"""m10_dequant_fast.py — 融合的反量化 kernel（Triton）：一個 kernel 把 FP8／INT8 的 block 轉成 BF16，直接寫進 KV 緩衝的視圖。

為什麼（09 §6）：主線量到真實 CPU 層的反量化很慢——Llama 一個 block（BF16 64 MiB）FP8 反量化約 0.23 ms，
有效頻寬約 0.4 TB/s〔算術：(32 MiB 讀＋64 MiB 寫)÷0.23 ms〕，MI300X HBM 是 5.3 TB/s。
現在的寫法（m10_codec.decode）是 PyTorch 拼的：view→float（寫 FP32）→乘 scale（再寫 FP32）→轉 BF16→copy 進視圖，
同一份資料來回 4–5 趟，還有好幾次 kernel 啟動。融合後只讀 1 次、寫 1 次。

結果必須和 m10_codec.decode **逐位元相同**（同樣是 FP32 相乘再四捨五入到 BF16），self-test 驗。
主線不用這支（07 凍結：判定用的速度是原本的寫法）；大掃描的「反量化加速」實驗（09 §6）才用，結果另外報。

出處（照實寫）：不是照某篇論文的程式碼。是兩個常見技巧——kernel 融合（多步合成一個 kernel、只讀寫一次）＋查表解碼
（FP8 只有 256 種碼）。相關前作見 docs/precision_20261013/cards/P07 §4（FLUTE 的查表反量化、vLLM FP8 Marlin 的位元技巧、
QServe／BitDecoding 把反量化塞進 attention）。bench 另放 torch.compile（PyTorch 官方編譯器自動融合同一段寫法）當對照組。

支援：fp8_head／fp8_channel／fp8_token（e4m3fn，scale 依粒度廣播）、int8_token、
      int4_token、int4_kivi（10/16 加：冒煙數據顯示 4-bit 的原寫法反量化比 FP8 慢 4–20 倍）。其他格式退回 m10_codec_x.decode。
INT4：一個 byte 放兩個值（偶數 index 在低 4 位，同 m10_codec._pack4），kernel 每個值讀它所在的 byte、取對應的半個 byte。
KIVI 的「乘 scale 再加 zero」要和 PyTorch 一樣分兩次捨入，所以 launch 時關掉 FMA 合併（enable_fp_fusion=False）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import triton
import triton.language as tl

import m10_codec as C

_LUT = {}


def _fp8_lut(device) -> torch.Tensor:
    """256 個 e4m3fn 碼 → FP32 值（精確）。編碼器把值截在 ±448，所以 NaN 碼不會出現。"""
    key = str(device)
    if key not in _LUT:
        _LUT[key] = torch.arange(256, dtype=torch.uint8).view(C.FP8).float().to(device)
    return _LUT[key]


@triton.jit
def _dq_kernel(q_ptr, s_ptr, lut_ptr, o_ptr, R, T, H,
               sq_l, sq_kv, sq_h, sq_t,
               ss_l, ss_kv, ss_h, ss_t, ss_d,
               so_l, so_kv, so_h, so_t,
               D: tl.constexpr, IS_FP8: tl.constexpr, ROWS: tl.constexpr):
    """每個 program 處理 ROWS 列（一列＝一個 (層, K/V, head, token) 的 D 個值）。q 的最後一維連續；scale 用 stride（0＝廣播）。"""
    pid = tl.program_id(0)
    # 列號（攤平的 L×2×H×T）。用 int64：寫進整個 KV 緩衝的視圖時，Llama 32K 的最遠偏移約 2.18e9 > 2^31−1（10/16 修，
    # CPU 直譯模式重現過：int32 會繞回、寫到緩衝前面；tests/dq_fast_offset_cpu_interp.py）
    r = (pid * ROWS + tl.arange(0, ROWS)).to(tl.int64)
    d = tl.arange(0, D)
    mask = r < R
    t = r % T
    h = (r // T) % H
    kv = (r // (T * H)) % 2
    l = r // (T * H * 2)
    qo = l * sq_l + kv * sq_kv + h * sq_h + t * sq_t
    so = l * ss_l + kv * ss_kv + h * ss_h + t * ss_t
    oo = l * so_l + kv * so_kv + h * so_h + t * so_t
    m2 = mask[:, None]
    if IS_FP8:
        code = tl.load(q_ptr + qo[:, None] + d[None, :], mask=m2, other=0).to(tl.int32)
        v = tl.load(lut_ptr + code)
    else:
        v = tl.load(q_ptr + qo[:, None] + d[None, :], mask=m2, other=0).to(tl.float32)
    s = tl.load(s_ptr + so[:, None] + d[None, :] * ss_d, mask=m2, other=1.0)
    tl.store(o_ptr + oo[:, None] + d[None, :], (v * s).to(tl.bfloat16), mask=m2)


@triton.jit
def _dq4_kernel(q_ptr, s_ptr, z_ptr, o_ptr, R, T, H, C,
                sq_l, sq_c, sq_h, sq_t,
                ss_l, ss_c, ss_h, ss_t, ss_d,
                so_l, so_c, so_h, so_t,
                D: tl.constexpr, TDIV: tl.constexpr, DDIV: tl.constexpr,
                HAS_ZERO: tl.constexpr, OFFSET8: tl.constexpr, ROWS: tl.constexpr):
    """INT4（兩個值一個 byte）→ BF16。列＝(層, 半邊 c, head, token)，C＝這次處理幾個半邊（int4_token 2 個、KIVI 一次 1 個）。
    scale／zero 的位置：(t // TDIV, d // DDIV)——KIVI 的 K 沿 token 每 32 個一組（TDIV＝32），V 沿 channel 每 32 個一組（DDIV＝32）。"""
    pid = tl.program_id(0)
    r = (pid * ROWS + tl.arange(0, ROWS)).to(tl.int64)   # int64：同 _dq_kernel 的說明
    d = tl.arange(0, D)
    mask = r < R
    t = r % T
    h = (r // T) % H
    c = (r // (T * H)) % C
    l = r // (T * H * C)
    m2 = mask[:, None]
    qo = l * sq_l + c * sq_c + h * sq_h + t * sq_t
    byte = tl.load(q_ptr + qo[:, None] + (d // 2)[None, :], mask=m2, other=0).to(tl.int32)
    nib = tl.where((d % 2 == 0)[None, :], byte & 0xF, byte >> 4)
    v = nib.to(tl.float32)
    if OFFSET8:
        v = v - 8.0
    so = l * ss_l + c * ss_c + h * ss_h + (t // TDIV) * ss_t
    sp = so[:, None] + (d // DDIV)[None, :] * ss_d
    y = v * tl.load(s_ptr + sp, mask=m2, other=1.0).to(tl.float32)
    if HAS_ZERO:
        y = y + tl.load(z_ptr + sp, mask=m2, other=0.0).to(tl.float32)
    oo = l * so_l + c * so_c + h * so_h + t * so_t
    tl.store(o_ptr + oo[:, None] + d[None, :], y.to(tl.bfloat16), mask=m2)


def supported(fmt: str) -> bool:
    return fmt in ("fp8_head", "fp8_channel", "fp8_token", "int8_token", "int4_token", "int4_kivi")


def _decode_int4(e, out, rows, num_warps=None):
    L, two, H, T, D = e.shape
    p = e.parts
    kw = dict(D=D, ROWS=rows, enable_fp_fusion=False)
    if num_warps:                                      # 09 §6.4 K3 調參才給；預設不傳（和原本相同）
        kw["num_warps"] = num_warps
    if e.fmt == "int4_token":
        q, s = p["q"], p["scale"]                      # q [L,2,H,T,D/2]、scale [L,2,H,T,1]
        assert q.is_contiguous() and q.stride(-1) == 1
        R = L * two * H * T
        _dq4_kernel[(triton.cdiv(R, rows),)](q, s, s, out, R, T, H, two,
                                             q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                                             s.stride(0), s.stride(1), s.stride(2), s.stride(3), 0,
                                             out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                                             TDIV=1, DDIV=1, HAS_ZERO=False, OFFSET8=True, **kw)
        return out
    G = C.KIVI_GROUP
    R = L * H * T
    for j, n in ((0, "k"), (1, "v")):                  # k_q／v_q [L,H,T,D/2]；K scale [L,H,T/G,1,D]、V scale [L,H,T,D/G,1]
        q, s, z, o = p[f"{n}_q"], p[f"{n}_scale"], p[f"{n}_zero"], out[:, j]
        assert q.is_contiguous() and s.stride() == z.stride()
        if n == "k":
            st, sd, tdiv, ddiv = s.stride(2), s.stride(4), G, 1
        else:
            st, sd, tdiv, ddiv = s.stride(2), s.stride(3), 1, G
        _dq4_kernel[(triton.cdiv(R, rows),)](q, s, z, o, R, T, H, 1,
                                             q.stride(0), 0, q.stride(1), q.stride(2),
                                             s.stride(0), 0, s.stride(1), st, sd,
                                             o.stride(0), 0, o.stride(1), o.stride(2),
                                             TDIV=tdiv, DDIV=ddiv, HAS_ZERO=True, OFFSET8=False, **kw)
    return out


def decode_fast(e, out: torch.Tensor | None = None, rows: int = 16, num_warps: int | None = None) -> torch.Tensor:
    if not supported(e.fmt):
        import m10_codec_x as X
        return X.decode(e, out)
    L, two, H, T, D = e.shape
    if out is None:
        out = torch.empty(e.shape, dtype=torch.bfloat16, device=next(iter(e.parts.values())).device)
    assert out.stride(-1) == 1 and out.dtype == torch.bfloat16
    if e.fmt.startswith("int4"):
        return _decode_int4(e, out, rows, num_warps)
    q, s = e.parts["q"], e.parts["scale"]
    assert q.is_contiguous() and q.shape == e.shape and q.stride(-1) == 1
    sb = s.expand(L, two, H, T, D)                      # 廣播 → stride 0
    R = L * two * H * T
    fp8 = e.fmt.startswith("fp8")
    grid = (triton.cdiv(R, rows),)
    _dq_kernel[grid](q if fp8 else q, sb, _fp8_lut(q.device), out, R, T, H,
                     q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                     sb.stride(0), sb.stride(1), sb.stride(2), sb.stride(3), sb.stride(4),
                     out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                     D=D, IS_FP8=fp8, ROWS=rows, **({"num_warps": num_warps} if num_warps else {}))
    return out


# ---- 對照組：PyTorch 官方編譯器（torch.compile／Inductor）自動融合同一段 PyTorch 寫法（不是手寫 kernel）
def _torch_ref(q, s, out):
    out.copy_((q.view(C.FP8).float() * s).to(torch.bfloat16))


def _torch_ref_int8(q, s, out):
    out.copy_((q.float() * s).to(torch.bfloat16))


def _torch_ref_int4_token(q, s, out):
    out.copy_(((C._unpack4(q).float() - 8) * s).to(torch.bfloat16))


def _torch_ref_kivi(kq, ks, kz, vq, vs, vz, out):
    L, _, H, T, D = out.shape
    G = C.KIVI_GROUP
    k = C._unpack4(kq).float().view(L, H, T // G, G, D) * ks.float() + kz.float()
    v = C._unpack4(vq).float().view(L, H, T, D // G, G) * vs.float() + vz.float()
    out.copy_(torch.stack((k.reshape(L, H, T, D), v.reshape(L, H, T, D)), dim=1).to(torch.bfloat16))


_COMPILED = {}


def decode_compiled(e, out):
    """torch.compile 版本：和 m10_codec.decode 一樣的運算，交給 Inductor 自動融合。只在 bench 用。"""
    key = "fp8" if e.fmt.startswith("fp8") else e.fmt
    fn = {"fp8": _torch_ref, "int8_token": _torch_ref_int8, "int4_token": _torch_ref_int4_token, "int4_kivi": _torch_ref_kivi}[key]
    if key not in _COMPILED:
        _COMPILED[key] = torch.compile(fn, dynamic=False)
    p = e.parts
    if key == "int4_kivi":
        _COMPILED[key](p["k_q"], p["k_scale"], p["k_zero"], p["v_q"], p["v_scale"], p["v_zero"], out)
    else:
        _COMPILED[key](p["q"], p["scale"], out)
    return out


# ====================================================================== 09 §6.4 方法掃描用（預設路徑不變）
@triton.jit
def _dqm_kernel(q_ptr, s_ptr, lut_ptr, o_ptr, bid_ptr, R, T, H, L,
                sq_b, sq_l, sq_kv, sq_h, sq_t,
                ss_b, ss_l, ss_kv, ss_h, ss_t, ss_d,
                so_l, so_kv, so_h, so_t,
                D: tl.constexpr, IS_FP8: tl.constexpr, BITS: tl.constexpr, ROWS: tl.constexpr):
    """_dq_kernel 的多 block 版：輸入第 0 維是 block（每個 block 連續），第 b 個寫到輸出的第 bid[b] 段（每段 T 個 token）。
    BITS（K4）：FP8 不查表，用位元運算拼出 FP32（e4m3fn：偏差 7 → FP32 偏差 127；次正規數＝尾數 × 2^-9），值和查表完全相同。"""
    pid = tl.program_id(0)
    r = (pid * ROWS + tl.arange(0, ROWS)).to(tl.int64)
    d = tl.arange(0, D)
    mask = r < R
    t = r % T
    h = (r // T) % H
    kv = (r // (T * H)) % 2
    l = (r // (T * H * 2)) % L
    b = r // (T * H * 2 * L)
    ob = tl.load(bid_ptr + b, mask=mask, other=0)
    qo = b * sq_b + l * sq_l + kv * sq_kv + h * sq_h + t * sq_t
    so = b * ss_b + l * ss_l + kv * ss_kv + h * ss_h + t * ss_t
    oo = l * so_l + kv * so_kv + h * so_h + (ob * T + t) * so_t
    m2 = mask[:, None]
    if IS_FP8:
        code = tl.load(q_ptr + qo[:, None] + d[None, :], mask=m2, other=0).to(tl.int32)
        if BITS:
            sgn = (code >> 7) & 1
            ex = (code >> 3) & 0xF
            mt = code & 0x7
            nrm = ((sgn << 31) | ((ex + 120) << 23) | (mt << 20)).to(tl.float32, bitcast=True)
            sub = mt.to(tl.float32) * 0.001953125
            sub = tl.where(sgn == 1, -sub, sub)
            v = tl.where(ex > 0, nrm, sub)
        else:
            v = tl.load(lut_ptr + code)
    else:
        v = tl.load(q_ptr + qo[:, None] + d[None, :], mask=m2, other=0).to(tl.float32)
    s = tl.load(s_ptr + so[:, None] + d[None, :] * ss_d, mask=m2, other=1.0)
    tl.store(o_ptr + oo[:, None] + d[None, :], (v * s).to(tl.bfloat16), mask=m2)


@triton.jit
def _dq4m_kernel(q_ptr, s_ptr, z_ptr, o_ptr, bid_ptr, R, T, H, C, L,
                 sq_b, sq_l, sq_c, sq_h, sq_t,
                 ss_b, ss_l, ss_c, ss_h, ss_t, ss_d,
                 so_l, so_c, so_h, so_t,
                 D: tl.constexpr, TDIV: tl.constexpr, DDIV: tl.constexpr,
                 HAS_ZERO: tl.constexpr, OFFSET8: tl.constexpr, ROWS: tl.constexpr):
    """_dq4_kernel 的多 block 版（同上）。"""
    pid = tl.program_id(0)
    r = (pid * ROWS + tl.arange(0, ROWS)).to(tl.int64)
    d = tl.arange(0, D)
    mask = r < R
    t = r % T
    h = (r // T) % H
    c = (r // (T * H)) % C
    l = (r // (T * H * C)) % L
    b = r // (T * H * C * L)
    ob = tl.load(bid_ptr + b, mask=mask, other=0)
    m2 = mask[:, None]
    qo = b * sq_b + l * sq_l + c * sq_c + h * sq_h + t * sq_t
    byte = tl.load(q_ptr + qo[:, None] + (d // 2)[None, :], mask=m2, other=0).to(tl.int32)
    nib = tl.where((d % 2 == 0)[None, :], byte & 0xF, byte >> 4)
    v = nib.to(tl.float32)
    if OFFSET8:
        v = v - 8.0
    so = b * ss_b + l * ss_l + c * ss_c + h * ss_h + (t // TDIV) * ss_t
    sp = so[:, None] + (d // DDIV)[None, :] * ss_d
    y = v * tl.load(s_ptr + sp, mask=m2, other=1.0).to(tl.float32)
    if HAS_ZERO:
        y = y + tl.load(z_ptr + sp, mask=m2, other=0.0).to(tl.float32)
    oo = l * so_l + c * so_c + h * so_h + (ob * T + t) * so_t
    tl.store(o_ptr + oo[:, None] + d[None, :], y.to(tl.bfloat16), mask=m2)


def group_parts(e, nb: int, device) -> dict:
    """「一次拆 nb 個 block」的暫存區：每個 part 多一個 block 維度（第 0 維），每個 block 連續。"""
    return {k: torch.empty((nb,) + tuple(t.shape), dtype=t.dtype, device=device) for k, t in e.parts.items()}


def sub_enc(e, gparts: dict, j: int):
    """暫存區裡第 j 個 block 的 Enc（視圖，給 copy_from 用）。"""
    return C.Enc(e.fmt, e.shape, {k: t[j] for k, t in gparts.items()})


def decode_multi(fmt: str, shape: tuple, gparts: dict, nb: int, out: torch.Tensor, bids: torch.Tensor,
                 rows: int = 16, num_warps: int | None = None, bits: bool = False) -> torch.Tensor:
    """一次 kernel 解 nb 個 block：gparts[k] 形狀 [nb, *part]；out 是 [L,2,H,T_out,D] 的 BF16（可以是整個 KV 緩衝）；
    第 b 個 block 寫到 out 的 token [bids[b]*T, (bids[b]+1)*T)，T＝shape[3]。和逐個 decode_fast 逐位元相同（自測驗）。"""
    assert supported(fmt) and out.stride(-1) == 1 and out.dtype == torch.bfloat16
    L, two, H, T, D = shape
    kw = dict(ROWS=rows, **({"num_warps": num_warps} if num_warps else {}))
    if fmt.startswith("fp8") or fmt == "int8_token":
        q, s = gparts["q"], gparts["scale"]
        assert q.is_contiguous()
        sb = s.expand(nb, L, two, H, T, D)
        R = nb * L * two * H * T
        _dqm_kernel[(triton.cdiv(R, rows),)](q, sb, _fp8_lut(q.device), out, bids, R, T, H, L,
                                             q.stride(0), q.stride(1), q.stride(2), q.stride(3), q.stride(4),
                                             sb.stride(0), sb.stride(1), sb.stride(2), sb.stride(3), sb.stride(4), sb.stride(5),
                                             out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                                             D=D, IS_FP8=fmt.startswith("fp8"), BITS=bool(bits), **kw)
        return out
    kw["enable_fp_fusion"] = False
    if fmt == "int4_token":
        q, s = gparts["q"], gparts["scale"]                  # [nb,L,2,H,T,D/2]、[nb,L,2,H,T,1]
        R = nb * L * two * H * T
        _dq4m_kernel[(triton.cdiv(R, rows),)](q, s, s, out, bids, R, T, H, two, L,
                                              q.stride(0), q.stride(1), q.stride(2), q.stride(3), q.stride(4),
                                              s.stride(0), s.stride(1), s.stride(2), s.stride(3), s.stride(4), 0,
                                              out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                                              D=D, TDIV=1, DDIV=1, HAS_ZERO=False, OFFSET8=True, **kw)
        return out
    G = C.KIVI_GROUP
    R = nb * L * H * T
    for j, n in ((0, "k"), (1, "v")):                      # q [nb,L,H,T,D/2]；K scale [nb,L,H,T/G,1,D]、V scale [nb,L,H,T,D/G,1]
        q, s, z, o = gparts[f"{n}_q"], gparts[f"{n}_scale"], gparts[f"{n}_zero"], out[:, j]
        assert q.is_contiguous() and s.stride() == z.stride()
        if n == "k":
            st, sd, tdiv, ddiv = s.stride(3), s.stride(5), G, 1
        else:
            st, sd, tdiv, ddiv = s.stride(3), s.stride(4), 1, G
        _dq4m_kernel[(triton.cdiv(R, rows),)](q, s, z, o, bids, R, T, H, 1, L,
                                              q.stride(0), q.stride(1), 0, q.stride(2), q.stride(3),
                                              s.stride(0), s.stride(1), 0, s.stride(2), st, sd,
                                              o.stride(0), 0, o.stride(1), o.stride(2),
                                              D=D, TDIV=tdiv, DDIV=ddiv, HAS_ZERO=True, OFFSET8=False, **kw)
    return out


def split_enc(e, U: int) -> list:
    """09 §6.4 G 組：把一個 block 的 Enc 沿 token 切成 T/U 段（每段連續、新配記憶體）。scale／zero 跟著切或整份複製，
    反量化是逐元素的，所以「切開來各自解」＝「整塊解」的對應段（自測驗）。只支援 fp8_*、int8_token、int4_token、int4_kivi、bf16。"""
    L, two, H, T, D = e.shape
    assert T % U == 0 and U % C.KIVI_GROUP == 0
    out = []
    for j in range(T // U):
        parts = {}
        for name, t in e.parts.items():
            ax = 2 if e.fmt == "int4_kivi" else 3          # KIVI 的 part 是 [L,H,T,...]；其他是 [L,2,H,T,...]
            n_ax = t.shape[ax]
            if n_ax == T:
                x = t.narrow(ax, j * U, U)
            elif e.fmt == "int4_kivi" and n_ax == T // C.KIVI_GROUP:
                g = C.KIVI_GROUP
                x = t.narrow(ax, j * U // g, U // g)
            elif n_ax == 1 and not e.fmt.startswith("fp8k0") and e.fmt != "int4_kivi":
                x = t
            else:
                raise ValueError(f"split_enc 不支援 {e.fmt} 的 {name} {tuple(t.shape)}")
            parts[name] = x.contiguous().clone()
        out.append(C.Enc(e.fmt, (L, two, H, U, D), parts))
    return out


_ZERO_BID = {}


def decode_variant(e, out=None, kernel: str = "k1"):
    """K 組的單 block 版本（還原器每個 block 呼叫一次）。k1＝decode_fast（預設）；k2＝torch.compile；
    k3:<rows>:<warps>＝調參；k4＝FP8 不查表（其他格式同 k1）。"""
    if out is None:
        out = torch.empty(e.shape, dtype=torch.bfloat16, device=next(iter(e.parts.values())).device)
    if kernel == "k1":
        return decode_fast(e, out)
    if kernel == "k2":
        return decode_compiled(e, out)
    if kernel.startswith("k3:"):
        _, r_, w_ = kernel.split(":")
        return decode_fast(e, out, rows=int(r_), num_warps=int(w_))
    if kernel == "k4":
        if not e.fmt.startswith("fp8"):
            return decode_fast(e, out)
        dev = out.device
        if str(dev) not in _ZERO_BID:
            _ZERO_BID[str(dev)] = torch.zeros(1, dtype=torch.int64, device=dev)
        return decode_multi(e.fmt, e.shape, {k: t.unsqueeze(0) for k, t in e.parts.items()}, 1, out,
                            _ZERO_BID[str(dev)], bits=True)
    raise ValueError(kernel)


def kernel_cfg(kernel: str) -> dict:
    """K 組設定 → decode_multi 的參數（S2／S4／S5 一次拆多個 block 時用同一套）。"""
    if kernel.startswith("k3:"):
        _, r_, w_ = kernel.split(":")
        return dict(rows=int(r_), num_warps=int(w_), bits=False)
    return dict(rows=16, num_warps=None, bits=(kernel == "k4"))


def install(kernel: str | None = None):
    """讓 m7_restore_harness 的載入線改用融合 kernel（不支援的格式自動退回原本的 decode）。大掃描的速度實驗才呼叫。
    kernel（09 §6.4 K 組）：None → 環境變數 M10_DQ_KERNEL，預設 k1（和原本完全相同）。"""
    import m7_restore_harness as RH
    import m10_codec_x as X
    X.install()
    kernel = kernel or os.environ.get("M10_DQ_KERNEL", "k1")
    other = X.decode
    if os.environ.get("M10_DEQUANT_FAST_X") == "1":      # 13 §5（Q3）：新格式也用融合 kernel（m10_dequant_fast_x；預設不接）
        import m10_dequant_fast_x as FX
        other = lambda e, out=None: FX.decode_x_fast(e, out) if FX.supported_x(e.fmt) else X.decode(e, out)  # noqa: E731
    if kernel == "k1":
        RH.decode = lambda e, out=None: decode_fast(e, out) if supported(e.fmt) else other(e, out)
    else:
        RH.decode = lambda e, out=None: decode_variant(e, out, kernel) if supported(e.fmt) else other(e, out)
    RH.DQ_KERNEL = kernel


def self_test(out_path=None, bench=False) -> bool:
    assert torch.cuda.is_available(), "要 GPU"
    res, ok = {"checks": {}}, True

    def check(name, cond, **info):
        nonlocal ok
        ok &= bool(cond)
        res["checks"][name] = {"pass": bool(cond), **info}
        print(("PASS " if cond else "FAIL ") + name, info if info else "", flush=True)

    x = C._synthetic().cuda()
    L, _, H, T, D = x.shape
    buf = torch.zeros((L, 2, H, 3 * T, D), dtype=torch.bfloat16, device="cuda")
    view = buf[:, :, :, T:2 * T, :]
    for fmt in ("fp8_head", "fp8_channel", "fp8_token", "int8_token", "int4_token", "int4_kivi"):
        e = C.encode(x, fmt)
        want = C.decode(e)
        got = decode_fast(e)
        check(f"bitwise[{fmt}]", torch.equal(got, want))
        buf.zero_()
        decode_fast(e, out=view)
        check(f"into_view[{fmt}]", torch.equal(view, want) and not bool(buf[:, :, :, :T].any())
              and not bool(buf[:, :, :, 2 * T:].any()))
        try:                                             # 官方編譯器的版本：記錄是否逐位元相同（不當成失敗條件）
            buf.zero_()
            decode_compiled(e, view)
            res["checks"][f"compiled_bitwise[{fmt}]"] = {"bitwise": bool(torch.equal(view, want))}
            print(f"INFO compiled_bitwise[{fmt}]", bool(torch.equal(view, want)), flush=True)
        except Exception as ex:                          # noqa: BLE001
            res["checks"][f"compiled_bitwise[{fmt}]"] = {"error": repr(ex)[:300]}
            print(f"INFO compiled[{fmt}] 失敗：{ex!r}"[:300], flush=True)
    if bench:                                            # 只在 GPU 沒有別人時跑才有意義
        res["bench"] = {}
        for name, shape in (("llama_block", (32, 2, 8, 512, 128)), ("qwen_block", (28, 2, 4, 512, 128))):
            xb = torch.randn(shape, device="cuda").to(torch.bfloat16)
            vb = torch.empty_like(xb)
            for fmt in ("fp8_head", "fp8_token", "int8_token", "int4_token", "int4_kivi"):
                e = C.encode(xb, fmt)
                for impl, fn in (("torch", lambda: C.decode(e, out=vb)), ("fused", lambda: decode_fast(e, out=vb)),
                                 ("torch_compile", lambda: decode_compiled(e, vb))):
                    try:
                        for _ in range(5):
                            fn()
                    except Exception as ex:              # noqa: BLE001  官方編譯器在 ROCm 上不一定能用：照實記
                        res["bench"][f"{name}|{fmt}|{impl}"] = {"error": repr(ex)[:300]}
                        print(f"{name} {fmt} {impl}: NOT_MEASURED（{ex!r}）"[:300], flush=True)
                        continue
                    torch.cuda.synchronize()
                    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    ts = []
                    for _ in range(50):
                        ev0.record()
                        fn()
                        ev1.record()
                        ev1.synchronize()
                        ts.append(ev0.elapsed_time(ev1))
                    ts.sort()
                    med = ts[len(ts) // 2]
                    gbps = (e.nbytes + xb.numel() * 2) / (med / 1e3) / 1e9
                    res["bench"][f"{name}|{fmt}|{impl}"] = {"median_ms": round(med, 4), "eff_GBps": round(gbps, 1)}
                    print(f"{name} {fmt} {impl}: {med:.4f} ms  ({gbps:.0f} GB/s 有效)", flush=True)
    res["all_pass"] = ok
    if out_path:
        json.dump(res, open(out_path, "w"), indent=1, ensure_ascii=False)
    print("ALL PASS" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test(a.out, a.bench) else 1)
    ap.print_help()
