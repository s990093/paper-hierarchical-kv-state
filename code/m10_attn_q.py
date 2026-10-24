"""m10_attn_q.py — 13 號計畫 §6（Q5 (b)）：「拆融合進 attention」的對照組微測。只報告，不進任何判定（換了 attention 寫法，違反 07）。

一個 Triton 的 attention（GQA、split-K 的 flash attention；新一輪的 query 對整段舊 context，全部看得到，不用遮罩），
三種讀 KV 的方式共用同一份程式碼，差別只在「載入 K、V 的 tile」那一步：
  bf16       讀 BF16（和讀壓縮的兩種是同一個 kernel，用來分開「kernel 本身」和「拆」的成本）
  fp8        讀 e4m3fn 碼，查表 × scale（Llama fp8_token：每 token；Qwen fp8_head：每 512 token 的 block 一個 per-head scale）
  int4_kivi  讀 4-bit：K 每 channel、沿 token 每 32 個一組；V 每 token、沿 channel 每 32 個一組（同 m10_codec）
另量 PyTorch SDPA 讀 BF16（現在模型用的路徑）當參考。新一輪自己那些 token 的 attention（BF16、因果）每種方式都一樣，不在這裡。
外部參考（13 §8b，只量時間）：ext_vllm_kivi＝vLLM-KIVI 的 Triton kernel（code/third_party/，原樣），decode（1 個 query）才跑；
  它的 4-bit 格式和我們的不同、每個 q head 各讀一份 KV（不共用 GQA），值是亂數，只看時間；量一層 × 層數次（同一層重複）。

正確性（寫死）：tile 裡拆出來的 BF16 值和 m10_codec.decode 逐位元相同（同一串 FP32 運算、關掉 FMA 合併）→
「讀壓縮」的輸出必須和「同一個 kernel 讀 decode 後的 BF16」**逐位元相同**；和 SDPA 只報最大誤差（累加順序不同）。
CPU 直譯模式（--device cpu）的 BF16 轉換是截斷，所以改比「同一個 kernel 讀截斷後的 BF16」。

  python code/m10_attn_q.py selftest --device cpu|cuda [--out x.json]
  python code/m10_attn_q.py bench --model llama --lengths 32768 65536 129024 --queries 256 1 --out-dir D [--states idle busy]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time

import torch
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m10_codec as C  # noqa: E402

SHAPES = {"llama": dict(L=32, Hq=32, Hkv=8, D=128), "qwen": dict(L=28, Hq=28, Hkv=4, D=128)}
FP8_FMT = {"llama": "fp8_token", "qwen": "fp8_head"}       # 主線的 FP8 粒度
CHUNK = 512
TARGET_PROGRAMS = 608                                     # 約 2 × MI300X 的 304 個 CU：split-K 的份數照這個定（每種讀法一樣）


@triton.jit
def _attn_kernel(q_ptr, o_ptr, m_ptr, l_ptr,
                 k_ptr, v_ptr, ks_ptr, vs_ptr, kz_ptr, vz_ptr, lut_ptr,
                 R, N, SPLIT_N, sm_scale,
                 sq_h, sq_r, sk_h, sk_n, sv_h, sv_n,
                 sks_h, sks_n, sks_d, svs_h, svs_n, svs_d,
                 so_h, so_s, so_r, sm_h, sm_s,
                 D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, MODE: tl.constexpr,
                 KT: tl.constexpr, KD: tl.constexpr, VT: tl.constexpr, VD: tl.constexpr, F32DOT: tl.constexpr):
    """program＝(kv head, 列塊, split)。q：[Hkv, R, D]（R＝GQA 群組大小 × query 數）。MODE 0 bf16、1 fp8、2 int4_kivi。
    碼：fp8 [Hkv, N, D] uint8；int4_kivi [Hkv, N, D/2]。scale（fp8、KIVI）與 zero（KIVI）：K 在 (n // KT, d // KD)、V 在 (n // VT, d // VD)。"""
    h = tl.program_id(0)
    rb = tl.program_id(1)
    s = tl.program_id(2)
    offs_r = rb * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    rmask = offs_r < R
    q = tl.load(q_ptr + h * sq_h + offs_r[:, None] * sq_r + offs_d[None, :], mask=rmask[:, None], other=0.0)
    m_i = tl.full((BM,), float("-inf"), tl.float32)
    l_i = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    n_lo = s * SPLIT_N
    n_hi = tl.minimum(n_lo + SPLIT_N, N)
    for n0 in range(n_lo, n_hi, BN):
        offs_n = (n0 + tl.arange(0, BN)).to(tl.int64)
        nmask = offs_n < n_hi
        nm2 = nmask[:, None]
        if MODE == 0:
            k = tl.load(k_ptr + h * sk_h + offs_n[:, None] * sk_n + offs_d[None, :], mask=nm2, other=0.0)
            v = tl.load(v_ptr + h * sv_h + offs_n[:, None] * sv_n + offs_d[None, :], mask=nm2, other=0.0)
        else:
            kp = h * sks_h + (offs_n // KT)[:, None] * sks_n + (offs_d // KD)[None, :] * sks_d
            vp = h * svs_h + (offs_n // VT)[:, None] * svs_n + (offs_d // VD)[None, :] * svs_d
            ksc = tl.load(ks_ptr + kp, mask=nm2, other=1.0).to(tl.float32)
            vsc = tl.load(vs_ptr + vp, mask=nm2, other=1.0).to(tl.float32)
            if MODE == 1:
                kc = tl.load(k_ptr + h * sk_h + offs_n[:, None] * sk_n + offs_d[None, :], mask=nm2, other=0).to(tl.int32)
                vc = tl.load(v_ptr + h * sv_h + offs_n[:, None] * sv_n + offs_d[None, :], mask=nm2, other=0).to(tl.int32)
                k = (tl.load(lut_ptr + kc) * ksc).to(tl.bfloat16)
                v = (tl.load(lut_ptr + vc) * vsc).to(tl.bfloat16)
            else:
                kb = tl.load(k_ptr + h * sk_h + offs_n[:, None] * sk_n + (offs_d // 2)[None, :], mask=nm2, other=0).to(tl.int32)
                vb = tl.load(v_ptr + h * sv_h + offs_n[:, None] * sv_n + (offs_d // 2)[None, :], mask=nm2, other=0).to(tl.int32)
                ev = (offs_d % 2 == 0)[None, :]
                ku = tl.where(ev, kb & 0xF, kb >> 4).to(tl.float32)
                vu = tl.where(ev, vb & 0xF, vb >> 4).to(tl.float32)
                k = (ku * ksc + tl.load(kz_ptr + kp, mask=nm2, other=0.0).to(tl.float32)).to(tl.bfloat16)
                v = (vu * vsc + tl.load(vz_ptr + vp, mask=nm2, other=0.0).to(tl.float32)).to(tl.bfloat16)
        if F32DOT:                                       # 只給 CPU 直譯模式：直譯器的 bf16 dot 不對（把位元當整數乘），改用 FP32
            qk = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32))) * sm_scale
        else:
            qk = tl.dot(q, tl.trans(k)) * sm_scale
        qk = tl.where(nmask[None, :], qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        if F32DOT:
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16).to(tl.float32), v.to(tl.float32))
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    tl.store(o_ptr + h * so_h + s * so_s + offs_r[:, None] * so_r + offs_d[None, :], acc, mask=rmask[:, None])
    tl.store(m_ptr + h * sm_h + s * sm_s + offs_r, m_i, mask=rmask)
    tl.store(l_ptr + h * sm_h + s * sm_s + offs_r, l_i, mask=rmask)


class KVSet:
    """一層的 context KV。mode bf16：k、v＝[Hkv, N, D] bf16。fp8：碼 [Hkv, N, D]＋scale [Hkv, N // sdiv]。
    int4_kivi：碼 [Hkv, N, D/2]；K scale／zero [Hkv, N/32, D]、V scale／zero [Hkv, N, D/32]（fp16）。"""

    def __init__(self, mode, k, v, ks=None, vs=None, kz=None, vz=None, sdiv=1):
        self.mode, self.k, self.v, self.ks, self.vs, self.kz, self.vz, self.sdiv = mode, k, v, ks, vs, kz, vz, sdiv

    @property
    def N(self):
        return self.k.shape[1]


_LUT = {}


def n_split_for(Hkv, R, BM, N, BN=64):
    rb = triton.cdiv(R, BM)
    return max(1, min(triton.cdiv(N, BN), round(TARGET_PROGRAMS / (Hkv * rb))))


def attn(q: torch.Tensor, kv: KVSet, n_split: int, BM: int, BN: int = 64) -> torch.Tensor:
    """q：[Hkv, R, D] bf16 → 輸出 [Hkv, R, D] bf16（split-K 的部分和在 PyTorch 裡合併）。"""
    Hkv, R, D = q.shape
    N = kv.N
    dev = q.device
    split_n = triton.cdiv(triton.cdiv(N, n_split), BN) * BN
    ns = triton.cdiv(N, split_n)
    o = torch.empty((Hkv, ns, R, D), dtype=torch.float32, device=dev)
    m = torch.empty((Hkv, ns, R), dtype=torch.float32, device=dev)
    l_ = torch.empty_like(m)
    if str(dev) not in _LUT:
        _LUT[str(dev)] = torch.arange(256, dtype=torch.uint8).view(C.FP8).float().to(dev)
    mode = {"bf16": 0, "fp8": 1, "int4_kivi": 2}[kv.mode]
    if mode == 0:
        ks = vs = kz = vz = q
        kss = vss = (0, 0, 0)
        kt, kd, vt, vd = 1, 1, 1, 1
    elif mode == 1:                                      # scale [Hkv, N // sdiv]：d 廣播
        ks, vs, kz, vz = kv.ks, kv.vs, kv.ks, kv.vs
        kss, vss = (ks.stride(0), ks.stride(1), 0), (vs.stride(0), vs.stride(1), 0)
        kt, kd, vt, vd = kv.sdiv, 1, kv.sdiv, 1
    else:
        ks, vs, kz, vz = kv.ks, kv.vs, kv.kz, kv.vz
        assert ks.stride() == kz.stride() and vs.stride() == vz.stride()
        kss, vss = tuple(ks.stride()), tuple(vs.stride())
        kt, kd, vt, vd = C.KIVI_GROUP, 1, 1, C.KIVI_GROUP
    grid = (Hkv, triton.cdiv(R, BM), ns)
    _attn_kernel[grid](q, o, m, l_, kv.k, kv.v, ks, vs, kz, vz, _LUT[str(dev)],
                       R, N, split_n, 1.0 / (D ** 0.5),
                       q.stride(0), q.stride(1), kv.k.stride(0), kv.k.stride(1), kv.v.stride(0), kv.v.stride(1),
                       *kss, *vss, o.stride(0), o.stride(1), o.stride(2), m.stride(0), m.stride(1),
                       D=D, BM=BM, BN=BN, MODE=mode, KT=kt, KD=kd, VT=vt, VD=vd,
                       F32DOT=os.environ.get("TRITON_INTERPRET") == "1", enable_fp_fusion=False,
                       num_warps=4, num_stages=1)    # 三種讀法同一組設定；fp8 查表在 2 個 stage 時要 139 KB 共享記憶體（MI300X 上限 64 KB，43a 冒煙抓到；離線編譯量的）
    mx = m.max(dim=1, keepdim=True).values
    w = torch.exp(m - mx)
    out = (o * w.unsqueeze(-1)).sum(1) / (l_ * w).sum(1).unsqueeze(-1)
    return out.to(torch.bfloat16)


def encode_layer(x: torch.Tensor, fmt: str):
    """x：[2, Hkv, N, D] bf16（一層；N 是 512 的倍數）→（KVSet, decode 後的 BF16 [2, Hkv, N, D]）。
    照 m10_codec 每 512 token 一個 block 編碼（scale 都是每個 block 自己算），再沿 token 接起來。"""
    _, H, N, D = x.shape
    encs = [C.encode(x[None, :, :, b * CHUNK:(b + 1) * CHUNK].contiguous(), fmt) for b in range(N // CHUNK)]
    dec = torch.cat([C.decode(e)[0] for e in encs], dim=2)
    if fmt.startswith("fp8"):
        q = torch.cat([e.parts["q"][0] for e in encs], dim=2)                # [2, H, N, D] uint8
        sc = torch.cat([e.parts["scale"][0] for e in encs], dim=2)[..., 0]   # token：[2, H, N]；head：[2, H, N/512]
        sdiv = 1 if fmt == "fp8_token" else CHUNK
        return KVSet("fp8", q[0].contiguous(), q[1].contiguous(), sc[0].contiguous(), sc[1].contiguous(), sdiv=sdiv), dec
    assert fmt == "int4_kivi", fmt
    p = lambda k, d: torch.cat([e.parts[k][0] for e in encs], dim=d)        # noqa: E731
    ks, kz = p("k_scale", 1)[:, :, 0].contiguous(), p("k_zero", 1)[:, :, 0].contiguous()       # [H, N/32, D]
    vs, vz = p("v_scale", 1)[..., 0].contiguous(), p("v_zero", 1)[..., 0].contiguous()         # [H, N, D/32]
    return KVSet("int4_kivi", p("k_q", 1).contiguous(), p("v_q", 1).contiguous(), ks, vs, kz, vz), dec


def _trunc(y):
    return (y.float().contiguous().view(torch.int32) & -65536).view(torch.float32).to(torch.bfloat16)


def _ref32_layer(x, fmt):
    """CPU 直譯模式用：每個 block 的 FP32 反量化值（不捨入），之後截斷成 BF16（直譯器的轉換）。"""
    _, H, N, D = x.shape
    outs = []
    for b in range(N // CHUNK):
        e = C.encode(x[None, :, :, b * CHUNK:(b + 1) * CHUNK].contiguous(), fmt)
        p = e.parts
        if fmt.startswith("fp8"):
            outs.append((p["q"].view(C.FP8).float() * p["scale"])[0])
        else:
            G, T = C.KIVI_GROUP, CHUNK
            k = C._unpack4(p["k_q"]).float().view(1, H, T // G, G, D) * p["k_scale"].float() + p["k_zero"].float()
            v = C._unpack4(p["v_q"]).float().view(1, H, T, D // G, G) * p["v_scale"].float() + p["v_zero"].float()
            outs.append(torch.stack((k.reshape(H, T, D), v.reshape(H, T, D)), dim=0))
    return _trunc(torch.cat(outs, dim=2))


def _sdpa(q4, K, V):
    """q4：[1, Hq, M, D]；K、V：[Hkv, N, D] → [1, Hq, M, D]（PyTorch SDPA，現在模型用的路徑；全部看得到）。"""
    return torch.nn.functional.scaled_dot_product_attention(q4, K.unsqueeze(0), V.unsqueeze(0), enable_gqa=True)


def pack_q(q4, Hkv):
    """[1, Hq, M, D] → [Hkv, G×M, D]（HF 的 GQA：q head h 用 kv head h // G）。"""
    _, Hq, M, D = q4.shape
    return q4[0].view(Hkv, Hq // Hkv, M, D).reshape(Hkv, (Hq // Hkv) * M, D).contiguous()


def unpack_o(o, Hq, M):
    Hkv, R, D = o.shape
    return o.view(Hkv, Hq // Hkv, M, D).reshape(1, Hq, M, D)


def _bm(R):
    return 16 if R <= 16 else 64


def build_ext(Hq, N, D, device):
    """vLLM-KIVI kernel 的輸入（只量時間）：每個 q head 一份 4-bit KV、block_size 16。它的 K scale／min 用 k_cache 的 stride
    索引到 d＜128（會碰到下一個 head 的位置），所以配一塊夠大的平面緩衝，讓所有讀取都在範圍內。"""
    g = torch.Generator(device="cpu").manual_seed(1)
    flat = N * Hq * (D // 2) + D
    t = dict(q=torch.randn((1, Hq, D), generator=g).half(),
             kc=torch.randint(0, 256, (N, Hq, D // 2), generator=g, dtype=torch.uint8),
             vc=torch.randint(0, 256, (N, Hq, D // 2), generator=g, dtype=torch.uint8),
             ks=(torch.rand(flat, generator=g) * 0.01).half(), km=(torch.rand(flat, generator=g) - 0.5).half(),
             vs=(torch.rand((N, Hq), generator=g) * 0.01).half(), vm=(torch.rand((N, Hq), generator=g) - 0.5).half(),
             bt=torch.arange(N // 16, dtype=torch.int32).view(1, -1), cl=torch.tensor([N], dtype=torch.int32))
    return {k: v.to(device) for k, v in t.items()}


def ext_call(e, D):
    from third_party.vllm_kivi_attention import kivi_paged_attention
    return kivi_paged_attention(e["q"], e["kc"], e["vc"], e["ks"], e["km"], e["vs"], e["vm"], e["bt"], e["cl"],
                                1.0 / (D ** 0.5), 16)


def _make_x(H, N, D, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn((2, H, N, D), generator=g) * 1.5).to(torch.bfloat16)
    x[0, :, :, 5] += 30                                   # K 的離群 channel（像 Qwen）
    return x.to(device)


# ---------------------------------------------------------------- 自測
def selftest(device: str, out_path=None) -> bool:
    torch.manual_seed(0)
    interp = device == "cpu"
    cases = [("llama", 2048, 256), ("llama", 2048, 1), ("qwen", 2048, 256), ("qwen", 2048, 1)] if interp else \
        [("llama", 32768, 256), ("llama", 32768, 1), ("qwen", 32768, 256), ("qwen", 32768, 1), ("llama", 8192, 1)]
    if interp:                                            # 直譯器很慢：縮小 head 數
        shapes = {"llama": dict(L=1, Hq=4, Hkv=2, D=128), "qwen": dict(L=1, Hq=7, Hkv=1, D=128)}
    else:
        shapes = SHAPES
    res, ok = [], True
    for model, N, M in cases:
        sh = shapes[model]
        Hq, Hkv, D = sh["Hq"], sh["Hkv"], sh["D"]
        x = _make_x(Hkv, N, D, device)
        q4 = (torch.randn((1, Hq, M, D)) * 0.5).to(torch.bfloat16).to(device)
        qp = pack_q(q4, Hkv)
        BM = _bm(qp.shape[1])
        ns = n_split_for(Hkv, qp.shape[1], BM, N)
        o_bf = attn(qp, KVSet("bf16", x[0].contiguous(), x[1].contiguous()), ns, BM)
        ref_sdpa = _sdpa(q4.float(), x[0].float(), x[1].float())
        err_bf = float((unpack_o(o_bf, Hq, M).float() - ref_sdpa).abs().max())
        for fmt in (FP8_FMT[model], "int4_kivi"):
            kv, dec = encode_layer(x, fmt)
            o_q = attn(qp, kv, ns, BM)
            base = _ref32_layer(x, fmt) if interp else dec
            o_ref = attn(qp, KVSet("bf16", base[0].contiguous(), base[1].contiguous()), ns, BM)
            same = torch.equal(o_q, o_ref)
            err = float((unpack_o(o_q, Hq, M).float() - _sdpa(q4.float(), base[0].float(), base[1].float())).abs().max())
            good = same and err < 0.05
            extra = {}
            if not same and BM == 16 and os.environ.get("M10_ATTN_BM64_CHECK") == "1":
                # H200（13_h200_repro §13）：BM＝16 時 Triton 走 mma.sync，融合讀法的運算元排法（kWidth 4）和讀 BF16 的（kWidth 2）
                # 不同 → 同一條 tensor core 指令裡沿 k 的加總順序不同。同一個 split、BM＝64（wgmma）再比一次：逐位元相同＝拆出來的值相同
                same64 = torch.equal(attn(qp, kv, ns, 64),
                                     attn(qp, KVSet("bf16", base[0].contiguous(), base[1].contiguous()), ns, 64))
                extra = dict(bitwise_vs_same_kernel_on_decoded_bm64=same64)
                good = same64 and err < 0.05
            res.append(dict(model=model, N=N, M=M, fmt=fmt, bitwise_vs_same_kernel_on_decoded=same,
                            max_abs_err_vs_sdpa_fp32=err, bf16_kernel_max_abs_err_vs_sdpa_fp32=err_bf, n_split=ns, **extra))
            print(("PASS " if good else "FAIL ") + f"{model} N={N} M={M} {fmt}：和同一 kernel 讀拆好的 BF16 逐位元"
                  f"{'相同' if same else '不同'}"
                  + (f"（BM＝64 {'相同' if extra['bitwise_vs_same_kernel_on_decoded_bm64'] else '不同'}）" if extra else "")
                  + f"；對 SDPA(FP32) 最大誤差 {err:.4g}（BF16 kernel 本身 {err_bf:.4g}）", flush=True)
            ok &= good
    if out_path:
        json.dump(dict(device=device, interp=interp, all_pass=ok, checks=res), open(out_path, "w"), indent=1)
    print("ALL PASS" if ok else "SOME FAILED", flush=True)
    return ok


# ---------------------------------------------------------------- 微測
def bench(a):
    from m10_speed import Busy, _watcher, run_id, ts
    sh = SHAPES[a.model]
    L, Hq, Hkv, D = sh["L"], sh["Hq"], sh["Hkv"], sh["D"]
    dev = "cuda"
    rid = os.environ.get("RUN_ID") or run_id()
    os.makedirs(a.out_dir, exist_ok=True)
    lv = json.load(open(a.busy_levels))
    watch, level = _watcher(os.path.join(a.out_dir, f"gpu_guard_{rid}.json"))
    busy_pid = set()
    rows, busy_stats = [], []
    fmt8 = FP8_FMT[a.model]
    for N in a.lengths:
        # 每層一份（不同的緩衝，避免 cache 讓重複的那一層變快）
        layers = {"bf16": [], "fp8": [], "int4_kivi": []}
        for li in range(L):
            x = _make_x(Hkv, N, D, dev, seed=li)
            layers["bf16"].append(KVSet("bf16", x[0].contiguous(), x[1].contiguous()))
            layers["fp8"].append(encode_layer(x, fmt8)[0])
            layers["int4_kivi"].append(encode_layer(x, "int4_kivi")[0])
            del x
        torch.cuda.synchronize()
        for M in a.queries:
            q4 = (torch.randn((1, Hq, M, D)) * 0.5).to(torch.bfloat16).to(dev)
            qp = pack_q(q4, Hkv)
            BM = _bm(qp.shape[1])
            ns = n_split_for(Hkv, qp.shape[1], BM, N)
            fns = {"sdpa_bf16": lambda kvl: [_sdpa(q4, kv.k, kv.v) for kv in kvl],
                   "bf16": lambda kvl: [attn(qp, kv, ns, BM) for kv in kvl],
                   "fp8": lambda kvl: [attn(qp, kv, ns, BM) for kv in kvl],
                   "int4_kivi": lambda kvl: [attn(qp, kv, ns, BM) for kv in kvl]}
            src = {"sdpa_bf16": "bf16", "bf16": "bf16", "fp8": "fp8", "int4_kivi": "int4_kivi"}
            if M == 1 and not a.no_ext:                  # 13 §8b 外部參考：vLLM-KIVI（只量時間）
                try:
                    ext = build_ext(Hq, N, D, dev)
                    ext_call(ext, D)
                    torch.cuda.synchronize()
                    fns["ext_vllm_kivi"] = lambda _kvl: [ext_call(ext, D) for _ in range(L)]
                    src["ext_vllm_kivi"] = "bf16"
                except Exception as ex:                  # 跑不起來就照實記，不影響其他讀法
                    print(f"[ext_vllm_kivi] N={N} 跑不起來：{ex!r}", flush=True)
                    rows.append(dict(run_id=rid, ts=ts(), model=a.model, length=N, queries=M, mode="ext_vllm_kivi",
                                     fp8_fmt="", gpu_state="", duty="", layers=L, n_split="", n=0, median_s="ERROR",
                                     p10_s=repr(ex)[:200], contaminated_gpu="", host_level=level))
            for st in a.states:
                duty = 0.0 if st == "idle" else float(lv[a.busy_key])
                with Busy(duty, watch) as busy:
                    if busy.p:
                        busy_pid.add(busy.p.pid)
                    order = list(fns)
                    times = {k: [] for k in order}
                    for k in order:                      # 暖機（也讓 Triton 編譯好）
                        for _ in range(a.warmup):
                            fns[k](layers[src[k]])
                    torch.cuda.synchronize()
                    for it in range(a.iters):
                        for k in (order if it % 2 == 0 else order[::-1]):      # 交替順序
                            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            e0.record()
                            fns[k](layers[src[k]])
                            e1.record()
                            e1.synchronize()
                            times[k].append(e0.elapsed_time(e1) / 1e3)
                    bad = {p_: v for p_, v in (watch.intruders.items() if watch else []) if p_ not in busy_pid}
                    cont = "NOT_CHECKED" if watch is None else bool(bad) or watch.started_clean is False
                    for k in order:
                        rows.append(dict(run_id=rid, ts=ts(), model=a.model, length=N, queries=M, mode=k,
                                         fp8_fmt=fmt8 if k == "fp8" else "", gpu_state=st, duty=duty, layers=L,
                                         n_split=ns if k not in ("sdpa_bf16", "ext_vllm_kivi") else "", n=len(times[k]),
                                         median_s=round(statistics.median(times[k]), 6),
                                         p10_s=round(sorted(times[k])[len(times[k]) // 10], 6),
                                         contaminated_gpu=cont, host_level=level))
                        print(f"[{a.model} N={N} M={M} {st}] {k}: {1e3 * statistics.median(times[k]):.3f} ms（{L} 層）", flush=True)
                if busy.p:
                    busy_pid.discard(busy.p.pid)
                    busy_stats.append(dict(length=N, queries=M, state=st, **(busy.stats or {})))
        del layers
        torch.cuda.empty_cache()
    if watch is not None:
        watch.__exit__(None, None, None)
    p = os.path.join(a.out_dir, f"attn_{a.model}.csv")
    new = not os.path.exists(p)
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    json.dump(busy_stats, open(os.path.join(a.out_dir, f"busy_{rid}.json"), "w"), indent=1)
    print("寫了", p, flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selftest")
    s.add_argument("--device", default="cuda")
    s.add_argument("--out")
    b = sub.add_parser("bench")
    b.add_argument("--model", choices=list(SHAPES), required=True)
    b.add_argument("--lengths", nargs="+", type=int, default=[32768, 65536, 129024])
    b.add_argument("--queries", nargs="+", type=int, default=[256, 1])
    b.add_argument("--states", nargs="+", default=["idle", "busy"])
    b.add_argument("--busy-levels", default=os.environ.get("M10_BUSY_LEVELS")       # H200：環境變數（10/19；沒設時不變）
                   or os.path.join(HERE, "..", "results", "m7_write_policy_mi300x", "busy_levels.json"))
    b.add_argument("--busy-key", default="busy2")
    b.add_argument("--warmup", type=int, default=5)
    b.add_argument("--iters", type=int, default=50)
    b.add_argument("--out-dir", required=True)
    b.add_argument("--no-ext", action="store_true", help="不跑外部參考 ext_vllm_kivi")
    a = ap.parse_args()
    if a.cmd == "selftest":
        sys.exit(0 if selftest(a.device, a.out) else 1)
    bench(a)


if __name__ == "__main__":
    main()
