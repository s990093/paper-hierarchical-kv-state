"""m10_dequant_fast_x.py — 大掃描新格式（m10_codec_x）的融合反量化 kernel（Triton）。13 號計畫 §5（Q3），只報告。

為什麼：大掃描複賽（43）的速度格子用原寫法拆新格式（PyTorch 一步一步做，每步一個 kernel）；融合 kernel 只支援主線的
FP8／INT8／INT4，新格式在真實 CPU 層的時間被拆主導（11 問題一 §5 第 2 點）。這裡補一個「每半邊（K 或 V）一個 kernel、
讀一次寫一次」的版本，直接寫進 KV 緩衝的視圖。

結果必須和 m10_codec_x.decode **逐位元相同**（GPU 上驗；CPU 直譯模式的 BF16 轉換是截斷，只能比截斷後的值）：
  每個值照 m10_codec_x.dec_half 的同一串 FP32 運算——KIVI 是「乘 scale 再加 zero」兩次捨入（launch 時關掉 FMA 合併）；
  其他格式只有一次 FP32 相乘；最後四捨五入成 BF16（PyTorch 是先 stack 成 FP32 再 .to(bf16)，每個值一樣）。
支援的半邊格式：bf16（直接複製）、int8_token、int4_token、fp8_head、fp8_token、fp8e5_head、fp8e5_token、kivi4、kivi2、mxfp4。
不支援（退回 m10_codec_x.decode）：nvfp4、int4h_token、int2h_g32（Hadamard 旋轉要另寫）。
只在 M10_DEQUANT_FAST_X=1 時由 m10_dequant_fast.install() 接上（預設不接，所有既有的 run 行為不變）。
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

import m10_codec as C
import m10_codec_x as X

SUPPORTED_HALF = {"bf16", "int8_token", "int4_token", "fp8_head", "fp8_token", "fp8e5_head", "fp8e5_token",
                  "kivi4", "kivi2", "mxfp4"}
_LUTS: dict = {}


def supported_x(fmt: str) -> bool:
    if not X.is_new(fmt):
        return False
    kf, vf = X.halves(fmt)
    return kf in SUPPORTED_HALF and vf in SUPPORTED_HALF


def _lut(kind: str, device) -> torch.Tensor:
    """查表：fp8（e4m3fn／e5m2 的 256 個碼 → FP32）、e2m1（16 個碼）、e8m0（256 個指數 → 2^(e−127)，和 dec_half 一樣用 torch.ldexp）。"""
    key = (kind, str(device))
    if key not in _LUTS:
        if kind == "e4m3":
            t = torch.arange(256, dtype=torch.uint8).view(C.FP8).float().to(device)
        elif kind == "e5m2":
            t = torch.arange(256, dtype=torch.uint8).view(X.E5).float().to(device)
        elif kind == "e2m1":
            t = torch.tensor(X._E2M1 + [-x for x in X._E2M1], device=device)
        else:
            t = torch.ldexp(torch.ones(256, device=device), torch.arange(256, device=device, dtype=torch.int32) - 127)
        _LUTS[key] = t.contiguous()
    return _LUTS[key]


@triton.jit
def _half_kernel(q_ptr, s_ptr, z_ptr, lut_ptr, lut2_ptr, o_ptr, R, T, H,
                 sq_l, sq_h, sq_t,
                 ss_l, ss_h, ss_t, ss_g,
                 so_l, so_h, so_t,
                 D: tl.constexpr, KIND: tl.constexpr, BITS: tl.constexpr,
                 TDIV: tl.constexpr, DDIV: tl.constexpr, ROWS: tl.constexpr):
    """列＝(層, head, token) 的 D 個值。KIND：0 int8（q×s）、1 int4 對稱（(u−8)×s）、2 fp8 查表（lut[u]×s）、
    3 KIVI（u×s＋z）、4 MXFP4（lut[u]×lut2[e8m0]）。BITS：每個值佔幾 bit（8／4／2）。
    scale（KIND 4 是 e8m0）的位置：(l, h, t // TDIV, d // DDIV)，ss_g＝最後那一維的 stride（0＝廣播）。"""
    pid = tl.program_id(0)
    r = (pid * ROWS + tl.arange(0, ROWS)).to(tl.int64)          # int64：寫進整個 KV 緩衝的視圖時偏移會超過 2^31（同 m10_dequant_fast）
    d = tl.arange(0, D)
    mask = r < R
    m2 = mask[:, None]
    t = r % T
    h = (r // T) % H
    l = r // (T * H)
    qo = l * sq_l + h * sq_h + t * sq_t
    if BITS == 8:
        raw = tl.load(q_ptr + qo[:, None] + d[None, :], mask=m2, other=0)
        if KIND == 0:
            u = raw.to(tl.float32)                              # int8_token：q 是有號的 int8
        else:
            u = raw.to(tl.int32)
    elif BITS == 4:
        byte = tl.load(q_ptr + qo[:, None] + (d // 2)[None, :], mask=m2, other=0).to(tl.int32)
        u = tl.where((d % 2 == 0)[None, :], byte & 0xF, byte >> 4)     # 偶數 index 在低 4 位（m10_codec._pack4）
    else:
        byte = tl.load(q_ptr + qo[:, None] + (d // 4)[None, :], mask=m2, other=0).to(tl.int32)
        u = (byte >> (2 * (d % 4))[None, :]) & 3                       # index 4j 在最低 2 位（m10_codec_x._pack2）
    so = l * ss_l + h * ss_h + (t // TDIV) * ss_t
    sp = so[:, None] + (d // DDIV)[None, :] * ss_g
    if KIND == 0:
        y = u * tl.load(s_ptr + sp, mask=m2, other=1.0).to(tl.float32)
    elif KIND == 1:
        y = (u.to(tl.float32) - 8.0) * tl.load(s_ptr + sp, mask=m2, other=1.0).to(tl.float32)
    elif KIND == 2:
        y = tl.load(lut_ptr + u) * tl.load(s_ptr + sp, mask=m2, other=1.0).to(tl.float32)
    elif KIND == 3:
        y = u.to(tl.float32) * tl.load(s_ptr + sp, mask=m2, other=1.0).to(tl.float32)
        y = y + tl.load(z_ptr + sp, mask=m2, other=0.0).to(tl.float32)
    else:
        e = tl.load(s_ptr + sp, mask=m2, other=127).to(tl.int32)
        y = tl.load(lut_ptr + u) * tl.load(lut2_ptr + e)
    oo = l * so_l + h * so_h + t * so_t
    tl.store(o_ptr + oo[:, None] + d[None, :], y.to(tl.bfloat16), mask=m2)


def _half(p: dict, f: str, role: str, o: torch.Tensor, rows: int, num_warps=None):
    """一個半邊（p＝這一半的 parts，o＝KV 緩衝的 [L, H, T, D] 視圖）。"""
    L, H, T, D = o.shape
    dev = o.device
    if f == "bf16":
        o.copy_(p["x"])
        return
    q = p["q"]
    assert q.is_contiguous() and q.stride(-1) == 1, (f, q.stride())
    dummy = q
    lut = lut2 = q
    z = None
    if f == "int8_token":
        s, kind, bits, tdiv, ddiv, sg = p["scale"], 0, 8, 1, 1, 0
    elif f == "int4_token":
        s, kind, bits, tdiv, ddiv, sg = p["scale"], 1, 4, 1, 1, 0
    elif f.startswith("fp8"):
        s, kind, bits, tdiv, ddiv, sg = p["scale"], 2, 8, 1, 1, 0
        lut = _lut("e5m2" if f.startswith("fp8e5") else "e4m3", dev)
    elif f in ("kivi4", "kivi2"):
        bits = 4 if f == "kivi4" else 2
        s, z, kind = p["scale"], p["zero"], 3
        assert s.stride() == z.stride()
        if role == "k":                                  # K scale [L,H,T/32,1,D]
            tdiv, ddiv, sg = X.G_KIVI, 1, s.stride(4)
        else:                                            # V scale [L,H,T,D/32,1]
            tdiv, ddiv, sg = 1, X.G_KIVI, s.stride(3)
    elif f == "mxfp4":
        s, kind, bits, tdiv, ddiv = p["e8m0"], 4, 4, 1, X.G_MX       # e8m0 [L,H,T,D/32,1]
        sg = s.stride(3)
        lut, lut2 = _lut("e2m1", dev), _lut("e8m0", dev)
    else:
        raise ValueError(f)
    st = s.stride(2) if s.shape[2] > 1 else 0            # fp8_head 的 scale [L,H,1,1]：沿 token 廣播
    R = L * H * T
    kw = dict(D=D, KIND=kind, BITS=bits, TDIV=tdiv, DDIV=ddiv, ROWS=rows, enable_fp_fusion=False)
    if num_warps:
        kw["num_warps"] = num_warps
    _half_kernel[(triton.cdiv(R, rows),)](q, s, z if z is not None else dummy, lut, lut2, o, R, T, H,
                                          q.stride(0), q.stride(1), q.stride(2),
                                          s.stride(0), s.stride(1), st, sg,
                                          o.stride(0), o.stride(1), o.stride(2), **kw)


def decode_x_fast(e, out: torch.Tensor | None = None, rows: int = 16, num_warps: int | None = None) -> torch.Tensor:
    if not supported_x(e.fmt):
        return X.decode(e, out)
    if out is None:
        out = torch.empty(e.shape, dtype=torch.bfloat16, device=next(iter(e.parts.values())).device)
    assert out.stride(-1) == 1 and out.dtype == torch.bfloat16
    kf, vf = X.halves(e.fmt)
    for j, (role, f) in enumerate((("k", kf), ("v", vf))):
        p = {k[2:]: t for k, t in e.parts.items() if k.startswith(role + ".")}
        _half(p, f, role, out[:, j], rows, num_warps)
    return out


def self_test(device: str, out_path=None) -> bool:
    """每種支援的格式（含 K／V 分開的組合）× 兩種形狀（Llama、Qwen 的 block，另加寫進大緩衝的視圖）：
    GPU → 和 m10_codec_x.decode 逐位元相同；CPU 直譯 → 和 FP32 參考值的截斷相同（直譯器轉 BF16 是截斷）。"""
    import json
    torch.manual_seed(0)
    interp = device == "cpu"
    fmts = ["int2_kivi", "mxfp4", "kv-int4_kivi-int2_kivi", "kv-int8_token-int4_kivi", "fp8e5_head", "fp8e5_token",
            "kv-fp8_token-int2_kivi", "kv-int4_token-mxfp4", "kv-bf16-int2_kivi", "kv-fp8_head-int8_token"]
    shapes = [(2, 2, 2, 512, 128), (3, 2, 4, 512, 128)] if interp else [(32, 2, 8, 512, 128), (28, 2, 4, 512, 128)]
    res, ok = [], True

    def ref_trunc(e):
        y = torch.stack([X.dec_half({k[2:]: t for k, t in e.parts.items() if k.startswith(r + ".")}, f, r,
                                    (e.shape[0], e.shape[2], e.shape[3], e.shape[4]))
                         for r, f in zip(("k", "v"), X.halves(e.fmt))], dim=1)
        return (y.contiguous().view(torch.int32) & -65536).view(torch.float32).to(torch.bfloat16)

    for shape in shapes:
        for fmt in fmts:
            x = (torch.randn(shape) * 2).to(torch.bfloat16)
            x[:, 0, :, :, 5] += 40                         # K 的離群 channel（像 Qwen）
            x[:, :, :, 7] = 0                              # 整列 0（MX 的 e8m0＝0、KIVI 的 scale＝0 分支）
            x = x.to(device)
            e = X.encode(x, fmt)
            want = ref_trunc(e) if interp else X.decode(e)
            got = decode_x_fast(e)
            # 寫進大緩衝的視圖（非連續，像還原時）：前後兩段要保持 0
            L, _, H, T, D = shape
            big = torch.zeros((L, 2, H, 3 * T, D), dtype=torch.bfloat16, device=device)
            decode_x_fast(e, out=big[:, :, :, T:2 * T])
            good = torch.equal(got, want) and torch.equal(big[:, :, :, T:2 * T], want) \
                and not big[:, :, :, :T].any() and not big[:, :, :, 2 * T:].any()
            n_diff = int((got != want).sum())
            res.append(dict(shape=list(shape), fmt=fmt, supported=supported_x(fmt), equal=bool(good), n_diff=n_diff))
            print(("PASS " if good else "FAIL ") + f"{fmt} {shape}（不同的值 {n_diff}）", flush=True)
            ok &= good
    for f in ("nvfp4", "int2h_g32", "int4h_token", "kv-nvfp4-int4_kivi"):
        sup = supported_x(f)
        res.append(dict(fmt=f, supported=sup))
        print(("PASS " if not sup else "FAIL ") + f"{f} 不支援 → 退回原寫法", flush=True)
        ok &= not sup
    if out_path:
        json.dump(dict(device=device, interp=interp, all_pass=ok, checks=res), open(out_path, "w"), indent=1)
    print("ALL PASS" if ok else "SOME FAILED", flush=True)
    return ok


if __name__ == "__main__":
    import argparse
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test(a.device, a.out) else 1)
