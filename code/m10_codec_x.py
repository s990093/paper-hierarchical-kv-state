"""m10_codec_x.py — 大掃描（docs/precision_20261013/09_sweep_plan.md）用的額外 KV 格式。

主線（07 凍結）用的 m10_codec.py 一個字都不改：舊格式一律原封不動轉給 m10_codec（逐位元相同，自測第 1 項驗）。
新格式全部「K、V 各自一半」編碼（K 和 V 可以用不同格式），反量化成 BF16 後走和 E0 相同的 attention（同 m10_codec）。

新格式（x 的一半＝[L, H, T, D]，D＝128；分組都沿 D，除非另寫）：
  fp8e5_head／fp8e5_token  e5m2（最大 57344），scale 每（層, head）／每（層, head, token）＝amax÷57344
  mxfp4         OCP MX：E2M1 值、每 32 個值一個 E8M0 scale（2 的次方）＝2^(floor(log2 amax)−2)
  nvfp4         NVIDIA NVFP4：E2M1 值、每 16 個值一個 E4M3 scale＝amax÷6÷g（截在 [2^-6, 448]），g＝每（層, K/V）的 FP32 全域 scale＝amax÷(448×6)；
                量化乘倒數 x·((1/g)/s)，和 torchao 參考實作逐值相同（10/15 比對）
  int2_kivi     KIVI 2-bit：K 每 channel（沿 token 每 32 個一組）、V 每 token（沿 channel 每 32 個一組），scale／zero 存 FP16
  int4h_token   先沿 D 做正規化 Hadamard 旋轉（QuaRot 式），再對稱 INT4 每（層, head, token）；解碼後轉回來
  int2h_g32     先 Hadamard 旋轉，再非對稱 2-bit、每（token, 32 個 channel）一組，scale／zero 存 FP16
  kv-<K 格式>-<V 格式>   K、V 用不同格式（KVTuner 的「K 比 V 敏感」），例 kv-int8_token-int4_kivi（K8V4）、
                kv-int4_kivi-int2_kivi（K4V2）。K／V 格式可以是上面任何一種或 bf16、int8_token、int4_token、fp8_head、fp8_token、int4_kivi。

寫死的細節（和 m10_codec 同一套規則）：
- 除以常數一律 _div_c（GPU／CPU 逐位元相同）；floor(log2) 用 frexp（精確，不靠 log2 的捨入）。
- E2M1 捨入：最近值、平手取偶數碼（OCP MX 規格）；超過 ±6 截在 ±6。碼＝符號位（bit 3）＋大小索引（0..7）。
- amax＝0 的組：MX 的指數設 −127（值全為 0）；NV 的 block scale 截在 2^-6 以上（同 torchao），整半全 0 時全域 scale 設 1
  （torchao 0.18 在這種情況會算出 NaN，我們量成 0）。
- 位元組＝Enc.nbytes（實際 tensor 大小），含 scale／zero／全域 scale。

自測：python code/m10_codec_x.py --self-test [--out x.json] [--gpu]
install()：把 m10_codec.encode／decode 與 m7_restore_harness.decode 換成這裡的分派版（只有大掃描的速度量測呼叫）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import torch

import m10_codec as C
from m10_codec import Enc, _div_c, _pack4, _unpack4

E5 = torch.float8_e5m2
E5_MAX = 57344.0
G_MX, G_NV, G_KIVI = 32, 16, C.KIVI_GROUP
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_E4M3_TINY = torch.finfo(torch.float8_e4m3fn).tiny          # 2^-6：NVFP4 block scale 的下限（同 torchao）
_E2M1_MID = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
_ALIAS = {"int4_kivi": "kivi4", "int2_kivi": "kivi2"}
HALF = ["bf16", "int8_token", "int4_token", "fp8_head", "fp8_token", "fp8e5_head", "fp8e5_token",
        "mxfp4", "nvfp4", "kivi4", "kivi2", "int4h_token", "int2h_g32"]
NEW_FULL = ["fp8e5_head", "fp8e5_token", "mxfp4", "nvfp4", "int2_kivi", "int4h_token", "int2h_g32"]
MIXED_EXAMPLES = ["kv-int8_token-int4_kivi", "kv-int4_kivi-int2_kivi"]
# m10_codec 原本的函式：install() 會把 C.encode／C.decode 換成這裡的分派版，舊格式必須呼叫原本的，不能呼叫 C.encode（會無限遞迴；
# 10/16 stage 31z 冒煙測試抓到）
_C_ENCODE, _C_DECODE = C.encode, C.decode


def is_new(fmt: str) -> bool:
    return fmt in NEW_FULL or fmt.startswith("kv-")


def halves(fmt: str) -> tuple[str, str]:
    if fmt.startswith("kv-"):
        rest = fmt[3:]
        for k in sorted(HALF + list(_ALIAS), key=len, reverse=True):     # K 格式名可能含底線，取最長前綴
            if rest.startswith(k + "-"):
                kf, vf = k, rest[len(k) + 1:]
                break
        else:
            raise ValueError(fmt)
    else:
        kf = vf = fmt
    kf, vf = _ALIAS.get(kf, kf), _ALIAS.get(vf, vf)
    assert kf in HALF and vf in HALF, (fmt, kf, vf)
    return kf, vf


# ---------------------------------------------------------------- 小工具
def _pack2(u: torch.Tensor) -> torch.Tensor:
    """u：0..3（最後一維是 4 的倍數）→ uint8，index 4j 放最低 2 位。"""
    u = u.to(torch.uint8)
    return u[..., 0::4] | (u[..., 1::4] << 2) | (u[..., 2::4] << 4) | (u[..., 3::4] << 6)


def _unpack2(p: torch.Tensor) -> torch.Tensor:
    out = torch.stack([(p >> s) & 3 for s in (0, 2, 4, 6)], dim=-1)
    return out.reshape(*p.shape[:-1], p.shape[-1] * 4)


_HAD = {}


def _hadamard(D: int, device) -> torch.Tensor:
    """正規化 Sylvester Hadamard（H＝Hᵀ、H·H＝I），FP32。"""
    key = (D, str(device))
    if key not in _HAD:
        assert D & (D - 1) == 0, D
        h = torch.ones((1, 1), dtype=torch.float64)
        while h.shape[0] < D:
            h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
        _HAD[key] = (h / math.sqrt(D)).float().to(device)
    return _HAD[key]


def _fwht(x: torch.Tensor) -> torch.Tensor:
    """沿最後一維的正規化 Walsh-Hadamard 轉換（Sylvester 順序，＝x @ _hadamard(D)）。用固定順序的加減蝴蝶，
    不用矩陣乘法：BLAS／rocBLAS 的加總順序不同，矩陣乘法在 CPU 和 GPU 上最後一位會不一樣（10/15 自測抓到）。
    H 對稱且 H·H＝I，所以同一個函式也是反轉換。"""
    D = x.shape[-1]
    assert D & (D - 1) == 0, D
    shape, y, h = x.shape, x, 1
    while h < D:
        y = y.reshape(*shape[:-1], D // (2 * h), 2, h)
        y = torch.stack((y[..., 0, :] + y[..., 1, :], y[..., 0, :] - y[..., 1, :]), dim=-2).reshape(shape)
        h *= 2
    return y * torch.full((), 1.0 / math.sqrt(D), dtype=y.dtype, device=y.device)


def _e2m1_enc(v: torch.Tensor) -> torch.Tensor:
    """v：已除以 scale 的 FP32 → 4-bit 碼（uint8，0..15）。最近值、平手取偶數碼、截在 ±6。"""
    a = v.abs().clamp_(max=6.0)
    mid = torch.tensor(_E2M1_MID, device=v.device)
    lo = torch.bucketize(a, mid, right=False)
    hi = torch.bucketize(a, mid, right=True)
    idx = torch.where(lo == hi, lo, torch.where(lo % 2 == 0, lo, hi))
    sign = (v < 0) & (idx > 0)
    return (idx | (sign.to(idx.dtype) << 3)).to(torch.uint8)


def _e2m1_dec(code: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(_E2M1 + [-x for x in _E2M1], device=code.device)
    return lut[code.long()]


def _grp(t: torch.Tensor, g: int) -> torch.Tensor:
    """[..., D] → [..., D/g, g]"""
    return t.view(*t.shape[:-1], t.shape[-1] // g, g)


# ---------------------------------------------------------------- 半邊編碼／解碼（t＝[L, H, T, D]）
def enc_half(t: torch.Tensor, f: str, role: str) -> dict:
    L, H, T, D = t.shape
    xf = t.float()
    if f == "bf16":
        return {"x": t.contiguous().clone()}
    if f in ("int8_token", "int4_token", "int4h_token"):
        if f == "int4h_token":
            xf = _fwht(xf)
        qmax = 127.0 if f == "int8_token" else 7.0
        amax = xf.abs().amax(dim=-1, keepdim=True)
        s = torch.where(amax > 0, _div_c(amax, qmax), torch.ones_like(amax))
        q = torch.round(xf / s).clamp_(-qmax, qmax)
        if f == "int8_token":
            return {"q": q.to(torch.int8), "scale": s}
        return {"q": _pack4(q + 8), "scale": s}
    if f.startswith("fp8"):
        e5 = f.startswith("fp8e5")
        dt, mx = (E5, E5_MAX) if e5 else (C.FP8, C.FP8_MAX)
        dims = (2, 3) if f.endswith("_head") else (3,)
        amax = xf.abs().amax(dim=dims, keepdim=True)
        s = torch.where(amax > 0, _div_c(amax, mx), torch.ones_like(amax))
        q = (xf / s).clamp_(-mx, mx).to(dt)
        return {"q": q.view(torch.uint8), "scale": s}
    if f == "mxfp4":
        g = _grp(xf, G_MX)
        amax = g.abs().amax(dim=-1, keepdim=True)
        _, ex = torch.frexp(amax)                          # amax＝m·2^ex，m∈[0.5,1) → floor(log2 amax)＝ex−1
        e = torch.where(amax > 0, ex - 1 - 2, torch.full_like(ex, -127)).clamp_(-127, 127)
        sc = torch.ldexp(torch.ones_like(amax), e)
        code = _e2m1_enc(g / sc)
        return {"q": _pack4(code.view(L, H, T, D)), "e8m0": (e + 127).to(torch.uint8)}
    if f == "nvfp4":
        gmax = xf.abs().amax(dim=(1, 2, 3), keepdim=True)   # 每（層）——這是 K 或 V 其中一半，所以等於每（層, K/V）
        gs = torch.where(gmax > 0, _div_c(gmax, C.FP8_MAX * 6.0), torch.ones_like(gmax))
        g = _grp(xf, G_NV)
        amax = g.abs().amax(dim=-1, keepdim=True)
        gu = gs.unsqueeze(-1)
        # 和 NVIDIA 的算法（torchao nvfp4_quantize＝MSLK triton kernel 的數值）相同：block scale 截在 [E4M3 最小正規數, 448]，
        # 量化用「乘倒數」x·((1/g)/s)；10/15 對 torchao 0.18 逐值比對過（09 §2.1）
        s8 = (_div_c(amax, 6.0) / gu).clamp_(min=_E4M3_TINY, max=C.FP8_MAX).to(C.FP8)
        recip = (torch.ones_like(gu) / gu) / s8.float()
        code = _e2m1_enc(g * recip)
        return {"q": _pack4(code.view(L, H, T, D)), "s8": s8.view(torch.uint8), "g": gs}
    if f in ("kivi4", "kivi2", "int2h_g32"):
        bits = 4 if f == "kivi4" else 2
        lv = float(2 ** bits - 1)
        if f == "int2h_g32":
            xf = _fwht(xf)
            g, red = _grp(xf, G_KIVI), -1
        elif role == "k":                                     # K：每 channel，沿 token 分組
            g, red = xf.view(L, H, T // G_KIVI, G_KIVI, D), 3
        else:                                                 # V：每 token，沿 channel 分組
            g, red = _grp(xf, G_KIVI), -1
        mn, mx = g.amin(dim=red, keepdim=True), g.amax(dim=red, keepdim=True)
        mn16, sc16 = mn.half(), _div_c(mx - mn, lv).half()
        sc = torch.where(sc16 > 0, sc16, torch.ones_like(sc16)).float()
        q = torch.round((g - mn16.float()) / sc).clamp_(0, lv).reshape(L, H, T, D)
        return {"q": _pack4(q) if bits == 4 else _pack2(q), "scale": sc.half(), "zero": mn16}
    raise ValueError(f)


def dec_half(p: dict, f: str, role: str, shape: tuple) -> torch.Tensor:
    L, H, T, D = shape
    if f == "bf16":
        return p["x"].float()
    if f == "int8_token":
        return p["q"].float() * p["scale"]
    if f in ("int4_token", "int4h_token"):
        y = (_unpack4(p["q"]).float() - 8) * p["scale"]
        return _fwht(y) if f == "int4h_token" else y
    if f.startswith("fp8"):
        dt = E5 if f.startswith("fp8e5") else C.FP8
        return p["q"].view(dt).float() * p["scale"]
    if f == "mxfp4":
        v = _grp(_e2m1_dec(_unpack4(p["q"])), G_MX)
        sc = torch.ldexp(torch.ones(p["e8m0"].shape, device=v.device), p["e8m0"].int() - 127)
        return (v * sc).view(L, H, T, D)
    if f == "nvfp4":
        v = _grp(_e2m1_dec(_unpack4(p["q"])), G_NV)
        eff = p["s8"].view(C.FP8).float() * p["g"].unsqueeze(-1)
        eff = torch.where(eff > 0, eff, torch.ones_like(eff))
        return (v * eff).view(L, H, T, D)
    if f in ("kivi4", "kivi2", "int2h_g32"):
        u = _unpack4(p["q"]) if f == "kivi4" else _unpack2(p["q"])
        u = u.float()
        if f == "int2h_g32" or role == "v":
            g = _grp(u, G_KIVI)
        else:
            g = u.view(L, H, T // G_KIVI, G_KIVI, D)
        y = (g * p["scale"].float() + p["zero"].float()).reshape(L, H, T, D)
        return _fwht(y) if f == "int2h_g32" else y
    raise ValueError(f)


# ---------------------------------------------------------------- 整個 block
def encode(x: torch.Tensor, fmt: str) -> Enc:
    if not is_new(fmt):
        return _C_ENCODE(x, fmt)
    assert x.dim() == 5 and x.shape[1] == 2 and x.dtype == torch.bfloat16, (x.shape, x.dtype)
    kf, vf = halves(fmt)
    parts = {}
    for j, (role, f) in enumerate((("k", kf), ("v", vf))):
        for k, t in enc_half(x[:, j], f, role).items():
            parts[f"{role}.{k}"] = t
    return Enc(fmt, tuple(x.shape), parts)


def decode(e: Enc, out: torch.Tensor | None = None) -> torch.Tensor:
    if not is_new(e.fmt):
        return _C_DECODE(e, out)
    L, _, H, T, D = e.shape
    kf, vf = halves(e.fmt)
    ys = []
    for role, f in (("k", kf), ("v", vf)):
        p = {k[2:]: t for k, t in e.parts.items() if k.startswith(role + ".")}
        ys.append(dec_half(p, f, role, (L, H, T, D)))
    y = torch.stack(ys, dim=1).to(torch.bfloat16)
    if out is None:
        return y
    out.copy_(y)
    return out


def install():
    """大掃描的速度量測用：讓 m10_speed／m7_restore_harness 認得新格式（舊格式照樣走 m10_codec 原本的函式）。"""
    import m7_restore_harness as RH
    C.encode, C.decode = encode, decode    # m10_speed 在函式內 `from m10_codec import encode` → 拿到這裡的版本
    RH.decode = decode


def bits_per_value(fmt: str, shape=(32, 2, 8, 512, 128)) -> float:
    """實際位元組 ÷ 值的個數 × 8（含 scale／zero）。用合成的 0 張量算形狀即可。"""
    x = torch.zeros(shape, dtype=torch.bfloat16)
    x[..., 0] = 1.0
    return encode(x, fmt).nbytes * 8 / x.numel()


# ---------------------------------------------------------------- 自測
def self_test(out_path=None, gpu=False) -> bool:
    res, ok = {"checks": {}}, True

    def check(name, cond, **info):
        nonlocal ok
        ok &= bool(cond)
        res["checks"][name] = {"pass": bool(cond), **info}
        print(("PASS " if cond else "FAIL ") + name, info if info else "", flush=True)

    x = C._synthetic()
    L, _, H, T, D = x.shape
    n = x.numel() // 2                                   # 一半（K 或 V）的值個數
    # 1. 舊格式透過這裡＝m10_codec 原本的結果（逐位元）
    for fmt in C.FORMATS:
        a, b = C.encode(x, fmt), encode(x, fmt)
        check(f"old_passthrough[{fmt}]", all(torch.equal(a.parts[k], b.parts[k]) for k in a.parts)
              and torch.equal(C.decode(a), decode(b)))
    # 1b. install() 之後舊格式仍然＝原本的結果（不遞迴）；測完還原
    import m7_restore_harness as RH
    saved = (C.encode, C.decode, RH.decode)
    try:
        install()
        for fmt in C.FORMATS:
            a, b = _C_ENCODE(x, fmt), C.encode(x, fmt)
            check(f"installed_old[{fmt}]", all(torch.equal(a.parts[k], b.parts[k]) for k in a.parts)
                  and torch.equal(_C_DECODE(a), C.decode(b)) and torch.equal(_C_DECODE(a), RH.decode(b)))
        e = C.encode(x, "mxfp4")
        check("installed_new[mxfp4]", e.fmt == "mxfp4" and torch.equal(RH.decode(e), decode(encode(x, "mxfp4"))))
    finally:
        C.encode, C.decode, RH.decode = saved
    # 2. 小工具
    u = torch.randint(0, 4, (3, 5, 128), dtype=torch.uint8)
    check("pack2_roundtrip", torch.equal(_unpack2(_pack2(u)), u))
    h = _hadamard(128, "cpu")
    check("hadamard_orthonormal", torch.allclose(h @ h, torch.eye(128), atol=1e-5))
    xx = torch.randn(7, 128)
    check("fwht_eq_matmul", torch.allclose(_fwht(xx), xx @ h, atol=1e-5) and torch.allclose(_fwht(_fwht(xx)), xx, atol=1e-5))
    v = torch.tensor([0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0, -0.25, -2.9, -6.5, 0.3])
    got = _e2m1_dec(_e2m1_enc(v)).tolist()
    want = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 0.0, -3.0, -6.0, 0.5]   # 平手取偶數碼、截在 6
    check("e2m1_round_ties_even", got == want, got=got, want=want)
    # 3. 新格式：往返、位元組、決定性、寫進非連續視圖
    expect = {
        "fp8e5_head": 2 * (n + L * H * 4), "fp8e5_token": 2 * (n + L * H * T * 4),
        "mxfp4": 2 * (n // 2 + n // G_MX), "nvfp4": 2 * (n // 2 + n // G_NV + L * 4),
        "int2_kivi": n // 4 + 2 * 2 * (L * H * (T // G_KIVI) * D) + n // 4 + 2 * 2 * (L * H * T * (D // G_KIVI)),
        "int4h_token": 2 * (n // 2 + L * H * T * 4), "int2h_g32": 2 * (n // 4 + 2 * 2 * (n // G_KIVI)),
        "kv-int8_token-int4_kivi": (n + L * H * T * 4) + (n // 2 + 2 * 2 * L * H * T * (D // G_KIVI)),
        "kv-int4_kivi-int2_kivi": (n // 2 + 2 * 2 * L * H * (T // G_KIVI) * D) + (n // 4 + 2 * 2 * L * H * T * (D // G_KIVI)),
    }
    errs = {}
    buf = torch.zeros((L, 2, H, 3 * T, D), dtype=torch.bfloat16)
    view = buf[:, :, :, T:2 * T, :]
    for fmt in NEW_FULL + MIXED_EXAMPLES:
        e = encode(x, fmt)
        y = decode(e)
        errs[fmt] = C.rel_err(x, y)
        check(f"roundtrip_finite[{fmt}]", bool(torch.isfinite(y.float()).all()) and y.shape == x.shape and
              y.dtype == torch.bfloat16, rel_err=errs[fmt])
        check(f"nbytes[{fmt}]", e.nbytes == expect[fmt], nbytes=e.nbytes, expect=expect[fmt],
              bits_per_value=round(e.nbytes * 8 / x.numel(), 4))
        e2 = encode(x, fmt)
        check(f"deterministic[{fmt}]", all(torch.equal(e.parts[k], e2.parts[k]) for k in e.parts))
        buf.zero_()
        decode(e, out=view)
        check(f"decode_into_view[{fmt}]", torch.equal(view, y) and not bool(buf[:, :, :, :T].any())
              and not bool(buf[:, :, :, 2 * T:].any()))
        zero_head = y[1, :, 2]
        check(f"zero_head[{fmt}]", bool((zero_head == 0).all()))
    # 4. 機制（合成資料，只看大小順序）：位元越少誤差越大；NV 的 scale 比 MX 細；旋轉讓 per-token INT4 的 K 誤差變小
    # （K 有固定偏移時 per-token 的 K8 反而比 per-channel 的 K4 差——這正是 Qwen 的情況，所以這裡只比 V。）
    check("err_order_bits_V", errs["kv-int8_token-int4_kivi"]["V"] < errs["kv-int4_kivi-int2_kivi"]["V"])
    check("nvfp4_le_mxfp4", errs["nvfp4"]["V"] <= errs["mxfp4"]["V"], nv=errs["nvfp4"], mx=errs["mxfp4"])
    # 旋轉只對「每個 token 不同 channel 的零星離群值」有用；固定偏移攤到所有 channel 反而變差（合成資料 K 就是後者）。
    g = torch.Generator().manual_seed(1)
    xr = torch.randn((L, 2, H, T, D), generator=g)
    ch = torch.randint(0, D, (L, 2, H, T, 1), generator=g)
    xr.scatter_(4, ch, 50.0 * torch.sign(torch.randn((L, 2, H, T, 1), generator=g)))
    xr = xr.to(torch.bfloat16)
    eh, ep = C.rel_err(xr, decode(encode(xr, "int4h_token"))), C.rel_err(xr, C.decode(C.encode(xr, "int4_token")))
    check("hadamard_helps_sparse_outliers", eh["K"] < ep["K"] and eh["V"] < ep["V"], rotated=eh, plain=ep)
    eh0, ep0 = C.rel_err_centered(x, decode(encode(x, "int4h_token"))), C.rel_err_centered(x, C.decode(C.encode(x, "int4_token")))
    res["note_rotation_constant_offset"] = {"rotated_cenK": eh0["K"], "plain_cenK": ep0["K"],
                                            "note": "固定偏移的 K：旋轉後 per-token INT4 變差（記錄，不判）"}
    # 5. GPU 與 CPU 逐位元相同
    if gpu and torch.cuda.is_available():
        xg = x.cuda()
        for fmt in NEW_FULL + MIXED_EXAMPLES:
            ec, eg = encode(x, fmt), encode(xg, fmt)
            same_enc = all(torch.equal(ec.parts[k], eg.parts[k].cpu()) for k in ec.parts)
            same_dec = torch.equal(decode(ec), decode(eg).cpu())
            check(f"gpu_eq_cpu[{fmt}]", same_enc and same_dec, enc_bitwise=same_enc, dec_bitwise=same_dec)
    else:
        res["checks"]["gpu_eq_cpu"] = "NOT_MEASURED（沒開 --gpu）"
    res["bits_per_value"] = {f: round(bits_per_value(f), 4) for f in
                             ["bf16"] + C.FORMATS[1:] + NEW_FULL + MIXED_EXAMPLES}
    res["all_pass"] = ok
    res["synthetic_note"] = "合成資料（高斯＋4 個 channel 偏移 200），誤差只驗機制，不代表真實 KV"
    if out_path:
        json.dump(res, open(out_path, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res["bits_per_value"], indent=1))
    print("ALL PASS" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test(a.out, a.gpu) else 1)
    ap.print_help()
