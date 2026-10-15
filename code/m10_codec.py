"""m10_codec.py — 精度這條線的 KV codec（03_plan_h200.md §1、§2.1、§2.5、§6）

一個 block 的 KV：x[L, 2(K/V), H, T, D]，BF16（T＝512，和 m7_model.host_chunk 同形狀）。
encode(x, fmt) → Enc（實際要存的位元組）；decode(enc, out) → 寫回 BF16（out 可以是 KV 緩衝的非連續視圖）。

格式（全部用 torch 的一般運算，不用 Hopper／MI300X 的 FP8 matmul；attention 路徑和 BF16 配置相同，§1）：
  bf16            不壓縮（對照用）
  fp8_head        e4m3fn，scale＝每（層, K/V, KV head）的 amax ÷ 448，用這個 block 自己算（§1 的預設）
  fp8_channel     e4m3fn，scale 每（層, K/V, head, channel）
  fp8_token       e4m3fn，scale 每（層, K/V, head, token）
  fp8k0_<g>       E1-K0：K 先減掉每（層, head, channel）的平均（平均另存 BF16），殘差做 fp8_<g>；V 照 fp8_<g>
  int8_token      對稱 INT8，scale 每（層, K/V, head, token）＝amax ÷ 127（vLLM int8_per_token_head 的分組）
  int4_token      對稱 INT4，scale 每（層, K/V, head, token）＝amax ÷ 7，兩個值塞一個 byte（第 2 階 (a)）
  int4_kivi       KIVI 式非對稱 INT4：K 每 channel、沿 token 每 32 個一組；V 每 token、沿 channel 每 32 個一組；
                  scale、zero 存 FP16（照 KIVI 原始實作）（第 2 階 (b)）

寫死的細節：
- e4m3fn 一律自己先截在 ±448 再轉型。torch 2.10／2.12（ROCm）超出範圍會變 NaN、不會飽和
  （2026-10-15 在 MI300X 上實測）；版本之間行為不同（P04 §3），所以不靠 torch 的轉型處理溢位。
- 不用 MI300X 原生的 float8_e4m3fnuz（最大 240，和 H200 的 e4m3fn 不同格式）。
- FP8 的位元組用 uint8 存（.view），搬運、pin 都不碰 float8 dtype。
- amax＝0 的群組 scale 設 1（全 0 的 head 不會除以 0）。
- 除以常數一律用 _div_c（除以同裝置的 0 維 tensor）。torch 在 GPU 上「除以 Python 純量」會改成乘倒數，
  和 CPU 的真除法差最後一位，scale 就不同（2026-10-15 MI300X 實測：scale 不同 → FP8 位元組不同）。
- scale 用 FP32 存（§1：Qwen 一個 block 的 fp8_head scale＝28×2×4×4 B＝896 B）；KIVI 照原作用 FP16。
- 存的位元組＝Enc.nbytes（各部分實際 tensor 的大小），限速器照這個計時（§1「限速器的位元組」）。

自測：python code/m10_codec.py --self-test [--out x.json]（CPU 一定跑；有 GPU 時另比 GPU 與 CPU 是否逐位元相同）
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field

import torch

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
KIVI_GROUP = 32
_FP8_GRAN = {"head": (3, 4), "channel": (3,), "token": (4,)}   # 對 x[L,2,H,T,D] 取 amax 的維度


@dataclass
class Enc:
    fmt: str
    shape: tuple
    parts: dict = field(default_factory=dict)

    @property
    def kind(self) -> str:
        """逐 block 來源記錄用：bf16／fp8／int8／int4。"""
        if self.fmt == "bf16":
            return "bf16"
        return "fp8" if self.fmt.startswith("fp8") else self.fmt.split("_")[0]

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.parts.values())

    def map(self, fn) -> "Enc":
        return Enc(self.fmt, self.shape, {k: fn(v) for k, v in self.parts.items()})

    def to(self, device, non_blocking=False) -> "Enc":
        return self.map(lambda t: t.to(device, non_blocking=non_blocking))

    def pin(self) -> "Enc":
        return self.map(lambda t: t.pin_memory())

    def empty_like(self, device) -> "Enc":
        return self.map(lambda t: torch.empty_like(t, device=device))

    def copy_from(self, src: "Enc", non_blocking=False) -> "Enc":
        assert src.fmt == self.fmt and src.parts.keys() == self.parts.keys()
        for k, t in self.parts.items():
            t.copy_(src.parts[k], non_blocking=non_blocking)
        return self


def _div_c(t: torch.Tensor, c: float) -> torch.Tensor:
    return t / torch.full((), c, dtype=t.dtype, device=t.device)


# ---------------------------------------------------------------- FP8
def _fp8_enc(x: torch.Tensor, gran: str):
    xf = x.float()
    amax = xf.abs().amax(dim=_FP8_GRAN[gran], keepdim=True)
    scale = torch.where(amax > 0, _div_c(amax, FP8_MAX), torch.ones_like(amax))
    q = (xf / scale).clamp_(-FP8_MAX, FP8_MAX).to(FP8)
    return q.view(torch.uint8), scale


def _fp8_dec(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.view(FP8).float() * scale


# ---------------------------------------------------------------- INT4 打包
def _pack4(u: torch.Tensor) -> torch.Tensor:
    """u：0..15 的整數（最後一維是偶數）→ uint8，偶數 index 放低 4 位。"""
    u = u.to(torch.uint8)
    return u[..., 0::2] | (u[..., 1::2] << 4)


def _unpack4(p: torch.Tensor) -> torch.Tensor:
    lo, hi = p & 0xF, p >> 4
    return torch.stack((lo, hi), dim=-1).flatten(-2)


# ---------------------------------------------------------------- encode / decode
def encode(x: torch.Tensor, fmt: str) -> Enc:
    assert x.dim() == 5 and x.shape[1] == 2 and x.dtype == torch.bfloat16, (x.shape, x.dtype)
    shape = tuple(x.shape)
    if fmt == "bf16":
        return Enc(fmt, shape, {"x": x.contiguous().clone()})
    if fmt.startswith("fp8_"):
        q, s = _fp8_enc(x, fmt[4:])
        return Enc(fmt, shape, {"q": q, "scale": s})
    if fmt.startswith("fp8k0_"):
        g = fmt[6:]
        # 平均用 float64 算再轉 BF16：CPU／GPU 的加總順序不同，float64 讓轉成 BF16 後幾乎不可能差一位
        mu = x[:, 0].double().mean(dim=2, keepdim=True).to(torch.bfloat16)   # 每（層, head, channel）
        r = x.float()
        r[:, 0] -= mu.float()                                   # 殘差留在 FP32 直接做 FP8，不先轉 BF16（避免兩次捨入）
        q, s = _fp8_enc(r, g)
        return Enc(fmt, shape, {"q": q, "scale": s, "k_mean": mu})
    if fmt == "int8_token":
        xf = x.float()
        amax = xf.abs().amax(dim=4, keepdim=True)
        s = torch.where(amax > 0, _div_c(amax, 127.0), torch.ones_like(amax))
        q = torch.round(xf / s).clamp_(-127, 127).to(torch.int8)
        return Enc(fmt, shape, {"q": q, "scale": s})
    if fmt == "int4_token":
        xf = x.float()
        amax = xf.abs().amax(dim=4, keepdim=True)
        s = torch.where(amax > 0, _div_c(amax, 7.0), torch.ones_like(amax))
        q = torch.round(xf / s).clamp_(-7, 7) + 8               # 1..15
        return Enc(fmt, shape, {"q": _pack4(q), "scale": s})
    if fmt == "int4_kivi":
        L, _, H, T, D = shape
        G = KIVI_GROUP
        assert T % G == 0 and D % G == 0
        # K：每 channel，沿 token 分組 → [L,H,T/G,G,D]，對 G 取 min/max
        k = x[:, 0].float().view(L, H, T // G, G, D)
        kmn, kmx = k.amin(dim=3, keepdim=True), k.amax(dim=3, keepdim=True)
        # V：每 token，沿 channel 分組 → [L,H,T,D/G,G]，對 G 取 min/max
        v = x[:, 1].float().view(L, H, T, D // G, G)
        vmn, vmx = v.amin(dim=4, keepdim=True), v.amax(dim=4, keepdim=True)
        parts = {}
        for name, t, mn, mx in (("k", k, kmn, kmx), ("v", v, vmn, vmx)):
            mn16, sc16 = mn.half(), _div_c(mx - mn, 15.0).half()
            sc = torch.where(sc16 > 0, sc16, torch.ones_like(sc16)).float()
            q = torch.round((t - mn16.float()) / sc).clamp_(0, 15)
            parts[f"{name}_q"] = _pack4(q.reshape(L, H, T, D))
            parts[f"{name}_scale"] = sc.half()
            parts[f"{name}_zero"] = mn16
        return Enc(fmt, shape, parts)
    raise ValueError(fmt)


def decode(e: Enc, out: torch.Tensor | None = None) -> torch.Tensor:
    """寫進 out（BF16，形狀 e.shape，可以是非連續視圖）；out=None 時新配一塊。"""
    p = e.parts
    if e.fmt == "bf16":
        y = p["x"]
    elif e.fmt.startswith("fp8_"):
        y = _fp8_dec(p["q"], p["scale"]).to(torch.bfloat16)
    elif e.fmt.startswith("fp8k0_"):
        y = _fp8_dec(p["q"], p["scale"])
        y[:, 0] += p["k_mean"].float()
        y = y.to(torch.bfloat16)
    elif e.fmt in ("int8_token",):
        y = (p["q"].float() * p["scale"]).to(torch.bfloat16)
    elif e.fmt == "int4_token":
        q = _unpack4(p["q"]).float() - 8
        y = (q * p["scale"]).to(torch.bfloat16)
    elif e.fmt == "int4_kivi":
        L, _, H, T, D = e.shape
        G = KIVI_GROUP
        k = _unpack4(p["k_q"]).float().view(L, H, T // G, G, D) * p["k_scale"].float() + p["k_zero"].float()
        v = _unpack4(p["v_q"]).float().view(L, H, T, D // G, G) * p["v_scale"].float() + p["v_zero"].float()
        y = torch.stack((k.reshape(L, H, T, D), v.reshape(L, H, T, D)), dim=1).to(torch.bfloat16)
    else:
        raise ValueError(e.fmt)
    if out is None:
        return y.clone() if y is p.get("x") else y
    out.copy_(y)
    return out


FORMATS = ["bf16", "fp8_head", "fp8_channel", "fp8_token", "fp8k0_head", "fp8k0_channel", "fp8k0_token",
           "int8_token", "int4_token", "int4_kivi"]


def rel_err(x: torch.Tensor, y: torch.Tensor) -> dict:
    """相對 L2 誤差，K、V 分開（§3.3 第 4 步記錄每個 FP8 block 的重建誤差）。"""
    out = {}
    for j, n in ((0, "K"), (1, "V")):
        a, b = x[:, j].float(), y[:, j].float()
        out[n] = float((a - b).norm() / a.norm().clamp_min(1e-30))
    return out


# ---------------------------------------------------------------- 自測
def _synthetic(L=4, H=4, T=512, D=128, seed=0):
    """合成 KV：V 高斯；K 高斯＋少數 channel 的大固定偏移（模仿 Qwen 的 k_proj bias，P04 §4）。
    只用來驗 codec 的機制，不代表真實 KV 的誤差大小（真實 KV 在 D0 量）。"""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((L, 2, H, T, D), generator=g)
    x[:, 0, :, :, :4] += 200.0                     # 4 個 channel 的固定偏移
    x[0, 1, 0, 7, 3] = 1e4                         # 一個極端值（測 amax 與截斷）
    x[1, :, 2] = 0.0                               # 一整個 head 全 0（測 amax＝0）
    return x.to(torch.bfloat16)


def self_test(out_path=None) -> bool:
    torch.manual_seed(0)
    res = {"torch": torch.__version__, "hip": torch.version.hip, "cuda": torch.version.cuda,
           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu-only", "checks": {}}
    ok = True

    def check(name, cond, **info):
        nonlocal ok
        ok &= bool(cond)
        res["checks"][name] = {"pass": bool(cond), **info}
        print(("PASS " if cond else "FAIL ") + name, info if info else "", flush=True)

    x = _synthetic()
    L, _, H, T, D = x.shape
    bf16_bytes = x.numel() * 2
    # 1. 每種格式：往返沒有 NaN／Inf、誤差、位元組
    expect_bytes = {
        "bf16": bf16_bytes,
        "fp8_head": x.numel() + L * 2 * H * 4,
        "fp8_channel": x.numel() + L * 2 * H * D * 4,
        "fp8_token": x.numel() + L * 2 * H * T * 4,
        "int8_token": x.numel() + L * 2 * H * T * 4,
        "int4_token": x.numel() // 2 + L * 2 * H * T * 4,
        "int4_kivi": x.numel() // 2 + 2 * 2 * (L * H * (T // KIVI_GROUP) * D + L * H * T * (D // KIVI_GROUP)),
    }
    for g in ("head", "channel", "token"):
        expect_bytes[f"fp8k0_{g}"] = expect_bytes[f"fp8_{g}"] + L * H * D * 2
    errs = {}
    for fmt in FORMATS:
        e = encode(x, fmt)
        y = decode(e)
        finite = bool(torch.isfinite(y.float()).all())
        er = rel_err(x, y)
        errs[fmt] = er
        check(f"roundtrip_finite[{fmt}]", finite and y.shape == x.shape and y.dtype == torch.bfloat16, rel_err=er)
        check(f"nbytes[{fmt}]", e.nbytes == expect_bytes[fmt], nbytes=e.nbytes, expect=expect_bytes[fmt],
              ratio_vs_bf16=round(e.nbytes / bf16_bytes, 5))
        check(f"deterministic[{fmt}]", all(torch.equal(a, b) for a, b in zip(e.parts.values(), encode(x, fmt).parts.values())))
    check("bf16_exact", errs["bf16"]["K"] == 0 and errs["bf16"]["V"] == 0)
    # 2. FP8：截斷比例、scale、全 0 head
    for g in ("head", "channel", "token"):
        e = encode(x, f"fp8_{g}")
        qf = e.parts["q"].view(FP8).float()
        # 「被截在 ±448」＝轉型前 |x/scale| 超過 448。scale＝amax/448，所以只可能是 FP32 除法的最後一位
        clamped = int(((x.float() / e.parts["scale"]).abs() > FP8_MAX * (1 + 1e-6)).sum())
        check(f"fp8_{g}_no_overflow", bool(torch.isfinite(qf).all()) and clamped == 0,
              n_clamped=clamped, frac_q_at_448=float((qf.abs() == FP8_MAX).float().mean()))
        zero_head = decode(e)[1, :, 2]
        check(f"fp8_{g}_zero_head", bool((zero_head == 0).all()))
    # amax 元素本身必須精確還原（scale＝amax/448 → 448 → ×scale）
    e = encode(x, "fp8_head")
    check("fp8_head_amax_exact", float(decode(e)[0, 1, 0, 7, 3]) == float(x[0, 1, 0, 7, 3]),
          got=float(decode(e)[0, 1, 0, 7, 3]), want=float(x[0, 1, 0, 7, 3]))
    # 3. K 去偏移應該讓 K 的誤差下降（合成資料有固定偏移，機制檢查）
    check("fp8k0_reduces_K_err", errs["fp8k0_head"]["K"] < errs["fp8_head"]["K"],
          fp8_head_K=errs["fp8_head"]["K"], fp8k0_head_K=errs["fp8k0_head"]["K"])
    # 4. INT4 打包往返
    u = torch.randint(0, 16, (3, 5, 128), dtype=torch.uint8)
    check("int4_pack_roundtrip", torch.equal(_unpack4(_pack4(u)), u))
    # 5. 解碼寫進非連續視圖（KV 緩衝的 chunk 視圖就是非連續的）
    buf = torch.zeros((L, 2, H, 3 * T, D), dtype=torch.bfloat16)
    view = buf[:, :, :, T:2 * T, :]
    for fmt in FORMATS:
        buf.zero_()
        decode(encode(x, fmt), out=view)
        check(f"decode_into_view[{fmt}]", torch.equal(view, decode(encode(x, fmt))) and
              not bool(buf[:, :, :, :T].any()) and not bool(buf[:, :, :, 2 * T:].any()))
    # 6. GPU 與 CPU 逐位元相同（同一份 codec 在不同裝置上存出來的位元組一樣）
    if torch.cuda.is_available():
        xg = x.cuda()
        for fmt in FORMATS:
            ec, eg = encode(x, fmt), encode(xg, fmt)
            same_enc = all(torch.equal(ec.parts[k], eg.parts[k].cpu()) for k in ec.parts)
            same_dec = torch.equal(decode(ec), decode(eg).cpu())
            check(f"gpu_eq_cpu[{fmt}]", same_enc and same_dec, enc_bitwise=same_enc, dec_bitwise=same_dec)
        # pinned → GPU 暫存 → 解碼進視圖（還原器的路徑）
        e = encode(x, "fp8_head").pin()
        st = e.empty_like("cuda")
        st.copy_from(e, non_blocking=True)
        gbuf = torch.zeros((L, 2, H, 2 * T, D), dtype=torch.bfloat16, device="cuda")
        decode(st, out=gbuf[:, :, :, T:])
        check("pinned_staging_decode", torch.equal(gbuf[:, :, :, T:].cpu(), decode(e)))
    else:
        res["checks"]["gpu_eq_cpu"] = "NOT_MEASURED (no GPU)"
    res["all_pass"] = ok
    res["synthetic_note"] = "合成資料（高斯＋4 個 channel 偏移 200），誤差只驗機制，不代表真實 KV"
    if out_path:
        json.dump(res, open(out_path, "w"), indent=1, ensure_ascii=False)
    print("ALL PASS" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test(a.out) else 1)
    ap.print_help()
