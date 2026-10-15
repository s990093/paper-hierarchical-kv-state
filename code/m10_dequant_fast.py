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

支援：fp8_head／fp8_channel／fp8_token（e4m3fn，scale 依粒度廣播）、int8_token。其他格式退回 m10_codec_x.decode。
"""
from __future__ import annotations

import argparse
import json
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
    r = pid * ROWS + tl.arange(0, ROWS)                  # 列號（攤平的 L×2×H×T）
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


def supported(fmt: str) -> bool:
    return fmt in ("fp8_head", "fp8_channel", "fp8_token", "int8_token")


def decode_fast(e, out: torch.Tensor | None = None, rows: int = 16) -> torch.Tensor:
    if not supported(e.fmt):
        import m10_codec_x as X
        return X.decode(e, out)
    L, two, H, T, D = e.shape
    q, s = e.parts["q"], e.parts["scale"]
    assert q.is_contiguous() and q.shape == e.shape and q.stride(-1) == 1
    if out is None:
        out = torch.empty(e.shape, dtype=torch.bfloat16, device=q.device)
    assert out.stride(-1) == 1 and out.dtype == torch.bfloat16
    sb = s.expand(L, two, H, T, D)                      # 廣播 → stride 0
    R = L * two * H * T
    fp8 = e.fmt.startswith("fp8")
    grid = (triton.cdiv(R, rows),)
    _dq_kernel[grid](q if fp8 else q, sb, _fp8_lut(q.device), out, R, T, H,
                     q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                     sb.stride(0), sb.stride(1), sb.stride(2), sb.stride(3), sb.stride(4),
                     out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                     D=D, IS_FP8=fp8, ROWS=rows)
    return out


# ---- 對照組：PyTorch 官方編譯器（torch.compile／Inductor）自動融合同一段 PyTorch 寫法（不是手寫 kernel）
def _torch_ref(q, s, out):
    out.copy_((q.view(C.FP8).float() * s).to(torch.bfloat16))


def _torch_ref_int8(q, s, out):
    out.copy_((q.float() * s).to(torch.bfloat16))


_COMPILED = {}


def decode_compiled(e, out):
    """torch.compile 版本：和 m10_codec.decode 一樣的運算，交給 Inductor 自動融合。只在 bench 用。"""
    fp8 = e.fmt.startswith("fp8")
    key = "fp8" if fp8 else "int8"
    if key not in _COMPILED:
        _COMPILED[key] = torch.compile(_torch_ref if fp8 else _torch_ref_int8, dynamic=False)
    _COMPILED[key](e.parts["q"], e.parts["scale"], out)
    return out


def install():
    """讓 m7_restore_harness 的載入線改用融合 kernel（不支援的格式自動退回原本的 decode）。大掃描的速度實驗才呼叫。"""
    import m7_restore_harness as RH
    import m10_codec_x as X
    X.install()
    RH.decode = lambda e, out=None: decode_fast(e, out) if supported(e.fmt) else X.decode(e, out)


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
    for fmt in ("fp8_head", "fp8_channel", "fp8_token", "int8_token"):
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
            for fmt in ("fp8_head", "fp8_token", "int8_token"):
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
