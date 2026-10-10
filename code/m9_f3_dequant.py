#!/usr/bin/env python
"""m9_f3_dequant.py — F3 工作 A：在 MI300X 上反量化「一個 64 MiB 的 KV chunk」要多久。

chunk 的形狀（Llama-3.1-8B，GQA）：512 token × 32 層 × K/V × 8 head × 128 dim
= 33,554,432 個值；BF16 = 64 MiB。這裡把它攤成 ROWS × 128，ROWS = 32·2·512·8 = 262,144，
每一列 = 一個 (層, K/V, token, head)。

量的東西（每一種都對照 FP32 參考答案檢查正確性）：
  INT8 → BF16，每 (token, head) 一個 FP32 scale（= D6 §2.1 的位元組模型：32+1 MiB）
  INT8 → BF16，每 32 個值一個 FP32 scale（per-group g=32）
  FP8 e4m3fnuz → BF16，整個 tensor 一個 scale（= vLLM --kv-cache-dtype fp8 的靜態縮放）
  FP8 e4m3fnuz → BF16，每 (token, head) 一個 scale（= fp8_per_token_head）
  INT4 → BF16，兩個 nibble 一個 byte，每 (token, head) 一個 FP32 scale（對稱，零點 8）
  INT4 → BF16，vLLM int4_per_token_head 的 scale 格式（scale 的 FP32 位元低 4 bit 放零點），不含 RHT

實作：plain torch（幾種寫法）、torch.compile、Triton、vLLM 的 convert_fp8（只有 FP8 有現成的獨立 kernel）。
計時：HIP event；每次量之前先讓 GPU 睡一下（torch.cuda._sleep），讓 CPU 把整段指令排進佇列，
      量到的是 GPU 上的執行時間，不含 CPU 發 kernel 的空檔。
cold：輪流用 8 組不同的 buffer（> MI300X 256 MB Infinity Cache）；hot：同一組 buffer。

用法（一定要包 GPU 鎖與 guard）：
  m7run f3-dequant flock /mlsteam/data/tiara/gpu.lock <py> code/m7_guard_run.py <py> code/m9_f3_dequant.py --out <csv>
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time
import traceback
import zlib
from datetime import datetime, timezone

RUN_ID = os.environ.get("RUN_ID", "no-run-id")
RUN_DIR = os.path.join(os.environ.get("TIARA_RUNS", "/mlsteam/data/tiara/runs"), RUN_ID)

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, help="repo 內的摘要 CSV（絕對路徑）")
ap.add_argument("--reps", type=int, default=50)
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--nsets", type=int, default=8, help="cold 模式輪流用幾組 buffer")
ap.add_argument("--skip-compile", action="store_true")
args = ap.parse_args()
OUT = os.path.abspath(args.out)
os.makedirs(RUN_DIR, exist_ok=True)
os.chdir(RUN_DIR)   # ROCm 當掉時的 gpucore.* 會掉在 cwd → 留在 run 目錄

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

assert torch.cuda.is_available(), "no GPU"
DEV = "cuda"
L, KV, T, H, D = 32, 2, 512, 8, 128
ROWS = L * KV * T * H            # 262,144
N = ROWS * D                     # 33,554,432
assert N * 2 == 64 * 1024**2, "BF16 chunk 應該剛好 64 MiB"
FP8 = torch.float8_e4m3fnuz if "gfx94" in torch.cuda.get_device_properties(0).gcnArchName else torch.float8_e4m3fn
TS = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
FAIL = []   # (variant, 完整錯誤)

# ───────────────────────────── Triton kernels ─────────────────────────────


@triton.jit
def _dq_rowscale(q_ptr, s_ptr, o_ptr, n_rows, G: tl.constexpr, BR: tl.constexpr):
    """每列 G 個值共用一個 FP32 scale；q 可以是 int8 或 fp8。"""
    pid = tl.program_id(0)
    r = pid * BR + tl.arange(0, BR)
    c = tl.arange(0, G)
    m = r[:, None] < n_rows
    off = r[:, None].to(tl.int64) * G + c[None, :]
    q = tl.load(q_ptr + off, mask=m, other=0).to(tl.float32)
    s = tl.load(s_ptr + r, mask=r < n_rows, other=0.0)
    tl.store(o_ptr + off, (q * s[:, None]).to(tl.bfloat16), mask=m)


@triton.jit
def _dq_tensorscale(q_ptr, o_ptr, scale, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    q = tl.load(q_ptr + off, mask=m, other=0).to(tl.float32)
    tl.store(o_ptr + off, (q * scale).to(tl.bfloat16), mask=m)


@triton.jit
def _dq_int4_sym(p_ptr, s_ptr, o_ptr, n_rows, HALF: tl.constexpr, BR: tl.constexpr):
    """p: [rows, HALF] uint8，低 nibble = 偶數 dim，高 nibble = 奇數 dim；值 = (nibble-8)*scale。"""
    pid = tl.program_id(0)
    r = pid * BR + tl.arange(0, BR)
    c = tl.arange(0, HALF)
    m = r[:, None] < n_rows
    p = tl.load(p_ptr + r[:, None].to(tl.int64) * HALF + c[None, :], mask=m, other=0).to(tl.int32)
    s = tl.load(s_ptr + r, mask=r < n_rows, other=0.0)[:, None]
    lo = ((p & 0xF) - 8).to(tl.float32) * s
    hi = (((p >> 4) & 0xF) - 8).to(tl.float32) * s
    v = tl.reshape(tl.join(lo, hi), (BR, 2 * HALF))
    r2 = pid * BR + tl.arange(0, BR)
    c2 = tl.arange(0, 2 * HALF)
    tl.store(o_ptr + r2[:, None].to(tl.int64) * (2 * HALF) + c2[None, :], v.to(tl.bfloat16),
             mask=r2[:, None] < n_rows)


@triton.jit
def _dq_int4_vllmfmt(p_ptr, sbits_ptr, o_ptr, n_rows, HALF: tl.constexpr, BR: tl.constexpr):
    """vLLM int4_per_token_head 的 scale 格式：FP32 scale 的位元，低 4 bit 換成零點。值 = (nibble-zp)*scale。不含 RHT。"""
    pid = tl.program_id(0)
    r = pid * BR + tl.arange(0, BR)
    c = tl.arange(0, HALF)
    m = r[:, None] < n_rows
    p = tl.load(p_ptr + r[:, None].to(tl.int64) * HALF + c[None, :], mask=m, other=0).to(tl.int32)
    bits = tl.load(sbits_ptr + r, mask=r < n_rows, other=0)
    zp = (bits & 0xF).to(tl.float32)[:, None]
    s = (bits & -16).to(tl.float32, bitcast=True)[:, None]
    lo = ((p & 0xF).to(tl.float32) - zp) * s
    hi = (((p >> 4) & 0xF).to(tl.float32) - zp) * s
    v = tl.reshape(tl.join(lo, hi), (BR, 2 * HALF))
    tl.store(o_ptr + r[:, None].to(tl.int64) * (2 * HALF) + tl.arange(0, 2 * HALF)[None, :],
             v.to(tl.bfloat16), mask=r[:, None] < n_rows)


def tr_rowscale(q, s, o, G, BR=64):
    n_rows = q.numel() // G
    _dq_rowscale[(triton.cdiv(n_rows, BR),)](q, s, o, n_rows, G=G, BR=BR)


def tr_tensorscale(q, scale, o, BLOCK=4096):
    _dq_tensorscale[(triton.cdiv(q.numel(), BLOCK),)](q, o, scale, q.numel(), BLOCK=BLOCK)


def tr_int4_sym(p, s, o, BR=64):
    _dq_int4_sym[(triton.cdiv(ROWS, BR),)](p, s, o, ROWS, HALF=D // 2, BR=BR)


def tr_int4_vllmfmt(p, sbits, o, BR=64):
    _dq_int4_vllmfmt[(triton.cdiv(ROWS, BR),)](p, sbits, o, ROWS, HALF=D // 2, BR=BR)


# ───────────────────────────── 資料產生 ─────────────────────────────


def make_set(kind: str):
    """回傳 (inputs dict, out tensor, ref fp32 tensor 的產生函式)。資料是隨機但合理的 KV 量級。"""
    g = torch.Generator(device=DEV)
    g.manual_seed(1234 + zlib.crc32(kind.encode()) % 1000)
    out = torch.empty(ROWS, D, dtype=torch.bfloat16, device=DEV)
    if kind == "int8_ptk":
        q = torch.randint(-127, 128, (ROWS, D), dtype=torch.int8, device=DEV, generator=g)
        s = (torch.rand(ROWS, 1, device=DEV, generator=g) * 0.05 + 1e-3).float()
        return {"q": q, "s": s}, out
    if kind == "int8_g32":
        q = torch.randint(-127, 128, (ROWS * D // 32, 32), dtype=torch.int8, device=DEV, generator=g)
        s = (torch.rand(ROWS * D // 32, 1, device=DEV, generator=g) * 0.05 + 1e-3).float()
        return {"q": q, "s": s}, out.view(ROWS * D // 32, 32)
    if kind == "fp8_tensor":
        x = (torch.randn(ROWS, D, device=DEV, generator=g) * 20).clamp(-240, 240).to(FP8)
        return {"x": x, "scale": 1.0, "scale_t": torch.tensor(1.0, device=DEV)}, out
    if kind == "fp8_ptk":
        x = (torch.randn(ROWS, D, device=DEV, generator=g) * 20).clamp(-240, 240).to(FP8)
        s = (torch.rand(ROWS, 1, device=DEV, generator=g) * 0.05 + 1e-3).float()
        return {"x": x, "s": s}, out
    if kind == "int4_sym":
        p = torch.randint(0, 256, (ROWS, D // 2), dtype=torch.uint8, device=DEV, generator=g)
        s = (torch.rand(ROWS, 1, device=DEV, generator=g) * 0.05 + 1e-3).float()
        return {"p": p, "s": s}, out
    if kind == "int4_vllmfmt":
        p = torch.randint(0, 256, (ROWS, D // 2), dtype=torch.uint8, device=DEV, generator=g)
        s = (torch.rand(ROWS, device=DEV, generator=g) * 0.05 + 1e-3).float()
        zp = torch.randint(0, 16, (ROWS,), dtype=torch.int32, device=DEV, generator=g)
        bits = (s.view(torch.int32) & -16) | zp
        return {"p": p, "sbits": bits.contiguous()}, out
    raise ValueError(kind)


def reference(kind, inp):
    """FP32 參考答案（攤平成 ROWS × D）。"""
    if kind == "int8_ptk":
        return inp["q"].float() * inp["s"]
    if kind == "int8_g32":
        return (inp["q"].float() * inp["s"]).view(ROWS, D)
    if kind == "fp8_tensor":
        return inp["x"].float() * inp["scale"]
    if kind == "fp8_ptk":
        return inp["x"].float() * inp["s"]
    if kind == "int4_sym":
        p = inp["p"].int()
        q = torch.stack([(p & 0xF) - 8, ((p >> 4) & 0xF) - 8], -1).view(ROWS, D).float()
        return q * inp["s"]
    if kind == "int4_vllmfmt":
        p = inp["p"].int()
        bits = inp["sbits"]
        zp = (bits & 0xF).float()[:, None]
        s = (bits & -16).view(torch.float32)[:, None]
        q = torch.stack([p & 0xF, (p >> 4) & 0xF], -1).view(ROWS, D).float()
        return (q - zp) * s
    raise ValueError(kind)


def in_bytes(inp):
    return sum(v.numel() * v.element_size() for v in inp.values() if torch.is_tensor(v) and v.dim() > 0)


# ───────────────────────────── 實作清單 ─────────────────────────────
# 每個實作：fn(inp, out) -> 結果 tensor（可能就是 out，也可能是新配置的）

def _int4_unpack_torch(p, zp=8):
    return torch.stack([(p & 0xF), (p >> 4)], -1).view(ROWS, D)


IMPLS = {
    "int8_ptk": {
        "torch_naive_fp32": lambda i, o: (i["q"].to(torch.float32) * i["s"]).to(torch.bfloat16),
        "torch_bf16": lambda i, o: i["q"].to(torch.bfloat16) * i["s"].to(torch.bfloat16),
        "torch_mul_out": lambda i, o: torch.mul(i["q"], i["s"], out=o),
        "triton": lambda i, o: (tr_rowscale(i["q"], i["s"], o, G=D), o)[1],
    },
    "int8_g32": {
        "torch_naive_fp32": lambda i, o: (i["q"].to(torch.float32) * i["s"]).to(torch.bfloat16),
        "torch_bf16": lambda i, o: i["q"].to(torch.bfloat16) * i["s"].to(torch.bfloat16),
        "torch_mul_out": lambda i, o: torch.mul(i["q"], i["s"], out=o),
        "triton": lambda i, o: (tr_rowscale(i["q"], i["s"], o, G=32), o)[1],
    },
    "fp8_tensor": {
        "torch_naive_fp32": lambda i, o: (i["x"].to(torch.float32) * i["scale"]).to(torch.bfloat16),
        "torch_to_bf16_only": lambda i, o: i["x"].to(torch.bfloat16),          # scale=1.0 時的純轉型
        "torch_copy_out": lambda i, o: o.copy_(i["x"]),                         # scale=1.0，一個 kernel
        "torch_bf16_mul": lambda i, o: i["x"].to(torch.bfloat16) * i["scale"],    # 兩個 kernel
        "vllm_convert_fp8": None,   # 下面填
        "triton": lambda i, o: (tr_tensorscale(i["x"], i["scale"], o), o)[1],
    },
    "fp8_ptk": {
        "torch_naive_fp32": lambda i, o: (i["x"].to(torch.float32) * i["s"]).to(torch.bfloat16),
        "torch_bf16": lambda i, o: i["x"].to(torch.bfloat16) * i["s"].to(torch.bfloat16),
        "triton": lambda i, o: (tr_rowscale(i["x"], i["s"], o, G=D), o)[1],
    },
    "int4_sym": {
        "torch_naive_fp32": lambda i, o: ((_int4_unpack_torch(i["p"]).to(torch.float32) - 8) * i["s"]).to(torch.bfloat16),
        "torch_bf16": lambda i, o: (_int4_unpack_torch(i["p"]).to(torch.bfloat16) - 8) * i["s"].to(torch.bfloat16),
        "triton": lambda i, o: (tr_int4_sym(i["p"], i["s"], o), o)[1],
    },
    "int4_vllmfmt": {
        "triton": lambda i, o: (tr_int4_vllmfmt(i["p"], i["sbits"], o), o)[1],
    },
}

# vLLM 的獨立 FP8 反量化 kernel（csrc/libtorch_stable/cache_kernels.cu convert_fp8，原始碼註明 "Only for testing"）
VLLM_VER = "not_imported"
try:
    import vllm  # noqa: E402
    from vllm import _custom_ops as vops  # noqa: E402
    VLLM_VER = vllm.__version__
    # 用 vLLM paged cache 的自然形狀：每個 block = 16 token × 8 head × 128 = 16,384 值；共 32·2·32 = 2048 個 block
    def _vllm_cvt(i, o):
        vops.convert_fp8(o.view(2048, 16384), i["x"].view(torch.uint8).view(2048, 16384), float(i["scale"]), "fp8")
        return o
    IMPLS["fp8_tensor"]["vllm_convert_fp8"] = _vllm_cvt
except Exception:  # noqa: BLE001
    FAIL.append(("import vllm._custom_ops", traceback.format_exc()))
    IMPLS["fp8_tensor"].pop("vllm_convert_fp8")

if not args.skip_compile:
    for kind in ("int8_ptk", "int8_g32", "fp8_tensor", "int4_sym"):
        base = IMPLS[kind]["torch_naive_fp32"]
        IMPLS[kind]["torch_compile"] = torch.compile(base, dynamic=False)


# ───────────────────────────── 計時 ─────────────────────────────

def _sleep_ok():
    try:
        torch.cuda._sleep(1000)
        torch.cuda.synchronize()
        return True
    except Exception:  # noqa: BLE001
        FAIL.append(("torch.cuda._sleep", traceback.format_exc()))
        return False


SLEEP_OK = _sleep_ok()
SLEEP_CYCLES = 2_000_000   # 約 1 ms 量級；只是讓 CPU 先把指令排好


def bench(fn, sets, cold: bool):
    """回傳每次的 ms 清單。"""
    times = []
    nsets = len(sets)
    total = args.warmup + args.reps
    for k in range(total):
        inp, out = sets[k % nsets] if cold else sets[0]
        torch.cuda.synchronize()
        if SLEEP_OK:
            torch.cuda._sleep(SLEEP_CYCLES)
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        fn(inp, out)
        en.record()
        en.synchronize()
        if k >= args.warmup:
            times.append(st.elapsed_time(en))
    return times


def pct(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    props = torch.cuda.get_device_properties(0)
    env = {"run_id": RUN_ID, "ts": TS, "gpu": props.name, "arch": props.gcnArchName,
           "torch": torch.__version__, "hip": torch.version.hip, "triton": triton.__version__,
           "vllm": VLLM_VER, "fp8_dtype": str(FP8), "sleep_ok": SLEEP_OK,
           "reps": args.reps, "warmup": args.warmup, "nsets": args.nsets,
           "shape": f"{L}L x {KV}KV x {T}tok x {H}h x {D}d = {N} values", "python": sys.version}
    print(json.dumps(env, ensure_ascii=False), flush=True)
    rows, raw = [], {}
    for kind, impls in IMPLS.items():
        sets = [make_set(kind) for _ in range(args.nsets)]
        ref = reference(kind, sets[0][0])
        ib = in_bytes(sets[0][0])
        ob = N * 2
        for name, fn in impls.items():
            tag = f"{kind}/{name}"
            try:
                res = fn(*sets[0])
                torch.cuda.synchronize()
                res = res.reshape(ROWS, D).float()
                err = (res - ref).abs()
                tol = ref.abs() * 2.0 ** -7 + 1e-30
                correct = bool((err <= tol).all().item())
                max_rel = float((err / (ref.abs() + 1e-30)).max().item())
                for cold in (True, False):
                    ts = bench(fn, sets, cold)
                    med = statistics.median(ts)
                    row = {"run_id": RUN_ID, "ts": TS, "precision": kind, "impl": name,
                           "cache_state": "cold" if cold else "hot", "n_reps": len(ts),
                           "median_ms": round(med, 5), "p90_ms": round(pct(ts, 0.9), 5),
                           "min_ms": round(min(ts), 5), "mean_ms": round(statistics.mean(ts), 5),
                           "in_bytes": ib, "out_bytes": ob,
                           "eff_GBps": round((ib + ob) / (med * 1e-3) / 1e9, 1),
                           "correct": correct, "max_rel_err": f"{max_rel:.3e}",
                           "gpu": props.name, "torch": torch.__version__, "triton": triton.__version__,
                           "vllm": VLLM_VER, "note": ""}
                    rows.append(row)
                    raw[f"{tag}/{row['cache_state']}"] = ts
                    print(f"[f3] {tag:32s} {row['cache_state']:4s} med={med:.4f} ms p90={row['p90_ms']:.4f} "
                          f"eff={row['eff_GBps']} GB/s correct={correct} max_rel={max_rel:.2e}", flush=True)
            except Exception:  # noqa: BLE001
                tb = traceback.format_exc()
                FAIL.append((tag, tb))
                print(f"[f3] 🔴 {tag} FAILED\n{tb}", flush=True)
                rows.append({"run_id": RUN_ID, "ts": TS, "precision": kind, "impl": name,
                             "cache_state": "NA", "n_reps": 0, "median_ms": "FAILED", "p90_ms": "FAILED",
                             "min_ms": "", "mean_ms": "", "in_bytes": ib, "out_bytes": ob, "eff_GBps": "",
                             "correct": "", "max_rel_err": "", "gpu": props.name, "torch": torch.__version__,
                             "triton": triton.__version__, "vllm": VLLM_VER,
                             "note": tb.strip().splitlines()[-1][:200]})
        del sets, ref
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(RUN_DIR, "f3_dequant_raw.json"), "w") as f:
        json.dump({"env": env, "raw_ms": raw, "failures": FAIL}, f, indent=1)
    with open(os.path.join(RUN_DIR, "f3_dequant.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[f3] wrote {len(rows)} rows -> {OUT}", flush=True)
    if FAIL:
        print(f"[f3] ⚠️ {len(FAIL)} failures（完整錯誤在 f3_dequant_raw.json）:", flush=True)
        for t, tb in FAIL:
            print(f"--- {t}\n{tb}", flush=True)
    return 0


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    print(f"[f3] wall {time.time() - t0:.1f} s", flush=True)
    sys.exit(rc)
