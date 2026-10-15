#!/usr/bin/env python3
"""gpu_smoke.py — 證明 GPU 是「真的能算」而不是「偵測得到」（03_plan_h200 §R0 的 GPU smoke）

從平台 B 的 lab_bin/gpu_smoke.py 改成廠商中立（torch.cuda 在 CUDA 與 ROCm 上都能用；原版一開始就 assert ROCm）。
做四件事：
  1. BF16 8192³ matmul（TFLOPS）
  2. 數值正確性（I@Y == Y；不對就以非 0 結束——不要用這張卡量任何東西）
  3. HBM 裝置內複製（1 GiB）
  4. pinned H2D／D2H（1 GiB）——限速層下限與 CPU 層錨點的數量級

⚠️ smoke test 等級，**不可寫進論文**：單一 stream、沒調 BLAS、每項 10 次。可用的是數量級與「算得對」。
   正式的 H2D／D2H 是 m7_calib.py c0（R2），要走 tiara.sbatch 記錄殼。

  python code/gpu_smoke.py [--json out.json] [--reps 10]
"""
import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime

import torch

GiB = 1 << 30


def timed(fn, reps):
    """每次都同步，回傳每次的秒數（第 0 次之前先暖機一次）。"""
    fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append(time.perf_counter() - t0)
    return out


def summ(vals):
    return {"median": round(statistics.median(vals), 3), "min": round(min(vals), 3), "max": round(max(vals), 3),
            "n": len(vals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="結果寫成 JSON")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--n", type=int, default=8192, help="matmul 的邊長")
    a = ap.parse_args()

    res = {"ts": datetime.now().astimezone().isoformat(timespec="seconds"),
           "torch": torch.__version__, "cuda": torch.version.cuda, "hip": torch.version.hip,
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "_note": "smoke test，不可寫進論文（03 §R0）"}
    print("=" * 66)
    print(f"torch {torch.__version__} | cuda {torch.version.cuda} | hip {torch.version.hip}")
    if not torch.cuda.is_available():
        res["error"] = "torch.cuda.is_available() == False"
        print("看不到 GPU", file=sys.stderr)
        _dump(a, res)
        return 2
    d = torch.device("cuda:0")
    p = torch.cuda.get_device_properties(0)
    arch = getattr(p, "gcnArchName", None) if torch.version.hip else f"sm_{p.major}{p.minor}"
    res.update(device_count=torch.cuda.device_count(), name=p.name, total_memory_gib=round(p.total_memory / GiB, 1),
               arch=arch, multi_processor_count=p.multi_processor_count)
    print(f"device: {p.name}  {p.total_memory / GiB:.1f} GiB  arch={arch}  visible={torch.cuda.device_count()}")
    try:
        res["cpu_affinity_count"] = len(os.sched_getaffinity(0))
    except AttributeError:
        pass
    print("=" * 66)

    # 1) 算力
    N = a.n
    x = torch.randn(N, N, device=d, dtype=torch.bfloat16)
    y = torch.randn(N, N, device=d, dtype=torch.bfloat16)
    ts = timed(lambda: x @ y, a.reps)
    tf = [2 * N ** 3 / t / 1e12 for t in ts]
    res["matmul_bf16_tflops"] = summ(tf) | {"N": N}
    print(f"[matmul  ] bf16 {N}^3   {statistics.median(ts)*1e3:8.2f} ms/iter   {statistics.median(tf):7.1f} TFLOPS (median)")
    del x, y

    # 2) 數值正確性
    e = torch.eye(1024, device=d, dtype=torch.float32)
    yy = torch.randn(1024, 1024, device=d, dtype=torch.float32)
    err = (e @ yy - yy).abs().max().item()
    res["correctness_max_abs_err"] = err
    res["correctness_ok"] = bool(err < 1e-4)
    print(f"[correct ] max|I@Y - Y| = {err:.3e}   -> {'OK' if err < 1e-4 else 'FAIL'}")

    # 3) HBM 裝置內複製（讀＋寫都算）
    g = torch.empty(GiB // 4, device=d, dtype=torch.float32)
    g2 = torch.empty_like(g)
    ts = timed(lambda: g2.copy_(g), a.reps)
    res["hbm_d2d_GBps"] = summ([2 * GiB / t / 1e9 for t in ts])
    print(f"[HBM d2d ] {res['hbm_d2d_GBps']['median']:8.1f} GB/s (median, read+write)")
    del g2

    # 4) pinned H2D／D2H
    h = torch.empty(GiB // 4, dtype=torch.float32, pin_memory=True)
    ts = timed(lambda: g.copy_(h, non_blocking=True), a.reps)
    res["h2d_pinned_GiBps"] = summ([GiB / t / GiB for t in ts])
    ts = timed(lambda: h.copy_(g, non_blocking=True), a.reps)
    res["d2h_pinned_GiBps"] = summ([GiB / t / GiB for t in ts])
    print(f"[H2D pin ] {res['h2d_pinned_GiBps']['median']:8.2f} GiB/s (median)")
    print(f"[D2H pin ] {res['d2h_pinned_GiBps']['median']:8.2f} GiB/s (median)")
    print("=" * 66)
    _dump(a, res)
    if not res["correctness_ok"]:
        print("數值不對，不要用這張卡量任何東西", file=sys.stderr)
        return 1
    print("GPU 可用：算得出來、數值正確、搬得動資料。")
    return 0


def _dump(a, res):
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        json.dump(res, open(a.json, "w"), indent=1)
        print(f"[ok] {a.json}")


if __name__ == "__main__":
    sys.exit(main())
