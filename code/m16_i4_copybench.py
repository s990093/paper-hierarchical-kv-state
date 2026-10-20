#!/usr/bin/env python3
"""m16_i4_copybench.py — 第三階段 I4：KV 從 CPU 載回 GPU 的「搬法」× GPU 忙不忙（MI300X）。

計畫：docs/stage3_20261019/experiment_matrix.md 的 I4（＋L2 的 X3）。**要 GPU**，只在第三階段窗口跑。
背景（舊實測）：vLLM 0.28 的 OffloadingConnector 在 MI300X 上 CPU 層只有 3.7–4.1 GiB/s，主因是「每次 copy 的固定成本 × copy 次數」
（`docs/research_20261010_followup/F1_offload_slowdown.md` §4.3–§4.6：約 3.5 µs／次送出、D2H 約 6 µs／次 GPU 時間；ROCm 7.2.2 的
hipMemcpyBatchAsync 沒有真的批次化，vLLM PR #43018）。F1 自己判定那是工程問題；這裡問的是**搬法和 GPU 忙的交互**：
SDMA（copy engine）不吃 CU，但每次 copy 有固定成本；用 kernel 搬（blit 或 gather kernel）沒有每次的固定成本，但會和忙碌的計算搶 CU。

## 版面（照 vLLM 的分頁 KV）
  GPU：每層一個 [2, n_gpu_blocks, BS, H, D]（K、V），BS＝16 token；CPU（pinned）：每層 K、V 各一個 [n_cpu_blocks, BS*H*D]（F1 的 rocmfix 拆法）。
  還原 R 個 block（預設 2048＝32K token）：CPU 的 block j → GPU 的 dst[j]；dst 有「連續」與「打散」兩種（分頁的 free list）。
  一個 block 每層 K、V 各一片（Llama：16×8×128×2 B＝32 KiB），R＝2048 → 32 層 × 2 × 2048＝131,072 片（和 F1 的 132,096 次同級）。

## 搬法（method）
  py_loop      Python 迴圈，每片一次 copy_（non_blocking）——最差的軟體路徑
  foreach      torch._foreach_copy_（C++ 迴圈，每片仍是一次 memcpy）——接近 vLLM 的 C++ 迴圈
  coalesce     來源與目的地都連續的那幾段合成一次 copy（dst 連續時：每層 K、V 各 1 次）
  stage        CPU 端先把 R 片收進一塊連續的 pinned 暫存（多執行緒 memcpy），一次 H2D，GPU 端 index_copy_ 散回分頁（＝DSA 的對應：CPU 端合併）
  zerocopy     Triton gather kernel 直接從 pinned host 記憶體讀（不經 copy engine，吃 CU）；載不起來就記 NOT_AVAILABLE
  （另外整個 process 用 HSA_ENABLE_SDMA=0 再跑一次：copy 改用 blit kernel。這要在 process 啟動前設，由 job 腳本分兩次跑。）
## 量什麼
  每個（method, dst 版面, GPU 狀態）：總時間（中位數、四分位距）、有效 GiB/s、每片 µs、process 的 CPU 時間、
  背景 busy 的 matmul 吞吐（Busy.stats，看搬法搶了多少計算）；每格第一次逐位元比對目的地。
  GPU 狀態：idle、busy2（m7_busy duty 0.7，和第二階段同一個背景負載）。
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

METHODS = ("py_loop", "foreach", "coalesce", "stage", "zerocopy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--bs", type=int, default=16, help="vLLM 的 block size（token）")
    ap.add_argument("--blocks", type=int, default=2048, help="要還原的 block 數（2048×16＝32K token）")
    ap.add_argument("--gpu-blocks", type=int, default=6144)
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--layouts", nargs="+", default=["contig", "scatter"])
    ap.add_argument("--gpu-states", nargs="+", default=["idle", "busy2"])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--stage-threads", type=int, default=8)
    ap.add_argument("--py-loop-max", type=int, default=2048, help="py_loop 太慢：只跑前幾個 block 再按比例報（照實標記）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from m10_speed import Busy, _watcher
    run_id = os.environ.get("RUN_ID", "NOT_SET")
    sdma = os.environ.get("HSA_ENABLE_SDMA", "default(1)")
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda:0")
    L, H, D, BS, R = a.layers, a.heads, a.head_dim, a.bs, a.blocks
    frag = BS * H * D                                     # 每片的元素數
    frag_bytes = frag * 2
    busy_lv = json.load(open(os.path.join(os.path.dirname(HERE), "results/m7_write_policy_mi300x/busy_levels.json")))

    g = torch.Generator().manual_seed(0)
    host = [[torch.randn((R, frag), generator=g, dtype=torch.float32).to(torch.bfloat16).pin_memory() for _ in range(2)]
            for _ in range(L)]
    gpu = [torch.empty((2, a.gpu_blocks, BS, H, D), dtype=torch.bfloat16, device=dev) for _ in range(L)]
    layouts = {"contig": list(range(R)),
               "scatter": torch.randperm(a.gpu_blocks, generator=g)[:R].tolist()}
    stage_buf = torch.empty((L, 2, R, frag), dtype=torch.bfloat16).pin_memory()
    stage_gpu = torch.empty((L, 2, R, frag), dtype=torch.bfloat16, device=dev)

    zc_kernel = None
    zc_err = None
    if "zerocopy" in a.methods:
        try:
            import triton
            import triton.language as tl

            @triton.jit
            def _gather(src_ptr, dst_ptr, dst_idx_ptr, FRAG: tl.constexpr, BLK: tl.constexpr):
                j = tl.program_id(0)          # 第 j 個來源 block
                c = tl.program_id(1)          # 片內第 c 段
                d = tl.load(dst_idx_ptr + j)
                offs = c * BLK + tl.arange(0, BLK)
                m = offs < FRAG
                x = tl.load(src_ptr + j.to(tl.int64) * FRAG + offs, mask=m)
                tl.store(dst_ptr + d.to(tl.int64) * FRAG + offs, x, mask=m)

            class _HostPtr:                   # 讓 Triton 把 pinned host 指標當成裝置指標（ROCm 的 UVA）；不行就丟例外
                def __init__(self, t):
                    self.t, self.dtype = t, t.dtype

                def data_ptr(self):
                    return self.t.data_ptr()
            zc_kernel = (_gather, _HostPtr)
        except Exception as ex:
            zc_err = repr(ex)

    def expected_check(dst_ids):
        bad = 0
        for li in range(L):
            for kv in range(2):
                got = gpu[li][kv].view(a.gpu_blocks, frag)[torch.tensor(dst_ids, device=dev)]
                bad += int(not torch.equal(got.cpu(), host[li][kv]))
        return bad

    def run_method(m, dst_ids, limit=None):
        n = R if limit is None else min(R, limit)
        torch.cuda.synchronize()
        c0, t0 = time.process_time(), time.perf_counter()
        if m == "py_loop":
            for li in range(L):
                for kv in range(2):
                    dv = gpu[li][kv].view(a.gpu_blocks, frag)
                    hv = host[li][kv]
                    for j in range(n):
                        dv[dst_ids[j]].copy_(hv[j], non_blocking=True)
        elif m == "foreach":
            dsts, srcs = [], []
            for li in range(L):
                for kv in range(2):
                    dv = gpu[li][kv].view(a.gpu_blocks, frag)
                    hv = host[li][kv]
                    dsts += [dv[dst_ids[j]] for j in range(n)]
                    srcs += [hv[j] for j in range(n)]
            torch._foreach_copy_(dsts, srcs, non_blocking=True)
        elif m == "coalesce":
            runs, s = [], 0                                    # 連續段：來源 j..j+k、目的地 d..d+k
            for j in range(1, n + 1):
                if j == n or dst_ids[j] != dst_ids[j - 1] + 1:
                    runs.append((s, j))
                    s = j
            for li in range(L):
                for kv in range(2):
                    dv = gpu[li][kv].view(a.gpu_blocks, frag)
                    hv = host[li][kv]
                    for s0, s1 in runs:
                        dv[dst_ids[s0]:dst_ids[s0] + (s1 - s0)].copy_(hv[s0:s1], non_blocking=True)
        elif m == "stage":
            import concurrent.futures as cf
            jobs = [(li, kv) for li in range(L) for kv in range(2)]

            def gather(x):                                     # 這個版面下來源本來就連續，仍照實做一次 CPU memcpy（DSA 的工作）
                li, kv = x
                stage_buf[li, kv, :n].copy_(host[li][kv][:n])
            with cf.ThreadPoolExecutor(a.stage_threads) as ex:
                list(ex.map(gather, jobs))
            stage_gpu[:, :, :n].copy_(stage_buf[:, :, :n], non_blocking=True)
            idx = torch.tensor(dst_ids[:n], device=dev)
            for li in range(L):
                for kv in range(2):
                    gpu[li][kv].view(a.gpu_blocks, frag).index_copy_(0, idx, stage_gpu[li, kv, :n])
        elif m == "zerocopy":
            if zc_kernel is None:
                raise RuntimeError(f"zerocopy NOT_AVAILABLE: {zc_err}")
            _gather, _HostPtr = zc_kernel
            idx = torch.tensor(dst_ids[:n], device=dev, dtype=torch.int64)
            BLK = 1024
            for li in range(L):
                for kv in range(2):
                    _gather[(n, (frag + BLK - 1) // BLK)](_HostPtr(host[li][kv]), gpu[li][kv], idx, FRAG=frag, BLK=BLK)
        else:
            raise ValueError(m)
        torch.cuda.synchronize()
        return time.perf_counter() - t0, time.process_time() - c0, n

    rows = []
    w, level = _watcher(os.path.join(a.out, f"gpu_guard_{run_id}_sdma{sdma}.json"))
    for st in a.gpu_states:
        duty = 0.0 if st == "idle" else float(busy_lv[st])
        with Busy(duty, watch=w) as b:
            for lay in a.layouts:
                dst_ids = layouts[lay]
                for m in a.methods:
                    lim = a.py_loop_max if m == "py_loop" else None
                    ts_ = []
                    status, bad = "ok", None
                    for rep in range(a.reps + 1):              # 第 0 次是暖機（Triton 編譯、配置），不計
                        for li in range(L):
                            gpu[li].zero_()
                        try:
                            t, c, n = run_method(m, dst_ids, lim)
                        except Exception as ex:
                            status = f"NOT_AVAILABLE: {ex!r}"[:300]
                            break
                        if rep == 0:
                            if lim is None:
                                bad = expected_check(dst_ids)
                            continue
                        ts_.append((t, c, n))
                    if not ts_:
                        rows.append(dict(run_id=run_id, ts=dt.datetime.now(dt.timezone.utc).isoformat(), sdma=sdma, gpu_state=st,
                                         layout=lay, method=m, status=status))
                        continue
                    tt = [x[0] for x in ts_]
                    n = ts_[0][2]
                    nbytes = L * 2 * n * frag_bytes
                    nfrag = L * 2 * n
                    med = statistics.median(tt)
                    rows.append(dict(
                        run_id=run_id, ts=dt.datetime.now(dt.timezone.utc).isoformat(), sdma=sdma, gpu_state=st, layout=lay,
                        method=m, status=status, n_blocks=n, scaled_from=(n if n < R else ""), bytes=nbytes, n_frag=nfrag,
                        t_med_s=round(med, 6), t_q1=round(statistics.quantiles(tt, n=4)[0], 6) if len(tt) >= 2 else "",
                        t_q3=round(statistics.quantiles(tt, n=4)[2], 6) if len(tt) >= 2 else "",
                        GiBps=round(nbytes / med / 2 ** 30, 3), us_per_frag=round(med / nfrag * 1e6, 3),
                        cpu_s_med=round(statistics.median(x[1] for x in ts_), 4), kv_bad=bad, reps=len(tt)))
                    print(rows[-1], flush=True)
        if b.stats:
            for r in rows:
                if r["gpu_state"] == st and "busy_stats" not in r:
                    r["busy_stats"] = json.dumps(b.stats)
    contaminated = bool(w and w.contaminated)
    if w:
        w.__exit__(None, None, None)
    for r in rows:
        r["contaminated"] = contaminated
        r["host_level"] = level
    keys = sorted({k for r in rows for k in r})
    path = os.path.join(a.out, f"copybench_sdma{sdma}.csv")
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    json.dump(dict(run_id=run_id, sdma=sdma, frag_bytes=frag_bytes, L=L, H=H, D=D, BS=BS, R=R, contaminated=contaminated,
                   zerocopy_error=zc_err, torch=torch.__version__), open(path.replace(".csv", ".json"), "w"), indent=1)
    if contaminated:
        open(os.path.join(os.environ.get("RUN_DIR", a.out), "CONTAMINATED"), "w").write("gpu_guard\n")
    print("寫好", path)


if __name__ == "__main__":
    main()
