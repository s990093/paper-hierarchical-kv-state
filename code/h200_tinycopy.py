"""h200_tinycopy.py — H200 上「Qwen fp8_head 在 GPU 忙時，只搬不拆也慢」的診斷微測（13_h200_repro §14 A；只報告）。

現象（13 號 Q5 (a)、Q1）：GPU 忙（背景 duty 1.0）時 Qwen fp8_head 只搬不拆 32K 0.118 s（閒 0.019）、128K 0.464 s（閒 0.071）；
Llama fp8_token、兩個模型的 int4_kivi 不變；MI300X（duty 0.7）也不變。blocks.csv 裡每個 block 送出拷貝都是 0.03 ms，時間全在最後等拷貝完成。
假設：fp8_head 每個 block 有一塊極小的拷貝（scale，Qwen 896 B），極小的 H2D 在背景佔滿 GPU 時要等時間片；
每塊多等約 1.5 ms × block 數（32K 64 個、128K 252 個）＝實測多出來的時間。

這支只量拷貝本身（pinned → GPU，non_blocking，一個 stream，64 次後 synchronize），閒／忙交替：
  single：每次一塊 size bytes（size 從 896 B 到 14 MB）
  block ：模仿一個 block 的兩塊（Qwen fp8_head：q 14.68 MB＋scale 896 B；Llama fp8_token：q 33.55 MB＋scale 1 MB）

  python code/h200_tinycopy.py --out-dir results/m10_diag_h200 [--reps 10]
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m10_speed import Busy, _watcher, run_id, ts  # noqa: E402

N_BLOCKS = 64                                           # 32K ÷ 512
SIZES = [896, 4096, 16384, 65536, 262144, 1048576, 14680064]
BLOCKS = {"qwen_fp8_head": [14680064, 896], "llama_fp8_token": [33554432, 1048576],
          "qwen_int4_kivi": None}                       # int4 的各塊由 m10_codec 實際編一次取大小（見 main）


def _bufs(sizes, dev):
    src = [torch.empty(s * N_BLOCKS, dtype=torch.uint8).pin_memory() for s in sizes]
    dst = [torch.empty(s * N_BLOCKS, dtype=torch.uint8, device=dev) for s in sizes]
    return src, dst


def _run(sizes, src, dst, stream):
    t0 = time.perf_counter()
    with torch.cuda.stream(stream):
        for b in range(N_BLOCKS):
            for s, a, d in zip(sizes, src, dst):
                d[b * s:(b + 1) * s].copy_(a[b * s:(b + 1) * s], non_blocking=True)
    stream.synchronize()
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--busy-levels", default=os.environ.get("M10_BUSY_LEVELS"))
    ap.add_argument("--busy-key", default="busy2")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rid = os.environ.get("RUN_ID") or run_id()
    duty = json.load(open(a.busy_levels))[a.busy_key]
    dev = "cuda"
    import m10_codec as C
    x = torch.randn((28, 2, 4, 512, 128)).to(torch.bfloat16)          # Qwen 一個 block：[L, K/V, H, T, D]
    e = C.encode(x, "int4_kivi")
    BLOCKS["qwen_int4_kivi"] = [t.numel() * t.element_size() for t in e.parts.values()]
    e8 = C.encode(x, "fp8_head")                         # 驗算：Qwen fp8_head 一個 block 的兩塊大小
    got = sorted(t.numel() * t.element_size() for t in e8.parts.values())
    assert got == sorted(BLOCKS["qwen_fp8_head"]), got
    pats = [("single", [s]) for s in SIZES] + [("block:" + k, v) for k, v in BLOCKS.items()]
    stream = torch.cuda.Stream()
    rows = []
    watch, level = _watcher(os.path.join(a.out_dir, f"gpu_guard_{rid}.json"))

    def contaminated():
        return "NOT_CHECKED" if watch is None else bool(watch.intruders) or watch.started_clean is False
    busy_stats = []
    for state in ("idle", "busy", "idle", "busy"):          # 交替兩輪，看漂移
        with Busy(duty if state == "busy" else 0.0, watch) as bz:
            for name, sizes in pats:
                src, dst = _bufs(sizes, dev)
                for _ in range(a.warmup):
                    _run(sizes, src, dst, stream)
                for r in range(a.reps):
                    w = _run(sizes, src, dst, stream)
                    rows.append(dict(run_id=rid, ts=ts(), gpu_state=state, pattern=name,
                                     sizes="+".join(map(str, sizes)), n_copies=N_BLOCKS * len(sizes), rep=r,
                                     wall_s=round(w, 6), contaminated_gpu=contaminated(), host_level=level))
                del src, dst
        if bz.stats:
            busy_stats.append(bz.stats)
    cont = contaminated()
    if watch is not None:
        watch.__exit__(None, None, None)
    with open(os.path.join(a.out_dir, "tinycopy.csv"), "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        if f.tell() == 0:
            wr.writeheader()
        wr.writerows(rows)
    summ = {}
    for name, sizes in pats:
        for st in ("idle", "busy"):
            v = [r["wall_s"] for r in rows if r["pattern"] == name and r["gpu_state"] == st]
            summ[f"{name}|{st}"] = dict(sizes=sizes, median_s=statistics.median(v), n=len(v),
                                        per_block_ms=1e3 * statistics.median(v) / N_BLOCKS)
    json.dump(dict(run_id=rid, duty=duty, busy=busy_stats, contaminated=cont, host_level=level, summary=summ),
              open(os.path.join(a.out_dir, f"tinycopy_{rid}.json"), "w"), indent=1)
    for k, v in summ.items():
        print(f"{k:28s} median {v['median_s']*1e3:9.3f} ms  ({v['per_block_ms']:.3f} ms／block)", flush=True)
    print("CLEAN" if cont is False else f"CONTAMINATED／{cont}", flush=True)


if __name__ == "__main__":
    main()
