"""timing.py — 量引擎每個請求要多久（給「完整地圖要跑多久」的估計用；06_simulator.md §9）。不是地圖、不看任何結果數字。

每份資料取一段，跑 1 格：session 引擎（E2＋C1，8 個時間設定＝GPU 閒／忙 × SSD 共用 1／2／4／8）與 block 引擎（E6＋C1，1 個時間設定），
記 秒／請求。輸出 $RUN_DIR/timing.csv。
"""
from __future__ import annotations

import csv
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sim  # noqa: E402
import trace as TR  # noqa: E402

RUN_DIR = os.environ.get("RUN_DIR", "/tmp/p3sim_timing_adhoc")
DATA = [("bailian", "coder", 0, None, 1.0), ("bailian", "traceA", 0, None, 1.0), ("mooncake", "conv-rule-strict", 0, None, 1.0),
        ("freeinference", "week13", 0, None, 1.0), ("wildchat", "1m", 200 * 86400, 7 * 86400, 1.0)]
SPECS8 = [("mi300x", g, "idle", sh, "cake", "mi300x_local") for g in ("idle", "busy") for sh in (1, 2, 4, 8)]


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    out = []
    only = sys.argv[1:] or None
    for src, var, a, d, frac in DATA:
        if only and f"{src}/{var}" not in only:
            continue
        t0 = time.time()
        P = TR.prepare(src, var, a, d, session_frac=frac)
        tp = time.time() - t0
        n = P["meta"]["n_req"]
        nb = sum(len(x) for x in P["full"])
        for cap in (32, 512):
            t1 = time.time()
            rows, q, ids, eng, vs = sim.run_cell(P, "E2", "C1", cap, "llama", SPECS8, H=1800)
            ts = time.time() - t1
            t2 = time.time()
            r2, _, _, _, _ = sim.run_cell_fifo1(P, "E6", "C1", cap, "llama", SPECS8[0], H=1800)
            tb = time.time() - t2
            row = dict(source=src, variant=var, window=f"{a}+{d}", n_req=n, mean_blocks_per_req=nb / max(n, 1), cap_GiB=cap,
                       prepare_s=tp, session_engine_8views_s=ts, session_s_per_req=ts / max(n, 1),
                       block_engine_1view_s=tb, block_s_per_req=tb / max(n, 1), fifo1_gpu_busy_frac=r2[0]["gpu_busy_frac"],
                       identities_ok=all(i["ok"] for i in ids))
            out.append(row)
            print(row, flush=True)
    with open(os.path.join(RUN_DIR, "timing.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run_id"] + list(out[0]))
        w.writeheader()
        for r in out:
            w.writerow(dict(run_id=os.environ.get("RUN_ID", "adhoc"), **r))


if __name__ == "__main__":
    main()
