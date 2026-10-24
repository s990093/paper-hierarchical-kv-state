"""timing.py — 量引擎每個請求要多久（給「完整地圖要跑多久」的估計用；06_simulator.md §9）。不是地圖、不看任何結果數字。

每份資料取一段，跑 1 格：session 引擎（E2＋C1，8 個時間設定＝GPU 閒／忙 × SSD 共用 1／2／4／8）與 block 引擎（E6＋C1，1 個時間設定），
記 秒／請求。輸出 $RUN_DIR/timing.csv。
2026-10-18（06 §16）：準備改用 prepare_cached（記第一次＝沒命中＋寫快取、第二次＝命中的秒數）；另量 O（降級時也用 oracle 決定）的一格；
--pols 可加別的策略。量法和 06 §9 相同（同一台共用機，有別人的負載：照錄 loadavg）。
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="*")
    ap.add_argument("--pols", nargs="*", default=["C1", "O"])
    ap.add_argument("--fmt", default="official")
    ap.add_argument("--caps", nargs="*", type=float, default=[32, 512])
    a_ = ap.parse_args()
    os.makedirs(RUN_DIR, exist_ok=True)
    out = []
    only = a_.only or None
    for src, var, a, d, frac in DATA:
        if only and f"{src}/{var}" not in only:
            continue
        t0 = time.time()
        P = TR.prepare_cached(src, var, a, d, session_frac=frac)
        tp = time.time() - t0
        hit0 = P["meta"]["cache"]["hit"]
        del P
        t0 = time.time()
        P = TR.prepare_cached(src, var, a, d, session_frac=frac)
        tp2 = time.time() - t0
        n = P["meta"]["n_req"]
        nb = sum(len(x) for x in P["full"])
        for cap in a_.caps:
            for pol in a_.pols:
                t1 = time.time()
                rows, q, ids, eng, vs = sim.run_cell(P, "E2", pol, cap, "llama", SPECS8, H=1800, fmt_path=a_.fmt)
                ts = time.time() - t1
                ok = all(i["ok"] for i in ids)
                del rows, q, ids, eng, vs
                t2 = time.time()
                r2, _, ids2, _, _ = sim.run_cell_fifo1(P, "E6", pol, cap, "llama", SPECS8[0], H=1800, fmt_path=a_.fmt)
                tb = time.time() - t2
                row = dict(source=src, variant=var, window=f"{a}+{d}", n_req=n, mean_blocks_per_req=nb / max(n, 1), cap_GiB=cap,
                           policy=pol, prepare_first_s=tp, prepare_first_was_cache_hit=hit0, prepare_cached_s=tp2,
                           session_engine_8views_s=ts, session_s_per_req=ts / max(n, 1),
                           block_engine_1view_s=tb, block_s_per_req=tb / max(n, 1), fifo1_gpu_busy_frac=r2[0]["gpu_busy_frac"],
                           identities_ok=ok and all(i["ok"] for i in ids2), loadavg_1m=os.getloadavg()[0])
                out.append(row)
                print(row, flush=True)
    p = os.path.join(RUN_DIR, "timing.csv")
    new = not os.path.exists(p)          # 同一個 run 目錄裡跑好幾次：附加（2026-10-18）
    with open(p, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run_id"] + list(out[0]))
        if new:
            w.writeheader()
        for r in out:
            w.writerow(dict(run_id=os.environ.get("RUN_ID", "adhoc"), **r))


if __name__ == "__main__":
    main()
