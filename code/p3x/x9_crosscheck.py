"""x9_crosscheck.py — 檢查 x9_census.cake() 與 m7_sim.sim_restore()（Cake 模式、start=0、單一限速層、無爭用）一致。

只做 start=0 的情況（sim_restore 一定從 block 0 開始重算）。f(i) 兩邊用同一份（x9_census.load_f，秒）。
用法： python code/p3x/x9_crosscheck.py --out results/p3_explore/x9
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import x9_census as X  # noqa: E402
from m7_restore_harness import GiB, Tier  # noqa: E402
import m7_sim  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run_id = os.environ["RUN_ID"]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    F = X.load_f()
    rows = []
    worst = 0.0
    for tier, gb in X.TIERS_GIBPS.items():
        for n in (1, 2, 5, 13, 32, 64, 80, 120):
            fs = [X.f_at(i, F) / 1e3 for i in range(n)]
            t = Tier(tier, read_Bps=gb * GiB, write_Bps=gb * GiB, c_s=0.0, mode="fifo", k_read=1.0)
            ref, meet, _, _ = m7_sim.sim_restore(n, [t] * n, fs, 0.0, "cake")
            mine = X.cake(0, n, X.CHUNK_BYTES / (gb * GiB) * 1e3, F)
            rel = abs(mine - ref * 1e3) / (ref * 1e3)
            worst = max(worst, rel)
            rows.append(dict(run_id=run_id, ts=now, tier=tier, n_chunks=n, m7_sim_ms=ref * 1e3, x9_cake_ms=mine,
                             rel_diff=rel, m7_meet=meet))
    with open(os.path.join(a.out, "t2_cake_crosscheck.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("worst rel diff", worst)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
