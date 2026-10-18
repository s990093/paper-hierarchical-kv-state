"""timing_ab.py — 同一台機器、同一段時間，舊程式（c0c179b 的 worktree）和新程式跑同一格的秒數（06 §16）。不看任何結果數字，只記秒數。

  python code/p3_sim/timing_ab.py --code-dir <repo 或 worktree> --pkl <P 的快取 .pkl> [--cap 32] [--tag old|new]

P 從快取的 pickle 讀（新舊兩邊用同一個 P；不計時）。格式表兩邊都用 placeholder（舊程式只認得它）。
一格＝session 引擎 E2＋C1、8 個時間設定（GPU 閒／忙 × SSD 共用 1／2／4／8），和 block 引擎 E6＋C1、1 個時間設定（06 §9 的同一個量法）。
輸出一列 JSON 到 stdout，並附加到 $RUN_DIR/timing_ab.csv。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-dir", required=True)
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--cap", type=float, default=32.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    here = os.path.join(os.path.abspath(a.code_dir), "code", "p3_sim")
    sys.path.insert(0, here)
    import sim  # noqa: E402  （a.code_dir 那一份）
    P = pickle.load(open(a.pkl, "rb"))
    fmt = os.path.join(here, "config", "format_table.PLACEHOLDER.json")
    specs8 = [("mi300x", g, "idle", sh, "cake", "mi300x_local") for g in ("idle", "busy") for sh in (1, 2, 4, 8)]
    t1 = time.time()
    rows, q, ids, eng, vs = sim.run_cell(P, "E2", "C1", a.cap, "llama", specs8, H=1800, fmt_path=fmt)
    ts = time.time() - t1
    ok = all(i["ok"] for i in ids)
    del rows, q, ids, eng, vs
    t2 = time.time()
    r2, _, ids2, _, _ = sim.run_cell_fifo1(P, "E6", "C1", a.cap, "llama", specs8[0], H=1800, fmt_path=fmt)
    tb = time.time() - t2
    n = len(P["full"])
    row = dict(run_id=os.environ.get("RUN_ID", "adhoc"), tag=a.tag, code_dir=a.code_dir, sim_file=sim.__file__,
               source=P["meta"]["source"], variant=P["meta"]["variant"], n_req=n, cap_GiB=a.cap,
               session_engine_8views_s=ts, session_s_per_req=ts / n, block_engine_1view_s=tb, block_s_per_req=tb / n,
               identities_ok=ok and all(i["ok"] for i in ids2), loadavg_1m=os.getloadavg()[0])
    print(json.dumps(row), flush=True)
    p = os.path.join(os.environ.get("RUN_DIR", "."), "timing_ab.csv")
    new = not os.path.exists(p)
    with open(p, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    main()
