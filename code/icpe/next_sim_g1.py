"""next_sim_g1.py — ICPE next G1：新程式（新功能全部關閉）的 X1 sweep 和舊輸出逐值比對（NEXT_PLAN §3b）。

  python -I code/icpe/next_sim_g1.py --ref results/icpe_h200/t1/p01/sim_V1/x1/x1_sweep.csv.gz --new <run>/out/x1/x1_sweep.csv \
      --out results/icpe_next/sim/g1_compare.csv

比法：兩邊都去掉 run_id、ts、elapsed_s（計時）、code_sha（程式雜湊，改了程式一定不同）四欄；欄位清單與順序必須相同；
依 (trace, rule, cpu_GiB, policy, slow, gpu, restore) 排序（sweep 的輸出順序本來就不固定：imap_unordered）；
每一格當字串比（sweep 寫檔時已經 round 到 6 位，字串相同＝值相同）。任一格不同 → exit 1，並印出前 20 個不同的格子。
只用標準庫。
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import socket
import sys
from datetime import datetime

DROP = ("run_id", "ts", "elapsed_s", "code_sha")
KEY = ("trace", "rule", "cpu_GiB", "policy", "slow", "gpu", "restore")


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", newline="") as fh:
        r = csv.reader(fh)
        head = next(r)
        rows = list(r)
    keep = [i for i, h in enumerate(head) if h not in DROP]
    cols = [head[i] for i in keep]
    ki = [cols.index(k) for k in KEY]
    out = [[row[i] for i in keep] for row in rows]
    out.sort(key=lambda x: tuple(x[i] for i in ki))
    runs = sorted({row[head.index("run_id")] for row in rows}) if "run_id" in head else []
    shas = sorted({row[head.index("code_sha")] for row in rows}) if "code_sha" in head else []
    return cols, out, ki, runs, shas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c0, r0, k0, run0, sha0 = load(a.ref)
    c1, r1, k1, run1, sha1 = load(a.new)
    cols_ok = c0 == c1
    n_ok = len(r0) == len(r1)
    diffs = []
    dup = 0
    if cols_ok and n_ok:
        keys0 = [tuple(x[i] for i in k0) for x in r0]
        dup = len(keys0) - len(set(keys0))
        for x, y in zip(r0, r1):
            if x != y:
                for j, (u, v) in enumerate(zip(x, y)):
                    if u != v:
                        diffs.append((tuple(x[i] for i in k0), c0[j], u, v))
    passed = cols_ok and n_ok and not diffs and dup == 0
    print(f"ref {a.ref}: {len(r0)} 列 {len(c0)} 欄 run_id={run0} code_sha={sha0}")
    print(f"new {a.new}: {len(r1)} 列 {len(c1)} 欄 run_id={run1} code_sha={sha1}")
    print(f"欄位相同 {cols_ok}；列數相同 {n_ok}；鍵值重複 {dup}；不同的格子 {len(diffs)}")
    if not cols_ok:
        print("  ref 有、new 沒有：", [c for c in c0 if c not in c1], "  new 有、ref 沒有：", [c for c in c1 if c not in c0])
    for d in diffs[:20]:
        print("  DIFF", d)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["run_id", "ts", "host", "ref", "ref_runs", "ref_code_sha", "new", "new_runs", "new_code_sha",
                        "n_rows_ref", "n_rows_new", "n_cols_compared", "cols_equal", "dup_keys", "n_diff_cells", "passed"])
        w.writerow([os.environ.get("RUN_ID", "adhoc"), datetime.now().astimezone().isoformat(timespec="seconds"),
                    socket.gethostname(), a.ref, ";".join(run0), ";".join(sha0), a.new, ";".join(run1), ";".join(sha1),
                    len(r0), len(r1), len(c0), cols_ok, dup, len(diffs), passed])
    print("G1", "PASS" if passed else "FAIL")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
