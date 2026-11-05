"""next_b_g0.py — G0 / G1 對拍：兩份 x1 sweep 輸出逐值比對（NEXT_PLAN_20261104 §3b）。

  python -I code/icpe/next_b_g0.py <參考 x1_sweep.csv[.gz]> <新跑的 x1_sweep.csv[.gz]> [--out <diff.csv>]

比法（和 icpe-sim 的 G1 相同）：去掉 run_id、ts、elapsed_s、code_sha 四欄，依
(trace, rule, cpu_GiB, policy, slow, gpu, restore) 排序後，每一格當字串比。欄位集合、列數、鍵都要相同。
逐值相同 → exit 0；任何不同 → 印前 20 個差異、exit 1。只用標準庫。
跨機器對拍（2026-11-05 加）：也去掉 collect 加的 host 欄；檔案裡有新功能的欄（GPU 層、預取目的地、慢層上限、雜訊）時，
那些欄也加進鍵（沒有這些欄的舊 sweep 鍵不變，G0／G1 的比法不受影響）；可讀 .xz。
"""
from __future__ import annotations

import argparse
import csv
import gzip
import lzma
import sys

DROP = ("run_id", "ts", "elapsed_s", "code_sha", "host")
KEY0 = ("trace", "rule", "cpu_GiB", "policy", "slow", "gpu", "restore")
KEY_NEW = ("gpu_GiB", "gpu_write", "gpu_policy", "pf_target", "slow_cap_frac", "slow_policy",
           "noise_model", "noise_lambda", "noise_draw", "noise_seed")
KEY = KEY0


def load(path):
    op = gzip.open if path.endswith(".gz") else lzma.open if path.endswith(".xz") else open
    with op(path, "rt", newline="") as fh:
        rd = csv.DictReader(fh)
        cols = [c for c in rd.fieldnames if c not in DROP]
        rows = [{c: r[c] for c in cols} for r in rd]
    return cols, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("new")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    cr, rr = load(a.ref)
    cn, rn = load(a.new)
    ok = True
    if cr != cn:
        print(f"欄位不同：ref only={sorted(set(cr) - set(cn))} new only={sorted(set(cn) - set(cr))} order_same={cr == cn}")
        ok = False
    cols = [c for c in cr if c in cn]
    global KEY
    KEY = KEY0 + tuple(k for k in KEY_NEW if k in cols)
    key = lambda r: tuple(r[k] for k in KEY)
    dr, dn = {}, {}
    for r in rr:
        assert key(r) not in dr, f"ref 鍵重複 {key(r)}"
        dr[key(r)] = r
    for r in rn:
        assert key(r) not in dn, f"new 鍵重複 {key(r)}"
        dn[key(r)] = r
    print(f"rows ref={len(rr)} new={len(rn)} cols_compared={len(cols)}")
    miss, extra = sorted(set(dr) - set(dn)), sorted(set(dn) - set(dr))
    if miss or extra:
        print(f"鍵不同：只在 ref {len(miss)}（例 {miss[:3]}）；只在 new {len(extra)}（例 {extra[:3]}）")
        ok = False
    diffs = []
    for k in sorted(set(dr) & set(dn)):
        for c in cols:
            if dr[k][c] != dn[k][c]:
                diffs.append((*k, c, dr[k][c], dn[k][c]))
    n_cells = len(set(dr) & set(dn)) * len(cols)
    print(f"cells compared={n_cells} differing={len(diffs)}")
    for d in diffs[:20]:
        print("  DIFF", d)
    if a.out and diffs:
        with open(a.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([*KEY, "col", "ref", "new"])
            w.writerows(diffs)
    if diffs:
        ok = False
    print("G0_RESULT", "IDENTICAL" if ok else "DIFFERENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
