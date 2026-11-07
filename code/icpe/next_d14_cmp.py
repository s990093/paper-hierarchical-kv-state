"""next_d14_cmp.py — icpe-data：PREREG 偏離 14 第二段的兩個核對（STATUS_DATA 2026-11-07 16:34，使用者在 icpe-main 的決定）。只讀 CSV。

  table X Y --skip-components O-PF "O-ALL(report)" --out <csv>
      兩張上限表（*_headroom_components.csv[.xz]）逐格比 7 欄（同 explore_apc：T_oracle、T_rival、headroom、best_rival、
      best_rival_gpu_policy、oracle_policy、oracle_gpu_policy；鍵 trace、rule、cpu_GiB（浮點）、slow、gpu、restore、component）。
      X 只取 Y 有的 trace（家族表 vs 家族表時兩邊相同）。--skip-components 以外的元件：兩邊格子一一對上、7 欄全部相同，否則 exit 1。
      被略過的元件只報告不同數。最後一行 D14_TABLE_RESULT IDENTICAL|DIFFERENT。
  sweep NEW OLD --skip-policies O-PF O-ALL --out <csv>
      兩份 x1_sweep：policy 不在 --skip-policies 的列，鍵一一對上、值逐字相同（排除 run_id、ts、host、code_sha、elapsed_s
      與偏離 14 的旗標欄）。最後一行 D14_SWEEP_RESULT IDENTICAL|DIFFERENT。
"""
from __future__ import annotations

import argparse
import csv
import lzma
import os
import socket
import sys
from datetime import datetime

TK = ["trace", "rule", "cpu_GiB", "slow", "gpu", "restore", "component"]
TCOLS = ["T_oracle", "T_rival", "headroom", "best_rival", "best_rival_gpu_policy", "oracle_policy", "oracle_gpu_policy"]
SK = ["trace", "rule", "cpu_GiB", "policy", "slow", "gpu", "restore", "gpu_GiB", "gpu_write", "gpu_policy", "pf_target",
      "noise_model", "noise_lambda", "noise_draw", "noise_seed"]
SKIP_COLS = {"run_id", "ts", "host", "code_sha", "elapsed_s", "pf_expire", "noise_overdue", "pf_gpu_recheck"}


def rows(path):
    op = lzma.open if path.endswith(".xz") else open
    with op(path, "rt", newline="") as fh:
        rd = csv.DictReader(fh)
        return rd.fieldnames, list(rd)


def meta():
    return dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"),
                host=socket.gethostname())


def tkey(r):
    return tuple(str(float(r[c])) if c == "cpu_GiB" else r[c] for c in TK)


def cmd_table(a):
    _, X = rows(a.x)
    _, Y = rows(a.y)
    traces = {r["trace"] for r in Y}
    X = [r for r in X if r["trace"] in traces]
    Xm, Ym = {}, {}
    for M, R, nm in ((Xm, X, "X"), (Ym, Y, "Y")):
        for r in R:
            k = tkey(r)
            if k in M:
                raise SystemExit(f"{nm}：重複的格 {k}")
            M[k] = r
    comps = sorted({k[-1] for k in Xm} | {k[-1] for k in Ym})
    out, bad = [], False
    for c in comps:
        kx = {k for k in Xm if k[-1] == c}
        ky = {k for k in Ym if k[-1] == c}
        both = kx & ky
        nd = {col: sum(Xm[k][col] != Ym[k][col] for k in both) for col in TCOLS}
        skipped = c in a.skip_components
        ok = skipped or (kx == ky and all(v == 0 for v in nd.values()))
        bad |= not ok
        out.append(dict(meta(), x=a.x, y=a.y, component=c, skipped=skipped, n_x=len(kx), n_y=len(ky), n_matched=len(both),
                        **{f"ndiff_{col}": v for col, v in nd.items()}, result="SKIPPED" if skipped else ("IDENTICAL" if ok else "DIFFERENT")))
        print(f"{c:16s} skipped={skipped!s:5s} x={len(kx)} y={len(ky)} matched={len(both)} ndiff={nd}")
    write(a.out, out)
    print("D14_TABLE_RESULT", "DIFFERENT" if bad else "IDENTICAL")
    return 1 if bad else 0


def cmd_sweep(a):
    hn, N = rows(a.new)
    ho, O = rows(a.old)
    kc = [c for c in SK if c in ho]
    if any(c not in hn for c in kc):
        raise SystemExit(f"NEW 缺鍵欄 {[c for c in kc if c not in hn]}")
    vc = [c for c in ho if c not in SKIP_COLS and c not in kc]
    miss = [c for c in vc if c not in hn]
    if miss:
        raise SystemExit(f"NEW 少了 OLD 有的欄 {miss}")
    skip = set(a.skip_policies)
    Nm, Om = {}, {}
    for M, R, nm in ((Nm, N, "NEW"), (Om, O, "OLD")):
        for r in R:
            k = tuple(r[c] for c in kc)
            if k in M:
                raise SystemExit(f"{nm}：重複的鍵 {k}")
            M[k] = r
    kn = {k for k, r in Nm.items() if r["policy"] not in skip}
    ko = {k for k, r in Om.items() if r["policy"] not in skip}
    both = kn & ko
    ndiff = {c: 0 for c in vc}
    nrow = 0
    for k in both:
        d = [c for c in vc if Nm[k][c] != Om[k][c]]
        for c in d:
            ndiff[c] += 1
        nrow += bool(d)
    sk_n = sum(r["policy"] in skip for r in Nm.values())
    sk_o = sum(r["policy"] in skip for r in Om.values())
    ok = kn == ko and nrow == 0
    rec = dict(meta(), new=a.new, old=a.old, n_new=len(kn), n_old=len(ko), n_matched=len(both), only_new=len(kn - ko),
               only_old=len(ko - kn), rows_differing=nrow, skipped_policies=";".join(sorted(skip)), skipped_rows_new=sk_n,
               skipped_rows_old=sk_o, result="IDENTICAL" if ok else "DIFFERENT")
    write(a.out, [rec])
    print(f"non-skipped rows new={len(kn)} old={len(ko)} matched={len(both)} only_new={len(kn - ko)} only_old={len(ko - kn)} "
          f"rows_differing={nrow} cols_differing={ {c: v for c, v in ndiff.items() if v} }; skipped rows new={sk_n} old={sk_o}")
    print("D14_SWEEP_RESULT", "IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


def write(path, recs):
    if not path:
        return
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        if new:
            w.writeheader()
        w.writerows(recs)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("table")
    s.add_argument("x"); s.add_argument("y")
    s.add_argument("--skip-components", nargs="*", default=[])
    s.add_argument("--out")
    s = sp.add_parser("sweep")
    s.add_argument("new"); s.add_argument("old")
    s.add_argument("--skip-policies", nargs="*", default=[])
    s.add_argument("--out")
    a = ap.parse_args()
    return cmd_table(a) if a.cmd == "table" else cmd_sweep(a)


if __name__ == "__main__":
    sys.exit(main())
