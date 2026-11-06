"""main_ab.py — icpe-main：版本之間的並列與逐格關係核對（版本 A 對版本 B；使用者 2026-11-06）。

  python -I code/icpe/main_ab.py cells --name <名稱> --x <headroom_components.csv> --y <headroom_components.csv> \
      --rel le|eq [--component O-GPU ...] --out <cellcheck.csv（附加）>
      逐格（(格, 元件)）核對 x 的 headroom 和 y 的關係：le＝x ≤ y（容許 1e-12，超過就算違反；剛好在容許內的另計）；eq＝完全相同。
      兩邊的 (格, 元件) 集合必須相同。違反就照實寫進輸出、exit 1。
      用途：版本 B（7 個 TTL）的 O-GPU ≤ 版本 A（3 個 TTL）——O-GPU 的 ④ 只用 BEL（兩版相同）、② 在 B 多了 8 個 TTL 規則（只少不多），
      所以每格只能變小或不變；偏離 5 算法的每個元件也一樣（④ 不變、② 多了規則）。正式 B 與偏離 5 的 O-GPU 必須完全相同。

  python -I code/icpe/main_ab.py table --bytrace <label>=<by_trace.csv> ... --verdict <label>=<verdict.csv> ... \
      --out-dir <dir> --out-prefix <p>
      by_trace 並列（每個 (trace, rule, 元件) 一列、每個版本的格數／最大／≥15% 格數），各版本的格數必須相同；
      verdict 並列（class 與 variant 兩層）。
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
from datetime import datetime

import pandas as pd

CELL = ["trace", "rule", "cpu_GiB", "slow", "gpu", "restore", "gpu_GiB", "gpu_write", "slow_cap_frac", "component"]
TOL = 1e-12


def meta():
    return dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"),
                host=socket.gethostname())


def cells(a):
    X, Y = pd.read_csv(a.x), pd.read_csv(a.y)
    if a.component:
        X, Y = X[X.component.isin(a.component)], Y[Y.component.isin(a.component)]
    M = X[CELL + ["headroom"]].merge(Y[CELL + ["headroom"]], on=CELL, how="outer", suffixes=("_x", "_y"), indicator=True)
    only = int((M._merge != "both").sum())
    M = M[M._merge == "both"]
    d = M.headroom_x - M.headroom_y
    if a.rel == "le":
        viol, within = int((d > TOL).sum()), int(((d > 0) & (d <= TOL)).sum())
        smaller, same = int((d < 0).sum()), int((d == 0).sum())
    else:
        viol, within, smaller, same = int((d != 0).sum()), 0, 0, int((d == 0).sum())
    ok = only == 0 and viol == 0 and len(M) > 0
    r = dict(meta(), name=a.name, rel=a.rel, components=";".join(a.component) or "all", file_x=a.x, file_y=a.y,
             n_x=len(X), n_y=len(Y), n_unmatched=only, n_matched=len(M), n_violate=viol, n_within_tol=within,
             n_x_smaller=smaller, n_equal=same, max_x_minus_y=float(d.max()) if len(d) else float("nan"),
             min_x_minus_y=float(d.min()) if len(d) else float("nan"), result="PASS" if ok else "FAIL")
    pd.DataFrame([r]).to_csv(a.out, mode="a", header=not os.path.exists(a.out), index=False)
    print(f"CELLS {a.name} ({a.rel}, {r['components']}): {r['result']} matched {len(M)} unmatched {only} violate {viol} "
          f"within_tol {within} x<y {smaller} x==y {same} max(x-y) {r['max_x_minus_y']}")
    sys.exit(0 if ok else 1)


def table(a):
    m = meta()
    parts = []
    for spec in a.bytrace:
        lab, p = spec.split("=", 1)
        d = pd.read_csv(p)
        d = d[["trace_class", "trace", "rule", "arrival", "tool_type", "component", "verdict_use", "n_cells", "max", "n_ge15",
               "n_ge15_write_saturated"]].copy()
        d["version"] = lab
        parts.append(d)
    T = pd.concat(parts, ignore_index=True)
    key = ["trace_class", "trace", "rule", "arrival", "tool_type", "component"]
    W = T.pivot_table(index=key, columns="version", values=["n_cells", "max", "n_ge15", "n_ge15_write_saturated"], aggfunc="first")
    W.columns = [f"{v}__{c}" for c, v in W.columns]
    W = W.reset_index()
    labs = [s.split("=", 1)[0] for s in a.bytrace]
    nc = W[[f"{l}__n_cells" for l in labs]]
    bad = W[nc.isna().any(axis=1) | (nc.nunique(axis=1) != 1)]
    W = W[key + [f"{l}__{c}" for l in labs for c in ("n_cells", "max", "n_ge15", "n_ge15_write_saturated")]]
    V = []
    for spec in a.verdict:
        lab, p = spec.split("=", 1)
        v = pd.read_csv(p)
        v["version"] = lab
        V.append(v[["version", "level", "trace_class", "trace", "rule", "arrival", "tool_type", "max_judged", "argmax_component",
                    "n_ge15_cells_judged", "ge15", "n_variants", "n_variants_ge15", "variants_ge15", "falsification"]])
    V = pd.concat(V, ignore_index=True) if V else pd.DataFrame()
    os.makedirs(a.out_dir, exist_ok=True)
    for name, X in (("by_trace", W), ("verdict", V)):
        X = X.copy()
        for i, (k, val) in enumerate(m.items()):
            X.insert(i, k, val)
        X.to_csv(os.path.join(a.out_dir, f"{a.out_prefix}_{name}.csv"), index=False)
    print(f"TABLE {a.out_prefix}: {len(W)} (trace, rule, component) rows, versions {labs}; n_cells equal across versions: "
          f"{'PASS' if not len(bad) else 'FAIL'}")
    if len(bad):
        print(bad[key + [f"{l}__n_cells" for l in labs]].to_string(index=False))
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("cells")
    c.add_argument("--name", required=True)
    c.add_argument("--x", required=True)
    c.add_argument("--y", required=True)
    c.add_argument("--rel", choices=["le", "eq"], required=True)
    c.add_argument("--component", nargs="*", default=[])
    c.add_argument("--out", required=True)
    t = sp.add_parser("table")
    t.add_argument("--bytrace", nargs="+", required=True)
    t.add_argument("--verdict", nargs="*", default=[])
    t.add_argument("--out-dir", required=True)
    t.add_argument("--out-prefix", required=True)
    a = ap.parse_args()
    cells(a) if a.cmd == "cells" else table(a)


if __name__ == "__main__":
    main()
