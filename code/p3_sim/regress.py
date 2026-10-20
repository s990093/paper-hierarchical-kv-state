"""regress.py — 兩次冒煙（smoke.py 的 smoke_cells.csv）逐欄比對（06 §14）。只比「舊的那次有的欄」，不印任何數值，只印哪些欄不同。

  python code/p3_sim/regress.py <舊 run 目錄> <新 run 目錄> [--map 'O(cpu-only)=O']

每一格的鍵＝(data, data_window, outer, policy, view, model, cpu_GiB, H_s)。不比的欄：elapsed_s、job_s、format_table（路徑：worktree 和 repo 不同；
sha256 照比）、run_id。--map 新=舊：新的那次的策略名對到舊的（例如 c0c179b 的 O ＝ 新的 O(cpu-only)）。
輸出 $RUN_DIR/regress_cells.csv（每格：一樣／哪些欄不同）與 regress_summary.csv（每個策略：格數、完全相同的格數、不同的欄）。
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import pandas as pd

SKIP = {"elapsed_s", "job_s", "format_table", "run_id"}
KEY = ["data", "data_window", "outer", "policy", "view", "model", "cpu_GiB", "H_s"]


def load(d):
    df = pd.read_csv(os.path.join(d, "smoke_cells.csv"), dtype=str, keep_default_na=False)
    for k in KEY:
        if k not in df.columns:
            df[k] = ""
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--map", nargs="*", default=[])
    a = ap.parse_args()
    out_dir = os.environ.get("RUN_DIR", ".")
    old, new = load(a.old), load(a.new)
    mp = dict(x.split("=", 1) for x in a.map)
    new = new.copy()
    new["policy_orig"] = new["policy"]
    new["policy"] = new["policy"].map(lambda p: mp.get(p, p))
    # 同名的策略（例如新的 O 和 map 過來的 O(cpu-only)→O）會撞鍵：map 過的優先，未 map 的同名列另外標成 policy+'(new)'
    if mp:
        clash = new["policy_orig"].isin(set(mp.values())) & ~new["policy_orig"].isin(set(mp))
        new.loc[clash, "policy"] = new.loc[clash, "policy_orig"] + "(new)"
    cols = [c for c in old.columns if c not in SKIP and c not in KEY]
    o = old.set_index(KEY)
    n = new.set_index(KEY)
    if not o.index.is_unique or not n.index.is_unique:
        print("鍵不唯一", o.index.duplicated().sum(), n.index.duplicated().sum())
        return 2
    rows = []
    missing_new = [k for k in o.index if k not in n.index]
    for k in o.index:
        if k not in n.index:
            rows.append(dict(zip(KEY, k), status="MISSING_IN_NEW", diff_cols=""))
            continue
        a_, b_ = o.loc[k], n.loc[k]
        diff = [c for c in cols if c not in n.columns or a_[c] != b_[c]]
        rows.append(dict(zip(KEY, k), status="SAME" if not diff else "DIFF", diff_cols=";".join(diff)))
    df = pd.DataFrame(rows)
    df.insert(0, "run_id", os.environ.get("RUN_ID", "adhoc"))
    df.to_csv(os.path.join(out_dir, "regress_cells.csv"), index=False)
    summ = []
    for pol, g in df.groupby("policy"):
        dc = sorted({c for s in g["diff_cols"] for c in s.split(";") if c})
        summ.append(dict(run_id=os.environ.get("RUN_ID", "adhoc"), policy=pol, n_cells=len(g), n_same=int((g.status == "SAME").sum()),
                         n_diff=int((g.status == "DIFF").sum()), n_missing=int((g.status == "MISSING_IN_NEW").sum()),
                         diff_cols=";".join(dc)))
    pd.DataFrame(summ).to_csv(os.path.join(out_dir, "regress_summary.csv"), index=False)
    for s in summ:
        print(f"{s['policy']}: cells={s['n_cells']} same={s['n_same']} diff={s['n_diff']} missing={s['n_missing']}"
              + (f" diff_cols={s['diff_cols'][:400]}" if s["diff_cols"] else ""), flush=True)
    print(f"old columns compared: {len(cols)}; old cells: {len(o)}; new cells: {len(n)}; missing in new: {len(missing_new)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
