"""x1_report.py — X1 的「值在哪裡」報表（只報告，不改判定；判定在 x1_sim.py analyze）。

讀 results/p3_explore/x1/x1_sweep*.csv，輸出：
  x1_value_vs_lru.csv   每格每個策略相對 LRU 的改善（回來請求平均 TTFT），加上各指標（中位數、p90、還原部分、全部請求總和）
  stdout                給文件用的表（markdown）
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "results", "p3_explore", "x1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", nargs="+", default=["x1_sweep.csv"])
    ap.add_argument("--out", default="x1_value_vs_lru.csv")
    a = ap.parse_args()
    d = pd.concat([pd.read_csv(os.path.join(OUT, x)) for x in a.sweep], ignore_index=True)
    fi = d.trace.str.startswith("fi:")
    if fi.any():
        g = d[fi]
        agg = g.groupby(["rule", "cpu_GiB", "policy", "slow", "gpu", "restore"]).agg(
            sum_ret=("sum_ret", "sum"), n_ret=("n_ret", "sum"), total_all=("total_all", "sum"),
            sum_ret_user=("sum_ret_user", "sum"), n_ret_user=("n_ret_user", "sum"),
            slow_write_GiB=("slow_write_GiB", "sum"), prefetch_GiB=("prefetch_GiB", "sum"),
            rho_w=("rho_w", "max"), run_id=("run_id", "first")).reset_index()
        agg["mean_ret"] = agg.sum_ret / agg.n_ret
        agg["mean_ret_user"] = agg.sum_ret_user / agg.n_ret_user
        agg["trace"] = "freeinference"
        d = pd.concat([d[~fi], agg], ignore_index=True)
    cell = ["trace", "rule", "cpu_GiB", "slow", "gpu", "restore"]
    mets = [m for m in ["mean_ret", "median_ret", "p90_ret", "mean_rest_ret", "total_all", "mean_ret_user"] if m in d]
    rows = []
    for k, g in d.groupby(cell):
        g = g.set_index("policy")
        if "LRU" not in g.index:
            continue
        for p, r in g.iterrows():
            row = dict(zip(cell, k), policy=p, src_run=r.run_id)
            for m in mets:
                base = g.loc["LRU", m]
                row[m] = r[m]
                row[f"gain_{m}"] = (base - r[m]) / base if (pd.notna(base) and base) else np.nan
            for m in ("slow_write_GiB", "rho_w", "prefetch_GiB"):
                row[m] = r.get(m, np.nan)
            rows.append(row)
    V = pd.DataFrame(rows)
    V.insert(0, "ts", datetime.now().astimezone().isoformat(timespec="seconds"))
    V.insert(0, "run_id", os.environ.get("RUN_ID", "adhoc"))
    V.to_csv(os.path.join(OUT, a.out), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 2000)
    pd.set_option("display.max_columns", 50)
    order = ["LRU", "LRU2", "ARC", "S3FIFO", "GDSF", "PENS", "LRU+NONE", "LRU+2HW", "LRU+2HA", "LRU+RPF",
             "O-BEL", "O-BELS", "O-DROP", "O-NOW", "O-PF", "O-ALL"]
    for tr in V.trace.unique():
        for ru in sorted(V[V.trace == tr].rule.unique()):
            X = V[(V.trace == tr) & (V.rule == ru) & (V.restore == "cake")]
            print(f"\n### {tr} [{ru}] Cake：回來請求平均 TTFT 相對 LRU 的改善 %（正＝比 LRU 快）")
            t = X.pivot_table(index=["slow", "gpu", "cpu_GiB"], columns="policy", values="gain_mean_ret")
            print((t[[c for c in order if c in t]] * 100).round(1).to_string())
            print(f"\n### {tr} [{ru}] Cake：LRU 的回來請求平均 TTFT（s）與 ρ_w")
            t = X[X.policy == "LRU"].pivot_table(index=["slow", "gpu"], columns="cpu_GiB", values="mean_ret")
            print(t.round(3).to_string())
            t = X[X.policy == "LRU"].pivot_table(index=["slow", "gpu"], columns="cpu_GiB", values="rho_w")
            print(t.round(2).to_string())


if __name__ == "__main__":
    main()
