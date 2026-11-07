#!/usr/bin/env python3
"""icpe-adv：老師 11/8 第 2–3 點——「oracle 比對手慢」的格子從哪來（只讀現有表，不跑模擬）。

故事：舊稿圖 2 的 −4.4%（FreeInference、CPU 16 GiB、SSD 級）是「完美預取比最好的不預測做法慢」。
這支程式回答兩件事：
  A. 那一格的原始 sweep 列：6 週各自的值、哪一週造成負值、搬運量（降級次數、從慢層讀回次數、預取量）。
  B. 新模擬器（偏離 14 修正版：版本 B 的 P1 表、版本 C 的 apc_d14 表）裡，每個元件有多少格 oracle 比對手慢、最慢多少。
oracle 的定義沒有改：④＝min over COMPONENT_POLICIES[c]（x1_sim.py），不含「不預測」的規則，所以 ④ 可以比 ② 慢。

輸出（--out 目錄）：
  adv_neg_A_cell_weeks.csv      A 的逐週列（old sweep）
  adv_neg_A_cell_check.csv      A 的 6 週彙總，和 x1fi_headroom_components.csv 那一列逐值核對
  adv_neg_B_by_variant.csv      B：表 × 變體 × 元件的計數（彙總，不含逐對話）
  adv_neg_B_by_component.csv    B：表 × 紀錄家族 × 元件的計數
每列帶 run_id、ts、host。
"""
import argparse
import glob
import gzip
import lzma
import os
import socket
import sys
from datetime import datetime

import pandas as pd

RUN_ID = os.environ.get("RUN_ID", "adhoc")
TS = datetime.now().astimezone().isoformat(timespec="seconds")
HOST = socket.gethostname()

OLD_SWEEP = "results/icpe_h200/t5/p12/x1/x1_sweep_fi.csv.gz"
OLD_COMP = "results/icpe_h200/t5/p12/x1/x1fi_headroom_components.csv"
CELL = dict(cpu_GiB=16, slow="ssd", restore="cake")
POLS = ["LRU", "LRU+RPF", "O-PF"]          # O-PF 的 ②＝{LRU, LRU+RPF}（x1_sim.py RIVALS），④＝{O-PF}
COLS = ["n_ret", "sum_ret", "mean_ret", "ret_cpu_loads", "ret_slow_loads", "ret_recompute",
        "prefetch_GiB", "n_demote", "slow_write_GiB", "slow_read_GiB", "rho_w"]


def stamp(df):
    df.insert(0, "host", HOST)
    df.insert(0, "ts", TS)
    df.insert(0, "run_id", RUN_ID)
    return df


def part_a(out):
    d = pd.read_csv(OLD_SWEEP, compression="gzip")
    m = (d.cpu_GiB == CELL["cpu_GiB"]) & (d.slow == CELL["slow"]) & (d.restore == CELL["restore"]) & d.policy.isin(POLS)
    w = d.loc[m, ["trace", "gpu", "policy", "run_id"] + COLS].rename(columns={"run_id": "src_run"})
    # 規則 6 式的自檢：mean_ret 必須等於 sum_ret ÷ n_ret（確認 sum_ret 的單位，彙總才能用它）
    bad = (w.sum_ret / w.n_ret - w.mean_ret).abs() > 1e-5
    assert not bad.any(), f"mean_ret != sum_ret/n_ret in {bad.sum()} rows"
    w = w.sort_values(["gpu", "trace", "policy"]).reset_index(drop=True)
    stamp(w.copy()).to_csv(f"{out}/adv_neg_A_cell_weeks.csv", index=False)

    comp = pd.read_csv(OLD_COMP)
    rows = []
    for gpu, g in w.groupby("gpu"):
        assert g.trace.nunique() == 6, f"expected 6 weeks, got {g.trace.nunique()}"
        agg = g.groupby("policy").agg(n_ret=("n_ret", "sum"), sum_ret=("sum_ret", "sum"))
        agg["T"] = agg.sum_ret / agg.n_ret
        t_or = agg.loc["O-PF", "T"]
        riv = agg.loc[["LRU", "LRU+RPF"], "T"]
        best = riv.idxmin()
        h = (riv.min() - t_or) / riv.min()
        ref = comp[(comp.cpu_GiB == 16) & (comp.slow == "ssd") & (comp.gpu == gpu) & (comp.restore == "cake")
                   & (comp.component == "O-PF")]
        assert len(ref) == 1, f"reference row count {len(ref)}"
        ref = ref.iloc[0]
        # 每週的 oracle 減對手：找出負值來自哪一週
        per = g.pivot(index="trace", columns="policy", values="mean_ret")
        per["oracle_minus_best_rival_s"] = per["O-PF"] - per[["LRU", "LRU+RPF"]].min(axis=1)
        worst = per.oracle_minus_best_rival_s.idxmax()
        rows.append(dict(gpu=gpu, T_oracle=t_or, T_LRU=agg.loc["LRU", "T"], T_LRU_RPF=agg.loc["LRU+RPF", "T"],
                         best_rival=best, headroom=h, ref_T_oracle=ref.T_oracle, ref_T_rival=ref.T_rival,
                         ref_best_rival=ref.best_rival, ref_headroom=ref.headroom,
                         match=bool(abs(h - ref.headroom) < 1e-9 and best == ref.best_rival),
                         n_weeks_oracle_slower=int((per.oracle_minus_best_rival_s > 0).sum()),
                         worst_week=worst, worst_week_oracle_minus_rival_s=per.loc[worst, "oracle_minus_best_rival_s"],
                         ref_src=OLD_COMP, ref_run=ref.run_id))
    r = pd.DataFrame(rows)
    stamp(r).to_csv(f"{out}/adv_neg_A_cell_check.csv", index=False)
    if not r.match.all():
        print("PART A MISMATCH vs reference row", file=sys.stderr)
        print(r.to_string(), file=sys.stderr)
        sys.exit(2)
    print(r.to_string())


def family(trace):
    t = str(trace).replace("nb:", "")
    for k in ["semianalysis", "tracelab", "openhands", "miniswe", "sharegpt", "wildchat", "lmsys", "ultrachat",
              "bailian", "cachewise"]:
        if t.startswith(k):
            return k
    if t.startswith("fi"):
        return "freeinference"
    return t  # conversation / toolagent（Mooncake）


def load_tables():
    tabs = []
    for f in sorted(glob.glob("results/icpe_next/main/verB_d14/*_p1_headroom_components.csv.xz")):
        tabs.append(("verB_d14_p1", f))
    for f in sorted(glob.glob("results/icpe_next/apc_d14/*/verC_headroom_components.csv")):
        tabs.append(("verC_d14", f))
    return tabs


def part_b(out):
    recs = []
    for tab, f in load_tables():
        d = pd.read_csv(f, compression="xz" if f.endswith(".xz") else None)
        if (d.trace.astype(str).str.contains("cachewise")).any():
            d = d[~d.trace.astype(str).str.contains("cachewise")]   # 偏離 15：論文版不含 CacheWise
        if d.empty:
            continue
        src_runs = ";".join(sorted(set(d.run_id.astype(str))))
        for (tr, rule, comp), g in d.groupby(["trace", "rule", "component"]):
            ws = g.write_saturated.astype(str).str.lower().eq("true")
            gi = g.loc[g.headroom.idxmin()]
            recs.append(dict(table=tab, file=f, src_analyze_runs=src_runs, trace=tr, rule=rule, family=family(tr), component=comp,
                             n_cells=len(g), n_neg=int((g.headroom < 0).sum()),
                             n_le_m1pct=int((g.headroom <= -0.01).sum()), n_le_m5pct=int((g.headroom <= -0.05).sum()),
                             n_le_m15pct=int((g.headroom <= -0.15).sum()),
                             n_le_m5pct_not_ws=int(((g.headroom <= -0.05) & ~ws).sum()),
                             n_ge_15pct=int((g.headroom >= 0.15).sum()),
                             min_headroom=gi.headroom, min_cpu_GiB=gi.cpu_GiB, min_slow=gi.slow, min_gpu=gi.gpu,
                             min_oracle=f"{gi.oracle_policy}/{gi.oracle_gpu_policy}/{gi.oracle_slow_policy}",
                             min_rival=f"{gi.best_rival}/{gi.best_rival_gpu_policy}/{gi.best_rival_slow_policy}",
                             min_T_oracle=gi.T_oracle, min_T_rival=gi.T_rival, min_write_saturated=gi.write_saturated))
    b = pd.DataFrame(recs)
    b["variant"] = b.trace.astype(str) + "|" + b.rule.astype(str)
    stamp(b.copy()).to_csv(f"{out}/adv_neg_B_by_variant.csv", index=False)
    s = (b.groupby(["table", "family", "component"])
         .agg(n_variants=("variant", "nunique"), n_cells=("n_cells", "sum"), n_neg=("n_neg", "sum"),
              n_le_m1pct=("n_le_m1pct", "sum"), n_le_m5pct=("n_le_m5pct", "sum"),
              n_le_m15pct=("n_le_m15pct", "sum"), n_le_m5pct_not_ws=("n_le_m5pct_not_ws", "sum"),
              n_ge_15pct=("n_ge_15pct", "sum"), min_headroom=("min_headroom", "min"))
         .reset_index())
    stamp(s).to_csv(f"{out}/adv_neg_B_by_component.csv", index=False)
    pd.set_option("display.width", 250)
    print(s[s.n_le_m5pct > 0].to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    part_a(a.out)
    part_b(a.out)
