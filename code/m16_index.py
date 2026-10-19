#!/usr/bin/env python3
"""m16_index.py — 第三階段的結果索引：從 results/m16_stage3_mi300x/ 各實驗的輸出抽關鍵數字，寫 docs/stage3_20261019/results.csv。

每列：exp、cell、metric、value、unit、kind（measured／simulated）、status（ok／CONTAMINATED／NOT_MEASURED）、run_id、source。
只讀檔、不碰 torch；沒有輸出的實驗寫一列 NOT_MEASURED（不跳過、不填 0）。

  python code/m16_index.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import statistics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results/m16_stage3_mi300x")
OUT = os.path.join(REPO, "docs/stage3_20261019/results.csv")
FIELDS = ["exp", "cell", "metric", "value", "unit", "kind", "status", "run_id", "source"]


def rel(p):
    return os.path.relpath(p, REPO)


def i1_prediction():
    rows = []
    for p in glob.glob(os.path.join(RES, "i1_concurrent_restore/prediction/prediction_*.json")):
        d = json.load(open(p))
        for tier, c in d["calibration_N1"].items():
            rows.append(dict(exp="I1-pred", cell=f"N=1|{tier}|idle", metric="sim/meas restore", value=c["ratio"], unit="ratio",
                             kind="simulated", status="ok", run_id=d["run_id"], source=rel(p)))
        for c in d["cells"]:
            cell = f"{c['workload']}|N={c['N']}|{c['gpu_state']}|{c['interference']}"
            for met in ("ttft_p95", "ttft_mean"):
                rows.append(dict(exp="I1-pred", cell=cell, metric=f"{met} gain per-request best vs rr×cake",
                                 value=c[f"{met}_gainA_perreq_vs_rr"], unit="fraction", kind="simulated", status="ok",
                                 run_id=d["run_id"], source=rel(p)))
                rows.append(dict(exp="I1-pred", cell=cell, metric=f"{met} gain joint vs per-request best",
                                 value=c[f"{met}_gainB_joint_vs_perreq"], unit="fraction", kind="simulated", status="ok",
                                 run_id=d["run_id"], source=rel(p)))
    return rows


def i1_measured():
    rows = []
    p = os.path.join(RES, "i1_concurrent_restore/interf_llama.json")
    if os.path.exists(p):
        d = json.load(open(p))
        st = "CONTAMINATED" if d.get("contaminated") else "ok"
        for k, v in d["summary"].items():
            rows.append(dict(exp="I1-interf", cell=k, metric="concurrent/solo (median)", value=v["k_median"], unit="ratio",
                             kind="measured", status=st, run_id=d["run_id"], source=rel(p)))
    p = os.path.join(RES, "i1_concurrent_restore/grid_llama.csv")
    if os.path.exists(p):
        cells = {}
        for r in csv.DictReader(open(p)):
            cells.setdefault((r["gpu_state"], r["workload"], r["N"], r["mode"], r["order"]), []).append(r)
        for k, rs in cells.items():
            st = "CONTAMINATED" if any(r["contaminated"] == "True" for r in rs) else "ok"
            rows.append(dict(exp="I1-grid", cell="|".join(k), metric="measured/predicted mean TTFT (median of reps)",
                             value=round(statistics.median(float(r["ratio_mean"]) for r in rs), 4), unit="ratio",
                             kind="measured", status=st, run_id=rs[0]["run_id"], source=rel(p)))
            rows.append(dict(exp="I1-grid", cell="|".join(k), metric="mean TTFT (median of reps)",
                             value=round(statistics.median(float(r["ttft_mean"]) for r in rs), 5), unit="s",
                             kind="measured", status=st, run_id=rs[0]["run_id"], source=rel(p)))
    return rows


def s1():
    rows = []
    p = os.path.join(RES, "s1_lossless/summary.json")
    if not os.path.exists(p):
        return rows
    d = json.load(open(p))
    for dn, s in d["dumps"].items():
        for k, v in s["bf16"].items():
            if k.startswith("all|") and "ratio" in v:
                rows.append(dict(exp="S1", cell=f"{dn}|{k}", metric="lossless ratio vs BF16", value=v["ratio"], unit="x",
                                 kind="measured", status="ok", run_id=d["run_id"], source=rel(p)))
            if k == "all|entropy":
                rows.append(dict(exp="S1", cell=f"{dn}|entropy", metric="ideal order-0 ratio (planes)",
                                 value=v["ideal_ratio_planes"], unit="x", kind="measured", status="ok", run_id=d["run_id"],
                                 source=rel(p)))
        for k, v in s["speed"].items():
            rows.append(dict(exp="S1", cell=f"{dn}|{k}", metric="decompress throughput", value=v["decomp_MBps"], unit="MB/s",
                             kind="measured", status="ok", run_id=d["run_id"], source=rel(p)))
    return rows


def i4():
    rows = []
    for p in glob.glob(os.path.join(RES, "i4_copy_batching/copybench_sdma*.csv")):
        for r in csv.DictReader(open(p)):
            st = "CONTAMINATED" if r.get("contaminated") == "True" else ("ok" if r.get("status") == "ok" else r.get("status"))
            rows.append(dict(exp="I4", cell=f"sdma={r['sdma']}|{r['gpu_state']}|{r['layout']}|{r['method']}",
                             metric="effective H2D", value=r.get("GiBps", ""), unit="GiB/s", kind="measured", status=st,
                             run_id=r["run_id"], source=rel(p)))
    return rows


PLANNED = ["S1", "I4", "I1-interf", "I1-grid", "C1a", "S2", "V0", "F2", "F1", "I3s", "I3g", "C1b"]


def main():
    rows = i1_prediction() + i1_measured() + s1() + i4()
    have = {r["exp"] for r in rows}
    for e in PLANNED:
        if e not in have:
            rows.append(dict(exp=e, cell="", metric="", value="", unit="", kind="", status="NOT_MEASURED", run_id="",
                             source=""))
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"寫好 {rel(OUT)}：{len(rows)} 列；有資料的實驗：{sorted(have)}")


if __name__ == "__main__":
    main()
