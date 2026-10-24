"""x7_compare.py — X7：會合點與 Cake 加速比的 3090／MI300X 並列表，和 2.2 的判定。只用 CPU、只讀 results/p3_explore/x7/ 的檔。

  python code/p3x/x7_compare.py --dir results/p3_explore/x7
讀：restore_3090_summary.csv（x7_summarize）、band_points.csv、band_edges.csv（x7_band）、mi300x_ref.json（x7_mi300x_ref）、constants_3090.json
寫：meet_compare.csv、model_check.csv、verdict.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime

import pandas as pd

SRID = os.environ.get("RUN_ID", "NO_RUN_ID")
# 格子裡的頻寬名 → MI300X meets.json 的名字（同一個頻寬才並列）
MI_NAME = {"cpu": "cpu", "mi_b16": "b16", "mi_b32": "b32", "mi_b48": "b48"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    D = lambda n: os.path.join(a.dir, n)  # noqa: E731
    S = pd.read_csv(D("restore_3090_summary.csv"))
    P = pd.read_csv(D("band_points.csv"))
    E = pd.read_csv(D("band_edges.csv"))
    M = json.load(open(D("mi300x_ref.json")))
    C = json.load(open(D("constants_3090.json")))
    t = datetime.now().astimezone().isoformat(timespec="seconds")
    S32 = S[(S.length == 32768) & (S.grid.isin(["grid_qwen", "grid_llama"]))]
    import x7_band as XB                                       # 同一個模型算任意頻寬
    spec = json.load(open(D("band_spec.json")))
    cases = {(c["platform"], c["model"], c["gpu_state"]): c for c in spec["cases"]}
    rows, chk = [], []
    for _, r in S32[(S32.config == "E0") & (S32["mode"] == "cake")].iterrows():
        bw = C["models"][r.model]["cpu_tier_h2d_GiBps_chunk_view"] if r.tier == "cpu" else float(r.tier_GiBps)
        pred = XB.evaluate(cases[("3090", r.model, r.gpu_state)], bw)
        mi_meas = M[r.model]["meets"]["E0"].get(MI_NAME.get(r.tier, "?"), {}).get(r.gpu_state, "")
        mi_bw = M[r.model]["h2d_GiBps"] if r.tier == "cpu" else float(r.tier_GiBps)
        mi_pred = XB.evaluate(cases[("MI300X", r.model, r.gpu_state)], mi_bw)
        lo = S32[(S32.model == r.model) & (S32.tier == r.tier) & (S32.gpu_state == r.gpu_state) & (S32.config == "E0")
                 & (S32["mode"] == "load_only")]
        co = S32[(S32.model == r.model) & (S32.gpu_state == r.gpu_state) & (S32["mode"] == "compute_only")]
        lo_t = float(lo.ttft_median_s.iloc[0]) if len(lo) else None
        co_t = float(co.ttft_median_s.iloc[0]) if len(co) else None
        sp = (min(x for x in (lo_t, co_t) if x is not None) / r.ttft_median_s) if (lo_t and co_t) else ""
        rows.append(dict(run_id=r.run_id, ts=r.ts, summarize_run_id=SRID, model=r.model, tier=r.tier, GiBps=bw, gpu_state=r.gpu_state,
                         b_3090_measured=r.b_median, b_3090_values=r.b_values, b_3090_model=pred["b_greedy"],
                         ttft_cake_3090_s=r.ttft_median_s, ttft_load_only_3090_s=lo_t if lo_t else "NOT_MEASURED",
                         ttft_compute_only_3090_s=co_t, speedup_measured_3090=round(sp, 4) if sp != "" else "NOT_MEASURED（只載入在這個頻寬沒跑）",
                         speedup_model_3090=round(pred["speedup_ttft"], 4), MI300X_GiBps=mi_bw,
                         b_MI300X_measured=mi_meas if mi_meas != "" else "NOT_MEASURED（MI300X 沒跑這個頻寬）",
                         b_MI300X_model=mi_pred["b_greedy"], speedup_model_MI300X=round(mi_pred["speedup_ttft"], 4)))
        chk.append(dict(run_id=r.run_id, ts=r.ts, summarize_run_id=SRID, model=r.model, tier=r.tier, GiBps=bw, gpu_state=r.gpu_state,
                        b_measured=r.b_median, b_model=pred["b_greedy"], abs_err=abs(r.b_median - pred["b_greedy"]),
                        within_1=abs(r.b_median - pred["b_greedy"]) <= 1))
    with open(D("meet_compare.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(D("model_check.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(chk[0]))
        w.writeheader()
        w.writerows(chk)
    ck = pd.DataFrame(chk)
    thr = [c for c in ck.itertuples() if c.tier != "cpu"]
    model_ok = sum(c.within_1 for c in thr) >= (2 / 3) * len(thr)
    # ---- 判定（§2.2）
    cpu = pd.DataFrame(rows)
    cpu = cpu[cpu.tier == "cpu"]
    a_cells = {f"{r.model}|{r.gpu_state}": r.b_3090_measured for r in cpu.itertuples()}
    cond_a = any(v >= 8 for v in a_cells.values())
    cond_i = all(v <= 3 for v in a_cells.values())
    e = E[(E.metric == "speedup_ttft") & (E.threshold == 1.3)]
    bands = {}
    cond_b = {"envelope": False, "main": False}
    cond_ii = {"envelope": True, "main": True}
    for m in ("qwen", "llama"):
        h2d = C["models"][m]["cpu_tier_h2d_GiBps_chunk_view"]
        for st in ("idle", "busy"):
            x = e[(e.platform == "3090") & (e.model == m) & (e.gpu_state == st)].iloc[0]
            y = e[(e.platform == "MI300X") & (e.model == m) & (e.gpu_state == st)].iloc[0]
            for kind in ("envelope", "main"):
                lo, hi = float(x[f"{kind}_lo_GiBps"]), float(x[f"{kind}_hi_GiBps"])
                mlo, mhi = float(y[f"{kind}_lo_GiBps"]), float(y[f"{kind}_hi_GiBps"])
                bands[f"{m}|{st}|{kind}"] = {"3090": [lo, hi], "MI300X": [mlo, mhi], "3090_cpu_h2d": h2d,
                                             "covers_cpu": lo <= h2d <= hi, "subset_of_MI300X": lo >= mlo and hi <= mhi}
                cond_b[kind] |= lo <= h2d <= hi
                cond_ii[kind] &= (lo >= mlo and hi <= mhi)
    if cond_b["envelope"] != cond_b["main"] or cond_ii["envelope"] != cond_ii["main"]:
        note = "外包絡與 main 的結論不同；判定取較不利於「有看頭」的那個（§2 補充 2）"
    else:
        note = "外包絡與 main 的結論相同"
    b_final = cond_b["envelope"] and cond_b["main"]
    ii_final = cond_ii["envelope"] or cond_ii["main"]          # 較不利於有看頭＝較容易判死路
    if cond_a or b_final:
        verdict = "有看頭"
    elif cond_i and ii_final:
        verdict = "死路"
    else:
        verdict = "可能"
    out = {"summarize_run_id": SRID, "ts": t, "verdict": verdict, "cond_a_cpu_meet_ge8": cond_a, "cond_b_band_covers_cpu": cond_b,
           "cond_i_cpu_meet_le3": cond_i, "cond_ii_band_subset_of_MI300X": cond_ii, "envelope_vs_main": note,
           "cpu_meet_cells": a_cells, "bands_ttft_1.3": bands, "model_validation": {
               "cells": len(thr), "within_1": int(sum(c.within_1 for c in thr)), "pass_2_of_3": model_ok}}
    json.dump(out, open(D("verdict.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
