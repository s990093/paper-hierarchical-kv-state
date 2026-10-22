"""p12_queue.py — ICPE P1-2 的 D（docs/h200/icpe/P1-2_cpu_symmetric_reruns.md §3）：H200 X4 的「排隊占比」拆解。

  python code/icpe/p12_queue.py --cfg results/p3_explore_h200/x4/x4_verdict_cfg.csv \
      --sweep <run 20261021-012002-p3h-x4-sweep>/x4_sweep.csv --out results/icpe_h200/p12/x4

D1：可進判定的格，share＝NP_qwait_p99_med ÷ NP_p99_med；中位數 ≥ 0.90 → 「p99 尾端仍幾乎全是排隊」成立。
D2：可進判定且 hr_mean_med ≥ 5% 的格，每個 seed 拆「該格平均的最好對手 − OP」的平均 TTFT 差＝Δ排隊＋Δ還原＋Δ其餘；
    每格取各項的 seed 中位數；≥ 50% 的格 Δ還原 ÷ Δ ≥ 0.5 → 「平均值的空間主要來自還原變快」，否則「主要來自排隊變短」。
只報告：QP 對 NP 的平均與 p99 改善。輸出 p12_queue_cells.csv、p12_queue_verdict.json（run_id、ts）。
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

CELL = ["trace", "rule", "p", "cpu_GiB", "slow", "fscale"]


def run_id():
    return os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc"


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = pd.read_csv(a.cfg)
    sw = pd.read_csv(a.sweep)
    el = cfg[cfg.eligible.astype(str) == "True"].copy()
    # D1
    el["share_q_p99"] = el.NP_qwait_p99_med / el.NP_p99_med
    d1 = dict(n_cells=int(len(el)), median=float(el.share_q_p99.median()), min=float(el.share_q_p99.min()),
              p10=float(el.share_q_p99.quantile(0.1)), max=float(el.share_q_p99.max()))
    d1["holds"] = bool(d1["median"] >= 0.90)
    # D2
    hot = el[el.hr_mean_med >= 0.05]
    rows = []
    for _, c in hot.iterrows():
        q = sw
        for k in CELL:
            q = q[q[k] == c[k]] if q[k].dtype == object else q[np.isclose(q[k].astype(float), float(c[k]))]
        rv = c.best_rival_mean
        comp = []
        for sd, x in q.groupby("seed"):
            x = x.set_index("policy")
            if rv not in x.index or "OP" not in x.index:
                continue
            d = x.loc[rv, "mean"] - x.loc["OP", "mean"]
            dq = x.loc[rv, "qwait_mean"] - x.loc["OP", "qwait_mean"]
            dr = x.loc[rv, "restore_mean"] - x.loc["OP", "restore_mean"]
            comp.append(dict(seed=sd, d=d, dq=dq, dr=dr, do=d - dq - dr, rr=dr / d if d else np.nan))
        if not comp:
            continue
        C = pd.DataFrame(comp)
        md, mq, mr, mo = (float(C[k].median()) for k in ("d", "dq", "dr", "do"))
        rows.append(dict(run_id=run_id(), ts=ts(), **{k: c[k] for k in CELL}, util=float(c.util), best_rival_mean=rv,
                         hr_mean_med=float(c.hr_mean_med), n_seeds=len(C), d_mean_s=md, d_qwait_s=mq, d_restore_s=mr, d_other_s=mo,
                         restore_share=mr / md if md else np.nan, median_seed_restore_share=float(C.rr.median()),
                         restore_dominant=bool(md and mr / md >= 0.5),
                         qp_vs_np_mean=float((x.loc["NP", "mean"] - x.loc["QP", "mean"]) / x.loc["NP", "mean"]) if "QP" in x.index else None))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(a.out, "p12_queue_cells.csv"), index=False)
    d2 = dict(n_cells=int(len(D)), n_restore_dominant=int(D.restore_dominant.sum()) if len(D) else 0)
    d2["restore_dominant_majority"] = bool(len(D) and d2["n_restore_dominant"] >= 0.5 * len(D))
    d2["conclusion"] = ("平均值的空間主要來自還原變快" if d2["restore_dominant_majority"] else "平均值的空間主要來自排隊變短") if len(D) else "沒有 hr_mean ≥5% 的格"
    d2["median_restore_share"] = float(D.restore_share.median()) if len(D) else None
    qp = dict(qp_vs_np_p99_median=float(el.hr_QP_vs_NP_p99_med.median()), qp_vs_np_p99_max=float(el.hr_QP_vs_NP_p99_med.max()))
    out = dict(run_id=run_id(), ts=ts(), src_cfg=a.cfg, src_sweep_run=sorted(sw.run_id.unique()), D1=d1, D2=d2, report_only_QP=qp,
               rule="P1-2 §3（事先登記 2026-10-22T04:14Z）")
    json.dump(out, open(os.path.join(a.out, "p12_queue_verdict.json"), "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    if len(D):
        pd.set_option("display.width", 220)
        print(D[CELL + ["util", "best_rival_mean", "hr_mean_med", "d_mean_s", "d_qwait_s", "d_restore_s", "d_other_s", "restore_share"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
