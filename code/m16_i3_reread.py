#!/usr/bin/env python3
"""m16_i3_reread.py — I3 模擬結果換基準重讀（只讀 CSV，不重跑模擬）。

為什麼（2026-10-20，使用者質疑探索「太多否定」）：I3 的判定只跟「每格事後挑出的最好現成規則」（7 選 1 的神諭）比。
實際部署不知道哪個最好，只會用一個固定的。這支把同一份 cells.csv／regret.csv 改成三種基準各報一次：
  (a) 神諭：每格最好的現成規則（原判定的基準）
  (b) 固定預設：vLLM 原生（ttft_ret_p95__vllm，不卸載）
  (b2) 固定 vLLM 原生卸載：vllm_tier（CPU＋SSD 串寫），每格都用它
  (c) 固定最佳：在同一組（tag）裡、所有格子都用同一個現成規則，事後挑中位數最好的那個（也是事後，對 κ 不利）
以及 regret.csv 的「搬錯硬體」比較：同一對設定下，把 A 的最好規則原封搬到 B（transplant）vs κ 感知，各離 B 的最好規則多遠。

gain＝1 − 候選 ÷ 基準（>0＝候選較好），和 I3 原報告同一定義。全部是〔模擬〕。

  python code/m16_i3_reread.py [--in results/m16_stage3_mi300x/i3_pause_sim]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from datetime import datetime, timezone

BASELINES = ["vllm", "vllm_cpu", "vllm_tier", "continuum", "continuum_cpu", "tokencake", "infercept"]
CANDS = ["kappa", "kappa_tier", "kappa_end"]
M = "ttft_ret_p95"


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summ(gs):
    return dict(n=len(gs), median=round(statistics.median(gs), 4), p10=round(q(gs, 0.1), 4), p90=round(q(gs, 0.9), 4),
                min=round(min(gs), 4), max=round(max(gs), 4), frac_ge_5pct=round(sum(g >= 0.05 for g in gs) / len(gs), 4),
                frac_le_m5pct=round(sum(g <= -0.05 for g in gs) / len(gs), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/m16_stage3_mi300x/i3_pause_sim")
    ap.add_argument("--out", default=None, help="預設＝--in")
    a = ap.parse_args()
    out = a.out or a.inp
    rid = os.environ.get("RUN_ID", "adhoc")
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cells = list(csv.DictReader(open(os.path.join(a.inp, "cells.csv"))))
    src_run = cells[0]["run_id"]
    tags = sorted({r["tag"] for r in cells})
    rows, summary = [], dict(run_id=rid, ts=ts, source_run_id=src_run, metric=M, kind="simulated", tags={})
    for tag in tags:
        cs = [r for r in cells if r["tag"] == tag]
        val = lambda r, k: float(r[f"{M}__{k}"]) if k in BASELINES else float(r[f"{M}_{k}"])
        # (c) 固定最佳：整組都用同一個現成規則，挑「對神諭的中位落差」最小的
        med_vs_oracle = {b: statistics.median(val(r, b) / float(r[f"{M}_best_baseline_val"]) for r in cs) for b in BASELINES}
        fixed_best = min(med_vs_oracle, key=med_vs_oracle.get)
        bases = {"oracle_per_cell": lambda r: float(r[f"{M}_best_baseline_val"]),
                 "fixed_default_vllm": lambda r: val(r, "vllm"),
                 # vLLM 0.28 原生的 CPU＋SSD 串寫：可部署、不需要事先知道答案的固定規則
                 "fixed_vllm_tier": lambda r: val(r, "vllm_tier"),
                 f"fixed_best_in_tag({fixed_best})": lambda r, b=fixed_best: val(r, b)}
        summary["tags"][tag] = dict(n_cells=len(cs), fixed_best=fixed_best,
                                    fixed_rules_median_ratio_vs_oracle={b: round(v, 4) for b, v in med_vs_oracle.items()},
                                    gains={})
        for c in CANDS:
            for bn, bf in bases.items():
                gs = [1 - val(r, c) / bf(r) for r in cs]
                s = summ(gs)
                summary["tags"][tag]["gains"][f"{c}_vs_{bn}"] = s
                rows.append(dict(run_id=rid, ts=ts, source_run_id=src_run, tag=tag, candidate=c, baseline=bn, **s))
    # 搬錯硬體：regret.csv 每一列＝一對設定（A→B），regret＝候選 ÷ B 的最好規則 − 1
    reg = list(csv.DictReader(open(os.path.join(a.inp, "regret.csv"))))
    summary["transplant"] = {}
    for what in sorted({r["what"] for r in reg}):
        for d in ("A→B", "B→A"):
            rs = [r for r in reg if r["what"] == what and r["dir"] == d]
            if not rs:
                continue
            tr = [float(r["regret_transplant"]) for r in rs]
            kp = [float(r["regret_kappa"]) for r in rs]
            summary["transplant"][f"{what} {d}"] = dict(
                n=len(rs), transplant_median=round(statistics.median(tr), 4), transplant_max=round(max(tr), 4),
                kappa_median=round(statistics.median(kp), 4), kappa_max=round(max(kp), 4),
                kappa_better_than_transplant=sum(k < t for k, t in zip(kp, tr)))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "reread_baselines.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(summary, open(os.path.join(out, "reread_baselines.json"), "w"), indent=1, ensure_ascii=False)
    print(f"〔模擬〕來源 run {src_run}；指標 {M}；gain＝1−候選÷基準（>0＝候選較好）")
    for tag, t in summary["tags"].items():
        print(f"\n[{tag}] {t['n_cells']} 格；整組固定最佳＝{t['fixed_best']}")
        for k, s in t["gains"].items():
            if k.startswith("kappa_vs") or k.endswith("_vs_fixed_vllm_tier"):
                print(f"  {k:45s} 中位 {s['median']:+.3f}  P10 {s['p10']:+.3f}  P90 {s['p90']:+.3f}  ≥5% 的格 {s['frac_ge_5pct']:.0%}")
    print("\n搬錯硬體（regret＝離 B 的最好規則多遠）")
    for k, s in summary["transplant"].items():
        print(f"  {k:28s} 原封搬過去 中位 {s['transplant_median']:+.3f}（最多 {s['transplant_max']:+.3f}）"
              f"｜κ 感知 中位 {s['kappa_median']:+.3f}（最多 {s['kappa_max']:+.3f}）｜κ 較好 {s['kappa_better_than_transplant']}/{s['n']}")


if __name__ == "__main__":
    main()
