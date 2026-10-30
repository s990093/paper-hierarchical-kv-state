"""e2e_curve.py — E2E-C 的 MPS 取捨曲線（只報告，不判定；判定之後加，docs/stage3_design_20261024/runs/E2E_combo.md）。

每個（模型, 背景, trace, 共用方式）：前景＝回來請求平均 TTFT 對 K2（stream）的改善 (K2 − K3(m))／K2，背景＝K3(m) 的 B（§3.4，含上限本身的代價）；
各取該組格子（2 慢層 × 2 CPU）的中位數與範圍。stream 那一點＝K2 自己（改善 0）。FreeInference 第 9–16 週有未驗證的異常（同檔），所以 trace 分開列。

  python code/s3/e2e_curve.py --root results/s3_h200/e2e            # 登記版（CPU 層 35.4 GiB/s）
  python code/s3/e2e_curve.py --root results/icpe_h200/t1/e2e       # T1 版（H200 的 CPU 層）
→ <root>/mps_tradeoff.csv、mps_tradeoff.png
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics
from collections import defaultdict
from datetime import datetime

MODES = ("stream", "mps", "mps87", "mps75", "mps50")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--half", default="test", choices=["test", "dev"])
    a = ap.parse_args()
    agg = defaultdict(lambda: dict(sum=0.0, n=0, BW=0.0, W=0.0))
    srcs = set()
    for p in sorted(glob.glob(os.path.join(a.root, "raw", "e2e_*.csv"))):
        for r in csv.DictReader(open(p)):
            if r["half"] != a.half:
                continue
            srcs.add(r["run_id"])
            k = (r["model"], r["trace"], r["slow"], int(r["cpu_GiB"]), r["bg"], r["cfg"], r["mode"])
            g = agg[k]
            g["sum"] += float(r["sum"]); g["n"] += int(r["n"])
            g["BW"] += float(r["B"]) * float(r["W"]); g["W"] += float(r["W"])
    val = {k: (g["sum"] / g["n"], g["BW"] / g["W"]) for k, g in agg.items() if g["n"] and g["W"]}
    groups = defaultdict(list)
    for (m, t, s, c, b, cfg, mode), (ttft, B) in val.items():
        if cfg not in ("K2", "K3"):
            continue
        k2 = val.get((m, t, s, c, b, "K2", "stream"))
        if k2 is None:
            continue
        gain = (k2[0] - ttft) / k2[0]
        groups[(m, b, t, mode)].append((gain, B))
    rid = os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc"
    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    rows = []
    for (m, b, t, mode), xs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], MODES.index(kv[0][3]))):
        g = [x[0] for x in xs]
        B = [x[1] for x in xs]
        rows.append(dict(run_id=rid, ts=ts, half=a.half, model=m, bg=b, trace=t, mode=mode, n_cells=len(xs),
                         gain_vs_K2_median=statistics.median(g), gain_min=min(g), gain_max=max(g),
                         B_median=statistics.median(B), B_min=min(B), B_max=max(B), src_runs=";".join(sorted(srcs))))
    out = os.path.join(a.root, f"mps_tradeoff{'' if a.half == 'test' else '_dev'}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("→", out, len(rows))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    combos = sorted({(r["model"], r["bg"]) for r in rows})
    fig, axes = plt.subplots(1, len(combos), figsize=(4.2 * len(combos), 3.6), squeeze=False)
    for ax, (m, b) in zip(axes[0], combos):
        for t, mk in (("conversation", "o-"), ("freeinference", "s--")):
            pts = [r for r in rows if r["model"] == m and r["bg"] == b and r["trace"] == t]
            if not pts:
                continue
            ax.plot([r["B_median"] for r in pts], [100 * r["gain_vs_K2_median"] for r in pts], mk, label=t)
            for r in pts:
                ax.annotate(r["mode"], (r["B_median"], 100 * r["gain_vs_K2_median"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.axvline(0.95, color="grey", lw=0.8, ls=":")
        ax.set_title(f"{m} / {b.split(':')[0]}", fontsize=9)
        ax.set_xlabel("background throughput B (median)")
        ax.set_ylabel("TTFT gain vs K2 (%)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    png = out.replace(".csv", ".png")
    fig.savefig(png, dpi=130)
    print("→", png)


if __name__ == "__main__":
    main()
