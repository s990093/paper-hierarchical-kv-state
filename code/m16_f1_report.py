#!/usr/bin/env python3
"""m16_f1_report.py — F1 前綴復用地圖的報告（只讀 summary_*.json，只用 CPU、不 import torch）。

設計與事先寫死的條件：docs/stage3_20261019/experiment_matrix.md 的 F1 卡、v0_f1_i3_howto.md §3、code/m16_queue/jobs/f1_common.sh 檔頭。
  成功：找到「復用收益 < 5%」或「變差」的區域，而且能用（重疊、容量、併發／到達）說清楚邊界。
  停止：所有區域收益都很大且單調 → 沒有研究問題，只當 baseline。
  另看：平均 TTFT、P95 TTFT、goodput（嚴的 SLO＝1.5×t1）選出的最佳設定會不會不同。
每格＝(L, 容量, 重疊, 到達)；設定＝off（不復用）、apc、apccpu（APC＋CPU 卸載）。收益＝1 − 指標(設定) ÷ 指標(基準)（TTFT 越小越好；goodput 用比例差）。

  python code/m16_f1_report.py [--in results/m16_stage3_mi300x/f1_prefix_reuse]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from datetime import datetime, timezone

PAT = re.compile(r"m16-f1-(\d+)-(off|apc|apccpu)-(ample|c075|c040)-([0-9.]+)-(burst|p1|p2)\.json$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/m16_stage3_mi300x/f1_prefix_reuse")
    a = ap.parse_args()
    rid = os.environ.get("RUN_ID", "adhoc")
    cells, contaminated, files = {}, [], {}
    for f in sorted(glob.glob(os.path.join(a.inp, "summary_*.json"))):
        m = PAT.search(f)
        if not m:
            continue
        L, cfg, cap, ov, arr = int(m.group(1)), m.group(2), m.group(3), m.group(4), m.group(5)
        d = json.load(open(f))
        if d.get("guard", {}).get("contaminated") or not d.get("timing_valid", True):
            contaminated.append(os.path.basename(f))
            continue
        gp = d.get("goodput") or []
        cells.setdefault((L, cap, ov, arr), {})[cfg] = dict(
            mean=d["ttft_s"]["mean"], p95=d["ttft_s"]["p95"], n=d["ttft_s"]["n"], n_failed=d.get("n_failed"),
            good_strict=gp[0]["frac_of_measured"] if gp else None, good_loose=gp[1]["frac_of_measured"] if len(gp) > 1 else None,
            run_id=d["run_id"])
        files[d["run_id"]] = os.path.basename(f)
    rows, small, flips = [], [], []
    for (L, cap, ov, arr), c in sorted(cells.items()):
        if not all(k in c for k in ("off", "apc", "apccpu")):
            continue
        g = lambda x, y, k: 1 - c[x][k] / c[y][k]
        r = dict(L=L, cap=cap, overlap=ov, arrival=arr,
                 apc_vs_off_mean=round(g("apc", "off", "mean"), 4), apc_vs_off_p95=round(g("apc", "off", "p95"), 4),
                 cpu_vs_apc_mean=round(g("apccpu", "apc", "mean"), 4), cpu_vs_apc_p95=round(g("apccpu", "apc", "p95"), 4),
                 best_mean=min(c, key=lambda k: c[k]["mean"]), best_p95=min(c, key=lambda k: c[k]["p95"]),
                 best_goodput=max(c, key=lambda k: (c[k]["good_strict"] or 0, -c[k]["mean"])),
                 **{f"{k}_{m}": round(c[k][m], 4) if c[k][m] is not None else None for k in c for m in ("mean", "p95", "good_strict")})
        rows.append(r)
        if r["apc_vs_off_mean"] < 0.05 or r["apc_vs_off_p95"] < 0.05:
            small.append(r)
        if len({r["best_mean"], r["best_p95"], r["best_goodput"]}) > 1:
            flips.append(r)
    verdict = ("SUCCESS（有收益 <5% 或變差的區域）" if small else "STOP（每一格 APC 收益都 ≥5%）")
    out = dict(run_id=rid, ts=datetime.now(timezone.utc).isoformat(timespec="seconds"), n_cells=len(rows),
               contaminated_excluded=contaminated, verdict=verdict, n_small_or_negative=len(small), n_metric_flips=len(flips),
               small_or_negative=small, metric_flips=flips, rows=rows, source_files=files,
               rule="成功：有 APC 收益 <5%（平均或 P95）或變差的格子；停止：全部 ≥5%。另報平均／P95／goodput（SLO 1.5×t1）選出的最佳設定是否不同。")
    json.dump(out, open(os.path.join(a.inp, "f1_report.json"), "w"), indent=1, ensure_ascii=False)
    L = ["# F1 前綴復用地圖：報告", "", f"run：{rid}；格子 {len(rows)}；排除污染 {len(contaminated)}；判定：**{verdict}**", "",
         "| L | 容量 | 重疊 | 到達 | APC 對不復用（平均／P95） | CPU 卸載對 APC（平均／P95） | 最佳（平均／P95／goodput） |",
         "|--:|:--|--:|:--|--:|--:|:--|"]
    for r in rows:
        L.append(f"| {r['L']} | {r['cap']} | {r['overlap']} | {r['arrival']} | {r['apc_vs_off_mean']:+.1%}／{r['apc_vs_off_p95']:+.1%} | "
                 f"{r['cpu_vs_apc_mean']:+.1%}／{r['cpu_vs_apc_p95']:+.1%} | {r['best_mean']}／{r['best_p95']}／{r['best_goodput']} |")
    L += ["", f"收益 <5% 或變差的格子：{len(small)}；三種指標選出不同最佳設定的格子：{len(flips)}", "", "規則：" + out["rule"]]
    open(os.path.join(a.inp, "f1_report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:4]))
    print(f"small/negative {len(small)}, metric flips {len(flips)}")


if __name__ == "__main__":
    main()
