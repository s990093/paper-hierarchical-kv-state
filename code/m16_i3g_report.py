#!/usr/bin/env python3
"""m16_i3g_report.py — I3g（agent 工具暫停時 KV 的去留，vLLM 0.28 端到端）的報告。只讀 requests_*.csv／summary_*.json，只用 CPU。

設計與事先寫死的判定：docs/stage3_20261019/experiment_log.md「I3g」（排進佇列前寫，不改）；驗證的對象＝i3_pause_sim.md §7 的預測：
  四個設定（同一個工作負載、Llama、TRITON_ATTN、HBM 壓到參考工作集的 0.25 倍）：
    a＝APC（被擠掉就重算）、b＝APC＋CPU、c＝APC＋CPU＋檔案層 /var/tmp（本地）、d＝同 c 但檔案層在 /mlsteam/data（NFS）
  模擬預測（回來那一輪的 TTFT P95，秒）：a 4.33、b 3.00、c 0.97、d 13.39。
  RANK：實測「回來那一輪」（turn ≥ 1、非 warm-up、成功）的 TTFT P95，c 最低且 d 最高 → SIM_RANK_CONFIRMED，否則 SIM_RANK_REJECTED。
  FLIP：P95(c) ≤ 0.95×P95(b) 且 P95(d) ≥ 1.05×P95(b)（加檔案層在本地 SSD 有幫助、在 NFS 反而變差）→ FLIP_CONFIRMED，否則 FLIP_NOT_CONFIRMED。
  另報每個設定的 實測／預測（容忍 ±30% 只是參考：模擬用的是 harness 的常數，vLLM 的重算快 1.7–2 倍、CPU 路徑慢約 12 倍）。
  timing_valid＝0（gpu_guard 污染）的 run 不算。

  python code/m16_i3g_report.py --in results/m16_stage3_mi300x/i3g_agentic
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from datetime import datetime, timezone

PRED = {"a": 4.33, "b": 3.00, "c": 0.97, "d": 13.39}
DESC = {"a": "APC（擠掉就重算）", "b": "APC＋CPU 卸載", "c": "APC＋CPU＋檔案層（本地 /var/tmp）", "d": "APC＋CPU＋檔案層（NFS）"}


def pct(v: list, q: float):
    if not v:
        return None
    s = sorted(v)
    k = (len(s) - 1) * q
    f = int(k)
    return s[f] + (s[min(f + 1, len(s) - 1)] - s[f]) * (k - f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/m16_stage3_mi300x/i3g_agentic")
    ap.add_argument("--pred", choices=("llama", "none"), default="llama",
                    help="none＝沒有模擬預測（Qwen，66）：RANK 改成和 Llama 的實測排序比（c 最低且 d 最高 → RANK_SAME_AS_LLAMA）")
    a = ap.parse_args()
    pred = PRED if a.pred == "llama" else {k: None for k in PRED}
    rid = os.environ.get("RUN_ID", "adhoc")
    cfg, bad = {}, []
    for f in sorted(glob.glob(os.path.join(a.inp, "summary_*.json"))):
        d = json.load(open(f))
        lab = (d.get("args", {}).get("label") or "")
        k = lab.rsplit("_", 1)[-1] if lab.startswith("i3g_") else None
        if k not in pred:
            continue
        if d.get("guard", {}).get("contaminated") or not d.get("timing_valid", True):
            bad.append(os.path.basename(f))
            continue
        rq = os.path.join(a.inp, f"requests_{d['run_id']}.csv")
        rows = list(csv.DictReader(open(rq)))
        ret = [float(r["ttft_s"]) for r in rows if r["turn"] not in ("", "0") and r["is_warmup"] in ("0", "False", "")
               and r["ok"] in ("1", "True") and r["ttft_s"]]
        first = [float(r["ttft_s"]) for r in rows if r["turn"] == "0" and r["is_warmup"] in ("0", "False", "") and r["ok"] in ("1", "True") and r["ttft_s"]]
        cfg[k] = dict(run_id=d["run_id"], label=lab, n_return=len(ret), p50=pct(ret, 0.5), p95=pct(ret, 0.95), mean=sum(ret) / len(ret) if ret else None,
                      first_p95=pct(first, 0.95), n_failed=d.get("n_failed"), pred_p95=pred[k], desc=DESC[k],
                      meas_over_pred=(pct(ret, 0.95) / pred[k]) if ret and pred[k] else None, metrics=d.get("server", {}).get("metrics") or None)
    have = all(k in cfg for k in pred)
    if have:
        p = {k: cfg[k]["p95"] for k in pred}
        rank = sorted(p, key=p.get)
        ok = rank[0] == "c" and rank[-1] == "d"
        rank_v = ("SIM_RANK_CONFIRMED" if ok else "SIM_RANK_REJECTED") if a.pred == "llama" else ("RANK_SAME_AS_LLAMA" if ok else "RANK_DIFFERS_FROM_LLAMA")
        flip_v = "FLIP_CONFIRMED" if p["c"] <= 0.95 * p["b"] and p["d"] >= 1.05 * p["b"] else "FLIP_NOT_CONFIRMED"
    else:
        rank, rank_v, flip_v = None, "INCOMPLETE", "INCOMPLETE"
    out = dict(run_id=rid, ts=datetime.now(timezone.utc).isoformat(timespec="seconds"), configs=cfg, contaminated_excluded=bad,
               rank_measured=rank, rank_sim=sorted(PRED, key=PRED.get) if a.pred == "llama" else None, pred=a.pred, verdict_rank=rank_v, verdict_flip=flip_v,
               rule=("RANK：回來那一輪 TTFT P95，c 最低且 d 最高 → SIM_RANK_CONFIRMED；FLIP：P95(c) ≤ 0.95×P95(b) 且 P95(d) ≥ 1.05×P95(b) → FLIP_CONFIRMED。"
                     "實測／預測的 ±30% 只是參考。污染的 run 不算。"))
    json.dump(out, open(os.path.join(a.inp, "i3g_report.json"), "w"), indent=1, ensure_ascii=False)
    f = lambda x, n=3: "—" if x is None else f"{x:.{n}f}"  # noqa: E731
    L = [f"# I3g：agent 工具暫停時 KV 的去留（vLLM 0.28 端到端；run {rid}）", "",
         f"判定：RANK **{rank_v}**（實測排名 {rank}；模擬 {out['rank_sim']}）；FLIP **{flip_v}**", "",
         "| 設定 | 說明 | 回來那一輪 n | P50 | P95 | 平均 | 第一輪 P95 | 模擬 P95 | 實測／模擬 | run |", "|:--|:--|--:|--:|--:|--:|--:|--:|--:|:--|"]
    for k in pred:
        c = cfg.get(k)
        if not c:
            L.append(f"| {k} | {DESC[k]} | NOT_MEASURED | | | | | {pred[k] or '—'} | | |")
            continue
        L.append(f"| {k} | {c['desc']} | {c['n_return']} | {f(c['p50'])} | {f(c['p95'])} | {f(c['mean'])} | {f(c['first_p95'])} | {pred[k] or '—'} | "
                 f"{f(c['meas_over_pred'], 2)} | {c['run_id']} |")
    L += ["", "規則：" + out["rule"], f"排除的污染 run：{bad or '無'}"]
    open(os.path.join(a.inp, "i3g_report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
