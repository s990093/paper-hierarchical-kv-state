#!/usr/bin/env python3
"""m16_i3g_rep_report.py — I3g 重跑（65_i3g_rep）的報告：b、c 各三次（r1＝64 單卡、r2＝卡 0、r3＝卡 1；r2／r3 兩張卡同時跑）。只用 CPU。

事先寫死的判定（experiment_log.md「I3g 重跑（65）」，排進佇列前寫）：
  指標＝回來那一輪（turn ≥ 1、非 warm-up、成功）的 TTFT P95。三對（r1、r2、r3，各自同一張卡、同一個條件）裡
  P95(c) ≤ 0.95×P95(b) 的對數 = 3 → C_BETTER_REPLICATED；= 0 → C_BETTER_NOT_REPLICATED；1–2 → MIXED。
  任一對缺（沒跑、污染）→ INCOMPLETE（照實寫幾對）。另報同一設定三次 P95 的最大／最小。

  python3 -I code/m16_i3g_rep_report.py --base results/m16_stage3_mi300x/i3g_agentic
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from datetime import datetime, timezone


def pct(v, q):
    s = sorted(v)
    k = (len(s) - 1) * q
    f = int(k)
    return s[f] + (s[min(f + 1, len(s) - 1)] - s[f]) * (k - f)


def ret_p95(path):
    rows = list(csv.DictReader(open(path)))
    v = [float(r["ttft_s"]) for r in rows if r["turn"] not in ("", "0") and r["is_warmup"] in ("0", "False", "") and r["ok"] in ("1", "True") and r["ttft_s"]]
    return (pct(v, 0.95) if v else None), len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/m16_stage3_mi300x/i3g_agentic")
    a = ap.parse_args()
    runs, bad = {}, []
    base = json.load(open(os.path.join(a.base, "i3g_report.json")))
    for k in ("b", "c"):
        c = base["configs"][k]
        p, n = ret_p95(os.path.join(a.base, f"requests_{c['run_id']}.csv"))
        runs[(k, "r1")] = dict(run_id=c["run_id"], p95=p, n=n, card=1, cond="64：單卡")
    for f in sorted(glob.glob(os.path.join(a.base, "rep", "summary_*.json"))):
        d = json.load(open(f))
        lab = d.get("args", {}).get("label") or ""
        parts = lab.split("_")
        if len(parts) != 3 or parts[0] != "i3g" or parts[1] not in ("b", "c") or parts[2] not in ("r2", "r3"):
            continue
        if d.get("guard", {}).get("contaminated") or not d.get("timing_valid", True):
            bad.append(os.path.basename(f))
            continue
        p, n = ret_p95(os.path.join(a.base, "rep", f"requests_{d['run_id']}.csv"))
        runs[(parts[1], parts[2])] = dict(run_id=d["run_id"], p95=p, n=n, n_failed=d.get("n_failed"), card={"r2": 0, "r3": 1}[parts[2]],
                                          cond="65：兩張卡同時跑")
    pairs = {}
    for r in ("r1", "r2", "r3"):
        b, c = runs.get(("b", r)), runs.get(("c", r))
        if b and c and b["p95"] and c["p95"]:
            pairs[r] = dict(p95_b=b["p95"], p95_c=c["p95"], ratio=c["p95"] / b["p95"], c_better=c["p95"] <= 0.95 * b["p95"])
    if len(pairs) < 3:
        v = f"INCOMPLETE（{len(pairs)}／3 對）"
    else:
        nb = sum(p["c_better"] for p in pairs.values())
        v = {3: "C_BETTER_REPLICATED", 0: "C_BETTER_NOT_REPLICATED"}.get(nb, "MIXED")
    spread = {}
    for k in ("b", "c"):
        v_ = [x["p95"] for (kk, _), x in runs.items() if kk == k and x["p95"] is not None]
        spread[k] = dict(min=min(v_) if v_ else None, max=max(v_) if v_ else None, n_runs=len(v_))
    out = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now(timezone.utc).isoformat(timespec="seconds"), verdict=v,
               runs={f"{k}_{r}": x for (k, r), x in sorted(runs.items())}, pairs=pairs, spread=spread, contaminated_excluded=bad,
               rule="三對（r1、r2、r3）裡 P95(c) ≤ 0.95×P95(b) 的對數 3 → C_BETTER_REPLICATED；0 → C_BETTER_NOT_REPLICATED；1–2 → MIXED；缺對 → INCOMPLETE。")
    od = os.path.join(a.base, "rep")
    json.dump(out, open(os.path.join(od, "i3g_rep_report.json"), "w"), indent=1, ensure_ascii=False)
    L = [f"# I3g 重跑：c（＋本地檔案層）對 b（＋CPU）（run {out['run_id']}）", "", f"判定：**{v}**", "",
         "| 對 | 條件 | 卡 | b 的 P95（秒） | c 的 P95（秒） | c／b | c 好 ≥5%？ | run（b／c） |", "|:--|:--|--:|--:|--:|--:|:--|:--|"]
    for r in ("r1", "r2", "r3"):
        b, c, p = runs.get(("b", r)), runs.get(("c", r)), pairs.get(r)
        if not p:
            fb = "NOT_MEASURED" if not b or b["p95"] is None else f"{b['p95']:.3f}"
            fc = "NOT_MEASURED" if not c or c["p95"] is None else f"{c['p95']:.3f}"
            L.append(f"| {r} | | | {fb} | {fc} | | | |")
            continue
        L.append(f"| {r} | {b['cond']} | {b['card']} | {p['p95_b']:.3f} | {p['p95_c']:.3f} | {p['ratio']:.3f} | {'是' if p['c_better'] else '否'} | "
                 f"{b['run_id']}／{c['run_id']} |")
    L += ["", "同一設定各次的 P95：" + "；".join(f"{k} {s['min']:.3f}–{s['max']:.3f} 秒（{s['n_runs']} 次）" if s["n_runs"] else f"{k} NOT_MEASURED" for k, s in spread.items()),
          "", "規則：" + out["rule"], f"排除的污染 run：{bad or '無'}"]
    open(os.path.join(od, "i3g_rep_report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
