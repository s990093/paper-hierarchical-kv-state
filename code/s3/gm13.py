#!/usr/bin/env python3
"""gm13.py — G-M13（docs/stage3_design_20261024/runs/G_M13_race_sim.md）：用 I3g 的逐請求 TTFT 重播「搬運和重算賽跑」。只用 CPU。

  run         → results/s3_h200/gm13/gm13_report.json、tables_gm13.md（平台是 MI300X 的 I3g 資料）
  --self-test 內插與規則
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
I3G = os.path.join(REPO, "results", "m16_stage3_mi300x", "i3g_agentic")
C6 = os.path.join(REPO, "results", "m7_write_policy_mi300x", "c6_vllm.csv")
OUT = os.path.join(REPO, "results", "s3_h200", "gm13")
RUNS = {"llama": dict(d="20261022-192954-m16-i3g-d", c="20261022-191335-m16-i3g-c")}
sys.path.insert(0, os.path.dirname(HERE))
from m16_i3g_report import pct  # noqa: E402  （同一個 P95 內插）


def run_id():
    return os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc"


def returns(run: str) -> list:
    rows = list(csv.DictReader(open(os.path.join(I3G, f"requests_{run}.csv"))))
    return [r for r in rows if r["turn"] not in ("", "0") and r["is_warmup"] in ("0", "False", "") and r["ok"] in ("1", "True") and r["ttft_s"]]


def cold_points() -> list:
    by = {}
    for r in csv.DictReader(open(C6)):
        if r["kind"] == "cold_compute":
            by.setdefault(int(r["L"]), []).append(float(r["ttft_s"]))
    return sorted((L, statistics.median(v)) for L, v in by.items())


def t_cold(L: float, pts: list) -> float:
    pts = [(0, 0.0)] + pts
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if L <= x1:
            return y0 + (y1 - y0) * (L - x0) / (x1 - x0)
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    return y1 + (y1 - y0) * (L - x1) / (x1 - x0)


def cmd_run(a):
    pts = cold_points()
    res = dict(run_id=run_id(), ts=datetime.now().astimezone().isoformat(timespec="seconds"), platform="MI300X", c6_points=pts,
               c6_src=os.path.relpath(C6, REPO), models={})
    for m, rr in RUNS.items():
        d, c = returns(rr["d"]), returns(rr["c"])
        p95_c = pct([float(r["ttft_s"]) for r in c], 0.95)
        out = {}
        for rule, k in (("H0", 1.0), ("H1", 2.0)):
            race, wins, wasted = [], 0, 0.0
            for r in d:
                td = float(r["ttft_s"])
                tc = k * t_cold(float(r["prompt_len"]), pts)
                if tc < td:
                    wins += 1
                    race.append(tc)
                else:
                    race.append(td)
                    wasted += max(0.0, td - (k - 1) * t_cold(float(r["prompt_len"]), pts))     # 讀先到：重算已經做了的時間
            out[rule] = dict(p95=pct(race, 0.95), p50=pct(race, 0.5), mean=statistics.fmean(race), frac_recompute_wins=wins / len(d),
                             wasted_gpu_s=wasted, n=len(d))
        p95_d = pct([float(r["ttft_s"]) for r in d], 0.95)
        out.update(p95_d=p95_d, p95_c=p95_c, n_c=len(c), threshold=1.5 * p95_c, runs=rr,
                   verdict="GM13-PASS" if out["H0"]["p95"] <= 1.5 * p95_c else "GM13-STOP")
        res["models"][m] = out
    res["models"]["qwen"] = dict(verdict="NOT_MEASURED", why="MI300X 上沒有 Qwen 的 vLLM 冷 prefill（c6_vllm.csv 只有 Llama）")
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "gm13_report.json"), "w"), indent=1, ensure_ascii=False)
    L = res["models"]["llama"]
    md = ["# G-M13：搬運和重算賽跑（重播 I3g，MI300X；登記 runs/G_M13_race_sim.md）", "",
          f"run {res['run_id']}；t_cold 點（L, s）：{pts}", "",
          "| 模型 | d 實測 P95 | c 實測 P95 | 門檻 1.5×c | H0 P95 | H0 重算贏 | H1 P95 | 判定 |", "|:--|--:|--:|--:|--:|--:|--:|:--|",
          f"| llama | {L['p95_d']:.2f} | {L['p95_c']:.2f} | {L['threshold']:.2f} | {L['H0']['p95']:.2f} | {100 * L['H0']['frac_recompute_wins']:.0f}% | "
          f"{L['H1']['p95']:.2f} | **{L['verdict']}** |", "| qwen | | | | | | | NOT_MEASURED |"]
    open(os.path.join(OUT, "tables_gm13.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


def self_test() -> bool:
    pts = [(8192, 0.44), (16384, 1.3), (32768, 4.58)]
    ok = abs(t_cold(8192, pts) - 0.44) < 1e-12 and abs(t_cold(4096, pts) - 0.22) < 1e-12 and abs(t_cold(24576, pts) - 2.94) < 1e-9
    print("self-test", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("cmd", nargs="?", default="run")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    cmd_run(a)
