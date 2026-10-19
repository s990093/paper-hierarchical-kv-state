"""x9b_diurnal.py — X9b 的 T5（I6：日夜週期下的寫入負載 ρ_KV）。BurstGPT、Azure LLM 2024。

判準在 X9 文件 §2.5（2026-10-17T13:41:37Z 寫定）。只做計數與算術，不擬合。
用法： python -I code/p3x/x9b_diurnal.py --out results/p3_explore/x9
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import statistics
from datetime import datetime, timezone

import numpy as np
import pandas as pd

DS = os.environ.get("TIARA_DATASETS", "/ssd8/hungwei/paper-hkv/datasets")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN_ID = os.environ.get("RUN_ID", "")
KiB = 1024
GiB = 1 << 30


def f0_ms():
    rows = list(csv.DictReader(open(os.path.join(REPO, "results/m7_write_policy_mi300x/calib_c1.csv"))))
    v = [float(r["ms"]) for r in rows if r["item"] == "f_chunk" and r["gpu_state"] == "idle" and r["chunk_idx"] == "0"]
    return statistics.median(v)


def hourly(t_s, tok):
    h = np.floor((t_s - t_s.min()) / 3600.0).astype(np.int64)
    D = np.bincount(h, weights=tok.astype(np.float64))
    return D[1:-1]   # 去掉頭尾不完整的小時


def runs_ge1(x):
    best = cur = 0
    for v in x:
        cur = cur + 1 if v >= 1.0 else 0
        best = max(best, cur)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert RUN_ID
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    traces = {}
    chk = {}
    # BurstGPT
    bg = {}
    for k in (1, 2, 3):
        d = pd.read_csv(f"{DS}/burstgpt/BurstGPT_without_fails_{k}.csv",
                        usecols=["Timestamp", "Request tokens", "Response tokens", "Total tokens"])
        ok = float(np.mean(d["Total tokens"] == d["Request tokens"] + d["Response tokens"]))
        chk[f"burstgpt_{k}_total_eq_sum"] = ok
        assert ok >= 0.99, f"BurstGPT_{k}: Total == Request + Response 只有 {ok:.4f}"
        bg[k] = d
    t1, t2 = bg[1]["Timestamp"].to_numpy(float), bg[2]["Timestamp"].to_numpy(float)
    cont = t2.min() >= t1.max() - 3600
    chk["burstgpt_2_continues_1"] = bool(cont)
    if cont:
        t12 = np.concatenate([t1, t2])
        tok12 = np.concatenate([(bg[1]["Request tokens"] + bg[1]["Response tokens"]).to_numpy(),
                                (bg[2]["Request tokens"] + bg[2]["Response tokens"]).to_numpy()])
        span = (t12.max() - t12.min()) / 86400
        chk["burstgpt_12_span_days"] = span
        assert abs(span - 121) / 121 <= 0.05, f"BurstGPT 前 4 個月跨度 {span:.1f} 天，README 寫 121 天"
        traces["burstgpt_first4months"] = (t12, tok12)
    else:
        s1 = (t1.max() - t1.min()) / 86400
        s2 = (t2.max() - t2.min()) / 86400
        chk["burstgpt_1_span_days"], chk["burstgpt_2_span_days"] = s1, s2
        assert abs(s1 + s2 - 121) / 121 <= 0.05, f"BurstGPT _1+_2 跨度 {s1+s2:.1f} 天，README 寫 121 天"
        traces["burstgpt_1"] = (t1, (bg[1]["Request tokens"] + bg[1]["Response tokens"]).to_numpy())
        traces["burstgpt_2"] = (t2, (bg[2]["Request tokens"] + bg[2]["Response tokens"]).to_numpy())
    t3 = bg[3]["Timestamp"].to_numpy(float)
    s3 = (t3.max() - t3.min()) / 86400
    chk["burstgpt_3_span_days"] = s3
    assert abs(s3 - 110) / 110 <= 0.05, f"BurstGPT_3 跨度 {s3:.1f} 天，README 寫 110 天"
    traces["burstgpt_3"] = (t3, (bg[3]["Request tokens"] + bg[3]["Response tokens"]).to_numpy())
    # Azure 2024
    for nm in ("conv", "code"):
        d = pd.read_csv(f"{DS}/azure-llm-2024/AzureLLMInferenceTrace_{nm}_1week.csv")
        # 20261017-214320 那次用 .astype("int64")/1e9，但 pandas 3 的解析度是 µs → 跨度少算 1000 倍，被規則 6 斷言擋下。改成明確換算秒
        tt = pd.to_datetime(d["TIMESTAMP"], utc=True, format="ISO8601")
        t = ((tt - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)
        span = (t.max() - t.min()) / 86400
        chk[f"azure2024_{nm}_span_days"] = span
        assert abs(span - 7) <= 0.2, f"Azure 2024 {nm} 跨度 {span:.2f} 天"
        assert (d["ContextTokens"] >= 0).all()
        traces[f"azure2024_{nm}"] = (t, (d["ContextTokens"] + d["GeneratedTokens"]).to_numpy())
    print("rule6", json.dumps(chk), flush=True)
    f0 = f0_ms()
    P0 = 512 / (f0 / 1e3)
    cfgs = []
    for bpt, pm, bw in itertools.product((("INT4", 40 * KiB), ("BF16", 128 * KiB)), (1, 2), (1.91, 0.239)):
        main_ = bpt[0] == "INT4" and pm == 1 and bw == 1.91
        cfgs.append(dict(bytes=bpt[0], bpt=bpt[1], P_mult=pm, ssd_GiBps=bw, main=main_))
    rows = []
    for name, (t, tok) in traces.items():
        D = hourly(t, tok)
        lam = D / D.max()
        shape = dict(n_hours=len(D), peak_over_mean=float(D.max() / D.mean()),
                     longest_run_lam_ge_0p8_h=runs_ge1(lam / 0.8), frac_hours_lam_ge_0p5=float(np.mean(lam >= 0.5)))
        for c in cfgs:
            rho = lam * P0 * c["P_mult"] * c["bpt"] / (c["ssd_GiBps"] * GiB)
            rows.append(dict(run_id=RUN_ID, ts=now, trace=name, **shape, bytes=c["bytes"], P_tok_s=P0 * c["P_mult"],
                             ssd_GiBps=c["ssd_GiBps"], main_setting=c["main"],
                             assumption=("measured" if c["main"] else "variant: " + ",".join(
                                 x for x, on in (("BF16", c["bytes"] == "BF16"), ("2x prefill [assume]", c["P_mult"] == 2),
                                                 ("8 GPUs/SSD [assume]", c["ssd_GiBps"] == 0.239)) if on)),
                             rho_max=float(rho.max()), frac_hours_rho_ge1=float(np.mean(rho >= 1)),
                             longest_run_rho_ge1_h=runs_ge1(rho), f0_ms=f0))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "x9b_t5_diurnal_rho.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
