#!/usr/bin/env python3
"""m16_i1_predict.py — I1 的事先預測（CPU、純 Python）：用第二階段實測的常數，在「計算與載入互不干擾」
（CacheFlow 的假設）與「載入在計算忙時變慢」兩種假設下，模擬 N 個 session 同時回來的各策略。

GPU 實測（m16_i1_multisession.py）跑完後，拿實測和這裡的預測比：差多少＝忽略干擾的模型錯多少。
這一支的輸出是〔模擬〕，不是量測；寫在 GPU 實測之前（事先登記），之後不改。

  python code/m16_i1_predict.py --out results/m16_stage3_mi300x/i1_concurrent_restore/prediction

常數來源（全部從檔案讀，記進輸出）：
  f(i)、chunk_bytes、真實 pinned H2D、b16／b32／b48 頻寬：results/m10_precision_mi300x/speed_<m>/bw.json（run 20261015-154558-m10-calibf-llama）
  新一輪（256 token）時間、單 session 的實測還原（尺準）：results/m10_precision_mi300x/speed_<m>/summary_fp8.csv（E0、cake、32768）
  GPU 忙時的放大：results/m7_write_policy_mi300x/busy_levels.json（重算 ×2.07，busy2）；H2D 在 busy2 下 +50% 是 08 §2.1 的算術
  （80.0→120.1 ms，L2 §0.1）——只當敏感度參數 k_load，不是「載入線遇到本身的重算線」的量測（那個是 NOT_MEASURED，I1 的 GPU 實驗才會量）。
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m16_mcake import Sess, apply_mode, simulate, stats, ORDERS, MODES  # noqa: E402

REPO = os.path.dirname(HERE)


def consts(model: str) -> dict:
    bwp = os.path.join(REPO, f"results/m10_precision_mi300x/speed_{model}/bw.json")
    bw = json.load(open(bwp))
    f_ms = bw["f_ms_median_by_chunk"]
    assert len(f_ms) == bw["n_blocks"] == 64, "f 的長度和 n_blocks 對不上"
    cb = bw["chunk_bytes"]
    t_load = {"cpu": cb / (bw["real_pinned_h2d_GiBps"] * 2 ** 30)}
    for k, v in bw["targets"].items():
        t_load[k] = cb / (v["GiBps"] * 2 ** 30)
    sp = os.path.join(REPO, f"results/m10_precision_mi300x/speed_{model}/summary_fp8.csv")
    meas = {}
    for r in csv.DictReader(open(sp)):
        if r["config"] == "E0" and r["mode"] == "cake" and r["length"] == "32768":
            meas[(r["bw_name"], r["gpu_state"])] = dict(restore=float(r["median_t_restore"]), ttft=float(r["median_ttft"]),
                                                        run_ids=r["run_ids"])
    t_new = {st: sorted(m["ttft"] - m["restore"] for (b, s), m in meas.items() if s == st) for st in ("idle", "busy")}
    t_new = {k: v[len(v) // 2] for k, v in t_new.items() if v}
    busy = json.load(open(os.path.join(REPO, "results/m7_write_policy_mi300x/busy_levels.json")))
    return dict(model=model, f_ms=f_ms, chunk_bytes=cb, t_load=t_load, t_new=t_new, meas=meas,
                rec_busy_factor=busy["_measured_slowdown"]["busy2"],
                sources=dict(bw=os.path.relpath(bwp, REPO), bw_run=bw["run_id"], summary=os.path.relpath(sp, REPO),
                             busy="results/m7_write_policy_mi300x/busy_levels.json", busy_runs=busy["_source_runs"]))


def run(C: dict, out: str):
    rid = os.environ.get("RUN_ID", dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S") + "-m16-i1pred")
    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    os.makedirs(out, exist_ok=True)
    rows = []
    # 尺準：N＝1 的模擬 vs 第二階段實測（GPU 閒）
    calib = {}
    for tier in ("cpu", "b16", "b32", "b48"):
        if (tier, "idle") not in C["meas"]:
            continue
        f = lambda i: C["f_ms"][i] / 1e3          # noqa: E731
        r = simulate([Sess(n=64, tier=tier)], "rr", f, C["t_load"], t_new=C["t_new"]["idle"])
        calib[tier] = dict(sim_restore=round(r["sessions"][0]["restored"], 4), meas_restore=C["meas"][(tier, "idle")]["restore"],
                           sim_meet=r["sessions"][0]["n_rec"])
        calib[tier]["ratio"] = round(calib[tier]["sim_restore"] / calib[tier]["meas_restore"], 3)
    # 掃描
    workloads = {
        "homog_cpu": lambda N: [Sess(n=64, tier="cpu") for _ in range(N)],
        "homog_b32": lambda N: [Sess(n=64, tier="b32") for _ in range(N)],
        "mix_cpu_b32": lambda N: [Sess(n=64, tier="cpu" if k % 2 == 0 else "b32") for k in range(N)],
        "mix_len": lambda N: [Sess(n=(16, 32, 64)[k % 3], tier="cpu" if k % 2 == 0 else "b32") for k in range(N)],
    }
    states = {"idle": dict(fx=1.0, t_new=C["t_new"]["idle"]),
              "busy2": dict(fx=C["rec_busy_factor"], t_new=C["t_new"].get("busy", C["t_new"]["idle"]))}
    for wname, mk in workloads.items():
        for N in (1, 2, 4, 8):
            for sname, st in states.items():
                f = lambda i, fx=st["fx"]: C["f_ms"][i] / 1e3 * fx   # noqa: E731
                for interf, (kr, kl) in {"none": (1.0, 1.0), "load+50%": (1.0, 1.5)}.items():
                    for mode in MODES:
                      for pol in ORDERS:
                        ss = apply_mode(mk(N), mode, f, C["t_load"])
                        r = simulate(ss, pol, f, C["t_load"], t_new=st["t_new"], k_rec=kr, k_load=kl)
                        tt = stats([s["ttft"] for s in r["sessions"]])
                        rows.append(dict(run_id=rid, ts=ts, model=C["model"], workload=wname, N=N, gpu_state=sname,
                                         interference=interf, mode=mode, policy=pol, ttft_mean=round(tt["mean"], 4),
                                         ttft_p50=round(tt["p50"], 4), ttft_p95=round(tt["p95"], 4),
                                         makespan=round(r["makespan"], 4),
                                         n_rec=sum(s["n_rec"] for s in r["sessions"]),
                                         n_load=sum(s["n_load"] for s in r["sessions"]),
                                         assign="".join(s["assign"] or "-" for s in r["sessions"])))
    with open(os.path.join(out, f"prediction_{C['model']}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # 每格三個比較（事先寫死）：
    #   A 各自 Cake＋輪流（rr×cake）→ 最好的「各請求各自最佳」（g5b × 任一順序）：第二階段的 G5b＋排隊順序就能拿到多少
    #   B 最好的「各自最佳」→ 最好的 joint（同時考慮所有 session）：**I1 的新意只看這一項**
    best = {}
    for r in rows:
        key = (r["workload"], r["N"], r["gpu_state"], r["interference"])
        best.setdefault(key, []).append(r)
    summ = []
    for key, rs in best.items():
        rr = next(x for x in rs if x["mode"] == "cake" and x["policy"] == "rr")
        pre = [x for x in rs if x["mode"] in ("cake", "g5b")]
        jo = [x for x in rs if x["mode"] == "joint"]
        cell = dict(workload=key[0], N=key[1], gpu_state=key[2], interference=key[3], rr_cake_p95=rr["ttft_p95"],
                    rr_cake_mean=rr["ttft_mean"])
        for met in ("ttft_p95", "ttft_mean"):
            bp = min(pre, key=lambda x: x[met])
            bj = min(jo, key=lambda x: x[met])
            cell[f"{met}_best_perreq"] = f"{bp['mode']}×{bp['policy']}={bp[met]}"
            cell[f"{met}_best_joint"] = f"{bj['mode']}×{bj['policy']}={bj[met]}"
            cell[f"{met}_gainA_perreq_vs_rr"] = round(1 - bp[met] / rr[met], 4)
            cell[f"{met}_gainB_joint_vs_perreq"] = round(1 - bj[met] / bp[met], 4)
        summ.append(cell)
    json.dump(dict(run_id=rid, ts=ts, kind="〔模擬〕事先預測，不是量測", calibration_N1=calib,
                   constants=dict(t_load_s=C["t_load"], t_new_s=C["t_new"], rec_busy_factor=C["rec_busy_factor"]),
                   sources=C["sources"], cells=summ),
              open(os.path.join(out, f"prediction_{C['model']}.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(calib, ensure_ascii=False))
    for x in summ:
        if x["N"] in (4, 8):
            print(x["workload"], x["N"], x["gpu_state"], x["interference"], "| P95 A", x["ttft_p95_gainA_perreq_vs_rr"],
                  "B", x["ttft_p95_gainB_joint_vs_perreq"], x["ttft_p95_best_perreq"], x["ttft_p95_best_joint"],
                  "| mean A", x["ttft_mean_gainA_perreq_vs_rr"], "B", x["ttft_mean_gainB_joint_vs_perreq"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(consts(a.model), a.out)


if __name__ == "__main__":
    main()
