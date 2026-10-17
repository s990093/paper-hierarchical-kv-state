"""x7_band_spec.py — X7：組 x7_band.py 的輸入（f(i)、block 位元組、新一輪時間、要列出的頻寬點），全部從檔案讀。只用 CPU。

  python code/p3x/x7_band_spec.py --out spec.json --calibf-qwen <run> --calibf-llama <run> --busy-qwen <run> --busy-llama <run> \
      --grid-qwen <run>... --grid-llama <run>...
GPU 忙的 f：3090＝閒的 f(i) × 實測變慢倍數（x7_busy.py，0.7:20:8192，block 0／32／63 三點之間線性內插）；
            MI300X＝閒的 f(i) × 2.0674（第一階段 C5，只量 block 32，calib_c5.csv duty 0.7 的中位數）。
新一輪時間：同一台、同一模型、32K 還原格子裡 t_new_turn 的中位數（GPU 閒／忙分開）。
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

RUNS = os.environ.get("TIARA_RUNS", "/ssd8/hungwei/paper-hkv/runs")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    for m in ("qwen", "llama"):
        ap.add_argument(f"--calibf-{m}", required=True)
        ap.add_argument(f"--busy-{m}", required=True)
        ap.add_argument(f"--grid-{m}", nargs="*", default=[])
        ap.add_argument(f"--tnew-ladder-{m}", help="沒有還原格子時：m7_calib c1 的 run，取 t_new256 在 block 64（只有 GPU 閒；忙＝NOT_MEASURED）")
    a = ap.parse_args()
    cases, points = [], []
    mi_c5 = pd.read_csv(os.path.join(REPO, "results", "m7_write_policy_mi300x", "calib_c5.csv"))
    mi_slow = float(mi_c5[mi_c5.duty == 0.7].slowdown.median())
    for m in ("qwen", "llama"):
        bw = json.load(open(os.path.join(RUNS, getattr(a, f"calibf_{m}"), "bw.json")))
        f = np.array(bw["f_ms_median_by_chunk"])
        b = pd.read_csv(os.path.join(RUNS, getattr(a, f"busy_{m}"), "busy.csv"))
        b = b[b.cfg == "0.7:20:8192"].groupby("chunk_idx").slowdown.median()
        sd = np.interp(np.arange(len(f)), b.index.values, b.values)
        if getattr(a, f"grid_{m}"):
            g = pd.concat([pd.read_csv(os.path.join(RUNS, r, "restores.csv")) for r in getattr(a, f"grid_{m}")])
            g = g[g.length == 32768]
            tn = g.groupby("gpu_state").t_new_turn.median().to_dict()
            tsrc = " ".join(getattr(a, f"grid_{m}"))
        else:                                              # 退路：只有 GPU 閒的新一輪時間（c1 的 t_new256@block64，1 次）
            lr = getattr(a, f"tnew_ladder_{m}")
            c1 = pd.read_csv(os.path.join(RUNS, lr, "calib_c1_mc128.csv"))
            tn = {"idle": float(c1[(c1.item == "t_new256") & (c1.chunk_idx == 64)].ms.median()) / 1e3, "busy": None}
            tsrc = f"{lr} (t_new256@block64, idle only; busy NOT_MEASURED → busy 的 TTFT 帶改用只看還原時間)"
        src = f"f={getattr(a, f'calibf_{m}')}; busy={getattr(a, f'busy_{m}')}; t_new={tsrc}"
        cases.append(dict(platform="3090", model=m, gpu_state="idle", f_ms=f.tolist(), chunk_bytes=bw["chunk_bytes"],
                          t_new_ms=tn["idle"] * 1e3, src=src))
        cases.append(dict(platform="3090", model=m, gpu_state="busy", f_ms=(f * sd).tolist(), chunk_bytes=bw["chunk_bytes"],
                          t_new_ms=tn["busy"] * 1e3 if tn["busy"] is not None else None, src=src, busy_slowdown_by_chunk=dict(zip(map(int, b.index), b.values.tolist()))))
        points.append(dict(name=f"3090_cpu_real_{m}", model=m, GiBps=bw["real_pinned_h2d_GiBps"]))
        for k, v in bw["targets"].items():
            points.append(dict(name=f"3090_{k}_{m}", model=m, GiBps=v["GiBps"]))
        # MI300X（m10 的 calib-f；新一輪＝m10 grid 的 t_new_turn 中位數）
        mbw = json.load(open(os.path.join(REPO, "results", "m10_precision_mi300x", f"speed_{m}", "bw.json")))
        mr = pd.read_csv(os.path.join(REPO, "results", "m10_precision_mi300x", f"speed_{m}", "restores.csv.gz"), low_memory=False)
        mr = mr[(mr.length == 32768) & (mr.run_id.str.contains("grid-fp8"))]
        mtn = mr.groupby("gpu_state").t_new_turn.median().to_dict()
        mf = np.array(mbw["f_ms_median_by_chunk"])
        msrc = f"f={mbw['run_id']}; busy=calib_c5 duty0.7 ×{mi_slow:.4f}; t_new={' '.join(mr.run_id.unique())}"
        cases.append(dict(platform="MI300X", model=m, gpu_state="idle", f_ms=mf.tolist(), chunk_bytes=mbw["chunk_bytes"],
                          t_new_ms=mtn["idle"] * 1e3, src=msrc))
        cases.append(dict(platform="MI300X", model=m, gpu_state="busy", f_ms=(mf * mi_slow).tolist(), chunk_bytes=mbw["chunk_bytes"],
                          t_new_ms=mtn["busy"] * 1e3, src=msrc))
        points.append(dict(name=f"MI300X_cpu_real_{m}", model=m, GiBps=mbw["real_pinned_h2d_GiBps"]))
        for k, v in mbw["targets"].items():
            points.append(dict(name=f"MI300X_{k}_{m}", model=m, GiBps=v["GiBps"]))
    # 第一階段的 A0 驗證：MI300X Llama 32K（phase1 C1 的 f，t_new＝C1 的 t_new256 在 block 64）
    c1 = pd.read_csv(os.path.join(REPO, "results", "m7_write_policy_mi300x", "calib_c1.csv"))
    f1 = c1[(c1.item == "f_chunk") & (c1.chunk_idx < 64)].groupby("chunk_idx").ms.median().values
    tn1 = float(c1[(c1.item == "t_new256") & (c1.chunk_idx == 64)].ms.median())
    cases.append(dict(platform="MI300X_phase1", model="llama", gpu_state="idle", f_ms=f1.tolist(), chunk_bytes=64 << 20,
                      t_new_ms=tn1, src="f,t_new=results/m7_write_policy_mi300x/calib_c1.csv " + " ".join(c1.run_id.unique())))
    a0 = pd.read_csv(os.path.join(REPO, "results", "m7_write_policy_mi300x", "summary_a0_speedup.csv"))
    for _, r in a0[a0.L == 32768].iterrows():
        points.append(dict(name=f"A0_{r.bw}", model="llama", GiBps=float(r.GiBps)))
    for nm, v in (("MI300X_tier_local", 6.98115), ("MI300X_tier_nfs", 0.331084), ("MI300X_tier_cpu_limiter", 35.4159)):
        points.append(dict(name=nm, GiBps=v))
    json.dump({"cases": cases, "points": points, "thresholds": [1.2, 1.3, 1.38], "mi300x_busy_slowdown": mi_slow},
              open(a.out, "w"), indent=1)
    print("cases", len(cases), "points", len(points))


if __name__ == "__main__":
    main()
