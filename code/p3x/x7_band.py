"""x7_band.py — X7：用兩條線模型（只用 CPU）算 Cake 的會合點與「有用的頻寬帶」，3090 與 MI300X 用同一個模型並列。

模型（和 m7_restore_harness 的兩條線同一個行為）：重算線從 block 0 往後、載入線從 block n−1 往前，
哪條線先空下來就領下一個 block（Restorer._g5_decide 的 G5b 模擬就是這個寫法）：
    tr=tl=0; p=0; q=n−1; while p≤q: if tr≤tl: tr+=f[p]; p+=1  else: tl+=ℓ; q−=1
    Cake＝max(tr, tl)；b＝p（重算的 block 數）
另報「最佳切點」min_m max(Σ_{i<m} f(i), (n−m)·ℓ)（m7_restore_harness.write_boundary 的算式，第一階段 C3 用它預測會合點）。
只算＝Σf；只載＝n·ℓ；ℓ＝block 位元組 ÷ 頻寬。加速比＝min(只算, 只載) ÷ Cake。
主要用 TTFT（還原＋新一輪 256 token，和 main.tex Fig. cake_band／A0 相同），另報只看還原時間的加速比。
帶＝加速比 ≥ 門檻的頻寬區間（對數格點 0.005–100 GiB/s，每十倍 200 點；端點線性內插）。block 是整數，加速比在門檻附近會鋸齒、
切成幾段；判定用「外包絡」（曾經 ≥ 門檻的最低到最高頻寬），另列含峰值的那一段（main）與段數。

  python code/p3x/x7_band.py --spec spec.json --out-dir results/p3_explore/x7
spec.json：{"cases": [{"platform":"3090","model":"qwen","gpu_state":"idle","f_ms":[...64 個...],
             "chunk_bytes":29360128,"t_new_ms":..., "src":"<run_id 或檔案>"}, ...],
            "points": [{"name":"cpu_real","GiBps":6.9}, ...],   # 另外在這些頻寬上列出模型值
            "thresholds":[1.2,1.3,1.38]}
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime


def greedy(f, ell):
    tr = tl = 0.0
    p, q = 0, len(f) - 1
    while p <= q:
        if tr <= tl:
            tr += f[p]
            p += 1
        else:
            tl += ell
            q -= 1
    return max(tr, tl), p


def optimal(f, ell):
    n = len(f)
    best, bm, R = None, 0, 0.0
    for m in range(n + 1):
        if m > 0:
            R += f[m - 1]
        T = max(R, (n - m) * ell)
        if best is None or T < best:
            best, bm = T, m
    return best, bm


def evaluate(case, gibps):
    f = [x / 1e3 for x in case["f_ms"]]
    n = len(f)
    ell = case["chunk_bytes"] / (gibps * (1 << 30))
    tc, b = greedy(f, ell)
    to, bo = optimal(f, ell)
    t_comp, t_load = sum(f), n * ell
    tn = case.get("t_new_ms")
    tn = None if tn is None else tn / 1e3                    # None＝新一輪時間沒量到 → TTFT 加速比退回只看還原（列 ttft_fallback）
    return dict(n_blocks=n, ell_ms=ell * 1e3, t_compute_s=t_comp, t_load_s=t_load, t_cake_s=tc, b_greedy=b,
                t_cake_opt_s=to, b_opt=bo, speedup_restore=min(t_comp, t_load) / tc,
                speedup_ttft=(min(t_comp, t_load) + tn) / (tc + tn) if tn is not None else min(t_comp, t_load) / tc,
                ttft_fallback="" if tn is not None else "t_new NOT_MEASURED: speedup_ttft=restore-only",
                cake_vs_load_restore=t_load / tc)


def bands(case, thr, key):
    xs = [10 ** (math.log10(0.005) + k / 200) for k in range(int(200 * (math.log10(100) - math.log10(0.005))) + 1)]
    ys = [evaluate(case, x)[key] for x in xs]
    out, cur = [], None
    for i, (x, y) in enumerate(zip(xs, ys)):
        inside = y >= thr
        if inside and cur is None:
            if i == 0:
                lo = x
            else:                                     # 內插（log x）
                x0, y0 = xs[i - 1], ys[i - 1]
                lo = 10 ** (math.log10(x0) + (thr - y0) / (y - y0) * (math.log10(x) - math.log10(x0)))
            cur = [lo, None, y, x]
        if inside and cur is not None and y > cur[2]:
            cur[2], cur[3] = y, x
        if not inside and cur is not None:
            x0, y0 = xs[i - 1], ys[i - 1]
            hi = 10 ** (math.log10(x0) + (y0 - thr) / (y0 - y) * (math.log10(x) - math.log10(x0)))
            cur[1] = hi
            out.append(cur)
            cur = None
    if cur is not None:
        cur[1] = xs[-1]
        out.append(cur)
    return [{"lo_GiBps": round(c[0], 4), "hi_GiBps": round(c[1], 4), "peak_speedup": round(c[2], 3),
             "peak_at_GiBps": round(c[3], 4)} for c in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    rid = os.environ.get("RUN_ID", "NO_RUN_ID")
    t = datetime.now().astimezone().isoformat(timespec="seconds")
    os.makedirs(a.out_dir, exist_ok=True)
    rows, summ = [], []
    for c in spec["cases"]:
        for p in spec["points"]:
            if p.get("model") and p["model"] != c["model"]:
                continue
            e = evaluate(c, p["GiBps"])
            rows.append(dict(run_id=rid, ts=t, platform=c["platform"], model=c["model"], gpu_state=c["gpu_state"],
                             point=p["name"], GiBps=p["GiBps"], **{k: round(v, 5) if isinstance(v, float) else v
                                                                   for k, v in e.items()},
                             f_src=c["src"]))
        for key in ("speedup_ttft", "speedup_restore"):
            for thr in spec["thresholds"]:
                bs = bands(c, thr, key)
                main = max(bs, key=lambda b_: b_["peak_speedup"]) if bs else None
                summ.append(dict(run_id=rid, ts=t, platform=c["platform"], model=c["model"], gpu_state=c["gpu_state"],
                                 metric=key, threshold=thr,
                                 envelope_lo_GiBps=min(b_["lo_GiBps"] for b_ in bs) if bs else "",
                                 envelope_hi_GiBps=max(b_["hi_GiBps"] for b_ in bs) if bs else "",
                                 main_lo_GiBps=main["lo_GiBps"] if main else "", main_hi_GiBps=main["hi_GiBps"] if main else "",
                                 peak_speedup=main["peak_speedup"] if main else "", peak_at_GiBps=main["peak_at_GiBps"] if main else "",
                                 n_fragments=len(bs), bands=json.dumps(bs), f_src=c["src"]))
    with open(os.path.join(a.out_dir, "band_points.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out_dir, "band_edges.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    for s in summ:
        if s["threshold"] == 1.3:
            print(s["platform"], s["model"], s["gpu_state"], s["metric"], "envelope", s["envelope_lo_GiBps"], s["envelope_hi_GiBps"],
                  "main", s["main_lo_GiBps"], s["main_hi_GiBps"], "peak", s["peak_speedup"], "@", s["peak_at_GiBps"])


if __name__ == "__main__":
    main()
