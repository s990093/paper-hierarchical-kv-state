#!/usr/bin/env python3
"""next_a_w6_analyze.py — W6a／W6c 的分析（PREREG §5、偏離 1、偏離 3）。在任何 W6 資料出現之前寫好、commit。

  python code/icpe/next_a_w6_analyze.py w6a    # results/icpe_next/a/w6a/gpu_requests.csv → w6a_runs.csv、w6a_verdict.json
  python code/icpe/next_a_w6_analyze.py w6c    # results/icpe_next/a/w6c/calib_c1.csv → w6c_f.csv、w6c_report.json、calib_for_sim/

W6a（PREREG §5，判準事先寫死）：
  乾淨的 run＝每一列 contaminated＝0、請求數完整（同 p02_analyze.py）。每個 (cpu, policy, rep) 取乾淨的最後一個。
  指標＝回來請求（kind＝1）的平均 TTFT。每個 cpu、每個 rep r：gap_r＝1 − TTFT(O-PF, r) ÷ TTFT(HINT-PF, r)；
  share_r＝(TTFT(LRU,r) − TTFT(HINT-PF,r)) ÷ (TTFT(LRU,r) − TTFT(O-PF,r))（HINT-PF 拿到 O-PF 改善的比例）。
  判定：只要有任何一個 (cpu, r) 的 gap_r ≥ 5% → ANY_GE_5（照實寫）；兩個 cpu 各 6 個 rep 都齊、而且全部 < 5% → ALL_LT_5
  （才可以寫「打字提示拿走大部分」）；有 rep 不齊而且其他都 < 5% → INCOMPLETE（缺的寫 NOT_MEASURED）。
  另報：每個 cpu 的 gap、share 的中位數與最小–最大。

W6c（只報告；敏感度的輸入）：
  每塊（chunk_idx）f_chunk、GPU 閒的中位數（同 p3_fvar.c1_f）。對照 V1（results/icpe_h200/p01/calib/calib_c1.csv）：
  0–79 塊逐塊比值（W6c ÷ V1）；80–255 塊 W6c 實測 ÷ V1 的線性外插（p3_fvar.c1_f 的 polyfit）。
  calib_for_sim/＝給 p3_fvar.py --variant V1 --calib 用的目錄：calib_c1.csv＝W6c 實測（整份）、calib_c5.csv＝V1 那份原樣複製
  （GPU 忙倍數不變；PREREG 偏離 1「其他設定不變」）。整份換、不接 V1 的 0–79 塊（PREREG 偏離 3）。
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
OUT = os.path.join(REPO, "results", "icpe_next", "a")
V1 = os.path.join(REPO, "results", "icpe_h200", "p01", "calib")
GAP = 0.05
CPUS, POLS, NREP = (128, 256), ("LRU", "HINT-PF", "O-PF"), 6


def run_id():
    return os.environ.get("RUN_ID", "adhoc")


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def w6a():
    import pandas as pd
    d = os.environ.get("NEXT_A_W6A_DIR", os.path.join(OUT, "w6a"))      # 覆寫只給自檢用（輸出寫到那個目錄）
    g = pd.read_csv(os.path.join(d, "gpu_requests.csv"))
    per = g.groupby("run_id").agg(cpu_GiB=("cpu_GiB", "first"), policy=("policy", "first"), rep=("rep", "first"),
                                  n=("r", "size"), contaminated=("contaminated", "max")).reset_index()
    nreq = int(per.n.max())
    problems = [f"run {r.run_id}（{r.cpu_GiB} {r.policy} rep{r.rep}）不完整或污染：n={r.n}/{nreq}, contaminated={r.contaminated}"
                for _, r in per[(per.contaminated != 0) | (per.n != nreq)].iterrows()]
    clean = per[(per.contaminated == 0) & (per.n == nreq)].sort_values("run_id").groupby(["cpu_GiB", "policy", "rep"]).tail(1)
    ret = g[(g.kind == 1) & g.run_id.isin(clean.run_id)]
    m = ret.groupby("run_id").ttft.mean().rename("ttft_ret_mean")
    clean = clean.merge(m, left_on="run_id", right_index=True)
    rows, gaps = [], []
    for c in CPUS:
        for r in range(NREP):
            x = clean[(clean.cpu_GiB == c) & (clean.rep == r)].set_index("policy").ttft_ret_mean
            if not all(p in x for p in POLS):
                rows.append(dict(cpu_GiB=c, rep=r, gap="NOT_MEASURED", share="NOT_MEASURED"))
                continue
            gap = 1 - x["O-PF"] / x["HINT-PF"]
            den = x["LRU"] - x["O-PF"]
            share = (x["LRU"] - x["HINT-PF"]) / den if den > 0 else float("nan")
            gaps.append(gap)
            rows.append(dict(cpu_GiB=c, rep=r, ttft_LRU=x["LRU"], ttft_HINT_PF=x["HINT-PF"], ttft_O_PF=x["O-PF"], gap=gap, share=share,
                             runs=";".join(clean[(clean.cpu_GiB == c) & (clean.rep == r)].run_id)))
    R = pd.DataFrame(rows)
    full = len(gaps) == len(CPUS) * NREP
    verdict = "ANY_GE_5" if any(x >= GAP for x in gaps) else ("ALL_LT_5" if full else "INCOMPLETE")
    summ = {}
    for c in CPUS:
        q = R[(R.cpu_GiB == c) & (R.gap != "NOT_MEASURED")]
        if len(q):
            gp, sh = q.gap.astype(float), q.share.astype(float)
            summ[str(c)] = dict(n=len(q), gap_median=float(gp.median()), gap_min=float(gp.min()), gap_max=float(gp.max()),
                                share_median=float(sh.median()), share_min=float(sh.min()), share_max=float(sh.max()))
    R.insert(0, "ts", ts())
    R.insert(0, "run_id", run_id())
    R.to_csv(os.path.join(d, "w6a_runs.csv"), index=False)
    v = dict(run_id=run_id(), ts=ts(), prereg="docs/icpe_brief_20261030/next/PREREG.md §5 W6a", threshold=GAP, verdict=verdict,
             n_pairs=len(gaps), n_expected=len(CPUS) * NREP, by_cpu=summ, problems=problems, clean_runs=sorted(clean.run_id))
    json.dump(v, open(os.path.join(d, "w6a_verdict.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v[k] for k in ("verdict", "n_pairs", "by_cpu", "problems")}, ensure_ascii=False, indent=1))


def w6c():
    import numpy as np
    import pandas as pd
    from p3_fvar import c1_f
    d = os.environ.get("NEXT_A_W6C_DIR", os.path.join(OUT, "w6c"))      # 覆寫只給自檢用
    f6, s6, i6, r6 = c1_f(os.path.join(d, "calib_c1.csv"))
    f1, s1, i1, r1 = c1_f(os.path.join(V1, "calib_c1.csv"))
    rows = []
    for i, y in enumerate(f6):
        if i < len(f1):
            rows.append(dict(chunk_idx=i, f_w6c_ms=y, f_v1_ms=f1[i], v1_kind="measured", ratio=y / f1[i]))
        else:
            e = i1 + s1 * i
            rows.append(dict(chunk_idx=i, f_w6c_ms=y, f_v1_ms=e, v1_kind="extrapolated", ratio=y / e))
    F = pd.DataFrame(rows)
    F.insert(0, "ts", ts())
    F.insert(0, "run_id", run_id())
    F.to_csv(os.path.join(d, "w6c_f.csv"), index=False)
    me, ex = F[F.v1_kind == "measured"].ratio, F[F.v1_kind == "extrapolated"].ratio
    rep = dict(run_id=run_id(), ts=ts(), w6c_runs=r6, v1_runs=r1, n_chunks_w6c=len(f6), n_chunks_v1=len(f1),
               w6c_fit=dict(slope_ms=s6, icpt_ms=i6), v1_fit=dict(slope_ms=s1, icpt_ms=i1),
               measured_overlap_ratio=dict(median=float(me.median()), min=float(me.min()), max=float(me.max())),
               vs_v1_extrapolation_ratio=(dict(median=float(ex.median()), min=float(ex.min()), max=float(ex.max())) if len(ex) else "NOT_MEASURED"),
               sum_ms_0_255=dict(w6c=float(np.sum(f6)), v1_measured_plus_extrap=float(F.f_v1_ms.sum())))
    if True:                                   # PREREG 偏離 3：整份換
        cd = os.path.join(d, "calib_for_sim")
        os.makedirs(cd, exist_ok=True)
        shutil.copyfile(os.path.join(d, "calib_c1.csv"), os.path.join(cd, "calib_c1.csv"))
        shutil.copyfile(os.path.join(V1, "calib_c5.csv"), os.path.join(cd, "calib_c5.csv"))
        rep["calib_for_sim"] = dict(mode="replace", calib_c1="W6c（整份）", calib_c5="V1 原樣複製")
    json.dump(rep, open(os.path.join(d, "w6c_report.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "w6a":
        w6a()
    elif cmd == "w6c":
        w6c()
    else:
        raise SystemExit("用法：next_a_w6_analyze.py w6a | w6c")
