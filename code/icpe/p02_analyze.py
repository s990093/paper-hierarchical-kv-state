"""p02_analyze.py — ICPE P0-2（docs/h200/icpe/P0-2_x1_gpu_validation.md §6）的比較與判定。

  python code/icpe/p02_analyze.py --dir results/icpe_h200/p02

讀 gpu_requests.csv（12 個 GPU run）、sim_requests.csv（S_orig；S_real 要先跑 x1_gpu.py sim --times gpu_requests.csv）。
判準（對 S_orig，三條都成立＝「模擬可信」）：
  1. 比值：10 格（容量 × 做法，rep 0）回來請求 TTFT 平均，實測 ÷ S_orig ∈ [0.9, 1.1]
  2. 排序：每個容量，S_orig 平均差 ≥5% 的每一對做法，GPU 上的先後相同
  3. headroom 正負號：O-BEL 對 min(LRU, FREQ2Q)；O-PF 對 LRU；HINT-PF 對 LRU；O-PF 對 HINT-PF（S_orig |headroom| < 2% 的列出、不判）
→ p02_verdict.json、p02_cells.csv、p02_pairs.csv、p02_signs.csv
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

POL = ["LRU", "FREQ2Q", "O-BEL", "HINT-PF", "O-PF"]
SIGNS = [("O-BEL", ("LRU", "FREQ2Q")), ("O-PF", ("LRU",)), ("HINT-PF", ("LRU",)), ("O-PF", ("HINT-PF",))]


def run_id():
    return os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc"


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    D = a.dir
    g = pd.read_csv(os.path.join(D, "gpu_requests.csv"))
    s = pd.read_csv(os.path.join(D, "sim_requests.csv"))
    problems = []
    # 每個 GPU run 要完整、沒有污染（污染的 run 照 §7 重跑；這裡只收乾淨的）
    runs = []
    for rid, x in g.groupby("run_id"):
        runs.append(dict(run_id=rid, cpu_GiB=int(x.cpu_GiB.iloc[0]), policy=x.policy.iloc[0], rep=int(x.rep.iloc[0]),
                         n=len(x), contaminated=int(x.contaminated.max())))
    R = pd.DataFrame(runs)
    nreq = int(s[s.ref == "S_orig"].groupby(["cpu_GiB", "policy"]).size().iloc[0])
    clean = R[(R.contaminated == 0) & (R.n == nreq)]
    for _, r in R[(R.contaminated != 0) | (R.n != nreq)].iterrows():
        problems.append(f"run {r.run_id}（{r.cpu_GiB} {r.policy} rep{r.rep}）不完整或污染：n={r.n}/{nreq}, contaminated={r.contaminated}")
    dup = clean.groupby(["cpu_GiB", "policy", "rep"]).size()
    if (dup > 1).any():
        problems.append(f"同一格有多個乾淨的 run：{dup[dup > 1].to_dict()}（取最後一個）")
    clean = clean.sort_values("run_id").groupby(["cpu_GiB", "policy", "rep"]).tail(1)
    g = g[g.run_id.isin(clean.run_id)]
    ret_g = g[g.kind == 1]
    so = s[(s.ref == "S_orig") & (s.kind == 1)]
    sr = s[(s.ref == "S_real") & (s.kind == 1)]
    sr = sr[sr.src_gpu_run.isin(clean.run_id)]
    cells = []
    for (cg, pol), x in ret_g[ret_g.rep == 0].groupby(["cpu_GiB", "policy"]):
        so_m = float(so[(so.cpu_GiB == cg) & (so.policy == pol)].ttft.mean())
        rid = x.run_id.iloc[0]
        srr = sr[sr.src_gpu_run == rid]
        m = x.merge(so[(so.cpu_GiB == cg) & (so.policy == pol)][["r", "ttft"]], on="r", suffixes=("", "_sim"))
        pr = m.ttft / m.ttft_sim
        gm = float(x.ttft.mean())
        cells.append(dict(run_id=run_id(), ts=ts(), cpu_GiB=cg, policy=pol, gpu_run=rid, n_ret=len(x), gpu_mean=gm, s_orig_mean=so_m,
                          ratio=gm / so_m, in_band=bool(0.9 <= gm / so_m <= 1.1),
                          s_real_mean=float(srr.ttft.mean()) if len(srr) else None,
                          ratio_real=(gm / float(srr.ttft.mean())) if len(srr) else None,
                          per_req_ratio_p10=float(pr.quantile(0.1)), per_req_ratio_med=float(pr.median()), per_req_ratio_p90=float(pr.quantile(0.9)),
                          gpu_median=float(x.ttft.median()), s_orig_median=float(so[(so.cpu_GiB == cg) & (so.policy == pol)].ttft.median()),
                          queued_frac_all=float((g[g.run_id == rid].queue_s > 1e-3).mean()),
                          queue_s_mean_all=float(g[g.run_id == rid].queue_s.mean()),
                          d2h_s_sum=float(g[g.run_id == rid].d2h_s.sum())))
    C = pd.DataFrame(cells)
    if len(C) != 10:
        problems.append(f"rep 0 的格子只有 {len(C)} 個（要 10）")
    # 2. 排序
    pairs = []
    for cg in sorted(C.cpu_GiB.unique()):
        c = C[C.cpu_GiB == cg].set_index("policy")
        for p, q in itertools.combinations([x for x in POL if x in c.index], 2):
            ds = (c.loc[p, "s_orig_mean"] - c.loc[q, "s_orig_mean"]) / min(c.loc[p, "s_orig_mean"], c.loc[q, "s_orig_mean"])
            dg = c.loc[p, "gpu_mean"] - c.loc[q, "gpu_mean"]
            judged = abs(ds) >= 0.05
            pairs.append(dict(run_id=run_id(), ts=ts(), cpu_GiB=cg, a=p, b=q, sim_rel_diff=ds, gpu_diff_s=dg, judged=bool(judged),
                              same_order=bool(np.sign(ds) == np.sign(dg)) if judged else None))
    Pp = pd.DataFrame(pairs)
    # 3. headroom 正負號
    signs = []
    for cg in sorted(C.cpu_GiB.unique()):
        c = C[C.cpu_GiB == cg].set_index("policy")
        for orc, rivals in SIGNS:
            if orc not in c.index or not all(r in c.index for r in rivals):
                continue
            rs, rg = min(c.loc[r, "s_orig_mean"] for r in rivals), min(c.loc[r, "gpu_mean"] for r in rivals)
            hs, hg = (rs - c.loc[orc, "s_orig_mean"]) / rs, (rg - c.loc[orc, "gpu_mean"]) / rg
            judged = abs(hs) >= 0.02
            signs.append(dict(run_id=run_id(), ts=ts(), cpu_GiB=cg, component=orc, rivals="/".join(rivals), headroom_sim=hs, headroom_gpu=hg,
                              judged=bool(judged), same_sign=bool(np.sign(hs) == np.sign(hg)) if judged else None))
    S = pd.DataFrame(signs)
    # 雜訊：LRU 兩次
    noise = []
    for cg in sorted(ret_g.cpu_GiB.unique()):
        x = ret_g[(ret_g.cpu_GiB == cg) & (ret_g.policy == "LRU")].groupby("rep").ttft.mean()
        if len(x) == 2:
            noise.append(dict(cpu_GiB=int(cg), rep0=float(x[0]), rep1=float(x[1]), rel_diff=float(abs(x[0] - x[1]) / x.mean())))
    c1 = bool(len(C) == 10 and C.in_band.all())
    c2 = bool(Pp[Pp.judged].same_order.all()) if len(Pp) else False
    c3 = bool(S[S.judged].same_sign.all()) if len(S) else False
    out = dict(run_id=run_id(), ts=ts(), k=int(g.k.iloc[0]) if len(g) else None, criteria=dict(ratio_in_band=c1, ranking=c2, headroom_sign=c3),
               passed=bool(c1 and c2 and c3 and not problems), n_clean_runs=len(clean), problems=problems,
               out_of_band=C[~C.in_band][["cpu_GiB", "policy", "ratio"]].to_dict("records") if len(C) else [],
               ranking_mismatch=Pp[Pp.judged & (Pp.same_order == False)].to_dict("records") if len(Pp) else [],  # noqa: E712
               sign_mismatch=S[S.judged & (S.same_sign == False)].to_dict("records") if len(S) else [],  # noqa: E712
               not_judged_pairs=Pp[~Pp.judged][["cpu_GiB", "a", "b", "sim_rel_diff"]].to_dict("records") if len(Pp) else [],
               not_judged_signs=S[~S.judged][["cpu_GiB", "component", "headroom_sim"]].to_dict("records") if len(S) else [],
               lru_noise=noise)
    C.to_csv(os.path.join(D, "p02_cells.csv"), index=False)
    Pp.to_csv(os.path.join(D, "p02_pairs.csv"), index=False)
    S.to_csv(os.path.join(D, "p02_signs.csv"), index=False)
    json.dump(out, open(os.path.join(D, "p02_verdict.json"), "w"), ensure_ascii=False, indent=1, default=str)
    pd.set_option("display.width", 220)
    print(C[["cpu_GiB", "policy", "n_ret", "gpu_mean", "s_orig_mean", "ratio", "s_real_mean", "ratio_real", "per_req_ratio_med"]].round(4).to_string(index=False))
    print(S[["cpu_GiB", "component", "rivals", "headroom_sim", "headroom_gpu", "judged", "same_sign"]].round(4).to_string(index=False))
    print(json.dumps({k: out[k] for k in ("criteria", "passed", "problems", "lru_noise")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
