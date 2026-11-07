#!/usr/bin/env python3
"""next_a_w6b.py — W6b：Mooncake 對話「淘汰順序」（O-ORD）的 GPU 縮小重播（PREREG §5、偏離 1、偏離 3 第 3 點；
跑哪些、幾次、怎麼判＝要在跑之前另外登記的 run set）。

為什麼是新的一支、不改 code/icpe/x1_gpu.py：W6c／W6a 的 job（24443、24444）排隊中，開跑時會讀這個 worktree 的 x1_gpu.py、
code/p3x/x1_sim.py，那兩支在它們跑完之前不能動。而 x1_sim 的 Engine 本來就有 O-ORD 的全部做法（RIVALS["O-ORD"]＝
LRU、LRU2、ARC、S3FIFO、GDSF、PENS；COMPONENT_POLICIES["O-ORD"]＝O-BEL、O-BELS），x1_gpu.py 的重播（replay、GpuView）
也和做法無關，只有 gpu 子命令的 --policy 限定 POLICIES5。所以這支 import x1_gpu，照原樣呼叫它的 replay／cmd_sim／cmd_gpu，
只換做法清單。

  selftest  （head node，CPU）8 個做法 × CPU 128／256：重播迴圈（模型時間）≡ Engine.run，逐請求逐位元（同 x1_gpu selftest (i)）。
            只記 PASS／FAIL、請求數、程式的 sha256；不記、不印任何 TTFT（登記前不看模擬數字）。
  sim       S_orig（給 --times＝GPU 的逐請求 CSV 時算 S_real）→ <out>/sim_requests.csv（= x1_gpu.cmd_sim，做法＝W6B_POLICIES）
  gpu       一個 GPU run（= x1_gpu.cmd_gpu；--policy 只收 W6B_POLICIES）→ <out>/gpu_requests.csv
  analyze   P0-2 §6 的三條判準（比值、排序、headroom 正負號），每格取各 rep 的中位數 → w6b_cells.csv、w6b_pairs.csv、
            w6b_signs.csv、w6b_runs.csv、w6b_verdict.json

環境同 W6a（next_a_w6.sh 的 w6a 部分）：P3_CONST=h200 P3_TIER=h200cpu M7_TIER_PARAMS=results/icpe_h200/t1/tier_params_h200cpu.json
X1_TRACE_DIR=$TIARA_HOME/datasets/traces；不設 M7_ATTN（sdpa，和 T1G P02、W6a 相同）。
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from datetime import datetime
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import x1_gpu as X1G  # noqa: E402  （import 時把 code/p3x、code 放進 sys.path）
import x1_sim as XS  # noqa: E402

W6B_RIVALS = ["LRU", "LRU2", "ARC", "S3FIFO", "GDSF", "PENS"]     # ②：O-ORD 的固定對手
W6B_ORACLES = ["O-BEL", "O-BELS"]                                 # ④：O-ORD 的元件
W6B_POLICIES = W6B_RIVALS + W6B_ORACLES
assert W6B_RIVALS == list(XS.RIVALS["O-ORD"]), XS.RIVALS["O-ORD"]
assert W6B_ORACLES == list(XS.COMPONENT_POLICIES["O-ORD"]), XS.COMPONENT_POLICIES["O-ORD"]
assert all(p in XS.POLICIES for p in W6B_POLICIES)
_MAKE_ENGINE_X1G = X1G.make_engine


def make_engine_w6b(P, policy, cap, views, K, lead=None):
    """x1_gpu.make_engine 不傳 gdsf_view（POLICIES5 沒有 GDSF），GDSF 的 Engine 會在 __init__ 讀 None.slow_name 而失敗
    （run 20261107-115225-icpe-next-w6b-selftest）。照 x1_sim 自己的 sweep（_job：gv = views[0] if gdsf）補上 views[0]。
    GDSF 只用這個 view 的固定屬性（slow_name、d、f）算成本；GpuView 是 View 的子類別、同一組 VIEW 參數，所以模擬與 GPU 重播的決策相同。"""
    if XS.POLICIES[policy][0] == "gdsf":
        return XS.Engine(P, policy, cap, views, K, gdsf_view=views[0])
    return _MAKE_ENGINE_X1G(P, policy, cap, views, K, lead)


X1G.make_engine = make_engine_w6b          # x1_gpu.replay 在呼叫時才查 make_engine，所以 replay／cmd_sim／cmd_gpu 都會用這個

K_SCALE = 8                                                       # 同 P0-2／T1G P02／W6a
BAND = (0.9, 1.1)
RANK_MIN = 0.05
SIGN_MIN = 0.02
OUT_DEFAULT = os.path.join(X1G.REPO, "results", "icpe_next", "a", "w6b")


def run_id():
    return os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc"


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def code_shas():
    return {os.path.relpath(p, X1G.REPO): sha(p)[:16] for p in (X1G.__file__, XS.__file__, X1G.XT.__file__, __file__)}


# ================================================================== selftest（CPU）
def cmd_selftest(a):
    import csv
    K = XS.constants()
    P = X1G.load_P(a.k)
    res = []
    for cg in X1G.CPU_GIB:
        cap = X1G.cap_bytes(cg, a.k)
        for pol in W6B_POLICIES:
            v1 = XS.View(K, *X1G.VIEW, P["nmax"], len(P["full"]))
            X1G.make_engine(P, pol, cap, [v1], K).run()
            _, v2, rows = X1G.replay(P, pol, cap, K, model_view=True)
            same = (bool(np.array_equal(v1.ttft, v2.ttft)) and v1.slow.bytes_r == v2.slow.bytes_r
                    and v1.slow.bytes_w == v2.slow.bytes_w and v1.pf_bytes == v2.pf_bytes)
            res.append(dict(run_id=run_id(), ts=ts(), check=f"replay_eq_engine_run[{cg}][{pol}]", passed=bool(same), k=a.k,
                            cpu_GiB=cg, policy=pol, n=len(rows)))
            print(("PASS " if same else "FAIL ") + res[-1]["check"], f"n={len(rows)}", flush=True)
    shas = code_shas()
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "selftest.csv")
    new = not os.path.exists(p)
    with open(p, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0].keys()) + ["code_sha"])
        if new:
            w.writeheader()
        for r in res:
            w.writerow(dict(r, code_sha=json.dumps(shas, sort_keys=True)))
    print(json.dumps(shas, indent=1), flush=True)
    if not all(r["passed"] for r in res):
        sys.exit(1)


# ================================================================== sim／gpu：照原樣呼叫 x1_gpu
def cmd_sim(a):
    X1G.cmd_sim(SimpleNamespace(out=a.out, k=a.k, cpu=list(X1G.CPU_GIB), policies=list(W6B_POLICIES), times=a.times))


def cmd_gpu(a):
    X1G.cmd_gpu(SimpleNamespace(out=a.out, k=a.k, cpu=a.cpu, policy=a.policy, rep=a.rep, max_req=a.max_req, csv=a.csv))


# ================================================================== analyze
def cmd_analyze(a):
    import pandas as pd
    D = a.dir
    g = pd.read_csv(os.path.join(D, "gpu_requests.csv"))
    s = pd.read_csv(os.path.join(D, "sim_requests.csv"))
    problems = []
    so_all = s[s.ref == "S_orig"]
    nreq = so_all.groupby(["cpu_GiB", "policy"]).size()
    if nreq.nunique() != 1:
        problems.append(f"S_orig 各格的請求數不同：{nreq.to_dict()}")
    nreq = int(nreq.iloc[0])
    runs = []
    for rid, x in g.groupby("run_id"):
        runs.append(dict(run_id=rid, cpu_GiB=int(x.cpu_GiB.iloc[0]), policy=x.policy.iloc[0], rep=int(x.rep.iloc[0]),
                         n=len(x), contaminated=int(x.contaminated.max()), gpu_mean_ret=float(x[x.kind == 1].ttft.mean())))
    R = pd.DataFrame(runs)
    for _, r in R[(R.contaminated != 0) | (R.n != nreq)].iterrows():
        problems.append(f"run {r.run_id}（{r.cpu_GiB} {r.policy} rep{r.rep}）不完整或污染：n={r.n}/{nreq}, contaminated={r.contaminated}（不用）")
    # 2026-11-07（使用者：每個 run 前等、跑完查重疊）：只用 overlap.csv（t1g.py overlap）裡 overlap＝0 的 run
    ovl = pd.read_csv(os.path.join(D, "overlap.csv"), dtype=str)
    ok_ovl = set(ovl[ovl.overlap == "0"].run_id)
    for _, r in R[~R.run_id.isin(ok_ovl)].iterrows():
        problems.append(f"run {r.run_id}（{r.cpu_GiB} {r.policy} rep{r.rep}）在 overlap.csv 沒有 overlap＝0 的紀錄 → 不用")
    clean = R[(R.contaminated == 0) & (R.n == nreq) & R.run_id.isin(ok_ovl)].sort_values("run_id")
    dup = clean.groupby(["cpu_GiB", "policy", "rep"]).size()
    if (dup > 1).any():
        problems.append(f"同一格同一 rep 有多個乾淨的 run：{dup[dup > 1].to_dict()}（取最後一個）")
    clean = clean.groupby(["cpu_GiB", "policy", "rep"]).tail(1)
    reps = sorted(range(a.reps))
    so = so_all[so_all.kind == 1]
    sr = s[(s.ref == "S_real") & (s.kind == 1) & s.src_gpu_run.isin(clean.run_id)]
    cells, missing = [], []
    for cg in X1G.CPU_GIB:
        for pol in W6B_POLICIES:
            x = clean[(clean.cpu_GiB == cg) & (clean.policy == pol)].set_index("rep")
            have = [r for r in reps if r in x.index]
            if len(have) < len(reps):
                missing.append(dict(cpu_GiB=cg, policy=pol, missing_reps=[r for r in reps if r not in x.index]))
            so_m = float(so[(so.cpu_GiB == cg) & (so.policy == pol)].ttft.mean())
            if not have:
                cells.append(dict(run_id=run_id(), ts=ts(), cpu_GiB=cg, policy=pol, n_reps=0, gpu_mean="NOT_MEASURED", s_orig_mean=so_m))
                continue
            gm = x.loc[have, "gpu_mean_ret"]
            srm = [float(sr[sr.src_gpu_run == x.loc[r, "run_id"]].ttft.mean()) for r in have
                   if (sr.src_gpu_run == x.loc[r, "run_id"]).any()]
            med = float(gm.median())
            cells.append(dict(run_id=run_id(), ts=ts(), cpu_GiB=cg, policy=pol, n_reps=len(have), gpu_mean=med,
                              gpu_min=float(gm.min()), gpu_max=float(gm.max()), s_orig_mean=so_m, ratio=med / so_m,
                              in_band=bool(BAND[0] <= med / so_m <= BAND[1]),
                              ratio_min=float(gm.min()) / so_m, ratio_max=float(gm.max()) / so_m,
                              s_real_median=float(np.median(srm)) if srm else None,
                              ratio_real=(med / float(np.median(srm))) if srm else None,
                              gpu_runs=";".join(x.loc[have, "run_id"])))
    C = pd.DataFrame(cells)
    M = C[C.n_reps > 0].copy()
    M["gpu_mean"] = M.gpu_mean.astype(float)
    full = (not missing) and len(M) == len(X1G.CPU_GIB) * len(W6B_POLICIES)
    # 2. 排序：每個容量，S_orig 相對差 ≥5% 的每一對，GPU（各 rep 中位數）的先後相同
    pairs = []
    for cg in X1G.CPU_GIB:
        c = M[M.cpu_GiB == cg].set_index("policy")
        for p, q in itertools.combinations([x for x in W6B_POLICIES if x in c.index], 2):
            ds = (c.loc[p, "s_orig_mean"] - c.loc[q, "s_orig_mean"]) / min(c.loc[p, "s_orig_mean"], c.loc[q, "s_orig_mean"])
            dg = c.loc[p, "gpu_mean"] - c.loc[q, "gpu_mean"]
            judged = abs(ds) >= RANK_MIN
            pairs.append(dict(run_id=run_id(), ts=ts(), cpu_GiB=cg, a=p, b=q, sim_rel_diff=ds, gpu_diff_s=dg, judged=bool(judged),
                              same_order=bool(np.sign(ds) == np.sign(dg)) if judged else None))
    Pp = pd.DataFrame(pairs)
    # 3. headroom 正負號：O-ORD＝min(O-BEL, O-BELS) 對 min(6 個對手)（判）；O-BEL、O-BELS 各自對 min(6 個對手)（只報告）
    signs = []
    for cg in X1G.CPU_GIB:
        c = M[M.cpu_GiB == cg].set_index("policy")
        if not all(p in c.index for p in W6B_POLICIES):
            continue
        rs, rg = min(c.loc[r, "s_orig_mean"] for r in W6B_RIVALS), min(c.loc[r, "gpu_mean"] for r in W6B_RIVALS)
        for comp, orcs, judged_row in (("O-ORD", W6B_ORACLES, True), ("O-BEL", ["O-BEL"], False), ("O-BELS", ["O-BELS"], False)):
            os_, og = min(c.loc[o, "s_orig_mean"] for o in orcs), min(c.loc[o, "gpu_mean"] for o in orcs)
            hs, hg = (rs - os_) / rs, (rg - og) / rg
            judged = judged_row and abs(hs) >= SIGN_MIN
            signs.append(dict(run_id=run_id(), ts=ts(), cpu_GiB=cg, component=comp, rivals="/".join(W6B_RIVALS),
                              best_rival_sim=min(W6B_RIVALS, key=lambda r: c.loc[r, "s_orig_mean"]),
                              best_rival_gpu=min(W6B_RIVALS, key=lambda r: c.loc[r, "gpu_mean"]),
                              headroom_sim=hs, headroom_gpu=hg, report_only=not judged_row, judged=bool(judged),
                              same_sign=bool(np.sign(hs) == np.sign(hg)) if judged else None))
    S = pd.DataFrame(signs)
    c1 = bool(full and M.in_band.all())
    c2 = bool(full and len(Pp) and Pp[Pp.judged].same_order.all())
    c3 = bool(full and len(S) and S[S.judged].same_sign.all())
    if not full:
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS" if (c1 and c2 and c3) else "FAIL"
    out = dict(run_id=run_id(), ts=ts(), prereg="PREREG §5 W6b、偏離 3 第 3 點、W6b 的 run set 登記", k=int(g.k.iloc[0]) if len(g) else None,
               reps_expected=len(reps), verdict=verdict, criteria=dict(ratio_in_band=c1, ranking=c2, headroom_sign=c3),
               n_clean_runs=len(clean), missing=missing, problems=problems,
               out_of_band=M[~M.in_band][["cpu_GiB", "policy", "ratio"]].to_dict("records"),
               ranking_mismatch=Pp[Pp.judged & (Pp.same_order == False)].to_dict("records") if len(Pp) else [],  # noqa: E712
               sign_mismatch=S[S.judged & (S.same_sign == False)].to_dict("records") if len(S) else [],  # noqa: E712
               not_judged_pairs=Pp[~Pp.judged][["cpu_GiB", "a", "b", "sim_rel_diff"]].to_dict("records") if len(Pp) else [],
               not_judged_signs=S[~S.judged][["cpu_GiB", "component", "headroom_sim"]].to_dict("records") if len(S) else [],
               code_sha=code_shas())
    R.insert(0, "ts", ts())
    R.insert(0, "an_run_id", run_id())
    R.to_csv(os.path.join(D, "w6b_runs.csv"), index=False)
    C.to_csv(os.path.join(D, "w6b_cells.csv"), index=False)
    Pp.to_csv(os.path.join(D, "w6b_pairs.csv"), index=False)
    S.to_csv(os.path.join(D, "w6b_signs.csv"), index=False)
    json.dump(out, open(os.path.join(D, "w6b_verdict.json"), "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: out[k] for k in ("verdict", "criteria", "n_clean_runs", "missing", "problems")}, ensure_ascii=False, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("selftest"); s.add_argument("--out", default=OUT_DEFAULT); s.add_argument("--k", type=int, default=K_SCALE)
    s = sp.add_parser("sim"); s.add_argument("--out", default=OUT_DEFAULT); s.add_argument("--k", type=int, default=K_SCALE)
    s.add_argument("--times", help="GPU 的逐請求 CSV（給了就算 S_real）")
    s = sp.add_parser("gpu"); s.add_argument("--out", default=OUT_DEFAULT); s.add_argument("--k", type=int, default=K_SCALE)
    s.add_argument("--cpu", type=int, required=True, choices=X1G.CPU_GIB); s.add_argument("--policy", required=True, choices=W6B_POLICIES)
    s.add_argument("--rep", type=int, default=0); s.add_argument("--max-req", type=int); s.add_argument("--csv", default="gpu_requests.csv")
    s = sp.add_parser("analyze"); s.add_argument("--dir", default=OUT_DEFAULT); s.add_argument("--reps", type=int, required=True)
    a = ap.parse_args()
    {"selftest": cmd_selftest, "sim": cmd_sim, "gpu": cmd_gpu, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
