"""explore_queue2.py — 二階排隊（qev，PREREG 偏離 19）對上限的影響：一個變體的分析。
用法：python -I explore_queue2.py --qsweep <qev 的 x1_sweep.csv> --ref <版本 C 的 verC_headroom_components.csv>
                                  --noq-sweep <偏離 10 的 x1_sweep.csv(.xz)> [--qpost-rows m=<一階新規則的 queue_rows.csv> orig=<偏離 11 的 queue_rows.csv>] --out <目錄>
  1. 結構檢查（和 --noq-sweep 比）：每個組合剛好一列、組合的集合相同、n_req／n_ret 相同；queue＝closed_event。
     （二階排隊會改快取狀態，所以快取相關的欄位本來就可以不同；不檢查 mean_ret 的大小關係。）
  2. analyze_v8（同版本 C 的算法）算二階的上限表，和版本 C 的同一格比。
  3. 負載＝二階裡那一格 (LRU, LRU) 的平均等待（qwait_mean）：輕 < 2 s、中 2–20 s、重 ≥ 20 s；重的只報告（偏離 19、同偏離 11）。
  4. 有 --qpost-rows 時，再和一階的同一格比（每個 tag 一組欄：m＝一階、到達不早於上一個完成（qpost=2，比較 2 的主對照）；
     orig＝偏離 11 B 的一階（qpost=1）），並列一階的負載等級。
輸出 queue2_rows.csv（每列有 run_id、ts、host）；印摘要；最後一行 QUEUE2_RESULT。"""
import argparse, os, socket, subprocess, sys
from datetime import datetime

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--qsweep", nargs="+", required=True)
ap.add_argument("--ref", required=True)
ap.add_argument("--noq-sweep", nargs="+", required=True)
ap.add_argument("--qpost-rows", nargs="*", default=[])
ap.add_argument("--out", required=True)
a = ap.parse_args()
REPO = os.environ["REPO"]
os.makedirs(a.out, exist_ok=True)
Q = pd.concat([pd.read_csv(p) for p in a.qsweep], ignore_index=True)
assert (Q.queue == "closed_event").all()
K = ["trace", "rule", "cpu_GiB", "policy", "slow", "gpu", "restore", "gpu_GiB", "gpu_write", "gpu_policy", "pf_target", "slow_cap_frac", "slow_policy"]
K = [k for k in K if k in Q.columns]
N = pd.concat([pd.read_csv(p) for p in a.noq_sweep], ignore_index=True)
for X in (Q, N):
    X["cpu_GiB"] = X.cpu_GiB.astype(float)
N = N[N.cpu_GiB.isin(Q.cpu_GiB.unique())]
ok = True
dupq, dupn = int(Q.duplicated(K).sum()), int(N.duplicated(K).sum())
M = Q.merge(N, on=K, suffixes=("", "_n"))
same_n = bool((M.n_req == M.n_req_n).all() and (M.n_ret == M.n_ret_n).all())
ok &= dupq == 0 and dupn == 0 and len(M) == len(Q) == len(N) and same_n
print(f"結構：二階 {len(Q)} 列、版本 C 同容量 {len(N)} 列、對上 {len(M)}；重複 {dupq}／{dupn}；n_req／n_ret 相同：{same_n}")
out = a.out
p = os.path.join(out, "qsweep.csv")
Q.drop(columns=["queue", "qwait_mean", "qwait_p90", "qwait_frac", "qsvc_mean"]).to_csv(p, index=False)
cp = subprocess.run([sys.executable, "-I", os.path.join(REPO, "code/icpe/rux_x1fast/v8_features/analyze_v8.py"), "--sweep", p,
                     "--out-dir", out, "--out-prefix", "queue2"], capture_output=True, text=True, env=dict(os.environ))
assert cp.returncode == 0, cp.stderr[-3000:]
H = pd.read_csv(os.path.join(out, "queue2_headroom_components.csv"))
V = pd.read_csv(a.ref)
C = ["trace", "rule", "cpu_GiB", "slow", "gpu", "restore", "component"]
for X in (H, V):
    X["cpu_GiB"] = X.cpu_GiB.astype(float)
V = V[V.trace.isin(H.trace.unique()) & V.cpu_GiB.isin(H.cpu_GiB.unique())]
J = V.merge(H, on=C, suffixes=("_noq", "_qev"))
ok &= len(J) == len(V) == len(H)
CELL = ["trace", "rule", "cpu_GiB", "slow", "gpu", "restore"]
load = Q[(Q.policy == "LRU") & (Q.gpu_policy == "LRU")][CELL + ["qwait_mean", "qsvc_mean"]].copy()
J = J.merge(load, on=CELL, how="left")
ok &= bool(J.qwait_mean.notna().all())
J["load"] = pd.cut(J.qwait_mean, [-1, 2, 20, np.inf], labels=["輕", "中", "重"])
J["d_hr_pt"] = (J.headroom_qev - J.headroom_noq) * 100
cols = CELL + ["component", "qwait_mean", "qsvc_mean", "load", "T_rival_noq", "T_rival_qev", "T_oracle_noq", "T_oracle_qev",
               "headroom_noq", "headroom_qev", "d_hr_pt", "best_rival_noq", "best_rival_qev", "best_rival_gpu_policy_noq", "best_rival_gpu_policy_qev",
               "oracle_policy_noq", "oracle_policy_qev", "oracle_gpu_policy_noq", "oracle_gpu_policy_qev", "write_saturated_qev"]
TAGS = []
for spec in a.qpost_rows:
    tg, pth = spec.split("=", 1)
    TAGS.append(tg)
    F = pd.read_csv(pth)
    F["cpu_GiB"] = F.cpu_GiB.astype(float)
    F = F[CELL + ["component", "qwait_mean", "load", "headroom_q"]].rename(
        columns={"qwait_mean": f"qwait_mean_qpost_{tg}", "load": f"load_qpost_{tg}", "headroom_q": f"headroom_qpost_{tg}"})
    J = J.merge(F, on=CELL + ["component"], how="left")
    ok &= bool(J[f"headroom_qpost_{tg}"].notna().all())
    J[f"d_qev_vs_qpost_{tg}_pt"] = (J.headroom_qev - J[f"headroom_qpost_{tg}"]) * 100
    cols += [f"qwait_mean_qpost_{tg}", f"load_qpost_{tg}", f"headroom_qpost_{tg}", f"d_qev_vs_qpost_{tg}_pt"]
meta = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"), host=socket.gethostname())
X = J[cols].copy()
for i, (k, v) in enumerate(meta.items()):
    X.insert(i, k, v)
X.to_csv(os.path.join(out, "queue2_rows.csv"), index=False)
pd.set_option("display.width", 250)
cake = J[J.restore == "cake"]


def agg(g):
    d = dict(n=len(g), n15_noq=int((g.headroom_noq >= 0.15).sum()), n15_qev=int((g.headroom_qev >= 0.15).sum()),
             up=int(((g.headroom_noq < 0.15) & (g.headroom_qev >= 0.15)).sum()), down=int(((g.headroom_noq >= 0.15) & (g.headroom_qev < 0.15)).sum()),
             med_d_pt=float(g.d_hr_pt.median()))
    for tg in TAGS:
        d.update({f"n15_qpost_{tg}": int((g[f"headroom_qpost_{tg}"] >= 0.15).sum()), f"med_d_vs_{tg}_pt": float(g[f"d_qev_vs_qpost_{tg}_pt"].median())})
    return pd.Series(d)


print(cake.groupby(["load", "component"], observed=True).apply(agg, include_groups=False).round(3).to_string())
print(cake.groupby("load", observed=True).qwait_mean.agg(["size", "min", "median", "max"]).round(2).to_string())
for tg in TAGS:
    print(f"負載等級：二階（列）× 一階 {tg}（欄），列數（每格 6 個元件算 6 列）")
    print(pd.crosstab(cake.load, cake[f"load_qpost_{tg}"]).to_string())
print("QUEUE2_RESULT", "E2E_OK" if ok else "E2E_FAIL", f"rows={len(J)}")
sys.exit(0 if ok else 1)
