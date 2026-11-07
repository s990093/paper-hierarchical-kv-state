"""探索（不是正式結果）：排隊的一階近似（qpost，SEMANTICS §2b）對上限的影響。
用法：python -I explore_queue.py --qsweep <qpost 的 x1_sweep.csv ...> --ref <版本 B 的 *_headroom_components.csv> --out <目錄>
                                 [--noq-sweep <同樣的格、沒排隊的 x1_sweep.csv>]（端到端檢查）
  1. （有 --noq-sweep 時）同一個組合：只有 TTFT 衍生的欄位可以不同，其他欄位逐格相同；有排隊的 mean_ret 不小於沒排隊的。
  2. analyze_v8（同版本 B 的算法）算有排隊的上限，和版本 B 的同一格比。
  3. 負載＝那一格 (LRU, LRU) 的平均等待（qwait_mean）：輕 < 2 s、中 2–20 s、重 ≥ 20 s。重的格重排很多，一階近似不可靠（V8_FEATURES §4c）。
輸出 queue_rows.csv（每列有 run_id、ts、host）；印摘要；最後一行 QUEUE_RESULT。"""
import argparse, os, socket, subprocess, sys
from datetime import datetime

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--qsweep", nargs="+", required=True)
ap.add_argument("--ref", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--noq-sweep", nargs="*", default=[])
a = ap.parse_args()
REPO = os.environ["REPO"]
os.makedirs(a.out, exist_ok=True)
Q = pd.concat([pd.read_csv(p) for p in a.qsweep], ignore_index=True)
assert (Q.queue == "closed_post").all()
K = ["trace", "rule", "cpu_GiB", "policy", "slow", "gpu", "restore", "gpu_GiB", "gpu_write", "gpu_policy", "pf_target", "slow_cap_frac", "slow_policy"]
K = [k for k in K if k in Q.columns]
ok_e2e = True
if a.noq_sweep:
    N = pd.concat([pd.read_csv(p) for p in a.noq_sweep], ignore_index=True)
    M = Q.merge(N, on=K, suffixes=("", "_n"))
    assert len(M) == len(Q), f"有排隊的 {len(Q)} 個組合只對上 {len(M)} 個（沒排隊的 sweep 少了格）"
    skip = {"run_id", "ts", "elapsed_s", "code_sha", "host", "queue", "qwait_mean", "qwait_p90", "qwait_frac"}
    def same_col(x, y):
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            return bool(((x == y) | (x.isna() & y.isna())).all())          # NaN＝NaN 算相同
        return bool((x.astype(str) == y.astype(str)).all())
    diffcols = sorted(c for c in Q.columns if c not in skip and c not in K and c + "_n" in M.columns and not same_col(M[c], M[c + "_n"]))
    ge = bool((M.mean_ret >= M.mean_ret_n - 1e-12).all())
    print(f"端到端：{len(M)} 個組合對上；不同的欄＝{diffcols}；有排隊的 mean_ret ≥ 沒排隊：{ge}")
    ttft_cols = {"n_ret_user", "sum_ret_user", "mean_ret_user", "sum_ret", "mean_ret", "median_ret", "p90_ret", "total_all", "mean_all"}
    ok_e2e = ge and set(diffcols) <= ttft_cols
p = os.path.join(out := a.out, "qsweep.csv")
Q.drop(columns=["queue", "qwait_mean", "qwait_p90", "qwait_frac"]).to_csv(p, index=False)
cp = subprocess.run([sys.executable, "-I", os.path.join(REPO, "code/icpe/rux_x1fast/v8_features/analyze_v8.py"), "--sweep", p,
                     "--out-dir", out, "--out-prefix", "queue"], capture_output=True, text=True, env=dict(os.environ))
assert cp.returncode == 0, cp.stderr[-3000:]
H = pd.read_csv(os.path.join(out, "queue_headroom_components.csv"))
V = pd.read_csv(a.ref)
C = ["trace", "rule", "cpu_GiB", "slow", "gpu", "restore", "component"]
for X in (H, V):
    X["cpu_GiB"] = X.cpu_GiB.astype(float)
V = V[V.trace.isin(H.trace.unique()) & V.cpu_GiB.isin(H.cpu_GiB.unique())]
J = V.merge(H, on=C, suffixes=("_noq", "_q"))
load = Q[(Q.policy == "LRU") & (Q.gpu_policy == "LRU")][["trace", "rule", "cpu_GiB", "slow", "gpu", "restore", "qwait_mean"]].copy()
load["cpu_GiB"] = load.cpu_GiB.astype(float)
J = J.merge(load, on=["trace", "rule", "cpu_GiB", "slow", "gpu", "restore"], how="left")
J["load"] = pd.cut(J.qwait_mean, [-1, 2, 20, np.inf], labels=["輕", "中", "重"])
J["d_hr_pt"] = (J.headroom_q - J.headroom_noq) * 100
meta = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"), host=socket.gethostname())
X = J[C + ["qwait_mean", "load", "T_rival_noq", "T_rival_q", "T_oracle_noq", "T_oracle_q", "headroom_noq", "headroom_q", "d_hr_pt",
           "best_rival_noq", "best_rival_q", "oracle_policy_noq", "oracle_policy_q"]].copy()
for i, (k, v) in enumerate(meta.items()):
    X.insert(i, k, v)
X.to_csv(os.path.join(out, "queue_rows.csv"), index=False)
pd.set_option("display.width", 250)
cake = J[(J.restore == "cake") & (J.rule != "min")]
S = cake.groupby(["trace", "component"]).agg(
    n=("headroom_noq", "size"), max_noq=("headroom_noq", "max"), max_q=("headroom_q", "max"),
    n15_noq=("headroom_noq", lambda x: int((x >= 0.15).sum())), n15_q=("headroom_q", lambda x: int((x >= 0.15).sum())),
    med_d_pt=("d_hr_pt", "median"), min_d_pt=("d_hr_pt", "min"), max_d_pt=("d_hr_pt", "max"))
print(S.round(4).to_string())
S2 = cake.groupby(["trace", "load"], observed=True).agg(n=("d_hr_pt", "size"), med_d_pt=("d_hr_pt", "median"), min_d_pt=("d_hr_pt", "min"),
                                                       max_d_pt=("d_hr_pt", "max"), wait_med=("qwait_mean", "median"))
print(S2.round(3).to_string())
print("QUEUE_RESULT", "E2E_OK" if ok_e2e else "E2E_FAIL", f"rows={len(J)}")
sys.exit(0 if ok_e2e else 1)
