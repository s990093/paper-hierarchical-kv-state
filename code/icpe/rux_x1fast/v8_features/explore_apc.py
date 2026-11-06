"""探索（不是正式結果）：block 粒度 APC／BELP 加進 GPU 規則之後，上限（headroom）怎麼變。
用法：python -I explore_apc.py <x1_sweep.csv> <verB 的 *_headroom_components.csv(.xz)> <輸出目錄>
  1. 拿掉 APC、BELP 的列 → next_sim_analyze（同版本 B 的算法：沒有 --rival-only-gpu）→ 和版本 B 表上同樣的格逐值比（重現檢查）。
  2. 只拿掉 BELP → next_sim_analyze：APC 是不預測的 GPU 規則，② 和 ④ 都有（SEMANTICS 2-5）。
  3. O-GPU 另外算：④＝NON_ORACLE × {BEL, BELP}、②＝NON_ORACLE × 不預測的 GPU 規則（含 APC）。
輸出 explore_apc_rows.csv（每列有 run_id、ts、host）並印摘要；最後一行 EXPLORE_RESULT。"""
import os, socket, subprocess, sys
from datetime import datetime

import numpy as np
import pandas as pd

sw, verb, out = sys.argv[1:4]
REPO = os.environ["REPO"]
PY = sys.executable
os.makedirs(out, exist_ok=True)
d = pd.read_csv(sw)
base = d[~d.gpu_policy.isin(["APC", "BELP"])]
apc = d[d.gpu_policy != "BELP"]
base.to_csv(os.path.join(out, "sweep_base.csv"), index=False)
apc.to_csv(os.path.join(out, "sweep_apc.csv"), index=False)
for tag in ("base", "apc"):
    cp = subprocess.run([PY, "-I", os.path.join(REPO, "code/icpe/next_sim_analyze.py"), "--sweep", os.path.join(out, f"sweep_{tag}.csv"),
                         "--out-dir", out, "--out-prefix", tag], capture_output=True, text=True, env=dict(os.environ))
    assert cp.returncode == 0, cp.stderr[-3000:]
    open(os.path.join(out, f"analyze_{tag}.log"), "w").write(cp.stdout + cp.stderr)
K = ["trace", "rule", "cpu_GiB", "slow", "gpu", "restore", "component"]
B = pd.read_csv(os.path.join(out, "base_headroom_components.csv"))
A = pd.read_csv(os.path.join(out, "apc_headroom_components.csv"))
V = pd.read_csv(verb)
V = V[V.trace.isin(B.trace.unique()) & V.cpu_GiB.isin(B.cpu_GiB.unique())]
for X in (B, V, A):
    X["cpu_GiB"] = X.cpu_GiB.astype(float)
# 1. 重現檢查
m = B.merge(V, on=K, suffixes=("", "_v"))
cols = ["T_oracle", "T_rival", "headroom", "best_rival", "best_rival_gpu_policy", "oracle_policy", "oracle_gpu_policy"]
ndiff = {c: int((m[c].astype(str) != m[c + "_v"].astype(str)).sum()) for c in cols}
print(f"重現檢查：版本 B 表上 {len(V)} 格×元件，對上 {len(m)}；各欄不同數 {ndiff}")
rep_ok = len(m) == len(V) and all(v == 0 for v in ndiff.values())
# 2. APC 前後
j = B.merge(A, on=K, suffixes=("_base", "_apc"))
j["d_rival"] = (j.T_rival_apc - j.T_rival_base) / j.T_rival_base
j["d_oracle"] = (j.T_oracle_apc - j.T_oracle_base) / j.T_oracle_base
j["d_headroom_pt"] = (j.headroom_apc - j.headroom_base) * 100
j["rival_uses_apc"] = j.best_rival_gpu_policy_apc == "APC"
j["oracle_uses_apc"] = j.oracle_gpu_policy_apc == "APC"
# 3. O-GPU with BELP
NON = ["LRU", "LRU2", "ARC", "S3FIFO", "GDSF", "PENS", "LRU+NONE", "LRU+2HW", "LRU+2HA", "LRU+RPF"]
CELL = ["trace", "rule", "cpu_GiB", "slow", "gpu", "restore"]
rows = []
for key, g in d[d.policy.isin(NON)].groupby(CELL):
    T = g.groupby("gpu_policy").mean_ret.min()
    no = T.drop(labels=[x for x in ("BEL", "BELP") if x in T.index])
    no_old = no.drop(labels=["APC"]) if "APC" in no.index else no
    r_new, r_old = no.min(), no_old.min()
    o_old = T.get("BEL", np.nan)
    o_new = min(T.get("BEL", np.inf), T.get("BELP", np.inf))
    rows.append(dict(zip(CELL, key), T_rival_old=r_old, T_oracle_old=o_old, headroom_old=(r_old - o_old) / r_old,
                     T_rival_new=r_new, T_oracle_new=o_new, headroom_new=(r_new - o_new) / r_new,
                     oracle_new_is_belp=bool(T.get("BELP", np.inf) < T.get("BEL", np.inf))))
G = pd.DataFrame(rows)
meta = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"), host=socket.gethostname())
for X, nm in ((j, "explore_apc_rows.csv"), (G, "explore_ogpu_belp_rows.csv")):
    X = X.copy()
    for i, (k, v) in enumerate(meta.items()):
        X.insert(i, k, v)
    X.to_csv(os.path.join(out, nm), index=False)
pd.set_option("display.width", 250)
cake = j[j.restore == "cake"]
S = cake.groupby(["trace", "rule", "component"]).agg(
    n=("headroom_base", "size"), max_base=("headroom_base", "max"), max_apc=("headroom_apc", "max"),
    n15_base=("headroom_base", lambda x: int((x >= 0.15).sum())), n15_apc=("headroom_apc", lambda x: int((x >= 0.15).sum())),
    med_d_rival=("d_rival", "median"), min_d_rival=("d_rival", "min"), med_d_hr_pt=("d_headroom_pt", "median"),
    max_d_hr_pt=("d_headroom_pt", "max"), min_d_hr_pt=("d_headroom_pt", "min"), rival_apc=("rival_uses_apc", "sum"), oracle_apc=("oracle_uses_apc", "sum"))
print(S.round(4).to_string())
Gs = G.groupby(["trace", "rule"]).agg(n=("headroom_old", "size"), max_old=("headroom_old", "max"), max_new=("headroom_new", "max"),
                                       med_old=("headroom_old", "median"), med_new=("headroom_new", "median"), belp_wins=("oracle_new_is_belp", "sum"))
print("O-GPU（④ 含 BELP）："); print(Gs.round(4).to_string())
print("EXPLORE_RESULT", "REPRODUCED" if rep_ok else "NOT_REPRODUCED", f"cells={len(j)}")
