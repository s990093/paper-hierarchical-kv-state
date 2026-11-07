"""探索（不是正式結果）：W5（CPU 層 INT4＋一次拆完）對上限的影響。
用法：python -I explore_w5.py <BF16 的 x1_sweep.csv> <W5 的 x1_sweep.csv（sweep_v8 RUX_W5=1）> <輸出目錄>
  base＝現在的 CPU 層（BF16）；forced＝每個組合都用 W5；allowed＝每個 (格, CPU 規則, GPU 規則, 慢層規則) 取兩種格式較快的
  （SEMANTICS 5-2：「允許 INT4」＝② 和 ④ 都可以事後挑格式）。三份都用 analyze_v8（同版本 B 的算法；有 APC、BELP 時照偏離 10）。
輸出 explore_w5_rows.csv（每列有 run_id、ts、host）；印摘要；最後一行 EXPLORE_W5_RESULT。"""
import os, socket, subprocess, sys
from datetime import datetime

import pandas as pd

fb, fw, out = sys.argv[1:4]
REPO = os.environ["REPO"]
os.makedirs(out, exist_ok=True)
B = pd.read_csv(fb)
W = pd.read_csv(fw)
assert (W.cpu_fmt == "int4_oneshot").all()
W = W.drop(columns=["cpu_fmt"])
K = ["trace", "rule", "cpu_GiB", "policy", "slow", "gpu", "restore", "gpu_GiB", "gpu_write", "gpu_policy", "pf_target", "slow_cap_frac", "slow_policy"]
K = [k for k in K if k in B.columns]
assert len(B) == len(W) and set(map(tuple, B[K].astype(str).values)) == set(map(tuple, W[K].astype(str).values)), "兩份 sweep 的組合不同"
M = B.merge(W, on=K, suffixes=("", "_w5"))
pick = M.mean_ret_w5 < M.mean_ret
A = B.copy().set_index(K)
Wi = W.set_index(K)
Mi = M.set_index(K)
idx = Mi.index[pick.values]
A.loc[idx, :] = Wi.loc[idx, A.columns]
A = A.reset_index()
paths = {}
for tag, X in (("base", B), ("forced", W), ("allowed", A)):
    p = os.path.join(out, f"sweep_{tag}.csv")
    X.to_csv(p, index=False)
    # analyze_v8＝next_sim_analyze＋「BELP 是 oracle、只進 O-GPU 的 ④」（偏離 10）；沒有 BELP 列時兩者逐值相同（check_analyze_v8）
    cp = subprocess.run([sys.executable, "-I", os.path.join(REPO, "code/icpe/rux_x1fast/v8_features/analyze_v8.py"), "--sweep", p, "--out-dir", out, "--out-prefix", tag],
                        capture_output=True, text=True, env=dict(os.environ))
    assert cp.returncode == 0, cp.stderr[-3000:]
    paths[tag] = os.path.join(out, f"{tag}_headroom_components.csv")
C = ["trace", "rule", "cpu_GiB", "slow", "gpu", "restore", "component"]
H = {t: pd.read_csv(p) for t, p in paths.items()}
J = H["base"][C + ["T_rival", "T_oracle", "headroom"]]
for t in ("forced", "allowed"):
    J = J.merge(H[t][C + ["T_rival", "T_oracle", "headroom"]].rename(columns={"T_rival": f"T_rival_{t}", "T_oracle": f"T_oracle_{t}", "headroom": f"headroom_{t}"}), on=C)
J["d_hr_forced_pt"] = (J.headroom_forced - J.headroom) * 100
J["d_hr_allowed_pt"] = (J.headroom_allowed - J.headroom) * 100
meta = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"), host=socket.gethostname())
X = J.copy()
for i, (k, v) in enumerate(meta.items()):
    X.insert(i, k, v)
X.to_csv(os.path.join(out, "explore_w5_rows.csv"), index=False)
pd.set_option("display.width", 250)
cake = J[J.restore == "cake"]
S = cake.groupby(["trace", "rule", "component"]).agg(
    n=("headroom", "size"), max_base=("headroom", "max"), max_forced=("headroom_forced", "max"), max_allowed=("headroom_allowed", "max"),
    n15_base=("headroom", lambda x: int((x >= 0.15).sum())), n15_forced=("headroom_forced", lambda x: int((x >= 0.15).sum())),
    n15_allowed=("headroom_allowed", lambda x: int((x >= 0.15).sum())),
    med_d_forced_pt=("d_hr_forced_pt", "median"), med_d_allowed_pt=("d_hr_allowed_pt", "median"),
    max_d_allowed_pt=("d_hr_allowed_pt", "max"), min_d_allowed_pt=("d_hr_allowed_pt", "min"))
print(S.round(4).to_string())
print(f"W5 較快的組合：{int(pick.sum())}／{len(pick)}")
print("EXPLORE_W5_RESULT", f"cells={len(J)}")
