"""W5 探索的後續（只讀 explore_w5.py 的輸出，不模擬）：上限上升超過門檻的 (格, 元件)，④ 與 ② 的最佳組合的
回來請求「CPU 載入／慢層載入／重算」次數，在 BF16 → W5 的相對變化。用法：python -I w5_mechanism.py <explore_w5 的輸出目錄> [門檻（百分點，預設 5）]"""
import os, sys
import pandas as pd
d = sys.argv[1]
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
H = {t: pd.read_csv(os.path.join(d, f"{t}_headroom_components.csv")) for t in ("base", "forced")}
S = {t: pd.read_csv(os.path.join(d, f"sweep_{t}.csv")) for t in ("base", "forced")}
C = ["trace", "rule", "cpu_GiB", "slow", "gpu", "restore", "component"]
J = H["base"].merge(H["forced"], on=C, suffixes=("_b", "_w"))
J = J[(J.rule != "min") & ((J.headroom_w - J.headroom_b) * 100 > thr)]
CELL = ["trace", "rule", "cpu_GiB", "slow", "gpu", "restore"]
out = []
for _, j in J.iterrows():
    rec = {k: j[k] for k in C}
    rec["d_hr_pt"] = round((j.headroom_w - j.headroom_b) * 100, 2)
    for side, pol, gp in (("oracle", "oracle_policy", "oracle_gpu_policy"), ("rival", "best_rival", "best_rival_gpu_policy")):
        for t, suf in (("base", "_b"), ("forced", "_w")):
            s = S[t]
            m = s
            for k in CELL:
                m = m[m[k] == j[k]]
            m = m[(m.policy == j[pol + suf]) & (m.gpu_policy == j[gp + suf])].iloc[0]
            for c in ("ret_cpu_loads", "ret_slow_loads", "ret_recompute"):
                rec[f"{side}_{c}_{t}"] = m[c]
        for c in ("ret_slow_loads", "ret_recompute"):
            b, w = rec[f"{side}_{c}_base"], rec[f"{side}_{c}_forced"]
            rec[f"{side}_{c}_rel"] = (w - b) / b if b else float("nan")
    out.append(rec)
X = pd.DataFrame(out)
X.to_csv("w5_mechanism_rows.csv", index=False)
pd.set_option("display.width", 250)
print(f"上升 > {thr} 個百分點的 (格, 元件)：{len(X)}")
print(X.groupby("component")[["d_hr_pt", "oracle_ret_slow_loads_rel", "rival_ret_slow_loads_rel", "oracle_ret_recompute_rel", "rival_ret_recompute_rel"]].median().round(3).to_string())
print(X.groupby(["component", "slow"]).size().to_string())
