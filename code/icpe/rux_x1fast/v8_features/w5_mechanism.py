"""W5 探索的後續（只讀 explore_w5.py 的輸出，不模擬）：上限上升超過門檻的 (格, 元件)，④ 與 ② 的最佳組合的
回來請求「CPU 載入／慢層載入／重算」次數，在 BF16 → W5 的相對變化。
用法：python -I w5_mechanism.py <explore_w5 的輸出目錄> [門檻（百分點，預設 5）] [W5 那一邊：forced（預設，探索用）｜allowed（偏離 11 登記的算法）]
  allowed：W5 那一邊的最佳組合取自 sweep_allowed（每個組合事後挑較快格式），另記那個組合挑到的格式（fmt_allowed：w5／bf16）。"""
import os, sys
import pandas as pd
d = sys.argv[1]
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
WS = sys.argv[3] if len(sys.argv) > 3 else "forced"
assert WS in ("forced", "allowed"), WS
H = {"base": pd.read_csv(os.path.join(d, "base_headroom_components.csv")), "forced": pd.read_csv(os.path.join(d, f"{WS}_headroom_components.csv"))}
S = {"base": pd.read_csv(os.path.join(d, "sweep_base.csv")), "forced": pd.read_csv(os.path.join(d, f"sweep_{WS}.csv"))}
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
            m = m[(m.policy == j[pol + suf]) & (m.gpu_policy == j[gp + suf])]
            assert len(m) == 1, (side, t, len(m))
            m = m.iloc[0]
            for c in ("ret_cpu_loads", "ret_slow_loads", "ret_recompute"):
                rec[f"{side}_{c}_{t}"] = m[c]
            if t == "forced" and WS == "allowed":
                # allowed 的列＝W5 的列 iff 同一組合的 W5 比 BF16 快（explore_w5 的 pick）；用同一組合的 BF16 列比
                mb = S["base"]
                for k in CELL:
                    mb = mb[mb[k] == j[k]]
                mb = mb[(mb.policy == m.policy) & (mb.gpu_policy == m.gpu_policy)]
                assert len(mb) == 1
                rec[f"{side}_fmt_allowed"] = "w5" if m.mean_ret < mb.iloc[0].mean_ret else "bf16"
        for c in ("ret_slow_loads", "ret_recompute"):
            b, w = rec[f"{side}_{c}_base"], rec[f"{side}_{c}_forced"]
            rec[f"{side}_{c}_rel"] = (w - b) / b if b else float("nan")
    out.append(rec)
X = pd.DataFrame(out)
X.insert(0, "w5_side", WS)
X.to_csv("w5_mechanism_rows.csv", index=False)
pd.set_option("display.width", 250)
print(f"W5 那一邊＝{WS}；上升 > {thr} 個百分點的 (格, 元件)：{len(X)}")
if not len(X):
    print("W5_MECH_RESULT rows=0"); sys.exit(0)
print(X.groupby("component")[["d_hr_pt", "oracle_ret_slow_loads_rel", "rival_ret_slow_loads_rel", "oracle_ret_recompute_rel", "rival_ret_recompute_rel"]].median().round(3).to_string())
print(X.groupby(["component", "slow"]).size().to_string())
print("W5_MECH_RESULT", f"rows={len(X)}")
