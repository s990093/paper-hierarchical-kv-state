"""檢查 analyze_v8.py：(a) O-GPU 以外的元件＝拿掉 BELP 列後用原本的 next_sim_analyze 算的；(b) O-GPU＝explore_apc.py 自己算的
（④＝NON_ORACLE × {BEL, BELP}）。用法：python -I check_analyze_v8.py <analyze_v8 的 headroom csv> <explore_apc 的輸出目錄>"""
import os, sys
import numpy as np
import pandas as pd
v8, ex = sys.argv[1:3]
H = pd.read_csv(v8)
A = pd.read_csv(os.path.join(ex, "apc_headroom_components.csv"))
G = pd.read_csv(os.path.join(ex, "explore_ogpu_belp_rows.csv"))
K = ["trace", "rule", "cpu_GiB", "slow", "gpu", "restore", "component"]
cols = ["T_oracle", "T_rival", "headroom", "best_rival", "best_rival_gpu_policy", "oracle_policy", "oracle_gpu_policy"]
h = H[H.component != "O-GPU"].merge(A[A.component != "O-GPU"], on=K, suffixes=("", "_a"))
na = sum(int((h[c].astype(str) != h[c + "_a"].astype(str)).sum()) for c in cols)
print(f"(a) O-GPU 以外：{len(h)} 列（analyze_v8 {int((H.component != 'O-GPU').sum())}、原本 {int((A.component != 'O-GPU').sum())}），不同 {na}")
g = H[(H.component == "O-GPU") & (H.rule != "min")].merge(G, on=["trace", "rule", "cpu_GiB", "slow", "gpu", "restore"])
nb = int((~np.isclose(g.headroom, g.headroom_new, rtol=0, atol=0)).sum() + (g.T_oracle != g.T_oracle_new).sum() + (g.T_rival != g.T_rival_new).sum())
print(f"(b) O-GPU：{len(g)} 列，不同 {nb}；④ 用 BELP 的列 {int((g.oracle_gpu_policy == 'BELP').sum())}")
ok = na == 0 and nb == 0 and len(h) == int((A.component != "O-GPU").sum()) and len(g) == len(G)
print("CHECK_RESULT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
