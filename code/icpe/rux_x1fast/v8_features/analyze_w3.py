"""W3 線索 (a) 的分析（SEMANTICS 4-3、4-4）：③＝每格在 NON_ORACLE × {CUEP, CUEO} 裡取最快（同 ② 的事後挑最快）；
  f_c＝(T② − T③) ÷ (T② − T④c)：每個元件 c（PR:21：「沒有簡單線索能拿走大部分」）；
  f_NP43＝(T① − T③) ÷ (T① − T④O-PF)（NP:43，和 C2 一樣的算法；T①＝(LRU, LRU) 的基準）。
② 與 ④ 來自上限表（next_sim_analyze／analyze_v8 的 *_headroom_components.csv：可以是正式表，也可以是同一個 sweep 算的），③ 來自 sweep 裡
GPU 規則＝CUEP／CUEO 的列。只用 cake、rule≠min 的格。
用法：python -I analyze_w3.py --sweep <x1_sweep.csv ...> --headroom <headroom_components.csv> --out <目錄> [--metric mean_ret]
輸出 w3_rows.csv（每列有 run_id、ts、host）；印摘要；最後一行 W3_RESULT。「大部分」的門檻沒登記（SEMANTICS 4-3 建議 f > 0.5），這裡只算數字。"""
import argparse, os, socket
from datetime import datetime

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--sweep", nargs="+", required=True)
ap.add_argument("--headroom", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--metric", default="mean_ret")
a = ap.parse_args()
NON = ["LRU", "LRU2", "ARC", "S3FIFO", "GDSF", "PENS", "LRU+NONE", "LRU+2HW", "LRU+2HA", "LRU+RPF"]
CELL = ["trace", "rule", "cpu_GiB", "slow", "gpu", "restore"]
d = pd.concat([pd.read_csv(p) for p in a.sweep], ignore_index=True)
c = d[d.gpu_policy.isin(["CUEP", "CUEO"]) & d.policy.isin(NON)]
if "slow_cap_frac" in c:
    c = c[c.slow_cap_frac.fillna(0) == 0]
assert len(c), "sweep 裡沒有 CUEP／CUEO 的列"
for x in (c,):
    x["cpu_GiB"] = x.cpu_GiB.astype(float)
n_combo = c.groupby(CELL).size()
three = c.loc[c.groupby(CELL)[a.metric].idxmin(), CELL + ["policy", "gpu_policy", a.metric]].rename(
    columns={"policy": "cue_cpu_policy", "gpu_policy": "cue_gpu_policy", a.metric: "T_cue"})
three["n_combos"] = n_combo.reindex(three.set_index(CELL).index).values
H = pd.read_csv(a.headroom)
H["cpu_GiB"] = H.cpu_GiB.astype(float)
H = H[(H.restore == "cake") & (H.rule != "min")]
J = H.merge(three, on=CELL)
J["f_c"] = (J.T_rival - J.T_cue) / (J.T_rival - J.T_oracle)
base = J.T_base
opf = H[H.component == "O-PF"].set_index(CELL).T_oracle.rename("T_oracle_OPF")
J = J.join(opf, on=CELL)
J["f_NP43"] = (base - J.T_cue) / (base - J.T_oracle_OPF)
J["headroom_ge15"] = J.headroom >= 0.15
J["cue_beats_rival"] = J.T_cue < J.T_rival
meta = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"), host=socket.gethostname())
X = J[CELL + ["component", "headroom", "headroom_ge15", "T_base", "T_rival", "best_rival", "best_rival_gpu_policy", "T_oracle", "oracle_policy",
              "oracle_gpu_policy", "T_cue", "cue_cpu_policy", "cue_gpu_policy", "n_combos", "f_c", "T_oracle_OPF", "f_NP43", "cue_beats_rival"]].copy()
for i, (k, v) in enumerate(meta.items()):
    X.insert(i, k, v)
os.makedirs(a.out, exist_ok=True)
X.to_csv(os.path.join(a.out, "w3_rows.csv"), index=False)
pd.set_option("display.width", 250)
S = J.groupby(["trace", "rule", "component"]).agg(
    n=("f_c", "size"), n_ge15=("headroom_ge15", "sum"), cue_beats_rival=("cue_beats_rival", "sum"),
    f_c_med=("f_c", "median"), f_c_max=("f_c", "max"),
    f_c_med_ge15=("f_c", lambda x: float(np.median(x[J.loc[x.index, "headroom_ge15"]])) if J.loc[x.index, "headroom_ge15"].any() else np.nan),
    n_f_gt_half_ge15=("f_c", lambda x: int(((x > 0.5) & J.loc[x.index, "headroom_ge15"]).sum())))
print(S.round(3).to_string())
print("f_NP43（每格一個）：中位數", round(float(J.drop_duplicates(CELL).f_NP43.median()), 3))
print("③ 選到的 GPU 規則：", J.drop_duplicates(CELL).cue_gpu_policy.value_counts().to_dict())
print("W3_RESULT", f"cells={J.drop_duplicates(CELL).shape[0]} rows={len(J)}")
