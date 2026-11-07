"""W3 線索 (a) 的分析（SEMANTICS 4-3、4-4）：③＝每格在 NON_ORACLE × {CUEP, CUEO} 裡取最快（同 ② 的事後挑最快）；
  f_c＝(T② − T③) ÷ (T② − T④c)：每個元件 c（PR:21：「沒有簡單線索能拿走大部分」）；
  f_NP43＝(T① − T③) ÷ (T① − T④O-PF)（NP:43，和 C2 一樣的算法；T①＝(LRU, LRU) 的基準）。
② 與 ④ 來自上限表（next_sim_analyze／analyze_v8 的 *_headroom_components.csv：可以是正式表，也可以是同一個 sweep 算的），③ 來自 sweep 裡
GPU 規則＝CUEP／CUEO 的列。只用 cake、rule≠min 的格。
用法：python -I analyze_w3.py --sweep <x1_sweep.csv ...> --headroom <headroom_components.csv> --out <目錄> [--metric mean_ret]
輸出 w3_rows.csv、w3_verdict.csv（每列有 run_id、ts、host）；印摘要；最後一行 W3_RESULT。
判定（PREREG 偏離 9，使用者 11/7 決定）：
  - 算判定的元件＝O-ORD、O-DROP、O-NOW、O-PF、O-GPU（偏離 7）；O-ALL(report) 只報告。
  - (格, 元件) 的上限 ≥ 0.15 才判；③ 要 NON_ORACLE 10 個 × {CUEP, CUEO} 共 20 個組合都在（cue_complete），少了那格＝NOT_MEASURED。
  - 「線索拿走大部分」＝f_c > 0.5。
  - 一個變體（紀錄變體 × rule）：任一個「上限 ≥15%、③ 齊全」的 (格, 元件) 的 f_c ≤ 0.5 → 推翻條件後半成立（沒有簡單線索能拿走大部分）；
    全部都 f_c > 0.5 → 不成立；沒有 ≥15% 的 (格, 元件) → 前半就不成立；有 ≥15% 但 ③ 都不齊 → 未定（NOT_MEASURED）。
  - 寫入飽和照算，另報格數（偏離 7 第 3 點）。"""
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
NEED = len(NON) * 2
three["cue_complete"] = three.n_combos == NEED
JUDGE = ["O-ORD", "O-DROP", "O-NOW", "O-PF", "O-GPU"]
H = pd.read_csv(a.headroom)
H["cpu_GiB"] = H.cpu_GiB.astype(float)
H = H[(H.restore == "cake") & (H.rule != "min") & H.trace.isin(c.trace.unique())]   # 只判 sweep 有跑到的紀錄
J = H.merge(three, on=CELL, how="left")                    # ③ 完全沒跑的格也留著（cue_complete＝False，不會被漏掉）
J["n_combos"] = J.n_combos.fillna(0).astype(int)
J["cue_complete"] = J.cue_complete.fillna(False).astype(bool)
J["f_c"] = (J.T_rival - J.T_cue) / (J.T_rival - J.T_oracle)
base = J.T_base
opf = H[H.component == "O-PF"].set_index(CELL).T_oracle.rename("T_oracle_OPF")
J = J.join(opf, on=CELL)
J["f_NP43"] = (base - J.T_cue) / (base - J.T_oracle_OPF)
J["headroom_ge15"] = J.headroom >= 0.15
J["cue_beats_rival"] = J.T_cue < J.T_rival
meta = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"), host=socket.gethostname())
J["judged"] = J.component.isin(JUDGE)
J["cue_takes_most"] = J.f_c > 0.5
sat = "write_saturated" if "write_saturated" in J.columns else None
X = J[CELL + ["component", "judged", "headroom", "headroom_ge15", "T_base", "T_rival", "best_rival", "best_rival_gpu_policy", "T_oracle", "oracle_policy",
              "oracle_gpu_policy", "T_cue", "cue_cpu_policy", "cue_gpu_policy", "n_combos", "cue_complete", "f_c", "cue_takes_most", "T_oracle_OPF", "f_NP43",
              "cue_beats_rival"] + ([sat] if sat else [])].copy()
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
# 判定（偏離 9）
V = []
for (tr, ru), g in J[J.judged].groupby(["trace", "rule"]):
    hi = g[g.headroom_ge15]
    ok = hi[hi.cue_complete]
    n_not = int((~ok.cue_takes_most).sum())
    if len(hi) == 0:
        verdict = "前半不成立（沒有上限 ≥15% 的格）"
    elif n_not > 0:
        verdict = "後半成立：沒有簡單線索能拿走大部分"
    elif len(ok) < len(hi):
        verdict = "未定（有 ≥15% 的格 ③ 沒跑齊）"
    else:
        verdict = "後半不成立：簡單線索拿走大部分（所有 ≥15% 的格 f > 0.5）"
    V.append(dict(trace=tr, rule=ru, n_cells_judged=len(g), n_ge15=len(hi), n_ge15_cue_complete=len(ok), n_ge15_f_le_half=n_not,
                  n_ge15_f_le_half_saturated=int(((~ok.cue_takes_most) & ok[sat].astype(str).str.lower().eq("true")).sum()) if sat else -1,
                  f_c_median_ge15=float(ok.f_c.median()) if len(ok) else float("nan"), f_c_min_ge15=float(ok.f_c.min()) if len(ok) else float("nan"),
                  verdict=verdict))
VD = pd.DataFrame(V)
for i, (k, v) in enumerate(meta.items()):
    VD.insert(i, k, v)
VD.to_csv(os.path.join(a.out, "w3_verdict.csv"), index=False)
print("判定（偏離 9）："); print(VD.drop(columns=list(meta)).to_string(index=False))
print("W3_RESULT", f"cells={J.drop_duplicates(CELL).shape[0]} rows={len(J)}")
