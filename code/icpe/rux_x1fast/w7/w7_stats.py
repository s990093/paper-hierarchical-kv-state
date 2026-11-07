"""w7_stats.py — 合成紀錄的實現值 vs 設定值（只讀 nb、diag、meta；不用 w7gen 的程式碼）。給每份紀錄一列；只報告，不判定。
用法：python -I w7_stats.py --nb-dir <dir> --names <name ...> [--out <csv>]
  輪數：沒有被上下文上限（1,048,576 token）截掉的 session 的平均輪數 vs turns_mean（截掉的 session 數＝gen_stats.n_cap_end）。
  觸發：回來的請求裡工具觸發（user_trigger＝false）的比例 vs tool_share。
  間隔：工具／人各自的平均間隔 vs a_tool_s／a_human_s；全部的平均 vs gap_mean_s；間隔÷該類平均的中位數 vs 分佈族的理論中位數
        （exp：ln2＝0.693；lognormal（σ＝1.5684）：exp(−σ²/2)＝0.292；lomax（α＝1.5）：(2^(2/3)−1)·0.5＝0.294）。
  fork：fork 的 session 比例 vs fork_frac（沒有活 session 可 fork 的次數＝gen_stats.n_fork_fallback）。
  起始內容：新對話第一個請求的 token 數（＝nall·512 的上界，所以用完整塊數×512 估）只報中位數。"""
import argparse, json, math, os
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--nb-dir", required=True); ap.add_argument("--names", nargs="+", required=True); ap.add_argument("--out", default="")
a = ap.parse_args()
TH = {"exp": math.log(2), "lognormal": math.exp(-0.5 * 1.5684286763930078 ** 2), "lomax": (2 ** (2 / 3) - 1) * 0.5}
rows = []
for nm in a.names:
    z = np.load(os.path.join(a.nb_dir, nm + ".npz")); g = np.load(os.path.join(a.nb_dir, nm + ".diag.npz"))
    m = json.load(open(os.path.join(a.nb_dir, nm + ".meta.json"))); P = m["params"]; G = m["gen_stats"]
    kind, user, sess, gap = z["kind"], z["user_trigger"], z["sess"], g["gap"]
    cont = kind == 1
    tool = cont & ~user
    hum = cont & user
    turns = np.bincount(sess)
    fam = P["gap_family"]
    rows.append(dict(name=nm, point=m["point"], seed=m["seed"], turns_mean=P["turns_mean"], turns_real_all=float(turns.mean()), n_cap=G["n_cap_end"],
                     tool_share=P["tool_share"], tool_share_real=float(tool.sum() / cont.sum()) if cont.any() else float("nan"),
                     gap_mean_s=P["gap_mean_s"], gap_mean_real=float(gap[cont].mean()) if cont.any() else float("nan"),
                     a_tool=G["a_tool_s"], a_tool_real=float(gap[tool].mean()) if tool.any() else float("nan"),
                     a_human=G["a_human_s"], a_human_real=float(gap[hum].mean()) if hum.any() else float("nan"),
                     family=fam, med_unit_theory=TH[fam],
                     med_unit_real=float(np.median(np.r_[gap[tool] / G["a_tool_s"], gap[hum] / G["a_human_s"]])) if cont.any() else float("nan"),
                     fork_frac=P["fork_frac"], fork_share_real=float((kind == 2).sum() / (kind != 1).sum()), n_fork_fallback=G["n_fork_fallback"]))
d = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(d.drop(columns=["name"]).round(3).to_string(index=False))
if a.out:
    d.to_csv(a.out, index=False)
