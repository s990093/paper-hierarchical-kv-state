"""decision_map.py — ICPE 的決策地圖（只報告；只讀 T1 已有的模擬輸出，不跑新的模擬）。

「在 Cake 之上，什麼條件下預測才值得做」：每個（trace, 慢層, GPU 狀態, CPU 容量），預知回來時間的 oracle 最多能贏最好的對稱對手多少
（X1 的 headroom，取 O-PF／O-ORD／O-DROP／O-NOW 裡最大的那個；Mooncake 用 rule＝min〔R1、R2 取小〕，FreeInference 用 session_id；還原＝Cake），
以及 conversation 上免費的打字提示（HINT-PF）拿到 oracle 預取改善的比例（P0-2 全尺寸，R1、Cake）。
門檻照 X1 的登記：≥15%「有看頭」、5–15%「灰區」、<5%「沒有空間」。

  python code/icpe/decision_map.py → results/icpe_h200/t1/decision_map.csv、decision_map.md
輸入（都是 T1：CPU 層＝H200 的常數）：
  results/icpe_h200/t1/p01/sim_V0/x1/x1_headroom_components.csv（conversation、toolagent）
  results/icpe_h200/t1/p12/x1/x1fi_headroom_components.csv（FreeInference 前 6 週）
  results/icpe_h200/t1/p02/fullsim_hint.csv（HINT-PF 全尺寸）
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T1 = os.path.join(REPO, "results", "icpe_h200", "t1")
VARIANTS = {"": "不截尾", "h3600": "截尾 H＝3,600 s（B2：O-DROP 大半是 trace 太短）"}
SRC = dict(mooncake=os.path.join(T1, "p01", "sim_V0", "x1", "x1_headroom_components.csv"),
           fi=os.path.join(T1, "p12", "x1", "x1fi_headroom_components.csv"),
           hint=os.path.join(T1, "p02", "fullsim_hint.csv"))


def zone(h):
    return "有看頭" if h >= 0.15 else ("灰區" if h >= 0.05 else "沒有空間")


def main():
    rid = os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc"
    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = ["# 決策地圖（T1：CPU 層＝H200 的常數；〔模擬〕；只報告）", "",
             f"run `{rid}`；來源：`results/icpe_h200/t1/p01/sim_V0/x1/x1{{,h3600}}_headroom_components.csv`、"
             f"`{os.path.relpath(SRC['fi'], REPO)}`（FreeInference 前 6 週，不截尾）、`{os.path.relpath(SRC['hint'], REPO)}`", "",
             "格內：oracle 最多贏最好的對稱對手的比例（最大的那個成分）；conversation 另附 HINT-PF 拿到 O-PF 改善的比例（>1＝O-PF 騰空間照 LRU，不是上界）。"
             "≥15% 有看頭、5–15% 灰區、<5% 沒有空間。", ""]
    for var, label in VARIANTS.items():
        one(var, label, rid, ts, lines)
    md = os.path.join(T1, "decision_map.md")
    open(md, "w").write("\n".join(lines) + "\n")
    print("→", md)


def one(var, label, rid, ts, lines):
    m = pd.read_csv(SRC["mooncake"].replace("x1_headroom", f"x1{var}_headroom"))
    m = m[(m.rule == "min") & (m.restore == "cake")]
    f = pd.read_csv(SRC["fi"])
    f = f[(f.rule == "session_id") & (f.restore == "cake")]
    d = pd.concat([m, f])
    key = ["trace", "slow", "gpu", "cpu_GiB"]
    best = d.loc[d.groupby(key).headroom.idxmax()][key + ["component", "headroom", "oracle_policy", "best_rival", "src_runs"]]
    h = pd.read_csv(SRC["hint"])
    h = h[(h.rule == "R1") & (h.restore == "cake")][["trace", "slow", "gpu", "cpu_GiB", "hint_vs_lru", "opf_vs_lru", "hint_share_of_opf", "run_id"]]
    h = h.rename(columns={"run_id": "hint_run"})
    out = best.merge(h, on=key, how="left")
    out["zone"] = out.headroom.map(zone)
    out.insert(0, "ts", ts)
    out.insert(0, "run_id", rid)
    out = out.sort_values(key)
    out.insert(2, "variant", var or "none")
    p = os.path.join(T1, f"decision_map{'_' + var if var else ''}.csv")
    out.to_csv(p, index=False)
    lines += [f"## {label}", ""]
    cpus = sorted(out.cpu_GiB.unique())
    lines.append("| trace | 慢層 | GPU | " + " | ".join(f"CPU {c} GiB" for c in cpus) + " |")
    lines.append("|:--|:--|:--|" + "--:|" * len(cpus))
    for (t, s, g), x in out.groupby(["trace", "slow", "gpu"], sort=True):
        cells = []
        for c in cpus:
            r = x[x.cpu_GiB == c]
            if r.empty:
                cells.append("—")
                continue
            r = r.iloc[0]
            txt = f"{100 * r.headroom:.1f}%（{r.component}）"
            if pd.notna(r.hint_share_of_opf):
                txt += f"；提示 {r.hint_share_of_opf:.2f}"
            cells.append(txt)
        lines.append(f"| {t} | {s} | {g} | " + " | ".join(cells) + " |")
    z = out.groupby(["trace", "zone"]).size().unstack(fill_value=0)
    lines += ["", "格數：" + "；".join(f"{t} " + "、".join(f"{k} {int(v)}" for k, v in r.items()) for t, r in z.iterrows()), ""]
    print("→", p, len(out), label)
    print(z.to_string())


if __name__ == "__main__":
    main()
