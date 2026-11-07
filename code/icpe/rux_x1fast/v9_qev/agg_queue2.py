"""agg_queue2.py — PREREG 偏離 19 的加總表（SemiAnalysis 12 個變體）：三種排隊並列。
  11B ＝ 偏離 11 B 的一階（qpost=1，沒有偏離 14 的 pfx）；基準＝舊版本 C（results/icpe_next/queue/<變體>/queue_rows.csv 的 headroom_noq）
  1m  ＝ 一階、到達不早於上一個完成（qpost=2，pfx=1）；基準＝修正版版本 C
  2nd ＝ 二階（qev，pfx=1）；基準＝修正版版本 C
  輸入：results/icpe_next/queue2/<變體>/queue2_rows.csv（已含 1m、11B 的上限與負載）、results/icpe_next/queue/<變體>/queue_rows.csv。
  只用 cake 的列。負載：各自那一格 (LRU, LRU) 的平均等待，輕 < 2 s、中 2–20 s、重 ≥ 20 s；重的只報告（偏離 19 §3）。
  差異的標示（使用者 11/7 約 17:1x）：
    11B → 1m：到達規則＋pfx。pfx 只動 O-PF／O-ALL 的列，所以分「非預取元件」（O-ORD、O-DROP、O-NOW、O-GPU：只有到達規則）和「預取元件」（O-PF、O-ALL(report)）。
    1m → 2nd：只差回饋。
  檢查：每個變體 324 列；非預取元件的兩個基準（舊、修正版版本 C）逐值相同。任何一項不過就 AGG_FAIL。
用法：python -I agg_queue2.py --base <results/icpe_next> --variants V ... --out <目錄>
輸出：agg_scen.csv（各情境 vs 自己的基準）、agg_feedback.csv（2nd vs 1m）、agg_arrival.csv（1m vs 11B）、agg_crosstab.csv、
      agg_rows.csv（全部格）、agg_summary.md；最後一行 AGG_RESULT。"""
import argparse, hashlib, json, os, socket, sys
from datetime import datetime

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--variants", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
CELL = ["trace", "rule", "cpu_GiB", "slow", "gpu", "restore"]
PF = ["O-PF", "O-ALL(report)"]
COMP = ["O-ORD", "O-DROP", "O-NOW", "O-PF", "O-GPU", "O-ALL(report)"]
TIERS = ["輕", "中", "重"]
T = 0.15
ok = True
inputs = []


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


parts = []
for v in a.variants:
    p2 = os.path.join(a.base, "queue2", v, "queue2_rows.csv")
    p11 = os.path.join(a.base, "queue", v, "queue_rows.csv")
    inputs += [dict(file=p2, sha256=sha(p2)), dict(file=p11, sha256=sha(p11))]
    Q = pd.read_csv(p2)
    O = pd.read_csv(p11)
    for X in (Q, O):
        X["cpu_GiB"] = X.cpu_GiB.astype(float)
    O = O[CELL + ["component", "headroom_noq"]].rename(columns={"headroom_noq": "headroom_noq_old"})
    J = Q.merge(O, on=CELL + ["component"], how="left", validate="one_to_one")
    good = len(Q) == 324 and len(O) == 324 and bool(J.headroom_noq_old.notna().all())
    npf = J[~J.component.isin(PF)]
    same = bool(np.array_equal(npf.headroom_noq.to_numpy(), npf.headroom_noq_old.to_numpy()))
    pfch = int((J[J.component.isin(PF)].headroom_noq != J[J.component.isin(PF)].headroom_noq_old).sum())
    print(f"{v}: 列 {len(Q)}／{len(O)}；非預取元件兩個基準逐值相同 {same}；預取元件基準有變的格 {pfch}／{int(J.component.isin(PF).sum())}")
    ok &= good and same
    J.insert(0, "variant", v)
    parts.append(J)
A = pd.concat(parts, ignore_index=True)
A = A[A.restore == "cake"].copy()
ok &= set(A.component.unique()) == set(COMP)
A["component"] = pd.Categorical(A.component, COMP, ordered=True)
# 每個情境：上限、基準、負載等級
S = {
    "11B": dict(hr="headroom_qpost_orig", base="headroom_noq_old", load="load_qpost_orig", wait="qwait_mean_qpost_orig"),
    "1m": dict(hr="headroom_qpost_m", base="headroom_noq", load="load_qpost_m", wait="qwait_mean_qpost_m"),
    "2nd": dict(hr="headroom_qev", base="headroom_noq", load="load", wait="qwait_mean"),
}
for k, s in S.items():
    A[f"tier_{k}"] = pd.Categorical(A[s["load"]], TIERS, ordered=True)
    ok &= bool(A[f"tier_{k}"].notna().all())


def scen(g, s):
    b, h = g[s["base"]], g[s["hr"]]
    return pd.Series(dict(n=len(g), n15_base=int((b >= T).sum()), n15=int((h >= T).sum()), up=int(((b < T) & (h >= T)).sum()),
                          down=int(((b >= T) & (h < T)).sum()), med_d_pt=float(((h - b) * 100).median())))


rows = []
for k, s in S.items():
    G = A.groupby([f"tier_{k}", "component"], observed=True).apply(lambda g: scen(g, s), include_groups=False).reset_index()
    G = G.rename(columns={f"tier_{k}": "tier"})
    G.insert(0, "scenario", k)
    rows.append(G)
SC = pd.concat(rows, ignore_index=True)


def pair(g, x, y):  # y 對 x：y − x
    hx, hy = g[S[x]["hr"]], g[S[y]["hr"]]
    return pd.Series(dict(n=len(g), n15_x=int((hx >= T).sum()), n15_y=int((hy >= T).sum()), up=int(((hx < T) & (hy >= T)).sum()),
                          down=int(((hx >= T) & (hy < T)).sum()), med_d_pt=float(((hy - hx) * 100).median()),
                          p10_d_pt=float(((hy - hx) * 100).quantile(0.1)), p90_d_pt=float(((hy - hx) * 100).quantile(0.9))))


FB = A.groupby(["tier_2nd", "component"], observed=True).apply(lambda g: pair(g, "1m", "2nd"), include_groups=False).reset_index()
FB = FB.rename(columns={"tier_2nd": "tier"})
A["comp_group"] = np.where(A.component.isin(PF), "預取元件（到達規則＋pfx）", "非預取元件（只有到達規則）")
AR = A.groupby(["tier_1m", "comp_group", "component"], observed=True).apply(lambda g: pair(g, "11B", "1m"), include_groups=False).reset_index()
AR = AR.rename(columns={"tier_1m": "tier"})
INTS = ["n", "n15_base", "n15", "n15_x", "n15_y", "up", "down"]
for D in (SC, FB, AR):
    for c in INTS:
        if c in D.columns:
            D[c] = D[c].astype(int)
CT = []
for x, y in (("2nd", "1m"), ("2nd", "11B"), ("1m", "11B")):
    c = pd.crosstab(A[f"tier_{x}"], A[f"tier_{y}"]) // len(COMP)  # 每格 6 個元件，換回格數
    c = c.reindex(index=TIERS, columns=TIERS, fill_value=0)
    for t1 in TIERS:
        for t2 in TIERS:
            CT.append(dict(row=x, col=y, tier_row=t1, tier_col=t2, cells=int(c.loc[t1, t2])))
CT = pd.DataFrame(CT)
WT = pd.concat([A.drop_duplicates(["variant"] + CELL).groupby(f"tier_{k}", observed=True)[s["wait"]].agg(["size", "min", "median", "max"])
                .rename_axis("tier").reset_index().assign(scenario=k) for k, s in S.items()], ignore_index=True)
meta = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"), host=socket.gethostname())
for name, D in (("agg_scen", SC), ("agg_feedback", FB), ("agg_arrival", AR), ("agg_crosstab", CT), ("agg_wait", WT),
                ("agg_rows", A.drop(columns=["run_id", "ts", "host"]))):
    D = D.copy()
    for i, (k, v) in enumerate(meta.items()):
        D.insert(i, k, v)
    D.to_csv(os.path.join(a.out, f"{name}.csv"), index=False)
pd.set_option("display.width", 250)
R = lambda D: D.round(2).to_string(index=False)
md = [f"# 偏離 19 加總表（{len(a.variants)} 個變體，cake，{len(A)} 列＝{len(A) // len(COMP)} 格 × 6 元件）", "",
      f"- run：{meta['run_id']}（{meta['host']}，{meta['ts']}）",
      "- 變體：" + "、".join(a.variants),
      "- 情境：11B＝偏離 11 B 一階（qpost=1，沒有 pfx，基準＝舊版本 C）；1m＝一階、到達不早於上一個完成（qpost=2，pfx=1，基準＝修正版版本 C）；2nd＝二階（qev，pfx=1，基準＝修正版版本 C）",
      "- 負載：各情境自己那一格 (LRU, LRU) 的平均等待；輕 < 2 s、中 2–20 s、**重 ≥ 20 s 只報告**", "- n15＝上限 ≥ 15% 的格數；up／down＝跨過 15%（升／降）；med_d_pt＝差的中位數（百分點）", "",
      "## 1. 各情境 vs 自己的基準（不含排隊）", "```", R(SC), "```", "",
      "## 2. 回饋：2nd vs 1m（依 2nd 的負載；x＝1m、y＝2nd，差＝2nd − 1m）", "```", R(FB), "```", "",
      "## 3. 到達規則（＋pfx）：1m vs 11B（依 1m 的負載；x＝11B、y＝1m，差＝1m − 11B）", "- 非預取元件只有到達規則不同；預取元件另外有 pfx（11B 沒有偏離 14 的修正）", "```", R(AR), "```", "",
      "## 4. 負載等級交叉（格數）", "```", R(CT.pivot_table(index=["row", "col", "tier_row"], columns="tier_col", values="cells", aggfunc="sum", sort=False)[TIERS].astype(int).reset_index()), "```", "",
      "## 5. 各負載的平均等待（s，每格一個）", "```", R(WT), "```", ""]
open(os.path.join(a.out, "agg_summary.md"), "w", encoding="utf-8").write("\n".join(md))
json.dump(dict(meta, variants=a.variants, inputs=inputs), open(os.path.join(a.out, "AGG_MANIFEST.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\n".join(md))
print("AGG_RESULT", "OK" if ok else "FAIL", f"variants={len(a.variants)} rows={len(A)}")
sys.exit(0 if ok else 1)
