"""m10_sweep_analyze.py — 大掃描海選的彙整、挑選手（09 §4.2，規則在看到任何海選結果之前寫死）、圖與表。CPU 就好。

輸入：m10_sweep.py screen 的逐列 CSV（.csv.gz，可多個 shard）。
輸出（--out-dir）：
  agg_<m>.csv          每個（格式, 放法, 佔比, 讀法）的平均：kl、first_ok、gold_lp 對 E0 的差、top1、讀到的低精度、位元組
  tests_<m>.csv        每一格（格式, 佔比）× 放法 X：Δ＝KL(隨機平均)−KL(X) 的配對 bootstrap CI 與三個條件
  finalists_<m>.json   選手（≤6，同格式 ≤2）＋前緣點（2 個）＋對照組；沒有勝出也照實寫
  tables_<m>.md        上面三樣的表（標「探索、未校正多重比較」）
  figs：sweep_kl_<m>.png（每個格式一格：KL 對佔比，每種放法一條線，隨機是帶子）、sweep_bytes_<m>.png（格式之間：KL 對平均位元／值）

用法：python code/m10_sweep_analyze.py --model llama --csv <screen_*.csv.gz…> --out-dir results/m10_sweep_mi300x/analysis
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

STRUCT = ["F", "M", "B", "ERR", "ATT", "AXE"]          # 和隨機比的放法（ORC 是作弊上限，不當選手）
RAND = ["R0", "R1", "R2", "R3", "R4"]
N_BOOT = 10_000
MAX_FINAL, MAX_PER_FMT, N_FRONTIER = 6, 2, 2
FRONTIER_DROP = -0.02                                 # first_ok 比 E0 掉不到 2 個百分點（點估計）


def load(paths: list[str], model: str) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df[df["model"] == model].copy()
    for c in ("kl", "gold_lp", "gold_lp_e0", "top1", "first_ok", "first_ok_e0", "frac", "bytes_stored", "bytes_read",
              "lp_read", "bits_per_value"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["fmt"] = df["fmt"].fillna("")
    df["d_gold_lp"] = df["gold_lp"] - df["gold_lp_e0"]
    df["d_first_ok"] = df["first_ok"] - df["first_ok_e0"]
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["fmt", "policy", "frac", "readout"], dropna=False)
    out = g.agg(n_items=("item_id", "nunique"), kl=("kl", "mean"), first_ok=("first_ok", "mean"),
                first_ok_e0=("first_ok_e0", "mean"), d_first_ok=("d_first_ok", "mean"), d_gold_lp=("d_gold_lp", "mean"),
                top1=("top1", "mean"), lp_read=("lp_read", "mean"), bytes_stored=("bytes_stored", "mean"),
                bytes_read=("bytes_read", "mean"), bits_per_value=("bits_per_value", "first")).reset_index()
    return out


def tests(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """09 §4.2 第 1 條：每一格（格式, 佔比），讀法 A，X ∈ STRUCT 對隨機平均（有幾個 seed 用幾個）。"""
    A = df[(df["readout"] == "A") & df["policy"].isin(STRUCT + RAND)]
    items = sorted(A["item_id"].unique())
    n = len(items)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(N_BOOT, n))       # 共用的重抽樣（所有檢定用同一組，結果可重現）
    rows = []
    for (fmt, frac), cell in A.groupby(["fmt", "frac"]):
        kl = cell.pivot_table(index="item_id", columns="policy", values="kl").reindex(items)
        fo = cell.pivot_table(index="item_id", columns="policy", values="first_ok").reindex(items)
        rs = [r for r in RAND if r in kl.columns]
        if not rs:
            continue
        kl_r = kl[rs].mean(axis=1)                   # 每題：隨機 seed 的平均
        fo_r = fo[rs].mean(axis=1)
        best_seed = min(kl[r].mean() for r in rs)
        for x in STRUCT:
            if x not in kl.columns:
                continue
            d = (kl_r - kl[x]).to_numpy()
            ok = ~np.isnan(d)
            if ok.sum() < n:                         # 有題缺列 → 只用齊的題（照實記 n_used）
                d = d[ok]
            m = len(d)
            boot = d[idx[:, :m] % m].mean(axis=1) if m else np.array([np.nan])
            lo, hi = np.percentile(boot, [2.5, 97.5]) if m else (np.nan, np.nan)
            delta = float(d.mean()) if m else np.nan
            rows.append(dict(fmt=fmt, frac=frac, policy=x, n_used=m, kl_x=float(kl[x].mean()), kl_rand=float(kl_r.mean()),
                             kl_best_seed=float(best_seed), delta=delta, ci_lo=float(lo), ci_hi=float(hi),
                             rel_delta=delta / float(kl_r.mean()) if kl_r.mean() > 0 else np.nan,
                             first_ok_x=float(fo[x].mean()), first_ok_rand=float(fo_r.mean()),
                             c1_ci=bool(lo > 0), c2_beats_best_seed=bool(kl[x].mean() < best_seed),
                             c3_first_ok=bool(fo[x].mean() >= fo_r.mean())))
    t = pd.DataFrame(rows)
    if len(t):
        t["winner"] = t["c1_ci"] & t["c2_beats_best_seed"] & t["c3_first_ok"]
    return t


def pick(t: pd.DataFrame, agg: pd.DataFrame) -> dict:
    res = {"rule": "09 §4.2（2026-10-15 22:50 UTC 寫死）", "finalists": [], "frontier": [], "controls": []}
    win = t[t["winner"]].sort_values("rel_delta", ascending=False) if len(t) else t
    per_fmt = {}
    for _, r in win.iterrows():
        if len(res["finalists"]) >= MAX_FINAL:
            break
        if per_fmt.get(r["fmt"], 0) >= MAX_PER_FMT:
            continue
        per_fmt[r["fmt"]] = per_fmt.get(r["fmt"], 0) + 1
        res["finalists"].append({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in
                                 r[["fmt", "frac", "policy", "delta", "ci_lo", "ci_hi", "rel_delta", "kl_x", "kl_rand",
                                    "first_ok_x", "first_ok_rand"]].items()})
    res["n_winners_total"] = int(len(win))
    if not res["finalists"]:
        res["note"] = "海選沒有任何格子勝出（09 §4.2 第 5 條）：複賽只做前緣點"
    A = agg[(agg["readout"] == "A") & (agg["policy"] != "ORC") & (agg["policy"] != "E0")]
    fr = A[A["d_first_ok"] > FRONTIER_DROP].sort_values(["bytes_stored", "kl"])
    for _, r in fr.head(N_FRONTIER).iterrows():
        res["frontier"].append(dict(fmt=r["fmt"], policy=r["policy"], frac=float(r["frac"]),
                                    bytes_stored=float(r["bytes_stored"]), d_first_ok=float(r["d_first_ok"]), kl=float(r["kl"])))
    cells = {(f["fmt"], f["frac"]) for f in res["finalists"]} | {(f["fmt"], f["frac"]) for f in res["frontier"]}
    for fmt, frac in sorted(cells):
        res["controls"].append(dict(fmt=fmt, frac=frac, policies=RAND, endpoint="ALL"))
    return res


def figs(agg: pd.DataFrame, model: str, out_dir: str, src: str):
    import matplotlib.pyplot as plt
    import m10_figs as MF
    A = agg[agg["readout"] == "A"]
    fmts = [f for f in A["fmt"].unique() if f]
    color = {p: MF.SLOTS[i] for i, p in enumerate(STRUCT)}
    nc = 4
    nr = (len(fmts) + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(3.1 * nc, 2.3 * nr), squeeze=False, sharex=True)
    for ax, fmt in zip(axes.flat, fmts):
        d = A[A["fmt"] == fmt]
        rr = d[d["policy"].isin(RAND)].groupby("frac")["kl"].agg(["min", "max", "mean"])
        ax.fill_between(rr.index * 100, rr["min"], rr["max"], color=MF.GRID, lw=0, label="隨機 5 個 seed 的範圍")
        ax.plot(rr.index * 100, rr["mean"], color=MF.MUTED, lw=1.2, label="隨機平均")
        for p in STRUCT:
            s = d[d["policy"] == p].sort_values("frac")
            if len(s):
                ax.plot(s["frac"] * 100, s["kl"], color=color[p], lw=1.4, label=p)
        o = d[d["policy"] == "ORC"].sort_values("frac")
        if len(o):
            ax.plot(o["frac"] * 100, o["kl"], color=MF.INK, lw=1.0, ls="--", label="ORC（作弊上限）")
        ax.set_yscale("log")
        ax.set_title(fmt)
    for ax in axes.flat[len(fmts):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("低精度佔比（%）")
    for ax in axes[:, 0]:
        ax.set_ylabel("KL（對 E0，log）")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, l, loc="upper center", ncol=min(len(l), 5), fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{model}：海選（探索、未校正多重比較）——讀法 A，KL 對低精度佔比", y=1.01)
    MF.source_note(fig, src)
    MF.save(fig, os.path.join(out_dir, f"sweep_kl_{model}.png"))
    plt.close(fig)
    # 格式之間：同樣的平均位元／值，誰的 KL 低（隨機平均那條線；每個家族一格，避免超過 8 色）
    fam = [("8 bit", ["fp8_head", "fp8_token", "fp8e5_head", "fp8k0_head", "fp8k0_token", "int8_token"]),
           ("4 bit", ["int4_token", "int4_kivi", "int4h_token", "mxfp4", "nvfp4", "kv-int4_kivi-int2_kivi"]),
           ("2–6 bit 混合", ["int2_kivi", "int2h_g32", "kv-int8_token-int4_kivi"])]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    R = A[A["policy"].isin(RAND)].groupby(["fmt", "frac"]).agg(kl=("kl", "mean"), bytes=("bytes_stored", "mean")).reset_index()
    for ax, (name, members) in zip(axes, fam):
        k = 0
        for f in members:
            s = R[R["fmt"] == f].sort_values("frac")
            if not len(s):
                continue
            ax.plot(s["bytes"] / 2**30, s["kl"], color=MF.SLOTS[k], lw=1.4, marker="o", ms=2.5, label=f)
            k += 1
        ax.set_yscale("log")
        ax.set_title(name)
        ax.set_xlabel("64 個 block 實際存的 GiB")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("KL（隨機放法的平均，log）")
    fig.suptitle(f"{model}：同樣的記憶體，哪種格式誤差小（海選、探索）", y=1.02)
    MF.source_note(fig, src)
    MF.save(fig, os.path.join(out_dir, f"sweep_bytes_{model}.png"))
    plt.close(fig)


def tables(agg, t, res, model, out_dir, src):
    L = [f"## {model} 大掃描海選（探索、未校正多重比較；09 §4.2 的規則挑選手）", "", f"資料：{src}", ""]
    L += ["### 選手", ""]
    if res["finalists"]:
        L += ["| 格式 | 佔比 | 放法 | Δ（隨機−X 的 KL） | 95% CI | 相對 | first_ok X／隨機 |", "|:--|--:|:--|--:|:--|--:|:--|"]
        for f in res["finalists"]:
            L.append(f"| {f['fmt']} | {f['frac']:.0%} | {f['policy']} | {f['delta']:.4g} | [{f['ci_lo']:.3g}, {f['ci_hi']:.3g}] | "
                     f"{f['rel_delta']:.1%} | {f['first_ok_x']:.3f}／{f['first_ok_rand']:.3f} |")
    else:
        L.append(res.get("note", "（無）"))
    L += ["", f"勝出的格子總數：{res['n_winners_total']}（三個條件都過的；每個模型最多取 {MAX_FINAL} 個、同格式最多 {MAX_PER_FMT} 個）", ""]
    L += ["### 前緣點（first_ok 比 E0 掉不到 2 個百分點、位元組最少）", "", "| 格式 | 放法 | 佔比 | GiB | first_ok 差 | KL |",
          "|:--|:--|--:|--:|--:|--:|"]
    for f in res["frontier"]:
        L.append(f"| {f['fmt']} | {f['policy']} | {f['frac']:.0%} | {f['bytes_stored'] / 2**30:.2f} | {f['d_first_ok']:+.3f} | {f['kl']:.4g} |")
    L += ["", "### 各格式在 25%／50%／75% 的 KL（讀法 A；隨機平均 vs 最好的結構化放法）", "",
          "| 格式 | 位元／值 | 25% 隨機 | 25% 最好 | 50% 隨機 | 50% 最好 | 75% 隨機 | 75% 最好 |", "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    A = agg[agg["readout"] == "A"]
    for fmt in [f for f in A["fmt"].unique() if f]:
        d = A[A["fmt"] == fmt]
        cells = []
        for fr in (0.25, 0.5, 0.75):
            x = d[np.isclose(d["frac"], fr)]
            r = x[x["policy"].isin(RAND)]["kl"].mean()
            s = x[x["policy"].isin(STRUCT)].sort_values("kl").head(1)
            cells += [f"{r:.3g}", f"{s['kl'].iloc[0]:.3g}（{s['policy'].iloc[0]}）" if len(s) else "—"]
        L.append(f"| {fmt} | {d['bits_per_value'].iloc[0]:.2f} | " + " | ".join(cells) + " |")
    open(os.path.join(out_dir, f"tables_{model}.md"), "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    df = load(a.csv, a.model)
    src = f"{len(a.csv)} 個 CSV；run_id {', '.join(sorted(df['run_id'].astype(str).unique()))[:120]}；{df['item_id'].nunique()} 題"
    agg = aggregate(df)
    agg.to_csv(os.path.join(a.out_dir, f"agg_{a.model}.csv"), index=False)
    t = tests(df)
    t.to_csv(os.path.join(a.out_dir, f"tests_{a.model}.csv"), index=False)
    res = pick(t, agg)
    res.update(model=a.model, csv=a.csv, n_items=int(df["item_id"].nunique()),
               run_ids=sorted(df["run_id"].astype(str).unique()))
    json.dump(res, open(os.path.join(a.out_dir, f"finalists_{a.model}.json"), "w"), indent=1, ensure_ascii=False, default=float)
    tables(agg, t, res, a.model, a.out_dir, src)
    figs(agg, a.model, a.out_dir, src)
    print(json.dumps({k: res[k] for k in ("n_winners_total", "finalists", "frontier")}, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
