"""analyze.py — 彙總 run_cell 的輸出：每個 ρ 的 regret（9 個值＝3 π × 3 seed，t 分布 95% CI）、相對 LRU、配對差、排名。
**合成資料、方法開發，不是結論。**

  python -I code/p3_lossdev/analyze.py A [B C D E ...] [--tag placeholder]
輸出：results/p3_lossdev/<exp>_summary.csv（有 run_id、ts）與 <exp>_summary.md。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402


def tci(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    m = float(x.mean())
    if n < 2:
        return m, np.nan, np.nan, n
    h = float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))
    return m, m - h, m + h, n


def load(exp, tag=""):
    fn = os.path.join(C.RES, f"{exp}{('_' + tag) if tag else ''}.csv")
    df = pd.read_csv(fn)
    if "error" in df:
        err = df[df["error"].notna()]
    else:
        err = df.iloc[:0]
    df = df[df["C_total"].notna()].copy()
    key = ["rho", "pi", "seed", "lam"]
    lru = df[df["method"] == "lru"].set_index(key)["C_total"].rename("C_lru")
    df = df.join(lru, on=key)
    df["rel_vs_lru"] = df["C_total"] / df["C_lru"] - 1
    den = df["C_lru"] - df["C_oracle"]
    df["gap_closed"] = np.where(den > 0, (df["C_lru"] - df["C_total"]) / den, np.nan)
    return df, err, fn


def summarize(df, lam=C.LAMBDA_MAIN, ref=None):
    d = df[df["lam"] == lam]
    rows = []
    for (m, rho), g in d.groupby(["method", "rho"]):
        r = dict(method=m, rho=rho, lam=lam)
        for col in ("regret", "rel_vs_lru", "gap_closed"):
            mu, lo, hi, n = tci(g[col])
            r.update({col: mu, f"{col}_lo": lo, f"{col}_hi": hi})
        r["n"] = n
        for col in ("auc_d1", "auc_d2", "cindex_d1", "brier_d1", "fit_s", "us_batch1", "us_per_row_b1024", "model_bytes", "n_viol", "frac_int2_d1"):
            if col in g:
                r[col] = float(np.nanmedian(g[col])) if col in ("us_batch1", "us_per_row_b1024", "model_bytes") else float(np.nanmean(g[col]))
        rows.append(r)
    s = pd.DataFrame(rows)
    # 每個 ρ：和該 ρ 最好的方法（regret 最小）配對比
    meth = d[d["kind"] == "method"]
    out = []
    for rho, g in s.groupby("rho"):
        gm = g[g["method"].isin(meth["method"].unique())]
        if len(gm) == 0:
            continue
        best = gm.sort_values("regret").iloc[0]["method"]
        pv = d[d["rho"] == rho].pivot_table(index=["pi", "seed"], columns="method", values="regret")
        for m in g["method"]:
            dd = pv[m] - pv[best] if (m in pv and best in pv) else pd.Series(dtype=float)
            mu, lo, hi, n = tci(dd)
            out.append(dict(method=m, rho=rho, best_in_rho=best, diff_vs_best=mu, diff_vs_best_lo=lo, diff_vs_best_hi=hi))
            if ref is not None and ref in pv and m in pv:
                mu, lo, hi, n = tci(pv[m] - pv[ref])
                out[-1].update(diff_vs_ref=mu, diff_vs_ref_lo=lo, diff_vs_ref_hi=hi, ref=ref)
    s = s.merge(pd.DataFrame(out), on=["method", "rho"], how="left")
    s["tie_with_best"] = (s["diff_vs_best_lo"] <= 0) | (s["diff_vs_best"] <= 0.01)
    s["rank_in_rho"] = s.groupby("rho")["regret"].rank(method="min")
    return s


def stable_table(s):
    """07 §2.9：五個 ρ 都和最好的打平（CI 含 0 或差 ≤ 1 pp），且 ρ＝0 時相對 LRU ≤ +1%。"""
    rows = []
    for m, g in s.groupby("method"):
        g = g.set_index("rho")
        tie_all = bool(g["tie_with_best"].all()) and len(g) == 5
        r0 = g.loc[0.0, "rel_vs_lru"] if 0.0 in g.index else np.nan
        r0hi = g.loc[0.0, "rel_vs_lru_hi"] if 0.0 in g.index else np.nan
        rows.append(dict(method=m, n_rho_tied=int(g["tie_with_best"].sum()), tie_all_rho=tie_all, rel_vs_lru_rho0=r0,
                         rel_vs_lru_rho0_hi=r0hi, stable=tie_all and (r0 <= 0.01), much_worse_than_lru_rho0=bool(r0 >= 0.05),
                         mean_rank=float(g["rank_in_rho"].mean())))
    return pd.DataFrame(rows).sort_values("mean_rank")


def fmt(x, p=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{p}f}"


def md_tables(exp, s, st, src, run_ids, err, lam=C.LAMBDA_MAIN):
    L = [f"# {exp} 彙總（λ={lam}）— 合成資料、方法開發，不是結論", "",
         f"來源：`{os.path.relpath(src, C.REPO)}`；run_id：{', '.join(run_ids)}。每格＝3 π × 3 seed＝9 個值的平均 [t 分布 95% CI]。", ""]
    rhos = sorted(s["rho"].unique())
    order = st["method"].tolist()
    base = [m for m in s["method"].unique() if m not in order]
    L.append("## regret（(C−C_oracle)/C_oracle，越小越好）")
    L.append("| 方法 | " + " | ".join(f"ρ={r}" for r in rhos) + " |")
    L.append("|:--|" + "--:|" * len(rhos))
    for m in order + sorted(base):
        g = s[s["method"] == m].set_index("rho")
        cells = []
        for r in rhos:
            if r in g.index:
                x = g.loc[r]
                star = "**" if x["tie_with_best"] else ""
                cells.append(f"{star}{fmt(x['regret'])}{star} [{fmt(x['regret_lo'])}, {fmt(x['regret_hi'])}]")
            else:
                cells.append("—")
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L += ["", "粗體＝和該 ρ 最好的方法配對差的 95% CI 含 0（或差 ≤ 1 個百分點）。", ""]
    L.append("## 相對 LRU 的成本（C/C_LRU − 1，負＝比 LRU 省）")
    L.append("| 方法 | " + " | ".join(f"ρ={r}" for r in rhos) + " |")
    L.append("|:--|" + "--:|" * len(rhos))
    for m in order + sorted(base):
        g = s[s["method"] == m].set_index("rho")
        L.append(f"| {m} | " + " | ".join((f"{fmt(g.loc[r, 'rel_vs_lru'] * 100, 1)}% [{fmt(g.loc[r, 'rel_vs_lru_lo'] * 100, 1)}, {fmt(g.loc[r, 'rel_vs_lru_hi'] * 100, 1)}]"
                                              if r in g.index else "—") for r in rhos) + " |")
    L += ["", "## 穩定度（§2.9）", "| 方法 | 打平最好的 ρ 數 | ρ＝0 相對 LRU | 穩定 | ρ＝0 比 LRU 差 ≥5% | 平均名次 |", "|:--|--:|--:|:--|:--|--:|"]
    for _, r in st.iterrows():
        L.append(f"| {r['method']} | {r['n_rho_tied']}/5 | {fmt(r['rel_vs_lru_rho0'] * 100, 1)}% (CI 上界 {fmt(r['rel_vs_lru_rho0_hi'] * 100, 1)}%) | "
                 f"{'是' if r['stable'] else '否'} | {'是' if r['much_worse_than_lru_rho0'] else '否'} | {fmt(r['mean_rank'], 1)} |")
    L += ["", "## 診斷（不參與排名；9 個值的平均）", "| 方法 | ρ | AUC D1 | AUC D2 | C-index D1 | Brier D1 | D1 壓成 INT2 的比例 | 讀到不過關格式的次數 |",
          "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for m in order:
        for r in rhos:
            g = s[(s["method"] == m) & (s["rho"] == r)]
            if len(g):
                x = g.iloc[0]
                L.append(f"| {m} | {r} | {fmt(x.get('auc_d1'))} | {fmt(x.get('auc_d2'))} | {fmt(x.get('cindex_d1'))} | {fmt(x.get('brier_d1'))} | "
                         f"{fmt(x.get('frac_int2_d1'))} | {fmt(x.get('n_viol'), 1)} |")
    if "us_batch1" in s and s["us_batch1"].notna().any():
        L += ["", "## 推論時間與大小（CPU 單執行緒；格子間的中位數）", "| 模型 | batch=1 µs | batch=1024 每列 µs | 模型 bytes | 訓練秒數（D1＋D2，2 執行緒） |", "|:--|--:|--:|--:|--:|"]
        for m in order:
            g = s[s["method"] == m]
            if g["us_batch1"].notna().any():
                L.append(f"| {m} | {fmt(g['us_batch1'].median(), 1)} | {fmt(g['us_per_row_b1024'].median(), 3)} | {int(g['model_bytes'].median()):,} | {fmt(g['fit_s'].median(), 1)} |")
    if len(err):
        L += ["", "## 失敗（完整錯誤在 CSV 的 traceback 欄）"]
        for _, r in err.iterrows():
            L.append(f"- {r['method']} ρ={r['rho']} π={r['pi']} seed={r['seed']}: `{r['error']}`")
    return "\n".join(L) + "\n"


def per_pi(df, lam=C.LAMBDA_MAIN):
    d = df[df["lam"] == lam]
    rows = []
    for (m, rho, pi), g in d.groupby(["method", "rho", "pi"]):
        mu, lo, hi, n = tci(g["regret"])
        r2, lo2, hi2, _ = tci(g["rel_vs_lru"])
        rows.append(dict(method=m, rho=rho, pi=pi, regret=mu, regret_lo=lo, regret_hi=hi, rel_vs_lru=r2, rel_vs_lru_lo=lo2, rel_vs_lru_hi=hi2, n=n))
    return pd.DataFrame(rows)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tag = ""
    if "--tag" in sys.argv:
        tag = sys.argv[sys.argv.index("--tag") + 1]
        args = [a for a in args if a != tag]
    refs = dict(A="bce_6h", B="lgbm", C="lgbm_n200_l63", D="feat_all", E="lgbm")
    run_id = os.environ.get("RUN_ID", "")
    for exp in args:
        df, err, src = load(exp, tag)
        run_ids = sorted(df["run_id"].unique())
        lams = sorted(df["lam"].unique())
        out_csv = []
        md = []
        for lam in lams:
            s = summarize(df, lam, ref=refs.get(exp))
            st = stable_table(s[s["method"].isin(df[df["kind"] == "method"]["method"].unique())])
            s["analysis_run_id"] = run_id
            s["source_run_ids"] = ";".join(run_ids)
            s["ts"] = C.now_iso()
            s["run_id"] = run_id or ";".join(run_ids)
            out_csv.append(s)
            md.append(md_tables(exp, s, st, src, run_ids, err if lam == C.LAMBDA_MAIN else err.iloc[:0], lam))
        name = f"{exp}{('_' + tag) if tag else ''}"
        pd.concat(out_csv).to_csv(os.path.join(C.RES, f"{name}_summary.csv"), index=False)
        pp = per_pi(df)
        pp["run_id"] = run_id or ";".join(run_ids)
        pp["ts"] = C.now_iso()
        pp.to_csv(os.path.join(C.RES, f"{name}_per_pi.csv"), index=False)
        open(os.path.join(C.RES, f"{name}_summary.md"), "w", encoding="utf-8").write("\n\n".join(md))
        print(f"[analyze] {name}: {len(df)} rows, errors={len(err)} → {name}_summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
