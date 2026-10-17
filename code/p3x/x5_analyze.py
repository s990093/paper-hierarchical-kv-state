"""x5_analyze.py — X5 的判定（照 X5 §2.4，開跑前寫的判準）＋只報告的表。

輸入：results/p3_explore/x5/x5_cells.csv（＋_mapped）、<raw>/ttft.npz、trace（session id，bootstrap 用）。
輸出：x5_headroom.csv、x5_verdict.json、x5_bootstrap.csv、x5_actions_summary.csv、tables_x5.md（都在 out-dir）。
headroom＝(T_最好固定規則 − T_oracle) ÷ T_最好固定規則；T＝回來請求完整 TTFT 平均；
最好固定規則＝R1–R6 最快的；oracle＝OA–OD 最快的。
判定：
  有看頭：主設定（llama、k0、reg）下 ∃ trace、CPU、(慢層, GPU) headroom ≥ 15%
  死路：reg 下所有格子（兩模型、兩反量化、全部 CPU／慢層／GPU）headroom < 5%
  其他＝可能
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x5_trace as XT  # noqa: E402

RULES = ["R1", "R2", "R3", "R4", "R5", "R6"]
ORACLES = ["OA", "OB", "OC", "OD"]
PARTIAL = ["R2b", "R4b", "R6b"]
CELL = ["trace", "model", "cpu_GiB", "variant", "dq", "gpu", "slow"]
RID = os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-p3x5-analyze-adhoc"


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def headroom_table(d: pd.DataFrame) -> pd.DataFrame:
    out = []
    for key, g in d.groupby(CELL):
        g = g.set_index("policy")
        rr = g.loc[[p for p in RULES if p in g.index]]
        oo = g.loc[[p for p in ORACLES if p in g.index]]
        pp = g.loc[[p for p in PARTIAL if p in g.index]]
        br, bo = rr.mean_ttft.idxmin(), oo.mean_ttft.idxmin()
        Tr, To = rr.mean_ttft.min(), oo.mean_ttft.min()
        row = dict(zip(CELL, key))
        row.update(run_id_cells=g.run_id.iloc[0], cpu_pct_ws=g.cpu_pct_ws.iloc[0], cpu_pct_sess_ws=g.cpu_pct_sess_ws.iloc[0],
                   slow_GiBps=g.slow_GiBps.iloc[0], n_ret=int(g.n_ret.iloc[0]),
                   best_rule=br, best_oracle=bo, T_rule=Tr, T_oracle=To, headroom=(Tr - To) / Tr,
                   headroom_median=(rr.loc[br].median_ttft - oo.loc[bo].median_ttft) / rr.loc[br].median_ttft,
                   headroom_p90=(rr.loc[br].p90_ttft - oo.loc[bo].p90_ttft) / rr.loc[br].p90_ttft,
                   restore_rule=rr.loc[br].mean_restore, restore_oracle=oo.loc[bo].mean_restore,
                   headroom_restore=(rr.loc[br].mean_restore - oo.loc[bo].mean_restore) / rr.loc[br].mean_restore,
                   headroom_sum_all=(rr.loc[br].sum_all_ttft - oo.loc[bo].sum_all_ttft) / rr.loc[br].sum_all_ttft,
                   oracle_violation=bool((rr.mean_ttft < To - 1e-12).any()),
                   rules_faster_than_oracle=",".join(rr.index[rr.mean_ttft < To - 1e-12]),
                   rule_frac_int4_long=rr.loc[br].frac_int4_long, oracle_frac_int4_long=oo.loc[bo].frac_int4_long,
                   rule_state_mix=f"BF16 {rr.loc[br].frac_BF16:.2f}/INT4 {rr.loc[br].frac_INT4:.2f}/SLOW {rr.loc[br].frac_SLOW:.2f}",
                   oracle_state_mix=f"BF16 {oo.loc[bo].frac_BF16:.2f}/INT4 {oo.loc[bo].frac_INT4:.2f}/SLOW {oo.loc[bo].frac_SLOW:.2f}/DROP {oo.loc[bo].frac_DROP:.2f}")
        for p in RULES + ORACLES + PARTIAL + ["ODn"]:
            if p in g.index:
                row[f"T_{p}"] = g.loc[p].mean_ttft
        if len(pp):
            bp = pp.mean_ttft.idxmin()
            row["best_partial"] = bp
            row["T_partial"] = pp.mean_ttft.min()
            row["headroom_partial"] = (Tr - pp.mean_ttft.min()) / Tr
            row["share_of_headroom_from_binary_return_info"] = ((Tr - pp.mean_ttft.min()) / (Tr - To)) if Tr > To else np.nan
        out.append(row)
    return pd.DataFrame(out)


def load_trace(nm, rule):
    if nm.startswith("freeinference"):
        import x5_fi
        return x5_fi.load()
    return XT.load(nm, rule)


def bootstrap(h: pd.DataFrame, npz_path: str, B: int = 2000, seed: int = 0, rule: str = "subtree") -> pd.DataFrame:
    z = np.load(npz_path)
    rng = np.random.default_rng(seed)
    out = []
    trs = {}
    for nm in h.trace.unique():
        tr = load_trace(nm, rule)
        H = float(os.environ.get("X5_CENSOR_H", "0"))
        msk = tr["ret"].copy()
        if H > 0:
            tc = float(tr["T"][-1]) - H
            prevT = np.where(tr["prev"] >= 0, tr["T"][np.maximum(tr["prev"], 0)], np.inf)
            msk &= prevT <= tc
        sid = tr["sid"][msk]
        u, inv = np.unique(sid, return_inverse=True)
        W = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=B).astype(np.float32)
        trs[nm] = (inv, len(u), W)
    for _, r in h.iterrows():
        inv, nu, W = trs[r.trace]
        base = f"{r.trace}|{r.model}|{r.cpu_GiB}|{r.variant}|{r.dq}|{r.gpu}|{r.slow}|"
        a = z[base + r.best_rule].astype(np.float64)
        o = z[base + r.best_oracle].astype(np.float64)
        sa = np.bincount(inv, weights=a, minlength=nu)
        so = np.bincount(inv, weights=o, minlength=nu)
        A, O = W @ sa, W @ so
        hb = (A - O) / A
        out.append(dict(run_id=RID, ts=ts(), **{k: r[k] for k in CELL}, best_rule=r.best_rule, best_oracle=r.best_oracle,
                        headroom=r.headroom, ci_lo=float(np.percentile(hb, 2.5)), ci_hi=float(np.percentile(hb, 97.5)),
                        n_sessions=nu, B=B))
    return pd.DataFrame(out)


def fmt_pct(x):
    return f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--raw-dir", required=True, help="x5_run 的 run 目錄（ttft.npz）")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--sfx", default="", help="'' 或 _strict 或 _fi")
    ap.add_argument("--rule", default="subtree")
    ap.add_argument("--variant", default="reg", help="主判定用的校正 variant（loadonly 敏感度用 loadonly）")
    a = ap.parse_args()
    sfx = a.sfx
    cp = os.path.join(a.out_dir, f"x5_cells{sfx}.csv")
    d = pd.read_csv(cp if os.path.exists(cp) else cp + ".gz")
    h = headroom_table(d)
    h.insert(0, "ts", ts())
    h.insert(0, "run_id", RID)
    h.to_csv(os.path.join(a.out_dir, f"x5_headroom{sfx}.csv"), index=False, float_format="%.6f")
    main_ = h[(h.model == "llama") & (h.dq == "k0") & (h.variant == a.variant)]
    reg = h[h.variant == a.variant]
    good = main_[main_.headroom >= 0.15]
    dead = bool((reg.headroom < 0.05).all())
    verdict = "有看頭" if len(good) else ("死路" if dead else "可能")
    exc = h[h.variant == "excess"]
    exc_main = exc[(exc.model == "llama") & (exc.dq == "k0")]
    if len(exc):
        verdict_excess = ("有看頭" if (exc_main.headroom >= 0.15).any() else ("死路" if (exc.headroom < 0.05).all() else "可能"))
    else:
        verdict_excess = "NOT_RUN"
    bs = bootstrap(main_, os.path.join(a.raw_dir, f"ttft{sfx}.npz"), B=a.B, rule=a.rule)
    bs.to_csv(os.path.join(a.out_dir, f"x5_bootstrap{sfx}.csv"), index=False, float_format="%.6f")
    vj = dict(run_id=RID, ts=ts(), verdict_registered=verdict, n_main_cells=len(main_), n_main_cells_ge15=int(len(good)),
              main_headroom_max=float(main_.headroom.max()), main_headroom_median=float(main_.headroom.median()),
              main_headroom_min=float(main_.headroom.min()),
              reg_all_cells=len(reg), reg_cells_ge5=int((reg.headroom >= 0.05).sum()), reg_cells_ge15=int((reg.headroom >= 0.15).sum()),
              reg_headroom_max=float(reg.headroom.max()),
              verdict_excess_variant_report_only=verdict_excess,
              excess_main_headroom_max=(float(exc_main.headroom.max()) if len(exc_main) else None),
              excess_main_cells_ge15=int((exc_main.headroom >= 0.15).sum()),
              oracle_violations=int(h.oracle_violation.sum()),
              main_cells_ge15_with_ci_lo_ge15=int(((bs.headroom >= 0.15) & (bs.ci_lo >= 0.15)).sum()),
              cells_ge15=good[CELL + ["headroom", "best_rule", "best_oracle"]].to_dict("records"))
    json.dump(vj, open(os.path.join(a.out_dir, f"x5_verdict{sfx}.json"), "w"), indent=1, ensure_ascii=False, default=str)
    # ---- 動作 × 距離回來的時間（只報告）
    ac = pd.read_csv(os.path.join(a.out_dir, f"x5_oracle_actions{sfx}.csv"))
    acm = ac[(ac.model == "llama") & (ac.dq == "k0") & (ac.variant == a.variant) & (ac.policy == "OD")]
    summ = acm.groupby(["trace", "cpu_GiB", "gpu", "slow", "time_to_return", "action"])["count"].sum().reset_index()
    summ.insert(0, "ts", ts())
    summ.insert(0, "run_id", RID)
    summ.to_csv(os.path.join(a.out_dir, f"x5_actions_summary{sfx}.csv"), index=False)
    # ---- 表
    L = []
    L.append(f"# X5 表（`code/p3x/x5_analyze.py`，run {RID}；cells run {h.run_id_cells.iloc[0]}）\n")
    L.append(f"判定（事先登記，主設定 llama／K0／reg）：**{verdict}**；excess 校正（只報告）：{verdict_excess}\n")
    for tr in ("conversation", "toolagent"):
        L.append(f"\n## 主設定 headroom — {tr}（Llama、原寫法反量化 K0、reg 校正；回來請求完整 TTFT 平均；括號＝session bootstrap 95% CI）\n")
        L.append("| CPU GiB (占工作集) | 慢層 | GPU | 最好規則 | T_規則 (s) | 最好 oracle | T_oracle (s) | headroom | 只算還原 | 中位數 | p90 | 部分 oracle（只知會不會回來） |")
        L.append("|--:|:--|:--|:--|--:|:--|--:|--:|--:|--:|--:|--:|")
        m = main_[main_.trace == tr].sort_values(["cpu_GiB", "slow", "gpu"])
        for _, r in m.iterrows():
            b = bs[(bs.trace == tr) & (bs.cpu_GiB == r.cpu_GiB) & (bs.slow == r.slow) & (bs.gpu == r.gpu)].iloc[0]
            L.append(f"| {r.cpu_GiB} ({r.cpu_pct_ws:.1f}%) | {r.slow} ({r.slow_GiBps:.2f}) | {r.gpu} | {r.best_rule} | {r.T_rule:.3f} | {r.best_oracle} | {r.T_oracle:.3f} | "
                     f"{fmt_pct(r.headroom)} [{fmt_pct(b.ci_lo)}, {fmt_pct(b.ci_hi)}] | {fmt_pct(r.headroom_restore)} | {fmt_pct(r.headroom_median)} | {fmt_pct(r.headroom_p90)} | "
                     f"{r.best_partial} {fmt_pct(r.headroom_partial)} |")
    L.append("\n## 全部格子的 headroom 範圍（只報告；variant × 模型 × 反量化 × trace）\n")
    L.append("| variant | 模型 | 反量化 | trace | 格子 | headroom 最小 | 中位 | 最大 | ≥15% 的格子 | ≥5% 的格子 | 最常見的最好規則 |")
    L.append("|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|:--|")
    for key, g in h.groupby(["variant", "model", "dq", "trace"]):
        L.append(f"| {' | '.join(map(str, key))} | {len(g)} | {fmt_pct(g.headroom.min())} | {fmt_pct(g.headroom.median())} | {fmt_pct(g.headroom.max())} | "
                 f"{int((g.headroom >= 0.15).sum())} | {int((g.headroom >= 0.05).sum())} | {g.best_rule.value_counts().index[0]} ({g.best_rule.value_counts().iloc[0]}) |")
    viol = h[h.oracle_violation]
    L.append(f"\n## oracle 比某個規則慢的格子：{len(viol)}\n")
    for _, r in viol.iterrows():
        L.append(f"- {r.variant}/{r.model}/{r.dq}/{r.trace}/{r.cpu_GiB}/{r.slow}/{r.gpu}: oracle {r.best_oracle} {r.T_oracle:.4f} > {r.rules_faster_than_oracle}")
    open(os.path.join(a.out_dir, f"tables_x5{sfx}.md"), "w").write("\n".join(L) + "\n")
    print(json.dumps(vj, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
