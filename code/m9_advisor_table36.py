"""m9_advisor_table36.py — PDF 附錄表 A2：表 6 背後的 36 組 GPU 比較，逐列列出。只讀既有的 CSV。

每一列：寫入時決定的策略 vs 最好的簡單策略 vs 延後版（同一條規則、CPU 滿了才做），回來請求 TTFT 中位數。
- 實驗 B（第一階段，15 組）：summary_b.csv，share I/O 模型；寫入時＝S5，延後版＝S4+
- 08 消融（16 組）：b2_verdict.csv（src=gpu）＋ b2_summary.csv 的 gpu_median；延後版＝S4B
- 第 1 輪（5 組）：r1_gpu_verdict.csv（stat=median）；寫入時＝S5L，延後版＝S4B
差距＝（對手 − 寫入時）÷ 對手，正值＝寫入時決定比較快。

用法：python code/m9_advisor_table36.py
輸出：results/m9_followup/advisor_table36.csv；docs/phase1_20261008/advisor_update_20261012/table36_rows.tex（\\input 用）
"""
import datetime
import os

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R7 = os.path.join(ROOT, "results", "m7_write_policy_mi300x")
RX = os.path.join(ROOT, "results", "m7_explore_mi300x")
OUT = os.path.join(ROOT, "results", "m9_followup", "advisor_table36.csv")
TEX = os.path.join(ROOT, "docs", "phase1_20261008", "advisor_update_20261012", "table36_rows.tex")
RID = os.environ.get("RUN_ID", "no-run-id")
TS = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
SIMPLE_B = ["S1", "S2b", "S4", "S4+", "S4+P", "S4L"]
DEV = {"nfs": "NFS", "local": "本地"}


def rows_b():
    b = pd.read_csv(os.path.join(R7, "summary_b.csv"))
    b = b[b.io_model == "share"]
    out = []
    for key, g in b.groupby(["cpu_tier", "ssd_dev", "gap_s", "ssd_frac", "wl_seed", "cpu_frac"]):
        tier, dev, gap, sfrac, seed, cfrac = key
        m = g.set_index("strategy")["median"]
        if "S5" not in m or "S4+" not in m:
            continue
        simple = m[[s for s in SIMPLE_B if s in m]]
        extra = []
        if tier != "harness":
            extra.append("CPU 3.69")
        if gap != 0:
            extra.append(f"間隔 {gap:g}s")
        if sfrac != 1.0:
            extra.append(f"SSD {sfrac:.0%}")
        setting = f"對話型、{DEV[dev]}、{cfrac:.0%}、s{seed}" + ("、" + "、".join(extra) if extra else "")
        out.append(dict(exp="B", setting=setting, wt="S5", t_wt=m["S5"], best=simple.idxmin(), t_best=simple.min(),
                        twin="S4+", t_twin=m["S4+"], src="summary_b.csv"))
    return out


def rows_08():
    v = pd.read_csv(os.path.join(R7, "b2_verdict.csv"))
    s = pd.read_csv(os.path.join(R7, "b2_summary.csv"))
    k = ["workload", "release", "cpu_gibps", "ssd_dev", "wl_seed", "cpu_frac"]
    out = []
    for _, r in v[v.src == "gpu"].iterrows():
        g = s[(s[k] == r[k]).all(axis=1)].set_index("strategy")["gpu_median"]
        setting = f"{ {'chat': '對話型', 'doc': '文件型'}[r.workload]}、CPU {r.cpu_gibps:.3g}、{DEV[r.ssd_dev]}、{r.cpu_frac:.0%}、s{r.wl_seed}"
        out.append(dict(exp="08", setting=setting, wt=r.strategy, t_wt=g[r.strategy], best=r.best_nwt, t_best=g[r.best_nwt],
                        twin="S4B", t_twin=g["S4B"], src=f"b2_verdict.csv；{r.run_id}"))
    return out


def rows_r1():
    r1 = pd.read_csv(os.path.join(RX, "r1_gpu_verdict.csv"))
    out = []
    for _, r in r1[r1.stat == "median"].iterrows():
        out.append(dict(exp="第 1 輪", setting=f"文件型、CPU 11.6、本地、50%、s{r.wl_seed}", wt="S5L",
                        t_wt=r.t_S5L, best=r.best_nwt, t_best=r.t_nwt, twin="S4B", t_twin=r.t_S4B, src=f"r1_gpu_verdict.csv；{r.run_id}"))
    return out


def main():
    d = pd.DataFrame(rows_b() + rows_08() + rows_r1())
    d["g_best"] = 100 * (d.t_best - d.t_wt) / d.t_best
    d["g_twin"] = 100 * (d.t_twin - d.t_wt) / d.t_twin
    d["both5"] = (d.g_best >= 5) & (d.g_twin >= 5)
    assert len(d) == 36, len(d)
    d.insert(0, "ts", TS)
    d.insert(0, "run_id", RID)
    d.to_csv(OUT, index=False)
    lines = []
    for i, r in enumerate(d.itertuples(), 1):
        def pct(x):
            s = f"{x:+.1f}\\%".replace("-", "−")
            return f"\\textbf{{{s}}}" if x >= 5 else s
        lines.append(f"{i} & {r.exp} & {r.setting.replace('%', '\\%')} & {r.wt} & {r.best} & {r.t_wt:.3f} & {r.t_best:.3f} & {r.t_twin:.3f} & "
                     f"{pct(r.g_best)} & {pct(r.g_twin)} \\\\")
        if i < len(d) and d.exp.iloc[i] != r.exp:
            lines.append("\\midrule")
    open(TEX, "w").write("\n".join(lines) + "\n")
    print(d.groupby("exp", sort=False).agg(n=("exp", "size"), beat_best=("g_best", lambda x: int((x >= 5).sum())),
                                           beat_twin=("g_twin", lambda x: int((x >= 5).sum())), both=("both5", "sum")))


if __name__ == "__main__":
    main()
