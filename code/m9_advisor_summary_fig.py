"""m9_advisor_summary_fig.py — 一張圖總結第一階段：什麼會改變使用者的等待時間？只讀既有的 CSV，不產生新的量測。

每一列是一個因素；每個點是一組 GPU 實測比較，橫軸是「兩邊的等待時間差多少（%）」。
- 有沒有 CPU 層：A1，32K、GPU 閒，全部都存（只有 SSD）vs 全放 CPU；本地 SSD、NFS 各一點
- GPU 忙不忙：A1，32K，GPU 閒 vs 忙 2.1 倍；全部都存、全放 CPU × 本地 SSD、NFS
- CPU 容量夠不夠：B 主設定，CPU 容量 25% vs 100%；有 CPU 層的 6 個策略
- 少寫前段：A1，先寫再丟 vs 一開始不寫，全部 48 組
- 寫入時決定 vs 延後決定：同一條規則，寫入時做 vs 滿了才做（B：S5 vs S4+；08：S5L vs S4B；第 1 輪：S5L vs S4B）

用法：python code/m9_advisor_summary_fig.py
輸出：docs/phase1_20261008/report/figs/advisor_summary.png；results/m9_followup/advisor_summary_data.csv
"""
import datetime
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R7 = os.path.join(ROOT, "results", "m7_write_policy_mi300x")
RX = os.path.join(ROOT, "results", "m7_explore_mi300x")
FIG = os.path.join(ROOT, "docs", "phase1_20261008", "report", "figs", "advisor_summary.png")
OUT = os.path.join(ROOT, "results", "m9_followup", "advisor_summary_data.csv")
RID = os.environ.get("RUN_ID", "no-run-id")
TS = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

FP = "/mlsteam/data/tiara/fonts/NotoSansCJKtc-Regular.otf"
font_manager.fontManager.addfont(FP)
plt.rcParams.update({"font.family": [font_manager.FontProperties(fname=FP).get_name(), "DejaVu Sans"],
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#0b0b0b",
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
C_FACT, C_MINE, INK, INK2 = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
NOISE = 6.2   # C4 雜訊底線 δ（07 §3.4）


def collect():
    pts = []
    a = pd.read_csv(os.path.join(R7, "a1_share.csv"))
    a = a[(a.L == 32768) & (a.gap == np.inf) & (a.contaminated_gpu == 0)]
    m = a.groupby(["ssd_dev", "gpu_state", "strategy"]).ttft.median()
    rid_a1 = a.run_id.iloc[0]
    for dev in ["local", "nfs"]:
        pts.append(("cpu", 100 * abs(m[dev, "idle", "CPUall"] / m[dev, "idle", "S0"] - 1), f"{dev}；全部都存 vs 全放 CPU", rid_a1))
        for s in ["S0", "CPUall"]:
            pts.append(("gpu", 100 * abs(m[dev, "busy2", s] / m[dev, "idle", s] - 1), f"{dev}；{s}；閒 vs 忙 2.1×", rid_a1))
    b = pd.read_csv(os.path.join(R7, "summary_b.csv"))
    bm = b[(b.io_model == "share") & (b.cpu_tier == "harness") & (b.ssd_dev == "nfs") & (b.gap_s == 0) & (b.ssd_frac == 1.0)
           & (b.wl_seed == 0)].pivot_table(index="strategy", columns="cpu_frac", values="median")
    for s in ["S1", "S2b", "S4", "S4+", "S4+P", "S5"]:
        pts.append(("cap", 100 * abs(bm.loc[s, 0.25] / bm.loc[s, 1.0] - 1), f"B 主設定；{s}；25% vs 100%", "summary_b.csv"))
    p = pd.read_csv(os.path.join(R7, "summary_a1_pairs.csv"))
    for _, r in p[(p.a == "S3") & (p.b == "S2a")].iterrows():
        pts.append(("less", abs(r.diff_pct), f"{r.ssd_dev}；{r.gpu_state}；gap {r.gap}；{r.L}", rid_a1))
    bs = b[b.io_model == "share"]
    for key, g in bs.groupby(["cpu_tier", "ssd_dev", "gap_s", "ssd_frac", "wl_seed", "cpu_frac"]):
        mm = g.set_index("strategy")["median"]
        if "S5" in mm and "S4+" in mm:
            pts.append(("when", 100 * abs(mm["S4+"] - mm["S5"]) / mm["S4+"], "B；S5 vs S4+；" + "／".join(map(str, key)), "summary_b.csv"))
    v = pd.read_csv(os.path.join(R7, "b2_verdict.csv"))
    for _, r in v[(v.src == "gpu") & (v.strategy == "S5L")].iterrows():
        pts.append(("when", 100 * abs(r.gain_s4b), f"08；S5L vs S4B；{r.workload}／{r.release}／s{r.wl_seed}／{r.cpu_frac}", r.run_id))
    r1 = pd.read_csv(os.path.join(RX, "r1_gpu_verdict.csv"))
    for _, r in r1[r1.stat == "median"].iterrows():
        pts.append(("when", 100 * abs(r.g_S4B), f"第 1 輪；S5L vs S4B；seed {r.wl_seed}", r.run_id))
    return pd.DataFrame(pts, columns=["factor", "abs_pct", "detail", "src_run_id"])


def main():
    d = collect()
    rows = [("cpu", "有沒有 CPU 層\n（全放 SSD vs 全放 CPU）", C_FACT),
            ("gpu", "GPU 忙不忙\n（閒 vs 忙 2.1 倍）", C_FACT),
            ("cap", "CPU 容量夠不夠\n（工作集的 25% vs 100%）", C_FACT),
            ("less", "少寫前段\n（先寫再丟 vs 一開始不寫）", C_MINE),
            ("when", "寫入時決定 vs 延後決定\n（同一條規則，寫入時做 vs 滿了才做）", C_MINE)]
    fig, ax = plt.subplots(figsize=(11, 6.6))
    ax.axvspan(0, NOISE, color="#e6e5e0", zorder=0, lw=0)
    ax.text(NOISE / 2, len(rows) - 0.35, f"重跑波動\n（{NOISE}%）", ha="center", va="bottom", fontsize=8, color=INK2)
    rng = np.random.default_rng(0)
    for i, (f, label, c) in enumerate(rows):
        y = len(rows) - 1 - i
        x = d[d.factor == f].abs_pct.values
        jit = rng.uniform(-0.18, 0.18, len(x))
        ax.scatter(x, y + jit, s=34, color=c, alpha=0.85, edgecolor="#fcfcfb", linewidth=0.8, zorder=3)
        if f in ("less", "when"):
            txt = f"{len(x)} 組；中位數 {np.median(x):.1f}%，{(x <= NOISE).mean() * 100:.0f}% 的組在波動內"
        else:
            txt = f"{len(x)} 組；差 {x.min():.0f}–{x.max():.0f}%"
        ax.text(45 if f in ("less", "when") else max(x.max(), NOISE) + 2.5, y, txt, va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)), [r[1] for r in rows][::-1])
    ax.tick_params(axis="y", length=0)
    for lab, (_, _, c) in zip(ax.get_yticklabels(), rows[::-1]):
        lab.set_color(INK if c == C_FACT else "#b44a1c")
    ax.set_xlim(0, 150)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel("兩邊的等待時間（TTFT）差多少（%）")
    ax.grid(axis="x", color="#e6e5e0", lw=0.6, zorder=0)
    ax.axhline(1.5, color="#c9c8c2", lw=0.8, ls="--")
    ax.text(149, 1.55, "↑ 會大幅改變等待時間的因素", ha="right", va="bottom", fontsize=8.5, color=C_FACT)
    ax.text(149, 1.45, "↓ 寫入時的決定（本研究在測的）：幾乎都在波動內", ha="right", va="top", fontsize=8.5, color="#b44a1c")
    ax.set_title("在 MI300X＋Llama-3.1-8B 上，什麼會改變使用者的等待時間？", loc="left", fontsize=12.5, color=INK, pad=26)
    ax.text(0, 1.012, "結論：寫入時決定放 CPU 還是 SSD，影響比重跑波動還小；真正的差別來自有沒有 CPU 層、容量夠不夠、GPU 忙不忙。",
            transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")
    fig.text(0.01, 0.015,
             "每個點是一組 GPU 實測比較（6 次的中位數）。資料：A1（run 20261008-134526-m7-a1）、實驗 B（summary_b.csv）、08 消融（b2_verdict.csv）、"
             "第 1 輪（r1_gpu_verdict.csv）；畫圖程式 code/m9_advisor_summary_fig.py", fontsize=7.3, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG, dpi=200)
    d.insert(0, "ts", TS)
    d.insert(0, "run_id", RID)
    d.to_csv(OUT, index=False)
    print(d.groupby("factor").abs_pct.describe().round(2))


if __name__ == "__main__":
    main()
