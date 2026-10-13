"""m9_advisor_breakdown.py — 給老師的「時間花在哪裡」比較圖（一張）。只讀既有的 A1 CSV，不產生新的量測。

Cake 還原時兩條線同時跑：GPU 從前面重算、I/O 從後面載入（CPU 層、SSD 層），兩條線都結束後再算新的 token。
每一列畫出這兩條線各花多久，就看得出等待時間卡在哪一邊。

用法：python code/m9_advisor_breakdown.py
輸出：docs/phase1_20261008/report/figs/advisor_breakdown.png；results/m9_followup/advisor_breakdown_data.csv
"""
import datetime
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = os.path.join(ROOT, "results", "m7_write_policy_mi300x", "a1_share.csv")
FIG = os.path.join(ROOT, "docs", "phase1_20261008", "report", "figs", "advisor_breakdown.png")
OUT = os.path.join(ROOT, "results", "m9_followup", "advisor_breakdown_data.csv")
RID = os.environ.get("RUN_ID", "no-run-id")
TS = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

FP = "/mlsteam/data/tiara/fonts/NotoSansCJKtc-Regular.otf"
font_manager.fontManager.addfont(FP)
plt.rcParams.update({"font.family": [font_manager.FontProperties(fname=FP).get_name(), "DejaVu Sans"],
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#0b0b0b",
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
C_GPU, C_SSD, C_CPU, C_NEW = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2"   # 參考調色盤前三格＋中性灰
INK, INK2 = "#0b0b0b", "#52514e"

STRATS = [("S0", "全部都存"), ("S2a", "先寫再丟（前段）"), ("S3", "一開始不寫（前段）"), ("S5", "依位置放（我的方法）")]
GPUS = [("idle", "GPU 閒"), ("busy2", "GPU 忙 2.1 倍")]


def main():
    a = pd.read_csv(SRC)
    q = a[(a.L == 32768) & (a.gap == np.inf) & (a.contaminated_gpu == 0) & (a.ssd_dev == "local")]
    g = q.groupby(["gpu_state", "strategy"]).agg(
        t_rec=("t_recompute", "median"), t_cpu=("t_load_cpu", "median"), t_ssd=("t_load_ssd", "median"),
        t_restore=("t_restore", "median"), t_new=("t_new", "median"), ttft=("ttft", "median"),
        meet=("meet", "median"), n_cpu=("n_load_cpu", "median"), n_ssd=("n_load_ssd", "median"),
        n=("ttft", "size"), src_run_id=("run_id", "first")).reset_index()

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    y, ticks, labels, rows = 0.0, [], [], []
    h = 0.30
    for gs, gname in GPUS:
        ax.text(-0.02, y + 0.15, gname, ha="right", va="center", fontsize=10.5, color=INK, fontweight="bold",
                transform=ax.get_yaxis_transform())
        y -= 0.55
        for s, sname in STRATS:
            r = g[(g.gpu_state == gs) & (g.strategy == s)].iloc[0]
            # 上：GPU 重算線；下：I/O 載入線（先 CPU 後 SSD）
            ax.barh(y + h / 2 + 0.02, r.t_rec, h, color=C_GPU, zorder=3)
            ax.barh(y - h / 2 - 0.02, r.t_cpu, h, color=C_CPU, zorder=3)
            ax.barh(y - h / 2 - 0.02, r.t_ssd, h, left=r.t_cpu, color=C_SSD, zorder=3)
            # 兩條線都結束後，算新的 token
            ax.barh(y + h / 2 + 0.02, r.t_new, h, left=r.t_restore, color=C_NEW, zorder=3)
            ax.plot([r.ttft, r.ttft], [y - h - 0.06, y + h + 0.06], color=INK, lw=1.6, zorder=4)
            ax.text(r.ttft + 0.015, y, f"{r.ttft:.2f} 秒　GPU 算 {int(r.meet)} 段、I/O 搬 {int(r.n_cpu + r.n_ssd)} 段",
                    va="center", fontsize=8.5, color=INK2)
            ticks.append(y)
            labels.append(sname)
            rows.append(dict(run_id=RID, ts=TS, gpu_state=gs, strategy=s, t_recompute=r.t_rec, t_load_cpu=r.t_cpu,
                             t_load_ssd=r.t_ssd, t_restore=r.t_restore, t_new=r.t_new, ttft=r.ttft, meet=r.meet,
                             n_load_cpu=r.n_cpu, n_load_ssd=r.n_ssd, n_reps=r.n, src="a1_share.csv",
                             src_run_id=r.src_run_id))
            y -= 1.0
        y -= 0.2

    # 重點標註：GPU 忙時，一開始不寫只能把前段全部重算
    r3 = g[(g.gpu_state == "busy2") & (g.strategy == "S3")].iloc[0]
    r0 = g[(g.gpu_state == "busy2") & (g.strategy == "S0")].iloc[0]
    ax.text(1.27, ticks[3] + 0.05,
            "怎麼看：每列上面是 GPU 重算、下面是 I/O 載入，同時開始。\n"
            "等待時間（黑線）＝比較長的那條＋算新 token。\n\n"
            "GPU 忙時：\n"
            f"・「全部都存」可以少算幾段（{int(g[(g.gpu_state == 'idle') & (g.strategy == 'S0')].meet.iloc[0])}→{int(r0.meet)} 段），改從 SSD 搬。\n"
            f"・「先寫再丟」「一開始不寫」沒有前段可搬，\n　只能把 {int(r3.meet)} 段算完；I/O 早就搬完，在旁邊等。\n"
            f"　→ 比「全部都存」慢 {100 * (r3.ttft / r0.ttft - 1):.0f}%",
            va="center", ha="left", fontsize=8.8, color=INK, linespacing=1.45,
            bbox=dict(boxstyle="round,pad=0.5", fc="#f3f2ee", ec="#d6d5cf", lw=0.8))
    ax.set_yticks(ticks, labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 2.05)
    ax.set_xlabel("時間（秒）")
    ax.grid(axis="x", color="#e6e5e0", lw=0.6, zorder=0)
    ax.legend(handles=[Patch(color=C_GPU, label="GPU 重算（從前面往後）"), Patch(color=C_CPU, label="從 CPU 載入"),
                       Patch(color=C_SSD, label="從 SSD 載入（從後面往前）"), Patch(color=C_NEW, label="算新的 token"),
                       plt.Line2D([], [], color=INK, lw=1.6, label="等待時間 TTFT")],
              loc="upper right", frameon=False, fontsize=8.5, ncol=1)
    ax.set_title("時間花在哪裡：Cake 還原時，GPU 重算和 I/O 載入兩條線同時跑，等待時間由比較長的那條決定", loc="left",
                 fontsize=11, color=INK, pad=12)
    fig.text(0.01, 0.01, "Llama-3.1-8B、32K（64 段）、SSD 層＝本地 SSD 實測參數、使用者等寫完才回來；每格 6 次的中位數。"
             f"run {g.src_run_id.iloc[0]}；畫圖程式 code/m9_advisor_breakdown.py", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0.04, 0.03, 1, 1))
    fig.savefig(FIG, dpi=200)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"{FIG}\n{OUT}: {len(rows)} 列")


if __name__ == "__main__":
    main()
