"""m9_advisor_figs.py — 給老師的 3 張圖（10/12 進度回覆用）。只讀既有的 CSV，不產生新的量測。

圖 1：省下的寫入有沒有縮短等待？GPU 忙時沒存前段要付多少？（A1，32K，寫完才回來）
圖 2：「寫入時決定」要同時贏過簡單策略和延後版才算數——GPU 上所有比較（第一階段 B、08、第 1 輪）
圖 3：讀取端（Cake）在中低頻寬最有用（A0，32K）

用法：python code/m9_advisor_figs.py
輸出：docs/phase1_20261008/report/figs/advisor_fig{1,2,3}.png；results/m9_followup/advisor_fig_data.csv（畫出來的每個數字）
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
FIG = os.path.join(ROOT, "docs", "phase1_20261008", "report", "figs")
OUT = os.path.join(ROOT, "results", "m9_followup", "advisor_fig_data.csv")
RID = os.environ.get("RUN_ID", "no-run-id")
TS = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

FP = "/mlsteam/data/tiara/fonts/NotoSansCJKtc-Regular.otf"
font_manager.fontManager.addfont(FP)
plt.rcParams.update({"font.family": [font_manager.FontProperties(fname=FP).get_name(), "DejaVu Sans"],
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "savefig.facecolor": "#fcfcfb", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})
# 參考調色盤的前三格（validate_palette：全部 PASS；aqua 對比 <3:1 → 每根柱子都直接標數字）
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRAY = "#0b0b0b", "#52514e", "#8a8984"
rows = []


def fig1():
    a = pd.read_csv(os.path.join(R7, "a1_share.csv"))
    q = a[(a.L == 32768) & (a.gap == np.inf) & a.strategy.isin(["S0", "S2a", "S3"]) & (a.contaminated_gpu == 0)]
    g = q.groupby(["ssd_dev", "gpu_state", "strategy"]).agg(ttft=("ttft", "median"), meet=("meet", "median"),
                                                             run_id=("run_id", "first")).reset_index()
    names = {"S0": "全部都存", "S2a": "先寫再丟（前段）", "S3": "一開始不寫（前段）"}
    cols = {"S0": C1, "S2a": C2, "S3": C3}
    gpus = [("idle", "GPU 閒"), ("busy1", "GPU 忙 1.5×"), ("busy2", "GPU 忙 2.1×")]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, (dev, dname) in zip(axs, [("local", "SSD 層＝本地 SSD 實測參數"), ("nfs", "SSD 層＝NFS 實測參數")]):
        x = np.arange(3)
        w = 0.26
        for k, s in enumerate(["S0", "S2a", "S3"]):
            ys, ms = [], []
            for gs, _ in gpus:
                r = g[(g.ssd_dev == dev) & (g.gpu_state == gs) & (g.strategy == s)].iloc[0]
                ys.append(r.ttft)
                ms.append(int(r.meet))
                rows.append(dict(fig=1, panel=dev, x=gs, series=s, value=r.ttft, meet=int(r.meet), src="a1_share.csv",
                                 src_run_id=r.run_id))
            bars = ax.bar(x + (k - 1) * w, ys, w - 0.02, color=cols[s], label=names[s], zorder=3)
            for b, y, m in zip(bars, ys, ms):
                ax.text(b.get_x() + b.get_width() / 2, y, f"{y:.2f} s\n算 {m} 段", ha="center", va="bottom",
                        fontsize=7, color=INK2)
        # GPU 忙 2.1× 時，「一開始不寫」比「全部都存」慢多少
        s0 = g[(g.ssd_dev == dev) & (g.gpu_state == "busy2") & (g.strategy == "S0")].ttft.iloc[0]
        s3 = g[(g.ssd_dev == dev) & (g.gpu_state == "busy2") & (g.strategy == "S3")].ttft.iloc[0]
        ax.annotate(f"比「全部都存」慢 {100 * (s3 / s0 - 1):.0f}%", xy=(2 + w, s3 * 1.13), xytext=(2 + w, s3 * 1.30), ha="center",
                    fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
        ax.set_xticks(x, [n for _, n in gpus])
        ax.set_ylim(0, max(g[g.ssd_dev == dev].ttft) * 1.45)
        ax.set_title(dname, loc="left", fontsize=10, color=INK)
        ax.set_ylabel("回來請求的 TTFT（秒）")
        ax.grid(axis="x", visible=False)
    axs[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("圖 1　省下的寫入沒有縮短等待；GPU 忙時，沒存的前段只能重算", x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.005, "Llama-3.1-8B、32K、使用者等寫完才回來、每格 6 次的中位數；「算 n 段」＝64 段裡 GPU 重算了幾段（Cake 的會合點）。"
             "run 20261008-134526-m7-a1", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(os.path.join(FIG, "advisor_fig1.png"), dpi=200)
    plt.close(fig)


def fig2():
    pts = []
    # 第一階段 B（GPU）：S5 vs S4+（延後版）與 vs 最好的簡單策略；share I/O 模型的所有設定
    b = pd.read_csv(os.path.join(R7, "summary_b.csv"))
    b = b[b.io_model == "share"]
    simple = ["S1", "S2b", "S4", "S4+", "S4+P", "S4L"]
    for key, g in b.groupby(["cpu_tier", "ssd_dev", "gap_s", "ssd_frac", "wl_seed", "cpu_frac"]):
        m = g.set_index("strategy")["median"]
        if "S5" not in m or "S4+" not in m:
            continue
        bs = m[[s for s in simple if s in m]].min()
        pts.append(dict(stage="第一階段（10/8）", gx=(bs - m["S5"]) / bs, gy=(m["S4+"] - m["S5"]) / m["S4+"],
                        src="summary_b.csv", detail="S5 vs S4+；" + "／".join(map(str, key))))
    # 08（GPU）：每個寫入時策略 vs 最好的不看位置、vs S4B
    v = pd.read_csv(os.path.join(R7, "b2_verdict.csv"))
    for _, r in v[v.src == "gpu"].iterrows():
        pts.append(dict(stage="08 消融（10/9）", gx=r.gain_nwt, gy=r.gain_s4b, src="b2_verdict.csv",
                        detail=f"{r.strategy}；{r.workload}／{r.release}／{r.cpu_gibps:.2f}／{r.ssd_dev}／s{r.wl_seed}／{r.cpu_frac}；{r.run_id}"))
    # 第 1 輪（GPU）：S5L vs 最好的不看位置、vs S4B
    r1 = pd.read_csv(os.path.join(RX, "r1_gpu_verdict.csv"))
    for _, r in r1[r1.stat == "median"].iterrows():
        pts.append(dict(stage="第 1 輪（10/9）", gx=r.g_nwt, gy=r.g_S4B, src="r1_gpu_verdict.csv",
                        detail=f"S5L；seed {r.wl_seed}；{r.run_id}"))
    p = pd.DataFrame(pts)
    p["gx"], p["gy"] = 100 * p.gx, 100 * p.gy
    XL, YL = (-45, 25), (-40, 55)
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.add_patch(plt.Rectangle((5, 5), XL[1] - 5, YL[1] - 5, color=C3, alpha=0.10, zorder=1, lw=0))
    ax.text(XL[1] - 0.5, YL[1] - 1.5, "要落在這裡才算「寫入時決定」有用：\n同時贏簡單策略和延後版各 ≥5%\n→ 0 個點", ha="right",
            va="top", fontsize=9, color=INK)
    ax.axhline(0, color=GRAY, lw=0.8, zorder=2)
    ax.axvline(0, color=GRAY, lw=0.8, zorder=2)
    styles = {"第一階段（10/8）": (C1, "o"), "08 消融（10/9）": (C2, "s"), "第 1 輪（10/9）": (C3, "^")}
    nclip = 0
    for st, (c, mk) in styles.items():
        q = p[p.stage == st]
        x = q.gx.clip(XL[0] + 1, None)
        nclip += int((q.gx < XL[0] + 1).sum())
        ax.scatter(x, q.gy.clip(YL[0] + 1, YL[1] - 1), s=46, color=c, marker=mk, edgecolor="#fcfcfb", linewidth=1.2,
                   zorder=4, label=f"{st}：{len(q)} 組")
    best = p.loc[(p.stage == "08 消融（10/9）") & (p.gx > 15)].iloc[0]
    ax.annotate(f"最好的一組：贏簡單策略 {best.gx:.1f}%，\n但只贏延後版 {best.gy:.1f}%（換 seed 就不見）",
                xy=(best.gx, best.gy), xytext=(-5, -30), fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
    if nclip:
        ax.text(XL[0] + 1.5, YL[0] + 2, f"◀ 另有 {nclip} 組在 −94% 到 −122%（寫穿遠勝，畫在邊上）", fontsize=7.5, color=INK2)
    ax.set_xlim(*XL)
    ax.set_ylim(*YL)
    ax.set_xlabel("比「最好的簡單策略」快多少（%）")
    ax.set_ylabel("比「延後版」快多少（%）\n（同一條規則，CPU 滿了才做）")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("圖 2　GPU 上所有比較：「寫入時決定」從來沒有同時贏過兩邊", loc="left", fontsize=12, color=INK)
    fig.text(0.01, 0.005, "每個點＝一組 GPU 實測比較（回來請求 TTFT 中位數）。延後版：第一階段為 S4+，08 與第 1 輪為 S4B。"
             "來源：summary_b.csv、b2_verdict.csv、r1_gpu_verdict.csv", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(FIG, "advisor_fig2.png"), dpi=200)
    plt.close(fig)
    for _, r in p.iterrows():
        rows.append(dict(fig=2, panel="", x=r.gx, series=r.stage, value=r.gy, meet=None, src=r.src, src_run_id=r.detail))


def fig3():
    a = pd.read_csv(os.path.join(R7, "a0.csv"))
    a = a[(a.L == 32768) & (a.contaminated_gpu == 0)]
    comp = a[a["mode"] == "compute_only"].ttft.median()
    t = a[a["mode"] != "compute_only"].groupby(["bw_name", "read_GiBps", "mode"]).agg(
        ttft=("ttft", "median"), meet=("meet", "median"), run_id=("run_id", "first")).reset_index()
    pts = []
    for (bn, bw), g in t.groupby(["bw_name", "read_GiBps"]):
        cake = g[g["mode"] == "cake"].iloc[0]
        load = g[g["mode"] == "load_only"].ttft.iloc[0]
        pts.append(dict(bw_name=bn, bw=bw, sp=min(comp, load) / cake.ttft, meet=int(cake.meet), run_id=cake.run_id))
    p = pd.DataFrame(pts).sort_values("bw")
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.axvspan(0.3, 4.0, color=C1, alpha=0.08, lw=0)
    ax.text(1.1, 2.18, "0.3–4 GiB/s：快 1.4–1.9 倍", ha="center", fontsize=9, color=INK)
    ax.axhline(1.0, color=GRAY, lw=0.8)
    ax.plot(p.bw, p.sp, color=C1, lw=2, marker="o", ms=7, mec="#fcfcfb", mew=1.2, zorder=3)
    label = {"nfs": "NFS 實測", "local": "本地 SSD 實測", "7Gbps": "7 Gbps", "25Gbps": "25 Gbps", "32Gbps": "32 Gbps",
             "56Gbps": "56 Gbps", "100Gbps": "100 Gbps"}
    for _, r in p.iterrows():
        below = r.bw_name in ("nfs", "32Gbps", "local")
        ax.text(r.bw, r.sp + (-0.05 if below else 0.05), f"{r.sp:.2f}×\n{label.get(r.bw_name, r.bw_name)}\n算 {r.meet}／64 段",
                ha="center", va="top" if below else "bottom", fontsize=7.5, color=INK2)
        rows.append(dict(fig=3, panel="", x=r.bw, series=r.bw_name, value=r.sp, meet=r.meet, src="a0.csv",
                         src_run_id=r.run_id))
    ax.axvline(4.1, color=C2, lw=1, ls="--")
    ax.text(4.3, 2.28, "vLLM 實際的 CPU 卸載 ≈4.1 GiB/s（C6）", fontsize=8, color=INK, va="top")
    ax.set_xscale("log")
    ax.set_xticks([0.33, 1, 3, 6.98, 11.6], ["0.33", "1", "3", "7", "11.6"])
    ax.set_ylim(0.8, 2.35)
    ax.set_xlabel("儲存層頻寬（GiB/s，對數座標）")
    ax.set_ylabel("Cake 比「只算」和「只搬」中較快的再快幾倍")
    ax.set_title("圖 3　讀取端（Cake）在中低頻寬最有用（32K）", loc="left", fontsize=12, color=INK)
    fig.text(0.01, 0.005, f"A0：每點 6 次的中位數；只算（不搬）＝{comp:.2f} s。註：harness 的重算比預設 vLLM 慢約 1.7–2 倍，"
             "頻寬帶的絕對位置待重量。", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(FIG, "advisor_fig3.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    d = pd.DataFrame(rows)
    d.insert(0, "ts", TS)
    d.insert(0, "run_id", RID)
    d.to_csv(OUT, index=False)
    print(f"{OUT}: {len(d)} 列；圖在 {FIG}/advisor_fig{{1,2,3}}.png")
