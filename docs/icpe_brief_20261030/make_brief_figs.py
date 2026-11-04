#!/usr/bin/env python3
"""Five summary figures for docs/icpe_brief_20261030/BRIEF_DISCORD.md (made for phone-size viewing, PNG).

    bash -c 'source code/env_h200.sh >/dev/null 2>&1; python docs/icpe_brief_20261030/make_brief_figs.py'

Rule: every number plotted is typed here once AND asserted to appear literally on its EVIDENCE.md row
(docs/icpe_paper_20261030/EVIDENCE.md) or read from a figure sidecar JSON; a mismatch aborts the script.
Nothing here is a new measurement. Writes figs/brief_sources.json (figure -> values -> evidence row).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "icpe_paper_20261030"
EVID = (PAPER / "EVIDENCE.md").read_text(encoding="utf-8").splitlines()
FI_SIDECAR = json.load(open(PAPER / "figs" / "fig9_fi_capacity.sources.json"))["plotted_values"]
OUT = HERE / "figs"
OUT.mkdir(exist_ok=True)

for f in Path.home().glob(".local/share/fonts/NotoSansTC-*.otf"):
    fm.fontManager.addfont(str(f))
BLUE, GRAY, ORANGE, AQUA, VIOLET = "#2a78d6", "#a09f98", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Noto Sans TC", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#c3c2b7", "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})
SOURCES = {}


def chk(prefix, *literals):
    """The EVIDENCE line that starts with `prefix` must contain every literal (so no number is invented)."""
    hits = [l for l in EVID if l.startswith(prefix)]
    assert hits, f"no EVIDENCE line starts with {prefix!r}"
    for lit in literals:
        assert any(lit in l for l in hits), f"{lit!r} not on EVIDENCE line {prefix!r}"


def save(fig, name, values, rows):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    SOURCES[name] = {"evidence_rows": rows, "values": values}
    print("wrote", name)


# ------------------------------------------------------------------ A: restore time by tier (Llama, 32K, GPU idle)
def fig_a():
    tiers = ["CPU\n記憶體", "b16\n8.83 GiB/s", "b32\n2.49", "b48\n0.72", "NFS 級\n0.33"]
    pre = ["| Llama／CPU／閒 |", "| Llama／b16／閒 |", "| Llama／b32／閒 |", "| Llama／b48／閒 |", "| Llama／NFS／閒 |"]
    bf16 = [0.0852, 0.4530, 1.6071, 5.5389, 12.082]
    int4c = [0.0290, 0.1883, 0.3772, 0.8765, 1.2412]
    recompute = [2.104, 2.105, 2.106, 2.106]          # NFS: NOT_MEASURED in M1 (recompute does not touch the tier)
    for p, a, b in zip(pre, bf16, int4c):
        chk(p, f"{a:g}", f"{b:g}")
    for p, r in zip(pre[:4], recompute):
        chk(p, f"{r:g}")
    chk("| b\\* 慢層", "8.83", "2.49", "0.722")
    chk("| NFS 級 |", "0.331")
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    x = range(len(tiers))
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], bf16, w, color=GRAY, label="BF16 整段搬回（基準）")
    b2 = ax.bar([i + w / 2 for i in x], int4c, w, color=BLUE, label="INT4 + Cake（本文建議）")
    ax.hlines(2.105, -0.55, 3.55, colors=ORANGE, linestyles=(0, (5, 3)), lw=1.8)
    ax.text(-0.5, 2.105 * 1.12, "整段重算 ≈2.1 s（跟放哪層無關）", color=ORANGE, ha="left", va="bottom", fontsize=10)
    ax.set_yscale("log")
    ax.set_ylim(0.015, 40)
    for bars in (b1, b2):
        for r in bars:
            v = r.get_height()
            ax.text(r.get_x() + r.get_width() / 2, v * 1.12, f"{v:.3g}", ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(list(x)); ax.set_xticklabels(tiers)
    ax.set_ylabel("把 32K 對話（4 GiB 的 KV）搬回 GPU 要幾秒（對數軸）")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0, 0.93))
    ax.set_title("停得越遠等越久；INT4 + Cake 把最慢那層從 12 秒壓到 1.2 秒", loc="left")
    ax.annotate("約 9.7 倍", xy=(4 + w / 2, 1.2412), xytext=(3.35, 5.4), color=BLUE, fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=1))
    fig.text(0.01, -0.03, "Llama-3.1-8B、GPU 閒、H200 實測。b16/b48 的 INT4 用逐塊解壓（偏慢）。b16~NFS 是限速模擬的慢層；NFS 級沒量整段重算。",
             fontsize=8.5, color=INK2)
    save(fig, "brief_a_tier_time.png",
         {"tiers": tiers, "bf16_full_load_s": bf16, "int4_cake_s": int4c, "full_recompute_s_measured_tiers": recompute,
          "ratio_NFS_bf16_over_int4cake": round(12.082 / 1.2412, 2), "ratio_note": "arithmetic from the two M1 cells"},
         ["M1", "§0 b*"])


# ------------------------------------------------------------------ B: same INT4, two dequant implementations
def fig_b():
    chk("| Llama／CPU／忙 |", "0.2383", "0.1208", "0.0342")
    chk("| M3-g |", "0.2325", "0.0342", "6.79")
    labels = ["BF16\n整段搬回", "BF16\nCake", "INT4 整段搬回\n逐塊解壓", "INT4 整段搬回\n一次拆完"]
    vals = [0.2383, 0.1208, 0.2325, 0.0342]
    cols = [GRAY, GRAY, ORANGE, BLUE]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    bars = ax.bar(labels, vals, color=cols, width=0.62)
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.006, f"{v:.4g} s", ha="center", fontsize=11)
    ax.set_ylim(0, 0.29)
    ax.set_ylabel("搬回要幾秒（越低越好）")
    ax.set_title("同樣是 INT4，解壓寫法不同就差 6.8 倍", loc="left")
    ax.annotate("", xy=(3, 0.0342 + 0.03), xytext=(2, 0.2325 - 0.02),
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.5, connectionstyle="arc3,rad=-0.25"))
    ax.text(2.5, 0.19, "6.8 倍", ha="center", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, -0.04, "Llama-3.1-8B、KV 放在 CPU 記憶體、GPU 忙、32K，H200 實測。逐塊解壓：每讀一塊就轉一次；一次拆完：整段讀完再轉。",
             fontsize=8.5, color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    save(fig, "brief_b_dequant.png", {"labels": labels, "seconds": vals, "ratio_M3g": "6.79 (seed 310) / 6.85 (seed 311)"}, ["M1", "M3-g"])


# ------------------------------------------------------------------ C: MPS trade-off (simulation)
def fig_c():
    rows = {
        "Llama · chunked prefill 背景 · 對話 trace": ("| Llama, cprefill, conversation |", BLUE,
            [(0.801, 0.0), (0.777, 50.4), (0.668, 71.7), (0.643, 77.8)]),
        "Llama · chunked prefill 背景 · agent trace": ("| Llama, cprefill, FreeInference |", AQUA,
            [(0.937, 0.0), (0.937, 62.9), (0.772, 76.0), (0.752, 79.5)]),
        "Llama · mixed 背景 · 對話 trace": ("| Llama, mixed, conversation |", ORANGE,
            [(0.697, 0.0), (0.632, 33.2), (0.607, 33.7), (0.602, 36.3), (0.470, 44.7)]),
    }
    for _, (p, _, pts) in rows.items():
        for k, (b, g) in enumerate(pts):
            if k == 0:
                chk(p, f"{b:.3f}")                 # stream B is column 1
            else:
                chk(p, f"{g:.1f}%／{b:.3f}")       # the rest are "gain%／B"
    names = ["stream 共用\n(基準)", "MPS\n不設上限", "背景限\n87%", "背景限\n75%", "背景限\n50%"]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.axvspan(0.95, 1.02, ymin=0, ymax=1, color="#e6f4ec", zorder=0)
    ax.text(0.985, 4, "登記的目標區\nB≥0.95 且 ≥15%\n沒有任何一點進去", ha="center", va="bottom", fontsize=8.5, color="#14724e")
    for lab, (_, c, pts) in rows.items():
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, "-o", color=c, lw=2, ms=6, label=lab)
    for (b, g), n in zip(rows["Llama · chunked prefill 背景 · 對話 trace"][2], names):
        ax.annotate(n.replace("\n", " "), (b, g), textcoords="offset points", xytext=(-6, 7), ha="right", fontsize=8.5, color=BLUE)
    ax.set_xlim(0.42, 1.02); ax.set_ylim(-4, 90)
    ax.set_xlabel("別的工作還剩多少處理量 B（1.0＝完全不受影響）")
    ax.set_ylabel("回來請求的等待比 stream 共用縮短 (%)")
    ax.set_title("替送回保留越多 GPU：回答越快，別的工作越慢", loc="left")
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=9, bbox_to_anchor=(0, -0.16))
    fig.text(0.01, -0.30, "模擬（H200 常數）。每條線由右往左：stream（gain＝0）→ MPS 不設上限 → 背景限 87% → 75% →（只有 mixed 有）50%。\nGPU 縮小重播的方向相同，但回答變快的幅度比模擬大 4.6–31.3 pp。",
             fontsize=8.5, color=INK2)
    save(fig, "brief_c_mps.png", {k: v[2] for k, v in rows.items()}, ["M4", "S2", "S3"])


# ------------------------------------------------------------------ D: ceiling vs CPU capacity (standard axes: x = capacity, y = headroom)
def fig_d():
    from make_ceiling_table import CAPS, load
    val, _ = load()
    chk("| B1 |", "50.5", "46.5", "27.6", "31.0")
    chk("| B3-T5 |", "4.89", "11.81")
    chk("| B3-X1 |", "0.30%")
    series = [("freeinference", "FreeInference（真實 agent 服務紀錄）", BLUE, "o"),
              ("conversation", "Mooncake conversation（對話）", ORANGE, "s"),
              ("toolagent", "Mooncake toolagent（工具／agent）", VIOLET, "^")]
    panels = [("O-PF", "預取：回來前先把資料搬好"), ("O-ORD", "淘汰順序：CPU 滿了，先擠掉最晚回來的")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.2), sharey=True)
    plotted = {}
    for ax, (comp, title) in zip(axes, panels):
        for tr, lab, c, m in series:
            pts = [(i, val[(tr, comp, "ssd", cap)]) for i, cap in enumerate(CAPS) if (tr, comp, "ssd", cap) in val]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, marker=m, lw=2.2, ms=6.5, label=lab, zorder=3)
            plotted[f"{tr}/{comp}"] = {CAPS[i]: round(v, 3) for i, v in pts}
            show = pts if tr != "freeinference" else [p for p in pts if p[0] in (0, 1, 3)]
            for i, v in show:
                ax.annotate(f"{v:.1f}" if abs(v) >= 1 else f"{v:.2f}", (i, v), textcoords="offset points", xytext=(0, 8 if v >= 0 else -14),
                            ha="center", fontsize=9, color=c, zorder=5)
        ax.axhline(0, color=INK2, lw=0.9, zorder=2)
        ax.set_xticks(range(len(CAPS))); ax.set_xticklabels(CAPS)
        ax.set_xlabel("CPU 記憶體容量（GiB）", fontsize=10.5)
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.set_ylim(-11, 58)
    axes[0].set_ylabel("完美預知讓 TTFT 縮短的 %", fontsize=10.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=3, frameon=False, fontsize=9.6, columnspacing=1.2, handletextpad=0.4)
    fig.text(0.01, 0.985, "完美預知最多能讓回來的使用者少等多少？取決於紀錄，也取決於 CPU 記憶體多大", fontsize=13, fontweight="bold", va="top")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.30, wspace=0.06)
    fig.text(0.01, 0.015,
             "TTFT＝使用者回來後，等到第一個字的時間。縮短的 %＝完美預知相對於「同一個動作下最好的不預測做法」。CPU 記憶體容量＝用來停放對話 KV 的\n"
             "CPU 記憶體大小。慢層＝SSD 級（讀 6.98 GiB/s）、Cake 還原、每格取 GPU 閒／忙的較大者；模擬、未截尾。Mooncake 只跑了 128–1,024 GiB，\n"
             "所以線從 128 開始。FreeInference 的 SSD 級 16 GiB 預取是負的：完美預取反而比最好的不預測做法慢。",
             fontsize=8.4, color=INK2, va="bottom", linespacing=1.4)
    save(fig, "brief_d_prediction_ceiling.png", plotted, ["B1", "B3-X1", "B3-T5", "x1_headroom_components.csv (rule=min, Cake, SSD class)", "fig9_fi_capacity.sources.json"])


# ------------------------------------------------------------------ E: typing hint on the GPU (standard grouped bars)
def fig_e():
    chk("| C2-GPU |", "0.278", "0.272", "0.237", "0.228", "0.81", "0.79", "14.5", "17.9", "4.0")
    chk("| B6 |", "8.1")
    chk("| GPU 驗證（B6）設定 |", "128／256 ÷ 8", "1/8")
    sets = {"16": (0.278, 0.237, 0.228, "0.81"), "32": (0.272, 0.237, 0.228, "0.79")}
    fig, ax = plt.subplots(figsize=(7.6, 5.5))
    w = 0.25
    names = (("LRU（不預測）", GRAY), ("打字提示預取（偵測到使用者開始打字就先搬）", BLUE), ("完美預取（事先知道何時回來）", AQUA))
    for i, (cap, (lru, hint, orc, ratio)) in enumerate(sets.items()):
        for j, (v, (lab, c)) in enumerate(zip((lru, hint, orc), names)):
            x = i + (j - 1) * (w + 0.02)
            ax.bar(x, v, w, color=c, label=lab if i == 0 else None, zorder=3)
            ax.text(x, v + 0.005, f"{v:.3f}", ha="center", fontsize=10.5)
        ax.text(i, 0.335, f"打字提示拿到完美預取\n改善量的 {ratio}", ha="center", fontsize=10.5, fontweight="bold", color=BLUE)
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"CPU 記憶體 {k} GiB\n（縮小 1/8，相當於全尺寸 {int(k) * 8} GiB）" for k in sets])
    ax.set_ylim(0, 0.40)
    ax.set_ylabel("回來請求的平均 TTFT（秒，越低越好）")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(loc="upper center", frameon=False, fontsize=9.6, ncol=1, bbox_to_anchor=(0.5, -0.17))
    fig.text(0.01, 0.985, "不用預測模型，使用者一開始打字就先搬：拿到完美預取好處的八成", fontsize=13, fontweight="bold", va="top")
    fig.subplots_adjust(top=0.90, bottom=0.34, left=0.11, right=0.98)
    fig.text(0.01, 0.012,
             "H200 實測，Mooncake 對話依 session 抽 1/8、CPU 容量同比縮小、Llama-3.1-8B、SSD 級慢層。完美預取只比打字提示再快 4.0%（16 GiB）。打字時間是合成的\n"
             "（Mooncake 沒有按鍵紀錄）。重播不含排隊；含排隊時完美預取比 LRU 快的幅度從 17.9% 掉到 8.1%（16 GiB）。",
             fontsize=8.4, color=INK2, va="bottom", linespacing=1.4)
    save(fig, "brief_e_typing_hint.png",
         {"CPU16": {"LRU_s": 0.278, "HINT_PF_s": 0.237, "O_PF_s": 0.228, "hint_over_oracle": 0.81},
          "CPU32": {"LRU_s": 0.272, "HINT_PF_s": 0.237, "O_PF_s": 0.228, "hint_over_oracle": 0.79}}, ["C2-GPU", "B6"])


# ------------------------------------------------------------------ S: schematic - why write-time placement does not help (interpretation)
def fig_s():
    from matplotlib.patches import FancyArrowPatch, Rectangle
    chkp("寫入准入對 SSD 寫入量與壽命的好處不在本文的量測範圍", "Cake 從前段重算，還原時大多不讀前段")
    N, K, KFULL = 64, 4, 24                      # 64 chunks of 512 tokens; K = example Cake meeting point; KFULL = illustration only
    ORG, ORGE = "#f4a261", "#c9772e"
    fig = plt.figure(figsize=(10.6, 6.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    X0, X1 = 24.0, 98.0
    bw = (X1 - X0) / N

    def row(y, colors, hatch_first=0, h=5.4):
        for i, c in enumerate(colors):
            hatch = "////" if i < hatch_first else None
            ax.add_patch(Rectangle((X0 + i * bw, y), bw * 0.9, h, fc=c, ec=ORGE if hatch else "white", lw=0.5 if hatch else 0.3, hatch=hatch, zorder=3))

    def xc(i):
        return X0 + i * bw
    ax.text(1.2, 97.5, "示意圖：為什麼「寫入時套用」沒有比「CPU 滿了才套用」快", fontsize=14, fontweight="bold", va="top")
    ax.text(1.2, 93.0, "一段 32K 對話切成 64 格（每格 512 個 token），第 1 格最舊、第 64 格最新", fontsize=10.5, color=INK2, va="top")
    # ① what happens when the user returns
    ax.text(1.2, 86.0, "① 使用者回來時，兩頭同時做", fontsize=12, fontweight="bold", va="center")
    row(72.0, ["#4a8fe0"] * N, hatch_first=0)
    row(72.0, [ORG] * K + ["#4a8fe0"] * (N - K), hatch_first=K)
    ax.add_patch(FancyArrowPatch((xc(0), 80.5), (xc(K), 80.5), arrowstyle="-|>", mutation_scale=14, color=ORGE, lw=2))
    ax.text(xc(0), 83.2, f"GPU 從第 1 格往後重算（例：前 {K} 格）", fontsize=10.3, color="#a85a1a", va="center")
    ax.add_patch(FancyArrowPatch((X1, 80.5), (xc(K), 80.5), arrowstyle="-|>", mutation_scale=14, color="#2a78d6", lw=2))
    ax.text(X1, 83.2, f"同時從儲存載入其餘 {N - K} 格", fontsize=10.3, color="#1e5aa8", va="center", ha="right")
    ax.text(xc(0), 68.0, "第 1 格", fontsize=9, color=INK2, ha="left"); ax.text(X1, 68.0, "第 64 格", fontsize=9, color=INK2, ha="right")
    ax.text(1.2, 76.0, "回來時怎麼拿", fontsize=10.3, color=INK2, va="center")
    ax.text(X0, 63.0, f"→ 前 {K} 格是「重算」出來的，根本不從儲存讀。所以它們事先放在哪，回來時都用不到。", fontsize=10.8, fontweight="bold", va="center")
    # ② CPU has room
    ax.text(1.2, 55.0, "② CPU 還有空位時：只有前 4 格放哪不同", fontsize=12, fontweight="bold", va="center")
    row(45.0, [GRAY] * K + [AQUA] * (N - K))
    row(36.0, [AQUA] * N)
    ax.text(1.2, 47.7, "寫入時套用", fontsize=10.8, va="center", fontweight="bold"); ax.text(1.2, 43.5, "對話一結束，前 4 格就放慢層", fontsize=8.6, va="center", color=INK2)
    ax.text(1.2, 38.7, "CPU 滿了才套用", fontsize=10.8, va="center", fontweight="bold"); ax.text(1.2, 34.5, "先全放 CPU，滿了才降級", fontsize=8.6, va="center", color=INK2)
    ax.add_patch(Rectangle((X0, 29.2), 2.2, 2.4, fc=AQUA, ec="none")); ax.text(X0 + 3.2, 30.4, "放 CPU", fontsize=9.5, va="center")
    ax.add_patch(Rectangle((X0 + 14, 29.2), 2.2, 2.4, fc=GRAY, ec="none")); ax.text(X0 + 17.2, 30.4, "放慢層", fontsize=9.5, va="center")
    # ③ CPU is tight
    ax.text(1.2, 22.0, "③ CPU 快滿時：兩種做法都先把「最前面的格子」降到慢層，擺法一樣", fontsize=12, fontweight="bold", va="center")
    row(13.0, [GRAY] * KFULL + [AQUA] * (N - KFULL))
    ax.text(1.2, 15.7, "兩種做法", fontsize=10.8, va="center", fontweight="bold")
    ax.text(xc(KFULL) + 1.0, 10.2, "（降幾格取決於 CPU 多大，這裡只是示意）", fontsize=9, color=INK2, va="center")
    ax.text(1.2, 4.9, f"結論：不同的只有前 {K} 格，而它們回來時靠重算、用不到儲存裡的副本；CPU 一緊，兩種擺法就一樣。實測 TTFT 差 −0.37～+1.16%（圖 1）。", fontsize=10.6, fontweight="bold", va="center")
    ax.text(1.2, 0.9, f"示意：前 {K} 格是用重算速度與 CPU 搬運速度算出的會合點的例子，實際值依設定而變。這個解釋是我們的判讀，沒有單獨驗證。", fontsize=8.4, color=INK2, va="center")
    save(fig, "brief_s_write_time_schematic.png", {"N": N, "example_meeting_point": K}, ["PAPER §7.1", "m7_write_policy.py"])


# ------------------------------------------------------------------ W: write-time placement vs apply-when-full (A1''/A1')
def fig_w():
    import csv
    import gzip
    import statistics as stt
    root = HERE.parents[1] / "results" / "icpe_h200" / "t1" / "gpu"
    chk("| A1'' |", "−0.37／−0.37／+0.01%", "rep 6", "δ\\* 5%")
    chk("| A1' |", "+1.16／+0.09／−0.02%", "δ\\* 5%")
    fracs = ("0.25", "0.5", "1.0")
    impls = (("a1pp", "fa3", "重算用 fa3"), ("a1p", "sdpa", "重算用 sdpa"))
    data = {}
    for d, tag, _ in impls:
        rows = list(csv.DictReader(gzip.open(root / d / "b_share.csv.gz", "rt")))
        assert not any(r["contaminated_gpu"] == "True" for r in rows), "contaminated rows"
        table = {r["cpu_frac"]: r for r in csv.DictReader(open(root / d / "analysis" / "table.csv"))}
        for cf in fracs:
            cell = {}
            for strat in ("S5", "S4+"):
                rr = [r for r in rows if r["strategy"] == strat and r["cpu_frac"] == cf and r["mode"] != "none"]
                assert len(rr) == 144, (d, strat, cf, len(rr))
                pooled = stt.median(float(r["ttft"]) for r in rr)
                per_rep = [stt.median(float(r["ttft"]) for r in rr if r["rep"] == str(k)) for k in range(6)]
                cell[strat] = (pooled, per_rep)
            diff = (cell["S4+"][0] - cell["S5"][0]) / cell["S4+"][0] * 100
            assert abs(diff - float(table[cf]["s5_vs_s4plus_pct"])) < 1e-6, (d, cf, diff)
            assert abs(cell["S4+"][0] - float(table[cf]["s4plus_median"])) < 1e-9
            data[(tag, cf)] = (cell, diff)
    diffs = [v[1] for v in data.values()]
    assert round(min(diffs), 2) == -0.37 and round(max(diffs), 2) == 1.16, (min(diffs), max(diffs))
    xl = ["25%", "50%", "100%"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.4), sharey=True, gridspec_kw={"wspace": 0.08})
    w = 0.34
    plotted = {}
    for ax, (d, tag, ttl) in zip(axes, impls):
        for i, cf in enumerate(fracs):
            cell, diff = data[(tag, cf)]
            for j, (strat, c) in enumerate((("S5", BLUE), ("S4+", GRAY))):
                x = i + (j - 0.5) * (w + 0.04)
                pooled, per_rep = cell[strat]
                ax.bar(x, pooled, w, color=c, zorder=3, label=("寫入時套用" if strat == "S5" else "CPU 滿了才套用") if i == 0 else None)
                ax.scatter([x + (k - 2.5) * 0.028 for k in range(6)], per_rep, s=11, color=INK, zorder=5, linewidths=0)
            top = max(max(cell["S5"][1]), max(cell["S4+"][1]))
            ax.text(i, top + 0.045, f"差 {diff:+.2f}%".replace("-", "−"), ha="center", fontsize=10.5, fontweight="bold", color=INK)
            plotted[f"{tag}/cpu_frac={cf}"] = {"S5_median_s": round(cell["S5"][0], 4), "S4plus_median_s": round(cell["S4+"][0], 4), "diff_pct": round(diff, 2)}
        ax.set_xticks(range(3)); ax.set_xticklabels(xl)
        ax.set_xlabel("CPU 能放下多少停放的 KV", fontsize=10.5)
        ax.set_title(ttl, loc="left", fontsize=11.5)
        ax.set_ylim(0, 1.4)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("回來請求的 TTFT 中位數（秒，越低越好）", fontsize=10.5)
    axes[0].legend(loc="upper right", frameon=False, fontsize=9.8)
    fig.text(0.01, 0.985, "同一條位置規則，只換「何時套用」：TTFT 幾乎一樣（差 −0.37～+1.16%，門檻 5%）", fontsize=13, fontweight="bold", va="top")
    fig.subplots_adjust(left=0.085, right=0.99, top=0.86, bottom=0.30)
    fig.text(0.01, 0.015,
             "規則＝依位置決定放哪層：前面的塊重算便宜，放慢層；後面的塊重算貴，放 CPU。寫入時套用＝對話一結束就這樣放；\n"
             "CPU 滿了才套用＝先全放 CPU，滿了才把位置最前面的塊降到慢層。「差」＝寫入時套用比另一版快的 %（正＝較快），要快 5% 以上才算贏。\n"
             "H200 實測、Llama-3.1-8B、32K、NFS 級慢層（限速模擬）、Cake 還原；8 個對話各 4 輪的小工作量，CPU 容量＝佔全部停放 KV 的比例。\n"
             "每個設定重複 6 次：柱＝全部回來請求的中位數（登記的統計量），黑點＝每次重複的中位數。fa3／sdpa＝重算時用的注意力實作。",
             fontsize=8.4, color=INK2, va="bottom", linespacing=1.4)
    save(fig, "brief_w_write_time.png", plotted, ["A1''", "A1'"])


# ------------------------------------------------------------------ T: two tables as images (Discord shows no markdown tables)
_TOK = set("–−‑～／／+-%.,:")
_NARROW = set("–−‑")
_CLOSE = set("，。、；：）」』！？")


def _is_tok(c):
    return (ord(c) < 128 and c not in " \n") or c in _TOK


def _wrap(text, units):
    """Wrap on display width (CJK = 2 units, ASCII = 1). A run of ASCII/number/range characters is never split,
    and closing punctuation never starts a line."""
    out, cur, w, i = [], "", 0, 0
    while i < len(text):
        if text[i] == "\n":
            out.append(cur); cur, w, i = "", 0, i + 1
            continue
        j = i + 1
        if _is_tok(text[i]):
            while j < len(text) and _is_tok(text[j]):
                j += 1
        tok = text[i:j]
        tw = sum(1 if (ord(c) < 128 or c in _NARROW) else 2 for c in tok)
        if w + tw > units and cur and tok[0] not in _CLOSE:
            out.append(cur); cur, w = "", 0
        cur += tok; w += tw; i = j
    out.append(cur)
    return out


def draw_table(name, title, headers, widths, rows, fills, foot, box_row=None, fs=11.2, width_in=11.8, bold_last=True):
    """rows: list of list of str; fills: list of list of colour or None; widths: relative column widths."""
    tot = sum(widths)
    colw = [width_in * 0.97 * w / tot for w in widths]
    lh, padv, padh = fs * 1.5 / 72, 0.14, 0.10
    def lines(cell, cw):
        return _wrap(cell, int(2 * (cw - 2 * padh) * 72 / fs * 0.97))
    hdr_lines = [lines(h, cw) for h, cw in zip(headers, colw)]
    hdr_h = max(len(l) for l in hdr_lines) * lh + 2 * padv
    row_ls = [[lines(c, cw) for c, cw in zip(r, colw)] for r in rows]
    row_h = [max(len(l) for l in ls) * lh + 2 * padv for ls in row_ls]
    foot_lines = foot.count("\n") + 1
    top_in, foot_in = 0.62, 0.30 + 0.17 * foot_lines
    H = top_in + hdr_h + sum(row_h) + foot_in
    fig = plt.figure(figsize=(width_in, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, width_in); ax.set_ylim(0, H); ax.axis("off")
    ax.text(0.15, H - 0.12, title, fontsize=13.5, fontweight="bold", va="top")
    x0, y = 0.15, H - top_in
    ax.add_patch(plt.Rectangle((x0, y - hdr_h), sum(colw), hdr_h, fc="#ecebe4", ec="none", zorder=1))
    x = x0
    for h, ls, cw in zip(headers, hdr_lines, colw):
        ax.text(x + padh, y - hdr_h / 2, "\n".join(ls), fontsize=fs, fontweight="bold", va="center", linespacing=1.5, color=INK)
        x += cw
    y -= hdr_h
    for ri, (r, ls, rh) in enumerate(zip(rows, row_ls, row_h)):
        x = x0
        for ci, (c, l, cw) in enumerate(zip(r, ls, colw)):
            f = fills[ri][ci]
            if f:
                ax.add_patch(plt.Rectangle((x, y - rh), cw, rh, fc=f, ec="none", zorder=1))
            ax.text(x + padh, y - rh / 2, "\n".join(l), fontsize=fs, va="center", linespacing=1.5, color=INK,
                    fontweight="bold" if (ci == 0 or (bold_last and ci == len(r) - 1)) else "normal", zorder=3)
            x += cw
        ax.plot([x0, x0 + sum(colw)], [y - rh, y - rh], color="#c3c2b7", lw=0.8, zorder=2)
        if box_row == ri:
            ax.add_patch(plt.Rectangle((x0, y - rh), sum(colw), rh, fc="none", ec=INK, lw=2.2, zorder=4))
        y -= rh
    ax.text(0.15, foot_in - 0.08, foot, fontsize=8.6, color=INK2, va="top", linespacing=1.45)
    save(fig, name, {"rows": [r[0] for r in rows]}, [])


def fig_tables():
    # --- table 1: what deciding at write time bought (all numbers are the EVIDENCE rows A1''/A1'/A2/A3/A4/A4')
    chk("| A1'' |", "−0.37／−0.37／+0.01%"); chk("| A1' |", "+1.16／+0.09／−0.02%")
    chk("| A2 |", "0/16"); chk("| A3 |", "存活 0/2", "2/5")
    chk("| A4 |", "+17.3／+18.8", "−4.9／−4.4", "−3.4／−1.9", "K2")
    chk("| A4' |", "−0.47～+0.08", "6 個 CI 都含 0")
    chkp("寫入准入對 SSD 寫入量與壽命的好處不在本文的量測範圍")
    GREY_, GREEN_, AMBER_ = "#eeede7", "#dff1e8", "#faecc8"
    rows = [
        ["依位置決定放哪層", "同一條規則，寫入時套用 vs CPU 滿了才套用；比回來請求的 TTFT（圖 1）",
         "差 −0.37～+1.16%（門檻 5%）。3 種 CPU 容量 × 2 種重算實作，各重複 6 次", "沒賺到"],
        ["寫入時決定，同時贏兩個對手各 5% 以上", "消融：頻寬、容量、負載型態、重算成本等 7 個條件。GPU 比 16 組；模擬挑出的 2 個設定再上 GPU",
         "0／16 組贏；2 個設定各只有 2／5 個隨機種子贏（要 3／5）", "沒賺到"],
        ["依位置決定精度（前／中／後段各 25% 的塊降精度）", "和「同比例、隨機挑」的對照比答對率。Llama 用 INT4、Qwen 用 FP8",
         "Llama INT4：差 −0.47～+0.08 pp，6 個信賴區間都含 0。Qwen FP8：前段 +17.3～+18.8、中段 −4.9～−4.4、後段 −3.4～−1.9 pp（GPU 閒／忙），方向不一",
         "沒有穩定的差距"],
        ["寫入准入（少寫慢層、保護 SSD 壽命）", "—", "沒量", "不在本文範圍"],
    ]
    fills = [[None, None, None, GREY_]] * 3 + [[None, None, None, None]]
    draw_table("brief_t1_write_time.png", "寫入時就決定的事：量到的都沒有賺到",
               ["寫入時決定什麼", "怎麼比", "量到什麼", "判定"], [2.5, 3.9, 4.6, 1.5], rows, fills,
               "H200 實測、Llama-3.1-8B 與 Qwen2.5-7B、32K。pp＝百分點（答對率對基準的差）。每次 32K 請求切成 64 塊；前／中／後段＝第 1–16／24–39／48–63 塊，隨機對照＝從第 1–63 塊隨機抽 16 塊；\n"
               "兩者存的位元組一樣，只有實際讀到的不同。〔判讀〕Qwen 前段「變好」是因為還原時大多不讀前段（Cake 從前段重算），不是真的好；它的 TTFT 反而慢 8.0／9.9%。")

    # --- table 2: format vs size vs slow-tier speed vs the usual benchmarks (absolute scores = BF16 score + measured diff)
    chk("| F1 |", "21.5–31.7%", "22.0–39.4%", "−48.3", "−36.2")
    chk("| F3 |", "37.4–49.1%", "36.7–54.5%", "−0.8 [−2.2, +0.3]")
    chk("| F6 |", "41.8–77.1%", "52.1–71.7%", "5.33×")
    chk("| D2-Q |", "LongBench +0.2 [−1.2, +1.5]", "LongBench +0.2 [−0.6, +1.1]")
    ro = json.load(open(HERE.parents[1] / "results" / "m10_precision_h200" / "analysis" / "report_only" / "report_only.json"))["items"]

    def score(model, fmt, bench):
        e0 = ro[f"{model}|e0acc|{bench}"]["result"]["acc"]
        if fmt == "bf16":
            return 100 * e0
        d = ro[f"{model}|{fmt}|{bench}"]["result"]["E1"]["A"]["diff"]     # reading A = load everything
        return 100 * (e0 + d)
    sc = {(m, f, b): round(score(m, f, b), 1) for m in ("llama", "qwen") for f in ("bf16", "fp8", "int4") for b in ("T1", "LB")}
    assert sc[("llama", "bf16", "T1")] == 99.7 and sc[("qwen", "fp8", "T1")] == 49.7 and sc[("qwen", "int4", "T1")] == 97.2
    assert sc[("qwen", "fp8", "LB")] == 6.6 and sc[("llama", "int4", "LB")] == 41.0 and sc[("qwen", "int4", "LB")] == 43.0
    assert ro["llama|e0acc|T1"]["result"]["n_rows"] == 360 and ro["llama|e0acc|LB"]["result"]["n_rows"] == 300
    RED_ = "#f8dcd3"

    def pair(f, b):
        return f"Llama {sc[('llama', f, b)]:.1f}\nQwen {sc[('qwen', f, b)]:.1f}"
    rows2 = [
        ["BF16（16 位元，原精度）", "1×（基準）", "基準", pair("bf16", "T1"), pair("bf16", "LB")],
        ["FP8（8 位元）", "1.94×", "21.5–39.4%", pair("fp8", "T1"), pair("fp8", "LB")],
        ["INT4（4 位元，KIVI）", "3.2×", "36.7–54.5%", pair("int4", "T1"), pair("int4", "LB")],
        ["INT2（2 位元，KIVI）", "5.33×", "41.8–77.1%", "沒測", "沒測\n（只有海選：誤差約是 INT4 的 30 倍）"],
    ]
    fills2 = [[None] * 5, [None, None, None, RED_, RED_], [None] * 5, [None, None, None, "#eeede7", "#eeede7"]]
    draw_table("brief_t2_format.png", "精度越低，搬得越快；但答對率守得住的，只有 INT4",
               ["格式（精度）", "KV 大小（壓縮）", "慢層還原比 BF16 快", "大海撈針答對率（%）", "LongBench 平均分數（0–100）"],
               [2.9, 1.7, 2.2, 2.4, 3.3], rows2, fills2,
               "H200 實測。慢層＝限速模擬的 b16／b32／b48（INT2 的範圍含 NFS 級），32K 單次還原、Cake 對 BF16。大海撈針＝多鍵 NIAH（180 題）；LongBench＝6 項長文問答（300 題）。\n"
               "分數＝BF16 的分數加上量到的差（全部載入的讀法）。紅底＝Qwen 的 FP8 崩壞（主任務也掉 32.2 個百分點）。Qwen 的 INT4 在大海撈針的 Cake＋GPU 忙讀法少 0.8 個\n"
               "百分點、信賴區間 [−2.2, +0.3]，沒過事先登記的門檻（下界 ≥ −2）。INT8、K4V2 未列入（沒有這兩項測驗）。", box_row=2, bold_last=False)


# ------------------------------------------------------------------ 0: the story in one picture (because -> so -> I adopt)
def fig_story():
    chk("| A2 |", "0/16")
    chk("| A1'' |", "−0.37")
    chk("| A1' |", "+1.16")
    chk("| B3-X1 |", "0.30%")
    chk("| B1 |", "50.5")
    chk("| Llama／NFS／閒 |", "12.082", "1.2412")
    chk("| M3-g |", "6.79")
    chk("| S2 |", "30.7–62.9%", "6.5 pp")
    chk("| C2-GPU |", "0.81")
    chk("| B3-T4 |", "99.29%")
    RED, AMB, GRN = "#c8472f", "#d49a1c", "#1f8f5f"
    rows = [
        ("假設一", "寫入時就依位置把 KV 放對層，\n比「滿了再搬」好", "沒用", RED,
         "H200 上 16 組比較 0 組贏；TTFT 差 −0.37～+1.16%",
         "為什麼：要搬的時候位置一樣看得到；Cake 又多半不讀前段"),
        ("轉向", "回來時怎麼送回：INT4 壓小 +\nCake 邊重算邊載入", "有用", GRN,
         "最慢那層 12.1 秒 → 1.24 秒，快 11.0–40.7%",
         "為什麼：慢層的時間花在搬位元組，INT4 小 3.2 倍"),
        ("意外", "INT4 怎麼解壓", "影響大", AMB,
         "同一種做法差 6.8 倍（CPU 記憶體層）",
         "為什麼：CPU 層搬得快，解壓成了主要成本"),
        ("GPU 共用", "MPS：搬回和別的工作同時跑", "取捨", AMB,
         "回答快 30.7–62.9%，別的工作最多慢 6.5 pp（模擬）",
         "為什麼：搬回拿到固定一份計算單元，不必輪流"),
        ("假設二", "預測誰會回來，能在這之上\n再省多少？", "看 trace", AMB,
         "agent 紀錄最多 0.30%；對話紀錄最多 50.5%",
         "為什麼：agent 紀錄 LRU 已命中 99.29%，沒東西可補"),
        ("免費訊號", "使用者一開始打字就預取", "有用", GRN,
         "拿到完美預取改善量的 0.81",
         "為什麼：打字的時間夠藏住搬回，不必預測模型"),
    ]
    trans = [
        "▼ 因為寫入時不用聰明（位置到搬的時候一樣看得到），\n所以改看回來的那一刻：我採用 Cake 當底，再把 KV 壓成 INT4",
        "▼ 因為 CPU 層搬得快，INT4 省下的搬運有限，\n所以把「怎麼解壓」當獨立變數：我採用兩種解壓實作並列比較",
        "▼ 因為真實服務的 GPU 不會只做搬回，\n所以讓搬回和別的工作同時跑：我採用 MPS，對照 stream 共用",
        "▼ 因為預測的價值要在最好的固定做法之上量，\n所以以上面的做法當底：我採用完美預測（oracle）量上限",
        "▼ 因為對話紀錄上預取有空間、但蓋預測器要花功夫，\n所以找免費的替代：我採用「使用者一開始打字」當預取訊號",
    ]
    RH, GH, HEAD, FOOT = 1.85, 0.95, 1.55, 0.9
    total = HEAD + FOOT + len(rows) * RH + len(trans) * GH
    fig = plt.figure(figsize=(7.4, total * 0.78))
    ax = fig.add_axes([0.01, 0.0, 0.98, 1.0])
    ax.set_xlim(0, 10); ax.set_ylim(0, total); ax.axis("off")
    ax.text(0.05, total - 0.35, "我們一開始這樣想，過程中發現…", fontsize=15, fontweight="bold", va="center")
    ax.text(0.05, total - 0.75, "起點：因為重算成本隨位置變貴、搬回成本不變，所以假設寫入時就依位置放好、\n再加預測；我採用「同一規則只換套用時間」和「完美預測上限」來驗證。",
            fontsize=10, color=INK2, va="top", linespacing=1.35)
    y = total - HEAD
    ax.text(0.05, FOOT / 2, "寫入時放置、送回、解壓：H200 實測。MPS、預測上限：模擬。打字提示：GPU 縮小重播（實測）。\n「為什麼」與轉折句是作者的解讀（99.29% 與 3.2 倍是量到的數字）。", fontsize=8.5, color=INK2, va="center", linespacing=1.4)
    for i, (tag, idea, verdict, col, result, why) in enumerate(rows):
        ax.add_patch(plt.Rectangle((0, y - RH + 0.07), 10, RH - 0.07, fc="#f6f5f1", ec="none"))
        ax.add_patch(plt.Rectangle((0, y - RH + 0.07), 0.14, RH - 0.07, fc=col, ec="none"))
        ax.text(0.4, y - 0.14, tag, fontsize=10.5, color=INK2, va="top")
        ax.text(0.4, y - 0.48, idea, fontsize=12.5, fontweight="bold", va="top", linespacing=1.3)
        ax.text(9.85, y - 0.14, verdict, fontsize=11.5, fontweight="bold", color="white", ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc=col, ec="none"))
        ax.text(0.4, y - 1.2, result, fontsize=11, va="top")
        ax.text(0.4, y - 1.52, why, fontsize=10, color=INK2, va="top")
        y -= RH
        if i < len(trans):
            ax.text(5, y - GH / 2 + 0.03, trans[i], fontsize=10, color=INK, ha="center", va="center", linespacing=1.35,
                    fontweight="bold")
            y -= GH
    save(fig, "brief_0_story.png", {"rows": [r[0] + ": " + r[4] for r in rows], "transitions": trans},
         ["A2", "A1''", "A1'", "B1", "B3-X1", "B3-T4", "M1", "M3-g", "S2", "C2-GPU"])


# ------------------------------------------------------------------ M: design method overview (two key points)
def fig_method():
    chk("| B6 |", "0.911–0.987")
    from matplotlib.patches import FancyBboxPatch
    fig = plt.figure(figsize=(7.4, 11.0))
    ax = fig.add_axes([0.01, 0.005, 0.98, 0.99])
    ax.set_xlim(0, 10); ax.set_ylim(0, 16.2); ax.axis("off")
    ax.text(0.05, 15.8, "設計方法：兩個重點", fontsize=15, fontweight="bold", va="center")

    def card(x, y, w, h, col, title, body, tsize=11, bsize=9.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white", ec=col, lw=2))
        ax.add_patch(FancyBboxPatch((x, y + h - 0.7), w, 0.7, boxstyle="round,pad=0.02,rounding_size=0.12", fc=col, ec=col, lw=2))
        ax.text(x + w / 2, y + h - 0.35, title, color="white", fontsize=tsize, fontweight="bold", ha="center", va="center")
        ax.text(x + 0.12, y + h - 0.9, body, fontsize=bsize, va="top", linespacing=1.4)

    # ---- key point 1: pre-registration
    ax.add_patch(plt.Rectangle((0, 10.35), 10, 5.0, fc="#f6f5f1", ec="none"))
    ax.text(0.15, 15.05, "重點一：判斷門檻，看到數字之前先登記", fontsize=12.5, fontweight="bold", va="top")
    ax.text(0.15, 14.55, "為什麼：門檻是研究的停損點，看過數字才定，容易往對專案有利的方向調", fontsize=9.6, color=INK2, va="top")
    card(0.1, 11.3, 3.0, 2.9, BLUE, "① 開跑前登記", "寫下：量什麼、\n對手是誰、什麼算贏\n\n連「登記前已經看過\n的數字」也列出")
    card(3.5, 11.3, 3.0, 2.9, VIOLET, "② 照登記去量", "登記後才改的做法，\n一律記成「偏離」，\n標出影響，\n以較保守的版本為準")
    card(6.9, 11.3, 3.0, 2.9, AQUA, "③ 機械式判定", "判定字固定：\n值得做／可能／死路\n\n沒過門檻的結果\n也照列")
    for x0, x1 in ((3.12, 3.48), (6.52, 6.88)):
        ax.annotate("", xy=(x1, 12.75), xytext=(x0, 12.75), arrowprops=dict(arrowstyle="->", lw=1.8, color=INK2, shrinkA=0, shrinkB=0))
    ax.text(0.15, 10.85, "門檻　時間：快 ≥5% 才算改進　｜　品質：答對率不低於 BF16 2 個百分點　｜　預測上限：≥15% 值得、<5% 死路",
            fontsize=9.3, va="center")

    # ---- key point 2: the oracle ladder
    ax.text(0.15, 9.95, "重點二：「完美預測上限」怎麼設計出來的", fontsize=12.5, fontweight="bold", va="top")
    ax.text(0.15, 9.45, "做法：同一份紀錄、同一套送回方法，只換最後一個條件，跑四種（長條是示意，不是實測）",
            fontsize=9.6, color=INK2, va="top")
    bars = [("① 底：INT4＋Cake，不預測", 9.0, "#c9c8c0"), ("② 最好的不預測對手", 7.6, "#898781"),
            ("③ 免費訊號：打字提示", 6.4, "#f0b9a4"), ("④ 完美預測（oracle）", 4.7, "#b0225f")]
    for k, (lab, w, c) in enumerate(bars):
        yy = 8.35 - k * 0.75
        ax.text(3.0, yy, lab, ha="right", va="center", fontsize=9.8)
        ax.add_patch(plt.Rectangle((3.15, yy - 0.24), w * 0.7, 0.48, fc=c, ec="none"))
    xo, xr = 3.15 + 4.7 * 0.7, 3.15 + 7.6 * 0.7
    ax.plot([xo, xo], [5.55, 4.75], color=INK2, lw=1, ls=":")
    ax.plot([xr, xr], [7.45, 4.75], color=INK2, lw=1, ls=":")
    ax.annotate("", xy=(xo, 4.75), xytext=(xr, 4.75), arrowprops=dict(arrowstyle="<->", lw=1.5, color=INK, shrinkA=0, shrinkB=0))
    ax.text((xo + xr) / 2, 4.5, "預測最多能省的部分", ha="center", va="top", fontsize=10.5, fontweight="bold")
    ax.text(0.15, 3.65, "上限 ＝ 1 −（④ 的 TTFT ÷ ② 的 TTFT）　④ 和 ② 做一樣的事，只多知道未來的一件事\n④ 都比 ② 快不到 15%，就不值得蓋預測器；<5% 算死路",
            fontsize=9.6, va="top", linespacing=1.45)

    # ---- three measurement layers (compact)
    ax.text(0.15, 2.55, "量法分三層", fontsize=11.5, fontweight="bold", va="top")
    ax.text(0.15, 2.05, "① GPU 直接量搬回（選格式與送法）　→　② trace 模擬（MPS 取捨、預測上限）　→　③ 縮小的 GPU 重播檢查②\n（實測÷模擬 0.911–0.987，策略排名相同）；每個量時間的 run 有 GPU 監看，出現外來程式就作廢重量",
            fontsize=9.3, va="top", linespacing=1.45)
    save(fig, "brief_m_method.png", {"panel_values": "no plotted data; B6 ratio range 0.911–0.987; ladder bars are schematic"}, ["B6", "§0", "§3.1-3.4 of PAPER.md"])


# ------------------------------------------------------------------ N: the lookup table as a decision tree
def fig_lookup():
    chk("| Llama／CPU／忙 |", "23.1%")
    chk("| Qwen／CPU／忙 |", "54.2%")
    chk("| E3 |", "1.52–3.52")
    chk("| M3-f |", "6.7%")
    chk("| Llama／b32／閒 |", "25.6%")
    chk("| Qwen／NFS／閒 |", "40.7%")
    chk("| Llama／b16／忙 |", "10.4%")
    from matplotlib.patches import FancyBboxPatch
    fig = plt.figure(figsize=(7.4, 9.6))
    ax = fig.add_axes([0.01, 0.005, 0.98, 0.99])
    ax.set_xlim(0, 10); ax.set_ylim(0, 14.0); ax.axis("off")
    ax.text(0.05, 13.55, "「查表」：KV 放在哪，就用哪一種做法", fontsize=15, fontweight="bold", va="center")
    ax.text(0.05, 13.05, "32K 對話；Llama-3.1-8B 與 Qwen2.5-7B；表在測過的條件下不翻轉", fontsize=9.8, color=INK2, va="center")

    def node(x, y, w, h, text, fc, ec, bold=False, size=10.3):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, fontweight="bold" if bold else "normal", linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.4, color=INK2, shrinkA=0, shrinkB=0))

    node(3.2, 11.0, 3.6, 1.1, "第一步：KV 停在哪裡？", "#f6f5f1", INK, True, 11.5)
    for x in (1.7, 5.0, 8.3):
        arrow(5.0, 11.0, x, 10.4)
    node(0.1, 9.2, 3.2, 1.2, "CPU 記憶體\n（搬得快）", "white", BLUE, True)
    node(3.4, 9.2, 3.2, 1.2, "b16\n（中等慢的限速層）", "white", "#898781", True, 10)
    node(6.7, 9.2, 3.2, 1.2, "慢儲存\nb32、b48、NFS 級", "white", VIOLET, True)
    arrow(8.3, 9.2, 8.3, 8.7)
    node(6.7, 7.3, 3.2, 1.4, "INT4＋Cake", "#e8e4fa", VIOLET, True, 12)
    ax.text(8.3, 7.1, "比次佳做法快 11.0–40.7%\n兩個模型、GPU 閒或忙，\n每一格都是它最快", ha="center", va="top", fontsize=9.2, linespacing=1.4)
    arrow(5.0, 9.2, 5.0, 8.7)
    node(3.4, 7.3, 3.2, 1.4, "INT4＋Cake 或\nINT4 整段搬回", "#eeeeea", "#898781", True, 11)
    ax.text(5.0, 7.1, "兩者差不到 10.4%，\n選哪個都行", ha="center", va="top", fontsize=9.2, linespacing=1.4)
    arrow(1.7, 9.2, 1.7, 8.7)
    node(0.1, 7.3, 3.2, 1.4, "第二步：\nINT4 怎麼解壓？", "#f6f5f1", BLUE, True, 10.8)
    # vertical rail with ticks to the three leaves
    ys = [5.35, 3.35, 1.35]
    ax.plot([0.35, 0.35], [7.3, ys[-1] + 0.4], color=INK2, lw=1.4)
    leaves = [
        ("一次拆完（整段讀完再轉）", "INT4 整段搬回", "GPU 忙時比 Cake 快 23.1%（Llama）、54.2%（Qwen）"),
        ("逐塊解壓（每讀一塊轉一次）", "Cake", "整段搬回反而比 Cake 慢 1.52–3.52 倍；\nINT4 在 CPU 層不比 BF16 快"),
        ("不確定是哪一種", "一律 Cake", "24 個測試格平均只比最好的多 6.7%"),
    ]
    for y, (cond, ans, note) in zip(ys, leaves):
        arrow(0.35, y + 0.4, 0.7, y + 0.4)
        ax.text(0.75, y + 1.05, "如果：" + cond, fontsize=9.6, va="center", color=INK, fontweight="bold")
        node(0.75, y, 2.1, 0.8, ans, "#e3eefb", BLUE, True, 10.3)
        ax.text(3.0, y + 0.4, note, fontsize=9.0, va="center", linespacing=1.35)
    ax.text(0.05, 0.45, "整段重算：32K 時每一格都不是最好｜只准 BF16（不壓縮）時：用 Cake｜SSD 級沒量\n† Qwen 在 GPU 忙時的多鍵 NIAH 品質有小的疑慮，見 11/11", fontsize=9.0, color=INK2, va="center", linespacing=1.5)
    save(fig, "brief_n_lookup.png", {"cpu_one_shot_gain_busy": {"Llama": 23.1, "Qwen": 54.2}, "slow_tier_gain_range": [11.0, 40.7],
                                      "per_block_slowdown": [1.52, 3.52], "always_cake_mean_extra_pct": 6.7, "b16_gap_pct": 10.4}, ["M1", "E3", "M3-f", "M3-g"])


# ------------------------------------------------------------------ helpers shared with make_paper_overview.py
PAPER_TEXT = (PAPER / "PAPER.md").read_text(encoding="utf-8")


def chkp(*literals):
    """Every literal must appear in PAPER.md (numbers quoted in the overview come from the paper text)."""
    for lit in literals:
        assert lit in PAPER_TEXT, f"{lit!r} not in PAPER.md"


if __name__ == "__main__":
    for f in (fig_story, fig_method, fig_lookup, fig_a, fig_b, fig_c, fig_s, fig_w, fig_tables, fig_d, fig_e):
        f()
    (OUT / "brief_sources.json").write_text(json.dumps(SOURCES, ensure_ascii=False, indent=1), encoding="utf-8")
