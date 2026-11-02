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


# ------------------------------------------------------------------ D: how much can perfect prediction save
def fig_d():
    chk("| B1 |", "50.5", "27.6")
    fi_pf128 = FI_SIDECAR["O-PF"]["nfs"]["128"] * 100          # 0.22 %
    fi_or128 = FI_SIDECAR["O-ORD"]["nfs"]["128"] * 100         # 0.30 %
    fi_pf32 = max(FI_SIDECAR["O-PF"][t]["32"] for t in ("b32", "ssd", "nfs")) * 100      # 4.89 %
    fi_or32 = max(FI_SIDECAR["O-ORD"][t]["32"] for t in ("b32", "ssd", "nfs")) * 100     # 11.81 %
    chk("| B3-T5 |", "4.89", "11.81")
    chk("| B3-X1 |", "0.30%")
    chk("| B3-T4 |", "99.29%")
    groups = [
        ("Mooncake 對話 trace\nCPU 128 GiB", 50.5, 27.6, ORANGE),
        ("FreeInference（agent trace）\nCPU 128 GiB", fi_pf128, fi_or128, BLUE),
        ("FreeInference\nCPU 32 GiB（上界最高的容量）", fi_pf32, fi_or32, BLUE),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    ypos = [2.2, 1.1, 0.0]
    h = 0.34
    for (lab, pf, orr, c), y in zip(groups, ypos):
        ax.barh(y + h / 2, pf, h, color=c, alpha=1.0, label="完美預取（先把資料搬回來）" if y == 2.2 else None)
        ax.barh(y - h / 2, orr, h, color=c, alpha=0.5, label="完美淘汰順序（擠出誰）" if y == 2.2 else None)
        for v, yy in ((pf, y + h / 2), (orr, y - h / 2)):
            ax.text(v + 1.2 if v < 1 else v + 0.8, yy, f"{v:.2f}%" if v < 10 else f"{v:.1f}%", va="center", fontsize=10.5, bbox=dict(fc="white", ec="none", pad=1))
    ax.set_yticks(ypos); ax.set_yticklabels([g[0] for g in groups])
    for xv, t in ((5, "5% 以下：死路"), (15, "15% 以上：值得做")):
        ax.axvline(xv, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.text(xv + 0.4, 2.72, t, fontsize=9, color=INK2, va="bottom")
    ax.set_xlim(0, 60); ax.set_ylim(-0.6, 3.0)
    ax.set_xlabel("就算預測完全準確，最多還能把等待縮短幾 %（模擬）")
    ax.set_title("預測值不值得，看固定規則還剩多少「沒命中」可補", loc="left")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    ax.text(58.5, 1.3, "128 GiB 時，回來的資料有\n99.29% 本來就在 CPU 層\n→ 沒有「沒命中」可補", ha="right", va="center", fontsize=9.5, color=BLUE)
    fig.text(0.01, -0.03, "Mooncake：代表格（SSD 級、GPU 閒）。FreeInference：前 6 週，每格取各慢層×GPU 閒忙的最大值（對預測最有利的算法）。",
             fontsize=8.5, color=INK2)
    save(fig, "brief_d_prediction_ceiling.png",
         {"mooncake_conv_cpu128_prefetch_pct": 50.5, "mooncake_conv_cpu128_eviction_pct": 27.6,
          "fi_cpu128_prefetch_pct": round(fi_pf128, 3), "fi_cpu128_eviction_pct": round(fi_or128, 3),
          "fi_cpu32_prefetch_max_pct": round(fi_pf32, 3), "fi_cpu32_eviction_max_pct": round(fi_or32, 3),
          "fi_cpu128_hit_rate_pct": 99.29},
         ["B1", "B3-X1", "B3-T5", "B3-T4", "fig9_fi_capacity.sources.json"])


# ------------------------------------------------------------------ E: typing hint on the GPU
def fig_e():
    chk("| C2-GPU |", "0.278", "0.272", "0.237", "0.228", "0.81", "0.79")
    sets = {"CPU 16 GiB": (0.278, 0.237, 0.228, "0.81"), "CPU 32 GiB": (0.272, 0.237, 0.228, "0.79")}
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    w = 0.26
    for i, (k, (lru, hint, orc, ratio)) in enumerate(sets.items()):
        for j, (v, c, lab) in enumerate(((lru, GRAY, "LRU（不預測）"), (hint, BLUE, "打字提示預取（免費訊號）"), (orc, AQUA, "完美預取"))):
            r = ax.bar(i + (j - 1) * (w + 0.02), v, w, color=c, label=lab if i == 0 else None)
            ax.text(i + (j - 1) * (w + 0.02), v + 0.004, f"{v:.3f}", ha="center", fontsize=10.5)
        ax.text(i, 0.335, f"打字提示拿到完美預取\n改善量的 {ratio}", ha="center", fontsize=10.5, fontweight="bold", color=BLUE)
    ax.set_xticks([0, 1]); ax.set_xticklabels(list(sets))
    ax.set_ylim(0, 0.40)
    ax.set_ylabel("回來請求的平均 TTFT（秒，越低越好）")
    ax.set_title("不用預測模型，使用者一開始打字就預取：拿到完美預取改善量的八成", loc="left", fontsize=12)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(loc="upper center", frameon=False, fontsize=9.5, ncol=3, bbox_to_anchor=(0.5, -0.1))
    fig.text(0.01, -0.12, "H200 GPU 縮小重播（Mooncake 對話 1/8、SSD 級、Llama）。縱軸從 0 起。打字時間是合成的（Mooncake 沒有按鍵紀錄）。只對預取有效。",
             fontsize=8.5, color=INK2)
    save(fig, "brief_e_typing_hint.png",
         {"CPU16": {"LRU_s": 0.278, "HINT_PF_s": 0.237, "O_PF_s": 0.228, "hint_over_oracle": 0.81},
          "CPU32": {"LRU_s": 0.272, "HINT_PF_s": 0.237, "O_PF_s": 0.228, "hint_over_oracle": 0.79}}, ["C2-GPU"])


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


# ------------------------------------------------------------------ O: one-glance overview (problem -> how -> what we found)
PAPER_TEXT = (PAPER / "PAPER.md").read_text(encoding="utf-8")


def chkp(*literals):
    """Every literal must appear in PAPER.md (numbers quoted in the overview come from the paper text)."""
    for lit in literals:
        assert lit in PAPER_TEXT, f"{lit!r} not in PAPER.md"


def fig_overview():
    chkp("11.0–40.7%", "6.8 倍", "30.7–62.9%", "6.5 個百分點", "50.5%", "0.81", "12.082", "1.2412")
    chk("| B3-X1 |", "0.30%")
    from matplotlib.patches import FancyBboxPatch
    GRN, GREY, GOLD, RED = "#1f8f5f", "#52514e", "#c58a10", "#c8472f"
    fig = plt.figure(figsize=(11.0, 7.0))
    ax = fig.add_axes([0.005, 0.005, 0.99, 0.99])
    ax.set_xlim(0, 16.6); ax.set_ylim(0, 11.4); ax.axis("off")

    def card(x, w, col, title):
        ax.add_patch(FancyBboxPatch((x, 1.0), w, 9.6, boxstyle="round,pad=0.02,rounding_size=0.18", fc="white", ec=col, lw=2.2))
        ax.add_patch(FancyBboxPatch((x, 9.8), w, 0.8, boxstyle="round,pad=0.02,rounding_size=0.18", fc=col, ec=col, lw=2.2))
        ax.text(x + w / 2, 10.2, title, color="white", fontsize=14, fontweight="bold", ha="center", va="center")

    def arrow(x0, x1, y=5.8):
        ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="-|>", lw=2.6, color=INK2, shrinkA=0, shrinkB=0, mutation_scale=20))

    xs, ws = [0.1, 5.4, 11.0], [4.6, 5.2, 5.5]
    card(xs[0], ws[0], GREY, "要解決的問題")
    card(xs[1], ws[1], BLUE, "我們怎麼做")
    card(xs[2], ws[2], GRN, "我們發現")
    arrow(xs[0] + ws[0] + 0.03, xs[1] - 0.03); arrow(xs[1] + ws[1] + 0.03, xs[2] - 0.03)

    # ---- column 1: the problem
    x0 = xs[0] + 0.3
    def tier(y, label, fc, ec, dashed=False):
        ax.add_patch(FancyBboxPatch((x0, y), 2.5, 0.95, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec=ec, lw=1.8, ls=(0, (4, 3)) if dashed else "-"))
        ax.text(x0 + 1.25, y + 0.475, label, ha="center", va="center", fontsize=10.5, fontweight="bold")
    tier(8.35, "GPU", "#dbe8fb", BLUE)
    tier(7.05, "CPU 記憶體", "#f6f5f1", INK2)
    tier(5.75, "慢儲存\n（SSD／網路）", "#f6f5f1", "#898781", True)
    ax.annotate("", xy=(x0 + 2.85, 8.8), xytext=(x0 + 2.85, 6.0), arrowprops=dict(arrowstyle="-|>", lw=2.2, color=ORANGE, shrinkA=0, shrinkB=0, mutation_scale=16))
    ax.text(x0 + 3.0, 7.4, "使用者\n回來時，\n先搬回\nGPU", fontsize=9.8, color=ORANGE, fontweight="bold", va="center", linespacing=1.3)
    ax.text(x0, 5.0, "對話停下來時，KV 被移出 GPU\n（KV＝已算好的中間結果）\n32K 對話的 KV 有 4.0 GiB", fontsize=9.4, va="top", linespacing=1.45)
    ax.annotate("", xy=(x0 + 4.0, 3.05), xytext=(x0, 3.05), arrowprops=dict(arrowstyle="<->", lw=1.3, color=INK, shrinkA=0, shrinkB=0))
    ax.text(x0 + 2.0, 3.13, "使用者等的時間＝首字延遲 TTFT", ha="center", va="bottom", fontsize=9.2, fontweight="bold")
    ax.add_patch(plt.Rectangle((x0, 2.15), 2.75, 0.6, fc=ORANGE, ec="none")); ax.text(x0 + 1.375, 2.45, "搬回 KV", color="white", ha="center", va="center", fontsize=9.8, fontweight="bold")
    ax.add_patch(plt.Rectangle((x0 + 2.75, 2.15), 1.25, 0.6, fc="#898781", ec="none")); ax.text(x0 + 3.375, 2.45, "算新訊息", color="white", ha="center", va="center", fontsize=9)
    ax.text(x0, 1.75, "搬得慢，使用者就等得久", fontsize=10.2, va="top", fontweight="bold")

    # ---- column 2: how
    x1 = xs[1] + 0.25
    steps = [
        ("先問", "① 回來時怎麼送回最快？", "格式（BF16／INT4）× 送法\n（整段搬回／重算／Cake）", "GPU 直接量 [實測]"),
        ("再問", "② GPU 還要做別的事怎麼分？", "MPS（保留給搬回的比例）\nvs 單純共用", "trace 模擬 ＋ GPU 檢查"),
        ("最後問", "③ 再加預測，還能省多少？", "「完美預測」的上限、寫入時預知、\n免費訊號", "trace 模擬 ＋ GPU 重播"),
    ]
    tops = [9.5, 6.95, 4.4]
    for (tag, q, detail, how), top in zip(steps, tops):
        ax.add_patch(FancyBboxPatch((x1, top - 2.35), ws[1] - 0.5, 2.35, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#f6f5f1", ec="none"))
        ax.text(x1 + 0.15, top - 0.1, tag, fontsize=9.2, color=INK2, va="top")
        ax.text(x1 + 0.15, top - 0.45, q, fontsize=10.6, fontweight="bold", va="top")
        ax.text(x1 + 0.15, top - 0.98, detail, fontsize=9.1, va="top", linespacing=1.4)
        ax.text(x1 + 0.15, top - 1.95, how, fontsize=9.1, color=BLUE, fontweight="bold", va="top")
    ax.text(x1, 1.55, "每一步的判定門檻，都在看到數字之前先登記", fontsize=9.4, color=INK, fontweight="bold", va="center")

    # ---- column 3: findings
    x2 = xs[2] + 0.25
    finds = [
        ("① 查表就夠", "慢儲存：INT4＋Cake，最慢那層\n12.1 秒 → 1.24 秒（快 11.0–40.7%）\nCPU 記憶體：看 INT4 怎麼解壓，\n差 6.8 倍", GRN, 2.6),
        ("② MPS 是取捨旋鈕", "回答快 30.7–62.9%，別的工作\n最多慢 6.5 個百分點；保留越多給\n搬回，別的工作越慢", GOLD, 2.2),
        ("③ 預測的價值有上限、看紀錄", "agent 紀錄：完美預測也省不到 15%\n（128 GiB 時只有 0.30%）\n對話紀錄：最多省 50.5%\n免費的打字訊號拿到 0.81\n寫入時預知：量不到收益", RED, 3.0),
    ]
    y = 9.5
    for t, body, col, h in finds:
        ax.add_patch(FancyBboxPatch((x2, y - h), ws[2] - 0.5, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white", ec=col, lw=1.8))
        ax.add_patch(plt.Rectangle((x2, y - h), 0.13, h, fc=col, ec="none"))
        ax.text(x2 + 0.3, y - 0.12, t, fontsize=10.8, fontweight="bold", va="top", color=INK)
        ax.text(x2 + 0.3, y - 0.65, body, fontsize=9.1, va="top", linespacing=1.45)
        y -= h + 0.2

    # ---- bottom banner
    ax.add_patch(FancyBboxPatch((0.1, 0.05), 16.4, 0.75, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#eef4fb", ec=BLUE, lw=1.6))
    ax.text(8.3, 0.43, "一句話：先查表選好做法、再分 GPU；要不要蓋預測器，先量「完美預測」的上限再決定", ha="center", va="center", fontsize=11.5, fontweight="bold")
    fig.savefig(OUT / "brief_o_overview.pdf", bbox_inches="tight", pad_inches=0.1)
    fig.savefig(OUT / "brief_o_overview.svg", bbox_inches="tight", pad_inches=0.1)
    save(fig, "brief_o_overview.png", {"claims": ["12.082 s -> 1.2412 s", "11.0-40.7%", "6.8x", "30.7-62.9%", "6.5 pp", "FreeInference 0.30% (CPU 128 GiB)", "Mooncake 50.5%", "typing hint 0.81"]},
         ["M1", "M3-g", "S2", "B1", "B3-X1", "C2-GPU", "PAPER.md abstract"])


if __name__ == "__main__":
    for f in (fig_overview, fig_story, fig_method, fig_lookup, fig_a, fig_b, fig_c, fig_d, fig_e):
        f()
    (OUT / "brief_sources.json").write_text(json.dumps(SOURCES, ensure_ascii=False, indent=1), encoding="utf-8")
