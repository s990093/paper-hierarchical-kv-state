"""產生 docs/RESEARCH_INTRO_20261006.md 用的圖（輸出到 figures/intro/）。

用法：python3 figures/specs/make_fig_intro.py

資料來源都寫在每張圖的註腳：
- 市場數字：Google I/O 2026 主題演講、The Decoder、OpenRouter 100 兆 token 研究。
- KV 大小：官方 config.json（2 × 層數 × KV heads × head_dim × 2 bytes）；參數量取自 Hugging Face API。
- 頻寬與重算時間：Cake（ICML'25）Fig. 1、Fig. 3 與引言的數字。
- 論文評測長度：各論文原文（見 docs/research_20260924/workloads_eval.md 的頁碼）。
示意圖（fig_intro_wall_schematic、fig_intro_lifecycle、fig_intro_l2_landscape、fig_intro_layers）不含量測數值。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "intro")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": ["PingFang HK", "Heiti TC", "Arial Unicode MS"],
    "font.size": 11,
    "axes.titlesize": 12.5,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# Okabe–Ito
BLUE, VERM, ORNG, GREEN, SKY, PURP, YEL = "#0072B2", "#D55E00", "#E69F00", "#009E73", "#56B4E9", "#CC79A7", "#F0E442"
INK, MUTED, GRID = "#1F2937", "#4B5563", "#D1D5DB"
LIGHT = {"blue": "#D6EAF8", "orng": "#FBE7C6", "green": "#D5EFE3", "grey": "#F3F4F6", "purp": "#EDE7F6", "red": "#FBE3D6"}


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)


def note(fig, text, y=-0.02):
    fig.text(0.01, y, text, fontsize=9, color=MUTED, ha="left", va="top", wrap=True)


# ---------------------------------------------------------------- F1 市場
def fig_market():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.05, 1]})
    # (a) Google 每月 token
    labels = ["2024-05", "2025-05", "2025-10", "2026-05"]
    vals = np.array([9.7e12, 480e12, 1.3e15, 3.2e15])
    txt = ["9.7 兆", "480 兆", "1,300 兆", "3,200 兆"]
    x = np.array([0, 12, 17, 24])  # 距 2024-05 的月數
    a.plot(x, vals, "-o", color=BLUE, lw=2.2, ms=7)
    for xi, v, t in zip(x, vals, txt):
        a.annotate(t, (xi, v), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=10.5, color=INK)
    a.set_yscale("log")
    a.set_ylim(3e12, 1.5e16)
    a.set_xticks(x, labels)
    a.set_xlim(-2, 26)
    a.set_ylabel("每月處理的 token 數（對數）")
    a.set_title("(a) Google 每月處理的 token 數：兩年約 330 倍", loc="left")
    a.grid(axis="y", color=GRID, lw=0.6)
    # (b) OpenRouter 平均長度
    cats = ["輸入（prompt）", "輸出（completion）", "整段序列"]
    start = np.array([1500, 150, 2000])
    end = np.array([6000, 400, 5400])
    xs = np.arange(len(cats))
    w = 0.36
    b.bar(xs - w / 2, start, w, color=LIGHT["blue"], edgecolor=BLUE, label="研究期間起點")
    b.bar(xs + w / 2, end, w, color=BLUE, edgecolor=BLUE, label="研究期間終點")
    for xi, s, e, st, et in zip(xs, start, end, ["≈1,500", "≈150", "<2,000"], ["6,000+", "≈400", "5,400+"]):
        b.text(xi - w / 2, s + 120, st, ha="center", fontsize=10)
        b.text(xi + w / 2, e + 120, et, ha="center", fontsize=10)
    b.text(0, 6900, "×4", ha="center", fontsize=12, color=VERM, weight="bold")
    b.text(1, 1300, "<×3", ha="center", fontsize=12, color=MUTED, weight="bold")
    b.set_xticks(xs, cats)
    b.set_ylim(0, 7800)
    b.set_ylabel("每個請求的平均 token 數")
    b.set_title("(b) OpenRouter：輸入越來越長，輸出沒跟上", loc="left")
    b.legend(frameon=False, loc="upper right")
    b.grid(axis="y", color=GRID, lw=0.6)
    note(fig, "來源：(a) Google I/O 2026 主題演講（2024-05、2025-05、2026-05）；2025-10 的數字引自 The Decoder（2025-10-10）。"
              "token 數有一部分來自推理模型的思考內容，不等於使用者數。 (b) Aubakirova 等，OpenRouter 100 兆 token 研究（arXiv 2601.10088）。")
    fig.tight_layout()
    save(fig, "fig_intro_market")


# ---------------------------------------------------------------- F2 KV 牆
MODELS = [  # 名稱, KV bytes/token（BF16）, BF16 權重 bytes, 顏色
    ("Llama-3.1-8B", 2 * 32 * 8 * 128 * 2, 8.030e9 * 2, GREEN),
    ("Qwen3-30B-A3B（MoE）", 2 * 48 * 4 * 128 * 2, 30.532e9 * 2, ORNG),
    ("Qwen3-32B（dense）", 2 * 64 * 8 * 128 * 2, 32.762e9 * 2, VERM),
    ("Llama-3.1-70B", 2 * 80 * 8 * 128 * 2, 70.554e9 * 2, PURP),
]
GPU_USABLE = 80e9 * 0.9


def kfmt(n):
    return f"{n/1e6:.0f}M" if n >= 1e6 else f"{n/1e3:.0f}K"


def fig_kv_wall():
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.9), gridspec_kw={"width_ratios": [1.1, 1]})
    n = np.logspace(np.log10(8192), np.log10(1.05e6), 200)
    for name, kv, w, c in MODELS:
        a.plot(n, n * kv / 1e9, color=c, lw=2.2, label=f"{name}：{kv/1024:.0f} KiB/token")
        room = GPU_USABLE - w
        if room > 0:
            cap = room / kv
            a.axhline(room / 1e9, color=c, lw=1.1, ls=(0, (5, 4)), alpha=0.9)
            a.plot([cap], [room / 1e9], "o", color=c, ms=7, zorder=5)
            a.annotate(f"≈{cap/1e3:.0f}K token 就滿", (cap, room / 1e9), textcoords="offset points",
                       xytext=(6, -14 if "8B" not in name else 6), fontsize=9.5, color=c)
    a.text(1.0e6, 2.2, "Llama-3.1-70B 的 BF16 權重\n（≈141 GB）單張 80 GB 放不下", ha="right", fontsize=9, color=PURP)
    a.set_xscale("log", base=2)
    a.set_yscale("log")
    ticks = [16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    a.set_xticks(ticks, ["16K", "32K", "64K", "128K", "256K", "512K", "1M"])
    a.set_xlim(12000, 1.1e6)
    a.set_ylim(1, 500)
    a.set_xlabel("一個請求的 context 長度（token）")
    a.set_ylabel("這個請求的 KV 大小（GB，對數）")
    a.set_title("(a) 單一請求的 KV vs 單張 80 GB GPU 扣掉權重後的空間（虛線）", loc="left", fontsize=11.5)
    a.legend(frameon=False, fontsize=9, loc="upper left")
    a.grid(color=GRID, lw=0.6)
    # (b) 500K、70B 級 KV 拿回 GPU 的時間
    kv_bytes = 500_000 * 327_680  # 163.8 GB
    rows = [
        ("仍在 GPU（HBM ≈2 TB/s）\n— 但單張放不下", kv_bytes / 2e12, None, GREEN),
        ("從 CPU 記憶體（≈25 GB/s）", kv_bytes / 25e9, None, BLUE),
        ("從本地 SSD（4 GB/s）", kv_bytes / 4e9, None, ORNG),
        ("從網路儲存（≈1 GB/s）", kv_bytes / 1e9, None, ORNG),
        ("從本地磁碟（0.5 GB/s）", kv_bytes / 0.5e9, None, ORNG),
        ("全部重算（由 Cake 的\n72K ≈ 30 s 推算）", 30 * 500 / 72, 30 * (500 / 72) ** 2, VERM),
    ]
    ys = np.arange(len(rows))[::-1]
    for y, (lab, t, t_hi, c) in zip(ys, rows):
        if t_hi is None:
            b.barh(y, t, color=c, alpha=0.85, height=0.55)
            b.text(t * 1.15, y, human(t), va="center", fontsize=10)
        else:
            b.barh(y, t_hi - t, left=t, color=c, alpha=0.35, height=0.55, hatch="///", edgecolor=c)
            b.text(t_hi * 1.12, y, f"{human(t)}～{human(t_hi)}", va="center", fontsize=10)
    b.set_yticks(ys, [r[0] for r in rows], fontsize=9.8)
    b.set_xscale("log")
    b.set_xlim(0.03, 6000)
    b.set_xticks([0.1, 1, 10, 60, 600, 3600], ["0.1 s", "1 s", "10 s", "1 分", "10 分", "1 時"])
    b.set_title("(b) 使用者回來時，把 500K token 的 KV（70B 級，≈164 GB）拿回 GPU", loc="left", fontsize=11.5)
    b.grid(axis="x", color=GRID, lw=0.6)
    note(fig, "〔算術〕(a) KV/token ＝ 2 × 層數 × KV heads × head_dim × 2 bytes（官方 config）；可用空間 ＝ 80 GB × 0.9 − BF16 權重；"
              "只算一個請求，未計啟動與工作區開銷。 (b) 頻寬取自 Cake 論文 Fig. 1、Fig. 3 的典型伺服器規格；"
              "重算的範圍：相對 72K，長度 ×6.9，線性部分 ×6.9、注意力的二次部分 ×48（Cake 引言：Llama2-70B、A100）。")
    fig.tight_layout()
    save(fig, "fig_intro_kv_wall")


def human(sec):
    if sec < 1:
        return f"{sec:.2f} s"
    if sec < 60:
        return f"{sec:.0f} s"
    if sec < 3600:
        return f"{sec/60:.1f} 分"
    return f"{sec/3600:.1f} 時"


# ---------------------------------------------------------------- F3 情境
def box(ax, x, y, w, h, fc, ec, lw=1.2, r=0.6, z=1):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z)
    ax.add_patch(p)
    return p


def arrow(ax, x1, y1, x2, y2, c=MUTED, lw=1.6, style="-|>", ls="-", ms=12, z=3):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms, color=c, lw=lw, linestyle=ls, zorder=z))


def chunks(ax, x0, y, n=8, w=2.0, gap=0.25, sel=None, fc_fn=None, alpha=1.0):
    xs = {}
    for i in range(n):
        x = x0 + i * (w + gap)
        xs[i] = x
        if sel is not None and i not in sel:
            continue
        fc = fc_fn(i) if fc_fn else LIGHT["blue"]
        ax.add_patch(Rectangle((x, y), w, 3.2, fc=fc, ec=INK, lw=0.8, alpha=alpha, zorder=2))
    return xs


def cost_color(i, n=8):
    cmap = matplotlib.colormaps["Blues"]
    return cmap(0.18 + 0.62 * i / (n - 1))


def fig_lifecycle():
    fig, ax = plt.subplots(figsize=(15, 7.0))
    ax.set_xlim(-17, 100); ax.set_ylim(0, 66); ax.axis("off")
    GY, CY, SY = 49, 37, 25  # 三層的 y
    lanes = [("GPU 記憶體（HBM）", "≈2 TB/s、≈80 GB", GY), ("CPU 記憶體（DRAM）", "≈25 GB/s、≈1.8 TB", CY), ("SSD／遠端儲存", "0.5–4 GB/s、≈26 TB", SY)]
    for name, spec, y in lanes:
        ax.add_patch(Rectangle((11, y - 0.6), 88.5, 5.6, fc="#FAFAFA", ec=GRID, lw=0.8, zorder=0))
        ax.text(-16.5, y + 3.0, name, fontsize=10.5, color=INK, va="center", weight="bold")
        ax.text(-16.5, y + 0.7, spec, fontsize=9, color=MUTED, va="center")
    cols = [(11.5, 31, "第 1 輪：prefill\n長 context（100K+）"), (32.5, 55, "閒置：使用者閱讀／agent 等工具\n（數分鐘～數小時）"),
            (56.5, 87.5, "第 2 輪到達：\n先把 KV 還原回 GPU"), (89, 99.5, "decode")]
    for x0, x1, t in cols:
        ax.text((x0 + x1) / 2, 61.6, t, ha="center", va="center", fontsize=10.5, color=INK, linespacing=1.4)
    for xd in (31.7, 55.8, 88.3):
        ax.plot([xd, xd], [17.0, 58.8], color=GRID, lw=1, ls=(0, (4, 3)), zorder=0)
    # 第 1 輪
    chunks(ax, 12.2, GY + 0.6, fc_fn=cost_color)
    ax.text(12.2, GY + 5.3, "前", fontsize=9, color=MUTED); ax.text(29.4, GY + 5.3, "後", fontsize=9, color=MUTED, ha="right")
    ax.text(21, GY + 5.3, "顏色越深＝重算越貴", ha="center", fontsize=8.8, color=MUTED)
    # 閒置
    for i in range(8):
        ax.add_patch(Rectangle((33.2 + i * 2.6, GY + 0.6), 2.2, 3.2, fc="#E5E7EB", ec=MUTED, lw=0.6, zorder=2))
    ax.text(43.6, GY + 5.3, "GPU 被其他使用者的 KV 佔滿", ha="center", fontsize=9, color=MUTED)
    chunks(ax, 33.4, CY + 0.6, sel=[6, 7], fc_fn=cost_color)
    chunks(ax, 33.4, SY + 0.6, sel=[0, 1, 2, 3, 4, 5], fc_fn=cost_color)
    arrow(ax, 27.5, GY + 0.2, 48.0, CY + 4.3, c=MUTED, lw=1.1, ls=(0, (4, 3)))
    arrow(ax, 18.0, GY + 0.2, 36.0, SY + 4.3, c=MUTED, lw=1.1, ls=(0, (4, 3)))
    ax.text(43.8, 20.6, "現狀：整段全存，閒置久了\n依 LRU 一路往下擠（不看位置）", ha="center", va="center", fontsize=8.8, color=MUTED, linespacing=1.4)
    # 第 2 輪：Cake 式還原
    xs = chunks(ax, 58.5, GY + 0.6, fc_fn=lambda i: LIGHT["blue"] if i < 4 else LIGHT["red"])
    arrow(ax, 58.5, GY + 5.6, 66.9, GY + 5.6, c=BLUE, lw=2)
    ax.text(58.5, GY + 7.3, "GPU 從前往後重算", fontsize=9, color=BLUE)
    ax.text(77.0, GY + 2.2, "← I/O 從後往前載入", fontsize=9, color=VERM, va="center")
    chunks(ax, 58.5, CY + 0.6, sel=[6, 7], fc_fn=cost_color, alpha=0.45)
    chunks(ax, 58.5, SY + 0.6, sel=[4, 5], fc_fn=cost_color, alpha=0.45)
    for i, src_y in zip(range(4, 8), [SY + 3.9, SY + 3.9, CY + 3.9, CY + 3.9]):
        arrow(ax, xs[i] + 1.0, src_y, xs[i] + 1.0, GY + 0.4, c=VERM, lw=1.5)
    ax.plot([67.6, 67.6], [GY - 0.4, GY + 4.6], color="#DC2626", lw=2, ls=(0, (3, 2)), zorder=4)
    ax.text(66.9, GY - 1.6, "會合點", ha="right", fontsize=9, color="#DC2626")
    ax.text(72, 20.6, "Cake：前段算、後段載，兩邊同時跑\n（越後面重算越貴，所以交給 I/O）", ha="center", va="center", fontsize=8.8, color=MUTED, linespacing=1.4)
    # decode
    chunks(ax, 89.6, GY + 0.6, n=4, w=2.0, gap=0.5, fc_fn=lambda i: LIGHT["green"])
    ax.text(94.4, GY - 1.6, "開始產生第一個字", ha="center", fontsize=8.8, color=MUTED)
    # 決策點
    dps = [(11.5, 31, "#7B3F98", "決策點 A（本研究的重點）", "prefill 寫入 KV 時，\n就決定每一段放哪一層、\n用幾 bit、存不存"),
           (32.5, 55, "#B45309", "決策點 B", "閒置期間：降到哪一層、\n要不要提前預取\n（要預測：會不會回來、何時回來）"),
           (56.5, 87.5, BLUE, "決策點 C（Cake／CacheFlow 已處理）", "讀取時：哪一段重算、哪一段載入\n＋兩邊差太多時的退路")]
    for x0, x1, c, t1, t2 in dps:
        box(ax, x0, 1.0, x1 - x0, 15.0, fc="white", ec=c, lw=1.6)
        ax.text((x0 + x1) / 2, 13.2, t1, ha="center", fontsize=10, color=c, weight="bold")
        ax.text((x0 + x1) / 2, 6.6, t2, ha="center", va="center", fontsize=9.1, color=INK, linespacing=1.55)
    ax.set_title("情境：一段長 context 的對話／agent，使用者隔了一段時間才回來", loc="left", fontsize=13)
    note(fig, "示意圖。各層頻寬與容量取自 Cake 論文 Fig. 1 的典型伺服器規格；8 個 chunk 與搬動路徑只是說明用。", y=0.03)
    save(fig, "fig_intro_lifecycle")


# ---------------------------------------------------------------- F4 撞牆示意
def restore_time(n_tokens, load_per_tok, a=1.0, b=1.0 / 65536, N=256):
    """示意用。把長度 n 的 context 切成 N 個 chunk：重算第 i 個 chunk ＝ a·C ＋ 2b·(i·C)·C（線性＋注意力），
    載入每 token 成本固定。回傳 (只重算, 只載入, 雙向會合)。"""
    C = n_tokens / N
    i = np.arange(N)
    c = a * C + 2 * b * (i * C) * C
    pre = np.concatenate([[0], np.cumsum(c)])
    load = (N - np.arange(N + 1)) * C * load_per_tok
    return pre[-1], N * C * load_per_tok, np.maximum(pre, load).min()


def crossing(n, y, level):
    k = np.argmax(np.asarray(y) > level)
    return np.interp(level, [y[k - 1], y[k]], [n[k - 1], n[k]])


def fig_wall_schematic():
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    n = np.logspace(np.log10(8192), np.log10(1.05e6), 160)
    L = 2.4
    rc, lo, ck = map(np.array, zip(*[restore_time(x, L) for x in n]))
    slo = np.interp(150_000, n, ck)
    # 示意：找一個「有效載入成本」讓目標曲線在 500K 碰到同一條上限
    lo_f, hi_f = 1.0, 10.0
    for _ in range(40):
        mid = (lo_f + hi_f) / 2
        tg = np.array([restore_time(x, L / mid)[2] for x in n])
        if crossing(n, tg, slo) < 500_000:
            lo_f = mid
        else:
            hi_f = mid
    tg = np.array([restore_time(x, L / hi_f)[2] for x in n])
    ax.axvspan(4096, 131072, color=SKY, alpha=0.10, zorder=0)
    ax.axvspan(100_000, 200_000, color="#9CA3AF", alpha=0.22, zorder=0)
    ax.plot(n, rc, color=VERM, lw=2, label="全部重算（超線性）")
    ax.plot(n, lo, color=ORNG, lw=2, ls=(0, (6, 3)), label="全部從 SSD 載入（線性）")
    ax.plot(n, ck, color=BLUE, lw=2.6, label="Cake：算＋載並行（不比兩者差）")
    ax.plot(n, tg, color=GREEN, lw=2.6, label="目標：寫入時分層＋預測（斜率變小）")
    ax.axhline(slo, color=INK, lw=1.2, ls=(0, (2, 2)))
    ax.text(8600, slo * 1.15, "互動可接受的 TTFT 上限（示意）", fontsize=9.5, color=INK)
    ax.text(8600, 1.2e8, "現有跨請求 KV 論文的評測範圍\n（多數 ≤ 32K，最長 128K）", fontsize=9.6, color=BLUE, va="top")
    ax.text(141_000, 1.2e8, "撞牆位置\n（假設 100K–200K，\n要先量出來）", ha="center", va="top", fontsize=9.4, color=INK)
    x_c = crossing(n, ck, slo); x_t = crossing(n, tg, slo)
    ax.plot([x_c], [slo], "o", color=BLUE, ms=7, zorder=5)
    ax.plot([x_t], [slo], "o", color=GREEN, ms=7, zorder=5)
    arrow(ax, x_c * 1.05, slo * 0.42, x_t * 0.97, slo * 0.42, c=GREEN, lw=2.2, ms=16)
    ax.text(np.sqrt(x_c * x_t), slo * 0.27, "把牆往後推到 ~500K", ha="center", fontsize=10.5, color=GREEN, weight="bold")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ticks = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    ax.set_xticks(ticks, ["8K", "16K", "32K", "64K", "128K", "256K", "512K", "1M"])
    ax.set_yticks([]); ax.minorticks_off()
    ax.set_xlim(7600, 1.12e6)
    ax.set_ylim(slo / 90, 2.0e8)
    ax.set_xlabel("context 長度（token，對數）")
    ax.set_ylabel("使用者回來時的 TTFT（對數，示意）")
    ax.legend(frameon=True, facecolor="white", edgecolor=GRID, loc="lower right", fontsize=9.8)
    ax.set_title("命題：把「長 context 還原」撞牆的長度往後推（示意圖，不是量測）", loc="left")
    note(fig, "示意圖：曲線形狀來自一階成本模型（重算 ＝ 線性項＋注意力的二次項；載入 ＝ 位元組 ÷ 頻寬；Cake ＝ 讓兩邊較慢者最小的會合點），"
              "數值沒有意義，綠線是「假設能做到」的樣子。撞牆的實際位置、能推多遠，都是待驗證的假設。", y=0.0)
    save(fig, "fig_intro_wall_schematic")


# ---------------------------------------------------------------- F5 L2 兩派
def fig_l2_landscape():
    fig, ax = plt.subplots(figsize=(15, 8.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 74); ax.axis("off")
    # 左：OS／系統派
    box(ax, 0.5, 18, 36, 50, fc="#EEF5FB", ec=BLUE, lw=1.8, r=1.2)
    ax.text(2, 65, "OS／系統派：用規則與成本模型決定", fontsize=12.5, color=BLUE, weight="bold")
    os_items = [
        ("替換", "LRU、ARC", "vLLM、SGLang"),
        ("准入", "用過兩次才寫下一層", "Dynamo、HiCache"),
        ("成本感知逐出", "閒置久＋重算便宜先丟", "Pensieve、AsymCache"),
        ("預取", "看排程佇列提前載入", "AttentionStore"),
        ("批次化、重疊", "搬運與計算錯開", "Strata"),
        ("算＋載並行", "前段算、後段載、會合", "Cake、CacheFlow"),
        ("損益平衡", "多長才值得存", "py-kvcache"),
        ("快速降級", "只用一次的前綴先丟", "Fancy-eviction"),
    ]
    for k, (t, d, ex) in enumerate(os_items):
        y = 60.5 - k * 4.6
        ax.text(2, y, f"• {t}", fontsize=10.2, color=INK, weight="bold")
        ax.text(11.6, y, d, fontsize=9.8, color=INK)
        ax.text(35.6, y, ex, fontsize=9.2, color=MUTED, ha="right")
    ax.text(2, 23.2, "優點：不用訓練、可解釋、換機器只要改成本常數", fontsize=9.8, color=GREEN)
    ax.text(2, 20.2, "缺點：對「未來」只能用最近、次數去猜", fontsize=9.8, color=VERM)
    # 右：預測派
    box(ax, 63.5, 18, 36, 50, fc="#FBF3EA", ec=ORNG, lw=1.8, r=1.2)
    ax.text(65, 65, "預測派：訓練一個模型去猜未來", fontsize=12.5, color="#B45309", weight="bold")
    box(ax, 65, 43, 33, 19.5, fc="white", ec=ORNG, lw=1.2)
    ax.text(66.2, 60.0, "GBDT／輕量模型（CPU、線上重訓）", fontsize=10.5, color="#B45309", weight="bold")
    for k, t in enumerate(["代表：LRB（CDN）、LCR／LARU（SGLang 的", "　　　KV 前綴）、SAECache（線上更新權重）",
                           "猜什麼：多久之後會再被用到", "為什麼重訓：負載會漂移", "　　　（LRB 每累積 128K 筆樣本換一次模型）",
                           "成本：CPU 上跑，不佔 GPU"]):
        ax.text(66.2, 56.8 - k * 2.45, t, fontsize=9.2, color=INK)
    box(ax, 65, 19.5, 33, 22, fc="white", ec=VERM, lw=1.2)
    ax.text(66.2, 39.0, "深度學習（GPU、離線、綁定 LLM）", fontsize=10.5, color=VERM, weight="bold")
    for k, t in enumerate(["代表：LPC（118M 文字嵌入模型）；KVP（每個", "　　　KV head 一個 RL agent）、ForesightKV、", "　　　LookaheadKV、TRIM-KV",
                           "猜什麼：對話會不會繼續；哪些 token 重要", "為什麼重訓：換一個 LLM 就要重收資料、重訓",
                           "成本：KVP 收 trace 用 7 台 × 8 GPU、訓練 8×H100", "註：KVP 等四篇是單一請求內丟 token（有損），不是分層"]):
        ax.text(66.2, 36.0 - k * 2.35, t, fontsize=9.0, color=INK)
    # 中：交集
    box(ax, 38.5, 30, 23, 26, fc="#F3EEF9", ec="#7B3F98", lw=2.0, r=1.2, z=4)
    ax.text(50, 52.6, "交集：學習增強", ha="center", fontsize=12, color="#7B3F98", weight="bold", zorder=5)
    ax.text(50, 49.4, "預測給建議、規則保底", ha="center", fontsize=10, color=INK, zorder=5)
    for k, t in enumerate(["LARU：預測錯時退回 LRU", "（有理論保證）", "Marconi：剛啟動時先用 LRU", "HALP：LRU 先篩、模型再排", "理論：Lykouris &", "Vassilvitskii（ICML'18）"]):
        ax.text(50, 45.4 - k * 2.5, t, ha="center", fontsize=9.3, color=INK, zorder=5)
    arrow(ax, 36.8, 43, 38.3, 43, c=BLUE, lw=2, ms=14)
    arrow(ax, 63.2, 43, 61.7, 43, c=ORNG, lw=2, ms=14)
    # 下：本研究
    box(ax, 0.5, 1.5, 99, 13.5, fc="white", ec="#7B3F98", lw=2.2, r=1.2)
    ax.text(2, 12.1, "本研究想站的位置", fontsize=11.5, color="#7B3F98", weight="bold")
    ax.text(2, 8.6, "時機：寫入時（prefill 結束）　範圍：依位置 × 多層（GPU 的 BF16／FP8／INT4、CPU、SSD、不存）× 100K–500K", fontsize=10, color=INK)
    ax.text(2, 5.6, "分工：硬體相關的成本（重算多貴、各層搬多快）用量測常數；模型只預測與硬體無關的「負載」（會不會回來、何時回來、讀取時 GPU 多忙）", fontsize=10, color=INK)
    ax.text(2, 2.8, "→ 換一台機器只要重量幾個常數，不必重訓；預測不準時退回規則", fontsize=10, color=INK)
    ax.set_title("L2（放在哪一層）的兩派做法與交集", loc="left", fontsize=13.5)
    note(fig, "整理自各篇原文。LRB：NSDI'20；LCR／LARU：arXiv 2509.20979；SAECache：arXiv 2605.18825；LPC：NeurIPS'25（118M 嵌入模型的描述引自 SAECache）；"
              "KVP：ICML'26 附錄 A.3；HALP：NSDI'23；Fancy-eviction：arXiv 2609.28870。", y=0.04)
    save(fig, "fig_intro_l2_landscape")


# ---------------------------------------------------------------- F6 L0–L5
def fig_layers():
    fig, ax = plt.subplots(figsize=(13.5, 6.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 64); ax.axis("off")
    rows = [
        ("L5", "誰在用、用多久？", "使用方式、快取契約（保留多久、怎麼計價）", "商業 API 的快取契約、agent 工作流、Keepalive"),
        ("L4", "怎麼找到筆記？", "命中規則：只認前綴，還是中段也能用", "CacheBlend、MEPIC、推理模型思考內容的 KV 失效"),
        ("L3", "怎麼搬筆記？", "傳輸：CPU／SSD／網路 → GPU", "Strata、vLLM 佈局改版、Tutti、Bottlenecks"),
        ("L2", "筆記放哪裡？", "放置、逐出、重算：GPU／CPU／SSD／不存", "Cake、CacheFlow、Pensieve、Bidaw、LMCache、Mooncake、LPC、LARU"),
        ("L1", "筆記寫多細？", "精度：BF16／FP8／INT4", "KIVI、KVTuner、QuantSpec、VeriCache"),
        ("L0", "要存的是什麼筆記？", "物件：KV、hybrid 模型的狀態", "Marconi、DASC、Sparse Prefix Caching"),
    ]
    for k, (lv, q, what, ex) in enumerate(rows):
        y = 54 - k * 9.2
        hl = lv == "L2"
        box(ax, 1, y, 82, 7.6, fc="#F3EEF9" if hl else "white", ec="#7B3F98" if hl else GRID, lw=2.4 if hl else 1.0, r=0.8)
        ax.text(3, y + 3.8, lv, fontsize=15, weight="bold", color="#7B3F98" if hl else INK, va="center")
        ax.text(9, y + 5.2, q, fontsize=11.5, weight="bold", color=INK, va="center")
        ax.text(9, y + 2.2, what, fontsize=9.8, color=MUTED, va="center")
        ax.text(81.5, y + 3.8, ex, fontsize=9.6, color=INK, va="center", ha="right")
    ax.text(84.5, 54 - 3 * 9.2 + 5.6, "◀ 論文最多、最擠", fontsize=10.5, color="#7B3F98", weight="bold", va="center")
    ax.text(84.5, 54 - 3 * 9.2 + 2.2, "本研究的核心", fontsize=10.5, color="#7B3F98", va="center")
    # 連帶的層
    for lv_idx, t in [(4, "連帶：依位置選精度"), (2, "連帶：每筆固定成本、限速"), (1, "連帶：前段不存 → 要「區段命中」")]:
        y = 54 - lv_idx * 9.2 + 3.8
        ax.text(84.5, y, f"◀ {t}", fontsize=9.6, color=MUTED, va="center")
    ax.set_title("這個領域都在做 KV cache 優化，但切入的層次不同：L0–L5", loc="left", fontsize=13.5)
    note(fig, "比喻：KV cache 是模型讀過一段文字後留下的「筆記」；重讀一次（重算）就是 prefill。各層的代表作見正文的 SOTA 總表。", y=0.03)
    save(fig, "fig_intro_layers")


# ---------------------------------------------------------------- F7 評測長度
def fig_eval_lengths():
    fig, ax = plt.subplots(figsize=(12.5, 6.3))
    rows = [  # 標籤, 低, 高, 類別, 註
        ("Bidaw（FAST'26）", 1066, 8168, "x", "公開 trace 的 p50–p99"),
        ("Cake（ICML'25）", 4096, 16384, "x", "合成 prompt"),
        ("HCache（EuroSys'25）", None, 16384, "x", "最長 16K"),
        ("LMCache（2025）", 10000, 20000, "x", "多輪 QA，每次查詢"),
        ("CachedAttention（ATC'24）", 2048, 32768, "x", "視窗 2K–4K，session ≤32K"),
        ("Strata（OSDI'26）", 681, 54797, "x", "各資料集的平均輸入"),
        ("Mooncake（FAST'25）", 8192, 131072, "x", "128K 是模擬資料"),
        ("CacheFlow（2026）", None, 131072, "x", "最長 128K"),
        ("Tutti（2026）", 3000, 200000, "io", "只處理 SSD→GPU 的搬運"),
        ("KVDrive（2026）", 60000, 360000, "io", "單一請求，沒有跨請求重用"),
    ]
    ys = np.arange(len(rows))[::-1]
    refs = [(18_700, GREEN, (0, (4, 3))), (32_000, GREEN, (0, (4, 3))), (124_018, GREEN, "-"), (500_000, "#DC2626", "-")]
    for x, c, ls in refs:
        ax.axvline(x, color=c, lw=1.5, ls=ls, zorder=1)
    top = len(rows) - 0.1
    ax.text(24_500, top, "生產 trace 平均\n18.7K／32.0K", ha="center", va="bottom", fontsize=9, color=GREEN, bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none"))
    ax.text(124_018, top, "agent 寫程式 trace\n中位數 124K", ha="center", va="bottom", fontsize=9, color=GREEN, bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none"))
    ax.text(500_000, top, "本研究目標\n500K", ha="center", va="bottom", fontsize=9, color="#DC2626", bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none"))
    for y, (lab, lo, hi, kind, nt) in zip(ys, rows):
        c = BLUE if kind == "x" else "#6B7280"
        if lo is None:
            ax.plot([hi], [y], marker="D", color=c, ms=8, zorder=3)
        else:
            ax.plot([lo, hi], [y, y], color=c, lw=7, solid_capstyle="butt", alpha=0.9, zorder=2)
        ax.text(hi * 1.18, y, nt, va="center", fontsize=9.2, color=MUTED, zorder=4,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.9))
    ax.set_yticks(ys, [r[0] for r in rows], fontsize=10)
    ax.set_xscale("log")
    ax.set_xlim(600, 1.6e6)
    ax.set_ylim(-0.7, len(rows) + 1.3)
    ax.set_xticks([1e3, 4096, 16384, 65536, 131072, 262144, 524288, 1048576], ["1K", "4K", "16K", "64K", "128K", "256K", "512K", "1M"])
    ax.minorticks_off()
    ax.set_xlabel("評測用到的 context 長度（token，對數）")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_title("現有論文評測到多長？跨請求重用＋分層的論文（藍）最長 128K；灰色只處理搬運或單一請求", loc="left", fontsize=12)
    note(fig, "整理自各篇原文（頁碼見 docs/research_20260924/workloads_eval.md）；◆ 表示原文只給上限。生產 trace 平均取自 arXiv 2609.28870（Chutes 18.7K、FreeInference 32.0K）；"
              "agent trace 中位數為我們從 TraceLab 公開資料算出（60.9% 的呼叫 ≥ 100K）。Bidaw 的 p50–p99 也是從其公開 trace 算出。")
    fig.tight_layout()
    save(fig, "fig_intro_eval_lengths")


if __name__ == "__main__":
    fig_market()
    fig_kv_wall()
    fig_lifecycle()
    fig_wall_schematic()
    fig_l2_landscape()
    fig_layers()
    fig_eval_lengths()
    print("written to", os.path.abspath(OUT))
