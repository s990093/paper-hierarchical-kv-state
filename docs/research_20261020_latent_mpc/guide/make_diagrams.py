"""產生入門導讀 PDF 用的示意圖與算術圖。

用法：python make_diagrams.py OUT_DIR
所有數字圖都只畫公式算出來的值（〔算術〕）或標明「示意」的玩具例子，沒有任何量測值。
"""
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

FONT_DIR = Path("/mlsteam/data/tiara/fonts")
for f in ["NotoSansCJKtc-Regular.otf", "NotoSansCJKtc-Bold.otf"]:
    fm.fontManager.addfont(str(FONT_DIR / f))
plt.rcParams["font.family"] = ["Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 跟 plan.html 同一套顏色
INK = "#171a24"
MUTED = "#5b6070"
RULE = "#d8dbe4"
LAT = "#3d4fc0"
LAT_SOFT = "#e4e7f8"
TXT = "#b4521e"
TXT_SOFT = "#f8e5d9"
GATE = "#8f5b00"
GATE_SOFT = "#f6ebd2"
GOOD = "#2e7d4f"
GOOD_SOFT = "#dcefe3"
BAD = "#b3412e"
BAD_SOFT = "#f6dcd6"
PANEL = "#ffffff"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "diagrams")
OUT.mkdir(parents=True, exist_ok=True)


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w * 100)
    ax.set_ylim(0, h * 100)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text="", fc=PANEL, ec=MUTED, fs=11, bold=False, color=INK, ls="-", lw=1.4, r=8, sub=None, subfs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls))
    if sub:
        ax.text(x + w / 2, y + h * 0.62, text, ha="center", va="center", fontsize=fs, color=color, fontweight="bold" if bold else "normal")
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=subfs, color=MUTED)
    elif text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, fontweight="bold" if bold else "normal", linespacing=1.4)


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.4, ls="-", rad=0.0, head=10):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=f"-|>,head_length={head/10*0.6},head_width={head/10*0.35}",
                                 connectionstyle=f"arc3,rad={rad}", color=color, lw=lw, ls=ls, mutation_scale=12))


def label(ax, x, y, s, fs=10, color=INK, ha="center", va="center", bold=False, **kw):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, fontweight="bold" if bold else "normal", **kw)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- 1. 故事：小明的兩種作法
def fig_story():
    fig, ax = canvas(10, 4.6)
    label(ax, 230, 440, "作法一：邊寫邊想（就像 CoT）", 13, bold=True)
    label(ax, 730, 440, "作法二：先在腦中試幾步，再下筆（就像規劃）", 13, bold=True)
    ax.plot([470, 470], [20, 450], color=RULE, lw=1)
    # 左：一條直線走到底
    ys = [370, 300, 230, 160, 80]
    names = ["題目", "第 1 步", "第 2 步（算錯了）", "第 3 步", "答案"]
    for i, (yy, n) in enumerate(zip(ys, names)):
        fc = BAD_SOFT if i in (2, 3, 4) else TXT_SOFT
        ec = BAD if i in (2, 3, 4) else TXT
        box(ax, 130, yy - 22, 200, 44, n, fc=fc, ec=ec, fs=11)
        if i:
            arrow(ax, 230, ys[i - 1] - 22, 230, yy + 22, color=TXT)
    label(ax, 345, 80, "✗", 20, color=BAD, ha="left")
    label(ax, 230, 30, "寫下去的每一步都收不回來；第 2 步錯，後面全錯。", 10, color=MUTED)
    # 右：三個候選，各自想像
    box(ax, 520, 360, 170, 44, "已經寫好的部分", fc=TXT_SOFT, ec=TXT, fs=11)
    cand = [("候選 A", 0.2, 560), ("候選 B", 0.7, 730), ("候選 C", 0.4, 900)]
    for n, sc, cx in cand:
        best = n == "候選 B"
        arrow(ax, 605, 360, cx, 290, color=TXT)
        box(ax, cx - 62, 248, 124, 40, n, fc=GOOD_SOFT if best else TXT_SOFT, ec=GOOD if best else TXT, fs=11, bold=best)
        ax.add_patch(Ellipse((cx, 165), 150, 90, fc=LAT_SOFT, ec=LAT, lw=1.2, ls="--"))
        label(ax, cx, 182, "腦中想像：", 9, color=LAT)
        label(ax, cx, 162, "接下來會怎樣？", 9, color=LAT)
        label(ax, cx, 138, f"看起來會對 {sc:.1f}", 10, color=GOOD if best else INK, bold=best)
        arrow(ax, cx, 248, cx, 212, color=LAT, ls="--")
    label(ax, 730, 80, "選 B 寫下去 → 寫完這一步，再重新想下一步", 11, color=GOOD, bold=True)
    label(ax, 730, 40, "（分數是示意，不是任何實驗結果）", 9, color=MUTED)
    save(fig, "fig_story.png")


# ---------------------------------------------------------------- 2. LLM 內部 + KV cache
def fig_llm():
    fig, ax = canvas(11.5, 5.2)
    toks = ["Betty", "有", "50", "元", "，"]
    xs = [160 + i * 90 for i in range(len(toks))]
    for x, t in zip(xs, toks):
        box(ax, x - 38, 20, 76, 34, t, fc=TXT_SOFT, ec=TXT, fs=10)
        ax.add_patch(FancyBboxPatch((x - 30, 80), 60, 18, boxstyle="round,pad=0,rounding_size=4", fc="#eef0f5", ec=MUTED, lw=0.8))
        arrow(ax, x, 54, x, 80, color=MUTED, head=7)
    label(ax, 105, 37, "token", 9, color=MUTED, ha="right")
    label(ax, 105, 89, "embedding", 9, color=MUTED, ha="right")
    # 層
    layers = [("第 1 層", 120), ("…", 175), ("第 ℓ 層", 230), ("…", 285), ("第 L 層", 340)]
    for name, yy in layers:
        if name == "…":
            label(ax, 340, yy + 16, "⋮", 16, color=MUTED)
            continue
        hl = "ℓ" in name
        box(ax, 112, yy, 456, 36, "", fc=LAT_SOFT if hl else "#f3f4f8", ec=LAT if hl else MUTED, lw=1.8 if hl else 1.0)
        label(ax, 105, yy + 18, name, 9, color=LAT if hl else MUTED, ha="right")
        for x in xs:
            ax.add_patch(FancyBboxPatch((x - 26, yy + 8), 52, 20, boxstyle="round,pad=0,rounding_size=3", fc=PANEL, ec=LAT if hl else RULE, lw=0.8))
    arrow(ax, 340, 98, 340, 120, color=MUTED)
    label(ax, 590, 262, "這一層、這個位置的向量", 10, color=LAT, ha="left", bold=True)
    label(ax, 590, 240, "＝ hidden state（隱藏狀態）", 10, color=LAT, ha="left", bold=True)
    label(ax, 590, 218, "＝ 本文說的「狀態 zₖ」", 10, color=LAT, ha="left", bold=True)
    arrow(ax, 586, 248, xs[-1] + 30, 248, color=LAT)
    # LM head 與機率
    box(ax, xs[-1] - 60, 392, 120, 32, "LM head", fc="#eef0f5", ec=MUTED, fs=10)
    arrow(ax, xs[-1], 376, xs[-1], 392, color=MUTED)
    probs = [("的", 0.42), ("但", 0.21), ("她", 0.18), ("…", 0.19)]
    for i, (w, p) in enumerate(probs):
        bx = 150 + i * 62
        ax.add_patch(FancyBboxPatch((bx, 420), 24, 60 * p / 0.42, boxstyle="round,pad=0,rounding_size=2", fc=TXT, ec="none"))
        label(ax, bx + 12, 408, w, 9)
    label(ax, 260, 500, "下一個 token 的機率（示意）", 9, color=MUTED)
    arrow(ax, xs[-1] - 60, 408, 405, 440, color=MUTED)
    # KV cache
    kx, ky = 830, 60
    label(ax, kx + 140, 500, "KV cache", 12, bold=True)
    label(ax, kx + 140, 476, "每一層、每個 token 都存一份 K 和 V", 10, bold=True)
    for li in range(6):
        for ti in range(5):
            x = kx + ti * 56
            y = ky + li * 55
            ax.add_patch(FancyBboxPatch((x, y + 18), 24, 22, boxstyle="round,pad=0,rounding_size=3", fc="#dfe3f7", ec=LAT, lw=0.6))
            ax.add_patch(FancyBboxPatch((x + 26, y + 18), 24, 22, boxstyle="round,pad=0,rounding_size=3", fc="#fbe7da", ec=TXT, lw=0.6))
        label(ax, kx - 10, ky + li * 55 + 29, f"層 {li+1 if li < 5 else 'L'}", 8, color=MUTED, ha="right")
    for ti, t in enumerate(toks):
        label(ax, kx + ti * 56 + 25, ky + 6, t, 8, color=MUTED)
    label(ax, kx + 140, 448, "K＝藍、V＝橘。新 token 只算新的一欄，", 8.8, color=MUTED)
    label(ax, kx + 140, 430, "舊的直接讀，不用重算。", 8.8, color=MUTED)
    label(ax, kx + 140, 408, "Llama-3.1-8B：每 token 128 KiB", 9.2, color=INK, bold=True)
    label(ax, kx + 140, 388, "4,096 token ＝ 512 MiB", 9.2, color=INK, bold=True)
    save(fig, "fig_llm.png")


# ---------------------------------------------------------------- 3. 讓模型多想一點的五種方法
def fig_tts():
    fig, axes = plt.subplots(1, 5, figsize=(12, 3.4))
    titles = ["(a) 直接 CoT", "(b) Self-consistency\n多條路投票", "(c) Best-of-N\n評分器挑一條", "(d) 步驟 beam\n每一步都剪枝", "(e) 樹搜尋／前瞻\n往下試再回來"]
    for ax, t in zip(axes, titles):
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis("off")
        ax.set_aspect("equal")
        ax.set_title(t, fontsize=10.5, color=INK, pad=4)

    def node(ax, x, y, c=TXT, fc=TXT_SOFT, r=4):
        ax.add_patch(plt.Circle((x, y), r, fc=fc, ec=c, lw=1.2, zorder=3))

    def edge(ax, a, b, c=TXT, ls="-", lw=1.2):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=c, lw=lw, ls=ls, zorder=1)

    ax = axes[0]
    pts = [(50, 88), (50, 66), (50, 44), (50, 22)]
    for i, p in enumerate(pts):
        node(ax, *p)
        if i:
            edge(ax, pts[i - 1], p)
    label(ax, 50, 8, "成本：1 條", 9, color=MUTED)
    ax = axes[1]
    node(ax, 50, 88)
    for x in [18, 38, 62, 82]:
        prev = (50, 88)
        for y in [62, 38]:
            node(ax, x, y)
            edge(ax, prev, (x, y))
            prev = (x, y)
        edge(ax, prev, (50, 16), c=MUTED, ls="--")
    ax.add_patch(FancyBboxPatch((38, 10), 24, 12, boxstyle="round,pad=0,rounding_size=2", fc=GOOD_SOFT, ec=GOOD))
    label(ax, 50, 16, "投票", 8, color=GOOD)
    label(ax, 50, 2, "成本：N 條", 9, color=MUTED)
    ax = axes[2]
    node(ax, 50, 88)
    for i, x in enumerate([18, 38, 62, 82]):
        prev = (50, 88)
        for y in [62, 38]:
            best = i == 2
            node(ax, x, y, c=GOOD if best else TXT, fc=GOOD_SOFT if best else TXT_SOFT)
            edge(ax, prev, (x, y), c=GOOD if best else TXT)
            prev = (x, y)
        label(ax, x, 26, ["0.2", "0.4", "0.9", "0.1"][i], 8, color=GOOD if best else MUTED)
    label(ax, 50, 14, "評分器（verifier／PRM）", 8, color=GATE)
    label(ax, 50, 2, "成本：N 條＋評分", 9, color=MUTED)
    ax = axes[3]
    node(ax, 50, 90)
    lvl1 = [18, 50, 82]
    for i, x in enumerate(lvl1):
        keep = i != 2
        node(ax, x, 66, c=TXT if keep else MUTED, fc=TXT_SOFT if keep else "#eeeeee")
        edge(ax, (50, 90), (x, 66), c=TXT if keep else MUTED)
        if not keep:
            label(ax, x + 9, 66, "✗", 10, color=BAD)
    for x0 in [18, 50]:
        for dx in [-10, 10]:
            keep = (x0, dx) in [(18, 10), (50, 10)]
            node(ax, x0 + dx, 42, c=TXT if keep else MUTED, fc=TXT_SOFT if keep else "#eeeeee")
            edge(ax, (x0, 66), (x0 + dx, 42), c=TXT if keep else MUTED)
    label(ax, 50, 20, "每步用 value／PRM 留前 k 名", 8, color=GATE)
    label(ax, 50, 2, "成本：每步 N 個候選", 9, color=MUTED)
    ax = axes[4]
    node(ax, 50, 90)
    for x in [25, 50, 75]:
        node(ax, x, 68)
        edge(ax, (50, 90), (x, 68))
        for yy in [50, 34, 18]:
            node(ax, x, yy, c=LAT, fc=LAT_SOFT, r=2.6)
        edge(ax, (x, 68), (x, 18), c=LAT, ls=":")
    label(ax, 50, 4, "成本：每個候選再往下試 H 步", 9, color=MUTED)
    fig.text(0.5, 0.005, "藍色虛線＝為了評估而「試著往下走」的部分。我們的計劃就是想把這一段改成在小模型裡想像。", ha="center", fontsize=9.5, color=LAT)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.08, wspace=0.08)
    save(fig, "fig_tts.png")


# ---------------------------------------------------------------- 4. MPC
def fig_mpc():
    fig, ax = canvas(10, 3.8)
    for k, x0 in enumerate([30, 575]):
        box(ax, x0, 160, 90, 50, f"狀態 s{'ₜ' if k == 0 else 'ₜ₊₁'}", fc=TXT_SOFT, ec=TXT, fs=11, bold=True)
        ends = [(x0 + 330, 300), (x0 + 330, 230), (x0 + 330, 160), (x0 + 330, 90)]
        scores = [0.3, 0.8, 0.5, 0.2]
        for (ex, ey), sc in zip(ends, scores):
            best = sc == max(scores)
            xs = np.linspace(x0 + 90, ex, 4)
            ys = np.linspace(185, ey, 4)
            ax.plot(xs, ys, color=GOOD if best else LAT, lw=1.6 if best else 1.0, ls="-" if best else "--")
            ax.scatter(xs[1:], ys[1:], s=16, color=GOOD if best else LAT, zorder=3)
            label(ax, ex + 12, ey, f"{sc:.1f}", 9, color=GOOD if best else MUTED, ha="left")
        label(ax, x0 + 210, 340, "① 想像 4 條、每條 H＝3 步的動作序列", 9.5, color=LAT)
        label(ax, x0 + 210, 40, "② 只執行最好那條的第 1 步", 9.5, color=GOOD, bold=True)
        ax.plot([x0 + 90, x0 + 90 + 80], [185, 185 + (300 - 185) / 3 * 0 + (230 - 185) / 3], color=GOOD, lw=4, alpha=0.35, solid_capstyle="round")
    arrow(ax, 470, 185, 570, 185, color=GOOD, lw=2)
    label(ax, 520, 210, "③ 觀察新狀態", 9, color=GOOD)
    label(ax, 520, 160, "④ 重新規劃", 9, color=GOOD)
    label(ax, 500, 8, "這就是 MPC（模型預測控制）：每一步都重新想，所以想像錯一點也還能修正。分數是示意。", 9.5, color=MUTED)
    save(fig, "fig_mpc.png")


# ---------------------------------------------------------------- 5. CEM 玩具例子 + 維度詛咒
def fig_cem():
    rng = np.random.default_rng(7)
    target = np.array([1.6, -1.0])

    def energy(x):
        d = x - target
        return (d ** 2).sum(-1) + 0.3 * np.sin(3 * x[..., 0]) * np.cos(3 * x[..., 1])

    gx, gy = np.meshgrid(np.linspace(-3.2, 3.2, 200), np.linspace(-3.2, 3.2, 200))
    E = energy(np.stack([gx, gy], -1))
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.4), gridspec_kw={"width_ratios": [1, 1, 1, 1.35]})
    mu, sd = np.zeros(2), np.ones(2) * 1.5
    for it in range(3):
        ax = axes[it]
        ax.contourf(gx, gy, E, levels=18, cmap="Blues_r", alpha=0.55)
        x = rng.normal(mu, sd, size=(64, 2))
        e = energy(x)
        elite = x[np.argsort(e)[:8]]
        ax.scatter(x[:, 0], x[:, 1], s=10, color=MUTED, alpha=0.6, label="64 個樣本")
        ax.scatter(elite[:, 0], elite[:, 1], s=22, color=S2, label="前 8 名（菁英）", zorder=3)
        ax.add_patch(Ellipse(mu, 4 * sd[0], 4 * sd[1], fc="none", ec=INK, lw=1.2, ls="--"))
        ax.scatter([target[0]], [target[1]], marker="*", s=120, color=GOOD, zorder=4)
        ax.set_title(f"第 {it+1} 輪", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-3.2, 3.2)
        mu, sd = elite.mean(0), elite.std(0) + 0.05
        if it == 0:
            ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    # 維度詛咒：樣本落在最佳點半徑 1 內的機率
    ax = axes[3]
    dims = np.array([1, 2, 4, 7, 8, 16, 32, 64])
    from math import gamma

    def chi2_cdf(r2, k, terms=400):
        # P(chi2_k <= r2) = 正規化的下不完全 gamma P(k/2, r2/2)，用級數展開
        a, x = k / 2.0, r2 / 2.0
        s, term = 0.0, 1.0 / a
        for n in range(terms):
            s += term
            term *= x / (a + n + 1)
        return math.exp(-x + a * math.log(x) - math.lgamma(a)) * s

    p = np.array([chi2_cdf(1.0, int(d)) for d in dims])
    ax.semilogy(dims, p, color=S1, lw=2, marker="o", ms=5)
    for d, v in zip(dims, p):
        if d in (2, 7, 32, 64):
            ax.annotate(f"{v:.1e}" if v < 0.01 else f"{v:.2f}", (d, v), textcoords="offset points", xytext=(-46 if d == 64 else 6, 4), fontsize=8, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims], fontsize=8)
    ax.set_xlabel("動作的維度", fontsize=9)
    ax.set_ylabel("落在目標半徑 1 內的機率", fontsize=8.5)
    ax.set_ylim(1e-50, 30)
    ax.set_title("維度越高，亂抽越難抽中", fontsize=11)
    ax.grid(alpha=0.3)
    ax.axvline(7, color=GOOD, ls=":", lw=1)
    ax.text(7.4, 1e-38, "V-JEPA 2-AC\n的 7 維", fontsize=8, color=GOOD)
    fig.text(0.5, 0.01, "左三格：玩具 2 維能量函數上的 CEM（★＝最低點，虛線橢圓＝目前的高斯分布）。右：x～N(0, I) 時 P(‖x‖ ≤ 1)，以卡方分布公式計算〔算術〕。", ha="center", fontsize=8.8, color=MUTED)
    fig.subplots_adjust(left=0.01, right=0.985, top=0.88, bottom=0.18, wspace=0.12)
    pos = axes[3].get_position(); axes[3].set_position([pos.x0 + 0.045, pos.y0, pos.width - 0.045, pos.height])
    save(fig, "fig_cem.png")
    return dict(zip(dims.tolist(), p.tolist()))


# ---------------------------------------------------------------- 6. 生成式 vs JEPA
def fig_jepa():
    fig, ax = canvas(10, 4.2)
    label(ax, 240, 395, "生成式：在原始空間預測", 12.5, bold=True, color=TXT)
    label(ax, 750, 395, "JEPA：在表徵空間預測", 12.5, bold=True, color=LAT)
    ax.plot([495, 495], [110, 400], color=RULE)
    box(ax, 40, 260, 120, 60, "看到的部分", sub="影格／前文", fc=TXT_SOFT, ec=TXT)
    box(ax, 200, 260, 120, 60, "生成器", fc="#eef0f5", ec=MUTED)
    box(ax, 360, 260, 120, 60, "預測的像素／字", fc=TXT_SOFT, ec=TXT, fs=10)
    arrow(ax, 160, 290, 200, 290)
    arrow(ax, 320, 290, 360, 290)
    box(ax, 360, 140, 120, 60, "真正的像素／字", fc=TXT_SOFT, ec=TXT, fs=10)
    arrow(ax, 420, 200, 420, 258, color=BAD, ls="--")
    label(ax, 430, 228, "逐像素比", 9, color=BAD, ha="left")
    label(ax, 250, 85, "連每片樹葉、每個用字都要猜對，", 9.5, color=MUTED)
    label(ax, 250, 63, "成本很高，也浪費在猜不到的細節上。", 9.5, color=MUTED)
    box(ax, 530, 260, 110, 60, "看到的部分", fc=TXT_SOFT, ec=TXT, fs=10)
    box(ax, 660, 260, 90, 60, "encoder", fc=LAT_SOFT, ec=LAT, fs=10)
    box(ax, 770, 260, 100, 60, "predictor", fc=LAT_SOFT, ec=LAT, fs=10)
    box(ax, 885, 260, 70, 60, "ẑ", fc=LAT_SOFT, ec=LAT, fs=13, bold=True)
    for a_, b_ in [(640, 660), (750, 770), (870, 885)]:
        arrow(ax, a_, 290, b_, 290, color=LAT)
    box(ax, 530, 140, 110, 60, "要預測的部分", fc=TXT_SOFT, ec=TXT, fs=10)
    box(ax, 660, 140, 140, 60, "target encoder", sub="EMA 或凍結", fc="#eef0f5", ec=MUTED, ls="--")
    box(ax, 885, 140, 70, 60, "z", fc="#eef0f5", ec=MUTED, fs=13, bold=True)
    arrow(ax, 640, 170, 660, 170)
    arrow(ax, 800, 170, 885, 170)
    arrow(ax, 920, 200, 920, 258, color=LAT, ls="--")
    label(ax, 930, 228, "L1", 10, color=LAT, ha="left", bold=True)
    label(ax, 745, 100, "只比「抽象的特徵」。但如果 target 也在學，", 9.2, color=MUTED)
    label(ax, 745, 80, "模型可以把所有東西都映射到同一點來作弊（崩塌）。", 9.2, color=MUTED)
    label(ax, 745, 50, "解法一：EMA＋stop-grad（V-JEPA 2 第一階段）", 9.2, color=LAT, bold=True)
    label(ax, 745, 28, "解法二：直接凍結 target（V-JEPA 2-AC 第二階段、本計劃）", 9.2, color=LAT, bold=True)
    save(fig, "fig_jepa.png")


# ---------------------------------------------------------------- 7. 類比斷點：H=1
def fig_h1():
    fig, ax = canvas(10, 3.9)
    label(ax, 500, 370, "同一個「想像一步」，在機器人和 LLM 上的價值完全不同", 13, bold=True)
    for row, (yy, who) in enumerate([(250, "機器手臂"), (100, "LLM")]):
        label(ax, 60, yy + 30, who, 12, bold=True, color=INK)
        box(ax, 130, yy, 300, 64, "想像這一步會怎樣", sub="world model 16 秒（V-JEPA 2-AC）" if row == 0 else "小 predictor：一次前向", fc=LAT_SOFT, ec=LAT)
        box(ax, 560, yy, 380, 64, "真的做這一步", sub="手臂動了，杯子倒了就回不去" if row == 0 else "一次 prefill；可以從 KV 分叉、隨時退回", fc=BAD_SOFT if row == 0 else GOOD_SOFT, ec=BAD if row == 0 else GOOD)
        label(ax, 495, yy + 32, "vs", 12, color=MUTED)
    label(ax, 750, 225, "又貴又不可逆 → 一定要先想像", 10, color=BAD, bold=True)
    label(ax, 750, 75, "便宜又可逆 → 做完還能拿到真實的 zₖ₊₁", 10, color=GOOD, bold=True)
    label(ax, 500, 30, "結論：H＝1 時，想像出來的 ẑₖ₊₁ 比不上免費拿到的真實 zₖ₊₁。世界模型只有在 H ≥ 2（往更後面想）時才可能有價值。", 10, color=LAT)
    save(fig, "fig_h1.png")


# ---------------------------------------------------------------- 8. 文獻地圖
def fig_litmap():
    cols = ["完全免訓練", "只訓練小頭\n（目標 LLM 凍結）", "微調 base", "從頭預訓練"]
    rows = ["世界模型＋規劃", "潛空間最佳化\n（梯度／PG／BO）", "取樣＋重排\n／提早停", "不搜尋"]
    cells = {
        (0, 0): [("RAP（文字空間 MCTS）", "n")],
        (0, 1): [("★ 方向 A：規劃推理步驟", "us"), ("Thoughts-as-Planning：\n規劃 prompt 編輯", "hi")],
        (0, 2): [("CoLA（MCTS on 潛動作）", "hi")],
        (0, 3): [],
        (1, 0): [("Soft Reasoning（BO）", "hi"), ("LatentSeek（PG）", "n"), ("LTPO（PG）", "n"), ("∇-Reasoner", "n")],
        (1, 1): [("LTO（潛獎勵模型）", "hi")],
        (1, 2): [],
        (1, 3): [("EBT（能量最小化）", "n")],
        (2, 0): [("LED", "n"), ("Hogwild!", "n"), ("方向 B", "n")],
        (2, 1): [("hidden state 探針", "hi"), ("SoftCoT++", "n")],
        (2, 2): [],
        (2, 3): [],
        (3, 0): [("Soft Thinking", "n"), ("KV Steering", "n"), ("LatentMAS", "n")],
        (3, 1): [("SoftCoT", "n"), ("C2C", "n")],
        (3, 2): [("Coconut", "n"), ("CODI", "n"), ("LLM-JEPA", "n"), ("Step STP", "n")],
        (3, 3): [("JEPA-Reasoner", "n"), ("NextLat", "n")],
    }
    fig, ax = canvas(11, 6.2)
    x0, y0, cw, rh = 150, 30, 232, 130
    for j, c in enumerate(cols):
        label(ax, x0 + j * cw + cw / 2, 585, c, 10.5, bold=True, color=MUTED)
    for i, r in enumerate(rows):
        yy = 560 - (i + 1) * rh
        label(ax, 140, yy + rh / 2, r, 10.5, bold=True, ha="right")
        for j in range(4):
            xx = x0 + j * cw
            items = cells[(i, j)]
            target = (i, j) == (0, 1)
            box(ax, xx + 4, yy + 4, cw - 8, rh - 8, fc=LAT_SOFT if target else (PANEL if items else "#fafbfc"), ec=LAT if target else RULE, lw=2 if target else 1, ls="-" if items else "--")
            ty = yy + rh - 22
            for name, kind in items:
                nl = name.count("\n") + 1
                h = 16 * nl + 6
                fc = {"us": LAT, "hi": GATE_SOFT, "n": "#f3f4f8"}[kind]
                ec = {"us": LAT, "hi": GATE, "n": RULE}[kind]
                tc = {"us": "white", "hi": GATE, "n": INK}[kind]
                box(ax, xx + 14, ty - h + 10, cw - 28, h, name, fc=fc, ec=ec, fs=8.6, color=tc, bold=kind != "n", r=4)
                ty -= h + 5
    label(ax, 550, 12, "位置是依原文判斷〔判讀〕。琥珀色＝高重疊、投入前要讀全文；靛藍＝本計劃。", 9.5, color=MUTED)
    save(fig, "fig_litmap.png")


# ---------------------------------------------------------------- 9. 方法 A
def fig_method():
    fig, ax = canvas(11, 4.4)
    box(ax, 8, 175, 172, 90, "凍結 LLM", sub="題目＋已完成步驟 → zₖ", fc=PANEL, ec=MUTED, ls="--", fs=12, bold=True)
    label(ax, 300, 410, "① 提出 N 個真實第 1 步", 11, color=TXT, bold=True)
    for i, yy in enumerate([320, 230, 140]):
        box(ax, 215, yy, 170, 56, f"候選 {['1', '2', 'N'][i]}（文字）", sub="prefill → 真實 zₖ₊₁", fc=TXT_SOFT, ec=TXT, fs=11, bold=True)
        arrow(ax, 180, 220, 213, yy + 28, color=TXT)
        arrow(ax, 385, yy + 28, 445, 220, color=LAT)
    box(ax, 445, 170, 175, 100, "② 潛動作 CEM", sub="8–32 維 · 64 樣本 × 5 輪", fc=LAT_SOFT, ec=LAT, fs=12, bold=True)
    box(ax, 665, 170, 165, 100, "③ 小 predictor", sub="20–60M · 想像 ẑₖ₊₂…ẑₖ₊H", fc=LAT_SOFT, ec=LAT, fs=12, bold=True)
    box(ax, 875, 170, 200, 100, "④ 能量", sub="−V(ẑₖ₊H)＋不確定度懲罰", fc=GATE_SOFT, ec=GATE, fs=12, bold=True)
    arrow(ax, 620, 220, 663, 220, color=LAT)
    arrow(ax, 830, 220, 873, 220, color=LAT)
    ax.add_patch(FancyArrowPatch((975, 270), (532, 270), connectionstyle="arc3,rad=0.35", arrowstyle="-|>,head_length=0.6,head_width=0.35", color=LAT, lw=1.4, mutation_scale=12))
    label(ax, 755, 365, "更新 CEM 分佈（5 輪）", 10, color=LAT, bold=True)
    ax.plot([300, 300, 950, 950], [140, 105, 105, 168], color=GATE, ls="--", lw=1.3)
    arrow(ax, 950, 120, 950, 168, color=GATE, ls="--")
    label(ax, 625, 88, "H＝1 捷徑：直接用 V(真實 zₖ₊₁)，不需要世界模型 → 這是 K2b 的對照組", 9.5, color=GATE)
    ax.plot([1060, 1060, 85, 85], [170, 40, 40, 173], color=TXT, lw=1.4)
    arrow(ax, 85, 60, 85, 173, color=TXT)
    label(ax, 570, 22, "⑤ 執行分數最好的候選（真文字，寫回 KV）→ 下一步重新規劃（MPC）", 10.5, color=TXT, bold=True)
    save(fig, "fig_method.png")


# ---------------------------------------------------------------- 10. 計算量算術
def compute_table():
    N, L, Np = 8, 40, 50e6
    bases = {"0.5B": 0.49e9, "1.5B": 1.54e9, "7.6B": 7.6e9, "8B": 8.0e9}
    rows = []
    for name, Nb in bases.items():
        u = Np / (Nb * L)
        for H in [1, 2, 4, 8]:
            rows.append(dict(base=name, H=H, text=N * H, a_small=N + N * 320 * u * (H - 1), a_big=N + N * 8000 * u * (H - 1), u=u))
    return rows


def fig_compute(rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    r76 = [r for r in rows if r["base"] == "7.6B"]
    Hs = [r["H"] for r in r76]
    x = np.arange(len(Hs))
    w = 0.26
    series = [("文字前瞻（真的往下寫 H 步）", "text", S2), ("A，CEM 64×5", "a_small", S1), ("A，CEM 800×10（V-JEPA 2-AC 預算）", "a_big", S3)]
    for k, (nm, key, c) in enumerate(series):
        vals = [r[key] for r in r76]
        bars = ax.bar(x + (k - 1) * (w + 0.02), vals, w, color=c, label=nm)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}" if v % 1 else f"{v:.0f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"H＝{h}" for h in Hs])
    ax.set_ylabel("每次決策的成本\n（單位：解碼一個候選步驟）", fontsize=9)
    ax.set_title("base 7.6B、N＝8、L＝40、predictor 50M", fontsize=10.5)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    names = ["0.5B", "1.5B", "7.6B", "8B"]
    ratios = [next(r for r in rows if r["base"] == n and r["H"] == 4) for n in names]
    ratios = [r["text"] / r["a_small"] for r in ratios]
    ax.barh(names, ratios, color=S1, height=0.55)
    for i, v in enumerate(ratios):
        ax.text(v + 0.05, i, f"{v:.2f}×", va="center", fontsize=9)
    ax.axvline(4, color=GATE, ls="--", lw=1.2)
    ax.text(4.02, 3.35, "上限 H＝4", fontsize=8.5, color=GATE)
    ax.set_xlim(0, 4.6)
    ax.invert_yaxis()
    ax.set_xlabel("文字前瞻 ÷ 方向 A（H＝4、CEM 64×5）", fontsize=9)
    ax.set_title("base 越大，想像越划算", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.01, "〔算術〕FLOPs 近似：一個想像步 ≈ N_pred／(N_base·L) 個解碼步。沒有量 wall-clock（NOT_MEASURED）。", ha="center", fontsize=8.8, color=MUTED)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.17, wspace=0.28)
    save(fig, "fig_compute.png")


# ---------------------------------------------------------------- 11. KV 與 24 GB 卡
def kv_numbers():
    GiB = 1024 ** 3
    models = {
        "Llama-3.1-8B": dict(params=8.03e9, layers=32, kvh=8, hd=128),
        "Qwen2.5-7B": dict(params=7.62e9, layers=28, kvh=4, hd=128),
    }
    out = {}
    for m, c in models.items():
        per_tok = 2 * c["layers"] * c["kvh"] * c["hd"] * 2
        out[m] = dict(per_tok=per_tok, weights_gib=c["params"] * 2 / GiB,
                      kv_text=1280 * per_tok / GiB, kv_a=320 * per_tok / GiB, kv_long=65536 * per_tok / GiB)
    return out


def fig_kv(kv):
    fig, ax = plt.subplots(figsize=(10, 3.4))
    cats = []
    w_vals, k_vals = [], []
    for m, d in kv.items():
        for nm, key in [("文字前瞻\nN8·H4·L40", "kv_text"), ("方向 A\n同設定", "kv_a"), ("長 CoT 搜尋\n16 分支×4K", "kv_long")]:
            cats.append(f"{m}\n{nm}")
            w_vals.append(d["weights_gib"])
            k_vals.append(d[key])
    y = np.arange(len(cats))
    ax.barh(y, w_vals, color="#c9cede", height=0.6, label="權重（BF16）")
    ax.barh(y, k_vals, left=w_vals, color=S2, height=0.6, label="搜尋分支多出來的 KV")
    for i, (wv, kv_) in enumerate(zip(w_vals, k_vals)):
        tx = wv + kv_ + 0.2
        if 22.0 < tx < 24.8:
            tx = 24.5
        ax.text(tx, i, f"{wv + kv_:.1f} GiB（KV {kv_*1024:.0f} MiB）" if kv_ < 1 else f"{wv + kv_:.1f} GiB（KV {kv_:.1f} GiB）", va="center", fontsize=8.5)
    ax.axvline(24, color=BAD, ls="--", lw=1.3)
    ax.text(24.2, len(cats) - 0.75, "RTX 3090 的 24 GiB", color=BAD, fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(cats, fontsize=7.8)
    ax.invert_yaxis()
    ax.set_xlim(0, 33)
    ax.set_xlabel("GiB（不含啟動時的 activation 與框架開銷）", fontsize=9)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.005, "〔算術〕每 token KV ＝ 2 × 層數 × KV 頭數 × head_dim × 2 bytes；層數與頭數讀自本機 config.json。", ha="center", fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.95, bottom=0.2)
    save(fig, "fig_kv.png")


# ---------------------------------------------------------------- 12. PoC 停損關
def fig_poc():
    fig, ax = canvas(11, 3.6)
    steps = [("P0", "資料", "取樣、切步、\n抽 hidden", False),
             ("K1", "訊號在不在", "探針 AUC\n≥ 0.70？", True),
             ("K2a", "健全性", "L1 ≤ copy\n的 0.8 倍？", False),
             ("K2b", "想像有沒有用", "Δρ ≥ ＋0.05\nvs V(真實 z)？", True),
             ("K3", "同計算量", "勝步驟 beam\n≥ 1.0 pt？", True),
             ("RQ3", "失敗軌跡", "只用對的 vs\n對＋錯", False)]
    fails = ["停下回報", "轉方向 B／C", "查切步與層", "世界模型多餘\n→ 寫成負面結果", "NO-GO\n→ 寫成分析", ""]
    xw = 172
    for i, ((sid, name, q, gate), f) in enumerate(zip(steps, fails)):
        x = 20 + i * xw
        box(ax, x, 160, xw - 22, 130, "", fc=GATE_SOFT if gate else PANEL, ec=GATE if gate else MUTED, lw=1.8 if gate else 1.1)
        label(ax, x + (xw - 22) / 2, 268, sid, 13, bold=True, color=GATE if gate else MUTED)
        label(ax, x + (xw - 22) / 2, 243, name, 10.5, bold=True)
        label(ax, x + (xw - 22) / 2, 198, q, 9.5, color=INK)
        if i < len(steps) - 1:
            arrow(ax, x + xw - 22, 225, x + xw, 225, color=GOOD, lw=1.6)
            label(ax, x + xw - 11, 240, "過", 8.5, color=GOOD)
        if f:
            arrow(ax, x + (xw - 22) / 2, 160, x + (xw - 22) / 2, 115, color=BAD)
            label(ax, x + (xw - 22) / 2, 85, f, 9, color=BAD)
    label(ax, 550, 20, "琥珀框＝停損關：門檻在開跑前寫死〔判讀〕。全程估計 1–2 GPU 天（NOT_MEASURED）。", 9.5, color=MUTED)
    save(fig, "fig_poc.png")


# ---------------------------------------------------------------- 13. 時程
def fig_timeline():
    fig, ax = plt.subplots(figsize=(10, 2.6))
    weeks = ["10/21–10/23", "第 1 週\n10/24–10/30", "第 2 週\n10/31–11/6", "第 3 週\n11/7–11/13", "第 4 週\n11/14–11/20"]
    lanes = ["GPU", "閱讀／寫作"]
    bars = [(0, 0, 1, "MI300X 只給 Tiara", TXT_SOFT, TXT),
            (0, 1, 1, "3090：P0＋K1", LAT_SOFT, LAT),
            (0, 2, 1, "K2a＋K2b", LAT_SOFT, LAT),
            (0, 3, 1, "K3", LAT_SOFT, LAT),
            (0, 4, 1, "RQ3＋判定", GATE_SOFT, GATE),
            (1, 1, 2, "讀完 4 篇高重疊論文全文；K2 開跑前決定是否撞題", "#f3f4f8", MUTED),
            (1, 3, 2, "寫 RUNLOG、整理結果圖", "#f3f4f8", MUTED)]
    for lane, start, span, txt, fc, ec in bars:
        ax.add_patch(FancyBboxPatch((start + 0.04, 1 - lane + 0.15), span - 0.08, 0.7, boxstyle="round,pad=0,rounding_size=0.08", fc=fc, ec=ec, lw=1.1))
        ax.text(start + span / 2, 1 - lane + 0.5, txt, ha="center", va="center", fontsize=9, color=ec if ec != MUTED else INK)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 2)
    ax.set_xticks(np.arange(5) + 0.5)
    ax.set_xticklabels(weeks, fontsize=8.5)
    ax.set_yticks([1.5, 0.5])
    ax.set_yticklabels(lanes, fontsize=9.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.xaxis.tick_top()
    fig.subplots_adjust(left=0.1, right=0.99, top=0.78, bottom=0.04)
    save(fig, "fig_timeline.png")


if __name__ == "__main__":
    fig_story()
    fig_llm()
    fig_tts()
    fig_mpc()
    cem = fig_cem()
    print("chi2 P(|x|<=1):", {k: f"{v:.3g}" for k, v in cem.items()})
    fig_jepa()
    fig_h1()
    fig_litmap()
    fig_method()
    rows = compute_table()
    fig_compute(rows)
    kv = kv_numbers()
    print("kv:", {m: {k: round(v, 3) for k, v in d.items()} for m, d in kv.items()})
    fig_kv(kv)
    fig_poc()
    fig_timeline()
