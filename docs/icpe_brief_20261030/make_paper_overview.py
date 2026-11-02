#!/usr/bin/env python3
"""Paper-style overview figure: (a) prediction-first workflow vs (b) ours: bound the value first, then decide.

    bash -c 'source code/env_h200.sh >/dev/null 2>&1; python docs/icpe_brief_20261030/make_paper_overview.py'

Visual language follows the reference the user pointed at (Stream-DiffVSR-style two-panel pipeline figure): modules are
drawn with role-specific glyphs instead of identical boxes (stacked cards = data, grey funnel = learned model,
small pills = settings, striped bar = time line, document icon = registration file), thick block arrows, a grey band for the
repeated inner loop, a blue box for persistent state, red numbered circles for what changed, and a step-by-step legend table.

Nothing here is a new measurement. The only numbers drawn (0.30%, 50.5%, 0.81, 128 GiB, 5%, 15%) are asserted against
docs/icpe_paper_20261030/PAPER.md and EVIDENCE.md. Panel (a) is a stylised reading of related-work abstracts
(PAPER.md section 11, "predictive management"), not a claim about any one paper.
Writes figs/paper_overview_{en,zh}.{png,pdf,svg}.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_brief_figs as B  # noqa: E402  (registers CJK fonts, loads EVIDENCE/PAPER text, chk/chkp helpers)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

RED, BLUE, INK, GREY, LGREY = "#c0272d", "#2a78d6", "#111111", "#6b6a66", "#e4e4e2"
ORANGE, LOAD = "#f4a261", "#4a8fe0"
OUT = HERE / "figs"

TXT = {
    "en": dict(
        title_a="(a) Prediction-first (common practice): build a predictor, then see the gain",
        title_b="(b) Ours: measure first, bound the value, then decide whether to build",
        trace="request trace", trace_b="3 real traces",
        a_train="Train a predictor", a_train_sub="who returns · when · where to place", a_train_mono="predictor.fit(trace)",
        a_band="for each returning request", a_band_mono="serve loop, predictor in the loop",
        pills=("place", "evict", "prefetch"), a_pol="Predictor-driven policy", a_pol_mono="policy(pred)",
        a_res="Restore parked KV", a_res_sub="load from CPU / storage", a_res_mono="restore(kv)",
        a_ttft="TTFT", a_out="report gain", a_out_sub="vs. LRU or a weak base",
        a_note="the gain is known only\nafter the predictor\nhas been built",
        state="Registered before the run", doc1="thresholds <5% · ≥15%", doc2="rival set per action", doc3="deviation log",
        state_mono="registration file", thr_note="thresholds",
        recompute="recompute", load="load",
        base="Base restore", base_sub="INT4 + Cake",
        band_b="for each cell: trace × slow tier × GPU idle/busy × CPU size", band_b_mono="same trace, same base, change ONE thing",
        lad=["base: INT4 + Cake", "best non-predictive rival", "free signal (typing hint)", "oracle (perfect foresight)"],
        bracket="what prediction can still save", formula="ceiling = 1 − T(oracle) / T(rival)",
        card1="FreeInference (agent trace)", card2="Mooncake (chat trace)", rival_s="rival", oracle_s="oracle", gap_s="gap",
        hint_note="typing hint gets ≈0.8 of it",
        scale_note="TTFT of returning requests, rival = 100%;  gap = 1 − T(oracle) / T(rival);  CPU 128 GiB (Mooncake: SSD class, GPU idle)",
        verdict="Verdict (registered)", v1="< 5%   stop", v2="5–15%   try a free signal", v3="≥ 15%   worth building",
        out_b="decision per trace", out_b1="FreeInference:\n≤ 0.30% → skip", out_b2="Mooncake:\n≤ 50.5% → typing\nhint first",
        note_b="the ceiling is known\nbefore any predictor\nis built",
        hdr_a="common practice (from abstracts, not verified in full text)", hdr_b="ours",
        rows=[
            ("Input", "a request trace; the predictor is trained on it", "the same trace, with the best fixed restore (INT4 + Cake) as the base"),
            ("Compare", "against LRU or a base without Cake", "against the best non-predictive rival for the same action (LRU, ARC, S3-FIFO ...)"),
            ("Foresight", "a learned predictor: who returns, when, where to place", "an oracle that knows ONE future quantity at a time (prefetch, eviction order, drop)"),
            ("Free signal", "rare (e.g. tool-reported progress used as a hint)", "user starts typing → prefetch; no model needed (0.81 of perfect prefetch)"),
            ("Decide", "the gain is measured after the predictor exists", "ceiling = 1 − T(oracle)/T(rival); thresholds registered before the run"),
            ("Output", "a predictor and the gain it reports", "a decision per trace: FreeInference ≤0.30% → skip; Mooncake ≤50.5% → typing hint"),
        ],
        fonts=["Liberation Sans", "DejaVu Sans"],
    ),
    "zh": dict(
        title_a="(a) 先建預測器、再看收益（常見做法）",
        title_b="(b) 本文：先量上限，再決定要不要蓋預測器",
        trace="請求紀錄", trace_b="3 份真實紀錄",
        a_train="訓練預測器", a_train_sub="誰會回來·何時·該放哪", a_train_mono="predictor.fit(trace)",
        a_band="每個回來的請求", a_band_mono="服務迴圈，預測器在迴圈裡",
        pills=("放置", "淘汰", "預取"), a_pol="依預測決策", a_pol_mono="policy(pred)",
        a_res="把停放的 KV 送回", a_res_sub="從 CPU／儲存載入", a_res_mono="restore(kv)",
        a_ttft="TTFT", a_out="報告收益", a_out_sub="對 LRU 或較弱的底",
        a_note="預測器蓋好之後，\n才知道收益多少",
        state="開跑前先登記", doc1="門檻 <5%・≥15%", doc2="每種動作的對手", doc3="偏離紀錄",
        state_mono="登記檔", thr_note="門檻",
        recompute="重算", load="載入",
        base="底：送回", base_sub="INT4 + Cake",
        band_b="每一格：紀錄 × 慢層 × GPU 閒忙 × CPU 容量", band_b_mono="同一份紀錄、同一個底，只改一件事",
        lad=["底：INT4 + Cake", "最好的不預測對手", "免費訊號（打字提示）", "oracle（完美預知）"],
        bracket="預測最多還能省的部分", formula="上限 = 1 − T(oracle) / T(對手)",
        card1="FreeInference（agent 紀錄）", card2="Mooncake（對話紀錄）", rival_s="對手", oracle_s="oracle", gap_s="差距",
        hint_note="打字提示拿到其中約 0.8",
        scale_note="回來請求的 TTFT，對手 = 100%；差距 = 1 − T(oracle) / T(對手)；CPU 128 GiB（Mooncake：SSD 級、GPU 閒）",
        verdict="判定（先登記）", v1="< 5%   不做", v2="5–15%   先試免費訊號", v3="≥ 15%   值得蓋",
        out_b="每份紀錄的決定", out_b1="FreeInference：\n≤ 0.30% → 不蓋", out_b2="Mooncake：\n≤ 50.5% → 先用\n打字提示",
        note_b="蓋預測器之前，\n就知道上限",
        hdr_a="常見做法（依摘要整理，未逐篇核對全文）", hdr_b="本文",
        rows=[
            ("輸入", "一份請求紀錄；預測器用它訓練", "同一份紀錄，加上最好的固定送回（INT4 + Cake）當底"),
            ("比較", "對 LRU 或沒有 Cake 的底", "對同動作下最好的不預測對手（LRU、ARC、S3-FIFO …）"),
            ("預知", "學出來的預測器：誰回來、何時、放哪", "oracle：一次只知道未來的一件事（預取、淘汰順序、丟棄）"),
            ("免費訊號", "少見（例：工具回報的進度當提示）", "使用者一開始打字就預取；不需要模型（拿到完美預取的 0.81）"),
            ("判定", "預測器蓋好之後才量收益", "上限 = 1 − T(oracle)/T(對手)；門檻開跑前先登記"),
            ("輸出", "一個預測器和它報告的收益", "每份紀錄一個決定：FreeInference ≤0.30% 不蓋；Mooncake ≤50.5% 先用打字提示"),
        ],
        fonts=["Noto Sans TC", "DejaVu Sans"],
    ),
}


def build(lang):
    t = TXT[lang]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": t["fonts"], "svg.fonttype": "none", "hatch.linewidth": 1.1})
    mono = {"family": "DejaVu Sans Mono"}
    fig = plt.figure(figsize=(14, 9.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 140); ax.set_ylim(0, 98); ax.axis("off")

    # ------------------------------------------------------------------ primitives
    def arrow(x0, y0, x1, y1, lw=3.4, col=INK, ls="-", ms=22):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=lw, color=col, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=ms), zorder=5)

    def txt(x, y, s, size=10.0, col=INK, bold=False, ha="center", va="center", ls=1.25, z=6):
        ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=col, fontweight="bold" if bold else "normal", linespacing=ls, zorder=z)

    def mono_txt(x, y, s, ha="center", col=GREY, size=8.4):
        fd = {} if any(ord(c) > 127 for c in s) else mono   # CJK text falls back to the sans font
        ax.text(x, y, s, ha=ha, va="center", fontsize=size + (0.4 if fd == {} else 0), color=col, fontdict=fd, zorder=6)

    def circle(x, y, n, r=1.45):
        ax.add_patch(Circle((x, y), r, fc="white", ec=RED, lw=2.2, zorder=8))
        ax.text(x, y - 0.05, str(n), ha="center", va="center", fontsize=11 if r > 1.2 else 9.5, fontweight="bold", color=INK, zorder=9)

    def label_block(cx, ytop, title, sub=None, mono_s=None, tsize=10.6):
        y = ytop
        txt(cx, y, title, size=tsize, bold=True); y -= 2.5
        if sub:
            txt(cx, y, sub, size=9.0, col=GREY); y -= 2.3
        if mono_s:
            mono_txt(cx, y, mono_s)

    def pill(cx, cy, w, h, s, fc, size=8.8, tc=INK):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={h / 2}", fc=fc, ec="none", zorder=5))
        txt(cx, cy, s, size=size, col=tc, z=7)

    # ------------------------------------------------------------------ glyphs
    def trace_thumb(x, y, w=11, h=8.2):
        for k in (2, 1, 0):
            ax.add_patch(Rectangle((x + 1.3 * k, y + 1.3 * k), w, h, fc="white", ec="#444444", lw=1.2, zorder=3 + (2 - k) * 0.01))
        for i, (a, b) in enumerate(((0.8, 3.6), (2.0, 5.2), (1.2, 7.4), (4.0, 8.6), (0.5, 2.4))):
            ax.add_patch(Rectangle((x + a, y + 0.9 + i * 1.5), b - a, 0.7, fc="#6fa8e8" if i % 2 == 0 else "#9aa0a8", ec="none", zorder=4))

    def net_glyph(x, y, w, h, col="#6b6b6b"):
        left = [(x, y), (x + 0.46 * w, y + 0.17 * h), (x + 0.46 * w, y + 0.83 * h), (x, y + h)]
        right = [(x + w, y), (x + 0.54 * w, y + 0.17 * h), (x + 0.54 * w, y + 0.83 * h), (x + w, y + h)]
        ax.add_patch(Polygon(left, closed=True, fc=col, ec="none", zorder=3))
        ax.add_patch(Polygon(right, closed=True, fc=col, ec="none", zorder=3))
        ax.add_patch(Rectangle((x + 0.46 * w, y + 0.17 * h), 0.08 * w, 0.66 * h, fc=col, ec="none", zorder=3))
        for fx, fh in ((0.08, 0.62), (0.20, 0.50), (0.32, 0.38)):
            for bx in (x + fx * w, x + w - (fx + 0.07) * w):
                ax.add_patch(Rectangle((bx, y + (h - fh * h) / 2), 0.07 * w, fh * h, fc="white", ec="none", zorder=4))

    def cake_glyph(x, y, w, h):
        wo = 0.55 * w
        ax.add_patch(Rectangle((x, y), wo, h, fc=ORANGE, ec="white", lw=0, hatch="////", zorder=3))
        ax.add_patch(Rectangle((x + wo, y), w - wo, h, fc=LOAD, ec="none", zorder=3))
        ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=INK, lw=1.3, zorder=4))
        ax.plot([x + wo, x + wo], [y - 0.7, y + h + 0.7], color=INK, lw=1.7, zorder=5)
        arrow(x + 0.2, y + h + 1.5, x + wo - 0.3, y + h + 1.5, lw=1.4, col="#c2570c", ms=11)
        arrow(x + w - 0.2, y + h + 1.5, x + wo + 0.3, y + h + 1.5, lw=1.4, col=LOAD, ms=11)
        txt(x + wo / 2, y + h + 3.3, t["recompute"], size=8.2, col="#c2570c")
        txt(x + wo + (w - wo) / 2, y + h + 3.3, t["load"], size=8.2, col=LOAD)

    def load_glyph(x, y, w, h):
        ax.add_patch(Rectangle((x, y), w, h, fc=LOAD, ec=INK, lw=1.3, zorder=3))
        arrow(x + 0.3, y + h + 1.5, x + w - 0.3, y + h + 1.5, lw=1.4, col=LOAD, ms=11)
        txt(x + w / 2, y + h + 3.3, t["load"], size=8.2, col=LOAD)

    def doc(x, y, w, h, col):
        f = 0.3 * w
        ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h - f), (x + w - f, y + h), (x, y + h)], closed=True, fc="white", ec=col, lw=1.5, zorder=4))
        ax.add_patch(Polygon([(x + w - f, y + h), (x + w - f, y + h - f), (x + w, y + h - f)], closed=True, fc="#cfe0f7", ec=col, lw=1.2, zorder=5))
        for k in range(3):
            ax.plot([x + 0.2 * w, x + 0.72 * w], [y + 0.2 * h + k * 0.17 * h] * 2, color=col, lw=0.9, zorder=5)

    # ================================================================= panel (a)
    txt(1, 96.3, t["title_a"], size=15.5, bold=True, ha="left")
    trace_thumb(3, 76.5)
    txt(10.2, 73.4, t["trace"], size=9.6)
    arrow(18.4, 82, 22.6, 82)
    net_glyph(23.5, 76, 16, 12)
    label_block(31.5, 72.8, t["a_train"], t["a_train_sub"], t["a_train_mono"])
    arrow(41.4, 82, 46.4, 82)

    ax.add_patch(Rectangle((47, 60), 69, 34, fc=LGREY, ec="none", zorder=1))
    txt(50, 91.3, t["a_band"], size=11.5, bold=True, ha="left")
    mono_txt(50, 88.6, t["a_band_mono"], ha="left")
    ax.add_patch(FancyBboxPatch((51, 74), 15, 13, boxstyle="round,pad=0,rounding_size=0.5", fc="white", ec=INK, lw=1.6, zorder=3))
    for yy, s, col in zip((83.6, 80.5, 77.4), t["pills"], ("#cfe8e0", "#fbe3b0", "#d5dcf6")):
        pill(58.5, yy, 11.8, 2.9, s, col)
    label_block(58.5, 71.6, t["a_pol"], None, t["a_pol_mono"])
    arrow(67.6, 80.5, 72.6, 80.5)
    load_glyph(74, 78, 21, 5)
    label_block(84.5, 74.4, t["a_res"], t["a_res_sub"], t["a_res_mono"])
    arrow(96.4, 80.5, 100.4, 80.5)
    ax.add_patch(Rectangle((101.5, 78.4), 5.2, 4.2, fc="#ee6a3a", ec="none", zorder=4))
    ax.add_patch(Rectangle((106.7, 78.4), 2.4, 4.2, fc="#8c8b86", ec="none", zorder=4))
    txt(105.3, 75.4, t["a_ttft"], size=9.2)
    arrow(117.0, 80.5, 122.2, 80.5)
    ax.add_patch(FancyBboxPatch((123, 74.5), 16.5, 12, boxstyle="round,pad=0,rounding_size=0.8", fc="white", ec=INK, lw=2.4, zorder=3))
    txt(131.25, 82.6, t["a_out"], size=10.6, bold=True)
    txt(131.25, 78.9, t["a_out_sub"], size=8.6, col=GREY)
    txt(131.25, 69.6, t["a_note"], size=9.0, ls=1.3)

    ax.plot([1, 139], [58.6, 58.6], color="#bdbdb8", lw=1.2)

    # ================================================================= panel (b)
    txt(1, 56.2, t["title_b"], size=15.5, bold=True, col=RED, ha="left")

    # registered state (blue) with document icons
    ax.add_patch(FancyBboxPatch((49, 45.4), 69.5, 9.2, boxstyle="round,pad=0,rounding_size=0.6", fc="#d9e8fb", ec=BLUE, lw=2.0, zorder=2))
    txt(51.5, 52.3, t["state"], size=11.5, bold=True, col=BLUE, ha="left")
    mono_txt(116.5, 52.3, t["state_mono"], ha="right", col=BLUE)
    for x0, s in ((51.5, t["doc1"]), (76.0, t["doc2"]), (98.5, t["doc3"])):
        doc(x0, 46.5, 4.2, 4.6, BLUE)
        txt(x0 + 5.5, 48.8, s, size=8.9, ha="left")

    trace_thumb(3, 30.5)
    circle(3.2, 41.5, 1)
    txt(10.2, 27.2, t["trace"], size=9.6)
    mono_txt(10.2, 25.0, t["trace_b"])
    arrow(18.4, 36, 22.6, 36)

    ax.add_patch(Rectangle((21, 18.5), 97.5, 26, fc=LGREY, ec="none", zorder=1))
    txt(23.5, 42.4, t["band_b"], size=11.0, bold=True, ha="left")
    cake_glyph(24, 29.4, 18, 5.2)
    label_block(33, 26.6, t["base"], t["base_sub"])
    mono_txt(33, 21.6, t["band_b_mono"].replace(", ", ",\n", 1), col=GREY, size=8.2)
    arrow(43.6, 32.0, 47.6, 32.0)

    # two mini bar charts on the same scale: how big is the gap between the best rival and perfect foresight?
    x_bar, full = 60.0, 24.0
    oracle_fi = full * (1 - 0.0030)      # FreeInference: largest oracle ceiling 0.30%  (B3-X1)
    oracle_mc = full * (1 - 0.505)       # Mooncake conversation, SSD class, GPU idle: prefetch oracle 50.5%  (B1)

    def mini(ytitle, y_riv, title, oracle_len, gap_txt, gap_col, first):
        txt(49, ytitle, title, size=9.8, bold=True, ha="left")
        for yy, lab, ln, col in ((y_riv, t["rival_s"], full, "#7f7e79"), (y_riv - 2.8, t["oracle_s"], oracle_len, "#b0225f")):
            ax.add_patch(Rectangle((x_bar, yy - 1.0), ln, 2.0, fc=col, ec="none", zorder=4))
            txt(x_bar - 1.2, yy, lab, size=8.8, ha="right")
        ax.plot([x_bar - 0.3, x_bar - 0.3], [y_riv - 3.9, y_riv + 1.4], color=GREY, lw=1.1, zorder=4)
        if first:
            circle(50.2, y_riv, 2, r=1.05); circle(50.2, y_riv - 2.8, 3, r=1.05)
        return x_bar + oracle_len

    xo1 = mini(39.0, 36.4, t["card1"], oracle_fi, "", RED, True)
    txt(85.8, 35.0, f'{t["gap_s"]} 0.30%', size=9.6, bold=True, ha="left", col="#8a1c20")
    ax.plot([49, 117], [31.2, 31.2], color="#c4c4c0", lw=1.0, zorder=3)
    xo2 = mini(29.7, 27.1, t["card2"], oracle_mc, "", RED, False)
    txt(85.8, 25.7, f'{t["gap_s"]} 50.5%', size=9.6, bold=True, ha="left", col="#1f6f48")
    xr = x_bar + full
    ax.plot([xo2, xo2], [23.7, 22.6], color=GREY, lw=1.1, ls=":", zorder=4)
    ax.plot([xr, xr], [26.0, 22.6], color=GREY, lw=1.1, ls=":", zorder=4)
    ax.annotate("", xy=(xo2, 22.6), xytext=(xr, 22.6), arrowprops=dict(arrowstyle="<->", lw=1.6, color=INK, shrinkA=0, shrinkB=0), zorder=5)
    hint_x = xr - 0.81 * (xr - xo2)      # typing hint reaches 0.81 of the gap (C2-GPU, GPU replay)
    ax.plot([hint_x], [23.6], marker="v", ms=7, color="#e8590c", zorder=6)
    circle(66.0, 21.2, 4, r=1.05)
    txt(67.5, 21.2, t["hint_note"], size=8.8, ha="left", col="#a8420a")
    txt(49, 19.0, t["scale_note"], size=7.6, ha="left", col=GREY)

    # verdict pills (registered thresholds); arrows from each gap to its verdict
    txt(107.3, 41.0, t["verdict"], size=10.2, bold=True)
    circle(97.8, 41.2, 5)
    for yy, s_, col in ((37.4, t["v1"], "#f6cfc8"), (32.0, t["v2"], "#f8e4b4"), (26.6, t["v3"], "#c9ebd8")):
        pill(107.3, yy, 19.0, 3.6, s_, col, size=8.8)
    arrow(93.4, 35.2, 97.0, 37.0, lw=1.8, ms=13, col="#8a1c20")
    arrow(93.4, 25.9, 97.0, 26.5, lw=1.8, ms=13, col="#1f6f48")
    ax.annotate("", xy=(107.3, 42.7), xytext=(107.3, 45.4), arrowprops=dict(arrowstyle="-|>", lw=1.8, color=BLUE, ls=(0, (4, 3)), shrinkA=0, shrinkB=0, mutation_scale=15), zorder=5)
    txt(108.8, 44.0, t["thr_note"], size=8.6, col=BLUE, ha="left")

    # output
    arrow(119.2, 32.5, 122.2, 32.5)
    ax.add_patch(FancyBboxPatch((123, 26), 16.5, 13.5, boxstyle="round,pad=0,rounding_size=0.8", fc="white", ec=RED, lw=2.4, zorder=3))
    txt(131.25, 37.6, t["out_b"], size=9.6, bold=True, col=RED)
    txt(131.25, 34.0, t["out_b1"], size=8.8, col=RED)
    txt(131.25, 29.3, t["out_b2"], size=8.8, col=RED)
    circle(123.4, 39.6, 6)
    txt(131.25, 22.2, t["note_b"], size=9.0, col=RED, ls=1.3)

    # ================================================================= legend table
    ax.add_patch(Rectangle((0.5, 0.6), 139, 17.6, fc="#f3f3f1", ec="none"))
    txt(16.0, 16.9, t["hdr_a"], size=10.0, col=GREY, bold=True, ha="left")
    txt(73.0, 16.9, t["hdr_b"], size=10.0, col=RED, bold=True, ha="left")
    for i, (lab, a, b) in enumerate(t["rows"]):
        y = 14.3 - i * 2.45
        circle(3.0, y, i + 1, r=1.0)
        txt(5.6, y, lab, size=9.6, bold=True, ha="left")
        txt(16.0, y, a, size=8.8, ha="left")
        txt(73.0, y, b, size=8.8, col=RED, ha="left")

    stem = OUT / f"paper_overview_{lang}"
    fig.savefig(str(stem) + ".png", dpi=170)
    fig.savefig(str(stem) + ".pdf")
    fig.savefig(str(stem) + ".svg")
    plt.close(fig)
    print("wrote", stem.name)


if __name__ == "__main__":
    B.chkp("0.30%", "50.5%", "0.81", "128 GiB")
    B.chk("| B3-X1 |", "0.30%")
    B.chk("| B1 |", "50.5")
    B.chk("| C2-GPU |", "0.81")
    for lang in ("en", "zh"):
        build(lang)
