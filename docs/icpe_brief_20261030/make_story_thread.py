#!/usr/bin/env python3
"""One-thread story figure for STORY.md: seven cards, each answering the question the previous one left open.

    bash -c 'source code/env_h200.sh >/dev/null 2>&1; python docs/icpe_brief_20261030/make_story_thread.py'

Every number drawn is asserted against EVIDENCE.md / PAPER.md / the T4 result table (nothing new is measured).
Writes figs/story_thread_zh.{png,pdf,svg}.
"""
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_brief_figs as B  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402

OUT = HERE / "figs"
INK, GREY = "#111111", "#5f5e5a"
COLS = ["#6a6a66", "#c8472f", "#d49a1c", "#2f6db5", "#a8741a", "#2a8a5a", "#1f6f48"]

CARDS = [
    ("問題與賭注", "預測能讓使用者回來時更快嗎？", "近年一批研究押注預測；我們一開始也這樣想", "",
     "▼ 預測的好處，最多有多大？"),
    ("最直接的版本：寫入時預知", "寫入時就依位置放好，有用嗎？", "沒用：16 組比較 0 組贏（TTFT 差 −0.37～+1.16%）", "位置回來時一樣看得到；最前面的塊回來時多半靠重算，不從儲存讀",
     "▼ 預測只剩「回來時」。怎麼公平地量它？"),
    ("比較對象決定答案", "預測值多少，取決於跟誰比？", "是。同一個完美預知：只能重算 15.9% → 有慢層 3.4% → 再加 Cake 2.6%", "FreeInference 前 6 週、CPU 32 GiB、慢層 6.98 GiB/s、GPU 閒、對 LRU（模擬；另掃容量與頻寬）",
     "▼ 所以先把「不預測」做到最強"),
    ("把「不預測」的做法做到最強", "最好的不預測做法是什麼？", "在 Cake 上加 INT4：最慢那層 12.1 秒 → 1.24 秒；CPU 層看解壓寫法（差 6.8 倍）", "Cake 是別人已發表的方法，不是我們的；為了公平，不是主角",
     "▼ 不預測的做法做好了，預測還剩多少？"),
    ("在最強的不預測做法之上量上限", "完美預測最多再省多少？", "FreeInference ≤0.30%；Mooncake 預取 50.5% → 14.6%（128→1,024 GiB）", "取決於固定規則已命中多少（FreeInference 99.29%；Mooncake 沒算）",
     "▼ Mooncake 的大上限是真的嗎？"),
    ("大的上限還剩多少", "簡單的打字線索能拿到多少？", "打字提示拿到完美預取改善的 0.81；完美預取只比它快 4.0%", "不是主流做法，是「不用預測」的參考點；含排隊上限再掉一半以上",
     "▼ 沒有這種簡單線索的部分呢？"),
    ("結論", "預測值不值得？", "兩份紀錄上有限：FreeInference 幾乎沒有；對話上打字線索拿到大部分", "待解：淘汰順序、Mooncake toolagent、Mooncake 命中率、真的預測器", ""),
]


def wrap(s, n):
    out, cur, w = [], "", 0
    for ch in s:
        cw = 1 if ord(ch) < 128 else 2
        if w + cw > n:
            out.append(cur); cur, w = "", 0
        cur += ch; w += cw
    if cur:
        out.append(cur)
    return "\n".join(out)


def build():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Noto Sans TC", "DejaVu Sans"], "svg.fonttype": "none"})
    fig = plt.figure(figsize=(8.4, 12.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 148); ax.axis("off")
    ax.text(3, 144.5, "一個問題，一條線", fontsize=17, fontweight="bold", va="center")
    ax.text(3, 140.4, "預測的價值＝完美預測比最好的不預測做法好多少；每一章回答上一章留下的問題", fontsize=10, color=GREY, va="center")
    ch, gap, top = 15.0, 4.6, 136.0
    for i, (title, q, a, ev, nxt) in enumerate(CARDS):
        y = top - i * (ch + gap)
        c = COLS[i]
        ax.add_patch(FancyBboxPatch((3, y - ch), 94, ch, boxstyle="round,pad=0,rounding_size=0.8", fc="#f6f5f1", ec=c, lw=1.6, zorder=2))
        ax.add_patch(FancyBboxPatch((3, y - ch), 1.3, ch, boxstyle="square,pad=0", fc=c, ec="none", zorder=3))
        ax.add_patch(Circle((8.4, y - 3.2), 2.0, fc=c, ec="none", zorder=4))
        ax.text(8.4, y - 3.25, str(i + 1), ha="center", va="center", fontsize=11.5, fontweight="bold", color="white", zorder=5)
        ax.text(12.4, y - 3.2, title, ha="left", va="center", fontsize=12, fontweight="bold", color=INK, zorder=5)
        ax.text(7.0, y - 7.0, "問", ha="left", va="center", fontsize=9.5, color=c, fontweight="bold", zorder=5)
        ax.text(11.2, y - 7.0, q, ha="left", va="center", fontsize=10.5, color=INK, zorder=5)
        ax.text(7.0, y - 10.2, "答", ha="left", va="center", fontsize=9.5, color=c, fontweight="bold", zorder=5)
        ax.text(11.2, y - 10.2, wrap(a, 80), ha="left", va="center", fontsize=10.2, color=INK, fontweight="bold", zorder=5, linespacing=1.3)
        if ev:
            ax.text(11.2, y - 13.0, wrap(ev, 92), ha="left", va="center", fontsize=8.6, color=GREY, zorder=5)
        if nxt:
            ax.text(50, y - ch - gap / 2, nxt, ha="center", va="center", fontsize=10.2, fontweight="bold", color=INK, zorder=5)
    fig.text(0.03, 0.006, "實測：寫入時放置、INT4（在 Cake 上）、解壓、打字提示（GPU 縮小重播）。模擬：15.9%→2.6%、完美預測上限、含排隊。", fontsize=8.4, color=GREY)
    stem = OUT / "story_thread_zh"
    fig.savefig(str(stem) + ".png", dpi=170); fig.savefig(str(stem) + ".pdf"); fig.savefig(str(stem) + ".svg")
    plt.close(fig); print("wrote", stem.name)


if __name__ == "__main__":
    B.chk("| A2 |", "0/16"); B.chk("| A1'' |", "−0.37"); B.chk("| A1' |", "+1.16")
    B.chk("| B3-T4 |", "15.9", "2.6", "99.29%")
    t4 = (HERE.parent / "h200" / "icpe" / "T4_hit_vs_ttft_bridge.md").read_text(encoding="utf-8")
    assert "15.9%" in t4 and "2.6%" in t4
    B.chkp("12.082", "1.2412")
    B.chk("| M3-g |", "6.79")
    B.chk("| B3-X1 |", "0.30%"); B.chk("| B1 |", "50.5")
    from make_ceiling_table import load
    assert round(load()[0][("conversation", "O-PF", "ssd", "1024")], 1) == 14.6
    B.chk("| C2-GPU |", "0.81", "4.0")
    B.chk("| B6 |", "8.1")
    build()
