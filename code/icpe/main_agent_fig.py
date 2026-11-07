"""main_agent_fig.py — icpe-main：agent 軌跡（偏離 17）一張圖：並行數 × 「完美預知最多能讓首字延遲少多少」，3 個 seed 的中位數與最小–最大。

  python -I code/icpe/main_agent_fig.py --verdict <agent_verC_verdict.csv> --verdict-noopf <agent_verC_noOPF_verdict.csv> \
      --out-dir <dir>

每個 seed 是一個變體，值＝該變體判定元件的最大值（main_summary 的 max_judged）；線＝3 個 seed 的中位數，誤差棒＝最小–最大。
實線＝版本 C（含 O-PF，待偏離 14 修正）；虛線＝判定不算 O-PF。另寫 <dir>/fig6_data.csv（每個點；run_id、ts、host、來源檔）。
"""
from __future__ import annotations

import argparse
import os
import socket
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SURF, INK, INK2, GRID, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0", "#8a8984"
TOOLS = [("OpenHands", "#2a78d6"), ("mini-SWE-agent", "#eb6834")]
plt.rcParams.update({"font.family": ["Noto Sans CJK TC", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.axisbelow": True, "legend.frameon": False})


def load(p):
    v = pd.read_csv(p)
    v = v[v.level == "variant"].copy()
    m = v.trace.str.extract(r"^nb:(?P<tool>miniswe|openhands)_N(?P<N>\d+)_s(?P<seed>\d+)$")
    assert not m.isna().any().any(), "verdict 裡有不是 agent 的變體"
    v["tool"] = m.tool.map({"miniswe": "mini-SWE-agent", "openhands": "OpenHands"})
    v["N"], v["seed"] = m.N.astype(int), m.seed.astype(int)
    g = v.groupby(["tool", "N"]).max_judged.agg(["median", "min", "max", "count"]).reset_index()
    assert (g["count"] == 3).all(), "每個 (工具, N) 要剛好 3 個 seed"
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdict", required=True)
    ap.add_argument("--verdict-noopf", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    meta = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"),
                host=socket.gethostname())
    C, X = load(a.verdict), load(a.verdict_noopf)
    rows = []
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    ax.axhline(15, color=REF, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(40, 16.2, "15%＝值得做預測（事先登記）", color=INK2, fontsize=8, va="bottom", ha="left")
    for tool, col in TOOLS:
        for G, ls, lab, src in ((C, "-", "", a.verdict), (X, (0, (3, 2)), "（不算 O-PF）", a.verdict_noopf)):
            g = G[G.tool == tool].sort_values("N")
            same = ls != "-" and (g["median"].values == C[C.tool == tool].sort_values("N")["median"].values).all()
            y, lo, hi = g["median"] * 100, g["min"] * 100, g["max"] * 100
            if not same:     # 不算 O-PF 和含 O-PF 完全相同時（mini-SWE-agent）不重畫，圖例寫明
                ax.errorbar(g.N, y, yerr=[y - lo, hi - y], color=col, lw=2 if ls == "-" else 1.2, ls=ls, marker="o" if ls == "-" else None,
                            ms=6, mec=SURF, mew=1.5, capsize=3, elinewidth=1, zorder=3 if ls == "-" else 2,
                            label=f"{tool}{lab}" + ("（不算 O-PF 也一樣）" if ls == "-" and tool == "mini-SWE-agent" else ""))
            for r in g.itertuples():
                rows.append(dict(meta, fig="fig6", series=f"{tool}{lab}", x=r.N, y=r.median * 100, y_min=r.min * 100, y_max=r.max * 100, src=src))
        last = C[(C.tool == tool) & (C.N == 72)].iloc[0]
        ax.text(73.5, last["median"] * 100 + (3 if tool == "OpenHands" else -3), f"{tool} {last['median'] * 100:.1f}%",
                color=INK, fontsize=8.5, va="center", ha="left", weight="bold")
    ax.set_xticks([16, 32, 64, 72])
    ax.set_xlim(12, 90)
    ax.set_ylim(0, 65)
    ax.set_xlabel("同時在跑的 agent 對話數（固定並行 N）")
    ax.set_ylabel("完美預知比最好的不預測做法少等（%）↑")
    ax.legend(loc="center left", bbox_to_anchor=(0.36, 0.42), fontsize=8)
    fig.tight_layout()
    os.makedirs(a.out_dir, exist_ok=True)
    fig.savefig(f"{a.out_dir}/fig6_agent.png", dpi=170)
    pd.DataFrame(rows).to_csv(f"{a.out_dir}/fig6_data.csv", index=False)
    print(pd.DataFrame(rows)[["series", "x", "y", "y_min", "y_max"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
