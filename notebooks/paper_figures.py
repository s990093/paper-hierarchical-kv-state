#!/usr/bin/env python3
"""產生 main.tex 使用的資料圖。輸出向量 PDF 至 notebooks/figures/。

與 analysis.py 的分工：analysis.py 服務 notebook（帶標題、螢幕尺寸、PNG），
這裡服務論文（無標題、單欄寬、向量 PDF、字級對齊內文）。兩者共用同一批
CSV，故不會漂移。

規則（CLAUDE.md 禁令 1、3）：
  * 每個數字都自 results/ 讀出，**沒有任何硬編值**
  * 讀不到就 raise，不用預設值
  * 圖上不畫標題——標題是 LaTeX caption 的工作

用法：
    /ssd7/hungwei/paper-hkv/venv/vllm/bin/python notebooks/paper_figures.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT = REPO / "notebooks" / "figures"

# main.tex 的版面：letter、margin 0.75in、twocolumn。
# 文字寬 = 8.5 - 1.5 = 7.0in；單欄 = (7.0 - 0.25)/2 = 3.375in。
COL, COL2 = 3.33, 6.95

# main.tex 的 \definecolor
CLR = {"gpu": "#C0392B", "cpu": "#2980B9", "ssd": "#27AE60",
       "drop": "#7F8C8D", "hl": "#B9770E", "ink": "#222222"}


def _need(p: Path) -> Path:
    if not p.exists():
        raise SystemExit(f"🔴 找不到 {p}\n   圖表不得以推估值繪製（CLAUDE.md 禁令 1）。")
    return p


def _rows(p: Path) -> list[dict]:
    with _need(p).open() as f:
        return list(csv.DictReader(f))


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Noto Serif CJK TC", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.unicode_minus": False,
        "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "legend.frameon": False,
        "legend.handlelength": 1.6, "legend.borderaxespad": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.grid": True, "grid.alpha": 0.30,
        "grid.linestyle": (0, (1, 2)), "grid.linewidth": 0.5,
        "lines.linewidth": 1.3, "lines.markersize": 3.5,
        "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.015,
        # 向量輸出且嵌入 TrueType，投稿系統才不會抱怨 Type 3
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {OUT.relative_to(REPO)}/{name}.pdf")


# ────────────────────────── 成本模型 ──────────────────────────

def cost_model(profile: str = "qwen-awq") -> dict:
    """自掃描目錄的自我描述 JSON 讀出成本常數（見 m4_sweep.dump_cost_model）。"""
    p = _need(RESULTS / "m4_oracle" / profile / "cost_model.json")
    d = json.loads(p.read_text())
    if d.get("model_profile") != profile:
        raise SystemExit(f"🔴 {p} 的 model_profile 是 {d.get('model_profile')}，非 {profile}")
    return d


# ───────────────────────── 圖 1：成本交叉 ─────────────────────────

def fig_crossover(profile: str = "qwen-awq") -> None:
    cm = cost_model(profile)
    c = cm["derived_ms_per_block"]
    pstar = cm["crossover_tokens"]
    xmax = 262144
    xs = list(range(0, xmax + 1, 1024))
    drop = [c["recompute_base"] + c["recompute_slope_per_token"] * x for x in xs]

    fig, ax = plt.subplots(figsize=(COL, 2.15))
    # 決策有意義的區間：P* 落在請求內部
    ax.axvspan(2 * pstar, 3.5 * pstar, color=CLR["hl"], alpha=0.10, lw=0)
    ax.plot(xs, drop, color=CLR["drop"], ls=(0, (4, 2)), label="Drop（丟棄後重算）")
    ax.axhline(c["ssd"], color=CLR["ssd"], label="SSD 取回")
    ax.axhline(c["cpu"], color=CLR["cpu"], label="CPU 取回")
    ax.plot([pstar], [c["ssd"]], "o", color=CLR["ink"], ms=4, zorder=5)
    ax.annotate(f"$P^{{*}}$ = {pstar:,}", (pstar, c["ssd"]),
                textcoords="offset points", xytext=(6, 6), fontsize=7,
                color=CLR["ink"])
    ax.annotate("決策有意義的區間\n2–3.5 $P^{*}$",
                (2.75 * pstar, max(drop) * 0.30), ha="center", fontsize=6.5,
                color=CLR["hl"])
    ax.set_xlabel("block 的絕對位置（token）")
    ax.set_ylabel("成本（ms / block）")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, max(drop) * 1.02)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(50000))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v//1000)}K"))
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.03))
    _save(fig, "fig_crossover")


# ─────────────────── 圖 2：headroom 的甜蜜點 ───────────────────

def fig_sweetspot(profile: str = "qwen-awq") -> None:
    """headroom 對請求長度。

    主線為固定壓力（5.4x）與固定重用率（80.9%）的掃描，僅長度變動——即
    論文表 \\ref{tab:sweet-spot} 的那一組。散點為其餘重用率下的量測，用以
    顯示峰值不是特定重用率的產物。
    """
    pstar = cost_model(profile)["crossover_tokens"]

    iso = []
    for d in sorted((RESULTS / "m4_oracle" / "isopressure").glob("L*")):
        for r in _rows(d / "headroom_surface.csv"):
            if r["policy"] == "oracle" and r["oracle_headroom_pct"]:
                iso.append((int(r["request_tokens"]),
                            float(r["oracle_headroom_pct"]),
                            float(r["reuse_pct"]), float(r["pressure_x"])))
    if not iso:
        raise SystemExit("🔴 isopressure/ 沒有 oracle 列")
    iso.sort()

    other = defaultdict(list)
    for r in _rows(RESULTS / "m4_oracle" / f"{profile}-surface-prefill"
                   / "headroom_surface.csv"):
        if r["policy"] == "oracle" and r["oracle_headroom_pct"]:
            other[round(float(r["reuse_pct"]))].append(
                (int(r["request_tokens"]), float(r["oracle_headroom_pct"])))

    fig, ax = plt.subplots(figsize=(COL, 2.15))
    ax.axvspan(2 * pstar, 3.5 * pstar, color=CLR["hl"], alpha=0.11, lw=0)
    ax.axvline(pstar, color=CLR["ink"], lw=0.6, ls=(0, (1, 2)))

    cmap = plt.get_cmap("Blues")
    for k, (u, pts) in enumerate(sorted(other.items())):
        pts.sort()
        ax.plot([x for x, _ in pts], [y for _, y in pts], "-", lw=0.8,
                color=cmap(0.30 + 0.16 * k), alpha=0.75, zorder=2,
                label="其餘重用率（59–89%）" if k == 0 else None)

    ax.plot([x for x, _, _, _ in iso], [y for _, y, _, _ in iso], "o-",
            color=CLR["ink"], lw=1.6, ms=4.2, zorder=4,
            label="壓力 5.4$\\times$、重用 80.9%")

    top = max(iso, key=lambda t: t[1])
    ax.annotate(f"峰值 {top[1]:.1f}%\n@ {top[0]//1024}K = {top[0]/pstar:.1f}$P^{{*}}$",
                (top[0], top[1]), textcoords="offset points", xytext=(-6, -26),
                ha="right", fontsize=6.5, color=CLR["ink"])
    ax.annotate("$P^{*}$", (pstar, 0.6), xytext=(3, 0), fontsize=6.5,
                textcoords="offset points", color=CLR["ink"])
    ax.annotate("2–3.5 $P^{*}$", (2.7 * pstar, 1.2), ha="center",
                fontsize=6.5, color=CLR["hl"])

    ax.set_xscale("log", base=2)
    ax.set_xlabel("請求長度（token）")
    ax.set_ylabel("Oracle headroom（%）")
    ax.set_xticks([32768, 65536, 131072, 262144, 524288])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v//1000)}K"))
    ax.set_ylim(0, 40)
    ax.legend(loc="upper right", handlelength=1.4)
    _save(fig, "fig_sweetspot")


# ───────────────── 圖 3：ε 是「精度 × 任務」的性質 ─────────────────

def _acc(rows: list[dict], key: str = "config") -> dict[str, float]:
    n: dict[str, int] = defaultdict(int)
    ok: dict[str, int] = defaultdict(int)
    for r in rows:
        n[r[key]] += 1
        ok[r[key]] += r["correct"] == "True"
    return {k: 100.0 * ok[k] / n[k] for k in n}


def fig_quality() -> None:
    order = ["bf16", "fp8", "int8", "int4"]
    label = {"bf16": "BF16", "fp8": "FP8", "int8": "INT8", "int4": "INT4"}
    gsm = _acc(_rows(RESULTS / "m5_quality" / "gsm8k_precision_n1000.csv"))
    ndl = _rows(RESULTS / "m5_quality" / "needle_ctx_sweep.csv")
    ctxs = sorted({int(r["prompt_tokens"]) for r in ndl})

    fig, axes = plt.subplots(1, 2, figsize=(COL2 * 0.72, 2.0),
                             gridspec_kw={"wspace": 0.30})
    a = axes[0]
    a.bar(range(len(order)), [gsm[c] for c in order], width=0.62,
          color=[CLR["ink"] if c == "bf16" else CLR["cpu"] for c in order])
    a.set_xticks(range(len(order)), [label[c] for c in order])
    a.set_ylim(0, 100)
    a.set_ylabel("正確率（%）")
    a.set_xlabel("(a) GSM8K 推理（n = 1,000）")
    for i, c in enumerate(order):
        a.annotate(f"{gsm[c]:.1f}", (i, gsm[c]), ha="center", va="bottom",
                   fontsize=6.5, xytext=(0, 1), textcoords="offset points")

    b = axes[1]
    for c in order:
        pts = [(x, _acc([r for r in ndl if int(r["prompt_tokens"]) == x])[c])
               for x in ctxs]
        b.plot([x for x, _ in pts], [y for _, y in pts], "o-",
               color=CLR["ink"] if c == "bf16" else None, label=label[c])
    b.set_xscale("log", base=2)
    b.set_ylim(-3, 103)
    b.set_xlabel("(b) 大海撈針檢索：上下文長度")
    b.set_ylabel("檢索正確率（%）")
    b.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v//1000)}K"))
    b.legend(loc="center left", ncol=1)
    _save(fig, "fig_quality")


# ─────────────── 圖 4：峰值隨算力移動 ───────────────

def fig_hwpeak() -> None:
    """headroom 的峰值隨加速器算力往更長的上下文移動。

    表格給不出「移動」；折線可以。實測線為實心，推算線為虛線並標明。
    """
    rows = _rows(RESULTS / "m4_oracle" / "hw_sweep.csv")
    hw: dict[str, list] = defaultdict(list)
    meta: dict[str, tuple] = {}
    for r in rows:
        k = r["hardware"]
        hw[k].append((int(r["request_tokens"]), float(r["oracle_headroom_pct"])))
        meta[k] = (int(r["crossover_tokens"]), r["measured"] == "1")
    order = sorted(hw, key=lambda k: meta[k][0])

    fig, ax = plt.subplots(figsize=(COL, 2.2))
    cmap = plt.get_cmap("plasma")
    for i, k in enumerate(order):
        pts = sorted(hw[k])
        xo, measured = meta[k]
        c = CLR["ink"] if measured else cmap(0.18 + 0.24 * i)
        ax.plot([x for x, _ in pts], [y for _, y in pts],
                "o-" if measured else "o--", color=c, lw=1.6 if measured else 1.1,
                ms=4.0 if measured else 3.0, zorder=4 if measured else 2,
                label=f"{k}　$P^{{*}}${xo/1000:.0f}K")
        px, py = max(pts, key=lambda t: t[1])
        ax.plot([px], [py], "*", color=c, ms=8, zorder=5)

    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({x for v in hw.values() for x, _ in v}))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v//1000)}K"))
    ax.set_xlabel("請求長度（token）")
    ax.set_ylabel("Oracle headroom（%）")
    ax.set_ylim(0, 46)
    ax.legend(loc="upper left", fontsize=6.0, handlelength=1.5,
              labelspacing=0.25, borderpad=0.2)
    ax.annotate("$\\star$ 為每條線的峰值", (0.97, 0.04), xycoords="axes fraction",
                ha="right", fontsize=6.2, color=CLR["ink"])
    _save(fig, "fig_hwpeak")


def main() -> int:
    style()
    print("產生論文圖表：")
    fig_crossover()
    fig_sweetspot()
    fig_quality()
    fig_hwpeak()
    print("完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
