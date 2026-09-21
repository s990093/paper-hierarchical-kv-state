#!/usr/bin/env python3
"""產生論文用的向量圖（PDF），全部來自平台 B 的實測 CSV。

與 `make_report_mi300x.py`（給人看的 24 頁報告）不同：這裡是**論文插圖**，
單欄寬、字級配合 10pt 內文、無標題（標題寫在 LaTeX 的 \\caption 裡）。

原則：
* 只畫實測值；每個函式的 docstring 標明來源檔與該圖回答的問題。
* 不畫「趨勢示意」——線都是資料點連起來的。
* 輸出到 notebooks/figures/，與論文現有的三張圖同目錄。

用法：python code/paper_figures_mi300x.py
"""
from __future__ import annotations

import csv
import glob
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402
from matplotlib import font_manager as fm          # noqa: E402

REPO = Path(__file__).resolve().parent.parent
R = REPO / "results"
OUT = REPO / "notebooks/figures"
OUT.mkdir(parents=True, exist_ok=True)

FONT = Path("/mlsteam/data/tiara/fonts/NotoSansCJKtc-Regular.otf")
if FONT.exists():
    fm.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({
    "axes.unicode_minus": False,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
COL = 3.33        # 單欄寬（吋）
COL2 = 7.0        # 雙欄寬

SHORT = {"b-llama8b": "Llama-3.1-8B", "b-ultralong8b-1m": "UltraLong-8B-1M",
         "b-qwen7b-1m": "Qwen2.5-7B-1M", "b-qwen3-30b-a3b": "Qwen3-30B-A3B",
         "b-seedoss36b": "Seed-OSS-36B", "b-qwen14b-1m": "Qwen2.5-14B-1M",
         "b-mistral-nemo12b": "Mistral-Nemo-12B"}
ORDER = ["b-llama8b", "b-ultralong8b-1m", "b-qwen7b-1m", "b-mistral-nemo12b",
         "b-qwen14b-1m", "b-qwen3-30b-a3b", "b-seedoss36b"]
C = plt.get_cmap("tab10").colors


def rows(p) -> list[dict]:
    p = Path(p)
    if not p.exists():
        return []
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def fig_recompute_position() -> str:
    """重算成本隨 block 絕對位置線性成長（論文 §2 觀察三的直接證據）。

    來源：results/m2_harness_mi300x/recompute_position_b-*.csv（取每個位置 3 次的最小值）
    """
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    for i, m in enumerate(ORDER):
        f = R / f"m2_harness_mi300x/recompute_position_{m}.csv"
        if not f.exists():
            f = R / f"m2_harness_mi300x/recompute_position_{m}_v2.csv"
        rs = rows(f)
        if not rs:
            continue
        g = defaultdict(list)
        for r in rs:
            if r["ttft_ms"]:
                g[int(r["cached_prefix_tokens"])].append(float(r["ttft_ms"]))
        xs = sorted(g)
        ax.plot([x / 1000 for x in xs], [min(g[x]) for x in xs], "-o", ms=2.2, lw=1.1,
                label=SHORT[m], color=C[i % 10])
    ax.set_xlabel("已快取前綴長度 $P$（千 token）")
    ax.set_ylabel("重算 2{,}048 token 的 TTFT（ms）")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=.3, which="both", lw=.4)
    ax.legend(fontsize=5.6, ncol=2, loc="upper left", framealpha=.9)
    p = OUT / "fig_b_recompute_position.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p.name


def fig_tier_costs() -> str:
    """六階動作空間的每 token 取回成本，跨 7 個模型。

    來源：results/m2_harness_mi300x/cost_constants_mi300x.csv
    """
    cc = rows(R / "m2_harness_mi300x/cost_constants_mi300x.csv")
    by = defaultdict(dict)
    for r in cc:
        try:
            by[r["model_key"]][r["tier"]] = float(r["us_per_token"])
        except ValueError:
            continue
    tiers = ["gpu_fp8", "gpu_int4", "cpu", "ssd", "drop"]
    names = ["GPU FP8", "GPU INT4", "CPU", "SSD", "重算"]
    ms = [m for m in ORDER if m in by]
    fig, ax = plt.subplots(figsize=(COL2, 2.4))
    w = 0.16
    for i, t in enumerate(tiers):
        ax.bar([x + i * w for x in range(len(ms))], [max(by[m].get(t, 0), 1e-3) for m in ms],
               w, label=names[i], color=C[i])
    ax.set_yscale("log")
    ax.set_xticks([x + 2 * w for x in range(len(ms))])
    ax.set_xticklabels([SHORT[m] for m in ms], rotation=12)
    ax.set_ylabel("取回成本（µs/token，對數）")
    ax.grid(axis="y", alpha=.3, which="both", lw=.4)
    ax.legend(ncol=5, fontsize=6.5, loc="upper left")
    p = OUT / "fig_b_tier_costs.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p.name


def fig_headroom() -> str:
    """Oracle headroom 對記憶體壓力的曲線，以及真實 trace 的位置。

    來源：results/m4_oracle_mi300x/oracle_*.csv
    """
    press, trace = defaultdict(dict), defaultdict(dict)
    for f in glob.glob(str(R / "m4_oracle_mi300x/oracle_*.csv")):
        for r in rows(f):
            if r.get("policy") != "oracle":
                continue
            m, w = r.get("model_profile", ""), r.get("workload", "")
            try:
                h = float(r["oracle_headroom_pct"])
            except (KeyError, ValueError):
                continue
            if w.startswith("pressure:"):
                press[m][float(w.split(":")[1].rstrip("x").split("(")[0])] = h
            elif w.startswith("trace:"):
                trace[m][w.split(":")[1]] = h
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    for i, m in enumerate(ORDER):
        if m not in press:
            continue
        xs = sorted(press[m])
        ax.plot(xs, [press[m][x] for x in xs], "-o", ms=2.5, lw=1.1,
                label=SHORT[m], color=C[i % 10])
    ax.axhspan(15, 60, color="#e8f5ea", zorder=0)
    ax.axhspan(5, 15, color="#fdf6e3", zorder=0)
    ax.text(1.05, 46, "GO", fontsize=6.5, color="#1a7f37")
    ax.text(1.05, 9, "問人", fontsize=6.5, color="#8a6d3b")
    ax.set_xlabel("記憶體壓力（工作集 / GPU 預算）")
    ax.set_ylabel("Oracle headroom（prefill-only，\\%）")
    ax.grid(alpha=.3, lw=.4)
    ax.legend(fontsize=5.6, ncol=2, framealpha=.9)
    p = OUT / "fig_b_headroom.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p.name


def fig_quality() -> str:
    """KV 精度的品質代價：撈針 vs GSM8K，跨模型。

    來源：results/m5_quality_mi300x/needle_*.csv、gsm8k_precision_*.csv
    """
    def acc(pat):
        out = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for f in glob.glob(str(R / f"m5_quality_mi300x/{pat}")):
            for r in rows(f):
                out[r.get("model_key", "")][r.get("config", "")][0] += (r.get("correct") == "True")
                out[r.get("model_key", "")][r.get("config", "")][1] += 1
        return out
    ndl, gsm = acc("needle_b*.csv"), acc("gsm8k_precision_*.csv")
    cfgs = ["bf16", "fp8", "fp8_ptk", "int8", "int4"]
    labels = ["BF16", "FP8", "FP8$_{ptk}$", "INT8", "INT4"]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.2), sharey=True)
    for ax, data, title in ((axes[0], ndl, "大海撈針（檢索）"), (axes[1], gsm, "GSM8K（推理）")):
        ms = [m for m in ORDER if m in data]
        w = 0.8 / max(1, len(ms))
        for i, m in enumerate(ms):
            ys = [100 * data[m][c][0] / data[m][c][1] if data[m][c][1] else float("nan")
                  for c in cfgs]
            ax.bar([x + i * w for x in range(len(cfgs))], ys, w, label=SHORT[m], color=C[i % 10])
        ax.set_xticks([x + 0.4 for x in range(len(cfgs))])
        ax.set_xticklabels(labels)
        ax.set_title(title, fontsize=7.5)
        ax.grid(axis="y", alpha=.3, lw=.4)
    axes[0].set_ylabel("正確率（\\%）")
    axes[0].set_ylim(0, 105)
    axes[0].legend(fontsize=5.4, ncol=2, loc="lower left", framealpha=.9)
    p = OUT / "fig_b_quality.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p.name


def fig_attention() -> str:
    """注意力重要度 vs 存取歷史：預測跨請求重用的能力，以及注意力的位置結構。

    來源：results/m5_attention_mi300x/alignment_*.json、attn_importance_*.csv
    """
    import json
    js = sorted(glob.glob(str(R / "m5_attention_mi300x/alignment_*.json")))
    data = [json.loads(Path(f).read_text()) for f in js]
    data.sort(key=lambda d: ORDER.index(d["model_key"]) if d["model_key"] in ORDER else 9)
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.2))
    ax = axes[0]
    xs = range(len(data))
    ax.bar([x - 0.19 for x in xs], [d["auc_attn_mass"] for d in data], 0.38,
           label="注意力重要度", color=C[3])
    ax.bar([x + 0.19 for x in xs], [d["auc_access_history"] for d in data], 0.38,
           label="存取歷史", color=C[0])
    ax.axhline(0.5, ls="--", lw=.8, c="#a94442")
    ax.text(len(data) - 1.1, 0.515, "隨機", fontsize=6, color="#a94442")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([SHORT.get(d["model_key"], "") for d in data], rotation=28, ha="right",
                       fontsize=5.6)
    ax.set_ylabel("預測重用的 AUC")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=.3, lw=.4)
    ax.legend(fontsize=6, loc="upper center")
    ax2 = axes[1]
    f = [x for x in sorted(glob.glob(str(R / "m5_attention_mi300x/attn_importance_b-*.csv")))
         if "smoke" not in x]
    if f:
        g = defaultdict(list)
        for r in rows(f[0]):
            g[int(r["block_in_doc"])].append(float(r["attn_mass"]))
        ks = sorted(g)
        ax2.plot(ks, [st.median(g[k]) for k in ks], lw=1.0, color=C[1])
        ax2.annotate("attention sink", xy=(ks[0], st.median(g[ks[0]])),
                     xytext=(len(ks) * .18, st.median(g[ks[0]]) * .8), fontsize=6,
                     arrowprops=dict(arrowstyle="->", lw=.7, color="#666"))
        ax2.annotate("recency", xy=(ks[-1], st.median(g[ks[-1]])),
                     xytext=(len(ks) * .55, st.median(g[ks[-1]]) * 1.9), fontsize=6,
                     arrowprops=dict(arrowstyle="->", lw=.7, color="#666"))
    ax2.set_xlabel("block 在文件中的位置")
    ax2.set_ylabel("注意力質量")
    ax2.grid(alpha=.3, lw=.4)
    p = OUT / "fig_b_attention.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p.name


def fig_prefill_share() -> str:
    """端到端 headroom 的天花板由 prefill 佔比決定。

    來源：results/m5_predictor_mi300x/policy_sim.csv
    """
    rs = [r for r in rows(R / "m5_predictor_mi300x/policy_sim.csv") if r["policy"] == "oracle"]
    seen, items = set(), []
    for r in rs:
        k = (r.get("trace", ""), r.get("model_profile", ""))
        if k in seen:
            continue
        seen.add(k)
        p_ms, d_ms = float(r["prefill_ms"]), float(r["decode_ms"])
        share = p_ms / (p_ms + d_ms)
        h = float(r["oracle_headroom_pct"])
        if h > 0:
            items.append((f"{k[0]}\n{SHORT.get(k[1], k[1])}", share, h, h / share))
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    xs = range(len(items))
    ax.bar([x - 0.2 for x in xs], [i[3] for i in items], 0.4, label="prefill-only（推算）",
           color=C[0])
    ax.bar([x + 0.2 for x in xs], [i[2] for i in items], 0.4, label="端到端（含 decode）",
           color=C[3])
    for x, i in zip(xs, items):
        ax.text(x, max(i[3], i[2]) + 1.0, f"prefill {100 * i[1]:.0f}\\%", ha="center", fontsize=5.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([i[0] for i in items], fontsize=5.8)
    ax.set_ylabel("Oracle headroom（\\%）")
    ax.grid(axis="y", alpha=.3, lw=.4)
    ax.legend(fontsize=6)
    p = OUT / "fig_b_prefill_share.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p.name


def main() -> int:
    made = [fig_recompute_position(), fig_tier_costs(), fig_headroom(),
            fig_quality(), fig_attention(), fig_prefill_share()]
    print("產生的論文插圖：")
    for m in made:
        print(f"  notebooks/figures/{m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
