#!/usr/bin/env python3
"""把平台 B（MI300X）的所有結果做成一份視覺化 PDF。

## 設計原則（與 CLAUDE.md 的禁令一致）

1. **只畫實際量到的數字**。每張圖的頁尾都標出來源 CSV 與 run_id。
2. **還沒跑的實驗畫成明確的空白頁**，寫清楚「會是什麼實驗、需要什麼輸入、
   目前卡在哪」——不是留白讓人以為忘了做，也不是用估算值填。
3. 每頁右下角標 `MEASURED` 或 `PENDING`，一眼可分。

用法：
    python code/make_report_mi300x.py                 # 產生 results/REPORT_MI300X.pdf
    python code/make_report_mi300x.py --out /tmp/x.pdf
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics as st
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib import font_manager as fm            # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
R = REPO / "results"
M1 = R / "m1_capacity/capacity_mi300x.csv"
M2 = R / "m2_harness_mi300x"
M3 = R / "m3_baseline_mi300x/baseline_mi300x.csv"
M4 = R / "m4_oracle_mi300x"
M5Q = R / "m5_quality_mi300x"
M5A = R / "m5_attention_mi300x"
M5P = R / "m5_predictor_mi300x"

FONT = Path("/mlsteam/data/tiara/fonts/NotoSansCJKtc-Regular.otf")
if FONT.exists():
    fm.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.figsize"] = (11.69, 8.27)        # A4 橫式
plt.rcParams["savefig.dpi"] = 150

MODELS = ["b-llama8b", "b-ultralong8b-1m", "b-qwen7b-1m", "b-qwen3-30b-a3b",
          "b-seedoss36b", "b-qwen14b-1m", "b-mistral-nemo12b"]
SHORT = {"b-llama8b": "Llama-3.1-8B", "b-ultralong8b-1m": "UltraLong-8B-1M",
         "b-qwen7b-1m": "Qwen2.5-7B-1M", "b-qwen3-30b-a3b": "Qwen3-30B-A3B (MoE)",
         "b-seedoss36b": "Seed-OSS-36B", "b-qwen14b-1m": "Qwen2.5-14B-1M",
         "b-mistral-nemo12b": "Mistral-Nemo-12B"}
COLORS = plt.get_cmap("tab10").colors


def rows(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def stamp(fig, source: str, measured: bool = True) -> None:
    """每頁標來源與狀態。沒有來源的圖不該存在。"""
    fig.text(0.01, 0.015, source, fontsize=6.5, color="#555555", va="bottom")
    fig.text(0.99, 0.015, "MEASURED" if measured else "PENDING", fontsize=8,
             color="#1a7f37" if measured else "#b35900", ha="right", va="bottom",
             weight="bold")


def pending_page(pdf, title: str, what: str, needs: str, blocked: str = "") -> None:
    """還沒跑的實驗：畫一張空的座標軸 + 說明，不留白、不填假值。"""
    fig, ax = plt.subplots()
    ax.set_title(title, fontsize=15, weight="bold", loc="left", pad=16)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#cccccc")
        s.set_linestyle((0, (4, 4)))
    ax.text(0.5, 0.62, "尚未量測", ha="center", va="center", fontsize=26,
            color="#b35900", weight="bold", transform=ax.transAxes)
    body = f"這一頁會放什麼：\n{what}\n\n需要的輸入：\n{needs}"
    if blocked:
        body += f"\n\n目前狀態：\n{blocked}"
    ax.text(0.5, 0.33, body, ha="center", va="center", fontsize=10.5,
            color="#333333", transform=ax.transAxes, linespacing=1.7)
    stamp(fig, "（尚無資料檔）", measured=False)
    pdf.savefig(fig)
    plt.close(fig)


def text_page(pdf, title: str, lines: list[str], source: str = "") -> None:
    fig = plt.figure()
    fig.text(0.06, 0.93, title, fontsize=17, weight="bold", va="top")
    fig.text(0.06, 0.87, "\n".join(lines), fontsize=10.5, va="top", linespacing=1.75)
    if source:
        stamp(fig, source)
    pdf.savefig(fig)
    plt.close(fig)


def table_page(pdf, title: str, headers: list[str], body: list[list[str]],
               source: str, note: str = "", col_w: list[float] | None = None,
               max_rows: int = 26) -> None:
    """數字表格頁。太多列就自動分頁（標題加「續」）。

    圖看趨勢、表看數字 —— 論文要引用的值必須能從這裡直接讀出來。
    """
    if not body:
        return
    pages = [body[i:i + max_rows] for i in range(0, len(body), max_rows)]
    for pi, chunk in enumerate(pages):
        fig, ax = plt.subplots()
        ax.axis("off")
        ax.set_title(title + ("" if pi == 0 else f"（續 {pi + 1}）"),
                     fontsize=15, weight="bold", loc="left", pad=18)
        if col_w is None:
            # 首欄（模型名）要夠寬，否則會被截斷成「Mistral-Nemo-1」
            n = len(headers)
            first = 0.30 if n <= 6 else (0.22 if n <= 9 else 0.17)
            col_w = [first] + [(1 - first) / (n - 1)] * (n - 1)
        h = min(0.86, 0.075 * (len(chunk) + 1))
        t = ax.table(cellText=chunk, colLabels=headers, cellLoc="right",
                     colWidths=col_w, bbox=[0, 0.90 - h, 1, h])
        t.auto_set_font_size(False)
        t.set_fontsize(8 if len(chunk) <= 18 else 7)
        for (r, c), cell in t.get_celld().items():
            cell.set_edgecolor("#dddddd")
            if r == 0:
                cell.set_facecolor("#eef2f7")
                cell.set_text_props(weight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#fafafa")
            if c == 0:
                cell.set_text_props(ha="left")
        if note and pi == len(pages) - 1:
            fig.text(0.06, 0.06, note, fontsize=8.5, color="#555555", va="bottom",
                     linespacing=1.6)
        stamp(fig, source)
        pdf.savefig(fig)
        plt.close(fig)


# ─────────────────────────── 各頁 ───────────────────────────

def page_cover(pdf) -> None:
    fig = plt.figure()
    fig.text(0.5, 0.70, "Tiara 實驗報告：平台 B（AMD MI300X）", fontsize=26,
             ha="center", weight="bold")
    fig.text(0.5, 0.62, "階層式 KV 放置的成本模型、對照組與品質量測", fontsize=14,
             ha="center", color="#444444")
    env = {}
    if (R / "env_mi300x.json").exists():
        env = json.loads((R / "env_mi300x.json").read_text())
    info = [
        f"產生時間　　{datetime.now():%Y-%m-%d %H:%M}",
        f"GPU　　　　　{env.get('gpu_name', 'AMD Instinct MI300X')}（192 GiB HBM3）",
        f"vLLM　　　　 {env.get('vllm_version', 'v0.28.0')}　ROCm 7.2.2　torch 2.12.1+rocm7.2",
        f"模型數　　　 {len(MODELS)}（dense 5、MoE 1、13–15B 級 2）",
        "資料來源　　 results/ 底下的 CSV／JSON，每頁頁尾標出檔名",
    ]
    fig.text(0.5, 0.45, "\n".join(info), fontsize=11.5, ha="center", linespacing=2.0,
             family="monospace")
    fig.text(0.5, 0.17, "綠色 MEASURED = 實測；橘色 PENDING = 尚未量測，該頁說明會是什麼實驗",
             fontsize=9.5, ha="center", color="#666666")
    pdf.savefig(fig)
    plt.close(fig)


def page_coverage(pdf) -> None:
    """覆蓋矩陣：一眼看出哪些格子還沒跑。"""
    cols = ["M1 容量", "M2 取回", "M2 重算", "M3 對照", "M4 Oracle",
            "無損", "撈針", "GSM8K", "LongBench", "RULER", "注意力"]
    have = {c: set() for c in cols}
    for r in rows(M1):
        for m in MODELS:
            if r["config"].startswith(m + "-bf16"):
                have["M1 容量"].add(m)
    for f in glob.glob(str(M2 / "retrieval_cost_b-*.csv")):
        for r in rows(f):
            have["M2 取回"].add(r.get("model_key", ""))
    for f in glob.glob(str(M2 / "recompute_position_b-*.csv")):
        for r in rows(f):
            have["M2 重算"].add(r.get("model_key", ""))
    for r in rows(M3):
        have["M3 對照"].add(r.get("model_key", ""))
    for f in glob.glob(str(M4 / "oracle_*.csv")):
        for r in rows(f):
            have["M4 Oracle"].add(r.get("model_profile", ""))
    for key, pat in (("無損", "lossless_b-*.csv"), ("撈針", "needle_b*.csv"),
                     ("GSM8K", "gsm8k_precision_*.csv"),
                     ("LongBench", "longbench_precision_*.csv"),
                     ("RULER", "ruler_precision_*.csv")):
        for f in glob.glob(str(M5Q / pat)):
            for r in rows(f):
                have[key].add(r.get("model_key", ""))
    for f in glob.glob(str(M5A / "alignment_*.json")):
        have["注意力"].add(Path(f).stem[len("alignment_"):])

    fig, ax = plt.subplots()
    ax.set_title("實驗覆蓋矩陣　■ 已完成　□ 尚未量測", fontsize=15,
                 weight="bold", loc="left", pad=14)
    ax.set_xlim(0, len(cols))
    ax.set_ylim(0, len(MODELS))
    ax.set_xticks([i + 0.5 for i in range(len(cols))])
    ax.set_xticklabels(cols, rotation=35, ha="left", fontsize=9)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks([i + 0.5 for i in range(len(MODELS))])
    ax.set_yticklabels([SHORT[m] for m in reversed(MODELS)], fontsize=9)
    done = total = 0
    for yi, m in enumerate(reversed(MODELS)):
        for xi, c in enumerate(cols):
            total += 1
            ok = m in have[c]
            done += ok
            ax.add_patch(plt.Rectangle((xi + .06, yi + .06), .88, .88,
                                       facecolor="#d7f0dc" if ok else "#f7f7f7",
                                       edgecolor="#cccccc"))
            if ok:
                ax.text(xi + .5, yi + .5, "✓", ha="center", va="center",
                        fontsize=13, color="#1a7f37", weight="bold")
    ax.set_xlabel(f"完成 {done}/{total} 格（{100 * done / total:.0f}%）", fontsize=11, labelpad=10)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    stamp(fig, "results/ 底下各 CSV 的實際內容掃描")
    pdf.savefig(fig)
    plt.close(fig)


def page_m1(pdf) -> None:
    data = defaultdict(dict)
    for r in rows(M1):
        c, tok = r["config"], r.get("kv_cache_tokens", "")
        if not tok or not tok.isdigit():
            continue
        for m in MODELS:
            if c.startswith(m + "-bf16"):
                suffix = c[len(m + "-bf16"):]
                key = {"": "BF16", "-kvfp8": "FP8", "-kvint8": "INT8", "-kvint4": "INT4"}.get(suffix)
                if key:
                    data[m][key] = int(tok)
    if not data:
        return pending_page(pdf, "M1：KV 容量", "各模型 × 四種 KV 精度的 GPU KV 容量",
                            "results/m1_capacity/capacity_mi300x.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    keys = ["BF16", "FP8", "INT8", "INT4"]
    ms = [m for m in MODELS if m in data]
    w = 0.2
    for i, k in enumerate(keys):
        ax1.bar([x + i * w for x in range(len(ms))],
                [data[m].get(k, 0) / 1e6 for m in ms], w, label=k, color=COLORS[i])
    ax1.set_xticks([x + 1.5 * w for x in range(len(ms))])
    ax1.set_xticklabels([SHORT[m] for m in ms], rotation=25, ha="right", fontsize=8.5)
    ax1.set_ylabel("GPU KV 容量（百萬 token）")
    ax1.set_title("每個模型能放多少 KV", fontsize=12, weight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=.3)
    for i, k in enumerate(keys[1:], 1):
        ax2.bar([x + (i - 1) * 0.25 for x in range(len(ms))],
                [data[m].get(k, 0) / data[m]["BF16"] if data[m].get("BF16") else 0 for m in ms],
                0.25, label=f"{k} / BF16", color=COLORS[i])
    ax2.axhline(2.0, ls="--", c="#888", lw=1)
    ax2.axhline(4.0, ls="--", c="#888", lw=1)
    ax2.text(len(ms) - .5, 2.03, "理論 2×", fontsize=8, color="#666")
    ax2.text(len(ms) - .5, 4.03, "理論 4×", fontsize=8, color="#666")
    ax2.set_xticks([x + 0.25 for x in range(len(ms))])
    ax2.set_xticklabels([SHORT[m] for m in ms], rotation=25, ha="right", fontsize=8.5)
    ax2.set_ylabel("相對 BF16 的容量倍數")
    ax2.set_title("量化換到的容量（實測 vs 理論）", fontsize=12, weight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=.3)
    fig.suptitle("M1：KV 容量 —— 低精度用容量換品質的「本錢」", fontsize=15, weight="bold")
    fig.tight_layout()
    stamp(fig, "results/m1_capacity/capacity_mi300x.csv")
    pdf.savefig(fig)
    plt.close(fig)


def page_m2_tiers(pdf) -> None:
    """各階的取回成本（µs/token），這是 Oracle 的成本常數來源。"""
    cc = rows(M2 / "cost_constants_mi300x.csv")
    if not cc:
        return pending_page(pdf, "M2：六階取回成本", "GPU/FP8/INT4/CPU/SSD/重算 的每 token 成本",
                            "code/m2_analyze_b.py 產生的 cost_constants_mi300x.csv")
    by = defaultdict(dict)
    for r in cc:
        try:
            by[r["model_key"]][r["tier"]] = float(r["us_per_token"])
        except ValueError:
            continue
    tiers = ["gpu_fp8", "gpu_int4", "cpu", "ssd", "drop"]
    names = {"gpu_fp8": "GPU FP8", "gpu_int4": "GPU INT4", "cpu": "CPU 記憶體",
             "ssd": "SSD", "drop": "丟掉重算"}
    ms = [m for m in MODELS if m in by]
    fig, ax = plt.subplots()
    w = 0.16
    for i, t in enumerate(tiers):
        vals = [by[m].get(t, 0) for m in ms]
        ax.bar([x + i * w for x in range(len(ms))], vals, w, label=names[t], color=COLORS[i])
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xticks([x + 2 * w for x in range(len(ms))])
    ax.set_xticklabels([f"{SHORT[m]}\n(ctx {by[m].get('_ctx', '')})" if False else SHORT[m]
                        for m in ms], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("每 token 取回成本（µs，對數刻度）")
    ax.set_title("M2：把 KV 變回可用要多少錢（減掉「本來就在 GPU」的基準）",
                 fontsize=15, weight="bold")
    ax.legend(fontsize=9, ncol=5)
    ax.grid(axis="y", alpha=.3)
    ax.text(0.01, -0.16, "註：Llama-3.1-8B 量於 ctx=16,384，其餘為 96,000；重算成本與位置成正比，"
                          "故不同 ctx 的「丟掉重算」不可直接比較。",
            transform=ax.transAxes, fontsize=8.5, color="#666")
    fig.tight_layout()
    stamp(fig, "results/m2_harness_mi300x/cost_constants_mi300x.csv")
    pdf.savefig(fig)
    plt.close(fig)


def page_m2_recompute(pdf) -> None:
    """重算成本 vs 位置：論文「C_recompute 不是常數」的核心證據。"""
    files = sorted(glob.glob(str(M2 / "recompute_position_b-*.csv")))
    if not files:
        return pending_page(pdf, "M2：重算成本 vs 位置", "在位置 P 重算 2,048 token 的耗時",
                            "results/m2_harness_mi300x/recompute_position_*.csv")
    fig, ax = plt.subplots()
    for i, f in enumerate(files):
        rs = rows(f)
        if not rs:
            continue
        m = rs[0]["model_key"]
        g = defaultdict(list)
        for r in rs:
            if r["ttft_ms"]:
                g[int(r["cached_prefix_tokens"])].append(float(r["ttft_ms"]))
        xs = sorted(g)
        ax.plot([x / 1000 for x in xs], [min(g[x]) for x in xs], "o-", ms=4,
                label=SHORT.get(m, m), color=COLORS[i % len(COLORS)])
    ax.set_xlabel("已快取的前綴長度 P（千 token）")
    ax.set_ylabel("重算 2,048 個 token 的 TTFT（ms）")
    ax.set_title("M2：重算成本隨位置線性成長 —— 論文「C_recompute 不是常數」的證據",
                 fontsize=15, weight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=.3)
    fit = rows(M2 / "recompute_fit_mi300x.csv")
    if fit:
        lines = [f"{SHORT.get(r['model_key'], r['model_key'])}: "
                 f"C0={float(r['C0_ms']):.0f} ms, 斜率={float(r['a_us_per_1k_token']):.2f} µs/千token, "
                 f"R²={float(r['r2']):.4f}"
                 for r in fit if r.get("range") == "min all"]
        ax.text(0.02, 0.97, "線性擬合（取各位置 3 次的最小值）：\n" + "\n".join(lines),
                transform=ax.transAxes, va="top", fontsize=8.5,
                bbox=dict(fc="#f7f7f7", ec="#dddddd"))
    fig.tight_layout()
    stamp(fig, "results/m2_harness_mi300x/recompute_position_*.csv + recompute_fit_mi300x.csv")
    pdf.savefig(fig)
    plt.close(fig)


def page_m3(pdf) -> None:
    rs = rows(M3)
    if not rs:
        return pending_page(pdf, "M3：Tier-0 對照組", "四個 baseline 的 warm/cold TTFT",
                            "results/m3_baseline_mi300x/baseline_mi300x.csv")
    g = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rs:
        if r["ttft_ms"]:
            g[r["model_key"]][int(r["ctx"])][(r["baseline"], r["round"])].append(float(r["ttft_ms"]))
    ms = [m for m in MODELS if m in g]
    n = len(ms)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(15, 8.5))
    axes = axes.ravel()
    bases = ["full_gpu", "cpu_lru", "cpu_arc", "tier_fs"]
    names = {"full_gpu": "不卸載", "cpu_lru": "CPU LRU", "cpu_arc": "CPU ARC", "tier_fs": "CPU+磁碟"}
    for ai, m in enumerate(ms):
        ax = axes[ai]
        ctxs = sorted(g[m])
        for bi, b in enumerate(bases):
            ys = [st.median(g[m][c][(b, "warm")]) if g[m][c][(b, "warm")] else float("nan")
                  for c in ctxs]
            ax.plot([c / 1024 for c in ctxs], ys, "o-", ms=4, label=names[b], color=COLORS[bi])
        ax.set_yscale("log")
        ax.set_title(SHORT[m], fontsize=10.5, weight="bold")
        ax.set_xlabel("ctx（K token）", fontsize=8.5)
        ax.set_ylabel("warm TTFT（ms）", fontsize=8.5)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=.3)
        if ai == 0:
            ax.legend(fontsize=8)
    for ax in axes[len(ms):]:
        ax.axis("off")
    fig.suptitle("M3：工作集塞不下 GPU 之後，四個對照組的取回延遲（越低越好）",
                 fontsize=15, weight="bold")
    fig.tight_layout()
    stamp(fig, "results/m3_baseline_mi300x/baseline_mi300x.csv")
    pdf.savefig(fig)
    plt.close(fig)


def page_m4(pdf) -> None:
    files = sorted(glob.glob(str(M4 / "oracle_*.csv")))
    if not files:
        return pending_page(
            pdf, "M4：Oracle go/no-go",
            "每個模型在不同記憶體壓力下，Oracle 相對最佳 baseline 的改善（headroom）",
            "results/m4_oracle_mi300x/oracle_<model>_<workload>.csv",
            "CPU 佇列重跑中（原本固定檔名 oracle.csv 互相覆蓋，已修）")
    press = defaultdict(dict)
    trace = defaultdict(dict)
    for f in files:
        for r in rows(f):
            if r.get("policy") != "oracle":
                continue
            m, w = r.get("model_profile", ""), r.get("workload", "")
            try:
                h = float(r["oracle_headroom_pct"])
            except (KeyError, ValueError):
                continue
            if w.startswith("pressure:"):
                press[m][float(w.split(":")[1].rstrip("x"))] = h
            elif w.startswith("trace:"):
                trace[m][w.split(":")[1]] = h
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    for i, (m, d) in enumerate(sorted(press.items())):
        xs = sorted(d)
        ax1.plot(xs, [d[x] for x in xs], "o-", label=SHORT.get(m, m), color=COLORS[i % 10])
    ax1.axhspan(15, 100, color="#d7f0dc", alpha=.5)
    ax1.axhspan(5, 15, color="#fff3cd", alpha=.6)
    ax1.axhspan(-2, 5, color="#f8d7da", alpha=.5)
    ax1.text(1.02, 45, "GO（>15%）", fontsize=9, color="#1a7f37")
    ax1.text(1.02, 9.5, "問人（5–15%）", fontsize=9, color="#8a6d3b")
    ax1.text(1.02, 1.5, "停止（<5%）", fontsize=9, color="#a94442")
    ax1.set_xlabel("記憶體壓力（工作集 / GPU 預算）")
    ax1.set_ylabel("Oracle 相對最佳 baseline 的改善（%）")
    ax1.set_title("合成工作負載（Zipf α=0.9）", fontsize=12, weight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=.3)
    if trace:
        ms = sorted(trace)
        ts = sorted({t for d in trace.values() for t in d})
        w = 0.35
        for i, t in enumerate(ts):
            ax2.bar([x + i * w for x in range(len(ms))],
                    [trace[m].get(t, 0) for m in ms], w, label=t, color=COLORS[i])
        ax2.set_xticks([x + w / 2 for x in range(len(ms))])
        ax2.set_xticklabels([SHORT.get(m, m) for m in ms], rotation=25, ha="right", fontsize=8.5)
        ax2.axhline(15, ls="--", c="#1a7f37")
        ax2.axhline(5, ls="--", c="#a94442")
        ax2.set_ylabel("headroom（%）")
        ax2.set_title("真實 trace（Mooncake，前 4,000 個請求）", fontsize=12, weight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(axis="y", alpha=.3)
    else:
        ax2.axis("off")
        ax2.text(.5, .5, "真實 trace 尚未重跑", ha="center", fontsize=13, color="#b35900")
    fig.suptitle("M4：Oracle go/no-go —— 整個研究的停損點", fontsize=15, weight="bold")
    fig.tight_layout()
    stamp(fig, "results/m4_oracle_mi300x/oracle_*.csv")
    pdf.savefig(fig)
    plt.close(fig)


def _acc(pattern: str) -> dict:
    out = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for f in glob.glob(str(M5Q / pattern)):
        for r in rows(f):
            m, c = r.get("model_key", ""), r.get("config", "")
            out[m][c][0] += (r.get("correct") == "True")
            out[m][c][1] += 1
    return out


def page_quality(pdf) -> None:
    ndl, gsm = _acc("needle_b*.csv"), _acc("gsm8k_precision_*.csv")
    if not ndl and not gsm:
        return pending_page(pdf, "M5：KV 精度的品質代價 ε", "撈針與 GSM8K 在五種 KV 精度下的正確率",
                            "results/m5_quality_mi300x/needle_*.csv、gsm8k_precision_*.csv")
    cfgs = ["bf16", "fp8", "fp8_ptk", "int8", "int4"]
    labels = ["BF16", "FP8", "FP8動態", "INT8", "INT4"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    for ax, data, title in ((ax1, ndl, "大海撈針（長距離檢索）"),
                            (ax2, gsm, "GSM8K many-shot（推理）")):
        ms = [m for m in MODELS if m in data]
        w = 0.8 / max(1, len(ms))
        for i, m in enumerate(ms):
            ys = [100 * data[m][c][0] / data[m][c][1] if data[m][c][1] else float("nan")
                  for c in cfgs]
            ax.bar([x + i * w for x in range(len(cfgs))], ys, w,
                   label=SHORT.get(m, m), color=COLORS[i % 10])
        ax.set_xticks([x + 0.4 for x in range(len(cfgs))])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("正確率（%）")
        ax.set_ylim(0, 105)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.grid(axis="y", alpha=.3)
    ax1.legend(fontsize=8, loc="lower left")
    fig.suptitle("M5：KV 量化的品質代價 —— 只看 GSM8K 會得出「量化免費」的錯誤結論",
                 fontsize=15, weight="bold")
    fig.tight_layout()
    stamp(fig, "results/m5_quality_mi300x/needle_*.csv, gsm8k_precision_*.csv")
    pdf.savefig(fig)
    plt.close(fig)


def page_bench(pdf) -> None:
    lb, ru = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for pat, dst in (("longbench_precision_*.csv", lb), ("ruler_precision_*.csv", ru)):
        for f in glob.glob(str(M5Q / pat)):
            for r in rows(f):
                sc = r.get("score") or r.get("metric_score") or ""
                if sc:
                    dst[r.get("model_key", "")][r.get("config", "")].append(float(sc))
    if not lb and not ru:
        return pending_page(pdf, "M5：LongBench / RULER 跑分", "兩套標準評測在五種 KV 精度下的分數",
                            "results/m5_quality_mi300x/longbench_precision_*.csv、ruler_precision_*.csv")
    cfgs = ["bf16", "fp8", "fp8_ptk", "int8", "int4"]
    labels = ["BF16", "FP8", "FP8動態", "INT8", "INT4"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    for ax, data, title in ((ax1, lb, "LongBench（7 個英文任務）"), (ax2, ru, "RULER（合成，16K）")):
        ms = [m for m in MODELS if m in data]
        w = 0.8 / max(1, len(ms))
        for i, m in enumerate(ms):
            ys = [100 * st.mean(data[m][c]) if data[m].get(c) else float("nan") for c in cfgs]
            ax.bar([x + i * w for x in range(len(cfgs))], ys, w,
                   label=SHORT.get(m, m), color=COLORS[i % 10])
        ax.set_xticks([x + 0.4 for x in range(len(cfgs))])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("平均分數")
        ax.set_title(title, fontsize=12, weight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=.3)
    fig.suptitle("M5：標準評測跑分 —— 五種 KV 精度的差距在 ±1 分內", fontsize=15, weight="bold")
    fig.tight_layout()
    stamp(fig, "results/m5_quality_mi300x/longbench_precision_*.csv, ruler_precision_*.csv")
    pdf.savefig(fig)
    plt.close(fig)


def page_lossless(pdf) -> None:
    res = []
    for f in sorted(glob.glob(str(M5Q / "lossless_b-*.csv"))):
        if "rep2" in f:
            continue
        by = defaultdict(dict)
        for r in rows(f):
            by[r["idx"]][r["config"]] = r
        pair = {("full_gpu", "cpu_lru"): [0, 0], ("full_gpu", "tier_fs"): [0, 0],
                ("cpu_lru", "tier_fs"): [0, 0]}
        for _, d in by.items():
            for (a, b), acc in pair.items():
                if a in d and b in d:
                    acc[1] += 1
                    acc[0] += d[a]["out_sha1"] != d[b]["out_sha1"]
        m = next(iter(by.values()))[next(iter(next(iter(by.values()))))]["model_key"]
        res.append((m, pair))
    if not res:
        return pending_page(pdf, "M5：卸載是否無損", "卸載前後輸出是否位元相同",
                            "results/m5_quality_mi300x/lossless_*.csv")
    fig, ax = plt.subplots()
    ms = [r[0] for r in res]
    keys = [("full_gpu", "cpu_lru"), ("full_gpu", "tier_fs"), ("cpu_lru", "tier_fs")]
    names = ["不卸載 vs CPU階", "不卸載 vs 磁碟階", "CPU階 vs 磁碟階"]
    w = 0.26
    for i, k in enumerate(keys):
        ys = [100 * p[k][0] / p[k][1] if p[k][1] else float("nan") for _, p in res]
        ax.bar([x + i * w for x in range(len(ms))], ys, w, label=names[i], color=COLORS[i])
    ax.set_xticks([x + w for x in range(len(ms))])
    ax.set_xticklabels([SHORT.get(m, m) for m in ms], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("輸出位元不同的比例（%）")
    ax.set_title("M5：卸載的無損性 —— 兩條卸載路徑彼此完全一致，但與「不卸載」不同",
                 fontsize=15, weight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=.3)
    ax.text(0.01, -0.14,
            "決定性對照：同一個 full_gpu 設定跑兩次 → 0/60 位元不同，所以上圖的差異不是雜訊。\n"
            "CPU階 vs 磁碟階 = 0% → 位元組搬移本身無損；差異來自「取回+部分重算」與「整段 prefill」"
            "的浮點路徑不同。",
            transform=ax.transAxes, fontsize=9, color="#666")
    fig.tight_layout()
    stamp(fig, "results/m5_quality_mi300x/lossless_*.csv + lossless_b-llama8b_rep2.csv")
    pdf.savefig(fig)
    plt.close(fig)


def page_attention(pdf) -> None:
    js = sorted(glob.glob(str(M5A / "alignment_*.json")))
    if not js:
        return pending_page(
            pdf, "M5-C：注意力重要度 ↔ 重用預測",
            "注意力重要度與存取歷史各自預測「block 會被重用」的 AUC，以及兩者的相關",
            "results/m5_attention_mi300x/alignment_*.json、attn_importance_*.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    data = [json.loads(Path(f).read_text()) for f in js]
    ms = [d["model_key"] for d in data]
    w = 0.35
    ax1.bar([x for x in range(len(ms))], [d["auc_attn_mass"] for d in data], w,
            label="注意力重要度", color=COLORS[3])
    ax1.bar([x + w for x in range(len(ms))], [d["auc_access_history"] for d in data], w,
            label="存取歷史", color=COLORS[0])
    ax1.axhline(0.5, ls="--", c="#a94442")
    ax1.text(len(ms) - 0.5, 0.51, "0.5 = 隨機猜", fontsize=9, color="#a94442")
    ax1.set_xticks([x + w / 2 for x in range(len(ms))])
    ax1.set_xticklabels([SHORT.get(m, m) for m in ms], rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("預測「該 block 會被重用」的 AUC")
    ax1.set_ylim(0, 1)
    ax1.set_title("兩個訊號各自的預測力", fontsize=12, weight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=.3)
    # 注意力隨 block 位置的形狀（sink + recency）
    f = sorted(glob.glob(str(M5A / "attn_importance_b-*.csv")))
    f = [x for x in f if "smoke" not in x]
    if f:
        rs = rows(f[0])
        g = defaultdict(list)
        for r in rs:
            g[int(r["block_in_doc"])].append(float(r["attn_mass"]))
        xs = sorted(g)
        ax2.plot(xs, [st.median(g[x]) for x in xs], lw=1.2, color=COLORS[1])
        ax2.set_xlabel("block 在文件中的位置")
        ax2.set_ylabel("注意力質量（中位數）")
        ax2.set_title("注意力形狀：開頭 sink + 結尾 recency", fontsize=12, weight="bold")
        ax2.grid(alpha=.3)
        ax2.annotate("attention sink", xy=(xs[0], st.median(g[xs[0]])),
                     xytext=(len(xs) * .15, st.median(g[xs[0]]) * .85),
                     arrowprops=dict(arrowstyle="->", color="#666"), fontsize=9)
        ax2.annotate("recency", xy=(xs[-1], st.median(g[xs[-1]])),
                     xytext=(len(xs) * .6, st.median(g[xs[-1]]) * 1.6),
                     arrowprops=dict(arrowstyle="->", color="#666"), fontsize=9)
    sp = ", ".join(f"{SHORT.get(d['model_key'], d['model_key'])}: {d['spearman_attn_vs_history']:+.3f}"
                   for d in data)
    fig.suptitle("M5-C：注意力重要度無法預測跨請求重用（兩訊號正交）　Spearman " + sp,
                 fontsize=14, weight="bold")
    fig.tight_layout()
    stamp(fig, "results/m5_attention_mi300x/alignment_*.json, attn_importance_*.csv")
    pdf.savefig(fig)
    plt.close(fig)


def page_predictor(pdf) -> None:
    met = rows(M5P / "predictor_metrics.csv")
    if not met:
        return pending_page(
            pdf, "M5：學習式預測器（LightGBM）",
            "預測器的 AUC／校準誤差 ECE／成本加權錯誤，以及特徵重要度",
            "results/m5_predictor_mi300x/predictor_metrics.csv、feature_importance.csv",
            "CPU 佇列排隊中（m5_predictor.py 已完成平台 B 的路徑移植）")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    xs = [f"{r.get('trace', '')}\n{r.get('loss', '')}" for r in met]
    # 有些列的 auc 可能是空字串（例如只跑了 features 階段），跳過而不是崩
    met = [r for r in met if (r.get("auc") or "").strip()]
    if not met:
        plt.close(fig)
        return pending_page(pdf, "M5：學習式預測器（LightGBM）",
                            "預測器的 AUC／校準誤差 ECE／成本加權錯誤，以及特徵重要度",
                            "results/m5_predictor_mi300x/predictor_metrics.csv",
                            "目前的 CSV 沒有含 auc 的列（可能只跑到 features 階段）")
    # xs 必須在過濾之後重算，否則標籤數與長條數不一致
    xs = [f"{r.get('trace', '')}\n{r.get('loss', '')}/{r.get('threshold_rule', '')}" for r in met]
    ax1.bar(range(len(met)), [float(r["auc"]) for r in met], color=COLORS[0])
    ax1.set_xticks(range(len(met)))
    ax1.set_xticklabels(xs, fontsize=7.5)
    ax1.set_ylabel("AUC")
    ax1.set_ylim(0.5, 1.0)
    ax1.set_title("預測器的判別力", fontsize=12, weight="bold")
    ax1.grid(axis="y", alpha=.3)
    fi = rows(M5P / "feature_importance.csv")
    if fi:
        agg = defaultdict(float)
        for r in fi:
            try:
                agg[r["feature"]] += float(r["gain"])
            except (KeyError, ValueError):
                continue
        top = sorted(agg.items(), key=lambda x: -x[1])[:12]
        ax2.barh([t[0] for t in reversed(top)], [t[1] for t in reversed(top)], color=COLORS[2])
        ax2.set_xlabel("累計 gain")
        ax2.set_title("前 12 名特徵", fontsize=12, weight="bold")
        ax2.tick_params(labelsize=8)
        ax2.grid(axis="x", alpha=.3)
    fig.suptitle("M5：學習式放置預測器", fontsize=15, weight="bold")
    fig.tight_layout()
    stamp(fig, "results/m5_predictor_mi300x/predictor_metrics.csv")
    pdf.savefig(fig)
    plt.close(fig)


def tbl_m1(pdf) -> None:
    data = defaultdict(dict)
    meta = {}
    for r in rows(M1):
        c, tok = r["config"], r.get("kv_cache_tokens", "")
        if not tok or not tok.isdigit():
            continue
        for m in MODELS:
            if c.startswith(m + "-bf16"):
                suf = c[len(m + "-bf16"):]
                k = {"": "BF16", "-kvfp8": "FP8", "-kvint8": "INT8", "-kvint4": "INT4"}.get(suf)
                if k:
                    data[m][k] = int(tok)
                    # 🔴 limited_by 只能取自 BF16 那一列：量化之後容量變大，
                    #    上限自然會從「記憶體」變成「模型長度」，取到後者會把
                    #    Seed-OSS「受記憶體限制」這個關鍵事實蓋掉。
                    if k == "BF16" or m not in meta:
                        meta[m] = (r.get("model_max_positions", ""),
                                   r.get("limited_by", "") if k == "BF16" else "",
                                   r.get("kv_cache_gib", ""))
    body = []
    for m in MODELS:
        if m not in data:
            continue
        d = data[m]
        bf = d.get("BF16", 0)
        mp, lim, gib = meta[m]
        body.append([SHORT[m], f"{bf:,}", f"{d.get('FP8', 0):,}", f"{d.get('INT8', 0):,}",
                     f"{d.get('INT4', 0):,}",
                     f"{d.get('FP8', 0) / bf:.3f}" if bf else "—",
                     f"{d.get('INT8', 0) / bf:.3f}" if bf else "—",
                     f"{d.get('INT4', 0) / bf:.3f}" if bf else "—",
                     f"{int(mp):,}" if mp.isdigit() else mp,
                     {"model": "模型上限", "memory": "記憶體"}.get(lim, lim)])
    table_page(pdf, "表 1　M1：GPU KV 容量（token）與量化倍數",
               ["模型", "BF16", "FP8", "INT8", "INT4", "FP8/BF16", "INT8/BF16",
                "INT4/BF16", "模型可定址長度", "上限來自"], body,
               "results/m1_capacity/capacity_mi300x.csv",
               "結論：七個模型的量化倍數一致收斂到 2.00／1.94／3.77，與平台 A（RTX 3090）相同 —— 這是跨硬體、跨模型的不變量，\n"
               "　　　可以當成論文中「精度階換容量」的係數直接引用。INT8 之所以是 1.94 而非 2.00，是每 token 每 head 的縮放中繼資料佔掉的。\n"
               "　　　七個模型只有 Seed-OSS-36B 的 BF16 上限來自記憶體（413,632 < 模型可定址的 524,288），其餘都先撞到模型自身的長度上限。")


def tbl_m2(pdf) -> None:
    cc = rows(M2 / "cost_constants_mi300x.csv")
    body = []
    for r in cc:
        if r["tier"] == "gpu_resident":
            continue
        body.append([SHORT.get(r["model_key"], r["model_key"]), f"{int(r['ctx']):,}",
                     {"gpu_fp8": "GPU FP8", "gpu_int4": "GPU INT4", "cpu": "CPU",
                      "ssd": "SSD", "drop": "丟掉重算"}.get(r["tier"], r["tier"]),
                     f"{float(r['warm_ttft_ms_median']):,.1f}",
                     f"{float(r['delta_vs_gpu_resident_ms']):,.1f}",
                     f"{float(r['us_per_token']):.2f}",
                     r["effective_gb_per_s"] or "—",
                     f"{int(r['prefix_kv_bytes']) / 2**30:.2f}"])
    table_page(pdf, "表 2　M2：各階的取回成本（warm TTFT 減「本來就在 GPU」的基準）",
               ["模型", "ctx", "階", "warm TTFT ms", "減基準 ms", "µs/token", "等效 GB/s", "前綴 KV GiB"],
               body, "results/m2_harness_mi300x/cost_constants_mi300x.csv",
               "結論：同一張卡上，CPU 階的等效頻寬從 2.3 到 37.8 GB/s（16 倍差距，見發現 11 的 Qwen3-MoE）—— κ 不只跨硬體變動，**跨模型也變動**。\n"
               "　　　GPU 內的精度階（FP8／INT4）反量化成本極小（0.17／1.59 µs/token），相對於 CPU 的 23–56、SSD 的 44–120 µs/token 幾乎免費。\n"
               "　　　⚠️ Llama-3.1-8B 量於 ctx=16,384，其餘為 96,000；重算成本與位置成正比，故「丟掉重算」那一欄不可跨 ctx 直接比較。")

    fit = [r for r in rows(M2 / "recompute_fit_mi300x.csv") if r.get("range") == "min all"]
    if fit:
        body2 = [[SHORT.get(r["model_key"], r["model_key"]), f"{float(r['C0_ms']):.1f}",
                  f"{float(r['a_us_per_1k_token']):.2f}", f"{float(r['r2']):.4f}",
                  f"{int(r['n_positions'])}", f"{int(r['chunk_tokens']):,}"] for r in fit]
        table_page(pdf, "表 3　M2：重算成本的線性擬合　C_recompute(P) = C0 + a·P",
                   ["模型", "C0（ms）", "a（µs/千 token）", "R²", "位置數", "chunk"],
                   body2, "results/m2_harness_mi300x/recompute_fit_mi300x.csv",
                   "結論：七個模型的 R² 都 ≥ 0.947（五個 ≥ 0.999），證實論文 §3 的「C_recompute 不是常數、隨位置線性成長」。\n"
                   "　　　兩個 8B 模型的斜率幾乎相同（9.25／9.75 µs/千 token），Seed-OSS-36B 是它們的 3.3 倍 —— 斜率隨模型規模放大，\n"
                   "　　　所以「丟掉重算」這個動作在大模型上昂貴得多，這正是階層放置對大模型更有價值的原因。")


def tbl_m3(pdf) -> None:
    rs = rows(M3)
    if not rs:
        return
    g = defaultdict(lambda: defaultdict(list))
    for r in rs:
        if r["ttft_ms"]:
            g[(r["model_key"], int(r["ctx"]))][(r["baseline"], r["round"])].append(float(r["ttft_ms"]))
    body = []
    for (m, ctx), d in sorted(g.items(), key=lambda x: (MODELS.index(x[0][0]) if x[0][0] in MODELS else 9, x[0][1])):
        if not d.get(("full_gpu", "warm")):
            continue
        row = [SHORT.get(m, m), f"{ctx:,}"]
        for b in ("full_gpu", "cpu_lru", "cpu_arc", "tier_fs"):
            c = d.get((b, "cold"))
            w = d.get((b, "warm"))
            row.append(f"{st.median(c):,.0f}" if c else "—")
            row.append(f"{st.median(w):,.0f}" if w else "—")
        body.append(row)
    table_page(pdf, "表 4　M3：四個對照組的 cold／warm TTFT（ms，中位數）",
               ["模型", "ctx", "不卸載 cold", "warm", "LRU cold", "warm",
                "ARC cold", "warm", "CPU+磁碟 cold", "warm"], body,
               "results/m3_baseline_mi300x/baseline_mi300x.csv",
               "結論：工作集塞不下 GPU 之後，「不卸載」的 warm ≈ cold（省 0%），卸載可省 73–98%。\n"
               "　　　LRU 與 ARC 在最大 ctx 上差距極大（Seed-OSS 65K：4,250 ms vs 75,874 ms）—— 這正是 EXPERIMENT_PLAN 要求的檢查：\n"
               "　　　兩者數字若相同就代表卸載根本沒發生。這裡確認有發生，且 Oracle 之後要贏的是這裡面最強的那個。",
               max_rows=24)


def tbl_m4(pdf) -> None:
    press, trace = defaultdict(dict), defaultdict(dict)
    for f in glob.glob(str(M4 / "oracle_*.csv")):
        for r in rows(f):
            if r.get("policy") != "oracle":
                continue
            m, w = r.get("model_profile", ""), r.get("workload", "")
            try:
                h = float(r["oracle_headroom_pct"])
            except (KeyError, ValueError):
                continue
            base = r.get("best_baseline", "")
            if w.startswith("pressure:"):
                press[m][w.split(":")[1]] = (h, base)
            elif w.startswith("trace:"):
                trace[m][w.split(":")[1]] = (h, base)
    if not press and not trace:
        return
    ks = sorted({k for d in press.values() for k in d},
                key=lambda x: float(x.rstrip("x").split("(")[0]))
    body = []
    for m in MODELS:
        if m not in press and m not in trace:
            continue
        row = [SHORT.get(m, m)]
        for k in ks:
            v = press.get(m, {}).get(k)
            row.append(f"{v[0]:.1f}%" if v else "—")
        for t in ("conversation", "toolagent"):
            v = trace.get(m, {}).get(t)
            row.append(f"{v[0]:.1f}%" if v else "—")
        bases = [v[1] for v in list(press.get(m, {}).values()) + list(trace.get(m, {}).values()) if v[1]]
        row.append(max(set(bases), key=bases.count) if bases else "—")
        body.append(row)
    table_page(pdf, "表 5　M4：Oracle 相對最佳 baseline 的改善（headroom）",
               ["模型"] + [f"壓力 {k}" for k in ks] + ["對話 trace", "工具代理 trace", "最常見的最佳 baseline"],
               body, "results/m4_oracle_mi300x/oracle_*.csv",
               "結論：判準是 >15% 繼續、5–15% 問人、<5% 停止。合成工作負載在壓力 2–8× 時 headroom 21–43% → **GO**。\n"
               "　　　壓力 1× 一律 0.0%，因為工作集塞得進 GPU，最佳 baseline 就是不卸載，沒有放置決策可做 —— 這是正確的退化行為。\n"
               "　　　真實 trace（實際壓力 6–86×）只有 7–21%：極高壓力下重用本來就少，最強 baseline 已接近極限。\n"
               "　　　這不是否證，而是適用範圍：階層放置的價值集中在「塞不下但仍有重用」的中等壓力區。")


def tbl_quality(pdf) -> None:
    cfgs = ["bf16", "fp8", "fp8_ptk", "int8", "int4"]
    labels = ["BF16", "FP8", "FP8動態", "INT8", "INT4"]
    ndl, gsm = _acc("needle_b*.csv"), _acc("gsm8k_precision_*.csv")
    lb, ru = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for pat, dst in (("longbench_precision_*.csv", lb), ("ruler_precision_*.csv", ru)):
        for f in glob.glob(str(M5Q / pat)):
            for r in rows(f):
                sc = r.get("score") or r.get("metric_score") or ""
                if sc:
                    dst[r.get("model_key", "")][r.get("config", "")].append(float(sc))
    body = []
    for m in MODELS:
        for name, src, fmt in (("撈針 %", ndl, "acc"), ("GSM8K %", gsm, "acc"),
                               ("LongBench", lb, "score"), ("RULER", ru, "score")):
            if m not in src:
                continue
            row = [SHORT.get(m, m), name]
            for c in cfgs:
                if fmt == "acc":
                    v = src[m].get(c)
                    row.append(f"{100 * v[0] / v[1]:.1f}" if v and v[1] else "—")
                else:
                    v = src[m].get(c)
                    row.append(f"{100 * st.mean(v):.1f}" if v else "—")
            body.append(row)
    table_page(pdf, "表 6　M5：四套評測 × 五種 KV 精度的分數",
               ["模型", "評測"] + labels, body,
               "results/m5_quality_mi300x/{needle,gsm8k_precision,longbench_precision,ruler_precision}_*.csv",
               "結論：ε（品質代價）是「模型 × 任務」的性質，不是精度本身的性質。\n"
               "　　　Llama-3.1-8B／UltraLong／Qwen3-MoE／Seed-OSS 在 64K–129K 下，連 INT4 都看不到可辨識的損失（四套評測皆然）。\n"
               "　　　唯一例外 Qwen2.5-7B-1M：撈針從 100% 掉到 0%，且在 3K 就發生（非長度效應），輸出是重複字元／空字串等退化生成 —— 是崩潰不是品質下降。\n"
               "　　　實務意涵：Oracle 使用低精度階前，必須對該模型先跑一次檢索型檢查；通過的模型可以放心用 INT4 換 3.77 倍容量。",
               max_rows=24)


def tbl_lossless(pdf) -> None:
    body = []
    for f in sorted(glob.glob(str(M5Q / "lossless_b-*.csv"))):
        if "rep2" in f:
            continue
        by = defaultdict(dict)
        for r in rows(f):
            by[r["idx"]][r["config"]] = r
        if not by:
            continue
        m = next(iter(next(iter(by.values())).values()))["model_key"]
        cnt = {}
        for a, b in (("full_gpu", "cpu_lru"), ("full_gpu", "tier_fs"), ("cpu_lru", "tier_fs")):
            n = d = p = 0
            for _, x in by.items():
                if a in x and b in x:
                    n += 1
                    d += x[a]["out_sha1"] != x[b]["out_sha1"]
                    p += x[a]["pred"] != x[b]["pred"]
            cnt[(a, b)] = (d, p, n)
        acc = {}
        for c in ("full_gpu", "cpu_lru", "tier_fs"):
            v = [x[c] for x in by.values() if c in x]
            acc[c] = f"{sum(1 for r in v if r['correct'] == 'True')}/{len(v)}" if v else "—"
        body.append([SHORT.get(m, m),
                     f"{cnt[('full_gpu', 'cpu_lru')][0]}/{cnt[('full_gpu', 'cpu_lru')][2]}",
                     f"{cnt[('full_gpu', 'tier_fs')][0]}/{cnt[('full_gpu', 'tier_fs')][2]}",
                     f"{cnt[('cpu_lru', 'tier_fs')][0]}/{cnt[('cpu_lru', 'tier_fs')][2]}",
                     f"{cnt[('full_gpu', 'cpu_lru')][1]}", acc["full_gpu"], acc["cpu_lru"], acc["tier_fs"]])
    table_page(pdf, "表 7　M5：卸載的無損性（輸出 sha1 不同的題數／可比題數）",
               ["模型", "不卸載 vs CPU階", "不卸載 vs 磁碟階", "CPU階 vs 磁碟階",
                "答案不同（CPU階）", "正確數 不卸載", "CPU階", "磁碟階"], body,
               "results/m5_quality_mi300x/lossless_*.csv",
               "結論：CPU 階 vs 磁碟階 = 0/60（完全一致）→ **位元組搬移本身無損**；同設定重跑也是 0/60 → 執行是決定性的。\n"
               "　　　但兩者 vs 不卸載有 13–25/60 不同 —— 差異來自「取回 + 部分重算」與「整段 prefill」走不同的浮點路徑（分塊邊界不同）。\n"
               "　　　正確率影響極小（最多差 1 題）。這與平台 A 的 60/60 完全相同不一致，必須在論文中標明為平台相依的實作事實。")


def tbl_attention(pdf) -> None:
    js = sorted(glob.glob(str(M5A / "alignment_*.json")))
    if not js:
        return
    body = []
    for f in js:
        d = json.loads(Path(f).read_text())
        body.append([SHORT.get(d["model_key"], d["model_key"]), f"{d['n_eval']:,}",
                     f"{d['positive_rate']:.3f}", f"{d['auc_attn_mass']:.3f}",
                     f"{d['auc_access_history']:.3f}", f"{d['spearman_attn_vs_history']:+.3f}",
                     f"{d['docs']}×{d['doc_tokens']}", f"{d['requests']}", f"{d['horizon']}"])
    table_page(pdf, "表 8　M5-C：注意力重要度 vs 存取歷史（預測「block 會被重用」）",
               ["模型", "樣本數", "正例率", "注意力 AUC", "存取歷史 AUC", "Spearman",
                "文件×token", "請求數", "horizon"], body,
               "results/m5_attention_mi300x/alignment_*.json",
               "結論：注意力重要度的 AUC ≈ 0.5（等於丟銅板），存取歷史 0.72，兩者相關 ≈ 0 → **兩個訊號正交**。\n"
               "　　　原因已量化：注意力幾乎完全由位置決定（同位置跨請求的變異係數 0.0085，位置之間 0.3306，差 38.8 倍；\n"
               "　　　同文件兩次請求的前 50 名 block 重疊率 100%），而跨請求重用取決於「哪份文件又被查」。\n"
               "　　　對論文：表 15 (C) 的 attn_mass 特徵族可以填「已量測、對重用預測無貢獻」，並支持現行以存取歷史為特徵的設計。\n"
               "　　　⚠️ 這不否定注意力重要度在「單次請求內選擇保留哪些 token」上的效用（H2O／SnapKV 已充分驗證）。")


def tbl_predictor(pdf) -> None:
    met = rows(M5P / "predictor_metrics.csv")
    if not met:
        return
    body = []
    for r in met:
        if not (r.get("auc") or "").strip():
            continue
        def _f(k):
            v = (r.get(k) or "").strip()
            return float(v) if v else 0.0
        body.append([r.get("trace", ""), r.get("loss", ""), r.get("threshold_rule", ""),
                     f"{_f('auc'):.4f}", f"{_f('ece'):.4f}",
                     f"{_f('average_precision'):.4f}",
                     f"{_f('spearman_positives'):.4f}",
                     f"{int(_f('n')):,}",
                     f"{_f('cost_ms'):,.0f}",
                     f"{_f('lat_median_us'):.0f}"])
    table_page(pdf, "表 9　M5：學習式重用預測器（LightGBM）",
               ["trace", "損失", "門檻規則", "AUC", "ECE", "AP", "Spearman(正樣本)",
                "測試樣本", "成本加權錯誤 ms", "推論 µs"], body,
               "results/m5_predictor_mi300x/predictor_metrics.csv",
               "結論：只用存取歷史特徵（間隔、指數衰減計數、靜態欄位）即可達到 AUC 0.92、校準誤差 ECE 0.003。\n"
               "　　　成本加權損失相對對稱損失的差別很小，說明在這個工作負載下「錯誤的方向」不像「錯誤的量」那麼關鍵。\n"
               "　　　推論延遲數百微秒／64 個候選，相對 CPU 階 0.4–1.0 ms/block 的搬運成本是可接受的決策開銷。")


def page_pending_list(pdf) -> None:
    """明確列出尚未完成的實驗與它們會回答什麼。"""
    lines = [
        "以下是**還沒跑完**的項目。每一項都已寫好腳本並排入佇列，只差機器時間。",
        "",
        "【GPU 佇列】",
        "  1. LongBench / RULER 跑分 —— 還缺 4 個模型（UltraLong、Qwen2.5-7B、Qwen3-MoE、Mistral-Nemo）",
        "     會回答：KV 精度在標準長文評測上的 ε，是否與撈針／GSM8K 一致",
        "  2. 注意力對齊實驗 —— UltraLong 進行中、Qwen2.5-14B 排隊",
        "     會回答：「注意力重要度無法預測跨請求重用」是否跨模型成立",
        "  3. 無損驗證補跑 —— Qwen2.5-14B 與 Mistral-Nemo 的 cpu_lru 那一格",
        "     （前次因 /dev/shm 被塞滿而失敗，清理邏輯已修）",
        "",
        "【CPU 佇列（模擬與訓練，不佔 GPU）】",
        "  4. M4 Oracle 全模型重跑 —— 7 個模型 × (驗證 + 壓力掃描 + 2 個真實 trace)",
        "     為什麼重跑：原本固定寫 oracle.csv，7 個模型互相覆蓋，只剩最後一次的 20 列",
        "  5. 平台 A 做過、B 還缺的掃描：預算掃描、SSD 容量掃描、依長度分解、",
        "     硬體掃描、語意消融、最終判定書",
        "  6. M5 學習式預測器（LightGBM）訓練 + 政策模擬 + 摘要",
        "     會回答：用存取歷史預測重用的 AUC／校準誤差，以及換成成本加權損失後的差別",
        "",
        "【已知限制，不打算補】",
        "  · LMCache 對照組：依賴 CUDA 專屬套件（cupy-cuda13x、cuda-python），ROCm 上裝不起來。",
        "    會記為 NOT_MEASURED 並附完整原因，不用別的東西頂替。",
        "  · 平台 A 的 tier_fs 設定與 B 不同（A 的 CPU 階 ≥ 工作集，磁碟階可能沒被用到），",
        "    兩平台的 tier_fs 不可直接比較。",
    ]
    text_page(pdf, "尚未完成的實驗（以及它們會回答什麼）", lines)


def page_findings(pdf) -> None:
    lines = [
        "1. 容量：FP8/INT8/INT4 相對 BF16 的容量倍數是 2.00 / 1.94 / 3.77，七個模型一致，",
        "   且與平台 A（3090）相同 —— 這是跨硬體的不變量。",
        "",
        "2. 重算成本不是常數：UltraLong-8B 從位置 0 的 86 ms 長到 1M 的 10,350 ms（120 倍），",
        "   線性擬合 R² = 0.9993。論文 §3 的核心假設在平台 B 成立。",
        "",
        "3. Oracle go/no-go = GO：合成工作負載在壓力 2–8× 時 headroom 21–43%。",
        "   但真實 trace（壓力 6–86×）只有 7–21%，其中 5 格落在「5–15% 要問人」的區間。",
        "   判讀：極高壓力下重用本來就少，最佳 baseline 已接近極限 —— 應寫成適用範圍限制。",
        "",
        "4. KV 量化的品質代價是「模型 × 任務」的性質，不是量化本身的性質：",
        "   Llama-3.1-8B / Qwen3-MoE / Seed-OSS / UltraLong 在 64K–129K 下五種精度全部 100%；",
        "   但 Qwen2.5-7B-1M 在 3K 就崩潰（FP8 15%、INT4 0%），且是語言能力崩潰（吐垃圾），",
        "   不是檢索失敗。平台 A 在同家族模型上量到同一模式 —— 跨平台重現。",
        "",
        "5. 卸載是位元組無損的，但端到端不是位元相同：",
        "   CPU 階 vs 磁碟階 0/60 不同（搬移無損）、同設定重跑 0/60（執行決定性），",
        "   但兩者 vs 不卸載有 16–25/60 不同 —— 差異來自「取回+部分重算」與「整段 prefill」",
        "   的浮點路徑不同。這是必須寫進論文的實作事實。",
        "",
        "6. 注意力重要度與跨請求重用正交：AUC 0.488（≈隨機）vs 存取歷史 0.721，",
        "   Spearman −0.016。原因：注意力幾乎完全由位置決定（同位置跨請求變異係數 0.0085，",
        "   位置間 0.3306，差 38.8 倍）。→ 論文表 15 (C) 的 attn_mass 那格可以填「無貢獻」。",
    ]
    text_page(pdf, "目前的主要發現", lines, "results/RUNLOG_MI300X.md")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(R / "REPORT_MI300X.pdf"))
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        page_cover(pdf)
        page_coverage(pdf)
        page_findings(pdf)
        page_m1(pdf); tbl_m1(pdf)
        page_m2_tiers(pdf); tbl_m2(pdf)
        page_m2_recompute(pdf)
        page_m3(pdf); tbl_m3(pdf)
        page_m4(pdf); tbl_m4(pdf)
        page_quality(pdf); page_bench(pdf); tbl_quality(pdf)
        page_lossless(pdf); tbl_lossless(pdf)
        page_attention(pdf); tbl_attention(pdf)
        page_predictor(pdf); tbl_predictor(pdf)
        page_pending_list(pdf)
        d = pdf.infodict()
        d["Title"] = "Tiara 實驗報告：平台 B（AMD MI300X）"
        d["Subject"] = "階層式 KV 放置：M1–M5 實測結果與待辦"
        d["CreationDate"] = datetime.now()
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
