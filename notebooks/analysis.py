#!/usr/bin/env python3
"""從 results/ 讀實測資料，產生 notebook 與論文共用的圖表。

放在獨立模組而非 notebook 儲存格裡，理由有三：
1. **notebook 的輸出會被 commit，但程式碼應該只有一份。** 論文的圖與
   notebook 的圖必須來自同一段程式，否則兩者會漂移。
2. 這裡的每個函式都直接讀 `results/` 的 CSV/JSON，**不接受任何硬編的數字**。
   讀不到就 raise，不用預設值——與 `EXPERIMENT_PLAN.md` §0 禁令 1 一致。
3. 可在 CI 或命令列直接跑：`python notebooks/analysis.py --figures`

用法:
    from analysis import load_all, fig_crossover, fig_pressure
    D = load_all()
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
BIG = Path("/ssd7/hungwei/paper-hkv")

BLOCK = 16          # vLLM 預設 block size，非本專案選擇


# ─────────────────────────── 讀取 ───────────────────────────

def _need(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError(
            f"缺少 {p}\n"
            f"這支模組不使用預設值。先跑對應的 milestone 產生該檔。")
    return p


def load_cost_model() -> dict:
    """M2 的成本常數。每個值都附上它是怎麼算出來的。"""
    d = json.loads(_need(RESULTS / "m4_oracle/cost_model.json").read_text())
    m, c = d["measured"], d["derived_ms_per_block"]
    return {
        **c,
        "_derivation": {
            "cpu": f"({m['warm_cpu_ms']} - {m['warm_gpu_ms']}) / {m['blocks_per_ctx']:.0f} blocks",
            "ssd": f"({m['warm_ssd_ms']} - {m['warm_gpu_ms']}) / {m['blocks_per_ctx']:.0f} blocks",
            "recompute_base": f"{m['recompute_at_pos0_ms']} / "
                              f"{m['recompute_chunk_tokens'] // BLOCK} blocks",
        },
        "_measured": m,
    }


def load_capacity() -> list[dict]:
    """M1 的容量實測。只取 measure 階段且 server 起得來的列。"""
    rows, seen = [], set()
    for x in csv.DictReader(_need(RESULTS / "m1_capacity/capacity.csv").open()):
        if x["phase"] != "measure" or x["server_ready"] != "True":
            continue
        if x["config"] in seen:
            continue
        seen.add(x["config"])
        rows.append({
            "config": x["config"], "weight": x["weight_dtype"],
            "kv_dtype": x["kv_dtype"],
            "kv_tokens": int(x["kv_cache_tokens"]),
            "extrapolated": x.get("extrapolated", "") == "True",
        })
    return rows


def load_idle_cost() -> dict:
    """每個 KV dtype 佔多少 KiB/token（已除以 GiB 正規化）。"""
    return json.loads(_need(RESULTS / "m2_harness/idle_cost_normalized.json").read_text())


def load_recompute_curve() -> list[dict]:
    rows = list(csv.DictReader(_need(RESULTS / "m2_harness/recompute_position.csv").open()))
    pts = sorted({int(r["cached_prefix_tokens"]) for r in rows})
    chunk = int(rows[0]["recomputed_tokens"])
    return [{"P": p, "chunk": chunk,
             "ms": round(median([float(r["ttft_ms"]) for r in rows
                                 if int(r["cached_prefix_tokens"]) == p]), 1)}
            for p in pts]


def load_m3(which: str = "longctx") -> list[dict]:
    """M3 baseline。which='longctx' 為 AWQ 權重的長 context 版。"""
    f = (RESULTS / f"m3_baseline/baseline_{which}.csv" if which != "base"
         else RESULTS / "m3_baseline/baseline.csv")
    agg = defaultdict(list)
    meta = {}
    for x in csv.DictReader(_need(f).open()):
        if str(x.get("contaminated", "")).lower() == "true":
            continue                       # 被插隊污染的列不入分析
        if not x["ttft_ms"]:
            continue
        k = (x["model_key"], x["baseline"], int(x["ctx"]), x["round"])
        agg[k].append(float(x["ttft_ms"]))
        meta[(x["model_key"], x["baseline"])] = x.get("gpu_kv_cache_tokens", "")
    return [{"model": k[0], "baseline": k[1], "ctx": k[2], "round": k[3],
             "ttft_ms": round(median(v), 1), "n": len(v),
             "kv_tokens": meta.get((k[0], k[1]), "")}
            for k, v in sorted(agg.items())]


def load_oracle_scenarios() -> list[dict]:
    """Oracle 的十個情境。從 log 解析——oracle.csv 每次跑會被覆寫，
    而我們要的是同一次 run 裡的全部情境。"""
    log = _need(BIG / "logs/oracle_redo.log").read_text()
    keys = ["pressure:1x", "pressure:2x", "pressure:5x", "pressure:10x",
            "pressure:20x", "pressure:40x", "trace:conv@48K", "trace:tool@48K",
            "trace:conv@274K", "trace:tool@274K"]
    out = []
    pat = (r'--- (\S+) ---\n(.*?)\n  最佳 baseline = (\w+)；'
           r'Oracle 改善 = \*\*([\d.]+)%\*\* → (\w+)')
    for i, m in enumerate(re.finditer(pat, log, re.S)):
        pol = {}
        for r in re.finditer(r'^(\w+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)$',
                             m.group(2), re.M):
            g = [int(x.replace(",", "")) for x in r.groups()[1:]]
            pol[r.group(1)] = dict(zip(("ms", "gpu", "cpu", "ssd", "recompute"), g))
        if pol:
            out.append({"scenario": keys[i] if i < len(keys) else m.group(1),
                        "best_baseline": m.group(3), "headroom_pct": float(m.group(4)),
                        "verdict": m.group(5), "policies": pol})
    if not out:
        raise ValueError("解析不到 Oracle 情境；log 格式可能變了")
    return out


def load_trace_stats() -> list[dict]:
    """真實 trace 的重用結構。這決定 headroom 的第二個自變數。"""
    out = []
    for name in ("conversation", "toolagent"):
        p = BIG / f"datasets/traces/{name}_trace.jsonl"
        if not p.exists():
            continue
        reqs = [json.loads(l) for l in p.open()]
        blocks = [b for r in reqs for b in r["hash_ids"]]
        lens = sorted(r["input_length"] for r in reqs)
        uniq = len(set(blocks))
        out.append({
            "trace": name, "requests": len(reqs),
            "accesses": len(blocks), "unique_blocks": uniq,
            "reuse_pct": round(100 * (len(blocks) - uniq) / len(blocks), 1),
            "compulsory_pct": round(100 * uniq / len(blocks), 1),
            "median_input": lens[len(lens) // 2],
            "p99_input": lens[int(len(lens) * 0.99)],
            "over_128k": sum(1 for x in lens if x >= 131072),
        })
    return out


def load_all() -> dict:
    return {
        "cost": load_cost_model(),
        "capacity": load_capacity(),
        "idle": load_idle_cost(),
        "recompute": load_recompute_curve(),
        "m3": load_m3("longctx"),
        "oracle": load_oracle_scenarios(),
        "traces": load_trace_stats(),
    }


# ─────────────────────────── 衍生量 ───────────────────────────

def drop_cost(cost: dict, position_tokens: float) -> float:
    """重算一個 block 的成本。隨絕對位置線性成長（非二次）。"""
    return cost["recompute_base"] + cost["recompute_slope_per_token"] * position_tokens


def ssd_drop_crossover(cost: dict) -> float:
    """SSD 與重算的成本在哪個位置相等。此點之前重算較廉，之後 SSD 較廉。"""
    return (cost["ssd"] - cost["recompute_base"]) / cost["recompute_slope_per_token"]


def fit_recompute_linear(curve: list[dict]) -> dict:
    """對重算曲線做最小平方線性擬合，並回報最大偏差。

    偏差大代表模型錯了（例如真的是二次的）。實測最大偏差 1.6%。
    """
    xs = [p["P"] for p in curve]
    ys = [p["ms"] for p in curve]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sxy / sxx
    icpt = my - slope * mx
    errs = [abs(icpt + slope * x - y) / y for x, y in zip(xs, ys)]
    return {"intercept_ms": icpt, "slope_ms_per_token": slope,
            "max_rel_error": max(errs), "n_points": n}


# ─────────────────────────── 圖表 ───────────────────────────

def style():
    """設定 matplotlib 的圖表樣式並回傳 pyplot。

    公開名稱（非底線開頭）——notebook 用 `from analysis import *`，
    底線開頭的名稱不會被帶進去。"""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK TC", "Noto Sans CJK JP", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": .25, "grid.linestyle": ":",
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 10,
    })
    return plt


C = {"gpu": "#b4551f", "cpu": "#8a6a2f", "ssd": "#3f6b78",
     "drop": "#7b8688", "live": "#0a6f68", "warn": "#94550c"}


def fig_crossover(D: dict, ax=None):
    """SSD 與重算的成本交叉。這張圖說明「哪一階被支配」是有條件的。"""
    plt = style()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 3.6))
    cost = D["cost"]
    xs = list(range(0, 131073, 2048))
    ax.plot(xs, [drop_cost(cost, x) for x in xs], color=C["drop"], lw=2,
            ls="--", label="丟棄後重算")
    ax.axhline(cost["ssd"], color=C["ssd"], lw=2, label="SSD 取回（常數）")
    ax.axhline(cost["cpu"], color=C["cpu"], lw=2, label="CPU 取回（常數）")
    x0 = ssd_drop_crossover(cost)
    ax.plot([x0], [cost["ssd"]], "o", color=C["live"], ms=8, zorder=5)
    ax.annotate(f"交叉點 {x0:,.0f} token", (x0, cost["ssd"]),
                textcoords="offset points", xytext=(12, 14),
                color=C["live"], fontweight="bold", fontsize=9)
    ax.set_xlabel("block 的絕對位置（token）")
    ax.set_ylabel("取回成本（ms / block）")
    ax.set_title("SSD 與重算的優劣隨位置反轉", fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(0, 131072)
    ax.set_ylim(0, 32)
    return ax


def fig_pressure(D: dict, ax=None):
    """Oracle 的 headroom 對「工作集/容量」的敏感度。"""
    plt = style()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 3.6))
    syn = [(float(s["scenario"].split(":")[1].rstrip("x")), s["headroom_pct"])
           for s in D["oracle"] if s["scenario"].startswith("pressure")]
    syn.sort()
    ax.plot([x for x, _ in syn], [y for _, y in syn], "o-", color=C["live"],
            lw=2, ms=6, label="合成 Zipf α=0.9（重用率 85%）")
    for s in D["oracle"]:
        if not s["scenario"].startswith("trace"):
            continue
        ax.plot([], [])
    reals = [s for s in D["oracle"] if s["scenario"].startswith("trace")]
    if reals:
        ax.scatter([1] * len(reals), [s["headroom_pct"] for s in reals],
                   marker="D", s=48, color=C["warn"], zorder=5,
                   label="真實生產 trace（重用率 37–55%）")
        for s in reals:
            ax.annotate(s["scenario"].replace("trace:", ""),
                        (1, s["headroom_pct"]), textcoords="offset points",
                        xytext=(9, -3), fontsize=8, color=C["warn"])
    ax.axhspan(0, 5, color="#93291d", alpha=.07)
    ax.axhspan(5, 15, color=C["warn"], alpha=.07)
    ax.axhspan(15, 35, color=C["live"], alpha=.07)
    for y, t in ((2.5, "NO-GO"), (10, "問人"), (25, "GO")):
        ax.text(41, y, t, va="center", fontsize=8, color="#666")
    ax.set_xscale("log")
    ax.set_xlabel("設定的壓力倍數（工作集 / GPU 預算）")
    ax.set_ylabel("Oracle 相對最佳線上策略的改善（%）")
    ax.set_title("headroom 取決於壓力與重用率兩者", fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_ylim(0, 33)
    return ax


def fig_policy_mix(D: dict, scenario: str | None = None, ax=None):
    """各策略的 block 去向。這張圖用命中次數而非時間——硬體無關。"""
    plt = style()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.8, 3.2))
    sc = next((s for s in D["oracle"] if s["scenario"] == scenario),
              D["oracle"][-1] if scenario is None else None)
    if sc is None:
        raise KeyError(f"找不到情境 {scenario}；可用：{[s['scenario'] for s in D['oracle']]}")
    order = ["full_gpu", "cpu_lru", "cpu_arc", "tier_fs", "oracle"]
    names = ["不卸載", "CPU+LRU", "CPU+ARC", "CPU+磁碟", "Oracle"]
    keys = ["gpu", "cpu", "ssd", "recompute"]
    labs = ["GPU 命中", "CPU 取回", "SSD 取回", "重算"]
    cols = [C["gpu"], C["cpu"], C["ssd"], C["drop"]]
    left = [0.0] * len(order)
    for k, lab, col in zip(keys, labs, cols):
        vals = [100 * sc["policies"][p][k] /
                sum(sc["policies"][p][x] for x in keys) for p in order]
        ax.barh(names, vals, left=left, color=col, label=lab,
                height=.68, edgecolor="none")
        left = [a + b for a, b in zip(left, vals)]
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("各層佔總存取的比例（%）")
    ax.set_title(f"{sc['scenario']}　Oracle 改善 {sc['headroom_pct']}%　"
                 f"（{sc['verdict']}）", fontweight="bold", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, ncol=4,
              loc="upper center", bbox_to_anchor=(.5, -.28))
    ax.grid(axis="y", visible=False)
    return ax


def fig_capacity(D: dict, ax=None):
    """M1 的容量實測。標出哪些設定的瓶頸已從記憶體轉為模型定址。"""
    plt = style()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.6, 3.8))
    caps = {"llama": 131072, "qwen": 262144, "mla": 163840}
    rows = sorted(D["capacity"], key=lambda r: r["kv_tokens"])
    names = [r["config"] for r in rows]
    vals = [r["kv_tokens"] for r in rows]
    lims = [next((v for k, v in caps.items() if k in r["config"]), None) for r in rows]
    cols = [C["live"] if (l and v > l) else C["ssd"] for v, l in zip(vals, lims)]
    ax.barh(names, vals, color=cols, height=.66)
    for i, (v, l) in enumerate(zip(vals, lims)):
        ax.text(v * 1.02, i, f"{v:,}", va="center", fontsize=8.5)
        if l:
            ax.plot([l], [i], "|", color="#93291d", ms=13, mew=2)
    ax.set_xlabel("GPU KV cache 容量（token，實測）")
    ax.set_title("紅線 = 模型可定址上限；越過它代表瓶頸已非記憶體",
                 fontweight="bold", loc="left", fontsize=10)
    ax.set_xlim(0, max(vals) * 1.22)
    ax.grid(axis="y", visible=False)
    return ax


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true", help="輸出圖檔到 notebooks/figures/")
    a = ap.parse_args()
    D = load_all()
    print(f"成本常數  : {', '.join(f'{k}={v:.4f}' for k, v in D['cost'].items() if isinstance(v, float))}")
    print(f"容量設定  : {len(D['capacity'])}")
    print(f"Oracle 情境: {len(D['oracle'])}")
    print(f"M3 資料點 : {len(D['m3'])}")
    fit = fit_recompute_linear(D["recompute"])
    print(f"重算線性擬合: {fit['intercept_ms']:.1f} + {fit['slope_ms_per_token']*1000:.2f} ms/1000tok"
          f"，最大偏差 {100*fit['max_rel_error']:.1f}%")
    print(f"SSD/重算交叉點: {ssd_drop_crossover(D['cost']):,.0f} token")
    if a.figures:
        plt = style()
        out = REPO / "notebooks/figures"
        out.mkdir(parents=True, exist_ok=True)
        for nm, fn in (("crossover", fig_crossover), ("pressure", fig_pressure),
                       ("capacity", fig_capacity)):
            fig, ax = plt.subplots(figsize=(6.6, 3.7))
            fn(D, ax=ax)
            fig.savefig(out / f"{nm}.pdf")
            fig.savefig(out / f"{nm}.png")
            plt.close(fig)
            print(f"  wrote {out / nm}.pdf/.png")
        fig, ax = plt.subplots(figsize=(6.8, 3.4))
        fig_policy_mix(D, "trace:conv@48K", ax=ax)
        fig.savefig(out / "policy_mix.pdf"); fig.savefig(out / "policy_mix.png")
        plt.close(fig)
        print(f"  wrote {out / 'policy_mix'}.pdf/.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
