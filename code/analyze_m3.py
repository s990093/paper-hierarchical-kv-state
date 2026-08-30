#!/usr/bin/env python3
"""把 results/m3_baseline/baseline.csv 整理成可讀的表 + 論文用的 LaTeX 片段。

## 這支腳本會拒絕做的事

* **不會**把 `concurrency_mode` 不同的列混在一起比較（平行跑的時間數字被自己人的
  PCIe 爭用污染，見 `m3_baseline.py` 的說明）。
* **不會**顯示 `contaminated=True` 的列。
* **不會**把有 `caveat` 的 baseline（例如缺編譯擴充的 lmcache）跟其他的並排而不標註。
* 沒有資料就印 `NOT_MEASURED`，不補值、不內插。

## 主要輸出

每個 (baseline, ctx) 的 cold / warm TTFT 中位數與差值。
**warm 相對 cold 的改善就是那一階卸載的價值**；`full_gpu` 是對照
（沒有第二階，warm 只能重算）。

用法:
    python code/analyze_m3.py
    python code/analyze_m3.py --mode serial --latex
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median, pstdev

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO / "results/m3_baseline/baseline.csv"
ORDER = ["full_gpu", "cpu_lru", "cpu_arc", "tier_fs", "lmcache"]


def load(path: Path, mode: str | None) -> list[dict]:
    if not path.exists():
        print(f"no such file: {path}")
        return []
    rows = list(csv.DictReader(path.open()))
    kept, dropped_contam, dropped_mode = [], 0, 0
    for r in rows:
        if str(r.get("contaminated", "")).lower() == "true":
            dropped_contam += 1
            continue
        if mode and r.get("concurrency_mode", "parallel") != mode:
            dropped_mode += 1
            continue
        kept.append(r)
    if dropped_contam:
        print(f"⚠️  丟棄 {dropped_contam} 列（GPU 被插隊污染）")
    if dropped_mode:
        print(f"ℹ️  丟棄 {dropped_mode} 列（concurrency_mode != {mode}）")
    return kept


def f(x: str | None) -> float | None:
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--model", default=None, help="只看某個 model_key")
    ap.add_argument("--mode", default=None, choices=["parallel", "serial"],
                    help="只看某一種 concurrency_mode（強烈建議指定）")
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()

    rows = load(Path(a.csv), a.mode)
    if a.model:
        rows = [r for r in rows if r["model_key"] == a.model]
    if not rows:
        print("NOT_MEASURED — 沒有符合條件的資料")
        return 1

    modes = {r.get("concurrency_mode", "parallel") for r in rows}
    if len(modes) > 1:
        print(f"🔴 CSV 裡混了 {modes} 兩種模式。加 --mode 指定一種，"
              f"不要混著比較。")
        return 2

    caveats = {r["baseline"]: r.get("caveat", "") for r in rows if r.get("caveat")}

    # (model, baseline, ctx, round) -> [ttft...]
    agg: dict[tuple, list[float]] = defaultdict(list)
    kv: dict[tuple, str] = {}
    for r in rows:
        t = f(r.get("ttft_ms"))
        if t is None:
            continue
        agg[(r["model_key"], r["baseline"], int(r["ctx"]), r["round"])].append(t)
        kv[(r["model_key"], r["baseline"])] = r.get("gpu_kv_cache_tokens", "?")

    for mk in sorted({k[0] for k in agg}):
        ctxs = sorted({k[2] for k in agg if k[0] == mk})
        bases = [b for b in ORDER if (mk, b) in kv] + \
                sorted({k[1] for k in agg if k[0] == mk} - set(ORDER))
        print(f"\n{'=' * 78}\nmodel = {mk}   mode = {list(modes)[0]}   "
              f"GPU KV cache = {kv.get((mk, bases[0]), '?')} tokens\n{'=' * 78}")
        print("cold = 第一次見到這些前綴；warm = 逐出後再送一次同樣的前綴")
        print("Δ% > 0 代表『第二階真的把 KV 取回來了』，full_gpu 是沒有第二階的對照\n")

        hdr = f"{'baseline':<12}{'ctx':>7}{'工作集':>10}{'cold ms':>11}{'warm ms':>11}{'Δ%':>8}  n"
        print(hdr)
        print("-" * len(hdr))
        for b in bases:
            for c in ctxs:
                cold = agg.get((mk, b, c, "cold"), [])
                warm = agg.get((mk, b, c, "warm"), [])
                if not cold or not warm:
                    print(f"{b:<12}{c:>7}{'':>10}{'NOT_MEASURED':>11}")
                    continue
                mc, mw = median(cold), median(warm)
                d = 100 * (mc - mw) / mc
                n = len(cold)
                ws = c * n  # 工作集 = N 個前綴 × ctx
                print(f"{b:<12}{c:>7}{ws:>10,}{mc:>11.1f}{mw:>11.1f}{d:>7.1f}%  {n}")
            print()

        if caveats:
            print("⚠️  不對等的比較（必須跟數字一起出現）:")
            for b, cv in caveats.items():
                print(f"    {b}: {cv}")

    if a.latex:
        print("\n% ---- LaTeX（貼進 main.tex 之前先確認 mode 與 caveat）----")
        print("% concurrency_mode = " + list(modes)[0])
        for b, cv in caveats.items():
            print(f"% CAVEAT {b}: {cv}")
        for mk in sorted({k[0] for k in agg}):
            for b in [x for x in ORDER if any(k[1] == x for k in agg)]:
                for c in sorted({k[2] for k in agg if k[0] == mk}):
                    cold = agg.get((mk, b, c, "cold"), [])
                    warm = agg.get((mk, b, c, "warm"), [])
                    if not cold or not warm:
                        continue
                    print(f"{mk} & {b} & {c} & {median(cold):.0f} & "
                          f"{median(warm):.0f} & "
                          f"{100 * (median(cold) - median(warm)) / median(cold):.1f}\\% \\\\")
    return 0


if __name__ == "__main__":
    sys.exit(main())
