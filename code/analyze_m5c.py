#!/usr/bin/env python3
"""把 M5-(c) 的 LongBench / RULER 逐題分數整理成論文用的表。

## 為什麼要配對（paired）統計

四個 KV 精度設定吃的是**同一批 prompt**（同樣的題目、同樣的順序、
`temperature=0`、固定 seed），唯一變動的是 `--kv-cache-dtype`。
所以「BF16 減 INT4」應該逐題相減再取平均，而不是把兩個獨立平均相減：
配對版本把題目難度的變異消掉，信賴區間會窄得多。
GSM8K 那一輪之所以量不出東西（±3.7pp），一部分就是沒有用上這一點。

CI 用 **paired bootstrap**（重抽題目、不重抽設定），10,000 次。
不假設常態，也不需要每個任務的分數是二元的
（LongBench 的 F1／ROUGE 是連續值，二項式 CI 在這裡不適用）。

## 這支腳本會拒絕做的事

* 沒有 CSV 就印 `NOT_MEASURED` 並以非零碼結束，不生表。
* 不跨任務加權平均原始分數（F1 與 ROUGE 的尺度不同）——
  巨觀平均是「先算每個任務的平均，再對任務取平均」，且一律附上逐任務值。

用法:
    python code/analyze_m5c.py                     # 兩個 suite 都印
    python code/analyze_m5c.py --suite ruler --latex
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results/m5_quality"
ORDER = ["bf16", "fp8", "int8", "int4"]
LABEL = {"bf16": "BF16", "fp8": "FP8", "int8": "INT8", "int4": "INT4"}

# 論文表格用的中文任務名 + 類別（類別是圖 4(c) 的分組依據）
TASK_META = {
    # LongBench
    "multifieldqa_en":      ("MultiFieldQA", "單文件 QA"),
    "qasper":               ("Qasper", "單文件 QA"),
    "hotpotqa":             ("HotpotQA", "多跳 QA"),
    "2wikimqa":             ("2WikiMQA", "多跳 QA"),
    "gov_report":           ("GovReport", "摘要"),
    "trec":                 ("TREC", "few-shot 分類"),
    "passage_retrieval_en": ("PassageRetrieval", "合成檢索"),
    # RULER
    "niah_multikey_2":      ("NIAH multikey-2", "多鍵檢索"),
    "niah_multikey_3":      ("NIAH multikey-3", "多鍵檢索"),
    "niah_multivalue":      ("NIAH multivalue", "多值回收"),
    "niah_multiquery":      ("NIAH multiquery", "多查詢回收"),
    "vt":                   ("VT", "多跳追蹤"),
    "cwe":                  ("CWE", "詞頻聚合"),
    "fwe":                  ("FWE", "詞頻聚合"),
}


def load(suite: str) -> list[dict]:
    p = RES / f"{suite}_precision.csv"
    if not p.exists():
        print(f"NOT_MEASURED: 找不到 {p}")
        return []
    return list(csv.DictReader(p.open()))


def by_item(rows: list[dict]) -> tuple[list[str], list[str], dict]:
    """回傳 (設定順序, 任務順序, {(config, task, idx): score})。"""
    cfgs = [c for c in ORDER if any(r["config"] == c for r in rows)]
    tasks: list[str] = []
    for r in rows:
        if r["task"] not in tasks:
            tasks.append(r["task"])
    s = {(r["config"], r["task"], int(r["idx"])): float(r["score"]) for r in rows}
    return cfgs, tasks, s


def paired_ci(base: list[float], other: list[float], n_boot: int = 10000,
              seed: int = 20260901) -> tuple[float, float, float]:
    """配對差值（base - other）的均值與 95% bootstrap CI，單位為百分點。"""
    d = [(b - o) * 100 for b, o in zip(base, other)]
    if not d:
        return float("nan"), float("nan"), float("nan")
    mean = sum(d) / len(d)
    rng = random.Random(seed)
    n = len(d)
    boots = sorted(sum(rng.choices(d, k=n)) / n for _ in range(n_boot))
    return mean, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]


def report(suite: str, rows: list[dict], latex: bool) -> None:
    cfgs, tasks, s = by_item(rows)
    base = cfgs[0]
    idxs = {t: sorted({k[2] for k in s if k[1] == t and k[0] == base}) for t in tasks}
    # 只留下**每個設定都有**的題目，配對才成立
    for t in tasks:
        idxs[t] = [i for i in idxs[t] if all((c, t, i) in s for c in cfgs)]

    per: dict[tuple[str, str], float] = {}
    for c in cfgs:
        for t in tasks:
            v = [s[(c, t, i)] for i in idxs[t]]
            per[(c, t)] = 100 * sum(v) / len(v) if v else float("nan")

    w = max(len(TASK_META.get(t, (t,))[0]) for t in tasks) + 2
    print(f"\n{'=' * 78}\n{suite}：KV 精度 × 任務（分數 = 官方指標 × 100）\n{'=' * 78}")
    print(f"{'任務':<{w}}{'類別':<14}{'n':>4}" + "".join(f"{LABEL[c]:>9}" for c in cfgs))
    for t in tasks:
        nm, cat = TASK_META.get(t, (t, "—"))
        print(f"{nm:<{w}}{cat:<14}{len(idxs[t]):>4}"
              + "".join(f"{per[(c, t)]:>9.2f}" for c in cfgs))

    macro = {c: sum(per[(c, t)] for t in tasks) / len(tasks) for c in cfgs}
    print("-" * 78)
    print(f"{'巨觀平均':<{w}}{'':<14}{'':>4}" + "".join(f"{macro[c]:>9.2f}" for c in cfgs))
    print(f"{'相對 BF16 保留':<{w}}{'':<14}{'':>4}"
          + "".join(f"{100 * macro[c] / macro[base]:>8.1f}%" for c in cfgs))

    print(f"\n配對差值（BF16 − X），單位百分點，95% bootstrap CI（10,000 次重抽）")
    print(f"{'任務':<{w}}" + "".join(f"{LABEL[c]:>26}" for c in cfgs[1:]))
    for t in tasks:
        nm = TASK_META.get(t, (t,))[0]
        line = f"{nm:<{w}}"
        for c in cfgs[1:]:
            m, lo, hi = paired_ci([s[(base, t, i)] for i in idxs[t]],
                                  [s[(c, t, i)] for i in idxs[t]])
            line += f"{m:>10.1f} [{lo:>6.1f},{hi:>6.1f}]"
        print(line)
    allb = [s[(base, t, i)] for t in tasks for i in idxs[t]]
    line = f"{'合併':<{w}}"
    for c in cfgs[1:]:
        m, lo, hi = paired_ci(allb, [s[(c, t, i)] for t in tasks for i in idxs[t]])
        line += f"{m:>10.1f} [{lo:>6.1f},{hi:>6.1f}]"
    print(line)

    if latex:
        print(f"\n% ── {suite}：貼進 main.tex 的表身 ──")
        for t in tasks:
            nm, cat = TASK_META.get(t, (t, "—"))
            cells = " & ".join(f"{per[(c, t)]:.1f}" for c in cfgs)
            print(f"\\texttt{{{nm}}} & {cat} & {len(idxs[t])} & {cells} \\\\")
        print("\\midrule")
        print("巨觀平均 & --- & --- & " + " & ".join(f"\\textbf{{{macro[c]:.1f}}}"
                                                 for c in cfgs) + " \\\\")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", nargs="*", default=["longbench", "ruler"])
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    missing = 0
    for suite in a.suite:
        rows = load(suite)
        if not rows:
            missing += 1
            continue
        report(suite, rows, a.latex)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
