#!/usr/bin/env python3
"""把 main.tex 裡 M5-(c) 的每個數字，對 `results/` 的 CSV 重新算一次。

CLAUDE.md 禁令 1 與 3 的執行工具：**論文裡的每個數字都要能追溯到一個輸出檔**。
這支腳本不讀論文的散文，只讀兩張表的儲存格、caption 裡的極值、
以及正文引用的配對 bootstrap 區間，逐一與 CSV 重算的值比對。

判準是**「論文印出來的一位小數，等於 CSV 值的一位小數」**，
不是「差距小於某個容忍值」——後者會放過 53.448 被寫成 53.5 這種錯
（本腳本第一次跑就抓到這一筆）。

bootstrap 區間是決定性的（`analyze_m5c.paired_ci` 的 seed 固定為 20260901），
所以區間也能逐字比對，不是「大約落在」。

用法:
    PYTHONPATH=/ssd7/hungwei/paper-hkv/pylibs python code/audit_m5c_claims.py
回傳 0 = 全部相符；1 = 有不符（會逐筆列出）。
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_m5c import ORDER, paired_ci  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results/m5_quality"

NAME2TASK = {
    "MultiFieldQA": ("longbench", "multifieldqa_en"), "Qasper": ("longbench", "qasper"),
    "HotpotQA": ("longbench", "hotpotqa"), "2WikiMQA": ("longbench", "2wikimqa"),
    "GovReport": ("longbench", "gov_report"), "TREC": ("longbench", "trec"),
    "PassageRetr.": ("longbench", "passage_retrieval_en"),
    "NIAH multikey-2": ("ruler", "niah_multikey_2"),
    "NIAH multikey-3": ("ruler", "niah_multikey_3"),
    "NIAH multivalue": ("ruler", "niah_multivalue"),
    "NIAH multiquery": ("ruler", "niah_multiquery"),
    "VT": ("ruler", "vt"), "CWE": ("ruler", "cwe"), "FWE": ("ruler", "fwe"),
}

fails: list[str] = []


def chk(label: str, claimed: str, actual: float) -> None:
    want = f"{actual:.1f}"
    ok = claimed == want
    if not ok:
        fails.append(f"{label}: 論文 {claimed}，CSV 算出 {actual:.4f} -> 應為 {want}")
    print(f"{'  ok' if ok else '🔴  '} {label:34s} 論文 {claimed:>6s}  CSV {actual:>8.3f}")


def load(suite: str) -> dict:
    p = RES / f"{suite}_precision.csv"
    if not p.exists():
        sys.exit(f"NOT_MEASURED: 找不到 {p}")
    per, order = defaultdict(dict), []
    for r in csv.DictReader(p.open()):
        per[(r["config"], r["task"])][int(r["idx"])] = float(r["score"])
        if r["task"] not in order:
            order.append(r["task"])
    # 🔴 任務順序必須與 analyze_m5c 一致（CSV 出現序，不是字典序）。
    #    合併全部題目的 bootstrap 是照這個順序把題目串起來再重抽，
    #    換個順序就換一組重抽序列，區間會差 0.1--0.2 pp。第一版用 sorted()，
    #    於是四個「不符」全是這個假警報。
    per["__order__"] = order
    return per


def main() -> int:
    data = {s: load(s) for s in ("longbench", "ruler")}
    tex = (REPO / "main.tex").read_text()

    def mean(suite: str, cfg: str, task: str) -> float:
        v = data[suite][(cfg, task)]
        return 100 * sum(v.values()) / len(v)

    # ── 附錄表 tab:eps-longctx 的逐任務儲存格 ──
    print("── 附錄表 tab:eps-longctx ──")
    cell = r"(?:\\textbf\{)?([\d.]+)\}?"
    pat = re.compile(r"\\texttt\{([^}]+)\} *& *[^&]+ *& *(\d+) *& *"
                     + r" *& *".join([cell] * 4) + r" *\\\\")
    seen = set()
    for name, n, *vals in pat.findall(tex):
        if name not in NAME2TASK:
            continue
        suite, task = NAME2TASK[name]
        seen.add(name)
        got_n = len(data[suite][("bf16", task)])
        if int(n) != got_n:
            fails.append(f"{name} 的 n：論文 {n}，CSV {got_n}")
        for cfg, v in zip(ORDER, vals):
            chk(f"{name}/{cfg}", v, mean(suite, cfg, task))
    # ── 附錄表的大海撈針一塊（列名不是 \texttt{}，另外比對）──
    ndl = defaultdict(lambda: [0, 0])
    for f, tag in (("needle_ctx_sweep.csv", None), ("needle_fair_32k.csv", 32245)):
        for r in csv.DictReader((RES / f).open()):
            k = (tag or int(r["prompt_tokens"]), r["config"])
            ndl[k][0] += r["correct"] == "True"
            ndl[k][1] += 1
    lens = sorted({k[0] for k in ndl})
    for tok in lens:
        lbl = f"{round(tok / 1000)}K"
        m = re.search(r"^%s +& *單鍵檢索 *& *(\d+) *& *" % lbl
                      + r" *& *".join([cell] * 4) + r" *\\\\", tex, re.M)
        if not m:
            fails.append(f"附錄表找不到大海撈針 {lbl} 那一列")
            continue
        n_claim, *vals = m.groups()
        if int(n_claim) != ndl[(tok, "bf16")][1]:
            fails.append(f"大海撈針 {lbl} 的 n：論文 {n_claim}，CSV {ndl[(tok, 'bf16')][1]}")
        for cfg, v in zip(ORDER, vals):
            ok_, n_ = ndl[(tok, cfg)]
            chk(f"大海撈針 {lbl}/{cfg}", v, 100 * ok_ / n_)

    if missing := set(NAME2TASK) - seen:
        fails.append(f"附錄表沒有這些任務的列：{sorted(missing)}")
    print(f"   覆蓋 {len(seen)}/{len(NAME2TASK)} 個任務")

    # ── 主表 tab:eps-task 的巨觀平均與全距 ──
    print("\n── 主表 tab:eps-task ──")
    tasks = {s: data[s]["__order__"] for s in data}

    def macro(suite: str, cfg: str) -> float:
        return sum(mean(suite, cfg, t) for t in tasks[suite]) / len(tasks[suite])

    row = re.compile(r"^(BF16|FP8|INT8|INT4) *& *[\d.]+\$\\times\$ *& *[\d.]+ *& *"
                     + r" *& *".join([cell] * 3) + r" *\\\\", re.M)
    rows = {m.group(1).lower(): m.groups()[1:] for m in row.finditer(tex)}
    if set(rows) != set(ORDER):
        fails.append(f"主表的列不齊：找到 {sorted(rows)}")
    for cfg in ORDER:
        if cfg not in rows:
            continue
        _, lb_c, ru_c = rows[cfg]
        chk(f"主表 LongBench 巨觀/{cfg}", lb_c, macro("longbench", cfg))
        chk(f"主表 RULER 巨觀/{cfg}", ru_c, macro("ruler", cfg))
    span = re.search(r"全距（pp） *& *-*-* *& *([\d.]+) *& *([\d.]+) *& *([\d.]+) *& *([\d.]+)", tex)
    if not span:
        fails.append("主表找不到「全距」那一列")
    else:
        for i, suite in ((2, "longbench"), (3, "ruler")):
            vals = [macro(suite, c) for c in ORDER]
            chk(f"主表 {suite} 全距", span.group(i + 1), max(vals) - min(vals))

    # ── caption 裡的「FP8 於這 14 個任務上最高只到 X、INT4 最高 Y」 ──
    print("\n── 附錄表題的極值句 ──")
    cap = re.search(r"FP8 於這 14 個任務上最高只到 ([\d.]+)、INT4 最高 ([\d.]+)", tex)
    if not cap:
        fails.append("附錄表題找不到 FP8/INT4 的極值句")
    else:
        for i, cfg in ((1, "fp8"), (2, "int4")):
            worst = max(mean(s, cfg, t) for s in data for t in tasks[s])
            chk(f"表題 {cfg} 最高", cap.group(i), worst)

    # ── 正文引用的配對 bootstrap 區間 ──
    print("\n── 正文的配對 bootstrap 區間（seed 固定，可逐字比對）──")
    def paired(suite: str, cfg: str, task: str | None):
        ts = [task] if task else tasks[suite]
        idx = [(t, i) for t in ts for i in sorted(data[suite][("bf16", t)])]
        return paired_ci([data[suite][("bf16", t)][i] for t, i in idx],
                         [data[suite][(cfg, t)][i] for t, i in idx])

    QUOTED = [   # (說明, suite, cfg, task, 論文寫的 mean/lo/hi)
        ("LongBench FP8 合併", "longbench", "fp8", None, ("51.8", "47.3", "56.3")),
        ("LongBench INT4 合併", "longbench", "int4", None, ("58.0", "53.7", "62.3")),
        ("LongBench INT8 合併", "longbench", "int8", None, ("1.3", "0.0", "2.8")),
        ("RULER INT8 合併", "ruler", "int8", None, ("12.5", "8.6", "16.7")),
        ("RULER INT8 multikey-3", "ruler", "int8", "niah_multikey_3", ("46.7", "30.0", "63.3")),
        ("RULER INT8 CWE", "ruler", "int8", "cwe", ("12.0", "4.7", "20.3")),
        ("RULER INT8 VT", "ruler", "int8", "vt", ("6.7", "-2.0", "16.0")),
        ("RULER INT8 FWE", "ruler", "int8", "fwe", ("-2.2", "-6.7", "2.2")),
        ("RULER INT8 multivalue", "ruler", "int8", "niah_multivalue", ("6.7", "2.5", "11.7")),
    ]
    for label, suite, cfg, task, (m, lo, hi) in QUOTED:
        gm, glo, ghi = paired(suite, cfg, task)
        for what, claimed, actual in (("mean", m, gm), ("lo", lo, glo), ("hi", hi, ghi)):
            chk(f"{label} {what}", claimed, actual)

    print()
    if fails:
        print(f"🔴 {len(fails)} 處不符：")
        for f in fails:
            print(f"   * {f}")
        return 1
    print("✅ 論文中 M5-(c) 的每個數字都與 results/ 的 CSV 相符")
    return 0


if __name__ == "__main__":
    sys.exit(main())
