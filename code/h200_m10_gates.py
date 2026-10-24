#!/usr/bin/env python3
"""H200 第二階段復現的停止條件（docs/precision_20261013/13_h200_repro.md §3）。只用 CPU，不改任何數字。

  python code/h200_m10_gates.py code              # §3-2：第二階段主線的程式和 MI300X（origin/main bcfe5b1）逐字相同
  python code/h200_m10_gates.py params            # 07 §2：CANDIDATE_PARAMS.json 的 sha256
  python code/h200_m10_gates.py e0  <out.json> <tune 的逐題 CSV...>     # §3-3：E0 難度 60–95%
  python code/h200_m10_gates.py fmt <h200 fmt_choice.json> <mi300x fmt_choice.json> <out.json>   # §3-4

條件不成立 → 印出原因、exit 1（呼叫端停下、回報）。
"""
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AMD_REF = "bcfe5b1"                     # 13 §1：MI300X 第二階段主線的程式（origin/main，2026-10-16）
MAINLINE = ["code/m10_codec.py", "code/m10_codec_x.py", "code/m10_quality.py", "code/m10_speed.py",
            "code/m10_analyze.py", "code/m10_fmt_select.py", "code/m10_kv_stats.py", "code/m10_longbench.py",
            "code/m10_verify.sh", "code/m10_verify_summary.py", "code/m7_restore_harness.py", "code/m7_model.py",
            "code/m7_busy.py", "code/gpu_guard.py", "code/test_m10_restore.py", "code/test_m7_correctness.py"]
TASKS_FILE = "code/m10_tasks.py"
# git diff bcfe5b1 944497e -- code/m10_tasks.py 的 sha256：只有 MODELS 讀 M7_LLAMA_GLOB／M7_QWEN_GLOB（commit 2955b35）
TASKS_DIFF_SHA = "01b8d52df472b1d25f8af742650acd22d9d6746f5618e184ac2f6ed52becdafb"
PARAMS = "results/m10_pilot_mi300x/e0_tune/CANDIDATE_PARAMS.json"
PARAMS_SHA = "00c1034faf038b6f41c21d141b4540a6c5d6e3635bd3f52e5210d3a1074412e2"   # 07 §2：00c1034f…74412e2
MAIN_TASKS = ["T2", "T3", "T4", "T5"]
E0_RANGE = (0.60, 0.95)                 # 03 §3.2、§7 D1


def git(*args) -> str:
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, check=True).stdout


def gate_code():
    bad = []
    d = git("diff", "--stat", AMD_REF, "--", *MAINLINE)          # 工作目錄 vs MI300X 的 commit（含沒 commit 的改動）
    if d.strip():
        bad.append("主線程式和 " + AMD_REF + " 不同：\n" + d)
    td = git("diff", AMD_REF, "--", TASKS_FILE)
    sha = hashlib.sha256(td.encode()).hexdigest()
    if sha != TASKS_DIFF_SHA:
        bad.append(f"{TASKS_FILE} 和 {AMD_REF} 的差不是預期的那段（模型路徑）：sha256 {sha}\n{td}")
    head = git("rev-parse", "--short", "HEAD").strip()
    if bad:
        print("[gate code] FAIL（13 §3-2）\n" + "\n".join(bad))
        return 1
    print(f"[gate code] PASS：HEAD {head} 的第二階段主線＝{AMD_REF}（{len(MAINLINE)} 個檔逐字相同；{TASKS_FILE} 只差模型路徑）")
    return 0


def gate_params():
    sha = hashlib.sha256(open(os.path.join(REPO, PARAMS), "rb").read()).hexdigest()
    if sha != PARAMS_SHA:
        print(f"[gate params] FAIL：{PARAMS} sha256 {sha} ≠ 07 §2 的 {PARAMS_SHA}")
        return 1
    print(f"[gate params] PASS：{PARAMS} sha256 {sha[:8]}…{sha[-7:]}（07 §2）")
    return 0


def gate_e0(out, csvs):
    """每個（模型, 任務）把兩輪合併算答對率。T1–T4 用 score_64（生成上限 64），T5 用 score_up（上限 120，GEN_LEN）。"""
    acc = defaultdict(list)
    cells = defaultdict(list)
    for p in csvs:
        for r in csv.DictReader(open(p, newline="")):
            col = "score_up" if r["task"] == "T5" else "score_64"
            v = float(r[col])
            acc[(r["model"], r["task"])].append(v)
            cells[(r["model"], r["task"], r["band"], r["turn"])].append(v)
    res = {"rule": "13 §3-3：主判準任務（T2–T5）E0 合併答對率在 60–95%；T1 只報告；T1–T4 用 score_64、T5 用 score_up",
           "range": E0_RANGE, "tasks": {}, "cells": {}, "fail": []}
    for (m, t), v in sorted(acc.items()):
        a = sum(v) / len(v)
        res["tasks"][f"{m}/{t}"] = {"n_rows": len(v), "acc": round(a, 4)}
        if t in MAIN_TASKS and not (E0_RANGE[0] <= a <= E0_RANGE[1]):
            res["fail"].append(f"{m}/{t} {a:.3f}")
    for (m, t, b, tu), v in sorted(cells.items()):
        res["cells"][f"{m}/{t}/{b}/turn{tu}"] = {"n": len(v), "acc": round(sum(v) / len(v), 4)}
    for m in ("llama", "qwen"):
        for t in MAIN_TASKS:
            if f"{m}/{t}" not in res["tasks"]:
                res["fail"].append(f"{m}/{t} 沒有資料")
    res["pass"] = not res["fail"]
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    for k, v in res["tasks"].items():
        print(f"[gate e0] {k:10s} n={v['n_rows']:3d} acc={v['acc']:.3f}")
    if res["fail"]:
        print("[gate e0] FAIL（13 §3-3）：" + "；".join(res["fail"]))
        return 1
    print("[gate e0] PASS")
    return 0


def gate_fmt(h200, mi300x, out):
    a = json.load(open(h200))["choice"]
    b = json.load(open(mi300x))["choice"]
    diff = [f"{m}/{t}: H200 {a[m][t]} vs MI300X {b[m][t]}" for m in ("llama", "qwen") for t in ("fp8", "int4")
            if a.get(m, {}).get(t) != b[m][t]]
    json.dump({"rule": "13 §3-4：H200 照 07 §3 重選；任一個和 MI300X 不同就停", "h200": a, "mi300x": b,
               "same": not diff, "diff": diff}, open(out, "w"), ensure_ascii=False, indent=1)
    if diff:
        print("[gate fmt] FAIL（13 §3-4，問使用者）：" + "；".join(diff))
        return 1
    print(f"[gate fmt] PASS：和 MI300X 相同 {a}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "code":
        sys.exit(gate_code())
    if cmd == "params":
        sys.exit(gate_params())
    if cmd == "e0" and len(sys.argv) >= 4:
        sys.exit(gate_e0(sys.argv[2], sys.argv[3:]))
    if cmd == "fmt" and len(sys.argv) == 5:
        sys.exit(gate_fmt(*sys.argv[2:5]))
    print(__doc__)
    sys.exit(2)
