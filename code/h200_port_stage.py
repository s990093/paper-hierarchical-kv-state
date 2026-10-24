#!/usr/bin/env python3
"""h200_port_stage.py — 把 MI300X 的佇列 stage 轉成 H200 版（13_h200_repro.md §12）。

來源：code/m10_queue_mi300x/queue/<stage>.sh，或 m10_restore2.py plan-44d 寫出的 44d_rs_longctx.sh。
只換：lib.sh 的位置、repo 與原始資料的路徑、results 的平台目錄（*_mi300x → *_h200）、run 名稱（m10- → m10h-）、
vLLM 的 python（MI300X 是另一個 venv 加 ROCm PATH；H200 是同一個 conda 環境，Qwen 用拿掉 DCA 的變體，13 §11.1）。
指令、題數、設定一律不動；43d 讀的 43 腳本換成 H200 的（H200 自己的選手）；例外只有 43b 的背景強度掃描（13_h200_repro §12：busy1、busy2 用 H200 主線的 duty，busy3 不跑）。
轉完檢查：非註解行不能再有 /mlsteam、_mi300x、/opt/rocm、「m7run m10-」，source 必須指到 H200 的 lib.sh。

  python code/h200_port_stage.py SRC DST --repo R --jobs J --raw-root T --nodca N
"""
import argparse
import os
import re
import sys

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--repo", required=True)
ap.add_argument("--jobs", required=True)
ap.add_argument("--raw-root", required=True, help="大的逐列檔放哪裡（MI300X 的 /mlsteam/data/tiara）")
ap.add_argument("--nodca", required=True, help="拿掉 DCA 的 Qwen 目錄（vLLM 用）")
ap.add_argument("--confirm43", required=True, help="H200 的 43 腳本（t2 的 CONFIRM-PLAN 寫的；43d 從它讀 43 的速度配置＝H200 自己的選手，13 §9）")
a = ap.parse_args()

s = open(a.src).read()
s = re.sub(r"^source \S*/lib\.sh$", f"source {a.jobs}/lib.sh", s, flags=re.M)
for old, new in [
    ("env PATH=/opt/rocm/bin:$PATH /mlsteam/workspace/venv/tiara-v028/bin/python", f"env M7_QWEN_GLOB={a.nodca} python"),
    ("/mlsteam/workspace/paper-hierarchical-kv-state", a.repo),
    ("/mlsteam/data/tiara/drop_m10", os.path.join(a.raw_root, "drop_m10h")),
    ("results/m10_restore2_mi300x", "results/m10_restore2_h200"),
    ("results/m10_match_mi300x", "results/m10_match_h200"),
    ("results/m10_drop_mi300x", "results/m10_drop_h200"),
    ("results/m10_xgpu_mi300x", "results/m10_xgpu_h200"),
    ("m7run m10-", "m7run m10h-"),
    ("$REPO/code/m10_queue_mi300x/queue/43_sweep_confirm.sh", a.confirm43),
]:
    s = s.replace(old, new)

if os.path.basename(a.src) == "43b_rs2_busy.sh":
    old_bl = re.search(r"echo '(\{\"idle\": 0\.0, \"busy1\": 0\.4, \"busy2\": 0\.7, \"busy3\": 0\.9, [^']*\})' > \$BL", s)
    if not old_bl:
        sys.exit("43b：找不到 MI300X 的背景強度表（busy_levels_q4.json 的 echo）")
    h200 = ('{"idle": 0.0, "busy1": 0.6, "busy2": 1.0, "_note": "H200（13_h200_repro §12）：busy1、busy2＝H200 主線'
            '（results/m7_write_policy_h200/busy_levels.json，duty 0.6／1.0）；MI300X 的 busy3（duty 0.9）不跑：H200 的 busy2 已經是 duty 1.0"}')
    s = s.replace(old_bl.group(1), h200)
    if s.count("--gpu-states busy1 busy2 busy3") != 1:
        sys.exit("43b：找不到 --gpu-states busy1 busy2 busy3")
    s = s.replace("--gpu-states busy1 busy2 busy3", "--gpu-states busy1 busy2")

bad = [f"{i}: {ln}" for i, ln in enumerate(s.splitlines(), 1)
       if not ln.lstrip().startswith("#") and re.search(r"/mlsteam|_mi300x|/opt/rocm|m7run m10-", ln)]
src_ok = re.search(rf"^source {re.escape(a.jobs)}/lib\.sh$", s, flags=re.M)
if bad or not src_ok:
    sys.exit(f"{a.src} 轉換後還有 MI300X 的東西（或 source 不對）：\n" + "\n".join(bad[:20]) + ("" if src_ok else "\n（沒有 source H200 的 lib.sh）"))
open(a.dst, "w").write(s)
print(f"[port] {os.path.basename(a.src)} → {a.dst}")
