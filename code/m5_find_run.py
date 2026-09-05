#!/usr/bin/env python3
"""挑出符合條件的訓練 run 目錄。給批次腳本用。

`m5_policy_sim.py --train-run` 需要一個明確的目錄，而同一個工作負載會有
十幾個 run（不同視窗／容量／特徵族／標籤模式），只靠檔名分不出來。
這支從每個 run 的 `train_meta.json` 比對欄位，找到唯一一個就印出路徑。

    python code/m5_find_run.py trace=lc128kz window_accesses=102702 \
        label_mode=uncensored num_leaves=63
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

BIG = Path(os.environ.get("PAPER_HKV_BIG", "/ssd7/hungwei/paper-hkv"))


def main() -> int:
    want = dict(kv.split("=", 1) for kv in sys.argv[1:])
    hits = []
    for d in sorted((BIG / "runs").glob("*-m5p-train-*")):
        f = d / "train_meta.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        if all(str(m.get(k, "")) == v for k, v in want.items()):
            hits.append(d)
    if not hits:
        print(f"🔴 沒有 run 符合 {want}", file=sys.stderr)
        return 1
    print(hits[-1])          # 同條件下取最新的一個
    if len(hits) > 1:
        print(f"   （{len(hits)} 個符合，取最新）", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
