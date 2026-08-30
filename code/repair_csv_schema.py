#!/usr/bin/env python3
"""修復 baseline.csv 的欄位錯位，並把所有列統一到目前的 schema。

## 出了什麼事

`write_csv()` 只在**檔案不存在**時寫 header：

```python
new = not p.exists()
w = csv.DictWriter(f, fieldnames=list(rows[0]))
if new: w.writeheader()
w.writerows(rows)
```

問題在於 `fieldnames` 取自**當下**的 row dict。實驗跑到一半我往 row dict 加了
`caveat` 與 `concurrency_mode` 兩個欄位，於是後來的列變成 22 / 23 個值，
而檔案開頭的 header 仍然只有 21 欄。

**結果是 608 列的欄位全部往右錯位**，而且 `csv.DictReader` 讀起來完全沒有報錯——
它只是把多出來的值塞進 `None` 這個 key。`caveat` 的內容跑到 `quality_score` 欄，
`concurrency_mode` 跑到 `quality_metric` 欄。

這正是 `EXPERIMENT_PLAN.md` §0 禁令 1 要防的東西：**看起來完全正常的錯誤數字**。

## 為什麼還救得回來

Python 的 dict 保持插入順序，而 row dict 的建構順序在程式碼裡是寫死的。
所以每一列的**值的順序**是正確的，錯的只有 header 的欄位數。
三個 schema 版本可以用「這一列有幾個值」直接分辨：

| 欄數 | 什麼時候寫的 | 缺哪些欄 |
|---|---|---|
| 21 | 最早 | `caveat`, `concurrency_mode` |
| 22 | 加了 caveat 之後 | `concurrency_mode` |
| 23 | 現行 | 無 |

21 / 22 欄的列缺 `concurrency_mode`，但那些 run 全部來自 `--all`（平行），
所以補 `parallel` 是事實而非猜測；另加 `mode_source` 欄標明它是補的還是原生的，
**不要讓補值看起來跟量測值一樣**。

用法:
    python code/repair_csv_schema.py                 # dry-run
    python code/repair_csv_schema.py --apply         # 寫回（會先備份）
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CSV = REPO / "results/m3_baseline/baseline.csv"

# 各 schema 版本的欄位順序，與 m3_baseline.py 裡 row dict 的建構順序一一對應。
HEAD = ["run_id", "ts", "baseline", "model_key", "model", "gpu", "ctx",
        "actual_prompt_tokens", "round", "prefix_idx", "ttft_ms", "tpot_ms",
        "total_ms", "gen_tokens", "gpu_kv_cache_tokens", "n_prefixes"]
TAIL = ["quality_score", "quality_metric", "contaminated", "guard_verdict", "log"]

SCHEMAS = {
    21: HEAD + TAIL,
    22: HEAD + ["caveat"] + TAIL,
    23: HEAD + ["caveat", "concurrency_mode"] + TAIL,
}
CANONICAL = SCHEMAS[23] + ["mode_source"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(CSV))
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    p = Path(a.csv)
    raw = list(csv.reader(p.open()))
    if not raw:
        print("empty")
        return 1
    header, data = raw[0], raw[1:]
    print(f"header 有 {len(header)} 欄；資料 {len(data)} 列")
    print("每列欄位數分布:", dict(Counter(len(r) for r in data)))

    out, bad = [], 0
    for r in data:
        sch = SCHEMAS.get(len(r))
        if sch is None:
            bad += 1
            continue
        d = dict(zip(sch, r))
        d.setdefault("caveat", "")
        if "concurrency_mode" in d:
            d["mode_source"] = "recorded"
        else:
            # 這些 run 全部來自 --all。是已知事實，不是猜測，但仍標明來源。
            d["concurrency_mode"] = "parallel"
            d["mode_source"] = "inferred_from_launch"
        out.append({k: d.get(k, "") for k in CANONICAL})

    if bad:
        print(f"🔴 {bad} 列的欄位數不屬於任何已知 schema，未處理")

    print(f"\n修好之後:")
    print("  concurrency_mode:", dict(Counter(r["concurrency_mode"] for r in out)))
    print("  mode_source     :", dict(Counter(r["mode_source"] for r in out)))
    print("  model × mode    :", dict(Counter(
        (r["model_key"], r["concurrency_mode"]) for r in out)))
    print("  quality_score   :", dict(Counter(r["quality_score"] for r in out)))

    # 抽驗：ttft_ms 必須全部是數字，否則代表還是錯位
    nonnum = [r for r in out if r["ttft_ms"] and not r["ttft_ms"]
              .replace(".", "", 1).replace("-", "", 1).isdigit()]
    print(f"  ttft_ms 非數字   : {len(nonnum)} 列 {'✅' if not nonnum else '🔴 仍然錯位'}")

    if not a.apply:
        print("\n(dry-run。加 --apply 才會寫回)")
        return 0

    bak = p.with_suffix(".csv.bak-misaligned")
    shutil.copy2(p, bak)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CANONICAL)
        w.writeheader()
        w.writerows(out)
    print(f"\n原檔備份到 {bak}")
    print(f"寫回 {p}：{len(out)} 列 × {len(CANONICAL)} 欄")
    return 0


if __name__ == "__main__":
    sys.exit(main())
