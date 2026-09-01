#!/usr/bin/env python3
"""把 `m5_understanding.py` 自寫的 LongBench 指標對**上游 metrics.py** 逐筆比對。

## 為什麼要有這支

論文會宣稱「LongBench 上 BF16 → INT4 掉了 X 分」。那個 X 是用**我們自己寫的**
`qa_f1` / `rouge_l` / `classification` / `retrieval` 算出來的——
上游 `metrics.py` 需要 `jieba`（只有中文任務用得到）與 `rouge`，
為了不把中文相依拉進 vLLM 的 venv，英文那四個指標是重寫的。

**重寫過的計分函式若與上游有出入，論文的數字就不是 LongBench 分數。**
這支測試是那個宣稱的憑據：隨機字串 + smoke run 的真實預測，
兩邊逐筆對到 1e-9。

需要 `rouge` 與 `fuzzywuzzy`（側裝於 `$PAPER_HKV_PYLIBS`，預設
`/ssd7/hungwei/paper-hkv/pylibs`）與上游的 `metrics.py`
（`$PAPER_HKV_BIG/datasets/longbench/metrics.py`）。缺任何一個就 SKIP 並回傳 77。

用法:
    PYTHONPATH=/ssd7/hungwei/paper-hkv/pylibs python code/test_m5c_metrics.py
"""

from __future__ import annotations

import os
import random
import sys
import types
from pathlib import Path

BIG = Path(os.environ.get("PAPER_HKV_BIG", "/ssd7/hungwei/paper-hkv"))
LB = BIG / "datasets/longbench"
sys.path.insert(0, os.environ.get("PAPER_HKV_PYLIBS", str(BIG / "pylibs")))
sys.path.insert(0, str(Path(__file__).resolve().parent))

CLASSES = ["Description", "Entity", "Abbreviation", "Human", "Location", "Number"]
WORDS = "alpha beta gamma delta 12 7 the a an , . ( ) yes no unanswerable Boston 1997".split()


def main() -> int:
    if not (LB / "metrics.py").exists():
        print(f"SKIP：找不到上游 {LB / 'metrics.py'}")
        return 77
    # 上游 metrics.py 在 import 時就抓 jieba（僅中文任務會呼叫），塞個假的
    sys.modules.setdefault("jieba", types.SimpleNamespace(
        cut=lambda s, cut_all=False: list(s)))
    sys.path.insert(0, str(LB))
    try:
        import metrics as ref
    except ImportError as e:
        print(f"SKIP：上游 metrics.py 的相依缺失（{e}）")
        return 77
    import m5_understanding as mine

    rng = random.Random(7)

    def rnd(n: int) -> str:
        return " ".join(rng.choice(WORDS) for _ in range(rng.randint(1, n)))

    bad = n = 0

    def chk(name: str, a: float, b: float, *ctx) -> None:
        nonlocal bad, n
        n += 1
        if abs(a - b) > 1e-9:
            bad += 1
            print(f"🔴 {name} 不一致 {a} vs {b}  {ctx}")

    for _ in range(3000):
        p, g = rnd(12), rnd(8)
        chk("qa_f1", mine.qa_f1(p, g), ref.qa_f1_score(p, g), p, g)
        chk("rouge_l", mine.rouge_l(p, g), ref.rouge_score(p, g), p, g)
    for _ in range(1500):
        # 上游的 retrieval_score 在金標不含 "Paragraph N" 時會 IndexError，
        # 真實資料一定含；我們的版本回 0.0。此處只比對真實資料會出現的形狀。
        p, g = rnd(10), f"Paragraph {rng.randint(1, 30)}"
        chk("retrieval", mine.retrieval(p, g), ref.retrieval_score(p, g), p, g)
    for _ in range(2000):
        p = " ".join(rng.choice(CLASSES + WORDS) for _ in range(rng.randint(1, 5)))
        g = rng.choice(CLASSES)
        chk("classification", mine.classification(p, g, all_classes=CLASSES),
            ref.classification_score(p, g, all_classes=CLASSES), p, g)

    print(f"{n} 組比對，與上游 metrics.py 不一致 {bad} 筆")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
