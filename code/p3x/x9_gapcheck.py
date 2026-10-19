"""x9_gapcheck.py — X9 T3 的事後一致性檢查（只報告，不改判定；看到 T3 結果之後才加，見 X9 文件 §6）。

問：WildChat 的回來間隔，在「1 小時內回來」這一段，和生產 trace（Bailian，2 小時視窗）長得像不像？
比的是條件分布 P(g ≤ x | g ≤ 1 h) 的分位數。兩邊的 g 定義都含生成時間：
  WildChat：assistant 訊息時間戳到下一個 assistant 訊息時間戳（x9_window.py 存的 t3_gaps_s.npy）
  Bailian ：父請求 timestamp 到子請求 timestamp（到達到到達）

用法： python -I code/p3x/x9_gapcheck.py --wildchat-gaps <t3_gaps_s.npy> --out results/p3_explore/x9
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone

import numpy as np

DS = "/ssd8/hungwei/paper-hkv/datasets/qwen-bailian-usagetraces-anon"
TR = {"bailian_traceA": "qwen_traceA_blksz_16.jsonl", "bailian_thinking": "qwen_thinking_blksz_16.jsonl",
      "bailian_coder": "qwen_coder_blksz_16.jsonl"}
RUN_ID = os.environ.get("RUN_ID", "")
QS = [0.1, 0.25, 0.5, 0.75, 0.9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wildchat-gaps", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert RUN_ID
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = []
    g = np.clip(np.load(a.wildchat_gaps), 0, None)
    sets = {"wildchat-1m": g}
    for name, fn in TR.items():
        rows = [json.loads(l) for l in open(os.path.join(DS, fn))]
        by = {r["chat_id"]: r for r in rows}
        gg = np.array([r["timestamp"] - by[r["parent_chat_id"]]["timestamp"] for r in rows
                       if r["parent_chat_id"] != -1 and r["parent_chat_id"] in by])
        assert (gg >= 0).mean() >= 0.99
        sets[name] = gg
    for name, x in sets.items():
        x1 = x[x <= 3600]
        row = dict(run_id=RUN_ID, ts=now, dataset=name, n_gaps=int(len(x)), n_le_1h=int(len(x1)),
                   **{f"q{int(q*100)}_given_le1h_s": float(np.quantile(x1, q)) for q in QS},
                   frac_le_60s_given_le1h=float(np.mean(x1 <= 60)), frac_le_600s_given_le1h=float(np.mean(x1 <= 600)))
        out.append(row)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "t3_gap_consistency.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
