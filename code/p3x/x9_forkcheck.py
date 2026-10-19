"""x9_forkcheck.py — X9 T1／T2 的事後診斷（只報告，不改判定；看到 T1 分岔率＝0、T2 共用占比之後才加，見 X9 文件 §6）。

問：Bailian 的 parent_chat_id 裡完全沒有兄弟（分岔率 0）。分岔（重新生成、編輯）會不會被記成「新的根 session」？
做法：對每個回來請求 c（有父），算
  L_own   ＝和自己父的共同前綴（block）
  L_other ＝和「另一個根 session 裡、比 c 早到的任何請求」的最長共同前綴（用逐 block 的前綴雜湊鏈）
報告 L_other ≥ L_own 的比例（別的 session 有一樣長、甚至更長的同一段歷史），並分 L_own 的長度。
另報：根請求（turn 1）中，和更早的另一個根共用 ≥64 block（1,024 token）前綴的比例。

用法： python -I code/p3x/x9_forkcheck.py --out results/p3_explore/x9
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone

DS = os.path.join(os.environ.get("TIARA_DATASETS", "/ssd8/hungwei/paper-hkv/datasets"), "qwen-bailian-usagetraces-anon")
TR = {"bailian_traceA": "qwen_traceA_blksz_16.jsonl", "bailian_thinking": "qwen_thinking_blksz_16.jsonl",
      "bailian_coder": "qwen_coder_blksz_16.jsonl", "bailian_traceB": "qwen_traceB_blksz_16.jsonl"}
RUN_ID = os.environ.get("RUN_ID", "")


def lcp(a, b):
    m = min(len(a), len(b))
    i = 0
    while i < m and a[i] == b[i]:
        i += 1
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert RUN_ID
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = []
    for name, fn in TR.items():
        rows = [json.loads(l) for l in open(os.path.join(DS, fn))]
        rows.sort(key=lambda r: (r["timestamp"], r["chat_id"]))
        by = {r["chat_id"]: r for r in rows}
        root = {}
        for r in rows:  # 依時間排序：父一定比子早（census 斷言過 ts 順序）
            p = r["parent_chat_id"]
            root[r["chat_id"]] = root.get(p, p) if (p != -1 and p in by) else r["chat_id"]
        # 前綴雜湊鏈 → 第一次出現的根（以及是否已有第二個根）
        seen = {}  # key -> (第一個根, 是否已被兩個以上的根用過)
        n_ret = n_other_ge_own = n_other_ge_own_long = n_long = 0
        n_root = n_root_dup64 = 0
        for r in rows:
            h = r["hash_ids"]
            rt = root[r["chat_id"]]
            # 查：和別的根的最長共同前綴
            key = 0
            l_other = 0
            keys = []
            for i, x in enumerate(h):
                key = hash((key, x))
                keys.append(key)
                s = seen.get(key)
                if s is None:
                    break
                if s[0] != rt or s[1]:  # 第一個根不是自己，或已有兩個以上的根 → 一定有別的根用過
                    l_other = i + 1
            # 繼續算剩下的 key（用於登記）
            for x in h[len(keys):]:
                key = hash((key, x))
                keys.append(key)
            p = r["parent_chat_id"]
            if p != -1 and p in by:
                n_ret += 1
                l_own = lcp(by[p]["hash_ids"], h)
                if l_other >= l_own and l_own > 0:
                    n_other_ge_own += 1
                if l_own >= 64:
                    n_long += 1
                    if l_other >= l_own:
                        n_other_ge_own_long += 1
            else:
                n_root += 1
                if l_other >= 64:
                    n_root_dup64 += 1
            for k in keys:
                s = seen.get(k)
                if s is None:
                    seen[k] = (rt, False)
                elif s[0] != rt and not s[1]:
                    seen[k] = (s[0], True)
        out.append(dict(run_id=RUN_ID, ts=now, trace=name, n_returns=n_ret,
                        frac_returns_other_session_prefix_ge_own=(n_other_ge_own / n_ret) if n_ret else float("nan"),
                        n_returns_own_ge64=n_long,
                        frac_long_returns_other_ge_own=(n_other_ge_own_long / n_long) if n_long else float("nan"),
                        n_roots=n_root, frac_roots_share_ge64_with_earlier_root=n_root_dup64 / n_root))
        print(json.dumps(out[-1]), flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "t1t2_forkcheck.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    main()
