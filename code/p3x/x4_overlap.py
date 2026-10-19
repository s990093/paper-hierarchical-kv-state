"""x4_overlap.py — 驗算 X5 的說法：conversation 的請求有多少在 toolagent 有相同的（timestamp, input_length, output_length）。"""
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

D = "/ssd8/hungwei/paper-hkv/datasets/mooncake/traces"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "p3_explore", "x4", "x4_overlap.csv")
key = lambda r: (r["timestamp"], r["input_length"], r["output_length"])  # noqa: E731
conv = [json.loads(x) for x in open(f"{D}/conversation_trace.jsonl")]
tool = [json.loads(x) for x in open(f"{D}/toolagent_trace.jsonl")]
tk = Counter(key(r) for r in tool)
m = sum(1 for r in conv if tk[key(r)] > 0)
# 同一把 key 在 toolagent 裡出現的那些請求，hash_ids 長度是否也相同（交叉驗算）
tl = {}
for r in tool:
    tl.setdefault(key(r), []).append(len(r["hash_ids"]))
mlen = sum(1 for r in conv if len(r["hash_ids"]) in tl.get(key(r), []))
rid = os.environ.get("RUN_ID", "adhoc")
ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
with open(OUT, "w") as f:
    f.write("run_id,ts,n_conv,n_tool,n_conv_with_same_ts_in_out_in_tool,frac,n_conv_same_key_and_same_n_blocks\n")
    f.write(f"{rid},{ts},{len(conv)},{len(tool)},{m},{m / len(conv):.4f},{mlen}\n")
print(f"conversation {len(conv)}，在 toolagent 有相同（timestamp, input, output）的 {m}（{m / len(conv):.1%}）；"
      f"其中 block 數也相同的 {mlen}", flush=True)
sys.exit(0)
