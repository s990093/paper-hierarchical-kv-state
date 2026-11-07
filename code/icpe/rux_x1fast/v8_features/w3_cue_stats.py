"""W3 線索 (a) 在 TraceLab 原始紀錄上的統計（只讀、不模擬）：cue＝這一輪的輸出有工具呼叫（`tools` 非空）對「下一輪由工具結果觸發」（first_input_event_type＝tool_result）。
用法：python -I w3_cue_stats.py <syfi_coding_trace.jsonl.gz>"""
import gzip, json, sys, collections
f = sys.argv[1]
S = collections.defaultdict(list)
fe = collections.Counter()
for ln in gzip.open(f, "rt"):
    r = json.loads(ln)
    S[r["session_id"]].append((r["round_index"], len(r.get("tools") or []), r.get("first_input_event_type"),
                               r["timing_events"][0]["timestamp"] if r.get("timing_events") else None))
    fe[r.get("first_input_event_type")] += 1
n = agree = cue = nxt_tool = last = 0
cm = collections.Counter()
for sid, rs in S.items():
    rs.sort()
    for i, (ri, nt, fet, ts) in enumerate(rs):
        n += 1
        if i + 1 == len(rs):
            last += 1
            continue
        c = nt > 0
        t = rs[i + 1][2] == "tool_result"
        cm[(c, t)] += 1
print("rounds", n, "sessions", len(S), "last_rounds", last)
print("first_input_event_type", dict(fe.most_common(8)))
print("cue(tools in output) x next_is_tool_result:", {f"cue={k[0]},next_tool={k[1]}": v for k, v in sorted(cm.items())})
