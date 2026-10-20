"""x8_traces.py — X8：把 agent／對照工作負載轉成同一種 session 請求表（第三階段探索 X8）

輸出（每個工作負載一個 .npz，放在 $RUN/traces/）欄位，一列＝一個請求，依到達時間排序：
  sid        session 編號（0 起）
  t_arr      到達時間（秒，trace 相對）
  t_idle     這個請求做完、KV 開始閒置的時間（秒）
  ctx_tok    這個請求的輸入 token 數（＝要有的 context）
  nold_tok   到達時可以從這個 session 上一份 KV 還原的前綴 token 數（第一個請求或 context 重設＝0）
  obj_tok    做完之後這個 session 留下的 KV token 數（輸入＋輸出）
  ret_kind   這個請求前面那段閒置是什麼：0＝session 第一個請求、1＝工具回來、2＝人／未知
  ttl_key    這個請求之後那段閒置的 Continuum 鍵（CacheWise＝工具名，'' ＝沒有工具／人類回合；其他 trace＝'*'）
  ttl_dur_s  CacheWise：這次工具呼叫的實際時長（秒，多個工具取最大）；其他＝nan（只給「前半段中位數」用）
  sess_start 這個 session 第一個請求的原始到達時間（CacheWise 切前後半用）
  shared_tok 這個請求開頭有多少 token 也出現在別的 session（模板、system prompt；block 層級快取只存一次）。
             只給 x8_sim --dedup 的敏感度用（登記之後才加）；CacheWise 沒有 hash，填 0。

規則 6（CLAUDE.md §1）：每個載入函式都有「用 A 欄驗算 B 欄」的斷言，比例寫進 meta，meta 進 x8_traces.csv。
下載的資料是不受信任的資料：只讀 JSON，不執行裡面的程式；用 python -I 跑。
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import glob
import json
import math
import os
import sys

import numpy as np

DS = os.environ.get("TIARA_DATASETS", "/ssd8/hungwei/paper-hkv/datasets")
MOON = os.path.join(DS, "mooncake", "traces")
BAIL = os.path.join(DS, "qwen-bailian-usagetraces-anon")
CW = os.path.join(DS, "cachewise-coding-traces", "repo", "parsed_traces")
BLK = 512  # 論文的 block（main.tex §2.1）


def _rid():
    return os.environ.get("RUN_ID", "adhoc")


def _ts():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _pack(name, recs, meta):
    recs.sort(key=lambda r: (r["t_arr"], r["ord"]))
    cols = ["sid", "t_arr", "t_idle", "ctx_tok", "nold_tok", "obj_tok", "ret_kind", "ttl_dur_s", "sess_start", "shared_tok"]
    out = {c: np.array([r[c] for r in recs]) for c in cols}
    out["ttl_key"] = np.array([r["ttl_key"] for r in recs], dtype=object)
    # session 編號重新從 0 排（依第一次出現）
    remap = {}
    out["sid"] = np.array([remap.setdefault(s, len(remap)) for s in out["sid"]], dtype=np.int64)
    meta["n_req"] = len(recs)
    meta["n_sessions"] = len(remap)
    ret = out["ret_kind"] > 0
    meta["n_returns"] = int(ret.sum())
    meta["n_sessions_with_return"] = int(len(np.unique(out["sid"][ret])))
    meta["ctx_tok_median"] = float(np.median(out["ctx_tok"]))
    meta["ctx_tok_p90"] = float(np.percentile(out["ctx_tok"], 90))
    meta["ctx_tok_max"] = float(out["ctx_tok"].max())
    meta["frac_ctx_gt_131072"] = float(np.mean(out["ctx_tok"] > 131072))
    gap = []
    last_idle = {}
    for s, ta, ti in zip(out["sid"], out["t_arr"], out["t_idle"]):
        if s in last_idle:
            gap.append(ta - last_idle[s])
        last_idle[s] = ti
    g = np.array(gap)
    for q in (10, 50, 90):
        meta[f"gap_s_p{q}"] = float(np.percentile(g, q)) if len(g) else float("nan")
    assert np.all(out["nold_tok"] <= out["ctx_tok"]), f"{name}: nold > ctx"
    return out, meta


# ------------------------------------------------------------------ Mooncake（前綴鏈重建）
def _strict_parents(rows):
    """R-strict（docs/phase3_20261017/research/calc/mooncake_chain2.py 的規則）：子請求最長的、等於某個較早請求
    「全部完整 block」的前綴；那個前綴最多被 3 個請求共用、候選父請求唯一、時間嚴格較早，才接。
    hash 是前綴鏈（載入時驗過 chain_bad_pos=chain_bad_prev=0），所以前綴 h[:L] 用 h[L-1] 當鍵。"""
    pop = collections.Counter()
    for r in rows:
        pop.update(set(r["hash_ids"]))
    cand = collections.defaultdict(list)
    for i, r in enumerate(rows):
        f = r["input_length"] // BLK
        if f >= 1:
            cand[r["hash_ids"][f - 1]].append(i)
    parent = [-1] * len(rows)
    amb = 0
    for j, r in enumerate(rows):
        h = r["hash_ids"]
        for L in range(len(h) - 1, 0, -1):
            ps = [i for i in cand.get(h[L - 1], []) if i < j and rows[i]["timestamp"] < r["timestamp"]]
            if ps:
                if pop[h[L - 1]] <= 3 and len(ps) == 1:
                    parent[j] = ps[0]
                elif len(ps) > 1:
                    amb += 1
                break
    return parent, amb


def load_mooncake(trace, rule="carry"):
    path = os.path.join(MOON, f"{trace}_trace.jsonl")
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    meta = dict(workload=f"mooncake_{trace}" + ("" if rule == "carry" else "_strict"), source=path, rule=rule)
    # 規則 6 (a)：hash_ids 的單位用 input_length 驗算（512 token）
    unit = sum(len(r["hash_ids"]) == math.ceil(r["input_length"] / BLK) for r in rows) / n
    meta["assert_unit512_rate"] = unit
    assert unit >= 0.95, f"{trace}: len(hash_ids)==ceil(input_length/512) 只有 {unit:.4f}"
    tss = [r["timestamp"] for r in rows]
    assert tss == sorted(tss), f"{trace}: timestamp 沒排序"
    # session 重建。第一版用 D4 的規則（父＝最近一個「最後一個完整 block（block≥1）」出現在子請求前綴裡的請求），
    # toolagent 會把 9,202 個共用 12 個 block 模板的請求併成一個 session（README §9 坑 #2）；所以改成：
    #   (1) 子請求含父請求全部完整 block（前綴鏈保證同位置）；
    #   (2) 子請求帶了父請求的輸出：child.input ≥ parent.input + parent.output − 512；
    #   (3) 每個父請求只能被接續一次（第一個符合的）；之後再符合的＝分叉，當新 session（不算回來）。
    cand = collections.defaultdict(list)   # 最後一個完整 block 的 hash → 還沒被接續的請求
    parent = [-1] * n
    continued = [False] * n
    n_fork = 0
    for ri, r in enumerate(rows):
        h = r["hash_ids"]
        hit_any = False
        for k in range(len(h) - 1, 0, -1):
            if h[k] not in cand:
                continue
            for p in reversed(cand[h[k]]):
                hit_any = True
                if continued[p]:
                    continue
                if r["input_length"] >= rows[p]["input_length"] + rows[p]["output_length"] - BLK:
                    parent[ri] = p
                    break
            if parent[ri] >= 0:
                break
        if parent[ri] >= 0:
            continued[parent[ri]] = True
        elif hit_any:
            n_fork += 1
        nfull = r["input_length"] // BLK
        if nfull >= 2:
            cand[h[nfull - 1]].append(ri)
    meta["n_prefix_hit_but_not_continuation"] = n_fork
    if rule == "strict":
        parent, amb = _strict_parents(rows)
        meta["strict_ambiguous_children"] = amb
    # 規則 6 (b)：子請求的長度要 ≥ 父請求的完整 block（由構造保證）；並驗「子 ≥ 父輸入＋父輸出 − 1 block」
    #             （下一輪會帶上一輪的輸出；不吻合＝前綴鏈把不同 session 接在一起，或 context 被截掉）
    pairs = [(ri, p) for ri, p in enumerate(parent) if p >= 0]
    grow = sum(rows[c]["input_length"] >= rows[p]["input_length"] + rows[p]["output_length"] - BLK for c, p in pairs)
    meta["n_parent_child_pairs"] = len(pairs)
    meta["assert_child_len_ge_parent_in_plus_out_rate"] = grow / max(len(pairs), 1)
    # 規則 6 (c)：timestamp 單位＝ms：間隔 ÷ 父輸出長度 < 1 ms/token（不可能的解碼速度）的比例要很小（D4：toolagent 1.1%）
    rat = [(rows[c]["timestamp"] - rows[p]["timestamp"]) / rows[p]["output_length"] for c, p in pairs
           if rows[p]["output_length"] >= 50]
    rat = np.array(rat, float)
    meta["frac_gap_lt_1ms_per_out_token"] = float(np.mean(rat < 1.0)) if len(rat) else float("nan")
    meta["assert_ts_ms_rate"] = 1 - meta["frac_gap_lt_1ms_per_out_token"]
    assert meta["assert_ts_ms_rate"] >= 0.95, f"{trace}: timestamp 單位不像 ms"
    if rule == "carry":   # R-carry 的定義就要求帶輸出；R-strict 不要求，只報告比例
        assert meta["assert_child_len_ge_parent_in_plus_out_rate"] >= 0.80, \
                f"{trace}: 只有 {meta['assert_child_len_ge_parent_in_plus_out_rate']:.3f} 的子請求帶了父請求的輸入＋輸出"
    # session id＝鏈的根；session 目前的 KV＝這個 session 最近一個請求
    sid = [0] * n
    for ri in range(n):
        sid[ri] = sid[parent[ri]] if parent[ri] >= 0 else ri
    shared = _shared_prefix([r["hash_ids"][:r["input_length"] // BLK] for r in rows], sid, BLK)
    cur = {}  # sid → 最近一個請求的 hash_ids
    recs = []
    fork_short = 0
    nret = 0
    for ri, r in enumerate(rows):
        s = sid[ri]
        h = r["hash_ids"]
        nold = 0
        kind = 0
        if s in cur and parent[ri] >= 0:
            kind = 2
            ch = cur[s]
            k = 0
            m = min(len(ch), len(h), r["input_length"] // BLK)  # 只算完整 block
            while k < m and ch[k] == h[k]:
                k += 1
            nold = k * BLK
            nret += 1
            p = parent[ri]
            fork_short += k < (rows[p]["input_length"] // BLK)
        cur[s] = h
        t = r["timestamp"] / 1000.0
        recs.append(dict(sid=s, t_arr=t, t_idle=t, ctx_tok=r["input_length"], nold_tok=min(nold, r["input_length"]),
                         obj_tok=r["input_length"] + r["output_length"], ret_kind=kind, ttl_key="*",
                         ttl_dur_s=float("nan"), sess_start=rows[s]["timestamp"] / 1000.0, ord=ri,
                         shared_tok=min(shared[ri], r["input_length"])))
    meta["frac_returns_shorter_than_parent_fullblocks(fork)"] = fork_short / max(nret, 1)
    meta["idle_start_rule"] = "t_idle=t_arr（trace 沒有完成時間）"
    _assert_session_growth(meta, recs, thr=0.95 if rule == "carry" else 0.0)  # strict：只報告比例（見 X8 §6）
    return _pack(meta["workload"], recs, meta)


def _shared_prefix(hlist, sid, blk):
    """每個請求開頭連續幾個（完整）block 的 hash 也出現在別的 session → 回傳 token 數。"""
    owner = {}
    multi = set()
    for h, s in zip(hlist, sid):
        for b in h:
            o = owner.setdefault(b, s)
            if o != s:
                multi.add(b)
    out = []
    for h in hlist:
        k = 0
        while k < len(h) and h[k] in multi:
            k += 1
        out.append(k * blk)
    return out


def _assert_session_growth(meta, recs, tol=BLK, thr=0.95):
    """規則 6：同一個 session 前後兩個請求，後一個的輸入要 ≥ 前一個留下的 KV（輸入＋輸出）− tol。
    這條驗的是「session 重建」本身（第一版只驗父請求，抓不到共用模板的錯誤合併）。"""
    last = {}
    ok = n = 0
    per = collections.Counter()
    for r in sorted(recs, key=lambda r: (r["t_arr"], r["ord"])):
        s = r["sid"]
        per[s] += 1
        if s in last and r["ret_kind"] > 0:
            n += 1
            ok += r["ctx_tok"] >= last[s] - tol
        last[s] = r["obj_tok"]
    meta["assert_session_consecutive_growth_rate"] = ok / max(n, 1)
    meta["max_requests_per_session"] = max(per.values())
    assert meta["assert_session_consecutive_growth_rate"] >= thr, \
        f"{meta['workload']}: 同 session 前後請求的長度成長只有 {ok / max(n, 1):.3f} 吻合（session 重建有問題？）"


# ------------------------------------------------------------------ Qwen-Bailian（明確 parent）
def load_bailian(fn):
    path = os.path.join(BAIL, fn)
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    B16 = 16
    meta = dict(workload="bailian_" + fn.split("_")[1], source=path)
    unit = sum(len(r["hash_ids"]) == math.ceil(r["input_length"] / B16) for r in rows) / n
    meta["assert_unit16_rate"] = unit
    assert unit >= 0.95, f"{fn}: len(hash_ids)==ceil(input_length/16) 只有 {unit:.4f}"
    tss = [r["timestamp"] for r in rows]
    assert tss == sorted(tss), f"{fn}: timestamp 沒排序"
    byid = {r["chat_id"]: i for i, r in enumerate(rows)}
    pairs = [(i, byid[r["parent_chat_id"]]) for i, r in enumerate(rows) if r["parent_chat_id"] in byid]
    meta["n_parent_child_pairs"] = len(pairs)
    if not pairs:
        meta["no_session_structure"] = True
        return None, meta
    # 規則 6：parent 欄用 hash 與長度驗算——子請求的 hash 前綴＝父請求的 hash（最後一個 block 可能因 padding 不同，README FAQ Q2）
    pref = sum(rows[c]["hash_ids"][:len(rows[p]["hash_ids"]) - 1] == rows[p]["hash_ids"][:-1] for c, p in pairs)
    grow = sum(rows[c]["input_length"] >= rows[p]["input_length"] + rows[p]["output_length"] - 4 * B16 for c, p in pairs)
    turn = sum(rows[c]["turn"] == rows[p]["turn"] + 1 for c, p in pairs)
    meta["assert_child_hash_prefix_rate"] = pref / len(pairs)
    meta["assert_child_len_ge_parent_in_plus_out_rate"] = grow / len(pairs)
    meta["assert_turn_plus1_rate"] = turn / len(pairs)
    assert meta["assert_child_hash_prefix_rate"] >= 0.95, "parent_chat_id 和 hash 前綴對不上"
    assert meta["assert_turn_plus1_rate"] >= 0.95, "turn 和 parent 對不上"
    # timestamp 單位＝秒：README 說秒；用 trace 長度驗算（README：兩小時）
    span = tss[-1] - tss[0]
    meta["ts_span_s"] = span
    assert 6000 <= span <= 7400, f"{fn}: 時間跨度 {span} 不像兩小時的秒數"
    parent = [-1] * n
    for c, p in pairs:
        parent[c] = p
    sid = [0] * n
    for i in range(n):
        sid[i] = sid[parent[i]] if parent[i] >= 0 else i
    shared = _shared_prefix([r["hash_ids"][:r["input_length"] // B16] for r in rows], sid, B16)
    cur = {}
    recs = []
    for i, r in enumerate(rows):
        s = sid[i]
        h = r["hash_ids"]
        nold, kind = 0, 0
        if s in cur and parent[i] >= 0:
            kind = 2
            ch = cur[s]
            k = 0
            m = min(len(ch), len(h))
            while k < m and ch[k] == h[k]:
                k += 1
            nold = min(k * B16, r["input_length"])
        cur[s] = h
        t = float(r["timestamp"])
        recs.append(dict(sid=s, t_arr=t, t_idle=t, ctx_tok=r["input_length"], nold_tok=nold,
                         obj_tok=r["input_length"] + r["output_length"], ret_kind=kind, ttl_key="*",
                         ttl_dur_s=float("nan"), sess_start=float(rows[s]["timestamp"]), ord=i,
                         shared_tok=min(shared[i], r["input_length"])))
    meta["idle_start_rule"] = "t_idle=t_arr（trace 沒有完成時間）"
    _assert_session_growth(meta, recs, tol=4 * B16)
    return _pack(meta["workload"], recs, meta)


# ------------------------------------------------------------------ CacheWise（原生 session，有工具時長）
def _P(s):
    return dt.datetime.fromisoformat(s).timestamp()


def load_cachewise():
    files = sorted(glob.glob(os.path.join(CW, "project_*", "session_*", "events.json")))
    meta = dict(workload="cachewise", source=CW, n_files=len(files))
    recs = []
    n_empty = 0
    tool_ok = tool_n = 0
    cr_pos = cr_match = 0
    pairs = app = 0
    lag_neg = lag_n = 0
    resets = 0
    for f in files:
        ev = json.load(open(f))
        if not ev:
            n_empty += 1
            continue
        msgs = collections.OrderedDict()
        tools = collections.defaultdict(list)
        for e in ev:
            if e["event_type"] == "llm_call":
                msgs.setdefault(e["message_id"], []).append(e)
            elif e["event_type"] == "tool_call":
                tools[e["associated_llm_call_id"]].append(e)
                if e.get("execution_time_ms") is not None and e.get("result_timestamp"):
                    tool_n += 1
                    # 規則 6 (a)：result_timestamp − timestamp ＝ execution_time_ms
                    tool_ok += abs(_P(e["result_timestamp"]) - _P(e["timestamp"]) - e["execution_time_ms"] / 1000) < 0.002
        R = []
        for mid, es in msgs.items():
            x = es[-1]
            ctx = x["input_tokens"] + x["cache_creation_input_tokens"] + x["cache_read_input_tokens"]
            tl = tools.get(mid, [])
            res = [_P(t["result_timestamp"]) for t in tl if t.get("result_timestamp")]
            dur = [t["execution_time_ms"] / 1000 for t in tl if t.get("execution_time_ms") is not None]
            names = sorted({t["tool_name"] for t in tl})
            R.append(dict(t0=_P(es[0]["timestamp"]), t1=_P(es[-1]["timestamp"]), ctx=ctx,
                          cr=x["cache_read_input_tokens"], out=x["output_tokens"],
                          res=max(res) if res else None, dur=max(dur) if dur else float("nan"),
                          names=names))
        R.sort(key=lambda r: r["t0"])
        sname = os.path.relpath(os.path.dirname(f), CW)
        s0 = R[0]["t0"]
        prev = None
        for k, r in enumerate(R):
            kind, nold, t_arr = 0, 0, r["t0"]
            if prev is not None:
                pairs += 1
                if r["cr"] > 0:
                    cr_pos += 1
                    # 規則 6 (b)：下一次的 cache_read ≈ 上一次的完整 context（append-only 的證據）
                    cr_match += abs(r["cr"] - prev["ctx"]) <= max(64, 0.02 * prev["ctx"])
                if r["ctx"] >= prev["ctx"]:
                    app += 1
                    nold = min(prev["ctx"] + prev["out"], r["ctx"])
                else:
                    resets += 1   # context 變短（壓縮、子代理切換）：當成新的 context，不還原
                    nold = 0
                if prev["res"] is not None:
                    kind = 1
                    t_arr = prev["res"]      # 工具做完，agent 立刻送下一個請求
                    lag_n += 1
                    lag_neg += (r["t0"] - prev["res"]) < -0.002
                else:
                    kind = 2                 # 人類回合：到達時間取第一個事件（高估間隔約一個 TTFT）
            ttl_key = "|".join(r["names"]) if r["names"] else ""
            recs.append(dict(sid=sname, t_arr=t_arr, t_idle=max(r["t1"], t_arr), ctx_tok=r["ctx"], nold_tok=nold,
                             obj_tok=r["ctx"] + r["out"], ret_kind=kind, ttl_key=ttl_key, ttl_dur_s=r["dur"],
                             sess_start=s0, ord=len(recs), shared_tok=0))
            prev = r
    meta["n_empty_sessions"] = n_empty
    meta["assert_tool_exec_time_rate"] = tool_ok / max(tool_n, 1)
    meta["assert_cache_read_eq_prev_ctx_rate(given cr>0)"] = cr_match / max(cr_pos, 1)
    meta["frac_pairs_append_only"] = app / max(pairs, 1)
    meta["frac_pairs_context_reset"] = resets / max(pairs, 1)
    meta["frac_tool_returns_next_t0_before_result"] = lag_neg / max(lag_n, 1)
    assert meta["assert_tool_exec_time_rate"] >= 0.95, "execution_time_ms 和時間戳對不上"
    assert meta["assert_cache_read_eq_prev_ctx_rate(given cr>0)"] >= 0.80, "cache_read 和上一輪 context 對不上"
    assert meta["frac_tool_returns_next_t0_before_result"] <= 0.05, "下一個請求比工具結果早"
    meta["idle_start_rule"] = "t_idle=該訊息最後一個事件；工具回來的到達＝工具結果時間；人類回合的到達＝第一個事件"
    _assert_session_growth(meta, recs, thr=0.90)   # context 重設（壓縮、子代理）是真的現象，約 3.6%
    return _pack("cachewise", recs, meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    metas = []
    jobs = [("mooncake_conversation", lambda: load_mooncake("conversation")),
            ("mooncake_toolagent", lambda: load_mooncake("toolagent")),
            ("mooncake_conversation_strict", lambda: load_mooncake("conversation", "strict")),
            ("mooncake_toolagent_strict", lambda: load_mooncake("toolagent", "strict")),
            ("bailian_coder", lambda: load_bailian("qwen_coder_blksz_16.jsonl")),
            ("bailian_traceB", lambda: load_bailian("qwen_traceB_blksz_16.jsonl")),
            ("cachewise", load_cachewise)]
    for name, fn in jobs:
        out, meta = fn()
        meta["workload"] = name
        if out is not None:
            np.savez(os.path.join(a.out, f"{name}.npz"), **out)
        print(name, json.dumps(meta, ensure_ascii=False, default=float), flush=True)
        metas.append(meta)
    os.makedirs(os.path.dirname(a.csv), exist_ok=True)
    with open(a.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "ts", "workload", "metric", "value"])
        for m in metas:
            for k, v in m.items():
                if k != "workload":
                    w.writerow([_rid(), _ts(), m["workload"], k, v])


if __name__ == "__main__":
    main()
