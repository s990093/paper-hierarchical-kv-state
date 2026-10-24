"""x2_load.py — X2（回來時間猜得到嗎）的 trace 載入、session 重建與單位斷言（CLAUDE.md §1 規則 6）。

每個 loader 回傳 (req, meta, checks)：
  req    : pandas.DataFrame，一列一個請求（回合），欄位
           sid（session 編號）、t（秒）、out（上一輪輸出長度：token；WildChat 是字元）、
           size_tok（這個請求結束時 session 的 KV token 數＝input＋output；WildChat 是換算值）
  meta   : dict，trace 名稱、觀察窗起訖（秒；設限用）、說明
  checks : list[dict]，每條「用 A 欄驗算 B 欄」的斷言：check、value、threshold、passed、note
斷言不過時 loader 丟 AssertionError（呼叫端照錄、停下該 trace），不會默默換規則。

下載的資料是不受信任的資料：這裡只讀 JSON／parquet，不執行資料目錄裡的任何程式。
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)

DATA = os.environ.get("TIARA_DATASETS", "/ssd8/hungwei/paper-hkv/datasets")
MC_DIR = os.path.join(DATA, "mooncake", "traces")
BL_DIR = os.path.join(DATA, "qwen-bailian-usagetraces-anon")
WC_DIR = os.path.join(DATA, "wildchat-1m", "data")
CW_DIR = os.path.join(DATA, "cachewise-coding-traces", "repo", "parsed_traces")

MC_TOL = 0.1          # §2.4a：子 input ≥ 父 input＋(1−0.1)×父 output（事後定的容差，照實列在文件 §6）
MC_MIN_CONSISTENT = 0.90


class ValidationError(AssertionError):
    """斷言不過：帶著全部的 checks，讓呼叫端照錄後停下。"""
    def __init__(self, msg, checks):
        super().__init__(msg)
        self.checks = checks


def _chk(checks, check, value, threshold, passed, note=""):
    checks.append(dict(check=check, value=float(value), threshold=str(threshold), passed=bool(passed), note=note))


def _finish(df, checks, meta):
    """排序、算每個請求的回合序號（session 內 1 起算）、上一次間隔、下一個請求時刻。"""
    df = df.sort_values(["t", "order"], kind="mergesort").reset_index(drop=True)
    df["turn"] = df.groupby("sid").cumcount() + 1
    g = df.groupby("sid")["t"]
    df["prev_t"] = g.shift(1)
    df["next_t"] = g.shift(-1)
    assert ((df["t"] - df["prev_t"]).dropna() >= 0).all(), "session 內時間倒退"
    if "t_idle" in df.columns:  # 有請求結束時間（FreeInference）：上一次間隔＝這次開始 − 上次閒置開始
        df["prev_gap"] = (df["t"] - df.groupby("sid")["t_idle"].shift(1)).clip(lower=0.0)  # 重疊的請求（極少）記 0
    else:
        df["prev_gap"] = df["t"] - df["prev_t"]
    meta["n_req"] = len(df)
    meta["n_sessions"] = int(df["sid"].nunique())
    return df.drop(columns=["order"]), meta, checks


# ---------------------------------------------------------------- Mooncake
def load_mooncake(name, strict=False, rule="d4"):
    """前綴鏈重建 session。驗算 (a)(b)(c) 見文件 §2.1。
    rule="d4"（登記的規則）：父＝最後一個完整 block（位置 ≥1）出現在子前綴裡的最近請求（D4 cmd_char）。
    strict=True（事後的敏感度，只報告）：在 d4 規則上切掉驗算 (a) 不過的父子連結，扇出 >1 的父只留最早的子。
    rule="lit"（事後，主 session 10:01Z 要求報第二種規則）：照抄文獻 agent 的嚴格規則
      （research/calc/mooncake_chain2.py）：子 j 從 L=len(h)−1 往下找「完整前綴＝h[:L]」的較早請求（時間嚴格較早）；
      第一個找到的 L，若 h[:L] 被 ≤3 個請求共用且候選父唯一才連，否則不連。"""
    import m8_trace_oracle as m8  # 只 import、不改；TRACE_DIR 在這個 process 內指到 ssd8
    m8.TRACE_DIR = MC_DIR
    rows, m8meta = m8.load_trace(name)  # 斷言 len(hash_ids)==ceil(input_length/512) ≥ 0.95、前綴鏈
    checks = []
    _chk(checks, "len(hash_ids)==ceil(input_length/512)", m8meta["unit_match_rate"], ">=0.95", m8meta["unit_match_rate"] >= 0.95,
         "m8_trace_oracle.load_trace")
    _chk(checks, "prefix_chain_bad_pos", m8meta["chain_bad_pos"], "==0", m8meta["chain_bad_pos"] == 0)
    n = len(rows)
    inp = np.array([r["input_length"] for r in rows], dtype=np.int64)
    out = np.array([r["output_length"] for r in rows], dtype=np.int64)
    ts = np.array([r["timestamp"] for r in rows], dtype=np.int64)
    lastfull, par = {}, np.full(n, -1, dtype=np.int64)
    if rule == "d4":
        for ri, r in enumerate(rows):
            h = r["hash_ids"]
            for k in range(len(h) - 1, 0, -1):  # 不算 block 0（conversation 全部請求共用）
                if h[k] in lastfull:
                    par[ri] = lastfull[h[k]]
                    break
            nf = r["input_length"] // m8.BLOCK_TOK
            if nf >= 2:
                lastfull[h[nf - 1]] = ri
    elif rule == "lit":
        # 前綴鏈已驗過（同一 hash 永遠同位置、同前一個 hash），所以「前綴 h[:L]」可以用 h[L−1] 代表
        from collections import Counter, defaultdict
        pop = Counter(b for r in rows for b in r["hash_ids"])  # 每個請求內 hash 不重複（m8 檢查 req_with_dup_hash）
        cand = defaultdict(list)
        for i, r in enumerate(rows):
            f = r["input_length"] // m8.BLOCK_TOK
            if f >= 1:
                cand[r["hash_ids"][f - 1]].append(i)
        n_amb = 0
        for j, r in enumerate(rows):
            h = r["hash_ids"]
            for L in range(len(h) - 1, 0, -1):
                ps = [i for i in cand.get(h[L - 1], []) if i < j and ts[i] < ts[j]]
                if ps:
                    if pop[h[L - 1]] <= 3 and len(ps) == 1:
                        par[j] = ps[0]
                    elif len(ps) > 1:
                        n_amb += 1
                    break
        _chk(checks, "lit_rule_ambiguous_children", float(n_amb), "report", True, "候選父不止一個而不連的子請求數")
    else:
        raise ValueError(rule)
    c = np.nonzero(par >= 0)[0]
    p = par[c]
    cons = inp[c] >= inp[p] + (1 - MC_TOL) * out[p]
    frac_cons = float(cons.mean()) if len(c) else float("nan")
    fan = np.bincount(p, minlength=n)
    fp = fan[fan > 0]
    gap_s = (ts[c] - ts[p]) / 1000.0
    ok_out = out[p] >= 50
    ratio = np.median(gap_s[ok_out] / out[p][ok_out])
    _chk(checks, f"child_input>=parent_input+(1-{MC_TOL})*parent_output", frac_cons, f">={MC_MIN_CONSISTENT}",
         frac_cons >= MC_MIN_CONSISTENT, f"n_pairs={len(c)}；容差 0 時 {float((inp[c] >= inp[p] + out[p]).mean()):.4f}")
    _chk(checks, "parent_fanout_frac_gt1", float((fp > 1).mean()), "report", True, f"max={int(fp.max())}")
    _chk(checks, "median(gap_s/parent_output_tok)", ratio, "0.01..10 s/token（timestamp 是毫秒）", 0.01 <= ratio <= 10)
    if strict:
        cut = c[~cons]
        par[cut] = -1
        c2 = np.nonzero(par >= 0)[0]
        firstchild = {}
        for i in c2:  # c2 依請求順序：每個父只留最早的子
            if par[i] in firstchild:
                par[i] = -1
            else:
                firstchild[par[i]] = i
        _chk(checks, "strict_links_cut_frac", float(1 - (par >= 0).sum() / max(len(c), 1)), "report", True,
             "事後敏感度：切掉的父子連結比例")
    root = np.arange(n)
    for i in range(n):
        if par[i] >= 0:
            root[i] = root[par[i]]
    _, cnt = np.unique(root, return_counts=True)
    _chk(checks, "max_requests_per_session", float(cnt.max()), "report", True,
         "一個 session 太大＝共用前綴把不同 session 併在一起（pitfall #2）")
    meta = dict(trace=f"mooncake_{name}" + ("_strict" if strict else "") + ("_lit" if rule == "lit" else ""), t_start=float(ts.min()) / 1000.0, t_end=float(ts.max()) / 1000.0,
                unit_out="token", note=("父＝最後一個完整 block（≥1）出現在子前綴的最近請求" if rule == "d4" else
                                        "文獻 agent 的嚴格規則（共用 ≤3、父唯一、時間嚴格較早）") + ("；strict 切連結" if strict else ""))
    for ck in checks:
        if not ck["passed"]:
            raise ValidationError(f"mooncake_{name}: 驗算不過 {ck}", checks)
    df = pd.DataFrame(dict(sid=root, t=ts / 1000.0, out=out.astype(float), size_tok=(inp + out).astype(float),
                           order=np.arange(n)))
    return _finish(df, checks, meta)


# ---------------------------------------------------------------- Qwen-Bailian
def load_bailian(name):
    path = os.path.join(BL_DIR, f"{name}_blksz_16.jsonl")
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    checks = []
    match = sum(len(r["hash_ids"]) == math.ceil(r["input_length"] / 16) for r in rows) / n
    _chk(checks, "len(hash_ids)==ceil(input_length/16)", match, ">=0.95", match >= 0.95)
    ts = np.array([r["timestamp"] for r in rows], dtype=float)
    _chk(checks, "timestamp_sorted", float(np.all(np.diff(ts) >= 0)), "==1", bool(np.all(np.diff(ts) >= 0)))
    idx = {r["chat_id"]: i for i, r in enumerate(rows)}
    assert len(idx) == n, "chat_id 不唯一"
    par = np.array([-1 if r["parent_chat_id"] == -1 else idx.get(r["parent_chat_id"], -2) for r in rows])
    _chk(checks, "parent_missing_frac", float((par == -2).mean()), "==0", bool((par == -2).sum() == 0))
    c = np.nonzero(par >= 0)[0]
    out = np.array([r["output_length"] for r in rows], dtype=np.int64)
    inp = np.array([r["input_length"] for r in rows], dtype=np.int64)
    if len(c):
        pref = np.mean([rows[i]["hash_ids"][:len(rows[par[i]]["hash_ids"]) - 1] ==
                        rows[par[i]]["hash_ids"][:-1] for i in c])
        turn_ok = np.mean([rows[i]["turn"] == rows[par[i]]["turn"] + 1 for i in c])
        later = np.mean(ts[c] >= ts[par[c]])
        g = ts[c] - ts[par[c]]
        ok = out[par[c]] >= 50
        rr = g[ok] / out[par[c]][ok]
        med = float(np.median(rr))
        fast = float(np.mean(rr < 0.001))
        _chk(checks, "child_hash_prefix==parent_hash[:-1]", pref, ">=0.95", pref >= 0.95)
        _chk(checks, "child_turn==parent_turn+1", turn_ok, ">=0.99", turn_ok >= 0.99)
        _chk(checks, "child_ts>=parent_ts", later, "==1", later == 1.0)
        _chk(checks, "median(gap_s/parent_output_tok)", med, "0.01..10 s/token（timestamp 是秒）", 0.01 <= med <= 10)
        _chk(checks, "frac(gap < 1ms*parent_output)", fast, "<0.05", fast < 0.05, "≥1000 token/s 的解碼＝單位不對")
        _chk(checks, "child_input>=parent_input+0.9*parent_output", float(np.mean(inp[c] >= inp[par[c]] + 0.9 * out[par[c]])),
             "report", True, "thinking 會去掉推理內容，只報告")
        fan = np.bincount(par[c], minlength=n)
        _chk(checks, "parent_fanout_frac_gt1", float((fan[fan > 0] > 1).mean()), "report", True)
    else:
        _chk(checks, "n_child_requests", 0.0, "report", True, "全部是根請求：沒有 session 結構")
    root = np.arange(n)
    for i in range(n):
        if par[i] >= 0:
            root[i] = root[par[i]]
    for ck in checks:
        if not ck["passed"]:
            raise ValidationError(f"bailian_{name.replace('qwen_', '')}: 驗算不過 {ck}", checks)
    meta = dict(trace=f"bailian_{name.replace('qwen_', '')}", t_start=float(ts.min()), t_end=float(ts.max()), unit_out="token",
                note="parent_chat_id 串成 session")
    df = pd.DataFrame(dict(sid=root, t=ts, out=out.astype(float), size_tok=(inp + out).astype(float), order=np.arange(n)))
    return _finish(df, checks, meta)


# ---------------------------------------------------------------- WildChat-1M
def load_wildchat(chars_per_token):
    """一個對話＝一個 session；一輪＝一個 assistant 訊息（只有 assistant 有 timestamp）。
    chars_per_token：Llama-3.1 tokenizer 在樣本上量的字元／token（measure_wildchat_cpt），用來換算 KV 大小。"""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    files = sorted(f for f in os.listdir(WC_DIR) if f.endswith(".parquet"))
    assert len(files) == 14, f"WildChat parquet 應有 14 個，找到 {len(files)}"
    parts, checks = [], []
    n_conv = n_top_ok = n_turn_ok = n_mono = n_alt = 0
    sid_base = 0
    for fn in files:
        tb = pq.read_table(os.path.join(WC_DIR, fn), columns=["timestamp", "turn", "conversation"])
        conv = tb.column("conversation").combine_chunks()
        off = conv.offsets.to_numpy()
        fl = conv.flatten()
        role = fl.field("role").to_numpy(zero_copy_only=False)
        mts = pc.cast(fl.field("timestamp"), pa.int64()).to_numpy(zero_copy_only=False)  # µs，user 是 null
        mts_valid = fl.field("timestamp").is_valid().to_numpy(zero_copy_only=False)
        clen = pc.utf8_length(fl.field("content")).to_numpy(zero_copy_only=False).astype(np.int64)
        top = pc.cast(tb.column("timestamp").combine_chunks(), pa.int64()).to_numpy(zero_copy_only=False)
        turn = tb.column("turn").to_numpy()
        nc = len(off) - 1
        conv_id = np.repeat(np.arange(nc), np.diff(off))
        is_a = role == "assistant"
        is_u = role == "user"
        # 斷言用的逐對話量
        n_user = np.bincount(conv_id[is_u], minlength=nc)
        a_valid = is_a & mts_valid
        n_a = np.bincount(conv_id[is_a], minlength=nc)
        n_av = np.bincount(conv_id[a_valid], minlength=nc)
        last_a = np.full(nc, -1, dtype=np.int64)
        np.maximum.at(last_a, conv_id[a_valid], mts[a_valid])
        n_top_ok += int(np.sum(last_a == top))
        n_turn_ok += int(np.sum(n_user == turn))
        # 交替 user/assistant
        alt = np.ones(nc, bool)
        pos_in = np.arange(len(role)) - off[conv_id]
        expect_u = (pos_in % 2) == 0
        bad = (expect_u & ~is_u) | (~expect_u & ~is_a)
        alt[np.unique(conv_id[bad])] = False
        n_alt += int(alt.sum())
        # 每輪：assistant 訊息；累積字元＝到這則 assistant 為止所有訊息
        cum = np.cumsum(clen)
        cum_conv = cum - np.r_[0, cum][off[conv_id]]
        ai = np.nonzero(a_valid)[0]
        d = pd.DataFrame(dict(sid=sid_base + conv_id[ai], t=mts[ai] / 1e6, out=clen[ai].astype(float),
                              size_tok=cum_conv[ai] / chars_per_token))
        # 單調：對話內 assistant timestamp 不遞減（d 依對話內位置排列）
        sv, tv = d["sid"].values, d["t"].values
        back = np.r_[False, (sv[1:] == sv[:-1]) & (tv[1:] < tv[:-1])]
        n_mono += int(nc - len(np.unique(sv[back])))
        assert np.all(n_a == n_av), f"{fn}: 有 assistant 訊息沒有 timestamp"
        parts.append(d)
        n_conv += nc
        sid_base += nc
    df = pd.concat(parts, ignore_index=True)
    _chk(checks, "top_timestamp==max(assistant timestamp)", n_top_ok / n_conv, ">=0.99", n_top_ok / n_conv >= 0.99)
    _chk(checks, "turn==n_user_messages", n_turn_ok / n_conv, ">=0.99", n_turn_ok / n_conv >= 0.99)
    _chk(checks, "assistant_ts_nondecreasing_in_conv", n_mono / n_conv, ">=0.99", n_mono / n_conv >= 0.99)
    _chk(checks, "alternating_user_assistant", n_alt / n_conv, "report", True)
    # 同一對話內 assistant 時間倒退的回合：照實丟掉該對話（不重排）
    sv, tv = df["sid"].values, df["t"].values
    back = np.r_[False, (sv[1:] == sv[:-1]) & (tv[1:] < tv[:-1])]
    df = df[~np.isin(sv, np.unique(sv[back]))].copy()
    # 物理檢查：下一輪回應 c 字元在 gap 秒內生成；gap < c/2000（>2000 字元/s）的比例
    df = df.sort_values(["sid", "t"], kind="mergesort")
    nxt_t = df.groupby("sid")["t"].shift(-1)
    nxt_c = df.groupby("sid")["out"].shift(-1)
    m = nxt_t.notna()
    gap = (nxt_t - df["t"])[m]
    fast = float(np.mean(gap.values < nxt_c[m].values / 2000.0))
    _chk(checks, "frac(gap_s < next_reply_chars/2000)", fast, "report",
         True, "assistant timestamp 若是回應完成時刻，gap 應 ≥ 下一則回應的生成時間；這個比例大＝timestamp 不是完成時刻")
    _chk(checks, "frac(gap==0)", float(np.mean(gap.values == 0)), "report", True, "timestamp 解析度 1 s")
    for ck in checks:
        if not ck["passed"]:
            raise ValidationError(f"wildchat: 驗算不過 {ck}", checks)
    df["order"] = np.arange(len(df))
    meta = dict(trace="wildchat_1m", t_start=float(df["t"].min()), t_end=float(df["t"].max()), unit_out="char",
                note=f"一輪＝一則 assistant；回來時刻＝下一則 assistant 的 timestamp；size_tok＝累積字元/{chars_per_token:.4f}",
                n_conv_raw=n_conv)
    return _finish(df, checks, meta)


def measure_wildchat_cpt(n_sample=2000, seed=0):
    """Llama-3.1 tokenizer 在 WildChat 隨機 n_sample 個對話上量的「字元／token」。回傳 (ratio, info)。"""
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tok_dir = os.environ.get("M7_LLAMA_GLOB")
    assert tok_dir and os.path.isdir(tok_dir), f"找不到 Llama tokenizer：{tok_dir}"
    tok = AutoTokenizer.from_pretrained(tok_dir)
    rng = np.random.default_rng(seed)
    files = sorted(f for f in os.listdir(WC_DIR) if f.endswith(".parquet"))
    per = int(math.ceil(n_sample / len(files)))
    nch = ntk = 0
    nconv = 0
    for fn in files:
        pf = pq.ParquetFile(os.path.join(WC_DIR, fn))
        nr = pf.metadata.num_rows
        pick = set(rng.choice(nr, size=min(per, nr), replace=False).tolist())
        tb = pf.read(columns=["conversation"])
        convs = tb.column("conversation").to_pylist()
        for i in sorted(pick):
            for msg in convs[i]:
                s = msg["content"] or ""
                nch += len(s)
                ntk += len(tok(s, add_special_tokens=False)["input_ids"])
            nconv += 1
    return nch / ntk, dict(n_conv=nconv, chars=nch, tokens=ntk, tokenizer=tok_dir, seed=seed)


# ---------------------------------------------------------------- CacheWise（只報告）
def load_cachewise():
    from datetime import datetime
    man = json.load(open(os.path.join(CW_DIR, "manifest.json")))
    checks, recs = [], []
    n_tool = n_tool_ok = 0
    last_ev = {}
    for proj in sorted(os.listdir(CW_DIR)):
        pd_ = os.path.join(CW_DIR, proj)
        if not os.path.isdir(pd_):
            continue
        for sess in sorted(os.listdir(pd_)):
            evs = json.load(open(os.path.join(pd_, sess, "events.json")))
            seen = {}
            sid = f"{proj}/{sess}"
            tmax = -1e30
            for e in evs:
                tt = datetime.fromisoformat(e["timestamp"]).timestamp()
                tmax = max(tmax, tt)
                if e["event_type"] == "llm_call":
                    rq = e.get("request_id")
                    if rq is None:  # 約 1% 的 llm_call 沒有 request_id：用 message_id，再沒有就自成一個請求
                        rq = ("msg", e.get("message_id")) if e.get("message_id") else ("idx", len(seen))
                    tot = (e.get("input_tokens") or 0) + (e.get("cache_creation_input_tokens") or 0) + \
                          (e.get("cache_read_input_tokens") or 0)
                    o = e.get("output_tokens") or 0
                    if rq not in seen:
                        seen[rq] = dict(sid=sid, t=tt, out=o, inp=tot)
                    else:
                        seen[rq]["out"] = max(seen[rq]["out"], o)
                        seen[rq]["inp"] = max(seen[rq]["inp"], tot)
                elif e["event_type"] == "tool_call" and e.get("execution_time_ms") is not None and e.get("result_timestamp"):
                    n_tool += 1
                    dt = (datetime.fromisoformat(e["result_timestamp"]).timestamp() - tt) * 1000.0
                    n_tool_ok += abs(dt - e["execution_time_ms"]) <= 1.0
                    tmax = max(tmax, datetime.fromisoformat(e["result_timestamp"]).timestamp())
            recs.extend(seen.values())
            last_ev[sid] = tmax
    _chk(checks, "tool execution_time_ms == result_ts - ts (±1 ms)", n_tool_ok / max(n_tool, 1), ">=0.95",
         n_tool_ok / max(n_tool, 1) >= 0.95, f"n_tool={n_tool}")
    _chk(checks, "manifest num_sessions", float(man["num_sessions"]), "report", True)
    for ck in checks:
        if not ck["passed"]:
            raise ValidationError(f"cachewise: 驗算不過 {ck}", checks)
    df = pd.DataFrame(recs)
    sids = {s: i for i, s in enumerate(sorted(df["sid"].unique()))}
    df["sess_end"] = df["sid"].map(last_ev)
    df["sid"] = df["sid"].map(sids)
    df["size_tok"] = (df["inp"] + df["out"]).astype(float)
    df["out"] = df["out"].astype(float)
    df["order"] = np.arange(len(df))
    meta = dict(trace="cachewise", t_start=float(df["t"].min()), t_end=float("nan"), unit_out="token",
                note="每個 session 的最後一個 request 設限在該 session 最後一個事件；session 時間戳各自平移，不做快照")
    sess_end = df[["sid", "sess_end"]].drop_duplicates().set_index("sid")["sess_end"]
    out = _finish(df.drop(columns=["inp"]), checks, meta)
    out[0]["sess_end"] = out[0]["sid"].map(sess_end)
    return out


# ---------------------------------------------------------------- FreeInference agentic trace（登記之後才加，README §4）
FI_DIR = os.path.join(DATA, "freeinference_agentic_trace", "release")


def load_freeinference(check_blocks=True):
    """session＝session_id（資料自己重建、不跨週；子 agent 也算一個 session）。
    一個請求的閒置從 end_ms 開始；回來＝同 session 下一個請求的 start_ms；設限在該週最後一個 end_ms。
    grp（閒置當下就知道的）＝這個請求的 finish_reason 是 tool_calls → 'tool'（等工具），否則 'user'（等使用者）。"""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    weeks = sorted(d for d in os.listdir(FI_DIR) if os.path.isdir(os.path.join(FI_DIR, d, "processed")))
    assert len(weeks) == 16, f"FreeInference 應有 16 週，找到 {len(weeks)}"
    checks, parts = [], []
    n_req = n_blk = n_step_ok = n_sess = n_se = n_ttft = 0
    dec = []
    lat_ok = lat_n = 0
    sid_base = 0
    n_sub = 0
    users = set()
    for wi, w in enumerate(weeks):
        pdir = os.path.join(FI_DIR, w, "processed")
        pf = pq.ParquetFile(os.path.join(pdir, "requests.parquet"))
        blen = []
        if check_blocks:
            for rg in range(pf.num_row_groups):  # block_ids 很大（全部 8 GB）：只取長度
                t = pf.read_row_group(rg, columns=["block_ids"])
                blen.append(pc.list_value_length(t.column("block_ids")).to_numpy(zero_copy_only=False))
                del t
            blen = np.concatenate(blen).astype(np.int64)
        tb = pf.read(columns=["session_id", "step", "start_ms", "end_ms", "ttft_ms", "step_trigger", "prompt_tokens",
                              "completion_tokens", "role_tokens", "finish_reason"])
        rt = tb.column("role_tokens").combine_chunks()
        rsum = np.zeros(len(tb), dtype=np.int64)
        for f in ("system", "user", "assistant", "tool", "tool_definitions"):
            rsum += pc.fill_null(rt.field(f), 0).to_numpy(zero_copy_only=False).astype(np.int64)
        if check_blocks:
            n_blk += int(np.sum(blen == rsum // 16))
        d = tb.drop(["role_tokens"]).to_pandas()
        n_req += len(d)
        st, en, tt = d["start_ms"].values, d["end_ms"].values, d["ttft_ms"].values
        n_se += int(np.sum(st <= en))
        okt = ~pd.isna(tt)
        n_ttft += int(np.sum((tt[okt] <= (en - st)[okt]))) + int((~okt).sum())
        ct = d["completion_tokens"].fillna(0).values
        m = (ct >= 50) & okt
        dec.append(((en - st - np.where(okt, tt, 0)) / np.maximum(ct, 1))[m])
        d = d.sort_values(["session_id", "step"], kind="mergesort").reset_index(drop=True)
        gstep = d.groupby("session_id")["step"]
        ok_step = (gstep.transform("min") == 0) & (gstep.transform("max") == gstep.transform("size") - 1)
        n_step_ok += int(d.loc[~d["session_id"].duplicated(), "session_id"].map(ok_step.groupby(d["session_id"]).all()).sum())
        n_sess += int(d["session_id"].nunique())
        # 工具 latency_ms（README：到下一次 LLM 請求的間隔）vs 下一個請求 start − 這個請求 end
        tc = pq.read_table(os.path.join(pdir, "tool_calls.parquet"), columns=["session_id", "step", "latency_ms"]).to_pandas()
        tl = tc.groupby(["session_id", "step"])["latency_ms"].max()
        nxt_start = d.groupby("session_id")["start_ms"].shift(-1)
        gap_ms = (nxt_start - d["end_ms"]).values
        key = pd.MultiIndex.from_arrays([d["session_id"], d["step"]])
        lat = tl.reindex(key).values
        mm = ~np.isnan(lat) & ~np.isnan(gap_ms)
        lat_ok += int(np.sum(np.abs(lat[mm] - gap_ms[mm]) <= np.maximum(1000.0, 0.05 * np.abs(gap_ms[mm]))))
        lat_n += int(mm.sum())
        se = pq.read_table(os.path.join(pdir, "sessions.parquet"), columns=["session_id", "user_id", "parent_session_id"]).to_pandas()
        n_sub += int(se["parent_session_id"].notna().sum())
        users.update(se["user_id"].dropna().tolist())
        codes = {s_: i for i, s_ in enumerate(sorted(d["session_id"].unique()))}
        wk_end = float(d["end_ms"].max()) / 1000.0
        part = pd.DataFrame(dict(
            sid=sid_base + d["session_id"].map(codes).values, t=d["start_ms"].values / 1000.0,
            t_idle=d["end_ms"].values / 1000.0, out=d["completion_tokens"].fillna(0).values.astype(float),
            size_tok=(d["prompt_tokens"].fillna(0) + d["completion_tokens"].fillna(0)).values.astype(float),
            grp=np.where(d["finish_reason"].values == "tool_calls", "tool", "user"),
            trig=d["step_trigger"].values, cens_at=wk_end, week=wi, week_start=float(d["start_ms"].min()) / 1000.0,
            finish=d["finish_reason"].astype(str).values))
        part["uturn"] = (part["trig"] == "user").groupby(part["sid"]).cumsum().values
        parts.append(part)
        sid_base += len(codes)
    df = pd.concat(parts, ignore_index=True)
    # 同一 session 的請求偶爾重疊（下一個在上一個結束前就開始）：閒置從「到目前為止所有請求的最晚結束」起算
    df = df.sort_values(["sid", "t"], kind="mergesort").reset_index(drop=True)
    prev_cm = df.groupby("sid")["t_idle"].cummax().groupby(df["sid"]).shift(1)
    n_ov = int(((df["t"] - prev_cm) < 0).sum())
    df["t_idle"] = df.groupby("sid")["t_idle"].cummax()
    _chk(checks, "requests_overlapping_previous (start < max previous end)", float(n_ov), "report", True,
         f"共 {len(df)} 個請求；閒置開始改用累積最晚結束")
    dec = np.concatenate(dec)
    med_dec = float(np.median(dec))
    if check_blocks:
        _chk(checks, "len(block_ids)==floor(sum(role_tokens)/16)", n_blk / n_req, ">=0.95", n_blk / n_req >= 0.95,
             "o200k_base、16-token、最後不滿一塊丟掉（README）")
    else:
        _chk(checks, "len(block_ids) check skipped", float("nan"), "report", True, "只在主分析 run 檢查（讀 8 GB 太久）")
    _chk(checks, "steps_consecutive_0..n-1_per_session", n_step_ok / n_sess, ">=0.99", n_step_ok / n_sess >= 0.99)
    _chk(checks, "start_ms<=end_ms", n_se / n_req, "==1", n_se == n_req)
    _chk(checks, "ttft_ms<=end_ms-start_ms", n_ttft / n_req, ">=0.99", n_ttft / n_req >= 0.99)
    _chk(checks, "median((end-start-ttft)/completion_tokens) ms/token", med_dec, "0.5..500（時間是毫秒）",
         0.5 <= med_dec <= 500, "解碼速度；秒會是 ~0.02、微秒會是 ~2e4")
    _chk(checks, "tool latency_ms == next.start - end (±max(1s,5%))", lat_ok / max(lat_n, 1), "report", True,
         f"n={lat_n}")
    _chk(checks, "n_subagent_sessions", float(n_sub), "report", True)
    _chk(checks, "n_users", float(len(users)), "report", True)
    for ck in checks:
        if not ck["passed"]:
            raise ValidationError(f"freeinference: 驗算不過 {ck}", checks)
    df["order"] = np.arange(len(df))
    meta = dict(trace="freeinference", t_start=float(df["t"].min()), t_end=float(df["cens_at"].max()), unit_out="token",
                note="閒置從 end_ms 起算；回來＝下一個請求 start_ms；設限在各週最後的 end_ms；size 用 o200k token × Llama 的 128 KiB")
    return _finish(df, checks, meta)


def _fi_week_blocks(w):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(os.path.join(FI_DIR, w, "processed", "requests.parquet"))
    n = ok = 0
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["block_ids", "role_tokens"])
        bl = pc.list_value_length(t.column("block_ids")).to_numpy(zero_copy_only=False).astype(np.int64)
        rt = t.column("role_tokens").combine_chunks()
        rs = np.zeros(len(t), dtype=np.int64)
        for f in ("system", "user", "assistant", "tool", "tool_definitions"):
            rs += pc.fill_null(rt.field(f), 0).to_numpy(zero_copy_only=False).astype(np.int64)
        n += len(t)
        ok += int(np.sum(bl == rs // 16))
        del t
    return w, n, ok


def freeinference_block_check(n_proc=4):
    """規則 6 的 block 斷言單獨跑（block_ids 全部 8 GB，逐週平行）。回傳 checks。"""
    from multiprocessing import Pool
    weeks = sorted(d for d in os.listdir(FI_DIR) if os.path.isdir(os.path.join(FI_DIR, d, "processed")))
    with Pool(n_proc) as pool:
        res = pool.map(_fi_week_blocks, weeks)
    n = sum(r[1] for r in res)
    ok = sum(r[2] for r in res)
    checks = []
    _chk(checks, "len(block_ids)==floor(sum(role_tokens)/16)", ok / n, ">=0.95", ok / n >= 0.95,
         f"n={n}；逐週 {[(r[0], round(r[2] / r[1], 6)) for r in res]}")
    return checks


LOADERS = {
    "freeinference": lambda **k: load_freeinference(check_blocks=False),  # block 斷言另跑（--fi-block-check）
    "mooncake_conversation": lambda **k: load_mooncake("conversation"),
    "mooncake_toolagent": lambda **k: load_mooncake("toolagent"),
    "mooncake_conversation_strict": lambda **k: load_mooncake("conversation", strict=True),  # 事後敏感度，只報告
    "mooncake_conversation_lit": lambda **k: load_mooncake("conversation", rule="lit"),  # 主 session 要求的第二種規則
    "bailian_traceA": lambda **k: load_bailian("qwen_traceA"),
    "bailian_traceB": lambda **k: load_bailian("qwen_traceB"),
    "bailian_thinking": lambda **k: load_bailian("qwen_thinking"),
    "bailian_coder": lambda **k: load_bailian("qwen_coder"),
    "wildchat_1m": lambda **k: load_wildchat(k["chars_per_token"]),
    "cachewise": lambda **k: load_cachewise(),
}
