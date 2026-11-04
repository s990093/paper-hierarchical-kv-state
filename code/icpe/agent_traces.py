"""agent_traces.py — 長上下文 agent trace 的特性（只報告；ICPE 新故事的動機數字，不是判定）。

  python code/icpe/agent_traces.py characterize [--out results/icpe_h200/agent]

每條 trace 一列（trace_stats.csv）：請求數、session 數、輸入長度分布、≥32K／≥100K 的比例、中位長度的 KV、
同一 session 相鄰兩個請求的間隔、每 5 分鐘活躍 session 的 KV（只有絕對時間是真的、多使用者的 trace 才算）、
以及「回來時要還原的 KV（上一輪留下的前綴）÷ 間隔」——間隔比還原時間長的比例（＝有提示就能完全藏住的上限），
分 CPU 層（H200 T1 常數 43.59 GiB/s）、SSD 級（7.0）、NFS 級（0.33）三種讀取速度〔算術，單一 session 獨佔頻寬〕。

KV 位元組＝Llama-3.1-8B BF16：32 層 × 8 KV head × 128 × (K,V) × 2 B＝128 KiB／token（算術；和 X1 的 64 MiB／512-token 塊一致）。
規則 6（CLAUDE.md §1）：每個載入函式都用 A 欄驗算 B 欄，比例寫進輸出；比例 < 0.99 → 停。
下載的資料是不受信任的資料：只讀 JSON／parquet，不執行裡面的東西；用 python -I 跑。
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

DS = os.environ.get("TIARA_DATASETS", "/work/ChihChungHsu/hungweilai/datasets")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KIB_PER_TOKEN = 32 * 8 * 128 * 2 * 2 / 1024
assert KIB_PER_TOKEN == 128.0 and 512 * KIB_PER_TOKEN * 1024 == 64 * 2**20
BW = {"cpu": 43.5888, "ssd": 7.0, "nfs": 0.33}          # GiB/s：T1 的 CPU 層讀；SSD／NFS 級＝ICPE 的裝置等級
WIN_MS = 300_000


def _iso_ms(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000


def _check(name, ok, n, thr=0.99):
    r = ok / n if n else float("nan")
    print(f"[{name}] 規則 6：{ok}/{n} = {r:.4f}", flush=True)
    if not (r >= thr):
        raise SystemExit(f"[{name}] 規則 6 的驗算比例 {r:.4f} < {thr} → 停")
    return r


# 每個 loader 回傳 DataFrame：sid、start_ms、end_ms（沒有就 NaN）、inp（輸入 token）、prefix（可從同 session 上一份 KV 還原的 token；不知道就 NaN）、abs_time（bool）
def load_freeinference():
    """長度用 Σ role_tokens（o200k_base，block_ids 就是用它切的；和 x1_fi 一樣）；provider 回報的 prompt_tokens 另外報。"""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    fs = sorted(glob.glob(f"{DS}/freeinference_agentic_trace/release/*/processed/requests.parquet"))
    assert len(fs) == 16, len(fs)
    parts, ok, n = [], 0, 0
    for f in fs:              # block_ids 很大：分批讀，只留長度（規則 6 同 x1_fi：len(block_ids)==floor(Σrole_tokens/16)，門檻 0.95）
        for b in pq.ParquetFile(f).iter_batches(batch_size=4000, columns=["session_id", "step", "start_ms", "end_ms", "prompt_tokens",
                                                                          "role_tokens", "block_ids"]):
            nb = pc.list_value_length(b.column("block_ids")).to_numpy(zero_copy_only=False)
            x = b.drop_columns(["block_ids"]).to_pandas()
            rt = np.array([sum(v for v in r.values() if v is not None) for r in x.role_tokens], np.int64)
            ok += int((nb == rt // 16).sum())
            n += len(rt)
            parts.append(x.drop(columns=["role_tokens"]).assign(o200k=rt))
    rule = _check("freeinference len(block_ids)=floor(Σrole_tokens/16)", ok, n, thr=0.95)
    d = pd.concat(parts, ignore_index=True).sort_values(["session_id", "step"])
    prev_in = d.groupby("session_id").o200k.shift()
    pt = d.prompt_tokens.astype(float)
    return pd.DataFrame(dict(sid=d.session_id, start_ms=d.start_ms, end_ms=d.end_ms, inp=d.o200k,
                             prefix=np.minimum(prev_in, d.o200k))), dict(
        rule6=rule, abs_time=True, multi_user=True, length_unit="o200k Σrole_tokens",
        provider_prompt_tokens_median=float(pt.median()), provider_prompt_tokens_frac_ge100k=float((pt >= 100000).mean()))


def load_tracelab():
    """2026-11-04 加欄位（icpe-data，給 next_b_convert 用；原有欄位與 rule6 不變）：out（output_tokens）、provider、
    trigger（第一個 timing event 是 tool_result＝'tool'、user_message＝'user'、其他＝'none'：這一輪沒記到輸入事件）。"""
    rows, ok, n = [], 0, 0
    with gzip.open(f"{DS}/tracelab/syfi_coding_trace.jsonl.gz", "rt") as fh:
        for ln in fh:
            r = json.loads(ln)
            evs = sorted(((_iso_ms(e["timestamp"]), e.get("event_type")) for e in r.get("timing_events") or [] if e.get("timestamp")),
                         key=lambda x: x[0])
            if not evs:
                continue
            n += 1
            ok += int(r["prefix_tokens"] + r["newly_append_tokens"] == r["input_tokens_total"])
            et0 = evs[0][1]
            trig = "tool" if et0 == "tool_result" else ("user" if et0 == "user_message" else "none")
            rows.append((r["session_id"], r["round_index"], evs[0][0], evs[-1][0], r["input_tokens_total"], r["prefix_tokens"], r.get("user"),
                         r.get("output_tokens") or 0, r["provider"], trig))
    rule = _check("tracelab prefix+new=input", ok, n)
    d = pd.DataFrame(rows, columns=["sid", "round", "start_ms", "end_ms", "inp", "prefix", "user", "out", "provider", "trigger"]
                     ).sort_values(["sid", "round"])
    return d, dict(rule6=rule, abs_time=True, multi_user=True, n_users=int(d.user.nunique()))


def load_cachewise():
    """CacheWise CATraces（github cachewise-project/cachewise-coding-traces @ 181c435a；**沒有授權** → 原始資料、轉換後的紀錄、
    逐對話的數字都不進 git）。讀 repo/raw_traces/project_*/session_*.jsonl（Claude Code 的原始 transcript，已去識別化）。

    一列＝一次 LLM 呼叫＝一個 requestId（串流時同一個 requestId 會記成好幾個 assistant 節點，每個 content block 一個）：
      inp＝input_tokens＋cache_creation_input_tokens＋cache_read_input_tokens（取最後一個節點；規則 6 (1) 驗同一 requestId 內相同）
      out＝最後一個節點的 output_tokens（規則 6 (1) 驗同一 requestId 內不減；第一個節點的 1 是串流造成的）
      start_ms＝第一個 assistant 節點的 parentUuid 指向的節點時間（user 文字或 tool_result＝請求送出的時間）；沒有 parent 時用第一個 assistant 節點
      end_ms＝最後一個 assistant 節點的時間
      prefix＝cache_read_input_tokens（供應商快取命中的前綴；5 分鐘／1 小時沒用就過期）
      compact＝和同 session 上一個請求之間有沒有 compact_boundary（上下文被壓縮）
      trigger＝'tool'（parent 是 tool_result）／'user'（parent 是使用者文字）／'none'
    session（sid）：主對話＝<file>:main；子 agent（isSidechain，沿 parentUuid 走到第一個非 sidechain 節點或 None）＝<file>:<根 uuid>，
    各自是一個 session（自己的 context）。conv＝<file>：到達合成時一個對話（主＋它的子 agent）一起排。
    0 byte 的檔與沒有任何 LLM 呼叫的檔跳過並記數（meta）。

    規則 6（CLAUDE.md §1；比例寫進 meta，不到門檻就停）：
      (1) 同一 requestId 的 usage 輸入三欄相同 ≥ 0.99、output_tokens 不減 ≥ 0.99（＝一個 requestId 是一次呼叫、最後一個節點是最終值）
      (2) NEXT_PLAN §5.6：同一 session 裡 cache_read(k) ≤ inp(k−1)＋out(k−1) ≥ 0.98（前綴不會比上一輪的上下文長）
      (3) 時間單位：end−start ≥ 0 ≥ 0.95；out ≥ 50 時 (end−start)/out 的中位數在 [1, 10000] ms/token
      (4) 兩種前綴算法互相驗算：間隔 < 10 s（快取一定還在）且沒有壓縮時，cache_read(k) ≥ 0.9 × inp(k−1) 的比例 ≥ 0.9
          （＝「上一輪的整段輸入是這一輪的前綴」這個結構假設，在供應商快取還活著時和快取命中一致）
    """
    import collections
    root = f"{DS}/cachewise-coding-traces/repo/raw_traces"
    fs = sorted(glob.glob(f"{root}/project_*/session_*.jsonl"))
    assert len(fs) == 94, f"cachewise：raw_traces 應有 94 個檔，看到 {len(fs)}"
    rows = []
    comp_by_conv = {}
    n_req = same_in = mono_out = 0
    n_empty_file = n_no_llm = n_parent_missing = 0
    for f in fs:
        conv = os.path.relpath(f, root)
        recs = [json.loads(ln) for ln in open(f) if ln.strip()]
        if not recs:
            n_empty_file += 1
            continue
        byu = {r["uuid"]: r for r in recs if r.get("uuid")}
        root_of = {}

        def sroot(r):
            u = r["uuid"]
            if u in root_of:
                return root_of[u]
            path, cur = [], r
            while True:
                path.append(cur["uuid"])
                p = byu.get(cur.get("parentUuid"))
                if p is None or not p.get("isSidechain"):
                    rt = cur["uuid"]
                    break
                if p["uuid"] in root_of:
                    rt = root_of[p["uuid"]]
                    break
                cur = p
            for x in path:
                root_of[x] = rt
            return rt

        comp = sorted(_iso_ms(r["timestamp"]) for r in recs if r.get("type") == "system" and r.get("subtype") == "compact_boundary")
        req = collections.OrderedDict()
        for r in recs:
            if r.get("type") == "assistant" and r.get("requestId"):
                sid = f"{conv}:{sroot(r)}" if r.get("isSidechain") else f"{conv}:main"
                req.setdefault((sid, r["requestId"]), []).append(r)
        if not req:
            n_no_llm += 1
            continue
        for (sid, rid), es in req.items():
            us = [e["message"]["usage"] for e in es]
            k3 = {(u.get("input_tokens", 0), u.get("cache_creation_input_tokens", 0), u.get("cache_read_input_tokens", 0)) for u in us}
            outs = [u.get("output_tokens", 0) for u in us]
            n_req += 1
            same_in += int(len(k3) == 1)
            mono_out += int(outs == sorted(outs))
            u = us[-1]
            inp = u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0)
            p = byu.get(es[0].get("parentUuid"))
            if p is None:
                n_parent_missing += 1
                t0, trig = _iso_ms(es[0]["timestamp"]), "none"
            else:
                t0 = _iso_ms(p["timestamp"])
                c = (p.get("message") or {}).get("content") if p.get("type") == "user" else None
                if isinstance(c, list) and any(isinstance(x, dict) and x.get("type") == "tool_result" for x in c):
                    trig = "tool"
                elif p.get("type") == "user":
                    trig = "user"
                else:
                    trig = "none"
            rows.append((conv, sid, rid, t0, _iso_ms(es[-1]["timestamp"]), inp, u.get("cache_read_input_tokens", 0), outs[-1], trig,
                         es[0]["message"].get("model")))
        comp_by_conv[conv] = comp
    r1a = _check("cachewise 同一 requestId 的輸入三欄相同", same_in, n_req)
    r1b = _check("cachewise 同一 requestId 的 output_tokens 不減", mono_out, n_req)
    d = pd.DataFrame(rows, columns=["conv", "sid", "rid", "start_ms", "end_ms", "inp", "prefix", "out", "trigger", "model"])
    # 順序＝檔案裡 requestId 的序號（去識別化後是每個檔從 req_00001 遞增，＝transcript 的先後）。不用 start_ms 排：
    # parent 節點的時間在 resume 的檔裡會被改寫，依時間排會把 128 個請求排錯（2026-11-04 驗算，見 STATUS_DATA）
    d["rnum"] = d.rid.str.extract(r"(\d+)$", expand=False).astype(np.int64)
    d = d.sort_values(["sid", "rnum"], kind="stable").reset_index(drop=True)
    g = d.groupby("sid", sort=False)
    prev_inp, prev_out, prev_end = g.inp.shift(), g.out.shift(), g.end_ms.shift()
    d["compact"] = [bool(pe == pe and any(pe <= x <= s for x in comp_by_conv[c])) for c, pe, s in zip(d.conv, prev_end, d.start_ms)]
    m = prev_inp.notna()
    r2 = _check("cachewise cache_read(k) ≤ inp(k−1)+out(k−1)", int((d.prefix[m] <= prev_inp[m] + prev_out[m]).sum()), int(m.sum()), thr=0.98)
    dur = d.end_ms - d.start_ms
    r3a = _check("cachewise end−start ≥ 0", int((dur >= 0).sum()), len(d), thr=0.95)
    q = d.out >= 50
    med = float((dur[q] / d.out[q]).median())
    print(f"[cachewise] 規則 6 (3)：(end−start)/out 中位數 {med:.2f} ms/token（out ≥ 50，n={int(q.sum())}）", flush=True)
    if not (1.0 <= med <= 1e4):
        raise SystemExit(f"[cachewise] (end−start)/out 中位數 {med} ms/token 不合理：時間欄不是毫秒？→ 停")
    gap = d.start_ms - prev_end
    m4 = m & (gap < 10_000) & ~d.compact & (prev_inp > 0)
    r4 = _check("cachewise 間隔<10s 時 cache_read(k) ≥ 0.9×inp(k−1)", int((d.prefix[m4] >= 0.9 * prev_inp[m4]).sum()), int(m4.sum()), thr=0.9)
    meta = dict(rule6=min(r1a, r1b), rule6_same_input=r1a, rule6_out_nondecreasing=r1b, rule6_prefix_le_prev=r2, rule6_dur_nonneg=r3a,
                rule6_ms_per_out_tok_median=med, rule6_struct_vs_cache_short_gap=r4, abs_time=False, multi_user=False,
                n_files=len(fs), n_empty_file=n_empty_file, n_file_no_llm=n_no_llm, n_conv=int(d.conv.nunique()), n_sess=int(d.sid.nunique()),
                n_sidechain_sess=int((~d.sid.str.endswith(":main")).groupby(d.sid).first().sum()), n_parent_missing=n_parent_missing,
                n_compact=int(d.compact.sum()))
    return d, meta


def load_bailian(name):
    """Qwen-Bailian（github alibaba-edu/qwen-bailian-usagetraces-anon @ 5f7439c5，Apache-2.0）：qwen_<name>_blksz_16.jsonl。
    欄位：chat_id、parent_chat_id、timestamp（秒，trace 相對）、input_length、output_length、type、turn、hash_ids（16 token 一塊、鏈式）。
    session＝沿 parent_chat_id 走到根（根的 chat_id）；prefix＝和 parent 共用的前導 hash 塊數 × 16（不超過 input_length）。
    hash_ids 的語意（上游 README「Anonymization Process」）：**每個 16-token 塊各自**做 salted SipHash-2-4 再重新編號——不是鏈式；
    同樣內容的塊出現在不同位置會是同一個 id。2026-11-04 traceA 的鏈式檢查違反 301,710 次就是這個原因（coder 剛好 0 次）。
    所以 chunk 鍵不能取第 32j+31 個 hash，要用整段前綴（next_b_convert.conv_bailian）。鏈式違反數只記在 meta，不再當停止條件。
    規則 6：
      (1) len(hash_ids)==ceil(input_length/16) ≥ 0.99（16-token 塊）
      (3) 時間單位：同 session 前後兩輪的間隔 ÷ 上一輪 output_length（output ≥ 50）的中位數，以秒解讀在 [0.001, 10] s/token
      (4) parent 欄位對 hash：有 parent 的請求，前導共用塊數 ≥ parent 的完整塊數的比例 ≥ 0.9（parent 的整段輸入是這一輪的前綴）
    """
    path = f"{DS}/qwen-bailian-usagetraces-anon/qwen_{name}_blksz_16.jsonl"
    recs = [json.loads(ln) for ln in open(path)]
    n = len(recs)
    ok1 = sum(len(r["hash_ids"]) == math.ceil(r["input_length"] / 16) for r in recs)
    r1 = _check(f"bailian-{name} len(hash_ids)=ceil(input/16)", ok1, n)
    pos_of, prev_of = {}, {}
    bad = 0
    for r in recs:
        p = -1
        for i, b in enumerate(r["hash_ids"]):
            if b in pos_of:
                bad += (pos_of[b] != i) + (prev_of[b] != p)
            else:
                pos_of[b], prev_of[b] = i, p
            p = b
    print(f"[bailian-{name}] 鏈式檢查違反 {bad} 次（逐塊 hash、不是鏈式，只記錄；見 docstring）", flush=True)
    idx = {r["chat_id"]: i for i, r in enumerate(recs)}
    rootc = {}
    sids, pref, par = [], [], []
    n_par, ok4, n_par_missing = 0, 0, 0
    for r in recs:
        pc = r.get("parent_chat_id")
        pi = idx.get(pc, -1) if pc not in (None, -1) else -1
        if pc not in (None, -1) and pi < 0:
            n_par_missing += 1
        root = rootc[recs[pi]["chat_id"]] if pi >= 0 else r["chat_id"]
        rootc[r["chat_id"]] = root
        sids.append(root)
        par.append(pi)
        if pi >= 0:
            ph, h = recs[pi]["hash_ids"], r["hash_ids"]
            k = 0
            while k < len(ph) and k < len(h) and ph[k] == h[k]:
                k += 1
            n_par += 1
            ok4 += int(k >= recs[pi]["input_length"] // 16)
            pref.append(min(k * 16, r["input_length"]))
        else:
            pref.append(0)
    r4 = _check(f"bailian-{name} 前導共用塊 ≥ parent 的完整塊", ok4, n_par, thr=0.9)
    d = pd.DataFrame(dict(sid=sids, chat_id=[r["chat_id"] for r in recs], parent=par, start_ms=[r["timestamp"] * 1000 for r in recs],
                          inp=[r["input_length"] for r in recs], out=[r["output_length"] for r in recs], prefix=pref,
                          turn=[r.get("turn") for r in recs]))
    d["end_ms"] = np.nan
    pm = d.parent >= 0
    gap_s = (d.start_ms[pm].values - d.start_ms.values[d.parent[pm].values]) / 1000
    pout = d.out.values[d.parent[pm].values]
    q = pout >= 50
    med = float(np.median(gap_s[q] / pout[q])) if q.any() else float("nan")
    print(f"[bailian-{name}] 規則 6 (3)：間隔÷上一輪輸出 中位數 {med:.4f} s/token（n={int(q.sum())}）", flush=True)
    if not (0.001 <= med <= 10):
        raise SystemExit(f"[bailian-{name}] 間隔÷輸出 中位數 {med} s/token 不合理：timestamp 不是秒？→ 停")
    return d, dict(rule6=r1, rule6_parent_prefix=r4, rule6_gap_s_per_out_tok_median=med, chain_bad=bad, abs_time=True, multi_user=True,
                   n_parent_missing=n_par_missing, n_sess=int(d.sid.nunique()))


def load_semianalysis():
    """主 agent 的請求一個 session；每個 subagent（巢狀的 requests）各自一個 session（它有自己的 context）。
    規則 6：上游 README 寫 `in`＝64-token hash 塊數 × 64 → 驗 len(hash_ids)×64＝in。"""
    rows, ok, n = [], 0, 0

    def add(sid, reqs, bs):
        nonlocal ok, n
        prev = set()
        for i, q in enumerate(reqs):
            n += 1
            h = q["hash_ids"]
            ok += int(len(h) * bs == q["in"])
            k = 0
            while k < len(h) and h[k] in prev:          # 前綴：連續命中同 session 上一個請求的塊數（hash_id_scope＝local）
                k += 1
            prev = set(h)
            t0 = q["t"] * 1000
            rows.append((sid, i, t0, t0 + q.get("api_time", np.nan) * 1000, q["in"], min(k * bs, q["in"]), q.get("type")))

    with open(f"{DS}/semianalysis_cc_traces/traces.jsonl") as fh:
        for ln in fh:
            r = json.loads(ln)
            bs = r["block_size"]
            main = [q for q in r["requests"] if q.get("type") != "subagent"]
            add(r["id"], main, bs)
            for q in r["requests"]:
                if q.get("type") == "subagent":
                    add(f"{r['id']}:{q['agent_id']}", q["requests"], bs)
    rule = _check("semianalysis len(hash_ids)×64=in", ok, n)
    d = pd.DataFrame(rows, columns=["sid", "i", "start_ms", "end_ms", "inp", "prefix", "type"])
    return d, dict(rule6=rule, abs_time=False, multi_user=False)


def load_exgentic():
    import pyarrow.parquet as pq
    rows, ok, n = [], 0, 0
    for f in sorted(glob.glob(f"{DS}/exgentic_agent_llm_traces/data/*.parquet")):
        for r in pq.read_table(f, columns=["session_id", "total_tokens", "spans"]).to_pylist():
            tot = 0
            for s in r["spans"] or []:
                a = s.get("attributes") or {}
                ti, to = a.get("gen_ai.usage.input_tokens"), a.get("gen_ai.usage.output_tokens")
                if ti is None or not s.get("start_time"):
                    continue
                tot += ti + (to or 0)
                rows.append((r["session_id"], _iso_ms(s["start_time"]), _iso_ms(s["end_time"]) if s.get("end_time") else np.nan, ti))
            n += 1
            ok += int(tot == r["total_tokens"])
    rule = _check("exgentic Σ(span input+output)=total_tokens", ok, n)
    d = pd.DataFrame(rows, columns=["sid", "start_ms", "end_ms", "inp"]).sort_values(["sid", "start_ms"])
    prev_in = d.groupby("sid").inp.shift()
    d["prefix"] = np.minimum(prev_in, d.inp)                         # 沒有 hash：上限＝上一個請求的輸入（〔假設〕前綴完全重用）
    return d, dict(rule6=rule, abs_time=False, multi_user=False)


def load_mooncake(name):
    rows, ok, n = [], 0, 0
    for ln in open(f"{DS}/traces/{name}_trace.jsonl"):
        r = json.loads(ln)
        n += 1
        ok += int(len(r["hash_ids"]) == math.ceil(r["input_length"] / 512))
        rows.append((r["timestamp"], r["input_length"]))
    rule = _check(f"mooncake-{name} len(hash_ids)=ceil(input/512)", ok, n)
    d = pd.DataFrame(rows, columns=["start_ms", "inp"]).assign(sid=np.nan, end_ms=np.nan, prefix=np.nan)
    return d, dict(rule6=rule, abs_time=True, multi_user=True, no_sessions=True)


def stats(name, d, meta):
    p = d.inp.astype(float)
    out = dict(trace=name, n_req=len(d), n_sess=int(d.sid.nunique()) if d.sid.notna().any() else None,
               inp_median=float(p.median()), inp_p90=float(p.quantile(.9)), inp_p99=float(p.quantile(.99)), inp_max=float(p.max()),
               frac_ge32k=float((p >= 32768).mean()), frac_ge100k=float((p >= 100000).mean()),
               kv_gib_at_median=float(p.median() * KIB_PER_TOKEN / 2**20), **{k: v for k, v in meta.items()})
    if d.sid.notna().any():
        d = d.sort_values(["sid", "start_ms"])
        prev_end = d.groupby("sid").end_ms.shift()
        prev_start = d.groupby("sid").start_ms.shift()
        gap = (d.start_ms - prev_end.where(prev_end.notna(), prev_start)) / 1000
        m = gap.notna() & d.prefix.notna()
        g, pre = gap[m].clip(lower=0), d.prefix[m].astype(float)
        out.update(n_returns=int(m.sum()), gap_s_median=float(g.median()), gap_s_p90=float(g.quantile(.9)), gap_s_p99=float(g.quantile(.99)),
                   prefix_median=float(pre.median()), kv_return_gib_median=float(pre.median() * KIB_PER_TOKEN / 2**20))
        for tier, bw in BW.items():
            t_restore = pre * KIB_PER_TOKEN / 2**20 / bw
            out[f"restore_s_median_{tier}"] = float(t_restore.median())
            out[f"frac_gap_ge_restore_{tier}"] = float((g >= t_restore).mean())
    if meta.get("abs_time") and meta.get("multi_user") and d.sid.notna().any():
        d = d.assign(win=(d.start_ms // WIN_MS))
        w = d.groupby(["win", "sid"]).inp.max().reset_index().groupby("win").agg(n=("sid", "size"), tok=("inp", "sum"))
        gib = w.tok * KIB_PER_TOKEN / 2**20
        out.update(active5m_median=float(w.n.median()), active5m_p90=float(w.n.quantile(.9)), active5m_max=int(w.n.max()),
                   active5m_kv_gib_median=float(gib.median()), active5m_kv_gib_p90=float(gib.quantile(.9)), active5m_kv_gib_max=float(gib.max()))
    return out


def cmd_characterize(a):
    os.makedirs(a.out, exist_ok=True)
    rid = os.environ.get("RUN_ID", datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc")
    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    res = []
    for name, fn in [("freeinference", load_freeinference), ("tracelab", load_tracelab), ("semianalysis_cc", load_semianalysis),
                     ("exgentic", load_exgentic), ("mooncake_conversation", lambda: load_mooncake("conversation")),
                     ("mooncake_toolagent", lambda: load_mooncake("toolagent"))]:
        d, meta = fn()
        s = stats(name, d, meta)
        res.append(dict(run_id=rid, ts=ts, **s))
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}, ensure_ascii=False), flush=True)
    pd.DataFrame(res).to_csv(os.path.join(a.out, "trace_stats.csv"), index=False)
    print(f"→ {os.path.join(a.out, 'trace_stats.csv')}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("characterize"); s.add_argument("--out", default=os.path.join(REPO, "results", "icpe_h200", "agent"))
    a = ap.parse_args()
    sys.exit({"characterize": cmd_characterize}[a.cmd](a))


if __name__ == "__main__":
    main()
