"""load_mooncake.py — Mooncake FAST'25 traces（kvcache-ai/Mooncake @ d750cdda，traces/{conversation,toolagent}_trace.jsonl）→ 統一格式。

Mooncake 沒有 session id；session 用前綴鏈重建。規格 §4 要求至少兩種規則（寬鬆／嚴格），各存一個 variant。
這裡存三種（variant 名稱 <trace>-rule-<規則>，trace＝conv／tool）：

  loose（寬鬆；照抄文獻 agent 的 research/calc/mooncake_chain.py）
      對每個請求 j，從 L＝len(h_j)−1 往下找「完整前綴（input_length//512 個 block）＝h_j[:L]」的較早請求 i（檔案順序 i<j，
      時間可以相同）；第一個找到的 L 就停，父＝這些候選裡最晚的那個。找不到 → 新 session 的第一輪。
      文獻 agent 量到「有子請求」的比例：conversation 43.9%、toolagent 44.1%。
  strict（嚴格；照抄 research/calc/mooncake_chain2.py）
      同上找第一個 L，但只有當 h_j[:L] 總共只被 ≤3 個請求含有、候選父唯一、且父的時間嚴格較早時才連；否則不連。
      文獻 agent：conversation 20.2%、toolagent 10.6%。
  d4（X2／X3 的 loader 用的規則，D4 cmd_char；為了和 X 系列對拍才存）
      父＝最後一個完整 block（位置 ≥1，完整 block 數 ≥2）出現在子請求 hash_ids 裡（由後往前找，位置 ≥1）的最近一個請求。

因為前綴鏈已驗過（同一個 hash 永遠在同一位置、前一個 hash 也一樣），「前綴 h[:L]」可以用 h[L−1] 代表，
「含有 h[:L] 的請求數」＝含有 block h[L−1] 的請求數。三個規則都可能分叉（一個父有多個子）→ 樹狀 session；
「下一輪」＝最早的子請求（schema.finalize）。

欄位對應
  t_start_ms ← timestamp（毫秒，trace 相對；斷言見下）
  t_end_ms ← t_start_ms + max(0, A + B×output_length + C×input_length)（estimated；常數見 service_constants.json）
  input/output_tokens ← input_length／output_length（trace 自己的 tokenizer）；block_ids ← hash_ids，block_tokens=512
  new_user_tokens ← max(0, input − 父 input − 父 output)；根＝input_length
  trigger ← unknown；user_id ← null
  obs_end_ms ← 檔案最大的 timestamp（conversation、toolagent 都是 3,536,999 ms＝58.9 分鐘）
  split ← §7 時間 60/20/20（session 開始時刻）

規則 6 的斷言（hard）
  len(hash_ids)==ceil(input_length/512) ≥ 0.95；timestamp 非遞減；前綴鏈：同一 hash 的位置與前一個 hash 都一致（0 個例外）；
  trace 跨度 ∈ [0.5, 2] 小時（README：約 1 小時；毫秒的驗算）；重建連結的 median(間隔秒／父 output token) ∈ 0.01..10
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

try:
    from . import schema as S
except ImportError:
    import schema as S

SOURCE = "mooncake"
DIR = os.path.join(S.DATASETS, "mooncake")
BLOCK = 512
TRACES = {"conv": "conversation", "tool": "toolagent"}
RULES = ("loose", "strict", "d4")
VARIANTS = [f"{t}-rule-{r}" for t in TRACES for r in RULES]
RULE_DOC = {
    "loose": "prefix-chain loose (research/calc/mooncake_chain.py): parent = latest earlier request whose full-block prefix == "
             "longest proper prefix of child (ties in time allowed)",
    "strict": "prefix-chain strict (research/calc/mooncake_chain2.py): as loose, but prefix shared by <=3 requests, unique "
              "candidate parent, parent strictly earlier; otherwise no link",
    "d4": "prefix-chain D4 (code/p3x/x2_load.py rule='d4'): parent = latest request whose last full block (pos>=1) appears in "
          "child hash_ids (pos>=1)",
}


def parents(rows, rule):
    n = len(rows)
    ts = [r["timestamp"] for r in rows]
    par = np.full(n, -1, dtype=np.int64)
    stats = {}
    if rule == "d4":
        lastfull = {}
        for j, r in enumerate(rows):
            h = r["hash_ids"]
            for k in range(len(h) - 1, 0, -1):
                if h[k] in lastfull:
                    par[j] = lastfull[h[k]]
                    break
            nf = r["input_length"] // BLOCK
            if nf >= 2:
                lastfull[h[nf - 1]] = j
        return par, stats
    pop = Counter(b for r in rows for b in r["hash_ids"])
    cand = defaultdict(list)  # 完整前綴的最後一個 block → 擁有這個完整前綴的請求（依檔案順序）
    for i, r in enumerate(rows):
        f = r["input_length"] // BLOCK
        if f >= 1:
            cand[r["hash_ids"][f - 1]].append(i)
    n_amb = n_multi = 0
    for j, r in enumerate(rows):
        h = r["hash_ids"]
        for L in range(len(h) - 1, 0, -1):
            lst = cand.get(h[L - 1])
            if not lst:
                continue
            if rule == "loose":
                ps = [i for i in lst if i < j]
                if ps:
                    par[j] = ps[-1]
                    n_multi += len(ps) > 1
                    break
            else:
                ps = [i for i in lst if i < j and ts[i] < ts[j]]
                if ps:
                    if pop[h[L - 1]] <= 3 and len(ps) == 1:
                        par[j] = ps[0]
                    elif len(ps) > 1:
                        n_amb += 1
                    break
    stats["children_with_multiple_candidate_parents"] = n_multi
    stats["ambiguous_not_linked"] = n_amb
    return par, stats


def load(variant, const):
    tkey, rule = variant.split("-rule-")
    name = TRACES[tkey]
    fn = f"traces/{name}_trace.jsonl"
    path = os.path.join(DIR, fn)
    ck = S.Checks(SOURCE, variant)
    rec = S.input_record(path, S.readme_sha256(os.path.join(DIR, "README.txt")).get(fn))
    ck.eq(f"sha256({fn})==README.txt", rec["sha256_match"], True)
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    n = len(rows)
    ck.ge("len(hash_ids)==ceil(input_length/512)", sum(len(r["hash_ids"]) == math.ceil(r["input_length"] / BLOCK) for r in rows) / n,
          0.95, "CLAUDE.md 規則 6：2026-08-31 踩過 16 vs 512")
    ts = np.array([r["timestamp"] for r in rows], dtype=np.int64)
    ck.eq("timestamp non-decreasing", bool(np.all(np.diff(ts) >= 0)), True)
    ck.rng("trace span hours", float((ts.max() - ts.min()) / 3.6e6), 0.5, 2.0, "README 約 1 小時；timestamp 是毫秒")
    pos_of, prev_of = {}, {}
    bad_pos = bad_prev = 0
    for r in rows:
        h = r["hash_ids"]
        for i, b in enumerate(h):
            p = h[i - 1] if i else -1
            if b in pos_of:
                bad_pos += pos_of[b] != i
                bad_prev += prev_of[b] != p
            else:
                pos_of[b], prev_of[b] = i, p
    ck.eq("prefix chain: same hash same position", bad_pos, 0)
    ck.eq("prefix chain: same hash same predecessor", bad_prev, 0)
    ck.report("requests with duplicate hash inside", sum(len(set(r["hash_ids"])) != len(r["hash_ids"]) for r in rows))
    ck.report("timestamp multiple of 1000 ms (frac)", float(np.mean(ts % 1000 == 0)), "解析度約 1 秒（D4）")
    ck.enforce()  # 單位／前綴鏈不對就不重建
    par, pst = parents(rows, rule)
    inp = np.array([r["input_length"] for r in rows], dtype=np.int64)
    out = np.array([r["output_length"] for r in rows], dtype=np.int64)
    c = np.nonzero(par >= 0)[0]
    p = par[c]
    g = (ts[c] - ts[p]) / 1000.0
    ok = out[p] >= 50
    ck.rng("median(gap_s/parent_output_tok) over links", float(np.median(g[ok] / out[p][ok])) if ok.any() else float("nan"),
           0.01, 10, "timestamp 是毫秒")
    has_child = np.zeros(n, bool)
    has_child[p] = True
    ck.report("frac requests with a child (returns)", float(has_child.mean()),
              "文獻 agent：loose 0.439／0.441、strict 0.202／0.106（conversation／toolagent）")
    ck.report("links with gap==0 (frac)", float(np.mean(g == 0)) if len(c) else 0.0)
    fan = np.bincount(p, minlength=n) if len(c) else np.zeros(n, int)
    ck.report("parent fanout>1 frac", float((fan[fan > 0] > 1).mean()) if len(c) else 0.0, f"max={int(fan.max()) if len(c) else 0}")
    ck.report("child input >= parent input + 0.9*parent output (frac)",
              float(np.mean(inp[c] >= inp[p] + 0.9 * out[p])) if len(c) else float("nan"),
              "X2 的一致性檢查（只報告；toolagent 在 X2 的 d4 規則下 0.79 被當成 BLOCKED）")
    for k, v in pst.items():
        ck.report(k, v)
    root = np.arange(n)
    for i in range(n):
        if par[i] >= 0:
            assert par[i] < i
            root[i] = root[par[i]]
    sizes = np.bincount(root, minlength=n)
    ck.report("max requests per session", int(sizes.max()), "大＝共用前綴把不同 session 併在一起")
    t_start = ts
    svc = np.round(np.maximum(0.0, const["A"] + const["B"] * out + const["C"] * inp)).astype(np.int64)
    nu = np.where(par >= 0, inp - inp[np.maximum(par, 0)] - out[np.maximum(par, 0)], inp)
    obs_end = int(ts.max())
    df = pd.DataFrame(dict(
        source=SOURCE, variant=variant, session_id=[f"{tkey}:{rule}:{r}" for r in root], user_id=None,
        session_rule=f"{rule}: {RULE_DOC[rule]}", trigger="unknown", t_start_ms=t_start,
        t_end_ms=pd.array(t_start + svc, dtype="Int64"), t_end_source="estimated",
        input_tokens=pd.array(inp, dtype="Int64"), output_tokens=pd.array(out, dtype="Int64"),
        cached_tokens=pd.array([pd.NA] * n, dtype="Int64"), tokenizer="mooncake(trace)",
        new_user_tokens=pd.array(np.maximum(nu, 0), dtype="Int64"), context_tokens=pd.array(inp, dtype="Int64"),
        block_ids=[r["hash_ids"] for r in rows], block_tokens=BLOCK, tool_name=None, tool_category=None,
        tool_latency_ms=pd.array([pd.NA] * n, dtype="Int64"), req_type=name, text_ref=[f"{fn}#line={i}" for i in range(n)],
        obs_end_ms=obs_end, t_start_source="trace", think_source="estimated", service_est_ms=pd.array(svc, dtype="Int64"),
        split_user=None, harness=None, is_subagent=None, parent_session_id=None, next_waited_for=None,
        provider_prompt_tokens=pd.array([pd.NA] * n, dtype="Int64"), parent_row=par, _order=np.arange(n)))
    df, fst = S.finalize(df, chain=False)
    ck.report("t_end_est clipped to next start", fst["t_end_est_clipped_to_next"])
    ck.report("t_end_est clipped to obs_end", fst["t_end_est_clipped_to_obs_end"])
    split, bounds, seg_end = S.time_split(df, int(ts.min()), obs_end)
    df["split"] = split
    sw = S.apply_split_window(df, seg_end)
    ck.report("split_drop rows (after boundary)", sw["split_drop_rows"])
    df = df.drop(columns=["parent_row", "_order"])
    extra = dict(trace=name, rule=rule, rule_doc=RULE_DOC[rule],
                 t_end_formula=f"t_end_ms = t_start_ms + max(0, {const['A']:.3f} + {const['B']:.5f}*output_tokens + "
                               f"{const['C']:.6f}*input_tokens)；超過下一輪開始或 obs_end 就截斷",
                 service_constants=dict(A=const["A"], B=const["B"], C=const["C"], run_id=const.get("run_id"), weeks=const["weeks"]),
                 obs_end_rule="檔案最大的 timestamp", split_rule="§7 時間 60/20/20（session 開始時刻）", split_bounds=bounds,
                 finalize=fst, rule_stats=pst, tier="R",
                 note="session 是重建的：「會不會回來」的標籤由規則決定（loose vs strict 差 2 倍以上），只適合當標籤雜訊的敏感度")
    ck.enforce()
    return df, ck, [rec], extra
