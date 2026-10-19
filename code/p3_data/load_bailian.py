"""load_bailian.py — Qwen-Bailian anonymized usage traces（alibaba-edu/qwen-bailian-usagetraces-anon @ 5f7439c5）→ 統一格式。

variant＝trace 檔：traceA（To-C）、thinking、coder；traceB（To-B）沒有任何 parent 連結（X2／X3 已觀察），
在這裡也跑，但「有 session 結構」的斷言會不過 → 不寫 turns.parquet（不當成 session 資料，照實記）。

欄位對應
  session_id ← 沿 parent_chat_id 找到的根請求（"<variant>:<root chat_id>"）；session_rule＝官方 parent_chat_id 鏈
  user_id ← null（README：不保留跨 session 或使用者裝置的關聯）
  turn_idx ← session 內依 (timestamp, 檔案順序)；parent_turn_idx ← 父請求
  trigger ← unknown（trace 沒有 user／tool 區分；type＝text/search/image/file/thinking 放 req_type）
  t_start_ms ← round(timestamp × 1000)（timestamp 是 trace 相對秒）
  t_end_ms ← t_start_ms + service_est_ms，t_end_source=estimated；
     service_est_ms＝max(0, A + B×output_length + C×input_length)，A、B、C 見 results/p3_data/service_constants.json
     （FreeInference 訓練週的中位數迴歸）。估的 t_end 超過子請求開始就截到子請求開始（schema.finalize 計數）
  think_ms ← next_t_start_ms − t_end_ms（think_source=estimated）
  input/output_tokens ← input_length／output_length（Qwen 的 tokenizer，trace 自己的）；cached_tokens ← null
  new_user_tokens ← max(0, input − 父 input − 父 output)；根請求＝input_length（含系統提示，上界）
  block_ids ← hash_ids（16-token SipHash），block_tokens=16
  obs_end_ms ← 7,200,000（README：每條 2 小時；斷言 ts_max ∈ [7100, 7200] s）
  split ← §7 依 session 開始時刻：前 60%／中 20%／後 20%（0–4320 s／–5760 s／–7200 s）

規則 6 的斷言（hard）
  len(hash_ids)==ceil(input_length/16) ≥ 0.95；timestamp 非遞減；chat_id 唯一；父請求都找得到；
  子 turn==父 turn+1 ≥ 0.99；子 hash 前綴==父 hash[:-1] ≥ 0.95；子 timestamp ≥ 父 timestamp；
  median(間隔秒／父 output token) ∈ 0.01..10（timestamp 是秒）；frac(間隔 < 1 ms × 父 output) < 0.05；有子請求（session 結構）

只報告（不擋；2026-10-18，06_simulator.md §10-2）：跨請求的原始 id 鏈性質（raw_id_chain_report）。上面的「子 hash 前綴＝父 hash[:-1]」
只看父子之間；跨請求「同一個 16-token id 永遠在同一個位置、前一個 id 也相同」沒有保證：traceA 不成立（52,168 個 id 出現在
不只一個位置），coder／thinking 成立。用 hash 的最後一個 id 代表整段前綴的程式（x1_fi 的寫法）在 traceA 會出錯。
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd

try:
    from . import schema as S
except ImportError:
    import schema as S

SOURCE = "bailian"
DIR = os.path.join(S.DATASETS, "qwen-bailian-usagetraces-anon")
VARIANTS = ["traceA", "thinking", "coder", "traceB"]
OBS_END_S = 7200.0


def raw_id_chain_report(id_lists):
    """跨請求的原始 id 鏈檢查（只報告）。依給的順序掃：每個 id 第一次出現的位置與前一個 id 當基準，
    之後每次出現位置不同 → 位置違反＋1；前一個 id 不同 → 前一個 id 違反＋1。
    回傳 dict(n_ids, n_ids_multi_position, position_violations, prev_id_violations)。和 p3_sim/trace.py 的算法相同。"""
    pos, prv, multi = {}, {}, set()
    pv = qv = 0
    for ids in id_lists:
        p_ = None
        for j, b in enumerate(ids):
            b = int(b)
            if b in pos:
                if pos[b] != j:
                    pv += 1
                    multi.add(b)
                qv += prv[b] != p_
            else:
                pos[b], prv[b] = j, p_
            p_ = b
    return dict(n_ids=len(pos), n_ids_multi_position=len(multi), position_violations=pv, prev_id_violations=qv)


def service_est(out_tok, in_tok, const):
    return np.maximum(0.0, const["A"] + const["B"] * np.asarray(out_tok, float) + const["C"] * np.asarray(in_tok, float))


def load(variant, const):
    fn = f"qwen_{variant}_blksz_16.jsonl"
    path = os.path.join(DIR, fn)
    exp = S.readme_sha256(os.path.join(DIR, "README.txt")).get(fn)
    ck = S.Checks(SOURCE, variant)
    rec = S.input_record(path, exp)
    ck.eq(f"sha256({fn})==README.txt", rec["sha256_match"], True)
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    n = len(rows)
    ck.ge("len(hash_ids)==ceil(input_length/16)", sum(len(r["hash_ids"]) == math.ceil(r["input_length"] / 16) for r in rows) / n,
          0.95, "README：hash_ids 一個＝16 token")
    ts = np.array([r["timestamp"] for r in rows], dtype=float)
    ck.eq("timestamp non-decreasing", bool(np.all(np.diff(ts) >= 0)), True)
    ck.rng("ts_max seconds (README: 2 hours)", float(ts.max()), 7100.0, OBS_END_S, "timestamp 是秒：2 小時＝7200")
    idx = {r["chat_id"]: i for i, r in enumerate(rows)}
    ck.eq("chat_id unique", len(idx), n)
    par = np.array([-1 if r["parent_chat_id"] == -1 else idx.get(r["parent_chat_id"], -2) for r in rows], dtype=np.int64)
    ck.eq("parent_chat_id found", int((par == -2).sum()), 0)
    c = np.nonzero(par >= 0)[0]
    ck.ge("n_child_requests (has session structure)", int(len(c)), 1, "traceB 全部 parent_chat_id=-1 → 不當成 session 資料")
    inp = np.array([r["input_length"] for r in rows], dtype=np.int64)
    out = np.array([r["output_length"] for r in rows], dtype=np.int64)
    turn = np.array([r["turn"] for r in rows], dtype=np.int64)
    if len(c):
        p = par[c]
        ck.ge("child turn==parent turn+1", float(np.mean(turn[c] == turn[p] + 1)), 0.99)
        ck.ge("child hash prefix==parent hash[:-1]",
              float(np.mean([rows[i]["hash_ids"][:len(rows[j]["hash_ids"]) - 1] == rows[j]["hash_ids"][:-1] for i, j in zip(c, p)])),
              0.95)
        ck.eq("child ts>=parent ts", float(np.mean(ts[c] >= ts[p])), 1.0)
        g = ts[c] - ts[p]
        ok = out[p] >= 50
        rr = g[ok] / out[p][ok]
        ck.rng("median(gap_s/parent_output_tok)", float(np.median(rr)) if ok.any() else float("nan"), 0.01, 10,
               "timestamp 是秒（毫秒會是 ~300）")
        ck.add("frac(gap < 1ms*parent_output)", float(np.mean(rr < 0.001)) if ok.any() else float("nan"), "<0.05",
               bool(ok.any() and np.mean(rr < 0.001) < 0.05), "≥1000 token/s 的解碼＝單位不對")
        fan = np.bincount(p, minlength=n)
        ck.report("parent fanout>1 frac", float((fan[fan > 0] > 1).mean()))
        nu_raw = inp[c] - inp[p] - out[p]
        ck.report("child input - parent input - parent output < 0 (frac)", float(np.mean(nu_raw < 0)), "thinking 會去掉推理內容")
    roots = par < 0
    ck.report("root turn==1 frac", float(np.mean(turn[roots] == 1)))
    rc = raw_id_chain_report(r["hash_ids"] for r in rows)
    ck.report("raw id cross-request chain: ids at >1 position (report)", rc["n_ids_multi_position"],
              "06_simulator §10-2：traceA 不是鏈式；不要用最後一個 hash 代表整段前綴")
    ck.report("raw id cross-request chain: position violations (report)", rc["position_violations"])
    ck.report("raw id cross-request chain: prev-id violations (report)", rc["prev_id_violations"])
    ck.enforce()  # 結構斷言不過就停在這裡（traceB）
    # 根
    root = np.arange(n)
    for i in range(n):  # 檔案依時間排序、父一定在前
        if par[i] >= 0:
            assert par[i] < i, "父請求在子請求之後（檔案順序）"
            root[i] = root[par[i]]
    depth = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if par[i] >= 0:
            depth[i] = depth[par[i]] + 1
    ck.ge("turn - root_turn == chain depth", float(np.mean(turn - turn[root] == depth)), 0.99, "規格 §4：turn 和鏈的深度一致")
    t_start = np.round(ts * 1000).astype(np.int64)
    svc = np.round(service_est(out, inp, const)).astype(np.int64)
    nu = np.where(par >= 0, inp - inp[np.maximum(par, 0)] - out[np.maximum(par, 0)], inp)
    df = pd.DataFrame(dict(
        source=SOURCE, variant=variant, session_id=[f"{variant}:{rows[r]['chat_id']}" for r in root], user_id=None,
        session_rule="official parent_chat_id chain", trigger="unknown", t_start_ms=t_start,
        t_end_ms=pd.array(t_start + svc, dtype="Int64"), t_end_source="estimated",
        input_tokens=pd.array(inp, dtype="Int64"), output_tokens=pd.array(out, dtype="Int64"),
        cached_tokens=pd.array([pd.NA] * n, dtype="Int64"), tokenizer="qwen(trace)",
        new_user_tokens=pd.array(np.maximum(nu, 0), dtype="Int64"), context_tokens=pd.array(inp, dtype="Int64"),
        block_ids=[r["hash_ids"] for r in rows], block_tokens=16, tool_name=None, tool_category=None,
        tool_latency_ms=pd.array([pd.NA] * n, dtype="Int64"), req_type=[r["type"] for r in rows], text_ref=[f"{fn}#line={i}" for i in range(n)],
        obs_end_ms=int(OBS_END_S * 1000), t_start_source="trace", think_source="estimated",
        service_est_ms=pd.array(svc, dtype="Int64"), split_user=None, harness=None, is_subagent=None,
        parent_session_id=None, next_waited_for=None, provider_prompt_tokens=pd.array([pd.NA] * n, dtype="Int64"),
        parent_row=par, _order=np.arange(n)))
    df, fst = S.finalize(df, chain=False)
    ck.report("t_end_est clipped to next start", fst["t_end_est_clipped_to_next"], "估的服務時間比觀察到的間隔長")
    ck.report("t_end_est clipped to obs_end", fst["t_end_est_clipped_to_obs_end"])
    split, bounds, seg_end = S.time_split(df, 0, int(OBS_END_S * 1000))
    df["split"] = split
    sw = S.apply_split_window(df, seg_end)
    ck.report("split_drop rows (after boundary)", sw["split_drop_rows"])
    df = df.drop(columns=["parent_row", "_order"])
    extra = dict(t_end_formula=f"t_end_ms = t_start_ms + max(0, {const['A']:.3f} + {const['B']:.5f}*output_tokens + "
                               f"{const['C']:.6f}*input_tokens)；超過子請求開始或 obs_end 就截斷",
                 service_constants=dict(A=const["A"], B=const["B"], C=const["C"], run_id=const.get("run_id"), weeks=const["weeks"]),
                 obs_end_rule="README：2 小時 → 7,200,000 ms", split_rule="§7 時間 60/20/20（session 開始時刻）",
                 split_bounds=bounds, finalize=fst, tier="R",
                 new_user_tokens_rule="max(0, input − parent.input − parent.output)；根＝input_length")
    ck.enforce()
    return df, ck, [rec], extra
