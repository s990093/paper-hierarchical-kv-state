"""x1_trace.py — X1：Mooncake 載入、單位斷言（CLAUDE.md 規則 6）、session 重建、重建驗算。

被 x1_sim.py import；單獨執行（`char` 子命令）時輸出 trace 特性與重建驗算到 results/p3_explore/x1/。

單位與欄位的驗算（規則 6：用 A 欄驗算 B 欄，斷言寫在載入函式裡）：
  (1) len(hash_ids) == ceil(input_length/512) 的比例 ≥ 0.95（hash_ids 一個＝512 token）。
  (2) 前綴鏈：同一個 hash 永遠在同一個位置、前一個 hash 也相同（違反次數必須是 0）。
  (3) timestamp 單位：重建出的「同 session 前後兩輪」間隔 ÷ 上一輪 output_length 的中位數，
      以毫秒解讀必須落在 [1, 10000] ms/token（秒會是 ~0.3、微秒會是 ~3e5，都會被擋下）。

session 重建規則（寫死，不看任何策略結果）：
  - 每個請求只存「完整 block」：stored(r) = hash_ids[0 : floor(input_length/512)]。最後一個不滿 512 token 的 block
    在下一輪內容一定會變（接上這輪的輸出），所以不存（和 vLLM 只快取滿 block 一致）。
  - 候選父請求 p：r 的 hash_ids 從後往前（位置 k ≥ 1，不看 block 0＝全部共用的 system prompt）找第一個
    「某個更早請求的最後一個完整 block」；同一個 hash 有多個請求時取最近的（D4 §4.1 / m8_trace_oracle 的做法）。
  - 接續（cont）：p 還沒有被接續過，且 input_length(r) ≥ input_length(p) + output_length(p)
    （下一輪的輸入一定包含上一輪的輸入加輸出）。r 加入 p 的 session。
  - 分叉（fork）：有候選父請求但不符合上一條（p 已被接續、或長度不夠）→ r 開新 session，和 p 共用前綴。
  - 新（new）：沒有候選父請求。
  - 「回來的請求」＝ cont。
  以上是規則 R1（寬）。規則 R2（嚴，2026-10-17T10:05Z 依 explore/README §4 第 2 條加，還沒看任何策略結果）：
  R1 的接續再加兩個條件，只留「沒有歧義的單鏈」：
    (a) 整份 trace 裡，以 p 為「最深候選父」的請求只有 r 一個（p 沒有分叉、沒有重新生成）；
    (b) p 的最後一個完整 block 不是任何其他請求的最後一個完整 block（沒有完全相同的 prompt）。
  不符合的 r 當分叉（開新 session，前綴照樣共用）。
"""
from __future__ import annotations

import bisect
import json
import math
import os
from collections import defaultdict

import numpy as np

TRACE_DIR = os.environ.get("X1_TRACE_DIR", "/ssd8/hungwei/paper-hkv/datasets/mooncake/traces")
BLOCK_TOK = 512
INF = 1 << 62


def load_trace(name, trace_dir=TRACE_DIR):
    path = os.path.join(trace_dir, f"{name}_trace.jsonl")
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    match = sum(len(r["hash_ids"]) == math.ceil(r["input_length"] / BLOCK_TOK) for r in rows)
    rate = match / n
    # 規則 6 (1)：用 input_length 驗算 hash_ids 的單位
    assert rate >= 0.95, f"{name}: len(hash_ids)==ceil(input_length/512) 只有 {rate:.4f}，hash_ids 不是 512 token？"
    tss = [r["timestamp"] for r in rows]
    assert tss == sorted(tss), f"{name}: timestamp 沒排序"
    # 規則 6 (2)：前綴鏈
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
    assert bad_pos == 0 and bad_prev == 0, f"{name}: 前綴鏈被違反 pos={bad_pos} prev={bad_prev}"
    meta = dict(trace=name, path=path, n_req=n, unit_match_rate=rate, ts_min=tss[0], ts_max=tss[-1],
                chain_bad_pos=bad_pos, chain_bad_prev=bad_prev, max_input_len=max(r["input_length"] for r in rows))
    return rows, meta


def _cands(rows):
    """每個請求的最深候選父（R1、R2 共用）。"""
    lastfull = {}
    cand = np.full(len(rows), -1, np.int64)
    for ri, r in enumerate(rows):
        h = r["hash_ids"]
        for k in range(len(h) - 1, 0, -1):
            if h[k] in lastfull:
                cand[ri] = lastfull[h[k]]
                break
        nfull = r["input_length"] // BLOCK_TOK
        if nfull >= 2:
            lastfull[h[nfull - 1]] = ri
    return cand


def reconstruct(rows, rule="R1"):
    """回傳 dict：parent、kind（'new'/'cont'/'fork'）、sess（session id）、cand_parent（候選父，fork 也有）。"""
    n = len(rows)
    strict_ok = None
    if rule == "R2":
        c0 = _cands(rows)
        claim = np.bincount(c0[c0 >= 0], minlength=n)
        lf_cnt = defaultdict(int)
        for r in rows:
            nf = r["input_length"] // BLOCK_TOK
            if nf >= 2:
                lf_cnt[r["hash_ids"][nf - 1]] += 1
        strict_ok = np.zeros(n, bool)
        for pi, r in enumerate(rows):
            nf = r["input_length"] // BLOCK_TOK
            strict_ok[pi] = nf >= 2 and claim[pi] == 1 and lf_cnt[r["hash_ids"][nf - 1]] == 1
    else:
        assert rule == "R1", rule
    lastfull = {}                 # 最後一個完整 block 的 hash → 最近的請求 index
    cand = np.full(n, -1, np.int64)
    parent = np.full(n, -1, np.int64)
    kind = np.empty(n, dtype=object)
    sess = np.full(n, -1, np.int64)
    continued = np.zeros(n, bool)
    nsess = 0
    for ri, r in enumerate(rows):
        h = r["hash_ids"]
        par = -1
        for k in range(len(h) - 1, 0, -1):
            if h[k] in lastfull:
                par = lastfull[h[k]]
                break
        cand[ri] = par
        if par >= 0 and not continued[par] and \
                r["input_length"] >= rows[par]["input_length"] + rows[par]["output_length"] and \
                (strict_ok is None or strict_ok[par]):
            parent[ri] = par
            continued[par] = True
            kind[ri] = "cont"
            sess[ri] = sess[par]
        else:
            kind[ri] = "fork" if par >= 0 else "new"
            sess[ri] = nsess
            nsess += 1
        nfull = r["input_length"] // BLOCK_TOK
        if nfull >= 2:
            lastfull[h[nfull - 1]] = ri
    return dict(parent=parent, cand=cand, kind=kind, sess=sess, n_sess=nsess, rule=rule)


def unit_check_time(rows, R):
    """規則 6 (3)：間隔 ÷ 上一輪輸出 token 數（毫秒解讀）的中位數必須在 [1, 10000] ms/token。"""
    par = R["parent"]
    g, ratio = [], []
    for ri in np.nonzero(par >= 0)[0]:
        p = par[ri]
        gap = rows[ri]["timestamp"] - rows[p]["timestamp"]
        g.append(gap)
        if rows[p]["output_length"] >= 50:
            ratio.append(gap / rows[p]["output_length"])
    g, ratio = np.array(g, float), np.array(ratio, float)
    med = float(np.median(ratio)) if len(ratio) else float("nan")
    assert 1.0 <= med <= 1e4, f"間隔 ÷ 上一輪輸出 token 的中位數 {med}（ms 解讀）不合理：timestamp 不是毫秒？"
    return g, ratio, med


def prepare(name, trace_dir=TRACE_DIR, rule="R1"):
    """x1_sim 用：載入、重建、block 重新編號、每個 block 的存取清單（請求 index，遞增）。"""
    rows, meta = load_trace(name, trace_dir)
    R = reconstruct(rows, rule)
    gaps, ratio, med = unit_check_time(rows, R)
    remap = {}
    full = []      # 每個請求的完整 block（重新編號）
    allb = []      # 每個請求的全部 block（含最後不滿的那個；只用在 prefill 長度）
    for r in rows:
        ids = [remap.setdefault(h, len(remap)) for h in r["hash_ids"]]
        nf = r["input_length"] // BLOCK_TOK
        full.append(ids[:nf])
        allb.append(len(ids))
    B = len(remap)
    acc = defaultdict(list)
    for ri, ids in enumerate(full):
        for b in ids:
            acc[b].append(ri)
    t = np.array([r["timestamp"] / 1000.0 for r in rows])   # 秒（規則 6 (3) 驗過是毫秒）
    meta.update(n_blocks_all=B, n_blocks_full=len(acc), gap_ratio_ms_per_tok_median=med)
    return dict(rows=rows, meta=meta, R=R, full=full, nall=allb, B=B, acc=dict(acc), t=t)


def next_access(acc, b, after_req):
    """block b 在請求 after_req 之後第一次被讀的請求 index；沒有回 INF。"""
    lst = acc.get(b)
    if not lst:
        return INF
    i = bisect.bisect_right(lst, after_req)
    return lst[i] if i < len(lst) else INF


# ------------------------------------------------------------------ Bailian：用明確的 parent 欄位對拍重建規則
def bailian_check(path):
    """Qwen-Bailian（block＝16 token，有 chat_id／parent_chat_id）：把同一套前綴鏈規則換成 16-token block，
    和資料自帶的 parent_chat_id 比。只用來檢查重建規則，不進 X1 的模擬。"""
    rows = [json.loads(l) for l in open(path)]
    blk = 16
    match = sum(len(r["hash_ids"]) == math.ceil(r["input_length"] / blk) for r in rows)
    assert match / len(rows) >= 0.95, f"bailian: len(hash_ids)==ceil(input_length/16) 只有 {match/len(rows):.4f}"
    id2idx = {r["chat_id"]: i for i, r in enumerate(rows)}
    lastfull, continued = {}, set()
    tp = fp = fn = 0
    n_true = 0
    out_rows = []
    for ri, r in enumerate(rows):
        h = r["hash_ids"]
        par = -1
        for k in range(len(h) - 1, 0, -1):
            if h[k] in lastfull:
                par = lastfull[h[k]]
                break
        ok = par >= 0 and par not in continued and \
            r["input_length"] >= rows[par]["input_length"] + rows[par]["output_length"]
        if ok:
            continued.add(par)
        true_par = id2idx.get(r.get("parent_chat_id"), -1) if r.get("parent_chat_id", -1) not in (-1, None) else -1
        if true_par >= 0:
            n_true += 1
        pred = par if ok else -1
        if pred >= 0 and pred == true_par:
            tp += 1
        elif pred >= 0:
            fp += 1
        if true_par >= 0 and pred != true_par:
            fn += 1
        nfull = r["input_length"] // blk
        if nfull >= 2:
            lastfull[h[nfull - 1]] = ri
    return dict(n_req=len(rows), n_true_parent=n_true, tp=tp, fp=fp, fn=fn,
                precision=tp / max(tp + fp, 1), recall=tp / max(n_true, 1))


if __name__ == "__main__":
    import argparse
    import csv
    import sys
    from datetime import datetime

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["char"])
    ap.add_argument("--traces", nargs="+", default=["conversation", "toolagent"])
    ap.add_argument("--rules", nargs="+", default=["R1", "R2"])
    ap.add_argument("--bailian", nargs="*", default=[
        "/ssd8/hungwei/paper-hkv/datasets/qwen-bailian-usagetraces-anon/qwen_traceA_blksz_16.jsonl",
        "/ssd8/hungwei/paper-hkv/datasets/qwen-bailian-usagetraces-anon/qwen_coder_blksz_16.jsonl"])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                                  "results", "p3_explore", "x1", "x1_trace_char.csv"))
    a = ap.parse_args()
    rid = os.environ.get("RUN_ID", "adhoc")
    ts = lambda: datetime.now().astimezone().isoformat(timespec="seconds")
    new = not os.path.exists(a.out)
    f = open(a.out, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["run_id", "ts", "trace", "metric", "value", "note"])
    put = lambda tr, m, v, note="": (w.writerow([rid, ts(), tr, m, v, note]), print(tr, m, v, note, flush=True))
    for name, rule in [(n_, r_) for n_ in a.traces for r_ in a.rules]:
        P = prepare(name, rule=rule)
        rows, R, meta = P["rows"], P["R"], P["meta"]
        name = f"{name}|{rule}"
        for k in ("n_req", "unit_match_rate", "ts_min", "ts_max", "chain_bad_pos", "chain_bad_prev", "max_input_len",
                  "n_blocks_all", "n_blocks_full", "gap_ratio_ms_per_tok_median"):
            put(name, k, meta[k], "載入驗證（規則 6）")
        kind = R["kind"]
        n = len(rows)
        for kk in ("new", "cont", "fork"):
            put(name, f"n_{kk}", int(np.sum(kind == kk)))
        put(name, "n_sessions", R["n_sess"])
        cnt = np.bincount(R["sess"], minlength=R["n_sess"])
        put(name, "sessions_with_ge2_requests", int(np.sum(cnt >= 2)), "≥1 次回來＝獨立回來事件的 session 數")
        put(name, "requests_per_session_mean", float(cnt.mean()))
        put(name, "requests_per_session_max", int(cnt.max()))
        # 候選父但被長度條件擋掉的（fork 中有候選父、父還沒被接續）
        cand, par = R["cand"], R["parent"]
        cont_mask = par >= 0
        n_cand = int(np.sum(cand >= 0))
        put(name, "n_with_candidate_parent", n_cand)
        lenfail = 0
        for ri in np.nonzero((cand >= 0) & (par < 0))[0]:
            p = cand[ri]
            if rows[ri]["input_length"] < rows[p]["input_length"] + rows[p]["output_length"]:
                lenfail += 1
        put(name, "fork_due_to_length_rule", lenfail, "有候選父，但 input(r) < input(p)+output(p)")
        # 驗算：cont 的新增 token = input(r) − input(p) − output(p)
        add = np.array([rows[ri]["input_length"] - rows[par[ri]]["input_length"] - rows[par[ri]]["output_length"]
                        for ri in np.nonzero(cont_mask)[0]], float)
        for q in (10, 50, 90):
            put(name, f"cont_new_user_tokens_p{q}", float(np.percentile(add, q)), "input(r)−input(p)−output(p)")
        gaps, ratio, med = unit_check_time(rows, R)
        for q in (10, 25, 50, 75, 90):
            put(name, f"cont_gap_s_p{q}", float(np.percentile(gaps, q)) / 1000.0, "同 session 兩輪的到達間隔")
        put(name, "cont_gap_ms_per_parent_out_tok_median", med, "D4 §4.1 同法（父輸出 ≥50 token）")
        put(name, "frac_cont_gap_lt_5ms_per_out_tok", float(np.mean(ratio < 5.0)),
            "間隔短於 5 ms/輸出 token（>200 tok/s 解碼，物理上很可疑）")
        # 用 1 s 解析度驗：間隔為 0 的 cont 比例
        put(name, "frac_cont_gap_zero", float(np.mean(gaps == 0)), "timestamp 解析度約 1 s")
        # 長度
        nf = np.array([len(x) for x in P["full"]])
        put(name, "full_blocks_per_req_mean", float(nf.mean()))
        put(name, "full_blocks_per_req_p50", float(np.median(nf)))
        put(name, "full_blocks_per_req_max", int(nf.max()))
        put(name, "cont_full_blocks_mean", float(nf[cont_mask].mean()))
        put(name, "working_set_GiB_bf16", P["meta"]["n_blocks_full"] * 64 / 1024, "不同的完整 block × 64 MiB")
        # 共用前綴：每個請求的已存前綴中，來自 session 父請求以外的部分
        # （block 0 被全部共用；這裡量「重用的 block 有多少不是來自自己 session 的上一輪」）
        seen_sess = {}
        own = cross = 0
        for ri, ids in enumerate(P["full"]):
            s = R["sess"][ri]
            for b in ids:
                if b in seen_sess:
                    if s in seen_sess[b]:
                        own += 1
                    else:
                        cross += 1
                    seen_sess[b].add(s)
                else:
                    seen_sess[b] = {s}
        put(name, "reuse_accesses_same_session", own, "完整 block 的重用存取，之前被同一個 session 寫過")
        put(name, "reuse_accesses_cross_session", cross, "之前只被別的 session 寫過（共用前綴、分叉）")
        b0 = defaultdict(int)
        for ids in P["full"]:
            if ids:
                b0[ids[0]] += 1
        put(name, "distinct_full_block0", len(b0))
        put(name, "top_block0_share", max(b0.values()) / max(sum(b0.values()), 1))
    for p in a.bailian or []:
        if not os.path.exists(p):
            put(os.path.basename(p), "bailian_check", "NOT_MEASURED", "檔案不存在")
            continue
        res = bailian_check(p)
        for k, v in res.items():
            put(os.path.basename(p), f"bailian_{k}", v, "前綴鏈重建規則（16-token block）對 parent_chat_id")
    f.close()
