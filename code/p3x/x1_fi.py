"""x1_fi.py — X1 事後加跑：FreeInference agentic trace（explore/README §4-1；登記之後才加入）。

資料：HF harvardMadsys/freeinference_agentic_trace @ 16d3c04d（CC-BY-4.0），release/<週>/processed/requests.parquet。
每週一個獨立的重播（session 不跨週、block id 每週各自編號；資料 README）。

單位與換算（CLAUDE.md 規則 6；斷言寫在 load_week）：
  (1) len(block_ids) == floor(Σ role_tokens / 16) 的比例 ≥ 0.95（16-token、o200k_base、最後不滿一塊丟掉）。
  (2) 鏈式前綴：同一個 block id 永遠在同一個位置、前一個 id 也相同（違反次數必須是 0）。
  (3) 時間單位：end_ms − start_ms ≥ ttft_ms 的比例 ≥ 0.95（兩欄都是毫秒才會成立；ttft 中位數另外報）。
  換算成 Llama 的 512-token chunk：連續 32 個 16-token block 併成一個 chunk，chunk j 的 key＝block_ids[32j+31]
  （鏈式 id，最後一個就決定整段前綴）；完整 chunk 數＝floor(len(block_ids)/32)。o200k_base 的 512 token
  直接當成 Llama 的 512 token——兩個 tokenizer 的 token 數比值沒有量（NOT_MEASURED），這是近似。
  prefill 長度＝ceil(Σ role_tokens / 512) 個 chunk。
session：資料自帶的 session_id（子 agent 是自己的 session）。「回來的請求」＝同一個 session 的第 2 個以後的請求
  （依 start_ms 排序；工具觸發與使用者發起都算，另外分開報）。
"""
from __future__ import annotations

import glob
import os
from collections import defaultdict

import numpy as np

FI_DIR = os.environ.get("X1_FI_DIR", os.path.join(os.environ.get("TIARA_DATASETS", "/ssd8/hungwei/paper-hkv/datasets"), "freeinference_agentic_trace/release"))
SUB = 32       # 32 × 16 token = 512 token


def weeks():
    return sorted(os.path.basename(os.path.dirname(os.path.dirname(p)))
                  for p in glob.glob(os.path.join(FI_DIR, "*", "processed", "requests.parquet")))


def load_week(week):
    import pyarrow.parquet as pq
    path = os.path.join(FI_DIR, week, "processed", "requests.parquet")
    t = pq.read_table(path, columns=["session_id", "step", "start_ms", "end_ms", "ttft_ms", "step_trigger",
                                     "role_tokens", "block_ids"]).to_pandas()
    t = t.sort_values(["start_ms", "session_id", "step"], kind="stable").reset_index(drop=True)
    n = len(t)
    rt = np.array([sum(v for v in d.values() if v is not None) for d in t.role_tokens], np.int64)
    nb = np.array([len(x) for x in t.block_ids], np.int64)
    rate = float(np.mean(nb == rt // 16))
    assert rate >= 0.95, f"{week}: len(block_ids)==floor(Σrole_tokens/16) 只有 {rate:.4f}"
    pos_of, prev_of = {}, {}
    bad = 0
    for ids in t.block_ids:
        p = -1
        for i, b in enumerate(ids):
            b = int(b)
            if b in pos_of:
                bad += (pos_of[b] != i) + (prev_of[b] != p)
            else:
                pos_of[b], prev_of[b] = i, p
            p = b
    assert bad == 0, f"{week}: 鏈式前綴被違反 {bad} 次"
    ok_t = (t.end_ms - t.start_ms) >= t.ttft_ms
    trate = float(ok_t[t.ttft_ms.notna()].mean())
    assert trate >= 0.95, f"{week}: end_ms−start_ms ≥ ttft_ms 只有 {trate:.4f}，時間欄不是同一單位？"
    meta = dict(trace=f"fi:{week}", path=path, n_req=n, unit_match_rate=rate, chain_bad=bad, time_check_rate=trate,
                ttft_ms_median=float(t.ttft_ms.median()), n_sessions=int(t.session_id.nunique()))
    return t, rt, meta


def prepare_fi(week):
    """回傳和 x1_trace.prepare 同形的 dict。"""
    t, rt, meta = load_week(week)
    remap = {}
    full, nall = [], []
    for ids, tok in zip(t.block_ids, rt):
        ids = [int(x) for x in ids]
        nf = len(ids) // SUB
        keys = [remap.setdefault(ids[SUB * j + SUB - 1], len(remap)) for j in range(nf)]
        full.append(keys)
        nall.append(max(int(-(-int(tok) // 512)), nf))
    sid = {}
    sess = np.empty(len(t), np.int64)
    parent = np.full(len(t), -1, np.int64)
    kind = np.empty(len(t), dtype=object)
    lastreq = {}
    for i, s in enumerate(t.session_id):
        if s not in sid:
            sid[s] = len(sid)
        sess[i] = sid[s]
        if s in lastreq:
            parent[i] = lastreq[s]
            kind[i] = "cont"
        else:
            kind[i] = "new"
        lastreq[s] = i
    acc = defaultdict(list)
    for ri, ks in enumerate(full):
        for b in ks:
            acc[b].append(ri)
    t0 = float(t.start_ms.iloc[0])
    tt = (t.start_ms.values.astype(float) - t0) / 1000.0
    R = dict(parent=parent, cand=parent.copy(), kind=kind, sess=sess, n_sess=len(sid), rule="session_id")
    meta.update(n_blocks_full=len(acc), n_blocks_all=len(remap))
    trig = np.array([x == "user" for x in t.step_trigger])
    return dict(rows=None, meta=meta, R=R, full=full, nall=nall, B=len(remap), acc=dict(acc), t=tt,
                user_trigger=trig)
