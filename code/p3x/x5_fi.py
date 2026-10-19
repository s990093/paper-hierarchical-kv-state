"""x5_fi.py — X5：FreeInference agentic trace（README §4 事後補充，登記之後才加；只報告＋第二個判定版本）。

資料：/ssd8/hungwei/paper-hkv/datasets/freeinference_agentic_trace（HF harvardMadsys/freeinference_agentic_trace @ 16d3c04d）。
選週（看任何 FreeInference 結果之前寫死）：照時間順序第一個 session 數 ≥ 1,000 的週＝2026-07-26_2026-08-01（1,198 個 session）。
單位（規則 6）：block_ids 一個＝16 token（o200k_base）；斷言 len(block_ids)==floor(sum(role_tokens)/16) 吻合率 ≥ 0.95。
轉成 512-token block：L＝sum(role_tokens)，nb＝ceil(L/512)；回來能讀的 k_eff＝和同一 session 上一個請求的
  16-token 共同前綴 ÷ 32（取整，只算完整的 512-token block）。
session＝session_id（資料自帶）。共用前綴：每個 session 第一個請求的前 2,048 個 16-token block 裡，被 ≥ 64 個 session
  共用的連續開頭（位置相同的 block id）→ s＝那段長度 ÷ 32。共用前綴的 block 常駐 CPU BF16（同 Mooncake）。
A 欄驗算 B 欄：接在上一個請求最後一個完整 512-block 後面的回來請求，L(r) ≥ L(q)+completion_tokens(q)−512 的比例 ≥ 0.8。
"""
from __future__ import annotations

import os

import numpy as np
import pyarrow.parquet as pq

FI_DIR = "/ssd8/hungwei/paper-hkv/datasets/freeinference_agentic_trace/release"
WEEK = "2026-07-26_2026-08-01"
B16 = 16
PER512 = 32
SHARED_SESS = 64
AB_MIN = 0.80


def load(week: str = WEEK) -> dict:
    path = os.path.join(FI_DIR, week, "processed", "requests.parquet")
    t = pq.read_table(path, columns=["session_id", "step", "start_ms", "prompt_tokens", "completion_tokens",
                                     "role_tokens", "block_ids"])
    sess = t.column("session_id").to_pylist()
    step = np.array(t.column("step").to_pylist())
    start = np.array(t.column("start_ms").to_pylist(), dtype=np.float64)
    comp = np.array(t.column("completion_tokens").to_pylist(), dtype=np.int64)
    rt = t.column("role_tokens").to_pylist()
    L = np.array([sum(v for v in d.values() if v is not None) for d in rt], dtype=np.int64)
    bl = t.column("block_ids").combine_chunks()
    offs = np.asarray(bl.offsets)
    vals = np.asarray(bl.values)
    nb16 = np.diff(offs)
    unit = float(np.mean(nb16 == L // B16))
    assert unit >= 0.95, f"FreeInference: len(block_ids)==floor(L/16) 只有 {unit:.4f}"
    n = len(sess)
    order = np.lexsort((step, start))                     # 依開始時間（同時間依 step）
    # session id
    uniq = {}
    sid_raw = np.array([uniq.setdefault(x, len(uniq)) for x in sess], dtype=np.int64)
    # 共用前綴：每個 session 的第一個請求
    first = {}
    for i in order:
        first.setdefault(sid_raw[i], i)
    M = 2048
    pos_ids = {}
    for j, i in first.items():
        a = vals[offs[i]:offs[i] + min(M, nb16[i])]
        for p, b in enumerate(a):
            pos_ids.setdefault((p, int(b)), set()).add(j)
    shared16 = {k for k, v in pos_ids.items() if len(v) >= SHARED_SESS}

    def shared_depth(i):
        a = vals[offs[i]:offs[i] + min(M, nb16[i])]
        d = 0
        while d < len(a) and (d, int(a[d])) in shared16:
            d += 1
        return d // PER512

    # 依時間排序後的陣列
    T = start[order] / 1000.0
    Lr = L[order]
    O = comp[order]
    sidr = sid_raw[order]
    nb = -(-Lr // 512)
    s = np.array([shared_depth(i) for i in order], dtype=np.int64)
    s = np.minimum(s, nb - 1).clip(0)
    prev = np.full(n, -1, dtype=np.int64)
    nxt = np.full(n, -1, dtype=np.int64)
    keff = np.zeros(n, dtype=np.int64)
    last_of = {}
    for r_, i in enumerate(order):
        j = sidr[r_]
        if j in last_of:
            q_, iq = last_of[j]
            prev[r_] = q_
            nxt[q_] = r_
            a = vals[offs[i]:offs[i + 1]]
            b = vals[offs[iq]:offs[iq + 1]]
            m = min(len(a), len(b))
            neq = np.nonzero(a[:m] != b[:m])[0]
            c16 = int(neq[0]) if len(neq) else m
            keff[r_] = max(c16 // PER512, s[r_])
        else:
            keff[r_] = s[r_]
        last_of[j] = (r_, i)
    ret = prev >= 0
    # 有些回來請求和上一個完全沒有共同的私有前綴（例如 context 被壓縮、重寫）：k_eff==s，照樣算回來（還原 0 個私有 block）
    q = prev[ret]
    cont = keff[ret] == (Lr[q] // 512)
    ab = Lr[ret][cont] >= Lr[q][cont] + O[q][cont] - 512
    ab_rate = float(ab.mean()) if cont.any() else float("nan")
    assert cont.sum() > 0 and ab_rate >= AB_MIN, f"FreeInference A/B 驗算 {ab_rate:.3f}（n={cont.sum()}）"
    gap = T[ret] - T[q]
    assert np.all(gap >= 0)
    keys = set()
    for r_, i in enumerate(order):
        if s[r_] > 0:
            keys.add((int(s[r_]), int(vals[offs[i] + s[r_] * PER512 - 1])))
    n_sh = int(sum(k[0] for k in keys))                    # 共用前綴的 512-block 數（巢狀的前綴重複算，偏大）
    nret = np.bincount(sidr[ret], minlength=len(uniq))
    meta = dict(trace=f"freeinference_{week}", rule="session_id", path=path, n_req=n, unit_match_rate=unit,
                chain_bad=0, n_distinct_blocks=0, n_shared_blocks=n_sh, n_hi_outdeg_nodes=len(shared16), gap_nodes=0,
                n_sessions=len(uniq), n_sessions_with_return=int((nret > 0).sum()), n_returns=int(ret.sum()),
                n_no_session=0, ab_n_cont=int(cont.sum()), ab_rate=ab_rate, frac_returns_cont=float(cont.mean()),
                frac_returns_lost_reuse=float(np.mean(keff[ret] <= s[ret])), lost_blocks_total=0,
                ts_span_s=float(T[-1] - T[0]), s_values=sorted(set(int(x) for x in s)),
                gap_s_p50=float(np.median(gap)), gap_s_p90=float(np.percentile(gap, 90)),
                keff_p50=float(np.median(keff[ret])), keff_p90=float(np.percentile(keff[ret], 90)),
                keff_max=int(keff[ret].max()), frac_ret_keff_gt64=float(np.mean(keff[ret] > 64)),
                frac_ret_keff_gt256=float(np.mean(keff[ret] > 256)),
                input_len_p50=float(np.median(Lr)), input_len_max=int(Lr.max()), nb_max=int(nb.max()))
    sid = sidr.copy()
    return dict(meta=meta, L=Lr, O=O, T=T, nb=nb, s=s, sid=sid, prev=prev, nxt=nxt, keff=keff, kseen=keff.copy(),
                ret=ret, n_sess=len(uniq))


if __name__ == "__main__":
    import json
    d = load()
    print(json.dumps(d["meta"], ensure_ascii=False, indent=1))
