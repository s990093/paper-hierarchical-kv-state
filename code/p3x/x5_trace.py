"""x5_trace.py — X5：Mooncake trace 載入、單位斷言（規則 6）、共用前綴偵測、session 重建。

事先寫死的規則見 docs/phase3_20261017/explore/X5_precision_capacity.md §2.1：
  - hash_ids 一個＝512 token：len(hash_ids)==ceil(input_length/512) 吻合率 ≥ 0.95；前綴鏈 0 違反。
  - 共用前綴＝前綴樹上出度（不同的下一個 hash，含請求結束）≥ SHARED_OUTDEG 的節點與其祖先。
    斷言：出度落在 [GAP_LO, SHARED_OUTDEG) 的節點數＝0（探索時看到 16–1023 是空的）。
  - session＝請求的第一個非共用 block；session 存的東西＝最近一次請求的私有 block。
  - 回來請求＝同一 session 的第 2 個以後的請求；能讀到的 block 數 k_eff＝和上一個請求 hash 的共同前綴長度。
  - A 欄驗算 B 欄：「接在上一輪最後一個完整 block 後面」的回來請求，
    input_length(r) ≥ input_length(q)+output_length(q)−512 的比例 ≥ 0.8。

只用 CPU。讀 trace 的 Python 用 -I 執行（下載資料是不受信任的資料；這裡只 json.loads）。
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict

import numpy as np

TRACE_DIR = os.environ.get("X5_TRACE_DIR", os.path.join(os.environ.get("TIARA_DATASETS", "/ssd8/hungwei/paper-hkv/datasets"), "mooncake/traces"))
BLOCK_TOK = 512
SHARED_OUTDEG = 64
GAP_LO = 16
AB_MIN = 0.80
INF = float("inf")


STRICT_POP = 3   # 嚴格規則（README §4 第 2 點、research/calc/mooncake_chain2.py）：匹配到的前綴最多被 3 個請求共用、父請求唯一


def load(name: str, rule: str = "subtree") -> dict:
    """rule：
      subtree（事先登記）：session＝共用前綴下的子樹，回來＝同一 session 的下一個請求。
      strict（10:01Z 事後補充）：只有「最長匹配的完整前綴最多被 STRICT_POP 個請求包含、而且只有一個更早的候選父請求」
              才算接在父請求的 session 後面；其他請求自己開新 session（之前的匹配當作沒有，重算）。"""
    assert rule in ("subtree", "strict")
    path = os.path.join(TRACE_DIR, f"{name}_trace.jsonl")
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    # --- 規則 6：hash_ids 的單位
    match = sum(len(r["hash_ids"]) == math.ceil(r["input_length"] / BLOCK_TOK) for r in rows)
    unit_rate = match / n
    assert unit_rate >= 0.95, f"{name}: len(hash_ids)==ceil(input_length/512) 只有 {unit_rate:.4f}"
    ts = [r["timestamp"] for r in rows]
    assert ts == sorted(ts), f"{name}: timestamp 沒排序"
    # --- 前綴鏈
    pos_of, prev_of = {}, {}
    bad = 0
    for r in rows:
        h = r["hash_ids"]
        for i, b in enumerate(h):
            p = h[i - 1] if i else -1
            if b in pos_of:
                bad += (pos_of[b] != i) + (prev_of[b] != p)
            else:
                pos_of[b], prev_of[b] = i, p
    assert bad == 0, f"{name}: 前綴鏈違反 {bad} 次"
    # --- 共用前綴：出度
    child = defaultdict(set)
    for r in rows:
        h = r["hash_ids"]
        for i, b in enumerate(h):
            child[b].add(h[i + 1] if i + 1 < len(h) else None)
    outdeg = {b: len(c) for b, c in child.items()}
    gap_nodes = sum(GAP_LO <= d < SHARED_OUTDEG for d in outdeg.values())
    hi_nodes = [b for b, d in outdeg.items() if d >= SHARED_OUTDEG]
    assert gap_nodes == 0, f"{name}: 出度在 [{GAP_LO},{SHARED_OUTDEG}) 的節點有 {gap_nodes} 個，共用前綴門檻要重看"
    shared = set()
    for b in hi_nodes:
        x = b
        while x != -1 and x not in shared:
            shared.add(x)
            x = prev_of[x]
    # --- 每個請求
    L = np.array([r["input_length"] for r in rows], dtype=np.int64)
    O = np.array([r["output_length"] for r in rows], dtype=np.int64)
    T = np.array(ts, dtype=np.float64) / 1000.0           # 毫秒 → 秒（D4 §4.1 驗過是毫秒）
    nb = np.array([len(r["hash_ids"]) for r in rows], dtype=np.int64)
    s = np.zeros(n, dtype=np.int64)
    for ri, r in enumerate(rows):
        h = r["hash_ids"]
        k = 0
        while k < len(h) and h[k] in shared:
            k += 1
        s[ri] = k
    strict_parent = None
    if rule == "strict":
        nreq = defaultdict(int)
        for r in rows:
            for b in r["hash_ids"]:
                nreq[b] += 1
        cand = defaultdict(list)                           # 完整前綴最後一個 block 的 hash → 請求
        for ri, r in enumerate(rows):
            fb = r["input_length"] // BLOCK_TOK
            if fb >= 1:
                cand[r["hash_ids"][fb - 1]].append(ri)
        strict_parent = np.full(n, -1, dtype=np.int64)
        for rj, r in enumerate(rows):
            h = r["hash_ids"]
            for Lp in range(len(h) - 1, 0, -1):
                ps = [i for i in cand.get(h[Lp - 1], []) if i < rj and rows[i]["timestamp"] < r["timestamp"]]
                if ps:
                    if nreq[h[Lp - 1]] <= STRICT_POP and len(ps) == 1 and h[Lp - 1] not in shared:
                        strict_parent[rj] = ps[0]
                    break
    sid = np.full(n, -1, dtype=np.int64)
    prev = np.full(n, -1, dtype=np.int64)                  # 同 session 的上一個請求
    nxt = np.full(n, -1, dtype=np.int64)
    keff = np.zeros(n, dtype=np.int64)                     # 回來時能從 session 讀到的 block 數（含共用前綴）
    kseen = np.zeros(n, dtype=np.int64)                    # 和所有歷史比的匹配深度（block cache 的上界）
    root_sid, last_of = {}, {}
    seen = set()
    for ri, r in enumerate(rows):
        h = r["hash_ids"]
        k = 0
        while k < len(h) and h[k] in seen:
            k += 1
        kseen[ri] = k
        seen.update(h)
        if s[ri] >= len(h):                                # 整個請求都是共用前綴：沒有 session
            keff[ri] = s[ri]
            continue
        if rule == "strict":
            p = strict_parent[ri]
            if p >= 0 and sid[p] >= 0:
                j = sid[p]
            else:
                j = len(root_sid)
                root_sid[("r", ri)] = j
        else:
            root = h[s[ri]]
            j = root_sid.setdefault(root, len(root_sid))
        sid[ri] = j
        if j in last_of:
            q = last_of[j]
            prev[ri] = q
            nxt[q] = ri
            hq = rows[q]["hash_ids"]
            c = 0
            while c < min(len(h), len(hq)) and h[c] == hq[c]:
                c += 1
            keff[ri] = c
        else:
            keff[ri] = min(kseen[ri], s[ri])
        last_of[j] = ri
    ret = prev >= 0
    assert np.all(keff[ret] > s[ret]), "回來請求的 k_eff 應該 > s（session 的第一個 block 一樣）"
    # --- A 欄驗算 B 欄
    q = prev[ret]
    cont = keff[ret] == (L[q] // BLOCK_TOK)                 # 接在上一輪最後一個完整 block 後面
    ab_ok = L[ret][cont] >= L[q][cont] + O[q][cont] - BLOCK_TOK
    ab_rate = float(ab_ok.mean()) if cont.any() else float("nan")
    assert cont.sum() > 0 and ab_rate >= AB_MIN, f"{name}: A/B 驗算只有 {ab_rate:.3f}（n={cont.sum()}）"
    gap = np.where(ret, T - np.where(ret, T[np.maximum(prev, 0)], 0), np.nan)
    assert np.all(gap[ret] >= 0), "回來間隔不可以是負的"
    nret_per_sid = np.bincount(sid[ret], minlength=len(root_sid))
    meta = dict(trace=name, rule=rule, path=path, n_req=n, unit_match_rate=unit_rate, chain_bad=bad,
                n_distinct_blocks=len(pos_of), n_shared_blocks=len(shared), n_hi_outdeg_nodes=len(hi_nodes),
                gap_nodes=gap_nodes, n_sessions=len(root_sid), n_sessions_with_return=int((nret_per_sid > 0).sum()),
                n_returns=int(ret.sum()), n_no_session=int((sid < 0).sum()),
                ab_n_cont=int(cont.sum()), ab_rate=ab_rate, frac_returns_cont=float(cont.mean()),
                frac_returns_lost_reuse=float(np.mean(kseen[ret] > keff[ret])),
                lost_blocks_total=int((kseen[ret] - keff[ret]).clip(0).sum()),
                ts_span_s=float(T[-1] - T[0]), s_values=sorted(set(int(x) for x in s)),
                gap_s_p50=float(np.median(gap[ret])), gap_s_p90=float(np.percentile(gap[ret], 90)),
                keff_p50=float(np.median(keff[ret])), keff_p90=float(np.percentile(keff[ret], 90)),
                keff_max=int(keff[ret].max()), frac_ret_keff_gt64=float(np.mean(keff[ret] > 64)),
                input_len_p50=float(np.median(L)), input_len_max=int(L.max()))
    return dict(meta=meta, L=L, O=O, T=T, nb=nb, s=s, sid=sid, prev=prev, nxt=nxt, keff=keff, kseen=kseen,
                ret=ret, n_sess=len(root_sid))


def map_lengths(tr: dict, alpha: float = 4.0, lo: int = 16, hi: int = 64) -> dict:
    """只報告：把每個請求的 block 數乘 alpha、夾在 [lo, hi]（8K–32K）。共用前綴 s 不變；
    k_eff 照比例放大，夾在 (s, 上一個請求的 block 數]。L＝新的 block 數 × 512。"""
    out = dict(tr)
    nb = np.clip(np.round(tr["nb"] * alpha).astype(np.int64), lo, hi)
    nb = np.maximum(nb, tr["s"] + 1)
    keff = tr["keff"].copy()
    ret = tr["ret"]
    pk = nb[np.maximum(tr["prev"], 0)]
    ke = np.round(tr["keff"] * alpha).astype(np.int64)
    ke = np.minimum(np.minimum(ke, pk), nb)
    ke = np.maximum(ke, tr["s"] + 1)
    keff = np.where(ret, ke, np.minimum(tr["keff"], tr["s"]))
    out.update(nb=nb, keff=keff, L=nb * BLOCK_TOK)
    out["meta"] = dict(tr["meta"], mapped=dict(alpha=alpha, lo=lo, hi=hi))
    return out


if __name__ == "__main__":
    import sys
    for nm in sys.argv[1:] or ["conversation", "toolagent"]:
        d = load(nm)
        print(json.dumps(d["meta"], ensure_ascii=False, indent=1))
