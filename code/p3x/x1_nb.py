"""x1_nb.py — ICPE next：icpe-data 轉好的新紀錄（TraceLab、CacheWise、SemiAnalysis、Bailian…）給 X1 模擬器吃的檔案格式。

x1_sim.prep("nb:<name>") 讀 $X1_NB_DIR/<name>.npz（np.load allow_pickle=False；不執行檔案裡的任何東西）。
icpe-data 用 write_nb(...) 寫（寫之前跑同一套檢查）。名稱建議帶上到達方式與種子，例：cachewise_N16_q1、tracelab_c2x。

格式（npz，全部是數值陣列＋一個 JSON 字串）：
  t          float64[n]   請求到達時間（秒；非遞減）
  sess       int64[n]     session id（0..S−1；同一個對話＝同一個 id；複本要換新 id）
  parent     int64[n]     同一個 session 的上一個請求的 index（第一個請求＝−1）；必須 < 自己
  kind       int8[n]      0＝new（session 的第一個請求）、1＝cont（回來的請求；parent ≥ 0）、2＝fork（開新 session、和別人共用前綴；parent＝−1）
  nall       int64[n]     prefill 的 512-token chunk 數（含最後不滿的那塊；＝ceil(總輸入 token / 512)）；≥ 這個請求的完整 chunk 數
  full_off   int64[n+1]   full_key 的切點：請求 i 的完整 chunk＝full_key[full_off[i]:full_off[i+1]]
  full_key   int64[m]     完整 512-token chunk 的鍵（鏈式：同一個鍵永遠在同一個位置、前一個鍵也一樣＝同一段前綴）
  user_trigger bool[n]    （選填）使用者發起的請求（x1_sim summarize 會另外報 mean_ret_user）
  meta_json  0 維字串     JSON：trace、來源、轉換 run_id、單位驗算結果等（原樣進 meta）
「回來的請求」＝kind 1（x1_sim 只對它算 TTFT 的平均）。和 x1_fi.prepare_fi 回傳同一種 dict。
"""
from __future__ import annotations

import json
import os
import numpy as np

NB_DIR = os.environ.get("X1_NB_DIR", "/ssd6/hungwei/icpe-next/datasets/x1_nb")
KIND = {0: "new", 1: "cont", 2: "fork"}


def check(t, sess, parent, kind, nall, full_off, full_key, name="nb"):
    """格式與一致性檢查；任一條不過 → AssertionError（寫檔前、讀檔後都跑）。全部用 numpy 向量化（TraceLab 有上億個 chunk）。
    回傳 (統計 dict, 稠密 id（照第一次出現的順序編號）)。"""
    n = len(t)
    for nm, x in (("sess", sess), ("parent", parent), ("kind", kind), ("nall", nall)):
        assert len(x) == n, f"{name}: {nm} 長度 {len(x)} ≠ {n}"
    assert len(full_off) == n + 1 and full_off[0] == 0 and full_off[-1] == len(full_key), f"{name}: full_off 不對"
    nf = np.diff(full_off)
    assert (nf >= 0).all(), f"{name}: full_off 不是非遞減"
    assert np.all(np.isfinite(t)) and (np.diff(t) >= 0).all(), f"{name}: t 不是非遞減的有限值"
    assert set(np.unique(kind).tolist()) <= {0, 1, 2}, f"{name}: kind 只能是 0／1／2"
    assert (nall >= nf).all(), f"{name}: nall < 完整 chunk 數的請求 {int((nall < nf).sum())} 個"
    idx = np.arange(n)
    cont = kind == 1
    assert ((parent >= 0) == cont).all(), f"{name}: kind＝cont ⇔ parent ≥ 0 不成立（{int(((parent >= 0) != cont).sum())} 個）"
    p = parent[cont]
    assert (p < idx[cont]).all(), f"{name}: parent 不在自己之前"
    assert (sess[p] == sess[cont]).all(), f"{name}: parent 和自己不是同一個 session"
    first = np.zeros(n, bool)
    if n:
        first[np.unique(sess, return_index=True)[1]] = True
    bad_first = int((first != (kind != 1)).sum())    # new、fork 都是 session 的第一個請求；cont 一定不是
    assert bad_first == 0, f"{name}: 「session 的第一個請求 ⇔ kind≠cont」不成立 {bad_first} 個"
    # 稠密 id（第一次出現的順序，和逐一 setdefault 的編號相同）
    m = len(full_key)
    if m:
        uq, fi, inv = np.unique(full_key, return_index=True, return_inverse=True)
        order = np.argsort(fi, kind="stable")
        rank = np.empty(len(order), np.int64)
        rank[order] = np.arange(len(order))
        ids = rank[inv.reshape(-1)]
        del uq, fi, inv, order, rank
    else:
        ids = np.zeros(0, np.int64)
    # 鏈式前綴（同 x1_trace 規則 6 (2)）：同一個鍵永遠在同一個位置、前一個鍵也一樣
    pos = np.arange(m) - np.repeat(full_off[:-1], nf)
    pred = np.full(m, -1, np.int64)
    if m:
        pred[1:] = ids[:-1]
        pred[pos == 0] = -1
    B = int(ids.max()) + 1 if m else 0
    pos_ref = np.empty(B, np.int64)
    pred_ref = np.empty(B, np.int64)
    pos_ref[ids] = pos
    pred_ref[ids] = pred
    bad = int((pos_ref[ids] != pos).sum())
    del pos, pos_ref
    bad += int((pred_ref[ids] != pred).sum())
    del pred, pred_ref
    assert bad == 0, f"{name}: 鏈式前綴被違反 {bad} 次"
    st = dict(n_req=n, n_sessions=int(len(np.unique(sess))), n_cont=int(cont.sum()), n_fork=int((kind == 2).sum()),
              n_keys=B, chain_bad=bad, full_chunks_total=int(m))
    return st, ids


def write_nb(path, t, sess, parent, kind, nall, full, user_trigger=None, meta=None):
    """full＝每個請求的完整 chunk 鍵清單（list of list of int）。先檢查再寫。"""
    t = np.asarray(t, np.float64)
    sess, parent, nall = (np.asarray(x, np.int64) for x in (sess, parent, nall))
    kind = np.asarray(kind, np.int8)
    full_off = np.zeros(len(full) + 1, np.int64)
    full_off[1:] = np.cumsum([len(x) for x in full])
    full_key = np.fromiter((k for x in full for k in x), np.int64, count=int(full_off[-1]))
    st, _ = check(t, sess, parent, kind, nall, full_off, full_key, os.path.basename(path))
    arrs = dict(t=t, sess=sess, parent=parent, kind=kind, nall=nall, full_off=full_off, full_key=full_key,
                meta_json=np.array(json.dumps(dict(meta or {}, **{"check": st}), ensure_ascii=False)))
    if user_trigger is not None:
        arrs["user_trigger"] = np.asarray(user_trigger, bool)
        assert len(arrs["user_trigger"]) == len(t)
    np.savez_compressed(path, **arrs)
    return st


def prepare_nb(name, nb_dir=None):
    """回傳和 x1_trace.prepare／x1_fi.prepare_fi 同形的 dict。"""
    path = os.path.join(nb_dir or NB_DIR, f"{name}.npz")
    z = np.load(path, allow_pickle=False)
    t, sess, parent, kind, nall = z["t"], z["sess"], z["parent"], z["kind"], z["nall"]
    full_off, full_key = z["full_off"], z["full_key"]
    st, ids = check(t, sess, parent, kind, nall, full_off, full_key, name)
    del full_key
    n, B = len(t), st["n_keys"]
    # 記憶體（TraceLab 約 2.2 億個 chunk）：清單裡放「同一個 id 同一個 int 物件」（同原本 setdefault 的做法，每格只佔 8 bytes 指標），
    # 不用 .tolist() 每格造一個新的 int（約 36 bytes）；中間的大陣列用完就丟
    bobj = np.empty(B, dtype=object)
    bobj[:] = list(range(B))
    if n:
        fo = bobj[ids]
        full = [fo[full_off[i]:full_off[i + 1]].tolist() for i in range(n)]
        del fo
    else:
        full = []
    nf = np.diff(full_off)
    req = np.repeat(np.arange(n, dtype=np.int64), nf)
    order = np.lexsort((req, ids))                   # 依 block、再依請求排序（同一個請求裡一個鍵只出現一次）
    ids_s = ids[order]
    del ids
    req_s = req[order]
    del req, order
    robj = np.empty(n, dtype=object)
    robj[:] = list(range(n))
    ro = robj[req_s]
    del req_s
    acc = {}
    if len(ids_s):
        cut = np.flatnonzero(np.diff(ids_s)) + 1
        heads = bobj[ids_s[np.r_[0, cut]]].tolist()
        bnd = np.r_[0, cut, len(ids_s)].tolist()
        del ids_s, cut
        for i, b in enumerate(heads):
            acc[b] = ro[bnd[i]:bnd[i + 1]].tolist()
    del ro
    remap_n = st["n_keys"]
    meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
    meta.update(trace=f"nb:{name}", path=path, n_blocks_full=len(acc), n_blocks_all=remap_n, **{f"load_{k}": v for k, v in st.items()})
    R = dict(parent=parent.astype(np.int64), cand=parent.astype(np.int64).copy(),
             kind=np.array([KIND[int(k)] for k in kind], dtype=object), sess=sess.astype(np.int64),
             n_sess=int(sess.max()) + 1 if len(sess) else 0, rule="nb")
    out = dict(rows=None, meta=meta, R=R, full=full, nall=nall.tolist(), B=remap_n, acc=acc,
               t=t.astype(np.float64) - float(t[0]))
    if "user_trigger" in z.files:
        out["user_trigger"] = z["user_trigger"].astype(bool)
    return out
