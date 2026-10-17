"""trace.py — 統一資料表（code/p3_data，/ssd8/.../p3_unified）→ 引擎的輸入 P（和 x1_sim.prep 同形）。

P 的欄位：full（每個請求存的完整 512-token block，重新編號）、nall（prefill 的 block 數，含最後不滿的）、acc（block → 讀到它的請求 index，
遞增）、tl（到達秒，從 0 起）、kindc（0 新、1 接續＝回來的請求）、sessl、parentl、obs_end（每個請求的觀察窗終點，秒）、
feats（不含未來資訊的特徵）、nmax、B、R.n_sess、meta。

block（CLAUDE.md 規則 6：用 A 欄驗算 B 欄，斷言寫在這裡）：
  * Mooncake（512-token）：完整 block＝block_ids[:input_tokens//512]；斷言 len(block_ids)＝ceil(input/512) 的比例 ≥ 0.95。
  * 16-token 的來源（Bailian：ceil；FreeInference：floor）：連續 32 個併成一個 512-token block；
    斷言 len(block_ids) 對 input_tokens 的 floor 或 ceil 吻合 ≥ 0.95。
  * block 的 key＝整段前綴（滾動）。x1_fi 用「第 32j+31 個 id」，只在 id 是鏈式時才對；Bailian traceA 的 id 不是鏈式（只報告，
    meta.checks 的 raw_id_chain_violations），所以一律用整段前綴。鏈式來源兩種 key 一一對應。
  * 沒有 block 的來源（WildChat、synth）：每個 session 自己一條鏈（session 之間不共用前綴，**這是近似**），
    完整 block 數＝input_tokens//512。WildChat 的 token 是 Llama tokenizer（03b §3.5）；synth 是 o200k_base（直接當 512 token 用，近似）。
  * 每份都斷言：同一個 key 永遠在同一個位置（前綴鏈，違反 0 次）、父請求不晚於子請求。
重播：開迴路（照 trace 的 t_start_ms）。03_data_design §5 說預設閉迴路；這版沒做（06_simulator.md §8）。
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CODE, "p3_data"))

import numpy as np  # noqa: E402

BLK = 512
SUB = 32
COLS = ["session_id", "turn_idx", "parent_turn_idx", "t_start_ms", "input_tokens", "output_tokens", "trigger",
        "obs_end_ms", "block_tokens", "censored"]


class TraceAssertion(AssertionError):
    pass


def load_unified(source, variant, with_blocks=True):
    import schema
    cols = COLS + (["block_ids"] if with_blocks else [])
    return schema.read_unified(source, variant, columns=cols), schema.read_manifest(source, variant)


def _window(df, t_from_s, dur_s):
    t0 = int(df.t_start_ms.min())
    lo = t0 + int(t_from_s * 1000)
    hi = lo + int(dur_s * 1000) if dur_s else None
    m = df.t_start_ms >= lo
    if hi is not None:
        m &= df.t_start_ms < hi
    out = df[m].copy()
    if hi is not None:
        out["obs_end_ms"] = np.minimum(out.obs_end_ms.values, hi)    # 觀察窗跟著切（截尾以切窗的結尾算）
    return out


def prepare(source, variant, t_from_s=0.0, dur_s=None, overlay=1, seed=0, max_req=None, session_frac=1.0, session_seed=0,
            key_mode="auto"):
    """回傳 P。overlay>1：把真實 session 錯開疊上去（半合成，meta.tier='H'）。
    session_frac<1：以 session 為單位、依 session_seed 抽樣保留（X4 的子集做法；serve=fifo1 時一張 GPU 撐不住整份 trace 才用）。"""
    df, man = load_unified(source, variant, with_blocks=source not in ("wildchat", "synth"))
    if dur_s is not None or t_from_s:
        df = _window(df, t_from_s, dur_s)
    if session_frac < 1.0:
        us = df.session_id.drop_duplicates().values           # 依第一次出現的順序（資料順序固定 → 可重現）
        keep = set(us[np.random.default_rng(session_seed).random(len(us)) < session_frac])
        df = df[df.session_id.isin(keep)]
    df = df.reset_index(drop=True)
    df["_ord"] = np.arange(len(df))
    has_blocks = "block_ids" in df.columns and df.block_tokens.notna().all()
    bt = int(df.block_tokens.iloc[0]) if has_blocks else None
    checks = {}
    it = df.input_tokens.fillna(0).astype(np.int64).values
    if has_blocks:
        nb = df.block_ids.apply(len).values
        fl, ce = float(np.mean(nb == it // bt)), float(np.mean(nb == -(-it // bt)))
        checks["len(block_ids)==floor(input/bt)"] = fl
        checks["len(block_ids)==ceil(input/bt)"] = ce
        if max(fl, ce) < 0.95:
            raise TraceAssertion(f"{source}/{variant}: block_ids 的單位對不上 input_tokens（floor {fl:.4f}、ceil {ce:.4f}，block_tokens={bt}）")
    # ---- 原始 id 有沒有前綴鏈的性質（同一個 id 永遠在同一個位置、前一個 id 也相同）：只報告。
    #      Bailian traceA 沒有（2026-10-18 冒煙抓到：52,168 個 id 出現在不只一個位置），所以 key 不能用「最後一個 id」。
    if has_blocks:
        pos, prv, bp = {}, {}, 0
        for ids in df.block_ids:
            p_ = None
            for j, b in enumerate(ids):
                b = int(b)
                if b in pos:
                    bp += (pos[b] != j) + (prv[b] != p_)
                else:
                    pos[b], prv[b] = j, p_
                p_ = b
        checks["raw_id_chain_violations(report)"] = int(bp)
        del pos, prv
    # ---- 每列的完整 512-token block：key＝整段前綴（滾動：上一個 block 的 key＋這 512 token 的 id）。
    #      原始 id 是鏈式時，這和 x1_fi 的「第 32j+31 個 id」一一對應（共用結構完全相同）；不是鏈式時（traceA）也正確。
    raw = []
    pref = {}
    fast = has_blocks and key_mode == "auto" and checks["raw_id_chain_violations(report)"] == 0
    checks["block_key_mode"] = "last_id(chained)" if fast else "full_prefix"
    for i in range(len(df)):
        n512 = int(it[i]) // BLK
        if has_blocks:
            ids = df.block_ids.iat[i]
            if bt not in (BLK, 16):
                raise TraceAssertion(f"block_tokens={bt} 沒有處理")
            g = 1 if bt == BLK else SUB
            if fast:     # 鏈式 id：最後一個 id 就決定整段前綴（x1_fi 的做法），和下面的整段前綴一一對應
                keys = [pref.setdefault(int(ids[g * j + g - 1]), len(pref)) for j in range(n512)]
            else:
                keys, prev = [], -1
                for j in range(n512):
                    k = (prev, tuple(int(x) for x in ids[g * j:g * j + g]))
                    prev = pref.setdefault(k, len(pref))
                    keys.append(prev)
        else:
            keys = None
        raw.append(keys)
    del pref
    # ---- session 與父
    sid_map = {}
    sess = np.empty(len(df), np.int64)
    for i, s in enumerate(df.session_id.values):
        sess[i] = sid_map.setdefault(s, len(sid_map))
    if not has_blocks:      # 每個 session 自己一條鏈
        for i in range(len(df)):
            raw[i] = [(int(sess[i]), j) for j in range(int(it[i]) // BLK)]
    pos = {(s, int(ti)): i for i, (s, ti) in enumerate(zip(sess, df.turn_idx.values))}
    par = np.full(len(df), -1, np.int64)
    ptx = df.parent_turn_idx.values
    for i in range(len(df)):
        p = ptx[i]
        if p == p and p is not None:
            par[i] = pos.get((int(sess[i]), int(p)), -1)    # 切窗切掉的父 → 當成新的
    tms = df.t_start_ms.values.astype(np.int64)
    obs = df.obs_end_ms.values.astype(np.int64)
    trig = df.trigger.values
    outt = df.output_tokens.fillna(0).astype(np.int64).values
    tier = "R" if source != "synth" else "S"
    # ---- 疊加（半合成）：copy c≥1 的每個 session 依 seed 錯開，block 換成另一個命名空間
    if overlay > 1:
        tier = "H"
        rng = np.random.default_rng(seed)
        t0, t1 = int(tms.min()), int(tms.max())
        span = max(t1 - t0, 1)
        sstart = {}
        for i in range(len(df)):
            sstart.setdefault(int(sess[i]), int(tms[i]))
            sstart[int(sess[i])] = min(sstart[int(sess[i])], int(tms[i]))
        n0, S0 = len(df), len(sid_map)
        T, O, SE, PA, RAW, IT, OT, TR = [tms], [obs], [sess], [par], list(raw), [it], [outt], [trig]
        for c in range(1, overlay):
            off = rng.integers(0, span, size=S0)
            delta = np.array([((sstart[int(s)] - t0 + int(off[s])) % span) - (sstart[int(s)] - t0) for s in sess], np.int64)
            T.append(tms + delta)
            O.append(obs + delta)
            SE.append(sess + c * S0)
            PA.append(np.where(par >= 0, par + c * n0, -1))
            RAW += [[(c, k) for k in keys] for keys in raw]
            IT.append(it)
            OT.append(outt)
            TR.append(trig)
        tms, obs, sess, par = np.concatenate(T), np.concatenate(O), np.concatenate(SE), np.concatenate(PA)
        raw, it, outt, trig = RAW, np.concatenate(IT), np.concatenate(OT), np.concatenate(TR)
    # ---- 子請求比父請求早開始（資料怪象；FreeInference week13 有 1 列早 14 ms）：子請求的到達時刻夾到父請求的到達時刻，
    #      計數記在 checks；超過 1e-4 的比例就丟錯（2026-10-18 timing run 20261018-030356-p3sim-timing 抓到，06 §10）
    nclamp = 0
    for _ in range(4):           # 連續好幾代都早的情況（理論上）要多跑幾次
        bad = (par >= 0) & (tms < np.where(par >= 0, tms[np.maximum(par, 0)], 0))
        nb_ = int(bad.sum())
        if not nb_:
            break
        nclamp += nb_
        tms = np.where(bad, tms[np.maximum(par, 0)], tms)
    checks["child_before_parent_clamped"] = nclamp
    if nclamp > 1e-4 * len(tms):
        raise TraceAssertion(f"{source}/{variant}: 子請求比父請求早開始的列 {nclamp} 個，超過 1e-4")
    # ---- 依到達時間排（同時間保留原順序；夾過的子請求排在父請求後面，因為父的原始順序在前）
    order = np.lexsort((np.arange(len(tms)), tms))
    if max_req is not None:
        order = order[:max_req]
    inv = np.full(len(tms), -1, np.int64)
    inv[order] = np.arange(len(order))
    remap = {}
    full, nall, kindc, parentl, sessl, tl, oe, feats = [], [], [], [], [], [], [], []
    tz = int(tms[order[0]])
    smap = {}
    for k, i in enumerate(order):
        keys = [remap.setdefault(x, len(remap)) for x in raw[i]]
        full.append(keys)
        nall.append(max(-(-int(it[i]) // BLK), len(keys)))
        p = int(inv[par[i]]) if par[i] >= 0 else -1
        if p >= k:
            raise TraceAssertion(f"{source}/{variant}: 父請求排在子請求之後（{p} ≥ {k}）")
        kindc.append(1 if p >= 0 else 0)
        parentl.append(p)
        sessl.append(smap.setdefault(int(sess[i]), len(smap)))
        tl.append((int(tms[i]) - tz) / 1000.0)
        oe.append((int(obs[i]) - tz) / 1000.0)
        feats.append((str(trig[i]), int(it[i]), int(outt[i])))
    # 前綴鏈：同一個 key 永遠在同一個位置
    where = {}
    bad = 0
    for keys in full:
        for j, b in enumerate(keys):
            w = where.setdefault(b, j)
            bad += w != j
    checks["chain_position_violations"] = bad
    if bad:
        raise TraceAssertion(f"{source}/{variant}: 前綴鏈被違反 {bad} 次（同一個 block key 出現在不同位置）")
    acc = defaultdict(list)
    for r, keys in enumerate(full):
        for b in keys:
            acc[b].append(r)
    n_sess = len(smap)
    meta = dict(source=source, variant=variant, tier=tier, overlay=overlay, overlay_seed=seed if overlay > 1 else None,
                session_frac=session_frac, session_seed=session_seed if session_frac < 1.0 else None,
                t_from_s=t_from_s, dur_s=dur_s, n_req=len(full), n_sess=n_sess, n_ret=int(sum(kindc)),
                block_src=(f"block_ids（{bt}-token）" if has_blocks else "每個 session 自己一條鏈（沒有 block_ids）"),
                n_blocks=len(remap), ws_GiB_bf16_llama=len(acc) * 64 / 1024.0, checks=checks,
                unified_run=man.get("run_id"), unified_output_sha256=man.get("output_sha256"),
                span_s=tl[-1] - tl[0] if tl else 0.0)
    return dict(full=full, nall=nall, acc=dict(acc), tl=tl, kindc=kindc, sessl=sessl, parentl=parentl, obs_end=oe,
                feats=feats, nmax=max(nall) + 1 if nall else 1, B=len(remap), R=dict(n_sess=n_sess), meta=meta)
