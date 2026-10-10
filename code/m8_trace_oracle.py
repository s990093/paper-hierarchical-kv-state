"""m8_trace_oracle.py — D4：工作負載本身的資訊值多少（一寫多讀、會不會回來）

子命令：
  char     描述每個 Mooncake trace 的重用結構（重用比例、每寫幾讀、重用距離、從不重用、2-hit 學習延遲、前綴共享）
  sim      容量模擬：CPU／SSD／重算，比較 OPT（Belady）、寫入時預測（WTP-a／WTP-b）、延後／線上策略
  scbench  SCBench 的共享 context 結構（每個 context 被讀幾次、context 多長）

單位（CLAUDE.md §1 規則 6）：
  - hash_ids 一個＝512 token。載入時斷言 len(hash_ids)==ceil(input_length/512) 的吻合率 ≥ 0.95，並回報。
  - timestamp 用資料自己驗證：(1) 總長度；(2) 同一 session 前後兩輪的間隔 ÷ 上一輪的 output_length。
  - hash 是否為「前綴鏈」（同一個 hash 永遠在同一個位置、前一個 hash 也相同）也在載入時驗證。

所有輸出 CSV 都有 run_id、ts 欄。大輸出放 $TIARA_RUNS/$RUN_ID/。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import OrderedDict, defaultdict
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RES7 = os.path.join(REPO, "results", "m7_write_policy_mi300x")
OUTDIR = os.path.join(REPO, "results", "m8_directions")
TRACE_DIR = "/mlsteam/data/tiara/datasets/traces"
SCB_DIR = "/mlsteam/data/tiara/datasets/scbench/data"
BLOCK_TOK = 512
MiB = 1 << 20
GiB = 1 << 30
CHUNK_BYTES = 64 * MiB  # Llama-3.1-8B：32 層 × 8 KV 頭 × 128 × 2(K,V) × 2 B × 512 token
INF = 1 << 62


def run_id():
    return os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc"


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def run_dir():
    d = os.path.join(os.environ.get("TIARA_RUNS", "/mlsteam/data/tiara/runs"), run_id())
    os.makedirs(d, exist_ok=True)
    return d


class Out:
    def __init__(self, path, fields):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        new = not os.path.exists(path)
        self.f = open(path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=["run_id", "ts"] + fields, extrasaction="ignore")
        if new:
            self.w.writeheader()
        self.rid = run_id()

    def row(self, **kw):
        kw.update(run_id=self.rid, ts=ts())
        for k, v in list(kw.items()):
            if isinstance(v, (float, np.floating)):
                kw[k] = round(float(v), 6)
        self.w.writerow(kw)
        self.f.flush()


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ 載入與單位驗證
def load_trace(name):
    path = os.path.join(TRACE_DIR, f"{name}_trace.jsonl")
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    match = sum(len(r["hash_ids"]) == math.ceil(r["input_length"] / BLOCK_TOK) for r in rows)
    rate = match / n
    # 規則 6：用 input_length 驗算 hash_ids 的單位
    assert rate >= 0.95, f"{name}: len(hash_ids)==ceil(input_length/512) 只有 {rate:.4f}，單位不是 512 token？"
    tss = [r["timestamp"] for r in rows]
    assert tss == sorted(tss), f"{name}: timestamp 沒排序"
    # 前綴鏈驗證：同一個 hash 永遠在同一位置、前一個 hash 也一樣
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
    dup_in_req = sum(len(set(r["hash_ids"])) != len(r["hash_ids"]) for r in rows)
    meta = dict(trace=name, n_req=n, unit_match_rate=rate, ts_min=tss[0], ts_max=tss[-1],
                chain_bad_pos=bad_pos, chain_bad_prev=bad_prev, req_with_dup_hash=dup_in_req,
                max_input_len=max(r["input_length"] for r in rows))
    return rows, meta


def build_access(rows):
    """把請求展開成 block 存取序列（依 timestamp、檔案順序；請求內依位置）。"""
    remap = {}
    blk, pos, req, tms = [], [], [], []
    for ri, r in enumerate(rows):
        for i, h in enumerate(r["hash_ids"]):
            b = remap.setdefault(h, len(remap))
            blk.append(b)
            pos.append(i)
            req.append(ri)
            tms.append(r["timestamp"])
    blk = np.array(blk, dtype=np.int64)
    N, B = len(blk), len(remap)
    nxt = np.full(N, INF, dtype=np.int64)
    prv = np.full(N, -1, dtype=np.int64)
    last = {}
    for a in range(N - 1, -1, -1):
        b = int(blk[a])
        if b in last:
            nxt[a] = last[b]
        last[b] = a
    last = {}
    for a in range(N):
        b = int(blk[a])
        if b in last:
            prv[a] = last[b]
        last[b] = a
    total = np.bincount(blk, minlength=B)
    occ = np.zeros(N, dtype=np.int64)  # 這是該 block 的第幾次存取（1 起算）
    seen = np.zeros(B, dtype=np.int64)
    for a in range(N):
        b = blk[a]
        seen[b] += 1
        occ[a] = seen[b]
    bpos = np.zeros(B, dtype=np.int64)
    bpos[blk] = np.array(pos)
    return dict(blk=blk, pos=np.array(pos, dtype=np.int64), req=np.array(req, dtype=np.int64),
                t=np.array(tms, dtype=np.int64), nxt=nxt, prv=prv, total=total, occ=occ, bpos=bpos, N=N, B=B)


def load_f():
    """C1 的 f(i)（ms），每個 chunk 取中位數；i ≥ 80 線性外插（最小平方）。"""
    import pandas as pd
    d = pd.read_csv(os.path.join(RES7, "calib_c1.csv"))
    d = d[(d.item == "f_chunk") & (d.gpu_state == "idle")]
    m = d.groupby("chunk_idx").ms.median().sort_index()
    x, y = m.index.values.astype(float), m.values.astype(float)
    slope, icpt = np.polyfit(x, y, 1)
    return y, float(slope), float(icpt), sorted(d.run_id.unique())


def f_of(pos_arr, fmeas, slope, icpt):
    n = len(fmeas)
    out = np.where(pos_arr < n, fmeas[np.minimum(pos_arr, n - 1)], icpt + slope * pos_arr)
    return out.astype(float)


# ------------------------------------------------------------------ char
def stack_distances(blk, prv):
    """每次重用存取的 stack distance（中間出現過的不同 block 數），Fenwick tree。"""
    N = len(blk)
    tree = [0] * (N + 1)

    def add(i, v):
        i += 1
        while i <= N:
            tree[i] += v
            i += i & -i

    def pre(i):  # sum of [0, i)
        s = 0
        while i > 0:
            s += tree[i]
            i -= i & -i
        return s

    out = np.full(N, -1, dtype=np.int64)
    for a in range(N):
        p = int(prv[a])
        if p >= 0:
            out[a] = pre(a) - pre(p + 1)
            add(p, -1)
        add(a, 1)
    return out


def pct(x, qs=(10, 25, 50, 75, 90, 99)):
    if len(x) == 0:
        return {f"p{q}": "NA" for q in qs}
    v = np.percentile(x, qs)
    return {f"p{q}": float(vv) for q, vv in zip(qs, v)}


def cmd_char(args):
    rd = run_dir()
    o_char = Out(os.path.join(OUTDIR, "d4_trace_char.csv"),
                 ["trace", "metric", "value", "unit", "note"])
    o_dist = Out(os.path.join(OUTDIR, "d4_reuse_dist.csv"),
                 ["trace", "quantity", "n", "p10", "p25", "p50", "p75", "p90", "p99", "unit"])
    for name in args.traces:
        rows, meta = load_trace(name)
        A = build_access(rows)
        N, B = A["N"], A["B"]
        blk, t, prv, occ, total, req = A["blk"], A["t"], A["prv"], A["occ"], A["total"], A["req"]
        log(f"[{name}] req={meta['n_req']} accesses={N} blocks={B} meta={meta}")
        put = lambda m, v, u="", note="": o_char.row(trace=name, metric=m, value=v, unit=u, note=note)
        for k, v in meta.items():
            if k != "trace":
                put(k, v, "", "載入驗證")
        span_ms = meta["ts_max"] - meta["ts_min"]
        put("ts_span_raw", span_ms, "raw", "若單位是 ms 則為 %.1f 分鐘" % (span_ms / 60000))

        # 用 session 前後兩輪驗證 timestamp 單位：子請求的間隔 ÷ 父請求的 output_length
        lastfull = {}  # 請求最後一個完整 block 的 hash → 請求 index
        gaps, ratio = [], []
        for ri, r in enumerate(rows):
            h = r["hash_ids"]
            par = None
            for k in range(len(h) - 1, 0, -1):  # 不算 block 0（共用 system prompt）
                if h[k] in lastfull:
                    par = lastfull[h[k]]
                    break
            if par is not None:
                g = r["timestamp"] - rows[par]["timestamp"]
                gaps.append(g)
                if rows[par]["output_length"] >= 50:
                    ratio.append(g / rows[par]["output_length"])
            nfull = r["input_length"] // BLOCK_TOK
            if nfull >= 2:
                lastfull[h[nfull - 1]] = ri
        gaps, ratio = np.array(gaps, float), np.array(ratio, float)
        put("n_parent_child_pairs", len(gaps), "", "子請求的前綴含父請求最後一個完整 block（block≥1）")
        o_dist.row(trace=name, quantity="parent_child_gap_raw", n=len(gaps), unit="raw ts", **pct(gaps))
        o_dist.row(trace=name, quantity="gap_raw_per_parent_output_token", n=len(ratio), unit="raw ts/token",
                   **pct(ratio))
        if len(ratio):
            put("frac_gap_lt_1raw_per_token", float(np.mean(ratio < 1.0)), "",
                "若單位是 ms：間隔 < 1 ms/輸出 token（物理上不可能的解碼速度）的比例")

        # 重用結構
        reuse = prv >= 0
        n_reuse = int(reuse.sum())
        reused_blocks = int((total >= 2).sum())
        put("accesses", N)
        put("distinct_blocks", B, "block", "工作集＝%.1f TiB（×64 MiB）" % (B * 64 / 1024 / 1024))
        put("frac_accesses_reuse", n_reuse / N, "", "無限容量時的命中率上限")
        put("frac_blocks_reused", reused_blocks / B)
        put("frac_blocks_never_reused", 1 - reused_blocks / B)
        put("reads_per_write", N / B, "", "總存取 ÷ 不同 block")
        put("reuses_per_reused_block", n_reuse / max(reused_blocks, 1))
        k = total[total >= 2] - 1
        for kk in (1, 2, 3):
            put(f"frac_reused_blocks_with_{kk}_reuse", float(np.mean(k == kk)))
        put("frac_reused_blocks_with_ge4_reuse", float(np.mean(k >= 4)))
        put("frac_reuse_accesses_from_blocks_with_1_reuse", float(k[k == 1].sum() / k.sum()))
        # 2-hit 學習延遲：第 2 次存取一定 miss
        sec = occ == 2
        put("twohit_lost_frac_of_reuse_accesses", float(sec.sum() / n_reuse), "",
            "2-hit 准入一定錯過的重用存取比例（每個被重用的 block 錯過第 2 次）")
        dt2 = (t[sec] - t[prv[sec]]) / 1000.0
        o_dist.row(trace=name, quantity="twohit_admit_delay_s(first->second access)", n=int(sec.sum()), unit="s(假設 ms)",
                   **pct(dt2))
        # 截尾效應：前半段寫入的 block
        tmid = meta["ts_min"] + span_ms / 2
        first_t = np.zeros(B, dtype=np.int64)
        first_t[blk[occ == 1]] = t[occ == 1]
        early = first_t <= tmid
        put("frac_blocks_never_reused_firsthalf", float(1 - (total[early] >= 2).mean()), "",
            "只看前半段第一次出現的 block（之後還有 ≥30 分鐘未來），去掉 trace 截尾效應")
        # 重用距離
        dts = (t[reuse] - t[prv[reuse]]) / 1000.0
        o_dist.row(trace=name, quantity="reuse_dist_time_s", n=n_reuse, unit="s(假設 ms)", **pct(dts))
        sd = stack_distances(blk, prv)
        sdb = sd[reuse] * 64 / 1024.0  # GiB
        o_dist.row(trace=name, quantity="reuse_dist_bytes_GiB", n=n_reuse, unit="GiB(×64MiB)", **pct(sdb))
        o_dist.row(trace=name, quantity="reuse_dist_blocks_frac_of_WS", n=n_reuse, unit="frac of distinct blocks",
                   **pct(sd[reuse] / B))
        np.save(os.path.join(rd, f"{name}_stackdist.npy"), sd)
        # 前綴共享
        b0 = [r["hash_ids"][0] for r in rows if r["hash_ids"]]
        cnt0 = defaultdict(int)
        for h in b0:
            cnt0[h] += 1
        c = np.array(sorted(cnt0.values(), reverse=True))
        put("distinct_block0", len(c))
        put("top1_block0_share_of_requests", float(c[0] / len(b0)))
        put("top10_block0_share_of_requests", float(c[:10].sum() / len(b0)))
        put("frac_requests_block0_shared_ge2", float(np.mean([cnt0[h] >= 2 for h in b0])))
        put("frac_requests_block0_shared_ge10", float(np.mean([cnt0[h] >= 10 for h in b0])))
        o_dist.row(trace=name, quantity="requests_per_block0", n=len(c), unit="requests", **pct(c))
        # 重用存取的位置
        o_dist.row(trace=name, quantity="reuse_access_position", n=n_reuse, unit="block idx", **pct(A["pos"][reuse]))
        # session 會不會回來（請求層級）：這個請求的任何 block 之後還會被讀
        ret = np.zeros(meta["n_req"], bool)
        np.logical_or.at(ret, req, A["nxt"] < INF)
        put("frac_requests_return", float(ret.mean()), "", "這個請求寫的任何 block 之後還有人讀")
        newblk = occ == 1
        retnew = np.zeros(meta["n_req"], bool)
        np.logical_or.at(retnew, req[newblk], A["nxt"][newblk] < INF)
        has_new = np.zeros(meta["n_req"], bool)
        has_new[req[newblk]] = True
        put("frac_requests_with_new_blocks_reused", float(retnew[has_new].mean()), "",
            "只看有新 block 的請求：它新寫的 block 之後有人讀（session 回來）")
        # 新 block 裡「同一請求有回來、但這個 block 沒被讀」的比例 → (b) 比 (a) 粗多少
        nb_ret = retnew[req[newblk]]
        nb_used = A["nxt"][newblk] < INF
        put("frac_new_blocks_in_returning_req_but_unused", float(np.mean(~nb_used[nb_ret])), "",
            "WTP-b 會收、但 WTP-a 不收的新 block 比例")
        log(f"[{name}] done char")
    log("char done; run dir", rd)


def session_returns(rows, A):
    """session 會不會回來：之後有請求的前綴包含這個請求最後一個完整 block（沒有完整 block 就看 block 0）。
    不能用「任何 block 之後有人讀」：conversation 全部請求共用同一個 block 0，那樣幾乎 100% 都算回來。"""
    req, pos, nxt = A["req"], A["pos"], A["nxt"]
    starts = np.r_[0, np.nonzero(np.diff(req))[0] + 1]
    ret = np.zeros(len(rows), bool)
    for ri, r in enumerate(rows):
        k = max(r["input_length"] // BLOCK_TOK - 1, 0)
        a = starts[ri] + k
        assert req[a] == ri and pos[a] == k
        ret[ri] = nxt[a] < INF
    return ret


def cmd_char2(args):
    """補充：timestamp 解析度、session 回來比例（child 定義）。"""
    o_char = Out(os.path.join(OUTDIR, "d4_trace_char.csv"), ["trace", "metric", "value", "unit", "note"])
    for name in args.traces:
        rows, meta = load_trace(name)
        A = build_access(rows)
        put = lambda m, v, u="", note="": o_char.row(trace=name, metric=m, value=v, unit=u, note=note)
        t = np.array([r["timestamp"] for r in rows])
        put("ts_frac_mod1000_in_{0,999}", float(np.mean((t % 1000 == 0) | (t % 1000 == 999))), "",
            "timestamp 落在 1000 的整數倍（或差 1）的比例：解析度約 1000 raw")
        put("ts_distinct_values", len(np.unique(t)))
        ret = session_returns(rows, A)
        put("frac_requests_session_returns", float(ret.mean()), "",
            "之後有請求的前綴包含它最後一個完整 block（session 被接著問）")
        newblk = A["occ"] == 1
        nb_req = A["req"][newblk]
        put("frac_new_blocks_in_session_returning_req", float(ret[nb_req].mean()), "", "WTP-b 會收的新 block 比例")
        put("frac_new_blocks_reused", float((A["nxt"][newblk] < INF).mean()), "", "WTP-a 會收的新 block 比例")
        ok = (A["nxt"][newblk] < INF)
        put("frac_new_reused_blocks_in_nonreturning_req", float(np.mean(~ret[nb_req][ok])), "",
            "會被重用、但所屬請求的 session 不回來（被別的 session 分叉讀走）的新 block 比例")
        log(name, "char2 done")


# ------------------------------------------------------------------ 快取策略（CPU 層）
class Rec:
    """記錄每個 block 第一次離開 CPU（淘汰或拒收）的存取 index，供 SSD 規則用。"""

    def __init__(self, A):
        B = A["B"]
        self.cnt = np.zeros(B, dtype=np.int64)
        self.total = A["total"]
        self.ev = np.full(B, INF, dtype=np.int64)    # 第一次離開 CPU
        self.ev2 = np.full(B, INF, dtype=np.int64)   # 第一次「存取 ≥2 次之後」離開 CPU
        self.evl = np.full(B, INF, dtype=np.int64)   # 第一次「之後還會被讀」時離開 CPU（WTP-a 用）
        self.n_ins = 0

    def out(self, x, a):
        if self.ev[x] == INF:
            self.ev[x] = a
        if self.cnt[x] >= 2 and self.ev2[x] == INF:
            self.ev2[x] = a
        if self.total[x] - self.cnt[x] >= 1 and self.evl[x] == INF:
            self.evl[x] = a


def run_lru(A, C, rec, admit=None, dead_drop=False):
    """LRU。admit(a) 回傳 False 時不收（拒收也算離開 CPU）。dead_drop：讀完最後一次就丟。"""
    blk, N = A["blk"], A["N"]
    total = A["total"]
    hit = np.zeros(N, bool)
    od = OrderedDict()
    cnt = rec.cnt
    for a in range(N):
        b = int(blk[a])
        cnt[b] += 1
        if b in od:
            hit[a] = True
            if dead_drop and cnt[b] >= total[b]:
                del od[b]
                continue
            od.move_to_end(b)
            continue
        if (admit is not None and not admit(a, b)) or (dead_drop and cnt[b] >= total[b]):
            rec.out(b, a)
            continue
        od[b] = None
        rec.n_ins += 1
        if len(od) > C:
            x, _ = od.popitem(last=False)
            rec.out(x, a)
    return hit


def run_2hit(A, C, rec):
    seen = set()

    def admit(a, b):
        if b in seen:
            return True
        seen.add(b)
        return False

    return run_lru(A, C, rec, admit=admit)


def run_slru(A, C, rec, prot_frac=0.8):
    blk, N = A["blk"], A["N"]
    hit = np.zeros(N, bool)
    P = max(1, int(C * prot_frac))
    prob, prot = OrderedDict(), OrderedDict()
    cnt = rec.cnt
    for a in range(N):
        b = int(blk[a])
        cnt[b] += 1
        if b in prot:
            hit[a] = True
            prot.move_to_end(b)
            continue
        if b in prob:
            hit[a] = True
            del prob[b]
            prot[b] = None
            if len(prot) > P:
                x, _ = prot.popitem(last=False)
                prob[x] = None
            continue
        prob[b] = None
        rec.n_ins += 1
        while len(prob) + len(prot) > C:
            if prob:
                x, _ = prob.popitem(last=False)
            else:
                x, _ = prot.popitem(last=False)
            rec.out(x, a)
    return hit


def run_arc(A, C, rec):
    """ARC（Megiddo & Modha, FAST'03）。"""
    blk, N = A["blk"], A["N"]
    hit = np.zeros(N, bool)
    T1, T2, B1, B2 = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
    p = 0.0
    cnt = rec.cnt

    def replace(b_in_B2, a):
        nonlocal p
        if T1 and (len(T1) > p or (b_in_B2 and len(T1) == p)):
            x, _ = T1.popitem(last=False)
            B1[x] = None
        else:
            x, _ = T2.popitem(last=False)
            B2[x] = None
        rec.out(x, a)

    for a in range(N):
        b = int(blk[a])
        cnt[b] += 1
        if b in T1:
            hit[a] = True
            del T1[b]
            T2[b] = None
            continue
        if b in T2:
            hit[a] = True
            T2.move_to_end(b)
            continue
        rec.n_ins += 1
        if b in B1:
            p = min(C, p + max(len(B2) / max(len(B1), 1), 1))
            replace(False, a)
            del B1[b]
            T2[b] = None
            continue
        if b in B2:
            p = max(0.0, p - max(len(B1) / max(len(B2), 1), 1))
            replace(True, a)
            del B2[b]
            T2[b] = None
            continue
        L1 = len(T1) + len(B1)
        if L1 == C:
            if len(T1) < C:
                B1.popitem(last=False)
                replace(False, a)
            else:
                x, _ = T1.popitem(last=False)
                rec.out(x, a)
        elif L1 < C:
            tot = L1 + len(T2) + len(B2)
            if tot >= C:
                if tot == 2 * C:
                    B2.popitem(last=False)
                replace(False, a)
        T1[b] = None
    return hit


def run_s3fifo(A, C, rec, small_frac=0.1):
    """S3-FIFO（Yang et al. SOSP'23）。S→M 門檻：在 S 裡被讀過 ≥1 次。"""
    blk, N = A["blk"], A["N"]
    hit = np.zeros(N, bool)
    S, M, G = OrderedDict(), OrderedDict(), OrderedDict()  # 尾端＝最舊（popitem(last=False)）
    freq = {}
    Sc = max(1, int(C * small_frac))
    Mc = C - Sc
    cnt = rec.cnt

    def evictM(a):
        while M:
            x, _ = M.popitem(last=False)
            if freq[x] > 0:
                freq[x] -= 1
                M[x] = None
            else:
                del freq[x]
                rec.out(x, a)
                return

    def evictS(a):
        while S:
            x, _ = S.popitem(last=False)
            if freq[x] >= 1:
                freq[x] = 0
                M[x] = None
                if len(M) > Mc:
                    evictM(a)
            else:
                del freq[x]
                G[x] = None
                if len(G) > Mc:
                    G.popitem(last=False)
                rec.out(x, a)
                return

    for a in range(N):
        b = int(blk[a])
        cnt[b] += 1
        if b in freq:
            hit[a] = True
            freq[b] = min(freq[b] + 1, 3)
            continue
        rec.n_ins += 1
        while len(S) + len(M) >= C:
            if len(S) >= Sc or not M:
                evictS(a)
            else:
                evictM(a)
        freq[b] = 0
        if b in G:
            del G[b]
            M[b] = None
        else:
            S[b] = None
    return hit


def run_belady(A, C, rec):
    """Belady MIN＋拒收：趕走下次使用最遠的；新 block 若比所有人都遠就不收。"""
    import heapq
    blk, nxt, N = A["blk"], A["nxt"], A["N"]
    hit = np.zeros(N, bool)
    incache = set()
    cur = {}
    heap = []
    cnt = rec.cnt
    for a in range(N):
        b = int(blk[a])
        cnt[b] += 1
        nu = int(nxt[a])
        if b in incache:
            hit[a] = True
            cur[b] = nu
            heapq.heappush(heap, (-nu, b))
            continue
        if nu >= INF:
            rec.out(b, a)
            continue
        if len(incache) < C:
            incache.add(b)
            cur[b] = nu
            heapq.heappush(heap, (-nu, b))
            rec.n_ins += 1
            continue
        while True:
            negnu, x = heap[0]
            if x in incache and cur[x] == -negnu:
                break
            heapq.heappop(heap)
        if -negnu > nu:
            heapq.heappop(heap)
            incache.discard(x)
            del cur[x]
            rec.out(x, a)
            incache.add(b)
            cur[b] = nu
            heapq.heappush(heap, (-nu, b))
            rec.n_ins += 1
        else:
            rec.out(b, a)
    return hit


def run_costbelady(A, C, rec, w_blk, K=32, seed=0):
    """看成本的 Belady 變體（近似）：抽 K 個，趕走 (下次使用距離 ÷ 命中省下的成本) 最大的。"""
    rng = random.Random(seed)
    blk, nxt, N = A["blk"], A["nxt"], A["N"]
    hit = np.zeros(N, bool)
    items, idx, cur = [], {}, {}
    cnt = rec.cnt
    w = w_blk

    def score(x, a):
        nu = cur[x]
        return float("inf") if nu >= INF else (nu - a) / w[x]

    for a in range(N):
        b = int(blk[a])
        cnt[b] += 1
        nu = int(nxt[a])
        if b in idx:
            hit[a] = True
            cur[b] = nu
            continue
        if nu >= INF or w[b] <= 0:
            rec.out(b, a)
            continue
        if len(items) < C:
            idx[b] = len(items)
            items.append(b)
            cur[b] = nu
            rec.n_ins += 1
            continue
        sb = (nu - a) / w[b]
        cand = [items[rng.randrange(len(items))] for _ in range(K)]
        best = max(cand, key=lambda x: score(x, a))
        if score(best, a) > sb:
            i = idx.pop(best)
            del cur[best]
            items[i] = b
            idx[b] = i
            cur[b] = nu
            rec.out(best, a)
            rec.n_ins += 1
        else:
            rec.out(b, a)
    return hit


def run_know_evict(A, C, rec, dead):
    """（事後加的對照組）同樣的未來知識，但在「淘汰時」才用：全部先收進 CPU（LRU），
    要騰位置時先趕「知識說之後沒人讀」的 block，沒有才趕 LRU。dead[a]＝這次存取之後這個 block 不會再被讀。"""
    blk, N = A["blk"], A["N"]
    hit = np.zeros(N, bool)
    od = OrderedDict()
    deadset = set()
    cnt = rec.cnt
    for a in range(N):
        b = int(blk[a])
        cnt[b] += 1
        if b in od:
            hit[a] = True
            od.move_to_end(b)
        else:
            od[b] = None
            rec.n_ins += 1
        if dead[a]:
            deadset.add(b)
        else:
            deadset.discard(b)
        while len(od) > C:
            if deadset:
                x = deadset.pop()
                del od[x]
            else:
                x, _ = od.popitem(last=False)
            rec.out(x, a)
    return hit


def gpu_filter(A, G):
    """GPU 層：固定 G 個 block 的 LRU、全收。回傳只含 GPU miss 的子序列（給 CPU 層的策略看），以及 GPU 命中數。"""
    blk, N = A["blk"], A["N"]
    od = OrderedDict()
    keep = np.ones(N, bool)
    for a in range(N):
        b = int(blk[a])
        if b in od:
            keep[a] = False
            od.move_to_end(b)
        else:
            od[b] = None
            if len(od) > G:
                od.popitem(last=False)
    idx = np.nonzero(keep)[0]
    sub = dict(blk=A["blk"][idx], pos=A["pos"][idx], req=A["req"][idx], t=A["t"][idx], N=len(idx), B=A["B"],
               bpos=A["bpos"])
    blk2 = sub["blk"]
    N2 = len(idx)
    nxt = np.full(N2, INF, dtype=np.int64)
    prv = np.full(N2, -1, dtype=np.int64)
    last = {}
    for a in range(N2 - 1, -1, -1):
        b = int(blk2[a])
        if b in last:
            nxt[a] = last[b]
        last[b] = a
    last = {}
    for a in range(N2):
        b = int(blk2[a])
        if b in last:
            prv[a] = last[b]
        last[b] = a
    occ = np.zeros(N2, dtype=np.int64)
    seen = np.zeros(A["B"], dtype=np.int64)
    for a in range(N2):
        seen[blk2[a]] += 1
        occ[a] = seen[blk2[a]]
    sub.update(nxt=nxt, prv=prv, occ=occ, total=np.bincount(blk2, minlength=A["B"]))
    assert int((occ == 1).sum()) == int((A["occ"] == 1).sum()), "第一次存取不可能是 GPU 命中"
    return sub, int(N - N2)


# ------------------------------------------------------------------ 成本
def ssd_write_idx(rule, A, rec):
    """每個 block 第一次寫進 SSD 的存取 index（INF＝從沒寫）。"""
    blk, occ, B = A["blk"], A["occ"], A["B"]
    w = np.full(B, INF, dtype=np.int64)
    if rule == "WT":
        first = occ == 1
        w[blk[first]] = np.nonzero(first)[0]
    elif rule == "WB":
        w = rec.ev.copy()
    elif rule == "WB2":
        w = rec.ev2.copy()
    elif rule == "H2":
        sec = occ == 2
        w[blk[sec]] = np.nonzero(sec)[0]
    elif rule == "WBR":  # WTP-a：被趕出 CPU、且之後還會被讀
        w = rec.evl.copy()
    elif rule == "WTR":  # WTP-a：寫入當下就知道之後會被讀 → 寫穿
        first = (occ == 1) & (A["nxt"] < INF)
        w[blk[first]] = np.nonzero(first)[0]
    elif rule == "NONE":
        pass
    else:
        raise ValueError(rule)
    return w


def cost_eval(A, hit, wssd, l_cpu, l_ssd, fpos, useful_blk, rule):
    """逐 block 相加模型。回傳 (總 TTFT ms, 只算重用的 ms, SSD 寫入 block 數)。"""
    blk, occ, N = A["blk"], A["occ"], A["N"]
    a_idx = np.arange(N)
    reuse = occ >= 2
    if rule == "NEED":  # OPT：之後要用、那時不在 CPU、SSD 比重算便宜的才寫，寫一次
        need = reuse & ~hit & useful_blk[blk]
        on_ssd = need
        nwrite = len(np.unique(blk[need]))
    else:
        wb = np.where(useful_blk, wssd, INF)
        on_ssd = wb[blk] < a_idx
        nwrite = int((wb < INF).sum())
    c = fpos.copy()
    c = np.where(reuse & on_ssd, np.minimum(c, l_ssd), c)
    c = np.where(reuse & hit, np.minimum(c, l_cpu), c)
    return float(c.sum()), float(c[reuse].sum()), nwrite, c


def cake_eval(A, cost_blk_access, fpos):
    """敏感度：Cake 重疊模型。每個請求的「之前出現過」前綴：GPU 從前面算、I/O 從後面載，取最好的分界；新 block 照算。"""
    req, occ = A["req"], A["occ"]
    total = 0.0
    N = A["N"]
    starts = np.r_[0, np.nonzero(np.diff(req))[0] + 1, N]
    for s, e in zip(starts[:-1], starts[1:]):
        o = occ[s:e]
        P = int(np.sum(o >= 2))  # 前綴鏈：重用的 block 一定在前面
        f = fpos[s:e]
        c = cost_blk_access[s:e]
        total += f[P:].sum()
        if P == 0:
            continue
        loadable = c[:P] < f[:P]  # 有比重算便宜的來源
        lcost = np.where(loadable, c[:P], 0.0)
        gcost = np.where(loadable, 0.0, f[:P])
        # 分界 m：[0,m) 全算；[m,P) 能載就載、不能載就算
        F = np.r_[0.0, np.cumsum(f[:P])]
        Lsuf = np.r_[np.cumsum(lcost[::-1])[::-1], 0.0]
        Gsuf = np.r_[np.cumsum(gcost[::-1])[::-1], 0.0]
        total += float(np.min(np.maximum(F + Gsuf, Lsuf)))
    return total


def cmd_sim(args):
    rd = run_dir()
    fmeas, slope, icpt, frun = load_f()
    params = json.load(open(os.path.join(RES7, "tier_params.json")))
    ssd_ms = {k: CHUNK_BYTES / (params[k]["read_GiBps"] * GiB) * 1e3 for k in ("local", "nfs")}
    fields = ["trace", "cpu_frac", "cpu_cap_blocks", "cpu_gibps", "ssd", "policy", "cpu_policy", "ssd_rule",
              "klass", "ttft_total_s", "ttft_reuse_s", "ttft_cake_s", "ssd_write_blocks", "ssd_write_GiB",
              "cpu_hits", "reuse_accesses", "cpu_inserts", "f_source"]
    if args.out != "d4_sim.csv":  # 第 2 次以後的 run（事後對照組、GPU 層敏感度）寫另一個檔，欄位多兩個
        fields += ["gpu_blocks", "gpu_hits"]
    o = Out(os.path.join(OUTDIR, args.out), fields)
    log(f"f: {len(fmeas)} 點, 外插 slope={slope:.4f} ms/chunk icpt={icpt:.3f}; ssd_ms={ssd_ms}")
    for name, G in [(n, g) for n in args.traces for g in args.gpu_blocks]:
        rows, meta = load_trace(name)
        A0 = build_access(rows)
        req_ret = session_returns(rows, A0)  # WTP-b：這個 session 會不會被接著問（見 session_returns）
        if G > 0:
            A, gpu_hits = gpu_filter(A0, G)
        else:
            A, gpu_hits = A0, 0
        B = A["B"]
        fpos = f_of(A["pos"], fmeas, slope, icpt)
        fblk = f_of(A["bpos"], fmeas, slope, icpt)
        n_reuse = int((A["occ"] >= 2).sum())
        log(f"[{name} G={G}] N={A['N']} B={B} reuse={n_reuse} gpu_hits={gpu_hits}")
        for frac in args.cpu_frac:
            C = max(1, int(B * frac))
            runs = {}

            def go(key, fn, *fa, **fk):
                rec = Rec(A)
                h = fn(A, C, rec, *fa, **fk)
                runs[key] = (h, rec)
                log(f"  [{name} {frac}] {key}: hits={int(h.sum())} ins={rec.n_ins}")

            go("LRU", run_lru)
            go("2HIT", run_2hit)
            go("SLRU", run_slru)
            go("ARC", run_arc)
            go("S3FIFO", run_s3fifo)
            go("BELADY", run_belady)
            total = A["total"]
            nxt = A["nxt"]
            go("WTPa", run_lru, admit=lambda a, b: nxt[a] < INF, dead_drop=True)  # 寫入時知道剩幾次讀＝之後還有沒有讀
            go("WTPb", run_lru, admit=lambda a, b: bool(req_ret[A["req"][a]]))
            if args.posthoc:
                go("KEVa", run_know_evict, dead=(nxt >= INF))
                go("KEVb", run_know_evict, dead=~req_ret[A["req"]])
            for gbps in args.cpu_gibps:
                l_cpu = CHUNK_BYTES / (gbps * GiB) * 1e3
                for ssd in args.ssd:
                    l_ssd = ssd_ms[ssd]
                    useful = l_ssd < fblk
                    # 看成本的 Belady：命中 CPU 省下 = min(SSD, 重算) − CPU
                    wb = np.minimum(fblk, np.where(useful, l_ssd, np.inf)) - l_cpu
                    rec = Rec(A)
                    hcb = run_costbelady(A, C, rec, wb)
                    runs_local = dict(runs)
                    runs_local["COSTBELADY"] = (hcb, rec)
                    log(f"  [{name} {frac} {gbps} {ssd}] COSTBELADY hits={int(hcb.sum())}")
                    combos = []
                    for cp in ("LRU", "2HIT", "SLRU", "ARC", "S3FIFO"):
                        for sr in ("WT", "WB", "WB2", "H2", "NONE"):
                            combos.append((f"{cp}+{sr}", cp, sr, "deferred"))
                    combos += [("WTPa+WBR", "WTPa", "WBR", "wtp_a"), ("WTPa+WTR", "WTPa", "WTR", "wtp_a"),
                               ("WTPb+WB", "WTPb", "WB", "wtp_b"),
                               ("OPT-BELADY", "BELADY", "NEED", "opt"), ("OPT-COST", "COSTBELADY", "NEED", "opt")]
                    if args.posthoc:
                        combos += [("KEVa+WBR", "KEVa", "WBR", "kev_a"), ("KEVa+WB", "KEVa", "WB", "kev_a"),
                                   ("KEVb+WB", "KEVb", "WB", "kev_b")]
                    for pol, cp, sr, kl in combos:
                        h, rec = runs_local[cp]
                        wssd = ssd_write_idx(sr, A, rec) if sr != "NEED" else None
                        T, Tr, nw, cvec = cost_eval(A, h, wssd, l_cpu, l_ssd, fpos, useful, sr)
                        Tc = cake_eval(A, cvec, fpos) if (args.cake and G == 0) else float("nan")
                        o.row(trace=name, cpu_frac=frac, cpu_cap_blocks=C, cpu_gibps=gbps, ssd=ssd, policy=pol,
                              cpu_policy=cp, ssd_rule=sr, klass=kl, ttft_total_s=T / 1e3, ttft_reuse_s=Tr / 1e3,
                              ttft_cake_s=Tc / 1e3, ssd_write_blocks=nw, ssd_write_GiB=nw * 64 / 1024,
                              cpu_hits=int(h.sum()), reuse_accesses=n_reuse, cpu_inserts=rec.n_ins,
                              f_source=f"calib_c1 {frun[0]}; i>=80 linear extrap", gpu_blocks=G, gpu_hits=gpu_hits)
    log("sim done; run dir", rd)


def cmd_nocache(args):
    """完全不重用（每個 block 都重算）的模型 TTFT，用來算「延後版拿到 oracle 好處的幾成」。"""
    fmeas, slope, icpt, frun = load_f()
    o = Out(os.path.join(OUTDIR, "d4_nocache.csv"), ["trace", "ttft_total_nocache_s", "ttft_reuse_nocache_s",
                                                      "ttft_compulsory_s", "f_source"])
    for name in args.traces:
        rows, meta = load_trace(name)
        A = build_access(rows)
        fpos = f_of(A["pos"], fmeas, slope, icpt)
        reuse = A["occ"] >= 2
        o.row(trace=name, ttft_total_nocache_s=fpos.sum() / 1e3, ttft_reuse_nocache_s=fpos[reuse].sum() / 1e3,
              ttft_compulsory_s=fpos[~reuse].sum() / 1e3, f_source=f"calib_c1 {frun[0]}; i>=80 linear extrap")


# ------------------------------------------------------------------ gap（判定）
def _pick(g, col):
    """一個類別裡挑 col 最小的那列；平手挑 SSD 寫入最少的。"""
    g = g.sort_values([col, "ssd_write_blocks"])
    return g.iloc[0]


def cmd_gap(args):
    import pandas as pd
    d = pd.read_csv(os.path.join(OUTDIR, args.sim))
    if args.src_run:
        d = d[d.run_id == args.src_run]
    assert len(d), "沒有資料"
    if "gpu_blocks" not in d.columns:
        d["gpu_blocks"] = 0
    if args.deferred_set == "prereg":
        # 只留第 2 節表格逐字列出的 7 個延後策略（SLRU／ARC／S3-FIFO 的 SSD 規則表上沒寫，取「淘汰時寫」WB）
        keep = {"LRU+WT", "LRU+WB", "2HIT+H2", "LRU+H2", "SLRU+WB", "ARC+WB", "S3FIFO+WB"}
        d = d[(d.klass != "deferred") | d.policy.isin(keep)]
    keys = ["trace", "cpu_frac", "cpu_gibps", "ssd", "gpu_blocks"]
    o = Out(os.path.join(OUTDIR, args.out),
            keys + ["metric", "src_sim_run", "best_deferred", "T_def", "W_def_policy", "W_def", "klass", "x_policy",
                    "T_x", "W_x", "gap_T", "gap_W"])
    rows = []
    for k, g in d.groupby(keys):
        D = g[g.klass == "deferred"]
        for metric in ("ttft_total_s", "ttft_reuse_s", "ttft_cake_s"):
            if g[metric].isna().all():
                continue
            bd = _pick(D, metric)
            Tdef = float(bd[metric])
            ok = D[D[metric] <= 1.05 * Tdef]
            wd = ok.sort_values(["ssd_write_blocks", metric]).iloc[0]
            Wdef = float(wd.ssd_write_blocks)
            for kl in sorted(set(g.klass) - {"deferred"}):
                x = _pick(g[g.klass == kl], metric)
                gT = (Tdef - float(x[metric])) / Tdef
                gW = (Wdef - float(x.ssd_write_blocks)) / Wdef if Wdef > 0 else 0.0
                r = dict(zip(keys, k))
                r.update(metric=metric, src_sim_run=g.run_id.iloc[0], best_deferred=bd.policy, T_def=Tdef,
                         W_def_policy=wd.policy, W_def=Wdef, klass=kl, x_policy=x.policy, T_x=float(x[metric]),
                         W_x=float(x.ssd_write_blocks), gap_T=gT, gap_W=gW)
                o.row(**r)
                rows.append(r)
    R = pd.DataFrame(rows)
    # 事先寫好的判定：SSD=local、GPU 層 0、主模型（ttft_total_s）；每個 trace 6 格取中位數
    P = R[(R.ssd == "local") & (R.gpu_blocks == 0) & (R.metric == "ttft_total_s")]
    ov = Out(os.path.join(OUTDIR, args.verdict_out),
             ["trace", "klass", "n_cells", "median_gap_T", "median_gap_W", "min_gap_T", "max_gap_T", "min_gap_W",
              "max_gap_W", "src_sim_run", "rule"])
    for (tr, kl), g in P.groupby(["trace", "klass"]):
        ov.row(trace=tr, klass=kl, n_cells=len(g), median_gap_T=float(g.gap_T.median()),
               median_gap_W=float(g.gap_W.median()), min_gap_T=float(g.gap_T.min()), max_gap_T=float(g.gap_T.max()),
               min_gap_W=float(g.gap_W.min()), max_gap_W=float(g.gap_W.max()), src_sim_run=g.src_sim_run.iloc[0],
               rule=args.rule_label)
        log(f"{tr:13s} {kl:6s} n={len(g)} med gap_T={g.gap_T.median():.4f} med gap_W={g.gap_W.median():.4f} "
            f"range T [{g.gap_T.min():.4f},{g.gap_T.max():.4f}] W [{g.gap_W.min():.4f},{g.gap_W.max():.4f}]")


# ------------------------------------------------------------------ SCBench
def cmd_scbench(args):
    from transformers import AutoTokenizer
    import glob
    tokp = glob.glob("/mlsteam/data/tiara/hf-cache/hub/models--unsloth--Llama-3.1-8B-Instruct/snapshots/*/")
    assert tokp, "找不到 Llama-3.1-8B tokenizer"
    tok = AutoTokenizer.from_pretrained(tokp[0])
    o = Out(os.path.join(OUTDIR, "d4_scbench.csv"),
            ["task", "n_examples", "turns_median", "turns_min", "turns_max", "ctx_tok_median", "ctx_tok_p90",
             "ctx_blocks_median", "reads_per_write", "frac_reads_lost_2hit", "tok_sampled", "note"])
    for fn in sorted(os.listdir(SCB_DIR)):
        if not fn.endswith(".jsonl"):
            continue
        task = fn[:-6]
        turns, toks = [], []
        n = 0
        with open(os.path.join(SCB_DIR, fn)) as fh:
            for line in fh:
                d = json.loads(line)
                n += 1
                turns.append(len(d["multi_turns"]))
                # scbench_vt 沒有 context 欄，長 context 放在 input（2026-10-10 第一次跑 KeyError，見 D4 §7）
                ctx = d["context"] if "context" in d else d["input"]
                if not isinstance(ctx, str):
                    ctx = json.dumps(ctx)
                if len(toks) < args.max_tok:
                    toks.append(len(tok(ctx, add_special_tokens=False)["input_ids"]))
        turns = np.array(turns)
        toks = np.array(toks)
        rpw = float(turns.mean())  # 每個 context 寫一次、每輪讀一次（multi-turn 模式第 1 輪就是寫入那次）
        o.row(task=task, n_examples=n, turns_median=float(np.median(turns)), turns_min=int(turns.min()),
              turns_max=int(turns.max()), ctx_tok_median=float(np.median(toks)), ctx_tok_p90=float(np.percentile(toks, 90)),
              ctx_blocks_median=float(np.median(np.ceil(toks / BLOCK_TOK))), reads_per_write=rpw,
              frac_reads_lost_2hit=float(np.mean(1.0 / np.maximum(turns - 1, 1) * (turns > 1))),
              tok_sampled=len(toks), note="reads=turns；第 1 輪＝寫入；2-hit 錯過第 2 輪（第 1 次重用）")
        log(task, n, np.median(turns), np.median(toks))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("char")
    a.add_argument("--traces", nargs="+", default=["conversation", "synthetic", "toolagent"])
    a = sp.add_parser("sim")
    a.add_argument("--traces", nargs="+", default=["conversation", "synthetic", "toolagent"])
    a.add_argument("--cpu-frac", nargs="+", type=float, default=[0.1, 0.25, 0.5])
    a.add_argument("--cpu-gibps", nargs="+", type=float, default=[3.69, 11.6])
    a.add_argument("--ssd", nargs="+", default=["local", "nfs"])
    a.add_argument("--cake", action="store_true")
    a.add_argument("--gpu-blocks", nargs="+", type=int, default=[0])
    a.add_argument("--posthoc", action="store_true", help="加事後對照組 KEVa/KEVb（同樣知識、淘汰時才用）")
    a.add_argument("--out", default="d4_sim.csv")
    a = sp.add_parser("gap")
    a.add_argument("--sim", default="d4_sim.csv")
    a.add_argument("--src-run", default="")
    a.add_argument("--out", default="d4_gap.csv")
    a.add_argument("--verdict-out", default="d4_verdict.csv")
    a.add_argument("--rule-label", default="D4 §2.3：SSD=local、GPU=0、ttft_total；6 格中位數")
    a.add_argument("--deferred-set", default="all", choices=["all", "prereg"])
    a = sp.add_parser("nocache")
    a.add_argument("--traces", nargs="+", default=["conversation", "synthetic", "toolagent"])
    a = sp.add_parser("char2")
    a.add_argument("--traces", nargs="+", default=["conversation", "synthetic", "toolagent"])
    a = sp.add_parser("scbench")
    a.add_argument("--max-tok", type=int, default=1000)
    args = ap.parse_args()
    {"char": cmd_char, "char2": cmd_char2, "sim": cmd_sim, "gap": cmd_gap, "nocache": cmd_nocache, "scbench": cmd_scbench}[args.cmd](args)


if __name__ == "__main__":
    main()
