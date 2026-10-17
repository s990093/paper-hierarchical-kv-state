"""x4_sim.py — X4：很多人同時回來時，尾端延遲有沒有預測能解的問題？〔模擬，只用 CPU，MI300X 常數〕

判準與模型寫在 docs/phase3_20261017/explore/X4_concurrency_tail.md §2（開跑前寫死，2026-10-17T09:47:18Z）。
重用 m7_sim（Tier、mk_tier、sim_restore、CB、D2H_BPS）、m7_write_policy（f(i)、tier 參數、SCHED、make_workload）、
m8_trace_oracle（build_access、run_lru、Rec：只用在 V2 驗證）。不改它們。

核心（serial 與事件引擎共用同一份）：
  block 層級的 CPU LRU（同時間先丟前面＝m7 S4L）＋慢層（不限容量、free release 降級）＋Cake 還原（可帶「預取中」的到達時間）。
  QP＝佇列預取（請求在 GPU 佇列裡等時，把它的慢層 block 讀回 CPU）；IP＝閒置時預取（GPU 閒、佇列空時）。
  IP 的候選排序：RP（最近存取）、GP（同 session 上一次間隔）、OP（真實下次存取＝oracle）。機制完全相同，只差資訊。

子命令：
  trace     載入兩個 Mooncake trace（規則 6 斷言），寫描述統計與各子集（p、seed）的大小
  validate  V1–V4（§2.5）；沒過 exit 1
  sweep     §2.3 主掃描（多行程），每格每 seed 每策略一列
  analyze   §2.4 判定
"""
from __future__ import annotations

import argparse
import bisect
import heapq
import json
import math
import os
import random
import sys
from collections import OrderedDict
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
REPO = os.path.dirname(CODE)
# 只用 CPU；成本常數固定用 MI300X（env_3090.sh 會把 M7_RES 指到 3090 的目錄，這裡蓋掉）
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
MI300X_RES = os.path.join(REPO, "results", "m7_write_policy_mi300x")
os.environ["M7_RES"] = MI300X_RES
os.environ["M7_TIER_PARAMS"] = os.path.join(MI300X_RES, "tier_params.json")
os.environ["M7_F_CSV"] = os.path.join(MI300X_RES, "calib_c1.csv")
os.environ.pop("M7_CHUNK_BYTES", None)
os.environ.pop("M7_CPU_GIBPS", None)
sys.path.insert(0, CODE)
import numpy as np  # noqa: E402

from m7_sim import CB, D2H_BPS, mk_tier, sim_restore  # noqa: E402
from m7_write_policy import SCHED, load_f, load_params, make_workload  # noqa: E402

GiB = 1 << 30
MiB = 1 << 20
assert CB == 64 * MiB, CB
BLOCK_TOK = 512
TRACE_DIR = os.environ.get("X4_TRACE_DIR", "/ssd8/hungwei/paper-hkv/datasets/mooncake/traces")
OUT = os.path.join(REPO, "results", "p3_explore", "x4")
RID = os.environ.get("RUN_ID", "adhoc")
RUNS = os.environ.get("TIARA_RUNS", "/ssd8/hungwei/paper-hkv/runs")
INF = float("inf")
B32 = dict(read_GiBps=1.4668, write_GiBps=1.4668, c_ms=0.0, k_read=1.0,
           note="b*=32 Llama：results/m10_precision_mi300x/speed_llama/bw.json（run 20261015-154558-m10-calibf-llama）；"
                "寫＝讀、k_read=1 是假設（限速層沒有寫入干擾校準）")


def now_ts():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_dir():
    d = os.path.join(RUNS, RID)
    os.makedirs(d, exist_ok=True)
    return d


# ================================================================== 成本常數
def params():
    P = load_params()
    P = dict(P)
    P["b32"] = dict(B32)
    return P


def f_ext(nmax=320):
    """MI300X C1 f(i)（秒），i ≥ 80 用最小平方直線外插（同 D4）。回傳 (f, slope_ms, icpt_ms, n_meas)。"""
    f = load_f()
    n = len(f)
    x = np.arange(n, dtype=float)
    slope, icpt = np.polyfit(x, np.array(f) * 1e3, 1)
    out = list(f) + [(icpt + slope * i) / 1e3 for i in range(n, nmax)]
    return out, float(slope), float(icpt), n


# ================================================================== trace 載入（規則 6）
def load_trace(name):
    path = os.path.join(TRACE_DIR, f"{name}_trace.jsonl")
    rows = [json.loads(line) for line in open(path)]
    n = len(rows)
    # (1) block 單位：hash_ids 一個＝512 token
    match = sum(len(r["hash_ids"]) == math.ceil(r["input_length"] / BLOCK_TOK) for r in rows) / n
    assert match >= 0.95, f"{name}: len(hash_ids)==ceil(input_length/512) 只有 {match:.4f}"
    ts = [r["timestamp"] for r in rows]
    assert ts == sorted(ts), f"{name}: timestamp 沒排序"
    # (2) 前綴鏈：同一 hash 永遠在同一位置、前一個 hash 也一樣
    pos_of, prev_of = {}, {}
    bad = 0
    for r in rows:
        h = r["hash_ids"]
        assert len(set(h)) == len(h), f"{name}: 請求內有重複 hash"
        for i, b in enumerate(h):
            p = h[i - 1] if i else None
            if b in pos_of:
                bad += (pos_of[b] != i) + (prev_of[b] != p)
            else:
                pos_of[b], prev_of[b] = i, p
    assert bad == 0, f"{name}: 前綴鏈違反 {bad} 次"
    # parent（D4 cmd_char 的定義）：最大的 k≥1 使 h[k] 是更早某請求最後一個完整 block
    lastfull = {}
    par = [-1] * n
    for ri, r in enumerate(rows):
        h = r["hash_ids"]
        for k in range(len(h) - 1, 0, -1):
            if h[k] in lastfull:
                par[ri] = lastfull[h[k]]
                break
        nf = r["input_length"] // BLOCK_TOK
        if nf >= 2:
            lastfull[h[nf - 1]] = ri
    # (3) timestamp 單位：父子間隔 ÷ 父輸出 token 數（ms/token）
    ratio, dlen, shared_ok = [], [], 0
    npair = 0
    for ri, p in enumerate(par):
        if p < 0:
            continue
        npair += 1
        c, q = rows[ri], rows[p]
        if q["output_length"] >= 50:
            ratio.append((c["timestamp"] - q["timestamp"]) / q["output_length"])
        dlen.append(c["input_length"] - q["input_length"] - q["output_length"])
        nsh = q["input_length"] // BLOCK_TOK   # 共用的 block 數（父請求的完整 block）
        shared_ok += nsh * BLOCK_TOK <= c["input_length"]
    med_ratio = float(np.median(ratio))
    assert 10 <= med_ratio <= 10000, f"{name}: 父子間隔 ÷ 父輸出 token 中位數 {med_ratio:.1f} ms/token，單位不是毫秒？"
    frac_len = float(np.mean(np.array(dlen) >= 0))
    assert frac_len >= 0.7, f"{name}: 子 input − 父 input ≥ 父 output 只有 {frac_len:.3f}（session 重建或 token 單位有問題？）"
    assert shared_ok == npair, f"{name}: 共用 block×512 > 子 input_length 有 {npair - shared_ok} 對"
    # session（前綴鏈的根）
    root = list(range(n))
    for i in range(n):
        if par[i] >= 0:
            root[i] = root[par[i]]
    # §2.7 事後修改 1：抽樣單位＝把分叉點（父請求有 ≥2 個子請求）的邊切掉之後的鏈
    nkid = [0] * n
    for p_ in par:
        if p_ >= 0:
            nkid[p_] += 1
    unit = list(range(n))
    in_unit = [False] * n
    for i in range(n):
        if par[i] >= 0 and nkid[par[i]] == 1:
            unit[i] = unit[par[i]]
            in_unit[i] = True
    span = (ts[-1] - ts[0]) / 1000.0
    meta = dict(trace=name, n_req=n, span_s=span, req_per_s=n / span, unit_match=match, chain_bad=bad,
                n_pairs=npair, frac_returning=npair / n, n_sessions=len(set(root)), n_units=len(set(unit)),
                n_in_unit_returns=sum(in_unit), max_session_req=max(np.bincount(root)), max_unit_req=max(np.bincount(unit)),
                med_gap_ms_per_parent_out_tok=med_ratio, frac_child_in_ge_parent_in_plus_out=frac_len,
                max_blocks=max(len(r["hash_ids"]) for r in rows),
                mean_input_tok=float(np.mean([r["input_length"] for r in rows])))
    # README §4.2（10:01Z 事後補充）：嚴格規則，照 docs/phase3_20261017/research/calc/mooncake_chain2.py：
    #   子請求 j 的最長 L（< len(h)）使某個更早（timestamp 嚴格較早）請求的完整前綴＝h[:L]；
    #   只有該前綴「私有」（含有它的請求 ≤3 個）而且候選父請求唯一時才連。前綴鏈 → h[:L] 由 h[L-1] 唯一決定。
    popc = {}
    for r in rows:
        for b_ in r["hash_ids"]:
            popc[b_] = popc.get(b_, 0) + 1
    cand = {}
    for i, r in enumerate(rows):
        nf = r["input_length"] // BLOCK_TOK
        if nf >= 1:
            cand.setdefault(r["hash_ids"][nf - 1], []).append(i)
    par_s = [-1] * n
    n_amb = 0
    for j, r in enumerate(rows):
        h = r["hash_ids"]
        for L in range(len(h) - 1, 0, -1):
            ps = [i for i in cand.get(h[L - 1], []) if i < j and rows[i]["timestamp"] < r["timestamp"]]
            if ps:
                if popc[h[L - 1]] <= 3 and len(ps) == 1:
                    par_s[j] = ps[0]
                else:
                    n_amb += 1
                break
    unit_s = list(range(n))
    for i in range(n):
        if par_s[i] >= 0:
            unit_s[i] = unit_s[par_s[i]]
    meta.update(n_strict_links=sum(x >= 0 for x in par_s), n_strict_ambiguous=n_amb,
                frac_req_with_strict_child=len({x for x in par_s if x >= 0}) / n,
                n_units_strict=len(set(unit_s)), max_unit_strict_req=max(np.bincount(unit_s)))
    rules = {"loose": (par, unit, in_unit), "strict": (par_s, unit_s, [x >= 0 for x in par_s])}
    return rows, rules, None, meta


def subset(rows, rules, rule, p, seed):
    """以 unit 為單位保留機率 p；同一 seed 的 u 固定，所以 p 小的子集包含在 p 大的裡面。
    rule=loose：§2.7（D4 的 parent，切掉分叉）；rule=strict：README §4.2 的嚴格規則。回來請求＝parent 也在子集裡。"""
    par, unit, in_unit = rules[rule]
    units = sorted(set(unit))
    rng = random.Random(1_000_003 * (seed + 1) + 17)
    u = {r: rng.random() for r in units}
    keep = [i for i in range(len(rows)) if u[unit[i]] < p]
    kidx = {i: k for k, i in enumerate(keep)}
    reqs = []
    for k, i in enumerate(keep):
        r = rows[i]
        pa = par[i]
        pk = kidx.get(pa, -1) if pa >= 0 else -1
        reqs.append(dict(t=r["timestamp"] / 1000.0, h=r["hash_ids"], ret=pk >= 0, ret_in_unit=pk >= 0 and in_unit[i],
                         par=pk, out=r["output_length"]))
    return finish_reqs(reqs), len({unit[i] for i in keep})


def finish_reqs(reqs):
    """n_old（前面出現過的前綴長度）、每個位置的下次使用時間、GP 的預測、session 間隔。"""
    seen = set()
    for q in reqs:
        h = q["h"]
        n_old = 0
        while n_old < len(h) and h[n_old] in seen:
            n_old += 1
        assert all(b not in seen for b in h[n_old:]), "前綴性質被破壞：前綴之後還有出現過的 block"
        q["n_old"] = n_old
        seen.update(h)
    nxt = {}
    for q in reversed(reqs):
        q["nu"] = [nxt.get(b, INF) for b in q["h"]]
        for b in q["h"]:
            nxt[b] = q["t"]
    for q in reqs:
        gap = q["t"] - reqs[q["par"]]["t"] if q["par"] >= 0 else INF
        q["gp"] = q["t"] + gap if gap < INF else INF
    return reqs


# ================================================================== 核心
class Sim:
    """policy: dict(qp=bool, ip=None|'RP'|'GP'|'OP')；promote: 請求結束後把用到的舊 block 放回 CPU。"""

    def __init__(self, reqs, cap_blocks, tiers, f, qp=False, ip=None, promote=True):
        self.R = reqs
        self.C = cap_blocks
        self.cpu_t, self.slow_t = tiers["cpu"], tiers["slow"]
        self.f = f
        self.qp, self.ip, self.promote = qp, ip, promote
        self.cpu = OrderedDict()      # block → None；LRU 在前
        self.slow = set()             # 有慢層副本的 block
        self.avail = {}               # 預取中的 block → 到達 CPU 的時間
        self.prefetched = set()       # 預取進來、還沒被用到的 block
        self.key = {}                 # IP 的預測鍵（越小越早用）
        self.ver = {}
        self.heap = []
        # QP 的增量記帳（2026-10-17 10:13Z：原本每次服務都重建保護集合，過載格 O(N²)；語意不變，見 qp_ref 對拍）
        self.qp_ref = False
        self.memb = {}                # block → 佇列中（含服務中）請求的集合（舊前綴含這個 block）
        self.cnt_pf = {}              # 請求 → 它的舊前綴裡「在慢層、不在 CPU」的 block 數
        self.qtail = 0
        self.S = dict(demote=0, ip_reads=0, qp_reads=0, pf_used=0, pf_evicted=0, ip_stop_guard=0, ip_stop_time=0,
                      ip_stop_empty=0, qp_stop_guard=0, promoted=0, wait_inflight=0)

    # ---- 預測鍵（IP 用；越小＝預測越早被用到）----
    def _mkkey(self, q, i):
        if self.ip == "OP":
            return q["nu"][i]
        if self.ip == "GP":
            return q["gp"]
        if self.ip == "RP":
            return -q["t"]
        return 0.0

    def _push(self, b):
        if self.ip is None:
            return
        k = self.key.get(b, INF)
        if k == INF:
            return
        v = self.ver.get(b, 0) + 1
        self.ver[b] = v
        heapq.heappush(self.heap, (k, v, b))

    def _pf_dec(self, b):
        for k in self.memb.get(b, ()):
            self.cnt_pf[k] -= 1

    def _enqueue(self, k):
        q = self.R[k]
        c = 0
        for b in q["h"][:q["n_old"]]:
            self.memb.setdefault(b, set()).add(k)
            c += (b not in self.cpu) and (b in self.slow)
        self.cnt_pf[k] = c

    def _dequeue(self, k):
        if k not in self.cnt_pf:
            return
        q = self.R[k]
        for b in q["h"][:q["n_old"]]:
            m = self.memb.get(b)
            if m is not None:
                m.discard(k)
                if not m:
                    del self.memb[b]
        del self.cnt_pf[k]

    def _evict_one(self, now):
        b, _ = self.cpu.popitem(last=False)
        self.avail.pop(b, None)
        for k in self.memb.get(b, ()):
            self.cnt_pf[k] += 1
        if b in self.prefetched:
            self.prefetched.discard(b)
            self.S["pf_evicted"] += 1
        if b not in self.slow:
            self.slow_t.reserve_write(CB, now)
            self.slow.add(b)
            self.S["demote"] += 1
        self._push(b)
        return b

    # ---- 還原（Cake，m7_sim.sim_restore 的邏輯＋預取中的到達時間）----
    def restore(self, q, t0):
        n = q["n_old"]
        h = q["h"]
        src, av = [], []
        for i in range(n):
            b = h[i]
            if b in self.cpu:
                src.append(self.cpu_t)
                av.append(self.avail.get(b, 0.0))
            else:
                src.append(self.slow_t)
                av.append(0.0)
        return cake(n, src, av, self.f, t0)

    # ---- 請求結束 ----
    def after_round(self, q, now):
        h, n_old = q["h"], q["n_old"]
        cpu = self.cpu
        for i in range(len(h)):
            b = h[i]
            self.key[b] = self._mkkey(q, i)
            if b in self.prefetched:
                self.prefetched.discard(b)
                self.S["pf_used"] += 1
            if i < n_old:
                if b in cpu:
                    cpu.move_to_end(b)
                    self.avail.pop(b, None)
                elif self.promote:
                    self._pf_dec(b)
                    cpu[b] = None
                    self.cpu_t.reserve_write(CB, now)
                    self.S["promoted"] += 1
            else:
                cpu[b] = None
                self.cpu_t.reserve_write(CB, now)
        while len(cpu) > self.C:
            self._evict_one(now)

    # ---- 佇列預取：服務中的請求還原完之後，替排隊中的請求把慢層 block 讀回 CPU，到 t_end 為止 ----
    # 2026-10-17 10:13Z 修正：只讀「已經存在於慢層」的 block（原版會去讀父請求還在排隊、還沒寫出來的 block）
    def queue_prefetch(self, j, t_go, t_end):
        if self.qp_ref:
            return self.queue_prefetch_ref(j, t_go, t_end)
        R = self.R
        k = max(self.qtail, j)
        while k < len(R) and (k == j or R[k]["t"] < t_end):
            self._enqueue(k)
            k += 1
        self.qtail = k
        for k in range(j + 1, self.qtail):
            if self.cnt_pf[k] == 0:
                continue
            q = R[k]
            for i in range(q["n_old"] - 1, -1, -1):      # 由後往前（Cake 的 I/O 方向）
                b = q["h"][i]
                if b in self.cpu or b not in self.slow:
                    continue
                st = max(self.slow_t.t_free, q["t"], t_go)
                if st >= t_end:
                    return
                if len(self.cpu) >= self.C:
                    v = next(iter(self.cpu))
                    if self.memb.get(v):
                        self.S["qp_stop_guard"] += 1
                        return
                    self._evict_one(st)
                done = self.slow_t.reserve_read(CB, st)
                self._pf_dec(b)
                self.cpu[b] = None
                self.avail[b] = done
                self.prefetched.add(b)
                self.S["qp_reads"] += 1

    def queue_prefetch_ref(self, j, t_go, t_end):
        """參考版（每次重建保護集合，O(N²)）：只用來對拍 queue_prefetch。"""
        R = self.R
        queued = []
        k = j + 1
        while k < len(R) and R[k]["t"] < t_end:
            queued.append(k)
            k += 1
        if not queued:
            return
        prot = set(R[j]["h"][:R[j]["n_old"]])
        for k in queued:
            prot.update(R[k]["h"][:R[k]["n_old"]])
        for k in queued:
            q = R[k]
            for i in range(q["n_old"] - 1, -1, -1):
                b = q["h"][i]
                if b in self.cpu or b not in self.slow:
                    continue
                st = max(self.slow_t.t_free, q["t"], t_go)
                if st >= t_end:
                    return
                if len(self.cpu) >= self.C:
                    v = next(iter(self.cpu))
                    if v in prot:
                        self.S["qp_stop_guard"] += 1
                        return
                    self._evict_one(st)
                done = self.slow_t.reserve_read(CB, st)
                self.cpu[b] = None
                self.avail[b] = done
                self.prefetched.add(b)
                self.S["qp_reads"] += 1

    # ---- 閒置時預取：GPU 閒、佇列空，[t0, t1) 之間一次送一個 block 的讀；t1 到了就停 ----
    def idle_prefetch(self, t0, t1):
        if self.ip is None or t1 <= t0:
            return
        cpu = self.cpu
        while True:
            st = max(t0, self.slow_t.t_free)
            if st >= t1:
                self.S["ip_stop_time"] += 1
                return
            c = None
            while self.heap:
                k, v, b = self.heap[0]
                if b in cpu or self.ver.get(b) != v:
                    heapq.heappop(self.heap)
                    continue
                if self.ip == "GP" and k < st:            # 已過期：GP 不預取
                    heapq.heappop(self.heap)
                    continue
                c = b
                break
            if c is None:
                self.S["ip_stop_empty"] += 1
                return
            kc = self.heap[0][0]
            if len(cpu) >= self.C:
                vb = next(iter(cpu))
                if not (self.key.get(vb, INF) > kc):     # 只有被換掉的預測比候選晚才換
                    self.S["ip_stop_guard"] += 1
                    return
                self._evict_one(st)
            heapq.heappop(self.heap)
            done = self.slow_t.reserve_read(CB, st)
            self._pf_dec(c)
            cpu[c] = None
            self.avail[c] = done
            self.prefetched.add(c)
            self.S["ip_reads"] += 1

    # ---- 服務一個請求 ----
    def serve(self, j, t_arr, t_go):
        q = self.R[j]
        n_old = q["n_old"]
        n_new = len(q["h"]) - n_old
        miss_cpu = sum(1 for b in q["h"][:n_old] if b not in self.cpu)
        inflight = sum(1 for b in q["h"][:n_old] if self.avail.get(b, 0.0) > t_go)
        self.S["wait_inflight"] += inflight
        t_rend, meet, nc, ns = self.restore(q, t_go)
        t_new = sum(self.f[n_old:n_old + n_new])
        ttft = t_rend + t_new - t_arr
        n_cap = n_new + (miss_cpu if self.promote else 0)
        T = t_arr + ttft + n_cap * CB / D2H_BPS
        return dict(j=j, t_arr=t_arr, t_go=t_go, T=T, ttft=ttft, qwait=t_go - t_arr, restore=t_rend - t_go,
                    prefill=t_new, ret=q["ret"], ret_in=q.get("ret_in_unit", q["ret"]), n_old=n_old, n_new=n_new, meet=meet, ld_cpu=nc, ld_slow=ns,
                    miss_cpu=miss_cpu, inflight=inflight)

    # ---- 序列模式（V1、V4）：一個接一個，m7_sim.simulate 的時間軸 ----
    def run_serial(self, t_start=100.0, ov=0.01, gap=0.0):
        rows = []
        t = t_start
        for j in range(len(self.R)):
            r = self.serve(j, t, t)
            self.after_round(self.R[j], r["T"])
            rows.append(r)
            t = r["T"] + ov + gap
        return rows

    # ---- 事件引擎：原始到達時間、一張 GPU、FIFO ----
    def run_events(self):
        rows = []
        R = self.R
        gpu_free = R[0]["t"] if R else 0.0
        busy = 0.0
        for j in range(len(R)):
            ta = R[j]["t"]
            if ta >= gpu_free:
                self.idle_prefetch(gpu_free, ta)
                t_go = ta
            else:
                t_go = gpu_free
            r = self.serve(j, ta, t_go)
            if self.qp:
                self.queue_prefetch(j, t_go, r["T"])
            self.after_round(R[j], r["T"])
            if self.qp:
                self._dequeue(j)
            busy += r["T"] - t_go
            gpu_free = r["T"]
            rows.append(r)
        return rows, busy


def cake(n, src, av, f, t0):
    """m7_sim.sim_restore（mode=cake）＋每個 block 最早可開始載入的時間 av[i]（預取中）。av 全 0 時逐位元相同。"""
    if n == 0:
        return t0, 0, 0, 0
    nc = ns = 0
    p, q, rt, lt, lstop = 0, n - 1, t0, t0, False
    while p <= q:
        if lstop or rt <= lt:
            rt += f[p]
            p += 1
        else:
            T = src[q]
            if T is None:
                lstop = True
                continue
            lt = T.reserve_read(CB, lt if av[q] <= lt else av[q])
            if T.name.startswith("cpu"):
                nc += 1
            else:
                ns += 1
            q -= 1
    return max(rt, lt), p, nc, ns


def mk_tiers(P, slow, cpu_gibps=None):
    cpu = mk_tier("cpu", P["cpu"], "share", cpu_gibps)
    s = mk_tier(f"ssd_{slow}", P[slow], "share")
    return {"cpu": cpu, "slow": s}


# ================================================================== 摘要
def pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


def summarize(rows, busy, sim, P, slow, cap_blocks, ws_blocks):
    ret = [r for r in rows if r["ret"]]
    tt = np.array([r["ttft"] for r in ret])
    rs = np.array([r["restore"] for r in ret])
    qw = np.array([r["qwait"] for r in ret])
    t0 = rows[0]["t_arr"]
    t1 = max(r["T"] for r in rows)
    span = t1 - t0
    W = P[slow]["write_GiBps"] * GiB
    new_b = sum(r["n_new"] for r in rows) * CB
    tin = np.array([r["ttft"] for r in rows if r["ret_in"]])
    out = dict(n_req=len(rows), n_ret=len(ret), n_ret_in=len(tin), p99_in=pct(tin, 99),
               mean_in=float(tin.mean()) if len(tin) else float("nan"), p99=pct(tt, 99), p90=pct(tt, 90), p50=pct(tt, 50),
               mean=float(tt.mean()) if len(tt) else float("nan"), max=float(tt.max()) if len(tt) else float("nan"),
               restore_p99=pct(rs, 99), restore_mean=float(rs.mean()) if len(rs) else float("nan"),
               qwait_p99=pct(qw, 99), qwait_mean=float(qw.mean()) if len(qw) else float("nan"),
               svc_p99=pct(tt - qw, 99), all_ttft_sum=float(sum(r["ttft"] for r in rows)),
               all_p99=pct([r["ttft"] for r in rows], 99), span_s=span, gpu_util=busy / span if span > 0 else float("nan"),
               rho_kv=new_b / (W * span), rho_strat=sim.S["demote"] * CB / (W * span),
               cap_blocks=cap_blocks, cap_GiB=cap_blocks * CB / GiB, ws_GiB=ws_blocks * CB / GiB,
               cap_pct_ws=100.0 * cap_blocks / ws_blocks, ld_slow=sum(r["ld_slow"] for r in rows),
               ld_cpu=sum(r["ld_cpu"] for r in rows), recompute_old=sum(r["meet"] for r in ret),
               miss_cpu_ret=sum(r["miss_cpu"] for r in ret))
    out.update(sim.S)
    return out


# ================================================================== trace 子命令
def cmd_trace(a):
    import pandas as pd
    f, slope, icpt, nmeas = f_ext()
    rows_out, sub_out = [], []
    for name in a.traces:
        rows, rules, _, meta = load_trace(name)
        nblk = sum(len(r["hash_ids"]) for r in rows)
        nextra = sum(max(0, len(r["hash_ids"]) - nmeas) for r in rows)
        meta.update(frac_block_positions_extrapolated=nextra / nblk, f_slope_ms=slope, f_icpt_ms=icpt)
        rows_out.append(dict(run_id=RID, ts=now_ts(), **meta))
        print(meta, flush=True)
        for rule, p, seed in [(ru, p_, sd) for ru in a.rules for p_ in a.p for sd in a.seeds]:
                reqs, ns = subset(rows, rules, rule, p, seed)
                ws = len({b for q in reqs for b in q["h"]})
                span = (reqs[-1]["t"] - reqs[0]["t"])
                sub_out.append(dict(run_id=RID, ts=now_ts(), trace=name, rule=rule, p=p, seed=seed, n_units=ns, n_req=len(reqs),
                                    n_ret=sum(q["ret"] for q in reqs), n_ret_in_unit=sum(q["ret_in_unit"] for q in reqs),
                                    req_per_s=len(reqs) / span,
                                    ret_per_s=sum(q["ret"] for q in reqs) / span, ws_blocks=ws, ws_GiB=ws * CB / GiB,
                                    new_blocks=sum(len(q["h"]) - q["n_old"] for q in reqs),
                                    old_blocks_ret=sum(q["n_old"] for q in reqs if q["ret"]),
                                    gpu_new_prefill_s=sum(sum(f[q["n_old"]:len(q["h"])]) for q in reqs),
                                    ge1000_units=ns >= 1000))
    pd.DataFrame(rows_out).to_csv(os.path.join(OUT, "x4_trace_desc.csv"), index=False)
    d = pd.DataFrame(sub_out)
    d.to_csv(os.path.join(OUT, "x4_subsets.csv"), index=False)
    pd.set_option("display.width", 250)
    print(d.groupby(["trace", "rule", "p"])[["n_units", "n_req", "n_ret", "n_ret_in_unit", "req_per_s", "ws_GiB",
                                     "gpu_new_prefill_s"]].agg(["mean", "min", "max"]).round(1).to_string())


# ================================================================== validate
def chatdoc_reqs(workload, seed, n_sess=8):
    sched = SCHED[workload]
    ev, _ = make_workload(seed, n_sess, len(sched))
    reqs = []
    for s, r in ev:
        n_new = sum(sched[:r])
        reqs.append(dict(t=0.0, h=[(s, i) for i in range(n_new)], ret=r >= 2, ret_in_unit=r >= 2, par=-1, out=0))
    reqs = finish_reqs(reqs)
    for q, (s, r) in zip(reqs, ev):
        assert q["n_old"] == sum(sched[:r - 1]), (q["n_old"], s, r)
    return reqs, sched


def m7_median(xs):
    xs = sorted(xs)
    n = len(xs)
    return (xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])) if n else float("nan")


def cmd_validate(a):
    import pandas as pd
    P = params()
    f, _, _, _ = f_ext()
    ok_all = True
    # ---- V1：r1_sim 的全部 S4L、free 格 ----
    ref = pd.read_csv(os.path.join(REPO, "results", "m7_explore_mi300x", "r1_sim_llama31_8b.csv"))
    ref = ref[(ref.strategy == "S4L") & (ref.release == "free")]
    v1 = []
    f80 = f[:80]
    for m in ref.itertuples():
        reqs, sched = chatdoc_reqs(m.workload, int(m.wl_seed))
        cap = int(m.cpu_frac * 8 * sum(sched))
        cg = None if abs(m.cpu_gibps - P["cpu"]["read_GiBps"]) < 1e-3 else float(m.cpu_gibps)
        sim = Sim(reqs, cap, mk_tiers(P, m.ssd_dev, cg), f80, promote=False)
        rows = sim.run_serial()
        ret = sorted(r["ttft"] for r in rows if r["ret"])
        med, p90 = m7_median(ret), ret[int(0.9 * (len(ret) - 1))]
        e1, e2 = 100 * (med / m.median - 1), 100 * (p90 / m.p90 - 1)
        v1.append(dict(run_id=RID, ts=now_ts(), ref_run_id=m.run_id, workload=m.workload, ssd_dev=m.ssd_dev,
                       cpu_frac=m.cpu_frac, wl_seed=m.wl_seed, cpu_gibps=m.cpu_gibps, ref_median=m.median, x4_median=med,
                       err_median_pct=e1, ref_p90=m.p90, x4_p90=p90, err_p90_pct=e2, ref_n_demote=m.n_demote,
                       x4_n_demote=sim.S["demote"], within_1pct=abs(e1) <= 1 and abs(e2) <= 1))
    v1 = pd.DataFrame(v1)
    v1.to_csv(os.path.join(OUT, "x4_validate_v1.csv"), index=False)
    frac = v1.within_1pct.mean()
    print(f"V1: {int(v1.within_1pct.sum())}/{len(v1)} 格 ≤1%（需要 ≥95%）；|err| 中位 {v1.err_median_pct.abs().median():.3g}%，"
          f"最大 {v1.err_median_pct.abs().max():.3g}%／p90 最大 {v1.err_p90_pct.abs().max():.3g}%；"
          f"demote 相同 {int((v1.ref_n_demote == v1.x4_n_demote).sum())}/{len(v1)}", flush=True)
    ok_all &= frac >= 0.95
    # ---- V3：cake() vs m7_sim.sim_restore，隨機版面 ----
    rng = random.Random(7)
    worst = 0.0
    for it in range(2000):
        slow = rng.choice(["local", "nfs", "b32"])
        cg = rng.choice([None, 11.6, 3.69, 1.0])
        n = rng.randint(0, 79)
        lay = [rng.random() < rng.random() for _ in range(n)]
        t0 = rng.uniform(0, 50)
        pre = rng.uniform(0, 3)          # tier 的初始忙碌（讀、寫）
        res = []
        for fn in ("m7", "x4"):
            T = mk_tiers(P, slow, cg)
            T["slow"].t_free = t0 + pre
            T["slow"].t_free_w = t0 + 2 * pre
            T["cpu"].t_free = t0 + 0.1 * pre
            loc = [T["cpu"] if x else T["slow"] for x in lay]
            if fn == "m7":
                res.append(sim_restore(n, loc, f80, t0, "cake"))
            else:
                res.append(cake(n, loc, [0.0] * n, f80, t0))
        worst = max(worst, abs(res[0][0] - res[1][0]))
        assert res[0][1:] == res[1][1:], (res, it)
    print(f"V3: 2000 個隨機版面，完成時間最大差 {worst:.3g} s（需要 0）", flush=True)
    ok_all &= worst == 0.0
    v34 = [dict(run_id=RID, ts=now_ts(), check="V3_cake_vs_sim_restore", n=2000, max_abs_diff_s=worst, ok=worst == 0.0)]
    # ---- V4：事件引擎＋到達間隔很大＋不預取＝序列模式 ----
    worst4 = worst4b = 0.0
    for wl in ("chat", "doc"):
        for slow in ("local", "nfs", "b32"):
            for promote in (False, True):
                for seed in range(3):
                    reqs, sched = chatdoc_reqs(wl, seed)
                    cap = int(0.25 * 8 * sum(sched))
                    s1 = Sim(reqs, cap, mk_tiers(P, slow), f80, promote=promote)
                    r1 = s1.run_serial(ov=1e4)
                    reqs2, _ = chatdoc_reqs(wl, seed)
                    for k, q in enumerate(reqs2):
                        q["t"] = 100.0 + 2e4 * k
                    s2 = Sim(reqs2, cap, mk_tiers(P, slow), f80, promote=promote)
                    r2, _ = s2.run_events()
                    d = max(abs(x["ttft"] - y["ttft"]) for x, y in zip(r1, r2))
                    worst4 = max(worst4, d)
                    # 2026-10-17 修正（§6）：序列模式改用和事件引擎完全相同的絕對到達時間，避免 1e6 s 量級的浮點捨入
                    reqs3, _ = chatdoc_reqs(wl, seed)
                    s3 = Sim(reqs3, cap, mk_tiers(P, slow), f80, promote=promote)
                    r3 = []
                    for k in range(len(reqs3)):
                        t = 100.0 + 2e4 * k
                        x = s3.serve(k, t, t)
                        s3.after_round(reqs3[k], x["T"])
                        r3.append(x)
                    worst4b = max(worst4b, max(abs(x["ttft"] - y["ttft"]) for x, y in zip(r3, r2)))
    print(f"V4（原版，序列模式時間軸從 100 s 起、ov=1e4）：每請求 TTFT 最大差 {worst4:.3g} s（原門檻 <1e-9；"
          f"2026-10-17 第一次跑 2.5e-9 沒過，診斷為絕對時間 ~1e6 s 的浮點捨入）", flush=True)
    print(f"V4（修正版，同一組絕對到達時間）：最大差 {worst4b:.3g} s（需要 0）", flush=True)
    ok_all &= worst4b == 0.0
    v34.append(dict(run_id=RID, ts=now_ts(), check="V4_events_vs_serial_original_ov1e4", n=2 * 3 * 2 * 3,
                    max_abs_diff_s=worst4, ok=worst4 < 1e-9))
    v34.append(dict(run_id=RID, ts=now_ts(), check="V4_events_vs_serial_same_abs_times", n=2 * 3 * 2 * 3,
                    max_abs_diff_s=worst4b, ok=worst4b == 0.0))
    # ---- V2：x4 載入 → m8_trace_oracle.run_lru 的命中數＝D4 的 cpu_hits ----
    import m8_trace_oracle as D4
    d4 = pd.concat([pd.read_csv(os.path.join(REPO, "results", "m8_directions", x))
                    for x in ("d4_sim.csv", "d4_sim2.csv", "d4_sim3.csv")], ignore_index=True)
    if "gpu_blocks" in d4:
        d4["gpu_blocks"] = d4["gpu_blocks"].fillna(0)
    d4 = d4[(d4.cpu_policy == "LRU") & (d4.gpu_blocks == 0) & d4.trace.isin(["conversation", "toolagent"])]
    d4 = d4.drop_duplicates(["trace", "cpu_frac", "cpu_cap_blocks", "cpu_hits"])
    v2 = []
    for name in ("conversation", "toolagent"):
        rows, rules, _, meta = load_trace(name)
        A = D4.build_access(rows)
        reqs, _ = subset(rows, rules, "loose", 1.01, 0)
        for m in d4[d4.trace == name].itertuples():
            C = int(m.cpu_cap_blocks)
            assert C == max(1, int(A["B"] * m.cpu_frac)), (C, A["B"], m.cpu_frac)
            hits = int(D4.run_lru(A, C, D4.Rec(A)).sum())
            T = mk_tiers(P, "local")
            s = Sim(reqs, C, T, f, promote=True)
            rr = s.run_serial()
            # x4 自己以請求為單位的 LRU：請求開始時，舊 block 在 CPU 的個數（不管 Cake 有沒有讀它）
            v2.append(dict(run_id=RID, ts=now_ts(), trace=name, cpu_frac=m.cpu_frac, cap_blocks=C, d4_run_id=m.run_id,
                           d4_cpu_hits=int(m.cpu_hits), x4_loader_run_lru_hits=hits, exact=hits == int(m.cpu_hits),
                           x4_request_lru_cpu_resident_old=s_cpu_resident(s, reqs, rr)))
            print("V2", v2[-1], flush=True)
    v2 = pd.DataFrame(v2)
    v2["x4_vs_d4_pct"] = 100 * (v2.x4_request_lru_cpu_resident_old / v2.d4_cpu_hits - 1)
    v2.to_csv(os.path.join(OUT, "x4_validate_v2.csv"), index=False)
    print(f"V2: {int(v2.exact.sum())}/{len(v2)} 格完全相同（需要全部）", flush=True)
    ok_all &= bool(v2.exact.all())
    pd.DataFrame(v34).to_csv(os.path.join(OUT, "x4_validate_v34.csv"), index=False)
    print("VALIDATE", "PASS" if ok_all else "FAIL", flush=True)
    pd.DataFrame(v34).to_csv(os.path.join(OUT, "x4_validate_v34.csv"), index=False)
    sys.exit(0 if ok_all else 1)


def s_cpu_resident(sim, reqs, rows):
    return int(sum(r["n_old"] - r["miss_cpu"] for r in rows))


# ================================================================== sweep
POLICIES = {"NP": (False, None), "QP": (True, None), "RP": (True, "RP"), "GP": (True, "GP"), "OP": (True, "OP"),
            "OPn": (False, "OP")}
_CACHE = {}


def _get_subset(name, p, seed, rule="loose"):
    k = (name, p, seed, rule)
    if k not in _CACHE:
        if name not in _CACHE:
            _CACHE[name] = load_trace(name)
        rows, rules, _, meta = _CACHE[name]
        reqs, ns = subset(rows, rules, rule, p, seed)
        ws = len({b for q in reqs for b in q["h"]})
        _CACHE[k] = (reqs, ns, ws)
    return _CACHE[k]


def _job(cfg):
    P = params()
    f, _, _, _ = f_ext()
    if cfg.get("fscale", 1.0) != 1.0:
        f = [x * cfg["fscale"] for x in f]
    reqs, ns, ws = _get_subset(cfg["trace"], cfg["p"], cfg["seed"], cfg.get("rule", "loose"))
    cap = int(round(cfg["cpu_GiB"] * GiB / CB))
    out = []
    for pol in cfg["policies"]:
        qp, ip = POLICIES[pol]
        sim = Sim(reqs, cap, mk_tiers(P, cfg["slow"]), f, qp=qp, ip=ip, promote=True)
        rows, busy = sim.run_events()
        sm = summarize(rows, busy, sim, P, cfg["slow"], cap, ws)
        out.append(dict({k: v for k, v in cfg.items() if k != "policies"}, policy=pol, n_units=ns, **sm))
    return out


def cmd_sweep(a):
    import multiprocessing as mp
    import pandas as pd
    cfgs = []
    for name in a.traces:
        for rule in a.rules:
            for p in a.p:
                for seed in a.seeds:
                    for cg in a.cpu_GiB:
                        for slow in a.slow:
                            cfgs.append(dict(trace=name, rule=rule, p=p, seed=seed, cpu_GiB=cg, slow=slow,
                                             fscale=a.fscale, policies=a.policies))
    # 同一個子集的 job 排在一起，讓每個 worker 的快取有用
    print("jobs", len(cfgs), "sims", len(cfgs) * len(a.policies), flush=True)
    t0 = datetime.now()
    res = []
    with mp.Pool(a.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, cfgs, chunksize=1)):
            res.extend(r)
            if i % 20 == 0:
                print(i, "/", len(cfgs), datetime.now() - t0, flush=True)
    d = pd.DataFrame(res)
    d.insert(0, "ts", now_ts())
    d.insert(0, "run_id", RID)
    fn = os.path.join(run_dir(), a.out)
    d.to_csv(fn, index=False)
    print(fn, len(d), "rows", datetime.now() - t0, flush=True)


# ================================================================== analyze
CELL = ["trace", "rule", "p", "cpu_GiB", "slow", "fscale"]
RIVALS = ["NP", "QP", "RP", "GP"]


def cmd_analyze(a):
    import pandas as pd
    d = pd.concat([pd.read_csv(x) for x in a.inp], ignore_index=True)
    if "rule" not in d:
        d["rule"] = "loose"      # 第一次主掃描（20261017-175817-p3x4-sweep）只有 loose
    d["rule"] = d["rule"].fillna("loose")
    rows = []
    for k, g in d.groupby(CELL + ["seed"]):
        pv = g.set_index("policy")
        if not all(x in pv.index for x in RIVALS + ["OP"]):
            continue
        rec = dict(zip(CELL + ["seed"], k))
        for metric in ("p99", "mean", "p90"):
            riv = pv.loc[RIVALS, metric]
            best = riv.idxmin()
            rec[f"best_rival_{metric}"] = best
            rec[f"rival_{metric}"] = float(riv.min())
            rec[f"op_{metric}"] = float(pv.loc["OP", metric])
            rec[f"hr_{metric}"] = (float(riv.min()) - float(pv.loc["OP", metric])) / float(riv.min())
            rec[f"op_slower_than_any_{metric}"] = bool((pv.loc["OP", metric] > riv * (1 + 1e-9)).any())
        for m2 in ("restore_p99", "svc_p99", "qwait_p99", "restore_mean"):
            r2 = pv.loc[RIVALS, m2]
            rec[f"hr_{m2}"] = (float(r2.min()) - float(pv.loc["OP", m2])) / float(r2.min()) if float(r2.min()) > 0 else 0.0
            rec[f"NP_{m2}"] = float(pv.loc["NP", m2])
        riv_in = pv.loc[RIVALS, "p99_in"]
        rec["hr_p99_in"] = (float(riv_in.min()) - float(pv.loc["OP", "p99_in"])) / float(riv_in.min())
        for pol in pv.index:
            rec[f"{pol}_p99"] = float(pv.loc[pol, "p99"])
        rec["util_np"] = float(pv.loc["NP", "gpu_util"])
        rec["rho_kv"] = float(pv.loc["NP", "rho_kv"])
        rec["n_units"] = int(pv.loc["NP", "n_units"])
        rec["n_ret"] = int(pv.loc["NP", "n_ret"])
        rec["cap_pct_ws"] = float(pv.loc["NP", "cap_pct_ws"])
        rec["ws_GiB"] = float(pv.loc["NP", "ws_GiB"])
        if "OPn" in pv.index:
            rec["hr_OPn_vs_NP_p99"] = (float(pv.loc["NP", "p99"]) - float(pv.loc["OPn", "p99"])) / float(pv.loc["NP", "p99"])
            rec["hr_QP_vs_NP_p99"] = (float(pv.loc["NP", "p99"]) - float(pv.loc["QP", "p99"])) / float(pv.loc["NP", "p99"])
        rows.append(rec)
    s = pd.DataFrame(rows)
    s.insert(0, "ts", now_ts())
    s.insert(0, "run_id", RID)
    s["src"] = ";".join(os.path.basename(os.path.dirname(x)) for x in a.inp)
    s.to_csv(os.path.join(run_dir(), a.out_seed), index=False)
    agg_spec = dict(n_seeds=("seed", "size"), util=("util_np", "mean"), util_max=("util_np", "max"),
                    rho_kv=("rho_kv", "mean"), n_units_min=("n_units", "min"), n_ret=("n_ret", "mean"),
                    cap_pct_ws=("cap_pct_ws", "mean"), ws_GiB=("ws_GiB", "mean"))
    for metric in ("p99", "mean", "p90"):
        agg_spec[f"hr_{metric}_med"] = (f"hr_{metric}", "median")
        agg_spec[f"hr_{metric}_min"] = (f"hr_{metric}", "min")
        agg_spec[f"n20_{metric}"] = (f"hr_{metric}", lambda x: int((x >= 0.20).sum()))
        agg_spec[f"n5_{metric}"] = (f"hr_{metric}", lambda x: int((x >= 0.05).sum()))
        agg_spec[f"rival_{metric}_med"] = (f"rival_{metric}", "median")
        agg_spec[f"op_{metric}_med"] = (f"op_{metric}", "median")
        agg_spec[f"best_rival_{metric}"] = (f"best_rival_{metric}", lambda x: x.value_counts().index[0])
        agg_spec[f"op_violations_{metric}"] = (f"op_slower_than_any_{metric}", "sum")
    agg_spec["hr_p99_in_med"] = ("hr_p99_in", "median")
    for m2 in ("restore_p99", "svc_p99", "qwait_p99", "restore_mean"):
        agg_spec[f"hr_{m2}_med"] = (f"hr_{m2}", "median")
        agg_spec[f"NP_{m2}_med"] = (f"NP_{m2}", "median")
    if "hr_OPn_vs_NP_p99" in s:
        agg_spec["hr_OPn_vs_NP_p99_med"] = ("hr_OPn_vs_NP_p99", "median")
        agg_spec["hr_QP_vs_NP_p99_med"] = ("hr_QP_vs_NP_p99", "median")
    for pol in ("NP", "QP", "RP", "GP", "OP", "OPn"):
        if f"{pol}_p99" in s:
            agg_spec[f"{pol}_p99_med"] = (f"{pol}_p99", "median")
    agg = s.groupby(CELL).agg(**agg_spec).reset_index()
    need = agg.n_seeds.apply(lambda n: 2 if n == 3 else (n // 2 + 1))
    for metric in ("p99", "mean", "p90"):
        agg[f"pass20_{metric}"] = (agg[f"n20_{metric}"] >= need) & (agg[f"hr_{metric}_med"] >= 0.20)
        agg[f"pass5_{metric}"] = (agg[f"n5_{metric}"] >= need) & (agg[f"hr_{metric}_med"] >= 0.05)
    agg["eligible"] = (agg.util <= 0.80) & (agg.n_units_min >= 1000)
    agg["rho_ge1"] = agg.rho_kv >= 1.0
    agg.insert(0, "ts", now_ts())
    agg.insert(0, "run_id", RID)
    agg["src"] = s["src"].iloc[0]
    agg.to_csv(os.path.join(OUT, a.out_cfg), index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 60)
    el = agg[agg.eligible]
    cols = CELL + ["util", "rho_kv", "n_units_min", "cap_pct_ws", "rival_p99_med", "op_p99_med", "best_rival_p99",
                   "hr_p99_med", "hr_p99_min", "n20_p99", "hr_mean_med", "hr_p90_med"]
    print(f"cells {len(agg)}  eligible {len(el)}  eligible&pass20_p99 {int(el.pass20_p99.sum())}  "
          f"eligible&pass5_p99 {int(el.pass5_p99.sum())}")
    print(el.sort_values("hr_p99_med", ascending=False)[cols].head(30).to_string())
    def verdict(e, metric):
        if len(e) == 0:
            return "（沒有能進判定的格）"
        if e[f"pass20_{metric}"].any():
            return "有看頭"
        if (e[f"hr_{metric}_med"] < 0.05).all():
            return "死路"
        return "可能"
    vrows = []
    for metric in ("p99", "mean", "p90", "p99_in", "restore_p99", "svc_p99"):
        if f"hr_{metric}_med" not in el:
            continue
        if metric not in ("p99", "mean", "p90"):
            el = el.assign(**{f"pass20_{metric}": False})
        for grp, e in [("ALL", el)] + [(f"rule={r}", el[el.rule == r]) for r in sorted(el.rule.unique())] + \
                [(f"trace={t}", el[el.trace == t]) for t in sorted(el.trace.unique())]:
            v = verdict(e, metric)
            mx = e[f"hr_{metric}_med"].max() if len(e) else float("nan")
            n5 = int((e[f"hr_{metric}_med"] >= 0.05).sum()) if len(e) else 0
            tag = "（§2.4 判定）" if metric == "p99" else "（只報告）"
            print(f"VERDICT[{metric}]{tag} {grp}: {v}  eligible cells={len(e)}  max hr_med={mx:.4f}  cells hr_med>=5%: {n5}")
            vrows.append(dict(run_id=RID, ts=now_ts(), metric=metric, group=grp, verdict=v, n_eligible=len(e),
                              max_hr_med=mx, n_cells_hr_med_ge5=n5, src=s["src"].iloc[0]))
    pd.DataFrame(vrows).to_csv(os.path.join(OUT, a.out_cfg.replace("_cfg", "_verdicts")), index=False)
    nonel = agg[~agg.eligible]
    print(f"report-only cells: {len(nonel)} (util>0.8: {int((agg.util > 0.8).sum())}, units<1000: "
          f"{int((agg.n_units_min < 1000).sum())}); max hr_p99_med among them {nonel.hr_p99_med.max():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["trace", "validate", "sweep", "analyze"])
    ap.add_argument("--traces", nargs="*", default=["conversation", "toolagent"])
    ap.add_argument("--p", type=float, nargs="*", default=[0.1, 0.15, 0.2, 0.25, 0.3, 0.4])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--cpu-GiB", dest="cpu_GiB", type=float, nargs="*", default=[32, 64, 128, 256, 512])
    ap.add_argument("--slow", nargs="*", default=["local", "b32", "nfs"])
    ap.add_argument("--policies", nargs="*", default=list(POLICIES))
    ap.add_argument("--fscale", type=float, default=1.0)
    ap.add_argument("--rules", nargs="*", default=["loose"])
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", default="x4_sweep.csv")
    ap.add_argument("--inp", nargs="*")
    ap.add_argument("--out-seed", default="x4_verdict_seed.csv")
    ap.add_argument("--out-cfg", default="x4_verdict_cfg.csv")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {"trace": cmd_trace, "validate": cmd_validate, "sweep": cmd_sweep, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
