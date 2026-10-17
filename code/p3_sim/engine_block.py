"""engine_block.py — block 引擎（serve=fifo1）：X4 的模型（x4_sim.Sim 的 run_events）＋格式＋記帳。給 E6（佇列預取）與它的 E2 基準。

和 X4 一樣（照抄算式，才能對 X4 的 QP 逐位元）：
  * 原始到達時間、一張 GPU、FIFO（一次服務一個請求，04 §8）；請求等 GPU 時就在佇列裡。
  * CPU 是 **block 層級**的 LRU（同時間先丟前面＝m7 S4L）；SSD 容量不限、副本一直留著；用到的舊 block 服務完放回 CPU（promote）。
  * Cake 還原（每個 block 最早可開始載入的時間＝預取中的到達時間）；TTFT＝還原＋新 prefill−到達；服務結束＝到達＋TTFT＋capture（D2H）。
  * QP（E6）：服務中的請求還原完之後，替排隊中的請求把「已經在 SSD、不在 CPU」的 block 由後往前讀回 CPU，到這次服務結束為止；
    CPU 滿而且 LRU 頭是排隊中請求要的 block → 停（X4 的 qp_stop_guard）。
多出來的：每個 block 的 CPU／SSD 副本格式與品質來歷、壓縮決定（每個請求寫新 block 時一次）、逐筆 I/O 紀錄。
E1（block 模式）：CPU 淘汰＝丟；還原只用最長的有存前綴（真實系統只認最長前綴，X6）。
"""
from __future__ import annotations

import bisect
import os
import sys
from array import array
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from engine import Chan, INF  # noqa: E402
from formats import FMT_NAMES  # noqa: E402

GiB = 1 << 30
INF_T = float("inf")


def finish_reqs(reqs):
    """＝x4_sim.finish_reqs：n_old（前面出現過的前綴長度）、每個位置的下次使用時間 nu。"""
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
        q["nu"] = [nxt.get(b, INF_T) for b in q["h"]]
        for b in q["h"]:
            nxt[b] = q["t"]
    return reqs


def reqs_from_P(P):
    """trace.prepare 的 P → X4 形式的請求（h＝完整 block；prefill 到 nall）。"""
    reqs = [dict(t=P["tl"][r], h=list(P["full"][r]), n_all=P["nall"][r], ret=P["kindc"][r] == 1,
                 ret_in_unit=P["kindc"][r] == 1, par=P["parentl"][r], out=0, s=P["sessl"][r]) for r in range(len(P["full"]))]
    return finish_reqs(reqs)


class BView:
    """一種時間設定（單一）。fb／dq 同 engine.View。d2h_Bps：capture 的速度（X4＝m7_sim.D2H_BPS）。"""

    def __init__(self, costs, mf, gpu, nmax, cpu_factor=1.0, ssd_share=1, d2h_Bps=None, label=None, fbytes=None, dq=None, f=None):
        self.key = (costs.platform, gpu, cpu_factor, ssd_share, "cake")
        self.label = label or f"{costs.platform}|gpu={gpu}|cpu_x{cpu_factor}|ssd/{ssd_share}|cake|fifo1"
        self.gpu, self.mode, self.cpu_factor, self.ssd_share = gpu, "cake", cpu_factor, ssd_share
        self.has_f = True
        self.f = f if f is not None else costs.f_array(nmax, gpu)
        self.fb = list(fbytes if fbytes is not None else mf.bytes)
        dql = dq if dq is not None else mf.dq_list(gpu)
        self.dq_ok = dql is not None and all(dql[x] is not None for x in range(1, len(FMT_NAMES)) if self.fb[x] is not None)
        self.dq = [0.0 if (dql is None or x is None) else x for x in (dql or [None] * len(FMT_NAMES))]
        c, s = costs.cpu, costs.ssd
        self.cpu = Chan("cpu", c["r"] / cpu_factor if cpu_factor != 1.0 else c["r"], c["w"], c["k"])
        self.slow = Chan("ssd", s["r"] / ssd_share if ssd_share != 1 else s["r"],
                         s["w"] / ssd_share if ssd_share != 1 else s["w"], s["k"])
        self.d2h = d2h_Bps
        self.ttft = self.rest = self.nC = self.nS = self.nR = self.wq = None


class BlockEngine:
    def __init__(self, reqs, outer, comp, cap_bytes, view, mf, qp=False, promote=True, block_tokens=512, H=0.0, obs_end=None,
                 x4_prefill=False):
        self.R = reqs
        self.outer = outer
        self.E = outer["code"]
        self.ssd_on = "ssd" in outer["tiers"]
        self.drop_on_evict = outer["on_cpu_evict"] == "drop"
        assert outer["on_cpu_evict"] in ("drop", "demote"), "block 引擎只做 E1／E2／E6"
        assert not outer.get("write_through") and outer.get("ssd_write_min_uses") is None and outer.get("idle_T") is None, \
            "block 引擎只做 E1／E2／E6（E3–E5 用 session 引擎）"
        self.comp = comp
        self.Cb = cap_bytes
        self.v = view
        self.mf = mf
        self.fb = view.fb
        self.qp, self.promote = qp, promote
        self.btok = block_tokens
        self.H = H
        self.obs_end = obs_end
        self.x4_prefill = x4_prefill        # True：prefill 只算到 len(h)（X4 的 h 含最後不滿的 block）
        self.cpu = OrderedDict()
        self.cpu_used = 0
        self.cfmt, self.sfmt, self.cq, self.sq = {}, {}, {}, {}
        self.owner = {}                  # block → 寫它的請求（ssd_fid 用的 ctx）
        self.ctxs = {}
        self.avail = {}
        self.prefetched = set()
        self.memb = {}
        self.cnt_pf = {}
        self.qtail = 0
        self.S = dict(demote=0, qp_reads=0, pf_used=0, pf_evicted=0, qp_stop_guard=0, promoted=0, wait_inflight=0, dropped=0)
        self.wbytes = {"store": 0, "promote": 0, "demote": 0, "prefetch_read": 0}
        self.wblocks_fmt = {}
        self.dec_fmt = {}
        acc = {}
        for j, q in enumerate(reqs):
            for b in q["h"]:
                acc.setdefault(b, []).append(j)
        self.acc = acc
        self.t_last = reqs[-1]["t"] if reqs else 0.0
        n = len(reqs)
        self.Ltok = np.zeros(n, np.int64)

    # ---- 小工具
    def _wfmt(self, tier, fid):
        k = (tier, fid)
        self.wblocks_fmt[k] = self.wblocks_fmt.get(k, 0) + 1

    def _cpu_room(self, nb):
        return self.cpu_used + nb <= self.Cb

    def _pf_dec(self, b):
        for k in self.memb.get(b, ()):
            self.cnt_pf[k] -= 1

    def _stored_prefix(self, q):
        """有存的最長前綴（E2／E6：SSD 副本一直在 → ＝n_old；E1：可能有洞）。"""
        n = 0
        for b in q["h"][:q["n_old"]]:
            if b in self.cpu or self.sfmt.get(b, 0):
                n += 1
            else:
                break
        return n

    def _enqueue(self, k):
        q = self.R[k]
        c = 0
        for b in q["h"][:q["n_old"]]:
            self.memb.setdefault(b, set()).add(k)
            c += (b not in self.cpu) and bool(self.sfmt.get(b, 0))
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
        fm = self.cfmt.pop(b)
        self.cpu_used -= self.fb[fm]
        self.avail.pop(b, None)
        for k in self.memb.get(b, ()):
            self.cnt_pf[k] += 1
        if b in self.prefetched:
            self.prefetched.discard(b)
            self.S["pf_evicted"] += 1
        if not self.sfmt.get(b, 0):
            if self.drop_on_evict or not self.ssd_on:
                self.S["dropped"] += 1
                for k in self.memb.get(b, ()):
                    self.cnt_pf[k] -= 1      # 沒有副本可以預取
            else:
                ow = self.owner.get(b)
                sf = self.comp.ssd_fid(self.ctxs[ow], fm) if ow is not None else fm
                self.sfmt[b] = sf
                q0 = self.cq.get(b, fm)
                self.sq[b] = sf if sf > q0 else q0
                nb = self.fb[sf]
                self.v.slow.write(nb, now)
                self.wbytes["demote"] += nb
                self._wfmt("ssd", sf)
                self.S["demote"] += 1
        return b

    # ---- Cake（＝x4_sim.cake，多了格式、解壓、品質）
    def restore(self, q, n, t0):
        v = self.v
        f, fb, dq = v.f, self.fb, v.dq
        h = q["h"]
        if n == 0:
            return t0, 0, 0, 0, 0
        nc = ns = 0
        wq = 0
        p, qq, rt, lt = 0, n - 1, t0, t0
        while p <= qq:
            if rt <= lt:
                rt += f[p]
                p += 1
            else:
                b = h[qq]
                if b in self.cpu:
                    fid = self.cfmt[b]
                    av = self.avail.get(b, 0.0)
                    lt = v.cpu.read(fb[fid], lt if av <= lt else av)
                    d = dq[fid]
                    if d:
                        lt = lt + d
                    nc += 1
                    qb = self.cq[b]
                else:
                    fid = self.sfmt[b]
                    lt = v.slow.read(fb[fid], lt)
                    d = dq[fid]
                    if d:
                        lt = lt + d
                    ns += 1
                    qb = self.sq[b]
                if qb > wq:
                    wq = qb
                qq -= 1
        return (rt if rt > lt else lt), p, nc, ns, wq

    # ---- 請求結束
    def _decide(self, j, q, L):
        h, n_old = q["h"], q["n_old"]
        tok = len(h) * self.btok
        # 這一輪要寫進 CPU 的第一個 block（新的、promote 回來的、或洞之後重算的）；oracle 看它之後有沒有被讀
        # （2026-10-18 測試 (iii)-fifo1 抓到：原本只看「沒見過的」block，沒有新 block 但有 promote 時被當成不會再讀 → int2）
        first_new = None
        for i, b in enumerate(h):
            if not (i < n_old and b in self.cpu) and (i >= n_old or i >= L or self.promote):
                first_new = b
                break
        ctx = dict(r=j, s=q.get("s", j), t=q["t"], n_blocks=len(h), tokens=tok, n_new=len(h) - n_old, nreq=None,
                   kind=1 if q["ret"] else 0, feats=None, mf=self.mf)
        if self.comp.needs_oracle:
            nx = INF
            if first_new is not None:
                lst = self.acc.get(first_new, [])
                i = bisect.bisect_right(lst, j)
                nx = lst[i] if i < len(lst) else INF
            oe = self.obs_end[j] if self.obs_end is not None else self.t_last
            ctx["next_read"] = None if nx >= INF else nx
            ctx["censored"] = (oe - q["t"]) < self.H if self.H > 0 else False
            mx = 0
            if nx < INF:
                lst = self.acc[first_new]
                for jj in lst[bisect.bisect_right(lst, j):]:
                    x = len(self.R[jj]["h"])
                    mx = x if x > mx else mx
            ctx["max_read_tokens"] = mx * self.btok
        fid = self.comp.cpu_fid(ctx)
        self.ctxs[j] = ctx
        if first_new is not None:
            k = (FMT_NAMES[fid], self.mf.bin_of(tok), (ctx.get("next_read") is not None) if self.comp.needs_oracle else None,
                 ctx.get("censored"))
            self.dec_fmt[k] = self.dec_fmt.get(k, 0) + 1
        return fid

    def after_round(self, j, q, now, L):
        h, n_old = q["h"], q["n_old"]
        cpu = self.cpu
        fid = self._decide(j, q, L)
        nbf = self.fb[fid]
        for i in range(len(h)):
            b = h[i]
            if b in self.prefetched:
                self.prefetched.discard(b)
                self.S["pf_used"] += 1
            if i < n_old and b in cpu:
                cpu.move_to_end(b)
                self.avail.pop(b, None)
            elif i < n_old and i < L and self.promote:
                self._pf_dec(b)
                cpu[b] = None
                self.cfmt[b] = fid
                sq = self.sq.get(b, 1)
                self.cq[b] = fid if fid > sq else sq
                self.owner[b] = j
                self.cpu_used += nbf
                self.v.cpu.write(nbf, now)
                self.wbytes["promote"] += nbf
                self._wfmt("cpu", fid)
                self.S["promoted"] += 1
            elif i >= n_old or i >= L:
                if i < n_old:
                    self._pf_dec(b)
                cpu[b] = None
                self.cfmt[b] = fid
                self.cq[b] = fid
                self.owner[b] = j
                self.cpu_used += nbf
                self.v.cpu.write(nbf, now)
                self.wbytes["store"] += nbf
                self._wfmt("cpu", fid)
        while self.cpu_used > self.Cb:
            self._evict_one(now)

    # ---- QP（＝x4_sim.Sim.queue_prefetch，增量版）
    def queue_prefetch(self, j, t_go, t_end):
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
            for i in range(q["n_old"] - 1, -1, -1):
                b = q["h"][i]
                sf = self.sfmt.get(b, 0)
                if b in self.cpu or not sf:
                    continue
                st = max(self.v.slow.t_free, q["t"], t_go)
                if st >= t_end:
                    return
                nb = self.fb[sf]
                while not self._cpu_room(nb):
                    v = next(iter(self.cpu))
                    if self.memb.get(v):
                        self.S["qp_stop_guard"] += 1
                        return
                    self._evict_one(st)
                done = self.v.slow.read(nb, st, 1)
                self.wbytes["prefetch_read"] += nb
                self._pf_dec(b)
                self.cpu[b] = None
                self.cfmt[b] = sf
                self.cq[b] = self.sq[b]
                self.cpu_used += nb
                self.avail[b] = done
                self.prefetched.add(b)
                self.S["qp_reads"] += 1

    def serve(self, j, t_arr, t_go):
        q = self.R[j]
        n_old = q["n_old"]
        L = self._stored_prefix(q) if (self.drop_on_evict or not self.ssd_on) else n_old
        n_end = len(q["h"]) if self.x4_prefill else max(q.get("n_all", len(q["h"])), len(q["h"]))
        miss_cpu = sum(1 for b in q["h"][:L] if b not in self.cpu)
        inflight = sum(1 for b in q["h"][:L] if self.avail.get(b, 0.0) > t_go)
        self.S["wait_inflight"] += inflight
        t_rend, meet, nc, ns, wq = self.restore(q, L, t_go)
        t_new = sum(self.v.f[L:n_end])
        ttft = t_rend + t_new - t_arr
        n_new = len(q["h"]) - n_old
        n_cap = n_new + (miss_cpu if self.promote else 0) + (n_old - L)
        T = t_arr + ttft + n_cap * self.fb[1] / self.v.d2h
        self.Ltok[j] = L * self.btok
        return dict(T=T, ttft=ttft, qwait=t_go - t_arr, restore=t_rend - t_go, L=L, nc=nc, ns=ns, meet=meet, wq=wq,
                    nR=meet + (n_end - L))

    def run_events(self):
        R = self.R
        n = len(R)
        v = self.v
        v.ttft, v.rest, v.qwait = np.zeros(n), np.zeros(n), np.zeros(n)
        v.nC, v.nS, v.nR, v.wq = np.zeros(n, np.int32), np.zeros(n, np.int32), np.zeros(n, np.int32), np.zeros(n, np.int8)
        gpu_free = R[0]["t"] if R else 0.0
        busy = 0.0
        for j in range(n):
            ta = R[j]["t"]
            t_go = ta if ta >= gpu_free else gpu_free
            r = self.serve(j, ta, t_go)
            if self.qp:
                self.queue_prefetch(j, t_go, r["T"])
            self.after_round(j, R[j], r["T"], r["L"])
            if self.qp:
                self._dequeue(j)
            busy += r["T"] - t_go
            gpu_free = r["T"]
            v.ttft[j], v.rest[j], v.qwait[j] = r["ttft"], r["restore"], r["qwait"]
            v.nC[j], v.nS[j], v.nR[j], v.wq[j] = r["nc"], r["ns"], r["nR"], r["wq"]
        self.busy = busy
        return self
