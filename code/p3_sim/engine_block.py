"""engine_block.py — block 引擎（serve=fifo1）：X4 的模型（x4_sim.Sim 的 run_events）＋格式＋記帳。給 E6（佇列預取）與它的 E2 基準。

和 X4 一樣（照抄算式，才能對 X4 的 QP 逐位元）：
  * 原始到達時間、一張 GPU、FIFO（一次服務一個請求，04 §8）；請求等 GPU 時就在佇列裡。
  * CPU 是 **block 層級**的 LRU（同時間先丟前面＝m7 S4L）；SSD 容量不限、副本一直留著；用到的舊 block 服務完放回 CPU（promote）。
  * Cake 還原（每個 block 最早可開始載入的時間＝預取中的到達時間）；TTFT＝還原＋新 prefill−到達；服務結束＝到達＋TTFT＋capture（D2H）。
  * QP（E6）：服務中的請求還原完之後，替排隊中的請求把「已經在 SSD、不在 CPU」的 block 由後往前讀回 CPU，到這次服務結束為止；
    CPU 滿而且 LRU 頭是排隊中請求要的 block → 停（X4 的 qp_stop_guard）。
多出來的：每個 block 的 CPU／SSD 副本格式與品質來歷、壓縮決定（每個請求寫新 block 時一次）、逐筆 I/O 紀錄。
E1（block 模式）：CPU 淘汰＝丟；還原只用最長的有存前綴（真實系統只認最長前綴，X6）。
2026-10-18（06 §11–§13）：ssd_moment 的策略在淘汰（降級寫 SSD）那一刻拿到 oracle／當下資訊；O 的「之後最長讀取」改用後綴最大值
（和 session 引擎同一個做法，結果不變、不再每次掃）；int2_on_read＝recompute；逐請求／逐 session 的新帳。舊的算式一行沒動。
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

from engine import Chan, INF, ok_token_limits, pass_masks, request_lengths  # noqa: E402
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
    rl = request_lengths(P)
    reqs = [dict(t=P["tl"][r], h=list(P["full"][r]), n_all=P["nall"][r], ret=P["kindc"][r] == 1,
                 ret_in_unit=P["kindc"][r] == 1, par=P["parentl"][r], out=0, s=P["sessl"][r], len_tok=rl[r])
            for r in range(len(P["full"]))]
    return finish_reqs(reqs)


class BView:
    """一種時間設定（單一）。fb／dq 同 engine.View。d2h_Bps：capture 的速度（X4＝m7_sim.D2H_BPS）。"""

    def __init__(self, costs, mf, gpu, nmax, cpu_factor=1.0, ssd_share=1, d2h_Bps=None, label=None, fbytes=None, dq=None, f=None):
        self.key = (costs.platform, gpu, cpu_factor, ssd_share, "cake")
        self.label = label or f"{costs.platform}|gpu={gpu}|cpu_x{cpu_factor}|ssd/{ssd_share}|cake|fifo1"
        self.gpu, self.mode, self.cpu_factor, self.ssd_share = gpu, "cake", cpu_factor, ssd_share
        self.has_f = True
        if costs.ssd_k_nm:
            raise NotImplementedError("block 引擎的放置依賴時間（佇列預取），沒有「k_read 沒量」的追蹤；/ssd8 視角只在 session 引擎（06 §24）")
        self.f = f if f is not None else costs.f_array(nmax, gpu)
        self.fb = list(fbytes if fbytes is not None else mf.bytes)
        dql = dq if dq is not None else mf.dq_list(gpu)
        self.dq_ok = dql is not None and all(dql[x] is not None for x in range(1, len(FMT_NAMES)) if self.fb[x] is not None)
        self.dq = [0.0 if (dql is None or x is None) else x for x in (dql or [None] * len(FMT_NAMES))]
        c, s = costs.cpu, costs.ssd
        self.cpu = Chan("cpu", c["r"] / cpu_factor if cpu_factor != 1.0 else c["r"], c["w"], c["k"], ext=False)
        self.slow = Chan("ssd", s["r"] / ssd_share if ssd_share != 1 else s["r"],
                         s["w"] / ssd_share if ssd_share != 1 else s["w"], s["k"])
        self.d2h = d2h_Bps
        self.ttft = self.rest = self.nC = self.nS = self.nR = self.wq = None
        nm = [x >= 1 and self.fb[x] is not None and (dql is None or dql[x] is None) for x in range(len(FMT_NAMES))]
        self.nmf = nm if any(nm) else None      # 逐請求的 NOT_MEASURED（06 §12.4）


class BlockEngine:
    def __init__(self, reqs, outer, comp, cap_bytes, view, mf, qp=False, promote=True, block_tokens=512, H=0.0, obs_end=None,
                 x4_prefill=False, int2_on_read="serve", unverified_len="prefix", H_extra=0.0, tier_rec=None, ev_log=False):
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
        # ---- 06 §11–§13 的新東西（不參與舊的算式）
        self.ssd_moment = bool(getattr(comp, "ssd_moment", False))
        assert int2_on_read in ("serve", "recompute", "recompute_unverified"), int2_on_read
        self.int2_on_read = int2_on_read
        assert unverified_len in ("prefix", "request"), unverified_len
        self.unverified_len = unverified_len
        self.req_strict = unverified_len == "request"
        cnt_, self.ord1 = {}, []          # ord1[j]＝請求 j 是它的 session 的第幾個請求（1 起算；R2 用）
        for j_, q_ in enumerate(reqs):
            k_ = q_.get("s", j_)
            cnt_[k_] = cnt_.get(k_, 0) + 1
            self.ord1.append(cnt_[k_])
        self.reqlen = [q["len_tok"] if "len_tok" in q else max(q.get("n_all", len(q["h"])), len(q["h"])) * block_tokens for q in reqs]
        self.Lqual = np.asarray(self.reqlen, np.int64) if self.req_strict else self.Ltok   # 品質帳用的讀取長度（06 §18）
        self.okmask = pass_masks(mf)
        self.rbin = [mf.bin_of(x) for x in self.reqlen] if self.req_strict else None
        self._sufreq = {}
        self.lim = ok_token_limits(mf)
        self.unv_on = int2_on_read == "recompute_unverified"
        if self.unv_on:        # 04 v4 §3.3：前 L 個的來歷在 L×512 都要 pass；任何長度都不 pass 的來歷（INT2）的副本被重算取代
            self.nro = frozenset(f for f in range(1, len(FMT_NAMES)) if self.lim[f] < 0)
        else:
            self.nro = frozenset(mf.never_read_only) if int2_on_read == "recompute" else None
        self.surp_blk = np.zeros(n, np.int32)     # 因為 INT2 改成重算的 block 數
        self.nprev = np.zeros(n, np.int32)        # ＝n_old（以前算過的前綴）
        self.nend = np.zeros(n, np.int32)         # prefill 到哪
        self.lo_bytes = np.zeros(n, np.int64)     # 服務開始時，有存前綴裡在 SSD（不在 CPU）的 block 的位元組（load-only 會讀的）
        self.s_ssd_w = {}                         # session → SSD 寫入位元組（降級算在 block 的 owner 請求的 session）
        self.cutH = (max(obs_end) - H) if (H > 0 and obs_end) else None
        self.s_ssd_w_H = {}                       # 截尾版：淘汰時刻 < 觀察窗終點 − H
        self.H_extra = float(H_extra)             # 第二個只記帳、不影響放置的截尾（例如 2H；06 §18）
        self.cutH2 = (max(obs_end) - self.H_extra) if (self.H_extra > 0 and obs_end) else None
        self.s_ssd_w_H2 = {}
        self.s_last = {}                          # session → 最近一個服務完的請求 index
        self.cur_j = -1
        self._sufmax = {}
        self._rep = (0, 0)                        # 這一輪的 INT2 取代區間 [L, L0)
        # ---- 重算計價（review #1 選項 a；同 engine.Engine）：被意外重算取代、而且只在 SSD（不在 CPU）的 block，照這個請求的輸入長度下過關的最省格式計價
        self.rc_blocks = np.zeros(n, np.int32)
        self.rc_bytes = np.zeros(n, np.int64)
        self._pass_b = [min(self.fb[f] for f in range(1, len(FMT_NAMES)) if m[f] and self.fb[f] is not None) for m in self.okmask]
        # ---- O-tier 的兩趟（同 engine.Engine；這個引擎和時間有關，第 2 趟的 CPU 側事件只能近似相同 → 數不一樣的請求數）
        self.rec = tier_rec if (tier_rec is not None and tier_rec.recording) else None
        self.ev_log = bool(ev_log)
        self._track = self.rec is not None or self.ev_log
        self._nw = {}
        self._evw, self._pfw = [], []
        self._miss = 0
        self._last_fid = 0
        self.ev_digest = np.zeros(n, np.uint64) if ev_log else None
        # 預取跳過「在觸發的請求的輸入長度下品質不過關」的副本（嚴格讀取規則；06 §26）：那些 block 服務時會被重算，讀了是白讀
        self.pf_skip = self.unv_on and self.req_strict
        self.S["pf_skipped"] = 0

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
        if self._track:
            self._evw.append(b)
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
                key = None
                if self._track or getattr(self.comp, "tier", False):
                    key = (b, self._nw.get(b, 0))
                    self._nw[b] = key[1] + 1
                if not self.ssd_moment:
                    sf = self.comp.ssd_fid(self.ctxs[ow], fm) if ow is not None else fm
                else:
                    if ow is not None:
                        mc = self._mctx(b, ow, now)
                        mc["ssd_key"] = key
                        sf = self.comp.ssd_fid(mc, fm)
                    else:
                        sf = fm
                    if sf < fm:
                        raise AssertionError(f"{self.comp.name}: ssd_fid 回傳 {sf} < 目前的 {fm}（格式只能往下壓）")
                if self.rec is not None:
                    self.rec.on_write(key)
                self.sfmt[b] = sf
                q0 = self.cq.get(b, fm)
                self.sq[b] = sf if sf > q0 else q0
                nb = self.fb[sf]
                self.v.slow.write(nb, now, 0, ow if ow is not None else -1)
                so = self.R[ow].get("s", ow) if ow is not None else -1
                self.s_ssd_w[so] = self.s_ssd_w.get(so, 0) + nb
                if self.cutH is not None and now < self.cutH:
                    self.s_ssd_w_H[so] = self.s_ssd_w_H.get(so, 0) + nb
                if self.cutH2 is not None and now < self.cutH2:
                    self.s_ssd_w_H2[so] = self.s_ssd_w_H2.get(so, 0) + nb
                self.wbytes["demote"] += nb
                self._wfmt("ssd", sf)
                self.S["demote"] += 1
        return b

    def _max_future_len(self, b, j):
        """block b 在請求 j 之後被讀到的請求裡最長的 len(h)（後綴最大值，06 §13；＝原本每次掃 lst[j 之後] 的結果）。"""
        lst = self.acc.get(b)
        if not lst:
            return 0
        sm = self._sufmax.get(b)
        if sm is None:
            R = self.R
            sm = [0] * len(lst)
            m = 0
            for i in range(len(lst) - 1, -1, -1):
                x = len(R[lst[i]]["h"])
                m = x if x > m else m
                sm[i] = m
            self._sufmax[b] = sm
        i = bisect.bisect_right(lst, j)
        return sm[i] if i < len(lst) else 0

    def _max_future_reqlen(self, b, j):
        """block b 在請求 j 之後被讀到的請求裡最長的「請求輸入長度」（token；嚴格版讀取長度；後綴最大值）。"""
        lst = self.acc.get(b)
        if not lst:
            return 0
        sm = self._sufreq.get(b)
        if sm is None:
            rl = self.reqlen
            sm = [0] * len(lst)
            m = 0
            for i in range(len(lst) - 1, -1, -1):
                x = rl[lst[i]]
                m = x if x > m else m
                sm[i] = m
            self._sufreq[b] = sm
        i = bisect.bisect_right(lst, j)
        return sm[i] if i < len(lst) else 0

    def _mctx(self, b, ow, now):
        """降級（寫 SSD）那一刻的 ctx（06 §11.2）。之後＝請求 index > cur_j（FIFO，照到達順序服務）。
        tokens／idle_s 用 owner 請求的 session 最近一個服務完的請求。"""
        s = self.R[ow].get("s", ow)
        ls = self.s_last.get(s, ow)
        after = self.cur_j
        ctx = dict(s=s, r=after, t=now, b=b, tokens=len(self.R[ls]["h"]) * self.btok, idle_s=now - self.R[ls]["t"],
                   cpu_ctx=self.ctxs.get(ow), mf=self.mf)
        if self.comp.needs_oracle:
            lst = self.acc.get(b, [])
            i = bisect.bisect_right(lst, after)
            nx = lst[i] if i < len(lst) else INF
            oe = self.obs_end[ls] if self.obs_end is not None else self.t_last
            ctx["next_read"] = None if nx >= INF else nx
            ctx["censored"] = (oe - now) < self.H if self.H > 0 else False
            ctx["max_read_tokens"] = self._max_future_len(b, after) * self.btok if nx < INF else 0
            ctx["max_read_req_tokens"] = self._max_future_reqlen(b, after) if nx < INF else 0
        ctx["req_tokens"] = self.reqlen[ls]
        ctx["nreq"] = self.ord1[ls]        # 這個 session 到 ls（最近一個服務完的請求）為止的請求數
        return ctx

    # ---- Cake（＝x4_sim.cake，多了格式、解壓、品質）
    def restore(self, q, n, t0):
        v = self.v
        f, fb, dq = v.f, self.fb, v.dq
        h = q["h"]
        nmf = v.nmf
        self._r_bS, self._r_nm = 0, False
        if n == 0:
            return t0, 0, 0, 0, 0
        bS = 0
        bad = False
        nc = ns = 0
        wq = 0
        # 06 §13 加速：Chan.read 內聯（同樣的浮點運算與順序；還原時沒有寫入，t_free_w 不變），結束時寫回
        cpu, slw = v.cpu, v.slow
        c_tf, c_tfw, c_r, c_k, c_busy, c_bytes = cpu.t_free, cpu.t_free_w, cpu.r, cpu.k, cpu.busy_r, cpu.bytes_r
        s_tf, s_tfw, s_r, s_k, s_busy, s_bytes = slw.t_free, slw.t_free_w, slw.r, slw.k, slw.busy_r, slw.bytes_r
        c_rec, s_rec, c_ext, s_ext = cpu.rec, slw.rec, cpu.ext, slw.ext
        crs, cre, crb, crk, crr, crd = cpu.rs.append, cpu.re.append, cpu.rb.append, cpu.rk.append, cpu.rr.append, cpu.rd.append
        srs, sre, srb, srk, srr, srd = slw.rs.append, slw.re.append, slw.rb.append, slw.rk.append, slw.rr.append, slw.rd.append
        c_cur, s_cur = cpu.cur_r, slw.cur_r
        incpu, cfmt, sfmt, cq, sq, avail = self.cpu, self.cfmt, self.sfmt, self.cq, self.sq, self.avail
        p, qq, rt, lt = 0, n - 1, t0, t0
        while p <= qq:
            if rt <= lt:
                rt += f[p]
                p += 1
            else:
                b = h[qq]
                if b in incpu:
                    fid = cfmt[b]
                    av = avail.get(b, 0.0)
                    now = lt if av <= lt else av
                    nb = fb[fid]
                    st = now if now > c_tf else c_tf
                    dur = nb / c_r
                    if c_tfw > st:
                        dur *= c_k
                    c_tf = st + dur
                    c_busy += dur
                    c_bytes += nb
                    if c_rec:
                        crs(st)
                        cre(c_tf)
                        crb(nb)
                        crk(0)
                        if c_ext:
                            crr(c_cur)
                            crd(st - now)
                    lt = c_tf
                    d = dq[fid]
                    if d:
                        lt = lt + d
                    nc += 1
                    qb = cq[b]
                else:
                    fid = sfmt[b]
                    nb = fb[fid]
                    st = lt if lt > s_tf else s_tf
                    dur = nb / s_r
                    if s_tfw > st:
                        dur *= s_k
                    s_tf = st + dur
                    s_busy += dur
                    s_bytes += nb
                    if s_rec:
                        srs(st)
                        sre(s_tf)
                        srb(nb)
                        srk(0)
                        if s_ext:
                            srr(s_cur)
                            srd(st - lt)
                    lt = s_tf
                    d = dq[fid]
                    if d:
                        lt = lt + d
                    ns += 1
                    bS += nb
                    qb = sq[b]
                if nmf and nmf[fid]:
                    bad = True
                if qb > wq:
                    wq = qb
                qq -= 1
        cpu.t_free, cpu.busy_r, cpu.bytes_r = c_tf, c_busy, c_bytes
        slw.t_free, slw.busy_r, slw.bytes_r = s_tf, s_busy, s_bytes
        self._r_bS, self._r_nm = bS, bad
        return (rt if rt > lt else lt), p, nc, ns, wq

    # ---- 請求結束
    def _decide(self, j, q, L):
        h, n_old = q["h"], q["n_old"]
        tok = len(h) * self.btok
        # 這一輪要寫進 CPU 的第一個 block（新的、promote 回來的、或洞之後重算的）；oracle 看它之後有沒有被讀
        # （2026-10-18 測試 (iii)-fifo1 抓到：原本只看「沒見過的」block，沒有新 block 但有 promote 時被當成不會再讀 → int2）
        first_new = None
        rlo, rhi = self._rep
        for i, b in enumerate(h):
            if not (i < n_old and b in self.cpu) and (i >= n_old or i >= L or self.promote):
                first_new = b
                break
            if rlo <= i < rhi:    # 有存但沒載入、要被重算的新 KV 取代的（04 §3.3「重算之後」，不論來歷）
                first_new = b
                break
        ctx = dict(r=j, s=q.get("s", j), t=q["t"], n_blocks=len(h), tokens=tok, n_new=len(h) - n_old, nreq=None,
                   kind=1 if q["ret"] else 0, feats=None, mf=self.mf)
        ctx["nreq"] = self.ord1[j]
        if self.comp.needs_oracle:
            nx = INF
            if first_new is not None:
                lst = self.acc.get(first_new, [])
                i = bisect.bisect_right(lst, j)
                nx = lst[i] if i < len(lst) else INF
            oe = self.obs_end[j] if self.obs_end is not None else self.t_last
            ctx["next_read"] = None if nx >= INF else nx
            ctx["censored"] = (oe - q["t"]) < self.H if self.H > 0 else False
            ctx["max_read_tokens"] = (self._max_future_len(first_new, j) if nx < INF else 0) * self.btok
            ctx["max_read_req_tokens"] = self._max_future_reqlen(first_new, j) if (nx < INF and first_new is not None) else 0
        ctx["req_tokens"] = self.reqlen[j]
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
        self._last_fid = fid
        nbf = self.fb[fid]
        rlo, rhi = self._rep
        for i in range(len(h)):
            b = h[i]
            if b in self.prefetched:
                self.prefetched.discard(b)
                self.S["pf_used"] += 1
            if rlo <= i < rhi and self.sfmt.get(b, 0):
                self.sfmt[b] = 0           # 重算的新 KV 取代：SSD 上的舊副本作廢（之後新副本若被降級寫 SSD 是一般的寫入）
                self.sq.pop(b, None)
                if self.rec is not None:
                    self.rec.on_super(b)
            if i < n_old and b in cpu and rlo <= i < rhi:
                # 重算出來的新 KV 取代 CPU 裡舊來歷的副本（照這一輪的決定寫）
                cpu.move_to_end(b)
                self.avail.pop(b, None)
                self.cpu_used += nbf - self.fb[self.cfmt[b]]
                self.cfmt[b] = fid
                self.cq[b] = fid
                self.owner[b] = j
                self.v.cpu.write(nbf, now)
                self.wbytes["store"] += nbf
                self._wfmt("cpu", fid)
            elif i < n_old and b in cpu:
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
            top = q["n_old"]
            if self.pf_skip:
                # 嚴格讀取規則：k 服務時，前綴裡第一個「來歷在 k 的輸入長度下不過關」的 block 起都會被重算 → 只預取它之前的（之後的白讀）
                okm_ = self.okmask[self.rbin[k]]
                cpu_, cq_, sq_, hk_ = self.cpu, self.cq, self.sq, q["h"]
                for i in range(top):
                    b = hk_[i]
                    if b in cpu_:
                        qb = cq_[b]
                    elif self.sfmt.get(b, 0):
                        qb = sq_[b]
                    else:
                        break
                    if not okm_[qb]:
                        self.S["pf_skipped"] += 1        # 次數：（排隊中的請求 k，這次預取）跳過了它的一段尾巴（同一個 k 會被多次檢查）
                        top = i
                        break
            for i in range(top - 1, -1, -1):
                b = q["h"][i]
                sf = self.sfmt.get(b, 0)
                if b in self.cpu or not sf:
                    continue
                st = max(self.v.slow.t_free, q["t"], t_go)
                if st >= t_end:
                    return
                t_rdy = q["t"] if q["t"] > t_go else t_go      # 準備好可以讀的時刻（read 裡 max(t_rdy, t_free)＝st，逐位元相同）
                nb = self.fb[sf]
                while not self._cpu_room(nb):
                    v = next(iter(self.cpu))
                    if self.memb.get(v):
                        self.S["qp_stop_guard"] += 1
                        return
                    self._evict_one(st)
                done = self.v.slow.read(nb, t_rdy, 1, k)
                if self.rec is not None:
                    self.rec.on_read(b, 3, self.reqlen[k], len(R[k]["h"]) * self.btok)
                if self._track:
                    self._pfw.append(b)
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
        self._rep = (0, 0)
        if self.unv_on:              # recompute_unverified：載入到第 i+1 個時，前 i+1 個的來歷在 (i+1)×512 都要 pass
            h = q["h"]
            cpu, cq, sq, lim, bt = self.cpu, self.cq, self.sq, self.lim, self.btok
            m_ = INF
            okm = self.okmask[self.rbin[j]] if self.req_strict else None
            for i in range(L):
                b = h[i]
                if okm is not None:          # unverified_len=request：請求的輸入長度下來歷不 pass → 從這裡起重算
                    bad_ = not okm[cq[b] if b in cpu else sq[b]]
                else:
                    x = lim[cq[b] if b in cpu else sq[b]]
                    if x < m_:
                        m_ = x
                    bad_ = (i + 1) * bt > m_
                if bad_:
                    self.surp_blk[j] = L - i
                    self._rep = (i, L)
                    L = i
                    break
        elif self.nro:               # int2_on_read=recompute：從第一個 INT2 來歷的 block 起都重算
            h = q["h"]
            cpu, cq, sq, nro = self.cpu, self.cq, self.sq, self.nro
            for i in range(L):
                b = h[i]
                if (cq[b] if b in cpu else sq[b]) in nro:
                    self.surp_blk[j] = L - i
                    self._rep = (i, L)
                    L = i
                    break
        rlo_, rhi_ = self._rep
        if rhi_ > rlo_:              # 重算計價：有存、沒載入、存放副本只在 SSD（CPU 沒有）的 block
            hh_, cpu_ = q["h"], self.cpu
            nrc = sum(1 for i_ in range(rlo_, rhi_) if hh_[i_] not in cpu_ and self.sfmt.get(hh_[i_], 0))
            if nrc:
                self.rc_blocks[j] = nrc
                self.rc_bytes[j] = nrc * self._pass_b[self.mf.bin_of(self.reqlen[j])]
        n_end = len(q["h"]) if self.x4_prefill else max(q.get("n_all", len(q["h"])), len(q["h"]))
        self.nprev[j] = n_old
        self.nend[j] = n_end
        # 一次掃完（06 §13：原本三個 generator；avail 只會有在 CPU 裡的 block，所以結果相同）
        cpu, sfmt, fb, avail = self.cpu, self.sfmt, self.fb, self.avail
        miss_cpu = inflight = lob = 0
        for b in q["h"][:L]:
            if b not in cpu:
                miss_cpu += 1
                lob += fb[sfmt[b]]
            elif avail and avail.get(b, 0.0) > t_go:
                inflight += 1
        self.lo_bytes[j] = lob
        self._miss = miss_cpu
        if self.rec is not None:     # O-tier 第 1 趟：被載入的 block 讀的是 CPU 還是 SSD 副本
            rl_, st_ = self.reqlen[j], len(q["h"]) * self.btok
            for b in q["h"][:L]:
                self.rec.on_read(b, 1 if b in cpu else 3, rl_, st_)
        self.S["wait_inflight"] += inflight
        self.v.cpu.cur_r = self.v.slow.cur_r = j
        t_rend, meet, nc, ns, wq = self.restore(q, L, t_go)
        t_new = sum(self.v.f[L:n_end])
        ttft = t_rend + t_new - t_arr
        n_new = len(q["h"]) - n_old
        n_cap = n_new + (miss_cpu if self.promote else 0) + (n_old - L)
        T = t_arr + ttft + n_cap * self.fb[1] / self.v.d2h
        self.Ltok[j] = L * self.btok
        return dict(T=T, ttft=ttft, qwait=t_go - t_arr, restore=t_rend - t_go, L=L, nc=nc, ns=ns, meet=meet, wq=wq,
                    nR=meet + (n_end - L), bS=self._r_bS, nm=self._r_nm)

    def run_events(self):
        R = self.R
        n = len(R)
        v = self.v
        v.ttft, v.rest, v.qwait = np.zeros(n), np.zeros(n), np.zeros(n)
        v.nC, v.nS, v.nR, v.wq = np.zeros(n, np.int32), np.zeros(n, np.int32), np.zeros(n, np.int32), np.zeros(n, np.int8)
        v.bS, v.nmreq, v.tgo, v.tend = np.zeros(n, np.int64), np.zeros(n, bool), np.zeros(n), np.zeros(n)
        v.cpu_r0 = np.zeros(n + 1, np.int64)      # 請求 j 的 CPU 讀取紀錄＝第 cpu_r0[j] 到 cpu_r0[j+1]−1 筆（只在還原裡讀 CPU）
        gpu_free = R[0]["t"] if R else 0.0
        busy = 0.0
        for j in range(n):
            ta = R[j]["t"]
            t_go = ta if ta >= gpu_free else gpu_free
            self.cur_j = j
            v.cpu_r0[j] = len(v.cpu.rs)
            r = self.serve(j, ta, t_go)
            if self.qp:
                self.queue_prefetch(j, t_go, r["T"])
            self.v.cpu.cur_r = self.v.slow.cur_r = j
            self.after_round(j, R[j], r["T"], r["L"])
            if self.ev_log:
                import hashlib
                hd = hashlib.blake2b(np.array([j, r["L"], self._miss, self._last_fid, len(self._evw), len(self._pfw)], np.int64).tobytes()
                                     + np.array(self._evw, np.int64).tobytes() + np.array(self._pfw, np.int64).tobytes(), digest_size=8).digest()
                self.ev_digest[j] = int.from_bytes(hd, "little")
            if self._track:
                self._evw, self._pfw = [], []
            self.s_last[R[j].get("s", j)] = j
            if self.qp:
                self._dequeue(j)
            busy += r["T"] - t_go
            gpu_free = r["T"]
            v.ttft[j], v.rest[j], v.qwait[j] = r["ttft"], r["restore"], r["qwait"]
            v.nC[j], v.nS[j], v.nR[j], v.wq[j] = r["nc"], r["ns"], r["nR"], r["wq"]
            v.bS[j], v.nmreq[j], v.tgo[j], v.tend[j] = r["bS"], r["nm"], t_go, r["T"]
        v.cpu_r0[n] = len(v.cpu.rs)
        self.busy = busy
        return self
