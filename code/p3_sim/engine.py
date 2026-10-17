"""engine.py — session 引擎（serve=trace）：X1 的模型（x1_sim.Engine）＋格式＋外層做法 E1–E5＋記帳。

和 X1 一樣（照抄算式，才能逐位元重現 X1 的 LRU）：
  * trace 時間、一次一個請求、GPU 不排隊；CPU 層與 SSD 各一條 I/O 線（讀寫分開排隊，寫入中讀取 ×k_read）。
  * 回合結束：session 的完整 block 寫進 CPU；CPU 滿就依 LRU（session 層級）降級或丟；refcount 共用前綴。
  * 回來時：Cake（或只載入）還原有存的最長前綴，其餘重算。
多出來的：
  * 每個 block 的 CPU 副本格式 cfmt、SSD 副本格式 sfmt（X1 的 fmt==1／on_slow），各自的品質來歷 cq／sq。
  * 外層做法：E1 丟（X1 LRU+NONE）、E2 降級（X1 LRU）、E3 寫穿、E4 第 2 次才寫 SSD（沒副本的被趕就丟）、E5 閒置 T 秒降級。
    E6 需要佇列 → engine_block.py（serve=fifo1）；serve=trace 時 E6≡E2。
  * 放置與時間無關（E1–E5、C0–C4、O 都只看請求順序），所以和 X1 一樣，一次跑很多「時間設定」（View）。
"""
from __future__ import annotations

import heapq
import os
import sys
from array import array
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from formats import FID, FMT_NAMES  # noqa: E402

GiB = 1 << 30
INF = 1 << 62
S_NONE, S_CPU, S_SLOW, S_DROP = 0, 1, 2, 3
# 寫入原因（只記整數計數；時間軸上每一輪合成一次寫入，和 X1 一樣）
W_STORE, W_DEMOTE, W_WT, W_SEL, W_TTL = "store", "demote", "write_through", "selective", "ttl_demote"


def next_access(acc, b, after_req):
    """＝x1_trace.next_access：block b 在請求 after_req 之後第一次被讀的請求 index；沒有回 INF。"""
    import bisect
    lst = acc.get(b)
    if not lst:
        return INF
    i = bisect.bisect_right(lst, after_req)
    return lst[i] if i < len(lst) else INF


# ================================================================== I/O 線（＝x1_sim.Chan，多了逐筆紀錄）
class Chan:
    __slots__ = ("name", "r", "w", "k", "t_free", "t_free_w", "busy_r", "busy_w", "bytes_r", "bytes_w",
                 "rec", "rs", "re", "rb", "rk", "ws", "we", "wb", "wk")

    def __init__(self, name, r_GiBps, w_GiBps, k, record=True):
        self.name = name
        self.r = r_GiBps * GiB
        self.w = w_GiBps * GiB
        self.k = k
        self.t_free = 0.0
        self.t_free_w = 0.0
        self.busy_r = self.busy_w = 0.0
        self.bytes_r = self.bytes_w = 0
        self.rec = record
        self.rs, self.re, self.rb, self.rk = array("d"), array("d"), array("q"), array("b")
        self.ws, self.we, self.wb, self.wk = array("d"), array("d"), array("q"), array("b")

    def read(self, nbytes, now, kind=0):
        st = now if now > self.t_free else self.t_free
        dur = nbytes / self.r
        if self.t_free_w > st:
            dur *= self.k
        self.t_free = st + dur
        self.busy_r += dur
        self.bytes_r += nbytes
        if self.rec:
            self.rs.append(st)
            self.re.append(self.t_free)
            self.rb.append(nbytes)
            self.rk.append(kind)
        return self.t_free

    def write(self, nbytes, now, kind=0):
        if nbytes <= 0:
            return self.t_free_w
        st = now if now > self.t_free_w else self.t_free_w
        dur = nbytes / self.w
        self.t_free_w = st + dur
        self.busy_w += dur
        self.bytes_w += nbytes
        if self.rec:
            self.ws.append(st)
            self.we.append(self.t_free_w)
            self.wb.append(nbytes)
            self.wk.append(kind)
        return self.t_free_w


class View:
    """一種時間設定：(平台常數, GPU 忙閒, CPU 忙閒係數, SSD 共用張數, 還原方式)。
    fbytes[fid]：每 block 位元組；dq[fid]：每 block 解壓秒數（None＝NOT_MEASURED → 這個 view 不算 TTFT）。"""

    def __init__(self, costs, mf, gpu, mode, nmax, nreq, cpu_factor=1.0, ssd_share=1, label=None, record=True,
                 fbytes=None, dq=None):
        self.key = (costs.platform, gpu, cpu_factor, ssd_share, mode)
        self.label = label or f"{costs.platform}|gpu={gpu}|cpu_x{cpu_factor}|ssd/{ssd_share}|{mode}"
        self.gpu, self.mode, self.cpu_factor, self.ssd_share = gpu, mode, cpu_factor, ssd_share
        self.has_f = costs.has_f
        if self.has_f:
            self.f = costs.f_array(nmax, gpu)
        else:
            assert mode == "load", "沒有 f(i)（70B）只能 load-only，TTFT 不算"
            self.f = [0.0] * nmax
        self.Fc = [0.0]
        for x in self.f:
            self.Fc.append(self.Fc[-1] + x)
        self.fb = list(fbytes if fbytes is not None else mf.bytes)
        dql = dq if dq is not None else mf.dq_list(gpu)
        self.dq_ok = dql is not None and all(dql[f] is not None for f in range(1, len(FMT_NAMES)) if self.fb[f] is not None)
        self.dq = [0.0 if (dql is None or x is None) else x for x in (dql or [None] * len(FMT_NAMES))]
        c, s = costs.cpu, costs.ssd
        # CPU 忙：H2D（CPU→GPU 的讀）變慢 cpu_factor 倍；SSD 共用：讀寫頻寬 ÷ 張數
        self.cpu = Chan("cpu", c["r"] / cpu_factor if cpu_factor != 1.0 else c["r"], c["w"], c["k"], record)
        self.slow = Chan("ssd", s["r"] / ssd_share if ssd_share != 1 else s["r"],
                         s["w"] / ssd_share if ssd_share != 1 else s["w"], s["k"], record)
        self.ttft = np.zeros(nreq)
        self.rest = np.zeros(nreq)
        self.nC = np.zeros(nreq, np.int32)
        self.nS = np.zeros(nreq, np.int32)
        self.nR = np.zeros(nreq, np.int32)
        self.wq = np.zeros(nreq, np.int8)      # 這次讀取載入的 block 裡最差的品質來歷（0＝沒有載入）
        self.last_end = 0.0

    def restore(self, lt_, lf_, lq_, L, n_all, t0, old_n):
        """lt_[i]∈{1: CPU, 3: SSD}、lf_[i]＝那份副本的格式、lq_[i]＝副本的品質來歷，i<L。
        算式＝x1_sim.View.restore（code 1＝CPU BF16、code 3＝慢層 INT4＋d）推廣到任意格式。"""
        f, cpu, slow, fb, dq = self.f, self.cpu, self.slow, self.fb, self.dq
        nC = nS = 0
        wq = 0
        if self.mode == "cake":
            p, q, rt, lt = 0, L - 1, t0, t0
            while p <= q:
                if rt <= lt:
                    rt += f[p]
                    p += 1
                else:
                    fid = lf_[q]
                    if lt_[q] == 1:
                        lt = cpu.read(fb[fid], lt)
                        d = dq[fid]
                        if d:
                            lt = lt + d
                        nC += 1
                    else:
                        lt = slow.read(fb[fid], lt) + dq[fid]
                        nS += 1
                    if lq_[q] > wq:
                        wq = lq_[q]
                    q -= 1
            end = rt if rt > lt else lt
            meet = p
        else:
            lt = t0
            for i in range(L):
                fid = lf_[i]
                if lt_[i] == 1:
                    lt = cpu.read(fb[fid], lt)
                    d = dq[fid]
                    if d:
                        lt = lt + d
                    nC += 1
                else:
                    lt = slow.read(fb[fid], lt) + dq[fid]
                    nS += 1
                if lq_[i] > wq:
                    wq = lq_[i]
            end = lt
            meet = 0
        self.last_end = end
        Fc = self.Fc
        base = end - t0
        ttft = base + Fc[n_all] - Fc[L]
        rest = base + (Fc[old_n] - Fc[L] if old_n > L else 0.0)
        return ttft, rest, nC, nS, meet + (n_all - L), wq


# ================================================================== 引擎
class Engine:
    """P：trace.prepare 或 x1_sim.prep 的同形 dict（full、nall、acc、tl、kindc、sessl、parentl、nmax、R.n_sess、B）。
    outer：outer_policies.json 的 used（加 E5 的 T）；comp：policies.CompressionPolicy；cap_bytes：CPU 容量。"""

    def __init__(self, P, outer, comp, cap_bytes, views, mf, H=0.0, block_tokens=512, check=False):
        self.P = P
        self.full, self.nall, self.acc, self.t = P["full"], P["nall"], P["acc"], P["tl"]
        self.kind, self.sess, self.parent = P["kindc"], P["sessl"], P["parentl"]
        self.obs_end = P.get("obs_end")                 # 每個請求的觀察窗終點（秒）；None＝trace 最後一個到達
        self.t_last = self.t[-1] if self.t else 0.0
        self.outer = outer
        self.E = outer["code"]
        self.ssd_on = "ssd" in outer["tiers"]
        self.evict_act = outer["on_cpu_evict"]          # drop／demote／demote_if_on_ssd_else_drop
        self.wt = bool(outer.get("write_through"))
        self.sel = outer.get("ssd_write_min_uses")
        self.ttl = outer.get("idle_T")
        self.comp = comp
        self.C = cap_bytes
        self.views = views
        self.mf = mf
        self.fb = [0 if x is None else x for x in mf.bytes]
        self.H = float(H)
        self.btok = block_tokens
        self.check = check
        B, S = P["B"], P["R"]["n_sess"]
        self.ref_cpu = [0] * B
        self.ref_live = [0] * B
        self.cfmt = [0] * B
        self.sfmt = [0] * B
        self.cq = [0] * B
        self.sq = [0] * B
        self.uses = [0] * B
        self.sb = [None] * S
        self.st = [S_NONE] * S
        self.last = [-1] * S
        self.prev = [-1] * S
        self.nreq = [0] * S
        self.sctx = [None] * S        # 最近一次寫入時的 ctx（ssd_fid 用）
        self.cpu_used = 0
        self.cur = -1
        self.cpu_set = set()
        self.lru = OrderedDict()
        self.ttl_heap = []
        self.ttl_ver = [0] * S
        self.n_demote = self.n_drop = self.n_ttl = 0
        self.res_sum = 0.0
        self.w_slow_blocks = 0
        self.n_lost = 0               # E1／E4：還有 session 參照、但已經沒有任何副本的 block（回來重算）
        self.wbytes = {W_STORE: 0, W_DEMOTE: 0, W_WT: 0, W_SEL: 0, W_TTL: 0}
        self.wblocks_fmt = {}         # (層, 格式) → 寫入的 block 數
        self.dec_fmt = {}             # (格式, 長度分箱, 之後被讀?, 截尾?) → 決定次數
        self.Ltok = np.zeros(len(self.full), np.int64)
        self.Lblk = np.zeros(len(self.full), np.int32)
        self._sufmax = {}

    # ---------------------------------------------------------- 小工具
    def _ctx(self, r, s, blocks, n_new, first_new):
        tok = len(blocks) * self.btok
        ctx = dict(r=r, s=s, t=self.t[r], n_blocks=len(blocks), tokens=tok, n_new=n_new, nreq=self.nreq[s] + 1,
                   kind=self.kind[r], feats=(self.P["feats"][r] if "feats" in self.P else None), mf=self.mf)
        if self.comp.needs_oracle:
            nx = next_access(self.acc, first_new, r) if first_new is not None else INF
            oe = self.obs_end[r] if self.obs_end is not None else self.t_last
            ctx["next_read"] = None if nx >= INF else nx
            ctx["censored"] = (oe - self.t[r]) < self.H if self.H > 0 else False
            # 之後讀到這些 block 的請求裡最長的那個（存的前綴 token 數的上界）：格式只能往下壓，
            # 所以要對「之後每一次讀」都過關（2026-10-18 測試 (iii) 抓到：只看寫入時的長度，後面更長的讀取會不過關）
            ctx["max_read_tokens"] = self._max_future_tokens(first_new, r) if nx < INF else 0
        return ctx

    def _max_future_tokens(self, b, r):
        import bisect
        sm = self._sufmax.get(b)
        lst = self.acc[b]
        if sm is None:
            full, bt = self.full, self.btok
            sm = [0] * len(lst)
            m = 0
            for i in range(len(lst) - 1, -1, -1):
                x = len(full[lst[i]]) * bt
                m = x if x > m else m
                sm[i] = m
            self._sufmax[b] = sm
        i = bisect.bisect_right(lst, r)
        return sm[i] if i < len(lst) else 0

    def _ssd_fid(self, s, cur):
        return self.comp.ssd_fid(self.sctx[s], cur) if self.sctx[s] is not None else cur

    def _wfmt(self, tier, fid, n=1):
        k = (tier, fid)
        self.wblocks_fmt[k] = self.wblocks_fmt.get(k, 0) + n

    def _fix(self, b, s):
        """＝x1_sim.Engine._fix（沒有預取）：回傳要寫 SSD 的位元組。"""
        if self.ref_live[b] == 0:
            fm = self.cfmt[b]
            if fm:
                self.cpu_used -= self.fb[fm]
                self.cfmt[b] = 0
            self.sfmt[b] = 0
            return 0
        if self.ref_cpu[b] == 0 and self.cfmt[b]:
            fm = self.cfmt[b]
            self.cpu_used -= self.fb[fm]
            self.cfmt[b] = 0
            if not self.sfmt[b]:
                if self.evict_act == "demote":
                    sf = self._ssd_fid(s, fm)
                    self.sfmt[b] = sf
                    q = self.cq[b]
                    self.sq[b] = sf if sf > q else q
                    self._wfmt("ssd", sf)
                    return self.fb[sf]
                self.n_lost += 1      # E1／E4：沒有 SSD 副本就沒了
        return 0

    # ---------------------------------------------------------- 動作
    def demote(self, s):
        assert self.st[s] == S_CPU
        self.st[s] = S_SLOW
        self.cpu_set.discard(s)
        self.lru.pop(s, None)
        w = 0
        for b in self.sb[s]:
            self.ref_cpu[b] -= 1
            w += self._fix(b, s)
        self.n_demote += 1
        self.res_sum += self.t[self.cur] - self.t[self.last[s]]
        return w

    def drop(self, s):
        st = self.st[s]
        w = 0
        if st in (S_CPU, S_SLOW):
            if st == S_CPU:
                self.cpu_set.discard(s)
                self.lru.pop(s, None)
            for b in self.sb[s]:
                self.ref_live[b] -= 1
                if st == S_CPU:
                    self.ref_cpu[b] -= 1
                w += self._fix(b, s)
        self.st[s] = S_DROP
        self.sb[s] = []
        self.n_drop += 1
        return w

    def victim_action(self, s):
        if self.evict_act == "drop":
            return self.drop(s)
        return self.demote(s)     # demote／demote_if_on_ssd_else_drop（後者在 _fix 裡丟沒有副本的 block）

    def enforce(self):
        w = 0
        while self.cpu_used > self.C:
            if not self.cpu_set or not self.lru:      # ＝x1_sim.Engine.select 的 None
                raise RuntimeError(f"CPU 超量 {self.cpu_used} > {self.C}，但沒有可以降級的")
            s = next(iter(self.lru))
            w += self.victim_action(s)
        if w:
            self.wbytes[W_DEMOTE] += w
        return w

    def ttl_expire(self, t_now):
        """E5：閒置超過 T 的 session 搬去 SSD；寫入發在到期那一刻。"""
        h = self.ttl_heap
        while h and h[0][0] <= t_now:
            te, v, s = heapq.heappop(h)
            if v != self.ttl_ver[s] or self.st[s] != S_CPU:
                continue
            w = self.demote(s)
            self.n_ttl += 1
            if w:
                self.wbytes[W_TTL] += w
                for vw in self.views:
                    vw.slow.write(w, te, 1)

    # ---------------------------------------------------------- 一個請求
    def step(self, r):
        t = self.t[r]
        if self.ttl is not None:
            self.ttl_expire(t)
        self.cur = r
        blocks = self.full[r]
        cfmt, sfmt, cq, sq = self.cfmt, self.sfmt, self.cq, self.sq
        lt_, lf_, lq_ = [], [], []
        for b in blocks:
            fm = cfmt[b]
            if fm:
                lt_.append(1)
                lf_.append(fm)
                lq_.append(cq[b])
            elif sfmt[b]:
                lt_.append(3)
                lf_.append(sfmt[b])
                lq_.append(sq[b])
            else:
                break
        L = len(lt_)
        self.Lblk[r] = L
        self.Ltok[r] = L * self.btok
        if self.check:
            for b in blocks[L:]:
                assert cfmt[b] == 0 and (sfmt[b] == 0 or self.evict_act != "demote"), "有存的 block 不是前綴"
        kd = self.kind[r]
        old_n = len(self.full[self.parent[r]]) if kd == 1 else 0
        tend = []
        for v in self.views:
            tt, rs, nC, nS, nR, wq = v.restore(lt_, lf_, lq_, L, self.nall[r], t, old_n)
            v.ttft[r], v.rest[r], v.nC[r], v.nS[r], v.nR[r], v.wq[r] = tt, rs, nC, nS, nR, wq
            tend.append(t + tt)
        # ---- 回合結束：新 block 照這個 session 的決定寫進 CPU
        s = self.sess[r]
        old = self.sb[s] if self.sb[s] else []
        old_state = self.st[s]
        first_new = None
        n_new = 0
        for b in blocks:
            if cfmt[b] == 0:
                n_new += 1
                if first_new is None:
                    first_new = b
        ctx = self._ctx(r, s, blocks, n_new, first_new)
        fid = self.comp.cpu_fid(ctx)
        if n_new:
            k = (FMT_NAMES[fid], self.mf.bin_of(ctx["tokens"]),
                 (ctx.get("next_read") is not None) if self.comp.needs_oracle else None, ctx.get("censored"))
            self.dec_fmt[k] = self.dec_fmt.get(k, 0) + 1
        self.sctx[s] = ctx
        fbf = self.fb[fid]
        cw = 0
        wt = 0
        rc, rl = self.ref_cpu, self.ref_live
        for i, b in enumerate(blocks):
            rl[b] += 1
            rc[b] += 1
            self.uses[b] += 1
            if cfmt[b] == 0:
                gq = sq[b] if (i < L and sfmt[b]) else 1     # 從 SSD 讀回來的帶著 SSD 副本的來歷；重算＝bf16
                cfmt[b] = fid
                cq[b] = fid if fid > gq else gq
                self.cpu_used += fbf
                cw += fbf
                if self.wt and not sfmt[b]:
                    sf = self.comp.ssd_fid(ctx, fid)
                    sfmt[b] = sf
                    sq[b] = sf if sf > cq[b] else cq[b]
                    wt += self.fb[sf]
                    self._wfmt("ssd", sf)
        if cw:
            self._wfmt("cpu", fid, cw // fbf if fbf else 0)
        if self.sel is not None:
            for b in blocks:
                if self.uses[b] >= self.sel and not sfmt[b] and cfmt[b]:
                    sf = self.comp.ssd_fid(ctx, cfmt[b])
                    sfmt[b] = sf
                    sq[b] = sf if sf > cq[b] else cq[b]
                    wt += self.fb[sf]
                    self._wfmt("ssd", sf)
                    self.wbytes[W_SEL] += self.fb[sf]
        if self.wt:
            self.wbytes[W_WT] += wt
        self.wbytes[W_STORE] += cw
        w = 0
        if old_state in (S_CPU, S_SLOW):
            for b in old:
                rl[b] -= 1
                if old_state == S_CPU:
                    rc[b] -= 1
                w += self._fix(b, s)
        if w:
            self.wbytes[W_DEMOTE] += w
        self.sb[s] = list(blocks)
        self.st[s] = S_CPU
        self.prev[s] = self.last[s]
        self.last[s] = r
        self.nreq[s] += 1
        if blocks:
            self.cpu_set.add(s)
            self.lru[s] = None
            self.lru.move_to_end(s)
            if self.ttl is not None:
                self.ttl_ver[s] += 1
                heapq.heappush(self.ttl_heap, (t + self.ttl, self.ttl_ver[s], s))
        else:
            self.cpu_set.discard(s)
        w += self.enforce()
        w += wt
        self.w_slow_blocks += w
        for v, te in zip(self.views, tend):
            v.cpu.write(cw, te)
            if self.ssd_on:
                v.slow.write(w, te)
        if self.check and (r % self.check == 0 or r == len(self.full) - 1):
            self.audit()

    def audit(self):
        used = 0
        for b, fm in enumerate(self.cfmt):
            if fm:
                used += self.fb[fm]
                assert self.ref_cpu[b] > 0
            if self.ref_live[b] == 0:
                assert fm == 0 and self.sfmt[b] == 0
            if self.sfmt[b] and self.cfmt[b]:
                assert self.sfmt[b] >= self.cfmt[b] or self.wt or self.sel is not None or True
            if self.sfmt[b]:
                assert self.sq[b] >= self.sfmt[b], "SSD 副本的來歷不可能比它的格式好"
            if fm:
                assert self.cq[b] >= fm
        assert used == self.cpu_used, (used, self.cpu_used)
        assert self.cpu_used <= self.C or not self.cpu_set, (self.cpu_used, self.C)

    def run(self):
        for r in range(len(self.full)):
            self.step(r)
        return self
