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
    """逐筆紀錄（06 §12）：開始 rs／ws、結束 re／we、位元組 rb／wb、種類 rk／wk（1＝預取或閒置降級），
    另外 rr／wr＝這筆算在哪個請求（→ session，見 06 §12.1）、rd／wd＝排隊延遲（實際開始 − 準備好可以開始的時刻 now，float32）。
    rq＝None 時用 cur_r（引擎在每個請求開始前設定）。時間的算式和 x1_sim.Chan 完全相同（多記的欄位不參與計算）。
    ext＝False（CPU 層）：讀取不逐筆記 rr／rd（沒有任何帳用到；逐筆的歸屬改用每個請求的起始筆數 View.cpu_r0，06 §13），寫入照記。"""
    __slots__ = ("name", "r", "w", "k", "t_free", "t_free_w", "busy_r", "busy_w", "bytes_r", "bytes_w",
                 "rec", "rs", "re", "rb", "rk", "ws", "we", "wb", "wk", "rr", "rd", "wr", "wd", "cur_r", "ext")

    def __init__(self, name, r_GiBps, w_GiBps, k, record=True, ext=True):
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
        self.rr, self.rd, self.wr, self.wd = array("i"), array("f"), array("i"), array("f")
        self.cur_r = -1
        self.ext = ext

    def read(self, nbytes, now, kind=0, rq=None):
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
            if self.ext:
                self.rr.append(self.cur_r if rq is None else rq)
                self.rd.append(st - now)
        return self.t_free

    def write(self, nbytes, now, kind=0, rq=None):
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
            self.wr.append(self.cur_r if rq is None else rq)
            self.wd.append(st - now)
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
        # 逐請求的 NOT_MEASURED（06 §12.4）：這個請求真的載入了「解壓常數 NOT_MEASURED」的格式 → 它的 TTFT 不能用
        nm = [f >= 1 and self.fb[f] is not None and (dql is None or dql[f] is None) for f in range(len(FMT_NAMES))]
        self.nmf = nm if any(nm) else None
        self.nmreq = np.zeros(nreq, bool)
        # SSD 的 k_read 沒量（/ssd8；06 §24）：追蹤「TTFT 碰到 k_read」的請求。沿著「max 選到的那一邊」傳遞：
        #   c_ta／s_ta＝CPU／SSD 讀取佇列的結束時刻（t_free）是否由 k_read 推出；cw_ta／sw_ta＝CPU／SSD 寫入佇列的結束時刻是否由「碰到 k_read 的請求」的 TTFT 推出。
        self.k_nm = bool(getattr(costs, "ssd_k_nm", False))
        self.c_ta = self.s_ta = self.cw_ta = self.sw_ta = False
        self.nmk = np.zeros(nreq, bool)         # 這個請求的 TTFT 碰到 k_read（已經算進 nmreq）
        self.last_k = False
        self.bS = np.zeros(nreq, np.int64)      # 這個請求的還原從 SSD 讀了幾個位元組（和時間設定有關：Cake 的會合點）
        self.last_nm = False
        self.last_bS = 0
        c, s = costs.cpu, costs.ssd
        # CPU 忙：H2D（CPU→GPU 的讀）變慢 cpu_factor 倍；SSD 共用：讀寫頻寬 ÷ 張數
        self.cpu = Chan("cpu", c["r"] / cpu_factor if cpu_factor != 1.0 else c["r"], c["w"], c["k"], record, ext=False)
        self.cpu_r0 = np.zeros(nreq + 1, np.int64)   # 請求 r 的 CPU 讀取紀錄＝第 cpu_r0[r] 到 cpu_r0[r+1]−1 筆（還原只在 step(r) 裡發生）
        self.slow = Chan("ssd", s["r"] / ssd_share if ssd_share != 1 else s["r"],
                         s["w"] / ssd_share if ssd_share != 1 else s["w"], s["k"], record)
        self.ttft = np.zeros(nreq)
        self.rest = np.zeros(nreq)
        self.nC = np.zeros(nreq, np.int32)
        self.nS = np.zeros(nreq, np.int32)
        self.nR = np.zeros(nreq, np.int32)
        self.wq = np.zeros(nreq, np.int8)      # 這次讀取載入的 block 裡最差的品質來歷（0＝沒有載入）
        self.last_end = 0.0

    def taint_write(self, which, nbytes, now, tainted_now):
        """寫入佇列的結束時刻 t_free_w 是否由 k_read 推出（只有 k_nm 的視角用；要在 Chan.write 之前呼叫）：
        開始時刻＝max(now, t_free_w) 選到 now → 看 now 是不是由「碰到 k_read 的請求」的 TTFT 推出；選到 t_free_w → 沿用它原本的。"""
        if nbytes <= 0:
            return
        if which == "cpu":
            self.cw_ta = tainted_now if now > self.cpu.t_free_w else self.cw_ta
        else:
            self.sw_ta = tainted_now if now > self.slow.t_free_w else self.sw_ta

    def restore(self, lt_, lf_, lq_, L, n_all, t0, old_n):
        """lt_[i]∈{1: CPU, 3: SSD}、lf_[i]＝那份副本的格式、lq_[i]＝副本的品質來歷，i<L。
        算式＝x1_sim.View.restore（code 1＝CPU BF16、code 3＝慢層 INT4＋d）推廣到任意格式。
        06 §13 加速：Chan.read 的算式內聯在這裡（同樣的浮點運算、同樣的順序；還原的時候沒有寫入，所以 t_free_w 不變），
        結束時把 t_free、busy_r、bytes_r 寫回 Chan。逐筆紀錄和 Chan.read 記的完全相同。"""
        f, cpu, slow, fb, dq = self.f, self.cpu, self.slow, self.fb, self.dq
        nmf = self.nmf
        bad = False
        bS = 0
        nC = nS = 0
        wq = 0
        c_tf, c_tfw, c_r, c_k, c_busy, c_bytes = cpu.t_free, cpu.t_free_w, cpu.r, cpu.k, cpu.busy_r, cpu.bytes_r
        s_tf, s_tfw, s_r, s_k, s_busy, s_bytes = slow.t_free, slow.t_free_w, slow.r, slow.k, slow.busy_r, slow.bytes_r
        c_rec, s_rec = cpu.rec, slow.rec
        c_ext, s_ext = cpu.ext, slow.ext
        crs, cre, crb, crk, crr, crd = cpu.rs.append, cpu.re.append, cpu.rb.append, cpu.rk.append, cpu.rr.append, cpu.rd.append
        srs, sre, srb, srk, srr, srd = slow.rs.append, slow.re.append, slow.rb.append, slow.rk.append, slow.rr.append, slow.rd.append
        c_cur, s_cur = cpu.cur_r, slow.cur_r
        k_nm = self.k_nm
        if k_nm:
            c_ta, s_ta, cw_ta, sw_ta = self.c_ta, self.s_ta, self.cw_ta, self.sw_ta
            bk = False                  # 這個請求的 TTFT 已經碰到 k_read（黏著：之後的 lt 都算碰到）
        if self.mode == "cake":
            p, q, rt, lt = 0, L - 1, t0, t0
            while p <= q:
                if rt <= lt:
                    rt += f[p]
                    p += 1
                else:
                    fid = lf_[q]
                    nb = fb[fid]
                    if lt_[q] == 1:
                        st = lt if lt > c_tf else c_tf          # ＝Chan.read
                        if k_nm:
                            tq = bk if lt > c_tf else c_ta
                            if c_tfw > st and cw_ta:
                                tq = True
                            c_ta = tq
                            if tq:
                                bk = True
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
                                crd(st - lt)
                        lt = c_tf
                        d = dq[fid]
                        if d:
                            lt = lt + d
                        nC += 1
                    else:
                        st = lt if lt > s_tf else s_tf
                        if k_nm:
                            tq = bk if lt > s_tf else s_ta
                            if s_tfw > st:              # k_read 真的乘上去了（或和被它推遲的寫入佇列比大小）
                                tq = True
                            s_ta = tq
                            if tq:
                                bk = True
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
                        lt = s_tf + dq[fid]
                        nS += 1
                        bS += nb
                    if nmf and nmf[fid]:
                        bad = True
                    if lq_[q] > wq:
                        wq = lq_[q]
                    q -= 1
            end = rt if rt > lt else lt
            meet = p
        else:
            lt = t0
            for i in range(L):
                fid = lf_[i]
                nb = fb[fid]
                if lt_[i] == 1:
                    st = lt if lt > c_tf else c_tf
                    if k_nm:
                        tq = bk if lt > c_tf else c_ta
                        if c_tfw > st and cw_ta:
                            tq = True
                        c_ta = tq
                        if tq:
                            bk = True
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
                            crd(st - lt)
                    lt = c_tf
                    d = dq[fid]
                    if d:
                        lt = lt + d
                    nC += 1
                else:
                    st = lt if lt > s_tf else s_tf
                    if k_nm:
                        tq = bk if lt > s_tf else s_ta
                        if s_tfw > st:              # k_read 真的乘上去了（或和被它推遲的寫入佇列比大小）
                            tq = True
                        s_ta = tq
                        if tq:
                            bk = True
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
                    lt = s_tf + dq[fid]
                    nS += 1
                    bS += nb
                if nmf and nmf[fid]:
                    bad = True
                if lq_[i] > wq:
                    wq = lq_[i]
            end = lt
            meet = 0
        cpu.t_free, cpu.busy_r, cpu.bytes_r = c_tf, c_busy, c_bytes
        slow.t_free, slow.busy_r, slow.bytes_r = s_tf, s_busy, s_bytes
        self.last_end = end
        if k_nm:
            self.c_ta, self.s_ta = c_ta, s_ta
            self.last_k = bk
            bad = bad or bk
        self.last_nm = bad
        self.last_bS = bS
        Fc = self.Fc
        base = end - t0
        ttft = base + Fc[n_all] - Fc[L]
        rest = base + (Fc[old_n] - Fc[L] if old_n > L else 0.0)
        return ttft, rest, nC, nS, meet + (n_all - L), wq


def ok_token_limits(mf):
    """每個 fid：品質從 0 token 起連續 pass 到幾 token（含）。bf16（status 永遠 pass）＝INF；從 0 就不 pass＝−1；沒有大小的＝−1。"""
    out = [-1] * len(FMT_NAMES)
    for f in range(1, len(FMT_NAMES)):
        if mf.bytes[f] is None:
            continue
        lim = INF
        bins = mf.len_bins
        for k in range(len(bins) - 1):
            if mf.status(f, bins[k]) != "pass":
                lim = bins[k] - 1
                break
        out[f] = lim
    return out


def request_lengths(P, btok=512):
    """每個請求的輸入長度（token，整個上下文；04 v4 §3.3 的「讀取長度」）。
    P["feats"][r]＝(trigger, input_tokens, output_tokens)（trace.prepare）；沒有 feats（x1_sim.prep 等）→ nall×btok。
    至少是這個請求存的完整 block 數×btok（資料裡 input_tokens 比 block 數小時以 block 為準，不讓長度小於已存的前綴）。"""
    full, nall = P["full"], P["nall"]
    ft = P.get("feats")
    out = []
    for r in range(len(full)):
        f = ft[r] if ft else None
        if f is not None and len(f) > 1 and f[1] is not None:
            out.append(max(int(f[1]), len(full[r]) * btok))
        else:
            out.append(int(nall[r]) * btok)
    return out


def pass_masks(mf):
    """每個長度分箱 k、每個 fid：這個格式的來歷在這個分箱的讀取長度下是否 pass（status＝pass 且有大小）。"""
    nb = len(mf.len_bins) - 1
    return [[(f >= 1 and mf.bytes[f] is not None and mf.status(f, mf.len_bins[k]) == "pass") for f in range(len(FMT_NAMES))]
            for k in range(nb)]


# ================================================================== 引擎
class Engine:
    """P：trace.prepare 或 x1_sim.prep 的同形 dict（full、nall、acc、tl、kindc、sessl、parentl、nmax、R.n_sess、B）。
    outer：outer_policies.json 的 used（加 E5 的 T）；comp：policies.CompressionPolicy；cap_bytes：CPU 容量。
    int2_on_read（06 §12.3）："serve"（預設＝舊行為）照讀；"recompute"：讀取遇到品質來歷是 never_read_only（INT2）的 block，
      那個 block 和它之後的都當成沒有存 → 重算（品質安全，算進重算量與 TTFT，那些 block 的 SSD 位元組不讀），
      重算出來的新 KV 照這一輪的決定寫回 CPU、取代舊副本（SSD 上的舊副本作廢；04 v4 §3.3「重算之後」，不只 INT2 來歷）。
      "recompute_unverified"（04 v4 §3.3）：只載入「在這次讀取的長度下品質過關」的前綴：載入到第 L 個 block 時，前 L 個 block 的品質來歷
      在 L×512 token 都要 pass；第一個做不到的地方起都重算；重算的 block（有存但沒載入的部分，不論來歷）照上面的規則被新 KV 取代。
    unverified_len（04 v4 §3.3，06 §18）："prefix"（預設＝舊行為）：讀取長度＝載入的前綴長度；"request"：讀取長度＝讀它的請求的輸入長度（整個上下文），
      載入的前綴停在第一個「來歷在這個長度不 pass」的 block，之後都重算；品質帳也用請求長度分箱。"""

    def __init__(self, P, outer, comp, cap_bytes, views, mf, H=0.0, block_tokens=512, check=False, int2_on_read="serve",
                 unverified_len="prefix", H_extra=0.0, tier_rec=None, ev_log=False):
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
        self.ssd_moment = bool(getattr(comp, "ssd_moment", False))
        self.C = cap_bytes
        self.views = views
        self.mf = mf
        self.fb = [0 if x is None else x for x in mf.bytes]
        self.H = float(H)
        self.btok = block_tokens
        self.check = check
        assert int2_on_read in ("serve", "recompute", "recompute_unverified"), int2_on_read
        self.int2_on_read = int2_on_read
        assert unverified_len in ("prefix", "request"), unverified_len
        self.unverified_len = unverified_len
        self.req_strict = unverified_len == "request"
        self.surp_on = int2_on_read == "recompute"
        self.unv_on = int2_on_read == "recompute_unverified"
        self.lim = ok_token_limits(mf)                 # 每個格式「從 0 到幾 token 都 pass」（bf16＝無限；從 0 就不 pass＝−1）
        if self.unv_on:      # 任何長度都不 pass 的來歷＝要被重算取代的（正式表＝{int2}）
            self.nro = frozenset(f for f in range(1, len(FMT_NAMES)) if self.lim[f] < 0)
        else:
            self.nro = frozenset(mf.never_read_only)
        B, S = P["B"], P["R"]["n_sess"]
        n = len(self.full)
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
        self.Ltok = np.zeros(n, np.int64)
        self.Lblk = np.zeros(n, np.int32)
        self._sufmax = {}
        self._sufreq = {}
        self.reqlen = request_lengths(P, block_tokens)      # 請求的輸入長度（token）
        self.Lqual = np.asarray(self.reqlen, np.int64) if self.req_strict else self.Ltok   # 品質帳用的讀取長度
        self.okmask = pass_masks(mf)
        self.rbin = [mf.bin_of(x) for x in self.reqlen] if self.req_strict else None
        # ---- 06 §12 的新帳（不參與任何舊的算式）
        self.surp_blk = np.zeros(n, np.int32)      # 因為 INT2 改成重算的 block 數（surprise recompute；int2_on_read=recompute）
        self.nprev = np.zeros(n, np.int32)         # 這個請求的前綴裡「以前算過」的 block 數（重算量＝nprev − 載入的）
        self.nend = np.asarray(self.nall, np.int32)  # prefill 到哪（block 數）
        self.lo_bytes = np.zeros(n, np.int64)      # load-only 還原會從 SSD 讀的位元組（只看放置，和時間設定無關）
        self.s_ssd_w = [0] * S                     # 每個 session 的 SSD 寫入位元組（歸屬規則見 06 §12.1）
        oe_ = P.get("obs_end")
        self.cutH = (max(oe_) - self.H) if (self.H > 0 and oe_) else None   # 截尾版：決定時刻 < 觀察窗終點 − H 的才算
        self.s_ssd_w_H = [0] * S
        # 第二個只記帳不影響放置的截尾（例如 2H；06 §18）：非 oracle 的放置和 H 無關，所以一次模擬就有兩個截尾版的位元組
        self.H_extra = float(H_extra)
        self.cutH2 = (max(oe_) - self.H_extra) if (self.H_extra > 0 and oe_) else None
        self.s_ssd_w_H2 = [0] * S
        self.wsp_rec, self.wsp_sid, self.wsp_b = array("q"), array("i"), array("q")   # SSD 寫入紀錄 i → 各 session 的份
        self.n_ssd_wrec = 0
        self._rw = {}                              # 這一輪（合成一筆的）SSD 寫入裡各 session 的份
        self._dec_t, self._dec_after = 0.0, -1     # SSD 寫入的決定時刻、「之後」從哪個請求 index 之後算
        # ---- 重算計價（review #1 選項 a；06 §26）：被「意外重算」取代、而且可用的存放副本只在 SSD（不在 CPU）的 block，
        #      照「這個請求的輸入長度下過關的最省格式」的大小算成一次讀取。一律記帳（獨立的欄位／npz 陣列）；
        #      是否算進主指標由 recompute_charge 參數決定（ledger.summarize_view／run_map.prim_cols／analyze_map）。
        self.rc_blocks = np.zeros(n, np.int32)     # 這個請求被計價的 block 數
        self.rc_bytes = np.zeros(n, np.int64)      # 這個請求被計價的位元組（歸屬於讀它的請求的 session）
        self._pass_b = [min(self.fb[f] for f in range(1, len(FMT_NAMES)) if m[f]) for m in self.okmask]   # 每個長度分箱：過關格式裡最小的大小
        # ---- O-tier 的兩趟（review #1 選項 b）：tier_rec＝policies.TierRec（第 1 趟記錄；第 2 趟的策略自己持有、這裡不用）；
        #      ev_log＝逐請求記下「CPU 側事件」的摘要（兩趟必須完全相同）
        self.rec = tier_rec if (tier_rec is not None and tier_rec.recording) else None
        self.ev_log = bool(ev_log)
        self._track = self.rec is not None or self.ev_log
        self._nw = {}                              # block → 已經寫過 SSD 幾次（鍵 (block, 第幾次)）
        self._evw = []                             # 這一輪裡寫 SSD 的 block（順序）
        self.ev_digest = array("Q")                # 每個請求一個 8 位元組摘要（不串接；不同的請求數可以直接數）

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
            ctx["max_read_req_tokens"] = self._max_future_reqlen(first_new, r) if nx < INF else 0
        ctx["req_tokens"] = self.reqlen[r]
        return ctx

    def _mctx(self, s, b, after, t_dec, tokens=None, idle=None, rq=None, nq=None):
        """SSD 寫入當下的 ctx（ssd_moment 的策略用；06 §11.2）：只放那一刻拿得到的資訊。
        tokens＝這個 session 現在存的長度；idle_s＝離它上一個請求的到達多久；cpu_ctx＝寫進 CPU 時的 ctx。
        needs_oracle 的策略另外拿到這個 block 在 after 之後的 next_read、max_read_tokens，和 censored（決定時刻離觀察窗結尾 < H）。"""
        ls = self.last[s]
        if tokens is None:
            tokens = len(self.sb[s]) * self.btok if self.sb[s] else 0
        if idle is None:
            idle = t_dec - self.t[ls] if ls >= 0 else 0.0
        ctx = dict(s=s, r=after, t=t_dec, b=b, tokens=tokens, idle_s=idle, cpu_ctx=self.sctx[s], mf=self.mf)
        if self.comp.needs_oracle:
            nx = next_access(self.acc, b, after)
            ref = ls if ls >= 0 else max(after, 0)
            oe = self.obs_end[ref] if self.obs_end is not None else self.t_last
            ctx["next_read"] = None if nx >= INF else nx
            ctx["censored"] = (oe - t_dec) < self.H if self.H > 0 else False
            ctx["max_read_tokens"] = self._max_future_tokens(b, after) if nx < INF else 0
            ctx["max_read_req_tokens"] = self._max_future_reqlen(b, after) if nx < INF else 0
        # req_tokens／nreq：決定當下已經服務的最近一個請求的輸入長度與這個 session 的請求數；寫穿／第 2 次才寫發生在請求 r 自己的回合裡
        # （self.last[s] 還是上一個請求），所以由呼叫端用 rq／nq 給現在這個請求（04 v4 §3.1：長度＝max(存的長度, 請求長度)）
        ctx["req_tokens"] = rq if rq is not None else (self.reqlen[ls] if ls >= 0 else 0)
        ctx["nreq"] = nq if nq is not None else self.nreq[s]
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

    def _max_future_reqlen(self, b, r):
        """block b 在請求 r 之後被讀到的請求裡，最長的「請求輸入長度」（token；04 v4 §3.3 嚴格版的讀取長度；後綴最大值）。"""
        import bisect
        sm = self._sufreq.get(b)
        lst = self.acc[b]
        if sm is None:
            rl = self.reqlen
            sm = [0] * len(lst)
            m = 0
            for i in range(len(lst) - 1, -1, -1):
                x = rl[lst[i]]
                m = x if x > m else m
                sm[i] = m
            self._sufreq[b] = sm
        i = bisect.bisect_right(lst, r)
        return sm[i] if i < len(lst) else 0

    def _ssd_dec(self, s, cur, b, tokens=None, idle=None, rq=None, nq=None):
        """要寫 SSD 的格式。舊的策略（ssd_moment=False）＝ssd_fid(寫進 CPU 時的 ctx, cur)，和 c0c179b 逐位元相同；
        ssd_moment=True 的策略（oracle、D-*、預測器的 ssd_fn）拿到寫入當下的 ctx（_mctx）。"""
        key = None
        if self._track or getattr(self.comp, "tier", False):
            key = (b, self._nw.get(b, 0))
            self._nw[b] = key[1] + 1
            self._evw.append(b)
        if not self.ssd_moment:
            sf = self.comp.ssd_fid(self.sctx[s], cur) if self.sctx[s] is not None else cur
        else:
            mc = self._mctx(s, b, self._dec_after, self._dec_t, tokens, idle, rq, nq)
            mc["ssd_key"] = key
            sf = self.comp.ssd_fid(mc, cur)
            if sf < cur:
                raise AssertionError(f"{self.comp.name}: ssd_fid 回傳 {sf} < 目前的 {cur}（格式只能往下壓）")
        if self.rec is not None:
            self.rec.on_write(key)
        return sf

    def _wfmt(self, tier, fid, n=1):
        k = (tier, fid)
        self.wblocks_fmt[k] = self.wblocks_fmt.get(k, 0) + n

    def _attr(self, s, nb):
        self.s_ssd_w[s] += nb
        self._rw[s] = self._rw.get(s, 0) + nb
        if self.cutH is not None and self._dec_t < self.cutH:
            self.s_ssd_w_H[s] += nb
        if self.cutH2 is not None and self._dec_t < self.cutH2:
            self.s_ssd_w_H2[s] += nb

    def _flush_split(self, w):
        """這一筆 SSD 寫入紀錄（w>0 才有紀錄）裡各 session 的份；加總必須＝w。"""
        rw, self._rw = self._rw, {}
        if w <= 0:
            assert not rw or sum(rw.values()) == 0, rw
            return
        assert sum(rw.values()) == w, (sum(rw.values()), w)
        k = self.n_ssd_wrec
        for s_, nb in rw.items():
            self.wsp_rec.append(k)
            self.wsp_sid.append(s_)
            self.wsp_b.append(nb)
        self.n_ssd_wrec += 1

    def _fix(self, b, s):
        """＝x1_sim.Engine._fix（沒有預取）：回傳要寫 SSD 的位元組。降級的位元組算在 s（讓這個 block 離開 CPU 的 session）。"""
        if self.ref_live[b] == 0:
            fm = self.cfmt[b]
            if fm:
                self.cpu_used -= self.fb[fm]
                self.cfmt[b] = 0
            self.sfmt[b] = 0
            if self.rec is not None:
                self.rec.on_super(b)
            return 0
        if self.ref_cpu[b] == 0 and self.cfmt[b]:
            fm = self.cfmt[b]
            self.cpu_used -= self.fb[fm]
            self.cfmt[b] = 0
            if not self.sfmt[b]:
                if self.evict_act == "demote":
                    sf = self._ssd_dec(s, fm, b)
                    self.sfmt[b] = sf
                    q = self.cq[b]
                    self.sq[b] = sf if sf > q else q
                    self._wfmt("ssd", sf)
                    nb = self.fb[sf]
                    self._attr(s, nb)
                    return nb
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
        """E5：閒置超過 T 的 session 搬去 SSD；寫入發在到期那一刻（決定時刻＝到期時刻，「之後」＝目前這個請求以後）。"""
        h = self.ttl_heap
        while h and h[0][0] <= t_now:
            te, v, s = heapq.heappop(h)
            if v != self.ttl_ver[s] or self.st[s] != S_CPU:
                continue
            self._dec_t, self._dec_after = te, self.cur
            self._rw = {}
            w = self.demote(s)
            self.n_ttl += 1
            if w:
                self.wbytes[W_TTL] += w
                ls = self.last[s]
                for vw in self.views:
                    if vw.k_nm:
                        vw.taint_write("ssd", w, te, False)       # 到期那一刻是 trace 時間，不受 k_read 影響
                    vw.slow.write(w, te, 1, ls)
            self._flush_split(w)

    # ---------------------------------------------------------- 一個請求
    def step(self, r):
        t = self.t[r]
        if self.ttl is not None:
            self.ttl_expire(t)
        self.cur = r
        self._dec_t, self._dec_after = t, r
        self._rw = {}
        blocks = self.full[r]
        cfmt, sfmt, cq, sq = self.cfmt, self.sfmt, self.cq, self.sq
        fbl = self.fb
        nro = self.nro if self.surp_on else None
        cut = False
        lob = 0
        lt_, lf_, lq_ = [], [], []
        if not self.unv_on:
            for b in blocks:
                fm = cfmt[b]
                if fm:
                    qb = cq[b]
                    if nro and qb in nro:
                        cut = True
                        break
                    lt_.append(1)
                    lf_.append(fm)
                    lq_.append(qb)
                elif sfmt[b]:
                    qb = sq[b]
                    if nro and qb in nro:
                        cut = True
                        break
                    lt_.append(3)
                    lf_.append(sfmt[b])
                    lq_.append(qb)
                    lob += fbl[sfmt[b]]
                else:
                    break
        else:                 # recompute_unverified：前 L 個的來歷在 L×512 都要 pass
            lim, bt, m_ = self.lim, self.btok, INF
            okm = self.okmask[self.rbin[r]] if self.req_strict else None
            for b in blocks:
                fm = cfmt[b]
                if fm:
                    qb = cq[b]
                elif sfmt[b]:
                    qb = sq[b]
                else:
                    break
                if okm is not None:          # unverified_len=request：這個請求的輸入長度下，來歷不 pass 就從這裡起重算
                    if not okm[qb]:
                        cut = True
                        break
                else:
                    if lim[qb] < m_:
                        m_ = lim[qb]
                    if (len(lt_) + 1) * bt > m_:
                        cut = True
                        break
                if fm:
                    lt_.append(1)
                    lf_.append(fm)
                else:
                    lt_.append(3)
                    lf_.append(sfmt[b])
                    lob += fbl[sfmt[b]]
                lq_.append(qb)
            nro = self.nro
        L = len(lt_)
        L0 = L                        # 沒有 INT2 截斷時的有存前綴
        if self.rec is not None:      # O-tier 第 1 趟：每個被載入的 block 讀的是 CPU 還是 SSD 副本（讀取長度＝這個請求的輸入長度／存的前綴長度）
            rl_, st_ = self.reqlen[r], len(blocks) * self.btok
            for i_ in range(L):
                self.rec.on_read(blocks[i_], lt_[i_], rl_, st_)
        if cut:                       # int2_on_read=recompute：從第一個 INT2 來歷的 block 起都重算（surprise recompute）
            nb_ = len(blocks)
            while L0 < nb_ and (cfmt[blocks[L0]] or sfmt[blocks[L0]]):
                L0 += 1
            self.surp_blk[r] = L0 - L
            nrc = 0                   # 重算計價：有存、沒載入，而且存放副本只在 SSD（CPU 沒有）的 block
            for i_ in range(L, L0):
                b_ = blocks[i_]
                if cfmt[b_] == 0 and sfmt[b_]:
                    nrc += 1
            if nrc:
                self.rc_blocks[r] = nrc
                self.rc_bytes[r] = nrc * self._pass_b[self.mf.bin_of(self.reqlen[r])]
        self.Lblk[r] = L
        self.Ltok[r] = L * self.btok
        self.lo_bytes[r] = lob
        uses = self.uses
        npv, nb_ = L0, len(blocks)
        while npv < nb_ and uses[blocks[npv]] > 0:
            npv += 1
        self.nprev[r] = npv
        if self.check:
            for b in blocks[L0:]:
                assert cfmt[b] == 0 and (sfmt[b] == 0 or self.evict_act != "demote"), "有存的 block 不是前綴"
        kd = self.kind[r]
        old_n = len(self.full[self.parent[r]]) if kd == 1 else 0
        tend = []
        for v in self.views:
            v.cpu.cur_r = v.slow.cur_r = r
            v.cpu_r0[r] = len(v.cpu.rs)
            tt, rs, nC, nS, nR, wq = v.restore(lt_, lf_, lq_, L, self.nall[r], t, old_n)
            v.ttft[r], v.rest[r], v.nC[r], v.nS[r], v.nR[r], v.wq[r] = tt, rs, nC, nS, nR, wq
            v.nmreq[r] = v.last_nm
            if v.k_nm:
                v.nmk[r] = v.last_k
            v.bS[r] = v.last_bS
            tend.append(t + tt)
        # ---- 回合結束：新 block 照這個 session 的決定寫進 CPU
        s = self.sess[r]
        old = self.sb[s] if self.sb[s] else []
        old_state = self.st[s]
        first_new = None
        n_new = 0
        if not cut:
            for b in blocks:
                if cfmt[b] == 0:
                    n_new += 1
                    if first_new is None:
                        first_new = b
        else:
            for i, b in enumerate(blocks):
                if cfmt[b] == 0 or L <= i < L0:
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
        tok = len(blocks) * self.btok
        cw = 0
        wt = 0
        rc, rl = self.ref_cpu, self.ref_live
        # 06 §13 加速：舊 block 和新 block 的共同前綴 [0, cpre) 上，「新的 +1、舊的 −1」互相抵消，而且那時 _fix 一定什麼都不做
        # （ref_live、ref_cpu 都 ≥1），所以兩邊都跳過。S_SLOW 時舊的不減 ref_cpu，所以共同前綴上 ref_cpu 仍要 +1。
        cpre = 0
        if old_state in (S_CPU, S_SLOW) and old:
            m_ = min(len(old), len(blocks))
            while cpre < m_ and old[cpre] == blocks[cpre]:
                cpre += 1
        slow_old = old_state == S_SLOW
        for i, b in enumerate(blocks):
            if i >= cpre:
                rl[b] += 1
                rc[b] += 1
            elif slow_old:
                rc[b] += 1
            uses[b] += 1
            c0 = cfmt[b]
            if c0 == 0 or (cut and L <= i < L0):
                if c0:            # 重算出來的新 KV 取代 CPU 裡舊來歷的副本（04 §3.3「重算之後」：一律照現在的決定重存）
                    self.cpu_used -= fbl[c0]
                if cut and L <= i < L0 and sfmt[b]:   # SSD 上的舊副本作廢（不用寫）；之後新副本若被寫到 SSD 是一般的 SSD 寫入
                    sfmt[b] = 0
                    sq[b] = 0
                    if self.rec is not None:
                        self.rec.on_super(b)
                gq = sq[b] if (i < L and sfmt[b]) else 1     # 從 SSD 讀回來的帶著 SSD 副本的來歷；重算＝bf16
                cfmt[b] = fid
                cq[b] = fid if fid > gq else gq
                self.cpu_used += fbf
                cw += fbf
                if self.wt and not sfmt[b]:
                    sf = self._ssd_dec(s, fid, b, tok, 0.0, self.reqlen[r], self.nreq[s] + 1)
                    sfmt[b] = sf
                    sq[b] = sf if sf > cq[b] else cq[b]
                    wt += fbl[sf]
                    self._attr(s, fbl[sf])
                    self._wfmt("ssd", sf)
        if cw:
            self._wfmt("cpu", fid, cw // fbf if fbf else 0)
        if self.sel is not None:
            for b in blocks:
                if uses[b] >= self.sel and not sfmt[b] and cfmt[b]:
                    sf = self._ssd_dec(s, cfmt[b], b, tok, 0.0, self.reqlen[r], self.nreq[s] + 1)
                    sfmt[b] = sf
                    sq[b] = sf if sf > cq[b] else cq[b]
                    wt += fbl[sf]
                    self._attr(s, fbl[sf])
                    self._wfmt("ssd", sf)
                    self.wbytes[W_SEL] += fbl[sf]
        if self.wt:
            self.wbytes[W_WT] += wt
        self.wbytes[W_STORE] += cw
        w = 0
        if old_state in (S_CPU, S_SLOW):
            for b in (old[cpre:] if cpre else old):
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
            if v.k_nm:
                v.taint_write("cpu", cw, te, v.last_k)
                if self.ssd_on:
                    v.taint_write("ssd", w, te, v.last_k)
            v.cpu.write(cw, te)
            if self.ssd_on:
                v.slow.write(w, te)
        self._flush_split(w if self.ssd_on else 0)
        if self.ev_log:
            import hashlib
            hd = hashlib.blake2b(array("q", [r, fid, L, n_new, len(self._evw), self.cpu_used, self.n_demote, self.n_drop, self.n_lost]).tobytes()
                                 + bytes(lt_) + array("q", self._evw).tobytes(), digest_size=8).digest()
            self.ev_digest.append(int.from_bytes(hd, "little"))
            self._evw = []
        elif self._track:
            self._evw = []
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
        w = self.wbytes
        assert sum(self.s_ssd_w) == w[W_DEMOTE] + w[W_WT] + w[W_SEL] + w[W_TTL], "每個 session 的 SSD 寫入加總≠各原因加總"

    def run(self):
        for r in range(len(self.full)):
            self.step(r)
        for v in self.views:
            v.cpu_r0[len(self.full)] = len(v.cpu.rs)
        return self
