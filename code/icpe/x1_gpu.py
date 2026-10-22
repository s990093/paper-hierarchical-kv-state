"""x1_gpu.py — ICPE P0-2（docs/h200/icpe/P0-2_x1_gpu_validation.md）：3b X1「猜誰會回來」的 GPU 驗證。

決策一律用 X1 的 Engine（code/p3x/x1_sim.py，import 不改檔）；這裡只加：
  * HINT-PF：LRU 順序＋打字提示預取（§3；Engine 的子類別，策略名註冊進 x1_sim.POLICIES）
  * 等比例縮小：依 R1 session 抽樣 1/k、CPU 容量 ÷k（§4）
  * GPU 重播：GpuView 把 View.restore 換成真的在 GPU 上還原（harness 的 Restorer＋新一輪 prefill），
    I/O 線仍是 X1 的 Chan（同一套保留公式），經 ChanTier 轉成真實時鐘給 Restorer 等；GPU 閒時虛擬時鐘直接跳到下一個到達（§5）

子命令：
  select    §4：k ∈ {4, 8, 16} 的模擬篩選 → <out>/select.json（只用模擬，在任何 GPU 重播之前）
  selftest  §7 (i)(ii)：模型執行器的重播 ≡ Engine.run（5 個做法、逐請求逐位元）；HINT-PF 在 L_a＝0 時 ≡ LRU；ChanTier 的時間換算
  sim       S_orig／S_real：縮小 trace 上的 X1 模擬，逐請求 TTFT → <out>/sim_*.csv（--times 給 GPU run 的 CSV＝S_real）
  gpu       一個 GPU run（一個容量 × 一個做法 × 一次）→ <out>/gpu_runs.csv（逐請求）；--max-req 給冒煙用
  fullsim   只報告：HINT-PF 在全尺寸 X1 格子（conversation、R1、Cake）的模擬，對 LRU 與 O-PF
環境：P3_CONST=h200、M7_RES=<H200 的 m7_write_policy_h200>（x1_sim.constants() 的檢查）；X1_TRACE_DIR 或 TIARA_DATASETS 指到 Mooncake。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import os
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
REPO = os.path.dirname(CODE)
sys.path.insert(0, os.path.join(CODE, "p3x"))
sys.path.insert(0, CODE)
# 注意：x1_sim 在 import 時 setdefault CUDA_VISIBLE_DEVICES＝""（沒設時藏卡）。Slurm job 裡已經設好，GPU 重播看得到配到的那張卡。

import numpy as np  # noqa: E402

import x1_sim as XS  # noqa: E402
import x1_trace as XT  # noqa: E402
from x1_sim import BF, GiB, INF, S_SLOW, Engine, View  # noqa: E402

POLICIES5 = ["LRU", "FREQ2Q", "O-BEL", "HINT-PF", "O-PF"]
XS.POLICIES["HINT-PF"] = ("lru", "demote", None, "hint")
VIEW = ("ssd", "idle", "cake")
CPU_GIB = (128, 256)
# X3 (a) 主設定（code/p3x/x3_hide.py::leads；H200 的 x3_typing_consts.json）
WPM_DESKTOP, CHARS_PER_WORD, TEMPLATE_OV, TPOT = 51.56, 5, 10, 0.050
CONSTS_X3 = os.path.join(REPO, "results", "p3_explore_h200", "x3", "x3_typing_consts.json")
NMAX_X3 = 256
SLACK = 0.0005                          # = m7_restore_harness.READ_SLACK_S（真實時鐘的背靠背規則）


def run_id():
    return os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc"


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ================================================================== §4 等比例縮小
def keep_session(sid: int, k: int) -> bool:
    return int(hashlib.sha1(str(int(sid)).encode()).hexdigest(), 16) % k == 0


def scale_P(P: dict, k: int) -> dict:
    """依 R1 session 抽樣 1/k：留下 session 通過 keep_session 的所有請求（cont 的父一定在同一個 session）。"""
    if k == 1:
        return P
    idx = [i for i, s in enumerate(P["sessl"]) if keep_session(s, k)]
    old2new = {o: n for n, o in enumerate(idx)}
    R = P["R"]
    par = [old2new[P["parentl"][i]] if P["parentl"][i] >= 0 else -1 for i in idx]
    for n, i in enumerate(idx):          # cont 的父必須也被留下來（同一個 session）
        assert P["parentl"][i] < 0 or P["parentl"][i] in old2new, (i, P["parentl"][i])
    full = [P["full"][i] for i in idx]
    acc = {}
    for ri, ids in enumerate(full):
        for b in ids:
            acc.setdefault(b, []).append(ri)
    t = np.array([P["tl"][i] for i in idx])
    meta = dict(P["meta"], scaled_k=k, n_req=len(idx), n_blocks_full=len(acc), orig_n_req=len(P["full"]))
    Q = dict(rows=[P["rows"][i] for i in idx], meta=meta,
             R=dict(parent=np.array(par), kind=np.array([R["kind"][i] for i in idx], dtype=object),
                    sess=np.array([P["sessl"][i] for i in idx]), n_sess=R["n_sess"], rule=R["rule"]),
             full=full, nall=[P["nall"][i] for i in idx], B=P["B"], acc=acc, t=t)
    Q["kindc"] = [P["kindc"][i] for i in idx]
    Q["sessl"] = [P["sessl"][i] for i in idx]
    Q["parentl"] = par
    Q["tl"] = [float(x) for x in t]
    Q["nmax"] = max(Q["nall"]) + 1
    Q["ws_GiB"] = len(acc) * BF / GiB
    Q["orig_index"] = idx
    return Q


def cap_bytes(cpu_gib: int, k: int) -> int:
    return (cpu_gib * GiB // k) // BF * BF


# ================================================================== §3 HINT-PF
def common_prefix(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def leads(P: dict, K: dict) -> list:
    """每個回來請求（kind＝cont）的 L_a（秒）；其他請求 None。X3 (a) 主設定的寫法，父＝X1 R1 的父。"""
    c = json.load(open(CONSTS_X3))
    cpt = float(c["cpt_llama_English"])
    cps = WPM_DESKTOP * CHARS_PER_WORD / 60.0
    f = XS.f_array(K, NMAX_X3 + 1, "idle")
    rows, par, t = P["rows"], P["parentl"], P["tl"]
    out = [None] * len(rows)
    for r, kd in enumerate(P["kindc"]):
        if kd != 1:
            continue
        p = par[r]
        q, rr = rows[p], rows[r]
        new_raw = rr["input_length"] - q["input_length"] - q["output_length"]
        typing = max(new_raw - TEMPLATE_OV, 0) * cpt / cps
        pp = par[p]
        p_cached = common_prefix(rows[pp]["hash_ids"], q["hash_ids"]) * 512 if pp >= 0 else 0
        gen_prefill = max(0, q["input_length"] - p_cached)
        cp = common_prefix(q["hash_ids"], rr["hash_ids"])
        n_old = min(cp * 512, q["input_length"] + q["output_length"])
        n_blk = min(NMAX_X3, math.ceil(n_old / 512))
        think = max(t[r] - t[p] - gen_prefill / 512.0 * f[n_blk] - q["output_length"] * TPOT, 0.0)
        out[r] = min(typing, think)
    return out


class HintEngine(Engine):
    """HINT-PF（§3）：預取機制同 O-PF；候選＝已收到提示、還沒回來、在慢層的 session（先收到提示的先）；
    騰空間只降級「沒有在等的提示」的 CPU session（LRU）；最後手段丟提示時間最晚的那個 session 的預取 block。"""

    def __init__(self, P, policy, cap, views, K, lead, **kw):
        super().__init__(P, policy, cap, views, K, **kw)
        assert self.pf == "hint"
        self.hint_t = [None] * len(P["full"])
        self.hheap = []
        self.sreq = {}
        for r, L in enumerate(lead):
            self.sreq.setdefault(self.sess[r], []).append(r)
            if L is not None and L > 0:
                h = self.t[r] - L
                self.hint_t[r] = h
                heapq.heappush(self.hheap, (h, r, self.sess[r]))
        self.sptr = {s: 0 for s in self.sreq}

    def _next_req(self, s):
        """session s 在目前請求之後的下一個請求（沒有＝None）。"""
        lst = self.sreq.get(s)
        if not lst:
            return None
        i = self.sptr[s]
        while i < len(lst) and lst[i] <= self.cur:
            i += 1
        self.sptr[s] = i
        return lst[i] if i < len(lst) else None

    def _active_hint(self, s, tau):
        r = self._next_req(s)
        if r is None:
            return None
        h = self.hint_t[r]
        return (h, r) if h is not None and h <= tau else None

    def demote(self, s, now_by_view=None):
        w = super().demote(s, now_by_view)
        tau = self.t[self.cur]
        a = self._active_hint(s, tau)
        if a is not None:                 # 提示已經來了、還沒回來：降級當下重新排進候選
            heapq.heappush(self.hheap, (max(a[0], tau), a[1], s))
        return w

    def unprefetch(self, s):
        super().unprefetch(s)
        tau = self.t[self.cur]
        a = self._active_hint(s, tau)
        if a is not None and self.st[s] == S_SLOW:
            heapq.heappush(self.hheap, (max(a[0], tau), a[1], s))

    def enforce(self):
        w = 0
        while self.cpu_used > self.C:
            s = self.select()
            if s is None:
                if self.pf_sessions:
                    tau = self.t[self.cur]

                    def hk(x):
                        a = self._active_hint(x, INF)
                        return (a[0] if a is not None else INF, -self.last[x])
                    s = max(self.pf_sessions, key=hk)
                    self.unprefetch(s)
                    continue
                raise RuntimeError(f"CPU 超量 {self.cpu_used} > {self.C}，但沒有可以降級的")
            w += self.victim_action(s)
        return w

    def make_room(self, cand, need, now, v):
        if self.C - self.cpu_used >= need:
            return True
        order = sorted((x for x in self.cpu_set if self._active_hint(x, now) is None), key=lambda x: self.last[x])
        w = 0
        for x in order:
            if self.C - self.cpu_used >= need:
                break
            self._order_remove(x)
            self.cpu_set.add(x)
            w += self.demote(x)
        if w:
            v.slow.write(w * self.i4, now)
            self.w_slow_blocks += w
        return self.C - self.cpu_used >= self.i4

    def advance_prefetch(self, t_from, t_to):
        v = self.views[0]
        slow = v.slow
        while self.hheap:
            h, r, s = self.hheap[0]
            st = max(slow.t_free, t_from, h)
            if st >= t_to:
                return
            if r <= self.cur or self.st[s] != S_SLOW or self.pfdone[s] or self._next_req(s) != r:
                heapq.heappop(self.hheap)
                continue
            bl = self.sb[s]
            m = v.m_cpu(len(bl))
            todo = [b for b in reversed(bl[m:]) if self.fmt[b] == 0 and self.on_slow[b]]
            if not todo:
                self.pfdone[s] = True
                heapq.heappop(self.hheap)
                continue
            if not self.make_room(s, len(todo) * self.i4, st, v):
                return
            for b in todo:
                st = max(slow.t_free, t_from, h)
                if st >= t_to:
                    return
                if self.C - self.cpu_used < self.i4:
                    return
                done = slow.read(self.i4, st)
                self.fmt[b] = 2
                self.pf_done[b] = done
                self.pf_owner[b] = s
                self.cpu_used += self.i4
                self.spf[s] += 1
                v.pf_bytes += self.i4
                self.pf_sessions.add(s)
            self.pfdone[s] = True
            heapq.heappop(self.hheap)


def make_engine(P, policy, cap, views, K, lead=None):
    if XS.POLICIES[policy][3] == "hint":
        return HintEngine(P, policy, cap, views, K, lead if lead is not None else leads(P, K))
    return Engine(P, policy, cap, views, K)


# ================================================================== §5 時鐘與 I/O 線的轉接
class Clock:
    """虛擬時間＝真實時間 − base。GPU 閒時 skip_to 直接跳到下一個到達（base 變小）。"""

    def __init__(self, v0: float):
        self.base = time.perf_counter() - v0

    def now(self) -> float:
        return time.perf_counter() - self.base

    def skip_to(self, v: float):
        d = v - self.now()
        if d > 0:
            self.base -= d

    def real(self, v: float) -> float:
        return v + self.base


class ChanTier:
    """給 m7_restore_harness.Restorer 用的「Tier」：保留算在 X1 的 Chan（虛擬時間），回傳真實時鐘的完成時間。
    ready：預取進 CPU 的 block，讀完前不能用（= x1_sim View.restore 的 pf_done）。"""

    def __init__(self, chan, clock: Clock, name: str, ready: float | None = None):
        self.chan, self.clock, self.name, self.ready = chan, clock, name, ready

    def reserve_read(self, nbytes: int, now: float | None = None) -> float:
        c = self.chan
        v = self.clock.now() if now is None else now
        st = c.t_free if 0.0 <= v - c.t_free < SLACK else v
        if self.ready is not None and self.ready > st:
            st = self.ready
        return self.clock.real(c.read(nbytes, st))


# ================================================================== §5 GPU 執行器
class GpuExec:
    def __init__(self, nmax: int, ring: int = 64):
        import torch
        from m10_codec import encode
        from m7_model import CHUNK, KVModel
        from m7_restore_harness import Restorer
        self.torch = torch
        self.CHUNK = CHUNK
        self.km = KVModel(max_len=(nmax + 1) * CHUNK)
        self.rs = Restorer(self.km)
        g = torch.Generator().manual_seed(20261023)
        self.ids = torch.randint(1000, min(120000, self.km.cfg.vocab_size), ((nmax + 1) * CHUNK,), generator=g).to(self.km.device)
        n = min(ring, nmax)
        self.km.prefill_chunked(self.ids[:n * CHUNK])
        torch.cuda.synchronize()
        self.bf, self.i4 = [], []
        for i in range(n):
            v = self.km.chunk_view(i)
            h = self.km.host_chunk()
            h.copy_(v)
            self.bf.append(h)
            e = encode(v, "int4_kivi")
            hp = e.map(lambda t: torch.empty(t.shape, dtype=t.dtype, pin_memory=True))
            hp.copy_from(e)
            self.i4.append(hp)
        torch.cuda.synchronize()
        self.R = n
        self.i4_bytes = self.i4[0].nbytes
        self.clock = None
        self.cur_tokens = 0              # 目前請求的 input_length（GpuView.restore 用）
        self.last = {}

    def d2h(self, positions: list) -> float:
        """回合結束：新進 CPU 的 block 真的 D2H（寫進 ring；內容不重要，複製是真的）。"""
        torch = self.torch
        t0 = time.perf_counter()
        for j, i in enumerate(positions):
            self.bf[j % self.R].copy_(self.km.chunk_view(i), non_blocking=True)
        torch.cuda.synchronize()
        return time.perf_counter() - t0


class GpuView(View):
    """View.restore 換成 GPU 上真的還原＋新一輪 prefill。I/O 線＝self.cpu／self.slow（Chan），由 ChanTier 轉給 Restorer。"""

    def __init__(self, K, ex: GpuExec, nmax, nreq):
        super().__init__(K, *VIEW, nmax, nreq)
        self.ex = ex

    def restore(self, lay, blocks, L, n_all, t0, old_n, pf_done):
        ex = self.ex
        clk = ex.clock
        tc = ChanTier(self.cpu, clk, "cpu")
        tsl = ChanTier(self.slow, clk, "ssd_local")
        host, loc = [], []
        nC = nS = 0
        for i in range(L):
            k = lay[i]
            if k == 1:
                host.append(ex.bf[i % ex.R]); loc.append(tc); nC += 1
            elif k == 2:
                host.append(ex.i4[i % ex.R]); loc.append(ChanTier(self.cpu, clk, "cpu", ready=pf_done[blocks[i]])); nC += 1
            else:
                host.append(ex.i4[i % ex.R]); loc.append(tsl); nS += 1
        C = ex.CHUNK
        tstart = time.perf_counter()
        if L:
            rec = ex.rs.restore(ex.ids, L, host, loc, "cake")
        else:
            rec = {"n_recompute": 0, "t_restore": 0.0, "n_load_cpu": 0, "n_load_ssd": 0, "n_copy_bound": 0, "t_dequant": 0.0}
        t_rest = time.perf_counter() - tstart
        ntok = ex.cur_tokens
        t_new = 0.0
        if ntok > L * C:
            t_new, _ = ex.rs.new_turn(ex.ids[L * C:ntok], L * C)
        ttft = time.perf_counter() - tstart
        meet = int(rec.get("n_recompute", 0))
        ex.last = dict(t_restore=t_rest, t_new=t_new, meet=meet, n_load_cpu=int(rec.get("n_load_cpu", 0)),
                       n_load_ssd=int(rec.get("n_load_ssd", 0)), n_copy_bound=int(rec.get("n_copy_bound", 0)),
                       t_dequant=float(rec.get("t_dequant", 0.0)), zero_new=int(ntok <= L * C))
        nR = meet + (n_all - L)
        return ttft, t_rest, int(rec.get("n_load_cpu", 0)), int(rec.get("n_load_ssd", 0)), nR


# ================================================================== 重播迴圈
def replay(P, policy, cap, K, ex=None, model_view=False, max_req=None, on_row=None, lead=None):
    """ex＝None 且 model_view：用模型時間（View.restore）跑同一個迴圈（selftest）。回傳 (engine, view, rows)。"""
    n = len(P["full"]) if max_req is None else min(max_req, len(P["full"]))
    if ex is None:
        view = View(K, *VIEW, P["nmax"], len(P["full"]))
    else:
        view = GpuView(K, ex, P["nmax"], len(P["full"]))
    eng = make_engine(P, policy, cap, [view], K, lead)
    eng.t = list(P["tl"])                      # 不改到 P（開始時間會換成實際的）
    if ex is not None:
        ex.clock = Clock(P["tl"][0])
    rows = []
    for r in range(n):
        a = P["tl"][r]
        if ex is not None:
            ex.clock.skip_to(a)
            start = ex.clock.now()
            eng.t[r] = start
            blocks = P["full"][r]
            pre = [eng.fmt[b] for b in blocks]
            ex.cur_tokens = P["rows"][r]["input_length"]
        else:
            start = a
        eng.step(r)
        row = dict(r=r, orig_r=P.get("orig_index", list(range(len(P["full"]))))[r], session=P["sessl"][r],
                   kind=P["kindc"][r], t_arr=a, t_start=start, queue_s=start - a, ttft=float(view.ttft[r]),
                   n_cpu=int(view.nC[r]), n_slow=int(view.nS[r]), n_recompute=int(view.nR[r]))
        if ex is not None:
            post = [eng.fmt[b] for b in blocks]
            newcpu = [i for i, (x, y) in enumerate(zip(pre, post)) if x != 1 and y == 1]
            row.update(ex.last, d2h_blocks=len(newcpu), d2h_s=ex.d2h(newcpu) if newcpu else 0.0)
        rows.append(row)
        if on_row:
            on_row(row)
    return eng, view, rows


def load_P(k: int):
    P = XS.prep("conversation", "R1")
    return scale_P(P, k)


# ================================================================== 子命令
def cmd_select(a):
    K = XS.constants()
    out = dict(run_id=run_id(), ts=ts(), rule="P0-2 §4：最小的 k 使 ρ̂ ≤ 0.30 且 O-PF 對 LRU 的 headroom 在 128/k 或 256/k ≥ 15%",
               candidates={})
    chosen = None
    for k in (4, 8, 16):
        P = load_P(k)
        span = P["tl"][-1] - P["tl"][0]
        c = dict(n_req=len(P["full"]), n_ret=int(sum(1 for x in P["kindc"] if x == 1)), span_s=span, cells={})
        for g in CPU_GIB:
            res = {}
            for pol in ("LRU", "O-PF"):
                _, v, _ = replay(P, pol, cap_bytes(g, k), K, model_view=True)
                ret = np.array(P["kindc"]) == 1
                res[pol] = dict(mean_ret=float(v.ttft[ret].mean()), sum_all=float(v.ttft.sum()))
            hr = (res["LRU"]["mean_ret"] - res["O-PF"]["mean_ret"]) / res["LRU"]["mean_ret"]
            c["cells"][g] = dict(cap_bytes=cap_bytes(g, k), **{f"{p}_mean_ret": res[p]["mean_ret"] for p in res},
                                 headroom_OPF_vs_LRU=hr, rho_hat=res["LRU"]["sum_all"] / span)
        c["rho_hat"] = c["cells"][128]["rho_hat"]
        c["ok_rho"] = c["rho_hat"] <= 0.30
        c["ok_headroom"] = any(c["cells"][g]["headroom_OPF_vs_LRU"] >= 0.15 for g in CPU_GIB)
        out["candidates"][k] = c
        print(k, json.dumps(c, ensure_ascii=False), flush=True)
        if chosen is None and c["ok_rho"] and c["ok_headroom"]:
            chosen = k
    out["chosen_k"] = chosen if chosen is not None else 16
    out["fallback"] = chosen is None
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "select.json"), "w"), ensure_ascii=False, indent=1)
    print("chosen k =", out["chosen_k"], "fallback" if out["fallback"] else "")


def cmd_selftest(a):
    K = XS.constants()
    res = []

    def rec(name, ok, **info):
        res.append(dict(run_id=run_id(), ts=ts(), check=name, passed=bool(ok), **{k: json.dumps(v) for k, v in info.items()}))
        print(("PASS " if ok else "FAIL ") + name, info, flush=True)

    P = load_P(a.k)
    cap = cap_bytes(CPU_GIB[0], a.k)
    for pol in POLICIES5:
        v1 = View(K, *VIEW, P["nmax"], len(P["full"]))
        e1 = make_engine(P, pol, cap, [v1], K)
        e1.run()
        _, v2, rows = replay(P, pol, cap, K, model_view=True)
        same = bool(np.array_equal(v1.ttft, v2.ttft)) and v1.slow.bytes_r == v2.slow.bytes_r and v1.pf_bytes == v2.pf_bytes
        rec(f"replay_eq_engine_run[{pol}]", same, n=len(rows), mean_ret=float(v2.ttft[np.array(P['kindc']) == 1].mean()))
    lead0 = [0.0 if x == 1 else None for x in P["kindc"]]
    _, vh, _ = replay(P, "HINT-PF", cap, K, model_view=True, lead=lead0)
    _, vl, _ = replay(P, "LRU", cap, K, model_view=True)
    rec("hint_lead0_eq_lru", bool(np.array_equal(vh.ttft, vl.ttft)), pf_bytes=vh.pf_bytes)
    L = leads(P, K)
    la = np.array([x for x in L if x is not None])
    rec("hint_leads_finite_nonneg", bool(len(la) and np.isfinite(la).all() and (la >= 0).all()), n=len(la),
        median=float(np.median(la)), p10=float(np.percentile(la, 10)), p90=float(np.percentile(la, 90)))
    _, vh2, _ = replay(P, "HINT-PF", cap, K, model_view=True)
    rec("hint_prefetches_something", vh2.pf_bytes > 0, pf_GiB=vh2.pf_bytes / GiB)
    # ChanTier：同一個 Chan，經轉接（虛擬時鐘）和直接呼叫的結果相同
    c1, c2 = XS.Chan("x", 6.98115234375, 1.91, 3.306), XS.Chan("x", 6.98115234375, 1.91, 3.306)
    clk = Clock(100.0)
    tier = ChanTier(c1, clk, "ssd")
    v_now = clk.now()
    c1.write(5 * BF, v_now); c2.write(5 * BF, v_now)
    d_real = tier.reserve_read(K["int4_bytes"], now=v_now)
    d_dir = c2.read(K["int4_bytes"], v_now)
    rec("chantier_matches_chan", abs((d_real - clk.base) - d_dir) < 1e-9, diff=(d_real - clk.base) - d_dir)
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "selftest.csv")
    new = not os.path.exists(p)
    with open(p, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0].keys()) + ["n", "mean_ret", "pf_bytes", "median", "p10", "p90", "pf_GiB", "diff"],
                           extrasaction="ignore")
        if new:
            w.writeheader()
        for r in res:
            w.writerow(r)
    if not all(r["passed"] for r in res):
        sys.exit(1)


SIM_FIELDS = ["run_id", "ts", "ref", "k", "cpu_GiB", "cap_bytes", "policy", "src_gpu_run", "r", "orig_r", "session", "kind",
              "t_start", "ttft", "n_cpu", "n_slow", "n_recompute"]


def cmd_sim(a):
    K = XS.constants()
    P = load_P(a.k)
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "sim_requests.csv")
    new = not os.path.exists(p)
    fh = open(p, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=SIM_FIELDS, extrasaction="ignore")
    if new:
        w.writeheader()
    if a.times:                                     # S_real：每個 GPU run 的實際開始時間
        import pandas as pd
        g = pd.read_csv(a.times)
        L0 = leads(P, K)                            # 提示時間＝原始到達 − L_a（和 GPU run 相同），所以 L' = L_a + (實際開始 − 原始到達)
        for rid, gg in g.groupby("run_id"):
            gg = gg.sort_values("r")
            assert len(gg) == len(P["full"]), (rid, len(gg), len(P["full"]))
            pol, cg = gg.policy.iloc[0], int(gg.cpu_GiB.iloc[0])
            Q = dict(P)
            Q["tl"] = [float(x) for x in gg.t_start.values]
            Q["t"] = np.array(Q["tl"])
            lead = [None if x is None else x + (q - o) for x, q, o in zip(L0, Q["tl"], P["tl"])]
            _, v, rows = replay(Q, pol, cap_bytes(cg, a.k), K, model_view=True, lead=lead)
            for row in rows:
                w.writerow(dict(run_id=run_id(), ts=ts(), ref="S_real", k=a.k, cpu_GiB=cg, cap_bytes=cap_bytes(cg, a.k), policy=pol,
                                src_gpu_run=rid, **row))
            print("S_real", rid, pol, cg, float(v.ttft[np.array(P["kindc"]) == 1].mean()), flush=True)
    else:
        for cg in a.cpu:
            for pol in a.policies:
                _, v, rows = replay(P, pol, cap_bytes(cg, a.k), K, model_view=True)
                for row in rows:
                    w.writerow(dict(run_id=run_id(), ts=ts(), ref="S_orig", k=a.k, cpu_GiB=cg, cap_bytes=cap_bytes(cg, a.k),
                                    policy=pol, src_gpu_run="", **row))
                print("S_orig", pol, cg, float(v.ttft[np.array(P["kindc"]) == 1].mean()), flush=True)
    fh.close()


GPU_FIELDS = ["run_id", "ts", "k", "cpu_GiB", "cap_bytes", "policy", "rep", "r", "orig_r", "session", "kind", "t_arr", "t_start",
              "queue_s", "ttft", "t_restore", "t_new", "meet", "n_cpu", "n_slow", "n_recompute", "n_load_cpu", "n_load_ssd",
              "n_copy_bound", "t_dequant", "zero_new", "d2h_blocks", "d2h_s", "contaminated"]


def cmd_gpu(a):
    import torch  # noqa: F401
    from gpu_guard import GpuWatcher, smi_index
    K = XS.constants()
    P = load_P(a.k)
    os.makedirs(a.out, exist_ok=True)
    rdir = os.path.join(os.environ.get("TIARA_RUNS", "."), run_id())
    p = os.path.join(a.out, a.csv)
    new = not os.path.exists(p)
    fh = open(p, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=GPU_FIELDS, extrasaction="ignore")
    if new:
        w.writeheader()
    with GpuWatcher(gpu=smi_index(0), own_root=os.getpid(), out_path=os.path.join(a.out, f"gpu_guard_{run_id()}.json")) as wat:
        ex = GpuExec(P["nmax"])
        assert ex.i4_bytes == K["int4_bytes"], (ex.i4_bytes, K["int4_bytes"])
        # 暖機（不記）：前 8 個請求用 LRU 在一份複本上跑一次
        replay(P, "LRU", cap_bytes(a.cpu, a.k), K, ex=ex, max_req=8)
        cap = cap_bytes(a.cpu, a.k)
        t0 = time.time()

        def on_row(row):
            w.writerow(dict(run_id=run_id(), ts=ts(), k=a.k, cpu_GiB=a.cpu, cap_bytes=cap, policy=a.policy, rep=a.rep,
                            contaminated=int(wat.contaminated), **row))
            if row["r"] % 100 == 0:
                fh.flush()
                print(f"[{row['r']}/{len(P['full'])} {time.time() - t0:.0f}s] ttft={row['ttft']:.3f} queue={row['queue_s']:.3f}", flush=True)
        replay(P, a.policy, cap, K, ex=ex, max_req=a.max_req, on_row=on_row)
    fh.close()
    if wat.contaminated:
        os.makedirs(rdir, exist_ok=True)
        open(os.path.join(rdir, "CONTAMINATED"), "w").write(json.dumps(sorted(wat.intruders)) + "\n")
        print("[x1_gpu] CONTAMINATED：這次的時間不可用", flush=True)
        sys.exit(3)


def cmd_fullsim(a):
    """只報告：HINT-PF 全尺寸（conversation、R1、Cake），對 LRU 與 O-PF；寫 <out>/fullsim_hint.csv。"""
    K = XS.constants()
    P = XS.prep("conversation", "R1")
    L = leads(P, K)
    out = []
    ret = np.array(P["kindc"]) == 1
    for cg in XS.CPU_GIB:
        for slow in XS.SLOWS:
            for gpu in XS.GPUS:
                res = {}
                for pol in ("LRU", "HINT-PF", "O-PF"):
                    v = View(K, slow, gpu, "cake", P["nmax"], len(P["full"]))
                    make_engine(P, pol, cg * GiB, [v], K, L).run()
                    res[pol] = float(v.ttft[ret].mean())
                gain_h = (res["LRU"] - res["HINT-PF"]) / res["LRU"]
                gain_o = (res["LRU"] - res["O-PF"]) / res["LRU"]
                out.append(dict(run_id=run_id(), ts=ts(), trace="conversation", rule="R1", cpu_GiB=cg, slow=slow, gpu=gpu,
                                restore="cake", LRU=res["LRU"], HINT_PF=res["HINT-PF"], O_PF=res["O-PF"],
                                hint_vs_lru=gain_h, opf_vs_lru=gain_o, hint_share_of_opf=(gain_h / gain_o) if gain_o > 0 else None))
                print(json.dumps(out[-1], ensure_ascii=False), flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "fullsim_hint.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("select"); s.add_argument("--out", required=True)
    s = sp.add_parser("selftest"); s.add_argument("--out", required=True); s.add_argument("--k", type=int, default=16)
    s = sp.add_parser("sim"); s.add_argument("--out", required=True); s.add_argument("--k", type=int, required=True)
    s.add_argument("--cpu", type=int, nargs="+", default=list(CPU_GIB)); s.add_argument("--policies", nargs="+", default=POLICIES5)
    s.add_argument("--times", help="GPU run 的逐請求 CSV（給了就算 S_real）")
    s = sp.add_parser("gpu"); s.add_argument("--out", required=True); s.add_argument("--k", type=int, required=True)
    s.add_argument("--cpu", type=int, required=True, choices=CPU_GIB); s.add_argument("--policy", required=True, choices=POLICIES5)
    s.add_argument("--rep", type=int, default=0); s.add_argument("--max-req", type=int); s.add_argument("--csv", default="gpu_requests.csv")
    s = sp.add_parser("fullsim"); s.add_argument("--out", required=True)
    a = ap.parse_args()
    {"select": cmd_select, "selftest": cmd_selftest, "sim": cmd_sim, "gpu": cmd_gpu, "fullsim": cmd_fullsim}[a.cmd](a)


if __name__ == "__main__":
    main()
