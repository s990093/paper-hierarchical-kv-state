"""m8_conc_sim.py — D1（H3）：多人同時用、寫入排隊時，延後版會不會輸？〔模擬，不碰 GPU〕

判準與模型定義寫在 docs/research_20261010_directions/D1_concurrency.md §2（開跑前寫死）。
重用 m7_sim／m7_write_policy 的 Tier、BState、make_workload、f(i)；不改它們。

新加的東西：
  1. 併發：N 個 session，poisson／bursty 到達，回來前有思考時間；一張 GPU，FIFO 排隊。
  2. drop 模型（release="drop"）：寫往 SSD 中的 chunk 鎖在 CPU、不能淘汰；搬移要等 SSD 寫完才空出格子；
     放不下就不等，照溢出規則（whole／tail／pa）丟新 chunk。SSD 寫入佇列可設上限 qmax。
  3. 還原修正：版面有「中間或尾巴的洞」時，不再整段重算（m7_sim.sim_restore 的 lstop），
     改成在「原本的貪婪 Cake」和「GPU 算前 m 段＋I/O 依序載入 m 之後有存的＋洞等前面都到再算」之間取最小。
     沒有這種洞時直接用 m7_sim.sim_restore，所以 concurrency=1 能逐格重現舊結果。

子命令：
  validate   D1 §2.6 的 6 格（serial、free/hold），對 r1_sim_llama31_8b.csv
  sweep      D1 §2.4 主掃描（drop 模型），多行程
  analyze    D1 §2.5 判定
  probe      §2.7 追加檢查（qmax 等），只報告
"""
from __future__ import annotations

import argparse
import bisect
import heapq
import itertools
import os
import random
import sys
from collections import deque
from datetime import datetime, timezone

os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m7_restore_harness import GiB, write_boundary  # noqa: E402
from m7_sim import CB, D2H_BPS, _Pool, mk_tier, sim_restore  # noqa: E402
from m7_write_policy import SCHED, BState, load_f, load_params, make_workload  # noqa: E402

OUT = os.path.join(HERE, "..", "results", "m8_directions")
RID = os.environ.get("RUN_ID", "adhoc")


def now_ts():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ================================================================== 還原（修正版）
def _rd(T, x):
    """T 這一層從時間 x 開始讀一個 chunk 要多久（不改狀態；和 Tier.reserve_read 同一個公式）。"""
    d = T.c_s + (CB / T.read_Bps if T.read_Bps else 0.0)
    if T.mode == "share" and T.t_free_w > x:
        d *= T.k_read
    return d


def _greedy_pure(n, loc, f, t0):
    """原本的 m7_sim.sim_restore（cake）不改狀態的版本：回傳 (結束, 重算數, 載入清單[(tier, 起始now)])。"""
    tf = {}
    p, q, rt, lt, lstop = 0, n - 1, t0, t0, False
    loads = []
    while p <= q:
        if lstop or rt <= lt:
            rt += f[p]
            p += 1
        else:
            T = loc[q]
            if T is None:
                lstop = True
                continue
            st = max(lt, tf.get(T.name, T.t_free))
            loads.append((T, lt))
            lt = st + _rd(T, st)
            tf[T.name] = lt
            q -= 1
    return max(rt, lt), p, loads


def _msched_pure(n, loc, f, t0, m, pref):
    """GPU 先算 [0,m)；I/O 依位置順序載入 [m,n) 裡有存的；[m,n) 的洞要等它前面的都到了才算。"""
    tf = {}
    lt = t0
    loads = []
    gpu = t0 + pref[m]
    last = t0
    rec = m
    for i in range(m, n):
        T = loc[i]
        if T is not None:
            st = max(lt, tf.get(T.name, T.t_free))
            loads.append((T, lt))
            lt = st + _rd(T, st)
            tf[T.name] = lt
            last = lt
        else:
            gpu = max(gpu, last) + f[i]
            rec += 1
    return max(gpu, lt), rec, loads


def restore_v2(n, loc, f, t0, mode="cake"):
    """回傳 (完成時間, 重算 chunk 數, n_load_cpu, n_load_ssd, kind)。會改動 tier 的讀取時鐘（和 sim_restore 一樣）。"""
    if n == 0:
        return t0, 0, 0, 0, "none"
    first = next((i for i in range(n) if loc[i] is not None), None)
    if first is None:
        return t0 + sum(f[:n]), n, 0, 0, "compute"
    if mode != "cake" or all(loc[i] is not None for i in range(first, n)):
        t, meet, nc, ns = sim_restore(n, loc, f, t0, mode)
        return t, meet, nc, ns, "cake"
    pref = [0.0]
    for x in f[:n]:
        pref.append(pref[-1] + x)
    best = _greedy_pure(n, loc, f, t0)
    kind = "greedy"
    for m in range(0, n + 1):
        r = _msched_pure(n, loc, f, t0, m, pref)
        if r[0] < best[0] - 1e-12:
            best, kind = r, f"msched{m}"
    end, rec, loads = best
    nc = ns = 0
    for T, at in loads:   # 照選好的順序真的預約讀取（讓之後的請求看到同一個 tier 狀態）
        T.reserve_read(CB, at)
        if T.name.startswith("cpu"):
            nc += 1
        else:
            ns += 1
    return end, rec, nc, ns, kind


# ================================================================== drop 模型的狀態
DISCARD_ALWAYS = {"S1", "S2b", "S4PD"}
DEMOTE_ALWAYS = {"S4", "S4+", "S4B", "S4W", "S4C", "S5L"}
POS_RULE = {"S4BT", "S4WT", "S5T", "S4WT50"}            # i<b_s 丟、i≥b_s 搬
# D1 §2.8 追加 1（事後）：S5T「前段不存」規則的延後版：先丟任何 session 的 i<b_c；i<max(b_c,b_s) 丟、否則搬
POS_RULE_D = {"S4BD", "S4WD", "S1D", "S4WD50", "S4CT", "S4CT50"}
# D1 §2.11 追加 4（事後、探索）：只挑尾段的預先清理
PRECLEAN_T = {"S4CT": 0.25, "S4CT50": 0.5}
# D1 §2.10 追加 3（事後、探索）：背景版水位線 50%
WATERMARK = {"S4W": 0.25, "S4WT": 0.25, "S4WD": 0.25, "S4WD50": 0.5, "S4WT50": 0.5}
ORDER = {"S1": "lru", "S4": "lru", "S5L": "lru", "S4BT": "lru", "S4WT": "lru", "S5T": "lru",
         "S2b": "cheap", "S4+": "cheap", "S4C": "cheap", "S4B": "lazyb", "S4W": "lazyb", "S4PD": "pensieve",
         "S4BD": "lazyb", "S4WD": "lazyb", "S1D": "lazyb", "S4WD50": "lazyb", "S4WT50": "lru",
         "S4CT": "lazyb", "S4CT50": "lazyb"}
POSTHOC = ["S4BD", "S4WD", "S1D"]
STRATS = ["S1", "S2b", "S4", "S4+", "S4B", "S4W", "S4C", "S4PD", "S4BT", "S4WT", "S5T", "S5L"]
TWINS = ["S1", "S2b", "S4", "S4+", "S4B", "S4W", "S4C", "S4PD", "S4BT", "S4WT"]
CAND, CAND_OV = "S5T", "pa"
OVERFLOWS = ["whole", "tail", "pa"]


class CState(BState):
    """release='drop'：寫往 SSD 中的 chunk 鎖在 CPU；搬移寫完才空出；放不下照溢出規則丟新 chunk，不等。"""

    def __init__(self, strategy, cpu_cap, tiers, f, P, overflow="pa", qmax=None, admit_first=False):
        assert strategy in ORDER, strategy
        self.admit_first = admit_first   # D1 §2.9 追加 2：新 chunk 也算淘汰候選（m7 BState 的「先放再淘汰」語意）
        super().__init__(strategy, cpu_cap, None, tiers, f, _Pool(), P, release="drop")
        self.overflow = overflow
        self.qmax = qmax
        self.wq = deque()          # SSD 寫入的完成時間（FIFO 通道，單調）
        self.stage = []            # heap (done, key)：寫完才釋放 CPU 格子的 chunk
        self.n_stage = 0
        self._bc, self._bs = {}, {}
        self.S = dict(new=0, policy_skip=0, overflow_drop=0, overflow_events=0, discard=0, demote=0, ssd_w=0,
                      ssd_rej=0, wt_skip=0, evict_ssdcopy=0, discard_new=0)

    # --- 分界（快取） ---
    def bc(self, n):
        if n not in self._bc:
            self._bc[n] = write_boundary(n, self.f, self.ell["cpu"])
        return self._bc[n]

    def bs(self, n):
        if n not in self._bs:
            self._bs[n] = write_boundary(n, self.f, self.ell["ssd"])
        return self._bs[n]

    # --- SSD 寫入佇列 ---
    def outstanding(self, now):
        while self.wq and self.wq[0] <= now:
            self.wq.popleft()
        return len(self.wq)

    def can_write(self, now):
        ok = self.qmax is None or self.outstanding(now) < self.qmax
        if not ok:
            self.S["ssd_rej"] += 1
        return ok

    def submit_ssd(self, key, now):
        c = self.c[key]
        done = self.t["ssd"].reserve_write(CB, now)
        self.wq.append(done)
        c["ssd"], c["done"] = True, done
        self.ssd_used += 1
        self.w_bytes["ssd"] += CB
        self.S["ssd_w"] += 1
        return done

    def release_done(self, now):
        while self.stage and self.stage[0][0] <= now:
            _, key = heapq.heappop(self.stage)
            c = self.c.get(key)
            if c is not None and c["stage"]:
                c["stage"] = False
                c["cpu"] = False
                self.cpu_used -= 1
                self.n_stage -= 1

    def locked(self, c, now):
        return c["stage"] or (c["ssd"] and c["done"] > now)

    # --- 淘汰 ---
    def victims(self, now, ev_idx, include_locked=False, extra=()):
        ks = [k for k, c in self.c.items() if c["cpu"] and (include_locked or not self.locked(c, now))] + list(extra)
        o = ORDER[self.st]
        if o == "lru":
            key = lambda k: (self.last[k[0]], k[1])  # noqa: E731
        elif o == "cheap":
            key = lambda k: (k[1], self.last[k[0]])  # noqa: E731
        elif o == "lazyb":
            bb = {s: self.bc(n) for s, n in self.hist.items()}
            key = lambda k: (0 if k[1] < bb[k[0]] else 1, self.last[k[0]], k[1])  # noqa: E731
        else:   # pensieve：V = f(i)/T
            def key(k):
                T = max(1, ev_idx - self.idle_since.get(k[0], ev_idx) + 1)
                return (self.f[k[1]] / T, k[1])
        ks.sort(key=key)
        return ks

    def action(self, k):
        if self.st in DISCARD_ALWAYS:
            return "discard"
        if self.st in DEMOTE_ALWAYS:
            return "demote"
        n = self.hist[k[0]]
        if self.st in POS_RULE_D:
            return "discard" if k[1] < max(self.bc(n), self.bs(n)) else "demote"
        return "discard" if k[1] < self.bs(n) else "demote"

    def _kill(self, k):
        c = self.c[k]
        if c["cpu"]:
            self.cpu_used -= 1
        if c["ssd"]:
            self.ssd_used -= 1
        del self.c[k]

    def act_on(self, k, now):
        """對一個沒鎖住的 CPU chunk 動手。回傳 'free'（立刻空出）、'stage'（之後才空）、'skip'。"""
        c = self.c[k]
        if c["ssd"] and c["done"] <= now:      # 有完成的 SSD 副本：丟 CPU 那份
            c["cpu"] = False
            self.cpu_used -= 1
            self.S["evict_ssdcopy"] += 1
            return "free"
        if self.action(k) == "discard":
            self._kill(k)
            self.S["discard"] += 1
            return "free"
        if not self.can_write(now):
            return "skip"
        done = self.submit_ssd(k, now)
        c["stage"] = True
        self.n_stage += 1
        heapq.heappush(self.stage, (done, k))
        self.S["demote"] += 1
        self.n_demote += 1
        return "stage"

    def free_now(self, k_need, now, ev_idx):
        freed = pend = 0
        for k in self.victims(now, ev_idx):
            if freed >= k_need:
                break
            c = self.c[k]
            instant = (c["ssd"] and c["done"] <= now) or self.action(k) == "discard"
            if not instant and pend >= k_need - freed:
                continue       # 已經送出夠多搬移，不要整個 CPU 都搬
            r = self.act_on(k, now)
            if r == "free":
                freed += 1
            elif r == "stage":
                pend += 1
        return freed

    # --- 寫入時 ---
    def placement(self, i, n_new):
        S = self.st
        if S in ("S1", "S2b"):
            return "wt"
        if S == "S5L":
            return "direct" if i < self.bc(n_new) else "cpu"
        if S == "S5T":
            if i < self.bc(n_new):
                return "skip"
            return "wt" if i >= self.bs(n_new) else "cpu"
        if S == "S1D":
            return "wt" if i >= self.bs(n_new) else "cpu"
        return "cpu"

    def after_round_drop(self, s, n_old, n_new, now, ev_idx):
        self.release_done(now)
        self.last[s] = ev_idx
        self.idle_since[s] = ev_idx
        self.hist[s] = n_new
        want = []
        for i in range(n_old, n_new):
            self.S["new"] += 1
            kind = self.placement(i, n_new)
            if kind == "skip":
                self.S["policy_skip"] += 1
            else:
                want.append((i, kind))
        need = len(want)
        free = self.cpu_cap - self.cpu_used
        if need > free and self.admit_first:      # 追加 2：新 chunk 也是候選，同一個淘汰順序
            newk = {(s, i) for i, _ in want}
            deficit, freed, pend, gone = need - free, 0, 0, set()
            for k in self.victims(now, ev_idx, extra=sorted(newk)):
                if freed >= deficit:
                    break
                if k in newk:
                    if self.action(k) == "discard":
                        gone.add(k)
                        freed += 1
                        self.S["discard_new"] += 1
                    continue
                c = self.c[k]
                instant = (c["ssd"] and c["done"] <= now) or self.action(k) == "discard"
                if not instant and pend >= deficit - freed:
                    continue
                r = self.act_on(k, now)
                if r == "free":
                    freed += 1
                elif r == "stage":
                    pend += 1
            want = [w for w in want if (s, w[0]) not in gone]
            need = len(want)
            free = self.cpu_cap - self.cpu_used
        if need > free:
            free += self.free_now(need - free, now, ev_idx)
        admit = max(0, min(need, free))
        if admit < need:
            self.S["overflow_events"] += 1
            self.S["overflow_drop"] += need - admit
            if self.overflow == "whole":
                want = []
            elif self.overflow == "tail":
                want = want[:admit]
            elif self.overflow == "pa":
                want = want[need - admit:]
            else:
                raise ValueError(self.overflow)
        for i, kind in want:
            key = (s, i)
            self.c[key] = {"host": None, "cpu": True, "ssd": False, "done": 0.0, "stage": False}
            self.cpu_used += 1
            self.t["cpu"].reserve_write(CB, now)
            self.w_bytes["cpu"] += CB
            if kind in ("wt", "direct"):
                if self.can_write(now):
                    done = self.submit_ssd(key, now)
                    if kind == "direct":
                        self.c[key]["stage"] = True
                        self.n_stage += 1
                        heapq.heappush(self.stage, (done, key))
                else:
                    self.S["wt_skip"] += 1
        assert self.cpu_used <= self.cpu_cap, (self.cpu_used, self.cpu_cap)
        S = self.st
        if S in WATERMARK:     # 背景：讓 CPU（扣掉之後會空出的暫存格）至少空 25%（追加 3 的 *50：50%）
            lo = self.cpu_cap - int(WATERMARK[S] * self.cpu_cap)
            if self.cpu_used - self.n_stage > lo:
                for k in self.victims(now, ev_idx):
                    if self.cpu_used - self.n_stage <= lo:
                        break
                    self.act_on(k, now)
        if S in PRECLEAN_T:          # 追加 4：淘汰順序最前面 x×cap 個裡，尾段（i ≥ max(b_c,b_s)）沒副本的先寫一份
            kk = int(PRECLEAN_T[S] * self.cpu_cap)
            for k in self.victims(now, ev_idx, include_locked=True)[:kk]:
                n = self.hist[k[0]]
                if k[1] >= max(self.bc(n), self.bs(n)) and not self.c[k]["ssd"] and self.can_write(now):
                    self.submit_ssd(k, now)
        if S == "S4C":               # 預先清理：淘汰順序最前面 25%×cap 個先寫一份到 SSD（複製、不移走）
            kk = int(0.25 * self.cpu_cap)
            for k in self.victims(now, ev_idx, include_locked=True)[:kk]:
                if not self.c[k]["ssd"] and self.can_write(now):
                    self.submit_ssd(k, now)

    def layout_drop(self, s, n, now):
        self.release_done(now)
        loc = []
        for i in range(n):
            c = self.c.get((s, i))
            if c is None:
                loc.append(None)
            elif c["cpu"]:
                loc.append(self.t["cpu"])
            elif c["ssd"] and c["done"] <= now:
                loc.append(self.t["ssd"])
            else:
                raise RuntimeError(f"chunk {(s, i)} 不在 CPU、SSD 也還沒寫完：不變量被破壞")
        return loc


# ================================================================== 一次模擬
def arrivals(seed, n_sess, lam, arrival, burst=8):
    rng = random.Random(1_000_003 * (seed + 1) + 7)
    t, out = 100.0, []
    if arrival == "poisson":
        for s in range(n_sess):
            t += rng.expovariate(lam)
            out.append(t)
    elif arrival == "bursty":
        for g in range(0, n_sess, burst):
            t += rng.expovariate(lam / burst)
            out += [t] * min(burst, n_sess - g)
    else:
        raise ValueError(arrival)
    return out, rng


def run(cfg, P, f):
    """cfg: dict(strategy, overflow, release, arrival, lam, think, n_sess, cpu_gibps, ssd_dev, workload, cpu_frac,
    qmax, seed, ov, gap)。回傳 (summary dict, rows)。"""
    sched = SCHED[cfg["workload"]]
    n_sess = cfg["n_sess"]
    tiers = {"cpu": mk_tier("cpu", P["cpu"], "share", cfg["cpu_gibps"]),
             "ssd": mk_tier(f"ssd_{cfg['ssd_dev']}", P[cfg["ssd_dev"]], "share")}
    cap = int(cfg["cpu_frac"] * n_sess * sum(sched))
    drop = cfg["release"] == "drop"
    if drop:
        st = CState(cfg["strategy"], cap, tiers, f, P, cfg["overflow"], cfg["qmax"], cfg.get("admit_first", False))
    else:
        st = BState(cfg["strategy"], cap, None, tiers, f, _Pool(), P, release=cfg["release"])
    ev, plan = make_workload(cfg["seed"], n_sess, len(sched))
    mode = "load_only" if cfg["strategy"] == "R0" else "cake"
    rows = []
    busy = 0.0

    def serve(s, r, t_arr, gpu_free, e):
        nonlocal busy
        n_old = sum(sched[:r - 1])
        n_new = n_old + sched[r - 1]
        t_start = max(t_arr, gpu_free)
        if drop:
            loc = st.layout_drop(s, n_old, t_start)
            t_go = t_start
        else:
            _, loc = st.layout(s, n_old)
            t_go = st.gate(t_start)
        t_end, rec, nc, ns, kind = restore_v2(n_old, loc, f, t_go, mode)
        t_new = sum(f[n_old:n_new])
        ttft = t_end + t_new - t_arr
        T = t_arr + ttft + (n_new - n_old) * CB / D2H_BPS
        busy += T - t_go
        if drop:
            st.after_round_drop(s, n_old, n_new, T, e)
        else:
            st.after_round(s, n_old, n_new, {i: None for i in range(n_old, n_new)}, T, e)
        rows.append(dict(ev=e, session=s, round=r, hist=n_old, t_arr=t_arr, t_start=t_start, T=T, ttft=ttft,
                         qwait=t_start - t_arr, gate=t_go - t_start, recompute=rec, ld_cpu=nc, ld_ssd=ns,
                         n_missing=sum(1 for x in loc if x is None), kind=kind, new_bytes=(n_new - n_old) * CB))
        return T

    if cfg["arrival"] == "serial":       # m7_sim.simulate 的序列：一個接一個
        t = 100.0
        for e, (s, r) in enumerate(ev):
            T = serve(s, r, t, t, e)
            t = T + cfg.get("ov", 0.01) + cfg.get("gap", 0.0)
    else:
        starts, rng = arrivals(cfg["seed"], n_sess, cfg["lam"], cfg["arrival"])
        # 思考時間事先抽好（每個 session、每一輪一個），所有策略用同一組亂數（common random numbers）
        think = {s: [rng.expovariate(1.0 / cfg["think"]) for _ in range(len(sched))] for s in range(n_sess)}
        heap = [(starts[s], s, s, 1) for s in range(n_sess)]
        heapq.heapify(heap)
        seq = n_sess
        gpu_free = 0.0
        e = 0
        while heap:
            t_arr, _, s, r = heapq.heappop(heap)
            gpu_free = serve(s, r, t_arr, gpu_free, e)
            e += 1
            if r < plan[s]:
                heapq.heappush(heap, (gpu_free + think[s][r], seq, s, r + 1))
                seq += 1
    return summarize2(rows, cfg, st, tiers, P, busy, cap), rows


def _q(xs, p):
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1)))] if xs else float("nan")


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return (xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])) if n else float("nan")


def summarize2(rows, cfg, st, tiers, P, busy, cap):
    ret = sorted(r["ttft"] for r in rows if r["round"] >= 2)
    svc = [r["ttft"] - r["qwait"] for r in rows if r["round"] >= 2]
    t0 = min(r["t_arr"] for r in rows)
    t1 = max(r["T"] for r in rows)
    span = t1 - t0
    W = P[cfg["ssd_dev"]]["write_GiBps"] * GiB
    newb = sum(r["new_bytes"] for r in rows)
    win = 30.0
    bins = {}
    for r in rows:
        bins[int((r["T"] - t0) // win)] = bins.get(int((r["T"] - t0) // win), 0) + r["new_bytes"]
    S = getattr(st, "S", {})
    out = dict(n_req=len(rows), n_ret=len(ret), median=median(ret), mean=sum(ret) / len(ret) if ret else float("nan"),
               p90=_q(ret, 0.9), p99=_q(ret, 0.99), svc_median=median(svc),
               qwait_mean=sum(r["qwait"] for r in rows if r["round"] >= 2) / max(1, len(ret)),
               span_s=span, gpu_util=busy / span if span > 0 else float("nan"),
               rho_kv=newb / (W * span), rho_peak=max(bins.values()) / (W * win),
               rho_strat=st.w_bytes["ssd"] / (W * span), cap_chunks=cap,
               new_chunks=newb // CB, w_ssd_GiB=st.w_bytes["ssd"] / GiB,
               recompute=sum(r["recompute"] for r in rows if r["round"] >= 2),
               ld_ssd=sum(r["ld_ssd"] for r in rows), ld_cpu=sum(r["ld_cpu"] for r in rows),
               n_general=sum(1 for r in rows if r["kind"] not in ("cake", "none", "compute")),
               n_missing_ret=sum(r["n_missing"] for r in rows if r["round"] >= 2),
               gate_mean=sum(r["gate"] for r in rows if r["round"] >= 2) / max(1, len(ret)))
    for k in ("policy_skip", "overflow_drop", "overflow_events", "discard", "demote", "ssd_w", "ssd_rej", "wt_skip",
              "evict_ssdcopy", "discard_new"):
        out[k] = S.get(k, "")
    out["drop_frac"] = (S["overflow_drop"] / S["new"]) if S and S.get("new") else (0.0 if S else "")
    return out


# ================================================================== validate
VAL_CELLS = [  # D1 §2.6（開跑前寫死）
    (35.4159, "free", "chat", "nfs", 0.25, 0, "S1"),
    (35.4159, "free", "chat", "nfs", 0.25, 0, "S4B"),
    (11.6, "free", "doc", "local", 0.5, 0, "S5L"),
    (3.69, "hold", "chat", "nfs", 0.5, 1, "S4W"),
    (3.69, "hold", "doc", "local", 0.25, 2, "S4C"),
    (11.6, "free", "chat", "local", 0.25, 3, "S2b"),
]


def validate(a):
    import pandas as pd
    P, f = load_params(), load_f()
    ref = pd.read_csv(os.path.join(HERE, "..", "results", "m7_explore_mi300x", "r1_sim_llama31_8b.csv"))
    out = []
    for cg, rel, wl, dev, cf, seed, strat in VAL_CELLS:
        m = ref[(ref.cpu_gibps.round(4) == round(cg, 4)) & (ref.release == rel) & (ref.workload == wl) &
                (ref.ssd_dev == dev) & (ref.cpu_frac == cf) & (ref.wl_seed == seed) & (ref.strategy == strat)]
        assert len(m) == 1, (len(m), cg, rel, wl, dev, cf, seed, strat)
        m = m.iloc[0]
        cfg = dict(strategy=strat, overflow="-", release=rel, arrival="serial", lam=0, think=0, n_sess=8,
                   cpu_gibps=None if abs(cg - P["cpu"]["read_GiBps"]) < 1e-3 else cg, ssd_dev=dev, workload=wl,
                   cpu_frac=cf, qmax=None, seed=seed, ov=0.01, gap=0.0)
        sm, rows = run(cfg, P, f)
        err = 100 * (sm["median"] / m["median"] - 1)
        out.append(dict(run_id=RID, ts=now_ts(), cpu_gibps=cg, release=rel, workload=wl, ssd_dev=dev, cpu_frac=cf,
                        wl_seed=seed, strategy=strat, ref_run_id=m["run_id"], ref_median=m["median"],
                        sim_median=sm["median"], err_median_pct=err, ref_p90=m["p90"], sim_p90=sm["p90"],
                        err_p90_pct=100 * (sm["p90"] / m["p90"] - 1), n_general_restores=sm["n_general"],
                        within_1pct=abs(err) <= 1.0))
    df = pd.DataFrame(out)
    fn = os.path.join(OUT, "d1_validate.csv")
    df.to_csv(fn, index=False)
    pd.set_option("display.width", 250)
    print(df.to_string())
    print(f"within 1%: {int(df.within_1pct.sum())}/{len(df)}  (需要 ≥4)")
    # 還原修正的單元檢查：保留頭（尾巴缺）不再整段重算
    import m7_sim
    f0 = f
    n = 64
    for k in (4, 16, 32):
        for cg in (35.4159, 3.69):
            tiers = {"cpu": mk_tier("cpu", P["cpu"], "share", cg)}
            loc = [tiers["cpu"]] * (n - k) + [None] * k
            old = m7_sim.sim_restore(n, list(loc), f0, 0.0, "cake")[0]
            tiers = {"cpu": mk_tier("cpu", P["cpu"], "share", cg)}
            loc = [tiers["cpu"]] * (n - k) + [None] * k
            new = restore_v2(n, loc, f0, 0.0)[0]
            tiers = {"cpu": mk_tier("cpu", P["cpu"], "share", cg)}
            ell = CB / tiers["cpu"].read_Bps
            lp_rs = (n - k) * ell + sum(f0[n - k:n])    # 載前綴＋算後綴（m=0）
            print(f"keep-head n={n} drop tail k={k} cpu={cg}: old sim_restore {old:.3f}s  new {new:.3f}s  "
                  f"load-prefix+recompute-suffix {lp_rs:.3f}s")


# ================================================================== sweep
GRID = dict(cpu_gibps=[3.69, 11.6], ssd_dev=["nfs", "local"], workload=["chat", "doc"], arrival=["poisson", "bursty"],
            lam=[0.05, 0.1, 0.2, 0.4], cpu_frac=[0.0625, 0.125, 0.25])
SEEDS = [0, 1, 2, 3, 4]

_P = _F = None


def add_derived(P):
    """D1 §2.10 追加 3：local8＝本地 SSD 讀寫各除以 8（8 張 GPU 平均分攤一顆 SSD）〔算術，假設，沒有量〕。"""
    lo = P["local"]
    P = dict(P)
    P["local8"] = dict(lo, read_GiBps=lo["read_GiBps"] / 8, write_GiBps=lo["write_GiBps"] / 8,
                       note="derived: local/8, assumption (8 GPUs share one NVMe), NOT measured")
    return P


def _init():
    global _P, _F
    _P, _F = add_derived(load_params()), load_f()


def _job(cfg):
    sm, _ = run(cfg, _P, _F)
    return dict(cfg, **sm)


def sweep_cfgs(grid, seeds, strats, overflows, qmax, think, n_sess, admit_first=False):
    keys = list(grid)
    for vals in itertools.product(*grid.values()):
        c = dict(zip(keys, vals))
        for seed in seeds:
            for strat in strats:
                for ov in overflows:
                    yield dict(strategy=strat, overflow=ov, release="drop", think=think, n_sess=n_sess, qmax=qmax,
                               seed=seed, admit_first=admit_first, **c)


def sweep(a):
    import multiprocessing as mp
    import pandas as pd
    grid = dict(GRID)
    if a.quick:
        grid = dict(cpu_gibps=[3.69], ssd_dev=["nfs"], workload=["chat"], arrival=["poisson"], lam=[0.2],
                    cpu_frac=[0.125])
    for k in ("cpu_gibps", "ssd_dev", "workload", "arrival", "lam", "cpu_frac"):
        v = getattr(a, k)
        if v:
            grid[k] = v
    cfgs = list(sweep_cfgs(grid, a.seeds, a.strats, a.overflows, a.qmax, a.think, a.n_sess, a.admit_first))
    print("jobs", len(cfgs), flush=True)
    t0 = datetime.now()
    with mp.Pool(a.procs, initializer=_init) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(_job, cfgs, chunksize=8)):
            res.append(r)
            if i % 2000 == 0:
                print(i, "/", len(cfgs), datetime.now() - t0, flush=True)
    df = pd.DataFrame(res)
    df.insert(0, "ts", now_ts())
    df.insert(0, "run_id", RID)
    rd = os.path.join(os.environ.get("TIARA_RUNS", "/mlsteam/data/tiara/runs"), RID)   # 大檔放 run 目錄（repo 檔要 <1 MB）
    os.makedirs(rd, exist_ok=True)
    fn = os.path.join(rd, a.out)
    df.to_csv(fn, index=False)
    print(fn, len(df), "rows", datetime.now() - t0)


# ================================================================== analyze
CFG_KEYS = ["cpu_gibps", "ssd_dev", "workload", "arrival", "lam", "cpu_frac", "qmax", "think", "n_sess"]


def analyze(a):
    import pandas as pd
    d = pd.concat([pd.read_csv(p) for p in a.inp], ignore_index=True)
    d["qmax"] = d["qmax"].fillna(-1)
    key = CFG_KEYS
    rows = []
    for k, g in d.groupby(key + ["seed"]):
        cand_name = a.cand
        c = g[(g.strategy == cand_name) & (g.overflow == a.cand_ov)]
        if len(c) != 1:
            continue
        c = c.iloc[0]
        tw = g[g.strategy.isin(a.twins)]
        best = tw.loc[tw.groupby("strategy")[a.metric].idxmin()]       # 每個對手取三種溢出規則裡最好的
        bv = best.set_index("strategy")[a.metric]
        gains = {st_: float(v) / c[a.metric] - 1 for st_, v in bv.items()}
        worst = min(gains, key=gains.get)
        rows.append(dict(zip(key + ["seed"], k), cand=cand_name, cand_ov=a.cand_ov, cand_val=c[a.metric],
                         min_gain=gains[worst], tightest_twin=worst,
                         tightest_twin_ov=best.set_index("strategy").loc[worst, "overflow"],
                         rho_kv=c.rho_kv, rho_peak=c.rho_peak, rho_strat_cand=c.rho_strat, gpu_util=c.gpu_util,
                         cand_drop_frac=c.drop_frac, n_ret=c.n_ret,
                         **{f"g_{s}": v for s, v in gains.items()}))
    s = pd.DataFrame(rows)
    s.insert(0, "ts", now_ts())
    s.insert(0, "run_id", RID)
    s["src"] = ";".join(os.path.basename(os.path.dirname(p)) for p in a.inp)
    rd = os.path.join(os.environ.get("TIARA_RUNS", "/mlsteam/data/tiara/runs"), RID)   # 每 seed 的表放 run 目錄（repo 只放每格的表）
    os.makedirs(rd, exist_ok=True)
    s.to_csv(os.path.join(rd if a.seed_to_run else OUT, a.out_seed), index=False)
    agg = s.groupby(key).agg(n_seeds=("seed", "size"), n10=("min_gain", lambda x: int((x >= 0.10).sum())),
                             n5=("min_gain", lambda x: int((x >= 0.05).sum())),
                             med_min_gain=("min_gain", "median"), worst_min_gain=("min_gain", "min"),
                             rho_kv=("rho_kv", "mean"), rho_peak=("rho_peak", "mean"),
                             rho_strat_cand=("rho_strat_cand", "mean"), gpu_util=("gpu_util", "mean"),
                             cand_drop_frac=("cand_drop_frac", "mean"),
                             tightest=("tightest_twin", lambda x: x.value_counts().index[0])).reset_index()
    agg["pass10"] = agg.n10 >= 3
    agg["pass5"] = agg.n5 >= 3
    agg["stable"] = (agg.rho_kv < 1) & (agg.gpu_util < 0.95)
    agg.insert(0, "ts", now_ts())
    agg.insert(0, "run_id", RID)
    agg["src"] = s["src"].iloc[0]
    agg.to_csv(os.path.join(OUT, a.out_cfg), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("configs", len(agg), " pass10", int(agg.pass10.sum()), " pass10&stable", int((agg.pass10 & agg.stable).sum()),
          " pass5", int(agg.pass5.sum()), " pass5&stable", int((agg.pass5 & agg.stable).sum()))
    print(agg.sort_values("med_min_gain", ascending=False).head(25).to_string())
    if agg.pass10.any() and agg.stable[agg.pass10].any():
        v = "有看頭"
    elif agg.pass10.any() or (agg.pass5 & agg.stable).any():
        v = "可能"
    else:
        v = "死路"
    print("VERDICT (D1 §2.5):", v)


def regress(a):
    """追加 1 的前置檢查：加了新策略之後，舊策略（S5T、S4BT）的數字要和主掃描逐列相同；再把新策略的列另存。"""
    import pandas as pd
    main_, ph = pd.read_csv(a.inp[0]), pd.read_csv(a.inp[1])
    k = ["strategy", "overflow", "seed"] + CFG_KEYS
    for d in (main_, ph):
        d["qmax"] = d["qmax"].fillna(-1)
    chk = a.regress_strats
    m = main_[main_.strategy.isin(chk)].merge(ph[ph.strategy.isin(chk)], on=k, suffixes=("", "_ph"))
    diff = (m["median"] - m["median_ph"]).abs().max()
    print(f"regression rows={len(m)} max|median diff|={diff:.3g}  max|mean diff|={(m['mean'] - m['mean_ph']).abs().max():.3g}")
    assert len(m) == len(main_[main_.strategy.isin(chk)]) and diff < 1e-9, "舊策略的結果變了"
    if not a.regress_extract:
        return
    out = ph[ph.strategy.isin(POSTHOC)]
    fn = os.path.join(os.path.dirname(a.inp[1]), "d1_sweep_posthoc_new.csv")
    out.to_csv(fn, index=False)
    print(fn, len(out))


def ovsum(a):
    """§2.7 次要報告：三種溢出規則彼此比較（真實系統預設 whole／tail 對 pa）。只看有溢出的列。"""
    import pandas as pd
    d = pd.concat([pd.read_csv(p) for p in a.inp], ignore_index=True)
    d["qmax"] = d["qmax"].fillna(-1)
    k = ["strategy", "seed"] + CFG_KEYS
    w = d.pivot_table(index=k, columns="overflow", values=a.metric).reset_index()
    df = d.pivot_table(index=k, columns="overflow", values="drop_frac").reset_index()
    w = w.merge(df, on=k, suffixes=("", "_drop"))
    w = w[(w["whole_drop"] > 0) | (w["tail_drop"] > 0) | (w["pa_drop"] > 0)]
    w["tail_vs_pa"] = w["tail"] / w["pa"] - 1      # 正＝pa 比 tail 快
    w["whole_vs_pa"] = w["whole"] / w["pa"] - 1
    out = w.groupby("strategy").agg(n=("pa", "size"), pa_drop=("pa_drop", "mean"), tail_drop=("tail_drop", "mean"),
                                    whole_drop=("whole_drop", "mean"),
                                    tail_vs_pa_med=("tail_vs_pa", "median"), tail_vs_pa_p90=("tail_vs_pa", lambda x: x.quantile(0.9)),
                                    whole_vs_pa_med=("whole_vs_pa", "median"),
                                    whole_vs_pa_p90=("whole_vs_pa", lambda x: x.quantile(0.9)),
                                    frac_pa_better5=("tail_vs_pa", lambda x: float((x >= 0.05).mean()))).reset_index()
    out.insert(0, "ts", now_ts())
    out.insert(0, "run_id", RID)
    out["metric"] = a.metric
    out["src"] = ";".join(os.path.basename(os.path.dirname(p)) for p in a.inp)
    out.to_csv(os.path.join(OUT, a.out_cfg), index=False)
    print(out.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "sweep", "analyze", "regress", "ovsum"])
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--admit-first", action="store_true", help="D1 §2.9 追加 2")
    ap.add_argument("--regress-strats", nargs="*", default=["S5T", "S4BT"])
    ap.add_argument("--seed-to-run", type=int, default=1, help="每 seed 的判定表寫到 run 目錄（2026-10-10 06:03 起；之前寫在 results/，已搬走）")
    ap.add_argument("--regress-extract", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--strats", nargs="*", default=STRATS)
    ap.add_argument("--overflows", nargs="*", default=OVERFLOWS)
    ap.add_argument("--qmax", type=int, default=None)
    ap.add_argument("--think", type=float, default=20.0)
    ap.add_argument("--n-sess", type=int, default=32)
    ap.add_argument("--cpu-gibps", type=float, nargs="*")
    ap.add_argument("--ssd-dev", nargs="*")
    ap.add_argument("--workload", nargs="*")
    ap.add_argument("--arrival", nargs="*")
    ap.add_argument("--lam", type=float, nargs="*")
    ap.add_argument("--cpu-frac", type=float, nargs="*")
    ap.add_argument("--out", default="d1_sweep.csv")
    ap.add_argument("--inp", nargs="*", default=["/mlsteam/data/tiara/runs/20261010-054631-d1-sweep/d1_sweep.csv"])
    ap.add_argument("--cand", default=CAND)
    ap.add_argument("--cand-ov", default=CAND_OV)
    ap.add_argument("--twins", nargs="*", default=TWINS)
    ap.add_argument("--metric", default="median")
    ap.add_argument("--out-seed", default="d1_verdict_seed.csv")
    ap.add_argument("--out-cfg", default="d1_verdict_cfg.csv")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {"validate": validate, "sweep": sweep, "analyze": analyze, "regress": regress, "ovsum": ovsum}[a.cmd](a)


if __name__ == "__main__":
    main()
