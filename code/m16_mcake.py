#!/usr/bin/env python3
"""m16_mcake.py — 第三階段 I1：N 個 session 同時回來時的「多 session Cake」排程核心（純 Python，不碰 torch／GPU）。

計畫：docs/stage3_20261019/experiment_matrix.md 的 I1。GPU 執行器在 m16_i1_multisession.py，用同一個 Sched；
這裡另有一個虛擬時鐘的模擬（simulate），給 --self-test 與「忽略干擾的預測」用。

## 模型
- session s：n_s 個 block（512 token），arrive_s 秒到達；KV 放在層 tier_s（cpu 或限速層）。
- 一條**計算線**（GPU 一條 compute stream）：從某個 session 的前面拿 block 重算（第 i 塊要 0..i−1 已在 GPU，所以每個 session 由前往後）。
- 每個層一條**載入線**（每個裝置一個佇列；cpu 與限速層互不阻擋）：從某個 session 的後面拿 block 載入。
- 每個 session 的前段指標 p_s、後段指標 q_s；p_s > q_s 且沒有進行中的 block → 還原完成；之後計算線（優先）做它的新一輪（new turn），做完＝TTFT。
- 單一 session 時，所有策略都等於 Cake（第二階段的還原器）。

## 兩個維度
順序（policy，計算線／載入線挑哪個 session）：
  rr      輪流：資源平分（每個 session 各自做，等於處理器共享）
  fcfs    先到先服務：兩條線都服務最早到、還沒完成的 session
  srpt    剩下的 block 最少的先（兩條線都是）
  global  計算線拿「位置最小」的前段 block（重算最便宜）、載入線拿「位置最大」的後段 block：把所有 session 當成一個大 Cake
  split   ＝ fcfs，但要求每個 session 都有 assign（舊名，保留相容）
怎麼還原（mode，每個 session 的 assign）：
  cake    assign＝''：計算線從前、載入線從後，會合（第二階段的 Cake）
  g5b     每個 session **各自**用第二階段 13 號的 G5b 規則（只看自己、單 session 的估計）決定 'L'（全部載入）或 ''（Cake）
  joint   auto_split：把所有 session 一起考慮，指定 'C'（全部重算）或 'L'（全部載入），讓兩條線的總量平衡
assign＝'L' 的 session 計算線不碰；'C' 的載入線不碰（任何 policy 都一樣）。
"""
from __future__ import annotations

import heapq
import threading
from dataclasses import dataclass, field

POLICIES = ("rr", "fcfs", "srpt", "global", "split")
ORDERS = ("rr", "fcfs", "srpt", "global")
MODES = ("cake", "g5b", "joint")


@dataclass
class Sess:
    n: int
    tier: str = "cpu"
    arrive: float = 0.0
    assign: str = ""                 # split 用：'C' 或 'L'
    p: int = 0
    q: int = -1
    inflight: int = 0
    t_restored: float | None = None
    t_ttft: float | None = None
    n_rec: int = 0
    n_load: int = 0

    def __post_init__(self):
        self.q = self.n - 1


class Sched:
    """執行緒安全。now 由呼叫端傳入（GPU 執行器用 perf_counter 相對時間；模擬用虛擬時間）。"""

    def __init__(self, sessions: list[Sess], policy: str, new_turn: bool = True):
        assert policy in POLICIES, policy
        if policy == "split":
            assert all(s.assign in ("C", "L") for s in sessions), "split 要每個 session 有 assign"
        self.S, self.policy, self.new_turn = sessions, policy, new_turn
        self.lock = threading.Lock()
        self.rr_c = 0
        self.rr_l: dict[str, int] = {}
        self.ready_new: list[int] = []          # 還原完成、等新一輪的 session（計算線優先做）

    # ---- 狀態 ----
    def _open(self, i: int, now: float) -> bool:
        s = self.S[i]
        return s.arrive <= now and s.p <= s.q

    def all_done(self) -> bool:
        with self.lock:
            return all((s.t_ttft is not None) if self.new_turn else (s.t_restored is not None) for s in self.S)

    def next_arrival(self, now: float) -> float | None:
        fut = [s.arrive for s in self.S if s.arrive > now and s.p <= s.q]
        return min(fut) if fut else None

    # ---- 挑選 ----
    def _pick(self, cands: list[int], side: str, tier: str | None = None) -> int:
        S, pol = self.S, self.policy
        if pol == "rr":
            if side == "c":
                k = min(cands, key=lambda i: (i - self.rr_c) % len(S))
                self.rr_c = (k + 1) % len(S)
            else:
                ptr = self.rr_l.get(tier, 0)
                k = min(cands, key=lambda i: (i - ptr) % len(S))
                self.rr_l[tier] = (k + 1) % len(S)
            return k
        if pol in ("fcfs", "split"):
            return min(cands, key=lambda i: (S[i].arrive, i))
        if pol == "srpt":
            return min(cands, key=lambda i: (S[i].q - S[i].p + 1, S[i].arrive, i))
        if pol == "global":
            if side == "c":
                return min(cands, key=lambda i: (S[i].p, S[i].arrive, i))
            return max(cands, key=lambda i: (S[i].q, -S[i].arrive, -i))
        raise ValueError(pol)

    def next_compute(self, now: float):
        """回傳 ('new', s) 新一輪、('rec', s, i) 重算、('wait', t) 等到 t、None 沒事可做了。"""
        with self.lock:
            if self.ready_new:
                return ("new", self.ready_new.pop(0))
            c = [i for i in range(len(self.S)) if self._open(i, now) and self.S[i].assign != "L"
                 and (self.policy != "split" or self.S[i].assign == "C")]
            if c:
                k = self._pick(c, "c")
                s = self.S[k]
                i = s.p
                s.p += 1
                s.inflight += 1
                return ("rec", k, i)
            pending = any((x.t_ttft is None) if self.new_turn else False for x in self.S)
            na = self.next_arrival(now)
            if na is not None:
                return ("wait", na)
            return ("wait", None) if pending else None

    def next_load(self, tier: str, now: float):
        """回傳 ('load', s, i)、('wait', t)、None。"""
        with self.lock:
            c = [i for i in range(len(self.S)) if self.S[i].tier == tier and self._open(i, now) and self.S[i].assign != "C"
                 and (self.policy != "split" or self.S[i].assign == "L")]
            if c:
                k = self._pick(c, "l", tier)
                s = self.S[k]
                i = s.q
                s.q -= 1
                s.inflight += 1
                return ("load", k, i)
            fut = [x.arrive for x in self.S if x.tier == tier and x.arrive > now and x.p <= x.q and x.assign != "C"
                   and (self.policy != "split" or x.assign == "L")]
            return ("wait", min(fut)) if fut else None

    def finish(self, k: int, kind: str, now: float):
        with self.lock:
            s = self.S[k]
            if kind == "new":
                s.t_ttft = now
                return
            s.inflight -= 1
            if kind == "rec":
                s.n_rec += 1
            else:
                s.n_load += 1
            if s.p > s.q and s.inflight == 0 and s.t_restored is None:
                s.t_restored = now
                if self.new_turn:
                    self.ready_new.append(k)


def g5b_assign(sessions: list[Sess], f_rec, t_load: dict[str, float]) -> list[str]:
    """第二階段 13 號 G5b（m7_restore_harness._g5_decide）的單 session 版：全部載入的時間 ≤ 模擬的 Cake 雙線 → 'L'，否則 ''（Cake）。
    每個 session 只看自己（不知道有別的 session 在搶）＝「各請求各自最佳」的基線。"""
    out = []
    for s in sessions:
        tl = t_load[s.tier]
        tr = tq = 0.0
        p, q = 0, s.n - 1
        while p <= q:
            if tr <= tq:
                tr += f_rec(p)
                p += 1
            else:
                tq += tl
                q -= 1
        out.append("L" if s.n * tl <= max(tr, tq) else "")
    return out


def apply_mode(sessions: list[Sess], mode: str, f_rec, t_load: dict[str, float]) -> list[Sess]:
    assert mode in MODES, mode
    if mode == "cake":
        a = [""] * len(sessions)
    elif mode == "g5b":
        a = g5b_assign(sessions, f_rec, t_load)
    else:
        a = auto_split(sessions, f_rec, t_load)
    for s, x in zip(sessions, a):
        s.assign = x
    return sessions


def auto_split(sessions: list[Sess], f_rec, t_load: dict[str, float]) -> list[str]:
    """split 的 auto：每個 session 全部重算的時間 R_s＝Σ f_rec(i)，全部載入 L_s＝n·t_load[tier]。
    貪婪（大的先）把 session 放到「放進去之後 max(計算線總量, 該層載入線總量) 最小」的那邊。"""
    R = [sum(f_rec(i) for i in range(s.n)) for s in sessions]
    Lt = [s.n * t_load[s.tier] for s in sessions]
    order = sorted(range(len(sessions)), key=lambda k: -max(R[k], Lt[k]))
    load_c, load_l = 0.0, {t: 0.0 for t in t_load}
    out = [""] * len(sessions)
    for k in order:
        tier = sessions[k].tier
        a = max(load_c + R[k], max(load_l.values()))
        ll = dict(load_l)
        ll[tier] += Lt[k]
        b = max(load_c, max(ll.values()))
        if a <= b:
            out[k] = "C"
            load_c += R[k]
        else:
            out[k] = "L"
            load_l = ll
    return out


# ---------------------------------------------------------------- 虛擬時鐘模擬（忽略干擾的預測；也是自測的參考）
def simulate(sessions: list[Sess], policy: str, f_rec, t_load: dict[str, float], t_new: float = 0.0,
             new_turn: bool = True, k_rec: float = 1.0, k_load: float = 1.0) -> dict:
    """f_rec(i)：第 i 塊重算秒數；t_load[tier]：每塊載入秒數。k_rec／k_load：兩條線同時忙時各自的放大倍數
    （1＝互不干擾，CacheFlow 的假設；>1＝干擾）。回傳每個 session 的還原完成、TTFT 與統計。"""
    S = [Sess(n=s.n, tier=s.tier, arrive=s.arrive, assign=s.assign) for s in sessions]
    sc = Sched(S, policy, new_turn=new_turn)
    tiers = sorted({s.tier for s in S})
    t = 0.0
    busy = {"c": None, **{tr: None for tr in tiers}}       # 每條線：(結束時間, kind, k)
    ev = []
    log = []

    def start_c(now):
        r = sc.next_compute(now)
        if r is None or r[0] == "wait":
            busy["c"] = None
            return r
        if r[0] == "new":
            dur = t_new
            busy["c"] = (now + dur, "new", r[1])
        else:
            _, k, i = r
            others = any(busy[tr] is not None for tr in tiers)
            dur = f_rec(i) * (k_rec if others else 1.0)
            busy["c"] = (now + dur, "rec", k)
            log.append(("rec", k, i, now, now + dur))
        heapq.heappush(ev, (busy["c"][0], "c"))
        return r

    def start_l(tr, now):
        r = sc.next_load(tr, now)
        if r is None or r[0] == "wait":
            busy[tr] = None
            return r
        _, k, i = r
        dur = t_load[tr] * (k_load if busy["c"] is not None else 1.0)
        busy[tr] = (now + dur, "load", k)
        log.append(("load", k, i, now, now + dur))
        heapq.heappush(ev, (busy[tr][0], tr))
        return r

    def kick(now):
        for tr in tiers:
            if busy[tr] is None:
                r = start_l(tr, now)
                if r is not None and r[0] == "wait" and r[1] is not None:
                    heapq.heappush(ev, (r[1], "arr"))
        if busy["c"] is None:
            r = start_c(now)
            if r is not None and r[0] == "wait" and r[1] is not None:
                heapq.heappush(ev, (r[1], "arr"))

    kick(0.0)
    guard = 0
    while not sc.all_done():
        guard += 1
        assert guard < 10_000_000, "模擬沒有收斂"
        if not ev:
            raise RuntimeError("沒有事件但還沒做完（排程死鎖）")
        t, line = heapq.heappop(ev)
        if line in ("c", *tiers) and busy[line] is not None and abs(busy[line][0] - t) < 1e-15:
            _, kind, k = busy[line]
            busy[line] = None
            sc.finish(k, kind, t)
        kick(t)
    out = []
    for s in S:
        out.append(dict(n=s.n, tier=s.tier, arrive=s.arrive, assign=s.assign, n_rec=s.n_rec, n_load=s.n_load,
                        restored=s.t_restored - s.arrive, ttft=(s.t_ttft - s.arrive) if s.t_ttft is not None else None))
    return dict(policy=policy, sessions=out, log=log, makespan=max(s.t_ttft if new_turn else s.t_restored for s in S))


def stats(vals: list[float]) -> dict:
    v = sorted(vals)
    n = len(v)

    def pct(p):
        if n == 1:
            return v[0]
        x = p * (n - 1)
        lo = int(x)
        hi = min(lo + 1, n - 1)
        return v[lo] + (v[hi] - v[lo]) * (x - lo)
    return dict(mean=sum(v) / n, p50=pct(0.5), p95=pct(0.95), max=v[-1], n=n)


# ---------------------------------------------------------------- 自測（只驗邏輯，不產生任何結果）
def self_test() -> bool:
    ok = True

    def check(name, cond):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name)
        ok &= bool(cond)

    f = lambda i: 0.010 + 0.0005 * i          # noqa: E731  重算隨位置變貴（合成值，只驗邏輯）
    tl = {"cpu": 0.004, "slow": 0.040}
    one = [Sess(n=64, tier="slow")]
    meets = {}
    for pol in POLICIES:
        ss = [Sess(n=64, tier="slow", assign="C")] if pol == "split" else one
        r = simulate(ss, pol, f, tl, t_new=0.0)
        meets[pol] = (r["sessions"][0]["n_rec"], round(r["sessions"][0]["restored"], 9))
    base = {k: v for k, v in meets.items() if k != "split"}
    check("N=1：rr／fcfs／srpt／global 都等於 Cake（會合點與時間相同）", len(set(base.values())) == 1)
    # 單一 session 的 Cake 會合點：和直接算的雙線一致
    tr = tl_ = 0.0
    p, q = 0, 63
    while p <= q:
        if tr <= tl_:
            tr += f(p)
            p += 1
        else:
            tl_ += tl["slow"]
            q -= 1
    check("N=1 的會合點＝手算的 Cake", meets["rr"][0] == p)
    # 守恆：每個 block 恰好一次
    for pol in POLICIES:
        ss = [Sess(n=32, tier="cpu" if k % 2 else "slow") for k in range(6)]
        if pol == "split":
            for k, s in enumerate(ss):
                s.assign = "C" if k % 3 == 0 else "L"
        r = simulate(ss, pol, f, tl, t_new=0.001)
        blocks = [(x[1], x[2]) for x in r["log"]]
        check(f"{pol}：每個 block 恰好做一次", len(blocks) == len(set(blocks)) == 6 * 32
              and all(s["n_rec"] + s["n_load"] == 32 for s in r["sessions"]))
        check(f"{pol}：每個 session 都有 TTFT ≥ 還原", all(s["ttft"] >= s["restored"] for s in r["sessions"]))
    # 相同 session、同時到：fcfs 的平均完成時間 < rr（處理器共享 vs 先到先服務）
    same = [Sess(n=64, tier="slow") for _ in range(8)]
    m_fcfs = stats([s["ttft"] for s in simulate(same, "fcfs", f, tl, 0.0)["sessions"]])
    m_rr = stats([s["ttft"] for s in simulate(same, "rr", f, tl, 0.0)["sessions"]])
    check("同樣大小、同時到：fcfs 平均 < rr 平均", m_fcfs["mean"] < m_rr["mean"])
    # split：assign 被遵守
    ss = [Sess(n=16, tier="cpu", assign="C"), Sess(n=16, tier="cpu", assign="L")]
    r = simulate(ss, "split", f, tl, 0.0)
    check("split：C 全重算、L 全載入", r["sessions"][0]["n_rec"] == 16 and r["sessions"][1]["n_load"] == 16)
    # auto_split 回傳每個 session 一個 C／L
    a = auto_split([Sess(n=64, tier="slow"), Sess(n=8, tier="cpu"), Sess(n=32, tier="slow")], f, tl)
    check("auto_split 每個 session 都有 C／L", all(x in ("C", "L") for x in a) and len(a) == 3)
    # mode：g5b 在載入很快時選 L、很慢時選 Cake；joint 每個都是 C／L；cake 全部 ''
    g = g5b_assign([Sess(n=64, tier="cpu"), Sess(n=64, tier="slow")], f, {"cpu": 0.0001, "slow": 10.0})
    check("g5b：載入極快 → L、極慢 → Cake", g == ["L", ""])
    for mode in MODES:
        ss = apply_mode([Sess(n=32, tier="cpu") for _ in range(4)], mode, f, tl)
        for pol in ORDERS:
            r = simulate(ss, pol, f, tl, 0.001)
            blocks = [(x[1], x[2]) for x in r["log"]]
            check(f"{mode}×{pol}：守恆", len(blocks) == len(set(blocks)) == 4 * 32)
    ss = apply_mode([Sess(n=32, tier="cpu"), Sess(n=32, tier="cpu")], "g5b", f, {"cpu": 0.0001, "slow": 1.0})
    r = simulate(ss, "rr", f, {"cpu": 0.0001, "slow": 1.0}, 0.0)
    check("assign＝L 的 session 計算線不碰（任何 policy）", all(x["n_rec"] == 0 for x in r["sessions"]))
    # 到達時間：晚到的 session 不會在到達前開始
    ss = [Sess(n=16, tier="cpu", arrive=0.0), Sess(n=16, tier="cpu", arrive=1.0)]
    r = simulate(ss, "rr", f, tl, 0.0)
    starts = [x[3] for x in r["log"] if x[1] == 1]
    check("晚到的 session 在到達後才開始", min(starts) >= 1.0 - 1e-12)
    # 干擾：k_rec、k_load > 1 時不會更快
    r1 = simulate(same, "global", f, tl, 0.0)["makespan"]
    r2 = simulate(same, "global", f, tl, 0.0, k_rec=1.3, k_load=1.5)["makespan"]
    check("有干擾時 makespan 不會變短", r2 >= r1)
    # global 的 makespan 不比 rr、fcfs 差太多（理想下應最短或打平；這裡只檢查 ≤ 1.05 倍）
    mk = {pol: simulate(same, pol, f, tl, 0.0)["makespan"] for pol in ("rr", "fcfs", "global")}
    check("同時到、相同 session：global 的 makespan ≤ 其他 ×1.05", mk["global"] <= 1.05 * min(mk["rr"], mk["fcfs"]))
    print("ALL PASS" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_test() else 1)
