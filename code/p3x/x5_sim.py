"""x5_sim.py — X5：session 級的精度×容量模擬（CPU-BF16／CPU-INT4／慢層 INT4／丟掉）。

策略（事先寫死，X5 §2.3）：
  R1 寫 BF16；滿了 LRU → 慢層
  R2 寫 INT4；滿了 LRU → 慢層
  R3 寫 BF16；滿了先壓「最大的」BF16（同大小取較舊）；沒有 BF16 才把 LRU 的 INT4 → 慢層
  R4 同 R3，但先壓 LRU 的 BF16
  R5 BF16 最多佔 50% 容量；超過壓 LRU BF16，否則 LRU INT4 → 慢層（沒有 INT4 時 LRU BF16 → 慢層）
  R6 成本感知：(還原時間增加 ÷ 釋放的位元組 ÷ 閒置時間) 最小的動作；寫入時 INT4 不比 BF16 慢就寫 INT4
  OA R1 的動作＋Belady 順序（不回來的丟）
  OB R4 的動作＋Belady 順序（不回來的丟）
  OC R2＋Belady 順序（不回來的丟）
  OD R6 的分數，閒置時間換成距離下次回來的時間、還原時間用下次真的要讀的 block 數；不回來的丟
  ODn OD 但不准丟（只報告：headroom 裡「丟掉不回來的」佔多少）
一次服務一個請求、沒有排隊；慢層容量無限；寫入、壓縮免費（X5 §2.2）。
"""
from __future__ import annotations

import heapq
from collections import OrderedDict

import numpy as np

NONE, BF, I4, SL, DR = 0, 1, 2, 3, 4
STATE_NAMES = {NONE: "NONE", BF: "BF16", I4: "INT4", SL: "SLOW", DR: "DROP"}
COST_AWARE = ("R6", "OD", "ODn", "R6b")
POLICIES = ("R1", "R2", "R3", "R4", "R5", "R6", "OA", "OB", "OC", "OD", "ODn", "R2b", "R4b", "R6b")
# 只報告的部分 oracle：R*b＝規則 R* ＋「知道誰不會再回來」（二元資訊），不回來的先降（慢層無限，降＝丟）。
RULES = ("R1", "R2", "R3", "R4", "R5", "R6")
ORACLES = ("OA", "OB", "OC", "OD")
TBINS = (10.0, 60.0, 300.0, 1200.0)


def tbin(dt):
    if not np.isfinite(dt):
        return "never"
    for i, b in enumerate(TBINS):
        if dt < b:
            return f"<{int(b)}s"
    return f">={int(TBINS[-1])}s"


def simulate(tr: dict, bytes_bf: float, bytes_i4: float, n_shared: int, C_bytes: float, policy: str,
             tabs: dict | None = None) -> dict:
    """tabs（成本感知策略才要）：{state: {s: np.array 還原時間 over k}}，state ∈ BF16/INT4/SLOW/DROP。"""
    assert policy in POLICIES
    T, sid, prev, nxt, keff, s_arr, nb = tr["T"], tr["sid"], tr["prev"], tr["nxt"], tr["keff"], tr["s"], tr["nb"]
    nS = tr["n_sess"]
    N = len(T)
    loc = np.zeros(nS, dtype=np.int8)
    n = np.zeros(nS, dtype=np.int64)
    s = np.zeros(nS, dtype=np.int64)
    last = np.zeros(nS)
    nextT = np.full(nS, np.inf)
    nextK = np.zeros(nS, dtype=np.int64)
    used = n_shared * bytes_bf
    C = C_bytes
    od_bf, od_i4 = OrderedDict(), OrderedDict()
    heap_big = []                      # R3：(−n, last, j)
    heap_bel = []                      # Belady：(−nextT, j, last)
    bf_bytes = 0.0
    state_at = np.full(N, -1, dtype=np.int8)
    acts = {"compress": 0, "demote": 0, "drop": 0}
    act_bins = {}
    write_fmt = {"R1": BF, "R2": I4, "R3": BF, "R4": BF, "R5": BF, "R6": BF, "OA": BF, "OB": BF, "OC": I4,
                 "OD": BF, "ODn": BF, "R2b": I4, "R4b": BF, "R6b": BF}[policy]
    binfo = policy.endswith("b")
    od_never = OrderedDict()           # R*b：在 CPU 裡、之後不會再回來的 session
    cost_aware = policy in COST_AWARE
    oracle = policy.startswith("O")
    srow = np.zeros(nS, dtype=np.int64)
    if cost_aware:
        assert tabs is not None
        svals = sorted(set(int(x) for x in s_arr))
        row_of = {v: i for i, v in enumerate(svals)}
        TB = {st: np.stack([tabs[st][v] for v in svals]) for st in ("BF16", "INT4", "SLOW", "DROP")}
        KMAX = TB["BF16"].shape[1] - 1

    def tget(st, sv, k):
        return TB[st][row_of[sv], k]

    def remove_cpu(j):
        nonlocal used, bf_bytes
        od_never.pop(j, None)
        if loc[j] == BF:
            od_bf.pop(j, None)
            used -= n[j] * bytes_bf
            bf_bytes -= n[j] * bytes_bf
        elif loc[j] == I4:
            od_i4.pop(j, None)
            used -= n[j] * bytes_i4

    def add_cpu(j, fmt):
        nonlocal used, bf_bytes
        loc[j] = fmt
        if fmt == BF:
            od_bf[j] = None
            used += n[j] * bytes_bf
            bf_bytes += n[j] * bytes_bf
            if policy == "R3":
                heapq.heappush(heap_big, (-int(n[j]), last[j], j))
        else:
            od_i4[j] = None
            used += n[j] * bytes_i4
        if policy in ("OA", "OB", "OC"):
            heapq.heappush(heap_bel, (-nextT[j], j, last[j]))
        if binfo and not np.isfinite(nextT[j]):
            od_never[j] = None

    def do(j, act, now):
        nonlocal used, bf_bytes
        acts[act] += 1
        if oracle:
            key = (act, STATE_NAMES[int(loc[j])], tbin(nextT[j] - now))
            act_bins[key] = act_bins.get(key, 0) + 1
        if act == "compress":
            assert loc[j] == BF
            od_bf.pop(j)
            used -= n[j] * (bytes_bf - bytes_i4)
            bf_bytes -= n[j] * bytes_bf
            loc[j] = I4
            od_i4[j] = None
            if policy in ("OA", "OB", "OC"):
                heapq.heappush(heap_bel, (-nextT[j], j, last[j]))
        else:
            remove_cpu(j)
            loc[j] = SL if act == "demote" else DR

    def lru_head(od):
        return next(iter(od)) if od else None

    def global_lru():
        a, b = lru_head(od_bf), lru_head(od_i4)
        if a is None:
            return b
        if b is None:
            return a
        return a if last[a] <= last[b] else b

    def bel_best(states):
        """最晚回來的（在 states 裡），不 pop。heap 裡同一個 j 可能有過期的項目，要驗證。"""
        tmp = []
        found = None
        while heap_bel:
            negt, j, lt = heapq.heappop(heap_bel)
            if lt != last[j] or -negt != nextT[j] or loc[j] not in (BF, I4):
                continue                      # 過期
            tmp.append((negt, j, lt))
            if loc[j] in states:
                found = j
                break
        for x in tmp:
            heapq.heappush(heap_bel, x)
        return found

    def evict_one(now, jw):
        if binfo and od_never:
            do(next(iter(od_never)), "demote", now)
            return
        if policy in ("R1", "R2", "R2b"):
            j = global_lru()
            do(j, "demote", now)
        elif policy in ("R3", "R4", "R4b"):
            if od_bf:
                if policy in ("R4", "R4b"):
                    j = lru_head(od_bf)
                else:
                    while True:
                        negn, lt, j = heap_big[0]
                        if loc[j] == BF and n[j] == -negn and last[j] == lt:
                            break
                        heapq.heappop(heap_big)
                do(j, "compress", now)
            else:
                do(lru_head(od_i4), "demote", now)
        elif policy == "R5":
            if od_bf and bf_bytes > 0.5 * C:
                do(lru_head(od_bf), "compress", now)
            elif od_i4:
                do(lru_head(od_i4), "demote", now)
            else:
                do(lru_head(od_bf), "demote", now)
        elif policy in ("OA", "OC"):
            j = bel_best((BF, I4))
            do(j, "drop" if not np.isfinite(nextT[j]) else "demote", now)
        elif policy == "OB":
            j = bel_best((BF, I4))
            if not np.isfinite(nextT[j]):
                do(j, "drop", now)
            else:
                jb = bel_best((BF,))
                if jb is not None:
                    do(jb, "compress", now)
                else:
                    do(bel_best((I4,)), "demote", now)
        else:  # R6, OD, ODn：向量化打分
            idx_bf = np.fromiter(od_bf.keys(), dtype=np.int64, count=len(od_bf))
            idx_i4 = np.fromiter(od_i4.keys(), dtype=np.int64, count=len(od_i4))
            cands = []   # (score array, idx array, act)
            for idx, st in ((idx_bf, "BF16"), (idx_i4, "INT4")):
                if len(idx) == 0:
                    continue
                sr = srow[idx]
                if policy in ("R6", "R6b"):
                    kk = s[idx] + n[idx]
                    w = now - last[idx]
                else:
                    kk = nextK[idx]
                    w = nextT[idx] - now
                kk = np.minimum(kk, KMAX)
                tnow = TB[st][sr, kk]
                nbytes = n[idx] * (bytes_bf if st == "BF16" else bytes_i4)
                opts = [("demote", TB["SLOW"][sr, kk], nbytes)]
                if st == "BF16":
                    opts.append(("compress", TB["INT4"][sr, kk], n[idx] * (bytes_bf - bytes_i4)))
                if policy == "OD":
                    opts.append(("drop", TB["DROP"][sr, kk], nbytes))
                for act, tafter, freed in opts:
                    d = tafter - tnow
                    with np.errstate(divide="ignore", invalid="ignore"):
                        sc = np.where(d <= 0, -1e30 + d / np.maximum(freed, 1), d / np.maximum(freed, 1) / np.maximum(w, 1e-9))
                    if policy in ("OD", "ODn"):
                        never = ~np.isfinite(nextT[idx])
                        if policy == "OD" and act == "drop":
                            sc = np.where(never, -1e31 - freed, sc)
                        elif policy == "ODn" and act == "demote":
                            sc = np.where(never, -1e31 - freed, sc)
                        else:
                            sc = np.where(never, np.inf, sc)
                    if policy in ("R6", "R6b"):
                        sc = np.where(idx == jw, np.where(d <= 0, sc, np.inf), sc)
                    cands.append((sc, idx, act))
            best = None
            for sc, idx, act in cands:
                i = int(np.argmin(sc))
                if best is None or sc[i] < best[0]:
                    best = (sc[i], int(idx[i]), act)
            if best is None or not np.isfinite(best[0]):
                # 只剩剛寫入的 session（R6 保護它）→ 照 LRU 降級
                j = global_lru()
                do(j, "demote", now)
            else:
                do(best[1], best[2], now)

    for r in range(N):
        j = sid[r]
        if j < 0:
            continue
        now = T[r]
        if prev[r] >= 0:
            state_at[r] = loc[j]
            remove_cpu(j)
        n[j] = nb[r] - s_arr[r]
        s[j] = s_arr[r]
        if cost_aware:
            srow[j] = row_of[int(s_arr[r])]
        last[j] = now
        if nxt[r] >= 0:
            nextT[j] = T[nxt[r]]
            nextK[j] = keff[nxt[r]]
        elif "fake_nextT" in tr and np.isfinite(tr["fake_nextT"][r]):
            # 穩健性（事後加，X9 的設限提醒）：trace 結束前 H 秒內才最後出現的 session 不准當「不回來」
            nextT[j] = tr["fake_nextT"][r]
            nextK[j] = nb[r]
        else:
            nextT[j] = np.inf
            nextK[j] = 0
        fmt = write_fmt
        if policy in ("R6", "OD", "ODn", "R6b"):
            kq = (s[j] + n[j]) if policy in ("R6", "R6b") else (nextK[j] if np.isfinite(nextT[j]) else s[j] + n[j])
            kq = min(int(kq), KMAX)
            fmt = I4 if tget("INT4", int(s[j]), kq) <= tget("BF16", int(s[j]), kq) else BF
        add_cpu(j, fmt)
        guard = 0
        while used > C + 1e-6:
            if not od_bf and not od_i4:
                break                                  # 共用前綴本身就超過容量（不會發生在我們的格子）
            evict_one(now, j)
            guard += 1
            assert guard < 100000
    return dict(state_at=state_at, acts=acts, act_bins=act_bins)


def evaluate(tr: dict, state_at: np.ndarray, tabs: dict, prefill: np.ndarray) -> dict:
    """回來請求的 TTFT：還原（state_at 決定）＋新一輪 prefill（所有策略相同）。"""
    ret = tr["ret"] & tr.get("eval_mask", np.ones(len(tr["ret"]), bool))
    idx = np.nonzero(ret)[0]
    st = state_at[idx]
    assert np.all(st >= 1), "回來請求的狀態不應該是 NONE"
    k = tr["keff"][idx]
    sv = tr["s"][idx]
    rest = np.zeros(len(idx))
    for code, name in ((BF, "BF16"), (I4, "INT4"), (SL, "SLOW"), (DR, "DROP")):
        m = st == code
        if m.any():
            for v in np.unique(sv[m]):
                mm = m & (sv == v)
                rest[mm] = tabs[name][int(v)][k[mm]]
    ttft = rest + prefill[idx]
    int4_read = np.isin(st, (I4, SL))
    long_ = k > 64
    return dict(n_ret=len(idx), mean_ttft=float(ttft.mean()), median_ttft=float(np.median(ttft)),
                p90_ttft=float(np.percentile(ttft, 90)), mean_restore=float(rest.mean()),
                sum_ret_ttft=float(ttft.sum()),
                frac_BF16=float(np.mean(st == BF)), frac_INT4=float(np.mean(st == I4)),
                frac_SLOW=float(np.mean(st == SL)), frac_DROP=float(np.mean(st == DR)),
                frac_int4_long=float(np.mean(int4_read & long_)),
                share_ttft_int4_long=float(ttft[int4_read & long_].sum() / ttft.sum()),
                ttft=ttft)
