"""x2_stats.py — X2 的無參數統計：Kaplan–Meier、life-table hazard、Harrell C-index（右設限）。

沒有任何從資料擬合的參數（CLAUDE.md §8）。Harrell C 用 numba 的 Fenwick tree，O(n log n)；
單元測試 `python -I code/p3x/x2_stats.py` 和暴力 O(n^2)、lifelines.utils.concordance_index 對拍。
"""
from __future__ import annotations

import numpy as np
from numba import njit


# ---------------------------------------------------------------- Harrell C
@njit(cache=False)
def harrell_counts(R, ev, s):
    """可比的一對 (i,j)：ev[i] 且 (R[j] > R[i]，或 R[j]==R[i] 且 j 設限)。
    分數越大＝預測越晚回來。回傳 (concordant, discordant, tied_score)：s[j]>s[i] 為一致。"""
    n = R.shape[0]
    if n < 2:
        return 0, 0, 0
    order_s = np.argsort(s, kind="mergesort")
    rank = np.empty(n, np.int64)
    r = 0
    prev = 0.0
    for idx in range(n):
        i = order_s[idx]
        if idx == 0 or s[i] != prev:
            r += 1
            prev = s[i]
        rank[i] = r
    m = r
    tree = np.zeros(m + 1, np.int64)
    order_R = np.argsort(-R, kind="mergesort")
    conc = 0
    disc = 0
    tie = 0
    inserted = 0
    p = 0
    while p < n:
        q = p
        Rv = R[order_R[p]]
        while q < n and R[order_R[q]] == Rv:
            q += 1
        for k in range(p, q):  # 同時間的設限者算「比較晚」
            i = order_R[k]
            if not ev[i]:
                j = rank[i]
                while j <= m:
                    tree[j] += 1
                    j += j & (-j)
                inserted += 1
        for k in range(p, q):
            i = order_R[k]
            if ev[i]:
                ri = rank[i]
                a = 0
                j = ri
                while j > 0:
                    a += tree[j]
                    j -= j & (-j)
                b = 0
                j = ri - 1
                while j > 0:
                    b += tree[j]
                    j -= j & (-j)
                conc += inserted - a
                tie += a - b
                disc += b
        for k in range(p, q):
            i = order_R[k]
            if ev[i]:
                j = rank[i]
                while j <= m:
                    tree[j] += 1
                    j += j & (-j)
                inserted += 1
        p = q
    return conc, disc, tie


def cindex(R, ev, s):
    R = np.asarray(R, np.float64)
    ev = np.asarray(ev, np.bool_)
    s = np.asarray(s, np.float64)
    ok = np.isfinite(s)
    c, d, t = harrell_counts(R[ok], ev[ok], s[ok])
    tot = c + d + t
    return ((c + 0.5 * t) / tot if tot else float("nan")), c, d, t


def brute_counts(R, ev, s):
    c = d = t = 0
    n = len(R)
    for i in range(n):
        if not ev[i]:
            continue
        for j in range(n):
            if j == i:
                continue
            if R[j] > R[i] or (R[j] == R[i] and not ev[j]):
                if s[j] > s[i]:
                    c += 1
                elif s[j] < s[i]:
                    d += 1
                else:
                    t += 1
    return c, d, t


# ---------------------------------------------------------------- Kaplan–Meier
def km(T, ev):
    """回傳 (times, S)：每個不同事件時間之後的存活。"""
    T = np.asarray(T, float)
    ev = np.asarray(ev, bool)
    o = np.argsort(T, kind="mergesort")
    T, ev = T[o], ev[o]
    ut, idx = np.unique(T, return_index=True)
    n = len(T)
    at_risk = n - idx
    d = np.add.reduceat(ev.astype(np.int64), idx)
    keep = d > 0
    S = np.cumprod(1.0 - d[keep] / at_risk[keep])
    return ut[keep], S


def km_at(times, S, grid):
    grid = np.asarray(grid, float)
    k = np.searchsorted(times, grid, side="right") - 1
    return np.where(k >= 0, S[np.clip(k, 0, None)], 1.0)


def km_median(times, S):
    k = np.nonzero(S <= 0.5)[0]
    return float(times[k[0]]) if len(k) else float("inf")


# ---------------------------------------------------------------- life table
def life_table(T, ev, edges):
    """actuarial：n＝箱開頭仍在風險中；q＝d/(n−c/2)；h＝d/((n−c/2−d/2)·寬)。最後一箱的寬是 ∞ 時 h 記 nan。"""
    T = np.asarray(T, float)
    ev = np.asarray(ev, bool)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        n = int(np.sum(T >= lo))
        inb = (T >= lo) & (T < hi)
        d = int(np.sum(inb & ev))
        c = int(np.sum(inb & ~ev))
        eff = n - c / 2.0
        q = d / eff if eff > 0 else float("nan")
        w = hi - lo
        h = d / ((eff - d / 2.0) * w) if (eff - d / 2.0) > 0 and np.isfinite(w) else float("nan")
        rows.append(dict(bin_lo=lo, bin_hi=hi, n_at_risk=n, d=d, c=c, q=q, hazard_per_s=h))
    return rows


def _selftest():
    rng = np.random.default_rng(0)
    from lifelines.utils import concordance_index
    for trial in range(300):
        n = int(rng.integers(2, 60))
        R = rng.integers(0, 8, n).astype(float)
        ev = rng.random(n) < 0.6
        s = rng.integers(0, 5, n).astype(float)
        a = harrell_counts(R, ev, s)
        b = brute_counts(R, ev, s)
        assert tuple(a) == tuple(b), (trial, a, b)
    # lifelines：同一個定義（分數越大＝越晚），無同時間事件時應一致
    for trial in range(50):
        n = 500
        R = rng.exponential(10, n)
        ev = rng.random(n) < 0.7
        s = R * rng.lognormal(0, 1, n)
        c1 = cindex(R, ev, s)[0]
        c2 = concordance_index(R, s, ev)
        assert abs(c1 - c2) < 1e-9, (c1, c2)
    # KM：無設限時＝經驗存活
    T = rng.exponential(5, 1000)
    t, S = km(T, np.ones(1000, bool))
    assert abs(km_at(t, S, [5.0])[0] - np.mean(T > 5.0)) < 1e-12
    print("x2_stats selftest OK（harrell=brute 300 次、=lifelines 50 次、KM 無設限＝經驗分布）")


if __name__ == "__main__":
    _selftest()
