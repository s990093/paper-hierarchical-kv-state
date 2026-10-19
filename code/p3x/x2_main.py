"""x2_main.py — X2：回來時間有沒有 LRU 以外的訊號（無參數統計，只用 CPU）。

用法（走記錄殼，CLAUDE.md §4.1）：
  source code/env_3090.sh
  TIARA_NO_GPU=1 m7run p3x2-<短名> python -I code/p3x/x2_main.py --trace bailian_traceA
  TIARA_NO_GPU=1 m7run p3x2-wccpt  python -I code/p3x/x2_main.py --measure-wildchat-cpt
  TIARA_NO_GPU=1 m7run p3x2-wc     python -I code/p3x/x2_main.py --trace wildchat_1m --cpt-run <RUN_ID>

產出（全部附 run_id、ts）：results/p3_explore/x2/
  x2_validation.csv  單位斷言與 session 重建驗算
  x2_summary.csv     session 數、回來比例、間隔分位數、設限比例
  x2_km.csv          Kaplan–Meier S(a)
  x2_hazard.csv      life-table hazard（全部、第 1 輪、≥2 輪）
  x2_bins.csv        依輪數／上一次間隔／上一輪輸出長度的條件分布
  x2_cindex.csv      C-index（快照集合 × 分數，加總後的主值、bootstrap CI、前後半段）
原始的逐快照計數：$TIARA_RUNS/$RUN_ID/snap_counts.csv.gz
判準見 docs/phase3_20261017/explore/X2_predictability.md §2（開跑前寫）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))
import x2_load  # noqa: E402
from x2_stats import cindex, harrell_counts, km, km_at, km_median, life_table  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.environ.get("X2_OUT") or os.path.join(REPO, "results", "p3_explore", "x2")  # X2_OUT 只給除錯用
BYTES_PER_TOK = 131072  # Llama-3.1-8B BF16：32 層 × 8 KV 頭 × 128 × 2(K,V) × 2 B
GiB = 1 << 30
CAPS_GIB = [128, 256, 512, 1024]
PRIMARY = "lru256GiB"
AGE_EDGES = [0, 10, 30, 60, 120, 300, 600, 1200, 3600, np.inf]
HZ_EDGES = [0, 5, 10, 20, 30, 45, 60, 90, 120, 180, 240, 300, 450, 600, 900, 1200, 1800, 2700, 3600, 5400, 7200]
HZ_EDGES_WC = HZ_EDGES + [3 * 3600, 6 * 3600, 12 * 3600, 86400, 3 * 86400, 7 * 86400, 30 * 86400]
KM_GRID = [10, 30, 60, 120, 300, 600, 1800, 3600]
KM_GRID_WC = KM_GRID + [6 * 3600, 86400, 7 * 86400]
N_BOOT = 1000


def run_id():
    return os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc"


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def run_dir():
    d = os.path.join(os.environ.get("TIARA_RUNS", "/ssd8/hungwei/paper-hkv/runs"), run_id())
    os.makedirs(d, exist_ok=True)
    return d


class Out:
    def __init__(self, name, fields):
        os.makedirs(OUT, exist_ok=True)
        path = os.path.join(OUT, name)
        new = not os.path.exists(path)
        self.f = open(path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=["run_id", "ts"] + fields, extrasaction="raise")
        if new:
            self.w.writeheader()

    def row(self, **kw):
        kw = {k: (round(float(v), 6) if isinstance(v, (float, np.floating)) else v) for k, v in kw.items()}
        kw.update(run_id=run_id(), ts=now())
        self.w.writerow(kw)
        self.f.flush()


def log(*a):
    print(*a, flush=True)


def pct(x, qs=(10, 25, 50, 75, 90, 99)):
    x = np.asarray(x, float)
    return {f"p{q}": (float(np.percentile(x, q)) if len(x) else float("nan")) for q in qs}


# ---------------------------------------------------------------- 回合（episode）
def t_idle(req):
    """閒置開始的時刻：有 t_idle（請求結束時間，FreeInference）就用它，否則用請求到達時間。"""
    return req["t_idle"].values if "t_idle" in req.columns else req["t"].values


def episodes(req, meta):
    ev = req["next_t"].notna().values
    base = t_idle(req)
    if meta["trace"] == "cachewise":
        cens = req["sess_end"].values - base
    elif "cens_at" in req.columns:
        cens = req["cens_at"].values - base
    else:
        cens = meta["t_end"] - base
    T = np.where(ev, req["next_t"].values - base, cens)
    T = np.maximum(T, 0.0)
    return T, ev


# ---------------------------------------------------------------- 描述統計、KM、hazard、分箱
def describe(req, meta, T, ev, is_wc):
    o_s = Out("x2_summary.csv", ["trace", "metric", "value", "note"])
    put = lambda m, v, note="": o_s.row(trace=meta["trace"], metric=m, value=v, note=note)
    nps = req.groupby("sid").size()
    put("n_requests", len(req))
    put("n_sessions", int(len(nps)))
    put("frac_sessions_ge2_requests", float((nps >= 2).mean()))
    put("n_sessions_ge2_requests", int((nps >= 2).sum()))
    put("requests_per_session_p50", float(nps.median()))
    put("requests_per_session_p99", float(np.percentile(nps, 99)))
    put("requests_per_session_max", int(nps.max()))
    put("frac_episodes_returned", float(ev.mean()), "回合之後同 session 有下一個請求（觀察窗內）")
    put("frac_episodes_censored", float(1 - ev.mean()), "trace 結束（或 session 結束）時還沒回來")
    span = meta["t_end"] - meta["t_start"] if np.isfinite(meta["t_end"]) else float("nan")
    put("trace_span_s", span)
    if ev.any():
        q = pct(T[ev])
        for k, v in q.items():
            put(f"gap_returned_{k}_s", v, "只算有回來的回合")
        p90 = q["p90"]
        put("frac_censored_with_cens_time_lt_gap_p90", float(np.mean((~ev) & (T < p90)) / max(np.mean(~ev), 1e-12)),
            "設限的回合中，設限時間 < 有回來間隔的 p90（尾端看不清楚）")
        put("frac_all_episodes_censored_lt_gap_p90", float(np.mean((~ev) & (T < p90))))
        t, S = km(T, ev)
        put("km_median_gap_s", km_median(t, S), "inf＝S 沒降到 0.5")
        put("km_S_at_max", float(S[-1]), "觀察窗內最後的存活（≈ 不回來的比例下界）")
    o_k = Out("x2_km.csv", ["trace", "stratum", "a_s", "S", "n"])
    grid = KM_GRID_WC if is_wc else KM_GRID
    strata = {"all": np.ones(len(T), bool), "turn1": req["turn"].values == 1, "turn_ge2": req["turn"].values >= 2}
    if "grp" in req.columns:  # FreeInference：閒置當下就知道在等工具還是等使用者
        for g_ in ("user", "tool"):
            strata[f"wait_{g_}"] = req["grp"].values == g_
        nt = req.groupby("sid")["trig"].shift(-1).values
        for g_ in ("user", "tool"):
            m = (req["grp"].values == g_) & ev
            put(f"wait_{g_}_frac_next_trigger_same", float(np.mean(nt[m] == g_)) if m.any() else float("nan"),
                "閒置當下的 grp（上一個請求 finish_reason）和下一個請求 step_trigger 一致的比例")
            mm = req["grp"].values == g_
            put(f"wait_{g_}_n_episodes", int(mm.sum()))
            put(f"wait_{g_}_frac_returned", float(ev[mm].mean()) if mm.any() else float("nan"))
            if (mm & ev).any():
                for k, v in pct(T[mm & ev]).items():
                    put(f"wait_{g_}_gap_returned_{k}_s", v)
                t_, S_ = km(T[mm], ev[mm])
                put(f"wait_{g_}_km_median_gap_s", km_median(t_, S_))
    for nm, m in strata.items():
        if m.sum() == 0 or ev[m].sum() == 0:
            continue
        t, S = km(T[m], ev[m])
        for a, s in zip(grid, km_at(t, S, grid)):
            o_k.row(trace=meta["trace"], stratum=nm, a_s=a, S=float(s), n=int(m.sum()))
    o_h = Out("x2_hazard.csv", ["trace", "stratum", "bin_lo", "bin_hi", "n_at_risk", "d", "c", "q", "hazard_per_s"])
    edges = (HZ_EDGES_WC if is_wc else HZ_EDGES) + [np.inf]
    shape = {}
    for nm, m in strata.items():
        if m.sum() == 0:
            continue
        rows = life_table(T[m], ev[m], edges)
        for r in rows:
            o_h.row(trace=meta["trace"], stratum=nm, **r)
        hz = [(r["hazard_per_s"], r["d"]) for r in rows if np.isfinite(r["hazard_per_s"])]
        pairs = [(a, b) for (a, da), (b, db) in zip(hz[:-1], hz[1:]) if da >= 30 and db >= 30]
        if pairs:
            shape[nm] = (float(np.mean([b < a for a, b in pairs])), len(pairs))
    for nm, (f, k) in shape.items():
        put(f"hazard_frac_adjacent_decreasing_{nm}", f, f"相鄰兩箱都 ≥30 事件的 {k} 對中，下一箱 hazard 較小的比例")
    # 分箱條件分布
    o_b = Out("x2_bins.csv", ["trace", "by", "bin", "n", "frac_censored", "km_median_s", "P_le_60s", "P_le_300s"])
    turn = req["turn"].values
    pg = req["prev_gap"].values
    outv = req["out"].values
    by = {
        "turn": [("1", turn == 1), ("2", turn == 2), ("3", turn == 3), ("4-5", (turn >= 4) & (turn <= 5)),
                 ("6-10", (turn >= 6) & (turn <= 10)), (">=11", turn >= 11)],
        "prev_gap_s": [("none", np.isnan(pg))] + [
            (f"{lo}-{hi}", (pg >= lo) & (pg < hi)) for lo, hi in [(0, 30), (30, 60), (60, 120), (120, 300), (300, 600)]]
        + [(">=600", pg >= 600)],
        "last_out": [(f"{lo}-{hi}", (outv >= lo) & (outv < hi)) for lo, hi in [(0, 64), (64, 256), (256, 512), (512, 1024)]]
        + [(">=1024", outv >= 1024)],
    }
    for b, lst in by.items():
        for nm, m in lst:
            n = int(m.sum())
            if n == 0:
                o_b.row(trace=meta["trace"], by=b, bin=nm, n=0, frac_censored=float("nan"), km_median_s=float("nan"),
                        P_le_60s=float("nan"), P_le_300s=float("nan"))
                continue
            if ev[m].sum():
                t, S = km(T[m], ev[m])
                med = km_median(t, S)
                s60, s300 = km_at(t, S, [60, 300])
            else:
                med, s60, s300 = float("inf"), 1.0, 1.0
            o_b.row(trace=meta["trace"], by=b, bin=nm, n=n, frac_censored=float(1 - ev[m].mean()), km_median_s=med,
                    P_le_60s=float(1 - s60), P_le_300s=float(1 - s300))


# ---------------------------------------------------------------- C-index
VARIANTS = [  # (名稱, 分數欄, 子集)；S2u 只有 FreeInference 有（登記後才加，只報告）
    ("S0", "S0", "full"), ("S2", "S2", "full"), ("S3", "S3", "full"), ("S2u", "S2u", "full"),
    ("S1fill", "S1fill", "full"), ("S4fill", "S4fill", "full"),
    ("S0", "S0", "ge2"), ("S1", "S1", "ge2"), ("S2", "S2", "ge2"), ("S3", "S3", "ge2"), ("S4", "S4", "ge2"),
]
STRATA_VARIANTS = [("S0", "S0", "full"), ("S2", "S2", "full"), ("S3", "S3", "full"),
                   ("S0", "S0", "ge2"), ("S1", "S1", "ge2"), ("S4", "S4", "ge2")]


def snapshot_grid(meta, is_wc, req=None):
    """回傳 (taus, lo, wk)：lo＝這個快照只看 t_idle ≥ lo 的請求（WildChat 過去 7 天）；
    wk＝只看這一週的請求（FreeInference；各週的時間在邊界會重疊），其他 trace 是 −1（不限）。"""
    if meta["trace"] == "freeinference":  # 登記後才加：每週內每小時一個，避開每週頭尾 6 h
        taus, lo, wk = [], [], []
        for w, g in req.groupby("week"):
            ws, we = float(g["week_start"].iloc[0]), float(g["cens_at"].iloc[0])
            tt = np.arange(ws + 6 * 3600, we - 6 * 3600 + 1e-9, 3600.0)
            taus.append(tt)
            lo.append(np.full(len(tt), ws))
            wk.append(np.full(len(tt), int(w)))
        return np.concatenate(taus), np.concatenate(lo), np.concatenate(wk)
    taus = _grid(meta, is_wc)
    lo = taus - 7 * 86400 if is_wc else np.full(len(taus), -np.inf)
    return taus, lo, np.full(len(taus), -1)


def _grid(meta, is_wc):
    if meta["trace"].startswith("mooncake"):
        return np.arange(600.0, meta["t_end"] - 600.0 + 1e-9, 30.0)
    if meta["trace"].startswith("bailian"):
        return np.arange(900.0, meta["t_end"] - 900.0 + 1e-9, 60.0)
    if is_wc:
        return np.arange(meta["t_start"] + 7 * 86400, meta["t_end"] - 7 * 86400 + 1e-9, 6 * 3600.0)
    raise ValueError(meta["trace"])


def scores(sub, tau):
    a = tau - t_idle(sub)
    pg = sub["prev_gap"].values
    S = dict(S0=a, S1=pg, S2=-sub["turn"].values.astype(float), S3=sub["out"].values, S4=pg - a)
    S["S1fill"] = np.where(np.isnan(pg), a, pg)
    S["S4fill"] = np.where(np.isnan(pg), 0.0, pg - a)
    # 事後加的（x2_posthoc.py，只報告）：2Q／SLRU 式的固定規則——先比輪數（多的先回來），同輪數再用 LRU
    S["S2lex"] = -sub["turn"].values.astype(float) * 1e9 + a
    if "uturn" in sub.columns:  # FreeInference（只報告）：只數使用者發起的回合
        S["S2u"] = -sub["uturn"].values.astype(float)
    return a, S


def c_counts(R, ev, s, m):
    m = m & np.isfinite(s)
    c, d, t = harrell_counts(R[m], ev[m], s[m])
    return c, d, t


def run_cindex(req, meta, is_wc, rd, variants=None, strata_variants=None, tag=""):
    variants = VARIANTS if variants is None else variants
    strata_variants = STRATA_VARIANTS if strata_variants is None else strata_variants
    taus, los, wks = snapshot_grid(meta, is_wc, req)
    oI = np.argsort(t_idle(req), kind="mergesort")
    req = req.iloc[oI].reset_index(drop=True)  # 依閒置開始時刻排序（沒有 t_idle 時＝原本的 t 順序）
    t_all = t_idle(req)
    nxt = req["next_t"].values
    nxt = np.where(np.isnan(nxt), np.inf, nxt)
    cens_at = req["cens_at"].values if "cens_at" in req.columns else np.full(len(req), meta["t_end"])
    size_b = req["size_tok"].values * BYTES_PER_TOK
    has_grp = "grp" in req.columns
    week = req["week"].values if "week" in req.columns else None
    recs = []
    set_sids = {}
    set_sids_ge2 = {}
    set_info = {}
    for si, (tau, lo_t, wk) in enumerate(zip(taus, los, wks)):
        lo = np.searchsorted(t_all, lo_t, side="left") if np.isfinite(lo_t) else 0
        hi = np.searchsorted(t_all, tau, side="right")
        idx = np.arange(lo, hi)
        idle = idx[nxt[idx] > tau]
        if wk >= 0:
            idle = idle[week[idle] == wk]
        if len(idle) == 0:
            continue
        sub = req.iloc[idle]
        a, S = scores(sub, tau)
        evs = np.isfinite(nxt[idle])
        R = np.where(evs, nxt[idle] - tau, cens_at[idle] - tau)
        assert np.all(R >= 0), "剩餘時間 < 0：設限時刻早於快照"
        # LRU 會留在 CPU 的：依 a 由小到大累加 KV 大小
        o = np.argsort(a, kind="mergesort")
        cum = np.cumsum(size_b[idle][o])
        sets = {"all_idle": np.ones(len(idle), bool)}
        for C in CAPS_GIB:
            m = np.zeros(len(idle), bool)
            m[o[cum <= C * GiB]] = True
            sets[f"lru{C}GiB"] = m
            set_info.setdefault(f"lru{C}GiB", []).append(bool(cum[-1] <= C * GiB))  # True＝全部閒置都放得下（不受容量限制）
        if has_grp:  # 同一個 LRU 集合（兩種一起佔 CPU），C 在「等使用者」「等工具」裡各自算
            gv = sub["grp"].values
            for base in list(sets):
                for g_ in ("user", "tool"):
                    sets[f"{base}|wait_{g_}"] = sets[base] & (gv == g_)
        ge2 = sub["turn"].values >= 2
        sids = sub["sid"].values
        for sname, sm in sets.items():
            set_sids.setdefault(sname, set()).update(sids[sm].tolist())
            set_sids_ge2.setdefault(sname, set()).update(sids[sm & ge2].tolist())
            for vname, col, subset in variants:
                if col not in S:
                    continue
                m = sm & (ge2 if subset == "ge2" else True)
                if m.sum() < 2:
                    continue
                c, d, t = c_counts(R, evs, S[col], m)
                recs.append(dict(snap=si, tau=tau, set=sname, variant=vname, subset=subset, strat="none",
                                 n=int(m.sum()), n_ev=int(evs[m].sum()), conc=c, disc=d, tie=t))
            # 條件版本：同一個已閒置時間箱子內比
            abin = np.searchsorted(AGE_EDGES, a, side="right") - 1
            for vname, col, subset in strata_variants:
                cc = dd = tt = 0
                nn = 0
                for b in np.unique(abin[sm]):
                    m = sm & (abin == b) & (ge2 if subset == "ge2" else True)
                    if m.sum() < 2:
                        continue
                    c, d, t = c_counts(R, evs, S[col], m)
                    cc += c
                    dd += d
                    tt += t
                    nn += int(m.sum())
                recs.append(dict(snap=si, tau=tau, set=sname, variant=vname, subset=subset, strat="agebin",
                                 n=nn, n_ev=-1, conc=cc, disc=dd, tie=tt))
        if si % 50 == 0:
            log(f"[{meta['trace']}] snapshot {si}/{len(taus)} tau={tau:.0f} idle={len(idle)}")
    sc = pd.DataFrame(recs)
    sc.to_csv(os.path.join(rd, f"snap_counts{tag}_{meta['trace']}.csv.gz"), index=False)
    return sc, set_sids, set_sids_ge2, set_info, len(taus)


def pooled(df):
    tot = df["conc"].sum() + df["disc"].sum() + df["tie"].sum()
    return (df["conc"].sum() + 0.5 * df["tie"].sum()) / tot if tot else float("nan"), int(tot)


def summarize_cindex(sc, set_sids, set_sids_ge2, set_info, n_grid, meta, fname="x2_cindex.csv"):
    o = Out(fname, ["trace", "set", "strat", "subset", "score", "C", "n_pairs", "C_lo95", "C_hi95",
                              "dC_vs_S0", "dC_lo95", "dC_hi95", "C_half1", "C_half2", "C_snap_median", "C_flipped",
                              "n_snap", "n_grid", "n_sessions_set", "n_sessions_set_ge2", "mean_set_size", "mean_events",
                              "frac_snap_all_idle_fit", "note"])
    rng = np.random.default_rng(12345)
    for (sname, strat), g in sc.groupby(["set", "strat"]):
        snaps = np.sort(g["snap"].unique())
        ns = len(snaps)
        boots = rng.integers(0, ns, size=(N_BOOT, ns))
        # 每個 (variant, subset) 的逐快照計數矩陣
        mats = {}
        for (v, sub), gg in g.groupby(["variant", "subset"]):
            gg = gg.set_index("snap").reindex(snaps).fillna(0)
            mats[(v, sub)] = (gg["conc"].values.astype(float), gg["disc"].values.astype(float),
                              gg["tie"].values.astype(float), gg["n"].values, gg["n_ev"].values)

        def bootC(key):
            c, d, t = mats[key][:3]
            num = (c + 0.5 * t)[boots].sum(1)
            den = (c + d + t)[boots].sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                return num / den

        half = ns // 2
        for (v, sub), (c, d, t, n, ne) in mats.items():
            den = c + d + t
            C = (c.sum() + 0.5 * t.sum()) / den.sum() if den.sum() else float("nan")
            Cf = (d.sum() + 0.5 * t.sum()) / den.sum() if den.sum() else float("nan")
            bc = bootC((v, sub))
            h1 = (c[:half].sum() + 0.5 * t[:half].sum()) / den[:half].sum() if den[:half].sum() else float("nan")
            h2 = (c[half:].sum() + 0.5 * t[half:].sum()) / den[half:].sum() if den[half:].sum() else float("nan")
            with np.errstate(invalid="ignore", divide="ignore"):
                per = (c + 0.5 * t) / den
            base = ("S0", sub)
            if v != "S0" and base in mats:
                cb, db, tb = mats[base][:3]
                Cb = (cb.sum() + 0.5 * tb.sum()) / (cb + db + tb).sum()
                dC = C - Cb
                bd = bc - bootC(base)
                dlo, dhi = np.nanpercentile(bd, [2.5, 97.5])
            else:
                dC = dlo = dhi = float("nan")
            fit = set_info.get(sname)
            o.row(trace=meta["trace"], set=sname, strat=strat, subset=sub, score=v, C=C, n_pairs=int(den.sum()),
                  C_lo95=float(np.nanpercentile(bc, 2.5)), C_hi95=float(np.nanpercentile(bc, 97.5)),
                  dC_vs_S0=dC, dC_lo95=float(dlo), dC_hi95=float(dhi), C_half1=h1, C_half2=h2,
                  C_snap_median=float(np.nanmedian(per)), C_flipped=Cf, n_snap=ns, n_grid=n_grid,
                  n_sessions_set=len(set_sids.get(sname, ())), n_sessions_set_ge2=len(set_sids_ge2.get(sname, ())),
                  mean_set_size=float(np.mean(n)), mean_events=float(np.mean(ne)) if strat == "none" else float("nan"),
                  frac_snap_all_idle_fit=(float(np.mean(fit)) if fit else float("nan")),
                  note=("primary" if (sname == PRIMARY and strat == "none") else ""))


def global_cindex(req, meta, T, ev):
    o = Out("x2_cindex_global.csv", ["trace", "subset", "score", "C", "n_pairs", "C_flipped", "n_episodes", "note"])
    pg = req["prev_gap"].values
    S = dict(S0=np.zeros(len(T)), S1=pg, S2=-req["turn"].values.astype(float), S3=req["out"].values)
    ge2 = req["turn"].values >= 2
    subs = [("full", np.ones(len(T), bool)), ("ge2", ge2)]
    if "grp" in req.columns:
        for g_ in ("user", "tool"):
            gm = req["grp"].values == g_
            subs += [(f"full|wait_{g_}", gm), (f"ge2|wait_{g_}", ge2 & gm)]
    for sub, m in subs:
        for k, s in S.items():
            if k == "S1" and sub.startswith("full"):
                continue
            mm = m & np.isfinite(s)
            C, c, d, t = cindex(T[mm], ev[mm], s[mm])
            tot = c + d + t
            o.row(trace=meta["trace"], subset=sub, score=k, C=C, n_pairs=int(tot),
                  C_flipped=((d + 0.5 * t) / tot if tot else float("nan")), n_episodes=int(mm.sum()),
                  note="回合結束當下（a=0）；S0 全部同分＝0.5；S4＝S1")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace")
    ap.add_argument("--measure-wildchat-cpt", action="store_true")
    ap.add_argument("--cpt-run", help="量 chars/token 的 run_id（讀它 run 目錄的 wildchat_cpt.json）")
    ap.add_argument("--no-cindex", action="store_true")
    ap.add_argument("--fi-block-check", action="store_true", help="FreeInference 的 block_ids 斷言（逐週平行，4 個 process）")
    args = ap.parse_args()
    rd = run_dir()
    if args.fi_block_check:
        o_v = Out("x2_validation.csv", ["trace", "check", "value", "threshold", "passed", "note"])
        checks = x2_load.freeinference_block_check()
        for ck in checks:
            o_v.row(trace="freeinference", **ck)
        log(checks)
        return 0 if all(c["passed"] for c in checks) else 5
    if args.measure_wildchat_cpt:
        r, info = x2_load.measure_wildchat_cpt()
        info["chars_per_token"] = r
        json.dump(info, open(os.path.join(rd, "wildchat_cpt.json"), "w"), indent=1)
        Out("x2_validation.csv", ["trace", "check", "value", "threshold", "passed", "note"]).row(
            trace="wildchat_1m", check="chars_per_token(Llama-3.1 tokenizer, random sample)", value=r,
            threshold="measure", passed=True, note=json.dumps(info, ensure_ascii=False))
        log("chars/token", r, info)
        return 0
    kw = {}
    if args.trace == "wildchat_1m":
        info = json.load(open(os.path.join(os.environ.get("TIARA_RUNS", "/ssd8/hungwei/paper-hkv/runs"), args.cpt_run,
                                           "wildchat_cpt.json")))
        kw["chars_per_token"] = info["chars_per_token"]
    o_v = Out("x2_validation.csv", ["trace", "check", "value", "threshold", "passed", "note"])
    try:
        req, meta, checks = x2_load.LOADERS[args.trace](**kw)
    except x2_load.ValidationError as e:
        for ck in e.checks:
            o_v.row(trace=args.trace, **ck)
        log("VALIDATION FAILED:", e)
        traceback.print_exc()
        return 5
    for ck in checks:
        o_v.row(trace=meta["trace"], **ck)
    log("meta", meta)
    json.dump(meta, open(os.path.join(rd, f"meta_{meta['trace']}.json"), "w"), indent=1, default=str)
    is_wc = meta["trace"] == "wildchat_1m"
    T, ev = episodes(req, meta)
    describe(req, meta, T, ev, is_wc)
    if not ev.any():
        log("沒有任何回來的回合：只寫描述統計")
        return 0
    global_cindex(req, meta, T, ev)
    if args.no_cindex or meta["trace"] == "cachewise":
        return 0
    sc, ss, ss2, info, n_grid = run_cindex(req, meta, is_wc, rd)
    summarize_cindex(sc, ss, ss2, info, n_grid, meta)
    log("done", meta["trace"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
