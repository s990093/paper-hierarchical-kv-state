"""x2_setspan.py — X2 事後的描述統計（只報告）：每個快照集合裡，閒置時間 a 涵蓋多長。

為什麼加：主結果裡 LRU（S0）的 C 在 Mooncake／Bailian 的 256 GiB 集合只有 0.51–0.62，在 WildChat 卻是 0.75。
猜測是集合涵蓋的閒置時間長短不同（流量大的叢集，256 GiB 只裝得下最近幾分鐘）。這支只量涵蓋範圍，不算任何分數。
用法：TIARA_NO_GPU=1 m7run p3x2-span-<短名> python -I code/p3x/x2_setspan.py --trace <trace> [--cpt-run <RUN_ID>]
產出：results/p3_explore/x2/x2_setspan.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x2_load  # noqa: E402
import x2_main as M  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--cpt-run")
    a = ap.parse_args()
    kw = {}
    if a.trace == "wildchat_1m":
        info = json.load(open(os.path.join(os.environ.get("TIARA_RUNS", "/ssd8/hungwei/paper-hkv/runs"), a.cpt_run,
                                           "wildchat_cpt.json")))
        kw["chars_per_token"] = info["chars_per_token"]
    req, meta, _ = x2_load.LOADERS[a.trace](**kw)
    setspan(req, meta)
    return 0


def setspan(req, meta):
    is_wc = meta["trace"] == "wildchat_1m"
    taus, los, wks = M.snapshot_grid(meta, is_wc, req)
    req = req.iloc[np.argsort(M.t_idle(req), kind="mergesort")].reset_index(drop=True)
    t_all = M.t_idle(req)
    nxt = np.where(np.isnan(req["next_t"].values), np.inf, req["next_t"].values)
    size_b = req["size_tok"].values * M.BYTES_PER_TOK
    turn = req["turn"].values
    acc = {}
    week = req["week"].values if "week" in req.columns else None
    for tau, lo_t, wk in zip(taus, los, wks):
        lo = np.searchsorted(t_all, lo_t, side="left") if np.isfinite(lo_t) else 0
        hi = np.searchsorted(t_all, tau, side="right")
        idx = np.arange(lo, hi)
        idle = idx[nxt[idx] > tau]
        if wk >= 0:
            idle = idle[week[idle] == wk]
        if len(idle) == 0:
            continue
        aa = tau - t_all[idle]
        o = np.argsort(aa, kind="mergesort")
        cum = np.cumsum(size_b[idle][o])
        ret = np.isfinite(nxt[idle])
        sets = {"all_idle": o}
        for C in M.CAPS_GIB:
            sets[f"lru{C}GiB"] = o[cum <= C * M.GiB]
        for nm, sel in sets.items():
            if len(sel) == 0:
                continue
            d = acc.setdefault(nm, dict(max_a=[], med_a=[], n=[], frac_turn1=[], frac_return=[], gib=[]))
            d["max_a"].append(aa[sel].max())
            d["med_a"].append(np.median(aa[sel]))
            d["n"].append(len(sel))
            d["frac_turn1"].append(np.mean(turn[idle][sel] == 1))
            d["frac_return"].append(np.mean(ret[sel]))
            d["gib"].append(size_b[idle][sel].sum() / M.GiB)
    o = M.Out("x2_setspan.csv", ["trace", "set", "n_snap", "median_max_idle_s", "median_median_idle_s",
                                 "median_n_sessions", "median_frac_turn1", "median_frac_return_before_end",
                                 "median_GiB", "note"])
    for nm, d in acc.items():
        o.row(trace=meta["trace"], set=nm, n_snap=len(d["n"]), median_max_idle_s=float(np.median(d["max_a"])),
              median_median_idle_s=float(np.median(d["med_a"])), median_n_sessions=float(np.median(d["n"])),
              median_frac_turn1=float(np.median(d["frac_turn1"])),
              median_frac_return_before_end=float(np.median(d["frac_return"])), median_GiB=float(np.median(d["gib"])),
              note="各快照的值取中位數；frac_return＝集合裡 trace 結束前會回來的比例")
    M.log("setspan done", meta["trace"])


if __name__ == "__main__":
    sys.exit(main())
