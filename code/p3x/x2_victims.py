"""x2_victims.py — X2 事後（只報告）：降級當下「挑最晚回來的那幾個」挑得準不準。

C-index 看整個排序；降級只需要挑出集合裡最晚回來的那一小撮。這支在每個快照的 LRU 集合裡，
照分數挑出「預測最晚回來」的前 k 個（k＝集合大小的 10%，至少 1 個），看它們之中：
  frac_ret_300s：300 s 內就回來的比例（挑錯、而且很快就要重新還原）
  frac_ret_end ：trace 結束前回來的比例
和「隨機挑」（＝集合的基準率）、oracle（照真實剩餘時間挑，設限的視為最晚）並列。
分數：S0（LRU）、S2lex（2Q 式：先比請求數再比閒置時間）。沒有參數。
用法：TIARA_NO_GPU=1 m7run p3x2-vic-<短名> python -I code/p3x/x2_victims.py --trace <trace> [--cpt-run <RUN_ID>]
產出：results/p3_explore/x2/x2_victims.csv
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

FRAC = 0.10


def victims(req, meta):
    is_wc = meta["trace"] == "wildchat_1m"
    taus, los, wks = M.snapshot_grid(meta, is_wc, req)
    req = req.iloc[np.argsort(M.t_idle(req), kind="mergesort")].reset_index(drop=True)
    t_all = M.t_idle(req)
    nxt = np.where(np.isnan(req["next_t"].values), np.inf, req["next_t"].values)
    cens_at = req["cens_at"].values if "cens_at" in req.columns else np.full(len(req), meta["t_end"])
    size_b = req["size_tok"].values * M.BYTES_PER_TOK
    has_grp = "grp" in req.columns
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
        sub = req.iloc[idle]
        a, S = M.scores(sub, tau)
        ev = np.isfinite(nxt[idle])
        R = np.where(ev, nxt[idle] - tau, np.inf)  # 設限的視為最晚（oracle 用）
        o = np.argsort(a, kind="mergesort")
        cum = np.cumsum(size_b[idle][o])
        sets = {}
        for C in M.CAPS_GIB:
            m = np.zeros(len(idle), bool)
            m[o[cum <= C * M.GiB]] = True
            sets[f"lru{C}GiB"] = m
            if has_grp:
                for g_ in ("user", "tool"):
                    sets[f"lru{C}GiB|wait_{g_}"] = m & (sub["grp"].values == g_)
        r300 = ev & (R <= 300)
        for sname, sm in sets.items():
            ii = np.nonzero(sm)[0]
            n = len(ii)
            if n < 2:
                continue
            k = max(1, int(np.ceil(FRAC * n)))
            d = acc.setdefault(sname, {nm: [[], []] for nm in ("random", "S0", "S2lex", "oracle")})
            d["random"][0].append(r300[ii].mean())
            d["random"][1].append(ev[ii].mean())
            for nm, sc in (("S0", S["S0"][ii]), ("S2lex", S["S2lex"][ii]), ("oracle", R[ii])):
                pick = ii[np.argsort(-sc, kind="mergesort")[:k]]
                d[nm][0].append(r300[pick].mean())
                d[nm][1].append(ev[pick].mean())
    o = M.Out("x2_victims.csv", ["trace", "set", "policy", "n_snap", "frac_ret_300s", "frac_ret_end", "note"])
    for sname, d in acc.items():
        for nm, (x3, xe) in d.items():
            o.row(trace=meta["trace"], set=sname, policy=nm, n_snap=len(x3), frac_ret_300s=float(np.mean(x3)),
                  frac_ret_end=float(np.mean(xe)),
                  note=f"每個快照挑預測最晚回來的前 {int(FRAC*100)}%；值＝快照平均；random＝集合的基準率")
    M.log("victims done", meta["trace"])


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
    victims(req, meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
