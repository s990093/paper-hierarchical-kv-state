"""x3_extra.py — X3 的補充分析（只報告，不改判定）：
  cond     條件在「閒置夠久」（才可能已被降級到慢層）的回來事件上，h 的分布：思考時間 ≥ 10／60／300 s；CacheWise 工具 ≥ 1／10／60 s
  bylen    h 依要還原的長度分箱（block 數），找提示在哪裡不夠
  wcts     WildChat 時間戳的語意檢查：「上一輪 assistant → 這一輪 assistant」的間隔和「這一輪」還是「上一輪」的輸出長度比較相關
主設定：Llama、GPU 閒、桌機打字速度、TPOT 50 ms、新 token＝原始增量 − 10（x3_hide 的 leads()）。
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from x3_hide import OUT, hide_stats, leads, load_events, load_f_llama, load_tables, ts  # noqa: E402

TIERS = ("nfs", "llama_b48", "llama_b32", "bw4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--events", nargs="+", required=True)
    a = ap.parse_args()
    tabs, _ = load_tables(a.tables)
    consts = json.load(open(os.path.join(OUT, "x3_typing_consts.json")))
    f = load_f_llama()
    evs = load_events(a.events)
    t0 = ts()
    cond, bylen = [], []
    for name, d in evs.items():
        L, n_blk = leads(name, d, consts, f)
        if name.startswith("cachewise_tool") or name == "freeinference_tool":
            idle, thr, sigs = L["c_tool"], (0, 1, 10, 60), ("c_tool",)
        else:
            idle, thr = L["b_think"], (0, 10, 60, 300)
            sigs = ("a_typing", "b_think") if "a_typing" in L else ("b_think",)
        for th in thr:
            sel = np.isfinite(idle) & (idle >= th)
            for tier in TIERS:
                for prec in ("int4", "bf16"):
                    tab = tabs[("llama", tier, prec, "idle")]
                    for sig in sigs:
                        r = dict(trace=name, idle_ge_s=th, frac_events_selected=round(float(sel.mean()), 4),
                                 sessions_selected=int(d.session[sel].nunique()), tier=tier, GiBps=round(tab["g"], 4),
                                 prec=prec, signal=sig)
                        r.update(hide_stats(n_blk[sel], L[sig][sel], tab, "full"))
                        cond.append(r)
        bins = [(1, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 256)]
        for lo, hi in bins:
            sel = (n_blk >= lo) & (n_blk <= hi)
            if sel.sum() == 0:
                continue
            for tier in TIERS:
                for prec in ("int4", "bf16"):
                    tab = tabs[("llama", tier, prec, "idle")]
                    for sig in sigs:
                        r = dict(trace=name, n_lo=lo, n_hi=hi, frac_events=round(float(sel.mean()), 4), tier=tier,
                                 GiBps=round(tab["g"], 4), prec=prec, signal=sig)
                        r.update(hide_stats(n_blk[sel], L[sig][sel], tab, "full"))
                        bylen.append(r)
        print("done", name, flush=True)
    for rs, fn in ((cond, "x3_cond_idle.csv"), (bylen, "x3_bylen.csv")):
        df = pd.DataFrame(rs)
        df.insert(0, "ts", t0)
        df.insert(0, "run_id", a.run_id)
        df.to_csv(os.path.join(OUT, fn), index=False)
        print("wrote", fn, len(df))
    # WildChat 時間戳語意
    if "wildchat_en" in evs:
        d = evs["wildchat_en"].sort_values(["session", "turn"])
        d["prev_gen"] = d.groupby("session").gen_tokens.shift(1)
        x = d.dropna(subset=["prev_gen"])
        x = x[(x.gap_s > 0)]
        r_cur = float(np.corrcoef(np.log1p(x.gap_s), np.log1p(x.gen_tokens))[0, 1])
        r_prev = float(np.corrcoef(np.log1p(x.gap_s), np.log1p(x.prev_gen))[0, 1])
        lo = x.gap_s / x.gen_tokens.clip(lower=1)
        lo_p = x.gap_s / x.prev_gen.clip(lower=1)
        big = x[x.gen_tokens >= 300]
        out = pd.DataFrame([dict(run_id=a.run_id, ts=t0, n=len(x), corr_log_gap_vs_log_this_turn_output=round(r_cur, 4),
                                 corr_log_gap_vs_log_prev_turn_output=round(r_prev, 4),
                                 p1_gap_per_this_output_token_s=round(float(np.percentile(lo[x.gen_tokens >= 50], 1)), 4),
                                 p1_gap_per_prev_output_token_s=round(float(np.percentile(lo_p[x.prev_gen >= 50], 1)), 4),
                                 frac_gap_lt_this_output_x20ms_for_ge300tok=round(float((big.gap_s < big.gen_tokens * 0.02).mean()), 4),
                                 note="若 assistant 時間戳＝回應結束，間隔應含這一輪的生成時間：和這一輪輸出長度較相關、且每 token 的下緣不會趨近 0")])
        out.to_csv(os.path.join(OUT, "x3_wildchat_tscheck.csv"), index=False)
        print(out.T)


if __name__ == "__main__":
    main()
