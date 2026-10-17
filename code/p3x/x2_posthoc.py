"""x2_posthoc.py — X2 事後加的對照（看過主結果才加，只報告、不改判定）。

S2lex：2Q／SLRU 式的固定規則（沒有參數）——先比「目前為止的請求數」（多的預測先回來），同請求數再用 LRU（閒置久的預測晚回來）。
目的：主結果裡 S2（輪數）的 ΔC 大，這裡看「頻率＋近期」這種不用學的經典組合能拿到多少。
用法：TIARA_NO_GPU=1 m7run p3x2-ph-<短名> python -I code/p3x/x2_posthoc.py --trace <trace> [--cpt-run <RUN_ID>]
產出：results/p3_explore/x2/x2_cindex_posthoc.csv（欄位同 x2_cindex.csv）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x2_load  # noqa: E402
import x2_main as M  # noqa: E402

PH_VARIANTS = [("S0", "S0", "full"), ("S2", "S2", "full"), ("S2lex", "S2lex", "full"),
               ("S0", "S0", "ge2"), ("S2", "S2", "ge2"), ("S2lex", "S2lex", "ge2")]
PH_STRATA = [("S0", "S0", "full"), ("S2lex", "S2lex", "full")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--cpt-run")
    ap.add_argument("--full-variants", action="store_true",
                    help="用主分析的全部分數（給 mooncake_conversation_strict 敏感度用）")
    a = ap.parse_args()
    kw = {}
    if a.trace == "wildchat_1m":
        info = json.load(open(os.path.join(os.environ.get("TIARA_RUNS", "/ssd8/hungwei/paper-hkv/runs"), a.cpt_run,
                                           "wildchat_cpt.json")))
        kw["chars_per_token"] = info["chars_per_token"]
    req, meta, checks = x2_load.LOADERS[a.trace](**kw)
    is_wc = meta["trace"] == "wildchat_1m"
    rd = M.run_dir()
    if a.full_variants:
        for ck in checks:
            M.Out("x2_validation.csv", ["trace", "check", "value", "threshold", "passed", "note"]).row(trace=meta["trace"], **ck)
        T, ev = M.episodes(req, meta)
        M.describe(req, meta, T, ev, is_wc)
        sc, ss, ss2, inf, n_grid = M.run_cindex(req, meta, is_wc, rd, variants=M.VARIANTS + PH_VARIANTS[2:3] + PH_VARIANTS[5:6],
                                                strata_variants=M.STRATA_VARIANTS, tag="_posthoc")
    else:
        sc, ss, ss2, inf, n_grid = M.run_cindex(req, meta, is_wc, rd, variants=PH_VARIANTS, strata_variants=PH_STRATA,
                                                tag="_posthoc")
    M.summarize_cindex(sc, ss, ss2, inf, n_grid, meta, fname="x2_cindex_posthoc.csv")
    M.log("posthoc done", meta["trace"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
