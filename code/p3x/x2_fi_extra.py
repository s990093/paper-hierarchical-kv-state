"""x2_fi_extra.py — FreeInference 只載入一次（block_ids 要讀 8 GB），跑三個事後分析（只報告）：
S2lex（x2_posthoc 同定義）、集合涵蓋時間（x2_setspan）、降級挑選（x2_victims）。
用法：TIARA_NO_GPU=1 m7run p3x2-fi-extra python -I code/p3x/x2_fi_extra.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x2_load  # noqa: E402
import x2_main as M  # noqa: E402
import x2_posthoc as P  # noqa: E402
import x2_setspan as SP  # noqa: E402
import x2_victims as V  # noqa: E402


def main():
    req, meta, _ = x2_load.load_freeinference(check_blocks=False)  # block_ids 的斷言在主分析 run 做過
    rd = M.run_dir()
    sc, ss, ss2, inf, n_grid = M.run_cindex(req, meta, False, rd, variants=P.PH_VARIANTS, strata_variants=P.PH_STRATA,
                                            tag="_posthoc")
    M.summarize_cindex(sc, ss, ss2, inf, n_grid, meta, fname="x2_cindex_posthoc.csv")
    SP.setspan(req, meta)
    V.victims(req, meta)
    M.log("fi extra done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
