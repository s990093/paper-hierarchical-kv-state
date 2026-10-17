"""x1_boot.py — X1 headroom 的 session-cluster bootstrap（只報告，不改判定）。

對 x1_headroom_components.csv 裡（正式判定那一版的）每一格：以「回來的請求所屬的 session」為重抽單位，
成對重抽 oracle 元件與最好對稱對手的逐請求 TTFT（sweep 存在 $RUN_DIR/req_*.npz），
headroom＝(Σ對手 − Σoracle) ÷ Σ對手（同一組重抽的回來請求），2,000 次，報 2.5／97.5 百分位。
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
OUT = os.path.join(HERE, "..", "..", "results", "p3_explore", "x1")


def find_npz(run_dirs, trace, rule, cpu, policy, slow, gpu, mode):
    key = f"ttft_{slow}_{gpu}_{mode}"
    for rd in run_dirs:
        for p in glob.glob(os.path.join(rd, f"req_{trace}_{rule}_{cpu}_{policy}_*.npz")):
            z = np.load(p)
            if key in z:
                return z[key]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headroom", default="x1_headroom_components.csv")
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--rule", default="min")
    ap.add_argument("--restore", default="cake")
    ap.add_argument("--min-headroom", type=float, default=0.05, help="只對 headroom ≥ 這個值的格重抽（省時間）")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--out", default="x1_bootstrap.csv")
    a = ap.parse_args()
    import x1_sim as X
    H = pd.read_csv(os.path.join(OUT, a.headroom))
    H = H[(H.rule == a.rule) & (H.restore == a.restore) & (H.headroom >= a.min_headroom) &
          (~H.trace.str.startswith("fi"))]
    rng = np.random.default_rng(20261017)
    rows = []
    cacheP = {}
    for _, r in H.iterrows():
        ru = r.rule_of_min if "rule_of_min" in r and isinstance(r.rule_of_min, str) else r.rule
        if (r.trace, ru) not in cacheP:
            cacheP[(r.trace, ru)] = X.prep(r.trace, ru)
        P = cacheP[(r.trace, ru)]
        kd = np.array(P["kindc"])
        sess = np.array(P["sessl"])
        ret = np.nonzero(kd == 1)[0]
        to = find_npz(a.run_dirs, r.trace, ru, int(r.cpu_GiB), r.oracle_policy, r.slow, r.gpu, r.restore)
        tr_ = find_npz(a.run_dirs, r.trace, ru, int(r.cpu_GiB), r.best_rival, r.slow, r.gpu, r.restore)
        if to is None or tr_ is None:
            rows.append(dict(**r[["trace", "cpu_GiB", "slow", "gpu", "restore", "component"]].to_dict(), rule_used=ru,
                             note="npz NOT_FOUND"))
            continue
        s_ret = sess[ret]
        uniq, inv = np.unique(s_ret, return_inverse=True)
        so = np.bincount(inv, weights=to[ret])
        sr = np.bincount(inv, weights=tr_[ret])
        point = (sr.sum() - so.sum()) / sr.sum()
        idx = rng.integers(0, len(uniq), size=(a.B, len(uniq)))
        Ro, Rr = so[idx].sum(1), sr[idx].sum(1)
        hb = (Rr - Ro) / Rr
        rows.append(dict(**r[["trace", "cpu_GiB", "slow", "gpu", "restore", "component", "oracle_policy",
                              "best_rival"]].to_dict(), rule_used=ru, n_sessions=len(uniq), n_ret=len(ret),
                         headroom_point=point, headroom_csv=r.headroom, ci_lo=float(np.percentile(hb, 2.5)),
                         ci_hi=float(np.percentile(hb, 97.5)), B=a.B))
    D = pd.DataFrame(rows)
    D.insert(0, "ts", datetime.now().astimezone().isoformat(timespec="seconds"))
    D.insert(0, "run_id", os.environ.get("RUN_ID", "adhoc"))
    D.to_csv(os.path.join(OUT, a.out), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(D.drop(columns=["run_id", "ts"]).to_string())


if __name__ == "__main__":
    main()
