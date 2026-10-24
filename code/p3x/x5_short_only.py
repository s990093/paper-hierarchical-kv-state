"""x5_short_only.py — X5 §4.5：只算 context ≤ 32K（k_eff ≤ 64）的回來請求，重算主設定 headroom 的格子。
目的：INT4 只在 32K 驗過品質，看判定是不是靠 > 32K 的回來請求。
用法：RUN_ID=<id> python -I code/p3x/x5_short_only.py --out-dir results/p3_explore/x5 \
        --sets "subtree::<grid run dir>" "strict:_strict:<strict run dir>"
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x5_trace as XT  # noqa: E402

RID = os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-p3x5-short-adhoc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sets", nargs="+", required=True, help="rule:sfx:rawdir")
    a = ap.parse_args()
    rows = []
    for spec in a.sets:
        rule, sfx, raw = spec.split(":")
        h = pd.read_csv(os.path.join(a.out_dir, f"x5_headroom{sfx}.csv"))
        z = np.load(os.path.join(raw, f"ttft{sfx}.npz"))
        m = h[(h.model == "llama") & (h.dq == "k0") & (h.variant == "reg")]
        trs = {nm: XT.load(nm, rule) for nm in m.trace.unique()}
        for _, r in m.iterrows():
            tr = trs[r.trace]
            k = tr["keff"][tr["ret"]]
            short = k <= 64
            base = f"{r.trace}|{r.model}|{r.cpu_GiB}|{r.variant}|{r.dq}|{r.gpu}|{r.slow}|"
            x = z[base + r.best_rule].astype(np.float64)
            o = z[base + r.best_oracle].astype(np.float64)
            rows.append(dict(run_id=RID, ts=datetime.now().astimezone().isoformat(timespec="seconds"), rule=rule,
                             trace=r.trace, cpu_GiB=r.cpu_GiB, slow=r.slow, gpu=r.gpu, best_rule=r.best_rule,
                             best_oracle=r.best_oracle, headroom_all=r.headroom,
                             headroom_le32k=(x[short].mean() - o[short].mean()) / x[short].mean(),
                             frac_returns_gt32k=float(1 - short.mean()), n_le32k=int(short.sum())))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(a.out_dir, "x5_headroom_le32k.csv"), index=False, float_format="%.6f")
    for rule, g in d.groupby("rule"):
        gg = g[g.headroom_all >= 0.15]
        print(rule, "cells>=15% (all):", len(gg), "-> le32k range", round(gg.headroom_le32k.min(), 4),
              round(gg.headroom_le32k.max(), 4), "still>=15%:", int((gg.headroom_le32k >= 0.15).sum()),
              "| any main cell le32k>=15%:", int((g.headroom_le32k >= 0.15).sum()))


if __name__ == "__main__":
    main()
