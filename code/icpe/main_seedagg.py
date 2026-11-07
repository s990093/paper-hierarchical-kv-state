"""main_seedagg.py — icpe-main：同一個設定的 3 個 seed（trace 名結尾 `_s1`／`_s2`／`_s3`）逐格取中位數和最小–最大（只報告）。

  python -I code/icpe/main_seedagg.py --headroom <headroom_components.csv> ... --label <標籤> --out-dir <dir> --out-prefix <prefix> \
      [--seeds 1 2 3] [--verdict-exclude O-PF]

用途（使用者 2026-11-07）：對話資料集（偏離 13）與 agent 軌跡（偏離 17）「每個 seed 當一個變體各自判，另報每格 3 個 seed 的中位數和
最小–最大」。判定照 main_summary.py（每個 seed 各自）；這支只出並列的彙總，不下判定。上限＝next_sim_analyze／analyze_v8 的 headroom 欄，不重算。

做法：
  - 設定＝trace 去掉 `_s<seed>`（例：nb:miniswe_N16_s2 → nb:miniswe_N16）。
  - 格＝(設定, rule, cpu_GiB, slow, gpu, restore, gpu_GiB, gpu_write, slow_cap_frac, component)。
  - 每格必須剛好有 --seeds 的每個 seed 各一列，不然就停（不補、不跳過）。
輸出（每列有 run_id、ts、host、label）：
  <prefix>_cells.csv     每格：3 個 seed 的上限、中位數、最小、最大、幾個 seed ≥15%
  <prefix>_summary.csv   每個 (設定, rule, 元件)：格數、中位數的最大值（和那格的最小–最大）、中位數 ≥15% 的格數、
                         至少 1 個／全部 seed ≥15% 的格數
  <prefix>_setting.csv   每個 (設定, rule)，只用判定元件（O-ALL(report) 不用；--verdict-exclude 再拿掉）：同上，跨元件
"""
from __future__ import annotations

import argparse
import os
import re
import socket
import sys
from datetime import datetime

import pandas as pd

JUDGED = ["O-ORD", "O-DROP", "O-NOW", "O-PF", "O-GPU"]
THR = 0.15
CELL = ["setting", "rule", "cpu_GiB", "slow", "gpu", "restore", "gpu_GiB", "gpu_write", "slow_cap_frac", "component"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headroom", nargs="+", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--verdict-exclude", nargs="*", default=[])
    a = ap.parse_args()
    assert set(a.verdict_exclude) <= set(JUDGED)
    meta = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"),
                host=socket.gethostname(), label=a.label)
    if a.verdict_exclude:
        meta["verdict_exclude"] = ";".join(a.verdict_exclude)
    H = pd.concat([pd.read_csv(p).assign(src_file=p) for p in a.headroom], ignore_index=True)
    m = H.trace.str.extract(r"^(?P<setting>.+)_s(?P<seed>\d+)$")
    if m.setting.isna().any():
        print(f"trace 名不是 ..._s<seed>：{sorted(H.trace[m.setting.isna()].unique())[:5]}：停", file=sys.stderr)
        sys.exit(1)
    H["setting"], H["seed"] = m.setting, m.seed.astype(int)
    assert not H.duplicated(CELL + ["seed"]).any(), "同一格同一個 seed 有重複列"
    want = sorted(a.seeds)
    S = H.groupby(CELL, sort=True).seed.apply(lambda s: sorted(s.tolist()))
    bad = S[S.map(lambda s: s != want)]
    if len(bad):
        print(f"{len(bad)} 格的 seed 不是 {want}：停；例 {bad.head(3).to_dict()}", file=sys.stderr)
        sys.exit(1)
    W = H.pivot_table(index=CELL, columns="seed", values="headroom", aggfunc="first")
    W.columns = [f"headroom_s{s}" for s in W.columns]
    sc = [f"headroom_s{s}" for s in want]
    W["median"] = W[sc].median(axis=1)
    W["min"] = W[sc].min(axis=1)
    W["max"] = W[sc].max(axis=1)
    W["n_seeds_ge15"] = (W[sc] >= THR).sum(axis=1)
    W = W.reset_index()

    def summ(g):
        i = g["median"].idxmax()
        return pd.Series(dict(n_cells=len(g), max_median=g.at[i, "median"], argmax_min=g.at[i, "min"], argmax_max=g.at[i, "max"],
                              argmax_component=g.at[i, "component"], argmax_cpu_GiB=g.at[i, "cpu_GiB"], argmax_slow=g.at[i, "slow"],
                              argmax_gpu=g.at[i, "gpu"], n_median_ge15=int((g["median"] >= THR).sum()),
                              n_any_seed_ge15=int((g.n_seeds_ge15 >= 1).sum()), n_all_seeds_ge15=int((g.n_seeds_ge15 == len(want)).sum())))

    SM = W.groupby(["setting", "rule", "component"], sort=True)[W.columns.tolist()].apply(summ).reset_index()
    use = [c for c in JUDGED if c not in a.verdict_exclude]
    ST = W[W.component.isin(use)].groupby(["setting", "rule"], sort=True)[W.columns.tolist()].apply(summ).reset_index()
    ST.insert(2, "components_used", ";".join(use))
    os.makedirs(a.out_dir, exist_ok=True)
    for name, X in (("cells", W), ("summary", SM), ("setting", ST)):
        X = X.copy()
        for i, (k, v) in enumerate(meta.items()):
            X.insert(i, k, v)
        X.to_csv(os.path.join(a.out_dir, f"{a.out_prefix}_{name}.csv"), index=False)
    print(f"seed 齊全 PASS：{W.setting.nunique()} 個設定、{len(W)} 格，每格 seed {want}")
    pd.set_option("display.width", 250)
    print(ST[["setting", "rule", "n_cells", "max_median", "argmax_min", "argmax_max", "argmax_component", "n_median_ge15",
              "n_any_seed_ge15", "n_all_seeds_ge15"]].to_string(index=False))


if __name__ == "__main__":
    main()
