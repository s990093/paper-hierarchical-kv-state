"""main_w3c.py — icpe-main：W3（偏離 9）的判定改用另一份上限表重算 f（偏離 10 第 5 點：「偏離 9 的 W3 用前半同一份表，
所以 T②、T④ 改用版本 C 重算 f」）。不重跑任何模擬：③ 用 TRX40 的 W3 sweep 算好的 T_cue（`w3_rows.csv`），② ④ 用給的上限表。

  python -I code/icpe/main_w3c.py --w3-rows <w3_rows.csv> ... --headroom <headroom_components.csv> ... --label <C|C_noBELP> \
      --out-dir <dir> --out-prefix <prefix>

照偏離 9 第 3、4 點（和補記）：
  - 只判 5 個判定元件（偏離 7）；O-ALL(report) 也算 f、只報告。
  - (格, 元件) 的上限 ≥0.15、而且 ③ 齊全（w3_rows 的 cue_complete）才判；f_c＝(T② − T③) ÷ (T② − T④)。
  - 變體：有任一個判的（格, 元件）f ≤ 0.5 → 後半成立；全部 f > 0.5 且沒有 ③ 不齊的 ≥15% 格 → 後半不成立；
    沒有 ≥15% 的格 → 前半不成立；有 ≥15% 但都 ③ 不齊 → 未定。
  - 寫入飽和照算，另報「≥15%、f ≤ 0.5 的格裡有幾格飽和」。
檢查：每個 w3_rows 的（格, 元件）都要在上限表裡找到，而且 T_cue 不是空的（③ 齊全時）；找不到就停。
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
from datetime import datetime

import numpy as np
import pandas as pd

JUDGED = ["O-ORD", "O-DROP", "O-NOW", "O-PF", "O-GPU"]
KEY = ["trace", "rule", "cpu_GiB", "slow", "gpu", "restore", "component"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w3-rows", nargs="+", required=True)
    ap.add_argument("--headroom", nargs="+", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args()
    meta = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"),
                host=socket.gethostname(), table=a.label)
    W = pd.concat([pd.read_csv(p).assign(src_w3=p) for p in a.w3_rows], ignore_index=True)
    H = pd.concat([pd.read_csv(p).assign(src_headroom=p) for p in a.headroom], ignore_index=True)
    H = H[H.trace.isin(W.trace.unique())]
    assert not H.duplicated(KEY).any(), "上限表的（格, 元件）有重複"
    W = W[["trace", "rule", "cpu_GiB", "slow", "gpu", "restore", "component", "T_cue", "cue_cpu_policy", "cue_gpu_policy",
           "cue_complete", "src_w3"]]
    M = W.merge(H[KEY + ["T_rival", "T_oracle", "headroom", "write_saturated", "src_headroom"]], on=KEY, how="left", indicator=True)
    miss = M[M._merge != "both"]
    if len(miss):
        print(f"w3_rows 有 {len(miss)} 個（格, 元件）在上限表裡找不到：停；例 {miss[KEY].head(3).to_dict('records')}", file=sys.stderr)
        sys.exit(1)
    M = M.drop(columns="_merge")
    M["judged"] = M.component.isin(JUDGED)
    M["ge15"] = M.headroom >= 0.15
    M["f_c"] = np.where(M.cue_complete, (M.T_rival - M.T_cue) / (M.T_rival - M.T_oracle), np.nan)
    M["cue_takes_most"] = M.f_c > 0.5
    if (M.cue_complete & M.T_cue.isna()).any():
        print("cue_complete 但 T_cue 是空的：停", file=sys.stderr)
        sys.exit(1)
    rows = []
    for (tr, ru), g in M[M.judged].groupby(["trace", "rule"], sort=True):
        j = g[g.ge15]
        jc = j[j.cue_complete]
        hit = jc[jc.f_c <= 0.5]
        if len(j) == 0:
            v = "前半不成立（沒有上限 ≥15% 的格）"
        elif len(hit):
            v = "後半成立：沒有簡單線索能拿走大部分"
        elif len(jc) and len(jc) == len(j):
            v = "後半不成立：簡單線索拿走大部分"
        else:
            v = "未定（≥15% 的格 ③ 不齊）"
        rows.append(dict(trace=tr, rule=ru, n_cells_judged=len(g), n_ge15=len(j), n_ge15_cue_complete=len(jc),
                         n_ge15_f_le_half=len(hit), n_ge15_f_le_half_saturated=int(hit.write_saturated.astype(bool).sum()),
                         f_c_median_ge15=(float(jc.f_c.median()) if len(jc) else np.nan),
                         f_c_max_ge15=(float(jc.f_c.max()) if len(jc) else np.nan),
                         n_cue_beats_rival=int((g.T_cue < g.T_rival).sum()), verdict=v))
    V = pd.DataFrame(rows)
    os.makedirs(a.out_dir, exist_ok=True)
    for name, X in (("rows", M), ("verdict", V)):
        X = X.copy()
        for i, (k, val) in enumerate(meta.items()):
            X.insert(i, k, val)
        X.to_csv(os.path.join(a.out_dir, f"{a.out_prefix}_{name}.csv"), index=False)
    pd.set_option("display.width", 250)
    print(V[["trace", "rule", "n_ge15", "n_ge15_f_le_half", "n_cue_beats_rival", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
