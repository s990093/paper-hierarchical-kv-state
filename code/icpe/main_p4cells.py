"""main_p4cells.py — icpe-main：P4（W4 雜訊）要跑的格子清單（PREREG §3 B、偏離 6 的 P4、偏離 8；使用者 2026-11-06）。

  python -I code/icpe/main_p4cells.py --headroom <family>=<版本 B 的 headroom_components.csv> ... \
      --components O-ORD O-DROP O-NOW O-PF O-GPU "O-ALL(report)" --out-dir <dir> [--thr 0.15]

選法（只做篩選，不重算上限）：
  - 上限＝next_sim_analyze.py 的 headroom 欄（版本 B＝P1 正式值，主指標 mean_ret）；headroom ≥ thr 的（格, 元件）都列。
  - 元件＝--components（使用者 11/6：5 個判定元件加上 O-ALL(report)）。
  - rule＝min 的列不列：min 是 R1、R2 兩者取小，min ≥ thr 一定代表 R1、R2 都 ≥ thr。程式自檢這件事，不成立就停。
  - 寫入飽和的格照列，另標 write_saturated（偏離 7 第 3 點）。
輸出（每列有 run_id、ts、host）：
  p4_cells.csv           每個（格, 元件）一列：格的座標、headroom、④／② 的做法、write_saturated
  p4_cells_distinct.csv  每個格一列：要加雜訊的元件清單（給 icpe-data 排模擬）
  p4_summary.csv         每個 (紀錄, rule, 元件) 的格數
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main_summary import classify  # noqa: E402

CELL = ["trace", "rule", "cpu_GiB", "slow", "gpu", "restore", "gpu_GiB", "gpu_write", "slow_cap_frac"]
KEEP = ["oracle_policy", "oracle_gpu_policy", "oracle_slow_policy", "T_oracle", "best_rival", "best_rival_gpu_policy",
        "best_rival_slow_policy", "T_rival", "headroom", "write_saturated", "T_base"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headroom", nargs="+", required=True, metavar="FAMILY=PATH")
    ap.add_argument("--components", nargs="+", required=True)
    ap.add_argument("--thr", type=float, default=0.15)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    meta = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"),
                host=socket.gethostname())
    parts = []
    for spec in a.headroom:
        fam, path = spec.split("=", 1)
        h = pd.read_csv(path)
        assert h.run_id.nunique() == 1, f"{path}：run_id 不只一個"
        assert set(h.metric) == {"mean_ret"} and set(h.dead_h) == {0}, f"{path}：不是主指標、不截尾的上限表"
        assert "rival_only_gpu" not in h.columns, f"{path}：是偏離 5 的算法（--rival-only-gpu），不是版本 B 正式值"
        miss = set(a.components) - set(h.component)
        assert not miss, f"{path}：沒有元件 {sorted(miss)}"
        h = h.rename(columns={"run_id": "src_analysis_run_id"})
        h.insert(0, "family_input", fam)
        h["src_file"] = path
        parts.append(h)
    H = pd.concat(parts, ignore_index=True)
    assert not H.duplicated(CELL + ["component"]).any(), "(格, 元件) 有重複"
    G = H[H.component.isin(a.components) & (H.headroom >= a.thr)].copy()

    # 自檢：rule＝min 的每一筆 ≥thr，R1、R2 同一格同一元件都要 ≥thr（所以 min 的列不用另列）
    mn = G[G.rule == "min"]
    sel = G[G.rule != "min"]
    key = [c for c in CELL if c != "rule"] + ["component"]
    have = sel.groupby(key).rule.apply(set)
    bad = [k for k, _ in mn.groupby(key) if not {"R1", "R2"} <= have.get(k, set())]
    assert not bad, f"rule=min ≥{a.thr} 但 R1／R2 沒有都 ≥{a.thr}：{bad[:5]}"
    print(f"自檢 PASS：rule=min ≥{a.thr} 的 {len(mn)} 筆，R1、R2 都在清單裡")

    cl = {t: classify(t) for t in sel.trace.unique()}
    sel = sel.assign(trace_class=sel.trace.map(lambda t: cl[t][0]), arrival=sel.trace.map(lambda t: cl[t][1]),
                     tool_type=sel.trace.map(lambda t: cl[t][3]), judged=sel.component != "O-ALL(report)")
    cols = ["family_input", "trace_class", "arrival", "tool_type"] + CELL + ["component", "judged"] + KEEP + ["src_analysis_run_id", "src_file"]
    C = sel[cols].sort_values(CELL + ["component"]).reset_index(drop=True)
    D = (C.groupby(["family_input", "trace_class", "arrival", "tool_type"] + CELL, sort=True)
         .agg(n_components=("component", "size"), components=("component", lambda s: ";".join(sorted(s))),
              max_headroom=("headroom", "max"), any_write_saturated=("write_saturated", "any")).reset_index())
    S = (C.groupby(["trace_class", "trace", "rule", "component"], sort=True)
         .agg(n_cells=("headroom", "size"), n_write_saturated=("write_saturated", "sum"),
              min_headroom=("headroom", "min"), max_headroom=("headroom", "max")).reset_index())
    sel_note = f"version B (P1 formal), headroom>={a.thr}, components={';'.join(a.components)}, rule=min not listed (implied by R1/R2)"
    os.makedirs(a.out_dir, exist_ok=True)
    for name, X in (("p4_cells", C), ("p4_cells_distinct", D), ("p4_summary", S)):
        X = X.copy()
        for i, (k, v) in enumerate(dict(meta, selection=sel_note).items()):
            X.insert(i, k, v)
        X.to_csv(os.path.join(a.out_dir, f"{name}.csv"), index=False)
    pd.set_option("display.width", 250)
    print(f"P4：{len(C)} 個（格, 元件），{len(D)} 個不同的格；依元件 {C.component.value_counts().to_dict()}")
    print(C.groupby("trace_class").agg(cell_comp=("component", "size")).join(D.groupby("trace_class").size().rename("cells")).to_string())
    print(f"→ {a.out_dir}/{{p4_cells,p4_cells_distinct,p4_summary}}.csv")


if __name__ == "__main__":
    main()
