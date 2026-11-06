"""main_pair.py — icpe-main：敏感度（P2／P3）和版本 A 的逐格配對（每組只換一個設定；使用者 2026-11-06）。

  python -I code/icpe/main_pair.py --name <名稱> --x <敏感度的 headroom_components.csv ...> --y <對照的 headroom_components.csv ...> \
      [--map COL=FROM:TO ...] [--by COL ...] --out-dir <dir> --out-prefix <p>

只做配對與相減，不重算任何上限（上限＝next_sim_analyze.py 的 headroom 欄）。
配法：
  - x 的每一列（格, 元件）先照 --map 換座標（例：restore=load:cake、gpu_write=wt:wb、slow=nvme807:ssd；只換 COL＝FROM 的列），
    再用 (格, 元件) 對到 y 的同一列。格＝(trace, rule, cpu_GiB, slow, gpu, restore, gpu_GiB, gpu_write, slow_cap_frac)。
  - x 的每一列都必須在 y 找到，而且只對到一列；換完座標後 x 不能有重複。不成立就照實印出、exit 1（不換方法硬配）。
  - y 多出來、沒被配到的列不算錯（y 是比較大的那份），另報列數。
  - 只用主指標的表（metric＝mean_ret、dead_h＝0）；兩邊都檢查。
輸出（每列有 run_id、ts、host）：
  <p>_pairs.csv          每個（格, 元件）一列：x 的原座標、headroom_x、headroom_y、diff＝x − y、跨 15% 的方向
                         （up＝y<0.15≤x；down＝x<0.15≤y；none）、兩邊的 write_saturated
  <p>_pairsum.csv        每個 (紀錄, rule, 元件[, --by]) 的配對數、diff 的最小／中位數／平均／最大、x>y／x<y／x＝y 的格數、
                         up／down 的格數、兩邊 ≥15% 的格數
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
THR = 0.15


def load(paths):
    parts = []
    for p in paths:
        h = pd.read_csv(p)
        assert h.run_id.nunique() == 1, f"{p}：run_id 不只一個"
        assert set(h.metric) == {"mean_ret"} and set(h.dead_h) == {0}, f"{p}：不是主指標、不截尾的上限表"
        h = h.rename(columns={"run_id": "src_analysis_run_id"})
        h["src_file"] = p
        parts.append(h)
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--x", nargs="+", required=True)
    ap.add_argument("--y", nargs="+", required=True)
    ap.add_argument("--map", nargs="*", default=[], metavar="COL=FROM:TO")
    ap.add_argument("--by", nargs="*", default=[])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args()
    meta = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"),
                host=socket.gethostname(), name=a.name, map=";".join(a.map) or "none")
    X, Y = load(a.x), load(a.y)
    K = CELL + ["component"]
    assert not Y.duplicated(K).any(), f"y 的 (格, 元件) 有 {int(Y.duplicated(K).sum())} 列重複"
    assert not X.duplicated(K).any(), f"x 的 (格, 元件) 有 {int(X.duplicated(K).sum())} 列重複"
    Xm = X.copy()
    for c in CELL:
        Xm["x_" + c] = X[c]
    n_mapped = {}
    for spec in a.map:
        col, ft = spec.split("=", 1)
        f, t = ft.split(":", 1)
        assert col in CELL, f"--map 的欄 {col} 不是格的座標"
        isnum = pd.api.types.is_numeric_dtype
        assert not isnum(X[col]) and not isnum(Y[col]), f"--map 只換文字欄（{col} 是數值欄）"
        m = X[col] == f
        assert m.any(), f"--map {spec}：x 沒有任何一列 {col}＝{f}"
        n_mapped[spec] = int(m.sum())
        Xm.loc[m, col] = t
    assert not Xm.duplicated(K).any(), f"換座標後 x 的 (格, 元件) 有 {int(Xm.duplicated(K).sum())} 列重複：--map 把不同的格換成同一格"
    keep_x = ["x_" + c for c in CELL] + K + ["headroom", "write_saturated", "src_analysis_run_id", "src_file"]
    keep_y = K + ["headroom", "write_saturated", "src_analysis_run_id", "src_file"]
    M = Xm[keep_x].merge(Y[keep_y], on=K, how="left", suffixes=("_x", "_y"), indicator=True)
    miss = M[M._merge != "both"]
    if len(miss):
        print(f"PAIR {a.name}: FAIL — x 有 {len(miss)} 列在 y 找不到對應的格（前 5 列）：")
        print(miss[["x_" + c for c in CELL] + ["component"]].head().to_string(index=False))
        sys.exit(1)
    assert len(M) == len(X), "配對後列數和 x 不同（y 有重複？）"
    n_y_unused = len(Y) - len(M)
    M["diff"] = M.headroom_x - M.headroom_y
    M["cross15"] = "none"
    M.loc[(M.headroom_y < THR) & (M.headroom_x >= THR), "cross15"] = "up"
    M.loc[(M.headroom_x < THR) & (M.headroom_y >= THR), "cross15"] = "down"
    cl = {t: classify(t) for t in M.trace.unique()}
    M["trace_class"] = M.trace.map(lambda t: cl[t][0])
    M["arrival"] = M.trace.map(lambda t: cl[t][1])
    M["tool_type"] = M.trace.map(lambda t: cl[t][3])
    P = M[["trace_class", "arrival", "tool_type"] + ["x_" + c for c in CELL] + K + ["headroom_x", "headroom_y", "diff", "cross15",
          "write_saturated_x", "write_saturated_y", "src_analysis_run_id_x", "src_file_x", "src_analysis_run_id_y", "src_file_y"]]
    P = P.rename(columns={c: "y_" + c for c in CELL}).rename(columns={"x_" + c: c for c in CELL})
    gk = ["trace_class", "arrival", "tool_type", "trace", "rule", "component"] + a.by
    S = (P.groupby(gk, sort=True)
         .agg(n_pairs=("diff", "size"), diff_min=("diff", "min"), diff_median=("diff", "median"), diff_mean=("diff", "mean"),
              diff_max=("diff", "max"), n_x_gt_y=("diff", lambda s: int((s > 0).sum())), n_x_lt_y=("diff", lambda s: int((s < 0).sum())),
              n_x_eq_y=("diff", lambda s: int((s == 0).sum())), n_up=("cross15", lambda s: int((s == "up").sum())),
              n_down=("cross15", lambda s: int((s == "down").sum())), n_ge15_x=("headroom_x", lambda s: int((s >= THR).sum())),
              n_ge15_y=("headroom_y", lambda s: int((s >= THR).sum())), max_x=("headroom_x", "max"), max_y=("headroom_y", "max"))
         .reset_index())
    os.makedirs(a.out_dir, exist_ok=True)
    for name, T in (("pairs", P), ("pairsum", S)):
        T = T.copy()
        for i, (k, v) in enumerate(meta.items()):
            T.insert(i, k, v)
        T.to_csv(os.path.join(a.out_dir, f"{a.out_prefix}_{name}.csv"), index=False)
    print(f"PAIR {a.name}: PASS — {len(P)} 對（x {len(X)} 列全部配到；y 沒被配到 {n_y_unused} 列）；換座標 {n_mapped or 'none'}；"
          f"x>y {int((P['diff'] > 0).sum())}、x<y {int((P['diff'] < 0).sum())}、x＝y {int((P['diff'] == 0).sum())}；"
          f"跨 15% up {int((P.cross15 == 'up').sum())}、down {int((P.cross15 == 'down').sum())}")
    print(f"→ {a.out_dir}/{a.out_prefix}_{{pairs,pairsum}}.csv")


if __name__ == "__main__":
    main()
