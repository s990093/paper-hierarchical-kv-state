"""main_rowcheck.py — icpe-main：sweep 檔的列數與網格完整性核對（版本 B 開算前；使用者 2026-11-06「先核對列數」）。

  python -I code/icpe/main_rowcheck.py --sweep <x1_sweep.csv.xz ...> --out <rowcheck.csv（附加）> \
      [--expect-cells 54] [--expect-cpu N] [--expect-gpu N] [--label <名稱>]

每個檔、每個 (trace, rule)：
  格數＝(cpu_GiB, slow, gpu, restore, gpu_GiB, gpu_write) 的組合數；CPU 規則數＝policy 種類；GPU 規則數＝gpu_policy 種類；
  列數必須＝格數 × CPU 規則數 × GPU 規則數（完整網格、沒有缺列），(格, policy, gpu_policy, pf_target) 不能重複。
  有給 --expect-* 就再核對那個數。任何一項不對就照實寫進輸出、exit 1（不換方法硬算）。
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
from datetime import datetime

import pandas as pd

CELL = ["cpu_GiB", "slow", "gpu", "restore", "gpu_GiB", "gpu_write"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--expect-cells", type=int, default=54)
    ap.add_argument("--expect-cpu", type=int, default=0)
    ap.add_argument("--expect-gpu", type=int, default=0)
    a = ap.parse_args()
    meta = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"),
                host=socket.gethostname(), label=a.label)
    rows, ok = [], True
    for p in a.sweep:
        d = pd.read_csv(p, dtype=str, keep_default_na=False)
        for (tr, ru), g in d.groupby(["trace", "rule"], sort=True):
            n_cell = g[CELL].drop_duplicates().shape[0]
            n_cpu, n_gpu = g.policy.nunique(), g.gpu_policy.nunique()
            dup = int(g.duplicated(CELL + ["policy", "gpu_policy", "pf_target"]).sum())
            full = len(g) == n_cell * n_cpu * n_gpu
            prob = []
            if not full:
                prob.append(f"rows {len(g)} != cells*cpu*gpu {n_cell * n_cpu * n_gpu}")
            if dup:
                prob.append(f"{dup} duplicate rows")
            if a.expect_cells and n_cell != a.expect_cells:
                prob.append(f"cells {n_cell} != {a.expect_cells}")
            if a.expect_cpu and n_cpu != a.expect_cpu:
                prob.append(f"cpu rules {n_cpu} != {a.expect_cpu}")
            if a.expect_gpu and n_gpu != a.expect_gpu:
                prob.append(f"gpu rules {n_gpu} != {a.expect_gpu}")
            ok &= not prob
            rows.append(dict(meta, file=p, trace=tr, rule=ru, n_rows=len(g), n_cells=n_cell, n_cpu_rules=n_cpu, n_gpu_rules=n_gpu,
                             n_dup=dup, cpu_rules=";".join(sorted(g.policy.unique())), gpu_rules=";".join(sorted(g.gpu_policy.unique())),
                             src_run_id=";".join(sorted(g.run_id.unique())), result="OK" if not prob else "FAIL", problem="; ".join(prob)))
    R = pd.DataFrame(rows)
    R.to_csv(a.out, mode="a", header=not os.path.exists(a.out), index=False)
    bad = R[R.result != "OK"]
    print(f"ROWCHECK {a.label}: {len(R)} (file, trace, rule), {int(R.n_rows.sum())} rows, "
          f"cells {sorted(R.n_cells.unique())} cpu {sorted(R.n_cpu_rules.unique())} gpu {sorted(R.n_gpu_rules.unique())}: "
          f"{'PASS' if ok else 'FAIL'}")
    if len(bad):
        print(bad[["file", "trace", "rule", "problem"]].to_string(index=False))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
