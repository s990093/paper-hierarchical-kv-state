"""main_overlap.py — icpe-main：同一個設定跑了兩次的 sweep 列逐格核對（例：cw_main_ttl7 對 cw_p1 的 _ttl；使用者 2026-11-06）。

  python -I code/icpe/main_overlap.py --name <名稱> --a <sweep ...> --b <sweep ...> --out <cmp.csv（附加）>

比法：兩邊都以字串讀入；鍵＝(trace, rule, cpu_GiB, policy, slow, gpu, restore, gpu_GiB, gpu_write, gpu_policy, pf_target)
（兩邊都有的才用）；去掉只記錄「誰、何時、在哪、多久、哪版程式」的欄（run_id、ts、host、elapsed_s、code_sha）。
A 的每一列都必須在 B 找到（A 是較小的那份）；重疊的列其餘每一欄字串都要相同。不同就印前幾個例子＋數值最大差，照實寫進 cmp.csv、exit 1。
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
from datetime import datetime

import numpy as np
import pandas as pd

META = ["run_id", "ts", "host", "elapsed_s", "code_sha"]
KEY = ["trace", "rule", "cpu_GiB", "policy", "slow", "gpu", "restore", "gpu_GiB", "gpu_write", "gpu_policy", "pf_target",
       "slow_cap_frac", "slow_policy"]


def load(paths):
    d = pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False) for p in paths], ignore_index=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    A, B = load(a.a), load(a.b)
    key = [k for k in KEY if k in A.columns and k in B.columns]
    val = sorted((set(A.columns) & set(B.columns)) - set(key) - set(META))
    res = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"),
               host=socket.gethostname(), name=a.name, files_a=len(a.a), files_b=len(a.b), rows_a=len(A), rows_b=len(B),
               code_sha_a=";".join(sorted(A.code_sha.unique())) if "code_sha" in A else "",
               code_sha_b=";".join(sorted(B.code_sha.unique())) if "code_sha" in B else "",
               only_cols_a=";".join(sorted(set(A.columns) - set(B.columns) - set(META))),
               only_cols_b=";".join(sorted(set(B.columns) - set(A.columns) - set(META))))
    da, db = int(A.duplicated(key).sum()), int(B.duplicated(key).sum())
    M = A[key + val].merge(B[key + val], on=key, how="left", suffixes=("_a", "_b"), indicator=True)
    miss = int((M._merge == "left_only").sum())
    M = M[M._merge == "both"]
    ne = np.zeros(len(M), dtype=bool)
    ex, ndiff, maxabs, maxrel = [], 0, 0.0, 0.0
    for c in val:
        x, y = M[c + "_a"].to_numpy(), M[c + "_b"].to_numpy()
        m = x != y
        if m.any():
            ndiff += int(m.sum())
            ne |= m
            ex += [f"{c}:{x[i]}|{y[i]}" for i in np.flatnonzero(m)[:2]]
            xf, yf = pd.to_numeric(pd.Series(x[m]), errors="coerce"), pd.to_numeric(pd.Series(y[m]), errors="coerce")
            ad = (xf - yf).abs()
            maxabs = max(maxabs, float(ad.max()) if ad.notna().any() else float("nan"))
            rd = ad / yf.abs().where(yf != 0)
            maxrel = max(maxrel, float(rd.max()) if rd.notna().any() else 0.0)
    ok = da == 0 and db == 0 and miss == 0 and ndiff == 0 and len(M) > 0 and not res["only_cols_a"] and not res["only_cols_b"]
    res.update(key_cols=";".join(key), n_value_cols=len(val), dup_a=da, dup_b=db, rows_a_missing_in_b=miss, rows_overlap=len(M),
               rows_differ=int(ne.sum()), n_diff_cells=ndiff, max_abs_diff=maxabs, max_rel_diff=maxrel,
               result="IDENTICAL" if ok else "DIFFERENT", detail="; ".join(ex[:6]))
    pd.DataFrame([res]).to_csv(a.out, mode="a", header=not os.path.exists(a.out), index=False)
    print(f"OVERLAP {a.name}: {res['result']} overlap {len(M)} rows (A {len(A)}, B {len(B)}); A missing in B {miss}; "
          f"dup A/B {da}/{db}; differ rows {int(ne.sum())} cells {ndiff} max_abs {maxabs} max_rel {maxrel}; "
          f"code_sha A={res['code_sha_a']} B={res['code_sha_b']} {res['detail']}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
