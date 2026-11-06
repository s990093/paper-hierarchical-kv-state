"""main_cmp.py — icpe-main：兩份 next_sim_analyze.py 輸出（headroom_components／violations）逐格比對。

  python -I code/icpe/main_cmp.py --name <名稱> --a <檔 A> --b <檔 B> --out <cmp.csv（附加）>

比法：兩邊都以字串讀入（可讀 .xz）；去掉只記錄「誰、何時、在哪台算」的欄（run_id、ts、host）；
欄位集合必須相同；依全部欄排序後逐格比字串。不同就印出前幾個例子、照實寫進 cmp.csv，exit 1。
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
from datetime import datetime

import pandas as pd

META = ["run_id", "ts", "host"]


def load(p):
    d = pd.read_csv(p, dtype=str, keep_default_na=False)
    return d.drop(columns=[c for c in META if c in d.columns])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    A, B = load(a.a), load(a.b)
    res = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"),
               host=socket.gethostname(), name=a.name, file_a=a.a, file_b=a.b, rows_a=len(A), rows_b=len(B),
               cols_a=A.shape[1], cols_b=B.shape[1])
    if set(A.columns) != set(B.columns):
        res.update(n_diff_cells="NA", result="COLUMNS_DIFFER",
                   detail=f"only_a={sorted(set(A.columns) - set(B.columns))} only_b={sorted(set(B.columns) - set(A.columns))}")
    elif len(A) != len(B):
        res.update(n_diff_cells="NA", result="ROWS_DIFFER", detail="")
    else:
        cols = sorted(A.columns)
        A = A[cols].sort_values(cols, kind="mergesort").reset_index(drop=True)
        B = B[cols].sort_values(cols, kind="mergesort").reset_index(drop=True)
        ne = A.ne(B)
        n = int(ne.values.sum())
        ex = ""
        if n:
            r, c = ne.values.nonzero()
            ex = "; ".join(f"row{r[i]}:{cols[c[i]]}:{A.iat[r[i], c[i]]}|{B.iat[r[i], c[i]]}" for i in range(min(5, len(r))))
        res.update(n_diff_cells=n, result="IDENTICAL" if n == 0 else "DIFFERENT", detail=ex)
    new = not os.path.exists(a.out)
    pd.DataFrame([res]).to_csv(a.out, mode="a", header=new, index=False)
    print(f"CMP {a.name}: {res['result']} rows {res['rows_a']}/{res['rows_b']} cols {res['cols_a']}/{res['cols_b']} "
          f"diff_cells={res['n_diff_cells']} {res['detail']}")
    sys.exit(0 if res["result"] == "IDENTICAL" else 1)


if __name__ == "__main__":
    main()
