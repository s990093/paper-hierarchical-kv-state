"""main_rulesplit.py — icpe-main：把 sweep 依 rule 拆成幾份（逐字複製，不解析數值），讓 Mooncake 的 R1、R2 分開餵給 next_sim_analyze.py。

  python -I code/icpe/main_rulesplit.py --in <x1_sweep.csv[.xz]> ... --rules R1 R2 --out-dir <dir>
  → <dir>/<rule>/<序號>_<原路徑攤平>.csv（只含 rule＝該值的列；表頭不變）

為什麼（使用者 2026-11-07 選「R1、R2 分開算」）：P4 的 Mooncake 只跑了版本 B ≥15% 的 R1／R2（偏離 8 第 3 點），
只有一個規則 ≥15% 的格，另一個規則沒有雜訊的列；next_sim_analyze.analyze() 的 rule＝min 對「有的規則」取小，
這些格的雜訊 min＝那一個規則，和沒有雜訊時的 min 不同，λ＝0 自檢因此停下（run 20261107-080102；診斷 20261107-081616：
R1／R2 0 個不同、min 31 個不同且全是只跑一個規則的格）。分開算時每次只有一個 rule，不會產生 min 列；min（只報告）另由 main_p4.py 算。

檢查（不過就 exit 1）：每個檔都有 rule 欄，每一列的 rule 都在 --rules 裡；寫出的各份列數加起來＝原檔列數；
重讀寫出的檔，每一列都和原檔對應的列逐字相同。另寫 <dir>/rulesplit_manifest.csv（run_id、ts、host、來源、rule、輸出、列數）。
"""
from __future__ import annotations

import argparse
import csv
import io
import lzma
import os
import socket
import sys
from datetime import datetime


def opentext(p):
    return io.TextIOWrapper(lzma.open(p), encoding="utf-8", newline="") if p.endswith(".xz") else open(p, encoding="utf-8", newline="")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", required=True)
    ap.add_argument("--rules", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    meta = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"),
                host=socket.gethostname())
    for r in a.rules:
        os.makedirs(os.path.join(a.out_dir, r), exist_ok=True)
    man = []
    for k, p in enumerate(a.inp):
        name = f"{k:03d}_" + os.path.relpath(p).replace("/", "__").removesuffix(".xz")
        outs = {r: os.path.join(a.out_dir, r, name) for r in a.rules}
        fh = {r: open(o, "w", encoding="utf-8", newline="") for r, o in outs.items()}
        w = {r: csv.writer(f, lineterminator="\n") for r, f in fh.items()}
        n = {r: 0 for r in a.rules}
        tot = 0
        with opentext(p) as f:
            rd = csv.reader(f)
            h = next(rd)
            if "rule" not in h:
                print(f"{p} 沒有 rule 欄：停", file=sys.stderr)
                sys.exit(1)
            i = h.index("rule")
            for x in w.values():
                x.writerow(h)
            for row in rd:
                tot += 1
                if len(row) != len(h) or row[i] not in w:
                    print(f"{p} 第 {tot + 1} 列：欄數 {len(row)}／{len(h)}，rule={row[i] if len(row) > i else None!r}（不在 {a.rules}）：停",
                          file=sys.stderr)
                    sys.exit(1)
                w[row[i]].writerow(row)
                n[row[i]] += 1
        for f in fh.values():
            f.close()
        assert sum(n.values()) == tot
        with opentext(p) as f:                      # 重讀核對：依原順序，每一列都在它那一份的同一個位置
            rd = csv.reader(f)
            h0 = next(rd)
            rr = {r: csv.reader(open(o, encoding="utf-8", newline="")) for r, o in outs.items()}
            for r in rr.values():
                assert next(r) == h0
            for row in rd:
                assert next(rr[row[i]]) == row, f"{p}：拆出來的列和原檔不同"
            for r, x in rr.items():
                assert next(x, None) is None, f"{outs[r]} 多出列"
        print(f"{p}：{tot} 列 → " + "、".join(f"{r} {n[r]}" for r in a.rules))
        for r in a.rules:
            man.append(dict(meta, source=p, rule=r, output=outs[r], rows=n[r], source_rows=tot))
    with open(os.path.join(a.out_dir, "rulesplit_manifest.csv"), "w", encoding="utf-8", newline="") as g:
        dw = csv.DictWriter(g, fieldnames=list(man[0]), lineterminator="\n")
        dw.writeheader()
        dw.writerows(man)
    print(f"檢查通過：{len(a.inp)} 個檔依 rule 拆成 {a.rules}，列數加總相同、逐字相同")


if __name__ == "__main__":
    main()
