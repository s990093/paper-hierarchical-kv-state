"""main_rowfilter.py — icpe-main：把 sweep 裡某一欄等於指定值的列拿掉（逐字複製其餘列，不解析數值）。

  python -I code/icpe/main_rowfilter.py --in <x1_sweep.csv[.xz]> ... --col gpu_policy --drop BELP --out-dir <dir>
  → <dir>/<序號>_<原路徑攤平>.csv

用途（使用者 2026-11-07）：版本 C（偏離 10）另外並列「④ 不含 BELP」（只報告）。把 BELP 的列拿掉後用 next_sim_analyze.py 原樣算：
APC 在那裡是不預測的 GPU 規則（進 ② 也進 O-ORD／O-DROP／O-NOW／O-PF 的 ④），O-GPU 的 ④＝NON_ORACLE × BEL。

檢查（不過就 exit 1）：每個檔都有 --col；拿掉的列數 > 0；寫出列數＋拿掉列數＝原列數；重讀寫出的檔，每一列都和原檔留下的列逐字相同、
順序不變，而且 --col 沒有任何一列等於 --drop。另寫 <dir>/rowfilter_manifest.csv（run_id、ts、host、來源、輸出、留下／拿掉列數）。
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
    ap.add_argument("--col", required=True)
    ap.add_argument("--drop", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    drop = set(a.drop)
    meta = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"),
                host=socket.gethostname())
    os.makedirs(a.out_dir, exist_ok=True)
    man = []
    for k, p in enumerate(a.inp):
        out = os.path.join(a.out_dir, f"{k:03d}_" + os.path.relpath(p).replace("/", "__").removesuffix(".xz"))
        kept = dropped = 0
        with opentext(p) as f, open(out, "w", encoding="utf-8", newline="") as g:
            r, w = csv.reader(f), csv.writer(g, lineterminator="\n")
            h = next(r)
            if a.col not in h:
                print(f"{p} 沒有 {a.col} 欄：停", file=sys.stderr)
                sys.exit(1)
            i = h.index(a.col)
            w.writerow(h)
            for row in r:
                if len(row) != len(h):
                    print(f"{p} 第 {kept + dropped + 2} 列有 {len(row)} 欄，表頭 {len(h)} 欄：停", file=sys.stderr)
                    sys.exit(1)
                if row[i] in drop:
                    dropped += 1
                else:
                    w.writerow(row)
                    kept += 1
        if dropped == 0:
            print(f"{p}：{a.col} 沒有任何一列是 {sorted(drop)}：停（檔案或欄位不對？）", file=sys.stderr)
            sys.exit(1)
        with opentext(p) as f, open(out, encoding="utf-8", newline="") as g:      # 重讀核對
            r0, r1 = csv.reader(f), csv.reader(g)
            assert next(r1) == next(r0)
            n = 0
            for x in r0:
                if x[i] in drop:
                    continue
                y = next(r1)
                assert y == x and y[i] not in drop, f"{out} 第 {n + 2} 列和來源不同"
                n += 1
            assert next(r1, None) is None and n == kept
        print(f"{p}：留 {kept} 列、拿掉 {dropped} 列（{a.col} ∈ {sorted(drop)}）→ {out}")
        man.append(dict(meta, source=p, output=out, kept=kept, dropped=dropped, col=a.col, drop=";".join(sorted(drop))))
    with open(os.path.join(a.out_dir, "rowfilter_manifest.csv"), "w", encoding="utf-8", newline="") as g:
        dw = csv.DictWriter(g, fieldnames=list(man[0]), lineterminator="\n")
        dw.writeheader()
        dw.writerows(man)
    print(f"檢查通過：{len(a.inp)} 個檔拿掉 {a.col} ∈ {sorted(drop)} 的列，其餘逐字相同")


if __name__ == "__main__":
    main()
