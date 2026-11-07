"""main_addnoisecols.py — icpe-main：給沒有雜訊欄的 sweep 補上 noise_* 欄（值＝next_sim_analyze.load() 本來會補的預設），
讓它能和 P4（W4 雜訊）的 sweep 一起餵給 next_sim_analyze.py（程式原樣使用；使用者 2026-11-07 選「我先補欄位再餵進去」）。

  python -I code/icpe/main_addnoisecols.py --exact <x1_sweep.csv[.xz]> ... --noisy <x1_sweep.csv[.xz]> ... --out-dir <dir>

為什麼要補：load() 只在「合併後整欄都沒有」時才補 noise_model="none" 等預設；main／_p1 的 sweep 沒有 noise_* 欄、P4 的有，
合併後 main／_p1 的列變成 NaN，等於 "none" 的列是 0（run 20261107-063333 的 KeyError；診斷 run 20261107-063414）。

做法（只動文字，不解析任何數值，所以原本每一格的字串完全不變）：
  - 每個 --exact 檔用 csv 模組逐列讀成字串，在列尾加 noise_model=none、noise_lambda=0.0、noise_draw=none、noise_seed=-1
    （和 next_sim_analyze.load() 的預設相同），寫成 <out-dir>/<序號>_<原路徑攤平>.csv。
  - 檢查（不過就 exit 1）：
    1. --exact 檔一個 noise_* 欄都沒有；--noisy 檔 4 個 noise_* 欄都有、而且 noise_model 沒有 "none"。
    2. load() 會補預設的其他欄（gpu_GiB、gpu_write、gpu_policy、pf_target、slow_cap_frac、slow_policy）：
       在全部檔案裡要嘛都有、要嘛都沒有（否則合併後一樣會變 NaN）。
    3. 寫出的檔重讀一次：列數相同、原本的欄逐字相同、新欄全是預設值。
  - 印出每個檔的列數；另寫 <out-dir>/addnoisecols_manifest.csv（每列有 run_id、ts、host、來源、輸出、列數、來源 sha256）。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import lzma
import os
import socket
import sys
from datetime import datetime

NZ = ["noise_model", "noise_lambda", "noise_draw", "noise_seed"]
NZ_DEFAULT = ["none", "0.0", "none", "-1"]
OTHER_DEFAULTED = ["gpu_GiB", "gpu_write", "gpu_policy", "pf_target", "slow_cap_frac", "slow_policy"]


def opentext(p):
    return io.TextIOWrapper(lzma.open(p), encoding="utf-8", newline="") if p.endswith(".xz") else open(p, encoding="utf-8", newline="")


def header(p):
    with opentext(p) as f:
        return next(csv.reader(f))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exact", nargs="+", required=True)
    ap.add_argument("--noisy", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    bad = []
    H = {p: header(p) for p in a.exact + a.noisy}
    for p in a.exact:
        has = [c for c in NZ if c in H[p]]
        if has:
            bad.append(f"--exact {p} 已經有 {has}")
    for p in a.noisy:
        miss = [c for c in NZ if c not in H[p]]
        if miss:
            bad.append(f"--noisy {p} 少了 {miss}")
    for c in OTHER_DEFAULTED:
        have = {c in h for h in H.values()}
        if len(have) > 1:
            bad.append(f"{c}：有的檔有、有的檔沒有 → 合併後會變 NaN（" + ", ".join(p for p, h in H.items() if c not in h)[:300] + " 沒有）")
    for p in a.noisy:
        if not [c for c in NZ if c not in H[p]]:
            with opentext(p) as f:
                r = csv.reader(f)
                i = next(r).index("noise_model")
                n_none = sum(1 for row in r if row[i] == "none")
            if n_none:
                bad.append(f"--noisy {p} 有 {n_none} 列 noise_model=none")
    if bad:
        print("檢查不過，停：\n  " + "\n  ".join(bad), file=sys.stderr)
        sys.exit(1)
    os.makedirs(a.out_dir, exist_ok=True)
    meta = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"),
                host=socket.gethostname())
    man = []
    for k, p in enumerate(a.exact):
        out = os.path.join(a.out_dir, f"{k:03d}_" + p.replace("/", "__").removesuffix(".xz"))
        n = 0
        with opentext(p) as f, open(out, "w", encoding="utf-8", newline="") as g:
            r, w = csv.reader(f), csv.writer(g, lineterminator="\n")
            h = next(r)
            w.writerow(h + NZ)
            for row in r:
                if len(row) != len(h):
                    print(f"{p} 第 {n + 2} 列有 {len(row)} 欄，表頭 {len(h)} 欄：停", file=sys.stderr)
                    sys.exit(1)
                w.writerow(row + NZ_DEFAULT)
                n += 1
        with opentext(p) as f, open(out, encoding="utf-8", newline="") as g:     # 重讀核對
            r0, r1 = csv.reader(f), csv.reader(g)
            h0, h1 = next(r0), next(r1)
            assert h1 == h0 + NZ, f"{out} 表頭不對"
            m = 0
            for x, y in zip(r0, r1, strict=True):
                assert y[:len(h0)] == x and y[len(h0):] == NZ_DEFAULT, f"{out} 第 {m + 2} 列和來源不同"
                m += 1
            assert m == n
        print(f"{p}：{n} 列 → {out}")
        man.append(dict(meta, source=p, output=out, rows=n, source_sha256=sha256(p)))
    for p in a.noisy:
        man.append(dict(meta, source=p, output="(unchanged, passed as is)", rows="", source_sha256=sha256(p)))
    with open(os.path.join(a.out_dir, "addnoisecols_manifest.csv"), "w", encoding="utf-8", newline="") as g:
        w = csv.DictWriter(g, fieldnames=list(man[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(man)
    print(f"檢查通過：{len(a.exact)} 個 exact 檔補了 {NZ}；{len(a.noisy)} 個 noisy 檔原樣使用")


if __name__ == "__main__":
    main()
