"""next_d14_store_table.py — icpe-data：把 next_d14_verb.sh 的 run 收進 git（results/icpe_next/b/analysis_d14/<家族>_p1/）。
檔案：<家族>_p1_headroom_components.csv.xz、<家族>_p1_violations.csv.xz、table_vs_verB.csv、rowcheck.csv、patch_summary.csv、manifest.json。
只在 run 的 stdout 有 D14_VERB_DONE 且 table_vs_verB.log 有 D14_TABLE_RESULT IDENTICAL 時才收（否則 exit 1）。
用法：python -I code/icpe/next_d14_store_table.py <家族> <run 目錄> [--dest results/icpe_next/b/analysis_d14]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import shutil
import socket
import sys


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fam")
    ap.add_argument("run")
    ap.add_argument("--dest", default="results/icpe_next/b/analysis_d14")
    a = ap.parse_args()
    R, f = a.run.rstrip("/"), a.fam
    if f"D14_VERB_DONE {f}" not in open(f"{R}/stdout.log").read():
        raise SystemExit("run 沒有 D14_VERB_DONE → 不收")
    if "D14_TABLE_RESULT IDENTICAL" not in open(f"{R}/table_vs_verB.log").read():
        raise SystemExit("table_vs_verB 不是 IDENTICAL → 不收")
    D = f"{a.dest}/{f}_p1"
    os.makedirs(D, exist_ok=True)
    for nm in (f"{f}_p1_headroom_components.csv", f"{f}_p1_violations.csv"):
        src = f"{R}/out/{nm}"
        with open(src, "rb") as fi, lzma.open(f"{D}/{nm}.xz", "wb", preset=9) as fo:
            shutil.copyfileobj(fi, fo)
    for nm in ("table_vs_verB.csv", "rowcheck.csv", "patch_summary.csv"):
        shutil.copy(f"{R}/{nm}", f"{D}/{nm}")
    files = {x: sha(f"{D}/{x}") for x in sorted(os.listdir(D)) if x != "manifest.json"}
    m = dict(run_id=os.path.basename(R), host=socket.gethostname(), run_dir=R, cmd=open(f"{R}/cmd.sh").read(),
             context=open(f"{R}/context.txt").read(),
             table_sha256_uncompressed=sha(f"{R}/out/{f}_p1_headroom_components.csv"), files=files,
             note=(f"PREREG 偏離 14 第一段的版本 B 檢查表（icpe-data；next_d14_verb.sh）：{f} 的 main＋P1 _ttl，原本的列換上 d14 列"
                   "（next_d14_patch.py）後用 next_sim_analyze 算；O-PF、O-ALL(report) 以外的元件和原本的版本 B 表逐值相同（table_vs_verB.csv）。"
                   "給第二段的重現檢查用；論文用的正式表由 icpe-main 另外算"))
    json.dump(m, open(f"{D}/manifest.json", "w"), ensure_ascii=False, indent=1)
    print(f"STORED {D} files={len(files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
