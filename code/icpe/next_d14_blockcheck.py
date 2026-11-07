"""next_d14_blockcheck.py — icpe-data：偏離 14 重跑 block 收進 git 前的核對（STATUS_DATA 2026-11-07 17:01）。只讀 run 目錄與清單。

每個跑完的 d14_<plan>/<block>（runs/next_b_queue_done.jsonl）：
  * exit_code＝0；stdout 有 `py_fallback_jobs=0`、`jobs=N workers=W code_sha=…`。
  * N＝清單的 jobs_rerun；P4 模型 P（flags 有 --noise-overdue）整個 block 重跑 → N＝jobs_block（count 的 WHOLE_BLOCK）。
  * rust_timing.jsonl 的行數＝N。
  * 有 jobs.csv（P4）：stdout 的 kept_jobs＝N、unmatched_cells=0。
  * cmd.sh 帶清單 flags 欄的每個旗標。
d14_p4verc 另外：N＝清單格子的 (t,r,c,p,g) 數 × 雜訊設定數（由 next_b_queue count 印的 expect_kept 給，--p4verc-expect plan/block=N）。
輸出 CSV（run_id、ts、host 每列都有），最後一行 D14_BLOCKCHECK PASS|FAIL n=…；FAIL 時 exit 1。
用法：python -I code/icpe/next_d14_blockcheck.py --list results/icpe_next/sim/dev14_rerun_list.csv --done <done.jsonl> \
        --plans d14_bl_main d14_bl_p1 [--only 'fi6w*'] --out <csv>
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import re
import socket
import sys
from datetime import datetime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--done", required=True)
    ap.add_argument("--plans", nargs="+", required=True)
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--p4verc-expect", nargs="*", default=[], help="plan/block=N")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = {(r["plan"], r["block"]): r for r in csv.DictReader(open(a.list, newline=""))}
    exp_vc = {k: int(v) for k, v in (x.split("=") for x in a.p4verc_expect)}
    done = {}
    for ln in open(a.done):
        r = json.loads(ln)
        done[(r["plan"], r["block"])] = r
    meta = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"),
                host=socket.gethostname())
    out, bad = [], 0
    for (plan, block), d in sorted(done.items()):
        if plan not in a.plans or (a.only and not any(fnmatch.fnmatch(block, p) for p in a.only)):
            continue
        run = d["run"]
        so = open(f"{run}/stdout.log", errors="replace").read()
        cmd = open(f"{run}/cmd.sh").read()
        rc = open(f"{run}/exit_code").read().strip()
        m = re.search(r"^jobs=(\d+) workers=(\d+) code_sha=(\S+)", so, re.M)
        n = int(m.group(1)) if m else -1
        nt = sum(1 for _ in open(f"{run}/rust_timing.jsonl")) if os.path.exists(f"{run}/rust_timing.jsonl") else -1
        why = []
        if rc != "0":
            why.append(f"rc={rc}")
        if "py_fallback_jobs=0" not in so:
            why.append("py_fallback")
        if plan == "d14_p4verc":
            want = exp_vc.get(f"{plan}/{block}")
            flags = ["RUX_PFX=1", "RUX_PGR=1", "RUX_NZO=resample"]
        else:
            base = plan[len("d14_"):]
            r = L.get((base, block))
            if r is None or r["action"] != "rerun":
                why.append("not_in_list")
                want, flags = None, []
            else:
                whole = "--noise-overdue" in r["flags"]
                want = int(r["jobs_block"] if whole else r["jobs_rerun"])
                flags = r["flags"].split()
        if want is None or n != want:
            why.append(f"jobs={n}!=want={want}")
        if nt != n:
            why.append(f"rust_timing={nt}!=jobs={n}")
        if os.path.exists(f"{run}/jobs.csv"):
            k = re.search(r"kept_jobs=(\d+)", so)
            u = re.search(r"unmatched_cells=(\d+)", so)
            if not k or int(k.group(1)) != n:
                why.append(f"kept_jobs={k and k.group(1)}")
            if not u or u.group(1) != "0":
                why.append(f"unmatched_cells={u and u.group(1)}")
        for f in flags:
            if f not in cmd:
                why.append(f"flag_missing:{f}")
        ok = not why
        bad += not ok
        out.append(dict(meta, plan=plan, block=block, block_run_id=d["run_id"], jobs=n, want=want, rust_timing=nt,
                        code_sha=m.group(3) if m else "", result="PASS" if ok else "FAIL", why=";".join(why)))
        print(f"{'PASS' if ok else 'FAIL'} {plan}/{block} jobs={n} want={want} rust_timing={nt} {';'.join(why)}")
    if not out:
        raise SystemExit("沒有符合的已完成 block")
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        if new:
            w.writeheader()
        w.writerows(out)
    print(f"D14_BLOCKCHECK {'FAIL' if bad else 'PASS'} n={len(out)} fail={bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
