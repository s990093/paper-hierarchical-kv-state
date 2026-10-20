"""summarize_stress.py — 壓力程式單獨跑（stress_selftest.sh）與記憶體 NUMA 分佈（stress_numa.sh）→ results/p3_calib_3090/cpu_stress_standalone.csv（2026-10-18）。

  python code/p3cal/summarize_stress.py --selftest <run_id> --numa <run_id> --out results/p3_calib_3090
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import datetime

RUNS = os.environ.get("TIARA_RUNS", "/ssd8/hungwei/paper-hkv/runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", required=True)
    ap.add_argument("--numa", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    d = os.path.join(RUNS, a.selftest)
    for ln in open(os.path.join(d, "stdout.log")):
        if ln.startswith("DONE "):
            j = json.loads(ln[5:])
            ts = datetime.fromtimestamp(os.path.getmtime(os.path.join(d, "stdout.log"))).astimezone().isoformat(timespec="seconds")
            rep = os.path.join(d, f"stress_n{j['threads']}_{'all' if j['cpus'] == 'ALL' else 'node1'}.csv")
            if os.path.exists(rep):
                r0 = list(csv.DictReader(open(rep)))
                ts = r0[0]["ts"] if r0 else ts
            rows.append({"run_id": a.selftest, "ts": ts, "kind": "standalone_copy", "threads": j["threads"], "cpus": j["cpus"],
                         "mib_per_buffer": j["mib"], "seconds": j["seconds"], "mean_copy_GiBps": j["mean_copy_GiBps"],
                         "median_1s_copy_GiBps": j["median_1s_GiBps"], "node0_anon_GiB": "", "node1_anon_GiB": "",
                         "node1_share": ""})
    d = os.path.join(RUNS, a.numa)
    for ln in open(os.path.join(d, "stdout.log")):
        m = re.match(r"N=(\d+) anon on node0 ([\d.]+) GiB, node1 ([\d.]+) GiB, node1 share ([\d.]+)", ln)
        if m:
            n = int(m.group(1))
            out = open(os.path.join(d, f"stress_n{n}.out")).read()
            j = json.loads([x for x in out.splitlines() if x.startswith("DONE ")][0][5:])
            rows.append({"run_id": a.numa, "ts": datetime.fromtimestamp(os.path.getmtime(os.path.join(d, f"numa_maps_n{n}.txt"))).astimezone().isoformat(timespec="seconds"),
                         "kind": "numa_placement_5s_after_start", "threads": n, "cpus": j["cpus"], "mib_per_buffer": j["mib"],
                         "seconds": j["seconds"], "mean_copy_GiBps": j["mean_copy_GiBps"], "median_1s_copy_GiBps": j["median_1s_GiBps"],
                         "node0_anon_GiB": m.group(2), "node1_anon_GiB": m.group(3), "node1_share": m.group(4)})
    with open(os.path.join(a.out, "cpu_stress_standalone.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
