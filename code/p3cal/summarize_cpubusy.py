"""summarize_cpubusy.py — cpu_busy_restore.py 的輸出整理成 results/p3_calib_3090/（2026-10-18）。

  python code/p3cal/summarize_cpubusy.py --run /ssd8/hungwei/paper-hkv/runs/<run_id> --out results/p3_calib_3090

變慢倍數的定義（寫死）：
  - 頻寬類（H2D／D2H GiB/s）：倍數＝N＝0 的中位數 ÷ 這個 (N, place) 的中位數（>1＝變慢）。
  - 時間類（t_restore、ttft、t_h2d、t_load、t_recompute）：倍數＝這個 (N, place) 的中位數 ÷ N＝0 的中位數。
  - N＝0 的基準＝兩段 N＝0（開頭、結尾）合在一起的中位數；另報兩段各自的中位數（看漂移）。
  - 同一個 (N, same) 有往上、往下兩段：合在一起取中位數；另報兩段各自的中位數。
輸出：cpubusy_h2d_raw.csv、cpubusy_restores_raw.csv、cpubusy_phases.csv（原樣複製，加 summarize_run_id）、
      cpubusy_slowdown.csv（每個 (N, place, 指標) 一列）、cpubusy_summary.json。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from collections import defaultdict
from datetime import datetime


def med(v):
    return round(statistics.median(v), 6) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rid = os.path.basename(a.run.rstrip("/"))
    srid = os.environ.get("RUN_ID", "NO_RUN_ID")
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(os.path.join(a.run, "CONTAMINATED")):
        raise SystemExit(f"{rid} 有 CONTAMINATED：時間欄不得進 results/（CLAUDE.md §3）")
    rc = open(os.path.join(a.run, "exit_code")).read().strip()
    if rc != "0":
        raise SystemExit(f"{rid} exit_code={rc}")
    raw = {}
    for k in ("h2d", "restores", "phases"):
        rows = list(csv.DictReader(open(os.path.join(a.run, f"{k}.csv"))))
        for r in rows:
            if r["run_id"] != rid:
                raise SystemExit(f"{k}.csv 的 run_id {r['run_id']} ≠ 目錄名 {rid}")
            r["summarize_run_id"] = srid
        raw[k] = rows
        with open(os.path.join(a.out, f"cpubusy_{k}{'_raw' if k != 'phases' else ''}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    bad_kv = [r for r in raw["restores"] if r["kv_bad"] not in ("", "0")]
    cont = [r for r in raw["restores"] if r["contaminated_gpu"] not in ("False",)]
    ph = {int(r["phase"]): r for r in raw["phases"]}
    meta = json.load(open(os.path.join(a.run, f"meta_{rid}.json")))
    gnode = int(meta["gpu_numa_node"])                    # 量測 process 與 place＝same 的壓力都在這個節點
    # 指標 → (N, place) → phase → 值
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in raw["h2d"]:
        data[("bw", r["item"] + f"_{r['size_mib']}MiB")][(int(r["N"]), r["place"])][int(r["phase"])].append(float(r["GiBps"]))
    for r in raw["restores"]:
        for col in ("t_restore", "ttft", "t_h2d", "t_load", "t_recompute", "b"):
            if r[col] == "":
                continue
            if col in ("t_h2d", "t_load") and r["mode"] == "compute_only":
                continue
            data[("time" if col != "b" else "count", f"{r['mode']}:{col}")][(int(r["N"]), r["place"])][int(r["phase"])].append(float(r[col]))
    out = []
    for (kind, metric), byk in sorted(data.items()):
        base_ph = byk.get((0, "same"), {})
        base_all = [x for v in base_ph.values() for x in v]
        b = med(base_all)
        for (N, place), byp in sorted(byk.items()):
            vals = [x for v in byp.values() for x in v]
            m = med(vals)
            phs = sorted(byp)
            if kind == "bw":
                sd = round(b / m, 4) if (b and m) else None
            elif kind == "time":
                sd = round(m / b, 4) if (b and m) else None
            else:
                sd = None
            out.append({
                "run_id": rid, "ts": now, "summarize_run_id": srid, "N": N, "place": place, "kind": kind, "metric": metric,
                "unit": "GiB/s" if kind == "bw" else ("s" if kind == "time" else "blocks"), "n": len(vals),
                "median": m, "min": min(vals), "max": max(vals), "baseline_N0_median": b,
                "slowdown_x": sd, "phases": " ".join(map(str, phs)),
                "median_by_phase": " ".join(f"{p}:{med(byp[p])}" for p in phs),
                "stress_copy_GiBps_by_phase": " ".join(f"{p}:{ph[p]['stress_copy_GiBps_median']}" for p in phs),
                "loadavg_by_phase": " ".join(f"{p}:{ph[p]['loadavg_start']}->{ph[p]['loadavg_end']}" for p in phs),
                "cpu_util_gpu_node_by_phase": " ".join(f"{p}:{ph[p][f'cpu_util_node{gnode}']}" for p in phs),
            })
    with open(os.path.join(a.out, "cpubusy_slowdown.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    key = ["H2D_pinned_1024MiB", "D2H_pinned_1024MiB", "H2D_pageable_1024MiB", "D2H_pageable_1024MiB",
           "H2D_pinned_chunk_into_kv_view_64MiB"]
    tbl = {}
    for r in out:
        if r["metric"] in key or r["metric"].startswith("H2D_pinned_qwen") or r["metric"] in (
                "cake:t_restore", "load_only:t_restore", "cake:ttft", "load_only:ttft", "compute_only:t_restore",
                "cake:b", "load_only:t_h2d", "cake:t_h2d"):
            tbl.setdefault(f"N={r['N']}:{r['place']}", {})[r["metric"]] = {"median": r["median"], "slowdown_x": r["slowdown_x"],
                                                                         "n": r["n"]}
    summ = {"run_id": rid, "ts": now, "summarize_run_id": srid, "gpu_phys": meta["gpu_phys"], "gpu_numa_node": gnode,
            "inner_verdict": meta["inner_verdict"], "peak_allocated_gib": meta["peak_allocated_gib"], "kv_bad_rows": len(bad_kv),
            "rows_with_contaminated_flag": len(cont),
            "phases": [{k: ph[p][k] for k in ("phase", "N", "place", "stress_cpus", "secs", "loadavg_start", "loadavg_end",
                                               "cpu_util_node0", "cpu_util_node1", "stress_copy_GiBps_median",
                                               "host_level", "host_foreign_gpus")} for p in sorted(ph)],
            "table": tbl}
    json.dump(summ, open(os.path.join(a.out, "cpubusy_summary.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summ, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
