"""x9b_ttl.py — X9b 的 T6（I8：固定 TTL 5 分鐘／1 小時 vs LRU），FreeInference。

判準在 X9 文件 §2.5（2026-10-17T13:41:37Z 寫定）。直接 import X8 的模擬器（code/p3x/x8_sim.py），不改它：
X8 第 1 層（有 TTL）當成 CPU 層、第 2 層容量 0、第 3 層＝SSD（∞）或沒有（0）。
用法： python code/p3x/x9b_ttl.py --out results/p3_explore/x9 --procs 6
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x8_sim  # noqa: E402

TRACE_DIR = "/ssd8/hungwei/paper-hkv/runs/20261017-180838-p3x8-traces3/traces"
RUN_ID = os.environ.get("RUN_ID", "")
C_LIST = [128, 256, 512, 1024]
POLS = [("LRU+SSD", "F-LRU", True), ("LRU-noSSD", "F-LRU", False),
        ("TTL300-drop", "F-TTL-300", False), ("TTL3600-drop", "F-TTL-3600", False),
        ("TTL300-SSD", "F-TTL-300", True), ("TTL3600-SSD", "F-TTL-3600", True)]
H_LIST = [0, 1800, 3600]
_D = {}


def task(args):
    C, name, pol, ssd = args
    if "d" not in _D:
        _D["d"], _D["meta"] = x8_sim.load_wl(TRACE_DIR, "fi", "orig")
    d = _D["d"]
    out, st = x8_sim.simulate(d, pol, C * 16, 0, 10 ** 18 if ssd else 0)
    ret = (d["ret_kind"] > 0) & d["eval"]
    rows = []
    for H in H_LIST:
        m = ret & ((d["t_end"] - d["t_dec"]) >= H) if H > 0 else ret
        oc = out[m]
        rows.append(dict(C_GiB=C, policy=name, x8_policy=pol, ssd=ssd, H_s=H, n_ret=int(m.sum()),
                         n_sess=int(len(np.unique(d["sid"][m]))),
                         frac_cpu=float(np.mean(oc == 1)), frac_ssd=float(np.mean(oc == 3)),
                         frac_dropped=float(np.mean(oc == 4)), frac_active=float(np.mean(oc == 5)),
                         frac_other=float(np.mean((oc == 0) | (oc == 2))),
                         ssd_write_GiB_eval=st["slow_w_blk"] * x8_sim.GIB_PER_BLK))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    assert RUN_ID
    jobs = [(C, n, p, s) for C in C_LIST for (n, p, s) in POLS]
    with ProcessPoolExecutor(a.procs) as ex:
        R = list(ex.map(task, jobs))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = [dict(run_id=RUN_ID, ts=now, trace="freeinference(x8 pack 20261017-180838-p3x8-traces3)", **r)
            for rr in R for r in rr]
    # 判準用的相對差：TTL300-drop vs TTL3600-drop 的 CPU 命中（H=0）
    ver = []
    for C in C_LIST:
        g = {r["policy"]: r for r in rows if r["C_GiB"] == C and r["H_s"] == 0}
        a3, a36 = g["TTL300-drop"]["frac_cpu"], g["TTL3600-drop"]["frac_cpu"]
        rel = abs(a36 - a3) / max(a3, a36, 1e-12)
        ver.append(dict(run_id=RUN_ID, ts=now, C_GiB=C, cpu_hit_TTL300_drop=a3, cpu_hit_TTL3600_drop=a36,
                        rel_diff_TTL300_vs_TTL3600=rel, cpu_hit_LRU=g["LRU+SSD"]["frac_cpu"],
                        cpu_or_ssd_hit_LRU_SSD=g["LRU+SSD"]["frac_cpu"] + g["LRU+SSD"]["frac_ssd"],
                        ssd_write_GiB_LRU_SSD=g["LRU+SSD"]["ssd_write_GiB_eval"],
                        ssd_write_GiB_TTL300_SSD=g["TTL300-SSD"]["ssd_write_GiB_eval"],
                        ssd_write_GiB_TTL3600_SSD=g["TTL3600-SSD"]["ssd_write_GiB_eval"]))
    os.makedirs(a.out, exist_ok=True)
    for fn, rs in (("x9b_t6_ttl_vs_lru.csv", rows), ("x9b_t6_ttl_verdict.csv", ver)):
        with open(os.path.join(a.out, fn), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rs[0].keys()))
            w.writeheader()
            w.writerows(rs)
    for r in ver:
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
