"""m10_sweep_sim.py — 大掃描（09）的速度推算：用「零件」時間推算任意低精度配置的 Cake 還原（會合點 b、還原時間）。

為什麼：大掃描有幾千種（格式 × 佔比 × 放法）組合，不可能每種都跑速度格子（一格 30 次 × 8 格）。
做法（照 m7_restore_harness.Restorer.restore 的 cake 模式）：
  - 重算線從 block 0 往後，每個 block 花 f[state][i]；
  - 載入線從 block n−1 往前，每個 block 花 l[tier][state][dtype][i]（限速讀＋H2D＋反量化，照實量的區間）；
  - 哪一條先空下來就拿下一個 block，直到兩邊相遇；還原時間＝兩條線較晚結束的那條。
零件來源：同一個 grid run 的 blocks.csv 裡「只算」「只載」兩條參考線（每個 block 的 t_end−t_start 中位數）。
新格式沒有參考線時，用 parts 指令量的每 block H2D＋反量化時間（09 §5）。

驗證（calib 指令）：拿 grid run 的參考線當零件，推算 8 種配置 × 8 格的會合點與還原時間，和同一 run 實測的 cake 中位數比。
推算只在驗證通過（09 §5 寫死的門檻）之後才用於大掃描。
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _open(p):
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p)


def load_parts(blocks_csv: str, length: int = 32768) -> dict:
    """參考線 → 每個 block 的中位數：f[state][i]、l[(tier, state, dtype)][i]（dtype＝stored_dtype）。"""
    fr = collections.defaultdict(list)
    lr = collections.defaultdict(list)
    for r in csv.DictReader(_open(blocks_csv)):
        if int(r["length"]) != length:
            continue
        dt = float(r["t_end"]) - float(r["t_start"])
        i = int(r["block"])
        if r["mode"] == "compute_only":
            fr[(r["gpu_state"], i)].append(dt)
        elif r["mode"] == "load_only":
            lr[(r["bw_name"], r["gpu_state"], r["stored_dtype"], i)].append(dt)
    f = collections.defaultdict(dict)
    for (s, i), v in fr.items():
        f[s][i] = statistics.median(v)
    l = collections.defaultdict(dict)
    for (bw, s, d, i), v in lr.items():
        l[(bw, s, d)][i] = statistics.median(v)
    return {"f": {s: [d[i] for i in sorted(d)] for s, d in f.items()},
            "l": {k: [d[i] for i in sorted(d)] for k, d in l.items()}}


def simulate(f: list, load: list) -> tuple[int, float, float, float]:
    """f[i]：重算 block i 的時間；load[i]：載入 block i 的時間。回傳（會合點 b＝重算的 block 數, 還原時間, 重算線結束, 載入線結束）。"""
    n = len(f)
    p, q, tr, tl = 0, n - 1, 0.0, 0.0
    while p <= q:
        if tr <= tl:                      # 重算線先空下來（同時空下來時重算線先拿，和 harness 的 rline 先跑一致）
            tr += f[p]
            p += 1
        else:
            tl += load[q]
            q -= 1
    return p, max(tr, tl), tr, tl


def predict(parts: dict, bw: str, state: str, lp_mask: list, lp_dtype: str) -> tuple[int, float]:
    f = parts["f"][state]
    lb = parts["l"][(bw, state, "bf16")]
    ll = parts["l"].get((bw, state, lp_dtype))
    load = [(ll[i] if m else lb[i]) for i, m in enumerate(lp_mask)]
    b, t, _, _ = simulate(f, load)
    return b, t


def cmd_calib(a):
    from m10_speed import layout
    parts = load_parts(a.blocks)
    summ = [r for r in csv.DictReader(open(a.summary)) if r["mode"] == "cake" and int(r["length"]) == 32768]
    out, errs = [], []
    for r in summ:
        cfg, bw, s = r["config"], r["bw_name"], r["gpu_state"]
        mask = layout(cfg, 64, 0) if cfg != "R25" else None
        if mask is None:                  # R25：每次 rep 的 seed 不同；用 5 個 seed 的推算平均（只拿來驗中位數）
            preds = [predict(parts, bw, s, layout("R25", 64, sd), a.lp_dtype) for sd in range(5)]
            b = statistics.median(p[0] for p in preds)
            t = statistics.median(p[1] for p in preds)
        else:
            b, t = predict(parts, bw, s, mask, a.lp_dtype)
        mb = [int(x) for x in r["b_values"].split()]
        mt = float(r["median_t_restore"])
        e = (t - mt) / mt
        errs.append(abs(e))
        out.append(dict(config=cfg, bw=bw, gpu_state=s, b_pred=b, b_meas=r["b_values"],
                        b_in_range=min(mb) - 1 <= b <= max(mb) + 1, t_pred=round(t, 4), t_meas=mt, rel_err=round(e, 4)))
    res = {"blocks": a.blocks, "summary": a.summary, "n_cells": len(out),
           "median_abs_rel_err": round(statistics.median(errs), 4), "max_abs_rel_err": round(max(errs), 4),
           "b_in_range_frac": round(sum(o["b_in_range"] for o in out) / len(out), 4), "cells": out}
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    for o in sorted(out, key=lambda o: (o["bw"], o["gpu_state"], o["config"])):
        print(f"{o['bw']:4s} {o['gpu_state']:4s} {o['config']:5s} b {o['b_pred']!s:>5}/{o['b_meas']:<12s} "
              f"t {o['t_pred']:.4f}/{o['t_meas']:.4f} ({o['rel_err']:+.1%})")
    print(f"median |err| {res['median_abs_rel_err']:.2%}  max |err| {res['max_abs_rel_err']:.2%}  b 在 ±1 內 {res['b_in_range_frac']:.0%}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("calib", help="用參考線推算 cake，和同一 run 的實測比")
    c.add_argument("--blocks", required=True, help="grid run 的 blocks.csv（可 .gz）")
    c.add_argument("--summary", required=True, help="同一 run 的 summary_<tier>.csv")
    c.add_argument("--lp-dtype", default="fp8", help="blocks.csv 的 stored_dtype（fp8／int8／int4）")
    c.add_argument("--out")
    a = ap.parse_args()
    {"calib": cmd_calib}[a.cmd](a)


if __name__ == "__main__":
    main()
