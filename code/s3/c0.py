#!/usr/bin/env python3
"""c0.py — 第三階段 C0（05 §6）：只用物理常數的「重算還是搬」預測器，在沒看過的格子上準不準。
判準：docs/stage3_design_20261024/runs/C0_cost_model.md（登記，不改）。只用 CPU、只讀已 commit 的結果。

  cells     建格子表 → results/s3_h200/c0/cells.csv（每格 Cake、全部載入、G5／G5b 的中位數與預測器的輸入）
  eval      四種切法的選錯代價 → results/s3_h200/c0/c0_report.json、tables_c0.md
  --self-test  預測器的算式（合成常數）
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import math
import os
import re
import statistics
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
R = os.path.join(REPO, "results")
OUT = os.path.join(R, "s3_h200", "c0")
GiB = 1 << 30
STD_BUSY = {"h200": "1.0", "mi300x": "0.7"}                  # 各平台的標準忙（parts 也是這個 duty）
MIN_REPS = 5


def run_id():
    return os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc"


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _open(p):
    return gzip.open(p, "rt", newline="") if p.endswith(".gz") else open(p, newline="")


# ============================================================================ 來源
def sources():
    """(平台, 來源名, kernel, restores 檔) 的清單（登記 §1）。"""
    out = []
    for plat in ("h200", "mi300x"):
        for m in ("llama", "qwen"):
            p = os.path.join(R, f"m10_precision_{plat}", f"speed_{m}", "restores.csv.gz")
            if os.path.exists(p):
                out.append((plat, "precision", "sdpa", p))
        base = os.path.join(R, f"m10_restore2_{plat}")
        for p in sorted(glob.glob(os.path.join(base, "q2", "*", "restores.csv"))):
            out.append((plat, "q2/" + os.path.basename(os.path.dirname(p)), "sdpa", p))
        for p in sorted(glob.glob(os.path.join(base, "q1", "*", "restores.csv"))):
            d = os.path.basename(os.path.dirname(p))
            if d.endswith(("_P0", "_G5")):
                out.append((plat, "q1/" + d, "sdpa", p))
        for p in sorted(glob.glob(os.path.join(base, "q4", "*_P0", "restores.csv"))):
            out.append((plat, "q4/" + os.path.basename(os.path.dirname(p)), "sdpa", p))
    for p in sorted(glob.glob(os.path.join(R, "icpe_h200", "p01", "grid", "*_g5b_*", "restores.csv"))):
        d = os.path.basename(os.path.dirname(p))
        out.append(("h200", "icpe/" + d, d.rsplit("_", 1)[-1], p))
    return out


def is_g5_dir(src: str) -> bool:
    return bool(re.search(r"(_G5$|_g5b$|_g5b_)", src))


def excluded_runs() -> set:
    """ICPE P0-1 (b) 標成和自己的 job 重疊的 run（results/icpe_h200/p01/grid/EXCLUDED_RUNS.csv）＋任何 run 目錄有 OVERLAP_OWN_JOB 的。"""
    p = os.path.join(R, "icpe_h200", "p01", "grid", "EXCLUDED_RUNS.csv")
    out = {r["run_id"] for r in csv.DictReader(open(p))} if os.path.exists(p) else set()
    runs = os.environ.get("TIARA_RUNS", os.path.join(os.path.expanduser("~"), "tiara-runs"))
    out |= {os.path.basename(os.path.dirname(f)) for f in glob.glob(os.path.join(runs, "*", "OVERLAP_OWN_JOB"))}
    return out


def build_cells():
    cells = {}
    excl = {}
    bad_runs = excluded_runs()
    for plat, src, kern, p in sources():
        for r in csv.DictReader(_open(p)):
            if r["run_id"] in bad_runs:
                excl["overlap_own_job"] = excl.get("overlap_own_job", 0) + 1
                continue
            if r.get("contaminated_gpu") == "True" or str(r.get("kv_bad", "0")) not in ("0", "", "0.0"):
                excl["contaminated_or_kv_bad"] = excl.get("contaminated_or_kv_bad", 0) + 1
                continue
            st, duty = r["gpu_state"], r["duty"]
            if st not in ("idle", "busy") or (st == "busy" and duty != STD_BUSY[plat]) or (st == "idle" and float(duty or 0) != 0):
                excl["nonstd_duty"] = excl.get("nonstd_duty", 0) + 1
                continue
            mode = r["mode"]
            if mode == "claim" or (mode == "cake" and not is_g5_dir(src)):
                opt = "cake"
            elif mode in ("loadall", "load_only"):
                opt = "full"
            elif mode == "cake" and is_g5_dir(src):
                opt = "g5"
            else:
                continue
            key = (plat, src.split("/")[0] if src.startswith(("q2", "q1", "q4")) else src, r["model"], int(r["length"]),
                   r["fmt"], r["config"], r["bw_name"], st, kern)
            c = cells.setdefault(key, dict(cake=[], full=[], g5=[], bytes=[], nlow=[], n=int(r["n_blocks"]), bw=set(), runs=set(),
                                           src=set()))
            c[opt].append(float(r["t_restore"]))
            c["src"].add(src)
            c["runs"].add(r["run_id"])
            if r.get("bw_GiBps") not in (None, "", "real"):
                c["bw"].add(float(r["bw_GiBps"]))
            if opt == "full":
                c["bytes"].append(int(float(r["bytes_read"])))
                c["nlow"].append(sum(int(float(r.get(k) or 0)) for k in ("n_load_fp8", "n_load_int8", "n_load_int4")))
    out = []
    for k, c in cells.items():
        if len(c["cake"]) < MIN_REPS or len(c["full"]) < MIN_REPS:
            continue
        plat, src, model, L, fmt, cfg, tier, st, kern = k
        out.append(dict(platform=plat, source=src, model=model, length=L, fmt=fmt, config=cfg, tier=tier, state=st, kernel=kern,
                        n_blocks=c["n"], t_cake=statistics.median(c["cake"]), t_full=statistics.median(c["full"]),
                        t_g5=statistics.median(c["g5"]) if len(c["g5"]) >= MIN_REPS else "", n_cake=len(c["cake"]),
                        n_full=len(c["full"]), bytes_full=statistics.median(c["bytes"]), n_low=statistics.median(c["nlow"]),
                        bw_GiBps=(min(c["bw"]) if c["bw"] else ""), src_dirs="|".join(sorted(c["src"])),
                        run_ids="|".join(sorted(c["runs"])), cake_reps=list(c["cake"]), full_reps=list(c["full"])))
    return out, excl


# ============================================================================ 微基準常數
def bw_json(plat: str, model: str, kern: str) -> dict:
    if kern == "fa3":
        p = os.path.join(R, "icpe_h200", "p01", "calib", f"speed_{model}", "bw.json")
    else:
        p = os.path.join(R, f"m10_precision_{plat}", f"speed_{model}", "bw.json")
    return json.load(open(p)) | {"_path": os.path.relpath(p, REPO)}


def fit_line(ys: list) -> tuple:
    n = len(ys)
    xs = list(range(n))
    mx, my = sum(xs) / n, sum(ys) / n
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - a * mx, a


def c5(plat: str, kern: str) -> dict:
    p = os.path.join(R, "icpe_h200", "p01", "calib", "calib_c5.csv") if kern == "fa3" else \
        os.path.join(R, f"m7_write_policy_{plat}", "calib_c5.csv")
    by = {}
    for r in csv.DictReader(open(p)):
        by.setdefault(r["duty"], []).append(float(r["slowdown"]))
    return {d: statistics.median(v) for d, v in by.items()} | {"_path": os.path.relpath(p, REPO)}


def parts(plat: str, model: str) -> dict:
    p = os.path.join(R, f"m10_sweep_{plat}", "parts", f"parts_{model}.json")
    d = json.load(open(p))["median_per_block"]
    return d | {"_path": os.path.relpath(p, REPO)}


FMT_PARTS = {"fp8_token": "fp8_token", "fp8_head": "fp8_head", "int4_kivi": "int4_kivi", "int8_token": "int8_token",
             "int4_token": "int4_token"}


def constants(cell) -> dict:
    """登記 §2：測試那一格的平台、模型、kernel 的微基準常數。"""
    plat, model, kern = cell["platform"], cell["model"], cell["kernel"]
    bw = bw_json(plat, model, kern)
    f0, a = fit_line(bw["f_ms_median_by_chunk"])
    pt = parts(plat, model)
    fmt = cell["fmt"]
    pf = FMT_PARTS.get(fmt, fmt)
    idle_l = pt["bf16|idle"]["t_h2d_ms"]
    busy_l = pt.get("bf16|busy", {}).get("t_h2d_ms")
    di = (pt.get(f"{pf}|idle") or {}).get("t_dequant_torch_ms")
    db = (pt.get(f"{pf}|busy") or {}).get("t_dequant_torch_ms")
    cc = c5(plat, kern)
    busy = cell["state"] == "busy"
    m_c = cc.get(STD_BUSY[plat]) if busy else 1.0
    if busy and m_c is None:
        raise SystemExit(f"{plat} {kern} 的 calib_c5 沒有 duty {STD_BUSY[plat]}（規則 7）")
    if cell["tier"] == "cpu":
        B = float(bw["real_pinned_h2d_GiBps"])
    else:
        B = float(cell["bw_GiBps"])
    return dict(f0=f0 / 1e3, alpha=a / 1e3, B=B * GiB, m_c=m_c, m_l=(busy_l / idle_l if busy and busy_l else 1.0),
                d=(di or 0.0) / 1e3, m_d=((db / di) if busy and di and db else 1.0), src_bw=bw["_path"], src_parts=pt["_path"],
                src_c5=cc["_path"])


def predict(cell, k: dict, c0: float = 0.0, c1: float = 0.0) -> dict:
    N = int(cell["n_blocks"])
    s = float(cell["bytes_full"]) / N
    q = float(cell["n_low"]) / N
    per_load = k["m_l"] * s / k["B"] + k["m_d"] * q * k["d"]
    cum, best_b, best = 0.0, 0, None
    for b in range(N + 1):
        t = max(k["m_c"] * cum, (N - b) * per_load)
        if best is None or t < best:
            best, best_b = t, b
        if b < N:
            cum += k["f0"] + k["alpha"] * b
    t_cake, t_full = best + c0, N * per_load + c1
    return dict(t_cake_pred=t_cake, t_full_pred=t_full, b_star=best_b, choice="cake" if t_cake < t_full else "full")


def regret(cell, choice: str) -> float:
    best = min(cell["t_cake"], cell["t_full"])
    return (cell["t_cake"] if choice == "cake" else cell["t_full"]) / best - 1


# ============================================================================ 切法
SPLITS = {
    "length": lambda c: ("train" if c["length"] == 32768 else "test"),
    "load": lambda c: ("train" if c["state"] == "idle" else "test"),
    "platform": lambda c: ("train" if c["platform"] == "mi300x" else "test"),
    "kernel": lambda c: (None if c["platform"] != "h200" else ("train" if c["kernel"] == "sdpa" else ("test" if c["kernel"] == "fa3" else None))),
}
PER_PLATFORM = {"length": True, "load": True, "platform": False, "kernel": False}


def fit_overheads(train: list) -> tuple:
    d0 = [c["t_cake"] - predict(c, constants(c))["t_cake_pred"] for c in train]
    d1 = [c["t_full"] - predict(c, constants(c))["t_full_pred"] for c in train]
    return (statistics.median(d0) if d0 else 0.0), (statistics.median(d1) if d1 else 0.0)


def evaluate(cells: list) -> dict:
    res = {}
    for name, fn in SPLITS.items():
        groups = {}
        for c in cells:
            side = fn(c)
            if side is None:
                continue
            g = c["platform"] if PER_PLATFORM[name] else "all"
            groups.setdefault(g, {"train": [], "test": []})[side].append(c)
        rows, fits = [], {}
        for g, d in groups.items():
            if not d["train"] or not d["test"]:
                continue
            c0, c1 = fit_overheads(d["train"])
            fits[g] = dict(c0=c0, c1=c1, n_train=len(d["train"]), n_test=len(d["test"]))
            for c in d["test"]:
                p = predict(c, constants(c), c0, c1)
                rows.append(dict(cell=c, pred=p, regret=regret(c, p["choice"]),
                                 regret_always_cake=regret(c, "cake"), regret_always_full=regret(c, "full"),
                                 regret_g5=(c["t_g5"] / min(c["t_cake"], c["t_full"]) - 1) if c["t_g5"] != "" else None))
        if not rows:
            res[name] = dict(status="NOT_MEASURED", fits=fits)
            continue
        rg = [r["regret"] for r in rows]
        g5 = [r["regret_g5"] for r in rows if r["regret_g5"] is not None]
        res[name] = dict(n_test=len(rows), mean=statistics.fmean(rg), worst=max(rg), fits=fits,
                         always_cake=dict(mean=statistics.fmean(r["regret_always_cake"] for r in rows),
                                          worst=max(r["regret_always_cake"] for r in rows)),
                         always_full=dict(mean=statistics.fmean(r["regret_always_full"] for r in rows),
                                          worst=max(r["regret_always_full"] for r in rows)),
                         g5=(dict(n=len(g5), mean=statistics.fmean(g5), worst=max(g5)) if g5 else None),
                         go=statistics.fmean(rg) <= 0.01 and max(rg) <= 0.05,
                         worst_cells=sorted(({k: r["cell"][k] for k in ("platform", "model", "length", "fmt", "config", "tier", "state",
                                                                        "kernel", "t_cake", "t_full")} | {"choice": r["pred"]["choice"],
                                                                                                          "regret": r["regret"]}
                                             for r in rows), key=lambda x: -x["regret"])[:5])
    meas = [v for v in res.values() if v.get("status") != "NOT_MEASURED"]
    verdict = "GC-GO" if len(meas) == 4 and all(v["go"] for v in meas) else ("GC-PARTIAL" if any(v.get("go") for v in meas) else "GC-FAIL")
    return dict(splits=res, verdict=verdict)


FIELDS = ["platform", "source", "model", "length", "fmt", "config", "tier", "state", "kernel", "n_blocks", "t_cake", "t_full", "t_g5",
          "n_cake", "n_full", "bytes_full", "n_low", "bw_GiBps", "src_dirs", "run_ids"]


def cmd_cells(a):
    cells, excl = build_cells()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "cells.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "ts"] + FIELDS)
        w.writeheader()
        rid = run_id()
        for c in cells:
            w.writerow({k: v for k, v in dict(c, run_id=rid, ts=ts()).items() if k in ["run_id", "ts"] + FIELDS})
    by = {}
    for c in cells:
        k = (c["platform"], c["kernel"], c["length"], c["state"])
        by[k] = by.get(k, 0) + 1
    print(json.dumps(dict(n_cells=len(cells), excluded_rows=excl), ensure_ascii=False))
    for k, v in sorted(by.items()):
        print(k, v)


def cmd_eval(a):
    cells, excl = build_cells()
    res = evaluate(cells)
    doc = dict(run_id=run_id(), ts=ts(), n_cells=len(cells), excluded_rows=excl, **res)
    os.makedirs(OUT, exist_ok=True)
    json.dump(doc, open(os.path.join(OUT, "c0_report.json"), "w"), indent=1, ensure_ascii=False, default=str)
    md = [f"# C0：還原方式預測（登記 runs/C0_cost_model.md）", "", f"run {doc['run_id']}；格子 {len(cells)}；判定 **{res['verdict']}**", "",
          "| 切法 | 測試格 | 平均代價 | 最差 | 過（≤1%／≤5%） | 永遠 Cake 平均／最差 | 永遠全部載入 平均／最差 | G5／G5b（有量的格） |",
          "|:--|--:|--:|--:|:--|--:|--:|--:|"]
    for k, v in res["splits"].items():
        if v.get("status") == "NOT_MEASURED":
            md.append(f"| {k} | NOT_MEASURED | | | | | | |")
            continue
        g5 = f"{100 * v['g5']['mean']:.1f}%／{100 * v['g5']['worst']:.1f}%（n＝{v['g5']['n']}）" if v["g5"] else "—"
        md.append(f"| {k} | {v['n_test']} | {100 * v['mean']:.2f}% | {100 * v['worst']:.1f}% | {'是' if v['go'] else '否'} | "
                  f"{100 * v['always_cake']['mean']:.1f}%／{100 * v['always_cake']['worst']:.1f}% | "
                  f"{100 * v['always_full']['mean']:.1f}%／{100 * v['always_full']['worst']:.1f}% | {g5} |")
    open(os.path.join(OUT, "tables_c0.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


def ref2sigma(cells: list) -> dict:
    """05 §2.1 的雜訊規則「事後」套到 C0-v1（只供參考，不改 GC-FAIL）：每個測試格，δ*＝max(5%, 2σ)，σ＝最好那一臂的
    t_restore 相對標準差（≥6 次才算）；選的那一臂離最好 ≤ δ* 就算 0 代價。2σ > 5% 或不到 6 次的格子不算、另計數。"""
    out = {}
    for name, fn in SPLITS.items():
        groups = {}
        for c in cells:
            side = fn(c)
            if side is None:
                continue
            g = c["platform"] if PER_PLATFORM[name] else "all"
            groups.setdefault(g, {"train": [], "test": []})[side].append(c)
        rg, n_nm, n_noisy = [], 0, 0
        for g, d in groups.items():
            if not d["train"] or not d["test"]:
                continue
            c0, c1 = fit_overheads(d["train"])
            for c in d["test"]:
                best = c["cake_reps"] if c["t_cake"] <= c["t_full"] else c["full_reps"]
                if len(best) < 6:
                    n_nm += 1
                    continue
                sr = statistics.stdev(best) / statistics.median(best)
                if 2 * sr > 0.05:
                    n_noisy += 1
                    continue
                ds = max(0.05, 2 * sr)
                r = regret(c, predict(c, constants(c), c0, c1)["choice"])
                rg.append(0.0 if r <= ds else r)
        out[name] = dict(n=len(rg), mean=(statistics.fmean(rg) if rg else None), worst=(max(rg) if rg else None),
                         n_not_measured=n_nm, n_noisy=n_noisy)
    return out


def cmd_ref2sigma(a):
    cells, _ = build_cells()
    res = dict(run_id=run_id(), ts=ts(), note="事後以 05 §2.1 的 2σ 規則重算，只供參考；C0-v1 的判定（GC-FAIL）不變", splits=ref2sigma(cells))
    json.dump(res, open(os.path.join(OUT, "c0_ref2sigma.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res["splits"], ensure_ascii=False))


def self_test() -> bool:
    k = dict(f0=0.01, alpha=0.0, B=10 * GiB, m_c=1.0, m_l=1.0, d=0.0, m_d=1.0)
    cell = dict(n_blocks=64, bytes_full=64 * (1 << 26), n_low=0)
    p = predict(cell, k)
    # 每塊載入 64 MiB／10 GiB/s＝6.25 ms、重算 10 ms → 會合點約 N·6.25/(10+6.25)＝24.6 → b*＝25（max(0.25, 0.244)）或 24
    ok = p["choice"] == "cake" and p["b_star"] in (24, 25) and abs(p["t_full_pred"] - 0.4) < 1e-9
    k2 = dict(k, B=1000 * GiB)
    ok &= predict(cell, k2)["choice"] == "full"
    ok &= abs(regret(dict(t_cake=1.0, t_full=2.0), "full") - 1.0) < 1e-12
    print("self-test", "PASS" if ok else "FAIL", p)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("cells", "eval", "ref2sigma"):
        s = sub.add_parser(name)
        s.add_argument("--out-dir", default="", help="輸出目錄（預設 results/s3_h200/c0；C0-v2 用 results/s3_h200/c0/v2，不覆蓋 v1）")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if getattr(a, "out_dir", ""):
        global OUT
        OUT = a.out_dir
    {"cells": cmd_cells, "eval": cmd_eval, "ref2sigma": cmd_ref2sigma}[a.cmd](a)


if __name__ == "__main__":
    main()
