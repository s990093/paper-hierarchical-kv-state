"""h200_longctx256_analyze.py — 256K（Qwen × 508 個 block × T2–T4）的彙整（13 §11.2；探索，不做判定）。CPU。

m10_longctx_analyze.py 的長度（32K／64K／128K）寫死在表與圖裡，而且 MI300X 也用它；所以 256K 另寫這支，
函式全部借用它（load、pick、harm、boot、main3、speed 的同一套定義），不改它。
256K 只有 T2–T4（T5 在這個長度出不了題），所以和其他長度比時一律只用 T2–T4 的題（同一個（任務, seed）配對）。

注意：m10_longctx_analyze.load() 會讀 --raw 底下所有 screen_qwen_*_s*.csv.gz；256K 的檔（screen_qwen_508_*）放進同一個目錄之後，
直接重跑 m10_longctx_analyze.py 會遇到 LEN 沒有 508 → KeyError。要重跑 32K–128K 的分析時，--raw 指到不含 508 檔的目錄。

  M10_PRECISION_DIR=results/m10_precision_h200 python code/h200_longctx256_analyze.py \
      --raw ~/longctx_m10h --lc results/m10_longctx_h200 --out-dir results/m10_longctx_h200/analysis
輸出：longctx256_qwen.json、tables_longctx256.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m10_longctx_analyze as LA  # noqa: E402

M = "qwen"
NB = 508
TOK = 260096
T234 = ("T2", "T3", "T4")
LEN = {**LA.LEN, NB: "256K"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--lc", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    rows = [r for r in LA.load(a.raw, M) if r["task"] in T234]
    nbs = sorted({r["nb"] for r in rows})
    if NB not in nbs:
        sys.exit(f"--raw 裡沒有 {M} 的 {NB} 個 block（screen_{M}_{NB}_s*.csv.gz）")
    res = dict(model=M, tasks=list(T234), note="只有 T2–T4；其他長度也只取 T2–T4 的題（13 §11.2）",
               run_ids=sorted({r["run_id"] for r in rows if r["nb"] == NB}),
               n_items={LEN[n]: len({r["key"] for r in rows if r["nb"] == n}) for n in nbs})
    # 1. 生成答對率（每個長度，T2–T4）
    res["acc"] = {}
    for nb in nbs:
        e = LA.pick(rows, nb, None, "E0", 0.0)
        ent = {"E0": LA.boot({k: r["gen"] for k, r in e.items() if r["gen"] is not None})}
        for f in LA.main3(M):
            x = LA.pick(rows, nb, f, "ALL", 1.0)
            ent[f] = LA.boot({k: r["gen"] for k, r in x.items() if r["gen"] is not None})
        res["acc"][LEN[nb]] = ent
    # 2. 傷害（ALL 對 E0）與「256K − 32K」「256K − 128K」（同一題配對）
    res["harm"] = {}
    for f in sorted({r["fmt"] for r in rows if r["policy"] == "ALL" and r["nb"] == NB}):
        ent = {}
        for metric in ("gen", "kl", "first"):
            per = {nb: LA.harm(rows, nb, f, metric) for nb in nbs}
            ent[metric] = {LEN[nb]: LA.boot(per[nb]) for nb in nbs}
            for lo_nb in (64, 252):
                if lo_nb in per:
                    common = set(per[NB]) & set(per[lo_nb])
                    d = LA.boot({k: per[NB][k] - per[lo_nb][k] for k in common})
                    if d:
                        d["grows_with_length"] = bool((d["hi"] < 0) if metric in ("gen", "first") else (d["lo"] > 0))
                    ent[metric][f"256K−{LEN[lo_nb]}"] = d
        res["harm"][f] = ent
    # 3. 位置（25%：F／M／B 對 R0）
    res["position"] = {}
    for f in LA.main3(M):
        r0 = LA.pick(rows, NB, f, "R0", 0.25)
        for pol in ("F", "M", "B"):
            x = LA.pick(rows, NB, f, pol, 0.25)
            common = set(x) & set(r0)
            g = LA.boot({k: x[k]["gen"] - r0[k]["gen"] for k in common if x[k]["gen"] is not None and r0[k]["gen"] is not None})
            kl = LA.boot({k: x[k]["kl"] - r0[k]["kl"] for k in common})
            res["position"][f"256K|{f}|{pol}"] = dict(gen_diff=g, kl_diff=kl)
    # 4. 讀法 B
    res["readout_b"] = {}
    for f in LA.main3(M):
        for ro in ("A", "B_idle", "B_busy"):
            x = LA.pick(rows, NB, f, "ALL", 1.0, ro)
            if x:
                res["readout_b"][f"256K|{f}|{ro}"] = dict(lp_read=round(float(np.mean([r["lp_read"] for r in x.values()])), 2),
                                                          kl=LA.boot({k: r["kl"] for k, r in x.items()}))
    # 5. 速度（E1 對 E0；cake 中位數）
    res["speed"] = {}
    for tier in ("fp8", "int4"):
        d = os.path.join(a.lc, f"speed_{M}_{TOK}_{LA.fmt_of(M, tier)}")
        p = os.path.join(d, "summary.csv")
        if not os.path.exists(p):
            res["speed"][f"256K|{tier}"] = "NOT_MEASURED"
            continue
        med = defaultdict(dict)
        for r in csv.DictReader(open(p)):
            if r["mode"] == "cake" and r["config"] in ("E0", "E1"):
                med[(r["bw_name"], r["gpu_state"])][r["config"]] = float(r["median_t_restore"])
        dirty = [os.path.basename(g) for g in glob.glob(os.path.join(d, "gpu_guard_*.json"))
                 if json.load(open(g)).get("verdict") != "CLEAN"]
        res["speed"][f"256K|{tier}"] = dict(cells={f"{k[0]}|{k[1]}": dict(E0=v.get("E0"), E1=v.get("E1"),
                                                                          gain=round((v["E0"] - v["E1"]) / v["E0"], 4)
                                                                          if "E0" in v and "E1" in v else None)
                                                   for k, v in sorted(med.items())}, dirty_guards=dirty)
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(a.out_dir, "longctx256_qwen.json"), "w"), indent=1, ensure_ascii=False)

    def ci(x, pct=True):
        if not x:
            return "—"
        s = 100 if pct else 1
        return f"{x['mean'] * s:+.1f} [{x['lo'] * s:+.1f}, {x['hi'] * s:+.1f}]（n＝{x['n']}）"

    def acc(x):
        return "—" if not x else f"{x['mean'] * 100:.1f} [{x['lo'] * 100:.1f}, {x['hi'] * 100:.1f}]"

    L = ["# 256K（Qwen × 508 個 block × T2–T4；13 §11.2；探索，不做判定）", "",
         f"run：{'、'.join(res['run_ids'])}；題數 {res['n_items']}（每個長度都只取 T2–T4）", "",
         "### 1. 生成答對率（%，95% CI；T2–T4）", "", "| 長度 | E0 | " + " | ".join(f"ALL {f}" for f in LA.main3(M)) + " |",
         "|:--|--:|" + "--:|" * 3]
    for nb in nbs:
        e = res["acc"][LEN[nb]]
        L.append(f"| {LEN[nb]} | {acc(e['E0'])} | " + " | ".join(acc(e[f]) for f in LA.main3(M)) + " |")
    L += ["", "### 2. 傷害（ALL 對 E0；生成答對率差，百分點；T2–T4 配對；CI 不含 0 且變差才標 ↑）", "",
          "| 格式 | 32K | 128K | 256K | 256K−32K | 256K−128K |", "|:--|:--|:--|:--|:--|:--|"]
    for f, ent in res["harm"].items():
        g = ent["gen"]
        fl = lambda k: " ↑" if g.get(k) and g[k].get("grows_with_length") else ""  # noqa: E731
        L.append(f"| {f} | {ci(g.get('32K'))} | {ci(g.get('128K'))} | {ci(g.get('256K'))} | "
                 f"{ci(g.get('256K−32K'))}{fl('256K−32K')} | {ci(g.get('256K−128K'))}{fl('256K−128K')} |")
    L += ["", "### 3. 位置（25%：F／M／B 對 R0；生成答對率差，百分點；KL 差負＝比隨機好）", "",
          "| 格式 | 放法 | 生成差 | KL 差 |", "|:--|:--|:--|:--|"]
    for k, v in res["position"].items():
        _, f, pol = k.split("|")
        L.append(f"| {f} | {pol} | {ci(v['gen_diff'])} | {ci(v['kl_diff'], pct=False)} |")
    L += ["", "### 4. 讀法 B（ALL）", "", "| 格式 | 讀法 | 讀到的低精度 block | KL |", "|:--|:--|--:|:--|"]
    for k, v in res["readout_b"].items():
        _, f, ro = k.split("|")
        L.append(f"| {f} | {ro} | {v['lp_read']} | {ci(v['kl'], pct=False)} |")
    L += ["", "### 5. 速度（E1 對 E0 快多少；cake 中位數，秒）", "", "| 階 | 格 | E0 | E1 | 快 |", "|:--|:--|--:|--:|--:|"]
    for k, v in res["speed"].items():
        if isinstance(v, str):
            L.append(f"| {k} | {v} | | | |")
            continue
        for c, x in v["cells"].items():
            gain = "" if x["gain"] is None else f"{x['gain'] * 100:+.1f}%"
            L.append(f"| {k.split('|')[1]} | {c.replace('|', '・')} | {x['E0']} | {x['E1']} | {gain} |")
        if v["dirty_guards"]:
            L.append(f"| {k} | gpu_guard 不是 CLEAN：{v['dirty_guards']}（時間不能用） | | | |")
    open(os.path.join(a.out_dir, "tables_longctx256.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
