"""m10_fmt_select.py — 20 題小測挑 FP8 scale 粒度與 INT4 分組（03 開頭「10/15 晚改動」寫死的規則）

規則：只看 E1@<格式>（讀法 A）對 E0 的平均 KL（所有題、所有輪），每個模型、每一階（FP8、INT4）各自選最小的。
**不讀、不印任何答對率**（score／pred 欄不碰）。

用法：python code/m10_fmt_select.py --csv <小測 CSV…> --out <fmt_choice.json>
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict

TIERS = {"fp8": ["fp8_head", "fp8_channel", "fp8_token"], "int4": ["int4_token", "int4_kivi"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kl = defaultdict(list)                          # (model, fmt) -> [kl]
    items = defaultdict(set)
    run_ids = set()
    for p in a.csv:
        for r in csv.DictReader(open(p)):
            if r["config"] == "E0" or r["mode"] != "A":
                continue
            name, _, fmt = r["config"].partition("@")
            if name != "E1" or not fmt:
                continue
            kl[(r["model"], fmt)].append(float(r["kl_vs_e0"]))     # 只取 KL
            items[(r["model"], fmt)].add(r["item_id"])
            run_ids.add(r["run_id"])
    out = {"rule": "每個模型、每一階，E1（讀法 A）對 E0 的平均 KL 最小的格式（03 開頭 10/15 晚改動）；不看答對率",
           "run_ids": sorted(run_ids), "csv": a.csv, "mean_kl": {}, "n_rows": {}, "n_items": {}, "choice": {}}
    models = sorted({m for m, _ in kl})
    for m in models:
        out["mean_kl"][m], out["n_rows"][m], out["n_items"][m], out["choice"][m] = {}, {}, {}, {}
        for tier, fmts in TIERS.items():
            got = {f: statistics.mean(kl[(m, f)]) for f in fmts if kl.get((m, f))}
            if set(got) != set(fmts):
                raise SystemExit(f"{m} {tier}：缺格式 {set(fmts) - set(got)}")
            ns = {f: len(kl[(m, f)]) for f in fmts}
            if len(set(ns.values())) != 1:
                raise SystemExit(f"{m} {tier}：各格式的列數不同 {ns}")
            for f in fmts:
                out["mean_kl"][m][f] = round(got[f], 6)
                out["n_rows"][m][f] = ns[f]
                out["n_items"][m][f] = len(items[(m, f)])
            out["choice"][m][tier] = min(fmts, key=lambda f: got[f])
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in ("mean_kl", "choice")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
