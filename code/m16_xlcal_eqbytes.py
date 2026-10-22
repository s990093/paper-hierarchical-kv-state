#!/usr/bin/env python3
"""m16_xlcal_eqbytes.py — 38k（xKV 事先校準的共用基底）的「同位元組」重新分析。只用 CPU、只讀 38k 的逐題檔。

**事後分析**（10/22 使用者在看過 38k 的 UNDETERMINED 之後要求）：只當探索，不改 38k 的判定（experiment_log.md「10/22 15:3xZ 起：補跑」）。
問題：共用基底不必每請求帶基底，所以同樣（或更少）的每請求位元組可以用更高的秩。38k 的主比較是「同秩」；這裡改比
「共用基底 k28＋INT4」對「每題 SVD k22＋INT4」——每請求位元組相近（Llama 0.085 vs 0.096，共用基底少 11%；Qwen 0.085 vs 0.082，多 4%）。
bootstrap 的亂數與 m16_xlcal3 不同，CI 端點可能差 0.1 個百分點；38k 的判定以 xlcal3_report.json 為準。
用 38k 新的 192 題（seed 66000 起；38k 的主判定題組）逐題配對，bootstrap 10,000 次（同 m16_xstack 的做法：百分位 95% CI）。

  python code/m16_xlcal_eqbytes.py --out-dir results/m16_stage3_mi300x/x_stack/v3
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import random
from datetime import datetime, timezone

REPORT = "results/m16_stage3_mi300x/x_stack/v3/xlcal3_report.json"
CMP = [("xl:k28:v42:cal+q:int4_kivi", "xl:k22:v34+q:int4_kivi", "共用基底 k28 vs 每題 SVD k22（每請求位元組相近，見位元組欄）"),
       ("xl:k22:v34:cal+q:int4_kivi", "xl:k22:v34+q:int4_kivi", "同秩 k22（38k 的主比較之一，對照用）"),
       ("xl:k28:v42:cal+q:int4_kivi", "xl:k28:v42+q:int4_kivi", "同秩 k28（38k 的主比較之一，對照用）")]
N_BOOT = 10000


def boot(v, seed=20261022):
    rng = random.Random(seed)
    n = len(v)
    ms = sorted(sum(v[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    return dict(mean=sum(v) / n, lo=ms[int(0.025 * N_BOOT)], hi=ms[int(0.975 * N_BOOT) - 1], p_le0=sum(m <= 0 for m in ms) / N_BOOT, n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    rep = json.load(open(REPORT))
    rows = []
    for f in rep["files_new"]:
        with (gzip.open(f, "rt") if f.endswith(".gz") else open(f)) as fh:
            rows += list(csv.DictReader(fh))
    new_items = {m: set(rep["variants"]["new192"]["models"][m]["items"]) for m in rep["variants"]["new192"]["models"]}
    by = {}
    for r in rows:
        if r["item_id"] in new_items.get(r["model"], ()):
            by.setdefault((r["model"], r["item_id"]), {})[r["spec"]] = r
    out = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now(timezone.utc).isoformat(timespec="seconds"), source_report=REPORT,
               source_run=rep["run_id"], kind="事後分析（探索，不是判定）", models={})
    L = ["# 38k 同位元組重新分析（事後探索，不改 38k 的判定）", "", f"來源：{REPORT}（run {rep['run_id']}）；只用新的 192 題。", "",
         "| 模型 | 比較 | 每請求位元組比（A／B） | 題數 | 答對率差 A−B（百分點）[95% CI] | P(≤0) |", "|:--|:--|:--|--:|:--|--:|"]
    for m in sorted(new_items):
        res = []
        for A, B, why in CMP:
            d, ra, rb = [], None, None
            for (mm, it), e in by.items():
                if mm == m and A in e and B in e:
                    d.append(float(e[A]["gen_score"]) - float(e[B]["gen_score"]))
                    ra, rb = float(e[A]["ratio_region"]), float(e[B]["ratio_region"])
            ci = boot(d) if d else None
            res.append(dict(A=A, B=B, why=why, ratio_A=ra, ratio_B=rb, diff=ci))
            if ci:
                L.append(f"| {m} | {why} | {ra:.4f}／{rb:.4f} | {ci['n']} | {100 * ci['mean']:+.1f} [{100 * ci['lo']:+.1f}, {100 * ci['hi']:+.1f}] | {ci['p_le0']:.3f} |")
        out["models"][m] = res
    L += ["", "ε＝2 個百分點的對照：CI 下界 ≥ −2 才算「不輸」。這是看過 38k 之後才提的比較，只當探索。"]
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(a.out_dir, "xlcal_eqbytes.json"), "w"), indent=1, ensure_ascii=False)
    open(os.path.join(a.out_dir, "xlcal_eqbytes.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
