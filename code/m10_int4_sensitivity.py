"""m10_int4_sensitivity.py — INT4 品質全過之後的敏感度檢查（探索，只報告，不改任何判定）。

使用者 10/16 問「INT4 兩個模型都過，會不會是題目太簡單、context 太短、程式有一半錯？」。
這裡只用已經有的品質列（讀法 A＝全部讀到低精度），回答「題目看不看得到低精度的傷害」：
  1. E0 的答對率（天花板／地板？）
  2. 每種格式和 E0 輸出逐字相同的比例、teacher forcing 的 KL（中位數、p90）、K 的相對誤差
     → 看是不是 INT8 < FP8 < INT4 的劑量反應
  3. 標準答案每個 token 的 log 機率變化（對 E0 配對），每個任務的平均與 95% CI
     （比答對率敏感：答案沒翻、但模型對答案的信心掉了也看得到）
輸入：quality/step1_<m>_*.csv（E0、FP8 的 E1、Step 2b 的 E1-K0／E1-INT8）、quality_sealed/int4end_<m>_*.csv。
用法：python code/m10_int4_sensitivity.py  → analysis/int4/sensitivity.{json,md}
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import statistics as st

R = os.environ.get("M10_PRECISION_DIR") or "/mlsteam/workspace/paper-hierarchical-kv-state/results/m10_precision_mi300x"   # H200：環境變數（10/19；沒設時不變）
CFGS = {"llama": [("step1", "E1s@int8_token", "INT8（每 token）"), ("step1", "E1s@fp8k0_token", "FP8 K 先減平均"),
                  ("step1", "E1", "FP8（主線）"), ("int4end", "E1@int4_kivi", "INT4 KIVI")],
        "qwen": [("step1", "E1s@int8_token", "INT8（每 token）"), ("step1", "E1s@fp8k0_head", "FP8 K 先減平均"),
                 ("step1", "E1", "FP8（主線）"), ("int4end", "E1@int4_kivi", "INT4 KIVI")]}


def load(pat):
    rows = []
    for f in sorted(glob.glob(pat)):
        rows += list(csv.DictReader(open(f)))
    return rows


def ci(x):
    mu = st.mean(x)
    return mu, 1.96 * st.stdev(x) / math.sqrt(len(x))


def main():
    out, md = {}, ["# INT4 敏感度檢查（讀法 A；探索，只報告）", ""]
    for m in ("llama", "qwen"):
        src = {"step1": load(f"{R}/quality/step1_{m}_*.csv"), "int4end": load(f"{R}/quality_sealed/int4end_{m}_*.csv")}
        runs = sorted({r["run_id"] for v in src.values() for r in v})
        key = lambda r: (r["item_id"], r["turn"], r["turn2_variant"])
        e0 = {key(r): r for r in src["step1"] if r["config"] == "E0" and r["mode"] == "A"}
        acc = {}
        for r in e0.values():
            acc.setdefault(f"{r['task']} 第{r['turn']}輪", []).append(float(r["score"]))
        res = {"run_ids": runs, "n_e0": len(e0), "e0_acc": {k: round(100 * st.mean(v), 1) for k, v in sorted(acc.items())},
               "formats": {}}
        md += [f"## {m}", "", "E0 答對率（%）：" + "、".join(f"{k} {v}" for k, v in res["e0_acc"].items()), "",
               "| 格式 | 和 E0 逐字相同 | KL 中位數 | KL p90 | K 相對誤差 | 每 token Δlog p（全部） | T2 | T3 | T4 | T5 |",
               "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for s, cfg, name in CFGS[m]:
            x = [r for r in src[s] if r["config"] == cfg and r["mode"] == "A" and key(r) in e0]
            kl = sorted(float(r["kl_vs_e0"]) for r in x)
            d = {}
            for r in x:
                d.setdefault(r["task"], []).append((float(r["gold_logprob"]) - float(e0[key(r)]["gold_logprob"]))
                                                   / int(r["gold_len"]))
            allx = sum(d.values(), [])
            f = dict(config=cfg, n=len(x), same_as_e0=round(st.mean(int(r["same_as_e0"]) for r in x), 3),
                     kl_median=round(st.median(kl), 4), kl_p90=round(kl[int(0.9 * len(kl))], 4),
                     relK_mean=round(st.mean(float(r["relK_read"]) for r in x), 4),
                     dlogp_all=[round(v, 4) for v in ci(allx)],
                     dlogp_task={t: [round(v, 4) for v in ci(v)] for t, v in sorted(d.items())})
            res["formats"][name] = f
            cell = lambda t: "%+.4f ± %.4f" % tuple(f["dlogp_task"][t]) if t in f["dlogp_task"] else "—"
            dall = "%+.4f ± %.4f" % tuple(f["dlogp_all"])
            md.append(f"| {name}（`{cfg}`） | {100 * f['same_as_e0']:.1f}% | {f['kl_median']} | {f['kl_p90']} | "
                      f"{f['relK_mean']} | {dall} | {cell('T2')} | {cell('T3')} | {cell('T4')} | {cell('T5')} |")
        md += ["", "run：" + "、".join(f"`{r}`" for r in runs), ""]
        out[m] = res
    md += ["每 token Δlog p＝（低精度的標準答案 log 機率 − E0 的）÷ 答案 token 數，同一題配對；± 是 95% CI 半寬。",
           "負的＝模型對標準答案的信心變低。exp(−0.01)≈0.99，也就是每個 token 的機率約低 1%。"]
    d = os.environ.get("M10_SENS_OUT") or f"{R}/analysis/int4"   # H200：判定沒走到 INT4，輸出另放（10/19；沒設時不變）
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(f"{d}/sensitivity.json", "w"), ensure_ascii=False, indent=1)
    open(f"{d}/sensitivity.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
