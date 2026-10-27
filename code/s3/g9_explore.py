#!/usr/bin/env python3
"""G9（投機式還原，04 §G）的零 GPU 探索：只讀 A0／A0-C 既有的 TF 與生成結果，不跑新的量測。

**探索性，不是登記的判定**（使用者 10/28「照建議」第 5 項）。回答一件事：用壓縮過的 KV 當「草稿」時，
它和完整 KV（E0）的輸出有多常一致——投機式還原的好處取決於這個比例。

兩個代理指標（都不是投機解碼真正的接受長度）：
  top1     TF：標準答案的每個位置上，配置的 argmax＝E0 的 argmax 的比例（a0_gpu.Item.metrics）。
           投機解碼用 greedy 驗證時，草稿 token 被接受的條件就是 argmax 相同；但 TF 走的是標準答案的路徑，
           不是 E0 自己生成的路徑。
  same     生成：整段輸出和 E0 完全相同（same_as_e0）。整段相同 → 每個草稿 token 都會被接受，所以是「全接受」的比例。

用法：python code/s3/g9_explore.py → results/s3_h200/g9/explore.json
"""
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a0 as A                                         # noqa: E402

OUT = os.path.join(A.REPO, "results", "s3_h200", "g9")


def per_item(rows: list, value: str) -> dict:
    """cid → {(item_id, turn): value}；同一題重複的列取第一個（同 a0.table）。"""
    return A.table(rows, value)


def summarize(tab: dict, fs: dict, tasks: dict) -> dict:
    out = {}
    for cid, v in sorted(tab.items()):
        if cid == "E0" or not v:
            continue
        vals = list(v.values())
        qs = statistics.quantiles(vals, n=10) if len(vals) >= 10 else None
        t5 = [x for k, x in v.items() if tasks.get(k) == "T5"]
        t24 = [x for k, x in v.items() if tasks.get(k) != "T5"]
        out[cid] = dict(f=fs.get(cid), n=len(vals), boot=A.boot(vals), p10=qs[0] if qs else None, median=statistics.median(vals),
                        mean_T2_T4=statistics.fmean(t24) if t24 else None, mean_T5=statistics.fmean(t5) if t5 else None)
    return out


def main():
    res = dict(run_id=A.run_id(), ts=A.ts(), note="探索性（不是登記的判定）；兩個指標都是代理，不是投機解碼的實際接受長度",
               tf_top1={}, gen_same={}, sources={})
    for cell in A.CELLS:
        fs = {c["cid"]: c["f"] for c in json.load(open(os.path.join(A.OUT, f"plan_{cell}.json")))["configs"]} \
            if os.path.exists(os.path.join(A.OUT, f"plan_{cell}.json")) else {}
        tf_paths = sorted(glob.glob(A.out_glob("tf", cell)))
        rows = A.read_rows(tf_paths)
        if rows:
            tasks = {(r["item_id"], r["turn"]): r["task"] for r in rows}
            res["tf_top1"][cell] = summarize(per_item(rows, "top1"), fs, tasks)
            res["sources"][f"tf_{cell}"] = sorted({r["run_id"] for r in rows})
        for name, pat in (("a0", A.out_glob("gen", cell)), ("a0c", os.path.join(A.RAW, f"a0c_gen_{cell}_s*.csv.gz"))):
            rows = A.read_rows(sorted(glob.glob(pat)))
            if not rows:
                continue
            tasks = {(r["item_id"], r["turn"]): r["task"] for r in rows}
            res["gen_same"].setdefault(cell, {})[name] = summarize(per_item(rows, "same_as_e0"), fs, tasks)
            res["sources"][f"gen_{name}_{cell}"] = sorted({r["run_id"] for r in rows})
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "explore.json")
    json.dump(res, open(p, "w"), indent=1, ensure_ascii=False, default=str)
    print("→", p)


if __name__ == "__main__":
    main()
