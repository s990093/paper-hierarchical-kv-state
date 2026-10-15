"""m10_e0_t4_breakdown.py — E0 調難度時 T4（HotpotQA）的拆解：題型、答案字數 × 兩種計分（CPU，不用 GPU）

為什麼：第 1、2 輪 T4 的 BF16 正確率低於 60%，要分清是模型不會，還是 string_match_part 太嚴
（"Steve Coogan" 對 "Stephen John Coogan" 算錯）。由 seed 重建題目（make_item 可重現）取得題型與答案字數。
用法：python code/m10_e0_t4_breakdown.py <逐題 CSV>...（只看 task==T4 的列）
"""
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m10_tasks as T  # noqa: E402


def main(paths):
    toks, meta = {}, {}
    by = collections.defaultdict(list)
    for p in paths:
        for r in csv.DictReader(open(p)):
            if r["task"] != "T4":
                continue
            m = r["model"]
            toks.setdefault(m, T.load_tokenizer(m))
            key = (m, r["item_id"], r["params"])
            if key not in meta:
                b1, b2 = r["item_id"].split("_")[1].split("-")
                it = T.make_item("T4", b1, b2, int(r["seed"]), toks[m], json.loads(r["params"]))
                meta[key] = {t["turn"]: (t.get("hotpot_type"), len(t["gold"][0].split())) for t in it["turns"]}
            typ, nw = meta[key][int(r["turn"])]
            x = (float(r["score_up"]), T.qa_f1(r["pred"], json.loads(r["gold"])))
            for k in (("type", typ), ("ans_words", "1-2" if nw <= 2 else "3+"), ("all", "-")):
                by[(m, os.path.basename(p)) + k].append(x)
    print("model,file,group,value,n,string_match_part,qa_f1")
    for k, v in sorted(by.items()):
        print(",".join(map(str, k)) + f",{len(v)},{sum(a for a, _ in v) / len(v):.3f},{sum(b for _, b in v) / len(v):.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
