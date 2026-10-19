"""x9_extra.py — X9 只報告的補充數字（不改判定）。

1. T1 的分類細節：每類邊的輸出長度、Δ、「多出來的 token」(Δ − 父輸出)；把死掉的 KV 改用「父的輸出 token」當分母。
   讀 x9_census.py 的逐邊檔（--census-run 指的 run 目錄）。
2. session 層級 vs block 層級的工作集：Σ_session（session 最長請求的 block 數）÷ 不重複 block 數。
   >1 表示用 session 當單位會重複計算共用的 block。

用法： python -I code/p3x/x9_extra.py --census-run <run 目錄> --out results/p3_explore/x9
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone

DS = os.environ.get("TIARA_DATASETS", "/ssd8/hungwei/paper-hkv/datasets")
BAILIAN = {
    "bailian_traceA": f"{DS}/qwen-bailian-usagetraces-anon/qwen_traceA_blksz_16.jsonl",
    "bailian_traceB": f"{DS}/qwen-bailian-usagetraces-anon/qwen_traceB_blksz_16.jsonl",
    "bailian_thinking": f"{DS}/qwen-bailian-usagetraces-anon/qwen_thinking_blksz_16.jsonl",
    "bailian_coder": f"{DS}/qwen-bailian-usagetraces-anon/qwen_coder_blksz_16.jsonl",
}
RUN_ID = os.environ.get("RUN_ID", "")


def ts():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census-run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert RUN_ID
    now = ts()
    src = os.path.basename(a.census_run.rstrip("/"))
    rows1 = []
    for name in ("bailian_traceA", "bailian_thinking", "bailian_coder"):
        fn = os.path.join(a.census_run, f"t1_edges_{name}.csv.gz")
        E = list(csv.DictReader(gzip.open(fn, "rt")))
        for e in E:
            for k in ("delta", "out_p", "in_p", "S", "Rplus", "Rminus"):
                e[k] = int(e[k])
        tot_out = sum(e["out_p"] for e in E)
        by = defaultdict(list)
        for e in E:
            by[e["cls"]].append(e)
        for cls, es in sorted(by.items()):
            extra = [e["delta"] - e["out_p"] for e in es]
            rows1.append(dict(run_id=RUN_ID, ts=now, source_run=src, trace=name, cls=cls, n_edges=len(es),
                              out_p_median=q([e["out_p"] for e in es], .5), delta_median=q([e["delta"] for e in es], .5),
                              extra_median=q(extra, .5), extra_q10=q(extra, .1), extra_q90=q(extra, .9),
                              in_p_median=q([e["in_p"] for e in es], .5)))
        # Bailian 每個父只有一個子（census：分岔率 0），所以逐邊加總＝逐父加總；這裡再斷言一次
        assert len({e['parent'] for e in E}) == len(E), f'{name}: 有父有兩個以上的子，逐邊加總不等於逐父'
        dm = sum(e["S"] - e["Rplus"] for e in E)
        dp = sum(e["S"] - e["Rminus"] for e in E)
        rows1.append(dict(run_id=RUN_ID, ts=now, source_run=src, trace=name, cls="ALL", n_edges=len(E),
                          dead_lb_over_parent_output=dm / tot_out, dead_ub_over_parent_output=dp / tot_out,
                          sum_parent_output_tok=tot_out))
    rows2 = []
    for name, path in BAILIAN.items():
        rws = [json.loads(l) for l in open(path)]
        by_id = {r["chat_id"]: r for r in rws}
        root = {}
        for r in rws:
            cur = r["chat_id"]
            path_ = []
            while cur not in root:
                path_.append(cur)
                p = by_id[cur]["parent_chat_id"]
                if p == -1 or p not in by_id:
                    root[cur] = cur
                    break
                cur = p
            rt = root[cur]
            for x in path_:
                root[x] = rt
        size = defaultdict(int)
        uniq = set()
        for r in rws:
            rt = root[r["chat_id"]]
            size[rt] = max(size[rt], len(r["hash_ids"]))
            uniq.update(r["hash_ids"])
        s = sum(size.values())
        rows2.append(dict(run_id=RUN_ID, ts=now, trace=name, n_sessions=len(size), sum_session_blocks=s,
                          unique_blocks=len(uniq), session_over_unique=s / len(uniq),
                          sum_session_GiB_llama=s * 16 * 128 / (1 << 20), unique_GiB_llama=len(uniq) * 16 * 128 / (1 << 20)))
    os.makedirs(a.out, exist_ok=True)
    for fn, rs in (("t1_breakdown.csv", rows1), ("t2_session_vs_block_workingset.csv", rows2)):
        keys = []
        for r in rs:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(os.path.join(a.out, fn), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rs)
        print(fn, json.dumps(rs, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
