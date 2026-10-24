"""x9b_switch.py — X9b 的 T7（I20：改 system prompt／工具定義之後會不會切回舊前綴），FreeInference。

判準在 X9 文件 §2.5（2026-10-17T13:41:37Z 寫定）。只做計數。
切回＝同 session 之後的請求 r 有 len(r) > L 且 r[L] == p[L]（鏈式 block：r 沿著 p 的舊鏈超過分岔點）。
用法： python -I code/p3x/x9b_switch.py --out results/p3_explore/x9 --procs 4
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x9_freeinf import lcp_chain, lcp_np  # noqa: E402

ROOT = os.path.join(os.environ.get("TIARA_DATASETS", "/ssd8/hungwei/paper-hkv/datasets"), "freeinference_agentic_trace/release")
RUN_ID = os.environ.get("RUN_ID", "")
MAIN_CAUSES = {"system_prompt", "tool_definitions"}


def one_week(wdir):
    rq = pq.read_table(os.path.join(wdir, "processed/requests.parquet"),
                       columns=["session_id", "step", "start_ms", "end_ms", "transition", "cause", "block_ids"])
    sid = rq.column("session_id").to_pylist()
    step = rq.column("step").to_numpy()
    st = rq.column("start_ms").to_numpy()
    en = rq.column("end_ms").to_numpy()
    trans = rq.column("transition").to_pylist()
    cause = rq.column("cause").to_pylist()
    bl = rq.column("block_ids").combine_chunks()
    off = bl.offsets.to_numpy()
    val = bl.values.to_numpy()
    nb = np.diff(off)
    order = sorted(range(len(sid)), key=lambda i: (sid[i], step[i]))
    by_s = defaultdict(list)
    for i in order:
        by_s[sid[i]].append(i)
    ev = []
    chk_n = chk_ok = 0
    for s, ix in by_s.items():
        ixa = np.array(ix)
        for j in range(len(ix) - 1):
            p, c = ix[j], ix[j + 1]
            if trans[c] not in ("mutation", "switch"):
                continue
            bp, bc = val[off[p]:off[p + 1]], val[off[c]:off[c + 1]]
            L = lcp_chain(bp, bc)
            if chk_n < 500:
                chk_n += 1
                chk_ok += lcp_np(bp, bc) == L
            if L >= nb[p]:
                continue
            later = ixa[j + 2:]
            back = -1
            if len(later):
                m = nb[later] > L
                cand = later[m]
                if len(cand):
                    hit = val[off[cand] + L] == bp[L]
                    if hit.any():
                        back = int(cand[int(np.argmax(hit))])
            ev.append(dict(session=s, kind=trans[c], cause=cause[c] or "", dead_tok=int(16 * (nb[p] - L)),
                           back=back >= 0,
                           gap_s=((st[back] - en[p]) / 1000.0) if back >= 0 else float("nan"),
                           steps=int(step[back] - step[p]) if back >= 0 else -1))
    return dict(week=os.path.basename(wdir), ev=ev, chk_n=chk_n, chk_ok=chk_ok)


def summ(evs, label, now):
    if not evs:
        return dict(run_id=RUN_ID, ts=now, subset=label, n_events=0)
    back = np.array([e["back"] for e in evs])
    dead = np.array([e["dead_tok"] for e in evs], dtype=float)
    gaps = np.array([e["gap_s"] for e in evs if e["back"]])
    steps = np.array([e["steps"] for e in evs if e["back"]])
    return dict(run_id=RUN_ID, ts=now, subset=label, n_events=len(evs),
                n_sessions_with_event=len({e["session"] for e in evs}),
                frac_events_switch_back=float(back.mean()),
                bytew_frac_switch_back=float(dead[back].sum() / dead.sum()),
                n_switch_back=int(back.sum()),
                gap_median_s=float(np.median(gaps)) if len(gaps) else float("nan"),
                gap_p90_s=float(np.quantile(gaps, 0.9)) if len(gaps) else float("nan"),
                frac_back_within_5min=float(np.mean(gaps <= 300)) if len(gaps) else float("nan"),
                steps_median=float(np.median(steps)) if len(steps) else float("nan"),
                dead_tok_sum=float(dead.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    assert RUN_ID
    weeks = sorted(glob.glob(os.path.join(ROOT, "*_*")))
    assert len(weeks) == 16
    with ProcessPoolExecutor(a.procs) as ex:
        R = list(ex.map(one_week, weeks))
    cn = sum(r["chk_n"] for r in R)
    co = sum(r["chk_ok"] for r in R)
    print("chain check", co, "/", cn, flush=True)
    assert cn == 0 or co / cn >= 0.99, f"鏈式前提：{co}/{cn}"
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    E = [e for r in R for e in r["ev"]]
    for e in E:
        e["session"] = e["session"]  # session_id 只在同一週內唯一也沒關係：session 不跨週
    rows = [summ([e for e in E if e["kind"] == "mutation" and e["cause"] in MAIN_CAUSES], "MAIN mutation:system_prompt+tool_definitions", now),
            summ([e for e in E if e["kind"] == "mutation"], "all mutation causes", now),
            summ([e for e in E if e["kind"] == "switch"], "switch", now)]
    for c in sorted({e["cause"] for e in E if e["kind"] == "mutation"}):
        rows.append(summ([e for e in E if e["kind"] == "mutation" and e["cause"] == c], f"mutation:{c}", now))
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "x9b_t7_switch_back.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
