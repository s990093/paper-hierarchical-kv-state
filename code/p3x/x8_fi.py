"""x8_fi.py — X8 W5：FreeInference agentic trace → x8 的 session 請求表（登記之後才加，見 X8 §2.5）

資料：/ssd8/hungwei/paper-hkv/datasets/freeinference_agentic_trace/release/<week>/processed/{requests,tool_calls}.parquet
     （HF harvardMadsys/freeinference_agentic_trace @ 16d3c04d，CC-BY-4.0；下載與 sha 見該目錄 README.txt）
欄位同 x8_traces.py 的 npz，另加 week（0..15）。
  到達＝start_ms、閒置開始＝end_ms；前綴＝同 session 前一個請求 block_ids 的共同前綴 × 16 token；
  ctx＝Σrole_tokens（o200k_base token，直接當 Llama token，不換算）；obj＝ctx＋completion_tokens；
  ret_kind：step_trigger＝tool → 1、其他（user）→ 2、session 第一個請求 → 0。
  ttl_key＝這個請求產生的工具名組合（排序後用 | 連），ttl_dur_s＝這些工具的 latency_ms 最大值。

規則 6 斷言（每週、全體都報）：
  (a) len(block_ids) == floor(Σrole_tokens/16)（README：最後不滿一塊丟掉）≥ 0.95
  (b) transition＝append 時，共同前綴＝前一個請求的全部 block ≥ 0.95
  (c) 工具觸發的請求：start_ms − 前一個 end_ms 與前一個請求工具的 max(latency_ms) 相差 ≤ max(1 s, 5%) ≥ 0.90
讀下載資料只用 pyarrow／numpy，不執行資料裡的任何程式；用 python -I 跑。
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.join(os.environ.get("TIARA_DATASETS", "/ssd8/hungwei/paper-hkv/datasets"), "freeinference_agentic_trace/release")
B16 = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", required=True)
    a = ap.parse_args()
    weeks = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
    assert len(weeks) == 16, weeks
    cols = dict(sid=[], t_arr=[], t_idle=[], ctx_tok=[], nold_tok=[], obj_tok=[], ret_kind=[], ttl_key=[], ttl_dur_s=[],
                sess_start=[], shared_tok=[], week=[])
    meta = collections.Counter()
    sid_map = {}
    t0 = time.time()
    for wi, w in enumerate(weeks):
        base = os.path.join(ROOT, w, "processed")
        tc = pq.read_table(os.path.join(base, "tool_calls.parquet"), columns=["session_id", "step", "name", "latency_ms"])
        tools = collections.defaultdict(list)
        for s, st, nm, lat in zip(tc.column("session_id").to_pylist(), tc.column("step").to_pylist(),
                                  tc.column("name").to_pylist(), tc.column("latency_ms").to_pylist()):
            tools[(s, st)].append((nm, lat))
        pf = pq.ParquetFile(os.path.join(base, "requests.parquet"))
        prev_sid, prev_blk, prev_end, prev_tools, sess_t0 = None, None, None, None, None
        wk = collections.Counter()
        for b in pf.iter_batches(batch_size=2048, columns=["session_id", "step", "start_ms", "end_ms", "step_trigger",
                                                            "completion_tokens", "role_tokens", "transition", "block_ids"]):
            sids = b.column("session_id").to_pylist()
            steps = b.column("step").to_pylist()
            st_ms = b.column("start_ms").to_numpy(zero_copy_only=False)
            en_ms = b.column("end_ms").to_numpy(zero_copy_only=False)
            trig = b.column("step_trigger").to_pylist()
            comp = b.column("completion_tokens").to_pylist()
            rt = b.column("role_tokens")
            rsum = np.zeros(len(sids), dtype=np.int64)
            for fi in range(rt.type.num_fields):
                v = rt.field(fi).to_numpy(zero_copy_only=False)
                rsum += np.nan_to_num(v.astype(float)).astype(np.int64)
            trans = b.column("transition").to_pylist()
            bl = b.column("block_ids")
            off = bl.offsets.to_numpy()
            val = bl.values.to_numpy()
            for k in range(len(sids)):
                blk = val[off[k]:off[k + 1]]
                s = sids[k]
                wk["n"] += 1
                wk["unit_ok"] += len(blk) == rsum[k] // B16
                gs = sid_map.setdefault((w, s), len(sid_map))
                kind, nold = 0, 0
                if s == prev_sid:
                    m = min(len(prev_blk), len(blk))
                    eq = prev_blk[:m] == blk[:m]
                    c = m if eq.all() else int(np.argmin(eq))
                    nold = c * B16
                    kind = 1 if trig[k] == "tool" else 2
                    if trans[k] == "append":
                        wk["append_n"] += 1
                        wk["append_ok"] += c == len(prev_blk)
                    if trig[k] == "tool" and prev_tools:
                        lat = [x for _, x in prev_tools if x is not None]
                        if lat:
                            wk["lat_n"] += 1
                            gap = st_ms[k] - prev_end
                            wk["lat_ok"] += abs(gap - max(lat)) <= max(1000, 0.05 * max(lat))
                    if st_ms[k] < prev_end:
                        wk["overlap"] += 1
                else:
                    sess_t0 = st_ms[k] / 1000.0
                tl = tools.get((s, steps[k]), [])
                names = sorted({nm for nm, _ in tl if nm is not None})
                lats = [x for _, x in tl if x is not None]
                ctx = int(rsum[k])
                cols["sid"].append(gs)
                cols["t_arr"].append(st_ms[k] / 1000.0)
                cols["t_idle"].append(max(en_ms[k], st_ms[k]) / 1000.0)
                cols["ctx_tok"].append(ctx)
                cols["nold_tok"].append(min(nold, ctx))
                cols["obj_tok"].append(ctx + int(comp[k] or 0))
                cols["ret_kind"].append(kind)
                cols["ttl_key"].append("|".join(names))
                cols["ttl_dur_s"].append(max(lats) / 1000.0 if lats else float("nan"))
                cols["sess_start"].append(sess_t0)
                cols["shared_tok"].append(0)
                cols["week"].append(wi)
                prev_sid, prev_blk, prev_end, prev_tools = s, blk, en_ms[k], tl
        r = dict(week=w, n_req=wk["n"], unit_rate=wk["unit_ok"] / wk["n"],
                 append_prefix_rate=wk["append_ok"] / max(wk["append_n"], 1),
                 tool_latency_gap_rate=wk["lat_ok"] / max(wk["lat_n"], 1), overlap=wk["overlap"])
        print(json.dumps(r), f"{time.time() - t0:.0f}s", flush=True)
        for kk, vv in wk.items():
            meta[kk] += vv
        assert r["unit_rate"] >= 0.95, r
        assert r["append_prefix_rate"] >= 0.95, r
        assert r["tool_latency_gap_rate"] >= 0.90, r
    out = {k: np.array(v, dtype=object if k == "ttl_key" else None) for k, v in cols.items()}
    o = np.lexsort((np.arange(len(out["t_arr"])), out["t_arr"]))
    out = {k: v[o] for k, v in out.items()}
    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, "freeinference.npz"), **out)
    ret = out["ret_kind"] > 0
    summ = dict(workload="freeinference", n_req=len(o), n_sessions=len(sid_map), n_returns=int(ret.sum()),
                n_tool_returns=int((out["ret_kind"] == 1).sum()), n_user_returns=int((out["ret_kind"] == 2).sum()),
                n_sessions_with_return=int(len(np.unique(out["sid"][ret]))),
                n_eval_sessions_with_return=int(len(np.unique(out["sid"][ret & (out["week"] >= 8)]))),
                assert_unit_rate=meta["unit_ok"] / meta["n"], assert_append_prefix_rate=meta["append_ok"] / meta["append_n"],
                assert_tool_latency_gap_rate=meta["lat_ok"] / meta["lat_n"], frac_overlap_prev=meta["overlap"] / meta["n"],
                ctx_tok_median=float(np.median(out["ctx_tok"])), ctx_tok_p90=float(np.percentile(out["ctx_tok"], 90)),
                frac_ctx_gt_131072=float(np.mean(out["ctx_tok"] > 131072)),
                frac_nold_over_ctx_returns=float(out["nold_tok"][ret].sum() / out["ctx_tok"][ret].sum()))
    print(json.dumps(summ), flush=True)
    rid = os.environ.get("RUN_ID", "adhoc")
    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    new = not os.path.exists(a.csv)
    with open(a.csv, "a", newline="") as fh:
        wr = csv.writer(fh)
        if new:
            wr.writerow(["run_id", "ts", "workload", "metric", "value"])
        for k, v in summ.items():
            if k != "workload":
                wr.writerow([rid, ts, "freeinference", k, v])


if __name__ == "__main__":
    main()
