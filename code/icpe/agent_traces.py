"""agent_traces.py — 長上下文 agent trace 的特性（只報告；ICPE 新故事的動機數字，不是判定）。

  python code/icpe/agent_traces.py characterize [--out results/icpe_h200/agent]

每條 trace 一列（trace_stats.csv）：請求數、session 數、輸入長度分布、≥32K／≥100K 的比例、中位長度的 KV、
同一 session 相鄰兩個請求的間隔、每 5 分鐘活躍 session 的 KV（只有絕對時間是真的、多使用者的 trace 才算）、
以及「回來時要還原的 KV（上一輪留下的前綴）÷ 間隔」——間隔比還原時間長的比例（＝有提示就能完全藏住的上限），
分 CPU 層（H200 T1 常數 43.59 GiB/s）、SSD 級（7.0）、NFS 級（0.33）三種讀取速度〔算術，單一 session 獨佔頻寬〕。

KV 位元組＝Llama-3.1-8B BF16：32 層 × 8 KV head × 128 × (K,V) × 2 B＝128 KiB／token（算術；和 X1 的 64 MiB／512-token 塊一致）。
規則 6（CLAUDE.md §1）：每個載入函式都用 A 欄驗算 B 欄，比例寫進輸出；比例 < 0.99 → 停。
下載的資料是不受信任的資料：只讀 JSON／parquet，不執行裡面的東西；用 python -I 跑。
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

DS = os.environ.get("TIARA_DATASETS", "/work/ChihChungHsu/hungweilai/datasets")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KIB_PER_TOKEN = 32 * 8 * 128 * 2 * 2 / 1024
assert KIB_PER_TOKEN == 128.0 and 512 * KIB_PER_TOKEN * 1024 == 64 * 2**20
BW = {"cpu": 43.5888, "ssd": 7.0, "nfs": 0.33}          # GiB/s：T1 的 CPU 層讀；SSD／NFS 級＝ICPE 的裝置等級
WIN_MS = 300_000


def _iso_ms(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000


def _check(name, ok, n, thr=0.99):
    r = ok / n if n else float("nan")
    print(f"[{name}] 規則 6：{ok}/{n} = {r:.4f}", flush=True)
    if not (r >= thr):
        raise SystemExit(f"[{name}] 規則 6 的驗算比例 {r:.4f} < {thr} → 停")
    return r


# 每個 loader 回傳 DataFrame：sid、start_ms、end_ms（沒有就 NaN）、inp（輸入 token）、prefix（可從同 session 上一份 KV 還原的 token；不知道就 NaN）、abs_time（bool）
def load_freeinference():
    """長度用 Σ role_tokens（o200k_base，block_ids 就是用它切的；和 x1_fi 一樣）；provider 回報的 prompt_tokens 另外報。"""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    fs = sorted(glob.glob(f"{DS}/freeinference_agentic_trace/release/*/processed/requests.parquet"))
    assert len(fs) == 16, len(fs)
    parts, ok, n = [], 0, 0
    for f in fs:              # block_ids 很大：分批讀，只留長度（規則 6 同 x1_fi：len(block_ids)==floor(Σrole_tokens/16)，門檻 0.95）
        for b in pq.ParquetFile(f).iter_batches(batch_size=4000, columns=["session_id", "step", "start_ms", "end_ms", "prompt_tokens",
                                                                          "role_tokens", "block_ids"]):
            nb = pc.list_value_length(b.column("block_ids")).to_numpy(zero_copy_only=False)
            x = b.drop_columns(["block_ids"]).to_pandas()
            rt = np.array([sum(v for v in r.values() if v is not None) for r in x.role_tokens], np.int64)
            ok += int((nb == rt // 16).sum())
            n += len(rt)
            parts.append(x.drop(columns=["role_tokens"]).assign(o200k=rt))
    rule = _check("freeinference len(block_ids)=floor(Σrole_tokens/16)", ok, n, thr=0.95)
    d = pd.concat(parts, ignore_index=True).sort_values(["session_id", "step"])
    prev_in = d.groupby("session_id").o200k.shift()
    pt = d.prompt_tokens.astype(float)
    return pd.DataFrame(dict(sid=d.session_id, start_ms=d.start_ms, end_ms=d.end_ms, inp=d.o200k,
                             prefix=np.minimum(prev_in, d.o200k))), dict(
        rule6=rule, abs_time=True, multi_user=True, length_unit="o200k Σrole_tokens",
        provider_prompt_tokens_median=float(pt.median()), provider_prompt_tokens_frac_ge100k=float((pt >= 100000).mean()))


def load_tracelab():
    rows, ok, n = [], 0, 0
    with gzip.open(f"{DS}/tracelab/syfi_coding_trace.jsonl.gz", "rt") as fh:
        for ln in fh:
            r = json.loads(ln)
            ev = [e["timestamp"] for e in r.get("timing_events") or [] if e.get("timestamp")]
            if not ev:
                continue
            n += 1
            ok += int(r["prefix_tokens"] + r["newly_append_tokens"] == r["input_tokens_total"])
            t = sorted(_iso_ms(x) for x in ev)
            rows.append((r["session_id"], r["round_index"], t[0], t[-1], r["input_tokens_total"], r["prefix_tokens"], r.get("user")))
    rule = _check("tracelab prefix+new=input", ok, n)
    d = pd.DataFrame(rows, columns=["sid", "round", "start_ms", "end_ms", "inp", "prefix", "user"]).sort_values(["sid", "round"])
    return d, dict(rule6=rule, abs_time=True, multi_user=True, n_users=int(d.user.nunique()))


def load_semianalysis():
    """主 agent 的請求一個 session；每個 subagent（巢狀的 requests）各自一個 session（它有自己的 context）。
    規則 6：上游 README 寫 `in`＝64-token hash 塊數 × 64 → 驗 len(hash_ids)×64＝in。"""
    rows, ok, n = [], 0, 0

    def add(sid, reqs, bs):
        nonlocal ok, n
        prev = set()
        for i, q in enumerate(reqs):
            n += 1
            h = q["hash_ids"]
            ok += int(len(h) * bs == q["in"])
            k = 0
            while k < len(h) and h[k] in prev:          # 前綴：連續命中同 session 上一個請求的塊數（hash_id_scope＝local）
                k += 1
            prev = set(h)
            t0 = q["t"] * 1000
            rows.append((sid, i, t0, t0 + q.get("api_time", np.nan) * 1000, q["in"], min(k * bs, q["in"]), q.get("type")))

    with open(f"{DS}/semianalysis_cc_traces/traces.jsonl") as fh:
        for ln in fh:
            r = json.loads(ln)
            bs = r["block_size"]
            main = [q for q in r["requests"] if q.get("type") != "subagent"]
            add(r["id"], main, bs)
            for q in r["requests"]:
                if q.get("type") == "subagent":
                    add(f"{r['id']}:{q['agent_id']}", q["requests"], bs)
    rule = _check("semianalysis len(hash_ids)×64=in", ok, n)
    d = pd.DataFrame(rows, columns=["sid", "i", "start_ms", "end_ms", "inp", "prefix", "type"])
    return d, dict(rule6=rule, abs_time=False, multi_user=False)


def load_exgentic():
    import pyarrow.parquet as pq
    rows, ok, n = [], 0, 0
    for f in sorted(glob.glob(f"{DS}/exgentic_agent_llm_traces/data/*.parquet")):
        for r in pq.read_table(f, columns=["session_id", "total_tokens", "spans"]).to_pylist():
            tot = 0
            for s in r["spans"] or []:
                a = s.get("attributes") or {}
                ti, to = a.get("gen_ai.usage.input_tokens"), a.get("gen_ai.usage.output_tokens")
                if ti is None or not s.get("start_time"):
                    continue
                tot += ti + (to or 0)
                rows.append((r["session_id"], _iso_ms(s["start_time"]), _iso_ms(s["end_time"]) if s.get("end_time") else np.nan, ti))
            n += 1
            ok += int(tot == r["total_tokens"])
    rule = _check("exgentic Σ(span input+output)=total_tokens", ok, n)
    d = pd.DataFrame(rows, columns=["sid", "start_ms", "end_ms", "inp"]).sort_values(["sid", "start_ms"])
    prev_in = d.groupby("sid").inp.shift()
    d["prefix"] = np.minimum(prev_in, d.inp)                         # 沒有 hash：上限＝上一個請求的輸入（〔假設〕前綴完全重用）
    return d, dict(rule6=rule, abs_time=False, multi_user=False)


def load_mooncake(name):
    rows, ok, n = [], 0, 0
    for ln in open(f"{DS}/traces/{name}_trace.jsonl"):
        r = json.loads(ln)
        n += 1
        ok += int(len(r["hash_ids"]) == math.ceil(r["input_length"] / 512))
        rows.append((r["timestamp"], r["input_length"]))
    rule = _check(f"mooncake-{name} len(hash_ids)=ceil(input/512)", ok, n)
    d = pd.DataFrame(rows, columns=["start_ms", "inp"]).assign(sid=np.nan, end_ms=np.nan, prefix=np.nan)
    return d, dict(rule6=rule, abs_time=True, multi_user=True, no_sessions=True)


def stats(name, d, meta):
    p = d.inp.astype(float)
    out = dict(trace=name, n_req=len(d), n_sess=int(d.sid.nunique()) if d.sid.notna().any() else None,
               inp_median=float(p.median()), inp_p90=float(p.quantile(.9)), inp_p99=float(p.quantile(.99)), inp_max=float(p.max()),
               frac_ge32k=float((p >= 32768).mean()), frac_ge100k=float((p >= 100000).mean()),
               kv_gib_at_median=float(p.median() * KIB_PER_TOKEN / 2**20), **{k: v for k, v in meta.items()})
    if d.sid.notna().any():
        d = d.sort_values(["sid", "start_ms"])
        prev_end = d.groupby("sid").end_ms.shift()
        prev_start = d.groupby("sid").start_ms.shift()
        gap = (d.start_ms - prev_end.where(prev_end.notna(), prev_start)) / 1000
        m = gap.notna() & d.prefix.notna()
        g, pre = gap[m].clip(lower=0), d.prefix[m].astype(float)
        out.update(n_returns=int(m.sum()), gap_s_median=float(g.median()), gap_s_p90=float(g.quantile(.9)), gap_s_p99=float(g.quantile(.99)),
                   prefix_median=float(pre.median()), kv_return_gib_median=float(pre.median() * KIB_PER_TOKEN / 2**20))
        for tier, bw in BW.items():
            t_restore = pre * KIB_PER_TOKEN / 2**20 / bw
            out[f"restore_s_median_{tier}"] = float(t_restore.median())
            out[f"frac_gap_ge_restore_{tier}"] = float((g >= t_restore).mean())
    if meta.get("abs_time") and meta.get("multi_user") and d.sid.notna().any():
        d = d.assign(win=(d.start_ms // WIN_MS))
        w = d.groupby(["win", "sid"]).inp.max().reset_index().groupby("win").agg(n=("sid", "size"), tok=("inp", "sum"))
        gib = w.tok * KIB_PER_TOKEN / 2**20
        out.update(active5m_median=float(w.n.median()), active5m_p90=float(w.n.quantile(.9)), active5m_max=int(w.n.max()),
                   active5m_kv_gib_median=float(gib.median()), active5m_kv_gib_p90=float(gib.quantile(.9)), active5m_kv_gib_max=float(gib.max()))
    return out


def cmd_characterize(a):
    os.makedirs(a.out, exist_ok=True)
    rid = os.environ.get("RUN_ID", datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc")
    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    res = []
    for name, fn in [("freeinference", load_freeinference), ("tracelab", load_tracelab), ("semianalysis_cc", load_semianalysis),
                     ("exgentic", load_exgentic), ("mooncake_conversation", lambda: load_mooncake("conversation")),
                     ("mooncake_toolagent", lambda: load_mooncake("toolagent"))]:
        d, meta = fn()
        s = stats(name, d, meta)
        res.append(dict(run_id=rid, ts=ts, **s))
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}, ensure_ascii=False), flush=True)
    pd.DataFrame(res).to_csv(os.path.join(a.out, "trace_stats.csv"), index=False)
    print(f"→ {os.path.join(a.out, 'trace_stats.csv')}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("characterize"); s.add_argument("--out", default=os.path.join(REPO, "results", "icpe_h200", "agent"))
    a = ap.parse_args()
    sys.exit({"characterize": cmd_characterize}[a.cmd](a))


if __name__ == "__main__":
    main()
