#!/usr/bin/env python3
"""m16_i3g_posthoc.py — I3g 的事後分析（不是事先寫死的判定；判定以 m16_i3g_report.py 為準）。只讀 requests_*.csv／i3g_report.json，只用 CPU。

  1. 誤差範圍：以 session 為單位的配對 bootstrap（64 個 session 有放回抽樣、四個設定用同一組抽樣，因為工作負載一樣），
     算「回來那一輪」TTFT P95 的差（x − b）與比（x ／ b）的 95% 區間。每個設定只跑一次，跑與跑之間的變異不在這個區間裡。
  2. d（NFS）慢在哪：vLLM 0.28 的 vllm:kv_offload_lookup_async_delay_seconds＝請求第一次被延後（下層查找還沒回來 RETRY，或找到了但正從下層搬回 CPU 的 HIT_PENDING）
     到不再延後的時間（kv_connector/v1/offloading/scheduler.py:564-575、:609-631、:975-987；tiering/manager.py:336-416）——**包含 promotion 的等待**，不只是存在檢查，拿它的平均和 TTFT 平均比；另報載入／寫出的位元組與時間。

  python3 -I code/m16_i3g_posthoc.py --in results/m16_stage3_mi300x/i3g_agentic
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from datetime import datetime, timezone

KEYS = ("a", "b", "c", "d")
B = 2000


def pct(v, q):
    s = sorted(v)
    k = (len(s) - 1) * q
    f = int(k)
    return s[f] + (s[min(f + 1, len(s) - 1)] - s[f]) * (k - f)


def ci(v):
    return [pct(v, 0.025), pct(v, 0.975)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/m16_stage3_mi300x/i3g_agentic")
    ap.add_argument("--seed", type=int, default=20261022)
    a = ap.parse_args()
    rep = json.load(open(os.path.join(a.inp, "i3g_report.json")))
    cfg = rep["configs"]
    by = {}
    for k in KEYS:
        rows = list(csv.DictReader(open(os.path.join(a.inp, f"requests_{cfg[k]['run_id']}.csv"))))
        s = {}
        for r in rows:
            if r["turn"] in ("", "0") or r["is_warmup"] not in ("0", "False", "") or r["ok"] not in ("1", "True") or not r["ttft_s"]:
                continue
            s.setdefault(r["session"], []).append(float(r["ttft_s"]))
        by[k] = s
    sess = sorted(set().union(*[set(by[k]) for k in KEYS]), key=int)
    assert all(set(by[k]) == set(sess) for k in KEYS), "四個設定回來那一輪的 session 不一樣"
    n = {k: sum(len(v) for v in by[k].values()) for k in KEYS}
    assert all(n[k] == cfg[k]["n_return"] for k in KEYS), (n, {k: cfg[k]["n_return"] for k in KEYS})
    p95 = {k: pct([x for v in by[k].values() for x in v], 0.95) for k in KEYS}
    rng = random.Random(a.seed)
    bs = {k: [] for k in KEYS}
    for _ in range(B):
        pick = [rng.choice(sess) for _ in sess]
        for k in KEYS:
            bs[k].append(pct([x for s in pick for x in by[k][s]], 0.95))
    pairs = {}
    for k in ("a", "c", "d"):
        diff = [x - y for x, y in zip(bs[k], bs["b"])]
        ratio = [x / y for x, y in zip(bs[k], bs["b"])]
        pairs[f"{k}_vs_b"] = dict(p95_diff=p95[k] - p95["b"], diff_ci=ci(diff), p95_ratio=p95[k] / p95["b"], ratio_ci=ci(ratio))
    per = {k: dict(p95=p95[k], p95_ci=ci(bs[k])) for k in KEYS}
    io = {}
    for k in KEYS:
        s = (cfg[k].get("metrics") or {}).get("series") or {}
        g = lambda name: s.get(name, {}).get("delta")  # noqa: E731
        num = lambda x: isinstance(x, (int, float))  # noqa: E731
        lk, lc = g("offload_lookup_async_delay_sum_s"), g("offload_lookup_async_delay_count")
        lb, lt, sb, st = g("offload_load_bytes"), g("offload_load_time_s"), g("offload_store_bytes"), g("offload_store_time_s")
        rows = list(csv.DictReader(open(os.path.join(a.inp, f"requests_{cfg[k]['run_id']}.csv"))))
        tt = [float(r["ttft_s"]) for r in rows if r["is_warmup"] in ("0", "False", "") and r["ok"] in ("1", "True") and r["ttft_s"]]
        io[k] = dict(ttft_mean_all=sum(tt) / len(tt), n_all=len(tt),
                     lookup_async_delay_mean=((lk / lc) if lc else 0.0) if num(lk) and num(lc) else "NOT_AVAILABLE",
                     lookup_async_delay_sum=lk if num(lk) else "NOT_AVAILABLE",
                     load_gb=lb / 1e9 if num(lb) else "NOT_AVAILABLE", load_gbps=(lb / lt / 1e9) if num(lb) and num(lt) and lt else "NOT_AVAILABLE",
                     store_gb=sb / 1e9 if num(sb) else "NOT_AVAILABLE", store_gbps=(sb / st / 1e9) if num(sb) and num(st) and st else "NOT_AVAILABLE",
                     tier_read_gb=(g("tiering_read_bytes") / 1e9) if num(g("tiering_read_bytes")) else "NOT_AVAILABLE",
                     tier_write_gb=(g("tiering_write_bytes") / 1e9) if num(g("tiering_write_bytes")) else "NOT_AVAILABLE",
                     prompt_cached_frac=(g("prompt_tokens_cached") / g("prompt_tokens")) if num(g("prompt_tokens_cached")) and g("prompt_tokens") else "NOT_AVAILABLE")
    out = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now(timezone.utc).isoformat(timespec="seconds"), posthoc=True,
               source_report=rep["run_id"], source_runs={k: cfg[k]["run_id"] for k in KEYS}, n_sessions=len(sess), n_return=n, B=B, seed=a.seed,
               per_config=per, vs_b=pairs, io=io)
    json.dump(out, open(os.path.join(a.inp, "i3g_posthoc.json"), "w"), indent=1, ensure_ascii=False)
    f = lambda x, d=2: x if isinstance(x, str) else f"{x:.{d}f}"  # noqa: E731
    L = [f"# I3g 事後分析（不是事先寫死的判定；run {out['run_id']}，讀 {rep['run_id']}）", "",
         f"誤差範圍＝以 session 為單位的配對 bootstrap（{len(sess)} 個 session、B＝{B}、seed {a.seed}）；每個設定只跑一次，跑與跑之間的變異不在區間裡。", "",
         "| 設定 | 回來那一輪 P95（秒） | 95% 區間 | 對 b 的差 | 差的 95% 區間 | 對 b 的比 | 比的 95% 區間 |", "|:--|--:|:--|--:|:--|--:|:--|"]
    for k in KEYS:
        r = per[k]
        q = pairs.get(f"{k}_vs_b")
        L.append(f"| {k} | {f(r['p95'], 3)} | [{f(r['p95_ci'][0])}, {f(r['p95_ci'][1])}] | " +
                 (f"{q['p95_diff']:+.2f} | [{q['diff_ci'][0]:+.2f}, {q['diff_ci'][1]:+.2f}] | {q['p95_ratio']:.2f} | [{q['ratio_ci'][0]:.2f}, {q['ratio_ci'][1]:.2f}] |"
                  if q else "（基準） | | | |"))
    L += ["", "| 設定 | TTFT 平均（全部請求） | 查下層＋搬回 CPU 的等待平均（秒／請求） | CPU→GPU 載入 GB | CPU→GPU 載入 GB/s | 寫出 GB | 寫出 GB/s | 檔案層讀 GB | 檔案層寫 GB | prompt 命中比例 |",
          "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for k in KEYS:
        r = io[k]
        L.append(f"| {k} | {f(r['ttft_mean_all'])} | {f(r['lookup_async_delay_mean'])} | {f(r['load_gb'], 0)} | {f(r['load_gbps'])} | {f(r['store_gb'], 0)} | "
                 f"{f(r['store_gbps'])} | {f(r['tier_read_gb'], 0)} | {f(r['tier_write_gb'], 0)} | {f(r['prompt_cached_frac'])} |")
    L += ["", "查下層＋搬回 CPU 的等待＝vllm:kv_offload_lookup_async_delay_seconds 的 sum／count（請求因下層查找或 promotion 被延後的時間）（count＝有等過的請求數；count＝0 記 0＝從沒等過）。a 沒有卸載，這些欄是 NOT_AVAILABLE。"]
    open(os.path.join(a.inp, "i3g_posthoc.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
