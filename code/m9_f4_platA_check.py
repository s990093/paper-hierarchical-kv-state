#!/usr/bin/env python3
"""F4 稽核（fork A）：main.tex 平台 A 數字 vs git 52803bc^ 的原始 CSV。

只讀：git 歷史（`git show 52803bc^:<path>`）與本機 Mooncake trace。
不碰 GPU、不改任何既有檔。
產出：
  results/audit_20261010/fragments/platA_evidence.json  （每一項重算的證據值）
  results/audit_20261010/fragments/claims_A.csv         （逐條主張的判定）
用法：python code/m9_f4_platA_check.py
"""
from __future__ import annotations

import collections
import csv
import io
import json
import os
import random
import statistics as st
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REV = "52803bc^"
OUT = REPO / "results/audit_20261010/fragments"
TRACES = Path("/mlsteam/data/tiara/datasets/traces")
RUN_ID = os.environ.get("RUN_ID", "NO_RUN_ID")
TS = datetime.now(timezone.utc).isoformat(timespec="seconds")


def gshow(path: str, rev: str = REV) -> str:
    return subprocess.run(["git", "-C", str(REPO), "show", f"{rev}:{path}"],
                          check=True, capture_output=True, text=True).stdout


def gcsv(path: str, rev: str = REV) -> list[dict]:
    return list(csv.DictReader(io.StringIO(gshow(path, rev))))


def gjson(path: str, rev: str = REV) -> dict:
    return json.loads(gshow(path, rev))


EV: dict[str, object] = {}


def ev(key: str, val):
    EV[key] = val
    return val


def lin(pts):
    n = len(pts)
    sx = sum(a for a, _ in pts); sy = sum(b for _, b in pts)
    sxx = sum(a * a for a, _ in pts); sxy = sum(a * b for a, b in pts)
    sl = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    b0 = (sy - sl * sx) / n
    mean = sy / n
    ss = sum((b - mean) ** 2 for _, b in pts)
    r2 = 1 - sum((b - (b0 + sl * a)) ** 2 for a, b in pts) / ss
    return b0, sl, r2


def pct(v, q):
    v = sorted(v)
    return v[int(q * (len(v) - 1))]


# ---------------------------------------------------------------- M1 容量
cap = gcsv("results/m1_capacity/capacity.csv")
capm = {(x["run_id"], x["config"]): int(x["kv_cache_tokens"]) for x in cap
        if x["phase"] == "measure" and x["verdict"] == "OK"}
ev("m1_capacity_measure", {f"{k[1]}|{k[0]}": v for k, v in capm.items()})
ev("m1_llama_bf16_kv_gib", [x["kv_gib"] for x in cap if x["config"] == "llama-bf16" and x["phase"] == "measure"])
ev("m1_verify_at_verdicts", {x["config"]: x["verdict"] for x in cap if x["phase"] == "verify_at"})
ev("ratio_547744_over_41648", round(547744 / 41648, 3))
ev("ratio_547744_over_48128", round(547744 / 48128, 3))
ev("ratio_399376_over_ref", {"/120320": round(399376 / 120320, 2), "/41648": round(399376 / 41648, 2)})

# capacity_by_dtype：KiB/token 正規化
cbd = gcsv("results/m2_harness/capacity_by_dtype.csv")
kib = collections.defaultdict(list)
for x in cbd:
    kib[x["kv_dtype_name"]].append(float(x["kv_cache_gib"]) * 1024 * 1024 / int(x["kv_cache_tokens"]))
ev("kib_per_tok", {k: [round(min(v), 3), round(st.median(v), 3), round(max(v), 3)] for k, v in kib.items()})
ev("kib_range_pct", {k: round(100 * (max(v) - min(v)) / st.median(v), 3) for k, v in kib.items()})
raw = [int(x["kv_cache_tokens"]) for x in cbd if x["kv_dtype_name"] == "int8"]
ev("raw_token_range_pct_int8", round(100 * (max(raw) - min(raw)) / max(raw), 2))
ev("cbd_host_contention", dict(collections.Counter(x["host_contention"] for x in cbd)))
b = st.median(kib["bf16"])
ev("vs_bf16", {k: round(b / st.median(v), 3) for k, v in kib.items()})
ev("scale_bytes_per_scale", {k: round((st.median(kib[k]) - ideal) * 1024 / 512, 2)
                             for k, ideal in (("int8", 64), ("int4", 32))})

# ---------------------------------------------------------------- M3 baselines
bl = gcsv("results/m3_baseline/baseline_longctx.csv")
g = collections.defaultdict(list)
for x in bl:
    g[(x["model_key"], int(x["ctx"]), x["baseline"], x["round"])].append(x)
ev("m3_longctx_contamination", {"|".join(k): v for k, v in collections.Counter((x["host_contention"], x["contaminated"], x["concurrency_mode"]) for x in bl).items()})
pd = {}
for (mk, ctx, bb, rnd), v in g.items():
    if bb == "full_gpu" and rnd == "cold":
        p0 = [x for x in v if x["prefix_idx"] == "0"][0]
        pf = float(p0["ttft_ms"]) / float(p0["actual_prompt_tokens"])
        pd[f"{mk}|{ctx}"] = {"prefix0_prefill_ms_per_tok": round(pf, 4), "prefix0_tpot": float(p0["tpot_ms"]),
                             "prefix0_ratio": round(float(p0["tpot_ms"]) / pf, 1),
                             "median_prefill": round(st.median(float(x["ttft_ms"]) / float(x["actual_prompt_tokens"]) for x in v), 4),
                             "median_tpot": round(st.median(float(x["tpot_ms"]) for x in v), 2),
                             "run_id": p0["run_id"]}
ev("prefill_decode_table", pd)
tab = {}
for ctx in (32768, 65536, 131072, 258048):
    for bb in ("full_gpu", "cpu_lru", "cpu_arc", "tier_fs", "lmcache"):
        c = g.get(("qwen-awq", ctx, bb, "cold")); w = g.get(("qwen-awq", ctx, bb, "warm"))
        if not c:
            continue
        cm = st.median(float(x["ttft_ms"]) for x in c); wm = st.median(float(x["ttft_ms"]) for x in w)
        tab[f"{ctx}|{bb}"] = {"cold_med": round(cm), "warm_med": round(wm), "speedup": round(cm / wm, 2), "run_id": c[0]["run_id"]}
ev("baseline_qwen_awq_medians", tab)
ev("tier_fs_258k_saving_s", round((tab["258048|tier_fs"]["cold_med"] - tab["258048|tier_fs"]["warm_med"]) / 1000, 1))
sp = [v["speedup"] for k, v in tab.items() if k.split("|")[0] in ("32768", "65536")]
ev("speedup_32k_65k_range", [min(sp), max(sp)])
ev("cpu_tier_tokens_24GiB_at_56KiB", round(24 * 1024 * 1024 / 56))
# 「prompt:生成 1092:1、99.8% 時間在 prefill」
b1 = gcsv("results/m3_baseline/baseline.csv")
shares = [float(x["ttft_ms"]) / float(x["total_ms"]) for x in b1 + bl
          if x["ttft_ms"] and x["total_ms"] and float(x["total_ms"]) > 0]
ev("ttft_share_of_total_all_m3_rows", {"median": round(100 * st.median(shares), 2), "min": round(100 * min(shares), 2), "max": round(100 * max(shares), 2)})
r16 = [x for x in bl if x["ctx"] == "16384"]
ev("ctx16384_gen_tokens", sorted(set(x["gen_tokens"] for x in r16)))
ev("ctx16384_ttft_share", round(100 * st.median(float(x["ttft_ms"]) / float(x["total_ms"]) for x in r16), 2))

# decode 擬合（照 52803bc^:code/m4_oracle.py load_decode_model 的演算法重算）
def decode_fit(mk):
    pts = []
    for rows in (b1, bl):
        rr = [x for x in rows if x.get("model_key") == mk and x.get("baseline") in ("", "full_gpu")
              and x["ttft_ms"] and x["total_ms"] and str(x.get("gen_tokens", "")).isdigit() and int(x["gen_tokens"]) > 0]
        gg = collections.defaultdict(list)
        for x in rr:
            gg[int(x.get("actual_prompt_tokens") or x["ctx"])].append(x)
        for c, v in gg.items():
            t = st.median(float(x["ttft_ms"]) for x in v); tot = st.median(float(x["total_ms"]) for x in v)
            n = st.median(int(x["gen_tokens"]) for x in v)
            if tot > t and n:
                pts.append((c // 16, (tot - t) / n))
    b0, sl, r2 = lin(pts)
    return {"base_ms": round(b0, 3), "ms_per_block": round(sl, 6), "r2": round(r2, 4), "n_points": len(pts)}


ev("decode_fit_llama_bf16", decode_fit("llama"))
ev("decode_fit_qwen_awq", decode_fit("qwen-awq"))
ev("decode_gbps_paper_slopes", {"llama-bf16_128KiB": round(16 * 128 * 1024 / 0.005581e-3 / 1e9, 1),
                                "qwen-awq_56KiB": round(16 * 56 * 1024 / 0.001267e-3 / 1e9, 1),
                                "qwen-awq_if_fp8_28KiB": round(16 * 28 * 1024 / 0.001267e-3 / 1e9, 1)})

# ---------------------------------------------------------------- Mooncake trace（本機）
tr = {}
for t in ("toolagent", "conversation"):
    p = TRACES / f"{t}_trace.jsonl"
    if not p.exists():
        tr[t] = "NOT_FOUND"
        continue
    r = [json.loads(l) for l in p.open()]
    il = [x["input_length"] for x in r]; ol = [x["output_length"] for x in r]
    tr[t] = {"requests": len(r), "input_median": st.median(il), "input_max": max(il),
             "output_median": st.median(ol), "output_p90": pct(ol, 0.9), "output_max": max(ol),
             "duration_s": (r[-1]["timestamp"] - r[0]["timestamp"]) / 1000}
ev("mooncake_local_trace", tr)

# ---------------------------------------------------------------- M4 oracle
def sweep(path):
    out = {}
    for x in gcsv(path):
        if x["ssd_gib"] not in ("512", "512.0"):
            continue
        d = float(x["decode_ms"]); p = float(x["prefill_ms"])
        out[f"{x['trace']}|{x['policy']}"] = {
            "sim_version": x["sim_version"], "device": x["device"], "best": x["best_baseline"],
            "headroom": x["oracle_headroom_pct"], "write_mibps": x["ssd_write_mibps"],
            "decode_share_pct": round(100 * d / (d + p), 1), "prefill_ms": p,
            "unique_blocks": x["unique_blocks"], "trace_duration_s": x["trace_duration_s"],
            "has_run_id": "run_id" in x}
    return out


SW = {k: sweep(f"results/m4_oracle/{k}ssd_sweep.csv") for k in ("", "qwen-awq/", "llama-awq/", "qwen-awq-oraclefix/")}
ev("ssd_sweep_512", SW)
wr = {}
for prof, key, kvb in (("llama-bf16", "", 128), ("qwen-awq", "qwen-awq/", 56), ("qwen-awq-fix", "qwen-awq-oraclefix/", 56)):
    for t in ("toolagent", "conversation"):
        tf = float(SW[key][f"{t}|tier_fs"]["write_mibps"]); orc = float(SW[key][f"{t}|oracle"]["write_mibps"])
        ub = int(SW[key][f"{t}|tier_fs"]["unique_blocks"]); dur = float(SW[key][f"{t}|tier_fs"]["trace_duration_s"])
        wr[f"{prof}|{t}"] = {"tier_fs": tf, "oracle": orc, "ratio": round(tf / orc, 2),
                             "lower_bound_mibps": round(ub * 16 * kvb * 1024 / 2**20 / dur)}
ev("write_bw", wr)
# prefill-only headroom 由 ssd_sweep 的 prefill_ms 推（qwen-awq，舊 oracle）
pf = {}
for t in ("toolagent", "conversation"):
    s = SW["qwen-awq/"]
    best = s[f"{t}|oracle"]["best"]
    bp = s[f"{t}|{best}"]["prefill_ms"]; op = s[f"{t}|oracle"]["prefill_ms"]
    pf[t] = {"prefill_only_headroom": round(100 * (bp - op) / bp, 2), "e2e": float(s[f"{t}|oracle"]["headroom"]),
             "ratio": round(float(s[f"{t}|oracle"]["headroom"]) / (100 * (bp - op) / bp), 3)}
ev("qwen_awq_mooncake_e2e_over_prefill", pf)


def surface(path):
    return {x["trace"]: float(x["oracle_headroom_pct"]) for x in gcsv(path) if x["policy"] == "oracle"}


iso = {}
for L in (32768, 65536, 131072, 262144, 524288):
    rr = [x for x in gcsv(f"results/m4_oracle/isopressure/L{L}/headroom_surface.csv") if x["policy"] == "oracle"][0]
    iso[L] = {"headroom": float(rr["oracle_headroom_pct"]), "verdict": rr["verdict"], "pressure": rr["pressure_x"],
              "reuse": rr["reuse_pct"], "sim_version": rr["sim_version"]}
ev("isopressure_qwen_awq", iso)
sp_ = surface("results/m4_oracle/qwen-awq-surface-prefill/headroom_surface.csv")
se_ = surface("results/m4_oracle/qwen-awq-surface-e2e/headroom_surface.csv")
rat = [se_[k] / sp_[k] for k in sp_ if sp_[k] > 0]
ev("surface_e2e_over_prefill", {"mean": round(st.mean(rat), 3), "median": round(st.median(rat), 3),
                                "sum_ratio": round(sum(se_.values()) / sum(sp_.values()), 3)})
sel = {k: v for k, v in se_.items() if any(s in k for s in ("L65536", "L131072", "L262144"))}
ev("surface_e2e_65k_262k", {"min": min(sel.values()), "max": max(sel.values()),
                            "rq5": {k: v for k, v in sel.items() if k.endswith("rq5")}})
hw = gcsv("results/m4_oracle/hw_sweep.csv")
ev("hw_sweep", {f"{x['compute_multiplier']}|{x['request_tokens']}": float(x["oracle_headroom_pct"]) for x in hw})
ev("hw_sweep_meta", {"run_id": hw[0]["run_id"], "sim_version": hw[0]["sim_version"]})
ev("m4_csv_sim_versions_without_run_id", "見 fork A 交接：除 hw_sweep 外 M4 CSV 皆無 run_id 欄；sim_version 0e7b989d／b0d39dd4 皆早於 oracle 修正 fbea1a98")

cmq = gjson("results/m4_oracle/qwen-awq/cost_model.json")
cml = gjson("results/m4_oracle/cost_model.json")
d = cmq["derived_ms_per_block"]
ev("cost_qwen_awq", {**d, "crossover_tokens": cmq["crossover_tokens"], "measured": {k: v for k, v in cmq["measured"].items() if k != "positions"}})
ev("cost_top_level_json_source", cml["measured"]["retrieval_csv"])


def costs(path, gpu_ms=None):
    r = gcsv(path)
    med = lambda tier, rnd: st.median(float(x["ttft_ms"]) for x in r if x["tier"] == tier and x["round"] == rnd)
    gw = med("gpu_resident", "warm"); ctx = int(r[0]["ctx"]); nb = ctx / 16
    return {"gpu_warm": gw, "cpu_warm": med("cpu", "warm"), "ssd_warm": med("ssd", "warm"), "drop_warm": med("drop", "warm"),
            "cpu_ms_blk": round((med("cpu", "warm") - gw) / nb, 4), "ssd_ms_blk": round((med("ssd", "warm") - gw) / nb, 4),
            "run_id": r[0]["run_id"], "host": {"|".join(k): v for k, v in collections.Counter((x["host_contention"], x["foreign_gpu_count"]) for x in r).items()},
            "has_contaminated_col": "contaminated" in r[0], "has_own_gpu_intruders_col": "own_gpu_intruders" in r[0]}


C = {"sata": costs("results/m2_harness/retrieval_cost_sata.csv"), "nvme": costs("results/m2_harness/retrieval_cost_nvme.csv")}
ev("cost_llama_bf16", C)
rp = gcsv("results/m2_harness/recompute_position.csv")
gg = collections.defaultdict(list)
for x in rp:
    gg[int(x["cached_prefix_tokens"])].append(float(x["ttft_ms"]))
pts = [(p, st.median(v)) for p, v in sorted(gg.items())]
base = pts[0][1] * 16 / 2048; slope = (pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0]) * 16 / 2048
ev("recompute_llama_bf16", {"base_ms_blk": round(base, 4), "alpha": round(slope, 7), "n_rows": len(rp), "positions": len(pts),
                            "twopoint_maxdev_pct": round(max(abs(y - (pts[0][1] + (pts[-1][1] - pts[0][1]) / pts[-1][0] * p)) / y for p, y in pts) * 100, 2),
                            "run_id": rp[0]["run_id"]})
ps = {k: round((C[k]["ssd_ms_blk"] - base) / slope) for k in C}
ps["qwen-awq"] = round((d["ssd"] - d["recompute_base"]) / d["recompute_slope_per_token"])
ev("P_star", ps)
ev("P_star_ratios", {"profile": round(ps["qwen-awq"] / ps["nvme"], 3), "device": round(ps["nvme"] / ps["sata"], 3),
                     "share_128K": {k: round(100 * v / 131072, 1) for k, v in ps.items()},
                     "mooncake_over_Pstar": round(6352 / ps["qwen-awq"], 3), "3.5Pstar": round(3.5 * ps["qwen-awq"])})
rq = gcsv("results/m2_harness/recompute_position_qwen-awq.csv")
gq = collections.defaultdict(list)
for x in rq:
    gq[int(x["cached_prefix_tokens"])].append(float(x["ttft_ms"]))
pq = [(p, st.median(v)) for p, v in sorted(gq.items())]
ev("recompute_qwen_awq", {"n_rows": len(rq), "positions": len(pq),
                          "twopoint_maxdev_pct": round(max(abs(y - (pq[0][1] + (pq[-1][1] - pq[0][1]) / pq[-1][0] * p)) / y for p, y in pq) * 100, 2),
                          "ls_maxdev_pct": round(max(abs(y - (lin(pq)[0] + lin(pq)[1] * p)) / y for p, y in pq) * 100, 2),
                          "run_ids": sorted(set(x["run_id"] for x in rq))})
# MI300X 實測 P*（只為核對 §sweet-spot 的外推，資料來自 52803bc^ 的 m4_oracle_mi300x）
mi = {}
for m in ("b-llama8b", "b-qwen7b-1m"):
    dd = gjson(f"results/m4_oracle_mi300x/cost_model_{m}.json")["derived_ms_per_block"]
    mi[m] = round((dd["ssd"] - dd["recompute_base"]) / dd["recompute_slope_per_token"])
ev("mi300x_measured_P_star", mi)
ev("mi300x_extrapolation", {"ratio": round(1307.4 / 71.2, 2), "P_hi": round(ps["qwen-awq"] * 1307.4 / 71.2),
                            "P_lo": round(ps["qwen-awq"] * 1307.4 / 71.2 / 2)})

# 磁碟頻寬
dbw = gcsv("results/m2_harness/disk_bw.csv"); dbs = gcsv("results/m2_harness/disk_bw_sustained.csv")
ev("disk_bw", {dev: {"read_med": st.median(float(x["read_mibps"]) for x in dbw if x["device"] == dev),
                     "write_med": st.median(float(x["write_mibps"]) for x in dbw if x["device"] == dev),
                     "host": sorted(set(x["host_contention"] for x in dbw if x["device"] == dev))}
               for dev in sorted(set(x["device"] for x in dbw))})
ev("disk_sustained", {"writes": [float(x["write_mibps"]) for x in dbs], "mean": round(st.mean(float(x["write_mibps"]) for x in dbs), 1),
                      "run_id": dbs[0]["run_id"], "host": sorted(set(x["host_contention"] for x in dbs))})
mib = 16384 * 128.11 / 1024
ev("eff_bw", {"bytes_MiB": round(mib, 1), "sata": round(mib / (C["sata"]["ssd_warm"] / 1000), 1),
              "nvme": round(mib / (C["nvme"]["ssd_warm"] / 1000), 1)})

# 模擬器驗證與缺口
ev("sim_validation", [{k: r.get(k) for k in ("ctx", "measured_full_warm_ms", "measured_lru_warm_ms", "measured_ratio",
                                              "sim_full_warm_ms_per_req", "sim_lru_warm_ms_per_req", "sim_ratio")}
                      for r in gjson("results/m4_oracle/simulator_validation.json")["rows"]])
gp = gcsv("results/m4_oracle/prefix_gap_probe.csv")
ev("prefix_gap", {"resident_pct": [min(float(x["post_gap_resident_pct"]) for x in gp), max(float(x["post_gap_resident_pct"]) for x in gp)],
                  "first_ever_pct": [min(float(x["post_gap_first_ever_pct"]) for x in gp), max(float(x["post_gap_first_ever_pct"]) for x in gp)]})

# ---------------------------------------------------------------- M5 品質
def acc(path):
    r = gcsv(path)
    out = {}
    for c in ("bf16", "fp8", "int8", "int4", "full_gpu", "cpu_lru", "tier_fs"):
        v = [x for x in r if x["config"] == c]
        if v:
            out[c] = round(100 * st.mean(x["correct"] == "True" for x in v), 2)
    out["model_key"] = sorted(set(x["model_key"] for x in r)); out["n_per_config"] = len([x for x in r if x["config"] == r[0]["config"]])
    out["run_ids"] = sorted(set(x["run_id"] for x in r)); out["kv_tokens"] = sorted(set(x["gpu_kv_cache_tokens"] for x in r))[:4]
    return out


Q = {p: acc(f"results/m5_quality/{p}.csv") for p in ("gsm8k_precision_n1000", "gsm8k_lossless", "needle_pilot_32k", "needle_fair_32k")}
ns = gcsv("results/m5_quality/needle_ctx_sweep.csv")
Q["needle_ctx_sweep"] = {x: {c: round(100 * st.mean(r["correct"] == "True" for r in ns if r["run_id"] == x and r["config"] == c), 1)
                             for c in ("bf16", "fp8", "int8", "int4")} for x in sorted(set(r["run_id"] for r in ns))}
Q["needle_ctx_sweep_prompt_tokens"] = {x: sorted(set(r["prompt_tokens"] for r in ns if r["run_id"] == x)) for x in sorted(set(r["run_id"] for r in ns))}
ll = gcsv("results/m5_quality/gsm8k_lossless.csv")
base_sha = {x["idx"]: x["out_sha1"] for x in ll if x["config"] == "full_gpu"}
Q["lossless_identical"] = {c: sum(base_sha[x["idx"]] == x["out_sha1"] for x in ll if x["config"] == c) for c in ("cpu_lru", "tier_fs")}
ev("quality", Q)
gsm = gcsv("results/m5_quality/gsm8k_precision_n1000.csv")
p = Q["gsm8k_precision_n1000"]["bf16"] / 100
ev("gsm8k_ci_unpaired_pp", round(196 * (2 * p * (1 - p) / 1000) ** 0.5, 2))
ev("gsm8k_power_n_for_2.8pp", {"50%power": round(1.96 ** 2 * 2 * p * (1 - p) / 0.028 ** 2),
                                "80%power": round((1.96 + 0.8416) ** 2 * 2 * p * (1 - p) / 0.028 ** 2)})


def suite(path, seed=0, B=10000):
    r = gcsv(path)
    tasks = list(dict.fromkeys(x["task"] for x in r))
    gg = collections.defaultdict(dict)
    for x in r:
        gg[(x["config"], x["task"])][x["idx"]] = float(x["score"])
    res = {"tasks": {}, "macro": {}, "paired": {}, "run_ids": {c: sorted(set(x["run_id"] for x in r if x["config"] == c)) for c in ("bf16", "fp8", "int8", "int4")},
           "n": {t: len(gg[("bf16", t)]) for t in tasks}, "level": dict(collections.Counter(x["level"] for x in r)),
           "model_key": sorted(set(x["model_key"] for x in r))}
    for c in ("bf16", "fp8", "int8", "int4"):
        pt = {t: round(100 * st.mean(gg[(c, t)].values()), 1) for t in tasks}
        res["tasks"][c] = pt
        res["macro"][c] = round(st.mean(100 * st.mean(gg[(c, t)].values()) for t in tasks), 2)
    for c in ("fp8", "int8", "int4"):
        dif = [100 * (gg[("bf16", t)][i] - gg[(c, t)][i]) for t in tasks for i in gg[("bf16", t)]]
        rnd = random.Random(seed)
        bs = sorted(st.mean(rnd.choice(dif) for _ in dif) for _ in range(B))
        res["paired"][c] = [round(st.mean(dif), 2), round(bs[int(0.025 * B)], 1), round(bs[int(0.975 * B) - 1], 1)]
    res["per_task_int8"] = {}
    for t in tasks:
        dif = [100 * (gg[("bf16", t)][i] - gg[("int8", t)][i]) for i in gg[("bf16", t)]]
        rnd = random.Random(seed)
        bs = sorted(st.mean(rnd.choice(dif) for _ in dif) for _ in range(B))
        res["per_task_int8"][t] = [round(st.mean(dif), 1), round(bs[int(0.025 * B)], 1), round(bs[int(0.975 * B) - 1], 1)]
    return res


ev("longbench", suite("results/m5_quality/longbench_precision.csv"))
ev("ruler", suite("results/m5_quality/ruler_precision.csv"))
ev("ga102", {"tflops": round(328 * 1695e6 * 128 / 1e12, 2), "flop_per_byte": round(328 * 1695e6 * 128 / 31.5e9)})
ev("pcie_arith", {"kv_bytes_GiB": round(126976 * 64 * 1024 / 2**30, 2), "ms_at_15.75GBps_GB": round(7.75e9 / 15.75e9 * 1000, 1),
                  "ms_at_15.75GBps_GiB": round(126976 * 65536 / 15.75e9 * 1000, 1),
                  "per_tok_ms": round(492.1 / 126976, 5), "hbm_ms_936": round(7.75e9 / 936e9 * 1000, 2),
                  "rel": [round(x / 8.3, 1) for x in (24.6, 98.4, 492.1)]})

# ---------------------------------------------------------------- 補充：serial vs parallel、未配對 CI 寬度
bs_rows = gcsv("results/m3_baseline/baseline.csv")
gsp = collections.defaultdict(list)
for x in bs_rows:
    gsp[(x["model_key"], x["baseline"], x["ctx"], x["round"], x["concurrency_mode"])].append(float(x["ttft_ms"]))
cells = []
for k in gsp:
    if k[4] == "serial" and (k[:4] + ("parallel",)) in gsp:
        sv = st.mean(gsp[k]); pv = st.mean(gsp[k[:4] + ("parallel",)])
        cells.append((k[:4], 100 * (sv - pv) / pv))
fg = [d for k, d in cells if k[1] == "full_gpu"]
fg_wo = [d for k, d in cells if k[1] == "full_gpu" and not (k[0] == "qwen" and k[2] == "32768" and k[3] == "warm")]
ev("serial_vs_parallel", {"cells": len(cells), "full_gpu_range": [round(min(fg), 1), round(max(fg), 1)],
                          "full_gpu_range_wo_qwen32k_warm": [round(min(fg_wo), 1), round(max(fg_wo), 1)],
                          "offload_cold_range": [round(min(d for k, d in cells if k[1] != "full_gpu" and k[3] == "cold"), 1),
                                                 round(max(d for k, d in cells if k[1] != "full_gpu" and k[3] == "cold"), 1)],
                          "offload_warm_range": [round(min(d for k, d in cells if k[1] != "full_gpu" and k[3] == "warm"), 1),
                                                 round(max(d for k, d in cells if k[1] != "full_gpu" and k[3] == "warm"), 1)],
                          "host": dict(collections.Counter(x["host_contention"] for x in bs_rows))})
lbr = gcsv("results/m5_quality/longbench_precision.csv")
a = [100 * float(x["score"]) for x in lbr if x["config"] == "bf16"]
b8 = [100 * float(x["score"]) for x in lbr if x["config"] == "int8"]
rnd = random.Random(0)
ub = sorted(st.mean(rnd.choice(a) for _ in a) - st.mean(rnd.choice(b8) for _ in b8) for _ in range(10000))
pl = EV["longbench"]["paired"]["int8"]
ev("longbench_int8_unpaired_vs_paired_width", {"unpaired": [round(ub[250], 1), round(ub[9749], 1)],
                                                "width_ratio": round((ub[9749] - ub[250]) / (pl[2] - pl[1]), 2)})

OUT.mkdir(parents=True, exist_ok=True)
# ---------------------------------------------------------------- 逐條主張（判定由 fork A 人工寫入；證據值見上方 EV）
G = "52803bc^:results/"
H = "HEAVY（整機 6 張外來 GPU 在用）"
CLAIMS = [
 # (section, line, claim_text, paper_value, kind, source_path, source_run_id, evidence_value, status, tag, note)
 ("觀察三", "330", "Qwen3-30B-A3B 在 3090 的 κ 為 27，dense 為 48–108", "27; 48–108", "推算",
  "git ea0165e:main.tex（舊算術版 tab:kappa）", "", "舊表以 100% MFU 算術估：8B 54×、70B 190×；現行實測 3090 dense κ_cpu 8.9–9.5",
  "不一致", "〔git ea0165e:main.tex:262〕〔檔案 main.tex:296-313〕", "7c76dba 換成實測表後此句未改；Qwen3-30B-A3B 從未在平台 A 量過（舊稿 L821：MoE 放不下）"),
 ("觀察三", "334", "萬一被需要時 6–190 倍的代價", "6–190×", "推算",
  "git ea0165e:main.tex", "", "舊算術 κ 6/21/54/190；現行實測 κ 全距 0.7–11.7", "不一致", "〔git ea0165e:main.tex:266〕",
  "舊估計殘留，與 tab:kappa 實測矛盾"),
 ("觀察三 fig:ladder", "359-361", "動作階梯圖：被需要時 CPU 1.0、SSD 9、DROP 12.3 µs/token", "1.0; 9; 12.3", "推算",
  "git 384165c:main.tex（舊圖『12.3 µs @100% MFU』）", "",
  "實測（tab:kappa）MI300X Llama-8B 16K：重算 46.7、CPU 30.5、SSD 64.9；3090：325.5、36.8、346.0 µs/token",
  "來源有疑慮", "〔git 384165c:main.tex〕〔算術〕", "100% MFU 算術值，標了 µs/token 單位、圖說只寫『示意』；與實測差一到兩個數量級"),
 ("觀察四 表", "403-406", "prefill/decode：16K 0.316/17.0/54×；65K 0.591/23.5/40×；127K 0.903/29.7/33×；258K 1.264/29.4/23×", "見文", "實測",
  G + "m3_baseline/baseline_longctx.csv（full_gpu cold）", "20260830-234043-m3-llama-awq-full_gpu; 20260831-012520-m3-qwen-awq-full_gpu",
  "只取 prefix_idx=0 單筆：0.3159/17.03/53.9；0.5910/23.47/39.7；0.9029/29.72/32.9；1.2637/29.36/23.2。4 前綴中位數 16K 為 0.335/16.87/50×",
  "來源有疑慮", "〔git 52803bc^:results/m3_baseline/baseline_longctx.csv〕", "n=1（第一個前綴）未說明；前三列 llama-awq、末列 qwen-awq；整機 " + H + "，contaminated=False"),
 ("觀察四", "411", "該平台 PCIe 為 Gen3 ×16（15.75 GB/s）", "Gen3; 15.75 GB/s", "外部規格",
  "main.tex:1076、tab:ratio", "", "同文 L1076 寫 PCIe 4.0×16、tab:ratio 寫 31.5 GB/s；repo 無 lspci 記錄", "不一致",
  "〔檔案 main.tex:1076〕〔檔案 main.tex:271〕", "全文自相矛盾；實際鏈路世代〔未查證〕"),
 ("觀察四", "411", "KV 7.75 GiB；移至主機 492 ms；HBM 讀 8.3 ms", "7.75 GiB; 492 ms; 8.3 ms", "算術",
  "", "", "126,976×64 KiB=7.75 GiB；7.75e9/15.75e9=492 ms（把 GiB 當 GB）；用 GiB 為 528 ms；7.75e9/936e9=8.3 ms",
  "不一致", "〔算術〕", "單位混用約 7%；且建立在 Gen3 假設上（見上一列）"),
 ("觀察四", "411", "prefill 攤提後每 token 0.0024 ms", "0.0024 ms", "算術", "", "", "492.1/126,976=0.0039 ms", "不一致", "〔算術〕", "算錯"),
 ("觀察四 表", "419-421", "卸載 5/20/100%：+24.6/+98.4/+492.1 ms；3.0/11.9/59.3×", "見文", "算術", "", "",
  "0.05/0.2/1×492.1；÷8.3 得 3.0/11.9/59.3（倍數是『增加量÷全駐留』）", "來源有疑慮", "〔算術〕", "算術自洽，但依賴 Gen3 15.75 GB/s 與 GiB/GB 混用"),
 ("觀察四 表", "440", "生成為主時 CPU/SSD 慢 3–59×", "3–59×", "算術", "", "", "同上表 3.0–59.3", "來源有疑慮", "〔算術〕", "同上"),
 ("觀察四", "446", "prompt:生成=1092:1，99.8% 時間在 prefill", "1092:1; 99.8%", "實測",
  G + "m3_baseline/baseline_longctx.csv", "20260830-234043-m3-llama-awq-full_gpu",
  "16,384/15=1092 ✓；同列 cold TTFT/總時間=95.6%；全部 longctx cold 列 92.6–99.7%", "不一致", "〔git 52803bc^:results/m3_baseline/baseline_longctx.csv〕",
  "99.8% 找不到對應列；比較像 token 比例而非時間比例"),
 ("雙平台設計", "451-453", "FLOP/byte 2,254 vs 20,752，相差 9.2 倍", "9.2×", "算術", "tab:ratio", "", "20,752/2,254=9.21", "一致", "〔算術〕",
  "2,254 本身見附錄 L1730 那列（71.16e12/31.5e9=2,259）"),
 ("雙平台設計", "455", "RTX 3090 可推至 8B 模型的約 262K 上下文", "262K", "設定",
  G + "m1_capacity/capacity.csv", "20260830-232943-m1-qwen-awq", "262,144 是 Qwen2.5-7B-1M（noDCA）的定址上限；Llama-3.1-8B AWQ 上限 131,072",
  "不一致", "〔git 52803bc^:results/RUNLOG.md:1025〕", "模型寫錯（7B Qwen 不是 8B）"),
 ("雙平台設計", "457", "平台 A 64K 與 128K 之間有容量懸崖", "64K–128K", "實測",
  G + "m1_capacity/capacity.csv", "20260830-232507-m1-llama-awq", "llama-awq KV 120,320 token（<128K）", "一致",
  "〔git 52803bc^:results/m1_capacity/capacity.csv〕", "記憶體量，非時間"),
 ("設定", "1076", "RTX 3090 24 GB、PCIe 4.0×16、sm_86", "24 GB; 4.0×16; sm_86", "外部規格",
  G + "env.json", "", "env.json：compute capability 8.6、23.684 GiB；PCIe 世代無記錄", "來源有疑慮", "〔git 52803bc^:results/env.json〕",
  "PCIe 與 L411 的 Gen3 互相矛盾"),
 ("設定", "1082", "Mooncake toolagent 23,608、conversation 12,031 請求；中位 6,352、最長 126,208", "見文", "實測",
  "/mlsteam/data/tiara/datasets/traces/*.jsonl；" + G + "RUNLOG.md:1439", "",
  "請求數 ✓；block 解碼後中位 toolagent 6,352／conversation 6,912（input_length 6,346／6,909）；最長 126,208（input_length 126,195）",
  "一致", "〔檔案 /mlsteam/data/tiara/datasets/traces/toolagent_trace.jsonl〕〔git 52803bc^:results/RUNLOG.md:1439〕", "中位 6,352 只屬 toolagent，文中當兩條 trace 共用"),
 ("設定", "1084", "重算需 225 µs 的 block vs 自 CPU 取回 4.2 µs", "225 µs; 4.2 µs", "推算",
  "git ea0165e:main.tex（舊算術 tab:kappa，3090／Llama-8B）", "", "實測 3090 Llama-8B 16K：重算 325.5、CPU 36.8 µs/token", "不一致",
  "〔git ea0165e:main.tex:931〕", "舊算術殘留"),
 ("measured fig:crossover", "1143", "qwen-awq NVMe 的 P*=37,717 token", "37,717", "實測",
  G + "m4_oracle/qwen-awq/cost_model.json", "20260901-024805-m2-retrieval; 20260831-223228-m2-recompute; 20260901-131533-m2-recompute",
  "(10.2447−3.5459)/0.00017760=37,717", "來源有疑慮", "〔git 52803bc^:results/m4_oracle/qwen-awq/cost_model.json〕",
  "時間常數全在 " + H + " 下量；CSV 沒有 contaminated／own_gpu_intruders 欄；Drop 斜率是兩端點連線不是擬合"),
 ("measured", "1147", "重算線性：48 個量測點、擬合偏差 <1.6%", "48; <1.6%", "實測",
  G + "m2_harness/recompute_position{,_qwen-awq}.csv", "20260830-195552-m2-recompute; 20260831-223228-m2-recompute",
  "48 列是 qwen-awq（11 位置），兩端點直線最大偏差 7.9%、最小平方 10.7%；1.6% 是 llama-bf16（15 列／5 位置，1.56%）", "不一致",
  "〔git 52803bc^:results/m2_harness/recompute_position_qwen-awq.csv〕", "把兩個剖面的數字拼在一起"),
 ("measured", "1147", "P* 隨成本剖面變動 3.5×", "3.5×", "算術", "tab:costmodels", "", "37,717/10,851=3.48", "來源有疑慮", "〔算術〕",
  "算術對；輸入常數為 HEAVY 下的時間量測"),
 ("measured", "1147", "128K 下落在 Drop 側的 block 8.3%（llama-bf16）至 28.8%（qwen-awq）", "8.3%; 28.8%", "算術", "tab:costmodels", "",
  "10,851/131,072=8.3%；37,717/131,072=28.8%", "來源有疑慮", "〔算術〕", "同上"),
 ("baselines 表說", "1159", "ctx 32K–65K 時五者相近（42–77×）", "42–77×", "實測",
  G + "m3_baseline/baseline_longctx.csv", "20260831-012520/022821/033010/124823/132605-m3-qwen-awq-*",
  "qwen-awq 4 前綴中位數 41.7–76.7×", "來源有疑慮", "〔git 52803bc^:results/m3_baseline/baseline_longctx.csv〕", "時間量測，" + H + "；contaminated=False"),
 ("tab:baseline-measured", "1164-1168", "131K：full_gpu 99,681/100,152；cpu_arc 102,702/72,883 1.4×；tier_fs 101,518/13,626 7.5×；lmcache 102,248/102,603", "見表", "實測",
  G + "m3_baseline/baseline_longctx.csv", "20260831-012520-m3-qwen-awq-full_gpu; 20260831-033010-m3-qwen-awq-cpu_arc; 20260831-124823-m3-qwen-awq-tier_fs; 20260831-132605-m3-qwen-awq-lmcache",
  "4 前綴中位數完全相符（tier_fs 7.45×）", "來源有疑慮", "〔git 52803bc^:results/m3_baseline/baseline_longctx.csv〕",
  "HEAVY；表題說五個系統但漏 cpu_lru（101,597/80,159，1.27×）；warm 經 PCIe，最受整機爭用影響"),
 ("tab:baseline-measured", "1170-1176", "258K：full_gpu 325,520/322,579；cpu_arc 327,295/325,767；tier_fs 326,393/43,664 7.5×；lmcache 333,046/335,034", "見表", "實測",
  G + "m3_baseline/baseline_longctx.csv", "同上", "中位數完全相符（tier_fs 7.48×）", "來源有疑慮", "〔git 52803bc^:results/m3_baseline/baseline_longctx.csv〕",
  "同上；漏 cpu_lru（330,702/331,720）"),
 ("baselines", "1180", "258K 只有 tier_fs 有效，省下 282.7 秒", "282.7 s", "實測", G + "m3_baseline/baseline_longctx.csv", "20260831-124823-m3-qwen-awq-tier_fs",
  "326,393−43,664=282.7 s", "來源有疑慮", "〔算術〕", "HEAVY"),
 ("baselines", "1180", "24 GiB 的 CPU 階可容 449,536 token", "449,536", "算術", "", "", "24 GiB÷56 KiB=449,390；449,536 在 git 與 RUNLOG 都找不到",
  "找不到來源", "〔算術〕", "差 0.03%，可能來自 vLLM log 的 block 數〔未查證〕"),
 ("baselines", "1181", "4 個 258K 前綴的工作集 1,032,192 token", "1,032,192", "算術", "", "", "4×258,048=1,032,192", "一致", "〔算術〕〔git 52803bc^:results/RUNLOG.md:1040〕", ""),
 ("decode-share", "1190", "Mooncake output_length：toolagent 中位 30、p90 507；conversation 中位 350、p90 597", "30/507; 350/597", "實測",
  "/mlsteam/data/tiara/datasets/traces/*.jsonl", "", "本機 trace 重算完全相符", "一致", "〔檔案 /mlsteam/data/tiara/datasets/traces/conversation_trace.jsonl〕", "外部資料集"),
 ("decode-share", "1190", "decode 佔端到端 48.8%（toolagent）與 53.8%（conversation）", "48.8%; 53.8%", "實測",
  G + "m4_oracle/*ssd_sweep.csv", "", "只出現在 52803bc^:code/m4_oracle.py 的 docstring；ssd_sweep 512 GiB：qwen-awq 31–34%、llama-awq 47–51%、llama-bf16 72–75%",
  "找不到來源", "〔git 52803bc^:code/m4_oracle.py:339〕", "沒有任何 CSV 產生這兩個數；剖面也沒寫"),
 ("decode-share", "1190", "llama-bf16（BF16 權重 15.2 GB）decode 比例升至 75%", "75%; 15.2 GB", "實測",
  G + "m4_oracle/ssd_sweep.csv", "20260831-200058-m4-decode（RUNLOG:1811；CSV 無 run_id 欄）",
  "tier_fs toolagent 75.4%、conversation 73.4%", "來源有疑慮", "〔git 52803bc^:results/m4_oracle/ssd_sweep.csv〕",
  "CSV 無 run_id 欄（違反 CLAUDE.md §4.3），靠 RUNLOG 才對得上；15.2 GB 只在 RUNLOG"),
 ("decode-share", "1197", "llama-bf16 decode 擬合 36.768+0.005581N，R²=0.999", "36.768; 0.005581", "實測",
  G + "m3_baseline/baseline{,_longctx}.csv", "20260830-*-m3-llama-full_gpu", "照 52803bc^:code/m4_oracle.py 演算法重算：36.768+0.005581N，R²=0.9994（4 個 ctx 點）",
  "一致", "〔git 52803bc^:code/m4_oracle.py:334〕", "只有 4 點"),
 ("decode-share", "1197", "qwen-awq decode 擬合 18.158+0.001267N", "18.158; 0.001267", "實測",
  G + "m3_baseline/baseline_longctx.csv", "20260831-012520-m3-qwen-awq-full_gpu", "同演算法重算：4.878+0.001579N，R²=0.9975（4 點）；18.158 只在 RUNLOG:1842",
  "不一致", "〔git 52803bc^:results/RUNLOG.md:1842〕", "用現存資料無法重現"),
 ("decode-share", "1197", "γ 換算 KV 讀取頻寬 376 與 581 GB/s（<936）", "376; 581 GB/s", "算術", "", "",
  "2 MiB/5.581 µs=376 ✓；16×56 KiB/1.267 µs=724（不是 581）；581 對應的是重算斜率 0.001579", "不一致", "〔算術〕",
  "文中印的 qwen 係數算不出 581；兩者來自不同版本的擬合"),
 ("decode-share", "1199", "端到端 headroom 約為 prefill headroom 的 0.88 倍", "0.88×", "實測",
  G + "m4_oracle/qwen-awq-surface-{e2e,prefill}/", "", "表面掃描 e2e/prefill 中位 0.898、加總比 0.860；Mooncake qwen-awq 由 prefill_ms 推得 0.67–0.69",
  "找不到來源", "〔git 52803bc^:results/m4_oracle/qwen-awq/ssd_sweep.csv〕", "0.88 沒有對應的計算"),
 ("decode-share", "1200", "qwen-awq 真實 trace 端到端 headroom 8.42%／9.23%", "8.42%; 9.23%", "實測",
  G + "m4_oracle/qwen-awq/ssd_sweep.csv", "（CSV 無 run_id；sim_version b0d39dd4）", "oracle 8.415／9.226 ✓；同設定 oracle 修正後（qwen-awq-oraclefix，fbea1a98，2026-09-05）為 12.335／12.003",
  "來源有疑慮", "〔git 52803bc^:results/m4_oracle/qwen-awq-oraclefix/ssd_sweep.csv〕〔git 52803bc^:results/RUNLOG.md:2861〕",
  "RUNLOG 自己寫『oracle 修好後 8.4%→12.3%，舊 sim_version 的 M4 CSV 全部要重跑』，論文仍用舊值"),
 ("decode-share", "1200", "65K–262K 端到端 headroom 17%–29%", "17–29%", "實測",
  G + "m4_oracle/qwen-awq-surface-e2e/headroom_surface.csv", "（無 run_id；sim 0e7b989d）", "rq5 三格 22.4/28.3/17.6；全部 12 格 6.7–28.7",
  "來源有疑慮", "〔git 52803bc^:results/m4_oracle/qwen-awq-surface-e2e/headroom_surface.csv〕", "取哪幾格沒說；oracle 修正前；成本常數 13:15 更新前跑的"),
 ("eps-task 無損表", "1210-1218", "64-shot GSM8K（前綴 11,029）60 題：三者 76.67%，cpu_lru／tier_fs 60/60 相同", "76.67%; 60/60", "實測",
  G + "m5_quality/gsm8k_lossless.csv", "20260830-231251-m5-lossless", "76.67% ×3；SHA-1 相同 60/60、60/60", "一致",
  "〔git 52803bc^:results/m5_quality/gsm8k_lossless.csv〕", "模型是 Llama-3.1-8B BF16（文中沒寫）；11,029 只在平台 B stdout 出現〔未查證於平台 A〕"),
 ("tab:eps-task", "1229-1240", "GSM8K 欄 77.9/76.2/76.4/75.1（n=1,000），表題寫 qwen-awq", "77.9/76.2/76.4/75.1", "實測",
  G + "m5_quality/gsm8k_precision_n1000.csv", "20260831-145649-m5-precision", "數值 ✓，但 model_key=llama（Llama-3.1-8B BF16，KV 41,595 token），不是 qwen-awq",
  "不一致", "〔git 52803bc^:results/m5_quality/gsm8k_precision_n1000.csv〕", "同一張表混兩個模型，表題錯"),
 ("tab:eps-task", "1237-1240", "大海撈針欄 100/5/95/0（32K，5 深度×4）", "100/5/95/0", "實測",
  G + "m5_quality/needle_pilot_32k.csv", "20260831-181930-m5-needle", "qwen-awq，32,245 token，n=20：100/5/95/0 ✓（needle_fair_32k 20260831-204132 同值）", "一致",
  "〔git 52803bc^:results/m5_quality/needle_pilot_32k.csv〕", "此檔也在 main 的 52803bc^ 裡，不只在 claude/ 分支（D6 的說法不完整）"),
 ("tab:eps-task", "1237-1240", "LongBench 欄 58.4/6.6/57.0/0.4", "58.4/6.6/57.0/0.4", "實測",
  G + "m5_quality/longbench_precision.csv", "20260901-193344/193346/193348/193350-m5-longbench", "巨觀 58.35/6.57/57.02/0.36 ✓", "一致",
  "〔git 52803bc^:results/m5_quality/longbench_precision.csv〕", "own_gpu_intruders=0；分數不受整機爭用影響"),
 ("tab:eps-task", "1237-1240", "RULER 欄 91.6/3.3/79.1/0.0", "91.6/3.3/79.1/0.0", "實測",
  G + "m5_quality/ruler_precision.csv", "20260901-193410/193413/195709/193408-m5-ruler", "巨觀 91.60/3.30/79.13/0.00 ✓", "一致",
  "〔git 52803bc^:results/m5_quality/ruler_precision.csv〕", ""),
 ("tab:eps-task", "1237-1240", "容量欄 1.00/2.00/1.94/3.77×", "1.00/2.00/1.94/3.77", "實測",
  G + "m2_harness/capacity_by_dtype.csv", "20260830-194119-m2-capacity", "128.11/64.04/66.04/34.02 KiB → 1.00/2.00/1.94/3.765", "一致",
  "〔git 52803bc^:results/m2_harness/capacity_by_dtype.csv〕", "記憶體量"),
 ("tab:eps-task", "1229,1242", "全距 2.8/100/58.0/91.6；同設定換任務 ε 差 21 至 36 倍", "21–36×", "算術", "", "",
  "58.0/2.8=20.7、100/2.8=35.7 ✓", "不一致", "〔算術〕", "2.8 來自 Llama-BF16 的 GSM8K，其餘三欄是 qwen-awq：倍數跨模型"),
 ("eps-task", "1247", "GSM8K n=1,000 差值 95% CI ±3.7pp", "±3.7pp", "算術", G + "m5_quality/gsm8k_precision_n1000.csv", "20260831-145649-m5-precision",
  "未配對常態近似 ±3.64pp", "一致", "〔算術〕", "Llama-BF16，不是 qwen-awq"),
 ("eps-task", "1247", "4K/8K/16K：BF16 100；INT8 95/90/90；FP8 5/0/5；INT4 0/0/0", "見文", "實測",
  G + "m5_quality/needle_ctx_sweep.csv", "20260831-184433/185342/190401-m5-needle", "prompt 3,997/8,023/16,141；數值完全相符", "一致",
  "〔git 52803bc^:results/m5_quality/needle_ctx_sweep.csv〕", ""),
 ("eps-task", "1249", "區分 2.8pp 需每設定 n≈1,760；7 個 f 值 12,320 請求", "1,760; 12,320", "算術", G + "RUNLOG.md:1676", "",
  "雙尾 α=0.05：50% 檢定力 n≈1,687；80% 檢定力 n≈3,447；7×1,760=12,320", "來源有疑慮", "〔算術〕〔git 52803bc^:results/RUNLOG.md:1676〕",
  "1,760 約等於 50% 檢定力，未說明假設；以常用 80% 需約 3,450"),
 ("eps-task", "1252", "LongBench 配對：FP8 掉 51.8 [47.3,56.3]、INT4 58.0 [53.7,62.3]", "51.8; 58.0", "實測",
  G + "m5_quality/longbench_precision.csv", "20260901-1933xx-m5-longbench", "重算（seed 0，10,000 次）51.78 [47.2,56.3]；57.99 [53.7,62.3]", "一致",
  "〔git 52803bc^:results/m5_quality/longbench_precision.csv〕", "bootstrap 隨機性 ±0.1"),
 ("eps-task", "1255", "INT8：GSM8K −1.5、LongBench −1.3 [0.0,2.8]、撈針 −5", "−1.5; −1.3; −5", "實測",
  G + "m5_quality/", "20260831-145649-m5-precision; 20260901-193348-m5-longbench; 20260831-181930-m5-needle", "77.9−76.4=1.5；1.33 [0.0,2.9]；100−95=5", "一致",
  "〔git 52803bc^:results/m5_quality/longbench_precision.csv〕", "GSM8K 是 Llama-BF16"),
 ("eps-task", "1255", "RULER INT8 79.1 vs 91.6，配對差 12.5 [8.6,16.7]；multikey_3 53.3，差 46.7 [30.0,63.3]", "12.5; 46.7", "實測",
  G + "m5_quality/ruler_precision.csv", "20260901-195709-m5-ruler", "12.47 [8.5,16.7]；46.7 [30.0,63.3]", "一致", "〔git 52803bc^:results/m5_quality/ruler_precision.csv〕", ""),
 ("eps-task", "1258", "FWE 是 14 任務中唯一 FP8 未全失者（21.1）、唯一 INT8 未掉者（87.8 vs 85.6，−2.2 [−6.7,2.2]）", "21.1; −2.2", "實測",
  G + "m5_quality/{ruler,longbench}_precision.csv", "20260901-1934xx-m5-ruler", "數值 ✓（−2.2 [−6.7,2.2]）；但 LongBench TREC 的 FP8 是 22.0，7 個 LongBench 任務 FP8 都 >0；INT8 在 HotpotQA 54.1>53.4、GovReport 34.1>34.0 也沒掉",
  "不一致", "〔git 52803bc^:results/m5_quality/longbench_precision.csv〕", "兩個『唯一』都不成立（若限 RULER 7 任務則成立）"),
 ("eps-task", "1258", "CWE 掉 12.0 [4.7,20.3]；VT 掉 6.7 [−2.0,16.0]；NIAH 四變體掉 6.7 至 46.7", "見文", "實測",
  G + "m5_quality/ruler_precision.csv", "20260901-195709-m5-ruler", "CWE 12.0 [4.7,20.3]；VT 6.7 [−2.0,16.0]；NIAH 10.0/46.7/6.7/7.5", "一致",
  "〔git 52803bc^:results/m5_quality/ruler_precision.csv〕", ""),
 ("eps-task", "1261", "數值分析：FP8 靜態 2.64%、動態 2.56%、INT8 0.65%；理論 3.6%／0.68%；0.65→95%、2.64→5%、11.76→0%", "見文", "推算",
  "git 52803bc^:code/quant_error.py；" + G + "RUNLOG.md:1964", "（無 run_id、無輸出檔）",
  "程式 docstring：『用高斯合成資料。真實 KV 有已知的離群值現象，實際數字可能不同』", "來源有疑慮", "〔git 52803bc^:code/quant_error.py〕",
  "合成資料被寫成『數值分析／與實測吻合』，沒揭露是高斯合成；沒有 run_id"),
 ("eps-task", "1264", "fp8_per_token_head 需 Triton，compute capability 8.6 拒絕 FP8 KV", "8.6", "實測", G + "RUNLOG.md:1971", "",
  "RUNLOG 記錄 ValueError: FP8 KV cache is not supported by the Triton attention backend on compute capability 8.6", "一致",
  "〔git 52803bc^:results/RUNLOG.md:1971〕", "只有錯誤訊息，沒有 run_id"),
 ("tab:writebw", "1280-1283", "寫入頻寬 tier_fs/Oracle：4,666/1,114；4,892/1,214；2,016/283；2,122/330 MiB/s", "見表", "實測",
  G + "m4_oracle/ssd_sweep.csv；" + G + "m4_oracle/qwen-awq/ssd_sweep.csv", "20260831-200058-m4-decode（llama-bf16，RUNLOG）；qwen-awq 無 run_id",
  "全部相符；oracle 修正後（fbea1a98）qwen-awq 的 Oracle 欄變 33.9／43.5", "來源有疑慮", "〔git 52803bc^:results/m4_oracle/qwen-awq-oraclefix/ssd_sweep.csv〕",
  "Oracle 欄是修正前的值；CSV 無 run_id 欄"),
 ("tab:writebw", "1273", "兩剖面 KV 每 token 位元組差 2.3×", "2.3×", "算術", "", "", "128/56=2.29", "一致", "〔算術〕", ""),
 ("write-feasibility", "1288", "Oracle 寫入頻寬是 tier_fs 的 1/4.0 至 1/7.1（成本感知放置放寬 4–7 倍）", "1/4.0–1/7.1", "實測",
  "同上", "", "修正前 4.19/4.03/7.13/6.44 ✓；修正後 qwen-awq 為 1/59.5、1/48.8", "來源有疑慮", "〔git 52803bc^:results/m4_oracle/qwen-awq-oraclefix/ssd_sweep.csv〕",
  "摘要也用了這個比值；oracle 修正後倍數變大一個數量級，數字過期"),
 ("write-feasibility", "1290", "SATA QLC 持續寫 181 MiB/s，tier_fs 超出 26×；NVMe 短測 2,512", "181; 26×; 2,512", "實測",
  G + "m2_harness/disk_bw{,_sustained}.csv", "20260831-162634-disk-bw; 20260831-162603-disk-bw", "189.6/171.9 平均 180.8；4,666/181=25.8；NVMe 寫中位 2,511.7", "來源有疑慮",
  "〔git 52803bc^:results/m2_harness/disk_bw_sustained.csv〕", "磁碟量測時整機 HEAVY（foreign_procs 7）；持續寫 n=2"),
 ("write-feasibility", "1290", "每 block 只寫一次的下界 3,086–3,208；qwen-awq 1,350–1,404 MiB/s", "見文", "算術",
  G + "m4_oracle/ssd_sweep.csv", "", "unique_blocks×16×KV/3,537 s：3,086/3,208/1,350/1,404 ✓", "一致", "〔算術〕", ""),
 ("sweet-spot", "1306", "壓力 5×、重用 80.9%：32K 8.94%、65K–262K 24.44–33.50%、524K 12.71%", "見文", "實測",
  G + "m4_oracle/isopressure/L*/headroom_surface.csv", "（無 run_id；sim 0e7b989d，2026-09-01 12:46）", "8.936/31.749/33.501/24.442/12.712 ✓", "來源有疑慮",
  "〔git 52803bc^:results/m4_oracle/isopressure/L131072/headroom_surface.csv〕", "oracle 修正前；也早於 13:15 的成本常數更新（RUNLOG:2160）；無 run_id"),
 ("sweet-spot", "1306", "峰值落在 3.5 P* 處", "3.5P*", "實測", G + "m4_oracle/hw_sweep.csv", "20260901-173251-m4-hw-sweep",
  "isopressure 峰在 131,072=3.48P* ✓；但同設定（5×、81%）的 hw_sweep 平台 A 實測線峰在 262,144=6.95P*（18.7%），131K 只有 17.5%", "不一致",
  "〔git 52803bc^:results/m4_oracle/hw_sweep.csv〕", "兩次掃描的峰位與峰值都不同（33.5% vs 18.7%），論文兩段各引一個"),
 ("fig:sweetspot 圖說", "1311", "細線三個重用率（59–89%）：四條線峰值皆落在 2–3.5P*", "2–3.5P*", "實測",
  G + "m4_oracle/qwen-awq-surface-prefill/headroom_surface.csv", "（無 run_id；sim 0e7b989d）", "rq1.2（59%）與 rq2（68%）峰在 65,536=1.74P*（<2P*=75,434）；只有 rq5、rq10 峰在 131,072",
  "不一致", "〔git 52803bc^:notebooks/paper_figures.py:135〕", "四條有兩條不在底色區"),
 ("hwpeak", "1316,1321", "峰值自平台 A 的 256K（圖說 262K）移至 6× 算力的 512K（524K），18.7%→39.5%", "18.7%; 39.5%", "推算",
  G + "m4_oracle/hw_sweep.csv", "20260901-173251-m4-hw-sweep", "18.678@262,144；39.49@524,288 ✓", "來源有疑慮", "〔git 52803bc^:results/m4_oracle/hw_sweep.csv〕",
  "6× 線為縮放 α 的推算（已揭露）；oracle 修正前（b0d39dd4）；與 L1306 平台 A 峰在 131K 矛盾"),
 ("hwpeak", "1326", "1,307.4/71.2=18.4×；MI300X 的 P* 346K–693K；甜蜜點 692K–2.4M", "見文", "推算", G + "m4_oracle_mi300x/cost_model_b-*.json", "",
  "外推算術 ✓（37,717×18.36=692,573）；但平台 B 已實測：Llama-3.1-8B P*≈14,461、Qwen2.5-7B-1M≈8,891 token", "不一致",
  "〔git 52803bc^:results/m4_oracle_mi300x/cost_model_b-llama8b.json〕〔算術〕", "預測被平台 B 實測否定（差 24–78 倍），論文沒更新"),
 ("hwpeak", "1326", "Llama-3.1-8B 於 1M 需 122 GB KV", "122 GB", "算術", "", "", "10^6×128 KiB=131 GB=122 GiB；2^20 token=128 GiB=137 GB", "不一致",
  "〔算術〕", "GB／GiB 混用（tab:fit 同一數字由主 session 查）"),
 ("hwpeak", "1328", "Mooncake 最長 126,208，僅為峰值的 1/10 至 1/19", "1/10–1/19", "推算", "", "", "126,208/1.21M=1/9.6；/2.43M=1/19.2", "來源有疑慮",
  "〔算術〕", "依賴已被平台 B 否定的外推（上一列）"),
 ("awq-unlock", "1337", "FP8-KV：Llama 41,648→83,312、Qwen 106,512→213,040，恰好兩倍", "見文", "實測",
  G + "m1_capacity/capacity.csv", "20260830-175957-m1-llama-bf16(-kvfp8); 20260830-180531-m1-qwen-bf16(-kvfp8)", "完全相符", "一致",
  "〔git 52803bc^:results/m1_capacity/capacity.csv〕", "記憶體量"),
 ("awq-unlock 表", "1345-1350", "KiB/tok 128.11/64.04/66.04/34.02；額外 +0.11/+0.04/+2.04/+2.02；vs BF16 1.00/2.00/1.94/3.77", "見表", "實測",
  G + "m2_harness/capacity_by_dtype.csv", "20260830-194119-m2-capacity", "中位 128.109/64.044/66.041/34.024；3.765", "一致",
  "〔git 52803bc^:results/m2_harness/capacity_by_dtype.csv〕", ""),
 ("awq-unlock", "1339", "正規化後全距降至 0.04%", "0.04%", "實測", G + "m2_harness/capacity_by_dtype.csv", "20260830-194119-m2-capacity", "int8 0.041%、int4 0.024%", "一致", "〔算術〕", ""),
 ("awq-unlock", "1355", "每 token 512 個 FP32 scale=2 KiB；int8 4.1、int4 4.0 bytes/scale", "512; 2 KiB; 4.1; 4.0", "算術", "", "",
  "2×32×8=512；(66.04−64)×1024/512=4.08；(34.02−32)×1024/512=4.04", "一致", "〔算術〕", ""),
 ("awq-unlock", "1358", "BF16 權重 15 GB，KV 預算 5.9 GB，容量上限 41,648", "15; 5.9; 41,648", "實測",
  G + "m1_capacity/capacity.csv；" + G + "m2_harness/capacity_by_dtype.csv", "20260830-175957-m1-llama-bf16; 20260830-194119-m2-capacity",
  "41,648 對應 kv_gib 5.084；5.88 GiB 對應 48,128 token", "不一致", "〔git 52803bc^:results/m1_capacity/capacity.csv〕",
  "5.9 與 41,648 是 KV pool 擺動的兩個不同模式（附錄 L1656 自己也說 5.09／5.88）"),
 ("awq-unlock", "1358", "AWQ-INT4 權重佔 4.7 GB，KV 預算約 15 GB", "4.7; 15", "實測", G + "m1_capacity/capacity.csv", "20260830-232507-m1-llama-awq",
  "120,320×128 KiB=14.7 GiB ✓；4.7 GB 在 CSV（AWQ 列 kv_gib 空白）與 RUNLOG 都找不到", "找不到來源", "〔git 52803bc^:results/m1_capacity/capacity.csv〕", ""),
 ("awq-unlock 表", "1365-1371", "單卡 KV 容量 41,648／120,320／240,656／273,872／547,744／399,376", "見表", "實測",
  G + "m1_capacity/capacity.csv", "20260830-175957/232507/232744/232943/233106/224248-m1-*", "measure 階段完全相符", "一致",
  "〔git 52803bc^:results/m1_capacity/capacity.csv〕", "是 KV pool 大小；後四者的 verify_at 都是 UNEXPECTED_FAIL（超過模型定址長度），不是可用上下文"),
 ("awq-unlock", "1377", "41,648 vs 240,656／547,744，相差 13.2×", "13.2×", "算術", G + "m1_capacity/capacity.csv", "", "547,744/41,648=13.15；若用 5.88 GiB 的 48,128 則 11.4×",
  "來源有疑慮", "〔算術〕", "比的是 Llama BF16 權重＋BF16 KV 對 Qwen AWQ＋FP8 KV：同時換模型、權重、KV 精度；分母取 KV pool 的低模式"),
 ("awq-unlock", "1377", "MLA KV 30.4 KiB/token（GQA 的 1/4.2）", "30.4; 1/4.2", "算術", "", "",
  "repo 內沒有 DeepSeek-V2-Lite 的 config；以公開 config 27 層×(512+64)×2 B=30.4 KiB〔未查證〕；128/30.4=4.2", "找不到來源", "〔算術〕〔未查證〕", ""),
 ("awq-unlock", "1377", "MLA 單卡 399,376 token，把壓力軸往後推 4.2×", "4.2×", "實測", G + "m1_capacity/capacity.csv", "20260830-224248-m1-mla-awq",
  "實測容量比：399,376/120,320（同為 AWQ 的 Llama）=3.32×", "不一致", "〔算術〕", "4.2× 是每 token 位元組比，不是容量比（權重大小不同）"),
 ("tab:verdict", "1397-1400", "端到端 headroom：qwen-awq 8.42/9.23%；llama-bf16 2.83/3.34%", "見表", "實測",
  G + "m4_oracle/{,qwen-awq/}ssd_sweep.csv", "20260831-200058-m4-decode（llama-bf16）；qwen-awq 無 run_id", "2.834/3.341/8.415/9.226 ✓；qwen-awq 修正後 12.335/12.003；llama-bf16 修正後未重跑",
  "來源有疑慮", "〔git 52803bc^:results/RUNLOG.md:2861〕", "全部是 oracle 修正前（sim 0e7b989d／b0d39dd4）"),
 ("tab:verdict", "1402", "qwen-awq 65K–131K prefill headroom 31.8–33.5%", "31.8–33.5%", "實測",
  G + "m4_oracle/isopressure/", "（無 run_id）", "31.749（→31.7）與 33.501", "來源有疑慮", "〔git 52803bc^:results/m4_oracle/isopressure/L65536/headroom_surface.csv〕",
  "31.749 先寫成 31.75 再進位成 31.8（重複進位）；oracle 修正前"),
 ("verdict", "1408", "Mooncake 中位 6,352 僅為 P* 的 0.17 倍", "0.17", "算術", "", "", "6,352/37,717=0.168", "一致", "〔算術〕", ""),
 ("verdict", "1408", "AWQ 權重使 decode 固定成本減半，headroom 升至 8–9%", "減半; 8–9%", "實測", "", "",
  "18.158/36.768=0.49，但 18.158 無法重現（重算 4.878）；8–9% 為修正前", "來源有疑慮", "〔git 52803bc^:results/RUNLOG.md:1842〕", ""),
 ("verdict", "1411", "五列相差 12×", "12×", "算術", "", "", "33.5/2.83=11.8", "來源有疑慮", "〔算術〕", "輸入是修正前的 headroom"),
 ("limitations", "1418", "依 Sibyl 80% 換算：生產 trace 6.7–7.4%、甜蜜點 25–27%", "6.7–7.4%; 25–27%", "推算", "", "",
  "0.8×8.42=6.74、0.8×9.23=7.38、0.8×31.8=25.4、0.8×33.5=26.8 ✓", "來源有疑慮", "〔算術〕", "推算已揭露；但底數是修正前 headroom"),
 ("limitations", "1429", "K/V 拆兩軸使 |A| 自 6 增至 36", "36", "算術", "", "", "6×6=36", "一致", "〔算術〕", ""),
 ("limitations", "1441", "KVP 需為 Qwen2.5-7B 的 112 個 layer-head 各訓練一個 agent（8 GPU DDP）", "112; 8", "外部規格", "", "",
  "28 層×4 KV head=112 ✓；README 的 8 GPU 說法本稽核未查", "找不到來源", "〔算術〕〔未查證〕", "外部文獻，非本 repo 量測"),
 ("附錄 A", "1645", "RTX 3090 sm_86、driver 550.163.01、CUDA 12.4、vLLM 0.28.0、PyTorch 2.13.0+cu129", "見文", "設定",
  G + "env.json", "", "全部相符（torch 編譯對 CUDA 12.9）", "一致", "〔git 52803bc^:results/env.json〕", ""),
 ("附錄 A hygiene", "1650", "該機器由二十餘位使用者共用", "二十餘", "設定", "CLAUDE.md §3", "", "CLAUDE.md：『/ssd7 底下有二十幾個使用者的目錄』", "一致", "〔檔案 CLAUDE.md:143〕", ""),
 ("附錄 A hygiene", "1653", "序列 vs 五卡並行 80 點：full_gpu ±2%，經 PCIe 的 warm −26% 至 −52%", "80; ±2%; −26–−52%", "實測",
  G + "m3_baseline/baseline.csv", "20260830-18xxxx-m3-*（serial／parallel）", "以平均重算 80 格：full_gpu −8.0～+1.4%（另一格 −52.8% 是 RUNLOG 說的容量假象）；卸載 warm −48.6～+19.2%",
  "來源有疑慮", "〔git 52803bc^:results/RUNLOG.md:814〕", "−52% 端點其實是 full_gpu 的容量假象；有一格 +19.2%；host_contention 全為 UNKNOWN"),
 ("附錄 A hygiene", "1656", "KV pool 5.09 與 5.88 GiB 擺動；原始 token 全距 13.5%；正規化 0.04%", "5.09; 5.88; 13.5%; 0.04%", "實測",
  G + "m2_harness/capacity_by_dtype.csv", "20260830-194119-m2-capacity", "13.47%、0.041% ✓", "一致", "〔git 52803bc^:results/m2_harness/capacity_by_dtype.csv〕", ""),
 ("附錄 A hygiene", "1659", "CPU 階 24 GiB、工作集 8 GiB；磁碟階查 2 次 vs 2,048；SSD 與 CPU 僅差 1.2%；縮到 1 GiB 後讀 3 GiB", "見文", "實測",
  G + "RUNLOG.md:541-556", "", "4×16,384×128 KiB=8 GiB ✓；550.9 vs 546.0 ms 差 0.9%（不是 1.2%）", "不一致", "〔git 52803bc^:results/RUNLOG.md:541〕",
  "小錯；被取代的量測沒有 CSV／run_id"),
 ("附錄 A cost", "1668", "qwen-awq NVMe：CPU 2,651.7、SSD 62,328.9 ms（96K，GPU 基準 860.7）；Drop 453.9→6,320.2 ms", "見文", "實測",
  G + "m2_harness/retrieval_cost_qwen-awq.csv；recompute_position_qwen-awq.csv", "20260901-024805-m2-retrieval; 20260831-223228-m2-recompute; 20260901-131533-m2-recompute",
  "中位數完全相符", "來源有疑慮", "〔git 52803bc^:results/m2_harness/retrieval_cost_qwen-awq.csv〕", "時間量測，" + H + "；無 contaminated／own_gpu_intruders 欄"),
 ("tab:costmodels", "1679-1686", "qwen-awq 欄 96,000／0.298／10.245／3.546／0.000178／37,717／258,048", "見表", "實測",
  G + "m4_oracle/qwen-awq/cost_model.json", "同上", "完全相符", "來源有疑慮", "〔git 52803bc^:results/m4_oracle/qwen-awq/cost_model.json〕", "HEAVY；Drop 斜率是兩端點連線（中間點偏差到 7.9%）"),
 ("tab:costmodels", "1679-1686", "llama-bf16 NVMe 欄 0.543／6.286／4.008／0.000210／10,851／24,576", "見表", "實測",
  G + "m2_harness/retrieval_cost_nvme.csv；recompute_position.csv", "20260830-223536-m2-retrieval; 20260830-195552-m2-recompute",
  "(719.4−163.3)/1024=0.543；(6600.5−163.3)/1024=6.286；P*=10,851 ✓", "來源有疑慮", "〔git 52803bc^:results/m2_harness/retrieval_cost_nvme.csv〕",
  "HEAVY；RUNLOG:598 記 NVMe 那次 lookup_async_delay 累計 733 s（根分割區共用 88%）"),
 ("tab:costmodels", "1679-1686", "llama-bf16 SATA 欄 0.588／5.536／4.008／0.000210／7,278／24,576", "見表", "實測",
  G + "m2_harness/retrieval_cost_sata.csv", "20260830-222852-m2-retrieval", "(736.9−134.4)/1024=0.588；(5803.3−134.4)/1024=5.536；P*=7,278 ✓", "來源有疑慮",
  "〔git 52803bc^:results/m2_harness/retrieval_cost_sata.csv〕", "HEAVY"),
 ("附錄 A cost", "1693", "P* 換剖面差 3.5×、換磁碟再 1.5×", "3.5×; 1.5×", "算術", "", "", "3.48；10,851/7,278=1.49", "來源有疑慮", "〔算術〕", "輸入為 HEAVY 下的時間量測"),
 ("附錄 A cost", "1696", "SATA 讀 522、NVMe 2,085 MiB/s（4.0×）；warm 5,803／6,600 ms；有效 353／311 MiB/s＝68%／15%；85% 非 I/O", "見文", "實測",
  G + "m2_harness/disk_bw.csv；retrieval_cost_{sata,nvme}.csv", "20260831-162603-disk-bw; 20260830-222852/223536-m2-retrieval",
  "521.8/2,084.7；5,803.3/6,600.5；2,049.8 MiB÷→353.2/310.5；67.6%/14.9%", "來源有疑慮", "〔git 52803bc^:results/m2_harness/disk_bw.csv〕",
  "數值相符；HEAVY；文中 2,048 MiB 實為 2,049.8"),
 ("附錄 A cost", "1699", "QLC 短測 492（n=3），16 GiB 長測 181（190、172），37%", "492; 181; 37%", "實測",
  G + "m2_harness/disk_bw{,_sustained}.csv", "20260831-162603-disk-bw; 20260831-162634-disk-bw", "491.9；189.6/171.9→180.8；36.8%", "來源有疑慮",
  "〔git 52803bc^:results/m2_harness/disk_bw_sustained.csv〕", "HEAVY；n=2"),
 ("附錄 A cost", "1701", "NVMe 短測寫 2,512；兩碟差 5.1×；SATA 短測高估 2.7×", "2,512; 5.1×; 2.7×", "實測",
  G + "m2_harness/disk_bw.csv", "20260831-162603-disk-bw", "2,511.7；2,512/492=5.1；492/181=2.7", "來源有疑慮", "〔git 52803bc^:results/m2_harness/disk_bw.csv〕", "HEAVY"),
 ("附錄 A oracle", "1711", "17,117 blocks 預算、20 萬次存取、約 3.5×10^9 次比較", "17,117; 3.5e9", "算術", "", "", "273,872/16=17,117；17,117×2e5=3.4e9", "一致", "〔算術〕", ""),
 ("附錄 A oracle", "1716", "2026-08-31：SSD 2 TiB 時 oracle headroom −3.16%", "−3.16%", "實測", G + "RUNLOG.md:1533", "", "RUNLOG 有記載；沒有對應 CSV／run_id", "來源有疑慮",
  "〔git 52803bc^:results/RUNLOG.md:1533〕", "只在 RUNLOG"),
 ("附錄 A oracle", "1719", "模擬器驗證：16K 9.00 vs 9.73（8%）、32K 12.27 vs 12.66（3%）；full_gpu 5,033 vs 5,864、cpu_lru 559 vs 603 ms", "見文", "實測",
  G + "m4_oracle/simulator_validation.json", "（JSON 無 run_id）", "完全相符", "來源有疑慮", "〔git 52803bc^:results/m4_oracle/simulator_validation.json〕",
  "文中稱『絕對值同樣相符』，但 full_gpu 模擬高 16.5%；檔案無 run_id"),
 ("附錄 A oracle", "1722", "缺口後仍留存 0.000%–0.464%；其中首次出現 61.8%–80.1%", "見文", "實測",
  G + "m4_oracle/prefix_gap_probe.csv", "（無 run_id）", "0.0/0.4644；61.83/80.09 ✓", "來源有疑慮", "〔git 52803bc^:results/m4_oracle/prefix_gap_probe.csv〕", "數值相符；檔案無 run_id；device=sata"),
 ("附錄 A ga102", "1730", "71.2 TFLOPS=328 TC×1,695 MHz×128；FLOP/byte 為此值÷31.5 GB/s", "71.2; 2,254", "外部規格", "", "",
  "328×1.695e9×128=71.16 TFLOPS ✓；71.16e12/31.5e9=2,259（tab:ratio 寫 2,254）", "不一致", "〔算術〕", "差 0.2%，極小"),
 ("tab:eps-longctx 圖說", "1739", "FP8 於 14 任務最高 22.0、INT4 最高 1.5", "22.0; 1.5", "實測", G + "m5_quality/longbench_precision.csv", "20260901-193346/193350-m5-longbench",
  "TREC FP8 22.0；GovReport INT4 1.5 ✓", "一致", "〔git 52803bc^:results/m5_quality/longbench_precision.csv〕", "注意：與 L1258『FWE 唯一未全失』互相矛盾"),
 ("tab:eps-longctx", "1747-1755", "LongBench 7 任務逐項分數與巨觀 58.4/6.6/57.0/0.4", "見表", "實測",
  G + "m5_quality/longbench_precision.csv", "20260901-193344/193346/193348/193350-m5-longbench", "28 格全部相符", "一致", "〔git 52803bc^:results/m5_quality/longbench_precision.csv〕", "n=50/任務"),
 ("tab:eps-longctx", "1758-1761", "撈針 4K/8K/16K/32K 四列", "見表", "實測", G + "m5_quality/needle_{ctx_sweep,pilot_32k}.csv",
  "20260831-184433/185342/190401/181930-m5-needle", "16 格全部相符", "一致", "〔git 52803bc^:results/m5_quality/needle_ctx_sweep.csv〕", "n=20"),
 ("tab:eps-longctx", "1764-1772", "RULER 7 任務逐項與巨觀 91.6/3.3/79.1/0.0", "見表", "實測",
  G + "m5_quality/ruler_precision.csv", "20260901-193410/193413/195709/193408-m5-ruler", "28 格全部相符", "一致", "〔git 52803bc^:results/m5_quality/ruler_precision.csv〕", "n=30/任務"),
 ("附錄 A bootstrap", "1778", "10,000 次配對 bootstrap；LongBench 350 題 INT8 差 1.3 [0.0,2.8]；未配對版寬約三倍", "1.3; 三倍", "實測",
  G + "m5_quality/longbench_precision.csv", "20260901-193344/193348-m5-longbench", "配對 1.33 [0.0,2.9] ✓；未配對重算 [−4.8,7.4]，寬 4.2 倍", "來源有疑慮",
  "〔算術〕", "『約三倍』沒有存檔的計算；本稽核重算約 4 倍"),
 ("附錄 A 限制", "1781", "每任務取前 50 筆，非全量 150–200 筆；32K 視窗", "50; 150–200", "設定", "/mlsteam/data/tiara/datasets/longbench/data/*.jsonl", "",
  "multifieldqa_en 150、其餘 6 個 200 ✓；最長 prompt 28,163 token", "一致", "〔檔案 /mlsteam/data/tiara/datasets/longbench/data/qasper.jsonl〕", ""),
 ("附錄 A 計分", "1784", "9,500 組隨機字串，容差 1e-9 下零筆不一致", "9,500; 0", "實測", G + "RUNLOG.md:2294", "", "RUNLOG 記載（code/test_m5c_metrics.py）；沒有 run_id 與輸出檔",
  "來源有疑慮", "〔git 52803bc^:results/RUNLOG.md:2294〕", "只在 RUNLOG"),
]

FIELDS = ["claim_id", "section", "line", "claim_text", "paper_value", "kind", "source_path", "source_run_id",
          "evidence_value", "status", "tag", "note"]
ALLOWED = {"一致", "不一致", "找不到來源", "來源有疑慮"}
with (OUT / "claims_A.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(FIELDS)
    for i, c in enumerate(CLAIMS, 1):
        assert len(c) == 11, c
        assert c[8] in ALLOWED, c
        w.writerow([f"A-{i:03d}", *c])
cnt = collections.Counter(c[8] for c in CLAIMS)
print("[claims_A]", len(CLAIMS), dict(cnt), "run_id", RUN_ID)
EV["_claims_counts"] = dict(cnt)

EV["_meta"] = {"run_id": RUN_ID, "ts": TS, "rev": REV}
(OUT / "platA_evidence.json").write_text(json.dumps(EV, ensure_ascii=False, indent=1, default=str))
print(json.dumps({k: EV[k] for k in ("_meta", "_claims_counts")}, ensure_ascii=False))
