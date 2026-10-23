#!/usr/bin/env python3
"""m16_index.py — 第三階段的結果索引：從 results/m16_stage3_mi300x/ 各實驗的輸出抽關鍵數字，寫 docs/stage3_20261019/results.csv。

每列：exp、cell、metric、value、unit、kind（measured／simulated）、status（ok／CONTAMINATED／NOT_MEASURED）、run_id、source。
只讀檔、不碰 torch；沒有輸出的實驗寫一列 NOT_MEASURED（不跳過、不填 0）。

  python code/m16_index.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import statistics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results/m16_stage3_mi300x")
OUT = os.path.join(REPO, "docs/stage3_20261019/results.csv")
FIELDS = ["exp", "cell", "metric", "value", "unit", "kind", "status", "run_id", "source"]


def rel(p):
    return os.path.relpath(p, REPO)


def i1_prediction():
    rows = []
    for p in glob.glob(os.path.join(RES, "i1_concurrent_restore/prediction/prediction_*.json")):
        d = json.load(open(p))
        for tier, c in d["calibration_N1"].items():
            rows.append(dict(exp="I1-pred", cell=f"N=1|{tier}|idle", metric="sim/meas restore", value=c["ratio"], unit="ratio",
                             kind="simulated", status="ok", run_id=d["run_id"], source=rel(p)))
        for c in d["cells"]:
            cell = f"{c['workload']}|N={c['N']}|{c['gpu_state']}|{c['interference']}"
            for met in ("ttft_p95", "ttft_mean"):
                rows.append(dict(exp="I1-pred", cell=cell, metric=f"{met} gain per-request best vs rr×cake",
                                 value=c[f"{met}_gainA_perreq_vs_rr"], unit="fraction", kind="simulated", status="ok",
                                 run_id=d["run_id"], source=rel(p)))
                rows.append(dict(exp="I1-pred", cell=cell, metric=f"{met} gain joint vs per-request best",
                                 value=c[f"{met}_gainB_joint_vs_perreq"], unit="fraction", kind="simulated", status="ok",
                                 run_id=d["run_id"], source=rel(p)))
    return rows


def i1_measured():
    rows = []
    p = os.path.join(RES, "i1_concurrent_restore/interf_llama.json")
    if os.path.exists(p):
        d = json.load(open(p))
        st = "CONTAMINATED" if d.get("contaminated") else "ok"
        for k, v in d["summary"].items():
            rows.append(dict(exp="I1-interf", cell=k, metric="concurrent/solo (median)", value=v["k_median"], unit="ratio",
                             kind="measured", status=st, run_id=d["run_id"], source=rel(p)))
    p = os.path.join(RES, "i1_concurrent_restore/grid_llama.csv")
    if os.path.exists(p):
        cells = {}
        for r in csv.DictReader(open(p)):
            cells.setdefault((r["gpu_state"], r["workload"], r["N"], r["mode"], r["order"]), []).append(r)
        for k, rs in cells.items():
            st = "CONTAMINATED" if any(r["contaminated"] == "True" for r in rs) else "ok"
            rows.append(dict(exp="I1-grid", cell="|".join(k), metric="measured/predicted mean TTFT (median of reps)",
                             value=round(statistics.median(float(r["ratio_mean"]) for r in rs), 4), unit="ratio",
                             kind="measured", status=st, run_id=rs[0]["run_id"], source=rel(p)))
            rows.append(dict(exp="I1-grid", cell="|".join(k), metric="mean TTFT (median of reps)",
                             value=round(statistics.median(float(r["ttft_mean"]) for r in rs), 5), unit="s",
                             kind="measured", status=st, run_id=rs[0]["run_id"], source=rel(p)))
    return rows


def s1():
    rows = []
    p = os.path.join(RES, "s1_lossless/summary.json")
    if not os.path.exists(p):
        return rows
    d = json.load(open(p))
    for dn, s in d["dumps"].items():
        for k, v in s["bf16"].items():
            if k.startswith("all|") and "ratio" in v:
                rows.append(dict(exp="S1", cell=f"{dn}|{k}", metric="lossless ratio vs BF16", value=v["ratio"], unit="x",
                                 kind="measured", status="ok", run_id=d["run_id"], source=rel(p)))
            if k == "all|entropy":
                rows.append(dict(exp="S1", cell=f"{dn}|entropy", metric="ideal order-0 ratio (planes)",
                                 value=v["ideal_ratio_planes"], unit="x", kind="measured", status="ok", run_id=d["run_id"],
                                 source=rel(p)))
        for k, v in s["speed"].items():
            rows.append(dict(exp="S1", cell=f"{dn}|{k}", metric="decompress throughput", value=v["decomp_MBps"], unit="MB/s",
                             kind="measured", status="ok", run_id=d["run_id"], source=rel(p)))
    return rows


def i4():
    rows = []
    for p in glob.glob(os.path.join(RES, "i4_copy_batching/copybench_sdma*.csv")):
        for r in csv.DictReader(open(p)):
            st = "CONTAMINATED" if r.get("contaminated") == "True" else ("ok" if r.get("status") == "ok" else r.get("status"))
            rows.append(dict(exp="I4", cell=f"sdma={r['sdma']}|{r['gpu_state']}|{r['layout']}|{r['method']}",
                             metric="effective H2D", value=r.get("GiBps", ""), unit="GiB/s", kind="measured", status=st,
                             run_id=r["run_id"], source=rel(p)))
    return rows


def i3s():
    rows = []
    p = os.path.join(RES, "i3_pause_sim/summary.json")
    if not os.path.exists(p):
        return rows
    d = json.load(open(p))
    rows.append(dict(exp="I3s", cell="main scan (96 cells)", metric="pre-registered kappa vs best existing rule (P95 TTFT)",
                     value="0/96 cells >=5%", unit="", kind="simulated", status="ok", run_id=d["run_id"], source=rel(p)))
    return rows


def i3r():
    rows = []
    p = os.path.join(RES, "i3_pause_sim/reread_baselines.json")
    if not os.path.exists(p):
        return rows
    d = json.load(open(p))
    for tag, t in d["tags"].items():
        for k in ("kappa_vs_oracle_per_cell", "kappa_vs_fixed_vllm_tier", "kappa_vs_fixed_default_vllm"):
            rows.append(dict(exp="I3r", cell=f"{tag}|{k}", metric="P95 TTFT gain 1-kappa/baseline (median over cells)",
                             value=t["gains"][k]["median"], unit="fraction", kind="simulated", status="ok",
                             run_id=d["run_id"], source=rel(p)))
    return rows


def k1():
    """K1：每個 dump 的中位數（層×head，不含整層串接列）。只讀 CSV。"""
    rows = []
    d = os.path.join(RES, "k1_kvstruct")
    big = "/mlsteam/data/tiara/runs/20261020-142819-m16-k1-kvstruct/k1_full"   # >1 MB 的檔（BIG_FILES.md）
    if not os.path.exists(os.path.join(big, "lowrank.csv")):
        return rows
    med = lambda v: round(statistics.median(v), 4)
    g = {}
    for r in csv.DictReader(open(os.path.join(big, "lowrank.csv"))):
        if r["head"] != "all" and r["rank"] == "32":
            g.setdefault((r["dump"], r["kind"]), []).append(float(r["rel_err"]))
            rid = r["run_id"]
    for (dn, kind), v in sorted(g.items()):
        rows.append(dict(exp="K1", cell=f"{dn}|{kind}|r=32", metric="low-rank rel. reconstruction error (median layer×head)",
                         value=med(v), unit="fraction", kind="measured", status="ok", run_id=rid,
                         source=os.path.join(big, "lowrank.csv")))
    g = {}
    for r in csv.DictReader(open(os.path.join(d, "xlayer.csv"))):
        if r["head"] != "all":
            g.setdefault((r["dump"], r["kind"]), []).append(float(r["p50"]))
    for (dn, kind), v in sorted(g.items()):
        rows.append(dict(exp="K1", cell=f"{dn}|{kind}|l vs l+1", metric="cross-layer cosine p50 (median over layer×head)",
                         value=med(v), unit="cosine", kind="measured", status="ok", run_id=rid,
                         source=rel(os.path.join(d, "xlayer.csv"))))
    g = {}
    for r in csv.DictReader(open(os.path.join(big, "adjtok.csv"))):
        if r["head"] != "all":
            g.setdefault((r["dump"], r["kind"]), []).append(float(r["frac_ge_0.9"]))
    for (dn, kind), v in sorted(g.items()):
        rows.append(dict(exp="K1", cell=f"{dn}|{kind}|t vs t+1", metric="adjacent-token cosine >=0.9 fraction (median)",
                         value=med(v), unit="fraction", kind="measured", status="ok", run_id=rid,
                         source=os.path.join(big, "adjtok.csv")))
    return rows


def c1a():
    """C1a：tables_c1a.md 的 T2–T4 列（recall@16 − 隨機 k/63，95% CI）。"""
    import re
    rows = []
    p = os.path.join(RES, "c1a_predictive/tables_c1a.md")
    if not os.path.exists(p):
        return rows
    t = open(p).read()
    rid = re.search(r"run：([^、；\n]+)", t).group(1)
    for sec in re.split(r"\n## ", t)[1:]:
        title = sec.split("\n")[0].strip()
        if "T2-T4" not in title:
            continue
        for line in sec.split("\n"):
            c = [x.strip() for x in line.split("|")[1:-1]]
            if len(c) >= 3 and c[0] in ("SNAP", "EXPATT", "KNORM", "KEYDIFF", "H2O", "RND0", "QRY1", "QUEST_MEAN", "QUEST_MAX",
                                        "FULLK_MEAN"):
                rows.append(dict(exp="C1a", cell=f"{title}|{c[0]}", metric="recall@16 minus random (pct pts) [95% CI]",
                                 value=c[1], unit="pct-pt", kind="measured", status="ok", run_id=rid, source=rel(p)))
    return rows


def x1():
    """X1：x1_report.json 的判定（每個模型、每個預算：最好的單一 vs 最好的組合，Δgold_lp 損失；TF 版 r）。〔teacher forcing 篩選〕"""
    rows = []
    p = os.path.join(RES, "x_stack/x1_report.json")
    if not os.path.exists(p):
        return rows
    d = json.load(open(p))
    for m, j in d["judge"]["models"].items():
        for b, v in j["budgets"].items():
            for side in ("single", "combo"):
                rows.append(dict(exp="X1", cell=f"{m}|budget {b}|best {side}: {v['best_' + side]} (ratio {v['best_' + side + '_ratio']})",
                                 metric="loss = -mean Δgold_lp vs E0 (teacher forcing)", value=round(v["loss_" + side], 4),
                                 unit="nats", kind="measured", status="ok", run_id=d["run_id"], source=rel(p)))
        rows.append(dict(exp="X1", cell=f"{m}|tf_M1", metric="r = B*(best combo)/B*(best single) within eps (TF screen)",
                         value=round(j["tf_M1"]["r"], 4), unit="ratio", kind="measured", status="ok", run_id=d["run_id"], source=rel(p)))
    return rows


def x2():
    """X2：x2_feasible.csv＋x2_unmapped.csv 的每列（生成答對率對 E0 的差）＋M1 的 r。"""
    rows = []
    d = os.path.join(RES, "x_stack")
    p = os.path.join(d, "x2_report.json")
    if not os.path.exists(p):
        return rows
    rep = json.load(open(p))
    for m, j in rep["judge"]["models"].items():
        rows.append(dict(exp="X2", cell=f"{m}|M1", metric="r = B*(best combo)/B*(best single) within eps (generation)",
                         value=j["M1"]["r"], unit="ratio", kind="measured", status="ok", run_id=rep["run_id"], source=rel(p)))
    for f in ("x2_feasible.csv", "x2_unmapped.csv"):
        for r in csv.DictReader(open(os.path.join(d, f))):
            rows.append(dict(exp="X2", cell=f"{r['model']}|{r['combo']}|ratio {r['ratio']}|n {r.get('n_items', '')}",
                             metric=f"gen. accuracy delta vs E0 [CI {r.get('acc_delta_lo', '')}, {r.get('acc_delta_hi', '')}]; feasible(eps)={r['feasible']}",
                             value=r["acc_delta"], unit="fraction", kind="measured", status="ok", run_id=r["run_id"],
                             source=rel(os.path.join(d, f))))
    return rows


def x3():
    rows = []
    p = os.path.join(RES, "x3_time/map_summary.json")
    if not os.path.exists(p):
        return rows
    d = json.load(open(p))
    rows.append(dict(exp="X3", cell="map verdict", metric="see map_report.md (STOP-0 holds; L/Q STOP-2a; X STOP-2b, A vs F +2.2%)",
                     value="NO-GO (X incomplete: xKV+INT4 unmapped)", unit="", kind="simulated", status="ok",
                     run_id=d.get("run_id", ""), source=rel(p)))
    return rows


def c1b():
    rows = []
    p = os.path.join(RES, "c1_selective_restore/c1b_cells.csv")
    if not os.path.exists(p):
        return rows
    for r in csv.DictReader(open(p)):
        if r["group"] != "T2-T4":
            continue
        rows.append(dict(exp="C1b", cell=f"{r['model']}|{r['cell']}", metric=f"turn-2 accuracy diff vs FULL [CI {r['diff_lo']}, {r['diff_hi']}]; moved {r['ratio_moved']}",
                         value=r["diff_mean"], unit="fraction", kind="measured", status="ok", run_id=r["run_id"], source=rel(p)))
    return rows


def vllm_rows():
    """V0（每個設定的 overall）、F2（burst makespan 中位數）、F1（f1_report.json 每格的收益）。"""
    import re
    rows = []
    for p in sorted(glob.glob(os.path.join(RES, "v0_vllm_smoke/v0check_*.json"))):
        d = json.load(open(p))
        rows.append(dict(exp="V0", cell=d["run_id"], metric="v0check overall (FAIL only at exact ties is accepted, see experiment_log 失敗 5)",
                         value=d.get("overall"), unit="", kind="measured", status="ok", run_id=d["run_id"], source=rel(p)))
    g = {}
    for p in glob.glob(os.path.join(RES, "f2_shared_prefix_burst/summary_*.json")):
        m = re.search(r"m16-f2-([A-Za-z]+)-P(\d+)-N(\d+)-r\d", p)
        if m:
            d = json.load(open(p))
            g.setdefault((m.group(1), m.group(2), m.group(3)), []).append((d["burst"]["makespan_s"], d["run_id"]))
    for (c, P, N), v in sorted(g.items()):
        rows.append(dict(exp="F2", cell=f"{c}|P{P}|N{N}", metric="burst makespan (median of reps)",
                         value=round(statistics.median(x[0] for x in v), 3), unit="s", kind="measured", status="ok",
                         run_id=v[0][1], source=rel(os.path.join(RES, "f2_shared_prefix_burst"))))
    p = os.path.join(RES, "f1_prefix_reuse/f1_report.json")
    if os.path.exists(p):
        d = json.load(open(p))
        for r in d["rows"]:
            cell = f"L{r['L']}|{r['cap']}|ov{r['overlap']}|{r['arrival']}"
            for k in ("apc_vs_off_mean", "cpu_vs_apc_mean"):
                rows.append(dict(exp="F1", cell=cell, metric=f"{k} (1 - TTFT ratio)", value=r[k], unit="fraction", kind="measured",
                                 status="ok", run_id=d["run_id"], source=rel(p)))
    return rows


def supp_rows():
    """10/22 補跑：K1b（Tucker／跨層）、X4a W 大／全部層、X3 翻轉格子（兩種背景）、I3g、38k 同位元組（事後）。"""
    rows = []
    p = os.path.join(RES, "k1_kvstruct/tucker/summary.json")
    if os.path.exists(p):
        d = json.load(open(p))
        rid = ";".join(d.get("sources", []))
        for k, v in d["q1"].items():
            rows.append(dict(exp="K1b", cell=f"{k}|adjacent layers", metric="token-subspace overlap (centered, top-64) vs same-token cosine; verdict " + v["verdict"],
                             value=round(v["overlap_centered"], 4), unit="fraction", kind="measured", status="ok", run_id=rid, source=rel(p)))
        for k, q in d["q2"].items():
            for t in q["table"]:
                for m in [m for m in t if m.endswith("_ratio") and t[m] is not None]:
                    rows.append(dict(exp="K1b", cell=f"{k}|rhe{t['rhe']}|{m[:-6]}", metric="relerr ÷ xKV-G4 at same bytes (structure only)",
                                     value=round(t[m], 4), unit="ratio", kind="measured", status="ok", run_id=rid, source=rel(p)))
    for sub, tag in (("wbig", "X4a-Wbig"), ("wall", "X4a-Wall")):
        p = os.path.join(RES, f"x_stack/x4/{sub}/x4w_report.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for m, dm in d["models"].items():
            for r in dm["pairs"]:
                c = r["gold_lp_diff"]
                rows.append(dict(exp=tag, cell=f"{m}|T{r['T']}|{r['bits']}", metric=f"gold_lp diff (W big − W4) [CI {c['lo']:.3f}, {c['hi']:.3f}]; mult {r['mult_wbig']} vs {r['mult_w4']}; verdict {dm['verdict']}",
                                 value=round(c["mean"], 4), unit="nats", kind="measured", status="ok", run_id=";".join(d["src_run_ids"]), source=rel(p)))
    for sub, tag in (("v3grid", "X3grid-matmul"), ("v3grid_mem", "X3grid-mem")):
        p = os.path.join(RES, f"x3_time/{sub}/x3grid_report.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for s in d["series"]:
            if s.get("status") != "OK":
                continue
            rows.append(dict(exp=tag, cell=f"{s['model']}|{s['tier']}", metric=f"flip={s['flip']}; best by duty {s['best_by_duty']}; best fixed {s['best_fixed']}; verdict {d['verdict']}",
                             value=round(s["adaptive_gain"], 4) if s.get("adaptive_gain") is not None else "", unit="adaptive gain (fraction)", kind="measured",
                             status="ok", run_id=";".join(d["src_run_ids"]), source=rel(p)))
    p = os.path.join(RES, "i3g_agentic/i3g_report.json")
    if os.path.exists(p):
        d = json.load(open(p))
        for k, c in d["configs"].items():
            rows.append(dict(exp="I3g", cell=f"{k}|{c['desc']}", metric=f"return-turn TTFT P95 (sim pred {c['pred_p95']}); rank {d['verdict_rank']}; flip {d['verdict_flip']}",
                             value=round(c["p95"], 4) if c.get("p95") is not None else "", unit="s", kind="measured", status="ok", run_id=c["run_id"], source=rel(p)))
    p = os.path.join(RES, "i3g_agentic/i3g_posthoc.json")
    if os.path.exists(p):
        d = json.load(open(p))
        for k, r in d["per_config"].items():
            q, io = d["vs_b"].get(f"{k}_vs_b"), d["io"][k]
            rows.append(dict(exp="I3g-posthoc", cell=k, metric=f"return-turn TTFT P95 session-bootstrap CI {r['p95_ci']}" +
                             (f"; ratio vs b {q['p95_ratio']:.3f} CI {q['ratio_ci']}" if q else "") + f"; lookup async delay mean {io['lookup_async_delay_mean']} s (post-hoc)",
                             value=round(r["p95"], 4), unit="s", kind="measured", status="ok", run_id=d["run_id"] + ";" + d["source_report"], source=rel(p)))
    p = os.path.join(RES, "i3g_agentic/rep/i3g_rep_report.json")
    if os.path.exists(p):
        d = json.load(open(p))
        for key, r in d["runs"].items():
            rows.append(dict(exp="I3g-rep", cell=f"{key}|{r['cond']}|card {r['card']}", metric=f"return-turn TTFT P95; verdict {d['verdict']}",
                             value=round(r["p95"], 4) if r.get("p95") is not None else "", unit="s", kind="measured", status="ok",
                             run_id=r["run_id"] + ";" + d["run_id"], source=rel(p)))
    p = os.path.join(RES, "i3g_agentic/qwen/i3g_report.json")
    if os.path.exists(p):
        d = json.load(open(p))
        for k, c in d["configs"].items():
            rows.append(dict(exp="I3g-qwen", cell=f"{k}|{c['desc']}", metric=f"return-turn TTFT P95 (Qwen); rank {d['verdict_rank']}; flip {d['verdict_flip']}",
                             value=round(c["p95"], 4) if c.get("p95") is not None else "", unit="s", kind="measured", status="ok", run_id=c["run_id"], source=rel(p)))
    p = os.path.join(RES, "i3g_agentic/qwen/i3g_posthoc.json")
    if os.path.exists(p):
        d = json.load(open(p))
        for k, r in d["per_config"].items():
            q, io = d["vs_b"].get(f"{k}_vs_b"), d["io"][k]
            rows.append(dict(exp="I3g-qwen-posthoc", cell=k, metric=f"return-turn TTFT P95 (Qwen) session-bootstrap CI {r['p95_ci']}" +
                             (f"; ratio vs b {q['p95_ratio']:.3f} CI {q['ratio_ci']}" if q else "") + f"; lookup async delay mean {io['lookup_async_delay_mean']} s (post-hoc)",
                             value=round(r["p95"], 4), unit="s", kind="measured", status="ok", run_id=d["run_id"] + ";" + d["source_report"], source=rel(p)))
    p = os.path.join(RES, "i3g_agentic/nfs_lookup/nfs_lookup_report.json")
    if os.path.exists(p):
        d = json.load(open(p))
        for lab, r in d["runs"].items():
            k = {"nfs": "d", "local": "c"}.get(lab)
            for ph, q in r["phases"].items():
                rho = d["rho"].get(k, {}).get(ph) if isinstance(d["rho"].get(k), dict) else None
                rows.append(dict(exp="nfs-lookup", cell=f"{lab}|{ph}", metric=f"single-thread access(F_OK) mean latency; rho vs I3g {k} = {rho}; verdict {d['verdict']} / {d['verdict_cause']}",
                                 value=round(q["all"]["mean"], 7), unit="s", kind="measured", status="ok", run_id=r["run_id"] + ";" + d["run_id"], source=rel(p)))
    p = os.path.join(RES, "x_stack/v3/xlcal_eqbytes.json")
    if os.path.exists(p):
        d = json.load(open(p))
        for m, rs in d["models"].items():
            for r in rs:
                c = r["diff"]
                rows.append(dict(exp="xlcal-eqbytes", cell=f"{m}|{r['A']} vs {r['B']}", metric=f"accuracy diff A−B [CI {c['lo']:.4f}, {c['hi']:.4f}]; bytes {r['ratio_A']} vs {r['ratio_B']} (post-hoc)",
                                 value=round(c["mean"], 4), unit="fraction", kind="measured", status="ok", run_id=d["run_id"] + ";" + d["source_run"], source=rel(p)))
    return rows


PLANNED = ["S1", "K1", "I4", "I1-interf", "I1-grid", "C1a", "X1", "X2", "X3", "V0", "F2", "F1", "I3s", "I3g", "C1b",
           "K1b", "X4a-Wbig", "X4a-Wall", "X3grid-matmul", "X3grid-mem", "xlcal-eqbytes", "I3g-posthoc", "I3g-rep", "I3g-qwen", "I3g-qwen-posthoc", "nfs-lookup"]


def main():
    rows = i1_prediction() + i1_measured() + s1() + i4() + i3s() + i3r() + k1() + c1a() + x1() + x2() + x3() + vllm_rows() + c1b() + supp_rows()
    have = {r["exp"] for r in rows}
    for e in PLANNED:
        if e not in have:
            rows.append(dict(exp=e, cell="", metric="", value="", unit="", kind="", status="NOT_MEASURED", run_id="",
                             source=""))
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"寫好 {rel(OUT)}：{len(rows)} 列；有資料的實驗：{sorted(have)}")


if __name__ == "__main__":
    main()
