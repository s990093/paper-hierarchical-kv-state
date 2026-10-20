"""Report-only (07 Sec.2 / Sec.6, 03 Sec.3.2 / Sec.3.3b) analyses that no prior file summarises:

* F30  T1 (multi-key NIAH, ceiling task) and LongBench-6 for the endpoints E1/E1s/E1sr at both tiers,
       plus T1 for the Llama INT4 mixed configs (Step 2 was reached for Llama).
* F31  per-block sensitivity map, Llama INT4-KIVI (the main model of the reached INT4 Step 2).

Sealing rule applied (REPORT.md Sec.6 / appendix): a sealed result is analysed only if the decision-flow node it
belongs to was executed. Not analysed here (ran, not analysed): FP8 mixed configs (stage 45, both models),
Qwen INT4 mixed configs (stage 46), Llama Step-2b rows (E1-K0/E1-INT8), sensitivity maps other than Llama INT4.

Statistics reuse the frozen code (code/m10_analyze.py: cond_rows, eps_check, cluster_boot); nothing new.
Also writes docs/precision_20261013/report_stage2/derived/report_only.json with every number plotted.
"""
from __future__ import annotations

import json
import os
import statistics
from collections import defaultdict

import numpy as np

import m10_analyze as A
from common import DERIVED, DOUBLE, FMT, OI, begin, panel_label, plt, read_csv, refline, rp, save

P = ("m10_precision_mi300x",)
MODELS = ["llama", "qwen"]
MLAB = {"llama": "Llama-3.1-8B", "qwen": "Qwen2.5-7B-1M"}
OUTJ = {}


def _rows(paths, strip_fmt=None):
    rows = []
    for p in paths:
        for r in read_csv(p):
            r["turn"] = int(r["turn"])
            r["score"] = float(r["score"])
            rows.append(r)
    if strip_fmt:
        keep = []
        for r in rows:
            name, _, f = r["config"].partition("@")
            if r["config"] == "E0":
                keep.append(r)
            elif f == strip_fmt:
                r["config"] = name
                keep.append(r)
        rows = keep
    return rows


def _shards(model, kind):
    d = rp(*P, "quality")
    names = sorted({f[:-4] if f.endswith(".csv") else f[:-7] for f in os.listdir(d)
                    if f.startswith(f"step1_{model}_{kind}") and (f.endswith(".csv") or f.endswith(".csv.gz"))})
    return [os.path.join(d, n + ".csv") for n in names]


def _sealed(name):
    d = rp(*P, "quality_sealed")
    return [os.path.join(d, name + ".csv")]


def analyse():
    """Return {(model, tier, task_kind): {config: {cond: eps_check dict}}}."""
    res = {}
    for m in MODELS:
        for kind, tasks, conds in (("T1", ["T1"], A.CONDS), ("LB", None, ["A"])):
            base = _rows(_shards(m, kind))
            lbt = sorted({r["task"] for r in base if r["task"].startswith("LB-")})
            tk = tasks or lbt
            ref = A.cond_rows(base, "E0", "A")
            ref = {k: r for k, r in ref.items() if r["task"] in tk}
            # FP8 tier endpoints (Step 1 node reached for both models)
            out = {}
            for cfg in ("E1", "E1s", "E1sr"):
                out[cfg] = {}
                for cond in conds:
                    x = {k: r for k, r in A.cond_rows(base, cfg, cond).items() if r["task"] in tk}
                    if x:
                        out[cfg][cond] = A.eps_check(ref, x, main_tasks=tk)
            res[(m, "fp8", kind)] = out
            # INT4 tier endpoints (Step 1' node reached for both models); E0 rows come from the Step-1 files
            r4 = _rows(_sealed(f"int4end_{m}_{kind}"), strip_fmt="int4_kivi")
            out = {}
            for cfg in ("E1", "E1s", "E1sr"):
                out[cfg] = {}
                for cond in conds:
                    x = {k: r for k, r in A.cond_rows(r4, cfg, cond).items() if r["task"] in tk}
                    if x:
                        out[cfg][cond] = A.eps_check(ref, x, main_tasks=tk)
            res[(m, "int4", kind)] = out
            if m == "llama" and kind == "T1":       # Step 2 reached for Llama: T1 of the mixed configs, vs E0 and vs R25
                rm = _rows(_sealed("int4mix_llama_T1"), strip_fmt="int4_kivi")
                out = {}
                for cfg in ("F25", "M25", "B25", "R25"):
                    out[cfg] = {}
                    for cond in conds:
                        x = A.cond_rows(rm, cfg, cond)
                        if x:
                            out[cfg][cond] = A.eps_check(ref, x, main_tasks=tk)
                vs_r = {}
                for cfg in ("F25", "M25", "B25"):
                    vs_r[cfg] = {}
                    for cond in ("B-idle", "B-busy"):
                        rr = A.cond_rows(rm, "R25", cond)
                        xx = A.cond_rows(rm, cfg, cond)
                        d, n, *_ = A.paired(rr, xx, tasks=tk)
                        p, lo, hi = A.cluster_boot(d)
                        vs_r[cfg][cond] = {"diff": round(p, 4), "ci95": [round(lo, 4), round(hi, 4)], "n_rows": n}
                res[("llama", "int4mix", "T1")] = out
                res[("llama", "int4mix_vsR25", "T1")] = vs_r
            # E0 accuracy for context
            e0 = statistics.mean(r["score"] for r in ref.values())
            res[(m, "e0acc", kind)] = {"acc": round(e0, 4), "n_rows": len(ref)}
    return res


def f30_t1_longbench():
    begin("F30_T1_LongBench_reportonly")
    res = analyse()
    OUTJ["T1_LB"] = {"|".join(k): v for k, v in res.items()}
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.7), gridspec_kw=dict(width_ratios=[1.25, 1.0, 1.0]))
    # (a) T1, endpoints, all four readings, both tiers & models
    ax = axes[0]
    labels, k = [], 0
    for m in MODELS:
        for tier in ("fp8", "int4"):
            for cfg in ("E1", "E1s", "E1sr"):
                for j, cond in enumerate(A.CONDS):
                    e = res[(m, tier, "T1")][cfg].get(cond)
                    if e is None:
                        continue
                    d, lo, hi = (100 * e["diff"], 100 * e["ci95"][0], 100 * e["ci95"][1])
                    col = FMT["fp8"] if tier == "fp8" else FMT["int4"]
                    ax.errorbar([k + 0.18 * (j - 1.5)], [d], yerr=[[d - lo], [hi - d]], fmt="o", ms=2.6, color=col,
                                mfc=col if cond == "A" else "white", elinewidth=0.6, lw=0)
                labels.append(f"{'L' if m == 'llama' else 'Q'} {tier.upper()} {cfg}")
                k += 1
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5.6)
    refline(ax, 0, color="#333333", ls="-", lw=0.5)
    refline(ax, -2, color=OI["verm"], ls="--")
    ax.set_ylabel("accuracy vs E0 (pp, 95% CI)")
    ax.set_title("T1 multi-key NIAH (4 readings each)")
    panel_label(ax, "a")
    e0t1 = {m: res[(m, "e0acc", "T1")]["acc"] for m in MODELS}
    ax.set_xlabel("filled = reading A; hollow = B idle / busy / idle->busy", fontsize=6.2)
    # (b) LongBench-6, reading A
    ax = axes[1]
    labels, k = [], 0
    for m in MODELS:
        for tier in ("fp8", "int4"):
            for cfg in ("E1", "E1s", "E1sr"):
                e = res[(m, tier, "LB")][cfg]["A"]
                d, lo, hi = (100 * e["diff"], 100 * e["ci95"][0], 100 * e["ci95"][1])
                col = FMT["fp8"] if tier == "fp8" else FMT["int4"]
                ax.errorbar([k], [d], yerr=[[d - lo], [hi - d]], fmt="o", ms=3, color=col, elinewidth=0.7, lw=0)
                labels.append(f"{'L' if m == 'llama' else 'Q'} {tier.upper()} {cfg}")
                k += 1
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5.6)
    refline(ax, 0, color="#333333", ls="-", lw=0.5)
    refline(ax, -2, color=OI["verm"], ls="--")
    ax.set_ylabel("score vs E0 (points, 95% CI)")
    ax.set_title("LongBench-6, reading A")
    panel_label(ax, "b")
    e0lb = {m: res[(m, "e0acc", "LB")]["acc"] for m in MODELS}

    # (c) T1 for Llama INT4 mixed configs vs R25 (reading B)
    ax = axes[2]
    vr = res[("llama", "int4mix_vsR25", "T1")]
    mk = {"F25": ">", "M25": "D", "B25": "<"}
    for i, cfg in enumerate(("F25", "M25", "B25")):
        for j, cond in enumerate(("B-idle", "B-busy")):
            e = vr[cfg][cond]
            d, lo, hi = (100 * e["diff"], 100 * e["ci95"][0], 100 * e["ci95"][1])
            ax.errorbar([i + (j - 0.5) * 0.25], [d], yerr=[[d - lo], [hi - d]], fmt=mk[cfg], ms=3.6, color=FMT["int4"],
                        mfc=FMT["int4"] if cond == "B-idle" else "white", elinewidth=0.8, lw=0)
    for yv in (2, -2):
        refline(ax, yv, color=OI["verm"], ls="--")
    refline(ax, 0, color="#333333", ls="-", lw=0.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["F25", "M25", "B25"])
    ax.set_ylabel("T1 accuracy vs R25 (pp, 95% CI)")
    ax.set_title("T1, Llama INT4 mixed vs R25 (b32)")
    ax.set_xlabel("filled = GPU idle, hollow = GPU busy", fontsize=6.2)
    panel_label(ax, "c")
    fig.tight_layout()
    return save(fig, "F30_T1_LongBench_reportonly")


def f31_sensmap_llama_int4():
    begin("F31_sensmap_llama_int4_reportonly")
    d = rp(*P, "quality_sealed")
    rows = read_csv(os.path.join(d, "sensmap_llama_int4_kivi_s0.csv")) + \
        read_csv(os.path.join(d, "sensmap_llama_int4_kivi_s1.csv"))
    by_block = defaultdict(dict)
    dl_block = defaultdict(dict)
    hit, miss = defaultdict(list), defaultdict(list)
    for r in rows:
        b = int(r["block"])
        iid = (r["item_id"], r["turn"])
        kl = float(r["kl_vs_e0"])
        by_block[b][iid] = [kl]
        dl_block[b][iid] = [float(r["gold_logprob"]) - float(r["gold_logprob_e0"])]
        needles = json.loads(r["needle_blocks"]) if r["needle_blocks"] not in ("", "[]") else []
        (hit if b in needles else miss)[r["item_id"]].append(kl)
    blocks = sorted(by_block)
    kl_pt, kl_lo, kl_hi, dl_pt, dl_lo, dl_hi = [], [], [], [], [], []
    for b in blocks:
        p, lo, hi = A.cluster_boot(by_block[b], n_boot=2000)
        kl_pt.append(p), kl_lo.append(lo), kl_hi.append(hi)
        p, lo, hi = A.cluster_boot(dl_block[b], n_boot=2000)
        dl_pt.append(p), dl_lo.append(lo), dl_hi.append(hi)
    hp, hlo, hhi = A.cluster_boot(hit, n_boot=2000)
    mp, mlo, mhi = A.cluster_boot(miss, n_boot=2000)
    n_items = len({r["item_id"] for r in rows})
    OUTJ["sensmap_llama_int4"] = {"run_ids": sorted({r["run_id"] for r in rows}), "n_items": n_items,
                                  "n_rows": len(rows),
                                  "per_block_mean_kl": dict(zip(blocks, [round(x, 6) for x in kl_pt])),
                                  "per_block_kl_ci": dict(zip(blocks, [[round(a, 6), round(b, 6)]
                                                                       for a, b in zip(kl_lo, kl_hi)])),
                                  "per_block_mean_dlogp": dict(zip(blocks, [round(x, 5) for x in dl_pt])),
                                  "answer_block_kl": [hp, hlo, hhi], "other_block_kl": [mp, mlo, mhi],
                                  "answer_rows": sum(len(v) for v in hit.values()),
                                  "other_rows": sum(len(v) for v in miss.values())}
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.25), gridspec_kw=dict(width_ratios=[2.2, 1.0]))
    ax = axes[0]
    for lo_b, hi_b, lab in ((1, 16, "front band"), (24, 39, "middle band"), (48, 63, "back band")):
        ax.axvspan(lo_b - 0.5, hi_b + 0.5, color="#f0f0f0", zorder=0)
        ax.text((lo_b + hi_b) / 2, 1.0, lab, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=6.0, color="#666666")
    kl_pt, kl_lo, kl_hi = map(np.array, (kl_pt, kl_lo, kl_hi))
    ax.fill_between(blocks, kl_lo, kl_hi, color=FMT["int4"], alpha=0.25, lw=0)
    ax.plot(blocks, kl_pt, color=FMT["int4"], lw=0.9, marker="o", ms=1.8)
    ax.set_yscale("log")
    ax.set_xlim(-0.8, 63.8)
    ax.set_xlabel("the single block stored as INT4-KIVI (all others BF16)")
    ax.set_ylabel("mean KL(answer || E0)\n(95% CI over items)")
    ax.set_title(f"Per-block sensitivity map, Llama, {n_items} items (reading A)", pad=10)
    ax = axes[1]
    ax.errorbar([0, 1], [hp, mp], yerr=[[hp - hlo, mp - mlo], [hhi - hp, mhi - mp]], fmt="o", ms=4,
                color=FMT["int4"], elinewidth=0.9, lw=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["block holds\nthe answer", "any other\nblock"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("mean KL (95% CI)")
    ax.set_title("Hit the answer vs not", pad=10)
    fig.tight_layout()
    return save(fig, "F31_sensmap_llama_int4_reportonly")


def f39_write_derived():
    """Not a figure: dump the numbers computed above for the report appendix."""
    os.makedirs(DERIVED, exist_ok=True)
    p = os.path.join(DERIVED, "report_only.json")
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(json.loads(json.dumps(OUTJ, default=float)))
    with open(p, "w") as f:
        json.dump(old, f, indent=1, ensure_ascii=False)
    return []


ALL = [f30_t1_longbench, f31_sensmap_llama_int4, f39_write_derived]
