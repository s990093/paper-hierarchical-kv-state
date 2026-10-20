"""Main pre-registered line (07_freeze_mi300x): concept, decision flow, K0, sources, endpoints, Step 2, quality-latency."""
from __future__ import annotations

import os
import statistics
from collections import defaultdict

import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from common import (CFG_MARK, DOUBLE, FMT, OI, SINGLE, TIER_LABEL, begin, load_json, panel_label, plt, read_csv,
                    refline, rp, save)

P = ("m10_precision_mi300x",)
MODELS = ["llama", "qwen"]
MLAB = {"llama": "Llama-3.1-8B", "qwen": "Qwen2.5-7B-1M"}
FMTNAME = {"llama": {"fp8": "fp8_token", "int4": "int4_kivi"}, "qwen": {"fp8": "fp8_head", "int4": "int4_kivi"}}


def fmt_choice():
    return load_json(rp(*P, "fmt_select", "fmt_choice.json"))["choice"]


# --------------------------------------------------------------------------------------------- F01
def f01_concept_cake():
    """Schematic of Cake bidirectional restore and which stored low-precision blocks are actually read."""
    begin("F01_concept_cake")
    n = 16
    rows = [("Slow tier, GPU idle: the two lines meet late", 9),
            ("Slow tier, GPU busy: recompute is slower, they meet earlier", 3),
            ("Fast tier (real CPU H2D): loading is far cheaper, they meet at the front", 1)]
    lp_front = set(range(1, 5))     # schematic F25-like placement (block 0 = sink, kept BF16)
    lp_back = set(range(12, 16))    # schematic B25-like placement
    fig, ax = plt.subplots(figsize=(DOUBLE, 3.0))
    ax.set_xlim(-0.3, n + 0.3)
    ax.set_ylim(-0.05, len(rows) * 1.5 - 0.05)
    ax.axis("off")
    for r, (title, b) in enumerate(rows):
        y = (len(rows) - 1 - r) * 1.5 + 0.3
        ax.text(0, y + 0.86, title, fontsize=7.2, ha="left", va="center", fontweight="bold")
        for i in range(n):
            lp = i in lp_front or i in lp_back
            recomputed = i < b
            face = "#d9d9d9" if recomputed else (FMT["fp8"] if lp else "#ffffff")
            ax.add_patch(Rectangle((i, y), 0.92, 0.58, facecolor=face, edgecolor="#333333", lw=0.5))
            if lp and recomputed:      # stored low precision but never read
                ax.add_patch(Rectangle((i, y), 0.92, 0.58, facecolor="none", edgecolor=FMT["fp8"], lw=0.0,
                                       hatch="////"))
        ax.annotate("", xy=(b - 0.05, y - 0.12), xytext=(0, y - 0.12),
                    arrowprops=dict(arrowstyle="->", lw=0.9, color="#444444"))
        ax.annotate("", xy=(b + 0.05, y - 0.12), xytext=(n - 0.08, y - 0.12),
                    arrowprops=dict(arrowstyle="->", lw=0.9, color=FMT["fp8"]))
        ax.plot([b - 0.04, b - 0.04], [y - 0.2, y + 0.66], color=OI["verm"], lw=1.1)
        ax.text(b + 0.1, y + 0.66, "meet point b", fontsize=6.2, ha="left", va="center", color=OI["verm"])
        if r == 0:
            ax.text(0.05, y - 0.3, "GPU recomputes from block 0 (exact BF16)", fontsize=6.2, ha="left", va="center",
                    color="#444444")
            ax.text(n - 0.1, y - 0.3, "I/O loads from the back (stored precision)", fontsize=6.2, ha="right",
                    va="center", color=FMT["fp8"])
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#d9d9d9", edgecolor="#333333", lw=0.5, label="recomputed (exact BF16)"),
               Patch(facecolor="#ffffff", edgecolor="#333333", lw=0.5, label="loaded BF16"),
               Patch(facecolor=FMT["fp8"], edgecolor="#333333", lw=0.5, label="loaded low precision (read)"),
               Patch(facecolor="#d9d9d9", edgecolor=FMT["fp8"], hatch="////", lw=0.5,
                     label="stored low precision, never read")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=6.4)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return save(fig, "F01_concept_cake")


# --------------------------------------------------------------------------------------------- F02
def f02_decision_flow():
    """Pre-registered decision flow; outcome labels are read from the frozen decision JSONs."""
    begin("F02_decision_flow")
    A = rp(*P, "analysis")
    k0 = {(m, t): load_json(rp(*P, f"speed_{m}" + ("_regrid" if (m == "qwen" and t == "fp8") else ""),
                                f"k0_verdict_{t}.json")) for m in MODELS for t in ("fp8", "int4")}
    dfp8 = load_json(f"{A}/fp8/decide_fp8.json")
    k3 = load_json(f"{A}/fp8/k3_qwen.json")
    dint4 = load_json(f"{A}/int4/decide_int4.json")
    st2 = load_json(f"{A}/int4/step2_llama_int4.json")
    q4 = {m: load_json(f"{A}/int4/quality_{m}_int4.json") for m in MODELS}
    sp4 = {m: load_json(f"{A}/int4/speed_{m}.json") for m in MODELS}

    def wins(v):
        return sum(c["E1_wins"] for c in v["cells"]), len(v["cells"])

    def tf(b):
        return "yes" if b else "no"

    eps4 = {m: all(q4[m]["configs"]["E1"][c]["eps_pass"] for c in ("A", "B-idle", "B-busy", "B-i2b")) for m in MODELS}
    fail_cells = {m: sorted({f"{c['bw']}/{c['gpu_state']}" for c in sp4[m]["cells"] if c["config"] == "E1"
                             and not c["not_worse"]}) for m in MODELS}
    nodes = [
        ("K0 (FP8): does all-FP8 restore faster?",
         f"E1 wins {wins(k0[('llama','fp8')])[0]}/{wins(k0[('llama','fp8')])[1]} cells (Llama), "
         f"{wins(k0[('qwen','fp8')])[0]}/{wins(k0[('qwen','fp8')])[1]} (Qwen)  ->  K0 false, continue", True),
        ("Step 1 (FP8 endpoints): is uniform FP8 enough?",
         f"Llama K1 = {tf(dfp8['llama']['K1'])};  Qwen K1 = {tf(dfp8['qwen']['K1'])}, K1s = {tf(dfp8['qwen']['K1s'])}", True),
        ("Step 2b (Qwen): does an 8-bit codec change fix it?",
         f"E1-K0 pass = {tf(k3['per_config']['E1s@fp8k0_head']['pass'])}, E1-INT8 pass = "
         f"{tf(k3['per_config']['E1s@int8_token']['pass'])}  ->  K3 = {tf(k3['K3'])}: FP8 tier needs no placement", True),
        ("K0 (INT4) + Step 1' (INT4 endpoints): is uniform INT4 enough?",
         f"K0: E1 wins {wins(k0[('llama','int4')])[0]}/{wins(k0[('llama','int4')])[1]} (Llama), "
         f"{wins(k0[('qwen','int4')])[0]}/{wins(k0[('qwen','int4')])[1]} (Qwen) -> K0 false.  "
         f"Quality eps, all 4 readings: Llama {tf(eps4['llama'])}, Qwen {tf(eps4['qwen'])}.\n"
         f"Speed-not-worse fails only on the real CPU tier (Llama {', '.join(fail_cells['llama'])}; "
         f"Qwen {', '.join(fail_cells['qwen'])})  ->  K1 = K1s = no", True),
        ("Route (03 Sec.1.1): uniform INT4 'not enough' for both -> Step 2 on Llama",
         "Qwen INT4 mixed configs split out (not part of the decision)", True),
        ("Step 2 (INT4, Llama): F25/M25/B25 vs R25",
         f"stable gap (cond. 1-2) = {tf(st2['stable_gap_1_2'])}  ->  K2 = {tf(st2['K2'])}: position adds nothing; STOP", True),
    ]
    hts = [1.05 if "\n" in body else 0.78 for _, body, _ in nodes]
    gap = 0.28
    total = sum(hts) + gap * (len(nodes) - 1) + 0.35
    fig, ax = plt.subplots(figsize=(DOUBLE, 3.7))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, total)
    ytop = total - 0.06
    for i, ((head, body, on_path), h) in enumerate(zip(nodes, hts)):
        y0 = ytop - h
        box = FancyBboxPatch((0.15, y0), 9.7, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                             facecolor="#f2f7fb" if on_path else "#ffffff", edgecolor=FMT["fp8"], lw=0.8)
        ax.add_patch(box)
        ax.text(0.3, ytop - 0.24, f"{i + 1}. {head}", fontsize=7.4, fontweight="bold", va="center")
        ax.text(0.45, ytop - (0.68 if "\n" in body else 0.53), body, fontsize=6.6, va="center",
                color="#222222", linespacing=1.2)
        if i < len(nodes) - 1:
            ax.annotate("", xy=(5.0, y0 - gap + 0.02), xytext=(5.0, y0 - 0.02),
                        arrowprops=dict(arrowstyle="->", lw=0.8, color="#444444"))
        ytop = y0 - gap
    ax.text(9.85, 0.05, "not reached: FP8 Step 2, Step 3 (sensitivity-aware rule S25)", fontsize=6.3, ha="right",
            color="#666666", style="italic")
    return save(fig, "F02_decision_flow")


# --------------------------------------------------------------------------------------------- F03
def k0_files():
    out = {}
    for m in MODELS:
        out[(m, "fp8")] = rp(*P, "speed_qwen_regrid" if m == "qwen" else "speed_llama", "k0_verdict_fp8.json")
        out[(m, "int4")] = rp(*P, f"speed_{m}", "k0_verdict_int4.json")
    return out


def f03_k0_speed():
    begin("F03_k0_speed")
    files = k0_files()
    tiers = ["cpu", "b16", "b32", "b48"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.45), sharey=True)
    for ax, m in zip(axes, MODELS):
        for j, tier in enumerate(("fp8", "int4")):
            v = load_json(files[(m, tier)])
            rows = [r for r in read_csv(os.path.join(os.path.dirname(files[(m, tier)]), "restores.csv"))
                    if r["run_id"] == v["run_id"] and r["mode"] == "cake" and r["config"] in ("E0", "E1")
                    and r["length"] == "32768"]
            for k, st in enumerate(("idle", "busy")):
                xs, ys, lo, hi = [], [], [], []
                for t_i, t in enumerate(tiers):
                    c = next(c for c in v["cells"] if c["bw_name"] == t and c["gpu_state"] == st)
                    e0 = c["median_E0_s"]
                    cell = [r for r in rows if r["bw_name"] == t and r["gpu_state"] == st]
                    p0 = {(r["launch"], r["rep"]): float(r["t_restore"]) for r in cell if r["config"] == "E0"}
                    p1 = {(r["launch"], r["rep"]): float(r["t_restore"]) for r in cell if r["config"] == "E1"}
                    keys = sorted(set(p0) & set(p1))
                    assert len(keys) == c["n_pairs"], (m, tier, t, st, len(keys), c["n_pairs"])
                    md = statistics.median(p0[k_] - p1[k_] for k_ in keys)
                    xs.append(t_i + (j - 0.5) * 0.36 + (k - 0.5) * 0.16)
                    ys.append(100 * md / e0)
                    lo.append(100 * c["paired_diff_ci95_s"][0] / e0)
                    hi.append(100 * c["paired_diff_ci95_s"][1] / e0)
                ys, lo, hi = np.array(ys), np.array(lo), np.array(hi)
                col = FMT["fp8"] if tier == "fp8" else FMT["int4"]
                ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt="o",
                            ms=3.6, mfc=col if st == "idle" else "white", mec=col, ecolor=col, elinewidth=0.7,
                            lw=0, label=f"{'FP8' if tier == 'fp8' else 'INT4-KIVI'}, GPU {st}")
        ds = load_json(files[(m, "fp8")])["delta_star"]
        refline(ax, 100 * ds, color="#777777", ls=":", lw=0.7)
        refline(ax, 0, color="#333333", ls="-", lw=0.6)
        ax.set_xticks(range(len(tiers)))
        ax.set_xticklabels([TIER_LABEL[t] for t in tiers])
        ax.set_title(MLAB[m])
        ax.set_xlim(-0.6, len(tiers) - 0.4)
        ax.text(len(tiers) - 0.45, 100 * ds + 2.0, f"win threshold delta* = {100 * ds:.2f}%", fontsize=6.2,
                ha="right", color="#555555")
        ax.set_xlabel("storage tier")
    axes[0].set_ylabel("E1 restore-time saving vs E0 (%)\nmedian paired diff / E0 median")
    axes[0].legend(loc="lower right", ncol=1, fontsize=6.3)
    fig.tight_layout()
    return save(fig, "F03_k0_speed")


# --------------------------------------------------------------------------------------------- F04
def f04_sources_read():
    """What is actually read: mean blocks recomputed / loaded BF16 / loaded low precision per config (32K, Cake)."""
    begin("F04_sources_read")
    srcs = {"llama": (rp(*P, "speed_llama", "restores.csv"), "20261015-161226-m10-grid-fp8-llama"),
            "qwen": (rp(*P, "speed_qwen_regrid", "restores.csv"), "20261015-194150-m10-regrid-fp8-qwen")}
    cfgs = ["E0", "E1", "E1s", "E1sr", "F25", "M25", "B25", "R25"]
    conds = [("b32", "idle"), ("b32", "busy"), ("cpu", "idle")]
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE, 3.5), sharex=True, sharey=True)
    for r, m in enumerate(MODELS):
        path, rid = srcs[m]
        rows = [x for x in read_csv(path) if x["run_id"] == rid and x["length"] == "32768" and x["mode"] == "cake"
                and x["contaminated_gpu"] == "False"]
        for c, (bw, st) in enumerate(conds):
            ax = axes[r, c]
            for i, cfg in enumerate(cfgs):
                xs = [x for x in rows if x["config"] == cfg and x["bw_name"] == bw and x["gpu_state"] == st]
                if not xs:
                    continue
                rec = statistics.mean(float(x["n_recompute"]) for x in xs)
                lb = statistics.mean(float(x["n_load_bf16"]) for x in xs)
                lp = statistics.mean(float(x["fp8_read"]) for x in xs)
                stored = statistics.mean(float(x["fp8_stored"]) for x in xs)
                y = len(cfgs) - 1 - i
                ax.barh(y, rec, color="#d9d9d9", edgecolor="white", lw=0.4, height=0.72)
                ax.barh(y, stored - lp, left=rec - (stored - lp), color="none", edgecolor=FMT["fp8"], hatch="//////",
                        lw=0, height=0.72)
                ax.barh(y, lb, left=rec, color="#ffffff", edgecolor="#555555", lw=0.4, height=0.72)
                ax.barh(y, lp, left=rec + lb, color=FMT["fp8"], edgecolor="white", lw=0.4, height=0.72)
                if cfg in ("F25", "B25", "E1"):
                    ax.text(65.5, y, f"{lp:.0f}/{stored:.0f}", fontsize=5.8, va="center", ha="left", color="#333333")
            ax.set_yticks(range(len(cfgs)))
            ax.set_yticklabels(cfgs[::-1])
            ax.set_xlim(0, 72)
            ax.grid(axis="y", visible=False)
            if r == 0:
                ax.set_title(f"{bw}, GPU {st}" if bw != "cpu" else "CPU (real H2D), GPU idle")
            if c == 0:
                ax.set_ylabel(MLAB[m])
        axes[r, 0].set_xticks([0, 16, 32, 48, 64])
    for ax in axes[1]:
        ax.set_xlabel("blocks (of 64; mean over restores)")
    handles = [Rectangle((0, 0), 1, 1, facecolor="#d9d9d9"), Rectangle((0, 0), 1, 1, facecolor="none",
                                                                       edgecolor=FMT["fp8"], hatch="//////", lw=0),
               Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#555555", lw=0.4),
               Rectangle((0, 0), 1, 1, facecolor=FMT["fp8"])]
    fig.legend(handles, ["recomputed (BF16)", "of which: stored FP8, not read", "loaded BF16", "loaded FP8 (read)"],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.text(0.99, 0.995, "numbers at right: FP8 blocks read / stored", fontsize=6, ha="right", va="top",
             color="#444444")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "F04_sources_read")


# --------------------------------------------------------------------------------------------- F05
COND_LAB = {"A": "A: load all", "B-idle": "B: Cake, idle", "B-busy": "B: Cake, busy", "B-i2b": "B: idle->busy (turn 2)"}


def _forest(ax, items, xlim=None, eps=-2.0):
    """items: list of (label, diff_pp, lo, hi, color, marker, eps_pass)."""
    for k, (lab, d, lo, hi, col, mk, ok) in enumerate(items):
        y = len(items) - 1 - k
        ax.errorbar([d], [y], xerr=[[d - lo], [hi - d]], fmt=mk, ms=3.6, color=col, mfc=col if ok else "white",
                    mec=col, elinewidth=0.8, lw=0)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([it[0] for it in items][::-1], fontsize=6.3)
    refline(ax, 0, axis="x", color="#333333", ls="-", lw=0.6)
    refline(ax, eps, axis="x", color=OI["verm"], ls="--", lw=0.7)
    ax.grid(axis="y", visible=False)
    if xlim:
        ax.set_xlim(*xlim)


def _q_items(qjson, cfgs, colfn, prefix=""):
    items = []
    for cfg in cfgs:
        for cond in ("A", "B-idle", "B-busy", "B-i2b"):
            v = qjson["configs"][cfg][cond]
            items.append((f"{prefix}{cfg.split('@')[0] if '@' not in cfg else cfg_label(cfg)}  {COND_LAB[cond]}",
                          100 * v["diff"], 100 * v["ci95"][0], 100 * v["ci95"][1], colfn(cfg),
                          CFG_MARK.get(cfg.split("@")[0], "o"), v["eps_pass"]))
    return items


def cfg_label(cfg):
    if cfg.startswith("E1s@fp8k0"):
        return "E1-K0"
    if cfg.startswith("E1s@int8"):
        return "E1-INT8"
    return cfg


def f05_endpoints_quality():
    begin("F05_endpoints_quality")
    A = rp(*P, "analysis")
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 6.4))
    # FP8 tier
    for c, m in enumerate(MODELS):
        q = load_json(f"{A}/fp8/quality_{m}_step1.json")
        items = _q_items(q, ["E1", "E1s", "E1sr"], lambda cfg: FMT["fp8"])
        if m == "qwen":
            q2 = load_json(f"{A}/fp8/quality_qwen_step2b.json")
            items += _q_items(q2, ["E1s@fp8k0_head", "E1s@int8_token"],
                              lambda cfg: FMT["fp8k0"] if "fp8k0" in cfg else FMT["int8"])
        xl = (-3.5, 2.5) if m == "llama" else (-38, 4)
        _forest(axes[0, c], items, xlim=xl)
        axes[0, c].set_ylim(-1.4, len(items) - 0.5)
        axes[0, c].set_title(f"FP8 tier: {MLAB[m]} ({FMTNAME[m]['fp8']})")
    # INT4 tier
    for c, m in enumerate(MODELS):
        q = load_json(f"{A}/int4/quality_{m}_int4.json")
        items = _q_items(q, ["E1", "E1s", "E1sr"], lambda cfg: FMT["int4"])
        _forest(axes[1, c], items, xlim=(-3.5, 2.5))
        axes[1, c].set_ylim(-1.4, len(items) - 0.5)
        axes[1, c].set_title(f"INT4 tier: {MLAB[m]} (int4_kivi)")
    for ax in axes.flat:
        ax.set_xlabel("accuracy difference vs E0 (pp, 95% CI)")
    axes[0, 1].set_xlabel("accuracy difference vs E0 (pp, 95% CI)  [wider x range]")
    for ax in axes.flat:
        ax.text(-2.0, -0.9, " eps", color=OI["verm"], fontsize=6.2, ha="left", va="center")
    q0 = load_json(f"{A}/fp8/quality_llama_step1.json")
    fig.text(0.01, 0.003, "filled = passes eps (CI lower bound > -2 pp and every task/turn >= -5 pp); hollow = fails. "
             f"Main tasks T2-T5; {q0['n_items']} items / {q0['configs']['E1']['A']['n_rows']} rows per cell.",
             fontsize=6.3)
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    return save(fig, "F05_endpoints_quality")


# --------------------------------------------------------------------------------------------- F06
def f06_step2_K2():
    begin("F06_step2_K2")
    A = rp(*P, "analysis")
    st2 = load_json(f"{A}/int4/step2_llama_int4.json")
    summ = read_csv(rp(*P, "speed_llama", "summary_int4.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.3), gridspec_kw=dict(width_ratios=[1.0, 1.25]))
    ax = axes[0]
    cfgs = ["F25", "M25", "B25"]
    for k, st in enumerate(("idle", "busy")):
        for i, cfg in enumerate(cfgs):
            c = next(x for x in st2["cond1"] if x["config"] == cfg and x["gpu_state"] == st)
            d, lo, hi = (100 * c["diff_vs_R25"], 100 * c["ci95"][0], 100 * c["ci95"][1])
            x = i + (k - 0.5) * 0.25
            ax.errorbar([x], [d], yerr=[[d - lo], [hi - d]], fmt=CFG_MARK[cfg], ms=3.8, color=FMT["int4"],
                        mfc=FMT["int4"] if st == "idle" else "white", elinewidth=0.8, lw=0,
                        label=f"GPU {st}" if i == 0 else None)
    for yv in (2, -2):
        refline(ax, yv, color=OI["verm"], ls="--")
    refline(ax, 0, color="#333333", ls="-", lw=0.6)
    ax.text(2.45, 2.15, "+/-2 pp gap required", fontsize=6.2, color=OI["verm"], ha="right", va="bottom")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["F25 (front)", "M25 (middle)", "B25 (back)"])
    ax.set_ylim(-3, 3)
    ax.set_ylabel("accuracy vs R25 (pp, 95% CI)")
    ax.set_title("Quality: placement vs random (reading B, b32)")
    ax.legend(loc="upper left", fontsize=6.3, ncol=2)
    panel_label(ax, "a")
    ax = axes[1]
    cfgs2 = ["E0", "E1", "E1s", "E1sr", "F25", "M25", "B25", "R25"]
    for k, st in enumerate(("idle", "busy")):
        ys = []
        for cfg in cfgs2:
            r = next(x for x in summ if x["length"] == "32768" and x["bw_name"] == "b32" and x["gpu_state"] == st
                     and x["config"] == cfg and x["mode"] == "cake")
            ys.append(float(r["median_t_restore"]))
        xs = np.arange(len(cfgs2)) + (k - 0.5) * 0.36
        cols = [FMT["bf16"] if c == "E0" else FMT["int4"] for c in cfgs2]
        ax.bar(xs, ys, width=0.34, color=[c if st == "idle" else "white" for c in cols], edgecolor=cols, lw=0.7,
               label=f"GPU {st}")
    ax.set_xticks(range(len(cfgs2)))
    ax.set_xticklabels(cfgs2)
    ax.set_ylabel("restore time, median (s)")
    ax.set_title("Speed: b32 tier, Cake (INT4-KIVI configs)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#777777", edgecolor="#777777", label="GPU idle"),
                       Patch(facecolor="white", edgecolor="#777777", label="GPU busy")], loc="upper right", fontsize=6.3)
    panel_label(ax, "b")
    fig.tight_layout()
    return save(fig, "F06_step2_K2")


# --------------------------------------------------------------------------------------------- F07
def f07_quality_latency():
    """Advisor's 'quality vs latency' plot (Figure A) for Llama, reading B, b32."""
    begin("F07_quality_latency")
    A = rp(*P, "analysis")
    q8 = load_json(f"{A}/fp8/quality_llama_step1.json")["configs"]
    q4 = load_json(f"{A}/int4/quality_llama_int4.json")["configs"]
    q4m = load_json(f"{A}/int4/quality_llama_step2.json")["configs"]
    s8 = read_csv(rp(*P, "speed_llama", "summary_fp8.csv"))
    s4 = read_csv(rp(*P, "speed_llama", "summary_int4.csv"))

    def t(summ, cfg, st):
        r = next(x for x in summ if x["length"] == "32768" and x["bw_name"] == "b32" and x["gpu_state"] == st
                 and x["config"] == cfg and x["mode"] == "cake")
        return float(r["median_t_restore"])

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5), sharey=True)
    for ax, st in zip(axes, ("idle", "busy")):
        cond = f"B-{st}"
        pts = [("E0", t(s8, "E0", st), 0, 0, 0, FMT["bf16"], "o")]
        for cfg in ("E1", "E1s", "E1sr"):
            v = q8[cfg][cond]
            pts.append((f"FP8 {cfg}", t(s8, cfg, st), 100 * v["diff"], 100 * v["ci95"][0], 100 * v["ci95"][1],
                        FMT["fp8"], CFG_MARK[cfg]))
        for cfg in ("E1", "E1s", "E1sr"):
            v = q4[cfg][cond]
            pts.append((f"INT4 {cfg}", t(s4, cfg, st), 100 * v["diff"], 100 * v["ci95"][0], 100 * v["ci95"][1],
                        FMT["int4"], CFG_MARK[cfg]))
        for cfg in ("F25", "M25", "B25", "R25"):
            v = q4m[cfg][cond]
            pts.append((f"INT4 {cfg}", t(s4, cfg, st), 100 * v["diff"], 100 * v["ci95"][0], 100 * v["ci95"][1],
                        FMT["int4"], CFG_MARK[cfg]))
        for lab, x, d, lo, hi, col, mk in pts:
            ax.errorbar([x], [d], yerr=[[d - lo], [hi - d]], fmt=mk, ms=3.8, color=col,
                        mfc=col if not lab.endswith("R25") else "white", elinewidth=0.7, lw=0)
        lab_pos = {"E0": ("E0", (4, 5)), "INT4 F25": ("F25", (3, -11)), "INT4 E1": ("INT4 E1 = E1s", (5, 7)),
                   "INT4 E1sr": ("E1sr", (5, -11)), "FP8 E1": ("FP8 E1 = E1s", (5, 9)),
                   "FP8 E1sr": ("FP8 E1sr", (5, -12)), "INT4 M25": ("M25", (-4, -13)), "INT4 B25": ("B25", (-17, 6)),
                   "INT4 R25": ("R25", (5, 6))}
        if st == "busy":
            lab_pos.update({"INT4 M25": ("M25", (4, 6)), "INT4 R25": ("R25", (5, -11))})
        for lab, x, d, *_ in pts:
            if lab in lab_pos:
                txt, off = lab_pos[lab]
                ax.annotate(txt, (x, d), xytext=off, textcoords="offset points", fontsize=6.0, color="#222222")
        refline(ax, -2, color=OI["verm"], ls="--")
        refline(ax, 0, color="#333333", ls="-", lw=0.5)
        ax.set_title(f"b32 tier, GPU {st} (reading B)")
        ax.set_xlabel("restore time, median of 30 restores (s)")
        ax.set_ylim(-3, 2)
    axes[0].set_ylabel("accuracy vs E0 (pp, 95% CI)")
    axes[0].text(axes[0].get_xlim()[0] + 0.02, -1.9, "eps = -2 pp", fontsize=6.2, color=OI["verm"], va="bottom")
    handles = [plt.Line2D([], [], marker="o", ls="", color=FMT["bf16"], ms=4, label="BF16 E0"),
               plt.Line2D([], [], marker="o", ls="", color=FMT["fp8"], ms=4, label="FP8 (fp8_token)"),
               plt.Line2D([], [], marker="o", ls="", color=FMT["int4"], ms=4, label="INT4-KIVI"),
               plt.Line2D([], [], marker="s", ls="", color="#555555", ms=4, label="E1s (sink BF16)"),
               plt.Line2D([], [], marker="D", ls="", color="#555555", ms=4, label="E1sr / M25"),
               plt.Line2D([], [], marker=">", ls="", color="#555555", ms=4, label="F25"),
               plt.Line2D([], [], marker="<", ls="", color="#555555", ms=4, label="B25"),
               plt.Line2D([], [], marker="o", ls="", mfc="white", color="#555555", ms=4, label="R25")]
    fig.legend(handles=handles, loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.06), fontsize=6.2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return save(fig, "F07_quality_latency")


# --------------------------------------------------------------------------------------------- F08
def f08_format_mechanism():
    """Why Qwen's FP8 fails: constant K-channel offsets; centered K error; format-selection KL."""
    begin("F08_format_mechanism")
    K = rp("m10_pilot_mi300x", "kv_stats")
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.25), gridspec_kw=dict(width_ratios=[1, 1.35, 1.1]))
    ax = axes[0]
    for i, m in enumerate(MODELS):
        rows = read_csv(f"{K}/kv_head_{m}.csv")
        v = np.array([float(r["max_absmean_K"]) for r in rows])
        jitter = (np.random.default_rng(0).random(len(v)) - 0.5) * 0.35
        ax.scatter(np.full(len(v), i) + jitter, v, s=4, color=["#444444", OI["purple"]][i], alpha=0.7, lw=0)
    ax.set_yscale("log")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Llama", "Qwen"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("max |mean K| over channels\n(per layer x KV head)")
    ax.set_title("Constant K-channel offset")
    panel_label(ax, "a")
    ax = axes[1]
    fmts = ["fp8_head", "fp8_channel", "fp8_token", "fp8k0_head", "int8_token", "int4_token", "int4_kivi"]
    w = 0.38
    for i, m in enumerate(MODELS):
        s = load_json(f"{K}/summary_kv_{m}.json")["codec_relerr_mean_over_prompts"]
        rel = [100 * s[f]["K"] if f in s else np.nan for f in fmts]
        cen = [100 * s[f]["K_centered"] if f in s else np.nan for f in fmts]
        x = np.arange(len(fmts)) + (i - 0.5) * w
        col = ["#444444", OI["purple"]][i]
        ax.bar(x, cen, width=w, color=col, alpha=0.85, label=f"{'Llama' if m == 'llama' else 'Qwen'}: centered K error")
        ax.scatter(x, rel, marker="_", s=30, color="white", zorder=3, lw=1.2)
        ax.scatter(x, rel, marker="_", s=22, color="#E69F00", zorder=4, lw=1.0,
                   label="plain relative-L2 K error" if i == 1 else None)
    ax.set_yscale("log")
    ax.set_xticks(range(len(fmts)))
    short = {"fp8_head": "FP8\nhead", "fp8_channel": "FP8\nchan", "fp8_token": "FP8\ntoken", "fp8k0_head": "FP8-K0\nhead",
             "int8_token": "INT8\ntoken", "int4_token": "INT4\ntoken", "int4_kivi": "INT4\nKIVI"}
    ax.set_xticklabels([short[f] for f in fmts], fontsize=6.0)
    ax.set_ylabel("K reconstruction error (%)")
    ax.set_title("Codec error on real K (5 x 32K prompts)")
    ax.legend(fontsize=5.8, loc="upper right")
    panel_label(ax, "b")
    ax = axes[2]
    fc = load_json(rp(*P, "fmt_select", "fmt_choice.json"))
    order = ["fp8_head", "fp8_channel", "fp8_token", "int4_token", "int4_kivi"]
    for i, m in enumerate(MODELS):
        kl = [fc["mean_kl"][m][f] for f in order]
        x = np.arange(len(order)) + (i - 0.5) * w
        col = ["#444444", OI["purple"]][i]
        ax.bar(x, kl, width=w, color=col, alpha=0.85, label="Llama" if m == "llama" else "Qwen")
        for f in fc["choice"][m].values():
            j = order.index(f)
            ax.text(j + (i - 0.5) * w, fc["mean_kl"][m][f] * 1.4, "*", ha="center", fontsize=8, color=col)
    ax.set_yscale("log")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([short[f] for f in order], fontsize=6.0)
    ax.set_ylabel("mean KL(E1 || E0), 20-item pilot")
    ax.set_title("Format selection (* = chosen)")
    panel_label(ax, "c")
    fig.tight_layout()
    return save(fig, "F08_format_mechanism")


# --------------------------------------------------------------------------------------------- F09
def f09_int4_sensitivity():
    begin("F09_int4_sensitivity")
    s = load_json(rp(*P, "analysis", "int4", "sensitivity.json"))
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.2))
    key_col = {"INT8": FMT["int8"], "FP8 K": FMT["fp8k0"], "FP8": FMT["fp8"], "INT4": FMT["int4"]}

    def col(name):
        for k, c in key_col.items():
            if name.startswith(k):
                return c
        return "#777777"

    def short(name):
        return (name.replace("（每 token）", " (token)").replace("（主線）", " (main)").replace("K 先減平均", "K0")
                .replace("FP8 K0", "FP8-K0"))
    ylo, yhi = -0.05, 0.02
    for c, m in enumerate(MODELS):
        f = s[m]["formats"]
        names = list(f.keys())
        x = np.arange(len(names)) + (c - 0.5) * 0.38
        axes[0].bar(x, [100 * f[n]["same_as_e0"] for n in names], width=0.36, color=[col(n) for n in names],
                    edgecolor=[col(n) for n in names], alpha=1.0 if m == "llama" else 0.45, lw=0.6)
        for j, n in enumerate(names):
            mu, hw = f[n]["dlogp_task"]["T5"]          # [mean, 95% half-width] (m10_int4_sensitivity.py)
            xx = j + (c - 0.5) * 0.3
            if mu - hw < ylo:
                axes[1].annotate(f"off-scale: {mu:+.2f}", (xx, ylo), xytext=(xx, ylo + 0.012), fontsize=6.0,
                                 ha="center", color=col(n), arrowprops=dict(arrowstyle="->", color=col(n), lw=0.7))
                continue
            axes[1].errorbar([xx], [mu], yerr=[[hw]], fmt="o" if m == "llama" else "s", ms=3.5, color=col(n),
                             mfc=col(n) if m == "llama" else "white", elinewidth=0.8, lw=0)
    for ax in axes:
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([short(n) for n in names], fontsize=6.3)
    axes[1].set_ylim(ylo, yhi)
    axes[0].set_ylabel("rows identical to E0 output (%)")
    axes[0].set_title("Dose-response: output agreement with BF16")
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(facecolor="#555555", label="Llama (left bar)"),
                            Patch(facecolor="#555555", alpha=0.45, label="Qwen (right bar)")], fontsize=6.2,
                   loc="lower left")
    axes[1].set_ylabel("T5 gold log-prob per token\nvs E0 (mean, 95% CI)")
    axes[1].set_title("Below the accuracy threshold (T5, reading A)")
    refline(axes[1], 0, color="#333333", ls="-", lw=0.5)
    axes[1].legend(handles=[plt.Line2D([], [], marker="o", ls="", color="#555555", label="Llama"),
                            plt.Line2D([], [], marker="s", ls="", mfc="white", color="#555555", label="Qwen")],
                   fontsize=6.2, loc="lower left")
    fig.tight_layout()
    return save(fig, "F09_int4_sensitivity")


ALL = [f01_concept_cake, f02_decision_flow, f03_k0_speed, f04_sources_read, f05_endpoints_quality, f06_step2_K2, f07_quality_latency,
       f08_format_mechanism, f09_int4_sensitivity]
