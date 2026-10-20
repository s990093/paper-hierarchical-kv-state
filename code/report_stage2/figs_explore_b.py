"""Exploratory restore-speed studies (report-only): dequant cost, restore round 2 (13), tiers / IO wall (12),
Cake gain vs load/compute ratio, harness vs vLLM, cross-GPU infra check.

Timing rule (CLAUDE.md Sec.3): a run's timings are used only if every gpu_guard JSON of that run is CLEAN and
no row is marked contaminated. Runs without any guard record are excluded (listed in the figure footnote).
"""
from __future__ import annotations

import glob
import math
import os
import statistics
from collections import defaultdict

import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

import common
from common import (DOUBLE, FMT, OI, SINGLE, TIER_COL, begin, load_json, panel_label, plt, read_csv, refline, rp,
                    save)

MODELS = ["llama", "qwen"]
MLAB = {"llama": "Llama-3.1-8B", "qwen": "Qwen2.5-7B-1M"}
TAGS = ["fp8", "int4"]
TAGCOL = {"fp8": FMT["fp8"], "int4": FMT["int4"], "bf16": FMT["bf16"]}
TAGLAB = {"fp8": "FP8", "int4": "INT4-KIVI", "bf16": "BF16"}
R2 = ("m10_restore2_mi300x",)
TIER_LABEL_SHORT = {"cpu": "CPU (real)", "b16": "b16", "b32": "b32", "b48": "b48"}


# ------------------------------------------------------------------------------------------ helpers
def guards_clean(d, rid):
    gs = glob.glob(os.path.join(d, f"gpu_guard_{rid}_*.json")) + glob.glob(os.path.join(d, f"gpu_guard_{rid}.json"))
    if not gs:
        return False
    return all(load_json(g).get("verdict") == "CLEAN" for g in gs)


def summary(d):
    """summary.csv of a speed dir -> (rows of its (single) run_id, run_id, usable)."""
    p = os.path.join(d, "summary.csv")
    rows = read_csv(p)
    rids = sorted({r["run_ids"] for r in rows})
    rid = rids[-1]
    rows = [r for r in rows if r["run_ids"] == rid]
    ok = guards_clean(d, rid) and all(r.get("contaminated_rows", "0") in ("0", "") for r in rows)
    return rows, rid, ok


def smed(rows, **kw):
    """Median restore time from a summary table (already a per-cell median)."""
    for r in rows:
        if all(str(r.get(k)) == str(v) for k, v in kw.items()):
            return float(r["median_t_restore"])
    return None


def bg_rate(d, rid):
    xs = [x["matmul_per_s"] for f in glob.glob(os.path.join(d, f"busy_{rid}_*.json")) for x in load_json(f)
          if x.get("state") == "busy" and "matmul_per_s" in x]
    return statistics.mean(xs) if xs else None


def plainlog(ax, axis="y"):
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(NullFormatter())


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main_summary(m, tag):
    if m == "qwen" and tag == "fp8":
        d = rp("m10_precision_mi300x", "speed_qwen_regrid")
        f = "summary_fp8.csv"
    else:
        d = rp("m10_precision_mi300x", f"speed_{m}")
        f = f"summary_{tag}.csv"
    rows = read_csv(os.path.join(d, f))
    rid = sorted({r["run_ids"] for r in rows if r["length"] == "32768"})[-1]
    rows = [r for r in rows if r["run_ids"] == rid]
    return rows, rid, guards_clean(d, rid)


# ------------------------------------------------------------------------------------------ F20
def f20_dequant_cost():
    begin("F20_dequant_cost")
    fig = plt.figure(figsize=(DOUBLE, 6.6))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1.05, 1.05], hspace=0.75, wspace=0.42)
    excluded = []
    # (a) per-block dequant, stage 40 'parts' (guarded), original torch vs fused Triton, idle / busy panels
    mainfp8 = {"llama": ["fp8_token", "fp8k0_token"], "qwen": ["fp8_head", "fp8k0_head"]}
    k = 0
    for m in MODELS:
        d = rp("m10_sweep_mi300x", "parts")
        rows = read_csv(os.path.join(d, f"parts_{m}.csv"))
        rid = rows[0]["run_id"]
        usable = guards_clean(d, rid) and all(r["contaminated_gpu"] == "False" for r in rows)
        fmts = mainfp8[m] + ["int8_token", "int4_token", "int4_kivi"]
        for st in ("idle", "busy"):
            ax = fig.add_subplot(gs[0, k])
            k += 1
            if not usable:
                excluded.append(rid)
                ax.text(0.5, 0.5, "excluded (guard)", transform=ax.transAxes, ha="center")
                continue
            for i, f in enumerate(fmts):
                col = FMT[common.fmt_family(f)]
                for j, cname in enumerate(("t_dequant_torch_ms", "t_dequant_fused_ms")):
                    v = [float(r[cname]) for r in rows if r["fmt"] == f and r["gpu_state"] == st and r[cname] != ""]
                    if not v:
                        continue
                    q1, q2, q3 = np.percentile(v, [25, 50, 75])
                    ax.errorbar([i + (j - 0.5) * 0.32], [q2], yerr=[[q2 - q1], [q3 - q2]], fmt="o", ms=3.4,
                                color=col, mfc="white" if j == 0 else col, elinewidth=0.6, lw=0)
            ax.set_yscale("log")
            plainlog(ax)
            ax.set_xticks(range(len(fmts)))
            ax.set_xticklabels([f.replace("_token", "").replace("_head", "").replace("_kivi", "-kivi")
                                for f in fmts], fontsize=5.8, rotation=40, ha="right")
            ax.set_xlim(-0.6, len(fmts) - 0.4)
            ax.set_title(f"{m.capitalize()}, GPU {st}", fontsize=7.0)
            if k == 1:
                ax.set_ylabel("dequant per block (ms)\nmedian, IQR over blocks")
                panel_label(ax, "a")
    hs = [plt.Line2D([], [], marker="o", mfc="white", color="#555555", ls="", label="original (step-by-step PyTorch)"),
          plt.Line2D([], [], marker="o", color="#555555", ls="", label="fused Triton kernel")]
    fig.legend(handles=hs, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.955), fontsize=6.3)
    # (b) real CPU tier, 32K, Cake: E0 / E1 original / E1 fused
    for c, m in enumerate(MODELS):
        ax = fig.add_subplot(gs[1, 2 * c:2 * c + 2])
        xt, xl = [], []
        for t_i, tag in enumerate(TAGS):
            mrows, mrid, mok = main_summary(m, tag)
            dq = rp("m10_sweep_mi300x", "speed_dqfast" + ("_int4" if tag == "int4" else "") + f"_{m}")
            frows, frid, fok = summary(dq)
            if not (mok and fok):
                excluded += [x for x, ok in ((mrid, mok), (frid, fok)) if not ok]
                continue
            for kk, st in enumerate(("idle", "busy")):
                x0 = t_i * 2 + kk
                vals = [smed(mrows, length=32768, bw_name="cpu", gpu_state=st, config="E0", mode="cake"),
                        smed(mrows, length=32768, bw_name="cpu", gpu_state=st, config="E1", mode="cake"),
                        smed(frows, length=32768, bw_name="cpu", gpu_state=st, config="E1", mode="cake")]
                cols = [FMT["bf16"], TAGCOL[tag], TAGCOL[tag]]
                faces = [FMT["bf16"], "white", TAGCOL[tag]]
                for j, (v, ec, fc) in enumerate(zip(vals, cols, faces)):
                    ax.bar(x0 + (j - 1) * 0.27, 1e3 * v, width=0.25, color=fc, edgecolor=ec, lw=0.7)
                xt.append(x0)
                xl.append(f"{TAGLAB[tag]}\nGPU {st}")
        ax.set_xticks(xt)
        ax.set_xticklabels(xl, fontsize=6.3)
        ax.set_ylabel("restore time, median (ms)")
        ax.set_title(f"{MLAB[m]}: real CPU tier, 32K, Cake", fontsize=7.2)
        ax.set_ylim(0, None)
        if c == 0:
            panel_label(ax, "b")
    hs = [plt.Rectangle((0, 0), 1, 1, facecolor=FMT["bf16"]),
          plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#555555"),
          plt.Rectangle((0, 0), 1, 1, facecolor="#555555")]
    fig.legend(hs, ["E0 (BF16)", "E1, original dequant (main grid, frozen)", "E1, fused dequant (stage 42)"],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.655), fontsize=6.2)
    # (c) 42c method finals: saving E0 - E1 on the CPU tier with paired 95% CI
    rep = load_json(rp("m10_sweep_mi300x", "dqsweep", "C", "report_C.json"))
    meths = ["S0", "S2k8", "S2k16", "S4", "S2k8+G4"]
    meths = [x for x in meths if x in rep["finalists"]]
    for c, (m, tag) in enumerate([(m, t) for m in MODELS for t in TAGS]):
        ax = fig.add_subplot(gs[2, c])
        for kk, st in enumerate(("idle", "busy")):
            for i, meth in enumerate(meths):
                cell = next((x for x in rep["cells"] if x["model"] == m and x["fmt"] == tag and x["method"] == meth
                             and x["bw"] == "cpu" and x["gpu_state"] == st), None)
                if cell is None or not cell["usable"]:
                    if cell is not None:
                        excluded.append(cell["run_id"])
                    continue
                d0 = 1e3 * (cell["median"]["E0"] - cell["median"]["E1"])
                lo, hi = (1e3 * x for x in cell["paired_diff_ci95_s"])
                ax.errorbar([i + (kk - 0.5) * 0.3], [d0], yerr=[[max(d0 - lo, 0)], [max(hi - d0, 0)]], fmt="o", ms=3.2,
                            color=TAGCOL[tag], mfc=TAGCOL[tag] if st == "idle" else "white", elinewidth=0.7, lw=0,
                            label=f"GPU {st}" if i == 0 else None)
        refline(ax, 0, color="#333333", ls="-", lw=0.6)
        ax.set_xticks(range(len(meths)))
        ax.set_xticklabels(meths, fontsize=5.8, rotation=40, ha="right")
        ax.set_title(f"{m.capitalize()} {TAGLAB[tag]}", fontsize=7.2)
        if c == 0:
            ax.set_ylabel("E1 saving vs E0 (ms)\npaired 95% CI")
            panel_label(ax, "c")
            ax.legend(fontsize=5.8, loc="lower left")
    fig.text(0.5, 0.0, "(c) 42c dequant-scheduling finals, real CPU tier, 32K, Cake: S0 per-block, S2k8/S2k16 = 8/16 blocks "
             "per kernel, S4 = all blocks once after loading,\nG4 = predicted meet (also speeds up E0, so its saving vs E0 "
             "looks smaller).  " +
             (f"Excluded (not CLEAN): {', '.join(sorted(set(excluded)))}" if excluded else "All runs gpu_guard CLEAN."),
             ha="center", fontsize=5.8)
    return save(fig, "F20_dequant_cost")


# ------------------------------------------------------------------------------------------ F21
Q4M = ["P0", "P1", "P2", "P3", "P4a", "P4b", "P5"]
Q4LAB = {"P0": "P0\nper-block", "P1": "P1\nio prio", "P2": "P2\n8-block", "P3": "P3\nS4 once", "P4a": "P4a\nCU 7/8",
         "P4b": "P4b\nCU 3/4", "P5": "P5\nP1+P2"}


def f21_busy_dequant():
    begin("F21_busy_dequant_Q4")
    fig = plt.figure(figsize=(DOUBLE, 4.9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.75, wspace=0.38)
    excluded = []
    bgr = defaultdict(list)
    for c, (m, tag) in enumerate([(m, t) for m in MODELS for t in TAGS]):
        ax = fig.add_subplot(gs[0, c])
        r0 = None
        for i, meth in enumerate(Q4M):
            d = rp(*R2, "q4", f"{m}_{tag}_{meth}")
            rows, rid, ok = summary(d)
            if not ok:
                excluded.append(rid)
                continue
            for j, mode in enumerate(("cake", "loadall")):
                v = smed(rows, bw_name="cpu", gpu_state="busy", config="E1", mode=mode)
                ax.plot([i + (j - 0.5) * 0.28], [1e3 * v], marker="o" if mode == "loadall" else "s", ms=3.6,
                        color=TAGCOL[tag], mfc=TAGCOL[tag] if mode == "loadall" else "white", ls="")
            if meth == "P0":
                e0c = smed(rows, bw_name="cpu", gpu_state="busy", config="E0", mode="cake")
                e0l = smed(rows, bw_name="cpu", gpu_state="busy", config="E0", mode="loadall")
                refline(ax, 1e3 * e0c, color="#333333", ls=":", lw=0.6)
                refline(ax, 1e3 * e0l, color="#333333", ls="-", lw=0.6)
                r0 = bg_rate(d, rid)
            r = bg_rate(d, rid)
            if r and r0:
                bgr[meth].append(r / r0)
        ax.set_xticks(range(len(Q4M)))
        ax.set_xticklabels(Q4M, fontsize=5.8, rotation=45)
        ax.set_ylim(0, None)
        ax.set_title(f"{m.capitalize()} {TAGLAB[tag]}", fontsize=7.2)
        if c == 0:
            ax.set_ylabel("E1 restore (ms)\nGPU busy, CPU tier, 32K")
            panel_label(ax, "a")
    hs = [plt.Line2D([], [], marker="s", mfc="white", color="#555555", ls="", label="E1 Cake"),
          plt.Line2D([], [], marker="o", color="#555555", ls="", label="E1 load-all"),
          plt.Line2D([], [], color="#333333", ls=":", lw=0.6, label="E0 Cake (P0)"),
          plt.Line2D([], [], color="#333333", ls="-", lw=0.6, label="E0 load-all (P0)")]
    fig.legend(handles=hs, fontsize=6.0, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    # (b) background throughput relative to P0
    ax = fig.add_subplot(gs[1, :2])
    for i, meth in enumerate(Q4M):
        v = bgr.get(meth, [])
        ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, s=10, color="#555555", zorder=3)
    refline(ax, 1.0, color="#333333", ls="-", lw=0.6)
    refline(ax, 0.95, color=OI["verm"], ls="--", lw=0.6)
    ax.text(len(Q4M) - 0.6, 0.955, "0.95 = pick-rule floor (13 Sec.3.2)", fontsize=5.8, color=OI["verm"], ha="right",
            va="bottom")
    ax.set_xticks(range(len(Q4M)))
    ax.set_xticklabels([Q4LAB[x] for x in Q4M], fontsize=5.9)
    ax.set_ylabel("background matmul/s\n(relative to P0)")
    ax.set_title("Who pays: background throughput", fontsize=7.2)
    panel_label(ax, "b")
    # (c) duty sweep, INT4
    lv = load_json(rp(*R2, "busy_levels_q4.json"))
    duties = [("busy1", lv["busy1"]), ("busy2", lv["busy2"]), ("busy3", lv["busy3"])]
    for c, m in enumerate(MODELS):
        ax = fig.add_subplot(gs[1, 2 + c])
        for meth, ls in (("P0", "--"), ("P3", "-")):
            d = rp(*R2, "q4duty", f"{m}_int4_{meth}")
            rows, rid, ok = summary(d)
            if not ok:
                excluded.append(rid)
                continue
            for mode, mk in (("cake", "s"), ("loadall", "o")):
                ys = [1e3 * smed(rows, bw_name="cpu", gpu_state=b, config="E1", mode=mode) for b, _ in duties]
                ax.plot([x for _, x in duties], ys, ls=ls, marker=mk, ms=3.3, color=FMT["int4"],
                        mfc=FMT["int4"] if mode == "loadall" else "white", lw=0.9,
                        label=f"{meth} {'S0' if meth == 'P0' else 'S4'} {'Cake' if mode == 'cake' else 'load-all'}")
        ax.set_xticks([x for _, x in duties])
        ax.set_xlabel("background duty")
        ax.set_ylim(0, None)
        ax.set_title(f"{m.capitalize()} INT4 vs duty", fontsize=7.2)
        if c == 0:
            ax.set_ylabel("E1 restore (ms)")
            panel_label(ax, "c")
            ax.legend(fontsize=5.5, loc="upper left")
    if excluded:
        fig.text(0.5, 0.0, f"Excluded (not CLEAN): {', '.join(sorted(set(excluded)))}", ha="center", fontsize=5.8)
    return save(fig, "F21_busy_dequant_Q4")


# ------------------------------------------------------------------------------------------ F22
def q2_ratios(rule):
    """Per-cell rule-median / best (best = min(Cake, load-all) on CPU, Cake on throttled tiers); mirrors
    m10_restore2.g5_ratios but reads through common.read_csv (source logging)."""
    sfx = "" if rule == "g5" else "_g5b"
    out, choose = {}, {}
    for m in MODELS:
        for tag in TAGS:
            for part in ("cpu", "slow"):
                d = rp(*R2, "q2", f"{m}_{tag}_{part}{sfx}")
                rows = read_csv(os.path.join(d, "restores.csv"))
                rid = max(r["run_id"] for r in rows)
                rows = [r for r in rows if r["run_id"] == rid]
                if not (guards_clean(d, rid) and all(r["contaminated_gpu"] == "False" for r in rows)):
                    continue
                cell = defaultdict(list)
                used = defaultdict(list)
                for r in rows:
                    cell[(r["mode"], r["bw_name"], r["gpu_state"], r["config"])].append(float(r["t_restore"]))
                    if r["mode"] == "cake":
                        used[(r["bw_name"], r["gpu_state"], r["config"])].append(r.get("meet_rule_used", ""))
                for (mode, bw, st, cf), v in list(cell.items()):
                    if mode != "cake" or cf not in ("E0", "E1", "R25"):
                        continue
                    g = statistics.median(v)
                    ref = [statistics.median(cell[(mm, bw, st, cf)]) for mm in ("claim", "loadall")
                           if (mm, bw, st, cf) in cell]
                    if ref:
                        k = f"{m}|{tag}|{bw}|{st}|{cf}"
                        out[k] = g / min(ref)
                        u = used[(bw, st, cf)]
                        choose[k] = sum(x == "none" for x in u) / len(u) if u else None
    return out, choose


def f22_meet_rule_g5():
    begin("F22_meet_rule_G5")
    pick = load_json(rp(*R2, "g5_pick.json"))
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.4), sharey=True)
    tiers = ["cpu", "b16", "b32", "b48"]
    res = {}
    for ax, rule in zip(axes, ("g5", "g5b")):
        r, ch = q2_ratios(rule)
        res[rule] = r
        for k, v in r.items():
            m, tag, bw, st, cf = k.split("|")
            x = tiers.index(bw) + {"E0": -0.22, "E1": 0.0, "R25": 0.22}[cf] + (0.06 if st == "busy" else -0.06)
            col = FMT["bf16"] if cf == "E0" else TAGCOL[tag]
            ax.scatter([x], [v], s=9, marker="o" if m == "llama" else "^", color=col if st == "idle" else "white",
                       edgecolors=col, lw=0.6, zorder=3)
            if ch.get(k) and ch[k] > 0.5 and bw != "cpu":
                ax.scatter([x], [v], s=34, marker="o", facecolors="none", edgecolors="#333333", lw=0.6, zorder=4)
        refline(ax, 1.0, color="#333333", ls="-", lw=0.6)
        refline(ax, 1.05, color="#999999", ls=":", lw=0.6)
        ax.set_xticks(range(len(tiers)))
        ax.set_xticklabels(["CPU (real H2D)", "b16", "b32", "b48"])
        gmv = pick["gm_g5"] if rule == "g5" else pick["gm_g5b"]
        common_k = sorted(set(r))
        ax.set_title(f"{'G5' if rule == 'g5' else 'G5b'}: geomean over {pick['n_common']} cells = {gmv:.3f}", fontsize=7.4)
        print(f"[check] {rule}: recomputed geomean {gm([r[k] for k in common_k]):.4f} over {len(common_k)} cells "
              f"(g5_pick.json {gmv:.4f}); max {max(r.values()):.3f} at {max(r, key=r.get)}")
        for t in tiers:
            vv = [v for k, v in r.items() if k.split("|")[2] == t]
            print(f"[check] {rule} tier {t}: geomean {gm(vv):.4f} n={len(vv)} min {min(vv):.3f} max {max(vv):.3f}; "
                  f"chose load-all in {sum(1 for k in r if k.split('|')[2] == t and (ch.get(k) or 0) > 0.5)} cells")
    axes[0].set_ylabel("rule median / per-cell best")
    hs = [plt.Line2D([], [], marker="o", ls="", color=FMT["bf16"], ms=3.5, label="E0"),
          plt.Line2D([], [], marker="o", ls="", color=FMT["fp8"], ms=3.5, label="E1/R25 FP8"),
          plt.Line2D([], [], marker="o", ls="", color=FMT["int4"], ms=3.5, label="E1/R25 INT4"),
          plt.Line2D([], [], marker="o", ls="", mfc="white", color="#555555", ms=3.5, label="GPU busy (hollow)"),
          plt.Line2D([], [], marker="^", ls="", color="#555555", ms=3.5, label="Qwen (triangle)"),
          plt.Line2D([], [], marker="o", ls="", mfc="none", color="#333333", ms=6, label="rule chose load-all (throttled tier)")]
    axes[1].legend(handles=hs, fontsize=5.8, loc="upper right")
    fig.text(0.5, -0.03, "best = min(Cake, load-all) on the CPU tier, Cake on throttled tiers (13 Sec.8c). "
             "Within each tier: E0 left, E1 middle, R25 right; ringed = throttled-tier cells where the rule "
             "chose load-all.", ha="center", fontsize=5.9)
    fig.tight_layout()
    return save(fig, "F22_meet_rule_G5")


# ------------------------------------------------------------------------------------------ F23
def f23_fused_attention_control():
    begin("F23_fused_attention_Q5")
    fig = plt.figure(figsize=(DOUBLE, 4.3))
    gsp = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.55, wspace=0.25)
    axes = [fig.add_subplot(gsp[0, 0]), fig.add_subplot(gsp[0, 1]), fig.add_subplot(gsp[1, :])]
    d = rp(*R2, "q5", "attn")
    for c, m in enumerate(MODELS):
        ax = axes[c]
        rows = read_csv(os.path.join(d, f"attn_{m}.csv"))
        rid = rows[0]["run_id"]
        if not guards_clean(d, rid):
            ax.text(0.5, 0.5, f"{rid}: not CLEAN, excluded", transform=ax.transAxes, ha="center")
            continue
        for mode, col in (("bf16", FMT["bf16"]), ("fp8", FMT["fp8"]), ("int4_kivi", FMT["int4"])):
            for st in ("idle", "busy"):
                pts = sorted((int(r["length"]), 1e3 * float(r["median_s"])) for r in rows
                             if r["mode"] == mode and r["queries"] == "1" and r["gpu_state"] == st
                             and r["contaminated_gpu"] == "False" and r["median_s"] not in ("", "ERROR"))
                if pts:
                    ax.plot([p[0] / 1024 for p in pts], [p[1] for p in pts], marker="o", ms=3, color=col,
                            mfc=col if st == "idle" else "white", ls="-" if st == "idle" else "--", lw=0.9,
                            label=f"{ {'bf16': 'BF16', 'fp8': 'FP8', 'int4_kivi': 'INT4-KIVI'}[mode]} {st}")
        ax.set_xscale("log", base=2)
        ax.set_xticks([32, 64, 126])
        ax.set_xticklabels(["32K", "64K", "128K"])
        ax.set_ylim(0, None)
        ax.set_xlabel("context (tokens)")
        ax.set_title(f"{MLAB[m]}: one decode step (our Triton kernel)", fontsize=6.5)
        if c == 0:
            ax.set_ylabel("attention per decode step,\nall layers (ms)")
            panel_label(ax, "a")
            ax.legend(fontsize=5.5, loc="upper left", ncol=1)
    # (c) restore: no-dequant lower bound vs S4+load-all vs P0 load-all (CPU tier, 32K, E1)
    ax = axes[2]
    xt, xl = [], []
    k = 0
    for m in MODELS:
        for tag in TAGS:
            for st in ("idle", "busy"):
                vals = []
                for dd, mode in ((rp(*R2, "q5", f"nodq_{m}_{tag}"), "cake"), (rp(*R2, "q4", f"{m}_{tag}_P3"), "loadall"),
                                 (rp(*R2, "q4", f"{m}_{tag}_P0"), "loadall")):
                    rows, rid, ok = summary(dd)
                    vals.append(smed(rows, bw_name="cpu", gpu_state=st, config="E1", mode=mode) if ok else None)
                for j, (v, fc) in enumerate(zip(vals, ("white", TAGCOL[tag], "#d9d9d9"))):
                    if v is not None:
                        ax.bar(k + (j - 1) * 0.27, 1e3 * v, width=0.25, color=fc, edgecolor=TAGCOL[tag], lw=0.7,
                               hatch="////" if j == 0 else None)
                xt.append(k)
                xl.append(f"{m.capitalize()} {TAGLAB[tag].split('-')[0]}\nGPU {st}")
                k += 1
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=6.0)
    ax.set_ylabel("E1 restore (ms)\nreal CPU tier, 32K")
    ax.set_title("Restore time: move-only lower bound (dequant inside attention) vs S4 + load-all", fontsize=7.0)
    hs = [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#555555", hatch="////"),
          plt.Rectangle((0, 0), 1, 1, facecolor="#555555"),
          plt.Rectangle((0, 0), 1, 1, facecolor="#d9d9d9", edgecolor="#555555")]
    ax.legend(hs, ["no dequant (move only)", "S4 dequant once + load-all", "per-block dequant + load-all (P0)"],
              fontsize=5.8, loc="upper right")
    panel_label(ax, "b")
    fig.text(0.5, 0.0, "vLLM 0.28 FP8-KV decode runs (q5/vllm_kvdecode) have no gpu_guard record and are not plotted.",
             ha="center", fontsize=5.9)
    return save(fig, "F23_fused_attention_Q5")


# ------------------------------------------------------------------------------------------ F24
def f24_longctx_restore_methods():
    begin("F24_longctx_restore_Q1")
    L = [32768, 65536, 129024]
    meths = [  # (label, dir key, mode, style)
        ("Cake, per-block dequant (P0)", "P0", "cake", dict(ls="--", marker="s", mfc="white")),
        ("load-all, per-block dequant (P0)", "P0", "loadall", dict(ls="--", marker="o")),
        ("load-all, dequant once (S4/P3)", "P3", "loadall", dict(ls="-", marker="o")),
        ("G5b rule (P0 dequant)", "G5b", "cake", dict(ls="-.", marker="D", mfc="white")),
        ("move only, no dequant", "nodq", "cake", dict(ls=":", marker="v")),
    ]

    def dir_for(m, tag, length, key):
        if length == 32768:
            return {"P0": rp(*R2, "q4", f"{m}_{tag}_P0"), "P3": rp(*R2, "q4", f"{m}_{tag}_P3"),
                    "G5b": rp(*R2, "q2", f"{m}_{tag}_cpu_g5b"), "nodq": rp(*R2, "q5", f"nodq_{m}_{tag}")}[key]
        return rp(*R2, "q1", f"{m}_{length}_{tag}_" + {"P0": "P0", "P3": "Q4_P3", "G5b": "G5b", "nodq": "nodq"}[key])

    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE, 3.9), sharex=True)
    excluded = []
    for c, (m, tag) in enumerate([(m, t) for m in MODELS for t in TAGS]):
        for r, st in enumerate(("idle", "busy")):
            ax = axes[r, c]
            for cfg, col_over in (("E0", FMT["bf16"]), ("E1", None)):
                for lab, key, mode, sty in meths:
                    if cfg == "E0" and key not in ("P0",):
                        continue
                    ys = []
                    for length in L:
                        rows, rid, ok = summary(dir_for(m, tag, length, key))
                        if not ok:
                            excluded.append(rid)
                            ys.append(np.nan)
                            continue
                        v = smed(rows, length=length, bw_name="cpu", gpu_state=st, config=cfg, mode=mode)
                        ys.append(np.nan if v is None else v)
                    col = col_over or TAGCOL[tag]
                    kw = dict(sty)
                    if kw.get("mfc") == "white":
                        pass
                    else:
                        kw["mfc"] = col
                    ax.plot([x / 1024 for x in L], ys, color=col, ms=3.0, lw=0.85,
                            label=(f"E0 (BF16) {'Cake' if mode == 'cake' else 'load-all'}" if cfg == "E0"
                                   else f"E1 {lab}"), **kw)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            plainlog(ax)
            ax.set_xticks([32, 64, 126])
            ax.set_xticklabels(["32K", "64K", "128K"])
            if r == 0:
                ax.set_title(f"{MLAB[m].split('-')[0]} {TAGLAB[tag]}", fontsize=7.2)
            if c == 0:
                ax.set_ylabel(f"GPU {st}\nrestore, CPU tier (s)")
            if r == 1:
                ax.set_xlabel("context")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06), fontsize=6.0)
    fig.text(0.5, -0.085, "Real CPU tier (pinned host memory, real H2D). 32K from 13-Q4/Q2/Q5 runs, 64K/128K from 13-Q1 (44d); "
             "128K = 129,024 tokens. Medians; color = stored format of E1.", ha="center", fontsize=5.8)
    if excluded:
        fig.text(0.5, -0.11, f"Excluded (not CLEAN): {', '.join(sorted(set(excluded)))}", ha="center", fontsize=5.8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return save(fig, "F24_longctx_restore_Q1")


# ------------------------------------------------------------------------------------------ F25
def tier_rows(m):
    d = rp("m10_tier_mi300x", m)
    gs = glob.glob(os.path.join(d, "gpu_guard_*.json"))
    ok = bool(gs) and all(load_json(g).get("verdict") == "CLEAN" for g in gs)
    rows = read_csv(rp("m10_tier_mi300x", f"summary_{m}.csv"))
    return rows, ok


def f25_tiers_iowall():
    begin("F25_tiers_iowall")
    iow = load_json(rp("m10_tier_mi300x", "iowall.json"))
    fig = plt.figure(figsize=(DOUBLE, 4.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1], hspace=0.55, wspace=0.42)
    for c, m in enumerate(MODELS):
        rows, ok = tier_rows(m)
        ax = fig.add_subplot(gs[0, 2 * c:2 * c + 2])
        if not ok:
            ax.text(0.5, 0.5, "guard not CLEAN: excluded", transform=ax.transAxes, ha="center")
            continue
        bwmap = {k: float(v) for k, v in iow[m]["slow_GiBps"].items()}
        bwmap["cpu"] = float(iow[m]["real_pinned_h2d_GiBps"])
        for prec, cfg_slow, cfg_cpu in (("bf16", "S-bf16", "E0"), ("int4", "S-int4", "E1")):
            for st in ("idle", "busy"):
                pts = []
                for r in rows:
                    if r["gpu_state"] != st:
                        continue
                    if (r["tier"] != "cpu" and r["config"] == cfg_slow) or (r["tier"] == "cpu" and r["config"] == cfg_cpu):
                        pts.append((bwmap[r["tier"]], float(r["median_t_restore"]), r["tier"]))
                pts.sort()
                col = TAGCOL[prec]
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3.2, lw=0.9, color=col,
                        mfc=col if st == "idle" else "white", ls="-" if st == "idle" else "--",
                        label=f"all {TAGLAB[prec]}, GPU {st}")
        for k, v in sorted(bwmap.items(), key=lambda x: x[1]):
            if k in ("nfs", "ssd", "cpu"):
                ax.axvline(v, color="#bbbbbb", lw=0.5, ls=":")
                ax.text(v, 1.02, {"nfs": "NFS-class", "ssd": "SSD-class", "cpu": "CPU real"}[k],
                        transform=ax.get_xaxis_transform(), fontsize=5.8, ha="center", color="#555555")
        be = iow[m]["A2"]["int4"]["breakeven"]
        txt = []
        for st, b in be.items():
            if b:
                ax.plot([b["GiBps"]], [0.0], marker="^", ms=5, color=FMT["int4"], mfc=FMT["int4"] if st == "idle" else "white",
                        transform=ax.get_xaxis_transform(), clip_on=False, zorder=5)
                txt.append(f"{st} {b['GiBps']:.1f}")
            else:
                txt.append(f"{st}: none up to the CPU tier")
        ax.text(0.98, 0.97, "INT4 break-even (GiB/s, triangles):\n" + "; ".join(txt), transform=ax.transAxes,
                fontsize=5.7, ha="right", va="top", color=FMT["int4"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        plainlog(ax)
        plainlog(ax, "x")
        ax.set_xlabel("tier bandwidth (GiB/s)")
        ax.set_title(f"{m.capitalize()}: whole 32K conversation on one tier", fontsize=7.2, pad=16)
        if c == 0:
            ax.set_ylabel("restore time, median (s)")
            ax.legend(fontsize=5.6, loc="lower left")
            panel_label(ax, "a", y=1.06)
    # (b) two-tier demotion: order x precision, at NFS and SSD class, GPU idle
    for c, m in enumerate(MODELS):
        rows, ok = tier_rows(m)
        if not ok:
            continue
        e0 = next(float(r["median_t_restore"]) for r in rows if r["tier"] == "cpu" and r["config"] == "E0"
                  and r["gpu_state"] == "idle")
        for j, tier in enumerate(("nfs", "ssd")):
            ax = fig.add_subplot(gs[1, 2 * c + j])
            for order, ls, mk in (("old", "-", "o"), ("new", "--", "s"), ("rnd", ":", "^")):
                for prec in ("bf16", "int4"):
                    pts = []
                    for frac in (25, 50, 75):
                        r = next((x for x in rows if x["tier"] == tier and x["gpu_state"] == "idle"
                                  and x["config"] == f"T{order}{frac}-{prec}"), None)
                        if r:
                            pts.append((frac, float(r["median_t_restore"])))
                    rall = next((x for x in rows if x["tier"] == tier and x["gpu_state"] == "idle"
                                 and x["config"] == f"S-{prec}"), None)
                    if rall:
                        pts.append((100, float(rall["median_t_restore"])))
                    pts = [(0, e0)] + pts
                    col = TAGCOL[prec]
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], ls=ls, marker=mk, ms=2.8, lw=0.8, color=col,
                            label=f"{ {'old': 'oldest first', 'new': 'newest first', 'rnd': 'random'}[order]}, "
                                  f"slow part {TAGLAB[prec].split('-')[0]}")
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_xlabel("% blocks demoted")
            ax.set_title(f"{m.capitalize()}: {'NFS' if tier == 'nfs' else 'SSD'}-class, idle", fontsize=7.0)
            if c == 0 and j == 0:
                ax.set_ylabel("restore (s)")
                panel_label(ax, "b")
            ax.set_ylim(0, None)
    h, l = fig.axes[2].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07), fontsize=5.8)
    fig.text(0.5, -0.125, "Stage 44 (12 plan): original per-block dequant, Cake, 2 launches x 5 reps.\n0% = whole conversation "
             "BF16 on the CPU tier (E0 of the same run); 100% = whole conversation on the slow tier.", ha="center",
             fontsize=5.8)
    return save(fig, "F25_tiers_iowall")


# ------------------------------------------------------------------------------------------ F26
def f26_cake_ratio():
    begin("F26_cake_ratio")
    import m10_cake_ratio as CR
    for _, _, f in CR.SRC:
        common._note(os.path.join(CR.P, f))
    cs = CR.cells()
    fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
    mk = {"cpu": "o", "b16": "s", "b32": "^", "b48": "D"}
    for c in cs:
        col = TAGCOL[c["fmt"]]
        ax.scatter([c["ratio"]], [c["gain"]], marker=mk[c["tier"]], s=13, color=col if c["gpu"] == "idle" else "white",
                   edgecolors=col, lw=0.7, zorder=3)
    refline(ax, 1.0, color="#333333", ls="-", lw=0.6)
    refline(ax, 2.0, color="#999999", ls=":", lw=0.6)
    ax.text(0.98, 2.03, "2x = each line does half", fontsize=5.6, color="#666666", va="bottom", ha="right",
            transform=ax.get_yaxis_transform())
    ax.text(0.98, 1.02, "Cake = best single mode", fontsize=5.6, color="#333333", va="bottom", ha="right",
            transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_xlabel("load-only / compute-only time ratio")
    ax.set_ylabel("Cake gain = min(C, L) / Cake")
    hs = [plt.Line2D([], [], marker=mk[t], ls="", color="#555555", ms=3.5, label=TIER_LABEL_SHORT[t]) for t in mk]
    hs += [plt.Line2D([], [], marker="o", ls="", color=TAGCOL[f], ms=3.5, label=TAGLAB[f]) for f in ("bf16", "fp8", "int4")]
    hs += [plt.Line2D([], [], marker="o", ls="", mfc="white", color="#555555", ms=3.5, label="GPU busy")]
    ax.legend(handles=hs, fontsize=5.6, ncol=2, loc="lower right")
    ax.set_title("Cake gain vs load/compute ratio (32K, main grids)", fontsize=7.4)
    return save(fig, "F26_cake_ratio")


# ------------------------------------------------------------------------------------------ F27
def f27_harness_vs_vllm():
    begin("F27_harness_vs_vllm")
    metas = {}
    for sub in ("vllm_ttft_guarded2", "vllm_ttft_guarded3"):
        mj = load_json(rp("m10_precision_mi300x", sub, "vllm_meta.json"))
        for k, v in mj.items():
            if all(g == "CLEAN" for g in v["guard"].values()):
                metas[k] = (v, sub)       # guarded3 (later) overrides guarded2 for llama|512
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.2), sharey=True)
    for ax, m in zip(axes, MODELS):
        for chunk, ls, mk in (("default", "-", "o"), ("512", "--", "s")):
            v, sub = metas[f"{m}|{chunk}"]
            pts = sorted((int(L), t) for L, t in v["median_ttft_s"].items())
            ax.plot([p[0] / 1024 for p in pts], [p[1] for p in pts], ls=ls, marker=mk, ms=3.3, color="#444444",
                    mfc="#444444" if chunk == "default" else "white", lw=0.9,
                    label=f"vLLM 0.28, chunk {v['max_num_batched_tokens']} ({v['run_id'][:15]})")
        # harness compute-only: M2 runs (14), 32K, all compute_only restores of the FP8 grid, GPU idle
        fm = {"llama": "fp8_token", "qwen": "fp8_head"}[m]
        d = rp("m10_match_mi300x", "m2", f"{m}_{fm}")
        rows = read_csv(os.path.join(d, "restores.csv"))
        rid = max(r["run_id"] for r in rows)
        ok = guards_clean(d, rid) and all(r["contaminated_gpu"] == "False" for r in rows if r["run_id"] == rid)
        v = [float(r["t_restore"]) for r in rows if r["run_id"] == rid and r["mode"] == "compute_only"
             and r["gpu_state"] == "idle"]
        if ok and v:
            h = statistics.median(v)
            dflt = metas[f"{m}|default"][0]["median_ttft_s"]["32768"]
            c512 = metas[f"{m}|512"][0]["median_ttft_s"]["32768"]
            ax.plot([32], [h], marker="*", ms=8, color=OI["verm"], ls="",
                    label=f"our harness, 512-token chunks (n={len(v)}):\n{h / dflt:.2f}x vLLM default, "
                          f"{h / c512:.2f}x vLLM 512-chunk")
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 16, 32])
        ax.set_xticklabels(["8K", "16K", "32K"])
        ax.set_xlabel("prompt length")
        ax.set_title(f"{MLAB[m]}: compute-only prefill, GPU idle", fontsize=7.2)
        ax.legend(fontsize=5.6, loc="upper left")
    axes[0].set_ylabel("time to first token / full recompute (s)")
    fig.tight_layout()
    return save(fig, "F27_harness_vs_vllm")


# ------------------------------------------------------------------------------------------ F28
def f28_xgpu_check():
    begin("F28_xgpu_check")
    v = load_json(rp("m10_xgpu_mi300x", "kfd", "verdict.json"))
    cells = [(k, c) for k, c in v["cells"].items() if isinstance(c, dict)]
    cells.sort(key=lambda kc: kc[1]["ratio"])
    fig, ax = plt.subplots(figsize=(SINGLE, 2.3))
    x = np.arange(len(cells))
    ax.scatter(x, [c["ratio"] for _, c in cells], s=10, color="#333333", zorder=3, label="B/A (card 0 busy / empty)")
    ax.scatter(x, [c["A2_over_A1"] for _, c in cells], s=10, facecolors="white", edgecolors="#888888", zorder=2,
               label="A2/A1 (same condition twice)")
    refline(ax, 1.08, color=OI["verm"], ls="--", lw=0.7)
    refline(ax, 1.02, color=OI["orange"], ls=":", lw=0.7)
    refline(ax, 1.0, color="#333333", ls="-", lw=0.5)
    ax.text(0, 1.083, "max limit 1.08", fontsize=5.8, color=OI["verm"], va="bottom")
    ax.text(len(cells) - 1, 1.023, "geomean limit 1.02", fontsize=5.8, color=OI["orange"], va="bottom", ha="right")
    worst = cells[-1][0]
    ax.annotate(worst.replace("|", " "), (x[-1], cells[-1][1]["ratio"]), xytext=(-105, -4), textcoords="offset points",
                fontsize=5.4, va="center")
    ax.set_xticks([])
    ax.set_xlabel(f"{v['n_cells']} timing cells (sorted)")
    ax.set_ylabel("restore-time ratio")
    ax.set_title(f"Cross-GPU interference: geomean {v['geomean_B_over_A']:.4f}, max {v['max_B_over_A']:.3f}"
                 f" -> {'pass' if v['pass_'] else 'fail'}", fontsize=6.8)
    ax.legend(fontsize=5.6, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    return save(fig, "F28_xgpu_check")


ALL = [f20_dequant_cost, f21_busy_dequant, f22_meet_rule_g5, f23_fused_attention_control, f24_longctx_restore_methods,
       f25_tiers_iowall, f26_cake_ratio, f27_harness_vs_vllm, f28_xgpu_check]
