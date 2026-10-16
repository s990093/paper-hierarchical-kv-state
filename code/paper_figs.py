"""Figures for the rewritten main.tex (2026-10-16).

Reads only files already in results/ (each traces to a run_id) and writes
  figures/*.pdf                         vector figures included by main.tex
  results/paper_20261016/fig_data.csv   every plotted value, with run_id, ts, src, src_run_id

No GPU, no new measurement: this script only re-plots existing results.
Usage: python3 code/paper_figs.py
"""
import csv
import datetime as dt
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
OUT = os.path.join(RES, "paper_20261016")
RUN_ID = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S") + "-paper-figs"
TS = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.6,
})
COL_W = 3.45  # IEEE column width, inches

rows = []  # fig_data.csv


def rec(fig, series, x, y, src, src_run_id, extra=""):
    rows.append(dict(run_id=RUN_ID, ts=TS, fig=fig, series=series, x=x, y=y,
                     extra=extra, src=src, src_run_id=src_run_id))


# ---------------------------------------------------------------- Fig: Cake band
def fig_cake_band():
    src = "results/m7_write_policy_mi300x/summary_a0_speedup.csv"
    data = list(csv.DictReader(open(os.path.join(ROOT, src))))
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    marks = {"4096": "o", "8192": "s", "16384": "^", "32768": "D"}
    for L in ["4096", "8192", "16384", "32768"]:
        pts = sorted((float(r["GiBps"]), min(float(r["speedup_vs_compute"]),
                                            float(r["speedup_vs_load"])))
                     for r in data if r["L"] == L)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker=marks[L], ms=3.2, lw=1.0,
                label=f"{int(L) // 1024}K tokens")
        for x, y in pts:
            rec("cake_band", f"L={L}", x, y, src, "20261008-133544-m7-a0")
    ax.axhline(1.0, color="0.4", lw=0.6, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("Tier read bandwidth (GiB/s, log scale)")
    ax.set_ylabel("Cake speedup over the\nfaster single path")
    ax.set_xticks([0.33, 0.8, 2.9, 7, 11.6])
    ax.set_xticklabels(["0.33", "0.8", "2.9", "7", "11.6"])
    ax.legend(ncol=2, frameon=False, loc="upper right")
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, "cake_band.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig: deferral test scatter
def fig_deferral():
    src = "results/m9_followup/advisor_fig_data.csv"
    data = [r for r in csv.DictReader(open(os.path.join(ROOT, src))) if r["fig"] == "2"]
    names = {"summary_b.csv": "Phase 1 (15 configs)",
             "b2_verdict.csv": "Ablation (16 comparisons)",
             "r1_gpu_verdict.csv": "Simulator survivor (5 seeds)"}
    marks = {"summary_b.csv": "o", "b2_verdict.csv": "s", "r1_gpu_verdict.csv": "^"}
    lo = -45.0
    fig, ax = plt.subplots(figsize=(COL_W, 2.25))
    ax.add_patch(plt.Rectangle((5, 5), 40, 60, color="tab:green", alpha=0.10, lw=0))
    ax.text(6.0, 44, "write-time wins\n(both $\\geq$5%):\nempty", fontsize=6.5,
            color="tab:green", va="top")
    for s in names:
        pts = [(float(r["x"]), float(r["value"]), r["src_run_id"]) for r in data if r["src"] == s]
        xs = [max(x, lo) for x, _, _ in pts]
        ys = [y for _, y, _ in pts]
        clipped = sum(1 for x, _, _ in pts if x < lo)
        ax.scatter(xs, ys, s=12, marker=marks[s], label=names[s] +
                   (f", {clipped} at $<${int(lo)}%" if clipped else ""),
                   edgecolor="k", linewidth=0.3, zorder=3)
        for x, y, rid in pts:
            rec("deferral", names[s], x, y, src, rid)
    ax.axvline(5, color="0.4", lw=0.6, ls="--")
    ax.axhline(5, color="0.4", lw=0.6, ls="--")
    ax.axvline(0, color="0.7", lw=0.4)
    ax.axhline(0, color="0.7", lw=0.4)
    ax.set_xlim(lo - 3, 25)
    ax.set_ylim(-35, 55)
    ax.set_xlabel("Gain over best position-blind policy (%)")
    ax.set_ylabel("Gain over deferred twin (%)")
    ax.legend(frameon=False, loc="upper left", fontsize=6.3, handletextpad=0.2)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, "deferral.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- helpers for m10 tables
A = os.path.join(RES, "m10_precision_mi300x", "analysis")


def md_speed_table(path):
    """Second table in tables_speed_<m>.md: restore medians per config."""
    txt = open(path, encoding="utf-8").read()
    out = {}
    for line in txt.splitlines():
        m = re.match(r"^\| (b16|b32|b48|cpu) \| (閒|忙) \|(.*)\|\s*$", line)
        if not m:
            continue
        cells = [c.strip() for c in m.group(3).split("|")]
        if len(cells) != 11:
            continue
        keys = ["E0", "E1", "E1s", "E1sr", "F25", "M25", "B25", "R25",
                "compute", "load_E0", "load_E1"]
        st = "idle" if m.group(2) == "閒" else "busy"
        out[(m.group(1), st)] = dict(zip(keys, map(float, cells)))
    return out


def md_read_table(path):
    """Read-accounting table: 'read/stored (meet)' per config."""
    txt = open(path, encoding="utf-8").read()
    out = {}
    for line in txt.splitlines():
        m = re.match(r"^\| (b16|b32|b48|cpu) \| (閒|忙) \|(.*)\|\s*$", line)
        if not m or "／" not in line:
            continue
        cells = [c.strip() for c in m.group(3).split("|")]
        if len(cells) != 8:  # skip the K0 table, whose meet column also contains '／'
            continue
        keys = ["E0", "E1", "E1s", "E1sr", "F25", "M25", "B25", "R25"]
        st = "idle" if m.group(2) == "閒" else "busy"
        d = {}
        for k, c in zip(keys, cells):
            mm = re.match(r"([\d.]+)／([\d.]+)（([\d.]+)）", c)
            d[k] = (float(mm.group(1)), float(mm.group(2)), float(mm.group(3)))
        out[(m.group(1), st)] = d
    return out


# ---------------------------------------------------------------- Fig: read accounting heatmap
def fig_reads():
    src = "results/m10_precision_mi300x/analysis/fp8/tables_speed_llama.md"
    rid = "20261015-161226-m10-grid-fp8-llama"
    t = md_read_table(os.path.join(ROOT, src))
    tiers = [("b16", "5.27 GiB/s"), ("b32", "1.47 GiB/s"), ("b48", "0.42 GiB/s"),
             ("cpu", "CPU (real H2D)")]
    cfgs = ["F25", "M25", "B25", "R25", "E1"]
    cols = [(b, s) for b, _ in tiers for s in ("idle", "busy")]
    M = np.zeros((len(cfgs), len(cols)))
    lab = [["" for _ in cols] for _ in cfgs]
    for j, (b, s) in enumerate(cols):
        for i, c in enumerate(cfgs):
            rd, stored, meet = t[(b, s)][c]
            M[i, j] = rd / stored
            lab[i][j] = f"{rd:g}" if c != "E1" else f"{rd:g}"
            rec("reads", c, f"{b}-{s}", rd, src, rid, extra=f"stored={stored:g};meet={meet:g}")
    fig, ax = plt.subplots(figsize=(COL_W, 1.75))
    im = ax.imshow(M, cmap="Greys", vmin=0, vmax=1.25, aspect="auto")
    for i in range(len(cfgs)):
        for j in range(len(cols)):
            ax.text(j, i, lab[i][j], ha="center", va="center", fontsize=6.5,
                    color="white" if M[i, j] > 0.75 else "black")
    ax.set_yticks(range(len(cfgs)))
    ax.set_yticklabels(["Front (16)", "Mid (16)", "Back (16)", "Rand (16)",
                        "All (64)"])
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["idle", "busy"] * len(tiers), fontsize=6)
    for k, (_, name) in enumerate(tiers):
        ax.text(2 * k + 0.5, -0.95, name, ha="center", va="bottom", fontsize=6.5)
        if k:
            ax.axvline(2 * k - 0.5, color="tab:red", lw=0.8)
    ax.tick_params(length=0)
    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(top=0.86)
    fig.savefig(os.path.join(FIG, "reads.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig: INT4 mixed placements (Llama)
def fig_int4_mixed():
    sp_src = "results/m10_precision_mi300x/analysis/int4/tables_speed_llama.md"
    sp_rid = "20261016-034902-m10-grid-int4-llama"
    sp = md_speed_table(os.path.join(ROOT, sp_src))
    q_e1_src = "results/m10_precision_mi300x/analysis/int4/quality_llama_int4.json"
    q_mx_src = "results/m10_precision_mi300x/analysis/int4/quality_llama_step2.json"
    qe1 = json.load(open(os.path.join(ROOT, q_e1_src)))
    qmx = json.load(open(os.path.join(ROOT, q_mx_src)))
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.9), sharey=True)
    marks = {"E0": ("k", "*"), "E1": ("tab:red", "o"), "F25": ("tab:blue", "<"),
             "M25": ("tab:purple", "s"), "B25": ("tab:orange", ">"), "R25": ("tab:green", "D")}
    names = {"E0": "BF16", "E1": "All", "F25": "Front", "M25": "Mid", "B25": "Back", "R25": "Rand"}
    for ax, st in zip(axes, ["idle", "busy"]):
        for c, (col, mk) in marks.items():
            x = sp[("b32", st)][c]
            if c == "E0":
                y, lo, hi, rids = 0.0, 0.0, 0.0, qe1["run_ids"]
            else:
                q = (qe1 if c == "E1" else qmx)
                cc = q["configs"][c]
                v = cc[_cond_key(cc, st)]
                y = 100 * v["diff"]
                lo, hi = 100 * v["ci95"][0], 100 * v["ci95"][1]
                rids = q["run_ids"]
            ax.errorbar([x], [y], yerr=[[y - lo], [hi - y]], fmt=mk, color=col, ms=4,
                        capsize=1.5, lw=0.7, label=names[c])
            rec("int4_mixed", f"{c}-{st}", x, y, f"{sp_src};{q_e1_src};{q_mx_src}",
                sp_rid + ";" + ";".join(rids[-7:]), extra=f"ci=[{lo:.2f},{hi:.2f}]")
        ax.axhspan(-2, -6, color="tab:red", alpha=0.08, lw=0)
        ax.axhline(-2, color="tab:red", lw=0.6, ls="--")
        ax.axhline(0, color="0.6", lw=0.4)
        ax.set_title(f"GPU {st}, 1.47 GiB/s tier", fontsize=7)
        ax.set_xlabel("Restore time (s)")
        ax.grid(alpha=0.25, lw=0.4)
    for ax in axes:
        ax.annotate("Front = BF16:\nno INT4 block read", xy=(0.99, 0), xytext=(0, -24),
                    textcoords="offset points", xycoords=("axes fraction", "data"),
                    ha="right", fontsize=5.8, color="tab:blue")
    axes[0].set_ylabel("Accuracy vs. BF16 (pp)")
    axes[0].set_ylim(-3, 2)
    axes[0].text(0.62, -2.6, "fails $\\varepsilon$ = 2 pp", fontsize=6, color="tab:red")
    axes[1].legend(frameon=False, fontsize=6, loc="lower right", ncol=2, handletextpad=0.1,
                   columnspacing=0.6)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, "int4_mixed.pdf"))
    plt.close(fig)


def _cond_key(cc, st):
    """Map GPU state to the read-B condition key used by m10_analyze."""
    want = {"idle": ["B-idle"], "busy": ["B-busy"]}[st]
    for k in cc:
        if k in want:
            return k
    raise KeyError(f"no read-B {st} key in {list(cc)}")


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    fig_cake_band()
    fig_deferral()
    fig_reads()
    fig_int4_mixed()
    with open(os.path.join(OUT, "fig_data.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(RUN_ID, len(rows), "rows")


if __name__ == "__main__":
    main()
