#!/usr/bin/env python3
"""Regenerate every data figure of the ICPE 2027 paper version (docs/icpe_paper_20261030) from result files.

    bash -c 'source code/env_h200.sh >/dev/null 2>&1; python docs/icpe_paper_20261030/figs/make_figs.py [fig3 fig4 ...]'

Rules this script follows:
  * Every plotted value is read from a file under results/ (no constants typed in from documents).
  * For each figure it writes figs/<name>.sources.json: the exact input files (repo-relative path, sha256),
    the run_id values read from those files, the plotted values, and cross-checks against
    (a) other result files that hold the same number and (b) the values quoted in the docs / task brief.
    A cross-check that fails is reported (status "DIFF"), never patched: the figure shows what the primary file says.
  * fig8 (T4 bridge) is skipped with a message when results/icpe_h200/t4/t4_verdict.json does not exist.

Diagrams fig1/fig2 are not made here: they are FigureSpec JSON in figs/specs/, rendered with
.claude/skills/figure-spec/scripts/figure_renderer.py.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib
import matplotlib.text  # noqa: F401  (Text objects for the min-font check)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

FIGS = Path(__file__).resolve().parent
REPO = FIGS.parents[2]
DPI = 200
MAX_PNG_BYTES = 300 * 1024
# ACM two-column layout: every figure is drawn at its physical placement width (no downscaling); all text >= 7 pt.
FULL_W = 7.0     # full text width (figure*)
COL_W = 3.33     # single column
MIN_FONT_PT = 7.0
_BW = "results/m10_precision_h200/speed_llama/bw.json"
_bw = json.load(open(REPO / _BW))
T1_NOTE = ("'T1 constants': the CPU tier and the GPU recompute cost were measured on the H200 (docs/h200/icpe/T1_h200_cpu_tier.md). "
           f"The b32 tier ({_bw['targets']['b32']['GiBps']} GiB/s) is also H200-derived: read from {_BW} "
           f"(run_id {_bw['run_id']}), the bandwidth that puts Cake's meeting point at block 32 given the H200 recompute speed. "
           "Only the SSD-class and NFS-class throttle constants carry over from an earlier platform.")

# ---------------------------------------------------------------- palette (dataviz reference instance, light mode)
BLUE = "#2a78d6"         # categorical slot 1: simulation (Cake planner, T1 constants) / primary condition
BLUE_LIGHT = "#86b6ef"   # blue ramp step 250: secondary condition of the same series (ordinal)
ORANGE = "#eb6834"       # slot 2: H200-measured points
AQUA = "#1baf7a"         # slot 3
VIOLET = "#6250d6"       # slot 7
INK = "#0b0b0b"
INK2 = "#52514e"         # secondary ink
MUTED = "#898781"        # muted ink / de-emphasised series
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BAND = "#f2f1ee"
WHITE = "#ffffff"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "hatch.linewidth": 0.8,
    "axes.edgecolor": BASELINE,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "savefig.facecolor": WHITE,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
})
THRESH_STYLE = dict(color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)


def rel(p):
    return str(Path(p).resolve().relative_to(REPO))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _runs(values):
    out = []
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        for x in str(v).split(";"):
            x = x.strip()
            if x and x not in out:
                out.append(x)
    return out


class Src:
    """Collects the inputs, plotted values and checks of one figure; writes the sidecar."""

    def __init__(self, name):
        self.name = name
        self.inputs = {}
        self.values = {}
        self.checks = []
        self.notes = []

    def _reg(self, path, run_ids, related=None):
        r = rel(path)
        ent = self.inputs.setdefault(r, {"path": r, "sha256": sha256(path), "run_id": [], "related_run_ids": {}})
        for x in run_ids:
            if x not in ent["run_id"]:
                ent["run_id"].append(x)
        for k, v in (related or {}).items():
            cur = ent["related_run_ids"].setdefault(k, [])
            for x in v:
                if x not in cur:
                    cur.append(x)

    def csv(self, relpath, related_cols=()):
        p = REPO / relpath
        d = pd.read_csv(p)
        if "run_id" not in d.columns:
            raise SystemExit(f"{relpath}: no run_id column")
        rel_ids = {c: _runs(d[c].unique()) for c in related_cols if c in d.columns}
        self._reg(p, _runs(d["run_id"].unique()), rel_ids)
        return d

    def json(self, relpath, related_keys=()):
        p = REPO / relpath
        d = json.load(open(p))
        if "run_id" not in d:
            raise SystemExit(f"{relpath}: no run_id field")
        rel_ids = {k: _runs([d[k]]) for k in related_keys if d.get(k)}
        self._reg(p, [d["run_id"]], rel_ids)
        return d

    def check(self, what, got, expected, tol, expected_from):
        ok = (got is not None and expected is not None and abs(got - expected) <= tol + 1e-12)
        self.checks.append({"what": what, "value_in_plotted_file": got, "expected": expected, "tolerance": tol,
                            "expected_from": expected_from, "status": "match" if ok else "DIFF"})
        if not ok:
            print(f"  [{self.name}] DIFF {what}: file {got!r} vs {expected!r} ({expected_from})")
        return ok

    def write(self, fig, fname):
        w, h = (float(x) for x in fig.get_size_inches())
        if abs(w - FULL_W) > 1e-6 and abs(w - COL_W) > 1e-6:
            raise SystemExit(f"{fname}: width {w} in is neither full ({FULL_W}) nor column ({COL_W})")
        fig.canvas.draw()
        sizes = [x.get_fontsize() for x in fig.findobj(matplotlib.text.Text) if x.get_visible() and x.get_text().strip()]
        small = min(sizes)
        if small < MIN_FONT_PT - 1e-9:
            raise SystemExit(f"{fname}: smallest text {small} pt < {MIN_FONT_PT} pt")
        layout = {"intended_width": "ACM full width (figure*)" if abs(w - FULL_W) < 1e-6 else "ACM single column",
                  "width_in": w, "height_in": round(h, 3), "dpi": DPI, "smallest_text_pt": small,
                  "note": "drawn at the physical placement size; include at width=\\linewidth / \\textwidth without scaling"}
        out = FIGS / fname
        fig.savefig(out, dpi=DPI, metadata={"Software": None})
        plt.close(fig)
        size = out.stat().st_size
        if size > MAX_PNG_BYTES:
            print(f"  [{self.name}] WARNING: {fname} is {size} bytes (> {MAX_PNG_BYTES})")
        side = {
            "figure": fname,
            "layout": layout,
            "script": rel(__file__),
            "rule": "every plotted value is read from the inputs below; nothing is typed in",
            "inputs": list(self.inputs.values()),
            "plotted_values": self.values,
            "checks": self.checks,
            "notes": self.notes,
        }
        sp = FIGS / f"{Path(fname).stem}.sources.json"
        with open(sp, "w") as f:
            json.dump(side, f, indent=1, ensure_ascii=False, default=float)
            f.write("\n")
        n_diff = sum(c["status"] != "match" for c in self.checks)
        print(f"  wrote {rel(out)} ({size // 1024} KB), {rel(sp)}; checks {len(self.checks) - n_diff} match, {n_diff} DIFF")


def pct_label(v):
    """v is a fraction; keeps the sign of tiny negative values (e.g. −0.00%)."""
    p = 100.0 * v
    s = f"{abs(p):.2f}%" if abs(p) < 10.0 else f"{abs(p):.1f}%"
    return ("−" if p < 0 else "") + s


def one(df, what):
    if len(df) != 1:
        raise SystemExit(f"expected exactly one row for {what}, got {len(df)}")
    return df.iloc[0]


# ================================================================ fig3: scorecard
X1_COMPS = ["O-PF", "O-ORD", "O-DROP", "O-NOW"]
ROW_LABEL = {
    "O-PF": "O-PF\nprefetch before return",
    "O-ORD": "O-ORD\neviction order",
    "O-DROP": "O-DROP\ndrop never-returning at turn end",
    "O-NOW": "O-NOW\ndrop never-returning on eviction",
    "X5": "X5\nknow who never returns",
}


def fig3():
    s = Src("fig3_scorecard")
    T1 = "results/icpe_h200/t1"
    T3 = "results/icpe_h200/t3"

    # ---- chat: Mooncake conversation, SSD class, GPU idle, Cake, rule=min (R1/R2 smaller), CPU 128/256 GiB
    def chat_cells(path):
        d = s.csv(path, related_cols=("src_runs",))
        d = d[(d.trace == "conversation") & (d.rule == "min") & (d.slow == "ssd") & (d.gpu == "idle") & (d.restore == "cake")]
        out = {}
        for c in X1_COMPS:
            for cpu in (128, 256):
                out[(c, cpu)] = float(one(d[(d.component == c) & (d.cpu_GiB == cpu)], f"{path} {c}@{cpu}").headroom)
        return out

    chat_t1 = chat_cells(f"{T1}/p01/sim_V0/x1/x1_headroom_components.csv")
    chat_t3 = chat_cells(f"{T3}/p01/sim_V0/x1/x1_headroom_components.csv")
    cmp1 = s.csv(f"{T1}/t1_compare.csv").set_index("item")
    cmp3 = s.csv(f"{T3}/t3_compare.csv").set_index("item")
    for (c, cpu), v in chat_t1.items():
        s.check(f"chat T1 {c}@{cpu}", v, float(cmp1.loc[f"B1.{c}@{cpu}", "new_h200_cpu"]), 1e-9, f"{T1}/t1_compare.csv B1")
    for (c, cpu), v in chat_t3.items():
        s.check(f"chat T3 {c}@{cpu}", v, float(cmp3.loc[f"B1.{c}@{cpu}", "t3_stair"]), 1e-9, f"{T3}/t3_compare.csv B1")
    doc_t1 = {("O-PF", 128): 50.5, ("O-PF", 256): 46.5, ("O-ORD", 128): 27.65, ("O-ORD", 256): 31.0,
              ("O-DROP", 128): 27.6, ("O-DROP", 256): 34.9}
    for k, q in doc_t1.items():
        tol = 0.005 if k == ("O-ORD", 128) else 0.05
        s.check(f"chat T1 {k[0]}@{k[1]} (%)", round(100 * chat_t1[k], 4), q, tol,
                "docs/h200/icpe/T1_h200_cpu_tier.md results table (B1 new)")

    # ---- chat X5: Mooncake conversation, main cells (llama, k0, registered variant) -> max
    x5 = s.csv("results/p3_explore_h200/x5/x5_headroom.csv", related_cols=("run_id_cells",))
    x5v = s.json("results/p3_explore_h200/x5/x5_verdict.json")
    x5m = x5[(x5.model == "llama") & (x5.dq == "k0") & (x5.variant == "reg")]
    s.check("X5 main cells (both Mooncake traces)", len(x5m), x5v["n_main_cells"], 0, "x5_verdict.json n_main_cells")
    s.check("X5 main max (both Mooncake traces)", float(x5m.headroom.max()), x5v["main_headroom_max"], 1e-6,
            "x5_verdict.json main_headroom_max")
    x5c = x5m[x5m.trace == "conversation"]
    if len(x5c) != 24:
        raise SystemExit(f"X5 conversation main cells: {len(x5c)} != 24")
    x5c_best = x5c.loc[x5c.headroom.idxmax()]
    x5_all_best = x5m.loc[x5m.headroom.idxmax()]
    chat_x5 = float(x5c_best.headroom)
    s.notes.append(
        f"x5_verdict.json main_headroom_max = {x5v['main_headroom_max']:.4f} is over 48 main cells of two Mooncake traces; "
        f"its argmax is trace '{x5_all_best.trace}' (cpu {x5_all_best.cpu_GiB} GiB, {x5_all_best.slow}, GPU {x5_all_best.gpu}). "
        f"The chat panel plots the max over the 24 conversation cells only: {chat_x5:.4f} "
        f"(cpu {x5c_best.cpu_GiB} GiB, {x5c_best.slow}, GPU {x5c_best.gpu}); "
        f"{int((x5c.headroom >= 0.15).sum())} of 24 conversation cells >= 15%.")

    # ---- agent: FreeInference X1, max over the 24 cells, per component (all returning; user-initiated)
    def agent_max(path, verdict_path):
        d = s.csv(path, related_cols=("src_runs",))
        v = s.csv(verdict_path, related_cols=("src_runs",)).set_index("component")
        out, arg = {}, {}
        for c in X1_COMPS:
            dc = d[d.component == c]
            if len(dc) != 24:
                raise SystemExit(f"{path} {c}: {len(dc)} cells != 24")
            r = dc.loc[dc.headroom.idxmax()]
            out[c] = float(r.headroom)
            arg[c] = f"cpu {r.cpu_GiB} GiB, {r.slow}, GPU {r.gpu}"
            s.check(f"{Path(path).name} {c} max", out[c], float(v.loc[c, "max_headroom_all"]), 1e-9, verdict_path)
        return out, arg

    ag_all, ag_all_arg = agent_max(f"{T1}/p12/x1/x1fi_headroom_components.csv", f"{T1}/p12/x1/x1fi_verdict.csv")
    ag_usr, ag_usr_arg = agent_max(f"{T1}/p12/x1/x1fiuser_headroom_components.csv", f"{T1}/p12/x1/x1fiuser_verdict.csv")
    ag3_all, _ = agent_max(f"{T3}/p12/x1/x1fi_headroom_components.csv", f"{T3}/p12/x1/x1fi_verdict.csv")
    ag3_usr, _ = agent_max(f"{T3}/p12/x1/x1fiuser_headroom_components.csv", f"{T3}/p12/x1/x1fiuser_verdict.csv")
    s.check("agent X1 max over components (all)", max(ag_all.values()), float(cmp1.loc["B3.x1fi.max", "new_h200_cpu"]), 1e-9, "t1_compare.csv B3")
    s.check("agent X1 max over components (user)", max(ag_usr.values()), float(cmp1.loc["B3.x1fiuser.max", "new_h200_cpu"]), 1e-9, "t1_compare.csv B3")
    s.check("agent X1 max (all) %, 2 decimals", round(100 * max(ag_all.values()), 2), 0.30, 0.005, "T1 doc B3 0.30%")
    s.check("agent X1 max (user) %, 2 decimals", round(100 * max(ag_usr.values()), 2), 1.52, 0.005, "T1 doc B3 1.52%")
    s.check("agent T3 X1 max (all)", max(ag3_all.values()), float(cmp3.loc["B3.x1fi.max", "t3_stair"]), 1e-9, "t3_compare.csv B3")

    # ---- agent X5: FreeInference main cells -> max (X5 does not read CPU-tier constants; not rerun under T1)
    x5fv = s.json("results/icpe_h200/p12/x5/x5_verdict_fi.json")
    x5f = s.csv("results/icpe_h200/p12/x5/x5_headroom_fi.csv", related_cols=("run_id_cells",))
    x5fm = x5f[(x5f.model == "llama") & (x5f.dq == "k0") & (x5f.variant == "reg")]
    s.check("agent X5 main cells", len(x5fm), x5fv["n_main_cells"], 0, "x5_verdict_fi.json n_main_cells")
    agent_x5 = float(x5fv["main_headroom_max"])
    s.check("agent X5 max (csv vs verdict)", float(x5fm.headroom.max()), agent_x5, 1e-6, "x5_headroom_fi.csv main cells")
    s.check("agent X5 max %, 2 decimals", round(100 * agent_x5, 2), 2.44, 0.005, "ICPE_EVIDENCE B3 2.44%")
    s.notes.append("X5 values come from the pre-T1 X5 analyses (X5 does not read the CPU-tier constants, so T1 did not rerun it; "
                   "docs/h200/icpe/T1_h200_cpu_tier.md). No T3 (STAIR) X5 exists, so X5 rows have no hollow marker.")
    s.notes.append("Agent X1 bars are the max over 24 cells (slow tier SSD/b32/NFS x GPU idle/busy x CPU 128-1024 GiB, Cake); "
                   "agent X5 is the max over its 24 main cells (Llama, tiers b16/b32/b48 x GPU idle/busy x CPU 128-1024 GiB).")

    s.values = {
        "units": "oracle headroom over the best symmetric rival, fraction (plotted x100 as %)",
        "chat_T1_cake": {f"{c}@{cpu}GiB": v for (c, cpu), v in chat_t1.items()},
        "chat_T3_stair": {f"{c}@{cpu}GiB": v for (c, cpu), v in chat_t3.items()},
        "chat_X5_max_conversation_main": chat_x5,
        "chat_X5_argmax": {"cpu_GiB": int(x5c_best.cpu_GiB), "slow": x5c_best.slow, "gpu": x5c_best.gpu},
        "agent_T1_max_all_returning": ag_all, "agent_T1_argmax_all": ag_all_arg,
        "agent_T1_max_user_initiated": ag_usr, "agent_T1_argmax_user": ag_usr_arg,
        "agent_T3_max_all_returning": ag3_all, "agent_T3_max_user_initiated": ag3_usr,
        "agent_X5_max_main": agent_x5,
    }

    # ---- plot
    rows = X1_COMPS + ["X5"]
    fig, axes = plt.subplots(2, 1, figsize=(FULL_W, 6.0), sharex=True,
                             gridspec_kw=dict(hspace=0.58, left=0.28, right=0.975, top=0.865, bottom=0.08))
    h = 0.34
    xmax = 60.0

    def panel(ax, vals_a, vals_b, t3_a, t3_b, x5_val, x5_color, title, lab_a, lab_b, lab_x5):
        ys = np.arange(len(rows))[::-1]
        for y, r in zip(ys, rows):
            if r == "X5":
                bars = [(y, x5_val, x5_color, None)]
            else:
                bars = [(y + h / 2 + 0.01, vals_a[r], BLUE, t3_a.get(r) if t3_a else None),
                        (y - h / 2 - 0.01, vals_b[r], BLUE_LIGHT, t3_b.get(r) if t3_b else None)]
            for yy, v, col, t3 in bars:
                ax.barh(yy, 100 * v, height=h, color=col, edgecolor="none", zorder=2)
                end = 100 * max(v, t3 if t3 is not None else v)
                if t3 is not None:
                    ax.plot(100 * t3, yy, marker="D", ms=5.5, mfc=WHITE, mec=INK, mew=1.1, zorder=4, ls="none")
                # labels of bars that end left of the 5% line sit just right of it (keeps them off the marks)
                lx = 5.6 if end < 4.4 else end + 0.8
                ax.text(lx, yy, pct_label(v), va="center", ha="left", fontsize=7.5, color=INK, zorder=5)
        ax.set_yticks(ys)
        ax.set_yticklabels([ROW_LABEL[r] for r in rows], fontsize=7.5, color=INK)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.xaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        for t in (5, 15):
            ax.axvline(t, **THRESH_STYLE)
        ax.set_xlim(0, xmax)
        ax.text(0.0, 1.21, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5, color=INK, fontweight="bold")
        ax.tick_params(labelbottom=True)
        handles = [Patch(color=BLUE, label=lab_a), Patch(color=BLUE_LIGHT, label=lab_b)]
        if lab_x5:
            handles.append(Patch(color=x5_color, label=lab_x5))
        handles.append(Line2D([], [], marker="D", ms=5.5, mfc=WHITE, mec=INK, mew=1.1, ls="none", label="same, STAIR planner (T3)"))
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.07), ncol=len(handles), fontsize=7.5,
                  handlelength=1.2, columnspacing=1.0, borderaxespad=0.2)

    panel(axes[0], {c: chat_t1[(c, 128)] for c in X1_COMPS}, {c: chat_t1[(c, 256)] for c in X1_COMPS},
          {c: chat_t3[(c, 128)] for c in X1_COMPS}, {c: chat_t3[(c, 256)] for c in X1_COMPS},
          chat_x5, MUTED,
          "Chat trace: Mooncake conversation (SSD class, GPU idle, Cake planner)",
          "CPU 128 GiB", "CPU 256 GiB", "X5: max over 24 cells")
    panel(axes[1], ag_all, ag_usr, ag3_all, ag3_usr, agent_x5, BLUE,
          "Agent trace: FreeInference (each bar = max over 24 cells)",
          "all returning requests", "user-initiated only", None)
    for ax in axes:
        ax.text(5 - 0.4, 1.005, "< 5%: dead end", transform=ax.get_xaxis_transform(), ha="right", va="bottom", fontsize=7, color=INK2)
        ax.text(15 + 0.4, 1.005, "≥ 15%: worth it", transform=ax.get_xaxis_transform(), ha="left", va="bottom", fontsize=7, color=INK2)
    axes[1].set_xlabel("oracle headroom over the best symmetric rival (%) · simulation, T1 constants")
    s.notes.append(T1_NOTE)
    s.write(fig, "fig3_scorecard.png")


# ================================================================ GPU scaled-down replay (B6 / C2)
# T1G = rerun with the H200 CPU-tier constants (paper's main version; docs/h200/icpe/T1G_gpu_rerun.md).
# OLD = original replay with the CPU tier throttled at 35.4 GiB/s (shown faint, for reference).
P02_T1G = ("results/icpe_h200/t1/gpu/p02/analysis/p02_cells.csv", "results/icpe_h200/t1/gpu/p02/analysis/p02_verdict.json")
P02_OLD = ("results/icpe_h200/p02/p02_cells.csv", "results/icpe_h200/p02/p02_verdict.json")
OLD_LABEL = "CPU tier 35.4 GiB/s (old)"


def p02_load(s, which):
    cells_path, verdict_path = which
    cells = s.csv(cells_path, related_cols=("gpu_run",))
    pv = s.json(verdict_path)
    if pv["run_id"] not in set(cells.run_id):
        raise SystemExit(f"{verdict_path} run_id {pv['run_id']} not in {cells_path}")
    if not pv["passed"]:
        print(f"  note: {verdict_path} passed = {pv['passed']}")
    k = int(pv["k"])
    gpu = {}
    for cpu in sorted(cells.cpu_GiB.unique()):
        g = cells[cells.cpu_GiB == cpu].set_index("policy")
        L, H, O = (float(g.loc[p, "gpu_mean"]) for p in ("LRU", "HINT-PF", "O-PF"))
        Ls, Hs, Os = (float(g.loc[p, "s_orig_mean"]) for p in ("LRU", "HINT-PF", "O-PF"))
        gpu[int(cpu)] = dict(lru_s=L, hint_s=H, opf_s=O, hint_vs_lru=1 - H / L, opf_vs_lru=1 - O / L,
                             hint_share_of_opf=(L - H) / (L - O),
                             sim_scaled_hint_share_of_opf=(Ls - Hs) / (Ls - Os),
                             gpu_runs={p: g.loc[p, "gpu_run"] for p in ("LRU", "HINT-PF", "O-PF")})
    return cells, pv, k, gpu


# ================================================================ fig4: typing hint vs oracle prefetch
def fig4():
    s = Src("fig4_hint")
    t1 = s.csv("results/icpe_h200/t1/p02/fullsim_hint.csv")
    t3 = s.csv("results/icpe_h200/t3/p02/fullsim_hint.csv")
    for d, nm in ((t1, "T1"), (t3, "T3")):
        if len(d) != 24 or set(d.trace) != {"conversation"} or set(d.slow) != {"ssd", "b32", "nfs"}:
            raise SystemExit(f"fullsim_hint {nm}: unexpected cells")
    _, pv, k, gpu = p02_load(s, P02_T1G)
    _, pv_old, k_old, gpu_old = p02_load(s, P02_OLD)
    if k_old != k:
        raise SystemExit("old and T1G replays use different k")
    for cpu, q in zip(sorted(gpu), (0.81, 0.79)):
        s.check(f"T1G GPU HINT-PF share of O-PF gain, CPU {cpu} GiB cell", round(gpu[cpu]["hint_share_of_opf"], 2), q, 0.005,
                "coordinator / docs/h200/icpe/T1G_gpu_rerun.md (0.81 / 0.79)")
    for cpu in sorted(gpu_old):
        s.check(f"old GPU HINT-PF share of O-PF gain, CPU {cpu} GiB cell", round(gpu_old[cpu]["hint_share_of_opf"], 2), 0.81, 0.005,
                "ICPE_EVIDENCE C2 (0.81 / 0.81, old replay)")
    for cpu, (qh, qo) in zip(sorted(gpu), ((14.5, 17.9), (12.9, 16.2))):
        s.check(f"T1G GPU HINT-PF over LRU, {cpu} (%)", round(100 * gpu[cpu]["hint_vs_lru"], 1), qh, 0.05, "T1G doc numbers table")
        s.check(f"T1G GPU O-PF over LRU, {cpu} (%)", round(100 * gpu[cpu]["opf_vs_lru"], 1), qo, 0.05, "T1G doc numbers table")
    t1_share_med = t1.groupby("slow").hint_share_of_opf.median().to_dict()
    for sl, q in (("ssd", 0.933), ("b32", 0.941), ("nfs", 1.118)):
        s.check(f"T1 median HINT-PF/O-PF share {sl}", round(t1_share_med[sl], 3), q, 0.0005, "T1 doc C2 (new)")
    s.check("T1 cells with share > 1", int((t1.hint_share_of_opf > 1).sum()), 4, 0, "T1 doc C2 (new)")
    s.check("T1 HINT-PF over LRU min (%)", round(100 * t1.hint_vs_lru.min(), 1), 6.9, 0.05, "T1 doc C2")
    s.check("T1 HINT-PF over LRU max (%)", round(100 * t1.hint_vs_lru.max(), 1), 49.1, 0.05, "T1 doc C2")
    s.check("T1 O-PF over LRU min (%)", round(100 * t1.opf_vs_lru.min(), 1), 7.0, 0.05, "T1 doc C2")
    s.check("T1 O-PF over LRU max (%)", round(100 * t1.opf_vs_lru.max(), 1), 54.2, 0.05, "T1 doc C2")
    above = {nm: d[d.hint_share_of_opf > 1][["cpu_GiB", "slow", "gpu", "hint_share_of_opf"]].to_dict(orient="records")
             for d, nm in ((t1, "cake_T1"), (t3, "stair_T3"))}
    s.values = {
        "units": "improvement of mean TTFT of returning requests over LRU, fraction (plotted x100)",
        "cake_T1": t1[["cpu_GiB", "slow", "gpu", "hint_vs_lru", "opf_vs_lru", "hint_share_of_opf"]].to_dict(orient="records"),
        "stair_T3": t3[["cpu_GiB", "slow", "gpu", "hint_vs_lru", "opf_vs_lru", "hint_share_of_opf"]].to_dict(orient="records"),
        "median_share_by_slow_T1": t1_share_med,
        "median_share_by_slow_T3": t3.groupby("slow").hint_share_of_opf.median().to_dict(),
        "cells_with_share_gt_1": above,
        "gpu_scaled_down_T1G_primary": {"k": k, "verdict_run_id": pv["run_id"], "cells": gpu},
        "gpu_scaled_down_old_35p4GiBps": {"k": k_old, "verdict_run_id": pv_old["run_id"], "cells": gpu_old},
    }
    s.notes.append(f"GPU points: scaled-down replay (k = {k}: 1/{k} of the sessions, CPU tier {128 // k} / {256 // k} GiB for the "
                   "128 / 256 GiB cells), Llama-3.1-8B, SSD class, Cake; gain computed from gpu_mean in p02_cells.csv "
                   "(LRU rep 0 run). 'sim_scaled_hint_share_of_opf' is the same ratio from the scaled-down simulation (s_orig_mean); not plotted.")
    s.notes.append("Primary GPU points = T1G rerun with the H200 CPU-tier constants (docs/h200/icpe/T1G_gpu_rerun.md); "
                   f"hollow gray diamonds = the original replay with the CPU tier throttled at 35.4 GiB/s ({P02_OLD[0]}).")

    fig, ax = plt.subplots(figsize=(COL_W, 4.05), gridspec_kw=dict(left=0.155, right=0.965, top=0.79, bottom=0.10))
    lim = 60
    xs = np.array([0, lim])
    ax.plot(xs, xs, color=MUTED, lw=1.0, zorder=1)
    ax.plot(xs, 0.9 * xs, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(47.5, 52.5, "y = x", ha="right", va="bottom", fontsize=7, color=INK2)
    ax.text(59.5, 47.0, "y = 0.9x", ha="right", va="top", fontsize=7, color=INK2)
    ax.scatter(100 * t1.opf_vs_lru, 100 * t1.hint_vs_lru, s=14, color=BLUE, edgecolor=WHITE, linewidth=0.6, zorder=3,
               label="simulation, Cake planner (T1), 24 cells")
    ax.scatter(100 * t3.opf_vs_lru, 100 * t3.hint_vs_lru, s=16, marker="s", facecolor="none", edgecolor=BLUE, linewidth=0.9,
               zorder=3, label="simulation, STAIR planner (T3), 24 cells")
    ox = [100 * gpu_old[c]["opf_vs_lru"] for c in sorted(gpu_old)]
    oy = [100 * gpu_old[c]["hint_vs_lru"] for c in sorted(gpu_old)]
    gx = [100 * gpu[c]["opf_vs_lru"] for c in sorted(gpu)]
    gy = [100 * gpu[c]["hint_vs_lru"] for c in sorted(gpu)]
    ax.scatter(gx, gy, s=30, marker="D", color=ORANGE, edgecolor=WHITE, linewidth=0.7, zorder=5,
               label=f"H200 measured, scaled down (k = {k}), 2 cells")
    ax.scatter(ox, oy, s=26, marker="D", facecolor="none", edgecolor=MUTED, linewidth=0.8, zorder=4,
               label=f"same, {OLD_LABEL}")
    shares = " / ".join(f"{gpu[c]['hint_share_of_opf']:.2f}" for c in sorted(gpu))
    ax.annotate(f"H200: HINT-PF gets\n{shares} of O-PF's\ngain over LRU",
                xy=(gx[0], gy[0]), xytext=(29, 10.5), fontsize=7, color=INK, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7, shrinkA=2, shrinkB=4))
    n1, n3 = len(above["cake_T1"]), len(above["stair_T3"])
    tiers = sorted({r["slow"] for v in above.values() for r in v})
    tier_txt = ", ".join({"nfs": "NFS class", "ssd": "SSD class"}.get(t, t) for t in tiers)
    ax.text(lim - 1, 1.2, f"above y = x: {n1} Cake and\n{n3} STAIR cells, all {tier_txt}"
            if len(tiers) == 1 else f"above y = x: {n1} Cake and\n{n3} STAIR cells ({tier_txt})",
            fontsize=7, color=INK2, ha="right", va="bottom")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_yticks([0, 20, 40, 60])
    ax.set_xlabel("O-PF (oracle prefetch) gain over LRU (%)")
    ax.set_ylabel("HINT-PF (typing hint) gain over LRU (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.17, 1.01), ncol=1, fontsize=7, handletextpad=0.3, borderaxespad=0.1,
              labelspacing=0.3)
    s.notes.append(T1_NOTE)
    s.notes.append("Simulation cells: Mooncake conversation, R1, SSD class / b32 / NFS class x GPU idle/busy x CPU 128-1024 GiB.")
    s.write(fig, "fig4_hint.png")


# ================================================================ fig5: INT4 vs BF16 restore time
def fig5():
    s = Src("fig5_int4")
    quoted = {"llama": (37.4, 49.1), "qwen": (36.7, 54.5)}
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.1), gridspec_kw=dict(left=0.08, right=0.99, top=0.79, bottom=0.18, wspace=0.18))
    vals = {}
    for ax, m in zip(axes, ("llama", "qwen")):
        base = f"results/m10_precision_h200/speed_{m}"
        v = s.json(f"{base}/k0_verdict_int4.json")
        spec = s.json(f"{base}/spec_grid_int4.json")
        bw = s.json(f"{base}/bw.json")
        if spec["run_id"] != v["run_id"]:
            raise SystemExit(f"{m}: spec run_id {spec['run_id']} != verdict run_id {v['run_id']}")
        model_name = Path(str(bw["model_path"]).split("models--")[-1].split("/")[0]).name.split("--")[-1]
        tiers = [t for t in spec["tiers"] if t["kind"] == "throttle"]
        tiers.sort(key=lambda t: -t["GiBps"])
        cells = {(c["bw_name"], c["gpu_state"]): c for c in v["cells"]}
        rows = []
        for t in tiers:
            for g in ("idle", "busy"):
                c = cells[(t["name"], g)]
                if c["contaminated_rows"] != 0:
                    raise SystemExit(f"{m} {t['name']} {g}: contaminated rows")
                rows.append(dict(tier=t["name"], GiBps=t["GiBps"], gpu=g, bf16_s=c["median_E0_s"], int4_s=c["median_E1_s"],
                                 gain=c["gain_E1_over_E0"], n_pairs=c["n_pairs"], int4_wins=c["E1_wins"]))
        g_lo, g_hi = min(r["gain"] for r in rows), max(r["gain"] for r in rows)
        s.check(f"{m} INT4 gain min over throttled tiers (%)", round(100 * g_lo, 2), quoted[m][0], 0.05 + 1e-9, "task brief / ICPE_EVIDENCE D2")
        s.check(f"{m} INT4 gain max over throttled tiers (%)", round(100 * g_hi, 2), quoted[m][1], 0.05 + 1e-9, "task brief / ICPE_EVIDENCE D2")
        cpu_cells = [dict(gpu=g, bf16_s=cells[("cpu", g)]["median_E0_s"], int4_s=cells[("cpu", g)]["median_E1_s"],
                          gain=cells[("cpu", g)]["gain_E1_over_E0"], int4_wins=cells[("cpu", g)]["E1_wins"])
                     for g in ("idle", "busy") if ("cpu", g) in cells]
        vals[m] = dict(model=model_name, context_tokens=spec["main_length"], throttled=rows, cpu_tier_not_plotted=cpu_cells,
                       delta_star=v["delta_star"])
        # bars
        w = 0.36
        x = np.arange(len(rows), dtype=float)
        x += np.repeat(np.arange(len(tiers)) * 0.5, 2)   # gap between tiers
        ax.bar(x - w / 2 - 0.01, [r["bf16_s"] for r in rows], width=w, color=MUTED, label="BF16 on the slow tier", zorder=2)
        ax.bar(x + w / 2 + 0.01, [r["int4_s"] for r in rows], width=w, facecolor=AQUA, edgecolor=WHITE, hatch="////", linewidth=0,
               label="INT4 on the slow tier", zorder=2)
        top = max(r["bf16_s"] for r in rows)
        for xi, r in zip(x, rows):
            ax.text(xi, r["bf16_s"] + 0.03 * top, f"−{100 * r['gain']:.1f}%", ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([r["gpu"] for r in rows], fontsize=7)
        for ti, t in enumerate(tiers):
            xc = x[2 * ti:2 * ti + 2].mean()
            ax.text(xc, -0.11, f"{t['name']}\n{t['GiBps']:.2f} GiB/s", transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=7, color=INK)
        ax.set_ylim(0, top * 1.15)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(f"{model_name}, {spec['main_length'] // 1024}K-token context", loc="left", fontsize=7.5, color=INK)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("median restore time (s)")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.10), ncol=2, fontsize=7.5)
    fig.text(0.99, 0.975, "numbers: INT4 vs BF16 restore-time change · H200 measured, throttled slow tiers, GPU idle / busy",
             ha="right", va="top", fontsize=7, color=INK2)
    s.values = {"units": "seconds (median restore time over n_pairs paired runs); gain = 1 - INT4/BF16", **vals}
    s.notes.append("Only the throttled slow tiers are plotted (as asked). The same verdict files also hold the unthrottled CPU tier, "
                   "where INT4 does not help: see plotted_values.<model>.cpu_tier_not_plotted.")
    s.notes.append("Tier bandwidths in the x labels are read per model from spec_grid_int4.json (the throttle targets differ between models).")
    s.write(fig, "fig5_int4.png")


# ================================================================ fig6: MPS trade-off
MPS_ORDER = ["stream", "mps", "mps87", "mps75", "mps50"]
MPS_LABEL = {"stream": "stream", "mps": "MPS 100%", "mps87": "MPS 87%", "mps75": "MPS 75%", "mps50": "MPS 50%"}
# label placement only (offset in points, ha, va) per (background, level)
LABEL_OFFSET = {
    ("cprefill:2048:65536", "stream"): (6, 4, "left", "bottom"),
    ("cprefill:2048:65536", "mps75"): (6, 4, "left", "bottom"),
    ("mixed:512:32:8192", "stream"): (6, 4, "left", "bottom"),
    ("mixed:512:32:8192", "mps75"): (-6, -2, "right", "top"),
}


def fig6():
    s = Src("fig6_mps")
    reg_path = "results/s3_h200/e2e/mps_tradeoff.csv"
    t1_path = "results/icpe_h200/t1/e2e/mps_tradeoff.csv"
    reg = s.csv(reg_path, related_cols=("src_runs",))
    t1v = s.csv(t1_path, related_cols=("src_runs",))
    rep = s.json("results/s3_h200/e2e_gpu/report.json")
    runs = s.csv("results/s3_h200/e2e_gpu/gpu_runs.csv").set_index("run_id")
    s.notes.append(f"Simulation curve = the registered E2E-C run ({reg_path}); the T1 rerun with H200 CPU-tier constants "
                   f"({t1_path}) is report-only per its registration (docs/h200/icpe/T1_h200_cpu_tier.md); differences listed in plotted_values.")
    s.notes.append("Curve points: test half, Llama-3.1-8B, Mooncake conversation, median over 4 cells (2 slow tiers x 2 CPU sizes); "
                   "y = (K2 - K3(m)) / K2 with K2 = INT4 + typing-hint prefetch under stream sharing; x = background throughput ratio B.")
    s.notes.append("Scaled-down points (hollow = simulation of the replayed workload, filled = H200 measured): Llama-3.1-8B, "
                   "conversation sampled 1/8, CPU 128 GiB scaled, SSD class, Cake, INT4 on the slow tier, HINT-PF; "
                   "gain = 1 - TTFT(m) / TTFT(stream); report.json verdict: " + str(rep.get("verdict")))
    s.notes.append("Registered E2E-C pass rule drawn as dashed lines: Delta > 15% and B >= 0.95 "
                   "(docs/stage3_design_20261024/runs/E2E_combo.md section 4); thresholds are rule constants, not data.")
    bgs = ["cprefill:2048:65536", "mixed:512:32:8192"]
    bg_title = {"cprefill:2048:65536": "background: chunked prefill (2048-token chunks, 64K context)",
                "mixed:512:32:8192": "background: mixed prefill/decode (512 / 32, 8K context)"}
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.8), sharey=True,
                             gridspec_kw=dict(left=0.085, right=0.99, top=0.755, bottom=0.13, wspace=0.08))
    vals = {}
    diffs = []
    allB = list(reg[(reg.model == "llama") & (reg.trace == "conversation") & (reg.half == "test") & reg.bg.isin(bgs)].B_median)
    allB += [c["B_gpu"] for c in rep["cells"].values() if c.get("status") == "ok"]
    allB += [c["sim"]["B"] for c in rep["cells"].values() if c.get("status") == "ok"]
    XMIN = math.floor((min(allB) - 0.03) * 20) / 20
    for ax, bg in zip(axes, bgs):
        r = reg[(reg.model == "llama") & (reg.trace == "conversation") & (reg.bg == bg) & (reg.half == "test")].set_index("mode")
        t = t1v[(t1v.model == "llama") & (t1v.trace == "conversation") & (t1v.bg == bg) & (t1v.half == "test")].set_index("mode")
        modes = [m for m in MPS_ORDER if m in r.index]
        for m in modes:
            if m in t.index:
                diffs.append(dict(bg=bg, mode=m, d_gain_pp=100 * (t.loc[m, "gain_vs_K2_median"] - r.loc[m, "gain_vs_K2_median"]),
                                  d_B=t.loc[m, "B_median"] - r.loc[m, "B_median"]))
        cx = [float(r.loc[m, "B_median"]) for m in modes]
        cy = [100 * float(r.loc[m, "gain_vs_K2_median"]) for m in modes]
        # registered pass region
        ax.axhline(15, **THRESH_STYLE)
        ax.axvline(0.95, **THRESH_STYLE)
        ax.add_patch(plt.Rectangle((0.95, 15), 0.10, 100, color=BAND, zorder=0, lw=0))
        ax.plot(cx, cy, color=BLUE, lw=2, zorder=3, marker="o", ms=5.5, mec=WHITE, mew=1.0,
                label="simulation (registered E2E-C, full size, median of 4 cells)")
        # scaled-down sim + GPU
        sx, sy, gx, gy, glab, skipped = [], [], [], [], [], []
        for m in MPS_ORDER:
            c = rep["cells"].get(f"{bg}|{m}")
            if c is None:
                continue
            run = c["gpu"]["run"]
            if int(runs.loc[run, "contaminated"]) != 0:
                raise SystemExit(f"{run} contaminated")
            if c.get("status") != "ok":
                skipped.append(dict(cell=f"{bg}|{m}", status=c.get("status"), reason=c.get("sim", {}).get("reason")))
                continue
            g_gpu = 0.0 if m == "stream" else c["gain_gpu"]
            g_sim = 0.0 if m == "stream" else c["gain_sim"]
            if m != "stream":
                st = rep["cells"][f"{bg}|stream"]
                s.check(f"{bg}|{m} gain_gpu = 1 - ttft/ttft(stream)", g_gpu, 1 - c["gpu"]["ttft"] / st["gpu"]["ttft"], 1e-9, "report.json gpu ttft")
                s.check(f"{bg}|{m} gain_sim = 1 - ttft/ttft(stream)", g_sim, 1 - c["sim"]["ret_ttft_mean"] / st["sim"]["ret_ttft_mean"], 1e-9,
                        "report.json sim ret_ttft_mean")
            sx.append(c["sim"]["B"]); sy.append(100 * g_sim)
            gx.append(c["B_gpu"]); gy.append(100 * g_gpu); glab.append(m)
            vals.setdefault(bg, {}).setdefault("scaled_down", {})[m] = dict(
                gpu_run=run, B_gpu=c["B_gpu"], gain_gpu=g_gpu, B_sim=c["sim"]["B"], gain_sim=g_sim,
                ttft_gpu_s=c["gpu"]["ttft"], ttft_sim_s=c["sim"]["ret_ttft_mean"])
        ax.plot(sx, sy, color=BLUE, lw=0.8, ls=(0, (2, 2)), zorder=2)
        ax.scatter(sx, sy, s=34, facecolor=WHITE, edgecolor=BLUE, linewidth=1.2, zorder=4,
                   label="simulation of the replayed workload (scaled down)")
        ax.plot(gx, gy, color=ORANGE, lw=0.8, zorder=2)
        ax.scatter(gx, gy, s=52, marker="D", color=ORANGE, edgecolor=WHITE, linewidth=1.0, zorder=5,
                   label="H200 measured (scaled down)")
        for x_, y_, m in zip(gx, gy, glab):
            dx, dy, ha, va = LABEL_OFFSET.get((bg, m), (6, -2, "left", "top"))
            ax.annotate(MPS_LABEL[m], (x_, y_), xytext=(dx, dy), textcoords="offset points", fontsize=7, color=INK, ha=ha, va=va)
        vals.setdefault(bg, {})["curve_registered"] = {m: dict(B_median=float(r.loc[m, "B_median"]),
                                                                gain_vs_K2_median=float(r.loc[m, "gain_vs_K2_median"]))
                                                       for m in modes}
        vals[bg]["not_plotted"] = skipped
        ax.set_title(bg_title[bg], loc="left", fontsize=7.5, color=INK)
        ax.set_xlim(XMIN, 1.0)
        ax.set_ylim(-5, 85)
        ax.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlabel("background throughput ratio B")
        ax.text(0.975, 50, "registered pass region", fontsize=7, color=INK2, va="center", ha="center", rotation=90)
    axes[0].set_ylabel("TTFT improvement over stream sharing (%)")
    h, l_ = axes[0].get_legend_handles_labels()
    h.append(Line2D([], [], **{k: v for k, v in THRESH_STYLE.items() if k != "zorder"}))
    l_.append("registered E2E-C rule: Δ > 15% and B ≥ 0.95")
    fig.legend(h, l_, loc="upper left", bbox_to_anchor=(0.075, 0.995), ncol=2, fontsize=7, handletextpad=0.4, columnspacing=1.0)
    fig.text(0.085, 0.835, "each line connects, in order: stream → MPS 100% → 87% → 75% → 50% (where available)",
             fontsize=7, color=INK2, ha="left", va="bottom")
    s.values = {"units": "B = background throughput ratio; gain fraction (plotted x100)", "per_background": vals,
                "t1_minus_registered_curve": diffs,
                "t1_minus_registered_max_abs_gain_pp": max(abs(d["d_gain_pp"]) for d in diffs),
                "t1_minus_registered_max_abs_B": max(abs(d["d_B"]) for d in diffs)}
    s.write(fig, "fig6_mps.png")


# ================================================================ fig7: simulator vs GPU
def fig7():
    s = Src("fig7_validation")
    cells, pv, k, _ = p02_load(s, P02_T1G)
    old, pv_old, _, _ = p02_load(s, P02_OLD)
    order = ["LRU", "FREQ2Q", "O-BEL", "HINT-PF", "O-PF"]
    pols = [p for p in order if p in set(cells.policy)] + sorted(set(cells.policy) - set(order))
    cpus = sorted(int(c) for c in cells.cpu_GiB.unique())
    s.check("T1G ratio min (3 decimals)", round(float(cells.ratio.min()), 3), 0.911, 0.0005, "coordinator / T1G doc B6 (0.911-0.987)")
    s.check("T1G ratio max (3 decimals)", round(float(cells.ratio.max()), 3), 0.987, 0.0005, "coordinator / T1G doc B6 (0.911-0.987)")
    s.check("T1G all cells in band (csv in_band)", float(cells.in_band.all()), 1.0, 0, "p02_cells.csv in_band")
    s.check("T1G ratio recomputed gpu_mean / s_orig_mean", float((cells.gpu_mean / cells.s_orig_mean - cells.ratio).abs().max()), 0.0, 1e-9, "p02_cells.csv")
    s.check("old ratio min (3 decimals)", round(float(old.ratio.min()), 3), 0.911, 0.0005, "ICPE_EVIDENCE B6 (old replay)")
    s.check("old ratio max (3 decimals)", round(float(old.ratio.max()), 3), 0.975, 0.0005, "ICPE_EVIDENCE B6 (old replay)")
    s.notes.append(f"Scaled-down replay: k = {k}, so the 128 / 256 GiB cells run with a {128 // k} / {256 // k} GiB CPU tier; "
                   f"Llama-3.1-8B, Mooncake conversation, SSD class, Cake; TTFT excludes queueing (registered). "
                   f"T1G p02_verdict.json passed = {pv['passed']}, n_clean_runs = {pv['n_clean_runs']}; "
                   f"old replay passed = {pv_old['passed']}, n_clean_runs = {pv_old['n_clean_runs']}. "
                   "Filled = T1G rerun with H200 CPU-tier constants (main); hollow gray = old replay, CPU tier throttled at 35.4 GiB/s.")
    fig, ax = plt.subplots(figsize=(COL_W, 3.0), gridspec_kw=dict(left=0.20, right=0.97, top=0.76, bottom=0.15))
    ax.axvspan(0.9, 1.1, color=BAND, zorder=0, lw=0)
    ax.axvline(1.0, color=BASELINE, lw=0.8, zorder=1)
    for x in (0.9, 1.1):
        ax.axvline(x, **THRESH_STYLE)
    markers = {cpus[0]: "o", cpus[-1]: "s"}
    ys = np.arange(len(pols))[::-1]
    vals = []
    old_vals = []
    for cpu, off in zip(cpus, (0.13, -0.13)):
        xo, yo = [], []
        for y, p in zip(ys, pols):
            r = one(old[(old.policy == p) & (old.cpu_GiB == cpu)], f"old {p}@{cpu}")
            xo.append(float(r.ratio)); yo.append(y + off)
            old_vals.append(dict(policy=p, cpu_GiB_full_size=cpu, ratio=float(r.ratio), gpu_run=r.gpu_run))
        ax.scatter(xo, yo, s=22 if markers[cpu] == "o" else 20, marker=markers[cpu], facecolor="none", edgecolor=MUTED,
                   linewidth=0.8, zorder=3, label=f"{cpu} GiB cell, {OLD_LABEL}")
    for cpu, off in zip(cpus, (0.13, -0.13)):
        xs_, ys_ = [], []
        for y, p in zip(ys, pols):
            r = one(cells[(cells.policy == p) & (cells.cpu_GiB == cpu)], f"{p}@{cpu}")
            xs_.append(float(r.ratio)); ys_.append(y + off)
            vals.append(dict(policy=p, cpu_GiB_full_size=cpu, cpu_GiB_replayed=cpu // k, ratio=float(r.ratio),
                             gpu_mean_s=float(r.gpu_mean), s_orig_mean_s=float(r.s_orig_mean), gpu_run=r.gpu_run, n_ret=int(r.n_ret)))
        ax.scatter(xs_, ys_, s=24 if markers[cpu] == "o" else 22, marker=markers[cpu], color=ORANGE, edgecolor=WHITE,
                   linewidth=0.6, zorder=4, label=f"{cpu} GiB cell (replayed at {cpu // k} GiB)")
    lo = min(vals, key=lambda v: v["ratio"])
    hi = max(vals, key=lambda v: v["ratio"])
    ax.text(1.0, -0.95, f"range {lo['ratio']:.3f}–{hi['ratio']:.3f};\nall {len(vals)} cells in [0.9, 1.1]",
            fontsize=7, color=INK2, ha="center", va="center")
    ax.set_yticks(ys)
    ax.set_yticklabels(pols, color=INK)
    ax.set_ylim(-1.45, len(pols) - 0.4)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0.86, 1.14)
    ax.set_xticks([0.9, 0.95, 1.0, 1.05, 1.1])
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("measured / simulated mean TTFT")
    hs, ls_ = ax.get_legend_handles_labels()
    from matplotlib.legend_handler import HandlerTuple
    # hs: [old 128, old 256, T1G 128, T1G 256]
    ax.legend([hs[2], hs[3], (hs[0], hs[1])], [ls_[2], ls_[3], f"both cells, {OLD_LABEL}"],
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)}, loc="lower left", bbox_to_anchor=(-0.22, 1.01), ncol=1,
              fontsize=7, handletextpad=0.4, labelspacing=0.3, borderaxespad=0.1)
    s.values = {"units": "ratio = gpu_mean / s_orig_mean", "k": k, "cells_T1G_primary": vals, "cells_old_35p4GiBps": old_vals}
    s.write(fig, "fig7_validation.png")


# ================================================================ fig8: T4 bridge (hit-ratio gap vs TTFT gap)
def fig8():
    vpath = REPO / "results/icpe_h200/t4/t4_verdict.json"
    if not vpath.exists():
        print("  fig8_bridge: SKIPPED - results/icpe_h200/t4/t4_verdict.json does not exist yet (T4 still running)")
        return
    s = Src("fig8_bridge")
    v = s.json("results/icpe_h200/t4/t4_verdict.json", related_keys=("sweep_run_id",))
    C = s.csv("results/icpe_h200/t4/t4_cells.csv", related_cols=("sweep_run_id",))
    W = s.csv("results/icpe_h200/t4/t4_weeks.csv")
    n_weeks = W.trace.nunique()
    rule = v["rule"]
    main = C[(C.slow == "ssd") & (C.gpu == "idle") & (C.subset == "ret")].sort_values("cpu_GiB")
    vm = pd.DataFrame(v["main"]).sort_values("cpu_GiB")
    for col in ("G_hit_pp", "G_rec", "G_load", "G_cake"):
        s.check(f"main {col} (cells csv vs verdict json), max abs diff",
                float((main[col].to_numpy() - vm[col].to_numpy()).__abs__().max()), 0.0, 1e-4, "t4_verdict.json main")
    caps = main.cpu_GiB.to_numpy()
    tier = {sl: C[(C.slow == sl) & (C.gpu == "idle") & (C.subset == "ret")].sort_values("cpu_GiB") for sl in ("ssd", "b32", "nfs")}
    nfs = tier["nfs"].set_index("cpu_GiB").G_cake
    s.values = {
        "units": "G_hit_pp in percentage points; G_rec / G_load / G_cake fractions (plotted x100)",
        "registered_rule": rule, "registered_verdict": v["verdict"], "robustness": v.get("robustness"),
        "weeks_in_trace": int(n_weeks),
        "main_ssd_idle_ret": main[["cpu_GiB", "G_hit_pp", "G_rec", "G_load", "G_cake", "G_rec_cake", "hr_lru_none", "hr_obel_none", "class"]].to_dict(orient="records"),
        "G_cake_by_slow_idle_ret": {sl: d[["cpu_GiB", "G_cake", "class"]].to_dict(orient="records") for sl, d in tier.items()},
    }
    s.notes.append("Brief from the coordinator: NFS-class G_cake about 18-28% at 16-32 GiB; file values: "
                   + ", ".join(f"{int(c)} GiB {100 * nfs.loc[c]:.1f}%" for c in nfs.index) + ".")
    s.notes.append(f"Policies (simulation, FreeInference, {n_weeks} weekly slices merged, Llama-3.1-8B costs, H200 CPU-tier constants): "
                   "G_hit = CPU-tier hit ratio of O-BEL+NONE minus LRU+NONE (pp); G_rec = oracle order vs LRU with no slow tier, misses "
                   "recomputed (load-only restore); G_load = with slow tier, load-only restore; G_cake = with slow tier, Cake restore.")

    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 3.0), gridspec_kw=dict(left=0.075, right=0.985, top=0.80, bottom=0.17, wspace=0.36))

    def logx(ax):
        ax.set_xscale("log", base=2)
        ax.set_xticks(caps)
        ax.set_xticklabels([str(int(c)) for c in caps])
        ax.minorticks_off()
        ax.set_xlim(caps.min() / 1.25, caps.max() * 1.25)
        ax.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlabel("CPU tier capacity (GiB)")

    def line(ax, x, y, col, lab, lw=2, ls="-", mk="o"):
        ax.plot(x, y, color=col, lw=lw, ls=ls, marker=mk, ms=4.5, mec=WHITE, mew=0.7, zorder=3, label=lab)

    a, b, c = axes
    logx(a)
    a.axhline(rule["hit_gap_pp"], **THRESH_STYLE)
    line(a, caps, main.G_hit_pp, INK2, "hit-ratio gap")
    a.set_ylabel("percentage points")
    a.set_title("CPU-tier hit ratio:\noracle order minus LRU (no slow tier)", loc="left", fontsize=7.5, color=INK)
    a.text(caps.max() * 1.2, rule["hit_gap_pp"] + 0.15, f"{rule['hit_gap_pp']:g} pp", ha="right", va="bottom", fontsize=7, color=INK2)
    a.set_ylim(bottom=min(0, float(main.G_hit_pp.min())) - 0.3)

    logx(b)
    for t in (rule["small"], rule["gap"]):
        b.axhline(100 * t, **THRESH_STYLE)
    line(b, caps, 100 * main.G_rec, BLUE, "no slow tier, recompute misses", mk="o")
    line(b, caps, 100 * main.G_load, AQUA, "slow tier, load only", mk="s")
    line(b, caps, 100 * main.G_cake, ORANGE, "slow tier + Cake", mk="^")
    b.set_ylabel("oracle gain vs LRU, mean TTFT (%)")
    b.set_title("TTFT: oracle order vs LRU\n(SSD class, GPU idle)", loc="left", fontsize=7.5, color=INK)

    logx(c)
    for t in (rule["small"], rule["gap"]):
        c.axhline(100 * t, **THRESH_STYLE)
    line(c, caps, 100 * tier["b32"].G_cake, MUTED, "b32", lw=1.6, mk=TIER_MARK["b32"])
    line(c, caps, 100 * tier["ssd"].G_cake, ORANGE, "SSD class", mk=TIER_MARK["ssd"])
    line(c, caps, 100 * tier["nfs"].G_cake, VIOLET, "NFS class", mk=TIER_MARK["nfs"])
    c.set_title("TTFT, slow tier + Cake,\nby slow tier (GPU idle)", loc="left", fontsize=7.5, color=INK)
    c.set_ylabel("oracle gain vs LRU (%)")
    ymax = max(100 * float(np.nanmax(main[["G_rec", "G_load", "G_cake"]].to_numpy())),
               max(100 * float(d.G_cake.max()) for d in tier.values()), 100 * rule["gap"])
    for ax in (b, c):
        ax.set_ylim(min(0, ax.get_ylim()[0]), ymax * 1.15)
        ax.text(caps.max() * 1.2, 100 * rule["gap"] + 0.3, f"{100 * rule['gap']:g}%", ha="right", va="bottom", fontsize=7, color=INK2)
        ax.text(caps.max() * 1.2, 100 * rule["small"] + 0.3, f"{100 * rule['small']:g}%", ha="right", va="bottom", fontsize=7, color=INK2)
    b.legend(loc="upper left", fontsize=7, ncol=1, handlelength=1.5, borderaxespad=0.3)
    c.legend(loc="upper left", fontsize=7, ncol=1, handlelength=1.5, borderaxespad=0.3)
    fig.text(0.075, 0.985, f"FreeInference ({n_weeks} weeks), returning requests, simulation, T1 constants", ha="left", va="top",
             fontsize=7.5, color=INK2)
    s.notes.append(T1_NOTE)
    s.write(fig, "fig8_bridge.png")


# ================================================================ fig9: FreeInference X1 headroom vs CPU-tier capacity (T5 + T1)
def fig9():
    s = Src("fig9_fi_capacity")
    T5 = "results/icpe_h200/t5/p12/x1/x1fi_headroom_components.csv"
    T1F = "results/icpe_h200/t1/p12/x1/x1fi_headroom_components.csv"
    d5 = s.csv(T5, related_cols=("src_runs",))
    d1 = s.csv(T1F, related_cols=("src_runs",))
    for d, nm in ((d5, "T5"), (d1, "T1")):
        if set(d.restore) != {"cake"} or set(d.trace) != {"freeinference"}:
            raise SystemExit(f"{nm}: unexpected restore/trace values")
    # T5's 128 GiB cells must equal T1's (the registered check; verified again here)
    key = ["cpu_GiB", "slow", "gpu", "component"]
    m = d5[d5.cpu_GiB == 128].merge(d1[d1.cpu_GiB == 128], on=key, suffixes=("_t5", "_t1"))
    s.check("T5 vs T1 at 128 GiB: cells compared", len(m), 24, 0, "4 components x 3 slow x 2 GPU")
    s.check("T5 vs T1 at 128 GiB: max |headroom diff|", float((m.headroom_t5 - m.headroom_t1).abs().max()), 0.0, 0.0, T1F)
    s.check("T5 vs T1 at 128 GiB: best_rival differs (cells)", float((m.best_rival_t5 != m.best_rival_t1).sum()), 0.0, 0, T1F)
    a = pd.concat([d5[d5.cpu_GiB < 128], d1], ignore_index=True)
    caps = np.array(sorted(a.cpu_GiB.unique()))
    tiers = ["b32", "ssd", "nfs"]
    curves, argmax = {}, {}
    for c in ("O-ORD", "O-PF"):
        for sl in tiers:
            sub = a[(a.component == c) & (a.slow == sl)]
            idx = sub.groupby("cpu_GiB").headroom.idxmax()
            best = sub.loc[idx].set_index("cpu_GiB").sort_index()
            if list(best.index) != list(caps):
                raise SystemExit(f"{c} {sl}: missing capacities")
            curves[(c, sl)] = best.headroom
            argmax[f"{c}|{sl}"] = [dict(cpu_GiB=int(ci), headroom=float(r.headroom), gpu=r.gpu, best_rival=r.best_rival,
                                        oracle_policy=r.oracle_policy, source="T5" if ci < 128 else "T1")
                                   for ci, r in best.iterrows()]
    ordn = curves[("O-ORD", "nfs")]
    for cpu, q in ((16, 5.9), (32, 11.8), (64, 3.6)):
        s.check(f"O-ORD NFS max over GPU @ {cpu} GiB (%)", round(100 * float(ordn.loc[cpu]), 1), q, 0.05, "coordinator / T5 doc")
    s.check("O-ORD NFS max over GPU @ 128 GiB (%)", round(100 * float(ordn.loc[128]), 2), 0.30, 0.005, "coordinator / T1 B3")
    s.check("cells >= 15% (all components, 16-1024 GiB)", float((a.headroom >= 0.15).sum()), 0.0, 0, "coordinator: nothing >= 15%")
    rep_ = s.json("results/icpe_h200/t5/t5_report.json")
    # T4 overlay: oracle order vs LRU only (slow tier + Cake), NFS class, GPU idle, returning requests
    t4 = s.csv("results/icpe_h200/t4/t4_cells.csv", related_cols=("sweep_run_id",))
    t4n = t4[(t4.slow == "nfs") & (t4.gpu == "idle") & (t4.subset == "ret")].sort_values("cpu_GiB")
    t4n_in = t4n[(t4n.cpu_GiB >= caps.min()) & (t4n.cpu_GiB <= caps.max())]
    s.values = {
        "units": "oracle headroom over the best symmetric rival (Cake), fraction; max over GPU idle/busy per slow tier",
        "capacities_GiB": [int(x) for x in caps],
        "O-ORD": {sl: {int(k): float(v) for k, v in curves[("O-ORD", sl)].items()} for sl in tiers},
        "O-PF": {sl: {int(k): float(v) for k, v in curves[("O-PF", sl)].items()} for sl in tiers},
        "argmax_cells": argmax,
        "T4_obel_vs_LRU_nfs_idle_cake": {int(r.cpu_GiB): float(r.G_cake) for r in t4n.itertuples()},
        "t5_report_run_id": rep_["run_id"],
    }
    s.notes.append("16-64 GiB from T5 (docs/h200/icpe/T5_fi_small_cpu.md), 128-1024 GiB from T1; T5's 128 GiB cells equal T1's "
                   "(checked here). Each point = max over GPU idle/busy for that slow tier; FreeInference first weeks, all returning "
                   "requests, Cake, H200 CPU-tier constants (simulation).")
    s.notes.append("Hollow markers: T4's G_cake (oracle order O-BEL vs LRU only, NFS class, GPU idle, returning requests); "
                   f"T4's 8 GiB point ({100 * float(t4n[t4n.cpu_GiB == 8].G_cake.iloc[0]) if (t4n.cpu_GiB == 8).any() else float('nan'):.1f}%) "
                   "is outside the plotted 16-1024 GiB range and not drawn. It is GPU idle only, while the lines take the max over GPU states.")

    col = {"ssd": ORANGE, "b32": MUTED, "nfs": VIOLET}
    name = {"ssd": "SSD class", "b32": "b32", "nfs": "NFS class"}
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.3), sharey=True,
                             gridspec_kw=dict(left=0.09, right=0.99, top=0.79, bottom=0.15, wspace=0.08))
    for ax, c, lw, ms in ((axes[0], "O-ORD", 2.0, 4.5), (axes[1], "O-PF", 1.2, 3.5)):
        ax.set_xscale("log", base=2)
        ax.set_xticks(caps)
        ax.set_xticklabels([f"{int(x):,}" for x in caps])
        ax.minorticks_off()
        ax.set_xlim(caps.min() / 1.3, caps.max() * 1.3)
        ax.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.axhline(0, color=BASELINE, lw=0.8, zorder=1)
        for th in (5, 15):
            ax.axhline(th, **THRESH_STYLE)
            ax.text(caps.max() * 1.25, th + 0.6, f"{th}%", ha="right", va="bottom", fontsize=7, color=INK2)
        for sl in tiers:
            y = curves[(c, sl)]
            ax.plot(caps, 100 * y.to_numpy(), color=col[sl], lw=lw, marker=TIER_MARK[sl], ms=ms, mec=WHITE, mew=0.6, zorder=3,
                    label=name[sl])
        ax.axvspan(caps.min() / 1.3, 96, color=BAND, zorder=0, lw=0)
        ax.text(16 * 1.02, 0.98, "T5", transform=ax.get_xaxis_transform(), fontsize=7, color=INK2, va="top")
        ax.text(128 * 1.05, 0.98, "T1", transform=ax.get_xaxis_transform(), fontsize=7, color=INK2, va="top")
        ax.set_xlabel("CPU tier capacity (GiB)")
    axes[0].scatter(t4n_in.cpu_GiB, 100 * t4n_in.G_cake, s=34, facecolor="none", edgecolor=VIOLET, linewidth=1.2, zorder=4,
                    label="NFS class, vs LRU only (T4, GPU idle)")
    axes[0].set_title("O-ORD: eviction order (main)", loc="left", fontsize=8, color=INK)
    axes[1].set_title("O-PF: prefetch before return", loc="left", fontsize=8, color=INK)
    axes[0].set_ylabel("headroom over best symmetric rival (%)")
    lo = min(-5.0, 100 * min(float(y.min()) for y in curves.values()) - 3)
    hi = max(30.0, 100 * float(t4n_in.G_cake.max()) + 3)
    axes[0].set_ylim(lo, hi)
    h, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="upper left", bbox_to_anchor=(0.08, 0.995), ncol=4, fontsize=7.5, handletextpad=0.4, columnspacing=1.2)
    fig.text(0.09, 0.855, "FreeInference (agent trace), simulation, Cake, T1 constants; each point = max over GPU idle / busy",
             fontsize=7, color=INK2, ha="left", va="bottom")
    s.notes.append(T1_NOTE)
    s.write(fig, "fig9_fi_capacity.png")


TIER_MARK = {"b32": "s", "ssd": "^", "nfs": "D"}   # slow tier -> marker (fig8 panel c, fig9), so identity is not hue-only

FIGURES = {"fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6, "fig7": fig7, "fig8": fig8, "fig9": fig9}

if __name__ == "__main__":
    want = sys.argv[1:] or list(FIGURES)
    for w in want:
        print(f"[{w}]")
        FIGURES[w]()
