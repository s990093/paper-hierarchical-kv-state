"""Exploratory studies (report-only): sweep (09), long context (10), token drop + match (14).

Figures F10-F19. Every number is read from results/ (see figs/SOURCES.md).
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from common import (DOUBLE, FMT, OI, TIER_COL, WHO, begin, fmt_family, load_json, panel_label, plt, read_csv,
                    read_text, refline, rp, save)

MODELS = ["llama", "qwen"]
MLAB = {"llama": "Llama-3.1-8B", "qwen": "Qwen2.5-7B-1M"}
LENS = ["32K", "64K", "128K"]
LEN_TOK = {"32K": 32768, "64K": 65536, "128K": 129024}
MAIN3 = {"llama": ["fp8_token", "int8_token", "int4_kivi"], "qwen": ["fp8_head", "int8_token", "int4_kivi"]}
SHORT = {"fp8_token": "FP8 (token)", "fp8_head": "FP8 (head)", "fp8e5_head": "FP8-e5m2", "fp8k0_token": "FP8-K0",
         "fp8k0_head": "FP8-K0", "int8_token": "INT8", "int4_kivi": "INT4-KIVI", "int4_token": "INT4 token",
         "int4h_token": "INT4+Had.", "mxfp4": "MXFP4", "nvfp4": "NVFP4", "int2_kivi": "INT2-KIVI",
         "int2h_g32": "INT2+Had.", "kv-int4_kivi-int2_kivi": "K4V2", "kv-int8_token-int4_kivi": "K8V4"}


def fcol(fmt):
    return FMT[fmt_family(fmt)]


# --------------------------------------------------------------------------------------------- F10
def f10_sweep_screen():
    """Sweep screen (stage 41): whole-context formats and structured placement vs random."""
    begin("F10_sweep_screen")
    A = rp("m10_sweep_mi300x", "analysis")
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.45), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    mk = {"llama": "o", "qwen": "s"}
    for m in MODELS:
        rows = [r for r in read_csv(f"{A}/agg_{m}.csv") if r["readout"] == "A" and r["policy"] == "ALL"]
        for r in rows:
            b = float(r["bits_per_value"])
            kl = float(r["kl"])
            d = 100 * float(r["d_first_ok"])
            c = fcol(r["fmt"])
            axes[0].scatter([b], [kl], marker=mk[m], s=16, color=c if m == "llama" else "white", edgecolor=c, lw=0.8,
                            zorder=3)
            axes[1].scatter([b], [d], marker=mk[m], s=16, color=c if m == "llama" else "white", edgecolor=c, lw=0.8,
                            zorder=3)
        # labels for the main formats only (selective)
        for r in rows:
            if m == "qwen" and r["fmt"] in ("fp8_head", "int4_kivi", "int4_token", "int2_kivi"):
                axes[0].annotate("Qwen 4-bit per-token" if r["fmt"] == "int4_token" else SHORT[r["fmt"]], (float(r["bits_per_value"]), float(r["kl"])), xytext=(3, 2),
                                 textcoords="offset points", fontsize=5.6, color="#333333")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("bits per value (blocks 1-63)")
    axes[0].set_ylabel("mean KL vs E0 (teacher forcing)")
    axes[0].set_title("All low precision: error")
    axes[1].set_xlabel("bits per value")
    axes[1].set_ylabel("first-token accuracy vs E0 (pp)")
    axes[1].set_title("All low precision: accuracy")
    refline(axes[1], 0, color="#333333", ls="-", lw=0.5)
    refline(axes[1], -2, color=OI["verm"], ls="--")
    for ax in axes[:2]:
        ax.set_xticks([3, 4, 5, 6, 8])
    hs = [plt.Line2D([], [], marker="o", ls="", color="#555555", ms=4, label="Llama (filled)"),
          plt.Line2D([], [], marker="s", ls="", mfc="white", color="#555555", ms=4, label="Qwen (hollow)")]
    fam = ["fp8", "fp8k0", "int8", "int4", "int4tok", "int2"]
    lab = {"fp8": "FP8", "fp8k0": "FP8-K0", "int8": "INT8 / K8V4", "int4": "INT4-KIVI", "int4tok": "4-bit per-token",
           "int2": "INT2 / K4V2"}
    hs += [plt.Line2D([], [], marker="o", ls="", color=FMT[f], ms=4, label=lab[f]) for f in fam]
    axes[1].legend(handles=hs, fontsize=5.6, loc="lower right", ncol=1, handletextpad=0.2, borderpad=0.2)
    ax = axes[2]
    nwin = {}
    for m in MODELS:
        t = [r for r in read_csv(f"{A}/tests_{m}.csv") if r["winner"] == "True"]
        x = [100 * float(r["rel_delta"]) for r in t]
        y = [100 * (float(r["first_ok_x"]) - float(r["first_ok_rand"])) for r in t]
        c = [fcol(r["fmt"]) for r in t]
        ax.scatter(x, y, marker=mk[m], s=7, c=c if m == "llama" else "none", edgecolors=c, lw=0.5, alpha=0.75)
        nwin[m] = len(t)
    refline(ax, 0, color="#333333", ls="-", lw=0.5)
    refline(ax, 2, color=OI["verm"], ls="--")
    ax.set_xlabel("KL reduction vs same-fraction random (%)")
    ax.set_ylabel("first-token accuracy vs random (pp)")
    ax.set_title("Structured placement 'winners'")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color="#555555", ms=3.5,
                                  label=f"Llama, {nwin['llama']} winning cells"),
                       plt.Line2D([], [], marker="s", ls="", mfc="white", color="#555555", ms=3.5,
                                  label=f"Qwen, {nwin['qwen']} winning cells")], fontsize=5.6, loc="upper right")
    for i, a in enumerate(axes):
        panel_label(a, "abc"[i])
    fig.tight_layout(w_pad=0.8)
    return save(fig, "F10_sweep_screen")


# --------------------------------------------------------------------------------------------- F11
_ROW = re.compile(r"^\|\s*(\S+@\S+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([+-]?[0-9.]+)\s*\[([+-]?[0-9.]+),\s*([+-]?[0-9.]+)\]")


def _detail_rows(md):
    """Parse confirm/detail_A.md: {model: {config: (GiB, acc, diff, lo, hi)}}."""
    out, cur = {}, None
    for line in md.splitlines():
        if line.startswith("## "):
            cur = line[3:].split("（")[0].strip()
            out[cur] = {}
        mm = _ROW.match(line)
        if mm and cur:
            cells = [x.strip() for x in line.strip().strip("|").split("|")]
            eps_a = cells[4] == "過" and cells[5] == "過"          # reading-A eps: main and per-family
            out[cur][mm.group(1)] = tuple(float(mm.group(i)) for i in range(2, 7)) + (eps_a,)
    return out


def _cfg_short(cfg):
    pol, fmt = cfg.split("@")
    pol = pol.replace("SW-", "").replace("E1s", "ALL")
    return f"{pol} {SHORT.get(fmt, fmt)}"


def f11_sweep_confirm():
    """Sweep confirm (stage 43): 12 finalists vs random and vs E0; frontier (whole-context) points."""
    begin("F11_sweep_confirm")
    C = rp("m10_sweep_mi300x", "confirm")
    det = _detail_rows(read_text(f"{C}/detail_A.md"))
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.9), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    ax = axes[0]
    items = []
    for m in MODELS:
        cj = load_json(f"{C}/confirm_{m}.json")
        for cfg, v in cj["finalists"].items():
            r = v["vs_random"]["A"]
            items.append((m, cfg, 100 * r["diff"], 100 * r["ci95"][0], 100 * r["ci95"][1], r["holm"]["reject"],
                          v["confirmed"]))
    for k, (m, cfg, d, lo, hi, rej, conf) in enumerate(items):
        y = len(items) - 1 - k
        c = fcol(cfg.split("@")[1])
        ax.errorbar([d], [y], xerr=[[d - lo], [hi - d]], fmt="o" if m == "llama" else "s", ms=3.5, color=c,
                    mfc=c if rej else "white", elinewidth=0.8, lw=0)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([f"{'L' if m == 'llama' else 'Q'}: {_cfg_short(cfg)}" for m, cfg, *_ in items][::-1],
                       fontsize=5.8)
    refline(ax, 0, axis="x", color="#333333", ls="-", lw=0.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("accuracy vs random, 5 seeds (pp)")
    ax.set_title("Finalist vs random (reading A)")
    ax.text(0.98, 0.02, "filled = Holm-significant\nconfirmed: 0 of 12", transform=ax.transAxes, fontsize=5.8,
            ha="right", va="bottom")
    panel_label(ax, "a")
    for j, m in enumerate(MODELS):
        ax = axes[1 + j]
        cj = load_json(f"{C}/confirm_{m}.json")
        for cfg, (gib, acc, d, lo, hi, eps_ok) in det[m].items():
            c = fcol(cfg.split("@")[1])
            is_all = cfg.startswith("E1s@")
            ax.errorbar([gib], [d], yerr=[[d - lo], [hi - d]], fmt="D" if is_all else ("o" if m == "llama" else "s"),
                        ms=3.6, color=c, mfc=c if eps_ok else "white", elinewidth=0.7, lw=0)
            if is_all:
                ax.annotate(SHORT.get(cfg.split("@")[1]), (gib, d), xytext=(4, -2), textcoords="offset points",
                            fontsize=5.6)
        refline(ax, -2, color=OI["verm"], ls="--")
        refline(ax, 0, color="#333333", ls="-", lw=0.5)
        ax.set_xlabel("stored KV per 32K context (GiB)")
        ax.set_title(f"{'Llama' if m == 'llama' else 'Qwen'}: quality vs storage")
        panel_label(ax, "bc"[j])
    axes[1].set_ylabel("accuracy vs E0 (pp, 95% CI)")
    axes[2].text(0.98, 0.98, "diamond = all-low-precision (frontier)\nfilled = passes eps (reading A)",
                 transform=axes[2].transAxes, fontsize=5.6, ha="right", va="top")
    fig.tight_layout(w_pad=1.2)
    return save(fig, "F11_sweep_confirm")


# --------------------------------------------------------------------------------------------- long-context helpers
def _lc(m):
    return load_json(rp("m10_longctx_mi300x", "analysis", f"longctx_{m}.json"))


# --------------------------------------------------------------------------------------------- F12
def f12_longctx_accuracy():
    """Long context: generated accuracy vs length (E0 + whole-context formats) and how much low precision Cake reads."""
    begin("F12_longctx_accuracy")
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.35), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    for j, m in enumerate(MODELS):
        d = _lc(m)
        ax = axes[j]
        series = [("E0", FMT["bf16"], "BF16 (E0)")] + [(f, fcol(f), SHORT[f]) for f in MAIN3[m]]
        for s_i, (key, col, lab) in enumerate(series):
            xs = np.arange(len(LENS)) + (s_i - 1.5) * 0.07
            v = [d["acc"][L][key] for L in LENS]
            y = np.array([100 * x["mean"] for x in v])
            lo = np.array([100 * x["lo"] for x in v])
            hi = np.array([100 * x["hi"] for x in v])
            ax.errorbar(xs, y, yerr=[y - lo, hi - y], marker="o", ms=3.2, color=col, lw=0.9, elinewidth=0.6,
                        label=lab)
        ax.set_xticks(range(len(LENS)))
        ax.set_xticklabels(LENS)
        ax.set_ylim(-3, 100)
        ax.set_xlabel("context length")
        ax.set_title(f"{MLAB[m]}")
        ax.legend(fontsize=5.8, loc="lower left")
        panel_label(ax, "ab"[j])
    axes[0].set_ylabel("generated accuracy (%)")
    fig.text(0.01, -0.05, "a, b: reading A, blocks 1..last low precision (ALL); 48 items per length (T2-T5, turn 1), "
             "95% bootstrap CI.\nc: meeting points simulated (m10_sweep_sim) from same-length reference grids; Llama and Qwen coincide.", fontsize=6.0)
    ax = axes[2]
    for m in MODELS:
        d = _lc(m)
        f = "int4_kivi"
        for ro, ls in (("B_idle", "-"), ("B_busy", "--")):
            ys = []
            for L in LENS:
                rb = d["readout_b"][f"{L}|{f}|{ro}"]["lp_read"]
                tot = d["readout_b"][f"{L}|{f}|A"]["lp_read"]
                ys.append(100 * rb / tot)
            ax.plot(np.arange(len(LENS)) + (0.04 if m == "qwen" else -0.04), ys, ls=ls,
                    marker="o" if m == "llama" else "s", ms=3.2,
                    color="#444444" if m == "llama" else OI["purple"],
                    mfc="#444444" if m == "llama" else "white", lw=0.9,
                    label=f"{'Llama' if m == 'llama' else 'Qwen'}, Cake GPU {ro.split('_')[1]}")
    ax.set_xticks(range(len(LENS)))
    ax.set_xticklabels(LENS)
    ax.set_ylim(50, 100)
    ax.set_xlabel("context length")
    ax.set_ylabel("INT4 blocks read (% of stored)")
    ax.set_title("Reading B (b32), INT4-KIVI")
    ax.legend(fontsize=5.6, loc="lower right")
    panel_label(ax, "c")
    fig.tight_layout()
    return save(fig, "F12_longctx_accuracy")


# --------------------------------------------------------------------------------------------- F13
def f13_longctx_harm():
    """Harm vs length (whole-context format minus E0, generated accuracy) for all 13 formats."""
    begin("F13_longctx_harm")
    cols = LENS + ["128K−32K"]
    cmap = LinearSegmentedColormap.from_list("ov", [OI["verm"], "#f4f4f4", OI["blue"]])
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.3))
    for j, m in enumerate(MODELS):
        d = _lc(m)
        fmts = sorted(d["harm"], key=lambda f: (fmt_family(f), f))
        M = np.array([[100 * d["harm"][f]["gen"][c]["mean"] for c in cols] for f in fmts])
        ax = axes[j]
        im = ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30), aspect="auto")
        for i, f in enumerate(fmts):
            for k, c in enumerate(cols):
                v = d["harm"][f]["gen"][c]
                txt = f"{100 * v['mean']:+.0f}"
                excl0 = v["lo"] > 0 or v["hi"] < 0
                if k == 3 and v.get("grows_with_length"):
                    txt += "↑"
                ax.text(k, i, txt, ha="center", va="center", fontsize=5.8,
                        color="white" if abs(100 * v["mean"]) > 22 else "#111111",
                        fontweight="bold" if excl0 else "normal")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(["32K", "64K", "128K", "128K\nminus 32K"])
        ax.set_yticks(range(len(fmts)))
        ax.set_yticklabels([SHORT[f] for f in fmts], fontsize=6.2)
        ax.grid(False)
        ax.axvline(2.5, color="white", lw=2)
        ax.set_title(f"{MLAB[m]}")
        panel_label(ax, "ab"[j])
    fig.subplots_adjust(wspace=0.42)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("accuracy, all-low-precision minus E0 (pp)\n(colour clipped at +/-30)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    fig.text(0.01, -0.01, "bold = 95% CI excludes 0;  ↑ = harm grows with length (128K-32K CI below 0); "
             "n = 48 items per length, paired by seed.", fontsize=6.2)
    fig.text(0.01, -0.045, "Qwen rows at 0% accuracy show +21 in the last column only because E0 itself drops with "
             "length (floor effect), not because the format recovers.", fontsize=6.2)
    return save(fig, "F13_longctx_harm")


# --------------------------------------------------------------------------------------------- F14
def f14_longctx_position():
    """Position at length: 25% low precision at front/middle/back minus random (R0)."""
    begin("F14_longctx_position")
    mk = {"F": ">", "M": "D", "B": "<"}
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE, 3.6), sharex=True)
    for r, m in enumerate(MODELS):
        d = _lc(m)
        for c, f in enumerate(MAIN3[m]):
            ax = axes[r, c]
            col = fcol(f)
            for p_i, pol in enumerate("FMB"):
                xs = np.arange(len(LENS)) + (p_i - 1) * 0.2
                v = [d["position"][f"{L}|{f}|{pol}"]["gen_diff"] for L in LENS]
                y = np.array([100 * x["mean"] for x in v])
                lo = np.array([100 * x["lo"] for x in v])
                hi = np.array([100 * x["hi"] for x in v])
                ax.errorbar(xs, y, yerr=[y - lo, hi - y], fmt=mk[pol], ms=3.4, color=col, elinewidth=0.7, lw=0,
                            label={"F": "front 25%", "M": "middle 25%", "B": "back 25%"}[pol])
            refline(ax, 0, color="#333333", ls="-", lw=0.5)
            for yv in (2, -2):
                refline(ax, yv, color=OI["verm"], ls="--", lw=0.6)
            ax.set_xticks(range(len(LENS)))
            ax.set_xticklabels(LENS)
            broken = (m == "qwen" and f == "fp8_head")
            ax.set_title(f"{'Llama' if m == 'llama' else 'Qwen'}: {SHORT[f]}" + ("  (broken format)" if broken else ""),
                         fontsize=7.3)
            if not broken:
                ax.set_ylim(-10, 10)
            if c == 0:
                ax.set_ylabel("accuracy vs random R0 (pp)")
    axes[0, 0].legend(fontsize=5.8, loc="lower left")
    for ax in axes[1]:
        ax.set_xlabel("context length")
    fig.text(0.01, -0.01, "reading A, 25% of blocks low precision (block 0 BF16), generated accuracy, 48 items/length, "
             "95% bootstrap CI; dashed = +/-2 pp. Note the wider y range for Qwen FP8 (head).", fontsize=6.0)
    fig.tight_layout()
    return save(fig, "F14_longctx_position")


# --------------------------------------------------------------------------------------------- F15
DS_WHO = {"KEYDIFF": "write", "KNORM": "write", "EXPATT": "write", "SNAP": "write", "TOVA": "write", "H2O": "write",
          "ADA": "write", "PYR": "write", "QRY": "query", "STREAM": "position", "NEW": "position", "RND0": "random",
          "RND1": "random", "RND2": "random", "ORC": "oracle", "NONE": "control", "MASK0": "control",
          "MASK0H": "control"}
WHO_LAB = {"write": "write-time score (query-agnostic)", "query": "restore-time, uses turn-1 question (QRY)",
           "position": "position (StreamingLLM)", "random": "random", "oracle": "oracle (keeps turn-1 answer block)",
           "control": "control (mask path, nothing dropped)"}


def _ds_parse(name):
    _, gran, meth, frac = name.split("-")
    return gran, meth, int(frac)


def f15_drop_ds2():
    """Token drop confirm (DS2): accuracy vs E0 per method/granularity/fraction and the turn-2 penalty."""
    begin("F15_drop_ds2")
    rep = load_json(rp("m10_drop_mi300x", "ds2", "ds2_report.json"))
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 5.0), gridspec_kw=dict(width_ratios=[1.35, 1]))
    for r, m in enumerate(MODELS):
        d = rep[m]
        names = sorted(d, key=lambda n: (_ds_parse(n)[0] != "T", ["control", "write", "query", "oracle", "position",
                                                                    "random"].index(DS_WHO[_ds_parse(n)[1]]),
                                         _ds_parse(n)[1], _ds_parse(n)[2]))
        names = [n for n in names if _ds_parse(n)[1] != "NONE"] + [n for n in names if _ds_parse(n)[1] == "NONE"]
        ax = axes[r, 0]
        for k, n in enumerate(names):
            g, meth, fr = _ds_parse(n)
            e = d[n]["eps"]
            y = len(names) - 1 - k
            pt, lo, hi = 100 * e["diff"], 100 * e["ci95"][0], 100 * e["ci95"][1]
            col = WHO[DS_WHO[meth]]
            ax.errorbar([pt], [y], xerr=[[pt - lo], [hi - pt]], fmt="o" if g == "T" else "s", ms=3.3, color=col,
                        mfc=col if e["eps_pass"] else "white", elinewidth=0.7, lw=0)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([f"{_ds_parse(n)[0]}  {_ds_parse(n)[1]}  {_ds_parse(n)[2]}%" for n in names][::-1],
                           fontsize=5.6)
        refline(ax, 0, axis="x", color="#333333", ls="-", lw=0.5)
        refline(ax, -2, axis="x", color=OI["verm"], ls="--")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{MLAB[m]}: dropped tokens vs E0 (reading A, both turns)")
        ax.set_xlabel("accuracy vs E0 (pp, 95% CI); filled = passes eps")
        ax.set_xlim(-80, 6)
        ax = axes[r, 1]
        sel = [n for n in names if _ds_parse(n)[1] not in ("NONE",)]
        for k, n in enumerate(sel):
            g, meth, fr = _ds_parse(n)
            col = WHO[DS_WHO[meth]]
            y = len(sel) - 1 - k
            ax.scatter([100 * d[n]["turn1"]], [y], marker="o" if g == "T" else "s", s=12, facecolor="white",
                       edgecolor=col, lw=0.8, zorder=3)
            ax.scatter([100 * d[n]["turn2"]], [y], marker="o" if g == "T" else "s", s=12, color=col, zorder=4)
            ax.plot([100 * d[n]["turn1"], 100 * d[n]["turn2"]], [y, y], color=col, lw=0.7)
        ax.set_yticks(range(len(sel)))
        ax.set_yticklabels([f"{_ds_parse(n)[0]} {_ds_parse(n)[1]} {_ds_parse(n)[2]}%" for n in sel][::-1],
                           fontsize=5.6)
        refline(ax, 0, axis="x", color="#333333", ls="-", lw=0.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(-80, 10)
        ax.set_title("Turn 1 (hollow) -> turn 2 (filled)")
        ax.set_xlabel("accuracy vs E0 per turn (pp, point estimate)")
    hs = [plt.Line2D([], [], marker="o", ls="", color=WHO[w], ms=4, label=WHO_LAB[w])
          for w in ("write", "query", "oracle", "position", "random", "control")]
    hs += [plt.Line2D([], [], marker="o", ls="", color="#555555", ms=4, label="T = per-head tokens"),
           plt.Line2D([], [], marker="s", ls="", color="#555555", ms=4, label="B = whole 512-token blocks")]
    fig.legend(handles=hs, loc="lower center", ncol=4, fontsize=5.8, bbox_to_anchor=(0.5, -0.045))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "F15_drop_ds2")


# --------------------------------------------------------------------------------------------- F16
def f16_drop_ds1():
    """Token drop screen (DS1, teacher forcing): gold log-prob loss vs drop fraction, by granularity."""
    begin("F16_drop_ds1")
    t = load_json(rp("m10_drop_mi300x", "ds1", "ds1_table.json"))
    grans = ["T", "P", "B"]
    gl = {"T": "per-head tokens (T)", "P": "16-token pages (P)", "B": "512-token blocks (B)"}
    hi_m = {"KEYDIFF": "write", "QRY": "query", "STREAM": "position", "ORC": "oracle"}
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE, 3.7), sharex=True, sharey=True)
    for r, m in enumerate(MODELS):
        for c, g in enumerate(grans):
            ax = axes[r, c]
            keys = [k for k in t if k.startswith(f"{m}|{g}|")]
            meths = sorted({k.split("|")[2] for k in keys} - {"MASK0", "MASK0H"})
            for meth in meths:
                pts = sorted((float(k.split("|")[3]), t[k]) for k in keys if k.split("|")[2] == meth)
                if not pts:
                    continue
                x = [100 * p[0] for p in pts]
                y = [p[1]["d_gold_lp"][0] for p in pts]
                lo = [p[1]["d_gold_lp"][1] for p in pts]
                hi = [p[1]["d_gold_lp"][2] for p in pts]
                if meth in hi_m:
                    col = WHO[hi_m[meth]]
                    ax.plot(x, y, marker="o", ms=2.6, color=col, lw=1.0, label=meth, zorder=4)
                    ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0, zorder=2)
                elif meth.startswith("RND"):
                    ax.plot(x, y, color=WHO["random"], lw=0.8, ls="--", label="random (3 seeds)" if meth == "RND0"
                            else None, zorder=3)
                else:
                    ax.plot(x, y, color="#9ecae1", lw=0.6, label="other write-time scores" if meth == meths[0]
                            or (meth == "ADA") else None, zorder=1)
            ax.set_yscale("symlog", linthresh=1)
            refline(ax, 0, color="#333333", ls="-", lw=0.5)
            if r == 0:
                ax.set_title(gl[g])
            if c == 0:
                ax.set_ylabel(f"{MLAB[m]}\ngold log-prob vs E0 (nats)")
            if r == 1:
                ax.set_xlabel("dropped fraction (%)")
            ax.set_xticks([10, 25, 50, 75, 90])
    h, l = [], []
    for a in axes.flat:
        hx, lx = a.get_legend_handles_labels()
        h += hx; l += lx
    seen, hh, ll = set(), [], []
    for a, b in zip(h, l):
        if b not in seen:
            seen.add(b); hh.append(a); ll.append(b)
    fig.legend(hh, ll, loc="lower center", ncol=7, fontsize=6.0, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "F16_drop_ds1")


# --------------------------------------------------------------------------------------------- F17
def _cells(sp, key):
    v = sp[key]
    if not v["usable"]:
        return None
    c = v["cells"]
    return dict(c) if isinstance(c, list) else c


def f17_drop_speed():
    """D3: restore time with dropped blocks / without, by drop position, per tier (GPU idle)."""
    begin("F17_drop_speed")
    sp = load_json(rp("m10_match_mi300x", "speed_14.json"))
    tiers = [("cpu", "loadall", "CPU\nload-all"), ("cpu", "cake", "CPU\nCake"), ("b16", "cake", "b16"),
             ("b32", "cake", "b32"), ("b48", "cake", "b48"), ("nfs", "cake", "NFS-class")]
    pos_col = {"F": OI["orange"], "B": OI["blue"], "R": OI["grey"]}
    pos_mk = {"F": ">", "B": "<", "R": "o"}
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 3.7), sharex=True, sharey=True)
    for r, m in enumerate(MODELS):
        cint = _cells(sp, f"d3/{m}_int4") or {}
        cnfs = _cells(sp, f"d3/{m}_nfs") or {}
        allc = {**cint, **cnfs}
        for c, base in enumerate(("E0", "E1s")):
            ax = axes[r, c]
            for t_i, (tier, mode, _) in enumerate(tiers):
                k0 = f"32768|{tier}|idle|{base}|{mode}"
                if k0 not in allc:
                    continue
                t0 = allc[k0]["median"]
                for p_i, pol in enumerate("FBR"):
                    for frac in (25, 50):
                        k = f"32768|{tier}|idle|{base}-D{pol}{frac}|{mode}"
                        if k not in allc:
                            continue
                        x = t_i + (p_i - 1) * 0.22 + (0.05 if frac == 50 else -0.05)
                        ax.scatter([x], [allc[k]["median"] / t0], marker=pos_mk[pol], s=16,
                                   color=pos_col[pol] if frac == 50 else "white", edgecolor=pos_col[pol], lw=0.8,
                                   zorder=3)
            for yv, lab in ((1.0, "no saving"), (0.75, "proportional to 25%"), (0.5, "proportional to 50%")):
                refline(ax, yv, color="#333333" if yv == 1 else "#888888", ls="-" if yv == 1 else ":", lw=0.6)
            ax.set_xticks(range(len(tiers)))
            ax.set_xticklabels([t[2] for t in tiers], fontsize=6.3)
            ax.set_ylim(0.3, 1.1)
            ax.set_title(f"{'Llama' if m == 'llama' else 'Qwen'}: base {'BF16 (E0)' if base == 'E0' else 'INT4-KIVI (E1s)'}"
                         + ("" if base == "E0" else "  [NFS: not run]"), fontsize=7.3)
    fig.supylabel("restore time, dropped / not dropped", fontsize=7.5, x=0.005)
    hs = [plt.Line2D([], [], marker=pos_mk[p], ls="", color=pos_col[p], ms=4, label=l)
          for p, l in (("F", "drop oldest (front)"), ("B", "drop newest (back)"), ("R", "drop random"))]
    hs += [plt.Line2D([], [], marker="o", ls="", color="#555555", ms=4, label="50% dropped (filled)"),
           plt.Line2D([], [], marker="o", ls="", mfc="white", color="#555555", ms=4, label="25% dropped (hollow)")]
    fig.legend(handles=hs, loc="lower center", ncol=5, fontsize=6.0, bbox_to_anchor=(0.5, -0.03))
    fig.text(0.99, 0.995, "32K, GPU idle, median of 10 restores; dotted = proportional saving", fontsize=6,
             ha="right", va="top", color="#444444")
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    return save(fig, "F17_drop_speed")


# --------------------------------------------------------------------------------------------- F18
M2_FMTS = {"llama": ["fp8_token", "int4_kivi", "mxfp4", "kv-int4_kivi-int2_kivi", "int2_kivi"],
           "qwen": ["fp8_head", "int4_kivi", "mxfp4", "kv-int4_kivi-int2_kivi", "int2_kivi"]}


def f18_match_formats_units():
    """M2: restore time vs bits/value per slow tier; M1: restore unit 256/128 relative to 512."""
    begin("F18_match_m2_m1")
    sp = load_json(rp("m10_match_mi300x", "speed_14.json"))
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.45), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    tiers = ["b16", "b32", "b48", "nfs"]
    for j, m in enumerate(MODELS):
        agg = {r["fmt"]: float(r["bits_per_value"]) for r in
               read_csv(rp("m10_sweep_mi300x", "analysis", f"agg_{m}.csv")) if r["policy"] == "ALL"
               and r["readout"] == "A"}
        ax = axes[j]
        for t in tiers:
            for st, ls in (("idle", "-"), ("busy", "--")):
                xs, ys = [], []
                for f in M2_FMTS[m]:
                    c = _cells(sp, f"m2/{m}_{f}")
                    if not c:
                        continue
                    k = f"32768|{t}|{st}|E1s|cake"
                    if k in c:
                        xs.append(agg[f])
                        ys.append(c[k]["median"])
                o = np.argsort(xs)
                ax.plot(np.array(xs)[o], np.array(ys)[o], ls=ls, marker="o", ms=2.8, color=TIER_COL[t], lw=0.9,
                        mfc=TIER_COL[t] if st == "idle" else "white", label=f"{t}, {st}" if j == 0 else None)
        ax.set_yscale("log")
        ax.set_yticks([0.2, 0.5, 1, 2, 3])
        ax.set_yticklabels(["0.2", "0.5", "1", "2", "3"])
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel("bits per value (K/V average)")
        ax.set_title(f"M2 {'Llama' if m == 'llama' else 'Qwen'}: all low precision, Cake")
        ax.set_xticks([3, 4, 5, 8])
        if j == 0:
            ax.set_ylabel("restore time, median (s)")
        panel_label(ax, "ab"[j])
    ax = axes[2]
    tiers1 = ["cpu", "b16", "b32", "b48"]
    for m in MODELS:
        base = _cells(sp, f"m1/{m}_u512")
        for u, mk in (("u256", "o"), ("u128", "s")):
            c = _cells(sp, f"m1/{m}_{u}")
            for cfg, xo in (("E0", -0.18), ("E1s", 0.18)):
                xs, ys = [], []
                for t_i, t in enumerate(tiers1):
                    k = f"32768|{t}|idle|{cfg}|cake"
                    if base and c and k in base and k in c:
                        xs.append(t_i + xo + (0.05 if m == "qwen" else -0.05))
                        ys.append(c[k]["median"] / base[k]["median"])
                col = FMT["bf16"] if cfg == "E0" else FMT["int4"]
                ax.scatter(xs, ys, marker=mk, s=12, color=col if m == "llama" else "white", edgecolor=col, lw=0.7,
                           zorder=3)
    refline(ax, 1.0, color="#333333", ls="-", lw=0.6)
    ax.set_xticks(range(len(tiers1)))
    ax.set_xticklabels(["CPU", "b16", "b32", "b48"])
    ax.set_ylabel("time / time with 512-token unit")
    ax.set_title("M1 restore unit, Cake (GPU idle)")
    hs = [plt.Line2D([], [], marker="o", ls="", color="#555555", ms=4, label="256-token unit"),
          plt.Line2D([], [], marker="s", ls="", color="#555555", ms=4, label="128-token unit"),
          plt.Line2D([], [], marker="o", ls="", color=FMT["bf16"], ms=4, label="E0 (BF16)"),
          plt.Line2D([], [], marker="o", ls="", color=FMT["int4"], ms=4, label="E1s (INT4-KIVI)"),
          plt.Line2D([], [], marker="o", ls="", mfc="white", color="#555555", ms=4, label="hollow = Qwen")]
    ax.legend(handles=hs, fontsize=5.6, loc="upper right")
    ax.set_ylim(0.8, 3.2)
    panel_label(ax, "c")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=8, fontsize=5.8, bbox_to_anchor=(0.36, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=1.0)
    return save(fig, "F18_match_m2_m1")


# --------------------------------------------------------------------------------------------- F19
def _guard_clean(d, rids):
    gs = [g for g in glob.glob(os.path.join(d, "gpu_guard_*.json")) if any(r in os.path.basename(g) for r in rids)]
    return bool(gs) and all(load_json(g).get("verdict") == "CLEAN" for g in gs)


def _lc_speed(m, tier, L):
    """{(bw, gpu): (E0, E1)} medians for cake, 32K from the main grids, 64K/128K from the long-context grids."""
    fmt = {"llama": {"fp8": "fp8_token", "int4": "int4_kivi"}, "qwen": {"fp8": "fp8_head", "int4": "int4_kivi"}}[m][tier]
    if L == "32K":
        d = rp("m10_precision_mi300x", "speed_qwen_regrid" if (m == "qwen" and tier == "fp8") else f"speed_{m}")
        p = os.path.join(d, f"summary_{tier}.csv")
    else:
        d = rp("m10_longctx_mi300x", f"speed_{m}_{LEN_TOK[L]}_{fmt}")
        p = os.path.join(d, "summary.csv")
    rows = [r for r in read_csv(p) if int(r["length"]) == LEN_TOK[L] and r["mode"] == "cake"
            and r["config"] in ("E0", "E1")]
    rids = {x for r in rows for x in r["run_ids"].split()}
    if not _guard_clean(d, rids):
        return None
    out = {}
    for r in rows:
        out.setdefault((r["bw_name"], r["gpu_state"]), {})[r["config"]] = float(r["median_t_restore"])
    return out


def f19_longctx_speed():
    """Long context speed: E1 vs E0 restore-time saving per tier at 32K/64K/128K (original dequant)."""
    begin("F19_longctx_speed")
    tiers = ["cpu", "b16", "b32", "b48"]
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 3.5), sharex=True, sharey=True)
    for r, m in enumerate(MODELS):
        for c, tier in enumerate(("fp8", "int4")):
            ax = axes[r, c]
            data = {L: _lc_speed(m, tier, L) for L in LENS}
            for t in tiers:
                for st, ls in (("idle", "-"), ("busy", "--")):
                    ys = []
                    for L in LENS:
                        v = (data[L] or {}).get((t, st), {})
                        ys.append(100 * (v["E0"] - v["E1"]) / v["E0"] if "E0" in v and "E1" in v else np.nan)
                    ax.plot(range(len(LENS)), ys, ls=ls, marker="s" if t == "cpu" else "o", ms=3, color=TIER_COL[t], lw=0.9,
                            mfc=TIER_COL[t] if st == "idle" else "white",
                            label=f"{'CPU' if t == 'cpu' else t}, GPU {st}")
            refline(ax, 0, color="#333333", ls="-", lw=0.6)
            ax.set_xticks(range(len(LENS)))
            ax.set_xticklabels(LENS)
            ax.set_title(f"{'Llama' if m == 'llama' else 'Qwen'}: "
                         f"{'FP8 (main)' if tier == 'fp8' else 'INT4-KIVI'}", fontsize=7.3)
            if c == 0:
                ax.set_ylabel("E1 saving vs E0 (%)")
            if r == 1:
                ax.set_xlabel("context length")
    axes[0, 0].legend(fontsize=5.6, ncol=2, loc="lower left")
    fig.text(0.01, -0.01, "Cake, median restore time; 32K: main grids (3x10), 64K/128K: long-context grids (2x5); "
             "unfused (frozen) dequant; all gpu_guard CLEAN. Tiers use the 32K-calibrated bandwidths.", fontsize=6.0)
    fig.tight_layout()
    return save(fig, "F19_longctx_speed")


ALL = [f10_sweep_screen, f11_sweep_confirm, f12_longctx_accuracy, f13_longctx_harm, f14_longctx_position,
       f15_drop_ds2, f16_drop_ds1, f17_drop_speed, f18_match_formats_units, f19_longctx_speed]
