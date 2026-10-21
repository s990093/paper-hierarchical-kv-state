"""Shared paths, loaders, palette and save helper for the stage-2 report figures.

Rules (docs/precision_20261013/report_stage2/REPORT.md, appendix):
* every number drawn comes from a file under results/ (no hard-coded data);
* CPU only: torch is blocked on import so nothing here can touch a GPU;
* each figure records the files it read (SOURCES) -> figs/SOURCES.md.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import os
import sys

# ---- hard block: no torch / no GPU from the report code --------------------------------------
sys.modules["torch"] = None  # any `import torch` (also indirect) now raises ImportError
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(REPO, "results")
OUT = os.path.join(REPO, "docs", "precision_20261013", "report_stage2", "figs")
DERIVED = os.path.join(REPO, "docs", "precision_20261013", "report_stage2", "derived")
# H200（docs/h200/STAGE2_H200_VS_MI300X.md）：REPORT_STAGE2_PLATFORM=h200 時讀 results/*_h200、寫 docs/h200/report_stage2_h200/。沒設時不變。
PLATFORM = os.environ.get("REPORT_STAGE2_PLATFORM", "mi300x")
_H200_SUB = {"vllm_ttft_guarded2": "vllm_ttft_guarded", "vllm_ttft_guarded3": "vllm_ttft_guarded",
             "speed_qwen_regrid": "speed_qwen"}          # H200 沒有 Qwen 重量那次；TTFT 的 guard 重量只有一份
if PLATFORM == "h200":
    OUT = os.path.join(REPO, "docs", "h200", "report_stage2_h200", "figs")
    DERIVED = os.path.join(REPO, "docs", "h200", "report_stage2_h200", "derived")
sys.path.insert(0, os.path.join(REPO, "code"))

SOURCES: dict[str, set] = {}
_CUR = [None]


def begin(fig_id: str):
    _CUR[0] = fig_id
    SOURCES.setdefault(fig_id, set())


def _note(path):
    if _CUR[0] is not None:
        SOURCES[_CUR[0]].add(os.path.relpath(path, REPO))


def rp(*parts) -> str:
    """Path under results/."""
    if PLATFORM == "h200" and parts:
        parts = (parts[0].replace("m10_pilot_mi300x", "m10_precision_h200").replace("_mi300x", "_h200"),) + \
            tuple(_H200_SUB.get(x, x) for x in parts[1:])
    return os.path.join(RES, *parts)


def load_json(path):
    _note(path)
    with open(path) as f:
        return json.load(f)


def read_csv(path):
    """Read a CSV; if the raw .csv is absent, fall back to the committed .csv.gz."""
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path = path + ".gz"
    _note(path)
    if path.endswith(".gz"):
        with gzip.open(path, "rt", newline="") as f:
            txt = f.read()
    else:
        with open(path, newline="") as f:
            txt = f.read()
    rows = [r for r in csv.DictReader(io.StringIO(txt)) if r.get("run_id", "") != "run_id"]
    return rows


def read_text(path):
    _note(path)
    with open(path) as f:
        return f.read()


# ---- palette (Okabe-Ito; fixed per entity, never cycled) -------------------------------------
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
          blue="#0072B2", verm="#D55E00", purple="#CC79A7", grey="#999999")

# formats (storage state of a block)
FMT = {
    "bf16": OI["black"],
    "fp8": OI["blue"],          # each model's selected FP8 (Llama fp8_token, Qwen fp8_head)
    "fp8k0": OI["sky"],         # FP8 with K mean removed (Step 2b)
    "int8": OI["green"],        # int8_token
    "int4": OI["verm"],         # int4_kivi (main INT4)
    "int4tok": OI["orange"],    # per-token 4-bit family (int4_token, mxfp4, nvfp4, ...)
    "int2": OI["purple"],       # int2_kivi and K4V2
    "other": OI["grey"],
}
FMT_LABEL = {"bf16": "BF16 (E0)", "fp8": "FP8", "fp8k0": "FP8-K0", "int8": "INT8", "int4": "INT4-KIVI",
             "int4tok": "4-bit per-token", "int2": "INT2-KIVI", "other": "other"}

# who decides what to drop (token-drop study only)
WHO = {"write": OI["blue"], "query": OI["verm"], "position": OI["orange"], "random": OI["grey"],
       "oracle": OI["purple"], "control": OI["black"]}

# tiers: one hue, light -> dark = slow
TIER_ORDER = ["cpu", "b16", "b32", "b48", "nfs"]
TIER_COL = {"cpu": "#9ecae1", "b16": "#4292c6", "b32": "#2171b5", "b48": "#08519c", "nfs": "#08306b"}
TIER_LABEL = {"cpu": "CPU\n(real H2D)", "b16": "b16", "b32": "b32", "b48": "b48", "nfs": "NFS-class\n0.33 GiB/s"}

# configurations: marker by role (color = format)
CFG_MARK = {"E0": "o", "E1": "o", "E1s": "s", "E1sr": "D", "F25": ">", "M25": "D", "B25": "<", "R25": "o"}

SINGLE = 3.4
DOUBLE = 7.0


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.8,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "errorbar.capsize": 1.5,
        "savefig.dpi": 200,
        "pdf.fonttype": 42,
        "figure.dpi": 100,
    })


def refline(ax, y, label=None, color="#555555", ls="--", lw=0.7, axis="y", **kw):
    if axis == "y":
        ax.axhline(y, color=color, ls=ls, lw=lw, zorder=1, **kw)
    else:
        ax.axvline(y, color=color, ls=ls, lw=lw, zorder=1, **kw)


def panel_label(ax, s, x=None, y=None):
    """Prefix the axes title with a bold panel letter (left-aligned title)."""
    t = ax.get_title()
    ax.set_title("")
    ax.set_title(f"$\\bf{{{s}}}$  {t}", loc="left")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    out = []
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"{name}.{ext}")
        fig.savefig(p, bbox_inches="tight", pad_inches=0.02, dpi=200)
        sz = os.path.getsize(p)
        if sz >= 1_000_000:
            raise SystemExit(f"{p} is {sz} bytes (>= 1 MB)")
        out.append((p, sz))
    plt.close(fig)
    return out


def write_sources_merge():
    """Merge this run's SOURCES into figs/SOURCES.json (so --only runs keep the others) and write SOURCES.md."""
    pj = os.path.join(OUT, "SOURCES.json")
    allsrc = {}
    if os.path.exists(pj):
        allsrc = {k: set(v) for k, v in json.load(open(pj)).items()}
    for k, v in SOURCES.items():
        allsrc[k] = set(v)
    json.dump({k: sorted(v) for k, v in sorted(allsrc.items())}, open(pj, "w"), indent=1)
    lines = ["# Figure data sources (auto-generated by code/report_stage2/make_figs.py)", "",
             "Every figure reads only these files (paths relative to the repo root).", ""]
    SOURCES.clear(); SOURCES.update(allsrc)
    for k in sorted(SOURCES):
        lines.append(f"## {k}")
        for p in sorted(SOURCES[k]):
            lines.append(f"- `{p}`")
        lines.append("")
    with open(os.path.join(OUT, "SOURCES.md"), "w") as f:
        f.write("\n".join(lines))


def fmt_family(fmt: str) -> str:
    """Map a codec name to a palette family key."""
    f = fmt.lower()
    if f in ("", "bf16", "e0", "none"):
        return "bf16"
    if f.startswith("fp8k0"):
        return "fp8k0"
    if f.startswith("fp8"):
        return "fp8"
    if f.startswith("int8") or f.startswith("kv-int8"):
        return "int8"
    if f == "int4_kivi":
        return "int4"
    if f.startswith("int2") or f.startswith("kv-int4"):
        return "int2"
    if f.startswith("int4") or f.startswith("mxfp4") or f.startswith("nvfp4"):
        return "int4tok"
    return "other"
