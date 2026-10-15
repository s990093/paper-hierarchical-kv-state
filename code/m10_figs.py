"""m10_figs.py — 精度這條線的圖（03 §8 的圖 A–D，加速度格子、逐 block 敏感度地圖）。只讀 results/ 的 CSV，不用 GPU。

每張圖右下角印資料來源（CSV 檔名與 run_id 數），每個數字都能追回 CSV（03 §6 m10_figs 那一列）。
顏色照 dataviz 的參考調色盤（類別色依固定順序、跟著配置走；分歧色藍↔紅、中點灰）。

用法：
  python code/m10_figs.py speed  --model llama --restores <restores.csv> --run <grid run_id> --out <png>
  python code/m10_figs.py sources --model llama --restores … --run … --bw b32 --out <png>          # 圖 B
  python code/m10_figs.py A      --model llama --quality <quality json> --summary <grid summary.csv> --delta-star 0.05 --out <png>
  python code/m10_figs.py C      --model llama --csv <逐題 CSV…> --configs E1 E1s … --out <png>     # 圖 C
  python code/m10_figs.py D      --model llama --csv … --configs E0 E1 … --out <png>                # 圖 D
  python code/m10_figs.py sens   --csv <sensmap CSV…> --out <png>
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import statistics
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ---- 調色盤（dataviz references/palette.md，淺色）
SURF, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COLOR = {"E0": INK2, "E1": SLOTS[0], "E1s": SLOTS[1], "E1sr": SLOTS[2], "F25": SLOTS[3], "M25": SLOTS[4],
         "B25": SLOTS[5], "R25": SLOTS[6], "E1-K0": SLOTS[7], "E1-INT8": MUTED,
         "compute_only": MUTED, "load_only": BASE}
ORDER = ["E0", "E1", "E1s", "E1sr", "F25", "M25", "B25", "R25", "E1-K0", "E1-INT8"]
DIV_NEG, DIV_MID, DIV_POS = "#e34948", "#f0efec", "#2a78d6"     # 掉分＝紅、0＝灰、加分＝藍

from matplotlib import font_manager  # noqa: E402

FONT = os.environ.get("M10_CJK_FONT", "/mlsteam/data/tiara/fonts/NotoSansCJKtc-Regular.otf")   # 字型檔（m9_advisor_figs 同一個）
_fam = ["DejaVu Sans"]
if os.path.exists(FONT):
    font_manager.fontManager.addfont(FONT)
    _fam = [font_manager.FontProperties(fname=FONT).get_name(), "DejaVu Sans"]
plt.rcParams.update({
    "font.family": _fam, "font.size": 9,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlecolor": INK, "axes.titlesize": 9.5,
    "legend.frameon": False, "legend.labelcolor": INK2})


def base_name(cfg: str) -> str:
    """'E1s@fp8k0_head' → 'E1-K0'；'E1s@int8_token' → 'E1-INT8'；'E1@int4_kivi' → 'E1'。"""
    name, _, fmt = cfg.partition("@")
    if fmt.startswith("fp8k0"):
        return "E1-K0"
    if fmt.startswith("int8"):
        return "E1-INT8"
    return name


def source_note(fig, text):
    fig.text(0.995, 0.005, text, ha="right", va="bottom", fontsize=6.5, color=MUTED)


def save(fig, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("wrote", out)


def read_csvs(paths):
    rows = []
    for p in paths:
        f = gzip.open(p, "rt", newline="") if p.endswith(".gz") else open(p, newline="")   # restores.csv 進 git 的是 .gz
        rows += [r for r in csv.DictReader(f) if r.get("run_id") != "run_id"]
    return rows


# ============================================================================ 速度格子
def fig_speed(a):
    rows = [r for r in read_csvs([a.restores]) if r["run_id"] == a.run and int(r["length"]) == 32768]
    cells = defaultdict(list)
    for r in rows:
        key = r["config"] if r["mode"] == "cake" else ("compute_only" if r["mode"] == "compute_only"
                                                      else f"load_only({r['config']})")
        cells[(r["bw_name"], r["gpu_state"], key)].append(float(r["t_restore"]))
    bws = [b for b in ("b16", "b32", "b48", "cpu") if any(k[0] == b for k in cells)]
    states = ["idle", "busy"]
    keys = [c for c in ORDER if any(k[2] == c for k in cells)] + ["compute_only"] + \
        sorted({k[2] for k in cells if k[2].startswith("load_only")})
    fig, axes = plt.subplots(len(states), len(bws), figsize=(3.0 * len(bws), 2.9 * len(states)), sharey=True,
                             squeeze=False)
    for i, st in enumerate(states):
        for j, bw in enumerate(bws):
            ax = axes[i][j]
            for y, k in enumerate(keys):
                v = cells.get((bw, st, k))
                if not v:
                    continue
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                c = COLOR.get(k, BASE if k.startswith("load_only") else INK2)
                ax.plot([q1, q3], [y, y], color=c, lw=2, solid_capstyle="round")
                ax.plot(med, y, "o", ms=6, color=c, mec=SURF, mew=1.5)
            ax.set_yticks(range(len(keys)))
            ax.set_yticklabels(keys)
            if i == 0 and j == 0:            # sharey：只翻一次（翻兩次會變回原樣）
                ax.invert_yaxis()
            ax.set_title(f"{bw}・GPU {'閒' if st == 'idle' else '忙'}")
            ax.set_xlabel("還原時間（s）")
            ax.set_xlim(left=0)
            # 只算那條比其他慢很多時（例：CPU 層 0.05 s 對 3 s）截掉、標數字，不然其他配置的差別全擠在 0 附近
            q3s = [np.percentile(cells[(bw, st, k)], 75) for k in keys if k != "compute_only" and cells.get((bw, st, k))]
            co = cells.get((bw, st, "compute_only"))
            if co and q3s and np.median(co) > 3 * max(q3s):
                xmax = 1.25 * max(q3s)
                ax.set_xlim(0, xmax)
                ax.annotate(f"只算 {np.median(co):.2f} s →", xy=(xmax, keys.index("compute_only")), xytext=(-2, 0),
                            textcoords="offset points", ha="right", va="center", fontsize=7, color=INK2)
    fig.suptitle(f"{a.model}：每個配置的還原時間（32K；點＝中位數、線＝四分位距；3 process × 10 次）", color=INK, fontsize=10)
    source_note(fig, f"{os.path.basename(a.restores)} run_id={a.run} n={len(rows)}")
    fig.tight_layout()
    save(fig, a.out)


# ============================================================================ 圖 B：讀到什麼
def fig_sources(a):
    rows = [r for r in read_csvs([a.restores]) if r["run_id"] == a.run and int(r["length"]) == 32768
            and r["mode"] == "cake" and r["bw_name"] == a.bw]
    cfgs = [c for c in ORDER if any(r["config"] == c for r in rows)]
    states = ["idle", "busy"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.42 * len(cfgs) + 1.3), sharey=True)
    for ax, st in zip(axes, states):
        for y, c in enumerate(cfgs):
            v = [r for r in rows if r["config"] == c and r["gpu_state"] == st]
            if not v:
                continue
            rec = statistics.median(int(r["n_recompute"]) for r in v)
            lbf = statistics.median(int(r["n_load_bf16"]) for r in v)
            llp = statistics.median(int(r["fp8_read"]) for r in v)
            stored = statistics.median(int(r["fp8_stored"]) for r in v)
            x0 = 0
            for w, col, lab in ((rec, "#86b6ef", "重算（精確 BF16）"), (lbf, INK2, "載入 BF16"), (llp, DIV_NEG, "載入低精度")):
                ax.barh(y, w, left=x0, color=col, edgecolor=SURF, linewidth=1.5, height=0.62,
                        label=lab if y == 0 else None)
                x0 += w
            unread = stored - llp
            ax.text(64.8, y, f"存了低精度、沒讀到 {unread:g}" if stored else "", va="center", fontsize=7, color=INK2)
        ax.set_yticks(range(len(cfgs)))
        ax.set_yticklabels(cfgs)
        if st == "idle":                     # sharey：只翻一次
            ax.invert_yaxis()
        ax.set_xlim(0, 64)
        ax.set_xlabel("block 數（共 64；三段各自取中位數）")
        ax.set_title(f"{a.bw}・GPU {'閒' if st == 'idle' else '忙'}")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.18), ncol=3)
    fig.suptitle(f"圖 B　{a.model}：Cake 還原時每個 block 從哪裡來（限速層 {a.bw}）", color=INK, fontsize=10)
    source_note(fig, f"{os.path.basename(a.restores)} run_id={a.run}")
    fig.tight_layout()
    save(fig, a.out)


# ============================================================================ 圖 A：品質對延遲
def fig_A(a):
    q = json.load(open(a.quality))
    summ = [r for r in csv.DictReader(open(a.summary)) if int(r["length"]) == 32768 and r["mode"] == "cake"]
    med = {(r["config"], r["bw_name"], r["gpu_state"]): float(r["median_t_restore"]) for r in summ}
    bws = [b for b in ("b16", "b32", "b48", "cpu") if any(k[1] == b for k in med)]
    fig, axes = plt.subplots(2, len(bws), figsize=(3.1 * len(bws), 6.0), squeeze=False)
    cfgs = [c for c in ORDER if c in q["configs"] or c == "E0"]
    for i, st in enumerate(("idle", "busy")):
        for j, bw in enumerate(bws):
            ax = axes[i][j]
            ax.axhline(-2, color=DIV_NEG, lw=1, ls="--")
            ax.axhline(0, color=BASE, lw=0.8)
            e1 = med.get(("E1", bw, st))
            if e1:
                ax.axvspan(e1 * (1 - a.delta_star), e1 * (1 + a.delta_star), color=SLOTS[0], alpha=0.08, lw=0)
            for c in cfgs:
                x = med.get((c, bw, st))
                if x is None:
                    continue
                col = COLOR.get(c, INK2)
                if c == "E0":
                    ax.plot(x, 0, "o", ms=7, color=col, mec=SURF, mew=1.5)
                    ax.annotate("E0", (x, 0), textcoords="offset points", xytext=(4, 4), fontsize=7, color=INK2)
                    continue
                for cond, mk, fill in (("A", "o", "none"), ("B-idle" if st == "idle" else "B-busy", "o", col)):
                    if cond != "A" and bw != "b32":
                        continue
                    e = q["configs"][c].get(cond)
                    if not e:
                        continue
                    y = 100 * e["diff"]
                    lo, hi = 100 * e["ci95"][0], 100 * e["ci95"][1]
                    ax.plot([x, x], [lo, hi], color=col, lw=1.5)
                    ax.plot(x, y, mk, ms=7, color=col, mfc=fill if fill != "none" else SURF, mec=col, mew=1.5)
                ax.annotate(c, (x, 100 * q["configs"][c]["A"]["diff"]), textcoords="offset points", xytext=(4, -9),
                            fontsize=7, color=INK2)
            ax.set_title(f"{bw}・GPU {'閒' if st == 'idle' else '忙'}")
            ax.set_xlabel("還原時間中位數（s）")
            if j == 0:
                ax.set_ylabel("對 E0 的正確率差（百分點）")
    fig.suptitle(f"圖 A　{q['model']}：品質對延遲（空心＝讀法 A、實心＝讀法 B〔只在 b32〕；虛線＝ε −2；淺藍帶＝E1 ±δ*）",
                 color=INK, fontsize=10)
    source_note(fig, f"{os.path.basename(a.quality)}；{os.path.basename(a.summary)}")
    fig.tight_layout()
    save(fig, a.out)


# ============================================================================ 圖 C：答案位置 × 低精度位置
def _paired_by(rows, cfg, cond_filter, key_fn):
    ref = {(r["item_id"], r["turn"]): float(r["score"]) for r in rows if r["config"] == "E0" and r["mode"] == "A"
           and r["turn2_variant"] in ("", "a")}
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["config"] != cfg or not cond_filter(r):
            continue
        k = (r["item_id"], r["turn"])
        out[key_fn(r)][r["item_id"]].append(float(r["score"]) - ref[k])
    return out


def _boot(d, n=2000):
    items = sorted(d)
    s = np.array([sum(d[i]) for i in items])
    c = np.array([len(d[i]) for i in items])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(items), size=(n, len(items)))
    st = s[idx].sum(1) / c[idx].sum(1)
    return s.sum() / c.sum(), *np.percentile(st, [2.5, 97.5])


CONDF = {"A": lambda r: r["mode"] == "A" and r["turn2_variant"] in ("", "a"),
         "B-idle": lambda r: r["mode"] == "B" and r["bw"] == "b32" and r["gpu_state"] == "idle" and r["turn2_variant"] in ("", "a"),
         "B-busy": lambda r: r["mode"] == "B" and r["bw"] == "b32" and r["gpu_state"] == "busy" and r["turn2_variant"] in ("", "a")}


def fig_C(a):
    rows = [r for r in read_csvs(a.csv) if r["model"] == a.model and r["task"] in ("T1", "T2", "T3", "T4")]
    if a.tasks:
        rows = [r for r in rows if r["task"] in a.tasks]
    bands = ["front", "mid", "back"]
    conds = [c for c in ("A", "B-idle", "B-busy") if any(CONDF[c](r) for r in rows)]
    cfgs = [c for c in a.configs if any(r["config"] == c for r in rows)]
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MID, DIV_POS])
    lim = a.vlim
    # 上下疊（10/15 晚：左右排時太扁、格內字太小，總標題壓到中間的小標題）
    fig, axes = plt.subplots(len(conds), 1, figsize=(0.95 * len(cfgs) + 2.6, 2.25 * len(conds) + 0.9), squeeze=False,
                             layout="constrained")
    for ax, cond in zip(axes[:, 0], conds):
        M = np.full((3, len(cfgs)), np.nan)
        txt = [[""] * len(cfgs) for _ in range(3)]
        for j, c in enumerate(cfgs):
            g = _paired_by(rows, c, CONDF[cond], lambda r: r["band"])
            for i, b in enumerate(bands):
                if g.get(b):
                    p, lo, hi = _boot(g[b])
                    M[i, j] = 100 * p
                    txt[i][j] = f"{100 * p:+.1f}\n±{100 * (hi - lo) / 2:.1f}"
        im = ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
        for i in range(3):
            for j in range(len(cfgs)):
                ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(cfgs)))
        ax.set_xticklabels([base_name(c) for c in cfgs])
        ax.set_yticks(range(3))
        ax.set_yticklabels(["答案在前", "答案在中", "答案在後"])
        ax.set_title({"A": "讀法 A（全載入）", "B-idle": "讀法 B・b32・閒", "B-busy": "讀法 B・b32・忙"}[cond])
        ax.grid(False)
    fig.colorbar(im, ax=axes[:, 0].tolist(), shrink=0.6, label="對 E0 的差（百分點）")
    fig.suptitle(f"圖 C　{a.model}{a.title_suffix}：答案位置 × 低精度位置（T1–T4，兩輪合併；格內＝點估計／CI 半寬）",
                 color=INK, fontsize=10)
    source_note(fig, " ".join(os.path.basename(p) for p in a.csv)[:160])
    save(fig, a.out)


# ============================================================================ 圖 D：答案位置 → 正確率
def fig_D(a):
    rows = [r for r in read_csvs(a.csv) if r["model"] == a.model and r["task"] in ("T1", "T2", "T3", "T4")
            and r["mode"] == "A" and r["turn2_variant"] in ("", "a")]
    cfgs = [c for c in a.configs if any(r["config"] == c for r in rows)]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for c in cfgs:
        bins = defaultdict(list)
        for r in rows:
            if r["config"] != c:
                continue
            nb = json.loads(r["needle_blocks"])
            if not nb:
                continue
            m = float(np.median(nb))
            k = int(min(15, max(0, m - 1)) // 4) if m < 17 else (4 + int((m - 24) // 4) if m < 40 else 8 + int((m - 48) // 4))
            bins[k].append(float(r["score"]))
        col = COLOR.get(base_name(c), INK2)
        dx = 0.35 * (cfgs.index(c) - (len(cfgs) - 1) / 2)          # 同一組的配置左右錯開，CI 才不會疊在一起
        for seg in ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)):
            xs = [k for k in seg if bins.get(k)]
            if not xs:
                continue
            centers = [dx + ((1 + 4 * k + 2) if k < 4 else ((24 + 4 * (k - 4) + 2) if k < 8 else (48 + 4 * (k - 8) + 2)))
                       for k in xs]
            ys = [100 * statistics.mean(bins[k]) for k in xs]
            hw = [100 * 1.96 * (statistics.pstdev(bins[k]) / len(bins[k]) ** 0.5) for k in xs]   # 常態近似，只看形狀
            ax.vlines(centers, [y - h for y, h in zip(ys, hw)], [y + h for y, h in zip(ys, hw)], color=col, lw=1, alpha=0.45)
            ax.plot(centers, ys, "-o", color=col, lw=2, ms=5, mec=SURF, mew=1.2,
                    label=base_name(c) if seg[0] == 0 else None)
    for x0, x1 in ((17, 24), (40, 48)):
        ax.axvspan(x0, x1, color=GRID, alpha=0.5, lw=0)
    ax.set_xlim(0, 64)
    ax.set_xlabel("答案所在的 block（needle_blocks 的中位數；灰帶＝沒有題目）")
    ax.set_ylabel("正確率（%；細線＝95% CI，常態近似）")
    ax.legend(ncol=len(cfgs), loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"圖 D　{a.model}{a.title_suffix}：答案位置 → 正確率（讀法 A，T1–T4，兩輪合併）", color=INK, fontsize=10, y=1.08)
    source_note(fig, " ".join(os.path.basename(p) for p in a.csv)[:160])
    save(fig, a.out)


# ============================================================================ 逐 block 敏感度地圖
def fig_sens(a):
    rows = read_csvs(a.csv)
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[(r["model"], r["fmt"])][int(r["block"])].append(float(r["kl_vs_e0"]))
    keys = sorted(groups)
    fig, axes = plt.subplots(len(keys), 1, figsize=(7.5, 1.9 * len(keys)), sharex=True, squeeze=False)
    for ax, k in zip(axes[:, 0], keys):
        g = groups[k]
        xs = sorted(g)
        m = [statistics.mean(g[x]) for x in xs]
        se = [statistics.stdev(g[x]) / np.sqrt(len(g[x])) if len(g[x]) > 1 else 0 for x in xs]
        ax.fill_between(xs, [max(1e-9, a_ - 1.96 * b_) for a_, b_ in zip(m, se)], [a_ + 1.96 * b_ for a_, b_ in zip(m, se)],
                        color=SLOTS[0], alpha=0.15, lw=0)
        ax.plot(xs, m, color=SLOTS[0], lw=2)
        ax.set_yscale("log")
        ax.set_title(f"{k[0]}・{k[1]}（每點 {len(g[xs[0]])} 題的平均 KL，帶＝95% CI）", loc="left")
        ax.set_ylabel("KL")
    axes[-1][0].set_xlabel("只有這一個 block 存成低精度")
    fig.suptitle("逐 block 敏感度地圖（03 §3.3b；第 1 輪；只報告）", color=INK, fontsize=10)
    source_note(fig, " ".join(os.path.basename(p) for p in a.csv)[:160])
    fig.tight_layout()
    save(fig, a.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("speed")
    p.add_argument("--model", required=True)
    p.add_argument("--restores", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("sources")
    p.add_argument("--model", required=True)
    p.add_argument("--restores", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--bw", default="b32")
    p.add_argument("--out", required=True)
    p = sub.add_parser("A")
    p.add_argument("--quality", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--delta-star", type=float, required=True)
    p.add_argument("--out", required=True)
    for n in ("C", "D"):
        p = sub.add_parser(n)
        p.add_argument("--model", required=True)
        p.add_argument("--csv", nargs="+", required=True)
        p.add_argument("--configs", nargs="+", required=True)
        p.add_argument("--tasks", nargs="*")
        p.add_argument("--vlim", type=float, default=10.0)
        p.add_argument("--title-suffix", default="")
        p.add_argument("--out", required=True)
    p = sub.add_parser("sens")
    p.add_argument("--csv", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"speed": fig_speed, "sources": fig_sources, "A": fig_A, "C": fig_C, "D": fig_D, "sens": fig_sens}[a.cmd](a)


if __name__ == "__main__":
    main()
