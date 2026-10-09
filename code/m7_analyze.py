"""m7_analyze.py — 第一階段的分析與圖（全部從 results/m7_write_policy_mi300x/*.csv 自動產生，不手打）

  python code/m7_analyze.py
產出：results/m7_write_policy_mi300x/summary_*.csv、verdict.json；docs/phase1_20261008/report/figs/*.png
判準照 phase1/05 §6（開跑前寫死）。
"""
from __future__ import annotations

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m7_restore_harness import write_boundary  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results", "m7_write_policy_mi300x")
FIG = os.path.join(HERE, "..", "docs", "phase1_20261008", "report", "figs")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": ["Noto Sans CJK TC", "DejaVu Sans"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984",
                     "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
for fp in ("/mlsteam/data/tiara/fonts/NotoSansCJKtc-Regular.otf",):
    if os.path.exists(fp):
        from matplotlib import font_manager
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = [font_manager.FontProperties(fname=fp).get_name(), "DejaVu Sans"]

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = \
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"
GRAY, GRAY2 = "#9a9893", "#c9c7c0"
INK2 = "#52514e"

V = {}   # verdict


def rd(name):
    p = os.path.join(R, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def med_ci(x, conf=0.95):
    """中位數與無母數 95% CI（二項分布順序統計量）。"""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    from scipy.stats import binom
    lo = int(binom.ppf((1 - conf) / 2, n, 0.5))
    hi = int(binom.isf((1 - conf) / 2, n, 0.5))
    lo = max(lo - 1, 0)
    hi = min(hi, n - 1)
    return float(np.median(x)), float(x[lo]), float(x[hi])


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =========================================================== C：校準
def calib():
    out = {}
    c1 = rd("calib_c1.csv")
    f = c1[c1.item == "f_chunk"].groupby("chunk_idx").ms.median()
    A = np.polyfit(f.index.values, f.values, 1)
    out["C1"] = {"f0_ms": float(f.iloc[0]), "f79_ms": float(f.iloc[-1]), "slope_ms_per_chunk": float(A[0]),
                 "intercept_ms": float(A[1]), "monotone_increasing": bool((np.diff(f.values) > -0.5).all()),
                 "rep_spread_max_pct": float((c1[c1.item == "f_chunk"].groupby("chunk_idx").ms.agg(
                     lambda s: (s.max() - s.min()) / s.median() * 100)).max()),
                 "run_ids": sorted(c1.run_id.unique())}
    out["C1"]["PASS"] = out["C1"]["monotone_increasing"] and A[0] > 0
    # 圖：f(i)
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.plot((f.index + 1) * 512 / 1024, f.values, color=BLUE, lw=2)
    ax.set_xlabel("chunk 結束位置（K token）")
    ax.set_ylabel("重算一個 chunk（ms）")
    ax.set_title("C1　逐 chunk 重算成本 f(i)（Llama-3.1-8B BF16，MI300X，GPU 閒）", loc="left", fontsize=9)
    save(fig, "c1_f.png")

    c4 = rd("c4_noise.csv")
    if c4 is not None:
        rows = []
        for (slot, pm), g in c4.groupby(["slot", "proc_mode"]):
            m, lo, hi = med_ci(g.ttft)
            rows.append({"slot": slot, "proc_mode": pm, "n": len(g), "median_s": m, "ci_lo": lo, "ci_hi": hi,
                         "delta_pct": (hi - lo) / 2 / m * 100, "cv_pct": g.ttft.std() / g.ttft.mean() * 100,
                         "meet_values": sorted(g.meet.unique().tolist())})
        t = pd.DataFrame(rows)
        t.to_csv(os.path.join(R, "summary_c4.csv"), index=False)
        delta = float(t.delta_pct.max())
        out["C4"] = {"table": rows, "delta_pct": delta, "PASS": delta <= 10.0}
        fig, ax = plt.subplots(figsize=(5.2, 2.4))
        for k, ((slot, pm), g) in enumerate(c4.groupby(["slot", "proc_mode"])):
            ax.scatter(g.ttft * 1e3, np.full(len(g), k) + np.random.default_rng(0).uniform(-.15, .15, len(g)),
                       s=14, color=[BLUE, ORANGE, AQUA, VIOLET][k % 4], edgecolor="white", linewidth=0.5)
        labs = [f"{s} / {'同 process' if p == 'same' else '每次重開'}" for s, p in c4.groupby(["slot", "proc_mode"]).groups]
        ax.set_yticks(range(len(labs)), labs)
        ax.set_xlabel("TTFT（ms）；S0，L=16K，GPU 閒，SSD 層＝本地 SSD 限速器")
        ax.set_title(f"C4　雜訊底線 δ = {delta:.2f}%", loc="left", fontsize=9)
        save(fig, "c4_noise.png")

    c7 = rd("calib_c7_self.csv")
    if c7 is not None:
        g = c7.groupby("set_GiBps").err_pct.agg(["median", "min", "max"]).reset_index()
        out["C7_self"] = {"table": g.to_dict("records"), "max_abs_err_pct": float(c7.err_pct.abs().max()),
                          "PASS": bool((g["median"].abs() <= 5).all())}
    an = rd("c7_anchor.csv")
    if an is not None:
        p = an.groupby(["device", "mode", "L", "io"]).ttft.median().unstack("io").reset_index()
        p["sim_over_real"] = p["sim"] / p["real"]
        p.to_csv(os.path.join(R, "summary_c7_anchor.csv"), index=False)
        rb = an[(an.io == "real") & (an.device != "cpu")]
        out["C7_anchor"] = {"table": p.to_dict("records"),
                            "page_cache_ok": bool((rb.read_bytes_delta >= (rb.n_load_ssd * 64 * 2**20) * 0.99).all()),
                            "kv_bad_total": int(pd.to_numeric(an.kv_bad, errors="coerce").fillna(0).sum())}
    V["calib"] = out
    return out


# =========================================================== C3 + A0
def a0():
    d = rd("a0.csv")
    if d is None:
        return
    P = json.load(open(os.path.join(R, "tier_params.json")))
    c1 = rd("calib_c1.csv")
    f = (c1[c1.item == "f_chunk"].groupby("chunk_idx").ms.median() / 1e3).tolist()
    g = d.groupby(["L", "bw_name", "read_GiBps", "mode"]).agg(ttft=("ttft", "median"), meet=("meet", "median"),
                                                               n=("ttft", "size")).reset_index()
    g.to_csv(os.path.join(R, "summary_a0.csv"), index=False)
    comp = g[g["mode"] == "compute_only"].set_index("L").ttft
    rows = []
    for (L, bw, gib), h in g[g["mode"] != "compute_only"].groupby(["L", "bw_name", "read_GiBps"]):
        lo = h[h["mode"] == "load_only"].ttft.iloc[0]
        ck = h[h["mode"] == "cake"].ttft.iloc[0]
        meet = h[h["mode"] == "cake"].meet.iloc[0]
        n = L // 512
        ell = 64 * 2**20 / (gib * 2**30)
        pred = write_boundary(n, f, ell)
        rows.append({"L": L, "bw": bw, "GiBps": gib, "ttft_compute": comp[L], "ttft_load": lo, "ttft_cake": ck,
                     "speedup_vs_compute": comp[L] / ck, "speedup_vs_load": lo / ck, "meet_measured": meet,
                     "meet_pred_alg1": pred, "meet_err": meet - pred})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(R, "summary_a0_speedup.csv"), index=False)
    # 三個趨勢條件
    c1_ok = []   # 對只算的加速隨長度變大（每個頻寬）
    for bw, h in t.groupby("bw"):
        s = h.sort_values("L").speedup_vs_compute.values
        c1_ok.append(bool(s[-1] >= s[0]))
    c2_ok = []   # 對只載的加速隨頻寬變大而變小（每個 L）
    for L, h in t.groupby("L"):
        s = h.sort_values("GiBps").speedup_vs_load.values
        c2_ok.append(bool(s[-1] <= s[0]))
    tol = 1.02
    c3 = bool(((t.ttft_cake <= t.ttft_compute * tol) & (t.ttft_cake <= t.ttft_load * tol)).all())
    V["A0"] = {"cond1_speedup_vs_compute_grows_with_L": f"{sum(c1_ok)}/{len(c1_ok)} 頻寬成立",
               "cond2_speedup_vs_load_shrinks_with_bw": f"{sum(c2_ok)}/{len(c2_ok)} 長度成立",
               "cond3_cake_not_slower_(2%tol)": c3,
               "PASS": all(c1_ok) and all(c2_ok) and c3,
               "C3_meet_abs_err_max": int(t.meet_err.abs().max()),
               "C3_within_1_chunk_frac": float((t.meet_err.abs() <= 1).mean()),
               "kv_bad_total": int(pd.to_numeric(d.kv_bad, errors="coerce").fillna(0).sum()),
               "run_ids": sorted(d.run_id.unique())}
    V["C3"] = {"PASS": bool((t.meet_err.abs() <= 1).all()), "max_abs_err_chunks": int(t.meet_err.abs().max()),
               "note": "預測＝演算法 1（GPU 閒的 f 中位數、限速器 ℓ）；實測＝Cake 動態會合的重算 chunk 數"}
    # 圖：A0
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.0))
    order = t.sort_values("GiBps").bw.unique()
    # 頻寬是有序量 → 單一色相由淺到深
    cols = plt.get_cmap("Blues")(np.linspace(0.35, 1.0, len(order)))
    for k, bw in enumerate(order):
        h = t[t.bw == bw].sort_values("L")
        gib = h.GiBps.iloc[0]
        lab = (f"{bw}（{gib:.2f} GiB/s）" if "Gbps" in bw else
               f"{ {'local':'本地 SSD','nfs':'NFS'}[bw] }實測參數（{gib:.2f} GiB/s）")
        axs[0].plot(h.L / 1024, h.speedup_vs_compute, marker="o", ms=4, lw=1.6, color=cols[k], label=lab)
        axs[1].plot(h.L / 1024, h.speedup_vs_load, marker="o", ms=4, lw=1.6, color=cols[k], label=lab)
    axs[0].set_title("Cake 對「只算」的加速", loc="left", fontsize=9)
    axs[1].set_title("Cake 對「只載」的加速", loc="left", fontsize=9)
    for ax in axs:
        ax.set_xlabel("歷史長度 L（K token）"); ax.axhline(1, color=GRAY, lw=0.8); ax.set_xscale("log", base=2)
        ax.set_xticks([4, 8, 16, 32], ["4", "8", "16", "32"])
    axs[1].set_yscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter
    axs[1].yaxis.set_major_locator(FixedLocator([0.8, 1, 1.5, 2, 3, 4, 6]))
    axs[1].yaxis.set_minor_locator(FixedLocator([]))
    axs[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    axs[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    axs[1].legend(fontsize=7, frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("A0　純 Cake 跑通（S0；SSD 層為模擬：CPU 記憶體＋限速器；每點 6 次中位數）", x=0.01, y=1.04, ha="left", fontsize=9.5)
    save(fig, "a0_cake.png")


# =========================================================== A1
def a1():
    d = rd("a1_share.csv")
    if d is None:
        return
    d = d.copy()
    idle = d.gpu_state == "idle"
    d["t_new_ratio"] = d.t_new / d[idle].groupby("L").t_new.min().reindex(d.L).values
    V["A1_gpu_slow_rows_idle"] = {"n": int((idle & (d.t_new_ratio > 1.15)).sum()), "of": int(idle.sum())}
    d["gap"] = d.gap.map(lambda x: "inf" if str(x) in ("inf", "nan") or (isinstance(x, float) and math.isinf(x)) else "0")
    g = d.groupby(["ssd_dev", "gpu_state", "gap", "L", "strategy"]).agg(
        ttft=("ttft", "median"), ttft_lo=("ttft", lambda x: med_ci(x)[1]), ttft_hi=("ttft", lambda x: med_ci(x)[2]),
        meet=("meet", "median"), n_rec=("n_recompute", "median"), n_cpu=("n_load_cpu", "median"),
        n_ssd=("n_load_ssd", "median"), t_rec=("t_recompute", "median"), t_load=("t_load", "median"),
        t_new=("t_new", "median"), b=("b", "first"), w_ssd=("w_chunks_ssd", "first"), w_cpu=("w_chunks_cpu", "first"),
        t_write=("t_write_ssd_s", "first"), n=("ttft", "size")).reset_index()
    g.to_csv(os.path.join(R, "summary_a1.csv"), index=False)
    c4 = V.get("calib", {}).get("C4", {}).get("delta_pct", np.nan)

    def cmp(a, b, **flt):
        rows = []
        h = g
        for k, v in flt.items():
            h = h[h[k] == v]
        for key, x in h.groupby(["ssd_dev", "gpu_state", "gap", "L"]):
            xa, xb = x[x.strategy == a], x[x.strategy == b]
            if len(xa) and len(xb):
                A, B = xa.iloc[0], xb.iloc[0]
                diff = (A.ttft - B.ttft) / B.ttft * 100
                sep = (A.ttft_hi < B.ttft_lo) or (B.ttft_hi < A.ttft_lo)
                rows.append(dict(zip(["ssd_dev", "gpu_state", "gap", "L"], key), a=a, b=b, ttft_a=A.ttft, ttft_b=B.ttft,
                                 diff_pct=diff, ci_separated=sep, beyond_delta=abs(diff) > c4))
        return rows
    comps = cmp("S3", "S2a") + cmp("S3", "S0") + cmp("S5", "S0") + cmp("S5", "CPUall")
    ct = pd.DataFrame(comps)
    ct.to_csv(os.path.join(R, "summary_a1_pairs.csv"), index=False)
    q1 = ct[(ct.a == "S3") & (ct.b == "S2a")]
    q1_sig = q1[(q1.diff_pct < 0) & q1.ci_separated & q1.beyond_delta]
    q2 = ct[(ct.a == "S3") & (ct.b == "S0") & (ct.gap == "inf")]
    s5 = ct[(ct.a == "S5") & (ct.b == "S0")]
    V["A1"] = {"Q1_S3_faster_than_S2a_significant_cells": f"{len(q1_sig)}/{len(q1)}",
               "Q1_cells": q1_sig[["ssd_dev", "gpu_state", "gap", "L", "diff_pct"]].to_dict("records"),
               "Q2_S3_vs_S0_gapinf": q2[["ssd_dev", "gpu_state", "L", "diff_pct", "ci_separated"]].to_dict("records"),
               "S5_vs_S0_diff_pct_range": [float(s5.diff_pct.min()), float(s5.diff_pct.max())],
               "kv_bad_total": int(pd.to_numeric(d.kv_bad, errors="coerce").fillna(0).sum()),
               "run_ids": sorted(d.run_id.unique())}
    # 圖 1：時間花在哪裡（L=16K、32K；gap=inf；每個 GPU 狀態；兩個 SSD 參數各一張）
    for dev in g.ssd_dev.unique():
        fig, axs = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
        for ax, L in zip(axs, (16384, 32768)):
            h = g[(g.ssd_dev == dev) & (g.L == L) & (g.gap == "inf")]
            labels, y = [], 0
            for gs in ("idle", "busy1", "busy2"):
                for st in ("S0", "S2a", "S3", "S5", "CPUall"):
                    r = h[(h.gpu_state == gs) & (h.strategy == st)]
                    if not len(r):
                        continue
                    r = r.iloc[0]
                    ax.barh(y + 0.18, r.t_rec * 1e3, height=0.32, color=BLUE, edgecolor="white", linewidth=1)
                    # 載入線：依 chunk 數把時間分給 CPU / SSD
                    tot = r.n_cpu + r.n_ssd
                    fc = r.n_cpu / tot if tot else 0
                    ax.barh(y - 0.18, r.t_load * 1e3 * fc, height=0.32, color=AQUA, edgecolor="white", linewidth=1)
                    ax.barh(y - 0.18, r.t_load * 1e3 * (1 - fc), left=r.t_load * 1e3 * fc, height=0.32,
                            color=ORANGE, edgecolor="white", linewidth=1)
                    ax.plot([r.ttft * 1e3], [y], marker="|", ms=14, mew=2, color="#0b0b0b")
                    ax.plot([r.ttft_lo * 1e3, r.ttft_hi * 1e3], [y, y], color="#0b0b0b", lw=0.8)
                    ax.text(r.ttft * 1e3, y + 0.42, f" {r.ttft*1e3:.0f} ms｜會合 {r.meet:.0f}/{L//512}",
                            fontsize=6.5, color=INK2, va="center")
                    labels.append(f"{st}｜{ {'idle':'GPU 閒','busy1':'忙1 (1.5×)','busy2':'忙2 (2.1×)'}[gs] }")
                    y += 1
                y += 0.5
            ax.set_yticks([i for i in np.arange(0, y) if True][:0])
            ax.set_title(f"L = {L//1024}K", loc="left", fontsize=9)
            ax.set_xlabel("ms")
        ys, lab, yy = [], [], 0
        for gs in ("idle", "busy1", "busy2"):
            for st in ("S0", "S2a", "S3", "S5", "CPUall"):
                ys.append(yy); yy += 1
            yy += 0.5
        axs[0].set_yticks(ys, labels[:len(ys)])
        axs[0].invert_yaxis()   # sharey：只翻一次
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        fig.legend(handles=[Patch(color=BLUE, label="重算線（GPU）"), Patch(color=AQUA, label="載入線：CPU 層"),
                            Patch(color=ORANGE, label="載入線：SSD 層"),
                            Line2D([], [], marker="|", ms=10, mew=2, color="#0b0b0b", lw=0, label="TTFT 中位數（線＝95% CI）")],
                   loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.04))
        fig.suptitle(f"圖 1　時間花在哪裡（A1，單一 session，寫完才回來；SSD 層＝{ {'local':'本地 SSD','nfs':'NFS'}[dev] } 實測參數的限速器）",
                     x=0.01, ha="left", fontsize=9.5)
        save(fig, f"fig1_breakdown_{dev}.png")


# =========================================================== A2
def a2():
    rows = []
    for m in ("fifo", "share"):
        d = rd(f"a2_{m}.csv")
        if d is None:
            continue
        d = d.copy()
        d["model"] = np.where(d.io == "real", "real", "sim_" + m)
        rows.append(d)
    if not rows:
        return
    d = pd.concat(rows)
    d["ms_per_load"] = d.t_load_ssd / d.n_load_ssd.replace(0, np.nan) * 1e3
    g = d.groupby(["device", "model", "bg_write"]).agg(ttft=("ttft", "median"), lo=("ttft", lambda x: med_ci(x)[1]),
                                                       hi=("ttft", lambda x: med_ci(x)[2]), meet=("meet", "median"),
                                                       ms_load=("ms_per_load", "median"),
                                                       bg_s=("bg_write_s", "median"), n=("ttft", "size"),
                                                       cont_disk=("contaminated_disk", "sum")).reset_index()
    g.to_csv(os.path.join(R, "summary_a2.csv"), index=False)
    real = g[g.model == "real"].set_index(["device", "bg_write"])
    V["A2"] = {f"{dev}_{bw}_vs_none_pct": float((real.loc[(dev, bw)].ttft / real.loc[(dev, "none")].ttft - 1) * 100)
               for dev in g.device.unique() for bw in ("S3_partial", "S0_full")}
    V["A2"]["S3_partial_vs_S0_full_pct"] = {dev: float((real.loc[(dev, "S3_partial")].ttft / real.loc[(dev, "S0_full")].ttft - 1) * 100)
                                            for dev in g.device.unique()}
    # 圖 2
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.0))
    for ax, dev in zip(axs, ("local", "nfs")):
        h = g[g.device == dev]
        models = [m for m in ("real", "sim_share", "sim_fifo") if m in h.model.values]
        x = np.arange(3)
        w = 0.8 / len(models)
        for k, m in enumerate(models):
            hh = h[h.model == m].set_index("bg_write").reindex(["none", "S3_partial", "S0_full"])
            xs = x + (k - (len(models) - 1) / 2) * w
            cap = {"local": 2400, "nfs": 2000}[dev]
            vals = (hh.ttft * 1e3).values
            ax.bar(xs, np.minimum(vals, cap), width=w * 0.92, color=[BLUE, AQUA, GRAY2][k],
                   label={"real": "真實裝置", "sim_share": "限速器（share，A2 校準）",
                          "sim_fifo": "限速器（FIFO，原設計）"}[m])
            ax.errorbar(xs, vals, yerr=[(hh.ttft - hh.lo) * 1e3, (hh.hi - hh.ttft) * 1e3], fmt="none",
                        ecolor="#0b0b0b", lw=0.8)
            for xi, v in zip(xs, vals):
                ax.text(xi, min(v, cap) + cap * 0.01, f"{v:.0f}" + (" ↑" if v > cap else ""), ha="center",
                        va="bottom", fontsize=6.5, color=INK2)
            ax.set_ylim(0, cap * 1.08)
        ax.set_xticks(x, ["背景不寫", "背景寫後段\n(S3 的量)", "背景全寫\n(S0/S2a 的量)"])
        ax.set_ylabel("還原者的 TTFT（ms）")
        ax.set_title({"local": "本地 SSD（/var/tmp）", "nfs": "NFS（/mlsteam/data）"}[dev], loc="left", fontsize=9)
    axs[1].legend(fontsize=7, frameon=False)
    fig.suptitle("圖 2　寫入會不會拖慢正在還原的人（A2：L=16K 的 session 用 Cake 還原，同時有人寫 32K 的 KV）",
                 x=0.01, y=1.04, ha="left", fontsize=9.5)
    save(fig, "fig2_write_interference.png")


# =========================================================== B
SIMPLE = ["S0", "S1", "S2b", "S4", "S4+"]  # 照 05 §6 寫死的 5 個；S4+P（Pensieve 原文）事後新增，另報；R0 只當參考；S5s 為探索組


def b():
    files = [f for f in os.listdir(R) if f.startswith("b_") and f.endswith(".csv")]
    if not files:
        return
    parts = []
    for f in files:
        if os.path.getsize(os.path.join(R, f)) == 0:   # 正在跑、表頭還沒寫出
            continue
        x = pd.read_csv(os.path.join(R, f))
        x["cpu_tier"] = ("vllm_" + f.split("_cpu")[1][:-4]) if "_cpu" in f else "harness"
        parts.append(x)
    d = pd.concat(parts)
    # GPU 變慢的短暫插曲（2026-10-08 在 A1 看到：無外來 pid，但純 GPU 計算忽然慢 1.3–2.5 倍、持續數秒）：
    # 每個請求自己的 t_new（算新的 8K）÷ 同一個歷史長度下的最小值 > 1.15 → 標記 gpu_slow
    d["t_new_ratio"] = d.t_new / d.groupby("hist_chunks").t_new.transform("min")
    d["gpu_slow"] = d.t_new_ratio > 1.15
    V["B_gpu_slow_events"] = {"total": int(d.gpu_slow.sum()), "of": int(len(d))}
    d = d[d["round"] > 1]   # 只看回來的請求
    keys = ["io_model", "cpu_tier", "ssd_dev", "gap_s", "ssd_frac", "wl_seed", "cpu_frac", "strategy"]
    rows = []
    for k, h in d.groupby(keys):
        m, lo, hi = med_ci(h.ttft)
        rows.append(dict(zip(keys, k), n=len(h), reps=h.rep.nunique(), median=m, ci_lo=lo, ci_hi=hi, mean=h.ttft.mean(),
                         p90=float(np.percentile(h.ttft, 90)), miss=h.n_missing.sum(), ld_cpu=h.n_load_cpu.sum(),
                         ld_ssd=h.n_load_ssd.sum(), recompute=h.n_recompute.sum(),
                         t_recompute=h.t_recompute.mean(), w_ssd_GiB=h.groupby("rep").w_bytes_ssd.max().median() / 2**30,
                         w_cpu_GiB=h.groupby("rep").w_bytes_cpu.max().median() / 2**30,
                         demote=h.groupby("rep").n_demote.max().median(), backlog=h.ssd_backlog_s.mean(),
                         kv_bad=pd.to_numeric(h.kv_bad, errors="coerce").fillna(0).sum(),
                         cont_gpu=int((h.contaminated_gpu.astype(str) == "True").sum())))
    t = pd.DataFrame(rows)
    # 排除 gpu_slow 事件後的中位數（敏感度）
    ex = d[~d.gpu_slow].groupby(keys).ttft.median().rename("median_excl_slow")
    sl = d.groupby(keys).gpu_slow.sum().rename("n_gpu_slow")
    t = t.merge(ex.reset_index(), on=keys, how="left").merge(sl.reset_index(), on=keys, how="left")
    t.to_csv(os.path.join(R, "summary_b.csv"), index=False)
    delta = V.get("calib", {}).get("C4", {}).get("delta_pct", np.nan)
    verdicts = []
    for k, h in t.groupby(keys[:-1]):
        h = h.set_index("strategy")
        simple = h.loc[[s for s in SIMPLE if s in h.index]]
        if "S5" not in h.index or not len(simple):
            continue
        best = simple["median"].idxmin()
        s5 = h.loc["S5"]
        bb = h.loc[best]
        gain = (bb["median"] - s5["median"]) / bb["median"] * 100
        sep = (s5.ci_hi < bb.ci_lo) or (bb.ci_hi < s5.ci_lo)
        s4p = h.loc["S4+"] if "S4+" in h.index else None
        # 成對：同一事件序列（同 rep、同 ev）的 S5 vs best
        sel = d
        for kk, vv in zip(keys[:-1], k):
            sel = sel[sel[kk] == vv]
        pa = sel[sel.strategy == "S5"].set_index(["rep", "ev"]).ttft
        pb = sel[sel.strategy == best].set_index(["rep", "ev"]).ttft
        j = pa.index.intersection(pb.index)
        paired = ((pb[j] - pa[j]) / pb[j] * 100) if len(j) else pd.Series(dtype=float)
        # 判準（05 §6）：S5 要比最好的簡單策略好 >5% 才有意義；CI 不重疊且 >δ 才算數
        if gain > 15 and sep and gain > delta:
            verdict = "> 15%：進第二階段"
        elif gain >= 5 and sep and gain > delta:
            verdict = "5–15%：找老師"
        elif gain >= 5:
            verdict = f"S5 中位數好 {gain:.1f}%，但 CI 重疊 → 不算數"
        elif gain > -5:
            verdict = f"差距 {gain:+.1f}%（< 5%）：簡單策略追平"
        else:
            verdict = f"S5 比最好的簡單策略差 {-gain:.1f}%" + ("（CI 不重疊）" if sep else "（中位數；CI 重疊）")
        verdicts.append(dict(zip(keys[:-1], k), best_simple=best, best_median=bb["median"], s5_median=s5["median"],
                             gain_pct=gain, ci_separated=sep, delta_pct=delta,
                             s4plus_median=None if s4p is None else s4p["median"],
                             s5_vs_s4plus_pct=None if s4p is None else (s4p["median"] - s5["median"]) / s4p["median"] * 100,
                             paired_mean_gain_pct=float(paired.mean()) if len(paired) else None,
                             paired_median_gain_pct=float(paired.median()) if len(paired) else None,
                             s4plusP_median=h.loc["S4+P"]["median"] if "S4+P" in h.index else None,
                             s5s_median=h.loc["S5s"]["median"] if "S5s" in h.index else None,
                             n_reps=int(s5.reps), verdict=verdict,
                             gain_pct_excl_slow=(bb["median_excl_slow"] - s5["median_excl_slow"]) / bb["median_excl_slow"] * 100,
                             best_simple_excl_slow=simple["median_excl_slow"].idxmin()))
    vt = pd.DataFrame(verdicts)
    vt.to_csv(os.path.join(R, "summary_b_verdict.csv"), index=False)
    V["B"] = vt.to_dict("records")
    # 圖 3：每個設定一張
    for k, h in t.groupby(keys[:-2]):
        cfg = dict(zip(keys[:-2], k))
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
        sty = {"S5": dict(color=BLUE, lw=2.6, ls="-", z=4), "S4+": dict(color=ORANGE, lw=2.0, ls=(0, (4, 2)), z=5),
               "S4": dict(color=VIOLET, lw=1.6, ls="-", z=3), "S4+P": dict(color=AQUA, lw=1.6, ls="-", z=3),
               "S1": dict(color=YELLOW, lw=1.4, ls="-", z=2), "S2b": dict(color=MAGENTA, lw=1.4, ls="-", z=2),
               "S0": dict(color=GREEN, lw=1.2, ls="-", z=1), "S3": dict(color=GRAY, lw=1.2, ls="-", z=1),
               "S5s": dict(color=GRAY, lw=1.2, ls=(0, (1, 1.5)), z=1)}
        for ax, stat in zip(axs, ("median", "p90")):
            for st in ["S0", "S3", "S5s", "S1", "S2b", "S4", "S4+P", "S5", "S4+"]:
                hh = h[h.strategy == st].sort_values("cpu_frac")
                if not len(hh):
                    continue
                y = sty[st]
                ax.plot(hh.cpu_frac * 100, hh[stat] * 1e3, marker="o", ms=4.5, lw=y["lw"], ls=y["ls"], color=y["color"],
                        label={"S5s": "S5s（探索）", "S4+P": "S4+P（Pensieve 原文）"}.get(st, st), zorder=y["z"])
            ax.set_xlabel("CPU 層容量（工作集的 %）")
            ax.set_ylabel(f"回來請求的 TTFT {'中位數' if stat == 'median' else 'P90'}（ms）")
            ax.set_xticks(sorted(h.cpu_frac.unique() * 100))
            ax.set_title({"median": "中位數", "p90": "P90"}[stat], loc="left", fontsize=9)
        axs[1].legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        r0 = h[h.strategy == "R0"]
        note = f"R0（只載，參考）中位數 {', '.join(f'{x*1e3:.0f}' for x in r0.sort_values('cpu_frac')['median'])} ms，未畫" if len(r0) else ""
        fig.suptitle(f"圖 3　簡單策略會不會追平（B：8 session × ≤4 輪，每輪 +8K；SSD 層＝{cfg['ssd_dev']} 限速器，"
                     f"干擾 {cfg['io_model']}，CPU 層 {cfg['cpu_tier']}，間隔 {cfg['gap_s']} s，SSD 容量 {cfg['ssd_frac']:.0%}，seed {cfg['wl_seed']}）\n{note}；S5 與 S4+ 重疊（虛線＝S4+）",
                     x=0.01, y=1.06, ha="left", fontsize=9)
        save(fig, f"fig3_b_{cfg['io_model']}_{cfg['cpu_tier']}_{cfg['ssd_dev']}_gap{cfg['gap_s']}_ssd{cfg['ssd_frac']}_seed{cfg['wl_seed']}.png")


if __name__ == "__main__":
    calib()
    for fn in (a0, a1, a2, b):
        try:
            fn()
        except Exception as e:  # 缺資料時記下來，不靜默
            import traceback
            V.setdefault("errors", []).append(f"{fn.__name__}: {e!r}")
            traceback.print_exc()
    json.dump(V, open(os.path.join(R, "verdict.json"), "w"), indent=1, ensure_ascii=False, default=str)
    print(json.dumps(V, indent=1, ensure_ascii=False, default=str)[:6000])
