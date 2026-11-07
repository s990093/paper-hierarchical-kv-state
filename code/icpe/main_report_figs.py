"""main_report_figs.py — icpe-main：2026-11-07 報告（以版本 B 為準）的 5 張圖，只讀已進 git 的彙總 CSV，不重算任何上限。

  python -I code/icpe/main_report_figs.py --verb-dir results/icpe_next/main/verB --p4-dir results/icpe_next/main/p4/noise \
      --w3-rows <tracelab_c1 的 w3_rows.csv> --out-dir <dir>

圖（PNG，淺色底）＋ fig_data.csv（每張圖畫的每個點；有 run_id、ts、host、來源檔）：
  fig1_verB_max.png     每個紀錄變體的最大上限（5 個判定元件；R1／R2；版本 B）
  fig2_components.png   上限 ≥15% 的（格, 元件）落在哪個元件（H2）
  fig3_cpu.png          上限 ≥15% 的格占比 vs CPU 容量（H3）
  fig4_p4.png           P4：中位數仍 ≥15% 的（格, 元件）占比 vs λ（P-B1 主、P-B2、M-B1）
  fig5_w3.png           W3（TraceLab c1）：22 個 ≥15% 的（格, 元件），④ 與 ③ 相對 ② 的 TTFT
"""
from __future__ import annotations

import argparse
import os
import socket
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

JUDGED = ["O-ORD", "O-DROP", "O-NOW", "O-PF", "O-GPU"]
SURF, INK, INK2, GRID, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0", "#8a8984"
CLS = [("raw", True, "原始到達・工具類", "#2a78d6"), ("synthetic", True, "合成到達・工具類", "#eb6834"),
       ("raw", False, "原始到達・非工具類（只報告）", "#1baf7a")]
plt.rcParams.update({"font.family": ["Noto Sans CJK TC", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.axisbelow": True, "legend.frameon": False})
ROWS = []
PREFIX, LABEL, P4_EXCLUDE, W3_EXPECT = "verB", "版本 B", [], 22   # main() 依參數改（版本 C：verC／版本 C）


def rec(fig, series, label, x, y, src):
    ROWS.append(dict(fig=fig, series=series, label=label, x=x, y=y, src=src))


def short(trace, rule):
    t = trace.split(":", 1)[1] if trace.startswith(("nb:", "fi:")) else trace
    if trace.startswith("fi:"):
        return f"FreeInference {t[5:10]}"
    m = {"toolagent": "Mooncake toolagent", "conversation": "Mooncake 對話"}
    if t in m:
        return f"{m[t]} {rule}"
    if t.startswith("tracelab_c"):
        c = t[-1]
        return "TraceLab c1" if c == "1" else f"TraceLab c{c}（壓縮 {c}×）"
    if t.startswith("semianalysis_") or t.startswith("cachewise_"):
        f, n, s = t.split("_")
        return f"{'SemiAnalysis' if f == 'semianalysis' else 'CacheWise'} {n} {s}"
    if t.startswith("bailian_"):
        return f"Bailian {t.split('_', 1)[1]}"
    return t


def fig1(verb, out):
    src = f"{verb}/{PREFIX}_verdict.csv"
    V = pd.read_csv(src)
    V = V[V.level == "variant"].copy()
    fig, ax = plt.subplots(figsize=(7.2, 9.6))
    y, ticks, labels = 0, [], []
    for arr, tool, name, col in CLS:
        g = V[(V.arrival == arr) & (V.tool_type == tool)].copy()
        g["lab"] = [short(t, r) for t, r in zip(g.trace, g.rule)]
        g = g.sort_values("max_judged", ascending=False)
        ys = y + 1.2 + np.arange(len(g))
        ax.scatter(g.max_judged * 100, ys, s=34, color=col, edgecolor=SURF, linewidth=1.5, zorder=3, label=name)
        for yy, (_, r) in zip(ys, g.iterrows()):
            rec("fig1", name, r.lab, round(float(r.max_judged) * 100, 3), int(yy), src)
        ticks += list(ys)
        labels += list(g.lab)
        ax.text(0.5, y + 0.1, name, color=INK, fontsize=9, ha="left", va="center", weight="bold")
        y = ys[-1] + 1.8
    ax.axvline(15, color=REF, linestyle="--", linewidth=1)
    ax.text(15.6, y - 0.6, "15%＝值得做預測（事先登記的門檻）", color=INK2, fontsize=8, va="center")
    ax.set_yticks(ticks, labels, fontsize=7.5)
    ax.set_xlim(0, 72)
    ax.set_ylim(y, -0.8)
    ax.set_xlabel("完美預知讓首字延遲（TTFT）比最好的不預測做法少多少（%）↑ 越大越值得做預測\n（每個變體：5 個判定元件 × 54 格裡的最大值）")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"圖 1　完美預知最多能少等多久（每個紀錄變體；{LABEL}）", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{out}/fig1_{PREFIX}_max.png", dpi=170)
    plt.close(fig)


def cls_of(df):
    return [next(n for a, t, n, _ in CLS if a == ar and t == tt) for ar, tt in zip(df.arrival, df.tool_type)]


def fig2(verb, out):
    src = f"{verb}/{PREFIX}_by_trace.csv"
    B = pd.read_csv(src)
    B = B[B.verdict_use].copy()
    B["cls"] = cls_of(B)
    S = B.groupby(["cls", "component"]).n_ge15.sum()
    comps = ["O-GPU", "O-ORD", "O-PF", "O-DROP", "O-NOW"]
    nice = {"O-GPU": "GPU 裡留誰\n(O-GPU)", "O-ORD": "CPU 淘汰順序\n(O-ORD)", "O-PF": "預取\n(O-PF)",
            "O-DROP": "丟不丟\n(O-DROP)", "O-NOW": "回合結束就丟\n(O-NOW)"}
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    w = 0.26
    x = np.arange(len(comps))
    for i, (_, _, name, col) in enumerate(CLS):
        v = [int(S.get((name, c), 0)) for c in comps]
        tot = sum(v)
        bars = ax.bar(x + (i - 1) * w, v, w - 0.03, color=col, label=f"{name}（共 {tot}）", zorder=3)
        for b, vv, c in zip(bars, v, comps):
            rec("fig2", name, c, c, vv, src)
            if vv:
                ax.text(b.get_x() + b.get_width() / 2, vv + 6, str(vv), ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(x, [nice[c] for c in comps])
    ax.set_ylabel("能少等 ≥15% 的（格, 元件）數")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"圖 2　能少等 ≥15% 的地方，是靠預測哪個動作（{LABEL}）", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{out}/fig2_components.png", dpi=170)
    plt.close(fig)


def fig3(verb, out):
    src = f"{verb}/{PREFIX}_by_cpu.csv"
    C = pd.read_csv(src)
    C = C[C.component.isin(JUDGED) & (C.rule != "min")].copy()
    C["cls"] = cls_of(C)
    T = C.groupby(["cls", "cpu_GiB"]).agg(n=("n_cells", "sum"), k15=("n_ge15", "sum")).reset_index()
    cpus = sorted(T.cpu_GiB.unique())
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(cpus))
    for _, _, name, col in CLS:
        g = T[T.cls == name].set_index("cpu_GiB").loc[cpus]
        share = g.k15 / g.n * 100
        ax.plot(x, share, color=col, linewidth=2, marker="o", markersize=5, markeredgecolor=SURF, label=name, zorder=3)
        for xx, c, s in zip(x, cpus, share):
            rec("fig3", name, f"{c:g} GiB", float(c), round(float(s), 3), src)
    ax.set_xticks(x, [("111\n(=GPU)" if abs(c - 111.0625) < 1e-6 else f"{c:g}") for c in cpus])
    ax.set_xlabel("CPU 記憶體容量（GiB；111＝和 GPU 的 KV 預算一樣大）")
    ax.set_ylabel("能少等 ≥15% 的格子比例（%）")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    ax.set_title(f"圖 3　CPU 記憶體多大時最值得預測：中間最大（{LABEL}）", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{out}/fig3_cpu.png", dpi=170)
    plt.close(fig)


def fig4(p4, out):
    src = f"{p4}/p4_by_class.csv"
    P = pd.read_csv(src)
    if P4_EXCLUDE:   # 論文版（偏離 15）：從逐變體表重新加總，拿掉這些紀錄
        src = f"{p4}/p4_by_trace.csv"
        T = pd.read_csv(src)
        T = T[T.judged & ~T.trace.str.contains("|".join(P4_EXCLUDE))]
        P = T.groupby(["arrival", "tool_type", "noise_model", "noise_draw", "noise_lambda"])[["n_cellcomp", "n_med_ge15"]].sum().reset_index()
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.3), sharey=True)
    series = [("P", "B1", "P-B1：λ 的預測換成亂猜，每回合抽一次（主）", "#2a78d6", "-"),
              ("P", "B2", "P-B2：同上，每次查詢重抽", "#eb6834", "--"),
              ("M", "B1", "M：回來時間 ×(1±λ)", "#1baf7a", ":")]
    for ax, (arr, tool, name, _) in zip(axs, CLS):
        g = P[(P.arrival == arr) & (P.tool_type == tool)]
        base = int(g[(g.noise_model == "P") & (g.noise_draw == "B1") & (g.noise_lambda == 0)].n_cellcomp.iloc[0])
        for m, d, lab, col, ls in series:
            h = g[(g.noise_model == m) & (g.noise_draw == d)].sort_values("noise_lambda")
            h = h[h.noise_lambda > 0]
            xs = [0.0] + list(h.noise_lambda)
            ys = [100.0] + list(h.n_med_ge15 / base * 100)
            ax.plot(xs[1:], ys[1:], color=col, linestyle=ls, linewidth=2, marker="o", markersize=4,
                    markeredgecolor=SURF, label=lab, zorder=3)
            ax.plot(xs[:2], ys[:2], color=col, linestyle=ls, linewidth=1, alpha=0.5, zorder=2)
            for xx, yy, nn in zip(h.noise_lambda, h.n_med_ge15 / base * 100, h.n_med_ge15):
                rec("fig4", f"{name}|{lab}", f"n={int(nn)}/{base}", float(xx), round(float(yy), 3), src)
        ax.scatter([0], [100], s=30, color=INK, zorder=4)
        ax.set_title(f"{name}\n（{base} 個（格, 元件））", fontsize=8.5, loc="left")
        ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4], ["0", "10%", "20%", "30%", "40%"])
        ax.set_xlabel("預測錯誤強度 λ")
        ax.set_ylim(0, 105)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("仍能少等 ≥15% 的比例（%）")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7.5, loc="lower center", ncol=2)
    fig.suptitle("圖 4　預測不準時，還剩多少值得做（10 個 seed 的中位數；λ＝0＝沒有雜訊；P4 以版本 B 選格）", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(f"{out}/fig4_p4.png", dpi=170)
    plt.close(fig)


def fig5(w3, out):
    W = pd.read_csv(w3)
    W = W[W.trace == "nb:tracelab_c1"]          # w3 檔可能含 c2、c4、FreeInference（main_w3c 的輸出）
    g = W[W.judged & (W.headroom >= 0.15)].copy()
    assert W3_EXPECT is None or len(g) == W3_EXPECT, f"W3 c1 判定的 ≥15% 格應為 {W3_EXPECT}，實際 {len(g)}"
    g["r4"] = g.T_oracle / g.T_rival
    g["r3"] = g.T_cue / g.T_rival
    g = g.sort_values(["component", "cpu_GiB", "slow", "gpu"]).reset_index(drop=True)
    gl = {"idle": "閒", "busy": "忙"}
    g["lab"] = [f"{c} {('111' if abs(cpu - 111.0625) < 1e-6 else f'{cpu:g}')} GiB {s} {gl.get(u, u)}" for c, cpu, s, u in zip(g.component, g.cpu_GiB, g.slow, g.gpu)]
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ys = np.arange(len(g))
    for yy, a, b in zip(ys, g.r4, g.r3):
        ax.plot([a, b], [yy, yy], color=GRID, linewidth=1.2, zorder=1)
    ax.scatter(g.r4, ys, s=30, color="#2a78d6", edgecolor=SURF, linewidth=1.2, zorder=3, label="④ 完美預知")
    ax.scatter(g.r3, ys, s=30, color="#eb6834", edgecolor=SURF, linewidth=1.2, zorder=3, label="③ 工具呼叫線索（不用預測；CUEP／CUEO 取快）")
    for yy, (_, r) in zip(ys, g.iterrows()):
        rec("fig5", "oracle", r.lab, round(float(r.r4), 4), int(yy), w3)
        rec("fig5", "cue", r.lab, round(float(r.r3), 4), int(yy), w3)
    ax.axvline(1, color=REF, linestyle="--", linewidth=1)
    ax.text(1.02, len(g) - 0.3, "② 最好的固定規則＝1", color=INK2, fontsize=8)
    ax.set_xscale("log")
    tk = [t for t in [0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8] if t <= max(4, float(g.r3.max()) * 1.1)]
    ax.set_xticks(tk, [f"{t:g}" for t in tk])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks(ys, g.lab, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("首字延遲 ÷ 最好的固定規則（對數軸）↓ 越小越好；<1＝比固定規則快")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=2)
    ax.set_title(f"圖 5　免費線索拿得走嗎？TraceLab c1 的 {len(g)} 格都比固定規則慢（{LABEL}）", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{out}/fig5_w3.png", dpi=170)
    plt.close(fig)


def table_main(verb, out):
    """主結果表（照 PBKV Table 1／SAGA Table 3 的排法：列＝紀錄，欄＝元件；格＝最大值與 ≥15% 的格數）。"""
    src = f"{verb}/{PREFIX}_by_trace.csv"
    B = pd.read_csv(src)
    B = B[B.verdict_use].copy()
    B["cls"] = cls_of(B)
    def grp(t, r):
        x = t.split(":", 1)[1] if t.startswith(("nb:", "fi:")) else t
        if t.startswith("fi:"):
            return "FreeInference（6 週）"
        if x.startswith("semianalysis_") or x.startswith("cachewise_"):
            f, n, _ = x.split("_")
            return f"{'SemiAnalysis' if f == 'semianalysis' else 'CacheWise'} 並行 {n[1:]}（3 個 seed）"
        return short(t, r)
    B["grp"] = [grp(t, r) for t, r in zip(B.trace, B.rule)]
    comps = ["O-GPU", "O-ORD", "O-PF", "O-DROP", "O-NOW"]
    rows = []
    for (cl, g), d in B.groupby(["cls", "grp"], sort=False):
        r = dict(cls=cl, group=g, n_variants=d[["trace", "rule"]].drop_duplicates().shape[0])
        for c in comps:
            e = d[d.component == c]
            r[f"{c}_max_pct"] = round(float(e["max"].max()) * 100, 1)
            r[f"{c}_n_ge15"] = int(e.n_ge15.sum())
            r[f"{c}_n_cells"] = int(e.n_cells.sum())
        r["max_pct"] = max(r[f"{c}_max_pct"] for c in comps)
        rows.append(r)
    T = pd.DataFrame(rows)
    order = [n for _, _, n, _ in CLS]
    T["o"] = T.cls.map(order.index)
    T = T.sort_values(["o", "max_pct"], ascending=[True, False]).drop(columns="o")
    T.insert(0, "src", src)
    T.to_csv(f"{out}/table_main_{PREFIX}.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verb-dir", required=True)
    ap.add_argument("--p4-dir", required=True)
    ap.add_argument("--w3-rows", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="verB", help="上限表的檔名前綴（verB／verC）")
    ap.add_argument("--label", default="版本 B")
    ap.add_argument("--p4-exclude", nargs="*", default=[], help="P4 加總時拿掉的紀錄（字串包含）；例：cachewise（偏離 15）")
    ap.add_argument("--w3-expect", type=int, default=22)
    a = ap.parse_args()
    global PREFIX, LABEL, P4_EXCLUDE, W3_EXPECT
    PREFIX, LABEL, P4_EXCLUDE, W3_EXPECT = a.prefix, a.label, a.p4_exclude, a.w3_expect
    os.makedirs(a.out_dir, exist_ok=True)
    fig1(a.verb_dir, a.out_dir)
    fig2(a.verb_dir, a.out_dir)
    fig3(a.verb_dir, a.out_dir)
    fig4(a.p4_dir, a.out_dir)
    fig5(a.w3_rows, a.out_dir)
    table_main(a.verb_dir, a.out_dir)
    meta = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"),
                host=socket.gethostname())
    D = pd.DataFrame(ROWS)
    for i, (k, v) in enumerate(meta.items()):
        D.insert(i, k, v)
    D.to_csv(f"{a.out_dir}/fig_data.csv", index=False)
    print(f"5 張圖、{len(D)} 個點 → {a.out_dir}")


if __name__ == "__main__":
    main()
