"""Cake 的增益 vs「存／算」比（只報告；CPU，只讀主線的速度格子）。

使用者 10/19 問：Cake 論文說「存（載入）和算差太多就沒意義」，我們的實驗在哪？
每格：只算（compute_only）C、只載（load_only，同配置）L、Cake K；比＝L ÷ C；增益＝min(C, L) ÷ K（> 1 才是 Cake 比單一做法好）。
來源：results/m10_precision_mi300x/speed_{llama,qwen}/summary_{fp8,int4}.csv（Qwen FP8 用 speed_qwen_regrid）。32K。
用法：python code/m10_cake_ratio.py → results/m10_precision_mi300x/analysis/cake_ratio.{md,png}
"""
import csv, os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(REPO, "results", "m10_precision_mi300x")
SRC = [("llama", "fp8", "speed_llama/summary_fp8.csv"), ("llama", "int4", "speed_llama/summary_int4.csv"),
       ("qwen", "fp8", "speed_qwen_regrid/summary_fp8.csv"), ("qwen", "int4", "speed_qwen/summary_int4.csv")]
TIERS = ["cpu", "b16", "b32", "b48"]


def cells():
    out = []
    for m, tag, f in SRC:
        rows = [r for r in csv.DictReader(open(os.path.join(P, f))) if r["length"] == "32768"]
        get = {(r["bw_name"], r["gpu_state"], r["mode"], r["config"]): r for r in rows}
        for st in ("idle", "busy"):
            for bw in TIERS:
                c = get.get((bw, st, "compute_only", "-"))
                for cfg in (["E0"] if tag == "fp8" else []) + ["E1"]:
                    l, k = get.get((bw, st, "load_only", cfg)), get.get((bw, st, "cake", cfg))
                    if not (c and l and k):
                        continue
                    C, L, K = (float(x["median_t_restore"]) for x in (c, l, k))
                    out.append(dict(model=m, fmt="bf16" if cfg == "E0" else tag, gpu=st, tier=bw, C=C, L=L, K=K,
                                    ratio=L / C, gain=min(C, L) / K, b=k["b_values"], run=k["run_ids"]))
    return out


def main():
    cs = cells()
    d = os.path.join(P, "analysis"); os.makedirs(d, exist_ok=True)
    L = ["# Cake 的增益 vs「載入 ÷ 重算」（只報告；`code/m10_cake_ratio.py`）", "",
         "32K、主線速度格子（原寫法的拆）。比＝只載 ÷ 只算；增益＝min(只算, 只載) ÷ Cake（> 1：Cake 比單一做法快）。", "",
         "| 層 | 比的範圍 | Cake 增益的範圍 | 格數 |", "|:--|:--|:--|--:|"]
    for t in TIERS:
        x = [c for c in cs if c["tier"] == t]
        if x:
            L.append(f"| {t} | {min(c['ratio'] for c in x):.3f}–{max(c['ratio'] for c in x):.3f} | "
                     f"{min(c['gain'] for c in x):.2f}–{max(c['gain'] for c in x):.2f}× | {len(x)} |")
    L += ["", "| 模型 | 格式 | GPU | 層 | 只算 (s) | 只載 (s) | 比 | Cake (s) | 會合點 b | 增益 |", "|:--|:--|:--|:--|--:|--:|--:|--:|:--|--:|"]
    for c in sorted(cs, key=lambda c: c["ratio"]):
        L.append(f"| {c['model']} | {c['fmt']} | {c['gpu']} | {c['tier']} | {c['C']:.3f} | {c['L']:.3f} | {c['ratio']:.3f} | {c['K']:.3f} | {c['b']} | {c['gain']:.2f}× |")
    L += ["", "run：" + "、".join(sorted({c["run"] for c in cs}))]
    open(os.path.join(d, "cake_ratio.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:10]))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        style = {"cpu": ("o", "#1f6fb2"), "b16": ("s", "#3a9e5f"), "b32": ("^", "#c9822b"), "b48": ("D", "#b23a48")}
        for t in TIERS:
            x = [c for c in cs if c["tier"] == t]
            ax.scatter([c["ratio"] for c in x], [c["gain"] for c in x], marker=style[t][0], s=36, color=style[t][1],
                       edgecolor="white", linewidth=1.5, label={"cpu": "CPU tier (real)", "b16": "b16 (~5 GiB/s)",
                                                                "b32": "b32 (~1.5 GiB/s)", "b48": "b48 (~0.4 GiB/s)"}[t])
        ax.axhline(1.0, color="#888", lw=1, ls="--")
        ax.axvline(1.0, color="#bbb", lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("load-only time / compute-only time (log)")
        ax.set_ylabel("Cake speedup vs the faster single method")
        ax.set_title("Cake helps only when load and compute are comparable (32K)")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(d, "cake_ratio.png"), dpi=150)
        print("wrote", os.path.join(d, "cake_ratio.png"))
    except ImportError:
        print("matplotlib 不在，沒畫圖")


if __name__ == "__main__":
    main()
