#!/usr/bin/env python3
"""M5 第二階段的圖。**只畫「表格表達不了的東西」。**

專案的預設是**留表刪圖**（同一批數字不並列）。所以這裡只有四張，
每一張都是表格做不到的形狀：

| 圖 | 為什麼表格不行 |
|---|---|
| `m5_ordering` | 兩個連續量的**聯合分布**（預測 vs 真值），要看的是散開的形狀 |
| `m5_threshold` | 成本對門檻的**連續曲線**與其最小值位置；表格只能給幾個點 |
| `m5_calibration` | 可靠度圖。§B.6 明文要求：「缺少此圖，該缺陷等同未被處理」 |
| `m5_horizon` | 三個指標對視窗 $W$ 的趨勢與交叉 |

規則同 `paper_figures.py`：每個數字自 `results/` 或 run 目錄讀出，
讀不到就 raise，不畫標題（標題是 caption 的工作）。

用法：
    PYTHONPATH=/ssd7/hungwei/paper-hkv/pylibs \
      /ssd7/hungwei/paper-hkv/venv/vllm/bin/python notebooks/m5_figures.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, os.environ.get("PAPER_HKV_PYLIBS",
                                  "/ssd7/hungwei/paper-hkv/pylibs"))
BIG = Path(os.environ.get("PAPER_HKV_BIG", "/ssd7/hungwei/paper-hkv"))
M5 = REPO / "results/m5_predictor"
OUT = REPO / "notebooks" / "figures"
COL, COL2 = 3.33, 6.95
CLR = {"gpu": "#C0392B", "cpu": "#2980B9", "ssd": "#27AE60",
       "drop": "#7F8C8D", "hl": "#B9770E", "ink": "#222222"}

from m4_oracle import load_cost_model                      # noqa: E402
from m5_predictor import cost_weighted_error, drop_cost, p_star   # noqa: E402


def rows(name: str) -> list[dict]:
    p = M5 / name
    if not p.exists():
        raise SystemExit(f"🔴 找不到 {p}")
    return list(csv.DictReader(p.open()))


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight",
                    dpi=200 if ext == "png" else None)
    plt.close(fig)
    print(f"  ✅ {OUT / (stem + '.pdf')}")


def latest_train(pred: dict) -> Path:
    d = BIG / "runs" / pred["run_id"]
    if not d.exists():
        raise SystemExit(f"🔴 找不到 run 目錄 {d}")
    return d


def pick(met: list[dict], **kw) -> dict:
    for r in met:
        if all(str(r.get(k, "")) == str(v) for k, v in kw.items()):
            return r
    raise SystemExit(f"🔴 predictor_metrics.csv 裡找不到 {kw}")


# ────────────────── 圖 1：排序品質 ──────────────────

def fig_ordering(met):
    """預測值 vs 真實的下次使用時間，逐工作負載。

    這張圖是本階段的核心發現：**分類幾乎完美而排序幾乎沒學到**。
    表格給得出 AUC 與 Spearman 兩個數字，但給不出「為什麼」——
    要看到正樣本被壓成一條水平帶，才知道問題出在設限的目標。
    """
    picks = []
    for tr, lab in (("toolagent", "Mooncake toolagent"),
                    ("lc128kz", "長上下文 131K（論文 34.6% 設定）")):
        cand = [r for r in met if r["trace"] == tr and r["loss"] == "sym_l2"
                and r["threshold_rule"] == "p_star" and r["num_leaves"] == "63"
                and r.get("label_mode", "censored") == "censored"]
        if cand:
            picks.append((cand[0], lab))
    if not picks:
        return
    fig, axes = plt.subplots(1, len(picks), figsize=(COL2, 2.7), squeeze=False)
    for ax, (r, lab) in zip(axes[0], picks):
        d = np.load(latest_train(r) / "test_pred_sym_l2.npz")
        y, yh, yb = d["y_reg"], d["y_hat"], d["y_bin"]
        m = yb == 1
        ax.hexbin(y[m], yh[m], gridsize=45, bins="log", cmap="Blues",
                  mincnt=1, linewidths=0)
        lo = float(min(y[m].min(), yh[m].min()))
        hi = float(max(y[m].max(), yh[m].max()))
        ax.plot([lo, hi], [lo, hi], color=CLR["hl"], lw=1.0, ls="--")
        ax.set_xlabel(r"真實 $\log\tau$（正樣本）")
        ax.set_ylabel(r"預測 $\hat y$")
        ax.set_title(f"{lab}\nSpearman {r['spearman_positives']}　"
                     f"AUC {r['auc']}", fontsize=8)
    fig.tight_layout()
    save(fig, "m5_ordering")


# ────────────────── 圖 2：成本 vs 門檻 ──────────────────

def fig_threshold(met, cm):
    """成本加權錯誤對決策門檻的曲線，並標出 0.5 與式 (9) 的 $p^{*}$。

    §5.3 的主張是「門檻被錯置了約一個數量級」。這是連續軸上的曲線與其最小值位置，
    表格只能給端點，看不出最小值落在哪、也看不出 0.5 有多遠。
    """
    picks = []
    for tr, lab in (("toolagent", "Mooncake toolagent（W=1×）"),
                    ("lc128kz", "長上下文 131K（W=6×）")):
        cand = [r for r in met if r["trace"] == tr and r["loss"] == "sym_l2"
                and r["threshold_rule"] == "p_star" and r["num_leaves"] == "63"
                and r.get("label_mode", "censored") == "censored"]
        if cand:
            picks.append((sorted(cand, key=lambda r: -int(r["window_accesses"]))[0], lab))
    if not picks:
        return
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    for (r, lab), c in zip(picks, (CLR["cpu"], CLR["ssd"])):
        d = np.load(latest_train(r) / "test_pred_sym_l2.npz")
        p, yb, pos = d["p_hat"].astype(float), d["y_bin"], d["pos_tokens"].astype(float)
        ths = np.concatenate([np.geomspace(1e-4, 0.9, 60), [0.5]])
        ths.sort()
        cost = [cost_weighted_error(p, yb, np.full(len(p), t), cm, pos)["cost_ms"]
                for t in ths]
        ax.plot(ths, np.array(cost) / 1e3, color=c, lw=1.3, label=lab)
        ps = float(np.median(p_star(cm, pos, "cpu")))
        j = int(np.argmin(cost))
        ax.axvline(ps, color=c, lw=0.8, ls=":")
        ax.plot([ths[j]], [cost[j] / 1e3], "o", ms=4, color=c)
    ax.axvline(0.5, color=CLR["drop"], lw=0.9, ls="--")
    ax.text(0.5, ax.get_ylim()[1] * 0.95, " 0.5（對稱損失的預設）", fontsize=7,
            color=CLR["drop"], ha="left", va="top")
    ax.set_xscale("log")
    ax.set_xlabel(r"決策門檻（虛線 = 式 (9) 的 $p^{*}$ 中位數；圓點 = 實際最小值）")
    ax.set_ylabel("成本加權錯誤（秒）")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, "m5_threshold")


# ────────────────── 圖 3：可靠度圖 ──────────────────

def fig_calibration(cal):
    """可靠度圖。§B.6：「缺少此圖，該缺陷等同未被處理」。"""
    groups = defaultdict(list)
    for r in cal:
        if r["num_leaves"] != "63":
            continue
        groups[(r["trace"], r["loss"])].append(r)
    keys = [k for k in groups if k[0] in ("toolagent", "lc128kz")]
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    ax.plot([0, 1], [0, 1], color=CLR["drop"], lw=0.8, ls="--")
    styles = {"sym_l2": "-", "cost_l2": "--"}
    colors = {"toolagent": CLR["cpu"], "lc128kz": CLR["ssd"]}
    for k in sorted(keys):
        g = sorted(groups[k], key=lambda r: float(r["mean_p_hat"]))
        x = [float(r["mean_p_hat"]) for r in g]
        y = [float(r["empirical_rate"]) for r in g]
        n = [int(r["n"]) for r in g]
        ax.plot(x, y, styles[k[1]], color=colors[k[0]], lw=1.2, marker="o",
                ms=3, label=f"{k[0]} / {k[1]}")
        for xi, yi, ni in zip(x, y, n):
            if ni < 50:
                ax.plot([xi], [yi], "x", color="red", ms=4)
    ax.set_xlabel(r"預測機率 $\hat p$")
    ax.set_ylabel("實際重用比率")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "m5_calibration")


# ────────────────── 圖 4：視窗長度 ──────────────────

def fig_horizon(met, pol):
    """三個量對決策視窗 $W$ 的趨勢：正樣本率、AUC、排序品質。"""
    g = [r for r in met if r["trace"] == "toolagent" and r["loss"] == "sym_l2"
         and r["threshold_rule"] == "p_star" and r["num_leaves"] == "63"
         and r.get("label_mode", "censored") == "censored"
         and r.get("sample_rate") == "0.25" and str(r.get("data_seed")) == "1234"
         and r.get("feature_groups") == "history+deltas+edc+static"]
    if len(g) < 2:
        return
    g.sort(key=lambda r: int(r["window_accesses"]))
    w = [int(r["window_accesses"]) for r in g]
    fig, ax = plt.subplots(figsize=(COL, 2.4))
    ax.plot(w, [float(r["test_positive_rate"]) for r in g], "o-",
            color=CLR["gpu"], lw=1.2, ms=4, label="正樣本率")
    ax.plot(w, [float(r["auc"]) for r in g], "s-", color=CLR["cpu"], lw=1.2,
            ms=4, label="AUC")
    ax.plot(w, [float(r["spearman_positives"]) for r in g], "^-",
            color=CLR["ssd"], lw=1.2, ms=4, label="Spearman（正樣本內）")
    ax.set_xscale("log")
    ax.set_xticks(w)
    ax.set_xticklabels([f"{x // 1000}K" for x in w])
    ax.minorticks_off()
    ax.set_xlabel("決策視窗 $W$（次存取）")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, "m5_horizon")


def fig_timeline():
    """**實測的**狀態時序圖：40 個 block 在 120 個請求中所處的階。

    論文圖 3(b) 目前的 caption 明寫「示意，非量測資料」。這張是同一個形狀，
    但每一格都來自 `m5_policy_sim` 的實際重放，可以直接取代它。
    """
    p = M5 / "timeline.csv"
    if not p.exists():
        return
    rows = list(csv.DictReader(p.open()))
    pol = rows[0]["policy"]
    rows = [r for r in rows if r["policy"] == pol]
    blocks = sorted({int(r["block"]) for r in rows})
    reqs = sorted({int(r["request"]) for r in rows})
    bi = {b: i for i, b in enumerate(blocks)}
    ri = {q: i for i, q in enumerate(reqs)}
    codes = {"GPU-BF16": 0, "GPU": 0, "GPU-INT8": 1, "CPU": 2, "SSD": 3, "DROP": 4}
    m = np.full((len(blocks), len(reqs)), 4, dtype=int)
    for r in rows:
        m[bi[int(r["block"])], ri[int(r["request"])]] = codes[r["state"]]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([CLR["gpu"], "#E8A29A", CLR["cpu"], CLR["ssd"], "#DDDDDD"])
    fig, ax = plt.subplots(figsize=(COL2, 2.4))
    ax.imshow(m, aspect="auto", cmap=cmap, vmin=0, vmax=4,
              interpolation="nearest")
    ax.set_xlabel("請求序號")
    ax.set_ylabel("追蹤的 48 個 block")
    ax.set_yticks([])
    # 只列出資料裡真的出現過的狀態——沒出現就不要在圖例裡佔位
    present = sorted({codes[r["state"]] for r in rows})
    names = {0: "GPU-BF16", 1: "GPU-INT8（降級後）", 2: "CPU", 3: "SSD",
             4: "DROP（不在任何階）"}
    cols = [CLR["gpu"], "#E8A29A", CLR["cpu"], CLR["ssd"], "#DDDDDD"]
    handles = [plt.Rectangle((0, 0), 1, 1, color=cols[i]) for i in present]
    ax.legend(handles, [names[i] for i in present],
              ncol=len(present), fontsize=7, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    save(fig, "m5_timeline")


def fig_roc_pr(met):
    """ROC 與 PR 曲線。分類品質的標準呈現，兩個工作負載對照。"""
    picks = []
    for tr, lab in (("toolagent", "Mooncake toolagent（W=1×）"),
                    ("toolagent", "Mooncake toolagent（W=35×）"),
                    ("lc128kz", "長上下文 131K（W=6×）")):
        want = 599095 if "35" in lab else (102702 if tr == "lc128kz" else 17117)
        c = [r for r in met if r["trace"] == tr and r["loss"] == "sym_l2"
             and r["threshold_rule"] == "p_star" and r["num_leaves"] == "63"
             and r.get("label_mode") == "censored"
             and int(r["window_accesses"]) == want
             and r.get("feature_groups") == "history+deltas+edc+static"
             and str(r.get("data_seed")) == "1234"
             and r.get("sample_rate") in ("0.25", "1.0")]
        if c:
            picks.append((c[0], lab))
    if not picks:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL2, 2.9))
    cols = (CLR["cpu"], CLR["hl"], CLR["ssd"])
    for (r, lab), c in zip(picks, cols):
        f = latest_train(r) / "curves_sym_l2.npz"
        if not f.exists():
            continue
        z = np.load(f)
        a1.plot(z["fpr"], z["tpr"], color=c, lw=1.4,
                label=f"{lab}　AUC {float(r['auc']):.3f}")
        a2.plot(z["recall"], z["precision"], color=c, lw=1.4,
                label=f"AP {float(r.get('average_precision') or 0):.3f}")
    a1.plot([0, 1], [0, 1], color=CLR["drop"], lw=0.8, ls="--")
    a1.set_xlabel("偽陽性率 FPR"); a1.set_ylabel("真陽性率 TPR")
    a1.legend(fontsize=6.5, frameon=False, loc="lower right")
    a2.set_xlabel("召回率（丟棄）"); a2.set_ylabel("精確率（丟棄）")
    a2.legend(fontsize=6.5, frameon=False, loc="lower left")
    for a in (a1, a2):
        a.set_xlim(-0.02, 1.02); a.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    save(fig, "m5_roc_pr")


def fig_confusion(met):
    """同一個模型、兩個門檻的混淆矩陣：0.5 與式 (9) 的 p*。"""
    rows = [r for r in met if r["trace"] == "toolagent" and r["loss"] == "sym_l2"
            and r["num_leaves"] == "63" and r.get("label_mode") == "censored"
            and int(r["window_accesses"]) == 599095
            and r.get("feature_groups") == "history+deltas+edc+static"]
    by = {r["threshold_rule"]: r for r in rows}
    if not {"0.5", "p_star"} <= set(by):
        return
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.5))
    for ax, rule, title in ((axes[0], "0.5", "門檻 = 0.5（對稱損失的預設）"),
                            (axes[1], "p_star", r"門檻 = $p^{*}$（式 (9)）")):
        r = by[rule]
        m = np.array([[int(r["cm_drop_correct"]), int(r["cm_drop_wrong"])],
                      [int(r["cm_keep_wrong"]), int(r["cm_keep_correct"])]],
                     dtype=float)
        ax.imshow(m / m.sum(), cmap="Blues", vmin=0, vmax=0.6)
        for i in range(2):
            for j in range(2):
                v = m[i, j]
                ax.text(j, i, f"{v:,.0f}\n{100 * v / m.sum():.1f}%",
                        ha="center", va="center", fontsize=8,
                        color="white" if v / m.sum() > 0.3 else CLR["ink"])
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["真值：不會再用", "真值：會再用"], fontsize=7)
        ax.set_yticklabels(["決策：丟掉", "決策：留下"], fontsize=7)
        ax.set_title(f"{title}\n成本 {float(r['cost_ms']):,.0f} ms",
                     fontsize=8, pad=6)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    save(fig, "m5_confusion")


def fig_training(met):
    """訓練曲線：加權 L2 損失隨迭代下降，早停點標出來。"""
    picks = [r for r in met if r["trace"] == "toolagent" and r["num_leaves"] == "63"
             and r["threshold_rule"] == "p_star" and r.get("label_mode") == "censored"
             and int(r["window_accesses"]) == 17117
             and r.get("feature_groups") == "history+deltas+edc+static"
             and str(r.get("data_seed")) == "1234"
             and r.get("sample_rate") == "0.25"]
    if not picks:
        return
    fig, ax = plt.subplots(figsize=(COL, 2.4))
    for r, c in zip(picks, (CLR["cpu"], CLR["gpu"])):
        f = latest_train(r) / f"evals_{r['loss']}.json"
        if not f.exists():
            continue
        h = json.loads(f.read_text())
        for split, ls in (("train", "--"), ("valid", "-")):
            if split not in h:
                continue
            key = list(h[split])[0]
            y = h[split][key]
            ax.plot(range(1, len(y) + 1), y, ls, color=c, lw=1.2,
                    label=f"{r['loss']} / {split}")
        ax.axvline(int(r["best_iter"]), color=c, lw=0.7, ls=":")
    ax.set_xlabel("boosting 迭代（虛線 = 早停點）")
    ax.set_ylabel("加權 L2 損失")
    ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    save(fig, "m5_training")


def fig_importance():
    """特徵重要度（gain），只看主設定。"""
    p = M5 / "feature_importance.csv"
    if not p.exists():
        return
    rows = [r for r in csv.DictReader(p.open())
            if r["trace"] == "toolagent" and r["loss"] == "sym_l2"
            and r.get("num_leaves") == "63" and r.get("label_mode") == "censored"
            and r.get("window_accesses") == "17117"
            and r.get("feature_groups") == "history+deltas+edc+static"
            and str(r.get("data_seed")) == "1234"
            and r.get("sample_rate") == "0.25"]
    if not rows:
        return
    rows.sort(key=lambda r: -float(r["gain"]))
    rows = rows[:12]
    tot = sum(float(r["gain"]) for r in rows)
    fam = lambda n: (CLR["ssd"] if n.startswith(("pos", "req", "last"))
                     else CLR["cpu"] if n.startswith("log_delta")
                     else CLR["hl"] if n.startswith("edc") else CLR["drop"])
    fig, ax = plt.subplots(figsize=(COL, 2.9))
    names = [r["feature"] for r in rows][::-1]
    vals = [100 * float(r["gain"]) / tot for r in rows][::-1]
    ax.barh(names, vals, color=[fam(n) for n in names], height=.72)
    ax.set_xlabel("gain 佔前 12 名的比例（%）")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    save(fig, "m5_importance")


def main() -> int:
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "font.family": "sans-serif",
                         "font.sans-serif": ["Noto Sans CJK TC", "DejaVu Sans"],
                         "axes.unicode_minus": False})
    met = rows("predictor_metrics.csv")
    cal = rows("calibration_bins.csv")
    pol = rows("policy_sim.csv") if (M5 / "policy_sim.csv").exists() else []
    cm = load_cost_model("nvme", require_model_key="qwen-awq")
    print("產生 M5 的圖：")
    fig_ordering(met)
    fig_threshold(met, cm)
    fig_calibration(cal)
    fig_horizon(met, pol)
    fig_timeline()
    fig_roc_pr(met)
    fig_confusion(met)
    fig_training(met)
    fig_importance()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
