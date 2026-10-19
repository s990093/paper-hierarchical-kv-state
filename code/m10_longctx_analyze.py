"""m10_longctx_analyze.py — 10 號長 context 的彙整（docs/precision_20261013/10_longctx_plan.md §5；探索，不做判定）。CPU。

輸入：--raw 裡 44c 的 screen_<m>_<n_blocks>_s*.csv.gz（m10_sweep screen 的欄位，含 n_blocks、gen_score）；
      44a 的速度格子 <LC>/speed_<m>_<長度>_<格式>/summary.csv（＋gpu_guard）；32K 對照用主線的 summary_fp8／summary_int4。
輸出（--out-dir）：longctx_<m>.json、tables_longctx.md、figs/L1_<m>_acc.png（長度 × 答對率）、figs/L2_<m>_harm.png（傷害隨長度）。

§5 的五件事（寫死）：
  1. 主圖：每個長度，E0 與主線三種格式（FP8 主線粒度、INT8、INT4 KIVI）的 ALL 生成答對率，以題為單位 bootstrap 95% CI。
  2. 傷害隨長度：每個格式，ALL 對 E0 的差（生成答對率、teacher forcing 的 KL、第一個 token 對不對），
     同一個（任務, seed）跨長度配對，算「128K 的傷害 − 32K 的傷害」（另報 64K − 32K）與 CI；CI 不含 0 才寫「隨長度變大」，標「探索」。
  3. 位置：每個長度，主線三種格式的 F／M／B 對 R0（25%）的生成答對率差與 KL 差（配對 CI）。題數小，只報方向與 CI。
  4. 讀法 B：每個長度讀到多少低精度 block、KL 多少（和讀法 A 並列）。
  5. 速度：E1 對 E0 快多少（每格），和 32K 主線並排；gpu_guard 不是 CLEAN 的格子標出來（時間不能用）。
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.dirname(HERE)
R = os.path.join(REPO, "results", "m10_precision_mi300x")
LEN = {64: "32K", 128: "64K", 252: "128K"}
TOKENS = {64: 32768, 128: 65536, 252: 129024}
N_BOOT, SEED = 10_000, 0


def fmt_of(m, tier):
    return json.load(open(os.path.join(R, "fmt_select", "fmt_choice.json")))["choice"][m][tier]


def main3(m):
    return [fmt_of(m, "fp8"), "int8_token", "int4_kivi"]


def load(raw, m):
    rows = []
    for p in sorted(glob.glob(os.path.join(raw, f"screen_{m}_*_s*.csv.gz"))):
        for r in csv.DictReader(gzip.open(p, "rt")):
            if r.get("run_id") == "run_id":           # 10/19：續跑時多寫了一次表頭（s2 第一次 OOM 只留下表頭）→ 跳過
                continue
            for k in ("kl", "first_ok", "first_ok_e0", "lp_read", "frac"):
                r[k] = float(r[k]) if r[k] not in ("", None) else None
            r["gen"] = float(r["gen_score"]) if r.get("gen_score") not in ("", None) else None
            r["nb"] = int(r["n_blocks"])
            r["key"] = (r["task"], r["seed"])
            rows.append(r)
    return rows


def boot(vals_by_item: dict):
    """{item: 值} → (平均, lo, hi, n)；以題為單位 bootstrap。"""
    v = np.array([vals_by_item[k] for k in sorted(vals_by_item)], dtype=np.float64)
    if len(v) == 0:
        return None
    rng = np.random.default_rng(SEED)
    s = v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(1)
    lo, hi = np.percentile(s, [2.5, 97.5])
    return dict(mean=round(float(v.mean()), 4), lo=round(float(lo), 4), hi=round(float(hi), 4), n=int(len(v)))


def pick(rows, nb, fmt, policy, frac, readout="A"):
    out = {}
    for r in rows:
        if r["nb"] == nb and r["readout"] == readout and r["policy"] == policy and \
                (policy == "E0" or (r["fmt"] == fmt and abs(r["frac"] - frac) < 1e-9)):
            out[r["key"]] = r
    return out


def harm(rows, nb, fmt, metric):
    """每題 ALL 對 E0 的傷害：gen＝gen(ALL)−gen(E0)；kl＝kl(ALL)；first＝first_ok−first_ok_e0。"""
    a = pick(rows, nb, fmt, "ALL", 1.0)
    e = pick(rows, nb, None, "E0", 0.0)
    out = {}
    for k, r in a.items():
        if metric == "gen":
            if r["gen"] is None or k not in e or e[k]["gen"] is None:
                continue
            out[k] = r["gen"] - e[k]["gen"]
        elif metric == "kl":
            out[k] = r["kl"]
        else:
            out[k] = r["first_ok"] - r["first_ok_e0"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/mlsteam/data/tiara/longctx_m10")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "results", "m10_longctx_mi300x", "analysis"))
    ap.add_argument("--lc", default=os.path.join(REPO, "results", "m10_longctx_mi300x"))
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    md = ["# 長 context（10_longctx_plan §5；探索，不做判定）", ""]
    allres = {}
    for m in ("llama", "qwen"):
        rows = load(a.raw, m)
        if not rows:
            md += [f"## {m}", "", "NOT_MEASURED（沒有 44c 的 screen CSV）", ""]
            continue
        nbs = sorted({r["nb"] for r in rows})
        fmts = sorted({r["fmt"] for r in rows if r["policy"] == "ALL"})
        res = dict(model=m, run_ids=sorted({r["run_id"] for r in rows}), lengths=[LEN[n] for n in nbs],
                   n_items={LEN[n]: len({r["key"] for r in rows if r["nb"] == n}) for n in nbs})
        # 1. 主圖
        res["acc"] = {}
        for nb in nbs:
            e = pick(rows, nb, None, "E0", 0.0)
            ent = {"E0": boot({k: r["gen"] for k, r in e.items() if r["gen"] is not None})}
            for f in main3(m):
                x = pick(rows, nb, f, "ALL", 1.0)
                ent[f] = boot({k: r["gen"] for k, r in x.items() if r["gen"] is not None})
            res["acc"][LEN[nb]] = ent
        # 2. 傷害隨長度
        res["harm"] = {}
        for f in fmts:
            ent = {}
            for metric in ("gen", "kl", "first"):
                per = {nb: harm(rows, nb, f, metric) for nb in nbs}
                ent[metric] = {LEN[nb]: boot(per[nb]) for nb in nbs}
                for hi_nb in (128, 252):
                    if 64 in per and hi_nb in per:
                        common = set(per[64]) & set(per[hi_nb])
                        d = boot({k: per[hi_nb][k] - per[64][k] for k in common})
                        if d:
                            worse = (d["hi"] < 0) if metric in ("gen", "first") else (d["lo"] > 0)
                            d["grows_with_length"] = bool(worse)
                        ent[metric][f"{LEN[hi_nb]}−32K"] = d
            res["harm"][f] = ent
        # 3. 位置（25%）
        res["position"] = {}
        for nb in nbs:
            for f in main3(m):
                r0 = pick(rows, nb, f, "R0", 0.25)
                for pol in ("F", "M", "B"):
                    x = pick(rows, nb, f, pol, 0.25)
                    common = set(x) & set(r0)
                    g = boot({k: x[k]["gen"] - r0[k]["gen"] for k in common
                              if x[k]["gen"] is not None and r0[k]["gen"] is not None})
                    kl = boot({k: x[k]["kl"] - r0[k]["kl"] for k in common})
                    res["position"][f"{LEN[nb]}|{f}|{pol}"] = dict(gen_diff=g, kl_diff=kl)
        # 4. 讀法 B
        res["readout_b"] = {}
        for nb in nbs:
            for f in main3(m):
                for ro in ("A", "B_idle", "B_busy"):
                    x = pick(rows, nb, f, "ALL", 1.0, ro)
                    if x:
                        res["readout_b"][f"{LEN[nb]}|{f}|{ro}"] = dict(
                            lp_read=round(float(np.mean([r["lp_read"] for r in x.values()])), 2),
                            kl=boot({k: r["kl"] for k, r in x.items()}))
        # 5. 速度
        res["speed"] = speed(a.lc, m)
        allres[m] = res
        json.dump(res, open(os.path.join(a.out_dir, f"longctx_{m}.json"), "w"), indent=1, ensure_ascii=False)
        md += report(res, m)
        if not a.no_fig:
            figs(res, m, os.path.join(a.out_dir, "figs"))
    open(os.path.join(a.out_dir, "tables_longctx.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


def speed(lc, m):
    out = {}
    for tier in ("fp8", "int4"):
        f = fmt_of(m, tier)
        base = os.path.join(R, "speed_qwen_regrid" if (m == "qwen" and tier == "fp8") else f"speed_{m}", f"summary_{tier}.csv")
        srcs = [("32K", base, None)] + [(LEN[nb], os.path.join(lc, f"speed_{m}_{TOKENS[nb]}_{f}", "summary.csv"),
                                         os.path.join(lc, f"speed_{m}_{TOKENS[nb]}_{f}")) for nb in (128, 252)]
        for L, p, d in srcs:
            if not os.path.exists(p):
                out[f"{L}|{tier}"] = "NOT_MEASURED"
                continue
            med = defaultdict(dict)
            for r in csv.DictReader(open(p)):
                if r["mode"] == "cake" and r["config"] in ("E0", "E1"):
                    med[(r["bw_name"], r["gpu_state"])][r["config"]] = float(r["median_t_restore"])
            dirty = []
            if d:
                dirty = [os.path.basename(g) for g in glob.glob(os.path.join(d, "gpu_guard_*.json"))
                         if json.load(open(g)).get("verdict") != "CLEAN"]
            out[f"{L}|{tier}"] = dict(cells={f"{k[0]}|{k[1]}": dict(E0=v.get("E0"), E1=v.get("E1"),
                                                                    gain=round((v["E0"] - v["E1"]) / v["E0"], 4)
                                                                    if "E0" in v and "E1" in v else None)
                                             for k, v in sorted(med.items())}, dirty_guards=dirty)
    return out


def _ci(x, pct=True):
    if not x:
        return "—"
    s = 100 if pct else 1
    return f"{s * x['mean']:+.1f} [{s * x['lo']:+.1f}, {s * x['hi']:+.1f}]（n＝{x['n']}）" if pct else \
        f"{x['mean']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}]（n＝{x['n']}）"


def report(res, m):
    L = [f"## {m}（run：{'、'.join(res['run_ids'])}；每個長度的題數 {res['n_items']}）", "",
         "### 1. 生成答對率（%，95% CI）", "", "| 長度 | E0 | " + " | ".join(f"ALL {f}" for f in main3(m)) + " |",
         "|:--|" + "--:|" * (1 + len(main3(m)))]
    for Lb, e in res["acc"].items():
        cell = lambda x: f"{100 * x['mean']:.1f} [{100 * x['lo']:.1f}, {100 * x['hi']:.1f}]" if x else "—"  # noqa: E731
        L.append(f"| {Lb} | {cell(e['E0'])} | " + " | ".join(cell(e[f]) for f in main3(m)) + " |")
    L += ["", "### 2. 傷害隨長度（ALL 對 E0；生成答對率差，百分點；「128K−32K」CI 不含 0 且變差才標 ↑）", "",
          "| 格式 | 32K | 64K | 128K | 64K−32K | 128K−32K |", "|:--|:--|:--|:--|:--|:--|"]
    for f, e in res["harm"].items():
        g = e["gen"]
        flag = lambda k: (" ↑" if g.get(k) and g[k].get("grows_with_length") else "")  # noqa: E731
        L.append(f"| {f} | {_ci(g.get('32K'))} | {_ci(g.get('64K'))} | {_ci(g.get('128K'))} | "
                 f"{_ci(g.get('64K−32K'))}{flag('64K−32K')} | {_ci(g.get('128K−32K'))}{flag('128K−32K')} |")
    L += ["", "（KL、第一個 token 的同一張表在 JSON 的 harm.<格式>.kl／first）", "",
          "### 3. 位置（25%：F／M／B 對 R0；生成答對率差，百分點；KL 差負＝比隨機好）", "",
          "| 長度 | 格式 | 放法 | 生成差 | KL 差 |", "|:--|:--|:--|:--|:--|"]
    for k, v in res["position"].items():
        Lb, f, p = k.split("|")
        L.append(f"| {Lb} | {f} | {p} | {_ci(v['gen_diff'])} | {_ci(v['kl_diff'], pct=False)} |")
    L += ["", "### 4. 讀法 B（ALL）", "", "| 長度 | 格式 | 讀法 | 讀到的低精度 block | KL |", "|:--|:--|:--|--:|:--|"]
    for k, v in res["readout_b"].items():
        Lb, f, ro = k.split("|")
        L.append(f"| {Lb} | {f} | {ro} | {v['lp_read']} | {_ci(v['kl'], pct=False)} |")
    L += ["", "### 5. 速度（E1 對 E0 快多少；cake 中位數）", "", "| 長度 | 階 | 格子 | E0 (s) | E1 (s) | E1 快多少 |", "|:--|:--|:--|--:|--:|--:|"]
    for k, v in res["speed"].items():
        Lb, t = k.split("|")
        if v == "NOT_MEASURED":
            L.append(f"| {Lb} | {t} | — | NOT_MEASURED | | |")
            continue
        for c, x in v["cells"].items():
            gain = f"{100 * x['gain']:+.1f}%" if x["gain"] is not None else "—"
            L.append(f"| {Lb} | {t} | {c} | {x['E0']} | {x['E1']} | {gain} |")
        if v["dirty_guards"]:
            L.append(f"| {Lb} | {t} | **gpu_guard 不是 CLEAN：{v['dirty_guards']}（時間不能用）** | | | |")
    L.append("")
    return L


def figs(res, m, out_dir):
    import m10_figs as F
    plt = F.plt
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    Ls = list(res["acc"])
    xs = range(len(Ls))
    for name, col in [("E0", F.INK2)] + list(zip(main3(m), F.SLOTS)):
        pts = [res["acc"][L].get(name) for L in Ls]
        ys = [100 * p["mean"] if p else np.nan for p in pts]
        err = [[100 * (p["mean"] - p["lo"]) if p else 0 for p in pts], [100 * (p["hi"] - p["mean"]) if p else 0 for p in pts]]
        ax.errorbar(list(xs), ys, yerr=err, color=col, marker="o", lw=2, capsize=3, label="E0（BF16）" if name == "E0" else f"ALL {name}")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(Ls)
    ax.set_ylabel("生成答對率（%）")
    ax.set_xlabel("context 長度")
    ax.legend(fontsize=7)
    ax.set_title(f"圖 L1　{m}：長度 × 答對率（T2–T5、讀法 A；95% CI；探索）")
    F.source_note(fig, "10_longctx_plan §5；" + "、".join(res["run_ids"][:3]) + ("…" if len(res["run_ids"]) > 3 else ""))
    F.save(fig, os.path.join(out_dir, f"L1_{m}_acc.png"))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fm = list(res["harm"])
    for j, (L, col) in enumerate(zip(["32K", "64K", "128K"], [F.SLOTS[0], F.SLOTS[1], F.SLOTS[2]])):
        ys, lo, hi = [], [], []
        for f in fm:
            x = res["harm"][f]["gen"].get(L)
            ys.append(100 * x["mean"] if x else np.nan)
            lo.append(100 * (x["mean"] - x["lo"]) if x else 0)
            hi.append(100 * (x["hi"] - x["mean"]) if x else 0)
        ax.errorbar([i + (j - 1) * 0.22 for i in range(len(fm))], ys, yerr=[lo, hi], fmt="o", color=col, capsize=2, label=L)
    ax.axhline(0, color=F.BASE, lw=1)
    ax.axhline(-2, color=F.SLOTS[7], lw=1, ls=":")
    ax.set_xticks(range(len(fm)))
    ax.set_xticklabels(fm, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("ALL 對 E0 的生成答對率差（百分點）")
    ax.legend(fontsize=7)
    ax.set_title(f"圖 L2　{m}：每個格式的傷害，隨長度（探索；虛線＝−2）")
    F.source_note(fig, "10_longctx_plan §5")
    F.save(fig, os.path.join(out_dir, f"L2_{m}_harm.png"))


if __name__ == "__main__":
    main()
