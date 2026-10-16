"""m10_confirm_analyze.py — 大掃描複賽（09 §4.3）的判定；CPU。規則 10/15 寫死，下面「讀法」那一條 10/16 釐清（沒有任何結果時）。

三條（每個模型、每個選手）：
  1. 品質不輸 E0：主線凍結的 `m10_analyze.py quality`（ε＝2 個百分點，四種情境 A／B-idle／B-busy／B-i2b 都過）。
  2. 贏過隨機：配對答對率差＝X − 同格 R0–R4 五個 seed 的平均（同一題、同一輪），以題為單位 bootstrap（m10_analyze 的
     cluster bootstrap，10,000 次、seed 0）。單邊 p＝bootstrap 統計量 ≤ 0 的比例；Holm 校正，族＝這個模型的所有選手，α＝0.025
     （沒有校正時等於「95% CI 下界 > 0」）。**主判準用讀法 A**（和海選挑選手的依據一致）；讀法 B 的閒、忙用同樣的檢定，列為次要。
     前緣點不是選手，只報它對 E0 的差和對 R 的差，不進 Holm。
  3. 速度不輸：同位元組（同格式、同佔比）的配置（X、R0–R4、同格的其他選手）裡，每個（頻寬, GPU 狀態）格子
     X 的還原時間中位數 ≤ 最快的 ×（1＋δ*）；用複賽實測的速度格子（confirm/speed_<m>/summary_<格式>.csv）。
     ALL（E1s）沒有同位元組的混合配置 → 寫「—」。速度格子的 gpu_guard 不是 CLEAN → 那個格子的速度不能用（CLAUDE.md §3）。
輸出：<dir>/confirm_<m>.json、<dir>/tables_confirm.md；m10_analyze 的輸出在 <dir>/analysis/。
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.dirname(HERE)
import m10_analyze as A  # noqa: E402

ALPHA = 0.025
RAND = ["R0", "R1", "R2", "R3", "R4"]


def boot_stats(diff_by_item, n_boot=A.N_BOOT, seed=A.BOOT_SEED):
    """和 m10_analyze.cluster_boot 同一個做法，另外回傳 bootstrap 統計量（算單邊 p 用）。"""
    items = sorted(diff_by_item)
    s = np.array([sum(diff_by_item[i]) for i in items], dtype=np.float64)
    n = np.array([len(diff_by_item[i]) for i in items], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(items), size=(n_boot, len(items)))
    stats = s[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(s.sum() / n.sum()), float(lo), float(hi), float((stats <= 0).mean())


def vs_random(rows, spec, cond):
    """X − 同格 R0–R4 平均；回傳 (點估計, lo, hi, 單邊 p, 配對列數)。只用主任務（m10_analyze.MAIN_TASKS）。"""
    name, _, fmt = spec.partition("@")
    _, pol, pct = name.split("-")
    x = A.cond_rows(rows, spec, cond)
    rs = [A.cond_rows(rows, f"SW-{r}-{pct}@{fmt}", cond) for r in RAND]
    d = defaultdict(list)
    for k, rx in x.items():
        if rx["task"] not in A.MAIN_TASKS:
            continue
        rr = [r.get(k) for r in rs]
        if any(v is None for v in rr):
            raise SystemExit(f"{spec} {cond}：R 少了 {k}")
        d[k[0]].append(rx["score"] - sum(v["score"] for v in rr) / len(rr))
    if not d:
        return None
    p, lo, hi, pv = boot_stats(d)
    return dict(diff=round(p, 4), ci95=[round(lo, 4), round(hi, 4)], p_one_sided=round(pv, 5),
                n_rows=sum(len(v) for v in d.values()))


def holm(pvals: dict, alpha=ALPHA) -> dict:
    order = sorted(pvals, key=pvals.get)
    out, m, still = {}, len(order), True
    for i, k in enumerate(order):
        thr = alpha / (m - i)
        still = still and pvals[k] <= thr
        out[k] = dict(p=pvals[k], threshold=round(thr, 5), reject=bool(still))
    return out


def speed_check(sdir, spec, same_bytes, dstar):
    """每個（頻寬, GPU 狀態）：X 的中位數 ≤ 同位元組裡最快的 ×（1＋δ*）。"""
    name, _, fmt = spec.partition("@")
    p = os.path.join(sdir, f"summary_{fmt}.csv")
    if not os.path.exists(p):
        return None
    med = defaultdict(dict)
    for r in csv.DictReader(open(p)):
        if r["mode"] == "cake" and r["length"] == "32768":
            med[(r["bw_name"], r["gpu_state"])][r["config"]] = float(r["median_t_restore"])
    guards = [json.load(open(g)) for g in glob.glob(os.path.join(sdir, "gpu_guard_*.json"))]
    dirty = [g for g in guards if g.get("verdict") != "CLEAN"]
    cells, ok = [], True
    for cell, m in sorted(med.items()):
        if name not in m:
            continue
        pool = {c: t for c, t in m.items() if c in same_bytes}
        best = min(pool.values())
        good = m[name] <= best * (1 + dstar)
        ok &= good
        cells.append(dict(bw=cell[0], gpu_state=cell[1], t_x=m[name], fastest=min(pool, key=pool.get), t_fastest=best,
                          not_worse=bool(good)))
    return dict(not_worse=bool(ok) and not dirty, cells=cells, n_dirty_guards=len(dirty))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(REPO, "results", "m10_sweep_mi300x", "confirm"))
    ap.add_argument("--plan", default=os.path.join(REPO, "results", "m10_sweep_mi300x", "confirm_plan.json"))
    a = ap.parse_args()
    plan = json.load(open(a.plan))["models"]
    md = ["# 大掃描複賽（09 §4.3；新 seed 50000 起）", ""]
    for m, pm in plan.items():
        csvs = sorted(glob.glob(os.path.join(a.dir, f"quality_{m}_*.csv")))
        if not csvs:
            md += [f"## {m}", "", "NOT_MEASURED（沒有複賽的品質 CSV）", ""]
            continue
        fin = json.load(open(pm["finalists_json"]))
        finalists = [f"SW-{f['policy']}-{int(round(f['frac'] * 100))}@{f['fmt']}" for f in fin["finalists"]]
        frontier = [(f"E1s@{f['fmt']}" if f["policy"] == "ALL" else f"SW-{f['policy']}-{int(round(f['frac'] * 100))}@{f['fmt']}")
                    for f in fin["frontier"]]
        out_a = os.path.join(a.dir, "analysis")
        subprocess.run([sys.executable, os.path.join(HERE, "m10_analyze.py"), "quality", "--model", m, "--csv", *csvs,
                        "--configs", *pm["configs"], "--tag", "confirm", "--out-dir", out_a], check=True,
                       stdout=subprocess.DEVNULL)
        q = json.load(open(os.path.join(out_a, f"quality_{m}_confirm.json")))
        rows = A.load_rows(csvs)
        rows = [r for r in rows if r["model"] == m]
        dstar = json.load(open(os.environ.get("CONFIRM_NOISE") or os.path.join(REPO, "results", "m10_precision_mi300x", f"speed_{m}", "noise.json")))["delta_star"]
        res = dict(model=m, delta_star=dstar, rule_reading="主判準＝讀法 A；B-idle、B-busy 次要（10/16 釐清）",
                   finalists={}, frontier={}, run_ids=q["run_ids"], n_items=q["n_items"])
        tests = {}
        for cond in ("A", "B-idle", "B-busy"):
            tests[cond] = {s: vs_random(rows, s, cond) for s in finalists}
        hol = {cond: holm({s: t["p_one_sided"] for s, t in v.items() if t}) for cond, v in tests.items()}
        for s in finalists + frontier:
            name, _, fmt = s.partition("@")
            pct = name.split("-")[2] if name.startswith("SW-") else None
            same = ({name} | {f"SW-{r}-{pct}" for r in RAND} |
                    {f.partition("@")[0] for f in finalists if f.endswith("@" + fmt) and f.split("-")[2].split("@")[0] == pct}) \
                if pct else set()
            eps = {c: (q["configs"].get(s, {}).get(c) or {}).get("eps_pass") for c in A.CONDS}
            ent = dict(eps=eps, eps_all=all(v is True for v in eps.values()),
                       speed=speed_check(os.path.join(a.dir, f"speed_{m}"), s, same, dstar) if pct else "—（ALL：沒有同位元組的混合配置）")
            if s in finalists:
                ent["vs_random"] = {c: dict(**tests[c][s], holm=hol[c].get(s)) if tests[c][s] else None for c in tests}
                ent["beats_random"] = bool(hol["A"].get(s, {}).get("reject"))
                ent["confirmed"] = bool(ent["eps_all"] and ent["beats_random"] and isinstance(ent["speed"], dict)
                                        and ent["speed"]["not_worse"])
                res["finalists"][s] = ent
            else:
                ent["vs_random_A"] = vs_random(rows, s, "A") if pct and not name.split("-")[1].startswith("R") else None
                res["frontier"][s] = ent
        json.dump(res, open(os.path.join(a.dir, f"confirm_{m}.json"), "w"), indent=1, ensure_ascii=False)
        md += [f"## {m}（δ*＝{dstar:.4f}；{res['n_items']} 題；run：{'、'.join(res['run_ids'])}）", "",
               "| 配置 | 角色 | ε 四種情境 | 贏隨機（讀法 A，Holm） | 對隨機的差 A（百分點，95% CI） | 速度不輸 | 複賽成立 |",
               "|:--|:--|:--|:--|:--|:--|:--|"]
        for s, e in res["finalists"].items():
            v = e["vs_random"]["A"]
            sp = e["speed"]["not_worse"] if isinstance(e["speed"], dict) else e["speed"]
            md.append(f"| {s} | 選手 | {'過' if e['eps_all'] else '沒過'} | {'是' if e['beats_random'] else '否'} | "
                      f"{100 * v['diff']:+.1f} [{100 * v['ci95'][0]:+.1f}, {100 * v['ci95'][1]:+.1f}] | {sp} | "
                      f"**{'是' if e['confirmed'] else '否'}** |")
        for s, e in res["frontier"].items():
            v = e.get("vs_random_A")
            sp = e["speed"]["not_worse"] if isinstance(e["speed"], dict) else e["speed"]
            dv = f"{100 * v['diff']:+.1f} [{100 * v['ci95'][0]:+.1f}, {100 * v['ci95'][1]:+.1f}]" if v else "—"
            md.append(f"| {s} | 前緣點 | {'過' if e['eps_all'] else '沒過'} | — | {dv} | {sp} | — |")
        if not res["finalists"]:
            md.append("\n海選沒有任何格子勝出（09 §4.2 第 5 條）：複賽只有前緣點。")
        md.append("")
    open(os.path.join(a.dir, "tables_confirm.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
