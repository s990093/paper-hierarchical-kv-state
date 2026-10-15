"""m10_analyze.py — 精度這條線的判定（03_plan_h200.md §2.4、§2.5、§1.1；10/15 晚改動補的細節）

**這支在看到任何低精度答對率之前寫好、commit**（事先登記的一部分）。判準照 03 寫死，這裡只是把它變成程式：

品質（每個模型、每個配置 X、每種情境）：
  情境  A        ：讀法 A（強制全載入），兩輪
        B-idle   ：讀法 B、b32、GPU 閒；第 2 輪留在 GPU（turn2_variant a）
        B-busy   ：讀法 B、b32、GPU 忙；第 2 輪留在 GPU
        B-i2b    ：讀法 B、b32、第 1 輪閒、第 2 輪換成忙再還原一次（第 1 輪取 B-idle 的列＋turn2_variant b 的列）
  主判準：主任務（T2–T5）所有列（兩輪合併），X 對 E0 的配對正確率差；以「題」為單位重抽的 bootstrap
          （10,000 次、seed 0），95% CI 下界 > −0.02。
  輔助  ：每個主任務族、每一輪，點估計的差 ≥ −0.05。
  ε 過  ＝主判準且輔助都過。
  檢定力：E1 對 E0（情境 A）的 CI 半寬 > 0.015 → 「題數不夠，先加題」，不能判「沒過」。
速度「不輸」（每個模型、X）：每個（頻寬, GPU 狀態）格子（32K、cake），X 的還原時間中位數 ≤ 最快的混合配置
  （F25、M25、B25、R25）的中位數 ×（1＋δ*）。
K1   ：E1 四種情境都 ε 過，而且速度不輸。
K1s  ：K1 不成立，但 E1s 或 E1sr 四種情境都 ε 過、而且速度不輸。
「這個模型過」＝K1 或 K1s。兩個模型怎麼合併照 03 §1.1 的表（decide_tier）。
K3   ：Step 2b 的 E1-K0 或 E1-INT8 四種情境都 ε 過、而且速度不輸（只在 FP8 階）。
Step 2（混合配置）：穩定差距第 1、2 條（第 3 條要新題目重跑）與 K2。

子命令：
  quality   --model m --csv <逐題 CSV…> [--configs …] → <out>/quality_<m>_<tag>.json 與表格 CSV
  speed     --model m --summary <grid summary.csv> --noise <noise.json> → speed_<m>.json（每格中位數、不輸與否）
  k3        --quality <quality json> --speed <2b 的 speed json…> → k3_<m>.json
  step2     --model m --csv … --speed … --summary … → 穩定差距第 1、2 條與 K2
  decide    --llama <quality json> --qwen <quality json> --speed-llama … --speed-qwen … → 03 §1.1 的下一步
  selftest  用假資料檢查 bootstrap 與判準（不讀任何實驗資料）
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict

import numpy as np

MAIN_TASKS = ["T2", "T3", "T4", "T5"]
POS_TASKS = ["T1", "T2", "T3", "T4"]          # 有位置帶的任務族（穩定差距第 2 條）
EPS_CI = -0.02
EPS_AUX = -0.05
POWER_HALFWIDTH = 0.015
N_BOOT = 10_000
BOOT_SEED = 0
MIXED = ["F25", "M25", "B25", "R25"]
CONDS = ["A", "B-idle", "B-busy", "B-i2b"]


# ============================================================================ 讀資料
def load_rows(paths):
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("run_id") == "run_id":           # 防重複表頭
                    continue
                if r["score"] == "BLIND":
                    raise SystemExit(f"{p} 有 BLIND 列：這是凍結前的試跑檔，不能拿來判定")
                r["turn"] = int(r["turn"])
                r["score"] = float(r["score"])
                rows.append(r)
    return rows


def cond_rows(rows, config, cond, bw="b32"):
    """回傳 {(item_id, turn): row}。"""
    out = {}
    for r in rows:
        if r["config"] != config:
            continue
        v = r["turn2_variant"]
        if cond == "A":
            ok = r["mode"] == "A" and v in ("", "a")
        elif cond == "B-idle":
            ok = r["mode"] == "B" and r["bw"] == bw and r["gpu_state"] == "idle" and v in ("", "a")
        elif cond == "B-busy":
            ok = r["mode"] == "B" and r["bw"] == bw and r["gpu_state"] == "busy" and v in ("", "a")
        elif cond == "B-i2b":
            ok = r["mode"] == "B" and r["bw"] == bw and r["gpu_state"] == "idle" and \
                ((r["turn"] == 1 and v == "") or (r["turn"] > 1 and v == "b"))
            # T5 單輪：沒有第 2 輪，B-i2b 只有第 1 輪（和 B-idle 相同）
        else:
            raise ValueError(cond)
        if ok:
            k = (r["item_id"], r["turn"])
            if k in out:
                raise SystemExit(f"重複的列：{config} {cond} {k}")
            out[k] = r
    return out


# ============================================================================ 統計
def cluster_boot(diff_by_item: dict, n_boot=N_BOOT, seed=BOOT_SEED):
    """diff_by_item：{item_id: [每一列的差]}。統計量＝所有列的平均（ratio estimator）。回傳 (點估計, lo, hi)。"""
    items = sorted(diff_by_item)
    if not items:
        return None, None, None
    s = np.array([sum(diff_by_item[i]) for i in items], dtype=np.float64)
    n = np.array([len(diff_by_item[i]) for i in items], dtype=np.float64)
    point = s.sum() / n.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(items), size=(n_boot, len(items)))
    stats = s[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def paired(ref: dict, x: dict, tasks=None, turn=None):
    """ref、x：{(item, turn): row}。回傳 {item: [差]}、配對列數、翻轉表、和 E0 輸出一致率。"""
    d = defaultdict(list)
    flips = {"e0_right_x_wrong": 0, "e0_wrong_x_right": 0}
    same, nsame, kl = 0, 0, []
    for k, rx in x.items():
        if tasks and rx["task"] not in tasks:
            continue
        if turn is not None and k[1] != turn:
            continue
        r0 = ref.get(k)
        if r0 is None:
            raise SystemExit(f"E0 少了 {k}")
        d[k[0]].append(rx["score"] - r0["score"])
        if r0["score"] >= 0.5 > rx["score"]:
            flips["e0_right_x_wrong"] += 1
        if rx["score"] >= 0.5 > r0["score"]:
            flips["e0_wrong_x_right"] += 1
        if rx.get("same_as_e0") not in ("", None):
            same += int(rx["same_as_e0"])
            nsame += 1
        if rx.get("kl_vs_e0") not in ("", None):
            kl.append(float(rx["kl_vs_e0"]))
    n = sum(len(v) for v in d.values())
    return d, n, flips, (same / nsame if nsame else None), (statistics.mean(kl) if kl else None)


def eps_check(ref, x, main_tasks=MAIN_TASKS):
    """主判準＋輔助（逐任務族、逐輪）。"""
    d, n, flips, same, kl = paired(ref, x, tasks=main_tasks)
    point, lo, hi = cluster_boot(d)
    aux = []
    for t in main_tasks:
        for turn in sorted({k[1] for k, r in x.items() if r["task"] == t}):
            dd, nn, *_ = paired(ref, x, tasks=[t], turn=turn)
            if nn == 0:
                continue
            p = sum(sum(v) for v in dd.values()) / nn
            aux.append({"task": t, "turn": turn, "n": nn, "diff": round(p, 4), "pass": p >= EPS_AUX})
    main_ok = lo is not None and lo > EPS_CI
    aux_ok = all(a["pass"] for a in aux)
    return {"n_rows": n, "n_items": len(d), "diff": _r(point), "ci95": [_r(lo), _r(hi)],
            "ci_halfwidth": _r((hi - lo) / 2 if lo is not None else None),
            "main_pass": bool(main_ok), "aux_pass": bool(aux_ok), "eps_pass": bool(main_ok and aux_ok),
            "aux": aux, "flips": flips, "same_as_e0": _r(same), "mean_kl_vs_e0": _r(kl, 6)}


def _r(x, k=4):
    return None if x is None else round(float(x), k)


def table(ref, x, group_keys=("task", "turn", "band")):
    """描述用：每個（任務, 輪, 帶）的 E0、X 正確率與配對差（含 bootstrap CI）。"""
    groups = defaultdict(dict)
    for k, rx in x.items():
        g = tuple(rx[c] if c != "turn" else k[1] for c in group_keys)
        groups[g][k] = rx
    out = []
    for g, xs in sorted(groups.items()):
        d, n, flips, same, kl = paired(ref, xs)
        p, lo, hi = cluster_boot(d, n_boot=2000)
        acc0 = statistics.mean(ref[k]["score"] for k in xs)
        accx = statistics.mean(r["score"] for r in xs.values())
        out.append({**dict(zip(group_keys, g)), "n": n, "acc_e0": round(acc0, 4), "acc_x": round(accx, 4),
                    "diff": _r(p), "ci_lo": _r(lo), "ci_hi": _r(hi), "flip_e0R_xW": flips["e0_right_x_wrong"],
                    "flip_e0W_xR": flips["e0_wrong_x_right"], "same_as_e0": _r(same), "mean_kl": _r(kl, 6),
                    "mean_lp_read": round(statistics.mean(int(r["lp_read"]) for r in xs.values()), 2)})
    return out


# ============================================================================ quality
def cmd_quality(a):
    rows = load_rows(a.csv)
    rows = [r for r in rows if r["model"] == a.model]
    if a.strip_fmt:                          # 例：INT4 階的 "E1@int4_kivi" → "E1"；只留 E0 和這個格式的列（同名的 FP8 列丟掉）
        rows = [r for r in rows if r["config"] == "E0" or r["config"].endswith("@" + a.strip_fmt)]
        for r in rows:
            name, _, f = r["config"].partition("@")
            if f == a.strip_fmt:
                r["config"] = name
    configs = a.configs or sorted({r["config"] for r in rows} - {"E0"})
    ref = cond_rows(rows, "E0", "A")
    res = {"model": a.model, "tag": a.tag, "csv": a.csv, "run_ids": sorted({r["run_id"] for r in rows}),
           "n_items": len({r["item_id"] for r in rows if not r["task"].startswith("LB-")}),
           "rule": {"main_tasks": MAIN_TASKS, "eps_ci_lower": EPS_CI, "eps_aux": EPS_AUX, "n_boot": N_BOOT,
                    "boot_seed": BOOT_SEED, "power_halfwidth": POWER_HALFWIDTH, "bootstrap_unit": "item"},
           "e0_accuracy": [], "configs": {}}
    # E0 各格答對率（只報告：60–95% 是難度要求，10/15 晚改動：不再依它排除任務族）
    cells = defaultdict(list)
    for (iid, turn), r in ref.items():
        cells[(r["task"], turn, r["band"])].append(r["score"])
    for (t, turn, band), v in sorted(cells.items()):
        m = statistics.mean(v)
        res["e0_accuracy"].append({"task": t, "turn": turn, "band": band, "n": len(v), "acc": round(m, 4),
                                   "in_60_95": 0.60 <= m <= 0.95})
    os.makedirs(a.out_dir, exist_ok=True)
    tab_rows = []
    for c in configs:
        res["configs"][c] = {}
        for cond in CONDS:
            x = cond_rows(rows, c, cond)
            x = {k: r for k, r in x.items() if not r["task"].startswith("LB-")}
            if not x:
                continue
            ent = eps_check(ref, x)
            ent["mean_lp_read"] = round(statistics.mean(int(r["lp_read"]) for r in x.values()), 3)
            ent["b_values"] = sorted({int(r["b"]) for r in x.values()})
            res["configs"][c][cond] = ent
            for t in table(ref, x):
                tab_rows.append({"model": a.model, "config": c, "cond": cond, **t})
            for t in table(ref, x, group_keys=("task",)):
                tab_rows.append({"model": a.model, "config": c, "cond": cond, "turn": "all", "band": "all", **t})
        # LongBench（只有讀法 A；不進判準）
        lb = {k: r for k, r in cond_rows(rows, c, "A").items() if r["task"].startswith("LB-")}
        if lb:
            for t in table(ref, lb, group_keys=("task",)):
                tab_rows.append({"model": a.model, "config": c, "cond": "A", "turn": 1, "band": "", **t})
    # 檢定力（E1 對 E0，情境 A）
    e1a = res["configs"].get("E1", {}).get("A")
    res["power_ok"] = None if e1a is None else bool(e1a["ci_halfwidth"] <= POWER_HALFWIDTH)
    tag = f"{a.model}_{a.tag}"
    json.dump(res, open(os.path.join(a.out_dir, f"quality_{tag}.json"), "w"), indent=1, ensure_ascii=False)
    keys = []
    for t in tab_rows:
        keys += [k for k in t if k not in keys]
    with open(os.path.join(a.out_dir, f"quality_{tag}_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(tab_rows)
    for c, v in res["configs"].items():
        for cond, e in v.items():
            print(f"{a.model} {c:12s} {cond:7s} n={e['n_rows']:5d} diff={e['diff']:+.4f} CI=[{e['ci95'][0]:+.4f},"
                  f"{e['ci95'][1]:+.4f}] eps={'PASS' if e['eps_pass'] else 'fail'} lp_read={e['mean_lp_read']}", flush=True)
    print("power_ok:", res["power_ok"])


# ============================================================================ speed
def cmd_speed(a):
    rows = list(csv.DictReader(open(a.summary)))
    dstar = json.load(open(a.noise))["delta_star"]
    cell = defaultdict(dict)
    for r in rows:
        if int(r["length"]) != a.main_length or r["mode"] != "cake":
            continue
        cell[(r["bw_name"], r["gpu_state"])][r["config"]] = float(r["median_t_restore"])
    if a.x_summary:                          # Step 2b：X（例 E1s@fp8k0_head）的中位數取自另一次 grid（跨 run，報告寫明）
        for r in csv.DictReader(open(a.x_summary)):
            if int(r["length"]) == a.main_length and r["mode"] == "cake" and r["config"] == a.x_from:
                cell[(r["bw_name"], r["gpu_state"])][a.configs[0]] = float(r["median_t_restore"])
    out = {"model": a.model, "summary": a.summary, "x_summary": a.x_summary, "delta_star": dstar,
           "rule": "X 不輸＝每格 median(X) ≤ min(F25,M25,B25,R25 的 median)×(1+δ*)（03 §2.4）", "cells": [], "not_worse": {}}
    for x in a.configs:
        ok_all = True
        for (bw, st), d in sorted(cell.items()):
            fastest_mixed = min(d[m] for m in MIXED)
            ok = d[x] <= fastest_mixed * (1 + dstar)
            ok_all &= ok
            out["cells"].append({"config": x, "bw": bw, "gpu_state": st, "median_x": d[x],
                                 "fastest_mixed": fastest_mixed, "ratio": round(d[x] / fastest_mixed, 4), "not_worse": ok})
        out["not_worse"][x] = bool(ok_all)
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(a.out_dir, f"speed_{a.model}{a.tag}.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out["not_worse"]))


# ============================================================================ decide
def model_verdict(q: dict, sp: dict, x_list):
    """回傳 {config: 四種情境都 ε 過 且 速度不輸}。少了情境 → None（不能判）。"""
    out = {}
    for x in x_list:
        v = q["configs"].get(x)
        if not v or any(c not in v for c in CONDS):
            out[x] = None
            continue
        out[x] = all(v[c]["eps_pass"] for c in CONDS) and bool(sp["not_worse"].get(x))
    return out


def cmd_decide(a):
    res = {"tier": a.tier}
    passed = {}
    for m, qp, sp in (("llama", a.llama, a.speed_llama), ("qwen", a.qwen, a.speed_qwen)):
        q, s = json.load(open(qp)), json.load(open(sp))
        v = model_verdict(q, s, ["E1", "E1s", "E1sr"])
        k1 = v["E1"]
        k1s = (not k1) and bool(v["E1s"] or v["E1sr"]) if k1 is not None else None
        res[m] = {"per_config": v, "K1": k1, "K1s": k1s, "power_ok": q["power_ok"],
                  "model_pass": (bool(k1) or bool(k1s)) if k1 is not None else None}
        passed[m] = res[m]["model_pass"]
    L, Q = passed["llama"], passed["qwen"]
    if res["llama"]["power_ok"] is False or res["qwen"]["power_ok"] is False:
        nxt = "題數不夠（E1 對 E0 的 CI 半寬 > 1.5 個百分點）：先加題再判，不能說「沒過」（03 §2.4）"
    elif L is None or Q is None:
        nxt = "資料不全，不能判"
    elif L and Q:
        nxt = "這一階兩個模型都不需要位置策略 → 往下一階（03 §2.5）" if a.tier == "fp8" else \
            "INT4 階也不需要位置策略 → 整個停（03 §2.5 第 3 步）"
    elif L and not Q:
        nxt = "只在 Qwen 上做 Step 2b（E1-K0、E1-INT8）；過 → K3 停；不過 → Step 2 在 Qwen 上做（03 §1.1）" if a.tier == "fp8" \
            else "INT4 階：Qwen 均勻不夠 → Step 2 在 Qwen 上做（03 §2.5 第 4 步、§1.1）"
    else:
        nxt = "在 Llama 上做 Step 2b；過 → K3 停；不過 → Step 2 在 Llama 上做（Qwen 不進混合配置）（03 §1.1）" if a.tier == "fp8" \
            else "INT4 階：Llama 均勻不夠 → Step 2 在 Llama 上做（03 §2.5 第 4 步、§1.1）"
    res["next"] = nxt
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(a.out_dir, f"decide_{a.tier}.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, ensure_ascii=False, indent=1))


# ============================================================================ K3（Step 2b）
def cmd_k3(a):
    """K3：E1-K0 或 E1-INT8 四種情境都 ε 過、而且速度不輸（03 §2.4）。--quality 給 quality json（含兩個 2b 配置），
    --speed 給每個 2b 配置各一個 speed json（cmd_speed --x-summary 產生）。"""
    q = json.load(open(a.quality))
    res = {"model": q["model"], "per_config": {}}
    for sp_path in a.speed:
        sp = json.load(open(sp_path))
        (x, nw), = sp["not_worse"].items()
        v = q["configs"].get(x)
        eps = None if not v or any(c not in v for c in CONDS) else all(v[c]["eps_pass"] for c in CONDS)
        res["per_config"][x] = {"eps_all_conds": eps, "speed_not_worse": nw,
                                "pass": None if eps is None else bool(eps and nw)}
    res["K3"] = any(bool(v["pass"]) for v in res["per_config"].values())
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(a.out_dir, f"k3_{q['model']}.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, ensure_ascii=False, indent=1))


# ============================================================================ Step 2（穩定差距、K2）
def cmd_step2(a):
    """穩定差距第 1、2 條（03 §2.4）。第 3 條（新題目重跑）要另外跑一批再用同一支判。"""
    rows = [r for r in load_rows(a.csv) if r["model"] == a.model]
    sp = json.load(open(a.speed))
    dstar = sp["delta_star"]
    med = {(c["config"], c["bw"], c["gpu_state"]): c["median_x"] for c in sp["cells"]}
    # speed json 只存了 --configs 的格子；混合配置的中位數另從 summary 讀
    summ = {(r["config"], r["bw_name"], r["gpu_state"]): float(r["median_t_restore"])
            for r in csv.DictReader(open(a.summary)) if int(r["length"]) == 32768 and r["mode"] == "cake"}
    out = {"model": a.model, "delta_star": dstar, "cond1": [], "cond2": {}}
    for st, cond in (("idle", "B-idle"), ("busy", "B-busy")):
        r25 = cond_rows(rows, "R25", cond)
        for x in ("F25", "M25", "B25"):
            xx = cond_rows(rows, x, cond)
            xx = {k: r for k, r in xx.items() if not r["task"].startswith("LB-")}
            d, n, *_ = paired(r25, xx, tasks=MAIN_TASKS)   # X − R25（正＝X 比較好）；和主判準同一組任務（T2–T5）
            p, lo, hi = cluster_boot(d)
            t_x, t_r = summ[(x, "b32", st)], summ[("R25", "b32", st)]
            ok = p is not None and p >= 0.02 and (lo > 0 or hi < 0) and t_x <= t_r * (1 + dstar)
            per_task = {}
            for t in POS_TASKS:
                dd, nn, *_ = paired(r25, xx, tasks=[t])
                if nn:
                    per_task[t] = round(sum(sum(v) for v in dd.values()) / nn, 4)
            out["cond1"].append({"config": x, "gpu_state": st, "n": n, "diff_vs_R25": _r(p), "ci95": [_r(lo), _r(hi)],
                                 "t_x": t_x, "t_R25": t_r, "cond1_pass": bool(ok), "per_pos_task": per_task})
    for e in out["cond1"]:
        if e["cond1_pass"]:
            same_dir = sum(1 for t, v in e["per_pos_task"].items() if (v > 0) == (e["diff_vs_R25"] > 0) and v != 0)
            out["cond2"][f"{e['config']}|{e['gpu_state']}"] = {"n_pos_tasks_same_direction": same_dir,
                                                               "cond2_pass": same_dir >= 2}
    out["stable_gap_1_2"] = any(v["cond2_pass"] for v in out["cond2"].values())
    out["K2"] = not out["stable_gap_1_2"]
    out["note"] = "穩定差距第 3 條（新 seed、同樣題數重跑，方向與顯著性都重現）要另外跑；K2＝第 1、2 條都沒有任何配置成立"
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(a.out_dir, f"step2_{a.model}_{a.tag}.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, ensure_ascii=False, indent=1))


# ============================================================================ selftest
def selftest(_a=None):
    """假資料：只檢查程式邏輯，不讀任何實驗資料。"""
    rng = np.random.default_rng(1)
    ok = True
    # 1. 差為 0 的配對 → 點估計 0、CI 含 0
    d = {f"i{k}": [0.0, 0.0] for k in range(100)}
    p, lo, hi = cluster_boot(d)
    ok &= p == 0 and lo == 0 and hi == 0
    # 2. 已知差 −0.1 → CI 在 −0.1 附近、下界 < −0.02 → 主判準不過
    d = {f"i{k}": [float(-(rng.random() < 0.1)), float(-(rng.random() < 0.1))] for k in range(800)}
    p, lo, hi = cluster_boot(d)
    ok &= -0.13 < p < -0.07 and lo < EPS_CI
    # 3. 列數不同的題（T5 單輪）用 ratio estimator
    d = {"a": [1.0, 1.0], "b": [0.0]}
    p, _, _ = cluster_boot(d, n_boot=10)
    ok &= abs(p - 2 / 3) < 1e-12
    # 4. eps_check 用假列
    ref, x = {}, {}
    for k in range(600):
        for turn in (1, 2):
            base = dict(item_id=f"T2_{k}", task=MAIN_TASKS[k % 4], band="front", turn=turn, lp_read="0",
                        same_as_e0="1", kl_vs_e0="0", b="0")
            if base["task"] == "T5" and turn == 2:
                continue
            s0 = float(rng.random() < 0.8)
            ref[(base["item_id"], turn)] = {**base, "score": s0}
            x[(base["item_id"], turn)] = {**base, "score": s0}
    e = eps_check(ref, x)
    ok &= e["eps_pass"] and e["diff"] == 0
    # 5. cond_rows：照 m10_quality.item_rows 的列形狀（兩輪題＋單輪 T5）
    rows = []
    for iid, task, turns in (("T2_x", "T2", (1, 2)), ("T5_y", "T5", (1,))):
        for cfg in ("E0", "E1"):
            for t in turns:
                rows.append(dict(config=cfg, item_id=iid, task=task, turn=t, mode="A", bw="", gpu_state="",
                                 turn2_variant="" if t == 1 else "a", score=1.0))
                for st, sc in (("idle", 0.5), ("busy", 0.25)):
                    rows.append(dict(config=cfg, item_id=iid, task=task, turn=t, mode="B", bw="b32", gpu_state=st,
                                     turn2_variant="" if t == 1 else "a", score=sc))
            if len(turns) > 1:
                rows.append(dict(config=cfg, item_id=iid, task=task, turn=2, mode="B", bw="b32", gpu_state="idle",
                                 turn2_variant="b", score=0.75))
    got = {c: cond_rows(rows, "E1", c) for c in CONDS}
    ok &= len(got["A"]) == 3 and len(got["B-idle"]) == 3 and len(got["B-busy"]) == 3 and len(got["B-i2b"]) == 3
    ok &= got["B-i2b"][("T2_x", 2)]["score"] == 0.75 and got["B-i2b"][("T2_x", 1)]["score"] == 0.5
    ok &= got["B-busy"][("T5_y", 1)]["score"] == 0.25 and got["A"][("T2_x", 2)]["score"] == 1.0
    print("selftest", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quality")
    q.add_argument("--model", required=True)
    q.add_argument("--csv", nargs="+", required=True)
    q.add_argument("--configs", nargs="*")
    q.add_argument("--tag", default="step1")
    q.add_argument("--strip-fmt", help="把 config 名稱裡的 @<這個格式> 拿掉（INT4 階用）")
    q.add_argument("--out-dir", required=True)
    s = sub.add_parser("speed")
    s.add_argument("--model", required=True)
    s.add_argument("--summary", required=True)
    s.add_argument("--noise", required=True)
    s.add_argument("--configs", nargs="+", default=["E1", "E1s", "E1sr"])
    s.add_argument("--main-length", type=int, default=32768)
    s.add_argument("--x-summary", help="X 的中位數改從這個 summary 取（Step 2b）；--configs 只能給一個名字")
    s.add_argument("--x-from", default="E1s", help="--x-summary 裡的 config 名稱")
    s.add_argument("--tag", default="")
    s.add_argument("--out-dir", required=True)
    d = sub.add_parser("decide")
    d.add_argument("--tier", choices=["fp8", "int4"], default="fp8")
    for m in ("llama", "qwen"):
        d.add_argument(f"--{m}", required=True)
        d.add_argument(f"--speed-{m}", required=True)
    d.add_argument("--out-dir", required=True)
    t = sub.add_parser("step2")
    t.add_argument("--model", required=True)
    t.add_argument("--csv", nargs="+", required=True)
    t.add_argument("--speed", required=True)
    t.add_argument("--summary", required=True)
    t.add_argument("--tag", default="seedA")
    t.add_argument("--out-dir", required=True)
    k = sub.add_parser("k3")
    k.add_argument("--quality", required=True)
    k.add_argument("--speed", nargs="+", required=True)
    k.add_argument("--out-dir", required=True)
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    {"quality": cmd_quality, "speed": cmd_speed, "decide": cmd_decide, "step2": cmd_step2, "k3": cmd_k3}[a.cmd](a)


if __name__ == "__main__":
    main()
