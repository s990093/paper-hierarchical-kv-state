"""p01_tools.py — ICPE P0-1（docs/h200/icpe/P0-1_recompute_speed.md）的小工具與分析。

  bw-nfs --out-dir D        §3.2：H200 的 speed_<m>/bw.json 複本，只加 targets.nfs（tier_params 的 nfs 讀頻寬）→ D/bw_<m>.json
  analyze-a                 §3.1：B 主設定 S5 對 S4+（m7_analyze 同一套，經 m7_repro_compare.analyze）→ p01/p01a_verdict.json、p01a_table.csv
  analyze-b                 §3.2＋§8 偏離 1：每格最佳方式（S4 下 Cake vs 全部載入）、翻轉；G5b 只報告 → p01/p01b_verdict.json、p01b_cells.csv
  analyze-c                 §3.3：X1／X5 三個版本的判定字、跨門檻的格子 → p01/p01c_verdict.json、p01c_crossings.csv
  verdict                   §4：總判定 → p01/p01_verdict.json

每個輸出都有 run_id、ts。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TIER_PARAMS = os.environ.get("M7_TIER_PARAMS") or os.path.join(REPO, "results", "m7_write_policy_mi300x", "tier_params.json")
PREC_H200 = os.path.join(REPO, "results", "m10_precision_h200")


def run_id():
    return os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc"


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def cmd_bw_nfs(a):
    tp = json.load(open(TIER_PARAMS))
    os.makedirs(a.out_dir, exist_ok=True)
    for m in ("llama", "qwen"):
        src = os.path.join(PREC_H200, f"speed_{m}", "bw.json")
        d = json.load(open(src))
        assert "nfs" not in d["targets"], src
        d["targets"]["nfs"] = {"b_star": None, "GiBps": float(tp["nfs"]["read_GiBps"]),
                               "source": f"{os.path.relpath(TIER_PARAMS, REPO)} 的 nfs.read_GiBps（03 §R2：限速層兩台都用 MI300X 的）"}
        d["_icpe_p01"] = {"copied_from": os.path.relpath(src, REPO), "run_id": run_id(), "ts": ts(),
                          "note": "只加 targets.nfs；f 與 b16／b32／b48 不動（P0-1 §3.2：限速層頻寬沿用舊值）"}
        out = os.path.join(a.out_dir, f"bw_{m}.json")
        json.dump(d, open(out, "w"), indent=1, ensure_ascii=False)
        print(m, {k: v["GiBps"] for k, v in d["targets"].items()}, "→", out)


P01 = os.path.join(REPO, "results", "icpe_h200", "p01")
OLD7 = os.path.join(REPO, "results", "m7_write_policy_h200")
MAIN_B = dict(io_model="share", cpu_tier="harness", ssd_dev="nfs", gap_s=0.0, ssd_frac=1.0, wl_seed=0)
TIE = 0.05


def _rdir():
    rid = os.environ.get("RUN_DIR")
    return rid if rid and os.path.isdir(rid) else None


def _winners():
    c = json.load(open(os.path.join(P01, "attn", "choice.json")))
    return {m: (None if v["stop_loss"] else v["winner"]) for m, v in c["models"].items()}, c


def cmd_analyze_a(a):
    import shutil
    import tempfile
    import numpy as np
    import pandas as pd
    sys.path.insert(0, os.path.join(REPO, "code"))
    import m7_repro_compare as C
    base = _rdir() or tempfile.mkdtemp(prefix="icpe_p01a_")
    stage = os.path.join(base, "stage")
    os.makedirs(stage, exist_ok=True)
    for f in os.listdir(OLD7):            # 舊目錄的校準（照 h200_brr.ball 的篩法：不放 B、A2、C7 錨點與衍生檔）
        pth = os.path.join(OLD7, f)
        if not os.path.isfile(pth) or not f.endswith((".csv", ".json")):
            continue
        if f.startswith(("b_", "b2_", "summary_b", "sim_", "repro_", "verdict", "a2_", "c7_anchor", "summary_a2", "summary_c7", "a2c7_")):
            continue
        shutil.copy2(pth, stage)
    for f in ("calib_c1.csv", "c4_noise.csv"):          # 勝者重量的 C1、C4 蓋掉舊的
        shutil.copy2(os.path.join(P01, "calib", f), os.path.join(stage, f))
    for f in os.listdir(os.path.join(P01, "b")):
        if (f.startswith("b_") and f.endswith(".csv")) or (f.startswith("b_workload_") and f.endswith(".json")):
            shutil.copy2(os.path.join(P01, "b", f), stage)
    tmp, V = C.analyze(stage, TIER_PARAMS, os.path.join(base, "figs"), "icpe_p01a")
    if V.get("errors"):
        print("m7_analyze errors:", V["errors"])
    delta = float(V["calib"]["C4"]["delta_pct"])
    dstar = max(5.0, delta)
    d = pd.read_csv(os.path.join(tmp, "summary_b_verdict.csv"))
    for k, v in MAIN_B.items():
        d = d[d[k].astype(str) == str(v)] if isinstance(v, str) else d[np.isclose(d[k].astype(float), float(v))]
    raw = pd.read_csv(os.path.join(P01, "b", "b_share.csv"))
    bad = int((raw.contaminated_gpu.astype(str) == "True").sum())
    kvb = int(pd.to_numeric(raw.kv_bad, errors="coerce").fillna(0).sum())
    ret = raw[raw["round"] > 1]
    old = pd.read_csv(os.path.join(REPO, "results", "m7_write_policy_h200_merged", "summary_b_verdict_all.csv"))
    for k, v in MAIN_B.items():
        old = old[old[k].astype(str) == str(v)] if isinstance(v, str) else old[np.isclose(old[k].astype(float), float(v))]
    rows = []
    for _, r in d.sort_values("cpu_frac").iterrows():
        cf = float(r.cpu_frac)
        q = ret[np.isclose(ret.cpu_frac, cf)]
        m4, m5 = q[q.strategy == "S4+"].ttft.mean(), q[q.strategy == "S5"].ttft.mean()
        o = old[np.isclose(old.cpu_frac.astype(float), cf)]
        rows.append(dict(run_id=run_id(), ts=ts(), cpu_frac=cf, n_reps=int(r.n_reps), s4plus_median=float(r.s4plus_median),
                         s5_median=float(r.s5_median), s5_vs_s4plus_pct=float(r.s5_vs_s4plus_pct), delta_star_pct=dstar,
                         passed=bool(float(r.s5_vs_s4plus_pct) <= dstar),
                         s5_vs_s4plus_mean_pct=float((m4 - m5) / m4 * 100), best_simple=r.best_simple, s5_vs_best_simple_pct=float(r.gain_pct),
                         old_merged_s5_vs_s4plus_pct=(float(o.s5_vs_s4plus_pct.iloc[0]) if len(o) else None),
                         src_runs=";".join(sorted(raw.run_id.unique()))))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(P01, "p01a_table.csv"), index=False)
    out = dict(run_id=run_id(), ts=ts(), c4_delta_pct=delta, delta_star_pct=dstar, n_cpu_frac=len(t),
               contaminated_rows=bad, kv_bad=kvb, passed=bool(len(t) == 3 and t.passed.all() and bad == 0),
               rule="P0-1 §3.1：每個 cpu_frac，S5 對 S4+（回來請求 TTFT 中位數，rep 0–5 合併；正＝S5 快）≤ +δ*", rows=rows,
               m7_analyze_errors=V.get("errors"))
    json.dump(out, open(os.path.join(P01, "p01a_verdict.json"), "w"), ensure_ascii=False, indent=1, default=str)
    print(t.drop(columns=["run_id", "ts", "src_runs"]).round(3).to_string(index=False))
    print("δ*", dstar, "passed", out["passed"])


def _grid_rows(m, kind, arm):
    import pandas as pd
    pth = os.path.join(P01, "grid", f"{m}_{kind}_{arm}", "restores.csv")
    d = pd.read_csv(pth)
    d["arm"], d["kind"] = arm, kind
    return d


def cmd_analyze_b(a):
    import numpy as np
    import pandas as pd
    win, _ = _winners()
    cells, g5 = [], []
    problems = []
    for m in ("llama", "qwen"):
        if not win.get(m):
            problems.append(f"{m}：S1 停損，(b) 沒跑")
            continue
        arms = ["sdpa", win[m]]
        d = pd.concat([_grid_rows(m, k, arm) for k in ("s4", "g5b") for arm in arms], ignore_index=True)
        nb = int((d.contaminated_gpu.astype(str) == "True").sum())
        kb = int(pd.to_numeric(d.kv_bad, errors="coerce").fillna(0).sum())
        if nb or kb:
            problems.append(f"{m}：contaminated_gpu {nb} 列、kv_bad {kb}")
        s4 = d[(d.kind == "s4") & d["mode"].isin(["cake", "loadall"])]
        med = s4.groupby(["arm", "bw_name", "gpu_state", "config", "mode"]).t_restore.agg(["median", "size"]).reset_index()
        best = {}
        for (arm, bw, st, cfg), q in med.groupby(["arm", "bw_name", "gpu_state", "config"]):
            q = q.set_index("mode")
            tc, tl = float(q.loc["cake", "median"]), float(q.loc["loadall", "median"])
            rel = (tl - tc) / min(tc, tl)
            w = "tie" if abs(rel) <= TIE else ("cake" if tc < tl else "loadall")
            best[(arm, bw, st, cfg)] = dict(cake=tc, loadall=tl, n_cake=int(q.loc["cake", "size"]), n_loadall=int(q.loc["loadall", "size"]),
                                            winner=w, loadall_vs_cake=rel)
        for bw in ("cpu", "b32", "nfs"):
            for st in ("idle", "busy"):
                for cfg in ("E0", "E1"):
                    o, n = best.get(("sdpa", bw, st, cfg)), best.get((win[m], bw, st, cfg))
                    if o is None or n is None:
                        problems.append(f"{m} {bw} {st} {cfg}：缺格")
                        continue
                    flip = o["winner"] != "tie" and n["winner"] != "tie" and o["winner"] != n["winner"]
                    weak = (o["winner"] == "tie") != (n["winner"] == "tie")
                    cells.append(dict(run_id=run_id(), ts=ts(), model=m, tier=bw, gpu=st, config=cfg, attn_new=win[m],
                                      sdpa_cake=o["cake"], sdpa_loadall=o["loadall"], sdpa_winner=o["winner"],
                                      new_cake=n["cake"], new_loadall=n["loadall"], new_winner=n["winner"],
                                      n_per_mode=min(o["n_cake"], o["n_loadall"], n["n_cake"], n["n_loadall"]),
                                      flip=bool(flip), weak_change=bool(weak)))
        # G5b（只報告，S0）
        q = d[d.kind == "g5b"]
        for (arm, bw, st, cfg), x in q.groupby(["arm", "bw_name", "gpu_state", "config"]):
            gm = x[x["mode"] == "cake"]
            c, l = x[x["mode"] == "claim"].t_restore.median(), x[x["mode"] == "loadall"].t_restore.median()
            bst = min(c, l)
            g5.append(dict(run_id=run_id(), ts=ts(), model=m, arm=arm, tier=bw, gpu=st, config=cfg, g5b=float(gm.t_restore.median()),
                           cake=float(c), loadall=float(l), g5b_over_best=float(gm.t_restore.median() / bst),
                           g5b_frac_chose_loadall=float((gm.meet_rule_used == "none").mean())))
    C = pd.DataFrame(cells)
    G = pd.DataFrame(g5)
    C.to_csv(os.path.join(P01, "p01b_cells.csv"), index=False)
    G.to_csv(os.path.join(P01, "p01b_g5b.csv"), index=False)
    g5sum = {arm: float(x.g5b_over_best.mean()) for arm, x in G.groupby("arm")} if len(G) else {}
    out = dict(run_id=run_id(), ts=ts(), n_cells=len(C), n_flips=int(C.flip.sum()) if len(C) else None,
               flips=C[C.flip].to_dict("records") if len(C) else [], weak_changes=C[C.weak_change].to_dict("records") if len(C) else [],
               passed=bool(len(C) == 24 and not C.flip.any() and not problems),
               g5b_mean_over_best_by_arm=g5sum, problems=problems,
               rule="P0-1 §3.2＋§8 偏離 1：S4 下每格 {Cake, 全部載入} 中位數較小者（差 ≤5% 平手）；兩邊都不是平手且不同＝翻轉；24 格 0 翻轉＝成立")
    json.dump(out, open(os.path.join(P01, "p01b_verdict.json"), "w"), ensure_ascii=False, indent=1, default=str)
    if len(C):
        print(C.drop(columns=["run_id", "ts"]).round(4).to_string(index=False))
    print("flips", out["n_flips"], "passed", out["passed"], "G5b mean ratio", g5sum, "problems", problems)


def _x1_word(dirx, prefix):
    import pandas as pd
    pth = os.path.join(dirx, f"{prefix}_verdict.csv")
    if not os.path.exists(pth):
        return None
    v = pd.read_csv(pth)
    w = v[v.rule == "min"].verdict_rule.unique()
    assert len(w) == 1, (pth, w)
    return str(w[0])


def cmd_analyze_c(a):
    import numpy as np
    import pandas as pd
    words, cross, problems = {}, [], []
    for V in ("V0", "V1", "V2"):
        d = os.path.join(P01, f"sim_{V}")
        if not os.path.isdir(d):
            words[V] = None
            continue
        x5 = os.path.join(d, "x5", "x5_verdict.json")
        words[V] = dict(x1_registered=_x1_word(os.path.join(d, "x1"), "x1"), x1_H1800=_x1_word(os.path.join(d, "x1"), "x1h1800"),
                        x1_H3600=_x1_word(os.path.join(d, "x1"), "x1h3600"),
                        x5_registered=json.load(open(x5))["verdict_registered"] if os.path.exists(x5) else None)
    # V0 回歸：和 git 的 results/p3_explore_h200 比（原登記的 X1 判定與 headroom、X5 判定與 headroom）
    g1 = os.path.join(REPO, "results", "p3_explore_h200", "x1")
    g5 = os.path.join(REPO, "results", "p3_explore_h200", "x5")
    reg = {}
    if words.get("V0"):
        k1 = ["trace", "rule", "cpu_GiB", "slow", "gpu", "restore", "component"]
        a0 = pd.read_csv(os.path.join(g1, "x1_headroom_components.csv"))
        b0 = pd.read_csv(os.path.join(P01, "sim_V0", "x1", "x1_headroom_components.csv"))
        mm = a0.merge(b0, on=k1, suffixes=("_git", "_v0"))
        reg["x1_headroom_max_abs_diff"] = float((mm.headroom_git - mm.headroom_v0).abs().max())
        reg["x1_cells"] = len(mm)
        k5 = ["trace", "model", "cpu_GiB", "variant", "dq", "gpu", "slow"]
        c0 = pd.read_csv(os.path.join(g5, "x5_headroom.csv"))
        e0 = pd.read_csv(os.path.join(P01, "sim_V0", "x5", "x5_headroom.csv"))
        m5 = c0.merge(e0, on=k5, suffixes=("_git", "_v0"))
        reg["x5_headroom_max_abs_diff"] = float((m5.headroom_git - m5.headroom_v0).abs().max())
        reg["x5_cells"] = len(m5)
        reg["x1_word_git"] = _x1_word(g1, "x1")
        reg["x5_word_git"] = json.load(open(os.path.join(g5, "x5_verdict.json")))["verdict_registered"]
        reg["identical"] = reg["x1_headroom_max_abs_diff"] == 0 and reg["x5_headroom_max_abs_diff"] == 0 and \
            reg["x1_word_git"] == words["V0"]["x1_registered"] and reg["x5_word_git"] == words["V0"]["x5_registered"]
        if not reg["identical"]:
            problems.append("V0 和 git 的 results/p3_explore_h200 不是逐值相同（§3.3：要停下來查）")
        # 跨門檻的格子
        k1 = ["trace", "rule", "cpu_GiB", "slow", "gpu", "restore", "component"]
        for V in ("V1", "V2"):
            if not words.get(V):
                continue
            b = pd.read_csv(os.path.join(P01, f"sim_{V}", "x1", "x1_headroom_components.csv"))
            mm = b0.merge(b, on=k1, suffixes=("_v0", "_v"))
            mm = mm[mm.rule == "min"]
            for thr in (0.15, 0.05):
                x = mm[(mm.headroom_v0 >= thr) != (mm.headroom_v >= thr)]
                for _, r in x.iterrows():
                    cross.append(dict(run_id=run_id(), ts=ts(), variant=V, exp="X1", threshold=thr, **{k: r[k] for k in k1},
                                      headroom_v0=r.headroom_v0, headroom_v=r.headroom_v))
            e = pd.read_csv(os.path.join(P01, f"sim_{V}", "x5", "x5_headroom.csv"))
            m5 = e0.merge(e, on=k5, suffixes=("_v0", "_v"))
            m5 = m5[(m5.variant == "reg") & (m5.dq == "k0") & (m5.model == "llama")]
            for thr in (0.15, 0.05):
                x = m5[(m5.headroom_v0 >= thr) != (m5.headroom_v >= thr)]
                for _, r in x.iterrows():
                    cross.append(dict(run_id=run_id(), ts=ts(), variant=V, exp="X5", threshold=thr, **{k: r[k] for k in k5},
                                      headroom_v0=r.headroom_v0, headroom_v=r.headroom_v))
    keys = ("x1_registered", "x1_H3600", "x5_registered")
    judged = {V: (None if not words.get(V) or not words.get("V0") else all(words[V][k] == words["V0"][k] for k in keys))
              for V in ("V1", "V2")}
    pd.DataFrame(cross).to_csv(os.path.join(P01, "p01c_crossings.csv"), index=False)
    out = dict(run_id=run_id(), ts=ts(), words=words, same_as_V0=judged, v0_regression=reg, n_crossings=len(cross),
               passed=bool(judged.get("V2") and (judged.get("V1") in (True, None)) and not problems), problems=problems,
               rule="P0-1 §3.3：V1、V2 的 X1 原登記、X1 截尾 H=3,600、X5 登記判定的字都和 V0 相同；V1 沒跑（S1 停損）時只用 V2 判")
    json.dump(out, open(os.path.join(P01, "p01c_verdict.json"), "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: out[k] for k in ("words", "same_as_V0", "v0_regression", "n_crossings", "passed", "problems")},
                     ensure_ascii=False, indent=1, default=str))


def cmd_verdict(a):
    v = {}
    for k in ("a", "b", "c"):
        pth = os.path.join(P01, f"p01{k}_verdict.json")
        v[k] = json.load(open(pth)) if os.path.exists(pth) else None
    _, choice = _winners()
    stop = {m: x["stop_loss"] for m, x in choice["models"].items()}
    parts = {k: (None if v[k] is None else bool(v[k]["passed"])) for k in v}
    overall = "成立" if all(parts.values()) else "不成立"
    out = dict(run_id=run_id(), ts=ts(), s1_choice={m: choice["models"][m]["winner"] for m in choice["models"]}, s1_stop_loss=stop,
               parts=parts, overall=overall, note="None＝那一部分沒有資料（NOT_MEASURED），不是成立")
    json.dump(out, open(os.path.join(P01, "p01_verdict.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("bw-nfs")
    b.add_argument("--out-dir", required=True)
    for c in ("analyze-a", "analyze-b", "analyze-c", "verdict"):
        sp.add_parser(c)
    a = ap.parse_args()
    {"bw-nfs": cmd_bw_nfs, "analyze-a": cmd_analyze_a, "analyze-b": cmd_analyze_b, "analyze-c": cmd_analyze_c,
     "verdict": cmd_verdict}[a.cmd](a)


if __name__ == "__main__":
    main()
