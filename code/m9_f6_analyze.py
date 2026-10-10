"""m9_f6_analyze.py — F6：D1 用新 seed（5–14）重新確認的膠水〔模擬分析，不碰 GPU〕

判準：docs/research_20261010_followup/README.md「追加 F6」（開 agent 前寫死）。
不改 code/m8_conc_sim.py（sha256 前 16 碼 cbe8e13145235450）；這裡只 import 它。

為什麼要膠水：
  * m8_conc_sim.analyze 把「每格」的表寫到 results/m8_directions/（F6 不准動既有目錄），
    而且通過門檻寫死成 n10 >= 3（D1 的 3／5）。F6 要 ≥6／10。
  * 所以這裡：把 m8.OUT 指到 run 目錄，原封不動呼叫 m8.analyze 算每 seed 的 min_gain，
    再讀它的每 seed 表，自己用 k-of-N 規則聚合。

子命令：
  verdict   每格判定（k 可設；F6 用 k=6），可一次跑多個 metric；每格表寫 results/m9_followup/f6_*.csv
  check     膠水驗證：k=3 套在 D1 的 seed 0–4 輸入上，必須逐格重現 D1 公佈的 *_cfg.csv（n10、n5、pass10、pass5、med_min_gain…）
  regress   模擬驗證：凍結程式重跑 seed 0–4 的列，和 D1 的 sweep CSV 逐列比
  summary   把 verdict 的表整理成 F6 判定（(a)+(b) ρ_KV<1 的通過格數；(c) D1 5 個過載格重現幾格）
"""
from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from datetime import datetime, timezone

os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m8_conc_sim as m8  # noqa: E402

RID = os.environ.get("RUN_ID", "adhoc")
RUNS = os.environ.get("TIARA_RUNS", "/mlsteam/data/tiara/runs")
RD = os.path.join(RUNS, RID)
OUT_REPO = os.path.join(HERE, "..", "results", "m9_followup")
KEY = m8.CFG_KEYS

TW16 = ["S1", "S2b", "S4", "S4+", "S4B", "S4W", "S4C", "S4PD", "S4BT", "S4WT", "S4BD", "S4WD", "S4WD50", "S4WT50",
        "S4CT", "S4CT50"]
TWINS = {"S5T": TW16 + ["S1D"], "S1D": list(TW16)}


def now_ts():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm(d):
    d = d.copy()
    d["qmax"] = d["qmax"].fillna(-1).astype(float)
    for c in ("cpu_gibps", "lam", "cpu_frac", "think"):
        d[c] = d[c].astype(float).round(6)
    d["n_sess"] = d["n_sess"].astype(int)
    return d


def per_seed(inp, cand, cand_ov, twins, metric, tag):
    """原封不動呼叫 m8.analyze（每格表改寫到 run 目錄），回傳它的每 seed 表。"""
    import pandas as pd
    os.makedirs(RD, exist_ok=True)
    m8.OUT = RD                      # 不寫 results/m8_directions/
    m8.RID = RID
    seed_fn, cfg_fn = f"{tag}_seed.csv", f"{tag}_m8cfg.csv"
    m8.analyze(Namespace(inp=inp, cand=cand, cand_ov=cand_ov, twins=twins, metric=metric, seed_to_run=1,
                         out_seed=seed_fn, out_cfg=cfg_fn))
    print(f"[f6] ↑ 上面 m8.analyze 印的 pass10／VERDICT 用的是它寫死的 n10>=3；F6 不用它，下面自己用 k-of-N 聚合。",
          flush=True)
    s = pd.read_csv(os.path.join(RD, seed_fn))
    # 規則 7：m8.analyze 遇到缺的對手會「安靜地不比」——這裡要明確檢查每個對手在每一列都有值
    gcols = [f"g_{t}" for t in twins]
    missing = [c for c in gcols if c not in s.columns]
    if missing:
        raise RuntimeError(f"對手完全沒出現在輸入裡：{missing}")
    nan_rows = int(s[gcols].isna().any(axis=1).sum())
    if nan_rows:
        raise RuntimeError(f"{nan_rows} 列（格×seed）缺某些對手的值")
    if s["cand_val"].isna().any():
        raise RuntimeError("候選的指標有 NaN")
    return _norm(s)


def aggregate(s, k, n_expected):
    import pandas as pd
    g = s.groupby(KEY)
    agg = g.agg(n_seeds=("seed", "size"), seeds=("seed", lambda x: " ".join(str(int(v)) for v in sorted(x))),
                n10=("min_gain", lambda x: int((x >= 0.10).sum())), n5=("min_gain", lambda x: int((x >= 0.05).sum())),
                med_min_gain=("min_gain", "median"), worst_min_gain=("min_gain", "min"),
                best_min_gain=("min_gain", "max"),
                cand_val_med=("cand_val", "median"), cand_val_min=("cand_val", "min"), cand_val_max=("cand_val", "max"),
                rho_kv=("rho_kv", "mean"), rho_kv_min=("rho_kv", "min"), rho_kv_max=("rho_kv", "max"),
                rho_peak=("rho_peak", "mean"), rho_strat_cand=("rho_strat_cand", "mean"),
                gpu_util=("gpu_util", "mean"), cand_drop_frac=("cand_drop_frac", "mean"),
                tightest=("tightest_twin", lambda x: x.value_counts().index[0])).reset_index()
    agg["k_req"] = k
    agg["pass10"] = agg.n10 >= k
    agg["pass5"] = agg.n5 >= k
    agg["rho_lt1"] = agg.rho_kv < 1
    agg["stable_d1"] = (agg.rho_kv < 1) & (agg.gpu_util < 0.95)     # D1 §2.5 的「穩定」，只報告
    bad = agg[agg.n_seeds != n_expected]
    if len(bad):
        raise RuntimeError(f"{len(bad)} 格的 seed 數不是 {n_expected}：\n{bad[KEY + ['n_seeds']].to_string()}")
    return agg


def verdict(a):
    import pandas as pd
    out = []
    for metric in a.metrics:
        tag = f"{a.tag}_{metric}"
        s = per_seed(a.inp, a.cand, a.cand_ov, a.twins, metric, tag)
        exp_seeds = sorted(a.seeds)
        got = sorted(int(x) for x in s.seed.unique())
        if got != exp_seeds:
            raise RuntimeError(f"seed 不對：要 {exp_seeds}，拿到 {got}")
        agg = aggregate(s, a.k, len(exp_seeds))
        agg.insert(0, "metric", metric)
        out.append(agg)
        p = agg[agg.pass10]
        print(f"[f6] {a.tag} cand={a.cand} metric={metric} twins={len(a.twins)} k={a.k}/{len(exp_seeds)}: "
              f"cells={len(agg)} rho<1={int(agg.rho_lt1.sum())} pass10={len(p)} pass10&rho<1={int(p.rho_lt1.sum())} "
              f"pass5={int(agg.pass5.sum())}", flush=True)
        if len(p):
            print(p[KEY + ["n10", "med_min_gain", "worst_min_gain", "rho_kv", "rho_kv_min", "rho_kv_max", "gpu_util",
                           "cand_val_med", "tightest"]].to_string(), flush=True)
        print("  top-8 by med_min_gain:", flush=True)
        print(agg.sort_values("med_min_gain", ascending=False).head(8)[
                  KEY + ["n10", "n5", "med_min_gain", "worst_min_gain", "rho_kv", "tightest"]].to_string(), flush=True)
    df = pd.concat(out, ignore_index=True)
    df.insert(0, "cand", a.cand)
    df.insert(0, "grid", a.tag)
    df.insert(0, "ts", now_ts())
    df.insert(0, "run_id", RID)
    df["n_twins"] = len(a.twins)
    df["twins"] = " ".join(a.twins)
    df["src"] = ";".join(os.path.basename(os.path.dirname(p)) for p in a.inp)
    fn = a.out or os.path.join(OUT_REPO, f"f6_{a.tag}_{a.cand}_cfg.csv")
    if not os.path.isabs(fn):        # 相對名稱 → run 目錄（冒煙測試用，不寫 repo）
        fn = os.path.join(RD, fn)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    df.to_csv(fn, index=False)
    print("[f6] wrote", fn, len(df), "rows", os.path.getsize(fn), "bytes")


def check(a):
    """膠水驗證：k=3 套在 D1 的 seed 0–4 輸入上，要逐格重現 D1 公佈的 cfg 表。"""
    import pandas as pd
    s = per_seed(a.inp, a.cand, a.cand_ov, a.twins, a.metrics[0], a.tag)
    agg = aggregate(s, a.k, 5)
    ref = _norm(pd.read_csv(a.ref))
    m = agg.merge(ref, on=KEY, how="outer", suffixes=("", "_d1"), indicator=True)
    only = m[m._merge != "both"]
    m = m[m._merge == "both"]
    res = dict(run_id=RID, ts=now_ts(), tag=a.tag, cand=a.cand, metric=a.metrics[0], k=a.k, ref=os.path.basename(a.ref),
               ref_run_id=ref.run_id.iloc[0], cells_glue=len(agg), cells_ref=len(ref), cells_unmatched=len(only),
               mism_n10=int((m.n10 != m.n10_d1).sum()), mism_n5=int((m.n5 != m.n5_d1).sum()),
               mism_pass10=int((m.pass10 != m.pass10_d1).sum()), mism_pass5=int((m.pass5 != m.pass5_d1).sum()),
               max_abs_diff_med_min_gain=float((m.med_min_gain - m.med_min_gain_d1).abs().max()),
               max_abs_diff_worst_min_gain=float((m.worst_min_gain - m.worst_min_gain_d1).abs().max()),
               max_abs_diff_rho_kv=float((m.rho_kv - m.rho_kv_d1).abs().max()),
               mism_tightest=int((m.tightest != m.tightest_d1).sum()),
               glue_pass10=int(agg.pass10.sum()), ref_pass10=int(ref.pass10.sum()),
               glue_pass5=int(agg.pass5.sum()), ref_pass5=int(ref.pass5.sum()),
               src=";".join(os.path.basename(os.path.dirname(p)) for p in a.inp))
    res["ok"] = (res["cells_unmatched"] == 0 and res["mism_n10"] == 0 and res["mism_n5"] == 0 and
                 res["mism_pass10"] == 0 and res["mism_pass5"] == 0 and res["max_abs_diff_med_min_gain"] < 1e-9)
    print("[f6-check]", res, flush=True)
    fn = os.path.join(RD, f"{a.tag}_check.csv")
    pd.DataFrame([res]).to_csv(fn, index=False)
    print("[f6-check] wrote", fn)
    if not res["ok"]:
        sys.exit(3)


NUMS = ["median", "mean", "p90", "p99", "svc_median", "qwait_mean", "span_s", "gpu_util", "rho_kv", "rho_peak",
        "rho_strat", "cap_chunks", "new_chunks", "w_ssd_GiB", "recompute", "ld_ssd", "ld_cpu", "n_ret", "n_req",
        "drop_frac", "overflow_drop", "discard", "demote", "ssd_w", "ssd_rej"]


def regress(a):
    """凍結程式重跑 seed 0–4 的列 vs D1 的 sweep CSV，逐列比。"""
    import numpy as np
    import pandas as pd
    new = _norm(pd.concat([pd.read_csv(p) for p in a.new], ignore_index=True))
    old = _norm(pd.concat([pd.read_csv(p) for p in a.old], ignore_index=True))
    if a.ssd_dev:
        old = old[old.ssd_dev.isin(a.ssd_dev)]
        new = new[new.ssd_dev.isin(a.ssd_dev)]
    k = ["strategy", "overflow", "seed"] + KEY
    for d, nm in ((new, "new"), (old, "old")):
        dup = int(d.duplicated(k).sum())
        if dup:
            raise RuntimeError(f"{nm} 有 {dup} 列重複的 key")
    m = new.merge(old, on=k, how="outer", suffixes=("", "_old"), indicator=True)
    only_new = m[m._merge == "left_only"]
    only_old = m[m._merge == "right_only"]
    b = m[m._merge == "both"]
    rows = []
    for c in NUMS:
        if c not in b.columns or c + "_old" not in b.columns:
            continue
        x = pd.to_numeric(b[c], errors="coerce").to_numpy(float)
        y = pd.to_numeric(b[c + "_old"], errors="coerce").to_numpy(float)
        both_nan = np.isnan(x) & np.isnan(y)
        diff = np.where(both_nan, 0.0, np.abs(x - y))
        rows.append(dict(col=c, max_abs_diff=float(np.nanmax(diff)) if len(diff) else 0.0,
                         n_diff=int((diff > 1e-9).sum() + (np.isnan(diff)).sum())))
    r = pd.DataFrame(rows)
    print(r.to_string(), flush=True)
    res = dict(run_id=RID, ts=now_ts(), tag=a.tag, rows_new=len(new), rows_old=len(old), rows_matched=len(b),
               rows_only_new=len(only_new), rows_only_old=len(only_old),
               only_old_strats=" ".join(sorted(only_old.strategy.unique())),
               only_new_strats=" ".join(sorted(only_new.strategy.unique())),
               cols_compared=len(r), cols_with_diff=int((r.n_diff > 0).sum()),
               max_abs_diff_median=float(r.set_index("col").loc["median", "max_abs_diff"]),
               max_abs_diff_any=float(r.max_abs_diff.max()),
               new_src=";".join(os.path.basename(os.path.dirname(p)) for p in a.new),
               old_src=";".join(os.path.basename(os.path.dirname(p)) for p in a.old))
    res["identical"] = res["cols_with_diff"] == 0 and res["rows_only_new"] == 0
    print("[f6-regress]", res, flush=True)
    pd.DataFrame([res]).to_csv(os.path.join(RD, f"{a.tag}_regress.csv"), index=False)


def summary(a):
    """F6 判定。讀 verdict 寫的 results/m9_followup/f6_<grid>_<cand>_cfg.csv。"""
    import pandas as pd
    rows, crows = [], []
    for grid in ("main", "local8", "l8q16"):
        for cand in ("S5T", "S1D"):
            fn = os.path.join(OUT_REPO, f"f6_{grid}_{cand}_cfg.csv")
            d = pd.read_csv(fn)
            for metric, g in d.groupby("metric", sort=False):
                p = g[g.pass10]
                pr = p[p.rho_lt1]
                rows.append(dict(run_id=RID, ts=now_ts(), grid=grid, cand=cand, metric=metric,
                                 in_verdict=(metric == "median"), k_req=int(g.k_req.iloc[0]), n_seeds=int(g.n_seeds.iloc[0]),
                                 cells=len(g), cells_rho_lt1=int(g.rho_lt1.sum()), pass10=len(p),
                                 pass10_rho_lt1=len(pr), pass10_rho_ge1=len(p) - len(pr),
                                 pass5=int(g.pass5.sum()), pass5_rho_lt1=int((g.pass5 & g.rho_lt1).sum()),
                                 max_med_min_gain=float(g.med_min_gain.max()),
                                 max_med_min_gain_rho_lt1=float(g[g.rho_lt1].med_min_gain.max()) if g.rho_lt1.any() else float("nan"),
                                 max_n10=int(g.n10.max()),
                                 max_n10_rho_lt1=int(g[g.rho_lt1].n10.max()) if g.rho_lt1.any() else -1,
                                 pass_rho_kv_min=float(p.rho_kv.min()) if len(p) else float("nan"),
                                 pass_rho_kv_max=float(p.rho_kv.max()) if len(p) else float("nan"),
                                 rho_kv_grid_min=float(g.rho_kv.min()), rho_kv_grid_max=float(g.rho_kv.max()),
                                 src_run=g.run_id.iloc[0], src_sweep=g.src.iloc[0]))
    s = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(s.drop(columns=["ts", "run_id"]).to_string(), flush=True)
    fn = os.path.join(OUT_REPO, "f6_summary.csv")
    s.to_csv(fn, index=False)
    print("[f6] wrote", fn)
    # (a)+(b)：ρ_KV<1 的格、中位數、k=6
    ab = s[(s.metric == "median") & s.grid.isin(["main", "local8"])]
    for cand in ("S5T", "S1D"):
        x = ab[ab.cand == cand]
        print(f"[F6 (a)+(b)] cand={cand}: cells rho<1 = {int(x.cells_rho_lt1.sum())}, "
              f"pass (>=10% vs every twin in >=6/10 seeds) = {int(x.pass10_rho_lt1.sum())}  "
              f"-> {'CONFIRMED (0 pass)' if int(x.pass10_rho_lt1.sum()) == 0 else 'NOT confirmed: claim 4 must be weakened'}")
    # (c)：D1 的 5 個過載格
    ref = _norm(pd.read_csv(a.d1_overload_ref))
    five = ref[ref.pass10].copy()
    if len(five) != 5:
        raise RuntimeError(f"D1 過載格應為 5 格，讀到 {len(five)}")
    five = five[five.ssd_dev == "local8"]
    for cand in ("S1D", "S5T"):
        d = _norm(pd.read_csv(os.path.join(OUT_REPO, f"f6_l8q16_{cand}_cfg.csv")))
        for metric in ("median", "mean", "p90"):
            dm = d[d.metric == metric]
            mm = five[KEY + ["n10", "med_min_gain", "worst_min_gain", "rho_kv", "tightest", "run_id"]].merge(
                dm[KEY + ["n10", "n5", "pass10", "med_min_gain", "worst_min_gain", "rho_kv", "rho_kv_min", "rho_kv_max",
                          "gpu_util", "cand_val_med", "cand_val_min", "cand_val_max", "tightest", "run_id", "seeds"]],
                on=KEY, how="left", suffixes=("_d1", ""))
            if mm.n10.isna().any():
                raise RuntimeError("F6 的 local8 q16 表缺 D1 的某個過載格")
            for _, r in mm.iterrows():
                crows.append(dict(run_id=RID, ts=now_ts(), cand=cand, metric=metric, in_verdict=(cand == "S1D" and metric == "median"),
                                  **{c: r[c] for c in KEY},
                                  d1_n10_of5=int(r.n10_d1), d1_med_min_gain=r.med_min_gain_d1, d1_rho_kv=r.rho_kv_d1,
                                  d1_tightest=r.tightest_d1, d1_run_id=r.run_id_d1,
                                  f6_n10_of10=int(r.n10), f6_n5_of10=int(r.n5), f6_reproduced=bool(r.n10 >= 6),
                                  f6_med_min_gain=r.med_min_gain, f6_worst_min_gain=r.worst_min_gain,
                                  f6_rho_kv=r.rho_kv, f6_rho_kv_min=r.rho_kv_min, f6_rho_kv_max=r.rho_kv_max,
                                  f6_gpu_util=r.gpu_util, f6_cand_val_med=r.cand_val_med, f6_cand_val_min=r.cand_val_min,
                                  f6_cand_val_max=r.cand_val_max, f6_tightest=r.tightest, f6_src_run=r.run_id,
                                  f6_seeds=r.seeds))
    c = pd.DataFrame(crows)
    fn = os.path.join(OUT_REPO, "f6_c_overload.csv")
    c.to_csv(fn, index=False)
    print(c.drop(columns=["ts", "run_id", "think", "n_sess", "d1_run_id", "f6_src_run", "f6_seeds"]).to_string(), flush=True)
    print("[f6] wrote", fn)
    cv = c[c.in_verdict]
    print(f"[F6 (c)] S1D median: D1 overload cells reproduced at >=6/10 = {int(cv.f6_reproduced.sum())}/5  -> "
          f"{'CONFIRMED (only wins under sustained overload)' if cv.f6_reproduced.any() else 'corner is NOISE (0 reproduce)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["verdict", "check", "regress", "summary"])
    ap.add_argument("--inp", nargs="*")
    ap.add_argument("--cand", default="S5T")
    ap.add_argument("--cand-ov", default="pa")
    ap.add_argument("--twins", nargs="*")
    ap.add_argument("--metrics", nargs="*", default=["median"])
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(5, 15)))
    ap.add_argument("--tag", default="f6")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref")
    ap.add_argument("--new", nargs="*")
    ap.add_argument("--old", nargs="*")
    ap.add_argument("--ssd-dev", nargs="*")
    ap.add_argument("--d1-overload-ref",
                    default=os.path.join(HERE, "..", "results", "m8_directions", "d1_afq16_S1D_cfg.csv"))
    a = ap.parse_args()
    if a.twins is None and a.cmd in ("verdict", "check"):
        a.twins = TWINS[a.cand]
    {"verdict": verdict, "check": check, "regress": regress, "summary": summary}[a.cmd](a)


if __name__ == "__main__":
    main()
