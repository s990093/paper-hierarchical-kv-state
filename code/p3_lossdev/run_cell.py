"""run_cell.py — 一個比較（A–E）× 一批格子：訓練、在 valid 挑門檻、在 test 用真實答案算 regret。**合成資料、方法開發，不是結論。**

  TIARA_NO_GPU=1 bash code/run3090.sh p3ld-A python -I code/p3_lossdev/run_cell.py --exp A [--cells all|pi0.6|smoke] [--workers 8]
輸出：$RUN_DIR/cells/<exp>_r<rho>_p<pi>_s<seed>.csv（每格一份）→ 合併成 results/p3_lossdev/<exp>[_<tag>].csv（每列有 run_id、ts）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as C  # noqa: E402

RHOS = (0.0, 0.25, 0.5, 0.75, 1.0)
PIS = (0.3, 0.6, 0.9)
SEEDS = (0, 1, 2)
H10, H1h, H6 = 600.0, 3600.0, 6 * 3600.0


def specs_for(exp):
    L = []
    if exp == "A":
        for H, nm in ((H10, "10m"), (H1h, "1h"), (H6, "6h")):
            L.append(dict(name=f"bce_{nm}", loss="bce", H=H))
        L += [dict(name="logt_imp", loss="logt_imp"), dict(name="hazard", loss="hazard"), dict(name="hazard_cond", loss="hazard_cond"),
              dict(name="cox", loss="cox"), dict(name="aft", loss="aft"), dict(name="aft_cond", loss="aft_cond"),
              dict(name="lava", loss="lava"), dict(name="lambdarank", loss="lambdarank"),
              dict(name="bce_w_tuned", loss="bce_w", H=H6), dict(name="bce_w_fixed", loss="bce_w", H=H6, fixed_thr=0.5),
              dict(name="quantile", loss="quantile")]
    elif exp == "B":
        L += [dict(name="lr", loss="bce", model="lr"), dict(name="lgbm", loss="bce", model="lgbm"),
              dict(name="xgb", loss="bce", model="xgb"), dict(name="mlp_2x64", loss="bce", model="mlp", size=dict(hidden=(64, 64))),
              dict(name="gru", loss="bce", model="gru"), dict(name="rsf", loss="surv", model="rsf"),
              dict(name="deepsurv", loss="surv", model="deepsurv")]
    elif exp == "C":
        for n in (50, 200, 800):
            for d, lv in ((4, 15), (8, 63), (12, 255)):
                L.append(dict(name=f"lgbm_n{n}_l{lv}", loss="bce", model="lgbm", size=dict(n=n, leaves=lv, depth=d), timing=True))
                L.append(dict(name=f"xgb_n{n}_d{d}", loss="bce", model="xgb", size=dict(n=n, depth=d), timing=True))
        for h, nm in (((32,), "1x32"), ((64, 64), "2x64"), ((256,) * 4, "4x256")):
            L.append(dict(name=f"mlp_{nm}", loss="bce", model="mlp", size=dict(hidden=h), timing=True))
        L.append(dict(name="lr", loss="bce", model="lr", timing=True))
    elif exp == "D":
        sets = {"F0": ["F0"], "F0+F1": ["F0", "F1"], "F0+F2t": ["F0", "F2t"], "F0+F2c": ["F0", "F2c"],
                "F0+F1+F2t": ["F0", "F1", "F2t"], "all": ["F0", "F1", "F2t", "F2c"],
                "all-F1": ["F0", "F2t", "F2c"], "all-F2t": ["F0", "F1", "F2c"], "all-F2c": ["F0", "F1", "F2t"]}
        for k, fams in sets.items():
            if k == "all-F2c":
                continue          # 和 F0+F1+F2t 相同，不重算（分析時共用）
            L.append(dict(name=f"feat_{k}", loss="bce", feats=sum((C.FAMILIES[f] for f in fams), [])))
    elif exp == "E":
        L += [dict(name="lgbm", loss="bce", fallback=True), dict(name="lr", loss="bce", model="lr", fallback=True)]
    else:
        raise ValueError(exp)
    return L


def cells_for(which):
    if which == "smoke":
        return [(0.5, 0.6, 0)]
    if which.startswith("list:"):        # list:r,p,s;r,p,s（重跑失敗的格子用）
        return [tuple(float(x) if i < 2 else int(x) for i, x in enumerate(c.split(","))) for c in which[5:].split(";")]
    if which == "pi0.6":
        return [(r, 0.6, s) for r in RHOS for s in SEEDS]
    return [(r, p, s) for r in RHOS for p in PIS for s in SEEDS]


# ------------------------------------------------------------------ 評估
def evaluate(cell, tab, s1v, s2v, s1t, s2t, lam, fixed_thr=None):
    va, te = cell.va1, cell.te1
    has2v = va.index.isin(cell.va2.index)
    has2t = te.index.isin(cell.te2.index)
    s2v = np.where(has2v, s2v, np.inf)
    s2t = np.where(has2t, s2t, np.inf)
    if fixed_thr is None:
        th1, th2 = C.tune(tab, va["S"].to_numpy(), va["tau_hat"].to_numpy(), va["M"].to_numpy(), va["ctx"].to_numpy(),
                          s1v, s2v, has2v, lam)
    else:
        th1 = {c: fixed_thr for c in range(4)}
        th2 = {c: fixed_thr for c in range(4)}
    f1, f2 = C.apply_policy(tab, te["ctx"].to_numpy(), s1t, s2t, has2t, th1, th2)
    tot, b, v, bad = C.cost(tab, te["S"].to_numpy(), te["tau"].to_numpy(), te["M"].to_numpy(), te["ctx"].to_numpy(), f1, f2, lam)
    return dict(C_total=float(tot.sum()), C_bytes=float(b.sum()), C_viol=float(v.sum()), n_viol=int(bad.sum()),
                frac_int2_d1=float((f1 == C.INT2).mean()), th1=json.dumps({int(k): float(x) for k, x in th1.items()}),
                th2=json.dumps({int(k): float(x) for k, x in th2.items()})), (f1, f2, th1, th2)


def fixed_cost(cell, tab, f1, f2, lam):
    te = cell.te1
    tot, b, v, bad = C.cost(tab, te["S"].to_numpy(), te["tau"].to_numpy(), te["M"].to_numpy(), te["ctx"].to_numpy(), f1, f2, lam)
    return dict(C_total=float(tot.sum()), C_bytes=float(b.sum()), C_viol=float(v.sum()), n_viol=int(bad.sum()),
                frac_int2_d1=float((f1 == C.INT2).mean()))


def diag(cell, s1t, s2t, prob):
    te = cell.te1
    read = np.isfinite(te["tau"].to_numpy())
    has2 = te.index.isin(cell.te2.index)
    r = dict(auc_d1=C.auc(read, s1t), auc_d2=C.auc(read[has2], s2t[has2]) if has2.any() else np.nan,
             cindex_d1=C.cindex(te["tau"].to_numpy(), s1t))
    r["brier_d1"] = float(np.mean((np.clip(s1t, 0, 1) - read) ** 2)) if prob else np.nan
    return r


def lru_scores(cell):
    z1v, z1t = np.zeros(len(cell.va1)), np.zeros(len(cell.te1))
    s2v = cell.align2(cell.va1, cell.va2, -cell.va2["a_log"].to_numpy())
    s2t = cell.align2(cell.te1, cell.te2, -cell.te2["a_log"].to_numpy())
    return z1v, s2v, z1t, s2t


def baselines(cell, tab, lam):
    te = cell.te1
    n = len(te)
    ctx = te["ctx"].to_numpy()
    pm = C.pass_min(tab, ctx)
    full = lambda c: np.full(n, c)
    rows = {
        "oracle": C.oracle_actions(tab, te["tau"].to_numpy(), ctx),
        "fixed_all_bf16": (full(C.BF16), full(C.BF16)),
        "fixed_all_int8": (full(C.INT8), full(C.INT8)),
        "fixed_all_int4": (full(C.INT4), full(C.INT4)),
        "fixed_paper_bf16_then_int4": (full(C.BF16), full(C.INT4)),
        "fixed_by_length": (np.where(ctx <= 32768, C.INT4, C.INT8), np.where(ctx <= 32768, C.INT4, C.INT8)),
        "fixed_passmin": (pm, pm),
        "fixed_all_int2": (full(C.INT2), full(C.INT2)),
    }
    out = {k: fixed_cost(cell, tab, f1, f2, lam) for k, (f1, f2) in rows.items()}
    s = lru_scores(cell)
    r, acts = evaluate(cell, tab, *s, lam)
    out["lru"] = r
    return out, acts, s


def fallback_variants(cell, tab, res_model, acts_model, s_model, acts_lru, s_lru, lam, seed):
    """退路：(i) 逐筆 band（|logit p − logit θ| < δ 改用 LRU 的動作，δ 在 valid 上挑）；(ii) 整體（valid 上不顯著好於 LRU 就全用 LRU）。"""
    s1v, s2v, s1t, s2t = s_model
    va, te = cell.va1, cell.te1
    _, _, th1m, th2m = acts_model
    _, _, th1l, th2l = acts_lru
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

    def policy(df, has2, s1, s2, lz1, lz2, delta):
        ctx = df["ctx"].to_numpy()
        fm1, fm2 = C.apply_policy(tab, ctx, s1, s2, has2, th1m, th2m)
        fl1, fl2 = C.apply_policy(tab, ctx, lz1, lz2, has2, th1l, th2l)
        pm = C.pass_min(tab, ctx)
        t1 = np.array([th1m.get(int(c), -np.inf) for c in pm])
        t2 = np.array([th2m.get(int(c), -np.inf) for c in pm])
        u1 = np.abs(lg(s1) - lg(np.clip(t1, 1e-6, 1 - 1e-6))) < delta
        u2 = has2 & (np.abs(lg(np.where(has2, s2, 0.5)) - lg(np.clip(t2, 1e-6, 1 - 1e-6))) < delta)
        f1 = np.where(u1, fl1, fm1)
        # D2：若 D1 用了 LRU 的動作，D2 也跟 LRU；否則照模型，但不確定時用 LRU
        f2 = np.where(u1 | u2, np.where(fl2 == C.INT2, C.INT2, f1), fm2)
        f2 = np.where(f1 == C.INT2, C.INT2, f2)
        return f1, f2

    has2v = va.index.isin(cell.va2.index)
    has2t = te.index.isin(cell.te2.index)
    lzv = (s_lru[0], np.where(has2v, s_lru[1], np.inf))
    lzt = (s_lru[2], np.where(has2t, s_lru[3], np.inf))
    s2vv = np.where(has2v, s2v, np.inf)
    s2tt = np.where(has2t, s2t, np.inf)
    best, bd = None, None
    for delta in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, np.inf):
        f1, f2 = policy(va, has2v, s1v, s2vv, *lzv, delta)
        cv = C.cost(tab, va["S"].to_numpy(), va["tau_hat"].to_numpy(), va["M"].to_numpy(), va["ctx"].to_numpy(), f1, f2, lam)[0].sum()
        if best is None or cv < best - 1e-9:
            best, bd = cv, delta
    f1, f2 = policy(te, has2t, s1t, s2tt, *lzt, bd)
    band = fixed_cost(cell, tab, f1, f2, lam)
    band["fb_delta"] = bd
    # 整體：valid 上 per-session 的成本差，bootstrap
    fm1, fm2 = C.apply_policy(tab, va["ctx"].to_numpy(), s1v, s2vv, has2v, th1m, th2m)
    fl1, fl2 = C.apply_policy(tab, va["ctx"].to_numpy(), *lzv, has2v, th1l, th2l)
    args = (tab, va["S"].to_numpy(), va["tau_hat"].to_numpy(), va["M"].to_numpy(), va["ctx"].to_numpy())
    dm = C.cost(*args, fm1, fm2, lam)[0] - C.cost(*args, fl1, fl2, lam)[0]
    g = pd.Series(dm).groupby(va["sess"].to_numpy()).sum().to_numpy()
    rng = np.random.default_rng(seed + 99)
    bs = np.array([g[rng.integers(0, len(g), len(g))].sum() for _ in range(1000)])
    use_lru = bool(np.quantile(bs, 0.975) >= 0)
    if use_lru:
        f1, f2 = C.apply_policy(tab, te["ctx"].to_numpy(), *lzt, has2t, th1l, th2l)
    else:
        f1, f2 = C.apply_policy(tab, te["ctx"].to_numpy(), s1t, s2tt, has2t, th1m, th2m)
    glob = fixed_cost(cell, tab, f1, f2, lam)
    glob["fb_use_lru"] = use_lru
    glob["fb_valid_diff_ci_hi"] = float(np.quantile(bs, 0.975))
    return band, glob


def timing(spec, out, cell, feats):
    import methods as Mth
    m, p = out["timing_model"]
    X = cell.te1[feats].to_numpy(dtype=float)
    model = spec.get("model", "lgbm")
    if model == "lgbm":
        bst = m.booster_
        fn = lambda Z: bst.predict(Z, num_threads=1)
        nb = len(bst.model_to_string().encode())
    elif model == "xgb":
        m.set_param({"nthread": 1})
        fn = lambda Z: m.inplace_predict(Z)
        nb = len(m.save_raw("ubj"))
    elif model == "mlp":
        import torch
        torch.set_num_threads(1)
        pr, net = m
        def fn(Z):
            with torch.no_grad():
                return net(torch.from_numpy(pr(Z)))
        nb = C.nbytes_torch(net)
    elif model == "lr":
        pr, lr = m
        fn = lambda Z: lr.predict_proba(pr(Z))
        nb = C.nbytes_pickle(m)
    else:
        return {}
    u1, ub = C.time_predict(fn, X)
    import torch
    torch.set_num_threads(Mth.NT)
    return dict(us_batch1=u1, us_per_row_b1024=ub, model_bytes=nb)


def run_one(exp, rho, pi, seed, table_path, run_dir, run_id):
    import methods as Mth
    t0 = time.time()
    tab = C.load_table(table_path)
    cell = Mth.Cell(rho, pi, seed)
    cell.table = tab
    base = dict(run_id=run_id, exp=exp, rho=rho, pi=pi, seed=seed, data_sha=cell.data_sha, table_sha=tab["sha"],
                code_sha=C.code_sha(), run_dir=run_dir, n_test_d1=len(cell.te1), n_test_d2=len(cell.te2),
                n_train_d1=len(cell.tr1), n_train_d2=len(cell.tr2), n_valid_d1=len(cell.va1),
                frac_read_test=float(np.isfinite(cell.te1["tau"]).mean()), c_latent=cell.meta["c"])
    lams = C.LAMBDAS if exp == "A" else (C.LAMBDA_MAIN,)
    if os.environ.get("P3LD_LAMS"):          # 事後加的敏感度（07 §2.10）：例如 P3LD_LAMS=16
        lams = tuple(float(x) for x in os.environ["P3LD_LAMS"].split(","))
    rows = []
    lru_cache = {}
    for lam in lams:
        bl, acts_lru, s_lru = baselines(cell, tab, lam)
        lru_cache[lam] = (acts_lru, s_lru, bl)
        for k, r in bl.items():
            rows.append(dict(base, method=k, kind="baseline", lam=lam, C_oracle=bl["oracle"]["C_total"], **r))
        if lam == lams[0]:
            dl = diag(cell, s_lru[2], np.where(np.isnan(s_lru[3]), 0, s_lru[3]), False)
            rows[-len(bl) + list(bl).index("lru")].update(dl)
    for spec in specs_for(exp):
        try:
            ts0 = time.time()
            out = Mth.scores_for(cell, spec)
            print(f"[spec] {exp} r{rho} p{pi} s{seed} {spec['name']} {time.time() - ts0:.1f}s", file=sys.stderr, flush=True)
        except Exception as e:  # 規則 2：記錄完整錯誤，不靜默跳過
            rows.append(dict(base, method=spec["name"], kind="method", lam=C.LAMBDA_MAIN, error=repr(e),
                             traceback=traceback.format_exc()[-2000:]))
            continue
        # 規則 7：分數有 NaN 不可以靜默變成「壓成 INT2」（NaN ≥ θ 是 False）；整個方法記成錯誤
        h2v, h2t = cell.va1.index.isin(cell.va2.index), cell.te1.index.isin(cell.te2.index)
        nn = dict(s1v=int(np.isnan(out["s1v"]).sum()), s1t=int(np.isnan(out["s1t"]).sum()),
                  s2v=int(np.isnan(out["s2v"][h2v]).sum()), s2t=int(np.isnan(out["s2t"][h2t]).sum()))
        if any(nn.values()):
            rows.append(dict(base, method=spec["name"], kind="method", lam=lams[0], error=f"NaN scores {nn}",
                             traceback=""))
            print(f"[nan] {exp} r{rho} p{pi} s{seed} {spec['name']} {nn}", file=sys.stderr, flush=True)
            continue
        dg = diag(cell, out["s1t"], np.where(np.isnan(out["s2t"]), 0, out["s2t"]), out["prob"])
        tim = {}
        if spec.get("timing") and out.get("timing_model") is not None:
            tim = timing(spec, out, cell, spec.get("feats", C.ALL_FEATS))
        for lam in lams:
            r, acts = evaluate(cell, tab, out["s1v"], out["s2v"], out["s1t"], out["s2t"], lam, out.get("fixed_thr"))
            oc = lru_cache[lam][2]["oracle"]["C_total"]
            rows.append(dict(base, method=spec["name"], kind="method", lam=lam, C_oracle=oc, fit_s=out["fit_s"], **r, **dg, **tim))
            if spec.get("fallback"):
                acts_lru, s_lru, _ = lru_cache[lam]
                band, glob = fallback_variants(cell, tab, r, acts, (out["s1v"], out["s2v"], out["s1t"], out["s2t"]),
                                               acts_lru, s_lru, lam, seed)
                rows.append(dict(base, method=spec["name"] + "+fb_band", kind="method", lam=lam, C_oracle=oc, **band))
                rows.append(dict(base, method=spec["name"] + "+fb_global", kind="method", lam=lam, C_oracle=oc, **glob))
    df = pd.DataFrame(rows)
    df["regret"] = (df["C_total"] - df["C_oracle"]) / df["C_oracle"]
    df["ts"] = C.now_iso()
    df["cell_wall_s"] = time.time() - t0
    os.makedirs(os.path.join(run_dir, "cells"), exist_ok=True)
    fn = os.path.join(run_dir, "cells", f"{exp}_r{rho}_p{pi}_s{seed}.csv")
    df.to_csv(fn, index=False)
    return fn, json.dumps(cell.meta["checks"], ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--cells", default="all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--table", default=None, help="格式表 JSON；預設 placeholder")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    run_id = os.environ.get("RUN_ID", "norun-" + time.strftime("%Y%m%d-%H%M%S"))
    run_dir = os.path.join(os.environ.get("TIARA_RUNS", "/ssd8/hungwei/paper-hkv/runs"), run_id)
    os.makedirs(run_dir, exist_ok=True)
    C.load_table(a.table)   # placeholder 不存在就寫出來
    cells = cells_for(a.cells)
    print(f"[run_cell] exp={a.exp} cells={len(cells)} workers={a.workers} run_dir={run_dir}", flush=True)
    files, fails = [], []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        fut = {ex.submit(run_one, a.exp, r, p, s, a.table, run_dir, run_id): (r, p, s) for r, p, s in cells}
        for f in as_completed(fut):
            try:
                fn, chk = f.result()
                files.append(fn)
                print(f"[done] {fut[f]} {fn} checks={chk}", flush=True)
            except Exception:
                fails.append((fut[f], traceback.format_exc()))
                print(f"[FAIL] {fut[f]}\n{traceback.format_exc()}", flush=True)
    if not files:
        print("[run_cell] 沒有任何格子成功", flush=True)
        return 1
    df = pd.concat([pd.read_csv(x) for x in sorted(files)], ignore_index=True)
    os.makedirs(C.RES, exist_ok=True)
    out = os.path.join(C.RES, f"{a.exp}{('_' + a.tag) if a.tag else ''}.csv")
    if a.cells not in ("all", "pi0.6") and not a.cells.startswith("list:"):
        out = os.path.join(run_dir, f"{a.exp}_{a.cells}.csv")   # smoke 不寫進 results/
    df.to_csv(out, index=False)
    n_err = int(df["error"].notna().sum()) if "error" in df else 0
    print(f"[run_cell] wrote {out} rows={len(df)} method_errors={n_err} cell_failures={len(fails)}", flush=True)
    return 0 if not fails and n_err == 0 else 5


if __name__ == "__main__":
    sys.exit(main())
