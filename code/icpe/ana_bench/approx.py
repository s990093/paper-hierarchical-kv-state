"""近似方法的誤差評估（只用已經跑完的正式 sweep 當標準答案；不跑模擬）。
方法：
  M1 跨 CPU 容量線性內插（log2 容量）：只跑部分容量，其他容量的 TTFT 用內插。
  M2 只看命中率：② 改成「回來時沒命中快層的塊最少」的對手（ret_slow_loads＋ret_recompute），不是最快的對手。
  M3 跨格子先淘汰對手（racing）：用一個 block 當校準，只留常勝的 k 個對手組合，其他 block 只跑這些。
  M4 兩階段（低保真度先篩）：讀 lowfi.py 的輸出（--lowfi <dir>），每格依低保真 TTFT 留下 ≤(1+δ)×最小值 的對手（至少 top-k），
     ② 用留下的對手的完整 TTFT。
判定一律照 PREREG：上限＝(T② − T④)/T②，看 ≥15%、≥5% 有沒有翻轉。輸出只放誤差與翻轉統計，不報任何格子的上限值。
用法：python approx.py --code <code_dir> --sweeps <csv(.xz)…> --out <dir> [--lowfi <dir> --lowfi-trace <nb:...>]
"""
import argparse, os, sys, json, math, socket
from datetime import datetime
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--code", required=True)
ap.add_argument("--sweeps", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--lowfi", default=None)
ap.add_argument("--lowfi-trace", default=None)
a = ap.parse_args()
sys.path.insert(0, os.path.join(a.code, "p3x")); sys.path.insert(0, a.code)
import x1_sim as XS  # noqa: E402

RUN_ID = os.environ.get("RUN_ID", "adhoc")
META = dict(run_id=RUN_ID, ts=datetime.now().astimezone().isoformat(timespec="seconds"), host=socket.gethostname())
CELL = ["trace", "cpu_GiB", "slow", "gpu"]
d = pd.concat([pd.read_csv(p) for p in a.sweeps], ignore_index=True)
d = d[(d.restore == "cake") & (d.pf_target == "cpu")].copy()
GP_NO = sorted(x for x in d.gpu_policy.unique() if x != "BEL")
COMPS = {c: (XS.COMPONENT_POLICIES[c], XS.RIVALS[c]) for c in XS.RIVALS}


def headrooms(T, allowed=None):
    """T：{(cell)->{(cpu_pol, gpu_pol): T}}。allowed：② 那邊只准用的 (cpu_pol, gpu_pol) 集合（None＝全部）。
    回傳 {(cell, comp): (h, T4, T2, best_rival)}。和 next_sim_analyze 的元件定義相同（O-ORD/O-DROP/O-NOW/O-PF/O-GPU）。"""
    out = {}
    for cell, t in T.items():
        for comp, (opol, rpol) in list(COMPS.items()) + [("O-GPU", (XS.NON_ORACLE, XS.NON_ORACLE))]:
            if comp == "O-GPU":
                oset = [(p, "BEL") for p in XS.NON_ORACLE]
            else:
                oset = [(p, q) for p in opol for q in GP_NO]
            rset = [(p, q) for p in rpol for q in GP_NO]
            if any(k not in t for k in oset + rset):
                continue
            if allowed is not None:
                rset = [k for k in rset if k in allowed] or rset
            T4 = min(t[k] for k in oset)
            r = min(rset, key=lambda k: t[k])
            out[(cell, comp)] = ((t[r] - T4) / t[r], T4, t[r], r)
    return out


def table(df, col="mean_ret"):
    T = {}
    for r in df[CELL + ["policy", "gpu_policy", col]].itertuples(index=False):
        T.setdefault(tuple(r[:4]), {})[(r[4], r[5])] = r[6]
    return T


def compare(H0, H1, name, extra):
    keys = [k for k in H0 if k in H1]
    h0 = np.array([H0[k][0] for k in keys]); h1 = np.array([H1[k][0] for k in keys])
    f15 = int(((h0 >= 0.15) != (h1 >= 0.15)).sum()); f5 = int(((h0 >= 0.05) != (h1 >= 0.05)).sum())
    up15 = int(((h0 < 0.15) & (h1 >= 0.15)).sum()); dn15 = int(((h0 >= 0.15) & (h1 < 0.15)).sum())
    err = np.abs(h1 - h0) * 100
    return dict(method=name, n=len(keys), flips15=f15, flips15_false_go=up15, flips15_false_nogo=dn15, flips5=f5,
                abs_err_pp_median=round(float(np.median(err)), 4) if len(err) else None,
                abs_err_pp_p99=round(float(np.percentile(err, 99)), 4) if len(err) else None,
                abs_err_pp_max=round(float(err.max()), 4) if len(err) else None, **extra)


T_true = table(d)
H_true = headrooms(T_true)
# 自檢：和 next_sim_analyze 對同一份 sweep 的定義一致（只比元件與列數，數值由同一套 min 算）
print("cells", len(T_true), "cell×comp", len(H_true), flush=True)
res = []

# ---------------- M1：跨容量內插
caps = sorted(d.cpu_GiB.unique())
L2 = {c: math.log2(c) for c in caps}
for name, keep in [("M1-interp keep{8,64,1024}", [8, 64, 1024]), ("M1-interp keep{8,32,128,512,1024}", [8, 32, 128, 512, 1024]),
                   ("M1-interp keep all but 111.06", [c for c in caps if abs(c - 111.0625) > 1e-9]),
                   ("M1-interp keep{8,16,32,64,128,256,1024}", [8, 16, 32, 64, 128, 256, 1024])]:
    keep = sorted(keep)
    Tp = {}
    for (tr, c, sl, gg), t in T_true.items():
        if c in keep:
            continue
        lo = max(k for k in keep if k < c); hi = min(k for k in keep if k > c)
        w = (L2[c] - L2[lo]) / (L2[hi] - L2[lo])
        tl, th = T_true.get((tr, lo, sl, gg)), T_true.get((tr, hi, sl, gg))
        Tp[(tr, c, sl, gg)] = {k: tl[k] + w * (th[k] - tl[k]) for k in t if k in tl and k in th}
    Hp = headrooms(Tp)
    res.append(compare({k: v for k, v in H_true.items() if k in Hp}, Hp, name,
                       dict(jobs_saved_frac=round(1 - len(keep) / len(caps), 3))))

# ---------------- M2：只看命中率
Mi = table(d.assign(miss=d.ret_slow_loads + d.ret_recompute), "miss")
Hp = {}
agree = tot = 0
for (cell, comp), (h, T4, T2, r) in H_true.items():
    t, m = T_true[cell], Mi[cell]
    rpol = XS.NON_ORACLE if comp == "O-GPU" else COMPS[comp][1]
    rset = [(p, q) for p in rpol for q in GP_NO]
    rm = min(rset, key=lambda k: (m[k], t[k]))
    Hp[(cell, comp)] = ((t[rm] - T4) / t[rm], T4, t[rm], rm)
    agree += rm == r; tot += 1
res.append(compare(H_true, Hp, "M2-hit-ratio picks ②", dict(proxy_picks_true_best_frac=round(agree / tot, 4), jobs_saved_frac=None)))
# 命中率排序 vs TTFT 排序：每格 rival 間的 Spearman
rhos = []
for cell, t in T_true.items():
    ks = [k for k in t if k[0] in XS.NON_ORACLE and k[1] in GP_NO]
    x = pd.Series([Mi[cell][k] for k in ks]).rank(); y = pd.Series([t[k] for k in ks]).rank()
    if x.std() > 0 and y.std() > 0:
        rhos.append(float(np.corrcoef(x, y)[0, 1]))
res.append(dict(method="M2 rank corr(miss, TTFT) across rivals per cell", n=len(rhos),
                spearman_median=round(float(np.median(rhos)), 3), spearman_min=round(float(min(rhos)), 3)))

# ---------------- M3：racing（一個 trace 校準，其他 trace 用）
traces = sorted(d.trace.unique())
if len(traces) > 1:
    cal = traces[0]
    wins = {}
    for (cell, comp), (h, T4, T2, r) in H_true.items():
        if cell[0] != cal:
            continue
        t = T_true[cell]
        for k, v in t.items():
            if k[0] in XS.NON_ORACLE and k[1] in GP_NO and v <= T2 * 1.01:
                wins[k] = wins.get(k, 0) + 1
    ranked = sorted(wins, key=lambda k: -wins[k])
    nriv = len([1 for p in XS.NON_ORACLE for q in GP_NO])
    for kk in (3, 5, 10, 20):
        allowed = set(ranked[:kk])
        Hp = headrooms({c: t for c, t in T_true.items() if c[0] != cal}, allowed)
        res.append(compare({k: v for k, v in H_true.items() if k[0][0] != cal}, Hp, f"M3-racing top{kk} rival combos (calib {cal})",
                           dict(rival_jobs_saved_frac=round(1 - kk / nriv, 3))))

# ---------------- M4：兩階段（低保真）
if a.lowfi:
    full = pd.read_csv(os.path.join(a.lowfi, "lowfi_full.csv"))
    # 驗證：Rust 完整版（依序相加的平均）對正式 sweep 的 mean_ret
    m = full.assign(trace=a.lowfi_trace).merge(d[d.trace == a.lowfi_trace][CELL + ["policy", "gpu_policy", "mean_ret"]],
                                               on=CELL + ["policy", "gpu_policy"], suffixes=("_rust", "_formal"))
    rel = (m.mean_ret_rust / m.mean_ret_formal - 1).abs()
    res.append(dict(method="Rust full vs formal sweep mean_ret", n=len(m), exact_equal=int((rel == 0).sum()),
                    max_rel_diff=float(rel.max()), median_rel_diff=float(rel.median())))
    Tt = {c: t for c, t in T_true.items() if c[0] == a.lowfi_trace}
    Ht = {k: v for k, v in H_true.items() if k[0][0] == a.lowfi_trace}
    tfull = full.job_s.sum()
    for f in sorted(os.listdir(a.lowfi)):
        if not f.startswith("lowfi_") or f == "lowfi_full.csv":
            continue
        lf = pd.read_csv(os.path.join(a.lowfi, f)).assign(trace=a.lowfi_trace)
        var = lf.variant.iloc[0]
        Tl = table(lf)
        # 低保真本身的排序品質：每格 rival 的 Spearman（低保真 vs 完整）
        rh = []
        for cell, t in Tt.items():
            ks = [k for k in t if k[0] in XS.NON_ORACLE and k[1] in GP_NO and k in Tl.get(cell, {})]
            x = pd.Series([Tl[cell][k] for k in ks]).rank(); y = pd.Series([t[k] for k in ks]).rank()
            if len(ks) > 2 and x.std() > 0 and y.std() > 0:
                rh.append(float(np.corrcoef(x, y)[0, 1]))
        for delta in (0.0, 0.02, 0.05, 0.10):
            for topk in (1, 3, 5):
                kept_n = tot_n = 0
                Hp = {}
                for (cell, comp), (h, T4, T2, r) in Ht.items():
                    t, tl = Tt[cell], Tl[cell]
                    rpol = XS.NON_ORACLE if comp == "O-GPU" else COMPS[comp][1]
                    rset = [(p, q) for p in rpol for q in GP_NO]
                    mn = min(tl[k] for k in rset)
                    srt = sorted(rset, key=lambda k: tl[k])
                    kept = set(srt[:topk]) | {k for k in rset if tl[k] <= mn * (1 + delta)}
                    kept_n += len(kept); tot_n += len(rset)
                    rk = min(kept, key=lambda k: t[k])
                    Hp[(cell, comp)] = ((t[rk] - T4) / t[rk], T4, t[rk], rk)
                res.append(compare(Ht, Hp, f"M4-two-stage {var} δ={delta} top{topk}",
                                   dict(rival_kept_frac=round(kept_n / tot_n, 3), stage1_cost_frac=round(lf.job_s.sum() / tfull, 3),
                                        lowfi_rank_spearman_median=round(float(np.median(rh)), 3) if rh else None)))

os.makedirs(a.out, exist_ok=True)
R = pd.DataFrame(res)
for i, (k, v) in enumerate(META.items()):
    R.insert(i, k, v)
R.to_csv(os.path.join(a.out, "approx_eval.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 60)
print(R.drop(columns=list(META)).to_string())
