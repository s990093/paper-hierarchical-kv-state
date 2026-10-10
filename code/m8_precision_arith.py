"""m8_precision_arith.py — D6：精度當作一層，和 Cake 雙向還原的互動（純算術，不碰 GPU）

判準事先寫在 docs/research_20261010_directions/D6_precision.md §2（2026-10-10T05:37:29Z）。

輸入（都是既有實測）：
  f(i)  results/m7_write_policy_mi300x/calib_c1.csv（run 20261008-130316-m7-c1，GPU 閒，每 chunk 中位數）
        量到 chunk 0–79；>=80 用線性擬合外插（128K 用到），輸出會標 f_source。
  頻寬  results/m7_write_policy_mi300x/tier_params.json（cpu 35.4、local 6.98 讀／1.91 寫、nfs 0.331 讀／0.601 寫）
        加上 3.69（vLLM CPU 實測，C6）、11.6 與 0.81／1.0／2.0／2.9 的掃描點。
還原模型（和 m7_restore_harness.write_boundary 同一個式子，加上反量化）：
  T = min over m >= m_min of max( sum_{i<m} f(i) + [GPU 線] sum_{i>=m} d_i ,  sum_{i>=m} l_i [+ I/O 線的反量化] )
  載入的區段一定是從尾巴往前連續的一段（harness 的載入線遇到沒存的 chunk 就停）。
  在這個模型裡，還原時間只和「載入了幾個、各精度幾個」有關，和排列無關（存了卻被重算的 chunk 只浪費預算）。
  所以「依位置混精度」的 oracle = 在預算內窮舉各精度的個數（每種精度一個計數）。

輸出（都有 run_id、ts）：
  results/m8_directions/d6_f_fit.csv          f(i) 的擬合
  results/m8_directions/d6_full_store.csv     問題 1：整段都存，BF16／FP8／INT8／INT4，32K／128K
  results/m8_directions/d6_budget_mix.csv.gz  問題 2：同預算，均勻精度 vs 依位置混精度（oracle）；未壓縮版在 run 目錄
  results/m8_directions/d6_deferral_cost.csv  問題 2：延後版（降級時才量化）多付的寫入量與時間
  results/m8_directions/d6_summary_gain.csv   帶內最大增益摘要；d6_summary_full.csv 問題 1 的摘要
  results/m8_directions/d6_threshold.csv      （--threshold）反量化門檻細掃
完整掃描另存一份到 $TIARA_RUNS/$RUN_ID/。

D6 的最終 run（2026-10-10）：
  source code/m7_env.sh
  m7run d6-arith python code/m8_precision_arith.py --dmeas fp8=0.087,int4=0.814 \
      --dmeas-src run_20260915-124058-m2-b-llama8b-retrieval_warmTTFT_delta_0.17_1.59_us_per_token   # → 20261010-055349-d6-arith
  m7run d6-thresh python code/m8_precision_arith.py --threshold                                    # → 20261010-055854-d6-thresh
  注意：m7run 的參數不可含空白或括號（runsh 用 $* 寫 cmd.sh，引號會消失）。
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone

os.environ.setdefault("OMP_NUM_THREADS", "8")      # 共用機器：限制 BLAS 執行緒
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
C1 = os.path.join(REPO, "results", "m7_write_policy_mi300x", "calib_c1.csv")
TP = os.path.join(REPO, "results", "m7_write_policy_mi300x", "tier_params.json")
OUT = os.path.join(REPO, "results", "m8_directions")

MiB = 1 << 20
GiB = 1 << 30
CHUNK_TOK = 512
# 每個 chunk 的位元組〔算術〕：Llama-3.1-8B 32 層 x 8 KV head x 128 dim x (K,V) x 512 token
#   BF16 2 B/元素 = 64 MiB；FP8（vLLM 靜態縮放，無中繼資料）= 32 MiB；
#   INT8／INT4 per-token-head：每 token 512 個 FP32 scale = 2 KiB → 每 chunk 多 1 MiB
BYTES = {"bf16": 64 * MiB, "fp8": 32 * MiB, "int8": 33 * MiB, "int4": 17 * MiB}
BAND = [0.331, 0.81, 1.0, 2.0, 2.9, 3.69, 6.98, 11.6]   # 判定用：0.3–12 GiB/s
EXTRA = [35.4]                                         # 帶外，只報告
CTX = {"32K": 64, "128K": 256}
BETAS = [0.125, 0.25, 0.375, 0.5, 0.75, 1.0]
QSETS = {
    "Q1": {"allowed": ["bf16", "int8"], "int4_tail_W": None},
    "Q2": {"allowed": ["bf16", "fp8", "int8", "int4"], "int4_tail_W": None},
    "Q3W1": {"allowed": ["bf16", "int8", "int4"], "int4_tail_W": 1},
    "Q3W4": {"allowed": ["bf16", "int8", "int4"], "int4_tail_W": 4},
}


def run_id():
    return os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc"


def ts():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Out:
    def __init__(self, path, fields):
        self.path = path
        self.f = open(path, "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=["run_id", "ts"] + fields, extrasaction="raise")
        self.w.writeheader()
        self.rid = run_id()

    def row(self, **kw):
        kw.update(run_id=self.rid, ts=ts())
        for k, v in list(kw.items()):
            if isinstance(v, (float, np.floating)):
                kw[k] = round(float(v), 6)
        self.w.writerow(kw)

    def close(self):
        self.f.close()


# ------------------------------------------------------------------ f(i)
def load_f():
    rows = list(csv.DictReader(open(C1)))
    rids = {r["run_id"] for r in rows}
    d = {}
    for r in rows:
        if r["item"] == "f_chunk" and r["gpu_state"] == "idle":
            d.setdefault(int(r["chunk_idx"]), []).append(float(r["ms"]))
    idx = sorted(d)
    assert idx == list(range(len(idx))), "calib_c1 的 chunk_idx 不連續"
    # 規則 6：用另一欄位驗算——pos_end 應該等於 (chunk_idx+1)*512
    for r in rows:
        if r["item"] == "f_chunk":
            assert int(r["pos_end"]) == (int(r["chunk_idx"]) + 1) * CHUNK_TOK, r
    med = np.array([float(np.median(d[i])) for i in idx]) / 1e3     # 秒
    x = np.arange(len(med))
    b, a = np.polyfit(x, med, 1)
    pred = a + b * x
    r2 = 1 - ((med - pred) ** 2).sum() / ((med - med.mean()) ** 2).sum()
    return med, a, b, r2, sorted(rids)


def f_array(n, med, a, b):
    f = np.empty(n)
    k = min(n, len(med))
    f[:k] = med[:k]
    if n > k:
        f[k:] = a + b * np.arange(k, n)
    return f


# ------------------------------------------------------------------ 還原時間
def counts_grid(n, k):
    """所有 k 維非負整數向量、總和 <= n。回傳 (M, k) int32。"""
    if k == 1:
        return np.arange(n + 1, dtype=np.int32)[:, None]
    out = []
    for c0 in range(n + 1):
        sub = counts_grid(n - c0, k - 1)
        out.append(np.hstack([np.full((sub.shape[0], 1), c0, dtype=np.int32), sub]))
    return np.vstack(out)


_GRID_CACHE = {}


def grid(n, k):
    if (n, k) not in _GRID_CACHE:
        g = counts_grid(n, k)
        _GRID_CACHE[(n, k)] = (g, g.astype(np.float64))
    return _GRID_CACHE[(n, k)]


def best_config(F, n, precs, ell, dq, budget, dloc, int4_tail_W=None):
    """在預算內窮舉各精度的載入個數。F[m] = sum_{i<m} f(i)。
    回傳 (T, counts dict, K=載入個數)。"""
    k = len(precs)
    G, Gf = grid(n, k)
    s = np.array([BYTES[p] for p in precs], dtype=np.float64)
    l = np.array([ell[p] for p in precs])
    d = np.array([dq[p] for p in precs])
    K = G.sum(1)
    ok = (Gf @ s) <= budget + 1e-6
    if int4_tail_W is not None and "int4" in precs:
        j = precs.index("int4")
        ok &= G[:, j] <= np.maximum(0, K - int4_tail_W)
    G, Gf, K = G[ok], Gf[ok], K[ok]
    if dloc == "gpu":
        gpu = F[n - K] + Gf @ d
        io = Gf @ l
    elif dloc == "io":     # 反量化在 I/O 線上做管線：每 chunk 取 max(ℓ, d)
        gpu = F[n - K]
        io = Gf @ np.maximum(l, d)
    else:
        raise ValueError(dloc)
    T = np.maximum(gpu, io)
    i = int(np.argmin(T))
    return float(T[i]), {p: int(G[i, j]) for j, p in enumerate(precs)}, int(K[i])


def nondominated(precs, dq):
    """去掉被支配的精度：位元組不比它少、反量化不比它便宜的就不用列舉（不影響 oracle 的最佳值）。"""
    keep = []
    for p in precs:
        dom = any((BYTES[q] <= BYTES[p] and dq[q] <= dq[p]) and (BYTES[q] < BYTES[p] or dq[q] < dq[p])
                  for q in precs if q != p)
        if not dom:
            keep.append(p)
    return keep


def uniform_T(F, n, p, ell, dq, budget, dloc):
    Kmax = min(n, int(budget // BYTES[p]))
    best = (None, None)
    for K in range(Kmax + 1):
        if dloc == "gpu":
            T = max(F[n - K] + K * dq[p], K * ell[p])
        else:
            T = max(F[n - K], K * max(ell[p], dq[p]))
        if best[0] is None or T < best[0]:
            best = (T, K)
    return best


def summarize(rid_rows):
    """從剛寫好的 CSV 算摘要（同一個 run，同一個 run_id）。"""
    import pandas as pd
    d2 = pd.read_csv(os.path.join(OUT, "d6_budget_mix.csv"))
    band = d2[d2.in_band]
    o = Out(os.path.join(OUT, "d6_summary_gain.csv"),
            ["qset", "d_mode", "d_loc", "ctx", "n_cells_in_band", "max_gain", "argmax_bw", "argmax_beta",
             "argmax_PA_counts", "argmax_U_best", "median_gain", "n_cells_ge5pct", "n_cells_ge15pct"])
    for (q, dm, dl, c), x in band.groupby(["qset", "d_mode", "d_loc", "ctx"], sort=True):
        i = x.gain_PA_vs_U.idxmax()
        o.row(qset=q, d_mode=dm, d_loc=dl, ctx=c, n_cells_in_band=len(x), max_gain=x.gain_PA_vs_U.max(),
              argmax_bw=x.loc[i, "bw_GiBps"], argmax_beta=x.loc[i, "beta"], argmax_PA_counts=x.loc[i, "PA_counts"],
              argmax_U_best=x.loc[i, "U_best_prec"], median_gain=x.gain_PA_vs_U.median(),
              n_cells_ge5pct=int((x.gain_PA_vs_U >= 0.05).sum()), n_cells_ge15pct=int((x.gain_PA_vs_U >= 0.15).sum()))
    o.close()
    d1 = pd.read_csv(os.path.join(OUT, "d6_full_store.csv"))
    d1 = d1[d1.T_cake_ms != "NOT_MEASURED"].copy()
    for c in ("T_cake_ms", "T_load_only_ms", "T_compute_only_ms", "meet_b", "T_cake_bf16_ms"):
        d1[c] = d1[c].astype(float)
    o = Out(os.path.join(OUT, "d6_summary_full.csv"),
            ["ctx", "bw_GiBps", "in_band", "d_mode", "d_loc", "prec", "T_cake_ms", "meet_b", "speedup_vs_bf16_cake",
             "cake_gain_vs_best_single_path", "bf16_equiv_bw_GiBps"])
    for _, r in d1.iterrows():
        best1 = min(r.T_load_only_ms, r.T_compute_only_ms)
        o.row(ctx=r.ctx, bw_GiBps=r.bw_GiBps, in_band=r.in_band, d_mode=r.d_mode, d_loc=r.d_loc, prec=r.prec,
              T_cake_ms=r.T_cake_ms, meet_b=int(r.meet_b), speedup_vs_bf16_cake=r.T_cake_bf16_ms / r.T_cake_ms,
              cake_gain_vs_best_single_path=best1 / r.T_cake_ms,
              bf16_equiv_bw_GiBps=r.bw_GiBps * BYTES["bf16"] / BYTES[r.prec])
    o.close()


def threshold(med, a, b):
    """事後加的（D6 §7 偏離 1）：Q1／Q2、反量化在 GPU 線，細掃 d，找混精度增益跨過 5%／15% 的門檻。"""
    o = Out(os.path.join(OUT, "d6_threshold.csv"),
            ["qset", "d_ms_per_chunk", "max_gain_in_band", "argmax_ctx", "argmax_bw", "argmax_beta", "argmax_PA_counts",
             "argmax_U_best"])
    for qname in ("Q1", "Q2"):
        q = QSETS[qname]
        for dv in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0):
            dq = {"bf16": 0.0, "fp8": dv / 1e3, "int8": dv / 1e3, "int4": dv / 1e3}
            best = None
            for ctx, n in CTX.items():
                f = f_array(n, med, a, b)
                F = np.concatenate([[0.0], np.cumsum(f)])
                for bw in BAND:
                    ell = {p: BYTES[p] / (bw * GiB) for p in BYTES}
                    for beta in BETAS:
                        X = beta * n * BYTES["bf16"]
                        Tu = {p: uniform_T(F, n, p, ell, dq, X, "gpu")[0] for p in q["allowed"]}
                        ub = min(Tu, key=Tu.get)
                        Tpa, cnt, K = best_config(F, n, nondominated(q["allowed"], dq), ell, dq, X, "gpu")
                        g = (Tu[ub] - min(Tpa, Tu[ub])) / Tu[ub]
                        if best is None or g > best[0]:
                            best = (g, ctx, bw, beta, ";".join(f"{p}={c}" for p, c in cnt.items() if c), ub)
            o.row(qset=qname, d_ms_per_chunk=dv, max_gain_in_band=best[0], argmax_ctx=best[1], argmax_bw=best[2],
                  argmax_beta=best[3], argmax_PA_counts=best[4], argmax_U_best=best[5])
    o.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", action="store_true", help="只跑 d 門檻細掃（Q1／Q2、GPU 線）")
    ap.add_argument("--dmeas", default="", help="實測反量化成本 ms/chunk，例如 fp8=0.087,int4=0.814；沒量到的精度不要給")
    ap.add_argument("--dmeas-src", default="", help="實測值的來源（run_id／檔案）")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rid = run_id()
    rundir = os.path.join(os.environ.get("TIARA_RUNS", "/mlsteam/data/tiara/runs"), rid)
    os.makedirs(rundir, exist_ok=True)

    med, a, b, r2, frids = load_f()
    if args.threshold:
        threshold(med, a, b)
        shutil.copy(os.path.join(OUT, "d6_threshold.csv"), os.path.join(rundir, "d6_threshold.csv"))
        print("wrote d6_threshold.csv")
        return 0
    tp = json.load(open(TP))
    print(f"f(i): {len(med)} chunks measured, fit f = {a*1e3:.3f} + {b*1e3:.4f}*i ms, R2={r2:.5f}, src={frids}")

    of = Out(os.path.join(OUT, "d6_f_fit.csv"), ["src_run_id", "n_measured", "a_ms", "b_ms_per_chunk", "r2",
                                                 "max_abs_resid_ms", "sum_f_32K_ms", "sum_f_128K_ms", "note"])
    x = np.arange(len(med))
    resid = np.abs(med - (a + b * x)).max() * 1e3
    of.row(src_run_id=";".join(frids), n_measured=len(med), a_ms=a * 1e3, b_ms_per_chunk=b * 1e3, r2=r2,
           max_abs_resid_ms=resid, sum_f_32K_ms=f_array(64, med, a, b).sum() * 1e3,
           sum_f_128K_ms=f_array(256, med, a, b).sum() * 1e3,
           note="chunk>=80 為線性外插〔算術：外插〕；32K 全部實測")
    of.close()

    # 反量化模式
    # 事先登記的是 d0／d5／d10；d0.1–d2 是事後加的敏感度點（找門檻用，見 D6 §7 偏離 1）
    dmodes = {}
    for dv in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0):
        dmodes[f"d{dv:g}"] = {"bf16": 0.0, "fp8": dv / 1e3, "int8": dv / 1e3, "int4": dv / 1e3}
    dmeas_known = set()
    if args.dmeas:
        dm = {"bf16": 0.0}
        for kv in args.dmeas.split(","):
            k_, v_ = kv.split("=")
            dm[k_.strip()] = float(v_) / 1e3
            dmeas_known.add(k_.strip())
        dmodes["dmeas"] = dm
    bws = BAND + EXTRA

    # ---------------- 問題 1：整段都存
    o1 = Out(os.path.join(OUT, "d6_full_store.csv"),
             ["ctx", "n_chunks", "bw_GiBps", "in_band", "prec", "chunk_MiB", "d_mode", "d_ms_per_chunk", "d_loc",
              "T_cake_ms", "meet_b", "n_loaded", "T_compute_only_ms", "T_load_only_ms", "T_cake_bf16_ms",
              "speedup_vs_bf16_cake", "f_source", "note"])
    for ctx, n in CTX.items():
        f = f_array(n, med, a, b)
        F = np.concatenate([[0.0], np.cumsum(f)])
        fsrc = "measured" if n <= len(med) else f"measured0-{len(med)-1}+linear_extrap"
        for bw in bws:
            ell = {p: BYTES[p] / (bw * GiB) for p in BYTES}
            for dname, dq in dmodes.items():
                for dloc in ("gpu", "io"):
                    Tb, _ = uniform_T(F, n, "bf16", ell, dmodes["d0"], n * BYTES["bf16"], dloc)
                    for p in ("bf16", "fp8", "int8", "int4"):
                        if dname == "dmeas" and p != "bf16" and p not in dmeas_known:
                            o1.row(ctx=ctx, n_chunks=n, bw_GiBps=bw, in_band=bw in BAND, prec=p, chunk_MiB=BYTES[p] / MiB,
                                   d_mode=dname, d_ms_per_chunk="NOT_MEASURED", d_loc=dloc, T_cake_ms="NOT_MEASURED",
                                   meet_b="", n_loaded="", T_compute_only_ms=F[n] * 1e3, T_load_only_ms="",
                                   T_cake_bf16_ms=Tb * 1e3, speedup_vs_bf16_cake="", f_source=fsrc,
                                   note="此精度沒有實測反量化成本")
                            continue
                        T, K = uniform_T(F, n, p, ell, dq, n * BYTES[p], dloc)
                        Tl = n * (ell[p] + dq[p]) if dloc == "gpu" else n * max(ell[p], dq[p])
                        o1.row(ctx=ctx, n_chunks=n, bw_GiBps=bw, in_band=bw in BAND, prec=p, chunk_MiB=BYTES[p] / MiB,
                               d_mode=dname, d_ms_per_chunk=dq[p] * 1e3, d_loc=dloc, T_cake_ms=T * 1e3, meet_b=n - K,
                               n_loaded=K, T_compute_only_ms=F[n] * 1e3, T_load_only_ms=Tl * 1e3,
                               T_cake_bf16_ms=Tb * 1e3, speedup_vs_bf16_cake=Tb / T, f_source=fsrc,
                               note="load_only 在 gpu 模式＝ℓ+d 串行（保守）"
                                    + (f"；d 實測來源 {args.dmeas_src}" if dname == "dmeas" else ""))
    o1.close()

    # ---------------- 問題 2：同預算，均勻 vs 依位置混精度
    o2 = Out(os.path.join(OUT, "d6_budget_mix.csv"),
             ["ctx", "n_chunks", "bw_GiBps", "in_band", "beta", "budget_GiB", "qset", "d_mode", "d_loc",
              "T_U_bf16_ms", "T_U_fp8_ms", "T_U_int8_ms", "T_U_int4_ms", "U_best_prec", "T_U_best_ms",
              "T_PA_best_ms", "PA_counts", "PA_n_loaded", "PA_meet_b", "gain_PA_vs_U", "T_compute_only_ms",
              "f_source", "note"])
    for ctx, n in CTX.items():
        f = f_array(n, med, a, b)
        F = np.concatenate([[0.0], np.cumsum(f)])
        fsrc = "measured" if n <= len(med) else f"measured0-{len(med)-1}+linear_extrap"
        for bw in bws:
            ell = {p: BYTES[p] / (bw * GiB) for p in BYTES}
            for beta in BETAS:
                X = beta * n * BYTES["bf16"]
                for qname, q in QSETS.items():
                    for dname, dq in dmodes.items():
                        allowed = list(q["allowed"])
                        note = ""
                        if dname == "dmeas":
                            miss = [p for p in allowed if p != "bf16" and p not in dmeas_known]
                            if miss:
                                note = "排除沒有實測反量化成本的精度：" + "/".join(miss)
                            allowed = [p for p in allowed if p == "bf16" or p in dmeas_known]
                        for dloc in ("gpu", "io"):
                            # 均勻：Q3 的 INT4 不能蓋到尾巴，所以均勻 INT4 不合法
                            Tu = {}
                            for p in ("bf16", "fp8", "int8", "int4"):
                                legal = p in allowed and not (q["int4_tail_W"] is not None and p == "int4")
                                Tu[p] = uniform_T(F, n, p, ell, dq, X, dloc)[0] if legal else None
                            ub = min((p for p in Tu if Tu[p] is not None), key=lambda p: Tu[p])
                            precs = nondominated(allowed, dq) if q["int4_tail_W"] is None else list(allowed)
                            Tpa, cnt, K = best_config(F, n, precs, ell, dq, X, dloc, q["int4_tail_W"])
                            Tpa = min(Tpa, Tu[ub])   # oracle 包含均勻（數值上保險）
                            o2.row(ctx=ctx, n_chunks=n, bw_GiBps=bw, in_band=bw in BAND, beta=beta, budget_GiB=X / GiB,
                                   qset=qname, d_mode=dname, d_loc=dloc,
                                   T_U_bf16_ms=Tu["bf16"] * 1e3 if Tu["bf16"] is not None else "",
                                   T_U_fp8_ms=Tu["fp8"] * 1e3 if Tu["fp8"] is not None else "",
                                   T_U_int8_ms=Tu["int8"] * 1e3 if Tu["int8"] is not None else "",
                                   T_U_int4_ms=Tu["int4"] * 1e3 if Tu["int4"] is not None else "",
                                   U_best_prec=ub, T_U_best_ms=Tu[ub] * 1e3, T_PA_best_ms=Tpa * 1e3,
                                   PA_counts=";".join(f"{p}={c}" for p, c in cnt.items() if c),
                                   PA_n_loaded=K, PA_meet_b=n - K, gain_PA_vs_U=(Tu[ub] - Tpa) / Tu[ub],
                                   T_compute_only_ms=F[n] * 1e3, f_source=fsrc,
                                   note=note + (f"；d 實測來源 {args.dmeas_src}" if dname == "dmeas" else ""))
        print(f"done ctx={ctx}", flush=True)
    o2.close()

    # ---------------- 延後版的代價（降級時才量化）
    o3 = Out(os.path.join(OUT, "d6_deferral_cost.csv"),
             ["ctx", "target_prec", "path", "bytes_final_GiB", "bytes_bf16_GiB", "extra_write_GiB",
              "extra_read_GiB", "ssd_write_amplification", "extra_io_time_s", "extra_transient_capacity_GiB",
              "quantize_compute_s", "note"])
    paths = {
        "gpu_to_cpu_then_cpu_to_ssd_quantize_at_demotion": None,
        "direct_to_local_ssd_bf16_then_rewrite": ("local", tp["local"]["write_GiBps"], tp["local"]["read_GiBps"]),
        "direct_to_nfs_bf16_then_rewrite": ("nfs", tp["nfs"]["write_GiBps"], tp["nfs"]["read_GiBps"]),
    }
    for ctx, n in CTX.items():
        for p in ("fp8", "int8", "int4"):
            fin = n * BYTES[p] / GiB
            bf = n * BYTES["bf16"] / GiB
            for pname, spec in paths.items():
                if spec is None:
                    o3.row(ctx=ctx, target_prec=p, path=pname, bytes_final_GiB=fin, bytes_bf16_GiB=bf,
                           extra_write_GiB=0.0, extra_read_GiB=0.0, ssd_write_amplification=1.0, extra_io_time_s=0.0,
                           extra_transient_capacity_GiB=bf - fin, quantize_compute_s="NOT_MEASURED",
                           note="CPU 層本來就存 BF16；降到 SSD 時才量化，SSD 只寫量化後的位元組。"
                                "代價＝CPU 上量化的運算（或繞 GPU 一趟的 PCIe）＋CPU 暫存 BF16 的時間窗")
                else:
                    dev, wbw, rbw = spec
                    t = bf / wbw + bf / rbw
                    o3.row(ctx=ctx, target_prec=p, path=pname, bytes_final_GiB=fin, bytes_bf16_GiB=bf,
                           extra_write_GiB=bf, extra_read_GiB=bf, ssd_write_amplification=(bf + fin) / fin,
                           extra_io_time_s=t, extra_transient_capacity_GiB=bf - fin, quantize_compute_s="NOT_MEASURED",
                           note=f"{dev}：先寫 BF16（{wbw:.3f} GiB/s）再讀回（{rbw:.3f} GiB/s）再寫量化版；"
                                "寫入時量化只寫 bytes_final")
    o3.close()

    summarize(None)
    for fn in ("d6_f_fit.csv", "d6_full_store.csv", "d6_budget_mix.csv", "d6_deferral_cost.csv",
               "d6_summary_gain.csv", "d6_summary_full.csv"):
        shutil.copy(os.path.join(OUT, fn), os.path.join(rundir, fn))
    # repo 檔案要 < 1 MB：完整的 budget_mix 只留在 run 目錄，repo 放 gzip
    import gzip
    src = os.path.join(OUT, "d6_budget_mix.csv")
    with open(src, "rb") as fi, gzip.open(src + ".gz", "wb", compresslevel=9) as fo:
        shutil.copyfileobj(fi, fo)
    os.remove(src)
    # m7run 與 runsh 各自取 date，可能差 1 秒 → run 目錄（cmd.sh、stdout）與 RUN_ID 不同名。留指標檔互指。
    if not os.path.exists(os.path.join(rundir, "cmd.sh")):
        root = os.path.dirname(rundir)
        stamp = datetime.strptime(rid[:15], "%Y%m%d-%H%M%S")
        short = rid[16:]
        for dt in (1, 2, 3, -1):
            cand = os.path.join(root, (stamp + timedelta(seconds=dt)).strftime("%Y%m%d-%H%M%S") + "-" + short)
            if os.path.exists(os.path.join(cand, "cmd.sh")) and "m8_precision_arith" in open(os.path.join(cand, "cmd.sh")).read():
                open(os.path.join(rundir, "POINTER.txt"), "w").write(f"cmd.sh/stdout.log 在 {cand}（m7run 與 runsh 的 RUN_ID 差 {dt} 秒）\n")
                open(os.path.join(cand, "POINTER.txt"), "w").write(f"CSV 的 run_id = {rid}，輸出副本在 {rundir}\n")
                break
    print("wrote", OUT, "and", rundir)


if __name__ == "__main__":
    sys.exit(main())
