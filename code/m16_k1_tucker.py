#!/usr/bin/env python3
"""m16_k1_tucker.py — K1b：Tucker（多維分解）多一個維度有沒有多省；「相鄰層不像」和「跨層低秩有效」怎麼同時成立。
只用 numpy＋scipy，不碰 torch／GPU。

為什麼（使用者 10/22）：「4.1 K1 多一個維度壓縮（Tucker）？」「相鄰層同一 token 的 KV 完全不像，這樣不能跨層？」「TD-MoE 有機會結合？」
設計與事先寫死的判定：docs/stage3_20261019/experiment_log.md「K1b」（看數字前寫的，不改）。摘要：
  Q1 跨層：相鄰層「同 token cosine」 vs 「token 子空間重疊」（每層前 RH 個左奇異向量；‖U_lᵀU_{l+1}‖²_F／RH）與 CKA；去平均版為準。
  Q2 在 xKV 每 4 層（X 系列用的）操作點的同樣位元組下，比 4 維／3 維 Tucker（HOSVD 上界＋任何 Tucker 的下界）、兩側分解、xKV 不同分組。
     GO：兩個模型的 K 都至少一個操作點，Tucker 上界或兩側分解 ≤ 0.9 × xKV-G4（兩題平均）。

張量 X[t, l, c]（c＝h·D＋d）。不去平均（照 m16_xstack 的 xKV）。位元組＝存的浮點數個數 ÷ (T·L·C)（同精度，只比結構；量化另外乘）。
  xKV 分組 G、秩 r：每組 X_g[T, G·C] 截斷 SVD → 存 (L/G)·T·r＋L·C·r。G＝1 是每層各自做；G＝L 就是 Tucker 的 token 模態。
  Tucker（N 維、秩 r_n）：存 Σ_n I_n·r_n＋Π_n r_n。任何 Tucker 的誤差² ≥ max_n（模態 n 丟掉的能量）（模態 n 展開的秩 ≤ r_n，Eckart–Young）；
     HOSVD 的誤差² ≤ Σ_n（模態 n 丟掉的能量）（De Lathauwer et al. 2000）。只算得出上下界，不組核心張量。
  兩側：每組先截斷 SVD 到 r_T（token 係數 A＝UΣ 跨 G 層共用），再把每層的 Σ·W_lᵀ（r_T×C）截斷到 r_C（每層自己的通道基底）。
     兩段誤差正交、可相加（左奇異向量互相正交）。存 (L/G)·T·r_T＋L·(r_T·r_C＋C·r_C)。
指標＝相對 Frobenius 誤差 ‖X−X̂‖／‖X‖。X1 已知它和答對率只中度相關（Spearman 0.50／0.59）——這裡只是篩選。

  python code/m16_k1_tucker.py --self-test
  python code/m16_k1_tucker.py run --dump <dir> --out <dir> [--groups 1 2 4 8 16 32] [--two 4 8] [--max-blocks N]
  python code/m16_k1_tucker.py report --in <dir>

執行：`OMP_NUM_THREADS=8 nice -n 19 taskset -c <8 核> python ...`；大檔（譜）寫到 --spectra（資料盤）。
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import m16_kvstruct as ks  # noqa: E402

RH = 64                                                   # 子空間重疊用的秩
OP = {"K": (8, 11, 16, 22, 28, 32, 42), "V": (8, 17, 22, 34, 42, 48, 64)}   # xKV-G4 的操作點（每 head 每層等效秩）
X_SERIES = {"K": (11, 22, 28), "V": (17, 34, 42)}         # X1–X4 實際用的（xl:k11:v17／k22:v34／k28:v42）
GO_FACTOR = 0.9
REF_G = 4
RGRID = np.geomspace(1 / 256, 1.0, 129)                   # 曲線取樣的位元組比例
FONT_DIR = "/mlsteam/data/tiara/fonts"                 # 和 m16_report_pdf.py 同一套


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def desc_eigvalsh(G: np.ndarray) -> np.ndarray:
    w = np.linalg.eigvalsh(G)[::-1]
    return np.clip(w, 0.0, None)


def desc_eigh(G: np.ndarray):
    w, v = np.linalg.eigh(G)
    return np.clip(w[::-1], 0.0, None), v[:, ::-1]


def cum0(lam: np.ndarray, n: int) -> np.ndarray:
    """cum0[r]＝前 r 個特徵值的和，r＝0..n（長度不夠補 0 個）。"""
    c = np.zeros(n + 1)
    k = min(n, lam.size)
    c[1:k + 1] = np.cumsum(lam[:k])
    c[k + 1:] = c[k]
    return c


def best_under(ratio: np.ndarray, err: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """每個 grid 值：位元組比例 ≤ grid 的設定裡最小的誤差（沒有 → nan）。"""
    o = np.argsort(ratio, kind="stable")
    r, e = ratio[o], np.minimum.accumulate(err[o])
    idx = np.searchsorted(r, grid, side="right") - 1
    out = np.full(grid.shape, np.nan)
    ok = idx >= 0
    out[ok] = e[idx[ok]]
    return out


# ---------------------------------------------------------------- 讀 dump
def load(dp: "ks.Dump", rope: dict, max_blocks: int | None = None):
    nb = dp.nb if not max_blocks else min(dp.nb, max_blocks)
    L, H, D, ch = len(dp.layers), dp.H, dp.D, dp.chunk
    T = nb * ch
    K = np.empty((T, L, H * D), np.float32)
    V = np.empty((T, L, H * D), np.float32)
    for bi in range(nb):
        U = np.asarray(dp.bf16[bi])                                    # [L, 2, H, chunk, D]
        c, s = ks.rope_cos_sin(rope["inv_freq"], dp.positions(bi), rope["attention_scaling"])
        sl = slice(bi * ch, (bi + 1) * ch)
        for j in range(L):
            kp = ks.bf16_to_f32(U[j, 0]).astype(np.float64)            # [H, chunk, D]
            K[sl, j] = ks.rope_invert(kp, c[None], s[None]).transpose(1, 0, 2).reshape(ch, H * D)
            V[sl, j] = ks.bf16_to_f32(U[j, 1]).transpose(1, 0, 2).reshape(ch, H * D)
    return K, V


# ---------------------------------------------------------------- 分析一種（K 或 V）
def layer_stats(X: np.ndarray, H: int, D: int, rh: int = RH) -> dict:
    """每層 Gram、相鄰層的 cosine／子空間重疊／CKA（去平均與不去平均）、Tucker 各模態的 Gram。"""
    T, L, C = X.shape
    Gl, E_l, E_direct = [], np.zeros(L), np.zeros(L)
    pairs = []
    prev = None
    for l in range(L):
        Xl = X[:, l].astype(np.float64)
        G = Xl.T @ Xl
        mu = Xl.mean(0)
        Gc = G - T * np.outer(mu, mu)
        E_l[l] = np.trace(G)
        E_direct[l] = float(np.einsum("tc,tc->", Xl, Xl))
        Gl.append(G)
        lam, Vv = desc_eigh(G)
        lamc, Vc = desc_eigh(Gc)
        k = min(rh, int((lam > lam[0] * 1e-12).sum()))
        kc = min(rh, int((lamc > lamc[0] * 1e-12).sum()))
        U = (Xl @ Vv[:, :k]) / np.sqrt(lam[:k])
        Uc = ((Xl - mu) @ Vc[:, :kc]) / np.sqrt(lamc[:kc])
        cur = dict(Xl=Xl, mu=mu, G=G, Gc=Gc, U=U, Uc=Uc, lam=lam, lamc=lamc)
        if prev is not None:
            A, B = prev, cur
            cos, _ = ks.cos_rows(A["Xl"], B["Xl"])
            cosc, _ = ks.cos_rows(A["Xl"] - A["mu"], B["Xl"] - B["mu"])
            Cx = A["Xl"].T @ B["Xl"]
            Cxc = Cx - T * np.outer(A["mu"], B["mu"])
            pairs.append(dict(
                l=l - 1, cos=float(cos.mean()), cos_centered=float(cosc.mean()),
                overlap=float(np.sum((A["U"].T @ B["U"]) ** 2) / min(A["U"].shape[1], B["U"].shape[1])),
                overlap_centered=float(np.sum((A["Uc"].T @ B["Uc"]) ** 2) / min(A["Uc"].shape[1], B["Uc"].shape[1])),
                cka=float(np.sum(Cx ** 2) / (np.linalg.norm(A["G"]) * np.linalg.norm(B["G"]))),
                cka_centered=float(np.sum(Cxc ** 2) / (np.linalg.norm(A["Gc"]) * np.linalg.norm(B["Gc"]))),
                overlap_random=rh / T))
        prev = cur
    # 規則 6：能量用兩條路徑算（Gram 的 trace、直接平方和）
    assert np.allclose(E_l, E_direct, rtol=1e-9), ("Gram trace 和直接平方和不符", E_l[:3], E_direct[:3])
    # Tucker 各模態（共用）的 Gram
    GC = sum(Gl)                                                                  # 模態 c（head·維度），跨層共用
    GD = sum(G[h * D:(h + 1) * D, h * D:(h + 1) * D] for G in Gl for h in range(H))   # 模態 d，跨層＋跨 head 共用
    GH = np.array([[sum(np.trace(G[a * D:(a + 1) * D, b * D:(b + 1) * D]) for G in Gl) for b in range(H)] for a in range(H)])
    GL = np.zeros((L, L))
    for t0 in range(0, T, 2048):
        Xt = X[t0:t0 + 2048].astype(np.float64)
        GL += np.einsum("tlc,tmc->lm", Xt, Xt)
    E = float(E_l.sum())
    for nm, M in (("C", GC), ("D", GD), ("H", GH), ("L", GL)):
        assert abs(np.trace(M) - E) <= 1e-8 * E, (f"模態 {nm} 的 trace ≠ 總能量", np.trace(M), E)
    per_layer_lam = [desc_eigvalsh(G) for G in Gl]
    return dict(E=E, E_l=E_l, pairs=pairs, Gl=Gl, per_layer_lam=per_layer_lam,
                mode_lam=dict(C=desc_eigvalsh(GC), D=desc_eigvalsh(GD), H=desc_eigvalsh(GH), L=desc_eigvalsh(GL)))


def gram_blocked(M: np.ndarray, bs: int = 4096) -> np.ndarray:
    """MᵀM（float64），每塊 bs 欄分開乘。單次 32768×32768 的 float64 矩陣乘在這台的 OpenBLAS 會 segfault
    （K1b 失敗 5：Llama G＝32，faulthandler 指在 `Xg.T @ Xg`），所以一律分塊。"""
    n = M.shape[1]
    out = np.empty((n, n))
    for i in range(0, n, bs):
        A = M[:, i:i + bs].astype(np.float64)
        for j in range(i, n, bs):
            B = A if j == i else M[:, j:j + bs].astype(np.float64)
            blk = A.T @ B
            out[i:i + bs, j:j + bs] = blk
            if j != i:
                out[j:j + bs, i:i + bs] = blk.T
    return out


def group_spectra(X: np.ndarray, G: int, Gl: list, vectors: bool = False):
    """xKV 分組 G：每組 X_g[T, G·C] 的特徵值（由大到小）與能量；vectors＝True 時也回傳右奇異向量（G·C ≤ T 時）。"""
    T, L, C = X.shape
    out = []
    for g in range(L // G):
        if G == 1:
            G_ = Gl[g]
        elif G * C <= T or vectors:
            G_ = gram_blocked(X[:, g * G:(g + 1) * G].reshape(T, G * C))
        else:
            G_ = gram_blocked(X[:, g * G:(g + 1) * G].reshape(T, G * C).T)
        Eg = float(np.trace(G_))
        if vectors:
            lam, W = desc_eigh(G_)
            out.append(dict(lam=lam, E=Eg, W=W))
        else:
            out.append(dict(lam=desc_eigvalsh(G_), E=Eg))
        del G_
    return out


def xkv_curve(spec: list, T: int, L: int, C: int, G: int, E: float):
    rmax = min(T, G * C)
    r = np.arange(1, rmax + 1)
    disc = sum(s["E"] - cum0(s["lam"], rmax)[1:] for s in spec)
    ratio = ((L / G) * T * r + L * C * r) / (T * L * C)
    return r, ratio, np.sqrt(np.clip(disc, 0, None) / E)


def tucker_bounds(dims: tuple, lams: tuple, E: float, grids: tuple):
    """N 維 Tucker：每個秩組合的 (位元組比例, 下界, 上界)。"""
    mesh = np.meshgrid(*grids, indexing="ij")
    rs = [m.ravel().astype(np.float64) for m in mesh]
    disc = [((E - cum0(lam, int(g.max()))[g.astype(int)]) / E) for lam, g in zip(lams, rs)]
    disc = [np.clip(d, 0, None) for d in disc]
    store = sum(I * r for I, r in zip(dims, rs)) + np.prod(np.stack(rs), axis=0)
    ratio = store / float(np.prod(dims))
    lo = np.sqrt(np.max(np.stack(disc), axis=0))
    hi = np.sqrt(np.minimum(np.sum(np.stack(disc), axis=0), 1.0))
    return ratio, lo, hi, rs


def two_sided(spec_v: list, T: int, L: int, C: int, G: int, H: int, E: float, rT_list):
    """兩側分解：回傳每個 (r_T, r_C) 的 (位元組比例, 誤差)。"""
    rows = []
    for rT in rT_list:
        if rT > G * C or rT > T:
            continue
        disc_T = sum(s["E"] - cum0(s["lam"], rT)[rT] for s in spec_v)
        kmax = min(rT, C)
        tail = np.zeros(kmax + 1)                                # tail[rc]＝每層截到 rc 時丟掉的能量（所有層加總）
        for s in spec_v:
            sq = np.sqrt(s["lam"][:rT])
            for j in range(G):
                M = sq[:, None] * s["W"][j * C:(j + 1) * C, :rT].T          # [r_T, C]
                sv = np.linalg.svd(M, compute_uv=False) ** 2
                c = cum0(sv, kmax)
                tail += c[-1] - c
        for rc in range(1, kmax + 1):
            ratio = ((L / G) * T * rT + L * (rT * rc + C * rc)) / (T * L * C)
            rows.append((rT, rc, ratio, float(np.sqrt(max(disc_T + tail[rc], 0.0) / E))))
    return rows


def analyze(X: np.ndarray, H: int, D: int, groups, two, log) -> dict:
    T, L, C = X.shape
    t0 = time.perf_counter()
    st = layer_stats(X, H, D)
    E = st["E"]
    log(f"  層統計 {time.perf_counter() - t0:.0f}s")
    curves, spectra, ops = {}, {}, {}
    xk = {}
    for G in groups:
        t1 = time.perf_counter()
        spec = group_spectra(X, G, st["Gl"])
        spectra[f"xkv_G{G}"] = [s["lam"] for s in spec]
        r, ratio, err = xkv_curve(spec, T, L, C, G, E)
        xk[G] = (r, ratio, err)
        curves[f"xkv_G{G}"] = best_under(ratio, err, RGRID)
        log(f"  xKV G={G} {time.perf_counter() - t1:.0f}s")
    lamT = np.concatenate(spectra[f"xkv_G{L}"]) if L in groups else None
    assert lamT is not None, "groups 要包含 L（Tucker 的 token 模態＝全部層一組）"
    ml = st["mode_lam"]
    gT = np.unique(np.round(np.geomspace(1, min(T, L * C), 96)).astype(int))
    # 4 維：token × 層 × head × 維度
    r4, lo4, hi4, _ = tucker_bounds((T, L, H, D), (lamT, ml["L"], ml["H"], ml["D"]), E,
                                    (gT, np.arange(1, L + 1), np.arange(1, H + 1), np.arange(1, D + 1)))
    gC = np.unique(np.round(np.geomspace(1, C, 64)).astype(int))
    r3, lo3, hi3, _ = tucker_bounds((T, L, C), (lamT, ml["L"], ml["C"]), E, (gT, np.arange(1, L + 1), gC))
    curves.update(tucker4_hi=best_under(r4, hi4, RGRID), tucker4_lo=best_under(r4, lo4, RGRID),
                  tucker3_hi=best_under(r3, hi3, RGRID), tucker3_lo=best_under(r3, lo3, RGRID))
    tw = {}
    for G in two:
        t1 = time.perf_counter()
        spec_v = group_spectra(X, G, st["Gl"], vectors=True)
        rT_list = sorted({rhe * G * H for rhe in (2, 4, 6, 8, 11, 16, 22, 28, 32, 42, 48, 64, 96)})
        rows = two_sided(spec_v, T, L, C, G, H, E, rT_list)
        a = np.array([[x[2], x[3]] for x in rows])
        tw[G] = a
        curves[f"two_G{G}"] = best_under(a[:, 0], a[:, 1], RGRID)
        del spec_v
        log(f"  兩側 G={G} {time.perf_counter() - t1:.0f}s（{len(rows)} 組）")
    # 操作點：xKV-G4 的位元組下，各方法最好的誤差
    def at(ratio_arr, err_arr, b):
        m = ratio_arr <= b * (1 + 1e-12)
        return float(err_arr[m].min()) if m.any() else None
    return dict(st=st, curves=curves, spectra=spectra, xk=xk, t4=(r4, lo4, hi4), t3=(r3, lo3, hi3), tw=tw, at=at, E=E)


def op_rows(res: dict, kind: str, H: int, L: int, T: int, C: int) -> list:
    r, ratio, err = res["xk"][REF_G]
    at = res["at"]
    rows = []
    for rhe in OP[kind]:
        rr = rhe * REF_G * H
        if rr > r[-1]:
            continue
        b, e_ref = float(ratio[rr - 1]), float(err[rr - 1])
        row = dict(kind=kind, rhe=rhe, in_x_series=rhe in X_SERIES[kind], bytes_ratio=b, factor=1 / b, xkv_G4=e_ref)
        for G, (_, ra, ea) in res["xk"].items():
            row[f"xkv_G{G}"] = at(ra, ea, b)
        r4, lo4, hi4 = res["t4"]
        r3, lo3, hi3 = res["t3"]
        row.update(tucker4_hi=at(r4, hi4, b), tucker4_lo=at(r4, lo4, b), tucker3_hi=at(r3, hi3, b), tucker3_lo=at(r3, lo3, b))
        for G, a in res["tw"].items():
            row[f"two_G{G}"] = at(a[:, 0], a[:, 1], b)
        rows.append(row)
    return rows


# ---------------------------------------------------------------- 子命令
def cmd_run(a) -> int:
    rid = os.environ.get("RUN_ID", "adhoc")
    ts = now()
    dp = ks.Dump(a.dump)
    rc, rope = ks.rope_check(dp)
    if rc["status"] not in ("PASS_PROBE", "PASS_DATA"):
        print(f"RoPE 核對沒過：{rc['status']} {rc['warnings']}", file=sys.stderr)
        return 2
    L, H, D = len(dp.layers), dp.H, dp.D
    if len(dp.layers) != dp.meta["L_total"]:
        print(f"這個 dump 只有部分層（{len(dp.layers)}／{dp.meta['L_total']}），跨層分析要全部層", file=sys.stderr)
        return 2
    groups = [g for g in a.groups if L % g == 0] + ([L] if L not in a.groups else [])
    two = [g for g in a.two if L % g == 0]
    log = lambda m: print(f"[{now()}] {dp.name} {m}", flush=True)  # noqa: E731
    t0 = time.perf_counter()
    K, Vv = load(dp, rope, a.max_blocks)
    T, C = K.shape[0], H * D
    log(f"讀入 T={T} L={L} H={H} D={D}：{time.perf_counter() - t0:.0f}s；groups={groups} two={two}")
    os.makedirs(a.out, exist_ok=True)
    out = dict(run_id=rid, ts=ts, dump=dp.name, model=dp.meta["model"], task=dp.meta["task"], source_run=dp.meta["run_id"],
               T=T, L=L, H=H, D=D, rope_status=rc["status"], groups=groups, two=two, kinds={})
    npz = {}
    for kind, X in (("K", K), ("V", Vv)):
        log(f"{kind}（K＝RoPE 前）")
        res = analyze(X, H, D, groups, two, log)
        st = res["st"]
        ml = st["mode_lam"]
        eff = lambda lam, q: int(np.searchsorted(np.cumsum(lam) / lam.sum(), q) + 1)  # noqa: E731
        out["kinds"][kind] = dict(
            E=res["E"], pairs=st["pairs"],
            mode_eff_rank={m: {f"e{int(q * 100)}": eff(ml[m], q) for q in (0.9, 0.95, 0.99)} | {"size": int(ml[m].size)}
                           for m in ("L", "H", "D", "C")},
            per_layer_eff_rank={f"e{int(q * 100)}": float(np.mean([eff(l_, q) for l_ in st["per_layer_lam"]])) for q in (0.9, 0.95, 0.99)},
            token_mode_eff_rank={f"e{int(q * 100)}": eff(np.concatenate(res["spectra"][f"xkv_G{L}"]), q) for q in (0.9, 0.95, 0.99)},
            curves={k: [None if np.isnan(x) else float(x) for x in v] for k, v in res["curves"].items()},
            op_points=op_rows(res, kind, H, L, T, C))
        for k, v in res["spectra"].items():
            for i, lam in enumerate(v):
                npz[f"{kind}_{k}_{i}"] = lam.astype(np.float32)
        for m, lam in ml.items():
            npz[f"{kind}_mode{m}"] = lam
    out["rgrid"] = RGRID.tolist()
    out["seconds"] = round(time.perf_counter() - t0, 1)
    json.dump(out, open(os.path.join(a.out, f"tucker_{dp.name}.json"), "w"), indent=1, ensure_ascii=False)
    if a.spectra:
        os.makedirs(os.path.dirname(a.spectra), exist_ok=True)
        np.savez_compressed(a.spectra, **npz)
    log(f"完成 {out['seconds']}s")
    return 0


def cmd_report(a) -> int:
    rid = os.environ.get("RUN_ID", "adhoc")
    ts = now()
    js = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.inp, "tucker_*.json")))]
    assert js, "沒有 tucker_*.json"
    # 操作點 CSV
    op_csv, pair_csv = [], []
    for j in js:
        for kind, d in j["kinds"].items():
            for r in d["op_points"]:
                op_csv.append(dict(run_id=j["run_id"], ts=j["ts"], dump=j["dump"], model=j["model"], task=j["task"], **r))
            for p in d["pairs"]:
                pair_csv.append(dict(run_id=j["run_id"], ts=j["ts"], dump=j["dump"], model=j["model"], task=j["task"], kind=kind, **p))
    for name, rows in (("op_points.csv", op_csv), ("xlayer_subspace.csv", pair_csv)):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(a.inp, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    models = sorted({j["model"] for j in js})
    # Q1
    q1 = {}
    for m in models:
        for kind in ("K", "V"):
            ps = [p for j in js if j["model"] == m for p in j["kinds"][kind]["pairs"]]
            q1[f"{m}_{kind}"] = {k: float(np.mean([p[k] for p in ps])) for k in
                                 ("cos", "cos_centered", "overlap", "overlap_centered", "cka", "cka_centered", "overlap_random")}
    def q1_verdict(d):
        if d["overlap_centered"] >= 0.5 and abs(d["cos"]) <= 0.1:
            return "TOKEN_SUBSPACE_SHARED"
        if d["overlap_centered"] < 0.2:
            return "NOT_EXPLAINED"
        return "PARTIAL"
    for k, d in q1.items():
        d["verdict"] = q1_verdict(d)
    # Q2：兩題平均的比值
    q2 = {}
    meths = None
    for m in models:
        for kind in ("K", "V"):
            rows = [r for r in op_csv if r["model"] == m and r["kind"] == kind]
            rhes = sorted({r["rhe"] for r in rows})
            meths = [k for k in rows[0] if k.startswith(("xkv_G", "tucker", "two_G"))]
            tab = []
            for rhe in rhes:
                rr = [r for r in rows if r["rhe"] == rhe]
                ref = np.mean([r["xkv_G4"] for r in rr])
                ent = dict(rhe=rhe, in_x_series=rr[0]["in_x_series"], factor=float(np.mean([r["factor"] for r in rr])), xkv_G4=float(ref))
                for k in meths:
                    vals = [r[k] for r in rr]
                    ent[k] = None if any(v is None for v in vals) else float(np.mean(vals))
                    ent[k + "_ratio"] = None if ent[k] is None else ent[k] / ref
                tab.append(ent)
            cand = [k for k in meths if k in ("tucker4_hi", "tucker3_hi") or k.startswith("two_G")]
            hit = [(t["rhe"], k, t[k + "_ratio"]) for t in tab for k in cand if t[k + "_ratio"] is not None and t[k + "_ratio"] <= GO_FACTOR]
            lo_lose = all((t["tucker4_lo_ratio"] or 0) > 1 and (t["tucker3_lo_ratio"] or 0) > 1 for t in tab)
            q2[f"{m}_{kind}"] = dict(table=tab, hits=hit, tucker_lower_bound_loses_everywhere=lo_lose)
    go = all(q2[f"{m}_K"]["hits"] for m in models)
    verdict = ("GO（Tucker／兩側在某些操作點 ≤0.9×xKV-G4，值得做 GPU 答對率）" if go else
               "STOP（Tucker 共用的維度沒有比 xKV-G4 多省；兩個模型的 K 沒有都達到 0.9×）")
    out = dict(run_id=rid, ts=ts, sources=[j["run_id"] for j in js], dumps=[j["dump"] for j in js], q1=q1, q2=q2,
               verdict_q2=verdict, go=go,
               eff_rank={f"{j['dump']}_{k}": dict(per_layer=d["per_layer_eff_rank"], token_mode=d["token_mode_eff_rank"], modes=d["mode_eff_rank"])
                         for j in js for k, d in j["kinds"].items()},
               rule="Q1：去平均重疊 ≥0.5 且 cosine ≤0.1 → 跨層共用的是 token 子空間；<0.2 → 不能這樣解釋。"
                    "Q2：兩個模型的 K 都至少一個 xKV-G4 操作點，Tucker 上界或兩側分解 ≤0.9×xKV-G4（兩題平均）→ GO，否則 STOP。")
    json.dump(out, open(os.path.join(a.inp, "summary.json"), "w"), indent=1, ensure_ascii=False)
    figs(js, a.inp)
    md(out, js, a.inp, meths)
    print(verdict)
    for k, d in q1.items():
        print(k, {kk: round(v, 3) if isinstance(v, float) else v for kk, v in d.items()})
    return 0


def md(out: dict, js: list, d: str, meths: list):
    f = lambda v, n=3: "—" if v is None else f"{v:.{n}f}"  # noqa: E731
    L_ = [f"# K1b：Tucker 與跨層共用（run {out['run_id']}）", "",
          f"來源：{', '.join(out['sources'])}（dump：{', '.join(out['dumps'])}）。規則：{out['rule']}", "",
          f"**Q2 判定：{out['verdict_q2']}**", "", "## Q1 相鄰層：同 token cosine vs token 子空間重疊（層平均、兩題平均）", "",
          "| 模型／種類 | cosine | cosine（去平均） | 子空間重疊 | 重疊（去平均） | CKA | CKA（去平均） | 隨機重疊 | 判讀 |", "|:--|--:|--:|--:|--:|--:|--:|--:|:--|"]
    for k, q in out["q1"].items():
        L_.append(f"| {k} | {f(q['cos'])} | {f(q['cos_centered'])} | {f(q['overlap'])} | {f(q['overlap_centered'])} | {f(q['cka'])} | "
                  f"{f(q['cka_centered'])} | {f(q['overlap_random'], 4)} | {q['verdict']} |")
    L_ += ["", "## 有效秩（95% 能量需要的維度）", "", "| dump／種類 | 每層各自（平均） | token 模態（全部層一組） | 共用層模態 | 共用 head 模態 | 共用維度模態 | 共用 head·維度模態 |",
           "|:--|--:|--:|--:|--:|--:|--:|"]
    for k, e in out["eff_rank"].items():
        mo = e["modes"]
        L_.append(f"| {k} | {e['per_layer']['e95']:.0f}／{mo['C']['size']} | {e['token_mode']['e95']} | {mo['L']['e95']}／{mo['L']['size']} | "
                  f"{mo['H']['e95']}／{mo['H']['size']} | {mo['D']['e95']}／{mo['D']['size']} | {mo['C']['e95']}／{mo['C']['size']} |")
    for k, q in out["q2"].items():
        L_ += ["", f"## Q2 {k}：xKV-G4 操作點的同樣位元組下的相對誤差（兩題平均；括號＝÷xKV-G4）", "",
               "| 等效秩 | 倍數（結構） | xKV-G4 | " + " | ".join(m for m in meths if m != "xkv_G4") + " |",
               "|--:|--:|--:|" + "--:|" * (len(meths) - 1)]
        for t in q["table"]:
            cells = [f"{f(t[m])}（{f(t[m + '_ratio'], 2)}）" for m in meths if m != "xkv_G4"]
            L_.append(f"| {t['rhe']}{'＊' if t['in_x_series'] else ''} | {t['factor']:.1f} | {f(t['xkv_G4'])} | " + " | ".join(cells) + " |")
        L_.append(f"\n≤0.9 的格子：{q['hits'] or '無'}；Tucker 下界在每個操作點都 >xKV-G4：{q['tucker_lower_bound_loses_everywhere']}")
    L_ += ["", "＊＝X1–X4 實際用的操作點。倍數只算結構（同精度），X 系列再乘係數量化（INT4 約 ×3.2–4）。tucker*_lo＝任何 Tucker 的誤差下界（最樂觀），"
           "tucker*_hi＝HOSVD 的上界；two_G*＝兩側分解（依序截斷）。"]
    open(os.path.join(d, "summary.md"), "w").write("\n".join(L_) + "\n")


def figs(js: list, d: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts = [p for p in glob.glob(os.path.join(FONT_DIR, "NotoSansCJKtc-*.otf"))]
    if not fonts:
        raise SystemExit(f"找不到字型 {FONT_DIR}/NotoSansCJKtc-*.otf（圖裡的中文會變方塊；規則 2：停）")
    for p in fonts:
        font_manager.fontManager.addfont(p)
    plt.rcParams.update({"font.family": "Noto Sans CJK TC", "axes.unicode_minus": True})
    rg = None
    models = sorted({j["model"] for j in js})
    fig, axes = plt.subplots(len(models), 2, figsize=(11, 4.2 * len(models)), squeeze=False)
    for i, m in enumerate(models):
        jm = [j for j in js if j["model"] == m]
        rg = np.array(jm[0]["rgrid"])
        for c, kind in enumerate(("K", "V")):
            ax = axes[i, c]
            keys = list(jm[0]["kinds"][kind]["curves"])
            for k in keys:
                ys = np.array([[np.nan if v is None else v for v in j["kinds"][kind]["curves"][k]] for j in jm]).mean(0)
                sty = dict(lw=1.2)
                if k.startswith("xkv_G"):
                    G = int(k[5:])
                    if G not in (1, REF_G, jm[0]["L"]) and G != 8:
                        continue
                    sty.update(lw=2.2 if G == REF_G else 1.2, color={1: "#999999", REF_G: "#1f5fa8", 8: "#5aa0e0"}.get(G, "#0b2f5c"),
                               label=f"xKV 每 {G} 層" if G > 1 else "每層各自 SVD")
                elif k.startswith("tucker4"):
                    sty.update(color="#c0392b", ls="-" if k.endswith("hi") else ":", label="4 維 Tucker " + ("上界（HOSVD）" if k.endswith("hi") else "下界（最樂觀）"))
                elif k.startswith("tucker3"):
                    continue
                elif k.startswith("two_G"):
                    sty.update(color="#27ae60", ls="--", label=f"兩側（token 跨 {k[5:]} 層＋每層通道基底）")
                ax.plot(1 / rg, ys, **sty)
            for t in (jm[0]["kinds"][kind]["op_points"]):
                if t["in_x_series"]:
                    ax.axvline(t["factor"], color="#bbbbbb", lw=0.6, ls="--")
            ax.set_xscale("log")
            ax.set_xlim(1.5, 64)
            ax.set_ylim(0, 1.0)
            ax.set_xlabel("結構壓縮倍數（同精度；不含量化）")
            ax.set_ylabel("相對誤差")
            ax.set_title(f"{m} {'K（RoPE 前）' if kind == 'K' else 'V'}（兩題平均）")
            ax.grid(alpha=0.3)
            if i == 0 and c == 0:
                ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "tucker_curves.png"), dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(models), figsize=(11, 3.8), squeeze=False)
    for i, m in enumerate(models):
        ax = axes[0, i]
        for kind, col in (("K", "#1f5fa8"), ("V", "#c0392b")):
            ps = {}
            for j in js:
                if j["model"] != m:
                    continue
                for p in j["kinds"][kind]["pairs"]:
                    ps.setdefault(p["l"], []).append(p)
            ls_ = sorted(ps)
            ax.plot(ls_, [np.mean([p["cos"] for p in ps[l]]) for l in ls_], color=col, ls=":", label=f"{kind} 同 token cosine")
            ax.plot(ls_, [np.mean([p["overlap_centered"] for p in ps[l]]) for l in ls_], color=col, ls="-", label=f"{kind} token 子空間重疊（去平均）")
        ax.axhline(RH / js[0]["T"], color="#999999", lw=0.6)
        ax.set_ylim(-0.1, 1.0)
        ax.set_xlabel("層 l（和 l+1 比）")
        ax.set_title(f"{m}：相鄰層")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "xlayer_subspace.png"), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- 自測
def self_test() -> bool:
    rng = np.random.default_rng(0)
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name + (f"  {info}" if info else ""))

    T, L, H, D = 512, 4, 2, 8
    C = H * D
    # 共用 token 子空間、每層自己的通道座標（隨機正交轉換）→ cosine≈0、重疊≈1
    A = rng.standard_normal((T, 6))
    X = np.empty((T, L, C), np.float32)
    for l in range(L):
        Q, _ = np.linalg.qr(rng.standard_normal((C, C)))
        X[:, l] = (A @ rng.standard_normal((6, C))) @ Q + 0.01 * rng.standard_normal((T, C))
    st = layer_stats(X, H, D, rh=6)
    p = st["pairs"][0]
    check("共用 token 子空間：去平均重疊 >0.95", p["overlap_centered"] > 0.95, f"{p['overlap_centered']:.3f}")
    check("共用 token 子空間：同 token cosine 小（|cos|<0.3）", abs(p["cos"]) < 0.3, f"{p['cos']:.3f}")
    # xKV 曲線＝真的截斷 SVD 誤差
    spec = group_spectra(X, 2, st["Gl"])
    r, ratio, err = xkv_curve(spec, T, L, C, 2, st["E"])
    rr = 5
    num = 0.0
    for g in range(2):
        Xg = X[:, g * 2:(g + 1) * 2].reshape(T, 2 * C).astype(np.float64)
        U, S, Vt = np.linalg.svd(Xg, full_matrices=False)
        num += np.sum((Xg - (U[:, :rr] * S[:rr]) @ Vt[:rr]) ** 2)
    check("xKV 曲線＝實際 SVD 重建誤差", abs(err[rr - 1] - np.sqrt(num / st["E"])) < 1e-6, f"{err[rr - 1]:.6f}")
    check("xKV 位元組＝(L/G)Tr＋LCr", abs(ratio[rr - 1] - ((L / 2) * T * rr + L * C * rr) / (T * L * C)) < 1e-12)
    # 兩側分解的誤差公式＝實際重建
    sv = group_spectra(X, 2, st["Gl"], vectors=True)
    rows = two_sided(sv, T, L, C, 2, H, st["E"], [6])
    rc = 3
    e_formula = [x for x in rows if x[0] == 6 and x[1] == rc][0][3]
    num = 0.0
    for g in range(2):
        Xg = X[:, g * 2:(g + 1) * 2].reshape(T, 2 * C).astype(np.float64)
        U, S, Vt = np.linalg.svd(Xg, full_matrices=False)
        Ar = U[:, :6]
        for j in range(2):
            M = S[:6, None] * Vt[:6, j * C:(j + 1) * C]
            u, s, vt = np.linalg.svd(M, full_matrices=False)
            Mh = (u[:, :rc] * s[:rc]) @ vt[:rc]
            num += np.sum((Xg[:, j * C:(j + 1) * C] - Ar @ Mh) ** 2)
    check("兩側分解誤差公式＝實際重建", abs(e_formula - np.sqrt(num / st["E"])) < 1e-6, f"{e_formula:.6f} vs {np.sqrt(num / st['E']):.6f}")
    # Tucker 上下界夾住真的 HOSVD
    Y = rng.standard_normal((40, 4, 2, 8))
    E = float(np.sum(Y ** 2))
    unf = lambda Z, n: np.moveaxis(Z, n, 0).reshape(Z.shape[n], -1)  # noqa: E731
    lams = tuple(desc_eigvalsh(unf(Y, n) @ unf(Y, n).T) for n in range(4))
    ranks = (10, 3, 2, 5)
    Us = [np.linalg.svd(unf(Y, n), full_matrices=False)[0][:, :k] for n, k in enumerate(ranks)]
    core = Y
    for n, U in enumerate(Us):
        core = np.moveaxis(np.tensordot(U.T, np.moveaxis(core, n, 0), axes=1), 0, n)
    Yh = core
    for n, U in enumerate(Us):
        Yh = np.moveaxis(np.tensordot(U, np.moveaxis(Yh, n, 0), axes=1), 0, n)
    e_true = np.sqrt(np.sum((Y - Yh) ** 2) / E)
    rt, lo, hi, rs = tucker_bounds(Y.shape, lams, E, tuple(np.array([k]) for k in ranks))
    check("Tucker 下界 ≤ HOSVD 誤差 ≤ 上界", lo[0] - 1e-9 <= e_true <= hi[0] + 1e-9, f"{lo[0]:.4f} ≤ {e_true:.4f} ≤ {hi[0]:.4f}")
    check("Tucker 位元組＝ΣI·r＋Πr", abs(rt[0] - (40 * 10 + 4 * 3 + 2 * 2 + 8 * 5 + 10 * 3 * 2 * 5) / Y.size) < 1e-12)
    # 全秩 → 誤差 0
    rt, lo, hi, _ = tucker_bounds(Y.shape, lams, E, tuple(np.array([s]) for s in Y.shape))
    check("全秩 Tucker 誤差上界＝0", hi[0] < 1e-6, f"{hi[0]:.2e}")
    # best_under
    bu = best_under(np.array([0.1, 0.2, 0.3]), np.array([0.5, 0.6, 0.2]), np.array([0.05, 0.15, 0.25, 0.35]))
    check("best_under", np.isnan(bu[0]) and np.allclose(bu[1:], [0.5, 0.5, 0.2]), str(bu))
    M = rng.standard_normal((300, 70)).astype(np.float32)
    Mg = M.astype(np.float64)
    check("分塊 Gram＝直接算（欄、列兩個方向）", np.allclose(gram_blocked(M, bs=16), Mg.T @ Mg) and np.allclose(gram_blocked(M.T, bs=16), Mg @ Mg.T))
    check("沒有 import torch／vllm／transformers", not [m for m in ks.FORBIDDEN if m in sys.modules])
    print("self-test：" + ("ALL PASS" if ok else "有 FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("run")
    p.add_argument("--dump", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--groups", type=int, nargs="+", default=[1, 2, 4, 7, 8, 14, 16, 28, 32])
    p.add_argument("--two", type=int, nargs="+", default=[4, 7, 8])
    p.add_argument("--max-blocks", type=int, default=None)
    p.add_argument("--spectra", default=None)
    p = sub.add_parser("report")
    p.add_argument("--in", dest="inp", required=True)
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if a.cmd == "run":
        rc = cmd_run(a)
    elif a.cmd == "report":
        rc = cmd_report(a)
    else:
        ap.print_help()
        rc = 1
    ks.assert_no_gpu_libs()
    sys.exit(rc)


if __name__ == "__main__":
    main()
