"""synth_ext.py — loss 開發用的合成資料（擴充 code/p3_data/synth.py，不改原檔）。**答案已知；合成資料、方法開發，不是結論。**

和 synth.py 的差別（docs/phase3_20261017/07_lossdev_synth.md §2.1）
  * 「回不回來、多久回來」由一個潛在分數 Y 決定（Gaussian copula），可以控制線索強度 ρ 與不回來的比例 π：
      Y = ρ·(√w·u_session + √(1−w)·m_turn) + √(1−ρ²)·η，u、m、η ~ N(0,1) 獨立。
      Y > c ⇒ 不回來（end）；否則 U'＝Φ(Y)/Φ(c)：U' < p_tool ⇒ 工具（間隔＝tool log-normal 的 U'/p_tool 分位數），
      否則人（human log-normal 的 (U'−p_tool)/(1−p_tool) 分位數）。Y 越大越晚回來、最大的不回來。
      c 用二分搜尋校準：session 平均輪數 E_u[1/P(end|u)]＝1/π（每個 session 恰好一個 end ⇒ 所有 D1 中 end 的比例＝π）。
  * u 是 session 層級（看得到的途徑：過去的間隔與次數＝F2t、F2c），m 是這一輪（看得到的途徑：中繼資料 meta_*＝F1）。
    ρ＝0：所有特徵和答案獨立。w 的上限：session 層的變異 ρ²w 要 < 1/3，session 長度的二階矩才有限（見 §2.10）。
  * context、輸出長度、服務時間沿用 synth.py 的擬合池與公式（FreeInference week01；擬合值，不是真實參數）。
  * 外層搬走時刻 M ~ LogUniform(m_lo, m_hi)，和答案獨立；τ > M 的輪在 a＝M 有第二個決定 D2。

  python -I code/p3_lossdev/synth_ext.py check   # 小樣本自我檢查（斷言）
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FIT_PATH = os.path.join(REPO, "results", "p3_data", "synth_fit_fi_week01.json")
CONST_PATH = os.path.join(REPO, "results", "p3_data", "service_constants.json")
KV_BYTES_LLAMA = 32 * 8 * 128 * 2 * 2   # 131072 B/token（架構算的，同 synth.py）

DEFAULTS = dict(n_d1=30_000, w=0.2, horizon_s=7 * 86400, m_lo_s=60.0, m_hi_s=4 * 3600.0, max_turns=2000)
SPLIT_FRACS = (0.6, 0.8)


class SynthAssertionError(AssertionError):
    pass


def _load_fit():
    if not os.path.exists(FIT_PATH):
        raise FileNotFoundError(f"{FIT_PATH} 不存在（先跑 code/p3_data/synth.py fit）")
    fj = json.load(open(FIT_PATH, encoding="utf-8"))
    const = json.load(open(CONST_PATH, encoding="utf-8"))
    return fj, const


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def mean_turns(c, rho, w, n_gh=80):
    """E_u[1/P(end|u)]，P(end|u)＝P(Y>c|u)。Gauss–Hermite 積分（u ~ N(0,1)）。"""
    x, wt = np.polynomial.hermite_e.hermegauss(n_gh)
    wt = wt / wt.sum()
    a = rho * np.sqrt(w)
    s = np.sqrt(max(1e-12, 1 - a * a))
    p_end = ndtr(-(c - a * x) / s)          # 不用 1−ndtr：尾端會變成 0
    return float(np.sum(wt / np.maximum(p_end, 1e-300)))


def calibrate_c(pi_end, rho, w):
    target = 1.0 / pi_end
    lo, hi = -8.0, 8.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mean_turns(mid, rho, w) > target:
            hi = mid          # 輪數太多 ⇒ end 太少 ⇒ c 要更小
        else:
            lo = mid
    return 0.5 * (lo + hi)


def generate(rho, pi_end, seed, **kw):
    """回傳 (d1: DataFrame, meta: dict)。d1 每列＝一個回合結束（D1 決定）；D2 由 tau > M 推得（見 make_d2）。"""
    p = dict(DEFAULTS, **kw)
    w = float(p["w"])
    if rho * rho * w >= 1 / 3:
        raise SynthAssertionError(f"ρ²w={rho * rho * w:.3f} ≥ 1/3：session 長度的二階矩無限（§2.10）")
    fj, const = _load_fit()
    lnp = fj["lognormal"]
    mix = fj["mixture"]
    p_tool = mix["tool"] / (mix["tool"] + mix["human"])
    pools = fj["pools"]
    fc = np.array(pools["first_context"], dtype=np.int64)
    inc_t = np.array(pools["ctx_increment_tool"], dtype=np.int64)
    inc_u = np.array(pools["ctx_increment_user"], dtype=np.int64)
    outs = np.array(pools["output_tokens"], dtype=np.int64)
    H = float(p["horizon_s"])
    c = calibrate_c(pi_end, rho, w)
    rng = np.random.default_rng(int(seed) + 1000 * int(round(rho * 100)) + 7 * int(round(pi_end * 100)))
    n_s = int(round(p["n_d1"] * pi_end))
    a_s = rho * np.sqrt(w)
    a_m = rho * np.sqrt(1 - w)
    a_e = np.sqrt(max(0.0, 1 - rho * rho))
    u = rng.standard_normal(n_s)
    s0 = np.sort(rng.uniform(0, H, size=n_s))
    # 每個 session 的狀態（向量化，一次一輪）
    active = np.ones(n_s, dtype=bool)
    t_cur = s0.copy()                      # 這一輪開始的時刻（秒）
    ctx = fc[rng.integers(0, len(fc), size=n_s)].astype(np.int64)
    trig_tool = np.zeros(n_s, dtype=np.int8)
    k = np.zeros(n_s, dtype=np.int64)
    sum_lg = np.zeros(n_s)
    min_lg = np.full(n_s, np.nan)
    prev_lg = np.full(n_s, np.nan)
    last_h_lg = np.full(n_s, np.nan)
    n_tool = np.zeros(n_s, dtype=np.int64)
    n_human = np.zeros(n_s, dtype=np.int64)
    rows = []
    n_trunc_turns = 0
    for step in range(int(p["max_turns"])):
        idx = np.nonzero(active)[0]
        if len(idx) == 0:
            break
        n = len(idx)
        m = rng.standard_normal(n)
        eta = rng.standard_normal(n)
        Y = a_s * u[idx] + a_m * m + a_e * eta
        out = outs[rng.integers(0, len(outs), size=n)]
        svc = np.maximum(0.0, const["A"] + const["B"] * out + const["C"] * ctx[idx]) / 1000.0
        t_end = t_cur[idx] + svc
        end = Y > c
        U = ndtr(Y) / ndtr(c)
        is_tool = (~end) & (U < p_tool)
        is_h = (~end) & ~is_tool
        q = np.where(is_tool, U / p_tool, (U - p_tool) / (1 - p_tool))
        q = np.clip(q, 1e-12, 1 - 1e-12)
        z = ndtri(q)
        mu = np.where(is_tool, lnp["tool"]["mu"], lnp["human"]["mu"])
        sg = np.where(is_tool, lnp["tool"]["sigma"], lnp["human"]["sigma"])
        gap = np.where(end, np.inf, np.exp(mu + sg * z) / 1000.0)    # 秒
        # 中繼資料（m 的雜訊觀測）
        meta_score = m + 0.3 * rng.standard_normal(n)
        meta_flag = (m + rng.standard_normal(n) > 0).astype(np.int8)
        mc = m + 0.5 * rng.standard_normal(n)
        meta_cat = np.searchsorted(ndtri(np.arange(1, 8) / 8.0) * np.sqrt(1.25), mc).astype(np.int8)
        M = np.exp(rng.uniform(np.log(p["m_lo_s"]), np.log(p["m_hi_s"]), size=n))
        in_win = t_cur[idx] < H
        rows.append(dict(
            sess=idx, turn_idx=k[idx].copy(), t_start=t_cur[idx].copy(), t_end=t_end, s0=s0[idx],
            ctx=ctx[idx].copy(), out=out, trig_tool=trig_tool[idx].copy(),
            meta_cat=meta_cat, meta_flag=meta_flag, meta_score=meta_score,
            prev_gap_log=prev_lg[idx].copy(), mean_gap_log=np.where(k[idx] > 0, sum_lg[idx] / np.maximum(k[idx], 1), np.nan),
            min_gap_log=min_lg[idx].copy(), last_human_gap_log=last_h_lg[idx].copy(),
            n_tool_ret=n_tool[idx].copy(), n_human_ret=n_human[idx].copy(),
            tau=gap, cls=np.where(end, 2, np.where(is_tool, 0, 1)).astype(np.int8), Y=Y, M=M, keep=in_win))
        # 更新狀態
        lg = np.log10(np.where(end, 1.0, np.maximum(gap, 1e-3)))
        cont = ~end
        ii = idx[cont]
        sum_lg[ii] += lg[cont]
        min_lg[ii] = np.where(np.isnan(min_lg[ii]), lg[cont], np.minimum(min_lg[ii], lg[cont]))
        prev_lg[ii] = lg[cont]
        hh = is_h[cont]
        last_h_lg[ii[hh]] = lg[cont][hh]
        n_tool[ii] += is_tool[cont]
        n_human[ii] += is_h[cont]
        trig_tool[ii] = is_tool[cont].astype(np.int8)
        inc = np.where(is_tool[cont], inc_t[rng.integers(0, len(inc_t), size=int(cont.sum()))],
                       inc_u[rng.integers(0, len(inc_u), size=int(cont.sum()))])
        ctx[ii] = np.maximum(1, ctx[ii] + inc)
        t_cur[ii] = t_end[cont] + gap[cont]
        k[ii] += 1
        active[idx[end]] = False
        active[ii] = t_cur[ii] < H                 # 下一輪開始超過觀察窗 ⇒ 不再產生（真實答案已記在這一輪）
    else:
        n_trunc_turns = int(active.sum())
    d = pd.DataFrame({kk: np.concatenate([r[kk] for r in rows]) for kk in rows[0]})
    d = d[d["keep"]].drop(columns=["keep"]).sort_values(["t_end", "sess"], kind="stable").reset_index(drop=True)
    d["hour"] = ((d["t_end"] // 3600) % 24).astype(np.int16)
    d["sess_age_log"] = np.log10(1.0 + d["t_end"] - d["s0"])
    d["ctx_log"] = np.log10(d["ctx"].astype(float))
    d["out_log"] = np.log10(1.0 + d["out"].astype(float))
    d["S"] = d["ctx"].astype(float) * KV_BYTES_LLAMA
    d["read"] = np.isfinite(d["tau"]).astype(np.int8)
    tr, va = SPLIT_FRACS[0] * H, SPLIT_FRACS[1] * H
    d["split"] = np.where(d["t_end"] < tr, "train", np.where(d["t_end"] < va, "valid", "test"))
    d["obs_end"] = np.where(d["split"] == "train", tr, np.where(d["split"] == "valid", va, H))
    meta = dict(rho=rho, pi_end=pi_end, seed=seed, params=p, c=c, p_tool=p_tool, n_sessions=n_s,
                n_d1=int(len(d)), n_trunc_sessions_maxturns=n_trunc_turns,
                fit_sha256=_sha(FIT_PATH), const_sha256=_sha(CONST_PATH))
    _checks(d, meta)
    return d, meta


def _checks(d, meta):
    """規則 6 式的斷言：用一個量驗算另一個量。不過就丟錯。"""
    chk = {}
    # (1) 校準：所有 D1 中「不回來」的比例 ≈ π（由 session 數／輪數驗算 c 的積分）。觀察窗截斷的 session 會讓比例略低。
    frac_end = float((d["cls"] == 2).mean())
    chk["frac_end"] = frac_end
    if abs(frac_end - meta["pi_end"]) > 0.03:
        raise SynthAssertionError(f"不回來的比例 {frac_end:.4f} 和 π={meta['pi_end']} 差 > 0.03")
    # (2) 單調：Y 越大越晚（同一類內 Spearman＝1；end 的 Y 全部 > 回來的）
    from scipy.stats import spearmanr
    for cl in (0, 1):
        sub = d[d["cls"] == cl]
        if len(sub) > 50:
            r = spearmanr(sub["Y"], sub["tau"]).statistic
            chk[f"spearman_Y_tau_cls{cl}"] = float(r)
            if r < 0.999:
                raise SynthAssertionError(f"cls={cl} 的 Y 與間隔不單調：spearman={r}")
    if (d["cls"] == 2).any() and (d["cls"] != 2).any():
        if d.loc[d["cls"] == 2, "Y"].min() <= d.loc[d["cls"] != 2, "Y"].max():
            raise SynthAssertionError("end 的 Y 不全大於回來的 Y")
    # (3) 特徵沒有偷看答案：ρ＝0 時 meta_score 和 Y 的相關 ≈ 0
    r_meta = float(np.corrcoef(d["meta_score"], d["Y"])[0, 1])
    chk["corr_meta_score_Y"] = r_meta
    if meta["rho"] == 0 and abs(r_meta) > 0.05:
        raise SynthAssertionError(f"ρ=0 但 corr(meta_score, Y)={r_meta}")
    # (4) 時間：t_end ≥ t_start；下一輪的 t_start ＝ 這一輪 t_end ＋ tau（同 session）
    if (d["t_end"] < d["t_start"]).any():
        raise SynthAssertionError("t_end < t_start")
    g = d.sort_values(["sess", "turn_idx"])
    nxt = g.groupby("sess")["t_start"].shift(-1)
    has = nxt.notna()
    err = np.abs(nxt[has] - (g.loc[has, "t_end"] + g.loc[has, "tau"]))
    chk["max_abs_next_start_err_s"] = float(err.max()) if has.any() else 0.0
    if has.any() and err.max() > 1e-6 * max(1.0, float(g["t_end"].max())):
        raise SynthAssertionError(f"下一輪開始 ≠ t_end＋tau：max err {err.max()}")
    # (5) 輪數＝過去回來次數＝工具＋人
    if not (d["turn_idx"] == d["n_tool_ret"] + d["n_human_ret"]).all():
        raise SynthAssertionError("turn_idx ≠ n_tool_ret + n_human_ret")
    # 只報告：回來的間隔中位數 vs 擬合值（copula 的選擇效應會讓它偏移，不擋）
    fj, _ = _load_fit()
    for cl, name in ((0, "tool"), (1, "human")):
        sub = d.loc[d["cls"] == cl, "tau"]
        if len(sub):
            chk[f"{name}_gap_p50_s"] = float(sub.median())
            chk[f"{name}_gap_p50_fit_s"] = fj["lognormal"][name]["q50_ms"] / 1000.0
    chk["frac_tool_among_returns"] = float((d["cls"] == 0).sum() / max(1, (d["cls"] != 2).sum()))
    chk["n_sessions_with_d1"] = int(d["sess"].nunique())
    meta["checks"] = chk


def observed(d, dp):
    """dp='D1'／'D2'：回傳該決定點的列（D2 只有 tau > M 的列），加上決定時刻 t_dec、已閒置 a、
    從決定起算的觀察時間 t_obs 與 e_obs（在該 split 的觀察窗結束處設限），以及 usable（決定時刻在觀察窗內）。"""
    if dp == "D1":
        x = d.copy()
        x["a"] = 0.0
    else:
        x = d[d["tau"] > d["M"]].copy()
        x["a"] = x["M"]
    x["t_dec"] = x["t_end"] + x["a"]
    rem_true = x["tau"] - x["a"]                       # 從決定起算的真實剩餘時間（inf＝不回來）
    win = x["obs_end"] - x["t_dec"]                    # 觀察窗剩多少
    x["e_obs"] = (rem_true <= win).astype(np.int8)
    x["t_obs"] = np.where(x["e_obs"] == 1, rem_true, win)
    x["usable"] = win > 0
    x["a_log"] = np.log10(1.0 + x["a"])
    return x


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "check":
        for rho in (0.0, 0.5, 1.0):
            for pi in (0.3, 0.9):
                d, meta = generate(rho, pi, 0)
                print(json.dumps(dict(rho=rho, pi=pi, c=round(meta["c"], 4), n=meta["n_d1"], **{k: (round(v, 4) if isinstance(v, float) else v)
                                                                                         for k, v in meta["checks"].items()}), ensure_ascii=False))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
