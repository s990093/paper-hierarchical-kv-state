"""synth.py — S 級全合成產生器（03_data_design.md §6）。**答案已知**；只用來測程式與機制，不能撐「學得到」的主張。

  python -I code/p3_data/synth.py fit            # 由 FreeInference week01 的統一表擬合起點參數 → results/p3_data/synth_fit_fi_week01.json
  python -I code/p3_data/build.py synth          # 依 PRESETS 產生 → p3_unified/synth/<variant>/

模型（每個 session 獨立）
  * session 開始時刻 ~ Uniform(0, horizon_s)（固定個數的 Poisson 過程），依序排好。
  * 每一輪之後「下一步」三選一，每輪獨立：tool（工具型，秒級）、human（人，分鐘級）、end（不回來）；機率 p_tool、p_human、p_end。
    → 每個 session 的輪數 ~ Geometric(p_end)。
  * 間隔（下一輪開始 − 這一輪結束）~ LogNormal(μ_c, σ_c)，c∈{tool, human}。
  * 第 0 輪的 context ~ 從 FreeInference week01 的第 0 輪 input_tokens 重抽；第 k 輪的 context ＝ 第 k−1 輪 ＋ 增量，
    增量從 FreeInference week01「同一種 trigger 的相鄰兩輪 input_tokens 差」重抽（可為負：壓縮），context 至少 1。
  * output_tokens 從 FreeInference week01 重抽；服務時間 ＝ max(0, A + B·output + C·input)（service_constants.json，同樣是擬合值）。
  * 觀察窗 [0, horizon_s]：開始時刻超過窗的輪丟掉；窗內最後一輪若真的還有下一輪 → 右設限。
  * 訊號特徵：s ＝「到下一輪的真實時間」的常態分數（依秩；end＝+∞ 排最後，同分隨機）；
    x_feat_rho_<ρ> ＝ ρ·s ＋ √(1−ρ²)·ε，ε~N(0,1) 所有 ρ 共用（common random numbers）。ρ＝0 特徵沒用，ρ＝1 特徵＝答案的排序。
  * KV 位元組：x_kv_bytes_llama ＝ context × 131,072（Llama-3.1-8B：32 層 × 8 KV head × 128 × 2(K,V) × 2 B）、
    x_kv_bytes_qwen ＝ context × 57,344（Qwen2.5-7B：28 × 4 × 128 × 2 × 2 B）。由架構算出，不是量測。
  * 真實答案：x_true_next_class（tool／human／end）、x_true_think_ms（end＝null；**不受觀察窗截斷**）。
同參數同 seed → 逐位元相同（test_p3_data.py 測 parquet 的 sha256）。所有亂數只從一個 np.random.default_rng(seed) 依固定順序抽。
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import ndtri  # noqa: E402

import schema as S  # noqa: E402

SOURCE = "synth"
FIT_PATH = os.path.join(os.environ.get("P3_DATA_RESULTS", os.path.join(S.REPO, "results", "p3_data")), "synth_fit_fi_week01.json")   # P3_DATA_RESULTS：見 build.py
CONST_PATH = os.path.join(os.environ.get("P3_DATA_RESULTS", os.path.join(S.REPO, "results", "p3_data")), "service_constants.json")
KV_BYTES = {"llama": 32 * 8 * 128 * 2 * 2, "qwen": 28 * 4 * 128 * 2 * 2}
Z90 = 1.2815515655446004  # Φ⁻¹(0.9)
DEFAULTS = dict(n_sessions=10_000, seed=0, horizon_s=7 * 86400, signal_strength=[0.0, 0.25, 0.5, 0.75, 1.0],
                mixture=None, lognormal=None)  # None＝用擬合值
PRESETS = {
    "default-seed0": dict(),
    "small-seed1": dict(n_sessions=500, seed=1, horizon_s=86400),
}


def fit(week_variant="week01"):
    """FreeInference 的統一表（已轉好）→ 起點參數。分位數擬合 log-normal：μ＝ln p50、σ＝(ln p90 − ln p10)/(2·1.2816)。"""
    df = S.read_unified("freeinference", week_variant)
    man = S.read_manifest("freeinference", week_variant)
    df = df.sort_values(["session_id", "turn_idx"])
    nt = df["next_trigger"].fillna("end").values
    th = df["think_ms"].to_numpy(dtype="float64", na_value=np.nan)
    p = {c: float(np.mean(nt == k)) for c, k in (("tool", "tool"), ("human", "user"), ("end", "end"))}
    ln = {}
    for c, k in (("tool", "tool"), ("human", "user")):
        x = th[(nt == k) & ~np.isnan(th) & (th > 0)]
        q10, q50, q90 = np.percentile(x, [10, 50, 90])
        ln[c] = dict(mu=float(np.log(q50)), sigma=float((np.log(q90) - np.log(q10)) / (2 * Z90)), n=int(len(x)),
                     q10_ms=float(q10), q50_ms=float(q50), q90_ms=float(q90),
                     n_nonpositive_excluded=int(np.sum((nt == k) & ~np.isnan(th) & (th <= 0))))
    first = df.loc[df["turn_idx"] == 0, "input_tokens"].astype(int).tolist()
    prev = df.groupby("session_id")["input_tokens"].shift(1)
    inc = (df["input_tokens"] - prev)
    m = prev.notna()
    inc_by = {t: inc[m & (df["trigger"] == t)].astype(int).tolist() for t in ("tool", "user")}
    outs = df["output_tokens"].dropna().astype(int).tolist()
    return dict(fitted_from=dict(source="freeinference", variant=week_variant, week=man["week"], run_id=man["run_id"],
                                 turns_sha256=man["output_sha256"]),
                label="擬合值（fitted）：FreeInference week01 的經驗分位數與經驗分布；不是真實參數",
                mixture=p, lognormal=ln, pools=dict(first_context=first, ctx_increment_tool=inc_by["tool"],
                                                     ctx_increment_user=inc_by["user"], output_tokens=outs),
                n_rows=int(len(df)), n_sessions=int(df["session_id"].nunique()))


def generate(params, fit_json, const):
    """回傳 (統一表 DataFrame, 統計)。只用一個 rng，抽的順序固定。"""
    rng = np.random.default_rng(int(params["seed"]))
    n_s = int(params["n_sessions"])
    H = int(params["horizon_s"] * 1000)
    mix = params.get("mixture") or fit_json["mixture"]
    lnp = params.get("lognormal") or fit_json["lognormal"]
    pools = fit_json["pools"]
    p_end = mix["end"]
    p_tool_c = mix["tool"] / (mix["tool"] + mix["human"])
    # 1. session 開始與輪數
    s0 = np.sort(rng.integers(0, H, size=n_s, endpoint=False))
    nturn = rng.geometric(p_end, size=n_s)
    T = int(nturn.sum())
    sess = np.repeat(np.arange(n_s), nturn)
    first = np.r_[0, np.cumsum(nturn)[:-1]]
    k = np.arange(T) - first[sess]
    last = k == nturn[sess] - 1
    # 2. 每輪之後的下一步
    u = rng.random(T)
    cls = np.where(last, 2, np.where(u < p_tool_c, 0, 1))          # 0 tool、1 human、2 end
    z = rng.standard_normal(T)
    mu = np.where(cls == 0, lnp["tool"]["mu"], lnp["human"]["mu"])
    sg = np.where(cls == 0, lnp["tool"]["sigma"], lnp["human"]["sigma"])
    gap = np.where(cls == 2, np.nan, np.round(np.exp(mu + sg * z)))
    # 3. context、輸出、服務時間
    fc = np.array(pools["first_context"], dtype=np.int64)
    it, iu = np.array(pools["ctx_increment_tool"], dtype=np.int64), np.array(pools["ctx_increment_user"], dtype=np.int64)
    ou = np.array(pools["output_tokens"], dtype=np.int64)
    c0 = fc[rng.integers(0, len(fc), size=n_s)]
    trig = np.empty(T, dtype=object)
    trig[k == 0] = "user"
    prev_cls = np.r_[2, cls[:-1]]
    trig[k > 0] = np.where(prev_cls[k > 0] == 0, "tool", "user")
    a_t = it[rng.integers(0, len(it), size=T)]
    a_u = iu[rng.integers(0, len(iu), size=T)]
    inc = np.where(k == 0, 0, np.where(trig == "tool", a_t, a_u))
    ctx = np.empty(T, dtype=np.int64)
    cs = np.cumsum(inc)
    ctx = c0[sess] + cs - cs[first[sess]]
    ctx = np.maximum(ctx, 1)   # 先算再截：截斷不累積（簡單、可重現；寫在文件）
    out = ou[rng.integers(0, len(ou), size=T)]
    svc = np.round(np.maximum(0.0, const["A"] + const["B"] * out + const["C"] * ctx)).astype(np.int64)
    eps = rng.standard_normal(T)
    tie = rng.random(T)
    # 4. 時間：t_start[k] = s0 + Σ_{j<k}(svc[j] + gap[j])
    step = svc + np.nan_to_num(gap, nan=0).astype(np.int64)
    cst = np.cumsum(step)
    before = cst - step - (cst - step)[first[sess]]
    t_start = s0[sess] + before
    t_end = t_start + svc
    # 5. 訊號特徵（全部 T 列的秩，觀察窗截斷之前）
    key = np.where(cls == 2, np.inf, gap)
    order = np.lexsort((tie, key))
    rank = np.empty(T, dtype=np.float64)
    rank[order] = np.arange(T)
    s = ndtri((rank + 0.5) / T)
    feats = {}
    for rho in params["signal_strength"]:
        feats[f"x_feat_rho_{int(round(rho * 100)):03d}"] = rho * s + np.sqrt(max(0.0, 1 - rho ** 2)) * eps
    # 6. 觀察窗
    keep = t_start < H
    n = int(keep.sum())
    df = pd.DataFrame(dict(
        source=SOURCE, variant=params["variant"], session_id=[f"s{int(x):06d}" for x in sess[keep]], user_id=None,
        session_rule="synthetic", trigger=trig[keep], t_start_ms=t_start[keep].astype(np.int64),
        t_end_ms=pd.array(t_end[keep], dtype="Int64"), t_end_source="synthetic", input_tokens=pd.array(ctx[keep], dtype="Int64"),
        output_tokens=pd.array(out[keep], dtype="Int64"), cached_tokens=pd.array([pd.NA] * n, dtype="Int64"),
        tokenizer="o200k_base(resampled from freeinference week01)", new_user_tokens=pd.array([pd.NA] * n, dtype="Int64"),
        context_tokens=pd.array(ctx[keep], dtype="Int64"), block_ids=None, block_tokens=pd.array([pd.NA] * n, dtype="Int32"),
        tool_name=None, tool_category=None, tool_latency_ms=pd.array([pd.NA] * n, dtype="Int64"), req_type="synthetic", text_ref=None,
        obs_end_ms=H, t_start_source="synthetic", think_source="synthetic", service_est_ms=pd.array(svc[keep], dtype="Int64"),
        split_user=None, harness=None, is_subagent=None, parent_session_id=None, next_waited_for=None,
        provider_prompt_tokens=pd.array([pd.NA] * n, dtype="Int64"),
        x_true_next_class=np.array(["tool", "human", "end"], dtype=object)[cls[keep]],
        x_true_think_ms=pd.array(gap[keep], dtype="Float64").astype("Int64"),
        x_kv_bytes_llama=ctx[keep] * KV_BYTES["llama"], x_kv_bytes_qwen=ctx[keep] * KV_BYTES["qwen"],
        _order=k[keep], **{c: v[keep] for c, v in feats.items()}))
    df, fst = S.finalize(df, chain=True, by_time=False)
    split, bounds, seg_end = S.time_split(df, 0, H)
    df["split"] = split
    sw = S.apply_split_window(df, seg_end)
    df = df.drop(columns=["_order"])
    st = dict(n_turns_generated=T, n_turns_in_window=n, finalize=fst, split_bounds=bounds, split_window=sw,
              censored_but_truly_returns=int((df["censored"] & (df["x_true_next_class"] != "end")).sum()))
    return df, st


def build_variant(variant):
    params = dict(DEFAULTS, **PRESETS[variant], variant=variant)
    if not os.path.exists(FIT_PATH):
        raise FileNotFoundError(f"{FIT_PATH} 不存在：先跑 synth.py fit（規則 7）")
    fj = json.load(open(FIT_PATH, encoding="utf-8"))
    const = json.load(open(CONST_PATH, encoding="utf-8"))
    df, st = generate(params, fj, const)
    ck = S.Checks(SOURCE, variant)
    # 產生器自己的一致性：觀察到的下一輪 ⇒ 真實答案一致
    obs = df["think_ms"].notna()
    ck.eq("observed think_ms == x_true_think_ms", bool((df.loc[obs, "think_ms"] == df.loc[obs, "x_true_think_ms"]).all()), True)
    ck.eq("observed next_trigger matches true class", bool(((df.loc[obs, "next_trigger"] == "tool") ==
                                                              (df.loc[obs, "x_true_next_class"] == "tool")).all()), True)
    ck.eq("not censored => true class != end", bool((df.loc[~df["censored"], "x_true_next_class"] != "end").all()), True)
    cr = df["censored"] & (df["x_true_next_class"] != "end")
    ck.eq("censored & returns => true next start >= horizon",
          bool(((df.loc[cr, "t_end_ms"] + df.loc[cr, "x_true_think_ms"]) >= params["horizon_s"] * 1000).all()), True)
    for rho in params["signal_strength"]:
        c = f"x_feat_rho_{int(round(rho * 100)):03d}"
        from scipy.stats import spearmanr
        key = np.where(df["x_true_next_class"] == "end", np.inf, df["x_true_think_ms"].to_numpy(dtype="float64", na_value=np.inf))
        r = spearmanr(df[c].values, key).statistic
        ck.report(f"spearman({c}, true time-to-next)", float(r), f"目標 ρ≈{rho}（常態分數的 Pearson＝ρ；Spearman 會略低）")
    doc = {"x_true_next_class": "真實的下一步（tool／human／end）", "x_true_think_ms": "真實的間隔（end＝null；不受觀察窗截斷）",
           "x_kv_bytes_llama": "context × 131072", "x_kv_bytes_qwen": "context × 57344"}
    for rho in params["signal_strength"]:
        doc[f"x_feat_rho_{int(round(rho * 100)):03d}"] = f"ρ={rho}：ρ·s＋√(1−ρ²)·ε"
    extra = dict(params={k: v for k, v in params.items()}, fit=dict(path=FIT_PATH, sha256=S.sha256_file(FIT_PATH),
                                                                      label=fj["label"], fitted_from=fj["fitted_from"],
                                                                      mixture=fj["mixture"], lognormal=fj["lognormal"]),
                 service_constants=dict(A=const["A"], B=const["B"], C=const["C"], run_id=const.get("run_id")),
                 kv_bytes_per_token=KV_BYTES, gen_stats=st, x_columns_doc=doc, tier="S",
                 content_sha256=content_hash(df), note="S 級：只測程式與機制，不能撐「學得到」的主張")
    ck.enforce()
    return df, ck, [dict(path=FIT_PATH, sha256=S.sha256_file(FIT_PATH)), dict(path=CONST_PATH, sha256=S.sha256_file(CONST_PATH))], extra


def content_hash(df):
    """內容雜湊（與 parquet 的編碼細節無關）：每欄轉成字串後的 sha256。"""
    import hashlib
    h = hashlib.sha256()
    for c in sorted(df.columns):
        h.update(c.encode())
        h.update(pd.util.hash_pandas_object(df[c].astype(str), index=False).values.tobytes())
    return h.hexdigest()


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "fit":
        fj = fit()
        fj.update(run_id=os.environ.get("RUN_ID", ""), ts=S.now_iso(), code=S.code_fingerprint())
        S.write_manifest(FIT_PATH, fj)
        S.log(json.dumps(dict(mixture=fj["mixture"], lognormal=fj["lognormal"]), ensure_ascii=False))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
