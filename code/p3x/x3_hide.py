"""x3_hide.py — X3：拿得到的提前量（x3_leads 的事件）× 需要的提前量（x3_restore 的 T(n,k) 表）→ 可藏起來的比例 h 的分布與判定。

定義見 docs/phase3_20261017/explore/X3_short_horizon_prefetch.md §2.1、§2.4：
  h＝(T_none − T(n,k)) ÷ (T_none − T_full)，k＝min(n, floor(L × 頻寬 ÷ 每 block 位元組))（整份預取）；
  只預取後段：k 再截在 n − b（b＝T_none 的會合點）。T_none − T_full < 1 ms 的事件不算（另報比例）。
  (a) L＝min(打字時間, 思考時間)；(b) L＝思考時間（上界）；(c) L＝工具執行時間（CacheWise）。

輸出（results/p3_explore/x3/）：
  x3_hide_summary.csv      原始長度；所有模型／層／精度／GPU；各訊號；整份 vs 後段
  x3_hide_mapped.csv       映射到 16／32／64 個 block（8K／16K／32K），只報告
  x3_hide_sens.csv         打字速度、TPOT、新 token 數的敏感度（Llama、GPU 閒）
  x3_lead_dist.csv         各 trace 各訊號的提前量分布（與層無關）
  x3_verdict.json          §2.2 的判定（照字面）
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import math
import os
import sys
from datetime import datetime, timezone

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
import p3_platform as PF  # noqa: E402  （P3_CONST：沒設＝MI300X；h200＝results/*_h200）
OUT = PF.explore_out("x3")
GiB = 1 << 30
NMAX = 256
# 文獻常數（出處見 X3 文件 §5）
WPM = {"desktop": 51.56,   # Dhakal et al. CHI'18, p.5（平均 51.56 WPM，SD 20.2）
       "mobile": 36.17}    # Palin et al. MobileHCI'19, p.6（平均 36.17 WPM，SD 13.22）
CHARS_PER_WORD = 5         # Dhakal et al. p.4：「one word consists of any five characters」
TEMPLATE_OV = 10           # 實測（x3_leads consts）
TPOTS = (0.020, 0.050, 0.100)
TPOT_MAIN = 0.050
HUMAN_TRACES = ("bailian_traceA", "mooncake_conversation", "mooncake_conversation_strict", "wildchat_en")   # §2.4-3
REAL_OK = {"bailian_traceA", "bailian_coder", "mooncake_conversation", "mooncake_toolagent", "wildchat_en",
           "mooncake_conversation_strict", "mooncake_toolagent_strict"}  # ≥1,000 session（原登記）
POST_REG = {"freeinference_tool", "freeinference_user"}   # README §4-1：登記之後才加，只進「含 FreeInference」版判定
TRACE_TOK = {"bailian": "qwen", "mooncake": "llama", "wildchat": "llama", "freeinference": "llama"}  # o200k 沒有實測，用 Llama 的字元/token 代替


def ts():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_tables(tdir):
    idx = json.load(open(os.path.join(tdir, "index.json")))
    tabs = {}
    for spec in idx["specs"]:
        m, tier, g, prec, gpu, variant = spec
        z = np.load(os.path.join(tdir, f"T_{m}_{tier}_{prec}_{gpu}_{variant}.npz"))
        tabs[(m, tier, prec, gpu)] = dict(T=z["T"], meet0=z["meet0"], nb=float(z["block_bytes"]), g=float(z["GiBps"]))
    return tabs, idx


def load_f_llama():
    sys.path.insert(0, HERE)
    from x3_restore import consts
    return np.array(consts("llama")["f_idle"])


def load_events(paths):
    evs = {}
    for p in paths:
        name = os.path.basename(p)[len("events_"):-len(".csv.gz")]
        d = pd.read_csv(p, compression="gzip")
        if len(d):
            evs[name] = d
    return evs


def leads(name, d, consts, f, wpm="desktop", tpot=TPOT_MAIN, typed_mode="raw_minus_ov"):
    """回傳 dict 訊號 → 提前量陣列（秒）。"""
    n_raw = np.ceil(d.n_old_tokens.values / 512).astype(int)
    n_blk = np.minimum(NMAX, n_raw)
    fpos = f[np.clip(n_blk, 0, NMAX)]
    cps = WPM[wpm] * CHARS_PER_WORD / 60.0
    out = {}
    if name.startswith("cachewise_tool") or name == "freeinference_tool":
        out["c_tool"] = d.lead_tool_s.values.astype(float)
        return out, n_blk
    if name.startswith("cachewise_human"):
        out["b_think"] = np.where(d.think_meas_s.values >= 0, d.think_meas_s.values, np.nan)   # 負的＝時間戳亂序，排除
        return out, n_blk
    if name == "freeinference_user":
        think = d.think_meas_s.values.astype(float)       # 實測：start_ms − 上一個 end_ms
        out["b_think"] = think
        cps_ = WPM[wpm] * CHARS_PER_WORD / 60.0
        typed = np.maximum(np.nan_to_num(pd.to_numeric(d.typed_tokens, errors="coerce").values, nan=0.0), 0.0)
        if typed_mode == "raw_minus_ov":
            pass   # role_tokens 只算訊息內容（tiktoken），沒有 chat template 的 token，不扣開銷
        typing = typed * consts["cpt_llama_English"] / cps_
        out["a_typing"] = np.minimum(typing, think)
        out["a_typing_uncapped"] = typing
        return out, n_blk
    prefill = d.gen_prefill_tokens.values / 512.0 * fpos
    think = d.gap_s.values - prefill - d.gen_tokens.values * tpot
    think = np.maximum(think, 0.0)
    out["b_think"] = think
    if name.startswith("wildchat"):
        typing = d.typed_chars.values / cps
    else:
        cpt = consts[f"cpt_{TRACE_TOK[name.split('_')[0]]}_English"]
        raw = d.new_tokens_raw.values.astype(float)
        typed = np.maximum(raw - TEMPLATE_OV, 0.0)
        if typed_mode == "nontext_zero" and "type" in d:
            typed = np.where(d["type"].fillna("").values == "text", typed, 0.0)
        elif typed_mode == "raw":
            typed = np.maximum(raw, 0.0)
        typing = typed * cpt / cps
    out["a_typing"] = np.minimum(typing, think)
    out["a_typing_uncapped"] = typing
    return out, n_blk


def hide_stats(n_blk, L, tab, scope, over=None):
    """over：要還原的長度超過 256 個 block（Llama 上限 128K）的事件；給了就把它們的 h 設成 0（下界）。"""
    T, meet0, nb, g = tab["T"], tab["meet0"], tab["nb"], tab["g"]
    ok = (n_blk >= 1) & np.isfinite(L)
    n = n_blk[ok]
    Lk = L[ok]
    t_none = T[n, 0]
    t_full = T[n, n]
    pen = t_none - t_full
    has = pen >= 1e-3
    k = np.minimum(n, np.floor(Lk * g * GiB / nb).astype(np.int64))
    if scope == "tail":
        k = np.minimum(k, n - meet0[n])
    t_L = T[n, k]
    h = np.where(has, (t_none - t_L) / np.where(has, pen, 1.0), np.nan)
    if over is not None:
        h = np.where(over[ok], 0.0, h)
        t_L = np.where(over[ok], t_none, t_L)
    hv = h[has]
    lead_need = n * nb / (g * GiB)
    lead_tail = (n - meet0[n]) * nb / (g * GiB)
    r = dict(n_events=int(ok.sum()), n_with_penalty=int(has.sum()), frac_no_penalty=round(1 - has.mean(), 4) if ok.any() else np.nan)
    if hv.size:
        r.update({f"h_p{q}": round(float(np.percentile(hv, q)), 4) for q in (10, 25, 50, 75, 90)})
        r.update(h_mean=round(float(hv.mean()), 4), frac_h_ge_0p5=round(float((hv >= 0.5).mean()), 4),
                 frac_h_ge_0p99=round(float((hv >= 0.99).mean()), 4),
                 frac_h_le_0=round(float((hv <= 0).mean()), 4),
                 time_weighted_hidden=round(float((t_none - t_L)[has].sum() / pen[has].sum()), 4),
                 penalty_ms_median=round(float(np.median(pen[has]) * 1e3), 2),
                 penalty_ms_mean=round(float(pen[has].mean() * 1e3), 2),
                 penalty_ms_p90=round(float(np.percentile(pen[has], 90) * 1e3), 2),
                 lead_need_full_s_median=round(float(np.median(lead_need[has])), 4),
                 lead_need_tail_s_median=round(float(np.median(lead_tail[has])), 4),
                 lead_avail_s_median=round(float(np.median(Lk[has])), 4),
                 frac_lead_ge_need_full=round(float((Lk[has] >= lead_need[has]).mean()), 4),
                 frac_lead_ge_need_tail=round(float((Lk[has] >= lead_tail[has]).mean()), 4),
                 restore_ms_mean_none=round(float(t_none[has].mean() * 1e3), 2),
                 restore_ms_mean_hint=round(float(t_L[has].mean() * 1e3), 2),
                 restore_ms_mean_full=round(float(t_full[has].mean() * 1e3), 2))
    return r


def ttft_stats(n_blk, L, tab, newtok, f):
    """回來請求的完整 TTFT（還原＋新一輪 prefill，prefill≈新 token÷512×f(n)）平均：不預取／用提示預取／完美預取。全部事件（含沒東西可藏的）。"""
    T, nb, g = tab["T"], tab["nb"], tab["g"]
    ok = (n_blk >= 1) & np.isfinite(L)
    n, Lk = n_blk[ok], L[ok]
    k = np.minimum(n, np.floor(Lk * g * GiB / nb).astype(np.int64))
    pre = np.maximum(newtok[ok], 0) / 512.0 * f[n]
    return dict(ttft_ms_mean_none=round(float((T[n, 0] + pre).mean() * 1e3), 2),
                ttft_ms_mean_hint=round(float((T[n, k] + pre).mean() * 1e3), 2),
                ttft_ms_mean_full=round(float((T[n, n] + pre).mean() * 1e3), 2),
                ttft_ms_median_none=round(float(np.median(T[n, 0] + pre) * 1e3), 2),
                ttft_ms_median_hint=round(float(np.median(T[n, k] + pre) * 1e3), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--tables", required=True, help="x3_restore tables run 目錄下的 tables/")
    ap.add_argument("--events", nargs="+", required=True, help="events_*.csv.gz")
    a = ap.parse_args()
    tabs, idx = load_tables(a.tables)
    tiers = idx["tiers"]
    consts = json.load(open(os.path.join(OUT, "x3_typing_consts.json")))
    f = load_f_llama()
    evs = load_events(a.events)
    t0 = ts()
    rows, mapped, sens, ldist = [], [], [], []
    for name, d in evs.items():
        nsess = int(d.session.nunique())
        L, n_blk = leads(name, d, consts, f)
        newtok = pd.to_numeric(d.typed_tokens if name.startswith(("wildchat", "freeinference_user")) else d.new_tokens_raw,
                               errors="coerce").values.astype(float) \
            if "new_tokens_raw" in d and not name.startswith(("cachewise", "freeinference_tool")) else np.zeros(len(d))
        newtok = np.nan_to_num(newtok)
        over = np.ceil(d.n_old_tokens.values / 512) > NMAX
        clipped = float(np.mean(over))
        for sig, x in L.items():
            xv = x[np.isfinite(x)]
            ldist.append(dict(trace=name, signal=sig, n=len(xv), sessions=nsess,
                              **{f"p{q}": round(float(np.percentile(xv, q)), 4) for q in (5, 10, 25, 50, 75, 90, 95)},
                              mean=round(float(xv.mean()), 4), frac_zero=round(float((xv <= 0).mean()), 4),
                              n_old_blocks_median=float(np.median(n_blk)), n_old_blocks_p90=float(np.percentile(n_blk, 90)),
                              frac_n_clipped_256=round(clipped, 5), frac_n_zero=round(float((n_blk == 0).mean()), 4)))
        for (m, tier, prec, gpu), tab in tabs.items():
            g = tab["g"]
            for sig, x in L.items():
                if sig == "a_typing_uncapped":
                    continue
                for scope in ("full", "tail"):
                    for clip in (("clip", "over0") if over.any() else ("clip",)):
                        r = dict(trace=name, sessions=nsess, real_trace=int(name in REAL_OK), post_reg=int(name in POST_REG),
                                 human_typing=int(name in HUMAN_TRACES), length="orig", model=m, tier=tier, GiBps=round(g, 4),
                                 in_band=int(0.33 <= g <= 4.0), prec=prec, gpu=gpu, signal=sig, scope=scope, wpm="desktop",
                                 tpot_ms=50, typed="raw_minus_ov", clip=clip, frac_over_128K=round(float(over.mean()), 4))
                        r.update(hide_stats(n_blk, x, tab, scope, over if clip == "over0" else None))
                        if scope == "full" and clip == "clip":
                            r.update(ttft_stats(n_blk, x, tab, newtok, f))
                        rows.append(r)
                # 映射長度（只 Llama、GPU 閒、整份）
                if m == "llama" and gpu == "idle":
                    for nm in (16, 32, 64):
                        nn = np.where(n_blk >= 1, nm, 0)
                        r = dict(trace=name, sessions=nsess, length=f"mapped_{nm * 512 // 1024}K", model=m, tier=tier,
                                 GiBps=round(g, 4), in_band=int(0.33 <= g <= 4.0), prec=prec, gpu=gpu, signal=sig, scope="full")
                        r.update(hide_stats(nn, x, tab, "full"))
                        mapped.append(r)
        # 敏感度（Llama、GPU 閒、整份、原始長度）
        if not name.startswith(("cachewise", "freeinference_tool")):
            for wpm in WPM:
                for tpot in TPOTS:
                    for tm in ("raw_minus_ov", "nontext_zero", "raw"):
                        if name.startswith(("wildchat", "freeinference")) and tm != "raw_minus_ov":
                            continue
                        L2, _ = leads(name, d, consts, f, wpm=wpm, tpot=tpot, typed_mode=tm)
                        for (m, tier, prec, gpu), tab in tabs.items():
                            if m != "llama" or gpu != "idle":
                                continue
                            for sig in ("a_typing", "b_think"):
                                r = dict(trace=name, sessions=nsess, length="orig", model=m, tier=tier, GiBps=round(tab["g"], 4),
                                         in_band=int(0.33 <= tab["g"] <= 4.0), prec=prec, gpu=gpu, signal=sig, scope="full",
                                         wpm=wpm, tpot_ms=int(tpot * 1e3), typed=tm)
                                r.update(hide_stats(n_blk, L2[sig], tab, "full"))
                                sens.append(r)
        print("done", name, flush=True)

    def dump(rs, fn):
        df = pd.DataFrame(rs)
        df.insert(0, "ts", t0)
        df.insert(0, "run_id", a.run_id)
        p = os.path.join(OUT, fn)
        df.to_csv(p, index=False)
        print("wrote", p, len(df))
        return df

    S = dump(rows, "x3_hide_summary.csv")
    dump(mapped, "x3_hide_mapped.csv")
    SE = dump(sens, "x3_hide_sens.csv")
    dump(ldist, "x3_lead_dist.csv")

    # ---- 判定（§2.2、§2.4，照字面）；兩個版本：原登記（不含 FreeInference）、含 FreeInference（README §4-1）
    def verdict(include_fi):
        base = S[(S.model == "llama") & (S.gpu == "idle") & (S.prec == "int4") & (S.length == "orig") & (S.scope == "full")
                 & (S.in_band == 1)]
        a_ok = base[(base.signal == "a_typing") & base.trace.isin(HUMAN_TRACES) & (base.real_trace == 1) & (base["clip"] == "clip")]
        parts = [a_ok]
        if include_fi:
            # FreeInference：(a)＝使用者發起的下一輪、(c)＝工具；超過 128K 的事件 h＝0（下界，over0）
            parts.append(base[(base.trace == "freeinference_user") & (base.signal == "a_typing") & (base["clip"] == "over0")])
            parts.append(base[(base.trace == "freeinference_tool") & (base.signal == "c_tool") & (base["clip"] == "over0")])
        main = pd.concat(parts)
        best = main.sort_values("h_p50", ascending=False).head(10)
        interesting = bool((main.h_p50 >= 0.5).any())
        allow = S.trace.isin(REAL_OK | (POST_REG if include_fi else set()))
        ac = S[allow & (S.length == "orig") & (S.scope == "full") & (S.signal.isin(["a_typing", "c_tool"]))]
        ac_se = SE[SE.trace.isin(REAL_OK | (POST_REG if include_fi else set())) & (SE.signal == "a_typing")]
        max_any = float(np.nanmax(np.r_[ac.h_p50.values, ac_se.h_p50.values]))
        dead = max_any < 0.2
        v = "有看頭" if interesting else ("死路" if dead else "可能")
        return dict(verdict=v, interesting_cells=int((main.h_p50 >= 0.5).sum()), main_cells=int(len(main)),
                    main_best=best[["trace", "signal", "tier", "GiBps", "h_p50", "frac_h_ge_0p5", "time_weighted_hidden",
                                    "n_with_penalty", "sessions"]].to_dict("records"),
                    main_worst=main.sort_values("h_p50").head(8)[["trace", "signal", "tier", "GiBps", "h_p50"]].to_dict("records"),
                    max_h_p50_over_all_a_c=max_any, dead=dead)
    out = dict(run_id=a.run_id, ts=ts(), rule="§2.2/§2.4 of X3_short_horizon_prefetch.md",
               v1_original_registration=verdict(False), v2_with_freeinference=verdict(True))
    json.dump(out, open(os.path.join(OUT, "x3_verdict.json"), "w"), indent=1, ensure_ascii=False, default=float)
    print(json.dumps(out, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
