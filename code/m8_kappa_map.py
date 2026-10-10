"""m8_kappa_map.py — D3「κ 地圖」：用事先算得出的量，預測每一格哪種策略家族會贏。

判準寫死在 docs/research_20261010_directions/D3_kappa_map.md §2（2026-10-10T05:37:39Z）。
本程式只讀既有 CSV（不碰 GPU、不改任何既有檔），輸出 results/m8_directions/d3_*.csv。

用法（走 m7run，讓每個數字都有 run_id）：
  HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= m7run d3-kappa-map python code/m8_kappa_map.py --out results/m8_directions

規則（細節見文件 §2.3）：
  R1   讀取端 Cake 有沒有用（0 參數；演算法 1）
  R1b  R1 + 一個固定同步開銷 c_sync（只用 A0 L=8K、32K 擬合）
  R1t  3090：從 SSD 載入整段 vs 整段重算（裝置頻寬）
  R2   任務規則 (b)：b/n<0.3 或 free ⇒ 寫入時領先 ≤5%
  R2'  寫入時 vs 延後雙胞胎
  R3   寫穿 vs 延後
  R4a/R4b  D＝P·(T_out−T_in)/(T_in+t_new) 判斷「放置重不重要」
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R7 = os.path.join(REPO, "results", "m7_write_policy_mi300x")
RX = os.path.join(REPO, "results", "m7_explore_mi300x")
MiB = 1 << 20
GiB = 1 << 30
TIE = 0.05            # 判準：差距 ≥5% 才算贏
RUN_ID = os.environ.get("RUN_ID", "NO_RUN_ID")
TS = datetime.now(timezone.utc).isoformat(timespec="seconds")

WT = ["S1", "S2b"]
LZ = ["S4", "S4+", "S4+P", "S4L", "S4B", "S4W", "S4C"]
WTP = ["S5", "S5L", "S5P", "S5c", "S5s"]
PLACEMENT = WT + LZ + ["S5", "S5L", "S5P", "S5c"]       # R4 的「放置策略」：排除 R0、S0、S3、S2a、S5s
TWIN = {"S5": ["S4+"], "S5P": ["S4+P"], "S5L": ["S4B", "S4L"], "S5c": ["S4+"]}
SCHED = {"chat": [16, 16, 16, 16], "doc": [64, 1, 1, 1]}
N_RET = {"chat": 32, "doc": 65}                          # 回來請求的平均歷史 chunk 數（§2.3 R2）


# ---------------------------------------------------------------------------
# 事先就知道的量
def load_f(path, scale=1.0):
    d = pd.read_csv(path)
    d = d[(d["item"] == "f_chunk") & (d["gpu_state"] == "idle")]
    med = d.groupby("chunk_idx")["ms"].median().sort_index()
    assert list(med.index) == list(range(len(med))), "f(i) 的 chunk_idx 不連續"
    return [x / 1e3 * scale for x in med.values]


F_LLAMA = load_f(os.path.join(R7, "calib_c1.csv"))
F_LONGA = load_f(os.path.join(RX, "calib_c1_longalpaca7b.csv"))
TP = json.load(open(os.path.join(R7, "tier_params.json")))
CHUNK = {"llama31_8b": 64 * MiB, "longalpaca7b": 256 * MiB}


def ell(bytes_, gibps):
    return bytes_ / (gibps * GiB)


def alg1(n, f, l, c_sync=0.0):
    """演算法 1：回傳 (T_comp, T_load, T_cake, m*)。T_cake＝min_m max(Σf[:m], (n−m)·ℓ)（+c_sync）。"""
    T_comp = sum(f[:n])
    T_load = n * l
    best, best_m, R = None, 0, 0.0
    for m in range(0, n + 1):
        if m > 0:
            R += f[m - 1]
        T = max(R, (n - m) * l)
        if best is None or T < best:
            best, best_m = T, m
    return T_comp, T_load, best + c_sync, best_m


def bclass(speedup):
    return "useful" if speedup >= 1 + TIE else "not_useful"


def winner3(tc, tl, tk):
    return ["compute_only", "load_only", "cake"][int(np.argmin([tc, tl, tk]))]


# ---------------------------------------------------------------------------
# 讀取端：A0、c7_anchor
def read_cells():
    a0 = pd.read_csv(os.path.join(R7, "a0.csv"))
    a0 = a0[a0["contaminated_gpu"].astype(str) != "True"]
    med = a0.groupby(["L", "bw_name", "read_GiBps", "mode"]).agg(t_restore=("t_restore", "median"),
                                                                 ttft=("ttft", "median"), n=("rep", "size"),
                                                                 meet=("meet", "median")).reset_index()
    comp = med[med["mode"] == "compute_only"].set_index("L")
    rows = []
    for (L, bw, g), grp in med[med["mode"] != "compute_only"].groupby(["L", "bw_name", "read_GiBps"]):
        g_ = grp.set_index("mode")
        n = L // 512
        tc, tl, tk = comp.loc[L, "t_restore"], g_.loc["load_only", "t_restore"], g_.loc["cake", "t_restore"]
        rows.append(dict(src="A0", run_id_src=a0["run_id"].iloc[0], device="throttled_ssd", io="sim", L=L, bw=bw,
                         gibps=g, n=n, ell_ms=ell(64 * MiB, g) * 1e3,
                         kappa_bar=ell(64 * MiB, g) / np.mean(F_LLAMA[:n]),
                         meas_comp=tc, meas_load=tl, meas_cake=tk,
                         meas_ttft_comp=comp.loc[L, "ttft"], meas_ttft_load=g_.loc["load_only", "ttft"],
                         meas_ttft_cake=g_.loc["cake", "ttft"], meas_meet=g_.loc["cake", "meet"]))
    # c7_anchor：真實裝置（io=real）與限速器（io=sim）
    c7 = pd.read_csv(os.path.join(R7, "c7_anchor.csv"))
    c7 = c7[c7["contaminated_gpu"].astype(str) != "True"]
    c0 = pd.read_csv(os.path.join(R7, "calib_c0.csv"))
    h2d = c0[(c0["item"] == "H2D_pinned_chunk_into_kv_view") & (c0["rep"] > 0)]["value"].astype(float).median()
    gib_real = {"cpu": h2d, "local": TP["local"]["measured_min_read_MiBps"] * MiB / GiB,
                "nfs": TP["nfs"]["measured_min_read_MiBps"] * MiB / GiB}
    gib_sim = {"cpu": TP["cpu"]["read_GiBps"], "local": TP["local"]["read_GiBps"], "nfs": TP["nfs"]["read_GiBps"]}
    m7 = c7.groupby(["device", "io", "L", "mode"]).agg(t_restore=("t_restore", "median"),
                                                       ttft=("ttft", "median")).reset_index()
    for (dev, io_, L), grp in m7.groupby(["device", "io", "L"]):
        g_ = grp.set_index("mode")
        n = L // 512
        g = gib_real[dev] if io_ == "real" else gib_sim[dev]
        rows.append(dict(src="c7_anchor", run_id_src=c7["run_id"].iloc[0], device=dev, io=io_, L=L, bw=f"{dev}-{io_}",
                         gibps=g, n=n, ell_ms=ell(64 * MiB, g) * 1e3, kappa_bar=ell(64 * MiB, g) / np.mean(F_LLAMA[:n]),
                         meas_comp=comp.loc[L, "t_restore"], meas_load=g_.loc["load_only", "t_restore"],
                         meas_cake=g_.loc["cake", "t_restore"], meas_ttft_comp=comp.loc[L, "ttft"],
                         meas_ttft_load=g_.loc["load_only", "ttft"], meas_ttft_cake=g_.loc["cake", "ttft"],
                         meas_meet=np.nan))
    d = pd.DataFrame(rows)
    d["meas_speedup"] = d[["meas_comp", "meas_load"]].min(axis=1) / d["meas_cake"]
    d["meas_class"] = d["meas_speedup"].apply(bclass)
    d["meas_winner"] = [winner3(a, b, c) for a, b, c in zip(d.meas_comp, d.meas_load, d.meas_cake)]
    return d


def apply_r1(d, c_sync, tag):
    out = []
    for _, r in d.iterrows():
        tc, tl, tk, m = alg1(int(r.n), F_LLAMA, r.ell_ms / 1e3, c_sync)
        out.append((tc, tl, tk, m))
    d[f"{tag}_comp"], d[f"{tag}_load"], d[f"{tag}_cake"], d[f"{tag}_meet"] = zip(*out)
    d[f"{tag}_speedup"] = d[[f"{tag}_comp", f"{tag}_load"]].min(axis=1) / d[f"{tag}_cake"]
    d[f"{tag}_class"] = d[f"{tag}_speedup"].apply(bclass)
    d[f"{tag}_winner"] = [winner3(a, b, c) for a, b, c in zip(d[f"{tag}_comp"], d[f"{tag}_load"], d[f"{tag}_cake"])]
    d[f"{tag}_hit"] = d[f"{tag}_class"] == d["meas_class"]
    d[f"{tag}_err"] = (d[f"{tag}_speedup"] - d["meas_speedup"]) / d["meas_speedup"]
    return d


def fit_csync(d):
    """只用 A0 的 L=8K、32K：類別命中最多，同分取 |誤差| 中位數最小。"""
    fit = d[(d.src == "A0") & (d.L.isin([8192, 32768]))].copy()
    best = None
    for c in np.arange(0.0, 0.0305, 0.0005):
        x = apply_r1(fit.copy(), c, "t")
        key = (-int(x["t_hit"].sum()), float(x["t_err"].abs().median()))
        if best is None or key < best[0]:
            best = (key, c)
    return best[1]


# ---------------------------------------------------------------------------
# 3090（git 歷史，commit 52803bc^）
def git_csv(path):
    try:
        s = subprocess.run(["git", "-C", REPO, "show", f"52803bc^:{path}"], capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        print(f"[3090] git show 失敗：{path}\n{e.stderr}", file=sys.stderr)
        return None
    return pd.read_csv(io.StringIO(s))


def r1t_3090():
    bw = git_csv("results/m2_harness/disk_bw.csv")
    rp = git_csv("results/m2_harness/recompute_position.csv")
    rows = []
    if bw is None or rp is None:
        return pd.DataFrame(rows)
    rp = rp[rp["model_key"] == "llama"]
    med = rp.groupby("cached_prefix_tokens")["ttft_ms"].median()
    b, a = np.polyfit(med.index.values.astype(float), med.values, 1)   # ttft_ms = a + b·P（2048-token block）
    ctx, blk = 16384, 2048
    sum_f = sum(a + b * p for p in range(0, ctx, blk)) / 1e3
    kvb = 131072 * ctx                                                  # Llama-3.1-8B BF16：128 KiB／token
    for name, dev, path in [("sata", "/dev/sdh", "results/m2_harness/retrieval_cost_sata.csv"),
                            ("nvme", "/dev/nvme0n1p3", "results/m2_harness/retrieval_cost_nvme.csv")]:
        rc = git_csv(path)
        if rc is None:
            continue
        mibps = bw[bw["device"] == dev]["read_mibps"].median()
        t_load = kvb / (mibps * MiB)
        w = rc[(rc["round"] == "warm") & (rc["model_key"] == "llama")]
        ssd = w[w["tier"] == "ssd"]["ttft_ms"].median() / 1e3
        drop = w[w["tier"] == "drop"]["ttft_ms"].median() / 1e3
        cpu = w[w["tier"] == "cpu"]["ttft_ms"].median() / 1e3
        pred = "ssd" if t_load < sum_f else "drop"
        meas = "ssd" if ssd < drop else "drop"
        rows.append(dict(platform="RTX3090", dev=name, run_id_src=rc["run_id"].iloc[0], run_id_bw=bw["run_id"].iloc[0],
                         run_id_f=rp["run_id"].iloc[0], device_read_MiBps=mibps, pred_t_load_s=t_load,
                         pred_sum_f_s=sum_f, f_fit_a_ms=a, f_fit_b_ms_per_tok=b, meas_warm_ssd_s=ssd,
                         meas_warm_drop_s=drop, meas_warm_cpu_s=cpu,
                         eff_ssd_MiBps_upper=kvb / MiB / ssd, pred_winner=pred, meas_winner=meas, hit=pred == meas,
                         host_contention=";".join(sorted(rc["host_contention"].astype(str).unique()))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 寫入端：把每一格整理成「策略 → 回來請求 TTFT 中位數」
def gpu_cells():
    cells = []

    def add(df, base, keys):
        df = df[(df["round"] >= 2) & df["ttft"].notna()]
        df = df[df["contaminated_gpu"].astype(str) != "True"]
        for k, g in df.groupby(keys):
            k = k if isinstance(k, tuple) else (k,)
            meta = dict(base, **dict(zip(keys, k)))
            med = g.groupby("strategy")["ttft"].median().to_dict()
            meta["run_ids"] = ";".join(sorted(g["run_id"].unique()))
            meta["n_req"] = int(g.groupby("strategy").size().min())
            cells.append((meta, med))

    k1 = ["io_model", "ssd_dev", "gap_s", "ssd_frac", "wl_seed", "cpu_frac"]
    add(pd.read_csv(os.path.join(R7, "b_share.csv")), dict(set="P1", model="llama31_8b", cpu_gibps=TP["cpu"]["read_GiBps"],
                                                           release="free", workload="chat", f_scale=1.0), k1)
    add(pd.read_csv(os.path.join(R7, "b_share_cpu3.69.csv")), dict(set="P1", model="llama31_8b", cpu_gibps=3.69,
                                                                   release="free", workload="chat", f_scale=1.0), k1)
    add(pd.read_csv(os.path.join(R7, "b_fifo.csv")), dict(set="P1", model="llama31_8b", cpu_gibps=TP["cpu"]["read_GiBps"],
                                                          release="free", workload="chat", f_scale=1.0), k1)
    k2 = k1 + ["cpu_gibps", "release", "workload"]
    add(pd.read_csv(os.path.join(R7, "b2_share.csv")), dict(set="A08", model="llama31_8b", f_scale=1.0), k2)
    add(pd.read_csv(os.path.join(RX, "r1_gpu_llama_doc_free_local.csv")), dict(set="R1", model="llama31_8b", f_scale=1.0), k2)
    return cells


def sim_cells():
    cells = []
    s = pd.read_csv(os.path.join(R7, "sim_sweep.csv"))
    for k, g in s.groupby(["cpu_gibps", "release", "workload", "ssd_dev", "f_scale", "wl_seed", "cpu_frac"]):
        meta = dict(set="SIM08", model="llama31_8b", io_model="share", gap_s=0.0, ssd_frac=1.0,
                    **dict(zip(["cpu_gibps", "release", "workload", "ssd_dev", "f_scale", "wl_seed", "cpu_frac"], k)),
                    run_ids=";".join(sorted(g["run_id"].unique())), n_req=np.nan)
        cells.append((meta, g.set_index("strategy")["median"].to_dict()))
    for model in ["llama31_8b", "longalpaca7b"]:
        r = pd.read_csv(os.path.join(RX, f"r1_sim_{model}.csv"))
        for k, g in r.groupby(["cpu_gibps", "release", "workload", "ssd_dev", "wl_seed", "cpu_frac"]):
            meta = dict(set=f"SIMR1_{model}", model=model, io_model="share", gap_s=0.0, ssd_frac=1.0, f_scale=1.0,
                        **dict(zip(["cpu_gibps", "release", "workload", "ssd_dev", "wl_seed", "cpu_frac"], k)),
                        run_ids=";".join(sorted(g["run_id"].unique())), n_req=np.nan)
            cells.append((meta, g.set_index("strategy")["median"].to_dict()))
    return cells


def fam_best(med, fam):
    xs = [(med[s], s) for s in fam if s in med and pd.notna(med[s])]
    return min(xs) if xs else (np.nan, None)


def gain(rival, me):
    """(對手 − 自己)／對手；正＝自己較快。"""
    return (rival - me) / rival


def features(meta):
    f0 = F_LLAMA if meta["model"] == "llama31_8b" else F_LONGA
    f = [x * meta["f_scale"] for x in f0]
    cb = CHUNK[meta["model"]]
    wl = meta["workload"]
    n = N_RET[wl]
    l_cpu = ell(cb, meta["cpu_gibps"])
    l_ssd = ell(cb, TP[meta["ssd_dev"]]["read_GiBps"])
    _, _, T_in, b_cpu = alg1(n, f, l_cpu)
    _, _, T_out, _ = alg1(n, f, l_ssd)
    t_new = sum(f[n:n + 16]) if wl == "chat" else f[n]
    P = 1.0 - meta["cpu_frac"]
    D = P * (T_out - T_in) / (T_in + t_new)
    return dict(n_ret=n, ell_cpu_ms=l_cpu * 1e3, ell_ssd_ms=l_ssd * 1e3, b_cpu=b_cpu, b_over_n=b_cpu / n,
                T_in=T_in, T_out=T_out, t_new=t_new, P=P, D=D)


def evaluate(cell):
    meta, med = cell
    r = dict(meta)
    r.update(features(meta))
    wt, wt_s = fam_best(med, WT)
    lz, lz_s = fam_best(med, LZ)
    wtp, wtp_s = fam_best(med, WTP)
    nwt, nwt_s = fam_best(med, WT + LZ)
    r.update(best_WT=wt_s, t_WT=wt, best_LZ=lz_s, t_LZ=lz, best_WTP=wtp_s, t_WTP=wtp, best_NWT=nwt_s, t_NWT=nwt)
    # 標籤
    r["lead_WTP_over_NWT"] = gain(nwt, wtp) if pd.notna(wtp) and pd.notna(nwt) else np.nan
    pl = [med[s] for s in PLACEMENT if s in med and pd.notna(med[s])]
    r["n_placement"] = len(pl)
    r["spread"] = max(pl) / min(pl) - 1 if len(pl) >= 2 else np.nan
    r["matters"] = bool(r["spread"] >= TIE) if pd.notna(r["spread"]) else None
    bestall = min(pl) if pl else np.nan
    r["gain_best_vs_S1"] = gain(med["S1"], bestall) if "S1" in med else np.nan
    if pd.notna(wt) and pd.notna(lz):
        g = gain(lz, wt)
        r["R3_meas"] = "WT" if g >= TIE else ("LZ" if gain(wt, lz) >= TIE else "tie")
        r["R3_pred"] = "WT" if meta["release"] == "hold" else "tie"
        r["R3_hit"] = r["R3_meas"] == r["R3_pred"]
    # R2（任務規則 b）
    if pd.notna(r["lead_WTP_over_NWT"]):
        r["R2_pred"] = "may_exceed" if (r["b_over_n"] >= 0.3 and meta["release"] == "hold") else "le5"
        r["R2_meas"] = "exceed" if r["lead_WTP_over_NWT"] > TIE else "le5"
    # R2' 寫入時 vs 延後雙胞胎（配對）
    pairs = []
    for s, twins in TWIN.items():
        if s not in med:
            continue
        tw = next((t for t in twins if t in med), None)
        if tw is None:
            continue
        g_tw = gain(med[tw], med[s])
        if meta["release"] == "free":
            hit = abs(g_tw) < TIE
        else:
            g_wt = gain(wt, med[s]) if pd.notna(wt) else np.nan
            hit = (g_tw >= TIE) and (g_wt < TIE)
        pairs.append(dict(strategy=s, twin=tw, gain_vs_twin=g_tw, hit=bool(hit)))
    r["R2p_pairs"] = json.dumps(pairs)
    r["R2p_n"] = len(pairs)
    r["R2p_hits"] = sum(p["hit"] for p in pairs)
    for s in sorted(set(PLACEMENT + WTP + ["R0", "S0", "S3"])):
        r[f"t_{s}"] = med.get(s, np.nan)
    return r


# ---------------------------------------------------------------------------
def score(name, dataset, y_true, y_pred, extra=""):
    y_true, y_pred = list(y_true), list(y_pred)
    n = len(y_true)
    hits = sum(a == b for a, b in zip(y_true, y_pred))
    if n:
        maj = pd.Series(y_true).value_counts().idxmax()
        base = sum(a == maj for a in y_true)
    else:
        maj, base = None, 0
    return dict(run_id=RUN_ID, ts=TS, rule=name, dataset=dataset, n=n, hits=hits, hit_rate=hits / n if n else np.nan,
                baseline_majority=maj, baseline_hits=base, baseline_rate=base / n if n else np.nan, note=extra)


def balanced_acc(y, yhat):
    y, yhat = np.array(y, bool), np.array(yhat, bool)
    tpr = (yhat & y).sum() / max(1, y.sum())
    tnr = (~yhat & ~y).sum() / max(1, (~y).sum())
    return 0.5 * (tpr + tnr)


def posthoc(W, rd, out):
    """〔事後探索〕看過預先登記規則的結果之後才加的分析，不進判定（文件 §7 列為偏離）。
    1. R4 改用 |D|（CPU 層比 SSD 慢時 D 會變負）
    2. 換一個標籤：「最好的策略比業界預設 S1 快 ≥5%」
    3. 同一組物理條件、只換 workload seed，標籤會不會翻：物理常數能解釋的上限
    4. R1 改用 TTFT（含 t_new）當標籤"""
    rows = []
    W = W.copy()
    W["absD"] = W["D"].abs()
    W["beats_S1"] = (W["gain_best_vs_S1"] >= TIE).where(W["gain_best_vs_S1"].notna())
    fit = W[(W["set"] == "P1") & W["matters"].notna()]
    for label in ["matters", "beats_S1"]:
        f_ = fit[fit[label].notna()]
        Ds = np.sort(f_["absD"].unique())
        cands = [0.0] + list((Ds[:-1] + Ds[1:]) / 2) + [Ds[-1] + 1e-9]
        th = max(cands, key=lambda t: (balanced_acc(f_[label].astype(bool), f_["absD"] >= t), -t))
        for nm, m in [("GPU P1", W["set"] == "P1"), ("GPU A08", W["set"] == "A08"), ("GPU R1", W["set"] == "R1"),
                      ("SIM08", W["set"] == "SIM08"), ("SIMR1 llama", W["set"] == "SIMR1_llama31_8b"),
                      ("SIMR1 longalpaca", W["set"] == "SIMR1_longalpaca7b")]:
            sub = W[m & W[label].notna()]
            y = sub[label].astype(bool)
            r = score(f"PH |D|>=th → {label}", nm, y, sub["absD"] >= th, f"th={th:.4f}(fit P1)")
            r["balanced_acc"] = balanced_acc(y, sub["absD"] >= th)
            rows.append(r)
            r2 = score(f"PH ref cpu_frac<1 → {label}", nm, y, sub["cpu_frac"] < 1)
            r2["balanced_acc"] = balanced_acc(y, sub["cpu_frac"] < 1)
            rows.append(r2)
    # 3. 只換 seed，標籤會不會翻
    phys = ["set", "model", "io_model", "ssd_dev", "gap_s", "ssd_frac", "cpu_frac", "cpu_gibps", "release",
            "workload", "f_scale"]
    for label in ["matters", "beats_S1", "R3_meas", "best_NWT", "R2_meas"]:
        for st, sub in W[W[label].notna()].groupby("set"):
            g = sub.groupby(phys, dropna=False)[label].agg(lambda x: x.astype(str).nunique())
            n_multi = sub.groupby(phys, dropna=False).size()
            g = g[n_multi >= 2]
            rows.append(dict(run_id=RUN_ID, ts=TS, rule=f"PH seed-flip {label}", dataset=st, n=len(g),
                             hits=int((g == 1).sum()), hit_rate=(g == 1).mean() if len(g) else np.nan,
                             baseline_majority="", baseline_hits=np.nan, baseline_rate=np.nan,
                             note="hits＝同一組物理條件、所有 seed 標籤相同的組數；1−hit_rate＝物理常數單獨無法決定的比例"))
    # 4. R1 改用 TTFT
    A0 = rd[rd.src == "A0"].copy()
    A0["meas_speedup_ttft"] = A0[["meas_ttft_comp", "meas_ttft_load"]].min(axis=1) / A0["meas_ttft_cake"]
    A0["meas_class_ttft"] = A0["meas_speedup_ttft"].apply(bclass)
    rows.append(score("PH R1 label=TTFT", "A0(all 28)", A0.meas_class_ttft, A0.R1_class,
                      f"cells where TTFT says not useful: {';'.join(f'{l}/{b}' for l, b in zip(A0[A0.meas_class_ttft=='not_useful'].L, A0[A0.meas_class_ttft=='not_useful'].bw))}"))
    pd.DataFrame(rows).to_csv(os.path.join(out, "d3_posthoc.csv"), index=False)
    print("\n〔事後探索〕")
    print(pd.DataFrame(rows)[["rule", "dataset", "n", "hits", "hit_rate", "baseline_rate", "note"]].to_string())


def b100(out):
    """§2.1 附加測試：第一階段 B、容量 100%（歷史全在 CPU 35.4 GiB/s），R1 預測 Cake 對只載（R0）有沒有用。
    實測標籤取 d3_write_cells.csv（同一 run 的前一步）裡 P1、nfs、gap 0、cpu_frac 1.0 那格的 t_R0 與 Cake 策略最好的中位數。"""
    W = pd.read_csv(os.path.join(out, "d3_write_cells.csv"))
    c = W[(W["set"] == "P1") & (W["cpu_frac"] == 1.0)].iloc[0]
    cake = min(c[f"t_{s}"] for s in ["S1", "S2b", "S4", "S4+", "S4+P", "S5"] if pd.notna(c[f"t_{s}"]))
    l = ell(64 * MiB, TP["cpu"]["read_GiBps"])
    rows = []
    for n in (16, 32, 48):
        tc, tl, tk, m = alg1(n, F_LLAMA, l)
        rows.append(dict(run_id=RUN_ID, ts=TS, n=n, pred_T_load=tl, pred_T_cake=tk, pred_speedup=min(tc, tl) / tk,
                         pred_class=bclass(min(tc, tl) / tk), meas_R0_ttft=c["t_R0"], meas_best_cake_ttft=cake,
                         meas_speedup_ttft=c["t_R0"] / cake, meas_class=bclass(c["t_R0"] / cake), src_runs=c["run_ids"]))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(out, "d3_b100_r0_vs_cake.csv"), index=False, float_format="%.5g")
    print(d.to_string())


def prereg_test(out):
    """§6 提議的樣本外 GPU 測試（LongAlpaca-7B，MHA，B 實驗從沒在 GPU 上跑過）。
    對幾個候選格先寫下地圖（R2'、R3、R4b）的預測；模擬器（r1_sim_longalpaca7b.csv）同格結果只當參考，用來避開「所有策略都一樣」的退化格。"""
    rows = []
    sim = pd.read_csv(os.path.join(RX, "r1_sim_longalpaca7b.csv"))
    Dstar = 0.6026     # R4b 的 D*（第一階段擬合，run 20261010-054706-d3-kappa-map）
    for cg, sd in [(3.69, "local"), (3.69, "nfs"), (11.6, "nfs"), (35.4159, "nfs")]:
        for cf in [0.25, 0.5]:
            for seed in [0, 1, 2]:
                meta = dict(model="longalpaca7b", workload="chat", cpu_gibps=cg, ssd_dev=sd, release="free",
                            cpu_frac=cf, f_scale=1.0)
                ft = features(meta)
                sm = sim[(np.isclose(sim.cpu_gibps, cg)) & (sim.workload == "chat") & (sim.ssd_dev == sd) &
                         (sim.release == "free") & (sim.cpu_frac == cf) & (sim.wl_seed == seed)]
                med = sm.set_index("strategy")["median"].to_dict()
                wt, _ = fam_best(med, WT)
                lz, _ = fam_best(med, LZ)
                pl = [med[x] for x in PLACEMENT if x in med]
                r = dict(run_id=RUN_ID, ts=TS, wl_seed=seed, **meta, **ft,
                         pred_R2p="|S5-S4+|<5% and |S5L-S4B|<5%", pred_R3="tie(|WT-LZ|<5%)",
                         pred_R4b=bool(ft["D"] >= Dstar), Dstar=Dstar,
                         sim_S5_vs_S4p=gain(med["S4+"], med["S5"]), sim_S5L_vs_S4B=gain(med["S4B"], med["S5L"]),
                         sim_WT_vs_LZ=gain(lz, wt), sim_spread=max(pl) / min(pl) - 1,
                         sim_src=";".join(sorted(sm["run_id"].unique())))
                for s_ in ["S1", "S2b", "S4", "S4+", "S4B", "S5", "S5L"]:
                    r[f"sim_{s_}"] = med.get(s_, np.nan)
                rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(out, "d3_gpu_test_prereg.csv"), index=False, float_format="%.5g")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(d[["cpu_gibps", "ssd_dev", "cpu_frac", "wl_seed", "b_over_n", "D", "pred_R4b", "sim_S5_vs_S4p", "sim_S5L_vs_S4B",
             "sim_WT_vs_LZ", "sim_spread", "sim_S1", "sim_S4", "sim_S4B", "sim_S5L"]].round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--b100", action="store_true", help="§2.1 的附加測試：第一階段 B 容量 100%（全在 CPU）時 R0（只載）vs Cake")
    ap.add_argument("--prereg-test", action="store_true", help="只算提議的 GPU 測試格的地圖預測（§6），不跑規則評分")
    a = ap.parse_args()
    if a.prereg_test:
        return prereg_test(a.out)
    if a.b100:
        return b100(a.out)
    os.makedirs(a.out, exist_ok=True)
    scores = []

    # ---------------- R1 / R1b ----------------
    rd = read_cells()
    rd = apply_r1(rd, 0.0, "R1")
    c_sync = fit_csync(rd)
    rd = apply_r1(rd, c_sync, "R1b")
    rd["R1b_c_sync_ms"] = c_sync * 1e3
    rd["R1b_role"] = np.where((rd.src == "A0") & rd.L.isin([8192, 32768]), "fit", "test")
    A0 = rd[rd.src == "A0"]
    scores.append(score("R1", "A0(all 28)", A0.meas_class, A0.R1_class,
                        f"median|speedup err|={A0.R1_err.abs().median():.4f}; winner3 hits={int((A0.R1_winner==A0.meas_winner).sum())}/{len(A0)}"))
    c7r = rd[(rd.src == "c7_anchor") & (rd.io == "real")]
    c7s = rd[(rd.src == "c7_anchor") & (rd.io == "sim")]
    scores.append(score("R1", "c7_anchor real", c7r.meas_class, c7r.R1_class, f"median|err|={c7r.R1_err.abs().median():.4f}"))
    scores.append(score("R1", "c7_anchor sim", c7s.meas_class, c7s.R1_class, f"median|err|={c7s.R1_err.abs().median():.4f}"))
    te = rd[rd.R1b_role == "test"]
    for nm, sub in [("A0 test(4K,16K)", te[te.src == "A0"]), ("c7_anchor real", te[(te.src == "c7_anchor") & (te.io == "real")]),
                    ("c7_anchor sim", te[(te.src == "c7_anchor") & (te.io == "sim")]),
                    ("A0 fit(8K,32K)", rd[rd.R1b_role == "fit"])]:
        scores.append(score("R1b", nm, sub.meas_class, sub.R1b_class,
                            f"c_sync={c_sync*1e3:.1f}ms; median|err|={sub.R1b_err.abs().median():.4f}"))
    rd.insert(0, "ts", TS)
    rd.insert(0, "run_id", RUN_ID)
    rd.to_csv(os.path.join(a.out, "d3_read_cells.csv"), index=False)

    # ---------------- 3090 ----------------
    t3 = r1t_3090()
    if len(t3):
        t3.insert(0, "ts", TS)
        t3.insert(0, "run_id", RUN_ID)
        t3.to_csv(os.path.join(a.out, "d3_3090_tier.csv"), index=False)
        scores.append(score("R1t", "3090 SSD vs drop (16K)", t3.meas_winner, t3.pred_winner,
                            f"contention={';'.join(t3.host_contention)}"))

    # ---------------- 寫入端 ----------------
    rows = [evaluate(c) for c in gpu_cells() + sim_cells()]
    W = pd.DataFrame(rows)
    W["is_gpu"] = W["set"].isin(["P1", "A08", "R1"])
    W.insert(0, "ts", TS)
    W.insert(0, "run_id", RUN_ID)

    # R4b：只用第一階段 GPU 擬合 D*
    fit = W[(W["set"] == "P1") & W["matters"].notna()]
    Ds = np.sort(fit["D"].unique())
    cands = [0.0] + list((Ds[:-1] + Ds[1:]) / 2) + [Ds[-1] + 1e-9]
    best = None
    for th in cands:
        ba = balanced_acc(fit["matters"].astype(bool), fit["D"] >= th)
        acc = ((fit["D"] >= th) == fit["matters"].astype(bool)).mean()
        key = (-ba, -acc, th)
        if best is None or key < best[0]:
            best = (key, th)
    Dstar = best[1]
    W["R4b_Dstar"] = Dstar
    W["R4b_pred"] = W["D"] >= Dstar
    W["R4a_pred_not"] = W["D"] < 0.05
    # 完整表（含模擬格的配對細節）放 run 目錄；進 git 的瘦身版 <1 MB（CLAUDE.md §2）
    rdir = os.path.join(os.environ.get("TIARA_RUNS", "/mlsteam/data/tiara/runs"), RUN_ID)
    if os.path.isdir(rdir):
        W.to_csv(os.path.join(rdir, "d3_write_cells_full.csv"), index=False)
    Ws = W.copy()
    Ws.loc[~Ws["is_gpu"], "R2p_pairs"] = ""
    Ws.to_csv(os.path.join(a.out, "d3_write_cells.csv"), index=False, float_format="%.5g")

    groups = [("GPU P1(第一階段)", W["set"] == "P1"), ("GPU A08", W["set"] == "A08"), ("GPU R1", W["set"] == "R1"),
              ("GPU 全部", W["is_gpu"]), ("SIM08", W["set"] == "SIM08"),
              ("SIMR1 llama", W["set"] == "SIMR1_llama31_8b"), ("SIMR1 longalpaca", W["set"] == "SIMR1_longalpaca7b")]
    for nm, m in groups:
        sub = W[m]
        # R2：安全性與準度
        s2 = sub[sub["R2_pred"].notna()]
        safe = s2[s2.R2_pred == "le5"]
        mayx = s2[s2.R2_pred == "may_exceed"]
        scores.append(score("R2 safety(pred le5)", nm, safe.R2_meas, safe.R2_pred))
        scores.append(score("R2 precision(pred may_exceed)", nm, mayx.R2_meas,
                            ["exceed"] * len(mayx)))
        scores.append(score("R2 as classifier", nm, s2.R2_meas, s2.R2_pred.replace({"may_exceed": "exceed"})))
        # R2'
        pairs = [p for js in sub["R2p_pairs"] for p in json.loads(js)]
        scores.append(dict(run_id=RUN_ID, ts=TS, rule="R2' pairs", dataset=nm, n=len(pairs),
                           hits=sum(p["hit"] for p in pairs),
                           hit_rate=(sum(p["hit"] for p in pairs) / len(pairs)) if pairs else np.nan,
                           baseline_majority="", baseline_hits=np.nan, baseline_rate=np.nan,
                           note=f"cells all-hit={int((sub.R2p_hits==sub.R2p_n).where(sub.R2p_n>0).sum())}/{int((sub.R2p_n>0).sum())}"))
        for rel in ["free", "hold"]:
            pr = [p for js in sub[sub.release == rel]["R2p_pairs"] for p in json.loads(js)]
            if pr:
                scores.append(dict(run_id=RUN_ID, ts=TS, rule=f"R2' pairs {rel}", dataset=nm, n=len(pr),
                                   hits=sum(p["hit"] for p in pr), hit_rate=sum(p["hit"] for p in pr) / len(pr),
                                   baseline_majority="", baseline_hits=np.nan, baseline_rate=np.nan, note=""))
        # R3
        s3 = sub[sub["R3_pred"].notna()]
        scores.append(score("R3", nm, s3.R3_meas, s3.R3_pred))
        # R4
        s4 = sub[sub["matters"].notna()]
        scores.append(score("R4b", nm, s4.matters.astype(bool), s4.R4b_pred, f"D*={Dstar:.4f}"))
        ce = s4[s4.R4a_pred_not & s4.matters.astype(bool)]
        scores.append(dict(run_id=RUN_ID, ts=TS, rule="R4a counterexamples(D<0.05 but matters)", dataset=nm,
                           n=int(s4.R4a_pred_not.sum()), hits=int(s4.R4a_pred_not.sum() - len(ce)),
                           hit_rate=(1 - len(ce) / s4.R4a_pred_not.sum()) if s4.R4a_pred_not.sum() else np.nan,
                           baseline_majority="", baseline_hits=np.nan, baseline_rate=np.nan, note=f"counterexamples={len(ce)}"))
        # 參考：只用 cpu_frac 的笨規則（c<1 ⇒ 重要）
        scores.append(score("R4-ref cpu_frac<1", nm, s4.matters.astype(bool), s4.cpu_frac < 1))
    S = pd.DataFrame(scores)
    S.to_csv(os.path.join(a.out, "d3_rule_scores.csv"), index=False)
    posthoc(W, rd, a.out)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 500)
    print(S[["rule", "dataset", "n", "hits", "hit_rate", "baseline_majority", "baseline_rate", "note"]].to_string())
    print(f"\nc_sync={c_sync*1e3:.2f} ms  D*={Dstar:.4f}")


if __name__ == "__main__":
    main()
