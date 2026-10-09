"""m7_round1_analyze.py — 破解計劃第 1 輪（11_round1_plan.md）的分析。

  python code/m7_round1_analyze.py h1      # 寫入量 Pareto（重分析第一階段 B 與 08 的 GPU 實測，不用 GPU）
  python code/m7_round1_analyze.py kappa   # 各模型的 f(i) → 不同頻寬下的 b/n（H2 篩選）

輸出到 results/m7_explore_mi300x/。每列帶 run_id（指回原始 GPU 列的 run）與 ts（分析時間）。
"""
import datetime
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "..", "results", "m7_write_policy_mi300x")
OUT = os.path.join(HERE, "..", "results", "m7_explore_mi300x")
TS = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
WT = ["S5", "S5s", "S5L", "S5P", "S5c"]
NWT = ["S0", "S1", "S2b", "S3", "S4", "S4+", "S4+P", "S4L", "S4B"]   # 不看位置＋延後版（S4B）
GiB = 1 << 30


# ------------------------------------------------------------------ H1
def _load_b():
    srcs = [("b_share.csv", "share", 35.4159), ("b_fifo.csv", "fifo", 35.4159),
            ("b_share_cpu3.69.csv", "share", 3.69), ("b2_share.csv", None, None)]
    ds = []
    for fn, io, cg in srcs:
        p = os.path.join(RES, fn)
        if not os.path.exists(p):
            p += ".gz"
        d = pd.read_csv(p)
        d["src"] = fn
        if "release" not in d:
            d["release"], d["workload"], d["cpu_gibps"] = "free", "chat", cg
        ds.append(d)
    return pd.concat(ds, ignore_index=True)


def h1():
    d = _load_b()
    d["cpu_gibps"] = d.cpu_gibps.round(2)
    cfg = ["src", "io_model", "ssd_dev", "gap_s", "ssd_frac", "wl_seed", "cpu_gibps", "release", "workload", "cpu_frac"]
    rows = []
    for key, g in d.groupby(cfg + ["strategy"]):
        c = dict(zip(cfg + ["strategy"], key))
        # 每個 rep 最後一列的累計寫入量；TTFT 只看回來的請求
        last = g.sort_values("ev").groupby("rep").tail(1)
        rows.append(dict(run_id=";".join(sorted(g.run_id.unique())), ts=TS, **c,
                         ttft_med=g[g["round"] >= 2].ttft.median(),
                         ssd_GiB=last.w_bytes_ssd.median() / GiB, cpu_GiB=last.w_bytes_cpu.median() / GiB,
                         n_demote=last.n_demote.median()))
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(OUT, "h1_write_volume.csv"), index=False)
    ver = []
    for key, g in s.groupby(cfg):
        c = dict(zip(cfg, key))
        best = g.ttft_med.min()
        cand = g[g.ttft_med <= 1.05 * best]
        nw = cand[cand.strategy.isin(NWT)]
        wt = cand[cand.strategy.isin(WT)]
        min_nw = nw.ssd_GiB.min() if len(nw) else float("nan")
        r = dict(c, best=g.loc[g.ttft_med.idxmin(), "strategy"], n_cand=len(cand),
                 cand=" ".join(cand.strategy), min_ssd_nwt=min_nw,
                 min_ssd_nwt_by=nw.loc[nw.ssd_GiB.idxmin(), "strategy"] if len(nw) else "",
                 wt_in_band=" ".join(wt.strategy), min_ssd_wt=wt.ssd_GiB.min() if len(wt) else float("nan"))
        # 兩邊都是 0（容量 100%，沒人寫 SSD）不算優勢
        r["wt_adv"] = bool(len(wt) and (len(nw) == 0 or (min_nw > 0 and wt.ssd_GiB.min() <= 0.8 * min_nw)))
        ver.append(r)
    v = pd.DataFrame(ver)
    v.insert(0, "ts", TS)
    v.to_csv(os.path.join(OUT, "h1_verdict.csv"), index=False)
    pd.set_option("display.width", 250)
    print(f"設定數 {len(v)}；寫入時決定在 5% TTFT 帶內、且 SSD 寫入 ≤ 不看位置／延後版最小值 0.8 倍：{int(v.wt_adv.sum())} 個"
          f"（{v.wt_adv.mean() * 100:.1f}%）；判準 ≥20% 才存活")
    print("\n各來源：")
    print(v.groupby("src").wt_adv.agg(["sum", "count"]))
    print("\n5% 帶內 SSD 寫入最少的策略（不看位置／延後版）：")
    print(v.min_ssd_nwt_by.value_counts())
    print("\n寫入時決定有優勢的設定：")
    print(v[v.wt_adv][["src", "release", "workload", "cpu_gibps", "ssd_dev", "gap_s", "ssd_frac", "wl_seed", "cpu_frac",
                       "best", "cand", "min_ssd_nwt", "min_ssd_nwt_by", "min_ssd_wt"]].round(2).to_string(index=False))
    # 各策略的 SSD 寫入量（容量 25%／50%，chat、free、NFS、gap 0，第一階段主設定）
    m = s[(s.src == "b_share.csv") & (s.ssd_dev == "nfs") & (s.gap_s == 0) & (s.ssd_frac >= 1) & (s.wl_seed == 0)]
    print("\n第一階段主設定（NFS、gap 0、seed 0）：SSD 寫入 GiB／TTFT 中位數")
    print(m.pivot_table(index="strategy", columns="cpu_frac", values=["ssd_GiB", "ttft_med"]).round(2).to_string())


# ------------------------------------------------------------------ kappa
MODELS = {   # 名稱 → (f 檔, chunk bytes 來源 json)
    "llama31_8b": ("calib_c1_llama31_8b.csv", None),
    "longalpaca7b": ("calib_c1_longalpaca7b.csv", "correct_longalpaca7b.json"),
    "qwen3_30b_a3b_grouped": ("calib_c1_qwen3_30b_a3b_grouped.csv", "correct_qwen3_30b_a3b.json"),
    "qwen3_30b_a3b_eager": ("calib_c1_qwen3_30b_a3b_eager.csv", "correct_qwen3_30b_a3b.json"),
}
CFG_BYTES = {"llama31_8b": 32 * 2 * 8 * 128 * 2, "longalpaca7b": 32 * 2 * 32 * 128 * 2,       # 〔算術〕config.json
             "qwen3_30b_a3b_grouped": 48 * 2 * 4 * 128 * 2, "qwen3_30b_a3b_eager": 48 * 2 * 4 * 128 * 2}


def load_f_csv(p):
    d = pd.read_csv(p)
    d = d[(d.item == "f_chunk") & (d.gpu_state == "idle")]
    m = d.groupby("chunk_idx").ms.median()
    return [float(m[i]) / 1e3 for i in sorted(m.index)], ";".join(sorted(d.run_id.unique())) if "run_id" in d else ""


def boundary(f, n, ell):
    """和 m7_write_policy.write_boundary 同一個定義：b = argmin_m max(sum f[:m], (n-m)·ell)。"""
    best, bm = None, 0
    for m in range(n + 1):
        t = max(sum(f[:m]), (n - m) * ell)
        if best is None or t < best - 1e-12:
            best, bm = t, m
    return bm, best


def kappa():
    rows = []
    for name, (fn, cj) in MODELS.items():
        p = os.path.join(OUT, fn)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            print(f"{name}: {fn} 不存在或還沒有資料 → NOT_MEASURED")
            continue
        f, rid = load_f_csv(p)
        cb = CFG_BYTES[name]
        tok_bytes = cb
        cb_chunk = cb * 512
        n = 64
        for bw in (0.33, 0.81, 1.0, 2.91, 3.69, 6.98, 11.6, 35.4159):
            ell = cb_chunk / (bw * GiB)          # 只算頻寬，不含每個 chunk 的固定開銷
            b, t = boundary(f, n, ell)
            rows.append(dict(run_id=rid, ts=TS, model=name, kv_bytes_per_token=tok_bytes, chunk_MiB=cb_chunk / 2 ** 20,
                             gibps=bw, ell_ms=ell * 1e3, f0_ms=f[0] * 1e3, f63_ms=f[63] * 1e3,
                             f_sum_s=sum(f[:n]), load_all_s=n * ell, cake_s=t, b=b, b_frac=b / n,
                             speedup_vs_best=min(sum(f[:n]), n * ell) / t))
    k = pd.DataFrame(rows)
    k.to_csv(os.path.join(OUT, "kappa_screen.csv"), index=False)
    pd.set_option("display.width", 250)
    print(k[["model", "chunk_MiB", "gibps", "ell_ms", "f0_ms", "f63_ms", "b", "b_frac", "speedup_vs_best"]].round(3).to_string(index=False))
    print("\n通過（快的層 3.69 或 35.4 GiB/s 時 b/n ≥ 30%）：")
    fast = k[k.gibps.isin([3.69, 35.4159])]
    print(fast.groupby("model").b_frac.max().round(3).to_string())


# ------------------------------------------------------------------ sim（H2 延伸〔模擬〕）
NWT0 = ["S0", "S1", "S2b", "S4", "S4+", "S4+P", "S4L"]


def sim():
    import glob
    out = []
    for fn in sorted(glob.glob(os.path.join(OUT, "r1_sim_*.csv"))):
        if fn.endswith(("_verdict.csv", "_configs.csv")):      # 本函式自己的輸出
            continue
        d = pd.read_csv(fn)
        cfg = ["model", "cpu_gibps", "release", "workload", "ssd_dev", "cpu_frac", "wl_seed"]
        for key, g in d.groupby(cfg):
            m = g.set_index("strategy")["median"]
            w = g.set_index("strategy")["w_ssd_GiB"]
            bn, b4, b4w = m[NWT0].min(), m["S4B"], m["S4W"]
            b4c = m["S4C"] if "S4C" in m else float("nan")
            wt = m[WT]
            bw = wt.idxmin()
            r = dict(run_id=g.run_id.iloc[0], ts=TS, **dict(zip(cfg, key)), best_nwt=m[NWT0].idxmin(), best_wt=bw,
                     t_nwt=bn, t_S4B=b4, t_S4W=b4w, t_wt=wt.min(),
                     t_S4C=b4c, g_nwt=(bn - wt.min()) / bn, g_S4B=(b4 - wt.min()) / b4, g_S4W=(b4w - wt.min()) / b4w,
                     g_S4C=(b4c - wt.min()) / b4c,
                     w_wt=w[bw], w_best_twin=w[[m[NWT0].idxmin(), "S4B", "S4W"]].min())
            r["pass_nwt_S4B"] = (r["g_nwt"] >= 0.05) and (r["g_S4B"] >= 0.05)
            r["pass_noC"] = r["pass_nwt_S4B"] and (r["g_S4W"] >= 0.05)
            r["pass_all"] = r["pass_noC"] and (r["g_S4C"] >= 0.05)        # NaN（舊檔沒有 S4C）→ False
            out.append(r)
    v = pd.DataFrame(out)
    v.to_csv(os.path.join(OUT, "r1_sim_verdict.csv"), index=False)
    cfg = ["model", "cpu_gibps", "release", "workload", "ssd_dev", "cpu_frac"]
    agg = v.groupby(cfg).agg(seeds=("wl_seed", "nunique"), n_pass=("pass_all", "sum"), n_pass_noW=("pass_nwt_S4B", "sum"), n_pass_noC=("pass_noC", "sum"),
                             g_nwt_med=("g_nwt", "median"), g_S4B_med=("g_S4B", "median"), g_S4W_med=("g_S4W", "median"), g_S4C_med=("g_S4C", "median"),
                             g_min_all=("g_nwt", "min")).reset_index()
    agg["survive"] = agg.n_pass >= 3
    agg.insert(0, "ts", TS)
    agg.to_csv(os.path.join(OUT, "r1_sim_configs.csv"), index=False)
    pd.set_option("display.width", 250)
    for mdl, g in v.groupby("model"):
        a2 = agg[agg.model == mdl]
        print(f"\n== {mdl}：{len(g)} 格（設定×seed）；同時贏「最好不看位置」與 S4B ≥5%：{int(g.pass_nwt_S4B.sum())} 格；"
              f"再加 S4W：{int(g.pass_noC.sum())} 格；再加 S4C（全部對照組）：{int(g.pass_all.sum())} 格；設定 {len(a2)} 個，≥3/5 seed 存活：{int(a2.survive.sum())} 個")
        top = a2.sort_values(["n_pass", "n_pass_noC", "g_nwt_med"], ascending=False).head(8)
        print(top[["cpu_gibps", "release", "workload", "ssd_dev", "cpu_frac", "n_pass", "n_pass_noC", "n_pass_noW", "g_nwt_med",
                   "g_S4B_med", "g_S4W_med", "g_S4C_med"]].round(3).to_string(index=False))


# ------------------------------------------------------------------ gpu（11 追加 4：Llama 存活設定的 GPU 確認）
def gpu():
    fn = os.path.join(OUT, "r1_gpu_llama_doc_free_local.csv")
    d = pd.read_csv(fn)
    print(f"{fn}: {len(d)} 列；run_id {sorted(d.run_id.unique())}")
    print(f"kv_bad 非 0：{int((d.kv_bad.fillna(0) != 0).sum())}；contaminated_gpu：{int(d.contaminated_gpu.astype(bool).sum())}")
    sim = pd.read_csv(os.path.join(OUT, "r1_sim_llama31_8b.csv"))
    sim = sim[(sim.cpu_gibps.round(1) == 11.6) & (sim.release == "free") & (sim.workload == "doc") & (sim.ssd_dev == "local")
              & (sim.cpu_frac == 0.5)]
    NW = ["S1", "S2b", "S4", "S4+", "S4+P", "S4L"]
    rows, ver = [], []
    for seed, g in d.groupby("wl_seed"):
        n_ev = g.ev.max() + 1
        for st, h in g.groupby("strategy"):
            full = h.groupby("rep").ev.nunique()
            ok = full[full == n_ev].index
            r = h[(h["round"] >= 2) & h.rep.isin(ok)]
            sm = sim[(sim.wl_seed == seed) & (sim.strategy == st)]
            rows.append(dict(run_id=";".join(sorted(h.run_id.unique())), ts=TS, wl_seed=seed, strategy=st, reps=len(ok),
                             n=len(r), median=r.ttft.median(), mean=r.ttft.mean(), p90=r.ttft.quantile(0.9),
                             sim_median=sm["median"].iloc[0] if len(sm) else float("nan")))
    t = pd.DataFrame(rows)
    t["sim_err"] = (t.sim_median - t["median"]) / t["median"]
    t.to_csv(os.path.join(OUT, "r1_gpu_summary.csv"), index=False)
    for seed, g in t.groupby("wl_seed"):
        m = g.set_index("strategy")
        if not {"S5L", "S4B", "S4W", "S4C"} <= set(m.index) or (m.reps < 3).any():
            print(f"seed {seed}：還沒跑完（策略 {len(m)}，rep 不齊）")
            continue
        for col in ("median", "mean"):
            bn = m.loc[m.index.isin(NW), col]
            x = m.at["S5L", col]
            r = dict(run_id=m.run_id.iloc[0], ts=TS, wl_seed=seed, stat=col, best_nwt=bn.idxmin(), t_nwt=bn.min(),
                     t_S4B=m.at["S4B", col], t_S4W=m.at["S4W", col], t_S4C=m.at["S4C", col], t_S5L=x,
                     g_nwt=(bn.min() - x) / bn.min(), g_S4B=(m.at["S4B", col] - x) / m.at["S4B", col],
                     g_S4W=(m.at["S4W", col] - x) / m.at["S4W", col], g_S4C=(m.at["S4C", col] - x) / m.at["S4C", col])
            r["pass"] = min(r["g_nwt"], r["g_S4B"], r["g_S4W"], r["g_S4C"]) >= 0.05
            ver.append(r)
    v = pd.DataFrame(ver)
    if len(v):
        v.to_csv(os.path.join(OUT, "r1_gpu_verdict.csv"), index=False)
    pd.set_option("display.width", 250)
    print(t[["wl_seed", "strategy", "reps", "n", "median", "mean", "p90", "sim_median", "sim_err"]].round(3).to_string(index=False))
    if len(v):
        print(v[["wl_seed", "stat", "best_nwt", "t_nwt", "t_S4B", "t_S4W", "t_S4C", "t_S5L", "g_nwt", "g_S4B", "g_S4W", "g_S4C",
                 "pass"]].round(3).to_string(index=False))
        med = v[v.stat == "median"]
        print(f"\n判定（中位數，11 追加 4）：通過的 seed {int(med['pass'].sum())}／{len(med)}；≥3／5 才算存活")
        print(f"模擬誤差：|err| ≤10% 的格 {int((t.sim_err.abs() <= 0.1).sum())}／{t.sim_err.notna().sum()}，中位 {t.sim_err.abs().median():.3f}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    {"h1": h1, "kappa": kappa, "sim": sim, "gpu": gpu}[sys.argv[1]]()
