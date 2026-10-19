"""summarize_grids.py — p3cal 補量的還原格子（X7 的 grid_*.sh，code/p3x/x7_restore.py）整理成 results/p3_calib_3090/（2026-10-18）。

  python code/p3cal/summarize_grids.py --out results/p3_calib_3090 --grid grid_llama=<run> --grid grid_llama=<run> \
      --grid grid_qwen=<run> --grid grid_fused=<run> ...

和 code/p3x/x7_summarize.py 的「還原格子」段落同一個分組、同一組欄位（restore_3090_v2_summary.csv 可以直接和
results/p3_explore/x7/restore_3090_summary.csv 並列）；同 switch 鄰居卡的 PCIe 流量用 code/p3x/x7_pcie_flag.flag（門檻 100 MB/s，同 X7）。
另外：
  - meet_check_v2.csv：E0 cake 的會合點實測 vs X7 的兩條線模型（code/p3x/x7_band.evaluate＋results/p3_explore/x7/band_spec.json，
    3090 的 f、忙的倍數、X7 的 CPU 層 H2D），同 X7 的 model_check.csv 定義（|實測中位數−模型| ≤1 block）。
  - lowprec_cpu_v2.csv：CPU 層（真實 H2D）BF16 → 低精度的還原時間變化（主線 PyTorch 反量化 vs 融合 Triton kernel）。
只讀 X7 的檔，不改。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "code", "p3x"))
import x7_band as XB  # noqa: E402
import x7_pcie_flag  # noqa: E402

RUNS = os.environ.get("TIARA_RUNS", "/ssd8/hungwei/paper-hkv/runs")
SRID = os.environ.get("RUN_ID", "NO_RUN_ID")
X7 = os.path.join(REPO, "results", "p3_explore", "x7")


def med(x):
    x = [v for v in x if v == v]
    return statistics.median(x) if x else float("nan")


def check_run(r):
    d = os.path.join(RUNS, r)
    ec = open(os.path.join(d, "exit_code")).read().strip() if os.path.exists(os.path.join(d, "exit_code")) else "MISSING"
    if ec != "0":
        raise SystemExit(f"{r}: exit_code={ec}，不進 results")
    if os.path.exists(os.path.join(d, "CONTAMINATED")):
        raise SystemExit(f"{r}: CONTAMINATED，不進 results（CLAUDE.md §3）")
    g = json.load(open(os.path.join(d, "gpu_guard.json")))
    if g.get("contaminated"):
        raise SystemExit(f"{r}: gpu_guard contaminated")
    return g["verdict"]


def write(path, rows):
    if not rows:
        return
    keys = list(rows[0])
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    pd.DataFrame(rows)[keys].to_csv(path, index=False)
    print("wrote", path, len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", action="append", required=True, help="<grid 名>=<run_id>")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    allg = []
    guards = {}
    for s in a.grid:
        key, r = s.split("=")
        guards[r] = check_run(r)
        x = x7_pcie_flag.flag(os.path.join(RUNS, r), 100.0)
        x["grid"] = key
        x["src_run"] = r
        allg.append(x)
    G = pd.concat(allg)
    bad = G[G.contaminated_gpu.astype(str) != "False"]
    if len(bad):
        raise SystemExit(f"{len(bad)} 列 contaminated_gpu≠False，不進 results")
    G["lp_read"] = G.n_load_fp8 + G.n_load_int4
    raw_cols = ["run_id", "ts", "grid", "model", "length", "config", "fmt", "mode", "tier", "tier_GiBps", "gpu_state", "rep",
                "order_idx", "b", "n_recompute", "n_load_bf16", "n_load_fp8", "n_load_int4", "bytes_read", "t_restore",
                "t_recompute", "t_load", "t_h2d", "t_dequant", "t_wait", "waiting_line", "t_new_turn", "ttft", "copy_bound",
                "kv_bad", "dequant_impl", "contaminated_gpu", "host_level", "gpu", "sibling_max_mbps", "sibling_busy"]
    G[raw_cols].assign(summarize_run_id=SRID).to_csv(os.path.join(a.out, "restore_3090_v2_raw.csv"), index=False)
    print("wrote restore_3090_v2_raw.csv", len(G))
    srows = []
    for k, x in G.groupby(["grid", "model", "length", "config", "fmt", "mode", "tier", "gpu_state", "dequant_impl"]):
        lp = x.lp_read.median()
        q = x[~x.sibling_busy]
        srows.append(dict(
            run_id=" ".join(sorted(x.run_id.unique())), ts=x.ts.min(), summarize_run_id=SRID, grid=k[0], model=k[1],
            length=k[2], config=k[3], fmt=k[4], mode=k[5], tier=k[6], tier_GiBps=x.tier_GiBps.iloc[0], gpu_state=k[7],
            n=len(x), n_launch=x.run_id.nunique(), b_median=med(x.b.tolist()), b_values=" ".join(map(str, sorted(x.b.unique()))),
            b_median_by_launch=" ".join(f"{r_}:{med(g_.b.tolist())}" for r_, g_ in x.groupby("run_id")),
            t_restore_median_s=round(x.t_restore.median(), 5),
            t_restore_median_by_launch=" ".join(f"{r_}:{round(g_.t_restore.median(), 5)}" for r_, g_ in x.groupby("run_id")),
            ttft_median_s=round(x.ttft.median(), 5), t_recompute_median_s=round(x.t_recompute.median(), 5),
            t_h2d_median_s=round(x.t_h2d.median(), 5), t_dequant_median_s=round(x.t_dequant.median(), 5), lp_read_median=lp,
            dequant_ms_per_lp_block=round(x.t_dequant.median() / lp * 1e3, 4) if lp > 0 else "",
            t_new_turn_median_s=round(x.t_new_turn.median(), 5),
            kv_bad_sum=pd.to_numeric(x.kv_bad, errors="coerce").fillna(0).sum(),
            kv_checked=int(pd.to_numeric(x.kv_bad, errors="coerce").notna().sum()),
            copy_bound_rows=int((x.copy_bound > 0).sum()), host_levels=" ".join(sorted(x.host_level.astype(str).unique())),
            dequant_impl=k[8], gpu=" ".join(map(str, sorted(x.gpu.unique()))),
            guard=" ".join(sorted({guards[r_] for r_ in x.run_id.unique()})),
            n_sibling_pcie_ge100MBps=int(x.sibling_busy.sum()), sibling_max_mbps=int(x.sibling_max_mbps.max()),
            b_median_sibling_quiet=med(q.b.tolist()) if len(q) else "NOT_MEASURED",
            t_restore_median_s_sibling_quiet=round(q.t_restore.median(), 5) if len(q) else "NOT_MEASURED"))
    write(os.path.join(a.out, "restore_3090_v2_summary.csv"), srows)
    S = pd.DataFrame(srows)

    # ---- 會合點：實測 vs X7 的兩條線模型
    spec = json.load(open(os.path.join(X7, "band_spec.json")))
    cases = {(c["platform"], c["model"], c["gpu_state"]): c for c in spec["cases"]}
    C = json.load(open(os.path.join(X7, "constants_3090.json")))
    chk = []
    for _, r in S[(S.config == "E0") & (S["mode"] == "cake") & (S.length == 32768)].iterrows():
        bw = C["models"][r.model]["cpu_tier_h2d_GiBps_chunk_view"] if r.tier == "cpu" else float(r.tier_GiBps)
        pred = XB.evaluate(cases[("3090", r.model, r.gpu_state)], bw)
        lo = S[(S.grid == r.grid) & (S.model == r.model) & (S.tier == r.tier) & (S.gpu_state == r.gpu_state) &
               (S.config == "E0") & (S["mode"] == "load_only") & (S.length == 32768)]
        co = S[(S.grid == r.grid) & (S.model == r.model) & (S.gpu_state == r.gpu_state) & (S["mode"] == "compute_only")
               & (S.length == 32768)]
        lo_t = float(lo.ttft_median_s.iloc[0]) if len(lo) else None
        co_t = float(co.ttft_median_s.iloc[0]) if len(co) else None
        sp = min(lo_t, co_t) / r.ttft_median_s if (lo_t and co_t) else "NOT_MEASURED（只載入在這個頻寬沒跑）"
        chk.append(dict(run_id=r.run_id, ts=r.ts, summarize_run_id=SRID, grid=r.grid, model=r.model, tier=r.tier, GiBps=bw,
                        gpu_state=r.gpu_state, b_measured=r.b_median, b_values=r.b_values, b_by_launch=r.b_median_by_launch,
                        b_model=pred["b_greedy"], abs_err=abs(r.b_median - pred["b_greedy"]),
                        within_1=abs(r.b_median - pred["b_greedy"]) <= 1, ttft_cake_s=r.ttft_median_s,
                        ttft_load_only_s=lo_t if lo_t else "NOT_MEASURED", ttft_compute_only_s=co_t,
                        speedup_ttft_measured=round(sp, 4) if not isinstance(sp, str) else sp,
                        speedup_ttft_model=round(pred["speedup_ttft"], 4), model_ttft_fallback=pred["ttft_fallback"],
                        b_median_sibling_quiet=r.b_median_sibling_quiet, n_sibling_busy=r.n_sibling_pcie_ge100MBps))
    write(os.path.join(a.out, "meet_check_v2.csv"), chk)

    # ---- CPU 層的低精度：BF16 → FP8／INT4，主線 vs 融合
    lp = []
    cpu = S[S.tier == "cpu"]
    for (g, m, mode, st, impl), x in cpu.groupby(["grid", "model", "mode", "gpu_state", "dequant_impl"]):
        b16 = x[x.config == "E0"]
        if not len(b16):
            continue
        t0 = float(b16.t_restore_median_s.iloc[0])
        for _, r in x[x.config == "E1"].iterrows():
            lp.append(dict(run_id=r.run_id, ts=r.ts, summarize_run_id=SRID, grid=g, model=m, mode=mode, gpu_state=st,
                           dequant_impl=impl, fmt=r.fmt, t_restore_bf16_s=t0, t_restore_lowprec_s=r.t_restore_median_s,
                           change_pct=round((r.t_restore_median_s / t0 - 1) * 100, 1),
                           dequant_ms_per_lp_block=r.dequant_ms_per_lp_block, t_h2d_lowprec_s=r.t_h2d_median_s,
                           t_h2d_bf16_s=float(b16.t_h2d_median_s.iloc[0]), n=r.n))
    write(os.path.join(a.out, "lowprec_cpu_v2.csv"), lp)


if __name__ == "__main__":
    main()
