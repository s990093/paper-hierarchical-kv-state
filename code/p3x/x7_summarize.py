"""x7_summarize.py — X7：把 3090 的原始 run 目錄整理成 results/p3_explore/x7/ 的 CSV、constants_3090.json，並和 MI300X 並列。只用 CPU。

  python code/p3x/x7_summarize.py --runs-json runs.json --out-dir results/p3_explore/x7

runs.json（哪些 run 進 results；被標 CONTAMINATED 或 exit≠0 的 run 會被拒絕）：
  {"c0": [...], "calibf_qwen": "...", "calibf_llama": "...", "ladder_llama": "...", "busy_qwen": "...", "busy_llama": "...",
   "dqbench": "...", "grid_qwen": [...], "grid_llama": [...], "grid_llama64k": [...], "grid_fused": [...]}
每個 CSV 都有 run_id（來源 run）與 ts（來源列的時間；彙總列取來源列最早的 ts），另有 summarize_run_id（這次整理的 run）。
MI300X 的對照值全部從 repo 裡已提交的檔案讀（附檔名與 run_id），不手抄。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from datetime import datetime

import pandas as pd

RUNS = os.environ.get("TIARA_RUNS", "/ssd8/hungwei/paper-hkv/runs")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRID = os.environ.get("RUN_ID", "NO_RUN_ID")
NM = "NOT_MEASURED"


def rd(r):
    return os.path.join(RUNS, r)


def check_run(r):
    d = rd(r)
    ec = open(os.path.join(d, "exit_code")).read().strip() if os.path.exists(os.path.join(d, "exit_code")) else "MISSING"
    if ec != "0":
        raise SystemExit(f"{r}: exit_code={ec}，不進 results")
    if os.path.exists(os.path.join(d, "CONTAMINATED")):
        raise SystemExit(f"{r}: CONTAMINATED，不進 results（CLAUDE.md §3）")
    g = json.load(open(os.path.join(d, "gpu_guard.json"))) if os.path.exists(os.path.join(d, "gpu_guard.json")) else None
    if g is not None and g.get("contaminated"):
        raise SystemExit(f"{r}: gpu_guard contaminated")
    return g["verdict"] if g else "NO_GUARD"


def write(path, rows):
    if not rows:
        return
    keys = list(rows[0])
    for r in rows[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("wrote", path, len(rows))


def med(x):
    x = [v for v in x if v == v]
    return statistics.median(x) if x else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-json", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    R = json.load(open(a.runs_json))
    os.makedirs(a.out_dir, exist_ok=True)
    O = lambda n: os.path.join(a.out_dir, n)  # noqa: E731
    cons = {"how_to_use": ("X1／X4／X5 加跑：f(i)＝models.<m>.f_ms_idle_32k（GPU 忙乘 busy_0.7_slowdown.by_chunk 內插）；"
                           "CPU 層 ℓ＝models.<m>.ell_cpu_ms（真實 H2D，不是限速器）；慢層用 MI300X 的 tier_params 或 3090 的 bstar_GiBps；"
                           "反量化＝dequant_bench_ms_per_block（單獨）或 models.<m>.dequant_ms_per_block_in_restore_cpu（還原時、主線寫法）。"
                           "結果標〔3090 常數〕，和 MI300X 常數分開列。"),
            "summarize_run_id": SRID, "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
            "platform": "A_3090 (RTX 3090 24 GiB, sm_86, PCIe Gen3 x16 behind PLX PEX8796, Xeon Gold 6248R)",
            "note": "〔3090 常數〕X7 量測。和 MI300X 常數分開列；所有值附 run_id。", "models": {}}

    # ---------------- H2D／D2H（C0、duplex）
    rows = []
    for r in R["c0"]:
        v = check_run(r)
        d = pd.read_csv(os.path.join(rd(r), "calib_c0.csv"))
        cmd = open(os.path.join(rd(r), "cmd.sh")).read().strip()
        gpu = [ln for ln in open(os.path.join(rd(r), "context.txt")) if ln.startswith("short=")][0].split("cuda_visible='")[1].split("'")[0]
        b = d[d.kind == "bw"].copy()
        b["value"] = b.value.astype(float)
        for (item, size), g in b.groupby(["item", "size_mib"]):
            rows.append(dict(run_id=r, ts=g.ts.min(), summarize_run_id=SRID, gpu=gpu, taskset="taskset" in cmd, item=item,
                             size_mib=size, median_GiBps=round(g.value.median(), 3), min_GiBps=round(g.value.min(), 3),
                             max_GiBps=round(g.value.max(), 3), n=len(g), guard=v))
        dp = os.path.join(rd(r), "calib_c0_duplex.csv")
        if os.path.exists(dp):
            x = pd.read_csv(dp)
            for case, g in x.groupby("case"):
                rows.append(dict(run_id=r, ts=g.ts.min(), summarize_run_id=SRID, gpu=gpu, taskset="taskset" in cmd,
                                 item=f"H2D_pinned_duplex_{case}", size_mib=1024, median_GiBps=round(g.h2d_GiBps.median(), 3),
                                 min_GiBps=round(g.h2d_GiBps.min(), 3), max_GiBps=round(g.h2d_GiBps.max(), 3), n=len(g), guard=v))
    write(O("h2d_3090.csv"), rows)
    h2d = pd.DataFrame(rows)
    view = h2d[(h2d.item == "H2D_pinned_chunk_into_kv_view") & (h2d.taskset)]
    cons["cpu_tier_h2d_llama_chunk_GiBps_c0"] = {"median_of_run_medians": round(view.median_GiBps.median(), 3),
                                                  "per_run": dict(zip(view.run_id, view.median_GiBps)), "gpu": sorted(set(view.gpu))}
    d2h = h2d[(h2d.item == "D2H_pinned_chunk_from_kv_view") & (h2d.taskset)]
    cons["cpu_tier_d2h_llama_chunk_GiBps_c0"] = {"median_of_run_medians": round(d2h.median_GiBps.median(), 3),
                                                  "per_run": dict(zip(d2h.run_id, d2h.median_GiBps))}
    dx = h2d[h2d.item.str.startswith("H2D_pinned_duplex") & (h2d.taskset)]
    kr = {}
    for r_, g_ in dx.groupby("run_id"):
        al = g_[g_.item == "H2D_pinned_duplex_alone"].median_GiBps.iloc[0]
        wd = g_[g_.item == "H2D_pinned_duplex_with_d2h"].median_GiBps.iloc[0]
        kr[r_] = round(al / wd, 4)
    cons["cpu_tier_k_read_duplex"] = {"per_run": kr, "definition": "H2D 單獨 ÷ H2D 同時有 D2H（中位數；MI300X tier_params 的 cpu k_read＝1.161 同定義）"}

    # ---------------- f(i)
    frows = []
    for m in ("qwen", "llama"):
        r = R[f"calibf_{m}"]
        v = check_run(r)
        bw = json.load(open(os.path.join(rd(r), "bw.json")))
        cf = pd.read_csv(os.path.join(rd(r), "calib_f.csv"))
        for i, fm in enumerate(bw["f_ms_median_by_chunk"]):
            g = cf[cf.chunk_idx == i]
            frows.append(dict(run_id=r, ts=g.ts.min(), summarize_run_id=SRID, model=m, chunk_idx=i, pos_end=(i + 1) * 512,
                              f_ms_median=fm, n_reps=len(g), f_ms_min=g.ms.min(), f_ms_max=g.ms.max(), source="m10_speed calib-f", guard=v))
        cons["models"][m] = {"chunk_bytes": bw["chunk_bytes"], "n_blocks_32k": bw["n_blocks"],
                             "f_ms_idle_32k": bw["f_ms_median_by_chunk"], "f_sum_s_32k": bw["f_sum_s"],
                             "f_src_run": r, "cpu_tier_h2d_GiBps_chunk_view": bw["real_pinned_h2d_GiBps"],
                             "cpu_tier_h2d_src_run": r,
                             "ell_cpu_ms": round(bw["chunk_bytes"] / (bw["real_pinned_h2d_GiBps"] * (1 << 30)) * 1e3, 4),
                             "bstar_GiBps": {k: v_["GiBps"] for k, v_ in bw["targets"].items()}, "bstar_src_run": r}
    r = R["ladder_llama"]
    v = check_run(r)
    lad = []
    for mc in (80, 96, 112, 120, 128, 136):
        p = os.path.join(rd(r), f"calib_c1_mc{mc}.csv")
        ok = os.path.exists(p) and os.path.getsize(p) > 0
        rc = [ln for ln in open(os.path.join(rd(r), "ladder_rc.txt")) if ln.startswith(f"mc={mc} ")]
        lad.append({"max_chunks": mc, "tokens": mc * 512, "rc": rc[0].split("rc=")[1].strip() if rc else "NOT_RUN"})
    c128 = pd.read_csv(os.path.join(rd(r), "calib_c1_mc128.csv"))
    ff = c128[c128.item == "f_chunk"]
    for _, x in ff.iterrows():
        frows.append(dict(run_id=r, ts=x.ts, summarize_run_id=SRID, model="llama", chunk_idx=int(x.chunk_idx),
                          pos_end=int(x.pos_end), f_ms_median=x.ms, n_reps=1, f_ms_min=x.ms, f_ms_max=x.ms,
                          source="m7_calib c1 --max-chunks 128 --reps 1 (64K ladder)", guard=v))
    write(O("f_3090.csv"), frows)
    oom_log = open(os.path.join(rd(r), "c1_mc136.log")).read()
    cons["models"]["llama"]["longest_fit"] = {"ladder": lad, "longest_ok_tokens": max(x["tokens"] for x in lad if x["rc"] == "0"),
                                              "oom_error": [ln for ln in oom_log.splitlines() if "OutOfMemoryError" in ln][:1],
                                              "f_ms_64k_reps1": ff.ms.tolist(), "src_run": r,
                                              "t_new256_ms": c128[c128.item == "t_new256"][["chunk_idx", "ms"]].to_dict("records")}

    # ---------------- GPU 忙
    brows = []
    for m in ("qwen", "llama"):
        r = R[f"busy_{m}"]
        v = check_run(r)
        b = pd.read_csv(os.path.join(rd(r), "busy.csv"))
        for (cfg, ci), g in b.groupby(["cfg", "chunk_idx"]):
            brows.append(dict(run_id=r, ts=g.ts.min(), summarize_run_id=SRID, model=m, cfg=cfg, chunk_idx=ci, n=len(g),
                              slowdown_median=round(g.slowdown.median(), 4), slowdown_min=round(g.slowdown.min(), 4),
                              slowdown_max=round(g.slowdown.max(), 4), ms_median=round(g.ms.median(), 3),
                              bg_alone_util_mean=g.bg_alone_util_mean.iloc[0], matmul_ms_ready=g.matmul_ms_ready.iloc[0],
                              bg_n_matmul=g.bg_n_matmul.iloc[0], bg_seconds=g.bg_seconds.iloc[0], guard=v))
        x = b[b.cfg == "0.7:20:8192"]
        cons["models"][m]["busy_0.7_slowdown"] = {"median_all_chunks": round(x.slowdown.median(), 4),
                                                  "by_chunk": {int(k): round(g.slowdown.median(), 4) for k, g in x.groupby("chunk_idx")},
                                                  "bg_alone_util_pct": x.bg_alone_util_mean.iloc[0], "cfg": "m7_busy --duty 0.7 --period-ms 20 --n 8192（MI300X 同一設定）",
                                                  "src_run": r}
    write(O("busy_3090.csv"), brows)

    # ---------------- 反量化 bench
    r = R["dqbench"]
    v = check_run(r)
    dq = json.load(open(os.path.join(rd(r), "dqfast_bench.json")))
    ctime = datetime.fromtimestamp(os.path.getmtime(os.path.join(rd(r), "dqfast_bench.json"))).astimezone().isoformat(timespec="seconds")
    drows = []
    for k, x in dq["bench"].items():
        shape, fmt, impl = k.split("|")
        drows.append(dict(run_id=r, ts=ctime, summarize_run_id=SRID, shape=shape, fmt=fmt, impl=impl,
                          median_ms=x.get("median_ms", NM), eff_GBps=x.get("eff_GBps", NM), error=x.get("error", ""), guard=v))
    sel = json.load(open(os.path.join(rd(r), "codec_selftest.json")))
    drows.append(dict(run_id=r, ts=ctime, summarize_run_id=SRID, shape="selftest", fmt="all", impl="m10_codec --self-test",
                      median_ms="", eff_GBps="", error="" if sel["all_pass"] else "SOME FAILED", guard=v))
    drows.append(dict(run_id=r, ts=ctime, summarize_run_id=SRID, shape="selftest", fmt="all", impl="m10_dequant_fast --self-test (bitwise)",
                      median_ms="", eff_GBps="", error="" if dq["all_pass"] else "SOME FAILED", guard=v))
    write(O("dequant_bench_3090.csv"), drows)
    cons["dequant_bench_ms_per_block"] = {k: dq["bench"][k].get("median_ms", "NOT_MEASURED: " + dq["bench"][k].get("error", "")[:200])
                                          for k in dq["bench"]}
    cons["dequant_bench_src_run"] = r

    # ---------------- 還原格子
    def grid(key):
        fr = []
        for r in R.get(key, []):
            check_run(r)
            import x7_pcie_flag                                 # 同 switch 鄰居卡的 PCIe 流量（§2.2 污染那條）
            x = x7_pcie_flag.flag(rd(r), 100.0)
            x["src_run"] = r
            fr.append(x)
        return pd.concat(fr) if fr else None

    srows = []
    allg = []
    for key in ("grid_qwen", "grid_llama", "grid_llama64k", "grid_fused"):
        g = grid(key)
        if g is None:
            continue
        g["grid"] = key
        allg.append(g)
        bad = g[g.contaminated_gpu.astype(str) != "False"]
        if len(bad):
            raise SystemExit(f"{key}: {len(bad)} 列 contaminated_gpu≠False，不進 results")
        g["lp_read"] = g.n_load_fp8 + g.n_load_int4
        for k, x in g.groupby(["model", "length", "config", "fmt", "mode", "tier", "gpu_state"]):
            lp = x.lp_read.median()
            srows.append(dict(run_id=" ".join(sorted(x.run_id.unique())), ts=x.ts.min(), summarize_run_id=SRID, grid=key,
                              model=k[0], length=k[1], config=k[2], fmt=k[3], mode=k[4], tier=k[5],
                              tier_GiBps=x.tier_GiBps.iloc[0], gpu_state=k[6], n=len(x), n_launch=x.run_id.nunique(),
                              b_median=med(x.b.tolist()), b_values=" ".join(map(str, sorted(x.b.unique()))),
                              t_restore_median_s=round(x.t_restore.median(), 5), ttft_median_s=round(x.ttft.median(), 5),
                              t_recompute_median_s=round(x.t_recompute.median(), 5), t_h2d_median_s=round(x.t_h2d.median(), 5),
                              t_dequant_median_s=round(x.t_dequant.median(), 5), lp_read_median=lp,
                              dequant_ms_per_lp_block=round(x.t_dequant.median() / lp * 1e3, 4) if lp > 0 else "",
                              t_new_turn_median_s=round(x.t_new_turn.median(), 5),
                              kv_bad_sum=pd.to_numeric(x.kv_bad, errors="coerce").fillna(0).sum(),
                              kv_checked=int(pd.to_numeric(x.kv_bad, errors="coerce").notna().sum()),
                              copy_bound_rows=int((x.copy_bound > 0).sum()), host_levels=" ".join(sorted(x.host_level.astype(str).unique())),
                              dequant_impl=x.dequant_impl.iloc[0],
                              gpu=" ".join(map(str, sorted(x.gpu.unique()))),
                              n_sibling_pcie_ge100MBps=int(x.sibling_busy.sum()),
                              sibling_max_mbps=int(x.sibling_max_mbps.max()),
                              b_median_sibling_quiet=med(x[~x.sibling_busy].b.tolist()) if (~x.sibling_busy).any() else "NOT_MEASURED",
                              t_restore_median_s_sibling_quiet=round(x[~x.sibling_busy].t_restore.median(), 5) if (~x.sibling_busy).any() else "NOT_MEASURED"))
    write(O("restore_3090_summary.csv"), srows)
    S = pd.DataFrame(srows)
    for m in ("qwen", "llama"):
        s = S[(S.model == m) & (S.length == 32768) & (S.grid == f"grid_{m}")] if len(S) else S
        if not len(s):
            why = "NOT_MEASURED: 沒有乾淨的還原格子 run（runs.json 的 rejected）"
            cons["models"][m].update(cpu_tier_meet_E0_32k=why, t_new_turn_256_s_32k=why,
                                     dequant_ms_per_block_in_restore_cpu=why, restore_grid_runs=[])
            continue
        cpu = s[(s.tier == "cpu") & (s.config == "E0") & (s["mode"] == "cake")]
        cons["models"][m]["cpu_tier_meet_E0_32k"] = {st: float(cpu[cpu.gpu_state == st].b_median.iloc[0]) for st in cpu.gpu_state}
        cons["models"][m]["t_new_turn_256_s_32k"] = {st: float(med(s[s.gpu_state == st].t_new_turn_median_s.tolist()))
                                                     for st in sorted(s.gpu_state.unique())}
        dqr = s[(s.tier == "cpu") & (s.config == "E1")]
        cons["models"][m]["dequant_ms_per_block_in_restore_cpu"] = {
            f"{r_.fmt}|{r_['mode']}|{r_.gpu_state}": r_.dequant_ms_per_lp_block for _, r_ in dqr.iterrows()}
        cons["models"][m]["restore_grid_runs"] = R.get(f"grid_{m}", [])
    json.dump(cons, open(O("constants_3090.json"), "w"), indent=1, ensure_ascii=False)
    print("wrote constants_3090.json")


if __name__ == "__main__":
    main()
