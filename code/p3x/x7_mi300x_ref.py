"""x7_mi300x_ref.py — X7：從 repo 裡已提交的 MI300X 結果檔讀出要和 3090 並列的常數（不手抄），寫 mi300x_ref.csv＋.json。只用 CPU。

  python code/p3x/x7_mi300x_ref.py --out-dir results/p3_explore/x7
每列：quantity、model、gpu_state、value、unit、source_file、run_id（來源檔裡的 run_id）、ts（來源列最早的 ts，沒有就空）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(REPO, "results")
SRID = os.environ.get("RUN_ID", "NO_RUN_ID")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    rows = []

    def add(q, model, st, val, unit, src, rid, ts=""):
        rows.append(dict(run_id=rid, ts=ts, summarize_run_id=SRID, quantity=q, model=model, gpu_state=st,
                         value=val, unit=unit, source_file=os.path.relpath(src, REPO)))

    ref = {}
    for m in ("llama", "qwen"):
        p = os.path.join(RES, "m10_precision_mi300x", f"speed_{m}", "bw.json")
        bw = json.load(open(p))
        f = bw["f_ms_median_by_chunk"]
        cf = pd.read_csv(os.path.join(RES, "m10_precision_mi300x", f"speed_{m}", "calib_f.csv"))
        t0 = cf.ts.min()
        add("f0_ms", m, "idle", f[0], "ms", p, bw["run_id"], t0)
        add("f63_ms", m, "idle", f[63], "ms", p, bw["run_id"], t0)
        add("f_slope_ms_per_block", m, "idle", round((f[63] - f[0]) / 63, 4), "ms/block", p, bw["run_id"], t0)
        add("f_sum_32k_s", m, "idle", bw["f_sum_s"], "s", p, bw["run_id"], t0)
        add("h2d_pinned_chunk_view_GiBps", m, "idle", bw["real_pinned_h2d_GiBps"], "GiB/s", p, bw["run_id"], t0)
        add("ell_cpu_real_ms", m, "idle", round(bw["chunk_bytes"] / (bw["real_pinned_h2d_GiBps"] * (1 << 30)) * 1e3, 4), "ms", p, bw["run_id"], t0)
        for k, v in bw["targets"].items():
            add(f"bstar_{k}_GiBps", m, "idle", v["GiBps"], "GiB/s", p, bw["run_id"], t0)
        ref[m] = {"f_ms_idle_32k": f, "chunk_bytes": bw["chunk_bytes"], "f_src_run": bw["run_id"],
                  "h2d_GiBps": bw["real_pinned_h2d_GiBps"], "bstar": {k: v["GiBps"] for k, v in bw["targets"].items()}}
        mp = os.path.join(RES, "m10_precision_mi300x", f"speed_{m}", "meets.json")
        mj = json.load(open(mp))
        for cfg in ("E0", "E1", "E1@int4_kivi"):
            for tier, e in mj["meets"].get(cfg, {}).items():
                for st in ("idle", "busy"):
                    if st in e:
                        add(f"meet_{cfg}_{tier}", m, st, e[st], "blocks", mp, " ".join(mj["source_run_ids"]))
        ref[m]["meets"] = {cfg: {t: {s: v for s, v in e.items() if not s.endswith("_all")} for t, e in mj["meets"][cfg].items()}
                           for cfg in ("E0", "E1", "E1@int4_kivi") if cfg in mj["meets"]}
        # CPU 層真實 H2D 的還原（32K）：中位數與每個低精度 block 的反量化時間
        d = pd.read_csv(os.path.join(RES, "m10_precision_mi300x", f"speed_{m}", "restores.csv.gz"), low_memory=False)
        d = d[(d.length == 32768) & (d.bw_name == "cpu") & (d.config.isin(["E0", "E1"])) & (d.run_id.str.contains("grid-(fp8|int4)"))]
        rp = os.path.join(RES, "m10_precision_mi300x", f"speed_{m}", "restores.csv.gz")
        for (rid, fmt, cfg, mode, st), g in d.groupby(["run_id", "fmt", "config", "mode", "gpu_state"]):
            tag = f"{cfg}{'' if cfg == 'E0' else '@' + fmt}|{mode}"
            add(f"cpu32k_t_restore_s|{tag}", m, st, round(g.t_restore.median(), 5), "s", rp, rid, g.ts.min())
            add(f"cpu32k_b|{tag}", m, st, g.b.median(), "blocks", rp, rid, g.ts.min())
            add(f"cpu32k_t_new_turn_s|{tag}", m, st, round(g.t_new_turn.median(), 5), "s", rp, rid, g.ts.min())
            if cfg == "E1":
                lp = g.fp8_read.median()
                add(f"cpu32k_dequant_ms_per_block|{tag}", m, st, round(g.t_dequant.median() / lp * 1e3, 4), "ms", rp, rid, g.ts.min())
    # 第一階段：C0、duplex、C1（Llama）、C5、tier_params
    base = os.path.join(RES, "m7_write_policy_mi300x")
    c0 = pd.read_csv(os.path.join(base, "calib_c0.csv"))
    b = c0[(c0.kind == "bw") & (c0.rep != 0)].copy()
    b["value"] = b.value.astype(float)
    for (item, size), g in b.groupby(["item", "size_mib"]):
        add(f"c0_{item}_{int(size)}MiB_median_GiBps", "-", "idle", round(g.value.median(), 3), "GiB/s",
            os.path.join(base, "calib_c0.csv"), " ".join(g.run_id.unique()), g.ts.min())
    dx = pd.read_csv(os.path.join(base, "calib_c0_duplex.csv"))
    for case, g in dx.groupby("case"):
        add(f"duplex_h2d_{case}_median_GiBps", "-", "idle", round(g.h2d_GiBps.median(), 3), "GiB/s",
            os.path.join(base, "calib_c0_duplex.csv"), " ".join(g.run_id.unique()), g.ts.min())
    c1 = pd.read_csv(os.path.join(base, "calib_c1.csv"))
    f1 = c1[c1.item == "f_chunk"].groupby("chunk_idx").ms.median()
    add("phase1_c1_f0_ms", "llama", "idle", round(f1.iloc[0], 3), "ms", os.path.join(base, "calib_c1.csv"), " ".join(c1.run_id.unique()), c1.ts.min())
    add("phase1_c1_f_last_ms", "llama", "idle", round(f1.iloc[-1], 3), "ms", os.path.join(base, "calib_c1.csv"), " ".join(c1.run_id.unique()), c1.ts.min())
    tn = c1[c1.item == "t_new256"].groupby("chunk_idx").ms.median()
    for k, v in tn.items():
        add(f"phase1_c1_t_new256_ms_at_block{k}", "llama", "idle", round(v, 3), "ms", os.path.join(base, "calib_c1.csv"), " ".join(c1.run_id.unique()))
    ref["phase1_llama_f_ms"] = f1.tolist()
    ref["phase1_llama_t_new256_ms"] = {int(k): float(v) for k, v in tn.items()}
    c5 = pd.read_csv(os.path.join(base, "calib_c5.csv"))
    for duty, g in c5.groupby("duty"):
        add(f"c5_slowdown_duty{duty}", "llama", "busy" if duty else "idle", round(g.slowdown.median(), 4), "x",
            os.path.join(base, "calib_c5.csv"), " ".join(g.run_id.unique()), g.ts.min())
    tp = json.load(open(os.path.join(base, "tier_params.json")))
    for t in ("cpu", "local", "nfs"):
        add(f"tier_params_{t}_read_GiBps", "-", "idle", round(tp[t]["read_GiBps"], 5), "GiB/s", os.path.join(base, "tier_params.json"),
            " ".join(tp["source_runs"]))
    a0 = pd.read_csv(os.path.join(base, "summary_a0_speedup.csv"))
    ref["a0_32k"] = a0[a0.L == 32768][["bw", "GiBps", "ttft_compute", "ttft_load", "ttft_cake", "meet_measured"]].to_dict("records")
    ref["a0_src"] = os.path.relpath(os.path.join(base, "summary_a0_speedup.csv"), REPO) + " (run 20261008-133544-m7-a0)"
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "mi300x_ref.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump(ref, open(os.path.join(a.out_dir, "mi300x_ref.json"), "w"), indent=1, ensure_ascii=False)
    print("rows", len(rows))


if __name__ == "__main__":
    main()
