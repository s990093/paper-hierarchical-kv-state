"""〔估計〕What if recompute were as fast as default vLLM?  (REPORT.md Sec.5.5, threat to validity)

Uses only files under results/:
* f(i) per 512-token chunk, GPU idle: results/m10_precision_mi300x/speed_<m>/bw.json (f_ms_median_by_chunk)
* throttled-tier bandwidths b16/b32/b48: same bw.json (targets.*.GiBps)
* bytes per block (BF16 / INT4-KIVI): results/m10_tier_mi300x/iowall.json (bytes_per_block)
* real-CPU-tier load-all medians (E0, E1 INT4, original dequant): results/m10_precision_mi300x/speed_<m>/summary_int4.csv
* harness compute-only 32K idle: same summary (mode compute_only)
* guarded vLLM default-chunk TTFT 32K: results/m10_precision_mi300x/vllm_ttft_guarded2/vllm_ttft.csv
k = harness compute-only / vLLM default TTFT; f is divided by k. Load time per block on a throttled tier = bytes / B
(dequant ignored there); on the CPU tier = load-all median / 64. The two-line simulation is code/m10_sweep_sim.simulate
(validated against measured meets in results/m10_tier_mi300x/iowall.json A2). Output: derived/whatif_recompute.json.
"""
from __future__ import annotations

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DERIVED, load_json, read_csv, rp  # noqa: E402

import m10_sweep_sim as sim  # noqa: E402

GiB = 1 << 30


def main():
    io = load_json(rp("m10_tier_mi300x", "iowall.json"))
    vl = read_csv(rp("m10_precision_mi300x", "vllm_ttft_guarded2", "vllm_ttft.csv"))
    out = {}
    for m in ("llama", "qwen"):
        bw = load_json(rp("m10_precision_mi300x", f"speed_{m}", "bw.json"))
        f = [x / 1000 for x in bw["f_ms_median_by_chunk"]]
        summ = [r for r in read_csv(rp("m10_precision_mi300x", f"speed_{m}", "summary_int4.csv"))
                if r["length"] == "32768"]
        C = next(float(r["median_t_restore"]) for r in summ if r["mode"] == "compute_only" and r["gpu_state"] == "idle"
                 and r["bw_name"] == "b16")
        v = statistics.median(float(r["ttft_s"]) for r in vl if r["model"] == m and r["chunking"] == "default"
                              and r["length"] == "32768")
        k = C / v
        bpb = io[m]["bytes_per_block"]
        res = {"harness_compute_only_32K_idle_s": C, "vllm_default_ttft_32K_s": v, "k": round(k, 3), "cells": []}
        for scale_name, s in (("measured_f", 1.0), ("f_div_k", 1.0 / k)):
            ff = [x * s for x in f]
            for tier in ("b16", "b32", "b48"):
                B = bw["targets"][tier]["GiBps"] * GiB
                for cfg, nbytes in (("E0", bpb["bf16"]), ("E1_int4", bpb["int4"])):
                    load = [nbytes / B] * len(ff)
                    b, t, tr, tl = sim.simulate(ff, load)
                    res["cells"].append({"f": scale_name, "tier": tier, "config": cfg, "meet_b": b, "t_cake_s": round(t, 4),
                                         "t_load_all_s": round(sum(load), 4), "t_compute_only_s": round(sum(ff), 4),
                                         "F25_blocks_read_if_lp": sum(1 for i in range(1, 17) if i >= b)})
            for cfg, sc in (("E0", "E0"), ("E1_int4", "E1")):
                L = next(float(r["median_t_restore"]) for r in summ if r["mode"] == "load_only" and r["bw_name"] == "cpu"
                         and r["gpu_state"] == "idle" and r["config"] == sc)
                load = [L / len(ff)] * len(ff)
                b, t, tr, tl = sim.simulate(ff, load)
                res["cells"].append({"f": scale_name, "tier": "cpu", "config": cfg, "meet_b": b, "t_cake_s": round(t, 4),
                                     "t_load_all_s": round(L, 4), "t_compute_only_s": round(sum(ff), 4),
                                     "cake_floor_first_block_s": round(ff[0], 4)})
        out[m] = res
    os.makedirs(DERIVED, exist_ok=True)
    with open(os.path.join(DERIVED, "whatif_recompute.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    for m, r in out.items():
        print(m, "k=", r["k"])
        for c in r["cells"]:
            print("  ", c)


if __name__ == "__main__":
    main()
