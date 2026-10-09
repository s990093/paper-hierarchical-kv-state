"""m7_round1_sim.py — 破解計劃第 1 輪 H2 延伸〔模擬〕：用某個模型的 f(i) 與 chunk 大小，跑 08 的掃描格，
加上背景版對照組 S4W，看寫入時決定能不能同時贏過所有延後版（11_round1_plan.md 的判準）。

chunk 大小要在 import 前用環境變數 M7_CHUNK_BYTES 設（m7_write_policy、m7_sim 在 import 時讀），所以每個模型一個 process：
  M7_CHUNK_BYTES=$((256<<20)) python code/m7_round1_sim.py --model longalpaca7b --f-csv results/m7_explore_mi300x/calib_c1_longalpaca7b.csv
輸出：results/m7_explore_mi300x/r1_sim_<model>.csv（每格每策略一列）
"""
import argparse
import datetime
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m7_sim  # noqa: E402
from m7_round1_analyze import load_f_csv  # noqa: E402
from m7_write_policy import CHUNK_BYTES, load_params  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "m7_explore_mi300x")
STRATS = ["S0", "S1", "S2b", "S4", "S4+", "S4+P", "S4L", "S4B", "S4W", "S4C", "S5", "S5s", "S5L", "S5P", "S5c"]
GRID = dict(cpu_gibps=[None, 11.6, 3.69, 1.0], release=["free", "hold"], workload=["chat", "doc"],
            ssd_dev=["nfs", "local"], cpu_frac=[0.25, 0.5], wl_seed=[0, 1, 2, 3, 4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--f-csv", default=None, help="不給就用第一階段的 calib_c1.csv（Llama-3.1-8B）")
    a = ap.parse_args()
    assert m7_sim.CB == CHUNK_BYTES, (m7_sim.CB, CHUNK_BYTES)
    f0, f_rid = load_f_csv(a.f_csv) if a.f_csv else (m7_sim.load_f(), "calib_c1.csv(phase1)")
    assert len(f0) >= 68, f"f(i) 只有 {len(f0)} 個 chunk，doc 負載要 68 個"
    P = load_params()
    rid = os.environ.get("RUN_ID", "no-run-id")
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    fn = os.path.join(OUT, f"r1_sim_{a.model}.csv")
    rows = []
    keys = list(GRID)
    for vals in itertools.product(*GRID.values()):
        c = dict(zip(keys, vals))
        for st in STRATS:
            sm = m7_sim.summarize(m7_sim.simulate(st, c["cpu_frac"], c["ssd_dev"], "share", 0.0, 1.0, c["wl_seed"],
                                                  c["cpu_gibps"], c["release"], c["workload"], P=P, f0=f0))
            rows.append(dict(run_id=rid, ts=ts, model=a.model, f_src=f_rid, chunk_MiB=CHUNK_BYTES / 2 ** 20,
                             cpu_gibps=c["cpu_gibps"] or P["cpu"]["read_GiBps"], **{k: c[k] for k in keys if k != "cpu_gibps"},
                             strategy=st, median=sm["median"], p90=sm["p90"], t_gate_mean=sm["t_gate_mean"],
                             w_ssd_GiB=sm["w_ssd_GiB"], n_demote=sm["n_demote"], recompute=sm["recompute"]))
    import pandas as pd
    pd.DataFrame(rows).to_csv(fn, index=False)
    print(f"{fn}: {len(rows)} 列")


if __name__ == "__main__":
    main()
