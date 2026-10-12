"""m9_f7_fscale_sim.py — 後續 F7：重算變快（f×0.5、×0.6）時，第 1 輪的結論還成立嗎？〔模擬〕

為什麼：F1 發現 harness 的重算比預設 vLLM（ROCM_ATTN）慢 1.7–2.0 倍；08 的 f×0.25–0.5 掃描只比了兩個對手。
判準寫在 docs/research_20261010_followup/README.md「追加 F7」（跑之前寫的）。
做法：和 m7_round1_sim.py 同一個 GRID、同 15 個策略，只把 f(i) 乘上 f_scale；判定邏輯照抄 m7_round1_analyze.sim()，
另外加 ≥10% 的版本，free／hold 分開。

用法：python code/m9_f7_fscale_sim.py run  --scales 0.5 0.6     → results/m9_followup/f7_sim_f<scale>.csv
      python code/m9_f7_fscale_sim.py judge                     → results/m9_followup/f7_verdict.csv、f7_configs.csv
"""
import argparse
import datetime
import itertools
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m7_sim  # noqa: E402
from m7_round1_sim import GRID, STRATS  # noqa: E402
from m7_write_policy import CHUNK_BYTES, load_params  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "m9_followup")
NWT = ["S0", "S1", "S2b", "S4", "S4+", "S4+P", "S4L"]          # 不看位置（同 m7_round1_analyze 的 NWT0）
WT = ["S5", "S5s", "S5L", "S5P", "S5c"]                        # 寫入時決定
CFG = ["cpu_gibps", "release", "workload", "ssd_dev", "cpu_frac"]
RID = os.environ.get("RUN_ID", "no-run-id")
TS = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _one(args):
    scale, c, st, P, f0 = args
    sm = m7_sim.summarize(m7_sim.simulate(st, c["cpu_frac"], c["ssd_dev"], "share", 0.0, 1.0, c["wl_seed"], c["cpu_gibps"],
                                          c["release"], c["workload"], f_scale=scale, P=P, f0=f0))
    return dict(run_id=RID, ts=TS, model="llama31_8b", f_scale=scale, chunk_MiB=CHUNK_BYTES / 2 ** 20,
                cpu_gibps=c["cpu_gibps"] or P["cpu"]["read_GiBps"], **{k: c[k] for k in GRID if k != "cpu_gibps"},
                strategy=st, median=sm["median"], mean=sm["mean"], p90=sm["p90"], w_ssd_GiB=sm["w_ssd_GiB"],
                n_demote=sm["n_demote"], recompute=sm["recompute"])


def run(scales):
    import pandas as pd
    P, f0 = load_params(), m7_sim.load_f()
    assert len(f0) >= 68, len(f0)
    keys = list(GRID)
    cells = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]
    for s in scales:
        jobs = [(s, c, st, P, f0) for c in cells for st in STRATS]
        with Pool(32) as p:
            rows = p.map(_one, jobs, chunksize=8)
        fn = os.path.join(OUT, f"f7_sim_f{s}.csv")
        pd.DataFrame(rows).to_csv(fn, index=False)
        print(f"{fn}: {len(rows)} 列", flush=True)


def judge():
    import glob
    import pandas as pd
    out = []
    for fn in sorted(glob.glob(os.path.join(OUT, "f7_sim_f*.csv"))):
        d = pd.read_csv(fn)
        for key, g in d.groupby(["f_scale"] + CFG + ["wl_seed"]):
            m = g.set_index("strategy")["median"]
            bn = m[NWT].min()
            wt = m[WT]
            r = dict(run_id=g.run_id.iloc[0], ts=TS, **dict(zip(["f_scale"] + CFG + ["wl_seed"], key)),
                     best_nwt=m[NWT].idxmin(), best_wt=wt.idxmin(), t_wt=wt.min(), t_nwt=bn,
                     g_nwt=(bn - wt.min()) / bn, g_S4B=(m["S4B"] - wt.min()) / m["S4B"],
                     g_S4W=(m["S4W"] - wt.min()) / m["S4W"], g_S4C=(m["S4C"] - wt.min()) / m["S4C"])
            gmin = min(r["g_nwt"], r["g_S4B"], r["g_S4W"], r["g_S4C"])
            r["g_min"] = gmin
            r["pass5"], r["pass10"] = gmin >= 0.05, gmin >= 0.10
            r["pass5_nwt_S4B"] = min(r["g_nwt"], r["g_S4B"]) >= 0.05      # 08 當時只比這兩個（對照用）
            out.append(r)
    v = pd.DataFrame(out)
    v.to_csv(os.path.join(OUT, "f7_verdict.csv"), index=False)
    a = v.groupby(["f_scale"] + CFG).agg(seeds=("wl_seed", "nunique"), n_pass5=("pass5", "sum"), n_pass10=("pass10", "sum"),
                                          n_pass5_2twins=("pass5_nwt_S4B", "sum"), g_min_med=("g_min", "median"),
                                          g_nwt_med=("g_nwt", "median")).reset_index()
    a["survive5"], a["survive10"] = a.n_pass5 >= 3, a.n_pass10 >= 3
    a["survive5_2twins"] = a.n_pass5_2twins >= 3
    a.insert(0, "ts", TS)
    a.insert(0, "run_id", RID)
    a.to_csv(os.path.join(OUT, "f7_configs.csv"), index=False)
    pd.set_option("display.width", 250)
    for (s, rel), g in a.groupby(["f_scale", "release"]):
        print(f"\n== f×{s}，{rel}：{len(g)} 個設定；存活（≥3/5 seed）— 全部對手 ≥5%：{int(g.survive5.sum())}，"
              f"≥10%：{int(g.survive10.sum())}；只比 08 的兩個對手 ≥5%：{int(g.survive5_2twins.sum())}")
        top = g.sort_values(["n_pass10", "n_pass5", "g_min_med"], ascending=False).head(6)
        print(top[CFG + ["n_pass5", "n_pass10", "n_pass5_2twins", "g_min_med", "g_nwt_med"]].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "judge"])
    ap.add_argument("--scales", type=float, nargs="*", default=[0.5, 0.6])
    a = ap.parse_args()
    run(a.scales) if a.cmd == "run" else judge()
