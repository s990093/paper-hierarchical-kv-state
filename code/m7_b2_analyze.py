"""08 消融 GPU 確認（b2_share.csv）的分析：GPU 實測 vs 模擬器預測，以及 08 §6＋§8 的判準。

用法：python code/m7_b2_analyze.py [b2 csv]
輸出：results/m7_write_policy_mi300x/b2_summary.csv（每格：GPU 中位數、模擬中位數、誤差）、
      b2_verdict.csv（每個設定 × 容量：寫入時決定 vs 最好的不看位置策略、vs S4B）
只統計回來的請求（round ≥ 2）；某格 rep 不齊（還在跑）就標 partial，不進判定。
"""
import datetime
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from m7_sim import load_f, load_params, simulate, summarize  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results", "m7_write_policy_mi300x")
WT = ["S5", "S5s", "S5L", "S5P", "S5c"]                     # 寫入時決定
NWT = ["S0", "S1", "S2b", "S4", "S4+", "S4+P", "S4L"]        # 寫入不看位置
TS = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")   # 分析時間；run_id 指回 GPU 原始列
CFG = ["workload", "release", "cpu_gibps", "ssd_dev", "io_model", "gap_s", "ssd_frac", "wl_seed"]


def main():
    fn = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RES, "b2_share.csv")
    d = pd.read_csv(fn)
    print(f"{fn}: {len(d)} 列, run_id: {sorted(d.run_id.unique())}")
    print(f"kv_bad 非空/非 0：{int((d.kv_bad.fillna(0) != 0).sum())}；contaminated_gpu：{int(d.contaminated_gpu.astype(bool).sum())}")
    # 每個 seed 的事件數不同（make_workload 每輪後 25% 機率不再回來：seed 0 有 32 個、seed 1 有 20 個），所以逐設定算
    n_evs = d.groupby(CFG).ev.max() + 1
    P, f0 = load_params(), load_f()
    out = []
    for key, g in d.groupby(CFG + ["cpu_frac", "strategy"]):
        c = dict(zip(CFG + ["cpu_frac", "strategy"], key))
        n_ev = n_evs[tuple(c[k] for k in CFG)]
        full = g.groupby("rep").ev.nunique()
        complete = int((full == n_ev).sum())
        gr = g[(g["round"] >= 2) & g.rep.isin(full[full == n_ev].index)]
        if gr.empty:
            continue
        sm = summarize(simulate(c["strategy"], c["cpu_frac"], c["ssd_dev"], c["io_model"], float(c["gap_s"]),
                                c["ssd_frac"], int(c["wl_seed"]), c["cpu_gibps"], c["release"], c["workload"],
                                P=P, f0=f0))
        gpu = gr.ttft.median()
        out.append(dict(run_id=";".join(sorted(g.run_id.unique())), ts=TS, **c, reps_complete=complete, n=len(gr), gpu_median=gpu, sim_median=sm["median"],
                        err=(sm["median"] - gpu) / gpu, gpu_t_gate_mean=gr.t_gate.mean(),
                        sim_t_gate_mean=sm["t_gate_mean"]))
    s = pd.DataFrame(out)
    s.to_csv(os.path.join(RES, "b2_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(s[["workload", "release", "wl_seed", "cpu_frac", "strategy", "reps_complete", "gpu_median", "sim_median",
             "err", "gpu_t_gate_mean", "sim_t_gate_mean"]].round(3).to_string(index=False))
    print(f"\n模擬誤差：|err| ≤10% 的格 {int((s.err.abs() <= 0.10).sum())}/{len(s)}，中位 {s.err.abs().median():.3f}，最大 {s.err.abs().max():.3f}")

    ver = []
    for key, g in s.groupby(CFG + ["cpu_frac"]):
        c = dict(zip(CFG + ["cpu_frac"], key))
        m = g.set_index("strategy")
        nwt = m[m.index.isin(NWT)]
        if nwt.empty or "S4B" not in m.index:
            continue
        best_n = nwt.gpu_median.idxmin()
        for w in [x for x in WT if x in m.index]:
            for src in ("gpu", "sim"):
                col = f"{src}_median"
                bn = nwt[col].min()
                ver.append(dict(run_id=m.at[w, "run_id"], ts=TS, **c, strategy=w, src=src, best_nwt=nwt[col].idxmin(), gain_nwt=(bn - m.at[w, col]) / bn,
                                gain_s4b=(m.at["S4B", col] - m.at[w, col]) / m.at["S4B", col],
                                reps=int(m.at[w, "reps_complete"])))
    v = pd.DataFrame(ver)
    if v.empty:
        print("\n還沒有可判定的設定（需要 S4B 和至少一個不看位置的策略都有完整 rep）")
        return
    v["pass"] = (v.gain_nwt >= 0.05) & (v.gain_s4b >= 0.05)
    v.to_csv(os.path.join(RES, "b2_verdict.csv"), index=False)
    print("\n判定（gain＝(對手−自己)/對手，正＝自己較快；pass＝兩個都 ≥5%）：")
    print(v[["workload", "release", "wl_seed", "cpu_frac", "strategy", "src", "best_nwt", "gain_nwt", "gain_s4b",
             "pass", "reps"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
