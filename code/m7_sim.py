"""m7_sim.py — B 實驗的虛擬時鐘模擬器（08_ablation_plan.md §5）

和 GPU harness 用**同一份** BState（寫入／淘汰）與同一個限速器 Tier，只把時間換成虛擬的：
  重算 chunk i  → f[i]（C1 實測中位數，GPU 閒）× f_scale
  新一輪 prefill → sum f[n_old:n_new]
  Cake           → 重算線從前、載入線從後，誰先空出來誰先拿下一個 chunk（和 Restorer.restore 的指標鎖同義）
  請求之間       → capture（D2H，C0 實測 44.8 GiB/s）＋ ov_between
沒有 GPU 雜訊、Python 開銷、執行緒同步；所以只拿來挑設定，不拿來下結論（08 §6）。

子命令：
  validate  逐格重現第一階段 B 的 9,300 列，輸出 sim_validate.csv
  sweep     08 §2 的單一因子掃描與組合，輸出 sim_sweep.csv
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m7_restore_harness import GiB, Tier  # noqa: E402
from m7_write_policy import RES, SCHED, BState, load_f, load_params, make_workload  # noqa: E402

MiB = 1 << 20
CB = int(os.environ.get("M7_CHUNK_BYTES", 64 * MiB))   # 換模型時和 m7_write_policy 一起設
D2H_BPS = 44.8 * GiB      # C0（calib_c0.csv）D2H 實測中位數，只用於 capture 的時間


class _Pool:
    def put(self, t):
        pass


def mk_tier(name, p, io_model, cpu_gibps=None):
    if name == "cpu" and cpu_gibps:
        p = dict(p, read_GiBps=cpu_gibps, write_GiBps=cpu_gibps)
    return Tier(name, read_Bps=p["read_GiBps"] * GiB, write_Bps=p["write_GiBps"] * GiB, c_s=p.get("c_ms", 0) / 1e3,
                mode=io_model, k_read=p.get("k_read", 1.0) if io_model == "share" else 1.0)


def sim_restore(n, loc, f, t0, mode):
    """回傳 (完成時間, meet, n_load_cpu, n_load_ssd)。loc[i] 是 Tier 或 None。"""
    if n == 0:
        return t0, 0, 0, 0
    nc = ns = 0
    if mode == "load_only":
        t, i = t0, 0
        while i < n and loc[i] is not None:
            t = loc[i].reserve_read(CB, t)
            if loc[i].name.startswith("cpu"):
                nc += 1
            else:
                ns += 1
            i += 1
        meet = i
        for j in range(i, n):
            t += f[j]
        return t, meet, nc, ns
    p, q, rt, lt, lstop = 0, n - 1, t0, t0, False
    while p <= q:
        if lstop or rt <= lt:
            rt += f[p]
            p += 1
        else:
            if loc[q] is None:
                lstop = True
                continue
            lt = loc[q].reserve_read(CB, lt)
            if loc[q].name.startswith("cpu"):
                nc += 1
            else:
                ns += 1
            q -= 1
    return max(rt, lt), p, nc, ns


def simulate(strategy, cpu_frac, ssd_dev="nfs", io_model="share", gap=0.0, ssd_frac=1.0, wl_seed=0,
             cpu_gibps=None, release="free", workload="chat", f_scale=1.0, ov_between=0.01, P=None, f0=None):
    P = P or load_params()
    f = [x * f_scale for x in (f0 or load_f())]
    sched = SCHED[workload]
    n_sess = 8
    total = n_sess * sum(sched)
    tiers = {"cpu": mk_tier("cpu", P["cpu"], io_model, cpu_gibps), "ssd": mk_tier(f"ssd_{ssd_dev}", P[ssd_dev], io_model)}
    st = BState(strategy, int(cpu_frac * total), None if ssd_frac >= 1 else int(ssd_frac * total), tiers, f, _Pool(), P,
                release=release)
    ev, _ = make_workload(wl_seed, n_sess, len(sched))
    t = 100.0     # 虛擬時鐘從 100 s 開始（避開 t_free=0 的初值）
    rows = []
    for e, (s, r) in enumerate(ev):
        n_old = sum(sched[:r - 1])
        n_new = n_old + sched[r - 1]
        _, loc = st.layout(s, n_old)
        t_arr = t
        t_go = st.gate(t_arr)
        mode = "load_only" if strategy == "R0" else "cake"
        t_end, meet, nc, ns = sim_restore(n_old, loc, f, t_go, mode)
        t_new = sum(f[n_old:n_new])
        ttft = t_end + t_new - t_arr
        t = t_arr + ttft + (n_new - n_old) * CB / D2H_BPS
        st.after_round(s, n_old, n_new, {i: None for i in range(n_old, n_new)}, t, e)
        rows.append(dict(ev=e, session=s, round=r, hist_chunks=n_old, ttft=ttft, t_gate=t_go - t_arr, meet=meet,
                         n_load_cpu=nc, n_load_ssd=ns, n_missing=sum(1 for x in loc if x is None),
                         w_ssd_GiB=st.w_bytes["ssd"] / GiB, n_demote=st.n_demote))
        t += ov_between + gap
    return rows


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return (xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])) if n else float("nan")


def summarize(rows):
    ret = [r["ttft"] for r in rows if r["round"] >= 2]
    ret.sort()
    return dict(median=median(ret), mean=sum(ret) / len(ret), p90=ret[int(0.9 * (len(ret) - 1))],
                t_gate_mean=sum(r["t_gate"] for r in rows if r["round"] >= 2) / len(ret),
                w_ssd_GiB=rows[-1]["w_ssd_GiB"], n_demote=rows[-1]["n_demote"],
                ld_ssd=sum(r["n_load_ssd"] for r in rows), recompute=sum(r["meet"] for r in rows if r["round"] >= 2))


# ------------------------------------------------------------------ validate
def validate(a):
    import pandas as pd
    P, f0 = load_params(), load_f()
    srcs = [("b_share.csv", "share", None), ("b_fifo.csv", "fifo", None), ("b_share_cpu3.69.csv", "share", 3.69)]
    out = []
    for fn, io, cg in srcs:
        d = pd.read_csv(os.path.join(RES, fn))
        d = d[d["round"] >= 2]
        for key, g in d.groupby(["ssd_dev", "gap_s", "ssd_frac", "wl_seed", "cpu_frac", "strategy"]):
            ssd_dev, gap, sf, seed, cf, strat = key
            if strat in ("S3",):   # S3 不存 → 和其他策略同一套程式，照樣模擬
                pass
            rows = simulate(strat, cf, ssd_dev, io, float(gap), sf, int(seed), cg, P=P, f0=f0, ov_between=a.ov)
            sm = summarize(rows)
            out.append(dict(io_model=io, cpu_gibps=cg or P["cpu"]["read_GiBps"], ssd_dev=ssd_dev, gap_s=gap,
                            ssd_frac=sf, wl_seed=seed, cpu_frac=cf, strategy=strat, reps=g.rep.nunique(),
                            meas_median=g.ttft.median(), sim_median=sm["median"],
                            meas_mean=g.ttft.mean(), sim_mean=sm["mean"],
                            err_median_pct=100 * (sm["median"] / g.ttft.median() - 1)))
    df = pd.DataFrame(out)
    df.insert(0, "ts", datetime.now().astimezone().isoformat(timespec="seconds"))
    df.insert(0, "run_id", os.environ.get("RUN_ID", "adhoc"))
    df.to_csv(os.path.join(RES, "sim_validate.csv"), index=False)
    w = df.err_median_pct.abs()
    print(df.to_string(max_rows=400))
    print(f"cells={len(df)}  |err|<=10%: {(w <= 10).mean():.1%}  median|err|={w.median():.1f}%  max={w.max():.1f}%")
    # 判準第二部分：實測 S5 vs 最好簡單策略 >5% 的格，方向要一致
    SIMPLE = ["S0", "S1", "S2b", "S4", "S4+"]
    agree = []
    for key, g in df.groupby(["io_model", "cpu_gibps", "ssd_dev", "gap_s", "ssd_frac", "wl_seed", "cpu_frac"]):
        g = g.set_index("strategy")
        simp = [s for s in SIMPLE if s in g.index]
        if "S5" not in g.index or not simp:
            continue
        bm = g.loc[simp, "meas_median"].min()
        bs = g.loc[simp, "sim_median"].min()
        dm = 100 * (bm / g.loc["S5", "meas_median"] - 1)   # 正＝S5 比較快
        ds = 100 * (bs / g.loc["S5", "sim_median"] - 1)
        agree.append(dict(cfg=key, meas_S5_vs_best=round(dm, 1), sim_S5_vs_best=round(ds, 1),
                          same_sign=(abs(dm) <= 5) or (dm * ds > 0)))
    ag = pd.DataFrame(agree)
    print(ag.to_string())
    print("direction agreement where |meas|>5%:", ag[ag.meas_S5_vs_best.abs() > 5].same_sign.mean())
    json.dump(dict(cells=len(df), frac_within_10pct=float((w <= 10).mean()), median_abs_err=float(w.median()),
                   max_abs_err=float(w.max()), direction=ag.astype({"cfg": str}).to_dict("records")),
              open(os.path.join(RES, "sim_validate.json"), "w"), indent=1, default=str)


# ------------------------------------------------------------------ sweep
STRATS = ["S0", "S1", "S2b", "S4", "S4+", "S4+P", "S4L", "S4B", "S4W", "S4C", "S5", "S5s", "S5L", "S5P", "S5c"]
BASE = dict(ssd_dev="nfs", io_model="share", gap=0.0, ssd_frac=1.0, wl_seed=0, cpu_gibps=None, release="free",
            workload="chat", f_scale=1.0)
FACTORS = {
    "cpu_gibps": [None, 11.6, 3.69, 1.0],
    "release": ["free", "hold"],
    "workload": ["chat", "doc"],
    "ssd_dev": ["nfs", "local"],
    "f_scale": [0.25, 0.5, 1.0, 2.0],
    "wl_seed": [0, 1, 2],
}


def sweep(a):
    import pandas as pd
    P, f0 = load_params(), load_f()
    cfgs = []
    for k, vals in FACTORS.items():           # 單一因子
        for v in vals:
            cfgs.append(dict(BASE, **{k: v}))
    combo_keys = ["cpu_gibps", "release", "workload", "ssd_dev", "f_scale"]
    for k1, k2 in itertools.combinations(combo_keys, 2):   # 兩兩組合（非基準水準）
        for v1 in FACTORS[k1]:
            for v2 in FACTORS[k2]:
                if v1 != BASE[k1] and v2 != BASE[k2]:
                    cfgs.append(dict(BASE, **{k1: v1, k2: v2}))
    for vals in itertools.product(*[FACTORS[k] for k in combo_keys]):   # 全組合（5 個因子）
        cfgs.append(dict(BASE, **dict(zip(combo_keys, vals))))
    seen, uniq = set(), []
    for c in cfgs:
        key = tuple(sorted(c.items(), key=lambda x: x[0]))
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    out = []
    for i, c in enumerate(uniq):
        for cf in a.cpu_fracs:
            for strat in STRATS:
                sm = summarize(simulate(strat, cf, P=P, f0=f0, ov_between=a.ov, **c))
                out.append(dict(cfg_id=i, **c, cpu_frac=cf, strategy=strat, **sm))
        if i % 50 == 0:
            print(i, "/", len(uniq), flush=True)
    df = pd.DataFrame(out)
    df["cpu_gibps"] = df.cpu_gibps.fillna(P["cpu"]["read_GiBps"])
    df.insert(0, "ts", datetime.now().astimezone().isoformat(timespec="seconds"))
    df.insert(0, "run_id", os.environ.get("RUN_ID", "adhoc"))
    df.to_csv(os.path.join(RES, "sim_sweep.csv"), index=False)
    print("configs", len(uniq), "rows", len(df))


def probe(a):
    """單一設定 × 多個 seed／容量／SSD，看候選設定穩不穩（輸出 sim_probe.csv，附加）。"""
    import pandas as pd
    P, f0 = load_params(), load_f()
    out = []
    for seed in a.seeds:
        for dev in a.ssd_devs:
            for cf in a.cpu_fracs:
                for strat in STRATS:
                    c = dict(BASE, cpu_gibps=a.cpu_gibps, release=a.release, workload=a.workload, ssd_dev=dev,
                             f_scale=a.f_scale, wl_seed=seed)
                    out.append(dict(c, cpu_frac=cf, strategy=strat,
                                    **summarize(simulate(strat, cf, P=P, f0=f0, ov_between=a.ov, **c))))
    df = pd.DataFrame(out)
    df.insert(0, "ts", datetime.now().astimezone().isoformat(timespec="seconds"))
    df.insert(0, "run_id", os.environ.get("RUN_ID", "adhoc"))
    path = os.path.join(RES, "sim_probe.csv")
    df.to_csv(path, index=False, mode="a", header=not os.path.exists(path))
    pd.set_option("display.width", 250)
    print(df.pivot_table(index=["wl_seed", "ssd_dev", "cpu_frac"], columns="strategy", values="median").round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "sweep", "probe"])
    ap.add_argument("--cpu-gibps", type=float, default=None)
    ap.add_argument("--release", default="free")
    ap.add_argument("--workload", default="chat")
    ap.add_argument("--f-scale", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--ssd-devs", nargs="*", default=["nfs", "local"])
    ap.add_argument("--ov", type=float, default=0.01, help="請求之間的額外開銷（秒）")
    ap.add_argument("--cpu-fracs", type=float, nargs="*", default=[0.125, 0.25, 0.5])
    a = ap.parse_args()
    {"validate": validate, "sweep": sweep, "probe": probe}[a.cmd](a)


if __name__ == "__main__":
    main()
