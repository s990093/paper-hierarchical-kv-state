"""m7_ablate_analyze.py — 08 消融：模擬掃描的彙整（sim_sweep.csv → sim_sweep_gain.csv）

gain_nwt   ＝最好的寫入時決定策略 vs 最好的「不看位置」策略（08 §6 預先列的 S0,S1,S2b,S4,S4+,S4+P,S4L）
gain_s4b   ＝最好的寫入時決定策略 vs S4B（同一個分界 b，等滿了才套用；08 §8 修正）
gain_brule ＝min(S4B, 寫入時決定) vs 最好的「不看位置」策略 → 「b 這個規則」本身值多少
全部用中位數，正值＝前者比較快，單位 %，分母是被比較的那一方（同 07 報告）。全部是〔模擬〕。
"""
import os
import sys

import pandas as pd

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "m7_write_policy_mi300x")
NWT = ["S0", "S1", "S2b", "S4", "S4+", "S4+P", "S4L"]
WT = ["S5", "S5s", "S5L", "S5P", "S5c"]
KEYS = ["cpu_gibps", "release", "workload", "ssd_dev", "f_scale", "wl_seed", "cpu_frac"]


def gains(d):
    rows = []
    for k, g in d.groupby(KEYS):
        m = g.set_index("strategy")["median"]
        bn, bw, b4 = m[NWT].min(), m[WT].min(), m["S4B"]
        rows.append(dict(zip(KEYS, k), best_nwt=m[NWT].idxmin(), t_nwt=bn, best_wt=m[WT].idxmin(), t_wt=bw, t_S4B=b4,
                         gain_nwt=100 * (bn - bw) / bn, gain_s4b=100 * (b4 - bw) / b4,
                         gain_brule=100 * (bn - min(bw, b4)) / bn,
                         S5_vs_S4p=100 * (m["S4+"] - m["S5"]) / m["S4+"],
                         S5P_vs_S4pP=100 * (m["S4+P"] - m["S5P"]) / m["S4+P"],
                         S5L_vs_S4L=100 * (m["S4L"] - m["S5L"]) / m["S4L"],
                         S5L_vs_S4B=100 * (b4 - m["S5L"]) / b4,
                         S5c_vs_S4p=100 * (m["S4+"] - m["S5c"]) / m["S4+"]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d = pd.read_csv(os.path.join(RES, sys.argv[1] if len(sys.argv) > 1 else "sim_sweep.csv"))
    d["cpu_gibps"] = d.cpu_gibps.round(2)
    r = gains(d).round(2)
    r.to_csv(os.path.join(RES, "sim_sweep_gain.csv"), index=False)
    pd.set_option("display.width", 250)
    base = dict(cpu_gibps=35.42, release="free", workload="chat", ssd_dev="nfs", f_scale=1.0, wl_seed=0)
    show = ["cpu_frac", "best_nwt", "t_nwt", "best_wt", "t_wt", "t_S4B", "gain_nwt", "gain_s4b", "gain_brule",
            "S5_vs_S4p", "S5L_vs_S4L", "S5c_vs_S4p"]
    for k in base:
        m = pd.Series(True, index=r.index)
        for kk, v in base.items():
            if kk != k:
                m &= r[kk] == v
        print(f"== 單一因子 {k}")
        print(r[m][[k] + show].to_string(index=False))
    print("\n== gain_nwt 前 20")
    print(r.sort_values("gain_nwt", ascending=False).head(20).to_string(index=False))
    print("\n== gain_s4b 前 20（寫入時決定 vs 延後套用同一個 b）")
    print(r.sort_values("gain_s4b", ascending=False).head(20).to_string(index=False))
    n = len(r)
    print(f"\ncells={n}  gain_nwt>=5: {(r.gain_nwt >= 5).sum()}  >=15: {(r.gain_nwt >= 15).sum()}  |  "
          f"gain_nwt>=5 AND gain_s4b>=5: {((r.gain_nwt >= 5) & (r.gain_s4b >= 5)).sum()}  |  "
          f"gain_brule>=5: {(r.gain_brule >= 5).sum()}  >=15: {(r.gain_brule >= 15).sum()}")
