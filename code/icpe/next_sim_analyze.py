"""next_sim_analyze.py — ICPE next：有 GPU 層這一軸的 X1 sweep 的上限（headroom）表（PREREG §1、§4；偏離 2 第 5 項）。

  python -I code/icpe/next_sim_analyze.py --sweep <x1_sweep.csv ...> --out-dir <dir> [--pf-target cpu|gpu] [--metric mean_ret] [--dead-h 0]

定義（照 PREREG；和 x1_sim analyze 同一種「事後挑最快」）：
  一格＝(trace, rule, cpu_GiB, slow, gpu, restore, gpu_GiB, gpu_write)。每一列的做法＝(CPU 規則, GPU 規則)。
  GPU 規則：BEL＝完美預測（④ 的一部分），其他（LRU、TTLA:τ、TTLB:τ）＝不預測。GPU 層關閉（gpu_policy＝none）時只有 none。
  元件 c ∈ RIVALS（O-ORD、O-DROP、O-NOW、O-PF）：
     ④＝min over {CPU ∈ COMPONENT_POLICIES[c]} × {GPU 不預測的規則}；②＝min over {CPU ∈ RIVALS[c]} × {GPU 不預測的規則}。
  元件 O-GPU（GPU 層的完美淘汰）：④＝min over {CPU ∈ NON_ORACLE} × {BEL}；②＝min over {CPU ∈ NON_ORACLE} × {GPU 不預測的規則}。
  O-ALL(report)：④＝(O-ALL, BEL)（GPU 層關閉時 (O-ALL, none)）；②＝min over NON_ORACLE × GPU 不預測的規則。只報告。
  上限＝(T② − T④) ÷ T②；① 基準＝(LRU, LRU)（關閉時 (LRU, none)），另列 T_base。
  預取目的地：--pf-target gpu 時，有預取的 CPU 規則（O-PF、LRU+RPF、O-ALL）用 pf_target＝gpu 的列，其餘照 cpu（沒有預取就沒有差別）。
  多個重建規則（Mooncake R1／R2）時另加 rule＝min（每格取兩規則中較小的上限），同 x1_sim analyze。
  這裡只算數字，不下判定字；門檻（≥15%、5%）照 PREREG §1 由讀表的人套用。
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CODE, "p3x"))
sys.path.insert(0, CODE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import x1_sim as XS  # noqa: E402

CELL = ["trace", "rule", "cpu_GiB", "slow", "gpu", "restore", "gpu_GiB", "gpu_write"]


def load(paths, pf_target, metric, dead_h):
    d = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    for c, v in (("gpu_GiB", 0), ("gpu_write", "none"), ("gpu_policy", "none"), ("pf_target", "cpu")):
        if c not in d.columns:
            d[c] = v
    pfp = {p for p, spec in XS.POLICIES.items() if spec[3]}
    keep = np.where(d.policy.isin(pfp), d.pf_target == pf_target, d.pf_target == "cpu")
    d = d[keep].copy()
    if metric != "mean_ret":
        d["mean_ret"] = d[metric]
    comp_pol = {k: list(v) for k, v in XS.COMPONENT_POLICIES.items()}
    if dead_h:
        d = d[~d.policy.isin(["O-DROP", "O-NOW", "O-ALL"])].copy()
        d.loc[d.policy.str.endswith(f"-H{dead_h}"), "policy"] = d.policy.str.replace(f"-H{dead_h}", "", regex=False)
    d = d[~d.policy.str.contains("-H")]
    dup = d.duplicated(CELL + ["policy", "gpu_policy"]).sum()
    assert dup == 0, f"同一格同一個 (CPU 規則, GPU 規則) 有 {dup} 列重複：sweep 重複寫入？"
    return d, comp_pol


def analyze(d, comp_pol):
    rows, viol = [], []
    src = ",".join(sorted(d.run_id.unique()))
    for key, g in d.groupby(CELL, sort=True):
        T = {(r.policy, r.gpu_policy): (float(r.mean_ret), float(r.rho_w)) for r in g.itertuples()}
        gp_all = sorted({gp for _, gp in T})
        gp_no = [x for x in gp_all if x != "BEL"]
        base_g = "LRU" if "LRU" in gp_all else "none"
        base = T.get(("LRU", base_g), (np.nan, np.nan))[0]

        def best(cpus, gps):
            c = [(T[(p, q)][0], p, q) for p in cpus for q in gps if (p, q) in T]
            return min(c) if c else None
        comps = [(c, comp_pol[c], gp_no, XS.RIVALS[c], gp_no) for c in XS.RIVALS]
        if "BEL" in gp_all:
            comps.append(("O-GPU", XS.NON_ORACLE, ["BEL"], XS.NON_ORACLE, gp_no))
        for comp, ocpu, ogpu, rcpu, rgpu in comps:
            need = [(p, q) for p in ocpu for q in ogpu] + [(p, q) for p in rcpu for q in rgpu]
            if any(x not in T for x in need):
                continue                                   # 這格沒有跑齊：不算（同 x1_sim analyze）
            o, r = best(ocpu, ogpu), best(rcpu, rgpu)
            To, Tr = o[0], r[0]
            rho_o, rho_r = T[(o[1], o[2])][1], T[(r[1], r[2])][1]
            rows.append(dict(zip(CELL, key), component=comp, oracle_policy=o[1], oracle_gpu_policy=o[2], T_oracle=To,
                             best_rival=r[1], best_rival_gpu_policy=r[2], T_rival=Tr, headroom=(Tr - To) / Tr,
                             write_saturated=bool(rho_o >= 1 or rho_r >= 1), rho_w_oracle=rho_o, rho_w_rival=rho_r,
                             T_base=base, src_runs=src))
            for p in rcpu:
                for q in rgpu:
                    if T[(p, q)][0] < To - 1e-12:
                        viol.append(dict(zip(CELL, key), component=comp, oracle_policy=o[1], oracle_gpu_policy=o[2], T_oracle=To,
                                         rival=p, rival_gpu_policy=q, T_rival=T[(p, q)][0], oracle_slower_by=(To - T[(p, q)][0]) / T[(p, q)][0]))
        og = "BEL" if "BEL" in gp_all else "none"
        if ("O-ALL", og) in T and all((p, q) in T for p in XS.NON_ORACLE for q in gp_no):
            r = best(XS.NON_ORACLE, gp_no)
            To = T[("O-ALL", og)][0]
            rows.append(dict(zip(CELL, key), component="O-ALL(report)", oracle_policy="O-ALL", oracle_gpu_policy=og, T_oracle=To,
                             best_rival=r[1], best_rival_gpu_policy=r[2], T_rival=r[0], headroom=(r[0] - To) / r[0],
                             write_saturated=bool(T[("O-ALL", og)][1] >= 1 or T[(r[1], r[2])][1] >= 1),
                             rho_w_oracle=T[("O-ALL", og)][1], rho_w_rival=T[(r[1], r[2])][1], T_base=base, src_runs=src))
    H = pd.DataFrame(rows)
    if len(H) and H.rule.nunique() > 1:
        ck = [c for c in CELL if c != "rule"] + ["component"]
        HC = H[H.component != "O-ALL(report)"]
        HM = HC.loc[HC.groupby(ck).headroom.idxmin().values].copy()
        sat = HC.groupby(ck).write_saturated.any()
        HM["write_saturated"] = sat.loc[[tuple(x) for x in HM[ck].values]].values
        HM["rule_of_min"] = HM["rule"]
        HM["rule"] = "min"
        H = pd.concat([H, HM], ignore_index=True)
    return H, pd.DataFrame(viol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-prefix", default="next")
    ap.add_argument("--pf-target", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--metric", default="mean_ret")
    ap.add_argument("--dead-h", type=int, default=0)
    a = ap.parse_args()
    d, comp_pol = load(a.sweep, a.pf_target, a.metric, a.dead_h)
    H, V = analyze(d, comp_pol)
    meta = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"),
                host=socket.gethostname(), pf_target=a.pf_target, metric=a.metric, dead_h=a.dead_h)
    os.makedirs(a.out_dir, exist_ok=True)
    for name, X in (("headroom_components", H), ("violations", V)):
        X = X.copy()
        for i, (k, v) in enumerate(meta.items()):
            X.insert(i, k, v)
        X.to_csv(os.path.join(a.out_dir, f"{a.out_prefix}_{name}.csv"), index=False)
    if len(H):
        S = H[H.restore == "cake"].groupby(["trace", "rule", "component"]).headroom.agg(
            n_cells="size", max="max", median="median", n_ge15=lambda x: int((x >= 0.15).sum()), n_ge5=lambda x: int((x >= 0.05).sum()))
        pd.set_option("display.width", 250)
        print(S.to_string())
    print(f"→ {a.out_dir}/{a.out_prefix}_headroom_components.csv（{len(H)} 列）、violations（{len(V)} 列）")


if __name__ == "__main__":
    main()
