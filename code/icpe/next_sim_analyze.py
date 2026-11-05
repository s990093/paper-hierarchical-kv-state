"""next_sim_analyze.py — ICPE next：有 GPU 層這一軸的 X1 sweep 的上限（headroom）表（PREREG §1、§4；偏離 2 第 5 項）。

  python -I code/icpe/next_sim_analyze.py --sweep <x1_sweep.csv ...> --out-dir <dir> [--pf-target cpu|gpu] [--metric mean_ret] [--dead-h 0]

定義（照 PREREG；和 x1_sim analyze 同一種「事後挑最快」）：
  一格＝(trace, rule, cpu_GiB, slow, gpu, restore, gpu_GiB, gpu_write)。每一列的做法＝(CPU 規則, GPU 規則)。
  GPU 規則：BEL＝完美預測（④ 的一部分），其他（LRU、TTLA:τ、TTLB:τ）＝不預測。GPU 層關閉（gpu_policy＝none）時只有 none。
  元件 c ∈ RIVALS（O-ORD、O-DROP、O-NOW、O-PF）：
     ④＝min over {CPU ∈ COMPONENT_POLICIES[c]} × {GPU 不預測的規則}；②＝min over {CPU ∈ RIVALS[c]} × {GPU 不預測的規則}。
  元件 O-GPU（GPU 層的完美淘汰）：④＝min over {CPU ∈ NON_ORACLE} × {BEL}；②＝min over {CPU ∈ NON_ORACLE} × {GPU 不預測的規則}。
  元件 O-SLOW（慢層容量上限時的完美淘汰；第二批）：④＝min over {CPU 不預測} × {GPU 不預測} × {慢層 BEL}；
     ②＝min over {CPU 不預測} × {GPU 不預測} × {慢層 LRU、LRU2、ARC、S3FIFO、GDSF、PENS}。其他元件的慢層一律取慢層不預測規則裡最快的。
  O-ALL(report)：④＝(O-ALL, BEL, BEL)（關閉的層用 none）；②＝min over 全部不預測的組合。只報告。
  --rival-only-gpu <GPU 規則…>（PREREG 偏離 4 第 2 項的補跑）：這些 GPU 規則只放進 ②，不放進 ④（④ 的 GPU 規則集合維持原本的）。
     影響 O-ORD／O-DROP／O-NOW／O-PF 與 O-SLOW 的 ④；O-GPU 的 ④ 只用 BEL、O-ALL(report) 的 ④ 是固定組合，本來就不受影響。
     不給（預設）＝和之前完全相同。給了會自檢：拿掉這些規則的列重算一次，兩邊的 (格, 元件) 集合必須相同、
     每一列的上限都不能變大（② 只多不少、④ 不變），不過就停。
  上限＝(T② − T④) ÷ T②；① 基準＝(LRU, LRU)（關閉時 (LRU, none)），另列 T_base。
  預取目的地：--pf-target gpu 時，有預取的 CPU 規則（O-PF、LRU+RPF、O-ALL）用 pf_target＝gpu 的列，其餘照 cpu（沒有預取就沒有差別）。
  多個重建規則（Mooncake R1／R2）時另加 rule＝min（每格取兩規則中較小的上限），同 x1_sim analyze。
  這裡只算數字，不下判定字；門檻（≥15%、5%）照 PREREG §1 由讀表的人套用。
  W4 雜訊（sweep 有 noise_* 欄時）：每個雜訊設定 (模型, λ, 抽法, seed) 各算一次——用到完美預知的組合用那個設定的列、
  其他（②）用沒有雜訊的列；輸出 <prefix>_noise_headroom.csv（逐 seed）與 <prefix>_noise_summary.csv（seed 間的中位數、
  最小–最大、平均±標準差；PREREG §3 B）。λ＝0 的上限必須和沒有雜訊的相同（自檢，不同就停）。
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

CELL = ["trace", "rule", "cpu_GiB", "slow", "gpu", "restore", "gpu_GiB", "gpu_write", "slow_cap_frac"]
NZ = ["noise_model", "noise_lambda", "noise_draw", "noise_seed"]


def load(paths, pf_target, metric, dead_h):
    d = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    for c, v in (("gpu_GiB", 0), ("gpu_write", "none"), ("gpu_policy", "none"), ("pf_target", "cpu"),
                 ("slow_cap_frac", 0.0), ("slow_policy", "none"),
                 ("noise_model", "none"), ("noise_lambda", 0.0), ("noise_draw", "none"), ("noise_seed", -1)):
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
    dup = d.duplicated(CELL + ["policy", "gpu_policy", "slow_policy"] + NZ).sum()
    assert dup == 0, f"同一格同一個 (CPU 規則, GPU 規則) 有 {dup} 列重複：sweep 重複寫入？"
    return d, comp_pol


def analyze(d, comp_pol, oracle_keys=None, rival_only=()):
    """oracle_keys（雜訊用）：只算「④ 那一邊的組合全部都在 oracle_keys 裡」的元件（避免拿沒有雜訊的列混進來）。
    rival_only：只放進 ②、不放進 ④ 的 GPU 規則（--rival-only-gpu）。"""
    rows, viol = [], []
    src = ",".join(sorted(d.run_id.unique()))
    for key, g in d.groupby(CELL, sort=True):
        T = {(r.policy, r.gpu_policy, r.slow_policy): (float(r.mean_ret), float(r.rho_w)) for r in g.itertuples()}
        gp_all = sorted({k[1] for k in T})
        sp_all = sorted({k[2] for k in T})
        gp_no = [x for x in gp_all if x != "BEL"]
        gp_or = [x for x in gp_no if x not in rival_only]     # ④ 用的 GPU 不預測規則（沒給 rival_only 時＝gp_no）
        sp_no = [x for x in sp_all if x != "BEL"]
        base = T.get(("LRU", "LRU" if "LRU" in gp_all else "none", "LRU" if "LRU" in sp_all else "none"), (np.nan, np.nan))[0]

        def combos(cpus, gps, sps):
            return [(p, q, u) for p in cpus for q in gps for u in sps]

        def best(cs):
            c = [(T[k][0],) + k for k in cs if k in T]
            return min(c) if c else None
        comps = [(c, combos(comp_pol[c], gp_or, sp_no), combos(XS.RIVALS[c], gp_no, sp_no)) for c in XS.RIVALS]
        rival_all = combos(XS.NON_ORACLE, gp_no, sp_no)
        if "BEL" in gp_all:
            comps.append(("O-GPU", combos(XS.NON_ORACLE, ["BEL"], sp_no), rival_all))
        if "BEL" in sp_all:
            comps.append(("O-SLOW", combos(XS.NON_ORACLE, gp_or, ["BEL"]), rival_all))
        for comp, oset, rset in comps:
            if any(x not in T for x in oset + rset):
                continue                                   # 這格沒有跑齊：不算（同 x1_sim analyze）
            if oracle_keys is not None and any(k not in oracle_keys for k in oset):
                continue
            o, r = best(oset), best(rset)
            To, Tr = o[0], r[0]
            rho_o, rho_r = T[o[1:]][1], T[r[1:]][1]
            rows.append(dict(zip(CELL, key), component=comp, oracle_policy=o[1], oracle_gpu_policy=o[2], oracle_slow_policy=o[3],
                             T_oracle=To, best_rival=r[1], best_rival_gpu_policy=r[2], best_rival_slow_policy=r[3], T_rival=Tr,
                             headroom=(Tr - To) / Tr, write_saturated=bool(rho_o >= 1 or rho_r >= 1), rho_w_oracle=rho_o,
                             rho_w_rival=rho_r, T_base=base, src_runs=src))
            for k in rset:
                if T[k][0] < To - 1e-12:
                    v = dict(zip(CELL, key), component=comp, oracle_policy=o[1], oracle_gpu_policy=o[2],
                             oracle_slow_policy=o[3], T_oracle=To, rival=k[0], rival_gpu_policy=k[1], rival_slow_policy=k[2],
                             T_rival=T[k][0], oracle_slower_by=(To - T[k][0]) / T[k][0])
                    if rival_only:
                        v["rival_gpu_rival_only"] = k[1] in rival_only    # ④ 沒有這個 GPU 規則（登記如此）
                    viol.append(v)
        ok_ = ("O-ALL", "BEL" if "BEL" in gp_all else "none", "BEL" if "BEL" in sp_all else "none")
        if ok_ in T and all(k in T for k in rival_all) and (oracle_keys is None or ok_ in oracle_keys):
            r = best(rival_all)
            To = T[ok_][0]
            rows.append(dict(zip(CELL, key), component="O-ALL(report)", oracle_policy="O-ALL", oracle_gpu_policy=ok_[1],
                             oracle_slow_policy=ok_[2], T_oracle=To, best_rival=r[1], best_rival_gpu_policy=r[2],
                             best_rival_slow_policy=r[3], T_rival=r[0], headroom=(r[0] - To) / r[0],
                             write_saturated=bool(T[ok_][1] >= 1 or T[r[1:]][1] >= 1),
                             rho_w_oracle=T[ok_][1], rho_w_rival=T[r[1:]][1], T_base=base, src_runs=src))
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
    ap.add_argument("--rival-only-gpu", nargs="+", default=[], metavar="GPU_RULE",
                    help="只放進 ②、不放進 ④ 的 GPU 規則（PREREG 偏離 4 第 2 項：補跑的 TTL），例：TTLA:1 TTLA:3 TTLB:1 …")
    a = ap.parse_args()
    ro = tuple(dict.fromkeys(a.rival_only_gpu))
    d, comp_pol = load(a.sweep, a.pf_target, a.metric, a.dead_h)
    if ro:
        assert "BEL" not in ro and "none" not in ro, f"--rival-only-gpu 不能含 BEL／none：{ro}"
        miss = [x for x in ro if x not in set(d.gpu_policy)]
        assert not miss, f"--rival-only-gpu 的 {miss} 在 sweep 的 gpu_policy 裡一列都沒有（打錯字？）"
        on_or = d[d.gpu_policy.isin(ro) & ~d.policy.isin(XS.NON_ORACLE)]
        print(f"--rival-only-gpu {' '.join(ro)}：這些 GPU 規則有 {int(d.gpu_policy.isin(ro).sum())} 列；"
              f"其中 CPU 規則不在 NON_ORACLE 的 {len(on_or)} 列（只會被 ④ 用到，所以不會用到）")
    exact = d[d.noise_model == "none"]
    H, V = analyze(exact, comp_pol, rival_only=ro)
    if ro:                             # 自檢：拿掉這些規則的列重算（＝原本的 GPU 規則集合）；格子集合相同、上限不變大
        ck = CELL + ["component"]
        H0, _ = analyze(exact[~exact.gpu_policy.isin(ro)], comp_pol)
        a1, a0 = H.set_index(ck).headroom, H0.set_index(ck).headroom
        assert a1.index.is_unique and a0.index.is_unique
        only1, only0 = a1.index.difference(a0.index), a0.index.difference(a1.index)
        assert len(only1) == 0 and len(only0) == 0, \
            f"自檢失敗：(格, 元件) 集合不同：只在新的 {len(only1)} 列、只在拿掉後的 {len(only0)} 列；例 {list(only1[:3])} {list(only0[:3])}"
        a0 = a0.loc[a1.index]
        up = a1 > a0
        assert not up.any(), f"自檢失敗：{int(up.sum())} 列的上限比拿掉這些規則時大；例 {list(a1.index[up.values][:3])}"
        print(f"--rival-only-gpu 自檢 PASS：{len(a1)} 列（格, 元件）和拿掉這些規則時相同；"
              f"上限沒有變大；有變小的 {int((a1 < a0).sum())} 列，相同的 {int((a1 == a0).sum())} 列")
    noisy = d[d.noise_model != "none"]
    NH = []
    if len(noisy):
        ck = CELL + ["component"]
        hx = H.set_index(ck).headroom
        for spec, g in noisy.groupby(NZ):
            keys = set(zip(g.policy, g.gpu_policy, g.slow_policy))
            base = exact[[k not in keys for k in zip(exact.policy, exact.gpu_policy, exact.slow_policy)]]
            h, _ = analyze(pd.concat([base, g], ignore_index=True), comp_pol, oracle_keys=keys, rival_only=ro)
            for c, v in zip(NZ, spec):
                h[c] = v
            if spec[1] == 0:           # λ＝0 自檢
                j = h.set_index(ck).headroom
                common = j.index.intersection(hx.index)
                dmax = float((j.loc[common] - hx.loc[common]).abs().max()) if len(common) else 0.0
                assert dmax == 0.0, f"λ＝0 的上限和沒有雜訊的不同（最大差 {dmax}）：{spec}"
                print(f"λ＝0 自檢 {spec}：{len(common)} 格和沒有雜訊的逐值相同")
            NH.append(h)
    meta = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"),
                host=socket.gethostname(), pf_target=a.pf_target, metric=a.metric, dead_h=a.dead_h)
    if ro:
        meta["rival_only_gpu"] = ";".join(ro)      # 沒給就不加這欄（和之前的輸出逐欄相同）
    os.makedirs(a.out_dir, exist_ok=True)
    outs = [("headroom_components", H), ("violations", V)]
    if NH:
        NHd = pd.concat(NH, ignore_index=True)
        gk = CELL + ["component", "noise_model", "noise_lambda", "noise_draw"]
        NS = NHd.groupby(gk).headroom.agg(n_seeds="size", median="median", min="min", max="max", mean="mean", sd="std").reset_index()
        ex = H.set_index(CELL + ["component"]).headroom.rename("headroom_exact")
        NS = NS.join(ex, on=CELL + ["component"])
        outs += [("noise_headroom", NHd), ("noise_summary", NS)]
    for name, X in outs:
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
