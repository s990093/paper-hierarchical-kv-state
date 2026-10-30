#!/usr/bin/env python3
"""t5.py — ICPE T5（docs/h200/icpe/T5_fi_small_cpu.md）：FreeInference 在 CPU 16–64 GiB 的 X1（對最好的對稱對手）。

  python code/icpe/t5.py check    §2：CPU 128 GiB 的列和 T1 逐值相同；每個（週, 容量）的策略清單完整
  python code/icpe/t5.py report   §3：判定字、各元件各容量的最大 headroom、熱組、和 T1（128–1,024 GiB）接起來的曲線

sweep 與 analyze 用 X1 原本的程式（code/icpe/p3_fvar.py --variant V0 … x1 sweep／analyze），這支只做檢查與整理。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
T1X = os.path.join(REPO, "results", "icpe_h200", "t1", "p12", "x1")
T5 = os.path.join(REPO, "results", "icpe_h200", "t5")
T5X = os.path.join(T5, "p12", "x1")
T1_RUN = "20261028-073351-icpe-t1-p12-x1fi-sweep"
KEY = ["trace", "rule", "cpu_GiB", "policy", "slow", "gpu", "restore"]


def run_id():
    import t1
    return t1.run_id()


def ts():
    import t1
    return t1.ts()


def cmd_check(a):
    import pandas as pd
    d = pd.read_csv(os.path.join(T5X, "x1_sweep_fi.csv"))
    ref = pd.read_csv(os.path.join(T1X, "x1_sweep_fi.csv.gz"))
    assert set(ref.run_id) == {T1_RUN}
    out = dict(run_id=run_id(), ts=ts(), sweep_run_id=sorted(d.run_id.unique()), checks=[])
    ok = True
    # 1. CPU 128 GiB 和 T1 逐值相同
    m = d[d.cpu_GiB == 128].merge(ref[ref.cpu_GiB == 128], on=KEY, suffixes=("", "_t1"), how="outer", indicator=True)
    only = m[m._merge != "both"]
    bad = []
    for c in ["n_demote", "n_drop", "ret_cpu_loads", "ret_slow_loads"]:
        x = m[(m._merge == "both") & (m[c] != m[c + "_t1"])]
        bad += [(r.trace, r.policy, r.slow, r.gpu, c, r[c + "_t1"], r[c]) for _, r in x.head(10).iterrows()]
    for c in ["mean_ret", "mean_ret_user"]:
        x = m[(m._merge == "both") & ((m[c] - m[c + "_t1"]).abs() > 1e-6)]
        bad += [(r.trace, r.policy, r.slow, r.gpu, c, r[c + "_t1"], r[c]) for _, r in x.head(10).iterrows()]
    c1 = len(only) == 0 and not bad
    ok &= c1
    out["checks"].append(dict(name="1-reproduce-T1-128", passed=bool(c1), n_rows=int((m._merge == "both").sum()),
                              n_unmatched=int(len(only)), mismatches=[list(map(str, b)) for b in bad[:20]]))
    # 2. 策略清單完整
    want = set(ref[ref.cpu_GiB == 128].policy)
    miss = []
    for (tr, c), g in d.groupby(["trace", "cpu_GiB"]):
        lack = want - set(g.policy)
        if lack:
            miss.append((tr, int(c), sorted(lack)))
    cells = sorted({(tr, int(c)) for tr, c in zip(d.trace, d.cpu_GiB)})
    c2 = not miss and len(cells) == 6 * 4
    ok &= c2
    out["checks"].append(dict(name="2-policy-set", passed=bool(c2), n_week_cpu=len(cells), missing=miss[:10]))
    out["passed"] = bool(ok)
    json.dump(out, open(os.path.join(T5, "t5_check.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    sys.exit(0 if ok else 1)


def cmd_report(a):
    import pandas as pd
    out = dict(run_id=run_id(), ts=ts())
    for pre in ("x1fi", "x1fiuser"):
        V = pd.read_csv(os.path.join(T5X, f"{pre}_verdict.csv"))
        H = pd.read_csv(os.path.join(T5X, f"{pre}_headroom_components.csv"))
        H1 = pd.read_csv(os.path.join(T1X, f"{pre}_headroom_components.csv"))
        hp = os.path.join(T5X, f"{pre}_hot_groups.csv")
        HOT = pd.read_csv(hp) if os.path.getsize(hp) > 1 else pd.DataFrame()
        Hc = H[H.restore == "cake"]
        # 每個元件、每個容量：最大 headroom 與在哪（慢層, GPU）、最好的對手
        mx = []
        for (comp, c), g in Hc.groupby(["component", "cpu_GiB"]):
            r = g.loc[g.headroom.idxmax()]
            mx.append(dict(component=comp, cpu_GiB=int(c), max_headroom=float(r.headroom), slow=r.slow, gpu=r.gpu,
                           best_rival=r.best_rival, oracle_policy=r.oracle_policy,
                           n_cells_ge15=int((g.headroom >= 0.15).sum()), n_cells=len(g)))
        # 128 GiB：T5 和 T1 的 headroom 要一樣（只報告；§2 已經逐列比過 sweep）
        k = ["component", "cpu_GiB", "slow", "gpu", "restore"]
        j = Hc[Hc.cpu_GiB == 128].merge(H1[(H1.cpu_GiB == 128) & (H1.restore == "cake")], on=k, suffixes=("", "_t1"))
        d128 = float((j.headroom - j.headroom_t1).abs().max()) if len(j) else float("nan")
        # 接起來的曲線：T5 的 16–64 ＋ T1 的 128–1,024（每個元件、每個容量的最大）
        both = pd.concat([Hc[Hc.cpu_GiB < 128], H1[H1.restore == "cake"]], ignore_index=True)
        curve = both.groupby(["component", "cpu_GiB"]).headroom.max().unstack()
        out[pre] = dict(verdict=sorted(V.verdict_rule.unique().tolist()), verdict_rows=V.drop(columns=["src_runs"]).to_dict(orient="records"),
                        max_by_component_cpu=mx, hot_groups=HOT.drop(columns=[c for c in ("run_id", "ts") if c in HOT]).to_dict(orient="records"),
                        max_abs_diff_vs_t1_at_128=d128,
                        curve_16_1024={comp: {str(int(c)): float(v) for c, v in row.items() if v == v} for comp, row in curve.iterrows()},
                        analyze_run_id=sorted(H.run_id.unique().tolist()))
    json.dump(out, open(os.path.join(T5, "t5_report.json"), "w"), ensure_ascii=False, indent=1)
    for pre in ("x1fi", "x1fiuser"):
        o = out[pre]
        print(f"=== {pre}: verdict {o['verdict']}  |Δ vs T1 @128| {o['max_abs_diff_vs_t1_at_128']:.2e}")
        for r in o["max_by_component_cpu"]:
            print(f"  {r['component']:7s} {r['cpu_GiB']:5d} GiB  max {100*r['max_headroom']:6.2f}%  ({r['slow']},{r['gpu']}) rival={r['best_rival']} ≥15%: {r['n_cells_ge15']}/{r['n_cells']}")
        print("  hot groups:", len(o["hot_groups"]))
        for h in o["hot_groups"]:
            print("   ", h)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("check")
    sp.add_parser("report")
    a = ap.parse_args()
    dict(check=cmd_check, report=cmd_report)[a.cmd](a)


if __name__ == "__main__":
    main()
