#!/usr/bin/env python3
"""next_a_w1_hitrate.py — NEXT_PLAN_20261104 W1：Mooncake 的 CPU 層命中率，定義照 T4 的 HR＝ΣC ÷ ΣU。

只用現有輸出，不跑模擬。唯一的計算是紀錄前處理（x1_sim.prep → t4.reusable，算每個請求的 U）。

定義（docs/h200/icpe/T4_hit_vs_ttft_bridge.md §2）：
  U[r]＝前面連續幾個 block 在 r 之前被讀過（acc[b][0] < r）——只由紀錄決定，和策略、容量、時間設定無關。
  C[r]＝r 到達時，有存的前綴裡在 CPU 層的 block 數（lay∈{1 BF16, 2 預取的 INT4}）。
  HR＝Σ_ret C ÷ Σ_ret U（ret＝回來的請求，kind＝cont）。

X1 輸出怎麼對到 C（為什麼只能用 restore=load 的列）：
  x1_sim.View.restore 在 load 模式逐塊載入有存的前綴：nC＝lay≠3 的塊數＝C、nS＝lay＝3 的塊數、nR＝n_all−L。
  cake 模式只載入後半段、前半段重算，所以 cake 列的 ret_cpu_loads ≤ C，不能當 C。
  ΣU 不在 X1 的任何欄位（ret_recompute 含從沒出現過的新 block），要從紀錄算。
  預取策略（O-PF、LRU+RPF、O-ALL*）一個 job 只跑一種時間設定 → 只拿得到 load 那一輪的 HR；cake 那一輪＝NOT_MEASURED。

自檢（任一條不過 → exit 1，不寫 results）：
  K1 T4 的對應：results/icpe_h200/t4/t4_sweep.csv 的 load 列 C_ret＝ret_cpu_loads、S_ret＝ret_slow_loads（逐列）；
     cake 列 ret_cpu_loads ≤ C_ret。→ 證明「load 列的 ret_cpu_loads＝C」。
  K2 紀錄對得上：prep 出來的回來請求數＝sweep 的 n_ret、總請求數＝n_req（每個 trace×rule）。
  K3 記帳：每個 load 列 ret_cpu_loads＋ret_slow_loads＋ret_recompute＝Σ_ret n_all，且 ret_cpu_loads＋ret_slow_loads ≤ Σ_ret U。
只報告：K4 非預取策略的 HR 在 12 種時間設定間是否相同（同一個 job 共用引擎狀態時應相同）。

FreeInference 的對照：T4 的 t4_sweep.csv（6 週、CPU 8–128 GiB、LRU／O-BEL／LRU+NONE／O-BEL+NONE），
6 週合併＝ΣC_ret ÷ ΣU_ret（同 T4）。FreeInference 256–1,024 GiB：T1 的 x1_sweep_fi 只有 cake 列 → NOT_MEASURED。

用法（head node，CPU）：
  P3_CONST=h200 P3_TIER=h200cpu M7_TIER_PARAMS=results/icpe_h200/t1/tier_params_h200cpu.json \\
  X1_TRACE_DIR=/work/ChihChungHsu/hungweilai/datasets/traces RUN_ID=<id> python code/icpe/next_a_w1_hitrate.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
REPO = os.path.dirname(CODE)
sys.path.insert(0, os.path.join(CODE, "p3x"))
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SRC_MC = os.path.join(REPO, "results", "icpe_h200", "t1", "p01", "sim_V1", "x1", "x1_sweep.csv.gz")
SRC_T4 = os.path.join(REPO, "results", "icpe_h200", "t4", "t4_sweep.csv")
OUT = os.path.join(REPO, "results", "icpe_next", "a", "w1")
PF_POL = ("O-PF", "LRU+RPF", "O-ALL", "O-ALL-H1800", "O-ALL-H3600")


def run_id():
    return os.environ.get("RUN_ID", "adhoc")


def now_ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def code_sha():
    h = hashlib.sha256()
    for p in (__file__, os.path.join(CODE, "p3x", "x1_sim.py"), os.path.join(CODE, "p3x", "x1_trace.py"),
              os.path.join(HERE, "t4.py")):
        h.update(open(p, "rb").read())
    return h.hexdigest()[:12]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def write(path, rows, rid, ts):
    keys = ["run_id", "ts"] + [k for k in rows[0].keys()]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(dict(run_id=rid, ts=ts, **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}))


def main():
    assert os.environ.get("P3_CONST") == "h200" and os.environ.get("P3_TIER") == "h200cpu", "要 P3_CONST=h200 P3_TIER=h200cpu"
    assert os.environ.get("X1_PLANNER", "cake") == "cake", "不要設 X1_PLANNER"
    rid, ts = run_id(), now_ts()
    checks, fail = {}, []

    # ---------------- K1：T4 的 load 列 C_ret＝ret_cpu_loads
    t4 = pd.read_csv(SRC_T4)
    ld, ck = t4[t4.restore == "load"], t4[t4.restore == "cake"]
    k1 = dict(n_load_rows=len(ld), n_cake_rows=len(ck),
              load_C_eq=int((ld.C_ret == ld.ret_cpu_loads).sum()), load_S_eq=int((ld.S_ret == ld.ret_slow_loads).sum()),
              cake_le=int((ck.ret_cpu_loads <= ck.C_ret).sum()), cake_lt=int((ck.ret_cpu_loads < ck.C_ret).sum()))
    k1["pass"] = bool(k1["load_C_eq"] == len(ld) and k1["load_S_eq"] == len(ld) and k1["cake_le"] == len(ck) and len(ld) > 0)
    checks["K1_t4_load_rows_equal_C"] = k1
    if not k1["pass"]:
        fail.append("K1")

    # ---------------- 紀錄前處理：每個 (trace, rule) 的 Σ_ret U、Σ_ret n_all
    import x1_sim as XS
    import t4 as T4
    mc = pd.read_csv(SRC_MC)
    src_rid = sorted(mc.run_id.unique().tolist())
    tr = {}
    for (trace, rule), g in mc.groupby(["trace", "rule"]):
        P = XS.prep(trace, rule)
        U = T4.reusable(P)
        kd = np.array(P["kindc"])
        ret = kd == 1
        nall = np.array(P["nall"], np.int64)
        nfull = np.array([len(x) for x in P["full"]], np.int64)
        tr[(trace, rule)] = dict(U_ret=int(U[ret].sum()), nall_ret=int(nall[ret].sum()), nfull_ret=int(nfull[ret].sum()),
                                 n_ret=int(ret.sum()), n_req=len(kd), ws_blocks=int(P["meta"]["n_blocks_full"]),
                                 unit_match_rate=float(P["meta"]["unit_match_rate"]))
        # K2
        ok = bool((g.n_ret == tr[(trace, rule)]["n_ret"]).all() and (g.n_req == tr[(trace, rule)]["n_req"]).all())
        checks[f"K2_{trace}_{rule}"] = dict(pass_=ok, prep_n_ret=tr[(trace, rule)]["n_ret"], sweep_n_ret=sorted(g.n_ret.unique().tolist()),
                                            prep_n_req=tr[(trace, rule)]["n_req"], sweep_n_req=sorted(g.n_req.unique().tolist()))
        if not ok:
            fail.append(f"K2 {trace} {rule}")

    # ---------------- K3 + 命中率（load 列）
    rows = []
    k3_bad = 0
    L = mc[mc.restore == "load"]
    for _, r in L.iterrows():
        t = tr[(r.trace, r.rule)]
        cpu, slw, rec = int(r.ret_cpu_loads), int(r.ret_slow_loads), int(r.ret_recompute)
        ok = (cpu + slw + rec == t["nall_ret"]) and (cpu + slw <= t["U_ret"])
        k3_bad += not ok
        rows.append(dict(src_run_id=r.run_id, trace=r.trace, rule=r.rule, cpu_GiB=int(r.cpu_GiB), cpu_frac_ws=float(r.cpu_frac_ws),
                         policy=r.policy, slow=r.slow, gpu=r.gpu,
                         hr_scope=("load_run_only" if r.policy in PF_POL else "engine_state"),
                         U_ret=t["U_ret"], C_ret=cpu, S_ret=slw, dropped_ret=t["U_ret"] - cpu - slw,
                         hr_cpu=cpu / t["U_ret"], hr_slow=slw / t["U_ret"], hr_stored=(cpu + slw) / t["U_ret"],
                         n_ret=int(r.n_ret), nall_ret=t["nall_ret"], n_drop=int(r.n_drop), n_demote=int(r.n_demote)))
    checks["K3_accounting"] = dict(pass_=k3_bad == 0, n_rows=len(L), n_bad=k3_bad)
    if k3_bad:
        fail.append("K3")

    # ---------------- K4（只報告）：非預取策略在 12 種時間設定間 C 是否相同
    H = pd.DataFrame(rows)
    nz = H[H.hr_scope == "engine_state"].groupby(["trace", "rule", "cpu_GiB", "policy"]).C_ret.nunique()
    checks["K4_view_independent_report_only"] = dict(n_groups=int(len(nz)), n_groups_varying=int((nz > 1).sum()),
                                                     varying=[list(map(str, k)) for k in nz[nz > 1].index.tolist()][:20])

    # ---------------- FreeInference（T4）：6 週合併
    fi_rows = []
    base = t4[(t4.slow == "ssd") & (t4.gpu == "idle") & (t4.restore == "load")]
    for (cap, pol), g in base.groupby(["cpu_GiB", "policy"]):
        assert len(g) == 6, (cap, pol, len(g))
        fi_rows.append(dict(src_run_id=";".join(sorted(g.run_id.unique())), trace="freeinference_6wk", rule="session_id",
                            cpu_GiB=int(cap), policy=pol, U_ret=int(g.U_ret.sum()), C_ret=int(g.C_ret.sum()), S_ret=int(g.S_ret.sum()),
                            hr_cpu=g.C_ret.sum() / g.U_ret.sum(), hr_slow=g.S_ret.sum() / g.U_ret.sum(),
                            hr_stored=(g.C_ret.sum() + g.S_ret.sum()) / g.U_ret.sum(), n_ret=int(g.nreq_ret.sum())))
    # T4 的 C 在 12 種時間設定間相同（同一引擎）：檢查一下，這樣 ssd/idle 的值代表全部
    var = t4.groupby(["trace", "cpu_GiB", "policy"]).C_ret.nunique()
    checks["K5_t4_C_view_independent_report_only"] = dict(n_groups=int(len(var)), n_varying=int((var > 1).sum()))

    checks["fail"] = fail
    os.makedirs(OUT, exist_ok=True)
    meta = dict(run_id=rid, ts=ts, code_sha=code_sha(), src_mooncake=os.path.relpath(SRC_MC, REPO), src_mooncake_sha256=sha256(SRC_MC),
                src_mooncake_run_id=src_rid, src_t4=os.path.relpath(SRC_T4, REPO), src_t4_sha256=sha256(SRC_T4),
                x1_trace_dir=os.environ.get("X1_TRACE_DIR", ""), traces={f"{k[0]}|{k[1]}": v for k, v in tr.items()},
                fi_256_1024="NOT_MEASURED（T1 x1_sweep_fi 只有 cake 列）", checks=checks)
    if fail:
        print("自檢沒過：", fail, file=sys.stderr)
        json.dump(meta, open(os.path.join(os.environ.get("RUN_DIR", "."), "w1_checks_FAILED.json"), "w"), ensure_ascii=False, indent=1)
        sys.exit(1)
    write(os.path.join(OUT, "w1_hitrate_mooncake.csv"), rows, rid, ts)
    write(os.path.join(OUT, "w1_hitrate_fi_t4.csv"), fi_rows, rid, ts)
    json.dump(meta, open(os.path.join(OUT, "w1_checks.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(checks, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
