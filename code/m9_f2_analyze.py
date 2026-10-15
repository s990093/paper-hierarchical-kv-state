"""m9_f2_analyze.py — F2 能耗 κ 的計算（不碰 GPU）

  python code/m9_f2_analyze.py cakee  --probe-run <run_id>
      由 probe 的 P_b、P_wait 與既有 C1 的 f(i)，照預先登記 §2.4 算每層的 Cake-E 切點 m_E
  python code/m9_f2_analyze.py final  --probe-run <run_id> --sweep-runs <run_id> [<run_id> ...]
      帳 A/B/C/D 的能量最佳 vs 時間最佳、節省、判定、κ_E、成本算術；寫 results/m9_followup/f2_*.csv
  python code/m9_f2_analyze.py gate   --probe-run <run_id> [--c1 ...]
      （2026-10-16，平台 C）sweep 之前檢查計數器能不能用（F2 §2.3 事先寫的兩條）：更新週期 ≤ 最短還原的 5%、
      ≥10 s 的區段 ΔE/Δt 和功率讀值差 ≤5%；另外核對 NVML 和 torch 是同一張卡。沒過 exit 1（停下來回報）
  其他平台：--res（輸出目錄）、--c1（f(i) 的 C1 CSV）、--chunk-mib（每 chunk MiB；LongAlpaca 256），預設＝MI300X 的值

所有輸出列都有 run_id（本次分析的 run）、src_run（原始量測的 run）與 ts。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RUNS = os.environ.get("TIARA_RUNS", "/mlsteam/data/tiara/runs")
RID = os.environ.get("RUN_ID") or (datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc")
RES = os.path.join(HERE, "..", "results", "m9_followup")
C1 = os.path.join(HERE, "..", "results", "m7_write_policy_mi300x", "calib_c1.csv")
N = 64
CHUNK_MIB = 64
GiB = 1 << 30
MiB = 1 << 20

TIERS = {"cpu35": 35.4159, "cpu11": 11.6, "cpu369": 3.69, "local": 6.98115234375, "nfs": 0.331083984375}


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def med(v):
    return statistics.median(v) if v else float("nan")


def rows(path):
    return list(csv.DictReader(open(path)))


TAG = ""


def writer(name, fields):
    os.makedirs(RES, exist_ok=True)
    if TAG:
        name = name.replace("f2_", f"f2_{TAG}_", 1)
    f = open(os.path.join(RES, name), "w", newline="")
    w = csv.DictWriter(f, fieldnames=["run_id", "ts"] + fields, extrasaction="ignore")
    w.writeheader()

    def row(**kw):
        kw.update(run_id=RID, ts=ts())
        for k, v in list(kw.items()):
            if isinstance(v, float):
                kw[k] = round(v, 6)
        w.writerow(kw)
        f.flush()
    return row


def load_f():
    d = defaultdict(list)
    for r in rows(C1):
        if r["item"] == "f_chunk" and r["gpu_state"] == "idle":
            d[int(r["chunk_idx"])].append(float(r["ms"]) / 1e3)
    return [med(d[i]) for i in range(N)]


def write_boundary(n, f, ell):
    # 與 code/m7_restore_harness.py:write_boundary 相同（這裡不 import torch）
    best, best_m, R = None, 0, 0.0
    for m in range(0, n + 1):
        if m > 0:
            R += f[m - 1]
        T = max(R, (n - m) * ell)
        if best is None or T < best:
            best, best_m = T, m
    return best_m


def probe_summary(run):
    P = defaultdict(list)
    erec = defaultdict(list)
    tch = defaultdict(list)
    val = []
    for r in rows(os.path.join(RUNS, run, "f2_probe.csv")):
        P[r["phase"]].append(float(r["P_W"]))
        # 功率讀值：瞬時（MI300X current_socket_power；NVML POWER_INSTANT）。沒有瞬時讀值時才退回 1 s 平均（NVML
        # nvmlDeviceGetPowerUsage），並在輸出標明是哪一種（2026-10-16）
        if r["cur_W_mean"] not in ("", "None"):
            val.append((r["phase"], float(r["P_W"]), float(r["cur_W_mean"]), "cur_W", float(r["secs"])))
        elif r.get("avg_W_mean") not in (None, "", "None"):
            val.append((r["phase"], float(r["P_W"]), float(r["avg_W_mean"]), "avg_W", float(r["secs"])))
        if r["phase"] == "e_rec":
            note = dict(x.split("=") for x in r["note"].split(";"))
            erec[int(note["chunk"])].append(float(note["J_per_chunk"]))
            tch[int(note["chunk"])].append(float(note["ms_per_chunk"]))
    return P, erec, tch, val


def cakee(a):
    P, erec, tch, val = probe_summary(a.probe_run)
    Pb = med(P["busy_compute_only_64"])
    Pw = med(P["wait_nfs_load_only"])
    Pidle = med(P["idle_model"])
    f = load_f()
    thr = Pw / (Pb - Pw)
    out = writer("f2_cakee.csv", ["src_run", "tier", "read_GiBps", "ell_ms", "P_b_W", "P_wait_W", "P_idle_W",
                                  "thr_kappa", "kappa0", "kappa63", "b_time", "m_E"])
    res = {}
    for t, gib in TIERS.items():
        ell = CHUNK_MIB * MiB / (gib * GiB)
        bT = write_boundary(N, f, ell)
        mE = 0
        while mE < N and f[mE] / ell < thr:
            mE += 1
        mE = min(mE, bT)
        res[t] = mE
        out(src_run=a.probe_run, tier=t, read_GiBps=gib, ell_ms=ell * 1e3, P_b_W=Pb, P_wait_W=Pw, P_idle_W=Pidle,
            thr_kappa=thr, kappa0=f[0] / ell, kappa63=f[63] / ell, b_time=bT, m_E=mE)
    print("CAKE_E_ARGS", " ".join(f"{t}={m}" for t, m in res.items()))
    print(json.dumps(dict(P_b=Pb, P_wait=Pw, P_idle=Pidle, ratio_b_wait=Pb / Pw, ratio_b_idle=Pb / Pidle, thr=thr)))


# ------------------------------------------------------------------ gate（2026-10-16，平台 C：NVML 的計數器要先重新驗證）
def _bdf(x):
    """'00000000:E5:00.0' 或 '00000000:e5:00' → (domain, bus, device)；讀不懂回 None。"""
    try:
        d, b, rest = str(x).split(":")[:3]
        return int(d, 16), int(b, 16), int(rest.split(".")[0], 16)
    except Exception:
        return None


def gate(a):
    """F2 §2.3 事先寫的兩條，在 sweep 之前判（只用 probe 與 C1 的 f(i)，不看任何還原結果）：
      1. 計數器更新週期（中位數）≤ 最短一次還原的 5%。最短還原＝各層用 f(i) 與頻寬算的 Cake 時間 max(R(m), (n−m)ℓ)
         的最小值（MI300X 報告寫「最短的一次還原約 110 ms」，同一個算法給 cpu35 約 107 ms）。超過時預先登記要改成
         「K 次一起量」，這支 sweep 沒有實作 → 停下來回報，不自己加。
      2. 同一段 ≥10 s 的區段，ΔE/Δt 和功率讀值的平均差 ≤5%（MI300X：10–30 s 區段差 0.23–0.69%）。
    另外：NVML 的 bdf 和 torch 的 PCI id 要是同一張卡（規則 7）。"""
    P, erec, tch, val = probe_summary(a.probe_run)
    meta = json.load(open(os.path.join(RUNS, a.probe_run, "f2_probe_meta.json")))
    c = meta["counter"]
    f = load_f()
    tpred = {}
    for t, gib in TIERS.items():
        ell = CHUNK_MIB * MiB / (gib * GiB)
        m = write_boundary(N, f, ell)
        tpred[t] = max(sum(f[:m]), (N - m) * ell)
    t_min = min(tpred.values())
    upd = c.get("update_period_ms_median")
    checks = {}
    checks["counter_changes"] = (c.get("n_changes") or 0) >= 2
    checks["update_period_le_5pct"] = upd is not None and upd <= 0.05 * t_min * 1e3
    long_ = [(ph, pc, pp, src, secs) for ph, pc, pp, src, secs in val if secs >= 10.0]
    diffs = [(ph, (pc - pp) / pp * 100, src, secs) for ph, pc, pp, src, secs in long_]
    checks["validation_rows_ge10s"] = len(diffs) > 0
    checks["validation_le_5pct"] = bool(diffs) and all(abs(d) <= 5.0 for _, d, _, _ in diffs)
    nb, tb = _bdf(meta.get("bdf")), _bdf(meta.get("torch_pci"))
    checks["same_gpu_nvml_torch"] = bool(nb and tb and nb == tb)   # 讀不懂任一邊＝無法確認＝不過（規則 7）
    ok = all(checks.values())
    out = writer("f2_gate.csv", ["src_run", "check", "ok", "value", "threshold", "note"])
    out(src_run=a.probe_run, check="update_period_le_5pct", ok=checks["update_period_le_5pct"], value=upd,
        threshold=0.05 * t_min * 1e3, note=f"ms；最短還原（預測）{t_min * 1e3:.1f} ms @ {min(tpred, key=tpred.get)}；p90 {c.get('update_period_ms_p90')}")
    out(src_run=a.probe_run, check="counter_changes", ok=checks["counter_changes"], value=c.get("n_changes"), threshold=2, note=f"res_uJ={c.get('res_uJ')}")
    for ph, d, src, secs in diffs:
        out(src_run=a.probe_run, check=f"validation_{ph}", ok=abs(d) <= 5.0, value=d, threshold=5.0, note=f"%；{src}；{secs:.1f} s")
    out(src_run=a.probe_run, check="same_gpu_nvml_torch", ok=checks["same_gpu_nvml_torch"], value=f"{meta.get('bdf')} | {meta.get('torch_pci')}", threshold="", note="")
    summ = dict(ok=ok, checks=checks, t_pred_ms={k: v * 1e3 for k, v in tpred.items()}, update_period_ms_median=upd,
                validation=[dict(phase=ph, diff_pct=d, src=src, secs=secs) for ph, d, src, secs in diffs])
    json.dump(summ, open(os.path.join(RUNS, RID, "f2_gate.json"), "w"), indent=1, ensure_ascii=False)
    print("F2_GATE", json.dumps(summ, ensure_ascii=False), flush=True)
    sys.exit(0 if ok else 1)


# ------------------------------------------------------------------ final
def key_of(r):
    mode = r["mode"]
    if mode in ("S", "CE"):
        return f"{mode}{int(r['m'])}" if mode == "S" else "CE"
    return mode


def final(a):
    P, erec, tch, val = probe_summary(a.probe_run)
    Pb = med(P["busy_compute_only_64"])
    Pw = med(P["wait_nfs_load_only"])
    Pidle = med(P["idle_model"])
    Pbare = med(P["idle_bare"])
    Pdma = med(P["dma_h2d"])

    # ---- 計數器驗證
    vo = writer("f2_counter_validation.csv", ["src_run", "phase", "P_counter_W", "P_poll_W", "diff_pct", "poll_src", "secs"])
    for ph, pc, pp, src, secs in val:
        vo(src_run=a.probe_run, phase=ph, P_counter_W=pc, P_poll_W=pp, diff_pct=(pc - pp) / pp * 100, poll_src=src, secs=secs)

    # ---- probe 摘要
    po = writer("f2_probe_summary.csv", ["src_run", "item", "median", "min", "max", "n", "unit"])
    for ph, v in P.items():
        po(src_run=a.probe_run, item=f"P_{ph}", median=med(v), min=min(v), max=max(v), n=len(v), unit="W")
    for i in sorted(erec):
        po(src_run=a.probe_run, item=f"e_rec_chunk{i}", median=med(erec[i]), min=min(erec[i]), max=max(erec[i]),
           n=len(erec[i]), unit="J/chunk")
        po(src_run=a.probe_run, item=f"t_rec_chunk{i}", median=med(tch[i]), min=min(tch[i]), max=max(tch[i]),
           n=len(tch[i]), unit="ms/chunk")
    meta = json.load(open(os.path.join(RUNS, a.probe_run, "f2_probe_meta.json")))
    for k, v in meta["counter"].items():
        po(src_run=a.probe_run, item=f"counter_{k}", median=v, min="", max="", n="", unit="")

    # ---- sweep 原始列（rep>=0）
    S = []
    for run in a.sweep_runs:
        for r in rows(os.path.join(RUNS, run, "f2_sweep.csv")):
            if int(r["rep"]) < 0:
                continue
            r["src_run"] = run
            for k in ("secs", "E_J", "P_W", "t_restore", "t_rline", "t_lline", "read_GiBps"):
                r[k] = float(r[k])
            # 切點＝重算的 chunk 數。harness 的 load_only 把 meet 記成「載入的前綴長度」（64），所以一律改用 n_recompute
            r["m"] = int(r["n_recompute"])
            S.append(r)
    bad = [r for r in S if r["kv_bad"] not in ("", "0")]
    # 只重算：所有層共用
    Rrows = [r for r in S if r["mode"] == "R"]
    R_E = [r["E_J"] for r in Rrows]
    cv_R = statistics.pstdev(R_E) / statistics.fmean(R_E) if len(R_E) > 1 else float("nan")

    def acct(r, acc, param=0.0):
        E, t = r["E_J"], r["secs"]
        if acc == "A":
            return E
        if acc == "B":
            return E - Pidle * t
        if acc == "C":
            return E + param * r["t_lline"]
        if acc == "D":
            return E + param * Pw * t
        raise ValueError(acc)
    accts = [("A", 0.0), ("B", 0.0), ("C", 50.0), ("C", 150.0), ("D", 0.5), ("D", 1.0), ("D", 2.0)]

    so = writer("f2_sweep_summary.csv", ["src_runs", "tier", "read_GiBps", "action", "m_median", "n", "secs_median",
                                         "secs_min", "secs_max", "E_A_median", "E_A_min", "E_A_max", "P_W_median",
                                         "E_B_median", "t_lline_median", "E_per_chunk_A"])
    vo2 = writer("f2_verdict.csv", ["src_runs", "accounting", "param", "tier", "T_star", "m_T", "E_star", "m_E",
                                    "differ", "E_Tstar", "E_Estar", "saving_pct", "time_cost_pct", "nonoverlap",
                                    "E_cake", "E_cakeE", "cakeE_vs_cake_pct", "tier_good", "tier_dead"])
    srcs = "+".join(a.sweep_runs)
    overall = {}
    tier_actions = {}
    for t in TIERS:
        acts = defaultdict(list)
        for r in S:
            if r["tier"] == t:
                acts[key_of(r)].append(r)
        acts["R"] = Rrows
        tier_actions[t] = acts
        for k, rs in sorted(acts.items(), key=lambda kv: med([x["m"] for x in kv[1]])):
            so(src_runs=srcs, tier=t, read_GiBps=TIERS[t], action=k, m_median=med([x["m"] for x in rs]), n=len(rs),
               secs_median=med([x["secs"] for x in rs]), secs_min=min(x["secs"] for x in rs),
               secs_max=max(x["secs"] for x in rs), E_A_median=med([x["E_J"] for x in rs]),
               E_A_min=min(x["E_J"] for x in rs), E_A_max=max(x["E_J"] for x in rs),
               P_W_median=med([x["P_W"] for x in rs]), E_B_median=med([acct(x, "B") for x in rs]),
               t_lline_median=med([x["t_lline"] for x in rs]), E_per_chunk_A=med([x["E_J"] for x in rs]) / N)
    for acc, prm in accts:
        goods, deads, cake_small = [], [], []
        for t, acts in tier_actions.items():
            mE = {k: med([acct(x, acc, prm) for x in rs]) for k, rs in acts.items()}
            mT = {k: med([x["secs"] for x in rs]) for k, rs in acts.items()}
            mm = {k: med([x["m"] for x in rs]) for k, rs in acts.items()}
            Ts = min(mT, key=mT.get)
            Es = min(mE, key=mE.get)
            differ = abs(mm[Ts] - mm[Es]) > 2
            saving = (mE[Ts] - mE[Es]) / abs(mE[Ts]) if mE[Ts] else float("nan")
            tcost = (mT[Es] - mT[Ts]) / mT[Ts]
            eT = [acct(x, acc, prm) for x in acts[Ts]]
            eE = [acct(x, acc, prm) for x in acts[Es]]
            nonov = max(eE) < min(eT)
            # Cake-E：m_E=0 的層 S(0) 就是只載入（f2_cakee.csv），sweep 沒有另外跑，用 L 代表
            ce = mE.get("CE", mE.get("L") if a.cakee_zero_is_l else None)
            cpct = (mE["C"] - ce) / abs(mE["C"]) * 100 if ce is not None else float("nan")
            good = differ and saving >= 0.10 and nonov
            dead = (not differ) or saving < 0.05
            goods.append(good)
            deads.append(dead)
            cake_small.append(abs(cpct) < 5 if ce is not None else True)
            vo2(src_runs=srcs, accounting=acc, param=prm, tier=t, T_star=Ts, m_T=mm[Ts], E_star=Es, m_E=mm[Es],
                differ=differ, E_Tstar=mE[Ts], E_Estar=mE[Es], saving_pct=saving * 100, time_cost_pct=tcost * 100,
                nonoverlap=nonov, E_cake=mE["C"], E_cakeE=ce if ce is not None else "", cakeE_vs_cake_pct=cpct,
                tier_good=good, tier_dead=dead)
        if any(goods):
            v = "有看頭"
        elif all(deads) or Pb / Pw <= 1.2 or all(cake_small):
            v = "死路"
        else:
            v = "可能"
        overall[f"{acc}{'' if acc in ('A', 'B') else prm}"] = v
    if cv_R > 0.10:
        overall["A"] = "NOT_MEASURABLE"

    # ---- 時間–能量的 Pareto 前緣（帳 A）：比「只載入」每快 1 秒要多花幾焦耳
    pa = writer("f2_pareto.csv", ["src_runs", "tier", "action", "m", "secs", "E_A", "on_frontier",
                                  "dt_vs_L_s", "dE_vs_L_J", "J_per_s_saved_vs_L"])
    for t, acts in tier_actions.items():
        pts = sorted(((med([x["secs"] for x in rs]), med([x["E_J"] for x in rs]), k, med([x["m"] for x in rs]))
                      for k, rs in acts.items()))
        tL, eL = med([x["secs"] for x in acts["L"]]), med([x["E_J"] for x in acts["L"]])
        best = float("inf")
        for tt, ee, k, m in pts:
            on = ee < best
            best = min(best, ee)
            dt, de = tL - tt, ee - eL
            pa(src_runs=srcs, tier=t, action=k, m=m, secs=tt, E_A=ee, on_frontier=on, dt_vs_L_s=dt, dE_vs_L_J=de,
               J_per_s_saved_vs_L=(de / dt) if dt > 1e-9 else "")

    # ---- κ_$ 算術（價格全是〔假設〕）：選能量最佳省下的電費 vs 多花的 GPU 時間租金
    co = writer("f2_cost.csv", ["src_runs", "accounting", "tier", "T_star", "E_star", "dE_J", "dt_s",
                                "price_kWh_assumed", "pue_assumed", "uusd_energy_saved_per_restore",
                                "gpu_usd_per_hr_assumed", "uusd_gputime_cost_per_restore",
                                "breakeven_gpu_usd_per_hr", "note"])
    for t, acts in tier_actions.items():
        mE = {k: med([x["E_J"] for x in rs]) for k, rs in acts.items()}
        mT = {k: med([x["secs"] for x in rs]) for k, rs in acts.items()}
        Ts, Es = min(mT, key=mT.get), min(mE, key=mE.get)
        dE, dt = mE[Ts] - mE[Es], mT[Es] - mT[Ts]
        for pe in (0.05, 0.10, 0.30):
            for gh in (1.0, 2.5, 6.0):
                pue = 1.3
                saved = dE / 3.6e6 * pe * pue
                cost = dt / 3600 * gh
                co(src_runs=srcs, accounting="A", tier=t, T_star=Ts, E_star=Es, dE_J=dE, dt_s=dt,
                   price_kWh_assumed=pe, pue_assumed=pue, uusd_energy_saved_per_restore=saved * 1e6,
                   gpu_usd_per_hr_assumed=gh, uusd_gputime_cost_per_restore=cost * 1e6,
                   breakeven_gpu_usd_per_hr=(saved / dt * 3600) if dt > 0 else "",
                   note="GPU 時間可另作他用時才算租金；GPU 本來就閒著時租金差＝0")

    # ---- κ_E
    ko = writer("f2_kappa_e.csv", ["src_runs", "tier", "chunk", "f_ms_probe", "ell_ms_measured", "kappa_T",
                                   "e_rec_J", "e_load_J", "kappa_E", "kappa_E_over_kappa_T"])
    for t, acts in (tier_actions.items() if not TAG else []):   # e_rec 是 Llama 的 probe，別的模型不算 κ_E
        L = acts["L"]
        e_load = med([x["E_J"] for x in L]) / N
        ell = med([x["secs"] for x in L]) / N
        for i in sorted(erec):
            ko(src_runs=srcs, tier=t, chunk=i, f_ms_probe=med(tch[i]), ell_ms_measured=ell * 1e3,
               kappa_T=med(tch[i]) / 1e3 / ell, e_rec_J=med(erec[i]), e_load_J=e_load,
               kappa_E=med(erec[i]) / e_load, kappa_E_over_kappa_T=(med(erec[i]) / e_load) / (med(tch[i]) / 1e3 / ell))

    summ = dict(P_b=Pb, P_wait=Pw, P_idle_model=Pidle, P_idle_bare=Pbare, P_dma=Pdma, ratio_b_wait=Pb / Pw,
                ratio_b_idle=Pb / Pidle, cv_R=cv_R, n_R=len(R_E), kv_bad_rows=len(bad), n_rows=len(S),
                verdicts=overall)
    json.dump(summ, open(os.path.join(RUNS, RID, "f2_final_summary.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summ, ensure_ascii=False, indent=1))


def main():
    global TAG, RES, C1, CHUNK_MIB
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["cakee", "final", "gate"])
    p.add_argument("--probe-run", required=True)
    p.add_argument("--sweep-runs", nargs="*", default=[])
    p.add_argument("--cakee-zero-is-l", type=int, default=1)
    p.add_argument("--tag", default="")
    p.add_argument("--res", default=RES, help="輸出目錄（預設 results/m9_followup＝MI300X）")
    p.add_argument("--c1", default=C1, help="f(i) 的 C1 CSV（預設 MI300X 第一階段的 calib_c1.csv）")
    p.add_argument("--chunk-mib", type=int, default=CHUNK_MIB, help="每 chunk MiB（Llama 64；LongAlpaca 256）")
    a = p.parse_args()
    TAG, RES, C1, CHUNK_MIB = a.tag, a.res, a.c1, a.chunk_mib
    os.makedirs(os.path.join(RUNS, RID), exist_ok=True)
    {"cakee": cakee, "final": final, "gate": gate}[a.cmd](a)


if __name__ == "__main__":
    main()
