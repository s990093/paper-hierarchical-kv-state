"""t1g.py — ICPE T1G（docs/h200/icpe/T1G_gpu_rerun.md）：CPU 層換成 H200 常數之後，重量 GPU 實測的 run。

給 runner（code/r_h200_t1g.sh）用的兩個排程小工具（量時間不和我們自己的 job 同時跑；memory h200-no-self-cotenancy-for-timing）：
  wait-clear --job J [--node N] [--max-min M]   等到節點上沒有我們自己的別的 job 在跑（每 60 s 查一次 squeue）；逾時 exit 2
  overlap --job J --start T0 --end T1 --step S --short SH --log CSV
                                                這一步的時間窗 [T0, T1] 裡，節點上有沒有我們自己的別的 job（sacct）；
                                                有 → 寫進 CSV、exit 1（照重疊重跑）；沒有 → 也寫一列、exit 0；查不到 → exit 2（規則 7）
  a3-plan --dir D --tier-params TP              T1G-2：存活設定＋策略 → D/a3_plan.tsv
  analyze --part a1pp|a1p|a2|p02|a3|all         T1G §3／T1G-2 §2：照 overlap.csv 篩出每一步最後一次乾淨的 run，套原本的分析程式；
                                                輸出在 results/icpe_h200/t1/gpu/<part>/analysis/，舊→新的對照寫 t1g_compare.json
                                                （要 pandas：在 env_h200.sh 的 conda 環境裡跑；登記 §5：資料收齊之前寫好）
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta

SLURM_BIN = "/cm/local/apps/slurm/current/bin"


def _now():
    return datetime.now().astimezone()


def _run(cmd):
    env = dict(os.environ, PATH=SLURM_BIN + ":" + os.environ.get("PATH", ""))
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} 失敗（exit {p.returncode}）：{p.stderr.strip()}")
    return p.stdout


def _node(a):
    return a.node or os.environ.get("SLURMD_NODENAME") or socket.gethostname().split(".")[0]


def cmd_wait_clear(a):
    node, t_end = _node(a), time.time() + a.max_min * 60
    n = 0
    while True:
        try:
            out = _run(["squeue", "-h", "-w", node, "-u", os.environ["USER"], "-t", "R", "-o", "%i %j"])
        except Exception as e:                      # 查不到 ≠ 沒有（規則 7）
            print(f"[t1g] squeue 查不到：{e} → 停", file=sys.stderr)
            return 2
        others = [ln for ln in out.split("\n") if ln.strip() and ln.split()[0] != str(a.job)]
        if not others:
            print(f"[t1g] {_now().isoformat(timespec='seconds')} 節點 {node} 上沒有我們自己的別的 job（等了 {n} 次）", flush=True)
            return 0
        if time.time() > t_end:
            print(f"[t1g] 等了 {a.max_min} 分鐘仍有我們自己的 job：{others} → 停", file=sys.stderr)
            return 2
        if n % 10 == 0:
            print(f"[t1g] {_now().isoformat(timespec='seconds')} 等我們自己的 job 結束：{others}", flush=True)
        n += 1
        time.sleep(60)


def _parse(s):
    if s in ("", "Unknown", "None"):
        return None
    return datetime.fromisoformat(s).astimezone()


def cmd_overlap(a):
    node = _node(a)
    t0, t1 = datetime.fromisoformat(a.start).astimezone(), datetime.fromisoformat(a.end).astimezone()
    since = (t0 - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%S")
    try:
        out = _run(["sacct", "-u", os.environ["USER"], "-X", "-P", "-n", "-S", since,
                    "-o", "JobID,JobName,NodeList,Start,End,State"])
    except Exception as e:
        print(f"[t1g] sacct 查不到：{e} → 停", file=sys.stderr)
        return 2
    hits = []
    for ln in out.split("\n"):
        f = ln.split("|")
        if len(f) < 6 or f[0] == str(a.job) or node not in f[2]:
            continue
        s, e = _parse(f[3]), _parse(f[4])
        if s is None:
            continue
        e = e or _now()
        if s < t1 and e > t0:
            hits.append(f"{f[0]}:{f[1]}:{f[3]}..{f[4]}")
    runs = sorted(glob.glob(os.path.join(os.environ.get("TIARA_RUNS", "."), f"*-{a.short}")))
    rid = os.path.basename(runs[-1]) if runs else "NOT_FOUND"
    new = not os.path.exists(a.log)
    os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
    with open(a.log, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["run_id", "ts", "job", "node", "step", "t_start", "t_end", "overlap", "overlapping_jobs"])
        w.writerow([rid, _now().isoformat(timespec="seconds"), a.job, node, a.step, a.start, a.end, int(bool(hits)), ";".join(hits)])
    print(f"[t1g] {a.step} run {rid} [{a.start} → {a.end}] 重疊：{hits or '無'}", flush=True)
    return 1 if hits else 0


A3_PREF = ("S5L", "S5c")           # T1G-2 §1 第 3 步：平手取原本 A／B／C 用過的策略，再平手取字母序第一個


def cmd_a3_plan(a):
    """T1G-2 §1 第 2、3 步：r1_sim_configs.csv 的存活設定（Llama）＋每格 best_wt 出現最多的策略 → a3_plan.tsv（runner 的 --part a3 讀）。"""
    from collections import Counter
    cfg = [r for r in csv.DictReader(open(os.path.join(a.dir, "r1_sim_configs.csv")))
           if r["model"] == "llama31_8b" and r["survive"] == "True"]
    ver = list(csv.DictReader(open(os.path.join(a.dir, "r1_sim_verdict.csv"))))
    tp = json.load(open(a.tier_params))
    tier_cpu = float(tp["cpu"]["read_GiBps"])
    key = ("release", "workload", "ssd_dev", "cpu_frac")
    rows = []
    for c in cfg:
        g = [v for v in ver if v["model"] == "llama31_8b" and abs(float(v["cpu_gibps"]) - float(c["cpu_gibps"])) < 1e-6
             and all(v[k] == c[k] for k in key)]
        assert len(g) == 5, (c, len(g))
        cnt = Counter(v["best_wt"] for v in g)
        top = max(cnt.values())
        tied = sorted(s for s, n in cnt.items() if n == top)
        wt = next((s for s in A3_PREF if s in tied), tied[0])
        cg = float(c["cpu_gibps"])
        over = "" if abs(cg - tier_cpu) < 1e-6 else f"{cg:g}"          # tier_params 的 CPU 層 → 不設 M7_CPU_GIBPS
        assert c["workload"] == "doc", c                                # 原本的 GPU 確認只做 doc（r_h200_r1h.sh）
        tag = f"{c['release']}_{c['ssd_dev']}_c{cg:.2f}_f{c['cpu_frac']}"
        rows.append([tag, over or "-", c["release"], c["ssd_dev"], c["cpu_frac"], wt, f"{cg:.4f}", dict(cnt)])
    with open(os.path.join(a.dir, "a3_plan.tsv"), "w") as fh:
        fh.write("# tag cpu_override release ssd cpu_frac wt cpu_gibps best_wt_counts\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    for r in rows:
        print("[t1g a3-plan]", *r)
    return 0


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T1D = os.path.join(REPO, "results", "icpe_h200", "t1")
T1G = os.path.join(T1D, "gpu")
TP = os.path.join(T1D, "tier_params_h200cpu.json")
M7H = os.path.join(REPO, "results", "m7_write_policy_h200")
P01 = os.path.join(REPO, "results", "icpe_h200", "p01")
MAIN_B = dict(io_model="share", cpu_tier="harness", ssd_dev="nfs", gap_s=0.0, ssd_frac=1.0, wl_seed=0)   # = p01_tools.MAIN_B


def _valid(prefix):
    """overlap.csv 裡步驟名以 prefix 開頭的：每一步取最後一次「沒有重疊、run 目錄 exit 0、沒有 CONTAMINATED」的 run_id；沒有 → None（NOT_MEASURED）。"""
    runs = os.environ.get("TIARA_RUNS", os.path.join(os.path.dirname(REPO), "tiara-runs"))
    rows = list(csv.DictReader(open(os.path.join(T1G, "overlap.csv")))) if os.path.exists(os.path.join(T1G, "overlap.csv")) else []
    out, tried = {}, {}
    for r in rows:
        if not r["step"].startswith(prefix):
            continue
        tried.setdefault(r["step"], []).append(r["run_id"])
        d = os.path.join(runs, r["run_id"])
        ec = open(os.path.join(d, "exit_code")).read().strip() if os.path.exists(os.path.join(d, "exit_code")) else "?"
        ok = r["overlap"] == "0" and ec == "0" and not os.path.exists(os.path.join(d, "CONTAMINATED"))
        if ok:
            out[r["step"]] = r["run_id"]
        else:
            out.setdefault(r["step"], None)
    return out, tried


def _run_env(cmd, env_extra):
    env = dict(os.environ, **env_extra)
    print("[t1g analyze] $", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, env=env, cwd=REPO)
    if p.returncode != 0:
        raise SystemExit(f"[t1g analyze] 失敗（exit {p.returncode}）：{' '.join(cmd)}")


def an_a1(part):
    import shutil
    import tempfile
    import numpy as np
    import pandas as pd
    sys.path.insert(0, os.path.join(REPO, "code"))
    import m7_repro_compare as C
    step = {"a1pp": "T1G-A1PP", "a1p": "T1G-A1P"}[part]
    val, tried = _valid(step)
    rid = val.get(step)
    src, outd = os.path.join(T1G, part), os.path.join(T1G, part, "analysis")
    os.makedirs(outd, exist_ok=True)
    if rid is None:
        doc = dict(run_id=run_id(), ts=ts(), part=part, status="NOT_MEASURED", tried=tried.get(step, []))
        json.dump(doc, open(os.path.join(outd, "verdict.json"), "w"), ensure_ascii=False, indent=1)
        return doc
    base = tempfile.mkdtemp(prefix=f"t1g_{part}_")
    stage = os.path.join(base, "stage")
    os.makedirs(stage)
    for f in os.listdir(M7H):          # 同 p01_tools.cmd_analyze_a 的篩法
        pth = os.path.join(M7H, f)
        if not os.path.isfile(pth) or not f.endswith((".csv", ".json")):
            continue
        if f.startswith(("b_", "b2_", "summary_b", "sim_", "repro_", "verdict", "a2_", "c7_anchor", "summary_a2", "summary_c7", "a2c7_")):
            continue
        shutil.copy2(pth, stage)
    if part == "a1pp":                 # fa3 的 C1、C4（同 A1''）
        for f in ("calib_c1.csv", "c4_noise.csv"):
            shutil.copy2(os.path.join(P01, "calib", f), os.path.join(stage, f))
    raw = pd.read_csv(os.path.join(src, "b_share.csv"))
    raw = raw[raw.run_id == rid]
    raw.to_csv(os.path.join(stage, "b_share.csv"), index=False)
    shutil.copy2(os.path.join(src, "b_workload_seed0.json"), stage)
    tmp, V = C.analyze(stage, TP, os.path.join(base, "figs"), f"t1g_{part}")
    delta = float(V["calib"]["C4"]["delta_pct"])
    dstar = max(5.0, delta)
    d = pd.read_csv(os.path.join(tmp, "summary_b_verdict.csv"))
    for k, v in MAIN_B.items():
        d = d[d[k].astype(str) == str(v)] if isinstance(v, str) else d[np.isclose(d[k].astype(float), float(v))]
    if part == "a1pp":
        old = pd.read_csv(os.path.join(P01, "p01a_table.csv"))
        old_ok = json.load(open(os.path.join(P01, "p01a_verdict.json")))["passed"]
    else:
        old = pd.read_csv(os.path.join(REPO, "results", "m7_write_policy_h200_merged", "summary_b_verdict_all.csv"))
        for k, v in MAIN_B.items():
            old = old[old[k].astype(str) == str(v)] if isinstance(v, str) else old[np.isclose(old[k].astype(float), float(v))]
        old_ok = None                  # 原本 A1' 的判定字在 ICPE_EVIDENCE（「cpu 0.25／0.5／1.0 都在 ±δ*」）；這裡只列舊數字
    bad = int((raw.contaminated_gpu.astype(str) == "True").sum())
    ret = raw[raw["round"] > 1]
    rows = []
    for _, r in d.sort_values("cpu_frac").iterrows():
        cf = float(r.cpu_frac)
        q = ret[np.isclose(ret.cpu_frac, cf)]
        m4, m5 = q[q.strategy == "S4+"].ttft.mean(), q[q.strategy == "S5"].ttft.mean()
        o = old[np.isclose(old.cpu_frac.astype(float), cf)]
        rows.append(dict(run_id=run_id(), ts=ts(), part=part, src_run=rid, cpu_frac=cf, n_reps=int(r.n_reps),
                         s4plus_median=float(r.s4plus_median), s5_median=float(r.s5_median), s5_vs_s4plus_pct=float(r.s5_vs_s4plus_pct),
                         delta_star_pct=dstar, passed=bool(float(r.s5_vs_s4plus_pct) <= dstar),
                         s5_vs_s4plus_mean_pct=float((m4 - m5) / m4 * 100),
                         old_s5_vs_s4plus_pct=(float(o.s5_vs_s4plus_pct.iloc[0]) if len(o) else None)))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(outd, "table.csv"), index=False)
    doc = dict(run_id=run_id(), ts=ts(), part=part, src_run=rid, status="DONE", c4_delta_pct=delta, delta_star_pct=dstar,
               contaminated_rows=bad, passed=bool(len(t) == 3 and t.passed.all() and bad == 0), old_passed=old_ok,
               rule="P0-1 §3.1：每個 cpu_frac，S5 對 S4+（回來請求 TTFT 中位數，rep 0–5）≤ +δ*；三格都過、污染列 0", rows=rows,
               m7_analyze_errors=V.get("errors"))
    json.dump(doc, open(os.path.join(outd, "verdict.json"), "w"), ensure_ascii=False, indent=1, default=str)
    return doc


def an_a2():
    import pandas as pd
    val, tried = _valid("T1G-A2-")
    outd = os.path.join(T1G, "a2", "analysis")
    os.makedirs(outd, exist_ok=True)
    rids = [r for r in val.values() if r]
    nm = sorted(s for s, r in val.items() if r is None)
    d = pd.read_csv(os.path.join(T1G, "a2", "b2_share.csv"))
    d = d[d.run_id.isin(rids)]
    fn = os.path.join(outd, "b2_share_valid.csv")
    d.to_csv(fn, index=False)
    _run_env([sys.executable, "code/m7_b2_analyze.py", fn], dict(M7_RES=outd, M7_TIER_PARAMS=TP, M7_F_CSV=os.path.join(M7H, "calib_c1.csv"),
                                                              M7_D2H_GIBPS="49.5"))
    v = pd.read_csv(os.path.join(outd, "b2_verdict.csv"))
    old = pd.read_csv(os.path.join(REPO, "results", "m7_write_policy_h200_merged", "b2_verdict.csv"))
    g, og = v[v.src == "gpu"], old[old.src == "gpu"]
    doc = dict(run_id=run_id(), ts=ts(), part="a2", src_runs=sorted(rids), not_measured_steps=nm, status="DONE" if not nm else "PARTIAL",
               gpu_rows=len(g), gpu_pass=int(g["pass"].sum()), old_gpu_rows=len(og), old_gpu_pass=int(og["pass"].sum()),
               rule="08 消融：寫入時決定的策略同時贏最好的不看位置與 S4B 各 ≥5%（GPU 列）")
    json.dump(doc, open(os.path.join(outd, "verdict.json"), "w"), ensure_ascii=False, indent=1, default=str)
    return doc


def an_p02():
    import shutil
    import pandas as pd
    val, tried = _valid("T1G-P02-")
    val.pop("T1G-P02-SMOKE", None)
    outd = os.path.join(T1G, "p02", "analysis")
    os.makedirs(outd, exist_ok=True)
    rids = [r for r in val.values() if r]
    nm = sorted(s for s, r in val.items() if r is None)
    g = pd.read_csv(os.path.join(T1G, "p02", "gpu_requests.csv"))
    g[g.run_id.isin(rids)].to_csv(os.path.join(outd, "gpu_requests.csv"), index=False)
    s = pd.read_csv(os.path.join(T1G, "p02", "sim_requests.csv"))
    s[s.ref == "S_orig"].to_csv(os.path.join(outd, "sim_requests.csv"), index=False)
    shutil.copy2(os.path.join(T1G, "p02", "selftest.csv"), outd)
    env = dict(P3_CONST="h200", P3_TIER="h200cpu", M7_TIER_PARAMS=TP)
    _run_env([sys.executable, "code/icpe/x1_gpu.py", "sim", "--out", outd, "--k", "8", "--times", os.path.join(outd, "gpu_requests.csv")], env)
    _run_env([sys.executable, "code/icpe/p02_analyze.py", "--dir", outd], env)
    new = json.load(open(os.path.join(outd, "p02_verdict.json")))
    old = json.load(open(os.path.join(REPO, "results", "icpe_h200", "p02", "p02_verdict.json")))
    doc = dict(run_id=run_id(), ts=ts(), part="p02", src_runs=sorted(rids), not_measured_steps=nm,
               status="DONE" if not nm else "PARTIAL", new=new, old=old)
    json.dump(doc, open(os.path.join(outd, "verdict.json"), "w"), ensure_ascii=False, indent=1, default=str)
    return doc


def an_a3():
    import pandas as pd
    a3 = os.path.join(T1G, "a3")
    plan = [ln.rstrip("\n").split("\t") for ln in open(os.path.join(a3, "a3_plan.tsv")) if ln.strip() and not ln.startswith("#")]
    cells = []
    for tag, over, rel, ssd, frac, wt, cg, *_ in plan:
        val, tried = _valid(f"T1G-A3-{tag}-")
        rids = [r for r in val.values() if r]
        nm = sorted(s for s, r in val.items() if r is None)
        fn = f"r1h_gpu_llama_doc_{tag}_valid.csv"
        d = pd.read_csv(os.path.join(a3, f"r1h_gpu_llama_doc_{tag}.csv"))
        d[d.run_id.isin(rids)].to_csv(os.path.join(a3, fn), index=False)
        _run_env([sys.executable, "code/m7_round1_analyze.py", "gpu", "--csv", fn, "--cpu", cg, "--release", rel, "--ssd", ssd, "--frac", frac,
              "--wt", wt, "--tag", tag], dict(M7_EXPLORE=a3, M7_RES=M7H))
        v = pd.read_csv(os.path.join(a3, f"r1_gpu_verdict_{tag}.csv"))
        med = v[v.stat == "median"]
        cells.append(dict(tag=tag, cpu_gibps=float(cg), release=rel, ssd=ssd, cpu_frac=float(frac), wt=wt, src_runs=sorted(rids),
                          not_measured_steps=nm, seeds_judged=len(med), seeds_pass=int(med["pass"].sum()),
                          survive=bool(int(med["pass"].sum()) >= 3)))
    doc = dict(run_id=run_id(), ts=ts(), part="a3", cells=cells, n_survive=sum(c["survive"] for c in cells),
               old="H200 自己 3 格 0/3（results/m7_explore_h200/r1_gpu_verdict_{A,B,C}.csv）",
               rule="寫入時決定策略的回來請求 TTFT 中位數同時贏最好的不看位置、S4B、S4W、S4C 各 ≥5%，≥3/5 seed → 存活")
    os.makedirs(os.path.join(a3, "analysis"), exist_ok=True)
    json.dump(doc, open(os.path.join(a3, "analysis", "verdict.json"), "w"), ensure_ascii=False, indent=1, default=str)
    return doc


def cmd_analyze(a):
    parts = ["a1pp", "a1p", "a2", "p02", "a3"] if a.part == "all" else [a.part]
    fn = {"a1pp": lambda: an_a1("a1pp"), "a1p": lambda: an_a1("a1p"), "a2": an_a2, "p02": an_p02, "a3": an_a3}
    cp = os.path.join(T1G, "t1g_compare.json")
    comp = json.load(open(cp)) if os.path.exists(cp) else {}
    for p in parts:
        comp[p] = fn[p]()
        print(f"[t1g analyze] {p}：", json.dumps({k: v for k, v in comp[p].items() if not isinstance(v, (list, dict))}, ensure_ascii=False, default=str))
    json.dump(comp, open(cp, "w"), ensure_ascii=False, indent=1, default=str)
    return 0


def run_id():
    from datetime import datetime as _d
    return os.environ.get("RUN_ID") or _d.now().strftime("%Y%m%d-%H%M%S") + "-adhoc"


def ts():
    return _now().isoformat(timespec="seconds")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("a3-plan"); s.add_argument("--dir", required=True); s.add_argument("--tier-params", required=True)
    s = sp.add_parser("analyze"); s.add_argument("--part", required=True, choices=["a1pp", "a1p", "a2", "p02", "a3", "all"])
    s = sp.add_parser("wait-clear"); s.add_argument("--job", required=True); s.add_argument("--node"); s.add_argument("--max-min", type=float, default=360)
    s = sp.add_parser("overlap"); s.add_argument("--job", required=True); s.add_argument("--node")
    s.add_argument("--start", required=True); s.add_argument("--end", required=True); s.add_argument("--step", required=True)
    s.add_argument("--short", required=True); s.add_argument("--log", required=True)
    a = ap.parse_args()
    sys.exit({"wait-clear": cmd_wait_clear, "overlap": cmd_overlap, "a3-plan": cmd_a3_plan, "analyze": cmd_analyze}[a.cmd](a))


if __name__ == "__main__":
    main()
