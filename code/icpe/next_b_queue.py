"""next_b_queue.py — ICPE next（icpe-data）：正式模擬的佇列（方案 A，STATUS_DATA 2026-11-05T00:20；PREREG 偏離 4（origin/icpe-next-a 776e747））。

  python -I code/icpe/next_b_queue.py list                      # 列出所有 plan 與 block、job 數
  python -I code/icpe/next_b_queue.py run --plans cw_main tl_main [--workers 32] [--dry-run]
  python -I code/icpe/next_b_queue.py collect --plans cw_main   # 把跑完的 block 彙總進 results/icpe_next/b/

每個 block＝一次 `p3_fvar.py --variant V1 ... x1 sweep`（一個紀錄變體或一組小紀錄），照 CLAUDE.md §4.1 的殼：
  /ssd6/hungwei/icpe-next/runs/<RUN_ID>/{cmd.sh, context.txt, stdout.log, stderr.log, exit_code, out/, req/}
  RUN_ID＝<YYYYmmdd-HHMMSS>-icpe-data-<plan>-<block>；req/＝逐請求 TTFT 的 npz（x1_boot.py 用；不進 git）。
規則：
  * 一個 block 失敗（exit ≠ 0）→ 整個佇列停下來、印錯誤（CLAUDE.md 規則 2）。不自動重試。
  * 跑完（exit 0）的 block 記在 /ssd6/hungwei/icpe-next/runs/next_b_queue_done.jsonl；重跑佇列會跳過它們。
  * nice -n 10、OMP 執行緒 1；程式＝本 worktree 的 HEAD（context.txt 記 commit 與 git status）。
  * 寫進 git 的只有彙總：results/icpe_next/b/<plan>/<block>/x1_sweep.csv.xz（加 host 欄）＋manifest.json。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import lzma
import os
import socket
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ROOT = "/ssd6/hungwei/icpe-next"
RUNS = f"{ROOT}/runs"
DONE = f"{RUNS}/next_b_queue_done.jsonl"
PY = f"{ROOT}/venv/bin/python"
NB_DIR = f"{ROOT}/datasets/x1_nb"
MOONCAKE = "/ssd8/hungwei/paper-hkv/datasets/mooncake/traces"
FI_DIR = "/ssd8/hungwei/paper-hkv/datasets/freeinference_agentic_trace/release"
FI_WEEKS = ["2026-05-17_2026-05-23", "2026-05-24_2026-05-30", "2026-05-31_2026-06-06",
            "2026-06-07_2026-06-13", "2026-06-14_2026-06-20", "2026-06-21_2026-06-27"]   # T1／T4／T5 用的 6 週

# ---------------------------------------------------------------- 方案 A 的共同設定
CPU9 = ["8", "16", "32", "64", "128", "256", "512", "1024", "111.0625"]
CPU_REP = ["111.0625", "128"]
POL17 = ["LRU", "LRU2", "ARC", "S3FIFO", "GDSF", "PENS", "FREQ2Q", "LRU+NONE", "LRU+2HW", "LRU+2HA", "LRU+RPF",
         "O-BEL", "O-BELS", "O-DROP", "O-NOW", "O-PF", "O-ALL"]
TTL3 = ["0.3", "10", "300"]
GPOL = ["LRU", "BEL"] + [f"TTLA:{t}" for t in TTL3] + [f"TTLB:{t}" for t in TTL3]
MAIN_VIEWS = ["--slows", "ssd", "b32", "nfs", "--modes", "cake"]
# PREREG 偏離 4 第 2 項的補跑規則：主結果裡 ④ 對 ② 上限 ≥10% 的每一格，用第 3 項原登記的 7 個 TTL 補跑 ②。
# 已跑 0.3／10／300 → 只補 1／3／30／100（TTLA、TTLB）；CPU 規則＝② 用到的全部不預測規則（x1_sim.NON_ORACLE，ttl7cells 會核對）。
TTL7_THR = 0.10
TTL7_EXTRA = ["1", "3", "30", "100"]
GPOL_TTL7 = [f"TTLA:{t}" for t in TTL7_EXTRA] + [f"TTLB:{t}" for t in TTL7_EXTRA]
NON_ORACLE = ["LRU", "LRU2", "ARC", "S3FIFO", "GDSF", "PENS", "LRU+NONE", "LRU+2HW", "LRU+2HA", "LRU+RPF"]


def sweep_args(cpus=CPU9, views=MAIN_VIEWS, write="wb", pf="cpu", gpol=GPOL, pols=POL17):
    return (["--cpu", *cpus, "--policies", *pols, *views, "--gpu-gib", "111.0625", "--gpu-write", write,
             "--gpu-policies", *gpol, "--pf-target", pf])


def nb(name):
    return dict(traces=[f"nb:{name}"], rules=["nb"])


def blocks():
    """plan → [(block 名稱, traces/rules, sweep 參數)]；順序＝使用者決定的優先順序。"""
    P = {}
    P["cw_main"] = [(f"cw_N{N}_s{s}", nb(f"cachewise_N{N}_s{s}"), sweep_args()) for s in (1, 2, 3) for N in (16, 32, 64, 72)]
    P["tl_main"] = [(f"tl_c{c}", nb(f"tracelab_c{c}"), sweep_args()) for c in (1, 2, 4)]
    P["old_main"] = [("mooncake", dict(traces=["conversation", "toolagent"], rules=["R1", "R2"]), sweep_args()),
                     ("fi6w", dict(traces=[f"fi:{w}" for w in FI_WEEKS], rules=["session_id"]), sweep_args())]
    P["sa_main"] = [(f"sa_N{N}_s{s}", nb(f"semianalysis_N{N}_s{s}"), sweep_args()) for s in (1, 2, 3) for N in (16, 32, 64, 72)]
    P["bl_main"] = [("bailian3", dict(traces=[f"nb:bailian_{f}" for f in ("traceA", "thinking", "coder")], rules=["nb"]), sweep_args())]
    # 敏感度代表格：每份紀錄 1 個變體 × CPU 111.0625、128
    rep = [("cw", nb("cachewise_N16_s1")), ("tl", nb("tracelab_c1")), ("sa", nb("semianalysis_N16_s1")), ("bl", nb("bailian_coder")),
           ("mc", dict(traces=["conversation", "toolagent"], rules=["R1"])), ("fi", dict(traces=[f"fi:{FI_WEEKS[0]}"], rules=["session_id"]))]
    S = []
    for k, tr in rep:
        S.append((f"{k}_nvme807", tr, sweep_args(CPU_REP, ["--slows", "nvme807", "--modes", "cake"])))
        S.append((f"{k}_load", tr, sweep_args(CPU_REP, ["--slows", "ssd", "b32", "nfs", "nvme807", "--modes", "load"])))
        S.append((f"{k}_wt", tr, sweep_args(CPU_REP, write="wt")))
        S.append((f"{k}_pfgpu", tr, sweep_args(CPU_REP, pf="gpu")))
    S.append(("cw_cacheprefix", nb("cachewise_N16_s1_cache"), sweep_args(CPU_REP)))
    S.append(("tl_cacheprefix", nb("tracelab_c1_cache"), sweep_args(CPU_REP)))
    # 截斷 128K（PREREG §4 敏感度 1）：Mooncake（最長 126,195）、Bailian coder（最長 25,777）截斷後完全相同 → 不跑
    for k, nm in (("cw", "cachewise_N16_s1_t128k"), ("tl", "tracelab_c1_t128k"), ("sa", "semianalysis_N16_s1_t128k"),
                  ("fi", f"fi_{FI_WEEKS[0]}_t128k")):
        S.append((f"{k}_t128k", nb(nm), sweep_args(CPU_REP)))
    P["sens_rep"] = S
    # ≥10% 補跑：results/icpe_next/b/<plan>_ttl7/cells.csv（ttl7cells 從登記的分析表產生）→ 每個 (trace, rule, 慢層組合) 一個 block，
    # CPU 容量＝這個組合裡要補的那些；GPU 閒／忙無法在 sweep 裡分開選，兩個都跑（多跑的只會讓 ② 更強）。
    for base in list(P):
        f = f"{REPO}/results/icpe_next/b/{base}_ttl7/cells.csv"
        if os.path.exists(f):
            P[f"{base}_ttl7"] = ttl7_blocks(f)
    return P


def ttl7_blocks(path):
    by = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            by.setdefault((r["trace"], r["rule"], float(r["cpu_GiB"])), set()).add(r["slow"])
    grp = {}
    for (tr, ru, c), sl in by.items():
        grp.setdefault((tr, ru, tuple(s for s in ("ssd", "b32", "nfs") if s in sl)), []).append(c)
    out = []
    for (tr, ru, sl), cs in sorted(grp.items()):
        cpus = [str(int(c)) if c == int(c) else repr(c) for c in sorted(cs)]   # 111.0625 不能用 :g（6 位有效數字）
        name = f"{tr.replace('nb:', '').replace(':', '_')}_{ru}_{'-'.join(sl)}"
        out.append((name, dict(traces=[tr], rules=[ru]),
                    sweep_args(cpus, ["--slows", *sl, "--modes", "cake"], gpol=GPOL_TTL7, pols=NON_ORACLE)))
    return out


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw).stdout


def done_set():
    if not os.path.exists(DONE):
        return {}
    out = {}
    for ln in open(DONE):
        r = json.loads(ln)
        out[(r["plan"], r["block"])] = r
    return out


def cmd_list(a):
    P = blocks()
    dn = done_set()
    for plan, bl in P.items():
        if a.plans and plan not in a.plans:
            continue
        print(f"== {plan}")
        for name, tr, args in bl:
            st = "DONE " + dn[(plan, name)]["run_id"] if (plan, name) in dn else ""
            print(f"  {name:18s} {' '.join(tr['traces'])[:70]:70s} {st}")


def run_block(plan, name, tr, args, workers, dry):
    rid = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-icpe-data-{plan}-{name}"
    run = f"{RUNS}/{rid}"
    env = {"RUN_ID": rid, "RUN_DIR": f"{run}/req", "P3_CONST": "h200", "P3_TIER": "h200cpu",
           "M7_TIER_PARAMS": f"{REPO}/results/icpe_h200/t1/tier_params_h200cpu.json", "M7_RES": f"{REPO}/results/m7_write_policy_h200",
           "X1_NB_DIR": NB_DIR, "X1_TRACE_DIR": MOONCAKE, "X1_FI_DIR": FI_DIR,
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    argv = [PY, "-I", "code/icpe/p3_fvar.py", "--variant", "V1", "--calib", f"{REPO}/results/icpe_h200/p01/calib",
            "--out-root", f"{run}/out", "x1", "sweep", "--traces", *tr["traces"], "--rules", *tr["rules"], *args, "--workers", str(workers)]
    cmd = "cd " + REPO + "\n" + "".join(f"export {k}={v}\n" for k, v in env.items()) + "exec nice -n 10 " + " ".join(argv) + "\n"
    if dry:
        print(cmd)
        return 0, rid
    os.makedirs(f"{run}/req", exist_ok=True)
    with open(f"{run}/cmd.sh", "w") as fh:
        fh.write(cmd)
    with open(f"{run}/context.txt", "w") as fh:
        fh.write(f"date: {datetime.now().astimezone().isoformat()}\nhost: {socket.gethostname()}\nplan: {plan}\nblock: {name}\n")
        fh.write("git HEAD: " + sh(f"git -C {REPO} rev-parse HEAD"))
        fh.write("git status:\n" + sh(f"git -C {REPO} status --porcelain"))
        fh.write("uptime: " + sh("uptime"))
        fh.write("free -g:\n" + sh("free -g"))
        fh.write("venv: " + sh(f"{PY} -I -c \"import sys,numpy,pandas,pyarrow;print(sys.version.split()[0],numpy.__version__,pandas.__version__,pyarrow.__version__)\""))
        for t in tr["traces"]:
            if t.startswith("nb:"):
                p = f"{NB_DIR}/{t[3:]}.npz"
                fh.write(f"trace {t}: sha256 {sha256(p)}  {p}\n")
    t0 = time.time()
    with open(f"{run}/stdout.log", "w") as fo, open(f"{run}/stderr.log", "w") as fe:
        rc = subprocess.call(["bash", f"{run}/cmd.sh"], stdout=fo, stderr=fe)
    with open(f"{run}/exit_code", "w") as fh:
        fh.write(f"{rc}\n")
    if rc == 0:
        with open(DONE, "a") as fh:
            fh.write(json.dumps(dict(plan=plan, block=name, run_id=rid, run=run, elapsed_s=round(time.time() - t0, 1),
                                     ts=datetime.now().astimezone().isoformat(timespec="seconds"))) + "\n")
    return rc, rid


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def cmd_run(a):
    P = blocks()
    dn = done_set()
    for plan in a.plans:
        for name, tr, args in P[plan]:
            if (plan, name) in dn:
                print(f"[skip] {plan}/{name} 已完成（{dn[(plan, name)]['run_id']}）", flush=True)
                continue
            print(f"[{datetime.now():%m-%d %H:%M:%S}] start {plan}/{name}", flush=True)
            rc, rid = run_block(plan, name, tr, args, a.workers, a.dry_run)
            print(f"[{datetime.now():%m-%d %H:%M:%S}] end   {plan}/{name} rc={rc} run={rid}", flush=True)
            if rc != 0:
                err = open(f"{RUNS}/{rid}/stderr.log").read()[-4000:]
                print(f"BLOCK FAILED {plan}/{name} rc={rc}\n--- stderr (tail) ---\n{err}", flush=True)
                return 1
    return 0


def cmd_collect(a):
    """跑完的 block → results/icpe_next/b/<plan>/<block>/：x1_sweep.csv.xz（加 host 欄）＋manifest.json。"""
    dn = done_set()
    host = socket.gethostname()
    n = 0
    for (plan, name), r in sorted(dn.items()):
        if a.plans and plan not in a.plans:
            continue
        src = f"{r['run']}/out/x1/x1_sweep.csv"
        dst = f"{REPO}/results/icpe_next/b/{plan}/{name}"
        if os.path.exists(f"{dst}/manifest.json") and json.load(open(f"{dst}/manifest.json"))["run_id"] == r["run_id"]:
            continue
        os.makedirs(dst, exist_ok=True)
        buf = io.StringIO()
        rows = 0
        with open(src, newline="") as fh:
            rd = csv.reader(fh)
            w = csv.writer(buf)
            hdr = next(rd)
            assert hdr[:2] == ["run_id", "ts"], hdr[:2]
            w.writerow(hdr[:2] + ["host"] + hdr[2:])
            for row in rd:
                w.writerow(row[:2] + [host] + row[2:])
                rows += 1
        with lzma.open(f"{dst}/x1_sweep.csv.xz", "wt", preset=9) as fh:
            fh.write(buf.getvalue())
        ctx = open(f"{r['run']}/context.txt").read()
        man = dict(plan=plan, block=name, run_id=r["run_id"], host=host, run_dir=r["run"], rows=rows, elapsed_s=r["elapsed_s"],
                   finished=r["ts"], cmd=open(f"{r['run']}/cmd.sh").read(), context=ctx,
                   note="方案 A（STATUS_DATA 2026-11-05T00:20；PREREG 偏離 4）；② 不含 block 粒度 APC（PREREG 偏離 2 第 3 項）")
        with open(f"{dst}/manifest.json", "w") as fh:
            json.dump(man, fh, ensure_ascii=False, indent=1)
        n += 1
        print(f"→ {dst}  rows={rows}")
    print(f"collected {n}")
    return 0


def cmd_traces(a):
    """轉好的紀錄 → results/icpe_next/b/traces/trace_summary.csv（每個變體一列的彙總；沒有逐對話的數字）。"""
    import glob
    host = socket.gethostname()
    rows = []
    for p in sorted(glob.glob(f"{NB_DIR}/*.meta.json")):
        m = json.load(open(p))
        L = m.get("loader") or {}
        ar = m.get("arrival") or {}
        rows.append(dict(run_id=m.get("run_id"), ts=m.get("ts"), host=m.get("host", host), name=m["name"], trace=m["trace"],
                         prefix_rule=m.get("prefix_rule") if m["trace"] in ("cachewise", "tracelab") else "",
                         arrival=ar.get("kind"), N=ar.get("N", ""), seed=ar.get("seed", ""), compress=ar.get("factor", ""),
                         copies=ar.get("copies", ""), n_conv=ar.get("n_conv", ""), active_median=ar.get("active_median", ""),
                         active_p10=ar.get("active_p10", ""), n_req=m["n_req"], n_sess=m["n_sess"], n_cont=m["n_cont"],
                         full_chunks_total=m["full_chunks_total"], n_keys=m["write_check"]["n_keys"], span_h=round(m["span_s"] / 3600, 3),
                         inp_tokens_median=m["inp_tokens_median"], inp_tokens_p90=m["inp_tokens_p90"], frac_inp_gt_128k=m["frac_inp_gt_128k"],
                         trunc_frac_req_cut=(m.get("trunc_128k") or {}).get("frac_req_cut", ""), time_fix_within_session=m["time_fix_within_session"],
                         rule6=json.dumps({k: v for k, v in L.items() if k.startswith("rule6")}, ensure_ascii=False),
                         npz_sha256=sha256(p.replace(".meta.json", ".npz"))))
    dst = f"{REPO}/results/icpe_next/b/traces"
    os.makedirs(dst, exist_ok=True)
    with open(f"{dst}/trace_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"→ {dst}/trace_summary.csv  rows={len(rows)}")
    return 0


def cmd_ttl7cells(a):
    """登記的分析表（next_sim_analyze.py 的 *_headroom_components.csv，3 個 TTL）→ results/icpe_next/b/<plan>_ttl7/cells.csv：
    任何元件（含 O-ALL(report)）上限 ≥ TTL7_THR 的格子。O-ALL(report) 也算：它的 ② 也用到 GPU 層的 TTL，多補只會讓 ② 更強。"""
    sys.path.insert(0, os.path.join(REPO, "code", "p3x"))
    sys.path.insert(0, os.path.join(REPO, "code"))
    import x1_sim as XS
    assert NON_ORACLE == XS.NON_ORACLE, (NON_ORACLE, XS.NON_ORACLE)
    assert set(sum(XS.RIVALS.values(), [])) <= set(NON_ORACLE)
    rows = list(csv.DictReader(open(a.headroom, newline="")))
    rids = {r["run_id"] for r in rows}
    assert len(rids) == 1 and "adhoc" not in rids, rids
    src = rids.pop()
    for r in rows:   # 只接受主結果的設定
        assert (r["restore"], r["gpu_write"], r["pf_target"], float(r["gpu_GiB"]), float(r["slow_cap_frac"])) == \
               ("cake", "wb", "cpu", 111.0625, 0.0), r
    key = ("trace", "rule", "cpu_GiB", "slow", "gpu")
    cell = {}
    for r in rows:
        h = float(r["headroom"]) if r["headroom"] not in ("", "nan") else float("nan")
        k = tuple(r[c] for c in key)
        c = cell.setdefault(k, dict(max_headroom_ttl3=float("-inf"), components_ge=[]))
        if h == h:
            c["max_headroom_ttl3"] = max(c["max_headroom_ttl3"], h)
            if h >= a.thr:
                c["components_ge"].append(r["component"])
    sel = [(k, v) for k, v in sorted(cell.items()) if v["components_ge"]]
    dst = f"{REPO}/results/icpe_next/b/{a.plan}_ttl7"
    os.makedirs(dst, exist_ok=True)
    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    with open(f"{dst}/cells.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "ts", "host", "src_analysis_run_id", "thr", *key, "max_headroom_ttl3", "components_ge_thr"])
        for k, v in sel:
            w.writerow([os.environ.get("RUN_ID", "adhoc"), ts, socket.gethostname(), src, a.thr, *k,
                        f"{v['max_headroom_ttl3']:.6f}", "|".join(v["components_ge"])])
    n_view = len({(k[0], k[1], k[2], k[3]) for k, _ in sel})
    print(f"→ {dst}/cells.csv  cells={len(sel)}／{len(cell)}  (trace,rule,cpu,slow)={n_view}  jobs={n_view * 2 * len(NON_ORACLE) * len(GPOL_TTL7)}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("traces")
    s = sp.add_parser("ttl7cells"); s.add_argument("--plan", required=True); s.add_argument("--headroom", required=True)
    s.add_argument("--thr", type=float, default=TTL7_THR)
    s = sp.add_parser("list"); s.add_argument("--plans", nargs="*", default=[])
    s = sp.add_parser("run"); s.add_argument("--plans", nargs="+", required=True); s.add_argument("--workers", type=int, default=32)
    s.add_argument("--dry-run", action="store_true")
    s = sp.add_parser("collect"); s.add_argument("--plans", nargs="*", default=[])
    a = ap.parse_args()
    sys.exit(dict(list=cmd_list, run=cmd_run, collect=cmd_collect, traces=cmd_traces, ttl7cells=cmd_ttl7cells)[a.cmd](a))


if __name__ == "__main__":
    main()
