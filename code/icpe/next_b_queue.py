"""next_b_queue.py — ICPE next（icpe-data）：正式模擬的佇列（方案 A，STATUS_DATA 2026-11-05T00:20；PREREG 偏離 4（origin/icpe-next-a 776e747））。

  python -I code/icpe/next_b_queue.py list                      # 列出所有 plan 與 block、job 數
  python -I code/icpe/next_b_queue.py run --plans cw_main tl_main [--workers 32] [--engine py|rust] [--dry-run]
  python -I code/icpe/next_b_queue.py collect --plans cw_main   # 把跑完的 block 彙總進 results/icpe_next/b/

每個 block＝一次 `p3_fvar.py --variant V1 ... x1 sweep`（一個紀錄變體或一組小紀錄），照 CLAUDE.md §4.1 的殼：
  /ssd6/hungwei/icpe-next/runs/<RUN_ID>/{cmd.sh, context.txt, stdout.log, stderr.log, exit_code, out/, req/}
  RUN_ID＝<YYYYmmdd-HHMMSS>-icpe-data-<plan>-<block>；req/＝逐請求 TTFT 的 npz（x1_boot.py 用；不進 git）。
規則：
  * 一個 block 失敗（exit ≠ 0）→ 整個佇列停下來、印錯誤（CLAUDE.md 規則 2）。不自動重試。
  * 跑完（exit 0）的 block 記在 /ssd6/hungwei/icpe-next/runs/next_b_queue_done.jsonl；重跑佇列會跳過它們。
  * nice -n 10、OMP 執行緒 1；程式＝本 worktree 的 HEAD（context.txt 記 commit 與 git status）。
  * --engine rust（PREREG 偏離 6；icpe-sim G1 PASS 0b0ef1e、icpe-data G0 PASS 後才可用）：x1 sweep 加 --engine rust，
    CPU 約 1.25×workers（Rust 執行緒 w＋摘要行程 w//4）；逐 job 時間記在 <run>/rust_timing.jsonl。只用在還沒開始的 block。
  * 寫進 git 的只有彙總：results/icpe_next/b/<plan>/<block>/x1_sweep.csv.xz（加 host 欄）＋manifest.json。
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
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
# 路徑：預設＝acvlab；別台（TRX40、H200）用環境變數指到自己的位置（STATUS_DATA 2026-11-05T09:xx 的分工）
ROOT = os.environ.get("NEXT_B_ROOT", "/ssd6/hungwei/icpe-next")
RUNS = f"{ROOT}/runs"
DONE = f"{RUNS}/next_b_queue_done.jsonl"
PY = os.environ.get("NEXT_B_PY", f"{ROOT}/venv/bin/python")
NB_DIR = os.environ.get("NEXT_B_NB_DIR", f"{ROOT}/datasets/x1_nb")
MOONCAKE = os.environ.get("NEXT_B_MOONCAKE", "/ssd8/hungwei/paper-hkv/datasets/mooncake/traces")
FI_DIR = os.environ.get("NEXT_B_FI_DIR", "/ssd8/hungwei/paper-hkv/datasets/freeinference_agentic_trace/release")
TAG = os.environ.get("NEXT_B_TAG", "icpe-data")      # run_id 裡的 session 名（acvlab＝icpe-data）
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
# PREREG 偏離 6 的 P1
COMP5 = ["O-BEL", "O-BELS", "O-DROP", "O-NOW", "O-PF"]                                   # x1_sim.COMPONENT_POLICIES 的全部規則
TRUNC6 = [f"{p}-H{h}" for h in (1800, 3600) for p in ("O-DROP", "O-NOW", "O-ALL")]     # 截尾版（--dead-h），只報告
GPOL16 = GPOL + GPOL_TTL7                                                                # LRU、BEL、TTLA／TTLB × 7


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
    # PREREG 偏離 6（P1–P3，照優先度）：每個主結果 block 補跑。P1：② 和 ④ 都用 7 個 TTL（只補 1／3／30／100，
    # 配 NON_ORACLE＋元件規則；④ 缺任何一個 TTL，next_sim_analyze 會靜默整列跳過 → check_p1 核對）＋截尾規則（只報告）。
    # P2／P3：規則集合同 sens_rep（17 × 8），從代表格擴到全部格子。
    for base in ("cw_main", "tl_main", "old_main", "sa_main", "bl_main"):
        k = base[:-5]
        P[f"{k}_p1"] = [x for name, tr, _ in P[base] for x in (
            (f"{name}_ttl", tr, sweep_args(gpol=GPOL_TTL7, pols=NON_ORACLE + COMP5)),
            (f"{name}_trunc", tr, sweep_args(gpol=GPOL16, pols=TRUNC6)))]
        P[f"{k}_p2"] = [x for name, tr, _ in P[base] for x in (
            (f"{name}_nvme807", tr, sweep_args(views=["--slows", "nvme807", "--modes", "cake"])),
            (f"{name}_load", tr, sweep_args(views=["--slows", "ssd", "b32", "nfs", "nvme807", "--modes", "load"])))]
        P[f"{k}_p3"] = [x for name, tr, _ in P[base] for x in (
            (f"{name}_wt", tr, sweep_args(write="wt")),
            (f"{name}_pfgpu", tr, sweep_args(pf="gpu")))]
    # 跨機器對拍（新功能開啟）：acvlab 先跑並 collect（--allow-xcheck）當參考；別台跑同一個 block，用 next_b_g0.py 逐值比，相同才開跑正式的
    xa = sweep_args(["64"], ["--slows", "ssd", "--modes", "cake"], gpol=["LRU", "BEL", "TTLA:10", "TTLB:10"],
                    pols=["LRU", "GDSF", "O-BEL", "O-PF"])
    P["xcheck"] = [("xc_sa", nb("semianalysis_N16_s1"), xa), ("xc_bl", nb("bailian_coder"), xa)]
    # ≥10% 補跑：results/icpe_next/b/<plan>_ttl7/cells.csv（ttl7cells 從登記的分析表產生）→ 每個 (trace, rule, 慢層組合) 一個 block，
    # CPU 容量＝這個組合裡要補的那些；GPU 閒／忙無法在 sweep 裡分開選，兩個都跑（多跑的只會讓 ② 更強）。
    for base in list(P):
        f = f"{REPO}/results/icpe_next/b/{base}_ttl7/cells.csv"
        if os.path.exists(f):
            P[f"{base}_ttl7"] = ttl7_blocks(f)
    P.update(p4_blocks())
    return P


# PREREG 偏離 6 的 P4（W4 雜訊，§3 B）：選格＝icpe-main 的清單（偏離 8；origin/icpe-next-main c868ef2）。
# 每個（格, 元件）要跑的組合＝next_sim_analyze 的 ④：O-ORD／O-DROP／O-NOW／O-PF＝該元件的 CPU 規則 × 15 個 GPU 不預測規則；
# O-GPU＝NON_ORACLE × BEL；O-ALL(report)＝(O-ALL, BEL)。雜訊：P 的 λ＝0–0.4、M 的 λ＝0–0.3，λ＞0 × 10 個 seed，λ＝0 一次（自檢）。
# 一個 block＝一個紀錄 × 模型（P／M，λ 清單不同）× 抽法（B1／B2）；只跑清單裡的 job（x1 sweep --jobs-from，icpe-sim）。
P4_CELLS = f"{REPO}/results/icpe_next/main/p4/p4_cells_distinct.csv"
P4_GP_NO = ["LRU"] + [f"TTL{s}:{t}" for s in "AB" for t in sorted(TTL3 + TTL7_EXTRA, key=float)]
P4_LAMS = {("P", "B1"): ["0", "0.1", "0.2", "0.3", "0.4"], ("P", "B2"): ["0.1", "0.2", "0.3", "0.4"],
           ("M", "B1"): ["0", "0.1", "0.2", "0.3"], ("M", "B2"): ["0.1", "0.2", "0.3"]}
P4_COLS = ["trace", "rule", "cpu_GiB", "policy", "gpu_policy", "slow", "gpu"]


def _cpu_str(v):
    x = float(v)
    return str(int(x)) if x.is_integer() else repr(x)


def p4_combos(XS, c):
    if c in XS.COMPONENT_POLICIES:
        return [(p, g) for p in XS.COMPONENT_POLICIES[c] for g in P4_GP_NO]
    if c == "O-GPU":
        return [(p, "BEL") for p in XS.NON_ORACLE]
    if c == "O-ALL(report)":
        return [("O-ALL", "BEL")]
    raise SystemExit(f"P4：不認得的元件 {c}")


def p4_blocks():
    if not os.path.exists(P4_CELLS):
        return {}
    XS = import_xs()
    by = {}
    for r in csv.DictReader(open(P4_CELLS)):
        assert r["gpu_GiB"] == "111.0625" and r["gpu_write"] == "wb" and r["restore"] == "cake" and float(r["slow_cap_frac"]) == 0, r
        sa = r["family_input"] == "SemiAnalysis"
        by.setdefault((sa, r["trace"]), []).append(r)
    out = {"p4_b1": [], "p4_b2": [], "sa_p4_b1": [], "sa_p4_b2": []}
    for (sa, trace), rows in sorted(by.items()):
        jobs = sorted({(trace, r["rule"], _cpu_str(r["cpu_GiB"]), p, g, r["slow"], r["gpu"])
                       for r in rows for c in r["components"].split(";") for p, g in p4_combos(XS, c)})
        cpus = sorted({j[2] for j in jobs}, key=float)
        rules = sorted({j[1] for j in jobs})
        slows = sorted({j[5] for j in jobs})
        pols = sorted({j[3] for j in jobs})
        gps = sorted({j[4] for j in jobs}, key=lambda g: (g != "LRU", g != "BEL", g))
        short = trace[3:] if trace.startswith("nb:") else trace
        for draw in ("B1", "B2"):
            for model in ("P", "M"):
                args = ["--cpu", *cpus, "--policies", *pols, "--slows", *slows, "--modes", "cake", "--gpu-gib", "111.0625",
                        "--gpu-write", "wb", "--gpu-policies", *gps, "--pf-target", "cpu", "--noise-models", model,
                        "--noise-lambdas", *P4_LAMS[(model, draw)], "--noise-draws", draw, "--noise-seeds", *map(str, range(10))]
                out[("sa_" if sa else "") + "p4_" + draw.lower()].append((f"{short}_{model}_{draw}",
                                                                           dict(traces=[trace], rules=rules, jobs=jobs, no_npz=True), args))
    return out


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


def import_xs():
    sys.path.insert(0, os.path.join(REPO, "code", "p3x"))
    sys.path.insert(0, os.path.join(REPO, "code"))
    import x1_sim as XS
    return XS


def check_p1(XS):
    """偏離 6 P1：④ 和 ② 必須用同一組 GPU 規則。next_sim_analyze 不加 --rival-only-gpu 時，某格的 ④ 缺任何一個 TTL，
    那個元件就整列跳過、不報錯（icpe-sim 11/6 提醒）→ 補跑的 CPU 規則必須正好是 NON_ORACLE ∪ COMPONENT_POLICIES。"""
    assert NON_ORACLE == XS.NON_ORACLE, (NON_ORACLE, XS.NON_ORACLE)
    comp = [p for v in XS.COMPONENT_POLICIES.values() for p in v]
    assert sorted(COMP5) == sorted(comp), (COMP5, comp)
    assert set(sum(XS.RIVALS.values(), [])) <= set(NON_ORACLE)
    for p in TRUNC6:
        assert p in XS.POLICIES, p


def parse_sweep(args):
    d, k = {}, None
    for x in args:
        if x.startswith("--"):
            k = x[2:]
            d[k] = []
        else:
            d[k].append(x)
    return d


def cmd_count(a):
    """每個 block 的 job 數（直接呼叫 x1_sim.jobs_for，和 sweep 開跑時印的 jobs= 同一個數）；不讀紀錄、不跑模擬。"""
    XS = import_xs()
    check_p1(XS)
    P = blocks()
    tot = 0
    for plan in a.plans:
        ptot = 0
        for name, tr, args in P[plan]:
            d = parse_sweep(args)
            for nm in d["slows"]:
                if nm not in XS.SLOWS and nm not in XS.EXTRA_SLOWS:
                    XS.EXTRA_SLOWS.append(nm)
            gs = XS.gpu_specs([float(x) for x in d["gpu-gib"]], d["gpu-write"], d["gpu-policies"], d["pf-target"])
            ns = XS.noise_specs(d["noise-models"], [float(x) for x in d["noise-lambdas"]], d["noise-draws"],
                                [int(x) for x in d["noise-seeds"]]) if d.get("noise-models") else None
            J = XS.jobs_for(tr["traces"], tr["rules"], [float(x) for x in d["cpu"]], d["policies"], d["modes"],
                            slows=d["slows"], gspecs=gs, nspecs=ns)
            if tr.get("jobs"):         # 同 --jobs-from：job 的某一個 view 的鍵在清單裡就保留
                keep = {(t, r, float(c), p, g, s, gp) for t, r, c, p, g, s, gp in tr["jobs"]}
                J = [j for j in J if any((j[0], j[1], float(j[2]), j[3], j[5]["pol"] if j[5] else "none", v[0], v[1]) in keep
                                         for v in j[4])]
            nv = len(tr["traces"]) * len(tr["rules"])
            print(f"{plan:10s} {name:28s} variants={nv} jobs={len(J):7d} per_variant={len(J) / nv:9.1f}")
            ptot += len(J)
        print(f"{plan:10s} TOTAL jobs={ptot}")
        tot += ptot
    print(f"ALL TOTAL jobs={tot}")
    return 0


def run_block(plan, name, tr, args, workers, dry, engine="py"):
    rid = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{TAG}-{plan}-{name}"
    run = f"{RUNS}/{rid}"
    # RUN_DIR＝x1_sim 存逐請求 npz 的地方（只用在 x1_boot）；P4 的 block 設 no_npz → 空字串＝不存（PREREG 的 P4 分析只用 sweep CSV；
    # 存的話 SA 約 1.2 TB、TL 約 0.9 TB〔推算〕，共用磁碟放不下；icpe-trx 11/6 15:15 的請求、STATUS_DATA 15:25）
    env = {"RUN_ID": rid, "RUN_DIR": "" if tr.get("no_npz") else f"{run}/req", "P3_CONST": "h200", "P3_TIER": "h200cpu",
           "M7_TIER_PARAMS": f"{REPO}/results/icpe_h200/t1/tier_params_h200cpu.json", "M7_RES": f"{REPO}/results/m7_write_policy_h200",
           "X1_NB_DIR": NB_DIR, "X1_TRACE_DIR": MOONCAKE, "X1_FI_DIR": FI_DIR,
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    if engine == "rust":
        env["X1_RUST_TIMING_LOG"] = f"{run}/rust_timing.jsonl"
    argv = [PY, "-I", "code/icpe/p3_fvar.py", "--variant", "V1", "--calib", f"{REPO}/results/icpe_h200/p01/calib",
            "--out-root", f"{run}/out", "x1", "sweep", "--traces", *tr["traces"], "--rules", *tr["rules"], *args, "--workers", str(workers)]
    if engine == "rust":
        argv += ["--engine", "rust"]
    if tr.get("jobs"):                 # P4：只跑清單裡的 job（清單寫在 run 目錄，sweep 印它的 sha256）
        argv += ["--jobs-from", f"{run}/jobs.csv"]
    cmd = "cd " + REPO + "\n" + "".join(f"export {k}={v}\n" for k, v in env.items()) + "exec nice -n 10 " + " ".join(argv) + "\n"
    if dry:
        print(cmd)
        return 0, rid
    os.makedirs(f"{run}/req", exist_ok=True)
    with open(f"{run}/cmd.sh", "w") as fh:
        fh.write(cmd)
    if tr.get("jobs"):
        with open(f"{run}/jobs.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(P4_COLS)
            w.writerows(tr["jobs"])
    with open(f"{run}/context.txt", "w") as fh:
        fh.write(f"date: {datetime.now().astimezone().isoformat()}\nhost: {socket.gethostname()}\nplan: {plan}\nblock: {name}\nengine: {engine}\n")
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
    # 'plan:pat1,pat2' ＝這個 plan 只跑名稱符合這些 glob 的 block（例 sens_rep:tl_cacheprefix,tl_t128k）；--only 對全部 plan 生效
    specs = [(x.split(":", 1)[0], x.split(":", 1)[1].split(",") if ":" in x else []) for x in a.plans]
    for plan, _ in specs:
        assert plan in P, f"沒有這個 plan：{plan}"
    if any(p.endswith("_p1") for p, _ in specs):
        check_p1(import_xs())
    for plan, pats in specs:
        for name, tr, args in P[plan]:
            if a.only and not any(fnmatch.fnmatch(name, pat) for pat in a.only):
                continue
            if pats and not any(fnmatch.fnmatch(name, pat) for pat in pats):
                continue
            if (plan, name) in dn:
                print(f"[skip] {plan}/{name} 已完成（{dn[(plan, name)]['run_id']}）", flush=True)
                continue
            print(f"[{datetime.now():%m-%d %H:%M:%S}] start {plan}/{name}", flush=True)
            rc, rid = run_block(plan, name, tr, args, a.workers, a.dry_run, a.engine)
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
        if plan == "xcheck" and not a.allow_xcheck:   # 參考只由 acvlab 收；別台的 xcheck 用 next_b_g0.py 比，不寫進 results/
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


def cmd_digest(a):
    """紀錄 npz 的內容指紋（savez_compressed 的 zip 有時間戳，檔案 sha256 每次轉換都不同）：
    依鍵名排序，sha256(鍵名、dtype、shape、位元組)。別台重新下載＋轉換後，指紋要和 acvlab 的相同才能用。
    不含 meta_json：它存 run_id／ts／host／絕對路徑，每台每次都不同（icpe-trx 11/5 09:07 指出；第一版含它，作廢）。
    meta_json 裡的彙總欄位另由 trace_summary.csv 比。"""
    import numpy as np
    for nm in a.names:
        p = f"{NB_DIR}/{nm}.npz"
        h = hashlib.sha256()
        with np.load(p, allow_pickle=False) as z:
            for k in sorted(z.files):
                if k == "meta_json":
                    continue
                x = np.ascontiguousarray(z[k])
                h.update(f"{k}|{x.dtype.str}|{x.shape}|".encode())
                h.update(x.tobytes())
        print(f"{nm} {h.hexdigest()}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("traces")
    s = sp.add_parser("ttl7cells"); s.add_argument("--plan", required=True); s.add_argument("--headroom", required=True)
    s.add_argument("--thr", type=float, default=TTL7_THR)
    s = sp.add_parser("list"); s.add_argument("--plans", nargs="*", default=[])
    s = sp.add_parser("count"); s.add_argument("--plans", nargs="+", required=True)
    s = sp.add_parser("run"); s.add_argument("--plans", nargs="+", required=True); s.add_argument("--workers", type=int, default=32)
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--engine", choices=["py", "rust"], default="py", help="rust＝PREREG 偏離 6（CPU 約 1.25×workers）")
    s.add_argument("--only", nargs="*", default=[], help="只跑名稱符合這些 glob 的 block（例 'sa_*'）")
    s = sp.add_parser("collect"); s.add_argument("--plans", nargs="*", default=[])
    s.add_argument("--allow-xcheck", action="store_true", help="只有 acvlab 收 xcheck 的參考")
    s = sp.add_parser("digest"); s.add_argument("names", nargs="+", help="NB_DIR 裡的紀錄名（不含 .npz）")
    a = ap.parse_args()
    sys.exit(dict(list=cmd_list, count=cmd_count, run=cmd_run, collect=cmd_collect, traces=cmd_traces, ttl7cells=cmd_ttl7cells,
                  digest=cmd_digest)[a.cmd](a))


if __name__ == "__main__":
    main()
