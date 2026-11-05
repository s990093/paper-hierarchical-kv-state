"""照正式 x1 sweep 的切法（jobs_for：TTL／預取的 job 一個時間設定一個 job、GDSF 每個 (慢層, GPU) 一個、其他 6 個時間設定合成一個 job）
跑一整個 block，用 Rust（只輸出摘要）比不同版本的時間，並逐列對正式 sweep 的 mean_ret（正式 CSV 的精度）。不是正式模擬、不進 results。
用法：python grid.py <code_dir> <K.json> <trace> <trace_dir> <formal_sweep.csv(.xz)> <work_dir> <out_dir> <threads> <label=bin[:fork]> ...
"""
import sys, os, json, time, subprocess, resource, csv
code, kjson, trace, tdir, formal, work, outd, threads = sys.argv[1:9]
specs = sys.argv[9:]
sys.path.insert(0, os.path.join(code, "p3x")); sys.path.insert(0, code)
import numpy as np
import pandas as pd
import x1_sim as XS
K = json.load(open(kjson)); XS._K = K
CPUS = [8, 16, 32, 64, 128, 256, 512, 1024, 111.0625]
POLS = ["LRU", "LRU2", "ARC", "S3FIFO", "GDSF", "PENS", "FREQ2Q", "LRU+NONE", "LRU+2HW", "LRU+2HA", "LRU+RPF", "O-BEL", "O-BELS",
        "O-DROP", "O-NOW", "O-PF", "O-ALL"]
GPOLS = ["LRU", "BEL", "TTLA:0.3", "TTLA:10", "TTLA:300", "TTLB:0.3", "TTLB:10", "TTLB:300"]
SL, GP = ["ssd", "b32", "nfs"], ["idle", "busy"]
VK = [(s, g, "cake") for s in SL for g in GP]
VID = {k: i for i, k in enumerate(VK)}
G = 111.0625

P = XS.prep(trace, "nb")
gspecs = XS.gpu_specs([G], ["wb"], GPOLS, ["cpu"])
J = XS.jobs_for([trace], ["nb"], CPUS, POLS, modes=["cake"], slows=SL, gspecs=gspecs)
print("jobs", len(J), flush=True)
rd0 = os.path.join(work, "grid_" + trace.replace(":", "_"))
os.makedirs(rd0, exist_ok=True)
vs = XS.make_views(K, P, VK)
fidx = {"idle": 0, "busy": 1}
with open(os.path.join(rd0, "views.txt"), "w") as f:
    for i, v in enumerate(vs):
        f.write(" ".join([str(i)] + [repr(float(x)) for x in (v.d, v.cpu.r, v.cpu.w, v.cpu.k, v.slow.r, v.slow.w, v.slow.k)]
                         + [v.mode, str(fidx[v.gpu])]) + "\n")
for v in vs:
    np.asarray(v.f, np.float64).tofile(os.path.join(rd0, f"f{fidx[v.gpu]}.f64"))
    np.asarray(v.Fc, np.float64).tofile(os.path.join(rd0, f"fc{fidx[v.gpu]}.f64"))
fi = XS.f_array(K, P["nmax"], "idle")
pfc = [0.0]
for x in fi:
    pfc.append(pfc[-1] + x)
np.asarray(pfc, np.float64).tofile(os.path.join(rd0, "fc_idle.f64"))
open(os.path.join(rd0, "i4.txt"), "w").write(str(K["int4_bytes"]))
meta = {}
with open(os.path.join(rd0, "jobs.txt"), "w") as f:
    for j, (tr, ru, c, p, keys, gs, nz, sc) in enumerate(J):
        o, a, h, pf = XS.POLICIES[p]
        go, gt = XS.parse_gpol(gs["pol"])
        f.write(" ".join([f"j{j}", o, a, h or "-", pf or "-", repr(float("inf")), repr(c * XS.GiB), str(int(gs["G"] * XS.GiB)), go,
                          "-" if gt is None else repr(gt), gs["write"], gs["pf"], ",".join(str(VID[k]) for k in keys)]) + "\n")
        meta[f"j{j}"] = (c, p, gs["pol"], keys)

fm = pd.read_csv(formal, dtype={"mean_ret": str})
fm = fm[(fm.restore == "cake") & (fm.pf_target == "cpu")]
fmd = {(r.cpu_GiB, r.policy, r.gpu_policy, r.slow, r.gpu): r.mean_ret for r in fm.itertuples()}

import shutil
for sp in specs:
    label, rest = sp.split("=", 1)
    fork = rest.endswith(":fork")
    binp = rest[:-5] if fork else rest
    rd = os.path.join(work, f"grid_{label}")
    if os.path.exists(rd):
        shutil.rmtree(rd)
    shutil.copytree(rd0, rd)
    env = dict(os.environ, X1FAST_THREADS=threads, X1FAST_SUMMARY_ONLY="1", X1FAST_FORK="1" if fork else "0")
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN); w0 = time.perf_counter()
    cp = subprocess.run(["nice", "-n", "19", binp, tdir, rd], env=env, capture_output=True, text=True)
    wall = time.perf_counter() - w0; r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    if cp.returncode != 0:
        print("FAILED", label, cp.stderr[-3000:], flush=True); sys.exit(2)
    rows, nmatch, ntot, rst = [], 0, 0, []
    for line in open(os.path.join(rd, "timing.txt")):
        x = line.split()
        if x[0] not in meta:
            continue
        c, p, gp, keys = meta[x[0]]
        means = [float(v) for v in x[2:2 + len(keys)]]
        if "fork_rstar" in x:
            rst.append(int(x[x.index("fork_rstar") + 1]))
        for k, m in zip(keys, means):
            fv = fmd[(c, p, gp, k[0], k[1])]
            ok = round(m, 6) == float(fv) or float(f"{m:.6f}") == float(fv)
            nmatch += ok; ntot += 1
    rec = dict(label=label, bin=binp, fork=fork, threads=int(threads), n_jobs=len(meta), rows_checked=ntot, rows_equal_formal=nmatch,
               wall_s=round(wall, 1), cpu_s=round((r1.ru_utime + r1.ru_stime) - (r0.ru_utime + r0.ru_stime), 1),
               fork_groups=len(rst) and None, fork_rstar_frac_median=round(float(np.median(rst)) / len(P["full"]), 4) if rst else None,
               stderr_tail=cp.stderr.strip().splitlines()[-1] if cp.stderr.strip() else "",
               ts=time.strftime("%Y-%m-%dT%H:%M:%S%z"), host="acvlab", run_id=os.environ.get("RUN_ID", "adhoc"))
    with open(os.path.join(outd, "grid.jsonl"), "a") as fo:
        fo.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
