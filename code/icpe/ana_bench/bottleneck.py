"""瓶頸分析：同一組 job（SemiAnalysis N16 s1、CPU 64 GiB、ssd／idle／cake），比
  (a) Python 參考實作：cProfile 的 tottime 依函式歸到 7 類；
  (b) Rust v1（直接移植）與 (c) Rust 最新版（快取＋共用前段）：程式內分段計時（奈秒）。
類別：0 TTL 到期、1 預取（含騰空間）、2 算版面、3 還原、4 回合結束（refcount）、5 GPU 擠出（含挑誰）、6 CPU 擠出（含挑誰）、7 其他。
不是 sweep、不進 results。用法：python bottleneck.py <code_dir> <K.json> <trace> <trace_dir> <work> <out.jsonl> <bin_v1> <bin_new>"""
import sys, os, json, time, subprocess, cProfile, pstats, shutil
code, kjson, trace, tdir, work, out, bin1, bin2 = sys.argv[1:9]
sys.path.insert(0, os.path.join(code, "p3x")); sys.path.insert(0, code)
import numpy as np
import x1_sim as XS
K = json.load(open(kjson)); XS._K = K; XS._GPU_COLS = True
CAT = ["TTL 到期", "預取（含騰空間）", "算版面", "還原", "回合結束（refcount）", "GPU 擠出（含挑誰）", "CPU 擠出（含挑誰）", "其他"]
FN = {"_g_expire": 0, "advance_prefetch": 1, "_advance_prefetch_gpu": 1, "make_room": 1, "_g_make_room": 1, "unprefetch": 1, "_pf_push": 1,
      "m_cpu": 1, "read": 1, "_step_gpu": 2, "step": 2, "restore": 3, "_turn_end_wb": 4, "_turn_end_wt": 4, "_fix_g": 4, "_fix": 4,
      "_g_add": 4, "_g_remove": 4, "write": 4, "_g_enforce": 5, "_g_evict_wb": 5, "_g_select": 5, "_g_next": 5, "_next_req_bl": 5,
      "_g_release": 5, "_g_free_wt": 5, "enforce": 6, "select": 6, "demote": 6, "drop": 6, "victim_action": 6, "next_req": 6,
      "excl_start": 6, "_order_add": 6, "_order_remove": 6, "key": 6, "dead": 6, "next_access": 6}
JOBS = [("LRU", "LRU"), ("LRU", "TTLA:10"), ("LRU", "TTLB:10"), ("O-BEL", "LRU"), ("O-BEL", "BEL"), ("GDSF", "LRU"),
        ("O-PF", "LRU"), ("LRU+RPF", "TTLA:10")]
CPU, VK = 64.0, [("ssd", "idle", "cake")]
P = XS.prep(trace, "nb")
recs = []
# (a) Python
for pol, gp in JOBS:
    gs = XS.gpu_specs([111.0625], ["wb"], [gp])[0]
    views = XS.make_views(K, P, VK)
    gv = views[0] if XS.POLICIES[pol][0] == "gdsf" else None
    pr = cProfile.Profile(); c0 = time.process_time(); pr.enable()
    XS.Engine(P, pol, CPU * XS.GiB, views, K, gdsf_view=gv, gpu_cap=int(gs["G"] * XS.GiB), gpu_policy=gs["pol"], gpu_write="wb").run()
    pr.disable(); tot = time.process_time() - c0
    st = pstats.Stats(pr).stats
    cat = [0.0] * 8
    for (fname, line, fn), (cc, nc, tt, ct, callers) in st.items():
        cat[FN.get(fn, 7)] += tt
    s = sum(cat)
    recs.append(dict(impl="python", policy=pol, gpu_policy=gp, total_s=round(tot, 3), share={CAT[i]: round(cat[i] / s, 4) for i in range(8)}))
    print(json.dumps(recs[-1], ensure_ascii=False), flush=True)
# (b)(c) Rust
rd0 = os.path.join(work, "bneck_" + trace.replace(":", "_"))
os.makedirs(rd0, exist_ok=True)
vs = XS.make_views(K, P, VK)
with open(os.path.join(rd0, "views.txt"), "w") as f:
    v = vs[0]
    f.write(" ".join(["0"] + [repr(float(x)) for x in (v.d, v.cpu.r, v.cpu.w, v.cpu.k, v.slow.r, v.slow.w, v.slow.k)] + ["cake", "0"]) + "\n")
np.asarray(vs[0].f, np.float64).tofile(os.path.join(rd0, "f0.f64")); np.asarray(vs[0].Fc, np.float64).tofile(os.path.join(rd0, "fc0.f64"))
np.asarray(vs[0].Fc, np.float64).tofile(os.path.join(rd0, "fc_idle.f64"))
open(os.path.join(rd0, "i4.txt"), "w").write(str(K["int4_bytes"]))
with open(os.path.join(rd0, "jobs.txt"), "w") as f:
    for j, (pol, gp) in enumerate(JOBS):
        o, a, h, pf = XS.POLICIES[pol]
        go, gt = XS.parse_gpol(gp)
        f.write(" ".join([f"j{j}", o, a, h or "-", pf or "-", repr(float("inf")), repr(CPU * XS.GiB), str(int(111.0625 * XS.GiB)), go,
                          "-" if gt is None else repr(gt), "wb", "cpu", "0"]) + "\n")
for impl, binp in (("rust_v1", bin1), ("rust_new", bin2)):
    rd = os.path.join(work, f"bneck_{impl}")
    if os.path.exists(rd):
        shutil.rmtree(rd)
    shutil.copytree(rd0, rd)
    subprocess.run(["nice", "-n", "19", binp, tdir, rd], env=dict(os.environ, X1FAST_THREADS="1", X1FAST_SUMMARY_ONLY="1", X1FAST_FORK="0"),
                   check=True, capture_output=True)
    for line in open(os.path.join(rd, "timing.txt")):
        x = line.split()
        if not x[0].startswith("j"):
            continue
        j = int(x[0][1:]); pol, gp = JOBS[j]
        pr = [int(v) for v in x[x.index("prof") + 1].split(",")]
        tot = float(x[1]) * 1e9
        cat = pr + [max(tot - sum(pr), 0)]
        recs.append(dict(impl=impl, policy=pol, gpu_policy=gp, total_s=round(float(x[1]), 4),
                         share={CAT[i]: round(cat[i] / tot, 4) for i in range(8)}))
        print(json.dumps(recs[-1], ensure_ascii=False), flush=True)
with open(out, "a") as fo:
    for r in recs:
        r.update(trace=trace, cpu_GiB=CPU, view="ssd/idle/cake", ts=time.strftime("%Y-%m-%dT%H:%M:%S%z"), host="acvlab",
                 run_id=os.environ.get("RUN_ID", "adhoc"))
        fo.write(json.dumps(r, ensure_ascii=False) + "\n")
