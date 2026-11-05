"""跑分：同一組 x1 job（不是 sweep、不進 results），量 wall 與 CPU 時間，並對每個 job 的逐請求輸出算 sha256。
用法：python bench.py <label> <code_dir> <K.json> <trace> <reps> <out.jsonl>"""
import sys, os, json, time, hashlib, platform
label, code, kjson, trace, reps, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6]
sys.path.insert(0, os.path.join(code, "p3x")); sys.path.insert(0, code)
import numpy as np
import x1_sim as XS
K = json.load(open(kjson)); XS._K = K; XS._GPU_COLS = True
ALL6 = [(s, g, "cake") for s in ("ssd", "b32", "nfs") for g in ("idle", "busy")]
JOBS = [("LRU", "TTLB:10", [("ssd", "idle", "cake")]), ("O-BEL", "LRU", [("ssd", "idle", "cake")]),
        ("LRU+RPF", "TTLA:10", [("ssd", "idle", "cake")]), ("GDSF", "LRU", ALL6)]
CPU = 64.0
t = time.time(); P = XS.prep(trace, "nb"); load_s = time.time() - t
impl = f"{platform.python_implementation()} {platform.python_version()}"
with open(out, "a") as fo:
    for rep in range(reps):
        for pol, gpol, keys in JOBS:
            gs = XS.gpu_specs([111.0625], ["wb"], [gpol])[0]
            w0, c0 = time.perf_counter(), time.process_time()
            views = XS.make_views(K, P, keys)
            gv = views[0] if XS.POLICIES[pol][0] == "gdsf" else None
            eng = XS.Engine(P, pol, CPU * XS.GiB, views, K, gdsf_view=gv, gpu_cap=int(gs["G"] * XS.GiB), gpu_policy=gs["pol"],
                            gpu_write=gs["write"], pf_target=gs.get("pf", "cpu")).run()
            wall, cpu = time.perf_counter() - w0, time.process_time() - c0
            h = hashlib.sha256()
            for v in views:
                for a in (v.ttft, v.rest, v.nC, v.nS, v.nR):
                    h.update(np.ascontiguousarray(a).tobytes())
                sm = XS.summarize(P, v, eng, gcols=True)
                h.update(json.dumps({k: (repr(x) if isinstance(x, float) else str(x)) for k, x in sorted(sm.items())}).encode())
            rec = dict(label=label, impl=impl, trace=trace, cpu_GiB=CPU, policy=pol, gpu_policy=gpol, n_views=len(keys), rep=rep,
                       wall_s=round(wall, 3), cpu_s=round(cpu, 3), load_s=round(load_s, 1), sha256=h.hexdigest()[:16],
                       x1_sim=os.path.realpath(XS.__file__), ts=time.strftime("%Y-%m-%dT%H:%M:%S%z"), host="acvlab",
                       run_id=os.environ.get("RUN_ID", "adhoc"))
            fo.write(json.dumps(rec) + "\n"); fo.flush(); print(json.dumps(rec), flush=True)
