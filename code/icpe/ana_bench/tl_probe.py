"""TraceLab（2.23 億 chunk）在 Rust 上的時間與記憶體，並逐列對正式 tl_main/tl_c1 的 mean_ret。不是 sweep、不進 results。
直接從 npz 寫 Rust 的二進位紀錄（不建 Python 的 list，省記憶體）；id 編號用 x1_nb.check（和 prepare_nb 相同）。
用法：python tl_probe.py <code_dir> <K.json> <npz> <formal_sweep.csv.xz> <work> <bin> <out.jsonl>"""
import sys, os, json, time, subprocess, resource
code, kjson, npz, formal, work, binp, out = sys.argv[1:8]
sys.path.insert(0, os.path.join(code, "p3x")); sys.path.insert(0, code)
import numpy as np
import pandas as pd
import x1_sim as XS
import x1_nb
K = json.load(open(kjson)); XS._K = K
name = os.path.basename(npz)[:-4]
tdir = os.path.join(work, f"trace_nb_{name}")
t0 = time.time()
if not os.path.exists(os.path.join(tdir, "DONE")):
    os.makedirs(tdir, exist_ok=True)
    z = np.load(npz, allow_pickle=False)
    t, sess, parent, kind, nall = z["t"], z["sess"], z["parent"], z["kind"], z["nall"]
    full_off, full_key = z["full_off"], z["full_key"]
    st, ids = x1_nb.check(t, sess, parent, kind, nall, full_off, full_key, name)
    n, B, S = len(t), int(st["n_keys"]), int(sess.max()) + 1
    assert ids.max() < B < 2**32
    nf = np.diff(full_off)
    req = np.repeat(np.arange(n, dtype=np.int64), nf)
    order = np.lexsort((req, ids))
    acc_req = req[order]
    aoff = np.zeros(B + 1, np.int64); np.cumsum(np.bincount(ids, minlength=B), out=aoff[1:])
    assert len(acc_req) == len(ids)
    np.array([n, B, S], np.int64).tofile(os.path.join(tdir, "meta.i64"))
    full_off.astype(np.int64).tofile(os.path.join(tdir, "req_off.i64")); ids.astype(np.uint32).tofile(os.path.join(tdir, "req_blk.u32"))
    nall.astype(np.int64).tofile(os.path.join(tdir, "nall.i64")); kind.astype(np.int64).tofile(os.path.join(tdir, "kind.i64"))
    sess.astype(np.int64).tofile(os.path.join(tdir, "sess.i64"))
    par = parent.astype(np.int64).copy(); par[kind != 1] = 0
    par.tofile(os.path.join(tdir, "parent.i64"))
    (t.astype(np.float64) - float(t[0])).tofile(os.path.join(tdir, "t.f64"))
    aoff.tofile(os.path.join(tdir, "acc_off.i64")); acc_req.tofile(os.path.join(tdir, "acc_req.i64"))
    open(os.path.join(tdir, "DONE"), "w").write("ok\n")
    nmax = int(nall.max()) + 1; nreq = n
    del z, full_key, ids, req, order, acc_req
else:
    nall = np.fromfile(os.path.join(tdir, "nall.i64"), np.int64); nmax = int(nall.max()) + 1; nreq = len(nall)
conv_s = time.time() - t0
VK = [(s, g, "cake") for s in ("ssd", "b32", "nfs") for g in ("idle", "busy")]
vs = [XS.View(K, s, g, m, nmax, nreq) for (s, g, m) in VK]
rd = os.path.join(work, f"tlrun_{name}")
os.makedirs(rd, exist_ok=True)
fidx = {"idle": 0, "busy": 1}
with open(os.path.join(rd, "views.txt"), "w") as f:
    for i, v in enumerate(vs):
        f.write(" ".join([str(i)] + [repr(float(x)) for x in (v.d, v.cpu.r, v.cpu.w, v.cpu.k, v.slow.r, v.slow.w, v.slow.k)] + [v.mode, str(fidx[v.gpu])]) + "\n")
for v in vs:
    np.asarray(v.f, np.float64).tofile(os.path.join(rd, f"f{fidx[v.gpu]}.f64")); np.asarray(v.Fc, np.float64).tofile(os.path.join(rd, f"fc{fidx[v.gpu]}.f64"))
np.asarray(vs[0].Fc, np.float64).tofile(os.path.join(rd, "fc_idle.f64"))
open(os.path.join(rd, "i4.txt"), "w").write(str(K["int4_bytes"]))
JOBS = [("LRU", "LRU", list(range(6))), ("O-BEL", "BEL", list(range(6))), ("LRU", "TTLA:10", [0]), ("O-PF", "LRU", [4])]
CPU = 64.0
with open(os.path.join(rd, "jobs.txt"), "w") as f:
    for j, (p, gp, vids) in enumerate(JOBS):
        o, a, h, pf = XS.POLICIES[p]; go, gt = XS.parse_gpol(gp)
        f.write(" ".join([f"j{j}", o, a, h or "-", pf or "-", repr(float("inf")), repr(CPU * XS.GiB), str(int(111.0625 * XS.GiB)), go,
                          "-" if gt is None else repr(gt), "wb", "cpu", ",".join(map(str, vids))]) + "\n")
r0 = resource.getrusage(resource.RUSAGE_CHILDREN); w0 = time.perf_counter()
cp = subprocess.run(["/usr/bin/time", "-v", "nice", "-n", "19", binp, tdir, rd], env=dict(os.environ, X1FAST_THREADS="1", X1FAST_SUMMARY_ONLY="1"),
                    capture_output=True, text=True)
wall = time.perf_counter() - w0
maxrss = [l for l in cp.stderr.splitlines() if "Maximum resident" in l]
assert cp.returncode == 0, cp.stderr[-2000:]
fm = pd.read_csv(formal, dtype={"mean_ret": str})
fm = fm[(fm.restore == "cake") & (fm.pf_target == "cpu") & (fm.cpu_GiB == CPU)]
fmd = {(r.policy, r.gpu_policy, r.slow, r.gpu): r.mean_ret for r in fm.itertuples()}
recs = []
for line in open(os.path.join(rd, "timing.txt")):
    x = line.split()
    if not x[0].startswith("j"):
        continue
    p, gp, vids = JOBS[int(x[0][1:])]
    means = [float(v) for v in x[2:2 + len(vids)]]
    eq = [float(f"{m:.6f}") == float(fmd[(p, gp, VK[k][0], VK[k][1])]) for m, k in zip(means, vids)]
    recs.append(dict(trace=f"nb:{name}", policy=p, gpu_policy=gp, n_views=len(vids), cpu_GiB=CPU, rust_job_s=float(x[1]),
                     rows_equal_formal=sum(eq), rows=len(eq)))
summary = dict(trace=f"nb:{name}", n_req=nreq, convert_s=round(conv_s, 1), rust_proc_wall_s=round(wall, 1),
               rust_maxrss=maxrss[0].strip() if maxrss else None, jobs=recs,
               ts=time.strftime("%Y-%m-%dT%H:%M:%S%z"), host="acvlab", run_id=os.environ.get("RUN_ID", "adhoc"))
open(out, "a").write(json.dumps(summary, ensure_ascii=False) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=1))
