"""兩階段篩選的實驗：同一份紀錄做出「完整」與幾種「低保真度」版本，用 Rust（只輸出摘要）跑和正式網格同形的單一時間設定 job，
輸出每個 (CPU 容量, CPU 規則, GPU 規則, 慢層, GPU 閒忙) 的回來請求平均 TTFT。不是正式模擬、不進 results。
低保真度：prefix<p>＝只留前 p 比例的請求；sample<R>＝依 session 雜湊只留 R 比例的 session，CPU、GPU 容量 ×R（頻寬不變）。
用法：python lowfi.py <code_dir> <K.json> <trace> <bin> <threads> <work_dir> <out_dir> <variant> ...
"""
import sys, os, json, time, subprocess, resource
code, kjson, trace, binp, threads, work, outd = sys.argv[1:8]
variants = sys.argv[8:]
sys.path.insert(0, os.path.join(code, "p3x")); sys.path.insert(0, code)
import numpy as np
import x1_sim as XS
K = json.load(open(kjson)); XS._K = K
CPUS = [8, 16, 32, 64, 128, 256, 512, 1024, 111.0625]
POLS = ["LRU", "LRU2", "ARC", "S3FIFO", "GDSF", "PENS", "FREQ2Q", "LRU+NONE", "LRU+2HW", "LRU+2HA", "LRU+RPF", "O-BEL", "O-BELS",
        "O-DROP", "O-NOW", "O-PF", "O-ALL"]
GPOLS = ["LRU", "BEL", "TTLA:0.3", "TTLA:10", "TTLA:300", "TTLB:0.3", "TTLB:10", "TTLB:300"]
VK = [(s, g, "cake") for s in ("ssd", "b32", "nfs") for g in ("idle", "busy")]
G = 111.0625

t0 = time.time(); P = XS.prep(trace, "nb"); print("prep_s", round(time.time() - t0, 1), flush=True)
full, tl = P["full"], np.asarray(P["tl"], np.float64)
n = len(full)
sess = np.asarray(P["sessl"], np.int64); kind = np.asarray(P["kindc"], np.int64); parent = np.asarray(P["parentl"], np.int64)
nall = np.asarray(P["nall"], np.int64)
B, S = int(P["B"]), int(P["R"]["n_sess"])


def write_trace(tdir, keep):
    """keep：要留的請求 index（遞增）。session id、block id 不變；請求重新編號；acc 由留下的請求重建。"""
    if os.path.exists(os.path.join(tdir, "DONE")):
        return
    os.makedirs(tdir, exist_ok=True)
    keep = np.asarray(keep, np.int64)
    newi = -np.ones(n, np.int64); newi[keep] = np.arange(len(keep))
    lens = np.fromiter((len(full[i]) for i in keep), np.int64, len(keep))
    off = np.zeros(len(keep) + 1, np.int64); np.cumsum(lens, out=off[1:])
    blk = np.fromiter((b for i in keep for b in full[i]), np.int64, int(off[-1]))
    req = np.repeat(np.arange(len(keep), dtype=np.int64), lens)
    o = np.lexsort((req, blk))
    bs, rs = blk[o], req[o]
    al = np.bincount(bs, minlength=B).astype(np.int64)
    aoff = np.zeros(B + 1, np.int64); np.cumsum(al, out=aoff[1:])
    par = parent[keep].copy()
    m = kind[keep] == 1
    par[m] = newi[par[m]]
    assert (par[m] >= 0).all(), "cont 的 parent 不在留下的請求裡"
    par[~m] = 0
    np.array([len(keep), B, S], np.int64).tofile(os.path.join(tdir, "meta.i64"))
    off.tofile(os.path.join(tdir, "req_off.i64")); blk.astype(np.uint32).tofile(os.path.join(tdir, "req_blk.u32"))
    nall[keep].tofile(os.path.join(tdir, "nall.i64")); kind[keep].tofile(os.path.join(tdir, "kind.i64"))
    sess[keep].tofile(os.path.join(tdir, "sess.i64")); par.tofile(os.path.join(tdir, "parent.i64"))
    tl[keep].tofile(os.path.join(tdir, "t.f64"))
    aoff.tofile(os.path.join(tdir, "acc_off.i64")); rs.tofile(os.path.join(tdir, "acc_req.i64"))
    open(os.path.join(tdir, "DONE"), "w").write("ok\n")


vs = XS.make_views(K, P, VK)
fidx = {"idle": 0, "busy": 1}
fi = XS.f_array(K, P["nmax"], "idle")
pfc = [0.0]
for x in fi:
    pfc.append(pfc[-1] + x)

for var in variants:
    if var == "full":
        keep, scale = np.arange(n), 1.0
    elif var.startswith("prefix"):
        p = float(var[6:]); keep, scale = np.arange(int(p * n)), 1.0
    elif var.startswith("sample"):
        R = float(var[6:])
        h = (sess.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(2**32)
        keep, scale = np.flatnonzero(h < np.uint64(int(R * 2**32))), R
    else:
        raise SystemExit(var)
    tdir = os.path.join(work, f"lf_{trace.replace(':', '_')}_{var}")
    write_trace(tdir, keep)
    rd = os.path.join(work, f"lfrun_{trace.replace(':', '_')}_{var}")
    os.makedirs(rd, exist_ok=True)
    with open(os.path.join(rd, "views.txt"), "w") as f:
        for i, v in enumerate(vs):
            f.write(" ".join([str(i)] + [repr(float(x)) for x in (v.d, v.cpu.r, v.cpu.w, v.cpu.k, v.slow.r, v.slow.w, v.slow.k)]
                             + [v.mode, str(fidx[v.gpu])]) + "\n")
    for v in vs:
        np.asarray(v.f, np.float64).tofile(os.path.join(rd, f"f{fidx[v.gpu]}.f64"))
        np.asarray(v.Fc, np.float64).tofile(os.path.join(rd, f"fc{fidx[v.gpu]}.f64"))
    np.asarray(pfc, np.float64).tofile(os.path.join(rd, "fc_idle.f64"))
    open(os.path.join(rd, "i4.txt"), "w").write(str(K["int4_bytes"]))
    tlast = float(tl[keep][-1])
    meta = {}
    with open(os.path.join(rd, "jobs.txt"), "w") as f:
        j = 0
        for c in CPUS:
            for p in POLS:
                o, a, hk, pf = XS.POLICIES[p]
                for gp in GPOLS:
                    go, gt = XS.parse_gpol(gp)
                    for vi, vk in enumerate(VK):
                        dead = float("inf")
                        f.write(" ".join([f"j{j}", o, a, hk or "-", pf or "-", repr(dead), repr(c * XS.GiB * scale),
                                          str(int(G * XS.GiB * scale)), go, "-" if gt is None else repr(gt), "wb", "cpu", str(vi)]) + "\n")
                        meta[f"j{j}"] = (c, p, gp, vk[0], vk[1])
                        j += 1
    env = dict(os.environ, X1FAST_THREADS=threads, X1FAST_SUMMARY_ONLY="1")
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN); w0 = time.perf_counter()
    cp = subprocess.run(["nice", "-n", "19", binp, tdir, rd], env=env, capture_output=True, text=True)
    wall = time.perf_counter() - w0; r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    if cp.returncode != 0:
        print("FAILED", var, cp.stderr[-3000:], flush=True); sys.exit(2)
    rows = []
    for line in open(os.path.join(rd, "timing.txt")):
        x = line.split()
        if x[0] in meta:
            c, p, gp, sl, gg = meta[x[0]]
            rows.append(dict(variant=var, cpu_GiB=c, policy=p, gpu_policy=gp, slow=sl, gpu=gg, job_s=float(x[1]), mean_ret=float(x[2])))
    import csv
    path = os.path.join(outd, f"lowfi_{var}.csv")
    with open(path, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=["run_id", "ts", "host", "trace"] + list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=time.strftime("%Y-%m-%dT%H:%M:%S%z"), host="acvlab", trace=trace, **r))
    print(json.dumps(dict(variant=var, n_req=int(len(keep)), frac_req=round(len(keep) / n, 4), n_jobs=len(rows), threads=int(threads),
                          wall_s=round(wall, 1), cpu_s=round((r1.ru_utime + r1.ru_stime) - (r0.ru_utime + r0.ru_stime), 1),
                          sum_job_s=round(sum(r["job_s"] for r in rows), 1))), flush=True)
