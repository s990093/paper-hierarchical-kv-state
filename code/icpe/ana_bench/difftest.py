"""Rust 全移植（x1fast）對原本 x1_sim.Engine 的逐位元對拍＋計時。不是 sweep、不進 results。
用法：python difftest.py <code_dir> <K.json> <trace> <matrix> <out.jsonl> <bin> <work_dir> [--no-ref]
matrix：broad（全部 17 個 CPU 規則 × GPU 設定 × 小容量，給小紀錄用）、narrow（代表組合，給大紀錄用）、speed（只計時）。
雜湊和 bench.py 相同（逐請求陣列＋summarize 的字典）。"""
import sys, os, json, time, hashlib, subprocess, resource
code, kjson, trace, matrix, out, binp, work = sys.argv[1:8]
noref = "--no-ref" in sys.argv
sys.path.insert(0, os.path.join(code, "p3x")); sys.path.insert(0, code)
import numpy as np
import x1_sim as XS
K = json.load(open(kjson)); XS._K = K; XS._GPU_COLS = True

t = time.time(); P = XS.prep(trace, "nb" if trace.startswith("nb:") else "R1"); load_s = time.time() - t
nreq = len(P["full"])
tdir = os.path.join(work, "trace_" + trace.replace(":", "_"))
if not os.path.exists(os.path.join(tdir, "DONE")):
    os.makedirs(tdir, exist_ok=True)
    full = P["full"]
    lens = np.fromiter((len(x) for x in full), np.int64, nreq)
    off = np.zeros(nreq + 1, np.int64); np.cumsum(lens, out=off[1:])
    blk = np.fromiter((b for x in full for b in x), np.int64, int(off[-1]))
    B, S = int(P["B"]), int(P["R"]["n_sess"])
    assert len(blk) == 0 or (blk.max() < B < 2**32 and blk.min() >= 0)
    acc = P["acc"]
    al = np.zeros(B, np.int64)
    for b, lst in acc.items():
        al[b] = len(lst)
    aoff = np.zeros(B + 1, np.int64); np.cumsum(al, out=aoff[1:])
    areq = np.empty(int(aoff[-1]), np.int64)
    for b, lst in acc.items():
        areq[aoff[b]:aoff[b + 1]] = lst
    assert len(areq) == len(blk), (len(areq), len(blk))       # 用 A 驗 B：acc 的總數＝full 的總數
    assert all(np.all(np.diff(areq[aoff[b]:aoff[b + 1]]) > 0) for b in list(acc)[:2000])
    np.array([nreq, B, S], np.int64).tofile(os.path.join(tdir, "meta.i64"))
    off.tofile(os.path.join(tdir, "req_off.i64")); blk.astype(np.uint32).tofile(os.path.join(tdir, "req_blk.u32"))
    np.asarray(P["nall"], np.int64).tofile(os.path.join(tdir, "nall.i64"))
    np.asarray(P["kindc"], np.int64).tofile(os.path.join(tdir, "kind.i64"))
    np.asarray(P["sessl"], np.int64).tofile(os.path.join(tdir, "sess.i64"))
    np.asarray(P["parentl"], np.int64).tofile(os.path.join(tdir, "parent.i64"))
    np.asarray(P["tl"], np.float64).tofile(os.path.join(tdir, "t.f64"))
    aoff.tofile(os.path.join(tdir, "acc_off.i64")); areq.tofile(os.path.join(tdir, "acc_req.i64"))
    open(os.path.join(tdir, "DONE"), "w").write("ok\n")

SL = ["ssd", "b32", "nfs"]
VKEYS = [(s, g, m) for s in SL for g in ("idle", "busy") for m in ("cake", "load")]
VID = {k: i for i, k in enumerate(VKEYS)}
ALLC = [k for k in VKEYS if k[2] == "cake"]
ALLV = list(VKEYS)

if matrix == "broad":
    CPUS = [8.0, 64.0]
    POLS = ["LRU", "LRU2", "ARC", "S3FIFO", "GDSF", "PENS", "FREQ2Q", "PRED", "LRU+NONE", "LRU+2HW", "LRU+2HA", "LRU+RPF",
            "O-BEL", "O-BELS", "O-DROP", "O-NOW", "O-PF", "O-ALL", "O-DROP-H3600"]
    GSETS = [None, ("wb", "LRU", "cpu"), ("wb", "BEL", "cpu"), ("wb", "TTLA:10", "cpu"), ("wb", "TTLB:0.3", "cpu"),
             ("wt", "LRU", "cpu"), ("wt", "TTLA:300", "cpu"), ("wb", "TTLB:10", "gpu"), ("wt", "LRU", "gpu")]
elif matrix == "narrow":
    CPUS = [64.0]
    POLS = ["LRU", "LRU2", "ARC", "S3FIFO", "GDSF", "PENS", "LRU+RPF", "O-BEL", "O-BELS", "O-DROP", "O-NOW", "O-PF", "O-ALL"]
    GSETS = [None, ("wb", "LRU", "cpu"), ("wb", "BEL", "cpu"), ("wb", "TTLA:10", "cpu"), ("wt", "TTLB:10", "cpu"), ("wb", "TTLB:10", "gpu")]
else:   # speed：正式網格裡最常見的形狀
    CPUS = [64.0]
    POLS = ["LRU", "LRU2", "ARC", "S3FIFO", "GDSF", "PENS", "O-BEL", "O-PF", "LRU+RPF"]
    GSETS = [("wb", "LRU", "cpu"), ("wb", "TTLA:10", "cpu"), ("wb", "TTLB:10", "cpu"), ("wb", "BEL", "cpu")]

JOBS = []
for c in CPUS:
    for p in POLS:
        o, a, h, pf = XS.POLICIES[p]
        for gsx in GSETS:
            if gsx is not None and gsx[2] == "gpu" and not pf:
                continue
            gs = None if gsx is None else dict(G=111.0625, write=gsx[0], pol=gsx[1], pf=gsx[2])
            tdep = pf or (gs is not None and XS.parse_gpol(gs["pol"])[1] is not None)
            if tdep:
                keys = [("nfs", "busy", "cake")] if len(JOBS) % 2 else [("b32", "idle", "load")]
            elif o == "gdsf":
                keys = [("ssd", "busy", "cake"), ("ssd", "busy", "load")]
            else:
                keys = ALLV if matrix == "broad" else ALLC
            JOBS.append((p, c, gs, keys))

# ---- 時間設定的常數（用原本的 View 建，保證同一份 f、Fc、頻寬）
rdir = os.path.join(work, f"run_{matrix}_{trace.replace(':', '_')}")
os.makedirs(rdir, exist_ok=True)
vs = XS.make_views(K, P, VKEYS)
fidx = {"idle": 0, "busy": 1}
with open(os.path.join(rdir, "views.txt"), "w") as f:
    for i, v in enumerate(vs):
        f.write(" ".join([str(i)] + [repr(float(x)) for x in (v.d, v.cpu.r, v.cpu.w, v.cpu.k, v.slow.r, v.slow.w, v.slow.k)]
                         + [v.mode, str(fidx[v.gpu])]) + "\n")
for v in vs:
    np.asarray(v.f, np.float64).tofile(os.path.join(rdir, f"f{fidx[v.gpu]}.f64"))
    np.asarray(v.Fc, np.float64).tofile(os.path.join(rdir, f"fc{fidx[v.gpu]}.f64"))
fi = XS.f_array(K, P["nmax"], "idle")
pfc = [0.0]
for x in fi:
    pfc.append(pfc[-1] + x)
np.asarray(pfc, np.float64).tofile(os.path.join(rdir, "fc_idle.f64"))
open(os.path.join(rdir, "i4.txt"), "w").write(str(K["int4_bytes"]))
with open(os.path.join(rdir, "jobs.txt"), "w") as f:
    for j, (p, c, gs, keys) in enumerate(JOBS):
        o, a, h, pf = XS.POLICIES[p]
        dead = float(P["tl"][-1] - float(p.split("-H")[1])) if "-H" in p else float("inf")
        if gs is None:
            gfields = ["0", "-", "-", "wb", "cpu"]
        else:
            go, gt = XS.parse_gpol(gs["pol"])
            gfields = [str(int(gs["G"] * XS.GiB)), go, "-" if gt is None else repr(gt), gs["write"], gs["pf"]]
        f.write(" ".join([f"j{j}", o, a, h or "-", pf or "-", repr(dead), repr(c * XS.GiB)] + gfields
                         + [",".join(str(VID[k]) for k in keys)]) + "\n")


class E:
    pass


def digest(views, eng):
    h = hashlib.sha256()
    for v in views:
        for a in (v.ttft, v.rest, v.nC, v.nS, v.nR):
            h.update(np.ascontiguousarray(a).tobytes())
        sm = XS.summarize(P, v, eng, gcols=True)
        h.update(json.dumps({k: (repr(x) if isinstance(x, float) else str(x)) for k, x in sorted(sm.items())}).encode())
    return h.hexdigest()[:16]


# ---- Rust：一個行程跑全部 job（紀錄只讀一次）
r0 = resource.getrusage(resource.RUSAGE_CHILDREN); w0 = time.perf_counter()
cp = subprocess.run([binp, tdir, rdir], capture_output=True, text=True)
rust_wall = time.perf_counter() - w0; r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
rust_cpu = (r1.ru_utime + r1.ru_stime) - (r0.ru_utime + r0.ru_stime)
if cp.returncode != 0:
    print("RUST FAILED", cp.returncode, cp.stderr[-3000:]); sys.exit(2)
tim = {}
for line in open(os.path.join(rdir, "timing.txt")):
    x = line.split()
    tim[x[0]] = float(x[1])          # 新版多一行 all_jobs_wall_s <秒> threads <n>

nsame = ndiff = 0
with open(out, "a") as fo:
    for j, (p, c, gs, keys) in enumerate(JOBS):
        od = os.path.join(rdir, "out", f"j{j}")
        views = XS.make_views(K, P, keys)
        for k, v in enumerate(views):
            v.ttft = np.fromfile(os.path.join(od, f"v{k}_ttft.f64"), np.float64)
            v.rest = np.fromfile(os.path.join(od, f"v{k}_rest.f64"), np.float64)
            v.nC = np.fromfile(os.path.join(od, f"v{k}_nc.i32"), np.int32)
            v.nS = np.fromfile(os.path.join(od, f"v{k}_ns.i32"), np.int32)
            v.nR = np.fromfile(os.path.join(od, f"v{k}_nr.i32"), np.int32)
            v.nG = np.fromfile(os.path.join(od, f"v{k}_ng.i32"), np.int32)
            ch = np.fromfile(os.path.join(od, f"v{k}_chan.f64"), np.float64).tolist()
            cb = [int(x) for x in np.fromfile(os.path.join(od, f"v{k}_chanb.i64"), np.int64).tolist()]
            v.cpu.t_free, v.cpu.t_free_w, v.cpu.busy_r, v.cpu.busy_w, v.slow.t_free, v.slow.t_free_w, v.slow.busy_r, v.slow.busy_w = ch
            v.cpu.bytes_r, v.cpu.bytes_w, v.slow.bytes_r, v.slow.bytes_w, v.pf_bytes = cb
        ei = [int(x) for x in np.fromfile(os.path.join(od, "eng.i64"), np.int64).tolist()]
        e = E()
        e.n_demote, e.n_drop, e.n_unpf, e.n_gevict, e.n_gexpire, e.g_wb_blocks, _ = ei
        e.res_sum = float(np.fromfile(os.path.join(od, "eng.f64"), np.float64)[0])
        e.G = 0 if gs is None else int(gs["G"] * XS.GiB)
        d_rust = digest(views, e)
        d_ref, ref_cpu = None, None
        if not noref:
            c0 = time.process_time()
            rv = XS.make_views(K, P, keys)
            gv = rv[0] if XS.POLICIES[p][0] == "gdsf" else None
            gkw = {} if gs is None else dict(gpu_cap=int(gs["G"] * XS.GiB), gpu_policy=gs["pol"], gpu_write=gs["write"], pf_target=gs["pf"])
            eng = XS.Engine(P, p, c * XS.GiB, rv, K, gdsf_view=gv, **gkw).run()
            ref_cpu = time.process_time() - c0
            d_ref = digest(rv, eng)
            if d_ref == d_rust:
                nsame += 1
            else:
                ndiff += 1
        rec = dict(trace=trace, matrix=matrix, job=f"j{j}", policy=p, cpu_GiB=c,
                   gpu="off" if gs is None else f"{gs['write']}:{gs['pol']}:pf{gs['pf']}", n_views=len(keys),
                   views=";".join("/".join(k) for k in keys), rust_s=round(tim[f"j{j}"], 4), ref_cpu_s=None if ref_cpu is None else round(ref_cpu, 3),
                   speedup=None if ref_cpu is None else round(ref_cpu / tim[f"j{j}"], 1), sha_rust=d_rust, sha_ref=d_ref,
                   identical=None if d_ref is None else d_ref == d_rust,
                   ts=time.strftime("%Y-%m-%dT%H:%M:%S%z"), host="acvlab", run_id=os.environ.get("RUN_ID", "adhoc"))
        fo.write(json.dumps(rec) + "\n"); fo.flush()
        if rec["identical"] is False:
            print("DIFF", json.dumps(rec), flush=True)
print(json.dumps(dict(trace=trace, matrix=matrix, n_jobs=len(JOBS), identical=nsame, different=ndiff, rust_proc_wall_s=round(rust_wall, 2),
                      rust_proc_cpu_s=round(rust_cpu, 2), rust_trace_load_s=tim.get("trace_load_s"), py_prep_s=round(load_s, 1))), flush=True)
