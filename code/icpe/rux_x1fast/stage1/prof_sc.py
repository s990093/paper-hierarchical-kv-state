"""icpe-rux 第 1 段（b）：慢層容量上限的「時間花在哪」。一個 job、不是 sweep；只讀 x1_sim／x1_rust，不改它們。

用法（repo 根目錄；env-rux.sh 已 source）：
  python -I code/icpe/rux_x1fast/stage1/prof_sc.py --trace nb:bailian_coder --reps 3 --out <run>/out [--cprofile] [--rust]

每個設定（同一個 job：CPU LRU × GPU-LRU wb 111.0625 GiB、CPU 111.0625 GiB、Cake）：
  nocap6  ：沒有上限，6 個時間設定（ssd／b32／nfs × idle／busy）＝v5 支援的對應 job
  nocap1  ：沒有上限，1 個時間設定（ssd idle）＝慢層 GDSF 的對應 job（jobs_for 把 GDSF 按 (慢層, GPU) 拆）
  sc:<P>  ：frac 0.5、慢層規則 P（LRU、LRU2、ARC、S3FIFO、PENS、BEL：6 個時間設定；GDSF：1 個）
Python 計時＝time.process_time()（單執行緒）；Rust（--rust，只有 nocap*：v5 不支援上限）＝x1fast 批次模式自己量的引擎時間。
每列都有 run_id、ts、host；digest＝逐請求陣列（ttft、rest、nC、nS、nR）的 sha256 前 16 碼。
--cprofile：每個設定另外跑一次 cProfile（不計時），並記慢層結構的操作次數（包一層計數，不改語意）。"""
import argparse, cProfile, hashlib, io, json, os, pstats, socket, sys, time
from datetime import datetime

ap = argparse.ArgumentParser()
ap.add_argument("--trace", required=True)
ap.add_argument("--rule", default="R1")
ap.add_argument("--cpu", type=float, default=111.0625)
ap.add_argument("--gpu", type=float, default=111.0625)
ap.add_argument("--policy", default="LRU")
ap.add_argument("--gpu-policy", default="LRU")
ap.add_argument("--configs", nargs="+", default=["nocap6", "nocap1", "sc:LRU", "sc:LRU2", "sc:ARC", "sc:S3FIFO", "sc:PENS", "sc:BEL", "sc:GDSF"])
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--cprofile", nargs="*", default=None, help="要 cProfile 的設定；只給旗標＝全部")
ap.add_argument("--rust", action="store_true")
ap.add_argument("--out", required=True)
a = ap.parse_args()

REPO = os.environ["REPO"]
sys.path.insert(0, os.path.join(REPO, "code", "p3x"))
sys.path.insert(0, os.path.join(REPO, "code", "icpe"))
sys.path.insert(0, os.path.join(REPO, "code"))
import numpy as np
import p3_fvar
os.makedirs(a.out, exist_ok=True)
note = {}
XS = p3_fvar.patch_x1("V1", os.path.join(REPO, "results/icpe_h200/p01/calib"), a.out, note)
K = XS.constants()
XS._K = K
XS._GPU_COLS = True
RUN_ID = os.environ.get("RUN_ID", "adhoc")
HOST = socket.gethostname()
OUT = open(os.path.join(a.out, "rows.jsonl"), "a")


def la():
    return float(open("/proc/loadavg").read().split()[0])


def emit(**r):
    r = dict(run_id=RUN_ID, ts=datetime.now().astimezone().isoformat(timespec="seconds"), host=HOST, trace=a.trace, rule=a.rule,
             cpu_GiB=a.cpu, policy=a.policy, gpu_policy=a.gpu_policy, threads=1, **r)
    OUT.write(json.dumps(r) + "\n")
    OUT.flush()
    print(json.dumps(r), flush=True)


t0 = time.time()
P = XS.prep(a.trace, a.rule)
prep_s = time.time() - t0
V6 = [(s, g, "cake") for s in ("ssd", "b32", "nfs") for g in ("idle", "busy")]
V1 = [("ssd", "idle", "cake")]
emit(kind="prep", prep_s=round(prep_s, 2), n_req=len(P["full"]), n_sess=int(P["R"]["n_sess"]), B=int(P["B"]),
     n_blocks_full=int(P["meta"]["n_blocks_full"]), chunks=int(sum(map(len, P["full"]))), load_1m=la())


def spec(cfg):
    if cfg == "nocap6":
        return V6, None
    if cfg == "nocap1":
        return V1, None
    pol = cfg.split(":")[1]
    return (V1 if pol == "GDSF" else V6), pol


def gkw():
    return dict(gpu_cap=int(a.gpu * XS.GiB), gpu_policy=a.gpu_policy, gpu_write="wb")


def make(cfg):
    vk, pol = spec(cfg)
    views = XS.make_views(K, P, vk)
    kw = gkw()
    if pol is not None:
        kw.update(slow_cap=0.5, slow_policy=pol)
    gdsf = XS.POLICIES[a.policy][0] == "gdsf" or pol == "GDSF"
    return views, XS.Engine(P, a.policy, a.cpu * XS.GiB, views, K, gdsf_view=views[0] if gdsf else None, **kw)


def digest(views):
    h = hashlib.sha256()
    for v in views:
        for x in (v.ttft, v.rest, v.nC, v.nS, v.nR):
            h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()[:16]


def counters(e):
    return dict(n_demote=e.n_demote, n_drop=e.n_drop, w_slow_blocks=e.w_slow_blocks, n_gevict=getattr(e, "n_gevict", None),
                sc_cap=getattr(e, "sc_cap", -1), n_sc_clean=getattr(e, "n_sc_clean", 0), n_sc_drop=getattr(e, "n_sc_drop", 0))


# ---- Python：交錯計時
for rep in range(a.reps):
    for cfg in a.configs:
        views, e = make(cfg)
        l0 = la(); w0 = time.perf_counter(); c0 = time.process_time()
        e.run()
        c1 = time.process_time(); w1 = time.perf_counter()
        emit(kind="py_time", engine="py", config=cfg, rep=rep, n_views=len(views), cpu_s=round(c1 - c0, 4), wall_s=round(w1 - w0, 4),
             load_1m_before=l0, load_1m_after=la(), digest=digest(views), **counters(e))

# ---- Rust v5（只有沒有上限的設定）
if a.rust:
    import shutil, subprocess
    import x1_rust as XR
    tdir, tsha = XR.prepare_trace(P, a.trace, a.rule)
    for rep in range(a.reps):
        for cfg in a.configs:
            if not cfg.startswith("nocap"):
                continue
            vk, _ = spec(cfg)
            views = XS.make_views(K, P, vk)
            jd = XR.write_job(XS, P, a.policy, a.cpu * XS.GiB, views, K, gkw())
            env = dict(os.environ, X1FAST_THREADS="1", X1FAST_FORK="0", X1FAST_SUMMARY_ONLY="0")
            env.pop("X1FAST_SERVE", None)
            l0 = la(); w0 = time.perf_counter()
            cp = subprocess.run([XR.binary(), tdir, jd], env=env, capture_output=True, text=True)
            w1 = time.perf_counter()
            assert cp.returncode == 0, cp.stderr[-3000:]
            tm = open(os.path.join(jd, "timing.txt")).read().split("\n")
            load_s = float(tm[0].split()[1])
            run_s = float([x for x in tm if x.startswith("j0 ")][0].split()[1])
            XR.read_outputs(jd, P, views, gkw(), a.policy)
            shutil.rmtree(jd, ignore_errors=True)
            emit(kind="rust_time", engine="rust_v5", config=cfg, rep=rep, n_views=len(views), cpu_s=round(run_s, 5),
                 wall_s=round(w1 - w0, 4), rust_trace_load_s=round(load_s, 4), load_1m_before=l0, load_1m_after=la(),
                 digest=digest(views), bin=XR.binary())

# ---- cProfile＋操作次數（不計時）
if a.cprofile is not None:
    for cfg in (a.cprofile or a.configs):
        views, e = make(cfg)
        ops = {}
        if getattr(e, "sc", None) is not None:
            so = e.sorder
            for nm in ("add", "remove", "select"):
                f = getattr(so, nm)

                def w(*x, _f=f, _nm=nm, **k):
                    ops[_nm] = ops.get(_nm, 0) + 1
                    if _nm == "select":
                        ops["select_mem_sum"] = ops.get("select_mem_sum", 0) + len(so.mem)
                        ops["select_mem_max"] = max(ops.get("select_mem_max", 0), len(so.mem))
                    return _f(*x, **k)
                setattr(so, nm, w)
            cq = e.clean_q
            ops["_cq"] = cq
        pr = cProfile.Profile()
        l0 = la(); c0 = time.process_time()
        pr.enable(); e.run(); pr.disable()
        c1 = time.process_time()
        tag = cfg.replace(":", "_")
        pr.dump_stats(os.path.join(a.out, f"cprofile_{tag}.prof"))
        for key in ("cumulative", "tottime"):
            s = io.StringIO()
            pstats.Stats(pr, stream=s).sort_stats(key).print_stats(30)
            open(os.path.join(a.out, f"cprofile_{tag}_{key}.txt"), "w").write(s.getvalue())
        st = pstats.Stats(pr).stats
        tot = sum(v[2] for v in st.values())
        fn = {}
        for (f, line, name), (cc, nc, tt, ct, callers) in st.items():
            k = f"{os.path.basename(f)}:{name}"
            o = fn.get(k, (0, 0.0, 0.0))
            fn[k] = (o[0] + nc, o[1] + tt, max(o[2], ct))
        top = sorted(fn.items(), key=lambda x: -x[1][1])[:20]
        ops.pop("_cq", None)
        emit(kind="cprofile", engine="py", config=cfg, cpu_s_profiled=round(c1 - c0, 3), load_1m_before=l0, tottime_sum=round(tot, 3),
             digest=digest(views), ops=ops, **counters(e),
             top20_tottime=[dict(fn=k, ncalls=v[0], tottime=round(v[1], 4), tot_share=round(v[1] / tot, 4), cumtime=round(v[2], 4)) for k, v in top])
OUT.close()
