"""icpe-rux v6 的速度：同一個 job，Python Engine 對 Rust v6（批次模式、單執行緒），交錯重複；每個設定也比逐請求指紋。
job＝CPU LRU × GPU-LRU wb 111.0625 GiB、CPU 111.0625 GiB、Cake（同第 1 段的 prof_sc.py）。
設定：nocap6／nocap1（沒有上限）、sc:<規則>（frac 0.5；GDSF 1 個時間設定，其他 6 個）。
用法：python -I speed_v6.py --trace <name> --py-reps 1 --rs-reps 3 --out <dir>
每列有 run_id、ts、host、threads＝1、load；Python＝process_time，Rust＝x1fast 自己量的引擎時間（不含讀紀錄）。"""
import argparse, hashlib, json, os, shutil, socket, subprocess, sys, time
from datetime import datetime

ap = argparse.ArgumentParser()
ap.add_argument("--trace", required=True)
ap.add_argument("--rule", default="R1")
ap.add_argument("--configs", nargs="+", default=["nocap6", "nocap1", "sc:LRU", "sc:LRU2", "sc:ARC", "sc:S3FIFO", "sc:GDSF", "sc:PENS", "sc:BEL"])
ap.add_argument("--py-reps", type=int, default=1)
ap.add_argument("--rs-reps", type=int, default=3)
ap.add_argument("--out", required=True)
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ["REPO"]
for p in ("code", "code/icpe", "code/p3x"):
    sys.path.insert(0, os.path.join(REPO, p))
sys.path.insert(0, HERE)
import numpy as np
import x1_rust_v6 as XR
sys.modules["x1_rust"] = XR
import p3_fvar
os.makedirs(a.out, exist_ok=True)
XS = p3_fvar.patch_x1("V1", os.path.join(REPO, "results/icpe_h200/p01/calib"), a.out, {})
K = XS.constants()
XS._K = K
P = XS.prep(a.trace, a.rule)
tdir, _ = XR.prepare_trace(P, a.trace, a.rule)
V6 = [(s, g, "cake") for s in ("ssd", "b32", "nfs") for g in ("idle", "busy")]
GKW = dict(gpu_cap=int(111.0625 * XS.GiB), gpu_policy="LRU", gpu_write="wb", pf_target="cpu")
CPU = 111.0625 * XS.GiB
fh = open(os.path.join(a.out, "speed.jsonl"), "a")


def la():
    return float(open("/proc/loadavg").read().split()[0])


def spec(cfg):
    if cfg == "nocap6":
        return V6, None
    if cfg == "nocap1":
        return V6[:1], None
    pol = cfg.split(":")[1]
    return (V6[:1] if pol == "GDSF" else V6), dict(frac=0.5, pol=pol)


def dig(views):
    h = hashlib.sha256()
    for v in views:
        for x in (v.ttft, v.rest, v.nC, v.nS, v.nR):
            h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()[:16]


def emit(**r):
    r = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"),
             host=socket.gethostname(), trace=a.trace, threads=1, **r)
    fh.write(json.dumps(r) + "\n")
    fh.flush()
    print(json.dumps(r), flush=True)


def run_py(cfg):
    vk, sc = spec(cfg)
    views = XS.make_views(K, P, vk)
    kw = dict(GKW)
    if sc:
        kw.update(slow_cap=sc["frac"], slow_policy=sc["pol"])
    gv = views[0] if sc and sc["pol"] == "GDSF" else None
    l0 = la(); c0 = time.process_time()
    e = XS.Engine(P, "LRU", CPU, views, K, gdsf_view=gv, **kw).run()
    c1 = time.process_time()
    return c1 - c0, l0, dig(views), [getattr(e, "n_sc_clean", 0), getattr(e, "n_sc_drop", 0)], len(vk)


def run_rs(cfg):
    vk, sc = spec(cfg)
    views = XS.make_views(K, P, vk)
    jd = XR.write_job(XS, P, "LRU", CPU, views, K, GKW, sc=sc)
    env = dict(os.environ, X1FAST_THREADS="1", X1FAST_FORK="0", X1FAST_SUMMARY_ONLY="0")
    env.pop("X1FAST_SERVE", None)
    env.pop("X1FAST_SC_CHECK", None)
    l0 = la()
    cp = subprocess.run([XR.binary(), tdir, jd], env=env, capture_output=True, text=True)
    assert cp.returncode == 0, cp.stderr[-3000:]
    run_s = float([x for x in open(os.path.join(jd, "timing.txt")).read().split("\n") if x.startswith("j0 ")][0].split()[1])
    e = XR.read_outputs(jd, P, views, GKW, "LRU")
    shutil.rmtree(jd, ignore_errors=True)
    return run_s, l0, dig(views), [getattr(e, "n_sc_clean", 0), getattr(e, "n_sc_drop", 0)], len(vk)


for rep in range(max(a.py_reps, a.rs_reps)):
    for cfg in a.configs:
        if rep < a.rs_reps:
            t, l0, d, c, nv = run_rs(cfg)
            emit(kind="time", engine="rust_v6", config=cfg, rep=rep, n_views=nv, cpu_s=round(t, 5), load_1m=l0, digest=d, sc_counts=c,
                 bin=XR.binary())
        if rep < a.py_reps:
            t, l0, d, c, nv = run_py(cfg)
            emit(kind="time", engine="py", config=cfg, rep=rep, n_views=nv, cpu_s=round(t, 4), load_1m=l0, digest=d, sc_counts=c)
