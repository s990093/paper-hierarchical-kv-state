"""icpe-rux v6 的冒煙測試：同一個 job，Python Engine 對 Rust v6（x1_rust_v6.run，批次模式），比逐請求陣列的 sha256、
I/O 線狀態與引擎計數（含慢層的 3 個）。不是正式對拍（正式的在 cmp_v6.py）；只用來在大跑之前抓明顯的錯。
用法：python -I smoke_v6.py --trace conversation --policies LRU O-BEL --gpu-policies LRU BEL --slow LRU ... --out <dir>"""
import argparse, hashlib, json, os, socket, sys, time
from datetime import datetime

ap = argparse.ArgumentParser()
ap.add_argument("--trace", required=True)
ap.add_argument("--rule", default="R1")
ap.add_argument("--cpus", nargs="+", type=float, default=[111.0625])
ap.add_argument("--policies", nargs="+", default=["LRU"])
ap.add_argument("--gpu-policies", nargs="+", default=["LRU"])
ap.add_argument("--gpu-write", default="wb")
ap.add_argument("--pf-target", default="cpu")
ap.add_argument("--slow", nargs="+", default=["LRU", "LRU2", "ARC", "S3FIFO", "GDSF", "PENS", "BEL"])
ap.add_argument("--views", default="all6", choices=["all6", "one"])
ap.add_argument("--noise", default=None, help="P|M,lam,B1|B2,seed")
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
XR.prepare_trace(P, a.trace, a.rule)
V6 = [(s, g, "cake") for s in ("ssd", "b32", "nfs") for g in ("idle", "busy")]
nz = None
if a.noise:
    m, lam, dr, sd = a.noise.split(",")
    nz = dict(model=m, lam=float(lam), draw=dr, seed=int(sd))
fh = open(os.path.join(a.out, "smoke.jsonl"), "a")


def dig(views, eng):
    h = hashlib.sha256()
    for v in views:
        for x in (v.ttft, v.rest, v.nC, v.nS, v.nR):
            h.update(np.ascontiguousarray(x).tobytes())
        h.update(np.array([v.cpu.t_free, v.cpu.t_free_w, v.cpu.busy_r, v.cpu.busy_w, v.slow.t_free, v.slow.t_free_w,
                           v.slow.busy_r, v.slow.busy_w], np.float64).tobytes())
        h.update(np.array([v.cpu.bytes_r, v.cpu.bytes_w, v.slow.bytes_r, v.slow.bytes_w, v.pf_bytes], np.int64).tobytes())
        if hasattr(v, "nG"):
            h.update(np.ascontiguousarray(np.asarray(v.nG, np.int32)).tobytes())
    cnt = [eng.n_demote, eng.n_drop, eng.n_unpf, getattr(eng, "n_gevict", 0), getattr(eng, "n_gexpire", 0),
           getattr(eng, "g_wb_blocks", 0), eng.w_slow_blocks, getattr(eng, "sc_cap", -1), getattr(eng, "n_sc_clean", 0),
           getattr(eng, "n_sc_drop", 0)]
    h.update(np.array(cnt, np.int64).tobytes())
    h.update(np.array([eng.res_sum], np.float64).tobytes())
    return h.hexdigest()[:16], cnt


bad = 0
for cpu in a.cpus:
    for pol in a.policies:
        for gp in a.gpu_policies:
            tdep = XS.POLICIES[pol][3] or XS.parse_gpol(gp)[1] is not None
            for sp in a.slow:
                one = a.views == "one" or tdep or sp == "GDSF" or XS.POLICIES[pol][0] == "gdsf"
                vk = [("ssd", "idle", "cake")] if one else V6
                gkw = dict(gpu_cap=int(111.0625 * XS.GiB), gpu_policy=gp, gpu_write=a.gpu_write, pf_target=a.pf_target)
                sc = dict(frac=0.5, pol=sp)
                gv = lambda vs: vs[0] if (XS.POLICIES[pol][0] == "gdsf" or sp == "GDSF") else None
                vp = XS.make_views(K, P, vk)
                kw = dict(gkw, slow_cap=0.5, slow_policy=sp)
                if nz:
                    kw["noise"] = nz
                t0 = time.process_time()
                ep = XS.Engine(P, pol, cpu * XS.GiB, vp, K, gdsf_view=gv(vp), **kw).run()
                tp = time.process_time() - t0
                dp, cp = dig(vp, ep)
                vr = XS.make_views(K, P, vk)
                jd = XR.write_job(XS, P, pol, cpu * XS.GiB, vr, K, gkw, nz=nz, sc=sc)
                import subprocess, shutil
                env = dict(os.environ, X1FAST_THREADS="1", X1FAST_FORK="0", X1FAST_SUMMARY_ONLY="0", X1FAST_SC_CHECK="500")
                env.pop("X1FAST_SERVE", None)
                cpr = subprocess.run([XR.binary(), XR._TDIR[(a.trace, a.rule)], jd], env=env, capture_output=True, text=True)
                if cpr.returncode != 0:
                    print("RUST FAIL", pol, gp, sp, cpr.stderr[-2000:], flush=True)
                    sys.exit(2)
                tr_ = float([x for x in open(os.path.join(jd, "timing.txt")).read().split("\n") if x.startswith("j0 ")][0].split()[1])
                er = XR.read_outputs(jd, P, vr, gkw, pol)
                shutil.rmtree(jd, ignore_errors=True)
                dr, cr = dig(vr, er)
                ok = dp == dr
                bad += not ok
                r = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"),
                         host=socket.gethostname(), trace=a.trace, cpu=cpu, policy=pol, gpu_policy=gp, gpu_write=a.gpu_write,
                         pf_target=a.pf_target, slow=sp, noise=a.noise, n_views=len(vk), same=ok, py_digest=dp, rust_digest=dr,
                         py_counts=cp, rust_counts=cr, py_s=round(tp, 4), rust_s=round(tr_, 5))
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                print(("SAME " if ok else "DIFF ") + f"cpu={cpu} {pol} gpu={gp} slow={sp} views={len(vk)} py={tp:.3f}s rust={tr_:.4f}s "
                      f"sc(clean,drop)={cp[8:10]} vs {cr[8:10]}" + ("" if ok else f" py_counts={cp} rust_counts={cr}"), flush=True)
print(f"SMOKE_RESULT bad={bad}", flush=True)
sys.exit(1 if bad else 0)
