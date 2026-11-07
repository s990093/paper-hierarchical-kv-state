"""icpe-rux v9（v8_dev14＋二階排隊 qev=1；PREREG 偏離 19）的驗證。下面的「v8」都是 v9 的基底 v8_dev14。比的是 x1fast 的原始輸出檔（逐請求陣列、I/O 線狀態、引擎計數）。
用法（rux_run.sh 的 run 目錄裡）：python -I smoke_v9.py --trace nb:semianalysis_N16_s1 --rule nb --out <dir> [--workers 12]
測試（每個印一行 PASS／FAIL，最後印 V9_RESULT）：
  T1 qev_spread_eq_noq：到達改成 r × 10^6 秒（不會排隊、不會重排）→ v9 qev 和 v8 沒排隊逐位元相同；等待全 0；服務順序＝0..n−1。
  T2 v9_noqev_eq_v8：真的到達、沒有 qev → v9 和 v8 逐位元相同（v9 只加了 qev 的路徑）。
  T3 qev_real_exact：真的到達、qev → Python 用 v9 輸出的服務時間照同樣的運算重算 closed-loop FCFS：到達、開始、等待、服務順序逐位元相同；
     TTFT＝服務＋等待；開始 ≥ 到達；工作守恆。
  T4 qev_audit_eq：X1FAST_QEV_AUDIT（每 k 個請求暴力重算「下次回來」與 front）不出錯，而且輸出和沒稽核逐位元相同。
  T5 qev_determinism：同一個 job 跑兩次逐位元相同。
另外每個 T3 的 job 報告（不判定）：平均等待、重排比例；和一階 qpost（v8）比 mean TTFT 的差。"""
import argparse, hashlib, json, os, shutil, socket, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

ap = argparse.ArgumentParser()
ap.add_argument("--trace", required=True)
ap.add_argument("--rule", default="nb")
ap.add_argument("--cpus", nargs="+", type=float, default=[16.0, 111.0625])
ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--audit-every", type=int, default=500)
ap.add_argument("--tests", nargs="+", default=["T1", "T2", "T3", "T4", "T5", "T6"])
ap.add_argument("--pols", nargs="+", default=None)
ap.add_argument("--gps", nargs="+", default=None)
ap.add_argument("--flags", nargs="*", default=[])     # 每個 job 都加的 jobs.txt 欄位（例：pfx=1，偏離 14 (i)）
a = ap.parse_args()
REPO = os.environ["REPO"]
for p in ("code", "code/icpe", "code/p3x"):
    sys.path.insert(0, os.path.join(REPO, p))
sys.path.insert(0, os.path.join(REPO, "code/icpe/rux_x1fast/v6_slowcap"))
import numpy as np
import x1_rust_v6 as XR
sys.modules["x1_rust"] = XR
import p3_fvar
os.makedirs(a.out, exist_ok=True)
XS = p3_fvar.patch_x1("V1", os.path.join(REPO, "results/icpe_h200/p01/calib"), a.out, {})
K = XS.constants()
XS._K = K
P = XS.prep(a.trace, a.rule)
TDIR, _ = XR.prepare_trace(P, a.trace, a.rule)
BIN = {}
for tag, d in (("v8", "v8_dev14"), ("v9", "v9_qev")):     # "v8"＝v9 的基底 v8_dev14（旗標沒開時和 v8 48e4efce 逐位元相同）
    XR.SRC, XR._BIN = os.path.join(REPO, "code/icpe/rux_x1fast", d), None
    BIN[tag] = XR.binary()
    print("binary", tag, BIN[tag], XR.src_sha()[:16], flush=True)
XR._BIN = None
V6 = [(s, g, "cake") for s in ("ssd", "b32", "nfs") for g in ("idle", "busy")]
GiB = XS.GiB
NREQ = len(P["full"])
ARR = np.fromfile(os.path.join(TDIR, "t.f64"), np.float64)
SESS = np.fromfile(os.path.join(TDIR, "sess.i64"), np.int64)
fh = open(os.path.join(a.out, "smoke_v9.jsonl"), "a")
FAIL = 0
P17 = ["LRU", "LRU2", "ARC", "S3FIFO", "GDSF", "PENS", "FREQ2Q", "LRU+NONE", "LRU+2HW", "LRU+2HA", "LRU+RPF",
       "O-BEL", "O-BELS", "O-DROP", "O-NOW", "O-PF", "O-ALL"]
GP = ["LRU", "BEL", "TTLA:10", "TTLB:3", "APC", "BELP"]


def job(policy, cpu, vk, gp, extras=()):
    views = XS.make_views(K, P, vk)
    go = gp.lower() if gp in ("APC", "BELP") else None
    gkw = dict(gpu_cap=int(111.0625 * GiB), gpu_policy="LRU" if go else gp, gpu_write="wb", pf_target="cpu")
    jd = XR.write_job(XS, P, policy, cpu * GiB, views, K, gkw)
    x = open(os.path.join(jd, "jobs.txt")).read().split()
    if go is not None:
        x[8] = go
    x += list(extras) + list(a.flags)
    open(os.path.join(jd, "jobs.txt"), "w").write(" ".join(x) + "\n")
    return jd


def run(tag, jd, nv, tdir=None, env_extra=None, outname="out"):
    """跑一次 → (digest, 每個檔的 bytes)。輸出放 jd/<outname>（同一個 job 目錄可以跑多次）。"""
    src = os.path.join(jd, "jobs.txt")
    jd2 = jd + "_" + outname
    os.makedirs(jd2, exist_ok=True)
    for f in os.listdir(jd):
        if not os.path.exists(os.path.join(jd2, f)):
            os.symlink(os.path.join(jd, f), os.path.join(jd2, f))
    shutil.rmtree(os.path.join(jd2, "out"), ignore_errors=True)
    env = dict(os.environ, X1FAST_THREADS="1", X1FAST_FORK="0", X1FAST_SUMMARY_ONLY="0", **(env_extra or {}))
    env.pop("X1FAST_SERVE", None)
    cp = subprocess.run([BIN[tag], tdir or TDIR, jd2], env=env, capture_output=True, text=True)
    if cp.returncode != 0:
        return "CRASH", {"stderr": cp.stderr[-3000:]}
    od = os.path.join(jd2, "out", "j0")
    names = [f"v{k}_{n}" for k in range(nv) for n in ("ttft.f64", "rest.f64", "nc.i32", "ns.i32", "nr.i32", "ng.i32", "chan.f64", "chanb.i64")]
    names += ["eng.i64", "eng.f64"] + (["eng_v8.i64"] if os.path.exists(os.path.join(od, "eng_v8.i64")) else [])
    h = hashlib.sha256()
    raw = {}
    for n in names:
        b = open(os.path.join(od, n), "rb").read()
        raw[n] = b
        h.update(n.encode() + b"\0" + b)
    for n in ("v0_qe_arr.f64", "v0_qe_start.f64", "v0_qe_svc.f64", "v0_qe_wait.f64", "v0_qe_order.i64", "qe_audit.i64", "v0_qpw.f64", "v0_qps.f64"):
        p = os.path.join(od, n)
        if os.path.exists(p):
            raw[n] = open(p, "rb").read()
    return h.hexdigest()[:16], raw


def cleanup(*jds):
    import glob
    for jd in jds:
        shutil.rmtree(jd, ignore_errors=True)
        for x in glob.glob(jd + "_*"):
            shutil.rmtree(x, ignore_errors=True)


def report(name, ok, **kw):
    global FAIL
    FAIL += 0 if ok else 1
    r = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"),
             host=socket.gethostname(), trace=a.trace, test=name, ok=bool(ok), **kw)
    fh.write(json.dumps(r) + "\n"); fh.flush()
    print(("PASS " if ok else "FAIL ") + name + " " + json.dumps(kw), flush=True)


def f64(raw, n):
    return np.frombuffer(raw[n], np.float64)


PREV = np.full(NREQ, -1, np.int64)          # 同一個 session 的上一個請求（沒有＝−1）
_lastp = {}
for _r in range(NREQ):
    _s = int(SESS[_r])
    PREV[_r] = _lastp.get(_s, -1)
    _lastp[_s] = _r


def early(arr, st, svc, w, tag):
    """（只報告；審查第 1 點）到達早於同 session 上一個請求完成的比例，以及這段「還送不出來的時間」占總等待的比例。"""
    m = PREV >= 0
    p = PREV[m]
    gap = (st[p] + svc[p]) - arr[m]
    ex = np.minimum(np.maximum(gap, 0.0), w[m])
    return {f"frac_early_{tag}": float((gap > 0).mean()), f"early_share_of_wait_{tag}": float(ex.sum() / w.sum()) if w.sum() > 0 else 0.0}


def closed_fcfs(t, sess, svc, maxrule=True):
    """Python 版（照 Rust 的運算順序）：closed-loop FCFS；到達＝紀錄時刻＋這個 session 上一個請求開始得比紀錄晚多少，
    maxrule＝True 時再取 max(…, 上一個完成)（v9 qev、qpost=2）；同時到照序號。"""
    import heapq
    n = len(t)
    nxt = [-1] * n
    first, prev = {}, {}
    for r in range(n):
        s = int(sess[r])
        if s not in first:
            first[s] = r
        else:
            nxt[prev[s]] = r
        prev[s] = r
    H = [(float(t[r]), r) for r in first.values()]
    heapq.heapify(H)
    free = -np.inf
    A, W, S, O = np.zeros(n), np.zeros(n), np.zeros(n), []
    while H:
        x, r = heapq.heappop(H)
        st = free if free > x else x
        A[r], W[r], S[r] = x, st - x, st
        O.append(r)
        free = st + svc[r]
        late = st - t[r]
        if nxt[r] >= 0:
            an = float(t[nxt[r]] + late)
            if maxrule and free > an:
                an = free
            heapq.heappush(H, (an, nxt[r]))
    return A, W, S, np.array(O, np.int64)


SPREAD = os.path.join(a.out, "trace_spread")
if not os.path.isdir(SPREAD):
    os.makedirs(SPREAD)
    for f in os.listdir(TDIR):
        if f != "t.f64":
            os.symlink(os.path.join(TDIR, f), os.path.join(SPREAD, f))
    (np.arange(NREQ, dtype=np.float64) * 1e6).tofile(os.path.join(SPREAD, "t.f64"))
CASES = [(pol, gp, cpu) for cpu in a.cpus for pol in (a.pols or P17) for gp in (a.gps or GP)]
VQ = {"light": ("ssd", "idle", "cake"), "heavy": ("nfs", "busy", "cake")}


def tdep(pol, gp):
    return XS.POLICIES[pol][3] is not None or gp.startswith("TTL") or XS.POLICIES[pol][0] == "gdsf"   # GDSF：時間設定要同慢層、同 GPU（check_job）


def t1(c):
    pol, gp, cpu = c
    tag = f"cpu={cpu} {pol} {gp}"
    jd0, jdq = job(pol, cpu, [VQ["heavy"]], gp), job(pol, cpu, [VQ["heavy"]], gp, ["qev=1"])
    r0, rq = run("v8", jd0, 1, tdir=SPREAD), run("v9", jdq, 1, tdir=SPREAD)
    ok = r0[0] == rq[0] and r0[0] != "CRASH"
    kw = dict(a=r0[0], b=rq[0])
    if rq[0] != "CRASH":
        w, o = f64(rq[1], "v0_qe_wait.f64"), np.frombuffer(rq[1]["v0_qe_order.i64"], np.int64)
        ok &= bool((w == 0).all() and np.array_equal(o, np.arange(NREQ)))
    else:
        kw["stderr"] = rq[1]["stderr"][-400:]
    cleanup(jd0, jdq)
    return f"qev_spread_eq_noq {tag}", ok, kw


def t2(c):
    pol, gp, cpu = c
    vk = [VQ["light"]] if tdep(pol, gp) else V6
    tag = f"cpu={cpu} {pol} {gp} views={len(vk)}"
    jd = job(pol, cpu, vk, gp)
    r8, r9 = run("v8", jd, len(vk), outname="o8"), run("v9", jd, len(vk), outname="o9")
    cleanup(jd)
    return f"v9_noqev_eq_v8 {tag}", r8[0] == r9[0] and r8[0] != "CRASH", dict(a=r8[0], b=r9[0])


def t345(c, load):
    pol, gp, cpu = c
    tag = f"cpu={cpu} {pol} {gp} {load}"
    jdq = job(pol, cpu, [VQ[load]], gp, ["qev=1"])
    jdp = job(pol, cpu, [VQ[load]], gp, ["qpost=1"])
    jdm = job(pol, cpu, [VQ[load]], gp, ["qpost=2"])
    out = []
    rq = run("v9", jdq, 1, outname="q")
    if rq[0] == "CRASH":
        out.append((f"qev_real_exact {tag}", False, dict(stderr=rq[1]["stderr"][-600:])))
    else:
        svc, st, w, ar = (f64(rq[1], f"v0_qe_{n}.f64") for n in ("svc", "start", "wait", "arr"))
        o = np.frombuffer(rq[1]["v0_qe_order.i64"], np.int64)
        tt = f64(rq[1], "v0_ttft.f64")
        A, W, S, O = closed_fcfs(ARR, SESS, svc)
        ok = bool(np.array_equal(A, ar) and np.array_equal(W, w) and np.array_equal(S, st) and np.array_equal(O, o)
                  and np.array_equal(svc + W, tt) and (st >= ar).all())
        so = st[o]
        comp = so + svc[o]
        ok &= bool(np.array_equal(so[1:], np.maximum(ar[o][1:], comp[:-1])))      # 工作守恆：開始＝max(到達, 前一個完成)
        rq8 = run("v8", jdp, 1, outname="p")
        info = dict(mean_wait=float(w.mean()), frac_wait=float((w > 0).mean()), frac_out_of_index_order=float(np.mean(o[1:] < np.maximum.accumulate(o)[:-1])),
                    mean_ttft_qev=float(tt.mean()))
        info.update(early(ar, st, svc, w, "qev"))
        ok &= info["frac_early_qev"] == 0.0                                       # 二階：沒有請求早於上一個完成
        if rq8[0] != "CRASH":
            info.update(mean_ttft_qpost=float(f64(rq8[1], "v0_ttft.f64").mean()), mean_wait_qpost=float(f64(rq8[1], "v0_qpw.f64").mean()))
            wp, sp_ = f64(rq8[1], "v0_qpw.f64"), f64(rq8[1], "v0_qps.f64")
            info.update(early(sp_ - wp, sp_, f64(rq8[1], "v0_ttft.f64") - wp, wp, "qpost"))
        rqm = run("v9", jdm, 1, outname="m")
        if rqm[0] != "CRASH":
            wm = f64(rqm[1], "v0_qpw.f64")
            info.update(mean_ttft_qpost2=float(f64(rqm[1], "v0_ttft.f64").mean()), mean_wait_qpost2=float(wm.mean()))
        out.append((f"qev_real_exact {tag}", ok, info))
        if "T4" in a.tests:
            ra = run("v9", jdq, 1, env_extra={"X1FAST_QEV_AUDIT": str(a.audit_every)}, outname="a")
            na = np.frombuffer(ra[1]["qe_audit.i64"], np.int64).tolist() if ra[0] != "CRASH" else None
            out.append((f"qev_audit_eq {tag}", ra[0] == rq[0] and bool(na and na[1] > 0),
                        dict(a=rq[0], b=ra[0], audits=na, stderr=ra[1].get("stderr", "")[-400:] if ra[0] == "CRASH" else "")))
        if "T5" in a.tests:
            r2 = run("v9", jdq, 1, outname="q2")
            out.append((f"qev_determinism {tag}", r2[0] == rq[0], dict(a=rq[0], b=r2[0])))
    cleanup(jdq, jdp, jdm)
    return out


def t6(c, load):
    """qpost=2（一階、到達不早於上一個完成）：(i) 到達拉開 → 和 v8 沒排隊逐位元相同、等待全 0；(ii) 真的到達 → Python 用沒排隊的 TTFT
    照同樣的運算重算等待、開始時刻，逐位元相同，其他輸出和沒排隊相同；(iii) v9 qpost=1 和 v8 qpost=1 逐位元相同（含等待、開始）。"""
    pol, gp, cpu = c
    tag = f"cpu={cpu} {pol} {gp} {load}"
    jd0 = job(pol, cpu, [VQ[load]], gp)
    jdm = job(pol, cpu, [VQ[load]], gp, ["qpost=2"])
    jdp = job(pol, cpu, [VQ[load]], gp, ["qpost=1"])
    out = []
    r0s, rms = run("v8", jd0, 1, tdir=SPREAD, outname="s0"), run("v9", jdm, 1, tdir=SPREAD, outname="sm")
    ok = r0s[0] == rms[0] and r0s[0] != "CRASH" and bool((f64(rms[1], "v0_qpw.f64") == 0).all())
    out.append((f"qpost2_spread_eq_noq {tag}", ok, dict(a=r0s[0], b=rms[0])))
    r0, rm = run("v8", jd0, 1, outname="r0"), run("v9", jdm, 1, outname="rm")
    if "CRASH" in (r0[0], rm[0]):
        out.append((f"qpost2_real_exact {tag}", False, dict(stderr=(rm if rm[0] == "CRASH" else r0)[1]["stderr"][-500:])))
    else:
        svc = f64(r0[1], "v0_ttft.f64")
        A, W, S, O = closed_fcfs(ARR, SESS, svc, maxrule=True)
        w, st, tt = f64(rm[1], "v0_qpw.f64"), f64(rm[1], "v0_qps.f64"), f64(rm[1], "v0_ttft.f64")
        ok = bool(np.array_equal(W, w) and np.array_equal(S, st) and np.array_equal(svc + W, tt))
        for nm in ("v0_rest.f64", "v0_nc.i32", "v0_ns.i32", "v0_nr.i32", "v0_ng.i32", "v0_chan.f64", "v0_chanb.i64", "eng.i64", "eng.f64"):
            ok &= r0[1][nm] == rm[1][nm]
        _, W1, _, _ = closed_fcfs(ARR, SESS, svc, maxrule=False)
        out.append((f"qpost2_real_exact {tag}", ok, dict(mean_wait_qpost2=float(W.mean()), mean_wait_qpost1=float(W1.mean()),
                                                         **early(A, S, svc, W, "qpost2"))))
    r8, r9 = run("v8", jdp, 1, outname="p8"), run("v9", jdp, 1, outname="p9")
    ok = r8[0] == r9[0] and r8[0] != "CRASH" and r8[1].get("v0_qpw.f64") == r9[1].get("v0_qpw.f64") and r8[1].get("v0_qps.f64") == r9[1].get("v0_qps.f64")
    out.append((f"qpost1_v9_eq_v8 {tag}", ok, dict(a=r8[0], b=r9[0])))
    cleanup(jd0, jdm, jdp)
    return out


with ThreadPoolExecutor(a.workers) as ex:
    if "T1" in a.tests:
        for r in ex.map(t1, CASES):
            report(*r[:2], **r[2])
    if "T2" in a.tests:
        for r in ex.map(t2, CASES):
            report(*r[:2], **r[2])
    if "T3" in a.tests:
        for rs in ex.map(lambda c: t345(c[0], c[1]), [(c, ld) for c in CASES for ld in ("light", "heavy")]):
            for r in rs:
                report(*r[:2], **r[2])
    if "T6" in a.tests:
        for rs in ex.map(lambda c: t6(c[0], c[1]), [(c, ld) for c in CASES for ld in ("light", "heavy")]):
            for r in rs:
                report(*r[:2], **r[2])
print(f"V9_RESULT {'PASS' if FAIL == 0 else 'FAIL'} fail={FAIL} trace={a.trace} cases={len(CASES)} flags={' '.join(a.flags) or '-'}", flush=True)
sys.exit(1 if FAIL else 0)
