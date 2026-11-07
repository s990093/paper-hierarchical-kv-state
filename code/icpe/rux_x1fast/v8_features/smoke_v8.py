"""icpe-rux v8 的驗證（沒有 Python 參考的功能）：退化檢查（新功能設成等於沒開 → 和 v7 逐位元相同）、精確重算、不變量。
用法：python -I smoke_v8.py --trace conversation --rule R1 --tests queue cue --out <dir>
每個測試印一行 PASS／FAIL，最後印 V8_RESULT。比的是 x1fast 的原始輸出檔（每個時間設定的逐請求陣列、I/O 線狀態、引擎計數）。"""
import argparse, hashlib, json, os, shutil, socket, subprocess, sys
from datetime import datetime

ap = argparse.ArgumentParser()
ap.add_argument("--trace", required=True)
ap.add_argument("--rule", default="R1")
ap.add_argument("--tests", nargs="+", default=["queue", "cue"])
ap.add_argument("--cpus", nargs="+", type=float, default=[8.0, 111.0625])
ap.add_argument("--out", required=True)
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
for tag, d in (("v7", "v7_fast"), ("v8", "v8_features")):
    XR.SRC, XR._BIN = os.path.join(REPO, "code/icpe/rux_x1fast", d), None
    BIN[tag] = XR.binary()
XR._BIN = None
V6 = [(s, g, "cake") for s in ("ssd", "b32", "nfs") for g in ("idle", "busy")]
ONE = [("ssd", "idle", "cake")]
GiB = XS.GiB
NREQ = len(P["full"])
fh = open(os.path.join(a.out, "smoke_v8.jsonl"), "a")
FAIL = 0
SESS = None


def job(policy, cpu, vk, gw="wb", gpol="LRU", pft="cpu", go=None, gttl=None, extras=(), files=None, sc=None, gcap=111.0625, nz=None):
    views = XS.make_views(K, P, vk)
    gkw = dict(gpu_cap=int(gcap * GiB), gpu_policy=gpol, gpu_write=gw, pf_target=pft)
    jd = XR.write_job(XS, P, policy, cpu * GiB, views, K, gkw, sc=sc, nz=nz)
    x = open(os.path.join(jd, "jobs.txt")).read().split()
    if go is not None:
        x[8] = go
    if gttl is not None:
        x[9] = gttl
    x += list(extras)
    open(os.path.join(jd, "jobs.txt"), "w").write(" ".join(x) + "\n")
    for nm, arr in (files or {}).items():
        arr.tofile(os.path.join(jd, nm))
    return jd, len(vk)


def run(tag, jd, nv, tdir=None, env_extra=None):
    """跑一次 → (digest, 每個檔的 bytes)。jd 跑完不刪（呼叫端可以再讀）；out 目錄每次清掉重跑。"""
    shutil.rmtree(os.path.join(jd, "out"), ignore_errors=True)
    env = dict(os.environ, X1FAST_THREADS="1", X1FAST_FORK="0", X1FAST_SUMMARY_ONLY="0", **(env_extra or {}))
    env.pop("X1FAST_SERVE", None)
    cp = subprocess.run([BIN[tag], tdir or TDIR, jd], env=env, capture_output=True, text=True)
    if cp.returncode != 0:
        return "CRASH", {"stderr": cp.stderr[-3000:]}
    od = os.path.join(jd, "out", "j0")
    names = [f"v{k}_{n}" for k in range(nv) for n in ("ttft.f64", "rest.f64", "nc.i32", "ns.i32", "nr.i32", "ng.i32", "chan.f64", "chanb.i64")]
    names += ["eng.i64", "eng.f64"] + (["eng_sc.i64"] if os.path.exists(os.path.join(od, "eng_sc.i64")) else [])
    h = hashlib.sha256()
    raw = {}
    for n in names:
        b = open(os.path.join(od, n), "rb").read()
        raw[n] = b
        h.update(n.encode() + b"\0" + b)
    for n in ("qwait.f64", "qstart.f64", "qsvc.f64", "qarr.f64"):
        p = os.path.join(od, n)
        if os.path.exists(p):
            raw[n] = open(p, "rb").read()
    return h.hexdigest()[:16], raw


def report(name, ok, **kw):
    global FAIL
    FAIL += 0 if ok else 1
    r = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"),
             host=socket.gethostname(), trace=a.trace, test=name, ok=bool(ok), **kw)
    fh.write(json.dumps(r) + "\n"); fh.flush()
    print(("PASS " if ok else "FAIL ") + name + " " + json.dumps(kw), flush=True)


def same(name, ra, rb, **kw):
    report(name, ra[0] == rb[0] and ra[0] != "CRASH", a=ra[0], b=rb[0], **kw)


def eng(raw):
    return np.frombuffer(raw["eng.i64"], np.int64).tolist()


def sess_of():
    global SESS
    if SESS is None:
        SESS = np.fromfile(os.path.join(TDIR, "sess.i64"), np.int64)
    return SESS


def cue_rand(seed, p=0.5):
    """合成的線索（只用來測引擎，不是資料）：每個請求獨立 Bernoulli(p)；session 的最後一個請求＝0。"""
    rng = np.random.default_rng(seed)
    c = (rng.random(NREQ) < p).astype(np.uint8)
    s = sess_of()
    last = {}
    for r in range(NREQ):
        last[int(s[r])] = r
    for r in last.values():
        c[r] = 0
    return c


# ------------------------------------------------------------------ W3 線索
def t_cue():
    z = np.zeros(NREQ, np.uint8)
    o = np.ones(NREQ, np.uint8)
    for cpu in a.cpus:
        for pol in ("LRU", "O-BEL", "LRU+RPF", "GDSF"):
            for gw in ("wb", "wt"):
                tdep = pol not in ("LRU", "O-BEL")
                vk = ONE if tdep else V6
                # 1. cue 全 0：CUEP、CUEO 都＝GPU LRU（v7）
                jd0, nv = job(pol, cpu, vk, gw=gw)
                r7 = run("v7", jd0, nv)
                for go in ("cuep", "cueo"):
                    jd, nv = job(pol, cpu, vk, gw=gw, go=go, extras=["cue=1"], files={"cue.u8": z})
                    same(f"cue0_{go}_eq_v7_lru cpu={cpu} {pol} {gw}", run("v8", jd, nv), r7)
                    shutil.rmtree(jd, ignore_errors=True)
                shutil.rmtree(jd0, ignore_errors=True)
                # 2. cue 全 1：CUEP＝TTLB 無限大（1 個時間設定）
                jdt, _ = job(pol, cpu, ONE, gw=gw, gpol="TTLB:10", gttl="inf")
                rt7 = run("v7", jdt, 1)
                jd, _ = job(pol, cpu, ONE, gw=gw, go="cuep", extras=["cue=1"], files={"cue.u8": o})
                same(f"cue1_cuep_eq_v7_ttlb_inf cpu={cpu} {pol} {gw}", run("v8", jd, 1), rt7)
                shutil.rmtree(jd, ignore_errors=True); shutil.rmtree(jdt, ignore_errors=True)
                # 3. 合成線索：CUEO 每次有工具呼叫都逐出 → GPU 逐出次數 ≥ 有線索、沒被丟的回合數；能跑完、和 LRU 不同
                c = cue_rand(1)
                jd, nv = job(pol, cpu, vk, gw=gw, go="cueo", extras=["cue=1"], files={"cue.u8": c})
                rr = run("v8", jd, nv)
                ok = rr[0] != "CRASH"
                if ok:
                    e = eng(rr[1])
                    ok = e[3] >= 0
                report(f"cue_rand_cueo_runs cpu={cpu} {pol} {gw}", ok, digest=rr[0], n_cue=int(c.sum()), eng=eng(rr[1]) if ok else None)
                shutil.rmtree(jd, ignore_errors=True)
                jd, nv = job(pol, cpu, vk, gw=gw, go="cuep", extras=["cue=1"], files={"cue.u8": c})
                rr = run("v8", jd, nv)
                report(f"cue_rand_cuep_runs cpu={cpu} {pol} {gw}", rr[0] != "CRASH", digest=rr[0])
                shutil.rmtree(jd, ignore_errors=True)


# ------------------------------------------------------------------ 排隊
def t_queue():
    # 1. 退化：把到達時刻改成 r × 10^6 秒（服務時間遠小於間隔 → 不會等）→ q=1 和沒有排隊逐位元相同，等待全 0
    td2 = os.path.join(a.out, "trace_spread")
    if not os.path.isdir(td2):
        os.makedirs(td2)
        for f in os.listdir(TDIR):
            if f != "t.f64":
                os.symlink(os.path.join(TDIR, f), os.path.join(td2, f))
        (np.arange(NREQ, dtype=np.float64) * 1e6).tofile(os.path.join(td2, "t.f64"))
    arr = np.fromfile(os.path.join(TDIR, "t.f64"), np.float64)
    for cpu in a.cpus:
        for pol, gp in (("LRU", "LRU"), ("O-BEL", "BEL"), ("LRU+RPF", "LRU"), ("GDSF", "TTLA:10"), ("O-PF", "TTLB:10")):
            for vk in ([V6[0]], [V6[5]]):
                jd0, _ = job(pol, cpu, vk, gpol=gp)
                jdq, _ = job(pol, cpu, vk, gpol=gp, extras=["q=1"])
                r0 = run("v8", jd0, 1, tdir=td2)
                rq = run("v8", jdq, 1, tdir=td2)
                same(f"queue_spread_eq_noqueue cpu={cpu} {pol} {gp} {vk[0]}", rq, r0)
                if rq[0] != "CRASH":
                    w = np.frombuffer(rq[1]["qwait.f64"], np.float64)
                    report(f"queue_spread_zero_wait cpu={cpu} {pol} {gp} {vk[0]}", bool((w == 0).all()), max_wait=float(w.max()))
                # 2. 原本的到達：精確重算（開始＝max(到達, 前一個開始＋服務)；TTFT＝服務＋等待），逐請求位元相同
                rq = run("v8", jdq, 1)
                if rq[0] == "CRASH":
                    report(f"queue_real cpu={cpu} {pol} {gp} {vk[0]}", False, crash=rq[1]["stderr"][-500:])
                else:
                    st = np.frombuffer(rq[1]["qstart.f64"], np.float64)
                    sv = np.frombuffer(rq[1]["qsvc.f64"], np.float64)
                    w = np.frombuffer(rq[1]["qwait.f64"], np.float64)
                    tt = np.frombuffer(rq[1]["v0_ttft.f64"], np.float64)
                    done = st + sv
                    exp = arr.copy()
                    exp[1:] = np.where(done[:-1] > arr[1:], done[:-1], arr[1:])
                    ok1 = np.array_equal(st, exp)
                    ok2 = np.array_equal(w, st - arr)
                    ok3 = np.array_equal(tt, sv + w)
                    rho = float(sv.sum() / (arr[-1] - arr[0])) if arr[-1] > arr[0] else float("nan")
                    report(f"queue_real_exact cpu={cpu} {pol} {gp} {vk[0]}", ok1 and ok2 and ok3, start_ok=ok1, wait_ok=ok2,
                           ttft_ok=ok3, rho=round(rho, 4), mean_wait=float(w.mean()), frac_wait=float((w > 0).mean()),
                           mean_svc=float(sv.mean()))
                shutil.rmtree(jd0, ignore_errors=True); shutil.rmtree(jdq, ignore_errors=True)
    # 3. 沒有 q 欄位時 v8＝v7（同一個 job，6 個時間設定）
    for cpu in a.cpus:
        jd, nv = job("LRU", cpu, V6, gpol="BEL")
        same(f"noq_v8_eq_v7 cpu={cpu}", run("v8", jd, nv), run("v7", jd, nv))
        shutil.rmtree(jd, ignore_errors=True)


# ------------------------------------------------------------------ W5
def w5_d1():
    """一次拆完的殘差 d1＝(T_meas − 64·i4 ÷ r_cpu) ÷ 64；T_meas＝P0-1b S4（fa3）32K、CPU、INT4 全部載入的還原中位數。"""
    import csv
    f = os.path.join(REPO, "results/icpe_h200/p01/grid/llama_s4_fa3/summary.csv")
    T = {}
    for r in csv.DictReader(open(f)):
        if r["length"] == "32768" and r["bw_name"] == "cpu" and r["config"] == "E1" and r["mode"] == "loadall":
            assert int(r["median_bytes_read"]) == 64 * K["int4_bytes"], r
            T[r["gpu_state"]] = float(r["median_t_restore"])
    return T


def w5_file(vk, mode, T=None):
    """w5.f64：每個時間設定 [dc1, dc2]。mode＝degen（dc1＝0、dc2＝d → 等於沒開）或 real（dc1＝dc2＝d1）。"""
    views = XS.make_views(K, P, vk)
    out = []
    for v, (_, g, _) in zip(views, vk):
        if mode == "degen":
            out += [0.0, float(v.d)]
        else:
            d1 = (T[g] - 64 * K["int4_bytes"] / v.cpu.r) / 64
            assert d1 > 0, (g, d1)
            out += [d1, d1]
    return np.array(out, np.float64)


def t_w5():
    T = w5_d1()
    BFB = 64 << 20
    i4 = K["int4_bytes"]
    for cpu in a.cpus:
        for pol in ("LRU", "LRU2", "ARC", "S3FIFO", "GDSF", "PENS", "O-BEL", "LRU+RPF", "O-PF", "LRU+2HA", "O-NOW"):
            for gp, gw in (("LRU", "wb"), ("BEL", "wb"), ("TTLA:10", "wb"), ("LRU", "wt")):
                tdep = pol not in ("LRU", "LRU2", "ARC", "S3FIFO", "O-BEL", "LRU+2HA", "O-NOW") or gp.startswith("TTL")
                vk = ONE if tdep else V6
                for sc in (None, dict(frac=0.5, pol="LRU")):
                    if sc and (gp != "LRU" or gw != "wb"):
                        continue
                    tag = f"cpu={cpu} {pol} {gp} {gw} sc={'on' if sc else 'off'}"
                    jd0, nv = job(pol, cpu, vk, gw=gw, gpol=gp, sc=sc)
                    r7 = run("v7", jd0, nv)
                    shutil.rmtree(jd0, ignore_errors=True)
                    # 1. 退化：w5＝BF、dc1＝0、dc2＝d → 和 v7 逐位元相同
                    jd, nv = job(pol, cpu, vk, gw=gw, gpol=gp, sc=sc, extras=[f"w5={BFB}"], files={"w5.f64": w5_file(vk, "degen")})
                    same(f"w5_degen_eq_v7 {tag}", run("v8", jd, nv), r7)
                    shutil.rmtree(jd, ignore_errors=True)
                    # 2. 真的 W5：CPU 讀的位元組＝CPU 命中塊數 × i4（BF16 位置與預取位置都是 INT4）；寫入是 i4 的整數倍
                    jd, nv = job(pol, cpu, vk, gw=gw, gpol=gp, sc=sc, extras=[f"w5={i4}"], files={"w5.f64": w5_file(vk, "real", T)})
                    rr = run("v8", jd, nv)
                    shutil.rmtree(jd, ignore_errors=True)
                    if rr[0] == "CRASH":
                        report(f"w5_real {tag}", False, crash=rr[1]["stderr"][-500:])
                        continue
                    ok, info = True, []
                    for k in range(nv):
                        nc = np.frombuffer(rr[1][f"v{k}_nc.i32"], np.int32).astype(np.int64)
                        cb = np.frombuffer(rr[1][f"v{k}_chanb.i64"], np.int64)
                        t7 = np.frombuffer(r7[1][f"v{k}_ttft.f64"], np.float64)
                        t8 = np.frombuffer(rr[1][f"v{k}_ttft.f64"], np.float64)
                        ok &= bool(cb[0] == nc.sum() * i4) and bool(cb[1] % i4 == 0)
                        info.append(dict(cpu_hits_v7=int(np.frombuffer(r7[1][f"v{k}_nc.i32"], np.int32).sum()), cpu_hits_w5=int(nc.sum()),
                                         mean_ttft_v7=float(t7.mean()), mean_ttft_w5=float(t8.mean())))
                    report(f"w5_real_bytes {tag}", ok, views=info[:2])


# ------------------------------------------------------------------ APC／BELP
def t_apc():
    NZ = dict(model="P", lam=0.3, draw="B1", seed=0)
    for cpu in a.cpus:
        for pol in ("LRU", "O-BEL", "GDSF", "LRU+RPF", "ARC", "O-NOW", "LRU+2HA"):
            tdep = pol not in ("LRU", "O-BEL", "ARC", "O-NOW", "LRU+2HA")
            vk = ONE if tdep else V6
            for gw in ("wb", "wt"):
                for sc in (None, dict(frac=0.5, pol="BEL")):
                    if sc and gw == "wt":
                        continue
                    tag = f"cpu={cpu} {pol} {gw} sc={'on' if sc else 'off'}"
                    for go, gref in (("apc", "LRU"), ("belp", "BEL")):
                        jd7, nv = job(pol, cpu, vk, gw=gw, gpol=gref, sc=sc)
                        r7 = run("v7", jd7, nv)
                        shutil.rmtree(jd7, ignore_errors=True)
                        jd, nv = job(pol, cpu, vk, gw=gw, gpol=gref, go=go, sc=sc)
                        # 1. 一次逐出整個 session ＝ v7 的 GPU LRU／BEL（含 wb 的「先寫進 CPU」路徑）
                        same(f"{go}_whole_eq_v7_{gref} {tag}", run("v8", jd, nv, env_extra={"X1FAST_APC_WHOLE": "1", "X1FAST_V8_CHECK": "200"}), r7)
                        # 2. 逐塊逐出：每 200 個請求從頭重算 ref_gpu、ref_gc、gpu_used（斷言在 Rust 裡）
                        rr = run("v8", jd, nv, env_extra={"X1FAST_V8_CHECK": "200"})
                        ok = rr[0] != "CRASH"
                        info = {}
                        if ok:
                            t7 = np.frombuffer(r7[1]["v0_ttft.f64"], np.float64); t8 = np.frombuffer(rr[1]["v0_ttft.f64"], np.float64)
                            g7 = np.frombuffer(r7[1]["v0_ng.i32"], np.int32); g8 = np.frombuffer(rr[1]["v0_ng.i32"], np.int32)
                            nblk = int(np.fromfile(os.path.join(jd, "out", "j0", "eng_v8.i64"), np.int64)[0])
                            info = dict(mean_ttft_v7=float(t7.mean()), mean_ttft_v8=float(t8.mean()), gpu_blocks_v7=int(g7.sum()),
                                        gpu_blocks_v8=int(g8.sum()), n_gblk=nblk, eng_v7=eng(r7[1]), eng_v8=eng(rr[1]))
                        else:
                            info = dict(crash=rr[1]["stderr"][-800:])
                        report(f"{go}_partial_audit {tag}", ok, **info)
                        if go == "belp":                        # 兩份實作互比：捷徑（預設）＝每塊全掃（X1FAST_BELP_SLOW=1）
                            same(f"belp_fast_eq_slow {tag}", run("v8", jd, nv), run("v8", jd, nv, env_extra={"X1FAST_BELP_SLOW": "1"}))
                        # 3. 不會逐出（GPU 很大）＝v7
                        shutil.rmtree(jd, ignore_errors=True)
                        jdb, nv = job(pol, cpu, vk, gw=gw, gpol=gref, go=go, sc=sc, gcap=1e6)
                        jdb7, _ = job(pol, cpu, vk, gw=gw, gpol=gref, sc=sc, gcap=1e6)
                        same(f"{go}_bigGPU_eq_v7 {tag}", run("v8", jdb, nv, env_extra={"X1FAST_V8_CHECK": "500"}), run("v7", jdb7, nv))
                        shutil.rmtree(jdb, ignore_errors=True); shutil.rmtree(jdb7, ignore_errors=True)
    # 4. 雜訊（B1）：BELP 一次整個 session ＝ v7 GPU BEL
    for cpu in a.cpus:
        jd7, _ = job("O-BEL", cpu, ONE, gpol="BEL", nz=NZ)
        jd, _ = job("O-BEL", cpu, ONE, gpol="BEL", go="belp", nz=NZ)
        same(f"belp_whole_eq_v7_BEL_noiseB1 cpu={cpu}", run("v8", jd, 1, env_extra={"X1FAST_APC_WHOLE": "1"}), run("v7", jd7, 1))
        rr = run("v8", jd, 1, env_extra={"X1FAST_V8_CHECK": "200"})
        report(f"belp_partial_audit_noiseB1 cpu={cpu}", rr[0] != "CRASH", digest=rr[0])
        shutil.rmtree(jd, ignore_errors=True); shutil.rmtree(jd7, ignore_errors=True)
    # 5. 和其他新功能一起：APC＋W5、APC＋排隊（能跑、稽核過）
    T = w5_d1()
    for cpu in a.cpus:
        jd, nv = job("LRU", cpu, V6, go="apc", extras=[f"w5={K['int4_bytes']}"], files={"w5.f64": w5_file(V6, "real", T)})
        rr = run("v8", jd, nv, env_extra={"X1FAST_V8_CHECK": "200"})
        report(f"apc_w5_audit cpu={cpu}", rr[0] != "CRASH", digest=rr[0])
        shutil.rmtree(jd, ignore_errors=True)
        jd, nv = job("LRU", cpu, ONE, go="apc", extras=["q=1"])
        rr = run("v8", jd, nv, env_extra={"X1FAST_V8_CHECK": "200"})
        report(f"apc_queue_audit cpu={cpu}", rr[0] != "CRASH", digest=rr[0])
        shutil.rmtree(jd, ignore_errors=True)


# ------------------------------------------------------------------ 排隊 closed-loop（q=2）
def wc_closed(arr0, sess, svc):
    """對照（近似）：工作守恆的 closed-loop FCFS 事件模擬——同樣的服務時間，但照「延後後的到達」排序服務（不是照紀錄順序）。
    每個 session 的下一個請求到達＝紀錄的時刻＋這個 session 累積的等待。回傳每個請求的等待。"""
    import heapq
    n = len(arr0)
    order = {}
    for r in range(n):
        order.setdefault(int(sess[r]), []).append(r)
    pos = {s: 0 for s in order}
    D = {s: 0.0 for s in order}
    H = [(arr0[v[0]], v[0], s) for s, v in order.items()]
    heapq.heapify(H)
    free, W = -np.inf, np.zeros(n)
    while H:
        a, r, s = heapq.heappop(H)
        st = max(a, free)
        W[r] = st - a
        free = st + svc[r]
        D[s] += W[r]
        pos[s] += 1
        if pos[s] < len(order[s]):
            nr = order[s][pos[s]]
            heapq.heappush(H, (arr0[nr] + D[s], nr, s))
    return W


def t_queue2():
    td2 = os.path.join(a.out, "trace_spread")
    arr = np.fromfile(os.path.join(TDIR, "t.f64"), np.float64)
    sess = sess_of()
    for cpu in a.cpus:
        for pol, gp in (("LRU", "LRU"), ("O-BEL", "BEL"), ("LRU+RPF", "LRU"), ("GDSF", "TTLA:10")):
            for vk in ([V6[0]], [V6[5]]):
                tag = f"cpu={cpu} {pol} {gp} {vk[0]}"
                jdq, _ = job(pol, cpu, vk, gpol=gp, extras=["q=2"])
                if os.path.isdir(td2):
                    jd0, _ = job(pol, cpu, vk, gpol=gp)
                    same(f"queue2_spread_eq_noqueue {tag}", run("v8", jdq, 1, tdir=td2), run("v8", jd0, 1, tdir=td2))
                    shutil.rmtree(jd0, ignore_errors=True)
                rq = run("v8", jdq, 1)
                shutil.rmtree(jdq, ignore_errors=True)
                if rq[0] == "CRASH":
                    report(f"queue2_real {tag}", False, crash=rq[1]["stderr"][-500:]); continue
                st, sv, w, tt, aa = (np.frombuffer(rq[1][k], np.float64) for k in ("qstart.f64", "qsvc.f64", "qwait.f64", "v0_ttft.f64", "qarr.f64"))
                # 精確重算（同樣的浮點運算順序）：到達＝紀錄＋累積等待；開始＝max(到達, 前一個開始＋服務)
                D, ok = {}, True
                prev = -np.inf
                for r in range(len(arr)):
                    s = int(sess[r])
                    ar = arr[r] + D.get(s, 0.0)
                    t0 = prev if prev > ar else ar
                    ok &= (ar == aa[r]) and (t0 == st[r]) and (t0 - ar == w[r]) and (sv[r] + w[r] == tt[r])
                    D[s] = D.get(s, 0.0) + (t0 - ar)
                    prev = st[r] + sv[r]
                wc = wc_closed(arr, sess, sv)
                report(f"queue2_real_exact {tag}", bool(ok), mean_wait=float(w.mean()), p90_wait=float(np.percentile(w, 90)),
                       max_wait=float(w.max()), frac_wait=float((w > 0).mean()), mean_svc=float(sv.mean()),
                       wc_mean_wait=float(wc.mean()), wc_p90_wait=float(np.percentile(wc, 90)))


# ------------------------------------------------------------------ 排隊的一階近似（qpost=1）
def closed_post(t, sess, svc):
    """Python 版（照 Rust 的運算順序）：工作守恆 closed-loop FCFS；到達＝紀錄時刻＋這個 session 上一個請求開始得比紀錄晚多少。"""
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
    W = np.zeros(n)
    S = np.zeros(n)
    while H:
        a, r = heapq.heappop(H)
        st = free if free > a else a
        W[r] = st - a
        S[r] = st
        free = st + svc[r]
        late = st - t[r]
        if nxt[r] >= 0:
            heapq.heappush(H, (float(t[nxt[r]] + late), nxt[r]))
    return W, S


def t_qpost():
    td2 = os.path.join(a.out, "trace_spread")
    if not os.path.isdir(td2):
        os.makedirs(td2)
        for f in os.listdir(TDIR):
            if f != "t.f64":
                os.symlink(os.path.join(TDIR, f), os.path.join(td2, f))
        (np.arange(NREQ, dtype=np.float64) * 1e6).tofile(os.path.join(td2, "t.f64"))
    arr = np.fromfile(os.path.join(TDIR, "t.f64"), np.float64)
    sess = sess_of()
    T = w5_d1()
    for cpu in a.cpus:
        for pol, gp, gw, extra in (("LRU", "LRU", "wb", []), ("O-BEL", "BEL", "wb", []), ("GDSF", "TTLA:10", "wb", []),
                                   ("LRU+RPF", "LRU", "wb", []), ("LRU", "LRU", "wt", []), ("O-PF", "TTLB:10", "wb", []),
                                   ("LRU", "apc", "wb", []), ("LRU", "LRU", "wb", ["w5"])):
            tdep = pol not in ("LRU", "O-BEL") or gp.startswith("TTL")
            vk = ONE if tdep else V6
            go = "apc" if gp == "apc" else None
            gpol = "LRU" if gp == "apc" else gp
            ex, fl = [], {}
            if extra == ["w5"]:
                ex, fl = [f"w5={K['int4_bytes']}"], {"w5.f64": w5_file(vk, "real", T)}
            tag = f"cpu={cpu} {pol} {gp} {gw} {'+'.join(extra) or '-'} views={len(vk)}"
            jd0, nv = job(pol, cpu, vk, gw=gw, gpol=gpol, go=go, extras=ex, files=fl)
            jdq, _ = job(pol, cpu, vk, gw=gw, gpol=gpol, go=go, extras=ex + ["qpost=1"], files=fl)
            # 1. 退化：到達改成 r × 10^6 秒 → 等待全 0 → 和沒開 qpost 逐位元相同
            r0s, rqs = run("v8", jd0, nv, tdir=td2), run("v8", jdq, nv, tdir=td2)
            same(f"qpost_spread_eq_noq {tag}", rqs, r0s)
            if rqs[0] != "CRASH":
                zw = all((np.fromfile(os.path.join(jdq, "out", "j0", f"v{k}_qpw.f64"), np.float64) == 0).all() for k in range(nv))
                report(f"qpost_spread_zero_wait {tag}", zw)
            # 2. 原本的到達：服務時間＝沒開 qpost 的 TTFT；Python 照同樣的運算重算等待、開始時刻、TTFT，逐位元相同
            r0 = run("v8", jd0, nv)
            rq = run("v8", jdq, nv)
            if rq[0] == "CRASH" or r0[0] == "CRASH":
                report(f"qpost_real_exact {tag}", False, crash=(rq[1] if rq[0] == "CRASH" else r0[1])["stderr"][-500:])
            else:
                ok, info = True, []
                for k in range(nv):
                    svc = np.frombuffer(r0[1][f"v{k}_ttft.f64"], np.float64)
                    tq = np.frombuffer(rq[1][f"v{k}_ttft.f64"], np.float64)
                    w = np.fromfile(os.path.join(jdq, "out", "j0", f"v{k}_qpw.f64"), np.float64)
                    st = np.fromfile(os.path.join(jdq, "out", "j0", f"v{k}_qps.f64"), np.float64)
                    W, S = closed_post(arr, sess, svc)
                    ok &= bool(np.array_equal(W, w) and np.array_equal(S, st) and np.array_equal(svc + W, tq))
                    # 其他輸出（還原時間、命中數、I/O 線、引擎計數）和沒開 qpost 相同：快取狀態不受影響
                    for nm in (f"v{k}_rest.f64", f"v{k}_nc.i32", f"v{k}_ns.i32", f"v{k}_nr.i32", f"v{k}_ng.i32", f"v{k}_chan.f64", f"v{k}_chanb.i64"):
                        ok &= r0[1][nm] == rq[1][nm]
                    inv = float(np.mean(S[1:] < np.maximum.accumulate(S)[:-1]))
                    info.append(dict(mean_wait=float(W.mean()), p90_wait=float(np.percentile(W, 90)), frac_wait=float((W > 0).mean()),
                                     mean_svc=float(svc.mean()), frac_reordered=inv))
                ok &= rq[1]["eng.i64"] == r0[1]["eng.i64"]
                report(f"qpost_real_exact {tag}", ok, views=info[:2] if nv > 1 else info)
            shutil.rmtree(jd0, ignore_errors=True); shutil.rmtree(jdq, ignore_errors=True)


for t in a.tests:
    {"cue": t_cue, "queue": t_queue, "queue2": t_queue2, "w5": t_w5, "apc": t_apc, "qpost": t_qpost}[t]()
print(f"V8_RESULT {'PASS' if FAIL == 0 else 'FAIL'} fail={FAIL} trace={a.trace}", flush=True)
sys.exit(1 if FAIL else 0)
