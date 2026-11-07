"""v8_dev14（v8＋PREREG 偏離 14 的規則）的驗證。比的是 x1fast 的原始輸出檔（逐請求陣列、I/O 線狀態、引擎計數）。
用法（rux_run.sh 的 run 目錄裡）：python -I smoke_v8_dev14.py --trace nb:semianalysis_N16_s1 --rule nb --ref-bin <icpe-sim 的 x1fast3> --out <dir>
測試（每個印一行 PASS／FAIL，最後印 D14_RESULT）：
  D1 off_eq_v8：旗標都沒開 → v8_dev14 和 v8（48e4efce）逐位元相同。含 APC／BELP、雜訊 B1／B2、預取到 CPU／GPU。
  D2 on_eq_ref：不開 APC／BELP、開 pfx=1＋nzo=resample＋pgr=1 → v8_dev14 和 icpe-sim 的 x1fast（--ref-bin，63b8e328）逐位元相同；旗標只開一個也比。
     雜訊：無、P／M × B1／B2 × λ∈{0, 0.3}。GPU：關、LRU、BEL、TTLA:10；預取到 CPU，O-PF／O-ALL／LRU+RPF 另跑預取到 GPU。
  D3 apc_whole：開 APC／BELP＋雜訊＋三個旗標，X1FAST_APC_WHOLE=1（一次逐出整個 session）→ APC＝GPU LRU、BELP＝GPU BEL（同一個 v8_dev14）。
     另外逐塊逐出時開 X1FAST_V8_CHECK=200（APC 的引用數稽核）要能跑完。
  D4 flags_matter（只報告）：開旗標和不開的差，確認旗標真的有作用（O-PF／O-ALL 的 pfx、模型 P 的 nzo）。"""
import argparse, hashlib, json, os, shutil, socket, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

ap = argparse.ArgumentParser()
ap.add_argument("--trace", required=True)
ap.add_argument("--rule", default="nb")
ap.add_argument("--ref-bin", required=True)
ap.add_argument("--cpus", nargs="+", type=float, default=[16.0, 111.0625])
ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--tests", nargs="+", default=["D1", "D2", "D3"])
ap.add_argument("--pols", nargs="+", default=None)
ap.add_argument("--gw", default="wb", choices=["wb", "wt"])   # wt：只預取到 CPU（wt＋預取到 GPU 在旗標全關時就會斷言，使用者決定不修；icpe-sim STATUS_SIM 15:56）
a = ap.parse_args()
REPO = os.environ["REPO"]
for p in ("code", "code/icpe", "code/p3x"):
    sys.path.insert(0, os.path.join(REPO, p))
sys.path.insert(0, os.path.join(REPO, "code/icpe/rux_x1fast/v6_slowcap"))
import numpy as np
import x1_rust_v6 as XR
sys.modules["x1_rust"] = XR
import p3_fvar
assert sys.version_info[:2] == (3, 12), "雜訊 B2／nzo=resample 的亂數順序照 CPython 3.12 的 set"
os.makedirs(a.out, exist_ok=True)
XS = p3_fvar.patch_x1("V1", os.path.join(REPO, "results/icpe_h200/p01/calib"), a.out, {})
K = XS.constants()
XS._K = K
P = XS.prep(a.trace, a.rule)
TDIR, _ = XR.prepare_trace(P, a.trace, a.rule)
BIN = {}
for tag, d in (("v8", "v8_features"), ("d14", "v8_dev14")):
    XR.SRC, XR._BIN = os.path.join(REPO, "code/icpe/rux_x1fast", d), None
    BIN[tag] = XR.binary()
    print("binary", tag, BIN[tag], XR.src_sha()[:16], flush=True)
XR._BIN = None
BIN["ref"] = a.ref_bin
print("binary ref", a.ref_bin, hashlib.sha256(open(a.ref_bin, "rb").read()).hexdigest()[:16], flush=True)
V6 = [(s, g, "cake") for s in ("ssd", "b32", "nfs") for g in ("idle", "busy")]
ONE = [("nfs", "busy", "cake")]
GiB = XS.GiB
fh = open(os.path.join(a.out, "smoke_v8_dev14.jsonl"), "a")
FAIL = 0
P17 = ["LRU", "LRU2", "ARC", "S3FIFO", "GDSF", "PENS", "FREQ2Q", "LRU+NONE", "LRU+2HW", "LRU+2HA", "LRU+RPF",
       "O-BEL", "O-BELS", "O-DROP", "O-NOW", "O-PF", "O-ALL"]
PFPOL = ("O-PF", "O-ALL", "LRU+RPF") if a.gw == "wb" else ()     # wt：不跑預取到 GPU


def nzs():
    out = [None]
    for m in ("P", "M"):
        for dr in ("B1", "B2"):
            for lam in (0.0, 0.3):
                out.append(dict(model=m, lam=lam, draw=dr, seed=7))
    return out


def job(policy, cpu, vk, gp="LRU", gcap=111.0625, pft="cpu", nz=None, extras=()):
    views = XS.make_views(K, P, vk)
    go = gp.lower() if gp in ("APC", "BELP") else None
    gkw = dict(gpu_cap=int(gcap * GiB), gpu_policy="LRU" if go else gp, gpu_write=a.gw, pf_target=pft) if gcap > 0 else dict(gpu_cap=0)
    jd = XR.write_job(XS, P, policy, cpu * GiB, views, K, gkw, nz=nz)
    x = open(os.path.join(jd, "jobs.txt")).read().split()
    if go is not None:
        x[8] = go
    x += list(extras)
    open(os.path.join(jd, "jobs.txt"), "w").write(" ".join(x) + "\n")
    return jd, len(vk)


def run(tag, jd, nv, env_extra=None, outname="out"):
    jd2 = jd + "_" + outname
    os.makedirs(jd2, exist_ok=True)
    for f in os.listdir(jd):
        if not os.path.exists(os.path.join(jd2, f)):
            os.symlink(os.path.join(jd, f), os.path.join(jd2, f))
    shutil.rmtree(os.path.join(jd2, "out"), ignore_errors=True)
    env = dict(os.environ, X1FAST_THREADS="1", X1FAST_FORK="0", X1FAST_SUMMARY_ONLY="0", **(env_extra or {}))
    env.pop("X1FAST_SERVE", None)
    cp = subprocess.run([BIN[tag], TDIR, jd2], env=env, capture_output=True, text=True)
    if cp.returncode != 0:
        return "CRASH", {"stderr": cp.stderr[-3000:]}
    od = os.path.join(jd2, "out", "j0")
    names = [f"v{k}_{n}" for k in range(nv) for n in ("ttft.f64", "rest.f64", "nc.i32", "ns.i32", "nr.i32", "ng.i32", "chan.f64", "chanb.i64")]
    names += ["eng.i64", "eng.f64"]
    h = hashlib.sha256()
    raw = {}
    for n in names:
        b = open(os.path.join(od, n), "rb").read()
        raw[n] = b
        h.update(n.encode() + b"\0" + b)
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


def tdep(pol, gp):
    return XS.POLICIES[pol][3] is not None or gp.startswith("TTL") or XS.POLICIES[pol][0] == "gdsf"


def nzt(nz):
    return "-" if nz is None else f"{nz['model']}{nz['draw']}l{nz['lam']}"


def d1(c):
    pol, gp, cpu, nz, pft = c
    vk = ONE if tdep(pol, gp) or nz is not None else V6
    tag = f"cpu={cpu} {pol} {gp} nz={nzt(nz)} pft={pft} views={len(vk)}"
    jd, nv = job(pol, cpu, vk, gp=gp, nz=nz, pft=pft)
    r8, rd = run("v8", jd, nv, outname="a"), run("d14", jd, nv, outname="b")
    cleanup(jd)
    return [(f"off_eq_v8 {tag}", r8[0] == rd[0] and r8[0] != "CRASH", dict(a=r8[0], b=rd[0], err=(rd if rd[0] == "CRASH" else r8)[1].get("stderr", "")[-300:]))]


def d2(c):
    pol, gp, gcap, cpu, nz, pft = c
    vk = ONE if tdep(pol, gp) or nz is not None else V6
    out = []
    for fl in (["pfx=1", "nzo=resample", "pgr=1"], ["pfx=1"], ["nzo=resample"], ["pgr=1"]):
        if fl == ["nzo=resample"] and (nz is None or nz["model"] != "P"):
            continue
        if fl == ["pgr=1"] and pft != "gpu":
            continue
        tag = f"cpu={cpu} {pol} {gp if gcap else 'noGPU'} nz={nzt(nz)} pft={pft} {'+'.join(fl)} views={len(vk)}"
        jd, nv = job(pol, cpu, vk, gp=gp, gcap=gcap, nz=nz, pft=pft, extras=fl)
        rr, rd = run("ref", jd, nv, outname="r"), run("d14", jd, nv, outname="d")
        if rr[0] == rd[0] == "CRASH":
            # 兩邊都斷言：比 panic 訊息（去掉 thread 編號與行號，兩支的行號不同），相同＝行為相同（參考版本來就會停），另外標出來
            import re
            def pm(x):
                m = re.search(r"panicked at [^\n]*\n([^\n]*)", x[1]["stderr"])
                return m.group(1) if m else x[1]["stderr"][-200:]
            same_crash = pm(rr) == pm(rd)
            out.append((f"on_eq_ref_both_crash {tag}", same_crash, dict(ref=pm(rr), d14=pm(rd))))
        else:
            out.append((f"on_eq_ref {tag}", rr[0] == rd[0] and rr[0] != "CRASH",
                        dict(a=rr[0], b=rd[0], err=(rd if rd[0] == "CRASH" else rr)[1].get("stderr", "")[-300:])))
        if len(fl) == 3 and "D4" in a.tests:
            j0, _ = job(pol, cpu, vk, gp=gp, gcap=gcap, nz=nz, pft=pft)
            r0 = run("d14", j0, nv, outname="z")
            if r0[0] != "CRASH" and rd[0] != "CRASH":
                t0, t1 = np.frombuffer(r0[1]["v0_ttft.f64"], np.float64), np.frombuffer(rd[1]["v0_ttft.f64"], np.float64)
                out.append((f"flags_matter {tag}", True, dict(changed=r0[0] != rd[0], mean_ttft_off=float(t0.mean()), mean_ttft_on=float(t1.mean()))))
            cleanup(j0)
        cleanup(jd)
    return out


def d3(c):
    pol, cpu, nz = c
    vk = ONE
    out = []
    for go, gref in (("APC", "LRU"), ("BELP", "BEL")):
        tag = f"cpu={cpu} {pol} {go} nz={nzt(nz)} pfx+nzo+pgr"
        fl = ["pfx=1", "nzo=resample", "pgr=1"]
        jr, _ = job(pol, cpu, vk, gp=gref, nz=nz, extras=fl)
        ja, _ = job(pol, cpu, vk, gp=go, nz=nz, extras=fl)
        rr = run("d14", jr, 1, outname="r")
        rw = run("d14", ja, 1, env_extra={"X1FAST_APC_WHOLE": "1", "X1FAST_V8_CHECK": "200"}, outname="w")
        out.append((f"apc_whole {tag}", rr[0] == rw[0] and rr[0] != "CRASH", dict(a=rr[0], b=rw[0], err=(rw if rw[0] == "CRASH" else rr)[1].get("stderr", "")[-300:])))
        rp = run("d14", ja, 1, env_extra={"X1FAST_V8_CHECK": "200"}, outname="p")
        out.append((f"apc_partial_audit {tag}", rp[0] != "CRASH", dict(d=rp[0], err=rp[1].get("stderr", "")[-300:] if rp[0] == "CRASH" else "")))
        cleanup(jr, ja)
    return out


pols = a.pols or P17
C1 = [(pol, gp, cpu, nz, pft) for cpu in a.cpus for pol in pols for gp in ("LRU", "BEL", "TTLA:10", "APC", "BELP")
      for nz in (None, dict(model="P", lam=0.3, draw="B1", seed=7), dict(model="M", lam=0.3, draw="B2", seed=7))
      for pft in (("cpu", "gpu") if (pol in PFPOL and gp not in ("APC", "BELP")) else ("cpu",))]   # APC／BELP 不支援預取進 GPU
C2 = [(pol, gp, gcap, cpu, nz, pft) for cpu in a.cpus for pol in pols for gp, gcap in (("LRU", 0), ("LRU", 111.0625), ("BEL", 111.0625), ("TTLA:10", 111.0625))
      for nz in (nzs() if pol.startswith("O-") else [None, dict(model="P", lam=0.3, draw="B1", seed=7)])   # 雜訊只影響 ④
      for pft in (("cpu", "gpu") if (pol in PFPOL and gcap > 0) else ("cpu",))]
C3 = [(pol, cpu, nz) for cpu in a.cpus for pol in pols for nz in (None, dict(model="P", lam=0.3, draw="B1", seed=7), dict(model="P", lam=0.3, draw="B2", seed=7))]
print(f"cases D1={len(C1)} D2={len(C2)} D3={len(C3)}", flush=True)
with ThreadPoolExecutor(a.workers) as ex:
    for t, fn, C in (("D1", d1, C1), ("D2", d2, C2), ("D3", d3, C3)):
        if t not in a.tests:
            continue
        for rs in ex.map(fn, C):
            for r in rs:
                report(*r[:2], **r[2])
print(f"D14_RESULT {'PASS' if FAIL == 0 else 'FAIL'} fail={FAIL} trace={a.trace} gw={a.gw}", flush=True)
sys.exit(1 if FAIL else 0)
