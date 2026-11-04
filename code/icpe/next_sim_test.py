"""next_sim_test.py — ICPE next（icpe-sim）：GPU 快取層與 nvme807 慢層的自檢（docs/icpe_brief_20261030/next/SIM_CHANGES.md）。

  P3_CONST=h200 P3_TIER=h200cpu M7_RES=results/m7_write_policy_h200 M7_TIER_PARAMS=results/icpe_h200/t1/tier_params_h200cpu.json \
  X1_TRACE_DIR=<Mooncake traces> python code/icpe/next_sim_test.py --out results/icpe_next/sim/next_sim_test.csv [--workers 8]

檢查（任一條不過 → exit 1）：
  T1 wt_tiny_eq_off：直寫、GPU 預算 1 byte（什麼都放不下）× 每種 CPU 規則 × GPU 規則 {LRU, BEL, TTLA:5, TTLB:5}
     ≡ GPU 層關閉（原本的路徑）：逐請求 TTFT／還原時間／nC／nS／重算數、CPU 與慢層的讀寫位元組與忙碌時間逐位元相同。
     （直寫的 CPU 層是原本回合結束程式的複本，這條驗證複本沒有改到行為。）
  T2 audit：wb／wt × GPU 規則 × 每種 CPU 規則，GPU 4 GiB、CPU 32 GiB、前 N 個請求，每 25 步重算 refcount、已用位元組、
     各層持有者（Engine.audit＋audit_gpu），沒有 AssertionError。
  T3 huge_gpu：GPU 預算大到放得下全部（1e6 GiB）：wb 與 wt 的逐請求 TTFT 相同（存下來的都在 GPU 上），回來的請求 nC＝nS＝0，
     wb 的 CPU／慢層寫入＝0。
  T4 restore_gpu_prefix：隨機 layout，lay[0:g]＝GPU：g＝L 時 TTFT＝新 block 的重算時間、nC＝nS＝0、重算數＝n_all−L；
     同一個 layout 把前綴換成 GPU（全新 I/O 線）：只載入不會變慢；Cake 最多慢「一塊的成本」（貪婪會合點的離散誤差，次數照報）。
  T5 nvme807：常數＝G-fio CSV 的值；寫＝讀×SSD 級寫讀比；k_read＝SSD 級。
  T6 gpu_pf_audit：預取直接搬回 GPU（PREREG §2 第 2 項）：wb／wt × {O-PF, LRU+RPF, O-ALL} × GPU 規則，GPU 4 GiB、CPU 32 GiB、
     前 N 個請求，每 25 步稽核（含預取進 GPU 的塊：主人、計數、GPU 位元組）；有預取進 GPU 的塊被用到（報次數）。
  T7 gpu_pf_tiny_eq_nopf：GPU 1 byte（放不下任何一塊）時，預取目的地＝GPU 的 O-PF／LRU+RPF ≡ 同樣設定但沒有預取的 LRU（逐位元）。
只用 CPU；不碰 GPU。
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import socket
import sys
import time
from datetime import datetime
from multiprocessing import get_context

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
REPO = os.path.dirname(CODE)
sys.path.insert(0, os.path.join(CODE, "p3x"))
sys.path.insert(0, CODE)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402

import x1_sim as XS  # noqa: E402

GiB = XS.GiB
GPOLS = ["LRU", "BEL", "TTLA:5", "TTLB:5"]
_K = None


def keys_for(pol, gpol=None):
    o, a, h, pf = XS.POLICIES[pol]
    tdep = pf or (gpol is not None and XS.parse_gpol(gpol)[1] is not None)
    if tdep or o == "gdsf":
        return [("nfs", "busy", "cake")]
    return list(XS.ALL_VIEWS)


def fingerprint(views):
    out = []
    for v in views:
        out.append((v.ttft.tobytes(), v.rest.tobytes(), v.nC.tobytes(), v.nS.tobytes(), v.nR.tobytes(),
                    v.cpu.bytes_r, v.cpu.bytes_w, v.cpu.busy_r, v.cpu.busy_w,
                    v.slow.bytes_r, v.slow.bytes_w, v.slow.busy_r, v.slow.busy_w, v.pf_bytes))
    return out


def run(P, pol, cpu_gib, keys, **gkw):
    views = XS.make_views(_K, P, keys)
    gv = views[0] if XS.POLICIES[pol][0] == "gdsf" else None
    eng = XS.Engine(P, pol, int(cpu_gib * GiB), views, _K, gdsf_view=gv, **gkw)
    eng.run()
    return eng, views


def t1_job(arg):
    trace, pol, cpu, gpol = arg
    P = XS.prep(trace)
    keys = keys_for(pol, gpol)
    _, v0 = run(P, pol, cpu, keys)
    _, v1 = run(P, pol, cpu, keys, gpu_cap=1, gpu_policy=gpol, gpu_write="wt")
    eq = fingerprint(v0) == fingerprint(v1)
    return f"wt_tiny_eq_off[{trace},{cpu}GiB,{pol},G:{gpol}]", eq, f"{len(keys)} 種時間設定" + ("逐位元相同" if eq else "：不同")


def t2_job(arg):
    trace, pol, gpol, wm, nsub = arg
    P = XS.prep(trace)
    sub = dict(P)
    for k in ("full", "nall", "tl", "kindc", "sessl", "parentl"):
        sub[k] = P[k][:nsub]
    keys = keys_for(pol, gpol)
    keys = keys[:2] if len(keys) > 2 else keys
    try:
        views = XS.make_views(_K, sub, keys)
        gv = views[0] if XS.POLICIES[pol][0] == "gdsf" else None
        eng = XS.Engine(sub, pol, 32 * GiB, views, _K, gdsf_view=gv, check=25, gpu_cap=4 * GiB, gpu_policy=gpol, gpu_write=wm)
        eng.run()
        det = f"前 {nsub} 個請求、GPU 4 GiB、CPU 32 GiB、每 25 步；GPU 逐出 {eng.n_gevict}、到期 {eng.n_gexpire}、GPU 命中 block {int(views[0].nG.sum())}"
        return f"audit[{trace},{wm},{pol},G:{gpol}]", True, det
    except AssertionError as e:
        return f"audit[{trace},{wm},{pol},G:{gpol}]", False, repr(e)


def t3_job(arg):
    trace, pol, cpu = arg
    P = XS.prep(trace)
    keys = keys_for(pol)
    ew, vw = run(P, pol, cpu, keys, gpu_cap=int(1e6 * GiB), gpu_policy="LRU", gpu_write="wb")
    et, vt = run(P, pol, cpu, keys, gpu_cap=int(1e6 * GiB), gpu_policy="LRU", gpu_write="wt")
    ret = np.array(P["kindc"]) == 1
    same = all(np.array_equal(a.ttft, b.ttft) for a, b in zip(vw, vt))
    noload = all(int(v.nC[ret].sum()) == 0 and int(v.nS[ret].sum()) == 0 for v in vw + vt)
    nowrite = all(v.cpu.bytes_w == 0 and v.slow.bytes_w == 0 for v in vw)
    ok = same and noload and nowrite
    return (f"huge_gpu[{trace},{cpu}GiB,{pol}]", ok,
            f"wb≡wt TTFT {same}；回來的請求 nC＝nS＝0 {noload}；wb 不寫 CPU／慢層 {nowrite}；GPU 命中 block {int(vw[0].nG[ret].sum())}")


def t4(rec):
    rng = random.Random(11)
    bad = mono = slack = 0
    for trial in range(400):
        n_all = rng.randint(1, 150)
        L = rng.randint(0, n_all)
        slow, gst, mode = rng.choice(XS.SLOWS), rng.choice(XS.GPUS), rng.choice(["cake", "load"])
        codes = [rng.choice([1, 3]) for _ in range(L)]
        g = rng.randint(0, L)
        pfd = [0.0] * max(L, 1)

        def restore(gg):
            v = XS.View(_K, slow, gst, mode, 300, 1)
            return v, v.restore([0] * gg + codes[gg:], list(range(L)), L, n_all, 1.0, L, pfd, gg)
        vL, oL = restore(L)
        if not (abs(oL[0] - (vL.Fc[n_all] - vL.Fc[L])) < 1e-12 and oL[2] == 0 and oL[3] == 0 and oL[4] == n_all - L):
            bad += 1
        v0, o0 = restore(0)
        _, og = restore(g)
        if og[0] > o0[0] + 1e-12:
            # Cake 的會合點是貪婪的雙指標：起點從 0 換成 g，會合點最多差一塊 → 只准慢「不到一塊的成本」；只載入必須不變慢
            step = max(max(v0.f[:L]) if L else 0.0, XS.BF / v0.cpu.r, v0.i4 / v0.slow.r + v0.d)
            if mode == "load" or og[0] - o0[0] > step:
                mono += 1
            else:
                slack += 1
    rec("restore_gpu_prefix", bad == 0 and mono == 0,
        f"400 個隨機 layout：g＝L 不符 {bad}；前綴換成 GPU 反而變慢（只載入，或 Cake 超過一塊）{mono}；"
        f"Cake 慢了不到一塊（貪婪會合點的離散誤差）{slack}")


def t6_job(arg):
    trace, pol, gpol, wm, nsub = arg
    P = XS.prep(trace)
    sub = dict(P)
    for k in ("full", "nall", "tl", "kindc", "sessl", "parentl"):
        sub[k] = P[k][:nsub]
    keys = [("nfs", "busy", "cake")]
    try:
        views = XS.make_views(_K, sub, keys)
        eng = XS.Engine(sub, pol, 32 * GiB, views, _K, check=25, gpu_cap=4 * GiB, gpu_policy=gpol, gpu_write=wm, pf_target="gpu")
        eng.run()
        return (f"gpu_pf_audit[{trace},{wm},{pol},G:{gpol}]", True,
                f"前 {nsub} 個請求、GPU 4 GiB、CPU 32 GiB、每 25 步；預取 {views[0].pf_bytes / GiB:.1f} GiB、GPU 命中 block {int(views[0].nG.sum())}、取消預取 {eng.n_unpf}")
    except AssertionError as e:
        return f"gpu_pf_audit[{trace},{wm},{pol},G:{gpol}]", False, repr(e)


def t7_job(arg):
    trace, pol, wm = arg
    P = XS.prep(trace)
    keys = [("nfs", "busy", "cake")]
    _, v0 = run(P, "LRU", 128, keys, gpu_cap=1, gpu_policy="LRU", gpu_write=wm)
    _, v1 = run(P, pol, 128, keys, gpu_cap=1, gpu_policy="LRU", gpu_write=wm, pf_target="gpu")
    eq = fingerprint(v0) == fingerprint(v1)
    return f"gpu_pf_tiny_eq_nopf[{trace},{wm},{pol}]", eq, "逐位元相同" if eq else "不同"


def t5(rec):
    tp = __import__("json").load(open(XS.PF.TIER_PARAMS))
    x = XS.extra_slow("nvme807", tp)
    rows = [r for r in csv.DictReader(open(XS.GFIO_CSV)) if r["dir"].startswith("/tmp/") and r["pattern"] == "seq"
            and r["io_bytes"] == str(64 << 20) and r["threads"] == "8"]
    want_r = float(rows[0]["gibps"])
    want_w = want_r * tp["local"]["write_GiBps"] / tp["local"]["read_GiBps"]
    ok = x["r"] == want_r and abs(x["w"] - want_w) < 1e-12 and x["k"] == tp["local"]["k_read"]
    rec("nvme807_consts", ok, f"讀 {x['r']} GiB/s、寫 {x['w']:.6f} GiB/s、k_read {x['k']}（{x['src_run']}；{XS.PF.TIER_PARAMS} 的 local）")


def main():
    global _K
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--traces", nargs="+", default=["conversation", "toolagent"])
    ap.add_argument("--cpu", nargs="+", type=int, default=[128])
    ap.add_argument("--audit-n", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    _K = XS.constants()
    XS._K = _K
    for tr in a.traces:
        XS.prep(tr)
    rid = os.environ.get("RUN_ID", "adhoc")
    host = socket.gethostname()
    rows, ok = [], True

    def rec(name, passed, detail):
        nonlocal ok
        ok &= bool(passed)
        rows.append(dict(run_id=rid, ts=datetime.now().astimezone().isoformat(timespec="seconds"), host=host,
                         test=name, passed=bool(passed), detail=detail))
        print(("PASS " if passed else "FAIL ") + name + " :: " + detail, flush=True)

    t0 = time.time()
    t4(rec)
    t5(rec)
    pols = [p for p in XS.POLICIES]
    J1 = [(tr, p, c, gp) for tr in a.traces for c in a.cpu for p in pols for gp in GPOLS]
    J2 = [(tr, p, gp, wm, a.audit_n) for tr in a.traces for p in pols for gp in GPOLS for wm in XS.GPU_WRITES]
    J3 = [(tr, p, c) for tr in a.traces for c in a.cpu for p in pols]
    J6 = [(tr, p, gp, wm, a.audit_n) for tr in a.traces for p in ("O-PF", "LRU+RPF", "O-ALL") for gp in GPOLS for wm in XS.GPU_WRITES]
    J7 = [(tr, p, wm) for tr in a.traces for p in ("O-PF", "LRU+RPF") for wm in XS.GPU_WRITES]
    with get_context("fork").Pool(a.workers) as pool:
        for fn, J in ((t1_job, J1), (t2_job, J2), (t3_job, J3), (t6_job, J6), (t7_job, J7)):
            for res in pool.imap(fn, J):
                rec(*res)
            print(f"[{fn.__name__}] {len(J)} 個，累計 {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
    print("next_sim_test", "OK" if ok else "FAILED", f"{sum(r['passed'] for r in rows)}/{len(rows)}", f"{time.time() - t0:.0f}s", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
