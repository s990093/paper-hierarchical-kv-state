"""e2e_gpu.py — E2E-C 第二段（docs/stage3_design_20261024/runs/E2E_gpu_mps.md）：MPS 取捨在 GPU 上驗證。

  gpu --bg idle|SPEC --mode stream|mps|mps87|mps75|mps50 [--rep R] [--max-req N] [--out DIR]   一個 GPU run（縮小版 conversation、HINT-PF、CPU 128）
  sim [--out DIR]                                                                                 同一份工作負載的模擬（S_orig；有 GPU 資料時另算 S_real）
  report [--out DIR]                                                                              登記 §3 的判準 → report.json
  --self-test                                                                                     BgView 在 idle 時和 x1_gpu.replay(model_view=True) 逐請求相同

環境同 T1G 的 P02：P3_CONST=h200、P3_TIER=h200cpu、M7_TIER_PARAMS＝T1 的檔、X1_TRACE_DIR。MPS 那幾格要 runner 先開 MPS daemon。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e2e as E  # noqa: E402（會把 code/、code/p3x、code/icpe 放進 sys.path）

import numpy as np  # noqa: E402

XS, XG = E.XS, E.XG
OUT = os.path.join(E.REPO, "results", "s3_h200", "e2e_gpu")
K_SCALE, CPU = 8, 128
POLICY = "HINT-PF"
SLOW = "ssd"
BGS, MODES = E.BGS, E.MODES
GPU_FIELDS = XG.GPU_FIELDS[:-1] + ["bg", "mode", "contaminated"]
RUN_FIELDS = ["run_id", "ts", "bg", "mode", "rep", "n_req", "max_req", "bg_solo", "bg_solo_sigma", "bg_co", "rho", "t_replay_s",
              "mps_pct", "contaminated"]
SIM_FIELDS = ["run_id", "ts", "ref", "src_gpu_run", "bg", "mode", "r", "kind", "t_start", "ttft", "src"]


def run_id():
    return E.run_id()


def ts():
    return E.ts()


def setup():
    E.use_model("llama")
    K = E.constants("llama")
    P = XG.load_P(K_SCALE)
    return K, P, XG.cap_bytes(CPU, K_SCALE)


def _append(path, fields, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


# ============================================================================ GPU
def cmd_gpu(a):
    import torch
    from gpu_guard import GpuWatcher, smi_index
    K, P, cap = setup()
    os.makedirs(a.out, exist_ok=True)
    rid = run_id()
    ex = XG.GpuExec(P["nmax"])
    assert ex.i4_bytes == K["int4_bytes"], (ex.i4_bytes, K["int4_bytes"])
    rows = []
    proc = bgobj = None
    with GpuWatcher(gpu=smi_index(0), own_root=os.getpid(), out_path=os.path.join(a.out, f"gpu_guard_{rid}.json")) as wat:
        XG.replay(P, "LRU", cap, K, ex=ex, max_req=8)                      # 暖機（不記），同 x1_gpu
        solo = sig = None
        pct = None
        if a.bg == "idle":
            rate = None
        elif a.mode == "stream":
            from bgengine import Background
            bgobj = Background(ex.km, a.bg)
            sl = bgobj.solo_rates(6, 1.0)
            solo, sig = statistics.median(sl), statistics.stdev(sl) / statistics.median(sl)
            bgobj.start()
            time.sleep(0.5)
            rate = bgobj.rate
        else:
            import b0
            pct = int(a.mode[3:] or 100)
            proc, rfn, solo, _tps, bsr = b0.start_bgproc(argparse.Namespace(model="llama"), a.bg, pct)
            sig = bsr if isinstance(bsr, float) else None
            rate = lambda t0, t1: rfn(t0, t1)[1]                          # noqa: E731
        t0 = time.time()

        def on_row(row):
            rows.append(dict(run_id=rid, ts=ts(), k=K_SCALE, cpu_GiB=CPU, cap_bytes=cap, policy=POLICY, rep=a.rep, bg=a.bg, mode=a.mode,
                             contaminated=int(wat.contaminated), **row))
            if row["r"] % 200 == 0:
                print(f"[{row['r']}/{len(P['full'])} {time.time() - t0:.0f}s] ttft={row['ttft']:.3f}", flush=True)
        try:
            ta = time.perf_counter()
            XG.replay(P, POLICY, cap, K, ex=ex, max_req=a.max_req, on_row=on_row)
            torch.cuda.synchronize()
            tb = time.perf_counter()
            co = rate(ta, tb) if rate else None
        finally:
            if proc is not None:
                torch.cuda.synchronize()                  # 自己的工作先做完，再請背景優雅地停（b0 的做法，S3-B0-FAIL-3）
                proc.terminate()
                proc.wait(120)
            if bgobj is not None:
                bgobj.stop()
    _append(os.path.join(a.out, a.csv), GPU_FIELDS, rows)
    run = dict(run_id=rid, ts=ts(), bg=a.bg, mode=a.mode, rep=a.rep, n_req=len(rows), max_req=a.max_req or "", bg_solo=solo,
               bg_solo_sigma=sig, bg_co=co, rho=(co / solo if (co is not None and solo) else None), t_replay_s=tb - ta, mps_pct=pct,
               contaminated=int(wat.contaminated))
    _append(os.path.join(a.out, a.runs_csv), RUN_FIELDS, [run])
    print("[e2e_gpu]", json.dumps(run, ensure_ascii=False), flush=True)
    if wat.contaminated:
        print("[e2e_gpu] CONTAMINATED：這次的時間不可用", flush=True)
        return 3
    return 0


# ============================================================================ 模擬
def sim_view(K, P, cap, fac, lead=None, Q=None):
    Q = Q or P
    v = E.BgView(K, SLOW, fac, Q["nmax"], len(Q["full"]))
    XG.make_engine(Q, POLICY, cap, [v], K, lead).run()
    assert v.i == len(Q["full"])
    return v


def cmd_sim(a):
    K, P, cap = setup()
    os.makedirs(a.out, exist_ok=True)
    cell = E.load_b0("llama")
    rid, rows, summ = run_id(), [], {}
    facs = {("idle", "stream"): E.IDLE}
    for bg in BGS:
        for m in MODES:
            try:
                facs[(bg, m)] = E.factors(cell, bg, m, "e2e")
            except E.NotMeasured as ex:
                summ[f"{bg}|{m}"] = dict(status="NOT_MEASURED", reason=str(ex))
    kind = np.asarray(P["kindc"])
    for (bg, m), fac in facs.items():
        v = sim_view(K, P, cap, fac)
        for r in range(len(P["full"])):
            rows.append(dict(run_id=rid, ts=ts(), ref="S_orig", src_gpu_run="", bg=bg, mode=m, r=r, kind=int(kind[r]),
                             t_start=float(P["tl"][r]), ttft=float(v.ttft[r]), src=int(v.src[r])))
        summ[f"{bg}|{m}"] = dict(status="ok", ret_ttft_mean=float(v.ttft[kind == 1].mean()), **bstat(np.asarray(P["tl"]), v.ttft, rho_of(v), fac["cap"]))
        print("S_orig", bg, m, round(summ[f"{bg}|{m}"]["ret_ttft_mean"], 4), flush=True)
    gp = os.path.join(a.out, "gpu_requests.csv")
    if os.path.exists(gp):                               # S_real（只報告）：GPU run 的實際開始時間，提示時間跟著移（同 x1_gpu sim --times）
        import pandas as pd
        g = pd.read_csv(gp)
        L0 = XG.leads(P, K)
        for grid, gg in g.groupby("run_id"):
            gg = gg.sort_values("r")
            if len(gg) != len(P["full"]):
                continue
            key = (gg.bg.iloc[0], gg["mode"].iloc[0])
            if key not in facs:
                continue
            Q = dict(P)
            Q["tl"] = [float(x) for x in gg.t_start.values]
            Q["t"] = np.array(Q["tl"])
            lead = [None if x is None else x + (q - o) for x, q, o in zip(L0, Q["tl"], P["tl"])]
            v = sim_view(K, P, cap, facs[key], lead=lead, Q=Q)
            for r in range(len(P["full"])):
                rows.append(dict(run_id=rid, ts=ts(), ref="S_real", src_gpu_run=grid, bg=key[0], mode=key[1], r=r, kind=int(kind[r]),
                                 t_start=Q["tl"][r], ttft=float(v.ttft[r]), src=int(v.src[r])))
    p = os.path.join(a.out, "sim_requests.csv")
    if os.path.exists(p):
        os.remove(p)                                     # 每次 sim 重寫（只有一份模擬；run_id 在每列）
    _append(p, SIM_FIELDS, rows)
    json.dump(dict(run_id=rid, ts=ts(), cells=summ), open(os.path.join(a.out, "sim_summary.json"), "w"), indent=1, ensure_ascii=False)
    print("→", p, len(rows))
    return 0


def rho_of(v):
    fac = v.fac
    if not fac["rho"]:
        return np.ones(len(v.src))
    rmap = {0: fac["rho"]["recompute1"]}
    for s in (1, 2, 3):
        rmap[s] = fac["rho"][v.action(s)]
    return np.array([rmap[int(s)] for s in v.src])


def bstat(t, ttft, rho, cap):
    """§3.4 的公式，整段重播一個時間窗：φ＝前景占用的聯集 ÷ 窗長，ρ̄＝占用時間加權的 ρ，B＝cap × (1 − φ(1 − ρ̄))。"""
    t, ttft, rho = np.asarray(t, float), np.asarray(ttft, float), np.asarray(rho, float)
    a, b = float(t.min()), float(t.max())
    lo, hi = t, np.minimum(t + ttft, b)
    ln = np.clip(hi - lo, 0, None)
    U, cs, ce = 0.0, None, None
    for x, y in sorted((float(x), float(y)) for x, y, l in zip(lo, hi, ln) if l > 0):
        if cs is None or x > ce:
            if cs is not None:
                U += ce - cs
            cs, ce = x, y
        else:
            ce = max(ce, y)
    if cs is not None:
        U += ce - cs
    phi = U / (b - a)
    rb = float((ln * rho).sum() / ln.sum()) if ln.sum() > 0 else 1.0
    return dict(phi=phi, rho_bar=rb, B=cap * (1 - phi * (1 - rb)))


# ============================================================================ 報告（§3）
def cmd_report(a):
    import pandas as pd
    g = pd.read_csv(os.path.join(a.out, "gpu_requests.csv"))
    runs = pd.read_csv(os.path.join(a.out, "gpu_runs.csv"))
    sim = json.load(open(os.path.join(a.out, "sim_summary.json")))["cells"]
    s = pd.read_csv(os.path.join(a.out, "sim_requests.csv"))
    _, P, _ = setup()
    nreq = len(P["full"])
    clean = runs[(runs.contaminated == 0) & (runs.n_req == nreq) & runs.max_req.isna()]
    clean = clean.sort_values("run_id").groupby(["bg", "mode", "rep"]).tail(1)
    gpu = {}
    for _, r in clean.iterrows():
        x = g[(g.run_id == r.run_id) & (g.kind == 1)]
        y = g[g.run_id == r.run_id].sort_values("r")
        gpu[(r.bg, r["mode"], int(r.rep))] = dict(run=r.run_id, ttft=float(x.ttft.mean()), p95=float(x.ttft.quantile(.95)), rho=r.rho,
                                                    solo=r.bg_solo, t_start=y.t_start.values, ttft_all=y.ttft.values)
    out, cells = dict(run_id=run_id(), ts=ts(), prereg="docs/stage3_design_20261024/runs/E2E_gpu_mps.md"), {}
    for bg in BGS:
        st_g, st_s = gpu.get((bg, "stream", 0)), sim.get(f"{bg}|stream", {})
        for m in MODES:
            G, S = gpu.get((bg, m, 0)), sim.get(f"{bg}|{m}", {})
            c = dict(gpu=None if G is None else {k: G[k] for k in ("run", "ttft", "p95", "rho", "solo")},
                     sim=S if S.get("status") == "ok" else S or {"status": "NOT_MEASURED"})
            if G is None or S.get("status") != "ok":
                c["status"] = "NOT_MEASURED"
                cells[f"{bg}|{m}"] = c
                continue
            c["status"] = "ok"
            c["ratio"] = G["ttft"] / S["ret_ttft_mean"]
            c["a_ok"] = bool(0.9 <= c["ratio"] <= 1.1)
            mps_solo = gpu.get((bg, "mps", 0), {}).get("solo")
            cap_gpu = 1.0 if m in ("stream", "mps") else (G["solo"] / mps_solo if mps_solo else None)
            if G["rho"] is not None and cap_gpu is not None:
                bs = bstat(G["t_start"], G["ttft_all"], np.full(len(G["ttft_all"]), G["rho"]), cap_gpu)
                c.update(cap_gpu=cap_gpu, phi_gpu=bs["phi"], B_gpu=bs["B"], d_rho=G["rho"] - S["rho_bar"], d_B=bs["B"] - S["B"])
                c["c_ok"] = bool(abs(c["d_rho"]) <= 0.05 and abs(c["d_B"]) <= 0.03)
            else:
                c["c_ok"] = None
            if m != "stream":
                if st_g is not None and st_s.get("status") == "ok":
                    gg, gs = 1 - G["ttft"] / st_g["ttft"], 1 - S["ret_ttft_mean"] / st_s["ret_ttft_mean"]
                    c.update(gain_gpu=gg, gain_sim=gs, b_ok=bool(np.sign(gg) == np.sign(gs) and abs(gg - gs) <= 0.10))
                else:
                    c["b_ok"] = None
            cells[f"{bg}|{m}"] = c
    ok = [c for c in cells.values() if c["status"] == "ok"]
    bl = [c["b_ok"] for c in ok if "b_ok" in c and c["b_ok"] is not None]
    al = [c["a_ok"] for c in ok]
    cl = [c["c_ok"] for c in ok if c["c_ok"] is not None]
    if len(ok) < 6:
        verdict = "NOT_DETERMINED"
    elif bl and all(bl):
        verdict = "MPS-GPU-VALID" if (al and np.mean(al) >= 0.8 and cl and np.mean(cl) >= 0.8) else "MPS-GPU-PARTIAL"
    else:
        verdict = "MPS-GPU-INVALID"
    rep = {}                                             # 只報告：rep 0／rep 1
    for (bg, m, r), G in gpu.items():
        if r == 1 and (bg, m, 0) in gpu:
            rep[f"{bg}|{m}"] = G["ttft"] / gpu[(bg, m, 0)]["ttft"] - 1
    sr = {}
    for grid, x in s[(s.ref == "S_real") & (s.kind == 1)].groupby("src_gpu_run"):
        G = next((v for v in gpu.values() if v["run"] == grid), None)
        if G is not None:
            sr[grid] = G["ttft"] / float(x.ttft.mean())
    idle = gpu.get(("idle", "stream", 0))
    out.update(verdict=verdict, n_cells_ok=len(ok), a_frac=float(np.mean(al)) if al else None, b_all=bool(bl and all(bl)),
               c_frac=float(np.mean(cl)) if cl else None, cells=cells, rep_noise=rep, s_real_ratio=sr,
               idle=None if idle is None else {"run": idle["run"], "ttft": idle["ttft"], "sim": sim.get("idle|stream")})
    json.dump(out, open(os.path.join(a.out, "report.json"), "w"), indent=1, ensure_ascii=False, default=float)
    print(json.dumps({k: out[k] for k in ("verdict", "n_cells_ok", "a_frac", "b_all", "c_frac")}, ensure_ascii=False))
    for k, c in cells.items():
        print(f"  {k:28s} {c['status']:12s} " + " ".join(f"{x}={c[x]:.4f}" for x in ("ratio", "gain_gpu", "gain_sim", "d_rho", "d_B")
                                                       if isinstance(c.get(x), float)))
    return 0


# ============================================================================ 自檢
def self_test():
    K, P, cap = setup()
    _, v0, _ = XG.replay(P, POLICY, cap, K, model_view=True)
    v1 = sim_view(K, P, cap, E.IDLE)
    ok = bool(np.array_equal(np.asarray(v0.ttft, float), np.asarray(v1.ttft, float)))
    print(f"[e2e_gpu self-test] idle BgView == x1_gpu.replay(model_view) HINT-PF：{ok}")
    t = np.array([0.0, 1.0, 10.0])
    b = bstat(t, np.array([2.0, 2.0, 0.0]), np.array([0.5, 0.5, 1.0]), 1.0)     # 聯集 [0,3]＝3／10；ρ̄＝0.5 → B＝1 − 0.3×0.5
    ok2 = abs(b["phi"] - 0.3) < 1e-12 and abs(b["B"] - 0.85) < 1e-12
    print(f"[e2e_gpu self-test] bstat：{ok2} {b}")
    return ok and ok2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    sp = ap.add_subparsers(dest="cmd")
    s = sp.add_parser("gpu"); s.add_argument("--bg", required=True, choices=("idle",) + BGS); s.add_argument("--mode", required=True, choices=MODES)
    s.add_argument("--rep", type=int, default=0); s.add_argument("--max-req", type=int); s.add_argument("--out", default=OUT)
    s.add_argument("--csv", default="gpu_requests.csv"); s.add_argument("--runs-csv", default="gpu_runs.csv")
    s = sp.add_parser("sim"); s.add_argument("--out", default=OUT)
    s = sp.add_parser("report"); s.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if a.cmd == "gpu" and a.bg == "idle" and a.mode != "stream":
        ap.error("idle 只配 stream")
    sys.exit({"gpu": cmd_gpu, "sim": cmd_sim, "report": cmd_report}[a.cmd](a))


if __name__ == "__main__":
    main()
