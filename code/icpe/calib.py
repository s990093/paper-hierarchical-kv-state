"""calib.py — ICPE T2（docs/h200/icpe/T2_parallel_calibration.md）：平行爭用校準。

  barrier --wave W --n N [--max-min M]     等同一波（job 名 icpe-cal-<W>-*）的 N 個 job 都在跑；逾時照跑（exit 0，印出來）
  wait-others --job J --wave W [--max-min M]
                                           等節點上沒有「這一波以外」的我們自己的 job（par 用；solo 用 t1g.py wait-clear）
  window --job J --cond solo|par --wave W --comp mb|rp --start T0 --end T1 --short SH --log CSV
                                           這個部件的時間窗寫一列；同時用 sacct 記窗裡不屬於校準的我們自己的 job（查不到 → exit 2）
  analyze                                  登記 §3–§4 → results/icpe_h200/calib/calib_report.json、calib_metrics.csv（要 numpy／pandas）
  self-test                                analyze 的分類與判準在合成資料上的自檢
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t1g  # noqa: E402（_run、_node、_now、_parse 共用）

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAL = os.path.join(REPO, "results", "icpe_h200", "calib")
PREFIX = "icpe-cal-"
MB_PRIMS = ["h2d_perblock", "h2d_batch", "recompute1", "e2e_cake_bf16_cpu", "e2e_cake_int4_cpu", "e2e_cake_int4_b32", "e2e_load_int4_b32"]
COVER = 0.9            # §3：覆蓋率 ≥ 0.9 才算同時跑
NBOOT, SEED = 10000, 20261029
EVENT_PAD_S = 120


def _own_running(node):
    out = t1g._run(["squeue", "-h", "-w", node, "-u", os.environ["USER"], "-t", "R", "-o", "%i %j"])
    return [ln.split(None, 1) for ln in out.split("\n") if ln.strip()]


def cmd_barrier(a):
    node, t_end, n = t1g._node(a), time.time() + a.max_min * 60, 0
    want = f"{PREFIX}{a.wave}-"
    while True:
        try:
            out = t1g._run(["squeue", "-h", "-u", os.environ["USER"], "-o", "%i %j %T"])
        except Exception as e:                       # 查不到 ≠ 沒有（規則 7）
            print(f"[calib] squeue 查不到：{e} → 停", file=sys.stderr)
            return 2
        mine = [ln.split() for ln in out.split("\n") if ln.strip() and ln.split()[1].startswith(want)]
        run = [x for x in mine if x[2] == "RUNNING"]
        if len(run) >= a.n:
            print(f"[calib] {t1g._now().isoformat(timespec='seconds')} 波 {a.wave}：{len(run)}/{a.n} 個在跑（等了 {n} 次）→ 開始", flush=True)
            return 0
        if time.time() > t_end:
            print(f"[calib] 波 {a.wave} 等了 {a.max_min} 分鐘只有 {len(run)}/{a.n} 個在跑：{mine} → 照跑（記下來）", flush=True)
            return 0
        if n % 5 == 0:
            print(f"[calib] 波 {a.wave}：{len(run)}/{a.n} 個在跑，等…", flush=True)
        n += 1
        time.sleep(20)


def cmd_wait_others(a):
    node, t_end, n = t1g._node(a), time.time() + a.max_min * 60, 0
    want = f"{PREFIX}{a.wave}-"
    while True:
        try:
            others = [x for x in _own_running(node) if x[0] != str(a.job) and not (len(x) > 1 and x[1].startswith(want))]
        except Exception as e:
            print(f"[calib] squeue 查不到：{e} → 停", file=sys.stderr)
            return 2
        if not others:
            print(f"[calib] {t1g._now().isoformat(timespec='seconds')} 節點 {node} 上沒有這一波以外的我們的 job（等了 {n} 次）", flush=True)
            return 0
        if time.time() > t_end:
            print(f"[calib] 等了 {a.max_min} 分鐘仍有：{others} → 停", file=sys.stderr)
            return 2
        if n % 10 == 0:
            print(f"[calib] 等這一波以外的 job 結束：{others}", flush=True)
        n += 1
        time.sleep(60)


def _gpu_index():
    try:
        out = t1g._run(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"])
        return ";".join(x.strip().replace(", ", "@") for x in out.strip().split("\n"))
    except Exception as e:     # noqa: BLE001
        return f"NOT_MEASURED({e})"


def cmd_window(a):
    node = t1g._node(a)
    t0, t1_ = datetime.fromisoformat(a.start).astimezone(), datetime.fromisoformat(a.end).astimezone()
    since = (t0 - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%S")
    try:
        out = t1g._run(["sacct", "-u", os.environ["USER"], "-X", "-P", "-n", "-S", since, "-o", "JobID,JobName,NodeList,Start,End,State"])
    except Exception as e:
        print(f"[calib] sacct 查不到：{e} → 停", file=sys.stderr)
        return 2
    other = []
    for ln in out.split("\n"):
        f = ln.split("|")
        if len(f) < 6 or f[0] == str(a.job) or node not in f[2] or f[1].startswith(PREFIX):
            continue
        s, e = t1g._parse(f[3]), t1g._parse(f[4])
        if s is None:
            continue
        e = e or t1g._now()
        if s < t1_ and e > t0:
            other.append(f"{f[0]}:{f[1]}")
    runs = sorted(glob.glob(os.path.join(os.environ.get("TIARA_RUNS", "."), f"*-{a.short}")))
    rid = os.path.basename(runs[-1]) if runs else "NOT_FOUND"
    new = not os.path.exists(a.log)
    os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
    with open(a.log, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["run_id", "ts", "job", "node", "wave", "cond", "comp", "t_start", "t_end", "gpu", "rc", "own_other_jobs"])
        w.writerow([rid, t1g._now().isoformat(timespec="seconds"), a.job, node, a.wave, a.cond, a.comp, a.start, a.end, _gpu_index(), a.rc,
                    ";".join(other)])
    print(f"[calib] {a.comp} run {rid} [{a.start} → {a.end}] 校準以外的我們的 job：{other or '無'}", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ 分析
def classify(W):
    """W：windows 的 DataFrame（每列一個部件窗）。回傳加上 k、cover_max_partial、cls 欄（solo／par／partial／excluded）。"""
    import pandas as pd
    W = W.copy()
    W["t0"] = pd.to_datetime(W.t_start, utc=True)
    W["t1"] = pd.to_datetime(W.t_end, utc=True)
    act = {j: list(zip(g.t0, g.t1)) for j, g in W.groupby("job")}
    ks, parts, cls = [], [], []
    for _, r in W.iterrows():
        L = (r.t1 - r.t0).total_seconds()
        covs = []
        for j, iv in act.items():
            if j == r.job:
                continue
            ov = sum(max(0.0, (min(r.t1, b) - max(r.t0, a)).total_seconds()) for a, b in iv)
            covs.append(ov / L if L > 0 else 0.0)
        k = sum(c >= COVER for c in covs)
        part = [c for c in covs if 0 < c < COVER]
        ks.append(k)
        parts.append(max(part) if part else 0.0)
        if isinstance(r.own_other_jobs, str) and r.own_other_jobs.strip():
            cls.append("excluded")
        elif part:
            cls.append("partial")
        elif k == 0:
            cls.append("solo")
        else:
            cls.append("par")
    W["k"], W["cover_partial_max"], W["cls"] = ks, parts, cls
    return W


def judge(S, P, rng):
    """登記 §4：回傳 dict（d、CI、δ*、σ、verdict）。"""
    import numpy as np
    S, P = np.asarray(S, float), np.asarray(P, float)
    out = dict(n_S=len(S), n_P=len(P))
    if len(S) < 6 or len(P) < 6:
        return dict(out, verdict="NOT_MEASURED")
    ms, mp = float(np.median(S)), float(np.median(P))
    sigma = float(np.std(S, ddof=1) / ms)
    dstar = max(0.05, 2 * sigma)
    d = mp / ms - 1
    bs = np.median(rng.choice(S, (NBOOT, len(S))), axis=1)
    bp = np.median(rng.choice(P, (NBOOT, len(P))), axis=1)
    lo, hi = (float(x) for x in np.percentile(bp / bs - 1, [2.5, 97.5]))
    out.update(med_S=ms, med_P=mp, sigma_rel=sigma, delta_star=dstar, d=d, ci_lo=lo, ci_hi=hi)
    if 2 * sigma > 0.05:
        return dict(out, verdict="NOISY")
    if d <= dstar and hi <= dstar:
        return dict(out, verdict="OK")
    if d > dstar and lo > 0:
        return dict(out, verdict="SLOW")
    return dict(out, verdict="INCONCLUSIVE")


def probe_ok(per_metric, kind):
    v = [x["verdict"] for m, x in per_metric.items() if m.startswith(kind + ":")]
    if any(x == "NOT_MEASURED" for x in v):
        return "NOT_MEASURED"
    judged = [x for x in v if x != "NOISY"]
    if kind == "mb" and len(judged) < 4:
        return "NOT_MEASURED"
    if kind == "rp" and not judged:
        return "NOT_MEASURED"
    return "OK" if all(x == "OK" for x in judged) else "NOT_OK"


def values(W):
    """每個 job、每個指標一個值（登記 §1）。"""
    import pandas as pd
    rows = []
    for _, r in W.iterrows():
        if r.comp == "mb":
            p = os.path.join(CAL, "raw", f"mb_{r.job}.csv")
            if not os.path.exists(p):
                continue
            d = pd.read_csv(p)
            d = d[(d["mode"] == "stream") & (d.bg == "idle") & (pd.to_numeric(d.own_gpu_intruders, errors="coerce").fillna(1) == 0)]
            d = d.assign(t=pd.to_numeric(d.t_med_s, errors="coerce")).dropna(subset=["t"])
            numa = d.gpu_numa.dropna().astype(str).iloc[0] if "gpu_numa" in d and d.gpu_numa.notna().any() else "NOT_MEASURED"
            for prim in MB_PRIMS:
                x = d[d.prim == prim].t
                if len(x):
                    rows.append(dict(job=r.job, comp="mb", metric=f"mb:{prim}", value=float(x.median()), n=len(x), gpu_numa=numa))
        else:
            p = os.path.join(CAL, "raw", f"rp_{r.job}.csv")
            if not os.path.exists(p):
                continue
            d = pd.read_csv(p)
            if (d.contaminated != 0).any():
                continue
            x = d[d.kind == 1].ttft
            if len(x):
                rows.append(dict(job=r.job, comp="rp", metric="rp:ret_ttft_mean", value=float(x.mean()), n=len(x), n_req=len(d)))
    return pd.DataFrame(rows)


def other_user_events(W):
    """只報告：窗前後各 120 s 內，節點上別的使用者的 job 開始／結束次數（只記數目）。"""
    try:
        t0 = (min(W.t0) - timedelta(days=2)).to_pydatetime().astimezone()          # pandas Timestamp（UTC）→ 本地時間給 sacct
        out = t1g._run(["sacct", "-a", "-X", "-P", "-n", "-S", t0.strftime("%Y-%m-%dT%H:%M:%S"), "-o", "JobID,User,NodeList,Start,End"])
    except Exception as e:     # noqa: BLE001
        return {"status": f"NOT_MEASURED（sacct：{e}）"}
    ev = []
    for ln in out.split("\n"):
        f = ln.split("|")
        if len(f) < 5 or f[1] == os.environ["USER"] or "hgx-cn01" not in f[2]:
            continue
        for s in (f[3], f[4]):
            t = t1g._parse(s)
            if t is not None:
                ev.append(t)
    res = {}
    for _, r in W.iterrows():
        a = (r.t0 - timedelta(seconds=EVENT_PAD_S)).to_pydatetime()
        b = (r.t1 + timedelta(seconds=EVENT_PAD_S)).to_pydatetime()
        res[f"{r.job}:{r.comp}"] = sum(a <= t <= b for t in ev)
    return {"status": "ok", "events_per_window": res}


def analyze(W, V, rng, events=None):
    import numpy as np
    W = classify(W)
    V = V.merge(W[["job", "comp", "cls", "k", "gpu"]], on=["job", "comp"], how="left")
    per = {}
    for m, x in V.groupby("metric"):
        S, P = x[x.cls == "solo"].value, x[(x.cls == "par") & (x.k >= 1)].value
        per[m] = judge(S, P, rng)
        per[m]["by_k"] = {int(k): dict(n=len(g), med=float(np.median(g.value)), d=float(np.median(g.value) / np.median(S) - 1) if len(S) else None)
                          for k, g in x[x.cls == "par"].groupby("k")}
    numa = V[V.comp == "mb"].groupby("job").gpu_numa.first().to_dict() if "gpu_numa" in V else {}
    for m, x in V.groupby("metric"):                        # 只報告（§4）：同一張卡上 solo 對 par、各 NUMA 節點
        S = x[x.cls == "solo"]
        by = {}
        for g, y in x.groupby("gpu"):
            s_, p_ = y[y.cls == "solo"].value, y[(y.cls == "par") & (y.k >= 1)].value
            by[str(g)] = dict(n_S=len(s_), n_P=len(p_), med_S=float(np.median(s_)) if len(s_) else None,
                              d=float(np.median(p_) / np.median(s_) - 1) if len(s_) and len(p_) else None)
        per[m]["by_gpu"] = by
        per[m]["numa_of_job"] = {int(j): str(numa.get(j)) for j in x.job}
        del S
    pk = W[(W.cls == "par") & (W.k >= 1)].k
    res = {kind: probe_ok(per, kind) for kind in ("mb", "rp")}
    if res["mb"] == "OK" and res["rp"] == "OK":
        verdict = "CAL-OK"
    elif "OK" in res.values():
        verdict = "CAL-PARTIAL"
    else:
        verdict = "CAL-NO"
    conc = int(1 + np.floor(np.median(pk))) if len(pk) else None
    return W, V, dict(verdict=verdict, probes=res, allowed_concurrency=conc if verdict != "CAL-NO" else 1, metrics=per,
                      n_windows={c: int((W.cls == c).sum()) for c in ("solo", "par", "partial", "excluded")},
                      k_values=sorted(int(k) for k in pk), events=events or {})


def cmd_analyze(a):
    import numpy as np
    import pandas as pd
    W = pd.read_csv(os.path.join(CAL, "calib_windows.csv"))
    bad = W[W.rc.astype(str) != "0"]
    W = W[W.rc.astype(str) == "0"]
    V = values(W)
    rng = np.random.default_rng(SEED)
    Wc = classify(W)
    W2, V2, rep = analyze(W, V, rng, events=other_user_events(Wc))
    rep.update(run_id=t1g.run_id(), ts=t1g.ts(), prereg="docs/h200/icpe/T2_parallel_calibration.md",
               nonzero_rc_windows=bad[["job", "comp", "rc"]].to_dict("records"))
    V2.insert(0, "ts", t1g.ts())
    V2.insert(0, "run_id", t1g.run_id())
    V2.to_csv(os.path.join(CAL, "calib_metrics.csv"), index=False)
    W2.drop(columns=["t0", "t1"]).to_csv(os.path.join(CAL, "calib_windows_classified.csv"), index=False)
    json.dump(rep, open(os.path.join(CAL, "calib_report.json"), "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: rep[k] for k in ("verdict", "probes", "allowed_concurrency", "n_windows", "k_values")}, ensure_ascii=False))
    for m, x in rep["metrics"].items():
        print(f"  {m:28s} {x['verdict']:13s} " + " ".join(f"{k}={x[k]:.4f}" for k in ("d", "ci_lo", "ci_hi", "delta_star", "sigma_rel") if k in x)
              + f" n_S={x['n_S']} n_P={x['n_P']}")
    return 0


def cmd_self_test(a):
    import numpy as np
    import pandas as pd
    base = datetime(2026, 10, 29, 12, 0, tzinfo=datetime.now().astimezone().tzinfo)

    def win(job, wave, cond, comp, s, e):
        return dict(job=job, wave=wave, cond=cond, comp=comp, t_start=(base + timedelta(minutes=s)).isoformat(),
                    t_end=(base + timedelta(minutes=e)).isoformat(), gpu="0", rc=0, own_other_jobs="")
    rows, t = [], 0
    for i in range(6):                                   # solo：一個接一個
        rows += [win(100 + i, "s", "solo", "mb", t, t + 3), win(100 + i, "s", "solo", "rp", t + 3, t + 10)]
        t += 12
    for w in range(3):                                   # par：三個一波，對齊
        for j in range(3):
            rows += [win(200 + 3 * w + j, f"w{w}", "par", "mb", t, t + 3), win(200 + 3 * w + j, f"w{w}", "par", "rp", t + 3, t + 10)]
        t += 12
    rows += [win(300, "x", "par", "mb", t, t + 3), win(300, "x", "par", "rp", t + 3, t + 10),       # 部分重疊
             win(301, "x", "par", "mb", t + 5, t + 8), win(301, "x", "par", "rp", t + 8, t + 15)]
    W = classify(pd.DataFrame(rows))
    assert (W[W.job < 200].cls == "solo").all(), W[W.job < 200]
    assert (W[(W.job >= 200) & (W.job < 300)].cls == "par").all() and (W[(W.job >= 200) & (W.job < 300)].k == 2).all()
    x = W[W.job >= 300].set_index(["job", "comp"])
    assert list(x.cls) == ["solo", "partial", "par", "partial"] and x.loc[(301, "mb"), "k"] == 1, x   # 301 的 mb 整段被 300 的 rp 蓋住
    rng = np.random.default_rng(1)
    noise = lambda n, m, s: m * (1 + s * rng.standard_normal(n))      # noqa: E731
    assert judge(noise(6, 1.0, 0.005), noise(9, 1.0, 0.005), np.random.default_rng(SEED))["verdict"] == "OK"
    assert judge(noise(6, 1.0, 0.005), noise(9, 1.2, 0.005), np.random.default_rng(SEED))["verdict"] == "SLOW"
    assert judge(noise(6, 1.0, 0.05), noise(9, 1.0, 0.05), np.random.default_rng(SEED))["verdict"] == "NOISY"
    assert judge(noise(5, 1.0, 0.005), noise(9, 1.0, 0.005), np.random.default_rng(SEED))["verdict"] == "NOT_MEASURED"
    V = []
    for _, r in W[W.cls != "partial"].iterrows():
        slow = 1.0 if r.cls == "solo" else (1.0 if r.comp == "mb" else 1.2)
        if r.comp == "mb":
            V += [dict(job=r.job, comp="mb", metric=f"mb:{p}", value=float(noise(1, slow, 0.003)[0])) for p in MB_PRIMS]
        else:
            V.append(dict(job=r.job, comp="rp", metric="rp:ret_ttft_mean", value=float(noise(1, slow, 0.003)[0])))
    _, _, rep = analyze(pd.DataFrame(rows), pd.DataFrame(V), np.random.default_rng(SEED))
    assert rep["probes"] == {"mb": "OK", "rp": "NOT_OK"} and rep["verdict"] == "CAL-PARTIAL" and rep["allowed_concurrency"] == 3, rep["probes"]
    print("[calib] self-test PASS")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("barrier"); s.add_argument("--wave", required=True); s.add_argument("--n", type=int, required=True)
    s.add_argument("--max-min", type=float, default=30); s.add_argument("--node")
    s = sp.add_parser("wait-others"); s.add_argument("--job", required=True); s.add_argument("--wave", required=True)
    s.add_argument("--max-min", type=float, default=360); s.add_argument("--node")
    s = sp.add_parser("window"); s.add_argument("--job", required=True); s.add_argument("--cond", required=True, choices=["solo", "par"])
    s.add_argument("--wave", required=True); s.add_argument("--comp", required=True, choices=["mb", "rp"])
    s.add_argument("--start", required=True); s.add_argument("--end", required=True); s.add_argument("--short", required=True)
    s.add_argument("--rc", default="0"); s.add_argument("--log", default=os.path.join(CAL, "calib_windows.csv")); s.add_argument("--node")
    sp.add_parser("analyze")
    sp.add_parser("self-test")
    a = ap.parse_args()
    fn = {"barrier": cmd_barrier, "wait-others": cmd_wait_others, "window": cmd_window, "analyze": cmd_analyze, "self-test": cmd_self_test}
    sys.exit(fn[a.cmd](a))


if __name__ == "__main__":
    main()
