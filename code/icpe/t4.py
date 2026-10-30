#!/usr/bin/env python3
"""t4.py — ICPE T4（docs/h200/icpe/T4_hit_vs_ttft_bridge.md）：同一段 FreeInference，命中率的差距 vs TTFT 的差距。

  python code/icpe/t4.py selftest            §3：重現 T1 的 X1 FreeInference 數字、記帳不變量、O-BEL+NONE 在 8 GiB 的 audit
  python code/icpe/t4.py sweep --workers 4   §1：6 週 × CPU 8／16／32／64／128 GiB × 4 個策略，每個 job 12 種時間設定
  python code/icpe/t4.py analyze             §2、§4：6 週合併的命中率與 TTFT 差距、每個容量的判定、總判定

不改 x1_sim.py：在同一個 process 裡把 O-BEL+NONE 加進 x1_sim.POLICIES，並包一層 View／Engine 記錄每個請求到達時
CPU 層、慢層各有幾個 block（不改它們的計算），之後才建 Pool（fork，子行程繼承）。
環境：P3_CONST=h200、P3_TIER=h200cpu（T1 的常數）；不設 X1_PLANNER（＝Cake）；TIARA_DATASETS 指到有 freeinference_agentic_trace 的目錄。
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
REPO = os.path.dirname(CODE)
sys.path.insert(0, os.path.join(CODE, "p3x"))
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

T1 = os.path.join(REPO, "results", "icpe_h200", "t1")
T4 = os.path.join(REPO, "results", "icpe_h200", "t4")
WEEKS = ["fi:2026-05-17_2026-05-23", "fi:2026-05-24_2026-05-30", "fi:2026-05-31_2026-06-06",
         "fi:2026-06-07_2026-06-13", "fi:2026-06-14_2026-06-20", "fi:2026-06-21_2026-06-27"]
RULE = "session_id"
CAPS = [8, 16, 32, 64, 128]
POLS = ["LRU", "O-BEL", "LRU+NONE", "O-BEL+NONE"]
MAIN = ("ssd", "idle")
HIT_GAP_PP, GAP, SMALL = 5.0, 0.15, 0.05      # §4
SUBSETS = ("ret", "user", "all")


def run_id():
    import t1
    return t1.run_id()


def ts():
    import t1
    return t1.ts()


def code_sha():
    h = hashlib.sha256()
    for p in (__file__, os.path.join(CODE, "p3x", "x1_sim.py"), os.path.join(CODE, "p3x", "x1_trace.py"),
              os.path.join(CODE, "p3x", "x1_fi.py")):
        h.update(open(p, "rb").read())
    return h.hexdigest()[:12]


# ================================================================== 包 x1_sim（不改它的計算）
_XS = None
_K = None
_U = {}


def xs():
    global _XS
    if _XS is None:
        assert os.environ.get("X1_PLANNER", "cake") == "cake", "T4 用原本的 Cake：不要設 X1_PLANNER"
        assert os.environ.get("P3_TIER") == "h200cpu" and os.environ.get("P3_CONST") == "h200", "要 P3_CONST=h200 P3_TIER=h200cpu"
        import x1_sim as XS
        assert "O-BEL+NONE" not in XS.POLICIES
        XS.POLICIES["O-BEL+NONE"] = ("bel", "none", None, None)     # §1：oracle 順序、丟（沒有慢層）

        class RecView(XS.View):
            """同 View；另外記錄每個請求到達時的 L（有存的前綴）與其中在 CPU 層的 block 數。"""

            def start_rec(self, n):
                self.hL = np.full(n, -1, np.int64)
                self.hC = np.zeros(n, np.int64)
                self._r = -1

            def restore(self, lay, blocks, L, n_all, t0, old_n, pf_done):
                r = self._r
                self.hL[r] = L
                self.hC[r] = sum(1 for k in lay if k != 3)
                return XS.View.restore(self, lay, blocks, L, n_all, t0, old_n, pf_done)

        class RecEngine(XS.Engine):
            def step(self, r):
                self.views[0]._r = r
                XS.Engine.step(self, r)

        XS.RecView, XS.RecEngine = RecView, RecEngine
        _XS = XS
    return _XS


def reusable(P):
    """U[r]＝前面連續幾個 block 在 r 之前被讀過（acc[b][0] < r）。"""
    acc, full = P["acc"], P["full"]
    U = np.zeros(len(full), np.int64)
    for r, ks in enumerate(full):
        u = 0
        for b in ks:
            if acc[b][0] < r:
                u += 1
            else:
                break
        U[r] = u
    return U


def prep(week):
    XS = xs()
    P = XS.prep(week, RULE)
    if week not in _U:
        _U[week] = reusable(P)
    return P, _U[week]


def run_one(week, cap, pol, keys, check=False):
    """回傳 (engine, views, 每個請求的 U, L, C)。"""
    XS = xs()
    P, U = prep(week)
    n = len(P["full"])
    views = [XS.RecView(K=_K, slow=keys[0][0], gpu=keys[0][1], mode=keys[0][2], nmax=P["nmax"], nreq=n)]
    views[0].start_rec(n)
    views += XS.make_views(_K, P, keys[1:])
    eng = XS.RecEngine(P, pol, cap * XS.GiB, views, _K, check=check).run()
    L, C = views[0].hL, views[0].hC
    assert (L >= 0).all(), "有請求沒有記到"
    nf = np.array([len(x) for x in P["full"]], np.int64)
    assert (C <= L).all() and (L <= U).all() and (U <= nf).all(), "記帳：C ≤ L ≤ U ≤ len(full) 不成立"
    if XS.POLICIES[pol][1] == "none":
        assert int((L - C).sum()) == 0, "丟的策略不該有慢層的 block"
    return P, eng, views, U, L, C, nf


def hit_fields(P, U, L, C, nf):
    kd = np.array(P["kindc"])
    ret = kd == 1
    m = dict(ret=ret, user=ret & P["user_trigger"], all=np.ones(len(kd), bool))
    out = {}
    for k in SUBSETS:
        x = m[k]
        out.update({f"U_{k}": int(U[x].sum()), f"C_{k}": int(C[x].sum()), f"S_{k}": int((L - C)[x].sum()),
                    f"nfull_{k}": int(nf[x].sum()), f"n_{k}": int(x.sum())})
    return out


def _job(job):
    week, cap, pol = job
    XS = xs()
    t0 = time.time()
    P, eng, views, U, L, C, nf = run_one(week, cap, pol, list(XS.ALL_VIEWS))
    hf = hit_fields(P, U, L, C, nf)
    out = []
    for v in views:
        sm = XS.summarize(P, v, eng)
        out.append(dict(trace=week, rule=RULE, cpu_GiB=cap, policy=pol, slow=v.slow_name, gpu=v.gpu, restore=v.mode,
                        **sm, **hf, elapsed_s=time.time() - t0))
    return out


def _init():
    global _K
    XS = xs()
    if _K is None:
        _K = XS.constants()
    return XS


# ================================================================== selftest
def cmd_selftest(a):
    import pandas as pd
    XS = _init()
    os.makedirs(T4, exist_ok=True)
    res = dict(run_id=run_id(), ts=ts(), code_sha=code_sha(), checks=[])
    ok = True

    def rec(name, passed, detail):
        nonlocal ok
        ok = ok and passed
        res["checks"].append(dict(name=name, passed=bool(passed), detail=detail))
        print(("PASS " if passed else "FAIL ") + name, json.dumps(detail, ensure_ascii=False)[:600], flush=True)

    # 1. 重現 T1
    ref = pd.read_csv(os.path.join(T1, "p12", "x1", "x1_sweep_fi.csv.gz"))
    assert set(ref.run_id) == {"20261028-073351-icpe-t1-p12-x1fi-sweep"}
    ints = ["ret_cpu_loads", "ret_slow_loads", "ret_recompute", "n_demote", "n_drop", "n_ret", "n_req"]
    flts = ["mean_ret", "mean_ret_user"]
    bad, n_cmp = [], 0
    for wk in (WEEKS[0], WEEKS[-1]):
        for pol in ("LRU", "O-BEL", "LRU+NONE"):
            for row in _job((wk, 128, pol)):
                if row["restore"] != "cake" or row["gpu"] != "idle":
                    continue
                r = ref[(ref.trace == wk) & (ref.cpu_GiB == 128) & (ref.policy == pol) & (ref.slow == row["slow"])
                        & (ref.gpu == "idle") & (ref.restore == "cake")]
                assert len(r) == 1, (wk, pol, row["slow"], len(r))
                r = r.iloc[0]
                n_cmp += 1
                for c in ints:
                    if int(r[c]) != int(row[c]):
                        bad.append((wk, pol, row["slow"], c, int(r[c]), int(row[c])))
                for c in flts:
                    if abs(float(r[c]) - float(row[c])) > 1e-6:
                        bad.append((wk, pol, row["slow"], c, float(r[c]), float(row[c])))
    rec("1-reproduce-T1", not bad and n_cmp == 18, dict(compared=n_cmp, mismatches=bad[:20]))
    # 2. 記帳（run_one 裡的斷言；這裡再看一次總數）
    P, eng, views, U, L, C, nf = run_one(WEEKS[0], 8, "LRU+NONE", list(XS.ALL_VIEWS))
    hf = hit_fields(P, U, L, C, nf)
    rec("2-accounting", hf["S_all"] == 0 and hf["n_all"] == len(P["full"]) and hf["U_all"] <= hf["nfull_all"],
        dict(policy="LRU+NONE", cpu_GiB=8, n_all=hf["n_all"], S_all=hf["S_all"], U_all=hf["U_all"], nfull_all=hf["nfull_all"]))
    # 3. O-BEL+NONE、8 GiB、audit 每 2,000 步
    try:
        P, eng, views, U, L, C, nf = run_one(WEEKS[0], 8, "O-BEL+NONE", list(XS.ALL_VIEWS), check=2000)
        rec("3-obel-none-audit", True, dict(week=WEEKS[0], cpu_GiB=8, n_drop=eng.n_drop, n_demote=eng.n_demote))
    except AssertionError as e:
        rec("3-obel-none-audit", False, dict(error=repr(e)))
    res["passed"] = ok
    json.dump(res, open(os.path.join(T4, "t4_selftest.json"), "w"), ensure_ascii=False, indent=1)
    print("selftest", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


# ================================================================== sweep
def cmd_sweep(a):
    import multiprocessing as mp
    _init()
    for wk in WEEKS:
        prep(wk)
    J = [(wk, c, p) for wk in WEEKS for c in CAPS for p in POLS]
    J.sort(key=lambda j: (-WEEKS.index(j[0]), j[1]))      # 大的週、小的容量先跑
    os.makedirs(T4, exist_ok=True)
    out_path = os.path.join(T4, "t4_sweep.csv")
    assert not os.path.exists(out_path), f"{out_path} 已存在（不覆寫）"
    sha = code_sha()
    print(f"jobs={len(J)} workers={a.workers} code_sha={sha}", flush=True)
    t0 = time.time()
    w = None
    with open(out_path, "w", newline="") as fh, mp.get_context("fork").Pool(a.workers) as pool:
        for done, res in enumerate(pool.imap_unordered(_job, J), 1):
            for row in res:
                row = dict(run_id=run_id(), ts=ts(), **row, code_sha=sha)
                if w is None:
                    w = csv.DictWriter(fh, fieldnames=list(row.keys()))
                    w.writeheader()
                w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})
            fh.flush()
            r0 = res[0]
            print(f"[{done}/{len(J)} {time.time()-t0:.0f}s] {r0['trace']} {r0['cpu_GiB']} {r0['policy']} job={r0['elapsed_s']:.1f}s",
                  flush=True)
    print("sweep done", round(time.time() - t0, 1), flush=True)


# ================================================================== analyze
def _rel(base, x):
    return 1.0 - x / base if base > 0 else float("nan")


def cmd_analyze(a):
    import pandas as pd
    d = pd.read_csv(os.path.join(T4, "t4_sweep.csv"))
    assert len(d) == len(WEEKS) * len(CAPS) * len(POLS) * 12, len(d)
    assert d.run_id.nunique() == 1
    sweep_rid = d.run_id.iloc[0]
    # 命中率：每 (週, 容量, 策略) 各一份（12 種時間設定相同）→ 先去重再合併 6 週
    hcols = [f"{p}_{k}" for k in SUBSETS for p in ("U", "C", "S", "nfull", "n")]
    h1 = d.drop_duplicates(["trace", "cpu_GiB", "policy"])
    chk = d.groupby(["trace", "cpu_GiB", "policy"])[hcols].nunique().max().max()
    assert chk == 1, "同一個 (週, 容量, 策略) 的命中記帳在不同時間設定之間不一樣"
    H = h1.groupby(["cpu_GiB", "policy"])[hcols].sum()
    T = d.groupby(["cpu_GiB", "policy", "slow", "gpu", "restore"])[["sum_ret", "n_ret", "sum_ret_user", "n_ret_user"]].sum()
    T["ttft_ret"] = T.sum_ret / T.n_ret
    T["ttft_user"] = T.sum_ret_user / T.n_ret_user

    def hr(c, p, k, tier=False):
        x = H.loc[(c, p)]
        num = x[f"C_{k}"] + (x[f"S_{k}"] if tier else 0)
        return 100.0 * num / x[f"U_{k}"] if x[f"U_{k}"] > 0 else float("nan")

    def tt(c, p, s, g, m, k):
        return float(T.loc[(c, p, s, g, m), "ttft_ret" if k == "ret" else "ttft_user"])

    cells = []
    for c in CAPS:
        for s in ("ssd", "b32", "nfs"):
            for g in ("idle", "busy"):
                for k in ("ret", "user"):
                    hl, ho = hr(c, "LRU+NONE", k), hr(c, "O-BEL+NONE", k)
                    row = dict(cpu_GiB=c, slow=s, gpu=g, subset=k,
                               hr_lru_none=hl, hr_obel_none=ho, G_hit_pp=ho - hl,
                               hr_cpu_lru=hr(c, "LRU", k), hr_cpu_obel=hr(c, "O-BEL", k),
                               hr_tier_lru=hr(c, "LRU", k, True), hr_tier_obel=hr(c, "O-BEL", k, True),
                               ttft_lru_none_load=tt(c, "LRU+NONE", s, g, "load", k), ttft_obel_none_load=tt(c, "O-BEL+NONE", s, g, "load", k),
                               ttft_lru_load=tt(c, "LRU", s, g, "load", k), ttft_obel_load=tt(c, "O-BEL", s, g, "load", k),
                               ttft_lru_cake=tt(c, "LRU", s, g, "cake", k), ttft_obel_cake=tt(c, "O-BEL", s, g, "cake", k),
                               ttft_lru_none_cake=tt(c, "LRU+NONE", s, g, "cake", k), ttft_obel_none_cake=tt(c, "O-BEL+NONE", s, g, "cake", k))
                    row["G_rec"] = _rel(row["ttft_lru_none_load"], row["ttft_obel_none_load"])
                    row["G_load"] = _rel(row["ttft_lru_load"], row["ttft_obel_load"])
                    row["G_cake"] = _rel(row["ttft_lru_cake"], row["ttft_obel_cake"])
                    row["G_rec_cake"] = _rel(row["ttft_lru_none_cake"], row["ttft_obel_none_cake"])
                    gap = row["G_hit_pp"] >= HIT_GAP_PP and row["G_rec"] >= GAP
                    if not gap:
                        cls = "NO-GAP"
                    elif row["G_cake"] < SMALL:
                        cls = "ABSORBED"
                    elif row["G_cake"] < GAP:
                        cls = "PARTLY"
                    else:
                        cls = "NOT-ABSORBED"
                    row["class"] = cls
                    cells.append(row)
    C = pd.DataFrame(cells)
    C.insert(0, "run_id", run_id())
    C.insert(1, "ts", ts())
    C["sweep_run_id"] = sweep_rid

    def overall(sub):
        cl = list(sub["class"])
        g = [x for x in cl if x != "NO-GAP"]
        if not g:
            return "BRIDGE-NO-GAP"
        if all(x == "ABSORBED" for x in g):
            return "BRIDGE-ABSORBED"
        return "BRIDGE-MIXED"

    main = C[(C.slow == MAIN[0]) & (C.gpu == MAIN[1]) & (C.subset == "ret")].sort_values("cpu_GiB")
    verdict = overall(main)
    robust = {f"{s}-{g}-{k}": overall(C[(C.slow == s) & (C.gpu == g) & (C.subset == k)])
              for s in ("ssd", "b32", "nfs") for g in ("idle", "busy") for k in ("ret", "user")}
    # 每週（只報告）：主設定、回來的請求
    wk = []
    for (tr, c), x in d[(d.slow == MAIN[0]) & (d.gpu == MAIN[1])].groupby(["trace", "cpu_GiB"]):
        def m(p, r):
            y = x[(x.policy == p) & (x.restore == r)].iloc[0]
            return float(y.mean_ret), y
        a_, ya = m("LRU+NONE", "load")
        b_, yb = m("O-BEL+NONE", "load")
        lc, _ = m("LRU", "cake")
        oc, _ = m("O-BEL", "cake")
        wk.append(dict(trace=tr, cpu_GiB=c, G_hit_pp=100.0 * (yb.C_ret / yb.U_ret - ya.C_ret / ya.U_ret) if ya.U_ret else float("nan"),
                       G_rec=_rel(a_, b_), G_cake=_rel(lc, oc), n_ret=int(ya.n_ret)))
    W = pd.DataFrame(wk)
    W.insert(0, "run_id", run_id())
    C.to_csv(os.path.join(T4, "t4_cells.csv"), index=False, float_format="%.6g")
    W.to_csv(os.path.join(T4, "t4_weeks.csv"), index=False, float_format="%.6g")
    cols = ["cpu_GiB", "G_hit_pp", "G_rec", "G_load", "G_cake", "G_rec_cake", "hr_lru_none", "hr_obel_none", "class"]
    out = dict(run_id=run_id(), ts=ts(), sweep_run_id=sweep_rid, code_sha=code_sha(),
               rule=dict(hit_gap_pp=HIT_GAP_PP, gap=GAP, small=SMALL, main=dict(slow=MAIN[0], gpu=MAIN[1], subset="ret")),
               verdict=verdict, main=main[cols].to_dict(orient="records"), robustness=robust,
               negative_gap_cells=int(((C.G_hit_pp < 0) | (C.G_rec < 0) | (C.G_cake < 0)).sum()))
    json.dump(out, open(os.path.join(T4, "t4_verdict.json"), "w"), ensure_ascii=False, indent=1)
    pd.set_option("display.width", 200)
    print(main[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("verdict", verdict)
    print("robustness", json.dumps(robust))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("selftest")
    s = sp.add_parser("sweep")
    s.add_argument("--workers", type=int, default=4)
    sp.add_parser("analyze")
    a = ap.parse_args()
    dict(selftest=cmd_selftest, sweep=cmd_sweep, analyze=cmd_analyze)[a.cmd](a)


if __name__ == "__main__":
    main()
