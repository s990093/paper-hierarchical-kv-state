#!/usr/bin/env python3
"""kvs_f0p.py — F0P：共享前綴的優先權反轉（F0 加兩個優先級；CPU 事件模擬，〔模擬〕）。

登記：docs/kv_sharing_hiding_20261026/runs/F0P_priority.md（模型、策略、網格、判準都在那裡；這支只照做）。
常數、f(i)、t_new、t_L 都和 F0 相同：直接 import kvs_f0（不改它）。

  python3 code/s3/kvs_f0p.py --self-test                    # 自測 (a)(b)(c)＋反轉的玩具例＋判定函式
  python3 code/s3/kvs_f0p.py run --slice decisive --jobs 8  # 決定性實驗的 12 格
  python3 code/s3/kvs_f0p.py run --slice full --jobs 8      # 全網格（決定性實驗沒 STOP 才跑）
  python3 code/s3/kvs_f0p.py report                         # 判定表

純 Python（不用 numpy），任何 CPU 都能跑。輸出都有 run_id、ts。
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import heapq
import json
import math
import os
import random
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
REPO = os.path.dirname(CODE)
sys.path.insert(0, HERE)
sys.path.insert(0, CODE)

import kvs_f0 as F0                                                    # noqa: E402
from kvs_f0 import LENS, PANELS, TIERS, f_of, pct, t_load_of, t_new_of, ts   # noqa: E402

OUT = os.path.join(REPO, "results", "s3_h200", "kvs_f0p")
NS = (4, 16, 64)
SPREADS = (0.0, 0.5, 2.0)
RHOS = (0.0, 0.5, 0.8)
RHOB = (0.0, 0.4, 0.8)
HS = (0.1, 0.3)
SEEDS = tuple(range(40))
POLICIES = ("A4", "I4", "D4", "C4", "A5", "I5")
INHERIT = {"I4", "I5"}
HELP = {"A5", "I5"}
ESCAPE = {"D4": "load", "C4": "rec"}
HI, LO = 0, 1                       # 數字小＝優先
MIN_N = 20                          # hsh／hexp 樣本數 < 20 → NOT_MEASURED（登記 §1.4）
DECISIVE = dict(panel=("h200", "llama"), tiers=TIERS, lens=("8K", "128K"), rhobs=(0.4, 0.8), N=16, spread=0.5, rho_o=0.5, h=0.3)
FIELDS = ["run_id", "ts", "slice", "platform", "model", "tier", "L", "n_blocks", "N", "spread", "rho_o", "rho_b", "h", "policy",
          "n_seeds", "lam_o", "lam_b", "util_c", "util_l", "T_w_s", "W_load_s", "t_new_s", "c_o_s", "t_load_block_s",
          "n_hsh", "hsh_p95", "hsh_mean", "n_hexp", "hexp_p95", "n_h", "h_p95", "n_l", "l_p95", "n_all", "all_p95",
          "n_rec_blocks", "n_load_blocks"]
_CONSTS = {}


def run_id() -> str:
    return os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-kvs-f0p-adhoc"


def run_dir() -> str:
    d = os.environ.get("RUN_DIR") or os.path.join(os.environ.get("TIARA_RUNS", os.path.join(REPO, "_big", "runs")), run_id())
    os.makedirs(d, exist_ok=True)
    return d


def consts(plat: str, model: str) -> dict:
    if (plat, model) not in _CONSTS:
        _CONSTS[(plat, model)] = F0.consts(plat, model)
    return _CONSTS[(plat, model)]


# ============================================================================ 到達（登記 §1.2）
def rates(cell: dict, C: dict) -> dict:
    n, tL = cell["n"], t_load_of(C, cell["tier"])
    tn, c_o = t_new_of(C, n), C["c_o"]
    lam_b = cell["rho_b"] / max(n * tL, tn) if cell["rho_b"] > 0 else 0.0
    lam_o = max(0.0, cell["rho_o"] - lam_b * tn) / c_o
    W = n * tL
    return dict(lam_b=lam_b, lam_o=lam_o, util_c=lam_o * c_o + lam_b * tn, util_l=lam_b * W, Tw=20.0 * max(W, tn, c_o),
                W=W, tn=tn, tL=tL, c_o=c_o)


def _rng(key: str) -> random.Random:
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))


def arrivals(cell: dict, C: dict, seed: int) -> tuple[list, list, list, float]:
    """共同亂數：時間不看 h（同格不同 h 的到達相同），優先級用另一條亂數流（h 大的 H 集合包含 h 小的）。
    回傳（共享請求、別人的工作、背景載入）各為 [(到達時間, 優先級)]，以及 H。"""
    key = f"f0p|{C['platform']}|{C['model']}|{cell['tier']}|{cell['n']}|{cell['N']}|{cell['spread']}|{cell['rho_o']}|{cell['rho_b']}|{seed}"
    rt, rc = _rng(key + "|t"), _rng(key + "|c")
    R = rates(cell, C)
    gt = [0.0] + sorted(rt.uniform(0.0, cell["spread"] * R["W"]) for _ in range(cell["N"] - 1))
    H = gt[-1] + R["W"] + 5.0

    def pois(lam):
        out = []
        if lam <= 0:
            return out
        t = -R["Tw"] + rt.expovariate(lam)
        while t <= H:
            out.append(t)
            t += rt.expovariate(lam)
        return out

    ot, bt = pois(R["lam_o"]), pois(R["lam_b"])
    cls = lambda: HI if rc.random() < cell["h"] else LO       # noqa: E731
    return [(t, cls()) for t in gt], [(t, cls()) for t in ot], [(t, cls()) for t in bt], H


# ============================================================================ 模擬
class Restore:
    """一份前綴的還原。kind：shared（共享 P）、pload（D4 的 H 私人載入）、prec（C4 的 H 私人重算）、bload（背景載入）。"""
    __slots__ = ("n", "kind", "p", "q", "inflight", "done_t", "waiters", "n_rec", "n_load", "owner", "cls", "lseq")

    def __init__(self, n: int, kind: str, owner: int, cls: int, lseq: int):
        self.n, self.kind, self.owner, self.cls, self.lseq = n, kind, owner, cls, lseq
        self.p, self.q, self.inflight = 0, n - 1, 0
        self.done_t = None
        self.waiters = []
        self.n_rec = self.n_load = 0


def simulate(cell: dict, C: dict, policy: str, g: list, o: list, b: list, trace: list | None = None) -> dict:
    """一個 episode。g／o／b＝[(到達, 優先級)]。trace（list）給自測：(線, 種類, 標籤, 塊, 開始, 結束)。"""
    assert policy in POLICIES, policy
    n, tL = cell["n"], t_load_of(C, cell["tier"])
    tnew, c_o = t_new_of(C, n), C["c_o"]
    helper, inherit, escape = policy in HELP, policy in INHERIT, ESCAPE.get(policy)
    gcls, ocls, bcls = [c for _, c in g], [c for _, c in o], [c for _, c in b]
    ev, seq, lseq = [], [0], [0]

    def push(t, kind, x):
        seq[0] += 1
        heapq.heappush(ev, (t, seq[0], kind, x))

    for k, (a, _) in enumerate(g):                # 推入順序同 F0：共享請求、別人，再加背景載入
        push(a, "ag", k)
    for k, (a, _) in enumerate(o):
        push(a, "ao", k)
    for k, (a, _) in enumerate(b):
        push(a, "ab", k)
    fin_g, fin_o, fin_b = [None] * len(g), [None] * len(o), [None] * len(b)
    cq = []                    # 計算線：(優先級, bundle, 準備好的時間, seq, 項目)
    loadq = []                 # 載入線：還沒載完的 Restore
    busy = {"c": None, "l": None}
    R = [None]                 # 共享 P 的還原
    stats = dict(rec=0, load=0, restores=[])

    def item_cls(item):
        k = item[0]
        if k == "new":
            return gcls[item[1]]
        if k == "other":
            return ocls[item[1]]
        if k == "bnew":
            return bcls[item[1]]
        return item[1].cls                                                   # prec

    def enq(t, item):
        seq[0] += 1
        heapq.heappush(cq, (item_cls(item), 0 if item[0] == "new" else 1, t, seq[0], item))

    def new_restore(kind, owner, cls, waiter):
        lseq[0] += 1
        r = Restore(n, kind, owner, cls, lseq[0])
        r.waiters.append(waiter)
        stats["restores"].append(r)
        return r

    def complete(r: Restore, t):
        if r.p > r.q and r.inflight == 0 and r.done_t is None:
            r.done_t = t
            for it in r.waiters:
                enq(t, it)
            r.waiters = []

    def start_load(t):
        loadq[:] = [r for r in loadq if r.p <= r.q]
        if not loadq:
            busy["l"] = None
            return
        r = min(loadq, key=lambda x: (x.cls, x.lseq))
        i = r.q
        r.q -= 1
        r.inflight += 1
        r.n_load += 1
        stats["load"] += 1
        busy["l"] = ("load", r, i)
        push(t + tL, "ld", None)
        if trace is not None:
            trace.append(("l", r.kind, r.lseq, i, t, t + tL))

    def start_compute(t):
        r = R[0]
        if helper and r is not None and r.done_t is None and r.p <= r.q and (not cq or r.cls <= cq[0][0]):
            i = r.p
            r.p += 1
            r.inflight += 1
            r.n_rec += 1
            stats["rec"] += 1
            dur = f_of(C, i)
            busy["c"] = ("rec", r, i)
            push(t + dur, "cd", None)
            if trace is not None:
                trace.append(("c", "help", r.lseq, i, t, t + dur))
            return
        if not cq:
            busy["c"] = None
            return
        item = heapq.heappop(cq)[-1]
        if item[0] == "prec":
            pr = item[1]
            i = pr.p
            pr.p += 1
            pr.inflight += 1
            pr.n_rec += 1
            stats["rec"] += 1
            dur = f_of(C, i)
            busy["c"] = ("prec", pr, i)
            tag = (pr.lseq, i)
        else:
            dur = tnew if item[0] in ("new", "bnew") else c_o
            busy["c"] = item
            tag = (item[1], -1)
        push(t + dur, "cd", None)
        if trace is not None:
            trace.append(("c", item[0], tag[0], tag[1], t, t + dur))

    def kick(t):
        if busy["l"] is None:
            start_load(t)
        if busy["c"] is None:
            start_compute(t)

    def g_arrive(k, t):
        r, c = R[0], gcls[k]
        if r is None:
            R[0] = new_restore("shared", k, c, ("new", k))
            loadq.append(R[0])
            return
        if r.done_t is not None:
            enq(t, ("new", k))
            return
        if escape and c == HI and r.cls == LO:                               # H 跟隨者遇到 L 的在途 R
            if escape == "load":
                loadq.append(new_restore("pload", k, HI, ("new", k)))
            else:
                enq(t, ("prec", new_restore("prec", k, HI, ("new", k))))
            return
        r.waiters.append(("new", k))
        if inherit and c < r.cls:
            r.cls = c

    while ev:
        t, _, kind, x = heapq.heappop(ev)
        if kind == "ag":
            g_arrive(x, t)
        elif kind == "ao":
            enq(t, ("other", x))
        elif kind == "ab":
            loadq.append(new_restore("bload", -1 - x, bcls[x], ("bnew", x)))
        elif kind == "ld":
            _, r, i = busy["l"]
            busy["l"] = None
            r.inflight -= 1
            complete(r, t)
        elif kind == "cd":
            bb = busy["c"]
            busy["c"] = None
            if bb[0] == "rec":
                bb[1].inflight -= 1
                complete(bb[1], t)
            elif bb[0] == "prec":
                pr = bb[1]
                pr.inflight -= 1
                if pr.p <= pr.q:
                    enq(t, ("prec", pr))
                else:
                    complete(pr, t)
            elif bb[0] == "new":
                fin_g[bb[1]] = t
            elif bb[0] == "bnew":
                fin_b[bb[1]] = t
            else:
                fin_o[bb[1]] = t
        kick(t)
    if any(x is None for x in fin_g + fin_o + fin_b):
        raise RuntimeError(f"有請求沒做完（{policy}，{cell}）")
    return dict(ttft_g=[f - a for f, (a, _) in zip(fin_g, g)], ttft_o=[f - a for f, (a, _) in zip(fin_o, o)],
                ttft_b=[f - a for f, (a, _) in zip(fin_b, b)], n_rec=stats["rec"], n_load=stats["load"], restores=stats["restores"])


# ============================================================================ 一格
def _p95(v: list, min_n: int = 1):
    return pct(v, 0.95) if len(v) >= max(min_n, 1) else "NOT_MEASURED"


def run_cell(args) -> tuple:
    """一格 × 全部策略 × 40 個 seed → 每個策略一列（P95 是所有 seed 的請求合起來算）。"""
    slice_, plat, model, cell, keep_raw = args
    C = consts(plat, model)
    Rt = rates(cell, C)
    arr = [arrivals(cell, C, s) for s in SEEDS]
    rows, raw = [], {}
    for pol in POLICIES:
        hsh, hexp, hh, ll, nrec, nload = [], [], [], [], 0, 0
        rq = []
        for g, o, b, _ in arr:
            r = simulate(cell, C, pol, g, o, b)
            owner_lo = g[0][1] == LO
            for k, ((a, c), x) in enumerate(zip(g, r["ttft_g"])):
                (hh if c == HI else ll).append(x)
                if c == HI:
                    hsh.append(x)
                    if k > 0 and owner_lo:
                        hexp.append(x)
                if keep_raw:
                    rq.append(("g", k, c, round(a, 6), round(x, 6)))
            for (a, c), x in list(zip(o, r["ttft_o"])) + list(zip(b, r["ttft_b"])):
                if a >= 0.0:                                                   # 只量 [0, H] 內到達的
                    (hh if c == HI else ll).append(x)
            nrec += r["n_rec"]
            nload += r["n_load"]
        allv = hh + ll
        rows.append(dict(slice=slice_, platform=plat, model=model, tier=cell["tier"], L=cell["L"], n_blocks=cell["n"], N=cell["N"],
                         spread=cell["spread"], rho_o=cell["rho_o"], rho_b=cell["rho_b"], h=cell["h"], policy=pol, n_seeds=len(SEEDS),
                         lam_o=Rt["lam_o"], lam_b=Rt["lam_b"], util_c=Rt["util_c"], util_l=Rt["util_l"], T_w_s=Rt["Tw"],
                         W_load_s=Rt["W"], t_new_s=Rt["tn"], c_o_s=Rt["c_o"], t_load_block_s=Rt["tL"],
                         n_hsh=len(hsh), hsh_p95=_p95(hsh, MIN_N), hsh_mean=sum(hsh) / len(hsh) if hsh else "NOT_MEASURED",
                         n_hexp=len(hexp), hexp_p95=_p95(hexp, MIN_N), n_h=len(hh), h_p95=_p95(hh), n_l=len(ll), l_p95=_p95(ll),
                         n_all=len(allv), all_p95=_p95(allv), n_rec_blocks=nrec, n_load_blocks=nload))
        if keep_raw:
            raw[pol] = rq
    return rows, (cell, raw) if keep_raw else None


def cells_of(slice_: str):
    if slice_ == "decisive":
        d = DECISIVE
        for tier in d["tiers"]:
            for L in d["lens"]:
                for rb in d["rhobs"]:
                    yield d["panel"], dict(tier=tier, L=L, n=LENS[L], N=d["N"], spread=d["spread"], rho_o=d["rho_o"], rho_b=rb, h=d["h"])
        return
    for panel in PANELS:
        for tier in TIERS:
            for L, n in LENS.items():
                for N in NS:
                    for sp in SPREADS:
                        for rho in RHOS:
                            for rb in RHOB:
                                for h in HS:
                                    yield panel, dict(tier=tier, L=L, n=n, N=N, spread=sp, rho_o=rho, rho_b=rb, h=h)


# ============================================================================ 指令
def cmd_run(a):
    rid, rd = run_id(), run_dir()
    jobs = [(a.slice, p[0], p[1], cell, a.slice == "decisive") for p, cell in cells_of(a.slice)]
    jobs.sort(key=lambda j: -j[3]["n"] * (1 + j[3]["rho_o"] + j[3]["rho_b"]))      # 大的先送，平行比較平均
    print(f"[f0p] {a.slice}：{len(jobs)} 格，每格 {len(POLICIES)} 個策略 × {len(SEEDS)} 個 seed → run {rid}", flush=True)
    if a.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(a.jobs) as ex:
            res = list(ex.map(run_cell, jobs, chunksize=1))
    else:
        res = [run_cell(j) for j in jobs]
    os.makedirs(a.out, exist_ok=True)
    by = {}
    for rows, _ in res:
        for r in rows:
            by.setdefault((r["platform"], r["model"]), []).append(r)
    for (plat, model), rows in by.items():                                          # 列的順序照格的參數排
        rows.sort(key=lambda r: (TIERS.index(r["tier"]), r["n_blocks"], r["N"], r["spread"], r["rho_o"], r["rho_b"], r["h"],
                                 POLICIES.index(r["policy"])))
        p = os.path.join(a.out, f"f0p_{a.slice}_{plat}_{model}.csv" + (".gz" if a.slice == "full" else ""))
        with (gzip.open(p, "wt", newline="") if p.endswith(".gz") else open(p, "w", newline="")) as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            for r in rows:
                r.update(run_id=rid, ts=ts())
                w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in ((k, r.get(k, "")) for k in FIELDS)})
        print(f"[f0p] → {p}（{len(rows)} 列）", flush=True)
    raws = [x for _, x in res if x]
    if raws:
        p = os.path.join(rd, f"f0p_{a.slice}_requests.json.gz")
        with gzip.open(p, "wt") as fh:
            json.dump([dict(cell=c, requests=r, fields=["role", "idx", "cls", "arrival", "ttft"]) for c, r in raws], fh)
        print(f"[f0p] 共享請求的明細 → {p}", flush=True)


def _read(path: str) -> list:
    rows = list(csv.DictReader(gzip.open(path, "rt") if path.endswith(".gz") else open(path)))
    for r in rows:
        for k in ("hsh_p95", "hexp_p95", "h_p95", "l_p95", "all_p95"):
            r[k] = float(r[k]) if r[k] not in ("NOT_MEASURED", "") else None
    return rows


def _cellkey(r):
    return (r["tier"], r["L"], r["N"], r["spread"], r["rho_o"], r["rho_b"], r["h"])


def _cells(rows: list) -> dict:
    cells = {}
    for r in rows:
        cells.setdefault(_cellkey(r), {})[r["policy"]] = r
    return cells


def _q(d: dict, num: str, den: str, col: str = "hsh_p95"):
    a, b = d[num][col], d[den][col]
    return None if a is None or b is None else a / b - 1.0


def _metrics(d: dict) -> dict:
    inv4, inv5 = _q(d, "A4", "I4"), _q(d, "A5", "I5")
    esc = None
    if d["I4"]["hsh_p95"] is not None and d["D4"]["hsh_p95"] is not None and d["C4"]["hsh_p95"] is not None:
        esc = 1.0 - min(d["D4"]["hsh_p95"], d["C4"]["hsh_p95"]) / d["I4"]["hsh_p95"]
    return dict(inv4=inv4, inv5=inv5, esc=esc, inv4_hexp=_q(d, "A4", "I4", "hexp_p95"), side_l=_q(d, "I4", "A4", "l_p95"),
                side_h=_q(d, "I4", "A4", "h_p95"))


def verdict_decisive(rows: list) -> dict:
    per, invalid = {}, 0
    for k, d in _cells(rows).items():
        m = _metrics(d)
        if m["inv4"] is None or m["inv5"] is None:
            invalid += 1
        per["|".join(k)] = m
    valid = [v for v in per.values() if v["inv4"] is not None and v["inv5"] is not None]
    stop = bool(valid) and all(v["inv4"] < 0.05 and v["inv5"] < 0.05 for v in valid)
    return dict(cells=per, n_cells=len(per), n_invalid=invalid, verdict="STOP-P" if stop else "CONTINUE")


def _summ(rows: list) -> dict:
    per = [dict(key="|".join(k), **_metrics(d)) for k, d in _cells(rows).items()]
    v4 = [p for p in per if p["inv4"] is not None]
    ve = [p for p in per if p["esc"] is not None]
    v5 = [p for p in per if p["inv5"] is not None]
    nv = len(v4)
    out = dict(n_cells=len(per), n_valid=nv, n_invalid=len(per) - nv,
               frac_inv4_gt15=sum(p["inv4"] > 0.15 for p in v4) / nv if nv else None,
               frac_esc_gt5=sum(p["esc"] > 0.05 for p in ve) / len(ve) if ve else None,
               max_inv4=max((p["inv4"] for p in v4), default=None), max_inv5=max((p["inv5"] for p in v5), default=None),
               median_inv4=sorted(p["inv4"] for p in v4)[nv // 2] if nv else None,
               all_below_5=bool(v4) and all(p["inv4"] < 0.05 for p in v4) and all(p["inv5"] < 0.05 for p in v5),
               max_esc=max((p["esc"] for p in ve), default=None),
               side_l_gt5=sum(1 for p in per if p["side_l"] is not None and p["side_l"] > 0.05),
               max_inv4_rhob0=max((p["inv4"] for p in v4 if p["key"].split("|")[5] in ("0.0", "0")), default=None))
    out["E"] = out["frac_inv4_gt15"] is not None and out["frac_inv4_gt15"] >= 0.10
    out["X"] = out["frac_esc_gt5"] is not None and out["frac_esc_gt5"] >= 0.10
    return out


def verdict_full(rows_by_plat: dict) -> dict:
    out = {plat: _summ(rows) for plat, rows in rows_by_plat.items()}
    ps = list(out.values())
    if ps and all(p["E"] and p["X"] for p in ps):
        v = "GO-P"
    elif ps and all(p["E"] for p in ps) and all(not p["X"] for p in ps):
        v = "INHERIT-ONLY"
    elif ps and all(p["all_below_5"] for p in ps):
        v = "STOP-P"
    else:
        v = "ASK"
    return dict(platforms=out, verdict=v)


def cmd_report(a):
    rep = dict(run_id=run_id(), ts=ts())
    dp = os.path.join(a.out, "f0p_decisive_h200_llama.csv")
    if os.path.exists(dp):
        rep["decisive"] = verdict_decisive(_read(dp))
    full = {plat: os.path.join(a.out, f"f0p_full_{plat}_llama.csv.gz") for plat in ("h200", "mi300x")}
    if all(os.path.exists(p) for p in full.values()):
        rep["full_llama"] = verdict_full({k: _read(p) for k, p in full.items()})
        qw = {plat: os.path.join(a.out, f"f0p_full_{plat}_qwen.csv.gz") for plat in ("h200", "mi300x")}
        if all(os.path.exists(p) for p in qw.values()):
            rep["full_qwen_replicate"] = verdict_full({k: _read(p) for k, p in qw.items()})
    json.dump(rep, open(os.path.join(a.out, "f0p_verdict.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: (v if k != "decisive" else {kk: vv for kk, vv in v.items() if kk != "cells"}) for k, v in rep.items()},
                     ensure_ascii=False, indent=1))
    if "decisive" in rep:
        print("決定性實驗每格（inv4、inv5、esc、inv4_hexp、副作用 L、副作用 H）：")
        for k, m in rep["decisive"]["cells"].items():
            print(" ", k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in m.items()})


# ============================================================================ 自測（只驗邏輯，不產生結果）
def self_test() -> bool:
    ok = True

    def check(name, cond):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name)
        ok &= bool(cond)

    def same(x, y):
        return len(x) == len(y) and all(abs(p - q) < 1e-9 for p, q in zip(x, y))

    # (a) 單一優先級、ρ_b＝0、不暖機、F0 的到達 → 和 F0 的 P4／P5 逐筆相同
    base = {"A4": "P4", "I4": "P4", "D4": "P4", "C4": "P4", "A5": "P5", "I5": "P5"}
    for plat, model in (("h200", "llama"), ("mi300x", "qwen")):
        C = consts(plat, model)
        for tier in TIERS:
            for L, n in LENS.items():
                for N, sp, rho in ((6, 0.5, 0.5), (16, 0.0, 0.8), (4, 2.0, 0.0)):
                    cell0 = dict(tier=tier, L=L, n=n, N=N, spread=sp, rho_o=rho)
                    good = True
                    for s in (0, 1):
                        gf, of, _ = F0.arrivals(cell0, C, s)
                        ref = {p: F0.simulate(cell0, C, p, gf, of) for p in ("P4", "P5")}
                        for c in (LO, HI):
                            for pol in POLICIES:
                                r = simulate(cell0, C, pol, [(t, c) for t in gf], [(t, c) for t in of], [])
                                good &= same(r["ttft_g"], ref[base[pol]]["ttft_g"]) and same(r["ttft_o"], ref[base[pol]]["ttft_o"])
                    check(f"(a) {plat}/{model} {tier} {L} N={N}：單一優先級時 6 個策略＝F0 的 P4／P5（逐筆）", good)
    C = consts("h200", "llama")
    # (b) ρ_b＝0（有暖機、混合優先級）→ A4 與 I4 逐筆相同（載入線上只有 R）
    for tier in TIERS:
        for L, n in LENS.items():
            cell = dict(tier=tier, L=L, n=n, N=16, spread=0.5, rho_o=0.8, rho_b=0.0, h=0.3)
            good = True
            for s in (0, 1, 2):
                g, o, b, _ = arrivals(cell, C, s)
                good &= not b and same(simulate(cell, C, "A4", g, o, b)["ttft_g"], simulate(cell, C, "I4", g, o, b)["ttft_g"])
            check(f"(b) {tier} {L}：ρ_b＝0 時 A4＝I4（inv4 恰好 0）", good)
    # (c) 守恆、一條線一次一件事、到達前不開始（有背景載入、混合優先級）
    for tier, L, rb in (("nvme", "8K", 0.8), ("nfs", "8K", 0.4), ("cpu", "32K", 0.8)):
        cell = dict(tier=tier, L=L, n=LENS[L], N=16, spread=0.5, rho_o=0.5, rho_b=rb, h=0.3)
        g, o, b, H = arrivals(cell, C, 4)
        check(f"(c) {tier} {L}：到達有暖機（別人或背景載入有負的到達）、優先級兩種都有",
              any(t < 0 for t, _ in o + b) and {c for _, c in g + o + b} == {HI, LO})
        for pol in POLICIES:
            tr = []
            r = simulate(cell, C, pol, g, o, b, trace=tr)
            blocks = {}
            for line, kind, tag, i, s, e in tr:
                if line == "l" or kind in ("help", "prec"):
                    blocks.setdefault(tag, []).append(i)
            cons = len(blocks) == len(r["restores"]) and all(sorted(v) == list(range(cell["n"])) for v in blocks.values())
            check(f"(c) {pol} {tier} {L}：每份還原的每一塊恰好做一次（{len(r['restores'])} 份）", cons)
            sc = sorted((s, e) for line, *_, s, e in tr if line == "c")
            sl = sorted((s, e) for line, *_, s, e in tr if line == "l")
            check(f"(c) {pol} {tier} {L}：每條線一次只做一件事",
                  all(sc[k + 1][0] >= sc[k][1] - 1e-12 for k in range(len(sc) - 1)) and
                  all(sl[k + 1][0] >= sl[k][1] - 1e-12 for k in range(len(sl) - 1)))
            arr_new = {("new", k): a for k, (a, _) in enumerate(g)}
            arr_new.update({("other", k): a for k, (a, _) in enumerate(o)})
            arr_new.update({("bnew", k): a for k, (a, _) in enumerate(b)})
            st = all(s >= arr_new[(kind, tag)] - 1e-12 for line, kind, tag, i, s, e in tr if line == "c" and kind in ("new", "other", "bnew"))
            bl = {rr.lseq: b[-1 - rr.owner][0] for rr in r["restores"] if rr.kind == "bload"}
            st &= all(s >= bl[tag] - 1e-12 for line, kind, tag, i, s, e in tr if line == "l" and kind == "bload")
            check(f"(c) {pol} {tier} {L}：請求在到達前不開始", st)
            if pol in ("A4", "I4"):
                check(f"(c) {pol}：P4 基底不重算", r["n_rec"] == 0)
    # 反轉的玩具例：nfs 8K；L 的背景載入在 −0.001 開始，owner 是 L，H 跟隨者在 0.01 到
    cell = dict(tier="nfs", L="8K", n=16, N=2, spread=0.0, rho_o=0.0, rho_b=0.0, h=0.0)
    tL, W, tn = t_load_of(C, "nfs"), 16 * t_load_of(C, "nfs"), t_new_of(C, 16)
    g, o, b = [(0.0, LO), (0.01, HI)], [], [(-0.001, LO)]
    ttf = {p: simulate(cell, C, p, g, o, b)["ttft_g"][1] for p in POLICIES}
    check("玩具：A4 的 H 跟隨者等背景＋R（≈ 2W＋t_new）", abs(ttf["A4"] - (2 * W - 0.001 + tn - 0.01)) < 1e-9)
    check("玩具：I4 的 H 跟隨者只等正在載的一塊＋R（≈ t_L＋W＋t_new）", abs(ttf["I4"] - (tL - 0.001 + W + tn - 0.01)) < 1e-9)
    check("玩具：D4 的 H 私人載入＝I4（都在 H，排在同一塊之後）", abs(ttf["D4"] - ttf["I4"]) < 1e-9)
    rec = sum(f_of(C, i) for i in range(16))
    check("玩具：C4 的 H 跟隨者自己重算（≈ Σf＋t_new），nfs 8K 比 I4 快", abs(ttf["C4"] - (rec + tn)) < 1e-9 and ttf["C4"] < ttf["I4"])
    check("玩具：A5（L 的 R 可幫忙）＜A4", ttf["A5"] < ttf["A4"])
    # 計算線的優先級：H 的別人插到排隊的 L 前面
    cell = dict(tier="cpu", L="8K", n=16, N=1, spread=0.0, rho_o=0.0, rho_b=0.0, h=0.0)
    o = [(0.0, LO), (0.001, LO), (0.002, LO), (0.003, HI)]
    r = simulate(cell, C, "A4", [(10.0, LO)], o, [])
    check("計算線：H 的別人（第 4 個到）比第 2、3 個 L 先做完", r["ttft_o"][3] + 0.003 < r["ttft_o"][1] + 0.001)
    # 共同亂數：時間不看 h，H 集合隨 h 單調
    c1 = dict(tier="nvme", L="8K", n=16, N=16, spread=0.5, rho_o=0.5, rho_b=0.4, h=0.1)
    c3 = dict(c1, h=0.3)
    a1, a3 = arrivals(c1, C, 7), arrivals(c3, C, 7)
    check("到達：同格不同 h 的時間相同", [t for t, _ in a1[0] + a1[1] + a1[2]] == [t for t, _ in a3[0] + a3[1] + a3[2]])
    check("到達：h＝0.1 的 H 集合 ⊆ h＝0.3 的", all(x[1] == LO or y[1] == HI for x, y in zip(a1[0] + a1[1] + a1[2], a3[0] + a3[1] + a3[2])))
    Rt = rates(c1, C)
    check("速率：λ_b＝ρ_b÷max(n·t_L, t_new)、計算線總使用率＝max(ρ_o, λ_b·t_new)",
          abs(Rt["lam_b"] - 0.4 / max(Rt["W"], Rt["tn"])) < 1e-12 and abs(Rt["util_c"] - max(0.5, Rt["lam_b"] * Rt["tn"])) < 1e-12)
    # 判定函式（合成數字）
    def fake(v4, v5, vd=None, vc=None, key=("nfs", "8K", "16", "0.5", "0.5", "0.4", "0.3")):
        base_r = dict(zip(("tier", "L", "N", "spread", "rho_o", "rho_b", "h"), key), hexp_p95=None, l_p95=1.0, h_p95=1.0)
        p = {"A4": 1.0 + v4, "I4": 1.0, "D4": vd or 1.0, "C4": vc or 1.0, "A5": 1.0 + v5, "I5": 1.0}
        return [dict(base_r, policy=k, hsh_p95=x) for k, x in p.items()]
    rows = []
    for j in range(12):
        rows += fake(0.04, 0.01, key=("nfs", "8K", "16", "0.5", "0.5", str(j), "0.3"))
    check("決定性判定：12 格都 <5% → STOP-P", verdict_decisive(rows)["verdict"] == "STOP-P")
    rows[0]["hsh_p95"] = 1.2
    check("決定性判定：一格 inv4＝20% → CONTINUE", verdict_decisive(rows)["verdict"] == "CONTINUE")
    big = sum((fake(0.3, 0.1, vd=0.8, key=("nfs", "8K", "16", "0.5", "0.5", str(j), "0.3")) for j in range(10)), [])
    small = sum((fake(0.01, 0.0, key=("cpu", "8K", "16", "0.5", "0.5", str(j), "0.3")) for j in range(10)), [])
    check("全網格判定：(E)(X) 都成立 → GO-P", verdict_full({"a": big + small, "b": big + small})["verdict"] == "GO-P")
    noesc = sum((fake(0.3, 0.1, key=("nfs", "8K", "16", "0.5", "0.5", str(j), "0.3")) for j in range(10)), [])
    check("全網格判定：(E) 成立、(X) 不成立 → INHERIT-ONLY", verdict_full({"a": noesc + small, "b": noesc + small})["verdict"] == "INHERIT-ONLY")
    check("全網格判定：全部 <5% → STOP-P", verdict_full({"a": small, "b": small})["verdict"] == "STOP-P")
    check("全網格判定：一個平台 GO、一個 STOP → ASK", verdict_full({"a": big + small, "b": small})["verdict"] == "ASK")
    print("ALL PASS" if ok else "SOME FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--slice", choices=("decisive", "full"), required=True)
    r.add_argument("--jobs", type=int, default=1)
    r.add_argument("--out", default=OUT)
    p = sub.add_parser("report")
    p.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    {"run": cmd_run, "report": cmd_report}.get(a.cmd, lambda _: ap.print_help())(a)


if __name__ == "__main__":
    main()
