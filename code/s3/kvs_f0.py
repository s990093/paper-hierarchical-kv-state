#!/usr/bin/env python3
"""kvs_f0.py — F0：一段共享前綴正在還原時，跟隨者該等、該自己算、還是幫忙（CPU 事件模擬，〔模擬〕）。

登記：docs/kv_sharing_hiding_20261026/runs/F0_follower_sim.md（模型、策略、網格、判準都在那裡；這支只照做）。

  python code/s3/kvs_f0.py --self-test                  # 只驗邏輯：守恆、N=1 等於 m16_mcake 的 Cake、到達前不開始…
  python code/s3/kvs_f0.py cal                          # F0-CAL (b)：N=1、32K 對 summary_fp8 的實測
  python code/s3/kvs_f0.py run --slice decisive         # 決定性實驗的 12 格
  python code/s3/kvs_f0.py run --slice full --jobs 16   # 全網格
  python code/s3/kvs_f0.py report                       # 判定表

純 Python（不用 numpy），head node 或任何 CPU 都能跑。輸出都有 run_id、ts。
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
sys.path.insert(0, CODE)

OUT = os.path.join(REPO, "results", "s3_h200", "kvs_f0")
MiB, GiB = 1 << 20, 1 << 30
PANELS = (("h200", "llama"), ("h200", "qwen"), ("mi300x", "llama"), ("mi300x", "qwen"))
TIERS = ("cpu", "nvme", "nfs")
LENS = {"8K": 16, "32K": 64, "128K": 256}
NS = (1, 4, 16, 64)
SPREADS = (0.0, 0.5, 2.0)
RHOS = (0.0, 0.5, 0.8)
INTERF = {"none": (1.0, 1.0), "rec+6%": (1.06, 1.0), "load+50%": (1.0, 1.5)}
SEEDS = tuple(range(20))
FIXED = ("P0", "P1", "P2", "P3", "P4", "P5")
OURS = "P6"
THETAS = {"P6": 1.0, "P7t0.25": 0.25, "P7t4": 4.0, "P7tinf": math.inf}     # P7＝{P0…P6} ∪ 這三個門檻（只報告）
POLICIES = FIXED + tuple(THETAS)
BUNDLE = {"P4", "P5"} | set(THETAS)
HELP = {"P5"} | set(THETAS)
DECISIVE = dict(panel=("h200", "llama"), tiers=TIERS, lens=("8K", "128K"), rhos=(0.0, 0.8), N=16, spread=0.5)
FIELDS = ["run_id", "ts", "slice", "platform", "model", "tier", "L", "n_blocks", "N", "spread", "rho_o", "interference", "policy",
          "n_seeds", "n_req", "n_group", "n_other", "ttft_p95", "ttft_mean", "group_p95", "group_mean", "other_p95", "other_mean",
          "restore_mean", "n_rec_blocks", "n_load_blocks", "t_load_block_s", "W_load_s", "c_o_s", "t_new_s"]


def run_id() -> str:
    return os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-kvs-f0-adhoc"


def ts() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def run_dir() -> str:
    d = os.environ.get("RUN_DIR") or os.path.join(os.environ.get("TIARA_RUNS", os.path.join(REPO, "_big", "runs")), run_id())
    os.makedirs(d, exist_ok=True)
    return d


# ============================================================================ 常數（登記 §1.2）
def _t_new32(plat: str, model: str) -> float:
    """＝m16_i1_predict.consts 的算法：E0、cake、32768、idle 各 tier 的（ttft − restore），排序後取 [len//2]。"""
    p = os.path.join(REPO, f"results/m10_precision_{plat}/speed_{model}/summary_fp8.csv")
    v = sorted(float(r["median_ttft"]) - float(r["median_t_restore"]) for r in csv.DictReader(open(p))
               if r["config"] == "E0" and r["mode"] == "cake" and r["length"] == "32768" and r["gpu_state"] == "idle")
    if not v:
        raise RuntimeError(f"{p}：沒有 E0 cake 32768 idle 的列")
    return v[len(v) // 2]


def consts(plat: str, model: str) -> dict:
    bwp = os.path.join(REPO, f"results/m10_precision_{plat}/speed_{model}/bw.json")
    bw = json.load(open(bwp))
    f = [x / 1e3 for x in bw["f_ms_median_by_chunk"]]
    if len(f) != 64 or bw["n_blocks"] != 64:
        raise RuntimeError(f"{bwp}：f 的長度 {len(f)}／n_blocks {bw['n_blocks']} 不是 64")
    n = len(f)
    mx, my = (n - 1) / 2, sum(f) / n                            # 最小平方 f(i)＝f0＋α·i（秒）
    alpha = sum((i - mx) * (y - my) for i, y in enumerate(f)) / sum((i - mx) ** 2 for i in range(n))
    f0 = my - alpha * mx
    tp = json.load(open(os.path.join(REPO, "results/m7_write_policy_mi300x/tier_params.json")))
    gb = {"cpu": float(bw["real_pinned_h2d_GiBps"]), "nfs": float(tp["nfs"]["read_GiBps"])}
    if plat == "h200":
        rows = [r for r in csv.DictReader(open(os.path.join(REPO, "results/s3_h200/gfio/gfio_20261026-104426-s3-gfio.csv")))
                if r["dir"] == "/tmp/hungweilai-s3/gfio" and r["pattern"] == "seq" and r["io_bytes"] == str(64 * MiB) and r["threads"] == "8"]
        if len(rows) != 1:
            raise RuntimeError(f"G-fio 的 /tmp seq 64MiB x8 應該剛好一列，找到 {len(rows)}")
        gb["nvme"] = float(rows[0]["gibps"])
    else:
        gb["nvme"] = float(tp["local"]["read_GiBps"])
    for k, v in bw["targets"].items():                           # b16／b32／b48：只在尺準（cal）用
        gb[k] = float(v["GiBps"])
    C = dict(platform=plat, model=model, f=f, f0=f0, alpha=alpha, chunk_bytes=int(bw["chunk_bytes"]), gibps=gb,
             t_new32=_t_new32(plat, model), sources=dict(bw=os.path.relpath(bwp, REPO), bw_run=bw["run_id"]))
    C["c_o"] = sum(f_of(C, i) for i in range(60, 64))
    return C


def f_of(C: dict, i: int) -> float:
    """第 i 塊（512 token）的重算秒數：i<64 用實測，之後用 f0＋α·i 外推。"""
    return C["f"][i] if i < 64 else C["f0"] + C["alpha"] * i


def t_new_of(C: dict, n: int) -> float:
    return C["t_new32"] * (C["f0"] + C["alpha"] * n) / (C["f0"] + C["alpha"] * 64)


def t_load_of(C: dict, tier: str) -> float:
    return C["chunk_bytes"] / (C["gibps"][tier] * GiB)


# ============================================================================ 模擬
class Restore:
    """一份前綴的還原。kind：shared（共享 P：載入線從後、可被幫忙從前）、pload（P0 的私人整份載入）、prec（P2 的私人整份重算）。"""
    __slots__ = ("n", "kind", "p", "q", "inflight", "done_t", "waiters", "n_rec", "n_load", "owner")

    def __init__(self, n: int, kind: str, owner: int):
        self.n, self.kind, self.owner = n, kind, owner
        self.p, self.q, self.inflight = 0, n - 1, 0
        self.done_t = None
        self.waiters = []
        self.n_rec = self.n_load = 0


def arrivals(cell: dict, C: dict, seed: int) -> tuple[list, list, float]:
    """共同亂數：同一格、同一個 seed，所有策略看到同樣的到達。回傳（共享請求到達、別人的到達、H）。"""
    key = f"{C['platform']}|{C['model']}|{cell['tier']}|{cell['n']}|{cell['N']}|{cell['spread']}|{cell['rho_o']}|{seed}"
    rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))
    W = cell["n"] * t_load_of(C, cell["tier"])
    g = [0.0] + sorted(rng.uniform(0.0, cell["spread"] * W) for _ in range(cell["N"] - 1))
    H = g[-1] + W + 5.0
    o = []
    if cell["rho_o"] > 0:
        lam = cell["rho_o"] / C["c_o"]
        t = rng.expovariate(lam)
        while t <= H:
            o.append(t)
            t += rng.expovariate(lam)
    return g, o, H


def simulate(cell: dict, C: dict, policy: str, g_arr: list, o_arr: list, k_rec: float = 1.0, k_load: float = 1.0,
             trace: list | None = None) -> dict:
    """一個 episode。回傳每個請求的 TTFT 與統計。trace（list）給自測：記錄 (線, 種類, 誰, 塊, 開始, 結束)。"""
    assert policy in POLICIES, policy
    n, tL = cell["n"], t_load_of(C, cell["tier"])
    tnew, c_o = t_new_of(C, n), C["c_o"]
    bundle, helper = policy in BUNDLE, policy in HELP
    theta = THETAS.get(policy)
    ev, seq = [], [0]

    def push(t, kind, x):
        seq[0] += 1
        heapq.heappush(ev, (t, seq[0], kind, x))

    for k, a in enumerate(g_arr):
        push(a, "ag", k)
    for k, a in enumerate(o_arr):
        push(a, "ao", k)
    arr_g, arr_o = list(g_arr), list(o_arr)
    fin_g, fin_o = [None] * len(g_arr), [None] * len(o_arr)
    restore_done = []
    cq = []                    # 計算線佇列：(類別, 準備好的時間, seq, 項目)
    loadq = []                 # 載入線：還要載的 Restore（先到先服務）
    busy = {"c": None, "l": None}
    R = [None]                 # 共享 P 的還原（去重的策略）
    stats = dict(rec=0, load=0)

    def enq(t, item):
        seq[0] += 1
        cls = 0 if (bundle and item[0] == "new") else 1
        heapq.heappush(cq, (cls, t, seq[0], item))

    def complete(r: Restore, t):
        if r.p > r.q and r.inflight == 0 and r.done_t is None:
            r.done_t = t
            restore_done.append((r.kind, t - (arr_g[r.owner] if r.owner >= 0 else 0.0)))
            for k in r.waiters:
                enq(t, ("new", k))
            r.waiters = []

    def start_load(t):
        while loadq and not (loadq[0].p <= loadq[0].q):
            loadq.pop(0)
        if not loadq:
            busy["l"] = None
            return
        r = loadq[0]
        i = r.q
        r.q -= 1
        r.inflight += 1
        r.n_load += 1
        stats["load"] += 1
        dur = tL * (k_load if busy["c"] is not None else 1.0)
        busy["l"] = ("load", r, i)
        push(t + dur, "ld", None)
        if trace is not None:
            trace.append(("l", r.kind, r.owner, i, t, t + dur))

    def start_compute(t):
        r = R[0]
        if helper and r is not None and r.done_t is None and r.p <= r.q:
            i = r.p
            if theta is None:                                                    # P5：一律幫忙
                do = True
            else:
                Q, W = len(cq), len(r.waiters)
                do = Q == 0 or (theta != math.inf and W * tL >= theta * Q * f_of(C, i))
            if do:
                r.p += 1
                r.inflight += 1
                r.n_rec += 1
                stats["rec"] += 1
                dur = f_of(C, i) * (k_rec if busy["l"] is not None else 1.0)
                busy["c"] = ("rec", r, i)
                push(t + dur, "cd", None)
                if trace is not None:
                    trace.append(("c", "help", r.owner, i, t, t + dur))
                return
        if not cq:
            busy["c"] = None
            return
        _, _, _, item = heapq.heappop(cq)
        kl = k_rec if busy["l"] is not None else 1.0
        if item[0] == "prec":
            pr = item[1]
            i = pr.p
            pr.p += 1
            pr.inflight += 1
            pr.n_rec += 1
            stats["rec"] += 1
            dur = f_of(C, i) * kl
            busy["c"] = ("prec", pr, i)
        else:
            dur = (tnew if item[0] == "new" else c_o) * kl
            busy["c"] = item
        push(t + dur, "cd", None)
        if trace is not None:
            trace.append(("c", item[0], item[1] if item[0] != "prec" else item[1].owner, busy["c"][2] if item[0] == "prec" else -1, t, t + dur))

    def kick(t):
        if busy["l"] is None:
            start_load(t)
        if busy["c"] is None:
            start_compute(t)

    def g_arrive(k, t):
        r = R[0]
        if policy == "P0":
            pr = Restore(n, "pload", k)
            pr.waiters.append(k)
            loadq.append(pr)
            return
        if r is None:
            r = R[0] = Restore(n, "shared", k)
            r.waiters.append(k)
            loadq.append(r)
            return
        if r.done_t is not None:
            enq(t, ("new", k))
            return
        follow_rec = policy == "P2" or (policy == "P3" and busy["c"] is None and not cq)
        if follow_rec:
            pr = Restore(n, "prec", k)
            pr.waiters.append(k)
            pr.q = n - 1
            enq(t, ("prec", pr))
        else:
            r.waiters.append(k)

    while ev:
        t, _, kind, x = heapq.heappop(ev)
        if kind == "ag":
            g_arrive(x, t)
        elif kind == "ao":
            enq(t, ("other", x))
        elif kind == "ld":
            _, r, i = busy["l"]
            busy["l"] = None
            r.inflight -= 1
            complete(r, t)
        elif kind == "cd":
            b = busy["c"]
            busy["c"] = None
            if b[0] == "rec":
                b[1].inflight -= 1
                complete(b[1], t)
            elif b[0] == "prec":
                pr = b[1]
                pr.inflight -= 1
                if pr.p <= pr.q:
                    enq(t, ("prec", pr))
                else:
                    complete(pr, t)
            elif b[0] == "new":
                fin_g[b[1]] = t
            else:
                fin_o[b[1]] = t
        kick(t)
    if any(x is None for x in fin_g) or any(x is None for x in fin_o):
        raise RuntimeError(f"有請求沒做完（{policy}，{cell}）")
    tg = [f - a for f, a in zip(fin_g, arr_g)]
    to = [f - a for f, a in zip(fin_o, arr_o)]
    return dict(ttft_g=tg, ttft_o=to, restores=restore_done, n_rec=stats["rec"], n_load=stats["load"])


def pct(v: list, p: float) -> float:
    s = sorted(v)
    if len(s) == 1:
        return s[0]
    x = p * (len(s) - 1)
    lo = int(x)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (x - lo)


def run_cell(args) -> list:
    """一格 × 全部策略 × 一種干擾 × 20 個 seed → 每個策略一列（P95 等是所有 seed 的請求合起來算）。"""
    slice_, plat, model, cell, interf, keep_raw = args
    C = consts(plat, model)
    kr, kl = INTERF[interf]
    rows, raw = [], {}
    arr = [arrivals(cell, C, s) for s in SEEDS]
    for pol in POLICIES:
        tg, to, rest, nrec, nload = [], [], [], 0, 0
        for g, o, _ in arr:
            r = simulate(cell, C, pol, g, o, kr, kl)
            tg += r["ttft_g"]
            to += r["ttft_o"]
            rest += [x[1] for x in r["restores"]]
            nrec += r["n_rec"]
            nload += r["n_load"]
        allv = tg + to
        rows.append(dict(slice=slice_, platform=plat, model=model, tier=cell["tier"], L=cell["L"], n_blocks=cell["n"], N=cell["N"],
                         spread=cell["spread"], rho_o=cell["rho_o"], interference=interf, policy=pol, n_seeds=len(SEEDS),
                         n_req=len(allv), n_group=len(tg), n_other=len(to), ttft_p95=pct(allv, 0.95), ttft_mean=sum(allv) / len(allv),
                         group_p95=pct(tg, 0.95), group_mean=sum(tg) / len(tg),
                         other_p95=pct(to, 0.95) if to else "NOT_MEASURED", other_mean=sum(to) / len(to) if to else "NOT_MEASURED",
                         restore_mean=sum(rest) / len(rest) if rest else "NOT_MEASURED", n_rec_blocks=nrec, n_load_blocks=nload,
                         t_load_block_s=t_load_of(C, cell["tier"]), W_load_s=cell["n"] * t_load_of(C, cell["tier"]), c_o_s=C["c_o"],
                         t_new_s=t_new_of(C, cell["n"])))
        if keep_raw:
            raw[pol] = dict(group=tg, other=to)
    return rows, (cell, interf, raw) if keep_raw else None


def cells_of(slice_: str):
    if slice_ == "decisive":
        d = DECISIVE
        for tier in d["tiers"]:
            for L in d["lens"]:
                for rho in d["rhos"]:
                    yield d["panel"], dict(tier=tier, L=L, n=LENS[L], N=d["N"], spread=d["spread"], rho_o=rho), ("none",)
        return
    for panel in PANELS:
        for tier in TIERS:
            for L, n in LENS.items():
                for N in NS:
                    for sp in SPREADS:
                        for rho in RHOS:
                            yield panel, dict(tier=tier, L=L, n=n, N=N, spread=sp, rho_o=rho), tuple(INTERF)


# ============================================================================ 指令
def cmd_run(a):
    rid, rd = run_id(), run_dir()
    jobs = [(a.slice, p[0], p[1], cell, itf, a.slice == "decisive") for p, cell, itfs in cells_of(a.slice) for itf in itfs]
    print(f"[f0] {a.slice}：{len(jobs)} 個 (格, 干擾)，每個 {len(POLICIES)} 個策略 × {len(SEEDS)} 個 seed → run {rid}", flush=True)
    if a.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(a.jobs) as ex:
            res = list(ex.map(run_cell, jobs, chunksize=4))
    else:
        res = [run_cell(j) for j in jobs]
    os.makedirs(a.out, exist_ok=True)
    by = {}
    for rows, _ in res:
        for r in rows:
            by.setdefault((r["platform"], r["model"]), []).append(r)
    for (plat, model), rows in by.items():
        # 全網格每個面板約 2.2 MB → 寫 .csv.gz（約 0.2 MB；repo 的檔案 <1 MB，CLAUDE.md §2）
        p = os.path.join(a.out, f"f0_{a.slice}_{plat}_{model}.csv" + (".gz" if a.slice == "full" else ""))
        with (gzip.open(p, "wt", newline="") if p.endswith(".gz") else open(p, "w", newline="")) as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            for r in rows:
                r.update(run_id=rid, ts=ts())
                w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in ((k, r.get(k, "")) for k in FIELDS)})
        print(f"[f0] → {p}（{len(rows)} 列）", flush=True)
    raws = [x for _, x in res if x]
    if raws:
        p = os.path.join(rd, f"f0_{a.slice}_requests.json.gz")
        with gzip.open(p, "wt") as fh:
            json.dump([dict(cell=c, interference=i, ttft=r) for c, i, r in raws], fh)
        print(f"[f0] 每個請求的明細 → {p}", flush=True)


def cmd_cal(a):
    """F0-CAL (b)：L＝32K、N＝1、ρ_o＝0，P5 的 TTFT ÷ summary_fp8 的 E0 cake 32768 idle median_ttft。"""
    rid = run_id()
    out = dict(run_id=rid, ts=ts(), rule="每個面板，有實測的 tier（cpu、b16、b32、b48）至少 3/4 落在 [0.9, 1.1]", panels={})
    ok_all = True
    for plat, model in PANELS:
        C = consts(plat, model)
        meas = {r["bw_name"]: float(r["median_ttft"]) for r in csv.DictReader(open(os.path.join(REPO, f"results/m10_precision_{plat}/speed_{model}/summary_fp8.csv")))
                if r["config"] == "E0" and r["mode"] == "cake" and r["length"] == "32768" and r["gpu_state"] == "idle"}
        res = {}
        for tier in ("cpu", "b16", "b32", "b48"):
            if tier not in meas:
                res[tier] = "NOT_MEASURED"
                continue
            cell = dict(tier=tier, L="32K", n=64, N=1, spread=0.0, rho_o=0.0)
            r = simulate(cell, C, "P5", [0.0], [])
            res[tier] = dict(sim=round(r["ttft_g"][0], 4), meas=meas[tier], ratio=round(r["ttft_g"][0] / meas[tier], 4))
        ins = [v for v in res.values() if isinstance(v, dict) and 0.9 <= v["ratio"] <= 1.1]
        ok = len(ins) >= 3
        ok_all &= ok
        out["panels"][f"{plat}/{model}"] = dict(tiers=res, pass_=ok, t_new32=C["t_new32"], f0=C["f0"], alpha=C["alpha"])
        print(f"[f0 cal] {plat}/{model}: {'PASS' if ok else 'FAIL'} {res}", flush=True)
    out["verdict"] = "PASS" if ok_all else "FAIL"
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "f0_cal.json"), "w"), indent=1, ensure_ascii=False)
    print(f"[f0 cal] {out['verdict']}", flush=True)


def _read(path: str) -> list:
    rows = list(csv.DictReader(gzip.open(path, "rt") if path.endswith(".gz") else open(path)))
    for r in rows:
        r["ttft_p95"] = float(r["ttft_p95"])
    return rows


def _cellkey(r):
    return (r["tier"], r["L"], r["N"], r["spread"], r["rho_o"])


def verdict_decisive(rows: list) -> dict:
    rows = [r for r in rows if r["interference"] == "none"]
    cells = {}
    for r in rows:
        cells.setdefault(_cellkey(r), {})[r["policy"]] = r["ttft_p95"]
    per = {}
    for k, d in cells.items():
        best = min(FIXED, key=lambda p: (d[p], p))
        per["|".join(k)] = dict(best_fixed=best, gain_P6=1 - d[OURS] / d[best])
    same = len({v["best_fixed"] for v in per.values()}) == 1
    small = all(v["gain_P6"] < 0.05 for v in per.values())
    return dict(cells=per, all_same_best_fixed=same, all_gain_below_5pct=small,
                verdict="STOP-F" if (same and small) else "CONTINUE", n_cells=len(per))


def verdict_full(rows_by_plat: dict) -> dict:
    out = {}
    for plat, rows in rows_by_plat.items():
        rows = [r for r in rows if r["interference"] == "none"]
        cells = {}
        for r in rows:
            cells.setdefault(_cellkey(r), {})[r["policy"]] = r["ttft_p95"]
        gm = {p: math.exp(sum(math.log(d[p]) for d in cells.values()) / len(cells)) for p in FIXED}
        rstar = min(FIXED, key=lambda p: (gm[p], p))
        gains = [1 - d[OURS] / d[rstar] for d in cells.values()]
        bests = {min(FIXED, key=lambda p: (d[p], p)) for d in cells.values()}
        c1 = sum(g > 0.15 for g in gains) / len(gains) >= 0.10
        c2 = min(gains) >= -0.05
        c3 = len(bests) > 1
        out[plat] = dict(n_cells=len(cells), R_star=rstar, geomean_p95=gm, frac_gain_gt_15=sum(g > 0.15 for g in gains) / len(gains),
                         min_gain=min(gains), max_gain=max(gains), best_fixed_rules=sorted(bests), i=c1, ii=c2, iii=c3,
                         all_below_5=all(g < 0.05 for g in gains))
    plats = list(out.values())
    if plats and all(p["i"] and p["ii"] and p["iii"] for p in plats):
        v = "GO-F"
    elif plats and (all(p["all_below_5"] for p in plats) or all(not p["iii"] for p in plats)):
        v = "STOP-F"
    else:
        v = "ASK"
    return dict(platforms=out, verdict=v)


def cmd_report(a):
    rep = dict(run_id=run_id(), ts=ts())
    cal = os.path.join(a.out, "f0_cal.json")
    rep["cal"] = json.load(open(cal))["verdict"] if os.path.exists(cal) else "NOT_MEASURED"
    dp = os.path.join(a.out, "f0_decisive_h200_llama.csv")
    if os.path.exists(dp):
        rep["decisive"] = verdict_decisive(_read(dp))
    full = {plat: os.path.join(a.out, f"f0_full_{plat}_llama.csv.gz") for plat in ("h200", "mi300x")}
    full = {k: (p if os.path.exists(p) else p[:-3]) for k, p in full.items()}       # 舊的未壓縮檔也讀
    if all(os.path.exists(p) for p in full.values()):
        rep["full_llama"] = verdict_full({k: _read(p) for k, p in full.items()})
    if rep["cal"] != "PASS":
        rep["note"] = "尺準沒有 PASS：照登記 §3，判定暫停（回報、問人）"
    json.dump(rep, open(os.path.join(a.out, "f0_verdict.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: (v if k not in ("decisive",) else {kk: vv for kk, vv in v.items() if kk != "cells"}) for k, v in rep.items()},
                     ensure_ascii=False, indent=1))


# ============================================================================ 自測（只驗邏輯，不產生結果）
def self_test() -> bool:
    import m16_mcake as MC
    ok = True

    def check(name, cond):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name)
        ok &= bool(cond)

    C = consts("h200", "llama")
    check("常數：f 有 64 個、外推斜率 > 0、c_o > 0、四個 tier 都有頻寬", len(C["f"]) == 64 and C["alpha"] > 0 and C["c_o"] > 0
          and all(t in C["gibps"] for t in ("cpu", "nvme", "nfs", "b16")))
    check("外推在 i<64 時不用（直接用實測）", f_of(C, 10) == C["f"][10] and abs(f_of(C, 64) - (C["f0"] + C["alpha"] * 64)) < 1e-15)
    # (a) N=1、ρ=0：P5 的還原＝m16_mcake 的 Cake（rr、單 session）
    for tier in ("cpu", "nvme", "nfs", "b32"):
        for L, n in LENS.items():
            cell = dict(tier=tier, L=L, n=n, N=1, spread=0.0, rho_o=0.0)
            r = simulate(cell, C, "P5", [0.0], [])
            f = (lambda i: f_of(C, i))
            m = MC.simulate([MC.Sess(n=n, tier="x")], "rr", f, {"x": t_load_of(C, tier)}, t_new=t_new_of(C, n))
            same = abs(r["restores"][0][1] - m["sessions"][0]["restored"]) < 1e-9 and abs(r["ttft_g"][0] - m["sessions"][0]["ttft"]) < 1e-9
            check(f"F0-CAL(a) {tier} {L}：P5（N=1）＝m16_mcake 的 Cake", same)
    # 守恆與基本性質（合成格）
    cell = dict(tier="nvme", L="8K", n=16, N=6, spread=0.5, rho_o=0.5)
    g, o, H = arrivals(cell, C, 3)
    g2, o2, _ = arrivals(cell, C, 3)
    check("共同亂數：同格同 seed 到達相同", g == g2 and o == o2)
    check("到達：第一個在 0、都在 [0, spread·W]、別人在 [0, H]", g[0] == 0.0 and max(g) <= 0.5 * 16 * t_load_of(C, "nvme") + 1e-12
          and all(0 <= x <= H for x in o))
    for pol in POLICIES:
        tr = []
        r = simulate(cell, C, pol, g, o, trace=tr)
        check(f"{pol}：每個請求都做完、TTFT ≥ 0", len(r["ttft_g"]) == len(g) and all(x >= 0 for x in r["ttft_g"] + r["ttft_o"]))
        # 每一份還原：每塊恰好一次（共享 P 的載入＋幫忙不重複；私人的各自完整）
        blocks = {}
        for line, kind, owner, i, s, e in tr:
            if (line == "l") or kind in ("help", "prec"):
                tag = "shared" if kind in ("shared", "help") else f"{kind}{owner}"
                blocks.setdefault(tag, []).append(i)
        cons = all(sorted(v) == list(range(cell["n"])) for v in blocks.values())
        check(f"{pol}：每份還原的每一塊恰好做一次", cons)
        lines_ok = all(e >= s for *_, s, e in tr)
        starts_c = sorted((s, e) for line, *_, s, e in tr if line == "c")
        starts_l = sorted((s, e) for line, *_, s, e in tr if line == "l")
        no_overlap = all(starts_c[k + 1][0] >= starts_c[k][1] - 1e-12 for k in range(len(starts_c) - 1)) and \
            all(starts_l[k + 1][0] >= starts_l[k][1] - 1e-12 for k in range(len(starts_l) - 1))
        check(f"{pol}：每條線一次只做一件事", lines_ok and no_overlap)
        if pol == "P0":
            check("P0：共享請求各載一份（載入塊數＝N·n）", r["n_load"] == len(g) * cell["n"] and r["n_rec"] == 0)
        if pol in ("P1", "P4"):
            check(f"{pol}：只載一份、不重算", r["n_load"] == cell["n"] and r["n_rec"] == 0)
        if pol == "P5":
            check("P5：載入＋幫忙＝n（去重）", r["n_load"] + r["n_rec"] == cell["n"])
    # 到達前不開始：晚到的別人
    tr = []
    simulate(dict(tier="cpu", L="8K", n=16, N=1, spread=0.0, rho_o=0.0), C, "P1", [0.0], [5.0], trace=tr)
    check("別人的工作在到達（5 s）之前不開始", all(s >= 5.0 - 1e-12 for line, kind, *_, s, e in tr if kind == "other"))
    # bundle：P 到了之後，等 P 的新一輪排在別人前面（P4）vs 先到先服務（P1）
    cellb = dict(tier="nfs", L="8K", n=16, N=3, spread=0.0, rho_o=0.0)
    ob = [0.01 * k for k in range(1, 40)]                  # 還原期間堆了一串別人的工作
    r1 = simulate(cellb, C, "P1", [0.0, 0.0, 0.0], ob)
    r4 = simulate(cellb, C, "P4", [0.0, 0.0, 0.0], ob)
    check("bundle：P4 的共享請求 TTFT ≤ P1", max(r4["ttft_g"]) <= max(r1["ttft_g"]) + 1e-12)
    # P6：GPU 閒（沒有別人）時＝P5（Cake）
    cell1 = dict(tier="nfs", L="32K", n=64, N=4, spread=0.0, rho_o=0.0)
    a5 = simulate(cell1, C, "P5", [0.0] * 4, [])
    a6 = simulate(cell1, C, "P6", [0.0] * 4, [])
    check("P6：沒有別人的工作時和 P5 相同", a5["ttft_g"] == a6["ttft_g"])
    # θ＝∞：只在計算線沒排隊時幫忙；有一大串別人時，P7tinf 的重算塊數 ≤ P6 ≤ P5
    cellh = dict(tier="nfs", L="32K", n=64, N=16, spread=0.0, rho_o=0.8)
    gh, oh, _ = arrivals(cellh, C, 0)
    nrec = {p: simulate(cellh, C, p, gh, oh)["n_rec"] for p in ("P5", "P6", "P7tinf")}
    check("幫忙的量：P7tinf ≤ P6 ≤ P5", nrec["P7tinf"] <= nrec["P6"] <= nrec["P5"])
    # P2：跟隨者自己整份重算（重算塊數＝跟隨者數 × n，如果都在載入期間到）
    cell2 = dict(tier="nfs", L="8K", n=16, N=4, spread=0.0, rho_o=0.0)
    r2 = simulate(cell2, C, "P2", [0.0, 0.0, 0.0, 0.0], [])
    check("P2：同時到的 3 個跟隨者各重算 n 塊；owner 載入 n 塊", r2["n_rec"] == 3 * 16 and r2["n_load"] == 16)
    # P3：計算線空著 → 跟隨者重算；有排隊 → 等
    r3 = simulate(cell2, C, "P3", [0.0, 0.0, 0.0, 0.0], [])
    check("P3：第一個跟隨者到時計算線空 → 重算；之後的看到排隊 → 等", r3["n_rec"] == 16 and r3["n_load"] == 16)
    # 干擾不會讓事情變快
    base = simulate(cellh, C, "P6", gh, oh)
    slow = simulate(cellh, C, "P6", gh, oh, k_rec=1.06, k_load=1.5)
    check("有干擾時共享請求的 P95 不會變短", pct(slow["ttft_g"], 0.95) >= pct(base["ttft_g"], 0.95) - 1e-9)
    # 判定函式（合成數字）
    fake = []
    for tier in TIERS:
        for L in ("8K", "128K"):
            for rho in ("0.0", "0.8"):
                for p in POLICIES:
                    v = {"P1": 2.0, "P6": 1.99}.get(p, 3.0)
                    fake.append(dict(tier=tier, L=L, N="16", spread="0.5", rho_o=rho, interference="none", policy=p, ttft_p95=v))
    vd = verdict_decisive(fake)
    check("決定性判定：全部同一條、改進 <5% → STOP-F", vd["verdict"] == "STOP-F" and vd["n_cells"] == 12)
    fake[0 * len(POLICIES) + POLICIES.index("P2")]["ttft_p95"] = 1.5      # 一格的最佳固定規則換成 P2
    check("決定性判定：有一格換規則 → CONTINUE", verdict_decisive(fake)["verdict"] == "CONTINUE")
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
    c = sub.add_parser("cal")
    c.add_argument("--out", default=OUT)
    p = sub.add_parser("report")
    p.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    {"run": cmd_run, "cal": cmd_cal, "report": cmd_report}.get(a.cmd, lambda _: ap.print_help())(a)


if __name__ == "__main__":
    main()
