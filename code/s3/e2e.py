#!/usr/bin/env python3
"""e2e.py — E2E-C（docs/stage3_design_20261024/runs/E2E_combo.md）：組合策略在 ICPE X1 模型上的端到端模擬（第一段，只用 CPU）。

配置（登記 §2）：K0 純 Cake（慢層 BF16、LRU、stream）；K1 慢層 INT4；K2 ＋HINT-PF；K3 ＋固定 MPS 等級 m*；O 為 O-PF＋m*。
背景（登記 §3）：B0 量到的倍數乘進 X1 的 View（重算 f、反量化 d、CPU 層讀取頻寬）；分解檢查 D 不過的模型改用端到端比值（§3.3）。
背景吞吐（§3.4）：B＝cap × (1 − φ·(1 − ρ̄))，含 MPS 上限本身的代價。

不改 x1_sim／x1_gpu／x1_fi，只 import。Qwen 的 X1 常數是新加的（登記 §7）；Qwen 時把 x1_sim.BF 換成 Qwen 的每塊位元組。

子命令：
  dcheck   §3.2：每個模型用 3.1（基本動作）還是 3.3（端到端比值）→ results/s3_h200/e2e/dcheck.json
  run      一個（模型, trace）的全部格子 → results/s3_h200/e2e/raw/e2e_<model>_<trace>.csv（每列一個（重播, 配置, 背景, 共用方式））
  report   §4–§5 的 m* 與判定 → results/s3_h200/e2e/e2e_report.json
  --self-test  背景＝閒時新的 View 和原本的 View 逐請求逐位元相同；Qwen 版 leads 用 Llama 參數時和 x1_gpu.leads 相同
"""
from __future__ import annotations

import argparse
import copy
import csv
import glob
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
REPO = os.path.dirname(CODE)
DS = os.environ.get("TIARA_DATASETS", "/work/ChihChungHsu/hungweilai/datasets")
os.environ.setdefault("P3_CONST", "h200")
os.environ.setdefault("X1_TRACE_DIR", os.path.join(DS, "traces"))
os.environ.setdefault("X1_FI_DIR", os.path.join(DS, "freeinference_agentic_trace", "release"))
for p in (CODE, os.path.join(CODE, "p3x"), os.path.join(CODE, "icpe")):
    sys.path.insert(0, p)

import numpy as np  # noqa: E402

import p3_platform as PF  # noqa: E402
import x1_sim as XS  # noqa: E402
import x1_fi as XF  # noqa: E402
import x1_gpu as XG  # noqa: E402

assert PF.PLAT == "h200", "E2E-C 只用 H200 的常數（登記 §1）"
GiB = XS.GiB
OUT = os.path.join(REPO, "results", "s3_h200", "e2e")
if PF.TIER:                                              # ICPE T1（docs/h200/icpe/T1_h200_cpu_tier.md）：CPU 層換 H200 的常數，輸出另放
    OUT = os.path.join(PF.T1_ROOT, "e2e")
B0RAW = os.path.join(REPO, "results", "s3_h200", "b0", "raw")
MODELS = ("llama", "qwen")
TRACES = ("conversation", "freeinference")
SLOWS = ("ssd", "nfs")
CPUS = (128, 256)
BGS = ("cprefill:2048:65536", "mixed:512:32:8192")
MPS_MODES = ("mps", "mps87", "mps75", "mps50")          # 平手時上限較寬的優先（登記 §4）
MODES = ("stream",) + MPS_MODES
B0_B32 = {"llama": 2.9404, "qwen": 1.4414}              # B0 的 b32 頻寬（GiB/s；B0 登記 §1.3）
BF_OF = {"llama": 64 * XS.MiB, "qwen": 28 * 4 * 128 * 2 * 2 * 512}   # 每塊 BF16（Qwen2.5-7B：28 層、4 KV head、head_dim 128）
B_MIN = 0.95
GO_GAIN, ASK_GAIN, GO_CELLS = 0.15, 0.05, 8
NOISE = 0.05                                             # 05 §2.1：2σ ≤ 5% 才可用
E2E_OF = {0: None, 1: "e2e_cake_bf16_cpu", 2: "e2e_cake_int4_cpu", 3: "e2e_cake_int4_b32"}


def run_id():
    return os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc"


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


class NotMeasured(Exception):
    pass


# ============================================================================ 常數
def constants(model: str) -> dict:
    """Llama＝ICPE 的 x1_sim.constants()（不改）；Qwen＝登記 §7。"""
    bf, XS.BF = XS.BF, BF_OF["llama"]        # x1_sim.constants() 斷言 Llama 的 BF16 位元組＝模組的 BF
    try:
        K = XS.constants()
    finally:
        XS.BF = bf
    if model == "llama":
        return K
    bw = json.load(open(PF.rdir("m10_precision_mi300x", "speed_qwen", "bw.json")))
    y = np.asarray(bw["f_ms_median_by_chunk"], float)
    assert len(y) == 64, len(y)
    slope, icpt = np.polyfit(np.arange(len(y), dtype=float), y, 1)
    iow = json.load(open(PF.IOWALL))
    assert abs(iow["qwen"]["bytes_per_block"]["bf16"] - BF_OF["qwen"]) < 1, iow["qwen"]["bytes_per_block"]
    import pandas as pd
    d = pd.read_csv(PF.rdir("m10_precision_mi300x", "speed_qwen", "restores.csv.gz"))
    rid4 = PF.run("20261016-052143-m10-grid-int4-qwen")
    g = d[(d.run_id == rid4) & (d.config == "E1") & (d["mode"] == "load_only") & (d.bw_name == "cpu") & (d.fp8_read > 0)]
    dq = (g.t_dequant / g.fp8_read).groupby(g.gpu_state).median()
    Q = copy.deepcopy(K)
    Q.update(f_ms=[float(x) for x in y], f_slope=float(slope), f_icpt=float(icpt), f_runs=[bw.get("run_id")],
             busy_factor=float("nan"), int4_bytes=int(iow["qwen"]["bytes_per_block"]["int4"]),
             dq_idle=float(dq["idle"]), dq_busy=float("nan"), dq_run=rid4)
    return Q


def use_model(model: str):
    """x1_sim 的引擎與 View 都讀模組層的 BF：換模型時一起換（登記 §7）。"""
    XS.BF = BF_OF[model]


# ============================================================================ B0
def load_b0(model: str) -> dict:
    cell = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(B0RAW, f"b0_{model}_b0_*.csv"))):
        for r in csv.DictReader(open(p)):
            cell[(r["bg"], r["mode"], r["prim"])].append(r)
    assert cell, f"沒有 B0 的 {model} 資料"
    # 規則 6：每塊位元組用 B0 自己記的搬運量交叉驗算（64 塊）
    hb = {int(r["bytes"]) for r in cell[("idle", "stream", "h2d_perblock")]}
    db = {int(r["bytes"]) for r in cell[("idle", "stream", "dq_int4")]}
    assert hb == {64 * BF_OF[model]}, (model, hb)
    assert len(db) == 1 and next(iter(db)) % 64 == 0, (model, db)
    return cell


def _med(rows, col):
    v = [float(r[col]) for r in rows if r.get(col) not in ("", None, "NOT_MEASURED")]
    return statistics.median(v) if v else None


def b0_stat(cell, bg, mode, prim):
    rows = cell.get((bg, mode, prim), [])
    t = _med(rows, "t_med_s")
    sg = _med(rows, "sigma_rel")
    return dict(t=t, sigma=sg, ok=bool(t and sg is not None and 2 * sg <= NOISE), rho=_med(rows, "bg_ratio"), n=len(rows))


def factors(cell, bg, mode, path):
    """一個（背景, 共用方式）的倍數、ρ、cap（登記 §3）。要用的不可用 → NotMeasured。"""
    def ratio(prim):
        a, b = b0_stat(cell, bg, mode, prim), b0_stat(cell, "idle", "stream", prim)
        if not (a["ok"] and b["ok"]):
            raise NotMeasured(f"{bg}×{mode}×{prim}：2σ 不過（{a['sigma']}／{b['sigma']}）")
        return a["t"] / b["t"]
    fac = dict(bg=bg, mode=mode, path=path, r_f=ratio("recompute1"))
    if path == "prim":
        fac.update(r_d=ratio("dq_int4"), r_h=ratio("h2d_perblock"))
        fac["r_e2e"] = {}
    else:
        fac["r_e2e"] = {a: ratio(a) for a in ("e2e_cake_bf16_cpu", "e2e_cake_int4_cpu", "e2e_cake_int4_b32", "e2e_cake_bf16_b32")}
    fac["rho"] = {a: b0_stat(cell, bg, mode, a)["rho"] for a in
                  ("recompute1", "e2e_cake_bf16_cpu", "e2e_cake_int4_cpu", "e2e_cake_int4_b32", "e2e_cake_bf16_b32")}
    if any(v is None for v in fac["rho"].values()):
        raise NotMeasured(f"{bg}×{mode}：缺 bg_ratio")
    solo = lambda m: _med([r for k, rs in cell.items() if k[0] == bg and k[1] == m for r in rs], "bg_solo_rate")  # noqa: E731
    fac["cap"] = 1.0 if mode in ("stream", "mps") else solo(mode) / solo("mps")
    return fac


IDLE = dict(bg="idle", mode="stream", path="prim", r_f=1.0, r_d=1.0, r_h=1.0, r_e2e={}, rho={}, cap=1.0)


# ============================================================================ View
class BgView(XS.View):
    """X1 的 View（Cake、GPU 閒的常數）乘上背景的倍數。每次 restore 記下這個請求讀了什麼（ρ 的對應，登記 §3.3／3.4）。"""

    def __init__(self, K, slow, fac, nmax, nreq, slow_bytes=None, dq=None, k0=False):
        super().__init__(K, slow, "idle", "cake", nmax, nreq, slow_bytes=slow_bytes, dq=dq)
        self.fac, self.k0 = fac, k0
        self.key = (slow, fac["bg"], fac["mode"], fac["path"])
        self.Fc_idle = self.Fc
        rf = fac["r_f"]
        if fac["path"] == "prim":
            if rf != 1.0:
                self.f = [x * rf for x in self.f]
                Fc = [0.0]
                for x in self.f:
                    Fc.append(Fc[-1] + x)
                self.Fc = Fc
            if fac["r_d"] != 1.0:
                self.d = self.d * fac["r_d"]
            if fac["r_h"] != 1.0:
                self.cpu.r = self.cpu.r / fac["r_h"]
        self.src = np.zeros(nreq, np.int8)
        self.i = 0

    def action(self, src):
        if src == 3 and self.k0:
            return "e2e_cake_bf16_b32"
        return E2E_OF[src]

    def restore(self, lay, blocks, L, n_all, t0, old_n, pf_done):
        tt, rs, nC, nS, nR = super().restore(lay, blocks, L, n_all, t0, old_n, pf_done)
        meet = nR - (n_all - L)
        src = 0
        if L > meet:
            if nS > 0:
                src = 3
            elif any(lay[i] == 2 for i in range(meet, L)):
                src = 2
            else:
                src = 1
        self.src[self.i] = src
        self.i += 1
        if self.fac["path"] == "e2e":
            rf = self.fac["r_f"]
            base = self.last_end - t0
            re = self.fac["r_e2e"][self.action(src)] if src else rf
            Fc = self.Fc_idle
            tt = base * re + (Fc[n_all] - Fc[L]) * rf
            rs = base * re + ((Fc[old_n] - Fc[L]) * rf if old_n > L else 0.0)
        return tt, rs, nC, nS, nR


# ============================================================================ 分解檢查 D（§3.2）
def t_model(K, model, fac, lay_code):
    Kd = copy.deepcopy(K)
    Kd["slow"]["b32"] = dict(r=B0_B32[model], w=B0_B32[model], k=1.0)
    v = BgView(Kd, "b32", fac, 65, 1)
    return v.restore([lay_code] * 64, list(range(64)), 64, 64, 0.0, 0, [0.0] * 64)[0]


def dcheck(model):
    use_model(model)
    K = constants(model)
    cell = load_b0(model)
    out = []
    for a, code in (("e2e_cake_bf16_cpu", 1), ("e2e_cake_int4_b32", 3)):
        ref = b0_stat(cell, "idle", "stream", a)
        t_idle = t_model(K, model, IDLE, code)
        for bg in BGS:
            for m in MODES:
                row = dict(model=model, action=a, bg=bg, mode=m)
                try:
                    fac = factors(cell, bg, m, "prim")
                    me = b0_stat(cell, bg, m, a)
                    if not (me["ok"] and ref["ok"]):
                        raise NotMeasured(f"{a} 的端到端 2σ 不過")
                    pred = t_model(K, model, fac, code) / t_idle
                    meas = me["t"] / ref["t"]
                    row.update(usable=True, pred=pred, meas=meas, err=pred / meas - 1)
                except NotMeasured as ex:
                    row.update(usable=False, why=str(ex))
                out.append(row)
    us = [r for r in out if r["usable"]]
    ok = [r for r in us if abs(r["err"]) <= 0.10]
    frac = len(ok) / len(us) if us else 0.0
    return dict(model=model, rows=out, n_usable=len(us), n_within=len(ok), frac=frac, path="prim" if us and frac >= 0.80 else "e2e")


# ============================================================================ trace 與提前量（§6）
def leads_conv(P, K, model):
    """x1_gpu.leads 的寫法，cpt 與 f(i) 換成這個模型的（Llama 時和 x1_gpu.leads 逐值相同，自測）。"""
    c = json.load(open(XG.CONSTS_X3))
    cpt = float(c[f"cpt_{model}_English"])
    cps = XG.WPM_DESKTOP * XG.CHARS_PER_WORD / 60.0
    f = XS.f_array(K, XG.NMAX_X3 + 1, "idle")
    rows, par, t = P["rows"], P["parentl"], P["tl"]
    out = [None] * len(rows)
    for r, kd in enumerate(P["kindc"]):
        if kd != 1:
            continue
        p = par[r]
        q, rr = rows[p], rows[r]
        new_raw = rr["input_length"] - q["input_length"] - q["output_length"]
        typing = max(new_raw - XG.TEMPLATE_OV, 0) * cpt / cps
        pp = par[p]
        p_cached = XG.common_prefix(rows[pp]["hash_ids"], q["hash_ids"]) * 512 if pp >= 0 else 0
        gen_prefill = max(0, q["input_length"] - p_cached)
        cp = XG.common_prefix(q["hash_ids"], rr["hash_ids"])
        n_old = min(cp * 512, q["input_length"] + q["output_length"])
        n_blk = min(XG.NMAX_X3, math.ceil(n_old / 512))
        think = max(t[r] - t[p] - gen_prefill / 512.0 * f[n_blk] - q["output_length"] * XG.TPOT, 0.0)
        out[r] = min(typing, think)
    return out


def leads_fi(P, week, model):
    """X3（ICPE C1）的 FreeInference 定義：使用者發起＝min(打字, gap)；工具觸發＝gap（c_tool）；gap < 0 → 0。"""
    import pyarrow.parquet as pq
    c = json.load(open(XG.CONSTS_X3))
    cpt = float(c[f"cpt_{model}_English"])
    cps = XG.WPM_DESKTOP * XG.CHARS_PER_WORD / 60.0
    t = pq.read_table(os.path.join(XF.FI_DIR, week, "processed", "requests.parquet"),
                      columns=["session_id", "step", "start_ms", "end_ms", "step_trigger", "role_tokens"]).to_pandas()
    t = t.sort_values(["start_ms", "session_id", "step"], kind="stable").reset_index(drop=True)   # 同 x1_fi.load_week
    assert len(t) == len(P["full"]), (week, len(t), len(P["full"]))
    tt = (t.start_ms.values.astype(float) - float(t.start_ms.iloc[0])) / 1000.0
    assert np.array_equal(tt, np.asarray(P["tl"])), f"{week}：和 x1_fi 的到達時間對不上（規則 6）"
    ut = np.array([(dict(x).get("user") if x is not None else None) for x in t.role_tokens], dtype=object)
    st, en = t.start_ms.values.astype(float), t.end_ms.values.astype(float)
    trig = t.step_trigger.values
    out = [None] * len(t)
    cnt = defaultdict(int)
    for r, kd in enumerate(P["kindc"]):
        if kd != 1:
            continue
        p = P["parentl"][r]
        gap = (st[r] - en[p]) / 1000.0
        if not gap >= 0:
            cnt["neg_gap"] += 1
            out[r] = 0.0
            continue
        if trig[r] == "tool":
            out[r] = gap
            cnt["tool"] += 1
        elif trig[r] == "user":
            if ut[r] is None or ut[p] is None:
                cnt["user_tok_missing"] += 1
                out[r] = 0.0
            else:
                out[r] = min(max(float(ut[r]) - float(ut[p]) - XG.TEMPLATE_OV, 0.0) * cpt / cps, gap)
                cnt["user"] += 1
        else:
            cnt["other_trigger"] += 1
            out[r] = 0.0
    return out, dict(cnt)


def replays(trace):
    """(名稱, P, 週) 的清單；FreeInference 每週一個獨立重播（同 X1）。"""
    if trace == "conversation":
        return [("conversation", None)]
    return [(f"fi:{w}", w) for w in XF.weeks()]


# ============================================================================ 一次重播的指標（§3.4、§4）
def window_stats(P, v, extra_mask=None):
    t = np.asarray(P["tl"], float)
    ttft = np.asarray(v.ttft, float)
    kind = np.asarray(P["kindc"])
    t0, t1 = float(t.min()), float(t.max())
    tm = (t0 + t1) / 2
    fac = v.fac
    rho = np.ones(len(t))
    if fac["rho"]:
        rmap = {0: fac["rho"]["recompute1"]}
        for s in (1, 2, 3):
            rmap[s] = fac["rho"][v.action(s)]
        rho = np.array([rmap[int(s)] for s in v.src])
    out = {}
    for half, (a, b) in (("dev", (t0, tm)), ("test", (tm, t1))):
        sel = (t >= a) & (t < b) if half == "dev" else (t >= a)
        ret = sel & (kind == 1)
        lo, hi = np.maximum(t, a), np.minimum(t + ttft, b)
        ln = np.clip(hi - lo, 0, None)
        # 聯集長度
        iv = sorted((float(x), float(y)) for x, y, l in zip(lo, hi, ln) if l > 0)
        U, cs, ce = 0.0, None, None
        for x, y in iv:
            if cs is None or x > ce:
                if cs is not None:
                    U += ce - cs
                cs, ce = x, y
            else:
                ce = max(ce, y)
        if cs is not None:
            U += ce - cs
        W = b - a
        rb = float((ln * rho).sum() / ln.sum()) if ln.sum() > 0 else 1.0
        phi = U / W if W > 0 else 0.0
        B = fac["cap"] * (1 - phi * (1 - rb))
        d = dict(sum=float(ttft[ret].sum()), n=int(ret.sum()), W=W, phi=phi, rho_bar=rb, B=B,
                 p95=float(np.percentile(ttft[ret], 95)) if ret.any() else None)
        if extra_mask is not None:
            um = ret & extra_mask
            d.update(sum_user=float(ttft[um].sum()), n_user=int(um.sum()))
        out[half] = d
    return out


# ============================================================================ run
def cmd_run(a):
    model, trace = a.model, a.trace
    use_model(model)
    K = constants(model)
    BF = BF_OF[model]
    K0K = dict(copy.deepcopy(K), int4_bytes=BF, dq_idle=0.0, dq_busy=0.0)
    cell = load_b0(model)
    dc = json.load(open(os.path.join(OUT, "dcheck.json")))
    path = dc[model]["path"]
    facs, nm = {}, {}
    for bg in BGS:
        for m in MODES:
            try:
                facs[(bg, m)] = factors(cell, bg, m, path)
            except NotMeasured as ex:
                nm[(bg, m)] = str(ex)
    rid = run_id()
    os.makedirs(os.path.join(OUT, "raw"), exist_ok=True)
    dest = os.path.join(OUT, "raw", f"e2e_{model}_{trace}.csv")
    rows = []
    for name, week in replays(trace):
        P = XS.prep(name, "R1") if trace == "conversation" else XS.prep(name)
        if trace == "conversation":
            lead, lcnt = leads_conv(P, K, model), {}
            umask = None
        else:
            lead, lcnt = leads_fi(P, week, model)
            umask = np.asarray(P["user_trigger"], bool)
        nmax, nreq = P["nmax"], len(P["full"])
        for slow in SLOWS:
            for cg in CPUS:
                cap = cg * GiB
                base = dict(run_id=rid, ts=ts(), model=model, trace=trace, replay=name, slow=slow, cpu_GiB=cg, path=path,
                            leads=json.dumps(lcnt, sort_keys=True))

                def emit(cfg, v):
                    ws = window_stats(P, v, umask)
                    for half, d in ws.items():
                        rows.append(dict(base, cfg=cfg, bg=v.fac["bg"], mode=v.fac["mode"], half=half, **d))
                # K0、K1：LRU，放置和時間無關 → 一次引擎跑所有背景（stream）
                live = [bg for bg in BGS if (bg, "stream") in facs]
                if live:
                    v0 = [BgView(K0K, slow, facs[(bg, "stream")], nmax, nreq, slow_bytes=BF, dq=0.0, k0=True) for bg in live]
                    XS.Engine(P, "LRU", cap, v0, K0K).run()
                    v1 = [BgView(K, slow, facs[(bg, "stream")], nmax, nreq) for bg in live]
                    XS.Engine(P, "LRU", cap, v1, K).run()
                    for v in v0:
                        assert v.i == nreq
                        emit("K0", v)
                    for v in v1:
                        assert v.i == nreq
                        emit("K1", v)
                for bg in BGS:
                    for m in MODES:
                        if (bg, m) not in facs:
                            continue
                        for cfg, pol in (("K2" if m == "stream" else "K3", "HINT-PF"), ("O", "O-PF")):
                            v = BgView(K, slow, facs[(bg, m)], nmax, nreq)
                            XG.make_engine(P, pol, cap, [v], K, lead).run()
                            assert v.i == nreq
                            emit(cfg, v)
                print(json.dumps(dict(replay=name, slow=slow, cpu=cg, rows=len(rows)), ensure_ascii=False), flush=True)
        _PC = XS._PCACHE
        _PC.clear()                                   # FreeInference 每週很大：用完就丟
    keys = list(rows[0].keys())
    with open(dest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    json.dump(dict(run_id=rid, ts=ts(), model=model, trace=trace, path=path, not_measured={f"{k[0]}|{k[1]}": v for k, v in nm.items()}),
              open(dest.replace(".csv", ".meta.json"), "w"), indent=1, ensure_ascii=False)
    print("→", dest, len(rows))


# ============================================================================ report（§4–§5）
def cmd_report(a):
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, "raw", "e2e_*.csv"))):
        rows += list(csv.DictReader(open(p)))
    assert rows, "沒有 E2E 的結果"
    agg = defaultdict(lambda: dict(sum=0.0, n=0, sum_user=0.0, n_user=0, BW=0.0, W=0.0, p95=[]))
    for r in rows:
        k = (r["model"], r["trace"], r["slow"], int(r["cpu_GiB"]), r["bg"], r["cfg"], r["mode"], r["half"])
        g = agg[k]
        g["sum"] += float(r["sum"]); g["n"] += int(r["n"])
        if r.get("n_user") not in (None, ""):
            g["sum_user"] += float(r["sum_user"]); g["n_user"] += int(r["n_user"])
        g["BW"] += float(r["B"]) * float(r["W"]); g["W"] += float(r["W"])
        if r.get("p95") not in (None, ""):
            g["p95"].append(float(r["p95"]))
    # P95（只報告，§5）：conversation 是單一重播的 P95；FreeInference 只有每週的 P95（沒有逐請求的列可合併）→ 報各週 P95 的中位數
    val = {k: dict(ttft=(g["sum"] / g["n"] if g["n"] else None), B=(g["BW"] / g["W"] if g["W"] else None),
                   ttft_user=(g["sum_user"] / g["n_user"] if g["n_user"] else None),
                   p95=(statistics.median(g["p95"]) if g["p95"] else None), n_p95=len(g["p95"])) for k, g in agg.items()}
    cells = [(m, t, s, c, b) for m in MODELS for t in TRACES for s in SLOWS for c in CPUS for b in BGS]

    def get(cellk, cfg, mode, half):
        return val.get(tuple(cellk) + (cfg, mode, half))
    # m*（§4）
    mstar = {}
    for m in MODELS:
        for b in BGS:
            devc = [c for c in cells if c[0] == m and c[4] == b]
            cand = []
            for mode in MPS_MODES:
                ok, gains = True, []
                for c in devc:
                    k2, k3 = get(c, "K2", "stream", "dev"), get(c, "K3", mode, "dev")
                    if not (k2 and k3 and k2["ttft"] and k3["ttft"] is not None):
                        ok = False
                        break
                    ok &= k3["B"] >= B_MIN
                    gains.append((k2["ttft"] - k3["ttft"]) / k2["ttft"])
                if ok and len(gains) == len(devc):
                    cand.append((statistics.median(gains), -MPS_MODES.index(mode), mode))
            mstar[f"{m}|{b}"] = max(cand)[2] if cand else None
    # 判定（§5）
    res, n_pass, n_ask, n_nm = [], 0, 0, 0
    per_model = defaultdict(int)
    for c in cells:
        k2 = get(c, "K2", "stream", "test")
        ms = mstar[f"{c[0]}|{c[4]}"]
        row = dict(model=c[0], trace=c[1], slow=c[2], cpu_GiB=c[3], bg=c[4], m_star=ms)
        if not k2 or k2["ttft"] is None:
            row.update(status="NOT_MEASURED")
            n_nm += 1
            res.append(row)
            continue
        if ms is None:
            k3 = dict(k2)
        else:
            k3 = get(c, "K3", ms, "test")
        gain = (k2["ttft"] - k3["ttft"]) / k2["ttft"]
        Bk3 = k3["B"]
        ok = gain > GO_GAIN and Bk3 >= B_MIN
        ask = gain >= ASK_GAIN and Bk3 >= B_MIN
        n_pass += ok
        n_ask += ask
        per_model[c[0]] += ok
        k0, k1 = get(c, "K0", "stream", "test"), get(c, "K1", "stream", "test")
        o = get(c, "O", ms or "stream", "test")
        row.update(status="PASS" if ok else "FAIL", gain_k3_vs_k2=gain, B_k3=Bk3,
                   ttft=dict(K0=k0 and k0["ttft"], K1=k1 and k1["ttft"], K2=k2["ttft"], K3=k3["ttft"], O=o and o["ttft"]),
                   B=dict(K0=k0 and k0["B"], K1=k1 and k1["B"], K2=k2["B"], K3=Bk3, O=o and o["B"]),
                   gain_k3_vs_k0=(k0["ttft"] - k3["ttft"]) / k0["ttft"] if k0 else None,
                   o_share=((k0["ttft"] - k3["ttft"]) / (k0["ttft"] - o["ttft"]) if (k0 and o and k0["ttft"] != o["ttft"]) else None),
                   pass_at_B090=bool(gain > GO_GAIN and Bk3 >= 0.90),
                   ttft_user=dict(K2=k2["ttft_user"], K3=k3["ttft_user"]) if c[1] == "freeinference" else None)
        # 只報告（§5）：P95 與開發集的數字；不進判定
        dv = {cfg: get(c, cfg, md, "dev") for cfg, md in (("K0", "stream"), ("K1", "stream"), ("K2", "stream"),
                                                           ("K3", ms or "stream"), ("O", ms or "stream"))}
        if ms is None:
            dv["K3"] = dv["K2"]
        row.update(p95=dict(kind="replay" if c[1] == "conversation" else "median_of_weekly",
                            K2=k2["p95"], K3=k3["p95"], K0=k0 and k0["p95"], O=o and o["p95"]),
                   dev=dict(ttft={x: y and y["ttft"] for x, y in dv.items()}, B={x: y and y["B"] for x, y in dv.items()},
                            gain_k3_vs_k2=((dv["K2"]["ttft"] - dv["K3"]["ttft"]) / dv["K2"]["ttft"]
                                           if dv["K2"] and dv["K3"] and dv["K2"]["ttft"] else None)))
        res.append(row)
    if n_nm > GO_CELLS:
        v = "NOT_DETERMINED"
    elif n_pass >= GO_CELLS and all(per_model[m] >= 1 for m in MODELS):
        v = "E2E-GO"
    elif n_ask >= GO_CELLS:
        v = "E2E-ASK"
    else:
        v = "E2E-NOGO"
    doc = dict(run_id=run_id(), ts=ts(), prereg="docs/stage3_design_20261024/runs/E2E_combo.md", verdict=v, n_pass=n_pass,
               n_ask_level=n_ask, n_not_measured=n_nm, pass_per_model=dict(per_model), m_star=mstar,
               dcheck=json.load(open(os.path.join(OUT, "dcheck.json"))), cells=res, sources=sorted({r["run_id"] for r in rows}))
    json.dump(doc, open(os.path.join(OUT, "e2e_report.json"), "w"), indent=1, ensure_ascii=False, default=str)
    print(json.dumps(dict(verdict=v, n_pass=n_pass, n_ask=n_ask, n_nm=n_nm, per_model=dict(per_model), m_star=mstar), ensure_ascii=False))


def cmd_dcheck(a):
    os.makedirs(OUT, exist_ok=True)
    doc = {m: dcheck(m) for m in MODELS}
    doc.update(run_id=run_id(), ts=ts())
    json.dump(doc, open(os.path.join(OUT, "dcheck.json"), "w"), indent=1, ensure_ascii=False, default=str)
    print(json.dumps({m: dict(path=doc[m]["path"], n_usable=doc[m]["n_usable"], n_within=doc[m]["n_within"]) for m in MODELS}))


# ============================================================================ 自測
def self_test() -> bool:
    ok = True

    def chk(name, cond):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name, flush=True)
        ok &= bool(cond)
    use_model("llama")
    K = XS.constants()
    P = XS.prep("conversation", "R1")
    nmax, nreq = P["nmax"], len(P["full"])
    L0 = XG.leads(P, K)
    L1 = leads_conv(P, K, "llama")
    chk("leads_conv（Llama 參數）＝x1_gpu.leads 逐值相同", L0 == L1)
    for pol in ("LRU", "HINT-PF"):
        for slow in ("ssd", "nfs"):
            a = XS.View(K, slow, "idle", "cake", nmax, nreq)
            b = BgView(K, slow, IDLE, nmax, nreq)
            XG.make_engine(P, pol, 128 * GiB, [a], K, L0).run()
            XG.make_engine(P, pol, 128 * GiB, [b], K, L0).run()
            chk(f"閒時 BgView ≡ View（{pol}、{slow}、128 GiB；{nreq} 個請求逐位元）", np.array_equal(a.ttft, b.ttft) and b.i == nreq)
    for m in MODELS:
        cell = load_b0(m)
        chk(f"B0 {m}：每塊位元組交叉驗算過、有 idle 參考", b0_stat(cell, "idle", "stream", "recompute1")["t"] is not None)
    Q = constants("qwen")
    chk("Qwen 常數：f 64 點、INT4 9,175,040 B、d > 0", len(Q["f_ms"]) == 64 and Q["int4_bytes"] == 9175040 and Q["dq_idle"] > 0)
    use_model("llama")
    print("self-test", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("dcheck")
    s = sub.add_parser("run")
    s.add_argument("--model", required=True, choices=MODELS)
    s.add_argument("--trace", required=True, choices=TRACES)
    sub.add_parser("report")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    {"dcheck": cmd_dcheck, "run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
