"""sim.py — 跑一格：(trace, 外層做法, 壓縮做法, CPU 容量, 模型, 一組時間設定) → 每個時間設定一列＋品質帳＋記帳恆等式。

外層做法的參數從 config/outer_policies.json 讀；serve=trace 用 engine.Engine（E1–E5；E6≡E2），serve=fifo1 用 engine_block（E2／E6）。
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from engine import Engine, View  # noqa: E402
from engine_block import BlockEngine, BView, reqs_from_P  # noqa: E402
from formats import CFG, cpu_busy_factor, load_costs, load_format_table  # noqa: E402
import ledger  # noqa: E402
from policies import make_policy  # noqa: E402

GiB = 1 << 30
OUTER = json.load(open(os.path.join(CFG, "outer_policies.json"), encoding="utf-8"))


def outer_cfg(code, T=None):
    """E5 要給 T（秒）；回傳引擎用的 dict。"""
    u = dict(OUTER[code]["used"])
    u["code"] = code
    if code == "E5":
        assert T in u["idle_demote_s"], f"E5 的 T 要是 {u['idle_demote_s']} 之一"
        u["idle_T"] = float(T)
    else:
        u["idle_T"] = None
    return u


def make_views(P, model, specs, mf_by_platform, record=True):
    """specs：[(platform, gpu, cpu ('idle'|'busy'), ssd_share, mode, ssd_profile)]。"""
    vs = []
    for (plat, gpu, cpu, share, mode, prof) in specs:
        costs = load_costs(plat, model, prof)
        cf = 1.0 if cpu == "idle" else cpu_busy_factor(plat)        # NOT_MEASURED → 丟錯，不跑
        mf = mf_by_platform[plat]
        vs.append(View(costs, mf, gpu, mode, P["nmax"], len(P["full"]), cpu_factor=cf, ssd_share=share,
                       label=f"{plat}|{prof}|gpu={gpu}|cpu={cpu}|ssd/{share}|{mode}", record=record))
    return vs


def run_cell(P, code, policy, cpu_GiB, model, specs, T=None, H=0.0, fmt_path=None, check=False, pol_kw=None,
             policy_obj=None):
    t0 = time.time()
    plats = sorted({s[0] for s in specs})
    mfs = {p: load_format_table(model, p, fmt_path) for p in plats}
    mf0 = mfs[plats[0]]
    for p in plats:
        assert mfs[p].bytes == mf0.bytes, "不同平台的格式大小必須相同（放置軌跡共用）"
    comp = policy_obj or make_policy(policy, mf0, **(pol_kw or {}))
    oc = outer_cfg(code, T)
    views = make_views(P, model, specs, mfs)
    eng = Engine(P, oc, comp, int(cpu_GiB * GiB), views, mf0, H=H, check=check).run()
    er = ledger.engine_row(eng)
    rows, qtabs, ids = [], [], []
    for v in views:
        row, qt = ledger.summarize_view(P, eng, v, H)
        rows.append(dict(source=P["meta"]["source"], variant=P["meta"]["variant"], data_tier=P["meta"]["tier"],
                         outer=code + (f"-T{int(T)}" if T else ""), policy=comp.name, model=model, cpu_GiB=cpu_GiB, H_s=H,
                         format_table=mf0.src["path"], format_table_sha256=mf0.src["sha256"], format_placeholder=mf0.placeholder,
                         n_req=P["meta"]["n_req"], n_sess=P["meta"]["n_sess"], **row, **er, elapsed_s=time.time() - t0))
        for q in qt:
            qtabs.append(dict(outer=rows[-1]["outer"], policy=comp.name, cpu_GiB=cpu_GiB, **q))
        for (n, a, b, ok) in ledger.accounting_identities(P, eng, v):
            ids.append(dict(outer=rows[-1]["outer"], policy=comp.name, cpu_GiB=cpu_GiB, view=v.label, identity=n, lhs=a, rhs=b, ok=ok))
    return rows, qtabs, ids, eng, views


def run_cell_fifo1(P, code, policy, cpu_GiB, model, spec, H=0.0, fmt_path=None, pol_kw=None, policy_obj=None):
    """serve=fifo1（一張 GPU、FIFO）：block 引擎，E1／E2／E6，單一時間設定 spec＝(platform, gpu, cpu, ssd_share, 'cake', ssd_profile)。"""
    t0 = time.time()
    plat, gpu, cpu, share, mode, prof = spec
    assert mode == "cake"
    mf = load_format_table(model, plat, fmt_path)
    comp = policy_obj or make_policy(policy, mf, **(pol_kw or {}))
    oc = outer_cfg(code)
    costs = load_costs(plat, model, prof)
    cf = 1.0 if cpu == "idle" else cpu_busy_factor(plat)
    reqs = reqs_from_P(P)
    nmax = max(max(len(q["h"]) for q in reqs), max(P["nall"])) + 1
    d2h = costs.cpu["w"] * GiB          # capture＝GPU→CPU：用 CPU 層的寫入速度（X4 用 C0 的 D2H 44.8 GiB/s；06 §8）
    v = BView(costs, mf, gpu, nmax, cpu_factor=cf, ssd_share=share, d2h_Bps=d2h,
              label=f"{plat}|{prof}|gpu={gpu}|cpu={cpu}|ssd/{share}|cake|fifo1")
    eng = BlockEngine(reqs, oc, comp, int(cpu_GiB * GiB), v, mf, qp=bool(oc.get("queue_prefetch")), H=H,
                      obs_end=P.get("obs_end")).run_events()
    er = ledger.engine_row(eng)
    row, qt = ledger.summarize_view(P, eng, v, H)
    import numpy as np
    ret = np.asarray(P["kindc"]) == 1
    row.update(qwait_mean_s=float(v.qwait[ret].mean()) if ret.any() else float("nan"),
               qwait_p99_s=float(np.percentile(v.qwait[ret], 99)) if ret.any() else float("nan"),
               gpu_busy_frac=eng.busy / max(P["tl"][-1] - P["tl"][0], 1e-9))
    rows = [dict(source=P["meta"]["source"], variant=P["meta"]["variant"], data_tier=P["meta"]["tier"], outer=code, serve="fifo1",
                 policy=comp.name, model=model, cpu_GiB=cpu_GiB, H_s=H, format_table=mf.src["path"],
                 format_table_sha256=mf.src["sha256"], format_placeholder=mf.placeholder, n_req=P["meta"]["n_req"],
                 n_sess=P["meta"]["n_sess"], **row, **er, elapsed_s=time.time() - t0)]
    qtabs = [dict(outer=code, policy=comp.name, cpu_GiB=cpu_GiB, **q) for q in qt]
    ids = [dict(outer=code, policy=comp.name, cpu_GiB=cpu_GiB, view=v.label, identity=n, lhs=a, rhs=b, ok=ok)
           for (n, a, b, ok) in ledger.accounting_identities(P, eng, v)]
    return rows, qtabs, ids, eng, [v]
