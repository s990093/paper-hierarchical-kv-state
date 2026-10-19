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
from formats import CFG, cpu_busy_factor, cpu_factor_of, load_costs, load_format_table  # noqa: E402
import ledger  # noqa: E402
from policies import TIER_NAMES, TierRec, make_policy  # noqa: E402

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
        cf = cpu_factor_of(plat, cpu)        # NOT_MEASURED → 丟錯，不跑
        mf = mf_by_platform[plat]
        vs.append(View(costs, mf, gpu, mode, P["nmax"], len(P["full"]), cpu_factor=cf, ssd_share=share,
                       label=f"{plat}|{prof}|gpu={gpu}|cpu={cpu}|ssd/{share}|{mode}", record=record))
    return vs


def _tier_stats(rec, eng2, n_mismatch, approx):
    """O-tier 兩趟的對照欄位（06 §26）。session 引擎：CPU 側事件必須完全相同（不同就在呼叫端丟錯，所以列上永遠是 0）；
    block 引擎（和時間有關）：數兩趟摘要不同的請求個數，標 otier_approx。"""
    nw = sum(eng2._nw.values())
    return dict(otier_pass1_ssd_writes=rec.n_write, otier_pass2_ssd_writes=nw, otier_ssd_copies_never_read=rec.n_write - len(rec.ssd_read),
                otier_cpu_event_mismatch_requests=int(n_mismatch), otier_approx=bool(approx))


def _digest_compare(d1, d2, label, strict):
    """兩趟的逐請求摘要比對；strict＝不同就丟 AssertionError（列出第一個不同的請求）。回傳不同的請求數。"""
    import numpy as np
    a, b = np.asarray(d1), np.asarray(d2)
    if len(a) != len(b):
        if strict:
            raise AssertionError(f"{label}: 兩趟的請求數不同 {len(a)} ≠ {len(b)}")
        return max(len(a), len(b))
    bad = np.nonzero(a != b)[0]
    if strict and len(bad):
        raise AssertionError(f"{label}: O-tier 兩趟的 CPU 側事件不相同：{len(bad)} 個請求的摘要不同，第一個是請求 {int(bad[0])}")
    return int(len(bad))


def _sess_meta(P, code, comp, cpu_GiB, model, H, mf, int2_on_read, serve, unverified_len="prefix"):
    return dict(source=P["meta"]["source"], variant=P["meta"]["variant"], outer=code, serve=serve, policy=comp.name, model=model,
                cpu_GiB=cpu_GiB, H_s=H, format_table=mf.src["path"], format_table_sha256=mf.src["sha256"], int2_on_read=int2_on_read,
                unverified_len=unverified_len, dq_set=mf.src.get("dq_set"), run_id=os.environ.get("RUN_ID", "adhoc"))


def run_cell(P, code, policy, cpu_GiB, model, specs, T=None, H=0.0, fmt_path=None, check=False, pol_kw=None,
             policy_obj=None, int2_on_read="serve", sess_out=None, unverified_len="prefix", H_extra=0.0, quality_rule="strict",
             recompute_charge="none"):
    """sess_out：給路徑就把每個 session 的帳寫成 npz（06 §12.1）。int2_on_read、unverified_len 見 engine.Engine。
    unverified_len 也決定 oracle 的讀取長度定義（pol_kw 沒給 o_read_len 時跟著它；06 §18）。"""
    t0 = time.time()
    pol_kw = dict(pol_kw or {})
    pol_kw.setdefault("o_read_len", unverified_len)
    plats = sorted({s[0] for s in specs})
    mfs = {p: load_format_table(model, p, fmt_path) for p in plats}
    assert quality_rule in ("strict", "loose"), quality_rule
    if quality_rule == "loose":      # 04 v4 §3.3 寬鬆版雙胞胎（只報告）：整套（O 與所有對手）用同一個寬鬆品質規則
        mfs = {p: m.loosened() for p, m in mfs.items()}
    mf0 = mfs[plats[0]]
    for p in plats:
        assert mfs[p].bytes == mf0.bytes, "不同平台的格式大小必須相同（放置軌跡共用）"
    oc = outer_cfg(code, T)
    tier = policy_obj is None and policy in TIER_NAMES
    rec = eng1 = None
    if tier:       # O-tier：第 1 趟＝一般的 O（只記錄，不需要時間設定）；第 2 趟＝tier 版（CPU 側決定相同，SSD 這一階查第 1 趟的記錄）
        rec = TierRec(True)
        comp1 = make_policy(TIER_NAMES[policy], mf0, **pol_kw)
        eng1 = Engine(P, oc, comp1, int(cpu_GiB * GiB), [], mf0, H=H, check=check, int2_on_read=int2_on_read,
                      unverified_len=unverified_len, H_extra=H_extra, tier_rec=rec, ev_log=True).run()
        rec.recording = False
        pol_kw2 = dict(pol_kw, tier_rec=rec, tier_strict=True)
        comp = make_policy(policy, mf0, **pol_kw2)
    else:
        comp = policy_obj or make_policy(policy, mf0, **pol_kw)
    views = make_views(P, model, specs, mfs)
    eng = Engine(P, oc, comp, int(cpu_GiB * GiB), views, mf0, H=H, check=check, int2_on_read=int2_on_read,
                 unverified_len=unverified_len, H_extra=H_extra, ev_log=tier).run()
    er = ledger.engine_row(eng)
    if tier:
        nm = _digest_compare(eng1.ev_digest, eng.ev_digest, f"{policy}", strict=True)
        er.update(_tier_stats(rec, eng, nm, approx=False))
        assert comp.n_tier_missing == 0
    rows, qtabs, ids = [], [], []
    for v in views:
        row, qt = ledger.summarize_view(P, eng, v, H, recompute_charge)
        rows.append(dict(source=P["meta"]["source"], variant=P["meta"]["variant"], data_tier=P["meta"]["tier"],
                         outer=code + (f"-T{int(T)}" if T else ""), policy=comp.name, model=model, cpu_GiB=cpu_GiB, H_s=H,
                         format_table=mf0.src["path"], format_table_sha256=mf0.src["sha256"], format_placeholder=mf0.placeholder,
                         format_table_kind=mf0.src.get("kind"), format_dq_set=mf0.src.get("dq_set"),
                         n_req=P["meta"]["n_req"], n_sess=P["meta"]["n_sess"], **row, **er, elapsed_s=time.time() - t0))
        for q in qt:
            qtabs.append(dict(outer=rows[-1]["outer"], policy=comp.name, cpu_GiB=cpu_GiB, **q))
        for (n, a, b, ok) in ledger.accounting_identities(P, eng, v):
            ids.append(dict(outer=rows[-1]["outer"], policy=comp.name, cpu_GiB=cpu_GiB, view=v.label, identity=n, lhs=a, rhs=b, ok=ok))
    if sess_out:
        ledger.write_per_session(sess_out, P, eng, views, _sess_meta(P, rows[-1]["outer"], comp, cpu_GiB, model, H, mf0,
                                                                      int2_on_read, "trace", unverified_len))
    return rows, qtabs, ids, eng, views


def run_cell_fifo1(P, code, policy, cpu_GiB, model, spec, H=0.0, fmt_path=None, pol_kw=None, policy_obj=None,
                   int2_on_read="serve", sess_out=None, reqs=None, unverified_len="prefix", H_extra=0.0, quality_rule="strict",
                   recompute_charge="none"):
    """serve=fifo1（一張 GPU、FIFO）：block 引擎，E1／E2／E6，單一時間設定 spec＝(platform, gpu, cpu, ssd_share, 'cake', ssd_profile)。"""
    t0 = time.time()
    plat, gpu, cpu, share, mode, prof = spec
    assert mode == "cake"
    mf = load_format_table(model, plat, fmt_path)
    assert quality_rule in ("strict", "loose"), quality_rule
    if quality_rule == "loose":
        mf = mf.loosened()
    pol_kw = dict(pol_kw or {})
    pol_kw.setdefault("o_read_len", unverified_len)
    oc = outer_cfg(code)
    costs = load_costs(plat, model, prof)
    cf = cpu_factor_of(plat, cpu)
    reqs = reqs_from_P(P) if reqs is None else reqs
    nmax = max(max(len(q["h"]) for q in reqs), max(P["nall"])) + 1
    d2h = costs.cpu["w"] * GiB          # capture＝GPU→CPU：用 CPU 層的寫入速度（X4 用 C0 的 D2H 44.8 GiB/s；06 §8）

    def mkview():
        return BView(costs, mf, gpu, nmax, cpu_factor=cf, ssd_share=share, d2h_Bps=d2h,
                     label=f"{plat}|{prof}|gpu={gpu}|cpu={cpu}|ssd/{share}|cake|fifo1")
    tier = policy_obj is None and policy in TIER_NAMES
    rec = eng1 = None
    if tier:       # O-tier：block 引擎的放置和時間有關（佇列預取），所以第 2 趟的 CPU 側事件只能近似相同：數不同的請求數、標 approx
        rec = TierRec(True)
        comp1 = make_policy(TIER_NAMES[policy], mf, **pol_kw)
        eng1 = BlockEngine(reqs, oc, comp1, int(cpu_GiB * GiB), mkview(), mf, qp=bool(oc.get("queue_prefetch")), H=H,
                           obs_end=P.get("obs_end"), int2_on_read=int2_on_read, unverified_len=unverified_len, H_extra=H_extra,
                           tier_rec=rec, ev_log=True).run_events()
        rec.recording = False
        comp = make_policy(policy, mf, **dict(pol_kw, tier_rec=rec, tier_strict=False))
    else:
        comp = policy_obj or make_policy(policy, mf, **pol_kw)
    v = mkview()
    eng = BlockEngine(reqs, oc, comp, int(cpu_GiB * GiB), v, mf, qp=bool(oc.get("queue_prefetch")), H=H,
                      obs_end=P.get("obs_end"), int2_on_read=int2_on_read,
                      unverified_len=unverified_len, H_extra=H_extra, ev_log=tier).run_events()
    er = ledger.engine_row(eng)
    if tier:
        nm = _digest_compare(eng1.ev_digest, eng.ev_digest, f"{policy}(block)", strict=False)
        er.update(_tier_stats(rec, eng, nm, approx=True))
        er["otier_tier_missing_keys"] = comp.n_tier_missing
    row, qt = ledger.summarize_view(P, eng, v, H, recompute_charge)
    import numpy as np
    ret = np.asarray(P["kindc"]) == 1
    row.update(qwait_mean_s=float(v.qwait[ret].mean()) if ret.any() else float("nan"),
               qwait_p99_s=float(np.percentile(v.qwait[ret], 99)) if ret.any() else float("nan"),
               gpu_busy_frac=eng.busy / max(P["tl"][-1] - P["tl"][0], 1e-9))
    rows = [dict(source=P["meta"]["source"], variant=P["meta"]["variant"], data_tier=P["meta"]["tier"], outer=code, serve="fifo1",
                 policy=comp.name, model=model, cpu_GiB=cpu_GiB, H_s=H, format_table=mf.src["path"],
                 format_table_sha256=mf.src["sha256"], format_placeholder=mf.placeholder, format_table_kind=mf.src.get("kind"),
                 format_dq_set=mf.src.get("dq_set"),
                 n_req=P["meta"]["n_req"],
                 n_sess=P["meta"]["n_sess"], **row, **er, elapsed_s=time.time() - t0)]
    qtabs = [dict(outer=code, policy=comp.name, cpu_GiB=cpu_GiB, **q) for q in qt]
    ids = [dict(outer=code, policy=comp.name, cpu_GiB=cpu_GiB, view=v.label, identity=n, lhs=a, rhs=b, ok=ok)
           for (n, a, b, ok) in ledger.accounting_identities(P, eng, v)]
    if sess_out:
        ledger.write_per_session(sess_out, P, eng, [v], _sess_meta(P, code, comp, cpu_GiB, model, H, mf, int2_on_read, "fifo1", unverified_len))
    return rows, qtabs, ids, eng, [v]
