"""test_p3_sim.py — 統一模擬器的不變量（docs/phase3_20261017/06_simulator.md §6）。只用 CPU。

  (i)   C0／C1＋E2＋不預取，在 X1 的設定上逐位元重現 X1 的 LRU（另加 E1＝X1 的 LRU+NONE）
  (ii)  一張 GPU、一次服務一個請求（X4 的事件引擎）時，E6 的佇列預取逐位元重現 X4 的 QP（E2＝X4 的 NP）；
        serve=trace（沒有佇列）時 E6≡E2
  (iii) O 的主指標不比任何「品質帳為 0」的 C 差，而且 O 的品質帳（fail）為 0；違反的格子寫進 $RUN_DIR，不藏
  (iv)  O 的容量單調（CPU 越大，SSD 讀寫總量不增）；違反照錄
  (v)   同設定、同 seed 逐位元重現（含疊加、session 抽樣）；不同 seed 會不同
  另外：記帳恆等式、品質來歷、截尾、預測器拿不到標籤、和 X8 的成本模型對拍。

**不讀 placeholder 格式表**：每個測試用 _test_format_table() 造的表（數值只為了測程式，和真實無關）。
執行：TIARA_NO_GPU=1 bash code/run3090.sh p3sim-test python -m pytest -q code/p3_sim/test_p3_sim.py
"""
from __future__ import annotations

import copy
import csv
import json
import math
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(CODE, "p3x"))
sys.path.insert(0, CODE)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import sim  # noqa: E402
import trace as TR  # noqa: E402
from engine import W_STORE, Engine, View  # noqa: E402
from engine_block import BlockEngine, BView  # noqa: E402
from formats import FID, Costs, ModelFormats, load_costs, load_format_table  # noqa: E402
from policies import PredictorPolicy, make_policy  # noqa: E402

GiB = 1 << 30
BF = 64 << 20
OUT = os.environ.get("RUN_DIR") or os.path.join(os.environ.get("TMPDIR", "/tmp"), "p3sim_test_adhoc")
os.makedirs(OUT, exist_ok=True)
SPECS = [("mi300x", "idle", "idle", 1, "cake", "mi300x_local"), ("mi300x", "busy", "idle", 4, "cake", "mi300x_local"),
         ("mi300x", "idle", "idle", 1, "load", "mi300x_local")]
FMT_TABLE = {}


def _write_rows(name, rows):
    if not rows:
        return
    p = os.path.join(OUT, name)
    keys = sorted({k for r in rows for k in r})
    new = not os.path.exists(p)
    with open(p, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run_id"] + keys)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(dict(run_id=os.environ.get("RUN_ID", "adhoc"), **r))


def _test_format_table(tmpdir):
    """測試用的格式表：數值只為了測程式（不是第二階段的實測）。8bit 每個長度都過；int4 <16K 過、≥16K 不過；int2 只給不再被讀的。"""
    p = os.path.join(tmpdir, "format_table.TEST.json")
    if p in FMT_TABLE:
        return p
    bins = [0, 16384, 1 << 40]
    d = dict(PLACEHOLDER=False, TEST_ONLY=True, run_id="test", len_bins_tokens=bins, never_read_only=["int2"],
             quality={"llama": {"bf16": ["pass", "pass"], "8bit": ["pass", "pass"], "int4": ["pass", "fail"], "int2": ["fail", "fail"]}},
             models={"llama": dict(kv_bytes_per_token_bf16=131072, block_tokens=512, formats={
                 "bf16": dict(bytes_per_block=BF, dequant_s_per_block={"mi300x": {"idle": 0.0, "busy": 0.0}}),
                 "8bit": dict(bytes_per_block=33 << 20, dequant_s_per_block={"mi300x": {"idle": 0.0002, "busy": 0.0008}}),
                 "int4": dict(bytes_per_block=20 << 20, dequant_s_per_block={"mi300x": {"idle": 0.0004, "busy": 0.0016}}),
                 "int2": dict(bytes_per_block=12 << 20, dequant_s_per_block={"mi300x": {"idle": 0.0005, "busy": 0.002}})})})
    json.dump(d, open(p, "w"))
    FMT_TABLE[p] = True
    return p


@pytest.fixture(scope="session")
def ft(tmp_path_factory):
    return _test_format_table(str(tmp_path_factory.mktemp("ft")))


_PC = {}


def _P(src, var, a, d, **kw):
    k = (src, var, a, d, tuple(sorted(kw.items())))
    if k not in _PC:
        _PC[k] = TR.prepare(src, var, a, d, **kw)
    return _PC[k]


# ===================================================================== (i) X1
def _mk_mf_x1(i4, dqi, dqb):
    return ModelFormats("llama", "x1", [None, BF, None, i4, None], {"idle": [None, 0.0, None, dqi, None], "busy": [None, 0.0, None, dqb, None]},
                        [0, 1 << 40], {}, [], False, {"path": "test", "sha256": "-"})


@pytest.mark.parametrize("trace,rule,cap", [("conversation", "R1", 128), ("conversation", "R1", 1024),
                                            ("conversation", "R2", 256), ("toolagent", "R1", 128)])
def test_i_x1_lru_bitwise(trace, rule, cap):
    import x1_sim
    K = x1_sim.constants()
    P = x1_sim.prep(trace, rule)
    out = []
    for comp_name, x1pol, code, Kx in (("C1", "LRU", "E2", K),
                                       ("C0", "LRU", "E2", dict(K, int4_bytes=BF, dq_idle=0.0, dq_busy=0.0)),
                                       ("C1", "LRU+NONE", "E1", K)):
        V1 = x1_sim.make_views(Kx, P, x1_sim.ALL_VIEWS)
        x1_sim.Engine(P, x1pol, cap * GiB, V1, Kx).run()
        mf = _mk_mf_x1(Kx["int4_bytes"], Kx["dq_idle"], Kx["dq_busy"])
        V2 = [View(Costs(Kx["f_ms"], Kx["f_slope"], Kx["f_icpt"], Kx["busy_factor"], Kx["cpu"], Kx["slow"][s], "x1"), mf, g, m,
                   P["nmax"], len(P["full"])) for (s, g, m) in x1_sim.ALL_VIEWS]
        Engine(P, sim.outer_cfg(code), make_policy(comp_name, mf), cap * GiB, V2, mf).run()
        for (s, g, m), a, b in zip(x1_sim.ALL_VIEWS, V1, V2):
            ok = (np.array_equal(a.ttft, b.ttft) and np.array_equal(a.rest, b.rest) and np.array_equal(a.nC, b.nC)
                  and np.array_equal(a.nS, b.nS) and np.array_equal(a.nR, b.nR) and a.slow.bytes_w == b.slow.bytes_w
                  and a.slow.bytes_r == b.slow.bytes_r and a.cpu.bytes_w == b.cpu.bytes_w)
            out.append(dict(test="i", trace=trace, rule=rule, cap_GiB=cap, x1_policy=x1pol, p3sim=f"{code}+{comp_name}", view=f"{s}|{g}|{m}",
                            n_req=len(P["full"]), bitwise=ok, max_abs_ttft_diff=float(np.max(np.abs(a.ttft - b.ttft)))))
    _write_rows("test_i_x1.csv", out)
    bad = [r for r in out if not r["bitwise"]]
    assert not bad, bad


# ===================================================================== (ii) X4
@pytest.mark.parametrize("trace,p,seed,capG,slow", [("conversation", 0.2, 0, 32, "local"), ("conversation", 0.3, 1, 64, "nfs"),
                                                    ("conversation", 0.4, 2, 128, "b32"), ("toolagent", 0.2, 0, 64, "local")])
def test_ii_x4_qp_bitwise(trace, p, seed, capG, slow):
    import x4_sim
    P4 = x4_sim.params()
    f, _, _, _ = x4_sim.f_ext()
    CB = x4_sim.CB
    mf = ModelFormats("llama", "x4", [None, CB, None, None, None], {"idle": [None, 0.0, None, None, None], "busy": [None, 0.0, None, None, None]},
                      [0, 1 << 40], {}, [], False, {"path": "test", "sha256": "-"})
    reqs, ns, ws = x4_sim._get_subset(trace, p, seed, "loose")
    cap = int(round(capG * GiB / CB))
    out = []
    for pol, qp, code in (("NP", False, "E2"), ("QP", True, "E6")):
        s4 = x4_sim.Sim(reqs, cap, x4_sim.mk_tiers(P4, slow), f, qp=qp, ip=None, promote=True)
        rows, busy = s4.run_events()
        c = dict(r=P4["cpu"]["read_GiBps"], w=P4["cpu"]["write_GiBps"], k=P4["cpu"]["k_read"])
        s = dict(r=P4[slow]["read_GiBps"], w=P4[slow]["write_GiBps"], k=P4[slow]["k_read"])
        v = BView(Costs([1.0], 0.0, 0.0, 1.0, c, s, "x4"), mf, "idle", 0, d2h_Bps=x4_sim.D2H_BPS, f=f)
        e = BlockEngine(copy.deepcopy(reqs), sim.outer_cfg(code), make_policy("C0", mf), cap * CB, v, mf, qp=qp, x4_prefill=True).run_events()
        a = np.array([r["ttft"] for r in rows])
        qa = np.array([r["qwait"] for r in rows])
        ok = (np.array_equal(a, v.ttft) and np.array_equal(qa, v.qwait) and s4.S["qp_reads"] == e.S["qp_reads"]
              and s4.S["demote"] == e.S["demote"] and s4.S["qp_stop_guard"] == e.S["qp_stop_guard"]
              and s4.S["pf_used"] == e.S["pf_used"] and s4.S["pf_evicted"] == e.S["pf_evicted"] and s4.S["promoted"] == e.S["promoted"])
        out.append(dict(test="ii", trace=trace, p=p, seed=seed, cap_GiB=capG, slow=slow, x4_policy=pol, p3sim=code, n_req=len(reqs),
                        bitwise=ok, max_abs_ttft_diff=float(np.max(np.abs(a - v.ttft))), qp_reads=e.S["qp_reads"], demote=e.S["demote"],
                        x4_gpu_util=busy / (rows[-1]["T"] - rows[0]["t_arr"])))
    _write_rows("test_ii_x4.csv", out)
    bad = [r for r in out if not r["bitwise"]]
    assert not bad, bad


def test_ii_trace_mode_e6_equals_e2(ft):
    P = _P("mooncake", "conv-rule-strict", 0, 900)
    r2, _, _, _, v2 = sim.run_cell(P, "E2", "C1", 32, "llama", SPECS, fmt_path=ft)
    r6, _, _, _, v6 = sim.run_cell(P, "E6", "C1", 32, "llama", SPECS, fmt_path=ft)
    for a, b in zip(v2, v6):
        assert np.array_equal(a.ttft, b.ttft) and a.slow.bytes_r == b.slow.bytes_r and a.slow.bytes_w == b.slow.bytes_w


# ===================================================================== (iii)、(iv) O
DATA = [("mooncake", "conv-rule-strict", 0, 1800, {}), ("bailian", "coder", 0, 1800, {}), ("synth", "small-seed1", 0, 4 * 3600, {})]
OUTERS = [("E2", None), ("E3", None), ("E4", None), ("E5", 300)]


def _cells(ft, src, var, a, d, kw, caps, H):
    P = _P(src, var, a, d, **kw)
    res = {}
    for code, T in OUTERS:
        for cap in caps:
            for pol in ("C0", "C1", "C2", "C3", "C4", "O"):
                rows, q, ids, eng, vs = sim.run_cell(P, code, pol, cap, "llama", SPECS[:1], T=T, H=H, fmt_path=ft, check=200)
                r = rows[0]
                assert all(i["ok"] for i in ids), [i for i in ids if not i["ok"]]
                res[(code, cap, pol)] = r
    return res


@pytest.mark.parametrize("src,var,a,d,kw", DATA)
@pytest.mark.parametrize("H", [0.0, 600.0])
def test_iii_oracle_dominates_quality_ok_fixed(ft, src, var, a, d, kw, H):
    res = _cells(ft, src, var, a, d, kw, [16, 64], H)
    out, viol = [], []
    for (code, cap, pol), r in res.items():
        if pol != "O":
            continue
        assert r["reads_fail"] == 0, f"O 的品質帳不是 0：{code} {cap} {r['reads_fail']}"
        for c in ("C0", "C1", "C2", "C3", "C4"):
            rc = res[(code, cap, c)]
            for metric in ("ssd_rw_GiB", "ssd_rw_p99_GiBps"):
                row = dict(test="iii", source=src, variant=var, H_s=H, outer=code, cap_GiB=cap, rival=c, metric=metric,
                           O=r[metric], rival_value=rc[metric], rival_reads_fail=rc["reads_fail"],
                           rival_quality_ok=rc["reads_fail"] == 0, O_le_rival=r[metric] <= rc[metric] + 1e-12)
                out.append(row)
                if row["rival_quality_ok"] and not row["O_le_rival"] and metric == "ssd_rw_GiB":
                    viol.append(row)
    _write_rows("test_iii_oracle.csv", out)
    _write_rows("test_iii_violations.csv", viol)
    assert not viol, f"O 的 SSD 讀寫總量比品質帳為 0 的固定做法大：{viol}"


@pytest.mark.parametrize("src,var,a,d,kw", DATA)
def test_iv_oracle_capacity_monotone(ft, src, var, a, d, kw):
    P = _P(src, var, a, d, **kw)
    caps = [4, 8, 16, 32, 64, 128, 256]
    out, viol = [], []
    for code, T in OUTERS:
        prev = None
        for cap in caps:
            rows, q, ids, eng, vs = sim.run_cell(P, code, "O", cap, "llama", SPECS[:1], T=T, fmt_path=ft)
            x = rows[0]["ssd_rw_GiB"]
            row = dict(test="iv", source=src, variant=var, outer=code, cap_GiB=cap, O_ssd_rw_GiB=x,
                       monotone=prev is None or x <= prev + 1e-12)
            out.append(row)
            if not row["monotone"]:
                viol.append(row)
            prev = x
    _write_rows("test_iv_monotone.csv", out)
    _write_rows("test_iv_violations.csv", viol)
    assert not viol, f"O 的容量單調被違反：{viol}"


def test_iii_fifo1_oracle(ft):
    """E6（block 引擎）也要：O 的品質帳 0、主指標不比品質帳 0 的 C 差（只在 block 引擎內比）。"""
    P = _P("bailian", "coder", 0, 1800, session_frac=0.3, session_seed=1)
    out, viol = [], []
    for code in ("E2", "E6"):
        res = {}
        for pol in ("C0", "C1", "C2", "C3", "C4", "O"):
            rows, q, ids, eng, vs = sim.run_cell_fifo1(P, code, pol, 16, "llama", SPECS[0], fmt_path=ft)
            assert all(i["ok"] for i in ids), [i for i in ids if not i["ok"]]
            res[pol] = rows[0]
        assert res["O"]["reads_fail"] == 0
        for c in ("C0", "C1", "C2", "C3", "C4"):
            row = dict(test="iii-fifo1", outer=code, rival=c, O=res["O"]["ssd_rw_GiB"], rival_value=res[c]["ssd_rw_GiB"],
                       rival_quality_ok=res[c]["reads_fail"] == 0, gpu_busy_frac=res[c]["gpu_busy_frac"])
            row["O_le_rival"] = row["O"] <= row["rival_value"] + 1e-12
            out.append(row)
            if row["rival_quality_ok"] and not row["O_le_rival"]:
                viol.append(row)
    _write_rows("test_iii_fifo1.csv", out)
    _write_rows("test_iii_violations.csv", viol)
    assert not viol, viol


# ===================================================================== (v) 重現
def _fingerprint(rows, vs):
    import hashlib
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps({k: v for k, v in r.items() if k != "elapsed_s"}, sort_keys=True, default=str).encode())
    for v in vs:
        for a in (v.ttft, v.rest, v.nC, v.nS, v.wq):
            h.update(np.ascontiguousarray(a).tobytes())
        h.update(bytes(memoryview(v.slow.rs)))
        h.update(bytes(memoryview(v.slow.ws)))
    return h.hexdigest()


def test_v_reproducible(ft):
    fps = []
    for seed in (7, 7, 8):
        _PC.clear()
        P = TR.prepare("bailian", "coder", 0, 1200, overlay=2, seed=seed, session_frac=0.5, session_seed=seed)
        assert P["meta"]["tier"] == "H"
        rows, q, ids, eng, vs = sim.run_cell(P, "E5", "O", 16, "llama", SPECS, T=300, H=300, fmt_path=ft)
        r2, _, _, _, v2 = sim.run_cell_fifo1(P, "E6", "C4", 16, "llama", SPECS[0], fmt_path=ft)
        fps.append(_fingerprint(rows + r2, vs + v2))
    _write_rows("test_v_repro.csv", [dict(test="v", seed=s, fingerprint=f) for s, f in zip((7, 7, 8), fps)])
    assert fps[0] == fps[1], "同設定同 seed 不同"
    assert fps[0] != fps[2], "不同 seed 卻完全相同（seed 沒有作用？）"


# ===================================================================== 其他
def test_quality_lineage_and_downward_only(ft):
    """C1：SSD 副本一律 int4；從 SSD 讀回來再寫進 CPU 的 block，從 CPU 讀時品質仍記 int4。"""
    P = _P("mooncake", "conv-rule-strict", 0, 1800)
    rows, q, ids, eng, vs = sim.run_cell(P, "E2", "C1", 8, "llama", SPECS[:1], fmt_path=ft, check=100)
    assert all(f == 3 for f in eng.sfmt if f), "C1 的 SSD 副本必須都是 int4"
    for b, f in enumerate(eng.cfmt):
        if f:
            assert eng.cq[b] >= f
        if eng.sfmt[b]:
            assert eng.sq[b] >= eng.sfmt[b]
    v = vs[0]
    cpu_only_int4 = int(np.sum((v.nS == 0) & (v.nC > 0) & (v.wq == 3)))
    assert cpu_only_int4 > 0, "應該有『全部從 CPU 讀、但品質來歷是 int4』的讀取（從 SSD promote 回來的）"
    _write_rows("test_lineage.csv", [dict(test="lineage", reads_cpu_only_lineage_int4=cpu_only_int4)])


def test_censoring_blocks_never_read(ft):
    """H 大於整個觀察窗：所有決定都在截尾區 → O 不准選 int2（never_read_only）。H=0 時會選到。"""
    P = _P("bailian", "coder", 0, 1800)
    _, _, _, e0, _ = sim.run_cell(P, "E2", "O", 16, "llama", SPECS[:1], H=0.0, fmt_path=ft)
    _, _, _, e1, _ = sim.run_cell(P, "E2", "O", 16, "llama", SPECS[:1], H=1e9, fmt_path=ft)
    assert any(k[0] == "int2" for k in e0.dec_fmt), "H=0 時 O 應該對不會再被讀的選 int2"
    assert not any(k[0] == "int2" for k in e1.dec_fmt), "全部截尾時 O 不准選 int2"


def test_predictor_interface_no_labels(ft):
    P = _P("mooncake", "conv-rule-strict", 0, 900)
    seen = []

    def fn(state, feats):
        seen.append(set(state))
        assert "next_read" not in state and "censored" not in state and "max_read_tokens" not in state
        return "bf16"
    mf = load_format_table("llama", "mi300x", ft)
    r0, _, _, _, v0 = sim.run_cell(P, "E2", "C0", 16, "llama", SPECS, fmt_path=ft)
    r1, _, _, _, v1 = sim.run_cell(P, "E2", None, 16, "llama", SPECS, fmt_path=ft, policy_obj=PredictorPolicy(mf, fn))
    assert seen
    for a, b in zip(v0, v1):
        assert np.array_equal(a.ttft, b.ttft) and a.slow.bytes_w == b.slow.bytes_w


def test_cost_model_vs_x8():
    """Cake 單層、均一 layout：引擎的 View.restore ＝ x8_sim.cake（X8 的成本表，|差| < 1e-9，X8 自檢同一門檻）。"""
    import x8_sim
    f8, ell, _ = x8_sim.load_costs()
    worst = 0.0
    mf = ModelFormats("llama", "x8", [None, BF, None, None, None], {"idle": [None, 0.0, None, None, None], "busy": [None, 0.0, None, None, None]},
                      [0, 1 << 40], {}, [], False, {"path": "test", "sha256": "-"})
    fl = [float(x) for x in f8]
    for tier in ("cpu", "local", "nfs"):
        for n in (1, 2, 5, 17, 64, 80, 157, 256):
            rate = BF / ell[tier] / GiB
            ch = dict(r=rate, w=rate, k=1.0)
            c = Costs([x * 1e3 for x in fl], 0.0, 0.0, 1.0, ch, ch, "x8")
            v = View(c, mf, "idle", "cake", len(fl), 1)
            tier_code = 1 if tier == "cpu" else 3
            tt, rs, nC, nS, nR, wq = v.restore([tier_code] * n, [1] * n, [1] * n, n, n, 0.0, n)
            t8, meet8 = x8_sim.cake(n, ell[tier], np.array(fl))
            worst = max(worst, abs(v.last_end - t8))
            assert nR == meet8 and abs(v.last_end - t8) < 1e-9, (tier, n, v.last_end, t8)
    _write_rows("test_x8_cost.csv", [dict(test="x8_cost", max_abs_diff_s=worst)])


def test_not_measured_refuses(ft, tmp_path):
    """NOT_MEASURED 的常數用到就丟錯（不靜默補值）。格式表用測試自己的，不靠 placeholder。"""
    from formats import NotMeasured
    with pytest.raises(NotMeasured):        # 3090 的 CPU 忙係數沒量
        sim.make_views(_P("mooncake", "conv-rule-strict", 0, 300), "llama", [("3090", "idle", "busy", 1, "cake", "mi300x_local")],
                       {"3090": load_format_table("llama", "mi300x", ft)})
    with pytest.raises(NotMeasured):        # 3090 /ssd8 沒量
        load_costs("3090", "llama", "ssd8_3090")
    d = json.load(open(ft))
    d["models"]["llama"]["formats"]["8bit"]["bytes_per_block"] = "NOT_MEASURED"
    p2 = str(tmp_path / "ft_nm.json")
    json.dump(d, open(p2, "w"))
    with pytest.raises(NotMeasured):        # 沒有大小的格式不准用
        make_policy("C3", load_format_table("llama", "mi300x", p2))


@pytest.mark.parametrize("src,var,a,d", [("bailian", "coder", 0, 1800), ("mooncake", "conv-rule-strict", 0, 1800),
                                         ("freeinference", "week01", 0, 86400)])
def test_block_keys_last_id_equals_full_prefix(src, var, a, d):
    """鏈式 id 的來源：「最後一個 id」與「整段前綴」兩種 key 給出完全相同的 P（traceA 不是鏈式，只能用整段前綴）。"""
    P1 = TR.prepare(src, var, a, d, key_mode="auto")
    P2 = TR.prepare(src, var, a, d, key_mode="prefix")
    assert P1["meta"]["checks"]["block_key_mode"] == "last_id(chained)"
    assert P2["meta"]["checks"]["block_key_mode"] == "full_prefix"
    assert P1["full"] == P2["full"] and P1["nall"] == P2["nall"] and P1["tl"] == P2["tl"]


# ===================================================================== 2026-10-18（06 §11–§13）新加的
SYN = ("synth", "small-seed1", 0, 4 * 3600)


def test_official_format_table_adapter():
    """正式表 → fid：位元組＝表的 bytes_per_block_code；解壓＝placeholder 同語意的欄；品質＝32K 的生成判準，32,768 本身在實測的分箱。"""
    import formats as FM
    d = json.load(open(FM.OFFICIAL_FORMAT_TABLE))
    rows = {(r["model"], r["fmt"]): r for r in d["rows"]}
    sha = FM.sha256_file(FM.OFFICIAL_FORMAT_TABLE)
    out = []
    assert FM.DEFAULT_DQ_SET == "cake_k0" and FM.OFFICIAL_DQ_COLS["mi300x"] == ("dq_mi300x_cake_k0_idle_ms", "dq_mi300x_cake_k0_busy_ms")
    # 只掃這張表能給的平台（2026-10-20：DQ_SETS 多了 h200 欄，H200 的表另外測，見 test_official_format_table_adapter_h200）
    tp = FM.TABLE_PLATFORMS[d.get("platform")]
    for model, dqs, plat in [(m, d_, p_) for m in ("llama", "qwen") for d_ in FM.DQ_SETS for p_ in FM.DQ_SETS[d_] if p_ in tp]:
        ci, cb = FM.DQ_SETS[dqs][plat]
        mf = load_format_table(model, plat, "official" if dqs == FM.DEFAULT_DQ_SET else f"official:{dqs}")   # 預設＝正式表＋cake_k0
        assert mf.src["dq_set"] == dqs
        assert mf.src["sha256"] == sha and mf.src["kind"] == "official" and not mf.placeholder
        for fid, fmt in FM.OFFICIAL_CONCRETE[model].items():
            c = rows[(model, fmt)]["cells"]
            assert mf.bytes[fid] == c["bytes_per_block_code"]["value"]
            for col, g in ((ci, "idle"), (cb, "busy")):
                v = c[col]["value"]
                assert (mf.dq[g][fid] is None) if isinstance(v, str) else abs(mf.dq[g][fid] - v / 1e3) < 1e-15
            out.append(dict(test="official_fmt", model=model, platform=plat, fid=fid, concrete=fmt,
                            status_32768=mf.status(fid, 32768), status_32769=mf.status(fid, 32769)))
        assert mf.bin_of(32768) == 0 and mf.bin_of(32769) == 1 and mf.bin_of(32768 + 512) == 1
        assert [mf.status(f, 32768) for f in (1, 2, 3, 4)] == ["pass", "pass", "pass", "unverified"]
        assert [mf.status(f, 1 << 20) for f in (1, 2, 3, 4)] == ["pass", "unverified", "unverified", "unverified"]
        assert [mf.status(f, 512) for f in (1, 2, 3, 4)] == ["pass", "pass", "pass", "unverified"]
        assert mf.never_read_only == {4}
        assert all(q["agrees_with_table_verdict"] for q in mf.src["quality"].values())
    # 04 v4 §9：MI300X 預設＝「還原中・Cake・K0」；舊的「只載入」欄還可以選（值不同）
    mfd, mfl = load_format_table("llama", "mi300x"), load_format_table("llama", "mi300x", "official:loadonly_k0")
    assert mfd.dq["idle"][4] == 0.0014996 and mfd.dq["busy"][4] == 0.0021866 and mfl.dq["idle"][4] is None
    assert mfd.dq["idle"][2] == 0.3867 / 1e3 and mfl.dq["idle"][2] == 0.2331 / 1e3
    assert mfd.src["sha256"] == mfl.src["sha256"]
    assert load_format_table("qwen", "mi300x").dq["idle"][4] is None and load_format_table("qwen", "mi300x").dq["idle"][2] == 0.166 / 1e3
    assert load_format_table("qwen", "mi300x").src["concrete"]["8bit"] == "int8_token"      # Qwen 的 FP8 不過 → INT8
    assert load_format_table("llama", "mi300x").src["concrete"]["8bit"] == "fp8_token"
    # FAIL 的格式（Qwen fp8_head）照判準算出 fail（驗算判準本身）
    assert FM.official_quality_32k(rows[("qwen", "fp8_head")])[0] == "fail"
    assert load_format_table("llama", "mi300x", "placeholder").placeholder
    _write_rows("test_official_fmt.csv", out)


def test_official_format_table_adapter_h200():
    """H200 的正式表（2026-10-20）：同一套轉換規則；表和平台對不上要丟錯（兩個方向）。"""
    import formats as FM
    d = json.load(open(FM.OFFICIAL_FORMAT_TABLE_H200))
    assert d["platform"] == "h200" and FM.TABLE_PLATFORMS["h200"] == ("h200",)
    rows = {(r["model"], r["fmt"]): r for r in d["rows"]}
    sha = FM.sha256_file(FM.OFFICIAL_FORMAT_TABLE_H200)
    out = []
    for model, dqs in [(m, d_) for m in ("llama", "qwen") for d_ in FM.DQ_SETS]:
        ci, cb = FM.DQ_SETS[dqs]["h200"]
        mf = load_format_table(model, "h200", "official_h200" if dqs == FM.DEFAULT_DQ_SET else f"official_h200:{dqs}")
        assert mf.src["dq_set"] == dqs and mf.src["dq_cols"] == [ci, cb]
        assert mf.src["sha256"] == sha and mf.src["kind"] == "official" and not mf.placeholder
        for fid, fmt in FM.OFFICIAL_CONCRETE[model].items():
            c = rows[(model, fmt)]["cells"]
            assert mf.bytes[fid] == c["bytes_per_block_code"]["value"]
            for col, g in ((ci, "idle"), (cb, "busy")):
                v = c[col]["value"]
                assert (mf.dq[g][fid] is None) if isinstance(v, str) else abs(mf.dq[g][fid] - v / 1e3) < 1e-15
            out.append(dict(test="official_fmt_h200", model=model, platform="h200", fid=fid, concrete=fmt,
                            status_32768=mf.status(fid, 32768), status_32769=mf.status(fid, 32769)))
        assert mf.bin_of(32768) == 0 and mf.bin_of(32769) == 1
        assert all(q["agrees_with_table_verdict"] for q in mf.src["quality"].values())
    for plat in ("mi300x", "3090"):
        with pytest.raises(ValueError):
            load_format_table("llama", plat, "official_h200")
    with pytest.raises(ValueError):
        load_format_table("llama", "h200", "official")
    _write_rows("test_official_fmt_h200.csv", out)


def test_official_table_sha_in_rows_and_ttft_request_rule():
    """每列都記正式表的 sha256；解壓 NOT_MEASURED 的格式只在真的被載入時才讓 TTFT（ttftq_*）變 NOT_MEASURED，位元組照算。"""
    import formats as FM
    P = _P(*SYN)
    sha = FM.sha256_file(FM.OFFICIAL_FORMAT_TABLE)
    r0, _, _, _, v0 = sim.run_cell(P, "E5", "C0", 4, "llama", SPECS[:1], T=300)
    assert r0[0]["format_table_sha256"] == sha and r0[0]["format_table_kind"] == "official"
    assert r0[0]["ttftq_valid"] and r0[0]["ttft_nm_ret_requests"] == 0          # C0 只用 bf16（解壓定義為 0）
    assert r0[0]["ttft_valid"] is True and r0[0]["format_dq_set"] == "cake_k0"  # Llama／MI300X 的 Cake・K0 欄四個格式都有值（04 v4 §9）
    rl, _, _, _, _ = sim.run_cell(P, "E5", "C0", 4, "llama", SPECS[:1], T=300, fmt_path="official:loadonly_k0")
    assert rl[0]["ttft_valid"] is False and rl[0]["format_dq_set"] == "loadonly_k0"    # 舊的只載入欄：Llama int2 沒有值 → 舊規則整個 view 不算
    # Qwen 的 int2（Cake・K0 欄）是 NOT_MEASURED：真的載入到它的請求 → ttftq_* 才是 NOT_MEASURED，位元組照算
    r1, _, _, e1, v1 = sim.run_cell(P, "E5", "D-idle", 4, "qwen", SPECS[:1], T=300, pol_kw={"d_idle_tau_s": 60.0})
    assert r1[0]["ttft_valid"] is False
    n_int2_reads = int(np.sum((v1[0].nS + v1[0].nC > 0) & (v1[0].wq == 4) & (np.asarray(P["kindc"]) == 1)))
    if n_int2_reads:
        assert r1[0]["ttft_nm_ret_requests"] > 0 and r1[0]["ttftq_mean_s"] == "NOT_MEASURED"
    assert r1[0]["ssd_write_GiB"] > 0
    _write_rows("test_ttft_rule.csv", [dict(test="ttft_rule", didle_int2_ret_reads=n_int2_reads,
                                            nm_ret=r1[0]["ttft_nm_ret_requests"])])


def _ssd_fmt_counts(eng):
    return {f: n for (t, f), n in eng.wblocks_fmt.items() if t == "ssd"}


@pytest.mark.parametrize("code,T", [("E2", None), ("E3", None), ("E4", None), ("E5", 300)])
def test_oracle_ssd_decision(ft, code, T):
    """O 在寫 SSD 那一刻再決定（06 §11.2）：品質帳 0；SSD 格式不比 CPU 好；cpu-only 版不會把 CPU 的 bf16／8bit 壓成 int2。"""
    P = _P(*SYN)
    r, _, ids, e, _ = sim.run_cell(P, code, "O", 4, "llama", SPECS[:1], T=T, H=600, fmt_path=ft, check=200)
    assert all(i["ok"] for i in ids)
    assert r[0]["reads_fail"] == 0
    for b, f in enumerate(e.sfmt):
        if f and e.cfmt[b]:
            assert f >= e.cfmt[b] or code in ("E3", "E4")     # 寫穿／第 2 次才寫：SSD 副本可以比 CPU 的更省；CPU 之後重寫則不比較
        if f:
            assert e.sq[b] >= f
    rc, _, idc, ec, _ = sim.run_cell(P, code, "O", 4, "llama", SPECS[:1], T=T, H=600, fmt_path=ft, pol_kw={"o_ssd_oracle": False})
    assert all(i["ok"] for i in idc) and rc[0]["policy"] == "O(cpu-only)" and rc[0]["reads_fail"] == 0
    _write_rows("test_oracle_ssd.csv", [dict(test="oracle_ssd", outer=code, O_ssd_int2_blocks=_ssd_fmt_counts(e).get(4, 0),
                                             O_cpuonly_ssd_int2_blocks=_ssd_fmt_counts(ec).get(4, 0))])


def test_oracle_ssd_reaches_int2_after_last_read(ft):
    """修正前做不到的事：block 在 CPU 時之後還會被讀（所以不是 int2），之後被降級時已經不會再被讀 → O 在降級時壓成 int2。
    用一個手造的小 P：session 0 讀兩次、session 1 把它擠出 CPU；cpu-only 的舊 O 不會寫 int2。"""
    full = [[0, 1], [0, 1], [2, 3, 4]]
    P = dict(full=full, nall=[2, 2, 3], acc={0: [0, 1], 1: [0, 1], 2: [2], 3: [2], 4: [2]}, tl=[0.0, 10.0, 20.0], kindc=[0, 1, 0],
             sessl=[0, 0, 1], parentl=[-1, 0, -1], obs_end=[1e6] * 3, nmax=8, B=5, R=dict(n_sess=2),
             meta=dict(source="hand", variant="x", tier="S", n_req=3, n_sess=2))
    mf = load_format_table("llama", "mi300x", ft)
    cap = 3 * (20 << 20)       # 放得下 3 個 int4 block：第 3 個請求寫 3 個 → 擠出 session 0
    out = {}
    for kw in ({}, {"o_ssd_oracle": False}):
        comp = make_policy("O", mf, **kw)
        v = View(load_costs("mi300x", "llama"), mf, "idle", "cake", 8, 3)
        e = Engine(P, sim.outer_cfg("E2"), comp, cap, [v], mf, check=1).run()
        out[comp.name] = (e.cfmt[:], e.sfmt[:])
    assert out["O"][1][0] == 4 and out["O"][1][1] == 4, out          # 降級時之後不會再被讀 → int2
    assert out["O(cpu-only)"][1][0] != 4, out                        # 舊 O：SSD 維持 CPU 寫入時的格式


def test_oracle_variants(ft):
    P = _P(*SYN)
    mf = load_format_table("llama", "mi300x", ft)
    res = {}
    for name in ("O", "O-len", "O-reuse", "O-loose"):
        r, _, ids, e, _ = sim.run_cell(P, "E2", name, 4, "llama", SPECS[:1], H=0.0, fmt_path=ft, check=200)
        assert all(i["ok"] for i in ids)
        res[name] = (r[0], e)
    assert not any(k[0] == "int2" for k in res["O-len"][1].dec_fmt), "O-len 不知道『不會再被讀』，不准用 int2"
    assert 4 not in _ssd_fmt_counts(res["O-len"][1])
    assert any(k[0] == "int2" for k in res["O-reuse"][1].dec_fmt)
    ol = make_policy("O-loose", mf)
    assert len({ol.read_fid(t) for t in (512, 16384, 32768, 1 << 20)}) == 1, "O-loose 的格式和長度無關"
    assert res["O"][0]["reads_fail"] == 0 and res["O-len"][0]["reads_fail"] == 0
    _write_rows("test_oracle_variants.csv", [dict(test="variants", policy=k, reads_fail=v[0]["reads_fail"],
                                                  reads_unverified=v[0]["reads_unverified"]) for k, v in res.items()])


def test_new_baselines(ft):
    P = _P(*SYN)
    for name, cpu_set, ssd_set in (("C4r", {1, 3}, None), ("C1-8", {1}, {2}), ("C6", {2}, {3}), ("D-len", {1}, {1, 3}),
                                   ("D-idle", {1}, {1, 3, 4})):
        r, _, ids, e, _ = sim.run_cell(P, "E5", name, 4, "llama", SPECS[:1], T=300, fmt_path=ft, check=200,
                                       pol_kw={"d_idle_tau_s": 600.0})
        assert all(i["ok"] for i in ids), name
        cf = {f for (t, f), n in e.wblocks_fmt.items() if t == "cpu"}
        sf = set(_ssd_fmt_counts(e))
        assert cf <= cpu_set, (name, cf)
        if ssd_set is not None:
            assert sf <= ssd_set, (name, sf)
    _, _, _, e, _ = sim.run_cell(P, "E5", "D-idle", 4, "llama", SPECS[:1], T=300, fmt_path=ft, pol_kw={"d_idle_tau_s": 200.0})
    assert _ssd_fmt_counts(e).get(4, 0) > 0, "E5（閒置 300 s 才降級）＋D-idle（>200 s）應該寫 int2"


def test_predictor_ssd_hook_no_labels(ft):
    P = _P(*SYN)
    mf = load_format_table("llama", "mi300x", ft)
    seen = []

    def ssd_fn(state, feats, cur):
        seen.append(set(state))
        assert not {"next_read", "censored", "max_read_tokens", "cpu_ctx"} & set(state)
        assert "next_read" not in state.get("cpu_state", {})
        return "int4" if state["tokens"] <= 32768 else cur
    pp = PredictorPolicy(mf, lambda s, f: "bf16", ssd_fn=ssd_fn, name="P-dlen")
    r1, _, _, e1, v1 = sim.run_cell(P, "E5", None, 4, "llama", SPECS[:1], T=300, fmt_path=ft, policy_obj=pp)
    r2, _, _, e2, v2 = sim.run_cell(P, "E5", "D-len", 4, "llama", SPECS[:1], T=300, fmt_path=ft)
    assert seen and e1.wblocks_fmt == e2.wblocks_fmt and np.array_equal(v1[0].ttft, v2[0].ttft), "同規則的預測器＝D-len"


@pytest.mark.parametrize("engine", ["session", "fifo1"])
def test_int2_on_read_recompute(ft, engine):
    """int2_on_read=recompute：讀到的 block 不會有 INT2 來歷；surprise 有計數；預設（serve）和不給參數逐位元相同。"""
    P = _P(*SYN)
    kw = dict(fmt_path=ft, pol_kw={"d_idle_tau_s": 200.0})
    if engine == "session":
        run = lambda **k: sim.run_cell(P, "E5", "D-idle", 2, "llama", SPECS[:1], T=300, check=200, **kw, **k)   # noqa: E731
    else:
        run = lambda **k: sim.run_cell_fifo1(P, "E2", "D-idle", 2, "llama", SPECS[0], **kw, **k)                # noqa: E731
    r0, _, i0, e0, v0 = run()
    rs, _, is_, es, vs = run(int2_on_read="serve")
    rr, _, ir, er, vr = run(int2_on_read="recompute")
    assert all(i["ok"] for i in i0 + is_ + ir)
    assert np.array_equal(v0[0].ttft, vs[0].ttft) and v0[0].slow.bytes_r == vs[0].slow.bytes_r
    n_int2 = int(np.sum((v0[0].nS + v0[0].nC > 0) & (v0[0].wq == 4)))
    assert not np.any((vr[0].nS + vr[0].nC > 0) & (vr[0].wq == 4)), "recompute 時不准讀到 INT2 來歷的 block"
    if n_int2:
        assert rr[0]["surprise_events"] > 0 and rr[0]["surprise_tokens"] == rr[0]["surprise_blocks"] * 512
    assert r0[0]["surprise_events"] == 0
    _write_rows("test_int2_recompute.csv", [dict(test="int2_recompute", engine=engine, serve_int2_reads=n_int2,
                                                 surprise_events=rr[0]["surprise_events"], surprise_blocks=rr[0]["surprise_blocks"])])


def test_per_session_and_common_window(ft, tmp_path):
    """每個 session 的帳加起來＝總量；同一格的每個策略用同一個窗口範圍；served／offered 的窗口內＋spill＝總位元組。"""
    P = _P(*SYN)
    wins = set()
    for pol in ("C0", "C1", "O", "D-idle"):
        p = str(tmp_path / f"{pol}.npz")
        rows, _, ids, e, vs = sim.run_cell(P, "E5", pol, 4, "llama", SPECS, T=300, H=600, fmt_path=ft, sess_out=p)
        assert all(i["ok"] for i in ids)
        z = np.load(p)
        assert z["ssd_write_bytes"].sum() == vs[0].slow.bytes_w
        for i, v in enumerate(vs):
            assert z["ssd_read_bytes"][i].sum() == v.slow.bytes_r
            assert z["reads"][i].sum() == rows[i]["reads"] and z["reads_fail"][i].sum() == rows[i]["reads_fail"]
        assert z["n_req"].sum() == len(P["full"]) and len(z["sess_key"]) == P["R"]["n_sess"]
        for row, v in zip(rows, vs):
            wins.add((row["win_lo_s"], row["win_hi_s"], row["cw_n_windows"]))
            import ledger
            lo, hi = ledger.win_range(P)
            nwin = row["cw_n_windows"]
            r, sp = ledger.spread_in_range(v.slow.rs, v.slow.re, v.slow.rb, lo, nwin)
            assert abs(r.sum() + sp - v.slow.bytes_r) <= 1e-6 * max(1, v.slow.bytes_r)
        for (rid, s_, nb) in zip(e.wsp_rec, e.wsp_sid, e.wsp_b):
            assert nb > 0 and 0 <= s_ < P["R"]["n_sess"]
    assert len(wins) == 1, wins
    # block 引擎
    p = str(tmp_path / "fifo1.npz")
    rows, _, ids, e, vs = sim.run_cell_fifo1(P, "E6", "C1", 2, "llama", SPECS[0], fmt_path=ft, sess_out=p)
    z = np.load(p)
    assert all(i["ok"] for i in ids) and z["ssd_read_bytes"][0].sum() == vs[0].slow.bytes_r
    assert z["ssd_write_bytes"].sum() == vs[0].slow.bytes_w


def test_loadonly_bytes_view_independent(ft):
    """load-only SSD 位元組只看放置：session 引擎的每個時間設定都一樣，而且等於 load-only 時間設定的 SSD 讀取。"""
    P = _P(*SYN)
    rows, _, _, e, vs = sim.run_cell(P, "E2", "C1", 4, "llama", SPECS, fmt_path=ft)
    assert len({r["ssd_loadonly_read_GiB"] for r in rows}) == 1
    lo = [v for v in vs if v.mode == "load"][0]
    assert lo.slow.bytes_r == int(e.lo_bytes.sum())


def test_block_oracle_sufmax_equals_scan(ft):
    """block 引擎的 O：後綴最大值＝原本每次掃之後的讀取（06 §8-15）。"""
    import bisect
    P = _P("bailian", "coder", 0, 600, session_frac=0.3, session_seed=1)
    _, _, _, e, _ = sim.run_cell_fifo1(P, "E2", "O", 8, "llama", SPECS[0], fmt_path=ft)
    rng = np.random.default_rng(0)
    bs = list(e.acc)
    for b in rng.choice(len(bs), size=min(300, len(bs)), replace=False):
        b = bs[int(b)]
        lst = e.acc[b]
        for j in set([-1, lst[0], lst[-1], lst[len(lst) // 2]]):
            mx = 0
            for jj in lst[bisect.bisect_right(lst, j):]:
                mx = max(mx, len(e.R[jj]["h"]))
            assert e._max_future_len(b, j) == mx


def test_prepare_cache(tmp_path):
    """P 的快取：第二次命中、內容相同；檔案被改過就丟錯（不靜默重算）。"""
    d = str(tmp_path / "cache")
    P1 = TR.prepare_cached(*SYN, cache_dir=d)
    P2 = TR.prepare_cached(*SYN, cache_dir=d)
    assert P1["meta"]["cache"]["hit"] is False and P2["meta"]["cache"]["hit"] is True
    P0 = TR.prepare(*SYN)
    for k in ("full", "nall", "tl", "kindc", "sessl", "parentl", "obs_end", "win", "sess_key", "acc"):
        assert P2[k] == P0[k], k
    pk = [f for f in os.listdir(d) if f.endswith(".pkl")][0]
    with open(os.path.join(d, pk), "r+b") as fh:
        fh.seek(100)
        fh.write(b"\x00\x01\x02")
    with pytest.raises(RuntimeError):
        TR.prepare_cached(*SYN, cache_dir=d)


@pytest.mark.parametrize("engine", ["session", "fifo1"])
def test_recompute_unverified(ft, engine):
    """04 v4 §3.3：int2_on_read=recompute_unverified → 每次讀取只載入在那個長度品質過關的前綴；品質帳（fail＋unverified）一律 0。"""
    from engine import INF, ok_token_limits
    assert ok_token_limits(load_format_table("llama", "mi300x")) == [-1, INF, 32768, 32768, -1]
    assert ok_token_limits(load_format_table("llama", "mi300x", ft)) == [-1, INF, INF, 16383, -1]
    P = _P(*SYN)
    out = []
    for pol in ("C2", "D-idle", "O-reuse", "C0"):
        kw = dict(fmt_path=ft, pol_kw={"d_idle_tau_s": 200.0}, int2_on_read="recompute_unverified")
        if engine == "session":
            r, _, ids, e, v = sim.run_cell(P, "E5", pol, 2, "llama", SPECS[:1], T=300, check=200, **kw)
        else:
            r, _, ids, e, v = sim.run_cell_fifo1(P, "E2", pol, 2, "llama", SPECS[0], **kw)
        assert all(i["ok"] for i in ids), (pol, [i for i in ids if not i["ok"]])
        assert r[0]["reads_fail"] == 0 and r[0]["reads_unverified"] == 0, (pol, r[0]["reads_fail"], r[0]["reads_unverified"])
        if pol == "C0":
            assert r[0]["surprise_events"] == 0
        out.append(dict(test="recompute_unverified", engine=engine, policy=pol, surprise_events=r[0]["surprise_events"]))
    assert any(x["surprise_events"] > 0 for x in out if x["policy"] == "C2"), "C2（全 int4）在 ≥16K 的讀取應該觸發重算"
    _write_rows("test_recompute_unverified.csv", out)


# ===================================================================== 2026-10-18（06 §18）嚴格讀取長度：unverified_len="request"
def _hand_P_strict():
    """手造：session 0 在 r0 寫 4 個 block；session 1（r1）把它擠出 CPU（降級到 SSD）；session 0 在 r2 回來，
    r2 的輸入是 40 個 block＝20,480 token（> 測試表的 16,384）、存的前綴只有 4 個 block＝2,048 token。"""
    full = [[0, 1, 2, 3], [10, 11, 12, 13], [0, 1, 2, 3, 4, 5]]
    acc = {}
    for r, ks in enumerate(full):
        for b in ks:
            acc.setdefault(b, []).append(r)
    return dict(full=full, nall=[4, 4, 40], acc=acc, tl=[0.0, 10.0, 20.0], kindc=[0, 0, 1], sessl=[0, 1, 0], parentl=[-1, -1, 0],
                obs_end=[1e6] * 3, nmax=64, B=14, R=dict(n_sess=2), win=(0.0, 100.0),
                meta=dict(source="hand", variant="strict", tier="S", n_req=3, n_sess=2))


def test_strict_request_length_hand(ft):
    """嚴格讀取長度（04 v4 §3.3）：同一個放置，prefix 版用載入的前綴長度（2,048 token → int4 過關、照讀），
    request 版用請求的輸入長度（20,480 token → int4 不過關 → 整段重算）；O、O-len 在 request 版挑 8bit（int4 在那個長度不過關），0 次重算。"""
    P = _hand_P_strict()
    cap = 4 * BF
    out = []
    res = {}
    for ul in ("prefix", "request"):
        for pol in ("C1", "O", "O-len", "O-reuse"):
            v = View(load_costs("mi300x", "llama"), load_format_table("llama", "mi300x", ft), "idle", "cake", 64, 3)
            mf = load_format_table("llama", "mi300x", ft)
            comp = make_policy(pol, mf, o_read_len=ul)
            e = Engine(P, sim.outer_cfg("E2"), comp, cap, [v], mf, check=1, int2_on_read="recompute_unverified", unverified_len=ul).run()
            res[(ul, pol)] = e
            out.append(dict(test="strict_hand", unverified_len=ul, policy=pol, surprise_blocks=int(e.surp_blk.sum()),
                            L_tok_r2=int(e.Ltok[2]), ssd_fmt_b0=e.sfmt[0], cpu_fmt_b0=e.cfmt[0]))
    assert res[("prefix", "C1")].surp_blk.sum() == 0 and res[("prefix", "C1")].Ltok[2] == 4 * 512
    assert res[("request", "C1")].surp_blk[2] == 4 and res[("request", "C1")].Ltok[2] == 0         # int4 來歷 → 整段重算
    for pol in ("O", "O-len"):
        e = res[("request", pol)]
        assert e.surp_blk.sum() == 0, pol
        assert e.sfmt[0] == 2 or e.cfmt[0] == 2, (pol, e.sfmt[0], e.cfmt[0])                        # 8bit：20,480 token 下過關、最省
        assert e.Ltok[2] == 4 * 512
    # prefix 版的 oracle 看載入的前綴長度（3,072 token）→ int4，和 request 版不同
    assert res[("prefix", "O")].sfmt[0] == 3 or res[("prefix", "O")].cfmt[0] == 3
    # request 版的品質帳用請求長度分箱：C1 的重算讓 reads_unverified／fail 為 0
    rr, _, ids, e2, _ = sim.run_cell(P, "E2", "C1", 4 * BF / GiB, "llama", SPECS[:1], fmt_path=ft, check=1,
                                     int2_on_read="recompute_unverified", unverified_len="request")
    assert all(i["ok"] for i in ids) and rr[0]["reads_fail"] == 0 and rr[0]["reads_unverified"] == 0 and rr[0]["surprise_blocks"] == 4
    assert rr[0]["unverified_len"] == "request"
    # serve 模式（舊行為）＋request 長度：只改品質帳的分箱（r2 的 4 個 int4 block 在 20,480 token 下 fail）
    rs, _, _, _, _ = sim.run_cell(P, "E2", "C1", 4 * BF / GiB, "llama", SPECS[:1], fmt_path=ft, check=1, unverified_len="request")
    rp, _, _, _, _ = sim.run_cell(P, "E2", "C1", 4 * BF / GiB, "llama", SPECS[:1], fmt_path=ft, check=1)
    assert rs[0]["reads_fail"] == 1 and rp[0]["reads_fail"] == 0 and rs[0]["surprise_events"] == 0
    _write_rows("test_strict_hand.csv", out)


@pytest.mark.parametrize("engine", ["session", "fifo1"])
def test_strict_oracle_invariants_synth(engine):
    """合成資料（只查不變量，不比較）：request 版嚴格規則下，O、O-len 沒有意外重算、品質帳為 0；非 oracle 的可以有；預設（serve）不變。"""
    P = _P(*SYN)
    kw = dict(int2_on_read="recompute_unverified", unverified_len="request")
    out = {}
    for pol in ("C0", "C1", "C2", "O", "O-len", "O-reuse", "O-loose", "D-idle"):
        if engine == "session":
            r, _, ids, e, _ = sim.run_cell(P, "E5", pol, 2, "llama", SPECS[:1], T=300, H=1800.0, check=200, **kw)
        else:
            r, _, ids, e, _ = sim.run_cell_fifo1(P, "E2", pol, 2, "llama", SPECS[0], H=1800.0, **kw)
        assert all(i["ok"] for i in ids), (pol, [i for i in ids if not i["ok"]])
        assert r[0]["reads_fail"] == 0 and r[0]["reads_unverified"] == 0, (pol, r[0]["reads_fail"], r[0]["reads_unverified"])
        out[pol] = r[0]
    for pol in ("C0", "O", "O-len"):
        assert out[pol]["surprise_events"] == 0 and out[pol]["surprise_blocks"] == 0, (pol, out[pol]["surprise_events"])
    assert out["C2"]["surprise_events"] > 0, "全 int4：> 32K 的請求（63%）應該重算"
    _write_rows("test_strict_synth.csv", [dict(test="strict_synth", engine=engine, policy=k, surprise_events=v["surprise_events"],
                                               surprise_blocks=v["surprise_blocks"], reads=v["reads"]) for k, v in out.items()])


# ===================================================================== 2026-10-18（04 §3.3「重算之後」）重算的新 KV 照現在的決定重存
def _hand_P_restore():
    """手造 4 個請求：s0 在 r0 寫 4 個 block（2,048 token）；s1 在 r1 寫 4 個；s0 在 r2 回來（輸入 40 個 block＝20,480 token，>16,384）、
    再在 r3 回來（輸入 41 個 block）。測試格式表：int4 在 <16K 過關、≥16K 不過關；bf16 都過關。"""
    full = [[0, 1, 2, 3], [10, 11, 12, 13], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 6]]
    acc = {}
    for r, ks in enumerate(full):
        for b in ks:
            acc.setdefault(b, []).append(r)
    return dict(full=full, nall=[4, 4, 40, 41], acc=acc, tl=[0.0, 10.0, 20.0, 30.0], kindc=[0, 0, 1, 1], sessl=[0, 1, 0, 0],
                parentl=[-1, -1, 0, 2], obs_end=[1e6] * 4, nmax=64, B=14, R=dict(n_sess=2), win=(0.0, 100.0),
                meta=dict(source="hand", variant="restore", tier="S", n_req=4, n_sess=2))


def _restore_run(engine, ft, pol, cap_MB, pol_kw=None, **kw):
    P = _hand_P_restore()
    cap = cap_MB / 1024.0
    kw = dict(dict(int2_on_read="recompute_unverified", unverified_len="request"), **kw)
    if engine == "session":
        r, _, ids, e, _ = sim.run_cell(P, "E2", pol, cap, "llama", SPECS[:1], fmt_path=ft, check=1, pol_kw=pol_kw, **kw)
        wst = e.wbytes[W_STORE]
        cf = lambda b: e.cfmt[b]   # noqa: E731
        sf = lambda b: e.sfmt[b]   # noqa: E731
    else:
        r, _, ids, e, _ = sim.run_cell_fifo1(P, "E2", pol, cap, "llama", SPECS[0], fmt_path=ft, pol_kw=pol_kw, **kw)
        wst = e.wbytes["store"]
        cf = lambda b: e.cfmt.get(b, 0) if b in e.cpu else 0   # noqa: E731
        sf = lambda b: e.sfmt.get(b, 0)   # noqa: E731
    assert all(i["ok"] for i in ids), [i for i in ids if not i["ok"]]
    assert r[0]["reads_fail"] == 0 and r[0]["reads_unverified"] == 0
    return r[0], e, cf, sf, wst


@pytest.mark.parametrize("engine", ["session", "fifo1"])
def test_recompute_restore_c2_restores_int4_again(ft, engine):
    """(i) C2（全 int4）：長請求重算後又存成 int4（SSD 上的舊副本作廢），所以下一個長讀取又重算；位元組與恆等式成立。
    cap＝150 MB：r1 把 s0（4 個 int4＝80 MB）擠到 SSD；r2 從 SSD 讀到的 4 個 block 不能用 → 重算，舊 SSD 副本作廢。"""
    r, e, cf, sf, wst = _restore_run(engine, ft, "C2", 150)
    sb = e.surp_blk
    assert [int(x) for x in sb] == [0, 0, 4, 6], list(sb)          # r2：4 個（來自 SSD）；r3：6 個又重算
    assert r["surprise_events"] == 2 and r["surprise_blocks"] == 10 and r["surprise_tokens"] == 10 * 512
    I4 = 20 << 20
    assert wst == (4 + 4 + 6 + 7) * I4, wst                        # r2 存 6 個（重算 4＋新 2）、r3 存 7 個（重算 6＋新 1）
    assert all(cf(b) == 3 for b in range(7)), [cf(b) for b in range(7)]
    assert all(sf(b) == 0 for b in range(7)), "舊 SSD 副本要作廢"
    assert all(sf(b) == 3 for b in (10, 11, 12, 13)), "s1 被擠到 SSD 的副本不受影響"


@pytest.mark.parametrize("engine", ["session", "fifo1"])
def test_recompute_restore_c4r_restores_bf16(ft, engine):
    """(ii) C4r（thr＝2,048 token＜重存時的長度 3,072）：重算後存成 BF16，下一個長讀取不再重算；對照 C2。"""
    r, e, cf, sf, wst = _restore_run(engine, ft, "C4r", 1024, pol_kw=dict(c4_threshold_tokens=2048))
    assert [int(x) for x in e.surp_blk] == [0, 0, 4, 0], list(e.surp_blk)
    assert r["surprise_events"] == 1 and r["surprise_blocks"] == 4
    assert all(cf(b) == 1 for b in range(7)), [cf(b) for b in range(7)]          # 0..5 重存成 BF16；6 是 r3 的新 block（3,584 token＞thr）
    assert all(cf(b) == 3 for b in (10, 11, 12, 13))
    assert wst == 8 * (20 << 20) + 7 * (64 << 20), wst  # r0、r1 各 4 個 int4（=8）；r2 存 6 個 bf16、r3 存 1 個 bf16（=7）


@pytest.mark.parametrize("engine", ["session", "fifo1"])
def test_recompute_restore_serve_unchanged(ft, engine):
    """serve 照舊：不重算、不重存（C2 的 block 一直是 int4，來歷也不變）。品質帳照舊記（請求長度 → fail）。"""
    P = _hand_P_restore()
    if engine == "session":
        r, _, ids, e, _ = sim.run_cell(P, "E2", "C2", 1.0, "llama", SPECS[:1], fmt_path=ft, check=1, unverified_len="request")
        wst = e.wbytes[W_STORE]
    else:
        r, _, ids, e, _ = sim.run_cell_fifo1(P, "E2", "C2", 1.0, "llama", SPECS[0], fmt_path=ft, unverified_len="request")
        wst = e.wbytes["store"]
    assert all(i["ok"] for i in ids) and r[0]["surprise_events"] == 0 and int(e.surp_blk.sum()) == 0
    assert wst == (4 + 4 + 2 + 1) * (20 << 20), wst
    assert r[0]["reads_fail"] > 0


@pytest.mark.parametrize("engine", ["session", "fifo1"])
def test_recompute_restore_oracle_no_surprise(ft, engine):
    """oracle 在嚴格規則下沒有意外重算，所以重存規則不碰它們。"""
    for pol in ("O", "O-len"):
        r, e, _, _, _ = _restore_run(engine, ft, pol, 1024)
        assert r["surprise_events"] == 0 and int(e.surp_blk.sum()) == 0, pol


# ===================================================================== 2026-10-18（04 §5）只用量到的 3090 常數：/ssd8 的 k_read 沒量
def test_ssd8_k_read_not_borrowed():
    c = load_costs("3090", "llama", "ssd8_3090_meas")
    assert c.ssd_k_nm and c.ssd["k"] == 1.0 and c.ssd["k"] != 3.306
    assert "NOT_MEASURED" in c.src["ssd"] and "3.306" not in c.src["ssd"]
    assert not load_costs("mi300x", "llama").ssd_k_nm
    with pytest.raises(NotImplementedError):                      # block 引擎沒有這個追蹤
        BView(c, load_format_table("llama", "mi300x"), "idle", 8)


def _k_view(ft, k_nm, tl):
    P = _hand_P_strict()
    P = dict(P, tl=tl)
    mf = load_format_table("llama", "mi300x", ft)
    costs = load_costs("mi300x", "llama")
    costs.ssd = dict(costs.ssd, k=1.0)
    costs.ssd_k_nm = k_nm
    v = View(costs, mf, "idle", "cake", 64, 3)
    e = Engine(P, sim.outer_cfg("E2"), make_policy("C0", mf), 4 * BF, [v], mf, check=1).run()
    return e, v


def test_ssd8_k_read_taint(ft):
    """手造：r1 把 s0 的 4 個 BF16 block 降級寫 SSD（256 MB，寫 ≈0.13 s）。r2 馬上（10.01 s）回來 → 讀 SSD 時寫入還沒完 → k_read 乘上去 → NOT_MEASURED；
    r2 晚到（30 s）→ 不碰 k_read → 有數字。k_nm 只加旗標，不改任何時間（和 k_nm＝False、k＝1.0 逐位元相同）。"""
    for tl, hit in (([0.0, 10.0, 10.01], True), ([0.0, 10.0, 30.0], False)):
        e0, v0 = _k_view(ft, False, tl)
        e1, v1 = _k_view(ft, True, tl)
        assert np.array_equal(v0.ttft, v1.ttft) and v0.slow.bytes_r == v1.slow.bytes_r and np.array_equal(v0.nC, v1.nC)
        assert v1.nS[2] > 0, "r2 要從 SSD 讀"
        assert bool(v1.nmk[2]) is hit and bool(v1.nmreq[2]) is hit, (tl, list(v1.nmk))
        assert not v1.nmk[0] and not v1.nmk[1] and not v0.nmreq.any()
        # 傳遞：碰到 k_read 的請求，它的 TTFT 決定了結尾的寫入時刻 → CPU 寫入佇列的結束時刻也算由 k_read 推出
        assert v1.cw_ta is hit, (tl, v1.cw_ta)


# ===================================================================== 2026-10-18 第二輪審查之後（04 v4 §3.1：C1-2、R2、長度規則 max）
@pytest.mark.parametrize("engine", ["session", "fifo1"])
def test_r2_single_request_int2_then_recompute_restore_bf16(ft, engine):
    """(i) R2：session 只來過一次 → INT2；它回來時讀到沒驗證的 INT2 → 重算（4 個 block），重算出來的 KV 照當時的決定（這時已 2 個請求 → BF16）重存；
    第 3 個請求不再重算。s1 只來過一次，一直是 INT2。"""
    r, e, cf, sf, wst = _restore_run(engine, ft, "R2", 1024, pol_kw=dict(len_rule="max"))
    I2, BFb = 12 << 20, 64 << 20
    assert [int(x) for x in e.surp_blk] == [0, 0, 4, 0], list(e.surp_blk)
    assert wst == 8 * I2 + 7 * BFb, wst              # r0、r1 各 4 個 INT2；r2 存 6 個 BF16（重算 4＋新 2）、r3 存 1 個 BF16
    assert all(cf(b) == FID["bf16"] for b in range(7)), [cf(b) for b in range(7)]
    assert all(cf(b) == FID["int2"] for b in (10, 11, 12, 13))


@pytest.mark.parametrize("engine", ["session", "fifo1"])
def test_c1_2_ssd_int2_recompute_and_restore(ft, engine):
    """(ii) C1-2（CPU BF16、SSD INT2）：s0 被擠到 SSD 後是 INT2；s0 回來時 SSD 上的 INT2 不能用 → 重算（4 個 block）並以 BF16 重存到 CPU
    （SSD 的舊 INT2 副本作廢）；品質帳 0、恆等式成立。"""
    r, e, cf, sf, wst = _restore_run(engine, ft, "C1-2", 450)
    assert int(e.surp_blk[2]) == 4 and int(e.surp_blk.sum()) >= 4, list(e.surp_blk)
    assert all(cf(b) == FID["bf16"] for b in range(6)), [cf(b) for b in range(6)]
    assert all(sf(b) in (0, FID["int2"]) for b in range(7)), [sf(b) for b in range(7)]
    assert r["reads_fail"] == 0 and r["reads_unverified"] == 0


@pytest.mark.parametrize("engine", ["session", "fifo1"])
def test_len_rule_max_uses_request_length(ft, engine):
    """(iii) len_rule="max"：長度門檻看 max(存的長度, 請求輸入長度)。只有 r0（一個 session、請求輸入長度 4,096 token＝nall 8、存 4 個 block＝2,048 token）；
    CPU 只放得下 1 個 block → 其餘被搬去 SSD。D-len(thr 2,048)：stored 規則 2,048 ≤ thr → 搬去 SSD 時壓成 INT4；max 規則＝max(2,048, 4,096) > thr → 維持 BF16。"""
    P = dict(full=[[0, 1, 2, 3]], nall=[8], acc={b: [0] for b in range(4)}, tl=[0.0], kindc=[0], sessl=[0], parentl=[-1], obs_end=[1e6], nmax=64,
             B=4, R=dict(n_sess=1), win=(0.0, 100.0), meta=dict(source="hand", variant="len", tier="S", n_req=1, n_sess=1))
    out = {}
    for rule in ("stored", "max"):
        kw = dict(int2_on_read="recompute_unverified", unverified_len="request")
        pk = dict(d_threshold_tokens=2048, len_rule=rule)
        if engine == "session":
            r, _, ids, e, _ = sim.run_cell(P, "E2", "D-len", 70 / 1024.0, "llama", SPECS[:1], fmt_path=ft, check=1, pol_kw=pk, **kw)
        else:
            r, _, ids, e, _ = sim.run_cell_fifo1(P, "E2", "D-len", 70 / 1024.0, "llama", SPECS[0], fmt_path=ft, pol_kw=pk, **kw)
        assert all(i["ok"] for i in ids)
        out[rule] = {k: v for k, v in r[0].items() if k.startswith("wblocks_ssd_")}
    assert out["stored"].get("wblocks_ssd_int4", 0) > 0 and out["stored"].get("wblocks_ssd_bf16", 0) == 0, out
    assert out["max"].get("wblocks_ssd_bf16", 0) > 0 and out["max"].get("wblocks_ssd_int4", 0) == 0, out


# ===================================================================== 第二輪審查：寬鬆規則、E6 的實際讀取（還原＋預取）、截尾版護欄欄
def test_loosened_quality_rule_official():
    """寬鬆版雙胞胎的品質規則：32K 過關的格式任何長度都過關，其他不變；大小、解壓時間、never_read_only 不變。"""
    mf = load_format_table("llama", "mi300x")
    lo = mf.loosened()
    assert lo.bytes == mf.bytes and lo.never_read_only == mf.never_read_only and lo.src.get("quality_rule") == "loose" and "quality_rule" not in mf.src
    for f in mf.available:
        for tok in (1000, 32768, 70000, 500000):
            want = mf.status(f, 32768) if f != FID["bf16"] else "pass"
            assert lo.status(f, tok) == want, (f, tok)
    assert mf.status(FID["int4"], 100000) != "pass" and lo.status(FID["int4"], 100000) == mf.status(FID["int4"], 32768)


@pytest.mark.parametrize("engine", ["session", "fifo1"])
def test_loose_rule_removes_unverified_recompute(engine):
    """C2（全 INT4）：請求長度 51,200 token（>32K）。strict：INT4 來歷在這個長度沒驗證 → 重算（意外重算 >0）；loose：INT4 在 32K 過關 → 不重算。品質帳都是 0。"""
    P = _hand_P_restore()
    P = dict(P, nall=[4, 4, 100, 101], nmax=128)
    res = {}
    for rule in ("strict", "loose"):
        kw = dict(int2_on_read="recompute_unverified", unverified_len="request", fmt_path="official", quality_rule=rule)
        if engine == "session":
            r, _, ids, e, _ = sim.run_cell(P, "E2", "C2", 150 / 1024.0, "llama", SPECS[:1], check=1, **kw)
        else:
            r, _, ids, e, _ = sim.run_cell_fifo1(P, "E2", "C2", 150 / 1024.0, "llama", SPECS[0], **kw)
        assert all(i["ok"] for i in ids)
        assert r[0]["reads_fail"] == 0 and r[0]["reads_unverified"] == 0
        res[rule] = r[0]["surprise_events"]
    # official 格式表的 Llama 沒有 INT4 的 32K 以上資料 → strict 在 51,200 token 重算；loose 用 32K 的結果 → 不重算
    assert res["strict"] > 0 and res["loose"] == 0, res


def test_block_actual_read_columns_and_attribution(ft):
    """block 引擎（E6）：ssd_H_actual_read_GiB＝實際 SSD 讀取（還原＋預取，含浪費的）中到達 < 終點 − H 的請求；per-session 的 ssd_read_bytes_H 加總相符；
    預取讀取歸屬於觸發預取的請求的 session；全部實際讀取（不截尾）＝SSD 通道讀取的位元組；session 引擎沒有預取 → 實際讀取＝還原讀取。
    另外 ttftq_H_*／ttftq_H2_*（截尾版護欄）只算到達 < 終點 − H（－2H）的回來請求。"""
    import ledger as LG
    P = _hand_P_restore()
    P = dict(P, tl=[0.0, 0.5, 1.0, 1.5], obs_end=[100.0] * 4, win=(0.0, 100.0))
    for H in (75.0, 1e-3 + 0.0):
        r, _, ids, e, vs = sim.run_cell_fifo1(P, "E6", "C2", 150 / 1024.0, "llama", SPECS[0], fmt_path=ft, H=H, H_extra=2 * H,
                                              int2_on_read="recompute_unverified", unverified_len="request")
        assert all(i["ok"] for i in ids)
        v = vs[0]
        act = LG.actual_read_by_req(v, 4)
        row = r[0]
        assert abs(row["ssd_actual_read_GiB"] * GiB - act.sum()) < 1 and abs(act.sum() - v.slow.bytes_r) < 1
        cut = 100.0 - H
        want = act[np.asarray(P["tl"]) < cut].sum()
        assert abs(row["ssd_H_actual_read_GiB"] * GiB - want) < 1, (row["ssd_H_actual_read_GiB"], want)
        want2 = act[np.asarray(P["tl"]) < 100.0 - 2 * H].sum()
        assert abs(row["ssd_H2_actual_read_GiB"] * GiB - want2) < 1
        ps = LG.per_session(P, e, vs)
        assert ps["ssd_read_bytes_H"].shape == (1, 2) and int(ps["ssd_read_bytes_H"][0].sum()) == int(round(want))
        assert int(ps["ssd_read_bytes_H2"][0].sum()) == int(round(want2))
        sl = np.asarray(P["sessl"])
        assert [int(x) for x in ps["ssd_read_bytes"][0]] == [int(round(act[sl == s_].sum())) for s_ in (0, 1)]   # 全部：依觸發的請求歸屬
        # 截尾版護欄欄存在、且只看截尾內的回來請求
        ret = np.asarray(P["kindc"]) == 1
        m = ret & (np.asarray(P["tl"]) < cut)
        assert abs(row["ttftq_H_mean_s"] - float(np.asarray(v.ttft)[m].mean())) < 1e-12 if m.any() else math.isnan(row["ttftq_H_mean_s"])
        m2 = ret & (np.asarray(P["tl"]) < 100.0 - 2 * H)
        assert abs(row["ttftq_H2_mean_s"] - float(np.asarray(v.ttft)[m2].mean())) < 1e-12 if m2.any() else math.isnan(row["ttftq_H2_mean_s"])
    # session 引擎：沒有預取 → 實際讀取＝還原讀取（bS）
    r, _, ids, e, vs = sim.run_cell(P, "E2", "C2", 150 / 1024.0, "llama", SPECS[:1], fmt_path=ft, H=75.0, int2_on_read="recompute_unverified",
                                    unverified_len="request")
    assert abs(r[0]["ssd_actual_read_GiB"] * GiB - float(np.asarray(vs[0].bS).sum())) < 1


# ===================================================================== 第三輪審查之後（06 §26）：重算計價、O-tier、E6 主指標、P 快取鍵
@pytest.mark.parametrize("engine", ["session", "fifo1"])
def test_recompute_charge_hand(ft, engine):
    """重算計價（選項 a；手算）：C1-2（CPU BF16、SSD INT2）＋手造 4 個請求（_hand_P_restore；測試表：8bit 33 MiB、bf16 64 MiB、int4 在 ≥16K 不過關）。
    s0 被擠到 SSD 變成 INT2，在 r2（輸入 20,480 token）回來：INT2 不能用 → 意外重算。計價的單位＝這個長度下過關的最省格式（8bit，33 MiB）。
    session 引擎：s0 整個 session 都降到 SSD → 4 個 block 都只在 SSD → 4×33 MiB。block 引擎（block 層級 LRU，CPU 450 MB 放得下 7 個 BF16）：
    只有被擠掉的 block 0 在 SSD、block 1–3 還在 CPU（CPU 副本不算 SSD 讀取）→ 1×33 MiB。其他請求（沒有意外重算）0。
    recompute_charge＝none：主指標讀取不含；pass_size：含（欄位 ssd_H_primary_read_GiB）；兩種的分量欄位（charged／primread）一樣；per-session 的 npz 歸屬於讀它的 session。"""
    import ledger as LG
    P = _hand_P_restore()
    MiB = 1 << 20
    want_blocks, want_bytes = (4, 4 * 33 * MiB) if engine == "session" else (1, 33 * MiB)
    outs = {}
    for chg in ("none", "pass_size"):
        kw = dict(fmt_path=ft, int2_on_read="recompute_unverified", unverified_len="request", H=1.0, recompute_charge=chg)
        if engine == "session":
            r, _, ids, e, vs = sim.run_cell(P, "E2", "C1-2", 450 / 1024.0, "llama", SPECS[:1], check=1, **kw)
        else:
            r, _, ids, e, vs = sim.run_cell_fifo1(P, "E2", "C1-2", 450 / 1024.0, "llama", SPECS[0], **kw)
        assert all(i["ok"] for i in ids)
        outs[chg] = (r[0], e, vs)
        assert [int(x) for x in e.rc_blocks] == [0, 0, want_blocks, 0], list(e.rc_blocks)
        assert [int(x) for x in e.rc_bytes] == [0, 0, want_bytes, 0], list(e.rc_bytes)
        row = r[0]
        assert row["recompute_charge"] == chg and abs(row["ssd_H_charged_recompute_GiB"] * GiB - want_bytes) < 1 and row["recompute_charge_blocks"] == want_blocks
        assert row["reads_fail"] == 0 and row["reads_unverified"] == 0
    a, b = outs["none"][0], outs["pass_size"][0]
    assert a["ssd_H_primread_GiB"] == b["ssd_H_primread_GiB"]                                 # 分量一樣
    assert a["ssd_H_primary_read_GiB"] == a["ssd_H_primread_GiB"]                             # none：不含
    assert abs((b["ssd_H_primary_read_GiB"] - b["ssd_H_primread_GiB"]) * GiB - want_bytes) < 1  # pass_size：含
    ps = LG.per_session(P, outs["pass_size"][1], outs["pass_size"][2])
    assert [int(x) for x in ps["recompute_charge_bytes_H"]] == [want_bytes, 0] and [int(x) for x in ps["recompute_charge_bytes"]] == [want_bytes, 0]   # 歸屬於讀它的 session s0
    # 沒有意外重算的做法（C0）：計價 0
    kw0 = dict(fmt_path=ft, int2_on_read="recompute_unverified", unverified_len="request", H=1.0, recompute_charge="pass_size")
    if engine == "session":
        r0, _, _, e0, _ = sim.run_cell(P, "E2", "C0", 450 / 1024.0, "llama", SPECS[:1], **kw0)
    else:
        r0, _, _, e0, _ = sim.run_cell_fifo1(P, "E2", "C0", 450 / 1024.0, "llama", SPECS[0], **kw0)
    assert int(np.asarray(e0.rc_bytes).sum()) == 0 and r0[0]["ssd_H_charged_recompute_GiB"] == 0.0


def _hand_P_wt():
    """手造：s0 在 r0 寫 4 個 block，在 r1 回來（輸入 40 個 block＝20,480 token）：E3（寫穿）的 SSD 副本寫了以後從來沒有從 SSD 讀過（一直在 CPU）。"""
    full = [[0, 1, 2, 3], [0, 1, 2, 3, 4, 5]]
    acc = {}
    for r, ks in enumerate(full):
        for b in ks:
            acc.setdefault(b, []).append(r)
    return dict(full=full, nall=[4, 40], acc=acc, tl=[0.0, 10.0], kindc=[0, 1], sessl=[0, 0], parentl=[-1, 0],
                obs_end=[1e6] * 2, nmax=64, B=6, R=dict(n_sess=1), win=(0.0, 100.0),
                meta=dict(source="hand", variant="wt", tier="S", n_req=2, n_sess=1))


def test_otier_hand_never_read_from_ssd_becomes_int2(ft):
    """O-tier（選項 b；手算）：E3 寫穿，s0 的 4 個 block 寫了 SSD 副本以後只在 CPU 被讀（r1 輸入 20,480 token）。
    一般的 O：之後 r1 會讀 → SSD 副本用 20,480 token 下過關的最省格式＝8bit（33 MiB）；O-tier：這些 SSD 副本從沒被從 SSD 讀 → INT2（12 MiB）。
    block 4、5（r1 新寫的，之後沒人讀）兩個都是 INT2。SSD 寫入：O＝4×33＋2×12 MiB、O-tier＝6×12 MiB；CPU 側事件相同（引擎比對）；品質帳 0、沒有意外重算。"""
    P = _hand_P_wt()
    MiB = 1 << 20
    res = {}
    for pol in ("O", "O-tier"):
        r, _, ids, e, _ = sim.run_cell(P, "E3", pol, 1024 / 1024.0, "llama", SPECS[:1], fmt_path=ft, check=1, H=1.0,
                                       int2_on_read="recompute_unverified", unverified_len="request")
        assert all(i["ok"] for i in ids)
        res[pol] = (r[0], e)
        assert r[0]["reads_fail"] == 0 and r[0]["reads_unverified"] == 0 and r[0]["surprise_events"] == 0
    assert res["O"][1].wbytes["write_through"] == 4 * 33 * MiB + 2 * 12 * MiB, res["O"][1].wbytes
    assert res["O-tier"][1].wbytes["write_through"] == 6 * 12 * MiB, res["O-tier"][1].wbytes
    t = res["O-tier"][0]
    assert t["otier_cpu_event_mismatch_requests"] == 0 and t["otier_approx"] is False and t["otier_pass1_ssd_writes"] == t["otier_pass2_ssd_writes"] == 6
    assert t["otier_ssd_copies_never_read"] == 6               # 6 份 SSD 副本都沒被從 SSD 讀過
    assert t["policy"] == "O-tier" and res["O"][0]["policy"] == "O"
    # CPU 這一階的格式和一般的 O 完全相同
    assert [res["O"][1].cfmt[b] for b in range(6)] == [res["O-tier"][1].cfmt[b] for b in range(6)]


def test_otier_ssd_read_keeps_pass_format_and_assert_trips(ft, monkeypatch):
    """(i) 有被從 SSD 讀的副本：O-tier 的格式要在「第一次從 SSD 讀之後的所有讀取」的最長長度下過關（_hand_P_restore：s0 在 r2、r3 回來，輸入 20,480／20,992 token → 8bit），
    所以位元組和一般的 O 一樣、品質帳 0、沒有意外重算。(ii) 第 2 趟的 CPU 側事件和第 1 趟不同就丟錯：把「有沒有被從 SSD 讀」硬改成永遠沒有 → INT2 被讀到 → 意外重算 → 事件不同 → AssertionError。"""
    import policies
    P = _hand_P_restore()
    kw = dict(fmt_path=ft, check=1, H=1.0, int2_on_read="recompute_unverified", unverified_len="request")
    ro, _, _, eo, _ = sim.run_cell(P, "E2", "O", 150 / 1024.0, "llama", SPECS[:1], **kw)      # 150 MB：s0 的 8bit block 被 s1 擠到 SSD，之後從 SSD 讀回
    rt, _, ids, et, _ = sim.run_cell(P, "E2", "O-tier", 150 / 1024.0, "llama", SPECS[:1], **kw)
    assert all(i["ok"] for i in ids) and rt[0]["otier_cpu_event_mismatch_requests"] == 0
    assert ro[0]["ssd_H_write_issued_GiB"] > 0 and rt[0]["otier_pass1_ssd_writes"] > 0 and rt[0]["ssd_H_primread_GiB"] > 0      # 真的有 SSD 寫入與 SSD 讀取
    assert rt[0]["ssd_H_write_issued_GiB"] == ro[0]["ssd_H_write_issued_GiB"] and rt[0]["reads_fail"] == 0 and rt[0]["surprise_events"] == 0
    monkeypatch.setattr(policies.TierRec, "info", lambda self, key: (False, 0, 0) if key in self.keys else None)
    with pytest.raises(AssertionError, match="CPU 側事件"):
        sim.run_cell(P, "E2", "O-tier", 150 / 1024.0, "llama", SPECS[:1], **kw)


def test_otier_block_engine_reports_mismatch_and_approx(ft):
    """block 引擎（和時間有關）：兩趟的 CPU 側事件只能近似相同 → 不丟錯，數出不同的請求數（這個小例子 0），標 otier_approx。"""
    P = dict(_hand_P_restore(), tl=[0.0, 0.5, 1.0, 1.5])
    r, _, ids, e, _ = sim.run_cell_fifo1(P, "E6", "O-tier", 150 / 1024.0, "llama", SPECS[0], fmt_path=ft, H=1.0,
                                         int2_on_read="recompute_unverified", unverified_len="request")
    assert all(i["ok"] for i in ids)
    row = r[0]
    assert row["otier_approx"] is True and row["otier_cpu_event_mismatch_requests"] == 0 and row["otier_tier_missing_keys"] == 0
    assert row["reads_fail"] == 0 and row["reads_unverified"] == 0 and row["surprise_events"] == 0


def test_loose_variant_tier_policy_exists(ft):
    """O-loose-tier（便宜的版本）：做得出來、第 1 趟用 O-loose；O-len 沒有 tier 版（只知道最長讀取，加了 tier 資訊就不是它）。"""
    P = _hand_P_wt()
    r, _, ids, e, _ = sim.run_cell(P, "E3", "O-loose-tier", 1.0, "llama", SPECS[:1], fmt_path=ft, check=1, H=1.0,
                                   int2_on_read="recompute_unverified", unverified_len="request", quality_rule="loose")
    assert all(i["ok"] for i in ids) and r[0]["policy"] == "O-loose-tier" and r[0]["otier_cpu_event_mismatch_requests"] == 0
    with pytest.raises(ValueError):
        make_policy("O-len-tier", load_format_table("llama", "mi300x", ft))


def test_e6_prefetch_skips_blocks_that_fail_at_trigger_length(ft):
    """E6 預取（嚴格讀取規則；直接在引擎狀態上呼叫 queue_prefetch）：排隊中的請求 r2（輸入 20,480 token）要的 block 0–3 在 SSD 上是 int4（在那個長度不過關），
    服務時會整段重算 → 預取它們是白讀 → 跳過（pf_skipped、預取讀取 0）；關掉跳過時同一個狀態預取 4 個 block。SSD 上是 8bit（過關）→ 照預取 4 個。
    主指標的讀取（ledger.primread_by_req）＝只載入的還原＋全部預取。"""
    from engine_block import BlockEngine, BView, reqs_from_P
    import ledger as LG
    mf = load_format_table("llama", "mi300x", ft)
    costs = load_costs("mi300x", "llama", "mi300x_local")
    P = dict(_hand_P_restore(), tl=[0.0, 0.5, 1.0, 1.5])
    res = {}
    for sfmt_name, skip in (("int4", True), ("int4", False), ("8bit", True)):
        reqs = reqs_from_P(P)
        v = BView(costs, mf, "idle", 64, d2h_Bps=costs.cpu["w"] * GiB)
        eng = BlockEngine(reqs, sim.outer_cfg("E6"), make_policy("C2", mf), 10 * BF, v, mf, qp=True, int2_on_read="recompute_unverified",
                          unverified_len="request", H=1.0, obs_end=P["obs_end"])
        eng.pf_skip = skip
        for b in range(4):
            eng.sfmt[b] = FID[sfmt_name]
            eng.sq[b] = FID[sfmt_name]
        eng.queue_prefetch(1, 0.0, 100.0)
        res[(sfmt_name, skip)] = (eng.S["qp_reads"], eng.S["pf_skipped"], eng.wbytes["prefetch_read"])
    assert res[("int4", True)] == (0, 2, 0), res          # 排隊中的 r2 與 r3（輸入都 ≥20K）各跳過一次
    assert res[("int4", False)][0] == 4 and res[("int4", False)][1] == 0 and res[("int4", False)][2] == 4 * (20 << 20), res
    assert res[("8bit", True)][0] == 4 and res[("8bit", True)][1] == 0, res
    # 整個 run 上：主指標的讀取＝lo_bytes＋預取（含浪費）
    r, _, ids, e, vs = sim.run_cell_fifo1(P, "E6", "C2", 150 / 1024.0, "llama", SPECS[0], fmt_path=ft, H=1.0, int2_on_read="recompute_unverified",
                                          unverified_len="request")
    assert all(i["ok"] for i in ids)
    pr = LG.primread_by_req(e, vs[0], 4)
    assert abs(pr.sum() - (float(np.asarray(e.lo_bytes).sum()) + e.wbytes["prefetch_read"])) < 1
    assert abs(r[0]["ssd_primread_GiB"] * GiB - pr.sum()) < 1


def test_prepare_cache_key_includes_unified_file_hash_and_trace_py(tmp_path, monkeypatch):
    """P 快取的鍵含 (i) trace.py（和 schema.py）的 hash、(ii) 統一表 turns.parquet 檔案本身的 sha256（每次重算；不信 manifest 記的）。
    假的統一表換內容（manifest 不動）→ 鍵變；換 trace 程式的 hash → 鍵變。"""
    import hashlib
    import json as js
    import pickle  # noqa: F401
    import schema
    src, var = "synth", "small-seed1"
    real_file, real_sha = TR.unified_file_sha256(src, var)
    h = hashlib.sha256(open(real_file, "rb").read()).hexdigest()
    assert real_sha == h
    keys = []
    for patch in ("none", "file", "code"):
        if patch == "file":
            monkeypatch.setattr(TR, "unified_file_sha256", lambda s, v: ("/fake/turns.parquet", "0" * 64))
        elif patch == "code":
            monkeypatch.setattr(TR, "unified_file_sha256", lambda s, v: (real_file, real_sha))
            monkeypatch.setattr(TR, "_code_hash", lambda: "1" * 64)
        cd = str(tmp_path / patch)
        monkeypatch.setattr(TR, "prepare", lambda *a, **k: dict(meta={}, dummy=1))
        TR.prepare_cached(src, var, cache_dir=cd, max_req=5)
        js_files = [f for f in os.listdir(cd) if f.endswith(".json")]
        assert len(js_files) == 1
        m = js.load(open(os.path.join(cd, js_files[0])))
        keys.append(js_files[0])
        assert m["params"]["unified_file_sha256"] in (real_sha, "0" * 64) and "code_sha256" in m["params"]
    assert len(set(keys)) == 3, keys
