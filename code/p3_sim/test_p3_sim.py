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
from engine import Engine, View  # noqa: E402
from engine_block import BlockEngine, BView  # noqa: E402
from formats import Costs, ModelFormats, load_costs, load_format_table  # noqa: E402
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
