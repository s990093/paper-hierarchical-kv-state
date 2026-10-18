"""test_map.py — run_map.py（執行器）與 analyze_map.py（判定）的測試（06_simulator.md §20、§21）。只用 CPU。

fixture：手造的 map_cells.csv ＋ per-session npz，答案是事先算好的（每個 session 的位元組是常數 → bootstrap 的 SE＝0 → CI＝點估計，
所以 GO／NO-GO／ASK／提早 GO／護欄／NOT_MEASURED／0÷0／有效 session 數 都能逐一對答案）；bootstrap 本身另外用有變異的資料測性質
（分層、配對、max-T 不比單格窄、seed 重現）。最後一個測試在合成切片 synth/small-seed1 上跑一小段執行器（不是真實資料）。
執行：TIARA_NO_GPU=1 bash code/run3090.sh p3sim-maptest python -m pytest -q code/p3_sim/test_map.py
"""
from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import analyze_map as AM  # noqa: E402
import run_map as RM  # noqa: E402

GiB = 1 << 30
OUT = os.environ.get("RUN_DIR") or os.path.join(os.environ.get("TMPDIR", "/tmp"), "p3map_test_adhoc")
os.makedirs(OUT, exist_ok=True)


# ===================================================================== fixture
class Fx:
    """spec：{trace: dict(slices=[名字...], S=每個切片的 session 數, n_req=每個 session 的請求數（None＝各 1）, source=來源群組)}。
    bytes_fn(trace, slice, olabel, mem, label, tag) → 每個 session 的位元組（GiB 單位的 float 陣列或常數）；ttft_fn(...) → (mean, p99) 或 None＝NOT_MEASURED。"""

    def __init__(self, tmp, spec, bytes_fn, ttft_fn=None, mems=AM.MEMS):
        self.tmp, self.spec, self.rows = str(tmp), spec, []
        self.bytes_fn, self.ttft_fn = bytes_fn, ttft_fn or (lambda *a: (1.0, 2.0))
        self.build(mems)

    def build(self, mems):
        n = 0
        for trace, sp in self.spec.items():
            for s in sp["slices"]:
                S = sp["S"]
                nreq = np.asarray(sp.get("n_req") if sp.get("n_req") is not None else np.ones(S, np.int64))
                for (ol, mem) in [(o, m) for o, _ in AM.PRIMARY_OUTERS for m in mems]:
                    eng = AM.ENGINE_OF[ol]
                    for label in AM.LABELS:
                        oracle = label in AM.ORACLE_LABELS
                        tags = ("H", "2H") if oracle else ("both",)
                        for tag in tags:
                            xh = np.broadcast_to(np.asarray(self.bytes_fn(trace, s, ol, mem, label, "H"), float), (S,)) * GiB
                            x2 = np.broadcast_to(np.asarray(self.bytes_fn(trace, s, ol, mem, label, "2H"), float), (S,)) * GiB
                            path = os.path.join(self.tmp, f"{n}.npz")
                            n += 1
                            ttft = self.ttft_fn(trace, s, ol, mem, label)
                            d = dict(n_req=nreq, sess_orig=np.arange(S), ttft_n=np.asarray(int(nreq.sum()) if False else 100), ttft_sum=np.asarray(0.0),
                                     ttft_nm=np.asarray(0 if ttft is not None else 1), ttft_hist=np.zeros(len(RM.TTFT_EDGES) - 1, np.int64),
                                     ttft_vals=np.zeros(100))
                            if ttft is not None:
                                m, p = ttft
                                d["ttft_sum"] = np.asarray(m * 100.0)
                                d["ttft_hist"][int(np.searchsorted(RM.TTFT_EDGES, max(p, 1e-4)) - 1)] = 100       # 全部 100 個請求都落在 p 的那一格
                                d["ttft_vals"] = np.full(100, float(p))          # 逐請求的 ttft（精確的合併 p99 用）：100 個都＝p
                            if tag == "both":
                                d.update(ssd_write_bytes_H=(xh * 0.5).astype(np.int64), loadonly_ssd_read_bytes_H=(xh * 0.5).astype(np.int64),
                                         ssd_write_bytes_H2=(x2 * 0.25).astype(np.int64), loadonly_ssd_read_bytes_H2=(x2 * 0.75).astype(np.int64))
                                sfx = ("_H", "_H2")
                                ph, p2 = float(d["ssd_write_bytes_H"].sum() + d["loadonly_ssd_read_bytes_H"].sum()) / GiB, \
                                    float(d["ssd_write_bytes_H2"].sum() + d["loadonly_ssd_read_bytes_H2"].sum()) / GiB
                            else:
                                x = xh if tag == "H" else x2
                                d.update(ssd_write_bytes_H=(x * 0.5).astype(np.int64), loadonly_ssd_read_bytes_H=(x * 0.5).astype(np.int64))
                                sfx = ("_H", "") if tag == "H" else ("", "_H")
                                pv = float(d["ssd_write_bytes_H"].sum() + d["loadonly_ssd_read_bytes_H"].sum()) / GiB
                                ph, p2 = (pv, float("nan")) if tag == "H" else (float("nan"), pv)
                            np.savez(path, **d)
                            self.rows.append(dict(run_id="fx", stage="A1", comp="a1_blk" if eng == "block" else "a1_sess", slice=s, trace_group=trace,
                                                  source_group=sp["source"], synthetic=sp.get("synthetic", False), semi_synthetic=False, overlay=1,
                                                  engine=eng, outer_code=ol.split("-")[0], T_s=1800 if ol.startswith("E5") else "", mem_GiB=mem,
                                                  model="llama", policy_label=label, policy_family=label, oracle=oracle, h_tag=tag, view_group="main",
                                                  view_idx=0, is_primary=True, k_gpus=1, job_id=f"{s}|{ol}|{mem}|{label}|{tag}", npz_path=path,
                                                  npz_sfx_H=sfx[0], npz_sfx_2H=sfx[1], prim_H_GiB=ph, prim_2H_GiB=p2,
                                                  ttftq_mean_s=(ttft[0] if ttft else AM.NOT), ttftq_p99_s=(ttft[1] if ttft else AM.NOT), n_ret=100,
                                                  ret_recompute_prefix_blocks=7, surprise_blocks=0))
        self.csv = os.path.join(self.tmp, "map_cells.csv")
        pd.DataFrame(self.rows).to_csv(self.csv, index=False)

    def analyze(self, mode="full", boot=50):
        return AM.analyze(AM.load_cells(self.csv), mode, boot, verbose=False)


BASE = {"C0": 100.0, "C1": 90.0, "C1-8": 95.0, "C2": 85.0, "C3": 92.0, "C6": 91.0, "C4-8192": 96.0, "C4-16384": 97.0, "C4-32768": 98.0,
        "C4r-8192": 93.0, "C4r-16384": 94.0, "C4r-32768": 99.0, "D-len-8192": 89.0, "D-len-16384": 88.0, "D-len-32768": 87.0, "D-idle-300": 86.0,
        "D-idle-1800": 84.0, "D-idle-3600": 83.0, "D-idle-14400": 82.0}        # 非 oracle：最好的是 D-idle-14400（82）


def mk_bytes(o_frac, cell=None, traces=None):
    """O 的位元組＝最好的對手 × (1 − o_frac)；只有 (cell, traces) 指定的格子用 o_frac，其他格子 O＝最好的對手（省 0）。"""
    def f(trace, s, ol, mem, label, tag):
        if label in AM.ORACLE_LABELS:
            on = (cell is None or (ol, mem) == cell) and (traces is None or trace in traces)
            fr = o_frac if on else 0.0
            return {"O": 82.0 * (1 - fr), "O-len": 82.0 * (1 - fr) + 5, "O-reuse": 82.0 * (1 - fr) + 8, "O-loose": 82.0 * (1 - fr) - 3}[label]
        return BASE[label]
    return f


def S5(S=120, fi=False):
    spec = {"Bailian coder": dict(slices=["a/coder"], S=S, source="alibaba"), "Bailian traceA": dict(slices=["a/traceA"], S=S, source="alibaba"),
            "Mooncake": dict(slices=["m/c"], S=S, source="mooncake"), "WildChat": dict(slices=["w/1m"], S=S, source="wildchat")}
    if fi:
        spec["FreeInference"] = dict(slices=["f/w01", "f/w05"], S=S, source="freeinference")
    return spec


# ===================================================================== 判定（事先算好的答案）
def test_go_case(tmp_path):
    """Bailian coder 的 (E2, 63) 省 30%（≥15%）、不同來源的 WildChat 同一格省 10%（≥5%）→ GO；同來源的 traceA 省 10% 不算。"""
    def f(trace, s, ol, mem, label, tag):
        if label in AM.ORACLE_LABELS and (ol, mem) == ("E2", 63.0):
            fr = {"Bailian coder": 0.30, "WildChat": 0.10, "Bailian traceA": 0.10, "Mooncake": 0.0}[trace]
            return 82.0 * (1 - fr) + {"O": 0, "O-len": 5, "O-reuse": 8, "O-loose": -3}[label]
        return mk_bytes(0.0)(trace, s, ol, mem, label, tag)
    fx = Fx(tmp_path, S5(), f)
    r = fx.analyze()
    v = r["verdict"]
    assert v["verdict"] == "GO", v
    assert v["go_cells"][0]["cell"] == ("E2", 63.0) and v["go_cells"][0]["strong"] == "Bailian coder" and v["go_cells"][0]["weak"] == "WildChat"
    row = next(x for x in r["rows"] if x["trace"] == "Bailian coder" and x["outer"] == "E2" and x["mem_GiB"] == 63.0 and x["h"] == "H")
    assert abs(row["savings"] - 0.30) < 1e-9 and row["best_label"] == "D-idle-14400" and abs(row["best_GiB"] - 82.0 * 120) < 1e-6
    assert abs(row["ci_lo"] - 0.30) < 1e-9 and abs(row["ci_hi"] - 0.30) < 1e-9        # 每個 session 一樣 → SE＝0 → CI＝點
    # 同來源（traceA）當另一份資料：不算（拿掉 WildChat 的 10%）
    def f2(trace, s, ol, mem, label, tag):
        if trace == "WildChat" and label in AM.ORACLE_LABELS:
            return BASE["D-idle-14400"]
        return f(trace, s, ol, mem, label, tag)
    r2 = Fx(_sub(tmp_path, "g2"), S5(), f2).analyze()
    assert r2["verdict"]["verdict"] == "ASK", r2["verdict"]          # traceA 的 10% 和 coder 同一家
    # 弱的那份只有 4.9%（<5%）→ 不算 GO
    def f3(trace, s, ol, mem, label, tag):
        if trace in ("WildChat", "Bailian traceA") and label in AM.ORACLE_LABELS:
            return 82.0 * (1 - 0.049) + {"O": 0, "O-len": 5, "O-reuse": 8, "O-loose": -3}[label] if (ol, mem) == ("E2", 63.0) else 82.0
        return f(trace, s, ol, mem, label, tag)
    r3 = Fx(_sub(tmp_path, "g3"), S5(), f3).analyze()
    assert r3["verdict"]["verdict"] == "ASK"


def _sub(tmp, name):
    p = os.path.join(str(tmp), name)
    os.makedirs(p, exist_ok=True)
    return p


def test_nogo_and_ask_cases(tmp_path):
    """所有資料所有主格都省 2%（CI 上界 <5%）→ NO-GO；都省 10%（不到 15% 也不是 <5%）→ ASK。"""
    r = Fx(_sub(tmp_path, "n"), S5(), mk_bytes(0.02)).analyze()
    assert r["verdict"]["verdict"] == "NO-GO", r["verdict"]
    assert all(x["ci_hi"] < 0.05 for x in r["rows"] if x["h"] == "H")
    r = Fx(_sub(tmp_path, "a"), S5(), mk_bytes(0.10)).analyze()
    assert r["verdict"]["verdict"] == "ASK", r["verdict"]
    # 一份不完整的資料（缺一個主格）→ 不能 NO-GO（ASK）
    fx = Fx(_sub(tmp_path, "i"), S5(), mk_bytes(0.02))
    df = pd.read_csv(fx.csv, low_memory=False)
    df = df[~((df.trace_group == "Mooncake") & (df.outer_code == "E4") & (df.mem_GiB == 435.0))]
    df.to_csv(fx.csv, index=False)
    r = fx.analyze()
    assert r["verdict"]["verdict"] in ("ASK", "GO") and r["verdict"]["verdict"] != "NO-GO" and not r["traces"]["Mooncake"]["complete"]


def test_early_go_rule(tmp_path):
    """§10：不含 FreeInference 的資料已滿足 GO → 早期定案；只有 FreeInference 滿足 → PENDING_B（full 模式才 GO）。"""
    spec = S5(fi=True)
    # FreeInference 有 3 個有效 session 以上、要 ≥100：用 n_req 讓 eff≥100（每週 S 個 session 各 1 個請求 → 每週 S；兩週合 2S）
    spec["FreeInference"]["S"] = 120

    def f(trace, s, ol, mem, label, tag):
        if label in AM.ORACLE_LABELS and (ol, mem) == ("E3", 170.0):
            fr = {"Bailian coder": 0.30, "WildChat": 0.08}.get(trace, 0.0)
            return 82.0 * (1 - fr) + {"O": 0, "O-len": 5, "O-reuse": 8, "O-loose": -3}[label]
        return BASE.get(label, 82.0)
    spec["Bailian coder"]["S"] = spec["WildChat"]["S"] = 120
    fx = Fx(_sub(tmp_path, "e"), spec, f)
    early = fx.analyze("early")["verdict"]
    assert early["verdict"].startswith("GO") and "早期" in early["verdict"], early
    assert "FreeInference" in early["not_used"]
    # 只有 FreeInference（0.30）＋ WildChat（0.08）滿足：early → PENDING_B；full → GO
    def g(trace, s, ol, mem, label, tag):
        if label in AM.ORACLE_LABELS and (ol, mem) == ("E3", 170.0):
            fr = {"FreeInference": 0.30, "WildChat": 0.08}.get(trace, 0.0)
            return 82.0 * (1 - fr) + {"O": 0, "O-len": 5, "O-reuse": 8, "O-loose": -3}[label]
        return BASE.get(label, 82.0)
    fx = Fx(_sub(tmp_path, "e2"), spec, g)
    assert fx.analyze("early")["verdict"]["verdict"] == "PENDING_B"
    full = fx.analyze("full")["verdict"]
    assert full["verdict"] == "GO" and full["go_cells"][0]["strong"] == "FreeInference", full


def test_ineligible_trace_excluded(tmp_path):
    """有效 session 數 <100 的資料不進判定：即使它省 40% 也不能讓 GO（另一份只有 10%）。"""
    spec = S5()
    spec["Bailian coder"]["S"] = 120          # 有效 120 ≥100
    spec["Bailian traceA"].update(S=120, n_req=np.r_[np.full(1, 10000), np.ones(119)])   # 一個 session 占絕大多數請求 → 有效 session 很少
    spec["WildChat"]["S"] = 120
    spec["Mooncake"]["S"] = 120

    def f(trace, s, ol, mem, label, tag):
        if label in AM.ORACLE_LABELS and (ol, mem) == ("E2", 63.0):
            fr = {"Bailian traceA": 0.40, "WildChat": 0.10}.get(trace, 0.0)
            return 82.0 * (1 - fr) + {"O": 0, "O-len": 5, "O-reuse": 8, "O-loose": -3}[label]
        return BASE.get(label, 82.0)
    r = Fx(_sub(tmp_path, "el"), spec, f).analyze()
    assert not r["traces"]["Bailian traceA"]["eligible"] and r["traces"]["Bailian traceA"]["eff_sessions"] < 100
    assert r["traces"]["Bailian coder"]["eligible"] and abs(r["traces"]["Bailian coder"]["eff_sessions"] - 120) < 1e-9
    assert r["verdict"]["verdict"] != "GO" and "Bailian traceA" in r["verdict"]["not_used"]


def test_guard_excludes_baseline_and_nm(tmp_path):
    """護欄：最省的對手（D-idle-14400，82）等待時間 1.2×C0 → 被排除，最好的變成下一個（D-idle-3600，83）；
    NOT_MEASURED 的最省對手 → 當成通過（仍是最好），並標在 guard_nm；O 不受護欄。"""
    def tt_bad(trace, s, ol, mem, label):
        return (1.2, 2.4) if label == "D-idle-14400" else (1.0, 2.0)

    def f(trace, s, ol, mem, label, tag):
        return mk_bytes(0.30)(trace, s, ol, mem, label, tag)
    r = Fx(_sub(tmp_path, "gb"), S5(), f, tt_bad).analyze()
    row = next(x for x in r["rows"] if x["trace"] == "Mooncake" and x["outer"] == "E2" and x["mem_GiB"] == 63.0 and x["h"] == "H")
    assert row["best_label"] == "D-idle-3600" and "D-idle-14400" in row["guard_excluded"].split(";")
    assert abs(row["savings"] - (83.0 - 82.0 * 0.7) / 83.0) < 1e-9
    # 平均 1.04×、p99 1.06×（p99 超過 1.05）→ 排除；平均 1.06×、p99 1.04× → 也排除；都 1.05×（≤）→ 通過
    ok, nmf = AM.guard_status((1.05, 2.1, False), (1.0, 2.0, False))
    assert ok and not nmf
    assert not AM.guard_status((1.04, 2.13, False), (1.0, 2.0, False))[0]
    assert not AM.guard_status((1.06, 2.08, False), (1.0, 2.0, False))[0]

    def tt_nm(trace, s, ol, mem, label):
        return None if label == "D-idle-14400" else (1.0, 2.0)
    r = Fx(_sub(tmp_path, "gn"), S5(), f, tt_nm).analyze()
    row = next(x for x in r["rows"] if x["trace"] == "Mooncake" and x["outer"] == "E2" and x["mem_GiB"] == 63.0 and x["h"] == "H")
    assert row["best_label"] == "D-idle-14400" and "D-idle-14400" in row["guard_nm"].split(";") and row["guard_excluded"] == ""
    assert AM.guard_status((None, None, True), (1.0, 2.0, False)) == (True, True)


def test_zero_over_zero_cell(tmp_path):
    """0/0：所有非 oracle 都是 0 位元組、O 也是 0 → 省多少定為 0、best_zero 旗標、CI＝[0,0]、沒有 NaN。"""
    def f(trace, s, ol, mem, label, tag):
        if (ol, mem) == ("E4", 435.0):
            return 0.0
        return mk_bytes(0.02)(trace, s, ol, mem, label, tag)
    r = Fx(_sub(tmp_path, "z"), S5(), f).analyze()
    z = [x for x in r["rows"] if x["outer"] == "E4" and x["mem_GiB"] == 435.0 and x["h"] == "H"]
    assert z and all(x["best_zero"] and x["savings"] == 0.0 and x["ci_lo"] == 0.0 and x["ci_hi"] == 0.0 and not math.isnan(x["q"]) for x in z)
    assert r["verdict"]["verdict"] == "NO-GO"           # 0/0 的格子上界 0 <5%，不阻擋 NO-GO
    # best＝0 但 O>0（不合理的資料）也定為 0，不產生 inf
    p, bz = AM.savings_point(0.0, 3.0)
    assert p == 0.0 and bz
    assert math.isnan(AM.savings_point(float("nan"), 1.0)[0])


def test_report_only_never_makes_go(tmp_path):
    """O-loose 省 60%（只報告）不會讓判定變 GO；2H 版、拆解另列。"""
    def f(trace, s, ol, mem, label, tag):
        if label == "O-loose":
            return 82.0 * 0.4
        return mk_bytes(0.0)(trace, s, ol, mem, label, tag)
    r = Fx(_sub(tmp_path, "ro"), S5(), f).analyze()
    assert r["verdict"]["verdict"] in ("ASK", "NO-GO") and r["verdict"]["verdict"] != "GO"
    rep = pd.DataFrame(r["reports"])
    ol = rep[(rep.item == "O-loose") & (rep.h == "H") & (rep.outer == "E2@session") & (rep.mem_GiB == 63.0) & (rep.trace == "Mooncake")]
    assert abs(float(ol.savings.iloc[0]) - 0.6) < 1e-6 or abs(float(ol.savings.iloc[0]) - (82 - 32.8) / 82) < 1e-6
    d1 = rep[(rep.item == "O_minus_Olen")].savings
    assert len(d1) > 0 and abs(float(d1.iloc[0]) - 5.0 / 82.0) < 1e-9          # (O-len 的位元組 − O 的)/最好的對手＝5/82（fixture 裡 O-len 比 O 多 5）
    assert set(rep.kind) >= {"primary_point", "report_oracle_variant", "report_decomp"}


def test_e1_reported_only(tmp_path):
    """E1（沒有 SSD）不進判定也沒有「省多少」：只報等待時間與重算（kind＝report_E1）；有 E1 的列不改判定。"""
    fx = Fx(_sub(tmp_path, "e1"), S5(), mk_bytes(0.02))
    df = pd.read_csv(fx.csv, low_memory=False)
    e1 = df[(df.outer_code == "E2") & (df.engine == "session")].copy()
    e1["outer_code"] = "E1"
    pd.concat([df, e1]).to_csv(fx.csv, index=False)
    r = fx.analyze()
    assert r["verdict"]["verdict"] == "NO-GO" and not any(x["outer"] == "E1" for x in r["rows"])
    rep = pd.DataFrame(r["reports"])
    e = rep[rep.kind == "report_E1"]
    assert len(e) > 0 and set(e.outer) == {"E1@session"} and (e.ret_recompute_prefix_blocks == 7 * 1).all() and e.ttftq_mean_s.astype(float).between(0.99, 1.01).all()
    assert not ((rep.outer == "E1@session") & (rep.kind != "report_E1")).any()          # E1 沒有「省多少」的列


def test_2h_columns_and_npz_check(tmp_path):
    """非 oracle 的 H 與 2H 兩個版本從同一列的兩組欄位／npz 後綴讀；npz 總和和 CSV 對不上就丟錯（規則 3）。"""
    def f(trace, s, ol, mem, label, tag):
        b = mk_bytes(0.30)(trace, s, ol, mem, label, tag)
        return b * (1.0 if tag == "H" else 2.0)
    fx = Fx(_sub(tmp_path, "h2"), S5(), f)
    r = fx.analyze()
    h = next(x for x in r["rows"] if x["trace"] == "Mooncake" and x["outer"] == "E2" and x["mem_GiB"] == 63.0 and x["h"] == "H")
    h2 = next(x for x in r["rows"] if x["trace"] == "Mooncake" and x["outer"] == "E2" and x["mem_GiB"] == 63.0 and x["h"] == "2H")
    assert abs(h["best_GiB"] * 2 - h2["best_GiB"]) < 1e-6 and abs(h["savings"] - h2["savings"]) < 1e-9
    df = pd.read_csv(fx.csv, low_memory=False)
    i = df.index[(df.policy_label == "D-idle-14400") & (df.outer_code == "E2") & (df.mem_GiB == 63.0) & (df.trace_group == "Mooncake")][0]   # 最好的對手
    df.loc[i, "prim_H_GiB"] += 1.0
    df.to_csv(fx.csv, index=False)
    with pytest.raises(AssertionError, match="不符"):
        fx.analyze()


# ===================================================================== bootstrap 的性質
def test_bootstrap_stratified_paired_maxT():
    """分層：層內重抽，層的 session 數不變 → 兩層各自常數時總和固定、SE＝0（沒分層會有變異）；max-T 的 q 不小於單格的 q；seed 重現。"""
    S1, S2 = 10, 30
    strata = [("w1", S1, {("a", 1): (np.full(S1, 1.0), np.full(S1, 0.5))}), ("w2", S2, {("a", 1): (np.full(S2, 100.0), np.full(S2, 50.0))})]
    b = AM.bootstrap_maxT(strata, [("a", 1)], 200, [1, 2])
    assert b["se"][0] == 0.0 and abs(b["theta_hat"][0] - 0.5) < 1e-12 and np.allclose(b["theta_star"], 0.5)
    # 有變異的資料：5 個格子、每個格子的 best／O 是同一組隨機數的不同縮放
    rng = np.random.default_rng(7)
    S = 60
    base = rng.lognormal(size=(S, 5))
    cells = [("c", i) for i in range(5)]
    st = [("t", S, {c: (base[:, i] * 10, base[:, i] * (7 + i * 0.5) * rng.uniform(0.7, 1.3, S)) for i, c in enumerate(cells)})]
    b1 = AM.bootstrap_maxT(st, cells, 400, [3, 4])
    b2 = AM.bootstrap_maxT(st, cells, 400, [3, 4])
    b3 = AM.bootstrap_maxT(st, cells, 400, [3, 5])
    assert np.array_equal(b1["theta_star"], b2["theta_star"]) and not np.array_equal(b1["theta_star"], b3["theta_star"])
    one = [AM.bootstrap_maxT([("t", S, {c: st[0][2][c]})], [c], 400, [3, 4]) for c in cells]
    assert all(b1["q"] >= o["q"] - 1e-9 for o in one), (b1["q"], [o["q"] for o in one])
    assert np.all(b1["hi"] - b1["lo"] >= 0) and np.allclose(b1["hi"] + b1["lo"], 2 * b1["theta_hat"])
    # 配對：同一次重抽用在所有格子 → 兩個完全相同的格子，θ* 逐列相同
    st2 = [("t", S, {("x", 0): st[0][2][("c", 0)], ("x", 1): st[0][2][("c", 0)]})]
    b4 = AM.bootstrap_maxT(st2, [("x", 0), ("x", 1)], 100, [3, 4])
    assert np.array_equal(b4["theta_star"][:, 0], b4["theta_star"][:, 1])


def test_hist_percentile_vs_exact():
    rng = np.random.default_rng(1)
    x = rng.lognormal(mean=1.0, sigma=1.5, size=50000)
    h = RM.ttft_arrays(x, 0)["ttft_hist"]
    for q in (50, 99):
        assert abs(AM.hist_pct(h, RM.TTFT_EDGES, q) / np.percentile(x, q) - 1) < 0.012
    assert math.isnan(AM.hist_pct(np.zeros(5, np.int64), np.arange(6) + 1.0, 99))


def test_effective_sessions():
    assert abs(AM.effective_sessions(np.ones(250)) - 250) < 1e-9
    assert abs(AM.effective_sessions([3, 1]) - 1 / ((3 / 4) ** 2 + (1 / 4) ** 2)) < 1e-12


def test_fi_pooled_guard_from_vals(tmp_path):
    """FreeInference 兩週合起來：護欄用 npz 的逐請求 ttft 合併（p99、平均都精確）；NOT_MEASURED 的週讓整體 NM。"""
    spec = S5(fi=True)
    for t in spec:
        spec[t]["S"] = 120

    def tt(trace, s, ol, mem, label):
        if trace == "FreeInference" and label == "D-idle-14400":
            return (1.0, 2.5) if s == "f/w01" else (1.0, 1.5)         # w01 p99 比 C0 的 2.0 大 25%、w05 小；各週 100 個請求 → 合併 p99 落在中間
        return (1.0, 2.0)
    r = Fx(_sub(tmp_path, "fi"), spec, mk_bytes(0.3), tt).analyze()
    row = next(x for x in r["rows"] if x["trace"] == "FreeInference" and x["outer"] == "E2" and x["mem_GiB"] == 63.0 and x["h"] == "H")
    # 合併的 100 個＠2.5 與 100 個＠1.5：p99＝2.5 → 比 C0 的 2.0 高 25% → 排除
    assert "D-idle-14400" in row["guard_excluded"].split(";") and row["best_label"] == "D-idle-3600"
    assert abs(row["best_GiB"] - 83.0 * 240) < 1e-6          # 兩週各 120 個 session


def test_pooled_p99_exact_vs_numpy(tmp_path):
    """幾週合起來的護欄 p99＝對全部請求直接 np.percentile（精確，不是直方圖）；平均＝總和÷個數；缺 ttft_vals 的 npz 明確拒絕；一週 NOT_MEASURED → 整體 NM。"""
    rng = np.random.default_rng(7)
    weeks = [rng.lognormal(0.5, 1.8, size=n) for n in (1000, 777, 20001)]
    rows = []
    for i, x in enumerate(weeks):
        pth = str(tmp_path / f"w{i}.npz")
        np.savez(pth, **RM.ttft_arrays(x, 0, keep_vals=True))
        rows.append(dict(npz_path=pth, ttftq_mean_s=float(x.mean()), ttftq_p99_s=float(np.percentile(x, 99))))
    m, p, nm = AM.trace_ttft(rows)
    allx = np.concatenate(weeks)
    assert p == float(np.percentile(allx, 99)) and not nm
    assert abs(m - allx.mean()) <= 1e-12 * allx.mean()
    h_est = AM.hist_pct(sum(np.load(r["npz_path"])["ttft_hist"] for r in rows), RM.TTFT_EDGES, 99)
    assert h_est != p and abs(h_est / p - 1) < 0.012        # 直方圖只是近似；這裡確認護欄不再用它
    old = str(tmp_path / "old.npz")
    d = RM.ttft_arrays(weeks[0], 0)
    assert "ttft_vals" not in d
    np.savez(old, **d)
    with pytest.raises(SystemExit):                          # 沒有 ttft_vals → 拒絕（不悄悄退回直方圖）
        AM.trace_ttft([rows[0], dict(npz_path=old, ttftq_mean_s=1.0, ttftq_p99_s=1.0)])
    nmp = str(tmp_path / "nm.npz")
    np.savez(nmp, **RM.ttft_arrays(weeks[0], 3, keep_vals=True))
    assert AM.trace_ttft([rows[0], dict(npz_path=nmp, ttftq_mean_s=1.0, ttftq_p99_s=1.0)])[2] is True
    # 執行器只替有好幾個切片的群組（FreeInference 的週）存逐請求 ttft
    assert RM.keeps_ttft_vals("freeinference/week03") and not RM.keeps_ttft_vals("bailian/coder") and not RM.keeps_ttft_vals("synth/small-seed1")


def test_guard_sensitivity_report(tmp_path):
    """護欄敏感度（04 §4，只報告）：1.05（登記的）、1.10、1.20、不設。D-idle-14400（最省）p99 比 C0 高 15%、D-idle-1800 高 25%、D-idle-3600 高 8%：
    1.05 排除這三個 → 最好的是 C2（85）；1.10 放回 3600；1.20 放回 14400；不設也一樣。判定只用 1.05。"""
    def tt(trace, s, ol, mem, label):
        return {"D-idle-14400": (1.0, 2.3), "D-idle-3600": (1.0, 2.16), "D-idle-1800": (1.0, 2.5)}.get(label, (1.0, 2.0))
    r = Fx(_sub(tmp_path, "gs"), S5(), mk_bytes(0.3), tt).analyze()
    got = {x["guard_thr"]: x for x in r["reports"] if x["kind"] == "report_guard_sens" and x["trace"] == "Bailian coder" and x["outer"] == "E2@session"
           and x["mem_GiB"] == 63.0 and x["h"] == "H"}
    assert set(got) == {"1.05", "1.10", "1.20", "none"}
    assert got["1.05"]["best_label"] == "C2" and abs(got["1.05"]["best_GiB"] - 85.0 * 120) < 1e-6
    assert got["1.10"]["best_label"] == "D-idle-3600" and abs(got["1.10"]["best_GiB"] - 83.0 * 120) < 1e-6
    assert got["1.20"]["best_label"] == "D-idle-14400" and abs(got["1.20"]["best_GiB"] - 82.0 * 120) < 1e-6
    assert got["none"]["best_label"] == "D-idle-14400" and got["none"]["guard_excluded"] == ""
    row = next(x for x in r["rows"] if x["trace"] == "Bailian coder" and x["outer"] == "E2" and x["mem_GiB"] == 63.0 and x["h"] == "H")
    assert row["best_label"] == got["1.05"]["best_label"] and abs(row["savings"] - got["1.05"]["savings"]) < 1e-12      # 判定那一列＝登記的 1.05
    assert abs(got["1.20"]["savings"] - (82.0 - 82.0 * 0.7) / 82.0) < 1e-9 and got["1.20"]["savings"] < got["1.05"]["savings"]      # 護欄越鬆，最好的對手越強，省的越少
    pref = str(tmp_path / "gs" / "out")
    AM.write_outputs(r, pref, "full", "x.csv")
    md = open(pref + "_map.md", encoding="utf-8").read()
    assert "護欄敏感度" in md and "D-idle-3600" in md
    js = json.load(open(pref + "_verdict.json", encoding="utf-8"))
    assert {g["guard_thr"] for g in js["guard_sensitivity"]} == {"1.05", "1.10", "1.20", "none"} and js["guard"] == 1.05


# ===================================================================== 執行器
def test_runner_enumeration_and_counts():
    """階段、元件、個數（04 §5、§10）；非 oracle 19 個、oracle 4 個 × (H, 2H)。"""
    kall = {(s, m, v): dict(k=1, hourmax=0.1, overload=False) for s in RM.SLICES for m in RM.A1_MEM + RM.A2_MEM for v in ("idle", "busy")}
    ids = set()
    for st, n_slices in (("A1", 8), ("A2", 8), ("B-A1", 12), ("B-A2", 12)):
        jobs = RM.enumerate_jobs(st, None, False, (), kall)
        assert len({j["slice"] for j in jobs}) == n_slices
        for j in jobs:
            assert (st, j["id"], j["slice"]) not in ids
            ids.add((st, j["id"], j["slice"]))
    a1 = RM.enumerate_jobs("A1", ["bailian/coder"], False, (), kall)
    assert sum(1 for j in a1 if j["comp"] == "a1_sess") == 4 * 3 * (19 + 4 * 2)           # 4 外層 × 3 記憶體 × (19＋4×2)
    assert sum(1 for j in a1 if j["comp"] == "a1_blk") == 2 * 3 * (19 + 4 * 2)            # E6、E2 × 3 記憶體 × 27，主視角
    a2 = RM.enumerate_jobs("A2", ["bailian/coder"], False, (), kall)
    assert sum(1 for j in a2 if j["comp"] == "a2_mem") == 7 * 5 * 27
    assert sum(1 for j in a2 if j["comp"] == "a2_blk") == 2 * 5 * 2 * 27                   # E6、E2 × 5 記憶體 × (閒、忙) × 27
    assert sum(1 for j in a2 if j["comp"] == "sens_qwen") == 4 * 3 * 27
    assert {j["overlay"] for j in a2 if j["comp"].startswith("sens_ov")} == {2, 4}
    assert len(RM.enumerate_jobs("A2", ["bailian/coder"], True, (), kall)) == len(a2) + 3 * 3 * 27
    assert {j["slice"] for j in RM.enumerate_jobs("A1", None, False, (), kall)} >= {f"freeinference/week{w:02d}" for w in (1, 5, 9, 13)}
    assert not any(j["slice"] == "freeinference/week02" for j in RM.enumerate_jobs("A1", None, False, (), kall))
    assert any(j["slice"] == "freeinference/week02" for j in RM.enumerate_jobs("B-A1", None, False, (), kall))


def test_runner_refuses_real_without_frozen_and_checks_doc(tmp_path):
    """沒給 --frozen 就拒跑真實資料；04 還含「未凍結」的 commit 被拒；H 和 04 §9 表對得上（≤0.1 s）。"""
    with pytest.raises(SystemExit, match="--frozen"):
        RM.main(["--stage", "A1", "--slices", "bailian/coder", "--out", str(tmp_path)])
    head = subprocess.run(["git", "-C", RM.REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if "未凍結" in RM.prereg_text(head):
        with pytest.raises(SystemExit, match="未凍結"):
            RM.main(["--stage", "A1", "--slices", "bailian/coder", "--frozen", head, "--out", str(tmp_path)])
    Hs, _ = RM.load_H()
    rows = RM.check_H_against_doc(Hs, RM.prereg_text(None))
    assert len(rows) == 8 and all(abs(a - b) <= 0.1 for _, a, b in rows)
    bad = RM.prereg_text(None).replace("3,488.7 | 3,488.7", "3,500.0 | 3,500.0", 1)
    with pytest.raises(AssertionError, match="差超過 0.1"):
        RM.check_H_against_doc(Hs, bad)


def test_frozen_check_accepts_frozen_text(monkeypatch):
    """04 在那個 commit 的版本沒有「未凍結」→ 通過，回傳 commit 全名與內容的 sha256；有 → 拒絕。"""
    real = RM.prereg_text
    monkeypatch.setattr(RM, "prereg_text", lambda frozen=None: "第三階段 地圖實驗 v4（已凍結）" if frozen else real(None))
    full, sha = RM.check_frozen("HEAD")
    assert len(full) == 40 and len(sha) == 64
    monkeypatch.setattr(RM, "prereg_text", lambda frozen=None: "草稿，未凍結")
    with pytest.raises(SystemExit, match="未凍結"):
        RM.check_frozen("HEAD")


def test_runner_resume_and_partial_cleanup(tmp_path):
    """續跑：json＋npz 都在的跳過；只有半個（npz 沒有 json、.tmp）的清掉重跑；FAILED 重跑。"""
    kall = {(s, m, v): dict(k=1, hourmax=0.1, overload=False) for s in RM.SLICES for m in RM.A1_MEM + RM.A2_MEM for v in ("idle", "busy")}
    jobs = RM.enumerate_jobs("A1", ["synth/small-seed1"], False, (), kall)[:6]
    out = str(tmp_path)
    todo, done, cl = RM.classify(out, jobs)
    assert len(todo) == 6 and not done
    for i, j in enumerate(jobs):
        open(j["npz_path"], "wb").write(b"x")
        if i < 3:
            json.dump(dict(job_id=j["id"], status="ok", rows=[]), open(j["json_path"], "w"))
        elif i == 3:
            json.dump(dict(job_id=j["id"], status="FAILED", rows=[]), open(j["json_path"], "w"))
        elif i == 4:
            open(j["json_path"] + ".tmp", "w").write("{")
    todo, done, cl = RM.classify(out, jobs)
    assert [j["id"] for j in done] == [j["id"] for j in jobs[:3]]
    assert [j["id"] for j in todo] == [j["id"] for j in jobs[3:]]
    assert os.path.exists(jobs[0]["npz_path"]) and not os.path.exists(jobs[5]["npz_path"]) and not os.path.exists(jobs[4]["json_path"] + ".tmp")
    # json 說 ok 但 npz 不見了 → 重跑
    os.remove(jobs[0]["npz_path"])
    todo, done, _ = RM.classify(out, jobs)
    assert jobs[0]["id"] in [j["id"] for j in todo]


def test_dry_run_counts_and_estimates(capsys):
    """--dry-run 印每階段的元件、工作數、格子數、〔估計〕CPU 小時（不跑任何模擬）；A1 的非 FreeInference 判定用的 session 工作數對得上。"""
    RM.main(["--stage", "A1", "--dry-run"])
    t = capsys.readouterr().out
    assert "階段 A1" in t and "a1_sess" in t and "a1_blk" in t and "〔估計〕" in t and "CPU 小時" in t


def test_sub_P_and_group_assignment():
    """E6 的子集：session 不被拆開、父請求在同一組、session 重新編號；分組用 sha1(session_id) mod k（和 stats_v4 的 k 一樣的規則）。"""
    import hashlib
    keys = [f"s{i}" for i in range(7)]
    P = dict(full=[[i] for i in range(10)], nall=[1] * 10, tl=[float(i) for i in range(10)], kindc=[0, 0, 1, 1, 0, 1, 0, 1, 1, 1],
             parentl=[-1, -1, 0, 1, -1, 2, -1, 3, 4, 5], sessl=[0, 1, 0, 1, 2, 0, 3, 1, 2, 0], obs_end=[100.0] * 10, win=(0.0, 100.0),
             meta=dict(n_req=10, n_sess=7, n_ret=6), R=dict(n_sess=7), nmax=4, B=10, sess_key=keys)
    g = RM.sess_group(keys, 3)
    assert list(g) == [int(hashlib.sha1(k.encode()).hexdigest(), 16) % 3 for k in keys]
    sl = np.asarray(P["sessl"])
    seen = 0
    for k in range(3):
        idx = np.nonzero(g[sl] == k)[0]
        if not len(idx):
            continue
        Pg = RM.sub_P(P, idx)
        seen += len(idx)
        assert len(Pg["full"]) == len(idx) and Pg["R"]["n_sess"] == len(set(sl[idx]))
        assert all(p < i for i, p in enumerate(Pg["parentl"]) if p >= 0) and max(Pg["sessl"]) == Pg["R"]["n_sess"] - 1
        assert [Pg["sess_key"][s] for s in Pg["sessl"]] == [keys[s] for s in sl[idx]]
    assert seen == 10


def test_pool_rows():
    """k 張 GPU 的列合併：位元組、次數相加；p99 取最大；沒有定義的時間類欄寫 NaN；軸欄不變。"""
    r1 = dict(ssd_write_GiB=1.0, n_req=3, reads=2, ssd_cw_write_p99_GiBps=0.5, ttft_mean_s=1.0, ssd_share=1, view="v", ssd_sat_any_frac=0.1,
              ssd_H_write_issued_GiB=2.0, surprise_events=1)
    r2 = dict(ssd_write_GiB=2.0, n_req=4, reads=3, ssd_cw_write_p99_GiBps=0.9, ttft_mean_s=2.0, ssd_share=1, view="v", ssd_sat_any_frac=0.3,
              ssd_H_write_issued_GiB=5.0, surprise_events=2)
    p = RM._pool_rows([r1, r2])
    assert p["ssd_write_GiB"] == 3.0 and p["n_req"] == 7 and p["reads"] == 5 and p["ssd_cw_write_p99_GiBps"] == 0.9
    assert p["ssd_H_write_issued_GiB"] == 7.0 and p["surprise_events"] == 3 and p["ssd_sat_any_frac"] == 0.3
    assert math.isnan(p["ttft_mean_s"]) and p["ssd_share"] == 1 and p["view"] == "v"


# ===================================================================== 端到端（合成切片；不是真實資料）
def test_end_to_end_synth_subset(tmp_path):
    """合成切片 synth/small-seed1：執行器跑幾個工作 → collect → 欄位與 npz 對得上（主指標＝寫入＋只載入的讀取；總和和 CSV 一致）→ analyze 的 npz 驗算通過。
    （完整的端到端：run3090 的 p3sim-mapsyn；這裡只跑 C0、C1、O 的 E2＠63 GiB，session 引擎與 block 引擎各一。）"""
    out = str(tmp_path / "map")
    sel = ["session__E2__m63__llama__C0__both", "session__E2__m63__llama__C1__both", "session__E2__m63__llama__O__H",
           "block__E6__m63__llama__C0__vidle__both", "block__E6__m63__llama__O__vidle__2H"]
    rc = RM.main(["--stage", "A1", "--slices", "synth/small-seed1", "--only", *sel, "--procs", "5", "--out", out, "--no-sens-views"])
    assert rc == 0
    d = pd.read_csv(os.path.join(out, "map_cells.csv"), low_memory=False)
    assert set(d.job_id) == set(sel) and d.synthetic.all() and (d.unverified_len_mode == "request").all() and (d.int2_on_read == "recompute_unverified").all()
    assert (d.reads_fail == 0).all() and (d.reads_unverified == 0).all()
    assert d[d.policy_label == "O"].surprise_events.max() == 0
    for _, r in d[d.is_primary].iterrows():
        h = "H" if r.h_tag in ("H", "both") else "2H"
        z = np.load(r.npz_path)
        sfx = r.npz_sfx_H if h == "H" else r.npz_sfx_2H
        tot = (z["ssd_write_bytes" + sfx].sum() + z["loadonly_ssd_read_bytes" + sfx].sum()) / GiB
        assert abs(tot - (r.prim_H_GiB if h == "H" else r.prim_2H_GiB)) < 1e-6 * max(1, tot), r.job_id
        assert abs(r.ssd_H_write_issued_GiB + r.ssd_H_loadonly_read_GiB - (r.prim_H_GiB if h == "H" else r.prim_2H_GiB)) < 1e-6 * max(1, tot)
        assert z["ttft_n"] == z["ttft_hist"].sum() and z["n_req"].sum() == r.n_req
    # 非 oracle 的 2H 版：和同一個放置的 H2 欄位一致（一次模擬兩個截尾）
    c1 = d[(d.job_id == "session__E2__m63__llama__C1__both") & d.is_primary].iloc[0]
    assert c1.prim_2H_GiB <= c1.prim_H_GiB + 1e-9 and c1.prim_2H_GiB > 0
    # 續跑：再跑一次什麼都不跑
    rc = RM.main(["--stage", "A1", "--slices", "synth/small-seed1", "--only", *sel, "--procs", "5", "--out", out, "--no-sens-views"])
    assert rc == 0
    with open(os.path.join(OUT, "test_e2e_subset.txt"), "a") as fh:
        fh.write(f"{os.environ.get('RUN_ID', 'adhoc')} rows={len(d)}\n")
