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
    bytes_fn(trace, slice, olabel, mem, label, tag) → 每個 session 的位元組（GiB 單位的 float 陣列或常數）；ttft_fn(...) → (mean, p99) 或 None＝NOT_MEASURED。
    T_s＝每個切片截尾後的窗口長度（秒；SSD 使用率用；頻寬 1 GiB/s 讀寫）；loose_fn(…)＝給了就另外造寬鬆版雙胞胎的列（rule=loose；主格、O 只有 H）。
    views＝另外造的非主視角的列（{view_name: bytes 倍率}）。"""

    def __init__(self, tmp, spec, bytes_fn, ttft_fn=None, mems=AM.MEMS, T_s=36000.0, loose_fn=None, views=None, rec_fn=None, ttft_o_fn=None,
                 k_fn=None, rc_fn=None, commits=("", "")):
        """k_fn(trace, slice, mem)→k_util（預設 1）；rc_fn(trace, slice, ol, mem, label, tag)→這個做法每個 session 被計價的 GiB（重算計價；預設 0）；
        commits＝(frozen_commit, code_commit) 欄。"""
        self.k_fn, self.rc_fn, self.commits = k_fn or (lambda *a: 1), rc_fn or (lambda *a: 0.0), commits
        self.tmp, self.spec, self.rows = str(tmp), spec, []
        self.bytes_fn, self.ttft_fn = bytes_fn, ttft_fn or (lambda *a: (1.0, 2.0))
        # 寬鬆版雙胞胎是完整性檢查的一部分（第三輪審查 #6）：沒給 loose_fn 就用和 strict 一樣的數字造一份
        self.T_s, self.loose_fn, self.views, self.rec_fn = T_s, loose_fn or bytes_fn, views or {}, rec_fn or (lambda *a: 7)
        self.n = 0
        self.build(mems)

    def _one(self, trace, sp, s, S, nreq, ol, mem, eng, label, tag, bfn, rule, view, vg="main", vidx=0, scale=1.0):
        oracle = label in AM.ORACLE_LABELS
        xh = np.broadcast_to(np.asarray(bfn(trace, s, ol, mem, label, "H"), float), (S,)) * GiB * scale
        x2 = np.broadcast_to(np.asarray(bfn(trace, s, ol, mem, label, "2H"), float), (S,)) * GiB * scale
        path = os.path.join(self.tmp, f"{self.n}.npz")
        self.n += 1
        ttft = self.ttft_fn(trace, s, ol, mem, label)
        d = dict(n_req=nreq, sess_orig=np.arange(S), ttft_n=np.asarray(100), ttft_sum=np.asarray(0.0), ttft_nm=np.asarray(0 if ttft is not None else 1),
                 ttft_hist=np.zeros(len(RM.TTFT_EDGES) - 1, np.int64), ttft_vals=np.zeros(100))
        m, p = ttft if ttft is not None else (0.0, 0.0)
        for pfx in ("tH", "tH2"):                       # 截尾版護欄的逐請求值（100 個都＝p；平均＝m）
            d.update({f"{pfx}_n": np.asarray(100), f"{pfx}_sum": np.asarray(m * 100.0), f"{pfx}_nm": np.asarray(0 if ttft is not None else 1),
                      f"{pfx}_vals": np.full(100, float(p))})
        if ttft is not None:
            d["ttft_sum"] = np.asarray(m * 100.0)
            d["ttft_vals"] = np.full(100, float(p))

        rcg = float(self.rc_fn(trace, s, ol, mem, label, tag))

        def put(sfx, x, wf):
            d["ssd_write_bytes" + sfx] = (x * wf).astype(np.int64)
            rd = (x * (1 - wf)).astype(np.int64)
            d["primread_bytes" + sfx] = rd[None, :]            # 主指標的讀取（只載入的還原＋全部預取）：兩個引擎同一個 npz 欄位（第三輪審查 #4）
            d["recompute_charge_bytes" + sfx] = np.full(S, int(rcg * GiB), np.int64)
            return float(d["ssd_write_bytes" + sfx].sum()) / GiB, float(rd.sum()) / GiB, rcg * S
        nan = float("nan")
        pw = dict(prim_w_H_GiB=nan, prim_r_H_GiB=nan, prim_w_2H_GiB=nan, prim_r_2H_GiB=nan, T_H_s=nan, T_2H_s=nan,
                  prim_rbase_H_GiB=nan, prim_rc_H_GiB=nan, prim_rbase_2H_GiB=nan, prim_rc_2H_GiB=nan)
        if tag == "both":
            w1, r1, c1 = put("_H", xh, 0.5)
            w2, r2, c2 = put("_H2", x2, 0.25)
            sfx, ph, p2 = ("_H", "_H2"), w1 + r1, w2 + r2
            pw.update(prim_w_H_GiB=w1, prim_r_H_GiB=r1, prim_w_2H_GiB=w2, prim_r_2H_GiB=r2, T_H_s=self.T_s, T_2H_s=self.T_s,
                      prim_rbase_H_GiB=r1, prim_rc_H_GiB=c1, prim_rbase_2H_GiB=r2, prim_rc_2H_GiB=c2)
        else:
            x = xh if tag == "H" else x2
            w1, r1, c1 = put("_H", x, 0.5)
            pv = w1 + r1
            if tag == "H":
                sfx, ph, p2 = ("_H", ""), pv, nan
                pw.update(prim_w_H_GiB=w1, prim_r_H_GiB=r1, T_H_s=self.T_s, prim_rbase_H_GiB=r1, prim_rc_H_GiB=c1)
            else:
                sfx, ph, p2 = ("", "_H"), nan, pv
                pw.update(prim_w_2H_GiB=w1, prim_r_2H_GiB=r1, T_2H_s=self.T_s, prim_rbase_2H_GiB=r1, prim_rc_2H_GiB=c1)
        np.savez(path, **d)
        tm, tp = (ttft if ttft else (AM.NOT, AM.NOT))
        self.rows.append(dict(run_id="fx", stage="A1", comp="a1_blk" if eng == "block" else "a1_sess", slice=s, trace_group=trace,
                              source_group=sp["source"], synthetic=sp.get("synthetic", False), semi_synthetic=False, overlay=1,
                              engine=eng, outer_code=ol.split("-")[0], T_s=1800 if ol.startswith("E5") else "", mem_GiB=mem,
                              model="llama", policy_label=label, policy_family=label, oracle=oracle, h_tag=tag, view_group=vg,
                              view_idx=vidx, is_primary=(view == AM.PRIMARY_VIEW), k_gpus=1, k_util=self.k_fn(trace, s, mem), recompute_charge="none",
                              frozen_commit=self.commits[0], code_commit=self.commits[1],
                              job_id=f"{s}|{ol}|{mem}|{label}|{tag}|{rule}|{view}", npz_path=path,
                              view_name=view, quality_rule=rule, npz_sfx_H=sfx[0], npz_sfx_2H=sfx[1], prim_H_GiB=ph, prim_2H_GiB=p2,
                              ttftq_mean_s=tm, ttftq_p99_s=tp, ttftq_H_mean_s=tm, ttftq_H_p99_s=tp, ttftq_H2_mean_s=tm, ttftq_H2_p99_s=tp,
                              n_ret=100, ssd_bw_r_GiBps=1.0, ssd_bw_w_GiBps=1.0, prim_read_col="x",
                              recompute_prefix_blocks=self.rec_fn(trace, s, ol, mem, label), ret_recompute_prefix_blocks=7, surprise_blocks=0, **pw))

    def build(self, mems):
        for trace, sp in self.spec.items():
            for s in sp["slices"]:
                S = sp["S"]
                nreq = np.asarray(sp.get("n_req") if sp.get("n_req") is not None else np.ones(S, np.int64))
                for (ol, mem) in [(o, m) for o, _ in AM.PRIMARY_OUTERS for m in mems]:
                    eng = AM.ENGINE_OF[ol]
                    for label in AM.LABELS:
                        oracle = label in AM.ORACLE_LABELS
                        for tag in (("H", "2H") if oracle else ("both",)):
                            self._one(trace, sp, s, S, nreq, ol, mem, eng, label, tag, self.bytes_fn, "strict", AM.PRIMARY_VIEW)
                            if self.loose_fn and label in AM.NEEDED and not (oracle and tag != "H"):
                                self._one(trace, sp, s, S, nreq, ol, mem, eng, label, "H" if oracle else "both", self.loose_fn, "loose", AM.PRIMARY_VIEW)
                    for vn, sc in self.views.items():        # 非主視角（例如 E6 的 GPU 忙）：只有點要報告 → C0 與 O 與幾個對手就夠，這裡全做
                        for label in AM.LABELS:
                            oracle = label in AM.ORACLE_LABELS
                            for tag in (("H", "2H") if oracle else ("both",)):
                                self._one(trace, sp, s, S, nreq, ol, mem, eng, label, tag, self.bytes_fn, "strict", vn, vidx=1 if eng == "session" else 0, scale=sc)
        self.csv = os.path.join(self.tmp, "map_cells.csv")
        pd.DataFrame(self.rows).to_csv(self.csv, index=False)

    def analyze(self, mode="full", boot=50, guard_ref="C0", expected="present", ovar="greedy", charge="none"):
        return AM.analyze(AM.load_cells(self.csv), mode, boot, verbose=False, guard_ref=guard_ref, expected=expected, oracle_variant=ovar, recompute_charge=charge)


BASE = {"C0": 100.0, "C1": 90.0, "C1-8": 95.0, "C2": 85.0, "C3": 92.0, "C6": 91.0, "C4-8192": 96.0, "C4-16384": 97.0, "C4-32768": 98.0,
        "C4r-8192": 93.0, "C4r-16384": 94.0, "C4r-32768": 99.0, "D-len-8192": 89.0, "D-len-16384": 88.0, "D-len-32768": 87.0, "D-idle-300": 86.0,
        "D-idle-1800": 84.0, "D-idle-3600": 83.0, "D-idle-14400": 82.0, "C1-2": 150.0, "R2": 140.0}        # 非 oracle：最好的是 D-idle-14400（82）


def mk_bytes(o_frac, cell=None, traces=None):
    """O 的位元組＝最好的對手 × (1 − o_frac)；只有 (cell, traces) 指定的格子用 o_frac，其他格子 O＝最好的對手（省 0）。"""
    def f(trace, s, ol, mem, label, tag):
        if label in AM.ORACLE_LABELS:
            on = (cell is None or (ol, mem) == cell) and (traces is None or trace in traces)
            fr = o_frac if on else 0.0
            return {"O": 82.0 * (1 - fr), "O-tier": 82.0 * (1 - fr), "O-len": 82.0 * (1 - fr) + 5, "O-reuse": 82.0 * (1 - fr) + 8, "O-loose": 82.0 * (1 - fr) - 3}[label]
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
            return 82.0 * (1 - fr) + {"O": 0, "O-tier": 0, "O-len": 5, "O-reuse": 8, "O-loose": -3}[label]
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
            return 82.0 * (1 - 0.049) + {"O": 0, "O-tier": 0, "O-len": 5, "O-reuse": 8, "O-loose": -3}[label] if (ol, mem) == ("E2", 63.0) else 82.0
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
    # 一份不完整的資料（缺一個主格）→ 不給判定（IncompleteMap；第二輪審查 #4）
    fx = Fx(_sub(tmp_path, "i"), S5(), mk_bytes(0.02))
    df = pd.read_csv(fx.csv, low_memory=False)
    df = df[~((df.trace_group == "Mooncake") & (df.outer_code == "E4") & (df.mem_GiB == 435.0))]
    df.to_csv(fx.csv, index=False)
    with pytest.raises(AM.IncompleteMap, match="缺"):
        fx.analyze()


def test_early_go_rule(tmp_path):
    """§10：不含 FreeInference 的資料已滿足 GO → 早期定案；只有 FreeInference 滿足 → PENDING_B（full 模式才 GO）。"""
    spec = S5(fi=True)
    # FreeInference 有 3 個有效 session 以上、要 ≥100：用 n_req 讓 eff≥100（每週 S 個 session 各 1 個請求 → 每週 S；兩週合 2S）
    spec["FreeInference"]["S"] = 120

    def f(trace, s, ol, mem, label, tag):
        if label in AM.ORACLE_LABELS and (ol, mem) == ("E3", 170.0):
            fr = {"Bailian coder": 0.30, "WildChat": 0.08}.get(trace, 0.0)
            return 82.0 * (1 - fr) + {"O": 0, "O-tier": 0, "O-len": 5, "O-reuse": 8, "O-loose": -3}[label]
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
            return 82.0 * (1 - fr) + {"O": 0, "O-tier": 0, "O-len": 5, "O-reuse": 8, "O-loose": -3}[label]
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
            return 82.0 * (1 - fr) + {"O": 0, "O-tier": 0, "O-len": 5, "O-reuse": 8, "O-loose": -3}[label]
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
    # best＝0 而 O>0：NaN＋旗標，不是 0%（第二輪審查 #10）；0/0 → 0
    p, bz, nf = AM.savings_point(0.0, 3.0)
    assert math.isnan(p) and bz and nf == "best_zero_o_pos"
    assert AM.savings_point(0.0, 0.0) == (0.0, True, "")
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
    with pytest.raises(AssertionError, match="不等於分量加總"):          # prim_H_GiB 和分量（寫入＋讀取＋重算計價）對不上
        fx.analyze()
    df.loc[i, "prim_H_GiB"] -= 1.0
    df.loc[i, "prim_w_H_GiB"] += 1.0          # 分量和 prim_H 一致地被改 → 擋不到分量與 npz 的差
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

    def tnpz(x, nm_=0):
        return {"tH_vals": x.astype(np.float64), "tH_n": np.asarray(len(x)), "tH_sum": np.asarray(float(x.sum())), "tH_nm": np.asarray(nm_)}
    for i, x in enumerate(weeks):
        pth = str(tmp_path / f"w{i}.npz")
        np.savez(pth, **RM.ttft_arrays(x, 0, keep_vals=True), **tnpz(x))
        rows.append(dict(npz_path=pth, oracle=False, view_idx=0, job_id=f"w{i}", ttftq_H_mean_s=float(x.mean()), ttftq_H_p99_s=float(np.percentile(x, 99))))
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
    with pytest.raises(SystemExit):                          # 沒有逐請求的值 → 拒絕（不悄悄退回直方圖）
        AM.trace_ttft([rows[0], dict(rows[0], npz_path=old, job_id="old")])
    nmp = str(tmp_path / "nm.npz")
    np.savez(nmp, **tnpz(weeks[0], 3))
    assert AM.trace_ttft([rows[0], dict(rows[0], npz_path=nmp, job_id="nm")])[2] is True
    # 非工作的第 0 個視角的列不能幾週合併（npz 只存第 0 個視角的 ttft）→ NoPooledTtft，不悄悄用錯的視角
    with pytest.raises(AM.NoPooledTtft):
        AM.trace_ttft([rows[0], dict(rows[1], view_idx=3)])
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
    """階段、元件、個數（04 §5、§10）；非 oracle 21 個、oracle 5 個（含 O-tier）× (H, 2H)；寬鬆版雙胞胎＝21 個對手＋O＋O-tier（只有 H）。"""
    kall = {(s, m, v): dict(k=1, hourmax=0.1, overload=False) for s in RM.SLICES for m in RM.A1_MEM + RM.A2_MEM for v in ("idle", "busy")}
    ids = set()
    for st, n_slices in (("A1", 8), ("A2", 8), ("B-A1", 12), ("B-A2", 12)):
        jobs = RM.enumerate_jobs(st, None, False, (), kall)
        assert len({j["slice"] for j in jobs}) == n_slices
        for j in jobs:
            assert (st, j["id"], j["slice"]) not in ids
            ids.add((st, j["id"], j["slice"]))
    a1 = RM.enumerate_jobs("A1", ["bailian/coder"], False, (), kall)
    assert sum(1 for j in a1 if j["comp"] == "a1_sess") == 4 * 3 * (21 + 5 * 2)           # 4 外層 × 3 記憶體 × (21＋5×2)（O、O-tier、O-len、O-reuse、O-loose）
    assert sum(1 for j in a1 if j["comp"] == "a1_blk") == 2 * 3 * (21 + 5 * 2)            # E6、E2 × 3 記憶體 × 31，主視角
    assert sum(1 for j in a1 if j["comp"] == "a1_loose") == 4 * 3 * (21 + 2)              # 寬鬆版雙胞胎：4 外層 × 3 記憶體 × (21 對手＋O、O-tier 的 H)
    assert sum(1 for j in a1 if j["comp"] == "a1_loose_blk") == 1 * 3 * (21 + 2)          # E6 only
    assert all(j["rule"] == "loose" and j["id"].endswith("__loose") for j in a1 if j["comp"].startswith("a1_loose"))
    assert len({j["id"] for j in a1}) == len(a1)
    a2 = RM.enumerate_jobs("A2", ["bailian/coder"], False, (), kall)
    assert sum(1 for j in a2 if j["comp"] == "a2_mem") == 7 * 5 * 31
    assert sum(1 for j in a2 if j["comp"] == "a2_blk") == 2 * 5 * 2 * 31                   # E6、E2 × 5 記憶體 × (閒、忙) × 31
    assert sum(1 for j in a2 if j["comp"] == "sens_qwen") == 4 * 3 * 31
    assert {j["overlay"] for j in a2 if j["comp"].startswith("sens_ov")} == {2, 4}
    assert len(RM.enumerate_jobs("A2", ["bailian/coder"], True, (), kall)) == len(a2) + 3 * 3 * 31
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
    sel = ["session__E2__m63__llama__C0__both", "session__E2__m63__llama__C1__both", "session__E2__m63__llama__O__H", "session__E2__m63__llama__O-tier__H",
           "block__E6__m63__llama__C0__vidle__both", "block__E6__m63__llama__O__vidle__2H"]
    rc = RM.main(["--stage", "A1", "--slices", "synth/small-seed1", "--only", *sel, "--procs", "5", "--out", out, "--no-sens-views", "--skip", "a1_loose", "a1_loose_blk"])
    assert rc == 0
    d = pd.read_csv(os.path.join(out, "map_cells.csv"), low_memory=False)
    assert set(d.job_id) == set(sel) and d.synthetic.all() and (d.unverified_len_mode == "request").all() and (d.int2_on_read == "recompute_unverified").all()
    assert (d.reads_fail == 0).all() and (d.reads_unverified == 0).all()
    assert d[d.policy_label == "O"].surprise_events.max() == 0 and d[d.policy_label == "O-tier"].surprise_events.max() == 0
    ot = d[d.policy_label == "O-tier"].iloc[0]          # O-tier：兩趟；session 引擎 CPU 側事件完全相同
    assert ot.otier_cpu_event_mismatch_requests == 0 and ot.otier_approx is False or str(ot.otier_approx) == "False"
    assert ot.otier_pass1_ssd_writes == ot.otier_pass2_ssd_writes > 0 and ot.prim_H_GiB <= d[(d.policy_label == "O") & (d.engine == "session")].prim_H_GiB.iloc[0] + 1e-9
    assert d.frozen_commit.isna().all() or (d.frozen_commit.fillna("") == "").all()    # 合成：沒有凍結 commit
    for _, r in d[d.is_primary].iterrows():
        h = "H" if r.h_tag in ("H", "both") else "2H"
        z = np.load(r.npz_path)
        sfx = r.npz_sfx_H if h == "H" else r.npz_sfx_2H
        blk = r.engine == "block"       # 兩個引擎同一個定義（第三輪審查 #4）：只載入的還原讀取＋全部預取（含浪費）；recompute_charge＝none 時不含重算計價
        rd = z["primread_bytes" + sfx][0].sum()
        tot = (z["ssd_write_bytes" + sfx].sum() + rd) / GiB
        assert abs(tot - (r.prim_H_GiB if h == "H" else r.prim_2H_GiB)) < 1e-6 * max(1, tot), r.job_id
        assert r.prim_read_col == "ssd_H_primread_GiB" and r.recompute_charge == "none" and r.k_util >= 1
        assert abs(r.ssd_H_write_issued_GiB + r.ssd_H_primread_GiB - (r.prim_H_GiB if h == "H" else r.prim_2H_GiB)) < 1e-6 * max(1, tot) or h != "H"
        assert abs(z["recompute_charge_bytes" + sfx].sum() / GiB - (r.prim_rc_H_GiB if h == "H" else r.prim_rc_2H_GiB)) < 1e-6 * max(1, tot)
        if blk:     # E6：主指標讀取 ≥ Cake 實際讀（load-only 讀全部，Cake 會自己重算一部分）
            assert r.ssd_H_primread_GiB >= r.ssd_H_actual_read_GiB - 1e-9
        assert r.view_name == AM.PRIMARY_VIEW and r.quality_rule == "strict" and r.cpu_busy_run_id == RM.CPU_BUSY_RUN
        assert z["ttft_n"] == z["ttft_hist"].sum() and z["n_req"].sum() == r.n_req
    # 非 oracle 的 2H 版：和同一個放置的 H2 欄位一致（一次模擬兩個截尾）
    c1 = d[(d.job_id == "session__E2__m63__llama__C1__both") & d.is_primary].iloc[0]
    assert c1.prim_2H_GiB <= c1.prim_H_GiB + 1e-9 and c1.prim_2H_GiB > 0
    # 續跑：再跑一次什麼都不跑
    rc = RM.main(["--stage", "A1", "--slices", "synth/small-seed1", "--only", *sel, "--procs", "5", "--out", out, "--no-sens-views",
                  "--skip", "a1_loose", "a1_loose_blk"])
    assert rc == 0
    # 寬鬆版雙胞胎的工作：另一個 id、rule=loose；C0 的位元組和 strict 一樣（BF16 任何規則都過關）
    out2 = str(tmp_path / "map2")
    loose = ["session__E2__m63__llama__C0__both__loose", "session__E2__m63__llama__O__H__loose"]
    assert RM.main(["--stage", "A1", "--slices", "synth/small-seed1", "--only", *loose, "--procs", "2", "--out", out2, "--skip", "a1_sess", "a1_blk",
                    "a1_loose_blk"]) == 0
    dl = pd.read_csv(os.path.join(out2, "map_cells.csv"), low_memory=False)
    assert set(dl.job_id) == set(loose) and (dl.quality_rule == "loose").all() and len(dl) == 2 and dl.view_idx.eq(0).all()
    assert abs(dl[dl.policy_label == "C0"].prim_H_GiB.iloc[0] - d[(d.policy_label == "C0") & (d.engine == "session")].prim_H_GiB.iloc[0]) < 1e-9
    with open(os.path.join(OUT, "test_e2e_subset.txt"), "a") as fh:
        fh.write(f"{os.environ.get('RUN_ID', 'adhoc')} rows={len(d)}\n")


# ===================================================================== 第二輪審查之後（06 §25）
def _row(r, trace, ol, mem, h="H"):
    return next(x for x in r["rows"] if x["trace"] == trace and x["outer"] == ol and x["mem_GiB"] == mem and x["h"] == h)


def test_guard_ref_two_choices(tmp_path):
    """護欄的參考（使用者決定中；兩種都實作）：O 的等待時間比 C0 快 20%。ref＝C0 時 D-idle-14400（0.9×C0 慢…＝0.9 ≤ 1.05）合格、最好的對手 82；
    ref＝min(C0,O)（O 的 0.8）時門檻變 0.84：D-idle-14400（0.9）出局，D-idle-3600（0.83）留下 → 83。兩種都在報告裡；取捨表有兩欄合格與否。"""
    def tt(trace, s, ol, mem, label):
        return {"O": (0.8, 1.6), "D-idle-14400": (0.9, 1.8), "D-idle-3600": (0.83, 1.65)}.get(label, (1.0, 2.0))
    fx = Fx(_sub(tmp_path, "gr"), S5(), mk_bytes(0.30), tt)
    r0 = fx.analyze(guard_ref="C0")
    r1 = fx.analyze(guard_ref="min_C0_O")
    a, b = _row(r0, "Mooncake", "E2", 63.0), _row(r1, "Mooncake", "E2", 63.0)
    assert a["best_label"] == "D-idle-14400" and a["guard_ref"] == "C0" and abs(a["best_GiB"] - 82.0 * 120) < 1e-6
    assert b["best_label"] == "D-idle-3600" and b["guard_ref"] == "min_C0_O" and abs(b["best_GiB"] - 83.0 * 120) < 1e-6
    assert "D-idle-14400" in b["guard_excluded"].split(";") and "C0" in b["guard_excluded"].split(";")
    for r in (r0, r1):       # 兩種參考都在報告（和判定用哪一種無關）
        gr = [x for x in r["reports"] if x["kind"] == "report_guard_ref" and x["trace"] == "Mooncake" and x["outer"] == "E2@session" and x["mem_GiB"] == 63.0]
        assert {x["guard_ref"] for x in gr} == {"C0", "min_C0_O"}
        assert next(x for x in gr if x["guard_ref"] == "C0")["best_label"] == "D-idle-14400"
        assert next(x for x in gr if x["guard_ref"] == "min_C0_O")["best_label"] == "D-idle-3600"
    t = [x for x in r0["tradeoff"] if x["trace"] == "Mooncake" and x["outer"] == "E2" and x["mem_GiB"] == 63.0]
    assert {x["policy"] for x in t} == set(AM.NEEDED)
    d14 = next(x for x in t if x["policy"] == "D-idle-14400")
    assert d14["eligible_ref_C0"] == "通過" and d14["eligible_ref_min_C0_O"] == "不合格" and abs(d14["primary_GiB"] - 82.0 * 120) < 1e-6
    assert d14["recompute_tokens"] == 7 * 512 and d14["ttftq_H_mean_s"] == 0.9 and d14["ttftq_H_p99_s"] == 1.8
    assert next(x for x in t if x["policy"] == "O")["eligible_ref_min_C0_O"] == "通過"
    assert next(x for x in t if x["policy"] == "C0")["eligible_ref_min_C0_O"] == "不合格"      # C0 自己也慢過 1.05×0.8


def test_o_must_pass_guard_and_util_gate_and_best_is_c0_flag(tmp_path):
    """(a) O 的等待時間 1.2×C0 → 這個格子不能支持 GO（flag o_guard_ok=False），即使省 30%；(b) 最好的對手的 SSD 使用率 <5%（T 很長）→ 不能支持 GO；
    (c) 最好的對手是 C0 時，GO 的格子標 best_is_C0。"""
    def f(trace, s, ol, mem, label, tag):
        if label in AM.ORACLE_LABELS and (ol, mem) == ("E2", 63.0):
            fr = {"Bailian coder": 0.30, "WildChat": 0.10}.get(trace, 0.0)
            return 82.0 * (1 - fr) + {"O": 0, "O-tier": 0, "O-len": 5, "O-reuse": 8, "O-loose": -3}[label]
        return mk_bytes(0.0)(trace, s, ol, mem, label, tag)
    ok = Fx(_sub(tmp_path, "ok"), S5(), f).analyze()
    assert ok["verdict"]["verdict"] == "GO"
    r = _row(ok, "Bailian coder", "E2", 63.0)
    assert r["o_guard_ok"] and r["go_ok"] and not r["best_is_C0"] and abs(r["best_util"] - (82.0 * 120) / 36000.0) < 1e-9
    assert abs(r["c0_util"] - (100.0 * 120) / 36000.0) < 1e-9 and r["ssd_busy"] and r["evaluable"]
    assert abs(r["best_GiB_per_h"] - 82.0 * 120 / 10.0) < 1e-9 and abs(r["O_GiB_per_h"] - 82.0 * 0.7 * 120 / 10.0) < 1e-6
    assert r["best_recompute_tokens"] == 7 * 512 and r["O_recompute_tokens"] == 7 * 512
    assert "最好=C0" not in "".join(AM._flags_of_row(r))
    # (a) O 慢
    bad = Fx(_sub(tmp_path, "ob"), S5(), f, lambda t, s, ol, mem, lab: (1.2, 2.4) if lab == "O" else (1.0, 2.0)).analyze()
    r = _row(bad, "Bailian coder", "E2", 63.0)
    assert not r["o_guard_ok"] and not r["go_ok"] and "O 沒過護欄" in AM._flags_of_row(r)
    assert bad["verdict"]["verdict"] != "GO", bad["verdict"]
    # (b) T 很長 → C0 的使用率 <5% → 「SSD 不忙」：這格 GO、NO-GO 都不算（第三輪審查 #5）；最好的對手的使用率照舊報告
    low = Fx(_sub(tmp_path, "ul"), S5(), f, T_s=36000.0 * 50).analyze()
    r = _row(low, "Bailian coder", "E2", 63.0)
    assert r["c0_util"] < 0.05 and not r["ssd_busy"] and not r["evaluable"] and not r["go_ok"] and any("SSD 不忙" in x for x in AM._flags_of_row(r))
    assert r["best_util"] < 0.05
    assert low["verdict"]["verdict"] != "GO"
    assert low["verdict"]["verdict"] == "ASK" and "沒有任何能判定的格子" in low["verdict"]["reason"]

    # (c) 最好的對手是 C0（所有非 oracle 一樣大，C0 排第一）
    def g(trace, s, ol, mem, label, tag):
        if label in AM.ORACLE_LABELS:
            fr = {"Bailian coder": 0.30, "WildChat": 0.10}.get(trace, 0.0) if (ol, mem) == ("E2", 63.0) else 0.0
            return 100.0 * (1 - fr)
        return 100.0
    c0 = Fx(_sub(tmp_path, "c0"), S5(), g).analyze()
    assert c0["verdict"]["verdict"] == "GO" and c0["verdict"]["go_cells"][0]["strong_best_is_C0"] and c0["verdict"]["go_cells"][0]["weak_best_is_C0"]
    assert "最好的對手是 C0？ Bailian coder：是" in c0["verdict"]["reason"]


def test_best_zero_o_positive_is_nan_and_blocks_go(tmp_path):
    """最好的對手＝0、O>0：省多少＝NaN＋旗標（不是 0%）；這個格子沒有 CI、不能支持 GO、也不阻擋別的格子。"""
    def f(trace, s, ol, mem, label, tag):
        if (ol, mem) == ("E4", 435.0):         # C0 照常（SSD 夠忙），其他對手 0、O 有 3
            return 3.0 if label in AM.ORACLE_LABELS else (BASE["C0"] if label == "C0" else 0.0)
        return mk_bytes(0.02)(trace, s, ol, mem, label, tag)
    r = Fx(_sub(tmp_path, "bz"), S5(), f).analyze()
    z = _row(r, "Mooncake", "E4", 435.0)
    assert math.isnan(z["savings"]) and z["nan_flag"] == "best_zero_o_pos" and math.isnan(z["ci_lo"]) and not z["go_ok"]
    assert r["verdict"]["verdict"] == "ASK", r["verdict"]           # 一個 NaN 的格子 → 不能 NO-GO（也不是 GO）


def test_primary_read_is_primread_for_both_engines(tmp_path):
    """兩個引擎的主指標讀取同一個定義（第三輪審查 #4）＝npz 的 primread_bytes_H[0]（只載入的還原讀取＋全部預取）；用錯欄 → 驗算和 CSV 對不上。"""
    fx = Fx(_sub(tmp_path, "bk"), S5(), mk_bytes(0.1))
    ix = AM.Index(AM.load_cells(fx.csv))
    rb = ix.row("Mooncake", "m/c", "block", "E6", 63.0, "C0", "both")
    rs = ix.row("Mooncake", "m/c", "session", "E2", 63.0, "C0", "both")
    z = np.load(rb["npz_path"])
    assert z["primread_bytes_H"].shape == (1, 120)
    _, xb = AM.sess_bytes(rb, "H")
    assert abs(xb.sum() / GiB - 100.0 * 120) < 1e-6          # 寫 50% ＋ 讀 50%
    _, xs = AM.sess_bytes(rs, "H")
    assert abs(xs.sum() / GiB - 100.0 * 120) < 1e-6
    np.savez(rb["npz_path"], **{k: (np.zeros_like(z[k]) if k == "primread_bytes_H" else z[k]) for k in z.files})
    with pytest.raises(AssertionError, match="不符"):
        AM.sess_bytes(rb, "H")


def test_loose_twin_and_views_reported_only(tmp_path):
    """寬鬆版雙胞胎：O 與所有對手用寬鬆規則的位元組（loose_fn）；只報告，不改判定。非主視角（E6 的 GPU 忙）的列出現在報告（kind＝view）。"""
    def lf(trace, s, ol, mem, label, tag):       # 寬鬆規則下 O 省 50%（對手 82）
        return 41.0 if label in ("O", "O-tier") else BASE[label]
    busy = "mi300x|cake|gpu=busy|cpu=idle|ssd/1|mi300x_local"
    r = Fx(_sub(tmp_path, "lv"), S5(), mk_bytes(0.0), loose_fn=lf, views={busy: 1.0}).analyze()
    lt = [x for x in r["reports"] if x["kind"] == "report_loose_twin"]
    assert len(lt) == 4 * 15 and all(abs(x["savings"] - 0.5) < 1e-9 and x["best_label"] == "D-idle-14400" for x in lt)
    assert r["verdict"]["verdict"] != "GO"
    vw = [x for x in r["reports"] if x["kind"] == "view" and x["h"] == "H"]
    assert any(x["outer"] == "E6@block" and x["view"] == busy for x in vw), {(x["outer"], x["view"]) for x in vw}
    pref = str(tmp_path / "lv" / "out")
    AM.write_outputs(r, pref, "full", "x.csv")
    md = open(pref + "_map.md", encoding="utf-8").read()
    assert "寬鬆版雙胞胎" in md and "其他視角" in md and "護欄的參考" in md
    assert os.path.exists(pref + "_tradeoff.csv")
    td = pd.read_csv(pref + "_tradeoff.csv")
    assert {"run_id", "ts", "primary_GiB", "ttftq_H_mean_s", "ttftq_H_p99_s", "recompute_tokens", "eligible_ref_C0", "eligible_ref_min_C0_O"} <= set(td.columns)
    js = json.load(open(pref + "_verdict.json", encoding="utf-8"))
    assert js["guard_ref"] == "C0" and len(js["loose_twin"]) == 60 and len(js["guard_ref_report"]) > 0


def test_completeness_hardcoded_slices(tmp_path):
    """每種模式寫死應有的切片：fixture 的切片名（a/coder…）不是真的 → 丟 IncompleteMap；缺 FreeInference 的週、只有 A1 就跑 full → 丟；合成模式要 synth/small-seed1。"""
    fx = Fx(_sub(tmp_path, "cs"), S5(fi=True), mk_bytes(0.1))
    for mode in ("early", "full"):
        with pytest.raises(AM.IncompleteMap, match="缺"):
            fx.analyze(mode, expected=None)
    assert AM.EXPECTED["early"]["FreeInference"] == [f"freeinference/week{w:02d}" for w in (1, 5, 9, 13)]
    assert len(AM.EXPECTED["full"]["FreeInference"]) == 16 and set(AM.EXPECTED["full"]) == set(AM.TRACES)
    assert AM.EXPECTED["early"]["Mooncake"] == ["mooncake/conv-rule-strict"] and AM.EXPECTED["early"]["Bailian coder"] == ["bailian/coder"]
    # 只有 A1 的 8 個切片＝early OK、full 缺 12 週
    spec = {"Bailian coder": dict(slices=["bailian/coder"], S=120, source="alibaba"), "Bailian traceA": dict(slices=["bailian/traceA"], S=120, source="alibaba"),
            "WildChat": dict(slices=["wildchat/1m"], S=120, source="wildchat"), "Mooncake": dict(slices=["mooncake/conv-rule-strict"], S=120, source="mooncake"),
            "FreeInference": dict(slices=AM.FI_A, S=120, source="freeinference")}
    a1 = Fx(_sub(tmp_path, "a1"), spec, mk_bytes(0.1))
    assert a1.analyze("early", expected=None)["verdict"]["verdict"]
    with pytest.raises(AM.IncompleteMap, match="缺切片"):
        a1.analyze("full", expected=None)
    # 多一個切片（B 的週混進 early）→ 也丟
    spec2 = dict(spec, FreeInference=dict(slices=AM.FI_A + ["freeinference/week02"], S=120, source="freeinference"))
    with pytest.raises(AM.IncompleteMap, match="多出"):
        Fx(_sub(tmp_path, "a2"), spec2, mk_bytes(0.1)).analyze("early", expected=None)
    # 合成模式
    syn = {"synth": dict(slices=["synth/small-seed1"], S=120, source="synthetic", synthetic=True)}
    sx = Fx(_sub(tmp_path, "sy"), syn, mk_bytes(0.1))
    assert sx.analyze("full", expected=AM.EXPECTED["synthetic"])["traces"]["synth"]["synthetic"]
    with pytest.raises(AM.IncompleteMap):
        sx.analyze("full", expected=None)
    # 缺一個做法的格子（刪掉一列）→ 丟
    df = pd.read_csv(sx.csv, low_memory=False)
    df[~((df.policy_label == "R2") & (df.outer_code == "E3") & (df.mem_GiB == 170.0))].to_csv(sx.csv, index=False)
    with pytest.raises(AM.IncompleteMap, match="缺 .* 個格子"):
        sx.analyze("full", expected=AM.EXPECTED["synthetic"])


def test_resolve_guard_ref():
    """護欄的參考＝04 §9 的 `護欄的參考：…`；沒有那一行 → 預設 C0 並寫明（未凍結）；命令列可以指定（來源寫明不是 04）。"""
    assert AM.resolve_guard_ref("auto", prereg_text="x\n護欄的參考：C0\n")[0] == "C0"
    assert AM.resolve_guard_ref("auto", prereg_text="x\n護欄的參考：min(C0, O)\n")[0] == "min_C0_O"
    ref, why = AM.resolve_guard_ref("auto", prereg_text="沒有")
    assert ref == "C0" and "未凍結" in why
    ref, why = AM.resolve_guard_ref("min_C0_O")
    assert ref == "min_C0_O" and "命令列" in why
    assert AM.resolve_guard_ref("auto")[0] in AM.GUARD_REFS          # 目前工作目錄的 04（凍結前沒有那一行 → C0）


def test_cpu_busy_factors_from_csv(tmp_path):
    """CPU 忙的係數從 cpubusy_slowdown.csv 讀（N=8 same → 1.35、N=48 other → 1.81），帶 run_id；四捨五入不等於登記值就停。不是抄進程式。"""
    lo, hi, rid = RM.load_cpu_busy()
    assert (lo, hi) == (1.35, 1.81) and rid.endswith("p3cal-cpubusy-qwen")
    assert RM.SENS_SPECS[0][2] == lo and RM.SENS_SPECS[1][2] == hi and RM.SENS_GROUPS[:2] == ["sens_cpu1.35", "sens_cpu1.81"]
    bad = str(tmp_path / "c.csv")
    txt = open(RM.CPUBUSY_CSV, encoding="utf-8").read().replace("1.3502", "1.2502")
    open(bad, "w", encoding="utf-8").write(txt)
    with pytest.raises(AssertionError, match="1.35"):
        RM.load_cpu_busy(bad)


def _mk_repo(tmp, inputs, frozen_guard="C0", extra="完美預測：greedy\n重算計價：none\n"):
    """暫時的 git repo：code/p3_sim/x.py、輸入檔、04（含鎖定表）。回傳 (repo, 凍結 commit)。"""
    import lockfile as LK
    repo = str(tmp / "repo")
    os.makedirs(os.path.join(repo, "code", "p3_sim"))
    os.makedirs(os.path.join(repo, os.path.dirname(LK.PREREG)))
    run = lambda *a: subprocess.run(["git", "-C", repo, *a], check=True, capture_output=True, text=True).stdout.strip()   # noqa: E731
    run("init", "-q")
    run("config", "user.email", "t@t")
    run("config", "user.name", "t")
    open(os.path.join(repo, "code", "p3_sim", "x.py"), "w").write("x = 1\n")
    for p, c in inputs.items():
        os.makedirs(os.path.dirname(os.path.join(repo, p)), exist_ok=True)
        open(os.path.join(repo, p), "w").write(c)
    run("add", "-A")
    run("commit", "-qm", "code")
    code = run("rev-parse", "HEAD")
    lock = "\n".join(f"| `{p}` | `{LK.sha256_file(os.path.join(repo, p))}` |" for p in inputs)
    doc = f"# 04\n\n<!-- LOCK:BEGIN -->\n| 路徑 | sha256 |\n|:--|:--|\n{lock}\n<!-- LOCK:END -->\n程式 commit（凍結）：`{code}`\n護欄的參考：{frozen_guard}\n{extra}"
    open(os.path.join(repo, LK.PREREG), "w", encoding="utf-8").write(doc)
    run("add", "-A")
    run("commit", "-qm", "freeze")
    return repo, run("rev-parse", "HEAD"), run


def test_freeze_gate_lockfile(tmp_path):
    """凍結閘門（第二輪審查 #5）：sha256 全對＋程式 commit＝凍結的／只有文件的後代＋乾淨 → 通過；輸入檔改了、程式在凍結後改了、工作目錄髒、缺必要輸入、「未凍結」→ 拒絕。"""
    import lockfile as LK
    req = ["a/in1.json", "b/in2.csv"]
    inputs = {"a/in1.json": "{}\n", "b/in2.csv": "x,y\n1,2\n"}
    repo, fz, run = _mk_repo(tmp_path, inputs, "min(C0, O)")
    info = LK.verify_freeze(fz, repo, required=req)
    assert info["guard_ref"] == "min_C0_O" and info["frozen"] == fz and info["files"].keys() == set(req) and len(info["prereg_sha256"]) == 64
    assert info["oracle_variant"] == "greedy" and info["recompute_charge"] == "none"
    assert LK.parse_params("護欄的參考：C0\n完美預測：tier\n重算計價：pass_size\n") == dict(guard_ref="C0", oracle_variant="tier", recompute_charge="pass_size")
    assert LK.parse_oracle_variant("完美預測：greedy 或 tier\n") is None and LK.parse_recompute_charge("  重算計價：none\n") is None      # 必須整行、一個值
    with pytest.raises(SystemExit, match="缺判定參數"):
        LK.require_params("護欄的參考：C0\n")
    # 04 缺判定參數的行 → 拒絕
    repo2, fz2, _ = _mk_repo(tmp_path / "np", inputs, "C0", extra="")
    with pytest.raises(SystemExit, match="缺判定參數的行"):
        LK.verify_freeze(fz2, repo2, required=req)
    # unified_inputs：20 個切片（階段 A 與 B）的 turns.parquet 絕對路徑
    ui = LK.unified_inputs()
    assert len(ui) == 20 and all(os.path.isabs(x) and x.endswith("turns.parquet") for x in ui) and len(set(ui)) == 20
    assert LK.parse_guard_ref("護欄的參考：C0\n") == "C0" and LK.parse_guard_ref("沒有") is None
    # 只有文件的後代 commit → 通過
    open(os.path.join(repo, "notes.md"), "w").write("hi\n")
    run("add", "-A")
    run("commit", "-qm", "docs only")
    assert LK.verify_freeze(fz, repo, required=req)["head"] != fz
    # 缺必要輸入
    with pytest.raises(SystemExit, match="缺這些必要的輸入檔"):
        LK.verify_freeze(fz, repo, required=req + ["c/missing.json"])
    # 輸入檔被改 → sha256 不符（也是髒的）
    open(os.path.join(repo, "b/in2.csv"), "w").write("x,y\n1,3\n")
    with pytest.raises(SystemExit, match="sha256 不符") as e:
        LK.verify_freeze(fz, repo, required=req)
    assert "不乾淨" in str(e.value)
    run("checkout", "--", "b/in2.csv")
    # 髒的工作目錄（已追蹤檔案）
    open(os.path.join(repo, "notes.md"), "a").write("more\n")
    with pytest.raises(SystemExit, match="不乾淨"):
        LK.verify_freeze(fz, repo, required=req)
    run("checkout", "--", "notes.md")
    # code/p3_sim 底下未追蹤的檔
    open(os.path.join(repo, "code", "p3_sim", "new.py"), "w").write("y = 2\n")
    with pytest.raises(SystemExit, match="未追蹤"):
        LK.verify_freeze(fz, repo, required=req)
    os.remove(os.path.join(repo, "code", "p3_sim", "new.py"))
    assert LK.verify_freeze(fz, repo, required=req)
    # 凍結之後程式被改（已 commit）→ 拒絕
    open(os.path.join(repo, "code", "p3_sim", "x.py"), "w").write("x = 2\n")
    run("add", "-A")
    run("commit", "-qm", "code changed after freeze")
    with pytest.raises(SystemExit, match="有修改"):
        LK.verify_freeze(fz, repo, required=req)
    # 04 還含「未凍結」→ 拒絕
    with pytest.raises(SystemExit, match="未凍結"):
        LK.verify_freeze(fz, repo, required=req, text="草稿，未凍結\n")
    # 沒有鎖定表
    with pytest.raises(SystemExit, match="沒有 `<!-- LOCK:BEGIN"):
        LK.verify_freeze(fz, repo, required=req, text="已凍結但沒有表\n")


def test_real_data_requires_explicit_run_id(tmp_path, monkeypatch):
    """真實資料：run_id 不給（或 adhoc）就拒絕；合成資料不需要。（凍結閘門本身用假的 verify_freeze 略過，另有 test_freeze_gate_lockfile。）"""
    import lockfile as LK
    monkeypatch.setattr(LK, "verify_freeze", lambda *a, **k: dict(frozen="x" * 40, guard_ref="C0", oracle_variant="greedy", recompute_charge="none"))
    monkeypatch.setattr(RM, "check_frozen", lambda c: ("f" * 40, "s" * 64))
    monkeypatch.delenv("RUN_ID", raising=False)
    with pytest.raises(SystemExit, match="run_id"):
        RM.main(["--stage", "A1", "--slices", "bailian/coder", "--frozen", "HEAD", "--out", str(tmp_path / "o1"), "--limit", "1"])
    monkeypatch.setenv("RUN_ID", "adhoc")
    with pytest.raises(SystemExit, match="run_id"):
        RM.main(["--stage", "A1", "--slices", "bailian/coder", "--frozen", "HEAD", "--out", str(tmp_path / "o2"), "--limit", "1"])
    monkeypatch.setenv("RUN_ID", "20261018-000000-test")
    with pytest.raises(SystemExit, match="official"):
        RM.main(["--stage", "A1", "--slices", "bailian/coder", "--frozen", "HEAD", "--out", str(tmp_path / "o3"), "--fmt", "placeholder", "--limit", "1"])


# ===================================================================== 第三輪審查之後（06 §26）
def test_oracle_variant_selects_role_and_reports_other(tmp_path):
    """完美預測：greedy｜tier（04 §9）：判定用的「O」換成 O-tier（比 O 省 10%）；另一個照報告（report_oracle_alt）。省多少、bootstrap、GO 都用判定用的那個。"""
    def f(trace, s, ol, mem, label, tag):
        if label in AM.ORACLE_LABELS and (ol, mem) == ("E2", 63.0):
            fr = {"Bailian coder": 0.30, "WildChat": 0.10}.get(trace, 0.0)
            base = 82.0 * (1 - fr)
            return {"O": base, "O-tier": base * 0.9, "O-len": base + 5, "O-reuse": base + 8, "O-loose": base - 3}[label]
        return mk_bytes(0.0)(trace, s, ol, mem, label, tag)
    fx = Fx(_sub(tmp_path, "ov"), S5(), f)
    g, t = fx.analyze(ovar="greedy"), fx.analyze(ovar="tier")
    rg, rt = _row(g, "Bailian coder", "E2", 63.0), _row(t, "Bailian coder", "E2", 63.0)
    assert rg["oracle_label"] == "O" and rt["oracle_label"] == "O-tier" and rt["O_alt_label"] == "O"
    assert abs(rg["savings"] - 0.30) < 1e-9 and abs(rt["savings"] - (1 - 0.7 * 0.9)) < 1e-9          # tier：O-tier 的位元組
    assert abs(rt["savings_O_alt"] - 0.30) < 1e-9 and abs(rg["savings_O_alt"] - (1 - 0.7 * 0.9)) < 1e-9
    assert g["verdict"]["verdict"] == t["verdict"]["verdict"] == "GO"
    alt = [x for x in t["reports"] if x["kind"] == "report_oracle_alt" and x["trace"] == "Bailian coder" and x["outer"] == "E2@session" and x["mem_GiB"] == 63.0]
    assert len(alt) == 1 and alt[0]["item"] == "O" and abs(alt[0]["savings"] - 0.30) < 1e-9
    assert {x["policy"] for x in t["tradeoff"] if x["is_O"]} == {"O-tier"} and {x["policy"] for x in t["tradeoff"] if x["is_O_alt"]} == {"O"}
    # 資料裡缺 O-tier → 不齊（兩個 O 都要）
    df = pd.read_csv(fx.csv, low_memory=False)
    df[df.policy_label != "O-tier"].to_csv(fx.csv, index=False)
    with pytest.raises(AM.IncompleteMap, match="缺"):
        fx.analyze(ovar="tier")


def test_recompute_charge_in_primary_metric(tmp_path):
    """重算計價（04 §9 `重算計價：`）：none＝不含；pass_size＝每個對手的主指標加上它的計價 GiB（分量欄 prim_rc_*；npz 歸屬到 session），O 沒有。
    C1-2（SSD 放 INT2 回來就重算）原本最省（位元組 60）；計價 100 之後變 160 → 不再是最好的對手 → 最好的對手換成 D-idle-14400（82）→ 省多少不同。bootstrap 的點估計和 CSV 一致。"""
    def f(trace, s, ol, mem, label, tag):
        if label == "C1-2":
            return 60.0
        return mk_bytes(0.30)(trace, s, ol, mem, label, tag)
    rcf = lambda trace, s, ol, mem, label, tag: 100.0 if label == "C1-2" else 0.0   # noqa: E731
    fx = Fx(_sub(tmp_path, "rc"), S5(), f, rc_fn=rcf)
    a, b = fx.analyze(charge="none"), fx.analyze(charge="pass_size")
    ra, rb = _row(a, "Mooncake", "E2", 63.0), _row(b, "Mooncake", "E2", 63.0)
    assert ra["best_label"] == "C1-2" and abs(ra["best_GiB"] - 60.0 * 120) < 1e-6 and ra["best_charged_GiB"] == 100.0 * 120
    assert rb["best_label"] == "D-idle-14400" and abs(rb["best_GiB"] - 82.0 * 120) < 1e-6 and rb["recompute_charge"] == "pass_size" and rb["O_charged_GiB"] == 0.0
    assert abs(ra["savings"] - (1 - 0.7 * 82 / 60)) < 1e-9 and abs(rb["savings"] - 0.30) < 1e-9
    tr = [x for x in b["tradeoff"] if x["trace"] == "Mooncake" and x["outer"] == "E2" and x["mem_GiB"] == 63.0 and x["policy"] == "C1-2"][0]
    assert abs(tr["primary_GiB"] - 160.0 * 120) < 1e-6 and abs(tr["ssd_util"] - 160.0 * 120 / 36000.0) < 1e-9


def test_c0_util_gate_with_k_and_nm_gate(tmp_path):
    """(a) C0 的 SSD 使用率＝(寫入÷寫頻寬＋讀取÷讀頻寬)÷(T×k)，k＝這個切片這個記憶體的 k（所有引擎同一個）：k 放大 → 使用率成比例變小；
    <5% 的格子「SSD 不忙」→ GO、NO-GO 都不算。(b) 護欄輸入（C0、O、任何對手）有 NOT_MEASURED → 這格 GO、NO-GO 都不算。
    (c) NO-GO 只看能判定的格子；每份進判定的資料至少要有 1 個能判定的格子。"""
    # (a) k=1：C0 使用率 100×120/36000＝0.333；k=10 → 0.0333 <5%
    ok = Fx(_sub(tmp_path, "k1"), S5(), mk_bytes(0.02)).analyze()
    assert abs(_row(ok, "Mooncake", "E2", 63.0)["c0_util"] - 100.0 * 120 / 36000.0) < 1e-9 and ok["verdict"]["verdict"] == "NO-GO"
    k10 = Fx(_sub(tmp_path, "k10"), S5(), mk_bytes(0.02), k_fn=lambda t, s, m: 10).analyze()
    r = _row(k10, "Mooncake", "E2", 63.0)
    assert abs(r["c0_util"] - 100.0 * 120 / 36000.0 / 10) < 1e-9 and not r["ssd_busy"] and not r["evaluable"] and "SSD 不忙" in r["why_not"]
    assert k10["verdict"]["verdict"] == "ASK" and "沒有任何能判定的格子" in k10["verdict"]["reason"]          # 全部不忙 → 不是 NO-GO
    # 只有 Mooncake 的 435 GiB 那幾格 k 很大（不忙）：其他格子照判定；NO-GO 只看能判定的格子
    part = Fx(_sub(tmp_path, "kp"), S5(), mk_bytes(0.02), k_fn=lambda t, s, m: 10 if (s == "m/c" and m == 435.0) else 1).analyze()
    assert part["verdict"]["verdict"] == "NO-GO" and part["verdict"]["evaluable_cells"]["Mooncake"] == 10 and part["verdict"]["evaluable_cells"]["WildChat"] == 15
    assert all(x["ssd_busy"] is False for x in part["rows"] if x["trace"] == "Mooncake" and x["mem_GiB"] == 435.0)

    # 不忙的格子即使省 30% 也不能支持 GO
    def f(trace, s, ol, mem, label, tag):
        if label in AM.ORACLE_LABELS and (ol, mem) == ("E2", 63.0):
            fr = {"Bailian coder": 0.30, "WildChat": 0.10}.get(trace, 0.0)
            return 82.0 * (1 - fr) + {"O": 0, "O-tier": 0, "O-len": 5, "O-reuse": 8, "O-loose": -3}[label]
        return mk_bytes(0.0)(trace, s, ol, mem, label, tag)
    go = Fx(_sub(tmp_path, "go"), S5(), f)
    assert go.analyze()["verdict"]["verdict"] == "GO"
    nogo = Fx(_sub(tmp_path, "gn"), S5(), f, k_fn=lambda t, s, m: 10 if (s == "a/coder" and m == 63.0) else 1).analyze()
    assert nogo["verdict"]["verdict"] != "GO" and not _row(nogo, "Bailian coder", "E2", 63.0)["go_ok"]
    # (b) 一個對手的等待時間是 NOT_MEASURED → 這格不判定
    nm = Fx(_sub(tmp_path, "nm"), S5(), f, lambda t, s, ol, mem, lab: None if (lab == "C6" and s == "a/coder" and ol == "E2" and mem == 63.0) else (1.0, 2.0)).analyze()
    r = _row(nm, "Bailian coder", "E2", 63.0)
    assert not r["evaluable"] and "C6" in r["why_not"] and not r["go_ok"] and nm["verdict"]["verdict"] != "GO"
    assert "護欄輸入有 NOT_MEASURED" in " ".join(AM._flags_of_row(r))
    # O 是 NOT_MEASURED（min(C0,O) 參考時過去等於護欄關掉）→ 這格不判定
    onm = Fx(_sub(tmp_path, "onm"), S5(), f, lambda t, s, ol, mem, lab: None if (lab == "O" and s == "a/coder" and ol == "E2" and mem == 63.0) else (1.0, 2.0)).analyze(guard_ref="min_C0_O")
    r = _row(onm, "Bailian coder", "E2", 63.0)
    assert not r["evaluable"] and not r["go_ok"] and onm["verdict"]["verdict"] != "GO"


def test_go_reason_includes_loose_twin_and_twin_is_part_of_completeness(tmp_path):
    """寬鬆版雙胞胎（#6）：GO 的理由附上那一格雙胞胎的省多少；缺任何一個雙胞胎的格子 → IncompleteMap（用 --skip 拿掉就不給判定）。"""
    def f(trace, s, ol, mem, label, tag):
        if label in AM.ORACLE_LABELS and (ol, mem) == ("E2", 63.0):
            fr = {"Bailian coder": 0.30, "WildChat": 0.10}.get(trace, 0.0)
            return 82.0 * (1 - fr) + {"O": 0, "O-tier": 0, "O-len": 5, "O-reuse": 8, "O-loose": -3}[label]
        return mk_bytes(0.0)(trace, s, ol, mem, label, tag)

    def lf(trace, s, ol, mem, label, tag):       # 寬鬆規則下 O 在 (E2,63) 省 50%（對手 82）
        if label in ("O", "O-tier") and (ol, mem) == ("E2", 63.0):
            return 41.0
        return f(trace, s, ol, mem, label, tag)
    fx = Fx(_sub(tmp_path, "tw"), S5(), f, loose_fn=lf)
    r = fx.analyze()
    g = r["verdict"]["go_cells"][0]
    assert r["verdict"]["verdict"] == "GO" and abs(g["strong_twin"] - 0.5) < 1e-9 and abs(g["weak_twin"] - 0.5) < 1e-9
    assert "寬鬆版雙胞胎的省多少" in r["verdict"]["reason"] and "50.0%" in r["verdict"]["reason"]
    df = pd.read_csv(fx.csv, low_memory=False)
    df = df[~((df.quality_rule == "loose") & (df.trace_group == "WildChat") & (df.outer_code == "E3") & (df.mem_GiB == 170.0) & (df.policy_label == "C6"))]
    df.to_csv(fx.csv, index=False)
    with pytest.raises(AM.IncompleteMap, match="寬鬆版雙胞胎缺"):
        fx.analyze()
    df2 = pd.read_csv(fx.csv, low_memory=False)
    df2[df2.quality_rule != "loose"].to_csv(fx.csv, index=False)          # 整個雙胞胎被 --skip 掉
    with pytest.raises(AM.IncompleteMap, match="寬鬆版雙胞胎缺"):
        fx.analyze()


def test_duplicate_cells_raise(tmp_path):
    """同一個格子兩列（例如兩次 run 的結果被併進同一個 csv）→ DuplicateCell，不准默默取第一列。"""
    fx = Fx(_sub(tmp_path, "dup"), S5(), mk_bytes(0.1))
    df = pd.read_csv(fx.csv, low_memory=False)
    pd.concat([df, df.iloc[[5]]]).to_csv(fx.csv, index=False)
    with pytest.raises(AM.DuplicateCell, match="重複"):
        fx.analyze()


def test_check_row_commits(tmp_path):
    """分析的列檢查：frozen_commit 唯一且＝鎖定值；code_commit 唯一、不帶 +dirty、＝鎖定的程式 commit 或它沒改 code/p3_sim、code/p3_data 的後代；run_id 不是 adhoc。"""
    import lockfile as LK
    inputs = {"a/in1.json": "{}\n"}
    repo, fz, run = _mk_repo(tmp_path / "rc", inputs, "C0")
    lock = LK.verify_freeze(fz, repo, required=list(inputs))
    cc = lock["code_commit"]
    ok = LK.check_row_commits([fz] * 3, [cc] * 3, lock, repo, ["20261019-x"] * 3)
    assert ok == []
    assert LK.check_row_commits([fz], [cc + "+dirty"], lock, repo, ["r"]) != []
    assert any("不唯一" in e for e in LK.check_row_commits([fz, fz], [cc, fz], lock, repo, ["r", "r"]))      # fz 本身（只改文件）是後代 → 但不唯一
    assert LK.check_row_commits([fz], [fz], lock, repo, ["r"]) == []                                       # 單獨用文件後代 commit：通過（code/p3_sim 沒變）
    assert any("frozen_commit" in e for e in LK.check_row_commits(["0" * 40], [cc], lock, repo, ["r"]))
    assert any("frozen_commit" in e for e in LK.check_row_commits([""], [cc], lock, repo, ["r"]))
    assert any("adhoc" in e for e in LK.check_row_commits([fz], [cc], lock, repo, ["adhoc"]))
    assert any("空白" in e for e in LK.check_row_commits([fz], [""], lock, repo, ["r"]))
    open(os.path.join(repo, "code", "p3_sim", "x.py"), "w").write("x = 99\n")
    run("add", "-A")
    run("commit", "-qm", "code changed")
    changed = run("rev-parse", "HEAD")
    assert any("有修改" in e for e in LK.check_row_commits([fz], [changed], lock, repo, ["r"]))


def _spec_a1():
    return {"Bailian coder": dict(slices=["bailian/coder"], S=120, source="alibaba"), "Bailian traceA": dict(slices=["bailian/traceA"], S=120, source="alibaba"),
            "WildChat": dict(slices=["wildchat/1m"], S=120, source="wildchat"), "Mooncake": dict(slices=["mooncake/conv-rule-strict"], S=120, source="mooncake"),
            "FreeInference": dict(slices=AM.FI_A, S=120, source="freeinference")}


def test_analysis_freeze_gate_main(tmp_path, monkeypatch, capsys):
    """真實資料的分析：沒有 --frozen 拒絕；命令列覆蓋護欄參考／完美預測／重算計價／n_boot 拒絕；verify_freeze 失敗拒絕；列的 commit 不對拒絕；
    參數只從凍結的 04 讀（用假的 verify_freeze 給 tier／pass_size，輸出 json 記下來）；重複格子丟錯；n_boot＝1000（這裡用合成資料跑小的 n_boot 另測）。"""
    import lockfile as LK
    fz, cc = "f" * 40, "c" * 40
    fx = Fx(_sub(tmp_path, "gm"), _spec_a1(), mk_bytes(0.1), commits=(fz, cc))
    pre = str(tmp_path / "gm" / "out")
    base = ["--cells", fx.csv, "--mode", "early", "--out-prefix", pre]
    assert AM.main(base) == 2 and "--frozen" in capsys.readouterr().err
    for extra in (["--guard-ref", "C0"], ["--oracle-variant", "tier"], ["--recompute-charge", "pass_size"], ["--boot", "50"]):
        assert AM.main(base + ["--frozen", "HEAD"] + extra) == 2
        assert "不准用命令列覆蓋" in capsys.readouterr().err
    lock = dict(frozen=fz, code_commit=cc, head=cc, prereg_sha256="s" * 64, files={"x": "y"}, guard_ref="min_C0_O", oracle_variant="tier", recompute_charge="pass_size")

    def bad(*a, **k):
        raise SystemExit("凍結檢查失敗：測試")
    monkeypatch.setattr(LK, "verify_freeze", bad)
    assert AM.main(base + ["--frozen", "HEAD"]) == 2 and "凍結檢查失敗" in capsys.readouterr().err
    monkeypatch.setattr(LK, "verify_freeze", lambda *a, **k: lock)
    monkeypatch.setattr(LK, "check_row_commits", lambda *a, **k: ["列的 code_commit 不唯一：測試"])
    assert AM.main(base + ["--frozen", "HEAD"]) == 2 and "commit 檢查失敗" in capsys.readouterr().err
    monkeypatch.undo()
    # 通過：用真的 check_row_commits 的簡化版（比對列的值）；參數從 lock 讀
    monkeypatch.setattr(LK, "verify_freeze", lambda *a, **k: lock)
    monkeypatch.setattr(LK, "check_row_commits", lambda fv, cv, lk, repo, rid: [] if (set(fv) == {fz} and set(cv) == {cc}) else ["不符"])
    assert AM.main(base + ["--frozen", "HEAD"]) == 0
    js = json.load(open(pre + "_verdict.json", encoding="utf-8"))
    assert js["guard_ref"] == "min_C0_O" and js["oracle_variant"] == "tier" and js["recompute_charge"] == "pass_size" and js["n_boot"] == 1000
    assert js["freeze"]["frozen"] == fz and "凍結" not in js["guard_ref_source"] or "04 §9" in js["guard_ref_source"]
    # 列的 frozen_commit 對不上（合成資料的空字串）→ 拒絕
    fx2 = Fx(_sub(tmp_path, "gm2"), _spec_a1(), mk_bytes(0.1), commits=("", cc))
    assert AM.main(["--cells", fx2.csv, "--mode", "early", "--out-prefix", str(tmp_path / "gm2" / "o"), "--frozen", "HEAD"]) == 2
    # 合成資料：命令列可以選參數
    sx = Fx(_sub(tmp_path, "gs"), {"synth": dict(slices=["synth/small-seed1"], S=120, source="synthetic", synthetic=True)}, mk_bytes(0.1))
    assert AM.main(["--cells", sx.csv, "--synthetic", "--boot", "40", "--oracle-variant", "tier", "--recompute-charge", "pass_size", "--out-prefix", str(tmp_path / "gs" / "o")]) == 0
    js2 = json.load(open(str(tmp_path / "gs" / "o") + "_verdict.json", encoding="utf-8"))
    assert js2["oracle_variant"] == "tier" and js2["recompute_charge"] == "pass_size" and js2["n_boot"] == 40


def test_c0_util_mode_synthetic_end_to_end(tmp_path, monkeypatch):
    """--c0-util（凍結前的例外）在合成切片上：只有 C0、15 列（4 個 session 外層 + E6，各 3 個記憶體）、沒有等待時間欄；使用率欄位可以用同一列的分量重算；
    k 是這個切片的 E6 的 k（kgpu_kmin_all.csv）；job json 裡也沒有等待時間。"""
    monkeypatch.setenv("RUN_ID", "20261018-000000-c0utiltest")
    out, csvp = str(tmp_path / "c0"), str(tmp_path / "c0_ssd_util.csv")
    assert RM.main(["--c0-util", "--slices", "synth/small-seed1", "--out", out, "--c0-util-csv", csvp, "--procs", "8"]) == 0
    d = pd.read_csv(csvp)
    assert len(d) == 15 and set(d.outer) == {"E2", "E3", "E4", "E5-T1800", "E6"} and set(d.mem_GiB) == {63.0, 170.0, 435.0}
    assert not any("ttft" in c for c in d.columns) and (d.run_id == "20261018-000000-c0utiltest").all() and (d.frozen_commit == "NOT_FROZEN").all()
    kall = RM.load_kall()
    for _, r in d.iterrows():
        assert r.k == kall[("synth/small-seed1", r.mem_GiB, "idle")]["k"]
        u = (r.c0_write_GiB / r.ssd_bw_w_GiBps + r.c0_read_GiB / r.ssd_bw_r_GiBps) / (r.T_H_s * r.k)
        assert abs(u - r.utilization) < 1e-9 * max(1, u) and abs(r.gib_per_hour - (r.c0_write_GiB + r.c0_read_GiB) / (r.T_H_s / 3600)) < 1e-6 * max(1, r.gib_per_hour)
    for root, _, files in os.walk(os.path.join(out, "jobs")):
        for fn in files:
            j = json.load(open(os.path.join(root, fn)))
            assert j["comp"] == "c0util" and all("ttft" not in k for r in j["rows"] for k in r)


def test_runner_resume_commit_check(tmp_path):
    """續跑的 commit 檢查：舊 json 的 frozen_commit／code_commit 和這次的不同 → 真實資料丟錯（不默默沿用）；合成資料當作沒做過重跑；相同＝跳過。"""
    kall = {(s, m, v): dict(k=1, hourmax=0.1, overload=False) for s in RM.SLICES for m in RM.A1_MEM + RM.A2_MEM for v in ("idle", "busy")}
    jobs = RM.enumerate_jobs("A1", ["synth/small-seed1"], False, (), kall)[:3]
    out = str(tmp_path)
    RM.classify(out, jobs)
    for j in jobs:
        open(j["npz_path"], "wb").write(b"x")
        json.dump(dict(job_id=j["id"], status="ok", rows=[], frozen_commit="F1", code_commit="C1"), open(j["json_path"], "w"))
    todo, done, _ = RM.classify(out, jobs, True, dict(frozen_commit="F1", code_commit="C1"))
    assert len(done) == 3 and not todo
    with pytest.raises(SystemExit, match="續跑檢查失敗"):
        RM.classify(out, jobs, True, dict(frozen_commit="F1", code_commit="C2"))
    with pytest.raises(SystemExit, match="續跑檢查失敗"):
        RM.classify(out, jobs, True, dict(frozen_commit="F2", code_commit="C1"))
    todo, done, _ = RM.classify(out, jobs, True, dict(frozen_commit="", code_commit="C9+dirty"))      # 合成：不同 → 重跑
    assert len(todo) == 3 and not done


def test_real_run_refuses_cli_charge_override_and_c0_util_mode(tmp_path, monkeypatch):
    """runner：真實資料（有 --frozen）不准 --recompute-charge；--c0-util 需要明確的 --out 與 run_id；C0 使用率列只有 C0 的使用率欄位（沒有等待時間）。"""
    import lockfile as LK
    monkeypatch.setattr(LK, "verify_freeze", lambda *a, **k: dict(frozen="x" * 40, guard_ref="C0", oracle_variant="greedy", recompute_charge="none"))
    monkeypatch.setattr(RM, "check_frozen", lambda c: ("f" * 40, "s" * 64))
    monkeypatch.setenv("RUN_ID", "20261018-000000-test")
    with pytest.raises(SystemExit, match="不准用 --recompute-charge"):
        RM.main(["--stage", "A1", "--slices", "bailian/coder", "--frozen", "HEAD", "--recompute-charge", "pass_size", "--out", str(tmp_path / "o"), "--limit", "1"])
    monkeypatch.delenv("RUN_ID")
    with pytest.raises(SystemExit, match="c0-util"):
        RM.main(["--c0-util", "--slices", "synth/small-seed1", "--out", str(tmp_path / "c")])
    kall = {(s, m, v): dict(k=2, hourmax=0.1, overload=False) for s in RM.SLICES for m in RM.A1_MEM + RM.A2_MEM for v in ("idle", "busy")}
    jobs = RM.c0util_jobs(["synth/small-seed1"], kall)
    assert len(jobs) == (4 + 1) * 3 and {j["pol"]["label"] for j in jobs} == {"C0"} and {j["comp"] for j in jobs} == {"c0util"}
    assert {(j["outer"], j["T"]) for j in jobs} == {("E2", None), ("E3", None), ("E4", None), ("E5", 1800), ("E6", None)}
    assert all(j["k"] == 2 for j in jobs if j["engine"] == "block")
    row = RM.c0util_row(dict(slice="s", outer_code="E5", T_s=1800, engine="session", mem_GiB=63.0, k_util=2, prim_w_H_GiB=3.0, prim_r_H_GiB=6.0, T_H_s=36000.0,
                             ssd_bw_w_GiBps=1.0, ssd_bw_r_GiBps=2.0, prim_read_col="x", H_base_s=1800.0, job_id="j", run_id="r", ts="t", code_commit="c",
                             frozen_commit="", ttftq_H_mean_s=1.0))
    assert row["outer"] == "E5-T1800" and abs(row["utilization"] - (3.0 / 1.0 + 6.0 / 2.0) / (36000.0 * 2)) < 1e-12 and abs(row["gib_per_hour"] - 9.0 / 10.0) < 1e-12
    assert not any("ttft" in k for k in row) and row["frozen_commit"] == "NOT_FROZEN"


def test_warm_caches_builds_each_slice_once_before_workers(tmp_path, monkeypatch):
    """第一次建 P 快取的峰值記憶體很大（33 GB）；warm_caches 要讓每個 (切片, 疊加) 只建一次，而且不超過 warm_procs 個同時（06 §26.8）。"""
    calls = tmp_path / "calls.txt"

    def fake_prepare_cached(src, var, overlay=1, seed=0, **kw):
        with open(calls, "a") as fh:
            fh.write(f"{src}/{var}/{overlay}\n")
        return dict(meta=dict(cache=dict(hit=False)))
    monkeypatch.setattr(RM.TR, "prepare_cached", fake_prepare_cached)
    j = lambda s, o: dict(slice=s, overlay=o)   # noqa: E731
    todo = [j("synth/small-seed1", 1)] * 5 + [j("synth/small-seed1", 2), j("synth/small-seed1", 1)]
    RM.warm_caches(todo, 2, str(tmp_path))
    got = sorted(open(calls).read().split())
    assert got == ["synth/small-seed1/1", "synth/small-seed1/2"] or len(got) == 2, got
    assert len(got) == 2
    assert "P 快取" in open(tmp_path / "runner.log", encoding="utf-8").read()
