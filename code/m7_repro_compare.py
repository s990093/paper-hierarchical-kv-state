"""m7_repro_compare.py — H200 上「第一階段的復現」的檢查與判定（03_plan_h200.md §R1–§R3）

子命令：
  r1-check <correctness_*.json>...
      R1：test_m7_correctness.py 的逐位元幾項（HF 兩次、重算兩次、主機往返）全部 True 才回 0。
      任何一項 False 或欄位不存在 → 回 1（r_h200.sh 整個 job 停；03 §R1、§9 第一列）。
  busy-levels --res <dir> [--targets 1.5 2.0]
      R2 的 C5：從 <dir>/calib_c5.csv 挑出讓重算慢約 1.5 倍、2 倍的 duty，寫 <dir>/busy_levels.json
      （A1 的 busy1／busy2 讀這個檔；MI300X 上是 0.4／0.7 → 1.48×／2.07×，那份是手寫的）。
      最接近的 duty 離目標超過 0.25× → 回 1（要加 duty 重量，不硬選）。
  compare --mi300x <dir> --h200 <dir> [--write-summaries] [--fig-dir <dir>] [--out-csv …] [--out-json …]
      R3：兩台都用 m7_analyze.py 的同一套算法（calib／a0／a1／b；在暫存目錄的複本上跑，不覆寫原檔），
      再照 03 §R3 那張表（開跑前寫死）判定。**只比方向**；兩台數字並排寫進 repro_vs_mi300x.csv，
      這張表不進精度這條線的任何圖或判定（03 §R3）。

判定（03 §R3）：δ* ＝ max(5%, H200 的 C4 雜訊 δ)（m7_analyze calib() 的 delta_pct）。
  (0) kv_bad：H200 的 A0、A1、B 的 kv_bad 合計必須是 0（03 §R1：正確性）。不過 → 停下回報
  (1) 讀取時 Cake 有用（A0）：至少一個（頻寬, L）格，Cake 比「只算」「只載」中較快的再快超過 δ*。
      加速＝min(只算, 只載) ÷ Cake − 1（m7_analyze a0() 的 summary_a0_speedup.csv）。不過 → 停下回報，不進 D0
  (2) 寫入時依位置決定沒有淨價值（B 主設定：share、harness、nfs、gap 0、ssd_frac 1、seed 0）：
      每個 cpu_frac 的 S5 對 S4+（m7_analyze b() 的 s5_vs_s4plus_pct＝(S4+ − S5) ÷ S4+，回來請求 TTFT 中位數）
      在 ±δ* 內 → 重現；S5 快超過 δ* → 不重現（和第一階段相反，停下回報）；S5 慢超過 δ* → 「要看」（方向沒相反，但超出 ±δ*）
  (3) GPU 忙時沒有備份的代價（A1 Q2：S3 對 S0、gap＝inf、busy1／busy2）：MI300X 上顯著的格子
      （CI 不重疊、超過 δ、S3 較慢），H200 上也是 S3 較慢 → 重現。只記錄回報，不擋 D0
  (4) C6（harness ÷ vLLM）：只報告，不判定
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

BITWISE_KEYS = ["hf_twice_bitwise_equal", "recompute_same_chunking_bitwise_equal", "host_roundtrip_bitwise_equal"]
MAIN_B = {"io_model": "share", "cpu_tier": "harness", "ssd_dev": "nfs", "gap_s": 0.0, "ssd_frac": 1.0, "wl_seed": 0}
BUSY_MAX_OFF = 0.25      # 最接近的 duty 離目標慢倍數超過這個 → 不硬選，回 1
BUSY_WARN_OFF = 0.10


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ============================================================ R1
def r1_check(a):
    ok = True
    for p in a.files:
        try:
            d = json.load(open(p))
        except Exception as e:                       # 讀不到不等於過（規則 7）
            print(f"FAIL {p}: 讀不到 {e!r}")
            ok = False
            continue
        bad = [k for k in BITWISE_KEYS if d.get(k) is not True]
        info = {k: d.get(k) for k in ("L", "own_vs_hf_maxabs_logit", "own_vs_hf_argmax_equal", "own_vs_hf_top5_overlap",
                                      "chunk_bytes", "torch")}
        print(("PASS " if not bad else "FAIL ") + os.path.basename(p), {k: d.get(k) for k in BITWISE_KEYS}, info)
        ok &= not bad
    print("R1 逐位元：" + ("全部通過" if ok else "有不過的項目 → 照 03 §R1：全部停、回報"))
    return 0 if ok else 1


# ============================================================ R2：busy levels
def busy_levels(a):
    import pandas as pd
    p = os.path.join(a.res, "calib_c5.csv")
    if not os.path.exists(p):
        print(f"FAIL 找不到 {p}（先跑 C5：m7_write_policy.py busy-calib）")
        return 1
    d = pd.read_csv(p)
    med = d[d.duty > 0].groupby("duty").slowdown.median()
    if med.empty:
        print("FAIL calib_c5.csv 沒有 duty > 0 的列")
        return 1
    print("C5：每個 duty 的重算慢倍數（中位數）\n" + med.round(3).to_string())
    out, ok = {"idle": 0.0}, True
    measured, offs = {}, {}
    for name, target in zip(("busy1", "busy2"), a.targets):
        duty = float((med - target).abs().idxmin())
        off = abs(float(med[duty]) - target)
        measured[name], offs[name] = round(float(med[duty]), 3), round(off, 3)
        out[name] = duty
        if off > BUSY_MAX_OFF:
            print(f"FAIL {name}：目標 {target}×，最接近的是 duty {duty} → {med[duty]:.3f}×（差 {off:.3f}）。"
                  f"加 duty 重跑 C5，不硬選（CLAUDE.md 規則 2）")
            ok = False
        elif off > BUSY_WARN_OFF:
            print(f"WARN {name}：duty {duty} → {med[duty]:.3f}×，離目標 {target}× 差 {off:.3f}")
    if ok and not out["busy1"] < out["busy2"]:
        print(f"FAIL busy1 的 duty（{out['busy1']}）不小於 busy2（{out['busy2']}）")
        ok = False
    if not ok:
        return 1
    out.update({"_measured_slowdown": measured, "_targets": dict(zip(("busy1", "busy2"), a.targets)),
                "_off_target": offs, "_source_runs": sorted(d.run_id.astype(str).unique().tolist()),
                "_rule": "每個 duty 取 slowdown 中位數，選最接近目標的（m7_repro_compare.py busy-levels）",
                "_written": ts(), "_written_by_run": os.environ.get("RUN_ID", "adhoc")})
    dst = os.path.join(a.res, "busy_levels.json")
    if os.path.exists(dst):
        print("覆寫舊的 busy_levels.json：", open(dst).read().strip())
    json.dump(out, open(dst, "w"), ensure_ascii=False)
    print("寫入", dst, json.dumps(out, ensure_ascii=False))
    return 0


# ============================================================ R3：比較
def analyze(src, tier_params, figdir, tag):
    """在 src 的複本上跑 m7_analyze 的 calib／a0／a1／b（同一套算法），回傳（複本目錄, verdict dict）。"""
    import m7_analyze as A
    tmp = tempfile.mkdtemp(prefix=f"m7repro_{tag}_")
    for f in os.listdir(src):
        if f.endswith((".csv", ".json")) and os.path.isfile(os.path.join(src, f)):
            shutil.copy2(os.path.join(src, f), tmp)
    note = None
    if not os.path.exists(os.path.join(tmp, "tier_params.json")):
        # H200 的限速層參數沿用 MI300X 的（03 §R2）；a0() 會讀這個檔
        if not tier_params or not os.path.exists(tier_params):
            raise FileNotFoundError(f"{src} 沒有 tier_params.json，也沒有給 --tier-params")
        shutil.copy2(tier_params, os.path.join(tmp, "tier_params.json"))
        note = f"tier_params.json 來自 {tier_params}"
    fd = os.path.join(figdir, tag)
    os.makedirs(fd, exist_ok=True)
    A.R, A.FIG = tmp, fd
    A.V.clear()
    errors = []
    for fn in (A.calib, A.a0, A.a1, A.b):
        try:
            fn()
        except Exception as e:                       # 缺資料要看得到，不靜默（和 m7_analyze 主程式同一個處理）
            import traceback
            traceback.print_exc()
            errors.append(f"{fn.__name__}: {e!r}")
    V = json.loads(json.dumps(A.V, default=str))
    V["errors"] = errors
    if note:
        V["note"] = note
    return tmp, V


def _read(d, name):
    import pandas as pd
    p = os.path.join(d, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def _run_ids(d):
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".csv") and not f.startswith("summary_"):
            try:
                with open(os.path.join(d, f)) as fh:
                    r = csv.DictReader(fh)
                    if r.fieldnames and "run_id" in r.fieldnames:
                        out[f] = sorted({row["run_id"] for row in r})
            except Exception:
                pass
    return out


def compare(a):
    import numpy as np
    import pandas as pd
    mi, h2 = os.path.abspath(a.mi300x), os.path.abspath(a.h200)
    out_csv = a.out_csv or os.path.join(h2, "repro_vs_mi300x.csv")
    out_json = a.out_json or os.path.join(h2, "repro_verdict.json")
    for p in (out_csv, out_json):
        if os.path.dirname(os.path.abspath(p)) == mi:
            sys.exit(f"不寫進 MI300X 的目錄：{p}")
    if a.write_summaries and mi == h2:
        sys.exit("--write-summaries 時 --h200 不能是 MI300X 的目錄（會覆寫 MI300X 的 summary）")
    figdir = a.fig_dir or tempfile.mkdtemp(prefix="m7repro_figs_")
    tp = a.tier_params or os.environ.get("M7_TIER_PARAMS") or os.path.join(mi, "tier_params.json")
    tm, Vm = analyze(mi, tp, figdir, "mi300x")
    th, Vh = analyze(h2, tp, figdir, "h200")

    rows = []

    def row(section, key, metric, vm, vh, unit="", note=""):
        def f(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "NOT_MEASURED"
            return round(float(x), 4) if isinstance(x, (int, float, np.floating, np.integer)) else x
        rows.append({"section": section, "key": key, "metric": metric, "mi300x": f(vm), "h200": f(vh),
                     "unit": unit, "note": note})

    crit = {}
    # ---- δ
    dm = Vm.get("calib", {}).get("C4", {}).get("delta_pct")
    dh = Vh.get("calib", {}).get("C4", {}).get("delta_pct")
    if dh is None:
        dstar, dnote = None, "H200 沒有 C4（c4_noise.csv）→ δ* 無法定，(1)(2) 判不了"
    else:
        dstar, dnote = max(5.0, float(dh)), "δ* = max(5%, H200 的 δ)"
    row("C4", "noise", "delta_pct", dm, dh, "%", "m7_analyze calib()：各 slot×proc_mode 的 CI 半寬 ÷ 中位數，取最大")
    row("C4", "noise", "delta_star", None if dm is None else max(5.0, dm), dstar, "%", dnote)

    # ---- 校準並排（只報告）
    for k in ("f0_ms", "f79_ms", "slope_ms_per_chunk"):
        row("C1", "f(i)", k, Vm.get("calib", {}).get("C1", {}).get(k), Vh.get("calib", {}).get("C1", {}).get(k), "ms")
    c0 = {t: _read(d, "calib_c0.csv") for t, d in (("mi300x", mi), ("h200", h2))}
    for item in ("H2D_pinned", "D2H_pinned"):
        vals = {}
        for t, d in c0.items():
            if d is None:
                vals[t] = None
                continue
            x = d[(d["item"] == item) & (pd.to_numeric(d["size_mib"], errors="coerce") == 1024)].copy()
            x["rep"] = pd.to_numeric(x["rep"], errors="coerce")
            x = x[x.rep > 0]                                   # 排除 rep 0（暖機；tier_params 的規則）
            vals[t] = pd.to_numeric(x["value"], errors="coerce").median() if len(x) else None
        row("C0", item + " 1 GiB", "GiB/s (median, rep>0)", vals["mi300x"], vals["h200"], "GiB/s")
    for t_case in ("alone", "with_d2h"):
        v = {}
        for t, d in (("mi300x", mi), ("h200", h2)):
            x = _read(d, "calib_c0_duplex.csv")
            v[t] = None if x is None else x[x.case == t_case].h2d_GiBps.median()
        row("C0", f"duplex {t_case}", "h2d_GiBps", v["mi300x"], v["h200"], "GiB/s")
    bm, bh = (json.load(open(os.path.join(d, "busy_levels.json"))) if os.path.exists(os.path.join(d, "busy_levels.json"))
              else {} for d in (mi, h2))
    for k in ("busy1", "busy2"):
        row("C5", k, "duty", bm.get(k), bh.get(k))
        row("C5", k, "measured_slowdown", bm.get("_measured_slowdown", {}).get(k), bh.get("_measured_slowdown", {}).get(k), "×")
    c7m = _read(mi, "calib_c7_self_fixed.csv")
    c7m = c7m if c7m is not None else _read(mi, "calib_c7_self.csv")
    c7h = _read(h2, "calib_c7_self.csv")
    row("C7-self", "限速器", "err_pct_min", None if c7m is None else c7m.err_pct.min(),
        None if c7h is None else c7h.err_pct.min(), "%", "MI300X 用修正後的 calib_c7_self_fixed.csv")
    row("C7-self", "限速器", "err_pct_max", None if c7m is None else c7m.err_pct.max(),
        None if c7h is None else c7h.err_pct.max(), "%")

    # ---- (0) kv_bad
    def kvbad(V, t):
        tot = 0
        for k in ("A0", "A1"):
            tot += int(V.get(k, {}).get("kv_bad_total", 0) or 0)
        sb = _read(t, "summary_b.csv")
        if sb is not None:
            tot += int(pd.to_numeric(sb.kv_bad, errors="coerce").fillna(0).sum())
        return tot
    kb_m, kb_h = kvbad(Vm, tm), kvbad(Vh, th)
    have_h = all(os.path.exists(os.path.join(th, f)) for f in ("summary_a0.csv", "summary_a1.csv", "summary_b.csv"))
    row("R1", "kv_bad", "total (A0+A1+B)", kb_m, kb_h)
    crit["0_kv_bad"] = {"status": ("NOT_MEASURED" if not have_h else "PASS" if kb_h == 0 else "FAIL"),
                        "h200_kv_bad_total": kb_h, "rule": "A0、A1、B 的 kv_bad 合計＝0（03 §R1）；FAIL → 停下回報"}

    # ---- (1) A0：Cake 有用
    sp = {t: _read(d, "summary_a0_speedup.csv") for t, d in (("mi300x", tm), ("h200", th))}
    for t, s in sp.items():
        if s is not None:
            s["gain_pct"] = (s[["ttft_compute", "ttft_load"]].min(axis=1) / s.ttft_cake - 1) * 100
    if sp["h200"] is None or dstar is None:
        crit["1_cake_useful"] = {"status": "NOT_MEASURED", "rule": "至少一格 gain > δ*"}
    else:
        h = sp["h200"]
        hit = h[h.gain_pct > dstar]
        band = sorted(hit[hit.L == hit.L.max()].bw.tolist()) if len(hit) else []
        crit["1_cake_useful"] = {
            "status": "REPRODUCED" if len(hit) else "NOT_REPRODUCED",
            "n_cells_over_delta_star": int(len(hit)), "n_cells": int(len(h)),
            "max_gain_pct": float(h.gain_pct.max()), "bands_at_max_L": band,
            "mi300x_bands_at_max_L": (sorted(sp["mi300x"][(sp["mi300x"].gain_pct > dstar) &
                                                         (sp["mi300x"].L == sp["mi300x"].L.max())].bw.tolist())
                                      if sp["mi300x"] is not None else None),
            "rule": "至少一個（頻寬, L）格：min(只算, 只載) ÷ Cake − 1 > δ*（03 §R3 第一列）；不過 → 停下回報，不進 D0"}
    keys = set()
    for s in sp.values():
        if s is not None:
            keys |= set(zip(s.L, s.bw))
    for L, bw in sorted(keys, key=lambda x: (x[0], str(x[1]))):
        def g(s, col):
            if s is None:
                return None
            x = s[(s.L == L) & (s.bw == bw)]
            return float(x[col].iloc[0]) if len(x) else None
        row("A0", f"L={L} bw={bw}", "cake_gain_vs_best_pct", g(sp["mi300x"], "gain_pct"), g(sp["h200"], "gain_pct"), "%")
        row("A0", f"L={L} bw={bw}", "meet_measured", g(sp["mi300x"], "meet_measured"), g(sp["h200"], "meet_measured"), "chunk")

    # ---- (2) B：S5 對 S4+
    bv = {t: _read(d, "summary_b_verdict.csv") for t, d in (("mi300x", tm), ("h200", th))}

    def main_rows(v):
        if v is None:
            return None
        m = v.copy()
        for k, val in MAIN_B.items():
            m = m[m[k].astype(str) == str(val)] if isinstance(val, str) else m[np.isclose(m[k].astype(float), val)]
        return m.sort_values("cpu_frac")
    bm_, bh_ = main_rows(bv["mi300x"]), main_rows(bv["h200"])
    fr = sorted(set(([] if bm_ is None else bm_.cpu_frac.tolist()) + ([] if bh_ is None else bh_.cpu_frac.tolist())))
    states = []
    for cf in fr:
        vm_ = None if bm_ is None or not (bm_.cpu_frac == cf).any() else float(bm_[bm_.cpu_frac == cf].s5_vs_s4plus_pct.iloc[0])
        vh_ = None if bh_ is None or not (bh_.cpu_frac == cf).any() else float(bh_[bh_.cpu_frac == cf].s5_vs_s4plus_pct.iloc[0])
        nrep_h = None if bh_ is None or not (bh_.cpu_frac == cf).any() else int(bh_[bh_.cpu_frac == cf].n_reps.iloc[0])
        row("B", f"main cpu_frac={cf}", "s5_vs_s4plus_pct", vm_, vh_, "%", "正＝S5 較快；MI300X 的主設定合併了 rep 0–5（main＋r36）")
        if vh_ is not None and dstar is not None:
            states.append({"cpu_frac": cf, "h200_pct": vh_, "n_reps_h200": nrep_h,
                           "state": "REPRODUCED" if abs(vh_) <= dstar else ("NOT_REPRODUCED" if vh_ > dstar else "CHECK")})
    if not states:
        crit["2_s5_vs_s4plus"] = {"status": "NOT_MEASURED", "rule": "主設定每個 cpu_frac 的 |S5 對 S4+| ≤ δ*"}
    else:
        st = [s["state"] for s in states]
        crit["2_s5_vs_s4plus"] = {
            "status": "NOT_REPRODUCED" if "NOT_REPRODUCED" in st else ("CHECK" if "CHECK" in st else "REPRODUCED"),
            "cells": states,
            "rule": "B 主設定每個 cpu_frac：|S5 對 S4+| ≤ δ* → 重現；S5 快超過 δ* → 不重現（和第一階段相反，停下回報）；"
                    "S5 慢超過 δ* → CHECK（方向沒相反，但超出 ±δ*，回報）"}

    # ---- (3) A1 Q2：GPU 忙時沒有備份比較慢
    pr = {t: _read(d, "summary_a1_pairs.csv") for t, d in (("mi300x", tm), ("h200", th))}

    def q2(p):
        if p is None:
            return None
        x = p[(p.a == "S3") & (p.b == "S0") & (p.gap.astype(str) == "inf") & p.gpu_state.isin(["busy1", "busy2"])]
        return x.set_index(["ssd_dev", "gpu_state", "L"])
    qm, qh = q2(pr["mi300x"]), q2(pr["h200"])
    sig = qm[(qm.diff_pct > 0) & qm.ci_separated.astype(bool) & qm.beyond_delta.astype(bool)] if qm is not None else None
    cells = []
    if qm is not None:
        for idx, r in qm.iterrows():
            hv = float(qh.loc[idx].diff_pct) if qh is not None and idx in qh.index else None
            row("A1-Q2", f"{idx[0]} {idx[1]} L={idx[2]}", "S3_vs_S0_diff_pct", r.diff_pct, hv, "%",
                "正＝沒有備份（S3）比較慢；" + ("MI300X 顯著" if sig is not None and idx in sig.index else "MI300X 不顯著"))
            if sig is not None and idx in sig.index:
                cells.append({"cell": list(map(str, idx)), "mi300x_pct": float(r.diff_pct), "h200_pct": hv,
                              "same_direction": None if hv is None else hv > 0})
    if qh is None or not cells:
        crit["3_busy_no_backup"] = {"status": "NOT_MEASURED", "rule": "MI300X 顯著的格子，H200 也是 S3 較慢"}
    else:
        same = [c["same_direction"] for c in cells if c["same_direction"] is not None]
        crit["3_busy_no_backup"] = {
            "status": "REPRODUCED" if same and all(same) and len(same) == len(cells) else "PARTIAL",
            "same_direction": f"{sum(same)}/{len(cells)}", "cells": cells,
            "rule": "MI300X 上顯著（CI 不重疊、> δ、S3 較慢）的格子，H200 上也是 S3 較慢；只記錄回報，不擋 D0"}

    # ---- (4) C6：harness ÷ vLLM（只報告）
    c6 = {}
    for t, d, tt in (("mi300x", mi, tm), ("h200", h2, th)):
        v = _read(d, "c6_vllm.csv")
        s0 = _read(tt, "summary_a0.csv")
        c6[t] = {}
        if v is None:
            continue
        for L, g in v.groupby("L"):
            cold = g[g.kind == "cold_compute"].ttft_s.median()
            warm = g[g.kind == "warm_cpu_load"].ttft_s.median()
            bw = []
            for m in g[g.kind == "warm_cpu_load"].offload_metrics.dropna():
                # m7_c6_vllm.py 把 metrics 的 JSON 截在 1,500 字（json.loads 讀不了）→ 用正規表示式抓這兩個計數器
                by = [float(x) for x in re.findall(r'kv_offload_total_bytes_total\{[^}]*CPU_to_GPU[^}]*\}"\s*:\s*([0-9.eE+-]+)', m)]
                tm_ = [float(x) for x in re.findall(r'kv_offload_total_time_total\{[^}]*CPU_to_GPU[^}]*\}"\s*:\s*([0-9.eE+-]+)', m)]
                if by and tm_ and sum(tm_) > 0:
                    bw.append(sum(by) / sum(tm_) / 2**30)
            harness = None
            if s0 is not None:
                hh = s0[(s0["mode"] == "compute_only") & (s0.L == L)]
                harness = float(hh.ttft.iloc[0]) if len(hh) else None
            c6[t][int(L)] = {"vllm_cold_compute_s": float(cold), "vllm_warm_cpu_load_s": float(warm),
                             "vllm_cpu_to_gpu_GiBps": float(np.median(bw)) if bw else None,
                             "harness_compute_only_s": harness,
                             "harness_over_vllm_cold": (harness / cold) if harness else None}
    for L in sorted(set(c6.get("mi300x", {})) | set(c6.get("h200", {}))):
        for k, unit in (("vllm_cold_compute_s", "s"), ("harness_compute_only_s", "s"), ("harness_over_vllm_cold", "×"),
                        ("vllm_cpu_to_gpu_GiBps", "GiB/s")):
            row("C6", f"L={L}", k, c6.get("mi300x", {}).get(L, {}).get(k), c6.get("h200", {}).get(L, {}).get(k), unit,
                "MI300X 的 vLLM 開 connector 後退回 TRITON_ATTN（F1）" if k == "harness_over_vllm_cold" else "")
    backend = {}
    for t, d in (("mi300x", mi), ("h200", h2)):
        p = os.path.join(d, "c6_vllm_backend.txt")
        backend[t] = open(p).read().strip()[-2000:] if os.path.exists(p) else "NOT_MEASURED"
    crit["4_c6"] = {"status": "REPORT_ONLY", "by_L": c6, "vllm_backend_log": backend,
                    "rule": "harness ÷ vLLM 只報告，不判定（03 §R3 第四列；D0 的 §5.2 會拆開）"}

    # ---- 總判定
    blocking = [crit["0_kv_bad"]["status"], crit["1_cake_useful"]["status"], crit["2_s5_vs_s4plus"]["status"]]
    if any(s in ("FAIL", "NOT_REPRODUCED") for s in blocking):
        overall = "NOT_REPRODUCED：停下回報，不進 D0（03 §R3）"
    elif any(s == "NOT_MEASURED" for s in blocking):
        overall = "INCOMPLETE：有判準缺資料（NOT_MEASURED），不能說重現"
    elif "CHECK" in blocking:
        overall = "CHECK：方向沒相反，但有格子超出 ±δ*，回報後再決定"
    else:
        overall = "REPRODUCED：(0)(1)(2) 都過；(3) 見下"
    res = {"generated": ts(), "run_id": os.environ.get("RUN_ID", "adhoc"), "overall": overall,
           "delta_pct_h200": dh, "delta_pct_mi300x": dm, "delta_star_pct": dstar, "criteria": crit,
           "mi300x_dir": mi, "h200_dir": h2, "tier_params": tp, "fig_dir": figdir,
           "analysis_errors": {"mi300x": Vm.get("errors"), "h200": Vh.get("errors")},
           "input_run_ids_h200": _run_ids(h2),
           "note": "只比方向；兩台的數字不放進精度這條線的任何圖或判定（03 §R3）。算法＝m7_analyze.py（calib／a0／a1／b）。"}
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "ts", "section", "key", "metric", "mi300x", "h200", "unit", "note"])
        w.writeheader()
        for r in rows:
            w.writerow({"run_id": res["run_id"], "ts": res["generated"], **r})
    json.dump(res, open(out_json, "w"), indent=1, ensure_ascii=False, default=str)
    if a.write_summaries:
        for f in os.listdir(th):
            if f.startswith("summary_") or f == "verdict.json":
                shutil.copy2(os.path.join(th, f), os.path.join(h2, f))
        json.dump(Vh, open(os.path.join(h2, "verdict.json"), "w"), indent=1, ensure_ascii=False, default=str)
    # 自我檢查：重算的 MI300X summary 和已提交的是否相同（算法有沒有變）
    drift = {}
    for f in ("summary_a0_speedup.csv", "summary_a1_pairs.csv", "summary_b_verdict.csv", "summary_c4.csv"):
        x, y = _read(mi, f), _read(tm, f)
        if x is None or y is None:
            drift[f] = "missing"
            continue
        num = [c for c in x.columns if c in y.columns and pd.api.types.is_numeric_dtype(x[c]) and pd.api.types.is_numeric_dtype(y[c])]
        try:
            with np.errstate(invalid="ignore"):         # 兩邊都是 NaN 的格子（例：缺的欄）不算差
                drift[f] = (float(np.nanmax(np.abs(x[num].to_numpy(float) - y[num].to_numpy(float))))
                            if len(x) == len(y) else f"rows {len(x)} vs {len(y)}")
        except Exception as e:
            drift[f] = repr(e)
    res["mi300x_recompute_vs_committed_maxabs"] = drift
    json.dump(res, open(out_json, "w"), indent=1, ensure_ascii=False, default=str)

    # ---- 印出
    print(f"\n=== 第一階段復現（03 §R3）  δ_H200={dh}  δ*={dstar} ===")
    for k, v in crit.items():
        extra = {kk: vv for kk, vv in v.items() if kk in ("n_cells_over_delta_star", "max_gain_pct", "bands_at_max_L",
                                                          "same_direction", "h200_kv_bad_total")}
        if k == "2_s5_vs_s4plus" and "cells" in v:
            extra["cells"] = [(c["cpu_frac"], round(c["h200_pct"], 2), c["state"]) for c in v["cells"]]
        print(f"  {k:18s} {v['status']:15s} {json.dumps(extra, ensure_ascii=False, default=str)}")
    print("  總判定：", overall)
    print("  MI300X 重算 vs 已提交 summary 的最大差：", json.dumps(drift, ensure_ascii=False))
    print("  輸出：", out_csv, out_json)
    return 0 if overall.startswith(("REPRODUCED", "CHECK")) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("r1-check")
    p.add_argument("files", nargs="+")
    p = sub.add_parser("busy-levels")
    p.add_argument("--res", default=os.environ.get("M7_RES"))
    p.add_argument("--targets", type=float, nargs=2, default=[1.5, 2.0])
    p = sub.add_parser("compare")
    p.add_argument("--mi300x", default=os.path.join(HERE, "..", "results", "m7_write_policy_mi300x"))
    p.add_argument("--h200", default=os.environ.get("M7_RES"))
    p.add_argument("--tier-params", help="H200 目錄沒有 tier_params.json 時用（預設 M7_TIER_PARAMS）")
    p.add_argument("--fig-dir", help="m7_analyze 的圖放哪（預設暫存目錄；圖的標題寫死 MI300X，不要進 git）")
    p.add_argument("--out-csv")
    p.add_argument("--out-json")
    p.add_argument("--write-summaries", action="store_true",
                   help="把 H200 的 summary_*.csv、verdict.json 寫回 --h200（和 m7_write_policy_mi300x 同結構）")
    a = ap.parse_args()
    if a.cmd == "busy-levels" and not a.res:
        sys.exit("busy-levels 要 --res 或 M7_RES")
    if a.cmd == "compare" and not a.h200:
        sys.exit("compare 要 --h200 或 M7_RES")
    sys.exit({"r1-check": r1_check, "busy-levels": busy_levels, "compare": compare}[a.cmd](a))
