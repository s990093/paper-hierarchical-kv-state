#!/usr/bin/env python3
"""F4 稽核：從 run 目錄的 stdout 重建平台 B（MI300X）遺失的 CSV。

問題：results/m5_quality_mi300x/*.csv、results/m2_harness_mi300x/cost_constants_mi300x.csv
      等原始 CSV 從來沒進 git（commit 12aac34「不含原始 CSV」），
      又在 commit 52803bc（推倒重來）時跟著工作目錄一起刪掉。
      run 目錄裡只剩 cmd.sh / context.txt / stdout.log / server.log。

這支程式只做「抄寫＋算術」，不生資料、不碰 GPU：
  1. m2 retrieval：stdout 有逐請求的 TTFT → 逐列重建（每列 = 一個請求）。
  2. m2 recompute：stdout 有逐 rep 的 TTFT → 逐列重建。
  3. cost_constants_mi300x：用 1 的逐列資料，照 52803bc^:code/m2_analyze_b.py 的算法重算，
     並和 20260919-150345-m2-analyze 的 stdout 逐格比對（自我檢查）。
  4. recompute_fit_mi300x：同上，照原算法重算擬合並比對。
  5. m5 品質：stdout 只有「每個設定的總分／每個任務的分數」，沒有逐題輸出 →
     只能重建「摘要層級」，granularity 欄寫明 summary_from_stdout。

每列都有 run_id（外層 run，cmd.sh 所在）、inner_run_id（程式自己開的內層目錄）、ts。
ts 不是逐請求的時間（stdout 沒有），而是外層 run 的開始時間（context.txt 第一行），
ts_source 欄寫明這一點。

用法：  source code/m7_env.sh; HIP_VISIBLE_DEVICES= m7run f4-rebuild python code/m9_f4_rebuild.py
"""
from __future__ import annotations

import csv
import os
import re
import statistics as st
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

RUNS = Path(os.environ.get("TIARA_RUNS", "/mlsteam/data/tiara/runs"))
REPO = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("F4_OUT", REPO / "results/audit_20261010"))  # F4_OUT 只供乾跑測試
AUDIT_RUN_ID = os.environ.get("RUN_ID", "NO_RUN_ID")
ANALYZE_RUN = "20260919-150345-m2-analyze"      # 原本產生 cost_constants_mi300x.csv 的 run
NOW = datetime.now().astimezone().isoformat(timespec="seconds")

# 原始 m2_analyze_b.py 的設定（52803bc^:code/m2_analyze_b.py 第 21–30 行，照抄）
KV_KIB = {"b-llama8b": 128.0, "b-qwen7b-1m": 56.0, "b-qwen3-30b-a3b": 96.0,
          "b-seedoss36b": 256.0, "b-ultralong8b-1m": 128.0}
RETRIEVAL = {
    "b-llama8b": "retrieval_cost_b-llama8b.csv",
    "b-qwen7b-1m": "retrieval_cost_b-qwen7b-1m_v2.csv",
    "b-qwen3-30b-a3b": "retrieval_cost_b-qwen3-30b-a3b_v2.csv",
    "b-seedoss36b": "retrieval_cost_b-seedoss36b_v2.csv",
}
TIERS = ["gpu_resident", "gpu_fp8", "gpu_int4", "cpu", "ssd", "drop"]

NUM = r"([0-9][0-9,]*\.?[0-9]*)"


def fnum(s: str) -> float:
    return float(s.replace(",", ""))


def run_start(d: Path) -> str:
    try:
        return (d / "context.txt").read_text().splitlines()[0].strip()
    except Exception:  # noqa: BLE001
        return ""


def exit_code(d: Path) -> str:
    try:
        return (d / "exit_code").read_text().strip()
    except Exception:  # noqa: BLE001
        return "MISSING"


def parse_id_time(name: str) -> datetime | None:
    try:
        return datetime.strptime(name[:15], "%Y%m%d-%H%M%S")
    except ValueError:
        return None


ALL_DIRS = sorted(p.name for p in RUNS.iterdir() if p.is_dir())


def find_inner(outer: str, kind: str, max_s: int = 240) -> str:
    """外層 run 開跑後 max_s 秒內、名稱為 <ts>-<kind> 的第一個目錄。"""
    t0 = parse_id_time(outer)
    if t0 is None:
        return ""
    for n in ALL_DIRS:
        if not n.endswith("-" + kind) or n == outer:
            continue
        t = parse_id_time(n)
        if t is not None and timedelta(0) <= t - t0 <= timedelta(seconds=max_s):
            return n
    return ""


def outer_runs(pattern_cmd: str) -> list[Path]:
    out = []
    for n in ALL_DIRS:
        if not n.startswith("202609"):
            continue
        d = RUNS / n
        c = d / "cmd.sh"
        if not c.exists() or not (d / "stdout.log").exists():
            continue
        txt = c.read_text()
        if pattern_cmd not in txt:
            continue
        # 排除佇列外殼（一個 cmd.sh 裡排了好幾個 job）
        if txt.count("--model") > 1 or "for " in txt or "$M" in txt:
            continue
        out.append(d)
    return out


def arg(cmd: str, name: str) -> str:
    m = re.search(rf"--{name}\s+(\S+)", cmd)
    return m.group(1) if m else ""


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"[f4] 🔴 {path.name}: 0 列，不寫")
        return
    fields = list(rows[0])
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[f4] wrote {len(rows):5d} rows -> {path}")


def common(d: Path, inner: str) -> dict:
    return {"audit_run_id": AUDIT_RUN_ID, "run_id": d.name, "inner_run_id": inner,
            "ts": run_start(d), "ts_source": "run_start",
            "exit_code": exit_code(d)}


def contamination(d: Path, inner: str, lines: list[str]) -> dict:
    host = ""
    own = ""
    for ln in lines:
        m = re.search(r"整機爭用：\s*(\S+?)(?:\s|（|$)", ln)
        if m and not host:
            host = m.group(1)
        if "本卡出現外來 process" in ln:
            own = ln.split("：", 1)[-1].strip()[:90]
    cf = ""
    if inner and (RUNS / inner / "CONTAMINATED").exists():
        cf = (RUNS / inner / "CONTAMINATED").read_text().strip()[:90]
    return {"host_contention_stdout": host, "own_gpu_foreign_process_stdout": own,
            "inner_CONTAMINATED_file": cf}


# ───────────────────────── 1. m2 retrieval 逐列 ─────────────────────────
def rebuild_retrieval() -> list[dict]:
    rows = []
    for d in outer_runs("m2_cost_model.py"):
        cmd = (d / "cmd.sh").read_text()
        if arg(cmd, "stage") != "retrieval":
            continue
        lines = (d / "stdout.log").read_text(errors="replace").splitlines()
        inner = find_inner(d.name, "m2-retrieval")
        base = {**common(d, inner), **contamination(d, inner, lines),
                "model_key": arg(cmd, "model"), "csv_suffix": arg(cmd, "csv-suffix"),
                "cmd_ctx": arg(cmd, "ctx")}
        out_csv = ""
        rep, tier, ctx = 0, "", ""
        for i, ln in enumerate(lines, 1):
            if "輸出檔：" in ln:
                out_csv = ln.split("輸出檔：", 1)[1].strip().split("/results/")[-1]
            m = re.search(r"=== 第 (\d+)/(\d+) 輪", ln)
            if m:
                rep = int(m.group(1))
            m = re.search(r"retrieval\s+(\S+)\s+n_prefixes=(\d+)\s+ctx=(\d+)", ln)
            if m:
                tier, ctx = m.group(1), m.group(3)
                continue
            m = re.match(r"\s+(cold|warm)\s+#(\d+)\s+ttft=([0-9.]+)ms", ln)
            if m and tier:
                rows.append({**base, "original_csv": out_csv, "repeat": rep or 1,
                             "tier": tier, "ctx": ctx, "round": m.group(1),
                             "prefix_idx": int(m.group(2)), "ttft_ms": float(m.group(3)),
                             "source_file": f"{d.name}/stdout.log", "source_line": i,
                             "granularity": "per_request_from_stdout"})
        # stage_retrieval 在整段結束時才 write_rows 一次；沒印出 wrote 就代表這批沒進 CSV
        wrote = any(re.search(r"wrote \d+ rows -> \S*retrieval_cost_b-", ln) for ln in lines)
        for r in rows:
            if r["run_id"] == d.name:
                r["written_to_csv"] = wrote
    return rows


# ───────────────────────── 2. m2 recompute 逐列 ─────────────────────────
def rebuild_recompute() -> list[dict]:
    rows = []
    for d in outer_runs("m2_cost_model.py"):
        cmd = (d / "cmd.sh").read_text()
        if arg(cmd, "stage") != "recompute":
            continue
        lines = (d / "stdout.log").read_text(errors="replace").splitlines()
        inner = find_inner(d.name, "m2-recompute")
        base = {**common(d, inner), **contamination(d, inner, lines),
                "model_key": arg(cmd, "model"), "csv_suffix": arg(cmd, "csv-suffix")}
        chunk = ""
        pending = []
        for i, ln in enumerate(lines, 1):
            m = re.search(r"recompute chunk=(\d+)", ln)
            if m:
                chunk = m.group(1)
            m = re.match(r"\s+P=\s*(\d+)\s+rep(\d+)\s+ttft=([0-9.]+)ms", ln)
            if m:
                pending.append({**base, "chunk_tokens": chunk,
                                "cached_prefix_tokens": int(m.group(1)), "rep": int(m.group(2)),
                                "ttft_ms": float(m.group(3)),
                                "source_file": f"{d.name}/stdout.log", "source_line": i,
                                "granularity": "per_rep_from_stdout"})
            m = re.search(r"wrote (\d+) rows -> (\S+)", ln)
            if m:
                for r in pending:
                    r["original_csv"] = m.group(2).split("/results/")[-1]
                    r["written_to_csv"] = True
                rows.extend(pending)
                pending = []
        for r in pending:            # 印了但沒寫進 CSV 的（run 中途失敗）
            r["original_csv"] = ""
            r["written_to_csv"] = False
            rows.append(r)
    return rows


# ───────────────────── 3. cost_constants 重算＋比對 ─────────────────────
def parse_analyze_stdout() -> tuple[dict, dict, dict]:
    p = RUNS / ANALYZE_RUN / "stdout.log"
    cc, fit, kap = {}, {}, {}
    sec = 0
    for ln in p.read_text().splitlines():
        if ln.startswith("=== 取回成本"):
            sec = 1
        elif ln.startswith("=== C_recompute"):
            sec = 2
        elif ln.startswith("=== κ"):
            sec = 3
        elif ln.startswith("=== P*"):
            sec = 4
        t = ln.split()
        if sec == 1 and len(t) >= 4 and t[0].startswith("b-"):
            if t[3] == "NOT_MEASURED":
                cc[(t[0], t[2])] = None
            else:
                cc[(t[0], t[2])] = {"warm": float(t[3]), "delta": float(t[4]),
                                    "us": float(t[5]), "gbs": t[6], "n": int(t[7])}
        if sec == 2 and len(t) >= 7 and t[0].startswith("b-"):
            fit[(t[0], f"{t[1]} {t[2]}")] = {"C0": float(t[3]), "a": float(t[4]),
                                              "r2": float(t[5]), "dev": float(t[6])}
        if sec == 3 and len(t) >= 4 and t[0].startswith("b-"):
            kap[t[0]] = (t[2], t[3])
    return cc, fit, kap


def cost_constants(ret_rows: list[dict]) -> list[dict]:
    t_an = parse_id_time(ANALYZE_RUN)
    cc_orig, _, kap_orig = parse_analyze_stdout()
    out = []
    # 原檔是 append 模式（write_rows），所以「原始 CSV」= 所有在 analyze 之前寫到該路徑的 run
    by_file = defaultdict(list)
    for r in ret_rows:
        fn = Path(r["original_csv"]).name
        if r["written_to_csv"] and parse_id_time(r["run_id"]) < t_an:
            by_file[fn].append(r)
    models = dict(RETRIEVAL)
    # 另外三個模型（原 cost_constants 沒有）也算一份，標 in_original=False
    for extra in ("b-qwen14b-1m", "b-mistral-nemo12b", "b-ultralong8b-1m"):
        models[extra] = f"retrieval_cost_{extra}.csv"
    for model, fname in models.items():
        rows = by_file.get(fname, [])
        if not rows:
            out.append({"audit_run_id": AUDIT_RUN_ID, "ts": NOW, "model_key": model,
                        "tier": "ALL", "status": "NO_SOURCE_ROWS", "source_csv": fname})
            continue
        ctx = int(rows[0]["ctx"])
        kib = KV_KIB.get(model)
        by = defaultdict(list)
        for r in rows:
            by[(r["tier"], r["round"])].append(r["ttft_ms"])
        run_ids = sorted({r["run_id"] for r in rows})
        base = st.median(by[("gpu_resident", "warm")]) if by[("gpu_resident", "warm")] else None
        for t in TIERS:
            wv = by[(t, "warm")]
            o = cc_orig.get((model, t), "ABSENT")
            row = {"audit_run_id": AUDIT_RUN_ID, "ts": NOW, "model_key": model, "ctx": ctx,
                   "tier": t, "in_original_cost_constants": model in RETRIEVAL,
                   "source_csv": fname, "run_id": ";".join(run_ids)}
            if not wv or base is None:
                row.update({"status": "NOT_MEASURED",
                            "orig_analyze_stdout": "NOT_MEASURED" if o is None else str(o)})
                out.append(row)
                continue
            w = st.median(wv)
            delta = w - base
            us = 1000 * delta / ctx
            gbs = ((ctx * kib * 1024 / 1e9) / (delta / 1000)) if (kib and delta > 0) else ""
            cv = by[(t, "cold")]
            row.update({"warm_ttft_ms_median": round(w, 1),
                        "delta_vs_gpu_resident_ms": round(delta, 1),
                        "us_per_token": round(us, 3),
                        "effective_gb_per_s": round(gbs, 3) if gbs != "" else "",
                        "prefix_kv_bytes": int(ctx * kib * 1024) if kib else "",
                        "cold_ttft_ms_median": round(st.median(cv), 1) if cv else "",
                        "n_warm": len(wv),
                        "ssd_full_prefix_read": ("UNRECOVERABLE(req_read_bytes 不在 stdout；"
                                                 "原 analyze stdout 無「未完整讀回」註記)"
                                                 if t == "ssd" else "")})
            if isinstance(o, dict):
                ok = (abs(o["warm"] - round(w, 1)) <= 0.051 and abs(o["us"] - round(us, 2)) <= 0.0051
                      and o["n"] == len(wv))
                row["orig_analyze_stdout"] = f"warm={o['warm']} us/tok={o['us']} n={o['n']}"
                row["status"] = "MATCH_ORIGINAL" if ok else "MISMATCH_ORIGINAL"
            else:
                row["orig_analyze_stdout"] = "ABSENT（原 cost_constants 沒有這個模型）"
                row["status"] = "NEW_NOT_IN_ORIGINAL"
            row["granularity"] = "recomputed_from_per_request_stdout_rows"
            out.append(row)
        # κ 列
        ws = {t: (st.median(by[(t, "warm")]) if by[(t, "warm")] else None) for t in TIERS}
        if base is not None and ws["drop"] is not None:
            rc = ws["drop"] - base
            for t in ("cpu", "ssd"):
                k = (rc / (ws[t] - base)) if ws[t] and ws[t] - base > 0 else None
                ok_orig = kap_orig.get(model, ("", ""))[0 if t == "cpu" else 1]
                out.append({"audit_run_id": AUDIT_RUN_ID, "ts": NOW, "model_key": model,
                            "ctx": ctx, "tier": f"kappa_{t}=drop/{t}(harness 整段 warm TTFT 減基準)",
                            "in_original_cost_constants": model in RETRIEVAL,
                            "source_csv": fname, "run_id": ";".join(run_ids),
                            "us_per_token": round(k, 4) if k else "N/A",
                            "orig_analyze_stdout": ok_orig or "ABSENT",
                            "status": ("MATCH_ORIGINAL" if (ok_orig and k and ok_orig == f"{k:.2f}x")
                                       or (ok_orig == "N/A" and not k) else
                                       ("NEW_NOT_IN_ORIGINAL" if not ok_orig else "MISMATCH_ORIGINAL")),
                            "granularity": "recomputed_from_per_request_stdout_rows"})
    return out


# ───────────────────── 4. recompute 擬合重算＋比對 ─────────────────────
def fit_linear(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx
    c0 = my - a * mx
    ss_res = sum((y - (c0 + a * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return c0, a, (1 - ss_res / ss_tot) if ss_tot else float("nan")


def recompute_fit(rec_rows: list[dict]) -> list[dict]:
    t_an = parse_id_time(ANALYZE_RUN)
    _, fit_orig, _ = parse_analyze_stdout()
    # 原 analyzer：glob recompute_position_b-*.csv，sorted，同一 model 後面的檔蓋掉前面的
    by_file = defaultdict(list)
    for r in rec_rows:
        if r.get("written_to_csv") and parse_id_time(r["run_id"]) < t_an:
            by_file[Path(r["original_csv"]).name].append(r)
    chosen = {}
    for fn in sorted(by_file):
        rows = by_file[fn]
        chosen[rows[0]["model_key"]] = (fn, rows)
    out = []
    for model, (fn, rows) in sorted(chosen.items()):
        by = defaultdict(list)
        for r in rows:
            by[r["cached_prefix_tokens"]].append(r["ttft_ms"])
        stats = {"med": {P: st.median(v) for P, v in sorted(by.items())},
                 "min": {P: min(v) for P, v in sorted(by.items())}}
        for stat in ("med", "min"):
            pts = sorted(stats[stat].items())
            for label, sel in ((f"{stat} P<=114688", [p for p in pts if p[0] <= 114688]),
                               (f"{stat} all", pts)):
                if len(sel) < 3:
                    continue
                xs = [float(p) for p, _ in sel]
                ys = [v for _, v in sel]
                c0, a, r2 = fit_linear(xs, ys)
                dev = max(abs(y - (c0 + a * x)) / y for x, y in zip(xs, ys)) * 100
                o = fit_orig.get((model, label))
                ok = (o is not None and abs(o["C0"] - round(c0, 1)) <= 0.051
                      and abs(o["a"] - round(1000 * a, 2)) <= 0.0051
                      and abs(o["r2"] - round(r2, 4)) <= 0.00005)
                out.append({"audit_run_id": AUDIT_RUN_ID, "ts": NOW, "model_key": model,
                            "stat": stat, "range": label, "n_positions": len(sel),
                            "n_rows": sum(len(by[P]) for P, _ in sel),
                            "C0_ms": round(c0, 2), "a_us_per_1k_token": round(1000 * a, 3),
                            "r2": round(r2, 5), "max_abs_dev_pct": round(dev, 2),
                            "max_P": max(int(x) for x in xs),
                            "C_at_maxP_ms": round(ys[-1], 1), "C_at_P0_ms": round(ys[0], 1),
                            "source_csv": fn,
                            "run_id": ";".join(sorted({r["run_id"] for r in rows})),
                            "orig_analyze_stdout": (f"C0={o['C0']} a={o['a']} R2={o['r2']}"
                                                    if o else "ABSENT"),
                            "status": "MATCH_ORIGINAL" if ok else ("MISMATCH_ORIGINAL" if o else
                                                                   "NO_ORIGINAL"),
                            "granularity": "recomputed_from_per_rep_stdout_rows"})
    return out


# ───────────────────────── 5. m5 品質摘要 ─────────────────────────
def rebuild_quality() -> list[dict]:
    rows = []
    for script, kind_of in (("m5_quality.py", None), ("m5_understanding.py", None)):
        for d in outer_runs(script):
            cmd = (d / "cmd.sh").read_text()
            lines = (d / "stdout.log").read_text(errors="replace").splitlines()
            mode = arg(cmd, "mode") or arg(cmd, "suite")
            inner = ""
            for ln in lines:
                m = re.search(r"run_id = (\S+)", ln)
                if m:
                    inner = m.group(1)
            if not inner:
                inner = find_inner(d.name, f"m5-{mode}")
            base = {**common(d, inner), **contamination(d, inner, lines),
                    "model_key": arg(cmd, "model"), "mode": mode,
                    "needle_ctx": arg(cmd, "needle-ctx"), "ctx_arg": arg(cmd, "ctx"),
                    "max_model_len_arg": arg(cmd, "max-model-len"),
                    "n_per_task_arg": arg(cmd, "n-per-task"), "n_test_arg": arg(cmd, "n-test"),
                    "original_csv": arg(cmd, "csv").split("/results/")[-1]}
            setting = ""
            in_ident = in_final = False
            final_hdr: list[str] = []
            for i, ln in enumerate(lines, 1):
                src = {"source_file": f"{d.name}/stdout.log", "source_line": i,
                       "granularity": "summary_from_stdout"}
                m = re.match(r"\[m5c\]\s+(\S+)\s+n=\s*(\d+) tok 中位\s+(\d+) max\s+(\d+)", ln)
                if m:
                    for met, v in (("prompt_tokens_median", m.group(3)), ("prompt_tokens_max", m.group(4))):
                        rows.append({**base, "setting": "_all", "task": m.group(1), "metric": met,
                                     "value": float(v), "n": int(m.group(2)), **src})
                m = re.search(r"max_model_len = ([0-9,]+)", ln)
                if m and mode != "needle":   # needle 模式這行是 GSM8K 前綴的，不是 needle server 的
                    rows.append({**base, "setting": "_all", "task": "_all",
                                 "metric": "max_model_len_effective", "value": fnum(m.group(1)),
                                 "n": "", **src})
                m = re.search(r"共用前綴 = ([0-9,]+) tokens", ln)
                if m and mode != "needle":
                    rows.append({**base, "setting": "_all", "task": "_all",
                                 "metric": "gsm8k_shared_prefix_tokens", "value": fnum(m.group(1)),
                                 "n": "", **src})
                m = re.search(r"大海撈針：ctx=([0-9,]+)", ln)
                if m:
                    base["needle_ctx"] = m.group(1).replace(",", "")
                m = re.match(r"\[m5c?\] === (\S+?)\s*（", ln)
                if m:
                    setting = m.group(1)
                    continue
                m = re.search(r"server up, GPU KV = ([0-9,]+) tokens", ln)
                if m and setting:
                    rows.append({**base, "setting": setting, "task": "_server",
                                 "metric": "gpu_kv_cache_tokens", "value": fnum(m.group(1)),
                                 "n": "", **src})
                m = re.search(r"→ (\S+): (\d+)/(\d+) = ([0-9.]+)%", ln)
                if m:
                    rows.append({**base, "setting": m.group(1), "task": "_all",
                                 "metric": "accuracy_pct", "value": float(m.group(4)),
                                 "n": int(m.group(3)), "n_correct": int(m.group(2)), **src})
                m = re.match(r"\[m5c\]\s+(\S+)\s+([0-9.]+)\s+\(n=(\d+)\)", ln)
                if m and setting:
                    rows.append({**base, "setting": setting, "task": m.group(1),
                                 "metric": "task_score_x100", "value": float(m.group(2)),
                                 "n": int(m.group(3)), **src})
                if ln.startswith("設定") and "與基準相同" in ln:
                    in_ident = True
                    continue
                if in_ident:
                    m = re.match(r"(\S+)\s+([0-9.]+)%\s+(\d+)/(\d+)", ln)
                    if m:
                        rows.append({**base, "setting": m.group(1), "task": "_all",
                                     "metric": "identical_to_baseline", "value": int(m.group(3)),
                                     "n": int(m.group(4)), **src})
                    elif ln.strip() and not ln.startswith("="):
                        in_ident = False
                if ln.startswith("任務") and "bf16" in ln:
                    final_hdr = ln.split()[1:]
                    in_final = True
                    continue
                if in_final:
                    t = ln.split()
                    if t and t[0] == "巨觀平均" and len(t) == len(final_hdr) + 1:
                        for s, v in zip(final_hdr, t[1:]):
                            rows.append({**base, "setting": s, "task": "_macro",
                                         "metric": "macro_avg_x100", "value": float(v), "n": "",
                                         **src})
                        in_final = False
            rows.extend(server_paperwork(base, inner))
    return rows


def server_paperwork(base: dict, inner: str) -> list[dict]:
    """內層目錄每個設定的 cmd.txt 與 server.log：實際的 --kv-cache-dtype、attention 後端、KV 容量。"""
    out = []
    if not inner or not (RUNS / inner).is_dir():
        return out
    for sd in sorted(p for p in (RUNS / inner).iterdir() if p.is_dir()):
        cmdt = (sd / "cmd.txt").read_text() if (sd / "cmd.txt").exists() else ""
        log = (sd / "server.log").read_text(errors="replace") if (sd / "server.log").exists() else ""
        dt = re.search(r"'kv_cache_dtype': '([^']+)'", log)
        be = re.findall(r"Overriding with (\S+?) out of|Using (\S+) (?:attention )?backend", log)
        kv = re.search(r"GPU KV cache size: ([0-9,]+) tokens", log)
        mml = re.search(r"--max-model-len (\d+)", cmdt)
        src = {"source_file": f"{inner}/{sd.name}/server.log", "source_line": "",
               "granularity": "server_paperwork"}
        out.append({**base, "setting": sd.name, "task": "_server",
                    "metric": "server_kv_cache_dtype_cmd",
                    "value": "", "value_str": (re.search(r"--kv-cache-dtype (\S+)", cmdt) or [None, "auto(未指定)"])[1],
                    "n": "", **src})
        out.append({**base, "setting": sd.name, "task": "_server", "metric": "server_kv_cache_dtype_log",
                    "value": "", "value_str": dt.group(1) if dt else "auto(未出現在 non-default args)",
                    "n": "", **src})
        out.append({**base, "setting": sd.name, "task": "_server", "metric": "server_attention_backend_log",
                    "value": "", "value_str": ";".join(sorted({a or b for a, b in be})) or "NOT_FOUND",
                    "n": "", **src})
        if kv:
            out.append({**base, "setting": sd.name, "task": "_server", "metric": "server_gpu_kv_cache_tokens_log",
                        "value": fnum(kv.group(1)), "n": "", **src})
        if mml:
            out.append({**base, "setting": sd.name, "task": "_server", "metric": "server_max_model_len_cmd",
                        "value": int(mml.group(1)), "n": "", **src})
    return out


def manifest(items: list[tuple[str, list[dict], str]]) -> list[dict]:
    out = []
    for fname, rows, what in items:
        out.append({"audit_run_id": AUDIT_RUN_ID, "ts": NOW, "file": f"results/audit_20261010/{fname}",
                    "n_rows": len(rows),
                    "n_source_runs": len({r.get("run_id", "") for r in rows}),
                    "what": what})
    return out


def main() -> int:
    print(f"[f4] audit run_id = {AUDIT_RUN_ID}；只讀 {RUNS}，不碰 GPU")
    ret = rebuild_retrieval()
    rec = rebuild_recompute()
    cc = cost_constants(ret)
    fits = recompute_fit(rec)
    q = rebuild_quality()
    write_csv(OUT / "m2_retrieval_rows_mi300x_rebuilt.csv", ret)
    write_csv(OUT / "m2_recompute_rows_mi300x_rebuilt.csv", rec)
    write_csv(OUT / "cost_constants_mi300x_rebuilt.csv", cc)
    write_csv(OUT / "recompute_fit_mi300x_rebuilt.csv", fits)
    q_srv = [r for r in q if r.get("granularity") == "server_paperwork"]
    q = [r for r in q if r.get("granularity") != "server_paperwork"]
    write_csv(OUT / "m5_quality_mi300x_summary_rebuilt.csv", q)
    write_csv(OUT / "m5_server_paperwork_mi300x.csv", q_srv)
    write_csv(OUT / "REBUILD_MANIFEST.csv", manifest([
        ("m2_retrieval_rows_mi300x_rebuilt.csv", ret,
         "逐請求 TTFT（cold/warm × tier × prefix × repeat），抄自各 m2 retrieval run 的 stdout。"
         "缺：req_read_bytes、chunk_queries、逐列 ts、host 爭用細節"),
        ("m2_recompute_rows_mi300x_rebuilt.csv", rec,
         "逐 rep 的重算 TTFT（位置 P × rep），抄自 m2 recompute stdout；fill 列不收（原 CSV 也不收）"),
        ("cost_constants_mi300x_rebuilt.csv", cc,
         "照 52803bc^:code/m2_analyze_b.py 的算法從上一檔重算；與 20260919-150345-m2-analyze stdout 逐格比對"),
        ("recompute_fit_mi300x_rebuilt.csv", fits,
         "照原算法重算 C0 + a·P 擬合，與 analyze stdout 比對"),
        ("m5_quality_mi300x_summary_rebuilt.csv", q,
         "品質摘要（每設定總分、每任務分數、與基準逐字相同數、GPU KV 容量）。"
         "逐題輸出不存在於任何地方 → 無法重建逐題 CSV"),
        ("m5_server_paperwork_mi300x.csv", q_srv,
         "每個品質 run × 精度設定的 server cmd.txt／server.log：實際 --kv-cache-dtype、attention 後端、KV 容量、max-model-len"),
    ]))
    # 自我檢查摘要
    for name, rows in (("cost_constants", cc), ("recompute_fit", fits)):
        c = defaultdict(int)
        for r in rows:
            c[r.get("status", "")] += 1
        print(f"[f4] {name} 比對原 analyze stdout：{dict(c)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
