#!/usr/bin/env python3
"""m8 D6：盤點「既有的」KV 精度品質資料與反量化／位元組成本資料。

- 只讀既有檔案，不跑任何新實驗，不碰 GPU。
- 平台 A（RTX 3090）的原始 CSV 只存在 git 分支上（main 已在 52803bc 刪除），用 `git show` 讀。
- 平台 B（MI300X）的原始 CSV 從未進 git，且本機也不存在；只能從 run 目錄的 stdout.log /
  server.log 解析（腳本自己印的摘要），逐深度資料只剩 RUNLOG_MI300X.md 的轉述。
- 本腳本只做計數與算術（平均、中位數、相減、單位換算），每一列都標來源檔。

輸出：
  results/m8_directions/d6_quality_inventory.csv
  results/m8_directions/d6_dequant_inventory.csv
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import statistics
import subprocess
import sys
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = Path("/mlsteam/data/tiara/runs")
BR = "origin/claude/llm-long-context-inference-27746c"
BR2 = "origin/claude/vigilant-feynman-0arh3n"
OUT = Path(os.environ.get("D6_OUT", str(REPO / "results" / "m8_directions")))
EXTRACTED_BY = os.environ.get("RUN_ID", "NO_RUN_ID")

PA, PB = "A_RTX3090", "B_MI300X"

Q_COLS = ["run_id", "ts", "source_file", "platform", "model", "ctx", "task", "kv_dtype",
          "n", "metric", "score", "depth", "notes"]
D_COLS = ["run_id", "ts", "source_file", "platform", "model", "kv_dtype", "what_timed",
          "value", "unit", "notes"]

A_MODEL = {
    "qwen-awq": "Qwen2.5-7B-Instruct-1M AWQ-INT4 noDCA (qwen-awq)",
    "llama": "Meta-Llama-3.1-8B-Instruct BF16 weights (llama; NousResearch)",
    "llama-awq": "Meta-Llama-3.1-8B-Instruct AWQ-INT4 (llama-awq)",
}
CFG_FROM_KV = {"auto": "bf16", "fp8": "fp8", "fp8_per_token_head": "fp8_ptk",
               "int8_per_token_head": "int8", "int4_per_token_head": "int4"}

qrows: list[dict] = []
drows: list[dict] = []


def q(**kw):
    r = {c: "" for c in Q_COLS}
    r.update({k: ("" if v is None else v) for k, v in kw.items()})
    r["notes"] = (r["notes"] + "; " if r["notes"] else "") + f"extracted_by={EXTRACTED_BY}"
    qrows.append(r)


def d(**kw):
    r = {c: "" for c in D_COLS}
    r.update({k: ("" if v is None else v) for k, v in kw.items()})
    r["notes"] = (r["notes"] + "; " if r["notes"] else "") + f"extracted_by={EXTRACTED_BY}"
    drows.append(r)


def git_show(path: str) -> tuple[str, str]:
    txt = subprocess.run(["git", "-C", str(REPO), "show", f"{BR}:{path}"],
                         check=True, capture_output=True, text=True).stdout
    blob = subprocess.run(["git", "-C", str(REPO), "rev-parse", f"{BR}:{path}"],
                          check=True, capture_output=True, text=True).stdout.strip()
    blob2 = subprocess.run(["git", "-C", str(REPO), "rev-parse", f"{BR2}:{path}"],
                           capture_output=True, text=True).stdout.strip()
    same = "identical_on_" + BR2.split("/")[-1] if blob2 == blob else "DIFFERS_on_" + BR2
    return txt, f"blob={blob[:10]} {same}"


def fmt(x, nd=2):
    return f"{x:.{nd}f}"


# ───────────────────────── 平台 A：品質 ─────────────────────────

def a_needle(path: str, label: str):
    txt, blob = git_show(path)
    rows = list(csv.DictReader(io.StringIO(txt)))
    src = f"git:{BR}:{path}"
    groups: "OrderedDict[tuple, list]" = OrderedDict()
    for r in rows:
        groups.setdefault((r["run_id"], r["config"]), []).append(r)
    base_sha = {}
    for (rid, cfg), g in groups.items():
        if cfg == "bf16":
            base_sha[rid] = {x["idx"]: x["out_sha1"] for x in g}
    for (rid, cfg), g in groups.items():
        ptoks = sorted({x["prompt_tokens"] for x in g})
        kvtok = sorted({x["gpu_kv_cache_tokens"] for x in g})
        lv = defaultdict(int)
        for x in g:
            lv[x["level"]] += 1
        common = (f"{label}; prompt_tokens={'/'.join(ptoks)}; gpu_kv_cache_tokens={'/'.join(kvtok)}; "
                  f"vllm_kv_dtype={g[0]['kv_dtype']}; contention_levels={dict(lv)} (score unaffected, "
                  f"latency void); {blob}")
        ok = sum(x["correct"] == "True" for x in g)
        q(run_id=rid, ts=g[0]["ts"], source_file=src, platform=PA, model=A_MODEL[g[0]["model_key"]],
          ctx=ptoks[0] if len(ptoks) == 1 else "/".join(ptoks), task="needle_single_key",
          kv_dtype=cfg, n=len(g), metric="needle_acc_pct", score=fmt(100 * ok / len(g)),
          depth="", notes=f"{ok}/{len(g)} correct; " + common)
        byd: "OrderedDict[str, list]" = OrderedDict()
        for x in g:
            byd.setdefault(x["needle_depth"], []).append(x)
        for dep, gg in byd.items():
            okd = sum(x["correct"] == "True" for x in gg)
            act = sorted({x["needle_depth_actual"] for x in gg})
            pos = sorted({x["needle_pos"] for x in gg})
            q(run_id=rid, ts=gg[0]["ts"], source_file=src, platform=PA,
              model=A_MODEL[g[0]["model_key"]], ctx=ptoks[0], task="needle_single_key",
              kv_dtype=cfg, n=len(gg), metric="needle_acc_pct_by_depth",
              score=fmt(100 * okd / len(gg)), depth=dep,
              notes=f"{okd}/{len(gg)} correct; needle_depth_actual={'/'.join(act)}; "
                    f"needle_pos_token={'/'.join(pos)}; whole KV in one dtype (not mixed); {blob}")
        if cfg != "bf16" and rid in base_sha:
            same = sum(1 for x in g if base_sha[rid].get(x["idx"]) == x["out_sha1"])
            q(run_id=rid, ts=g[0]["ts"], source_file=src, platform=PA,
              model=A_MODEL[g[0]["model_key"]], ctx=ptoks[0], task="needle_single_key",
              kv_dtype=cfg, n=len(g), metric="identical_output_vs_bf16_count", score=same,
              depth="", notes=f"out_sha1 equality vs bf16 same idx; {blob}")


def a_understanding(path: str, suite: str, ctx_note: str, ctx: str):
    txt, blob = git_show(path)
    rows = list(csv.DictReader(io.StringIO(txt)))
    src = f"git:{BR}:{path}"
    by: "OrderedDict[tuple, list]" = OrderedDict()
    for r in rows:
        by.setdefault((r["config"], r["task"]), []).append(r)
    macro = defaultdict(list)
    first = {}
    for (cfg, task), g in by.items():
        m = 100 * statistics.mean(float(x["score"]) for x in g)
        macro[cfg].append(m)
        first.setdefault(cfg, g[0])
        pt = [int(x["prompt_tokens"]) for x in g]
        trunc = sum(1 for x in g if x["truncated"] not in ("0", "", "False"))
        q(run_id=g[0]["run_id"], ts=g[0]["ts"], source_file=src, platform=PA,
          model=A_MODEL[g[0]["model_key"]], ctx=ctx, task=f"{suite}:{task}", kv_dtype=cfg,
          n=len(g), metric=f"{g[0]['metric']}_x100_mean", score=fmt(m), depth="",
          notes=f"{ctx_note}; prompt_tokens median={int(statistics.median(pt))} max={max(pt)}; "
                f"truncated={trunc}; vllm_kv_dtype={g[0]['kv_dtype']}; {blob}")
    for cfg, ms in macro.items():
        g0 = first[cfg]
        nrow = sum(len(g) for (c, _t), g in by.items() if c == cfg)
        q(run_id=g0["run_id"], ts=g0["ts"], source_file=src, platform=PA,
          model=A_MODEL[g0["model_key"]], ctx=ctx, task=f"{suite}:macro_{len(ms)}tasks",
          kv_dtype=cfg, n=nrow, metric="macro_mean_of_task_means_x100",
          score=fmt(statistics.mean(ms)), depth="",
          notes=f"{ctx_note}; computed here = mean of the {len(ms)} task means above; {blob}")


def a_gsm8k(path: str, task: str):
    txt, blob = git_show(path)
    rows = list(csv.DictReader(io.StringIO(txt)))
    src = f"git:{BR}:{path}"
    by: "OrderedDict[tuple, list]" = OrderedDict()
    for r in rows:
        by.setdefault((r["run_id"], r["config"]), []).append(r)
    base = {}
    for (rid, cfg), g in by.items():
        if cfg in ("bf16", "full_gpu"):
            base[rid] = {x["idx"]: x["out_sha1"] for x in g}
    for (rid, cfg), g in by.items():
        ok = sum(x["correct"] == "True" for x in g)
        kvd = CFG_FROM_KV.get(g[0]["kv_dtype"], g[0]["kv_dtype"])
        t = task if task.startswith("gsm8k_many") else f"{task}:{cfg}"
        q(run_id=rid, ts=g[0]["ts"], source_file=src, platform=PA, model=A_MODEL[g[0]["model_key"]],
          ctx="NOT_RECORDED", task=t, kv_dtype=kvd, n=len(g), metric="gsm8k_acc_pct",
          score=fmt(100 * ok / len(g)), depth="",
          notes=f"{ok}/{len(g)}; n_shot={g[0]['n_shot']}; prompt token count not in CSV; "
                f"config={cfg}; gpu_kv_cache_tokens={g[0]['gpu_kv_cache_tokens']}; {blob}")
        if rid in base and cfg not in ("bf16", "full_gpu"):
            same = sum(1 for x in g if base[rid].get(x["idx"]) == x["out_sha1"])
            q(run_id=rid, ts=g[0]["ts"], source_file=src, platform=PA,
              model=A_MODEL[g[0]["model_key"]], ctx="NOT_RECORDED", task=t, kv_dtype=kvd,
              n=len(g), metric="identical_output_vs_baseline_count", score=same, depth="",
              notes=f"out_sha1 equality vs bf16/full_gpu same idx; config={cfg}; {blob}")


# ───────────────────────── 平台 B：品質（stdout） ─────────────────────────

def ctx_ts(run: Path) -> str:
    try:
        return (run / "context.txt").read_text(errors="replace").splitlines()[0].strip()
    except Exception as e:  # noqa: BLE001
        return f"NO_CONTEXT({e})"


def rundir_ts(name: str) -> datetime:
    return datetime.strptime(name[:15], "%Y%m%d-%H%M%S")


ALL_RUNS = sorted(p.name for p in RUNS.iterdir() if p.is_dir() and re.match(r"\d{8}-\d{6}-", p.name))


def inner_run(wrapper: str, suffix: str) -> str:
    t0 = rundir_ts(wrapper)
    for n in ALL_RUNS:
        if n.endswith(suffix) and n != wrapper:
            dt = (rundir_ts(n) - t0).total_seconds()
            if 0 <= dt <= 300:
                return n
    return "NOT_FOUND"


def b_model(stdout: str) -> str:
    m = re.search(r"模型(?:剖面)? (b-[\w.-]+) (?:->|→) (\S+)", stdout)
    return f"{m.group(2).rstrip('/').split('/')[-1]} BF16 weights ({m.group(1)})" if m else "UNKNOWN"


def b_m5_quality(wrapper: str):
    run = RUNS / wrapper
    out = (run / "stdout.log").read_text(errors="replace")
    cmd = (run / "cmd.sh").read_text(errors="replace")
    rc = (run / "exit_code").read_text().strip() if (run / "exit_code").exists() else "NA"
    mode = re.search(r"--mode (\w+)", cmd).group(1)
    model = b_model(out)
    inner = inner_run(wrapper, f"-m5-{mode}")
    contam = (RUNS / inner / "CONTAMINATED").exists() if inner != "NOT_FOUND" else False
    src = f"{RUNS}/{wrapper}/stdout.log"
    lost_csv = re.search(r"--csv (\S+)", cmd).group(1).split("/paper-hierarchical-kv-state/")[-1]
    base_note = (f"exit_code={rc}; inner_run={inner}{' CONTAMINATED(scores kept, latency void)' if contam else ''}; "
                 f"original CSV {lost_csv} NOT in git nor on disk (lost); parsed from stdout summary")
    if rc != "0":
        return
    if mode == "needle":
        m = re.search(r"大海撈針：ctx=([\d,]+)、(\d+) 個深度 × (\d+) 次重複", out)
        ctx = m.group(1).replace(",", "")
        task, metric = "needle_single_key", "needle_acc_pct"
        note = base_note + f"; requested --needle-ctx={ctx} (actual prompt tokens only in lost CSV); depths 0.05/0.25/0.5/0.75/0.95 x4 (per-depth rows lost)"
    elif mode == "precision":
        m = re.search(r"(\d+)-shot 共用前綴 = ([\d,]+) tokens", out)
        ctx = m.group(2).replace(",", "")
        task, metric = "gsm8k_many_shot", "gsm8k_acc_pct"
        note = base_note + f"; {m.group(1)}-shot shared prefix {ctx} tokens (ctx = prefix tokens)"
    elif mode == "lossless":
        m = re.search(r"(\d+)-shot 共用前綴 = ([\d,]+) tokens", out)
        ctx = m.group(2).replace(",", "")
        task, metric = "gsm8k_lossless", "gsm8k_acc_pct"
        note = base_note + "; placement tiers (CPU/SSD) with BF16 KV, not a precision test"
    else:
        return
    for cfg, k, n, pct in re.findall(r"→ (\w+): (\d+)/(\d+) = ([\d.]+)%", out):
        kvd = "bf16" if mode == "lossless" else cfg
        t = f"{task}:{cfg}" if mode == "lossless" else task
        q(run_id=wrapper, ts=ctx_ts(run), source_file=src, platform=PB, model=model, ctx=ctx,
          task=t, kv_dtype=kvd, n=n, metric=metric, score=pct, depth="",
          notes=f"{k}/{n}; " + note)
    tbl = out.split("無損驗證")[-1] if "無損驗證" in out else ""
    for cfg, pct, same, tot in re.findall(r"^(\w+)\s+([\d.]+)%\s+(\d+)/(\d+)\s", tbl, re.M):
        if cfg in ("bf16", "full_gpu"):
            continue
        kvd = "bf16" if mode == "lossless" else cfg
        t = f"{task}:{cfg}" if mode == "lossless" else task
        q(run_id=wrapper, ts=ctx_ts(run), source_file=src, platform=PB, model=model, ctx=ctx,
          task=t, kv_dtype=kvd, n=tot, metric="identical_output_vs_baseline_count", score=same,
          depth="", notes="out_sha1 equality vs bf16/full_gpu (printed table); " + note)


def b_m5c(wrapper: str):
    run = RUNS / wrapper
    out = (run / "stdout.log").read_text(errors="replace")
    cmd = (run / "cmd.sh").read_text(errors="replace")
    rc = (run / "exit_code").read_text().strip() if (run / "exit_code").exists() else "NA"
    if rc != "0":
        return
    suite = re.search(r"--suite (\w+)", cmd).group(1)
    model = b_model(out)
    m = re.search(r"\[m5c\] run_id = (\S+)", out)
    inner = m.group(1) if m else "NOT_FOUND"
    contam = (RUNS / inner / "CONTAMINATED").exists()
    mml = re.search(r"max_model_len = ([\d,]+)", out).group(1).replace(",", "")
    rctx = re.search(r"--ctx (\d+)", cmd)
    ctx = rctx.group(1) if rctx else mml
    ctx_note = (f"RULER --ctx {ctx}, max_model_len={mml}" if rctx else
                f"LongBench window max_model_len={mml} (middle-truncation)")
    lost_csv = re.search(r"--csv (\S+)", cmd).group(1).split("/paper-hierarchical-kv-state/")[-1]
    note = (f"exit_code={rc}; inner_run={inner}{' CONTAMINATED(scores kept, latency void)' if contam else ''}; "
            f"original CSV {lost_csv} lost; parsed from stdout final table; {ctx_note}")
    toks = {t: (a, b) for t, a, b in re.findall(r"^\[m5c\]\s+(\w+)\s+n=\s*\d+ tok 中位\s+(\d+) max\s+(\d+)", out, re.M)}
    ns = {t: n for t, n in re.findall(r"^\[m5c\]\s+(\w+)\s+[\d.]+\s+\(n=(\d+)\)", out, re.M)}
    sec = out.split("× KV 精度")[-1]
    lines = sec.splitlines()
    hdr = next(l for l in lines if l.startswith("任務"))
    cfgs = hdr.split()[1:]
    for l in lines[lines.index(hdr) + 1:]:
        if l.startswith("---") or not l.strip():
            continue
        if l.startswith("相對"):
            break
        parts = l.split()
        name, vals = parts[0], parts[1:]
        if len(vals) != len(cfgs):
            continue
        is_macro = name == "巨觀平均"
        for cfg, v in zip(cfgs, vals):
            t = f"{suite}:macro_{len(ns)}tasks" if is_macro else f"{suite}:{name}"
            nn = sum(int(x) for x in ns.values()) if is_macro else ns.get(name, "")
            tk = "" if is_macro else f"; prompt_tokens median={toks.get(name, ('?', '?'))[0]} max={toks.get(name, ('?', '?'))[1]}"
            q(run_id=wrapper, ts=ctx_ts(run), source_file=f"{RUNS}/{wrapper}/stdout.log", platform=PB,
              model=model, ctx=ctx, task=t, kv_dtype=cfg, n=nn,
              metric="macro_mean_of_task_means_x100" if is_macro else "official_metric_x100_mean",
              score=v, depth="", notes=note + tk)


def b_qwen7b_depth_from_runlog():
    """RUNLOG_MI300X.md 的逐深度轉述（原 CSV 已遺失）。只收錄文字裡明寫的數字。"""
    path = "results/RUNLOG_MI300X.md"
    txt, blob = git_show(path)
    lines = txt.splitlines()
    idx = next(i for i, l in enumerate(lines) if l.startswith("Qwen-7B 逐深度"))
    seg = "\n".join(lines[idx:idx + 3])
    depths = re.search(r"（([\d.／]+)）", seg).group(1).split("／")
    wrapper = "20260916-065920-m5-b-qwen7b-needle"
    ts = ctx_ts(RUNS / wrapper)
    src = f"git:{BR}:{path}#L{idx + 1}-{idx + 3}"
    for name, vals in re.findall(r"(BF16|INT8|FP8|FP8-ptk|INT4) ([\d/]+|全 0)", seg):
        kvd = {"BF16": "bf16", "INT8": "int8", "FP8": "fp8", "FP8-ptk": "fp8_ptk", "INT4": "int4"}[name]
        vs = ["0"] * len(depths) if vals == "全 0" else vals.split("/")
        for dep, v in zip(depths, vs):
            q(run_id=wrapper, ts=ts, source_file=src, platform=PB,
              model="Qwen2.5-7B-Instruct-1M-noDCA BF16 weights (b-qwen7b-1m)", ctx="131072",
              task="needle_single_key", kv_dtype=kvd, n=4, metric="needle_acc_pct_by_depth",
              score=v, depth=dep,
              notes=("SECONDARY SOURCE: transcribed in RUNLOG from lost CSV needle_b-qwen7b-1m.csv; "
                     "not re-verifiable; consistent with stdout totals; "
                     + ("'全 0' in RUNLOG -> 0 at every depth; " if vals == "全 0" else "")
                     + f"requested ctx 131072 (RUNLOG: actual 129,067); {blob}"))


# ───────────────────────── 反量化／取回成本 ─────────────────────────

WARM_DESC = ("warm TTFT: client-side HTTP stream time-to-first-token for re-sending the IDENTICAL "
             "{ctx}-token prompt whose KV is fully resident in the GPU prefix cache in this KV dtype "
             "(1 request, max_tokens=8, temp 0); delta vs BF16 = attention over cached KV with in-kernel "
             "dequant + any backend difference; NOT a separate dequant kernel; separate vLLM server per tier")
COLD_DESC = ("cold TTFT: first send of the {ctx}-token prompt = full prefill + writing KV in this dtype "
             "(quantize-on-write included); separate vLLM server per tier")


def summarize_tiers(rows, ctx, plat, model, run_id, ts, src, extra):
    w = defaultdict(list)
    for r in rows:
        if r["round"] == "warm" and r["ttft"] is not None:
            w[r["tier"]].append(r["ttft"])
    if "gpu_resident" not in w:
        return
    b = statistics.median(w["gpu_resident"])
    bmin, bmax = min(w["gpu_resident"]), max(w["gpu_resident"])
    for tier in ("gpu_fp8", "gpu_int4"):
        v = w.get(tier)
        if not v:
            d(run_id=run_id, ts=ts, source_file=src, platform=plat, model=model,
              kv_dtype=tier.replace("gpu_", ""), what_timed="warm TTFT delta vs gpu_resident",
              value="NOT_MEASURED", unit="", notes=f"no valid warm sample for {tier}; {extra}")
            continue
        m = statistics.median(v)
        overlap = not (min(v) > bmax or max(v) < bmin)
        verdict = ("RANGE_OVERLAPS_BASELINE -> indistinguishable from 0 (A-platform rule: NOT_MEASURED)"
                   if overlap else "range separated from baseline")
        kvd = tier.replace("gpu_", "")
        common = (f"derived here (arithmetic only) from n={len(v)} warm samples median={m:.3f} "
                  f"range {min(v):.1f}-{max(v):.1f} vs gpu_resident n={len(w['gpu_resident'])} median={b:.3f} "
                  f"range {bmin:.1f}-{bmax:.1f}; {verdict}; {extra}")
        d(run_id=run_id, ts=ts, source_file=src, platform=plat, model=model, kv_dtype=kvd,
          what_timed=f"median warm TTFT delta vs BF16 gpu_resident (ctx={ctx})", value=fmt(m - b, 3),
          unit="ms per request", notes=common)
        d(run_id=run_id, ts=ts, source_file=src, platform=plat, model=model, kv_dtype=kvd,
          what_timed=f"median warm TTFT delta / ctx (ctx={ctx})", value=fmt(1000 * (m - b) / ctx, 3),
          unit="us per cached token", notes=common)
        d(run_id=run_id, ts=ts, source_file=src, platform=plat, model=model, kv_dtype=kvd,
          what_timed=f"median warm TTFT delta / (ctx/16) (ctx={ctx}; 16-token vLLM block)",
          value=fmt((m - b) / (ctx / 16), 5), unit="ms per 16-token block", notes=common)


def a_tiers_csv(path: str, contam_note: str):
    txt, blob = git_show(path)
    rows = list(csv.DictReader(io.StringIO(txt)))
    src = f"git:{BR}:{path}"
    sel = [r for r in rows if r["tier"].startswith("gpu_")]
    if not sel:
        return
    model = A_MODEL[sel[0]["model_key"]]
    ctx = int(sel[0]["ctx"])
    for r in sel:
        kvd = {"gpu_resident": "bf16"}.get(r["tier"], r["tier"].replace("gpu_", ""))
        desc = (WARM_DESC if r["round"] == "warm" else COLD_DESC).format(ctx=ctx)
        d(run_id=r["run_id"], ts=r["ts"], source_file=src, platform=PA, model=model, kv_dtype=kvd,
          what_timed=desc, value=r["ttft_ms"], unit="ms (TTFT)",
          notes=(f"tier={r['tier']}; vllm_kv_dtype={r['kv_dtype']}; gpu_kv_cache_tokens={r['gpu_kv_cache_tokens']}; "
                 f"host_contention={r['host_contention']} foreign_gpu_count={r['foreign_gpu_count']}; "
                 f"{contam_note}; {blob}"))
    rr = [{"tier": r["tier"], "round": r["round"],
           "ttft": float(r["ttft_ms"]) if r["ttft_ms"] else None} for r in sel]
    summarize_tiers(rr, ctx, PA, model, sel[0]["run_id"], sel[0]["ts"], src, f"{contam_note}; {blob}")


def b_retrieval(wrapper: str):
    run = RUNS / wrapper
    out = (run / "stdout.log").read_text(errors="replace")
    cmd = (run / "cmd.sh").read_text(errors="replace")
    rc = (run / "exit_code").read_text().strip() if (run / "exit_code").exists() else "NA"
    model = b_model(out)
    ctx = int(re.search(r"--ctx (\d+)", cmd).group(1))
    inner = inner_run(wrapper, "-m2-retrieval")
    src = f"{RUNS}/{wrapper}/stdout.log"
    ts = ctx_ts(run)
    cont = re.search(r"整機爭用：(\S+)\s+外來 process (\d+) 個", out)
    note0 = (f"exit_code={rc}; inner_run={inner}; host contention={cont.group(1) if cont else '?'} "
             f"foreign_procs={cont.group(2) if cont else '?'}; original CSV results/m2_harness_mi300x/"
             f"retrieval_cost_{model.split('(')[-1].rstrip(')')}.csv lost (not in git, not on disk)")
    rows, tier, rnd = [], None, 0
    for l in out.splitlines():
        m = re.match(r"\[m2\] === 第 (\d+)/\d+ 輪", l)
        if m:
            rnd = int(m.group(1))
        m = re.match(r"\[m2\] retrieval (\w+)\s", l)
        if m:
            tier = m.group(1)
        m = re.match(r"\s+(cold|warm) #(\d+) ttft=([\d.]+|None)ms", l)
        if m and tier and tier.startswith("gpu_"):
            v = None if m.group(3) == "None" else float(m.group(3))
            rows.append({"tier": tier, "round": m.group(1), "ttft": v, "rep": rnd})
    for r in rows:
        kvd = {"gpu_resident": "bf16"}.get(r["tier"], r["tier"].replace("gpu_", ""))
        desc = (WARM_DESC if r["round"] == "warm" else COLD_DESC).format(ctx=ctx)
        d(run_id=wrapper, ts=ts, source_file=src, platform=PB, model=model, kv_dtype=kvd,
          what_timed=desc, value="NOT_MEASURED(None)" if r["ttft"] is None else r["ttft"],
          unit="ms (TTFT)", notes=f"tier={r['tier']}; interleaved round {r['rep']}/3; {note0}")
    # 腳本自己印的摘要（論文 0.17 / 1.59 的出處）
    for name, wm, dl, us in re.findall(r"^(gpu_fp8|gpu_int4)\s+([\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)$", out, re.M):
        d(run_id=wrapper, ts=ts, source_file=src, platform=PB, model=model, kv_dtype=name.replace("gpu_", ""),
          what_timed=f"AS PRINTED by m2_cost_model.py: (median warm TTFT - median gpu_resident warm TTFT)/ctx, ctx={ctx}",
          value=us, unit="us per cached token",
          notes=f"printed warm median={wm} ms, delta={dl} ms; {note0}")
    for name in re.findall(r"^(gpu_fp8|gpu_int4)\s+NOT_MEASURED", out, re.M):
        d(run_id=wrapper, ts=ts, source_file=src, platform=PB, model=model, kv_dtype=name.replace("gpu_", ""),
          what_timed=f"AS PRINTED by m2_cost_model.py (ctx={ctx})", value="NOT_MEASURED", unit="",
          notes=f"all warm samples returned ttft=None (no token streamed); {note0}")
    summarize_tiers(rows, ctx, PB, model, wrapper, ts, src, note0)


# ───────────────────────── KV 位元組／token ─────────────────────────

def a_capacity():
    path = "results/m2_harness/capacity_by_dtype.csv"
    txt, blob = git_show(path)
    for r in csv.DictReader(io.StringIO(txt)):
        kib = float(r["kv_cache_gib"]) * 1024 * 1024 / int(r["kv_cache_tokens"])
        d(run_id=r["run_id"], ts=r["ts"], source_file=f"git:{BR}:{path}", platform=PA, model=A_MODEL[r["model_key"]],
          kv_dtype=r["kv_dtype_name"],
          what_timed="capacity (not timed): kv_cache_gib*2^20/kv_cache_tokens from vLLM startup log",
          value=fmt(kib, 2), unit="KiB per token",
          notes=(f"rep={r['rep']}; tokens={r['kv_cache_tokens']}; gib={r['kv_cache_gib']} (2-dec rounding => ~±0.1%); "
                 f"flag={r['kv_cache_dtype_flag']}; contention irrelevant to bytes; {blob}"))
    path = "results/m2_harness/idle_cost_normalized.json"
    txt, blob = git_show(path)
    js = json.loads(txt)
    for k, v in js.items():
        d(run_id="20260830-194119-m2-capacity", ts="2026-08-30 (run)", source_file=f"git:{BR}:{path}", platform=PA,
          model=A_MODEL["llama"], kv_dtype=k,
          what_timed="capacity (not timed): repo-normalized KiB/token over 3 reps",
          value=v["kib_per_token"], unit="KiB per token",
          notes=f"overhead_kib_vs_ideal={v['overhead_kib']}; rel_bf16={v['rel_bf16']}; spread_pct={v['spread_pct']}; {blob}")
    path = "results/m1_capacity/capacity.csv"
    txt, blob = git_show(path)
    for r in csv.DictReader(io.StringIO(txt)):
        if r["phase"] == "measure" and r["kv_gib"] and r["kv_cache_tokens"]:
            kib = float(r["kv_gib"]) * 1024 * 1024 / int(r["kv_cache_tokens"])
            d(run_id=r["run_id"], ts=r["ts"], source_file=f"git:{BR}:{path}", platform=PA,
              model=f"{r['model'].rstrip('/').split('/')[-1]} ({r['weight_dtype']} weights; {r['config']})",
              kv_dtype=CFG_FROM_KV.get(r["kv_dtype"], r["kv_dtype"]),
              what_timed="capacity (not timed): kv_gib*2^20/kv_cache_tokens (M1 measure phase)",
              value=fmt(kib, 2), unit="KiB per token",
              notes=f"tokens={r['kv_cache_tokens']}; kv_gib={r['kv_gib']}; config={r['config']}; {blob}")


def b_capacity():
    for name in ALL_RUNS:
        if "-m1-b-" not in name:
            continue
        sl = RUNS / name / "measure" / "server.log"
        ct = RUNS / name / "measure" / "cmd.txt"
        if not sl.exists() or not ct.exists():
            continue
        log = sl.read_text(errors="replace")
        cmd = ct.read_text(errors="replace")
        g = re.findall(r"Available KV cache memory: ([\d.]+) GiB", log)
        t = re.findall(r"GPU KV cache size: ([\d,]+) tokens", log)
        if not g or not t:
            continue
        gib, tok = float(g[-1]), int(t[-1].replace(",", ""))
        kv = re.search(r"--kv-cache-dtype (\S+)", cmd)
        kvd = CFG_FROM_KV.get(kv.group(1), kv.group(1)) if kv else "bf16"
        mdl = re.search(r"vllm serve (\S+)", cmd).group(1).rstrip("/").split("/")[-1]
        ver = "vllm0.28.0" if "tiara-v028" in cmd else "vllm0.19.1(venv tiara)"
        contam = (RUNS / name / "CONTAMINATED").exists()
        d(run_id=name, ts=f"{name[:8]}-{name[9:15]} (run dir)", source_file=str(sl), platform=PB,
          model=f"{mdl} BF16 weights", kv_dtype=kvd,
          what_timed="capacity (not timed): 'Available KV cache memory' GiB*2^20 / 'GPU KV cache size' tokens",
          value=fmt(gib * 1024 * 1024 / tok, 2), unit="KiB per token",
          notes=(f"tokens={tok}; gib={gib} (2-dec rounding => <0.01%); {ver}; "
                 f"{'CONTAMINATED flag in run dir (RUNLOG_MI300X finding 6 says false positive; rerun exists)' if contam else 'clean'}"))


# ───────────────────────── main ─────────────────────────

def main() -> int:
    # 平台 A 品質
    a_needle("results/m5_quality/needle_pilot_32k.csv", "needle pilot 32K (RUNLOG A wrapper 20260831-181834-m5-needle-pilot)")
    a_needle("results/m5_quality/needle_fair_32k.csv", "needle 32K rerun named 'fair' (commit 1944b20: the fair fp8_per_token_head comparison failed on sm_86 with Triton-backend ValueError; only the 4 original configs exist)")
    a_needle("results/m5_quality/needle_ctx_sweep.csv", "needle ctx sweep 4K/8K/16K (RUNLOG A wrapper 20260831-182840-m5-needle-ctxsweep)")
    a_understanding("results/m5_quality/longbench_precision.csv", "longbench",
                    "LongBench 7 EN tasks, first 50 each, window max_model_len=32768 (m5_understanding default)", "32768")
    a_understanding("results/m5_quality/ruler_precision.csv", "ruler",
                    "RULER 7 tasks, --ctx 16384, first 30 each", "16384")
    a_gsm8k("results/m5_quality/gsm8k_precision.csv", "gsm8k_many_shot")
    a_gsm8k("results/m5_quality/gsm8k_precision_n1000.csv", "gsm8k_many_shot_n1000")
    a_gsm8k("results/m5_quality/gsm8k_lossless.csv", "gsm8k_lossless")
    # 平台 B 品質
    for name in ALL_RUNS:
        if not name.startswith("202609"):
            continue
        p = RUNS / name
        if not (p / "cmd.sh").exists():
            continue
        cmd = (p / "cmd.sh").read_text(errors="replace")
        if "m5_quality.py" in cmd:
            b_m5_quality(name)
        elif "m5_understanding.py" in cmd:
            b_m5c(name)
    b_qwen7b_depth_from_runlog()
    # 反量化
    a_tiers_csv("results/m2_harness/retrieval_cost_precision_tiers_quiet.csv",
                "QUIET (clean) per RUNLOG A 'M2 補充'")
    a_tiers_csv("results/m2_harness/retrieval_cost_precision_tiers.csv",
                "HEAVY contention: timing VOID per CLAUDE.md §3 (RUNLOG A says ordering wrong); inventory only")
    a_tiers_csv("results/m2_harness/retrieval_cost_llama-awq.csv",
                "HEAVY contention: timing VOID per CLAUDE.md §3; gpu_kv_cache_tokens differ across rounds; inventory only")
    a_tiers_csv("results/m2_harness/retrieval_cost_qwen-awq.csv",
                "HEAVY contention: timing VOID per CLAUDE.md §3; gpu_kv_cache_tokens differ across rounds; inventory only")
    for name in ALL_RUNS:
        p = RUNS / name
        if name.startswith("202609") and (p / "cmd.sh").exists():
            cmd = (p / "cmd.sh").read_text(errors="replace")
            if "m2_cost_model.py" in cmd and "--stage retrieval" in cmd and "--tiers" not in cmd:
                b_retrieval(name)
    for plat, model in ((PA, "any"), (PB, "any")):
        for kvd in ("int8", "fp8_ptk"):
            d(run_id="NONE", ts="", source_file="(searched: git branches, /mlsteam/data/tiara/runs)", platform=plat,
              model=model, kv_dtype=kvd, what_timed="read/dequant cost", value="NOT_MEASURED", unit="",
              notes=("no retrieval run ever included this tier (--tiers had gpu_fp8/gpu_int4 only)"
                     + ("; fp8_ptk cannot run on sm_86 (Triton backend ValueError)" if plat == PA and kvd == "fp8_ptk" else "")))
    d(run_id="NONE", ts="", source_file="(searched)", platform="A+B", model="any", kv_dtype="fp8/int8/int4",
      what_timed="standalone dequant kernel time / chunk-level dequant microbenchmark", value="NOT_MEASURED", unit="",
      notes="no kernel-level or per-chunk dequant timing exists; only end-to-end warm-TTFT deltas")
    # 位元組
    a_capacity()
    b_capacity()

    OUT.mkdir(parents=True, exist_ok=True)
    for fn, cols, rows in (("d6_quality_inventory.csv", Q_COLS, qrows),
                           ("d6_dequant_inventory.csv", D_COLS, drows)):
        with open(OUT / fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"[d6] wrote {len(rows)} rows -> {OUT / fn} ({(OUT / fn).stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
