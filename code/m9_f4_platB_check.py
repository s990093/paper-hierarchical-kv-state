#!/usr/bin/env python3
"""F4 稽核（fork B）：從平台 B（MI300X）的 run 目錄 stdout／result.json 抽出 main.tex §平台 B 引用的數字。

只讀 /mlsteam/data/tiara/runs/ 底下 2026-09 的 run 目錄，不碰 GPU、不改任何既有檔。
產出：results/audit_20261010/fragments/platB_evidence.csv（每列一個抽出的量，含來源 run_id 與 stdout 行號）。
用法：python code/m9_f4_platB_check.py
"""
import csv
import glob
import json
import os
import re
from datetime import datetime, timezone

RUNS = "/mlsteam/data/tiara/runs"
REPO = "/mlsteam/workspace/paper-hierarchical-kv-state"
OUT = os.path.join(REPO, "results/audit_20261010/fragments/platB_evidence.csv")
AUDIT_RUN_ID = os.environ.get("RUN_ID", "NO_RUN_ID")
TS = datetime.now(timezone.utc).isoformat(timespec="seconds")

rows = []


def add(group, model, key, value, run_id, src, line="", note=""):
    rows.append(dict(audit_run_id=AUDIT_RUN_ID, ts=TS, group=group, model=model, key=key,
                     value=value, source_run_id=run_id, source_file=src, source_line=line, note=note))


def read_lines(p):
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()
    except FileNotFoundError:
        return []


def model_from_cmd(cmd):
    m = re.search(r"--model (\S+)", cmd)
    return m.group(1) if m else ""


# ---------- 1. M1 容量（measure/result.json 與 gpu_guard.json） ----------
cap = {}
for rj in sorted(glob.glob(f"{RUNS}/2026091[5-7]-*-m1-b-*/measure/result.json")):
    rdir = os.path.dirname(os.path.dirname(rj))
    rid = os.path.basename(rdir)
    m = re.match(r"\d{8}-\d{6}-m1-(b-.+?)-bf16(?:-kv(fp8|int8|int4))?(?:-rerun)?$", rid)
    if not m:
        continue
    model, dt = m.group(1), m.group(2) or "bf16"
    d = json.load(open(rj))
    g = {}
    gp = os.path.join(rdir, "gpu_guard.json")
    if os.path.exists(gp):
        g = json.load(open(gp))
    if not d.get("ready") or not d.get("kv_cache_tokens"):
        continue
    cap[(model, dt)] = (d["kv_cache_tokens"], rid, g.get("verdict", "NO_GUARD"))
for (model, dt), (tok, rid, verdict) in sorted(cap.items()):
    add("m1_capacity", model, f"kv_tokens_{dt}", tok, rid, f"{rid}/measure/result.json", "", f"guard={verdict}")
for model in sorted({k[0] for k in cap}):
    if (model, "bf16") not in cap:
        continue
    base = cap[(model, "bf16")][0]
    for dt in ("fp8", "int8", "int4"):
        if (model, dt) in cap:
            add("m1_capacity", model, f"ratio_{dt}", round(cap[(model, dt)][0] / base, 4),
                cap[(model, dt)][1], "算術：kv_tokens_dt / kv_tokens_bf16", "", f"bf16_run={cap[(model, 'bf16')][1]}")

# ---------- 2. M2 重算成本 vs 位置（stdout 的 P=… repN ttft=…ms） ----------
REC = {
    "b-llama8b": "20260915-164331-m2-b-llama8b-recompute",
    "b-qwen7b-1m": "20260915-164640-m2-b-qwen7b-recompute",
    "b-qwen3-30b-a3b": "20260915-165438-m2-b-qwen3moe-recompute",
    "b-seedoss36b": "20260915-221352-m2-b-seedoss-recompute-v2",
    "b-ultralong8b-1m": "20260915-231329-m2-b-ultralong-recompute",
    "b-qwen14b-1m": "20260917-144949-m2-b-qwen14b-recompute",
    "b-mistral-nemo12b": "20260917-183724-m2-b-nemo12b-recompute",
}
pat = re.compile(r"P=\s*([\d,]+)\s+rep(\d)\s+ttft=([\d.]+)ms")
for model, rid in REC.items():
    lines = read_lines(f"{RUNS}/{rid}/stdout.log")
    pts = {}
    for ln in lines:
        mm = pat.search(ln)
        if mm:
            P = int(mm.group(1).replace(",", ""))
            pts.setdefault(P, []).append(float(mm.group(3)))
    if not pts:
        add("m2_recompute", model, "fit", "NO_DATA", rid, f"{rid}/stdout.log")
        continue
    for stat in ("min", "median"):
        xs, ys = [], []
        for P in sorted(pts):
            v = sorted(pts[P])
            y = v[0] if stat == "min" else v[len(v) // 2]
            xs.append(P)
            ys.append(y)
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        a = sxy / sxx
        c0 = my - a * mx
        ss_res = sum((y - (c0 + a * x)) ** 2 for x, y in zip(xs, ys))
        ss_tot = sum((y - my) ** 2 for y in ys)
        r2 = 1 - ss_res / ss_tot
        add("m2_recompute", model, f"{stat}_C0_ms", round(c0, 2), rid, f"{rid}/stdout.log")
        add("m2_recompute", model, f"{stat}_slope_ms_per_1k_tok", round(a * 1000, 3), rid, f"{rid}/stdout.log",
            "", "單位：每 1,000 token 的 P 增加多少 ms（= µs/token）")
        add("m2_recompute", model, f"{stat}_R2", round(r2, 4), rid, f"{rid}/stdout.log", "", f"n_pos={n}")
        add("m2_recompute", model, f"{stat}_Pmax", max(xs), rid, f"{rid}/stdout.log")
        add("m2_recompute", model, f"{stat}_y_at_P0", round(ys[0], 1), rid, f"{rid}/stdout.log")
        add("m2_recompute", model, f"{stat}_y_at_Pmax", round(ys[-1], 1), rid, f"{rid}/stdout.log")
    guard = [l for l in lines if "整機爭用" in l]
    add("m2_recompute", model, "contention_line", guard[0].strip() if guard else "NONE", rid, f"{rid}/stdout.log")

# ---------- 3. M4 壓力掃描（所有 --pressure run 的 Oracle 改善） ----------
for so in sorted(glob.glob(f"{RUNS}/2026091*-m4-*/stdout.log")):
    rid = so.split("/")[-2]
    cmd = " ".join(read_lines(f"{RUNS}/{rid}/cmd.sh"))
    if "--pressure" not in cmd and "--trace" not in cmd:
        continue
    model = model_from_cmd(cmd)
    lines = read_lines(so)
    cur = None
    for i, ln in enumerate(lines, 1):
        mm = re.match(r"--- pressure:(\d+)x ---", ln.strip())
        if mm:
            cur = f"pressure{mm.group(1)}x"
        mm2 = re.search(r"最佳 baseline = (\S+)；Oracle 改善 = \*\*([-\d.]+)%\*\*", ln)
        if mm2:
            key = cur if "--pressure" in cmd else "trace_" + re.search(r"--trace (\S+)", cmd).group(1)
            add("m4_oracle", model, key, float(mm2.group(2)), rid, f"{rid}/stdout.log", i, f"best={mm2.group(1)}")

# ---------- 4. M5 品質：撈針、GSM8K、無損 ----------
for so in sorted(glob.glob(f"{RUNS}/2026091*-m5*/stdout.log")):
    rid = so.split("/")[-2]
    cmd = " ".join(read_lines(f"{RUNS}/{rid}/cmd.sh"))
    if "m5_quality.py" not in cmd:
        continue
    mode = re.search(r"--mode (\S+)", cmd).group(1)
    model = model_from_cmd(cmd)
    lines = read_lines(so)
    ctx = re.search(r"--needle-ctx (\d+)", cmd)
    ctxs = ctx.group(1) if ctx else ""
    for i, ln in enumerate(lines, 1):
        mm = re.search(r"→ (\S+): (\d+)/(\d+) = ([\d.]+)%", ln)
        if mm:
            add(f"m5_{mode}", model, f"acc_{mm.group(1)}", float(mm.group(4)), rid, f"{rid}/stdout.log", i,
                f"{mm.group(2)}/{mm.group(3)} ctx={ctxs}")
        mm = re.match(r"^(bf16|fp8|fp8_ptk|int8|int4|full_gpu|cpu_lru|tier_fs)\s+([\d.]+)%\s+(\d+)/(\d+)\s", ln)
        if mm:
            add(f"m5_{mode}", model, f"identical_{mm.group(1)}", f"{mm.group(3)}/{mm.group(4)}", rid,
                f"{rid}/stdout.log", i, f"ctx={ctxs}")
        if "整機爭用" in ln or "外來 process：" in ln:
            add(f"m5_{mode}", model, "contention_line", ln.strip()[:160], rid, f"{rid}/stdout.log", i)

# ---------- 5. LongBench／RULER（巨觀平均與逐任務） ----------
for so in sorted(glob.glob(f"{RUNS}/2026091*-m5c-b-*/stdout.log")):
    rid = so.split("/")[-2]
    cmd = " ".join(read_lines(f"{RUNS}/{rid}/cmd.sh"))
    if "m5_understanding.py" not in cmd:
        continue
    suite = re.search(r"--suite (\S+)", cmd).group(1)
    model = model_from_cmd(cmd)
    lines = read_lines(so)
    hdr = None
    for i, ln in enumerate(lines, 1):
        if re.match(r"^(任務)\s+bf16\s+fp8\s+int8\s+int4\s+fp8_ptk", ln):
            hdr = ["bf16", "fp8", "int8", "int4", "fp8_ptk"]
            continue
        if hdr:
            parts = ln.split()
            if len(parts) == 6 and re.match(r"^[\d.]+$", parts[1]):
                vals = [float(x) for x in parts[1:]]
                task = parts[0]
                for dt, v in zip(hdr, vals):
                    add(f"m5c_{suite}", model, f"{task}_{dt}", v, rid, f"{rid}/stdout.log", i)
                add(f"m5c_{suite}", model, f"{task}_spread", round(max(vals) - min(vals), 2), rid,
                    f"{rid}/stdout.log", i, "五精度最大－最小")
                if task == "巨觀平均":
                    hdr = None
    for i, ln in enumerate(lines, 1):
        if "外來 process：" in ln:
            add(f"m5c_{suite}", model, "contention_line", ln.strip()[:160], rid, f"{rid}/stdout.log", i)
            break

# ---------- 6. 磁碟頻寬 ----------
for so in sorted(glob.glob(f"{RUNS}/2026091*-m3-diskbw/stdout.log")):
    rid = so.split("/")[-2]
    for i, ln in enumerate(read_lines(so), 1):
        mm = re.search(r"(\S+)\s+中位：寫 ([\d,]+) MiB/s、讀 ([\d,]+) MiB/s", ln)
        if mm:
            add("disk_bw", mm.group(1), "median_read_MiBps", int(mm.group(3).replace(",", "")), rid,
                f"{rid}/stdout.log", i)
            add("disk_bw", mm.group(1), "median_write_MiBps", int(mm.group(2).replace(",", "")), rid,
                f"{rid}/stdout.log", i)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {len(rows)} rows -> {OUT}")

# 方便人看：撈針、GSM8K、LongBench、RULER 的摘要
from collections import defaultdict
summ = defaultdict(dict)
for r in rows:
    if r["group"] in ("m5_needle", "m5_precision") and r["key"].startswith("acc_"):
        summ[(r["group"], r["model"], r["note"].split("ctx=")[-1], r["source_run_id"])][r["key"][4:]] = r["value"]
for k, v in sorted(summ.items()):
    vals = [v[x] for x in ("bf16", "fp8", "fp8_ptk", "int8", "int4") if x in v]
    print(k, v, "spread=", round(max(vals) - min(vals), 2) if vals else None)
for r in rows:
    if r["group"].startswith("m5c_") and r["key"].startswith("巨觀平均"):
        print(r["group"], r["model"], r["key"], r["value"], r["source_run_id"])
for r in rows:
    if r["group"] in ("m2_recompute", "m1_capacity", "disk_bw") and "contention" not in r["key"]:
        print(r["group"], r["model"], r["key"], r["value"], r["source_run_id"], r["note"])
for r in rows:
    if r["group"] == "m4_oracle":
        print(r["group"], r["model"], r["key"], r["value"], r["source_run_id"], r["note"])
