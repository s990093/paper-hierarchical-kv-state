"""m10_restore2.py — 13 號計畫（還原的第二輪；docs/precision_20261013/13_restore2_plan.md）的工具。只報告。

  cumask --out J        GPU（43a 冒煙）：CU 數 → P4a／P4b 的 HSA_CU_MASK；背景 matmul 單獨跑的時間（有／沒有遮罩）驗證遮罩有生效
  env <方法> --model m --fmt f
                        印出 Q4 方法（P0–P5）的環境變數（K* 與 S 組由 m10_dqsweep 給；P4 的遮罩讀 cumask.json）
  smoke-check --dir D   43a 冒煙：KV 逐位元、meet_rule_used、nodq、背景吞吐、新格式 kernel → D/OK_restore
  pick-q4               13 §3.2：帶到長 context 的方法 → q4_pick.json
  pick-c                13 §7：42c 的冠軍（CPU 層 E1 中位數對 8 格的幾何平均最小）→ c_pick.json
  plan-44d --jobs J     13 §7：寫 J/queue/44d_rs_longctx.sh
  report                各段的表 → results/m10_restore2_mi300x/tables_13.md、report_13.json（還沒跑的段寫 NOT_MEASURED 或略過）
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
# H200（10/19；沒設時不變）：M10_RESTORE2_DIR、M10_SWEEP_DIR、M10_PRECISION_DIR
R2 = os.environ.get("M10_RESTORE2_DIR") or os.path.join(REPO, "results", "m10_restore2_mi300x")
SWROOT = os.environ.get("M10_SWEEP_DIR") or os.path.join(REPO, "results", "m10_sweep_mi300x")
SW = os.path.join(SWROOT, "dqsweep")
PREC = os.environ.get("M10_PRECISION_DIR") or os.path.join(REPO, "results", "m10_precision_mi300x")
MODELS = ("llama", "qwen")
TAGS = ("fp8", "int4")
# 13 §3.1：Q4 的方法＝（S 組代號, 另外的環境變數）；"@p4a" 等由 cumask.json 換成實際遮罩
Q4 = {"P0": ("S0", {}), "P1": ("S0", {"M10_RS_PRIO": "1"}), "P2": ("S2k8", {}), "P3": ("S4", {}),
      "P4a": ("S0", {"M10_BUSY_CU_MASK": "@p4a"}), "P4b": ("S0", {"M10_BUSY_CU_MASK": "@p4b"}),
      "P5": ("S2k8", {"M10_RS_PRIO": "1"})}
MASKED = ("P4a", "P4b")


def fmt_of(m, tag):
    return json.load(open(os.path.join(PREC, "fmt_select", "fmt_choice.json")))["choice"][m][tag]


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def latest_rows(d):
    p = os.path.join(d, "restores.csv")
    if not os.path.exists(p):
        return []
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return []
    last = max(r["run_id"] for r in rows)
    return [r for r in rows if r["run_id"] == last]


def usable(d, rows):
    """gpu_guard 全部 CLEAN、沒有污染列（CLAUDE.md §3）。"""
    if not rows:
        return False
    rid = rows[0]["run_id"]
    gs = [json.load(open(g)) for g in glob.glob(os.path.join(d, f"gpu_guard_{rid}_*.json"))]
    return bool(gs) and all(g.get("verdict") == "CLEAN" for g in gs) and all(r["contaminated_gpu"] == "False" for r in rows)


def med(rows, **kw):
    v = [float(r["t_restore"]) for r in rows if all(r[k] == str(x) for k, x in kw.items())]
    return statistics.median(v) if v else None


# ---------------------------------------------------------------- cumask（GPU）
def _busy_ready_ms(mask: str, duty=0.7, reps=3):
    out = []
    for _ in range(reps):
        env = dict(os.environ)
        if mask:
            env["HSA_CU_MASK"] = mask
        p = subprocess.Popen([sys.executable, os.path.join(HERE, "m7_busy.py"), "--duty", str(duty)],
                             stdout=subprocess.PIPE, text=True, env=env)
        line = p.stdout.readline()
        p.terminate()
        p.wait(30)
        if not line.startswith("READY"):
            raise SystemExit(f"m7_busy 沒有啟動：{line!r}")
        out.append(float(line.split("matmul_ms=")[1].split()[0]))
    return statistics.median(out), out


def cmd_cumask(a):
    import torch
    n_cu = torch.cuda.get_device_properties(0).multi_processor_count
    del torch
    masks = {"none": "", "p4a": f"0:0-{n_cu * 7 // 8 - 1}", "p4b": f"0:0-{n_cu * 3 // 4 - 1}"}
    res = {"n_cu": n_cu, "masks": masks, "matmul_ms": {}, "rule": "遮罩生效＝背景單獨跑的 matmul 時間 ≥ 沒遮罩的 ×1.05（p4a）／×1.10（p4b）"}
    for k, mk in masks.items():
        res["matmul_ms"][k] = _busy_ready_ms(mk)
    base = res["matmul_ms"]["none"][0]
    res["effective"] = {"p4a": res["matmul_ms"]["p4a"][0] >= 1.05 * base, "p4b": res["matmul_ms"]["p4b"][0] >= 1.10 * base}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))
    if not all(res["effective"].values()):
        raise SystemExit("HSA_CU_MASK 沒有生效（13 §3.1）→ P4 拿掉")


# ---------------------------------------------------------------- env
def method_env(meth: str, m: str, fmt: str) -> dict:
    import m10_dqsweep as D
    s, extra = Q4[meth]
    env = D.method_env(s, m, fmt)
    for k, v in extra.items():
        if v.startswith("@"):
            cm = json.load(open(os.path.join(R2, "smoke", "cumask.json")))
            v = cm["masks"][v[1:]]
        env[k] = v
    return env


def cmd_env(a):
    print(" ".join(f"{k}={v}" for k, v in method_env(a.method, a.model, a.fmt).items()))


# ---------------------------------------------------------------- 冒煙檢查
def cmd_smoke_check(a):
    S, fails = a.dir, []

    def chk(ok, msg):
        print(("PASS " if ok else "FAIL ") + msg, flush=True)
        if not ok:
            fails.append(msg)

    rows = latest_rows(os.path.join(S, "g5"))
    chk(bool(rows) and all(r["kv_bad"] in ("", "0") for r in rows), "g5：KV 逐位元")
    chk({r["mode"] for r in rows} == {"cake", "loadall", "claim"}, f"g5：三種模式 {sorted({r['mode'] for r in rows})}")
    chk(all(r["meet_rule_used"] in ("none", "claim") for r in rows if r["mode"] == "cake"),
        f"g5：每列有決定 {sorted({r['meet_rule_used'] for r in rows if r['mode'] == 'cake'})}")
    chk(all(r["n_recompute"] == "0" for r in rows if r["mode"] == "loadall"), "g5：loadall 不重算")
    rows = latest_rows(os.path.join(S, "nodq"))
    e1 = [r for r in rows if r["config"] == "E1"]
    chk(bool(e1) and all(float(r["t_dequant"]) == 0 and r["n_recompute"] == "0" for r in e1), "nodq：E1 不拆、不重算")
    chk(all(r["kv_bad"] in ("", "0", "SKIP_nodq") for r in rows), "nodq：kv_bad")
    for name in ("p5mask",):
        d = os.path.join(S, name)
        rows = latest_rows(d)
        chk(bool(rows) and all(r["kv_bad"] in ("", "0") for r in rows), f"{name}：KV 逐位元（S2k8＋優先權＋遮罩、GPU 忙）")
        bs = [x for f in glob.glob(os.path.join(d, "busy_*.json")) for x in json.load(open(f))]
        # 遮罩只在 cumask 生效（OK_cumask）時才傳；H200（CUDA 沒有 HSA_CU_MASK）沒生效 → 只檢查背景吞吐（10/19；MI300X 有 OK_cumask，行為不變）
        need_mask = os.path.exists(os.path.join(S, "OK_cumask"))
        chk(bool(bs) and all(x.get("n_matmul", 0) > 0 and (x.get("cu_mask") or not need_mask) for x in bs),
            f"{name}：背景吞吐有記錄{'、遮罩有傳' if need_mask else '（遮罩沒生效，不傳）'} {bs[:1]}")
    rows = latest_rows(os.path.join(S, "dqx"))
    chk(bool(rows) and all(r["kv_bad"] in ("", "0") for r in rows), "dqx：新格式融合 kernel 的還原 KV 逐位元（int2_kivi）")
    if not fails:
        open(os.path.join(S, "OK_restore"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    print("ALL PASS" if not fails else f"{len(fails)} FAIL")
    sys.exit(1 if fails else 0)


# ---------------------------------------------------------------- 挑選規則
def _bg_rate(d, state="busy"):
    xs = [x["matmul_per_s"] for f in glob.glob(os.path.join(d, "busy_*.json")) for x in json.load(open(f))
          if x.get("state") == state and "matmul_per_s" in x]
    return statistics.mean(xs) if xs else None


def cmd_pick_q4(a):
    """13 §3.2：背景吞吐 ≥ P0 的 95% 的方法裡，「GPU 忙、全載入、E1 中位數」對 4 格的幾何平均最小；和 P0 差不到 3% 取 P0。P4 不參加。"""
    tab, notes = {}, []
    for meth in Q4:
        vals, ratios = [], []
        for m in MODELS:
            for tag in TAGS:
                d = os.path.join(R2, "q4", f"{m}_{tag}_{meth}")
                rows = latest_rows(d)
                if not usable(d, rows):
                    notes.append(f"{meth} {m} {tag}：沒有可用的資料")
                    continue
                v = med(rows, mode="loadall", config="E1", gpu_state="busy")
                if v:
                    vals.append(v)
                r0, r = _bg_rate(os.path.join(R2, "q4", f"{m}_{tag}_P0")), _bg_rate(d)
                if r0 and r:
                    ratios.append(r / r0)
        tab[meth] = dict(n_cells=len(vals), gm_busy_loadall_E1=gm(vals) if len(vals) == 4 else None,
                         bg_ratio_vs_P0=statistics.mean(ratios) if ratios else None)
    cand = {k: v for k, v in tab.items() if k not in MASKED and v["gm_busy_loadall_E1"]
            and (k == "P0" or (v["bg_ratio_vs_P0"] is not None and v["bg_ratio_vs_P0"] >= 0.95))}
    if "P0" not in cand:
        pick = None
    else:
        best = min(cand, key=lambda k: cand[k]["gm_busy_loadall_E1"])
        pick = "P0" if cand[best]["gm_busy_loadall_E1"] >= cand["P0"]["gm_busy_loadall_E1"] * 0.97 else best
    out = dict(rule="13 §3.2", table=tab, candidates=sorted(cand), pick=pick, notes=notes)
    os.makedirs(R2, exist_ok=True)
    json.dump(out, open(os.path.join(R2, "q4_pick.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if pick is None:
        raise SystemExit("Q4：P0 沒有完整的可用資料，挑不出來")


def cmd_pick_c(a):
    """13 §7：42c 的 5 個選手裡，CPU 層 E1 中位數對 8 格（2 模型 × 2 格式 × 閒／忙）的幾何平均最小的。"""
    plan = json.load(open(os.path.join(SW, "plan_C.json")))
    tab = {}
    for meth in plan["finalists"]:
        vals = []
        for m in MODELS:
            for tag in TAGS:
                d = os.path.join(SW, "C", f"{m}_{tag}_{meth.replace('+', '_')}")
                rows = latest_rows(d)
                if not usable(d, rows):
                    continue
                for st in ("idle", "busy"):
                    v = med([r for r in rows if r["mode"] == "cake"], config="E1", gpu_state=st, bw_name="cpu")
                    if v:
                        vals.append(v)
        tab[meth] = dict(n_cells=len(vals), gm_cpu_E1=gm(vals) if len(vals) == 8 else None)
    ok = {k: v for k, v in tab.items() if v["gm_cpu_E1"]}
    pick = min(ok, key=lambda k: ok[k]["gm_cpu_E1"]) if ok else None
    out = dict(rule="13 §7", table=tab, pick=pick)
    os.makedirs(R2, exist_ok=True)
    json.dump(out, open(os.path.join(R2, "c_pick.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if pick is None:
        raise SystemExit("42c：沒有完整的可用資料")


def g5_ratios(rule: str):
    """13 §8c：每格「G5（或 G5b）中位數 ÷ 最好」；最好＝CPU 層 min(Cake, 全載入)、限速層 Cake。只用可用的 run。"""
    sfx = "" if rule == "g5" else "_g5b"
    out = {}
    for m in MODELS:
        for tag in TAGS:
            for part in ("cpu", "slow"):
                d = os.path.join(R2, "q2", f"{m}_{tag}_{part}{sfx}")
                rows = latest_rows(d)
                if not usable(d, rows):
                    continue
                for bw in {r["bw_name"] for r in rows}:
                    for st in ("idle", "busy"):
                        for cf in ("E0", "E1", "R25"):
                            g = med(rows, mode="cake", bw_name=bw, gpu_state=st, config=cf)
                            ref = [x for x in (med(rows, mode="claim", bw_name=bw, gpu_state=st, config=cf),
                                               med(rows, mode="loadall", bw_name=bw, gpu_state=st, config=cf)) if x]
                            if g and ref:
                                out[f"{m}|{tag}|{bw}|{st}|{cf}"] = g / min(ref)
    return out


def cmd_pick_g5(a):
    """13 §8c：G5 和 G5b 各自對共同的格子取「÷ 最好」的幾何平均，小的帶到 44d；差不到 1% 取 G5（先登記的）。"""
    r5, r5b = g5_ratios("g5"), g5_ratios("g5b")
    common = sorted(set(r5) & set(r5b))
    res = dict(rule="13 §8c", n_common=len(common), gm_g5=gm([r5[k] for k in common]) if common else None,
               gm_g5b=gm([r5b[k] for k in common]) if common else None)
    if not common:
        res["pick"] = "g5" if r5 else None
    else:
        res["pick"] = "g5b" if res["gm_g5b"] < res["gm_g5"] * 0.99 else "g5"
    os.makedirs(R2, exist_ok=True)
    json.dump(res, open(os.path.join(R2, "g5_pick.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


# ---------------------------------------------------------------- 44d（長 context）
def cmd_plan_44d(a):
    """13 §7＋§7 的補充（10/17 寫，還沒有任何 13 號的結果）：
    P0、G5：CPU＋b32；冠軍、Q4 帶過來的、nodq：只有 CPU。loadall 只在 CPU。2 launch × 5 rep、閒／忙、E0 E1 R25。"""
    import m10_dqsweep as D
    c = json.load(open(os.path.join(R2, "c_pick.json")))["pick"]
    q = json.load(open(os.path.join(R2, "q4_pick.json")))["pick"]
    runs = [("P0", "S0", {}, "cpu b32", "loadall")]
    if c and c != "S0":
        runs.append((f"C_{c.replace('+', '_')}", c, {}, "cpu", "loadall"))
    if q and q != "P0":
        s_, ex = Q4[q]
        runs.append((f"Q4_{q}", s_, ex, "cpu", "loadall"))
    gp = os.path.join(R2, "g5_pick.json")
    g5rule = (json.load(open(gp)).get("pick") if os.path.exists(gp) else None) or "g5"
    runs.append(("G5" if g5rule == "g5" else "G5b", "S0", {"M10_MEET": g5rule}, "cpu b32", "loadall claim"))
    runs.append(("nodq", "S0", {"M10_MEET": "none", "M10_DQ_SCHED": "nodq"}, "cpu", ""))
    lines = [f"# 13 §7（Q1）長 context（由 m10_restore2.py plan-44d 寫，{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}）：",
             f"#   42c 冠軍 {c}、Q4 帶過來的 {q}、G5 的規則 {g5rule}；方法 {' '.join(r[0] for r in runs)}",
             f"source {a.jobs}/lib.sh", "set -uo pipefail", f"R={R2}/q1", "mkdir -p $R"]
    for L in (129024, 65536):
        for m in MODELS:
            for tag in TAGS:
                f = fmt_of(m, tag)
                for name, s_, ex, bws, xm in runs:
                    env = D.method_env(s_, m, f)
                    for k, v in ex.items():
                        if v.startswith("@"):
                            v = json.load(open(os.path.join(R2, "smoke", "cumask.json")))["masks"][v[1:]]
                        env[k] = v
                    if name == "nodq":
                        env.pop("M10_DQ_GROUP", None)
                    envs = " ".join(f"{k}={v}" for k, v in env.items())
                    cfgs = "E0 E1" if name == "nodq" else "E0 E1 R25"
                    xopt = f"--extra-modes {xm} --extra-tiers cpu" if xm else ""
                    dd = f"$R/{m}_{L}_{tag}_{name}"
                    cmd = (f"python $REPO/code/m10_speed.py grid --model {m} --fmt {f} --base-fmt {fmt_of(m, 'fp8')} --out-dir {dd} "
                           f"--bw-json $OUT/speed_{m}/bw.json --bw-names {bws} --configs {cfgs} --refs --lengths "
                           f"--main-length {L} {xopt} --launches 2 --reps 5 --launch0 300 --blocks-out \\$TIARA_RUNS/\\$RUN_ID/blocks.csv")
                    lines.append(f"m7run m10-rs2-q1-{m}-{L}-{tag}-{name} env {envs} bash -c \"{cmd}\" "
                                 f"|| echo \"44d：{m} {L} {tag} {name} 失敗\" >> $JOBS/NEEDS_HUMAN")
    lines.append("exit 0")
    p = os.path.join(a.jobs, "queue", "44d_rs_longctx.sh")
    open(p, "w").write("\n".join(lines) + "\n")
    print("寫了", p)


# ---------------------------------------------------------------- 報告（CPU）
def _paired(rows, a_mode, b_mode, cfg, st, bw="cpu"):
    """同一個（launch, rep）裡 a 和 b 的配對差（b − a；正＝a 快）與 bootstrap 95% CI。"""
    from m10_speed import bootstrap_ci
    k = lambda r: (r["launch"], r["rep"])                                  # noqa: E731
    A = {k(r): float(r["t_restore"]) for r in rows if (r["mode"], r["config"], r["gpu_state"], r["bw_name"]) == (a_mode, cfg, st, bw)}
    B = {k(r): float(r["t_restore"]) for r in rows if (r["mode"], r["config"], r["gpu_state"], r["bw_name"]) == (b_mode, cfg, st, bw)}
    ks = sorted(set(A) & set(B))
    d = [B[x] - A[x] for x in ks]
    if len(d) < 2:
        return None
    lo, hi = bootstrap_ci(d)
    return dict(n=len(d), median_diff=round(statistics.median(d), 5), ci95=[round(lo, 5), round(hi, 5)])


def _f(x, k=3):
    return "—" if x is None else f"{x:.{k}f}"


def _cell(rows, **kw):
    v = med(rows, **kw)
    return v


def report_q4(md, out):
    md += ["## Q4：GPU 忙時的「拆」（13 §3）", "",
           "真實 CPU 層、32K；還原時間中位數（s）。cake＝Cake（各方法自己的拆法）、loadall＝全部載入；背景吞吐＝GPU 忙時背景 matmul／秒 ÷ 同格 P0。", ""]
    for m in MODELS:
        for tag in TAGS:
            md += [f"### {m}・{fmt_of(m, tag)}", "",
                   "| 方法 | 閒 E0／E1 cake | 閒 E0／E1 loadall | 忙 E0／E1 cake | 忙 E0／E1 loadall | 忙 loadall 拆（E1，ms） | 背景吞吐 | 可用 |",
                   "|:--|:--|:--|:--|:--|--:|--:|:--|"]
            r0 = _bg_rate(os.path.join(R2, "q4", f"{m}_{tag}_P0"))
            for meth in Q4:
                d = os.path.join(R2, "q4", f"{m}_{tag}_{meth}")
                rows = latest_rows(d)
                if not rows:
                    md.append(f"| {meth} | NOT_MEASURED |  |  |  |  |  |  |")
                    continue
                c = {(mo, st, cf): med(rows, mode=mo, gpu_state=st, config=cf) for mo in ("cake", "loadall")
                     for st in ("idle", "busy") for cf in ("E0", "E1")}
                dq = [float(r["t_dequant"]) for r in rows if (r["mode"], r["gpu_state"], r["config"]) == ("loadall", "busy", "E1")]
                rr = _bg_rate(d)
                ratio = rr / r0 if rr and r0 else None
                ok = usable(d, rows)
                out.setdefault("q4", {})[f"{m}|{tag}|{meth}"] = dict(med={"|".join(k): v for k, v in c.items()}, bg_ratio=ratio,
                                                                     usable=ok, run_id=rows[0]["run_id"])
                md.append(f"| {meth} | {_f(c['cake', 'idle', 'E0'])}／{_f(c['cake', 'idle', 'E1'])} | {_f(c['loadall', 'idle', 'E0'])}／"
                          f"{_f(c['loadall', 'idle', 'E1'])} | {_f(c['cake', 'busy', 'E0'])}／{_f(c['cake', 'busy', 'E1'])} | "
                          f"{_f(c['loadall', 'busy', 'E0'])}／{_f(c['loadall', 'busy', 'E1'])} | "
                          f"{_f(1e3 * statistics.median(dq), 1) if dq else '—'} | {_f(ratio, 2)} | {'是' if ok else '**否**'} |")
            md.append("")
    pk = os.path.join(R2, "q4_pick.json")
    if os.path.exists(pk):
        md += [f"帶到長 context 的方法（13 §3.2）：**{json.load(open(pk))['pick']}**", ""]
    md += ["#### 背景強度掃描（13 §3.3；int4_kivi；CPU 層；E1 中位數 s，cake／loadall）", "",
           "| 模型 | 方法 | busy1（0.4） | busy2（0.7） | busy3（0.9） |", "|:--|:--|:--|:--|:--|"]
    for d in sorted(glob.glob(os.path.join(R2, "q4duty", "*"))):
        rows = latest_rows(d)
        if not rows:
            continue
        name = os.path.basename(d)
        cells = [f"{_f(med(rows, mode='cake', gpu_state=b, config='E1'))}／{_f(med(rows, mode='loadall', gpu_state=b, config='E1'))}"
                 for b in ("busy1", "busy2", "busy3")]
        md.append(f"| {name.split('_')[0]} | {name.split('_')[-1]} | " + " | ".join(cells) + f" |{'' if usable(d, rows) else ' **不可用**'}")
    md.append("")


def report_q2(md, out):
    md += ["## Q2：會合規則 G5（13 §4）", "",
           "還原時間中位數（s）；G5＝cake 列；最好＝min(Cake, 全載入)（限速層只有 Cake）；選全載入＝G5 決定不重算的比例。", "",
           "| 模型 | 格式 | 頻寬 | GPU | 配置 | G5 | Cake | 全載入 | G5 ÷ 最好 | 選全載入 | 可用 |",
           "|:--|:--|:--|:--|:--|--:|--:|--:|--:|--:|:--|"]
    for m in MODELS:
        for tag in TAGS:
            for part in ("cpu", "slow", "cpu_g5b", "slow_g5b"):
                d = os.path.join(R2, "q2", f"{m}_{tag}_{part}")
                rows = latest_rows(d)
                if not rows:
                    md.append(f"| {m} | {tag}{'（G5b）' if 'g5b' in part else ''} | {part.split('_')[0]} | NOT_MEASURED |  |  |  |  |  |  |  |")
                    continue
                tlab = f"{tag}（G5b）" if "g5b" in part else tag
                ok = usable(d, rows)
                for bw in sorted({r["bw_name"] for r in rows}, key=lambda x: ("cpu", "b16", "b32", "b48").index(x)):
                    for st in ("idle", "busy"):
                        for cf in ("E0", "E1", "R25"):
                            g = med(rows, mode="cake", bw_name=bw, gpu_state=st, config=cf)
                            c = med(rows, mode="claim", bw_name=bw, gpu_state=st, config=cf)
                            la = med(rows, mode="loadall", bw_name=bw, gpu_state=st, config=cf)
                            best = min(x for x in (c, la) if x is not None) if (c or la) else None
                            gr = [r for r in rows if (r["mode"], r["bw_name"], r["gpu_state"], r["config"]) == ("cake", bw, st, cf)]
                            frac = sum(r["meet_rule_used"] == "none" for r in gr) / len(gr) if gr else None
                            out.setdefault("q2", {})[f"{m}|{tlab}|{bw}|{st}|{cf}"] = dict(g5=g, claim=c, loadall=la, frac_none=frac,
                                                                                        usable=ok)
                            md.append(f"| {m} | {tlab} | {bw} | {'閒' if st == 'idle' else '忙'} | {cf} | {_f(g)} | {_f(c)} | {_f(la)} | "
                                      f"{_f(g / best if g and best else None, 3)} | {_f(frac, 2)} | {'是' if ok else '**否**'} |")
    gp = os.path.join(R2, "g5_pick.json")
    if os.path.exists(gp):
        g = json.load(open(gp))
        md += ["", f"G5 對 G5b（13 §8c；{g['n_common']} 個共同格子的「÷ 最好」幾何平均）：G5 {_f(g.get('gm_g5'), 3)}、G5b {_f(g.get('gm_g5b'), 3)} → 44d 用 **{g['pick']}**"]
    md.append("")


def report_q3(md, out):
    md += ["## Q3：新格式的融合 kernel（13 §5）", "",
           "真實 CPU 層，32K；中位數（s）。原寫法＝43 的速度格子（`results/m10_sweep_mi300x/confirm/speed_<m>/summary_<格式>.csv`）。", "",
           "| 模型 | 格式 | 配置 | 閒：原寫法 → 融合 | 忙：原寫法 → 融合 | 可用 |", "|:--|:--|:--|:--|:--|:--|"]
    for d in sorted(glob.glob(os.path.join(R2, "q3", "speed_*"))):
        m, f = os.path.basename(d)[len("speed_"):].split("_", 1)
        sp = glob.glob(os.path.join(d, "summary_*.csv"))
        if not sp:
            continue
        new = {(r["gpu_state"], r["config"]): float(r["median_t_restore"]) for r in csv.DictReader(open(sp[0]))
               if r["bw_name"] == "cpu" and r["mode"] == "cake" and r["length"] == "32768"}
        op = os.path.join(SWROOT, "confirm", f"speed_{m}", f"summary_{f}.csv")
        old = {(r["gpu_state"], r["config"]): float(r["median_t_restore"]) for r in csv.DictReader(open(op))
               if r["bw_name"] == "cpu" and r["mode"] == "cake" and r["length"] == "32768"} if os.path.exists(op) else {}
        guards = [json.load(open(g)) for g in glob.glob(os.path.join(d, "gpu_guard_*.json"))]
        ok = bool(guards) and all(g.get("verdict") == "CLEAN" for g in guards)
        for cf in sorted({c for _, c in new}):
            out.setdefault("q3", {})[f"{m}|{f}|{cf}"] = dict(new={s_: new.get((s_, cf)) for s_ in ("idle", "busy")},
                                                            old={s_: old.get((s_, cf)) for s_ in ("idle", "busy")}, usable=ok)
            md.append(f"| {m} | {f} | {cf} | {_f(old.get(('idle', cf)))} → {_f(new.get(('idle', cf)))} | "
                      f"{_f(old.get(('busy', cf)))} → {_f(new.get(('busy', cf)))} | {'是' if ok else '**否**'} |")
    md.append("")


def report_q5(md, out):
    md += ["## Q5：拆融合進 attention（對照組；13 §6）", "", "### (a) 只搬不拆的還原（E1，中位數 s）", "",
           "| 模型 | 格式 | 頻寬 | 閒：只搬不拆／全載入（P0） | 忙：只搬不拆／全載入（P0） | 可用 |", "|:--|:--|:--|:--|:--|:--|"]
    for m in MODELS:
        for tag in TAGS:
            d = os.path.join(R2, "q5", f"nodq_{m}_{tag}")
            rows = latest_rows(d)
            p0 = latest_rows(os.path.join(R2, "q4", f"{m}_{tag}_P0"))
            if not rows:
                continue
            for bw in ("cpu", "b32"):
                cells = []
                for st in ("idle", "busy"):
                    a = med(rows, mode="cake", bw_name=bw, gpu_state=st, config="E1")
                    b = med(p0, mode="loadall", bw_name=bw, gpu_state=st, config="E1") if p0 else None
                    out.setdefault("q5a", {})[f"{m}|{tag}|{bw}|{st}"] = dict(nodq=a, loadall_P0=b)
                    cells.append(f"{_f(a)}／{_f(b)}")
                md.append(f"| {m} | {tag} | {bw} | {cells[0]} | {cells[1]} | {'是' if usable(d, rows) else '**否**'} |")
    md += ["", "### (b) attention 微測（中位數 ms，所有層合計；同一個 Triton kernel 讀 bf16／fp8／int4_kivi；sdpa＝PyTorch SDPA 讀 BF16；ext＝vLLM-KIVI，只量時間）", ""]
    ap = glob.glob(os.path.join(R2, "q5", "attn", "attn_*.csv"))
    for p in sorted(ap):
        rows = [r for r in csv.DictReader(open(p)) if r["median_s"] not in ("", "ERROR")]
        modes = [x for x in ("sdpa_bf16", "bf16", "fp8", "int4_kivi", "ext_vllm_kivi") if any(r["mode"] == x for r in rows)]
        md += [f"#### {os.path.basename(p)}", "", "| context | query | GPU | " + " | ".join(modes) + " | 可用 |",
               "|:--|:--|:--|" + "--:|" * len(modes) + ":--|"]
        for L in sorted({int(r["length"]) for r in rows}):
            for M in sorted({int(r["queries"]) for r in rows}, reverse=True):
                for st in ("idle", "busy"):
                    v = {r["mode"]: float(r["median_s"]) for r in rows
                         if int(r["length"]) == L and int(r["queries"]) == M and r["gpu_state"] == st}
                    cont = {r["contaminated_gpu"] for r in rows if int(r["length"]) == L and r["gpu_state"] == st}
                    if v:
                        md.append(f"| {L} | {M} | {'閒' if st == 'idle' else '忙'} | " + " | ".join(_f(1e3 * v[x], 2) if x in v else "—" for x in modes)
                                  + f" | {'是' if cont == {'False'} else '**否**'} |")
        md.append("")
    md += ["### (c) 組合（〔算術〕，用 (a)(b) 的實測）", "",
           "每生成 1 個 token 多花 Δ＝decode（1 個 query）讀壓縮 − 讀 BF16（同一個 kernel）；還原省下＝全載入（P0）− 只搬不拆；"
           "划算的 token 數＝還原省下 ÷ Δ（Δ ≤ 0 寫「一直划算」）。32K、CPU 層。", "",
           "| 模型 | 格式 | GPU | 還原省下（ms） | Δ（ms／token） | 划算的 token 數 |", "|:--|:--|:--|--:|--:|--:|"]
    for m in MODELS:
        ap_ = os.path.join(R2, "q5", "attn", f"attn_{m}.csv")
        if not os.path.exists(ap_):
            continue
        rows = [r for r in csv.DictReader(open(ap_)) if r["median_s"] not in ("", "ERROR")]
        for tag, mode in (("fp8", "fp8"), ("int4", "int4_kivi")):
            for st in ("idle", "busy"):
                v = {r["mode"]: float(r["median_s"]) for r in rows if r["length"] == "32768" and r["queries"] == "1" and r["gpu_state"] == st}
                x = out.get("q5a", {}).get(f"{m}|{tag}|cpu|{st}", {})
                if mode in v and "bf16" in v and x.get("nodq") and x.get("loadall_P0"):
                    save = x["loadall_P0"] - x["nodq"]
                    dlt = v[mode] - v["bf16"]
                    be = "一直划算" if dlt <= 0 else f"{save / dlt:.0f}"
                    md.append(f"| {m} | {tag} | {'閒' if st == 'idle' else '忙'} | {1e3 * save:.1f} | {1e3 * dlt:.3f} | {be} |")
    md += ["", "### (d) vLLM 0.28 ROCm：BF16 KV vs FP8 KV（attention 裡拆）", "",
           "| 模型 | KV | context | prefill＋1（s） | 每個 token 的 decode（ms） |", "|:--|:--|:--|--:|--:|"]
    mp = os.path.join(R2, "q5", "vllm_kvdecode", "vllm_kvdecode_meta.json")
    if os.path.exists(mp):
        for k, v in json.load(open(mp)).items():
            m, kd = k.split("|")
            for L, r in v["result"].items():
                md.append(f"| {m} | {kd} | {L} | {r['median_t1_s']:.3f} | {1e3 * r['decode_per_token_s']:.2f} |")
    md.append("")


def report_q1(md, out):
    md += ["## Q1：長 context（13 §7；2 launch × 5 rep）", "", "中位數（s）；cake／loadall（G5 那一列是 g5／claim／loadall）。", "",
           "| 模型 | 長度 | 格式 | 方法 | 頻寬 | GPU | E0 | E1 | R25 | 可用 |", "|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for d in sorted(glob.glob(os.path.join(R2, "q1", "*"))):
        rows = latest_rows(d)
        if not rows:
            continue
        m, L, tag, *meth = os.path.basename(d).split("_")
        ok = usable(d, rows)
        for bw in sorted({r["bw_name"] for r in rows}):
            for st in ("idle", "busy"):
                cells = []
                for cf in ("E0", "E1", "R25"):
                    vs = [med(rows, mode=mo, bw_name=bw, gpu_state=st, config=cf) for mo in ("cake", "claim", "loadall")]
                    vs = [_f(v) for v in vs if v is not None]
                    cells.append("／".join(vs) if vs else "—")
                out.setdefault("q1", {})[f"{os.path.basename(d)}|{bw}|{st}"] = cells
                md.append(f"| {m} | {L} | {tag} | {'_'.join(meth)} | {bw} | {'閒' if st == 'idle' else '忙'} | " + " | ".join(cells)
                          + f" | {'是' if ok else '**否**'} |")
    md.append("")


def cmd_report(a):
    out, md = {}, ["# 13 號：還原的第二輪（只報告；docs/precision_20261013/13_restore2_plan.md）", "",
                   "所有表都是 32K、真實 CPU 層，除非另寫；「可用」＝gpu_guard 全 CLEAN、沒有污染列（CLAUDE.md §3）。", ""]
    for f in (report_q4, report_q2, report_q3, report_q5, report_q1):
        try:
            f(md, out)
        except Exception as ex:                                 # 某一段還沒跑或格式不對：照實寫，不擋其他段
            md += [f"（{f.__name__} 出錯：{ex!r}）", ""]
    os.makedirs(R2, exist_ok=True)
    json.dump(out, open(os.path.join(R2, "report_13.json"), "w"), indent=1, ensure_ascii=False, default=str)
    open(os.path.join(R2, "tables_13.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("cumask")
    s.add_argument("--out", required=True)
    s = sub.add_parser("env")
    s.add_argument("method", choices=list(Q4))
    s.add_argument("--model", required=True)
    s.add_argument("--fmt", required=True)
    s = sub.add_parser("smoke-check")
    s.add_argument("--dir", default=os.path.join(R2, "smoke"))
    sub.add_parser("pick-q4")
    sub.add_parser("pick-c")
    sub.add_parser("pick-g5")
    s = sub.add_parser("plan-44d")
    s.add_argument("--jobs", required=True)
    sub.add_parser("report")
    a = ap.parse_args()
    {"cumask": cmd_cumask, "env": cmd_env, "smoke-check": cmd_smoke_check, "pick-q4": cmd_pick_q4,
     "pick-c": cmd_pick_c, "pick-g5": cmd_pick_g5, "plan-44d": cmd_plan_44d, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
