"""m10_restore2.py — 13 號計畫（還原的第二輪；docs/precision_20261013/13_restore2_plan.md）的工具。只報告。

  cumask --out J        GPU（43a 冒煙）：CU 數 → P4a／P4b 的 HSA_CU_MASK；背景 matmul 單獨跑的時間（有／沒有遮罩）驗證遮罩有生效
  env <方法> --model m --fmt f
                        印出 Q4 方法（P0–P5）的環境變數（K* 與 S 組由 m10_dqsweep 給；P4 的遮罩讀 cumask.json）
  smoke-check --dir D   43a 冒煙：KV 逐位元、meet_rule_used、nodq、背景吞吐、新格式 kernel → D/OK_restore
  pick-q4               13 §3.2：帶到長 context 的方法 → q4_pick.json
  pick-c                13 §7：42c 的冠軍（CPU 層 E1 中位數對 8 格的幾何平均最小）→ c_pick.json
  plan-44d --jobs J     13 §7：寫 J/queue/44d_rs_longctx.sh
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
R2 = os.path.join(REPO, "results", "m10_restore2_mi300x")
SW = os.path.join(REPO, "results", "m10_sweep_mi300x", "dqsweep")
MODELS = ("llama", "qwen")
TAGS = ("fp8", "int4")
# 13 §3.1：Q4 的方法＝（S 組代號, 另外的環境變數）；"@p4a" 等由 cumask.json 換成實際遮罩
Q4 = {"P0": ("S0", {}), "P1": ("S0", {"M10_RS_PRIO": "1"}), "P2": ("S2k8", {}), "P3": ("S4", {}),
      "P4a": ("S0", {"M10_BUSY_CU_MASK": "@p4a"}), "P4b": ("S0", {"M10_BUSY_CU_MASK": "@p4b"}),
      "P5": ("S2k8", {"M10_RS_PRIO": "1"})}
MASKED = ("P4a", "P4b")


def fmt_of(m, tag):
    return json.load(open(os.path.join(REPO, "results", "m10_precision_mi300x", "fmt_select", "fmt_choice.json")))["choice"][m][tag]


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
        chk(bool(bs) and all(x.get("n_matmul", 0) > 0 and x.get("cu_mask") for x in bs), f"{name}：背景吞吐有記錄、遮罩有傳 {bs[:1]}")
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
    json.dump(out, open(os.path.join(R2, "c_pick.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if pick is None:
        raise SystemExit("42c：沒有完整的可用資料")


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
    runs.append(("G5", "S0", {"M10_MEET": "g5"}, "cpu b32", "loadall claim"))
    runs.append(("nodq", "S0", {"M10_MEET": "none", "M10_DQ_SCHED": "nodq"}, "cpu", ""))
    lines = [f"# 13 §7（Q1）長 context（由 m10_restore2.py plan-44d 寫，{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}）：",
             f"#   42c 冠軍 {c}、Q4 帶過來的 {q}；方法 {' '.join(r[0] for r in runs)}",
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
    s = sub.add_parser("plan-44d")
    s.add_argument("--jobs", required=True)
    a = ap.parse_args()
    {"cumask": cmd_cumask, "env": cmd_env, "smoke-check": cmd_smoke_check, "pick-q4": cmd_pick_q4,
     "pick-c": cmd_pick_c, "plan-44d": cmd_plan_44d}[a.cmd](a)


if __name__ == "__main__":
    main()
