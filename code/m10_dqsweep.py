"""m10_dqsweep.py — 09 §6.4 反量化的方法掃描：怎麼轉（K）× 何時轉／藏延遲（S）× 切多細（G）。只報告，不改主線判定。

  list B            印出海選（B 段）的方法代號
  env <方法> --model m --fmt f
                    印出這個方法的環境變數（shell：`export $(python m10_dqsweep.py env S1 --model qwen --fmt int4_kivi)`）；
                    K* 從 A 段的 choice_<m>.json 讀，沒有就用 k1
  kernels --model m --out-dir D
                    A 段：K 組 kernel 微測（寫到整個 KV 緩衝的視圖，和還原時一樣），GPU 閒／忙，挑 K*
  smoke --model m --fmts f ...
                    冒煙（stage 39）：真模型，每個方法（含 S×G 組合的樣本）E0／E1／R25 各還原一次，KV 逐位元檢查
  plan --sw D --jobs J
                    B 段跑完：照 09 §6.4.3 的分數挑複賽選手，寫 J/queue/42c_dqsweep_confirm.sh 與 D/plan_C.json
  report --sw D     C 段跑完（CPU）：寫 D/C/report_C.json、D/C/tables_C.md（09 §6.4.3-3 的報告項目；只報告）

方法的定義（09 §6.4.1）：S0 逐塊照順序（現在）；S1 邊搬邊拆；S2k<k> 多個 block 一組；S3 重算優先；S4 最後一次拆；
S5 會合後才拆；G1 256 token；G2 128 token；G4 預估會合。組合用「+」：例 S1+G4。
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SW_DEFAULT = os.path.join(REPO, "results", "m10_sweep_mi300x", "dqsweep")

METHODS = {
    "S0": {},
    "S1": {"M10_DQ_SCHED": "pipe"},
    "S2k2": {"M10_DQ_SCHED": "group", "M10_DQ_GROUP": "2"},
    "S2k4": {"M10_DQ_SCHED": "group", "M10_DQ_GROUP": "4"},
    "S2k8": {"M10_DQ_SCHED": "group", "M10_DQ_GROUP": "8"},
    "S2k16": {"M10_DQ_SCHED": "group", "M10_DQ_GROUP": "16"},
    "S3": {"M10_DQ_PRIO": "1"},
    "S4": {"M10_DQ_SCHED": "defer"},
    "S5": {"M10_DQ_SCHED": "defer_meet"},
    "G1": {"M10_UNIT": "256"},
    "G2": {"M10_UNIT": "128"},
    "G4": {"M10_MEET": "estimate"},
}
S_GROUP = ["S1", "S2k2", "S2k4", "S2k8", "S2k16", "S3", "S4", "S5"]
G_GROUP = ["G1", "G2", "G4"]
B_METHODS = ["S0"] + S_GROUP + G_GROUP
FMTS_A = ["fp8_token", "fp8_head", "int8_token", "int4_token", "int4_kivi"]
K3_GRID = [f"k3:{r}:{w}" for r in (4, 8, 16, 32, 64) for w in (1, 2, 4, 8)]
SHAPES = {"llama": (32, 2, 8, 512, 128), "qwen": (28, 2, 4, 512, 128)}
CONFIGS_C = ["E0", "E1", "E1s", "E1sr", "F25", "M25", "B25", "R25"]


def kstar(sw: str, model: str, fmt: str) -> str:
    p = os.path.join(sw, "A", f"choice_{model}.json")
    if not os.path.exists(p):
        return "k1"
    return json.load(open(p))["choice"].get(fmt, "k1")


def method_env(name: str, model: str, fmt: str, sw: str = SW_DEFAULT) -> dict:
    env = {"M10_DEQUANT_FAST": "1", "M10_DQ_KERNEL": kstar(sw, model, fmt)}
    for part in name.split("+"):
        env.update(METHODS[part])
    return env


def _opts(env: dict) -> dict:
    return dict(unit=int(env.get("M10_UNIT", "512")), sched=env.get("M10_DQ_SCHED", "serial"),
                group=int(env.get("M10_DQ_GROUP", "1")), compute_prio=env.get("M10_DQ_PRIO") == "1",
                meet=env.get("M10_MEET", "claim"))


# ---------------------------------------------------------------------- A 段：kernel 微測
def cmd_kernels(a):
    import torch
    import m10_codec as C
    import m10_dequant_fast as DF
    from m10_speed import Busy, _watcher, run_id, ts
    os.makedirs(a.out_dir, exist_ok=True)
    rid = run_id()
    L, two, H, T, D = SHAPES[a.model]
    tmax = 64 * 512 + 512
    kv = torch.zeros((L, two, H, tmax, D), dtype=torch.bfloat16, device="cuda")
    lo, hi = 63 * 512, 64 * 512
    view = kv[:, :, :, lo:hi]
    torch.manual_seed(0)
    x = (torch.randn(SHAPES[a.model], device="cuda") * 2).to(torch.bfloat16)
    x[:, 0, :, :, 5] += 40                                   # K 的離群通道
    variants = ["k0", "k1", "k2", "k4"] + K3_GRID
    duty = json.load(open(os.path.join(REPO, "results", "m7_write_policy_mi300x", "busy_levels.json")))["busy2"]
    watch, level = _watcher(os.path.join(a.out_dir, f"gpu_guard_{rid}_kernels_{a.model}.json"))
    busy_pid = set()
    encs, wants, fns, bit = {}, {}, {}, {}
    for fmt in FMTS_A:
        e = C.encode(x, fmt)
        encs[fmt], wants[fmt] = e, C.decode(e)
        for k in variants:
            if k == "k4" and not fmt.startswith("fp8"):
                continue
            fn = (lambda e=e: C.decode(e, out=view)) if k == "k0" else (lambda e=e, k=k: DF.decode_variant(e, view, k))
            try:
                kv.zero_()
                fn()
                torch.cuda.synchronize()
                ok = torch.equal(view, wants[fmt]) and not bool(kv[:, :, :, :lo].any()) and not bool(kv[:, :, :, hi:].any())
            except Exception as ex:                         # noqa: BLE001  照實記（例：torch.compile 在 ROCm 上不能用）
                print(f"{fmt} {k}: 執行失敗 {ex!r}"[:300], flush=True)
                bit[(fmt, k)] = f"ERROR {type(ex).__name__}"
                continue
            bit[(fmt, k)] = ok
            fns[(fmt, k)] = fn
            print(f"{fmt} {k}: 逐位元 {'相同' if ok else '不同'}", flush=True)
    rows = []
    for st in ("idle", "busy"):
        with Busy(0.0 if st == "idle" else duty, watch) as busy:
            if busy.p:
                busy_pid.add(busy.p.pid)
            for (fmt, k), fn in fns.items():
                for _ in range(5):
                    fn()
                torch.cuda.synchronize()
                tt = []
                for _ in range(50):
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    fn()
                    e1.record()
                    e1.synchronize()
                    tt.append(e0.elapsed_time(e1))
                tt.sort()
                rows.append(dict(run_id=rid, ts=ts(), model=a.model, fmt=fmt, kernel=k, gpu_state=st,
                                 median_ms=round(tt[25], 5), p10_ms=round(tt[5], 5), p90_ms=round(tt[45], 5),
                                 bitwise=bit[(fmt, k)], host_level=level))
            if busy.p:
                busy_pid.discard(busy.p.pid)
    contaminated = None
    if watch is not None:
        bad = {p: v for p, v in watch.intruders.items() if p not in busy_pid}
        contaminated = bool(bad) or watch.started_clean is False
        watch.__exit__(None, None, None)
    for r in rows:
        r["contaminated_gpu"] = contaminated if contaminated is not None else "NOT_CHECKED"
    p = os.path.join(a.out_dir, f"kernels_{a.model}.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # 挑 K*（09 §6.4.3-1）：逐位元相同的 k1／k3／k4 裡，閒＋忙的中位數最小；差不到 3% 照 k1 → k3 → k4 取前面的
    med = {(r["fmt"], r["kernel"], r["gpu_state"]): r["median_ms"] for r in rows}
    choice, scores = {}, {}
    for fmt in FMTS_A:
        cand = {k: med[(fmt, k, "idle")] + med[(fmt, k, "busy")] for k in ["k1", "k4"] + K3_GRID
                if bit.get((fmt, k)) is True and (fmt, k, "idle") in med}
        scores[fmt] = {k: round(v, 5) for k, v in sorted(cand.items(), key=lambda kv_: kv_[1])}
        best = min(cand.values())
        for cls in (["k1"], K3_GRID, ["k4"]):
            ok = {k: v for k, v in cand.items() if k in cls and v <= best * 1.03}
            if ok:
                choice[fmt] = min(ok, key=ok.get)
                break
    out = dict(run_id=rid, model=a.model, rule="09 §6.4.3-1：逐位元相同的 k1／k3／k4 裡閒＋忙中位數最小；差 <3% 照 k1→k3→k4",
               choice=({} if contaminated else choice),          # 污染 → 時間不能用，B／C 退回 k1（kstar 讀不到就用 k1）
               choice_if_clean=choice, scores=scores, contaminated=contaminated,
               bitwise={f"{f}|{k}": v for (f, k), v in bit.items()},
               k0_ms={f: [med.get((f, "k0", s)) for s in ("idle", "busy")] for f in FMTS_A},
               k2_ms={f: [med.get((f, "k2", s)) for s in ("idle", "busy")] for f in FMTS_A})
    json.dump(out, open(os.path.join(a.out_dir, f"choice_{a.model}.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({"choice": choice, "contaminated": contaminated}, ensure_ascii=False), flush=True)
    if contaminated:
        raise SystemExit("GPU 監控報污染：時間不能用（CLAUDE.md §3），要重量")


# ---------------------------------------------------------------------- 冒煙（stage 39）
def cmd_smoke(a):
    import torch
    import m7_restore_harness as RH
    import m10_dequant_fast as DF
    from m10_speed import Bench, layout
    names = B_METHODS + ["S1+G4", "S2k4+G1", "S4+G2", "S3+G4"]
    names.sort(key=lambda nm: -_opts(method_env(nm, a.model, "int4_kivi", a.sw))["unit"])   # 同一個單位的放一起，少重做寫入
    ok, t0 = True, time.time()
    for fmt in a.fmts:
        bench = Bench(a.model, 64, fmt, 256, unit=512)
        cur = None
        for name in names:
            env = method_env(name, a.model, fmt, a.sw)
            o = _opts(env)
            if o["unit"] != cur:
                bench.unit, bench.k = o["unit"], 512 // o["unit"]
                bench.capture(64)
                cur = o["unit"]
            DF.install(env["M10_DQ_KERNEL"])
            bench.rs = RH.Restorer(bench.km)
            bench.rs.time_h2d = True
            bench.rs.set_options(**o)
            for cfg in ("E0", "E1", "R25"):
                for mode in ("cake",) + (("load_only", "compute_only") if o["unit"] != 512 else ()):
                    rec = bench.restore(layout(cfg, 64, 7), {"kind": "cpu", "name": "cpu"}, mode, 64)
                    bad = bench.kv_bad(rec, 64)
                    srcs = sorted(i for i, _, _, _ in rec["blocks"])
                    once = srcs == list(range(64 * bench.k))
                    good = bad == 0 and once
                    ok &= good
                    print(("PASS" if good else "FAIL"), a.model, fmt, name, cfg, mode, f"meet={rec['meet']} kv_bad={bad}",
                          f"t={rec['t_restore'] * 1e3:.1f}ms dq={rec['t_dequant'] * 1e3:.1f}ms", flush=True)
        del bench
        torch.cuda.empty_cache()
    DF.install("k1")
    print(f"耗時 {time.time() - t0:.0f}s；" + ("ALL PASS" if ok else "SOME FAILED"), flush=True)
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------- B → C
def _latest_rows(d: str) -> list:
    p = os.path.join(d, "restores.csv")
    if not os.path.exists(p):
        return []
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return []
    last = max(r["run_id"] for r in rows)
    return [r for r in rows if r["run_id"] == last]


def _gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def cmd_plan(a):
    sw, jobs = a.sw, a.jobs
    problems, med = [], {}
    for m in ("llama", "qwen"):
        for meth in B_METHODS:
            rows = [r for r in _latest_rows(os.path.join(sw, "B", f"{m}_{meth}")) if r["mode"] == "cake"]
            if not rows:
                problems.append(f"{m} {meth}：沒有資料")
                continue
            if any(r["kv_bad"] not in ("", "0") for r in rows):
                problems.append(f"{m} {meth}：KV 逐位元檢查沒過 → 淘汰")
                continue
            if any(r["contaminated_gpu"] != "False" for r in rows):
                problems.append(f"{m} {meth}：GPU 監控報污染（時間不能用，要重量）")
                continue
            for st in ("idle", "busy"):
                for cfg in ("E0", "E1"):
                    v = [float(r["t_restore"]) for r in rows if r["gpu_state"] == st and r["config"] == cfg]
                    if v:
                        med[(m, meth, st, cfg)] = statistics.median(v)
    score, detail = {}, {}
    for meth in B_METHODS:
        per_model = []
        for m in ("llama", "qwen"):
            rs_ = []
            for st in ("idle", "busy"):
                e0s = [v for (mm, _, s, c), v in med.items() if mm == m and s == st and c == "E0"]
                if (m, meth, st, "E1") in med and e0s:
                    rs_.append(med[(m, meth, st, "E1")] / min(e0s))
            if len(rs_) == 2:
                per_model.append(_gm(rs_))
                detail[f"{m}|{meth}"] = [round(x, 4) for x in rs_]
        if len(per_model) == 2:
            score[meth] = _gm(per_model)
    ranked = sorted((k for k in score if k != "S0"), key=score.get)
    top3 = ranked[:3]
    bs = min((k for k in score if k in S_GROUP), key=score.get, default=None)
    bg = min((k for k in score if k in G_GROUP), key=score.get, default=None)
    combo = f"{bs}+{bg}" if bs and bg else None
    finalists = top3 + ([combo] if combo else ranked[3:4]) + ["S0"]
    finalists = list(dict.fromkeys(finalists))
    champ = top3[0] if top3 else None
    plan = dict(rule="09 §6.4.3-2：分數＝各模型、各 GPU 狀態「方法的 E1 中位數 ÷ 所有方法最快的 E0 中位數」取幾何平均；"
                     "前 3 名＋S 最好×G 最好＋S0", score={k: round(v, 4) for k, v in sorted(score.items(), key=lambda x: x[1])},
                detail=detail, finalists=finalists, champion=champ, problems=problems)
    os.makedirs(sw, exist_ok=True)
    json.dump(plan, open(os.path.join(sw, "plan_C.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(plan, ensure_ascii=False, indent=1), flush=True)
    if problems:
        open(os.path.join(jobs, "NEEDS_HUMAN"), "a").write("dqsweep B：" + "；".join(problems) + "\n")
    if not champ:
        raise SystemExit("沒有可用的方法，不寫 42c")
    lines = [f"# 09 §6.4.3-3 複賽（由 m10_dqsweep.py plan 寫，{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}）："
             f"選手 {' '.join(finalists)}；冠軍 {champ} 和 S0 另加 b32（E0 E1 R25）。只算／只載的參考線不跑（09 §6.4.4）",
             f"source {jobs}/lib.sh", "set -uo pipefail", f"SW={sw}", "mkdir -p $SW/C"]
    for m in ("llama", "qwen"):
        lines.append(f"G=$(fmt_of {m} fp8)")
        for fmt_expr, tag in (("$G", "fp8"), ("int4_kivi", "int4")):
            for meth in finalists:
                mm = meth.replace("+", "_")
                runs = [("cpu", " ".join(CONFIGS_C), "")]
                if meth in (champ, "S0"):
                    runs.append(("b32", "E0 E1 R25", "_b32"))
                for bw, cfgs, suf in runs:
                    d = f"$SW/C/{m}_{tag}_{mm}{suf}"
                    cmd = (f"python $REPO/code/m10_speed.py grid --model {m} --fmt {fmt_expr} --base-fmt $G --out-dir {d} "
                           f"--bw-json $OUT/speed_{m}/bw.json --bw-names {bw} --configs {cfgs} --refs --lengths "
                           f"--launches 3 --reps 10 --launch0 200 --blocks-out \\$TIARA_RUNS/\\$RUN_ID/blocks.csv")
                    lines.append(f"( export $(python3 $REPO/code/m10_dqsweep.py env {meth} --model {m} --fmt {fmt_expr}); "
                                 f"m7run m10-dqs-C-{m}-{tag}-{mm}{suf} bash -c \"{cmd}\" ) "
                                 f"|| echo \"dqsweep C {m} {tag} {meth}{suf} 失敗\" >> $JOBS/NEEDS_HUMAN")
    lines.append("exit 0")
    open(os.path.join(jobs, "queue", "42c_dqsweep_confirm.sh"), "w").write("\n".join(lines) + "\n")
    print("寫了", os.path.join(jobs, "queue", "42c_dqsweep_confirm.sh"), flush=True)


# 並排用（09 §6.4.3-3）：主線（K0、S0、G0；33a／20／22）與 42（K1、S0、G0）的速度格子
REF_SUMMARY = {
    ("llama", "fp8"): ("m10_precision_mi300x/speed_llama/summary_fp8.csv", "m10_sweep_mi300x/speed_dqfast_llama/summary.csv"),
    ("llama", "int4"): ("m10_precision_mi300x/speed_llama/summary_int4.csv", "m10_sweep_mi300x/speed_dqfast_int4_llama/summary.csv"),
    ("qwen", "fp8"): ("m10_precision_mi300x/speed_qwen_regrid/summary_fp8.csv", "m10_sweep_mi300x/speed_dqfast_qwen/summary.csv"),
    ("qwen", "int4"): ("m10_precision_mi300x/speed_qwen/summary_int4.csv", "m10_sweep_mi300x/speed_dqfast_int4_qwen/summary.csv"),
}
MIXED = ["F25", "M25", "B25", "R25"]


def _ref_med(path, bw, st, cfg="E1", length="32768"):
    p = os.path.join(REPO, "results", path)
    if not os.path.exists(p):
        return None
    for r in csv.DictReader(open(p)):
        if (r["length"], r["bw_name"], r["gpu_state"], r["config"], r["mode"]) == (length, bw, st, cfg, "cake"):
            return float(r["median_t_restore"])
    return None


def cmd_report(a):
    """C 段跑完：每個方法 E1 對 E0 的配對差與 CI、「速度不輸」比值、反量化／重算／等待時間、重算幾塊；和主線、42 並排。只報告。"""
    from collections import Counter
    from m10_speed import bootstrap_ci
    sw = a.sw
    plan = json.load(open(os.path.join(sw, "plan_C.json")))
    fins, champ = plan["finalists"], plan["champion"]
    out = dict(rule="09 §6.4.3-3（只報告）：配對＝同一個（launch, rep, 頻寬, GPU 狀態）的 E0、E1；差＝E0−E1（正＝E1 快），"
                    "bootstrap 95% CI（m10_speed.bootstrap_ci）；速度不輸比值＝E1 中位數 ÷ min(F25,M25,B25,R25 中位數)，"
                    "≤ 1＋δ* 算不輸（03 §2.4；δ* 是主線凍結的 noise.json）；gpu_guard 不是 CLEAN 或有污染列 → 那個 run 的時間不能用",
               finalists=fins, champion=champ, cells=[], problems=[])
    for m in ("llama", "qwen"):
        dstar = json.load(open(os.path.join(REPO, "results", "m10_precision_mi300x", f"speed_{m}", "noise.json")))["delta_star"]
        for tag in ("fp8", "int4"):
            for meth in fins:
                for suf in ("", "_b32"):
                    if suf and meth not in (champ, "S0"):
                        continue
                    name = f"{m}_{tag}_{meth.replace('+', '_')}{suf}"
                    d = os.path.join(sw, "C", name)
                    rows = [r for r in _latest_rows(d) if r["mode"] == "cake" and r["length"] == "32768"]
                    if not rows:
                        out["problems"].append(f"{name}：NOT_MEASURED（沒有資料）")
                        continue
                    rid = rows[0]["run_id"]
                    guards = [json.load(open(g)) for g in glob.glob(os.path.join(d, f"gpu_guard_{rid}_*.json"))]
                    dirty = [g.get("verdict") for g in guards if g.get("verdict") != "CLEAN"]
                    n_cont = sum(r["contaminated_gpu"] != "False" for r in rows)
                    n_bad = sum(r["kv_bad"] not in ("", "0") for r in rows)
                    usable = bool(guards) and not dirty and n_cont == 0
                    if not usable:
                        out["problems"].append(f"{name}：時間不能用（guard {dirty or '缺'}、污染列 {n_cont}）→ 要重量")
                    if n_bad:
                        out["problems"].append(f"{name}：KV 逐位元檢查有 {n_bad} 列沒過 → 方法淘汰")
                    for bw, st in sorted({(r["bw_name"], r["gpu_state"]) for r in rows}):
                        v = [r for r in rows if (r["bw_name"], r["gpu_state"]) == (bw, st)]
                        med = {c: statistics.median(float(r["t_restore"]) for r in v if r["config"] == c)
                               for c in sorted({r["config"] for r in v})}
                        e0 = {(r["launch"], r["rep"]): float(r["t_restore"]) for r in v if r["config"] == "E0"}
                        e1 = {(r["launch"], r["rep"]): float(r["t_restore"]) for r in v if r["config"] == "E1"}
                        keys = sorted(set(e0) & set(e1))
                        diffs = [e0[k] - e1[k] for k in keys]
                        lo, hi = bootstrap_ci(diffs) if len(diffs) >= 2 else (None, None)
                        x1 = [r for r in v if r["config"] == "E1"]
                        mixed = [med[c] for c in MIXED if c in med]
                        ratio = med["E1"] / min(mixed) if mixed and "E1" in med else None
                        ref33, ref42 = (_ref_med(p, bw, st) for p in REF_SUMMARY[(m, tag)])
                        out["cells"].append(dict(
                            model=m, fmt=tag, fmt_name=rows[0]["fmt"], method=meth, bw=bw, gpu_state=st, run_id=rid,
                            usable=usable, kv_bad_rows=n_bad, delta_star=dstar, n_pairs=len(keys),
                            median={c: round(x, 5) for c, x in med.items()},
                            gain_E1_over_E0=round((med["E0"] - med["E1"]) / med["E0"], 4),
                            paired_diff_ci95_s=[lo, hi],
                            not_worse_ratio=round(ratio, 4) if ratio else None,
                            not_worse=(ratio <= 1 + dstar) if ratio else None,
                            E1_t_dequant=round(statistics.median(float(r["t_dequant"]) for r in x1), 5),
                            E1_t_recompute=round(statistics.median(float(r["t_recompute"]) for r in x1), 5),
                            E1_t_wait=round(statistics.median(float(r["t_wait"]) for r in x1), 5),
                            b_E0=dict(Counter(int(float(r["b"])) for r in v if r["config"] == "E0")),
                            b_E1=dict(Counter(int(float(r["b"])) for r in x1)),
                            waiting_line_E1=dict(Counter(r["waiting_line"] for r in x1)),
                            ref_E1_mainline=ref33, ref_E1_stage42=ref42))
    json.dump(out, open(os.path.join(sw, "C", "report_C.json"), "w"), indent=1, ensure_ascii=False)
    ST = {"idle": "閒", "busy": "忙"}
    md = ["# 反量化方法掃描：複賽（09 §6.4.3-3；只報告）", "",
          f"選手 {'、'.join(fins)}；冠軍 {champ}。還原時間中位數（s），32K；配對差＝E0−E1（正＝E1 快）。"
          "「不輸比」＝E1 ÷ 混合配置最快的，≤ 1＋δ* 算不輸。主線＝原寫法（K0、S0、G0），42＝融合 kernel（K1、S0、G0）。", ""]
    for m in ("llama", "qwen"):
        for tag in ("fp8", "int4"):
            cs = [c for c in out["cells"] if c["model"] == m and c["fmt"] == tag]
            if not cs:
                continue
            md += [f"## {m}・{cs[0]['fmt_name']}（δ*＝{100 * cs[0]['delta_star']:.2f}%）", "",
                   "| 頻寬 | GPU | 方法 | E0 | E1 | E1 快多少 | 配對差 95% CI | 不輸比 | 反量化／重算／等待（E1） | 重算塊數 E0→E1 | E1：主線／42 | 可用 |",
                   "|:--|:--|:--|--:|--:|--:|:--|:--|:--|:--|:--|:--|"]
            for c in sorted(cs, key=lambda c: (c["bw"] != "cpu", c["bw"], c["gpu_state"] != "idle", fins.index(c["method"]))):
                lo, hi = c["paired_diff_ci95_s"]
                ci = f"[{lo:+.4f}, {hi:+.4f}]" if lo is not None else "—"
                nw = "—" if c["not_worse_ratio"] is None else f"{c['not_worse_ratio']:.3f}（{'過' if c['not_worse'] else '沒過'}）"
                bb = lambda d: "/".join(f"{k}×{n}" for k, n in sorted(d.items()))
                ref = "／".join("—" if x is None else f"{x:.3f}" for x in (c["ref_E1_mainline"], c["ref_E1_stage42"]))
                md.append(f"| {c['bw']} | {ST[c['gpu_state']]} | {c['method']} | {c['median']['E0']:.3f} | {c['median']['E1']:.3f} | "
                          f"{100 * c['gain_E1_over_E0']:+.1f}% | {ci} | {nw} | {1e3 * c['E1_t_dequant']:.1f}／"
                          f"{1e3 * c['E1_t_recompute']:.1f}／{1e3 * c['E1_t_wait']:.1f} ms | {bb(c['b_E0'])} → {bb(c['b_E1'])} | "
                          f"{ref} | {'是' if c['usable'] else '**否**'} |")
            md.append("")
    md += ["## 問題", ""] + ([f"- {p}" for p in out["problems"]] or ["無"])
    open(os.path.join(sw, "C", "tables_C.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list")
    s.add_argument("which", choices=["B"])
    s = sub.add_parser("env")
    s.add_argument("method")
    s.add_argument("--model", required=True)
    s.add_argument("--fmt", required=True)
    s.add_argument("--sw", default=SW_DEFAULT)
    s = sub.add_parser("kernels")
    s.add_argument("--model", choices=list(SHAPES), required=True)
    s.add_argument("--out-dir", required=True)
    s = sub.add_parser("smoke")
    s.add_argument("--model", choices=list(SHAPES), required=True)
    s.add_argument("--fmts", nargs="+", required=True)
    s.add_argument("--sw", default=SW_DEFAULT)
    s = sub.add_parser("plan")
    s.add_argument("--sw", default=SW_DEFAULT)
    s.add_argument("--jobs", required=True)
    s = sub.add_parser("report")
    s.add_argument("--sw", default=SW_DEFAULT)
    a = ap.parse_args()
    if a.cmd == "list":
        print(" ".join(B_METHODS))
    elif a.cmd == "env":
        print(" ".join(f"{k}={v}" for k, v in method_env(a.method, a.model, a.fmt, a.sw).items()))
    else:
        sys.path.insert(0, HERE)
        {"kernels": cmd_kernels, "smoke": cmd_smoke, "plan": cmd_plan, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
