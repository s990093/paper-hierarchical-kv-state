"""x5_run.py — X5 的格子：trace × 模型 × CPU GiB × 成本設定（校正 variant × 反量化 × GPU × 慢層）× 策略。

用法（CPU only）：
  CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" RUN_ID=<id> python -I code/p3x/x5_run.py \
      --out-dir results/p3_explore/x5 --raw-dir /ssd8/hungwei/paper-hkv/runs/<id> --procs 8 [--mapped]
產出：
  <out>/x5_trace_meta.csv、x5_cost_constants.json、x5_cost_validation.csv、x5_cells[_mapped].csv、x5_oracle_actions[_mapped].csv
  <raw>/ttft[_mapped].npz（每格每個策略的回來請求 TTFT，給 bootstrap 用）
每個 CSV 都有 run_id、ts。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x5_costs as XC  # noqa: E402
import x5_sim as XS  # noqa: E402
import x5_trace as XT  # noqa: E402

GiB = 1 << 30
TRACES = ("conversation", "toolagent")
MODELS = ("llama", "qwen")
CPU_GIB = (128, 256, 512, 1024)
VARIANTS = ("reg", "excess")
NONCOST = ("R1", "R2", "R3", "R4", "R5", "OA", "OB", "OC", "R2b", "R4b")
COST = ("R6", "OD", "ODn", "R6b")
RID = os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-p3x5-adhoc"
_C = {}


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


RULE = os.environ.get("X5_RULE", "subtree")


def get_trace(name, mapped):
    key = ("tr", name, mapped)
    if key not in _C:
        if name.startswith("freeinference"):
            import x5_fi
            tr = x5_fi.load()
        else:
            tr = XT.load(name, RULE)
        if mapped:
            tr = XT.map_lengths(tr)
        _C[key] = tr
    return _C[key]


def get_costs(model, variant):
    key = ("c", model, variant)
    if key not in _C:
        _C[key] = XC.Costs(model, variant)
    return _C[key]


def get_tabs(model, variant, gpu, slow, dq, svals):
    key = ("tab", model, variant, gpu, slow, dq, tuple(svals))
    if key not in _C:
        c = get_costs(model, variant)
        _C[key] = {st: {v: c.table(st, v, gpu, slow, dq) for v in svals} for st in XC.STATES}
    return _C[key]


def get_prefill(name, mapped, model, variant, gpu):
    key = ("pre", name, mapped, model, gpu)
    if key not in _C:
        tr = get_trace(name, mapped)
        c = get_costs(model, variant)
        _C[key] = np.array([c.prefill(int(k), int(L), gpu) for k, L in zip(tr["keff"], tr["L"])])
    return _C[key]


def settings(variants, dqs):
    for v in variants:
        for dq in dqs:
            for g in XC.GPUS:
                for sl in XC.SLOW:
                    yield v, dq, g, sl


def nonret_sum(tr, tabs, pre):
    """非回來請求（新 session）的 TTFT 總和：共用前綴從 CPU-BF16 還原＋prefill；所有策略相同。"""
    m = ~tr["ret"] & (tr["sid"] >= 0)
    k = tr["keff"][m]
    sv = tr["s"][m]
    rest = np.array([tabs["BF16"][int(a)][int(b)] for a, b in zip(sv, k)])
    return float((rest + pre[m]).sum())


def row_common(name, model, C, v, dq, g, sl, pol, tr, c):
    ws = (tr["meta"]["n_distinct_blocks"] or (tr["sess_ws_blocks"] + tr["meta"]["n_shared_blocks"])) * c.bytes["BF16"]
    sess_ws = tr["sess_ws_blocks"] * c.bytes["BF16"]
    return dict(run_id=RID, ts=ts(), trace=name, model=model, cpu_GiB=C, cpu_pct_ws=100 * C * GiB / ws,
                cpu_pct_sess_ws=100 * C * GiB / sess_ws, variant=v, dq=dq, gpu=g, slow=sl,
                slow_GiBps=c.B[sl], policy=pol)


def task(args):
    kind, name, model, C, pol, mapped, sets = args
    t0 = time.time()
    tr = get_trace(name, mapped)
    svals = sorted(set(int(x) for x in tr["s"]))
    out_rows, arrays, actrows = [], {}, []
    if kind == "noncost":
        c0 = get_costs(model, "reg")
        res = XS.simulate(tr, c0.bytes["BF16"], c0.bytes["INT4"], tr["meta"]["n_shared_blocks"], C * GiB, pol)
        sim_list = [(s, res) for s in sets]
    else:
        sim_list = []
        for s in sets:
            v, dq, g, sl = s
            c = get_costs(model, v)
            tabs = get_tabs(model, v, g, sl, dq, svals)
            res = XS.simulate(tr, c.bytes["BF16"], c.bytes["INT4"], tr["meta"]["n_shared_blocks"], C * GiB, pol, tabs)
            sim_list.append((s, res))
    for (v, dq, g, sl), res in sim_list:
        c = get_costs(model, v)
        tabs = get_tabs(model, v, g, sl, dq, svals)
        pre = get_prefill(name, mapped, model, v, g)
        e = XS.evaluate(tr, res["state_at"], tabs, pre)
        ttft = e.pop("ttft")
        row = row_common(name, model, C, v, dq, g, sl, pol, tr, c)
        row.update(e)
        row["sum_all_ttft"] = e["sum_ret_ttft"] + nonret_sum(tr, tabs, pre)
        row.update({f"n_{k}": vv for k, vv in res["acts"].items()})
        out_rows.append(row)
        arrays[f"{name}|{model}|{C}|{v}|{dq}|{g}|{sl}|{pol}"] = ttft.astype(np.float32)
        if pol in ("OD", "OB"):
            for (act, st, b), cnt in res["act_bins"].items():
                actrows.append(dict(row_common(name, model, C, v, dq, g, sl, pol, tr, c), action=act, from_state=st,
                                    time_to_return=b, count=cnt))
    return out_rows, arrays, actrows, time.time() - t0, args[:5]


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


CENSOR_H = float(os.environ.get("X5_CENSOR_H", "0"))


def apply_censor(tr, H):
    """穩健性：t_c＝最後一個請求的時間 − H；只評估上一輪在 t_c 之前的回來請求；
    t_c 之後才最後出現的 session，oracle／部分 oracle 當作在 T_end＋H 回來（不准丟）。"""
    if H <= 0 or "eval_mask" in tr:
        return tr
    Tend = float(tr["T"][-1])
    tc = Tend - H
    prevT = np.where(tr["prev"] >= 0, tr["T"][np.maximum(tr["prev"], 0)], np.inf)
    tr["eval_mask"] = prevT <= tc
    fake = np.full(len(tr["T"]), np.nan)
    late_last = (tr["nxt"] < 0) & (tr["T"] > tc) & (tr["sid"] >= 0)
    fake[late_last] = Tend + H
    tr["fake_nextT"] = fake
    tr["meta"] = dict(tr["meta"], censor_H=H, censor_tc=tc, n_eval_returns=int((tr["ret"] & tr["eval_mask"]).sum()),
                      n_late_last_sessions=int(late_last.sum()))
    return tr


def prepare_trace(name, mapped):
    tr = get_trace(name, mapped)
    apply_censor(tr, CENSOR_H)
    # session 工作集：每個 session 最大的私有 block 數總和
    mx = np.zeros(tr["n_sess"], dtype=np.int64)
    m = tr["sid"] >= 0
    np.maximum.at(mx, tr["sid"][m], (tr["nb"] - tr["s"])[m])
    tr["sess_ws_blocks"] = int(mx.sum())
    return tr


def init_worker(mapped):
    for name in TRACES:
        prepare_trace(name, mapped)


def init_worker_cfg(mapped, rule, traces, nmax, censor_h=0.0, models=None):
    global RULE, TRACES, CENSOR_H
    RULE, TRACES, CENSOR_H = rule, traces, censor_h
    XC.NMAX = nmax
    init_worker(mapped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--mapped", action="store_true", help="只報告：長度映射到 8K–32K（只跑 Llama、reg）")
    ap.add_argument("--rule", default="subtree", choices=["subtree", "strict"], help="Mooncake 的 session 重建規則")
    ap.add_argument("--fi", action="store_true", help="FreeInference（事後補充；只跑 reg）")
    ap.add_argument("--variants", default="reg,excess")
    ap.add_argument("--censor-H", type=float, default=0.0, help="穩健性（事後加）：只評估距離 trace 結束 ≥H 秒的決定")
    ap.add_argument("--main-only", action="store_true", help="只跑主設定（llama、k0）")
    a = ap.parse_args()
    global RULE, TRACES, CENSOR_H
    CENSOR_H = a.censor_H
    RULE = a.rule
    os.environ["X5_RULE"] = a.rule
    if a.fi:
        TRACES = ("freeinference",)
        XC.NMAX = 2500
    a.variants = tuple(a.variants.split(","))
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.raw_dir, exist_ok=True)
    sfx = (("_mapped" if a.mapped else "") + ("_strict" if a.rule == "strict" else "") + ("_fi" if a.fi else "")
           + (f"_cens{int(a.censor_H)}" if a.censor_H > 0 else "") + ("_loadonly" if a.variants == ("loadonly",) else ""))
    models = ("llama",) if (a.mapped or a.main_only) else MODELS
    variants = ("reg",) if a.mapped else a.variants
    # ---- trace 特性＋兩個 trace 重疊的檢查
    meta_rows = []
    for name in TRACES:
        tr = prepare_trace(name, a.mapped)
        mt = dict(run_id=RID, ts=ts(), mapped=a.mapped, **{k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                                                           for k, v in tr["meta"].items()},
                  sess_ws_blocks=tr["sess_ws_blocks"])
        meta_rows.append(mt)
    if not a.mapped and not a.fi:
        import collections
        rows = {nm: [json.loads(l) for l in open(os.path.join(XT.TRACE_DIR, f"{nm}_trace.jsonl"))] for nm in TRACES}
        k = {nm: collections.Counter((r["timestamp"], r["input_length"], r["output_length"]) for r in rows[nm]) for nm in TRACES}
        ov = sum((k["conversation"] & k["toolagent"]).values())
        for mt in meta_rows:
            mt["overlap_rows_same_ts_in_out_with_other_trace"] = ov
    write_csv(os.path.join(a.out_dir, f"x5_trace_meta{sfx}.csv"), meta_rows)
    # ---- 成本常數與驗證（只在主 run 寫）
    if not a.mapped and not a.fi and a.rule == "subtree":
        consts, val = {"run_id": RID, "ts": ts()}, []
        for m in MODELS:
            for v in VARIANTS:
                c = get_costs(m, v)
                if v == "reg":
                    consts[m] = c.constants()
                    consts[m]["slow_gate"] = c.slow_gate()
                    if not c.slow_gate()["passed"]:
                        print(f"[x5] 慢層推算沒過 gate：{c.slow_gate()}", file=sys.stderr)
                        sys.exit(3)
                for r in c.validation:
                    row = dict(run_id=RID, ts=ts(), variant=v, **r)
                    if v == "excess":
                        # excess 的「用的值」：用 restore() 重新算（含差的校正）
                        st = "SLOW" if r["kind"] == "slow" and r["state"] == "INT4" else r["state"]
                        if r["kind"] == "slow" and r["state"] == "BF16":
                            continue
                        p = c.restore(st, 0, r["n_blocks"], r["gpu"], r["tier"] if r["kind"] == "slow" else "b16", r["dq"])
                        row["pred_used_s"], row["err_used"] = p, (p - r["meas_s"]) / r["meas_s"]
                    val.append(row)
        json.dump(consts, open(os.path.join(a.out_dir, "x5_cost_constants.json"), "w"), indent=1, ensure_ascii=False)
        write_csv(os.path.join(a.out_dir, "x5_cost_validation.csv"), val)
    # ---- 任務
    sets_all = list(settings(variants, ("k0",) if a.main_only else XC.DQS))
    tasks = []
    for name in TRACES:
        for m in models:
            for C in CPU_GIB:
                for pol in NONCOST:
                    tasks.append(("noncost", name, m, C, pol, a.mapped, sets_all))
                for pol in COST:
                    for s in sets_all:
                        tasks.append(("cost", name, m, C, pol, a.mapped, [s]))
    print(f"[x5] {len(tasks)} tasks, procs={a.procs}", flush=True)
    rows, arrays, acts = [], {}, []
    t0 = time.time()
    with Pool(a.procs, initializer=init_worker_cfg, initargs=(a.mapped, a.rule, TRACES, XC.NMAX, CENSOR_H)) as pool:
        for i, (rr, ar, ac, dt, key) in enumerate(pool.imap_unordered(task, tasks, chunksize=2)):
            rows += rr
            arrays.update(ar)
            acts += ac
            if i % 50 == 0:
                print(f"[x5] {i + 1}/{len(tasks)} {key} {dt:.1f}s elapsed {time.time() - t0:.0f}s", flush=True)
    rows.sort(key=lambda r: (r["trace"], r["model"], r["cpu_GiB"], r["variant"], r["dq"], r["gpu"], r["slow"], r["policy"]))
    write_csv(os.path.join(a.out_dir, f"x5_cells{sfx}.csv"), rows)
    write_csv(os.path.join(a.out_dir, f"x5_oracle_actions{sfx}.csv"), acts)
    np.savez_compressed(os.path.join(a.raw_dir, f"ttft{sfx}.npz"), **arrays)
    print(f"[x5] done {len(rows)} rows in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
