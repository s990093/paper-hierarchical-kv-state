"""p11_tools.py — ICPE P1-1（docs/h200/icpe/P1-1_xkv_int4_h200.md）的 CPU 工具。

  plan      §2：寫 m16_xstack gen 的 --selection 檔（P1-1 的配置）＋核對新題 seed 67000 不和既有題組重疊 → p11/p11_selection.json
  report-q  §2：每個配置對 E0、對 q:int4_kivi 的配對差、ε、MLPerf 99%；校準 vs 每題 SVD（XLCAL 規則）；和 MI300X X4b 的相容 → p11/p11q_report.json、tables_p11q.md
  report-t  §3：慢層 8 格的「低秩＋INT4 比 INT4 快」判定 → p11/p11t_report.json、tables_p11t.md
每個輸出都有 run_id、ts。不 import torch。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
REPO = os.path.dirname(CODE)
sys.path.insert(0, CODE)
import m16_xstack as XS  # noqa: E402

P11 = os.path.join(REPO, "results", "icpe_h200", "p11")
SEED0, N, TASKS = 67000, 48, ("T2", "T3", "T4", "T5")
CONFIGS = {   # §2 的表（順序＝每題 SVD 的配置排在一起，給 --xl-cache）
    "qwen": [("q:int4_kivi", "ref_int4"), ("xl:k34:v51:cal+q:int4_kivi", "main"), ("xl:k34:v51+q:int4_kivi", "x4b_same")],
    "llama": [("q:int4_kivi", "ref_int4"), ("xl:k29:v44:cal+q:int4_kivi", "main"), ("xl:k19:v29:cal+q:int4_kivi", "main"),
              ("xl:k29:v44+q:int4_kivi", "x4b_same"), ("xl:k19:v29+q:int4_kivi", "x4b_same")],
}
PAIRS = {"qwen": [("xl:k34:v51:cal+q:int4_kivi", "xl:k34:v51+q:int4_kivi")],
         "llama": [("xl:k29:v44:cal+q:int4_kivi", "xl:k29:v44+q:int4_kivi"), ("xl:k19:v29:cal+q:int4_kivi", "xl:k19:v29+q:int4_kivi")]}
USED = {"x1": (XS.X1_SEED0, 24), "calib": (XS.CALIB_SEED0, 2), "x2": (XS.X2_SEED0, 24), "x2b": (XS.X2B_SEED0, XS.X2B_N),
        "x4a": (64000, 24), "x4b": (65000, 48), "xlcal3": (66000, 48), "c1a": (70000, 45), "lc128k": (XS.LC_SEED0, XS.LC_N)}
X4_REPORT = os.path.join(REPO, "results", "m16_stage3_mi300x", "x_stack", "x4", "x4_report.json")


def run_id():
    return os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc"


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def cmd_plan(a):
    new = {(t, s) for t, _, _, s in XS.plan_interleaved_cpu(list(TASKS), N, SEED0)}
    ov = {}
    for k, (s0, n) in USED.items():
        used = {(t, s) for t, _, _, s in XS.plan_interleaved_cpu(list(TASKS), n, s0)}
        ov[k] = len(new & used)
    if any(ov.values()):
        raise SystemExit(f"新題 seed {SEED0} 和既有題組重疊：{ov}（規則 2：停）")
    models = {}
    for m, cfgs in CONFIGS.items():
        cf = []
        for spec, role in cfgs:
            st = XS.validate(XS.parse_spec(spec), XS.MODEL_INFO[m], XS.N_BLOCKS - 1)
            if XS.canonical(st) != spec:
                raise XS.SpecError(f"{spec} 不是標準寫法（{XS.canonical(st)}）")
            cf.append(dict(spec=spec, roles=[role], tiers=["core"], budgets=[], ratio_region=XS.predict_ratio(spec, m),
                           why={"main": "P1-1 主配置（校準基底）", "x4b_same": "和 MI300X X4b 同一個配置（每題 SVD）",
                                "ref_int4": "可部署的強對照"}[role]))
        models[m] = dict(configs=cf)
    sel = dict(go=True, which="icpe_p11", reason="ICPE P1-1：xKV＋INT4 在 H200 重現（docs/h200/icpe/P1-1_xkv_int4_h200.md §2）",
               tasks=list(TASKS), seed0=SEED0, n=N, pairs={m: [list(p) for p in v] for m, v in PAIRS.items()}, models=models,
               seed_overlap=ov, run_id=run_id(), ts=ts())
    os.makedirs(P11, exist_ok=True)
    p = os.path.join(P11, "p11_selection.json")
    if os.path.exists(p):
        old = json.load(open(p))
        keys = ("go", "which", "tasks", "seed0", "n", "pairs", "models")
        if {k: old.get(k) for k in keys} != {k: sel.get(k) for k in keys}:
            raise SystemExit(f"{p} 已存在而內容不同 → 停（不默默蓋掉）")
        print(f"{p} 已存在、內容相同 → 不重寫")
        return
    json.dump(sel, open(p, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({m: [(c["spec"], round(1 / c["ratio_region"], 2)) for c in v["configs"]] for m, v in models.items()}, ensure_ascii=False),
          "seed 重疊：", ov)


def _mi300x_x4b():
    """MI300X X4b（32K）每個配置對 E0 的差（x4_report.json），找不到就 None。"""
    if not os.path.exists(X4_REPORT):
        return {}
    r = json.load(open(X4_REPORT))
    out = {}

    def walk(o, model=None):
        if isinstance(o, dict):
            if "spec" in o and "diff" in o and isinstance(o.get("diff"), dict) and model:
                out.setdefault((model, o["spec"]), o["diff"])
            for k, v in o.items():
                walk(v, k if k in ("llama", "qwen") else model)
        elif isinstance(o, list):
            for v in o:
                walk(v, model)
    walk(r.get("by_ctx", {}).get("32K", r))
    return out


def cmd_report_q(a):
    rows, partial, files, _ = XS._gen_rows(a.raw)
    mi = _mi300x_x4b()
    rep = dict(run_id=run_id(), ts=ts(), files=files, incomplete_items=partial, src_run_ids=sorted({r["run_id"] for r in rows}),
               eps=XS.EPS, rule_xlcal=XS.XLCAL_TEXT["rule"], models={})
    L = ["# P1-1 (Q)：xKV＋INT4 在 H200 的生成品質（192 題、seed 67000；判定事先寫死，見 docs/h200/icpe/P1-1_xkv_int4_h200.md §2）", ""]
    for m in CONFIGS:
        specs = [s for s, _ in CONFIGS[m]]
        by = {}
        for r in rows:
            if r["model"] == m and r["spec"] in set(specs) | {"E0"}:
                by.setdefault(r["item_id"], {})[r["spec"]] = r
        items = sorted(i for i, d in by.items() if "E0" in d and all(s in d for s in specs))
        M = dict(n_items=len(items), configs={}, pairs={})
        e0 = [float(by[i]["E0"]["gen_score"]) for i in items]
        i4 = [float(by[i]["q:int4_kivi"]["gen_score"]) for i in items]
        acc0 = statistics.mean(e0) if items else None
        for s, role in CONFIGS[m]:
            sc = [float(by[i][s]["gen_score"]) for i in items]
            d = XS.boot2([x - y for x, y in zip(sc, e0)]) if items else None
            d4 = XS.boot2([x - y for x, y in zip(sc, i4)]) if items and s != "q:int4_kivi" else None
            acc = statistics.mean(sc) if items else None
            c = dict(role=role, acc=acc, acc_e0=acc0, diff_vs_e0=d, diff_vs_int4=d4, ratio=float(by[items[0]][s]["ratio_region"]) if items else None,
                     eps_pass=bool(d and d["lo"] >= -XS.EPS), mlperf99=bool(items and acc >= 0.99 * acc0))
            o = mi.get((m, s))
            if o is not None and d is not None:
                c["mi300x_x4b_diff"] = o
                c["compatible_with_mi300x"] = bool(d["lo"] <= o["hi"] and o["lo"] <= d["hi"])
            M["configs"][s] = c
        vs = []
        for cal, ref in PAIRS[m]:
            d = XS.boot2([float(by[i][cal]["gen_score"]) - float(by[i][ref]["gen_score"]) for i in items]) if items else None
            v = XS.xlcal_verdict(d) if d else "INCOMPLETE"
            M["pairs"][f"{cal} vs {ref}"] = dict(cal_minus_ref=d, verdict=v)
            vs.append(v)
        M["xlcal_aggregate"] = "INCOMPLETE" if len(items) < XS.MIN_ITEMS else XS.xlcal_aggregate(vs)
        main = [s for s, r in CONFIGS[m] if r == "main"]
        if m == "qwen":
            M["Q_a_qwen_8x_near_lossless"] = bool(M["configs"][main[0]]["mlperf99"]) if items else None
        rep["models"][m] = M
        L += [f"## {m}（E0 答對率 {None if acc0 is None else round(100 * acc0, 1)}%，{len(items)} 題）", "",
              "| 配置 | 角色 | 倍數 | 答對率 | 對 E0（百分點，CI） | 對 INT4 | ε | MLPerf 99% | MI300X X4b 對 E0 | 相容 |",
              "|:--|:--|--:|--:|--:|--:|:--|:--|--:|:--|"]
        for s, c in M["configs"].items():
            if c["acc"] is None:
                continue
            L.append(f"| {s} | {c['role']} | {1 / c['ratio']:.2f} | {100 * c['acc']:.1f} | {XS._ci(c['diff_vs_e0'], 100, 1)} | "
                     f"{XS._ci(c['diff_vs_int4'], 100, 1) if c['diff_vs_int4'] else '—'} | {'過' if c['eps_pass'] else '沒過'} | "
                     f"{'過' if c['mlperf99'] else '沒過'} | {XS._ci(c['mi300x_x4b_diff'], 100, 1) if c.get('mi300x_x4b_diff') else '—'} | "
                     f"{({True: '相容', False: '不相容'}).get(c.get('compatible_with_mi300x'), '—')} |")
        L += ["", "校準基底 − 每題 SVD（XLCAL 規則）：" + "；".join(f"{k}：{XS._ci(e['cal_minus_ref'], 100, 1) if e['cal_minus_ref'] else 'NA'} {e['verdict']}"
                                                     for k, e in M["pairs"].items()) + f"；模型判定 {M['xlcal_aggregate']}", ""]
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(rep, open(os.path.join(a.out_dir, "p11q_report.json"), "w"), ensure_ascii=False, indent=1, default=str)
    open(os.path.join(a.out_dir, "tables_p11q.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def cmd_report_t(a):
    """§3：慢層 8 格（模型 × {b32, nfs} × {閒, 忙}）；INT4 端到端＝P0-1 (b) sdpa 組的 S4 的 min(Cake, 全部載入)；xKV＝loadpath。"""
    import pandas as pd
    lp = pd.read_csv(os.path.join(a.time_dir, "loadpath.csv"))
    rows, problems = [], []
    if (lp.contaminated.astype(str) == "True").any():
        problems.append("loadpath 有污染列（不進判定）")
        lp = lp[lp.contaminated.astype(str) != "True"]
    for m in ("llama", "qwen"):
        g = pd.read_csv(os.path.join(a.p01_grid, f"{m}_s4_sdpa", "restores.csv"))
        g = g[(g.config == "E1") & g["mode"].isin(["cake", "loadall"])]
        for tier in ("b32", "nfs", "cpu"):
            for duty, st in ((0.0, "idle"), (1.0, "busy")):
                q = g[(g.bw_name == tier) & (g.gpu_state == st)]
                ti4 = min(q[q["mode"] == "cake"].t_restore.median(), q[q["mode"] == "loadall"].t_restore.median()) if len(q) else None
                x = lp[(lp.model == m) & (lp.tier == tier) & (lp.duty.astype(float) == duty)]
                lp4 = x[x.seg == "int4_kivi"].meas_s.astype(float)
                tlp4 = float(lp4.iloc[0]) if len(lp4) else None      # 只報告：INT4 用同一種量法（全部載入的載入線）
                for _, r in x.iterrows():
                    rows.append(dict(run_id=run_id(), ts=ts(), model=m, tier=tier, gpu=st, seg=r.seg, cal=bool(r.cal), t_meas_s=float(r.meas_s),
                                     t_int4_e2e_s=ti4, speedup_vs_int4=(ti4 / float(r.meas_s)) if ti4 else None,
                                     faster=bool(ti4 and float(r.meas_s) < 0.95 * ti4), bytes_lb_s=float(r.bytes_lb_s) if "bytes_lb_s" in r else None,
                                     t_int4_lp_s=tlp4, speedup_vs_int4_lp=(tlp4 / float(r.meas_s)) if tlp4 else None,
                                     judged=tier in ("b32", "nfs") and str(r.seg).startswith("xkv_")))   # int4_kivi 本身是對照，不進判定
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(a.out_dir, "p11t_cells.csv"), index=False)
    out = dict(run_id=run_id(), ts=ts(), problems=problems, rule="P1-1 §3：每格 :cal 版全部載入 < INT4 端到端最好 ×0.95；慢層 8 格都成立＝成立")
    for cal in (True, False):
        J = T[T.judged & (T.cal == cal)]                     # judged 只含 xkv_* 的段（每格三個秩組合都要快）
        key = "cal" if cal else "per_item"
        per = J.groupby(["model", "tier", "gpu"]).faster.all() if len(J) else None
        out[key] = dict(n_cells=int(len(per)) if per is not None else 0, n_faster=int(per.sum()) if per is not None else 0,
                        holds=bool(per is not None and len(per) == 8 and per.all()))
    json.dump(out, open(os.path.join(a.out_dir, "p11t_report.json"), "w"), ensure_ascii=False, indent=1, default=str)
    print(T.drop(columns=["run_id", "ts"]).round(4).to_string(index=False))
    print(json.dumps(out, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("plan")
    q = sp.add_parser("report-q")
    q.add_argument("--raw", nargs="+", required=True)
    q.add_argument("--out-dir", default=P11)
    t = sp.add_parser("report-t")
    t.add_argument("--time-dir", default=os.path.join(P11, "time"))
    t.add_argument("--p01-grid", default=os.path.join(REPO, "results", "icpe_h200", "p01", "grid"))
    t.add_argument("--out-dir", default=P11)
    a = ap.parse_args()
    {"plan": cmd_plan, "report-q": cmd_report_q, "report-t": cmd_report_t}[a.cmd](a)


if __name__ == "__main__":
    main()
