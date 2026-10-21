"""m16_xlcal3.py — xKV 事先校準的基底 vs 每題 SVD：加題數（10/21 第三輪；38k*）。

38c（v2/tables_xlcal.md）兩個模型都 UNDETERMINED（96 題）。使用者 10/21：加題數把它判出來，**只比有 INT4 的兩對**
（xl:k22:v34:cal+q:int4_kivi vs xl:k22:v34+q:int4_kivi、xl:k28:v42:cal+q:int4_kivi vs xl:k28:v42+q:int4_kivi；只降秩那對不做）＋E0。
新題 seed 66000 起、T2–T5 各 48 題（192 題）、32K，同一題同一行程配對；GPU 這段直接用 m16_xstack.py gen（不改它）＋--calib-xl＋--xl-cache；
校準基底沿用 38c 算好的 calib_xl_<m>.pt（不重算；check-calib 核對 sha256＝38c 的摘要）。

判定照 m16_xstack.XLCAL_TEXT 原文（rule）；只把 aggregate 的「三對」改成「這兩對」（XLCAL3_TEXT['aggregate']，理由寫明）。
報告並列「只用 38c 的 96 題」「只用新的 192 題」「合併 288 題」；**主要判定讀新的 192 題**（10/21 改：兩對的限制是看過 38c 才定的，只有新題是獨立檢驗），合併只當參考。輸出 results/m16_stage3_mi300x/x_stack/v3/（不覆蓋 v2）。

子命令（全部 CPU、不 import torch）：
  plan          寫 v3/xlcal3_selection.json（m16_xstack gen 的 --selection 格式；已有而內容不同 → 停）
  check-calib   38c 的校準檔 sha256＝v2/calib_xl_<m>.json 記的 → v3/calib_xl_sha.json；不符或不在 → exit 1
  report        三種並列的報告 → v3/xlcal3_report.json、v3/tables_xlcal3.md
  --self-test
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m16_xstack as XS  # noqa: E402

OUT_V3 = os.path.join(XS.OUT_DIR, "v3")
SEED0, N_PER_TASK, TASKS = 66000, 48, ("T2", "T3", "T4", "T5")      # 新題：T2–T5 各 48 題＝192 題
PAIRS = (("xl:k22:v34:cal+q:int4_kivi", "xl:k22:v34+q:int4_kivi"), ("xl:k28:v42:cal+q:int4_kivi", "xl:k28:v42+q:int4_kivi"))
CALIB_PT = os.path.join(XS.DATA, "calib_xl_{m}.pt")
CALIB_V2 = os.path.join(XS.OUT_DIR_V2, "calib_xl_{m}.json")
# 不能重疊的題（seed0, 每任務題數, 任務）：X1、PCA／xKV 校準、X2（＝38c 的生成題）、X2b、X4a、X4b、C1a（X4d 用）、44c 128K 與 Qwen 256K
USED = {"x1": (XS.X1_SEED0, 24, TASKS), "calib": (XS.CALIB_SEED0, 2, TASKS), "x2": (XS.X2_SEED0, 24, TASKS),
        "x2b": (XS.X2B_SEED0, XS.X2B_N, TASKS), "x4a": (64000, 24, TASKS), "x4b": (65000, 48, TASKS),
        "c1a": (70000, 45, ("T2", "T3", "T4")), "c1a_t5": (70000, 15, ("T5",)), "lc128k": (XS.LC_SEED0, XS.LC_N, TASKS)}
VARIANTS = (("old96", "只用 38c 的 96 題（X2 的題）"), ("new192", "只用新的 192 題（seed 66000 起；主要判定）"), ("merged288", "合併 288 題（參考：含已看過的 96 題）"))

XLCAL3_TEXT = {
    "rule": XS.XLCAL_TEXT["rule"],
    "aggregate": "（10/21 改動：使用者指定只比有 INT4 的兩對，只降秩那對不做）每個模型：**這兩對**都 CAN_PRECOMPUTE → CAN_PRECOMPUTE；"
                 "任一對 MUST_PER_ITEM → MUST_PER_ITEM；其他 → UNDETERMINED；題數 < 48 → INCOMPLETE。整體：兩個模型都 CAN → CAN；"
                 "任一模型 MUST → MUST；其他照實寫。原文（XLCAL_TEXT['aggregate']）是三對；判定門檻（ε＝2 個百分點、CI 上下界的規則）不變。",
    "variants": "並列三種題組：只用 38c 的 96 題、只用新的 192 題、合併 288 題（每題四個配置＋E0 都有、同一題同一行程配對；兩組題的 seed 不重疊）。"
                "主要判定讀新的 192 題（獨立檢驗）；合併 288 題只當參考（含看過結果之後才限定兩對的 96 題）。10/21 在看到新資料之前改定。",
    "caveat": "「只比 INT4 的兩對」是看到 38c 的結果之後決定的（使用者 10/21）。38c 的 96 題在兩對的 aggregate 下，Llama 會從 UNDETERMINED 變成"
              " CAN_PRECOMPUTE（差別全在拿掉的只降秩那對：−3.7 [−7.6, +0.2]）；所以合併 288 題含了已經看過的 96 題，"
              "**「只用新的 192 題」才是事先寫死、沒看過資料的獨立檢驗**——兩者不一致時照實並列，判讀由人。",
    "design": "新題 seed 66000 起、T2–T5 各 48 題（和 X1／X2／X2b／X4a／X4b／校準／C1a（X4d）／44c 都不重疊，self-test 核對）；"
              "GPU 用 m16_xstack.py gen（和 38c 同一條路徑：每題 SVD 是每個配置各做一次，--xl-cache 在這兩對沒有可共用的 SVD、等於沒作用）；"
              "校準基底沿用 38c 的 calib_xl_<m>.pt（sha256 見 check-calib）。",
    "bytes": XS.XLCAL_TEXT["bytes"],
}


# ============================================================================ plan
def build_plan() -> dict:
    models = {}
    for m in XS.MODEL_INFO:
        cf = []
        for c, ref in PAIRS:
            for s, role in ((c, "xlcal"), (ref, "xlcal_ref")):
                st = XS.validate(XS.parse_spec(s), XS.MODEL_INFO[m], XS.N_BLOCKS - 1)
                if XS.canonical(st) != s:
                    raise XS.SpecError(f"{s} 不是標準寫法")
                cf.append(dict(spec=s, roles=[role], tiers=["core"], budgets=[],
                               why="校準基底版" if role == "xlcal" else "每題 SVD 的同名配置（同一題、同一個行程配對）"))
        models[m] = dict(configs=cf)
    return dict(go=True, which="xlcal3", reason="xKV 校準基底 vs 每題 SVD，只比 INT4 的兩對（使用者 10/21）", tasks=list(TASKS),
                seed0=SEED0, n=N_PER_TASK, pairs=[list(p) for p in PAIRS], models=models)


PLAN_KEYS = ("go", "which", "tasks", "seed0", "n", "pairs", "models")


def cmd_plan(a):
    p = build_plan()
    if os.path.exists(a.out):
        old = json.load(open(a.out))
        if {k: old.get(k) for k in PLAN_KEYS} == {k: p.get(k) for k in PLAN_KEYS}:
            print(f"{a.out} 已有、內容相同（run {old.get('run_id')}）→ 不重寫", flush=True)
            return
        raise SystemExit(f"{a.out} 已存在但內容不同 → 不默默改事先寫死的計畫")
    p.update(run_id=os.environ.get("RUN_ID", "adhoc"), ts=XS.ts())
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(p, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"寫好 {a.out}：" + "；".join(f"{m} {len(v['configs'])} 個配置" for m, v in p["models"].items()), flush=True)


# ============================================================================ check-calib
def calib_status(pt: str, v2json: str) -> dict:
    """校準檔的 sha256（現在算）和 38c 的摘要（v2/calib_xl_<m>.json）記的比；檔或摘要不在 → ok＝False（規則 7）。"""
    e = dict(file=pt, summary=v2json)
    if not os.path.exists(pt) or not os.path.exists(v2json):
        return dict(e, ok=False, why="檔案或 38c 的摘要不在")
    s = json.load(open(v2json))
    e.update(sha256=XS.sha256(pt), expected=s.get("sha256"), calib_run_id=s.get("run_id"), n_tokens=s.get("n_tokens"), R=s.get("R"))
    e["ok"] = bool(e["sha256"] == e["expected"])
    if not e["ok"]:
        e["why"] = "sha256 和 38c 的摘要不同（校準檔被重算或換過）"
    return e


def cmd_check_calib(a):
    res = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=XS.ts(),
               models={m: calib_status(a.calib.format(m=m), a.v2_summary.format(m=m)) for m in XS.MODEL_INFO})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    for m, e in res["models"].items():
        print(f"{m}：{'OK' if e['ok'] else 'NG'} sha256 {e.get('sha256')}（38c 記的 {e.get('expected')}；calib run {e.get('calib_run_id')}）"
              + ("" if e["ok"] else f" — {e['why']}"), flush=True)
    if not all(e["ok"] for e in res["models"].values()):
        raise SystemExit(1)


# ============================================================================ report
def variant_stats(rows: list, model: str) -> dict:
    """一個模型、一組題：E0 與四個配置都有的題 → 每個配置對 E0、每一對（校準 − 每題 SVD）、判定。"""
    specs = [s for p in PAIRS for s in p]
    by = {}
    for r in rows:
        if r["model"] == model and r["spec"] in set(specs) | {"E0"}:
            by.setdefault(r["item_id"], {})[r["spec"]] = r
    items = sorted(i for i, d in by.items() if "E0" in d and all(s in d for s in specs))
    info = XS.MODEL_INFO[model]
    cb = info["L"] * 2 * info["H"] * XS.CHUNK * info["D"] * 2 * (XS.N_BLOCKS - 1)
    out = dict(n_items=len(items), configs={}, pairs={})
    for s in specs:
        sc = [float(by[i][s]["gen_score"]) for i in items]
        e0 = [float(by[i]["E0"]["gen_score"]) for i in items]
        d = XS.boot2([x - y for x, y in zip(sc, e0)]) if items else None
        out["configs"][s] = dict(acc=statistics.mean(sc) if items else None, acc_e0=statistics.mean(e0) if items else None, diff=d,
                                 eps_pass=bool(d and d["lo"] >= -XS.EPS), ratio=float(by[items[0]][s]["ratio_region"]) if items else None,
                                 bytes_region=int(by[items[0]][s]["bytes_region"]) if items else None)
    vs = []
    for c, ref in PAIRS:
        d = XS.boot2([float(by[i][c]["gen_score"]) - float(by[i][ref]["gen_score"]) for i in items]) if items else None
        v = XS.xlcal_verdict(d)
        shared = sum(x for k, x in XS.predict_bytes(XS.parse_spec(c), info, XS.N_BLOCKS - 1).items() if k.startswith("shared."))
        bc = out["configs"][c]["bytes_region"]
        out["pairs"][f"{c} vs {ref}"] = dict(cal_minus_ref=d, verdict=v, bytes_shared_basis=shared,
                                             ratio_cal=out["configs"][c]["ratio"], ratio_ref=out["configs"][ref]["ratio"],
                                             amortized_ratio=({str(N): round((bc + shared / N) / cb, 6) for N in (1, 8, 64)} if bc else None))
        vs.append(v)
    out["verdict"] = "INCOMPLETE" if len(items) < XS.MIN_ITEMS else XS.xlcal_aggregate(vs)
    out["items"] = items
    return out


def overall(by_model: dict) -> str:
    vals = list(by_model.values())
    if vals and all(v == "CAN_PRECOMPUTE" for v in vals):
        return "CAN_PRECOMPUTE"
    if "MUST_PER_ITEM" in vals:
        return "MUST_PER_ITEM"
    return "；".join(f"{m} {v}" for m, v in by_model.items())


def cmd_report(a):
    if os.path.abspath(a.out_dir) in (os.path.abspath(XS.OUT_DIR), os.path.abspath(XS.OUT_DIR_V2)):
        raise SystemExit("輸出只寫 x_stack/v3/（不覆蓋第一版與 v2 的檔）")
    old, pold, fold, _ = XS._gen_rows(a.raw_old)
    new, pnew, fnew, _ = XS._gen_rows(a.raw_new)
    ov = {r["item_id"] for r in old} & {r["item_id"] for r in new}
    if ov:
        raise SystemExit(f"38c 的題和新題重疊：{sorted(ov)[:5]}（seed 應該不重疊，規則 2：停）")
    rid = os.environ.get("RUN_ID", "adhoc")
    cal = json.load(open(a.calib_sha)) if a.calib_sha and os.path.exists(a.calib_sha) else None
    rep = dict(run_id=rid, ts=XS.ts(), rule_text=XLCAL3_TEXT, eps=XS.EPS, pairs=[list(p) for p in PAIRS], files_old=fold, files_new=fnew,
               src_run_ids_old=sorted({r["run_id"] for r in old}), src_run_ids_new=sorted({r["run_id"] for r in new}),
               incomplete_items_old=pold, incomplete_items_new=pnew, calib_xl=cal or "NOT_FOUND（check-calib 沒跑）",
               contention_new=dict(rows_with_intruders=sum(1 for r in new if r.get("own_gpu_intruders") not in ("0", "NOT_CHECKED")),
                                   host_levels=sorted({r.get("host_level", "") for r in new})), variants={})
    for key, desc in VARIANTS:
        rows = old if key == "old96" else new if key == "new192" else old + new
        V = rep["variants"][key] = dict(desc=desc, models={})
        for m in XS.MODEL_INFO:
            V["models"][m] = variant_stats(rows, m)
        V["verdict_by_model"] = {m: V["models"][m]["verdict"] for m in XS.MODEL_INFO}
        V["verdict"] = overall(V["verdict_by_model"])
    rep["verdict"] = rep["variants"]["new192"]["verdict"]                 # 主要判定＝新的 192 題（獨立檢驗；10/21 看到新資料前改定）
    rep["verdict_by_model"] = rep["variants"]["new192"]["verdict_by_model"]
    rep["verdict_merged288_reference"] = rep["variants"]["merged288"]["verdict_by_model"]
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(rep, open(os.path.join(a.out_dir, "xlcal3_report.json"), "w"), indent=1, ensure_ascii=False, default=str)
    L = ["# xKV 事先校準的基底 vs 每題 SVD：加題數（10/21 第三輪；只比 INT4 的兩對；判定事先寫死，見最後）", "",
         f"**整體判定（新的 192 題＝獨立檢驗）：{rep['verdict']}**（每個模型：{json.dumps(rep['verdict_by_model'], ensure_ascii=False)}）；"
         f"合併 288 題（參考）：{json.dumps(rep['verdict_merged288_reference'], ensure_ascii=False)}", "",
         f"38c 的 run：{len(rep['src_run_ids_old'])} 個；新的 run：{'、'.join(rep['src_run_ids_new']) or '（沒有）'}；"
         f"不齊全的題（沒用）：38c {len(pold)}、新 {len(pnew)}", ""]
    if cal:
        L += ["校準基底（沿用 38c，不重算）：" + "；".join(f"{m} sha256 {e.get('sha256')}（{'＝' if e['ok'] else '≠'} 38c 記的；calib run {e.get('calib_run_id')}）"
                                               for m, e in cal["models"].items()), ""]
    else:
        L += ["校準基底：check-calib 的結果不在（NOT_FOUND）", ""]
    for m in XS.MODEL_INFO:
        L += [f"## {m}", "", "| 校準版 vs 每題 SVD | " + " | ".join(d for _, d in VARIANTS) + " |", "|:--|" + "--:|" * len(VARIANTS)]
        for c, ref in PAIRS:
            k = f"{c} vs {ref}"
            cells = []
            for key, _ in VARIANTS:
                M = rep["variants"][key]["models"][m]
                e = M["pairs"][k]
                cells.append(f"{XS._ci(e['cal_minus_ref'], 100, 1)} {e['verdict']}（n {M['n_items']}）")
            L.append(f"| {k} | " + " | ".join(cells) + " |")
        L.append("| **模型判定** | " + " | ".join(f"**{rep['variants'][key]['models'][m]['verdict']}**" for key, _ in VARIANTS) + " |")
        M = rep["variants"]["merged288"]["models"][m]
        L += ["", "合併 288 題，每個配置對 E0：", "", "| 配置 | 位元組比 | 答對率 | 對 E0（百分點，CI） | ε |", "|:--|--:|--:|--:|:--|"]
        for s, c in M["configs"].items():
            if c["acc"] is not None:
                L.append(f"| {s} | {c['ratio']:.4f} | {100 * c['acc']:.1f} | {XS._ci(c['diff'], 100, 1)} | {'過' if c['eps_pass'] else '沒過'} |")
        L += ["", "位元組（合併）：" + "；".join(f"{k}：每請求 {e['ratio_cal']}／{e['ratio_ref']}、共用基底 {e['bytes_shared_basis'] / 1e6:.1f} MB、"
                                            f"攤提 N＝1／8／64：{'／'.join(str((e['amortized_ratio'] or {}).get(x)) for x in ('1', '8', '64'))}"
                                            for k, e in M["pairs"].items()), ""]
    L += ["## 事先寫死的判定與改動", ""] + [f"- **{k}**：{v}" for k, v in XLCAL3_TEXT.items()]
    open(os.path.join(a.out_dir, "tables_xlcal3.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:40]))


# ============================================================================ --self-test（CPU；不 import torch）
def self_test() -> bool:
    import csv
    import gzip
    import random
    import tempfile
    res, ok = {}, True

    def check(name, cond, **kw):
        nonlocal ok
        ok &= bool(cond)
        res[name] = {"pass": bool(cond), **kw}
        print(("PASS " if cond else "FAIL ") + name, json.dumps(kw, default=str, ensure_ascii=False)[:300], flush=True)
    # 1. 新題和所有用過的題不重疊；192 題
    new = {(t, s) for t, _, _, s in XS.plan_interleaved_cpu(list(TASKS), N_PER_TASK, SEED0)}
    used = {k: {(t, s) for t, _, _, s in XS.plan_interleaved_cpu(list(ts), n, s0)} for k, (s0, n, ts) in USED.items()}
    check("seeds_disjoint_xlcal3", len(new) == 192 and all(not (new & v) for v in used.values()),
          overlaps={k: len(new & v) for k, v in used.items()}, new_seeds=[SEED0, SEED0 + N_PER_TASK - 1])
    # 2. 計畫：兩對 INT4（＋各自的每題 SVD 版）、和 v2 的 xlcal_pairs 的 INT4 兩對相同；不默默覆蓋
    p = build_plan()
    v2pairs = [tuple(x) for x in XS.xlcal_pairs() if "+q:int4_kivi" in x[0]]
    check("plan_two_int4_pairs", list(PAIRS) == v2pairs and all(len(v["configs"]) == 4 for v in p["models"].values())
          and p["seed0"] == SEED0 and p["n"] == N_PER_TASK, pairs=PAIRS)
    with tempfile.TemporaryDirectory() as d:
        po = os.path.join(d, "s.json")
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            cmd_plan(SimpleNamespace(out=po))
            cmd_plan(SimpleNamespace(out=po))
            j = json.load(open(po))
            j["n"] = 24
            json.dump(j, open(po, "w"))
            try:
                cmd_plan(SimpleNamespace(out=po))
                refused = False
            except SystemExit:
                refused = True
        check("plan_no_silent_overwrite", refused)
        # 3. check-calib：sha 相同 → ok；不同／不在 → 不 ok
        f = os.path.join(d, "c.pt")
        open(f, "wb").write(b"calib-bytes")
        sj = os.path.join(d, "c.json")
        json.dump(dict(sha256=XS.sha256(f), run_id="R"), open(sj, "w"))
        okc = calib_status(f, sj)["ok"]
        json.dump(dict(sha256="0" * 64, run_id="R"), open(sj, "w"))
        bad = calib_status(f, sj)
        check("check_calib_sha", okc and not bad["ok"] and not calib_status(os.path.join(d, "none.pt"), sj)["ok"], bad=bad.get("why"))
        # 4. 報告：合成的 38c（96 題，含只降秩那對）＋新（192 題）→ 三種並列、配對、判定（兩對）、INCOMPLETE、不准寫 v2
        rnd = random.Random(5)
        po_old, po_new = os.path.join(d, "old_s0.csv.gz"), os.path.join(d, "new_s0.csv.gz")
        expect = {}
        for path, n, s0, tag in ((po_old, 96, XS.X2_SEED0, "old"), (po_new, 192, SEED0, "new")):
            with gzip.open(path, "wt", newline="") as fo:
                w = csv.DictWriter(fo, fieldnames=XS.FIELDS_X2)
                w.writeheader()
                for m in XS.MODEL_INFO:
                    specs = [s for pr in PAIRS for s in pr] + (["xl:k11:v17:cal", "xl:k11:v17"] if tag == "old" else [])
                    for k in range(n):
                        e0 = int(rnd.random() < 0.8)
                        rows = [dict(spec="E0", gen_score=e0, ratio_region=1.0, bytes_region="")]
                        for s in specs:
                            r = int(rnd.random() < 0.8)
                            if m == "llama" and s == PAIRS[1][0]:
                                r = 0 if k % 2 else r                       # Llama 第二對：校準版每兩題掉一題 → MUST
                            by = XS.bytes_summary(XS.predict_bytes(XS.parse_spec(s), XS.MODEL_INFO[m], 63), XS.MODEL_INFO[m], 63, 64)
                            rows.append(dict(spec=s, gen_score=r, ratio_region=by["ratio_region"], bytes_region=by["bytes_region"]))
                        expect.setdefault((m, tag), []).append({x["spec"]: x["gen_score"] for x in rows})
                        for r_ in rows:
                            w.writerow(dict(r_, run_id=f"SYN-{tag}", ts="t", model=m, ctx="32K", item_id=f"T2_s{s0 + k}", n_rows_item=len(rows),
                                            own_gpu_intruders=0, host_level="QUIET"))
        out = os.path.join(d, "v3")
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            cmd_report(SimpleNamespace(raw_old=[po_old], raw_new=[po_new], calib_sha=None, out_dir=out))
        rep = json.load(open(os.path.join(out, "xlcal3_report.json")))
        V = rep["variants"]
        n_ok = all(V[k]["models"][m]["n_items"] == n for k, n in (("old96", 96), ("new192", 192), ("merged288", 288)) for m in XS.MODEL_INFO)
        c, ref = PAIRS[0]
        want = statistics.mean([x[c] - x[ref] for tag in ("old", "new") for x in expect[("qwen", tag)]])
        got = V["merged288"]["models"]["qwen"]["pairs"][f"{c} vs {ref}"]["cal_minus_ref"]["mean"]
        agg_two = len(V["merged288"]["models"]["qwen"]["pairs"]) == 2 and V["merged288"]["models"]["llama"]["verdict"] == "MUST_PER_ITEM"
        check("report_three_variants_pairing", n_ok and abs(got - want) < 1e-12 and agg_two and rep["verdict"] == "MUST_PER_ITEM"
              and os.path.exists(os.path.join(out, "tables_xlcal3.md")), merged_mean=(got, want), verdicts={k: V[k]["verdict"] for k in V})
        few = variant_stats([r for r in XS.read_rows([po_new]) if r["item_id"] < f"T2_s{SEED0 + 30}"], "qwen")
        check("report_incomplete_below_48", few["verdict"] == "INCOMPLETE" and few["n_items"] < 48, n=few["n_items"])
        try:
            cmd_report(SimpleNamespace(raw_old=[], raw_new=[], calib_sha=None, out_dir=XS.OUT_DIR_V2))
            g = False
        except SystemExit:
            g = True
        check("report_refuses_v1_v2_dirs", g)
    check("aggregate_rule_two_pairs", XS.xlcal_aggregate(["CAN_PRECOMPUTE", "CAN_PRECOMPUTE"]) == "CAN_PRECOMPUTE"
          and XS.xlcal_aggregate(["CAN_PRECOMPUTE", "MUST_PER_ITEM"]) == "MUST_PER_ITEM"
          and XS.xlcal_aggregate(["CAN_PRECOMPUTE", "UNDETERMINED"]) == "UNDETERMINED" and XLCAL3_TEXT["rule"] == XS.XLCAL_TEXT["rule"])
    # 5. job 用的 m16_xstack gen 指令照樣解析（--calib-xl、--xl-cache 是已有的旗子）
    ap = XS.build_parser()
    a = ap.parse_args(["gen", "--model", "llama", "--ctx", "32K", "--tier", "core", "--n", "48", "--seed0", str(SEED0), "--selection",
                       os.path.join(OUT_V3, "xlcal3_selection.json"), "--calib-xl", "/x/calib_xl_llama.pt", "--xl-cache", "--shard", "0/12",
                       "--budget-min", "90", "--out", "/x/gen_llama_32K_s0.csv.gz"])
    a2 = build_parser().parse_args(["report", "--raw-old", "/x/a", "--raw-new", "/x/b"])
    check("cli_parse", a.xl_cache and a.calib_xl and a.seed0 == SEED0 and a2.out_dir == OUT_V3)
    bad_mods = [x for x in ("torch", "vllm", "transformers") if x in sys.modules]
    check("no_torch_imported", not bad_mods, imported=bad_mods)
    print(f"{'ALL PASS' if ok else 'SOME FAILED'}（{len(res)} 項）", flush=True)
    return ok


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan")
    p.add_argument("--out", default=os.path.join(OUT_V3, "xlcal3_selection.json"))
    c = sp.add_parser("check-calib")
    c.add_argument("--calib", default=CALIB_PT, help="{m}＝模型")
    c.add_argument("--v2-summary", default=CALIB_V2)
    c.add_argument("--out", default=os.path.join(OUT_V3, "calib_xl_sha.json"))
    r = sp.add_parser("report")
    r.add_argument("--raw-old", nargs="+", required=True, help="38c 的生成逐列（$XD/xlcal/gen_*_32K_s*.csv.gz）")
    r.add_argument("--raw-new", nargs="+", required=True)
    r.add_argument("--calib-sha", default=os.path.join(OUT_V3, "calib_xl_sha.json"))
    r.add_argument("--out-dir", default=OUT_V3)
    return ap


def main():
    if "--self-test" in sys.argv[1:]:
        sys.exit(0 if self_test() else 1)
    a = build_parser().parse_args()
    {"plan": cmd_plan, "check-calib": cmd_check_calib, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
