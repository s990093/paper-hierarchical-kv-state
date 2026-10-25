#!/usr/bin/env python3
"""m16_x3grid.py — X3 補跑：「最快的還原格式會不會隨 GPU 忙碌翻轉」的完整實測格子（10/22 使用者要求；38j 只有 4 對）。

不寫新的 GPU 程式：格子交給 m16_xtime.py validate 量（同一條端到端路徑：第二階段的還原器＋載入線），這支只做兩件 CPU 的事：
  build   做一個 map 目錄（flip_cells.json＝下面的格子；map_cells.csv 連到 X3 map v2 的大檔，給 validate 填預測值）
  report  讀 validate_cells.csv，判「翻轉」與「依情況選」的價值
設計與事先寫死的判定：docs/stage3_20261019/experiment_log.md「X3 翻轉格子」（排進佇列之前寫，不改）。
  格子：2 模型 × 3 層（cpu_pinned、b16、ssd_meas）× 4 個 GPU 忙碌度（duty 0、0.4、0.7、0.9）；每格 6 個動作：
  raw（BF16）、int4_kivi@oneshot、k4v2@perblock、xkv_k22v34@oneshot、xkv_k22v34_int4@oneshot、xkv_k11v17_int4@oneshot。

  python code/m16_x3grid.py build --out <map 目錄>
  python code/m16_x3grid.py report --val <validate 目錄> --out-dir <dir>
  python code/m16_x3grid.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timezone

MAP2_BIG = "/mlsteam/data/tiara/runs/20261021-123143-m16-x3-map2/x3_full/map_cells.csv"
MODELS = ("llama", "qwen")
TIERS = ("cpu_pinned", "b16", "ssd_meas")
DUTIES = (0.0, 0.4, 0.7, 0.9)
ACTIONS = ("raw", "int4_kivi@oneshot", "k4v2@perblock", "xkv_k22v34@oneshot", "xkv_k22v34_int4@oneshot", "xkv_k11v17_int4@oneshot")
MARGIN = 0.05                     # 翻轉的兩端，最快的都要比對方的選擇快 ≥5%


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def dkey(d) -> str:
    return f"{float(d):.1f}"


def cells(pred: dict) -> list:
    out = []
    for m in MODELS:
        for t in TIERS:
            for d in DUTIES:
                meas = [dict(action=a, role=f"G{i}", pred_s=pred.get((m, t, dkey(d), a))) for i, a in enumerate(ACTIONS)]
                out.append(dict(cell=f"{m}|{t}|{d}", mode="G", model=m, tier=t, duty=d, priority=0,
                                A="raw", S="raw", F="raw", measure=meas))   # A＝F → validate 的 STOP-3 不算 A 對 F（這裡不用）
    return out


def read_pred(path: str) -> dict:
    p = {}
    for r in csv.DictReader(open(path)):
        if r["grid"] == "named" and r["T_s"]:
            p[(r["model"], r["tier"], dkey(r["duty"]), r["action"])] = float(r["T_s"])
    return p


def cmd_build(a):
    os.makedirs(a.out, exist_ok=True)
    pred = read_pred(MAP2_BIG)
    cs = cells(pred)
    miss = [(c["cell"], x["action"]) for c in cs for x in c["measure"] if x["pred_s"] is None]
    tiers = {(r[0], r[1]) for r in pred}
    for m in MODELS:
        for t in TIERS:
            if (m, t) not in tiers:
                raise SystemExit(f"map v2 沒有 {m} 的 {t} 層 → 格子設計錯（規則 2：停）")
    json.dump(dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=now(), worth_validate=True,
                   reason=f"X3 翻轉格子（10/22 補跑）：{len(cs)} 格 × {len(ACTIONS)} 個動作；不是 map 挑的", modes=["G"], cells=cs,
                   source_map=MAP2_BIG, missing_pred=miss), open(os.path.join(a.out, "flip_cells.json"), "w"), indent=1, ensure_ascii=False)
    link = os.path.join(a.out, "map_cells.csv")
    if not os.path.exists(link):
        os.symlink(MAP2_BIG, link)
    print(f"{len(cs)} 格、{len(cs) * len(ACTIONS)} 個量測；沒有預測值的 {len(miss)} 個 → {a.out}")
    return 0


def gm(v):
    return math.exp(sum(math.log(x) for x in v) / len(v))


def analyze(rows: list) -> dict:
    ok = [r for r in rows if r.get("meas_s") not in ("", None) and str(r.get("contaminated")) in ("False", "")]
    t = {}
    for r in ok:
        t[(r["model"], r["tier"], dkey(r["duty"]), r["action"])] = float(r["meas_s"])
    series = []
    for m in MODELS:
        for tier in TIERS:
            per = {}
            for d in DUTIES:
                v = {a: t[(m, tier, dkey(d), a)] for a in ACTIONS if (m, tier, dkey(d), a) in t}
                if v:
                    per[dkey(d)] = v
            if len(per) < 2:
                series.append(dict(model=m, tier=tier, status="INCOMPLETE", n_duties=len(per)))
                continue
            best = {d: min(v, key=v.get) for d, v in per.items()}
            lo, hi = dkey(DUTIES[0]), dkey(DUTIES[-1])
            flip, margins = False, None
            if lo in per and hi in per and best[lo] != best[hi]:
                bl, bh = best[lo], best[hi]
                m_lo = per[lo][bh] / per[lo][bl] - 1 if bh in per[lo] else None     # 閒時：閒的最快比忙的最快快多少
                m_hi = per[hi][bl] / per[hi][bh] - 1 if bl in per[hi] else None     # 忙時：反過來
                margins = (m_lo, m_hi)
                flip = m_lo is not None and m_hi is not None and m_lo >= MARGIN and m_hi >= MARGIN
            common = [a for a in ACTIONS if all(a in v for v in per.values())]
            fixed = {a: gm([per[d][a] / per[d][best[d]] for d in per]) for a in common}
            bf = min(fixed, key=fixed.get) if fixed else None
            series.append(dict(model=m, tier=tier, status="OK", best_by_duty=best, flip=flip, flip_margins=margins,
                               best_fixed=bf, adaptive_gain=(1 - 1 / fixed[bf]) if bf else None,
                               times={d: v for d, v in per.items()}))
    by_model = {m: any(s.get("flip") for s in series if s["model"] == m) for m in MODELS}
    n_flip = sum(bool(s.get("flip")) for s in series)
    verdict = ("FLIP_REPRODUCED（兩個模型都至少一條翻轉）" if all(by_model.values()) else
               "FLIP_PARTIAL（只有一個模型翻轉）" if any(by_model.values()) else "NO_FLIP（沒有任何一條翻轉）")
    return dict(series=series, n_flip=n_flip, flip_by_model=by_model, verdict=verdict, n_rows=len(rows), n_ok=len(ok),
                n_contaminated=sum(str(r.get("contaminated")) not in ("False", "") for r in rows))


def cmd_report(a):
    rid = os.environ.get("RUN_ID", "adhoc")
    rows = list(csv.DictReader(open(os.path.join(a.val, "validate_cells.csv"))))
    res = analyze(rows)
    res.update(run_id=rid, ts=now(), src_run_ids=sorted({r["run_id"] for r in rows}), val_dir=a.val,
               rule=(f"每條（模型, 層）：duty 0 與 0.9 的最快動作不同，而且兩端最快的都比對方的選擇快 ≥{MARGIN:.0%} → 翻轉。"
                     "兩個模型都至少一條翻轉 → FLIP_REPRODUCED；只有一個 → FLIP_PARTIAL；都沒有 → NO_FLIP。"
                     "另報：依情況選（每個 duty 各挑最快）對「固定最好的一種」的幾何平均好處。污染列不算。"))
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(a.out_dir, "x3grid_report.json"), "w"), indent=1, ensure_ascii=False)
    L = [f"# X3 翻轉格子（run {rid}）", "", f"來源：{', '.join(res['src_run_ids'])}；{res['n_ok']}／{res['n_rows']} 列可用（污染 {res['n_contaminated']}）",
         "", f"**判定：{res['verdict']}**（翻轉 {res['n_flip']} 條）", "", "規則：" + res["rule"], ""]
    for s in res["series"]:
        if s["status"] != "OK":
            L.append(f"## {s['model']} {s['tier']}：{s['status']}")
            continue
        L += [f"## {s['model']} {s['tier']}：翻轉 {'是' if s['flip'] else '否'}；固定最好＝{s['best_fixed']}；依情況選好 {s['adaptive_gain']:.1%}", "",
              "| duty | " + " | ".join(ACTIONS) + " | 最快 |", "|--:|" + "--:|" * len(ACTIONS) + ":--|"]
        for d, v in s["times"].items():
            L.append(f"| {d} | " + " | ".join(f"{v[x]:.4f}" if x in v else "—" for x in ACTIONS) + f" | {s['best_by_duty'][d]} |")
        L.append("")
    open(os.path.join(a.out_dir, "x3grid_report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:8]))
    return 0


def self_test() -> bool:
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name + (f"  {info}" if info else ""))
    cs = cells({})
    check("格子數＝2×3×4", len(cs) == 24)
    check("每格 6 個動作", all(len(c["measure"]) == 6 for c in cs))
    rows = []
    for m in MODELS:
        for t in TIERS:
            for d in DUTIES:
                for a in ACTIONS:
                    v = {"raw": 1.0, "int4_kivi@oneshot": 0.5 + 0.1 * d, "xkv_k22v34@oneshot": 0.3 + 1.0 * d}.get(a, 2.0)
                    rows.append(dict(model=m, tier=t, duty=str(d), action=a, meas_s=str(v), contaminated="False", run_id="x"))
    r = analyze(rows)
    check("假資料：閒時低秩最快、忙時 INT4 最快 → 每條都翻轉", r["n_flip"] == 6 and r["verdict"].startswith("FLIP_REPRODUCED"), str(r["n_flip"]))
    s = r["series"][0]
    check("依情況選的好處 > 0", s["adaptive_gain"] > 0, f"{s['adaptive_gain']:.3f}")
    for x in rows:
        if x["action"] == "xkv_k22v34@oneshot":
            x["contaminated"] = "True"
    r = analyze(rows)
    check("污染列不算 → 沒有翻轉", r["n_flip"] == 0 and r["verdict"].startswith("NO_FLIP"))
    print("self-test：" + ("ALL PASS" if ok else "有 FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    sp = ap.add_subparsers(dest="cmd")
    b = sp.add_parser("build")
    b.add_argument("--out", required=True)
    r = sp.add_parser("report")
    r.add_argument("--val", required=True)
    r.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if a.cmd == "build":
        sys.exit(cmd_build(a))
    if a.cmd == "report":
        sys.exit(cmd_report(a))
    ap.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
