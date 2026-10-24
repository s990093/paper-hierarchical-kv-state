"""h200_a2c7.py — A2（寫入干擾）與 C7 錨點的分析，H200 只跑限速層（job 20643；使用者 2026-10-16：「都用模擬限速」）

子命令：
  regress
      MI300X：在 git 資料的複本上跑 m7_analyze 的 calib()、a2()，summary_a2.csv、summary_c7_anchor.csv 要和 git 逐格相同，
      verdict 的 A2、C7_anchor 要和 git 的 verdict.json 相同（m7_analyze 改成「沒有 real 列也能算」之後的回歸）。
  run
      H200：在 $M7_RES 的複本上跑 calib()、a2()（m7_analyze 同一套），只把 summary_a2.csv、summary_c7_anchor.csv、
      a2c7_verdict.json 寫回 $M7_RES（calib() 也會重寫 summary_c4.csv 等，那些留在複本，不覆寫 R3 的）。
      再和 MI300X 並排：a2_vs_mi300x.csv、c7_anchor_vs_mi300x.csv（限速層兩台都有；真實裝置只有 MI300X，H200 寫 NOT_MEASURED）。
複本與圖放 $RUN_DIR（沒有就放暫存目錄）。
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.dirname(HERE)
MI = os.path.join(REPO, "results", "m7_write_policy_mi300x")
RUN_ID = os.environ.get("RUN_ID", "adhoc")
TS = datetime.now().astimezone().isoformat(timespec="seconds")


def _run_dir():
    """r_h200_cpu.sh 的 hrun 只給 RUN_ID；run 目錄＝$TIARA_RUNS/$RUN_ID（存在才用）"""
    p = os.path.join(os.environ.get("TIARA_RUNS", ""), RUN_ID)
    return p if os.environ.get("TIARA_RUNS") and os.path.isdir(p) else None


def _stage(src, name):
    base = os.environ.get("RUN_DIR") or _run_dir() or tempfile.mkdtemp(prefix="h200_a2c7_")
    d = os.path.join(base, name)
    os.makedirs(os.path.join(d, "figs"), exist_ok=True)
    for f in os.listdir(src):
        p = os.path.join(src, f)
        if os.path.isfile(p) and f.endswith((".csv", ".json")):
            shutil.copy2(p, d)
    return d


def _an(d):
    import m7_analyze as A
    A.R, A.FIG = d, os.path.join(d, "figs")
    A.V.clear()
    A.calib()
    A.a2()
    return json.loads(json.dumps(A.V, default=str))


def regress():
    d = _stage(MI, "mi300x_copy")
    V = _an(d)
    ok = True
    for fn in ("summary_a2.csv", "summary_c7_anchor.csv"):
        a, b = pd.read_csv(os.path.join(MI, fn)), pd.read_csv(os.path.join(d, fn))
        same = a.shape == b.shape and a.equals(b)
        print(fn, a.shape, b.shape, "IDENTICAL" if same else "DIFF")
        ok &= same
    g = json.load(open(os.path.join(MI, "verdict.json")))
    for k, a, b in (("A2", g.get("A2"), V.get("A2")), ("C7_anchor", g["calib"].get("C7_anchor"), V["calib"].get("C7_anchor"))):
        same = json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
        print("verdict", k, "IDENTICAL" if same else f"DIFF\n git {a}\n new {b}")
        ok &= same
    sys.exit(0 if ok else 1)


def run():
    h2 = os.environ["M7_RES"]
    assert os.path.abspath(h2) != os.path.abspath(MI), "M7_RES 是 MI300X 的目錄（沒有 source env_h200.sh？）"
    d = _stage(h2, "h200_copy")
    V = _an(d)
    for fn in ("summary_a2.csv", "summary_c7_anchor.csv"):
        x = pd.read_csv(os.path.join(d, fn))
        x.insert(0, "ts", TS)
        x.insert(0, "run_id", RUN_ID)
        x.to_csv(os.path.join(h2, fn), index=False)
    out = {"run_id": RUN_ID, "ts": TS, "A2": V.get("A2"), "C7_anchor": V["calib"].get("C7_anchor"),
           "C4_delta_pct": V["calib"].get("C4", {}).get("delta_pct")}
    json.dump(out, open(os.path.join(h2, "a2c7_verdict.json"), "w"), indent=1, ensure_ascii=False, default=str)
    pd.set_option("display.width", 250)
    # A2 並排：同一個（device, model, bg_write）兩台的 TTFT 中位數；H200 沒有 real 列 → NOT_MEASURED
    k = ["device", "model", "bg_write"]
    hm = pd.read_csv(os.path.join(d, "summary_a2.csv"))
    mm = pd.read_csv(os.path.join(MI, "summary_a2.csv"))
    a = mm[k + ["ttft", "n", "meet"]].merge(hm[k + ["ttft", "n", "meet"]], on=k, how="outer", suffixes=("_mi300x", "_h200"))
    for x in ("mi300x", "h200"):   # 和同一個 device、model 的「背景不寫」比（m7_analyze a2() 的 *_vs_none_pct）
        base = a[a.bg_write == "none"].set_index(["device", "model"])[f"ttft_{x}"]
        a[f"vs_none_pct_{x}"] = [(getattr(r, f"ttft_{x}") / base.get((r.device, r.model), float("nan")) - 1) * 100 for r in a.itertuples()]
    a = a.astype(object).where(a.notna(), "NOT_MEASURED")
    a.insert(0, "ts", TS)
    a.insert(0, "run_id", RUN_ID)
    a.to_csv(os.path.join(h2, "a2_vs_mi300x.csv"), index=False)
    print(a.drop(columns=["run_id", "ts"]).to_string(index=False))
    # C7 錨點並排：sim 兩台都有；real 只有 MI300X
    k = ["device", "mode", "L"]
    hc = pd.read_csv(os.path.join(d, "summary_c7_anchor.csv"))
    mc = pd.read_csv(os.path.join(MI, "summary_c7_anchor.csv"))
    c = mc[k + ["real", "sim", "sim_over_real"]].rename(columns=lambda s: s if s in k else s + "_mi300x")
    c = c.merge(hc[k + ["sim"]].rename(columns={"sim": "sim_h200"}), on=k, how="outer")
    c["real_h200"] = "NOT_MEASURED"
    c["sim_h200_over_sim_mi300x"] = c.sim_h200 / c.sim_mi300x
    c.insert(0, "ts", TS)
    c.insert(0, "run_id", RUN_ID)
    c.to_csv(os.path.join(h2, "c7_anchor_vs_mi300x.csv"), index=False)
    print(c.drop(columns=["run_id", "ts"]).round(4).to_string(index=False))
    print(json.dumps(out, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    {"regress": regress, "run": run}[sys.argv[1]]()
