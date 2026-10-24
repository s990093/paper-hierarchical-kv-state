"""h200_brr.py — B 重跑（job 20631）之後的合併與新舊並排（2026-10-16，使用者：「照重疊重跑」，兩組都報）

子命令：
  merge
      建 results/m7_write_policy_h200_merged/＝「沒有和我們自己的 F2 job 重疊」的那一組 B：
        b_share.csv          舊的 R3-B-main（run 20261016-063929-m7h-b-nfs-gap0，只和 20417 自己）＋重跑的 b_share.csv（6 個 -rr run）
        b_fifo.csv、b_share_cpu3.69.csv   重跑的（舊的整個 run 都和 F2 重疊）
        b2_share.csv         舊的 G1-s1、G2-s0（12:38 F2 結束後才開始）＋重跑的 G1-s0-rr
        b_workload_seed*.json 舊目錄的（seed 0–2 和重跑目錄逐位元相同，merge 時再比一次）
      每一列原封不動（run_id 指回原始 run）。斷言：兩部分的欄位相同、(設定, rep, ev) 不重複、每個 run 都 contaminated_gpu=False。
      另寫 MANIFEST.csv（每個檔：來源目錄、run_id、列數）。
  ball --old <dir> --new <dir>
      B 全部設定（m7_analyze b()，和 run 20261016-124102-h200-b-all-settings 同一套）在 --new 上重算：
      暫存目錄放 --old 的校準檔（C4 的 δ 從這裡來）＋ --new 的 b_*.csv、b_workload_*.json，
      不放 b_vs_mi300x_all.csv（檔名也是 b_*.csv，b() 會把它當成 B 的資料讀進去）。
      輸出 --new/summary_b_verdict_all.csv、--new/b_vs_mi300x_all.csv（和舊的同格式）
  stage --old <dir> --new <dir> [--dst <dir>]
      給只讀「一個 M7_RES 目錄」的程式（m8_kappa_map 的 D3）用：--dst（預設＝這個 run 目錄的 res/）放 --old 的所有
      .csv／.json／.csv.gz，再用 --new 的 b_share、b_fifo、b_share_cpu3.69、b2_share、b_workload_*.json 蓋掉（舊的 b_share.csv.gz 刪掉）。
      另寫 STAGE.csv（每個檔來自哪裡）。
  side --old <dir> --new <dir>
      新舊並排：summary_b_verdict_all、sim_validate、b2_summary、b2_verdict、h1_verdict
      → --new/old_vs_new_*.csv（每一列：設定鍵＋舊值＋新值）
"""
from __future__ import annotations

import argparse
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
OLD = os.path.join(REPO, "results", "m7_write_policy_h200")
RR = os.path.join(REPO, "results", "m7_write_policy_h200_rerun")
MERGED = os.path.join(REPO, "results", "m7_write_policy_h200_merged")
RUN_ID = os.environ.get("RUN_ID", "adhoc")
TS = datetime.now().astimezone().isoformat(timespec="seconds")

R3_MAIN = "20261016-063929-m7h-b-nfs-gap0"
B2_KEEP = ["20261016-132601-m7h-b2-g1-doc-hold-cpu3.69-s1", "20261016-142019-m7h-b2-g2-chat-hold-s0"]
B2_DROP = "20261016-120250-m7h-b2-g1-doc-hold-cpu3.69-s0"
B_KEY = ["strategy", "cpu_frac", "ssd_frac", "ssd_dev", "gap_s", "wl_seed", "io_model", "rep", "ev"]
B2_KEY = B_KEY + ["cpu_gibps", "release", "workload"]


def _read(d, fn):
    p = os.path.join(d, fn)
    return pd.read_csv(p if os.path.exists(p) else p + ".gz")


def _join(old, new, key, what):
    assert list(old.columns) == list(new.columns), f"{what}：欄位不同\n舊 {list(old.columns)}\n新 {list(new.columns)}"
    m = pd.concat([old, new], ignore_index=True)
    dup = m.duplicated(key, keep=False)
    assert not dup.any(), f"{what}：(設定, rep, ev) 重複 {int(dup.sum())} 列\n{m[dup].head()}"
    bad = m[m.contaminated_gpu.astype(str) != "False"]
    assert bad.empty, f"{what}：有 contaminated_gpu≠False 的列：{sorted(bad.run_id.unique())}"
    return m


def _run_dir():
    """r_h200_cpu.sh 的 hrun 只給 RUN_ID；run 目錄＝$TIARA_RUNS/$RUN_ID（存在才用）"""
    p = os.path.join(os.environ.get("TIARA_RUNS", ""), RUN_ID)
    return p if os.environ.get("TIARA_RUNS") and os.path.isdir(p) else None


def merge(a):
    os.makedirs(MERGED, exist_ok=True)
    man = []
    ob = _read(OLD, "b_share.csv")
    assert R3_MAIN in set(ob.run_id), f"舊 b_share 裡沒有 {R3_MAIN}"
    o = ob[ob.run_id == R3_MAIN]
    n = _read(RR, "b_share.csv")
    assert not n.run_id.str.endswith("-rr").eq(False).any(), "重跑的 b_share 裡有不是 -rr 的 run"
    # R3-B-main 是 rep 0–2；重跑的 r36 是 rep 3–5（--rep-start 3）。主設定之外的重跑 run 沒有和 R3-B-main 同設定
    m = _join(o, n, B_KEY, "b_share")
    m.to_csv(os.path.join(MERGED, "b_share.csv"), index=False)
    for rid, g in m.groupby("run_id"):
        man.append(dict(file="b_share.csv", src=OLD if rid == R3_MAIN else RR, src_run_id=rid, rows=len(g)))
    for fn in ("b_fifo.csv", "b_share_cpu3.69.csv"):
        d = _read(RR, fn)
        assert d.run_id.str.endswith("-rr").all() and (d.contaminated_gpu.astype(str) == "False").all(), fn
        shutil.copy2(os.path.join(RR, fn), os.path.join(MERGED, fn))
        for rid, g in d.groupby("run_id"):
            man.append(dict(file=fn, src=RR, src_run_id=rid, rows=len(g)))
    ob2 = _read(OLD, "b2_share.csv")
    assert set(ob2.run_id) == set(B2_KEEP) | {B2_DROP}, f"舊 b2_share 的 run 和預期不同：{sorted(set(ob2.run_id))}"
    n2 = _read(RR, "b2_share.csv")
    assert len(set(n2.run_id)) == 1 and n2.run_id.iloc[0].endswith(B2_DROP.split("-", 2)[2] + "-rr"), sorted(set(n2.run_id))
    m2 = _join(ob2[ob2.run_id.isin(B2_KEEP)], n2, B2_KEY, "b2_share")
    m2.to_csv(os.path.join(MERGED, "b2_share.csv"), index=False)
    for rid, g in m2.groupby("run_id"):
        man.append(dict(file="b2_share.csv", src=RR if rid.endswith("-rr") else OLD, src_run_id=rid, rows=len(g)))
    for s in range(5):
        fn = f"b_workload_seed{s}.json"
        po, pr = os.path.join(OLD, fn), os.path.join(RR, fn)
        if os.path.exists(pr):
            assert json.load(open(po)) == json.load(open(pr)), f"{fn} 新舊不同"
        shutil.copy2(po, os.path.join(MERGED, fn))
        man.append(dict(file=fn, src=OLD, src_run_id="", rows=0))
    mf = pd.DataFrame(man)
    mf.insert(0, "ts", TS)
    mf.insert(0, "run_id", RUN_ID)
    mf["src"] = mf.src.map(lambda p: os.path.relpath(p, REPO))
    mf.to_csv(os.path.join(MERGED, "MANIFEST.csv"), index=False)
    print(mf.to_string(index=False))
    print(f"b_share {len(m)} 列（舊 R3-B-main {len(o)}＋重跑 {len(n)}）；b2_share {len(m2)} 列"
          f"（舊 {int(ob2.run_id.isin(B2_KEEP).sum())}＋重跑 {len(n2)}）→ {MERGED}")


def ball(a):
    import m7_repro_compare as C
    rd = _run_dir()
    stage = os.path.join(rd, "stage") if rd else tempfile.mkdtemp(prefix="h200_ball_")
    os.makedirs(stage, exist_ok=True)
    for f in os.listdir(a.old):           # 舊目錄的校準等（不含 B 的資料與 B 的衍生檔）
        p = os.path.join(a.old, f)
        if not os.path.isfile(p) or not f.endswith((".csv", ".json")):
            continue
        if f.startswith(("b_", "b2_", "summary_b", "sim_", "repro_", "verdict")):
            continue
        # 12:41 那次（run 20261016-124102-h200-b-all-settings）的輸入沒有這兩種：job 20643 之後才有，而且只有限速層。
        # B 的判定只用到 calib() 的 C4 δ，所以不放
        if f.startswith(("a2_", "c7_anchor", "summary_a2", "summary_c7", "a2c7_")):
            continue
        shutil.copy2(p, stage)
    for f in os.listdir(a.new):
        if (f.startswith("b_") and f.endswith(".csv") and f != "b_vs_mi300x_all.csv") or \
           (f.startswith("b_workload_") and f.endswith(".json")):
            shutil.copy2(os.path.join(a.new, f), stage)
    print("暫存：", stage, sorted(os.listdir(stage)))
    tmp, V = C.analyze(stage, os.environ["M7_TIER_PARAMS"], os.path.join(rd or stage, "figs"), "h200")
    print("calib C4 δ:", V.get("calib", {}).get("C4", {}).get("delta_pct"), "errors:", V.get("errors"))
    d = pd.read_csv(os.path.join(tmp, "summary_b_verdict.csv"))
    d.insert(0, "ts", TS)
    d.insert(0, "run_id", RUN_ID)
    d.to_csv(os.path.join(a.new, "summary_b_verdict_all.csv"), index=False)
    mi = pd.read_csv(os.path.join(REPO, "results", "m7_write_policy_mi300x", "summary_b_verdict.csv"))
    k = ["io_model", "cpu_tier", "ssd_dev", "gap_s", "ssd_frac", "wl_seed", "cpu_frac"]
    for x in (d, mi):
        for c in k:
            x[c] = x[c].astype(str)
    m = mi.merge(d, on=k, how="outer", suffixes=("_mi300x", "_h200"))
    cols = k + ["n_reps_mi300x", "n_reps_h200", "s5_vs_s4plus_pct_mi300x", "s5_vs_s4plus_pct_h200", "gain_pct_mi300x",
                "gain_pct_h200", "best_simple_mi300x", "best_simple_h200"]
    m = m[cols]
    m.insert(0, "ts", TS)
    m.insert(0, "run_id", RUN_ID)
    m.to_csv(os.path.join(a.new, "b_vs_mi300x_all.csv"), index=False)
    pd.set_option("display.width", 250)
    print(m.drop(columns=["run_id", "ts"]).round(3).to_string(index=False))
    print("H200 rows", len(d), "MI300X rows", len(mi), "errors", V.get("errors"))
    if V.get("errors"):
        sys.exit(1)


B_FILES = ("b_share.csv", "b_fifo.csv", "b_share_cpu3.69.csv", "b2_share.csv")


def stage(a):
    rd = _run_dir()
    dst = a.dst or (os.path.join(rd, "res") if rd else None)
    assert dst, "沒有 --dst，也不在 hrun 的 run 目錄裡"
    os.makedirs(dst, exist_ok=True)
    rows = []
    for f in sorted(os.listdir(a.old)):
        p = os.path.join(a.old, f)
        if os.path.isfile(p) and f.endswith((".csv", ".json", ".csv.gz")):
            shutil.copy2(p, dst)
            rows.append(dict(file=f, src=a.old))
    for f in sorted(os.listdir(a.new)):
        if f in B_FILES or (f.startswith("b_workload_") and f.endswith(".json")):
            shutil.copy2(os.path.join(a.new, f), dst)
            rows = [r for r in rows if r["file"] != f] + [dict(file=f, src=a.new)]
            gz = os.path.join(dst, f + ".gz")
            if os.path.exists(gz):
                os.remove(gz)
                rows = [r for r in rows if r["file"] != f + ".gz"]
    for f in B_FILES:
        assert any(r["file"] == f and r["src"] == a.new for r in rows), f"--new 裡沒有 {f}"
    st = pd.DataFrame(rows)
    st["src"] = st.src.map(lambda p: os.path.relpath(p, REPO))
    st.insert(0, "ts", TS)
    st.insert(0, "run_id", RUN_ID)
    st.to_csv(os.path.join(dst, "STAGE.csv"), index=False)
    print(st.to_string(index=False))
    print("→", dst)


SIDE = {   # 檔名：（設定鍵, 要並排的值）
    "summary_b_verdict_all.csv": (["io_model", "cpu_tier", "ssd_dev", "gap_s", "ssd_frac", "wl_seed", "cpu_frac"],
                                  ["n_reps", "best_simple", "gain_pct", "s5_vs_s4plus_pct", "verdict"]),
    "sim_validate.csv": (["io_model", "cpu_gibps", "ssd_dev", "gap_s", "ssd_frac", "wl_seed", "cpu_frac", "strategy"],
                         ["reps", "meas_median", "sim_median", "err_median_pct"]),
    "b2_summary.csv": (["workload", "release", "cpu_gibps", "ssd_dev", "io_model", "gap_s", "ssd_frac", "wl_seed", "cpu_frac",
                        "strategy"], ["reps_complete", "gpu_median", "sim_median", "err"]),
    "b2_verdict.csv": (["workload", "release", "cpu_gibps", "ssd_dev", "io_model", "gap_s", "ssd_frac", "wl_seed", "cpu_frac",
                        "strategy", "src"], ["best_nwt", "gain_nwt", "gain_s4b", "pass", "reps"]),
    "h1_verdict.csv": (["src", "io_model", "ssd_dev", "gap_s", "ssd_frac", "wl_seed", "cpu_gibps", "release", "workload",
                        "cpu_frac"], ["best", "cand", "min_ssd_nwt", "min_ssd_wt", "wt_adv"]),
}
OLD_LOC = {"h1_verdict.csv": os.path.join(REPO, "results", "m7_explore_h200")}


def side(a):
    pd.set_option("display.width", 250)
    for fn, (key, val) in SIDE.items():
        po = os.path.join(OLD_LOC.get(fn, a.old), fn)
        pn = os.path.join(a.new, fn)
        if not (os.path.exists(po) and os.path.exists(pn)):
            print(f"== {fn}：缺檔（舊 {os.path.exists(po)}，新 {os.path.exists(pn)}）→ 跳過")
            continue
        o, n = pd.read_csv(po), pd.read_csv(pn)
        for x in (o, n):
            for c in key:
                x[c] = x[c].astype(str)
        rid = [c for c in ("run_id",) if c in o.columns]
        m = o[key + val + rid].merge(n[key + val + rid], on=key, how="outer", suffixes=("_old", "_new"))
        m.insert(0, "ts", TS)
        m.insert(0, "run_id", RUN_ID)
        out = os.path.join(a.new, "old_vs_new_" + fn)
        m.to_csv(out, index=False)
        print(f"== {fn}：舊 {len(o)} 列、新 {len(n)} 列、並排 {len(m)} 列 → {out}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("merge")
    p = sp.add_parser("stage")
    p.add_argument("--old", default=OLD)
    p.add_argument("--new", default=MERGED)
    p.add_argument("--dst")
    for c in ("ball", "side"):
        p = sp.add_parser(c)
        p.add_argument("--old", default=OLD)
        p.add_argument("--new", default=MERGED)
    a = ap.parse_args()
    {"merge": merge, "stage": stage, "ball": ball, "side": side}[a.cmd](a)


if __name__ == "__main__":
    main()
