"""main_p4.py — icpe-main：P4（W4 雜訊）的彙總表（PREREG 第 6 項、§3 B／C、偏離 6、偏離 8）。只彙總、只報格數，不下新的判定字。

  python -I code/icpe/main_p4.py --cells results/icpe_next/main/p4/p4_cells.csv.xz \
      --summary <family>=<prefix>_noise_summary.csv ... --perseed <family>=<prefix>_noise_headroom.csv ... \
      --metric mean_ret|p90_ret --exact-min <版本 B 的 mc 上限表> --out-dir <dir> --out-prefix <prefix> [--report-only]

輸入：next_sim_analyze.py 的雜訊輸出（每個雜訊設定、每個 seed 一列的 noise_headroom；seed 間的 noise_summary），
     和偏離 8 的選格清單（版本 B 上限 ≥15% 的每個（格, 元件），Mooncake 只列 R1／R2）。
使用者 2026-11-06 約 21:13 的決定（在 icpe-main，看任何 P4 數字之前；STATUS_MAIN 21:14 節）：
  1. 只報告格數：每種雜訊設定（模型, λ, 抽法）下，10 個 seed 的中位數仍 ≥15%／≥5% 的（格, 元件）數，
     加上 PREREG §3 B 要的 seed 間中位數、最小–最大、平均±標準差（逐格檔）。不下新的判定字。
  2. B1 主、B2 並列（主模型照 PREREG 是 P）。
  3. 佇列 seed：每個佇列 seed 分開為主（＝每個紀錄變體，例 semianalysis_N64_s1）；另報 3 × 10＝30 個值合起來的中位數與範圍。
     使用者 2026-11-07：只合「3 個佇列 seed 都有選到」的格（湊滿 30 個值；λ＝0 是 3 個值），湊不滿的另列格數。
  4. 全部到齊再算。
沿用：p90 只報告（--report-only）；寫入飽和照列另標（用版本 B 沒有雜訊時的標記）；Mooncake rule＝min 只報告；O-ALL(report) 只報告。
Mooncake（使用者 2026-11-07「R1、R2 分開算」）：summary／perseed 裡不能有 rule＝min；min 在這裡逐 seed 取 R1、R2 較小的上限，
  只算兩個規則都有選到的格，λ＝0 要和 --exact-min（版本 B）的 min 逐值相同。
檢查（不過就 exit 1）：
  - 雜訊設定剛好是 EXPECTED（P：B1 λ 0／.1／.2／.3／.4、B2 λ .1–.4；M：B1 λ 0–.3、B2 λ .1–.3），λ＝0 的 seed 數 1、其他 10。
  - 選格清單的每個（格, 元件）× 每個雜訊設定都有一列，n_seeds 對；逐 seed 檔的列數和 summary 一致。
  - mean_ret 時：summary 的 headroom_exact ＝ 選格清單的 headroom（版本 B）逐格完全相同。
輸出（每列有 run_id、ts、host）：
  <prefix>_cells.csv        逐（格, 元件, 雜訊設定）：seed 間中位數、最小、最大、平均、標準差、沒有雜訊的值
  <prefix>_by_trace.csv     紀錄變體 × rule × 元件 × 雜訊設定：格數、中位數 ≥15%／≥5% 的格數、各格中位數的範圍
  <prefix>_by_trace_wide.csv  同上，B1／B2 並排
  <prefix>_by_class.csv     到達方式 × 工具類 × 雜訊設定（只算判定元件、R1／R2）：格數加總、還有 ≥1 格中位數 ≥15% 的變體數
  <prefix>_pooled_cells.csv 3 × 10 合起來（CacheWise、SemiAnalysis）：每個（N, 格, 元件, 雜訊設定）一列
  <prefix>_pooled.csv       合起來的格數；湊不滿 3 個佇列 seed 的另列
  <prefix>_extra.csv        next_sim_analyze 算出來、但不在選格清單裡的列（例：Mooncake rule＝min）的數量
"""
from __future__ import annotations

import argparse
import os
import re
import socket
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from main_summary import classify  # noqa: E402

JUDGED = ["O-ORD", "O-DROP", "O-NOW", "O-PF", "O-GPU"]
THR, THR5 = 0.15, 0.05
CELL = ["trace", "rule", "cpu_GiB", "slow", "gpu", "restore", "gpu_GiB", "gpu_write", "slow_cap_frac"]
CC = CELL + ["component"]
SPEC = ["noise_model", "noise_draw", "noise_lambda"]
EXPECTED = {("P", "B1"): [0.0, 0.1, 0.2, 0.3, 0.4], ("P", "B2"): [0.1, 0.2, 0.3, 0.4],
            ("M", "B1"): [0.0, 0.1, 0.2, 0.3], ("M", "B2"): [0.1, 0.2, 0.3]}
QRE = re.compile(r"^(?:nb:)?(cachewise|semianalysis)_N(\d+)_s(\d+)$")


def readfam(specs):
    parts = []
    for s in specs:
        fam, path = s.split("=", 1)
        x = pd.read_csv(path)
        assert x.run_id.nunique() == 1, f"{path}：run_id 不只一個"
        x = x.rename(columns={"run_id": "src_analysis_run_id"}).drop(columns=["ts", "host"])
        x.insert(0, "family_input", fam)
        parts.append(x)
    X = pd.concat(parts, ignore_index=True)
    X["noise_lambda"] = X.noise_lambda.astype(float).round(6)
    return X


def fail(msgs):
    print("檢查不過，停：\n  " + "\n  ".join(msgs[:60]) + (f"\n  …共 {len(msgs)} 項" if len(msgs) > 60 else ""), file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--summary", nargs="+", required=True, metavar="FAMILY=PATH")
    ap.add_argument("--perseed", nargs="+", required=True, metavar="FAMILY=PATH")
    ap.add_argument("--metric", required=True, choices=["mean_ret", "p90_ret"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--exact-min", default="", help="版本 B 含 rule＝min 的上限表（Mooncake；和 --metric 同一個指標）")
    a = ap.parse_args()
    meta = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"),
                host=socket.gethostname(), metric=a.metric, report_only=a.report_only)

    P = pd.read_csv(a.cells)
    assert not P.duplicated(CC).any(), "選格清單有重複的（格, 元件）"
    assert set(P.rule) <= {"R1", "R2", "nb", "fi"} and "min" not in set(P.rule), f"選格清單的 rule 有 {sorted(set(P.rule))}"
    P = P[CC + ["headroom", "write_saturated"]].rename(columns={"headroom": "headroom_exact_verB",
                                                               "write_saturated": "write_saturated_verB"})
    NS, NH = readfam(a.summary), readfam(a.perseed)
    for X, nm in ((NS, "summary"), (NH, "perseed")):
        assert X.metric.eq(a.metric).all(), f"{nm} 的 metric 不是 {a.metric}"
        assert X.pf_target.eq("cpu").all() and X.dead_h.eq(0).all(), f"{nm}：pf_target／dead_h 不是 cpu／0"

    # 1. 雜訊設定剛好是 EXPECTED
    bad = []
    got = {(m, d): sorted(g.noise_lambda.unique()) for (m, d), g in NS.groupby(["noise_model", "noise_draw"])}
    if got != EXPECTED:
        bad.append(f"雜訊設定不是預期的：{got}（預期 {EXPECTED}）")
    exp_seeds = np.where(NS.noise_lambda == 0, 1, 10)
    if (NS.n_seeds.values != exp_seeds).any():
        k = NS[NS.n_seeds.values != exp_seeds]
        bad.append(f"n_seeds 不對的 {len(k)} 列；例 {k[CC + SPEC + ['n_seeds']].head(3).to_dict('records')}")
    nh_n = NH.groupby(CC + SPEC).size().rename("n_perseed")
    J = NS.join(nh_n, on=CC + SPEC)
    if (J.n_perseed != J.n_seeds).any():
        bad.append(f"逐 seed 檔的列數和 summary 不一致：{int((J.n_perseed != J.n_seeds).sum())} 列")
    if bad:
        fail(bad)

    # 2. 選格清單 × 雜訊設定 都要有
    specs = pd.DataFrame([dict(noise_model=m, noise_draw=d, noise_lambda=l) for (m, d), ls in EXPECTED.items() for l in ls])
    want = P.merge(specs, how="cross")
    M = want.merge(NS, on=CC + SPEC, how="left", indicator=True)
    miss = M[M._merge != "both"]
    if len(miss):
        fail([f"缺：{r}" for r in miss[CC + SPEC].head(60).to_dict("records")] + [f"共缺 {len(miss)} 列（選格清單 {len(P)} ×（雜訊設定 {len(specs)}））"])
    M = M.drop(columns="_merge")
    if a.metric == "mean_ret":
        d = (M.headroom_exact - M.headroom_exact_verB).abs().max()
        if not d == 0.0:
            fail([f"summary 的 headroom_exact 和選格清單（版本 B）不同：最大差 {d}"])

    # Mooncake rule＝min（只報告）。使用者 2026-11-07 選「R1、R2 分開算」：next_sim_analyze 每次只餵一個 rule，不產生 min 列
    # （一起餵時，只跑了一個規則的格會把那個規則當成 min，λ＝0 自檢不過；run 20261107-080102、診斷 20261107-081616）。
    # 這裡照 analyze() 的定義（每格每元件取 R1、R2 較小的上限；O-ALL(report) 沒有 min）逐 seed 算，只算兩個規則都有選到的格；
    # λ＝0 的每一格要和版本 B 的 min（--exact-min）完全相同，不同就停。
    if (NS.rule == "min").any():
        fail(["summary 裡有 rule＝min 的列：Mooncake 應該 R1、R2 分開算（使用者 2026-11-07）"])
    oc = [c for c in CC if c != "rule"]
    r12 = P[P.rule.isin(["R1", "R2"]) & (P.component != "O-ALL(report)")]
    both = r12.groupby(oc).rule.nunique()
    both = both[both == 2].reset_index().drop(columns="rule")
    mins = pd.DataFrame()
    if len(both):
        if not a.exact_min:
            fail(["有 R1／R2 的格，要給 --exact-min（版本 B 含 rule＝min 的上限表）"])
        X = NH[NH.rule.isin(["R1", "R2"])].merge(both, on=oc, how="inner")
        g = X.groupby(oc + SPEC + ["noise_seed"])
        if not (g.size() == 2).all():
            fail([f"rule＝min：有 {int((g.size() != 2).sum())} 個（格, 元件, 設定, seed）不是剛好 R1、R2 各一列"])
        per = g.agg(headroom=("headroom", "min"), family_input=("family_input", "first"),
                    src_analysis_run_id=("src_analysis_run_id", lambda x: ";".join(sorted(set(x))))).reset_index()
        per["rule"] = "min"
        E = pd.read_csv(a.exact_min)
        E = E[E.rule == "min"][CC + ["headroom"]].rename(columns={"headroom": "headroom_exact"})
        per = per.merge(E, on=CC, how="left")
        if per.headroom_exact.isna().any():
            fail([f"rule＝min：{int(per.headroom_exact.isna().sum())} 列在 --exact-min 裡找不到"])
        z = per[per.noise_lambda == 0]
        dz = (z.headroom - z.headroom_exact).abs().max() if len(z) else np.nan
        if not (len(z) and dz == 0.0):
            fail([f"rule＝min 的 λ＝0 自檢：{len(z)} 列，和版本 B 的 min 最大差 {dz}"])
        mins = per.groupby(["family_input", "rule"] + oc + SPEC).agg(
            n_seeds=("headroom", "size"), median=("headroom", "median"), min=("headroom", "min"), max=("headroom", "max"),
            mean=("headroom", "mean"), sd=("headroom", "std"), headroom_exact=("headroom_exact", "first"),
            src_analysis_run_id=("src_analysis_run_id", "first")).reset_index()
        ws = r12.groupby(oc).write_saturated_verB.any().rename("write_saturated_verB")
        mins = mins.join(ws, on=oc)
        mins["headroom_exact_verB"] = np.nan
        if len(mins) != len(both) * len(specs) or (mins.n_seeds.values != np.where(mins.noise_lambda == 0, 1, 10)).any():
            fail([f"rule＝min：應有 {len(both)} 個（格, 元件）× {len(specs)} 個設定＝{len(both) * len(specs)} 列、seed 數 1／10；實際 {len(mins)} 列"])
        print(f"rule＝min（只報告）：{len(both)} 個（格, 元件）；λ＝0 的 {len(z)} 列和版本 B 的 min 逐值相同")
    C = pd.concat([M, mins], ignore_index=True)

    # 3. 不在選格清單裡的列（只記數量）
    keyset = set(map(tuple, C[CC + SPEC].values.tolist()))
    ex = NS[[t not in keyset for t in map(tuple, NS[CC + SPEC].values.tolist())]]
    EX = ex.groupby(["family_input", "trace", "rule", "component"]).size().rename("n_rows").reset_index()

    cl = {t: classify(t) for t in C.trace.unique()}
    for i, nm in enumerate(["trace_class", "arrival", "arrival_detail", "tool_type"]):
        C[nm] = C.trace.map(lambda t: cl[t][i])
    C["judged"] = C.component.isin(JUDGED) & (C.rule != "min")
    C["med_ge15"] = C["median"] >= THR
    C["med_ge5"] = C["median"] >= THR5
    C["exact_ge15"] = C.headroom_exact >= THR

    # 4. 逐變體
    gk = ["family_input", "trace_class", "arrival", "arrival_detail", "tool_type", "trace", "rule", "component", "judged"] + SPEC
    BT = C.groupby(gk, sort=True).apply(lambda g: pd.Series(dict(
        n_cellcomp=len(g), n_exact_ge15=int(g.exact_ge15.sum()),
        n_med_ge15=int(g.med_ge15.sum()), n_med_ge5=int(g.med_ge5.sum()),
        n_med_ge15_write_saturated_verB=int((g.med_ge15 & g.write_saturated_verB.astype(bool)).sum()),
        min_of_medians=g["median"].min(), median_of_medians=g["median"].median(), max_of_medians=g["median"].max(),
        max_sd=g["sd"].max(), min_exact=g.headroom_exact.min(), max_exact=g.headroom_exact.max(),
        src_analysis_run_id=";".join(sorted(g.src_analysis_run_id.unique())))), include_groups=False).reset_index()
    for c in ["n_cellcomp", "n_exact_ge15", "n_med_ge15", "n_med_ge5", "n_med_ge15_write_saturated_verB"]:
        BT[c] = BT[c].astype(int)
    W = BT.pivot_table(index=[c for c in gk if c != "noise_draw"], columns="noise_draw",
                       values=["n_cellcomp", "n_med_ge15", "n_med_ge5", "median_of_medians", "max_of_medians"], aggfunc="first")
    W.columns = [f"{v}_{d}" for v, d in W.columns]
    W = W.reset_index()

    # 5. 到達方式 × 工具類（只算判定元件、R1／R2；格數加總；還有 ≥1 格中位數 ≥15% 的變體數）
    rows = []
    for k, g in BT[BT.judged].groupby(["arrival", "tool_type"] + SPEC, sort=True):
        v = g.groupby(["trace", "rule"]).agg(n=("n_cellcomp", "sum"), n15=("n_med_ge15", "sum"))
        hit = v[v.n15 > 0]
        rows.append(dict(zip(["arrival", "tool_type"] + SPEC, k), n_variants_in_p4=len(v), n_cellcomp=int(g.n_cellcomp.sum()),
                         n_med_ge15=int(g.n_med_ge15.sum()), n_med_ge5=int(g.n_med_ge5.sum()),
                         n_variants_with_med_ge15=len(hit), variants_with_med_ge15=";".join(f"{t}/{r}:{int(n)}" for (t, r), n in hit.n15.items()),
                         max_of_medians=g.max_of_medians.max()))
    BC = pd.DataFrame(rows)

    # 6. 3 × 10 合起來（CacheWise、SemiAnalysis）
    Q = C[C.trace.str.match(QRE) & (C.rule != "min")].copy()
    Q["qfam"] = Q.trace.map(lambda t: QRE.match(t).group(1))
    Q["N"] = Q.trace.map(lambda t: int(QRE.match(t).group(2)))
    Q["qseed"] = Q.trace.map(lambda t: int(QRE.match(t).group(3)))
    pk = ["family_input", "trace_class", "qfam", "N"] + [c for c in CC if c != "trace"] + SPEC
    nq = Q.groupby(pk).qseed.agg(lambda s: tuple(sorted(s))).rename("qseeds").reset_index()
    full = nq[nq.qseeds == (1, 2, 3)]
    part = nq[nq.qseeds != (1, 2, 3)]
    NHk = NH[["family_input"] + CC + SPEC + ["noise_seed", "headroom"]].merge(
        Q[["family_input", "trace_class", "qfam", "N", "qseed"] + CC + SPEC], on=["family_input"] + CC + SPEC, how="inner")  # 只用選到的（格, 元件）
    NHk = NHk.merge(full[pk], on=pk, how="inner")
    exq = Q.merge(full[pk], on=pk, how="inner").groupby(pk).headroom_exact.agg(exact_min="min", exact_max="max").reset_index()
    PC = NHk.groupby(pk).headroom.agg(n_values="size", median="median", min="min", max="max", mean="mean", sd="std").reset_index()
    PC = PC.merge(exq, on=pk, how="left")
    exp_n = np.where(PC.noise_lambda == 0, 3, 30)
    if len(PC) != len(full) or (PC.n_values.values != exp_n).any():
        fail([f"3 × 10 合起來：{len(full)} 個 key 應有 {len(full)} 列、每列 30（λ＝0 是 3）個值；實際 {len(PC)} 列，"
              f"值數不對 {int((PC.n_values.values != exp_n).sum())} 列"])
    PC["judged"] = PC.component.isin(JUDGED)
    rows = []
    gk2 = ["family_input", "trace_class", "qfam", "N", "component"] + SPEC
    for k, g in nq.groupby(gk2, sort=True):
        f = PC.merge(g[pk], on=pk, how="inner")
        rows.append(dict(zip(gk2, k), judged=k[4] in JUDGED, n_keys_full=len(f), n_pooled_med_ge15=int((f["median"] >= THR).sum()),
                         n_pooled_med_ge5=int((f["median"] >= THR5).sum()),
                         max_pooled_median=(f["median"].max() if len(f) else np.nan),
                         n_keys_partial_1qs=int((g.qseeds.map(len) == 1).sum()), n_keys_partial_2qs=int((g.qseeds.map(len) == 2).sum())))
    PS = pd.DataFrame(rows)

    os.makedirs(a.out_dir, exist_ok=True)
    cells_out = C[["family_input", "trace_class", "arrival", "tool_type", "judged"] + CC + SPEC +
                  ["n_seeds", "median", "min", "max", "mean", "sd", "headroom_exact", "write_saturated_verB", "src_analysis_run_id"]]
    for name, X in (("cells", cells_out), ("by_trace", BT), ("by_trace_wide", W), ("by_class", BC),
                    ("pooled_cells", PC), ("pooled", PS), ("extra", EX)):
        X = X.copy()
        for i, (k, v) in enumerate(meta.items()):
            X.insert(i, k, v)
        X.to_csv(os.path.join(a.out_dir, f"{a.out_prefix}_{name}.csv"), index=False)
    print(f"檢查 PASS：選格清單 {len(P)} 個（格, 元件）× {len(specs)} 個雜訊設定＝{len(M)} 列都有，n_seeds 對"
          + ("，headroom_exact＝版本 B" if a.metric == "mean_ret" else "")
          + f"；rule＝min（只報告）{len(mins)} 列；不在清單的 {len(ex)} 列（{EX.groupby('rule').n_rows.sum().to_dict() if len(EX) else {}}）")
    print(f"3 × 10：湊滿的 key {len(full)} 個、湊不滿 {len(part)} 個")
    print(f"→ {a.out_dir}/{a.out_prefix}_*.csv")


if __name__ == "__main__":
    main()
