"""main_summary.py — icpe-main：把各類紀錄的 next_sim_analyze.py 上限表合併成正式彙總表與判定（PREREG §1、§4）。

  python -I code/icpe/main_summary.py --version <標籤> --headroom <family>=<headroom_components.csv> ... \
      --sweep <x1_sweep.csv.xz> ... --out-dir <dir> --out-prefix <prefix> [--expect-cells 54] [--verdict-exclude O-PF]

只做彙總與套用登記的門檻，不重算任何上限（上限＝next_sim_analyze.py 的 headroom 欄）。

判定的規則（PREREG §1；PREREG 沒寫到的部分由使用者 2026-11-06 在 icpe-main 決定，見 STATUS_MAIN）：
  - 上限 ≥15% 才算值得做（GE15）；≥5% 只列格數。
  - 算判定的元件＝O-ORD、O-DROP、O-NOW、O-PF、O-GPU（PREREG §4 的 ④ 元件＋GPU 層的完美淘汰）；O-ALL(report) 只報告。
  - 原始到達與合成到達各下一個判定字，不合併（使用者 11/6：「原始／合成各下一個判定字」；PREREG §4「分開報」）。
  - Mooncake R1、R2 各自下判定；rule＝min 的列只報告（使用者 11/6：「R1、R2 各自下判定」）。
  - 寫入頻寬飽和的格子照算，另外標格數（使用者 11/6：「照算、另外標格數」）。
  - 推翻條件＝「真實的工具類紀錄上限 ≥15%」且「沒有簡單線索拿走大部分」。簡單線索（W3）沒登記也沒跑 → NOT_MEASURED，
    所以 ≥15% 成立時推翻條件只能寫「未定」；≥15% 不成立時推翻條件不成立。
  - 工具類＝NEXT_PLAN W2／§5.2 的分法：TraceLab、CacheWise、SemiAnalysis（coding agent）、FreeInference（agent）、
    Mooncake toolagent；Mooncake conversation（聊天）與 Bailian（聊天／API 服務，NEXT_PLAN §5.2「不是工具 agent」）只報告。
輸出（每列有 run_id、ts、host）：
  <prefix>_by_trace.csv     紀錄變體 × rule × 元件：格數、最大、中位數、最小、≥15%／≥5% 格數、其中寫入飽和的格數、最大值那格
  <prefix>_by_cpu.csv       再細分到 CPU 容量（每個容量 6 格＝3 慢層 × GPU 閒／忙），附 cpu_frac_ws（PREREG §4 敏感度 2）
  <prefix>_verdict.csv      level＝variant（每個變體）與 level＝class（到達方式 × 工具類）的判定
"""
from __future__ import annotations

import argparse
import os
import re
import socket
from datetime import datetime

import pandas as pd

JUDGED = ["O-ORD", "O-DROP", "O-NOW", "O-PF", "O-GPU"]
REPORT_ONLY = ["O-ALL(report)"]
THR, THR5 = 0.15, 0.05
CELL = ["trace", "rule", "cpu_GiB", "slow", "gpu", "restore", "gpu_GiB", "gpu_write", "slow_cap_frac"]
CHAT = {"sharegpt": "ShareGPT", "lmsys": "LMSYS-Chat-1M", "ultrachat": "UltraChat"}


def classify(trace):
    """(紀錄類別, 到達方式, 到達細節, 工具類?)；不認得就停（CLAUDE.md 規則 7：查不到不等於沒有）。"""
    t = trace.split(":", 1)[1] if trace.startswith("nb:") else trace
    if t.startswith("cachewise_N"):
        n = t.split("_")[1]
        return "CacheWise", "synthetic", f"fixed_concurrency_{n}", True
    if t.startswith("semianalysis_N"):
        n = t.split("_")[1]
        return "SemiAnalysis", "synthetic", f"fixed_concurrency_{n}", True
    if t in ("tracelab_c1", "tracelab_c2", "tracelab_c4"):
        c = int(t[-1])
        return "TraceLab", ("raw" if c == 1 else "synthetic"), ("original" if c == 1 else f"start_compressed_{c}x"), True
    if t.startswith("bailian_"):
        return "Bailian", "raw", "original", False
    if t == "conversation":
        return "Mooncake", "raw", "original", False
    if t == "toolagent":
        return "Mooncake", "raw", "original", True
    if trace.startswith("fi:"):
        return "FreeInference", "raw", "original", True
    # agent 軌跡（偏離 17＋更正）：固定並行 → 合成到達；工具類、下判定（使用者 2026-11-07 14:0x，在 icpe-main 用 AskUserQuestion）
    if t.startswith("miniswe_N"):
        return "mini-SWE-agent", "synthetic", f"fixed_concurrency_{t.split('_')[1]}", True
    if t.startswith("openhands_N"):
        return "OpenHands", "synthetic", f"fixed_concurrency_{t.split('_')[1]}", True
    # 對話資料集（偏離 13）：對話開始都是 Poisson → 合成到達；聊天＝非工具類、只報告（偏離 7 與 Bailian／Mooncake conversation 的先例；
    # STATUS_MAIN 11:33）。到達細節照登記的標籤：T60／T600＝合成思考時間；WildChat＝真實的對話內間隔。
    m = re.match(r"^(sharegpt|lmsys|ultrachat)_T(60|600)_s\d+$", t)
    if m:
        return CHAT[m.group(1)], "synthetic", f"poisson_start_synthetic_think_T{m.group(2)}", False
    if re.match(r"^wildchat_s\d+$", t):
        return "WildChat-1M", "synthetic", "poisson_start_real_intra_gaps", False
    raise ValueError(f"不認得的紀錄 {trace!r}：先在 classify() 寫明它的到達方式與類別")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--headroom", nargs="+", required=True, metavar="FAMILY=PATH")
    ap.add_argument("--sweep", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--expect-cells", type=int, default=54)
    ap.add_argument("--report-only", action="store_true", help="只出 by_trace／by_cpu、不下判定（例：p90，PREREG §1 只報告）")
    ap.add_argument("--verdict-exclude", nargs="*", default=[], metavar="COMPONENT",
                    help="判定時不用這些元件（by_trace 照列）；例：O-PF 待偏離 14 的修正版取代時，看判定靠不靠它（使用者 2026-11-07）")
    ap.add_argument("--trace-regex", default="", metavar="REGEX",
                    help="只用 trace 符合的列（re.search）；例：對話資料集的 T600 只報告、和 T60 分開跑（使用者 2026-11-07 11:33）")
    a = ap.parse_args()
    meta = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"),
                host=socket.gethostname(), version=a.version)
    if a.verdict_exclude:
        assert set(a.verdict_exclude) <= set(JUDGED), f"--verdict-exclude 只能是 {JUDGED}"
        meta["verdict_exclude"] = ";".join(a.verdict_exclude)      # 沒給就不加這欄（和之前的輸出逐欄相同）

    parts = []
    for spec in a.headroom:
        fam, path = spec.split("=", 1)
        h = pd.read_csv(path)
        assert h.run_id.nunique() == 1, f"{path}：run_id 不只一個"
        h = h.rename(columns={"run_id": "src_analysis_run_id"}).drop(columns=["ts", "host"])
        h.insert(0, "family_input", fam)
        h["src_file"] = path
        parts.append(h)
    H = pd.concat(parts, ignore_index=True)
    if a.trace_regex:                                              # 沒給就不加這欄（和之前的輸出逐欄相同）
        keep = H.trace.str.contains(a.trace_regex, regex=True)
        assert keep.any(), f"--trace-regex {a.trace_regex!r} 沒有符合的 trace"
        print(f"--trace-regex {a.trace_regex!r}：用 {H.trace[keep].nunique()} 個 trace、拿掉 {H.trace[~keep].nunique()} 個")
        H = H[keep].reset_index(drop=True)
        meta["trace_regex"] = a.trace_regex
    dup = H.duplicated(CELL + ["component"]).sum()
    assert dup == 0, f"(格, 元件) 有 {dup} 列重複：同一個紀錄放進了兩份輸入？"
    cl = {t: classify(t) for t in H.trace.unique()}
    H["trace_class"] = H.trace.map(lambda t: cl[t][0])
    H["arrival"] = H.trace.map(lambda t: cl[t][1])
    H["arrival_detail"] = H.trace.map(lambda t: cl[t][2])
    H["tool_type"] = H.trace.map(lambda t: cl[t][3])

    # 格數自檢：每個 (trace, rule) 的每個元件都要剛好 expect_cells 格；rule＝min 沒有 O-ALL(report)（同 next_sim_analyze）
    bad = []
    for (tr, ru), g in H.groupby(["trace", "rule"]):
        want = JUDGED + ([] if ru == "min" else REPORT_ONLY)
        for c in want:
            n = int((g.component == c).sum())
            if n != a.expect_cells:
                bad.append(f"{tr}/{ru}/{c}: {n} 格（應為 {a.expect_cells}）")
        extra = set(g.component) - set(want)
        if extra:
            bad.append(f"{tr}/{ru}: 多出元件 {sorted(extra)}")
    assert not bad, "格數自檢失敗（next_sim_analyze 會在缺組合時默默跳過元件）：\n  " + "\n  ".join(bad)

    # cpu_frac_ws：每個 (trace, rule, cpu_GiB) 只能有一個值
    S = pd.concat([pd.read_csv(p, usecols=["trace", "rule", "cpu_GiB", "cpu_frac_ws"]) for p in a.sweep], ignore_index=True)
    S = S.drop_duplicates()
    assert not S.duplicated(["trace", "rule", "cpu_GiB"]).any(), "cpu_frac_ws 在同一個 (trace, rule, cpu_GiB) 有多個值"

    key = ["trace_class", "trace", "rule", "arrival", "arrival_detail", "tool_type", "component"]
    rows = []
    for k, g in H.groupby(key, sort=True):
        i = g.headroom.idxmax()
        ge15 = g.headroom >= THR
        rows.append(dict(zip(key, k), judged=k[-1] in JUDGED,
                         verdict_use=(k[-1] in JUDGED and k[2] != "min" and k[-1] not in a.verdict_exclude),
                         n_cells=len(g), max=g.headroom.max(), median=g.headroom.median(), min=g.headroom.min(),
                         n_ge15=int(ge15.sum()), n_ge5=int((g.headroom >= THR5).sum()),
                         n_ge15_write_saturated=int((ge15 & g.write_saturated.astype(bool)).sum()),
                         n_write_saturated=int(g.write_saturated.astype(bool).sum()),
                         argmax_cpu_GiB=g.at[i, "cpu_GiB"], argmax_slow=g.at[i, "slow"], argmax_gpu=g.at[i, "gpu"],
                         argmax_oracle=f"{g.at[i, 'oracle_policy']}|{g.at[i, 'oracle_gpu_policy']}",
                         argmax_rival=f"{g.at[i, 'best_rival']}|{g.at[i, 'best_rival_gpu_policy']}",
                         src_analysis_run_id=g.src_analysis_run_id.iloc[0], src_file=g.src_file.iloc[0]))
    BT = pd.DataFrame(rows)

    rows = []
    for k, g in H.groupby(key + ["cpu_GiB"], sort=True):
        tr, ru, cpu = k[1], k[2], k[-1]
        f = S[(S.trace == tr) & (S.rule == ru) & (S.cpu_GiB == cpu)].cpu_frac_ws
        rows.append(dict(zip(key + ["cpu_GiB"], k), cpu_frac_ws=(float(f.iloc[0]) if len(f) == 1 else float("nan")),
                         n_cells=len(g), max=g.headroom.max(), median=g.headroom.median(), min=g.headroom.min(),
                         n_ge15=int((g.headroom >= THR).sum()), n_ge5=int((g.headroom >= THR5).sum()),
                         src_analysis_run_id=g.src_analysis_run_id.iloc[0]))
    BC = pd.DataFrame(rows)

    # 判定
    V = []
    U = BT[BT.verdict_use]
    for k, g in U.groupby(["trace_class", "trace", "rule", "arrival", "arrival_detail", "tool_type"], sort=True):
        hit = g[g.n_ge15 > 0]
        V.append(dict(level="variant", trace_class=k[0], trace=k[1], rule=k[2], arrival=k[3], arrival_detail=k[4], tool_type=k[5],
                      max_judged=g["max"].max(), argmax_component=g.loc[g["max"].idxmax(), "component"],
                      n_ge15_cells_judged=int(g.n_ge15.sum()),
                      ge15_components=";".join(f"{r.component}:{r.n_ge15}" for r in hit.itertuples()),
                      ge15="YES" if len(hit) else "NO"))
    VV = pd.DataFrame(V)
    for arr in ("raw", "synthetic"):
        for tool in (True, False):
            g = VV[(VV.arrival == arr) & (VV.tool_type == tool)]
            if not len(g):
                continue
            hit = g[g.ge15 == "YES"]
            cond = "成立" if len(hit) else "不成立"
            if tool:
                fals = "未定：≥15% 成立，但簡單線索（W3）NOT_MEASURED" if len(hit) else "不成立（≥15% 不成立）"
            else:
                fals = "不適用（不是工具類紀錄，只報告）"
            V.append(dict(level="class", arrival=arr, tool_type=tool, n_variants=len(g), n_variants_ge15=len(hit),
                          variants_ge15=";".join(f"{r.trace}/{r.rule}" for r in hit.itertuples()),
                          max_judged=g.max_judged.max(), ge15=cond, falsification=fals))
    VV = pd.DataFrame(V)

    os.makedirs(a.out_dir, exist_ok=True)
    outs = [("by_trace", BT), ("by_cpu", BC)] + ([] if a.report_only else [("verdict", VV)])
    if a.report_only:
        BT["verdict_use"] = False
    for name, X in outs:
        X = X.copy()
        for i, (k, v) in enumerate(meta.items()):
            X.insert(i, k, v)
        X.to_csv(os.path.join(a.out_dir, f"{a.out_prefix}_{name}.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(f"格數自檢 PASS：{H.groupby(['trace', 'rule']).ngroups} 個 (trace, rule)，每個元件 {a.expect_cells} 格")
    if not a.report_only:
        print(VV[VV.level == "class"][["arrival", "tool_type", "n_variants", "n_variants_ge15", "max_judged", "ge15", "falsification"]].to_string(index=False))
    print(f"→ {a.out_dir}/{a.out_prefix}_*.csv：{[n for n, _ in outs]}（{len(BT)}／{len(BC)}／{len(VV)} 列）")


if __name__ == "__main__":
    main()
