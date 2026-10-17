"""crosscheck.py — 統一表 vs X1–X9 的 loader（code/p3x/，只讀它們的輸出 CSV，不改、不執行它們）。

  python -I code/p3_data/crosscheck.py      → results/p3_data/crosscheck.csv

每一列：同一份資料、同一個量，統一表算出來的值（unified）vs X 系列 CSV 裡的值（x_value，附 X 的 run_id 與檔名）。
X 的值一律從 CSV 讀，不手抄。定義不同的量會在 note 寫清楚（例如 X2 的「下一輪」＝同 session 依時間的下一個請求，
統一表的「下一輪」＝最早的子請求；兩者只在有分叉的樹狀 session 不同）。統一表這邊也另外算一次「X 的定義」，證明差異只來自定義。
"""
from __future__ import annotations

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import schema as S  # noqa: E402

X = os.path.join(S.REPO, "results", "p3_explore")
OUT = os.path.join(S.REPO, "results", "p3_data", "crosscheck.csv")
rows = []


def xval(path, where, col):
    """X 的 CSV 裡符合 where 的那一列的 col（必須剛好一列；最後一個 run 優先）。回傳 (值, run_id)。"""
    d = pd.read_csv(os.path.join(X, path), dtype=str)
    m = np.ones(len(d), bool)
    for k, v in where.items():
        m &= d[k].values == v
    d = d[m]
    if len(d) == 0:
        return None, None
    r = d.iloc[-1]
    return r[col], r["run_id"]


def pctl_str(s, q):
    """X3 的分位數字串 'p1=.. p5=..' 取 pq。"""
    for t in str(s).split():
        k, v = t.split("=")
        if k == f"p{q}":
            return float(v)
    return None


def put(source, variant, metric, uval, xv, xref, xfile, note="", tol=1e-9, rel=False):
    try:
        xf = float(xv)
    except (TypeError, ValueError):
        xf = float("nan")
    if xv is None or np.isnan(xf):
        match = "NO_X_VALUE"
    else:
        d = abs(float(uval) - xf)
        lim = tol * max(1.0, abs(xf)) if rel else tol
        match = "MATCH" if d <= lim else "DIFF"
    rows.append(dict(run_id=os.environ.get("RUN_ID", ""), ts=S.now_iso(), source=source, variant=variant, metric=metric,
                     unified=uval, x_value=xv, match=match, x_run_id=xref, x_file=xfile, note=note))


def timechain(df):
    """X2 的定義：session 內依 (t_start, turn_idx) 的下一個請求。回傳 (has_next 比例, 間隔秒 array)。"""
    d = df.sort_values(["session_id", "t_start_ms", "turn_idx"], kind="mergesort")
    nxt = d.groupby("session_id")["t_start_ms"].shift(-1)
    g = (nxt - d["t_start_ms"]).dropna().values / 1000.0
    return float(nxt.notna().mean()), g


def linkgaps(df):
    """X3 的定義：每一條父子連結，子 t_start − 父 t_start（秒）。"""
    key = df.set_index(["session_id", "turn_idx"])["t_start_ms"]
    c = df[df["parent_turn_idx"].notna()]
    pt = key.reindex(pd.MultiIndex.from_arrays([c["session_id"], c["parent_turn_idx"].astype(int)])).values
    return (c["t_start_ms"].values - pt) / 1000.0


def q(a, p):
    return float(np.percentile(np.asarray(a, float), p))


COLS = ["session_id", "turn_idx", "parent_turn_idx", "n_children", "trigger", "next_trigger", "t_start_ms", "t_end_ms",
        "think_ms", "gap_ms", "censored"]


def main():
    # ------------------------------------------------------------ Bailian：X2 x2_summary、X3 x3_checks_bailian
    for v, x2n, x3n in (("traceA", "bailian_traceA", "traceA"), ("thinking", "bailian_thinking", "thinking"),
                        ("coder", "bailian_coder", "coder")):
        df = S.read_unified("bailian", v, COLS)
        f = "x2/x2_summary.csv"
        for met, uv in (("n_requests", len(df)), ("n_sessions", df["session_id"].nunique()),
                        ("n_sessions_ge2_requests", int((df.groupby("session_id").size() >= 2).sum())),
                        ("frac_episodes_censored", float(df["censored"].mean()))):
            xv, xr = xval(f, dict(trace=x2n, metric=met), "value")
            put("bailian", v, met, uv, xv, xr, f, tol=1e-6)
        g = df["gap_ms"].dropna().values / 1000.0
        for p in (10, 50, 90, 99):
            xv, xr = xval(f, dict(trace=x2n, metric=f"gap_returned_p{p}_s"), "value")
            put("bailian", v, f"gap_returned_p{p}_s", q(g, p), xv, xr, f, "子 t_start − 父 t_start（沒有分叉，所以＝X2 的定義）", tol=1e-3)
        f3 = "x3/x3_checks_bailian.csv"
        xv, xr = xval(f3, dict(trace=x3n, check="return_events"), "value")
        put("bailian", v, "n_links(return_events)", int(df["parent_turn_idx"].notna().sum()), xv, xr, f3)
    # ------------------------------------------------------------ Mooncake
    for v, x2n, x3f, x3n, x4n in (("conv-rule-d4", "mooncake_conversation", "x3/x3_checks_mooncake_loose.csv", "conversation", "conversation"),
                                  ("conv-rule-strict", "mooncake_conversation_lit", "x3/x3_checks_mooncake_strict.csv", "conversation", "conversation"),
                                  ("tool-rule-d4", None, "x3/x3_checks_mooncake_loose.csv", "toolagent", "toolagent"),
                                  ("tool-rule-strict", None, "x3/x3_checks_mooncake_strict.csv", "toolagent", "toolagent"),
                                  ("conv-rule-loose", None, None, None, None), ("tool-rule-loose", None, None, None, None)):
        df = S.read_unified("mooncake", v, COLS)
        ns = df["session_id"].nunique()
        hc, g_tc = timechain(df)
        lg = linkgaps(df)
        if x2n:
            f = "x2/x2_summary.csv"
            xv, xr = xval(f, dict(trace=x2n, metric="n_sessions"), "value")
            put("mooncake", v, "n_sessions", ns, xv, xr, f)
            xv, xr = xval(f, dict(trace=x2n, metric="frac_episodes_censored"), "value")
            put("mooncake", v, "frac_censored (X2 time-chain def, recomputed)", 1 - hc, xv, xr, f, "X2：session 內依時間的下一個請求", tol=1e-6)
            put("mooncake", v, "frac_censored (unified tree def: no child)", float(df["censored"].mean()), xv, xr, f,
                "統一表：沒有子請求才算設限；和 X2 不同是因為分叉", tol=1e-6)
            for p in (10, 50, 90, 99):
                xv, xr = xval(f, dict(trace=x2n, metric=f"gap_returned_p{p}_s"), "value")
                put("mooncake", v, f"gap_p{p}_s (X2 time-chain def, recomputed)", q(g_tc, p), xv, xr, f, tol=1e-3)
        if x3f:
            xv, xr = xval(x3f, dict(trace=x3n, check="return_events"), "value")
            put("mooncake", v, "n_links(return_events)", len(lg), xv, xr, x3f)
            xv, xr = xval(x3f, dict(trace=x3n, check="sessions_with_return"), "value")
            put("mooncake", v, "n_sessions_ge2(sessions_with_return)", int((df.groupby("session_id").size() >= 2).sum()), xv, xr, x3f)
            xv, xr = xval(x3f, dict(trace=x3n, check="gap_s_pctl"), "value")
            for p in (10, 50, 90, 99):
                put("mooncake", v, f"link_gap_p{p}_s", q(lg, p), pctl_str(xv, p), xr, x3f, "每條父子連結的間隔（X3 的定義）",
                    tol=0.6)
        if x4n:
            f4 = "x4/x4_trace_desc.csv"
            if "strict" in v:
                xv, xr = xval(f4, dict(trace=x4n), "n_units_strict")
                put("mooncake", v, "n_sessions", ns, xv, xr, f4)
                xv, xr = xval(f4, dict(trace=x4n), "frac_req_with_strict_child")
                put("mooncake", v, "frac_has_child", float((df["n_children"] > 0).mean()), xv, xr, f4, tol=1e-9)
            else:
                xv, xr = xval(f4, dict(trace=x4n), "n_sessions")
                put("mooncake", v, "n_sessions", ns, xv, xr, f4)
                xv, xr = xval(f4, dict(trace=x4n), "n_pairs")
                put("mooncake", v, "n_links", len(lg), xv, xr, f4)
                xv, xr = xval(f4, dict(trace=x4n), "frac_returning")
                put("mooncake", v, "frac_returning (X4 = time-chain def, recomputed)", hc, xv, xr, f4, tol=1e-9)
        if v.endswith("d4"):
            fx = "x9/t1_census.csv"
            xv, xr = xval(fx, dict(trace=x2n or ("mooncake_" + x3n)), "n_edges") if x3n else (None, None)
            xv, xr = xval(fx, dict(trace="mooncake_" + x3n), "n_edges")
            put("mooncake", v, "n_links vs X9 census n_edges", len(lg), xv, xr, fx, "X9 的父規則和 D4 不完全相同（見 03b §4）")
            xv, xr = xval(fx, dict(trace="mooncake_" + x3n), "n_roots")
            put("mooncake", v, "n_sessions vs X9 census n_roots", ns, xv, xr, fx)
        if v.endswith("loose"):
            put("mooncake", v, "frac_has_child (vs research/calc lit loose)", float((df["n_children"] > 0).mean()),
                {"conv-rule-loose": 0.4393649738176378, "tool-rule-loose": 0.4407827854964419}[v], "lit-agent",
                "docs/phase3_20261017/research/calc/mooncake_" + ("conversation" if v.startswith("conv") else "toolagent") + ".txt")
    # ------------------------------------------------------------ FreeInference：X3 x3_checks_freeinference、X9 fi_t1_census／fi_t3_window_censoring
    fi = pd.concat([S.read_unified("freeinference", f"week{i:02d}", COLS) for i in range(1, 17)], ignore_index=True)
    f9 = "x9/fi_t1_census.csv"
    for met, uv, col in (("n_requests", len(fi), "n_req"), ("n_sessions", fi["session_id"].nunique(), "n_sessions")):
        xv, xr = xval(f9, dict(dataset="freeinference"), col)
        put("freeinference", "ALL", met, uv, xv, xr, f9)
    u = fi[fi["next_trigger"] == "user"]
    t = fi[fi["next_trigger"] == "tool"]
    f9w = "x9/fi_t3_window_censoring.csv"
    xv, xr = xval(f9w, dict(dataset="freeinference", variant="user_returns_only", window="1h"), "n_finite_gaps")
    put("freeinference", "ALL", "n_user_returns (incl. negative)", len(u), xv, xr, f9w)
    xv, xr = xval(f9w, dict(dataset="freeinference", variant="user_returns_only", window="1h"), "gap_median_s")
    put("freeinference", "ALL", "user think median s (X9 clips negatives to 0)", float(np.median(np.maximum(u["think_ms"].astype(float), 0)) / 1000),
        xv, xr, f9w, tol=1e-3)
    xv, xr = xval(f9w, dict(dataset="freeinference", variant="all_requests", window="1h"), "n_finite_gaps")
    put("freeinference", "ALL", "n_returns all (user+tool)", int(fi["think_ms"].notna().sum()), xv, xr, f9w)
    f3 = "x3/x3_checks_freeinference.csv"
    un = u[u["think_ms"] >= 0]
    tn = t[t["think_ms"] >= 0]
    xv, xr = xval(f3, dict(trace="freeinference", check="user_return_events"), "value")
    put("freeinference", "ALL", "n_user_returns (X3 excludes negative)", len(un), xv, xr, f3)
    xv, xr = xval(f3, dict(trace="freeinference", check="tool_return_events"), "value")
    put("freeinference", "ALL", "n_tool_returns (X3 excludes negative)", len(tn), xv, xr, f3)
    xv, xr = xval(f3, dict(trace="freeinference", check="negative_gap_excluded"), "value")
    put("freeinference", "ALL", "n negative gaps", int((fi["think_ms"] < 0).sum()), xv, xr, f3)
    xv, xr = xval(f3, dict(trace="freeinference", check="user_think_s_pctl"), "value")
    for p in (10, 50, 90, 99):
        put("freeinference", "ALL", f"user_think_p{p}_s", q(un["think_ms"] / 1000, p), pctl_str(xv, p), xr, f3, tol=5e-4, rel=True)
    xv, xr = xval(f3, dict(trace="freeinference", check="tool_lead_s_pctl"), "value")
    for p in (10, 50, 90, 99):
        put("freeinference", "ALL", f"tool_gap_p{p}_s", q(tn["think_ms"] / 1000, p), pctl_str(xv, p), xr, f3, tol=5e-4, rel=True)
    xv, xr = xval(f3, dict(trace="freeinference", check="user_sessions"), "value")
    put("freeinference", "ALL", "sessions with a user return", int(u["session_id"].nunique()), xv, xr, f3,
        "X3 用 week/session_id 當鍵；session_id 跨週不重複時兩者相同")
    xv, xr = xval(f3, dict(trace="freeinference", check="tool_sessions"), "value")
    put("freeinference", "ALL", "sessions with a tool return", int(t["session_id"].nunique()), xv, xr, f3)
    del fi
    # ------------------------------------------------------------ WildChat：X2、X9
    wc = S.read_unified("wildchat", "1m", ["session_id", "turn_idx", "t_end_ms", "censored"])
    f = "x2/x2_summary.csv"
    for met, uv in (("n_requests", len(wc)), ("n_sessions", wc["session_id"].nunique()),
                    ("frac_episodes_censored", float(wc["censored"].mean()))):
        xv, xr = xval(f, dict(trace="wildchat_1m", metric=met), "value")
        put("wildchat", "1m", met, uv, xv, xr, f, "統一表另外丟掉不交替的對話（X2 只丟時間倒退的）", tol=1e-6)
    d = wc.sort_values(["session_id", "turn_idx"])
    g = (d.groupby("session_id")["t_end_ms"].shift(-1) - d["t_end_ms"]).dropna().values / 1000.0
    for p in (10, 50, 90, 99):
        xv, xr = xval(f, dict(trace="wildchat_1m", metric=f"gap_returned_p{p}_s"), "value")
        put("wildchat", "1m", f"assistant_to_assistant_gap_p{p}_s (X2 def)", q(g, p), xv, xr, f, tol=1e-3)
    f9 = "x9/t3_window_censoring.csv"
    xv, xr = xval(f9, dict(dataset="wildchat-1m", window="1h"), "n_finite_gaps")
    put("wildchat", "1m", "n_returns", len(g), xv, xr, f9)
    del wc, d
    # ------------------------------------------------------------ CacheWise：X2 cachewise2、X3
    cw = S.read_unified("cachewise", "catraces", COLS)
    f = "x2/x2_summary.csv"
    for met, uv in (("n_requests", len(cw)), ("n_sessions", cw["session_id"].nunique())):
        xv, xr = xval(f, dict(trace="cachewise", metric=met), "value")
        put("cachewise", "catraces", met, uv, xv, xr, f, "統一表只取主線（isSidechain=False）；X2 用 events.json（含 sidechain）")
    f3 = "x3/x3_checks_cachewise.csv"
    h = cw[cw["next_trigger"] == "user"]
    xv, xr = xval(f3, dict(trace="cachewise", check="human_prompt_events"), "value")
    put("cachewise", "catraces", "n_user_returns", len(h), xv, xr, f3, "X3：raw 裡每則文字 user 訊息；統一表：每個由文字 user 觸發的請求")
    xv, xr = xval(f3, dict(trace="cachewise", check="human_think_s_pctl"), "value")
    for p in (10, 50, 90, 99):
        put("cachewise", "catraces", f"user_think_p{p}_s", q(h["think_ms"] / 1000, p), pctl_str(xv, p), xr, f3,
            "X3：減「前一則 assistant 紀錄（含 sidechain）」；統一表：減上一個主線請求的最後一則", tol=0.5)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n = pd.Series([r["match"] for r in rows]).value_counts().to_dict()
    S.log(f"crosscheck: {len(rows)} 列 {n} → {OUT}")
    for r in rows:
        if r["match"] != "MATCH":
            S.log(f"  {r['match']} {r['source']}/{r['variant']} {r['metric']}: unified={r['unified']} x={r['x_value']} ({r['x_file']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
