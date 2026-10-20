"""load_burstgpt.py — BurstGPT release v2.0（HPMLL/BurstGPT）→ 統一格式。

只有 BurstGPT_3.csv（與 BurstGPT_without_fails_3.csv）有 `Session ID`、`Elapsed time` 欄（2026-10-17 讀 header 確認）；
BurstGPT_1／_2 只有 Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type → 依規格 §4「不當成 session 資料」，不轉。
variant v3 ＝ BurstGPT_3.csv（含失敗請求，Response tokens＝0 的保留：使用者看到失敗後的重送也是回來）。
只有 Log Type＝Conversation log 的列有 Session ID（斷言）；API log 沒有 session → 不進統一表（列數記在 manifest）。

欄位對應
  session_id ← Session ID；user_id ← null；trigger ← user（conversation 模式：每個請求是使用者送出的）
  t_start_ms ← Timestamp × 1000（秒，解析度 1 s）；t_end_ms ← (Timestamp + Elapsed time) × 1000（t_end_source=trace：
     README「從送出到拿到完整回應」）；think_ms ← 下一輪 Timestamp − 這一輪 t_end（可能 <0：上一個回應還沒完成就送下一個，保留原值、計數）
  input/output_tokens ← Request／Response tokens（provider 的 tokenizer）；context_tokens ← Request tokens
  new_user_tokens ← max(0, Request − 上一輪 Request − 上一輪 Response)；第一輪＝Request（同 Bailian 的定義；
     只有 59% 的下一輪 Request ≥ 上一輪 Request＋Response，所以這是有雜訊的上界，見 manifest）
  req_type ← Model；obs_end_ms ← max(Timestamp + Elapsed) × 1000
  split ← §7 沒有列 BurstGPT；比照 Bailian：時間 60/20/20（session 開始時刻）

規則 6 的斷言（hard）
  Total tokens == Request + Response ≥ 0.99；Session ID 非空 ⇔ Conversation log ≥ 0.999；
  時間跨度 ∈ [105, 115] 天（README：110 天；Timestamp 是秒）；Elapsed time ≥ 0；
  median(Elapsed／Response tokens) ∈ 0.001..1 s/token（Elapsed 是秒）；Timestamp 非遞減
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

try:
    from . import schema as S
except ImportError:
    import schema as S

SOURCE = "burstgpt"
DIR = os.path.join(S.DATASETS, "burstgpt")
VARIANTS = ["v3"]
FILES = {"v3": "BurstGPT_3.csv"}


def load(variant):
    fn = FILES[variant]
    path = os.path.join(DIR, fn)
    ck = S.Checks(SOURCE, variant)
    rec = S.input_record(path, S.readme_sha256(os.path.join(DIR, "README.txt")).get(fn))
    ck.eq(f"sha256({fn})==README.txt", rec["sha256_match"], True)
    d = pd.read_csv(path)
    need = ["Timestamp", "Session ID", "Elapsed time", "Model", "Request tokens", "Response tokens", "Total tokens", "Log Type"]
    ck.eq("columns include Session ID and Elapsed time", all(c in d.columns for c in need), True, f"columns={list(d.columns)}")
    ck.enforce()
    n_all = len(d)
    ck.ge("Total tokens == Request + Response", float(np.mean(d["Total tokens"] == d["Request tokens"] + d["Response tokens"])), 0.99)
    conv = d["Log Type"] == "Conversation log"
    ck.ge("Session ID present <=> Conversation log", float(np.mean(d["Session ID"].notna() == conv)), 0.999)
    span = (d["Timestamp"].max() - d["Timestamp"].min()) / 86400
    ck.rng("time span days (README: 110)", float(span), 105, 115, "Timestamp 是秒")
    ck.eq("Timestamp non-decreasing", bool(d["Timestamp"].is_monotonic_increasing), True)
    ck.ge("Elapsed time >= 0", float(np.mean(d["Elapsed time"] >= 0)), 1.0)
    d["_row"] = np.arange(len(d))
    c = d[d["Session ID"].notna()].copy()
    m = c["Response tokens"] >= 50
    ck.rng("median(Elapsed/Response tokens) s/token", float(np.median(c["Elapsed time"][m] / c["Response tokens"][m])), 0.001, 1,
           "Elapsed 是秒")
    ck.report("rows Response tokens == 0 (failures, kept)", int((c["Response tokens"] == 0).sum()))
    ck.report("API log rows (no session, excluded)", int((~conv).sum()))
    ck.report("Timestamp integer seconds (frac)", float(np.mean(c["Timestamp"] % 1 == 0)), "解析度 1 s")
    n = len(c)
    c = c.reset_index(drop=True)
    t_start = np.round(c["Timestamp"].values * 1000).astype(np.int64)
    t_end = np.round((c["Timestamp"].values + c["Elapsed time"].values) * 1000).astype(np.int64)
    obs_end = int(t_end.max())
    df = pd.DataFrame(dict(
        source=SOURCE, variant=variant, session_id=c["Session ID"].values, user_id=None,
        session_rule="official Session ID (Conversation log only)", trigger="user", t_start_ms=t_start,
        t_end_ms=pd.array(t_end, dtype="Int64"), t_end_source="trace",
        input_tokens=pd.array(c["Request tokens"].values, dtype="Int64"), output_tokens=pd.array(c["Response tokens"].values, dtype="Int64"),
        cached_tokens=pd.array([pd.NA] * n, dtype="Int64"), tokenizer="openai(provider)",
        new_user_tokens=pd.array([pd.NA] * n, dtype="Int64"), context_tokens=pd.array(c["Request tokens"].values, dtype="Int64"),
        block_ids=None, block_tokens=pd.array([pd.NA] * n, dtype="Int32"), tool_name=None, tool_category=None,
        tool_latency_ms=pd.array([pd.NA] * n, dtype="Int64"), req_type=c["Model"].values, text_ref=[f"{fn}#row={i}" for i in c["_row"].values], obs_end_ms=obs_end,
        t_start_source="trace", think_source="trace", service_est_ms=pd.array([pd.NA] * n, dtype="Int64"), split_user=None,
        harness=None, is_subagent=None, parent_session_id=None, next_waited_for=None,
        provider_prompt_tokens=pd.array([pd.NA] * n, dtype="Int64"), _order=np.arange(n)))
    df, fst = S.finalize(df, chain=True)
    # new_user_tokens：需要上一輪（finalize 之後才知道順序）
    srt = df.sort_values(["session_id", "turn_idx"], kind="mergesort")
    g = srt.groupby("session_id", sort=False)
    pr = g["input_tokens"].shift(1)
    po = g["output_tokens"].shift(1)
    nu = (srt["input_tokens"] - pr - po)
    hist = (nu >= 0)[pr.notna()]
    ck.report("Request >= prev Request + prev Response (frac)", float(hist.mean()),
              "低＝請求不一定帶完整歷史；new_user_tokens 是有雜訊的上界")
    nu = nu.where(pr.notna(), srt["input_tokens"]).clip(lower=0)
    df.loc[srt.index, "new_user_tokens"] = nu.astype("Int64").values
    ck.report("think_ms<0 (next sent before response complete)", fst["think_negative"])
    split, bounds, seg_end = S.time_split(df, int(t_start.min()), obs_end)
    df["split"] = split
    sw = S.apply_split_window(df, seg_end)
    ck.report("split_drop rows (after boundary)", sw["split_drop_rows"])
    df = df.drop(columns=["_order"])
    extra = dict(file=fn, rows_in_file=int(n_all), rows_conversation=int(n),
                 excluded=dict(BurstGPT_1="沒有 Session ID／Elapsed time 欄", BurstGPT_2="沒有 Session ID／Elapsed time 欄",
                               api_log_rows=int((~conv).sum())),
                 t_end_rule="(Timestamp + Elapsed time) × 1000（trace）", obs_end_rule="max(Timestamp + Elapsed) × 1000",
                 split_rule="時間 60/20/20（session 開始時刻；§7 沒列 BurstGPT，比照 Bailian）", split_bounds=bounds,
                 finalize=fst, tier="R",
                 new_user_tokens_rule="max(0, Request − prev Request − prev Response)；第一輪＝Request")
    ck.enforce()
    return df, ck, [rec], extra
