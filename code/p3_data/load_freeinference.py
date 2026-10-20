"""load_freeinference.py — FreeInference agentic trace（HF harvardMadsys/freeinference_agentic_trace @ 16d3c04d）→ 統一格式。

一週一個 variant（week01 … week16；session 不跨週，README）。只讀 release/<week>/processed/*.parquet（不讀、不執行 raw）。

欄位對應（規格 §3）
  session_id／user_id／harness／parent_session_id ← sessions.parquet（子 agent 也是一個 session，is_subagent=True）
  turn_idx ← step（斷言：每個 session 的 step＝0..n−1）
  trigger ← step_trigger（user／tool）；next_waited_for ← 下一個請求的 waited_for
  t_start_ms／t_end_ms ← start_ms／end_ms（t_end_source=trace，gateway 實測）
  input_tokens＝context_tokens ← Σ role_tokens（o200k_base，和 block_ids 同一個 tokenizer）
  output_tokens ← completion_tokens、cached_tokens ← cache_read_tokens（**provider 的 tokenizer**，和 input 不同）
  provider_prompt_tokens ← prompt_tokens（provider 的 tokenizer）
  new_user_tokens ← max(0, role_tokens.user[k] − role_tokens.user[k−1])；step 0 取 role_tokens.user[0]（上界：harness 可能把別的東西放在 user 角色）
  block_ids ← block_ids（16-token、週內編號；只在同一個 variant 內可比），block_tokens=16
  tool_name／tool_category／tool_latency_ms ← tool_calls.parquet 中這個請求的回應所呼叫的工具（依 call_index 用 | 串；時長取最大）
  req_type ← model
  obs_end_ms ← 這一週所有請求最大的 end_ms（觀察到的週末；週邊界的時區 README 沒寫，所以不用名目邊界）
  split ← 週（§7：week01–10 train、11–12 valid、13–16 test）；split_user ← user_split(user_id)（80/10/10，雜湊）

規則 6 的斷言（hard，不過就丟 DataAssertionError）
  (a) len(block_ids)==floor(Σrole_tokens/16) ≥ 0.95（README：o200k_base、16-token、最後不滿一塊丟掉）
  (b) 每個 session 的 step 連續 0..n−1 ≥ 0.99
  (c) 工具 latency_ms（README：到下一次 LLM 請求的間隔）== 下一個請求 start_ms − 這個請求 end_ms（±1 s）≥ 0.95
  (d) 時間單位：(end−start−ttft)/completion_tokens 的中位數在 0.5..500 ms/token；start_ms 落在目錄名的日期範圍（±1 天）≥ 0.999
  (e) processed/*.parquet 的 sha256 與列數 == tables.json；requests 的 session 集合 == sessions.parquet
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

try:
    from . import schema as S
except ImportError:  # 直接當 script 跑
    import schema as S

SOURCE = "freeinference"
ROOT = os.path.join(S.DATASETS, "freeinference_agentic_trace", "release")
N_WEEKS = 16
TRAIN_WEEKS, VALID_WEEKS = range(1, 11), range(11, 13)   # §7；其餘（13–16）test
SESSION_RULE = "official session_id（FreeInference 用重複的 tool-call id 與 assistant 訊息重建；不跨週）"
TOKENIZER = "o200k_base(input,block_ids,new_user);provider(output,cached)"
REQ_COLS = ["session_id", "step", "link_reason", "start_ms", "end_ms", "ttft_ms", "model", "provider", "step_trigger",
            "waited_for", "prompt_tokens", "completion_tokens", "cache_read_tokens", "role_tokens", "transition",
            "finish_reason"]
ROLES = ("system", "user", "assistant", "tool", "tool_definitions")


def weeks():
    ws = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d, "processed")))
    if len(ws) != N_WEEKS:
        raise S.DataAssertionError(f"FreeInference 應有 {N_WEEKS} 週，找到 {len(ws)}：{ws}", [])
    return ws


def week_split(week_no):
    return "train" if week_no in TRAIN_WEEKS else ("valid" if week_no in VALID_WEEKS else "test")


def _role_sum(rt):
    tot = np.zeros(len(rt), dtype=np.int64)
    user = None
    for f in ROLES:
        v = pc.fill_null(pc.struct_field(rt, f), 0).to_numpy(zero_copy_only=False).astype(np.int64)
        tot += v
        if f == "user":
            user = v
    return tot, user


def load_week(week_no, with_blocks=True):
    """回傳 (df, checks, inputs, extra, writer)。writer 把 block_ids 依 row group 串流寫入（不進 pandas）。"""
    wname = weeks()[week_no - 1]
    variant = f"week{week_no:02d}"
    pdir = os.path.join(ROOT, wname, "processed")
    ck = S.Checks(SOURCE, variant)
    # (e) 輸入檔雜湊與列數
    tj = json.load(open(os.path.join(pdir, "tables.json")))
    inputs = []
    for fn in ("requests.parquet", "sessions.parquet", "tool_calls.parquet"):
        rec = S.input_record(os.path.join(pdir, fn), tj["files"][fn]["sha256"])
        inputs.append(rec)
        ck.eq(f"sha256({fn})==tables.json", rec["sha256_match"], True)
    inputs.append(S.input_record(os.path.join(ROOT, wname, "manifest.json")))
    pf = pq.ParquetFile(os.path.join(pdir, "requests.parquet"))
    ck.eq("rows(requests)==tables.json", pf.metadata.num_rows, tj["files"]["requests.parquet"]["rows"])
    tb = pf.read(columns=REQ_COLS)
    n = tb.num_rows
    role_sum, role_user = _role_sum(tb.column("role_tokens").combine_chunks())
    d = tb.drop(["role_tokens"]).to_pandas()
    del tb
    # source 已依 (session_id, step) 排序：block_ids 串流寫入靠這個 1:1 對齊
    sid = d["session_id"].values
    same = sid[1:] == sid[:-1]
    sorted_ok = bool(np.all((sid[1:] > sid[:-1]) | (same & (d["step"].values[1:] > d["step"].values[:-1]))))
    ck.eq("requests sorted by (session_id, step)", sorted_ok, True, "block_ids 串流對齊的前提")
    # (a) block 單位
    rg_rows = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
    if with_blocks:
        blen = np.concatenate([pc.list_value_length(pf.read_row_group(i, columns=["block_ids"]).column("block_ids"))
                               .to_numpy(zero_copy_only=False) for i in range(pf.num_row_groups)]).astype(np.int64)
        ck.ge("len(block_ids)==floor(sum(role_tokens)/16)", float(np.mean(blen == role_sum // 16)), 0.95,
              "o200k_base、16-token、最後不滿一塊丟掉（README）")
        ck.report("len(block_ids)==ceil(sum(role_tokens)/16)", float(np.mean(blen == -(-role_sum // 16))), "應該低（README 說丟掉）")
    # (b) step 連續
    g = d.groupby("session_id")["step"]
    ok = (g.transform("min") == 0) & (g.transform("max") == g.transform("size") - 1)
    ck.ge("steps consecutive 0..n-1 per session", float(ok.groupby(d["session_id"]).all().mean()), 0.99)
    # session 集合
    se = pq.read_table(os.path.join(pdir, "sessions.parquet")).to_pandas()
    ck.eq("rows(sessions)==tables.json", len(se), tj["files"]["sessions.parquet"]["rows"])
    ck.eq("set(requests.session_id)==set(sessions.session_id)", set(d["session_id"].unique()) == set(se["session_id"]), True)
    ck.eq("sessions.user_id non-null", int(se["user_id"].isna().sum()), 0)
    sub_par_ok = se["parent_session_id"].dropna().isin(se["session_id"]).mean() if se["parent_session_id"].notna().any() else 1.0
    ck.report("subagent parent_session_id in same week", float(sub_par_ok))
    # (d) 時間單位
    st, en, tt = d["start_ms"].values, d["end_ms"].values, d["ttft_ms"].values.astype(float)
    ct = d["completion_tokens"].fillna(0).values.astype(float)
    ck.ge("start_ms<=end_ms", float(np.mean(st <= en)), 0.999)
    okt = ~np.isnan(tt)
    ck.ge("ttft_ms<=end_ms-start_ms", float(np.mean((tt[okt] <= (en - st)[okt]))) if okt.any() else 1.0, 0.99)
    m = okt & (ct >= 50)
    dec = (en - st - np.where(okt, tt, 0))[m] / ct[m]
    # 2026-10-17 第一次跑：這條原本是 hard，week01 得 0.055 ms/token 而不過（provider B 的 ttft_ms≈整段時長，
    # (end−start−ttft) 不是解碼時間；FAILED.json 在 scratch run）。改成只報告，單位改用整段時長驗算（下一條，hard）。
    ck.report("median((end-start-ttft)/completion_tokens) ms/token", float(np.median(dec)) if m.any() else float("nan"),
              "不是單位驗算：部分 provider 的 ttft_ms≈end−start（緩衝後一次送出）")
    ck.report("frac(end-start-ttft <= 1 ms | completion>=50)", float(np.mean((en - st - np.where(okt, tt, 0))[m] <= 1)) if m.any()
              else float("nan"))
    mm_ = ct >= 50
    ck.rng("median((end-start)/completion_tokens) ms/token", float(np.median((en - st)[mm_] / ct[mm_])) if mm_.any() else float("nan"),
           0.5, 500, "整段時長／輸出 token：時間是毫秒（秒會是 ~0.04、微秒 ~4e4）")
    d0 = datetime.strptime(wname[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    d1 = datetime.strptime(wname[11:21], "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
    lo, hi = (d0 - timedelta(days=1)).timestamp() * 1000, (d1 + timedelta(days=1)).timestamp() * 1000
    ck.ge("start_ms within week dir dates ±1 day", float(np.mean((st >= lo) & (st < hi))), 0.999,
          f"{wname}；毫秒 epoch 的驗算")
    trig_ok = d["step_trigger"].isin(["user", "tool"]).mean()
    ck.eq("step_trigger in {user,tool}", float(trig_ok), 1.0)
    ck.report("first step trigger==user", float((d.loc[d["step"] == 0, "step_trigger"] == "user").mean()))
    # 工具
    tc = pq.read_table(os.path.join(pdir, "tool_calls.parquet"),
                       columns=["session_id", "step", "call_index", "name", "category", "latency_ms"]).to_pandas()
    ck.eq("rows(tool_calls)==tables.json", len(tc), tj["files"]["tool_calls.parquet"]["rows"])
    tc = tc.sort_values(["session_id", "step", "call_index"], kind="mergesort")
    agg = tc.groupby(["session_id", "step"]).agg(
        tool_name=("name", lambda x: "|".join("" if v is None else str(v) for v in x)),
        tool_category=("category", lambda x: "|".join("" if v is None else str(v) for v in x)),
        tool_latency_ms=("latency_ms", "max"))
    key = pd.MultiIndex.from_arrays([d["session_id"], d["step"]])
    agg = agg.reindex(key)
    nxt_start = d.groupby("session_id")["start_ms"].shift(-1).values
    gap = nxt_start - en
    lat = agg["tool_latency_ms"].values.astype(float)
    mm = ~np.isnan(lat) & ~np.isnan(gap)
    ck.ge("tool latency_ms == next.start_ms - end_ms (±1 s)", float(np.mean(np.abs(lat[mm] - gap[mm]) <= 1000)) if mm.any()
          else float("nan"), 0.95, f"n={int(mm.sum())}；README：latency＝到下一次 LLM 請求的間隔")
    ck.report("tool latency_ms == next.start_ms - end_ms (exact)", float(np.mean(lat[mm] == gap[mm])) if mm.any() else float("nan"))
    ck.report("rows with start_ms < previous end_ms (overlap)", int(np.sum(gap[~np.isnan(gap)] < 0)))
    # 組統一表（列的順序＝source 的順序）
    u = se.set_index("session_id")
    du = role_user
    prev_user = pd.Series(du).groupby(d["session_id"].values).shift(1).values
    nu = np.where(np.isnan(prev_user), du, du - np.nan_to_num(prev_user))
    ck.report("new_user_tokens negative before clip (role_tokens.user shrank)", int(np.sum(nu < 0)))
    obs_end = int(en.max())
    out = pd.DataFrame(dict(
        source=SOURCE, variant=variant, session_id=d["session_id"].values,
        user_id=d["session_id"].map(u["user_id"]).values, session_rule=SESSION_RULE,
        trigger=d["step_trigger"].values, t_start_ms=st.astype(np.int64), t_end_ms=pd.array(en, dtype="Int64"),
        t_end_source="trace", input_tokens=pd.array(role_sum, dtype="Int64"),
        output_tokens=d["completion_tokens"].astype("Int64").values, cached_tokens=d["cache_read_tokens"].round().astype("Int64").values,
        tokenizer=TOKENIZER, new_user_tokens=pd.array(np.maximum(nu, 0).astype(np.int64), dtype="Int64"),
        context_tokens=pd.array(role_sum, dtype="Int64"), block_ids=None, block_tokens=16,
        tool_name=agg["tool_name"].values, tool_category=agg["tool_category"].values,
        tool_latency_ms=pd.Series(lat).round().astype("Int64").values, req_type=d["model"].values,
        text_ref=[f"release/{wname}/processed/requests.parquet#row={i}" for i in range(n)],
        obs_end_ms=obs_end, t_start_source="trace", think_source="trace", service_est_ms=pd.array([pd.NA] * n, dtype="Int64"),
        harness=d["session_id"].map(u["harness"]).values,
        is_subagent=d["session_id"].map(u["parent_session_id"].notna()).values.astype(bool),
        parent_session_id=d["session_id"].map(u["parent_session_id"]).values,
        provider_prompt_tokens=d["prompt_tokens"].astype("Int64").values,
        _order=np.arange(n), _step=d["step"].values, _waited=d["waited_for"].values))
    out, fstats = S.finalize(out, chain=True, by_time=False)
    ck.eq("turn_idx==step", bool((out["turn_idx"].values == out["_step"].values).all()), True)
    ck.report("child (next step) starts before parent start", fstats["child_starts_before_parent"])
    ck.report("think_ms<0 (next start before this end)", fstats["think_negative"])
    out["next_waited_for"] = pd.Series(out["_waited"].values).groupby(out["session_id"].values).shift(-1).values
    ck.report("next_trigger==tool but next_waited_for!=tool",
              int(((out["next_trigger"] == "tool") & (out["next_waited_for"] != "tool")).sum()),
              "ask_user_tool／subagent：工具觸發，但間隔是在等人或等子 agent")
    out["split"] = week_split(week_no)
    out["split_user"] = S.user_split(out["user_id"].values)
    S.apply_split_window(out, {"train": None, "valid": None, "test": None})
    out = out.drop(columns=["_order", "_step", "_waited"])
    extra = dict(week=wname, week_no=week_no, obs_end_rule="這一週最大的 end_ms",
                 split_rule="week01-10 train / week11-12 valid / week13-16 test（§7）；split_user＝sha256(salt+user_id) 80/10/10",
                 finalize=fstats, source_manifest=json.load(open(os.path.join(ROOT, wname, "manifest.json"))) |
                 {"files": "（略，見原檔）"},
                 new_user_tokens_rule="max(0, role_tokens.user[k] − role_tokens.user[k−1])；step 0＝role_tokens.user[0]",
                 tier="R")

    def writer(df_out, path):
        base = S.to_arrow(df_out)
        bi = base.schema.get_field_index("block_ids")
        w = pq.ParquetWriter(path, S.ARROW_SCHEMA, compression="zstd")
        try:
            off = 0
            for i, nr in enumerate(rg_rows):
                chunk = base.slice(off, nr)
                if with_blocks:
                    b = pf.read_row_group(i, columns=["block_ids"]).column("block_ids").combine_chunks()
                    if b.type != pa.large_list(pa.int64()):
                        b = b.cast(pa.large_list(pa.int64()))
                    chunk = chunk.set_column(bi, S.ARROW_SCHEMA.field("block_ids"), b)
                w.write_table(chunk, row_group_size=nr)
                off += nr
            assert off == len(df_out), (off, len(df_out))
        finally:
            w.close()

    ck.enforce()  # loader 自己丟錯（規格 §4）
    return out, ck, inputs, extra, writer


def service_constants(train_weeks=TRAIN_WEEKS, n_fit=200_000, seed=0):
    """沒有回應結束時間的來源（Bailian、Mooncake、WildChat）用的服務時間公式：
       service_est_ms = A + B × output_tokens + C × input_tokens
       A、B、C：FreeInference **訓練週**（week01–10）上 (end_ms − start_ms) 對 (completion_tokens, Σrole_tokens)
       的中位數迴歸（statsmodels QuantReg q=0.5；固定 seed 抽 n_fit 列）。
       原本打算用 TTFT 中位數＋解碼 ms/token 中位數，但部分 provider 的 ttft_ms≈整段時長（見 load_week 的報告欄），
       (end−start−ttft) 不是解碼時間，所以改用整段時長直接迴歸。跨來源粗估（模型、硬體、批次、prefix cache 都不同），標 estimated。"""
    import statsmodels.api as sm
    ws = weeks()
    parts = []
    for wn in train_weeks:
        t = pq.read_table(os.path.join(ROOT, ws[wn - 1], "processed", "requests.parquet"),
                          columns=["start_ms", "end_ms", "ttft_ms", "completion_tokens", "role_tokens"])
        tot, _ = _role_sum(t.column("role_tokens").combine_chunks())
        d = t.drop(["role_tokens"]).to_pandas()
        d["in_tok"] = tot
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    n_all = len(d)
    d = d[d["completion_tokens"].notna() & (d["end_ms"] >= d["start_ms"])]
    dur = (d["end_ms"] - d["start_ms"]).values.astype(float)
    out = d["completion_tokens"].values.astype(float)
    inp = d["in_tok"].values.astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(d), size=min(n_fit, len(d)), replace=False)
    X = np.column_stack([np.ones(len(idx)), out[idx], inp[idx]])
    fit = sm.QuantReg(dur[idx], X).fit(q=0.5)
    a, b, c = (float(v) for v in fit.params)
    pred = a + b * out + c * inp
    resid = dur - pred
    tt = d["ttft_ms"].astype(float).values
    return dict(formula="service_est_ms = A + B*output_tokens + C*input_tokens (clip >= 0)", A=a, B=b, C=c,
                fit="statsmodels QuantReg q=0.5 on (end_ms-start_ms) ~ 1 + completion_tokens + sum(role_tokens)",
                n_rows_weeks=int(n_all), n_fit=int(len(idx)), seed=seed,
                abs_resid_ms_pctl_all=S.pctl(np.abs(resid), (50, 90)),
                frac_pred_le_dur_all=float(np.mean(pred <= dur)),
                dur_ms_pctl=S.pctl(dur, (10, 50, 90)), output_tokens_pctl=S.pctl(out, (10, 50, 90)),
                input_tokens_pctl=S.pctl(inp, (10, 50, 90)), ttft_ms_pctl=S.pctl(tt, (10, 50, 90)),
                source="freeinference", weeks=[ws[w - 1] for w in train_weeks],
                note="FreeInference 訓練週（week01–10）的中位數迴歸；跨來源粗估，只用來把服務時間從間隔裡扣掉")
