"""load_wildchat.py — WildChat-1M（HF allenai/WildChat-1M @ 7d6490e4，ODC-BY，內容是真實使用者對話）→ 統一格式。

文字**不進**統一表（text_ref 指回原檔）。只讀 parquet。
一輪＝(第 k 則 user 訊息, 第 k 則 assistant 回應)。一個對話＝一個 session。

時間（README）：assistant 的 timestamp＝「backend 收到 ChatGPT 完整回應的時刻」→ t_end_ms（trace，解析度 1 s）。
  user 訊息**沒有** timestamp（全部 null，斷言）→ t_start_ms 用公式估（t_start_source=estimated）：
     t_start_ms[k] = t_end_ms[k] − service_est_ms[k]，service_est_ms = max(0, A + B×output_tokens + C×input_tokens)
     （A、B、C 見 results/p3_data/service_constants.json，FreeInference 訓練週的中位數迴歸；跨來源粗估）
     且 t_start_ms[k] ≥ t_end_ms[k−1]（使用者不可能在上一個回應完成前送出：估的服務時間比觀察到的間隔長時截斷，計數回報）
  think_ms[k] = t_start_ms[k+1] − t_end_ms[k]（think_source=estimated）；gap_ms 是兩個估的開始時間之差
token（WildChat 沒有 token 數）：Llama-3.1-8B-Instruct 的 tokenizer.json（tokenizers 函式庫，add_special_tokens=False，
  不加 chat template）逐訊息切；tokenizer 欄寫 llama-3.1。
  input_tokens＝context_tokens＝這一輪之前所有訊息＋這一則 user 訊息的 token 和；output_tokens＝這一則 assistant；
  new_user_tokens＝這一則 user 訊息的 token 數（實際打字量，不是上界；但含貼上的內容）
session_id ← conversation_hash＋":"＋第一則訊息的 turn_identifier（README：conversation_hash 不唯一）
user_id ← 對話層的 hashed_ip（README：對話中最常見的 hashed IP；跨對話串同一人是啟發式，header 沒有用）
req_type ← model；trigger ← user；obs_end_ms ← 整個資料集最大的 assistant timestamp
split ← §7 依時間（session 開始時刻，跨度 60/20/20）；split_user ← user_split(hashed_ip)

規則 6 的斷言（hard，對話層級的比例）
  turn == user 訊息數 == len(conversation)/2 ≥ 0.99；對話層 timestamp == 最後一則 assistant timestamp ≥ 0.99；
  assistant timestamp 非空 == 1；user timestamp 全是 null ≥ 0.999（否則 t_start 不該用估的）；
  對話內 assistant timestamp 非遞減 ≥ 0.99（倒退的對話整個丟掉，計數）；user／assistant 交替 ≥ 0.99（不交替的丟掉）；
  chars/token 的中位數 ∈ [1, 8]（tokenizer 真的有在切）
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

try:
    from . import schema as S
except ImportError:
    import schema as S

SOURCE = "wildchat"
DIR = os.path.join(S.DATASETS, "wildchat-1m")
VARIANTS = ["1m"]
TOK_PATH = os.path.join(os.environ.get("M7_LLAMA_GLOB", os.path.join(
    os.environ.get("TIARA_HOME", "/ssd8/hungwei/paper-hkv"),
    "hf-cache/hub/models--unsloth--Llama-3.1-8B-Instruct/snapshots/4699cc75b550f9c6f3173fb80f4703b62d946aa5")), "tokenizer.json")
CACHE = os.path.join(os.environ.get("TIARA_HOME", "/ssd8/hungwei/paper-hkv"), "tmp", "p3_data_cache")


def _tok_counts(texts, sha, fn):
    """每則訊息的 token 數；以檔案 sha256＋tokenizer sha256 為鍵快取（重跑不用再切）。"""
    from tokenizers import Tokenizer
    os.makedirs(CACHE, exist_ok=True)
    tsha = S.sha256_file(TOK_PATH)[:16]
    cp = os.path.join(CACHE, f"wildchat_{sha[:16]}_{tsha}.npy")
    if os.path.exists(cp):
        v = np.load(cp)
        if len(v) == len(texts):
            return v
    tok = Tokenizer.from_file(TOK_PATH)
    out = np.zeros(len(texts), dtype=np.int64)
    B = 50_000
    for i in range(0, len(texts), B):
        enc = tok.encode_batch([t if t is not None else "" for t in texts[i:i + B]], add_special_tokens=False)
        out[i:i + B] = [len(e.ids) for e in enc]
        S.log(f"  tokenize {fn} {min(i + B, len(texts))}/{len(texts)}")
    np.save(cp + ".tmp.npy", out)
    os.replace(cp + ".tmp.npy", cp)
    return out


def _one_file(fi, fn, path, sha, ck_acc):
    tb = pq.read_table(path, columns=["conversation_hash", "model", "timestamp", "conversation", "turn", "hashed_ip"])
    conv = tb.column("conversation").combine_chunks()
    off = conv.offsets.to_numpy()
    fl = conv.flatten()
    nc = len(off) - 1
    nmsg = np.diff(off)
    role = fl.field("role").to_numpy(zero_copy_only=False).astype(object)
    mts_valid = fl.field("timestamp").is_valid().to_numpy(zero_copy_only=False)
    mts = pc.cast(pc.cast(fl.field("timestamp"), pa.timestamp("us")), pa.int64()).to_numpy(zero_copy_only=False)
    tid = fl.field("turn_identifier").to_numpy(zero_copy_only=False)
    clen = pc.utf8_length(fl.field("content")).to_numpy(zero_copy_only=False).astype(np.int64)
    ntok = _tok_counts(fl.field("content").to_pylist(), sha, fn)
    top = pc.cast(pc.cast(tb.column("timestamp").combine_chunks(), pa.timestamp("us")), pa.int64()).to_numpy(zero_copy_only=False)
    turn = tb.column("turn").to_numpy()
    cid = np.repeat(np.arange(nc), nmsg)
    pos = np.arange(len(role)) - off[cid]
    is_u, is_a = role == "user", role == "assistant"
    expect_u = (pos % 2) == 0
    bad_alt = (expect_u & ~is_u) | (~expect_u & ~is_a)
    alt = np.ones(nc, bool)
    alt[np.unique(cid[bad_alt])] = False
    alt &= (nmsg % 2 == 0) & (nmsg > 0)
    n_user = np.bincount(cid[is_u], minlength=nc)
    av = is_a & mts_valid
    last_a = np.full(nc, np.iinfo(np.int64).min, dtype=np.int64)
    np.maximum.at(last_a, cid[av], mts[av])
    # 對話內 assistant timestamp 非遞減
    ai = np.nonzero(is_a)[0]
    back = np.zeros(nc, bool)
    same = cid[ai][1:] == cid[ai][:-1]
    dec = same & (mts[ai][1:] < mts[ai][:-1])
    back[cid[ai][1:][dec]] = True
    a = ck_acc
    a["n_conv"] += nc
    a["turn_eq_user"] += int(np.sum(turn == n_user))
    a["turn_eq_half"] += int(np.sum(turn * 2 == nmsg))
    a["top_eq_last"] += int(np.sum(top == last_a))
    a["assist_ts_valid"] += int(np.sum(mts_valid[is_a]))
    a["n_assist"] += int(is_a.sum())
    a["user_ts_null"] += int(np.sum(~mts_valid[is_u]))
    a["n_user"] += int(is_u.sum())
    a["mono"] += int(np.sum(~back))
    a["alt"] += int(alt.sum())
    a["chars"] += int(clen.sum())
    a["tokens"] += int(ntok.sum())
    keep = alt & ~back
    a["dropped_nonalt"] += int((~alt).sum())
    a["dropped_back"] += int((alt & back).sum())
    # 每輪：user 訊息 j=2k、assistant j+1
    first = off[:-1]
    uidx = np.nonzero(is_u & keep[cid] & expect_u)[0]
    aidx = uidx + 1
    c_of = cid[uidx]
    # 這一輪之前所有訊息的 token 和
    cum = np.cumsum(ntok)
    before = cum[uidx] - ntok[uidx] - np.r_[0, cum][first[c_of]]
    inp = before + ntok[uidx]
    sid = np.array([f"{h}:{t}" for h, t in zip(tb.column("conversation_hash").to_numpy(zero_copy_only=False)[c_of], tid[first[c_of]])],
                   dtype=object)
    hip = tb.column("hashed_ip").to_numpy(zero_copy_only=False)[c_of]
    model = tb.column("model").to_numpy(zero_copy_only=False)[c_of]
    rows_in_file = c_of
    d = pd.DataFrame(dict(session_id=sid, user_id=hip, t_end_ms=mts[aidx] // 1000, input_tokens=inp, output_tokens=ntok[aidx],
                          new_user_tokens=ntok[uidx], req_type=model,
                          text_ref=[f"data/{fn}#row={r}#msg={p}" for r, p in zip(rows_in_file, pos[uidx])],
                          _pos=pos[uidx], _file=fi))
    return d


def load(variant, const):
    files = sorted(f for f in os.listdir(os.path.join(DIR, "data")) if f.endswith(".parquet"))
    ck = S.Checks(SOURCE, variant)
    ck.eq("n parquet files", len(files), 14)
    exp = S.readme_sha256(os.path.join(DIR, "README.txt"))
    inputs, parts = [], []
    acc = {k: 0 for k in ("n_conv", "turn_eq_user", "turn_eq_half", "top_eq_last", "assist_ts_valid", "n_assist", "user_ts_null",
                          "n_user", "mono", "alt", "chars", "tokens", "dropped_nonalt", "dropped_back")}
    for fi, fn in enumerate(files):
        path = os.path.join(DIR, "data", fn)
        rec = S.input_record(path, exp.get(f"data/{fn}"))
        inputs.append(rec)
        ck.eq(f"sha256(data/{fn})==README.txt", rec["sha256_match"], True)
        parts.append(_one_file(fi, fn, path, rec["sha256"], acc))
        S.log(f"wildchat {fn}: turns={len(parts[-1])}")
    inputs.append(dict(path=TOK_PATH, sha256=S.sha256_file(TOK_PATH), note="tokenizer"))
    nc = acc["n_conv"]
    ck.ge("turn == n_user_messages", acc["turn_eq_user"] / nc, 0.99)
    ck.ge("turn == len(conversation)/2", acc["turn_eq_half"] / nc, 0.99, "規格 §4：輪數＝訊息數÷2")
    ck.ge("conversation timestamp == last assistant timestamp", acc["top_eq_last"] / nc, 0.99, "README：最後一輪的時間")
    ck.eq("assistant timestamp non-null", acc["assist_ts_valid"] / acc["n_assist"], 1.0)
    ck.ge("user timestamp is null", acc["user_ts_null"] / acc["n_user"], 0.999, "user 沒有時間 → t_start 用估的")
    ck.ge("assistant timestamp non-decreasing in conversation", acc["mono"] / nc, 0.99)
    ck.ge("alternating user/assistant (even length)", acc["alt"] / nc, 0.99)
    ck.rng("chars per token (all messages)", acc["chars"] / max(acc["tokens"], 1), 1.0, 8.0)
    ck.report("conversations dropped: non-alternating", acc["dropped_nonalt"])
    ck.report("conversations dropped: assistant timestamp goes back", acc["dropped_back"])
    ck.report("conversations read", nc)
    d = pd.concat(parts, ignore_index=True)
    del parts
    ck.eq("n unique session_id == kept conversations", int(d["session_id"].nunique()),
          int(nc - acc["dropped_nonalt"] - acc["dropped_back"]), "conversation_hash＋第一個 turn_identifier")
    n = len(d)
    # 估的 t_start：t_end − service；不早於上一輪的 t_end
    d = d.sort_values(["session_id", "_pos"], kind="mergesort").reset_index(drop=True)
    svc = np.round(np.maximum(0.0, const["A"] + const["B"] * d["output_tokens"].values + const["C"] * d["input_tokens"].values)
                   ).astype(np.int64)
    t_end = d["t_end_ms"].values.astype(np.int64)
    prev_end = d.groupby("session_id")["t_end_ms"].shift(1).values
    t_start = t_end - svc
    clip = ~np.isnan(prev_end) & (t_start < np.nan_to_num(prev_end, nan=-np.inf))
    t_start = np.where(clip, np.nan_to_num(prev_end).astype(np.int64), t_start)
    ck.report("t_start_est clipped to previous t_end", int(clip.sum()), "估的服務時間比觀察到的間隔長")
    ck.report("frac(end-to-end gap < service_est of next reply)", float(clip.sum() / max(1, (~np.isnan(prev_end)).sum())))
    obs_end = int(t_end.max())
    df = pd.DataFrame(dict(
        source=SOURCE, variant=variant, session_id=d["session_id"].values, user_id=d["user_id"].values,
        session_rule="conversation (conversation_hash + first turn_identifier)", trigger="user", t_start_ms=t_start,
        t_end_ms=pd.array(t_end, dtype="Int64"), t_end_source="trace", input_tokens=pd.array(d["input_tokens"].values, dtype="Int64"),
        output_tokens=pd.array(d["output_tokens"].values, dtype="Int64"), cached_tokens=pd.array([pd.NA] * n, dtype="Int64"),
        tokenizer="llama-3.1(tokenizers,no-chat-template)", new_user_tokens=pd.array(d["new_user_tokens"].values, dtype="Int64"),
        context_tokens=pd.array(d["input_tokens"].values, dtype="Int64"), block_ids=None,
        block_tokens=pd.array([pd.NA] * n, dtype="Int32"), tool_name=None, tool_category=None,
        tool_latency_ms=pd.array([pd.NA] * n, dtype="Int64"), req_type=d["req_type"].values, text_ref=d["text_ref"].values,
        obs_end_ms=obs_end, t_start_source="estimated", think_source="estimated", service_est_ms=pd.array(svc, dtype="Int64"),
        harness=None, is_subagent=None, parent_session_id=None, next_waited_for=None,
        provider_prompt_tokens=pd.array([pd.NA] * n, dtype="Int64"), _order=d["_pos"].values))
    del d
    df, fst = S.finalize(df, chain=True, by_time=False)
    ck.report("think_ms<0", fst["think_negative"])
    split, bounds, seg_end = S.time_split(df, int(df["t_start_ms"].min()), obs_end)
    df["split"] = split
    df["split_user"] = S.user_split(df["user_id"].values)
    sw = S.apply_split_window(df, seg_end)
    ck.report("split_drop rows (after boundary)", sw["split_drop_rows"])
    df = df.drop(columns=["_order"])
    extra = dict(t_start_formula=f"t_start_ms = max(t_end_ms − max(0, {const['A']:.3f} + {const['B']:.5f}*output_tokens + "
                                 f"{const['C']:.6f}*input_tokens), 上一輪 t_end_ms)",
                 service_constants=dict(A=const["A"], B=const["B"], C=const["C"], run_id=const.get("run_id"), weeks=const["weeks"]),
                 tokenizer_path=TOK_PATH, obs_end_rule="資料集最大的 assistant timestamp",
                 split_rule="§7 依時間 60/20/20（session 開始時刻）；split_user＝sha256(salt+hashed_ip) 80/10/10",
                 split_bounds=bounds, finalize=fst, counts=acc, tier="R", license="ODC-BY（LICENSE.md）；內容是真實使用者對話",
                 new_user_tokens_rule="這一則 user 訊息的 Llama-3.1 token 數")
    ck.enforce()
    return df, ck, inputs, extra
