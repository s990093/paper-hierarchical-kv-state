"""schema.py — 第三階段統一資料格式（docs/phase3_20261017/03_data_design.md §3）。

一列＝一個 session 的一輪（turn）。每個 source／variant 存成
  $TIARA_DATASETS/p3_unified/<source>/<variant>/turns.parquet  ＋  manifest.json

這支只放「所有來源共用」的東西：
  * 欄位、型別（COLUMNS）與 validate()；
  * Checks：「用 A 欄驗算 B 欄」的斷言紀錄（CLAUDE.md 規則 6）。任何 hard 斷言不過 → enforce() 丟 DataAssertionError，
    呼叫端（build.py）照錄、不寫 turns.parquet（規則 2、規則 7：查不到不等於沒有）；
  * finalize()：由 (session, 開始時間, 父列) 算 turn_idx、next_t_start_ms、think_ms、censored；
  * 切分（§7）：time_split()、week_split()、user_split()、apply_split_window()；
  * write_variant()：寫 parquet＋manifest（輸入 sha256、程式 sha256／git hash、斷言、統計）。

規格之外加的欄位（全部列在 EXTRA，理由見 03b_data_report.md §2）：
  parent_turn_idx、n_children、next_trigger、gap_ms、t_start_source、think_source、service_est_ms、
  split_user、split_drop、split_censored、split_obs_end_ms、harness、is_subagent、parent_session_id、
  next_waited_for、provider_prompt_tokens。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATASETS = os.environ.get("TIARA_DATASETS", "/ssd8/hungwei/paper-hkv/datasets")
UNIFIED = os.environ.get("P3_UNIFIED", os.path.join(DATASETS, "p3_unified"))
SCHEMA_VERSION = 1

# (欄位, arrow 型別, 可不可以是 null, 說明)。順序＝寫檔順序。
SPEC = [
    ("source", pa.string(), False, "來源名稱"),
    ("variant", pa.string(), False, "版本（週、trace 檔、重建規則…）"),
    ("session_id", pa.string(), False, "session（重建的看 session_rule）"),
    ("user_id", pa.string(), True, "使用者；來源沒有就 null"),
    ("session_rule", pa.string(), False, "session 怎麼來（官方欄位或重建規則名）"),
    ("turn_idx", pa.int32(), False, "session 內依 (t_start_ms, 原始順序) 的序號，0 起算"),
    ("trigger", pa.string(), False, "user／tool／unknown"),
    ("t_start_ms", pa.int64(), False, "請求開始（毫秒；各來源自己的時間原點）"),
    ("t_end_ms", pa.int64(), True, "回應結束；t_end_source 說是 trace 還是估的"),
    ("t_end_source", pa.string(), False, "trace／estimated"),
    ("input_tokens", pa.int64(), True, "輸入 token（tokenizer 欄）"),
    ("output_tokens", pa.int64(), True, "輸出 token"),
    ("cached_tokens", pa.int64(), True, "來源回報的 prefix-cache 命中 token"),
    ("tokenizer", pa.string(), False, "token 數用哪個 tokenizer"),
    ("new_user_tokens", pa.int64(), True, "這一輪新加的使用者輸入（打字量上界）；算法見各 loader"),
    ("context_tokens", pa.int64(), True, "這一輪開始時的累積 context"),
    ("block_ids", pa.large_list(pa.int64()), True, "前綴 block id（有的來源才有）"),
    ("block_tokens", pa.int32(), True, "一個 block 幾個 token（16、512…）"),
    ("tool_name", pa.string(), True, "這一輪回應呼叫的工具（多個用 | 串）"),
    ("tool_category", pa.string(), True, "工具類別（多個用 | 串）"),
    ("tool_latency_ms", pa.int64(), True, "這一輪的工具時長（多個取最大）"),
    ("req_type", pa.string(), True, "Bailian type、WildChat／FreeInference model 等"),
    ("text_ref", pa.string(), True, "文字在原始檔的位置（文字本身不進表）"),
    ("next_t_start_ms", pa.int64(), True, "同 session 下一輪（樹狀來源＝最早的子請求）的開始"),
    ("think_ms", pa.int64(), True, "next_t_start_ms − t_end_ms（主標籤）"),
    ("censored", pa.bool_(), False, "觀察窗內沒有下一輪＝右設限"),
    ("obs_end_ms", pa.int64(), False, "這一列的觀察窗終點"),
    ("split", pa.string(), False, "train／valid／test（§7）"),
]
EXTRA = [
    ("parent_turn_idx", pa.int32(), True, "父請求的 turn_idx（鏈＝上一輪；樹狀來源＝重建的父；根＝null）"),
    ("n_children", pa.int32(), False, "以這一列為父的請求數（>1＝分叉）"),
    ("next_trigger", pa.string(), True, "下一輪的 trigger：think_ms 是人的思考（user）還是工具時長（tool）"),
    ("gap_ms", pa.int64(), True, "next_t_start_ms − t_start_ms（到達間隔，含服務時間）"),
    ("t_start_source", pa.string(), False, "trace／estimated（WildChat 只有回應完成時刻）"),
    ("think_source", pa.string(), False, "trace／estimated／not_available"),
    ("service_est_ms", pa.int64(), True, "估計的服務時間（只有 estimated 的來源有）；公式在 manifest"),
    ("split_user", pa.string(), True, "依 user 的切分（FreeInference、WildChat）"),
    ("split_drop", pa.bool_(), False, "這一輪開始在它的切分邊界之後：用 split 時要丟掉（§7）"),
    ("split_censored", pa.bool_(), False, "在切分窗內看到的設限（邊界之後的下一輪＝設限）"),
    ("split_obs_end_ms", pa.int64(), False, "在切分窗內的觀察窗終點"),
    ("harness", pa.string(), True, "FreeInference 的 agent harness"),
    ("is_subagent", pa.bool_(), True, "FreeInference：子 agent session"),
    ("parent_session_id", pa.string(), True, "FreeInference：子 agent 的父 session"),
    ("next_waited_for", pa.string(), True, "FreeInference：下一個請求的 waited_for（tool／user／ask_user_tool／subagent）"),
    ("provider_prompt_tokens", pa.int64(), True, "FreeInference：provider 自己 tokenizer 的 prompt token"),
]
COLUMNS = SPEC + EXTRA
COL_NAMES = [c[0] for c in COLUMNS]
ARROW_SCHEMA = pa.schema([pa.field(n, t, nullable=nl) for n, t, nl, _ in COLUMNS])
TRIGGERS = {"user", "tool", "unknown"}
SPLITS = {"train", "valid", "test"}
SRC_FLAGS = {"trace", "estimated", "synthetic"}
THINK_FLAGS = {"trace", "estimated", "not_available", "synthetic"}
# 來源自己的額外欄位：名稱以 x_ 開頭，接在 COLUMNS 後面寫進 parquet（例如 synth 的真實答案與訊號特徵）


class DataAssertionError(AssertionError):
    """規則 6 的斷言不過。帶著全部 checks，讓 build.py 照錄後停下（不寫 turns.parquet）。"""

    def __init__(self, msg, checks):
        super().__init__(msg)
        self.checks = checks


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Checks:
    """斷言紀錄。hard=True 的不過就在 enforce() 丟錯；hard=False 只報告。"""

    def __init__(self, source, variant):
        self.source, self.variant, self.rows = source, variant, []

    def add(self, name, value, threshold, passed, note="", hard=True):
        v = value
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        self.rows.append(dict(check=name, value=v, threshold=str(threshold), passed=bool(passed),
                              hard=bool(hard), note=note))
        return bool(passed)

    def ge(self, name, value, thr, note="", hard=True):
        return self.add(name, value, f">={thr}", (value == value) and value >= thr, note, hard)

    def eq(self, name, value, thr, note="", hard=True):
        return self.add(name, value, f"=={thr}", value == thr, note, hard)

    def rng(self, name, value, lo, hi, note="", hard=True):
        return self.add(name, value, f"{lo}..{hi}", (value == value) and lo <= value <= hi, note, hard)

    def report(self, name, value, note=""):
        return self.add(name, value, "report", True, note, hard=False)

    def failed(self):
        return [r for r in self.rows if r["hard"] and not r["passed"]]

    def enforce(self):
        bad = self.failed()
        if bad:
            raise DataAssertionError(f"{self.source}/{self.variant}: {len(bad)} 條斷言不過：{bad}", self.rows)

    def summary(self):
        hard = [r for r in self.rows if r["hard"]]
        return dict(n_hard=len(hard), n_hard_pass=sum(r["passed"] for r in hard), n_report=len(self.rows) - len(hard))


# ----------------------------------------------------------------------------- 雜湊、版本
def sha256_file(path, bufsize=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def code_fingerprint(files=None):
    """loader 的 git hash＋檔案 sha256（p3_data 還沒 commit，光 git hash 不夠）。"""
    files = files or sorted(os.path.join(HERE, f) for f in os.listdir(HERE) if f.endswith(".py"))
    try:
        head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "code/p3_data"], capture_output=True, text=True,
                               timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001 —— 查不到要讓人知道（規則 7）
        head, dirty = f"GIT_FAILED: {e!r}", ""
    return dict(git_head=head, p3_data_uncommitted=bool(dirty),
                files={os.path.basename(f): sha256_file(f) for f in files})


def readme_sha256(readme_path):
    """各資料目錄 README.txt 列的「sha256 bytes path」→ {path: sha256}。"""
    out = {}
    if not os.path.exists(readme_path):
        return out
    for line in open(readme_path, encoding="utf-8"):
        p = line.split()
        if len(p) == 3 and len(p[0]) == 64 and p[1].isdigit():
            out[p[2]] = p[0]
    return out


def input_record(path, expected_sha=None):
    s = sha256_file(path)
    return dict(path=path, bytes=os.path.getsize(path), sha256=s, expected_sha256=expected_sha,
                sha256_match=(None if expected_sha is None else s == expected_sha))


# ----------------------------------------------------------------------------- 主要計算
def finalize(df, chain=True, parent_col="parent_row", by_time=True):
    """由 session_id、t_start_ms、t_end_ms、t_end_source、trigger、obs_end_ms 算標籤。

    chain=True：同一 session 依 (t_start_ms, _order) 排，父＝上一列。
    chain=False：df[parent_col] 是父列的位置（df 目前的位置，−1＝根）；樹狀（Bailian、Mooncake）。
    「下一輪」＝最早的子列（依 t_start_ms、再依原始順序）。回傳新的 DataFrame（列的順序不變）與統計 dict。
    estimated 的 t_end_ms 若超過下一輪開始或觀察窗終點，就截到那裡（計數回報；trace 的 t_end 不截）。
    """
    n = len(df)
    df = df.reset_index(drop=True)
    if "_order" not in df.columns:
        df["_order"] = np.arange(n)
    if by_time:
        order = np.lexsort((df["_order"].values, df["t_start_ms"].values, df["session_id"].values))
    else:  # 依原始順序（FreeInference 的 step）：session 內不重排
        order = np.lexsort((df["_order"].values, df["session_id"].values))
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)
    sid_s = df["session_id"].values[order]
    new_sess = np.r_[True, sid_s[1:] != sid_s[:-1]]
    # turn_idx（排序後的位置 → session 內序號）
    grp_start = np.maximum.accumulate(np.where(new_sess, np.arange(n), 0))
    turn_sorted = np.arange(n) - grp_start
    turn_idx = np.empty(n, dtype=np.int64)
    turn_idx[order] = turn_sorted
    if chain:
        par_sorted = np.where(new_sess, -1, np.arange(n) - 1)
        par = np.full(n, -1, dtype=np.int64)
        par[order] = np.where(par_sorted >= 0, order[np.maximum(par_sorted, 0)], -1)
    else:
        par = df[parent_col].values.astype(np.int64)
    c = np.nonzero(par >= 0)[0]
    p = par[c]
    # 父子一定同 session、父不晚於子
    assert np.all(df["session_id"].values[c] == df["session_id"].values[p]), "父子不在同一 session"
    back = df["t_start_ms"].values[p] > df["t_start_ms"].values[c]
    if by_time:
        assert not back.any(), "父請求晚於子請求"
    stats = {"child_starts_before_parent": int(back.sum())}
    # 最早的子：子依排序位置（inv）取最小
    first_child = np.full(n, -1, dtype=np.int64)
    if len(c):
        s = pd.Series(inv[c]).groupby(p).min()
        first_child[s.index.values] = order[s.values]
    n_children = np.bincount(p, minlength=n).astype(np.int64) if len(c) else np.zeros(n, dtype=np.int64)
    has = first_child >= 0
    t_start = df["t_start_ms"].values.astype(np.int64)
    nxt = np.where(has, t_start[np.maximum(first_child, 0)], 0)
    trig = df["trigger"].values
    obs_end = df["obs_end_ms"].values.astype(np.int64)
    assert np.all(obs_end >= t_start), "obs_end_ms 早於 t_start_ms"
    assert np.all(nxt[has] <= obs_end[has]), "下一輪在觀察窗之外"
    # 估計的 t_end：截到下一輪開始與觀察窗終點
    t_end = df["t_end_ms"].astype("Int64")
    est = (df["t_end_source"].values == "estimated")
    te = t_end.to_numpy(dtype="float64", na_value=np.nan)
    over_next = est & has & ~np.isnan(te) & (te > nxt)
    te = np.where(over_next, nxt, te)
    over_obs = est & ~np.isnan(te) & (te > obs_end)
    te = np.where(over_obs, obs_end, te)
    stats["t_end_est_clipped_to_next"] = int(over_next.sum())
    stats["t_end_est_clipped_to_obs_end"] = int(over_obs.sum())
    stats["n_estimated_t_end"] = int(est.sum())
    out = df.copy()
    out["t_end_ms"] = pd.Series(te).round().astype("Int64").values
    out["turn_idx"] = turn_idx.astype(np.int32)
    out["parent_turn_idx"] = pd.array(np.where(par >= 0, turn_idx[np.maximum(par, 0)], -1), dtype="Int32")
    out.loc[par < 0, "parent_turn_idx"] = pd.NA
    out["n_children"] = n_children.astype(np.int32)
    out["next_t_start_ms"] = pd.array(nxt, dtype="Int64")
    out.loc[~has, "next_t_start_ms"] = pd.NA
    out["next_trigger"] = pd.array(np.where(has, trig[np.maximum(first_child, 0)], None), dtype=object)
    out["gap_ms"] = out["next_t_start_ms"] - pd.array(t_start, dtype="Int64")
    out["think_ms"] = out["next_t_start_ms"] - out["t_end_ms"]
    out["censored"] = ~has
    stats["n_rows"] = n
    stats["n_with_next"] = int(has.sum())
    stats["n_fork_parents"] = int((n_children > 1).sum())
    stats["max_children"] = int(n_children.max()) if n else 0
    th = out["think_ms"].dropna()
    stats["think_negative"] = int((th < 0).sum())
    return out, stats


def session_start(df):
    return df.groupby("session_id")["t_start_ms"].transform("min").values


def time_split(df, t0, t1, fracs=(0.6, 0.2, 0.2)):
    """§7：依 session 開始時刻落在 trace 跨度的前 60%／中 20%／後 20%。回傳 (split 陣列, 邊界 dict)。"""
    span = t1 - t0
    b1 = int(t0 + fracs[0] * span)
    b2 = int(t0 + (fracs[0] + fracs[1]) * span)
    s0 = session_start(df)
    split = np.where(s0 < b1, "train", np.where(s0 < b2, "valid", "test"))
    seg_end = {"train": b1, "valid": b2, "test": None}
    return split, dict(t0=int(t0), t1=int(t1), b_train_valid=b1, b_valid_test=b2, fracs=list(fracs)), seg_end


def user_split(user_ids, fracs=(0.8, 0.1, 0.1), salt="p3_data_user_split_v1"):
    """依 user 切：sha256(salt+user_id) 的前 8 bytes ÷ 2^64 落在哪一段。null user → null。"""
    out = []
    cache = {}
    for u in user_ids:
        if u is None or (isinstance(u, float) and np.isnan(u)):
            out.append(None)
            continue
        if u not in cache:
            x = int.from_bytes(hashlib.sha256((salt + str(u)).encode()).digest()[:8], "big") / 2 ** 64
            cache[u] = "train" if x < fracs[0] else ("valid" if x < fracs[0] + fracs[1] else "test")
        out.append(cache[u])
    return np.array(out, dtype=object)


def apply_split_window(df, seg_end):
    """§7：跨切分邊界的 session 歸到它開始的那段；邊界之後的輪 split_drop=True；
    窗內最後一輪若下一輪在邊界之後 → split_censored=True、split_obs_end_ms＝邊界。"""
    end = np.array([seg_end.get(s) if seg_end.get(s) is not None else np.iinfo(np.int64).max for s in df["split"].values],
                   dtype=np.int64)
    t_start = df["t_start_ms"].values.astype(np.int64)
    drop = t_start >= end
    nxt = df["next_t_start_ms"].to_numpy(dtype="float64", na_value=np.nan)
    after = ~np.isnan(nxt) & (nxt >= end)
    df["split_drop"] = drop
    df["split_censored"] = df["censored"].values | after
    df["split_obs_end_ms"] = np.minimum(df["obs_end_ms"].values.astype(np.int64), end)
    return dict(split_drop_rows=int(drop.sum()), split_censored_extra=int((after & ~drop).sum()))


# ----------------------------------------------------------------------------- 統計
def pctl(x, qs=(10, 50, 90, 99)):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {f"p{q}": None for q in qs}
    v = np.percentile(x, qs)
    return {f"p{q}": float(r) for q, r in zip(qs, v)}


def describe(df):
    """manifest／summary.csv 用的統計。think_ms 分位數只算 next_trigger ∈ {user, unknown}（人的思考）；
    工具觸發的間隔另外列（§5：分開統計）。"""
    th = df["think_ms"].to_numpy(dtype="float64", na_value=np.nan)
    nt = df["next_trigger"].values
    human = np.isin(nt, ["user", "unknown"]) & ~np.isnan(th)
    tool = (nt == "tool") & ~np.isnan(th)
    q_h = pctl(th[human])
    q_t = pctl(th[tool])
    cens = df["censored"].values
    rem = (df["obs_end_ms"].values.astype(float) - df["t_end_ms"].to_numpy(dtype="float64", na_value=np.nan))
    p90 = q_h["p90"]
    short = (cens & (rem < p90)).mean() if p90 is not None else float("nan")
    d = dict(
        n_rows=int(len(df)), n_sessions=int(df["session_id"].nunique()),
        n_users=int(df["user_id"].dropna().nunique()),
        n_turns_user=int((df["trigger"] == "user").sum()), n_turns_tool=int((df["trigger"] == "tool").sum()),
        n_turns_unknown=int((df["trigger"] == "unknown").sum()),
        n_sessions_ge2=int((df.groupby("session_id").size() >= 2).sum()),
        censored_frac=float(cens.mean()),
        censored_short_frac=float(short),
        n_think_human=int(human.sum()), n_think_tool=int(tool.sum()),
        think_ms_p10=q_h["p10"], think_ms_p50=q_h["p50"], think_ms_p90=q_h["p90"], think_ms_p99=q_h["p99"],
        tool_gap_ms_p50=q_t["p50"], tool_gap_ms_p90=q_t["p90"],
        think_negative=int((th < 0).sum()),
        t_end_source=",".join(sorted(set(df["t_end_source"].unique()))),
        think_source=",".join(sorted(set(df["think_source"].unique()))),
        split_counts={s: int((df["split"] == s).sum()) for s in ("train", "valid", "test")},
    )
    return d


# ----------------------------------------------------------------------------- 驗證與寫檔
def validate(df):
    """欄位齊全、型別可轉、值域合法。不合 → AssertionError（這是程式錯，不是資料錯）。"""
    missing = [c for c in COL_NAMES if c not in df.columns]
    assert not missing, f"缺欄位 {missing}"
    for n, _, nullable, _ in COLUMNS:
        if not nullable:
            assert not df[n].isna().any(), f"{n} 不准 null"
    assert set(df["trigger"].unique()) <= TRIGGERS, set(df["trigger"].unique())
    assert set(df["split"].unique()) <= SPLITS, set(df["split"].unique())
    assert set(df["t_end_source"].unique()) <= SRC_FLAGS
    assert set(df["t_start_source"].unique()) <= SRC_FLAGS
    assert set(df["think_source"].unique()) <= THINK_FLAGS
    k = df[["session_id", "turn_idx"]]
    assert not k.duplicated().any(), "(session_id, turn_idx) 重複"
    # censored ⇔ 沒有下一輪
    assert (df["censored"].values == df["next_t_start_ms"].isna().values).all()
    # think_ms＝next − t_end（有值時）
    m = df["think_ms"].notna()
    assert (df.loc[m, "think_ms"] == df.loc[m, "next_t_start_ms"] - df.loc[m, "t_end_ms"]).all()
    # 每個 session 只有一個 split
    assert (df.groupby("session_id")["split"].nunique() == 1).all(), "session 跨切分"
    return True


def to_arrow(df, block_col=None):
    """DataFrame → Arrow table（照 ARROW_SCHEMA）。block_col：另外給的 Arrow 陣列（FreeInference 不進 pandas）。"""
    arrs = []
    for n, t, _, _ in COLUMNS:
        if n == "block_ids":
            if block_col is not None:
                arrs.append(block_col)
            elif df[n].isna().all():
                arrs.append(pa.nulls(len(df), type=t))
            else:
                arrs.append(pa.array([None if v is None or (isinstance(v, float) and np.isnan(v)) else list(v)
                                      for v in df[n].tolist()], type=t))
            continue
        s = df[n]
        if pa.types.is_string(t):
            arrs.append(pa.array(s.astype(object).where(s.notna(), None), type=t, from_pandas=True))
        else:
            arrs.append(pa.array(s, type=t, from_pandas=True))
    fields = list(ARROW_SCHEMA)
    for c in sorted(c for c in df.columns if c.startswith("x_")):
        a = pa.array(df[c].astype(object).where(df[c].notna(), None) if df[c].dtype == object else df[c], from_pandas=True)
        arrs.append(a)
        fields.append(pa.field(c, a.type, nullable=True))
    return pa.Table.from_arrays(arrs, schema=pa.schema(fields))


def variant_dir(source, variant):
    return os.path.join(UNIFIED, source, variant)


def test_rows_sha256(df):
    """§7：測試集凍結時記雜湊（session_id、turn_idx、t_start_ms，排序後）。"""
    t = df.loc[df["split"] == "test", ["session_id", "turn_idx", "t_start_ms"]].sort_values(["session_id", "turn_idx"])
    h = hashlib.sha256()
    for a, b, c in t.itertuples(index=False):
        h.update(f"{a}\t{b}\t{c}\n".encode())
    return h.hexdigest(), int(len(t))


def write_manifest(path, man):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


def write_variant(df, source, variant, checks, inputs, extra_manifest, tier="R", writer=None):
    """validate → enforce 斷言 → 寫 turns.parquet＋manifest.json。writer：自訂寫檔函式（FreeInference 串流 block_ids）。"""
    checks.enforce()
    validate(df)
    d = variant_dir(source, variant)
    os.makedirs(d, exist_ok=True)
    pth = os.path.join(d, "turns.parquet")
    tmp = pth + ".tmp"
    if writer is None:
        pq.write_table(to_arrow(df), tmp, compression="zstd")
    else:
        writer(df, tmp)
    os.replace(tmp, pth)
    desc = describe(df)
    th, nt = test_rows_sha256(df)
    man = dict(schema_version=SCHEMA_VERSION, source=source, variant=variant, tier=tier,
               run_id=os.environ.get("RUN_ID", ""), ts=now_iso(), output=pth, output_sha256=sha256_file(pth),
               columns=[dict(name=n, type=str(t), nullable=nl, doc=doc) for n, t, nl, doc in COLUMNS] +
               [dict(name=c, type=str(df[c].dtype), nullable=True, doc=extra_manifest.get("x_columns_doc", {}).get(c, ""))
                for c in sorted(c for c in df.columns if c.startswith("x_"))],
               inputs=inputs, code=code_fingerprint(), checks=checks.rows, checks_summary=checks.summary(),
               stats=desc, test_rows_sha256=th, test_rows=nt, **extra_manifest)
    write_manifest(os.path.join(d, "manifest.json"), man)
    return man


def read_unified(source, variant, columns=None):
    return pd.read_parquet(os.path.join(variant_dir(source, variant), "turns.parquet"), columns=columns)


def read_manifest(source, variant):
    return json.load(open(os.path.join(variant_dir(source, variant), "manifest.json"), encoding="utf-8"))


def log(*a):
    print(f"[{now_iso()}]", *a, flush=True, file=sys.stderr)
