"""test_p3_data.py — 統一載入層的測試（03_data_design.md §8）。

  CUDA_VISIBLE_DEVICES="" python -I -m pytest -q code/p3_data/test_p3_data.py        （要先 cd 到 repo；-I 下用 -m pytest）

分三組：
  1. 單元：finalize（鏈／樹／估的 t_end 截斷）、切分窗、Checks.enforce。
  2. 斷言會不會真的丟錯：造一份單位錯的假 Bailian／Mooncake 檔（16 vs 512 token，CLAUDE.md 規則 6 的那次事故），loader 必須丟 DataAssertionError。
  3. 合成：同參數同 seed 逐位元相同（parquet 位元組）；不同 seed 不同；ρ＝0／1 的訊號。
  4. 已轉好的統一表（有才跑）：每個 manifest 的 hard 斷言全過、validate 通過、FreeInference 串流寫入的 block_ids 對齊、
     和文獻 agent／X 系列的小樣本對拍（session 數、分位數）。
"""
from __future__ import annotations

import glob
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import schema as S  # noqa: E402


def _base(n, **kw):
    d = dict(source="t", variant="v", user_id=None, session_rule="r", trigger="user", t_end_source="trace",
             obs_end_ms=10_000, _order=np.arange(n))
    d.update(kw)
    return pd.DataFrame(d)


# ----------------------------------------------------------------------------- 1. 單元
def test_finalize_chain():
    df = _base(5, session_id=["a", "a", "b", "a", "b"], t_start_ms=[0, 100, 50, 300, 70],
               t_end_ms=pd.array([10, 150, 60, 310, 80], dtype="Int64"))
    out, st = S.finalize(df, chain=True)
    assert out["turn_idx"].tolist() == [0, 1, 0, 2, 1]
    assert out["next_t_start_ms"].tolist()[:3] == [100, 300, 70]
    assert pd.isna(out["next_t_start_ms"].iloc[3]) and pd.isna(out["next_t_start_ms"].iloc[4])
    assert out["think_ms"].tolist()[:3] == [90, 150, 10]
    assert out["censored"].tolist() == [False, False, False, True, True]
    assert out["parent_turn_idx"].tolist()[1] == 0 and pd.isna(out["parent_turn_idx"].iloc[0])


def test_finalize_tree_earliest_child_and_fork():
    # 0 是根；1、2 都是 0 的子（分叉），2 比 1 早；3 是 1 的子
    df = _base(4, session_id=["s"] * 4, t_start_ms=[0, 500, 200, 900], t_end_ms=pd.array([10, 510, 210, 910], dtype="Int64"),
               parent_row=[-1, 0, 0, 1])
    out, st = S.finalize(df, chain=False)
    assert out["n_children"].tolist() == [2, 1, 0, 0]
    assert out["next_t_start_ms"].iloc[0] == 200          # 最早的子
    assert out["next_t_start_ms"].iloc[1] == 900
    assert out["censored"].tolist() == [False, False, True, True]
    assert st["n_fork_parents"] == 1


def test_estimated_t_end_clipped_to_next():
    df = _base(2, session_id=["s", "s"], t_start_ms=[0, 100], t_end_ms=pd.array([500, 150], dtype="Int64"),
               t_end_source="estimated")
    out, st = S.finalize(df, chain=True)
    assert out["t_end_ms"].iloc[0] == 100 and out["think_ms"].iloc[0] == 0
    assert st["t_end_est_clipped_to_next"] == 1


def test_trace_t_end_not_clipped():
    df = _base(2, session_id=["s", "s"], t_start_ms=[0, 100], t_end_ms=pd.array([500, 150], dtype="Int64"))
    out, st = S.finalize(df, chain=True)
    assert out["think_ms"].iloc[0] == -400 and st["think_negative"] == 1   # 實測值不改，只計數


def test_split_window():
    df = _base(4, session_id=["a", "a", "a", "b"], t_start_ms=[10, 50, 80, 70],
               t_end_ms=pd.array([11, 51, 81, 71], dtype="Int64"), obs_end_ms=100)
    out, _ = S.finalize(df, chain=True)
    split, bounds, seg_end = S.time_split(out, 0, 100)
    out["split"] = split
    S.apply_split_window(out, seg_end)
    assert out["split"].tolist() == ["train", "train", "train", "valid"]
    assert out["split_drop"].tolist() == [False, False, True, False]        # 第 3 輪在 train 邊界（60）之後
    assert out["split_censored"].tolist() == [False, True, True, True]       # 第 2 輪的下一輪在邊界之後 → 設限
    assert out["split_obs_end_ms"].tolist()[1] == 60


def test_user_split_deterministic():
    a = S.user_split(["u1", "u2", None, "u1"])
    b = S.user_split(["u1", "u2", None, "u1"])
    assert list(a) == list(b) and a[0] == a[3] and a[2] is None


def test_checks_enforce():
    ck = S.Checks("t", "v")
    ck.ge("ok", 1.0, 0.95)
    ck.report("info", 0.1)
    ck.enforce()
    ck.ge("bad", 0.5, 0.95)
    with pytest.raises(S.DataAssertionError):
        ck.enforce()


# ----------------------------------------------------------------------------- 2. 斷言真的會丟錯
CONST = dict(A=1000.0, B=10.0, C=0.01, weeks=["test"])


def _fake_rows(block, n=40):
    rng = np.random.default_rng(0)
    rows, t = [], 0
    for i in range(n):
        L = int(rng.integers(600, 3000))
        t += int(rng.integers(1000, 5000))
        rows.append(dict(chat_id=i, parent_chat_id=-1, timestamp=t, input_length=L,
                         output_length=100, type="text", turn=1, hash_ids=list(range(i * 1000, i * 1000 + -(-L // block)))))
    return rows


def test_bailian_wrong_unit_raises(tmp_path, monkeypatch):
    import load_bailian as LB
    rows = _fake_rows(512)      # 用 512-token 的 block 數，但 Bailian 應該是 16
    for r in rows:
        r["timestamp"] = r["timestamp"] / 1000
    p = tmp_path / "qwen_traceA_blksz_16.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    monkeypatch.setattr(LB, "DIR", str(tmp_path))
    with pytest.raises(S.DataAssertionError) as e:
        LB.load("traceA", CONST)
    bad = {c["check"] for c in e.value.checks if c["hard"] and not c["passed"]}
    assert "len(hash_ids)==ceil(input_length/16)" in bad


def test_mooncake_wrong_unit_raises(tmp_path, monkeypatch):
    import load_mooncake as LM
    rows = [dict(timestamp=r["timestamp"], input_length=r["input_length"], output_length=r["output_length"], hash_ids=r["hash_ids"])
            for r in _fake_rows(16)]      # 16-token 的 block 數，但 Mooncake 是 512（2026-08-31 的事故反過來）
    (tmp_path / "traces").mkdir()
    (tmp_path / "traces" / "conversation_trace.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    monkeypatch.setattr(LM, "DIR", str(tmp_path))
    with pytest.raises(S.DataAssertionError) as e:
        LM.load("conv-rule-strict", CONST)
    bad = {c["check"] for c in e.value.checks if c["hard"] and not c["passed"]}
    assert "len(hash_ids)==ceil(input_length/512)" in bad


# ----------------------------------------------------------------------------- 3. 合成
def _synth(seed, n=300):
    import synth as SY
    fj = json.load(open(SY.FIT_PATH)) if os.path.exists(SY.FIT_PATH) else None
    if fj is None:
        pytest.skip("synth_fit_fi_week01.json 不存在")
    const = json.load(open(SY.CONST_PATH))
    p = dict(SY.DEFAULTS, n_sessions=n, seed=seed, horizon_s=86400, variant="t")
    df, _ = SY.generate(p, fj, const)
    buf = io.BytesIO()
    pq.write_table(S.to_arrow(df), buf, compression="zstd")
    return df, buf.getvalue()


def test_synth_bitwise_reproducible():
    import hashlib
    d1, b1 = _synth(7)
    d2, b2 = _synth(7)
    d3, b3 = _synth(8)
    assert hashlib.sha256(b1).hexdigest() == hashlib.sha256(b2).hexdigest()
    pd.testing.assert_frame_equal(d1, d2)
    assert hashlib.sha256(b1).hexdigest() != hashlib.sha256(b3).hexdigest()
    S.validate(d1)


def test_synth_signal_strength():
    from scipy.stats import spearmanr
    d, _ = _synth(3, n=2000)
    key = np.where(d["x_true_next_class"] == "end", np.inf, d["x_true_think_ms"].to_numpy(dtype="float64", na_value=np.inf))
    assert abs(spearmanr(d["x_feat_rho_000"], key).statistic) < 0.05
    assert spearmanr(d["x_feat_rho_100"], key).statistic > 0.999


# ----------------------------------------------------------------------------- 4. 已轉好的統一表
MANIFESTS = sorted(glob.glob(os.path.join(S.UNIFIED, "*", "*", "manifest.json")))
need = pytest.mark.skipif(not MANIFESTS, reason="還沒有轉好的統一表")


@need
@pytest.mark.parametrize("mp", MANIFESTS, ids=[os.path.relpath(m, S.UNIFIED) for m in MANIFESTS])
def test_manifest_hard_checks_pass_and_valid(mp):
    m = json.load(open(mp))
    assert m["checks_summary"]["n_hard"] == m["checks_summary"]["n_hard_pass"]
    cols = [c for c in S.COL_NAMES if c != "block_ids"]
    df = pd.read_parquet(m["output"], columns=cols)
    df["block_ids"] = None
    S.validate(df)
    assert S.sha256_file(m["output"]) == m["output_sha256"]


@need
def test_freeinference_block_ids_aligned():
    """串流寫入的 block_ids 要和同一列的 input_tokens 對得上（floor(Σrole_tokens/16)）。"""
    p = os.path.join(S.UNIFIED, "freeinference", "week01", "turns.parquet")
    if not os.path.exists(p):
        pytest.skip("freeinference/week01 未轉")
    t = pq.read_table(p, columns=["input_tokens", "block_ids", "session_id", "turn_idx"])
    nb = np.array([len(x) for x in t.column("block_ids").to_pylist()])
    inp = t.column("input_tokens").to_numpy()
    assert np.mean(nb == inp // 16) >= 0.95


def _lit_calc(name, key):
    """research/calc 的文獻 agent 輸出（獨立的程式）。"""
    p = os.path.join(S.REPO, "docs", "phase3_20261017", "research", "calc", name)
    return [l for l in open(p, encoding="utf-8") if key in l]


@need
def test_crosscheck_freeinference_week01_vs_lit():
    """文獻 agent 第 1 週：125 session、785 使用者發起、9,953 工具觸發；使用者發起的下一輪 p10／50／90／99＝25／175／3,587／34,687 s。"""
    p = os.path.join(S.UNIFIED, "freeinference", "week01", "turns.parquet")
    if not os.path.exists(p):
        pytest.skip()
    d = pd.read_parquet(p, columns=["session_id", "trigger", "next_trigger", "think_ms"])
    assert d["session_id"].nunique() == 125
    assert (d["trigger"] == "user").sum() == 785 and (d["trigger"] == "tool").sum() == 9953
    q = np.percentile(d.loc[d["next_trigger"] == "user", "think_ms"].astype(float) / 1000, [10, 50, 90, 99])
    assert np.allclose(q, [25, 175, 3587, 34687], rtol=0.005)


@need
def test_crosscheck_bailian_thinking_vs_lit():
    """文獻 agent：thinking 9,612 個根、1,200 個子；子減父 p10／50／90／99＝5.1／28.6／255.3／1,533.3 s。"""
    p = os.path.join(S.UNIFIED, "bailian", "thinking", "turns.parquet")
    if not os.path.exists(p):
        pytest.skip()
    d = pd.read_parquet(p, columns=["session_id", "parent_turn_idx", "gap_ms"])
    assert d["session_id"].nunique() == 9612 and d["parent_turn_idx"].notna().sum() == 1200
    q = np.percentile(d["gap_ms"].dropna().astype(float) / 1000, [10, 50, 90, 99])
    assert np.allclose(q, [5.1, 28.6, 255.3, 1533.3], atol=0.06)


@need
@pytest.mark.parametrize("variant,frac", [("conv-rule-loose", 0.4393649738176378), ("conv-rule-strict", 0.20197822292411272),
                                          ("tool-rule-loose", 0.4407827854964419), ("tool-rule-strict", 0.10623517451711284)])
def test_crosscheck_mooncake_return_frac_vs_lit(variant, frac):
    p = os.path.join(S.UNIFIED, "mooncake", variant, "turns.parquet")
    if not os.path.exists(p):
        pytest.skip()
    d = pd.read_parquet(p, columns=["n_children"])
    assert abs((d["n_children"] > 0).mean() - frac) < 1e-12


# ----------------------------------------------------------------------------- 5. 跨請求的原始 id 鏈（只報告；06_simulator §10-2）
def test_raw_id_chain_report_unit():
    import load_bailian as LB
    chained = [[1, 2, 3], [1, 2, 4], [1, 5]]
    assert LB.raw_id_chain_report(chained) == dict(n_ids=5, n_ids_multi_position=0, position_violations=0, prev_id_violations=0)
    broken = [[1, 2, 3], [2, 3], [9, 1]]      # 2、3、1 都換了位置；前一個 id 換了的：2（None）、1（9）
    r = LB.raw_id_chain_report(broken)
    assert r["n_ids_multi_position"] == 3 and r["position_violations"] == 3 and r["prev_id_violations"] == 2


@need
@pytest.mark.parametrize("variant,multi,posv", [("traceA", 52168, 220196), ("coder", 0, 0), ("thinking", 0, 0)])
def test_bailian_raw_id_chain(variant, multi, posv):
    """06 §10-2 記的值：traceA 52,168 個 id 出現在不只一個位置、位置違反 220,196 次；coder、thinking 都是 0。"""
    import load_bailian as LB
    p = os.path.join(S.UNIFIED, "bailian", variant, "turns.parquet")
    if not os.path.exists(p):
        pytest.skip()
    d = pd.read_parquet(p, columns=["block_ids"])
    r = LB.raw_id_chain_report(d["block_ids"])
    assert r["n_ids_multi_position"] == multi and r["position_violations"] == posv, r
