"""x9_window.py — X9 的 T3：1–2 小時的 trace 把「晚點回來」當成「不回來」的比例（WildChat-1M）。

判準在 docs/phase3_20261017/explore/X9_wild_ideas.md §2.2 T3（2026-10-17T09:45:06Z 寫定）。
無參數：M_W = Σ_{g 有限} min(1, g/W) / Σ_{全部} min(1, g/W)，g＝∞（最後一輪）時 min(1,g/W)=1。

用法： python -I code/p3x/x9_window.py --out results/p3_explore/x9 --rundir $RUN_DIR
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

DATA = "/ssd8/hungwei/paper-hkv/datasets/wildchat-1m/data/train-*.parquet"
RUN_ID = os.environ.get("RUN_ID", "")
WINDOWS_S = {"1h": 3600.0, "2h": 7200.0}
US = 1e6


def ts():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(*a):
    print(f"[{ts()}]", *a, flush=True)


def one_file(path):
    tb = pq.read_table(path, columns=["conversation_hash", "timestamp", "turn", "conversation"])
    n_conv = tb.num_rows
    conv = tb.column("conversation").combine_chunks()
    parent = pc.list_parent_indices(conv).to_numpy()
    flat = pc.list_flatten(conv)
    role = pc.struct_field(flat, "role").to_numpy(zero_copy_only=False)
    mts = pc.struct_field(flat, "timestamp")
    ts_null = pc.is_null(mts).to_numpy(zero_copy_only=False)
    ts_us = pc.cast(pc.fill_null(mts, 0), "int64").to_numpy()
    top_turn = tb.column("turn").to_numpy()
    top_ts = pc.cast(tb.column("timestamp"), "int64").to_numpy()
    is_user = role == "user"
    is_asst = role == "assistant"
    # 規則 6 (a)：頂層 turn ＝ user 訊息數
    n_user = np.bincount(parent[is_user], minlength=n_conv)
    turn_ok = int(np.sum(n_user == top_turn))
    # 哪些 role 有時間戳
    role_ts = {}
    for r in ("user", "assistant"):
        m = role == r
        role_ts[r] = dict(n=int(m.sum()), n_ts_nonnull=int((m & ~ts_null).sum()))
    # 規則 6 (b)：同一對話內有時間戳的訊息不遞減
    has = ~ts_null
    p_h, t_h = parent[has], ts_us[has]
    same = p_h[1:] == p_h[:-1]
    nondec = int(np.sum(t_h[1:][same] >= t_h[:-1][same]))
    n_pairs = int(same.sum())
    # 規則 6 (c)：頂層 timestamp ＝最後一個有時間戳訊息
    last_ts = np.full(n_conv, -1, dtype=np.int64)
    np.maximum.at(last_ts, p_h, t_h)  # 最大的時間戳（不依賴重複索引的寫入順序）
    top_eq = int(np.sum(last_ts == top_ts))
    # 回合結束＝assistant 訊息；回來＝下一個 assistant 訊息（user 沒有時間戳時，照 §2.2）
    use_user = role_ts["user"]["n_ts_nonnull"] >= 0.99 * role_ts["user"]["n"] and role_ts["user"]["n"] > 0
    am = is_asst & has
    pa, ta = parent[am], ts_us[am]
    if use_user:
        um = is_user & has
        pu, tu = parent[um], ts_us[um]
        # 第 j 個 assistant 的回來＝第 j+1 個 user：以對話內的序號對齊
        rank_a = _rank(pa)
        rank_u = _rank(pu)
        key_u = {(int(p), int(r)): int(t) for p, r, t in zip(pu, rank_u, tu)}
        gaps, finals = [], 0
        for p, r, t in zip(pa, rank_a, ta):
            nxt = key_u.get((int(p), int(r) + 1))
            if nxt is None:
                finals += 1
            else:
                gaps.append((nxt - t) / US)
        gaps = np.array(gaps)
        conv_of_gap = None
    else:
        samea = pa[1:] == pa[:-1]
        gaps = ((ta[1:] - ta[:-1]) / US)[samea]
        finals = int(len(np.unique(pa)))
        conv_of_gap = pa[:-1][samea]
    # 有 ≥2 個 assistant 訊息的對話
    na = np.bincount(pa, minlength=n_conv)
    finals_multi = int(np.sum(na >= 2))
    return dict(path=os.path.basename(path), n_conv=n_conv, turn_ok=turn_ok, role_ts=role_ts,
                nondec=nondec, n_pairs=n_pairs, top_eq=top_eq, use_user=bool(use_user),
                gaps=gaps, finals=finals, finals_multi=finals_multi,
                n_conv_with_asst_ts=int(np.sum(na >= 1)))


def _rank(p):
    r = np.zeros(len(p), dtype=np.int64)
    for i in range(1, len(p)):
        r[i] = r[i - 1] + 1 if p[i] == p[i - 1] else 0
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    assert RUN_ID, "RUN_ID 未設"
    files = sorted(glob.glob(DATA))
    log("files", len(files))
    assert len(files) == 14, f"WildChat-1M 應有 14 個 parquet，找到 {len(files)}"
    with ProcessPoolExecutor(a.procs) as ex:
        res = list(ex.map(one_file, files))
    n_conv = sum(r["n_conv"] for r in res)
    turn_rate = sum(r["turn_ok"] for r in res) / n_conv
    nondec_rate = sum(r["nondec"] for r in res) / max(1, sum(r["n_pairs"] for r in res))
    top_rate = sum(r["top_eq"] for r in res) / n_conv
    role_ts = {k: dict(n=sum(r["role_ts"][k]["n"] for r in res),
                       n_ts_nonnull=sum(r["role_ts"][k]["n_ts_nonnull"] for r in res)) for k in ("user", "assistant")}
    use_user = {r["use_user"] for r in res}
    chk = dict(n_conv=n_conv, turn_eq_user_count_rate=turn_rate, ts_nondecreasing_rate=nondec_rate,
               top_ts_eq_last_msg_rate=top_rate, role_ts=role_ts, return_time_from=("user" if use_user == {True} else
                                                                                   "assistant" if use_user == {False} else "mixed"))
    log("checks", json.dumps(chk))
    # 規則 6 斷言
    assert turn_rate >= 0.99, f"頂層 turn ＝ user 訊息數 只有 {turn_rate:.4f}"
    assert nondec_rate >= 0.99, f"時間戳不遞減 只有 {nondec_rate:.4f}"
    gaps = np.concatenate([r["gaps"] for r in res])
    finals = sum(r["finals"] for r in res)
    finals_multi = sum(r["finals_multi"] for r in res)
    neg = int(np.sum(gaps < 0))
    g = np.clip(gaps, 0, None)
    rows = []
    now = ts()
    for wname, W in WINDOWS_S.items():
        w = np.minimum(1.0, g / W)
        num = float(w.sum())
        M = num / (num + finals)
        M_multi = num / (num + finals_multi)
        rows.append(dict(run_id=RUN_ID, ts=now, dataset="wildchat-1m", window=wname, W_s=W,
                         n_conv=n_conv, n_turn_ends=int(len(g) + finals), n_finite_gaps=int(len(g)),
                         n_final=finals, n_final_multi_turn=finals_multi,
                         expected_censored_returning=num, M_W=M, M_W_multi_turn_only=M_multi,
                         n_negative_gaps_clipped=neg))
    qs = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    gq = np.quantile(g, qs) if len(g) else [float("nan")] * len(qs)
    dist = dict(run_id=RUN_ID, ts=now, dataset="wildchat-1m", n_finite_gaps=int(len(g)),
                **{f"gap_q{int(q*100)}_s": float(v) for q, v in zip(qs, gq)},
                frac_gt_1h=float(np.mean(g > 3600)), frac_gt_2h=float(np.mean(g > 7200)),
                frac_gt_24h=float(np.mean(g > 86400)), frac_gt_5min=float(np.mean(g > 300)),
                frac_gt_10min=float(np.mean(g > 600)), return_time_from=chk["return_time_from"],
                turn_eq_user_count_rate=turn_rate, ts_nondecreasing_rate=nondec_rate, top_ts_eq_last_msg_rate=top_rate,
                user_ts_nonnull=role_ts["user"]["n_ts_nonnull"], user_msgs=role_ts["user"]["n"],
                asst_ts_nonnull=role_ts["assistant"]["n_ts_nonnull"], asst_msgs=role_ts["assistant"]["n"])
    os.makedirs(a.out, exist_ok=True)
    for fn, rs in (("t3_window_censoring.csv", rows), ("t3_gap_distribution.csv", [dist])):
        with open(os.path.join(a.out, fn), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rs[0].keys()))
            w.writeheader()
            w.writerows(rs)
        log("寫入", fn, json.dumps(rs))
    np.save(os.path.join(a.rundir, "t3_gaps_s.npy"), gaps)


if __name__ == "__main__":
    main()
