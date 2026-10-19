"""x9_freeinf.py — X9 T1／T2／T3 的 FreeInference 版（登記後加；判準見 X9 文件 §2.4，2026-10-17T10:03:00Z 寫定）。

只讀 processed/{requests,sessions}.parquet；不訓練、不估任何查表。
用法： python -I code/p3x/x9_freeinf.py --out results/p3_explore/x9 --rundir $RUN_DIR --procs 4
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = os.path.join(os.environ.get("TIARA_DATASETS", "/ssd8/hungwei/paper-hkv/datasets"), "freeinference_agentic_trace/release")
RUN_ID = os.environ.get("RUN_ID", "")
WINDOWS_S = {"1h": 3600.0, "2h": 7200.0}


def ts():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(*a):
    print(f"[{ts()}]", *a, flush=True)


def lcp_np(a, b):
    m = min(len(a), len(b))
    if m == 0:
        return 0
    ne = a[:m] != b[:m]
    k = int(np.argmax(ne))
    return m if not ne[k] else k


def lcp_chain(a, b):
    """鏈式 block（README：block id 由整段前綴決定）→ a[k]==b[k] 蘊含所有 j<k 也相等，所以可以二分搜尋。
    每週抽 2,000 條邊用 lcp_np 驗證（規則 6 式的交叉驗算）。"""
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[mid - 1] == b[mid - 1]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def lead_chain(arr, mask):
    """arr 從頭開始、mask 為真的最長連續段（鏈式 → 單調，二分搜尋）。"""
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mask[arr[mid - 1]]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def lead_full(arr, mask):
    if len(arr) == 0:
        return 0
    m = mask[arr]
    k = int(np.argmin(m))
    return len(arr) if m[k] else k


def one_week(wdir):
    week = os.path.basename(wdir)
    rq = pq.read_table(os.path.join(wdir, "processed/requests.parquet"),
                       columns=["session_id", "step", "start_ms", "end_ms", "completion_tokens", "role_tokens",
                                "transition", "cause", "step_trigger", "block_ids"])
    ss = pq.read_table(os.path.join(wdir, "processed/sessions.parquet"),
                       columns=["session_id", "user_id", "parent_session_id"]).to_pydict()
    user_of = dict(zip(ss["session_id"], ss["user_id"]))
    is_sub = {s: (p is not None) for s, p in zip(ss["session_id"], ss["parent_session_id"])}
    sid = rq.column("session_id").to_pylist()
    step = rq.column("step").to_numpy()
    st = rq.column("start_ms").to_numpy()
    en = rq.column("end_ms").to_numpy()
    comp = pc.fill_null(rq.column("completion_tokens"), 0).to_numpy()
    rt = rq.column("role_tokens").combine_chunks()
    role_sum = np.zeros(len(sid), dtype=np.int64)
    for f in ("system", "user", "assistant", "tool", "tool_definitions"):
        role_sum += pc.fill_null(pc.struct_field(rt, f), 0).to_numpy()
    trans = rq.column("transition").to_pylist()
    cause = rq.column("cause").to_pylist()
    trig = rq.column("step_trigger").to_pylist()
    bl = rq.column("block_ids").combine_chunks()
    off = bl.offsets.to_numpy()
    val = bl.values.to_numpy()
    n_blk = np.diff(off)
    N = len(sid)

    def blocks(i):
        return val[off[i]:off[i + 1]]

    # 依 session、step 排序
    order = sorted(range(N), key=lambda i: (sid[i], step[i]))
    by_s = defaultdict(list)
    for i in order:
        by_s[sid[i]].append(i)
    # 規則 6 (c)
    n_pairs = ok_step = ok_ts = 0
    for s, ix in by_s.items():
        for a, b in zip(ix[:-1], ix[1:]):
            n_pairs += 1
            ok_step += step[b] > step[a]
            ok_ts += st[b] >= st[a]
    # 規則 6 (a)
    m160 = role_sum >= 160
    ratio = 16.0 * n_blk[m160] / role_sum[m160]
    unit_ok = int(np.sum((ratio >= 0.9) & (ratio <= 1.1)))
    unit_n = int(m160.sum())
    # ---- 逐邊 ----
    E = []  # (p, c, L)
    for s, ix in by_s.items():
        for a, b in zip(ix[:-1], ix[1:]):
            E.append((a, b, lcp_chain(blocks(a), blocks(b))))
    rng = np.random.default_rng(0)
    samp = rng.choice(len(E), size=min(2000, len(E)), replace=False) if E else []
    chain_ok = sum(lcp_np(blocks(E[k][0]), blocks(E[k][1])) == E[k][2] for k in samp)
    chain_n = len(samp)
    # 規則 6 (b)
    app_n = app_ok = 0
    for a, b, L in E:
        if trans[b] == "append":
            app_n += 1
            app_ok += L >= n_blk[a] - 1
    # ---- T1 ----
    S = Dm = Dp = 0
    cls_cnt = Counter()
    conf = Counter()
    Dm_lab = Counter()
    S_lab = Counter()
    sess_with_edge = set()
    for a, b, L in E:
        npb = int(n_blk[a])
        ip = 16 * npb
        d = 16 * (int(n_blk[b]) - npb)
        op = int(comp[a])
        full = L >= npb - 1
        if full and d >= op:
            cls = "APPEND"
        elif full:
            cls = "STRIP"
        else:
            cls = "REWRITE"
        s_ = ip + op
        if cls == "APPEND":
            rp = rm = s_
        elif cls == "STRIP":
            rp = ip + min(op, max(d, 0))
            rm = min(16 * L, ip)
        else:
            rp = rm = min(16 * L, ip)
        S += s_
        Dm += s_ - rp
        Dp += s_ - rm
        cls_cnt[cls] += 1
        lab = f"{trans[b]}|{cause[b]}"
        conf[(cls, trans[b])] += 1
        Dm_lab[lab] += s_ - rp
        S_lab[lab] += s_
        sess_with_edge.add(sid[a])
    # 只報告（事後加，見 X9 §6）：每一步「新寫進去」的 KV（token）＝第一步的全部輸入＋輸出；之後＝超出和前一步共同前綴的輸入＋輸出
    written = 0
    for s_, ix in by_s.items():
        written += 16 * int(n_blk[ix[0]]) + int(comp[ix[0]])
    for a, b, L in E:
        written += 16 * (int(n_blk[b]) - L) + int(comp[b])
    # ---- T2 ----
    # session 的不重複 block：第一個請求的全部＋之後每個請求超出「和前一個共同前綴」的部分（歸納下等於全部的聯集）
    Lof = {(a, b): L for a, b, L in E}
    sess_uniq = {}
    for s, ix in by_s.items():
        parts = [blocks(ix[0])] + [blocks(b)[Lof[(a, b)]:] for a, b in zip(ix[:-1], ix[1:])]
        sess_uniq[s] = np.unique(np.concatenate(parts))
    allu = np.concatenate(list(sess_uniq.values()))
    u, cnt = np.unique(allu, return_counts=True)
    shared_s = u[cnt >= 2]
    n_unique = len(u)
    sum_sess = int(sum(len(x) for x in sess_uniq.values()))
    by_user = defaultdict(list)
    for s, x in sess_uniq.items():
        by_user[user_of.get(s, s)].append(x)
    uu = np.concatenate([np.unique(np.concatenate(v)) for v in by_user.values()])
    u2, cnt2 = np.unique(uu, return_counts=True)
    shared_u = u2[cnt2 >= 2]
    del allu, uu, u, cnt, u2, cnt2

    maxid = int(val.max()) if len(val) else 0
    mask_s = np.zeros(maxid + 1, dtype=bool)
    mask_s[shared_s] = True
    mask_u = np.zeros(maxid + 1, dtype=bool)
    mask_u[shared_u] = True
    lead_ok = lead_n = 0
    for k in samp:
        a_, b_, L_ = E[k]
        if L_ > 0:
            lead_n += 1
            lead_ok += lead_chain(blocks(b_)[:L_], mask_s) == lead_full(blocks(b_)[:L_], mask_s)

    sumL = sumS = sumU = 0
    sumL_ns = sumS_ns = 0
    shares = []
    for a, b, L in E:
        if L <= 0:
            continue
        pre = blocks(b)[:L]
        P = lead_chain(pre, mask_s)
        Pu = lead_chain(pre, mask_u)
        sumL += L
        sumS += min(P, L)
        sumU += min(Pu, L)
        shares.append(min(P, L) / L)
        if not is_sub.get(sid[b], False):
            sumL_ns += L
            sumS_ns += min(P, L)
    # ---- T3 ----
    gaps, gaps_user = [], []
    finals = 0
    for s, ix in by_s.items():
        for a, b in zip(ix[:-1], ix[1:]):
            g = (st[b] - en[a]) / 1000.0
            gaps.append(g)
            if trig[b] == "user":
                gaps_user.append(g)
        finals += 1
    return dict(week=week, n_req=N, n_sessions=len(by_s), n_pairs=n_pairs, ok_step=ok_step, ok_ts=ok_ts,
                unit_ok=unit_ok, unit_n=unit_n, app_n=app_n, app_ok=app_ok,
                chain_ok=int(chain_ok), chain_n=int(chain_n), lead_ok=int(lead_ok), lead_n=int(lead_n),
                n_edges=len(E), sess_with_edge=len(sess_with_edge), S=S, Dm=Dm, Dp=Dp, cls_cnt=dict(cls_cnt), written=written,
                conf={f"{k[0]}|{k[1]}": v for k, v in conf.items()}, Dm_lab=dict(Dm_lab), S_lab=dict(S_lab),
                sumL=sumL, sumS=sumS, sumU=sumU, sumL_ns=sumL_ns, sumS_ns=sumS_ns, shares=np.array(shares),
                n_unique=n_unique, sum_sess=sum_sess, n_sub=sum(1 for s in by_s if is_sub.get(s, False)),
                gaps=np.array(gaps), gaps_user=np.array(gaps_user), finals=finals,
                n_users=len(by_user))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    assert RUN_ID
    weeks = sorted(glob.glob(os.path.join(ROOT, "*_*")))
    log("weeks", len(weeks))
    assert len(weeks) == 16, f"應有 16 週，找到 {len(weeks)}"
    with ProcessPoolExecutor(a.procs) as ex:
        R = list(ex.map(one_week, weeks))
    for r in R:
        log(r["week"], "req", r["n_req"], "sess", r["n_sessions"], "edges", r["n_edges"],
            "unit", r["unit_ok"], "/", r["unit_n"], "append_ok", r["app_ok"], "/", r["app_n"])
    tot = lambda k: sum(r[k] for r in R)  # noqa: E731
    unit_rate = tot("unit_ok") / tot("unit_n")
    app_rate = tot("app_ok") / tot("app_n")
    step_rate = tot("ok_step") / tot("n_pairs")
    ts_rate = tot("ok_ts") / tot("n_pairs")
    chain_rate = tot("chain_ok") / max(1, tot("chain_n"))
    lead_rate = tot("lead_ok") / max(1, tot("lead_n"))
    chk = dict(unit_rate=unit_rate, append_label_lcp_rate=app_rate, step_inc_rate=step_rate, start_nondec_rate=ts_rate,
               chain_lcp_sample_rate=chain_rate, chain_lead_sample_rate=lead_rate)
    log("rule6", json.dumps(chk))
    # 二分搜尋的前提（鏈式 block）不成立就停
    assert chain_rate >= 0.99 and lead_rate >= 0.99, f"鏈式前提：LCP 抽樣一致 {chain_rate:.4f}、共用前綴抽樣一致 {lead_rate:.4f}"
    assert unit_rate >= 0.95, f"規則 6 (a)：16·len(block_ids)/role_tokens ∈[0.9,1.1] 只有 {unit_rate:.4f}"
    assert app_rate >= 0.95, f"規則 6 (b)：append 標籤的前綴延續只有 {app_rate:.4f}"
    assert step_rate >= 0.99 and ts_rate >= 0.99, f"規則 6 (c)：step {step_rate:.4f} start {ts_rate:.4f}"
    now = ts()
    out = a.out
    os.makedirs(out, exist_ok=True)
    # T1
    cls = Counter()
    conf = Counter()
    Dm_lab = Counter()
    S_lab = Counter()
    for r in R:
        cls.update(r["cls_cnt"])
        conf.update(r["conf"])
        Dm_lab.update(r["Dm_lab"])
        S_lab.update(r["S_lab"])
    nE = tot("n_edges")
    t1 = dict(run_id=RUN_ID, ts=now, dataset="freeinference", n_req=tot("n_req"), n_sessions=tot("n_sessions"),
              n_sessions_with_edge=tot("sess_with_edge"), n_edges=nE, **{f"rule6_{k}": v for k, v in chk.items()},
              frac_APPEND=cls["APPEND"] / nE, frac_STRIP=cls["STRIP"] / nE, frac_REWRITE=cls["REWRITE"] / nE,
              sum_S_tok=tot("S"), sum_Dminus_tok=tot("Dm"), sum_Dplus_tok=tot("Dp"),
              dead_lb_frac=tot("Dm") / tot("S"), dead_ub_frac=tot("Dp") / tot("S"),
              sum_written_tok=tot("written"), dead_lb_over_written=tot("Dm") / tot("written"),
              dead_ub_over_written=tot("Dp") / tot("written"))
    lab_rows = []
    for lab in sorted(S_lab, key=lambda k: -S_lab[k]):
        lab_rows.append(dict(run_id=RUN_ID, ts=now, dataset="freeinference", label=lab, sum_S_tok=S_lab[lab],
                             sum_Dminus_tok=Dm_lab[lab], share_of_total_Dminus=Dm_lab[lab] / max(1, tot("Dm")),
                             dead_lb_frac_within=Dm_lab[lab] / S_lab[lab] if S_lab[lab] else float("nan")))
    for k, v in sorted(conf.items()):
        lab_rows.append(dict(run_id=RUN_ID, ts=now, dataset="freeinference", label=f"confusion:{k}", n_edges=v))
    # T2
    shares = np.concatenate([r["shares"] for r in R])
    t2 = dict(run_id=RUN_ID, ts=now, dataset="freeinference", n_returns=int(len(shares)),
              bytew_shared_frac=tot("sumS") / tot("sumL"), bytew_shared_frac_crossuser=tot("sumU") / tot("sumL"),
              bytew_shared_frac_nonsubagent=tot("sumS_ns") / max(1, tot("sumL_ns")),
              median_share=float(np.median(shares)), p90_share=float(np.quantile(shares, 0.9)),
              p10_share=float(np.quantile(shares, 0.1)), frac_share_ge_0p5=float(np.mean(shares >= 0.5)),
              session_over_unique_sum16weeks=tot("sum_sess") / tot("n_unique"),
              n_subagent_sessions=tot("n_sub"), n_users_weeksum=tot("n_users"))
    # T3
    g = np.concatenate([r["gaps"] for r in R])
    gu = np.concatenate([r["gaps_user"] for r in R])
    finals = tot("finals")
    neg = int(np.sum(g < 0))
    neg_u = int(np.sum(gu < 0))
    g = np.clip(g, 0, None)
    gu = np.clip(gu, 0, None)
    t3 = []
    for wn, W in WINDOWS_S.items():
        for name, gg in (("all_requests", g), ("user_returns_only", gu)):
            num = float(np.minimum(1.0, gg / W).sum())
            t3.append(dict(run_id=RUN_ID, ts=now, dataset="freeinference", variant=name, window=wn, W_s=W,
                           n_finite_gaps=int(len(gg)), n_final=finals, M_W=num / (num + finals),
                           gap_median_s=float(np.median(gg)), gap_p90_s=float(np.quantile(gg, 0.9)),
                           frac_gt_1h=float(np.mean(gg > 3600)), frac_gt_24h=float(np.mean(gg > 86400)),
                           n_negative_gaps_clipped=neg if name == "all_requests" else neg_u))
    for fn, rs in (("fi_t1_census.csv", [t1]), ("fi_t1_labels.csv", lab_rows), ("fi_t2_shared_prefix.csv", [t2]),
                   ("fi_t3_window_censoring.csv", t3)):
        keys = []
        for r in rs:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(os.path.join(out, fn), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rs)
        log("寫入", fn, json.dumps(rs[:3], ensure_ascii=False, default=float))
    np.save(os.path.join(a.rundir, "fi_gaps_s.npy"), g)
    np.save(os.path.join(a.rundir, "fi_shares.npy"), shares)


if __name__ == "__main__":
    main()
