"""x9_census.py — X9 的 T1（回合結束就死的 KV）與 T2（跨 session 共用前綴）。

判準在 docs/phase3_20261017/explore/X9_wild_ideas.md §2.2（2026-10-17T09:45:06Z 寫定）。
只讀 trace、不訓練、不估任何查表；全部是描述統計。

用法：
  python -I code/p3x/x9_census.py --out results/p3_explore/x9 --rundir $RUN_DIR
環境變數 RUN_ID 必須有（run3090.sh 會給）。
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

DS = os.environ.get("TIARA_DATASETS", "/ssd8/hungwei/paper-hkv/datasets")
BAILIAN = {
    "bailian_traceA": f"{DS}/qwen-bailian-usagetraces-anon/qwen_traceA_blksz_16.jsonl",
    "bailian_traceB": f"{DS}/qwen-bailian-usagetraces-anon/qwen_traceB_blksz_16.jsonl",
    "bailian_thinking": f"{DS}/qwen-bailian-usagetraces-anon/qwen_thinking_blksz_16.jsonl",
    "bailian_coder": f"{DS}/qwen-bailian-usagetraces-anon/qwen_coder_blksz_16.jsonl",
}
MOONCAKE = {
    "mooncake_conversation": f"{DS}/mooncake/traces/conversation_trace.jsonl",
    "mooncake_toolagent": f"{DS}/mooncake/traces/toolagent_trace.jsonl",
}

# 成本常數〔MI300X 常數〕：results/m7_write_policy_mi300x/calib_c1.csv（run 20261008-130316-m7-c1）
CALIB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "results/m7_write_policy_mi300x/calib_c1.csv")
CHUNK_TOK = 512
CHUNK_BYTES = 64 * (1 << 20)  # Llama-3.1-8B BF16，一個 512-token chunk
GiB = 1 << 30
TIERS_GIBPS = {"bstar32": 1.47, "nfs": 0.331083984375}  # b*=32 層（main.tex Table speed）、NFS（tier_params.json）

RUN_ID = os.environ.get("RUN_ID", "")


def ts():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(*a):
    print(f"[{ts()}]", *a, flush=True)


# ---------------------------------------------------------------------------
# 載入（規則 6 的斷言寫在這裡）
# ---------------------------------------------------------------------------
def load_bailian(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            rows.append(json.loads(line))
    n = len(rows)
    blk_ok = sum(len(r["hash_ids"]) == math.ceil(r["input_length"] / 16) for r in rows)
    rate = blk_ok / n
    # 規則 6 (a)：用 input_length 驗算 hash_ids 的單位（16 token／block）
    assert rate >= 0.99, f"{path}: len(hash_ids)==ceil(input_length/16) 只有 {rate:.4f}"
    by_id = {}
    for r in rows:
        assert r["chat_id"] not in by_id, f"重複的 chat_id {r['chat_id']}"
        by_id[r["chat_id"]] = r
    edges = [(by_id[r["parent_chat_id"]], r) for r in rows if r["parent_chat_id"] != -1 and r["parent_chat_id"] in by_id]
    missing_parent = sum(1 for r in rows if r["parent_chat_id"] != -1 and r["parent_chat_id"] not in by_id)
    # 沒有父子邊的 trace（traceB 全是單輪）：比例記 NaN，不做 (b) 斷言（20261017-174657 那次把 0/0 當 0 而誤觸斷言）
    ts_ok = sum(c["timestamp"] >= p["timestamp"] for p, c in edges) / len(edges) if edges else float("nan")
    # 規則 6 (b)：用 timestamp 驗算父子方向
    assert not edges or ts_ok >= 0.99, f"{path}: 子的 timestamp ≥ 父 只有 {ts_ok:.4f}"
    turn_ok = sum(c["turn"] == p["turn"] + 1 for p, c in edges) / len(edges) if edges else float("nan")
    chk = dict(n_req=n, blk_unit_rate=rate, ts_order_rate=ts_ok, turn_plus1_rate=turn_ok,
               n_edges=len(edges), n_missing_parent=missing_parent)
    return rows, by_id, edges, chk


def roots_of(rows, by_id):
    root = {}
    for r in rows:
        cid = r["chat_id"]
        path = []
        cur = cid
        while True:
            if cur in root:
                rt = root[cur]
                break
            path.append(cur)
            p = by_id[cur]["parent_chat_id"]
            if p == -1 or p not in by_id:
                rt = cur
                break
            cur = p
        for x in path:
            root[x] = rt
    return root


def lcp(a, b):
    m = min(len(a), len(b))
    i = 0
    while i < m and a[i] == b[i]:
        i += 1
    return i


# ---------------------------------------------------------------------------
# T1
# ---------------------------------------------------------------------------
def t1_edges(edges):
    out = []
    for p, c in edges:
        hp, hc = p["hash_ids"], c["hash_ids"]
        L = lcp(hp, hc)
        n_p = len(hp)
        full = L >= n_p - 1
        d = c["input_length"] - p["input_length"]
        op = p["output_length"]
        ip = p["input_length"]
        if full and d >= op:
            cls = "APPEND"
        elif full:
            cls = "STRIP"
        else:
            cls = "REWRITE"
        S = ip + op
        if cls == "APPEND":
            rp = rm = S
        elif cls == "STRIP":
            rp = ip + min(op, max(d, 0))
            rm = min(16 * L, ip)
        else:
            rp = rm = min(16 * L, ip)
        out.append(dict(parent=p["chat_id"], child=c["chat_id"], ptype=p["type"], cls=cls, L=L, n_p=n_p,
                        delta=d, out_p=op, in_p=ip, S=S, Rplus=rp, Rminus=rm,
                        same_input=int(hp == hc)))
    return out


def t1_summary(name, rows, edges, E, root, chk):
    by_parent = defaultdict(list)
    for e in E:
        by_parent[e["parent"]].append(e)
    S = Dm = Dp = 0
    S_t = Counter()
    Dm_t = Counter()
    Dp_t = Counter()
    Dm_cls = Counter()
    for pid, es in by_parent.items():
        s = es[0]["S"]
        bestp = max(es, key=lambda e: e["Rplus"])
        dm = s - bestp["Rplus"]
        dp = s - max(e["Rminus"] for e in es)
        t = es[0]["ptype"]
        S += s
        Dm += dm
        Dp += dp
        S_t[t] += s
        Dm_t[t] += dm
        Dp_t[t] += dp
        Dm_cls[bestp["cls"]] += dm
    cls_cnt = Counter(e["cls"] for e in E)
    nE = len(E)
    nEd = nE if nE else float("nan")
    # 分岔：有兄弟的子請求
    kids = Counter(e["parent"] for e in E)
    sib = [e for e in E if kids[e["parent"]] >= 2]
    regen = sum(e["same_input"] for e in E if kids[e["parent"]] >= 2)
    # 兄弟之間輸入相同（重新生成）：用 hash 比對兄弟，而不是和父比
    by_p_hash = defaultdict(Counter)
    for p, c in edges:
        by_p_hash[p["chat_id"]][tuple(c["hash_ids"])] += 1
    regen_sib = 0
    for pid, cnt in by_p_hash.items():
        if sum(cnt.values()) >= 2:
            regen_sib += sum(v for v in cnt.values() if v >= 2)
    roots = set(root.values())
    root_sizes = Counter(root.values())
    multi_roots = sum(1 for v in root_sizes.values() if v >= 2)
    parents_with_child = len(by_parent)
    parent_roots = len({root[pid] for pid in by_parent})
    summ = dict(trace=name, **chk, n_roots=len(roots), n_roots_multi=multi_roots,
                n_parents_with_child=parents_with_child, n_roots_of_parents=parent_roots,
                n_req_without_child=len(rows) - parents_with_child,
                frac_APPEND=cls_cnt["APPEND"] / nEd, frac_STRIP=cls_cnt["STRIP"] / nEd,
                frac_REWRITE=cls_cnt["REWRITE"] / nEd,
                sum_S_tok=S, sum_Dminus_tok=Dm, sum_Dplus_tok=Dp,
                dead_lb_frac=Dm / S if S else float("nan"), dead_ub_frac=Dp / S if S else float("nan"),
                dead_lb_share_STRIP=Dm_cls["STRIP"] / Dm if Dm else float("nan"),
                dead_lb_share_REWRITE=Dm_cls["REWRITE"] / Dm if Dm else float("nan"),
                fork_rate=len(sib) / nE if nE else float("nan"),
                regen_rate_sibling_same_input=regen_sib / nE if nE else float("nan"),
                child_same_input_as_parent=sum(e["same_input"] for e in E) / nE if nE else float("nan"),
                n_children_with_sibling=len(sib))
    by_type = []
    for t in sorted(S_t):
        by_type.append(dict(trace=name, ptype=t, sum_S_tok=S_t[t], dead_lb_frac=Dm_t[t] / S_t[t],
                            dead_ub_frac=Dp_t[t] / S_t[t],
                            n_parents=sum(1 for es in by_parent.values() if es[0]["ptype"] == t)))
    return summ, by_type


# ---------------------------------------------------------------------------
# T2
# ---------------------------------------------------------------------------
def multi_session_blocks(rows, rootkey):
    first = {}
    multi = set()
    for r in rows:
        rt = rootkey(r)
        for h in r["hash_ids"]:
            f = first.get(h)
            if f is None:
                first[h] = rt
            elif f != rt:
                multi.add(h)
    return multi


def leading_in(h, s):
    i = 0
    for x in h:
        if x in s:
            i += 1
        else:
            break
    return i


def load_f():
    import statistics
    rows = list(csv.DictReader(open(CALIB)))
    g = defaultdict(list)
    for r in rows:
        if r["item"] == "f_chunk" and r["gpu_state"] == "idle":
            g[int(r["chunk_idx"])].append(float(r["ms"]))
    f = [statistics.median(g[i]) for i in sorted(g)]
    # 線性外插（最小平方，block ≥ len(f)）〔外插〕
    n = len(f)
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(f) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, f)) / sum((x - mx) ** 2 for x in xs)
    icpt = my - slope * mx
    return f, slope, icpt


def f_at(i, F):
    f, slope, icpt = F
    return f[i] if i < len(f) else icpt + slope * i


def cake(start, n, ell, F):
    """Cake 兩條線：重算從 start 往後、載入從 n-1 往前（m7_sim.sim_restore 同一規則：rt<=lt 就重算）。回傳 ms。"""
    if n <= start:
        return 0.0
    p, q, rt, lt = start, n - 1, 0.0, 0.0
    while p <= q:
        if rt <= lt:
            rt += f_at(p, F)
            p += 1
        else:
            lt += ell
            q -= 1
    return max(rt, lt)


def t2_for(name, rows, ret, root_of_row, blk_tok, F):
    """ret: list of (child_row, L_blocks, prefix_tok_cap)."""
    multi = multi_session_blocks(rows, root_of_row)
    recs = []
    for c, L, cap in ret:
        if L <= 0:
            continue
        P = leading_in(c["hash_ids"], multi)
        sh = min(P, L)
        pref_tok = min(blk_tok * L, cap)
        sh_tok = min(blk_tok * sh, pref_tok)
        n = math.ceil(pref_tok / CHUNK_TOK)
        s = sh_tok // CHUNK_TOK
        rec = dict(child=c.get("chat_id", c.get("_idx")), L=L, P=P, shared=sh, share=sh / L,
                   pref_tok=pref_tok, shared_tok=sh_tok, n_chunks=n, shared_chunks=s)
        for tier, gb in TIERS_GIBPS.items():
            ell = CHUNK_BYTES / (gb * GiB) * 1e3
            full = cake(0, n, ell, F)
            part = cake(s, n, ell, F)
            rec[f"cake_full_ms_{tier}"] = full
            rec[f"cake_sharedfree_ms_{tier}"] = part
            rec[f"saving_frac_{tier}"] = (full - part) / full if full > 0 else 0.0
        recs.append(rec)
    return recs


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def t2_summary(name, recs, n_sessions):
    if not recs:
        return dict(trace=name, n_returns=0)
    sL = sum(r["L"] for r in recs)
    ssh = sum(r["shared"] for r in recs)
    shares = [r["share"] for r in recs]
    out = dict(trace=name, n_returns=len(recs), n_sessions=n_sessions,
               bytew_shared_frac=ssh / sL, median_share=q(shares, 0.5), p90_share=q(shares, 0.9),
               p10_share=q(shares, 0.1), frac_returns_share_ge_0p5=sum(s >= 0.5 for s in shares) / len(shares),
               median_pref_tok=q([r["pref_tok"] for r in recs], 0.5),
               median_shared_tok=q([r["shared_tok"] for r in recs], 0.5),
               frac_returns_with_shared_chunk=sum(r["shared_chunks"] >= 1 for r in recs) / len(recs))
    for tier in TIERS_GIBPS:
        out[f"median_saving_frac_{tier}"] = q([r[f"saving_frac_{tier}"] for r in recs], 0.5)
        out[f"mean_saving_frac_{tier}"] = sum(r[f"saving_frac_{tier}"] for r in recs) / len(recs)
        tf = sum(r[f"cake_full_ms_{tier}"] for r in recs)
        tp = sum(r[f"cake_sharedfree_ms_{tier}"] for r in recs)
        out[f"sum_saving_frac_{tier}"] = (tf - tp) / tf if tf else 0.0
    return out


# ---------------------------------------------------------------------------
# Mooncake（只報告）：前綴鏈重建
# ---------------------------------------------------------------------------
def load_mooncake(path):
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    rate = sum(len(r["hash_ids"]) == math.ceil(r["input_length"] / CHUNK_TOK) for r in rows) / n
    assert rate >= 0.95, f"{path}: len(hash_ids)==ceil(input_length/512) 只有 {rate:.4f}"
    tsorted = all(rows[i]["timestamp"] <= rows[i + 1]["timestamp"] for i in range(n - 1))
    for i, r in enumerate(rows):
        r["_idx"] = i
    # 父＝最近一個「hash_ids[:-1] 是 r 的前綴」的較早請求（共同前綴 ≥2 block；取最長的）
    key2idx = {}
    parent = [-1] * n
    Lp = [0] * n
    for i, r in enumerate(rows):
        h = r["hash_ids"]
        for k in range(len(h), 1, -1):  # k≥2：不讓共用的 block 0 當成父子關係
            j = key2idx.get(tuple(h[:k]))
            if j is not None:
                parent[i] = j
                Lp[i] = lcp(rows[j]["hash_ids"], h)
                break
        if len(h) >= 2:
            key2idx[tuple(h[:-1])] = i
    chk = dict(n_req=n, blk_unit_rate=rate, ts_sorted=tsorted,
               n_edges=sum(1 for p in parent if p >= 0))
    return rows, parent, Lp, chk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rundir", required=True)
    a = ap.parse_args()
    assert RUN_ID, "RUN_ID 未設（用 code/run3090.sh 跑）"
    os.makedirs(a.out, exist_ok=True)
    F = load_f()
    log("f(i) 讀入", len(F[0]), "個 block；外插 slope=%.4f icpt=%.3f" % (F[1], F[2]))
    t1rows, t1type, t2rows = [], [], []
    for name, path in BAILIAN.items():
        log("載入", name)
        rows, by_id, edges, chk = load_bailian(path)
        log(name, chk)
        root = roots_of(rows, by_id)
        E = t1_edges(edges)
        summ, bt = t1_summary(name, rows, edges, E, root, chk)
        log("T1", json.dumps(summ, ensure_ascii=False))
        t1rows.append(summ)
        t1type += bt
        if E:
            with gzip.open(os.path.join(a.rundir, f"t1_edges_{name}.csv.gz"), "wt") as fh:
                w = csv.DictWriter(fh, fieldnames=list(E[0].keys()))
                w.writeheader()
                w.writerows(E)
        ret = [(c, lcp(p["hash_ids"], c["hash_ids"]), p["input_length"]) for p, c in edges]
        recs = t2_for(name, rows, ret, lambda r: root[r["chat_id"]], 16, F)
        s2 = t2_summary(name, recs, len(set(root.values())))
        s2["source"] = "bailian_parent_chat_id"
        log("T2", json.dumps(s2, ensure_ascii=False))
        t2rows.append(s2)
        if recs:
            with gzip.open(os.path.join(a.rundir, f"t2_returns_{name}.csv.gz"), "wt") as fh:
                w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
                w.writeheader()
                w.writerows(recs)
    # Mooncake：T2 只報告；T1 的 APPEND／STRIP 也只報告（重建的父子，不是明確欄位）
    for name, path in MOONCAKE.items():
        log("載入", name)
        rows, parent, Lp, chk = load_mooncake(path)
        log(name, chk)
        # 根
        rootm = [0] * len(rows)
        for i in range(len(rows)):
            rootm[i] = i if parent[i] < 0 else rootm[parent[i]]
        edges = [(rows[parent[i]], rows[i]) for i in range(len(rows)) if parent[i] >= 0]
        # T1 報告版：block＝512，所以「整份延續」＝L ≥ n_p − 1 同定義
        E = []
        for p, c in edges:
            L = lcp(p["hash_ids"], c["hash_ids"])
            full = L >= len(p["hash_ids"]) - 1
            d = c["input_length"] - p["input_length"]
            cls = "APPEND" if (full and d >= p["output_length"]) else ("STRIP" if full else "REWRITE")
            E.append(cls)
        cc = Counter(E)
        t1rows.append(dict(trace=name, **chk, n_roots=len(set(rootm)),
                           frac_APPEND=cc["APPEND"] / max(1, len(E)), frac_STRIP=cc["STRIP"] / max(1, len(E)),
                           frac_REWRITE=cc["REWRITE"] / max(1, len(E)), note="report_only_reconstructed_parent"))
        ret = [(rows[i], Lp[i], rows[parent[i]]["input_length"]) for i in range(len(rows)) if parent[i] >= 0]
        recs = t2_for(name, rows, ret, lambda r: rootm[r["_idx"]], CHUNK_TOK, F)
        s2 = t2_summary(name, recs, len(set(rootm)))
        s2["source"] = "mooncake_reconstructed_report_only"
        log("T2", json.dumps(s2, ensure_ascii=False))
        t2rows.append(s2)
    now = ts()

    def dump(fn, rs):
        keys = []
        for r in rs:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(os.path.join(a.out, fn), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["run_id", "ts"] + keys)
            w.writeheader()
            for r in rs:
                w.writerow(dict(run_id=RUN_ID, ts=now, **r))
        log("寫入", os.path.join(a.out, fn))

    dump("t1_census.csv", t1rows)
    dump("t1_by_type.csv", t1type)
    dump("t2_shared_prefix.csv", t2rows)


if __name__ == "__main__":
    main()
