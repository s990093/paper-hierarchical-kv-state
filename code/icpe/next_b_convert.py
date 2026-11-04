"""next_b_convert.py — ICPE next（icpe-data）：新紀錄 → X1 模擬器的 nb 格式（code/p3x/x1_nb.py，icpe-sim 定的介面）。

  python -I code/icpe/next_b_convert.py cachewise   --N 16 32 64 72 --seeds 1 2 3 --copies 5 [--prefix-rule struct|cache]
  python -I code/icpe/next_b_convert.py tracelab    --compress 1 2 4 [--prefix-rule struct|cache]
  python -I code/icpe/next_b_convert.py semianalysis --N 16 32 64 72 --seeds 1 2 3
  python -I code/icpe/next_b_convert.py bailian     --files traceA traceB thinking coder
  共同：--out-dir（預設 $X1_NB_DIR 或 /ssd6/hungwei/icpe-next/datasets/x1_nb）、--trunc-128k（敏感度 1：每個 session 遇到第一個
  輸入 >131072 token 的請求就截斷，那一輪和之後都不算；名稱加 _t128k）、--p3x（x1_nb.py 所在目錄，預設 repo 的 code/p3x）

輸出：<out-dir>/<name>.npz（x1_nb.write_nb 寫，寫前檢查鏈式前綴、parent、session、時間）＋ <name>.meta.json（同一份 meta）。
**這些檔不進 git**（CacheWise 沒有授權；其他也是大檔）。git 裡只放彙總（next_b_trace_summary.csv）。

轉換規則（NEXT_PLAN §5.6、§5.7；PREREG §2 第 7 項、§3 C／D、§4）
  * chunk＝512 token；nall＝ceil(輸入 token/512)；完整 chunk 數＝floor(輸入 token/512)（同 x1_fi、x1_trace：最後不滿的不存）。
  * 回來的請求（kind 1）＝同一個 session 的第 2 個以後的請求；parent＝同一個 session 的上一個請求。
  * chunk 鍵：
      - TraceLab、CacheWise（沒有 hash，只有 token 數）：請求 k 的前 shared 塊沿用請求 k−1 的鍵，其餘給新鍵；
        shared＝min(floor(P_k/512), 完整塊數(k−1), 完整塊數(k))，P_k＝可從上一輪還原的前綴 token（--prefix-rule）：
          cache ：P_k＝min(供應商快取命中, inp(k−1))（TraceLab prefix_tokens；CacheWise cache_read_input_tokens）
          struct：inp(k) ≥ inp(k−1) 而且（CacheWise）中間沒有 compact_boundary → P_k＝inp(k−1)（上一輪的整段輸入是這一輪的前綴）；
                  否則同 cache。理由：供應商快取 5 分鐘／1 小時過期後命中變 0，但上下文其實是接續的（STATUS_DATA 2026-11-04 的驗算）
        不同 session 之間不共用（沒有內容資訊）。
      - SemiAnalysis（64-token hash，只在同一條紀錄內可比）：chunk j 的鍵＝(紀錄 id, hash_ids[0:8(j+1)]) 這整段前綴的編號
        （鏈式；同一條紀錄的主 agent 與子 agent 前綴相同就共用）。
      - Bailian（16-token 鏈式 hash）：chunk j 的鍵＝hash_ids[32j+31]（鏈式 id 的最後一個決定整段前綴；同 x1_fi）。
  * 時間：秒。同一個 session 內強制非遞減（t_k＝max(t_k, t_{k−1})，修正數記在 meta）；對話內的間隔一律用紀錄的真值。
  * 到達合成（只合成「對話什麼時候開始」）：
      - TraceLab：原始絕對時間；壓縮 c×：對話（session）開始時間 s → T0＋(s−T0)/c，對話內的偏移不動。c＝1 就是原始。
      - CacheWise、SemiAnalysis：固定並行 N：同時保持 N 個對話；一個結束（最後一個請求的 end）就從佇列補下一個，補上的時間＝那個結束時間。
        佇列＝全部對話（CacheWise：--copies 份複本，每份換新 session id、複本之間不共用鍵）用 numpy default_rng(seed).permutation 打亂。
        一個「對話」＝CacheWise 的一個檔（主＋子 agent）／SemiAnalysis 的一條紀錄（主＋子 agent），一起排、保留彼此的相對時間。
      - Bailian：原始時間，每份各自一個檔。
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import agent_traces as AT  # noqa: E402

CHUNK = 512
WIN = 131072


def _x1nb(p3x):
    sys.path.insert(0, p3x)
    import x1_nb
    return x1_nb


def _readme_sha(ds_dir, rel):
    """資料目錄 README.txt 記錄的 sha256（下載時對過上游）＋現在重算的 sha256；不同就停（規則 2）。"""
    p = os.path.join(ds_dir, rel)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    got = h.hexdigest()
    want = None
    rd = os.path.join(ds_dir, "README.txt")
    if os.path.exists(rd):
        for ln in open(rd):
            x = ln.split()
            if len(x) >= 3 and x[-1] == rel and len(x[0]) == 64:
                want = x[0]
    if want is not None and want != got:
        raise SystemExit(f"{p}: sha256 {got} ≠ README.txt 記錄的 {want} → 停")
    return dict(file=p, sha256=got, readme_sha256=want)


# ------------------------------------------------------------------ chunk 鍵
class KeyGen:
    def __init__(self):
        self.n = 0
        self.tab = {}

    def fresh(self, m):
        a = list(range(self.n, self.n + m))
        self.n += m
        return a

    def of(self, obj):
        k = self.tab.get(obj)
        if k is None:
            k = self.tab[obj] = self.n
            self.n += 1
        return k


def keys_by_prefix(d, kg, prefix_col):
    """d：同一個 session 的請求已依時間排好、有 inp、<prefix_col>（可還原的前綴 token）。回傳每列的完整 chunk 鍵。"""
    out = [None] * len(d)
    prev = {}
    sids, inps, prefs = d.sess.values, d.inp.values, d[prefix_col].values
    n_shared = 0
    for i in range(len(d)):
        s = sids[i]
        nf = int(inps[i]) // CHUNK
        pk = prev.get(s)
        if pk is None:
            ks = kg.fresh(nf)
        else:
            sh = min(int(prefs[i]) // CHUNK, len(pk), nf)
            ks = pk[:sh] + kg.fresh(nf - sh)
            n_shared += sh
        out[i] = ks
        prev[s] = ks
    return out, n_shared


def prefix_eff(d, rule, compact=None):
    g = d.groupby("sess", sort=False)
    pin = g.inp.shift()
    cache = np.minimum(d.prefix.astype(float), pin)
    if rule == "cache":
        return cache.fillna(0).astype(np.int64)
    assert rule == "struct", rule
    grow = d.inp >= pin
    if compact is not None:
        grow &= ~compact
    return pd.Series(np.where(grow, pin, cache), index=d.index).fillna(0).astype(np.int64)


# ------------------------------------------------------------------ 到達合成
def fixed_concurrency(conv_dur, N, seed):
    """conv_dur：每個對話的長度（秒）。回傳每個對話的開始時間（秒），依 default_rng(seed).permutation 的順序補位。"""
    order = np.random.default_rng(seed).permutation(len(conv_dur))
    start = np.zeros(len(conv_dur))
    heap = [0.0] * min(N, len(order))
    heapq.heapify(heap)
    for c in order:
        t0 = heapq.heappop(heap)
        start[c] = t0
        heapq.heappush(heap, t0 + float(conv_dur[c]))
    return start, order


def active_profile(starts, durs, step=60.0):
    """每 step 秒同時在跑的對話數（只報告）。"""
    if len(starts) == 0:
        return {}
    T = float(np.max(starts + durs))
    grid = np.arange(0.0, T + step, step)
    a = np.searchsorted(np.sort(starts), grid, side="right") - np.searchsorted(np.sort(starts + durs), grid, side="right")
    return dict(active_median=float(np.median(a)), active_p10=float(np.percentile(a, 10)), active_max=int(a.max()),
                span_h=T / 3600)


# ------------------------------------------------------------------ 共用：組 nb 陣列並寫檔
def finalize(name, d, keys, out_dir, x1nb, meta, trunc):
    """d：有 sess（任意 hashable）、t（秒）、inp、user（bool）；keys：每列的完整 chunk 鍵。"""
    d = d.copy()
    d["keys"] = keys
    n0 = len(d)
    if trunc:
        over = d.inp > WIN
        cut = over.groupby(d.sess, sort=False).cummax()
        meta["trunc_128k"] = dict(n_req_before=n0, n_req_cut=int(cut.sum()), frac_req_cut=float(cut.mean()),
                                  n_sess_touched=int(d.sess[cut].nunique()))
        d = d[~cut]
    # 同一個 session 內時間非遞減
    t = d.t.values.copy()
    fix = 0
    last = {}
    for i, s in enumerate(d.sess.values):
        if s in last and t[i] < last[s]:
            t[i] = last[s]
            fix += 1
        last[s] = t[i]
    d["t"] = t
    d["_o"] = np.arange(len(d))
    d = d.sort_values(["t", "_o"], kind="stable").reset_index(drop=True)
    sid_map = {}
    sess = np.array([sid_map.setdefault(s, len(sid_map)) for s in d.sess.values], np.int64)
    parent = np.full(len(d), -1, np.int64)
    kind = np.zeros(len(d), np.int8)
    lastreq = {}
    for i, s in enumerate(sess.tolist()):
        if s in lastreq:
            parent[i] = lastreq[s]
            kind[i] = 1
        lastreq[s] = i
    nall = np.array([max(int(math.ceil(x / CHUNK)), len(k)) for x, k in zip(d.inp.values, d["keys"].values)], np.int64)
    tt = d.t.values - d.t.values[0] if len(d) else d.t.values
    meta.update(name=name, n_req=int(len(d)), n_sess=int(len(sid_map)), n_cont=int((kind == 1).sum()),
                time_fix_within_session=fix, full_chunks_total=int(sum(len(k) for k in d["keys"].values)),
                inp_tokens_median=float(d.inp.median()), inp_tokens_p90=float(d.inp.quantile(.9)),
                frac_inp_gt_128k=float((d.inp > WIN).mean()), span_s=float(tt[-1]) if len(tt) else 0.0,
                ws_GiB_bf16_upper=float(len({k for ks in d["keys"].values for k in ks}) * 64 / 1024))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.npz")
    st = x1nb.write_nb(path, tt, sess, parent, kind, nall, list(d["keys"].values), user_trigger=d.user.values.astype(bool), meta=meta)
    meta["write_check"] = st
    with open(os.path.join(out_dir, f"{name}.meta.json"), "w") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1, default=str)
    print(f"→ {path}  n_req={meta['n_req']} n_sess={meta['n_sess']} chunks={meta['full_chunks_total']} span_h={meta['span_s']/3600:.1f}",
          flush=True)
    return meta


def base_meta(trace, a):
    return dict(trace=trace, run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"),
                host=os.uname().nodename, prefix_rule=getattr(a, "prefix_rule", None), argv=sys.argv[1:])


# ------------------------------------------------------------------ 各紀錄
def conv_cachewise(a, x1nb):
    src = AT.DS + "/cachewise-coding-traces"
    d0, lm = AT.load_cachewise()
    d0 = d0.rename(columns={"sid": "sess0"})
    out = []
    for cp in range(a.copies):
        x = d0.copy()
        x["copy"] = cp
        x["sess"] = [f"c{cp}|{s}" for s in x.sess0]
        x["convk"] = [f"c{cp}|{c}" for c in x.conv]
        out.append(x)
    d = pd.concat(out, ignore_index=True)
    d = d.sort_values(["sess", "start_ms", "end_ms"], kind="stable").reset_index(drop=True)
    d["pe"] = prefix_eff(d, a.prefix_rule, compact=d.compact)
    kg = KeyGen()
    keys, n_shared = keys_by_prefix(d, kg, "pe")
    d["keys"] = keys
    c0 = d.groupby("convk").start_ms.min()
    c1 = d.assign(e=np.fmax(d.end_ms, d.start_ms)).groupby("convk").e.max()
    convs = list(c0.index)
    dur = ((c1[convs] - c0[convs]) / 1000).values
    res = []
    for N in a.N:
        for seed in a.seeds:
            st, order = fixed_concurrency(dur, N, seed)
            smap = dict(zip(convs, st))
            x = d.copy()
            x["t"] = [smap[c] + (s - c0[c]) / 1000 for c, s in zip(x.convk, x.start_ms)]
            x["user"] = x.trigger == "user"
            name = f"cachewise_N{N}_s{seed}" + ("" if a.prefix_rule == "struct" else f"_{a.prefix_rule}") + ("_t128k" if a.trunc_128k else "")
            m = base_meta("cachewise", a)
            m.update(loader=lm, source=dict(dir=src, commit="181c435a090d328d00bbbee4c8eeb27d32f3abd2", license="NONE (internal research only)"),
                     arrival=dict(kind="fixed_concurrency", N=N, seed=seed, copies=a.copies, n_conv=len(convs),
                                  order_head=[str(convs[i]) for i in order[:3]], **active_profile(st, dur)),
                     n_shared_chunks=n_shared)
            res.append(finalize(name, x[["sess", "t", "inp", "user"]], list(x["keys"].values), a.out_dir, x1nb, m, a.trunc_128k))
    return res


def conv_tracelab(a, x1nb):
    src = AT.DS + "/tracelab"
    sha = _readme_sha(src, "syfi_coding_trace.jsonl.gz")
    d, lm = AT.load_tracelab()
    d = d.rename(columns={"sid": "sess"}).sort_values(["sess", "round"], kind="stable").reset_index(drop=True)
    d["pe"] = prefix_eff(d, a.prefix_rule)
    kg = KeyGen()
    keys, n_shared = keys_by_prefix(d, kg, "pe")
    T0 = d.start_ms.min()
    s0 = d.groupby("sess").start_ms.transform("min")
    res = []
    for c in a.compress:
        x = d.copy()
        x["t"] = ((T0 + (s0 - T0) / c) + (x.start_ms - s0) - T0) / 1000
        x["user"] = x.trigger == "user"
        name = f"tracelab_c{c}" + ("" if a.prefix_rule == "struct" else f"_{a.prefix_rule}") + ("_t128k" if a.trunc_128k else "")
        m = base_meta("tracelab", a)
        m.update(loader=lm, source=dict(sha, release="v0.0.2"), arrival=dict(kind="original" if c == 1 else "compress", factor=c),
                 n_shared_chunks=n_shared)
        res.append(finalize(name, x[["sess", "t", "inp", "user"]], keys, a.out_dir, x1nb, m, a.trunc_128k))
    return res


def conv_semianalysis(a, x1nb):
    src = AT.DS + "/semianalysis_cc_traces"
    sha = _readme_sha(src, "traces.jsonl")
    kg = KeyGen()
    rows = []
    ok = n = 0
    with open(f"{src}/traces.jsonl") as fh:
        for ln in fh:
            r = json.loads(ln)
            bs = r["block_size"]
            assert bs == 64, f"semianalysis block_size={bs}"
            per = CHUNK // bs

            def add(sid, reqs):
                nonlocal ok, n
                for q in reqs:
                    h = q["hash_ids"]
                    n += 1
                    ok += int(len(h) * bs == q["in"])
                    nf = q["in"] // CHUNK
                    ks, pv = [], None
                    for j in range(nf):
                        pv = kg.of((pv, r["id"], tuple(h[per * j:per * (j + 1)])))
                        ks.append(pv)
                    t0 = q["t"]
                    rows.append((r["id"], sid, t0, t0 + (q.get("api_time") or 0.0), q["in"], q.get("type"), ks))

            add(r["id"], [q for q in r["requests"] if q.get("type") != "subagent"])
            for q in r["requests"]:
                if q.get("type") == "subagent":
                    add(f"{r['id']}:{q['agent_id']}", q["requests"])
    rule = AT._check("semianalysis len(hash_ids)×64=in", ok, n)
    d = pd.DataFrame(rows, columns=["conv", "sess", "start_s", "end_s", "inp", "type", "keys"])
    c0 = d.groupby("conv").start_s.min()
    c1 = d.groupby("conv").end_s.max()
    convs = list(c0.index)
    dur = (c1[convs] - c0[convs]).values
    res = []
    for N in a.N:
        for seed in a.seeds:
            st, order = fixed_concurrency(dur, N, seed)
            smap = dict(zip(convs, st))
            x = d.copy()
            x["t"] = [smap[c] + (s - c0[c]) for c, s in zip(x.conv, x.start_s)]
            x["user"] = False
            name = f"semianalysis_N{N}_s{seed}" + ("_t128k" if a.trunc_128k else "")
            m = base_meta("semianalysis", a)
            m.update(loader=dict(rule6=rule), source=dict(sha, revision="23f152f6f0f9399a85901b89a6458def0ef16729"),
                     arrival=dict(kind="fixed_concurrency", N=N, seed=seed, n_conv=len(convs), **active_profile(st, dur)))
            res.append(finalize(name, x[["sess", "t", "inp", "user"]], list(x["keys"].values), a.out_dir, x1nb, m, a.trunc_128k))
    return res


def conv_bailian(a, x1nb):
    src = AT.DS + "/qwen-bailian-usagetraces-anon"
    res = []
    for f in a.files:
        sha = _readme_sha(src, f"qwen_{f}_blksz_16.jsonl")
        d, lm = AT.load_bailian(f)
        recs = [json.loads(ln) for ln in open(f"{src}/qwen_{f}_blksz_16.jsonl")]
        sub = CHUNK // 16
        keys = []
        for r in recs:
            h = r["hash_ids"]
            nf = r["input_length"] // CHUNK
            keys.append([int(h[sub * j + sub - 1]) for j in range(nf)])
        d = d.rename(columns={"sid": "sess"})
        d["t"] = d.start_ms / 1000
        d["user"] = True
        name = f"bailian_{f}" + ("_t128k" if a.trunc_128k else "")
        m = base_meta("bailian", a)
        m.update(loader=lm, source=dict(sha, commit="5f7439c51ec248a0c585f7d90a41a6f57773b912"), arrival=dict(kind="original"))
        res.append(finalize(name, d[["sess", "t", "inp", "user"]], keys, a.out_dir, x1nb, m, a.trunc_128k))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", choices=["cachewise", "tracelab", "semianalysis", "bailian"])
    ap.add_argument("--N", nargs="+", type=int, default=[16, 32, 64, 72])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--copies", type=int, default=5)
    ap.add_argument("--compress", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--files", nargs="+", default=["traceA", "traceB", "thinking", "coder"])
    ap.add_argument("--prefix-rule", choices=["struct", "cache"], default="struct")
    ap.add_argument("--trunc-128k", action="store_true")
    ap.add_argument("--out-dir", default=os.environ.get("X1_NB_DIR", "/ssd6/hungwei/icpe-next/datasets/x1_nb"))
    ap.add_argument("--p3x", default=os.path.join(REPO, "code", "p3x"))
    a = ap.parse_args()
    x1nb = _x1nb(a.p3x)
    fn = dict(cachewise=conv_cachewise, tracelab=conv_tracelab, semianalysis=conv_semianalysis, bailian=conv_bailian)[a.trace]
    res = fn(a, x1nb)
    summ = os.path.join(a.out_dir, "next_b_trace_summary.jsonl")
    with open(summ, "a") as fh:
        for m in res:
            fh.write(json.dumps({k: v for k, v in m.items() if k not in ("loader",)}, ensure_ascii=False, default=str) + "\n")
    print(f"→ {summ}")


if __name__ == "__main__":
    main()
