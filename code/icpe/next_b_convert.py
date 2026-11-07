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
      - Bailian（16-token、**逐塊** salted SipHash，不是鏈式；上游 README）：chunk j 的鍵＝hash_ids[0:32(j+1)] 這整段前綴的編號
        （同 SemiAnalysis；2026-11-04 traceA 的鏈式檢查違反 301,710 次，所以不能取 hash_ids[32j+31]）。
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
            if len(x) >= 2 and x[-1] == rel and len(x[0]) == 64:
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
        kg = KeyGen()
        keys = []
        for r in recs:          # 逐塊 hash（不是鏈式）→ chunk j 的鍵＝hash_ids[0:32(j+1)] 這整段前綴的編號（同 SemiAnalysis）
            h = r["hash_ids"]
            nf = r["input_length"] // CHUNK
            ks, pv = [], None
            for j in range(nf):
                pv = kg.of((pv, tuple(h[sub * j:sub * (j + 1)])))
                ks.append(pv)
            keys.append(ks)
        d = d.rename(columns={"sid": "sess"})
        d["t"] = d.start_ms / 1000
        d["user"] = True
        name = f"bailian_{f}" + ("_t128k" if a.trunc_128k else "")
        m = base_meta("bailian", a)
        m.update(loader=lm, source=dict(sha, commit="5f7439c51ec248a0c585f7d90a41a6f57773b912"), arrival=dict(kind="original"))
        res.append(finalize(name, d[["sess", "t", "inp", "user"]], keys, a.out_dir, x1nb, m, a.trunc_128k))
    return res


def conv_fi(a, x1nb):
    """FreeInference 的一週 → nb（只為了截斷 128K 的敏感度；主結果仍走 x1_sim 的 fi: 入口）。
    讀 code/p3x/x1_fi.prepare_fi 的結果（不改那支檔；它自己做規則 6 的三項斷言）。nall＝max(ceil(Σrole_tokens/512), 完整塊數)，
    所以「輸入 > 131072 token」＝nall > 256；這裡用 inp＝nall×512 當截斷判準（只影響判準與 meta 的長度統計）。"""
    os.environ.setdefault("X1_FI_DIR", "/ssd8/hungwei/paper-hkv/datasets/freeinference_agentic_trace/release")
    sys.path.insert(0, a.p3x)
    import x1_fi
    res = []
    for w in a.weeks:
        P = x1_fi.prepare_fi(w)
        d = pd.DataFrame(dict(sess=P["R"]["sess"], t=P["t"], inp=np.asarray(P["nall"], np.int64) * CHUNK, user=P["user_trigger"]))
        name = f"fi_{w}" + ("_t128k" if a.trunc_128k else "")
        m = base_meta("freeinference", a)
        m.update(loader={k: v for k, v in P["meta"].items() if k != "path"}, source=dict(dir=x1_fi.FI_DIR, revision="16d3c04d"),
                 arrival=dict(kind="original"), note="inp＝nall×512（截斷判準用；真正的 token 數在 x1_fi）")
        res.append(finalize(name, d, [list(k) for k in P["full"]], a.out_dir, x1nb, m, a.trunc_128k))
    return res


# ------------------------------------------------------------------ 公開對話資料集（使用者 11/7：ShareGPT、WildChat、LMSYS-Chat-1M、UltraChat；
# 登記＝STATUS_DATA 2026-11-07「對話資料集補登記」那節，跑模擬之前 commit）
# 共同規則：
#   * 合格的對話：訊息嚴格 user、assistant 交替、由 user 開始（最後可以停在 user）、內容都是字串、user 至少 2 輪；
#     WildChat 另外要每個 assistant 都有 timestamp、最後一則是 assistant。不合格的依原因計數（meta.filter）。
#   * 抽樣：合格的對話依載入順序編號，numpy default_rng(0).choice(不放回) 抽 CHAT_N＝9,000 個（CachedAttention 的 9K sessions）。
#   * token：Llama-3.1-8B-Instruct 的 tokenizer（NousResearch 鏡像 @ d10aef79，tokenizer.json sha256 固定），每則訊息單獨切，
#     再照 Llama 3.1 的對話格式組起來：<|begin_of_text|>，每則訊息＝<|start_header_id|>角色<|end_header_id|>\n\n 內容 <|eot_id|>；
#     不加 system 訊息。第 k 個 user 訊息觸發第 k 個請求，請求的輸入＝到這則 user 訊息為止的整段＋assistant 的開頭（4 個 token）。
#     所以上一個請求的輸入一定是下一個請求的前綴（只往後加）。
#   * chunk 鍵：整段 token 序列每 512 個一塊做鏈式 blake2b（前一塊的摘要＋這一塊的 token），內容相同的前綴（不同對話也一樣）共用鍵，
#     和 vLLM 前綴快取的行為一致。
#   * 規則 6（用 A 驗算 B）：把前 CHAT_CHECK 個抽到的對話照同一格式組成「文字」，整段一次切 token，和逐則切再組起來的長度比；
#     差異（|整段−逐則|／整段）的中位數 >1% 就停。另外檢查每個請求的鍵是上一個請求的鍵的延伸（write_nb 也會查）。
#   * 時間：
#       - ShareGPT、LMSYS-Chat-1M、UltraChat（沒有時間）：照 Pensieve／CachedAttention。對話開始＝Poisson，λ＝1 個對話／秒
#         （對話順序＝default_rng(seed).permutation）；同一個對話裡兩個請求的間隔＝思考時間 ~ 指數分佈，平均 T＝60 s（主）／600 s（敏感度）。
#         同一個 seed 的 T60 與 T600 用同一組單位指數亂數（T600＝T60 的間隔 ×10），到達順序也相同。
#         不加生成時間：模擬器不模擬解碼，也沒有量過的每 token 解碼時間（NOT_MEASURED，不自己編）。
#       - WildChat-1M：對話內用真實時間。第 k 個請求的偏移＝第 k 個 assistant 的 timestamp − 第 1 個 assistant 的 timestamp
#         （伺服器收到完整回覆的時間；請求本身的到達時間沒有記錄，用回覆完成時間代替，誤差是相鄰兩輪生成時間的差）。
#         對話開始一樣合成：Poisson λ＝1／s、同樣的 seed 與順序規則。原始的絕對時間太稀（一年約 100 萬個對話），不用。
CHAT_N = 9000
CHAT_LAMBDA = 1.0
CHAT_T = (60, 600)
CHAT_CHECK = 300
CHAT_STATS = {}          # 載入時的特殊情況計數（例：UltraChat 一行兩個物件）；寫進 meta.filter.load_stats
TOK_SHA = "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4"
BOT, SHD, EHD, EOT, NL2 = 128000, 128006, 128007, 128009, 271
ROLE_TOK = {"user": [882], "assistant": [78191]}
CHAT_SRC = dict(
    sharegpt=dict(dir="sharegpt", files=["HTML_cleaned_raw_dataset/sg_90k_part1_html_cleaned.json",
                                         "HTML_cleaned_raw_dataset/sg_90k_part2_html_cleaned.json"],
                  hf="anon8231489123/ShareGPT_Vicuna_unfiltered", rev="192ab2185289094fc556ec8ce5ce1e8e587154ca", license="Apache-2.0"),
    wildchat=dict(dir="wildchat-1m", files=[f"data/train-{i:05d}-of-00014.parquet" for i in range(14)],
                  hf="allenai/WildChat-1M", rev="7d6490e462285cf85d91eabea0f9a954fbddcd1f", license="ODC-BY"),
    lmsys=dict(dir="lmsys-chat-1m", files=None, hf="lmsys/lmsys-chat-1m", rev="200748d9d3cddcc9d782887541057aca0b18c5da",
               license="LMSYS-Chat-1M Dataset License Agreement (no redistribution)"),
    ultrachat=dict(dir="ultrachat", files=[f"train_{i}.jsonl" for i in range(10)], hf="openbmb/UltraChat",
                   rev="f220fe796ce3ed62fbe1681b45ce6cbc9c6cabe0", license="MIT"),
)


def _tokenizer(path, pylibs):
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if h != TOK_SHA:
        raise SystemExit(f"{path}: tokenizer sha256 {h} ≠ {TOK_SHA} → 停")
    if pylibs:
        sys.path.insert(0, pylibs)
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(path)
    for s, want in (("<|begin_of_text|>", BOT), ("<|start_header_id|>", SHD), ("<|end_header_id|>", EHD), ("<|eot_id|>", EOT)):
        assert tk.token_to_id(s) == want, (s, tk.token_to_id(s))
    for s, want in (("\n\n", [NL2]), ("user", ROLE_TOK["user"]), ("assistant", ROLE_TOK["assistant"])):
        assert tk.encode(s, add_special_tokens=False).ids == want, s
    import tokenizers
    return tk, dict(file=path, sha256=h, tokenizers_version=tokenizers.__version__)


def _iter_chat(src, d, files):
    """依固定順序逐一產生 (對話 id, [(role, content, ts秒或None)])；role 已正規化成 user／assistant，其他角色原樣保留（之後判不合格）。"""
    if src == "sharegpt":
        rmap = {"human": "user", "gpt": "assistant"}
        for f in files:
            for r in json.load(open(os.path.join(d, f))):
                yield r.get("id"), [(rmap.get(m.get("from"), str(m.get("from"))), m.get("value"), None) for m in (r.get("conversations") or [])]
    elif src == "ultrachat":
        # 上游有 3 行是兩個 JSON 物件直接接在一起（icpe-trx 查到：train_3 第 127,819 行、train_4 第 82,900 行、train_7 第 90,760 行；
        # 檔案 sha256＝HF 的 LFS 記錄，是上游本來就這樣）。規則（STATUS_DATA 11:30 補寫）：用 raw_decode 依序拆成多筆，每筆都是一個對話；
        # 拆出來的行數記在 CHAT_STATS["ultrachat_multi_obj_lines"]。拆不開（剩下的不是 JSON）就停。
        dec = json.JSONDecoder()
        for f in files:
            with open(os.path.join(d, f)) as fh:
                for ln in fh:
                    s = ln.strip()
                    pos, objs = 0, []
                    while pos < len(s):
                        r, pos = dec.raw_decode(s, pos)
                        objs.append(r)
                        while pos < len(s) and s[pos].isspace():
                            pos += 1
                    if len(objs) > 1:
                        CHAT_STATS["ultrachat_multi_obj_lines"] = CHAT_STATS.get("ultrachat_multi_obj_lines", 0) + 1
                    for r in objs:
                        yield r.get("id"), [("user" if i % 2 == 0 else "assistant", c, None) for i, c in enumerate(r.get("data") or [])]
    elif src in ("wildchat", "lmsys"):
        import pyarrow.parquet as pq
        idc = "conversation_hash" if src == "wildchat" else "conversation_id"
        for f in files:
            pf = pq.ParquetFile(os.path.join(d, f))
            for g in range(pf.num_row_groups):
                for r in pf.read_row_group(g, columns=[idc, "conversation"]).to_pylist():
                    msgs = []
                    for m in r["conversation"] or []:
                        ts = m.get("timestamp") if src == "wildchat" else None
                        msgs.append((m.get("role"), m.get("content"), ts.timestamp() if ts is not None else None))
                    yield r[idc], msgs
    else:
        raise SystemExit(src)


def _chat_ok(src, msgs):
    for i, (role, c, ts) in enumerate(msgs):
        if role != ("user" if i % 2 == 0 else "assistant"):
            return "not_alternating"
        if not isinstance(c, str):
            return "content_not_str"
        if src == "wildchat" and role == "assistant" and ts is None:
            return "assistant_no_ts"
    if sum(1 for m in msgs if m[0] == "user") < 2:
        return "lt2_user_turns"
    if src == "wildchat" and msgs[-1][0] != "assistant":
        return "ends_with_user"
    return None


def conv_chat(a, x1nb):
    src = a.source
    S = CHAT_SRC[src]
    d = os.path.join(a.src_dir or AT.DS, S["dir"])
    files = S["files"]
    if files is None:                     # LMSYS 的 parquet 檔名有雜湊尾碼：依名稱排序取 data/*.parquet
        files = sorted(f"data/{x}" for x in os.listdir(os.path.join(d, "data")) if x.endswith(".parquet"))
        assert len(files) == 6, files
    shas = [_readme_sha(d, f) for f in files]
    assert all(s["readme_sha256"] for s in shas), "README.txt 沒有記 sha256 → 停"
    tk, tok_meta = _tokenizer(a.tokenizer, a.pylibs)
    # 第一輪：合格與否，只記編號
    filt, elig, n_all = {}, [], 0
    for i, (cid, msgs) in enumerate(_iter_chat(src, d, files)):
        n_all += 1
        why = _chat_ok(src, msgs) or "ok"
        filt[why] = filt.get(why, 0) + 1
        if why == "ok":
            elig.append(i)
    load_stats = dict(CHAT_STATS)          # 只算第一輪（第二輪會再讀一次）
    n_pick = min(CHAT_N, len(elig))
    pick = set(np.asarray(elig)[np.sort(np.random.default_rng(0).choice(len(elig), size=n_pick, replace=False))].tolist())
    print(f"[{src}] 對話 {n_all}，合格 {len(elig)}，抽 {n_pick}；{filt}", flush=True)
    # 第二輪：取出抽到的對話
    convs = [(cid, msgs) for i, (cid, msgs) in enumerate(_iter_chat(src, d, files)) if i in pick]
    assert len(convs) == n_pick, (len(convs), n_pick)
    flat = [c for _, msgs in convs for (_, c, _) in msgs]
    ids = [e.ids for e in tk.encode_batch(flat, add_special_tokens=False)]
    # 組序列、算請求
    kg = KeyGen()
    req = []          # (對話序號, 請求序號 k, inp, keys, 第 k 個 assistant 的 ts)
    p = n_ext_ok = n_ext = 0
    seq_lens = []
    for ci, (cid, msgs) in enumerate(convs):
        seq = [BOT]
        reqpos = []
        for (role, _, ts) in msgs:
            if role == "assistant":
                seq += ids[p] + [EOT]          # assistant 的開頭在 user 那一步已經加了
            else:
                seq += [SHD] + ROLE_TOK["user"] + [EHD, NL2] + ids[p] + [EOT] + [SHD] + ROLE_TOK["assistant"] + [EHD, NL2]
                reqpos.append(len(seq))
            p += 1
        ats = [ts for (role, _, ts) in msgs if role == "assistant"]
        H, pv = [], b""
        for j in range(len(seq) // CHUNK):
            pv = hashlib.blake2b(pv + np.asarray(seq[CHUNK * j:CHUNK * (j + 1)], np.int32).tobytes(), digest_size=16).digest()
            H.append(kg.of(pv))
        prev = None
        for k, pos in enumerate(reqpos):
            ks = H[:pos // CHUNK]
            if prev is not None:
                n_ext += 1
                n_ext_ok += int(ks[:len(prev)] == prev)
            req.append((ci, k, pos, ks, ats[k] if k < len(ats) else None))
            prev = ks
        seq_lens.append(len(seq))
    assert p == len(flat)
    rule_ext = AT._check(f"{src} 請求的鍵＝上一個請求的鍵的延伸", n_ext_ok, n_ext, thr=1.0)
    # 規則 6：整段文字一次切 vs 逐則切再組
    diffs = []
    for cid, msgs in convs[:CHAT_CHECK]:
        txt = "<|begin_of_text|>" + "".join(
            (f"<|start_header_id|>user<|end_header_id|>\n\n{c}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" if role == "user"
             else f"{c}<|eot_id|>") for (role, c, _) in msgs)
        n_whole = len(tk.encode(txt, add_special_tokens=False).ids)
        n_parts = 1 + sum((len(tk.encode(c, add_special_tokens=False).ids) + (9 if role == "user" else 1)) for (role, c, _) in msgs)
        diffs.append(abs(n_whole - n_parts) / max(n_whole, 1))
    diffs = np.asarray(diffs)
    tok_check = dict(n=len(diffs), median=float(np.median(diffs)), p90=float(np.quantile(diffs, .9)), max=float(diffs.max()),
                     frac_exact=float((diffs == 0).mean()))
    print(f"[{src}] 規則 6（整段切 vs 逐則切）：{tok_check}", flush=True)
    if not tok_check["median"] <= 0.01:
        raise SystemExit(f"[{src}] 整段切與逐則切的長度差中位數 {tok_check['median']:.4f} > 1% → 停")
    R = pd.DataFrame(req, columns=["conv", "k", "inp", "keys", "ats"])
    n_conv = len(convs)
    res = []
    variants = [None] if src == "wildchat" else list(CHAT_T)
    for seed in a.seeds:
        rng = np.random.default_rng(seed)
        order = rng.permutation(n_conv)
        st_o = np.cumsum(rng.exponential(1.0 / CHAT_LAMBDA, n_conv))
        start = np.empty(n_conv)
        start[order] = st_o
        unit = rng.exponential(1.0, len(R))             # 每個請求一個單位指數亂數；每個對話的第一個不用
        for T in variants:
            x = R[["conv", "k", "inp"]].copy()
            if T is not None:
                g = np.where(x.k.values == 0, 0.0, unit * T)
                off = pd.Series(g).groupby(x.conv.values).cumsum().values
                arrival = dict(kind="poisson_sessions+exp_think", lam_per_s=CHAT_LAMBDA, think_mean_s=T, seed=seed)
                name = f"{src}_T{T}_s{seed}"
            else:
                a0 = R.groupby("conv").ats.transform("first")
                off = (R.ats - a0).values.astype(float)
                assert np.isfinite(off).all()
                arrival = dict(kind="poisson_sessions+real_gaps(assistant_ts)", lam_per_s=CHAT_LAMBDA, seed=seed)
                name = f"{src}_s{seed}"
            x["t"] = start[x.conv.values] + off
            x["sess"] = x.conv
            x["user"] = True
            gaps = pd.Series(off).groupby(x.conv.values).diff().dropna()
            dur = pd.Series(off).groupby(x.conv.values).max().values
            m = base_meta(src, a)
            m.update(source=dict(hf=S["hf"], revision=S["rev"], license=S["license"], files=shas), tokenizer=tok_meta,
                     filter=dict(n_conv_all=n_all, n_eligible=len(elig), n_sampled=n_pick, reasons=filt, sample_seed=0,
                                 load_stats=load_stats),
                     rule6=dict(keys_extend=rule_ext, tokenize_whole_vs_parts=tok_check),
                     conv_tokens=dict(median=float(np.median(seq_lens)), p90=float(np.quantile(seq_lens, .9)), max=int(max(seq_lens))),
                     arrival=dict(arrival, n_conv=n_conv, **active_profile(start, dur)),
                     gap_s=dict(median=float(gaps.median()), p10=float(gaps.quantile(.1)), p90=float(gaps.quantile(.9)),
                                frac_lt_0=float((gaps < 0).mean())))
            res.append(finalize(name, x[["sess", "t", "inp", "user"]], list(R["keys"].values), a.out_dir, x1nb, m, False))
    return res


# ------------------------------------------------------------------ 公開 agent 軌跡（PREREG 偏離 17；STATUS_DATA 2026-11-07 12:55）
AGENT_SRC = dict(
    miniswe=dict(dir="miniswe-qwen3-30b", hf="tarsur385/qwen3-30b-a3b-instruct-2507-swebench-verified-mini-swe-agent",
                 rev="58389eb5426db5a3ae6159faa37498072b4735ff", license="MIT"),
    openhands=dict(dir="openhands-eval", hf="OpenHands/openhands-evaluation-outputs", rev="aa8977805b4cefd317001d80ddf1ad52790e9d23",
                   license="MIT", file="outputs/SWE-bench_Lite-test/CodeActAgent/claude-3-5-sonnet-20241022_maxiter_100_N_v2.2-no-hint/output.jsonl",
                   src="openhands-src"),
)
ROLE_TOK_ALL = {"system": [9125], "user": [882], "assistant": [78191]}   # ipython 由 tokenizer 取得並核對


def conv_agent(a, x1nb):
    import next_agent_traces as NAT
    src = a.source
    S = AGENT_SRC[src]
    base = a.src_dir or AT.DS
    d = os.path.join(base, S["dir"])
    tk, tok_meta = _tokenizer(a.tokenizer, a.pylibs)
    roles = dict(ROLE_TOK_ALL)
    roles["ipython"] = tk.encode("ipython", add_special_tokens=False).ids
    for r, ids in roles.items():
        assert tk.encode(r, add_special_tokens=False).ids == ids, r
    if src == "miniswe":
        files = [ln.split()[1] for ln in open(os.path.join(d, "README.txt")) if ln.strip() and not ln.startswith("#")]
        shas = [_readme_sha(d, f) for f in files if f.startswith("trajectories/")]
        assert len(shas) == 499 and all(s["readme_sha256"] for s in shas), len(shas)
        sess, st = NAT.load_miniswe(d)
        rule_c = AT._check("miniswe api_calls＝成功＋FormatError", st["api_eq"], st["n_traj"], thr=1.0)
        shas = dict(n_files=len(shas), readme=os.path.join(d, "README.txt"))
    else:
        shas = [_readme_sha(d, S["file"])]
        sd = os.path.join(base, S["src"])
        src_sha = [_readme_sha(sd, f"{c}/{f}") for c in ("61874438105ba42dfc2bcb955d329d5c925e41d5", "6333c6ad58d790e8728bd3c8ecd5850173b1280e")
                   for f in ("codeact_agent.py", "function_calling.py", "event.py")]
        assert all(s["readme_sha256"] for s in shas + src_sha)
        sess, st = NAT.load_openhands(os.path.join(d, S["file"]), sd)
        rule_c = None
        shas = dict(data=shas, source=src_sha)
    print(f"[{src}] 軌跡 {st['n_traj']}，合格 {st['n_ok']}，成功呼叫 {st['calls_ok']}；{ {k: v for k, v in st.items() if k not in ('n_traj', 'n_ok', 'calls_ok')} }", flush=True)
    # token 化（逐則切，再組；同偏離 13）
    flat = [txt for s in sess for (_, txt) in s["msgs"]]
    ids = [e.ids for e in tk.encode_batch(flat, add_special_tokens=False)]
    kg = KeyGen()
    rows = []          # (sess, k, inp, keys, t_req, t_end, user)
    p = 0
    ratios, first_ratio, n_ext_ok, n_ext = [], [], 0, 0
    conv_dur, conv_t0 = [], []
    for si, s in enumerate(sess):
        seq, mstart = [BOT], []
        for (role, _) in s["msgs"]:
            mstart.append(len(seq))
            seq += [SHD] + roles[role] + [EHD, NL2] + ids[p] + [EOT]
            p += 1
        H, pv = [], b""
        for j in range(len(seq) // CHUNK):
            pv = hashlib.blake2b(pv + np.asarray(seq[CHUNK * j:CHUNK * (j + 1)], np.int32).tobytes(), digest_size=16).digest()
            H.append(kg.of(pv))
        prev = None
        t0 = s["calls"][0]["t_req"]
        for k, c in enumerate(s["calls"]):
            assert s["msgs"][c["m"]][0] == "assistant"
            inp = mstart[c["m"]] + 4                      # 到這則 assistant 訊息之前＋assistant 開頭
            ks = H[:inp // CHUNK]
            if prev is not None:
                n_ext += 1
                n_ext_ok += int(ks[:len(prev[0])] == prev[0])
                if c["ptok"] is not None and prev[2] is not None and c["ptok"] > prev[2]:
                    ratios.append((inp - prev[1]) / (c["ptok"] - prev[2]))
            elif c["ptok"]:
                first_ratio.append(inp / c["ptok"])
            rows.append((si, k, inp, ks, c["t_req"] - t0, c["t_end"] - t0, k == 0))
            prev = (ks, inp, c["ptok"])
        conv_t0.append(t0)
        conv_dur.append(max(c["t_end"] for c in s["calls"]) - t0)
    assert p == len(flat)
    rule_a = AT._check(f"{src} 請求的鍵＝上一個請求的鍵的延伸", n_ext_ok, n_ext, thr=1.0)
    r = np.asarray(ratios)
    rule_b = dict(n=len(r), median=float(np.median(r)), p10=float(np.quantile(r, .1)), p90=float(np.quantile(r, .9)),
                  frac_in_0p5_2=float(((r >= 0.5) & (r <= 2.0)).mean()), first_call_total_ratio_median=float(np.median(first_ratio)))
    print(f"[{src}] 規則 6 (b)（Llama 增量 ÷ 供應商 prompt_tokens 增量）：{rule_b}", flush=True)
    if not (0.8 <= rule_b["median"] <= 1.25 and rule_b["frac_in_0p5_2"] >= 0.9):
        raise SystemExit(f"[{src}] 規則 6 (b) 不過：{rule_b} → 停")
    R = pd.DataFrame(rows, columns=["conv", "k", "inp", "keys", "off", "end", "user"])
    neg = int((R.groupby("conv").off.diff() < 0).sum())
    dur = np.asarray(conv_dur)
    res = []
    for N in a.N:
        for seed in a.seeds:
            st0, order = fixed_concurrency(dur, N, seed)
            x = R[["conv", "inp", "user"]].copy()
            x["t"] = st0[R.conv.values] + R.off.values
            x["sess"] = x.conv
            name = f"{src}_N{N}_s{seed}"
            m = base_meta(src, a)
            gaps = R.groupby("conv").off.diff().dropna()
            m.update(source=dict(hf=S["hf"], revision=S["rev"], license=S["license"], files=shas), tokenizer=tok_meta,
                     loader=st, rule6=dict(keys_extend=rule_a, prompt_tokens_increment=rule_b, api_calls=rule_c),
                     arrival=dict(kind="fixed_concurrency", N=N, seed=seed, n_conv=len(sess), **active_profile(st0, dur)),
                     gap_s=dict(median=float(gaps.median()), p10=float(gaps.quantile(.1)), p90=float(gaps.quantile(.9)), n_negative=neg),
                     conv_dur_s=dict(median=float(np.median(dur)), p90=float(np.quantile(dur, .9)), max=float(dur.max())))
            res.append(finalize(name, x[["sess", "t", "inp", "user"]], list(R["keys"].values), a.out_dir, x1nb, m, False))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", choices=["cachewise", "tracelab", "semianalysis", "bailian", "fi", "chat", "agent"])
    ap.add_argument("--source", choices=list(CHAT_SRC) + list(AGENT_SRC), help="chat／agent：哪一份資料集")
    ap.add_argument("--src-dir", default=None, help="chat：資料集的上層目錄（預設 agent_traces.DS）；底下是 sharegpt／wildchat-1m／lmsys-chat-1m／ultrachat")
    ap.add_argument("--tokenizer", default="/ssd7/hungwei/hf-cache/huggingface/hub/models--NousResearch--Meta-Llama-3.1-8B-Instruct/"
                    "snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77/tokenizer.json")
    ap.add_argument("--pylibs", default=os.environ.get("NEXT_B_PYLIBS", "/ssd6/hungwei/icpe-next/pylibs"),
                    help="chat：裝 tokenizers 的旁路目錄（python -I 不讀 PYTHONPATH）")
    ap.add_argument("--weeks", nargs="+", default=["2026-05-17_2026-05-23"])
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
    if a.trace in ("chat", "agent") and not a.source:
        ap.error("chat／agent 要給 --source")
    fn = dict(cachewise=conv_cachewise, tracelab=conv_tracelab, semianalysis=conv_semianalysis, bailian=conv_bailian, fi=conv_fi,
              chat=conv_chat, agent=conv_agent)[a.trace]
    res = fn(a, x1nb)
    summ = os.path.join(a.out_dir, "next_b_trace_summary.jsonl")
    with open(summ, "a") as fh:
        for m in res:
            fh.write(json.dumps({k: v for k, v in m.items() if k not in ("loader",)}, ensure_ascii=False, default=str) + "\n")
    print(f"→ {summ}")


if __name__ == "__main__":
    main()
