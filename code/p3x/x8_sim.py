"""x8_sim.py — X8：四種去處（GPU 釘住／CPU／慢層／丟）的 session 層級模擬，oracle vs 固定規則

判準與設定事先寫在 docs/phase3_20261017/explore/X8_agents_longctx.md §2（2026-10-17T09:43:22Z）。

模型（〔模擬〕；成本〔MI300X 常數〕）：
  * 一次一個請求、沒有排隊：TTFT＝還原（Cake 兩條線，m7_sim.sim_restore 的同一演算法）＋新一輪 prefill。
  * 每個 session 閒置時整份 KV 在一層：G（GPU 釘住預算）→ C（CPU）→ S（慢層，容量不限；或沒有慢層）→ 丟。
  * 層的容量用 block（512 token × 64 MiB，Llama-3.1-8B BF16）。
  * 放置軌跡和成本無關（只有 O-GDS 例外），所以每個 (G, C, 慢層有/無) 先跑軌跡，再用不同成本表算 TTFT。

策略（參數事先寫死）：
  F-LRU         G 當 LRU 前綴快取 → C LRU → S
  F-TTL-T       T ∈ {0,2,10,60,300} s：釘在 G T 秒（擠不下時先趕最早到期的），到期降到 C；C 以下 LRU
  F-Cont        Continuum 式：T＝trace 前半段的工具時長中位數（CacheWise：依工具組合；其他：前半段回來間隔中位數）
  O-Belady      知道下次回來時間：每層滿了把「下次回來最晚」的往下推；永遠不回來的直接丟
  O-GDS         知道下次回來時間與要還原的 block 數：趕「(距回來時間 × 大小) ÷ 留在這層省下的還原時間」最大的
  P-Ret         （只報告，登記後才加）F-LRU，但知道「會不會回來」：不回來的直接丟；不知道何時回來
  H-Turn2       （只報告，看過結果才加）F-LRU，但 session 第一個請求留下的 KV 在每一層都先被趕（FIFO）；不用未來資訊
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RES7 = os.path.join(REPO, "results", "m7_write_policy_mi300x")
BLK = 512
GIB_PER_BLK = 64 / 1024        # 64 MiB
NMAX = int(os.environ.get("X8_NMAX", 1024))   # FreeInference 最長 2,460 個 block → 跑它時設 2560（只影響表的長度，不影響其他格子的數字）
BUSY = 2.07                    # main.tex §3.4：背景 matmul 0.7 duty → 重算慢 2.07×
INF = float("inf")

G_LIST = [0, 16, 64]           # GiB
C_LIST = [128, 256, 512, 1024]  # GiB
SLOWS = ["local", "nfs", "none"]
GPUS = ["idle", "busy"]
TTLS = [0, 2, 10, 60, 300]
FIXED = ["F-LRU"] + [f"F-TTL-{t}" for t in TTLS] + ["F-Cont"]
ORACLES = ["O-Belady", "O-GDS"]
POSTHOC = ["H-Turn2"]          # 只報告（看過結果才加，main.tex「Post-hoc rivals」）：F-LRU，但 session 第一個請求留下的 KV 先趕（無參數）
PARTIAL = ["P-Ret"]            # 只報告（登記之後才加，不進判定）：F-LRU＋只知道「會不會回來」（不回來的直接丟）
LENS = ["orig", "64K", "128K"]
MAXCTX = 131072


def rid():
    return os.environ.get("RUN_ID", "adhoc")


def now_ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ------------------------------------------------------------------ 成本
def load_costs():
    import pandas as pd
    d = pd.read_csv(os.path.join(RES7, "calib_c1.csv"))
    d = d[(d.item == "f_chunk") & (d.gpu_state == "idle")]
    m = d.groupby("chunk_idx").ms.median().sort_index()
    x, y = m.index.values.astype(float), m.values.astype(float) / 1e3
    slope, icpt = np.polyfit(x, y, 1)
    f = np.array([y[i] if i < len(y) else icpt + slope * i for i in range(NMAX)])
    tp = json.load(open(os.path.join(RES7, "tier_params.json")))
    ell = {k: tp[k]["ell_chunk_ms"] / 1e3 for k in ("cpu", "local", "nfs")}
    meta = dict(f_run=sorted(d.run_id.unique()), f_measured_blocks=len(y), f_slope_ms=slope * 1e3, f_icpt_ms=icpt * 1e3,
                tier_runs=tp["source_runs"], ell_ms={k: v * 1e3 for k, v in ell.items()})
    return f, ell, meta


def cake(n, ell, f):
    """Cake 兩條線：重算從前、載入從後，誰先空誰拿下一個（同 m7_sim.sim_restore、單一層、無爭用）。"""
    p, q, rt, lt = 0, n - 1, 0.0, 0.0
    while p <= q:
        if rt <= lt:
            rt += f[p]
            p += 1
        else:
            lt += ell
            q -= 1
    return max(rt, lt), p


def cost_tables(f, ell):
    T = {}
    for g in GPUS:
        fg = f * (BUSY if g == "busy" else 1.0)
        cum = np.concatenate([[0.0], np.cumsum(fg)])
        T[(g, "D")] = cum.copy()                         # 丟：全部重算
        for tier in ("cpu", "local", "nfs"):
            T[(g, tier)] = np.array([cake(n, ell[tier], fg)[0] for n in range(NMAX + 1)])
        T[(g, "cum")] = cum
        T[(g, "f")] = fg
    return T


def selftest(f, ell):
    """和 m7_sim.sim_restore（第一階段已驗證的實作）逐格比對。"""
    sys.path.insert(0, os.path.join(REPO, "code"))
    import m7_sim
    from m7_restore_harness import GiB, Tier
    CB = 64 * (1 << 20)
    worst = 0.0
    for tier in ("cpu", "local", "nfs"):
        for n in (1, 2, 5, 17, 64, 80, 157, 256):
            T = Tier(tier if tier == "cpu" else "ssd_" + tier, read_Bps=CB / ell[tier], write_Bps=CB / ell[tier])
            t_ref, meet_ref, _, _ = m7_sim.sim_restore(n, [T] * n, list(f), 0.0, "cake")
            t_me, meet_me = cake(n, ell[tier], f)
            worst = max(worst, abs(t_ref - t_me))
            assert meet_ref == meet_me and abs(t_ref - t_me) < 1e-9, (tier, n, t_ref, t_me, meet_ref, meet_me)
    return worst


# ------------------------------------------------------------------ 工作負載
def load_wl(trace_dir, wl, length, dedup=False):
    base = {"mc_conv": "mooncake_conversation", "mc_tool": "mooncake_toolagent", "bl_coder": "bailian_coder",
            "cw_orig": "cachewise", "cw_stack": "cachewise",
            "mc_conv_s": "mooncake_conversation_strict", "mc_tool_s": "mooncake_toolagent_strict",
            "fi": "freeinference", "fi_clip": "freeinference"}[wl]
    z = np.load(os.path.join(trace_dir, base + ".npz"), allow_pickle=True)
    d = {k: z[k] for k in z.files}
    if wl == "cw_stack":   # 對齊：每個 session 都從 0 開始（最大重疊；登記於 X8 §2.4）
        off = d["sess_start"]
        d["t_arr"] = d["t_arr"] - off
        d["t_idle"] = d["t_idle"] - off
    o = np.lexsort((np.arange(len(d["t_arr"])), d["t_arr"]))
    d = {k: v[o] for k, v in d.items()}
    meta = dict(k_scale=1.0, frac_clipped=0.0)
    if wl == "fi_clip":   # 敏感度（只報告）：截到 Llama-3.1-8B 的上限 131,072 token
        meta["frac_clipped"] = float(np.mean(d["ctx_tok"] > MAXCTX))
        d["ctx_tok"] = np.minimum(d["ctx_tok"], MAXCTX)
        d["obj_tok"] = np.minimum(d["obj_tok"], MAXCTX)
        d["nold_tok"] = np.minimum(d["nold_tok"], d["ctx_tok"])
    if length != "orig":
        L = {"64K": 65536, "128K": 131072}[length]
        k = L / float(np.median(d["ctx_tok"]))
        ctx = np.round(d["ctx_tok"] * k)
        meta["frac_clipped"] = float(np.mean(ctx > MAXCTX))
        d["ctx_tok"] = np.minimum(ctx, MAXCTX)
        d["obj_tok"] = np.minimum(np.round(d["obj_tok"] * k), MAXCTX)
        d["nold_tok"] = np.minimum(np.round(d["nold_tok"] * k), d["ctx_tok"])
        d["shared_tok"] = np.minimum(np.round(d["shared_tok"] * k), d["ctx_tok"])
        meta["k_scale"] = k
    n = len(d["sid"])
    # 下一個同 session 請求
    nxt = np.full(n, -1, dtype=np.int64)
    last = {}
    for i in range(n - 1, -1, -1):
        s = int(d["sid"][i])
        if s in last:
            nxt[i] = last[s]
        last[s] = i
    d["nxt"] = nxt
    # 穩健性（事後加，主 session 10/17 X9 提醒）：每個回來請求的「決定時間」＝同 session 前一個請求的閒置開始；
    # 觀察窗結束＝trace 最後一個到達（FreeInference：該週最後一個到達，session 不跨週）
    t_dec = np.full(n, np.nan)
    for i in range(n):
        j = nxt[i]
        if j >= 0:
            t_dec[j] = d["t_idle"][i]
    d["t_dec"] = t_dec
    if "week" in d:
        wend = {int(w): float(d["t_arr"][d["week"] == w].max()) for w in np.unique(d["week"])}
        d["t_end"] = np.array([wend[int(w)] for w in d["week"]])
    else:
        d["t_end"] = np.full(n, float(d["t_arr"].max()))
    # 評估段
    if wl.startswith("fi"):
        d["eval"] = d["week"] >= 8    # 後 8 週評估（session 不跨週）
        train_req = d["week"] < 8
        meta["split"] = "weeks 9-16 eval, weeks 1-8 Continuum statistics"
    elif wl.startswith("cw"):
        ss = d["sess_start"]
        cut = float(np.median(np.unique(ss)))
        d["eval"] = ss > cut          # 依 session 原始開始時間的後半
        train_req = ss <= cut
        meta["split"] = f"sessions with original start > median ({cut:.0f})"
    else:
        tmid = 0.5 * (d["t_arr"].min() + d["t_arr"].max())
        d["eval"] = d["t_arr"] >= tmid
        train_req = d["t_arr"] < tmid
        meta["split"] = f"t_arr >= {tmid:.1f} s"
    # Continuum 式 TTL（只用前半段）
    ttl = np.zeros(n)
    if wl.startswith("cw") or wl.startswith("fi"):
        key = d["ttl_key"]
        dur = d["ttl_dur_s"]
        m_tr = train_req & (key != "") & np.isfinite(dur)
        med = {}
        for kk in np.unique(key[m_tr]):
            med[kk] = float(np.median(dur[m_tr & (key == kk)]))
        gmed = float(np.median(dur[m_tr]))
        for i in range(n):
            kk = key[i]
            ttl[i] = 0.0 if kk == "" else med.get(kk, gmed)
        meta["cont_ttl"] = f"per tool-combination median from first-half sessions; {len(med)} keys; global {gmed:.3f}s"
    else:
        gaps = []
        for i in range(n):
            j = nxt[i]
            if j >= 0 and d["t_arr"][j] < (0.5 * (d["t_arr"].min() + d["t_arr"].max())):
                gaps.append(d["t_arr"][j] - d["t_idle"][i])
        gmed = float(np.median(gaps))
        ttl[:] = gmed
        meta["cont_ttl"] = f"global median of first-half return gaps = {gmed:.3f}s (n={len(gaps)})"
    d["cont_ttl"] = ttl
    d["nb_old"] = (d["nold_tok"] // BLK).astype(np.int64)
    d["obj_blk"] = np.ceil(d["obj_tok"] / BLK).astype(np.int64)
    d["sh_blk"] = np.zeros(n, dtype=np.int64)
    if dedup:   # 敏感度（登記之後才加，只報告）：跨 session 共用的前綴只存一次、永遠在 GPU、不占任何層
        d["sh_blk"] = (d["shared_tok"] // BLK).astype(np.int64)
        d["obj_blk"] = np.maximum(np.ceil((d["obj_tok"] - d["sh_blk"] * BLK) / BLK), 1).astype(np.int64)
        meta["dedup_frac_tokens_shared"] = float(d["shared_tok"].sum() / d["ctx_tok"].sum())
    ret = d["ret_kind"] > 0
    meta["n_eval_returns"] = int((ret & d["eval"]).sum())
    meta["n_eval_sessions_with_return"] = int(len(np.unique(d["sid"][ret & d["eval"]])))
    return d, meta


def prefill_new(d, T, g):
    """新一輪 prefill：從 nb_old 個 block 之後一路算到 ctx（最後一個 block 照 token 比例）。"""
    fg = T[(g, "f")]
    cum = T[(g, "cum")]
    ctx = d["ctx_tok"].astype(np.int64)
    full = ctx // BLK
    rem = (ctx % BLK) / BLK
    nb = np.maximum(d["nb_old"], np.minimum(d["sh_blk"], full))
    return (cum[full] - cum[np.minimum(nb, full)]) + rem * fg[np.minimum(full, NMAX - 1)]


# ------------------------------------------------------------------ 模擬（放置軌跡）
class TierH:
    __slots__ = ("cap", "used", "heap")

    def __init__(self, cap):
        self.cap = cap
        self.used = 0
        self.heap = []


def simulate(d, policy, G_blk, C_blk, slow_cap, cost=None):
    """回傳每個請求到達時 session KV 的位置碼：0=沒有前一份/第一個請求、1=G、2=C、3=S、4=丟了、5=還在用（active）。
    以及評估段的釘住 GiB·s、慢層寫入 block 數。"""
    n = len(d["sid"])
    sid, t_arr, t_idle, nxt = d["sid"], d["t_arr"], d["t_idle"], d["nxt"]
    obj_blk, nb_old, cont = d["obj_blk"], d["nb_old"], d["cont_ttl"]
    evalm = d["eval"]
    t_eval0 = float(t_arr[evalm].min()) if evalm.any() else INF
    S = int(sid.max()) + 1
    loc = np.zeros(S, dtype=np.int8)        # 0=無、1=G、2=C、3=S、4=丟、5=active
    size = np.zeros(S, dtype=np.int64)
    tok = np.zeros(S, dtype=np.int64)       # 目前有效的 heap entry 編號
    tnext = np.full(S, INF)
    nbnext = np.zeros(S, dtype=np.int64)
    tidle_obj = np.zeros(S)
    tiers = {1: TierH(G_blk), 2: TierH(C_blk), 3: TierH(slow_cap)}
    out = np.zeros(n, dtype=np.int8)
    seq = [0]
    oracle = policy.startswith("O-")
    if policy in ("F-LRU", "P-Ret", "H-Turn2"):
        ttl_of = lambda i: INF
    elif policy.startswith("F-TTL-"):
        T0 = float(policy.split("-")[-1])
        ttl_of = lambda i: T0
    elif policy == "F-Cont":
        ttl_of = lambda i: float(cont[i])
    else:
        ttl_of = None
    stats = dict(pin_gib_s=0.0, slow_w_blk=0, t_last=None)

    def key_for(tier_id, s, t_i, ttl):
        if oracle:
            if policy == "O-Belady":
                return -tnext[s]
            # O-GDS：(距回來時間 × 大小) ÷ 留在這層比下一層省的時間；越大越先趕
            nb = int(nbnext[s])
            here = {1: 0.0, 2: cost["C"][nb], 3: cost["S"][nb]}[tier_id]
            below = {1: cost["C"][nb], 2: (cost["S"][nb] if slow_cap > 0 else cost["D"][nb]), 3: cost["D"][nb]}[tier_id]
            sav = below - here
            if sav <= 0:
                return -INF
            return -((tnext[s] - t_i) * size[s] / sav)
        if tier_id == 1:
            return t_i + ttl        # LRU（ttl=∞ 時用 t_idle 排）或 TTL 到期時間
        return t_i                  # C、S：LRU

    def insert(tier_id, s, t_i, ttl, now):
        while tier_id <= 3:
            tr = tiers[tier_id]
            if size[s] > tr.cap:
                tier_id += 1
                continue
            seq[0] += 1
            k = key_for(tier_id, s, t_i, ttl) if not (tier_id == 1 and ttl == INF and not oracle) else t_i
            heapq.heappush(tr.heap, (k, seq[0], s, t_i))
            tok[s] = seq[0]
            loc[s] = tier_id
            tr.used += size[s]
            if tier_id == 3 and now >= t_eval0:
                stats["slow_w_blk"] += int(size[s])
            # 擠出去
            while tr.used > tr.cap:
                kk, sq, v, tv = heapq.heappop(tr.heap)
                if tok[v] != sq or loc[v] != tier_id:
                    continue
                tr.used -= size[v]
                loc[v] = 0
                tok[v] = 0
                insert(tier_id + 1, v, tv, INF, now)
            return
        loc[s] = 4   # 沒有地方：丟

    def remove(s):
        L = int(loc[s])
        if L in (1, 2, 3):
            tiers[L].used -= size[s]
        tok[s] = 0

    def advance(t):
        tl = stats["t_last"]
        if tl is not None and t > tl:
            a = max(tl, t_eval0)
            if t > a:
                stats["pin_gib_s"] += tiers[1].used * GIB_PER_BLK * (t - a)
        stats["t_last"] = t

    # 事件：(時間, 優先, seq, 種類, 參數)；同時間 IDLE(0) → EXP(1) → ARR(2)
    ev = [(float(t_arr[i]), 2, i, "A", i) for i in range(n)]
    heapq.heapify(ev)
    cur_req = np.full(S, -1, dtype=np.int64)
    while ev:
        t, pr, sq, kind, i = heapq.heappop(ev)
        advance(t)
        if kind == "A":
            s = int(sid[i])
            out[i] = 0 if d["ret_kind"][i] == 0 else (loc[s] if loc[s] != 0 else 4)
            remove(s)
            loc[s] = 5
            cur_req[s] = i
            heapq.heappush(ev, (float(t_idle[i]), 0, i, "I", i))
        elif kind == "I":
            s = int(sid[i])
            if cur_req[s] != i or loc[s] != 5:
                continue                        # 已經有更新的請求
            size[s] = int(obj_blk[i])
            j = nxt[i]
            tnext[s] = float(t_arr[j]) if j >= 0 else INF
            nbnext[s] = int(nb_old[j]) if j >= 0 else 0
            tidle_obj[s] = float(t_idle[i])
            if (oracle or policy == "P-Ret") and j < 0:
                loc[s] = 4                      # 永遠不回來：直接丟
                continue
            if oracle:
                insert(1, s, float(t_idle[i]), 0.0, t)
            elif policy == "H-Turn2":
                ti = float(t_idle[i]) - (1e12 if d["ret_kind"][i] == 0 else 0.0)
                insert(1, s, ti, INF, t)
            else:
                ttl = ttl_of(i)
                if ttl <= 0:
                    insert(2, s, float(t_idle[i]), 0.0, t)
                else:
                    insert(1, s, float(t_idle[i]), ttl, t)
                    if ttl < INF and loc[s] == 1:
                        heapq.heappush(ev, (float(t_idle[i]) + ttl, 1, int(tok[s]), "E", s))
        else:  # TTL 到期
            s = i
            if loc[s] == 1 and tok[s] == sq:
                remove(s)
                loc[s] = 0
                insert(2, s, tidle_obj[s], 0.0, t)   # CPU 的 LRU 鍵＝最後一次使用（閒置開始），不是到期時間
    return out, stats


# ------------------------------------------------------------------ 一個 task＝(工作負載, 長度, G, C)
_MEMO = {}
H_LIST = [0, 1800, 3600]   # 0＝登記的評估；1800／3600＝事後加的穩健性（只評估離觀察窗結束 ≥H 秒的決定）


def cell_stats(res, m, d, wl, length, G, C, slow, g, H, wmeta):
    us, inv = np.unique(d["sid"][m], return_inverse=True)
    X = {p: res[p]["ttft"][m] for p in res}
    R = {p: res[p]["rest"][m].mean() for p in res}
    fx = min(FIXED, key=lambda p: X[p].mean())
    orc = min(ORACLES, key=lambda p: X[p].mean())
    mf, mo = X[fx].mean(), X[orc].mean()
    hr = (mf - mo) / mf
    rng = np.random.default_rng(12345)
    sf = np.bincount(inv, weights=X[fx], minlength=len(us))
    so = np.bincount(inv, weights=X[orc], minlength=len(us))
    B = 1000
    hb = np.empty(B)
    for b0 in range(0, B, 100):   # 分批，避免大 trace 的 (B × session) 矩陣太大
        idx = rng.integers(0, len(us), size=(min(100, B - b0), len(us)))
        hb[b0:b0 + len(idx)] = 1 - so[idx].sum(1) / sf[idx].sum(1)
    viol = [p for p in FIXED + PARTIAL + POSTHOC if X[p].mean() < mo - 1e-12]
    mp = X["P-Ret"].mean()
    mt = X["H-Turn2"].mean()
    return dict(workload=wl, length=length, G_GiB=G, C_GiB=C, slow=slow, gpu=g, H_s=H,
                best_fixed=fx, best_oracle=orc, mean_fixed_s=mf, mean_oracle_s=mo, headroom=hr,
                ci_lo=float(np.percentile(hb, 2.5)), ci_hi=float(np.percentile(hb, 97.5)),
                hr_vs_TTL0=(X["F-TTL-0"].mean() - mo) / X["F-TTL-0"].mean(),
                hr_vs_LRU=(X["F-LRU"].mean() - mo) / X["F-LRU"].mean(),
                hr_median=(np.median(X[fx]) - np.median(X[orc])) / np.median(X[fx]),
                hr_p90=(np.percentile(X[fx], 90) - np.percentile(X[orc], 90)) / np.percentile(X[fx], 90),
                hr_PRet_vs_fixed=(mf - mp) / mf,
                hr_restore_only=((R[fx] - R[orc]) / R[fx] if R[fx] > 0 else float("nan")),
                abs_gap_ms=(mf - mo) * 1e3,
                pin_gib_fixed=res[fx]["pin"], pin_gib_oracle=res[orc]["pin"],
                hr_vs_Turn2=(mt - mo) / mt, Turn2_gain_vs_fixed=(mf - mt) / mf,
                share_of_headroom_from_whether=((mf - mp) / (mf - mo) if mf > mo else float("nan")),
                n_ret=int(m.sum()), n_sess=len(us), oracle_violations="|".join(viol), cont_ttl=wmeta["cont_ttl"])


def rest_offset(code, sh, nb, g, slow, T, ell):
    """還原 block [sh, nb)（前 sh 個是共用前綴、已在 GPU）。"""
    if code in (0, 1, 5) or nb <= sh:
        return 0.0
    tier = {2: "cpu", 3: slow, 4: "D"}[code]
    if tier == "none":
        tier = "D"
    if tier == "D":
        c = T[(g, "cum")]
        return float(c[nb] - c[sh])
    k = (tier, g, sh, nb)
    if k not in _MEMO:
        _MEMO[k] = cake(nb - sh, ell[tier], T[(g, "f")][sh:nb])[0]
    return _MEMO[k]


def run_task(args):
    wl, length, G, C, trace_dir, outdir = args[:6]
    dedup = len(args) > 6 and args[6]
    t0 = time.time()
    f, ell, cmeta = load_costs()
    T = cost_tables(f, ell)
    d, wmeta = load_wl(trace_dir, wl, length, dedup)
    ret = (d["ret_kind"] > 0) & d["eval"]
    allm = d["eval"]
    sidr = d["sid"][ret]
    us, inv = np.unique(sidr, return_inverse=True)
    pre = {g: prefill_new(d, T, g) for g in GPUS}
    rows = []
    boot = []
    traj = {}
    for pol in FIXED + ["O-Belady"] + PARTIAL + POSTHOC:
        for sc in ("inf", "0"):
            traj[(pol, sc)] = simulate(d, pol, G * 16, C * 16, 10 ** 18 if sc == "inf" else 0)
    for slow in SLOWS:
        sc = "0" if slow == "none" else "inf"
        for g in GPUS:
            rc = {0: np.zeros(NMAX + 1), 1: np.zeros(NMAX + 1), 2: T[(g, "cpu")],
                  3: T[(g, slow)] if slow != "none" else T[(g, "D")], 4: T[(g, "D")], 5: np.zeros(NMAX + 1)}
            costd = dict(C=T[(g, "cpu")], S=(T[(g, slow)] if slow != "none" else T[(g, "D")]), D=T[(g, "D")])
            res = {}
            for pol in FIXED + ORACLES + PARTIAL + POSTHOC:
                if pol == "O-GDS":
                    o, st = simulate(d, pol, G * 16, C * 16, 10 ** 18 if sc == "inf" else 0, cost=costd)
                else:
                    o, st = traj[(pol, sc)]
                nb = np.minimum(d["nb_old"], NMAX)
                rest = np.zeros(len(o))
                if dedup:
                    sh = np.minimum(d["sh_blk"], nb)
                    for ii in np.nonzero(o > 1)[0]:
                        rest[ii] = rest_offset(int(o[ii]), int(sh[ii]), int(nb[ii]), g, slow, T, ell)
                else:
                    for code in range(6):
                        m = o == code
                        rest[m] = rc[code][nb[m]]
                ttft = rest + pre[g]
                x = ttft[ret]
                span = max(float(d["t_arr"][allm].max() - d["t_arr"][allm].min()), 1e-9)
                res[pol] = dict(ttft=ttft, rest=rest, pin=st["pin_gib_s"] / span)
                oc = o[ret]
                rows.append(dict(workload=wl + ("+dedup" if dedup else ""), length=length, G_GiB=G, C_GiB=C, slow=slow, gpu=g, policy=pol,
                                 n_ret=len(x), n_sess=len(us), mean_s=x.mean(), median_s=float(np.median(x)),
                                 p90_s=float(np.percentile(x, 90)), restore_mean_s=rest[ret].mean(),
                                 total_all_s=ttft[allm].sum(),
                                 frac_G=np.mean(oc == 1), frac_C=np.mean(oc == 2), frac_S=np.mean(oc == 3),
                                 frac_D=np.mean(oc == 4), frac_active=np.mean(oc == 5),
                                 pin_gib_avg=st["pin_gib_s"] / span, slow_write_GiB=st["slow_w_blk"] * GIB_PER_BLK,
                                 k_scale=wmeta["k_scale"], frac_clipped=wmeta["frac_clipped"]))
            for H in H_LIST:
                mH = ret & ((d["t_end"] - d["t_dec"]) >= H) if H > 0 else ret
                if mH.sum() == 0:
                    boot.append(dict(workload=wl + ("+dedup" if dedup else ""), length=length, G_GiB=G, C_GiB=C,
                                     slow=slow, gpu=g, H_s=H, n_ret=0, n_sess=0, cont_ttl=wmeta["cont_ttl"]))
                    continue
                boot.append(cell_stats(res, mH, d, wl + ("+dedup" if dedup else ""), length, G, C, slow, g, H, wmeta))
    return rows, boot, dict(wl=wl, length=length, G=G, C=C, sec=time.time() - t0, wmeta=wmeta, cmeta=cmeta)


def write_csv(path, rows):
    if not rows:
        return
    keys = ["run_id", "ts"] + list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        r0, t0 = rid(), now_ts()
        for r in rows:
            r = {k: (round(float(v), 6) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}
            w.writerow(dict(run_id=r0, ts=t0, **r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--rawdir", required=True)
    ap.add_argument("--wl", nargs="+", default=["mc_conv", "mc_tool", "bl_coder", "cw_orig", "cw_stack"])
    ap.add_argument("--lens", nargs="+", default=LENS)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--dedup", action="store_true", help="敏感度：共用前綴只存一次（只報告）")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    f, ell, cmeta = load_costs()
    worst = selftest(f, ell)
    print("selftest vs m7_sim.sim_restore: PASS, max |dt| =", worst, flush=True)
    print("costs", json.dumps(cmeta, default=str), flush=True)
    T = cost_tables(f, ell)
    for nb in (16, 64, 128, 256):
        print(f"restore n={nb}: " + ", ".join(f"{g}/{k}={T[(g, k)][nb]:.3f}s" for g in GPUS
                                            for k in ("cpu", "local", "nfs", "D")), flush=True)
    tasks = [(w, L, G, C, a.traces, a.rawdir, a.dedup) for w in a.wl for L in a.lens for G in G_LIST for C in C_LIST]
    os.makedirs(a.outdir, exist_ok=True)
    os.makedirs(a.rawdir, exist_ok=True)
    allrows, allboot, metas = [], [], []
    with Pool(a.procs) as pool:
        for rows, boot, m in pool.imap_unordered(run_task, tasks):
            allrows += rows
            allboot += boot
            metas.append(m)
            print(f"done {m['wl']} {m['length']} G={m['G']} C={m['C']} {m['sec']:.1f}s", flush=True)
    write_csv(os.path.join(a.outdir, f"x8_cells{a.tag}.csv"), allrows)
    write_csv(os.path.join(a.outdir, f"x8_headroom{a.tag}.csv"), allboot)
    json.dump(metas, open(os.path.join(a.rawdir, "task_meta.json"), "w"), default=str, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
