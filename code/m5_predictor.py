#!/usr/bin/env python3
"""Milestone 5 第二階段 — 未來效用預測器（實作 + 訓練，PoC）。

`EXPERIMENT_PLAN.md` §6 把 M5 切成三步，這支腳本是**第二步**：

    1. 成本模型 + recency（不訓練）
    2. **未來效用預測器（GBDT，main.tex §5.2）**  ← 本檔
    3. 成本敏感損失的比較（main.tex §5.3）        ← 本檔的 (B) 區塊

第 3 步在這裡一起做，因為「訓練一個預測器」與「用哪個損失訓練」共用同一條
資料管線，分兩支腳本只會讓兩邊的特徵定義飄掉。

## 這支腳本做什麼

    trace ──重放──▶ 每次存取的特徵 ──兩個標記機制──▶ 樣本
                                            │
                                            ├─▶ 對稱 L2   (baseline)
                                            └─▶ 成本加權 L2 (main.tex 式 10)
                                                    │
                                              isotonic 校準
                                                    │
                                          依式 (9) 的門檻 p* 決策
                                                    │
                                    校準品質 (ECE) + 成本加權錯誤 (ms)

## 🔴 這是 trace 驅動的 PoC，不是端到端系統

論文 §5.2 的特徵有六族，這裡只做得到三族：

| 特徵族 | 本檔 | 為什麼 |
|---|---|---|
| deltas（前 k 次存取間隔） | ✅ | trace 有 |
| EDC（10 個指數衰減計數器） | ✅ | trace 有 |
| 靜態（位置、請求長度…） | 🟡 部分 | Mooncake 沒有 `token_type`，`layer_id`/`head_id` 在 trace 層級不存在 |
| pooled key/value 統計 | ❌ NOT_AVAILABLE | 要跑模型才有 |
| `attn_mass`（近 8 步的注意力質量） | ❌ NOT_AVAILABLE | 要跑模型才有 |

**所以本檔量到的是「僅存取歷史」那一組的下界**，不是論文 §5.2 的完整特徵集。
表 15 的 (C) 區塊（特徵集消融）**做不到**，不要拿這裡的數字去填它。

## 🔴 時鐘的單位

虛擬時間 t = **全域存取計數**（每存取一個 block 加 1），與 LRB 相同的作法。
EDC 的半衰期 $2^9$–$2^{18}$ 在這條 trace 上等於 1–500 個請求
（toolagent 平均 538 次存取／請求），涵蓋範圍合理。
標籤 $\\tau$ 與視窗 $W$ 都用同一個時鐘，單位是「次存取」。

## 🔴 門檻 p* 是怎麼從實測常數算出來的

式 (9) 是 $p^{*} = 1/(1+\\kappa)$，$\\kappa = c_{FN}/c_{FP}$。把它實例化到本設定：

* 判斷錯誤而**丟掉**了會被重用的 block：多付 $C_{drop}(pos) - C_{cpu}$（重算 vs 取回）
* 判斷錯誤而**留下**了不會被重用的 block：佔掉一個 CPU 槽位，其機會成本
  即該槽位對別人的價值 $C_{cpu}$

於是 $\\kappa(pos) = (C_{drop}(pos) - C_{cpu}) / C_{cpu}$，

    p*_cpu(pos) = C_cpu / C_drop(pos)
    p*_ssd(pos) = C_ssd / C_drop(pos)

兩件事立刻掉出來，且**都與已量到的結果一致**：

1. 門檻隨 block 的絕對位置變動（`llama-bf16`：0.147 @ pos 0 → 0.019 @ pos 126K），
   全部遠低於 0.5——這正是 §5.3 說的「門檻被錯置了一個數量級」。
2. `p*_ssd > 1` 時代表 SSD 這一階**在數學上不可選**（丟掉重算比放硬碟便宜），
   其邊界恰為 $C_{ssd} = C_{drop}(pos)$，即成本模型算出的交叉點 $P^{*}$。

**這不是新的假設，是把已量到的常數代進式 (9)。**

## 用法

    # 一次做完（特徵 + 訓練 + 評估）
    python code/m5_predictor.py all --trace toolagent --model qwen-awq --device nvme

    # 只重跑訓練（吃上一次的特徵）
    python code/m5_predictor.py train --trace toolagent --model qwen-awq
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, os.environ.get("PAPER_HKV_PYLIBS",
                                  "/ssd7/hungwei/paper-hkv/pylibs"))

from m4_invariants import check_trace_units                      # noqa: E402
from m4_oracle import (BLOCK, MODEL_PROFILES, CostModel,         # noqa: E402
                       load_cost_model, longctx_trace, mooncake_outputs,
                       mooncake_trace, profile, reuse_rate)

REPO = Path(__file__).resolve().parent.parent
BIG = Path(os.environ.get("PAPER_HKV_BIG", "/ssd7/hungwei/paper-hkv"))
OUT = REPO / "results/m5_predictor"
INDEX = OUT / "features_index.json"

# EDC 的十個半衰期（存取數）。main.tex 演算法 1：C_i <- 1 + C_i * 2^(-Δ1/2^(9+i))
EDC_N = 10
EDC_HALFLIVES = np.array([2.0 ** (9 + i) for i in range(EDC_N)], dtype=np.float64)


# ────────────────────── 長上下文工作負載 ──────────────────────

def session_trace(doc_blocks: int, turns: int, n_sessions: int,
                  tail_blocks: int, seed: int = 1234, long_frac: float = 0.5,
                  long_turns: int = 12, concurrency: int = 4) -> list[list[int]]:
    r"""長上下文的**多輪會談**：一份長文件被同一個 session 連續查 `turns` 次，
    每次附上不同的尾巴（新的問題）。

    🔴 為什麼不用 `m4_oracle.longctx_trace`（Zipf 抽文件）：
    128K 級的請求下，GPU 預算（qwen-awq 273,872 token = 17,117 blocks）只裝得下
    **約兩個請求**。Zipf 抽樣把同一份文件的兩次查詢隔開幾十上百個請求，
    重用距離遠超過快取壽命，於是**正樣本率是 0**——量不到任何東西。

    真實世界裡 128K 請求會發生重用的形態就是**多輪對話**：同一份長文件連續被追問。
    所以重用必須發生在**相鄰請求之間**，這個產生器就照那個形狀造。

    副作用（正是本實驗要的）：重用發生在文件的**所有位置**上（0 到 doc_tokens），
    於是高 $\kappa$ 區（位置 32K–128K，$\kappa$ 37–60）**第一次有正樣本**。
    Mooncake 上 8K 以上的位置一個正樣本都沒有，(B) 區塊因此無從檢驗。

    🔴 **會談長度必須是隨機的，否則問題是假的。** 第一版把 turns 寫死成 3，
    於是「n_acc == 2」就完全等價於「不會再被用到」——AUC 1.0、Bayes 誤差 0，
    量不出任何與損失函數有關的東西。這裡改成**雙峰混合**：
    `long_frac` 的會談是長的（平均 `long_turns` 輪），其餘是短的（1–3 輪）。
    於是 (i) `n_acc` 仍然有訊息（看過五次 -> 多半是長會談），
    (ii) 但早期的每一次決策都**真的有不確定性**——這正是式 (9) 的門檻
    與成本敏感損失唯一會產生差別的地方。

    ⚠️ 這是合成流量。回報時必須標 `workload=longctx-session` 與所用的參數。
    """
    rng = random.Random(seed)
    out: list[list[int]] = []
    nxt = 0
    started = 0

    def new_session():
        nonlocal nxt, started
        base, nxt = nxt, nxt + doc_blocks
        k = (rng.randint(max(1, long_turns // 2), long_turns * 3 // 2)
             if rng.random() < long_frac else rng.randint(1, max(1, turns)))
        started += 1
        return {"base": base, "left": k}

    # 🔴 `concurrency` 個會談**交錯**進行，不是一個做完再做下一個。
    #    序列化的版本讓每次重用的距離都恰好是「一個請求」，
    #    於是 τ 幾乎是常數、排序品質無從量起（Spearman 變成 nan），
    #    而且快取壓力也不對——真實服務同時有多個 session 在跑。
    #    交錯之後重用距離 ≈ concurrency 個請求，壓力由
    #    concurrency × doc_blocks / GPU 容量決定，成為一條可掃的軸。
    active = [new_session() for _ in range(max(1, concurrency))]
    while active:
        s_ = active[rng.randrange(len(active))]
        tail = [nxt + k for k in range(tail_blocks)]
        nxt += tail_blocks
        out.append([s_["base"] + b for b in range(doc_blocks)] + tail)
        s_["left"] -= 1
        if s_["left"] <= 0:
            active.remove(s_)
            if started < n_sessions:
                active.append(new_session())
    return out


# 🔴 每個工作負載回報的參數不同，但 `samples.csv` 只能有一組欄位。
#    不固定 schema 的話，跑完 Mooncake 再跑合成負載就會被 write_csv 的守門擋下
#    （而那個守門是對的：欄位變動的 append 會整排錯位）。
WORKLOAD_FIELDS = ("workload", "doc_tokens", "n_docs", "requests_arg",
                   "tail_frac", "alpha", "turns", "sessions", "long_frac",
                   "long_turns", "concurrency")


def _wmeta(**kw) -> dict:
    return {k: kw.get(k, "") for k in WORKLOAD_FIELDS}


def load_workload(a) -> tuple[list[list[int]], list[int] | None, dict]:
    """回傳 (trace, outputs, meta)。outputs 用於 decode 成本。"""
    if a.workload == "mooncake":
        tr = mooncake_trace(a.trace, limit=a.limit_requests)
        return tr, mooncake_outputs(a.trace), _wmeta(workload="mooncake")
    doc_b = max(1, a.doc_tokens // BLOCK)
    if a.workload == "longctx-zipf":
        # 🔴 直接重現論文 §6.8 那個 headroom 最高的點（`surface:L131072:rq10`）：
        #    qwen-awq / NVMe / prefill-only / 重用 89.06% / 壓力 6.41× /
        #    最佳 baseline = tier_fs / **headroom 34.65%**
        #    （`results/m4_oracle/qwen-awq-surface-prefill/headroom_surface.csv`）。
        #    在 Mooncake 上量預測器等於在 headroom 只有 8% 的地方量——
        #    那裡就算策略完美也拿不到什麼。方法該在它宣稱有用的區間被檢驗。
        tail_b = max(1, int(doc_b * a.tail_frac))
        tr = longctx_trace(a.n_docs, doc_b, tail_b, a.requests, a.alpha, a.seed)
        pool = mooncake_outputs("conversation")
        rr = random.Random(a.seed)
        outs = [pool[rr.randrange(len(pool))] for _ in range(len(tr))]
        return tr, outs, _wmeta(workload="longctx-zipf",
                                doc_tokens=a.doc_tokens, n_docs=a.n_docs,
                                requests_arg=a.requests, tail_frac=a.tail_frac,
                                alpha=a.alpha)
    tail_b = max(1, int(doc_b * a.tail_frac))
    tr = session_trace(doc_b, a.turns, a.sessions, tail_b, a.seed,
                       long_frac=a.long_frac, long_turns=a.long_turns,
                       concurrency=a.concurrency)
    # 合成請求的輸出長度：自真實 Mooncake 抽樣，「decode 佔多少」才不是編的
    pool = mooncake_outputs("conversation")
    rr = random.Random(a.seed)
    outs = [pool[rr.randrange(len(pool))] for _ in range(len(tr))]
    return tr, outs, _wmeta(workload="longctx-session", doc_tokens=a.doc_tokens,
                            turns=a.turns, sessions=a.sessions,
                            tail_frac=a.tail_frac, long_frac=a.long_frac,
                            long_turns=a.long_turns, concurrency=a.concurrency)


# ────────────────────────── 特徵 ──────────────────────────

def feature_names(k_deltas: int) -> list[str]:
    return (["n_acc", "log_age_since_first"]
            + [f"log_delta_{i + 1}" for i in range(k_deltas)]
            + [f"edc_{i}" for i in range(EDC_N)]
            + ["pos_tokens", "pos_frac", "req_len_tokens",
               "last_pos_tokens", "pos_delta_tokens", "req_gap"])


FEATURE_GROUPS = {
    "history": ("n_acc", "log_age_since_first"),
    "deltas": ("log_delta_",),
    "edc": ("edc_",),
    "static": ("pos_tokens", "pos_frac", "req_len_tokens", "last_pos_tokens",
               "pos_delta_tokens", "req_gap"),
}


def group_columns(names: list[str], groups: list[str]) -> list[int]:
    """回傳所選特徵族對應的欄位索引。用於 (C) 區塊的**簡化版**消融——
    只涵蓋存取歷史那三族，`pooled KV` 與 `attn_mass` 需要引擎，此處沒有。"""
    keep = []
    for i, n in enumerate(names):
        for g in groups:
            if any(n == pre or n.startswith(pre)
                   for pre in FEATURE_GROUPS[g]):
                keep.append(i)
                break
    if not keep:
        raise SystemExit(f"🔴 特徵族 {groups} 選不到任何欄位")
    return keep


class Replay:
    """單趟重放 trace，對**每一次存取**產生「決策當下看得到的」特徵。

    🔴 特徵一律取自更新狀態**之前**。寫反了就是把「這次存取」本身
       洩漏進特徵裡（delta_1 會變成 0，模型立刻學會「delta_1==0 => 一定會用」），
       而且準確率會漂亮得看不出有問題。`test_m5_predictor.py` 有對應的測試。
    """

    def __init__(self, trace: list[list[int]], k_deltas: int = 16):
        self.trace = trace
        self.k = k_deltas
        n = max(max(r) for r in trace) + 1
        self.n_blocks = n
        self.last_t = np.full(n, -1, dtype=np.int64)
        self.first_t = np.full(n, -1, dtype=np.int64)
        self.last_ri = np.full(n, -1, dtype=np.int64)
        self.n_acc = np.zeros(n, dtype=np.int32)
        self.last_pos = np.zeros(n, dtype=np.int32)
        self.deltas = np.full((n, k_deltas), np.nan, dtype=np.float32)
        self.edc = np.zeros((n, EDC_N), dtype=np.float32)
        self.n_feat = len(feature_names(k_deltas))

    def run(self, on_chunk, chunk: int = 200_000) -> int:
        """重放全部存取。每累積 `chunk` 列就呼叫一次 `on_chunk(X, meta, t0)`。

        meta 欄位：block, t, req_index, pos_tokens, req_len_tokens
        """
        k, nf = self.k, self.n_feat
        X = np.empty((chunk, nf), dtype=np.float32)
        meta = np.empty((chunk, 5), dtype=np.int64)
        fill = 0
        t = 0
        t0 = 0
        hl = EDC_HALFLIVES
        for ri, req in enumerate(self.trace):
            rl = len(req) * BLOCK
            for pi, b in enumerate(req):
                pos = pi * BLOCK
                lt = self.last_t[b]
                # ---- 特徵（狀態更新之前） ----
                row = X[fill]
                na = self.n_acc[b]
                row[0] = na
                row[1] = math.log1p(t - self.first_t[b]) if na else np.nan
                if na:
                    d1 = t - lt
                    row[2] = math.log1p(d1)
                    row[3:2 + k] = self.deltas[b, :k - 1]
                    row[2 + k:2 + k + EDC_N] = self.edc[b]
                else:
                    d1 = 0
                    row[2:2 + k] = np.nan
                    row[2 + k:2 + k + EDC_N] = 0.0
                j = 2 + k + EDC_N
                row[j] = pos
                row[j + 1] = pos / rl
                row[j + 2] = rl
                row[j + 3] = self.last_pos[b] if na else np.nan
                row[j + 4] = pos - self.last_pos[b] if na else np.nan
                row[j + 5] = ri - self.last_ri[b] if na else np.nan
                meta[fill] = (b, t, ri, pos, rl)
                fill += 1
                # ---- 狀態更新 ----
                if na:
                    self.deltas[b, 1:] = self.deltas[b, :-1]
                    self.deltas[b, 0] = math.log1p(d1)
                    self.edc[b] = 1.0 + self.edc[b] * np.exp2(-d1 / hl)
                else:
                    self.first_t[b] = t
                    self.edc[b] = 1.0
                self.last_t[b] = t
                self.last_ri[b] = ri
                self.last_pos[b] = pos
                self.n_acc[b] = na + 1
                t += 1
                if fill == chunk:
                    on_chunk(X[:fill], meta[:fill], t0)
                    t0 = t
                    fill = 0
        if fill:
            on_chunk(X[:fill], meta[:fill], t0)
        return t


# ─────────────────────── 標籤（兩個機制） ───────────────────────

class Labeler:
    """待標記佇列。**負樣本只能由機制 (b) 取得**（main.tex §5.2）。

    (a) 該 block 下次被存取時，以兩次存取的間隔為標籤；
    (b) 樣本離開長度為 W 的滑動視窗仍未被存取 -> 立即標記為「> W」。

    只有 (a) 的話，訓練集裡不會有任何「後來沒被用到」的 block，
    而那正是 Drop 這個動作的目標群體。
    """

    def __init__(self, window: int):
        self.W = window
        self.pending: dict[int, list[int]] = defaultdict(list)  # block -> sample ids
        self.queue: deque = deque()          # (t_sample, block, sample_id)
        self.t_of: dict[int, int] = {}       # sample_id -> 記錄特徵的時刻
        self.tau = {}                        # sample_id -> tau
        self.censored = {}                   # sample_id -> 0/1
        self.n_a = 0
        self.n_b = 0

    def add(self, sid: int, block: int, t: int) -> None:
        self.t_of[sid] = t
        self.pending[block].append(sid)
        self.queue.append((t, block, sid))

    def on_access(self, block: int, t: int) -> None:
        """機制 (a)：這個 block 又被用到了，把它所有待標記樣本結案。"""
        lst = self.pending.pop(block, None)
        if not lst:
            return
        for sid in lst:
            if sid in self.tau:              # 已被機制 (b) 標掉
                continue
            self.tau[sid] = t - self.t_of.pop(sid)
            self.censored[sid] = 0
            self.n_a += 1

    def expire(self, t: int) -> None:
        """機制 (b)：把離開視窗的樣本標成「> W」。"""
        W = self.W
        while self.queue and t - self.queue[0][0] > W:
            ts, block, sid = self.queue.popleft()
            if sid in self.tau:
                continue
            self.tau[sid] = W + 1
            self.censored[sid] = 1
            self.t_of.pop(sid, None)
            self.n_b += 1
            lst = self.pending.get(block)
            if lst:
                try:
                    lst.remove(sid)
                except ValueError:
                    pass
                if not lst:
                    self.pending.pop(block, None)


def label_only(meta: np.ndarray, window: int, sample_rate: float,
               max_per_block: int, out_dir: Path, seed: int,
               uncensored: bool = False) -> dict:
    r"""不重放、不算特徵，只用既有的 `meta.npy` 重新產生一組標籤。

    敏感度掃描要跑十幾組 (W, 取樣率, seed)，而**特徵完全不受這三者影響**——
    只有「收哪些樣本」與「標籤怎麼算」會變。重放一次 12.7M 次存取要兩分鐘，
    只算標籤三十秒。這支函式讓整個掃描從一小時降到十分鐘。

    `uncensored=True`：標籤改用**真實的下次使用時刻**（沒有再被用到就是 $\infty$，
    以 `n_accesses` 當哨兵），不在 $W$ 處設限。
    這一版是為了修「排序」：設限會把所有「$W$ 之外」的 block 壓成同分，
    而策略的逐出順序吃的就是那個排序。
    **部署上的代價要講清楚**：這種標籤要等到該 block 真的再被用到（或確定不會），
    延遲遠大於 $W$；線上系統要嘛用很長的 $W$，要嘛接受標籤更慢。
    $\hat p$ 的目標仍是 $\Pr[\tau \le W]$，所以門檻那一側不受影響。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks = np.asarray(meta[:, 0])
    total = len(blocks)
    rng = np.random.default_rng(seed)
    take = rng.random(total) < sample_rate
    if max_per_block:
        seen: dict[int, int] = {}
        for i in np.nonzero(take)[0]:
            b = int(blocks[i])
            j = seen.get(b, 0) + 1
            seen[b] = j
            if not (j <= max_per_block or rng.random() < max_per_block / j):
                take[i] = False
    sids = np.nonzero(take)[0].astype(np.int64)
    # 真實的下次使用時刻（單趟反向掃描）
    nxt = np.full(total, np.inf)
    last: dict[int, int] = {}
    for t in range(total - 1, -1, -1):
        b = int(blocks[t])
        nxt[t] = last.get(b, np.inf)
        last[b] = t
    gap = nxt[sids] - sids
    keep = (sids + window) < total            # 視窗完整可觀測
    sids, gap = sids[keep], gap[keep]
    cen = (gap > window).astype(np.int8)
    if uncensored:
        tau = np.where(np.isinf(gap), float(total), gap)
    else:
        tau = np.where(cen == 1, float(window + 1), gap)
    np.savez(out_dir / "labels.npz", sample_ids=sids, tau=tau, censored=cen)
    return {"n_accesses": total, "n_samples": int(len(sids)),
            "n_dropped_tail": int((~keep).sum()),
            "positive_rate": round(float((cen == 0).mean()), 4),
            "n_labeled_by_next_access": int((cen == 0).sum()),
            "n_labeled_by_window": int((cen == 1).sum()),
            "censored_pct": round(100.0 * float(cen.mean()), 3),
            "window_accesses": window, "sample_rate": sample_rate,
            "max_per_block": max_per_block, "label_mode":
            "uncensored" if uncensored else "censored", "seconds": 0.0}


def build_samples(rep: Replay, window: int, sample_rate: float,
                  max_per_block: int, out_dir: Path, seed: int = 1234) -> dict:
    """重放一次，寫出 (1) 每次存取的特徵矩陣 (2) 被抽樣的樣本與其標籤。

    取樣**以 block 為單位、每次存取獨立抽**（main.tex §5.2：
    「取樣以 block 為單位而非以請求為單位，以避免偏向熱門 block」）——
    即對每一次 block 存取以固定機率 `sample_rate` 收樣本，
    而不是「挑一個請求就把它整串 block 收進來」（那才會偏向熱門請求）。

    🔴 這裡踩過一次坑，值得記下來：第一版把上句讀成「每個 block 最多收 C 個樣本」
       （對第 j 次存取以機率 min(1, C/j) 收），結果正樣本率從 35% 掉到 **0.3%**，
       因為重用幾乎全部集中在少數極熱的 block 上，而那個上限正好把它們削掉 600 倍。
       更關鍵的是**它與策略實際面對的分布不一致**：驅逐決策是逐次存取發生的
       （每次存取都會重新 admit，因而每次存取最終都對應一次驅逐決策），
       所以訓練分布必須是逐存取的。
       抓到的方法是拿它跟另一條路徑的數字對——M4 實測 `cpu_lru` 的 GPU 命中率
       是 34.9%，而「下次存取距今 ≤ W」的比例是 34.8%，兩者必須一致。

    `max_per_block > 0` 時才額外套用每個 block 的取樣上限（診斷用，非預設）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_acc_total = sum(len(r) for r in rep.trace)
    nf = rep.n_feat
    Xmm = np.lib.format.open_memmap(out_dir / "X.npy", mode="w+",
                                    dtype=np.float32, shape=(n_acc_total, nf))
    Mmm = np.lib.format.open_memmap(out_dir / "meta.npy", mode="w+",
                                    dtype=np.int64, shape=(n_acc_total, 5))
    rng = np.random.default_rng(seed)
    lab = Labeler(window)
    seen = np.zeros(rep.n_blocks, dtype=np.int32)
    sample_ids: list[int] = []               # 全域存取序號（= X 的列號）
    t_start = time.time()

    def on_chunk(X, meta, t0):
        Xmm[t0:t0 + len(X)] = X
        Mmm[t0:t0 + len(X)] = meta
        blocks = meta[:, 0]
        ts = meta[:, 1]
        for i in range(len(X)):
            b = int(blocks[i])
            t = int(ts[i])
            lab.expire(t)
            lab.on_access(b, t)              # 先結案，再決定要不要收新樣本
            take = rng.random() < sample_rate
            if take and max_per_block:
                seen[b] += 1
                j = seen[b]
                take = j <= max_per_block or rng.random() < max_per_block / j
            if take:
                sid = t0 + i
                lab.add(sid, b, t)
                sample_ids.append(sid)

    total = rep.run(on_chunk)
    lab.expire(total + window + 1)           # 清空佇列（這批樣本下面會被丟掉）
    Xmm.flush(); Mmm.flush()

    sids_all = np.array(sorted(sample_ids), dtype=np.int64)
    # 🔴 只保留「視窗完整可觀測」的樣本：t + W < 總存取數。
    #    trace 的最後 W 次存取裡記下的樣本，其「> W」是**資料截斷的產物**，
    #    不是觀測到的事實。不擋掉的話，測試集（時序切分下就是尾巴那段）
    #    會被灌進一整批假的負樣本——冒煙測試裡正樣本率因此只剩 1.3%，
    #    而 AUC 看起來是 0.9998。指標會漂亮，但量的是資料截斷。
    keep = (sids_all + window) < total
    sids = sids_all[keep]
    n_tail = int((~keep).sum())
    tau = np.array([lab.tau[int(s)] for s in sids], dtype=np.float64)
    cen = np.array([lab.censored[int(s)] for s in sids], dtype=np.int8)
    np.savez(out_dir / "labels.npz", sample_ids=sids, tau=tau, censored=cen)
    return {
        "n_accesses": total,
        "n_samples": int(len(sids)),
        "n_dropped_tail": n_tail,
        "positive_rate": round(float((cen == 0).mean()), 4),
        "n_labeled_by_next_access": lab.n_a,
        "n_labeled_by_window": lab.n_b,
        "censored_pct": round(100.0 * float(cen.mean()), 3),
        "window_accesses": window,
        "sample_rate": sample_rate,
        "max_per_block": max_per_block,
        "seconds": round(time.time() - t_start, 1),
    }


# ─────────────────────── isotonic 校準（PAVA） ───────────────────────

def isotonic_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """加權 PAVA。回傳 (x 的節點, 對應的單調值)。

    自行實作而不裝 scikit-learn：校準是本檔唯一需要的統計工具，
    PAVA 只有二十行，而多裝一個相依就多一個版本要記進附錄。
    """
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order]
    # 相同 x 先合併成一個帶權點
    ux, start = np.unique(xs, return_index=True)
    sums = np.add.reduceat(ys, start)
    cnts = np.diff(np.append(start, len(ys))).astype(np.float64)
    vals, wts, npts = [], [], []
    for v, w in zip(sums / cnts, cnts):
        vals.append(v); wts.append(w); npts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, n2 = vals.pop(), wts.pop(), npts.pop()
            v1, w1, n1 = vals.pop(), wts.pop(), npts.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wts.append(w1 + w2)
            npts.append(n1 + n2)
    out = np.empty(len(ux))
    i = 0
    for v, n in zip(vals, npts):
        out[i:i + n] = v
        i += n
    return ux, out


def isotonic_predict(xk: np.ndarray, yk: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.clip(np.interp(x, xk, yk), 0.0, 1.0)


# ─────────────────────── 成本與門檻 ───────────────────────

def drop_cost(cm: CostModel, pos_tokens: np.ndarray) -> np.ndarray:
    return cm.recompute_base + cm.recompute_slope_per_token * pos_tokens


def p_star(cm: CostModel, pos_tokens: np.ndarray, tier: str = "cpu") -> np.ndarray:
    """式 (9) 在本設定的實例化。見檔頭〈門檻 p* 是怎麼算出來的〉。"""
    c = cm.cpu if tier == "cpu" else cm.ssd
    return c / drop_cost(cm, pos_tokens)


def cost_weighted_error(p_hat: np.ndarray, y: np.ndarray, thr: np.ndarray,
                        cm: CostModel, pos: np.ndarray) -> dict:
    """成本加權錯誤率（main.tex §B.6 的兩個診斷之一）：錯掉多少毫秒。

    決策：p_hat < thr -> Drop，否則留在 CPU。
    y=1（W 內會被用到）而丟掉 -> 多付 C_drop(pos) - C_cpu
    y=0（不會被用到）而留著 -> 白佔一個槽位，記 C_cpu
    """
    dc = drop_cost(cm, pos)
    dropped = p_hat < thr
    fn = dropped & (y == 1)
    fp = (~dropped) & (y == 0)
    ms_fn = float(np.sum(dc[fn] - cm.cpu))
    ms_fp = float(cm.cpu * np.count_nonzero(fp))
    n = len(y)
    return {
        "n": n,
        "drop_rate": round(float(dropped.mean()), 4),
        "err_rate": round(float((fn | fp).mean()), 4),
        "fn_rate": round(float(fn.mean()), 4),
        "fp_rate": round(float(fp.mean()), 4),
        "cost_ms": round(ms_fn + ms_fp, 2),
        "cost_ms_fn": round(ms_fn, 2),
        "cost_ms_fp": round(ms_fp, 2),
        "cost_us_per_decision": round(1000.0 * (ms_fn + ms_fp) / n, 3),
    }


POS_BINS = [0, 4096, 8192, 16384, 32768, 65536, 131072, 10 ** 9]


def cost_by_position(p_hat, y, cm, pos, rule: str) -> list[dict]:
    r"""依 block 的絕對位置分箱回報成本加權錯誤。

    §5.3 給了一個可證偽的預測：**加權訓練與「對稱訓練 + 移門檻」的差距
    應隨 κ 增大而擴大**。論文打算靠換平台（3090 vs MI300X）來變動 κ，
    但 κ 在**同一台機器上**就已經隨 block 位置變動了——
    $\kappa(pos) = (C_{drop}(pos) - C_{cpu}) / C_{cpu}$，
    在 qwen-awq/NVMe 上由 pos 0 的 ~11 增到 pos 128K 的 ~89（8 倍）。
    所以這個預測**不必等平台 B 就能先檢驗一次**。
    """
    rows = []
    for lo, hi in zip(POS_BINS[:-1], POS_BINS[1:]):
        m = (pos >= lo) & (pos < hi)
        if not m.any():
            continue
        thr = (np.full(int(m.sum()), 0.5) if rule == "0.5"
               else p_star(cm, pos[m], "cpu"))
        r = cost_weighted_error(p_hat[m], y[m], thr, cm, pos[m])
        kappa = (drop_cost(cm, pos[m]) - cm.cpu) / cm.cpu
        rows.append({"pos_lo": lo, "pos_hi": hi,
                     "kappa_median": round(float(np.median(kappa)), 2),
                     "p_star_median": round(float(np.median(
                         p_star(cm, pos[m], "cpu"))), 4),
                     "positive_rate": round(float(y[m].mean()), 4), **r})
    return rows


def ece(p_hat: np.ndarray, y: np.ndarray, bins: int = 15) -> tuple[float, list]:
    """期望校準誤差 + 可靠度表。**用表不用圖**（同一批數字只呈現一次）。"""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p_hat, edges[1:-1]), 0, bins - 1)
    rows, tot = [], 0.0
    for b in range(bins):
        m = idx == b
        n = int(m.sum())
        if not n:
            continue
        conf = float(p_hat[m].mean())
        acc = float(y[m].mean())
        tot += n * abs(conf - acc)
        rows.append({"bin_lo": round(float(edges[b]), 4),
                     "bin_hi": round(float(edges[b + 1]), 4),
                     "n": n, "mean_p_hat": round(conf, 4),
                     "empirical_rate": round(acc, 4),
                     "gap": round(acc - conf, 4)})
    return tot / len(p_hat), rows


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman 等級相關。自行實作，不為了一個統計量多裝 scipy。"""
    if len(a) < 2:
        return float("nan")

    def rank(x):
        o = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=np.float64)
        s = x[o]
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[j + 1] == s[i]:
                j += 1
            r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return r

    ra, rb = rank(a), rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    d = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d else float("nan")


def auc(p_hat: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney U 形式的 AUC（不依賴 sklearn）。"""
    order = np.argsort(p_hat, kind="mergesort")
    ranks = np.empty(len(p_hat), dtype=np.float64)
    sp = p_hat[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1 = float(np.count_nonzero(y == 1))
    n0 = float(len(y) - n1)
    if n0 == 0 or n1 == 0:
        return float("nan")
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# ─────────────────────── 訓練 ───────────────────────

def time_split(sids: np.ndarray, tau: np.ndarray, censored: np.ndarray,
               window: int, train_frac: float, calib_frac: float,
               embargo: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """依 trace 的**時間順序**切分（main.tex §5.2：隨機切分會洩漏未來）。

    另加一道 embargo：一個樣本在 $t$ 記錄，但它的標籤要到
    $t + \\min(\\tau, W)$ 才確定。若訓練集裡含有「標籤在測試期才確定」的樣本，
    等於用測試期的資訊訓練——這是時序切分的常見漏洞，切分點本身擋不住。
    """
    n = len(sids)
    i_split = int(n * train_frac)
    t_split = int(sids[i_split])
    settle = sids + np.where(censored == 1, window, tau).astype(np.int64)
    tr = np.arange(i_split)
    dropped = 0
    if embargo:
        keep = settle[:i_split] <= t_split
        dropped = int((~keep).sum())
        tr = tr[keep]
    i_cal = int(len(tr) * (1.0 - calib_frac))
    info = {"n_total": n, "t_split": t_split, "embargo_dropped": dropped,
            "n_train": int(i_cal), "n_calib": int(len(tr) - i_cal),
            "n_test": int(n - i_split)}
    return tr[:i_cal], tr[i_cal:], np.arange(i_split, n), info


def train_one(X, y, w, Xv, yv, wv, seed: int, rounds: int,
              num_leaves: int = 63) -> tuple:
    import lightgbm as lgb
    params = {"objective": "l2", "learning_rate": 0.05, "num_leaves": num_leaves,
              "min_data_in_leaf": 100, "feature_fraction": 0.9,
              "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
              "seed": seed, "deterministic": True, "num_threads": 8}
    t0 = time.time()
    ds = lgb.Dataset(X, label=y, weight=w, free_raw_data=False)
    dv = lgb.Dataset(Xv, label=yv, weight=wv, reference=ds, free_raw_data=False)
    booster = lgb.train(params, ds, num_boost_round=rounds, valid_sets=[dv],
                        callbacks=[lgb.early_stopping(30, verbose=False)])
    return booster, round(time.time() - t0, 2)


def predict_latency_us(booster, X: np.ndarray, k: int = 64,
                       repeats: int = 200) -> dict:
    """熱路徑延遲：對 k 個候選評分一次要多久。

    🔴 論文引的 30 μs 是 LRB 的數字，不是本檔量到的。這裡量自己的。
       單執行緒，因為預測器跑在服務執行緒上，不能假設有 8 核可用。
    """
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(X), size=(repeats, k))
    ts = []
    for r in range(repeats):
        batch = np.ascontiguousarray(X[idx[r]])
        t0 = time.perf_counter()
        booster.predict(batch, num_threads=1)
        ts.append((time.perf_counter() - t0) * 1e6)
    ts = np.array(ts)
    # 🔴 這是共用機器上的時間量測。CLAUDE.md §3 的污染規則講的是 GPU，
    #    但 CPU 也會被別人佔——負載一起記下來，否則這個數字無法解讀。
    return {"k": k, "median_us": round(float(np.median(ts)), 1),
            "p90_us": round(float(np.percentile(ts, 90)), 1),
            "per_candidate_us": round(float(np.median(ts)) / k, 3),
            "loadavg_1m": round(os.getloadavg()[0], 1),
            "n_cpu": os.cpu_count()}


# ─────────────────────── 主流程 ───────────────────────

def write_csv(path: Path, rows: list[dict]) -> None:
    """append 一批列。**欄位與既有檔頭不同就中止。**

    🔴 `csv.DictWriter` 在 append 模式下不會檢查檔頭。加一個欄位再 append，
       新列會照新順序寫進舊檔頭底下——欄位整排錯位，而檔案看起來完全正常。
       這種錯要等到有人用某一欄去畫圖才會發現。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    keys = list(rows[0])
    if not new:
        with path.open(newline="") as f:
            old = next(csv.reader(f), [])
        if old and set(old) == set(keys) and old != keys:
            # 只是順序不同（不同分支組 dict 的次序不一樣）——照既有檔頭重排即可。
            # 真正該擋的是「欄位集合」變了。
            keys = old
        if old and old != keys:
            raise SystemExit(
                f"🔴 {path} 的欄位與這批新列不同，拒絕 append。\n"
                f"   既有：{old}\n   新的：{keys}\n"
                f"   差異：+{sorted(set(keys) - set(old))} "
                f"-{sorted(set(old) - set(keys))}\n"
                f"   要換 schema 就先把舊檔移到 results/superseded/。")
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def load_index() -> dict:
    return json.loads(INDEX.read_text()) if INDEX.exists() else {}


def save_index(d: dict) -> None:
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")


# `samples.csv` 的欄位順序。兩條路徑（完整重放 vs 只重算標籤）組 dict 的次序不同，
# 不釘死就會在 append 時被 schema 守門擋下。
SAMPLE_FIELDS = (
    "run_id", "ts", "workload", "trace", "model_profile", "window_accesses",
    "window_mult", "sample_rate", "seed", "label_mode", "k_deltas",
    "max_per_block", "n_accesses", "n_samples", "n_dropped_tail",
    "positive_rate", "n_labeled_by_next_access", "n_labeled_by_window",
    "censored_pct", "requests", "unique_blocks", "reuse_rate", "gpu_blocks",
    "median_len_tokens_sim", "median_len_tokens_src", "seconds",
    "run_dir", "x_dir", "doc_tokens", "n_docs", "requests_arg", "tail_frac",
    "alpha", "turns", "sessions", "long_frac", "long_turns", "concurrency")


def stage_features(a, cm, prof) -> Path:
    trace, outs, wmeta = load_workload(a)
    if a.workload == "mooncake" and not a.limit_requests:
        units = check_trace_units(a.trace, trace)
    else:
        if a.workload == "mooncake":
            print(f"  ⚠️ --limit-requests {a.limit_requests}：跳過 trace 單位檢查，"
                  f"這一輪是冒煙測試，不是結果")
        units = {"checked": False}
    gpu_blocks = (a.gpu_tokens or prof["gpu_kv_tokens"]) // BLOCK
    window = a.window or int(a.window_mult * gpu_blocks)
    tag = (a.trace if a.workload == "mooncake" else
           f"lc{a.doc_tokens // 1024}k" +
           ("z" if a.workload == "longctx-zipf" else "s"))
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-m5p-feat-{tag}-{a.model}"
    root = BIG / "runs" / run_id
    root.mkdir(parents=True, exist_ok=True)
    print(f"[特徵] run={run_id}　workload={wmeta['workload']}")
    print(f"[特徵] W = {window:,} 次存取（= {a.window_mult}× GPU 容量 "
          f"{gpu_blocks:,} blocks）")
    if a.reuse_features:
        # 特徵不受 (W, 取樣率, seed, 標籤模式) 影響，只重算標籤
        src = Path(a.reuse_features)
        srcm = json.loads((src / "features_meta.json").read_text())
        x_dir = Path(srcm.get("x_dir", str(src)))
        meta = np.load(x_dir / "meta.npy", mmap_mode="r")
        t0 = time.time()
        stats = label_only(meta, window, a.sample_rate, a.max_per_block, root,
                           a.seed, uncensored=a.label_mode == "uncensored")
        stats["seconds"] = round(time.time() - t0, 1)
        stats["k_deltas"] = srcm["k_deltas"]
        feat_names = srcm["feature_names"]
        print(f"[特徵] 重用 {x_dir.name} 的特徵矩陣，只重算標籤")
    else:
        rep = Replay(trace, k_deltas=a.deltas)
        stats = build_samples(rep, window, a.sample_rate, a.max_per_block, root,
                              seed=a.seed)
        stats["label_mode"] = "censored"
        stats["k_deltas"] = a.deltas
        x_dir = root
        feat_names = feature_names(a.deltas)
        if a.label_mode == "uncensored":
            meta = np.load(root / "meta.npy", mmap_mode="r")
            st2 = label_only(meta, window, a.sample_rate, a.max_per_block, root,
                             a.seed, uncensored=True)
            stats.update({k: v for k, v in st2.items() if k != "seconds"})
    stats.update({"run_id": run_id, "ts": datetime.now().astimezone().isoformat(),
                  "trace": a.trace if a.workload == "mooncake" else tag,
                  "model_profile": a.model,
                  "requests": len(trace),
                  "gpu_blocks": gpu_blocks, "window_mult": a.window_mult,
                  "unique_blocks": len({b for r in trace for b in r}),
                  "reuse_rate": round(1 - len({b for r in trace for b in r})
                                      / max(1, sum(len(r) for r in trace)), 4),
                  "seed": a.seed,
                  "median_len_tokens_sim": units.get("median_sim", ""),
                  "median_len_tokens_src": units.get("median_src", ""),
                  "run_dir": str(root), "x_dir": str(x_dir), **wmeta})
    (root / "features_meta.json").write_text(
        json.dumps({**stats, "feature_names": feat_names},
                   indent=2, ensure_ascii=False) + "\n")
    write_csv(OUT / "samples.csv", [{k: stats.get(k, "") for k in SAMPLE_FIELDS}])
    idx = load_index()
    idx[(f"{stats['trace']}:{a.model}:w{a.window_mult}:k{stats['k_deltas']}"
         f":r{a.sample_rate}:s{a.seed}:{stats['label_mode']}")] = str(root)
    save_index(idx)
    print(f"[特徵] {stats['n_accesses']:,} 次存取 -> {stats['n_samples']:,} 個樣本"
          f"（機制 a {stats['n_labeled_by_next_access']:,}／"
          f"機制 b {stats['n_labeled_by_window']:,}；"
          f"尾巴視窗不完整而丟棄 {stats['n_dropped_tail']:,}；"
          f"正樣本率 {100 * stats['positive_rate']:.1f}%），{stats['seconds']}s")
    return root


def stage_train(a, cm, prof, root: Path) -> dict:
    fm = json.loads((root / "features_meta.json").read_text())
    window = fm["window_accesses"]
    names = fm["feature_names"]
    x_dir = Path(fm.get("x_dir", str(root)))
    X = np.load(x_dir / "X.npy", mmap_mode="r")
    meta = np.load(x_dir / "meta.npy", mmap_mode="r")
    lab = np.load(root / "labels.npz")
    sids, tau, cen = lab["sample_ids"], lab["tau"], lab["censored"]
    run_id = (f"{datetime.now():%Y%m%d-%H%M%S}-m5p-train-"
              f"{fm.get('trace', a.trace)}-{a.model}")
    out_root = BIG / "runs" / run_id
    out_root.mkdir(parents=True, exist_ok=True)

    i_tr, i_cal, i_te, info = time_split(sids, tau, cen, window, a.train_frac,
                                         a.calib_frac, not a.no_embargo)
    print(f"[切分] train {len(i_tr):,}／calib {len(i_cal):,}／test {len(i_te):,}"
          f"（embargo 丟掉 {info['embargo_dropped']:,}）")

    cols = group_columns(names, a.feature_groups)
    names = [names[i] for i in cols]
    Xs = np.asarray(X[sids])[:, cols]        # 只取被抽樣的列與所選特徵族
    pos = np.asarray(meta[sids][:, 3], dtype=np.float64)
    y_reg = np.log(np.maximum(tau, 1.0))
    y_bin = (cen == 0).astype(np.int8)
    w_cost = drop_cost(cm, pos) / cm.cpu     # 式 (10) 的 w(a_i)
    w_sym = np.ones_like(w_cost)

    rows, cal_rows, imp_rows, pos_rows = [], [], [], []
    boosters, calibs = {}, {}
    base = {"run_id": run_id, "ts": datetime.now().astimezone().isoformat(),
            "trace": fm.get("trace", a.trace), "model_profile": a.model,
            "device": a.device, "window_accesses": window, "k_deltas": fm["k_deltas"],
            "n_train": len(i_tr), "n_calib": len(i_cal), "n_test": len(i_te),
            "embargo_dropped": info["embargo_dropped"],
            "num_leaves": a.num_leaves,
            "test_positive_rate": round(float(y_bin[i_te].mean()), 4),
            "features_run": root.name,
            "workload": fm.get("workload", "mooncake"),
            "label_mode": fm.get("label_mode", "censored"),
            "sample_rate": fm.get("sample_rate", ""),
            "data_seed": fm.get("seed", ""),
            "feature_groups": "+".join(a.feature_groups),
            "test_trace": fm.get("trace", a.trace)}
    for loss, w in (("sym_l2", w_sym), ("cost_l2", w_cost)):
        booster, secs = train_one(Xs[i_tr], y_reg[i_tr], w[i_tr],
                                  Xs[i_cal], y_reg[i_cal], w[i_cal],
                                  a.seed, a.rounds, a.num_leaves)
        booster.save_model(str(out_root / f"model_{loss}.txt"))
        boosters[loss] = booster
        yhat_cal = booster.predict(Xs[i_cal])
        yhat_te = booster.predict(Xs[i_te])
        # 🔴 校準集只有單一類別時，isotonic 會把**所有**輸入映到同一個值，
        #    於是 p̂ 變成常數：AUC 恰為 0.5、Spearman 是 nan、ECE 等於基底率的差。
        #    指標看起來「有數字」，實際上模型的輸出完全沒被用到。
        #    合成的會談式工作負載很容易踩到（訓練期尾端剛好全是同一種樣本）。
        #    退路是**訓練期內等距抽樣**當校準集——它仍然全部早於測試期，不洩漏未來。
        calib_split = "tail"
        if y_bin[i_cal].min() == y_bin[i_cal].max():
            tr_all = np.concatenate([i_tr, i_cal])
            stride = max(2, len(tr_all) // max(1, len(i_cal)))
            i_cal2 = tr_all[::stride]
            print(f"  ⚠️ 校準集只有單一類別（{len(i_cal):,} 列全為 "
                  f"y={int(y_bin[i_cal][0])}），改用訓練期等距抽樣 "
                  f"{len(i_cal2):,} 列")
            i_cal = i_cal2
            calib_split = "strided"
            yhat_cal = booster.predict(Xs[i_cal])
        if y_bin[i_cal].min() == y_bin[i_cal].max():
            raise SystemExit(
                "🔴 整個訓練期都只有單一類別，校準無從做起。"
                "這代表工作負載或視窗 W 設定有問題，不是模型的問題。")
        # 校準：ŷ 越大代表越久才會再被用到 -> 機率越低，故對 -ŷ 做等張迴歸
        xk, yk = isotonic_fit(-yhat_cal, y_bin[i_cal].astype(np.float64))
        np.savez(out_root / f"calib_{loss}.npz", xk=xk, yk=yk)
        calibs[loss] = (xk, yk)
        p_te = isotonic_predict(xk, yk, -yhat_te)
        e, bins = ece(p_te, y_bin[i_te])
        lat = predict_latency_us(booster, Xs[i_te])
        # 固定開銷有多大：k=1 與 k=64 幾乎一樣，就代表量到的是 Python 端的
        # 呼叫成本而不是模型本身。這一條直接決定「30 μs」這個論證能不能引用。
        lat["k1_median_us"] = predict_latency_us(booster, Xs[i_te], k=1)["median_us"]
        lat["k256_median_us"] = predict_latency_us(booster, Xs[i_te],
                                                   k=256)["median_us"]
        for b in bins:
            cal_rows.append({**base, "loss": loss, **b})
        # 測試段的預測存起來：門檻掃描與可靠度圖都從這裡畫，不必重訓
        np.savez_compressed(out_root / f"test_pred_{loss}.npz",
                            p_hat=p_te.astype(np.float32),
                            y_hat=yhat_te.astype(np.float32),
                            y_bin=y_bin[i_te], y_reg=y_reg[i_te].astype(np.float32),
                            pos_tokens=pos[i_te].astype(np.float32))
        for n_, g in zip(names, booster.feature_importance("gain")):
            imp_rows.append({**base, "loss": loss, "feature": n_,
                             "gain": round(float(g), 2)})
        # 🔴 策略的逐出順序吃的是「下次使用時刻」的排序，不是二元標籤。
        #    AUC 高不代表排序好：AUC 只問「會不會在 W 內被用到」，
        #    Bélády 要問「誰先被用到」。兩者在本 trace 上差很多。
        sp_pos = spearman(yhat_te[y_bin[i_te] == 1],
                          y_reg[i_te][y_bin[i_te] == 1])
        sp_all = spearman(yhat_te, y_reg[i_te])
        for rule in ("0.5", "p_star"):
            for pr in cost_by_position(p_te, y_bin[i_te], cm, pos[i_te], rule):
                pos_rows.append({**base, "loss": loss, "threshold_rule": rule,
                                 **pr})
            thr = (np.full(len(i_te), 0.5) if rule == "0.5"
                   else p_star(cm, pos[i_te], "cpu"))
            m = cost_weighted_error(p_te, y_bin[i_te], thr, cm, pos[i_te])
            rows.append({**base, "loss": loss, "threshold_rule": rule,
                         "thr_median": round(float(np.median(thr)), 4),
                         "train_seconds": secs, "best_iter": booster.best_iteration,
                         "ece": round(e, 4), "auc": round(auc(p_te, y_bin[i_te]), 4),
                         "rmse_log_tau": round(float(np.sqrt(np.mean(
                             (yhat_te - y_reg[i_te]) ** 2))), 4),
                         "spearman_positives": round(sp_pos, 4),
                         "spearman_all": round(sp_all, 4),
                         "calib_split": calib_split,
                         "calib_positive_rate": round(
                             float(y_bin[i_cal].mean()), 4),
                         **m, **{f"lat_{k}": v for k, v in lat.items()}})
        print(f"[{loss}] {secs}s／{booster.best_iteration} 棵　ECE {e:.4f}　"
              f"AUC {auc(p_te, y_bin[i_te]):.4f}　"
              f"Spearman(正樣本) {sp_pos:.4f}　"
              f"推論 {lat['median_us']} μs/64 候選")
    # 兩個平凡基線，提供成本的尺度
    for name, p_const in (("always_keep", 1.0), ("always_drop", 0.0)):
        thr = p_star(cm, pos[i_te], "cpu")
        m = cost_weighted_error(np.full(len(i_te), p_const), y_bin[i_te],
                                thr, cm, pos[i_te])
        rows.append({**base, "loss": name, "threshold_rule": "p_star",
                     "thr_median": round(float(np.median(thr)), 4),
                     "train_seconds": 0.0, "best_iter": 0, "ece": "",
                     "auc": "", "rmse_log_tau": "",
                     "spearman_positives": "", "spearman_all": "",
                     "calib_split": "", "calib_positive_rate": "", **m,
                     **{f"lat_{k}": "" for k in ("k", "median_us", "p90_us",
                                                "per_candidate_us",
                                                "loadavg_1m", "n_cpu",
                                                "k1_median_us",
                                                "k256_median_us")}})
    write_csv(OUT / "predictor_metrics.csv", rows)
    write_csv(OUT / "calibration_bins.csv", cal_rows)
    write_csv(OUT / "feature_importance.csv", imp_rows)
    write_csv(OUT / "cost_by_position.csv", pos_rows)
    (out_root / "train_meta.json").write_text(json.dumps(
        {**base, "split": info, "features_run_dir": str(root),
         "x_dir": str(x_dir), "feature_cols": cols, "feature_names": names,
         "cost_model": cm.source}, indent=2, ensure_ascii=False) + "\n")
    # ---- 跨工作負載泛化（main.tex §5.2 承諾要報的第三層因應）----
    cross_rows = []
    if a.cross_test:
        for other in a.cross_test:
            key = (f"{other}:{a.model}:w{a.window_mult}:k{fm['k_deltas']}"
                   f":r{fm.get('sample_rate', 0.25)}:s{fm.get('seed', 1234)}"
                   f":{fm.get('label_mode', 'censored')}")
            odir = load_index().get(key)
            if not odir:
                print(f"  ⚠️ 跨測試找不到 {key} 的特徵，跳過")
                continue
            cross_rows += cross_eval(Path(odir), cols, cm, boosters, calibs,
                                     base, a)
        if cross_rows:
            write_csv(OUT / "cross_workload.csv", cross_rows)
    print(f"[訓練] 模型與校準表 -> {out_root}")
    return {"run_dir": out_root, "rows": rows}


def cross_eval(odir: Path, cols, cm, boosters, calibs, base, a) -> list[dict]:
    """於 A 訓練、於 B 測試。B 的**整段**都當測試集（模型沒看過 B 的任何一列）。"""
    ofm = json.loads((odir / "features_meta.json").read_text())
    ox = Path(ofm.get("x_dir", str(odir)))
    X = np.load(ox / "X.npy", mmap_mode="r")
    meta = np.load(ox / "meta.npy", mmap_mode="r")
    lab = np.load(odir / "labels.npz")
    sids, tau, cen = lab["sample_ids"], lab["tau"], lab["censored"]
    Xs = np.asarray(X[sids])[:, cols]
    pos = np.asarray(meta[sids][:, 3], dtype=np.float64)
    y_bin = (cen == 0).astype(np.int8)
    out = []
    for loss, booster in boosters.items():
        yhat = booster.predict(Xs)
        xk, yk = calibs[loss]
        p = isotonic_predict(xk, yk, -yhat)
        e, _ = ece(p, y_bin)
        m = cost_weighted_error(p, y_bin, p_star(cm, pos, "cpu"), cm, pos)
        out.append({**base, "loss": loss, "threshold_rule": "p_star",
                    "test_trace": ofm["trace"],
                    "test_workload": ofm.get("workload", "mooncake"),
                    "test_positive_rate": round(float(y_bin.mean()), 4),
                    "ece": round(e, 4), "auc": round(auc(p, y_bin), 4),
                    "spearman_positives": round(spearman(
                        yhat[y_bin == 1], np.log(np.maximum(tau, 1.0))[y_bin == 1]), 4),
                    **m})
        print(f"[跨測試] {base['trace']} -> {ofm['trace']}　{loss}　"
              f"AUC {out[-1]['auc']:.4f}　ECE {e:.4f}　"
              f"成本 {m['cost_ms']:,.0f} ms")
    return out


def main() -> int:
    global OUT, INDEX
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["features", "train", "all"])
    ap.add_argument("--trace", default="toolagent")
    ap.add_argument("--workload", default="mooncake",
                    choices=["mooncake", "longctx-session", "longctx-zipf"],
                    help="longctx-session = 合成的長上下文多輪會談，"
                         "見 session_trace 的說明")
    ap.add_argument("--doc-tokens", type=int, default=131072,
                    help="longctx-session：共用文件的長度（token）")
    ap.add_argument("--turns", type=int, default=4,
                    help="longctx-session：同一份文件被連續追問幾次")
    ap.add_argument("--sessions", type=int, default=250)
    ap.add_argument("--long-frac", type=float, default=0.5,
                    help="長會談的比例（其餘為 1–turns 輪的短會談）")
    ap.add_argument("--long-turns", type=int, default=12,
                    help="長會談的平均輪數")
    ap.add_argument("--n-docs", type=int, default=12,
                    help="longctx-zipf：文件數（論文 34.6% 那個點是 12）")
    ap.add_argument("--requests", type=int, default=120,
                    help="longctx-zipf：請求數（論文那個點是 120）")
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--concurrency", type=int, default=4,
                    help="同時進行幾個會談。重用距離 ≈ 這個數字（單位：請求），"
                         "壓力 = concurrency × 文件長度 ÷ GPU 預算")
    ap.add_argument("--tail-frac", type=float, default=0.02,
                    help="longctx-session：每輪各自不同的尾巴佔文件的比例")
    ap.add_argument("--model", default="qwen-awq", choices=list(MODEL_PROFILES))
    ap.add_argument("--device", default="nvme", choices=["sata", "nvme"])
    ap.add_argument("--gpu-tokens", type=int, default=None,
                    help="覆寫剖面的 GPU 預算（只影響 W 的預設值）")
    ap.add_argument("--window", type=int, default=None,
                    help="決策視窗 W（次存取）。預設 = window-mult × GPU blocks")
    ap.add_argument("--window-mult", type=float, default=1.0)
    ap.add_argument("--deltas", type=int, default=16,
                    help="保留幾個 delta 特徵（main.tex §5.2：k ≤ 32）")
    ap.add_argument("--sample-rate", type=float, default=0.25,
                    help="每次 block 存取被收成訓練樣本的機率")
    ap.add_argument("--max-per-block", type=int, default=0,
                    help="每個 block 的取樣上限（0 = 不設限；設了會讓訓練分布"
                         "與策略實際面對的分布不一致，見 build_samples 的說明）")
    ap.add_argument("--limit-requests", type=int, default=None)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--calib-frac", type=float, default=0.15,
                    help="訓練段最後這一比例拿來做 isotonic 校準與 early stopping")
    ap.add_argument("--no-embargo", action="store_true",
                    help="關掉 embargo（只用於證明它有效，不要用來產生結果）")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--num-leaves", type=int, default=63,
                    help="GBDT 的容量。§5.3 的論證是『有限容量的模型必須決定"
                         "把擬合能力放在哪』，所以容量是 (B) 區塊的自變數之一")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--reuse-features", default=None,
                    help="重用這個 run 的 X.npy／meta.npy，只重算標籤（快 4 倍）")
    ap.add_argument("--label-mode", default="censored",
                    choices=["censored", "uncensored"],
                    help="uncensored = 標籤用真實的下次使用時刻，不在 W 設限。"
                         "修的是『排序』那一側，見 label_only 的說明")
    ap.add_argument("--feature-groups", nargs="*",
                    default=["history", "deltas", "edc", "static"],
                    choices=list(FEATURE_GROUPS),
                    help="(C) 區塊的簡化版消融：只用這些特徵族訓練")
    ap.add_argument("--cross-test", nargs="*", default=[],
                    help="於本 trace 訓練後，另外在這些 trace 上測試（跨工作負載泛化）")
    ap.add_argument("--out-dir", default=str(OUT),
                    help="結果 CSV 的目錄。冒煙測試請指到 scratch，不要污染 results/")
    a = ap.parse_args()

    OUT = Path(a.out_dir)
    INDEX = OUT / "features_index.json"

    prof = profile(a.model)
    cm = load_cost_model(a.device, require_model_key=prof["cost_model_key"])
    print(f"[成本] {a.model}/{a.device}：CPU {cm.cpu:.3f}、SSD {cm.ssd:.3f}、"
          f"重算 {cm.recompute_base:.3f} + {cm.recompute_slope_per_token:.6f}×位置 ms/block")
    print(f"[門檻] p*_cpu：pos 0 -> {float(p_star(cm, np.array([0.0]))[0]):.4f}、"
          f"pos 128K -> {float(p_star(cm, np.array([131072.0]))[0]):.4f}"
          f"（式 (9)，非 0.5）")

    root = None
    if a.stage in ("features", "all"):
        root = stage_features(a, cm, prof)
    if a.stage in ("train", "all"):
        if root is None:
            tag = (a.trace if a.workload == "mooncake" else
                   f"lc{a.doc_tokens // 1024}k" +
                   ("z" if a.workload == "longctx-zipf" else "s"))
            key = (f"{tag}:{a.model}:w{a.window_mult}:k{a.deltas}"
                   f":r{a.sample_rate}:s{a.seed}:{a.label_mode}")
            idx = load_index()
            if a.features_dir:
                root = Path(a.features_dir)
            elif key in idx:
                root = Path(idx[key])
            else:
                raise SystemExit(f"🔴 找不到特徵 {key}\n   有的是：{list(idx)}")
        stage_train(a, cm, prof, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
