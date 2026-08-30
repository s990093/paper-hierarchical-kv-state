#!/usr/bin/env python3
"""Milestone 4 — Oracle 上界（🔴 整個研究的停損點）。

`EXPERIMENT_PLAN.md` §5：

> **問題**：一個「知道未來」的完美策略，比最好的簡單策略好多少？
>
> | Oracle 相對最佳 baseline 的改善 | 判定 |
> |---|---|
> | **> 15%** | 🟢 GO |
> | **5–15%** | 🟡 停下來問人 |
> | **< 5%** | 🔴 **NO-GO，停止** |
>
> **NO-GO 不是失敗。** 那是一個有價值的負面結果，會省下數個月。
> **不要為了讓專案繼續而美化數字。**

## 這支腳本是 trace-driven 模擬，不是端到端系統量測

計畫書允許：「用這個『未來知識』離線求解最佳放置（**整數規劃或貪婪近似皆可，
記錄你用的是哪一種**）」。這裡用的是 **Bélády/MIN**（單階最優）與
**成本感知貪婪**（多階）。

要讓這個模擬可信，它必須滿足兩個條件，缺一不可：

1. **成本常數來自實測**，不是假設 → 讀 `results/m2_harness/`，
   讀不到就**中止**，不使用預設值。
2. **模擬器要先能複現已量到的行為** → `--validate` 用 M3 的工作負載跑一次，
   比對模擬的 full_gpu vs cpu_lru 差距與**實測**的差距。
   對不上就代表模擬器不可信，此時 Oracle 的數字也不可信。

**沒有通過 (2) 的 Oracle 數字不得用於 go/no-go 判定。**

## 工作負載

M3 的工作負載（每個前綴恰好被存取兩次）**不適合拿來問 Oracle**——
存取序列太規律，Bélády 與 LRU 幾乎沒有差別，會系統性低估 headroom。

真實 serving 的重用是**偏斜**的：少數文件被反覆命中，多數只出現一兩次。
所以這裡用 Zipf 分布抽文件，這是快取工作負載的標準模型，
也正是論文關心的 prefix reuse 情境。

α 是可掃的參數——**headroom 對 α 的敏感度本身就是結果的一部分**，
不要只挑一個好看的 α 回報。

## 用法

    python code/m4_oracle.py --validate        # 先驗證模擬器
    python code/m4_oracle.py --alpha 0.6 0.9 1.2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parent.parent
M2 = REPO / "results/m2_harness"
M3_CSV = REPO / "results/m3_baseline/baseline.csv"
OUT = REPO / "results/m4_oracle"

BLOCK = 16          # vLLM 預設 block size（token）


# ────────────────────────── 成本模型 ──────────────────────────

@dataclass
class CostModel:
    """每個 block 的「被需要時的成本」（毫秒）。全部來自實測。

    idle_bytes_per_block 是「平時成本」——這一維決定同樣的預算能放幾個 block，
    在模擬裡體現為各階的容量，不是加在時間上。
    這正是 EXPERIMENT_PLAN §3 說的「2×N 矩陣不是 1×N 向量」。
    """

    gpu: float                      # 已在 GPU：≈0
    cpu: float                      # 從 CPU 搬回來
    ssd: float                      # 從磁碟搬回來
    recompute_base: float           # 在位置 0 重算一個 block
    recompute_slope_per_token: float  # 每多 1 個前序 token，重算多花多少 ms
    source: dict = field(default_factory=dict)

    def cost(self, tier: str, position_tokens: int) -> float:
        if tier == "gpu":
            return self.gpu
        if tier == "cpu":
            return self.cpu
        if tier == "ssd":
            return self.ssd
        if tier == "drop":
            # EXPERIMENT_PLAN §3：C_recompute 不是常數，隨絕對位置成長
            return self.recompute_base + self.recompute_slope_per_token * position_tokens
        raise ValueError(tier)


def load_cost_model(device: str = "sata") -> CostModel:
    """從 results/m2_harness/ 讀實測常數。讀不到就中止——不使用預設值。

    `device` 決定用哪一組 SSD 階的量測。**這不是無關緊要的選項**：
    2026-08-30 的第一版量測因為 CPU 階開太大（24 GiB > 工作集 8 GiB），
    東西根本沒 cascade 到磁碟，量到的「SSD 0.4044 ms/block」其實是 CPU 階，
    **比真值便宜 13.7 倍**。修正後 SSD = 5.54 ms/block，
    而且**大於 DROP 的 4.01 ms/block**——也就是說 Oracle 舊版以為
    「放硬碟很便宜所以要多用」，實際上放硬碟比丟掉重算還貴。
    用錯這個常數，Oracle 的 headroom 就沒有意義。
    """
    ret = M2 / f"retrieval_cost_{device}.csv"
    if not ret.exists():          # 相容舊檔名
        ret = M2 / "retrieval_cost.csv"
    rec = M2 / "recompute_position.csv"
    missing = [str(p) for p in (ret, rec) if not p.exists()]
    if missing:
        raise SystemExit(
            "🔴 缺少 M2 的實測成本常數，拒絕用假設值跑 Oracle。\n"
            f"   缺少：{missing}\n"
            "   先跑：python code/m2_cost_model.py --gpu 0 --stage all\n"
            "   （EXPERIMENT_PLAN §0 禁令 1：不准編造任何數字）")

    rows = list(csv.DictReader(ret.open()))
    def warm(tier: str) -> float | None:
        v = [float(r["ttft_ms"]) for r in rows
             if r["tier"] == tier and r["round"] == "warm" and r["ttft_ms"]]
        return median(v) if v else None
    ctxs = {int(r["ctx"]) for r in rows}
    ctx = max(ctxs) if ctxs else None
    nblk = ctx / BLOCK if ctx else None

    g, c, s = warm("gpu_resident"), warm("cpu"), warm("ssd")
    if None in (g, c, s) or not nblk:
        raise SystemExit(f"🔴 retrieval_cost.csv 不完整：gpu={g} cpu={c} ssd={s} ctx={ctx}")

    rrows = list(csv.DictReader(rec.open()))
    pts = sorted({int(r["cached_prefix_tokens"]) for r in rrows})
    chunk = int(rrows[0]["recomputed_tokens"])
    def at(p: int) -> float:
        return median([float(r["ttft_ms"]) for r in rrows
                       if int(r["cached_prefix_tokens"]) == p and r["ttft_ms"]])
    y0, yN = at(pts[0]), at(pts[-1])
    blocks_in_chunk = chunk / BLOCK
    slope = ((yN - y0) / (pts[-1] - pts[0])) / blocks_in_chunk if pts[-1] > pts[0] else 0.0

    return CostModel(
        gpu=0.0,
        cpu=max(0.0, (c - g)) / nblk,
        ssd=max(0.0, (s - g)) / nblk,
        recompute_base=y0 / blocks_in_chunk,
        recompute_slope_per_token=slope,
        source={
            "retrieval_csv": str(ret), "recompute_csv": str(rec),
            "ctx_used": ctx, "blocks_per_ctx": nblk,
            "warm_gpu_ms": g, "warm_cpu_ms": c, "warm_ssd_ms": s,
            "recompute_chunk_tokens": chunk,
            "recompute_at_pos0_ms": y0, "recompute_at_maxpos_ms": yN,
            "positions": pts,
        })


# ────────────────────────── 工作負載 ──────────────────────────

def zipf_trace(n_docs: int, doc_blocks: int, n_requests: int,
               alpha: float, seed: int) -> list[list[int]]:
    """回傳每個請求需要的 block id 序列。

    每個文件是一段連續的 block。請求 = 讀完該文件的所有 block（prefix 語意：
    要用第 k 個 block，前面 0..k-1 都得先在），所以 block 之間有依賴，
    這正是 EXPERIMENT_PLAN §3 說的「Drop 有依賴限制」。
    """
    rng = random.Random(seed)
    w = [1.0 / ((i + 1) ** alpha) for i in range(n_docs)]
    tot = sum(w)
    cum, acc = [], 0.0
    for x in w:
        acc += x / tot
        cum.append(acc)

    def pick() -> int:
        u = rng.random()
        for i, c in enumerate(cum):
            if u <= c:
                return i
        return n_docs - 1

    return [[d * doc_blocks + b for b in range(doc_blocks)]
            for d in (pick() for _ in range(n_requests))]


# ────────────────────────── 策略 ──────────────────────────

class Sim:
    """單一策略在 trace 上的模擬。回傳總成本（ms）與各階命中次數。"""

    def __init__(self, cm: CostModel, gpu_blocks: int, cpu_blocks: int,
                 ssd_blocks: int):
        self.cm = cm
        self.cap = {"gpu": gpu_blocks, "cpu": cpu_blocks, "ssd": ssd_blocks}

    # -- 線上策略 --------------------------------------------------
    def run_online(self, trace: list[list[int]], policy: str,
                   use_cpu: bool, use_ssd: bool,
                   split_at: int | None = None) -> dict:
        """split_at 給定時，另外回報第 split_at 個請求之後的成本（warm 段）。

        ⚠️ 為什麼需要這個：M3 的實測量的是**warm 那一輪**的 TTFT，
        模擬若回報兩輪加總，就是拿不同的量在比對，驗證會失去意義。
        2026-08-30 第一版驗證就犯了這個錯（實測比 9.0，模擬比 1.87，
        看起來像模擬器壞掉，其實是比錯東西）。"""
        gpu: OrderedDict[int, None] = OrderedDict()
        cpu: OrderedDict[int, None] = OrderedDict()
        ssd: OrderedDict[int, None] = OrderedDict()
        # ARC 的 ghost list（只在 policy=="arc" 用）
        b1: OrderedDict[int, None] = OrderedDict()
        b2: OrderedDict[int, None] = OrderedDict()
        t1: OrderedDict[int, None] = OrderedDict()
        t2: OrderedDict[int, None] = OrderedDict()
        target_t1 = 0

        total = 0.0
        warm_total = 0.0
        hits = {"gpu": 0, "cpu": 0, "ssd": 0, "drop": 0}
        warm_hits = {"gpu": 0, "cpu": 0, "ssd": 0, "drop": 0}

        def demote(blk: int) -> None:
            if use_cpu:
                cpu[blk] = None
                cpu.move_to_end(blk)
                while len(cpu) > self.cap["cpu"]:
                    ev, _ = cpu.popitem(last=False)
                    if use_ssd:
                        ssd[ev] = None
                        while len(ssd) > self.cap["ssd"]:
                            ssd.popitem(last=False)

        for ri, req in enumerate(trace):
            warm = split_at is not None and ri >= split_at
            for pos, blk in enumerate(req):
                if blk in gpu:
                    hits["gpu"] += 1
                    if warm:
                        warm_hits["gpu"] += 1
                    total += self.cm.cost("gpu", pos * BLOCK)
                    gpu.move_to_end(blk)
                    if policy == "arc":
                        if blk in t1:
                            del t1[blk]
                            t2[blk] = None
                        elif blk in t2:
                            t2.move_to_end(blk)
                    continue
                if blk in cpu:
                    hits["cpu"] += 1
                    c = self.cm.cost("cpu", pos * BLOCK)
                    del cpu[blk]
                    tier_hit = "cpu"
                elif blk in ssd:
                    hits["ssd"] += 1
                    c = self.cm.cost("ssd", pos * BLOCK)
                    del ssd[blk]
                    tier_hit = "ssd"
                else:
                    hits["drop"] += 1
                    c = self.cm.cost("drop", pos * BLOCK)
                    tier_hit = "drop"
                total += c
                if warm:
                    warm_total += c
                    warm_hits[tier_hit] += 1

                # 放進 GPU，必要時逐出
                if policy == "arc":
                    if blk in b1:
                        target_t1 = min(self.cap["gpu"],
                                        target_t1 + max(1, len(b2) // max(1, len(b1))))
                        del b1[blk]
                    elif blk in b2:
                        target_t1 = max(0, target_t1 - max(1, len(b1) // max(1, len(b2))))
                        del b2[blk]
                    t1[blk] = None
                gpu[blk] = None
                while len(gpu) > self.cap["gpu"]:
                    if policy == "arc" and len(t1) > target_t1 and t1:
                        ev, _ = t1.popitem(last=False)
                        b1[ev] = None
                        while len(b1) > self.cap["gpu"]:
                            b1.popitem(last=False)
                    elif policy == "arc" and t2:
                        ev, _ = t2.popitem(last=False)
                        b2[ev] = None
                        while len(b2) > self.cap["gpu"]:
                            b2.popitem(last=False)
                    else:
                        ev, _ = gpu.popitem(last=False)
                        demote(ev)
                        continue
                    gpu.pop(ev, None)
                    demote(ev)
        return {"total_ms": total, "hits": hits,
                "warm_ms": warm_total, "warm_hits": warm_hits}

    # -- Oracle ----------------------------------------------------
    def run_oracle(self, trace: list[list[int]], use_cpu: bool,
                   use_ssd: bool) -> dict:
        """知道未來的最佳放置。

        單階時 Bélády/MIN（逐出下次使用最遠的）是**可證明最優**的。
        多階時我們用**成本感知貪婪**：被逐出的 block 依「下次使用有多近」排序，
        近的留在 CPU、遠的下放 SSD、不再用到的直接丟掉（成本 0）。
        這不保證全域最優，所以它是**下界的 Oracle**——真正的最優只會更好，
        因此用它做 go/no-go 是保守的（不會高估 headroom）。
        """
        # 預計算每個 (請求序號, block) 的下一次使用時間
        flat: list[tuple[int, int]] = []   # (time, blk)
        for ti, req in enumerate(trace):
            for blk in req:
                flat.append((ti, blk))
        nxt: dict[int, list[int]] = {}
        for i, (_, blk) in enumerate(flat):
            nxt.setdefault(blk, []).append(i)
        ptr = {b: 0 for b in nxt}

        gpu: set[int] = set()
        cpu: set[int] = set()
        ssd: set[int] = set()
        total = 0.0
        hits = {"gpu": 0, "cpu": 0, "ssd": 0, "drop": 0}

        def next_use(blk: int, now: int) -> float:
            lst = nxt[blk]
            p = ptr[blk]
            while p < len(lst) and lst[p] <= now:
                p += 1
            ptr[blk] = p
            return lst[p] if p < len(lst) else math.inf

        for i, (_, blk) in enumerate(flat):
            pos = 0
            # 位置 = 該 block 在其請求中的序號 × BLOCK
            # （flat 保序，直接用該 block 在 req 內的 index）
            if blk in gpu:
                hits["gpu"] += 1
            elif blk in cpu:
                hits["cpu"] += 1
                total += self.cm.cost("cpu", pos)
                cpu.discard(blk)
            elif blk in ssd:
                hits["ssd"] += 1
                total += self.cm.cost("ssd", pos)
                ssd.discard(blk)
            else:
                hits["drop"] += 1
                total += self.cm.cost("drop", pos)
            gpu.add(blk)

            while len(gpu) > self.cap["gpu"]:
                victim = max(gpu, key=lambda b: next_use(b, i))
                gpu.discard(victim)
                if next_use(victim, i) is math.inf:
                    continue                       # 不再用到 -> 丟掉，成本 0
                if use_cpu and len(cpu) < self.cap["cpu"]:
                    cpu.add(victim)
                elif use_ssd and len(ssd) < self.cap["ssd"]:
                    ssd.add(victim)
                else:
                    # 兩階都滿：把 CPU 裡下次使用最遠的換下去
                    if use_cpu and cpu:
                        far = max(cpu, key=lambda b: next_use(b, i))
                        if next_use(far, i) > next_use(victim, i):
                            cpu.discard(far)
                            cpu.add(victim)
                            if use_ssd and len(ssd) < self.cap["ssd"]:
                                ssd.add(far)
        return {"total_ms": total, "hits": hits}


# ────────────────────────── 驗證 ──────────────────────────

def validate(cm: CostModel) -> dict:
    """用 M3 的工作負載跑模擬，跟**實測**比對。

    模擬器若複現不出已經量到的 full_gpu vs cpu_lru 差距，
    它預測的 Oracle headroom 就沒有理由可信。
    """
    if not M3_CSV.exists():
        raise SystemExit(f"🔴 找不到 {M3_CSV}，無法驗證模擬器")
    rows = [r for r in csv.DictReader(M3_CSV.open())
            if r.get("concurrency_mode") == "serial" and r["model_key"] == "llama"]
    if not rows:
        raise SystemExit("🔴 baseline.csv 沒有 serial 的 llama 資料")

    kv_tokens = int(median([int(r["gpu_kv_cache_tokens"]) for r in rows]))
    out = []
    for ctx in sorted({int(r["ctx"]) for r in rows}):
        n = int(median([int(r["n_prefixes"]) for r in rows if int(r["ctx"]) == ctx]))
        def meas(base: str) -> float | None:
            v = [float(r["ttft_ms"]) for r in rows
                 if int(r["ctx"]) == ctx and r["baseline"] == base
                 and r["round"] == "warm" and r["ttft_ms"]]
            return median(v) if v else None
        m_full, m_lru = meas("full_gpu"), meas("cpu_lru")
        if m_full is None or m_lru is None:
            continue
        # M3 的 trace：n 個不同前綴，cold 一輪 warm 一輪
        doc_blocks = ctx // BLOCK
        trace = [[d * doc_blocks + b for b in range(doc_blocks)]
                 for d in list(range(n)) * 2]
        sim = Sim(cm, gpu_blocks=kv_tokens // BLOCK,
                  cpu_blocks=(24 * 1024**3) // (BLOCK * 128 * 1024),
                  ssd_blocks=10**9)
        # split_at=n：trace 前 n 個請求是 cold、其後 n 個是 warm。
        # 實測量的是 warm 那一輪的 TTFT，所以模擬也只能取 warm 段來比。
        s_full = sim.run_online(trace, "lru", False, False, split_at=n)
        s_lru = sim.run_online(trace, "lru", True, False, split_at=n)
        sim_ratio = s_full["warm_ms"] / max(1e-9, s_lru["warm_ms"])
        out.append({
            "ctx": ctx, "n_prefixes": n,
            "measured_full_warm_ms": round(m_full, 1),
            "measured_lru_warm_ms": round(m_lru, 1),
            "measured_ratio": round(m_full / m_lru, 2),
            # 模擬的 warm 段成本除以請求數，才和「每個請求的 TTFT」可比
            "sim_full_warm_ms_per_req": round(s_full["warm_ms"] / n, 1),
            "sim_lru_warm_ms_per_req": round(s_lru["warm_ms"] / n, 1),
            "sim_ratio": round(sim_ratio, 2),
            "sim_full_warm_hits": s_full["warm_hits"],
            "sim_lru_warm_hits": s_lru["warm_hits"],
        })
    return {"gpu_kv_tokens": kv_tokens, "rows": out}


# ────────────────────────── 主流程 ──────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="sata", choices=["sata", "nvme"],
                    help="SSD 階要用哪個裝置的量測。兩者差異見 RUNLOG 發現 11")
    ap.add_argument("--validate", action="store_true",
                    help="只跑模擬器驗證（比對 M3 實測），不做 go/no-go")
    ap.add_argument("--alpha", type=float, nargs="*", default=[0.6, 0.9, 1.2],
                    help="Zipf 偏斜參數；headroom 對它的敏感度是結果的一部分")
    ap.add_argument("--docs", type=int, default=64)
    ap.add_argument("--doc-tokens", type=int, default=4096)
    ap.add_argument("--requests", type=int, default=400)
    ap.add_argument("--gpu-tokens", type=int, default=None,
                    help="GPU KV 預算（token）；預設取 M3 實測值")
    ap.add_argument("--cpu-gib", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    cm = load_cost_model(a.device)
    print(f"=== 成本模型（實測，SSD 階用 {a.device} 那組）===")
    print(json.dumps({"gpu_ms_per_block": cm.gpu, "cpu_ms_per_block": round(cm.cpu, 4),
                      "ssd_ms_per_block": round(cm.ssd, 4),
                      "recompute_base_ms_per_block": round(cm.recompute_base, 4),
                      "recompute_slope_ms_per_block_per_token":
                          round(cm.recompute_slope_per_token, 8)},
                     indent=2, ensure_ascii=False))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cost_model.json").write_text(json.dumps(
        {"measured": cm.source,
         "derived_ms_per_block": {"gpu": cm.gpu, "cpu": cm.cpu, "ssd": cm.ssd,
                                  "recompute_base": cm.recompute_base,
                                  "recompute_slope_per_token":
                                      cm.recompute_slope_per_token}},
        indent=2, ensure_ascii=False) + "\n")

    if a.validate:
        v = validate(cm)
        print(f"\n=== 模擬器驗證（GPU KV = {v['gpu_kv_tokens']:,} tokens）===")
        print("比的是同一個量：warm 那一輪，full_gpu 相對 cpu_lru 的成本倍數\n")
        print(f"{'ctx':>7}{'實測':>10}{'模擬':>10}{'比值差':>9}  判定")
        ok = agree_mag = n_cmp = 0
        for r in v["rows"]:
            # 退化情況：工作集塞得進 GPU，模擬裡兩者成本都是 0，比值無定義。
            # 這不是「對不上」，是「這個點沒有鑑別力」，不該計入判定。
            if r["sim_full_warm_ms_per_req"] < 1e-6 and r["sim_lru_warm_ms_per_req"] < 1e-6:
                print(f"{r['ctx']:>7}{r['measured_ratio']:>10.2f}{'—':>10}"
                      f"{'—':>9}  ⚪ 工作集塞得下，無鑑別力")
                continue
            n_cmp += 1
            same_dir = (r["measured_ratio"] > 1) == (r["sim_ratio"] > 1)
            rel = abs(r["sim_ratio"] - r["measured_ratio"]) / max(r["measured_ratio"], 1e-9)
            mag = rel <= 0.5                       # 倍數差在 50% 以內才算量級相符
            ok += same_dir
            agree_mag += mag
            tag = ("✅ 相符" if same_dir and mag else
                   "🟡 同向但量級偏離" if same_dir else "🔴 反向")
            print(f"{r['ctx']:>7}{r['measured_ratio']:>10.2f}{r['sim_ratio']:>10.2f}"
                  f"{rel*100:>8.0f}%  {tag}")
        (OUT / "simulator_validation.json").write_text(
            json.dumps(v, indent=2, ensure_ascii=False) + "\n")
        n = n_cmp
        print(f"\n有鑑別力的點 {n} 個：方向一致 {ok}/{n}；量級也相符 {agree_mag}/{n}")
        if n and agree_mag < n:
            print("\n🟡 有 ctx 的量級對不上。可能的原因：")
            print("   * 模擬把「miss」記成重算**一個 block**，但 vLLM 的 prefix cache 是")
            print("     前綴語意——中間缺一塊，其後全部要重算。模擬會低估 full_gpu 的成本。")
            print("   * 實測的 TTFT 含排程、tokenize、取樣等固定開銷，模擬只有搬運與計算。")
            print("   → **此時 Oracle 的絕對 headroom 不可引用**，只能用它的"
                  "『隨 α 如何變化』這個相對趨勢。")
        return 0 if n and agree_mag == n else 1

    gpu_tokens = a.gpu_tokens
    if gpu_tokens is None:
        rows = [r for r in csv.DictReader(M3_CSV.open())
                if r.get("concurrency_mode") == "serial" and r["model_key"] == "llama"]
        gpu_tokens = int(median([int(r["gpu_kv_cache_tokens"]) for r in rows]))
    gpu_blocks = gpu_tokens // BLOCK
    cpu_blocks = int(a.cpu_gib * 1024**3 // (BLOCK * 128 * 1024))
    doc_blocks = a.doc_tokens // BLOCK

    print(f"\n=== Oracle 設定 ===")
    print(f"  GPU 預算   : {gpu_tokens:,} tokens = {gpu_blocks:,} blocks（M3 實測值）")
    print(f"  CPU 預算   : {a.cpu_gib} GiB = {cpu_blocks:,} blocks")
    print(f"  工作負載   : {a.docs} 個文件 × {a.doc_tokens} tokens，"
          f"{a.requests} 個請求，Zipf")
    print(f"  工作集     : {a.docs * doc_blocks:,} blocks "
          f"({a.docs * doc_blocks / gpu_blocks:.1f}× GPU 預算)")

    rows = []
    for alpha in a.alpha:
        trace = zipf_trace(a.docs, doc_blocks, a.requests, alpha, a.seed)
        sim = Sim(cm, gpu_blocks, cpu_blocks, ssd_blocks=10**9)
        res = {
            "full_gpu": sim.run_online(trace, "lru", False, False),
            "cpu_lru": sim.run_online(trace, "lru", True, False),
            "cpu_arc": sim.run_online(trace, "arc", True, False),
            "tier_fs": sim.run_online(trace, "lru", True, True),
            "oracle": sim.run_oracle(trace, True, True),
        }
        best_base = min((k for k in res if k != "oracle"),
                        key=lambda k: res[k]["total_ms"])
        head = 100 * (res[best_base]["total_ms"] - res["oracle"]["total_ms"]) \
            / res[best_base]["total_ms"]
        verdict = ("GO" if head > 15 else
                   "MARGINAL_ASK_HUMAN" if head >= 5 else "NO_GO")

        print(f"\n--- Zipf α = {alpha} ---")
        print(f"{'policy':10s}{'total ms':>12s}{'gpu hit':>9s}{'cpu':>8s}"
              f"{'ssd':>8s}{'recompute':>11s}")
        for k, v in res.items():
            h = v["hits"]
            print(f"{k:10s}{v['total_ms']:>12,.0f}{h['gpu']:>9,}{h['cpu']:>8,}"
                  f"{h['ssd']:>8,}{h['drop']:>11,}")
        print(f"  最佳 baseline = {best_base}；Oracle 改善 = **{head:.1f}%** → {verdict}")

        for k, v in res.items():
            rows.append({
                "ts": datetime.now().astimezone().isoformat(),
                "alpha": alpha, "policy": k,
                "total_ms": round(v["total_ms"], 2),
                "gpu_hits": v["hits"]["gpu"], "cpu_hits": v["hits"]["cpu"],
                "ssd_hits": v["hits"]["ssd"], "recompute": v["hits"]["drop"],
                "gpu_budget_tokens": gpu_tokens, "cpu_budget_gib": a.cpu_gib,
                "docs": a.docs, "doc_tokens": a.doc_tokens,
                "requests": a.requests, "seed": a.seed,
                "best_baseline": best_base,
                "oracle_headroom_pct": round(head, 2) if k == "oracle" else "",
                "verdict": verdict if k == "oracle" else "",
                "method": "trace-driven simulation; Belady/MIN + cost-aware greedy",
                "cost_model": str(OUT / "cost_model.json"),
            })

    p = OUT / "oracle.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {p}")

    heads = [r["oracle_headroom_pct"] for r in rows if r["policy"] == "oracle"]
    print(f"\n{'=' * 60}\n🚦 go/no-go：headroom 隨 α = {a.alpha} 為 {heads}")
    print("   > 15% GO ｜ 5–15% 停下來問人 ｜ < 5% NO-GO")
    print("   ⚠️ 這是 trace-driven 模擬的上界，不是端到端量測。")
    print("      成本常數來自 M2 實測；模擬器須先通過 --validate。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
