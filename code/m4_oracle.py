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
import heapq
import math
import os
import random
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parent.parent
M2 = REPO / "results/m2_harness"
M5 = REPO / "results/m5_quality"
M3 = REPO / "results/m3_baseline"
M3_CSV = REPO / "results/m3_baseline/baseline.csv"
OUT = REPO / "results/m4_oracle"

# 模擬器的版本戳記 = 本檔內容的 SHA-1 前 8 碼。
# 每一份結果 CSV 都要帶著它。2026-08-31 一天之內改了六次模擬器語意，
# 而每一版的輸出看起來都完全正常——沒有戳記就無法事後分辨哪一列是哪一版產生的。
def sim_version() -> str:
    """只雜湊「會影響模擬數字」的那些原始碼，不是整個檔案。

    🔴 第一版雜湊整個檔案，結果在掃描跑到一半時新增一個
       `load_precision_tiers`（模擬路徑根本不會呼叫它）就讓戳記改變，
       整批有效資料被當成舊版丟掉。
       戳記要反映**行為**，不是檔案位元組。

    納入雜湊的是：Sim 類別（逐出、成本記帳、前綴語意、預取、目的地規則）、
    成本模型的載入與推導、以及兩個工作負載產生器。
    改這些之中任何一個，數字就可能改變，戳記就該改變。
    加註解、加無關的載入器、改 CLI 說明，則不影響。
    """
    import hashlib
    import inspect
    parts = []
    for name in ("Sim", "load_cost_model", "mooncake_trace", "zipf_trace",
                 "CostModel"):
        obj = globals().get(name)
        if obj is not None:
            try:
                parts.append(inspect.getsource(obj))
            except OSError:
                parts.append(name)
    parts.append(f"BLOCK={BLOCK};MOONCAKE_BLOCK={MOONCAKE_BLOCK}")
    parts.append(repr(sorted(MODEL_PROFILES.items())))
    return hashlib.sha1("".join(parts).encode()).hexdigest()[:8]

BLOCK = 16          # vLLM 預設 block size（token）
# Mooncake trace 的 hash_id 粒度（token）。由資料實測得到，見 mooncake_trace()。
MOONCAKE_BLOCK = 512

# 各裝置的**持續**寫入頻寬（MiB/s），實測值，見 results/m2_harness/disk_bw*.csv。
# 🔴 這張表必須與 `--device` 綁在一起使用：用 A 裝置的成本常數配 B 裝置的
#    頻寬上限，就是裝置混用。2026-08-31 的第一版可行性分析正是這樣做的
#    （SATA 的成本常數 + NVMe 的頻寬），結論因此不成立。
#
# sata：Samsung 870 QVO（/ssd7）。1 GiB 短測 492，但那落在 QLC 的 SLC 快取裡；
#       16 GiB 長測只剩 181。KV 階是持續寫入，所以用 181。
# nvme：Crucial P3（/）。1 GiB 測試 2,512。
DEVICE_WRITE_MIBPS = {"sata": 181.0, "nvme": 2512.0}

# 各裝置的掛載點。用來回報「這顆碟到底有多大／還剩多少」。
# 🔴 這也必須跟著 --device 走：先前 --device nvme 但 fs-root 仍指 /ssd7，
#    於是報表印出 SATA 的 7.2 TiB 容量，卻在講 NVMe 的結果。
DEVICE_FS_ROOT = {"sata": "/ssd7", "nvme": "/home/hungwei"}


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


# ────────────────────── 模型剖面（防止混用） ──────────────────────
# 🔴 2026-08-31 抓到的錯誤：先前用 `--gpu-tokens 273872`（qwen-awq 的實測容量）
#    去配 M2 的成本常數，而 M2 **整組都是在 llama、BF16 權重、ctx=16384 量的**。
#    等於「A 模型的記憶體預算 × B 模型的搬運/重算成本」，兩邊不可通約。
#    CPU 階的 blocks 數也一樣：128 KiB/token 是 Llama-3.1-8B 的 GQA BF16 值，
#    換模型就不對。
#
#    修法：把「預算 / KV 每 token 大小 / 成本模型來源」綁成一個剖面，
#    要換模型就得整組換，且成本模型的 model_key 必須對得上，對不上直接拒跑。
#
# kv_bytes_per_token 的算法（BF16 KV）：
#   層數 × KV head 數 × head_dim × 2(K,V) × 2 bytes
#   Llama-3.1-8B: 32 × 8 × 128 × 2 × 2 = 131,072 = 128 KiB  ✔ 與實測 kv_gib 相符
MODEL_PROFILES = {
    # 唯一一個目前有**自洽成本模型**的剖面。M2 全部量自這個設定。
    "llama-bf16": {
        "gpu_kv_tokens": 48_128,          # M2/M3 serial 實測（gpu_mem_util 相同）
        "kv_bytes_per_token": 131_072,    # 32×8×128×2×2
        "cost_model_key": "llama",
        "source": "results/m2_harness/*.csv (model_key=llama, ctx=16384)",
    },
    # 以下剖面**沒有**對應的成本量測。要用必須先跑 M2 的對應設定，
    # 否則 load_cost_model 會拒絕。列在這裡是為了記錄實測容量。
    # 🔴 llama-awq 在 M3 跑的是 **FP8 KV**（m3_baseline 的設定裡就寫著
    #    "extra": ["--kv-cache-dtype", "fp8"]），所以每 token 是 64 KiB 不是 128。
    #    寫成 128 會讓 decode 的擬合斜率換算出 1,030 GB/s——超過 3090 的峰值
    #    936 GB/s，物理上不可能。check_decode_bandwidth 抓到了這個錯。
    #    自我檢查：120,320 token × 64 KiB = 7.3 GiB，加 AWQ 權重 5.7 GB
    #    在 24 GB 卡上合理。
    "llama-awq": {"gpu_kv_tokens": 120_320, "kv_bytes_per_token": 65_536,
                  "cost_model_key": "llama-awq", "source": "M1 capacity.csv",
                  "kv_dtype": "fp8"},
    # 🔴 Qwen2.5-7B 的 KV 幾何與 Llama-3.1-8B 不同：
    #    28 層 × 4 KV head × 128 dim × 2(K,V) × 2 bytes = 57,344 = 56 KiB/token。
    #    先前這裡照抄 Llama 的 128 KiB，錯了 2.29 倍。
    #    自我檢查：273,872 × 56 KiB = 14.6 GiB（24 GB 卡放得下 ✅）
    #             273,872 × 128 KiB = 33.4 GiB（放不下 🔴）
    #    這個矛盾本來就該被抓到，所以下面加了 _check_profiles()。
    "qwen-awq": {"gpu_kv_tokens": 273_872, "kv_bytes_per_token": 57_344,
                 "cost_model_key": "qwen-awq", "source": "M1 capacity.csv"},
}


# 這張卡的實體上限。KV 預算 × 每 token 大小若超過它，剖面一定寫錯了。
CARD_VRAM_GIB = 24.0


def _check_profiles() -> None:
    """剖面的自我一致性檢查。在匯入時就跑，錯了立刻中止。

    🔴 2026-08-31 抓到：qwen-awq 的 kv_bytes_per_token 照抄了 Llama 的
       128 KiB，但 Qwen2.5-7B 是 56 KiB。照錯的值算，273,872 token 要
       33.4 GiB，超過整張 24 GB 的卡——這個矛盾在程式裡放了一整天沒被發現，
       因為沒有人去乘那兩個數字。
    """
    for name, p in MODEL_PROFILES.items():
        gib = p["gpu_kv_tokens"] * p["kv_bytes_per_token"] / 1024**3
        if gib > CARD_VRAM_GIB:
            raise SystemExit(
                f"🔴 剖面 {name} 自相矛盾：KV 預算 {p['gpu_kv_tokens']:,} token "
                f"× {p['kv_bytes_per_token'] / 1024:.0f} KiB/token = {gib:.1f} GiB，"
                f"超過整張卡的 {CARD_VRAM_GIB:.0f} GiB。\n"
                f"   最可能的原因：kv_bytes_per_token 照抄了別的模型。"
                f"算法 = 層數 × KV head 數 × head_dim × 2(K,V) × 2 bytes。")


_check_profiles()


def profile(name: str) -> dict:
    if name not in MODEL_PROFILES:
        raise SystemExit(f"🔴 未知剖面 {name}；可用：{list(MODEL_PROFILES)}")
    return MODEL_PROFILES[name]


def load_decode_model(model_key: str = "llama") -> dict:
    """從 M3 的實測擬合 decode 每一步的成本。

    ## 為什麼需要

    模擬器原本只模 prefill。但用 trace 裡真實的 `output_length` 換算，
    decode 佔了總時間的 **48.8%（toolagent）／53.8%（conversation）**。
    我先前以為是 0.2%，那是因為 M3 的 harness 只生成 12–28 個 token。

    **放置決策只能優化 prefill**：decode 期間該請求的 KV 必須整份在 GPU 裡，
    每一步都要讀完，沒有放置的自由度。所以端到端的 headroom
    等於 prefill 的 headroom 乘上 prefill 的時間佔比。

    ## 模型

        每步成本 = base + slope × 該請求的 block 數

    base 是讀權重的固定成本，slope 是讀 KV 的成本（與 KV 大小成正比）。
    擬合自 `results/m3_baseline/baseline*.csv` 的
    `(total_ms − ttft_ms) / gen_tokens`。

    ## 物理檢查（這是抓錯的關鍵）

    slope 換算成頻寬必須低於卡的峰值（3090 是 936 GB/s）：

        llama-bf16（128 KiB/token）  381 GB/s = 41%   ✅ R²=0.9999
        qwen-bf16 （ 56 KiB/token）  839 GB/s = 90%   ✅ R²=0.8378
        llama-awq 若誤用 128 KiB    1030 GB/s = 110%  🔴 不可能

    最後那一列就是這樣抓到 `llama-awq` 其實跑 FP8 KV（64 KiB/token）的。
    """
    from statistics import median
    files = [M3 / "baseline.csv", M3 / "baseline_longctx.csv"]
    pts: list[tuple[int, float]] = []
    for f in files:
        if not f.exists():
            continue
        rows = [x for x in csv.DictReader(f.open())
                if x.get("model_key") == model_key
                and x.get("baseline") in (None, "", "full_gpu")
                and x.get("ttft_ms") and x.get("total_ms")
                and str(x.get("gen_tokens", "")).isdigit()
                and int(x["gen_tokens"]) > 0]
        g: dict[int, list] = {}
        for x in rows:
            c = int(x.get("actual_prompt_tokens") or x["ctx"])
            g.setdefault(c, []).append(x)
        for c, v in g.items():
            t = median(float(x["ttft_ms"]) for x in v)
            tot = median(float(x["total_ms"]) for x in v)
            n = median(int(x["gen_tokens"]) for x in v)
            if tot > t and n:
                pts.append((c // BLOCK, (tot - t) / n))
    if len(pts) < 2:
        raise SystemExit(
            f"🔴 找不到 model_key={model_key!r} 的 decode 量測，拒絕用預設值。\n"
            f"   需要 results/m3_baseline/baseline*.csv 裡至少兩個不同 ctx 的"
            f" full_gpu serial 列。")
    n = len(pts)
    sx = sum(a for a, _ in pts); sy = sum(b for _, b in pts)
    sxx = sum(a * a for a, _ in pts); sxy = sum(a * b for a, b in pts)
    slope = (n * sxy - sx * sy) / max(1e-12, n * sxx - sx * sx)
    base = (sy - slope * sx) / n
    mean = sy / n
    ss = sum((b - mean) ** 2 for _, b in pts)
    r2 = 1 - sum((b - (base + slope * a)) ** 2 for a, b in pts) / ss if ss else 1.0
    return {"decode_base_ms": base, "decode_ms_per_block": slope,
            "r2": r2, "n_points": n, "model_key": model_key,
            "source": [str(f) for f in files if f.exists()]}


def check_decode_bandwidth(dm: dict, kv_bytes_per_block: int,
                           peak_gbps: float = 936.0) -> float:
    """decode 的 slope 換算成 KV 讀取頻寬，超過卡的峰值就中止。

    🔴 這個檢查抓到過真實的錯誤：把 llama-awq（實際跑 FP8 KV，64 KiB/token）
       當成 128 KiB/token 去算，得到 1030 GB/s = 峰值的 110%。
       物理上不可能，代表 KV 大小用錯了。
    """
    ms = dm["decode_ms_per_block"]
    if ms <= 0:
        return float("nan")
    gbps = kv_bytes_per_block / (ms / 1000) / 1e9
    if gbps > peak_gbps:
        raise SystemExit(
            f"🔴 decode 的擬合斜率換算出 {gbps:,.0f} GB/s 的 KV 讀取頻寬，"
            f"超過卡的峰值 {peak_gbps:,.0f} GB/s。\n"
            f"   物理上不可能 -> 最可能是 kv_bytes_per_token 用錯了"
            f"（例如把跑 FP8 KV 的設定當成 BF16）。")
    return gbps


# ────────────────── GPU 上的精度階（六階動作空間缺的兩階） ──────────────────
# 論文的動作空間有六階，但 M2 原本只量了四階
# （gpu_resident / cpu / ssd / drop）。GPU-FP8 與 GPU-INT4 是
# 「住在 GPU 但精度降低」的狀態：容量變大，但每次讀取要反量化。
#
# 一個精度階由三個數字描述：
#   capacity_x    容量倍數        —— M1 實測（results/m1_capacity/capacity.csv）
#   dequant_ms    每 block 的反量化成本 —— M2 的 retrieval_cost*_precision_tiers.csv
#   epsilon       品質代價        —— M5（results/m5_quality/）
#
# 🔴 三個都量到之前，不得把精度階加進 Oracle 的動作空間。
#    缺任何一個就回報 NOT_MEASURED，不用預設值（禁令 1）。

# M1 實測的容量倍數。三個剖面都恰好 2.00×（fp8），故直接記為常數；
# int4 的 3.77× 來自 llama-bf16 那組（34.02 vs 128.11 KiB/token）。
PRECISION_CAPACITY_X = {"bf16": 1.00, "fp8": 2.00, "int8": 1.94, "int4": 3.77}


def load_quality(pattern: str = "gsm8k_precision*.csv") -> dict:
    """讀 M5 的品質量測，算出每個 KV 精度的正確率與**相對 BF16 的退化 ε**。

    🔴 一定要跟著回報信賴區間。n=120 時單一比例的標準誤約 3.7 個百分點，
       兩個比例相減的標準誤約 5.2 個百分點——那組資料的排序是非單調的
       （int8 竟高於 bf16、int4 高於 fp8），那是雜訊不是訊號。
       n=1000 才把單點標準誤壓到約 1.3、差值到約 1.8 個百分點。

    回傳 {dtype: {n, correct, acc, epsilon_pp, ci95_pp, distinguishable}}。
    `distinguishable` 為 False 時，該 ε 與 0 無法區分，**不得寫進論文當作
    「精度降低造成 X 個百分點的損失」**。
    """
    import glob
    from math import sqrt
    files = sorted(glob.glob(str(M5 / pattern)))
    if not files:
        return {}
    # 取樣本數最多的那一份（n=1000 優先於 n=120）
    best, best_n = None, -1
    for f in files:
        rows = list(csv.DictReader(open(f)))
        if len(rows) > best_n:
            best, best_n = rows, len(rows)
    if not best:
        return {}
    agg: dict[str, list[int]] = {}
    for r in best:
        k = r.get("kv_dtype") or r.get("config") or "?"
        ok = str(r.get("correct", "")).strip().lower() in ("true", "1")
        a = agg.setdefault(k, [0, 0])
        a[0] += ok
        a[1] += 1
    base_key = next((k for k in agg if k in ("auto", "bf16")), None)
    base_acc = agg[base_key][0] / agg[base_key][1] if base_key else None
    out = {}
    for k, (c, n) in agg.items():
        acc = c / n
        se = sqrt(acc * (1 - acc) / n)
        row = {"n": n, "correct": c, "acc_pct": round(100 * acc, 2),
               "se_pp": round(100 * se, 2), "source": files[0]}
        if base_acc is not None and k != base_key:
            b_acc = base_acc
            b_n = agg[base_key][1]
            se_d = sqrt(acc * (1 - acc) / n + b_acc * (1 - b_acc) / b_n)
            eps = 100 * (b_acc - acc)
            ci = 1.96 * 100 * se_d
            row.update({"epsilon_pp": round(eps, 2), "ci95_pp": round(ci, 2),
                        "distinguishable": abs(eps) > ci})
        elif k == base_key:
            row.update({"epsilon_pp": 0.0, "ci95_pp": 0.0,
                        "distinguishable": True, "is_baseline": True})
        out[k] = row
    return out


def load_precision_tiers(device: str = "nvme") -> dict:
    """讀 GPU 精度階的反量化成本。量不到、或量得到但與零無法區分，一律回
    NOT_MEASURED，不猜。

    量法（見 overnight_v2.sh 步驟 1）：與 `gpu_resident` 完全相同的設定
    （工作集塞得進 GPU、不逐出），只換 `--kv-cache-dtype`。
    warm TTFT 減掉 bf16 那一列，差額即為每 block 的反量化成本。

    🔴 兩道守門，兩道都是踩過坑才加的：

    1. **只吃 `host_contention == QUIET` 的列。** 2026-08-31 那份
       `retrieval_cost_precision_tiers.csv` 18 列全是 HEAVY，依 CLAUDE.md §3
       時間欄作廢；但這支函式照樣從它算出 fp8=0.025、int4=0.0165 ms/block，
       而且 **int4 比 fp8 便宜**——物理上說不通。污染的數字看起來完全正常，
       所以必須在載入端擋掉，不能靠人記得。

    2. **階內全距與階間差同量級時，回 NOT_MEASURED。** n=3 時單看中位數會
       把雜訊讀成訊號。判準用非參數的「全距是否重疊」：某階的最小值若不高於
       基準的最大值，該階的反量化成本就與零無法區分。
    """
    cands = [M2 / "retrieval_cost_precision_tiers_quiet.csv",
             M2 / f"retrieval_cost_precision_tiers_{device}.csv",
             M2 / "retrieval_cost_precision_tiers.csv"]
    p = next((c for c in cands if c.exists()), None)
    out: dict[str, dict] = {}
    for name, mult in PRECISION_CAPACITY_X.items():
        out[name] = {"capacity_x": mult, "dequant_ms_per_block": "NOT_MEASURED",
                     "source": str(cands[-1])}
    if p is None:
        return out
    all_rows = list(csv.DictReader(p.open()))
    rows = [r for r in all_rows if r.get("host_contention") == "QUIET"]
    for v in out.values():
        v["source"] = str(p)
    if not rows:
        levels = sorted({r.get("host_contention", "?") for r in all_rows})
        for v in out.values():
            v["reason"] = f"{p.name} 的整機爭用為 {levels}，非 QUIET → 時間欄作廢"
        return out
    ctxs = {int(r["ctx"]) for r in rows if r.get("ctx")}
    if not ctxs:
        return out
    nblk = max(ctxs) / BLOCK

    def warm(tier: str) -> list[float]:
        return [float(r["ttft_ms"]) for r in rows
                if r["tier"] == tier and r["round"] == "warm" and r["ttft_ms"]]

    b = warm("gpu_resident")
    if not b:
        return out
    base = median(b)
    out["bf16"].update({"dequant_ms_per_block": 0.0, "warm_ms": round(base, 2),
                        "warm_range": [round(min(b), 2), round(max(b), 2)], "n": len(b)})
    for name, tier in (("fp8", "gpu_fp8"), ("int4", "gpu_int4")):
        w = warm(tier)
        if not w:
            continue
        med = median(w)
        rec = out[name]
        rec.update({"warm_ms": round(med, 2),
                    "warm_range": [round(min(w), 2), round(max(w), 2)],
                    "baseline_warm_ms": round(base, 2),
                    "baseline_range": [round(min(b), 2), round(max(b), 2)],
                    "n": len(w),
                    "delta_ms": round(med - base, 2)})
        # 全距重疊 → 與零無法區分（n 小，用非參數判準）
        if min(w) <= max(b):
            rec["distinguishable"] = False
            rec["reason"] = (f"warm 全距 {min(w):.1f}–{max(w):.1f} 與基準 "
                             f"{min(b):.1f}–{max(b):.1f} 重疊，n={len(w)}")
            continue
        rec["distinguishable"] = True
        rec["dequant_ms_per_block"] = round((med - base) / nblk, 5)
    return out


def load_cost_model(device: str = "sata",
                    require_model_key: str | None = None) -> CostModel:
    """從 results/m2_harness/ 讀實測常數。讀不到就中止——不使用預設值。

    `device` 決定用哪一組 SSD 階的量測。**這不是無關緊要的選項**：
    2026-08-30 的第一版量測因為 CPU 階開太大（24 GiB > 工作集 8 GiB），
    東西根本沒 cascade 到磁碟，量到的「SSD 0.4044 ms/block」其實是 CPU 階，
    **比真值便宜 13.7 倍**。修正後 SSD = 5.54 ms/block，
    而且**大於 DROP 的 4.01 ms/block**——也就是說 Oracle 舊版以為
    「放硬碟很便宜所以要多用」，實際上放硬碟比丟掉重算還貴。
    用錯這個常數，Oracle 的 headroom 就沒有意義。
    """
    # 檔名慣例（見 m2_cost_model.out_csv）：
    #   llama（BF16 權重）維持舊檔名；其他模型帶 model_key。
    #   device 後綴只有 SSD 階需要（SATA vs NVMe 差 13.7 倍）。
    mk = require_model_key or "llama"
    tag = "" if mk == "llama" else f"_{mk}"
    cands = [M2 / f"retrieval_cost{tag}_{device}.csv",
             M2 / f"retrieval_cost{tag}.csv"]
    ret = next((c for c in cands if c.exists()), cands[0])
    rec_c = [M2 / f"recompute_position{tag}.csv"]
    rec = next((c for c in rec_c if c.exists()), rec_c[0])
    missing = [str(p) for p in (ret, rec) if not p.exists()]
    if missing:
        raise SystemExit(
            "🔴 缺少 M2 的實測成本常數，拒絕用假設值跑 Oracle。\n"
            f"   缺少：{missing}\n"
            f"   先跑：python code/m2_cost_model.py --gpu 0 --stage all --model {mk}\n"
            "   （EXPERIMENT_PLAN §0 禁令 1：不准編造任何數字）")

    rows = list(csv.DictReader(ret.open()))
    if require_model_key is not None:
        have = sorted({r.get("model_key", "") for r in rows})
        if have != [require_model_key]:
            raise SystemExit(
                f"🔴 成本模型與所選剖面不相容，拒絕混用。\n"
                f"   剖面要求 model_key = {require_model_key!r}\n"
                f"   但 {ret} 裡是 {have}\n"
                f"   「A 模型的記憶體預算 × B 模型的搬運成本」不可通約。\n"
                f"   要用這個剖面，先跑該模型的 M2：\n"
                f"     python code/m2_cost_model.py --gpu 0 --stage all "
                f"--model {require_model_key}")
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

    # 🔴 外插警示：重算成本是位置的線性函數，也是模擬裡的主導成本項。
    #    擬合的位置上限若遠小於工作負載的最大位置，該係數就是外插而非量測。
    #    理論上 prefill 的每個 block 要對前面所有 block 做注意力，
    #    成本本來就隨位置線性成長，所以線性外插有依據——但仍是外插。
    fit_max = max(pts) if pts else 0
    print(f"[cost] 重算成本的位置擬合上限 = {fit_max:,} token。"
          f"超過此位置的成本為線性外插。")
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
            "position_fit_max_tokens": fit_max,
        })


# ────────────────────────── 工作負載 ──────────────────────────

TRACES = BIG_TRACES = Path(
    os.environ.get("PAPER_HKV_TRACES",
                   "/ssd7/hungwei/paper-hkv/datasets/traces"))


def mooncake_trace(name: str, limit: int | None = None) -> list[list[int]]:
    """讀 Mooncake 的**真實生產流量** trace。

    這是 Zipf 合成 trace 的替代品，也是它的檢驗。

    來源：Mooncake (FAST'25, Moonshot AI) 隨論文釋出的 trace
    https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/traces

    格式每列一個請求：
        {"timestamp": ms, "input_length": n, "output_length": n,
         "hash_ids": [block hash...]}

    `hash_ids` **就是 block 層級的前綴共用資訊**——共用前綴的請求會共用開頭的
    hash_ids。這跟模擬器需要的 `list[list[block_id]]` 完全同構，可直接餵入。

    實測解析結果（與論文回報值相符，可作為解析正確的佐證）：

    | trace | 請求數 | 前綴重用率 | 論文回報 | 擬合 Zipf α |
    |---|---|---|---|---|
    | conversation | 12,031 | **36.6%** | ~40% | **0.37** |
    | toolagent | 23,608 | **55.3%** | 59% | **0.58** |

    ⚠️ **真實 α 落在 0.37–0.58，是我先前合成掃描（0.4–1.5）的最低端**，
    而那正是 Oracle headroom 最大的區域。所以合成掃描不但涵蓋了真實情況，
    還把大部分算力花在比真實更「集中」（headroom 更小）的區域——**偏保守**。
    但這仍是事後的比對，正式結果要用真實 trace 直接跑。
    """
    p = TRACES / f"{name}_trace.jsonl"
    if not p.exists():
        raise SystemExit(
            f"🔴 找不到 {p}\n"
            f"   下載：curl -L -o {p} https://raw.githubusercontent.com/"
            f"kvcache-ai/Mooncake/main/FAST25-release/traces/{name}_trace.jsonl")
    # 🔴 2026-08-31 修正：Mooncake 的 hash_ids 是 **512-token 的 block**，
    #    不是本模擬器的 16-token block。實測（input_length / len(hash_ids)）：
    #        conversation 中位 496.3、平均 483.3、5-95% 416-511
    #        toolagent    中位 487.9、平均 482.3、5-95% 444-510
    #    先前直接把每個 hash_id 當成一個 block，造成三重低估：
    #      1. 工作集少算 32 倍
    #      2. **每個 block 的絕對位置少算 32 倍** -> 重算成本被算得太便宜
    #         （式 eq:recompute-position 是位置的線性函數），
    #         於是 DROP 這個動作看起來比實際划算太多
    #      3. 請求長度中位數 6,909 token 被當成 216 token
    #    所有 2026-08-31 16:30 之前的 trace 類數字都受影響，已作廢重跑。
    #
    #    修法：每個 hash_id 展開成 MOONCAKE_BLOCK // BLOCK = 32 個連續 block，
    #    並依 input_length 裁掉最後一塊多出來的部分（最後一塊通常不滿 512）。
    #    這是**資料的正確解碼**，不是假設。
    exp = MOONCAKE_BLOCK // BLOCK
    out = []
    _MOONCAKE_OUTPUTS[name] = outs = []
    for i, line in enumerate(p.open()):
        if limit and i >= limit:
            break
        rec = json.loads(line)
        want = max(1, -(-int(rec["input_length"]) // BLOCK))   # ceil
        out.append([h * exp + k for h in rec["hash_ids"]
                    for k in range(exp)][:want])
        outs.append(int(rec.get("output_length", 0)))
    return out


# 每條 trace 的輸出長度，由 mooncake_trace() 順便填好。
# 🔴 decode 佔真實負載的 48.8%（toolagent）／53.8%（conversation）的時間，
#    而放置決策對它無能為力——decode 期間該請求的 KV 必須整份在 GPU 裡。
#    模擬器若只算 prefill，回報的 headroom 會是端到端值的兩倍。
_MOONCAKE_OUTPUTS: dict[str, list[int]] = {}


def mooncake_outputs(name: str) -> list[int]:
    """該條 trace 每個請求的輸出（生成）token 數。需先呼叫 mooncake_trace()。"""
    if name not in _MOONCAKE_OUTPUTS:
        mooncake_trace(name)
    return _MOONCAKE_OUTPUTS[name]


def trace_duration_s(name: str) -> float | None:
    """trace 的真實牆鐘時長（秒）。用來把「寫了幾個 block」換算成頻寬需求。

    沒有這個就無法判斷一個策略在真機上跑不跑得動：
    寫入次數再多，攤在一小時上跟攤在一秒上是兩回事。
    """
    p = TRACES / f"{name}_trace.jsonl"
    if not p.exists():
        return None
    ts = [json.loads(l)["timestamp"] for l in p.open()]
    return (max(ts) - min(ts)) / 1000.0


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


def longctx_trace(n_docs: int, doc_blocks: int, tail_blocks: int,
                  n_requests: int, alpha: float, seed: int) -> list[list[int]]:
    """長上下文的合成流量：**一份長的共用文件 + 每個請求各自不同的尾巴**。

    `zipf_trace` 產生的是「整份文件」的請求，重用不是 0% 就是 100%。
    真實的長上下文服務不長那樣——典型形態是一份很長的文件（合約、程式庫、
    對話歷史）被反覆查詢，每次附上不同的問題。共用的是前綴，不同的是尾巴。
    這正是 prefix cache 的目標情境，也是論文要談的。

    參數與可觀察量的關係（重用率是 headroom 天花板的唯一決定因素）：

        總存取     = n_requests × (doc_blocks + tail_blocks)
        不重複     = 實際碰到的文件數 × doc_blocks + n_requests × tail_blocks
        重用率     = 1 − 不重複 / 總存取

    所以要瞄準某個重用率，就調 `tail_blocks`。這比調 alpha 直觀得多，
    也讓「我們假設了什麼」變成一個可以寫進論文的數字。

    ⚠️ 這是合成資料。真實 trace（Mooncake）最長只有 126,195 token，
       沒有 512K 的資料可以驗證這個形態。用它產生的結果必須標明
       `synthetic=longctx` 與所用的重用率。
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

    # 尾巴用獨立的 id 空間，保證每個請求的尾巴都是新的
    tail_base = n_docs * doc_blocks
    out = []
    for j in range(n_requests):
        d = pick()
        body = [d * doc_blocks + b for b in range(doc_blocks)]
        tail = [tail_base + j * tail_blocks + b for b in range(tail_blocks)]
        out.append(body + tail)
    return out


def reuse_rate(trace: list[list[int]]) -> float:
    """實際的重用率 = 1 − 不重複 block 數 / 總存取數。"""
    acc = sum(len(r) for r in trace)
    return 1.0 - len({b for r in trace for b in r}) / max(1, acc)


# ────────────────────────── 策略 ──────────────────────────

class Sim:
    """單一策略在 trace 上的模擬。回傳總成本（ms）與各階命中次數。"""

    def __init__(self, cm: CostModel, gpu_blocks: int, cpu_blocks: int,
                 ssd_blocks: int, decode: dict | None = None):
        self.cm = cm
        self.cap = {"gpu": gpu_blocks, "cpu": cpu_blocks, "ssd": ssd_blocks}
        # decode 的成本模型（load_decode_model 的回傳）。None = 只模 prefill。
        self.decode = decode

    def decode_ms(self, n_blocks: int, out_tokens: int) -> float:
        """一個請求的 decode 總成本。

        🔴 這一項**與放置無關**：decode 期間該請求的 KV 必須整份在 GPU 裡，
           每一步都要讀完，沒有搬到 CPU/SSD 或丟掉重算的自由度。
           所以它對每個策略都是同一個常數，只會**稀釋** headroom。
        """
        if not self.decode or out_tokens <= 0:
            return 0.0
        return out_tokens * (self.decode["decode_base_ms"]
                             + self.decode["decode_ms_per_block"] * n_blocks)

    # -- 系統語意的兩個共用模型 -----------------------------------
    @staticmethod
    def _gap_index(req, gpu, cpu, ssd, enabled: bool) -> int:
        """回傳「第一個在任何一階都找不到的 block」在請求中的序號。

        🔴 前綴語意：vLLM 的 cache lookup 是**連續前綴**，不是任意集合。
           GPU 端的 `find_longest_cache_hit` 與卸載端的
           `_lookup_complete_chunks`（docstring 明寫 "prefix lookup"）
           都只回傳一個 **token 數**——一個 token 數表達不了
           「block 0,1,4,5 命中、2,3 沒有」這種形狀。
           所以第一個缺口之後的 block **全部要重算**，即使它們還躺在 CPU 裡。

           先前的模型把每個 block 當成獨立事件，系統性**低估**了
           full_gpu 的成本，因而低估 headroom。
           這個修正對 Oracle 與所有 baseline 一致套用。
        """
        if not enabled:
            return len(req)
        for j, b in enumerate(req):
            if b not in gpu and b not in cpu and b not in ssd:
                return j
        return len(req)

    @staticmethod
    def _flush(compute: float, transfer: float, prev_compute: float,
               prefetch: bool) -> tuple[float, float]:
        """把一個請求的「計算」與「傳輸」合成關鍵路徑上的時間。

        🔴 預取：卸載連接器的取回是**非同步**的——vLLM 的
           `OffloadingConnectorScheduler` 在排程階段就發出 load，
           與前一個請求的 forward 重疊。原本的模型在存取當下才記傳輸成本，
           等於假設預取從不發生，**高估**了所有用到 CPU/SSD 的策略。

           重疊的上界取「前一個請求的計算時間」：計算佔 SM、傳輸走
           PCIe/NVMe，硬體不同可以並行；而預取所需的 GPU 空間來自
           前一個請求已消化完的 block（prefill 每個 block 只讀一次）。
           這是**上界不是實測**，開啟時結果要在論文裡明確標註。
        """
        if prefetch:
            return compute + max(0.0, transfer - prev_compute), compute
        return compute + transfer, compute

    # -- 線上策略 ---------------------------------------------------
    def run_online(self, trace: list[list[int]], policy: str,
                   use_cpu: bool, use_ssd: bool,
                   split_at: int | None = None,
                   prefix_semantics: bool = True,
                   prefetch: bool = False,
                   per_request: bool = False,
                   outputs: list[int] | None = None) -> dict:
        """split_at 給定時，另外回報第 split_at 個請求之後的成本（warm 段）。

        ⚠️ 為什麼需要這個：M3 的實測量的是**warm 那一輪**的 TTFT，
        模擬若回報兩輪加總，就是拿不同的量在比對，驗證會失去意義。
        2026-08-30 第一版驗證就犯了這個錯（實測比 9.0，模擬比 1.87，
        看起來像模擬器壞掉，其實是比錯東西）。

        prefix_semantics：見 `_gap_index` 的說明。
        prefetch：見 `_flush` 的說明。兩者都同樣套用在 Oracle 上，
        因為它們是**系統能力**，不是策略能力——只給 Oracle 就是作弊。
        """
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
        prev_compute = 0.0        # 上一個請求的計算時間，供預取重疊用
        writes = {"cpu": 0, "ssd": 0}
        per_req: list[float] = []
        decode_total = 0.0

        def demote(blk: int) -> None:
            # 🔴 寫入計量。模擬的成本模型只有「讀回來」的價格，
            #    **把 block 寫下去是免費的**——但真實硬體不是。
            #    toolagent 若把每個新 block 都寫一份到磁碟，需要持續
            #    7,172 MiB/s；這台機器的 SATA QLC 只有 ~380 MB/s（差 19 倍），
            #    NVMe 標稱 3,000 MB/s 也差 2.4 倍。
            #    所以「寫入次數」本身就是一個可行性判準：
            #    寫入頻寬需求超過裝置能力的策略，在真機上根本跑不出來。
            #    這一項對 tier_fs（無差別下放）的傷害大於 Oracle（選擇性下放），
            #    所以忽略它會**低估** headroom。
            if use_cpu:
                cpu[blk] = None
                cpu.move_to_end(blk)
                writes["cpu"] += 1
                while len(cpu) > self.cap["cpu"]:
                    ev, _ = cpu.popitem(last=False)
                    if use_ssd:
                        ssd[ev] = None
                        writes["ssd"] += 1
                        while len(ssd) > self.cap["ssd"]:
                            ssd.popitem(last=False)

        def admit(blk: int) -> None:
            """放進 GPU，必要時逐出（含 ARC 的 ghost list 記帳）。"""
            nonlocal target_t1
            if policy == "arc":
                if blk in b1:
                    target_t1 = min(self.cap["gpu"],
                                    target_t1 + max(1, len(b2) // max(1, len(b1))))
                    del b1[blk]
                elif blk in b2:
                    target_t1 = max(0, target_t1 - max(1, len(b1) // max(1, len(b2))))
                    del b2[blk]
                if blk not in t1 and blk not in t2:
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

        for ri, req in enumerate(trace):
            warm = split_at is not None and ri >= split_at
            gap = self._gap_index(req, gpu, cpu, ssd, prefix_semantics)
            req_compute = 0.0      # 重算 + GPU 命中：佔用 SM，無法與傳輸重疊
            req_transfer = 0.0     # CPU/SSD 取回：走 PCIe/NVMe，可被預取隱藏
            for pos_i, blk in enumerate(req):
                pos = pos_i * BLOCK
                if pos_i > gap:
                    # 落在前綴缺口之後：即使還在某一層也用不到，一律重算
                    tier_hit, c = "drop", self.cm.cost("drop", pos)
                    cpu.pop(blk, None)
                    ssd.pop(blk, None)
                elif blk in gpu:
                    tier_hit, c = "gpu", self.cm.cost("gpu", pos)
                    gpu.move_to_end(blk)
                    if policy == "arc":
                        if blk in t1:
                            del t1[blk]
                            t2[blk] = None
                        elif blk in t2:
                            t2.move_to_end(blk)
                    hits["gpu"] += 1
                    if warm:
                        warm_hits["gpu"] += 1
                    req_compute += c
                    continue                      # 已在 GPU，不需重新 admit
                elif blk in cpu:
                    tier_hit, c = "cpu", self.cm.cost("cpu", pos)
                    del cpu[blk]
                elif blk in ssd:
                    tier_hit, c = "ssd", self.cm.cost("ssd", pos)
                    del ssd[blk]
                else:
                    tier_hit, c = "drop", self.cm.cost("drop", pos)
                hits[tier_hit] += 1
                if warm:
                    warm_hits[tier_hit] += 1
                if tier_hit in ("cpu", "ssd"):
                    req_transfer += c
                else:
                    req_compute += c
                admit(blk)
            c_req, prev_compute = self._flush(req_compute, req_transfer,
                                              prev_compute, prefetch)
            # decode 接在 prefill 之後，與放置無關（見 decode_ms 的說明）
            d_req = self.decode_ms(len(req), outputs[ri]) if outputs else 0.0
            decode_total += d_req
            c_req += d_req
            total += c_req
            if per_request:
                per_req.append(c_req)
            if warm:
                warm_total += c_req
        out = {"total_ms": total, "hits": hits, "writes": writes,
               "decode_ms": decode_total,
               "prefill_ms": total - decode_total,
               "warm_ms": warm_total, "warm_hits": warm_hits}
        if per_request:
            out["per_request_ms"] = per_req
        return out

    # -- Oracle ----------------------------------------------------
    def run_oracle(self, trace: list[list[int]], use_cpu: bool,
                   use_ssd: bool, prefix_semantics: bool = True,
                   prefetch: bool = False, dest: str = "best",
                   per_request: bool = False,
                   outputs: list[int] | None = None) -> dict:
        """知道未來的最佳放置。

        單階時 Bélády/MIN（逐出下次使用最遠的）是**可證明最優**的。
        多階時我們用**成本感知貪婪**：被逐出的 block 依「下次使用有多近」排序，
        近的留在 CPU、遠的下放 SSD、不再用到的直接丟掉（成本 0）。
        這不保證全域最優，所以它是**下界的 Oracle**——真正的最優只會更好，
        因此用它做 go/no-go 是保守的（不會高估 headroom）。

        ⚠️ Oracle 的逐出**沒有**做成前綴感知。真正的最佳解會刻意讓保留的
        block 構成連續前綴（避免製造缺口），那是逐出與前綴結構的聯合最佳化。
        現在的版本一樣會被缺口懲罰，所以它仍然是下界，headroom 仍為保守估計。
        """
        # dest="best"：兩種目的地規則都跑，取較佳者。
        # 這是合法的——Oracle 的定義是「我們能構造出的最佳**離線**策略」，
        # 而兩種規則都是可實作的策略，取其較佳者仍然可實作。
        # 需要這個安全網是因為：成本感知規則用的是**邊際成本的估計**，
        # 估計不準時可能做出比無腦 cascade 更差的決策
        # （2026-08-31 實測 headroom 掉到 -3.16%）。
        # Oracle 若輸給 baseline，go/no-go 就失去意義。
        if dest == "best":
            outs = {d: self.run_oracle(trace, use_cpu, use_ssd,
                                       prefix_semantics, prefetch, d,
                                       per_request, outputs)
                    for d in ("cascade", "cost-aware", "marginal")}
            k = min(outs, key=lambda d: outs[d]["total_ms"])
            outs[k]["dest_chosen"] = k
            outs[k]["dest_alternatives_ms"] = {d: round(v["total_ms"], 2)
                                               for d, v in outs.items()}
            return outs[k]

        # (請求序號, block, 該 block 在請求中的序號)
        # 第三項不可省：重算成本隨 block 的絕對位置線性成長
        # （式 eq:recompute-position）。先前這裡固定用 0，等於讓 Oracle 永遠
        # 以位置 0 的價格重算，而線上策略付全價——**Oracle 自己作弊**。
        # 2026-08-31 由「壓力 0.5× 時五個策略命中數完全相同、時間卻差 9.7%」
        # 這個矛盾抓出來。
        flat: list[tuple[int, int, int]] = []
        pos_of: list[int] = []      # 每一次存取的**絕對位置**（token）
        # 🔴 tail_of[i]：若第 i 次存取的 block 未命中，這個請求要付的重算總量。
        #    前綴語意下，缺一塊 -> 其後全部重算，所以「丟掉一個 block」的
        #    真實邊際成本不是它自己那一塊，而是**它到請求結尾的整條尾巴**。
        #    先前的成本感知規則只比單一 block，於是做出局部便宜、全域昂貴的
        #    決策——2026-08-31 實測在 SSD=2 TiB 時讓 Oracle 反而**輸給**
        #    tier_fs（headroom -3.16%），那在定義上不可能，就是這個 bug。
        tail_of: list[float] = []
        for ti, req in enumerate(trace):
            n = len(req)
            acc = 0.0
            tails = [0.0] * n
            for k in range(n - 1, -1, -1):
                acc += self.cm.cost("drop", k * BLOCK)
                tails[k] = acc
            for pi, blk in enumerate(req):
                flat.append((ti, blk, pi))
                pos_of.append(pi * BLOCK)
                tail_of.append(tails[pi])
        nxt: dict[int, list[int]] = {}
        for i, (_, blk, _pi) in enumerate(flat):
            nxt.setdefault(blk, []).append(i)
        ptr = {b: 0 for b in nxt}

        gpu: set[int] = set()
        cpu: set[int] = set()
        ssd: set[int] = set()
        heap: list[tuple[float, int]] = []   # (-next_use, blk)，max-heap
        nu_cache: dict[int, float] = {}      # blk -> 入堆時記錄的 next_use
        total = 0.0
        decode_total = 0.0
        per_req: list[float] = []
        hits = {"gpu": 0, "cpu": 0, "ssd": 0, "drop": 0}
        # 逐出的去向。`free` = 該 block 之後再也用不到，丟掉成本 0。
        # 🔴 這個計數是檢驗「多階層有沒有用」的關鍵：若 free 佔 100%，
        #    代表最佳策略永遠找得到免費的犧牲者，CPU/SSD 階對最佳解毫無貢獻，
        #    此時量到的 headroom 全部來自 **GPU 階內的逐出選擇**，
        #    不能拿來支持論文的六階動作空間。
        evict = {"free": 0, "to_cpu": 0, "to_ssd": 0, "swap_cpu": 0, "lost": 0,
                 "drop_by_choice": 0, "swap_ssd": 0}
        # CPU 階：max-heap on next_use（留下次使用最近的）
        # SSD 階：min-heap on 節省量（把最不划算的先換掉）
        # 兩者都用 lazy deletion，否則每次逐出要掃整個集合。
        cpu_h: list[tuple[float, int]] = []
        ssd_h: list[tuple[float, int]] = []
        prev_compute = 0.0

        def next_use(blk: int, now: int) -> float:
            lst = nxt[blk]
            p = ptr[blk]
            while p < len(lst) and lst[p] <= now:
                p += 1
            ptr[blk] = p
            return lst[p] if p < len(lst) else math.inf

        i = -1
        for ri, req in enumerate(trace):
            gap = self._gap_index(req, gpu, cpu, ssd, prefix_semantics)
            req_compute = 0.0
            req_transfer = 0.0
            for pi, blk in enumerate(req):
                i += 1
                pos = pi * BLOCK          # 與 run_online 一致
                if pi > gap:
                    hits["drop"] += 1
                    req_compute += self.cm.cost("drop", pos)
                    cpu.discard(blk)
                    ssd.discard(blk)
                elif blk in gpu:
                    hits["gpu"] += 1
                    req_compute += self.cm.cost("gpu", pos)
                elif blk in cpu:
                    hits["cpu"] += 1
                    req_transfer += self.cm.cost("cpu", pos)
                    cpu.discard(blk)
                elif blk in ssd:
                    hits["ssd"] += 1
                    req_transfer += self.cm.cost("ssd", pos)
                    ssd.discard(blk)
                else:
                    hits["drop"] += 1
                    req_compute += self.cm.cost("drop", pos)
                gpu.add(blk)
                nu = next_use(blk, i)
                nu_cache[blk] = nu
                heapq.heappush(heap, (-nu if nu != math.inf else float("-inf"), blk))

                while len(gpu) > self.cap["gpu"]:
                    # 用 max-heap + lazy deletion 取代 max(gpu, key=...)。
                    # 原本每次逐出要掃整個 GPU 集合：預算 17,117 blocks、
                    # 20 萬次存取 = 35 億次運算，Python 跑不完（2026-08-31 卡死）。
                    # heap 版每次逐出 O(log n)。lazy deletion：條目過期就丟掉重取。
                    while heap:
                        negu, blk_h = heap[0]
                        if blk_h not in gpu or -negu != nu_cache.get(blk_h):
                            heapq.heappop(heap)      # 過期條目
                            continue
                        break
                    if not heap:
                        victim = next(iter(gpu))
                    else:
                        victim = heapq.heappop(heap)[1]
                    gpu.discard(victim)
                    j = next_use(victim, i)
                    if j is math.inf:
                        evict["free"] += 1
                        continue                   # 不再用到 -> 丟掉，成本 0

                    # 🔴 成本感知的目的地選擇（dest="cost-aware"）
                    #
                    # 舊版（dest="cascade"）無條件往下推：CPU 有位子就放 CPU、
                    # 否則放 SSD。但 SSD 是**固定** 5.536 ms/block，而丟掉重算是
                    # 4.008 + 0.00021×位置——位置 < 7,278 token 時**丟掉比放 SSD 便宜**。
                    # 真實 trace 的中位請求只有 6,352 token，整段都在交叉點以下，
                    # 所以舊版把大量 block 塞進 SSD，之後用比重算更貴的價格讀回來。
                    # 那不是「成本感知貪婪」，那只是 cascade。
                    #
                    # 這一項讓 Oracle 變強 → headroom 上升。因為 NO-GO 判定
                    # 依賴「Oracle 已經夠強」，這個修正對 NO-GO 的可信度是必要的。
                    # 丟掉的邊際成本：前綴語意下是「整條尾巴」，
                    # 不是單一 block。這是上界（尾巴也可能因為別的缺口而
                    # 本來就要重算），所以它讓 Oracle **偏向保留**——
                    # 保守的方向，不會高估 headroom。
                    # 🔴 2026-09-05：dest="marginal" 是第三條規則。
                    #
                    # cost-aware 用 tail_of[j]（整條尾巴）當丟掉的邊際成本。
                    # 前綴語意下缺一塊會使其後全部重算，所以尾巴是**上界**——
                    # 但那條尾巴是該請求所有缺塊**共用**的，整條算給每一個 block
                    # 等於重複計價，門檻因而被壓得太低（什麼都想留）。
                    # 後果是可量到的：qwen-awq/NVMe 下，多給 oracle 一個
                    # 512 GiB 的 SSD 階，它反而**慢 3.3%**（21.04M vs 20.35M ms）——
                    # 最佳策略不可能因為多了一個可用資源而變差。它把 25 萬個 block
                    # 寫上 SSD 再用 10.245 ms 讀回，而重算只要 3.5–4 ms。
                    # marginal 改用單一 block 的重算成本（下界），
                    # 而 dest="best" 取三條規則的較佳者，所以 oracle 只會更緊。
                    drop_c = (tail_of[j] if (prefix_semantics and dest != "marginal")
                              else self.cm.cost("drop", pos_of[j]))
                    if dest == "cascade":
                        cpu_ok = use_cpu
                        ssd_ok = use_ssd
                    else:
                        cpu_ok = use_cpu and self.cm.cpu < drop_c
                        ssd_ok = use_ssd and self.cm.ssd < drop_c

                    if cpu_ok and len(cpu) < self.cap["cpu"]:
                        cpu.add(victim)
                        heapq.heappush(cpu_h, (-j, victim))
                        evict["to_cpu"] += 1
                    elif cpu_ok and cpu:
                        # CPU 滿：換掉「下次使用最遠」的那個（CPU 永遠比重算便宜，
                        # 所以這裡純粹是 Bélády，跟成本無關）
                        while cpu_h:
                            nj, bh = cpu_h[0]
                            if bh not in cpu:
                                heapq.heappop(cpu_h)
                                continue
                            break
                        if cpu_h and -cpu_h[0][0] > j:
                            far = heapq.heappop(cpu_h)[1]
                            cpu.discard(far)
                            cpu.add(victim)
                            heapq.heappush(cpu_h, (-j, victim))
                            evict["swap_cpu"] += 1
                            victim = far          # 被擠下來的往下一階試
                            j = next_use(far, i)
                            if j is math.inf:
                                evict["free"] += 1
                                continue
                            drop_c = (tail_of[j]
                                      if (prefix_semantics and dest != "marginal")
                                      else self.cm.cost("drop", pos_of[j]))
                            ssd_ok = (use_ssd and
                                      (dest == "cascade" or self.cm.ssd < drop_c))
                        # else: victim 留著往 SSD 試

                    if victim in cpu:
                        continue
                    save = drop_c - self.cm.ssd      # 放 SSD 相對丟掉省多少
                    if ssd_ok and len(ssd) < self.cap["ssd"]:
                        ssd.add(victim)
                        heapq.heappush(ssd_h, (save, victim))
                        evict["to_ssd"] += 1
                    elif ssd_ok and ssd:
                        while ssd_h:
                            sv, bh = ssd_h[0]
                            if bh not in ssd:
                                heapq.heappop(ssd_h)
                                continue
                            break
                        if ssd_h and ssd_h[0][0] < save:
                            out = heapq.heappop(ssd_h)[1]
                            ssd.discard(out)
                            ssd.add(victim)
                            heapq.heappush(ssd_h, (save, victim))
                            evict["swap_ssd"] += 1
                        else:
                            evict["lost"] += 1
                    elif use_cpu or use_ssd:
                        # 有階層可用，但成本上不划算 -> 主動丟掉
                        evict["drop_by_choice" if dest != "cascade" else "lost"] += 1
                    else:
                        evict["lost"] += 1
            c_req, prev_compute = self._flush(req_compute, req_transfer,
                                              prev_compute, prefetch)
            d_req = self.decode_ms(len(req), outputs[ri]) if outputs else 0.0
            decode_total += d_req
            c_req += d_req
            total += c_req
            if per_request:
                per_req.append(c_req)
        out = {"total_ms": total, "hits": hits, "evict": evict,
               "decode_ms": decode_total, "prefill_ms": total - decode_total,
               "writes": {"cpu": evict["to_cpu"] + evict["swap_cpu"],
                          "ssd": evict["to_ssd"] + evict["swap_ssd"]}}
        if per_request:
            out["per_request_ms"] = per_req
        return out


# ────────────────────────── 驗證 ──────────────────────────

def validate(cm: CostModel, sem: dict | None = None) -> dict:
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
        sem = sem or {}
        s_full = sim.run_online(trace, "lru", False, False, split_at=n, **sem)
        s_lru = sim.run_online(trace, "lru", True, False, split_at=n, **sem)
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
    ap.add_argument("--trace", choices=["conversation", "toolagent", "mooncake"],
                    help="改用 Mooncake 的真實生產 trace，取代合成的 Zipf。"
                         "給了這個就忽略 --alpha")
    ap.add_argument("--trace-limit", type=int, default=None,
                    help="只取前 N 個請求（trace 很大時用）")
    ap.add_argument("--pressure", type=float, nargs="*", default=None,
                    help="掃『工作集 / GPU 預算』比。合成工作負載會依此縮放，"
                         "使每個比例都有意義。這是 headroom 的主要自變數")
    ap.add_argument("--docs", type=int, default=64)
    ap.add_argument("--doc-tokens", type=int, default=4096)
    ap.add_argument("--requests", type=int, default=0,
                    help="請求數。0 = 自動取 10×文件數。"
                         "🔴 固定 400 會讓高壓力設定名不副實："
                         "『pressure:8x』配了 535 篇文件，但 400 個請求只碰得到"
                         "其中一部分，實際壓力只有 2.8×。標籤是旋鈕設定值，"
                         "實際值一律以 realized_pressure_x 欄位為準")
    ap.add_argument("--model", default="llama-bf16", choices=list(MODEL_PROFILES),
                    help="模型剖面：一次鎖定「GPU 預算 + KV 每 token 大小 + "
                         "成本模型來源」三者，避免混用。目前只有 llama-bf16 "
                         "有對應的 M2 成本量測")
    ap.add_argument("--gpu-tokens", type=int, default=None,
                    help="覆寫剖面的 GPU KV 預算（token）。"
                         "⚠️ 覆寫成別的模型的容量就是混用，只在做敏感度分析時用")
    ap.add_argument("--cpu-gib", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--lookup", choices=["prefix", "per-block"], default="prefix",
                    help="cache 查詢語意。prefix=照 vLLM 實際行為（缺口之後全部重算）；"
                         "per-block=舊模型，每個 block 獨立命中（會低估 baseline 成本）")
    ap.add_argument("--prefetch", action="store_true",
                    help="允許 CPU/SSD 取回與前一個請求的計算重疊（非同步 load 的上界）。"
                         "同時套用於 Oracle 與所有 baseline")
    a = ap.parse_args()
    SEM = {"prefix_semantics": a.lookup == "prefix", "prefetch": a.prefetch}
    print(f"[語意] lookup={a.lookup}  prefetch={a.prefetch}")

    prof = profile(a.model)
    cm = load_cost_model(a.device, require_model_key=prof["cost_model_key"])
    print(f"=== 模型剖面 {a.model} ===")
    print(f"  GPU KV 預算       : {prof['gpu_kv_tokens']:,} tokens")
    print(f"  KV 每 token 大小  : {prof['kv_bytes_per_token'] / 1024:.0f} KiB")
    print(f"  成本模型來源      : {prof['source']}")
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
        v = validate(cm, SEM)
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

    gpu_tokens = a.gpu_tokens or prof["gpu_kv_tokens"]
    if a.gpu_tokens and a.gpu_tokens != prof["gpu_kv_tokens"]:
        print(f"  ⚠️ 覆寫 GPU 預算 {prof['gpu_kv_tokens']:,} → {a.gpu_tokens:,}"
              f"（敏感度分析；不是剖面的自洽值）")
    gpu_blocks = gpu_tokens // BLOCK
    cpu_blocks = int(a.cpu_gib * 1024**3 // (BLOCK * prof["kv_bytes_per_token"]))
    doc_blocks = a.doc_tokens // BLOCK

    print(f"\n=== Oracle 設定 ===")
    print(f"  GPU 預算   : {gpu_tokens:,} tokens = {gpu_blocks:,} blocks（M3 實測值）")
    print(f"  CPU 預算   : {a.cpu_gib} GiB = {cpu_blocks:,} blocks")
    print(f"  工作負載   : {a.docs} 個文件 × {a.doc_tokens} tokens，"
          f"{a.requests} 個請求，Zipf")
    print(f"  工作集     : {a.docs * doc_blocks:,} blocks "
          f"({a.docs * doc_blocks / gpu_blocks:.1f}× GPU 預算)")

    rows = []
    if a.pressure:
        # 🔴 為什麼需要這個：headroom 的主要自變數是「工作集 / GPU 預算」，
        # 不是 Zipf 的 α。2026-08-31 實測——把預算從 48,128 換成 273,872 之後，
        # 合成工作負載（64 文件 × 4,096 token = 16,384 blocks）竟然小於預算
        # （17,117 blocks），整個塞得進 GPU、完全不逐出，三個 α 因此給出
        # 完全相同的 9.66%——那是**假的 headroom**。
        # 固定文件大小、依目標比例調整文件數，讓每個點都真的有記憶體壓力。
        cases = [(f"pressure:{r:g}x", None) for r in a.pressure]
    else:
        cases = ([("trace:" + a.trace, None)] if a.trace
                 else [(f"zipf:{al}", al) for al in a.alpha])
    for label, alpha in cases:
        if a.pressure:
            ratio = float(label.split(":")[1].rstrip("x"))
            n_docs = max(2, round(ratio * gpu_blocks / doc_blocks))
            n_req = a.requests or max(400, 10 * n_docs)
            trace = zipf_trace(n_docs, doc_blocks, n_req, 0.9, a.seed)
            uniq = len({b for r in trace for b in r})
            print(f"\n[oracle] 名目壓力 {ratio:g}×：{n_docs} 個文件 × "
                  f"{a.doc_tokens} token、{n_req:,} 個請求；"
                  f"實際碰到 {uniq:,} 個不重複 block / 預算 {gpu_blocks:,} blocks"
                  f" = **實際壓力 {uniq / gpu_blocks:.1f}×**")
        elif a.trace:
            n_req = 0
            trace = mooncake_trace(a.trace, a.trace_limit)
            nb = len({b for r in trace for b in r})
            print(f"\n[oracle] 真實 trace「{a.trace}」：{len(trace):,} 個請求，"
                  f"{sum(len(r) for r in trace):,} 次 block 存取，"
                  f"{nb:,} 個不重複 block（工作集 = {nb / gpu_blocks:.1f}× GPU 預算）")
        else:
            n_req = a.requests or 400
            trace = zipf_trace(a.docs, doc_blocks, n_req, alpha, a.seed)
        sim = Sim(cm, gpu_blocks, cpu_blocks, ssd_blocks=10**9)
        res = {
            "full_gpu": sim.run_online(trace, "lru", False, False, **SEM),
            "cpu_lru": sim.run_online(trace, "lru", True, False, **SEM),
            "cpu_arc": sim.run_online(trace, "arc", True, False, **SEM),
            "tier_fs": sim.run_online(trace, "lru", True, True, **SEM),
            "oracle": sim.run_oracle(trace, True, True, **SEM),
        }
        best_base = min((k for k in res if k != "oracle"),
                        key=lambda k: res[k]["total_ms"])
        head = 100 * (res[best_base]["total_ms"] - res["oracle"]["total_ms"]) \
            / res[best_base]["total_ms"]
        verdict = ("GO" if head > 15 else
                   "MARGINAL_ASK_HUMAN" if head >= 5 else "NO_GO")

        print(f"\n--- {label} ---")
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
                "alpha": alpha if alpha is not None else "",
                "workload": label, "policy": k,
                "total_ms": round(v["total_ms"], 2),
                "gpu_hits": v["hits"]["gpu"], "cpu_hits": v["hits"]["cpu"],
                "ssd_hits": v["hits"]["ssd"], "recompute": v["hits"]["drop"],
                "model_profile": a.model,
                "gpu_budget_tokens": gpu_tokens, "cpu_budget_gib": a.cpu_gib,
                "docs": a.docs, "doc_tokens": a.doc_tokens,
                "requests": len(trace), "seed": a.seed,
                "lookup": a.lookup, "prefetch": int(a.prefetch),
                # 🔴 workload 標籤是旋鈕的**設定值**；實際壓力以此欄為準。
                #    兩者在高壓力時差很多（名目 8× → 實際 2.8×）。
                "unique_blocks": len({b for r_ in trace for b in r_}),
                "realized_pressure_x": round(
                    len({b for r_ in trace for b in r_}) / gpu_blocks, 3),
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


# 必須在 Sim / load_cost_model / MODEL_PROFILES 都定義完之後才算得出來。
SIM_VERSION = sim_version()
