# 回覆老師 — Tiara 現況盤點與下一步

**日期**：2026-09-19　**寫給**：Jess Chih-Chung Hsu
**產生方式**：全部數字在本機重新量測或從 `results/` 逐檔核對，每一格標出處。沒查到的寫「不存在」，不補、不估。

---

## 0. 在回答三張表之前，有三件事實要先更正

老師的 feedback 建立在我先前給的資訊上。盤點後發現**其中有三項是錯的**，而且都朝「高估自己」的方向錯。先更正，後面的討論才有意義。

### 0.1 設備：不是虛擬機，我手上有 MI300X

我先前說「目前是虛擬機」。實際上這是 **MLSteam 平台上的 Kubernetes container**，直接掛載一張實體 GPU：

> **1 × AMD Instinct MI300X，192 GiB HBM，304 CU，gfx942，PCIe Gen5 ×16**
> 主機是 Gigabyte G593-ZX1-AAX1，`lspci` 看得到 **8 張 MI300X**，本容器分到 1 張。

所以老師說的「如果沒有 MI300X，就不要用規格推算」——**這一點我有設備，不需要推算，而且已經量過了**。平台 B 的所有數字都是這張卡上的真機量測。

但另一個方向要收斂：**容器有資源配額，不是整台機器**。

| 項目 | 容器實際拿到（cgroup） | 主機總量（僅供參考，我拿不到） |
|---|---|---|
| CPU | **32 顆**（`cpu.max = 3200000 100000`） | 192 執行緒（2 × EPYC 9684X 96C） |
| RAM | **434.8 GiB**（`memory.max`，現用 341.5 GiB） | 2,267.5 GiB |
| GPU | **1 × MI300X 192 GiB** | 8 × MI300X |
| `/dev/shm` | **178.8 GiB** ← CPU KV 階的實際上限 | — |

**這張表 `results/hw_mi300x.json` 早就記對了**（`quota.*` 與 `host_view.*` 分開存）。問題只在我對外講的時候混用了。

### 0.2 數字：我給老師的五個數字，有四個在 repo 裡不存在

這是最嚴重的一項。我逐一 grep 過整個 repo（排除 `.git`／HTML／PDF／ipynb）：

| 我跟老師說的 | 核對結果 | 證據 |
|---|---|---|
| 合成 workload **+9.87%** | ❌ **不存在。** 全 repo 沒有任何結果被報成 9.87%。所有 `9.87` 命中都是 CSV 數值欄的巧合子字串（如 `1506929.87`、`42929.877`） | `grep -rn "9\.87"` |
| Mooncake **+0.76%** | ✅ **是真的。** oracle headroom 11.50%，Tiara +0.76%，吃到 headroom 的 6.62% | `PAPER_DELTAS.md:213`、`results/RUNLOG.md:2670` |
| 原始 CacheWise oracle **0%** | ❌ **不存在。** CacheWise 全 repo 只有 4 次命中，**全部是 related-work 文字**（`PAPER_DELTAS.md:196` 明寫它是「**可能**的 trace 來源」）。沒有 CacheWise trace 檔、沒有 loader、沒有任何結果 | `grep -rni "cachewise"` |
| 重排後 oracle **39.57%** | ❌ **不存在。** 唯一的 `39.57` 是一個 **TPOT 值**（`cpu_lru,b-qwen14b-1m,...,8000.601,39.57,9068.983` 的中間欄），不是 headroom。**而且 repo 裡根本沒有 session 重排／shuffle 腳本** | `results/m3_baseline_mi300x/baseline_mi300x.csv:691`；`grep -rln "shuffle\|reorder" code/` 只命中 RULER 的 needle 擺放與訓練樣本打散 |
| 真機 **3,600 requests** inconclusive | ❌ **不存在。** 所有 `3600` 命中都是 **request timeout**（`code/m3_baseline.py:448 REQ_TIMEOUT=3600`）或 DOI。**Tiara 的策略從來沒有在真機端到端跑過**（見 0.3） | `grep -rn "3600"` |

**而且真實數字比我報的好**：合成長上下文的實際結果是 **+15.40%，吃到 oracle headroom 的 81.85%**（`PAPER_DELTAS.md:211`），不是 +9.87%。

> 我不確定那四個數字是從哪裡來的——可能是我記錯，也可能是某次 AI 對話產生後我沒回頭核對。無論原因，**這正是 `CLAUDE.md` 規則 1（不准編造任何數字）要防的失敗模式，而它漏到了跟老師的對話裡。**
> 往後我跟老師報的每個數字，都會附 `檔案:行號`。

### 0.3 Plugin：老師的猜測是對的，而且比老師想的更嚴重

老師說：

> 「我目前看到的可能只是 CPU tier 裡面『要丟誰』，並不是圖上寫的 GPU 16／8／4-bit、CPU、SSD、DROP 全部一起控制。」

**正確，而且要再退一步：連「CPU tier 要丟誰」都不是在真機上跑的。**

- `/mlsteam/workspace/src/vllm-v0.28.0/` 的 `git status` **完全乾淨**，零修改。
- `vllm/v1/kv_offload/cpu/policies/` 只有原廠的 `lru.py`、`arc.py`、`base.py`、`factory.py`。
- 沒有任何名為 `tiara` 的套件被安裝進任何 venv。
- 專案對 vLLM 做的事，只是**用 CLI/JSON 設定原廠的 `OffloadingConnector`** 來跑 baseline（`code/m3_baseline.py:320-341`）。

**Tiara 的所有策略邏輯都活在 Python 模擬器裡**（`code/m4_oracle.py` 的 `class Sim`、`code/m5_policy_sim.py` 的 `run_learned()`），由真機量到的成本常數驅動。

而且 `OPEN_ISSUES.md:126-129`（B7）**自己早就寫下了這個限制**：

> vLLM 的 `CachePolicy` ABC 只有 `get/insert/remove/touch/evict/clear`，
> 也就是只能決定「從 CPU 階逐出哪個 block」。
> **`DROP→重算` 與 GPU 精度降級無法透過這個介面表達**，需要改 `OffloadingSpec` 與 attention 路徑。

所以論文 Fig. 2「Tiara 掛在 vLLM 的 CachePolicy 介面上」是**過度宣稱**，六態動作空間裡有四態掛不進去。

**目前六態動作空間的真實完成度：**

| 動作 | 模擬器 | 真機 | 說明 |
|---|---|---|---|
| GPU 逐出順序（誰先走） | ✅ 已實作已跑 | ❌ | Bélády `m4_oracle.py:1298`；學習式 `m5_policy_sim.py:262` |
| CPU 階留存／逐出 | ✅ 已實作已跑 | ❌ | `m4_oracle.py:1355`、`m5_policy_sim.py:201` |
| 逐出去向 CPU / SSD / DROP | ✅ 已實作已跑 | ❌ | `m4_oracle.py:1343-1408`、`m5_policy_sim.py:201-235` |
| DROP → 重算（成本隨位置線性） | ✅ 已實作已跑 | ❌ | `CostModel.cost("drop", pos)`，`m4_oracle.py:177` |
| **GPU-INT8 降級** | 🟡 **只有平台 A 跑過** | ❌ | `m5_policy_sim.py:326-341`；MI300X 的 35 列全是 `precision=off, downgrades=0` |
| **GPU-FP8／GPU-INT4** | ❌ **從未進入策略** | ❌ | oracle 根本沒有精度階（`m4_oracle.py:89` 註明 `load_precision_tiers()` 模擬路徑不會呼叫）；FP8/INT4 因品質 gate 被刻意排除 |
| 品質約束 ε 綁住策略 | ❌ 只有文件 | ❌ | `OPEN_ISSUES.md:115-118`：ε 只量到 per-dtype，沒有 per-policy |

---

## 1. 老師要的第二張表：設備與實驗

### 1.1 硬體（全部本機實測，2026-09-19）

| 項目 | 實測值 | 量測方式 |
|---|---|---|
| **GPU** | 1 × MI300X，192.0 GiB HBM，304 CU，gfx942 | `rocm-smi`，`torch.cuda` |
| GPU HBM 頻寬 | **3,901 GB/s**（D2D 讀+寫） | 2 GiB buffer × 10 iter |
| GPU 算力 | **658 TFLOP/s**（bf16 GEMM 8192³）／638（16384³） | rocBLAS `torch.mm` |
| **PCIe H2D（pinned）** | **57.6 GB/s** ← CPU→GPU 載入 KV 的真實上限 | 2 GiB pinned copy |
| PCIe D2H（pinned） | 48.6 GB/s | 同上 |
| **CPU** | 32 顆配額（主機 2 × EPYC 9684X 96C） | cgroup `cpu.max` |
| CPU DRAM memcpy | 15.4 GB/s（單執行緒） | numpy `copyto` |
| **RAM 配額** | **434.8 GiB**（現用 341.5）；`/dev/shm` 178.8 GiB | cgroup `memory.max` |
| **本地碟** | Broadcom **GBT3916-MR-32PD**（MegaRAID volume）**不是 NVMe** | `/sys/block/sda/device/model`；`raw_nvme_visible=False` |
| 本地碟 循序讀（8 GiB 檔） | **4,978 MiB/s**（O_DIRECT, bs=4M） | fio — **重現了專案量到的 4,771–5,527** ✅ |
| 本地碟 循序讀（**64 GiB 檔**） | **5,728 MiB/s** | fio — 檔案放大 8 倍**反而更快**，排除快取假象 |
| 本地碟 隨機 128K 讀（64 GiB） | **6,406 MiB/s**（QD=8, 51.2k IOPS） | fio — 隨機**沒有**比循序慢 |
| 本地碟 4K 讀 | 166 MB/s | dd O_DIRECT QD=1（低佇列深度才會掉） |
| **NFS**（`/mlsteam/data/tiara`） | NFSv4.1 over **10 GbE**，rsize/wsize 64K | `findmnt` |
| NFS 循序讀／寫 | **363 / 682 MB/s** | 與專案量到的 **289 / 683** 一致 ✅ |
| NFS 隨機 128K 讀 | **118 MiB/s**（QD=8）← 比循序掉 2.5× | fio |
| ⚠️ Page cache 污染 | 同檔案：O_DIRECT **2.7 GB/s** vs 有 cache **10.8 GB/s**（**4× 膨脹**） | dd 對照 |

**兩個要修正的說法：**

1. **`CLAIM_EVIDENCE_MI300X.md:94` 與 `RUNLOG_MI300X.md:789` 寫「本地 NVMe」是錯的。**
   容器內沒有任何 `/dev/nvme*`，底層是 Broadcom MegaRAID volume。
   而你自己的 `results/hw_mi300x.json` 已經寫了 `storage.raw_nvme_visible = False` ——
   **是散文跟自己的資料矛盾，不是資料錯。** 改成「本地 overlay（Broadcom MegaRAID volume，底層碟型號不可見）」即可。

2. **SSD 階的位置是 `/var/tmp`（容器 overlay），是暫態儲存。**
   真實部署的 SSD 階會是持久化裝置。這要寫進 threats to validity。

> **附帶一個被推翻的假設（記錄下來當方法學示範）**
> 我一開始懷疑「5,392 MiB/s 是 RAID 控制器快取，因為測試檔只有 8 GiB」。
> 決定性測試：把檔案放大到 **64 GiB**（遠超任何控制器快取）重量。
> 結果是 **5,728 MiB/s，比 8 GiB 的 4,978 還快**；隨機 128K QD=8 也有 **6,406 MiB/s**。
> **假設被推翻。磁碟數字是真的，而且原本的量測偏保守。**
> 這台的本地碟是 32 顆實體碟的 Broadcom MegaRAID 陣列（`GBT3916-MR-32PD`），確實很快。

### 1.2 🔴 最重要的發現：傳輸成本不是硬體常數，同一條 PCIe 上跨模型差 16.9 倍

這一節用的是 **vLLM 自己的計數器**（`vllm:kv_offload_load_bytes` / `load_time`），
資料來自專案既有的 `results/m2_harness_mi300x/connector_transfer_mi300x.csv`（30 列），
**不含端到端雜訊、不含排程、不含我的推論**。對照組是我本 session 實測的硬體上限。

| 模型 | KV KiB/token | block 大小 | **CPU 階載入** | **佔 PCIe 上限** | ms/block | n | 組內變異 |
|---|---|---|---|---|---|---|---|
| b-qwen3-30b-a3b (MoE) | 96 | 1.50 MiB | **38.28 GB/s** | **66.5%** | 0.041 | 3 | 1.7% |
| b-llama8b | 128 | 2.00 MiB | 4.47 GB/s | 7.8% | 0.469 | 6 | 1.7% |
| b-ultralong8b-1m | 128 | 2.00 MiB | 4.40 GB/s | 7.6% | 0.477 | 5 | 2.9% |
| b-seedoss36b | 256 | 4.00 MiB | 4.36 GB/s | 7.6% | 0.963 | 5 | 4.1% |
| b-qwen7b-1m | 56 | 0.88 MiB | **2.27 GB/s** | **3.9%** | 0.405 | 11 | 4.2% |

**對照：本 session 實測 PCIe H2D (pinned) = 57.6 GB/s。**

**再現性（我對這個發現做了自我攻擊，它扛住了）：**
30 次量測，跨 **5 個模型 × 多個 run_id × 多天 × 兩個實驗階段（`m2_retrieval` 與 `m3_baseline`）× 兩種逐出策略（lru / arc）× 載入量 2.6–20.5 GiB**。
**組內變異全部 < 4.2%，組間差 17.1 倍。** 這不是雜訊，是系統性的模型相依效應。

三件事同時成立：

1. **同一台機器、同一條 PCIe Gen5 ×16、同一個 vLLM，載入頻寬跨模型差 17.1 倍**（2.27 ↔ 38.28 GB/s）。
   **PCIe 不會因為換模型而變慢——所以這不是硬體性質。**
2. 出現**三個不同的區間**，不是連續變化：
   - **38 GB/s**（66% PCIe）：只有 Qwen3-30B-A3B，唯一的 **MoE** 模型
   - **≈4.4 GB/s**（7.6%）：llama8b / ultralong8b / seedoss36b ——
     這三個之間 **ms/block 與 block 大小成正比**（2.00 MiB→0.469 ms，4.00 MiB→0.963 ms，剛好 2 倍），
     所以它們是**卡在同一道 4.4 GB/s 的天花板上**
   - **2.27 GB/s**（3.9%）：Qwen2.5-7B-1M，block 最小（0.88 MiB）。
     若也在 4.4 GB/s 天花板上，它應該是 0.20 ms/block，實際是 0.405 —— **小 block 另外被懲罰了 2 倍**
3. 四個模型只吃到 PCIe 的 **3.9–7.8%**。

> **一個被推翻的假設（照實記錄）**
> 我先假設是「每次 block 傳輸的固定開銷」主導全部五個模型。
> 檢驗：若成立，`ms/block` 應該幾乎與 block 大小無關。
> 實際：block 大小變動 **4.6×**，`ms/block` 變動 **23.4×** → **對全體而言假設被推翻**。
> 修正後的讀法（見上）：**不是單一機制，是三個區間**，而區間之間的差異**機制未知**。

**為什麼這件事會改寫整篇論文：**

論文現在的核心 claim 是「**κ 跨硬體變動 32 倍**，所以單一策略不適用所有硬體」。
但上面這張表顯示：**光是傳輸項本身，在同一個硬體上跨模型就變動 16.9 倍，而且變動來源是軟體不是物理。**

也就是說：
- 「跨硬體」可能**選錯了自變數**——真正的變異來自「這個 (模型, 實作路徑) 組合有多會用硬體」
- Tiara 的成本模型 **每一個決策常數都是從這個被軟體壓扁的數字推出來的**
- MI300X 上學習式策略會輸，**可能不是因為「階間差距被硬體壓縮」（現在論文的解釋），而是因為傳輸項被軟體壓縮了 13 倍**，讓所有階看起來一樣貴

**這是目前最高價值的一條線索**，因為它同時是：
- 一個可能獨立成立的系統貢獻：多階 KV 系統的瓶頸在軟體路徑，而且以「模型相依」的方式呈現
- 一個會推翻我自己現有解釋的風險 → 正好符合老師要的「設計實驗看它會不會被推翻」
- **只需要一張 GPU 就能做完**，設備完全足夠

**下一步（在寫進論文之前必須做）：**
1. 找出 Qwen3-30B-A3B 為什麼能到 66% 而其他四個只有 7.6%（MoE？pinned vs pageable？batching？`/dev/shm` 的額外複製？）
2. 確認 4.4 GB/s 天花板是什麼（用 vLLM 的 profiler 或 `nsys`/`rocprof` 追一次 offload 路徑）
3. 如果能修掉，重量 κ——**κ_ssd 可能從 0.72 翻成 > 2，整個 SSD 階的結論會反轉**

### 1.3 Baseline 重現狀況

| Baseline | 平台 A（7×3090） | 平台 B（MI300X） | 跑不了的原因 |
|---|---|---|---|
| `full_gpu` | ✅ | ✅ | — |
| `cpu_lru`（vLLM 原廠） | ✅ | ✅ | — |
| `cpu_arc`（vLLM 原廠） | ✅ | ✅ | — |
| `tier_fs`（CPU+disk） | ✅ | ✅ | ⚠️ 平台 A 的 CPU 階 24 GiB ≥ 工作集 → **磁碟階可能一次都沒用到**，兩平台不可直接比較（`OPEN_ISSUES` D1） |
| **LMCache** | 🟡 跑了但**殘廢** | ❌ **NOT_MEASURED** | wheel 缺編譯好的 `c_ops`/`cuda_ops`；B 上依賴 `cupy-cuda13x`、`cuda-python`、`cufile-python`，**ROCm 裝不起來** |
| **AdaptCache** | ❌ | ❌ | **從未嘗試。** repo 內 0 次命中 |
| **Cake** | ❌ | ❌ | **從未嘗試。** repo 內 0 次命中 |
| **Bidaw** | ❌ | ❌ | **從未嘗試。** repo 內 0 次命中 |
| **Strata** | ❌ | ❌ | **從未嘗試。** repo 內 0 次命中 |
| **MTDS** | ❌ | ❌ | 只有文獻註記，論文付費牆 |
| KVP / ForesightKV / LookaheadKV | ❌ | ❌ | **刻意排除**：未釋出權重（`EXPERIMENT_PLAN.md:486`） |

**老師這一點完全命中**：目前比的四個 baseline 裡，三個是 vLLM 原廠的 LRU/ARC/tiering，一個是殘廢的 LMCache。**沒有一個是這個題目的 SOTA。**

### 1.4 統計嚴謹度（這是最大的方法學漏洞）

| 類別 | 現況 |
|---|---|
| 真機量測（M1/M2/M3/M5） | ✅ 有 `--repeats`，報 median + range；品質有 95% CI 與 `distinguishable` flag |
| **模擬結果（所有 oracle / headroom / Tiara 數字）** | 🔴 **每一個都是 single run，`--seed 1234`，沒有重複、沒有變異數、沒有 CI** |
| 污染偵測 | ✅ `gpu_guard.py` 全程監看，`CONTAMINATED` 的 run 時間欄作廢 |
| 自我稽核 | ✅ `verify_results_b.py` 77/77 涵蓋，但有 **5 FAIL / 2 WARN** 待解 |

**所有 headroom 百分比都沒有誤差棒。** 老師問「+0.76% 是不是還在 noise 裡」——**我現在無法回答，因為從來沒量過 noise。** 這正是下面第 3 節要做的事。

---

## 2. 老師要的第三件事：會殺死 Tiara 的實驗

老師給的順序是對的，我照做，而且加一個前置步驟（因為 0.1 發現配額比我以為的小）：

### 步驟 0（新增）— 先算容量，確認 eviction 真的會發生

在跑任何策略之前，先做一件純算術的事：**工作集 vs 各階容量**。

```
每 token 的 KV（bf16）= 2 × layers × kv_heads × head_dim × 2 bytes
  Qwen2.5-7B-1M  (28層, 4 kv_head, 128) =  56 KiB/token
  Qwen2.5-14B-1M (48層, 8 kv_head, 128) = 192 KiB/token
```

**如果 trace 的工作集 < GPU 階 + CPU 階，那 oracle headroom 必然精確等於 0**，而且那不是「policy 不好」，是「eviction 從來沒被觸發」。
平台 A 的 `tier_fs` 已經踩到這個坑（`OPEN_ISSUES` D1：CPU 階 24 GiB ≥ 工作集 16 GiB）。
**這個檢查要寫成 assertion 放進 loader，每個 trace 跑之前自動擋。**

### 步驟 1 — A/A test（✅ 本次已執行，而且結論改變了實驗設計）

**做法**：保留真實 trace 的原始順序，同一設定（`--model b-llama8b --trace toolagent --trace-limit 4000 --cpu-gib 96`）只換 seed，跑 8 次。

**結果**：

| seed | 1234 | 99 | 7 | 42 | 2026 | 31337 | 555 | 8888 |
|---|---|---|---|---|---|---|---|---|
| headroom | 20.7% | 20.7% | 20.7% | 20.7% | 20.7% | 20.7% | 20.7% | 20.7% |

**變異 = 0，完全決定性。** 另外我把一次獨立重跑的完整輸出與原始 `stdout.log` 逐行比對，
五個 policy 的 total_ms 全部逐位元相同（`full_gpu 1,857,677`、`oracle 1,370,923`…）。

> **所以「換 seed 跑 N 次」對這個專案量不到任何東西。**
> 真實 trace 走原始順序時，seed 只影響合成工作負載的生成，不影響 trace 重播。
> 這推翻了我原本寫的 A/A 設計——**先做這個實驗，省下了後面所有基於它的分析。**

### 步驟 1'（修正後的正確設計）— 對成本常數做不確定性傳播

模擬器本身沒有雜訊，所以 headroom 的不確定性**全部來自輸入的成本常數**。
而那些常數是真機量的，**本來就有變異**（M2 有 `--repeats`，記了 median 與 range）。

**正確的 kill experiment 是這樣：**

1. 從 `results/m2_harness_mi300x/` 取出每個成本常數的**實測分布**
   （`recompute_base`、`recompute_slope`、`cpu_ms_per_block`、`ssd_ms_per_block`）。
2. 在那些分布上取樣（或直接用 min/max 做最壞情況），**每組常數重跑一次模擬器**（單次約 25 秒，很便宜）。
3. 得到 **headroom 的分布與 95% CI**，以及 **Tiara vs `tier_fs` 差值的 CI**。

**判準（事先講好，不事後調整）：**
- 若 **Tiara − tier_fs 的 95% CI 跨過 0** → 這個結果不存在，claim 要改。
- 若 headroom 的 CI 下界 < 5% → 依 `CLAUDE.md` 規則 4 判 NO-GO。

這一步會直接檢驗 **+0.76%**（平台 A）——以目前 §1.2 發現的傳輸項可以差 16.9 倍來看，
**+0.76% 幾乎確定會被成本常數的不確定性吞掉。**

### 步驟 1''— 真機層的雜訊（還沒做）

模擬層清乾淨之後，真機層仍要量：同設定重跑 N≥5 次，量 TTFT／端到端時間的變異
（vLLM 排程、GPU 時脈、NFS 抖動）。**這一步還沒做。**

### 步驟 2 — 在原始順序上跑 oracle，確認 headroom 真的存在

**不重排 session。** 如果原始順序的 oracle headroom < 5%，就照 `CLAUDE.md` 規則 4 判 **NO-GO**。

目前 MI300X 的 oracle 結果（Mooncake 前 4,000 requests，原始順序，`results/m4_oracle_mi300x/`）：

| 模型 | conversation | toolagent |
|---|---|---|
| b-llama8b | 20.91% GO | 20.66% GO |
| b-mistral-nemo12b | 19.93% GO | 20.37% GO |
| b-qwen7b-1m | 15.32% GO | 11.54% MARGINAL |
| b-qwen14b-1m | 14.58% MARGINAL | 14.92% MARGINAL |
| b-qwen3-30b-a3b | 9.83% MARGINAL | 9.50% MARGINAL |
| b-seedoss36b | 6.94% MARGINAL | 7.20% MARGINAL |

**這是 prefill-only 口徑。** 換成端到端要乘上 prefill 佔比（MI300X 上 decode 佔 78–82%），
所以端到端 headroom 只剩 **3.76–4.41%**（`CLAIM_EVIDENCE` B3）。

> **這一步其實已經做完，而且結論不樂觀**：端到端 headroom 只有 4% 上下，
> 而 A/A 雜訊還沒量。**很有可能整個可操作空間都在雜訊裡。**

### 步驟 3 — 最強 baseline + 一個簡單 heuristic

老師說：「如果 strongest baseline 加一個簡單 heuristic 就能追平，那就不要再換更大的 predictor。」

**這件事的答案已經有了，而且是否定的**（`results/m5_predictor_mi300x/policy_sim.csv`，`RUNLOG_MI300X.md:772` 發現 19）：

| workload | 模型 | requests | oracle headroom | tiara_sym_l2 | tiara_cost_l2 |
|---|---|---|---|---|---|
| toolagent | b-llama8b | 7,413 | 3.76% | **−33.20%** | −34.99% |
| conversation | b-llama8b | 3,909 | 4.00% | **−45.26%** | −43.29% |
| lc128kz | b-llama8b | 186 | 8.33% | **−47.32%** | −29.65% |
| toolagent | b-seedoss36b | 7,351 | 4.41% | **−9.68%** | −9.58% |

（負數 = **比最佳 baseline `tier_fs` 更差**。同類工作負載在平台 A 是 **+81.85%**。）

**預測器本身沒壞**（AUC 0.917–0.922，ECE 0.003–0.004）。是可操作空間被硬體壓扁了：
MI300X 上四個 baseline 的端到端差距只有 **1.6%**。

而且平台 A 的消融（`RUNLOG.md:2684`）顯示：
- 長上下文：**排序**貢獻全部，成本模型門檻貢獻 **+0.00%**
- Mooncake：反過來，cascade 目的地貢獻 **−7.79%**

**兩個工作負載上有效的機制不是同一個。這通常代表方法沒有抓到不變量。**

---

## 3. 老師要的第一張表：SOTA matrix

**完整版在 → [`SOTA_MATRIX_20260919.md`](SOTA_MATRIX_20260919.md)**
（104 個 agent、22 份一手全文、110 條 claim、取前 25 條做 3 票對抗式驗證 → **12 條確認、13 條推翻**）

### 3.1 結論：五篇不重疊，而是把決策空間切成不相交的幾塊

**沒有任何一篇同時決定「放哪一階 + 幾位元 + 要不要重算」。**

| | Cake | AdaptCache | Strata | Bidaw | MTDS |
|---|---|---|---|---|---|
| recompute-vs-load | ✅ **唯一** | ❌ | — | — | ❓ |
| bit-width | ❌ 外生 | ✅ **唯一** | ❌ **零壓縮** | — | ❓ |
| tier placement | — | ✅ DRAM/SSD | ✅ HBM/DRAM | ✅ | ❓ |
| **成本模型** | ❌ **無** | 🟡 greedy MCKP | ❌ | — | ❓ |
| predictive 訊號 | ❌ | ❌ LFU | ❌ | 🟡 單一特徵 | ❓ |
| 真實儲存裝置 | ❌ **全模擬** | ✅ | 🟡 只在 1 個子實驗 | ✅ | ❓ |
| 公開 trace | ❌ | ❌ | 🟡 Poisson 合成 | ❌ **專有** | ❓ |

**MTDS 是空白列**：零條 claim 通過驗證，**不可用標題或摘要推斷填補**。矩陣目前是四篇寬。

### 3.2 🔴 老師要的那句話，有答案了

> 「最強方法在什麼真實 workload、什麼硬體條件下會做錯決定？」

**Cake 有 5 個實測的失敗點。** 它的 compute-vs-load 是**無條件的雙指標對撞，沒有成本模型、沒有估計器、沒有 fallback**。Table 6 有四個 sub-1.0× 的格子，最糟的是：

> **Llama-3.1-70B @ GPU 利用率 12.5% / 32 Gbps → 0.80× vs I/O-only，
> 也就是 TTFT 比「什麼都不做、單純去載」還高 25%。**

**而且作者自己把修法寫成未來工作**（逐字）：
> "incorporating a **fallback mechanism** that dynamically adjusts resource allocation when one resource significantly outperforms the other" / "incorporating an **estimation mechanism**"

**→ Tiara 的 κ 成本模型，正是 Cake 承認自己缺的那個東西。**

⚠️ 精確度：要寫 "demonstrably/measured" 不是 "provably"；不可寫「Cake 沒有 scheduler」（它有另一個 priority scheduler，只是不管 compute/load 切分）；最糟那點發生在 12.5% 利用率，是特定 regime。

### 3.3 The Gap（high confidence）

> **一個明確的成本模型，同時仲裁「重算 vs 載入」「放哪一階」「幾位元」，
> 在某資源明顯佔優時有 fallback，且由 predictive 訊號驅動。
> 五篇裡兩篇把其中一部分寫成自己的未來工作，沒有一篇實作。**

耦合關係（gap 的核心論證）：
```
改 bit-width → 改位元組數 → 改載入時間 → 移動 Cake 的交叉點 → 改變該放哪一階
```
AdaptCache 固定交叉點、Cake 固定 bit-width，另一篇獨立模擬研究更直接寫「we hold those fixed to isolate placement」。

**這正是 Tiara 六態動作空間的設計動機。gap 是真的。**

### 3.4 但要對自己誠實：Tiara 只佔了 gap 的一半，而且在模擬器裡

| gap 的組成 | Tiara |
|---|---|
| 真機量測的成本模型 | ✅ **最強的一塊** |
| 仲裁 recompute-vs-load / placement | ✅ 模擬器內 |
| **仲裁 bit-width** | 🟡 只有 INT8、只有平台 A |
| 資源不平衡 fallback | ❌ |
| predictive 訊號（GBDT AUC 0.917–0.922） | ✅ |
| **在真實 serving engine 上跑** | ❌ **從未** |
| 公開 trace ✅ / 回報變異 ❌ | 🟡 |

### 3.5 三個對我們特別有利的方法學發現

1. **⭐ 回報變異數會「超越」而非只是符合這個領域的現行水準。**
   驗證結論：這五篇**沒有任何一篇**回報 seed、變異、CI 或 A/A。
   **→ 老師要求的 A/A test 本身就是 contribution，不只是研究誠信。**
   （Strata/Bidaw 的變異情況未窮舉驗證，不可斷言。）

2. **⭐ 模擬器在頂會可接受**：ICML'25 接受 Cake **整條儲存路徑都是解析式模擬**；OSDI'26 接受 Strata 的 Mooncake 模擬子實驗。
   要避免的反例是「模擬器校準模型本身沒有驗證」——
   **我們的 `simulator_validation.json`（4 個模型中 3 個在 14–29% 內吻合）正是那篇缺的東西，必須放進論文。**

3. **⭐ 「公開 trace 有沒有 headroom」是公開的未解問題。**
   Bidaw 是唯一的正 headroom 證據，但建立在**未公開的專有 trace** 上。
   研究報告明文說：在可釋出的公開 trace 上重現 Belady-gap 量測，**本身就是一個可獨立成立的貢獻**。
   **→ 我們的負面結果（原始順序端到端 headroom 只有 4%）在這個框架下變成資料點，不是失敗。**

### 3.6 所以「Tiara 要打誰」

| 系統 | 定位 |
|---|---|
| **vLLM** | ❌ **不是對手**，是底座。老師說得對 |
| **Cake** | ✅ **主要對手**，有 5 個實測失敗點可打 |
| **AdaptCache** | 🟡 次要。唯一同做 bit-width，但只是 5 頁 workshop 論文，**不能當主要打擊對象** |
| **Strata** | ❌ 非直接對手（lossless、無 bit-width）。引用它，不要宣稱贏它 |
| **Bidaw** | ❌ **是盟友**。引用它證明 headroom 在多輪對話上存在 |

---

## 4. Deep research — ✅ 已完成（結果見 §3 與 `SOTA_MATRIX_20260919.md`）

原本列的三個問題，兩個有答案、一個仍是未解：

| 原問題 | 結果 |
|---|---|
| 五篇有沒有做硬體自適應？ | ✅ **沒有。** Cake 的 chunk 大小是離線調好的常數（COMP 512 / FETCH 128），唯一的 runtime 訊號是二元 residency check。AdaptCache 的品質曲線是離線建的。**「κ 變動時會做錯決定」這個 gap 成立。** |
| 真實 trace 有沒有 headroom 的公開證據？ | 🟡 **未解，而且這本身是機會。** Bidaw 是唯一的正證據但 trace 專有；近零 headroom 的證據只有 preprint 等級，且粒度不同（block-hash prefix vs 每使用者對話）。 |
| 這個領域怎麼證明超過雜訊？ | ✅ **不證明。** 沒有任何一篇報 seed／變異／CI／A/A。**我們報就贏。** |

### 4.1 下一輪 deep research 還要查的（新的，來自本輪的 open questions）

1. **MTDS 全文**（Springer 付費牆）——矩陣的空白列。可試 `semantic-scholar` / `openalex` skill 或學校 VPN。
2. **聯合最佳化 vs 循序組合**：有沒有人量過「先選 bit-width、再跑 Cake」跟「同時決定」的差距？
   研究報告明文說這個問題「**determines whether the identified gap is worth a paper**」。
3. **共用／虛擬化硬體上的 TTFT/TBT 雜訊水準**——本輪查不到任何一手證據。
4. 本輪**完全沒驗證到**的相關系統：LMCache 本體、CacheGen、CacheBlend、Mooncake、InfiniGen、FlexGen、AttentionStore、LayerKV、ALISA、KVPress、RadixAttention。

### 4.2 可以用的 skill

| Skill | 用途 |
|---|---|
| `novelty-check` | 查新：確認 §3.3 的 gap 沒被做過 |
| `arxiv` / `semantic-scholar` / `openalex` | 補 MTDS 全文與上面第 4 點 |
| `kill-argument` | **老師要的對抗式審查**：先寫最強拒稿理由，再逐點反駁 |
| `result-to-claim` | 判定現有結果支持／不支持什麼 claim |
| `experiment-audit` | 實驗誠實度稽核 |

⚠️ 這些 skill 原設計要 Codex MCP 當 zero-context reviewer，本機沒有。降級成 self-review 時**必須在 RUNLOG 標註「非 cross-model」**（`CLAUDE.md` §5）。

### 4.3 可以用的工具

專案已 vendored 22 個 skill，這幾個直接對應老師要的東西：

| Skill | 用途 |
|---|---|
| `novelty-check` | 查新：確認 gap 沒被做過 |
| `arxiv` / `semantic-scholar` / `openalex` | 抓五篇全文與引用圖 |
| `kill-argument` | **老師要的對抗式審查**：先寫最強拒稿理由，再逐點反駁 |
| `result-to-claim` | 判定現有結果支持／不支持什麼 claim |
| `experiment-audit` | 實驗誠實度稽核 |

⚠️ 這些 skill 原設計要 Codex MCP 當 zero-context reviewer，本機沒有。降級成 self-review 時**必須在 RUNLOG 標註「非 cross-model」**（`CLAUDE.md:5`）。

---

## 5. 「Tiara 到底要打誰」— 最核心的一句 claim

老師要一句話。SOTA 查完之後，我可以給一個**候選**，但它現在還不能寫進論文（理由在 §5.4）。

### 5.0 候選 claim

> **多階 KV cache 的三個決策——重算 vs 載入、放哪一階、幾位元——是耦合的：
> 改變位元寬會移動重算/載入的交叉點，進而改變最佳放置。
> 現有系統各自固定其中兩個來隔離第三個（Cake 固定位元寬、AdaptCache 固定交叉點），
> 因此在資源不平衡時會做出可量測的錯誤決策——Cake 實測 5 個設定輸給單一資源 baseline，最差 0.80×。
> Tiara 用真機量測的成本模型同時仲裁三者。**

**打擊對象 = Cake（主）+ AdaptCache（次）。不是 vLLM。**

### 5.1 原本的 claim 已經被自己的數據推翻了一部分

論文現在寫的是：

> κ 跨硬體變動達 **32 倍**，所以單一放置策略不適用所有硬體。

**實測只有 5.8–6.2 倍**（`CLAIM_EVIDENCE` B1），而且 32 倍那個數字是**算術估計不是量測**。
更麻煩的是：κ 對 **ctx 與模型**的敏感度（同一張 MI300X：Llama-8B@16K κ_cpu=1.53，Seed-OSS-36B@96K κ_cpu=11.65，跨越一個數量級）
**比跨硬體的變動還大**。也就是說「硬體決定策略」這個框架，可能選錯了自變數。

### 5.2 但手上有四個真的、而且反直覺的發現

這四個都是**實測**：

1. **傳輸成本不是硬體常數**：同一條 PCIe 上跨模型差 **16.9 倍**，四個模型只吃到 PCIe 的 7.6%（§1.2）。
   → **論文「κ 跨硬體變動」的自變數可能選錯了。**

2. **κ_ssd 在兩個平台上都 < 1**（3090: 0.82–0.94；MI300X: 0.72）
   → **在這些位置，重算比從 SSD 取回便宜。**
   這跟「加 SSD 階可以省時間」的直覺**相反**，也跟 MTDS／Bidaw 這類多階系統的前提方向相反。
   ⚠️ 但若發現 1 的軟體天花板能修掉，這條可能會反轉——**兩者互相牽動，要一起驗證。**

3. **學習式策略在 MI300X 上輸給最簡單的兩階 baseline 9.7–47.3%**，
   但在 3090 上贏 81.85%。**同一份程式碼、同一個工作負載、只換硬體。**

4. **query 側訊號（注意力重要度）對跨請求重用預測完全無效**：7 個模型 AUC 0.480–0.500（= 隨機）。
   原因已量化：注意力幾乎完全由**位置**決定（同位置跨請求變異係數 0.0085，位置間 0.3306，差 38.8 倍）。
   → 「用注意力挑該留哪個 block」這個在文獻裡很常見的直覺，**對 cross-request placement 是錯的**。

### 5.3 三個候選方向（等 SOTA 表出來再決定）

| 方向 | 一句話 | 風險 | 設備夠嗎 |
|---|---|---|---|
| **B. 傳輸路徑效率**（⬆ 升為首選） | 「多階 KV 系統的有效傳輸頻寬不是硬體性質：同一條 PCIe 上跨模型差 16.9 倍，四個模型只用到 7.6%。所有建立在『傳輸 ≈ 硬體頻寬』假設上的放置策略，都在錯誤的成本常數上最佳化」 | 機制未知，可能查出來是設定問題；但**即使是設定問題，那本身也是可報告的發現** | ✅ **一張 GPU 就夠** |
| **A. 縮成 measurement paper** | 「多階 KV placement 的可操作空間取決於 (模型架構, ctx, 硬體, 實作路徑)；我們量了 7 模型 × 2 平台，並給出判斷該不該做複雜策略的判準」 | 最安全，四個負面發現全部變成貢獻；novelty 偏弱，要靠量測完整度取勝 | ✅ 已有大部分資料 |
| **C. 保留 Tiara，claim 縮到「什麼時候該用複雜策略」** | 「我們給出一個 cheap test，事前判斷這個 (workload, hardware) 值不值得做 learned placement」 | 需要 A/A + 多硬體點才站得住；MI300X 的負面結果反而變成證據 | 🟡 只有 1 個硬體點（3090 在另一台） |

| **D. 佔領 §3.3 的 gap**（SOTA 查完後新增） | 「三個決策是耦合的，現有系統各自固定兩個隔離第三個，因此在資源不平衡時做錯決定」 | **必須先證明聯合最佳化贏過最佳循序組合**，否則 gap 不值一篇論文 | 🟡 需補 FP8/INT4 進策略 |

**我的傾向：先做 D 的決定性實驗，B 當第二，A 當保底。**

### 5.4 為什麼那句 claim 現在還不能寫

1. **Tiara 的 bit-width 只有 INT8、只有平台 A**；FP8/INT4 從未進入策略，oracle 根本沒有精度階。
   **claim 講「同時仲裁三者」，但第三者目前只有一格。**
2. **從未在真實 serving engine 上跑過**（§0.3）。
3. **最關鍵**：還沒證明「聯合最佳化贏過最佳的循序組合」（先選 bit-width、再跑 Cake）。
   研究報告把這列為 open question，並明寫它
   「**determines whether the identified gap is worth a paper**」。

### 5.5 🎯 那個決定成敗的實驗（只需要模擬器，設備完全足夠）

**問題**：聯合決定 (bit-width, tier, recompute) 有沒有贏過「先選 bit-width、再最佳化 tier/recompute」？

**做法**（全部在現有模擬器內，成本常數已經量好）：

| 組別 | 策略 |
|---|---|
| S1 循序-最佳 | 對每個 bit-width ∈ {BF16, FP8, INT8, INT4} **各跑一次**全域最佳的 tier/recompute 決策，取最好的那個 |
| S2 聯合 | 每個 block 獨立同時決定 (bit-width, tier, recompute) |
| S3 oracle | 聯合 + 未來已知（上界） |

**判準（事先講好）**：
- 若 **S2 − S1 的 95% CI 跨過 0** → **gap 不值一篇論文，claim 要換。**
- 若 S2 明顯 > S1 → gap 成立，而且**差距的大小就是這篇論文的貢獻量級**。

**這個實驗會殺死自己的想法，而且只要幾小時。** 它比再多跑幾次 baseline 有價值得多。

⚠️ 前置條件：要先把 FP8/INT4 接進 oracle 的動作空間（目前 `load_precision_tiers()` 存在但模擬路徑不呼叫，`m4_oracle.py:89`）。

---

## 6. 我接下來要做的事（依序，不跳步）

### 已完成（本次）
- [x] SOTA matrix（§3、`SOTA_MATRIX_20260919.md`）—— 12 條確認、13 條推翻
- [x] 硬體與實驗表（§1）—— 全部本機實測
- [x] 數字對帳（§0.2）—— 發現 4/5 個數字不存在
- [x] **A/A test 的模擬層**（§2 步驟 1）—— 8 個 seed 全部 20.7%，**變異 = 0**
- [x] 發現傳輸成本跨模型差 17.1 倍（§1.2）

### 待做（依優先序）
- [ ] **1. §5.5 的決定性實驗**：聯合 vs 循序組合。
      **這一個實驗決定整個 gap 值不值得寫成論文。** 只需模擬器，幾小時。
      前置：把 FP8/INT4 接進 oracle 動作空間（`m4_oracle.py:89`）。
- [ ] **2. 追 §1.2 的 4.4 GB/s 天花板**。用 `rocprof` 追一次 CPU 階 offload，
      找出 Qwen3-30B-A3B 為什麼能到 66% 而其他四個只有 7.6%。**可能改寫論文的自變數。**
- [ ] **3. 成本常數的不確定性傳播**（§2 步驟 1'）——給 headroom 與 Tiara−`tier_fs` 差值的 95% CI。
      **在這之前不報任何百分比**，包括 +0.76% 和 +15.40%。
- [ ] **4. 真機層 A/A**（§2 步驟 1''）——同設定重跑 N≥5。
- [ ] **5. 把「工作集 vs 容量」assertion 寫進 trace loader**。
- [ ] **6. 修正文件裡的事實錯誤**：
      「本地 NVMe」→ Broadcom MegaRAID volume（`CLAIM_EVIDENCE_MI300X.md:94`、`RUNLOG_MI300X.md:789`）；
      Fig. 2 的 plugin 過度宣稱（六態有四態掛不進 `CachePolicy`）；
      κ 的 32 倍 → 實測 5.8–6.2 倍。
- [ ] **7. 把 `simulator_validation.json` 寫進論文**——§3.5 發現這正是同類模擬研究缺的東西。
- [ ] **8. 補 MTDS 全文**（Springer 付費牆），填上矩陣空白列。
- [ ] **9. 決定方向 A / B / C / D**，跟老師確認後再動 `main.tex`。

---

## 附錄：這份盤點怎麼產生的

| 類別 | 方法 |
|---|---|
| 硬體數字 | 本機重新量測（`rocm-smi`、`torch.cuda`、`fio`、`dd` O_DIRECT、cgroup） |
| 五個數字的核對 | `grep -rn` 全 repo，排除 `.git`／HTML／PDF／ipynb |
| plugin 完成度 | 讀 `code/` 實作 + `git status` on `src/vllm-v0.28.0` + `OPEN_ISSUES.md` B7 |
| 實驗結果 | 直接讀 `results/**/*.csv` 與 `RUNLOG*.md`，標行號 |
| SOTA | ⏳ deep research 進行中，本版未完成 |

**本文件所有數字都可追溯。任何一格如果找不到出處，請當成錯的，回來問我。**
