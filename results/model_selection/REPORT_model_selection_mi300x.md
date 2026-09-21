# 平台 B（單張 MI300X）選模報告 — dense / MoE / hybrid

- **產生時間**：2026-09-15
- **run_id**：`20260915-084837-model-survey`（原始檔在 `/mlsteam/data/tiara/runs/20260915-084837-model-survey/`）
- **限制條件**（使用者指定）：BF16 權重為主、目標 1M 上下文、偏向較強／較大的模型、依 dense／MoE／hybrid 三類分開選。

> 🔴 **這份報告裡的容量、κ、prefill 時間全部是由 `config.json` 算出來的算術估計（`ARITHMETIC_NOT_MEASURED`），不是量測值。**
> 真值要靠 M1（vLLM 啟動時印出的 `GPU KV cache size`）和 M2（成本常數）量出來。
> 論文要引用時只能引 M1/M2 的量測，**不能引這張表**。

---

## 0. 結論先講

| 類別 | 主選 | 備選 | 單卡 BF16 能否跑到 1M（估算） | 卡在哪 |
|---|---|---|---|---|
| **dense** | **Seed-OSS-36B-Instruct**（強、原生 512K） | Llama-3.1-Nemotron-8B-UltraLong-1M（1M 錨點，與平台 A 同架構） | Seed：❌ 懸崖約 **399K**；UltraLong-8B：✅ 約 1.23M | dense 30B 級每 token KV 256 KiB，1M 就要 256 GiB |
| **MoE** | **Qwen3-30B-A3B-Instruct-2507** | GLM-4.7-Flash（MLA） | ✅ 約 1.18M（KV 96 GiB） | 262K 以上要靠 DCA，**vLLM 0.19.1 沒有 DCA 後端** |
| **hybrid** | **NVIDIA-Nemotron-3-Nano-30B-A3B**（模型卡原生 1M，並附 RULER@1M） | Qwen3.6-35B-A3B／Qwen3.6-27B、Kimi-Linear-48B-A3B | ✅ KV 在 1M 時只要 6 GiB | 🔴 **vLLM 0.19.1 的 OffloadingConnector 不支援 hybrid** |

**有兩個阻擋條件要先由你決定，實驗才有辦法往下走**（細節在 §4）：

1. **hybrid 類在目前的 vLLM 0.19.1 上沒辦法跟 OffloadingConnector 一起跑。** `OffloadingConnector` 要到 **v0.21.0** 才實作 `SupportsHMA`，而 v0.21 需要 torch 2.11；ROCm 版只有 nightly（`2.11.0.dev20260206`）。那個 nightly **之前在這台機器上裝成功過**（`logs/torch-nightly-20260915-075354.log`），但是 vLLM 並沒有用它編過。
   → 要做 hybrid，就得把 vLLM 升到 ≥0.21。這是「靜態讀程式碼」得出的結論，**還沒有實際跑過來確認**。
2. **「BF16 + 1M + 強模型」三個條件在單卡上同時成立的 dense 模型不存在。** 30B 級的 dense 模型 KV 本身就放不下。能讓 1M 放得進單卡的強模型只有 MoE（GQA KV 小）和 hybrid（大部分層沒有 KV）。

---

## 1. 方法

| 步驟 | 做法 | 產出（皆在 `results/model_selection/`） |
|---|---|---|
| 候選清單 | 掃 HF 上 30 多家機構 2025 年以後 >7B 的模型，加上長上下文關鍵字搜尋，得到 57 個候選 | `hf_fetch_index.json` |
| 架構參數 | 抓每個 repo 的 `config.json`／`README.md`／safetensors 參數量（含 HF commit sha） | 原始檔在 run 目錄的 `raw/` |
| 記憶體估算 | 逐層計算：GQA、MLA、滑動視窗、chunked、GDN、Mamba1/2、KDA、Lightning、short-conv，公式與 vLLM 0.19.1 的 `kv_cache_interface`／`mamba_utils` 一致 | `estimates_mi300x.csv` |
| 主流程度 | 從 HF papers 搜 16 組 KV／長上下文關鍵字，取 2025-01 以後的 **490 篇** arXiv 全文，統計各模型被提及的篇數 | `paper_model_frequency_{min3,any}.csv` |
| 框架可行性 | 讀 vLLM 0.19.1 原始碼：架構是否已註冊、HMA 與 connector 的相容性、有沒有 DCA 後端 | 見 §4 |

**估算假設**：
- VRAM 192 GiB（實測，`hw_mi300x.json`）× `gpu_memory_utilization` 0.90，再扣 8 GiB runtime 餘裕（假設值），剩下的是權重 + KV 的預算。
- κ₀ 用和論文 `tab:kappa` 相同的簡化模型：2·N_active FLOP/token、MFU 100%、算力 1307.4 TFLOPS、鏈路 63 GB/s。
- 程式：`code/model_survey_{fetch,estimate,papers,papers_strict}.py`。

---

## 2. 主流論文實際在用哪些模型

以 490 篇全文為樣本，嚴格版要求**全文提到 ≥3 次**，用來排除只在 related work 帶過一次的情況：

| 模型 | ≥3 次 | 2025 → 2026 | 類別 |
|---|---|---|---|
| Llama-3.1-8B | **35.3%** | 42.2% → 29.6% | dense |
| Mistral-7B | 12.9% | 16.6% → 9.7% | dense |
| Qwen3-8B | 12.4% | 4.9% → **18.7%** | dense |
| Qwen2.5-7B | 12.0% | 15.2% → 9.4% | dense |
| Llama-3-8B | 12.0% | 20.2% → 5.2% | dense |
| Qwen2.5-14B | 5.9% | 5.4% → 6.4% | dense |
| Qwen3.5/3.6 | 4.7% | 0% → **8.6%** | hybrid |
| Qwen3-32B | 4.3% | 1.3% → 6.7% | dense |
| Llama-3.1-70B | 3.1% | 3.6% → 2.6% | dense |
| GLM-4.5/4.6/4.7 | 2.9% | 0.9% → 4.5% | MoE |
| Qwen2.5-7B-1M | 2.2% | 2.7% → 1.9% | dense |
| Qwen3-30B-A3B | 2.2% | 0.9% → 3.4% | MoE |
| Jamba | 2.0% | 2.2% → 1.9% | hybrid |
| gpt-oss-120b | 1.8% | 1.3% → 2.2% | MoE |
| Kimi-Linear | 1.0%（任一次 6.3%） | 0.4% → 1.5% | hybrid |
| Nemotron-3 | 0.4%（任一次 1.4%） | 0% → 0.7% | hybrid |

**判讀**：
- KV 論文的主流仍然是 **7–8B dense**。這就是平台 A 已經在用的 Llama-3.1-8B／Qwen2.5-7B。
- 2026 年上升最快的是 **Qwen3-8B** 與 **Qwen3.5/3.6 hybrid**。
- 「較強的模型」在這個領域的論文裡本來就少，選它是為了強化論文論點，不是在跟隨主流。
- 限制：這是關鍵字提及的統計，**提及不等於拿來評測**。

---

## 3. 各類候選的估算表（單卡 MI300X，BF16 權重，BF16 KV）

欄位說明：
- **權重**：BF16 權重佔用。
- **KV/tok**：只算全注意力層與 MLA 層。
- **1M 總量**：權重 + KV + 每序列的 state。
- **懸崖**：KV 預算剛好用完時的上下文長度。
- **κ₀**：未計入 state checkpoint。
- **κ₀ ckpt**：計入 prefix cache 時的 state checkpoint（見 §5）。

### 3.1 dense

| 模型 | 權重 GiB | KV/tok KiB | 1M 總量 GiB | 懸崖（BF16 KV） | 宣稱上下文 | κ₀ |
|---|---|---|---|---|---|---|
| **Seed-OSS-36B-Instruct** | 67.3 | 256 | 323 ❌ | **399K** | 512K 原生 | 13.3 |
| Qwen3-32B | 61.0 | 256 | 317 ❌ | 425K | 32K 原生 / 131K 需 YaRN | 12.0 |
| granite-4.2-30b | 54.5 | 256 | 311 ❌ | 452K | 128K | 10.8 |
| EXAONE-4.5-33B | 64.0 | 256（3/4 層 SWA） | 129 ✅ | 1.64M（HMA 開）/ **413K**（HMA 關） | 262K | 50.5 |
| gemma-4-31B-it | 58.3 | 880 含 SWA | 139 ✅ | 1.39M（HMA 開）/ **127K**（HMA 關） | 256K | 36.8 |
| Mistral-Small-3.2-24B | 44.7 | 160 | 205 ❌ | 787K | 131K | 14.1 |
| Qwen2.5-14B-Instruct-1M | 27.5 | 192 | 220 ❌ | 750K（FP8 KV：1.5M） | 1M，需 DCA | 7.2 |
| **Llama-3.1-Nemotron-8B-UltraLong-1M** | 15.0 | 128 | **143 ✅** | 1.23M | **原生 1M** | 5.9 |
| glm-4-9b-chat-1m | 17.7 | 80 | 98 ✅ | 1.93M | 1M | 11.2 |
| Llama-3.3-70B／Apertus-70B／Llama-3-70B-1048k | 131 | 320 | 451 ❌ | 109K | 128K／65K／1M | 20.8 |

### 3.2 MoE

| 模型 | 權重 GiB | 啟用參數 | KV/tok KiB | 1M 總量 GiB | 懸崖 | 宣稱上下文 | κ₀ |
|---|---|---|---|---|---|---|---|
| **Qwen3-30B-A3B-Instruct-2507** | 56.9 | 3.3B | 96 | **153 ✅** | 1.18M | 262K 原生，1M 需 DCA+MInference | **3.2** |
| GLM-4.7-Flash | 58.2 | 3B | 52.9（MLA） | 111 ✅ | 2.11M | 203K | 5.3 |
| gemma-4-26B-A4B-it | 48.1 | 3.8B | 220 含 SWA | 68 ✅ | 6.1M（HMA 開）/ 556K（HMA 關） | 256K | 17.9 |
| gpt-oss-20b | 39.0 | 3.6B | 48 含 SWA | 63 ✅ | 5.5M／2.7M | 131K | 14.1 |
| Phi-3.5-MoE | 78.0 | 6.6B | 128 | 206 ❌ | 711K | 128K | 4.9 |
| Hunyuan-A13B | 149.7 | 13B | 128 | 278 ❌ | 123K | 256K | 9.6 |
| Llama-4-Scout／GLM-4.5-Air／gpt-oss-120b／Mistral-Small-4 | 202–222 | — | — | 權重本身放不下 ❌ | 0 | — | — |

### 3.3 hybrid（線性層、SSM 層與全注意力層混合）

| 模型 | 權重 GiB | 全注意力層比例 | KV/tok KiB | state/序列 MiB | 1M 總量 GiB | 宣稱上下文 | κ₀ / κ₀ ckpt |
|---|---|---|---|---|---|---|---|
| **Nemotron-3-Nano-30B-A3B** | 58.8 | 6/52 層（Mamba2） | **6** | 46.8 | **65 ✅** | 1M（RULER@1M 86.3，引自模型卡） | 54.9 / 11.4 |
| Nemotron-3.5-Lightning-30B-A3B | 61.3 | 同上 | 6 | 46.8 | 67 ✅ | 1M | 47.1 / 9.8 |
| Nemotron-Labs-3-Puzzle-75B-A9B | 145.8 | 8/88 | 8 | 122 | 154 ✅（只剩 19 GiB 預算） | 1M（RULER@1M 92.2） | 106 / 17.7 |
| **Qwen3.6-35B-A3B** | 67.0 | 10/40（GDN） | 20 | 61.4 | 87 ✅ | 262K 原生，1M 需 YaRN | 14.1 / 3.5 |
| Qwen3.6-27B（dense FFN） | 51.7 | 16/64（GDN） | 64 | 147 | 116 ✅ | 262K 原生，1M 需 YaRN | 40.9 / 10.2 |
| Kimi-Linear-48B-A3B | 91.5 | 7/27（MLA+KDA） | 7.9 | 41.4 | 99 ✅ | **原生 1M** | 35.9 / 9.4 |
| Jamba2-Mini（52B/12B） | 96.1 | 4/32（Mamba1） | 16 | 8.3 | 112 ✅ | 256K | 70.6 / 9.2 |
| granite-4.0-h-small | 60.0 | 4/40 | 16 | 73.7 | 76 ✅ | 128K | 51.8 / 5.2 |
| Falcon-H1-34B | 62.7 | 每層並行 | 144 | 146 | 207 ❌ | 262K | 22.0 / 11.0 |
| Qwen3-Next-80B-A3B | 151.5 | 12/48 | 24 | 37.7 | 176 ❌（懸崖 580K） | 1M 需 YaRN | 11.8 / 3.0 |
| Nemotron-3-Super-120B／Qwen3.5-122B／MiniMax-M1 | 230+ | — | — | — | 權重放不下 ❌ | — | — |

---

## 4. 框架可行性（vLLM 0.19.1 @ ROCm；靜態讀程式碼，未實跑）

| 項目 | 發現 | 證據 | 影響 |
|---|---|---|---|
| **A. hybrid × OffloadingConnector** | 只要設定 `--kv-transfer-config`，vLLM 就會自動關閉 HMA。HMA 一關，Mamba／GDN／KDA 的 state 規格無法和注意力 KV 統一，`unify_hybrid_kv_cache_specs` 會丟 `ValueError` | `vllm/config/vllm.py:1227-1244`、`v1/core/kv_cache_utils.py:1212-1219`；`OffloadingConnector` 在 0.19.1 **不是** `SupportsHMA` | 🔴 **所有 hybrid 在 0.19.1 上都跑不了論文的 CPU／SSD 階** |
| B. 哪一版開始支援 | `OffloadingConnector(KVConnectorBase_V1, SupportsHMA)` 從 **v0.21.0** 開始出現；v0.21–v0.26 的 CMake 需要 torch 2.11，v0.27 以後需要 2.13 | `git show <tag>:.../offloading_connector.py` 與 `CMakeLists.txt` | ROCm 能取得的 torch 2.11 只有 nightly；2.13 沒有 ROCm 版 |
| **C. 滑動視窗模型 × connector** | HMA 關閉後，SWA／chunked 層改配全長 KV | `unify_hybrid_kv_cache_specs` | gemma-4-31B 的懸崖從 1.39M 掉到 **127K**；EXAONE-4.5 掉到 413K |
| **D. DCA** | 0.19.1 的 attention backend registry 裡沒有 dual-chunk backend（grep 為空） | `v1/attention/backends/registry.py` | Qwen2.5-1M、Qwen3-30B-A3B 超過 262K 時，品質數字無效（與平台 A 同樣的問題） |
| E. 架構是否註冊 | 未註冊：MiniCPM-SALA、LongCat 兩款、Cohere2Moe、Solar-Open、Intern-S2-Mobius、Qwen3.8-Flash-Next；EXAONE-4.5 的外層 wrapper 未註冊 | `model_executor/models/registry.py` | 這幾個直接排除 |
| F. 模型載入時間 | 權重放在 NFS（循序讀 643.9 MB/s，實測），60 GiB 約 1.7 分鐘，150 GiB 約 4 分鐘 | `storage_bench.csv` | 可以接受 |

---

## 5. 對論文論點的意義（這一節是推論，要靠 M1/M2 驗證）

1. **三類剛好把 κ 往三個方向推，正好是論文需要的「常數改變，策略就改變」的實例。**
   - dense：κ₀ 由模型大小決定，UltraLong-8B 約 6、Seed-36B 約 13。
   - MoE：啟用參數只有 3B，但 KV 和同規模 dense 相當，**κ₀ 最低**（Qwen3-30B-A3B 約 3.2）。
   - hybrid：KV 極小，只看 KV 時 κ₀ 很高（Nemotron-Nano 約 55）。
   - 論文 §2 預測「MoE 的 Drop 門檻應高於 dense」，現在在同一張卡上可以檢驗。
2. 🔴 **hybrid 有一個論文目前沒有涵蓋的成本項：prefix cache 的 state checkpoint。**
   - vLLM 在 `mamba_cache_mode="all"` 下，每個 block 邊界都要存一份全部線性層的 state。
   - 攤到每個 token，Nemotron-Nano 約 **23 KiB/token**，是它 KV 的 **3.8 倍**；Qwen3.6-27B 約 192 KiB/token，是 KV 的 3 倍。
   - 計入後 κ₀ 從 55 掉到 11。所以 hybrid 的「傳輸成本」主要是 state，不是 KV。
   - **這是論文六階動作空間沒有的一種狀態。** 如果要做 hybrid，動作空間和 oracle 都需要擴充。
3. **1M 在 dense 上只有小模型做得到。** Seed-OSS-36B 的懸崖（約 399K）落在它的原生上下文（512K）以內，量到的是**純記憶體限制**。這和平台 A 用 Qwen2.5-7B-1M 的論證方式一樣，反而是乾淨的設定。

---

## 6. 建議的實驗組合與決策點

**A 案（不升級 vLLM，今天就能開始 M1/M2）**：
- dense：Seed-OSS-36B（到 399K）+ UltraLong-8B-1M（到 1M，與平台 A 同架構，κ 可以跨平台比）。
- MoE：Qwen3-30B-A3B-2507（到 262K；更長只量延遲，不宣稱品質）。
- hybrid：**做不了**。

**B 案（升級到 vLLM v0.21–v0.26 + torch 2.11 nightly）**：
- A 案全部，再加 hybrid：Nemotron-3-Nano-30B-A3B（1M）+ Qwen3.6-35B-A3B。
- 代價：
  - 重新編譯 vLLM（上次花了 40–90 分鐘）。
  - nightly torch 有穩定性風險。
  - 版本仍然和平台 A（v0.28.0）不一致，κ 跨平台比較依舊要標 `NOT_COMPARABLE`。

**需要你決定**：
1. 選 A 案還是 B 案（也就是要不要做 hybrid）？
2. 「強模型」的定義：dense 接受 Seed-OSS-36B 到不了 1M，還是寧可只用能到 1M 的小模型？
3. 權重授權：Llama 系列官方 repo 是 gated，這台機器沒有 HF token。UltraLong 是 NVIDIA 的公開 repo，不受影響。

在你決定之前，**我沒有下載任何模型，也沒有啟動任何 GPU 量測**。
