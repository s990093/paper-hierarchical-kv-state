# 平台 B（AMD MI300X）選模報告：dense / MoE / hybrid

產生時間：2026-09-15T09:16:10+00:00　run_id：`20260915-084837-model-survey`

產生方式：`code/model_survey_fetch.py` → `code/model_survey_estimate.py` → `code/model_survey_papers.py` → `code/model_survey_papers_strict.py` → **`code/model_survey_report.py`（本檔由它產生，勿手改）**

> ⚠️ **本報告的記憶體、可達上下文、κ、prefill 時間全部是由 HF `config.json` 算出的估計值，不是量測。**
> CSV 每列都帶 `estimate_kind=ARITHMETIC_NOT_MEASURED`。真值由 M1（vLLM 啟動時印的 `GPU KV cache size`）
> 與 M2（成本常數）量出；估計與實測的差距本身就是 M1 要回報的第一項。
> 論文使用率是「全文提到 ≥3 次」的篇數比例，是『實際評測用』的**代理**，不是逐篇人工確認。

---

## 0. 結論

**單卡 192 GB、BF16 權重、1M 上下文、要夠強、要是論文常見模型——這五個條件在三類架構上的交集完全不同。**

1. **Dense：「強」與「1M」在單卡上互斥。** KV 隨層數 × KV head 線性成長，Llama-3.3-70B 每 token 320 KiB，1M 需 320 GiB，BF16 權重 131.4 GiB 之後只剩 33.4 GiB → 最長 **109K**（FP8 KV 219K）。30B 級也只到 425K（Qwen3-32B）。**能在單卡跑到 1M 的 dense 只有 8–9B 級**：Llama-3.1-8B 架構 1M 需 128 GiB，放得下。
2. **MoE：強的放不下，放得下的 KV 不省。** Llama-4-Scout（202.4 GiB）、gpt-oss-120b（BF16 217.6 GiB；官方只發 MXFP4）、GLM-4.5-Air、Mistral-Small-4 的 **BF16 權重都超過單卡**。放得下的 30B 級 MoE（Qwen3-30B-A3B）KV 與 dense 同形（96 KiB/token），1M 需 96 GiB，估計可放。
3. **Hybrid：唯一能同時滿足「夠強 + BF16 + 1M」的一類**，因為只有 1/4–1/8 的層存 KV。Qwen3.6-27B 1M 只需 64 GiB、Nemotron-Labs-3-Puzzle-75B-A9B 8.1 GiB。**但 vLLM 0.19.1 的 OffloadingConnector 不支援 hybrid**（見 §5.1）——論文的 CPU/SSD 階在這一類上目前跑不起來。
4. **論文實際常用的模型幾乎都是 7–8B dense。** 490 篇 2025–2026 的 KV cache／長上下文論文中，Llama-3.1-8B 佔 **35.3%**；強模型很少：Qwen3-32B 4.3%、Llama-3.1-70B 3.1%、Qwen3-30B-A3B 2.2%、gpt-oss-120b 1.8%。2026 年上升最快的是 Qwen3-8B（4.9% → 18.7%）與 Qwen3.5/3.6（0.0% → 8.6%）。
5. **跨平台 κ 的主張需要一個兩平台都量過的錨點模型。** 平台 A 唯一自洽的 BF16 成本剖面是 `llama-bf16`（`NousResearch/Meta-Llama-3.1-8B-Instruct`），所以不論其餘怎麼選，**Llama-3.1-8B BF16 必須在平台 B 的集合裡**。

## 1. 需要你決定的事（決定之前不下載、不開跑）

| # | 決定 | 選項 | 我的建議 | 理由 |
|---|---|---|---|---|
| D1 | **hybrid 要不要納入論文？** | (a) 納入，平台 B 升級 vLLM ≥ 0.21.0<br>(b) 只做 dense/MoE，hybrid 只量容量與 κ（不跑卸載） | **(b) 先做，(a) 另開分支驗證** | 論文 §B.2「明確排除的架構」目前把 hybrid SSM 列為定義上不同的問題；且 0.19.1 卸載路徑對 hybrid 不可用、0.21+ 需 torch 2.11 ROCm nightly、Mamba state 的卸載語意在 v0.28.0 程式碼裡找不到處理（未實跑） |
| D2 | **dense 的「強」與「1M」怎麼取捨** | (a) 8B 跑到 1M + 70B 跑到記憶體上限<br>(b) 只用 70B | **(a)** | 70B 在 BF16 權重下最多 109K，1M 物理上放不下；8B 是跨平台錨點，且 UltraLong-1M 與它同架構（成本常數相同）、原生 1M |
| D3 | **MoE 是否堅持 BF16 權重** | (a) 堅持：Qwen3-30B-A3B（+ GLM-4.7-Flash 當 MLA 代表）<br>(b) 放寬：加 gpt-oss-120b（MXFP4） | **(a)** | 論文附錄把平台 B 的權重精度定為「BF16 主、FP8 對照」；gpt-oss-120b 沒有官方 BF16，且只有 131K |
| D4 | **強 hybrid 用哪一個**（若 D1 選 a） | Qwen3-Next-80B-A3B / Nemotron-Labs-3-Puzzle-75B-A9B / Kimi-Linear-48B-A3B | 論文常見度選 Qwen3-Next；原生 1M 選 Puzzle-75B | Qwen3-Next BF16 KV 最多 580K，1M 要 FP8 KV（平台 A 上 FP8 靜態縮放把檢索打到 5%，品質風險）；Puzzle-75B BF16 KV 就放得下 1M，但論文常見度 0.4% |

## 2. 推薦選模

「κ₀」為論文 tab:kappa 同一算法的下界（線性層 2N_active、100% MFU、MI300X 1307.4 TFLOPS；傳輸 = 遠端位置仍需保留的 KV 位元組 / 63 GB/s）。「1M 放得下」欄為 BF16 KV / FP8 KV。「使用率」為全文提到 ≥3 次的論文比例（2026 年）。

| 類別 | 角色 | 模型 | 總／啟用 | BF16 權重 | KV/token | 1M 需 KV | 1M 放得下 | 最長（BF16 KV） | 宣稱上下文 | κ₀ | 0.19.1 卸載 | 使用率 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | 錨點（≤128K） | **Llama-3.1-8B-Instruct**<br><sub>用 NousResearch 鏡像以對齊平台 A</sub> | 8.0B／8.0B | 15.0 GiB | 128.0 KiB | 128.0 GiB | ✓ / ✓ | 1.17M | 131K | 5.9 | 🟢 可 | 29.6% |
| dense | 1M 點（同架構） | **Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct**<br><sub>F32 發布，下載約 2 倍</sub> | 8.0B／8.0B | 15.0 GiB | 128.0 KiB | 128.0 GiB | ✓ / ✓ | 1.17M | 1.02M | 5.9 | 🟢 可 | 0.7% |
| dense | 強（規模軸） | **Llama-3.3-70B-Instruct** | 70.5B／70.5B | 131.4 GiB | 320.0 KiB | 320.0 GiB | ✗ / ✗ | 109K | 131K | 20.8 | 🟢 可 | 0.7% |
| dense | 選配：同家族對照 | **Qwen3-32B** | 32.8B／32.8B | 61.0 GiB | 256.0 KiB | 256.0 GiB | ✗ / ✗ | 425K | 131K | 12.0 | 🟢 可 | 6.7% |
| MoE | 主（常見） | **Qwen3-30B-A3B-Instruct-2507** | 30.5B／3.3B | 56.9 GiB | 96.0 KiB | 96.0 GiB | ✓ / ✓ | 1.12M | 0.96M | 3.2 | 🟢 可 | 3.4% |
| MoE | 選配：MLA 代表 | **GLM-4.7-Flash** | 31.2B／3.0B | 58.2 GiB | 52.9 KiB | 52.9 GiB | ✓ / ✓ | 2.02M | 203K | 5.3 | 🟢 可 | 4.5% |
| hybrid | 主（常見） | **Qwen3.6-27B**<br><sub>與 Qwen3.8-27B config 完全相同</sub> | 27.8B／27.8B | 51.7 GiB | 64.0 KiB + 147 MiB state | 64.1 GiB | ✓ / ✓ | 1.76M | 0.96M | 40.9 | 🔴 不可（hybrid） | 8.6% |
| hybrid | 強（候選 1） | **Qwen3-Next-80B-A3B-Instruct** | 81.3B／3.0B | 151.5 GiB | 24.0 KiB + 38 MiB state | 24.0 GiB | ✗ / ✓ | 580K | 0.96M | 11.8 | 🔴 不可（hybrid） | 1.9% |
| hybrid | 強（候選 2） | **NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16** | 78.3B／9.0B | 145.8 GiB | 8.0 KiB + 122 MiB state | 8.1 GiB | ✓ / ✓ | 2.35M | 1.00M | 105.9 | 🔴 不可（hybrid） | 0.7% |
| hybrid | 強（候選 3） | **Kimi-Linear-48B-A3B-Instruct** | 49.1B／3.0B | 91.5 GiB | 7.9 KiB + 41 MiB state | 7.9 GiB | ✓ / ✓ | 9.30M | 1.00M | 35.9 | 🔴 不可（hybrid） | 1.5% |

**主集合（dense 錨點 + 1M 點 + 70B、MoE 主、hybrid 主）的 BF16 權重合計 270 GiB**；`/mlsteam/data/tiara`（NFS）剩 937 GB。UltraLong 為 F32 發布，下載量約為表中值的 2 倍。

### 為什麼是這幾個

- **Llama-3.1-8B（錨點）**：論文使用率第一（全期 35.3%）；平台 A 的 `llama-bf16` 剖面就是它，是 κ 跨硬體比較唯一能直接對上的一組常數。原生 128K；估計 BF16 KV 最長 1.17M，所以 1M 的瓶頸是模型而非記憶體 → 1M 點改用同架構的 UltraLong-1M（config 除 RoPE 縮放外逐欄相同）。
- **Llama-3.3-70B（強 dense）**：論文 tab:kappa 預測 MI300X 上 κ 由 8B 的 6 增至 70B 的 21，這是那條預測的直接檢驗；估計 κ₀ 5.9 → 20.8。它同時是「192 GB 卡上單請求容量壓力仍存在」的實例（§2 觀察一(c)）：BF16 權重後只剩 33.4 GiB 給 KV。
- **Qwen3-30B-A3B（MoE 主）**：論文 §2「MoE 的 Drop 門檻應高於 dense」的預測需要一對同家族的 dense/MoE——Qwen3-32B（κ₀ 12.0）對 Qwen3-30B-A3B（κ₀ 3.2），同 tokenizer、同訓練世代。⚠️ 卡上 1M 走 DCA+MInference，而 **vLLM 0.19.1 沒有 DCA backend**（與平台 A 在 0.28.0 的發現相同）→ 262,144 以上只能改 YaRN，品質未驗證。
- **GLM-4.7-Flash（MoE 選配）**：卡上自稱 30B 級最強；MLA（每 token 52.9 KiB，約 Qwen3-30B-A3B 的一半）讓 KV 格式本身成為一個變因；GLM-4.5+ 家族 2026 使用率 4.5%。
- **Qwen3.6-27B（hybrid 主）**：Qwen3.5/3.6 是 2026 論文最常用的 hybrid 家族（8.6%）；3/4 層 Gated DeltaNet、1/4 層全注意力，1M 只需 64.1 GiB KV。Qwen3.8-27B 與它 config 完全相同（成本常數可共用），要更新的品質可直接換。

## 3. 論文實際用什麼模型（證據）

方法：HF papers 搜尋 16 組關鍵字（KV cache offloading / compression / eviction / quantization、prefix caching、long-context inference、sparse attention、hybrid linear attention 等）→ 2025-01-01 之後、標題或摘要與 KV／長上下文相關 → 抓 arXiv HTML 全文 → 490 篇可用（2025：223、2026：267）。每篇每模型只算一次；**「≥3 次」欄排除 related work 順帶提一次的情形**（例：DeepSeek-R1 任一次 23.3%，≥3 次只剩 5.3%）。

### 3.1 dense

| 模型家族 | 架構 | ≥3 次（全期） | 2025 | 2026 | 任一次 |
|---|---|---|---|---|---|
| Llama-3.1-8B | dense | 35.3% | 42.2% | 29.6% | 41.8% |
| Mistral-7B | dense | 12.9% | 16.6% | 9.7% | 20.4% |
| Qwen3-8B | dense | 12.4% | 4.9% | 18.7% | 17.1% |
| Qwen2.5-7B | dense | 12.0% | 15.2% | 9.4% | 17.3% |
| Llama-3-8B | dense | 12.0% | 20.2% | 5.2% | 17.8% |
| Llama-3.2-1B/3B | dense | 9.8% | 12.6% | 7.5% | 11.6% |
| Qwen3-4B | dense | 9.4% | 4.9% | 13.1% | 11.8% |
| Llama-2-7B | dense | 7.3% | 13.9% | 1.9% | 11.2% |
| Qwen2.5-14B | dense | 5.9% | 5.4% | 6.4% | 7.3% |
| Qwen3-32B | dense | 4.3% | 1.3% | 6.7% | 7.3% |
| R1-Distill-Llama-8B | dense | 3.5% | 5.4% | 1.9% | 5.5% |
| Gemma-3 | dense | 3.5% | 3.1% | 3.7% | 4.5% |
| Qwen3-14B | dense | 3.3% | 0.9% | 5.2% | 4.5% |
| Llama-3.1-70B | dense | 3.1% | 3.6% | 2.6% | 8.0% |

### 3.2 MoE

| 模型家族 | 架構 | ≥3 次（全期） | 2025 | 2026 | 任一次 |
|---|---|---|---|---|---|
| DeepSeek-V3/V3.x | MoE(MLA) | 5.5% | 3.1% | 7.5% | 23.1% |
| DeepSeek-R1(671B) | MoE(MLA) | 5.3% | 8.5% | 2.6% | 23.3% |
| GLM-4.5/4.6/4.7 | MoE | 2.9% | 0.9% | 4.5% | 3.9% |
| Qwen3-30B-A3B | MoE | 2.2% | 0.9% | 3.4% | 4.1% |
| Mistral-Small/Nemo | dense/MoE | 1.8% | 2.2% | 1.5% | 3.5% |
| gpt-oss-120b | MoE | 1.8% | 1.3% | 2.2% | 8.0% |
| Qwen3-235B-A22B | MoE | 1.6% | 0.4% | 2.6% | 3.9% |
| Gemma-4 | dense/MoE | 1.6% | 0.0% | 3.0% | 2.2% |
| Kimi-K2 | MoE(MLA) | 1.4% | 0.9% | 1.9% | 3.7% |
| DeepSeek-V2/V2.5 | MoE(MLA) | 1.2% | 1.3% | 1.1% | 16.3% |
| gpt-oss-20b | MoE | 1.2% | 0.0% | 2.2% | 7.3% |
| Llama-4-Scout | MoE | 0.8% | 1.8% | 0.0% | 1.0% |
| MiniMax-M2 | MoE | 0.8% | 0.0% | 1.5% | 1.4% |
| Llama-4-Maverick | MoE | 0.6% | 1.3% | 0.0% | 0.8% |

### 3.3 hybrid / SSM

| 模型家族 | 架構 | ≥3 次（全期） | 2025 | 2026 | 任一次 |
|---|---|---|---|---|---|
| Qwen3.5/3.6 | hybrid | 4.7% | 0.0% | 8.6% | 6.3% |
| Mamba/Mamba2 | SSM | 2.4% | 1.3% | 3.4% | 3.9% |
| Jamba | hybrid | 2.0% | 2.2% | 1.9% | 8.2% |
| MiniMax-M1/Text-01 | hybrid | 1.2% | 1.8% | 0.7% | 6.7% |
| Zamba2 | hybrid | 1.2% | 1.8% | 0.7% | 2.9% |
| Qwen3-Next-80B-A3B | hybrid | 1.2% | 0.4% | 1.9% | 3.1% |
| Kimi-Linear | hybrid | 1.0% | 0.4% | 1.5% | 6.3% |
| Nemotron-3 | hybrid | 0.4% | 0.0% | 0.7% | 1.4% |
| Falcon-H1 | hybrid | 0.4% | 0.0% | 0.7% | 1.2% |
| Nemotron-H | hybrid | 0.2% | 0.0% | 0.4% | 1.2% |

**讀法**：dense 7–8B 是這個領域的「預設實驗模型」；強模型的使用率都在個位數百分比。所以「強」與「常見」要分開滿足——錨點用常見的（審稿人可對照），規模／架構軸用強的（主張才有意義）。

## 4. 三類候選的完整估算

欄位：**KV/token** 為全注意力＋MLA 層（遠端位置仍需保留的部分）；**SWA** 層的 KV 每序列只存視窗內；**state** 為線性層（GDN/Mamba/KDA/Lightning/conv）每序列的常數狀態。**1M 放得下** = BF16 KV / FP8 KV。**最長** 為 BF16 KV 下的記憶體上限（未與模型宣稱上限取小）。**卸載後最長** = 開 OffloadingConnector（0.19.1 關閉 HMA，SWA 層改配全長 KV）時的上限。

### 4.1 Dense transformer

| 模型 | 發布 | 總／啟用 | BF16 權重 | 注意力 | KV/token | state/序列 | 1M 需 KV | 1M 放得下 | 最長 | 卸載後最長 | 宣稱上下文（方式） | κ₀ | vLLM 0.19.1 卸載 | 論文 ≥3 次 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Apertus-70B-Instruct-2509 | 2025-09-01 | 70.6B／70.6B | 131.5 GiB | GQA | 320.0 KiB | — | 320.0 GiB | ✗ / ✗ | 109K | 109K | 66K（原生） | 20.8 | 🟢 可 | — |
| Llama-3.3-70B-Instruct | 2024-12-06 | 70.5B／70.5B | 131.4 GiB | GQA | 320.0 KiB | — | 320.0 GiB | ✗ / ✗ | 109K | 109K | 131K（原生（llama3 RoPE）） | 20.8 | 🟢 可 | 1.4% |
| Llama-3-70B-Instruct-Gradient-1048k | 2024-05-03 | 70.5B／70.5B | 131.4 GiB | GQA | 320.0 KiB | — | 320.0 GiB | ✗ / ✗ | 109K | 109K | 1.00M（RoPE θ 放大 + 長序列微調） | 20.8 | 🟢 可 | — |
| Seed-OSS-36B-Instruct | 2025-08-20 | 36.1B／36.1B | 67.3 GiB | GQA | 256.0 KiB | — | 256.0 GiB | ✗ / ✗ | 399K | 399K | 524K（原生（config）） | 13.3 | 🟢 可 | 0.2% |
| EXAONE-4.5-33B | 2026-04-04 | 34.4B／34.4B | 64.0 GiB | GQA+SWA4096 | 64.0 KiB | — | 64.8 GiB | ✓ / ✓ | 1.56M | 413K | 262K（原生） | 50.5 | ⚪ 0.19.1 無此架構 | — |
| Qwen3-32B | 2025-04-27 | 32.8B／32.8B | 61.0 GiB | GQA | 256.0 KiB | — | 256.0 GiB | ✗ / ✗ | 425K | 425K | 131K（32,768 原生；131,072 需 YaRN） | 12.0 | 🟢 可 | 4.3% |
| gemma-4-31B-it | 2026-03-11 | 31.3B／31.3B | 58.3 GiB | GQA+SWA1024 | 80.0 KiB | — | 80.8 GiB | ✓ / ✓ | 1.32M | 127K | 262K（原生 256K） | 36.8 | 🟡 可，SWA 失去省記憶體 | 1.6% |
| granite-4.2-30b | 2026-08-07 | 29.3B／29.3B | 54.5 GiB | GQA | 256.0 KiB | — | 256.0 GiB | ✗ / ✗ | 452K | 452K | 131K（128K 原生（卡：可延伸至 512K）） | 10.8 | 🟢 可 | — |
| granite-4.1-30b | 2026-04-06 | 28.9B／28.9B | 53.8 GiB | GQA | 256.0 KiB | — | 256.0 GiB | ✗ / ✗ | 455K | 455K | 131K（config） | 10.6 | 🟢 可 | — |
| Mistral-Small-3.2-24B-Instruct-2506 | 2025-06-19 | 24.0B／24.0B | 44.7 GiB | GQA | 160.0 KiB | — | 160.0 GiB | ✗ / ✓ | 787K | 787K | 131K（原生） | 14.1 | 🟢 可 | 1.8% |
| Qwen2.5-14B-Instruct-1M | 2025-01-23 | 14.8B／14.8B | 27.5 GiB | GQA | 192.0 KiB | — | 192.0 GiB | ✗ / ✓ | 750K | 750K | 0.96M（DCA+稀疏注意力（需 Qwen 客製 vLLM；>262,144 無 DCA 會退化）） | 7.2 | 🟢 可 | 5.9% |
| glm-4-9b-chat-1m | 2024-06-04 | 9.5B／9.5B | 17.7 GiB | GQA | 80.0 KiB | — | 80.0 GiB | ✓ / ✓ | 1.84M | 1.84M | 1.00M（原生（模型卡：1M）） | 11.2 | 🟢 可 | 1.2% |
| Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct | 2025-03-04 | 8.0B／8.0B | 15.0 GiB | GQA | 128.0 KiB | — | 128.0 GiB | ✓ / ✓ | 1.17M | 1.17M | 1.02M（原生（1M 序列持續預訓練）） | 5.9 | 🟢 可 | 1.0% |
| Llama-3.1-Nemotron-8B-UltraLong-4M-Instruct | 2025-03-04 | 8.0B／8.0B | 15.0 GiB | GQA | 128.0 KiB | — | 128.0 GiB | ✓ / ✓ | 1.17M | 1.17M | 4.09M（原生（4M 序列持續預訓練）） | 5.9 | 🟢 可 | 1.0% |
| Llama-3.1-8B-Instruct | 2025-02-15 | 8.0B／8.0B | 15.0 GiB | GQA | 128.0 KiB | — | 128.0 GiB | ✓ / ✓ | 1.17M | 1.17M | 131K（原生（llama3 RoPE）） | 5.9 | 🟢 可 | 35.3% |
| Qwen2.5-7B-Instruct-1M | 2025-01-23 | 7.6B／7.6B | 14.2 GiB | GQA | 56.0 KiB | — | 56.0 GiB | ✓ / ✓ | 2.69M | 2.69M | 0.96M（DCA+稀疏注意力（需 Qwen 客製 vLLM；>262,144 無 DCA 會退化）） | 12.8 | 🟢 可 | 2.2% |

### 4.2 MoE

| 模型 | 發布 | 總／啟用 | BF16 權重 | 注意力 | KV/token | state/序列 | 1M 需 KV | 1M 放得下 | 最長 | 卸載後最長 | 宣稱上下文（方式） | κ₀ | vLLM 0.19.1 卸載 | 論文 ≥3 次 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mixtral-8x22B-Instruct-v0.1 | 2024-04-16 | 140.6B／39.2B | 261.9 GiB | GQA | 224.0 KiB | — | 224.0 GiB | ✗ 權重放不下 | 放不下 | — | 66K（原生） | 16.5 | —（BF16 權重放不下） | 0.2% |
| Mistral-Small-4-119B-2603 | 2026-01-23 | 119.4B／6.5B | 222.4 GiB | MLA | 22.5 KiB | — | 22.5 GiB | ✗ 權重放不下 | 放不下 | — | 262K（模型卡 256K（config 寫 1,048,576）） | 27.2 | —（BF16 權重放不下） | 1.8% |
| gpt-oss-120b | 2025-08-04 | 116.8B／5.1B | 217.6 GiB | GQA+SWA128 | 36.0 KiB | — | 36.0 GiB | ✗ 權重放不下 | 放不下 | — | 131K（YaRN（config）） | 13.3 | —（BF16 權重放不下） | 1.8% |
| GLM-4.5-Air | 2025-07-20 | 110.5B／12.0B | 205.8 GiB | GQA | 184.0 KiB | — | 184.0 GiB | ✗ 權重放不下 | 放不下 | — | 131K（config） | 6.1 | —（BF16 權重放不下） | 2.9% |
| Llama-4-Scout-17B-16E-Instruct | 2025-04-05 | 108.6B／17.0B | 202.4 GiB | GQA+chunked8192 | 48.0 KiB | — | 49.1 GiB | ✗ 權重放不下 | 放不下 | — | 10.00M（config 10M（iRoPE）） | 33.3 | —（BF16 權重放不下） | 0.8% |
| Solar-Open-100B | 2025-12-10 | 102.7B／12.0B | 191.2 GiB | GQA | 192.0 KiB | — | 192.0 GiB | ✗ 權重放不下 | 放不下 | — | 131K（YaRN 128K） | 5.9 | —（BF16 權重放不下） | — |
| Hunyuan-A13B-Instruct | 2025-06-25 | 80.4B／13.0B | 149.7 GiB | GQA | 128.0 KiB | — | 128.0 GiB | ✗ / ✗ | 123K | 123K | 262K（原生 256K（config 預設 32K，需改 max_position_embeddings）） | 9.6 | 🟢 可 | 0.0% |
| LongCat-Flash-Lite-Sparse | 2026-07-31 | 69.1B／3.0B | 128.8 GiB | MLA | 31.5 KiB | — | 31.5 GiB | ✓ / ✓ | 1.14M | 1.14M | 1.00M（模型卡：原生 1M） | 9.0 | ⚪ 0.19.1 無此架構 | — |
| LongCat-Flash-Lite | 2026-01-27 | 69.1B／3.0B | 128.7 GiB | MLA | 31.5 KiB | — | 31.5 GiB | ✓ / ✓ | 1.15M | 1.15M | 262K（YaRN 256K） | 9.0 | ⚪ 0.19.1 無此架構 | — |
| Phi-3.5-MoE-instruct | 2024-08-17 | 41.9B／6.6B | 78.0 GiB | GQA | 128.0 KiB | — | 128.0 GiB | ✗ / ✓ | 711K | 711K | 131K（LongRoPE 128K） | 4.9 | 🟢 可 | 0.0% |
| GLM-4.7-Flash | 2026-01-19 | 31.2B／3.0B | 58.2 GiB | MLA | 52.9 KiB | — | 52.9 GiB | ✓ / ✓ | 2.02M | 2.02M | 203K（config） | 5.3 | 🟢 可 | 2.9% |
| Qwen3-30B-A3B-Instruct-2507 | 2025-07-28 | 30.5B／3.3B | 56.9 GiB | GQA | 96.0 KiB | — | 96.0 GiB | ✓ / ✓ | 1.12M | 1.12M | 0.96M（262,144 原生；1M 需 DCA+MInference（卡：約需 240 GB GPU 記憶體）） | 3.2 | 🟢 可 | 2.2% |
| Qwen3-Coder-30B-A3B-Instruct | 2025-07-31 | 30.5B／3.3B | 56.9 GiB | GQA | 96.0 KiB | — | 96.0 GiB | ✓ / ✓ | 1.12M | 1.12M | 0.95M（262,144 原生；1M 需 YaRN） | 3.2 | 🟢 可 | 2.2% |
| North-Mini-Code-1.0 | 2026-06-05 | 30.5B／3.0B | 56.8 GiB | GQA+SWA4096 | 26.0 KiB | — | 26.3 GiB | ✓ / ✓ | 4.14M | 1.10M | 256K（原生 256K） | 10.9 | ⚪ 0.19.1 無此架構 | — |
| Trinity-Mini | 2025-12-01 | 26.1B／3.0B | 48.7 GiB | GQA+SWA2048 | 16.0 KiB | — | 16.1 GiB | ✓ / ✓ | 7.25M | 1.81M | 131K（原生 128K） | 17.6 | 🟡 可，SWA 失去省記憶體 | — |
| gemma-4-26B-A4B-it | 2026-03-11 | 25.8B／3.8B | 48.1 GiB | GQA+SWA1024 | 20.0 KiB | — | 20.2 GiB | ✓ / ✓ | 5.83M | 556K | 262K（原生 256K） | 17.9 | 🟡 可，SWA 失去省記憶體 | 1.6% |
| ERNIE-4.5-21B-A3B-PT | 2025-06-28 | 21.9B／3.0B | 40.9 GiB | GQA | 56.0 KiB | — | 56.0 GiB | ✓ / ✓ | 2.21M | 2.21M | 131K（原生） | 5.0 | 🟢 可 | — |
| gpt-oss-20b | 2025-08-04 | 20.9B／3.6B | 39.0 GiB | GQA+SWA128 | 24.0 KiB | — | 24.0 GiB | ✓ / ✓ | 5.24M | 2.62M | 131K（YaRN（config）） | 14.1 | 🟡 可，SWA 失去省記憶體 | 1.2% |

### 4.3 Hybrid（線性層 + 注意力）

| 模型 | 發布 | 總／啟用 | BF16 權重 | 注意力 | KV/token | state/序列 | 1M 需 KV | 1M 放得下 | 最長 | 卸載後最長 | 宣稱上下文（方式） | κ₀ | vLLM 0.19.1 卸載 | 論文 ≥3 次 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MiniMax-M1-80k-hf | 2025-07-01 | 456.1B／45.9B | 849.5 GiB | GQA | lightning | 40.0 KiB | 140.0 | 40.1 GiB | ✗ 權重放不下 | 放不下 | — | 0.95M（原生 1M（Lightning 注意力）） | 108.0 | —（BF16 權重放不下） | 1.2% |
| Qwen3.8-Flash-Next | 2026-08-24 | 180.0B／6.0B | 335.3 GiB | GQA | gdn | 24.0 KiB | 110.1 | 24.1 GiB | ✗ 權重放不下 | 放不下 | — | 0.95M（262,144 原生；1,000,000 需延伸） | 23.5 | —（BF16 權重放不下） | 4.7% |
| Qwen3.5-122B-A10B | 2026-02-24 | 125.1B／10.0B | 233.0 GiB | GQA | gdn | 24.0 KiB | 146.5 | 24.1 GiB | ✗ 權重放不下 | 放不下 | — | 0.96M（262,144 原生；1,010,000 需 YaRN） | 39.2 | —（BF16 權重放不下） | 4.7% |
| NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | 2026-03-10 | 123.6B／12.0B | 230.2 GiB | GQA | mamba2 | 8.0 KiB | 162.3 | 8.2 GiB | ✗ 權重放不下 | 放不下 | — | 1.00M（模型卡 1M（RULER@1M 已報）） | 141.2 | —（BF16 權重放不下） | 0.4% |
| Ling-2.6-flash | 2026-04-28 | 107.5B／7.4B | 200.2 GiB | MLA | lightning | 4.5 KiB | 28.0 | 4.5 GiB | ✗ 權重放不下 | 放不下 | — | 262K（131,072 原生；262,144 需 YaRN） | 154.8 | —（BF16 權重放不下） | — |
| Qwen3-Next-80B-A3B-Instruct | 2025-09-09 | 81.3B／3.0B | 151.5 GiB | GQA | gdn | 24.0 KiB | 37.7 | 24.0 GiB | ✗ / ✓ | 580K | — | 0.96M（262,144 原生；1,010,000 需 YaRN） | 11.8 | 🔴 不可（hybrid） | 1.2% |
| Qwen3-Coder-Next | 2026-01-30 | 79.7B／3.0B | 148.4 GiB | GQA | gdn | 24.0 KiB | 37.7 | 24.0 GiB | ✗ / ✓ | 714K | — | 262K（原生 256K（卡未宣稱 1M）） | 11.8 | 🔴 不可（hybrid） | 1.2% |
| NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16 | 2026-06-24 | 78.3B／9.0B | 145.8 GiB | GQA | mamba2 | 8.0 KiB | 122.2 | 8.1 GiB | ✓ / ✓ | 2.35M | — | 1.00M（模型卡 1M（config 預設 256K；RULER@1M 已報）） | 105.9 | 🔴 不可（hybrid） | 0.4% |
| AI21-Jamba2-Mini | 2026-01-06 | 51.6B／12.0B | 96.1 GiB | GQA | mamba1 | 16.0 KiB | 8.3 | 16.0 GiB | ✓ / ✓ | 4.30M | — | 262K（原生 256K） | 70.6 | 🔴 不可（hybrid） | 2.0% |
| Kimi-Linear-48B-A3B-Instruct | 2025-10-30 | 49.1B／3.0B | 91.5 GiB | MLA | kda | 7.9 KiB | 41.4 | 7.9 GiB | ✓ / ✓ | 9.30M | — | 1.00M（原生 1M） | 35.9 | 🔴 不可（hybrid） | 1.0% |
| Intern-S2-Mobius | 2026-07-29 | 36.0B／—B | 67.0 GiB | GQA | gdn | 20.0 KiB | 61.4 | 20.1 GiB | ✓ / ✓ | 4.89M | — | 262K（config（模型卡未宣稱更長）） | — | ⚪ 0.19.1 無此架構 | — |
| Qwen3.5-35B-A3B | 2026-02-24 | 36.0B／3.0B | 67.0 GiB | GQA | gdn | 20.0 KiB | 61.4 | 20.1 GiB | ✓ / ✓ | 4.89M | — | 0.96M（262,144 原生；1,010,000 需 YaRN） | 14.1 | 🔴 不可（hybrid） | 4.7% |
| Qwen3.6-35B-A3B | 2026-04-15 | 36.0B／3.0B | 67.0 GiB | GQA | gdn | 20.0 KiB | 61.4 | 20.1 GiB | ✓ / ✓ | 4.89M | — | 0.96M（262,144 原生；1,010,000 需 YaRN） | 14.1 | 🔴 不可（hybrid） | 4.7% |
| Qwen-AgentWorld-35B-A3B | 2026-06-22 | 34.7B／3.0B | 64.6 GiB | GQA | gdn | 20.0 KiB | 61.4 | 20.1 GiB | ✓ / ✓ | 5.01M | — | 262K（原生） | 14.1 | 🔴 不可（hybrid） | — |
| Falcon-H1-34B-Instruct | 2025-05-01 | 33.6B／33.6B | 62.7 GiB | GQA | mamba2 | 144.0 KiB | 146.1 | 144.1 GiB | ✗ / ✓ | 743K | — | 262K（config） | 22.0 | 🔴 不可（hybrid） | 0.4% |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 | 2026-08-01 | 32.9B／3.0B | 61.3 GiB | GQA | mamba2 | 6.0 KiB | 46.8 | 6.0 GiB | ✓ / ✓ | 17.24M | — | 1.00M（模型卡 1M） | 47.1 | 🔴 不可（hybrid） | 0.4% |
| granite-4.0-h-small | 2025-09-16 | 32.2B／8.8B | 60.0 GiB | GQA | mamba2 | 16.0 KiB | 73.7 | 16.1 GiB | ✓ / ✓ | 6.55M | — | 131K（模型卡 128K） | 51.8 | 🔴 不可（hybrid） | 0.0% |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | 2025-12-04 | 31.6B／3.5B | 58.8 GiB | GQA | mamba2 | 6.0 KiB | 46.8 | 6.0 GiB | ✓ / ✓ | 17.65M | — | 1.00M（模型卡 1M（config 預設 256K；RULER@1M 已報）） | 54.9 | 🔴 不可（hybrid） | 0.4% |
| Qwen3.5-27B | 2026-02-24 | 27.8B／27.8B | 51.7 GiB | GQA | gdn | 64.0 KiB | 146.8 | 64.1 GiB | ✓ / ✓ | 1.76M | — | 0.96M（262,144 原生；1,010,000 需 YaRN） | 40.9 | 🔴 不可（hybrid） | 4.7% |
| Qwen3.6-27B | 2026-04-21 | 27.8B／27.8B | 51.7 GiB | GQA | gdn | 64.0 KiB | 146.8 | 64.1 GiB | ✓ / ✓ | 1.76M | — | 0.96M（262,144 原生；1,010,000 需 YaRN） | 40.9 | 🔴 不可（hybrid） | 4.7% |
| Qwen3.8-27B | 2026-08-05 | 27.8B／27.8B | 51.7 GiB | GQA | gdn | 64.0 KiB | 146.8 | 64.1 GiB | ✓ / ✓ | 1.76M | — | 0.95M（262,144 原生；1,000,000 需延伸） | 40.9 | 🔴 不可（hybrid） | 4.7% |
| LFM2-24B-A2B | 2026-02-24 | 23.8B／2.3B | 44.4 GiB | GQA | shortconv | 20.0 KiB | 0.2 | 20.0 GiB | ✓ / ✓ | 6.02M | — | 128K（config） | 10.8 | 🔴 不可（hybrid） | — |
| MiniCPM-SALA | 2026-02-11 | 9.5B／9.5B | 17.7 GiB | GQA | lightning | 8.0 KiB | 24.0 | 8.0 GiB | ✓ / ✓ | 18.39M | — | 1.00M（模型卡 1M+（config 524,288）） | 111.5 | ⚪ 0.19.1 無此架構 | — |

## 5. 影響實驗設計的發現

### 5.1 🔴 vLLM 0.19.1 的 OffloadingConnector 不支援 hybrid（靜態讀碼，未實跑）

證據鏈（`/mlsteam/workspace/src/vllm`，tag v0.19.1）：

1. `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:44`：`class OffloadingConnector(KVConnectorBase_V1)`——**不是** `SupportsHMA` 子類別。
2. `vllm/config/vllm.py:1227-1244`：設了 `--kv-transfer-config` 且使用者沒指定時，**自動關閉 hybrid KV cache manager（HMA）**。
3. `vllm/v1/core/kv_cache_utils.py:1160-1219`（`unify_hybrid_kv_cache_specs`）：HMA 關閉時，滑動視窗層被改成全注意力規格（**失去省記憶體**）；Mamba／線性注意力層無法與注意力層統一 → `ValueError: Hybrid KV cache manager is disabled but failed to convert the KV cache specs to one unified type.`
4. 若強制開 HMA：`vllm/distributed/kv_transfer/kv_connector/factory.py:58-61` 直接拒絕（`Connector OffloadingConnector does not support HMA`）。
5. `git show <tag>:.../offloading_connector.py`：**v0.21.0 起才是 `SupportsHMA`**（v0.19.1、v0.20.0 不是；v0.21.0 到 v0.29.0 皆是）。v0.21.0–v0.26.0 需要 torch 2.11（ROCm 只有 nightly `2.11.0.dev20260206`），v0.27.0 起需要 torch 2.13（ROCm 無）。
6. 即使 v0.28.0（平台 A 的版本）宣告 `SupportsHMA`，`vllm/v1/kv_offload/` 與 connector 內**找不到任何 Mamba／state 的處理**——state 是否被卸載、被怎麼卸載，未知。

**後果**：hybrid 模型在平台 B 目前可以量容量（M1）與重算成本，但論文的 CPU/SSD 階（M2 retrieval、M3 baselines）跑不起來。

### 5.2 🟡 滑動視窗模型在卸載模式下的記憶體會暴增

Gemma-4-31B：HMA 開啟時 1M 需 80.8 GiB（50/60 層只存 1,024 token）；開 OffloadingConnector 後所有層都存全長，1M 需 880.0 GiB，最長由 1.32M 掉到 **127K**。gpt-oss、Llama-4、EXAONE-4.5、Trinity 同理。**baseline 與 Tiara 會在不同的有效容量下比較**——這類模型若入選，需在 M3 明確記錄。

### 5.3 🟡 hybrid 開 prefix caching 時，「每 token 要存的東西」是 attention KV 的約 3 倍

vLLM `mamba_cache_mode="all"`（開 prefix caching 時的預設）在每個 block 邊界存一份全部線性層的 state；block 大小由「單層 attention 頁 ≥ 單層 state 頁」決定（`vllm/model_executor/models/config.py`）。

| 模型 | attention KV/token | block 大小 | state 攤到每 token | 快取前綴每 token 合計 | κ₀（只算 KV） | κ₀（含 state） |
|---|---|---|---|---|---|---|
| Qwen3.6-27B | 64.0 KiB | 784 | 191.8 KiB | 255.8 KiB | 40.9 | 10.2 |
| Qwen3-Next-80B-A3B-Instruct | 24.0 KiB | 544 | 70.9 KiB | 94.9 KiB | 11.8 | 3.0 |
| NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16 | 8.0 KiB | 3,136 | 39.9 KiB | 47.9 KiB | 105.9 | 17.7 |
| Kimi-Linear-48B-A3B-Instruct | 7.9 KiB | 1,920 | 22.1 KiB | 30.0 KiB | 35.9 | 9.4 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | 6.0 KiB | 2,096 | 22.9 KiB | 28.9 KiB | 54.9 | 11.4 |

**這對論文是好消息也是風險**：hybrid 的 κ 取決於「要不要存 state checkpoint」，而那正是放置決策的一部分——動作空間在 hybrid 上多出一個維度。論文 §B.2 目前以「Marconi 已處理」排除 hybrid；若納入，這一段要重寫。

### 5.4 🟡 1M 的長度延伸方式在 vLLM 0.19.1 上有兩個缺口

- **DCA 不可用**：`vllm/` 內除模型與 RoPE 之外找不到 dual-chunk 的 attention backend（與平台 A 在 0.28.0 的崩潰結論一致）。受影響：Qwen2.5-7B/14B-Instruct-1M、Qwen3-30B-A3B-Instruct-2507 的 1M 模式。
- **YaRN 延伸屬於外推**：Qwen3-Next、Qwen3.5/3.6/3.8 的 262,144 → 1M 需 YaRN。CLAUDE.md §8 規定不要為了掃更長而開 YaRN（品質退化無法歸因）；若採用，需另建「full-KV + 同 YaRN 設定」的基準線。**原生 1M 的只有**：UltraLong-8B-1M、GLM-4-9B-1M、Kimi-Linear、Nemotron-3 系列、MiniCPM-SALA、LongCat-Flash-Lite-Sparse。

### 5.5 ⚪ 版本對齊

平台 A = vLLM v0.28.0，平台 B = v0.19.1（REPORT_MI300X §6 #3）。在對齊之前，κ 的跨硬體數字一律標 `NOT_COMPARABLE`；若為了 hybrid 升到 v0.21–v0.26，仍與 A 不同版。v0.27+ 需要 ROCm 沒有的 torch 2.13。

## 6. 方法、常數與假設

| 項目 | 值 | 來源 |
|---|---|---|
| VRAM | 206,141,652,992 B | `results/hw_mi300x.json`（實測） |
| 可用比例 | 172.8 GiB（gpu_memory_utilization 0.90） | vLLM 預設 |
| runtime 餘裕 | 8.0 GiB | **假設值**（activation / workspace / graph）；M1 量真值 |
| 1M | 1,048,576 token | `--max-model-len 1048576` |
| 算力 | 1307.4 TFLOPS | 論文 tab:ratio（規格值，非量測） |
| 主機鏈路 | 63.0 GB/s | PCIe Gen5 ×16 單向理論值（論文同表） |
| 權重 | safetensors 參數量 × 2 B | HF API（FP8/MXFP4 發布者為換算的 BF16 大小） |
| 啟用參數 | 模型卡宣稱值；無則由 config 算（標於 CSV `active_src`） | README.md |

逐層公式與 vLLM 0.19.1 `kv_cache_interface.py`、`mamba_utils.py` 一致，完整寫在 `code/model_survey_estimate.py` 開頭。已人工驗算：Qwen2.5-7B-1M 56 KiB/token、Llama-3.1-8B 128 KiB/token，與論文表 tab:kvsize 相同；Llama-3.1-8B/70B 的 κ₀ 與論文 tab:kappa 的 6×/21× 相同。

**已知的估計偏差方向**：(1) runtime 餘裕 8 GiB 對 1M 級 prefill 可能偏低（Qwen 卡上稱 30B-A3B 跑 1M 約需 240 GB，含 activation）——邊界上的模型要以 M1 為準；(2) 多模態模型的權重含 vision tower；(3) Gemma-4 的 K=V 全域層仍存兩份 KV（vLLM `gemma4.py`）；(4) LongCat-Sparse 與 Qwen3.8-Flash-Next 的稀疏索引器 KV 未計入。

## 7. 檔案位置

| 內容 | 路徑 |
|---|---|
| 候選模型原始 config / README / API | `/mlsteam/data/tiara/runs/20260915-084837-model-survey/raw/<org>__<name>/` |
| 估算 CSV（每模型一列，含 run_id/ts） | `results/model_survey/model_estimates_mi300x.csv` |
| 論文全文與逐篇命中 | `/mlsteam/data/tiara/runs/20260915-084837-model-survey/papers/text/`、`results/model_survey/paper_mentions.csv` |
| 論文模型頻率 | `results/model_survey/paper_model_frequency_any.csv`、`paper_model_frequency_min3.csv` |
| 程式 | `code/model_survey_{fetch,estimate,papers,papers_strict,report}.py` |

