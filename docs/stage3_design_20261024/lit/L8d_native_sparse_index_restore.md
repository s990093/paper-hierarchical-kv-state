# L8d：原生稀疏注意力的 learned indexer，能不能拿來當分層 KV 回復的「免費索引」？

- **日期**：2026-10-24
- **執行者**：文獻 agent（Claude；只做網路搜尋與閱讀，沒有跑 GPU、沒有安裝套件，除本檔外沒有改任何 repo 檔案）
- **怎麼查的**：
  - 約 28 次 WebSearch / WebFetch，另外用 Hugging Face API 讀 `config.json`、README、safetensors 大小與參數數。
  - 原文 PDF 只從 arxiv.org / github.com / huggingface.co 下載到 scratchpad 的 `l8d_pdfs/`，用 `pdftotext -layout` 轉成文字後閱讀。
  - 頁碼為 PDF 頁；DeepSeek-V3.2-Exp 報告引用的是印刷頁碼。
  - **讀了全文或相關章節的**：DeepSeek-V3.2-Exp 報告（全文 6 頁）、DeepSeek-V3.2 arXiv（§2、§3.2.1）、NSA（§3–5）、MoBA（§2–3）、MiniCPM4 技術報告（§1、§2.1、§5.3）、InfLLM-V2（§3–4）、NOSA（§1–3）、DeepSeek-V4（§2.3、§3.5、§4.2.1）、DeepSeek-V4.1-Flash（摘要、§2.3、§3.2.1）、ESS、Janus（§1–4）、SAC（§1–3）、FlashMemory-DS-V4（§2–3）、IndexCache（§1）、Native Top-k study（§1–3）。
  - **讀了的程式碼**：DeepSeek-V3.2-Exp `inference/model.py`（Indexer class）、DeepSeek-V4-Flash `inference/model.py`、MiniCPM4.1-8B `modeling_minicpm.py`。
- **審查方式**：**self-review，不是 cross-model review。** 數字都經過「原文或 config 再對一次」，但沒有找獨立 reviewer。
- **證據等級**：
  - 〔全文〕讀了原文的相關段落
  - 〔部分〕只讀了部分章節
  - 〔摘要〕只讀摘要或搜尋摘要
  - 〔程式碼〕來自 config 或原始碼
  - 〔官方文件〕來自 README、部落格或官方 docs
  - 〔標題〕只確認了標題、編號或存在
- 文中「未找到」一律指「本次搜尋沒有找到」，不代表不存在。`NOT_MEASURED` / `NOT_VERIFIED` 表示沒有實測或沒有核實。

---

## 0. 結論

**一、原生 indexer 能不能當 restore index？要分兩種情況。**

- **對原生稀疏模型本身：可以，而且是無損的。**
  - 原因：這類模型每一層本來就只讀 indexer 選中的條目，所以「只回復 indexer 選中的 union」對它是無損的。這是由架構保證的，不是靠近似。
  - 索引本身也夠小，放得進 GPU：
    - DSA 每 token 每層 132 B，約為 BF16 MLA KV 的 11.5%、vLLM FP8 KV 的 20.1%。
    - InfLLM-v2 每 token 每層 40 B，約為 KV 的 3.9%。
    - 128K 上下文時，DSA 索引約 0.98 GiB/序列，MiniCPM4.1 約 0.16 GiB/序列（§2）。

**二、但有三個硬限制，直接打在我們的設計上。**

- **限制 (i)：逐層相依。** 第 ℓ 層的 indexer query 來自第 ℓ−1 層的輸出，所以「要回復哪些 block」沒辦法在跑第 0 層之前一次決定。
  - Janus 的對策：用較早層的 hidden state 預測，recall > 96%；漏掉的再補讀（corrective reads），整體仍然無損。
  - 所以「先拿新問題算出 restore 集合、再一次從 NFS 讀回」這個流程，**不能直接套用**在原生稀疏模型上。
- **限制 (ii)：粒度不合。**
  - DSA 是 token 級 top-2048，InfLLM-v2 是 64-token block（含 window 共 96 個 block ≈ 6,144 token）。
  - 把「問題的每個 token × 每層 × 每個 KV head」的 union 映射到我們的 512-token block 之後，可能接近全部 block。
  - 這一點**沒有任何文獻量過 = NOT_MEASURED**，是最大的未知數。
- **限制 (iii)：dense 模型沒有原生索引。** Llama-3.1-8B、Qwen2.5-7B-1M 拿不到「免費」索引。InfLLM-V2 宣稱只要 5B tokens 就能把 dense 模型轉成稀疏，但在我們的 4×H200 / 48h 限制內是否可行，NOT_MEASURED。

**三、原生 indexer 並不是 full attention 的好近似。**

- Meituan 的研究在 HELMET 128K 上量到：DeepSeek-V3.2-Exp 的 indexer top-k 和主模型逐 head exact top-k 的重疊平均約 60%（Native Top-k study §3, p.5〔全文〕）。
- 模型 end-to-end 的表現仍然好，因為它就是在這個 indexer 下訓練的。
- 所以，原生 index 對「答案 block」的 recall，跟我們的 full-K oracle 比起來是多少，仍然未知。**沒有找到任何跨輪（query 改變）的量測。**

**四、新穎性：「原生索引 + 從慢速層 + 跨輪 + 補讀」已大致被做掉。**

- **Janus**（arXiv 2609.36938，2026-09-29）的設定：
  - 模型：DeepSeek-V4-Flash/Pro、GLM-5.2
  - 儲存：KV 放在 SSD
  - 情境：多輪 agent 的 append prefill
  - 方法：用模型自己的 indexer 預測要讀什麼，漏掉的再補讀，結果無損
- 這對「原生稀疏模型的無損 restore」算是**完全重疊**。
- 剩下的空間：
  - **(a) fidelity-graded restore**：沒被選中的 block 用低精度回復，或乾脆不回復
  - **(b) 比 exact union 還小的有損預算**，以及它的 quality–bytes 曲線
  - **(c) dense 模型**
  - **(d) 用 512-token / NFS 粒度量 union 大小**
  - **(e) 跨輪 index recall 對 full-K oracle 的量測**
- 另外值得一提：DeepSeek 自己的生產設計（V4 §3.5.2、V4.1 §3.2.1）在 prefix 命中時仍然是**整段回復** compressed KV，不做選擇性回復〔全文〕。

**五、H200 上的候選模型。**

- **最佳候選：MiniCPM4.1-8B**（InfLLM-v2，8.19B，BF16 16.4 GB，1×H200）。
  - 稀疏推論只能用 HF transformers（≥ 4.56）加 `infllmv2_cuda_impl`（feature_infer 分支），或 CPM.cu。vLLM / SGLang 只支援 dense 模式〔官方文件〕。
  - 關鍵優點：**同一份權重可以切換 dense / sparse**。所以 native index、full-K oracle、Quest 三者可以在同一份 KV 上直接比較。
- **DSA 系（token 級 lightning indexer）次選**：LongCat-Flash-Lite-Sparse（69B，約 3B 啟用，BF16 138.3 GB，SGLang PR #32918），建議 2×H200。
- **生產級**：DeepSeek-V4-Flash（284B，159.6 GB），4×H200。
- **放不進 4×H200 的**：DeepSeek-V3.2（689.5 GB）、GLM-5 FP8（756.2 GB）。

---

## 1. Q1：原生 block/token 選擇模型

### 1a. 指定的四種機制（皆讀原文）

| 模型／機制 | 選擇粒度 | 索引存什麼（維度、dtype） | top-k 預算 | 選擇的共享範圍 | 訓練方式 | 評測 ctx | 參數 | 權重 | 框架支援 | 來源 |
|---|---|---|---|---|---|---|---|---|---|---|
| **DeepSeek-V3.2-Exp / V3.2：DSA（lightning indexer）** | **token 級**（fine-grained token selection） | 每 token 一個 indexer key k^I（128 維，64 個 indexer head 共用）。推論碼存成 **FP8 e4m3 + 每 128 維 1 個 FP32 scale**。query 端有 64 head × 128 維與 per-head 權重 w，打分用 ReLU。〔程式碼〕`Indexer` 內 `k_cache` / `k_scale_cache` | **每個 query token 選 2048 個 KV token**（`index_topk`=2048） | 以 MLA 的 MQA 模式實作：選出的 latent **由所有 query head 共用**；每層各自選 | 從 V3.1-Terminus（已延到 128K）**繼續訓練**：<br>① dense warm-up 只訓 indexer，用 KL 對齊「各 head 加總、L1 正規化」的主注意力分布，1000 steps、2.1B tokens<br>② sparse 階段全模型訓練 15000 steps、943.7B tokens（indexer 輸入 detach，只吃 KL loss）<br>③ 後訓練 | 128K（`max_position_embeddings` 163,840） | 671B 總 / 37B 啟用〔官方文件：V4 README 比較表〕；HF safetensors 計數 685.4B（含 MTP） | [HF V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp)、[HF V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)，MIT，FP8 689.5 GB | **vLLM**：2025-09-29 起支援，Hopper/Blackwell，block size 只支援 64，FP8 MLA KV 656 B/token〔官方文件〕<br>**SGLang**：NSA backend 與 HiSparse〔官方文件〕<br>**TensorRT-LLM**：有 V3.2 KV reuse〔官方文件〕<br>**HF transformers**：有 PR #48788 "[DSA] Attend to the indexer's top-k latents only"，合併狀態 NOT_VERIFIED〔標題〕 | V3.2-Exp 報告 §1 p.2、§2.1 p.2–3、Table 1 p.4〔全文〕；V3.2 arXiv §2.1 p.3–4〔全文〕；config.json〔程式碼〕 |
| **NSA（DeepSeek/PKU，ACL'25）** | **block 級**：<br>• 壓縮 block l=32、stride d=16<br>• 選擇 block l′=64，選 n=16 個（固定含 1 個 initial、2 個 local）<br>• 另有 sliding window w=512 | 壓縮 key K̃^cmp：用可學習的 MLP φ（含 block 內位置編碼）把 l 個 key 壓成 1 個；d_k=192。selection block 的分數直接沿用 compression branch 的注意力分數（Eq. 8–9）。dtype 原文未述。三個 branch 各有**獨立的 K/V** | n=16 個 64-token block，外加 window 512 | **同一 GQA group 內的 head 共用**（分數加總，Eq. 10）；每層各自選 | **從頭預訓練**：27B 總 / 3B 啟用 MoE，30 層，GQA 4 組、64 head。預訓練 token 數兩處不一致：p.3 寫 260B、p.10 寫 270B（8K 長度）。之後用 YaRN 在 32K 上做 continued training 與 SFT | 64K NIAH；LongBench | 27B | **未找到官方權重**（HF 上有社群的 `zen-E/NSA-1B`，未驗證） | 官方推論框架：未找到 | arXiv 2502.11089 §3.3 p.6–8、§4.1 p.9–10、Table 4 p.14〔全文〕 |
| **MoBA（Moonshot）** | **block 級**，block size B 與 top-k 依實驗不同：<br>• scaling 實驗：B=512，top-3（含當前 block）<br>• hybrid 實驗：B=2048<br>• Llama-8B-1M：B=4096，top-12 | **無參數 gating**：分數是 q 與「block 內 K 的 mean-pool」的內積（Eq. 6）。mean-pool 是即時算的，沒有快取。dtype 未述 | top-k 個 block | 以單一 head 描述，多 head 時逐 head 做〔全文；GQA 下的共享方式原文未明說〕；Llama-8B-1M 最後 3 層保持 full attention | **繼續訓練**：Llama-3.1-8B → 1M context；1M 之後啟用 MoBA 訓練 100B tokens，SFT 從 32K 漸進到 1M。**注意：評測時 MoBA 只用於 prefill，生成階段切回 full attention** | RULER@128K：0.7818（full attention 0.7849）；1M NIAH | 8B（scaling 實驗另有 568M–2.1B） | **未找到官方權重**。README 註明 MoBA 必須繼續訓練，不能直接套到現成模型 | MoBA repo（flash-attn 2.6.3；提供 HF Llama 範例） | arXiv 2502.13189 Eq. 6 p.3、Table 1 p.6、§3.3 p.8–9、Table 2 p.9〔全文〕；README〔官方文件〕 |
| **MiniCPM4 / 4.1：InfLLM v2（InfLLM-V2）** | **KV 是 block 級（64 token），query 是 token 級**。會強制加上 init block 1 個，加上 local window 2048（= 32 個 block） | **semantic kernel**：<br>• K 的 mean-pool，kernel 32、stride 16（KC1）<br>• 粗粒度 kernel 128、stride 64（KC2，用來近似 LSE）<br>block 分數取「與該 block 相交的各 kernel 分數」的最大值（Eq. 1）。**沒有額外參數**，存的是 BF16 壓縮 key（程式碼裡的 `InfLLMv2CacheLayer.compress_k_cache` / `compress_k2_cache`）。若 `use_nope`=true，還會另存不含 RoPE 的 K | config `topk`=64；程式碼裡 `topk` = 64 + window/block = **96 個 block ≈ 6,144 個可見 token**，與 InfLLM-V2 論文的 \|I\|=96（init 1 + topk 63 + local 32）一致。`dense_len`=8192：低於這個長度走 dense | **同一 query group 內的 head 共用 top-k block**（MiniCPM4 有 2 個 KV head，每個對應 16 個 query head）；每層各自選 | MiniCPM4：8.3T tokens（4K）預訓練，之後擴到 32K（20B tokens，LongRoPE）。報告 p.6 寫「4K→128K」，但 §5 p.29–30 寫「訓練到 32K，128K 是用 YaRN 達到」，兩處不一致，以 §5 為準<br>InfLLM-V2 論文：短序列 dense 預訓練後，只用 **5B tokens** 做長文微調就轉成 sparse | MiniCPM4-8B：32K，128K（YaRN）跑 RULER-NIAH<br>MiniCPM4.1-8B：原生 64K，README 稱修改 LongRoPE factor 後驗證到 131,072 | 8.19B（HF 計數） | [MiniCPM4-8B](https://huggingface.co/openbmb/MiniCPM4-8B)、[MiniCPM4.1-8B](https://huggingface.co/openbmb/MiniCPM4.1-8B)、[InfLLM-V2-Long-Sparse-Base](https://huggingface.co/openbmb/InfLLM-V2-Long-Sparse-Base)，Apache-2.0 | **稀疏模式**：HF transformers + infllmv2_cuda_impl（SM80/SM90）、CPM.cu<br>**vLLM / SGLang：只有 dense 模式**〔官方文件〕 | MiniCPM4 arXiv 2506.07900 §2.1 p.7–10、§5.3 p.29–30〔全文〕；InfLLM-V2 arXiv 2509.24663 §3.2 p.4–5、§4.1 p.7〔全文〕；README 與 modeling_minicpm.py〔程式碼〕 |

### 1b. 其他 2025–2026 年的開源原生選擇模型（多數只讀了 config 或 README）

| 模型（日期） | 機制與粒度 | 索引內容 | top-k | 參數／權重大小 | 框架 | 證據 |
|---|---|---|---|---|---|---|
| **DeepSeek-V4-Flash**（2026-04） | **CSA**：每 m=4 個 token（有重疊）壓成 1 個 entry，再用 lightning indexer 選 top-k。與 **HCA**（m′=128，dense）交錯使用；前 2 層是純 SWA；另有 128-token 的 sliding window 分支 | 壓縮後的 indexer key K^IComp，c_I=128；indexer 的 QK 計算用 **FP4**（§2.3.4 p.13）。參考碼在 BF16 buffer 上模擬 FP4。主 KV：RoPE 維用 BF16、其餘 FP8，head_dim 512 | **512 個壓縮 entry**（約等於 2048 個原始 token）；64 個 indexer head | 284B 總 / 13B 啟用；FP4 + FP8 混合，159.6 GB；MIT | SGLang day-0（Hopper/Blackwell，ShadowRadix prefix cache，HiSparse 只用在 C4）；vLLM v0.22.0 有 DSv4 與 KV offloading（#43142） | arXiv 2606.19348 §2.3.1 p.9–10、§4.2.1 p.24〔全文〕；config〔程式碼〕 |
| **DeepSeek-V4.1-Flash**（2026-09） | **CSA2**：跨層重用 KV 與 index。**Hierarchical Sparse Indexer**：第一個 Full 層先選 2048 個 block、每個 8 個 position，得到 16,384 個候選；後面的層只在這個候選池裡打分 | 32 個 index head × 128；FP4 KV cache | 512 | 552B backbone（HF 計數 763.2B），510.3 GB | NOT_VERIFIED | arXiv 2609.19969 摘要、§2.3.2 p.11–12〔部分〕 |
| **GLM-5**（2026-02） | DSA | 32 個 index head × 128；MLA 512+64；78 層 | 2048 | HF 計數 753.9B；FP8 756.2 GB；MIT（744B/40B 為二手來源數字） | SGLang HiSparse（GLM-5.1）〔官方文件〕 | config〔程式碼〕；arXiv 2602.15763〔標題，二手〕 |
| **GLM-5.3-Flash**（2026-08） | 混合架構：KDA 線性注意力 + DSA，每 4 層 1 層 DSA（45 層中 11 層）。`index_kpool`=4，含義 NOT_VERIFIED | 32 × 128 | 2048 | 321.3B，FP8 328.3 GB | NOT_VERIFIED | config〔程式碼〕 |
| **LongCat-Flash-Lite-Sparse**（2026-07/08） | **LSA**：在 DSA 上加三件事：<br>• Streaming-aware：預算內固定含 16 個 init token、1024 個 local token<br>• Cross-Layer Indexing<br>• Hierarchical Indexing：先粗選 block，再在 block 內選 token | 16 個 index head × 128；MLA 512+64 | 2048 | 69B 總 / 約 3B 啟用，BF16 138.3 GB；MIT | SGLang PR #32918；README 稱單張 1×H20-141G 可跑（見 §5 的疑點） | README〔官方文件〕；arXiv 2608.01662〔摘要〕 |
| **MiniMax-M3 / M3.1-Flash**（2026-06 / 10） | **block 級**：block 128、top-16 block、local 1 個 block、分數取 max；M3 前 3 層是 dense | 4 個 index head × 128（索引張量的精確形狀 NOT_VERIFIED） | 16 個 block | M3：427B / 854.2 GB<br>M3.1-Flash：220.5B / 252.4 GB（MXFP8） | SGLang HiSparse（文件寫明 index K 留在 GPU）〔官方文件〕 | config〔程式碼〕 |
| **MiniCPM-SALA**（2026-02） | 25% 層用 InfLLM-v2（32 層中 8 層），75% 層用 Lightning 線性注意力 | 同 InfLLM-v2 | 同 InfLLM-v2 | 9.48B，19.0 GB，Apache-2.0；可到 1M | HF transformers；SGLang fork | README、config〔官方文件／程式碼〕 |
| **NOSA-1B / 3B / 8B**（2026-01） | 在 InfLLMv2 上把選擇拆成 **query-aware + query-agnostic**（eviction head，ED-DMA），並設 locality 下界，**原生就是為 CPU offload 設計** | block 分數加上 eviction 分數（dtype NOT_VERIFIED） | k = kq + ke | 8B backbone 形狀與 MiniCPM4 相同（2 個 KV head），config 上限 32K；Apache-2.0 | 自家 NOSI offload 系統，HF | arXiv 2510.13602 §2–3〔全文〕；HF〔程式碼〕 |
| 社群模型 | `fxmeng/DSV2-Lite-DSA-Sparse-1250`（把 DSA 改裝進 DeepSeek-V2-Lite）、`zen-E/MoBA-1B` | — | — | — | — | 只在 HF 確認存在〔標題〕，未驗證 |

---

## 2. Q2：索引大小 vs KV 大小（「索引放 GPU、KV 放 CPU/SSD」行不行）

公式中的符號：

| 符號 | 意義 |
|---|---|
| n_kv | KV head 數 |
| d | head 維度 |
| s | 壓縮 stride |
| L | 層數 |
| B | 每元素 bytes |

「每 token 每層」的索引一律按原始 token 攤提。

| 模型 | 索引 bytes / token / 層（公式） | KV bytes / token / 層 | 索引／KV | 全部層合計（每 token） | 32K：索引 / KV | 128K：索引 / KV | 放得進 GPU？ |
|---|---|---|---|---|---|---|---|
| **DeepSeek-V3.2(-Exp)**，L=61 | 128×1（FP8）+ (128/128)×4（FP32 scale）= **132**〔程式碼〕 | MLA (512+64)×2 = 1152（BF16）；vLLM FP8 格式為 **656**〔官方文件〕 | **11.5%**（對 BF16）／**20.1%**（對 FP8） | 索引 8,052 B；KV 40,016 B（FP8） | 0.246 GiB / 1.22 GiB（FP8） | 0.983 GiB / 4.89 GiB（FP8） | **可以**。交叉驗證：ESS 寫「indexer 佔總 cache 的 16.8%」，而 132/(132+656) = 16.75% ✓〔全文 p.3〕 |
| **GLM-5**，L=78 | 假設與 DSA 同格式（132；dtype NOT_VERIFIED） | 1152 / 656 | 11.5% / 20.1% | 10,296 B | 0.314 GiB / 1.56 GiB | 1.257 GiB / 6.25 GiB | 可以 |
| **LongCat-Flash-Lite-Sparse** | 假設 132（NOT_VERIFIED） | MLA 1152（BF16） | 11.5%（每個注意力層） | 注意力層數 NOT_VERIFIED | — | — | 可以 |
| **DeepSeek-V4-Flash**（只算 CSA 層） | (128×0.5 FP4 + 128/32×1 B scale)/4 = **17**；若照參考碼存 BF16 buffer：128×2/4 = 64（實際服務時的格式 NOT_VERIFIED） | (448×1 FP8 + 64×2 BF16)/4 = **144**；HCA 層 576/128 = 4.5；SWA 固定 128 token | **11.8%**（FP4）／44%（BF16） | — | — | — | 可以 |
| **MiniCPM4/4.1-8B（InfLLM-v2）**，L=32 | KC1：2×128×2/16 = 32；KC2：2×128×2/64 = 8；合計 **40**〔程式碼〕 | 2×128×2×2 = **1024** | **3.9%**。若 `use_nope`=true，再加不含 RoPE 的 K 512 B → 53.9% | 1,280 B | 0.039 GiB / 1.0 GiB | 0.156 GiB / 4.0 GiB | **非常可以** |
| **NSA（論文 27B 設定）** | 4×192×2/16 = **96**（dtype 為假設） | selection branch 的 KV：4×(192+128)×2 = 2560 | 3.75% | — | — | — | 可以，但沒有權重 |
| **MoBA（Llama-8B-1M 設定）** | 8×128×2/4096 = **0.5** | 8×128×2×2 = 4096 | 0.012% | — | — | — | 可以，但沒有權重 |
| 參考：**我們的 Quest**（Llama-3.1-8B，512-token block） | min+max：2×8×128×2/512 = **8** | 4096 | 0.2% | — | — | — | 可以，但 recall 不夠（已實測） |

〔判讀〕以下為解讀，非原文結論：

- **單一序列：索引都放得進 GPU。** 128K 時 DSA 約 1 GiB，InfLLM-v2 約 0.16 GiB。
- **多 session 時，索引本身也會變成要分層管理的物件。** Janus 把 indexer key 視為持久的 session 狀態，每輪從 SSD 整批讀回（因為讀法是確定的，可以提前 prefetch）〔全文 p.4〕。
- **索引小，不代表「restore 集合」小。** 對 DSA 來說，restore 集合是「問題 token × 61 層」各自 top-2048 的 union。
  - IndexCache 量到相鄰層 top-k 重疊 70–100%（30B DSA 模型，以 GLM-4.7-Flash 為基底）〔全文 p.2〕，這對 union 不爆炸是好消息。
  - 但映射到 512-token block 後的 union 比例 **NOT_MEASURED**。

---

## 3. Q3：原生稀疏 × offload / 分層 / 回復，誰做過了？

比較基準是我們的 idea：**restore 時、多輪、從慢速層讀回、可補讀修正**。

| 系統／論文 | 做了什麼 | 索引放哪 | restore 時？ | 多輪？ | 慢速層 | 補讀修正 | 與我們重疊程度 |
|---|---|---|---|---|---|---|---|
| **Janus**（arXiv 2609.36938，2026-09-29，SJTU 等 8 位作者，cs.DC，venue NOT_VERIFIED）〔全文 §1–4〕 | 多輪 agent session 的 **append prefill**。用模型自己的 indexer，餵入較早層的 hidden state h^(ℓ−d) 預測要讀的 KV（DS-V4 用 d=2、GLM-5.2 用 d=4）；predicted reads 與計算重疊，漏掉的用 corrective reads 補齊，**輸出無損**。<br>• 預測 recall > 96%，冗餘 < 5%（V4）／< 10%（GLM-5.2）（p.7）<br>• 約 99% decode 階段需要的 history KV 已在 append prefill 時載入（p.5）<br>• TTFT 比 Tutti / LMCache 快 1.22–1.85×（平均） | indexer key 是持久狀態，每輪從 SSD **整批**讀回 | ✓ | ✓（agent rounds） | NVMe SSD（GPU 直接發起 I/O） | ✓（corrective reads） | **完全重疊**（針對原生稀疏模型的無損版本）。沒做到的：fidelity 分級、比 exact union 還小的有損預算、dense 模型、512-token 粒度、NFS |
| **FlashMemory-DS-V4**（arXiv 2606.09079 v3，Tencent 等 15 位作者，專案已暫停）〔全文 §2–3〕 | 在 DS-V4-Flash 的 D-server 上加 Memory Indexer：**沿用原生 compressed indexer key**，改用 sigmoid 加門檻，每 τ=64 個 decode 步預測一次，把 chunk 從 **CPU cold pool** 搬到 GPU；P-server 把完整 KV 匯出到 D 端的 host memory | 原生 indexer key（在 GPU） | ✗（decode 時的 lookahead） | 用 LongMemEval 評測（一次給完整長 prompt） | CPU DRAM | 部分（之後再由原生 indexer 從已搬上來的子集選 top-k） | **部分重疊** |
| **SGLang HiSparse**（LMSYS blog 2026-04-10；docs）〔官方文件〕 | 適用 DSA（V3.2、GLM-5.1）、DeepSeek V4、MiniMax M3。完整 KV 放 CPU pinned memory，GPU 上是 hot buffer（例如 6144 個 slot），miss 用 LRU 換入。**只在 decode instance 啟用，必須搭配 PD disaggregation**，範例中關閉 radix cache | MiniMax M3：index K 留 GPU；DSA 的情況文件未說明 | ✗ | ✗（文件未討論） | CPU | miss 換入 | **部分重疊** |
| **ESS**（arXiv 2512.10576，Baidu，12 位作者）〔全文 p.1–3〕 | 對象 DeepSeek-V3.2-Exp：**indexer cache 留在 GPU（佔 16.8%）**，latent cache offload 到 CPU；量了同一層連續 decode 步之間的相似度；**結果來自模擬**（32K 時 +69.4%） | GPU | ✗ | ✗ | CPU | LRU / prefetch | **部分重疊** |
| **SAC**（arXiv 2606.19746，PKU / Alibaba）〔部分 §1–3〕 | CXL 分離式 KV 池：prefill 端寫入全部 KV，decode 端**只按需讀 top-k**（不像 RDMA 方案要整段 prefix 拉回）。indexer key 存在 GPU（p.2）。對象 DeepSeek-V3.2 + SGLang | GPU | prefix 重用（P→D） | ✗ | CXL 記憶體 | ✗（每步直接讀） | **部分重疊** |
| **NOSA / NOSI**（arXiv 2510.13602 v3，Tsinghua；v3 頁首標 EMNLP 2026 Main，但搜尋另見 icml.cc 2026 條目，venue 衝突）〔全文 §2–3〕 | 可訓練稀疏 + locality 約束 + CPU offload；量了相鄰 decode 步的 locality γ(t) | GPU | ✗ | ✗ | CPU | ✗ | **部分重疊** |
| **DeepSeek-V4 on-disk KV**（§3.5.2 p.23）與 **V4.1 persistent KV**（§3.2.1 p.19）〔全文〕 | 所有 compressed KV 落盤；prefix 命中時**讀回整段**，尾端不完整的 block 重算；SWA 用 checkpoint 或 replay；V4.1 寫明「Global KV is stored in its entirety」 | — | ✓ | ✓（多輪 session） | SSD | ✗ | **相關但不同**（整段回復） |
| **vLLM OffloadingConnector（v0.22.0：多層 tier、DSv4 #43142）**；**AMD Infera**〔官方文件〕 | KV offload 到多層 tier。Infera 對 DSA 是「latent + co-packed indexer 一起 offload、一起回復」，並警告兩者必須都回復，否則 top-k 會出錯 | 跟 KV 一起 | ✓ | — | CPU / disk / NFS | ✗ | **相關但不同**（整段回復） |
| **TensorRT-LLM V3.2 KV reuse**；**LMCache**（Janus 的 baseline） | prefix 以整個 block 為單位重用 | — | ✓ | ✓ | 多層 | ✗ | **相關但不同** |
| **IndexCache**（arXiv 2603.12201）、V4.1 Hierarchical Sparse Indexer、LongCat 的 HI / CLI | 降低 indexer 本身的成本：跨層重用 index、先選 block 候選池 | GPU | ✗ | ✗ | ✗ | ✗ | **相關但不同**（但「block 級候選池」可以借來當 restore 粒度） |
| **Beyond Memory Leaderboards**（arXiv 2607.16848） | 論文標題裡的「budgeted context restoration」指的是文字層級的 RAG 記憶，不是 KV | — | — | — | — | — | 名稱相似，**無關** |

---

## 4. Q4：多輪 / query 改變情境下的 indexer 品質證據

1. **indexer ≠ exact top-k。** Native Top-k study（Meituan，arXiv 2512.03494）〔全文 §3 p.5〕：
   - 在 HELMET 128K 的 decode 步上，DeepSeek-V3.2-Exp 的 lightning indexer 與主模型逐 head exact top-k 的重疊「averages approximately 60%」。
   - 不過模型 end-to-end 表現仍然好。
   - 只測 decode，**沒有測換問題後的情況**。
2. **步與步之間 locality 高。**
   - NOSA：InfLLMv2 在 16K 輸入、選 4K 時，多數層的 locality γ(t) ≥ 0.8〔全文 p.3〕。
   - ESS：V3.2-Exp 在 LongBench V2 上，同層連續步的相似度高（Fig. 2）〔全文 p.3〕。
   - **兩者都是同一輪內的 decode，不是跨輪。**
3. **跨層重疊高。** IndexCache：相鄰層 top-k 重疊 70–100%〔全文 p.2〕。這有利於 union 較小、也有利於用早層去預測後層（Janus 就是靠這一點）。
4. **跨輪（換一個新問題）唯一的直接證據是 Janus。** 多輪 agent 的 append prefill 下，用早層 hidden 預測 exact union 的 recall > 96%〔全文 p.7〕。但它量的是「預測 vs 模型自己的 indexer」，**不是對 full-K oracle 或答案 block**。
5. **oracle 預算的旁證。** FlashMemory〔全文 §3.3 p.10〕：
   - 在 LongBench-v2 / LongMemEval / RULER 上，保留注意力權重最高的 10–25% CSA chunk（加上全部 HCA）就能 100% 維持 baseline 準確度。
   - 但 **MRCR 即使給到 50% golden chunk 仍掉約 2%**。
   - 這與我們「Qwen 用 oracle 1/4 就無損」的觀察方向一致，但任務相依性強。
6. **已回報的失敗模式。**
   - **MRCR 崩掉**：FlashMemory 的 learned lookahead indexer 讓 MRCR 從 76.0 掉到 48.0（p.10）〔全文〕。
   - **與上下文無關的問題仍會抓很多 chunk**：sigmoid 門檻會漏背景機率，500K 時保留的 chunk 絕對量變成約 2.5 倍（p.10）〔全文〕。
   - **MoBA 評測時 decode 切回 full attention**〔全文 p.9〕：原生選擇在生成階段的品質沒有被直接驗證。
   - **InfLLM-v2 低於 8K 走 dense**（`dense_len`）〔官方文件〕。
   - **DeepSeek-V3.2 在新的 user message 到來時會丟掉歷史推理內容**（§3.2.1 p.9）〔全文〕：跨輪 prefix 會被改寫，restore 集合必須能處理 prefix 的刪改。
7. **SCBench**（ICLR 2025）〔摘要〕：多輪下，sub-O(n) 記憶的方法常常撐不住。但它評的是 training-free 方法，**未找到 NSA / MoBA / DSA / InfLLM-v2 在 SCBench 上的結果**。HySparse2（Xiaomi，arXiv 2609.26368）有評多輪 agentic 任務，但屬於架構論文〔摘要〕。
8. **缺口：** 未找到任何文獻量「原生 index 對答案 block 的跨輪 recall vs full-K oracle」，也未找到在 512-token 粒度下量 union 大小的。

---

## 5. Q5：在 1–4×H200 141GB（每 job 48h）上的可行性

| 模型 | 權重大小 | 至少幾張 H200 | 稀疏路徑可用的框架 | 32K–128K | 備註 |
|---|---|---|---|---|---|
| **MiniCPM4.1-8B** ★ | BF16 16.4 GB | **1** | HF transformers ≥ 4.56 + `infllmv2_cuda_impl`（feature_infer 分支，支援 SM80/SM90；README 列的是 H100，H200 同為 SM90），或 CPM.cu | 原生 64K；128K 要改 LongRoPE（README 稱驗證到 131,072） | dense / sparse 同一份權重可切換；**vLLM / SGLang 只有 dense** |
| MiniCPM4-8B | 約 16 GB | 1 | 同上 | 32K；128K 用 YaRN | — |
| InfLLM-V2-Long-Sparse-Base | F32 32.7 GB | 1 | HF | 32K | base 模型，沒有 chat 能力 |
| NOSA-8B | 約 16 GB（BF16） | 1 | HF + NOSI（ModelScope Docker；要設 `TORCH_CUDA_ARCH_LIST`） | config 上限 32K | 原生就為 offload 設計 |
| MiniCPM-SALA | 19.0 GB | 1 | HF；SGLang fork | 可到 1M | 只有 8/32 層是稀疏，線性層的狀態必須整份回復 |
| **LongCat-Flash-Lite-Sparse** | BF16 138.3 GB | **2**（README 稱 1×H20-141G 可跑，但權重就佔 138.3 GB，KV 空間存疑，NOT_VERIFIED） | SGLang（PR #32918，`--nsa-prefill-backend fa3`） | 原生 1M | **最小的 DSA 系 token 級模型** |
| **DeepSeek-V4-Flash** | FP4+FP8 159.6 GB | 記憶體上要 2；已公開的配置用 4 | SGLang（LMSYS 在 H200 上以 TP=4 跑 benchmark）；NVIDIA Dynamo 的 4×H200 recipe（DP4+EP，FP4 走 MARLIN）；vLLM（社群回報有版本相依問題） | 1M | 生產級 CSA 架構；工程量大 |
| MiniMax-M3.1-Flash | MXFP8 252.4 GB | 2–4 | SGLang | 1M | MXFP8 在 Hopper 上能不能跑 NOT_VERIFIED |
| GLM-5.3-Flash | FP8 328.3 GB | 3–4 | NOT_VERIFIED | 1M | 混合 KDA + DSA |
| DeepSeek-V3.2(-Exp) | FP8 689.5 GB | **> 4 ✗** | vLLM 範例是 8×H200 | — | 放不下 |
| GLM-5（FP8） | 756.2 GB | ✗ | — | — | 放不下 |
| DeepSeek-V4.1-Flash | 510.3 GB | 邊緣 ✗ | — | — | 扣掉 KV 空間後不建議 |
| MiniMax-M3 | 854.2 GB | ✗ | — | — | 放不下 |

**最小可行選擇：**

- **block 級**：MiniCPM4.1-8B（同骨幹的 NOSA-8B、InfLLM-V2-Long-Sparse-Base 可當對照）。
- **token 級 DSA 系**：LongCat-Flash-Lite-Sparse（2×H200）。
- **生產級**：DeepSeek-V4-Flash（4×H200）。

---

## 6. 建議的最小實驗〔判讀〕

1. **模型**：MiniCPM4.1-8B、1×H200、HF 稀疏路徑（設 `dense_len`=-1），沿用我們現有的 32K 兩輪任務；第一輪後保留 KV 與 `compress_k` / `compress_k2`。
2. **原生排序**：第二輪的問題 token 先跑一次 prefill，記錄每層、每個 KV head 的 InfLLM-v2 block 分數與 top-96 選擇。把 64-token block 映射到 512-token block，同時算兩樣東西：
   - **exact union 大小**：這就是無損 restore 的位元組預算
   - **block 排名**：依 max 或 sum 原生分數排序
3. **對照組**（同一份 KV）：
   - (a) **full-K oracle**：dense 模式下，第二輪問題對各 512-block 的注意力質量
   - (b) **Quest** min/max
   - (c) **隨機**
4. **指標**：
   - 答案 block 的 recall@{4, 8, 16}（總共 64 個 512-block）
   - 原生排名與 oracle 排名 top-k 的 Jaccard
   - union 佔全部 block 的比例
5. **品質**：只回復每種排名的前 1/4、1/2，分別在 sparse 與 dense 模式下跑第二輪，與完整回復比較分數。
6. **可預先性**：用前 d 層（例如 d=2、4）的 hidden 預測後層的選擇（Janus 的作法），量 pre-pass recall。這決定能不能「先決定、再一次從 NFS 讀」。
7. 所有數字照 RUNLOG 協定記錄；在實測之前一律為 NOT_MEASURED。

---

## 7. 參考文獻（IEEE 格式）

| # | 文獻 | 狀態 |
|---|---|---|
| [1] | DeepSeek-AI, "DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention," tech. report, Sep. 2025. https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf | 〔全文〕技術報告 |
| [2] | DeepSeek-AI, "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models," arXiv:2512.02556, Dec. 2025. https://arxiv.org/abs/2512.02556 | 〔部分 §2、§3.2.1〕技術報告 |
| [3] | J. Yuan *et al.*（15 位作者），"Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention," in *Proc. ACL 2025 (Long Papers)*, pp. 23078–23097. arXiv:2502.11089. https://aclanthology.org/2025.acl-long.1126/ | 〔全文〕venue 與頁碼取自 [1] 的參考文獻；是否得獎 NOT_VERIFIED |
| [4] | E. Lu *et al.*（25 位作者），"MoBA: Mixture of Block Attention for Long-Context LLMs," arXiv:2502.13189, Feb. 2025；NeurIPS 2025. https://arxiv.org/abs/2502.13189 | 〔全文〕；NeurIPS 2025 來自 proceedings 頁面〔摘要〕 |
| [5] | MiniCPM Team, "MiniCPM4: Ultra-Efficient LLMs on End Devices," arXiv:2506.07900, Jun. 2025. https://arxiv.org/abs/2506.07900 | 〔部分 §2.1、§5.3〕技術報告 |
| [6] | W. Zhao *et al.*（13 位作者），"InfLLM-V2: Dense-Sparse Switchable Attention for Seamless Short-to-Long Adaptation," arXiv:2509.24663, Sep. 2025；ICLR 2026. https://arxiv.org/abs/2509.24663 | 〔部分 §3–4〕；ICLR 2026 來自 proceedings 頁面〔摘要〕 |
| [7] | Y. Huang *et al.*（12 位作者），"NOSA: Native and Offloadable Sparse Attention," arXiv:2510.13602 v3, Sep. 2026. https://arxiv.org/abs/2510.13602 | 〔部分 §2–3〕；venue 衝突（頁首 EMNLP 2026 Main、另見 icml.cc 條目）→ venue NOT_VERIFIED |
| [8] | DeepSeek-AI, "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence," arXiv:2606.19348, Apr. 2026. https://arxiv.org/abs/2606.19348 | 〔部分 §2.3、§3.5、§4.2.1〕 |
| [9] | DeepSeek-AI, "DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression," arXiv:2609.19969, Sep. 2026. https://arxiv.org/abs/2609.19969 | 〔部分〕 |
| [10] | W. He *et al.*（8 位作者），"Efficient Agentic LLM Serving over SSD-based Sparse KV Storage"（Janus），arXiv:2609.36938, Sep. 2026. https://arxiv.org/abs/2609.36938 | 〔部分 §1–4〕venue NOT_VERIFIED |
| [11] | Y. Wang *et al.*（15 位作者），"FlashMemory-DeepSeek-V4: Lightning Index Ultra-Long Context via Lookahead Sparse Attention," arXiv:2606.09079 v3, Jul. 2026. https://arxiv.org/abs/2606.09079 | 〔部分 §2–3〕venue NOT_VERIFIED |
| [12] | X. Chen *et al.*（12 位作者），"ESS: An Offload-Centric Latent-Cache Management Architecture for DeepSeek-V3.2-Exp," arXiv:2512.10576, Dec. 2025. https://arxiv.org/abs/2512.10576 | 〔部分〕venue NOT_VERIFIED |
| [13] | R. Ma *et al.*（10 位作者），"SAC: Disaggregated KV Cache System for Sparse Attention LLMs with CXL," arXiv:2606.19746, Jun. 2026. https://arxiv.org/abs/2606.19746 | 〔部分〕venue NOT_VERIFIED |
| [14] | D. Xiu *et al.*（7 位作者），"A Preliminary Study on the Promises and Challenges of Native Top-k Sparse Attention," arXiv:2512.03494, Dec. 2025. https://arxiv.org/abs/2512.03494 | 〔部分 §1–3〕 |
| [15] | Y. Bai *et al.*（8 位作者），"IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse," arXiv:2603.12201, Mar. 2026. https://arxiv.org/abs/2603.12201 | 〔部分 §1〕 |
| [16] | W. Zan *et al.*（14 位作者），"LongCat Sparse Attention: Taming the Lightning via Streaming-aware Hierarchical Cross-Layer Indexing," arXiv:2608.01662, Aug. 2026. https://arxiv.org/abs/2608.01662 | 〔摘要〕＋ HF README〔官方文件〕 |
| [17] | J. Wei *et al.*, "HySparse2: Hybrid Sparse Attention with Two-Level KV Sharing," arXiv:2609.26368, Sep. 2026. https://arxiv.org/abs/2609.26368 | 〔摘要〕 |
| [18] | Y. Li *et al.*, "SCBench: A KV Cache-Centric Analysis of Long-Context Methods," ICLR 2025. arXiv:2412.10319. https://arxiv.org/abs/2412.10319 | 〔摘要〕 |
| [19] | Z. Xie, Z. Huang, T. Huang, "HiSparse," LMSYS blog, 2026-04-10. https://lmsys.org/blog/2026-04-10-sglang-hisparse ；SGLang docs https://docs.sglang.io/advanced_features/hisparse_guide.html | 〔官方文件〕 |
| [20] | vLLM Team, "DeepSeek-V3.2-Exp in vLLM: Fine-Grained Sparse Attention in Action," 2025-09-29. https://vllm.ai/blog/2025-09-29-deepseek-v3-2 | 〔官方文件〕 |
| [21] | LMSYS, "DeepSeek-V4 on Day 0," 2026-04-25. https://www.lmsys.org/blog/2026-04-25-deepseek-v4/ | 〔官方文件〕 |
| [22] | AMD, "Infera 0.1 – KV-Cache Offload." https://rocm.docs.amd.com/projects/infera/en/latest/features/kv_cache_offload.html | 〔官方文件〕 |
| [23] | Hugging Face model cards 與 config.json：deepseek-ai/DeepSeek-V3.2(-Exp)、DeepSeek-V4-Flash、DeepSeek-V4.1-Flash；zai-org/GLM-5、GLM-5.3-Flash；openbmb/MiniCPM4-8B、MiniCPM4.1-8B、InfLLM-V2-Long-Sparse-Base、MiniCPM-SALA、NOSA-8B；meituan-longcat/LongCat-Flash-Lite-Sparse；MiniMaxAI/MiniMax-M3、M3.1-Flash。https://huggingface.co/ | 〔程式碼〕/〔官方文件〕，2026-10-24 讀取 |
| [24] | OpenBMB, infllmv2_cuda_impl README. https://github.com/OpenBMB/infllmv2_cuda_impl ；MoonshotAI MoBA README. https://github.com/MoonshotAI/MoBA ；thunlp NOSA README. https://github.com/thunlp/NOSA | 〔官方文件〕 |
| [25] | Zhipu AI / Z.ai, "GLM-5: from Vibe Coding to Agentic Engineering," arXiv:2602.15763, Feb. 2026. | 〔標題，二手來源〕未開原文 |
