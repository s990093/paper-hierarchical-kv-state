# 第三階段研究地圖（文獻＋缺口＋要做的實驗）

> **日期**：2026-10-19（UTC）。**性質**：彙整 `lit/L0`–`L3`（四個文獻 agent，全部 self-review，不是 cross-model）；`lit/L4`（Cascade、CacheSmith、CacheFlow 全文查新）跑中，回來後補。
> **讀法**：每一列只寫結論與出處；細節、數字的條件、閱讀深度（〔全文〕〔部分〕〔摘要〕）都在對應的 L 檔。**不同硬體、batch、context 的數字只並列、不排名。**「沒找到」＝這輪搜尋沒找到，不等於不存在（CLAUDE.md 規則 7）。

---

## 0. 一頁結論

1. **影片的「英特爾 KV Cache 四件套」只有 KV Shrink 查得到可安裝的東西**：GitHub `intel/intel-accel-for-llm`，是 vLLM V1 KV connector，用 QAT／IAA／CPU 做 **DEFLATE 無損壓縮**，先把 BF16 做 byte-shuffle、**只壓高位元組（sign＋exponent）**，搬運用 DSA＋GDRCopy 批次送出；沒有 ROCm 後端、README 沒有效能數字。KV Cascade、KV Fuse、KV Infinity 只出現在媒體報導裡各一句話，**應視為行銷的概念整理，不是標準套件**。「agent 一輪約六成時間在搬資料」查不到出處（NOT_FOUND）。（L0 §1–§2）
2. **前兩階段的教訓「位置不是有用的訊號」和文獻一致**：好的低精度格式不傷答對率（第二階段），業界用量化多於 eviction（YAKV），eviction 在多輪會壞（SCBench、KVzip、RocketKV-MT、EpiCache）。（L1 §1、§4）
3. **文獻共同的盲點是「GPU 只跑自己」**：預取／重疊論文（InfiniGen、ShadowKV、KVPR、FreeKV、HCache）都沒量背景負載；CacheFlow 假設計算與載入互不干擾。**我們在 MI300X 上已經量到干擾**：GPU 忙時 H2D 慢 46–50%、FP8 反量化慢 2.4–3.8 倍（第二階段 08 §2.1、09 §6.1；L2 §0.1）。這是第三階段最強的切入點。（L2 §0、G2）
4. **「單請求」的結論多，「多請求／併發」的邊界少**：復用的打平點只有單請求前沿（py-kvcache），vLLM 本身有「命中但正在載入 → 延後」的機制（#44294，L40S 上開卸載反而慢約 2 倍），MI300X 上沒人量。（L3 §0、缺口 2）
5. **agent 工具暫停的 KV 去留**：現有系統只做二選一或三選一（InferCept、Continuum、Tokencake、KVFlow），κ 不實測、沒有 SSD；**最大未知是 Cascade（arXiv 2608.06557）**，L4 讀全文中。（L3 缺口 1）
6. **壓縮救不了 NFS 的還原速度，但能縮短預取提前量**：stage 44 實測 NFS 級對 CPU 慢 28–34 倍，INT4 只拉到 18–19 倍；無損約 1.3× 更小。（L2 §0 第 3 點）

---

## 1. 依方向（任務 §4 的 A–H）

| 方向 | 主要前作（出處） | 已經清楚的 | 還缺的（＝我們的機會） | 對應實驗 |
|:--|:--|:--|:--|:--|
| **A** eviction／重要性 | H2O、SnapKV、PyramidKV、Ada-KV、DuoAttention、Expected Attention、KVzip、DefensiveKV、KVP；agent：Online compaction、AgentKV、CommitKV（L1 §1） | 寫入時依問題刪除在多輪會壞；延後到下一輪問題來才壓可救回（2608.00902）；kvpress 的逐 head 遮罩不省記憶體，只有實體壓實才少搬 | 在 **Cake 還原**下、**同位元組**比較「有損刪除 vs 低精度」；刪除只在載入區有意義（L1 G1） | **S2** |
| **Cascade** 選擇性載入 | Quest、InfLLM、ShadowKV、MagicPIG、RetroInfer、ArkVale、OmniKV、RocketKV、HashAttention；還原時：IMPRESS、ContiguousKV、CacheTune（L1 §2） | decode 時的選擇性載入很多；多數只在 CUDA 上 | 「下一輪實際用到哪些 block」的實測；依重要度排還原順序、與 Cake 結合；GPU 忙時的淨收益（L1 G2–G4） | **C1a → C1b** |
| **B** 量化／壓縮 | 有損：第二階段已涵蓋；無損：SplitZip（BF16 指數，1.31–1.33×）、ZipNN、KVServe、CacheGen（L2 §1） | 盈虧公式已有（KVServe 加總、SplitZip 管線）；卸載層幾乎都不壓縮 | **解壓／反量化搶 GPU** 沒進任何盈虧模型；AMD／CPU 端無損沒人量 | **S1** |
| **C** 卸載／分層 | LMCache、Mooncake、Strata／HiCache、Dynamo KVBM、llm-d、vLLM OffloadingConnector、Pensieve、IMPRESS、Bidaw、HCache（L2 §2） | 2026 年都上線了；hipFile 要 ROCm 7.14（我們 7.2.2）、本機看不到 /dev/nvme | ROCm 上 vLLM CPU 層只有 4.1 GiB/s（vs pinned 50）——**送出成本**而非頻寬（F1 舊分析） | **I4** |
| **D** 預取／重疊 | InfiniGen、ShadowKV、KVPR、FreeKV、HCache、Cake、CachedAttention、Tokencake、KVFlow（L2 §3） | 單 GPU、無背景負載下有效 | GPU 忙時 SDMA vs blit、重疊何時變負（L2 G2）；壓縮 × 預取提前量（L2 G1） | **I4、I2、I3** |
| **E** prefix／復用 | vLLM APC、RadixAttention、Marconi、CacheBlend、EPIC、KVLink、APE、Block-Attention（L3 §1） | 單請求打平前沿（py-kvcache）；路由層門檻 | 重疊 × 容量 × **併發**的邊界；#44294 延後機制在 MI300X（L3 缺口 2） | **F1、F2** |
| **F** paging／壓力 | PagedAttention、vAttention、Jenga（L3 §2） | V1 搶佔只有重算 | 本次不單獨做 | — |
| **G** goodput／尾延遲 | DistServe、Sarathi-Serve、Llumnix、Mooncake、Cascade（L3 §3） | goodput 定義成熟 | 換成 P95／goodput 目標後，放置決定會不會翻轉（L3 缺口 3） | **F1**（同時算兩個目標） |
| **H** agent 多輪 | InferCept、Continuum、Tokencake、KVFlow、MORI、Parrot、Autellix（L3 §4） | 二／三選一的去留規則 | 四選一＋實測 κ＋跨硬體翻轉（L3 缺口 1）；公開 trace：Exgentic、SemiAnalysis、TraceLab、Mooncake（L3 §6） | **I3s → I3g** |

---

## 2. 缺口排序（給第三階段選題用；實驗結果出來後更新）

| # | 缺口 | 新意 | MI300X 上能不能 PoC | 最小證偽 | 何時停 | 新意風險 |
|:--|:--|:--|:--|:--|:--|:--|
| 1 | **GPU 忙時的搬運與重疊**（SDMA vs blit、重疊何時變負、多 session 共用鏈路與 GPU） | 前作都假設不干擾，我們已有干擾的實測 | 能（I4、I1） | SDMA 開關在每個 duty 差 <5%；聯合決定比各自 Cake 的 P95 好 <5% | 兩者都不成立 | 中：CacheFlow／CacheSmith 要看全文（L4） |
| 2 | **復用不划算的併發邊界**（含 #44294） | 單請求前沿已有，併發沒有；AMD 沒人量 | 能（V0→F2→F1） | 所有格子 APC＋卸載不比不復用差 >5%，且邊界在 C＝1 與 C＝64 間不動 | 同左 | 中：只重現 issue 只值一個工程修正 |
| 3 | **下一輪會用到哪些 block 的可預測性**（選擇性還原的前提） | 實測資料沒人公開 | 能（C1a，只要 forward） | 寫入時分數、摘要都≈隨機 | 同左 → 寫負面結果 | 低（是量測） |
| 4 | **agent 暫停的四選一＋實測 κ** | 延續 Tiara 主張（κ 跨硬體變動） | 先模擬（I3s） | κ 感知比最好的現成規則好 <5%，或兩平台選的動作相同 | 同左；或 Cascade 已做 | **高：Cascade 待查** |
| 5 | **等位元組下刪除 vs 降精度** | Cake 還原下沒人比 | 能（S2） | 量化在三個預算都不輸 | 同左 → 「刪除被量化支配」 | 低（多半是乾淨的負面結果） |
| 6 | **無損壓縮在 AMD 的盈虧與預取提前量** | AMD／CPU 端沒人量 | 能（S1） | BF16 無損 <1.15–1.2× | 同左 | 中：SplitZip 已在 H200 做 |
| — | 壓縮＋丟棄＋卸載聯合模型（L2 G4） | — | — | — | — | **最高**：很可能退化成「全 INT4＋G5b」 |

---

## 3. AMD／MI300X 這台機器能做與不能做的（L0 §3、L2、L3 §7.0）

- **能**：vLLM 0.28（APC、OffloadingConnector、TieringOffloadingSpec 的檔案層）、`--num-gpu-blocks-override`、`/metrics` 命中率；Triton kernel；CPU 端 zstd／zlib（容器 32 核）；`HSA_ENABLE_SDMA`（SDMA vs blit）；我們自己的 Restorer（Cake、G5b、drop、各種格式與融合反量化）。
- **不能**：QAT／IAA／DSA／AMX（EPYC 沒有）；hipFile 直通（看不到 NVMe、ROCm 7.2.2 太舊）；nvCOMP；CUDA-only 的稀疏 attention 官方實作（Quest、ShadowKV、DuoAttention、RetrievalAttention 等，L1）；LMCache（ROCm wheel 是 torch 2.11，期限內不建）。
- **環境更正**：容器記憶體 434.8 GiB（主機 2.2 TB 不能全用）；本地磁碟是 overlay（sda），實測讀 7,943–9,152 MiB/s；NFS 實測讀 377–447 MiB/s（第一階段 C2），限速器用的 0.331／6.98 GiB/s 是**參數**不是量測（`stage2_factcheck_and_toolbox.md` §1）。

---

## 4. 文獻檔索引

| 檔 | 內容 | 卡片數 |
|:--|:--|--:|
| `lit/L0_intel_suite_and_amd.md` | Intel 四件套查證、六成時間說法、AMD／ROCm 對應、48 條連結 | — |
| `lit/L1_eviction_selective_loading.md` | 方向 A＋Cascade：52 張卡（2025 下半年–2026 約 25 篇）、G1–G4、X1–X4 | 52 |
| `lit/L2_compression_offload_prefetch.md` | B 10＋C 9＋D 12 張卡、G1–G5、X1–X6、委託裡三個歷史數字的核實 | 31 |
| `lit/L3_reuse_serving_agentic.md` | E／F／G／H 約 45 篇、缺口 1–4、可用 trace 與產生器、X1–X4 | ~45 |
| `lit/L4_novelty_fulltext.md` | （跑中）Cascade、CacheSmith、CacheFlow、Where Should the KV Cache Live? 全文查新 | — |
