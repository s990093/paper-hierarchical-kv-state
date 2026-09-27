# Tiara 新穎性審查與 SOTA 盤點（審稿人視角）

- **日期**：2026-09-24（Semantic Scholar 查詢時間見 §4.3，UTC）
- **範圍**：(1) 35 篇本地全文的 baseline 引用圖；(2) 程式碼與整合狀態；(3) 新穎性威脅 (a)–(g)；(4) SOTA 判準與引用數；(5) 結論。
- **不重做**：`docs/PAPER_GUIDE.md`、`docs/RELATED_WORK_WEAKNESSES.md`、以及協調者提供的 `scratchpad/remote/`（SOTA_MATRIX_20260919、PAPERS_EXPLAINED*、ADVISOR_REPLY_20260919、EXPERIMENTS_20260919、results/*MI300X*）。本文只補它們沒涵蓋的部分，並在附錄 A 列出衝突。
- **未修改任何專案檔案。**

### 證據等級（每一條結論都標）

| 標記 | 意思 |
|---|---|
| 🟩 | 我本人讀過一手全文（本地 `papers_txt/` 或本次下載到 `scratchpad/extra_txt/` 的 PDF 抽文字），引號內為原文，附頁碼 |
| 🟨 | 一手網頁／官方文件／原始碼（附 URL）；或 WebFetch 回傳的原文摘錄（附 URL，可能經小模型摘要，關鍵句已盡量取逐字引號） |
| 🟥 | 只看到搜尋摘要或標題，**未查證**；本文不據此下結論 |
| （我們的判讀） | 推論，不是原文 |

**已遵守的紅線**：SOTA_MATRIX §6 與 PAPERS_EXPLAINED_20260920 §8.3 共 24 條被 3 票推翻的 claim，本文一條都不引用（例如「Strata 沒有 compute-vs-load policy」「Strata 只有 LRU、沒有任何 policy」「Bidaw 證明 ShareGPT 零 headroom」）。凡涉及 Strata，只引逐字原文。

---

## 0. 結論先講（審稿人摘要）

| 主張 | 判決 | 最致命的先行工作 | Tiara 還剩什麼 | 新實測（remote）造成的影響 |
|---|---|---|---|---|
| **發現 1**：重算成本隨位置線性、SSD 固定 → 交叉點 P\*，**寫入時**依位置決定要不要寫 SSD，寫入頻寬省 4–7× | **部分已知（窄新）** | Pensieve（前段先丟、依重算成本逐出）、py-kvcache（依硬體 break-even「不載也不存」）、Dynamo KVBM／SGLang HiCache／vLLM 的寫入過濾、AsymCache（位置線性成本逐出） | 「以**區塊絕對位置**對 **SSD 取回成本**的交叉點，套在 **SSD 寫入路徑**上」這個組合，本次未查到 | MI300X 上 P\* 只有 Llama-8B 為正（3,396 token），另 3 個模型 < 0（SSD 永遠較便宜）→ 規則在快 NVMe 平台上退化；傳輸項被 vLLM ROCm 描述符粒度主導 → P\* 是**實作性質**，不只是硬體性質 |
| **發現 2**：三個決策（幾 bit／放哪／算或載）互相打架 | **部分已知，且被自己的數據轉向** | EvicPress／AdaptCache（「壓縮與放置必須聯合」）、CacheGen（每 chunk 精度或送文字重算）、QEvict（精度與駐留聯合）、KV Survey O7/C5 | 「78 萬次降級、+0.004%」是 Tiara 模擬器自身 LRU 排序下的現象，審稿人會視為設計缺陷而非發現 | 全域 INT4＋原廠 LRU 就拿到 oracle headroom 的 91.8%／103.9% → 量化與放置是**替代品**；「必須聯合」在純延遲目標下不成立，只在品質約束下可能成立，而 ε 在平台 B 量不出來 |
| **發現 3**：學習式預測器沒把握時比 LRU 差 2.7–5.6×，需要「認輸退回 LRU」 | **已被做過** | **LCR/LARU（arXiv 2509.20979）**：對 SGLang RadixTree 的 KV prefix cache 做 learning-augmented LRU，1-consistent、O(k)-robust，「Even with poor predictions … remains close to LRU」；另有 learning-augmented caching 理論（ICML'18、APPROX'20、SODA'22）、HALP、Marconi（α=0 退回 LRU） | 多階、異質動作成本（含重算）下的 robust fallback；以「分數擠在一起」而非「偵測到誤判」觸發 | MI300X 上 AUC 0.917–0.922 的預測器仍輸 tier_fs 9.7–47.3% → 失敗主因不是「沒把握」而是「可操作空間小」；且「最佳 baseline」在 27% 的自助抽樣中會翻轉 → 「vs best baseline」的比較本身不穩 |
| **轉向**：Cake 式雙向重疊 → 前 1/(1+κ) 讀回時不從 SSD 載 → 多輪只追加、會合點單調後移 → 那段**永遠不必寫入**（寫入端准入、SSD 當有寫入預算的快取） | **部分已知（窄新，機制層級）** | **HCache**（重疊還原下「some layers may not even need to be stored because they can be recomputed from tokens」，layer 維度）、**CacheFlow**（兩指標最佳切點 ℓ = L·T_io/(T_comp+T_io)）、Pensieve（前段丟、只留文字）、HILOS／HBF／KVBM（寫入量與耐久度當約束） | token 維度＋寫入時＋「append-only → 會合點單調」這個論證；本次未查到有人寫出 | 需先回答：會合點在讀取時是動態的（Cake 自己寫「the merging point … shifts accordingly」）、前綴共享、vLLM/SGLang 只認連續前綴命中且尾巴先逐出、MI300X 上規則退化 |

**最該正面引用並劃清界線的 5 篇**：Pensieve（EuroSys'25）、HCache（EuroSys'25）、Cake（ICML'25）＋CacheFlow（arXiv 2604.25080）、py-kvcache（arXiv 2609.11744）、LCR/LARU（arXiv 2509.20979）。工業界的 Dynamo KVBM 磁碟過濾、SGLang HiCache 寫入策略、vLLM `store_threshold`／`max_offload_tokens` 必須當 baseline，不只是引用。詳見 §5.3。

---

## 1. Baseline 引用圖

### 1.1 方法與限制

- 逐篇 grep 評測章節的 `Baseline(s)`／`We compare`／`compared with` 段落，讀原段落後抽出系統名，附頁碼（`=== [page N] ===` 標記）。檔案：`scratchpad/papers_txt/<key>.txt`。統計腳本：`scratchpad/research/baseline_graph.py`。
- **選樣偏差**：這 35 篇是本專案自選的清單，不是隨機樣本；「被比最多」只代表**在這個清單裡**。所以 §1.4 另外補了清單外、本次讀過全文的 2025–2026 論文，以及 Semantic Scholar 引用語境。
- 「vLLM」「LRU」「full prefill」是**地板**，不是研究 SOTA（教授的說法與此一致）；統計時照實列出，但判定 SOTA 時排除。

### 1.2 論文 → 它比較的系統（35 篇，🟩，附頁碼）

**第 1 類 多層儲存**

| 論文 | baseline（頁碼） |
|---|---|
| Strata（OSDI'26） | vLLM v0.8.5；vLLM-LMCache（LMCache v0.2.1）；TensorRT-LLM v0.17.0；TRT-LLM-HiCache（CPU offload）；SGLang v0.4.5；SGLang-HiCache（作者自建、layer-wise `cudaMemcpyAsync`，「in line with prior work including CachedAttention, Pensieve and FlashGen」）（p9）；消融在 SGLang-HiCache 之上疊 Strata-IO／Strata-Schedule-Only／Strata-IO-LPM（p11） |
| Bidaw（FAST'26） | vLLM、CachedAttention、FlashGen、ideal（p11）；「Since CachedAttention and FlashGen are closed-source, we implement CachedAttention and FlashGen based on vLLM」（p11）；逐出比較：queue-enhanced（CachedAttention）、LFU、LRU、FIFO（p12） |
| MTDS（C&IS'26） | 原生 vLLM（開 offloading）、Mooncake、LMCache（p10）：「they are limited to fixed strategies, namely either fully reusing all KV caches or fully recomputing them」 |
| FlexGen（ICML'23） | DeepSpeed ZeRO-Inference、HF Accelerate、Petals（p7） |
| CachedAttention（ATC'24） | 只有 Recomputation（RE）（p9） |
| Mooncake（FAST'25） | vLLM-[4M]／vLLM-[20M]（p15–17）；trace 分析另比 LRU／LFU／LengthAwareCache（p7–8） |
| KVDrive（PACMMOD'26） | Original（全 GPU）、FlexGen、Quest、ShadowKV、PQCache、MagicPIG、RetroInfer（±E）（p15）；逐出消融對 LRU（p17） |
| Tutti（preprint'26） | vLLM 0.12／0.17 HBM；LMCache-DRAM(-LW)、LMCache-SSD、LMCache-GDS（p8） |
| LMCache（preprint'25） | Basic vLLM v0.10.2、vLLM v0.11.0 CPU offloading、Commercial #1／#2（p11）；vLLM native PD（p13） |

**第 2 類 可恢復驅逐**

| 論文 | baseline |
|---|---|
| ArkVale（NeurIPS'24） | StreamingLLM、H2O、TOVA（p8） |
| QEvict（preprint'26） | StreamingLLM、SnapKV、AdaKV、CriticalKV、DefensiveKV、Layer-DefensiveKV；量化：KIVI、KVQuant、ZipCache（p10） |

**第 3 類 以 query 挑 KV**

| 論文 | baseline |
|---|---|
| InfiniGen（OSDI'24） | 環境：UVM、FlexGen；方法：H2O、Quantization（p9） |
| Quest（ICML'24） | H2O、TOVA、StreamingLLM（p6） |
| ShadowKV（ICML'25） | Quest、Loki、InfiniGen（p7）；吞吐量對 full attention（p8） |

**第 4 類 重算 vs 傳輸**

| 論文 | baseline |
|---|---|
| Cake（ICML'25） | Compute-only：vLLM v0.6.2 chunked prefill；I/O-only：LMCache v0.1.4（p5） |
| KVPR（ACL-F'25） | DeepSpeed Inference、HF Accelerate、FlexGen（p6） |
| HCache（EuroSys'25） | DeepSpeed-MII（重算）、AttentionStore（作者在 DeepSpeed-MII 上重做成 KV offload baseline）、Ideal（p9） |
| CacheBlend（EuroSys'25） | Full KV recompute、Prefix caching（SGLang 技術）、Full KV reuse（PromptCache）、MapReduce、MapRerank（p11） |

**第 5 類 學習式驅逐**

| 論文 | baseline |
|---|---|
| KVP（ICML'26） | TOVA、SnapKV、Random、StreamingLLM、LagKV、KeyDiff、K-Norm（p7）；JudgeQ（附錄 A.4） |
| ForesightKV（ICML'26） | SnapKV、H2O、R-KV（p6、p16）；附錄另有 Duo-Attention、RPC 等（p19） |
| LookaheadKV（ICLR'26） | SnapKV、PyramidKV、StreamingLLM、LAQ、SpecKV（p7） |
| TRIM-KV（ICLR'26） | SeerAttn-R、R-KV、SnapKV、H2O、StreamingLLM（p7）；附錄 KeyDiff（p17）、LocRet（p19） |
| Marconi（MLSys'25） | Vanilla inference、vLLM+、SGLang+（SGLang+ 用 Marconi 的准入但逐出仍是 LRU）（p8） |
| SAECache（preprint'26） | LRU（vLLM 式）、LPC（NeurIPS'25）（p8） |

**第 6 類 品質／壓縮／放置最佳化**

| 論文 | baseline |
|---|---|
| KVServe（SIGCOMM'26） | CacheGen、KIVI、DuoAttention（p10） |
| EvicPress（preprint'25） | Prefill（vLLM v0.11.2）、Eviction-only（LRU）、keydiff／knorm／snapkv＋LRU、IMPRESS（p8） |
| AdaptCache（BigMem'25） | Without Compression、KIVI LRU、StreamingLLM LRU、Prefill（p2） |
| LeoAM（preprint'25） | H2O-like、H2O-like-chunked、Prefetch-based（InfiniGen 式）、Full cache（p10） |
| OrbitFlow（VLDB'26） | DeepSpeed-Inference、FlexGen、FlexGen+、SLO-aware Offloading、Dynamic Heuristic（自家變體）（p9） |
| Het-Mem（CAL'25） | Ideal、Static Placement、Reactive Scheduling、Page Granularity Scheduling（p3–4） |

**第 7、8 類**

| 論文 | baseline |
|---|---|
| KIVI（ICML'24） | 16-bit 全精度與 fake-quant 組合（p3、p6） |
| KVTuner（ICML'25） | KIVI-8、KIVI-4、KIVI-K8V4、per-token-asym（p8） |
| Bottlenecks（MLSys'26） | 無 baseline；量測對象為 vLLM v0.10.1＋LMCache v0.3.5（p7） |
| YAKV（preprint'26） | ShadowKV、ArkVale、LRQK、InfiniGen（p8） |
| KV Survey（ACL-F'26） | 不適用 |

### 1.3 反向統計：被最多篇拿來當 baseline（35 篇內）

| 次數 | 系統 | 誰拿它比 |
|---|---|---|
| 9 | **vLLM**（地板） | Strata、Bidaw、MTDS、Cake、EvicPress、Mooncake、Tutti、LMCache、Marconi |
| 7 | **LRU**（地板） | AdaptCache、Bidaw、EvicPress、Het-Mem、KVDrive、Mooncake、SAECache |
| 7 | StreamingLLM | AdaptCache、ArkVale、KVP、LookaheadKV、QEvict、Quest、TRIM-KV |
| 6 | Full prefill／重算（地板） | AdaptCache、CacheBlend、CachedAttention、EvicPress、HCache、Marconi |
| 6 | H2O | ArkVale、ForesightKV、InfiniGen、LeoAM、Quest、TRIM-KV |
| 6 | SnapKV | EvicPress、ForesightKV、KVP、LookaheadKV、QEvict、TRIM-KV |
| 4 | **LMCache** | Cake、MTDS、Strata、Tutti |
| 4 | KIVI | AdaptCache、KVServe、KVTuner、QEvict |
| 4 | FlexGen | InfiniGen、KVDrive、KVPR、OrbitFlow |
| 4 | DeepSpeed（ZeRO-Inference／Inference／MII） | FlexGen、HCache、KVPR、OrbitFlow |
| 3 | SGLang（含 +／prefix caching） | CacheBlend、Marconi、Strata |
| 3 | InfiniGen | LeoAM、ShadowKV、YAKV |
| 3 | TOVA | ArkVale、KVP、Quest |
| 2 | **CachedAttention／AttentionStore** | Bidaw、HCache（兩者都是**重做**，因為閉源） |
| 2 | Quest、ShadowKV、R-KV、KeyDiff、K-Norm、LFU、HF Accelerate | 略 |
| 1 | Mooncake、IMPRESS、FlashGen、LPC、CacheGen、TensorRT-LLM、… | 略 |
| **0** | **Strata、Bidaw、MTDS、AdaptCache、EvicPress、Cake、KVPR、HCache、Pensieve** | 35 篇內**沒有任何一篇**拿它們當 baseline |

### 1.4 清單外（本次讀過全文）的 baseline 選擇與引用語境

🟩 = 我讀過該論文全文段落。

| 論文（2025–2026） | 比了誰 | 對 SOTA 判定的意義 |
|---|---|---|
| **CacheFlow**（arXiv 2604.25080，UIUC） | vLLM、SGLang（HiCache）、LMCache v0.3.1、**Cake**（「the state-of-the-art hybrid restoration approach」，p7）🟩 | Cake 在「重算 vs 載回」格被當成 SOTA 混合法 |
| KVFlow（NeurIPS'25） | SGLang、SGLang w/ HiCache（p7）🟩 | HiCache 是 SGLang 系的放置機制 baseline |
| IMPRESS（FAST'25） | ReComp、AS-like（「Since it is not open-source, we reimplement it」）、AS+H2O+LRU／LFU（p11）🟩 | CachedAttention 第三次被重做 |
| AsymCache（arXiv 2606.02964） | vLLM-LRU、Max-score、Pensieve+MSA（p8–9）🟩 | Pensieve 的位置感知逐出被當 baseline |
| LCR／LARU（arXiv 2509.20979） | SGLang LRU、FPB（盲從預測）、HF（HALP 式 LRU 預篩 4 個候選）（p7）🟩 | 學習式＋fallback 這一格的直接對手 |
| Continuum（arXiv 2511.02230） | vLLM、Autellix+、InferCept（p9–10）🟩 | InferCept 是代理／中斷負載的慣用 baseline |
| CacheWise（arXiv 2606.16824） | vLLM（block-level LRU）、InferCept（p9）🟩 | 同上 |
| Pensieve（EuroSys'25） | vLLM v0.2.0、TensorRT-LLM v0.12.0（p9）；逐出策略對 LRU（p12）🟩 | — |
| py-kvcache（arXiv 2609.11744） | LMCache、vLLM KV Offload、llm-d（p1–2、p6）🟩 | 寫入/載入門檻格的最新對手 |
| Semantic Scholar 引用語境（🟨，下限） | SparKV（IEEE IoT-J 2026）：「we implement a strong hybrid baseline based on [25]」（[25]=Cake）；FlashGen（ASPLOS'25）比 CachedAttention；SYMPHONY、KunServe、LAMPS 比 InferCept；EvicPress 比 IMPRESS | 查詢：`/paper/{id}/citations?fields=contexts,…`，只取含 compare/baseline 字樣的語境，結果存 `research/s2/s2_cites.json` |

### 1.5 各決策格「事實上的 SOTA baseline」（被比最多＋夠新＋有開源）

| 決策格 | 事實上的 SOTA baseline | 依據 | 注意 |
|---|---|---|---|
| ① 精度／壓縮（單層） | **KIVI**（量化；4/35；[jy-yuan/KIVI](https://github.com/jy-yuan/KIVI)）；token dropping 用 **SnapKV**（6/35；[FasterDecoding/SnapKV](https://github.com/FasterDecoding/SnapKV)）、**H2O**（6/35；[FMInference/H2O](https://github.com/FMInference/H2O)）、**StreamingLLM**（7/35；[mit-han-lab/streaming-llm](https://github.com/mit-han-lab/streaming-llm)） | 被比次數最多、皆開源 | 這些都只在 GPU 內，**不是**多階放置的對手 |
| ①×② 壓縮＋層級 | 「**固定壓縮＋LRU**」（AdaptCache、EvicPress 都拿它當主 baseline）＋ **EvicPress** 本身 | 最新、唯一同格 | EvicPress 無官方程式碼（§2）；35 篇內與 S2 引用語境中**沒有人**拿 EvicPress 當 baseline（S2：7 次引用、0 次比較語境） |
| ② 放哪一層（機制） | **LMCache**（4/35；開源；vLLM／SGLang／Dynamo／llm-d 均整合）與 **SGLang HiCache**（= Strata 的上游實作；KVFlow、CacheFlow 拿它比） | 被比最多＋已併入主流框架 | vLLM 原生 `OffloadingConnector` 是**底座**不是對手；但它的 `store_threshold`／`max_offload_tokens` 是寫入准入格的必比 baseline（§2.4） |
| ③ 逐出 | 工業地板 **LRU**；學術 de facto **CachedAttention 的 queue-enhanced**（Bidaw、HCache、IMPRESS 都重做它，閉源）；代理負載 **InferCept**（開源 [WukLab/InferCept](https://github.com/WukLab/InferCept)）；學習式 prefix 逐出 **LPC**（NeurIPS'25，開源 [yangdsh/LPC](https://github.com/yangdsh/LPC)）；位置／重算成本感知 **Pensieve**（EuroSys'25，被 AsymCache 當 baseline；程式碼未找到） | 被比次數＋頂會＋開源 | Bidaw（FAST'26）太新，尚無人比 |
| ④ 重算 vs 載回 | 端點：vLLM prefill（compute-only）、LMCache（load-only）；混合：**Cake**（CacheFlow、SparKV 把它當 SOTA 混合法；無公開程式碼） | 被當 SOTA 混合 baseline | CacheFlow 摘要宣稱「reduces Time-To-First-Token (TTFT) by 10%–62% over existing advances」（比較對象含 vLLM、SGLang HiCache、LMCache、Cake；🟩），但只是 preprint、未見程式碼 |
| ⑤ 聯合（精度×層級×重算） | **無公認 SOTA**。最接近：EvicPress（壓縮×層級）、MTDS（層級×重算） | S2：EvicPress 0 次、MTDS 0 次比較語境 | 這一格空著，是機會也是風險：沒有人比過，代表社群尚未認可這一格的重要性（我們的判讀） |

---

## 2. 程式碼與整合狀態

### 2.1 總表

| 系統 | 開源？ | Repo／文件 | 併入主流框架？ | 證據與備註 |
|---|---|---|---|---|
| **Strata**（OSDI'26） | 論文無 repo URL；機制在 SGLang HiCache | [sgl-project/sglang](https://github.com/sgl-project/sglang)；[HiCache design](https://docs.sglang.io/advanced_features/hicache_design.html)；[LMSYS blog](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/) | ✅ SGLang（詳 §2.2） | 🟩 摘要：「Implemented as part of SGLang and deployed in production」（p2） |
| **SGLang HiCache** | ✅ | 同上；預設值見 [fields/memory.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/arg_groups/fields/memory.py) | ✅（SGLang 本體；Dynamo 亦有 HiCache 整合頁） | 🟨 寫入策略三種；L3 後端 file／mooncake／hf3fs／nixl／aibrix 等；ROCm 見 §2.2 |
| **MTDS**（C&IS'26） | 🟥 未能確認 | 論文只有 Data Availability（ShareGPT、Random 資料集，p15）；GitHub 有同名 [nosakii-outlook/MTDS](https://github.com/nosakii-outlook/MTDS)，README 僅兩句（「Paper accepted at Dec.10.」） | ❌ | 無法確認是官方完整實作 |
| **Bidaw**（FAST'26） | 系統碼：未找到；**trace 公開** | [ShipengHu-777/Interactive-conversation-workload](https://github.com/ShipengHu-777/Interactive-conversation-workload)（作者 GitHub 只有此 repo 與個人頁） | ❌ | 🟩 自述在 vLLM 上重做 CachedAttention、FlashGen（p11） |
| **FlashGen**（ASPLOS'25，Jeong & Ahn） | 未找到 | [DOI 10.1145/3676641.3716245](https://dl.acm.org/doi/10.1145/3676641.3716245) | ❌ | 🟩 Bidaw 稱其閉源；機制：優先排 KV 能放進 GPU 的請求＋inclusive caching（capacity layer 永遠有一份）（Bidaw p4） |
| **CachedAttention／AttentionStore**（ATC'24） | 未找到 | [USENIX 頁](https://www.usenix.org/conference/atc24/presentation/gao-bin-cost) | ❌ | 🟩 Bidaw（p11）、IMPRESS（p11）都說閉源並重做 |
| **Cake**（ICML'25） | 未找到 | 作者頁只有 PDF 連結（[shuoweijin.com](https://shuoweijin.com/publication/cake/)） | ❌ | 🟩 附錄 B：「We implement Cake by extending LMCache … and integrating it with vLLM」（p11）；SOTA_MATRIX 也判「程式碼 ❌」，一致 |
| **AdaptCache**（BigMem'25） | 未找到 | — | ❌ | 3 頁 workshop；未指名 serving engine（與 SOTA_MATRIX 一致） |
| **EvicPress**（preprint'25） | 官方未找到 | GitHub 有 [MrBottleTree/evicpress-core](https://github.com/MrBottleTree/evicpress-core)（無 README，無法確認） | ❌ | 🟩 實作在 vLLM v0.11.2＋LMCache v0.3.9post2 上，約 3K 行（p7） |
| **LMCache** | ✅ | [LMCache/LMCache](https://github.com/LMCache/LMCache)；[設定文件](https://docs.lmcache.ai/api_reference/configurations.html) | ✅ vLLM（`LMCacheConnectorV1`）、SGLang；[Dynamo 整合頁](https://docs.nvidia.com/dynamo/v1.0.1/integrations/lm-cache)；llm-d 列為替代實作；[加入 PyTorch 生態](https://pytorch.org/blog/lmcache-joins-pytorch-ecosystem/) | 🟩 論文：8 種後端、4 種處理器（含 AMD）、2 個引擎（p16）。⚠️ remote RUNLOG_MI300X：在 MI300X 上因 CUDA 專屬依賴裝不起來（NOT_MEASURED） |
| **Mooncake** | ✅ | [kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake) | ✅ Transfer Engine 為 vLLM v1 PD 的 KV connector；Mooncake Store 為 SGLang HiCache L3 後端（[設計頁](https://kvcache-ai.github.io/Mooncake/design/hicache-design.html)）；LMCache 後端；[加入 PyTorch 生態](https://pytorch.org/blog/mooncake-joins-pytorch-ecosystem/) | 🟨 |
| **CacheBlend**（EuroSys'25） | ✅ | [YaoJiayi/CacheBlend](https://github.com/YaoJiayi/CacheBlend) | ✅ LMCache 的 blending 功能（[文件](https://docs.lmcache.ai/kv_cache_optimizations/blending.html)） | 🟨 |
| **HCache**（EuroSys'25） | 未找到 | — | ❌ | 網路搜尋（含 thustorage 組織與作者頁）未見官方 repo |
| **KVDrive**（S2 標 Proc. ACM Manag. Data，DOI 10.1145/3802077） | 未找到 | — | ❌ | — |
| **Tutti**（preprint'26） | ✅ | [xPU-IO/Tutti](https://github.com/xPU-IO/Tutti)（v0.1.1，基於 GeminiFS） | ❌（獨立 KV 物件儲存） | 🟨 README |
| **Pensieve**（EuroSys'25） | 未找到 | — | ❌ | 搜尋無果 |
| InferCept（ICML'24） | ✅ | [WukLab/InferCept](https://github.com/WukLab/InferCept) | ❌ | — |
| KVPR（ACL-F'25） | ✅ | [chaoyij/KVPR](https://github.com/chaoyij/KVPR) | ❌ | 🟩 論文自述 |
| Marconi（MLSys'25） | ✅ | [ruipeterpan/marconi](https://github.com/ruipeterpan/marconi) | ❌ | artifact |
| LPC（NeurIPS'25） | ✅ | [yangdsh/LPC](https://github.com/yangdsh/LPC) | ❌（改 vLLM 0.7.3） | README |
| HILOS（ASPLOS'26） | ✅ | [hongsunjang/HILOS](https://github.com/hongsunjang/HILOS) | ❌ | 🟩 摘要 |
| py-kvcache（arXiv 2609.11744） | ✅ | github.com/atlarge-research/py-kvcache | ❌（vLLM KV Offload connector） | 🟩 摘要 |
| **vLLM OffloadingConnector** | ✅ | [KV Offloading Usage](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/)（[原始 md](https://github.com/vllm-project/vllm/blob/main/docs/features/kv_offloading_usage.md)）；[blog 2026-01-08](https://vllm.ai/blog/2026-01-08-kv-offloading-connector)；[tiered blog 2026-09-10](https://vllm.ai/blog/2026-09-10-tiered-kv-offloading) | 本體 | 詳 §2.4 |
| **NVIDIA Dynamo KVBM** | ✅ | [KVBM Guide](https://docs.nvidia.com/dynamo/v1.1.1/user-guides/kv-cache-offloading)；[設定參考](https://docs.nvidia.com/dynamo/reference/components/kvbm-configuration) | 整合 vLLM、TensorRT-LLM、SGLang | 詳 §2.4 |
| **llm-d** | ✅ | [Tiered Prefix Cache](https://llm-d.ai/docs/well-lit-paths/foundations/tiered-prefix-cache)；[KV offloader](https://llm-d.ai/docs/0.7/architecture/advanced/kv-management/kv-offloader) | 用 vLLM `OffloadingConnector`（HBM→CPU→filesystem）；LMCache、SGLang HiCache 為替代實作 | 詳 §2.4 |

### 2.2 Strata 與 SGLang HiCache 的關係（使用者要求確認）

**結論**：Strata 是在 SGLang HiCache 之上做的研究版本，其 I/O 機制已出現在 upstream HiCache；「HiCache 是 Strata 的上游實作」可以寫，但要注意兩處不一致。

證據鏈：
1. 🟩 Strata 摘要：「Implemented as part of SGLang and deployed in production」（p2）；架構圖與 §4 用的資料結構就是 **HiRadixTree**，「an extension to SGLang's RadixTree」（p6）。
2. 🟩 Strata 自己的評測把 **SGLang-HiCache** 當 baseline（layer-wise `cudaMemcpyAsync` 版，p9），並「On top of SGLang-HiCache」疊出 Strata-IO／Strata-Schedule 做消融（p11）。也就是說，**論文裡的 SGLang-HiCache 指的是 Strata 之前的版本**。
3. 🟨 LMSYS blog「SGLang HiCache」（2025-09-10）作者是 Zhiqiang Xie（= Strata 第一作者），內容包含 HiRadixTree、「GPU-assisted I/O kernels」（CPU–GPU 傳輸最高 3×）、「page-first」layout、三種寫入策略、Mooncake／3FS／NIXL 後端。Tutti 的參考文獻也把這篇 blog 列在 Strata 旁邊（tutti2026 p14）。
4. 🟨 SGLang HiCache design 文件：HiRadixTree「records where that KV cache is stored—whether in local GPU memory, CPU memory, L3 storage」；`kernel` I/O backend＝GPU-assisted kernels；prefetch 策略 `best_effort`／`wait_complete`／`timeout`。**文件中未見 delay-hit 感知排程的敘述** → Strata §4.3 的 cache-aware scheduler 是否完整進入 upstream：**未查證**。

**兩處不一致（寫論文時要注意）**：
- **預設寫入策略不同**：Strata 論文寫「The third and default policy is selective-write-through … The default threshold is set to 2 when write bandwidth is abundant」（p9）；但 SGLang main 的 CLI 預設是 `hicache_write_policy = "write_through"`（[fields/memory.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/arg_groups/fields/memory.py)，2026-09-24 抓取；[best practices 文件](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/hicache_best_practices.mdx) 範例也都用 `write_through`）。引用時要寫明是哪一個版本。
- 同檔的 L3 後端清單（2026-09-24）：`file, sim, mooncake, npu_memcache, hf3fs, nixl, aibrix, dynamic, eic, simm, mori, shm, tensorcast`；I/O backend 預設 `kernel`、host layout 預設 `page_first`。

**ROCm 支援**（🟨）：
- [PR #37152](https://github.com/sgl-project/sglang/pull/37152)「[ROCm] Widen the HiCache JIT copy rounds and enable the K-only host pool」，2026-09-19 merged。
- [PR #40570](https://github.com/sgl-project/sglang/pull/40570)「[AMD] Enable HiCache for GLM-5.2 MI355X throughput recipe」，2026-09-21 merged。
- AMD fork [xiaobochen-amd/sglang PR #28](https://github.com/xiaobochen-amd/sglang/pull/28)（2026-08-30）：「The kernel backend faults on ROCm on the first transfer, so hierarchical cache on ROCm works only if the user explicitly passes `--hicache-io-backend direct`」，並報「12 FAULT / 0 PASS across every kernel case」對 `direct`「11 PASS / 0 FAULT」。
- → **HiCache 在 ROCm 上可用，但 GPU-assisted `kernel` 路徑到 2026-09 才陸續修好**；在 MI300X 上跑 HiCache 當 baseline 時要記錄用的是 `kernel` 還是 `direct`。

**寫入策略原文**（🟩 Strata p9，這段對轉向很重要）：
> 「A single write-once policy is insufficient for hierarchical context caching because different serving workloads impose different trade-offs among **write bandwidth, cache capacity, durability, and future reuse**. Writing every generated KV page immediately maximizes reuse and persistence, but it can waste host memory and write bandwidth for one-off requests.」

### 2.3 Bidaw 比的 CachedAttention 與 FlashGen

- **FlashGen** ≠ FlexGen。它是 Jeong & Ahn，「Accelerating LLM Serving for Multi-turn Dialogues with Efficient Resource Management」，ASPLOS'25（S2：30 次引用、2 次 influential）。🟩 Bidaw 的描述：「the compute engine schedules requests by prioritizing those whose KVs can fit within the available GPU memory. Second, FlashGen adopts inclusive caching, where a copy of KVs in the performance layer is also maintained in the capacity layer」（p4）；「FlashGen enables re-computation on some requests」（p11）。
- **CachedAttention**：queue-enhanced 逐出，「combines the past KV accesses information with the queuing information」（Bidaw p4）。
- **Bidaw 怎麼重做**：🟩「Since CachedAttention and FlashGen are closed-source, we implement CachedAttention and FlashGen based on vLLM」（p11）。Bidaw 自己也用 inclusive caching：「we will maintain a copy of KVs in the capacity layer to avoid the large write traffic during eviction」（p6）——也就是**所有 KV 都寫進 SSD**。
- **對 Tiara 的意義**：這是「在同一底座上重做閉源 baseline」的 FAST 級先例（IMPRESS 與 HCache 也這樣重做 AttentionStore）。

### 2.4 主流框架的「寫入／載入」旋鈕（與轉向直接相關）

| 框架 | 旋鈕 | 原文（🟨） | 方向 |
|---|---|---|---|
| vLLM `OffloadingConnector` | `store_threshold` | 「Min lookups before a block is offloaded. Values ≥ 2 are rejected by `TieringOffloadingSpec`.」（單階 spec 才有） | 依次數准入 |
| vLLM | `max_offload_tokens`（per-request，experimental） | 「Only the first `max_offload_tokens` tokens of the request are offloaded; blocks beyond that point are skipped on the store path. This is useful when a known prefix (e.g., a system prompt or shared context) is worth caching but later request-specific tokens are not.」 | **依位置准入，但只存前段**（與 Tiara 相反） |
| vLLM | `max_load_tokens`（per-request，experimental） | 「tokens beyond the aligned load cap are recomputed」 | 載入前段、**重算後段**（與 Cake 相反，與 MTDS Partial Load 同向） |
| vLLM | `eviction_policy` | 內建 `lru`／`arc`，或自訂 `CachePolicy`（`cache_policy_module_path`） | 只管 CPU 主階逐出；remote ADVISOR_REPLY 已指出 DROP 與精度降級無法經此介面表達 |
| vLLM | 多階 | 「Only the CPU primary tier has direct GPU access. Secondary tiers cannot read from or write to GPU memory」；secondary 支援 `fs`／`obj`／`p2p`；「OffloadingConnector currently supports CUDA, ROCm, and XPU only」；tiered 框架「available in vLLM since v0.22」（tiered blog） | — |
| vLLM prefix caching | 逐出順序 | 「The freed blocks are added to the tail of the free queue in the *reverse* order. This is because the last block of a request must hash more tokens and is less likely to be reused by other requests.」（[設計文件](https://docs.vllm.ai/en/stable/design/prefix_caching/)，協調者與我各自查證） | **尾巴先逐出** |
| Dynamo KVBM | 磁碟過濾 | 「Disk offload filtering is enabled by default to **extend SSD lifespan**. The current policy only offloads KV blocks from CPU to disk if the blocks have frequency ≥ 2.」；「Frequency doubles on cache hit (initialized at 1) and decrements by 1 on each time decay step」；關閉：`DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER=true`（[KVBM Guide](https://docs.nvidia.com/dynamo/v1.1.1/user-guides/kv-cache-offloading)）；設定參考另列 offload policy `pass_all`／`presence`／`presence_lfu`（`min_lfu_count` 預設 8） | **依頻率准入，理由明寫是 SSD 壽命** |
| SGLang HiCache | 寫入策略 | `write_through`（預設）／`write_through_selective`（hit count 超過門檻才寫）／`write_back`（逐出時才寫） | 依次數准入 |
| SGLang | 逐出策略 | [evict_policy.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/evict_policy.py)：LRU、LFU、FIFO、MRU、FILO、Priority、**TLRU（Tail-Optimized LRU, arXiv 2510.15152）**、SLRU | TLRU 把超過 TTFT 預算的**尾端** token 標成「TEL-safe」先逐出 |
| LMCache | 設定 | `cache_policy`（LRU/LFU/FIFO，預設 LRU）、`min_retrieve_tokens`（命中太少就「skip retrieve」）、`save_decode_cache`（預設 false）、`save_unfull_chunk`（預設 false） | 有載入門檻，**未見依成本的寫入准入** |
| llm-d | tiered prefix cache | 用 vLLM `OffloadingConnector`；「The connector does not evict data from the shared tier -- capacity is managed by the storage system or by an external controller」 | — |

**（我們的判讀）一個實作層的硬傷**：vLLM／SGLang 的前綴語意是「只算連續前綴命中」，而且尾巴先逐出。Tiara 轉向要「不存前段、只存後段」，那些後段 block 在原生查找路徑上**永遠不會被算成命中**，除非改 scheduler／connector 支援「前段重算＋後段載入」（Cake／Pensieve 式）。vLLM 現有的兩個 per-request 旋鈕 `max_offload_tokens`、`max_load_tokens` 都是「保前段」方向。這與 remote ADVISOR_REPLY 指出的「六態有四態無法經 CachePolicy 表達」是同一類問題，要在論文裡當成**必須實作的機制**處理，不能只在模擬器裡成立。

---

## 3. 新穎性威脅 (a)–(g)

### (a) 依位置／重算成本在**寫入時**決定要不要存 SSD（或 CPU）

**判決：「依位置／重算成本決定逐出與丟棄」已被做過；「寫入時准入」已有依次數、依前綴長度、依位置（方向相反）三種做法；「以區塊絕對位置對 SSD 取回成本的交叉點 P\* 做 SSD 寫入准入」本次未查到 → 部分已知（窄新）。**

| 先行工作 | 做了什麼（原文） | 何時決定 | 與 Tiara 的差距 |
|---|---|---|---|
| **Pensieve**（EuroSys'25）🟩 [arXiv 2312.05516](https://arxiv.org/abs/2312.05516) | 「It evicts cached data to the next tier (or discards it), preferring conversations that have been inactive for longer and/or those that are **cheaper to recompute**. The eviction is done at the granularity of a chunk of tokens」（p2）；「it preferentially evicts tokens from the **leading end** of a conversation's history context … leading tokens of a conversation are cheaper to recompute than trailing ones」（p6）；retention value「V = Cost(s,l)/T」，Cost 由離線 profiling 內插（p6）；「When the CPU cache runs out of space, the same eviction policy (§4.3.1) is used to decide which KV-tokens to **drop**」（p6）；結果：「the GPU cache generally holds the request's latest tokens and the CPU cache holds the middle, while the earlier ones may have been dropped」（p7）；對 LRU：「beyond which Pensieve's eviction policy outperforms LRU」（p12） | **逐出時**（GPU→CPU→丟） | **使用者的印象正確**：Pensieve 確實依位置相關的重算成本逐出，而且逐出方向正是「前段先丟」。差距：沒有 SSD 階；分數是「重算成本 ÷ 閒置時間」，**沒有拿重算成本去跟取回成本比**（沒有交叉點）；不是寫入時決定 |
| **AsymCache／MSA**（arXiv 2606.02964，北大，2026-06）🟩 | 「a cache eviction policy that jointly optimizes hit rate and **position-aware recomputation cost**」（p1）；「block_id ← argmin_B f_B(t)·ΔT_B」（p4）；「The term (l1 + q1) corresponds to the immutable **positional index** of the cache block」（p5）；「If the recovery cost of all cache blocks is a uniform constant, our algorithm degrades to the conventional LRU strategy」（p5） | 逐出時，GPU 駐留 | 線性位置成本模型與 Tiara 的「3.546 + 0.000178×位置」同型；差距：只管 GPU 駐留（還原靠重算或從 host swap，p1），沒有 SSD、沒有寫入准入 |
| RAGCache（ACM TOCS；[arXiv 2404.12457](https://arxiv.org/abs/2404.12457)）🟩 | PGDSF：「calculates a priority based on the frequency, size of key-value tensors, last access time, and **prefix-aware recomputation cost**」（p4）；Priority = Clock + Frequency × Cost / Size（p5） | 逐出時，GPU／host／free | 文件級、RAG 限定 |
| **InferCept**（ICML'24）🟩 [arXiv 2402.01869](https://arxiv.org/abs/2402.01869) | 「we deduce three waste-calculation equations to quantify the GPU memory waste of Discard, Preserve, and Swap」（p2）；Waste_Discard = T_fwd(C)×C×M + T_fwd(C)×C_other×M（p5）；執行時「swaps as much context … as allowed to CPU memory, determines whether to preserve or discard the **remaining** context of each paused request based on the GPU memory waste amount」（p15） | **中斷發生時**（≈寫入時） | 決策時點接近 Tiara；但單位是「請求」、代價是 GPU 記憶體×時間，不看 block 位置；只有 GPU／CPU |
| Continuum（[arXiv 2511.02230](https://arxiv.org/abs/2511.02230)，頁首標 PVLDB 20(1) 2027）🟩 | 「selectively pins the KV cache in GPU memory with a time-to-live value determined by the **reload cost** and potential queueing delay induced by eviction」（p1）；批評 InferCept「makes its KV preserve decision based solely on the reload cost」（p2） | 工具呼叫時 | GPU 駐留時間，不是 SSD 准入 |
| MemServe（[arXiv 2406.17565](https://arxiv.org/abs/2406.17565)）🟩 | 式 (2)：transfer(y_p, y′_p) ≤ exec(x,y_p) − exec(x,y′_p) 則傳，「otherwise, we opt for recomputation」（p7） | 讀取時（跨實例） | — |
| **MTDS**（C&IS'26）🟩 | Partial Load：「the loader partially loads the matching prefix into GPU memory. The GPU then recomputes the remaining unmatched tokens」（p6）；「KV cache loading time exhibits a non-linear slow-then-fast growth trend, while the KV cache recomputation time grows linearly」，因此只在 (NoT1, NoT3) 區間載入（p6）；多個匹配長度時選最接近 NoT2 的（p6–7） | **讀取時** | 決定的是「載入多長的前綴」，方向是**載前段、算後段**（與 Cake 相反）；它量到的重算是對長度線性（等於每 token 常數，沒有位置效應）——在它的短上下文（1K token）情境下合理（我們的判讀） |
| **py-kvcache**（[arXiv 2609.11744](https://arxiv.org/abs/2609.11744)，VU Amsterdam／IBM，2026-09-10）🟩 | 「External KV caching should therefore be treated as a **setup specific admission decision**」（p1）；「The cache should avoid **loading or storing** prefixes below the measured break-even point」（p14）；實作：每節點×模型離線校準 break-even 長度，「If loading is predicted to cost more than recomputation, the manager declines the load」（p17） | 設計上講「不存」，實作是**讀取時**閘門 | **最接近發現 1 的框架**：硬體相依的交叉點、明講要當 admission。差距：門檻是「整段前綴長度」，不是「區塊的絕對位置」；沒有做寫入時准入；沒談寫入頻寬／耐久度 |
| LMCache 論文 🟩 | 「LMCache's KV cache loading should be adaptive: under low bandwidth, loading should be enabled only when the context length surpasses the **crossover point**」（p15）；32 Gbps 時要超過 256K 才贏 prefill（p14–15） | 讀取時 | 同上 |
| Strata／SGLang HiCache 🟩🟨 | selective-write-through：「A backup is triggered only if this counter exceeds a configurable threshold」（p9） | **寫入時** | 依存取**次數**，不依成本 |
| vLLM 🟨 | `store_threshold`（依次數）；`max_offload_tokens`（**只存前 N token**） | **寫入時** | 有位置維度的寫入准入，但方向是保前段 |
| Dynamo KVBM 🟨 | CPU→disk 只收 frequency ≥ 2，「to extend SSD lifespan」 | **寫入時** | 依頻率，理由是 SSD 壽命 |
| Tail-Optimized LRU（NeurIPS'25；[arXiv 2510.15152](https://arxiv.org/abs/2510.15152)；已在 SGLang `TLRUStrategy`）🟩🟨 | 「reallocates KV cache capacity to prioritize high-latency conversations by evicting cache entries that are unlikely to affect future turns」（p1）；SGLang 實作：只需保留 L + Q̂ − threshold 個 token，超出者「TEL-safe, i.e. free to evict」 | 逐出時 | 依位置＋TTFT 預算，但保前段、丟尾段 |
| HCache（EuroSys'25）🟩 | 還原計畫（依硬體 profiling）決定哪些層存 hidden state、哪些存 KV、哪些重算；「some layers may not even need to be stored because they can be recomputed from tokens」（p10） | **寫入格式在寫入時已決定** | 見 (c) |
| 其他「全部都存」的系統 🟩 | CachedAttention「save KV caches for all requests」（p1）；IMPRESS「all prefix KVs are stored on disks in chunks」（p6）；KVDrive「all KV entries are first persisted to SSD as the full backing」（p13）；EvicPress 新 KV「saved … to the remote disk, which is assumed to have unlimited space」（p5）；Bidaw inclusive caching（p6） | — | 這些正好是 Tiara 可以打的「寫入全收」基線 |
| 其他已查、無關的 | CacheWise（reuse-aware 逐出，工具 metadata 預測）、KVFlow（workflow-aware，CPU 當二級）、LayerKV（層級 offload 到 CPU）、DéjàVu（為容錯把 KV 串流到本地 SSD／遠端 CPU，全寫）、IMPRESS（只讀重要 token，有損）：**皆未見依位置的 SSD 寫入准入** | — | — |

**Tiara 剩下的差異**（我們的判讀）：
1. 寫入時、區塊級、**以絕對位置**對 **SSD 取回成本**比較：Pensieve 有位置但沒有 SSD 與交叉點；py-kvcache 有交叉點但是整段長度且在讀取時；KVBM／HiCache 有寫入准入但依次數。三者合起來才是 Tiara 的規則。
2. 量化「寫入頻寬省 4–7×」：這是可以寫的數字，但**只在 P\* 夠大的平台**成立（見 §5.2）。

**審稿人會怎麼打**：「這是 Pensieve 的逐出分數搬到寫入路徑，再加 py-kvcache 的 break-even。你要證明寫入時決定比 Pensieve 式的逐出時決定好在哪（寫入時資訊更少）。」→ 必須對 Pensieve 式 retention value、KVBM freq≥2、HiCache selective、vLLM `store_threshold` 做同底座比較。

### (b) 精度＋層級＋重算三者**聯合**決策

**判決：以「每個 block 同時決定 (bit, tier, recompute)」的形式，本次未查到 → 形式上新；但「聯合比分開好」這個動機已被 EvicPress／AdaptCache 主張過，而 Tiara 自己的數據（全域 INT4＋LRU 拿走 91.8%／103.9% headroom）反而削弱「必須聯合」。**

| 先行工作 | 做了什麼 | 缺哪一軸 |
|---|---|---|
| **EvicPress** 🟩 | 效用「Util(method, ratio, device) = (α·quality − TTFT)·frequency」（p6）；選項 = 壓縮法 × 壓縮率 × 下層裝置（p7）；新 KV 一律存到「assumed to have unlimited space」的 remote disk（p5）；粒度「context-level, instead of chunk-level … or token-level」（p6）；Prefill 只是 baseline（p8） | **沒有丟棄＋重算這個動作**；也明確拒絕 chunk／token 粒度（正好與 Tiara 的位置級相反，可引用來劃界） |
| **AdaptCache** 🟩 | 「For each storage tier, we will greedily decide the compression choice (to evict, store in full quality or compress each entry)」（p2）；未來頻率＝歷史頻率（p2） | 文中「evict」是移到下層還是刪除**沒說明**（未查證）；重算只當靜態 baseline（與 SOTA_MATRIX 一致） |
| **CacheGen**（SIGCOMM'24）🟩 [arXiv 2310.07240](https://arxiv.org/abs/2310.07240) | 「When the bandwidth is too low, CacheGen can also fall back to sending a chunk in **text format** and leave it to the LLM to recompute the KV cache of the chunk」（p2）；每 chunk 挑「the least compression loss (i.e., text format or lowest encoding level) with an expected delay still within the SLO」（p7） | 精度＋重算在**每 chunk** 聯合，但是網路串流、讀取時；沒有多階放置 |
| QEvict 🟩 | 「QEvict jointly determines residency and precision through full-precision, quantized, and evicted tiers」（p10） | 只在 GPU 內，丟了就沒了（無重算） |
| RDKV（[arXiv 2605.08317](https://arxiv.org/abs/2605.08317)）🟨 | 「assigns each token or channel a bit-width ranging from full precision down to zero bits」（摘要，WebFetch） | GPU 內，0 bit＝丟，不重算 |
| LeoAM、KVDrive 🟩（PAPER_GUIDE 已整理） | 多階＋INT4（LeoAM）；多階＋量化列為 future work（KVDrive） | 無重算動作 |
| KV Survey（ACL-F'26）附錄 G.2 🟩（PAPER_GUIDE 已整理） | 提出多階中間狀態的概念 | 無實作 |

**與 remote 實測的關係**：EXPERIMENTS_20260919 #1 在 {BF16, INT4} 下顯示全域 INT4＋原廠 LRU 就達到 oracle headroom 的 91.8%（toolagent）／103.9%（conversation），切到 INT4 後剩 2.02%／3.52%。PAPERS_EXPLAINED_20260920 判定「量化與放置是替代品」這個**新的**發現在窄化後存活（沒人量過 headroom 隨 bit-width 的變化）。所以 (b) 的「聯合決策」在純延遲目標下不再是賣點；只在「品質預算 q」的中間區間可能有價值，而 ε 在平台 B 目前量不出來（n=120、CI ±10pp）。

### (c) 以 Cake 式雙向重疊為前提，推出「前段永遠不必寫入」

**判決：這個原則在 layer 維度已被 HCache 發表；token 維度的「前段丟、後段留」佈局已被 Pensieve 以逐出方式實現；最佳切點公式已被 CacheFlow 寫出。「token 維度＋寫入時＋append-only 讓會合點單調後移，所以前段永遠不必寫入」這個完整論證，本次未查到。→ 部分已知（窄新，機制層級）。**

**先回答協調者的問題：「目前是否有任何系統主張『因為前段在雙向還原時會被重算，所以前段不必寫入／應優先逐出前段』？」**

| 問法 | 有沒有人 | 證據 |
|---|---|---|
| 「前段應**優先逐出**」 | **有，但理由不是雙向還原** | Pensieve：因為前段重算便宜而優先逐出並丟棄（p6–7）；AsymCache：成本項隨位置線性，所以「favors the retention of blocks located at later positions」（p5） |
| 「因為**還原時會被重算**，所以**不必存**」 | **有，但在 layer 維度** | HCache：「Combined with the zero-bubble scheduler, some layers may not even need to be stored because they can be recomputed from tokens」（p10）；Table 3：30B 模型的排程是「40 H + 8 RE」（8 層重算、不存）（p10）。HCache 還討論過 token-wise 切分（「the first two tokens are managed via HCache and the rest … KV offload」），最後因 GEMM 效率選 layer-wise（p6–7） |
| 「部分狀態改存便宜表示／重算，**以減少 SSD 寫入**」 | **有** | HILOS（ASPLOS'26）：「The X-cache mechanism, with a cache rate of α%, lowers storage writes by approximately α/2 %」，並做了以 PBW 計的耐久度分析（p11）；對應的重算與 NSP 上的 attention 同時執行（p5） |
| 「雙向還原的最佳切點」 | **有** | CacheFlow（arXiv 2604.25080）：「For a split at position ℓ, where segments [0, ℓ) are recomputed and [ℓ, L) are loaded … ℓ = L·T_io/(T_comp+T_io)」（p5）——這就是 Tiara 的 1/(1+κ)。但 CacheFlow 全文未見「因此前段不必存」的推論（我 grep storage／store／persist 無相關句） |
| 「因為雙向還原，**token 維度的前段在寫入時就不必寫進 SSD**」 | **未查到** | Cake 本身明寫評測「precompute and store all requests' KV cache in advance」（p5）；讀取靠 prefix hash 找最後一個存著的 chunk、GPU 從 0 往後、I/O 從尾往前（協調者查證的附錄 A 事實） |
| 反方向的工業做法 | **有，而且是主流** | vLLM `max_offload_tokens` 只存前 N token；vLLM 逐出時尾巴先走；SGLang LRU「recursively evicts leaf nodes」（協調者查證，arXiv 2312.07104）；TLRU 丟尾段 |

**這個轉向必須正面回答的五個反論**（我們的判讀，每一條都有文獻支撐）：
1. **會合點是讀取時的動態量，不是寫入時的常數。** Cake 原文：「whenever I/O or compute availability changes, the merging point of the two processes shifts accordingly」（p5），而且 Cake 的 adaptive 模式把 prefix-cache 的 prefill 排在 decode 與非 prefix 請求之後（p5）。GPU 忙的時候會合點會往**前**移，這時被 Tiara 省掉的前段就必須全部重算。「永遠不必寫」只在「讀取時的算力不少於寫入時假設」才成立。
2. **前段往往是最常被共享的部分。** vLLM 文件把「尾巴較少被別人重用」當成尾巴先逐出的理由；系統提示、工具說明都在前段。對多租戶的共享前綴，每個請求都重算前段，總算力成本會被放大（我們的判讀）。
3. **前綴命中語意。** 見 §2.4：不存前段後，後段在 vLLM／SGLang 的原生查找路徑上不會命中，必須實作 Cake 式載入器（Cake 沒有公開程式碼）。
4. **重疊模型與 P\* 模型給的答案不同。** P\* 是「逐塊、不重疊」的交叉點；在雙向重疊下，即使 L < P\*，I/O 仍會分走尾端一段，所以尾端**一定要存**；而 L 很大時重疊切點 ℓ\* 可能大於或小於 P\*。論文要講清楚用的是哪一個，兩者的寫入節省不同（我們的判讀）。
5. **平台相依。** remote RUNLOG_MI300X：P\*（SSD vs DROP，chunk=2,048）只有 Llama-3.1-8B 為正（3,396 token），Qwen2.5-7B-1M、Qwen3-30B-A3B、Seed-OSS-36B 都 < 0（SSD 一直較便宜），且作者自註外推合理性未驗證。在這種平台上「前段不寫」幾乎不省任何東西。

**Tiara 剩下的差異**：token 維度（HCache 做的是 layer 維度）、寫入時（Cake／CacheFlow 是讀取時）、append-only 多輪下會合點單調的論證（未見）、把 SSD 當有寫入預算的快取並量化寫入量（見 (d)，已有人做一般性的版本）。

### (d) SSD 的**寫入量／寫入頻寬／耐久度**當設計限制

**判決：已被做過（包含頂會論文與工業系統），而且有人已寫出寫入准入的成本條件。Tiara 能補的只剩「每塊的節省量 S 隨位置變化」這一項。**

| 先行工作 | 原文 | 類型 |
|---|---|---|
| **HILOS**（ASPLOS'26，SNU；[arXiv 2502.09921](https://arxiv.org/abs/2502.09921)）🟩 | 「Storage write endurance is often a cost concern … endurance is mainly limited by the total write volume … The X-cache mechanism … lowers storage writes by approximately α/2 % … delayed KV cache writeback defers writes … reducing write amplification」；以「each 3.84 TB SSD supports 7.008 PBW」估可服務請求數，「HILOS improves endurance by 1.34× to 1.47×」（p11） | 頂會、量了耐久度 |
| **HBF 特性化**（[arXiv 2608.11668](https://arxiv.org/abs/2608.11668)，北大，2026-08）🟩 | 「Writes outnumber reads on every trace … a TLC tier wears out sooner than the SSD pool it replaced」；結論要「reuse-aware placement, **write budgeting**, and thermal coordination」（p1）；寫入准入條件「Writing is net-positive only when ρS > AC, i.e. ρ > ρ★ = AC/S」（ρ：每寫入位元組換回的有用讀取位元組；S：每讀取位元組省下的時間；A：寫入放大；C：每位元組容量與寫入預算成本）（p6）；SSD 參考值「vendor 3-DWPD rating gives 38.4 TB/day」（p6） | preprint、Qwen-Bailian 真實 trace、Mooncake 式 KV pooling |
| HBF-GR（[arXiv 2609.07175](https://arxiv.org/abs/2609.07175)，Huawei，2026-09）🟩 | 「conventional LRU KV cache management tightly couples KV cache writes with cache misses, generating excessive write traffic that rapidly exhausts flash endurance … LRU-K extends HBF lifetime from about one year under conventional LRU to over six years with a moderate K = 10」（p1） | 生成式推薦，不是 LLM prefix cache，但「寫入准入延長快閃壽命」同構 |
| **Dynamo KVBM** 🟨 | 「Disk offload filtering is enabled by default to extend SSD lifespan … frequency ≥ 2」 | 工業、預設開啟 |
| Strata 🟩 | 寫入策略是「write bandwidth, cache capacity, durability, and future reuse」之間的取捨（p9） | 頂會 |
| Tutti 🟩 | 讀寫同時進行會讓 PCIe 頻寬崩潰，所以「Write requests are handled only after the critical-path reads have been scheduled」（p7） | 寫入頻寬干擾讀取 |
| Kareto（[arXiv 2603.08739](https://arxiv.org/abs/2603.08739)，浙大／阿里）🟩 | 「writes (from DRAM eviction) and reads (for KV reloading) compete for the same I/O channel」（p5） | — |
| HiFC（S2 標 NeurIPS'25）🟥 | 搜尋摘要提到 pSLC 與耐久度提升；OpenReview 需驗證，**未讀到原文，不據此下結論** | — |

**Tiara 剩下的差異**（我們的判讀）：HBF 特性化論文的 ρ★ 模型把 S 當成裝置層級的常數；Tiara 的位置成本模型可以把它寫成 **S(p) = max(0, 重算(p) − 取回)**，p < P\* 時 S=0，寫入就一定不划算。這是一個乾淨的延伸，但它是「把兩篇的模型接起來」，novelty 偏窄。另外，Tiara 在 RTX 3090＋SATA SSD（181 MiB/s 持續寫入）上的量測有實務價值，因為上述論文都在資料中心級硬體上。

### (e) 學習式 KV 預測器＋退回 LRU（robust fallback）

**判決：已被做過。**

**協調者點名的 arXiv 2509.20979 查證結果**（🟩，v2 全文）：
- **題目**：Toward Robust and Efficient ML-Based GPU Caching for Modern Inference（浙大、南洋理工、南航、快手；v1 2025-09-25、v2 2026-04-24）。
- **快取的是什麼**：**是 KV prefix cache**。「We integrate LCR into SGLang by replacing RadixTree's default LRU eviction policy with LARU」（p6）；「LARU also operates at the tree node level」（p7）；逐出候選限於 leaf node 以符合 SGLang 的 prefix 語意（p6–7）。另外也做 DLRM embedding（HugeCTR）。
- **演算法**：learning-augmented LRU。以 LRU phase 偵測誤判：「If the requested item was previously evicted due to a prediction-driven eviction within the current LRU phase, then the predictions in this phase were erroneous」（Lemma 1，p4）；偵測到就做一次 LRU 逐出並把信心 λ 減半，候選集縮成 LRU 佇列前 ⌊λk⌋ 個（p4）；「Theorem 1. LARU is 1-consistent and O(k)-robust」（p4）。預測器是 CPU 上的 LightGBM，線上訓練（p6）。
- **baseline**：SGLang 原生 LRU、FPB（Follow Prediction Blindly）、HF（仿 HALP，LRU 先挑 4 個候選再給 ML 排）（p2、p7）。
- **負載與模型**：Qwen-Bailian（阿里公開 trace A，>40,000 個多輪請求）、Online-QA（公司 A 內部，7,268 請求）、Aibrix-Synthetic；Qwen2.5-32B、DeepSeek-R1-671B（p7）。
- **結果**：P99 TTFT 在 Aibrix-Synthetic 降 6.0–10.7%，在 Online-QA 降 13.5–28.3%（p7）；「Even with poor predictions, performance degrades gracefully and remains close to LRU」（p1）。
- **它明說的問題定義**：「Most simply adopt FPB (Follow Prediction Blindly), which evicts based on predicted priorities and can fail severely when predictions are wrong」（p2）——**這正是 Tiara 發現 3 觀察到的現象（比 LRU 差 2.7–5.6×）的名稱與解法**。

**對 Tiara「會認輸的預測器」的衝擊**：直接命中。Tiara 的學習式策略就是 LCR 所說的 FPB；「沒把握時退回 LRU」就是 LCR 的 HF／LARU。在單一層級、單位成本（命中／未命中）的設定下，LCR 已經給出有理論保證的版本，而且就在 SGLang 的 KV prefix cache 上實作。審稿人會要求 Tiara 直接比 LARU。

**其他先例**：
- 理論：Lykouris & Vassilvitskii，Competitive caching with machine learned advice（ICML'18；S2 549 次引用）；Wei，Better and Simpler Learning-Augmented Online Caching（APPROX'20，BlindOracle&LRU）；Bansal et al.，**Learning-Augmented Weighted Paging**（SODA'22，[arXiv 2011.09076](https://arxiv.org/abs/2011.09076)）——**異質成本（weighted）的預測式分頁也已有理論**。
- 系統：HALP（NSDI'23，YouTube CDN，用 LRU 預篩候選再由學習模型排序，見 LCR p2 的描述）；Marconi 🟩：「setting α to 0 falls back to LRU … On startup, Marconi sets α to 0 … continuing to use LRU while bookkeeping … Marconi adopts the α value that maximizes the hit rate」（p7）；LPC（NeurIPS'25）🟨：「These insights, combined with last access timestamps, inform more effective cache management」（[NeurIPS 頁](https://neurips.cc/virtual/2025/poster/117662)），是否有 fallback：未查證。
- 反例（沒有 fallback）：CacheWise 🟩「we leave a thorough study of drift robustness to future work」（p8）；KVP、ForesightKV、LookaheadKV（PAPER_GUIDE 已整理）。
- 旁證：Where Should the KV Cache Live?（[arXiv 2609.16215](https://arxiv.org/abs/2609.16215)，Vizuara，模擬器、合成負載）🟩 摘要：「the shipped predicted-reuse policy is byte-identical to recency … a genuine EWMA predictor … still finishes second to reuse-frequency on the very workloads prediction was meant to win」（p1）。依 PAPERS_EXPLAINED_20260920 的紅線，只能當「領域現象」引用，不能當已證實結論。

**Tiara 剩下的差異**（我們的判讀）：(1) 多階＋含重算的異質動作成本下的 robust 保證（weighted paging 有理論、沒有這種系統）；(2) 用「分數分散度」而非「偵測到誤判」來觸發退回；(3)「AUC 0.9995 但逐出順序沒學到」這個評估指標錯配的觀察。三者都偏窄。

### (f) 用重算 FLOPs／重算成本做成本感知逐出

**判決：已被做過（而且有 30 年的古典根源）。**

| 先行工作 | 成本項 |
|---|---|
| Marconi（MLSys'25）🟩 | 「flop efficiency = Total FLOPs across layers / Memory consumption of all states」；S(n) = recency(n) + α·flop_efficiency(n)（p6–7）。它自己說「Traditional prefix caching systems designed for Transformers don't need to consider FLOP efficiency because KVs' FLOP efficiency is near-constant」（p6）——這句話與 Tiara／Pensieve／AsymCache 的位置線性成本**相衝突**，可以拿來凸顯長上下文時假設失效（我們的判讀） |
| Pensieve 🟩 | V = Cost(s,l)/T（p6） |
| AsymCache 🟩 | f_B(t)·ΔT_B，ΔT_B 隨位置線性（p4–5） |
| RAGCache 🟩 | Frequency × Cost / Size（p5） |
| InferCept 🟩／Continuum 🟩 | waste 模型／TTL 依 reload cost |
| GDSF（Cherkasova 1998）、GreedyDual-Size | 古典成本感知替換；Marconi 自己寫「Cost-aware cache eviction for objects with variable sizes is a well-studied problem (e.g., GDSF (Cherkasova, 1998))」（p7） |

**Tiara 剩下的差異**：把成本項擴成「動作相依」（CPU／SSD／DROP 各自的成本），而不是單一重算成本。這是 (a)(b) 的一部分，不是獨立貢獻。

### (g) 「κ 或類似比值隨硬體變動，所以放置策略要依硬體調整」

**判決：已被做過；而且 Tiara 自己的新數據削弱了「硬體」這個自變數。**

| 先行工作 | 原文 |
|---|---|
| **Bottlenecks**（MLSys'26）🟩 | κ_crit = F_pf/B_kv × BW_PCIe/C_eff = κ_M × κ_HW（p4）；「given a model and hardware platform, the architect should determine the system's κcrit and compare against the workload's κratio」（p5）；「B200 with PCIe 5.0 achieves κHW = 13.5, only 40% of H100's 34」（p6） |
| py-kvcache 🟩 | 「external KV caching has **hardware and workload dependent break-even points**, motivating cache admission and bypass policies」（p2）；break-even「frontier shifts between GPUs, models, SSD」（p11） |
| LMCache 🟩 | 交叉點隨頻寬移動（p14–15） |
| HCache 🟩 | 「generate a bubble-free restoration scheme according to the hardware setups … we profile the transmission and computation speed of a specific hardware platform offline」（p7） |
| KVPR 🟩 | LP 求切點，「The optimal split point l depends on the current sequence length s′」（p5）；依 GPU 速度與傳輸速度 |
| Cake 🟩 | 不同 GPU 與儲存的等效頻寬比較（Fig. 3，p4）；會合點隨資源移動（p5） |
| MatKV（ICDE'26；[arXiv 2512.22195](https://arxiv.org/abs/2512.22195)）🟩 | 以 Gray 的 five-minute rule 推出「ten-day rule」：文件至少每 10 天被取用一次，存 SSD 才比 GPU 重算划算（p2–4） |
| CacheTune（[arXiv 2605.24022](https://arxiv.org/abs/2605.24022)）🟩 | 「hardware-aware adaptive recomputation-ratio tuning to balance computation and data movement across heterogeneous cache pools」（摘要） |
| MTDS 🟩 | 邊緣裝置 vs 雲端的差異是其動機（p9） |

**Tiara 自己的新數據**（remote EXPERIMENTS #2、CLAIM_EVIDENCE B1）：同一張 MI300X 上跨模型傳輸差 17.1×，主因是 vLLM 在 ROCm 上的每筆描述符固定成本（12.9–16.3 µs）；實測跨平台 κ_cpu 只差 5.8–6.2×，不是 32×；κ 對 ctx 與模型的敏感度大於對硬體。→ 「κ 跨硬體變動」不能當貢獻；PAPERS_EXPLAINED_20260920 也判定 Bottlenecks 已拿走 achieved-vs-peak 的一半。

---

## 4. 怎麼判斷 SOTA

### 4.1 回答使用者的問題：「總不可能就是看比較多篇，或選 ICML 那種就算？還是跟我做的類似也算？」

三個直覺都只對一部分：

- **「看比較多篇」**：被當 baseline 的次數是好指標，但它有兩個偏差。第一，地板會贏（vLLM 9 次、LRU 7 次，但它們不是研究 SOTA）。第二，新論文會輸（Strata、Bidaw、EvicPress 在 35 篇內都是 0 次，只因為太新）。所以要看「**同一問題設定下、最近 12–18 個月、同儕審查過的**論文拿誰比」。
- **「選 ICML 那種」**：venue 只能當過濾器，不能當判準。Cake（ICML'25）的整條 I/O 路徑是模擬的（p5）；AdaptCache 是 3 頁 workshop；而 Pensieve、HCache 在 EuroSys。系統題目的「強」要看評測是否在真實機制上，不是看 ML 或 systems venue。
- **「跟我做的類似也算」**：只有在**問題設定等價**時才算。判斷方法是把問題寫成一個五元組，至少前兩項要相同：
  1. 決策變數（例：寫不寫 SSD／這段要算還是載／幾個 bit）；
  2. 決策時點與可用資訊（寫入時？讀取時？知道未來嗎？）；
  3. 目標與約束（TTFT？寫入預算？品質 ε？）；
  4. 負載型態（多輪、共享前綴、單請求長文）；
  5. 硬體型態（消費卡＋SATA？資料中心＋NVMe？）。
  例：Cake 的五元組是（切點；讀取時、KV 已全部存好；TTFT；單請求；GPU＋儲存頻寬）。它是「重算 vs 載回」格的 SOTA，但**不是**「寫不寫 SSD」這一格的 SOTA，因為第 2 項不同。

### 4.2 一套可以照做的判準（依重要性）

1. **問題設定等價**（上面的五元組，前兩項必須相同）。
2. **被後續、同設定、同儕審查的論文拿來當 baseline**，而且**沒被更新的同設定論文打敗**（若被打敗，SOTA 換成打敗它的那篇）。實作：用 Semantic Scholar `/paper/{id}/citations?fields=contexts,intents,isInfluential` 抓引用語境，篩出含「compare／baseline」的句子（本次已跑，見 `research/s2/s2_cites.json`；這是下限，因為語境抽取不完整）。
3. **可重現與整合**：有開源程式碼、已併入 vLLM／SGLang／Dynamo／llm-d 的，審稿人會預期你比；閉源的要「在同一底座重做」（Bidaw、IMPRESS、HCache 都這樣做過）。
4. **頂會同儕審查**：當過濾器與加權，不單獨成立。
5. **引用數**：只當平手時的參考。Semantic Scholar 的 influential citation 是模型判定「實質使用或延伸」的引用，比總數好，但仍反映注意力而非效能。

**審稿人實際會要求的 baseline 組合**（依本題）：
- (i) 同格最強的**開源生產系統**：SGLang HiCache（`write_through`／`write_through_selective`）、vLLM `OffloadingConnector`（`store_threshold`、`max_offload_tokens`）、Dynamo KVBM（freq≥2 磁碟過濾）、LMCache；
- (ii) 同格最強的**已發表演算法**（閉源就重做）：Pensieve 的 retention value、Cake 的雙向載入、py-kvcache 的 break-even gate、LARU（學習式那一格）；
- (iii) **簡單但抓到關鍵的 heuristic**：全域 INT4、freq≥2 過濾（remote EXPERIMENTS 已示範「一個簡單 heuristic 就關掉 gap」）；
- (iv) **oracle** 上界。

### 4.3 Semantic Scholar 數據（2026-09-24 查詢）

查詢方式：`POST https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,venue,year,citationCount,influentialCitationCount,externalIds`（批次 1 於 **2026-09-24T02:27:44Z**；批次 2 於 02:52:24Z；批次 3 於 03:01:54Z）；沒有 arXiv ID 的用 `/paper/search/match`（02:28–02:54Z）。原始 JSON：`research/s2/s2_batch.json`、`s2_batch2.json`、`s2_batch3.json`、`s2_match.json`、`s2_match2.json`。

**教授五篇**

| 論文 | S2 識別碼 | S2 venue | citationCount | influentialCitationCount |
|---|---|---|---|---|
| Cake | arXiv:2410.03065 | ICML | 38 | 6 |
| AdaptCache | arXiv:2509.00105 | arXiv.org | 5 | 0 |
| Strata | arXiv:2508.18572（search/match 只回傳 arXiv 條目，**OSDI'26 版未見獨立條目**） | arXiv.org | 32 | 4 |
| Bidaw | CorpusId:286257770（無 arXiv） | FAST | 2 | 0 |
| MTDS | DOI:10.1007/s40747-025-02200-4 | Complex & Intelligent Systems | 1 | 0 |

**指定的另外五篇**

| 論文 | S2 識別碼 | S2 venue | citationCount | influentialCitationCount |
|---|---|---|---|---|
| Pensieve | arXiv:2312.05516 | EuroSys | 88 | 8 |
| CachedAttention | arXiv:2403.19708 | USENIX ATC | 252 | 21 |
| LMCache | arXiv:2510.09665 | arXiv.org | 147 | 25 |
| Mooncake | arXiv:2407.00079 | ACM Trans. on Storage（DOI 10.1145/3773772） | 274 | 25 |
| CacheBlend | arXiv:2405.16444 | ACM TOCS（DOI 10.1145/3790254） | 268 | 40 |

**本文用到的其他論文（供參考）**

| 論文 | citationCount / influential | 論文 | citationCount / influential |
|---|---|---|---|
| InferCept（ICML'24） | 41 / 8 | HCache（EuroSys'25） | 51 / 5 |
| IMPRESS（FAST'25） | 36 / 3 | FlashGen（ASPLOS'25） | 30 / 2 |
| Marconi（MLSys'25） | 37 / 1 | KVPR（ACL-F'25） | 16 / 0 |
| CacheGen（SIGCOMM'24） | 320 / 16 | RAGCache（TOCS） | 154 / 19 |
| LCR／LARU（2509.20979） | 5 / 0 | AsymCache（2606.02964） | 1 / 0 |
| py-kvcache（2609.11744） | 0 / 0 | CacheFlow（2604.25080） | 2 / 0 |
| HILOS（ASPLOS'26） | 5 / 2 | HBF 特性化（2608.11668） | 3 / 0 |
| Tail-Optimized LRU（NeurIPS'25） | 6 / 0 | LPC（NeurIPS'25） | 6 / 1 |
| HALP（NSDI'23） | 35 / 5 | LRB（NSDI'20） | 190 / 30 |
| Lykouris & Vassilvitskii（ICML'18） | 549 / 55 | Learning-Aug. Weighted Paging（SODA'22） | 37 / 0 |
| Bottlenecks（2601.19910） | 1 / 0 | EvicPress（2512.14946） | 7 / 0 |
| KVDrive | 5 / 2 | Tutti | 5 / 0 |
| vLLM（SOSP'23） | 8,458 / 1,165 | SGLang（NeurIPS'24） | 1,445 / 168 |

**怎麼讀這些數字**（我們的判讀）：教授五篇的引用數都很低（Bidaw 2、MTDS 1、AdaptCache 5），這不代表它們不重要，而是太新。所以對它們只能用判準 1–3，不能用 5。相對地，CachedAttention（252/21）與 CacheBlend（268/40）已是被廣泛使用的節點；Pensieve（88/8）的引用數遠高於教授五篇，但在 35 篇內沒被當 baseline——它是本題**最容易被忽略、卻最接近 Tiara** 的一篇。

---

## 5. 結論（審稿人口吻）

### 5.1 新穎性等級

> **給作者的總評**：本稿包含三個觀察與一個轉向提案。查過 2024–2026 的文獻後，每一個組成部分都有很接近的先行工作；剩下的是一個特定的組合，而它的價值取決於平台。

**發現 1（P\*、寫入時依位置准入、寫入頻寬省 4–7×）— 部分已知（窄新）。**
位置線性的重算成本是 Cake 的核心 insight（「Compute cost increases for later tokens, while I/O cost remains constant regardless of token position」，p4），Pensieve（p4、p6）與 AsymCache（p5）也用了；依此把**前段先丟**是 Pensieve 的逐出策略（p6–7）；以硬體相依的 break-even 做 admission 是 py-kvcache（p1、p14、p17）與 LMCache（p15）的結論；寫入時的過濾已存在於 SGLang HiCache、vLLM、Dynamo KVBM（後者明寫為了 SSD 壽命）。本次沒找到的是「以**區塊絕對位置**對 **SSD 取回成本**的交叉點，套在 **SSD 寫入路徑**上」這個組合。這是一個 incremental but clean 的系統貢獻，前提是作者能展示它比 Pensieve 式逐出＋KVBM 式過濾更好，而且說清楚它在哪些平台有用。

**發現 2（三個決策互相打架）— 部分已知，而且被作者自己的新數據轉向。**
「壓縮與放置必須聯合」是 EvicPress 與 AdaptCache 的動機；CacheGen 已在每個 chunk 聯合決定精度與送文字重算（p2、p7）；QEvict 聯合決定精度與駐留（p10）。「78 萬次降級、+0.004%」只說明 Tiara 模擬器裡的 LRU 排序讓精度階留不住，審稿人會把它讀成設計缺陷。更要緊的是 remote EXPERIMENTS #1：全域 INT4＋原廠 LRU 拿到 91.8%／103.9% 的 oracle headroom。**在純延遲目標下，數據支持的是「量化與放置是替代品」，不是「三格必須一起決定」。** 如果作者要保留聯合決策，必須先量準 ε 並把論點限縮到品質預算的中間區間。

**發現 3（學習式預測器要會認輸）— 已被做過。**
arXiv 2509.20979（LCR/LARU）在 SGLang 的 KV prefix cache 上實作了 learning-augmented LRU，有 1-consistency 與 O(k)-robustness，並明確把「Follow Prediction Blindly」當成要修正的失敗模式；理論上有 ICML'18、APPROX'20、SODA'22（含異質成本）；系統上有 HALP 與 Marconi（α=0 即退回 LRU）。remote 的 MI300X 數據（AUC 0.917–0.922 仍輸 tier_fs 9.7–47.3%）顯示 Tiara 的失敗主因是可操作空間小，不是預測不確定，所以單純加上 LARU 式 fallback 也未必能轉負為正。這個發現只能當「我們採用 LARU 式保護」的工程選擇，不能當貢獻。

**轉向（Cake 式雙向重疊 → 前段永遠不必寫入）— 部分已知（窄新，機制層級）。**
「在重疊還原中會被重算的部分就不必存」這個原則，HCache 已在 layer 維度寫出並實作（p10、Table 3）；最佳切點公式 ℓ = L·T_io/(T_comp+T_io) 已見於 CacheFlow（p5）；token 維度「前段丟、只留文字、要用時重算」的佈局是 Pensieve 的逐出結果（p7）；以減少寫入延長 SSD 壽命已見於 HILOS（ASPLOS'26）與 KVBM。本次沒找到的是「token 維度＋寫入時＋append-only 使會合點單調」的完整論證。這個論證是有意思的，但它成立需要（1）讀取時算力不少於假設（Cake 自己說會合點會隨資源移動）；（2）前段不被其他請求共享；（3）改掉 vLLM／SGLang 只認連續前綴命中的語意；（4）平台上 P\* 夠大（MI300X 上 3/4 模型 P\* < 0）。作者必須把這四個條件寫成適用範圍，並在真機上實作 Cake 式載入器，否則審稿人會把它當成紙上推導。

### 5.2 新實測（remote）對評級的影響

| 發現 | 實測前的直覺評級 | 考慮 remote 實測後 | 關鍵數據 |
|---|---|---|---|
| 發現 1 | 部分已知 | **部分已知，且適用範圍明顯縮小** | MI300X P\* 只有 Llama-8B 為正（3,396 token）、其餘 3 模型 < 0；κ_ssd 在 16K 時兩平台都 < 1；傳輸項受 vLLM ROCm 描述符粒度主導（同卡跨模型 17.1×）→ P\* 是「模型 × ctx × 硬體 × 實作路徑」的函數 |
| 發現 2 | 部分已知 | **方向反轉**：支持的是「替代品」不是「必須聯合」 | 全域 INT4＋LRU = 91.8%／103.9% oracle headroom；INT4 後剩 2.02%／3.52% |
| 發現 3 | 已被做過（LCR） | **已被做過，且失敗機制不同於論文敘述** | AUC 0.92 仍輸 9.7–47.3%；最佳 baseline 在 27% 抽樣中翻轉 |
| 轉向 | 部分已知 | **部分已知，且需要 P\* 夠大的平台** | 同發現 1；另外 Tiara 策略從未在真機端到端跑過（ADVISOR_REPLY §0.3） |

（我們的判讀）綜合來看，最能站得住的組合是：**把 SSD 當有寫入預算的快取（HBF 特性化的 ρ★ 條件）＋ Tiara 的位置相依節省 S(p) ＋ 一個「事前判斷這個平台值不值得做」的判準（請求長度 ÷ P\*）**，並把發現 3 降為工程細節、把發現 2 改寫成替代品。

### 5.3 必須正面引用並劃清界線的 5 篇

| # | 論文 | 為什麼必引 | 劃界句（建議） |
|---|---|---|---|
| 1 | **Pensieve**（EuroSys'25） | 位置相依的重算成本逐出、前段先丟，最接近發現 1 與轉向 | 「Pensieve 在逐出時以重算成本÷閒置時間排序並丟棄前段；我們在**寫入 SSD 時**以重算成本與 **SSD 取回成本**的交叉點決定是否寫入，並處理 Pensieve 沒有的 SSD 階與寫入預算。」 |
| 2 | **HCache**（EuroSys'25） | 「會被重算的部分不必存」的先例（layer 維度） | 「HCache 在 layer 維度依硬體決定哪些層不存；我們在 token 維度依位置決定，並論證 append-only 對話下的單調性。」 |
| 3 | **Cake**（ICML'25）＋ **CacheFlow**（arXiv 2604.25080） | 雙向還原與最佳切點是轉向的前提；CacheFlow 已把 Cake 當 SOTA 比 | 「Cake／CacheFlow 在讀取時決定切點、假設 KV 已全部存好；我們把切點的單調性用在寫入端。」同時承認會合點的動態性 |
| 4 | **py-kvcache**（arXiv 2609.11744） | 硬體相依的 break-even、「avoid loading or storing prefixes below the measured break-even point」 | 「py-kvcache 以整段前綴長度在讀取時閘門；我們以區塊絕對位置在寫入時准入。」 |
| 5 | **LCR/LARU**（arXiv 2509.20979） | 學習式 KV prefix 逐出＋有保證的 LRU 退回 | 「我們採用 LARU 式保護；我們的貢獻不在 fallback 本身。」 |

另外必須**當 baseline 跑**（不只是引用）：Dynamo KVBM 磁碟過濾（freq≥2）、SGLang HiCache `write_through_selective`、vLLM `store_threshold`；寫入耐久度部分引 HILOS（ASPLOS'26）與 HBF 特性化（arXiv 2608.11668）。

### 5.4 最低限度要補的東西（依轉向）

1. 同一底座（vLLM）重做：Pensieve retention value、KVBM freq≥2、HiCache selective、vLLM `store_threshold`、py-kvcache gate；量**寫入位元組、TTFT、各階命中率**，並固定比某一個 baseline（remote EXPERIMENTS 已示範「最佳 baseline」會翻轉）。
2. 實作 Cake 式「前段重算＋後段載入」的載入器，否則「不存前段」在真機上等於浪費後段。
3. 掃讀取時的算力爭用（GPU 忙碌度）與前綴共享比例，畫出「永遠不必寫」的成立區間。
4. 兩個平台都要報 P\*，並把「請求長度 ÷ P\*」當成事前判準。

---

## 附錄 A：與既有文件的衝突與更正

| # | 既有文件的說法 | 本次證據 | 誰較強 | 建議 |
|---|---|---|---|---|
| 1 | SOTA_MATRIX §7：「MTDS … 零條 claim 通過驗證。沒有抽到任何一手內文」 | 使用者另外提供的 Springer PDF 已抽成 `papers_txt/mtds2026.txt`；本文引用的 MTDS 內容（p5–7 三種載入策略與 NoT1–NoT3 區間、p10 baseline）皆逐字取自該全文 | 本次（一手全文） | MTDS 那一列可以補，但要標明來源是使用者提供的 PDF |
| 2 | PAPERS_EXPLAINED_20260920 把 HCache、KVPR 評為 🟢 低威脅（證據等級 B：只讀摘要） | 全文顯示 HCache 在 layer 維度實作了「會被重算的就不存」（p10、Table 3），KVPR 的切點是「前段由 activation 重算、後段傳 KV」（p4–5） | 本次（全文）；但注意該文件的威脅對象是 (A)(B) 兩個發現，不是寫入端准入 | 對轉向 (c) 而言，HCache 應上調為**高**威脅、KVPR 為中 |
| 3 | RESEARCH_STORY §3：「Strata … 不把重算當成選項」；PAPER_GUIDE：「放哪、丟誰用的都是 LRU」 | SOTA_MATRIX §6 已推翻「Strata 沒有 compute-vs-load policy」與「Strata 完全沒有 eviction/admission/placement policy，只有 LRU」 | SOTA_MATRIX（3 票） | 只引原文：「recomputation becomes increasingly costly as context length grows, making it an unattractive alternative」（p5）、「For all memory layers, the LRU algorithm serves as the default eviction policy」（p9） |
| 4 | Strata 論文：預設 selective-write-through、門檻 2（p9） | SGLang main CLI 預設 `write_through`（2026-09-24 原始碼） | 兩者都對，版本不同 | 寫明引用的是論文版還是 upstream 版 |
| 5 | RESEARCH_STORY：「MTDS 整段二選一」 | MTDS 在多個匹配長度中選最接近 NoT2 的一個來載入，其餘重算（p6–7）——是「選前綴長度」，方向是載前段、算後段 | 本次（全文） | 改寫成「MTDS 在讀取時選擇要載入的前綴長度，不依區塊位置的重算成本切分」 |
| 6 | RESEARCH_STORY §4 表：「Cake 讀的時候、依位置」 | 一致；另補 Cake 原文「the merging point … shifts accordingly」（p5）作為轉向的主要反論 | — | — |
| 7 | 使用者印象：「Pensieve 的逐出會考慮重算成本且與位置有關」 | **正確**（p2、p6、p7） | — | 必須寫進 related work 的第一段 |

## 附錄 B：本次讀過的一手來源（除 35 篇本地檔以外）

全文 PDF 抽文字存於 `scratchpad/extra_txt/<key>.txt`（原 PDF 在 `scratchpad/extra_pdf/`）：

| key | 論文 | URL |
|---|---|---|
| pensieve2025 | Stateful LLM Serving with Pensieve（EuroSys'25） | https://arxiv.org/abs/2312.05516 |
| msa2026 | Multi-Segment Attention／AsymCache | https://arxiv.org/abs/2606.02964 |
| infercept2024 | InferCept（ICML'24） | https://arxiv.org/abs/2402.01869 |
| robustmlcache2025 | Toward Robust and Efficient ML-Based GPU Caching（LCR/LARU） | https://arxiv.org/abs/2509.20979 |
| impress2025 | IMPRESS（FAST'25） | https://www.usenix.org/system/files/fast25-chen-weijian-impress.pdf |
| continuum2025 | Continuum | https://arxiv.org/abs/2511.02230 |
| ragcache2024 | RAGCache | https://arxiv.org/abs/2404.12457 |
| cachegen2024 | CacheGen（SIGCOMM'24） | https://arxiv.org/abs/2310.07240 |
| kvflow2025 | KVFlow | https://arxiv.org/abs/2507.07400 |
| layerkv2024 | LayerKV | https://arxiv.org/abs/2410.00428 |
| memserve2024 | MemServe | https://arxiv.org/abs/2406.17565 |
| dejavu2024 | DéjàVu（ICML'24） | https://arxiv.org/abs/2403.01876 |
| cachewise2026 | CacheWise | https://arxiv.org/abs/2606.16824 |
| hbfchar2026 | HBF Sucks? A Full-Stack Characterization of HBF for KV-Centric LLM Serving | https://arxiv.org/abs/2608.11668 |
| hbfrec2026 | Enabling HBF for Generative Recommendation Serving with Write-Aware KV Cache Policy | https://arxiv.org/abs/2609.07175 |
| matkv2025 | MatKV（ICDE'26） | https://arxiv.org/abs/2512.22195 |
| adaptreuse2026 | Adaptive KV Cache Reuse（CacheTune） | https://arxiv.org/abs/2605.24022 |
| kvadmission2025 | KV Admission（WG-KV；GPU 內有損准入，與 SSD 准入不同，避免名詞混淆） | https://arxiv.org/abs/2512.17452 |
| kvmgmtsurvey2026 | From Tensor Buffer to Distributed Memory Hierarchy（survey；DG3 把「按各階取回延遲定價的逐出」列為 open problem） | https://arxiv.org/abs/2607.02574 |
| wherekv2026 | Where Should the KV Cache Live? | https://arxiv.org/abs/2609.16215 |
| pykvcache2026 | Building py-kvcache | https://arxiv.org/abs/2609.11744 |
| cacheflow2026 | CacheFlow | https://arxiv.org/abs/2604.25080 |
| nearstorage2025 | HILOS（ASPLOS'26） | https://arxiv.org/abs/2502.09921 |
| tlru2025 | Tail-Optimized Caching for LLM Inference（NeurIPS'25） | https://arxiv.org/abs/2510.15152 |
| motiered2026 | Kareto | https://arxiv.org/abs/2603.08739 |
| smartgen2026、predmultitier2026、kvmem2026、cheops2025、swiftcache2026 | 已查，與 (a)–(g) 無直接關係 | 見各自 arXiv |

其他工具輸出：`research/baseline_graph.py`（反向統計）、`research/s2/*.json`（Semantic Scholar 原始回應）、`research/vllm_kv_offloading_usage.md`（vLLM 文件原始 md）。
