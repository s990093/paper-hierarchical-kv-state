# 工作負載與評測方法學查證（Tiara）

> **查證日期**：2026-09-24。
> **範圍**：本地 35 篇文字檔（`scratchpad/papers_txt/<key>.txt`，頁碼一律以檔內 `=== [page N] ===` 為準，寫成「p9」），以及網路一手來源（URL 附在各節與附錄 B）。
> **分工**：已先讀 `docs/RELATED_WORK_WEAKNESSES.md`。本文只補它缺的欄位（baseline、指標、命中率定義、到達過程、重用型態）。模型、資料、硬體只在需要時重述。
> **對齊**：已掃過協調者提供的平台 B 記錄（`scratchpad/remote/`：`results/RUNLOG_MI300X.md`、`results/CLAIM_EVIDENCE_MI300X.md`、`SOTA_MATRIX_20260919.md`、`ADVISOR_REPLY_20260919.md`、`EXPERIMENTS_20260919.md`、`main_remote.tex`）。§6 的建議與這些記錄對齊。
> **標記**：
> * 「pN」＝本地文字檔頁碼。
> * 「（計算）」＝本報告直接下載公開檔案算出的數字，方法見附錄 A。
> * 「（判讀）」＝我們的推論。
> * 「未查證」＝查不到或未確認。
> * 注意力類型（MHA／GQA／MQA／MLA）依 HF `config.json` 判定（2026-09-24 取得；gated 模型改用未 gated 的鏡像 config），不是論文原文。

---

## 0. 結論先講（10 條）

1. **「沒有統一標準」可以量化。** 有 baseline 的 32 篇共用了 **64 個不同的 baseline**，其中 **41 個只出現在一篇**。496 個論文配對中，**72.6% 的 baseline 集合完全不交集**，平均 Jaccard 只有 **0.056**。只看多層放置／跨請求重用的那 20 篇，零交集仍有 57.4%。最常見的共同對手只有 **LRU（9 篇）** 與 **vLLM（9 篇）**。
   **ARC 在 35 篇中 0 篇被當 baseline**（文字檔裡的 "ARC" 全部是 ARC-Challenge 資料集或澳洲研究委員會）。所以「只能跟 LRU、ARC 比」這個現象來自 vLLM `OffloadingConnector` 內建的兩個策略，不是文獻慣例。
2. **到達過程**：主實驗用真實時間戳的只有 **2 篇**（Bidaw、Mooncake）；用 Poisson 的 **10 篇**；用固定或手調間隔的 **4 篇**；沒有到達過程（單請求或批次）的 **19 篇**。
3. **命中率沒有共同定義**。15 篇報了命中率，粒度至少有 6 種：token 加權前綴、區塊、請求級、單層或任一層、decode 時 top-K 已在 GPU 的比例、相鄰步驟重疊。
   同一份 Mooncake `toolagent` trace（無限容量）依定義可以是 **1.1%**（請求完整命中）、**55.3%**（區塊）、**57.1%**（token）、**66.9%**（逐請求平均）或 **100%**（請求至少命中一塊）（計算，§2.2）。
4. **「prefix hit 算 full 還是 partial？」** 凡是寫出公式的論文都算 **partial**：Marconi 的 token hit rate、SAECache 的 prefill = prompt×(1−hit)、LMCache 的 `hit_tokens`、Strata 的 per-page matching。若只算 full，真實 trace 上幾乎是 0：Mooncake 1.0–1.1%、Alibaba Bailian 1.5–12.4%（計算）。
5. **公開 trace 的現況要更新。** 2025–2026 新釋出了同時具備「長上下文＋真實時間＋多使用者」的資料：
   * **TraceLab**：CC BY 4.0；357,161 次 LLM 呼叫、43 位開發者；input 中位 **124,018 token**，60.9% ≥100K（計算）。
   * **Inferact Codex SWE-bench Pro**：MIT；每次呼叫 input 中位 63,917。
   * **MLPerf Agentic**：2026-09；含內容與 `delay_seconds`。

   但這三份的重用率都是 **94–99.5%**。專案的說法應改為：「長上下文的真實 trace 已經有了，但都是高重用（agent 累積歷史）；**長上下文＋中低重用**的公開資料仍然沒有。」
6. **Bidaw 的 trace 是公開的。** 論文 p11 腳註指向 `github.com/ShipengHu-777/Interactive-conversation-workload`（2026-02-13 建立；repo 沒有 license 檔）。`SOTA_MATRIX_20260919` 寫「專有／未公開」，**需要更正**。
   以資料自身交叉驗證：1,268,346 輪、56,573 位使用者、平均 22.42 輪／人（原文 22.4）（計算）。
7. **Alibaba ATC'25 的 trace 已公開**：`alibaba-edu/qwen-bailian-usagetraces-anon`，Apache-2.0，16-token SipHash 區塊。四條 2 小時 trace 的 input 中位 574–4,540 token，最長 89,286；無限容量下 token 級重用 46.2–66.5%（計算）。
8. **AdaptCache 的設定無法重現。** 「LongBench 6 資料集、1,100 段 context」用官方資料湊不出來：v1 六集合計 1,800；qmsum 沒有 `_e` 版；`_e` 五集加 qmsum 是 1,700。它也沒有釋出程式碼或 query。
   **EvicPress** 實際用的是 12 個資料集、555 段 context、GPT-5 產生的 query，同樣沒有釋出。它的 Table 2 長度沒有標單位；與官方字數比對後，應該是**字元數**（判讀）。
9. **統一評測的既有嘗試**：沒有一個同時標準化「CPU/SSD 分層＋跨請求重用＋品質」。
   * 最接近的是 **MLPerf Agentic**：重用規則明訂、有真實延遲、有併發掃描；但不規範分層，只報吞吐–互動性的 Pareto 曲線。
   * **SCBench**：量共享 context 下的品質，但重用率是構造出來的。
   * vLLM（`timed_trace`、`prefix_repetition`）與 SGLang（`mooncake --use-trace-timestamps`、`generated-shared-prefix`、`agentic-trace`）都已經能重播帶時間與 hash 的 trace。
10. **多使用者可以把數字衝高，但那是工作點的選擇。**
    * 文獻中最大的倍數（LMCache 14×、Mooncake 525%、EvicPress 3.6×）都出現在負載掃描裡 baseline 已經飽和的點，也就是「同 TTFT 下的吞吐」這類指標在膝點附近被放大。
    * Cake 的 +26% 只是一個 23 個請求的合成例子（逐字核對屬實，§5.1）。
    * 反例：SAECache 在 Chatbot-Arena 反而慢 12–34%；Bidaw 在 ShareGPT 上增益縮水；平台 B 的 M4 在真實 trace（壓力 6–86×）上的 headroom 反而比合成 Zipf（2–8×）低。
    * 誠實的做法是報整條負載曲線，並固定對照組。

---

## 1. 評測設定矩陣

欄位：模型（注意力）｜上下文｜資料／負載｜到達過程｜重用型態｜**Baselines**｜**主要指標**｜硬體｜頁碼。
「無」代表該篇沒有這項設定（例如單請求 benchmark 沒有到達過程）。

### 1.1 多層放置／跨請求重用（20 篇）

| 論文 | 模型（注意力） | 上下文 | 資料／負載 | 到達過程 | 重用型態 | Baselines | 主要指標 | 硬體 | 頁 |
|---|---|---|---|---|---|---|---|---|---|
| **Bidaw** FAST'26 | OPT-6.7B/13B/30B（MHA）、Qwen-7B/14B（Qwen1，MHA） | 動機實驗的歷史 ≤2,048；公開 trace 含歷史的 prompt 中位 1,066、p99 8,168（計算） | 自家 interactive conversation trace（>1M 輪）＋ShareGPT（只跑 OPT-13B） | **真實時間戳**，以抽樣使用者調整 users/min；ShareGPT 用 Poisson | 每位使用者的多輪歷史 | vLLM（全重算）、CachedAttention、FlashGen（兩者依 vLLM 重做）、「全部從 host 載入」上界；驅逐比較 queue-enhanced／LFU／LRU／FIFO；Belady ghost cache | 平均端到端延遲 vs 到達率；**吞吐＝相近延遲下可撐的 users/min**；DRAM miss rate；排隊時間 CDF | 1×A800、200GB DRAM、4×SATA RAID-5 1.5GB/s | p4–p5、p10–p13 |
| **CachedAttention** ATC'24 | LLaMA-65B（MHA）、LLaMA-2-13B（MHA）、LLaMA-2-70B（GQA）、Falcon-40B（GQA，config kv=8）、Mistral-7B-32K（GQA） | 2K／4K 視窗，溢出時截掉一半；session ≤32K、≤40 輪 | ShareGPT 9K sessions，約 52K 輪（平均 5.75） | Poisson，新 session λ=1.0/s | 多輪 session 歷史 | RE（全重算＋截斷）；驅逐消融 LRU、FIFO | hit rate（DRAM＋disk）、TTFT、prefill 吞吐、GPU time、成本 | 4×A100、128GB DRAM、10TB SSD | p3、p8–p11 |
| **Mooncake** FAST'25 | dummy「LLaMA2-70B 同架構」（GQA） | 8K（ArXiv）到 128K（模擬） | ArXiv Summarization、L-Eval、模擬資料（16K–128K、cache 50%）、23,000 條真實請求 | 公開與模擬資料用 Poisson；**真實 trace 依原時間戳重播** | 512-token block-hash 前綴共享 | vLLM（vLLM-[4M]、[20M]）；過載實驗比「兩階段前拒絕」；快取分析比 LRU／LFU／LengthAwareCache | P90 TTFT／TBT（對 SLO 正規化）、SLO 下吞吐、拒絕數；Table 1 cache hit ratio | 8×A800/節點、800Gbps RDMA | p7–p8、p15–p17 |
| **Strata** OSDI'26 | Llama-3.1-8B/70B（GQA）、Qwen2.5-14B-1M（GQA）；DeepSeek-V3（MLA，只在磁碟子實驗）；Mistral-24B（GQA，動機圖） | 平均輸入 ShareGPT 680.9、ReviewMT 17,708、LooGLE 21,613、NarrativeQA 54,797 | LooGLE（105 ctx／2,410 q）、NarrativeQA（50／1,461）、ReviewMT（100／1,092）、ShareGPT（200,869 q）；Mooncake Tool-Agent 只用於 delay-hit 模擬 | Poisson；ShareGPT 另插 60 s thinking time；in-flight 上限 128 | 長文件被多人反覆查、多代理對話、聊天 | vLLM、vLLM-LMCache、TRT-LLM、TRT-LLM-HiCache、SGLang、SGLang-HiCache | 平均 TTFT、output token 吞吐；cache hit rate | 8×H200、8×H20＋NVMe 7GB/s、GH200 | p9–p12 |
| **MTDS** C&IS'26 | GPT-2 1.5B（MHA）、LLaMa-2-7B（MHA）、LLaMa-3-8B（GQA）、Qwen-3-14B（GQA） | ShareGPT（短）；Random 資料集為固定長度（數值未抄錄） | ShareGPT、合成 Random | Poisson λ=1 | ShareGPT 多輪（判讀） | 原版 vLLM（開卸載與重用）、Mooncake、LMCache | TTFT、prefill 吞吐、DRAM（active）hit rate、GPU time | 4×A10、64GB DRAM、2TB SSD | p10–p12 |
| **Tutti** preprint'26 | Llama3-8B（GQA）；GLM-4-9B-Chat-1M（GQA，2 卡 TP） | LEval 3K–200K；LooGLE 多數 >100K | LEval、LooGLE 各子集輪流抽樣 | Poisson（原文：資料集沒有原生時間戳） | 長文件跨 session 重用 | vLLM 0.12.0／0.17.0 的 HBM-only、LMCache-DRAM-LW、LMCache-SSD、LMCache-GDS | TTFT、ITL、各階 hit rate、儲存頻寬、GPU bubble、成本 | 2×H100、4×Solidigm SSD | p8–p10 |
| **LMCache** preprint'25 | Llama-3.1-8B、Sao10K-L3-8B、Llama-3.1-70B、Qwen2.5-Coder-32B、Qwen3-Coder-480B-FP8（皆 GQA） | 多輪 QA 每 query 10K（8B 為 20K） | 模擬多輪文件 QA、LongBench TriviaQA、vLLM random（PD 實驗）、公司 F／G 的長度分布 | 多輪 QA：先 40 位使用者，再依 QPS 加入；TriviaQA 用 vLLM 腳本（Poisson）；公司 trace 由數天壓成 1 小時 | 多輪文件 QA、遠端集中共享 | basic vLLM v0.10.2（GPU prefix cache）、vLLM v0.11.0 CPU offloading、兩家商用服務；PD 比 vLLM 原生 PD | TTFT、ITL、**同 TTFT 下的吞吐** | 8×H100（GMI Cloud） | p10–p13、p15–p16 |
| **HCache** EuroSys'25 | Llama2-7B/13B、OPT-30B（MHA） | 擴到 16K | ShareGPT4（每輪 input 66.8、output 358.8）、L-Eval（平均 context 16,340） | session 間 Poisson，同 session **固定 30 s**；L-Eval 的 context 以 Zipf 合成到達 | 多輪、長文件重用 | Recomputation（DeepSpeed-MII）、KV offload（AttentionStore 重做）、Ideal | TTFT、TBT、restoration speed；GPU cache hit ratio | 4×A100-40G、4×PM9A3 | p3–p4、p9–p10、p13 |
| **Cake** ICML'25 | LongAlpaca-7B/13B（MHA）、LLaMA-3.1-8B/70B（GQA；70B 用 FP8 權重） | 4K–16K（每 2K 一點） | 取 LongChat／TriviaQA／NarrativeQA 的長度，prompt 用合成 token（原文："only token length matters"） | 無（單請求）；§5.7 為一個 16K 加 22 個短請求 | 前綴快取（所有 KV 預先存好） | vLLM v0.6.2 chunked prefill（compute-only，budget 512／1024）、LMCache v0.1.4（I/O-only） | TTFT speedup；§5.7 完工時間與吞吐 | 2×A100 NVLink、1×H100；**I/O 以延遲模擬** 7–100 Gbps | p5–p8 |
| **CacheBlend** EuroSys'25 | Mistral-7B、Yi-34B、Llama-70B（皆 GQA；後兩者 8-bit） | 512-token chunk，每個 query 取 top-6 | 2WikiMQA、Musique、SAMSum、MultiNews；Musique／2WikiMQA extended（各 1,500 個原始 query，GPT-4 再各生 3 個相似 query，共 6,000） | 掃描平均 request rate（分布未說明） | RAG 的非前綴 chunk 重用 | Full KV recompute、Prefix caching（SGLang，**假設載入零延遲**）、Full KV reuse（PromptCache）、MapReduce、MapRerank | F1、ROUGE-L、TTFT、吞吐 | 2×A40 | p11–p12 |
| **EvicPress** preprint'25 | Llama-3.1-8B、Qwen2.5-14B、Mistral-7B-v0.3、Qwen3-30B-A3B（GQA）；LongChat-7B（MHA） | Table 2 平均 12K–108K，單位未標，推定為字元，約 2K–25K token（判讀，§3.5） | LongBench **12** 個資料集抽 555 段 context；每段由 GPT-5 生 100 題（50 訓練／50 測試） | QPS 掃描（分布未說明）；§6.3 用 Azure trace 的時間戳（內容用自生 query） | 同一 context 被多次查詢 | Prefill（vLLM v0.11.2）、Eviction only（LRU）、keydiff／knorm／snapkv＋LRU、IMPRESS | quality score（MiniLM 嵌入與未壓縮答案的 cosine）、TTFT、ITL、E2E；CPU-hit／SSD-hit request % | 1×H100、80GB DRAM、800GB SSD、remote 假設無限 | p7–p10 |
| **AdaptCache** BigMem'25 workshop | Llama-3.1-8B（GQA） | 未給 | 6 個 LongBench 資料集共 1,100 段 context（只用引用 [1,4–6,10,15,16] 指名） | Poisson（多種 rate） | 同一 context 重用 | Without Compression（DRAM／SSD 卸載）、KIVI＋LRU、StreamingLLM＋LRU、Prefill | TTFT；品質（與 prefill 答案比 F1／ROUGE-L／CodeBLEU）；DRAM hit rate | 1×A100、100GB DRAM、400GB SSD 1GB/s | p1–p3 |
| **Marconi** MLSys'25 | 自建 7B Hybrid（Attention＋SSM）、Jamba-1.5-Mini（hybrid；config 需登入，注意力類型未查證） | LMSys 輸出常達數千 token（見其 Fig. 6） | LMSys、ShareGPT、SWE-Agent on SWE-Bench | **手動調整** inter-session 與 inter-request 間隔 | 多輪／agent 前綴 | Vanilla（不做 prefix cache）、vLLM+（block 32）、SGLang+（LRU） | **token hit rate**、P5／P50／P95 TTFT；刻意不量品質 | 8×A100-40GB | p8–p9 |
| **SAECache** preprint'26 | Qwen2.5-1.5B（GQA，只做整合驗證）；主結果是 trace-driven 模擬 | 未查證 | ShareGPT（74% 多輪）、LMSys（33%）、Chatbot-Arena（12%）＋4 個合成負載；Qwen-Bailian 與 CC-Bench 只用來擬合輪間間隔 | **固定注入間隔** {0.02, 0.03, 0.05, 0.08} s | 多輪；模板化的單輪 | LRU、LPC | prefix cache hit ratio、mean TTFT | A40 | p4、p8–p9、p16 |
| **FlexGen** ICML'23 | OPT-6.7B/30B/175B（MHA） | prompt 補齊到 512／1024；生成 32 | 合成 | 無（離線批次） | 無 | DeepSpeed ZeRO-Inference、HF Accelerate、Petals | generation throughput；HELM | T4 16GB | p7–p8 |
| **KVPR** ACL Findings'25 | OPT-6.7B/13B/30B（MHA） | prompt 256／512／1024；生成 32／128 | 沿用 FlexGen 的補齊資料 | 無 | 無 | DeepSpeed Inference、HF Accelerate（延遲）；FlexGen（吞吐） | decode latency／throughput（5 次平均） | A100 40GB | p6 |
| **OrbitFlow** VLDB'26 | LLaMA3-8B/70B（GQA） | 每批 ≤32K；70B 到 128K | 從 ShareGPT 抽樣合成 | Poisson（預設 0.97 req/min） | 無（請求內卸載） | DeepSpeed-Inference、FlexGen、FlexGen+、SLO-aware Offloading、Dynamic Heuristic | TBT／TPOT SLO 達成率、吞吐（req/min）、E2E | A5000 24GB、4×A6000 | p9–p10 |
| **Het-Mem** IEEE CAL'25 | LLaMA-3.1-8B（GQA） | NarrativeQA 約 30K prompt＋10K decode | 單一負載＋合成的重要度變化 trace | 無 | 無（decode 期放置） | Unlimited HBM、Static Placement、Reactive（LRU）、Page Granularity（Quest 式）、SA-Guided（上界） | 正規化 tokens/s、HBM hit rate | 模擬 GH200（HBM 設 24GB） | p3–p4 |
| **LeoAM** preprint'25 | LongChat-7B-32k、Yarn-Llama-2-13B-128k、OPT-6.7B（MHA） | LongBench、PG-19（長度未抄錄） | 準確度：COPA／RTE／PIQA／OpenBookQA＋PG-19；速度：LongBench | 無（batch 1／4／8） | 無 | H2O-like、H2O-like-chunked、Prefetch-based（InfiniGen 式）、Full cache | accuracy、推論延遲 | RTX 4090、120GB、SSD 7GB/s | p9–p10 |
| **KVDrive** preprint'26 | Llama-3-8B-1048K、Qwen3-8B/14B、Phi-4-mini（GQA） | 60K–360K | LongBench、RULER | 無（批次） | 無 | Original（全 GPU）、FlexGen、Quest、ShadowKV、PQCache、MagicPIG、RetroInfer(E)、RetroInfer；GPU 快取 LRU vs LA | generation 吞吐、accuracy、GPU 命中率 | L20、H20、RTX 4090 | p15–p18 |

### 1.2 單請求：壓縮、稀疏檢索、學習式驅逐、量測（14 篇）

| 論文 | 模型（注意力） | 上下文 | 資料 | 到達 | 重用 | Baselines | 主要指標 | 硬體 | 頁 |
|---|---|---|---|---|---|---|---|---|---|
| **ArkVale** NeurIPS'24 | LongChat-7b-v1.5-32k（MHA） | passkey 10K–30K；LongBench | LongBench 6 集（HotpotQA、NarrativeQA、Qasper、GovReport、TriviaQA、PassageRetrieval）＋passkey | 無 | 無 | StreamingLLM、H2O、TOVA | LongBench 分數、passkey、page recall accuracy、decode 延遲、相對吞吐 | 1×A100 | p7–p10 |
| **QEvict** preprint'26 | Llama-3.1-8B、Qwen2.5-7B、Mistral-7B-v0.2（GQA） | LongBench；RULER 32K | LongBench 12 任務、RULER、GSM8K | 無 | 無 | StreamingLLM、SnapKV、AdaKV、CriticalKV、DefensiveKV、Layer-DefensiveKV；KIVI、KVQuant、ZipCache | 官方任務指標；TTFT、TPOT、吞吐、peak GPU；**以實測位元組比 ρ 對齊記憶體預算** | A100 | p7、p10 |
| **KVP** ICML'26 | Qwen2.5-7B、Phi-4（GQA） | 訓練約 4K；RULER 評到 128K | RULER-4K、OASST2-4K、BoolQ、ARC-Challenge 等 | 無 | 無 | TOVA、SnapKV、Random、StreamingLLM、LagKV、KeyDiff、K-Norm（附錄另有 JudgeQ） | accuracy、perplexity | 訓練 8×H100；微基準 B200 | p6–p8 |
| **ForesightKV** ICML'26 | Qwen3-1.7B/4B、R1-Distill-Qwen-7B（GQA） | 生成 8K／16K／32K；budget 1K–4K | AIME2024／2025 | 無 | 無 | SnapKV、H2O、R-KV | pass@1（32 次平均）、最大併發 batch／吞吐 | A800 | p6–p7 |
| **LookaheadKV** ICLR'26 | Llama3.1-8B、Llama3.2-1B/3B、Qwen3-1.7B/4B/8B（GQA） | 訓練 ≤16K；評 4K–32K | LongBench、RULER、LongProc、MT-Bench | 無 | 無 | SnapKV、PyramidKV、StreamingLLM、LAQ、SpecKV | LongBench 平均、RULER、MT-Bench（Qwen3-235B 當評審）、TTFT overhead | 未查證 | p6–p8 |
| **TRIM-KV** ICLR'26 | Qwen3-1.7B–14B、R1-Distill（GQA） | 32K（吞吐）；128K（LongMemEval、SCBench） | AIME24、GSM8K、MATH-500、LongProc、LongMemEval_S、LongBench-v2、**SCBench** | 無 | SCBench／LongMemEval 依多輪多 session 協定（p18） | SeerAttn-R、R-KV、SnapKV、H2O、StreamingLLM | pass@1（AIME 64 次、GSM8K／MATH 8 次）、長上下文分數、decode 吞吐 | 4×H100（訓練） | p7–p10、p18 |
| **InfiniGen** OSDI'24 | OPT-6.7B/13B/30B、Llama-2-7B/13B（MHA） | 敏感度實驗 1,920；Llama-2-7B-32K 到 32K | COPA、OpenBookQA、WinoGrande、PIQA、RTE；WikiText-2、PTB；PG-19 | 無 | 無 | 環境 UVM、FlexGen；方法 H2O、量化 | accuracy、perplexity、延遲 | RTX A6000 | p9–p13 |
| **Quest** ICML'24 | LongChat-v1.5-7b-32k、Yarn-Llama-2-7b-128k（MHA） | passkey 10K／100K | PG19、passkey、LongBench 6 集 | 無 | 無 | H2O、TOVA、StreamingLLM | perplexity、passkey、LongBench、speedup | RTX 4090（kernel） | p5–p6 |
| **ShadowKV** ICML'25 | Llama-3-8B-1M、GLM-4-9B-1M、Llama-3.1-8B、Yi-9B-200K（GQA）；NIAH 另測 Phi-3-Mini-128K（MHA）、Qwen2-7B-128K | RULER 128K；LongBench 取 >4K 的樣本 | RULER、LongBench、NIAH | 無 | 無 | Quest、Loki、InfiniGen（各有全卸載與只卸 V 兩版）；吞吐對照為全注意力的最大 batch | accuracy、decode 吞吐、batch size | A100 | p7–p8、p17 |
| **KIVI** ICML'24 | Llama-2-7B/13B（MHA）、Falcon-7B（MQA）、Mistral-7B（GQA）、LongChat-7B（MHA）、Llama-3-8B（GQA） | NIAH 到約 27K–30K token | CoQA、TruthfulQA、GSM8K、LongBench、NIAH | 無 | 無 | 全精度，以及假量化的變體 | accuracy、peak memory、吞吐 | A100 | p6–p8、p14 |
| **KVTuner** ICML'25 | Llama-3.1-8B、Qwen2.5-3B–32B、Mistral-7B（GQA） | LongBench 20 集 | GSM8K 等數學題、LongBench | 無 | 無 | KIVI-8／KIVI-4／KIVI-K8V4、per-token-asym；吞吐以 KV8 為基準 | accuracy、decode 吞吐 | 未查證 | p8–p9 |
| **KVServe** SIGCOMM'26 | Qwen2.5-7B/32B、Llama-3.1-8B（GQA） | 未抄錄 | 剖析用 GSM8K、HumanEval、Multi-News、Qasper；未見過的 2WikiMQA、HotpotQA | 未說明 | 無（PD 之間的傳輸） | Default（BF16）、CacheGen、KIVI、DuoAttention | JCT、相對準確度（97% 門檻）、壓縮比 | 剖析 4×A100；decode H100；prefill 4090／5090／Pro 6000／H100 | p10–p11 |
| **YAKV** preprint'26 | Llama-3.1-8B、Qwen3-4B/30B-A3B/32B（GQA） | 自建 10.0K–63.5K（平均 20.1K）；Text2JSON／MultiNeedle 平均約 38K | 自建 context-intensive＋NIAH、RULER、LongBench、LongProc | 無 | 無 | ShadowKV、InfiniGen、ArkVale、LRQK | accuracy；TPOT、吞吐 | H200、A100、B200 | p4、p8–p9、p24 |
| **Bottlenecks** MLSys'26 | Llama-3.1-70B、Qwen3-235B-A22B（GQA）；MLA 延後處理 | 掃描 K（已快取）與 T（新 prefill）；FinQA 文件 60K–450K | prompt＝K 個 "Hi"＋T 個隨機單 token 詞；κ_ratio 的分布用 ShareGPT／NarrativeQA／FinQA 統計 | 每個設定 200 個請求（vLLM benchmark tool） | 合成的「K 已快取＋T 新」 | **無**（量測研究） | mean TTFT±std、GPU 利用率 | H100、B200；vLLM v0.10.1＋LMCache v0.3.5；關閉 prefix caching | p5–p7 |

`kvsurvey2026`（System-Aware KV Cache Optimization survey；GitHub 標示 ACL 2026）不是評測論文，不列入矩陣，放在 §4 討論。

### 1.3 統計：把「沒有統一標準」具體化

**到達過程**（34 篇有實驗的論文）：

| 類別 | 篇數 | 論文 |
|---|---|---|
| 主實驗用真實時間戳 | **2** | Bidaw、Mooncake |
| 只在子實驗或微基準用真實時間 | 2 | EvicPress（§6.3 Azure）、Strata（Mooncake 模擬） |
| 用真實 trace 但只取長度分布、時間被壓縮 | 1 | LMCache |
| Poisson | **10** | CachedAttention、HCache（session 間）、Strata、OrbitFlow、Tutti、MTDS、AdaptCache、LMCache（TriviaQA）、Mooncake（公開／模擬）、Bidaw（ShareGPT） |
| 固定或手調間隔 | 4 | HCache（30 s）、Strata（ShareGPT 的 60 s）、Marconi、SAECache |
| 分布未說明 | 3 | EvicPress、CacheBlend、KVServe |
| 無到達過程（單請求或批次） | **19** | ArkVale、Bottlenecks、Cake、FlexGen、ForesightKV、Het-Mem、InfiniGen、KIVI、KVDrive、KVP、KVPR、KVTuner、LeoAM、LookaheadKV、QEvict、Quest、ShadowKV、TRIM-KV、YAKV |

**合成或生成的負載**：15 篇的主負載含合成成分。
* 補齊或只看長度：FlexGen、KVPR、Cake、Bottlenecks、MTDS。
* 用 LLM 生 query：EvicPress（GPT-5）、CacheBlend（GPT-4）。
* 抽樣後再合成：OrbitFlow。
* 合成到達或順序：HCache（Zipf）、Strata（重排 cache distance）。
* 其他：LMCache、SAECache、Het-Mem、Mooncake（simulated）、AdaptCache（query 來源未說明）。

**與 LRU／ARC 比較**：
* 把 LRU 當 baseline 的有 9 篇：AdaptCache、Bidaw、CachedAttention、EvicPress、Het-Mem、KVDrive、Marconi（SGLang+）、Mooncake、SAECache。
* **ARC：0 篇。**

**baseline 重疊**（計算，方法見附錄 A.4）：
* 32 篇共 64 個 baseline 名稱，41 個只出現一次。
* 出現次數前幾名：LRU 9、vLLM 9、StreamingLLM 7、H2O 6、SnapKV 6、Prefill／全重算 6、KIVI 4、LMCache 4、FlexGen 4。
* 496 個配對中 360 個（72.6%）零交集，平均 Jaccard 0.056。
* 只看多層／重用 20 篇：190 個配對中 57.4% 零交集，平均 Jaccard 0.091。扣掉 LRU 類古典策略與全重算之後，系統級的共同對手只剩 vLLM（9）、LMCache（4）、DeepSpeed（3）、FlexGen（3）。

**模型注意力**：
* 8 篇的模型**全是 MHA**：ArkVale、Bidaw、FlexGen、HCache、InfiniGen、KVPR、LeoAM、Quest。
* 含 MLA 的只有 Strata 的一個磁碟子實驗；Bottlenecks 明說把 MLA 延後（p7）。
* 只有 Marconi 用 hybrid。

**指標**：
* 以 TTFT 為主指標的 14 篇。
* 品質指標至少 9 種：官方任務指標、F1、ROUGE-L、嵌入 cosine（EvicPress）、對未壓縮答案的相似度（AdaptCache）、相對準確度門檻（KVServe）、pass@1、perplexity、LLM 評審（MT-Bench）。
* 刻意不量品質的也有：Marconi（p8："prefix reusing is exact"）、Strata、Bidaw、Mooncake（dummy model）、Cake。

**24 GB 級單卡**：只有 4 篇在這類卡上做實驗，其中消費級只有 RTX 4090（LeoAM、KVDrive）；另外兩篇是工作站或資料中心卡（OrbitFlow 的 A5000、MTDS 的 A10）。

---

## 2. 命中率定義稽核

### 2.1 逐篇：粒度、full／partial、算哪一層

| 論文 | 名稱 | 粒度 | full／partial | 算哪一層 | 原文（節錄） | 頁 | 數字 |
|---|---|---|---|---|---|---|---|
| Marconi | token hit rate | **token** | partial（前綴；SSM 只能在 checkpoint 精確命中） | GPU prefix cache | "We define token hit rate as the ratio of the number of tokens that skipped prefill over the total number of input tokens." | p8 | SWE-Bench trace：SGLang+ 16.4%、Marconi 32.7%（p9） |
| SAECache | prefix cache hit ratio | **token**（由公式推得） | partial | GPU（vLLM evictor） | "prefill_tokens = prompt_length × (1 − hit_ratio)" | p8 | 領先 4.8–5.9 百分點 |
| LMCache | prefix cache hit ratio；API 回傳 `hit_tokens`／`hit_chunks` | token，對齊 chunk | partial | 任一層（CPU／disk／remote） | "returns a list of (instance_id, storage_device, hit_tokens)"；"prefix cache hit ratios drop from roughly 85% to 45% when truncating" | p10、p15–p16 | 公司 G 的生產環境 50%（p16） |
| Strata | cache hit rate | **page**（1–512 token） | partial | 分 device／host（Fig. 7），報總和 | "cache matching is performed on a per-page basis"；"increasing the page size leads to a significant drop in the KV cache hit rate" | p4、p10、p12 | 長上下文約 95%；page 512 時少 2.4% |
| Mooncake | cache hit rate（Table 1）；Cache Ratio（Table 2）；可重用上限 | **block**（512 token） | 前綴區塊 | 假設單一全域池 | Table 1 容量 Inf…1,000 blocks；"up to only 50% of the KVCache can be reused" | p8、p15、p18 | LRU 0.30–0.51 |
| CachedAttention | cache hit rate | 未定義（判讀：以 session KV 為單位） | 未定義 | **任一層**（DRAM＋disk），另外分層報 | "total KV cache hit rates, including both DRAM and disk hit rates" | p9、p11 | 86／71／89／90%；10T 時 CA 86%、LRU 58%、FIFO 48% |
| Bidaw | performance layer hit／miss rate；hit potential | **請求級**（一次存取＝一個請求載入該使用者的歷史） | 未定義；且快取的是 tensor 6，不是 KV | **只算 DRAM** | "Each dot represents a KV access corresponding to one request"；hit potential＝Belady 下的命中率 | p5–p6、p8–p9、p12–p13 | 容量 40.1% 時約 20%；miss rate −57.6%／−69.9% |
| MTDS | active／inactive hit；DRAM hit rate | 未定義 | 未定義 | DRAM＝active、SSD＝inactive | "we refer to KV cache hits occurring in DRAM as efficient hits or active hits, while those occurring in SSD are referred to as inactive hits" | p3、p12 | 91.4／89.2／85.1／73.6% |
| Tutti | 各儲存階的 cache hit rate | 未定義 | 未定義 | 判讀：以「該階為容量上限」計 | Table 1：HBM 8／4%、DRAM 53／24%、SSD 84／86%（LEval／LooGLE）；Fig. 2 還把 "hit rate = 75%" 當控制變因 | p3、p9 | 同一份資料 4%–86% |
| EvicPress | CPU-hit／SSD-hit request %；cache hit tokens | 圖上是請求級，文字寫 token，**兩者混用** | 未定義 | 分層 | 圖 9 軸 "CPU-hit request %"；"most of the cache hit tokens reside on CPU DRAM" | p9–p10 | — |
| AdaptCache | DRAM cache hit rate | 未定義（判讀：一個 KV entry＝一段 context） | 不適用（壓縮會改變 entry） | 只算 DRAM | "KIVI LRU achieves a DRAM cache hit rate of 38% … our method achieves hit rates of 81%, 56%, 44%, and 11%" | p2–p3 | 隨 α 從 81% 到 11% |
| HCache | cache hit ratio | 判讀：context 級 | 未定義 | GPU LRU cache | "With a uniform arrival pattern, the cache hit ratio is 15% … rises to 94%" | p13 | 15%→94% |
| KVDrive | hit rate | token（每步被選中的 critical entries） | 不適用（不是前綴） | GPU | "approximately 80% of critical entries are served directly from the in-GPU cache" | p12、p17 | 70.0–91.0% |
| ShadowKV | KV cache hit rate／chunk hit rate | chunk（本步選中者，上一步是否已在 GPU） | 不適用 | GPU | "The KV cache has a high hit rate, reducing computations and data movements by over 60%"；"chunk hit rate … around 60%" | p4、p9 | 約 60% |
| Het-Mem | HBM hit rate | decode 時的每次存取（模擬） | 不適用 | HBM | Fig. 5 "The HBM hit rates" | p4 | — |
| Bottlenecks | κ_ratio（重用 token／prefill token，**不是命中率**） | 每個請求 | — | — | 中位數：ShareGPT 100、NarrativeQA 5,000、FinQA 10,000 | p5–p6 | — |
| Cake | 無自己的命中率；引用 AttentionStore「約 80% 的命中發生在磁碟層」 | — | — | — | — | p2 | — |

### 2.2 同一份 trace，不同定義（計算）

做法：依時間戳排序，從空快取開始重播。前綴只算到第一個缺口為止（vLLM 語意）。最後一個不完整的 block 只有在整個前綴都命中時才計入。定義如下：
* **block**：命中的前綴 block 數 ÷ 總 block 數。
* **token**：命中的 token 數 ÷ 總 input token 數。
* **請求：任一**：至少命中一個 block 的請求比例。
* **請求：完整**：所有 block 都命中的請求比例。
* **逐請求平均**：每個請求的命中比例取平均。

| trace（block 大小） | 容量 | block | token | 請求：任一 | 請求：完整 | 逐請求平均 |
|---|---|---|---|---|---|---|
| Mooncake `toolagent`（512） | 無限 | 55.3% | 57.1% | 100.0% | **1.1%** | 66.9% |
| 同上 | LRU 1,000 blocks（=512K token） | 34.0% | 35.1% | 99.9% | 0.1% | 54.9% |
| Mooncake `conversation`（512） | 無限 | 36.6% | 37.4% | 100.0% | **1.0%** | 40.9% |
| 同上 | LRU 1,000 blocks | **4.4%** | 4.5% | 100.0% | 0.1% | 16.2% |
| Mooncake `synthetic`（512，Poisson） | 無限 | 64.0% | 65.1% | 44.6% | 5.3% | 43.2% |
| Bailian traceA，To-C 聊天（16） | 無限 | 57.9% | 58.1% | 99.5% | 2.3% | 59.9% |
| Bailian traceB，To-B API（16） | 無限 | 54.2% | 54.6% | 98.7% | 4.4% | 59.9% |
| Bailian thinking（16） | 無限 | 46.2% | 46.2% | 92.1% | 12.4% | 57.5% |
| Bailian coder（16） | 無限 | 66.4% | 66.5% | 99.6% | 1.5% | 52.8% |

**讀法**：
1. **完整命中幾乎為 0**，因為最後一個 block 通常含這輪新打的字。
2. **任一命中幾乎為 100%**，因為第一個 block 多半是共用的系統提示。「請求命中率」如果不說清楚是哪一種，數字可以在 1% 與 100% 之間任選。
3. token 與 block 級差距小（0–2 點），但**逐請求平均**與 token 級可以差 10 點以上，而且方向不固定：toolagent 高 9.8 點，Bailian coder 低 13.7 點。逐請求平均讓每個請求等權，所以數字偏向「短請求的命中比例」；token 級則偏向長請求。
4. 容量一有限，差距最大的是 `conversation`：無限時 37.4%，1,000 blocks 時只剩 4.5%。

**與原文比對**：
* 我們用公開檔重算 Mooncake Table 1（p8，LRU，容量 Inf／100K／50K／30K／10K／1K blocks）得 0.553／0.552／0.551／0.537／0.460／0.340。原文是 0.51／0.51／0.50／0.48／0.40／0.30，**系統性低約 0.04**。原文沒給定義，差異原因未查證。**連 trace 作者自己的命中率，外人都無法照定義重現。**
* 原文 p7 寫這份 trace「average input length of 7,590 tokens」，公開檔算出 **8,590**（差恰好 1,000，判讀為筆誤）。另外，`arxiv-trace/mooncake_trace.jsonl` 與 FAST'25 的 `toolagent_trace.jsonl` 都是 23,608 筆，平均 8,590 對 8,596，各定義的命中率完全相同（判讀：同一批請求）。
* 專案說「Mooncake toolagent／conversation 重用率 37–57%」，與上表 token 級、無限容量的 37.4% 與 57.1% 一致。

### 2.3 文獻中「設定一改，命中率就變」的證據

| 變因 | 證據 | 頁 |
|---|---|---|
| 到達分布（合成的偏斜度） | HCache：同一份 L-Eval，uniform 到達時 GPU 命中 15%，Zipf α=2.0 時 94% | p13 |
| page／block 大小 | Strata：page 從 1 放大到 512，命中率明顯下降，平均與 P90 TTFT 分別升到 2× 與 2.9× | p4 |
| 請求順序 | Strata：同一份 LooGLE 排成 min／shuffle／max cache distance。delay-hit 緩解只在 min distance 有 +42%；I/O 機制在 shuffle／max 分別有 +76%／+95%。**哪個機制重要，由順序決定** | p12 |
| 截斷策略 | LMCache：公司 F 的 trace 在「只保留最新 token」的截斷下，prefix 命中從約 85% 掉到 45% | p15 |
| 資料集本身 | Mooncake Table 2：ArXiv 約 0%、L-Eval >80%、真實 trace 約 50%。p18："the real reusability in our online traces is much smaller than the results reproduced by open-source benchmarks" | p15、p18 |
| 最低一層是哪一層 | Tutti Table 1：同一份資料，HBM 4–8%、DRAM 24–53%、SSD 84–86% | p9 |
| 重用的來源 | SAECache：多輪重用 99.28%（ShareGPT 型）與 99.96%（AgentBank 型）發生在同一 session 內；單輪請求跨 session 的重用 <0.01% | p4 |
| 缺口之後的 block | Tiara 自己的模擬器：第一個缺口之後仍留在任一層的 block 只有 0.000%–0.464%，所以「前綴語意 vs 逐 block 語意」對前綴結構的 trace 影響可忽略 | `main_remote.tex` L1722 |

### 2.4 回答「prefix hit 是 full 還是 partial？」

* **文獻慣例是 partial**：token 加權，前綴算到第一個缺口為止，粒度是引擎的 block 或 page（vLLM／Bailian 16 token、Mooncake 512、LMCache chunk 256、Strata page 1–32 預設）。
* **只算 full 在真實 trace 上沒有意義**：Mooncake 1.0–1.1%、Bailian 1.5–12.4%（§2.2）。
* **一定要寫清楚的 4 件事**：
  1. 粒度與 block 大小；
  2. 是 token 加權還是請求級；
  3. 只算 GPU 還是任一層；
  4. 是否排除暖機期，以及容量是否有限。

  缺一項，數字就不可比。§6.3 的協定把這 4 項定為必填。

---

## 3. 公開負載與 trace 清單（全部上網查證，2026-09-24）

欄位說明：
* **時間戳**：是否有每個請求或每一輪的到達時間。
* **內容／hash**：能否精確算出重用。
* **長度**：若有標「計算」，是本報告用公開檔算的。

### 3.1 生產 trace（有時間戳）

| 名稱與 URL | 時間戳 | 內容／hash | 多輪／身分 | 長度（token） | License | 35 篇中誰用過 | 備註 |
|---|---|---|---|---|---|---|---|
| **Mooncake FAST'25 traces**，`kvcache-ai/Mooncake/FAST25-release`（https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release） | ms，重播用 | **前綴 block hash，512 token**（README 明載；`ceil(input/512)=len(hash_ids)` 在四個檔 100% 成立，計算） | 無 session id，靠 hash 推斷 | conversation：12,031 筆，中位 6,909、p90 27,367、最大 126,195。toolagent：23,608 筆，中位 6,346、p90 16,810、最大 126,195。synthetic：3,993 筆，中位 11,587、最大 191,378（計算） | repo 為 Apache-2.0（trace 沒有另外標 license） | Mooncake；Strata（toolagent 模擬）；Bidaw 引用 conversation 的統計（p4）；Tiara | synthetic 那份是 Poisson 到達。SGLang 內建下載與重播（§4） |
| **Alibaba Qwen-Bailian**（ATC'25 "KVCache Cache in the Wild"），https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon | 秒（到小數第三位），每條 2 小時 | **salted SipHash，16-token block** | `chat_id`、`parent_chat_id`、`turn`、`type` | traceA（To-C）：43,058 筆，中位 1,046、最大 89,286。traceB（To-B API）：172,800 筆，中位 574，全為單輪。thinking：10,812 筆，中位 3,680、最大 51,622。coder：43,011 筆，中位 4,540、最大 25,777。**≥32K 的比例：0%–0.18%**（計算） | Apache-2.0 | 35 篇中只有 SAECache（只用來擬合輪間間隔，p16） | README 另列 LMetric（OSDI'26）使用。官方重播器：https://github.com/blitz-serving/trace-replayer 。README FAQ：最後一個 block 可能含 padding，hash 會變（使完整命中偏低） |
| **Bidaw Interactive-conversation-workload**，https://github.com/ShipengHu-777/Interactive-conversation-workload | 整數秒，涵蓋 21.35 h | **只有長度**（`User_id, Timestamp, Query_length, Response_length, Round_index`） | user_id＋round_index | 1,268,346 輪、56,573 位使用者；query 平均 35.3、response 平均 44.6；輪數平均／中位／p90＝22.42／18／45，**與原文 p4 的 22／18／45 一致**；同一使用者的輪間隔 p50 43 s、p99 142 s（計算） | **repo 沒有 license**（GitHub API `license=null`） | Bidaw | 重用只能在「prompt＝完整歷史」的假設下推算：含歷史的 prompt 中位 1,066、最大 47,624，其中歷史佔 97.7%（計算） |
| **Azure LLM Inference 2023**（Splitwise），https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md | 有 | 只有長度（`TIMESTAMP, ContextTokens, GeneratedTokens`） | 無 | code：8,819 筆，中位 1,469、最大 7,437。conv：19,366 筆，中位 1,020、最大 14,050。檔內時間為 2023-11-16 18:15–19:14（計算；md 寫 "November 11th"） | CC-BY-4.0 | EvicPress §6.3（只用時間戳）；Bidaw 只引用 | 無法算重用 |
| **Azure LLM Inference 2024**（DynamoLLM），https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md | 有，一週（2024-05） | 只有長度 | 無 | 未計算 | CC-BY-4.0 | 35 篇中未見使用 | 同上 |
| **BurstGPT**（KDD'25），https://github.com/HPMLL/BurstGPT | 秒 | 只有長度 | v2.0（2026-01-15）的 `BurstGPT_3` 才有 `Session ID` 與 `Elapsed time`；`data/BurstGPT_1.csv` 沒有 | BurstGPT_3：5,344,021 筆（conversation log 233,617）；request tokens 中位 327、≥32K 只有 0.029%；55,920 個對話 session，中位 2 輪；session 內輪間隔 p50 131 s（計算） | CC-BY-4.0 | 只有 Bidaw 引用 | vLLM 的 `burstgpt` 選項只隨機抽 GPT-4 列的長度，**不用時間戳**（§4） |
| **ServeGen**（NSDI'26），https://github.com/alibaba/ServeGen | 主資料是每位 client 的分布，不是逐請求；`conversations_hashed.json` 有 unix 時間戳 | 對話檔：`output_tokens` 為逐 token 雜湊（count/len=1.0）；`input_tokens` **語意未文件化**：count/len 比值 p10／p50／p90＝15／511／5,872，既不是固定 block，也不是「本輪新增 token」（計算） | 對話檔：1,616 段對話、5,720 輪，時間 2025-02-16～17 | 對話檔 input_token_count 中位 7,750、p90 55,321、最大 262,644，18.1% ≥32K（計算） | Apache-2.0 | 35 篇中未見使用 | **長上下文＋真實時間＋多輪**，但重用無法由 hash 驗證；83% 的相鄰輪次滿足「下一輪 input ≥ 上一輪 input＋output」（計算）。另有 offline batch traces（ACDC，SOSP'26） |
| **Splitwise traces** | 同 Azure 2023 | — | — | — | — | — | Azure 2023 那份就是 Splitwise 的資料 |
| **TraceLab**（UW SyFI，2026），https://github.com/uw-syfi/TraceLab ；資料 `releases/v0.0.1/syfi_coding_trace.jsonl.gz` | **ms，逐事件**（user message、tool emit 等），2025-09-23 → 2026-06-04 | **沒有內容與 hash**，工具輸入被刪除；有 `prefix_tokens`／`newly_append_tokens`／provider 的 cache split | session、round、43 位匿名開發者（Claude Code 與 Codex） | 357,161 次呼叫、4,265 個 session；input 中位 **124,018**、p90 256,767、p99 822,895；**60.9% ≥100K**；47.0% >131,072；9.7% >262,144（計算） | 資料 **CC BY 4.0**；程式碼 Apache-2.0 | 無（2026 新資料） | `prefix+new=input` 在 100% 的列成立；Claude 列的 `prefix_tokens`＝`cache_read` 完全相同。所以 95.7% 是 **provider 實際命中率**，不是理想的可重用上限（計算＋判讀）。輪間隔 p50 8.8 s、p99 1,782 s；同一個 5 分鐘內的活躍 session p90 為 4 個，它們最新 context 的總和 p90 約 1.06M token（計算） |

### 3.2 對話內容資料集

| 名稱與 URL | 時間戳 | 內容 | 多輪／身分 | License | 誰用過 | 備註 |
|---|---|---|---|---|---|---|
| **ShareGPT**（`anon8231489123/ShareGPT_Vicuna_unfiltered`），https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered | **無** | 有（`id`, `conversations[{from,value}]`） | 多輪；無使用者身分 | HF 標 apache-2.0（內容爬自 sharegpt.com，實際權利未查證） | CachedAttention、Bidaw、HCache（ShareGPT4）、Marconi、OrbitFlow、SAECache、Strata、MTDS、Bottlenecks | V3 的 id 帶 `_0` 之類的後綴，repo 附 `split_long_conversation.py`。判讀：長對話被切段，多輪鏈可能不完整 |
| **LMSYS-Chat-1M**，https://huggingface.co/datasets/lmsys/lmsys-chat-1m | **無時間戳欄位** | 有 | 多輪；1M 段對話（2023-04～08） | 需同意 LMSYS-Chat-1M License Agreement（gated，禁止再散布） | Marconi、SAECache（LMSys） | — |
| **Chatbot Arena conversations**，https://huggingface.co/datasets/lmsys/chatbot_arena_conversations | **有**（`tstamp`） | 有 | 平均 1.2 輪；33K 段；匿名 user id | 使用者 prompt CC-BY-4.0；模型輸出 CC-BY-NC-4.0 | SAECache | 幾乎是單輪 |
| **WildChat-1M**，https://huggingface.co/datasets/allenai/WildChat-1M ；**WildChat-4.8M**（2025-08-08 建立） | **有，逐則 assistant 訊息**（user 訊息為 null，已用 datasets-server 抽樣確認） | 有 | 多輪；`hashed_ip` 可串同一使用者的多段對話；837,989 段 | ODC-BY（兩版皆 ungated） | **35 篇中 0 篇** | 公開資料裡唯一同時有內容、逐輪時間、使用者串連的聊天集。時間是回應完成的時間，不是請求到達時間（判讀） |

### 3.3 長上下文 benchmark（單請求或共享 context）

| 名稱與 URL | 形態 | 規模／長度 | License | 誰用過 | 備註 |
|---|---|---|---|---|---|
| **SCBench**（ICLR'25），https://huggingface.co/datasets/microsoft/SCBench ；程式 https://github.com/microsoft/MInference/tree/main/scbench | 12 個任務；**兩種共享模式**：multi-turn（預設）與 multi-request（`--same_context_different_query`） | 922 列；context 299K–3.17M 字元；每列 2 個以上的 turn | MIT | TRIM-KV（p9–p10） | 重用是構造出來的：所有 turn 共用整段 context。`main_remote.tex` L1616 記為重用 80% |
| **LongBench v1／LongBench-E**，https://github.com/THUDM/LongBench | 21 個任務，單請求 | 4,750 筆；多數任務平均 5K–15K 字；LongBench-E 按長度 0–4K／4–8K／8K+ 均勻抽樣 | MIT | ArkVale、EvicPress、AdaptCache、KIVI、KVDrive、KVP、KVTuner、LeoAM、LMCache、LookaheadKV、QEvict、Quest、ShadowKV、TRIM-KV、YAKV 等 | §3.5 有 6 個資料集的實際筆數 |
| **LongBench v2**，https://huggingface.co/datasets/zai-org/LongBench-v2 | 選擇題 | 503 題 | apache-2.0（HF 標籤） | TRIM-KV | — |
| **RULER**，https://github.com/NVIDIA/RULER | 合成、長度可調 | 自訂 | Apache-2.0（程式碼） | KVDrive、KVP、LookaheadKV、QEvict、ShadowKV、YAKV | — |
| **LooGLE**，https://github.com/bigai-nlco/LooGLE | 長文件 QA、摘要 | 多數 >100K（Tutti p9） | repo MIT；HF 資料 CC-BY-SA-4.0 | Strata、Tutti | — |
| **L-Eval**，https://github.com/OpenLMLab/LEval | 20 個子任務 | 3K–200K（Tutti p9） | **GPL-3.0** | HCache、Mooncake、Tutti | — |

### 3.4 Agent／coding trace（2025–2026 新釋出的特別重要）

| 名稱與 URL | 時間 | 內容／hash | 規模與長度 | 重用 | License | 備註 |
|---|---|---|---|---|---|---|
| **TraceLab**（見 §3.1） | 真實，逐事件 | 無內容；有 prefix／new 的切分 | input 中位 124K | provider 命中 95.7% | CC BY 4.0 | **目前最貼近 Tiara 目標長度的真實 trace** |
| **Inferact codex_swebenchpro_traces**（2026-05-06），https://huggingface.co/datasets/Inferact/codex_swebenchpro_traces | README 給**統計**：呼叫間隔平均 10.5 s、p50 5.2、p99 81.4 s（含模型回應、工具執行、agent 處理時間）。檔內有沒有逐呼叫時間戳：**未查證**（HF viewer 只顯示 `conversations` 欄） | **有內容**（ShareGPT 格式，218 MB） | 610 個 trial、20,230 次呼叫；每次呼叫 input 平均 68,329、p50 63,917、p90 114,888、p99 166,322；結尾 context p50 80,488（README） | README：整體 cache hit 94.2%（1,301.5M／1,382.3M token）；93.8% 的 trial 第一次呼叫就命中共用的 11,520-token 前綴 | MIT | vLLM×Mooncake 部落格用它做 agentic 評測（https://vllm.ai/blog/2026-05-06-mooncake-store）。有內容，所以能用我們的 tokenizer 切成 16-token block 精確計算重用（判讀） |
| **MLPerf Agentic Inference dataset**（MLCommons，2026），https://endpoints.mlcommons-storage.org/index.html ；說明 https://github.com/mlcommons/endpoints/tree/main/examples/10_Agentic_Inference | 每則訊息的 `delay_seconds`（工具或使用者延遲） | **有內容**（JSONL，一列一則訊息，含 system、tools、tool_calls、tool_results、reasoning_content） | 613 條軌跡：113 條由 DeepSWE 任務產生的**生成**軌跡，500 條 Workato **合成**客服對話。估算（字元÷4，未 tokenize）：DeepSWE 請求 context 中位約 84K、最大約 238K，80.5% ≥32K；Workato 中位約 41K，500 段共用同一份約 36K token 的系統提示＋工具目錄（計算） | 理想重用：DeepSWE 99.5%、Workato 99.0%（字元加權，計算）。延遲：DeepSWE p50 0.15 s；Workato p50 7.74 s（計算） | Apache-2.0（編彙）；Workato CC BY 4.0；DeepSWE 為寬鬆授權的組合（LICENSES.txt） | 官方規則：軌跡內 prefix cache 完全允許；同一輪資料集內允許系統提示共用；**跨輪重複禁止**（以 salt 強制） |
| **CC-Bench trajectories**（Z.ai），https://huggingface.co/datasets/zai-org/CC-Bench-trajectories | 軌跡內含 SDK 精度的時間（SAECache p16） | 有（`trajectory` 字串＋token 總數） | SAECache 記為 370 條 Claude Code 軌跡，5 個 LLM、6 類任務 | 未計算 | MIT | SAECache 用來擬合 agent 的輪間間隔（p16） |
| **SWE-bench agent 軌跡**：`nebius/SWE-agent-trajectories`（CC-BY-4.0）、`nebius/SWE-rebench-openhands-trajectories`（CC-BY-4.0）、`SWE-bench/SWE-smith-trajectories`（MIT）、`SWE-Gym/OpenHands-SFT-Trajectories`（MIT） | **無**（欄位沒有時間） | 有 | 1 萬到 10 萬筆級 | 未計算 | 如左 | Marconi 用 SWE-Agent on SWE-Bench；SGLang `agentic-trace` 可載入 OpenHands／SWE-smith 格式 |
| **CacheWise**（arXiv 2606.16824） | — | — | 自收的 coding assistant trace | — | — | 摘要沒有提到釋出，**未查證** |
| **Continuum**（arXiv 2511.02230） | — | — | SWE-Bench、BFCL、OpenHands 的 trace 重播 | — | — | 摘要頁看不出有釋出，未查證 |

### 3.5 AdaptCache 與 EvicPress 的 LongBench 設定核對

**AdaptCache**（`adaptcache2025`，3 頁）：
* **原文沒有列出資料集名稱**。p2 只寫 "1,100 contexts from six LongBench datasets [1, 4–6, 10, 15, 16]"。對照參考文獻，[1] 是 LongBench 本身，其餘六個是 SAMSum、LongCoder（=LCC）、TriviaQA、RepoBench、HotpotQA、QMSum，也就是使用者列的六個。
* **沒有任何 `_e` 字樣**，所以「lcc_e、repobench_p_e」是否為原設定，未查證。
* 同一段內前後矛盾：p2 開頭寫 "three LongBench datasets"，評測段寫 "six"。
* **筆數對不上**（計算，資料取自官方 `zai-org/LongBench` 的 `data.zip`）：

  | 資料集 | v1 筆數 | `_e` 筆數 |
  |---|---|---|
  | qmsum | 200 | **不存在** |
  | samsum | 200 | 300 |
  | triviaqa | 200 | 300 |
  | hotpotqa | 200 | 300 |
  | lcc | 500 | 300 |
  | repobench-p | 500 | 300 |

  v1 合計 1,800；`_e` 五集加 qmsum 為 1,700。**兩者都不是 1,100**，抽樣方式沒有說明。
* **query 來源**：原文只說估計器「samples ten entries from each dataset, using questions generated by GPT 4o」（p2），**沒有說評測用的重複 query 從哪來**。`RELATED_WORK_WEAKNESSES.md` 寫「問題由 GPT-4o 產生」，嚴格說只成立於剖析步驟。
* **程式碼**：搜尋不到 repo，arXiv 頁也沒有連結。**未公開**（2026-09-24 查證）。

**EvicPress**（`evicpress2025`）：
* 實際用的是 LongBench **12** 個資料集（Table 2，p7）：narrativeqa、qasper、multifieldqa_en、hotpotqa、2wikimqa、musique、gov_report、qmsum、multi_news、trec、triviaqa、samsum。**不含 lcc 與 repobench-p**。
* 抽 555 段 context，每段由 **GPT-5** 生 100 題（50 訓練／50 測試）（p8）。
* **Table 2 的單位沒有標**。與官方 task.md 的 Avg len 比對：narrativeqa 108K／18,409＝5.9、qasper 24K／3,619＝6.6、multi_news 12K／2,113＝5.7、samsum 34K／6,258＝5.4；12 個資料集的比值都在 5.4–6.6。判讀：這是**字元數**，不是 token。所以 "108K" 大約只有 25K token 量級。
* **程式碼與 query**：arXiv HTML 沒有任何 repo 或 dataset 連結。**未公開**。

**結論**：兩篇都無法照原設定重現。若 Tiara 要用這組資料，應自己產生 query 並公開，並寫明「與 AdaptCache 的資料選集相同，筆數與 query 不同」。另外，這六集的長度偏短：v1 平均 1,235–10,614 字；`_e` 版平均 5,546–6,685 字，最大 23,278。遠低於 Tiara 的目標區間（65K+）。

### 3.6 小結：「沒有公開資料同時具備長 context 與真實重用率」要怎麼改

| | 中低重用（≤70%） | 高重用（≥90%） |
|---|---|---|
| **短到中等長度（中位 <10K）＋真實時間** | Mooncake（37–57%）、Bailian（46–67%）、Azure／BurstGPT（只有長度，算不出重用） | Bidaw（假設 prompt＝完整歷史時為 97.7%） |
| **長上下文（中位 ≥40K）＋真實時間** | **找不到** | TraceLab（95.7%，provider 實際命中）、Codex SWE-bench Pro（94.2%）、MLPerf Agentic（99%+，內容為生成或合成） |
| **長上下文＋沒有真實時間** | 無 | SCBench（構造出來的約 80%）、LongBench 重複查詢（重用率由研究者決定） |

**待確認的候選**：ServeGen 的對話檔是長上下文（中位 7,750、最大 262,644）、有真實時間、有多輪，但 `input_tokens` 的語意不明，重用無法驗證。

**建議的新說法**（供寫作階段參考，本文不改 `main.tex`）：
> 「長上下文且帶真實時間的公開 trace 在 2026 年已出現（TraceLab、Codex SWE-bench Pro、MLPerf Agentic），但它們都是 agent 型流量，重用率 94–99.5%；中低重用的長上下文 trace 仍然沒有公開資料。」

這也提出一個值得在論文正面回應的問題：**真實世界的長上下文，本來就以高重用的 agent 型為主嗎？** 若是，Tiara 的價值應該從「預測誰會被重用」轉向「重用必然發生時，誰該留在快層」，也就是容量與時間局部性的問題（判讀）。

---

## 4. 統一評測的既有嘗試

| 名稱 | 標準化了什麼 | 負載／到達 | 跨請求重用 | CPU/SSD 分層 | 品質 | 來源 |
|---|---|---|---|---|---|---|
| **Rethinking KV Cache Compression**（MLSys'25） | 把壓縮方法放進生產級框架量：①在 FlashAttention／PagedAttention 下吞吐不如預期；②壓縮會讓**輸出變長**，端到端延遲反而增加；③看逐樣本準確度而不是平均；並釋出工具 | 未涵蓋 | 摘要未提 | 摘要未提 | 有（逐樣本） | https://arxiv.org/abs/2503.24000 ；code https://github.com/LLMkvsys/rethink-kv-compression |
| **NVIDIA kvpress＋leaderboard** | 統一的 press 介面（主要在 prefill 壓縮；`DecodingPress` 為實驗功能）；把 context 與 question 分開（壓一次、回答多題）；CLI 評 LooGLE、RULER、ZeroSCROLLS、InfiniteBench、LongBench v1／v2、NIAH。**Leaderboard 只有 RULER 4,096 token**，每個模型、每種 press 在壓縮比 0.25–0.94 下的準確度 | 單請求 | 無 | 無（只有 transformers 的 QuantizedCache） | 有（只有準確度，沒有延遲） | https://github.com/NVIDIA/kvpress （Apache-2.0）；https://huggingface.co/spaces/nvidia/kvpress-leaderboard （由 `benchmark/ruler__4096__<model>__<press>__<ratio>` 目錄確認） |
| **"KV Cache Compression, But What Must We Give in Return?"**（EMNLP'24 Findings） | 10 種以上方法、5 類（量化、token 丟棄、prompt 壓縮、線性時間模型、hybrid），7 類任務；15 個 LongBench 資料集＋passkey；模型 Llama-3-8B、Mistral-7B-v0.2、LongChat-7B。逐條結論未逐字查證 | 單請求 | 無 | 無 | 有 | https://arxiv.org/abs/2407.01527 ；https://github.com/henryzhongsc/longctx_bench （含完整 log） |
| **SCBench**（ICLR'25） | KV 生命週期四階段（生成、壓縮、取回、載入）；12 個任務 × 2 種共享模式；8 類方法、8 個模型；`kv_type` 選項 dense／kivi／snapkv／quest／pyramidkv／streamingllm | multi-turn／multi-request | **有，但由構造決定** | 無（只有「載入」這個階段的概念） | 有 | §3.3 |
| **LMCache benchmarks** | `long_doc_qa`：合成文件，預設 20,000 token、重複 2 次，`--repeat-mode random/tile/interleave`（等於調整重用距離），附 `--hit-miss-ratio`。`multi_round_qa`：users × rounds × QPS，含共用系統提示與個別歷史，可改用 ShareGPT。另有 `multi_doc_qa`（CacheBlend）、`rag`、`storage_backend_io`（磁碟、io_uring 微基準） | QPS（到達分布未查證） | 有（合成） | 有（由 LMCache 後端決定） | 無 | https://github.com/LMCache/LMCache/tree/dev/benchmarks （Apache-2.0） |
| **vLLM `vllm bench serve`** | `--dataset-name`：sharegpt、burstgpt、sonnet、random、random-mm、random-rerank、hf、custom、custom_audio、custom_image、**prefix_repetition**（預設前綴 256、後綴 256、10 組前綴、輸出 128）、spec_bench、speed_bench、**timed_trace**（JSONL 帶時間與 hash，預設 16-token chunk，可縮放時間）。到達：`--request-rate` 配 `--burstiness`（1.0＝Poisson，其他為 gamma），另有 ramp-up。`benchmarks/multi_turn/`：「Benchmark KV Cache Offloading with Multi-Turn Conversations」（合成對話、num-clients、max-active-conversations） | Poisson／gamma／trace 時間 | prefix_repetition（合成）；timed_trace（依 hash） | multi_turn 基準用來測卸載 | 無 | https://github.com/vllm-project/vllm/blob/main/vllm/benchmarks/datasets/datasets.py 、`vllm/benchmarks/serve.py`、`benchmarks/multi_turn/README.md`（main 分支，2026-09-24） |
| **SGLang `sglang.benchmark.serving`**（舊名 `bench_serving`） | `--dataset-name`：agentic-trace（OpenHands／SWE-smith 格式，把真實回覆接回下一輪）、sharegpt、custom、openai、random、random-ids、**generated-shared-prefix**（組數、每組 prompt 數、系統提示長度、問題長度、輪數；組的分布 uniform 或 **zipf**）、mmmu、image、**mooncake**（conversation／toolagent／synthetic／arxiv，`--mooncake-slowdown-factor`、`--mooncake-num-rounds`）、longbench_v2、speed-bench。`--request-rate` 為 Poisson（預設 inf，全部在時間 0 送出）；**`--use-trace-timestamps` 只對 mooncake 有效** | Poisson 或 trace 時間 | 有 | HiCache 另有其評測（未查證） | 無 | https://github.com/sgl-project/sglang/tree/main/python/sglang/benchmark |
| **MLPerf Inference v6.0**（2026-04） | 標準 LLM 情境（GPT-OSS-120B、DeepSeek-R1 含 interactive、Llama-3.1-405B 取樣自 LongBench／RULER 等）；accuracy gate＋Offline／Server 等情境 | 依情境 | 未規範 | 未規範 | 有 gate | https://mlcommons.org/2026/04/mlperf-inference-v6-0-results/ （搜尋摘要） |
| **MLPerf Inference v6.1 Agentic**（2026-09-16） | 613 條軌跡；**保留資料集給的輪間延遲**；併發掃描（活躍使用者數）；**明訂 prefix cache 規則**；報吞吐對 E2E 互動性的 Pareto 曲線；accuracy gate（inline＋SWE-bench）；模型 Kimi K3（MoE＋KDA／Gated MLA）、Qwen3.6-35B-A3B（Gated DeltaNet hybrid）、DeepSeek-V4-Pro | 真實延遲＋閉迴路併發 | **有，且有規則** | 不規範（由提交者決定） | 有 | https://mlcommons.org/2026/07/agentic-inference-for-mlperf-inference/ ；README 見 §3.4 |
| **MLPerf v6.1 Edge Agentic** | Qwen3.6-27B Q4_K_M GGUF；20 段對話、1,007 輪；context 固定 32K；SingleStream（併發 1）；報 TTFT、TPOT、每輪 E2E 的 p50／p90／p99／max；3% 準確度門檻 | 單流 | 軌跡內 | 無 | 有 | https://mlcommons.org/2026/07/mlperf-inference-v61-edge-agentic/ |
| **InferenceMAX**（SemiAnalysis，現改名 InferenceX） | ISL／OSL 三組：1K/1K、1K/8K、8K/1K（輸入長度隨機取 80–100%）；Pareto 分析。原文："We opted for benchmarking requests that are random sequences to avoid prefix caching…" | 隨機 | **刻意避開** | 無 | 無 | https://newsletter.semianalysis.com/p/inferencemax-open-source-inference ；https://github.com/SemiAnalysisAI/InferenceX |
| **llm-d `inference-perf`／tiered prefix cache** | 「250 prefix groups × 5 prompts each on a 60-second Poisson interval」（guide）；部落格版：每個 query＝一個看過的 2,000-token 使用者系統提示＋256-token 問題，輸出 256，40 QPS，使用者池大小可變；分層 HBM→CPU RAM→檔案系統 | Poisson | 有（合成） | **有** | 無 | https://github.com/llm-d/llm-d/blob/main/guides/tiered-prefix-cache/README.md ；https://llm-d.ai/blog/native-kv-cache-offloading-to-any-file-system-with-llm-d （2026-02-10） |
| **KV survey 附錄 G.3**（`kvsurvey2026`） | p9 C6："inconsistent metric definitions and measures across tools, preventing reliable apples-to-apples comparisons"。p26–27 建議：指標含尾端延遲、goodput、**"KV hit rate in memory tiers"**、跨層 I/O 量、KV 相關 stall；負載含多租戶或突發、長上下文、異質（agent）；報告含漸增的 context 長度、準確度對記憶體曲線、硬體拓撲 | 原則，沒有實作 | — | — | — | 本地 p9、p26–p27 |

**小結**：
* 沒有任何一個同時標準化**分層放置＋跨請求重用（含真實時間）＋品質約束**。
* 品質類（kvpress、EMNLP'24、Rethinking）都是單請求。
* 服務類（vLLM、SGLang、LMCache、llm-d、InferenceMAX）不量品質；而且 InferenceMAX **刻意避開** prefix cache。
* MLPerf Agentic 是第一個把「真實延遲＋重用規則＋併發」寫進標準的，但它不規範分層，也不量分層的成本。
* survey 把「KV hit rate in memory tiers」列為建議指標，但沒有給定義，正好是 §2 的空缺。

---

## 5. 多使用者

### 5.1 Cake 逐字核對（`cake2025` p8，§5.7）

原文：
> "In this evaluation, we give an example to evaluate the performance of the adaptive scheduling algorithm in Cake. We begin by sending a prefix-caching request of length 16K … To simulate a burst of incoming requests from other users, we then randomly generate 22 additional requests with sequence lengths ranging from 32 to 448, following a spiked distribution."
> "… reduces the overall finish time from 1.5s to 1.19s, improving throughput by 26%."

**使用者筆記逐項核對結果：屬實。** 另外要補充 5 點：
1. 原文自稱是 "an example"，**只有一次執行**，沒有變異數。
2. "spiked distribution" 沒有定義。
3. 對照組是 vLLM 的預設排程（先把 prefix 請求做完），不是另一個快取系統。
4. 23 個請求裡只有 1 個有 prefix cache，**不是多使用者重用**，而是「一個重用請求加上其他使用者的算力競爭」。
5. 26% 是完工時間換算來的（1.5／1.19 = 1.26）。

其他章節的「多使用者」也是模擬：p6 用 vLLM token budget 的佔比代表「其他使用者佔用的 GPU 利用率」。

### 5.2 其他論文怎麼做多租戶或並發

| 論文 | 做法 | 頁 |
|---|---|---|
| Bidaw | 使用者到達率（每分鐘新使用者數）掃描；「吞吐」定義為相近延遲下可撐的 users/min；只評「單卡在去除冗餘計算後算得完」的到達率 | p4–p5、p12 |
| CachedAttention | session 以 Poisson λ=1.0/s 到達；LLaMA-13B 用 2 卡、24 batches | p8–p9 |
| LMCache | 多輪 QA 先放 40 位使用者，其餘依 QPS 加入；報「同 TTFT 下的吞吐」 | p11 |
| EvicPress | QPS 掃描；"At the same TTFT level, EVICPRESS achieves 2.0 to 3.6× higher request processing rate" | p9 |
| Mooncake | 真實 trace 依原時間重播；過載實驗加快重播速度；"up to a 525% increase in throughput in certain simulated scenarios"；真實流量下 "approximately 75% more requests while adhering to the SLOs" | p1、p16 |
| Strata | Poisson，in-flight 上限 128；**delay hit**：同一 context 的多個請求在第一次 miss 還沒解決時就到達，有效命中率下降；min cache distance 時 delay-hit 緩解帶來 +42% | p3、p5、p10、p12 |
| Tutti | 子資料集輪流抽樣模擬多 session，Poisson；"Under a 1s TTFT SLO, Tutti increases the effective request rate by 50% over DRAM and by 100% over GDS" | p9–p10 |
| HCache | session 用 Poisson＋固定 30 s；13B 受 GPU 記憶體限制，「各方法吞吐幾乎一樣」；7B／30B 最多多撐 11% 請求 | p9–p10 |
| OrbitFlow | batch 4；"attainment drops to around 50% as the arrival rate increases" | p9–p10 |
| SAECache | 固定注入間隔、GPU 飽和的高負載區；Chatbot-Arena 的 TTFT 比最佳 baseline 慢 12–34% | p8 |
| MTDS | batch 20、Poisson λ=1 | p10 |
| Marconi | 手動改變 session 間與請求間的間隔 | p8 |
| KVDrive、ShadowKV、ForesightKV、YAKV | 「多使用者」只等於 batch size 或最大併發 batch，沒有到達過程 | 各見 §1 |
| MLPerf Agentic | 活躍使用者數（target concurrency）掃描＋資料集給的延遲 | §4 |

### 5.3 「多使用者能不能把數據衝高？」

**能，而且有兩個獨立的機制。**

1. **容量壓力只在多使用者時才出現。**
   * 單請求 benchmark 沒有跨請求的驅逐，分層的價值是 0。
   * Tiara 平台 B 的 M4 在壓力 1× 時 headroom 一律 0%（`RUNLOG_MI300X` L561）。
   * Bidaw Fig. 5（p5）顯示：同時快取的 KV 總量隨到達率上升，並超過 200 GB 的 performance layer。
   * AdaptCache p1：以 Llama-3.1-70B 為例，兩台 H100 節點加 150 GB CPU 記憶體只放得下 500K token，約 30 位使用者的聊天歷史。
2. **排隊在飽和膝點會放大比值。**
   * 「同 TTFT 下的吞吐」或「高 QPS 下的 TTFT」這類指標，在 baseline 接近飽和時會非線性放大。
   * 文獻中最大的倍數都出現在這種點：LMCache 2.3–14×（p11）、Mooncake 525%（p1）、EvicPress 2.0–3.6×（p9）、Tutti +100%（p10）。
   * 同一篇在低負載點的差距小得多。例如 LMCache 在 QPS=1 時 TTFT 小 1.9–8.1×，但那個倍數本身也受 baseline 排隊影響（判讀）。

**但不保證會衝高。反例**：
* SAECache 在 Chatbot-Arena（12% 多輪）上多佇列的額外開銷超過命中收益，TTFT 反而慢 12–34%（p8）。
* Bidaw 換到 ShareGPT（Poisson 時間）後，吞吐增益「decreases evidently」（p12）。
* HCache 13B 在 GPU 記憶體受限時，各方法吞吐一樣（p10）。
* Tiara 平台 B 的 M4：合成 Zipf 在壓力 2–8× 時 headroom 21–43%；真實 trace 在 6–86× 時反而只有 7–21%（`RUNLOG_MI300X` L582–595）。壓力太高時重用消失，最強 baseline 已經拿走能拿的。
* TraceLab 的實際併發：43 位開發者、8 個月，同一個 5 分鐘內的活躍 session 中位 1、p90 4、最大 24（計算）。「很多人同時用長 context」在這份真實資料裡並不普遍。

**判讀**：多使用者讓分層「有事可做」，是必要條件；但報出來的倍數大小幾乎由「選哪個工作點」決定。誠實的報法：
* (a) 報整條負載曲線與膝點位置，不只報峰值比值；
* (b) 對固定的一個 baseline 比（見 §6.3）；
* (c) 同時報 prefill-only 與端到端 headroom。

---

## 6. 建議

### 6.0 與平台 B 記錄對齊：哪些已做、哪些已被推翻、哪些要更正

| 項目 | 狀態 | 依據 |
|---|---|---|
| Mooncake conversation／toolagent／synthetic，各取前 4,000 請求做 Oracle | **已做** | `RUNLOG_MI300X` L149、L582 |
| 合成 Zipf α=0.9、壓力 1–8× 掃描 | **已做** | `RUNLOG_MI300X` L552 |
| 24 份 LongBench 文件 × 4,096、Zipf 重複（注意力對齊實驗） | **已做** | `RUNLOG_MI300X` L696 |
| 品質：needle、GSM8K、LongBench 7 英文任務、RULER × 5 精度 × 7 模型 | **已做** | `CLAIM_EVIDENCE` A8；`main_remote.tex` L1252–1258 |
| SCBench 已下載（31 檔，大小與 HF 相符） | 已下載，未見用於評測 | `RUNLOG_MI300X` L153 |
| 只看 GSM8K 判定 ε | **已被推翻**（GSM8K 對量化不敏感，撈針可掉 100pp） | `CLAIM_EVIDENCE` A9 |
| 「vs 最佳 baseline」當主結果 | **已被推翻**（27% 的 bootstrap 抽樣會讓最佳 baseline 換人） | `EXPERIMENTS` §3.3 |
| 不標口徑的 headroom | **已被推翻**（端到端 ≈ prefill-only × 0.2） | `CLAIM_EVIDENCE` B3 |
| 全 BF16 前提下的 GO | **受挑戰**（全域切 INT4＋LRU 就拿到 oracle 空間的 91.8%） | `EXPERIMENTS` §1.2 |
| 模擬層 A/A（換 seed） | **無意義**（trace 重播是決定性的） | `ADVISOR_REPLY` L235–250 |
| 「工作集 ≤ 各階容量總和則 headroom ≡ 0」要寫成 loader assertion | 尚未完成 | `ADVISOR_REPLY` L231–233、L564 |
| `SOTA_MATRIX`：「Bidaw trace 專有／不公開」 | **需更正**：已公開（§3.1），可以重跑 Bidaw 的工作負載，也能實作它的 previous-answer 驅逐來比 | 本報告 §3.1 |
| `SOTA_MATRIX`：「Azure／BurstGPT／LMSys 在五篇中未出現」 | 放到 35 篇仍大致成立：LMSys 有 Marconi、SAECache；Azure 只有 EvicPress 的微基準；BurstGPT 只被引用 | 本報告 §3 |
| `SOTA_MATRIX` 被推翻的「Alibaba 理想命中 62%／54%」 | 本報告**另外**用公開檔算出 token 級無限容量 58.1%（traceA）、54.6%（traceB）。這是新計算，**不是**那條被推翻的引用；引用時必須附定義 | §2.2 |
| `main_remote.tex` L1082、L1616：「目標長度區間無公開 trace」 | **需修正措辭**：長上下文真實 trace 已存在，但都是高重用（§3.6） | §3.4 |

### 6.1 建議的工作負載（4 個，各有分工）

**W1｜Mooncake FAST'25 `toolagent`＋`conversation`（保留，作為生產重用的錨點）**
* **理由**：
  * 唯一被大量引用、有 ms 時間戳與前綴 hash 的生產 trace；
  * 重用 37–57% 落在「中度」；
  * Strata、SGLang 都支援重播，審稿人熟悉。
* **取得**：`kvcache-ai/Mooncake/FAST25-release/traces/`（Apache-2.0 repo）。
* **相對已做的延伸**：
  * 改用**全量**（12,031／23,608），而不只前 4,000 筆；
  * 暖機窗排除規則寫進 manifest；
  * 報 §6.3 的命中率三件組與無限容量上限。無限容量上限就是「強制 miss 下界」：token 級 37.4%／57.1%（計算）。

**W2｜TraceLab（新增，取代或補充「重用率與長度由我們設定」的合成長上下文）**
* **理由**：
  * 目前唯一公開、**真實時間、多使用者（43 人）、長上下文**（中位 124K、60.9% ≥100K）的 trace；
  * CC BY 4.0；
  * 直接回應 `main_remote.tex` L1411 與 L1616 自承的限制。
* **取得**：`github.com/uw-syfi/TraceLab/releases/v0.0.1/syfi_coding_trace.jsonl.gz`（53.6 MB）。
* **注意**：
  1. 沒有內容與 hash。只能重建「session 內前綴鏈」（本輪＝上一輪前綴＋新增），跨 session 的共用無法辨識。`round_index=0` 且 `prefix_tokens=0` 的列只有 1,167 筆，而 session 共 4,265 個（計算）。若每個 session 都有 round 0，代表多數 session 的第一輪就帶 prefix，可能是跨 session 共用系統提示或 session 接續。「每個 session 都有 round 0」這一點未驗證，兩種成因也無法由資料區分（判讀）。
  2. `prefix_tokens` 是 provider 的**實際**命中（Claude 列與 `cache_read` 完全相同），所以是可重用前綴的**下界**，不是理想上限。
  3. token 是 Claude／OpenAI 的 tokenizer 算的，換成 Qwen／Llama 要換算，並依 CLAUDE.md 規則 6 在 loader 裡加斷言。最自然的一條是 `prefix+new==input`（已驗證 100% 成立）。
  4. 9.7% 的呼叫 >262,144（Qwen noDCA 上限），47.0% >131,072（Llama-3.1 上限），要事先定好截斷或過濾規則，並記錄比例。
* **備選**：
  * **Inferact Codex SWE-bench Pro**（MIT，有內容，可以用我們的 tokenizer 切 16-token block 精確算重用；但逐呼叫時間戳未查證）；
  * **MLPerf Agentic**（有內容與延遲，但 DeepSWE 軌跡為生成、Workato 為合成）。

  三者都是 agent 型、高重用。要在論文寫明「長上下文＋中低重用」仍然沒有公開資料。

**W3｜品質：沿用 LongBench＋RULER＋needle，另外加 SCBench 的 multi-turn 模式**
* **理由**：這三套已做完。需要補的是 ε 的量測：樣本數要夠，而且要用對量化敏感的任務（RULER 的多鍵、多值、CWE）。
* **SCBench 的角色**：唯一評「壓縮後跨輪使用」的 benchmark（KV 生命週期裡的載入與重用），適合量「GPU-INT4 狀態下被跨輪重用」的品質代價。
* **限制**：它的重用率是構造的，**只能拿來量 ε，不能拿來支撐命中率或 headroom 的主張**。
* **取得**：`microsoft/SCBench`（MIT），平台 B 已經下載。

**W4｜短上下文、人類節奏的負對照：Alibaba Bailian traceA（首選）或 Bidaw trace**
* **理由**：
  * Bailian 同時有真實時間、16-token hash、`parent_chat_id` 多輪、Apache-2.0；
  * 16-token 剛好等於 vLLM 預設 block，vLLM `timed_trace` 的預設 chunk 也是 16（判讀：預設值相同，能不能直接吃還需實測）。
  * 它的 context 很短（中位 1,046），用來展示「短 context 下 κ 感知放置沒有價值」。這與 `main_remote.tex` L1328「今日的生產流量量不到放置決策的價值」同一方向，能把負面結果變成有依據的適用範圍。
* **Bidaw trace 的用途**：它是公開資料裡唯一有「每使用者 >20 輪、人類思考時間」的 trace。若要與 Bidaw 的 previous-answer 驅逐正面比較，就用它。
* **取得**：見 §3.1。

**不建議當主負載的**：
* AdaptCache／EvicPress 的 LongBench 組合：無法重現，且過短（§3.5）；
* LMSYS-Chat-1M：沒有時間戳，且 license 禁止再散布；
* Azure／BurstGPT：只有長度，算不出重用。

### 6.2 ShareGPT：要不要用、怎麼用

* **不要把 ShareGPT 當時間相關的負載。**
  * Bidaw §5.3（p12）示範過：時間戳換成 Poisson 後，依時間的驅逐策略 "no longer reduces miss rates"。
  * Tiara 的預測器特徵（EDC／delta）本身是時間的函數（`main_remote.tex` L992），用 Poisson 補時間會讓預測器評估無效。
  * Strata 給 ShareGPT 插的 60 s 固定 thinking time（p10）一樣是人造的。
* **若審稿人要求，只用兩種方式**：
  1. **只拿長度與內容分布**，當「短 context、低壓力」的補充，並明說沒有用它的時間；
  2. **時間從真實 trace 移植**，並明說是移植的。移植來源依內容相近度排序：
     * WildChat-1M 的逐則 assistant 時間戳；
     * BurstGPT_3 的 Session ID 輪間隔（p50 131 s，計算）；
     * Bidaw trace 的輪間隔（p50 43 s，計算）；
     * Bailian 的 parent→child 間隔。

     SAECache 對 Bailian 聊天擬合的 log-normal 參數也可參考：μ≈4.1、σ≈1.0，以 log 秒為單位（p17）。
* **更好的替代品**：直接用 **WildChat-1M**。它有內容、逐輪時間、`hashed_ip`，ODC-BY，**35 篇中沒有人用過**，能同時回答「為什麼不跑 ShareGPT」與「時間戳哪來」。

### 6.3 統一評測協定草案（v0）

**A. 每個 run 的 manifest（必填；缺一項，結果不得寫進 `results/`）**
1. 模型 HF id＋revision；注意力類型與 KV bytes/token（由 config 算）；各層的 KV dtype。
2. 引擎與版本、connector 名稱；**block／page 大小（token）**；trace 的 hash 粒度（Mooncake 512、Bailian 16）以及換算方式。
3. 各層容量（bytes）與「工作集÷GPU 容量」的壓力。
4. **斷言**：主張分層有價值時，工作集必須大於 GPU＋CPU 容量，否則 headroom 恆為 0（`ADVISOR_REPLY` L231–233）。
5. 到達過程：原生時間戳（寫明時間縮放倍數）、Poisson（寫明 rate 與 burstiness）或閉迴路併發 N（寫明延遲來源）。
6. trace 切片（全量或前 N 筆）與暖機排除規則；請求順序（原始或打亂；Strata p12 證明順序會改變結論）；併發上限。
7. 硬體與**實測**頻寬（fio＋引擎計數器）；重複次數與 CI 方法。真機層的 A/A 至少 5 次；模擬層不做 seed A/A。

**B. 必報指標**
1. 延遲：TTFT、TPOT／ITL、E2E 的 p50／p90／p99；decode 佔總時間的比例。
2. 吞吐：明訂 SLO（TTFT 與 TPOT 門檻）下的 goodput，並附**整條負載曲線**，不只峰值比值（§5.3）。
3. headroom：每次都標 **prefill-only 或端到端**，並附 bootstrap CI（`CLAIM_EVIDENCE` B3、`EXPERIMENTS` §3.3）。
4. 品質：逐任務的保留率，以及**最差任務**（`main_remote.tex` 的 $Q(\pi)$ 式），用配對 bootstrap CI；任務集必須含對量化敏感的任務。
5. **快取**。在 `main_remote.tex` L2039「服務層級分佈」（每次存取由哪一階滿足，加上 GPU 常駐率與重算率）之外，再固定報以下四項：
   * **H1（主）**：token 加權的前綴命中率，採 Marconi 的定義（跳過 prefill 的 token ÷ 總 input token）。前綴算到第一個缺口（vLLM 語意），分「只算 GPU」與「任一層」兩個版本。
   * **H2**：請求級的「任一命中」與「完整命中」。
   * **H3**：trace 原生粒度的 block 命中率。
   * **上限**：同一 trace 在無限容量下的 token 級重用，也就是強制 miss 的下界。

   §2.2 的表是這四項在 Mooncake 與 Bailian 上的參考值。
6. 搬移：各層讀寫的位元組數、持續寫入頻寬（`main_remote.tex` tab:writebw）、等待 KV 的 stall 時間比例。

**C. 對照組（固定一組，不再「vs 當次最佳」）**
1. `full_gpu`（不卸載）。
2. vLLM 原生 `lru`、`arc`，並註明 ARC 在 35 篇中無人使用，只因為 vLLM 內建才納入。
3. `tier_fs`（**主對照**：固定用它比，另外報 baseline 排名的穩定度）。
4. LMCache（能跑才放；平台 B 標 NOT_MEASURED）。
5. **強制加入簡單 heuristic**：全域 INT4 KV＋LRU（`EXPERIMENTS` §1.2 顯示它吃掉 91.8% 的 oracle 空間）。
6. 至少一個已發表的放置或驅逐策略，在同一成本模型下重做。候選：CachedAttention 的佇列前瞻驅逐、Bidaw 的 previous-answer 驅逐（需人類節奏的 trace，W4）、Marconi 的 FLOP-aware 驅逐（改成 GQA 版）。目前 AdaptCache、Cake、Bidaw、Strata 都還沒有人嘗試過（`ADVISOR_REPLY` L195–198）。

**D. 命中率的一句話定義（可直接放進論文）**
> 「命中率指 token 加權的連續前綴命中：一個請求的命中量為其前綴中、自第一個 block 起連續存在於某一層的 token 數，第一個缺口之後一律視為未命中；block 大小為 B token（vLLM 預設 16；以 Mooncake trace 重播時為其 hash 粒度 512，須註明）。分別對 GPU（BF16）與任一層計算，並另報請求級的完整命中與任一命中。所有數字排除前 X 分鐘暖機，並附同一 trace 於無限容量下的上限。」

### 6.4 還沒查證或需要後續處理的

* Inferact Codex traces 的檔內是否有逐呼叫時間戳：未查證（只看了 README 與 viewer 欄位）。
* ServeGen `input_tokens` 的語意：未文件化；它的 NSDI'26 PDF 超過抓取上限，未讀。
* MLPerf Agentic 的 token 數是字元÷4 的估算，沒有真正 tokenize。
* Jamba-1.5-Mini 的注意力 config 需要登入：未查證。
* AdaptCache 的 1,100 如何抽樣：原文未說明，無法查證。
* Mooncake Table 1 與我們的重算差約 0.04：原因未查證。
* EMNLP'24 Findings benchmark 的逐條結論：只核到摘要與 README，未逐字核對論文內文。
* CacheWise、Continuum 的 trace 是否釋出：未查證。
* SGLang HiCache 自己的評測腳本：未查證。

---

## 附錄 A　本報告自行計算的方法（可重現）

所有下載都以串流餵給 Python，**沒有寫入專案目錄**。輔助腳本放在 `scratchpad/tools/`：`kw.py`（頁碼檢索）、`mc_hit.py`、`bailian.py`。

**A.1 Mooncake**（`mc_hit.py`）
1. 讀 JSONL 並依 `timestamp` 排序。
2. 單位斷言：`ceil(input_length/512) == len(hash_ids)`，四個檔都 100% 成立。
3. 前綴匹配：從第一個 hash 開始，連續存在就算命中，第一個不存在即停止。
4. token 級：命中 block × 512；若整個前綴都命中，改計 `input_length`。
5. LRU：以 block 為單位，每次存取移到尾端，超過容量時從頭部逐出。

**A.2 Bailian**（`bailian.py`）：同 A.1，block 大小 16；單位斷言 `ceil(input/16) == len(hash_ids)` 100% 成立；只算無限容量。檔案從 `media.githubusercontent.com` 的 LFS 連結串流取得。

**A.3 其他**
* **Bidaw**：以空白分隔 7 個 part 並略過標頭；檢查 `round_index` 在每位使用者內連續、時間隨輪次單調（兩者皆 100% 成立）；「含歷史 prompt」＝該使用者先前所有 query＋response，加上本輪 query（假設，非原文）。
* **BurstGPT_3**：只統計 `Log Type` 為 Conversation 且 `Session ID` 非空的 session。
* **TraceLab**：round 的時間取該 round 最早的 `timing_events`；活躍 session 以 5 分鐘分桶，同一桶內各 session 取最大的 input。
* **MLPerf Agentic**：略過 `_type` metadata 列；一個請求＝緊接在 assistant 列之前的非 assistant 列；context＝system＋tools＋所有先前訊息的字元數；token≈字元÷4（估算）。
* **LongBench**：從 `zai-org/LongBench` 的 `data.zip` 讀入記憶體（沒有落地），逐檔計數；`length` 欄取平均。

**A.4 baseline 重疊**
* 每篇的 baseline 集合依 §1 表格正規化：「Full KV／FP16／Ideal／上界」不算對手；同名系統合併，例如 DeepSpeed-Inference 與 ZeRO-Inference 合併為 DeepSpeed。
* 計算全部 496 個配對的交集與 Jaccard。
* 子群為 §1.1 的 20 篇。

## 附錄 B　查證來源 URL（2026-09-24）

**Trace 與資料集**
* Mooncake：https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release
* Alibaba Bailian：https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon ；trace replayer：https://github.com/blitz-serving/trace-replayer
* Bidaw trace：https://github.com/ShipengHu-777/Interactive-conversation-workload
* Azure：https://github.com/Azure/AzurePublicDataset （2023／2024 的 md）
* BurstGPT：https://github.com/HPMLL/BurstGPT （Release v2.0）
* ServeGen：https://github.com/alibaba/ServeGen ；https://arxiv.org/abs/2505.09999
* TraceLab：https://github.com/uw-syfi/TraceLab ；https://syfi.cs.washington.edu/blog/2026-06-25-tracelab/
* Inferact Codex SWE-bench Pro：https://huggingface.co/datasets/Inferact/codex_swebenchpro_traces ；https://vllm.ai/blog/2026-05-06-mooncake-store
* MLPerf Agentic：https://mlcommons.org/2026/07/agentic-inference-for-mlperf-inference/ ；https://github.com/mlcommons/endpoints/tree/main/examples/10_Agentic_Inference ；https://endpoints.mlcommons-storage.org/index.html
* MLPerf Edge Agentic：https://mlcommons.org/2026/07/mlperf-inference-v61-edge-agentic/ ；v6.1 結果：https://mlcommons.org/2026/09/mlperf-inference-v6-1-results/
* CC-Bench：https://huggingface.co/datasets/zai-org/CC-Bench-trajectories
* SWE 軌跡：https://huggingface.co/datasets/nebius/SWE-agent-trajectories 、https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories 、https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories 、https://huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories
* 對話資料：https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered 、https://huggingface.co/datasets/lmsys/lmsys-chat-1m 、https://huggingface.co/datasets/lmsys/chatbot_arena_conversations 、https://huggingface.co/datasets/allenai/WildChat-1M 、https://huggingface.co/datasets/allenai/WildChat-4.8M
* 長上下文 benchmark：https://huggingface.co/datasets/microsoft/SCBench 、https://arxiv.org/abs/2412.10319 、https://github.com/THUDM/LongBench 、https://huggingface.co/datasets/zai-org/LongBench-v2 、https://github.com/NVIDIA/RULER 、https://github.com/bigai-nlco/LooGLE 、https://github.com/OpenLMLab/LEval
* CacheWise：https://arxiv.org/abs/2606.16824 ；Continuum：https://arxiv.org/abs/2511.02230
* AdaptCache：https://arxiv.org/abs/2509.00105 ；EvicPress：https://arxiv.org/abs/2512.14946

**評測工具與統一評測的嘗試**
* Rethinking KV Cache Compression：https://arxiv.org/abs/2503.24000 ；https://github.com/LLMkvsys/rethink-kv-compression
* kvpress：https://github.com/NVIDIA/kvpress ；https://huggingface.co/spaces/nvidia/kvpress-leaderboard
* EMNLP'24 Findings benchmark：https://arxiv.org/abs/2407.01527 ；https://github.com/henryzhongsc/longctx_bench
* LMCache benchmarks：https://github.com/LMCache/LMCache/tree/dev/benchmarks
* vLLM：https://github.com/vllm-project/vllm （`vllm/benchmarks/datasets/datasets.py`、`vllm/benchmarks/serve.py`、`benchmarks/multi_turn/`）
* SGLang：https://github.com/sgl-project/sglang/tree/main/python/sglang/benchmark
* InferenceMAX：https://newsletter.semianalysis.com/p/inferencemax-open-source-inference ；https://vllm.ai/blog/2025-10-09-blackwell-inferencemax
* llm-d：https://github.com/llm-d/llm-d/blob/main/guides/tiered-prefix-cache/README.md ；https://llm-d.ai/blog/native-kv-cache-offloading-to-any-file-system-with-llm-d
* MLPerf v6.0：https://mlcommons.org/2026/04/mlperf-inference-v6-0-results/

**模型 config（注意力類型判定）**：https://huggingface.co/<model>/resolve/main/config.json ，模型包括 THUDM/glm-4-9b-chat-1m、microsoft/Phi-4-mini-instruct、01-ai/Yi-9B-200K、01-ai/Yi-34B、tiiuae/falcon-40b、tiiuae/falcon-7b、Qwen/Qwen-7B、Qwen/Qwen-14B、Yukang/LongAlpaca-7B/13B、lmsys/longchat-7b-v1.5-32k、microsoft/Phi-3-mini-128k-instruct、microsoft/phi-4、Qwen3 系列、deepseek-ai/DeepSeek-V3、Qwen2.5-1M 系列、gradientai/Llama-3-8B-Instruct-Gradient-1048k、mistralai/Mistral-Small-24B-Instruct-2501、NousResearch 的 Llama-2／3.1 鏡像、unsloth/mistral-7b-instruct-v0.3、huggyllama/llama-65b、facebook/opt-13b、openai-community/gpt2-xl。
