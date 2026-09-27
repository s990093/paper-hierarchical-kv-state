# 相關工作的短板總表

> 2026-09-23 逐篇讀原文整理。PDF 在 `papers/`（已 gitignore），下載來源見 `papers/SOURCES.tsv`。
> **每一格都來自原文**；「§x」「原文：」是出處。沒有寫出處的判斷會標「（我們的判讀）」。
> Strata（OSDI'26）與 MTDS（C&IS'26）由使用者另外提供 PDF，已於同日補讀並加入 §2.1。

---

## 0. 先看結論：八種反覆出現的短板

| 短板 | 白話 | 中招的論文 |
|---|---|---|
| 🧬 **綁模型架構** | 方法只在某種 attention 結構上划算／成立 | **Bidaw**、**HCache**（只對 MHA 有利）；**Marconi**（只給 Attention+SSM 混合模型）；CacheBlend（只限 Transformer）；Yandex/YAKV（只限 full attention） |
| 🗓 **模型太舊** | 主實驗用 2023 年以前的 MHA 模型（OPT、LLaMA-1、Llama-2-7B/13B、LongChat），而現在的主流（Llama-3、Qwen2 以後）都是 GQA | FlexGen、KVPR、InfiniGen、ArkVale、Quest、LeoAM、HCache、Bidaw |
| ⏱ **時間戳是假的** | 資料集沒有到達時間，只好用 Poisson 補，或手動設定間隔 | CachedAttention、HCache（每輪固定間隔 30 s）、OrbitFlow、Tutti、Marconi、LMCache（把好幾天的 trace 壓縮成 1 小時）、Bidaw（ShareGPT 那組） |
| 🧪 **負載是合成的或生成的** | prompt 補成同一長度、只看長度不看內容，或讓 GPT 生問題 | FlexGen、KVPR、Cake、EvicPress（GPT-5 生 query）、AdaptCache（GPT-4o 生 query）、SAECache（請求模擬器）、Bottlenecks（重複 prompt） |
| 🎯 **只測特定領域** | 只在一種任務型態上驗證 | ForesightKV、TRIM-KV（數學推理）；CacheBlend（RAG）；Bidaw（短訊息聊天）；KVDrive、ShadowKV、ArkVale、QEvict、Quest（單請求長文 QA） |
| 📏 **品質在短任務上量** | 宣稱做長上下文，準確度卻在短任務上量 | **LeoAM**（「< 1% 準確度下降」是在 COPA/RTE/PIQA/OpenBookQA 上量的）、InfiniGen（同一組 few-shot 任務 + WikiText） |
| 💻 **模擬而非實測** | 延遲或頻寬是用算的，或整篇是模擬器 | Cake（I/O 延遲是算出來的 delay）、Het-Mem（整篇都是模擬）、SAECache（trace-driven simulation） |
| 🔒 **靠外部資料或不可重現** | 主結果建立在拿不到的 trace 或模型上 | Mooncake（dummy model + Kimi trace）、LMCache（公司 F/G 的 trace，只用長度分布；baseline 含商用服務） |

---

## 1. ShareGPT 與時間戳：誰跑了、跑的是真的還是假的

| 論文 | 有沒有跑 ShareGPT | 到達時間從哪來 | 備註 |
|---|---|---|---|
| **Bidaw** | 有，**但只跑 OPT-13B** | 自家 trace 是真實時間戳；ShareGPT 那組用 Poisson | 原文 §5.3：時間戳換成模擬的以後，剔除策略「**no longer reduces miss rates**」 |
| CachedAttention | 有（主負載，9K sessions，平均 5.75 輪） | Poisson。原文："no public request arrival timestamp" | |
| HCache | 有（ShareGPT4） | Session 之間 Poisson，**同一 session 的輪次固定間隔 30 s** | 原文："do not reuse the KV cache on GPU for a fair comparison" |
| Marconi | 有（另有 LMSys、SWE-Bench） | 手動改變 inter-session / inter-request 間隔 | |
| OrbitFlow | 有（抽樣） | Poisson。原文："publicly available production traces ... are limited" | |
| SAECache | 有（另有 LMSys、Chatbot-Arena） | Trace-driven simulation | Chatbot-Arena 上 TTFT 比最好的 baseline **慢 12–34%** |
| Bottlenecks | 有，但只用來算 κ_ratio 的分布 | 不適用（prefill-only 微基準） | |
| Mooncake | 沒有 | **真實 Kimi trace（23,000 條）** | 部分實驗改用 Poisson |
| LMCache | 沒有 | 公司 F/G 的 trace，但**只取長度分布**，並把好幾天的 trace 壓縮成 1 小時 | |
| 其他（KVDrive、Tutti、ArkVale、QEvict、KVP、LookaheadKV、ForesightKV、Cake、KVPR、CacheBlend、KVServe、EvicPress、AdaptCache、LeoAM、InfiniGen、Quest、ShadowKV、Het-Mem） | **沒有** | 單請求 benchmark（LongBench / RULER / LEval …）或合成負載 | |
| **Tiara（本文）** | **沒有**（`main.tex` 全文 0 次） | 真實 Mooncake trace（toolagent + conversation） | 審稿人可能會問「為什麼不跑 ShareGPT」，見 §4 |

---

## 2. 逐篇短板表

欄位說明：**測了什麼** = 模型｜資料｜硬體。**短板**用白話寫，括號內附出處。

### 2.1 多層儲存（GPU / CPU / SSD 放置）

| 論文 | 做什麼 | 測了什麼 | 短板 |
|---|---|---|---|
| **Bidaw**<br>FAST'26 | 多輪對話的 KV 放在 DRAM 和 SSD 兩層。排程會看 KV 在哪一層；剔除的依據是「上一輪回答多長」；不存 KV，改存 **tensor 6** | OPT-6.7B/13B/30B、Qwen-7B/14B（**Qwen1 世代**）｜自家業界 trace ＋ ShareGPT（只跑 OPT-13B）｜1× A800 80GB、4× SATA SSD RAID-5 1.5 GB/s | ① **剔除策略靠真實的人類互動時間戳**。ShareGPT 改用 Poisson 以後，原文說剔除策略「no longer reduces miss rates」（§5.3）<br>② **tensor 6 = normalized activation，只對 MHA 有利**。原文：「for GQA-based LLMs, KV is smaller and caching the storage-efficient tensor above is no longer beneficial」（§4）。原文引用的 Llama 是 **LLaMA-1**（參考文獻 [42]），Qwen 是 **Qwen1 技術報告**（[6]）；Llama-3、Qwen2 以後的模型都是 GQA<br>③ **上下文極短**：query 平均 36 token、回答平均 45 token；tensor 6 的分析只用了 2048-token 的歷史（圖 14）<br>④ 只在一台機器上測 |
| **FlexGen**<br>ICML'23 | 用線性規劃決定權重、KV、activation 在 GPU/CPU/disk 各放幾成 | OPT-6.7B/30B/175B｜**合成資料，所有 prompt 補成同一長度**（512/1024），生成 32 token；吞吐量用 **dummy weights**｜T4 16GB | ① 目標是離線批次吞吐量，不管延遲<br>② 配置事先算好，執行期不變<br>③ 負載是合成的，沒有重用<br>④ 只測 OPT |
| **CachedAttention**<br>ATC'24 | 多輪對話 session 的 KV 存在 HBM/DRAM/SSD，靠排程佇列預取 | LLaMA-1 65B、LLaMA-2 13B/70B、Falcon-40B｜ShareGPT 9K sessions｜4× A100 80GB | ① 時間戳是 Poisson 模擬的<br>② **唯一的對照組是「全部重算」**<br>③ 主要是 2K/4K context window 世代的模型（長上下文只補了 Mistral-7B 32K）。**但它不是只有 MHA**：原文寫 LLaMA-70B 與 Falcon-40B「using the group query attention」<br>④ 預取只看得到已經進佇列的請求（我們的判讀） |
| **Mooncake**<br>FAST'25 | 把叢集裡閒置的 DRAM/SSD 池化成共用的 KVCache 層 | 「**dummy model** that follows the same architecture as LLaMA2-70B」｜23,000 條真實 trace ＋ Poisson 模擬｜8× A800 / 節點、800 Gbps RDMA | ① 叢集等級，需要 RDMA，單機用不上<br>② 用 dummy model，**沒有品質量測**<br>③ 原文：「up to only 50% of the KVCache can be reused in our current workloads」<br>④ decode 時間假設是固定值，request-level 預測留給 future work |
| **KVDrive**<br>preprint'26 | 以 prefill 的重要性剖析決定 HBM/DRAM/SSD 的初始放置，全量 KV 寫進 SSD 當備份 | Llama-3-8B-1048K、Qwen3-8B/14B、Phi-4-mini｜LongBench、RULER｜L20、H20、RTX 4090 | ① **只測單請求長文理解**，沒有多輪或跨請求重用<br>② 初始的 HBM/DRAM/SSD 分層來自 prefill 的重要性剖析；decode 時每一步挑 top-K 做稀疏 attention（近似，原文自評「negligible accuracy loss」）<br>③ 原文：「all KV entries are first persisted to SSD as the full backing」（寫入量＝全量）<br>④ 量化列為 future work |
| **Strata**<br>OSDI'26 | SGLang 的分層快取（HiCache）。GPU 協助的 I/O 讓零碎的 page 可以整批搬；cache-aware 排程器處理 delay hit、平衡 batch；已在生產環境部署 | Llama-3.1-8B/70B、Qwen2.5-14B-1M、DeepSeek-V3｜LooGLE、NarrativeQA、ReviewMT、ShareGPT，**全部用 Poisson**（原文：「individual query timestamps are not available」）｜8× H200、H20 + NVMe 7 GB/s、GH200 | ① **放置策略就是 LRU**。原文：「For all memory layers, the Least Recently Used (LRU) algorithm serves as the default eviction policy」<br>② **不把重算當成選項**：原文認為長上下文時重算「increasingly costly ... an unattractive alternative」<br>③ 有「寫入時決定」的機制（selective-write-through：存取次數超過門檻（預設 2）才寫到下層），但依據是**次數**，不是成本<br>④ H200 平台的磁碟不到 1 GiB/s，所以 SSD 另外在 H20 上測<br>⑤ 原文自承：I/O kernel 會搶 SM；排程不保證公平；從更慢的階預取只能靠佇列延遲來藏；只支援 dense attention |
| **MTDS**<br>C&IS'26 | 邊緣裝置的多層 KV 儲存。依「預測的載入時間 vs 重算時間」選擇全載、部分載或不載；看佇列中的請求決定卸載優先序；DRAM 用兩級門檻加上滑動視窗 LRU 來逐出 | GPT-2 1.5B、LLaMa-2 7B、LLaMa-3 8B、Qwen-3 14B｜ShareGPT + Poisson（λ=1）、固定長度的合成資料｜4× A10 24GB、64 GB DRAM、2 TB SSD、PCIe Gen4 | ① **算還是載是整段一起決定**（Full / Partial / No Load），不會在同一段前綴內依位置切分<br>② 「未來命中機率」其實是**已經進佇列的請求**之間的前綴比對，看不到還沒抵達的請求<br>③ 決定發生在**讀取時**<br>④ **沒有精度階**，也沒有量品質<br>⑤ 模型偏小（含 GPT-2），時間戳是 Poisson<br>⑥ 原文 future work：跨裝置協調、結合壓縮 |
| **Tutti**<br>preprint'26 | 由 GPU 直接對 NVMe 發 I/O，繞過 CPU | 主要 Llama3-8B 單卡（擴展性實驗用 GLM-4-9B-1M 兩卡）｜LEval、LooGLE 輪流抽樣 + Poisson｜H100、PCIe 5.0 | ① **只解決 I/O 路徑**，放置和驅逐不是它的貢獻（我們的判讀；依據是貢獻列表，以及全文 evict 只出現 3 次）<br>② 依賴 GeminiFS（GPU 檔案系統）和 NVIDIA Green Contexts<br>③ 時間戳是模擬的（原文："datasets lack native timestamps"）<br>④ 主要只測一個模型 |

### 2.2 可恢復驅逐

| 論文 | 做什麼 | 測了什麼 | 短板 |
|---|---|---|---|
| **ArkVale**<br>NeurIPS'24 | 被驅逐的 page 在 CPU 留備份，用 page digest 估重要性，需要時召回 | **只有 LongChat-7b-v1.5-32k**（Llama-2 衍生，MHA）｜LongBench 6 個資料集 + passkey｜1× A100 | ① **只測一個模型**<br>② 原文自承（§7）：每個 page 都要在 CPU 備份，prefill 期間的備份延遲難以 overlap；CPU 放不下時要落到 disk（未實作）<br>③ 前兩層不套用<br>④ 單請求，沒有跨請求重用<br>⑤ Yandex 的研究指出，它的 digest 近似在 context-intensive 任務上失準 |
| **QEvict**<br>preprint'26 | 中等重要度的視窗改存 INT2，之後需要時再反量化回來 | Llama-3.1-8B、Mistral-7B、Qwen2.5-7B｜LongBench、RULER 32K、GSM8K｜A100 | ① **只有一個位置（GPU 內）**，只做精度分層<br>② 原文自承：必須定期把 attention score 算出來，用 FlashAttention-2 時會拉低 decode 吞吐量；fused kernel 留給 future work<br>③ 只測 decoder-only 模型和固定大小的視窗 |

### 2.3 學習式未來效用預測

| 論文 | 做什麼 | 測了什麼 | 短板 |
|---|---|---|---|
| **KVP**<br>ICML'26（Apple） | 每個 head 一個 RL agent，對 token 排序後驅逐 | Qwen2.5-7B、Phi-4｜RULER（到 128K）、OASST2-4k、零樣本 BoolQ 等｜訓練：8× H100（<30 分鐘）；微基準：B200 | ① 原文自承：「lacks an optimized end-to-end inference pipeline, and we therefore report policy-level wall-clock costs rather than **full-system latency**」<br>② 每個 head 和每層用同一個 budget；各 head 的 agent 彼此獨立<br>③ reward 用 future attention 當代理，不是下游任務<br>④ 動作只有保留或丟棄；換模型就要重收 trace、重新訓練 |
| **ForesightKV**<br>ICML'26 | 先建 golden eviction oracle 做監督學習，再用 GRPO 強化學習微調 | Qwen3-1.7B/4B、DeepSeek-R1-Distill-Qwen-7B（**全是推理模型**）｜**AIME2024、AIME2025**｜A800 | ① **只針對數學推理的長生成**（原文："tailored for long-context reasoning tasks"）<br>② 需要兩階段訓練（監督學習 + RL）<br>③ 動作只有保留或丟棄 |
| **LookaheadKV**<br>ICLR'26（Samsung） | 加入可學習的 lookahead token 和 LoRA，不必生成就能預測重要性 | Llama3.1-8B、Llama3.2-1B/3B、Qwen3-1.7B/4B/8B｜LongBench、RULER、MT-Bench｜實機型號未查到（H100 只出現在「we simulate the execution ... on a single NVIDIA H100」的理論估算） | ① 原文自承：「unable to conduct experiments on larger-sized models」<br>② 原文自承：「currently focuses on the **prefill** KV cache eviction」，decode 階段的驅逐留給 future work<br>③ 要改模型（加 LoRA），每個模型都要訓練 |
| **Marconi**<br>MLSys'25 | Hybrid（Attention + SSM）模型的 prefix cache，效用同時算重用機率、省下的計算、記憶體 | 自建 7B Hybrid、Jamba-1.5-Mini｜LMSys、ShareGPT、SWE-Bench｜8× A100-40GB | ① **只適用 Hybrid 模型**，純 Transformer 上就退化成一般的 prefix cache（我們的判讀）<br>② 到達間隔是手動設定的<br>③ 主要指標是 token hit rate<br>④ 沒有自己的 CPU/SSD 階（全文只在描述別的系統時提到 host memory） |
| **SAECache**<br>preprint'26 | 依 token 的語意類型（system prompt、工具輸出、推理鏈等）分成不同佇列驅逐 | 只用 Qwen2.5-1.5B 在 A40 上**驗證整合**；主實驗是 trace-driven 模擬｜ShareGPT、LMSys、Chatbot-Arena ＋ 4 個合成負載 | ① 原文自承：單一小模型時「vLLM ... **rarely fills the KV cache to capacity**」，所以改用模擬<br>② Chatbot-Arena 上 TTFT 比最好的 baseline 慢 12–34%<br>③ ⚠️ **數字自相矛盾**：摘要和圖 4 說明寫 756×，引言寫 44×，Table 1 寫 **42×**（92.3% ÷ 2.2%）。見 §4 |
| **TRIM-KV**<br>ICLR'26 | 學一個 retention gate，block 建立時評分一次，之後按指數衰減排序 | Qwen3-1.7B/4B/8B/14B、R1-Distill｜AIME24、GSM8K、MATH-500、LongProc、LongBench-V2｜訓練用 4× H100 | ① **以數學推理為主**<br>② 要訓練 gate（等於改模型）<br>③ ⚠️ `refs.bib` 標為「venue UNVERIFIED」，但 PDF 首頁寫「Published as a conference paper at **ICLR 2026**」 |

### 2.4 重算 vs 傳輸

| 論文 | 做什麼 | 測了什麼 | 短板 |
|---|---|---|---|
| **Cake**<br>ICML'25 | GPU 從前面往後算、I/O 從後面往前載，兩邊在中點會合 | LLaMA 3.1-8B/70B、Long-Alpaca-13B 等｜**合成 prompt**（4k–16k 每 2k 取一點，原文："only token length matters"）｜2× A100、1× H100 | ① **I/O 頻寬是模擬的**。原文：「We simulate the chunk I/O loading process by calculating the appropriate delay time」<br>② 原文：「we precompute and store all requests' KV cache in advance」，也就是假設 KV 已經在儲存層，不處理放置和驅逐<br>③ 原文自承：序列短時反而輸給純計算 |
| **KVPR**<br>ACL Findings'25 | 用 profiler 求出「傳 activation 再重算 KV」與「直接傳 KV」的最佳切分 | **OPT-6.7B/13B/30B**｜沿用 FlexGen 的補齊資料（256/512/1024）｜A100、PCIe 4.0 | 原文 Limitations 自承：<br>① 只支援單 GPU 或 data-parallel，不支援 TP<br>② 「we do not consider scenarios where the KV cache is loaded from **disk** or network storage」<br>③ 只在開始時 profile 一次，假設硬體條件不變 |
| **HCache**<br>EuroSys'25 | 不存 KV，改存 hidden state，要用時再由它重建 KV | Llama2-7B/13B、OPT-30B（原生 2K/4K，原文「expand the maximum context length ... to 16K」）｜ShareGPT4、L-Eval（Zipf 合成）｜4× A100-40G、4× PM9A3 SSD | ① **只適用 MHA**。原文 §7：「HCache can currently support LLMs using the **MHA** mechanisms ... Llama2, Gemma, Phi2, and Qwen1.5」；GQA 要「changing the model structure, which is beyond the scope」<br>② 為了公平比較，**關掉了 GPU 上的 KV 重用**<br>③ 同一 session 的輪次固定間隔 30 s |
| **CacheBlend**<br>EuroSys'25 Best Paper | RAG：多個已快取 chunk 的 KV 拼接起來，只挑一小部分 token 重算 | Mistral-7B、Yi-34B、Llama-70B（後兩個是 8-bit 量化）｜2WikiMQA、Musique、SAMSum、MultiNews ＋ 合成的 chunk 重用｜2× A40 | ① **只針對 RAG 的多 chunk 拼接**<br>② 選擇性重算是有損的<br>③ 原文自承：「only applies to language models with **transformer** structures」、「haven't tested ... more models and datasets with different quantization settings」、沒有測跨節點共享 |

### 2.5 品質約束、壓縮與放置最佳化

| 論文 | 做什麼 | 測了什麼 | 短板 |
|---|---|---|---|
| **KVServe**<br>SIGCOMM'26 | PD 分離時，KV 過網路前自動挑壓縮組態，並以品質門檻作為約束 | Qwen2.5-7B/32B、Llama-3.1-8B｜GSM8K、HumanEval、Multi-News、Qasper（訓練用）＋ 2WikiMQA、HotpotQA（沒見過的）｜RTX 5090、RTX 4090、Pro 6000、H100 | ① **決策變數只有壓縮組態，不含 KV 放在哪一層**<br>② 品質門檻是固定的 97% relative accuracy<br>③ 沒有 Limitations 段落 |
| **EvicPress**<br>preprint'25 | 對每個 context 聯合決定壓縮率和放在哪一層 | Llama-3.1-8B、Qwen2.5-14B、LongChat-7B、Mistral-7B、Qwen3-30B（MoE）｜LongBench 555 個 context，**每個由 GPT-5 生 100 個問題**｜1× H100 | 原文 §8 自承：<br>① 只在單節點測<br>② 假設各層頻寬「known and relatively stable」<br>③ DRAM 充足或兩層頻寬相近時，優勢會消失<br>另外：假設 remote storage 無限大；只從 3 種 token-dropping 方法中挑；在 musique 上輸給 keydiff+LRU |
| **AdaptCache**<br>BigMem workshop'25 | 同一團隊的前身：用 greedy 決定壓縮法、壓縮率和放置 | **只有 Llama-3.1-8B**｜LongBench 6 個資料集（1,100 段上下文），問題由 GPT-4o 產生，Poisson 到達｜1× A100、100 GB DRAM、400 GB SSD（1 GB/s） | ① 3 頁的 workshop 初步結果<br>② 原文：「estimate the future cache hit frequency ... using its **historical** hit frequency」，只看過去<br>③ 品質用 α 加權，不是約束；相對於 prefill，TTFT 降 56% 的代價是「within 15% quality drop」<br>④ **沒有列出實際的候選壓縮法與壓縮率**：全文只在概念上提到 token dropping 與 quantization，baseline 用 KIVI（2-bit）與 StreamingLLM |
| **LeoAM**<br>preprint'25 | 單張消費卡上做長上下文：KV 分塊放 GPU/CPU/SSD，並做 INT4 壓縮 | LongChat-7B-32k、Yarn-Llama-2-13B-128k、OPT-6.7B｜**準確度用 COPA/RTE/PIQA/OpenBookQA**（短 few-shot）+ PG-19；速度用 LongBench｜RTX 4090 | ① ⚠️ **「less than a 1% accuracy drop」是在短 few-shot 任務上量的，不是長上下文任務**（`main.tex` §3 引用了這個數字）<br>② 模型都是 MHA 舊世代<br>③ 不同記憶體容量是用限制比例來「emulate」的 |
| **OrbitFlow**<br>VLDB'26 | 用 ILP 在延遲 SLO 和容量限制下決定每層的 KV 放哪 | LLaMA3-8B（1× RTX A5000 24GB、PCIe 3.0）、LLaMA3-70B（4× A6000）｜**從 ShareGPT 抽樣 + Poisson**｜batch 4、每批 ≤ 32K | ① 無損設計（原文：「KV offloading does not compromise accuracy」），ILP 裡沒有品質變數<br>② 只有 GPU 和 CPU 兩層（全文只在描述別人時提到 SSD）<br>③ 原文："publicly available production traces for long-context serving are limited"，所以負載是合成的 |
| **Het-Mem**<br>IEEE CAL'25 | 把 HBM + off-package DRAM 的放置問題形式化，用 simulated annealing 求上界 | LLaMA-3.1-8B｜**一種負載**：NarrativeQA 約 30K prompt + 10K decode｜**GH200 的模擬參數，HBM 刻意設成 24 GB** | ① **整篇都是模擬**，沒有實機量測<br>② 原文：「Rather than proposing a specific scheduling policy」，沒有提出策略<br>③ 上界用到未來的存取資訊<br>④ 只有一個模型、一種負載，也沒有 SSD |

### 2.6 以 query 為訊號的稀疏檢索

| 論文 | 做什麼 | 測了什麼 | 短板 |
|---|---|---|---|
| **InfiniGen**<br>OSDI'24 | 用前一層的輸入預演 attention，只把重要的 KV 從 CPU 預取回來 | **OPT-6.7B/13B/30B、Llama-2-7B/13B**｜**準確度用 COPA/OpenBookQA/WinoGrande/PIQA/RTE** + WikiText-2/PTB；速度用 PG-19｜RTX A6000、PCIe 3.0 | ① **準確度在短任務上量**（敏感度實驗的輸入只有 1,920 token）<br>② 要**離線修改模型權重**（原文："skewing the ... query and key matrices"）<br>③ 模型是舊世代<br>④ Yandex 的研究指出，它在 context-intensive 任務上會漏掉需要的 key |
| **Quest**<br>ICML'24 | 用每個 page 的 min/max key 估重要性上界，取 top-k 做 attention | LongChat-7B-32k、Yarn-Llama-2-7B-128k｜PG19、passkey、LongBench 6 個資料集｜RTX 4090（kernel） | ① **只省「讀取量」，不省「容量」**：原文："reducing the memory movement"，全文沒有提到 CPU，KV 全部仍放在 GPU<br>② 前兩層不能用<br>③ 模型是舊世代 |
| **ShadowKV**<br>ICML'25 | key 做低秩壓縮留在 GPU，value 卸載到 CPU，執行時做稀疏選取 | Llama-3-8B-1M、GLM-4-9B-1M、Llama-3.1-8B、Yi-9B-200K（另有 Phi-3、Qwen2）｜RULER、LongBench、NIAH｜A100 | ① Yandex 的研究指出：rank 160 的 SVD「was sufficient for low context-intensity tasks they evaluate on」，但在 context-intensive 任務上「can no longer reliably select」<br>② 只有 GPU 和 CPU 兩層（全文 0 次提及 SSD 或 disk）<br>③ 單請求 |

### 2.7 服務層與量測研究

| 論文 | 做什麼 | 測了什麼 | 短板 |
|---|---|---|---|
| **LMCache**<br>preprint'25 | 企業級的 KV 快取層（CPU/disk/遠端後端、PD 傳輸） | Llama-3.1-8B/70B、Qwen2.5-72B、Qwen2.5-Coder-32B、Qwen3-Coder-480B｜合成的多輪文件 QA（10K doc）、TriviaQA + Poisson、公司 F/G 的 trace｜8× H100（雲端） | ① 真實 trace **只用了輸入和輸出的長度分布**，而且「stretch the original trace which lasts for several days」壓縮成 1 小時，重用的時間結構因此變形（我們的判讀）<br>② 放置策略不是它的貢獻（它提供 API）<br>③ 部分 baseline 是商用服務（原文：「accessed on September 10th」），不可重現 |
| **Bottlenecks**<br>MLSys'26 | 分析 KV 卸載什麼時候會變成 memory-bound；**定義 κ_crit** | Llama-3.1-70B、Qwen3-235B-A22B｜ShareGPT、NarrativeQA、FinQA（用來統計 κ_ratio）｜H100、B200 | ① 只做 prefill（輸出限 1 token）<br>② prompt 是用重複片段拼出來的<br>③ MLA 模型量不出來，原文："defer comprehensive MLA evaluation to future work"<br>④ 「99% 延遲在傳輸」的前提是容量壓力存在<br>⚠️ **它的 κ_crit 與我們的 κ 是同一個量**，見 §4 |
| **Yandex / YAKV**<br>preprint'26 | 在 context-intensive 任務上重新檢驗 KV 卸載方法 | Llama-3.1-8B、Qwen3-4B/30B/32B 等｜自建的 context-intensive 評測集 + NIAH、RULER、LongBench 等｜H200、A100、B200 | 原文自承：YAKV 是「minimal working KV offloading system」，沒有 prefetch 和 adaptive budget；「limited to models trained with full attention」（不含 MLA 和 Gated DeltaNet） |
| **KIVI**<br>ICML'24 | 2-bit 非對稱量化：K 沿 channel 量化、V 沿 token 量化 | Llama-2-7B/13B、Falcon-7B、Mistral-7B、LongChat｜GSM8K、LongBench、NIAH｜A100 | ① 最近的 token 要保留全精度的 residual 區<br>② 只有 GPU 一個位置 |
| **KVTuner**<br>ICML'25 | 依各層敏感度做混合精度 KV 量化 | Llama-3.1-8B、Qwen2.5-3B~32B、Mistral-7B｜以 GSM8K 等數學題為主 | ① 每個模型要離線搜尋一次<br>② 近無損的位元數隨模型而變（Llama-3.1-8B 3.25-bit、Qwen2.5-7B 4.0-bit）<br>③ 評測偏重數學 |

---

## 3. 我們可以打的共同缺口（我們的判讀，依據是上面的表）

1. **沒有人同時用「真實時間戳」和「目前主流的 GQA 模型」。** 有真實時間戳的（Bidaw、Mooncake）用的是 OPT/Qwen1 或 dummy model；用 GQA 新模型的（KVDrive、ShadowKV、KVP、LookaheadKV…）都是單請求 benchmark，沒有到達時間。
2. **「省空間的中間表示」這條路在 GQA 上是死路。** Bidaw 的 tensor 6 和 HCache 的 hidden state 都只在 MHA 上划算，兩篇原文都承認這一點。
3. **品質證據大多在錯的任務上。** 做長上下文放置的系統，有一半以上的準確度是在短 few-shot 任務或 NIAH 上量的；Yandex 已經證明 NIAH 會高估。
4. **學習式驅逐都不是端到端系統。** KVP 自承沒有端到端延遲；LookaheadKV 只做 prefill；ForesightKV 只做數學推理。

---

## 4. ⚠️ 讀原文時抓到、會打到我們自己的四件事

以下是**待處理的問題**，不是結論。依 CLAUDE.md §1-5，`main.tex` 的修改另開 paper-writing session 處理。

| # | 問題 | 證據 | 影響 |
|---|---|---|---|
| 1 | **Bottlenecks 已經定義了我們的 κ** | Bottlenecks 式 (6)：κ_crit = (F_pf / B_kv) × (BW_PCIe / C_eff)，並拆成 κ_M（模型）× κ_HW（硬體），還討論了 MoE 與 MLA。我們的 κ = t_recompute / t_transfer，以 2N_params FLOP 換算，代數上是同一個量。`main.tex:226` 只引它「99% 延遲在傳輸」 | **貢獻 1「κ 跨硬體變動 32 倍」的新穎性要重新定位。** 我們仍獨有的部分：κ 隨 block 絕對位置增長、把 κ 用作決策門檻 p* = 1/(1+κ)、MI300X 與 3090 的實測。建議在 §2 正面引用它並劃清差異 |
| 2 | **SAECache 的「756×」站不住** | 它的 Table 1 寫 92.3% ÷ 2.2% = **42×**，引言寫 44×，只有摘要和圖 4 說明寫 756× | `main.tex:957` 和 `PRIMER.md`（3 處）引用 756 倍，應改成 42×。另外 `main.tex:957` 寫「極冷的工具輸出」，但它的 Table 1 裡工具輸出是 23.0%，最冷的是**推理鏈（2.2%）** |
| 3 | **HCache 只對 MHA 有利** | 見 §2.4 | `main.tex:555` 用 HCache 論證「hidden state 可單步重建 KV，從而切斷依賴鏈」。但我們用的兩個模型都是 GQA。依模型 config 計算（非量測；數值取自本 repo 的 `results/m1_capacity/model_configs.json`）：Qwen2.5-7B 每層每 token 的 hidden state 是 3,584 維，KV 是 2×4×128 = 1,024 維，**hidden state 反而是 KV 的 3.5 倍**；Llama-3.1-8B 是 4,096 對 2,048，**2 倍**。這條論證要加上「僅限 MHA」 |
| 4 | **LeoAM 的「< 1%」是短任務上的數字** | 見 §2.5 | `main.tex` §3 說它是「唯一給出數值上界者」。可以保留，但應註明是在短任務上量的 |

其他小事：
- TRIM-KV 的 venue 可以從「UNVERIFIED」更新為 ICLR 2026（PDF 首頁為證）。
- 我們自己也沒跑 ShareGPT。可以預先準備理由：它沒有時間戳，而 Bidaw §5.3 正好示範了「用 Poisson 補時間戳會讓時間相關的策略失效」。

---

## 5. 這份表的可信度

**逐字查過原文的**：所有「原文：」「原文自承」引號內的句子、各篇的模型／資料／硬體清單、SAECache 的 42×／44×／756× 三處矛盾、Bottlenecks 的 κ_crit 公式、HCache 與 Bidaw 的 MHA 限定句。

**我們的推論（表內已標「我們的判讀」）**：
- CachedAttention 的預取只看得到已進佇列的請求
- Marconi 在純 Transformer 上退化成一般 prefix cache
- LMCache 把 trace 從數天壓成 1 小時會讓重用的時間結構變形
- Tutti 的貢獻不在放置和驅逐
- §3 的四個共同缺口
- 「Bottlenecks 的 κ_crit 與本文的 κ 是同一個量」：這是把兩邊公式展開後的代數比對（本文 `main.tex:266` 用 2N_params FLOP/token 除以 FLOPS，對上 KV 大小除以鏈路頻寬；Bottlenecks 式 (6) 是 F_pf/B_kv × BW/C_eff，其中 F_pf = 2 × params）。本文額外有的是「κ 隨 block 絕對位置增長」與把 κ 用作決策門檻

**第一版寫錯、已更正的（2026-09-23 同日）**：
1. Bidaw 引用的 Llama 寫成 Llama-2 → 實為 LLaMA-1（參考文獻 [42]）
2. CachedAttention 被列入「只用 MHA 舊模型」 → 原文寫 LLaMA-70B、Falcon-40B 使用 GQA，已移除
3. KVDrive 寫成「重要性只在 prefill 量一次」 → prefill 只決定初始分層，decode 每步重選 top-K
4. LookaheadKV 的硬體寫成 H100 → 那是理論估算用的模擬設定，實機型號未查到
5. KVP 的硬體寫成 H100 → 訓練 8× H100、微基準 B200

**沒讀到的**：無（Strata、MTDS 已補讀）。各篇的附錄只用關鍵字掃過，沒有逐頁讀完。
