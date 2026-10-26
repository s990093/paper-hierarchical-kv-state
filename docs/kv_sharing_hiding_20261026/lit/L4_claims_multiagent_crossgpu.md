# L4：筆記宣稱查證＋多 agent／RAG 共享 context 的取回重疊＋跨 GPU 共享 KV＋GPU 閒置填補

- **日期**：2026-10-26
- **執行者**：文獻 agent。**這是 self-review，不是 cross-model review。**
- **查了什麼**：
  1. 使用者貼的聊天機器人筆記裡的 4 個事實宣稱（vLLM APC、LMCache、SwiftCache、ConServe），外加 1 個隱含宣稱。
  2. 多 agent／RAG 共享 context、而且把取回和計算重疊的系統：RAGCache、TeleRAG、KVCOMM、Cache-Craft，另補 2025–2026 新作：TokenDance、PCR、CacheSmith、CacheWeaver 等；KVFlow、Strata 只補 repo 沒寫到的排程細節。
  3. 跨 GPU 共享 KV：TokenLake、MemServe、Mooncake Store、Infinite-LLM、Harvest（只補新事實）、Aqua、LMCache P2P、llm-d P2P。
  4. GPU 閒置填補：HyGen、Echo、Valve。只記它們有沒有處理 KV 的還原或共享。
  5. vLLM 上游的 issue／PR：同一段共享前綴正在載入時，其他請求怎麼辦。
- **證據等級**：
  - 〔全文〕：本文主體逐頁讀過。
  - 〔部分〕：讀過標明的節或頁。
  - 〔摘要〕：只讀 arXiv abs 頁或搜尋摘錄。
  - 〔文件〕：官方文件或官方部落格。
  - 〔程式碼〕：原始碼。
  - 〔PR〕：GitHub issue 或 PR。
  - 〔判讀〕：我的推論，原文沒有直接寫。
- **頁碼**：一律用 **PDF 頁**（從 PDF 第 1 頁起算），不是期刊頁碼。
- **原始檔**：下載的 PDF 和抽出的文字放在 scratchpad（不進 repo）：`/private/tmp/claude-501/-Users-hungwei-Desktop-Proj-paper-hierarchical-kv-state--claude-worktrees-hidden-latency-kv-cache-330751/6c0adbfb-479b-42f7-986a-069d8cf5ea05/scratchpad/L4/pdfs/<名稱>/`。
- **可信度提醒**：WebFetch 讀網頁時，內容先經工具內的小模型整理。所以凡是只靠 WebFetch 得到的數字（llm-d 部落格、LMCache 部落格、vLLM 文件、各篇 abs 頁）都比本機 PDF 抽字的可信度低一級。llm-d 的關鍵數字我抓了兩次，兩次一致，也和 repo 卡片一致。
- **repo 已有、本文不重做**：
  - `cards/D8_SwiftCache.md`、`cards/D5_DroidSpeak.md`、`cards/D8_llmd_P2P.md`
  - `stage3_20261019/lit/L3_reuse_serving_agentic.md`：§1 E 的 APC、CacheBlend；§4 H 的 InferCept、Continuum、KVFlow、Tokencake
  - `stage3_design_20261024/mechanisms/03_bandwidth_compute.md`：DualPath、Harvest 摘要
  - `stage3_design_20261024/lit/L8b_contention_aware_restore.md`：ConServe 一行
  - `research_20260924/novelty_sota.md`：RAGCache 的 PGDSF、MemServe 的式 (2)

---

## §0 結論

1. **筆記的 4 個宣稱都成立，但有兩處需要補正。**
   - ConServe 的 PMLR v306 版是 **ICML 2026**，但首版 arXiv 是 2024-10，當時標題不同、作者 7 人。
   - SwiftCache 的「共享」是**借記憶體空間和 NVLink 頻寬**：只有主模型的 KV 存在別的模型的空閒區塊裡，**沒有讓兩個模型讀同一份 KV 內容**。
   - 見 §1。
2. **「共享＋取回只做一次＋藏在別人的計算後面＋懂搬運的排程」這個組合，已經有論文做了。** 最接近的不是筆記提到的 ConServe 或 SwiftCache，而是 **Strata** 和 **KVFlow**。
   - **Strata**〔部分〕：
     - delay hit：同一段 context 正在算的時候，其他請求延到下一輪，排在佇列最前面。
     - bundle hit：共用同一段 context 的請求放進同一個 batch。
     - balanced batch：把要載入的請求配上足夠的計算量，讓載入被藏住。
     - bubble filling：載入期間改跑 decode batch。
   - **KVFlow**〔部分〕：
     - 每個 KV 節點記一個狀態（在 GPU、在 CPU、載入中、卸載中）。
     - 節點在載入中的請求先跳過，避免重複載入，改跑已就緒的 agent。
     - 依 agent 步驟圖預取，讓搬運藏在別的 agent 的計算後面。
   - LMCache〔部分〕、Cache-Craft〔部分〕、PCR〔部分〕也各有一部分：讓請求回佇列、或趁排隊時預取。
   - **威脅等級：高**（如果新提案只主張這個組合本身）。
3. **工程上還沒解決的空缺**：很多請求同時在等**同一份正在載入的共享 KV** 時，每個等待者該怎麼辦。
   - vLLM 上游在 2026 年連開了 4 個 issue／PR，都卡在這裡：#29087 避免重複載入（已合併）→ #44294 結果所有請求排成一列等，TTFT 膨脹到 12 倍 → #44295 用一個「GPU 有沒有壓力」的旗標做啟發式修正（未合併）→ #44882 要求「一次載入、多個請求共用」的核心機制（仍開著）。
   - 我**沒找到**任何論文把「每個等待者該等、該重算、還是先跑別的」寫成依藏延遲窗口與 κ 決定的規則。查不到不等於沒有。
4. **跨 GPU 借空閒 HBM 已經很擠**：SwiftCache、Harvest、Aqua、Infinite-LLM、TokenLake、AlignedServe。
   - **沒找到**的是：共享 KV 由一張卡取回一次，再經 NVLink 扇出（multicast）給多張卡上的多個請求，而且排程知道扇出的時間。〔查不到≠沒有〕
5. **GPU 閒置填補**這條線裡，只有 ConServe 處理 KV 還原：它增量 checkpoint 離線請求的 KV，GPU 有空時在背景把 KV 預取回來，和計算完全重疊。HyGen、Echo、Valve 只有 prefix 共享或回收記憶體，**沒有共享 KV 的取回重疊**。

---

## §1 任務 1：筆記宣稱查證表

| # | 筆記宣稱 | 判定 | 原文位置與證據 |
|:--|:--|:--|:--|
| a | vLLM 已經支援 Automatic Prefix Caching | **正確** | 官方設計文件〈Automatic Prefix Caching〉（docs.vllm.ai/en/latest/design/prefix_caching.html）〔文件〕：每個完整 block 的 hash 由「父 block hash＋本 block 的 token＋額外鍵（LoRA ID、多模態 hash、cache_salt）」組成；只快取完整 block；預設 sha256（v0.11 起）；以 reference count 讓多個請求共用 block；LRU 淘汰時請求的最後一個 block 最先被丟。功能頁（features/automatic_prefix_caching）〔文件〕：只加速 prefill，不加速 decode。V1 預設開啟：repo 已由 0.19.1 原始碼確認（`config/cache.py:66-68`，見 L3 §E1）〔程式碼，沿用 repo〕 |
| b | LMCache 支援跨請求、跨引擎的 KV Cache 共用 | **正確**（「跨引擎」的意思要說清楚） | 技術報告 arXiv 2510.09665v2〔部分〕：摘要說它把 vLLM、SGLang 產生的 KV 搬出 GPU，在引擎之間、查詢之間共用（PDF p.1）；圖 2 把「跨查詢重用」（context caching）和「跨引擎傳遞」（P/D 分離）分成兩種情境（p.3）；§7 有 P2P 共享：本地未命中時，經控制器的 `batched_p2p_lookup` 找出哪個 peer 有哪些 chunk，再去抓（p.10）。官方 quickstart〈Share KV cache across multiple LLMs〉〔文件〕：範例是**同一個模型**的兩個 vLLM 實例，經 Redis 或 P2P（NIXL）共用。〔判讀〕「跨引擎」指同一模型的多個引擎實例，或 P/D 之間；**不是跨模型**，也沒有宣稱 vLLM↔SGLang 之間互讀（NOT_VERIFIED） |
| c | 2026 年的 SwiftCache（arXiv 2606.16135）探討不同模型之間共享閒置 GPU 記憶體與 NVLink 頻寬 | **正確**；repo 卡片的描述也正確 | 標題〈SwiftCache: Efficient LLM Serving for Multi-turn Conversations with Heterogeneous KV Cache Sharing〉；第一作者 Jianmin Hu，通訊作者 Minxian Xu，共 8 人（SIAT／中科院、阿里巴巴、澳門大學）；arXiv v1 2026-06-15；abs 頁沒寫 venue，程式放在匿名連結（p.13）→ venue NOT_VERIFIED〔判讀：可能在雙盲審稿中〕。〔全文〕摘要寫明讓異質模型共用同一台機器裡沒用到的 GPU 記憶體和 NVLink 頻寬（p.1）。**重要補正**：它的「共享」是**記憶體空間的借用**。KV 需求高的模型（master）把自己的 prefix cache 存進需求低的模型（worker）的空閒 KV 區塊；worker 不讀 master 的 KV。原文 §6 自稱共享的是異質模型之間的 KV 記憶體緩衝區，連架構不同的模型都可以（p.12）。細節見 §1.c |
| d | 2026 年的 ConServe 已研究利用線上推理的 GPU 閒置時間執行離線工作，使用更細粒度的排程與搶占機制控制干擾（proceedings.mlr.press/v306/qiao26i.html） | **正確**；但年份要補正，筆記也漏了 KV 部分 | PMLR v306＝**第 43 屆 ICML（2026，首爾）**：PDF p.1 頁尾〔全文〕；proceedings 頁列 pp. 101429–101443〔文件〕。標題〈ConServe: Fine-Grained GPU Harvesting for LLM Online and Offline Co-Serving〉，Y. Qiao 等 13 人（UC Berkeley、UCLA、Intel、UC Davis、Rice）。三個機制：依 SLO 的 token 級排程、**iteration 內逐層搶占**、**增量 KV checkpoint**（p.1、§4）。arXiv 2410.01228 是同一條演進線：v1（2024-10-02）標題不同、作者 7 人；v2（2025-09-03）換成現標題、13 人，摘要數字和 ICML 版相同（2.2 倍吞吐、尾延遲 2.9 倍）〔摘要〕。細節見 §1.d |
| e | 隱含：「單純實作 KV 共用已經不是新研究」 | **成立** | 生產系統：vLLM APC（a）、SGLang RadixAttention（repo L3 §E2）、LMCache（b）、Mooncake Store（§4）。論文：RAGCache、Cache-Craft（SIGMOD'25）、KVCOMM（NeurIPS'25）、TokenDance（2026）、CacheBlend（EuroSys'25，repo）。**另一個隱含宣稱不成立**：筆記把 ConServe、SwiftCache 當成這個交集最近的前作。對「共享 × 藏延遲 × 排程」來說，Strata 和 KVFlow 更近（§3.9、§7） |

### §1.c SwiftCache 細節（〔全文〕，arXiv 2606.16135v1）

- **動機量測**（p.2 圖 1）：
  - 兩張 96 GB H20、LWM-1M-Text；PCIe 4.0 峰值 32 GB/s，NVLink 400 GB/s。
  - 5K 歷史 token 時，PCIe 載入約 75 ms，計算約 27 ms。
  - 搬運占比從 73.2%（5K）升到 86.2%（50K）。
- **為什麼可以借**：
  - 不同工作負載的 prefix 命中率差很多：多輪對話 90.7%、QA 81.8%、摘要 6.3%、程式補全 0.1%（p.4 表 1）。
  - 不同模型的 GPU 之間很少直接通訊，所以 NVLink 閒著（p.4 §2.4）。
- **機制**：
  1. **Layer stream cache**：master 的本地 GPU 只留目前那一層的 KV；下一層的 KV 在上一層的 FFN 執行時，從 worker GPU 非同步載入；新產生的 KV 寫回 worker（p.5–6 §3.2–3.3）。
  2. **Elastic cache**：worker 改用 block-major layout，讓縮放是 O(1)；用 master 和 worker 每塊元素數的最小公倍數決定最小縮放單位（p.6–8 §3.4）。
  3. **縮放規則**（Alg. 1，p.8）：worker 來了放不下的請求就擴；週期性看過去時間窗 t（例如 1 分鐘）內最長的請求，決定要不要縮。
  4. **排程**：master 和 worker 各自 FCFS、iteration 級；請求依 KV 需求分派給 master 或 worker（p.5 §3.1）。
- **實作**（p.9 §4）：
  - 約 8.6K 行 Python／Triton，加約 1.8K 行 C++。
  - 控制訊息走 ZeroMQ，KV 經 `torch.distributed` 跨卡傳。
  - 支援 Qwen3、LLaMA2。
- **評估設定**（p.9 §5）：
  - 4×H20 96 GB；NVLink 雙向 400 GB/s；PCIe 4.0 共 32 GB/s，四張卡共用；主機記憶體 128 GB。
  - 對照 vLLM 0.14.0＋LMCache 0.3.15、SGLang 0.5.3 HiCache；兩個對照組都配 96 GB CPU 卸載空間。
  - ShareGPT、L-Eval，Poisson 到達。
- **關鍵數字**：
  - P99 TTFT：對 SGLang HiCache 約少 67%（ShareGPT）、69%（L-Eval）；對 vLLM＋LMCache 約少 54%、58%（p.10 §5.1）。
  - 對 worker 的干擾：TTFT 最多 +9.7%、TPOT 最多 +6.5%（p.10 §5.2）。
  - 最大 context：對 vLLM chunked prefill（8K）1.58–3.98 倍（p.11 §5.3）。
  - Qwen3-32B 的載入、存回分別占總延遲 <1.4%、<1.1%；算進重疊後 <0.1%；P99 載入 <11 ms（p.11 §5.4）。
- **限制**：
  - 只在單台機器內（跨機器沒有 NVLink），列為未來工作（p.13 §7）。
  - worker 回收被借走的區塊時，master 放在那裡的 KV 怎麼處理：**本次沒讀到**（NOT_VERIFIED）。
- **四個概念**〔判讀〕：
  - 用 NVLink 取代 PCIe ＝ **latency reduction**。
  - 逐層預取 ＝ 把自己的搬運藏在**自己的** FFN 後面（hiding）。
  - 沒有「一次取回、多人用」：只有 master 一個讀者。

### §1.d ConServe 細節（〔全文〕，ICML 2026 版 qiao26i.pdf）

- **問題量測**（Llama-3.1 8B、單張 H100）：
  - 負載 2,000 tokens/s 時，超過 70% 的 GPU 資源閒置（p.3 圖 2）。
  - 單純的可搶占排程讓 P99 TTFT、TBT 分別變成 3.59 倍、3.01 倍；離線吞吐少 4.27 倍（p.3）。
  - iteration 級搶占可讓線上請求延遲最多 970 ms（p.2）。
  - 記憶體壓力下最多 69% 的 GPU 時間花在重算或 swap（p.3）。
- **機制**：
  1. **延遲模型**：Latency(P,C)＝k1·P＋k2·P(P+C)＋k3·P＋k4(P+C)＋k5，P 是要算的 token 數、C 是已在 KV 裡的 context 長度。離線擬合，20 分鐘內完成，誤差 <4%。每個 iteration 先放線上請求，再用 `can_schedule` 塞離線 token，直到 TBT 預算用完（p.4–5 §4.2、Alg. 1）。
  2. **逐層搶占**：層與層之間插 safepoint；一次檢查約 20 µs（讀 pinned 主機記憶體 7 µs＋NVLink broadcast 13 µs）；實際每 4 層一個（p.5 §4.3；p.9）。
  3. **增量 KV checkpoint**：每個 iteration 後非同步把新 token 的 KV 存到主機；搶占時只丟 GPU 上的副本。GPU 有空時在背景**增量預取**回來，論文說和計算完全重疊、不在關鍵路徑上（p.5 圖 5(d)；p.6 §4.4）。
     - 例：Qwen-2.5-14B 每 token 192 KB，在 64 GB/s 下超過 349K tokens/s；一個 2,048-token chunk 約 10 ms（p.6）。
- **評估設定**（p.6 §6.1）：
  - DGX 8×H100，NVLink 900 GB/s。
  - 模型：Llama-3.1 8B、Qwen-2.5 14B（各 1 張卡）、Llama-3.1 70B（TP4）。
  - 線上用 BurstGPT；離線用 DuReader、MultiNews、VCSUM。
- **關鍵數字**：
  - 合成負載：P99 TTFT、TBT 分別在最佳值的 25%、19% 以內；平均達最佳離線吞吐的 82.3%（p.7 §6.2）。
  - 真實負載：對最好的基線，P99 TTFT 平均少 2.04 倍、TBT 少 2.86 倍、離線吞吐多 2.23 倍（p.14 附錄 表 2）。
  - KV 傳輸 kernel 37 GB/s，最大的一次傳輸 7 ms，可完全重疊；70B TP4 合計 132 GB/s（p.9 §6.5）。
- **和本題的關係**〔判讀〕：
  - ConServe 是筆記裡**唯一真的有「KV 還原藏在計算後面」**的。但對象是被搶占的**離線請求自己的 KV**：沒有共享、沒有多個讀者，排程也沒有把「還原要多久」當成輸入，因為它假設背景預取一定藏得住。
  - 原文 §7 說同期的 HyGen 只做延遲預測和 iteration 級的 prefix 共享（p.9）。

---

## §2 四個概念的對照（後面各表用的尺）

| 概念 | 定義（本文用法） | 例子（出處見各節） |
|:--|:--|:--|
| latency hiding | 搬運照樣發生，但和**有用的計算**同時跑，不在關鍵路徑上。要分清楚藏在**自己的**計算後面，還是**別人的**計算或排隊時間後面 | 自己：LMCache 逐層管線、Cache-Craft、SwiftCache、PCR 逐層。別人：LMCache 請求回佇列、KVFlow status-aware、Strata bubble filling、Cache-Craft／PCR 排隊時預取、ConServe 背景預取 |
| latency reduction | 讓搬運本身變快 | NVLink 取代 PCIe（SwiftCache、Harvest、Aqua）、批次 memcpy（PCR 的 `cudaMemcpyBatchAsync`）、Strata 的 GPU 協助 I/O |
| work avoidance | 不重算已經算過的 | APC、LMCache、RAGCache、Cache-Craft、KVCOMM、TokenDance |
| speculation | 依預測提早做，可能白做 | RAGCache 推測式生成、TeleRAG lookahead 預取、KVFlow 依步驟圖預取、CacheScout 學習式預取 |

---

## §3 任務 2：多 agent／RAG 共享 context，取回與計算重疊

### 3.0 總表

| 系統 | 讀到 | (1) 共享什麼 | (2) 取回做一次、給多人？ | (3) 藏在誰的計算後面 | (4) 排程規則 |
|:--|:--|:--|:--|:--|:--|
| RAGCache | 〔部分〕 | 文件的 KV，依文件順序組成前綴（knowledge tree） | GPU 上的節點可以讓多個請求沿重疊路徑共用（p.5）。host→GPU 的取回沒寫去重（NOT_FOUND） | 藏的是**檢索**：同一請求的 CPU 向量搜尋和它自己的推測式 prefill 重疊。KV 搬運**沒有重疊**，直接算進命中延遲（p.4） | OrderPriority＝已快取長度÷要算的長度；有窗口防餓死。推測只在文件改變、而且待處理請求少於 max_prefill_bs 時啟動 |
| TeleRAG | 〔部分〕 | **不是 KV**：共享的是 IVF 索引叢集 | **有**：把相似的查詢分成 micro-batch，讓預取的叢集盡量重疊；GPU 上留常用叢集 | 同一查詢的 pre-retrieval LLM 生成 | 預取量＝B_link×平均 t_LLM；依快取重疊度把 micro-batch 派給 GPU |
| KVCOMM | 〔部分〕 | 同一段文字在不同前綴下（非前綴）；同一個模型 | base KV 在共享記憶體只存一份，多個 agent 修正後共用 | **沒有**。原文承認卸到 CPU 時搬運 1,260+ ms／agent，說管線化和本方法正交 | 無（框架層的 anchor 比對） |
| Cache-Craft | 〔部分〕 | RAG chunk 的 KV，任意位置（非前綴）＋部分重算 | 沒寫 | 自己：逐層預載。別人：**請求還在排隊時就非同步預載** | 預載深度 L_p＝(L−1)(1−T_prefill/T_load)+1；輪到執行時還沒載好，就從頭 prefill |
| TokenDance | 〔部分〕 | 多 agent 每一輪 All-Gather 的共同輸出塊（非前綴） | **重用的計算**每輪只做一次；Master 的傳輸是否每個 Mirror 各載一次：〔判讀〕看 Alg. 1 是各載一次 | 自己：LMCache 逐層 ping-pong 傳輸路徑 | vLLM V1 adapter 每個排程步把同一輪的請求分組 |
| PCR | 〔部分〕 | 文件前綴（prefix tree） | 〔判讀〕chunk 進了 DRAM 就讓後面的請求直接命中；in-flight 去重沒寫 | 自己：逐層，三條 stream。別人：**佇列中的請求先預取 SSD→DRAM** | look-ahead LRU（看待處理佇列）；預取窗口 4 個請求 |
| CacheSmith | 〔部分〕 | RAG chunk 的 KV＋選擇性重算 | 沒寫 | 自己：用 WarmUp 層當計算窗口，蓋住 H2D 搬運 | 依 SLO 剩餘時間組 batch，配重算預算 |
| KVFlow | 〔部分〕 | 多 agent workflow 的共享前綴節點（radix tree） | **有**：節點狀態是「載入中」就不再發重複的載入 | **別的 agent 的計算**：status-aware 排程 | 依步驟圖的「幾步後執行」預取；載入中的請求先跳過 |
| Strata | 〔部分〕 | 長文件 context（多人反覆查同一份） | **有**：delay hit 延後到下一輪最前面；bundle hit 同 batch | **別人的**：balanced batch 把載入配上計算；bubble filling 在載入時跑 decode | loading_bound（載入÷計算）門檻 100；延後門檻 100 token |

### 3.1 RAGCache（〔部分〕：PDF p.1–9、p.10–11 的數字段）

- C. Jin, Z. Zhang, X. Jiang, F. Liu, X. Liu, X. Liu, X. Jin（北大、字節跳動），arXiv 2404.12457v2（2024-04-25）。venue：搜尋只有第三方聚合頁寫 ACM TOCS 2025 → **NOT_VERIFIED**（repo novelty_sota 也寫 TOCS，同樣沒開 ACM DL）。
- **共享什麼**：檢索到的文件的 KV，用 knowledge tree 依文件 ID 順序組成前綴樹，因為文件順序會影響 KV（p.4–5）。多個請求可以沿重疊的路徑共用節點（p.5）。
- **取回**：
  - GPU 滿了，就把優先度最低的葉節點換到主機記憶體。
  - 「只換出一次」：主機保留一份，直到整個快取都淘汰它（p.6–7）。
  - host→GPU 的 KV 搬運，原文只在 p.4 把搬運時間加進命中延遲（A10G 上命中延遲仍比完整 prefill 低最多 3.9 倍）。我**沒讀到**把這段搬運和計算重疊的設計。
- **重疊**：dynamic speculative pipelining（p.7–8 §5.3）。
  - 向量搜尋分成幾個階段；每階段把目前的 top-k 候選送給 LLM 做推測式生成；候選改變就終止前一個推測。
  - 定理 5.1（G/G/1）：階段結束時佇列是空的、而且文件改變了，就開始推測；否則繼續搜尋。
  - 動態版：只在文件改變、而且待處理請求少於 `max_prefill_bs` 時才推測（p.8–9）。
- **排程**：cache-aware reordering，OrderPriority＝已快取長度÷要算的長度；有窗口避免餓死（p.7 §5.2）。
- **評估**：
  - AWS g5.16xlarge（A10G 24 GB、PCIe 4.0 x16），192 GiB 主機快取；大模型用 2×H800（p.9）。
  - vLLM v0.3.0、Faiss IVF（1024 叢集、top-k＝2）。
  - 平均 TTFT 比 vLLM 少 1.2–4 倍，比 SGLang 少 1.1–3.5 倍（p.10）。
  - 摘要：TTFT 最多 4 倍、吞吐最多 2.1 倍（對 vLLM＋Faiss）；對 SGLang TTFT 最多 3.5 倍、吞吐最多 1.8 倍（p.2）。
  - 推測式管線最多讓 TTFT 少 1.6 倍（p.11）。
- **四個概念**〔判讀〕：work avoidance（文件 KV 重用）＋speculation（推測式生成）＋hiding（藏的是檢索，不是 KV 搬運）。

### 3.2 TeleRAG（〔部分〕：p.1–2、p.4–10）

- C.-Y. Lin, K. Kamahori, Y. Liu 等 14 人（UW 為主），arXiv 2502.20969v4（2026-05-18）。PDF p.1 頁尾印 **MLSys 2026**（第 9 屆，Bellevue）。
- **共享什麼**：IVF 向量索引的叢集資料。**不是 KV**。
- **機制**：
  - 用初始查詢 q_in 預測要用的 IVF 叢集。在 pre-retrieval 的 LLM 生成期間，以 GPU DMA 非同步預取到 GPU。
  - 檢索時 GPU 搜預取到的叢集，CPU 同時搜沒預取的，最後在 GPU 合併（p.6 §4.1）。
  - 預取量＝B_link×平均 t_LLM（在校準集上估），用固定位元組預算而不是叢集數（p.6–7）。
- **一次取回、多人用**：
  - batch 時把總預算平分給各查詢（p.6）。
  - 用貪婪的 L2 距離把相似查詢分成 micro-batch，讓預取的叢集盡量重疊（p.6 §4.2）。
  - GPU 上保留常用叢集；cache-aware 排程把 micro-batch 派給快取重疊最多的 GPU（p.6）。
- **評估**：
  - 硬體：RTX4090（PCIe4 32 GB/s）、H100（PCIe5 64 GB/s）、8×H200（p.8 表 2）。
  - 資料：wiki_dpr，索引 61 GB、4,096 叢集、nprobe 256、top-k 3（p.7–8）。
  - 摘要：單查詢平均延遲最多少 1.98 倍、batch 吞吐平均高 1.83 倍（p.1）。**但同一版本的引言寫**：RTX4090 單查詢平均 1.53 倍；batch 8、H100 吞吐平均 1.98 倍；H200 上 4 卡對 1 卡 3.8 倍（p.2）。兩處不一致，引用時要標清楚出處。
  - 預取用 PyTorch copy，會占一部分 SM，所以 LLM 延遲略升（p.10）。這是「藏延遲本身有干擾」的直接證據。
- **四個概念**〔判讀〕：speculation＋hiding（藏在自己的生成後面）＋「一次取回、多人用」（micro-batch 分組）。**這是三向交集的完整樣板，只是搬的東西不是 KV。**

### 3.3 KVCOMM（〔部分〕：p.1–5、p.8–10、p.15、p.29–30）

- H. Ye, Z. Gao, M. Ma 等 11 人（Duke、MIT、NVIDIA），arXiv 2510.12872v2（2025-11-01），comments 寫 NeurIPS 2025（PDF p.1 頁尾亦同）。
- **共享什麼**：同一段文字（檢索段落、上游 agent 的輸出）出現在不同 agent 的不同前綴之後。KV 會偏移；用一組 anchor 記錄過去觀察到的偏移，內插出新前綴下的偏移（p.1–2）。
  - 設定：有向的多 agent 圖，每個節點是**同一個** RoPE LLM checkpoint，只是角色模板不同（p.5）。
  - 原文限制只適用同質 agent（p.15 §6.1）。
- **一次取回、多人用**：base KV 存在共享記憶體，各 agent 拿去修正（p.5、p.16 詞彙表）。
- **重疊**：**沒有**。
  - 原文附錄說：anchor 全卸到 CPU 時，softmax 100+ ms；4K context 每個 agent 的卸載可達 1,260+ ms。作者認為管線化可以緩解，但和 KVCOMM 正交（p.29 §6.4.4；p.30 表 A.6）。
  - 〔判讀〕這是「共享 × 藏延遲」的空缺被作者自己點名的例子。
- **評估**：
  - 單張 H100；anchor pool 20；對照 CacheBlend（固定重算偏差最大的 20% token）（p.9）。
  - reuse rate 超過 70%（摘要）。
  - 5 個全連接 agent、每個 1K 輸入：最多 7.8 倍，TTFT 約 430→55 ms（p.1）。
  - 三個 agent 平均 prefill 加速約 6.7 倍（p.10）。

### 3.4 Cache-Craft（〔部分〕：p.1、p.7–9、p.11）

- S. Agarwal, S. Sundaresan, S. Mitra 等 9 人（Adobe Research、IIT），arXiv 2502.15734v1（2025-02-05），comments 寫 SIGMOD 2025（只看 arXiv 欄位，沒開 ACM DL）。
- **共享什麼**：RAG chunk 的 KV，可放在任意位置。
  - 存的時候拿掉 RoPE，用的時候依新位置補回（p.8 §4）。
  - 每個 chunk 存 M 個變體；依重算量評分 CFO＝CCI×(1−β′) 選變體；依 1/CFO 累加的頻率淘汰（p.7）。
- **取回**：
  - GPU／CPU／SSD 三層，依使用頻率放（p.7 §3.4.2）。
  - **逐層預載**：第 l+1 層的 chunk KV 在第 l 層計算時載入，和新 chunk 的 prefill、命中 chunk 的重算重疊。
  - 載入比計算慢時，預先載 L_p 層：L_p＝(L−1)(1−T_prefill/T_load)+1（p.7 式 16）。
- **藏在排隊時間後面**（p.8）：
  - 請求還在佇列時就非同步把 chunk KV 搬到 GPU；**輪到執行時還沒到，就從輸入 token 重新 prefill**。
  - 例：Sys-X 一個請求的 5 個 chunk，CPU 載入 0.03 s、SSD 0.59 s；典型排隊 0.32 s；負載高時排隊可以完全蓋住 SSD 載入。
- **評估**：
  - p4de.24xlarge（8×A100-80GB、1,152 GB DRAM、8 TB NVMe 讀 16 GB/s、PCIe4 x16 64 GB/s）；LLaMA-3 8B（TP1）、70B（TP4）（p.9）。
  - 摘要：比 prefix caching 少 51% 的重複計算、比完整重算少 75%；吞吐 1.6 倍、端到端延遲少 2 倍。
  - 70B、30% 重算：高負載延遲少 2.07 倍，中負載 2.26 倍；扣掉排隊時間後是 3.22 倍、2.64 倍；高負載平均排隊約 7.15 s（p.11）。
- 〔判讀〕Cache-Craft 的「沒載好就重算」是一條二選一的排程規則，但門檻只是「到時有沒有到」，沒有估「再等多久」或 κ。

### 3.5 TokenDance（〔部分〕：p.1–8、p.10–11）

- Z. Bian, F. Wu, C. Zhang, H. Dong, Y. Liang, Y. Zhuo（北大、上海交大），arXiv 2604.03143v1（2026-04-03），comments：審稿中，沒有 venue。〔判讀〕第一作者和末位作者與 Tokencake 相同。
- **共享什麼**：多 agent 每一輪 All-Gather 時，所有 agent 的 prompt 都含同一組上輪輸出塊，但各自前面的私有歷史長度不同（非前綴）。
  - 8-agent GenerativeAgents 一輪中，兩兩區塊相似度 91–97%（p.2）。
  - A100-80GB、Qwen2.5-14B：10 個多 agent session（250 個子請求）吃掉 41.5 GiB KV，占 pool 的 99.3%（p.1）。
- **一次做、多人用**：
  - 把優化單位從「請求」拉到「一輪」：RoPE 修正和重要位置選擇每輪只做一次（p.4–6 圖 7）；預設後端是 CacheBlend 的選擇性重算（p.6）。
  - 儲存用 Master-Mirror：一份完整 Master，其他存成 block-sparse 差異，每份只占完整 cache 的 10–20%（p.6）。
- **重疊**：還原時在 LMCache 的逐層 ping-pong 傳輸路徑上直接套差異，不另外組出完整的 Mirror（p.7 §4.4；p.8 §5）。
  - 〔判讀〕Alg. 1 是**每個 Mirror 請求各載一次 Master chunk**，所以傳輸本身沒有「一次給多人」。
  - 卸載、預取：除了沿用 LMCache 的 CPU pool，沒有討論。
- **評估**（p.8 §6.1；摘要；p.10–11）：
  - A100 80GB；Qwen2.5-7B／14B；GenerativeAgents、AgentSociety trace。
  - 在 SLO 下比 vLLM prefix caching 多撐最多 2.7 倍的同時 agent。
  - 每個 agent 的 KV 儲存少最多 17.5 倍；比逐請求的位置無關快取（PIC）prefill 快最多 1.9 倍。
  - fused restore 比組出完整 Mirror 快 1.3–2.6 倍（10 agent、QPS 1：0.59 vs 0.43 ms／Mirror）。

### 3.6 PCR（〔部分〕：p.1–3、p.7–8、p.11–14）

- W. Wang, X. Hou, P. Tang 等 8 人（上海交大、華東師大），arXiv 2603.23049v1（2026-03-24），頁首標「Manuscript submitted to ACM」，venue NOT_VERIFIED。
- **三個機制**：
  1. prefix tree＋**look-ahead LRU**：看等待佇列裡的請求，提高「快要用到」的 chunk 的優先度。
  2. **逐層重疊**：CPU→GPU、GPU 計算、GPU→CPU 各一條 CUDA stream。
  3. **佇列預取**：檢索比推論快，所以佇列裡的請求已經知道要哪些文件，可以先把 KV 從 SSD 預取到 DRAM（p.11 §4.4）。預取器有 look-ahead 窗口；已在 DRAM 的不動；DRAM 和 SSD 都沒有的整段重算（p.12）。窗口設 4（p.13）。
- 用 `cudaMemcpyBatchAsync`（CUDA ≥12.8）一次搬多個不連續 block：32 GB/s 下 0.261 ms（p.13）。
- **評估**：
  - 2×A6000 或 2×RTX4090；PCIe4 實測約 24 GB/s；SSD 讀約 3 GB/s，寫約 500 MB/s（p.7、p.14）。
  - 對照 vLLM、LMCache 和兩個簡化版。
  - 平均 TTFT 最多快 2.47 倍（摘要）。
- 〔判讀〕「同一 chunk 正在從 SSD 搬時，另一個請求也要它」怎麼處理，原文沒寫。

### 3.7 CacheSmith（〔部分〕：p.1–4）

- X. Ma, D. Lee, A. Shah, M. Jeon（POSTECH、Microsoft Research），arXiv 2610.18925v1（2026-10-13）。
- 把選擇性重算拆成 WarmUp 等三個階段；依 SLO 剩餘時間組 batch，決定每個請求的重算預算。
- **phase-aware 預取**：WarmUp 層先做完整 prefill、不讀快取的 context，把這段計算當窗口，蓋住 host→GPU 的 KV 搬運（p.2）。
- quality-gated SLO 達成率最多 +27.5%（p.1）。
- 〔判讀〕藏在**自己的** WarmUp 計算後面；沒有多個讀者的設計。

### 3.8 KVFlow（只補 repo L3 §H4 沒寫的排程細節；〔部分〕p.5–6 §3.3）

- 每個 KV 節點有 4 種狀態：在 GPU、備份在 CPU、載入中、卸載中。
- 排程器檢查請求需要的所有節點；有「載入中」的就**跳過，避免重複載入**，等所有節點都就緒才派出。背景載入執行緒完成後更新狀態（p.6）。
- 依 Agent Step Graph，保守地預取所有可能下一步執行的 agent；同時進行的預取數有上限（p.5）。
- 多個 agent 共用的節點，淘汰優先度取它下面各 agent 的最小值（最不該丟），所以共享節點留得最久（p.5）。
- 〔判讀〕KVFlow 已經同時有：共享前綴節點、節點狀態防重複載入、把預取藏在別的 agent 的計算後面、依「幾步後執行」排程。缺的是：
  - 窗口不夠時（原文承認目前 agent 的執行比預取短就會卡，p.5–6），沒有「改重算」的選項。
  - repo L3 已記：它假設 PCIe 比重算快很多。

### 3.9 Strata（只補 repo 卡片沒寫的排程細節；〔部分〕p.4、p.6–7）

- Z. Xie 等 8 人，arXiv 2508.18572v1（2025-08-26），comments 寫「under peer review」。repo（`RELATED_WORK_WEAKNESSES.md`、`PAPER_GUIDE.md`）記為 OSDI'26，這次沒重查。
- **delay hit**（p.4 §3.2）：
  - 借用網路領域的概念：第一次未命中還沒解決時，又有多個請求要同一份資料。
  - LLM 服務裡，多個請求要同一段 context、又被放進同一個 batch，就會重複 prefill。
  - 非同步排程器先準備下一個 batch，讓這個窗口拉長到整個 batch 的執行時間。
- **處理**（p.6 §4.3.1）：
  - 在 HiRadixTree 加「暫態節點」，標成 in-queue（有請求引用了新 context）或 in-flight（對應的 cache 正在計算）。
  - 符合暫態節點的請求延到下一輪，但放在佇列最前面；只有命中暫態節點超過 100 個 token 才延。
  - 原文另列了第二種情境，但抽字在換頁處斷掉 → NOT_VERIFIED。
- **balanced batch＋bundle hit**（p.6–7 §4.3.2、Alg. 1）：
  - 組 batch 時，算總載入量÷總計算量；超過門檻（預設 100，依硬體與模型 profile）就把請求移到低優先清單。
  - 加進一個請求後，優先把和它**共用同一段 context** 的請求也加進來（bundle hit）。原文說這樣省 GPU 記憶體和裝置頻寬。
- **bubble filling**（p.7 §4.3.3）：某請求要載入長 context 時，先不跑準備好的 prefill batch，改跑 decode batch。理由：decode 吃 HBM 頻寬，載入吃 PCIe，互相干擾小。
- **評估**（p.7 §5.1）：
  - 8×H200（PCIe5 64 GB/s）與 GH200。
  - 對照 vLLM v0.8.5＋LMCache v0.2.1、TensorRT-LLM v0.17.0、SGLang v0.4.5。
  - Llama-3.1-8B、Qwen2.5-14B-1M（單卡）、Llama-3.1-70B（TP4）；LooGLE、NarrativeQA、ReviewMT、ShareGPT；Poisson 到達。
  - 摘要：TTFT 比 vLLM＋LMCache 少最多 5 倍，比 TensorRT-LLM 快 3.75 倍。
- 〔判讀〕這是目前讀到**最完整的三向交集**。repo 已記它的兩個弱點：淘汰只用 LRU；不把重算當選項。本次再補一個：delay hit 的延後是固定門檻，「該等還是該算」只看命中 token 數，**不看等的人要等多久、窗口裡有多少別人的計算**。

### 3.10 只讀摘要或 HTML 的 2025–2026 新作

| 論文 | arXiv（abs 頁已開） | 讀到 | 和本題的關係 |
|:--|:--|:--|:--|
| K. Tan, R. Gu, M. Li, "CacheWeaver: Cache-Aware Evidence Ordering for Efficient Grounded RAG Inference" | 2606.19667v2（2026-09-04） | 〔部分〕（HTML §3.4，經 WebFetch） | 重排檢索證據來提高前綴命中；**次要優化**：檢索器在跑時，同時預熱共享的 system prompt，原文說主要幫冷啟動和被淘汰後的恢復，沒有單獨量測。RTX4060Ti／4090／4090D，vLLM 0.8.5.post1、0.17.1。中位 TTFT 少約 20–33%（摘要） |
| J. Wang 等 13 人, "From Prefix Cache to Fusion RAG Cache"（FusionRAG） | 2601.12904（2026-01-19） | 〔摘要〕 | 離線把相關 chunk 的資訊嵌進每個 chunk；線上重算 <15% token；TTFT 比完整 attention 少 2.66–9.39 倍。摘要沒提重疊 |
| S. Wang, R. Ren, L. Gui, "ForkKV: Scaling Multi-LoRA Agent Serving via Copy-on-Write Disaggregated KV Cache" | 2604.06370（2026-04-07） | 〔摘要〕 | 多 LoRA agent：共享的 base KV＋各自的小殘差，fork＋copy-on-write；吞吐最多 3.0 倍。摘要沒提搬運、預取 |
| S. Woo 等 12 人, "PrefillShare: A Shared Prefill Module for KV Reuse in Multi-LLM Disaggregated Serving" | 2602.12029v2（2026-09-26） | 〔摘要〕 | 凍結的共享 prefill 模組＋各任務的 decoder，讓**不同的微調模型**共用同一份 KV；p95 延遲少 4.5 倍、吞吐 2 倍。摘要沒提重疊 |
| R. Zhang, C. Kim, L. Hu, "A Policy-Driven Runtime Layer for Agentic LLM Serving" | 2605.27744v2（2026-07-30） | 〔摘要〕 | 以 CacheScout 為例：線上學 agent 轉移矩陣，做淘汰與**步驟之間的預取**；命中率 +13–37 個百分點、TTFT −12–29%。搜尋結果把它叫「CacheSage」，但 abs 頁沒有這個名字。repo L3 §H10 的 2608.14624 也是 CacheScout |
| F. Bai 等 6 人, "AlignedServe: Orchestrating Prefix-aware Batching …" | 2605.23389（2026-05-22） | 〔摘要〕 | 「GPU-Prefetch-For-GPU」：一張卡替另一張卡預取 KV；這裡的「prefix-aware」指 KV 長度相近，**不是共享前綴**。decode 吞吐最多 1.98 倍。見 §4 |

---

## §4 任務 3：跨 GPU 共享 KV（用別張卡的閒置 HBM）

### 4.0 總表

| 系統 | 讀到 | 共享粒度 | 傳輸路徑 | 排程有沒有考慮傳輸延遲 | 能否和計算重疊 |
|:--|:--|:--|:--|:--|:--|
| SwiftCache | 〔全文〕 | 16-token block（vLLM 式），借別的模型的 KV 區塊 | master↔worker GPU，NVLink（torch.distributed） | 沒有。依 KV 需求分派、FCFS；彈性縮放看過去時間窗的最長請求 | 有：逐層，下一層的載入藏在上一層 FFN 後面（自己的計算） |
| Harvest | 〔部分〕 | 物件（KV 項目、MoE 專家權重） | 同一台機器的 peer GPU HBM，NVLink；權威副本在主機 DRAM | 沒有（原文自承不感知拓撲、不感知同機其他負載） | 沒評估；KV 部分只量搬運延遲 |
| Aqua | 〔摘要〕 | KV、LoRA 狀態（AQUA Tensors） | 鄰卡 HBM，NVLink／NVSwitch | NOT_VERIFIED（摘要：為搶占式、分時推論做分頁） | NOT_VERIFIED |
| Infinite-LLM | 〔部分〕 | 依 token 切的 KV（rManager 管 block） | 實例之間；**傳 query、不傳 KV**（DistAttention） | gManager 決定借貸關係 | 有：query 傳輸和 MicroAttention 重疊；KV 傳到債主時和推論重疊 |
| TokenLake | 〔部分〕 | **segment**（多個 block） | 跨實例的 GPU HBM pool；NVLink／IB；**傳 query 到 segment 所在處** | 用二分圖匹配把通訊量降到最低（看量，不是看延遲〔判讀〕） | 有：逐層 Put 和後續計算重疊；query 通訊被本地 self-attention 蓋住 |
| MemServe | 〔部分〕 | 依 radix tree 的 token block；by-request 時把 2L 個 block 合成一塊 | 實例之間（HBM 或 DRAM），NCCL send/recv | **有**：式 (1) 選實例時含排隊；式 (2) 傳輸時間 ≤ 少算的時間才傳，否則重算 | P→D 逐層傳輸和計算重疊（低負載時最好） |
| Mooncake Store | 〔文件〕 | 物件（Put／Get），物件切成 slice | 每個 Client 貢獻一段記憶體；client 之間直接傳（Transfer Engine、GPUDirect RDMA、多網卡） | Store 文件沒寫；Mooncake 的 Conductor 在 repo L3 §G3 | Store 文件沒寫 |
| LMCache P2P | 〔部分〕＋〔文件〕 | chunk | 本地 CPU 後端之上的唯讀共享層；NIXL（RDMA） | 論文只寫「例如挑命中 chunk 最多的 peer」 | 部落格沒提 |
| llm-d P2P | 〔文件〕 | block（依 hash） | producer 的 CPU 卸載層 → consumer 的 CPU 層，NIXL／UCX、RDMA；不經 GPU | **部分**：靜態門檻 `minCachedTokenDelta`，依量到的打平點校準；平手時看 peer 的等待佇列深度 | **沒有**：未來工作才提預取，讓傳輸疊在空閒時間 |
| AlignedServe | 〔摘要〕 | NOT_VERIFIED | 一張卡替另一張卡預取（GPU-Prefetch-For-GPU），到排程時再經 NVLink 送去 decode | NOT_VERIFIED | NOT_VERIFIED |

### 4.1 TokenLake（〔部分〕：p.1–8）

- B. Wu, Z. Zhang, Y. Zhong, G. Huang, Y. Zhu, X. Liu, X. Jin（北大、StepFun），arXiv 2508.17219v1（2025-08-24）。
- **問題**：cache-aware routing 會造成三件事（p.1、p.3）：
  1. 高命中的實例頻寬被吃滿、低命中的閒著（負載不均）。
  2. 共享前綴被複製到多個實例（冗餘）。
  3. 整段前綴必須住在同一個實例（碎片）。
  - P/D 分離讓這三件事更嚴重。
- **機制**：
  - 宣告式介面：把 query tensor、prefix cache、cache 相關操作交給 TokenLake，讓排程器不用管快取（p.1、p.4）。
  - segment 層級：一般 segment 用 hash 均勻分散；**heavy hitter**（常被讀的 segment）在實例過載時選擇性複製到負載最低的實例（p.6–7 §4.3）。
  - 二分圖匹配分派，降低 query 與新 cache 的通訊量（p.7）。
  - 查詢引擎、傳輸引擎是獨立的 process，和計算引擎放在同一張卡；零拷貝；逐層 Put 和後續計算重疊（p.5）。
  - 主張：segment 夠大時，分散查詢比本地查詢還快，因為多張卡的頻寬和算力可以一起處理 prefix attention（p.6）。
- **評估**（p.8）：
  - 8×A100-80GB 伺服器，NVLink 400 GB/s，4×200 Gbps IB；LWM-1M-Text。
  - LooGLE（平均 24K）、SCBench（平均 227K、5 輪）、ShareGPT；Poisson 到達。
  - 吞吐對 SGLang-Router 最多 2.6 倍、對 Mooncake 式 cache-centric P/D 最多 2.0 倍；命中率 2.0 倍、2.1 倍（p.1）。
- 〔判讀〕它是「去重＋共享 segment 給多個請求」的跨卡版本，但做法是**把計算搬到資料那裡**，不是把資料取回一次再扇出。

### 4.2 MemServe（〔部分〕：p.1–2、p.4、p.6–8）

- C. Hu, H. Huang, J. Hu 等 11 人（華為雲、中科院計算所、北大），arXiv 2406.17565v3（2024-12-21），venue NOT_VERIFIED。
- MemPool 管整個叢集的 CPU DRAM 與 GPU HBM，提供記憶體、索引、分散傳輸三類 API。`transfer_with_insert` 傳完直接插進對方的索引（p.1–4）。
- **傳輸方式**（p.6 §5.2）：
  - by-layer 傳輸會和計算重疊，低負載時 JCT 最好。
  - 但它用 NCCL send/recv，一次只送一個 block，高負載時呼叫次數太多。
  - 改成 by-request-agg：把每層的 2 個 block 合成一塊，呼叫次數少 2L 倍。
- **排程**（p.7 §5.3；p.8 §6）：
  - 式 (1)：選實例時，把排隊（該實例未完成請求的執行時間總和）和自己的執行時間加起來取最小。
  - 式 (2)：別的實例快取得更多時，傳輸時間 ≤ 少算的時間才傳，否則重算。傳輸時間＝資料量÷最大頻寬；每個 batch 套用一次（repo novelty_sota 已記式 (2)）。
  - 全域排程器選最長共同前綴的實例，並列出哪些其他實例有額外的 KV。
- **評估**：單台 8×H800-80G；P/D 分離的 JCT 最多改善 42%，加 context caching 再改善 29%（p.2）。
- 〔判讀〕MemServe 有「傳輸 vs 重算」，但**沒有**「藏延遲的窗口」：式 (2) 比的是傳輸時間和計算時間，沒有扣掉可以藏住的部分。

### 4.3 Mooncake Store（〔文件〕：GitHub `docs/source/design/store/mooncake-store.md`，commit 868a6d05ab，2026-10-24）

- 分散式 KV cache 儲存：key 由內容 hash 得到，所以值寫入後不變；物件層級的 Put／Get／Remove。
- **多副本**：同一物件可以存多份來分散熱點；同一物件的 slice 放在不同 segment；best-effort。
- **傳輸**：Transfer Engine 零拷貝，多網卡 GPUDirect RDMA 合併頻寬；大物件 striping、平行 I/O；資料從一個 Client 直接傳到另一個，不經 Master。
- **淘汰**：用量達 90% 時，以近似 LRU 淘汰 5%；有 lease（預設 10 s）保護正在讀的物件。
- **其他**：soft pin 留給 system prompt 這類常用物件；preferred segment 讓物件放近一點；RAM→SSD 多層。
- 文件**沒寫**排程是否考慮傳輸延遲、是否和計算重疊。這兩點在 Mooncake 論文和 Conductor（repo L3 §G3）。
- 另有 P2P Store：BitTorrent 式，checkpoint 分發用，不是 KV（`p2p-store.md`）。
- vLLM 上有一個開著的 RFC #38474「Add Mooncake Store Connector for Shared KV Cache Reuse」（2026-03-29）〔PR〕。
- 〔判讀〕Mooncake Store 對熱門共享 KV 的做法是**多存幾份**（以空間換頻寬），和「取回一次、多人用」方向相反。

### 4.4 Infinite-LLM（〔部分〕：p.1–6）

- B. Lin, C. Zhang, T. Peng 等（阿里巴巴、上海交大、北大），arXiv 2401.02669v2（2024-07-04）。
- DistAttention 把 attention 切成 MicroAttention。KV 不夠的實例（debtor）向有空間的實例（creditor）借記憶體，把一部分 KV 放過去；計算時**只傳 query**（幾 KB），在 creditor 算部分 attention 再合併（p.5–6）。
- 重疊：query 傳輸和 MicroAttention 重疊；KV 傳到 creditor 時和推論重疊（p.6 圖 6）。
- 評估：32 張 A100、最長 2,000K token，吞吐 1.35–3.4 倍（p.1–2）。
- 〔判讀〕對象是**同一個模型的多個實例**，借的是容量；不涉及多個請求共享同一份 KV。

### 4.5 Harvest（只補 repo 03_bandwidth_compute 沒寫的；〔部分〕：p.1–3、p.6–8）

- N. Gopal, K. Kaffes（Columbia），arXiv 2602.00328v1（2026-01-30；PDF 標 Preprint，2026-02-03）。
- **語意**：
  - peer HBM 是 best-effort 的暫時快取層；空間消失時回收，並呼叫應用註冊的回呼。
  - 正確性不靠它：要嘛主機 DRAM 留權威副本，要嘛能接受遺失、事後重建（p.1、p.3）。
- **KV 的評估只是搬運延遲的微基準**（p.7 §5.3–5.4）：
  - DeepSeekV3、Mistral-Large-3、Kimi-K2 的 KV（FP16），100–8,000 個 entry。
  - CPU→GPU vs peer GPU→GPU：Kimi-K2 約 5.42–5.68 倍；Mistral-Large-3 約 3–5.65 倍。
  - 機器是 Azure NC80adis H100 v5（2×H100、12 條 NVLink、PCIe5）（p.3）。
- **原文的限制**（p.8 §7）：
  - 不感知拓撲，也不感知同一個 NVLink 叢集裡的其他負載。
  - 只在一台兩卡機器上評估；沒研究 NVLink 被別的 collective 或租戶占用的情況。
  - 原文把「依爭用預算頻寬」列為有前景的方向。
- 原文提到 Aqua 是近親：把 KV 移到經 NVLink／NVSwitch 相連的其他 GPU 的 HBM，用來加速搶占式分時推論的分頁（p.3）。

### 4.6 Aqua（〔摘要〕）

- A. Vijaya Kumar, G. Antichi, R. Singh, "AQUA: Network-Accelerated Memory Offloading for LLMs in Scale-Up GPU Domains," arXiv 2407.21255v3（2025-02-21）。venue：搜尋摘錄寫 ASPLOS 2025，abs 頁沒有 → NOT_VERIFIED。
- 在 8×H100 上，反應速度比既有最佳做法好 20 倍；單一長 prompt 推論吞吐 4 倍（摘要）。

### 4.7 LMCache P2P、llm-d P2P（新事實）

- **LMCache 論文**（〔部分〕）：
  - 控制器有 lookup／move／pin／compress 等 API；`batched_p2p_lookup` 回傳（實例、裝置、命中 chunk 數），由本地自行選 peer（p.10 表 3）。
  - 本地 CPU 往磁碟、遠端物件儲存同時寫時，用 reference count 避免複製（p.7 §5.3）。這是「一次寫、多個目的地」，不是「一次讀、多個讀者」。
- **LMCache 部落格**〈p2p-1〉（2026-01-21；騰訊、Tensormesh 作者）〔文件〕：
  - P2P 是建在 LocalCPU 後端上的唯讀共享層；容量受叢集 RAM 限制；假設有 RDMA 網卡。
  - 控制器 metadata 查詢約 8 µs。
  - 刻意構造的測試：Llama 3.1 8B、Long Doc QA、約 62 GB 不重複 KV；從 peer 拉的實例平均 TTFT 0.490 s，冷啟動的實例 2.028 s。GPU 型號沒寫。
  - 沒提重疊，也沒提排程。
- **llm-d 部落格**（2026-08-15；IBM Research、Red Hat）〔文件，兩次 WebFetch 一致；repo 卡片已記 GLM 打平點與 1.2–1.3 s 下限〕：
  - **block 層級**；CPU 卸載層之間 CPU 對 CPU 搬，不經 GPU。
  - 路由器先依負載選 endpoint；peer 比自己多快取的 token 數超過 `minCachedTokenDelta`（依量到的打平點校準的靜態門檻），才指定它當 KV 來源；幾個 peer 差不多時，依它們的等待佇列深度挑。
  - gpt-oss-120b、H200：48K 時拉 235 ms vs 重算 1,988 ms。
  - **目前的拉取在關鍵路徑上**；未來工作才提預取，讓傳輸疊在空閒時間。

---

## §5 任務 4：GPU 閒置填補（只記和 KV 還原或共享有關的部分）

| 系統 | 讀到 | 有沒有處理 KV 還原或共享 |
|:--|:--|:--|
| ConServe（ICML 2026） | 〔全文〕 | **有還原，沒有共享**：離線請求的 KV 增量 checkpoint 到主機，被搶占時丟掉 GPU 副本，GPU 有空時背景增量預取，和計算重疊（§1.d） |
| HyGen（NeurIPS 2025；T. Sun, P. Wang, F. Lai；arXiv 2501.14808v4，2025-10-30） | 〔部分〕p.1–8 | **只有 prefix 共享的排序，沒有 KV 還原**：被搶占的離線請求**保留執行狀態**（原文說架構支援丟棄、保留、swap，但目前用保留，p.4）；離線請求依 prefix 樹的 DFS 順序排，讓 prefix 盡量共用，另有公平性延伸（p.6）。prefix 共享的效果是**模擬**的：把相鄰離線請求的共享前綴長度扣掉，最多 4 倍離線吞吐（p.8）。吞吐 3.87–5.84 倍（p.8；v4 摘要寫 3.9–5.8 倍） |
| Echo（arXiv 2504.03651，2025-03-01，Z. Wang 等 10 人） | 〔摘要〕 | KV 管理器依任務類型設優先度，用 prefix 共享減少重算；摘要沒寫還原、swap-in。離線吞吐最多 3.3 倍 |
| Valve（arXiv 2604.07874，2026-04-09，F. Liu 等 11 人） | 〔摘要〕 | 生產環境的線上、離線混部；每個線上請求最多搶占一次運算（亞毫秒）＋限速的子層級記憶體回收；摘要沒寫被回收的 KV 怎麼還原。部署在 8,054 張 GPU，叢集利用率 +34.6%，線上 TTFT <5%、TPOT <2% 的影響 |

〔判讀〕這條線的重點是「讓路」，KV 只是被搶占時要處理的狀態；**沒有一篇把共享 KV 的取回當成填空檔的工作**。

---

## §6 工程實務的證據：vLLM 上「共享前綴正在載入」的四個 issue／PR（〔PR〕，`gh` 唯讀查詢，2026-10-26）

| 編號 | 狀態 | 內容 |
|:--|:--|:--|
| #29087 | PR，2026-01-21 合併 | OffloadingConnector「Prevent redundant loads」：用 `_blocks_being_loaded` 避免同一批 block 被重複從 CPU 載入 |
| #44294 | issue，2026-06-02 開，2026-10-07 因過期自動關閉（不是修好） | 2×L40S（TP2）、Qwen3-8B、vLLM 0.19.1、50 個併發請求、共享 10K token 前綴：凡是引用到正在載入的 block 的請求**整個被延後**，排成一列；TTFT 11.46 s（0.17.1 是 0.90 s；不卸載是 5.95 s），吞吐 −39.9%。報告者說只在 prefix caching 開著時發生 |
| #44295 | PR，仍開著 | 修法：GPU 有壓力（上一步有請求帶外部 token）才延後，否則當作沒命中、直接算。50 併發從 −40.7% 改善到 −32.4%；報告者說剩下的落差是真實的卸載成本 |
| #44882 | feature request，2026-06-08 開，仍開著 | 要求核心機制：同一段外部前綴只配置、載入一次，後到的請求掛上同一份並增加 reference count；還在載入時，後到者等同一次載入。列出三個現有選項都不好：重複載入（浪費 I/O 和 GPU 空間）、延後後到者（排成一列）、當作沒命中（重算已知的命中） |

〔判讀〕這四筆說明「一次取回、多人用」在生產引擎裡還是**開著的設計問題**。三個選項的取捨（等、算、重複載），正好取決於「等的人能不能先做別的事」和「重算多貴」，也就是藏延遲的窗口和 κ。

---

## §7 對三向交集（KV 共享 × 藏延遲 × 排程）的意義

### 7.1 已經被做掉的部分（逐項附出處）

1. **共享本身（work avoidance）**：
   - 完全相同前綴：vLLM APC（§1 a）、SGLang RadixAttention（repo L3 §E2）、LMCache（§1 b）、Mooncake Store（§4.3）。
   - 非前綴＋部分重算：CacheBlend（repo）、Cache-Craft（§3.4）、KVCOMM（§3.3）、TokenDance（§3.5）、FusionRAG（§3.10）。
   - 跨模型：DroidSpeak（repo D5）、PrefillShare（§3.10）。
2. **共享 KV 的取回只做一次**：
   - KVFlow 的節點狀態防重複載入（§3.8，p.6）。
   - Strata 的 delay hit 延後＋bundle hit（§3.9，p.6–7）。
   - vLLM #29087（§6）。
   - 取回的不是 KV、但同一個樣板：TeleRAG 的 micro-batch 預取重疊（§3.2，p.6）。
   - 跨卡去重：TokenLake 的 segment 去重＋heavy hitter 複製（§4.1，p.6–7）。
3. **藏在別人的計算或排隊時間後面**：
   - Strata 的 balanced batch 與 bubble filling（§3.9，p.6–7）。
   - KVFlow 的 status-aware 排程（§3.8，p.6）。
   - LMCache 讓請求回佇列（§1 b；論文 p.9，圖 13 在 p.14 的 1.46 倍）。
   - Cache-Craft、PCR 的排隊時預取（§3.4 p.8；§3.6 p.11–12）。
   - ConServe 的背景預取（§1.d，p.6）。
4. **排程把搬運當成輸入**：
   - Strata 的 loading_bound 門檻（p.7）。
   - MemServe 的式 (1)(2)（p.7）。
   - TeleRAG 的 B_link×t_LLM（p.6）。
   - Cache-Craft 的 L_p 與「沒到就重算」（p.7–8）。
   - llm-d 的 `minCachedTokenDelta`＋佇列深度（§4.7）。
   - RAGCache 的推測閘門（p.8–9）。
   - CacheSmith 依 SLO 的排程（p.2）。
5. **借別張卡的空閒 HBM**：
   - SwiftCache（§1.c）、Harvest（§4.5）、Aqua（§4.6）、Infinite-LLM（§4.4）、TokenLake（§4.1）、AlignedServe（§3.10）。

### 7.2 明確沒被做的部分（查不到 ≠ 沒有）

| 空缺 | 最近的前作與差距 | 怎麼查的 |
|:--|:--|:--|
| **G1**：很多請求在等**同一個 in-flight 的共享載入**時，**每個等待者**該等、該重算、還是先跑別的工作，依「窗口裡有多少可以藏的別人的計算」與 κ 決定 | Strata 只延後「正在**計算**」的 context，門檻是固定的 100 token，也不把重算當選項。KVFlow 只會「跳過、等就緒」。Cache-Craft 是「到時沒到就重算」二選一。vLLM #44295 只看一個 GPU 有沒有壓力的旗標，#44882 還開著 | 讀了 Strata、KVFlow、Cache-Craft、PCR、LMCache、RAGCache 的排程段落；`gh search` vLLM issue／PR（關鍵字：dedup、in-flight、being loaded、same prefix）；WebSearch 搜「load once shared by multiple requests」「in-flight KV load coalescing」 |
| **G2**：共享 KV 由一張卡取回一次，**經 NVLink 扇出**給多張卡上的多個請求，而且排程知道扇出的時間 | SwiftCache 只有一個讀者。TokenLake 是把 query 搬到資料那裡。Mooncake Store 用多副本。AlignedServe 是一卡替一卡預取（只讀摘要）。Harvest、Aqua 借的是容量 | WebSearch 搜「broadcast／multicast shared prefix KV NVLink data-parallel replicas」；讀了 SwiftCache、TokenLake、Harvest 全文或部分；Mooncake Store 文件 |
| **G3**：藏延遲的窗口和「取回 vs 重算」的切換，依**硬體會變的成本比 κ**一起決定 | MemServe 的式 (2) 比傳輸和計算，但不扣能藏住的部分。Strata 的 loading_bound 門檻要依硬體 profile，但不重算。KVFlow 假設 PCIe 快。Cache-Craft 的 L_p 只決定預載深度 | 同上，另加 repo L3 §H 的 InferCept、Continuum、Tokencake |
| **G4**：多 agent 共享 context（非前綴、要修正）**加上**卸載、預取、重疊 | KVCOMM 自己說卸載 1,260+ ms／agent、管線化留給以後（p.29）。TokenDance 只沿用 LMCache 的逐層路徑，Master 每個 Mirror 各載一次〔判讀〕 | 讀了 KVCOMM 附錄、TokenDance §4–6；WebSearch 搜「multi-agent shared KV prefetch overlap」 |

### 7.3 對新提案的威脅等級

| 新提案若主張… | 威脅 | 理由 |
|:--|:--|:--|
| 「共享 KV × 藏延遲 × 排程」這個**組合本身**是新的 | **高** | Strata（delay hit、bundle hit、balanced batch、bubble filling）和 KVFlow（狀態防重複＋藏在別的 agent 後面＋依步驟圖預取）已經把組合做完；LMCache、Cache-Craft、PCR 也各有一部分 |
| 用 NVLink 或鄰卡 HBM 讓共享 KV 的取回變快 | **高** | 這是 latency reduction，不是 hiding；SwiftCache、Harvest、Aqua、AlignedServe 已做 |
| G1：等待者的決策（等／算／先做別的）依窗口與 κ | **中** | vLLM 社群在 2026 年連開 4 筆 issue／PR 還沒有定論，代表問題是真的、而且沒有標準答案；但 Strata 的延後和 bundle hit 很近，審稿人一定會拿來比。新意要靠「不同硬體上決策會翻轉」的實測撐起來 |
| G2：一次取回＋NVLink 扇出＋排程 | **中** | 沒找到直接前作；但 TokenLake 的「搬 query 不搬資料」和 Mooncake 的多副本是兩種現成的替代做法，必須正面比較；還要先量 NVLink 扇出 vs 多條 PCIe 各自取回（repo 03 M06 已列 Beluga 的警告：PCIe 頻寬不能直接相加） |
| G3：κ 決定窗口與切換 | **低～中** | 沒找到；但 MemServe 的式 (2) 與 Strata 的 loading_bound 是近親。延續 Tiara「κ 跨硬體變 32 倍」的主張，差異化最清楚 |
| G4：多 agent 非前綴共享＋卸載預取 | **中** | KVCOMM 自己點名是空缺；但 TokenDance（2026-04）已經在 LMCache 上做 collective reuse，下一步很可能就是這個 |

---

## §8 參考清單（IEEE 格式；arXiv abs 頁皆已開過確認存在）

1. W. Kwon *et al.*, "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023；vLLM 設計文件 "Automatic Prefix Caching," https://docs.vllm.ai/en/latest/design/prefix_caching.html 與 https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html（2026-10-26 讀取）.
2. Y. Liu, J. Yao, Y. Cheng *et al.*, "LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference," arXiv:2510.09665v2, Dec. 2025.（PDF 首頁作者順序；README 引用寫 Cheng et al.；venue NOT_VERIFIED）
3. LMCache, "P2P KV Cache Sharing" 與 quickstart "Share KV cache across multiple LLMs," https://docs.lmcache.ai/ ；部落格 https://blog.lmcache.ai/en/2026/01/21/p2p-1 （2026-01-21）.
4. J. Hu, M. Xu, S. Wang, C. Ma, M. Shen, K. Ye, L. Qu, C. Xu, "SwiftCache: Efficient LLM Serving for Multi-turn Conversations with Heterogeneous KV Cache Sharing," arXiv:2606.16135v1, Jun. 2026.（venue NOT_VERIFIED）
5. Y. Qiao, S. Yu, S. Anzai, H. Ma, S. Yang, Y. Wang, M. Kim, Y. Wu, Y. Zhou, J. Xing, J. E. Gonzalez, I. Stoica, H. Xu, "ConServe: Fine-Grained GPU Harvesting for LLM Online and Offline Co-Serving," in *Proc. ICML*, PMLR vol. 306, pp. 101429–101443, 2026；arXiv:2410.01228（v1 2024-10 標題不同；v2 2025-09）.
6. C. Jin, Z. Zhang, X. Jiang, F. Liu, X. Liu, X. Liu, X. Jin, "RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation," arXiv:2404.12457v2, Apr. 2024.（ACM TOCS NOT_VERIFIED）
7. C.-Y. Lin, K. Kamahori, Y. Liu *et al.*, "TeleRAG: Efficient Retrieval-Augmented Generation Inference with Lookahead Retrieval," in *Proc. MLSys*, 2026（PDF 頁尾）；arXiv:2502.20969v4.
8. H. Ye, Z. Gao, M. Ma *et al.*, "KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based Multi-agent Systems," in *Proc. NeurIPS*, 2025；arXiv:2510.12872v2.
9. S. Agarwal, S. Sundaresan, S. Mitra *et al.*, "Cache-Craft: Managing Chunk-Caches for Efficient Retrieval-Augmented Generation," SIGMOD 2025（arXiv comments）；arXiv:2502.15734v1.
10. Z. Bian, F. Wu, C. Zhang, H. Dong, Y. Liang, Y. Zhuo, "TokenDance: Scaling Multi-Agent LLM Serving via Collective KV Cache Sharing," arXiv:2604.03143v1, Apr. 2026.
11. W. Wang, X. Hou, P. Tang *et al.*, "PCR: A Prefetch-Enhanced Cache Reuse System for Low-Latency RAG Serving," arXiv:2603.23049v1, Mar. 2026.
12. X. Ma, D. Lee, A. Shah, M. Jeon, "CacheSmith: Accurate RAG Serving under Latency SLOs with Adaptive KV Reuse," arXiv:2610.18925v1, Oct. 2026.
13. Z. Pan *et al.*, "KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows," NeurIPS 2025；arXiv:2507.07400.
14. Z. Xie *et al.*, "Strata: Hierarchical Context Caching for Long Context Language Model Serving," arXiv:2508.18572v1, Aug. 2025.（arXiv：審稿中；repo 記 OSDI'26，本次未重查）
15. K. Tan, R. Gu, M. Li, "CacheWeaver: Cache-Aware Evidence Ordering for Efficient Grounded RAG Inference," arXiv:2606.19667v2, 2026.
16. J. Wang *et al.*, "From Prefix Cache to Fusion RAG Cache: Accelerating LLM Inference in Retrieval-Augmented Generation," arXiv:2601.12904, Jan. 2026.
17. S. Wang, R. Ren, L. Gui, "ForkKV: Scaling Multi-LoRA Agent Serving via Copy-on-Write Disaggregated KV Cache," arXiv:2604.06370, Apr. 2026.
18. S. Woo *et al.*, "PrefillShare: A Shared Prefill Module for KV Reuse in Multi-LLM Disaggregated Serving," arXiv:2602.12029v2, 2026.
19. R. Zhang, C. Kim, L. Hu, "A Policy-Driven Runtime Layer for Agentic LLM Serving," arXiv:2605.27744v2, 2026.
20. F. Bai *et al.*, "AlignedServe: Orchestrating Prefix-aware Batching to Build a High-throughput and Computing-efficient LLM Serving System," arXiv:2605.23389, May 2026.
21. B. Wu, Z. Zhang, Y. Zhong, G. Huang, Y. Zhu, X. Liu, X. Jin, "TokenLake: A Unified Segment-level Prefix Cache Pool for Fine-grained Elastic Long-Context LLM Serving," arXiv:2508.17219v1, Aug. 2025.
22. C. Hu, H. Huang, J. Hu *et al.*, "MemServe: Flexible Mem Pool for Building Disaggregated LLM Serving with Caching," arXiv:2406.17565v3, Dec. 2024.
23. Mooncake, "Mooncake Store" design doc, https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/design/store/mooncake-store.md （commit 868a6d05ab, 2026-10-24）.
24. B. Lin, C. Zhang, T. Peng *et al.*, "Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache," arXiv:2401.02669v2, Jul. 2024.
25. N. Gopal, K. Kaffes, "Harvest: Opportunistic Peer-to-Peer GPU Caching for LLM Inference," arXiv:2602.00328v1, Jan. 2026.
26. A. Vijaya Kumar, G. Antichi, R. Singh, "AQUA: Network-Accelerated Memory Offloading for LLMs in Scale-Up GPU Domains," arXiv:2407.21255v3, 2025.（ASPLOS 2025 NOT_VERIFIED）
27. N. Guy, L. Schour, M. Ayoub, "Pull, Don't Recompute: Peer-to-Peer KV Cache Sharing in llm-d," llm-d blog, Aug. 15, 2026, https://llm-d.ai/blog/p2p-kv-cache-sharing-llm-d .
28. T. Sun, P. Wang, F. Lai, "HyGen: Efficient LLM Serving via Elastic Online-Offline Request Co-location," NeurIPS 2025；arXiv:2501.14808v4.
29. Z. Wang *et al.*, "Echo: Efficient Co-Scheduling of Hybrid Online-Offline Tasks for Large Language Model Serving," arXiv:2504.03651, Mar. 2025.
30. F. Liu *et al.*, "Valve: Production Online-Offline Inference Colocation with Jointly-Bounded Preemption Latency and Rate," arXiv:2604.07874, Apr. 2026.
31. vLLM GitHub: PR #29087（"OffloadingConnector: Prevent redundant loads"）；issue #44294；PR #44295；issue #44882；RFC #38474. https://github.com/vllm-project/vllm .

---

## 附錄：本次的不確定性與沒做的事

- **只讀摘要的**：Aqua、ForkKV、PrefillShare、FusionRAG、AlignedServe、Echo、Valve、2605.27744。它們和本題的關係只根據摘要判斷。
- **WebFetch 讀的**：llm-d、LMCache 部落格、vLLM 文件、CacheWeaver 的 HTML，都經工具內的小模型整理。llm-d 的數字讀了兩次、彼此一致，也和 repo 卡片一致；其他只讀一次。
- **NOT_VERIFIED 的 venue**：RAGCache 的 TOCS、Strata 的 OSDI'26（arXiv 寫審稿中）、Aqua 的 ASPLOS'25、SwiftCache 的 venue。
- **沒讀到或抽字斷掉的**：Strata 的 delay hit 第二種情境（換頁處斷掉）；SwiftCache 在 worker 回收區塊時 master 的 KV 怎麼辦；RAGCache 的 host→GPU 搬運是否重疊（讀到的段落裡沒有）。
- **TeleRAG v4 的數字前後不一致**：摘要是單查詢 1.98 倍、batch 1.83 倍；引言是單查詢 1.53 倍、batch 1.98 倍。引用時要標明出處。
- **沒查的**：SGLang HiCache 現行程式碼是否已實作 Strata 的 delay hit 與 bubble filling；LMCache 現行版本如何處理 in-flight 的重複載入；NVIDIA Dynamo KVBM 的對應行為。這三項都 NOT_VERIFIED。
- **搜尋用過的關鍵字**（WebSearch）：
  - multi-agent shared KV prefetch overlap 2026
  - RAG serving KV retrieval overlap prefetch chunk cache 2026
  - load once shared by multiple requests deduplicate transfer
  - broadcast／multicast shared prefix KV NVLink data-parallel replicas
  - shared context multi-agent one copy many agents
  - online offline co-serving harvest idle GPU KV preemption
  - TokenLake、HyGen、Aqua、Mooncake Store、LMCache P2P
- **GitHub 唯讀查詢**（`gh`）：
  - vLLM 的 issue／PR：#29087、#44294、#44295、#44882、#38474、#44223。
  - Mooncake 的 `docs/source/design` 目錄。
