# KV Cache 論文導讀：35 篇，分 8 類

> **這份文件是什麼**：把 Tiara 相關的 35 篇論文，用白話逐篇講清楚。每篇都回答六件事：
> 它想解決什麼問題、怎麼做、實驗怎麼測、成果多少、有什麼短板、跟我們的關係。
>
> **可信度**：每一篇都讀了原文（PDF 在 `papers/`，來源見 `papers/SOURCES.tsv`）。
> 「成果」欄的數字**全部取自該論文的摘要或正文**，不是我們量的。
> 「短板」的逐條出處見 [`RELATED_WORK_WEAKNESSES.md`](RELATED_WORK_WEAKNESSES.md)。
> 標「（我們的判讀）」的是推論，不是原文。
>
> Strata（OSDI'26）與 MTDS（C&IS'26）由使用者另外提供 PDF，已補讀。
>
> 2026-09-23 整理。

---

## 目錄

- [第 0 章 先懂這幾個名詞](#第-0-章-先懂這幾個名詞)
- [地圖：35 篇怎麼分類](#地圖35-篇怎麼分類)
- [第 1 類 多層儲存系統：KV 放 GPU、CPU 還是 SSD？](#第-1-類-多層儲存系統kv-放-gpucpu-還是-ssd)
- [第 2 類 可恢復驅逐：丟掉的還能撿回來嗎？](#第-2-類-可恢復驅逐丟掉的還能撿回來嗎)
- [第 3 類 用「當下的 query」挑 KV：這一步要讀哪些？](#第-3-類-用當下的-query-挑-kv這一步要讀哪些)
- [第 4 類 重算 vs 傳輸：要用時，算回來還是搬回來？](#第-4-類-重算-vs-傳輸要用時算回來還是搬回來)
- [第 5 類 學習式驅逐：用模型預測誰未來有用](#第-5-類-學習式驅逐用模型預測誰未來有用)
- [第 6 類 品質、壓縮與放置最佳化：有損的時候怎麼取捨？](#第-6-類-品質壓縮與放置最佳化有損的時候怎麼取捨)
- [第 7 類 KV 量化：一個數字要用幾個 bit？](#第-7-類-kv-量化一個數字要用幾個-bit)
- [第 8 類 量測研究與綜述：別人怎麼看這整個領域](#第-8-類-量測研究與綜述別人怎麼看這整個領域)
- [附錄 A：一表總覽](#附錄-a一表總覽)
- [附錄 B：讀原文時抓到、跟我們有關的事](#附錄-b讀原文時抓到跟我們有關的事)

---

## 第 0 章 先懂這幾個名詞

讀這批論文會一直遇到下面這些詞。先用一個比喻把它們串起來：

> **把 LLM 想成一個在讀長篇文件的人。** 他每讀一個字，就在筆記本上記一行摘要（這就是 **KV**）。
> 之後要寫下一個字時，他得回頭翻**整本筆記**（這就是 **attention**）。
> 筆記本放在手邊的桌上最快（**GPU 記憶體 / HBM**），但桌子很小；
> 放不下的可以收進抽屜（**CPU 記憶體 / DRAM**）或倉庫（**SSD**），拿回來比較慢；
> 或者乾脆撕掉，要用時再把原文重讀一遍重寫筆記（**重算 / recompute**）。

| 名詞 | 白話 |
|---|---|
| **KV cache** | 模型對每個已讀 token 記下的「鍵（K）」和「值（V）」。有了它，生成下一個 token 時就不必把前文全部重算一次。代價是它**隨上下文長度線性變大**，長上下文時很容易比模型權重還大 |
| **prefill / decode** | **prefill** = 一次讀完整段輸入、建出 KV（算力密集）；**decode** = 一個一個吐出輸出 token，每吐一個都要讀一遍全部 KV（頻寬密集） |
| **TTFT / TPOT / TBT** | 首個 token 要等多久 / 之後平均每個 token 多久 / 相鄰兩個 token 間隔多久。TTFT 主要看 prefill，後兩者主要看 decode |
| **MHA vs GQA** | **MHA**（Multi-Head Attention）：每個 attention head 都有自己的 K、V。**GQA**（Grouped-Query Attention）：好幾個 head **共用**一組 K、V，所以 KV 小很多。**2023 年以前的模型（OPT、LLaMA-1、Llama-2-7B/13B、Qwen1）多是 MHA；Llama-3、Qwen2 以後都是 GQA**。這個差別決定了好幾篇論文的方法還能不能用（見 Bidaw、HCache） |
| **多層儲存 / 卸載（offload）** | KV 放不下 GPU，就搬到 CPU 記憶體或 SSD。越下層越大、越便宜、越慢 |
| **驅逐（eviction）** | 空間不夠時決定「誰要離開」。可能是搬到下一層，也可能直接丟掉 |
| **重算（recompute）** | 丟掉的 KV 要用時，再從原文 token 算一次。不佔儲存，但花算力，而且**越後面的 token 越貴**（要看的前文更長） |
| **prefix caching / 跨請求重用** | 很多請求有相同的開頭（系統提示、多輪對話的歷史、同一份文件）。把這段的 KV 存起來，下一個請求就能直接用，不必重做 prefill |
| **稀疏 attention** | 每一步只讀「最重要的一小部分 KV」，而不是全部。省頻寬，但屬於近似，可能選錯 |
| **量化（quantization）** | 把 KV 從 16 bit 壓成 8、4、2 bit。省空間、省頻寬，但會損失精度 |
| **有損 vs 無損** | 搬到 CPU、SSD 再搬回來：位元組一模一樣，是**無損**的；量化、丟 token、稀疏 attention 則是**有損**的，可能影響答案品質 |
| **時間戳 / Poisson 到達** | 真實服務裡，請求在什麼時間點抵達，決定了「兩次使用同一段 KV 之間隔多久」。很多公開資料集（如 ShareGPT）**沒有時間戳**，研究者只好用 Poisson 分布亂數產生。**這會讓依賴時間規律的策略失效**（Bidaw 就親身示範了） |
| **ShareGPT** | 網友分享的 ChatGPT 多輪對話，是評測多輪對話服務最常用的公開資料集。有對話內容，**沒有到達時間** |
| **LongBench / RULER / NIAH** | 常見的長上下文評測。LongBench 是真實任務（問答、摘要、程式），RULER 與 NIAH（大海撈針）偏合成。**NIAH 太簡單，會高估方法的能力**（見 Yandex） |
| **κ（kappa）** | 本專案用來表示「重算一段 KV 的成本 ÷ 把它傳回來的成本」。κ 越大，代表重算越不划算 |

---

## 地圖：35 篇怎麼分類

| 類別 | 它們在回答的問題 | 論文 |
|---|---|---|
| **1. 多層儲存系統** | KV 太多，放不下 GPU，要放 GPU、CPU 還是 SSD？怎麼搬才快？ | Bidaw、FlexGen、CachedAttention、Mooncake、KVDrive、Tutti、LMCache、**Strata**、**MTDS** |
| **2. 可恢復驅逐** | 丟掉（或壓扁）的 KV，後來又變重要了，能不能撿回來？ | ArkVale、QEvict |
| **3. 用當下 query 挑 KV** | 這一步生成時，只讀哪幾塊 KV 就夠了？ | InfiniGen、Quest、ShadowKV |
| **4. 重算 vs 傳輸** | 要用一段不在手邊的 KV 時，算回來和搬回來哪個快？能不能兩個一起做？ | Cake、KVPR、HCache、CacheBlend |
| **5. 學習式驅逐** | 能不能訓練一個模型，預測哪些 KV 未來還會用到？ | KVP、ForesightKV、LookaheadKV、TRIM-KV、Marconi、SAECache |
| **6. 品質、壓縮與放置最佳化** | 允許有損壓縮時，怎麼在「速度」和「答案品質」之間取捨？放置能不能寫成最佳化問題？ | KVServe、EvicPress、AdaptCache、LeoAM、OrbitFlow、Het-Mem |
| **7. KV 量化** | KV 用幾個 bit 存，才不會傷到品質？ | KIVI、KVTuner |
| **8. 量測研究與綜述** | 整個領域的瓶頸在哪？評測方法可靠嗎？ | Bottlenecks、Yandex（YAKV）、KV Survey |

**Tiara 站在哪裡**：它想把第 1 類的「放哪一層」、第 4 類的「丟掉後重算」、第 7 類的「用幾個 bit」統一成**同一把梯子**（GPU-BF16 → GPU-FP8 → GPU-INT4 → CPU → SSD → 丟掉重算），再用第 5 類的「學習式預測」決定每塊 KV 站在梯子的哪一階，並以第 6 類的「品質約束」當作限制條件。

---

## 第 1 類 多層儲存系統：KV 放 GPU、CPU 還是 SSD？

> **這一類的共同想法**：GPU 放不下，就往下層放；重點在「搬得夠快」和「常用的放上層」。
> **這一類的共同盲點**：大多數只看「過去誰被用過」來決定放哪；也沒有把「丟掉重算」當成一個選項。

### Bidaw（FAST'26，清華）
**一句話**：多輪聊天的 KV 存在「記憶體＋SSD」兩層，讓「算的一方」和「存的一方」互相知道對方的狀態。

- **想解決什麼**：多輪對話時，每一輪都要把前面幾輪的 KV 從 CPU 記憶體或 SSD 拿回 GPU。原文量到，比起「全部都在記憶體裡」的理想情況，既有方法的延遲**最多高 3.8 倍**、吞吐量**最多低 2.0 倍**。原因是 GPU 排程和儲存層各做各的。
- **怎麼做**（三招）：
  1. **排程看 KV 在哪**：把「KV 在記憶體」和「KV 在 SSD」的請求分開排，再依 KV 大小重新排序。這樣 GPU 不會卡在等某個從 SSD 慢慢搬的請求。
  2. **用「上一輪回答多長」預測下次何時再用**：作者觀察到，模型上一輪回答越長，使用者讀完再回覆的時間通常越久，這段 KV 下次被用到的時間也越晚。所以回答長的，優先趕到 SSD。
  3. **不存 KV，改存 tensor 6**：模型運算中有好幾個中間張量都能換算回 KV。作者算「每 MB 能省下多少計算」，發現第 6 個張量（normalized activation）最划算（51.0 GFLOPs/MB，KV 本身是 30.5），就改存它，要用時再花一步轉回 KV。
- **實驗**：OPT-6.7B/13B/30B、Qwen-7B/14B（Qwen1）｜合作廠商的真實聊天 trace（超過一百萬輪，平均 query 36 token、回答 45 token、平均 22 輪）＋ ShareGPT（只跑 OPT-13B）｜1× A800、200 GB 記憶體、4 顆 SATA SSD 組 RAID-5（1.5 GB/s）
- **成果**：延遲最多降 **3.58 倍**、吞吐量最多升 **1.83 倍**，接近「全部 KV 都在記憶體」的理論上限。
- **短板**：
  - **第 2 招要靠真實的人類互動時間戳。** ShareGPT 沒有時間戳，用 Poisson 模擬以後，原文自己寫這招「no longer reduces miss rates」。
  - **第 3 招只對 MHA 有利。** 原文寫明 GQA 模型「KV is smaller ... no longer beneficial」。它舉的 LLaMA 是 LLaMA-1、Qwen 是 Qwen1，現在的主流模型都用不上。
  - 對話非常短（36 token 的問題），不是長上下文場景。
  - 只在一台機器上測。
- **跟 Tiara 的關係**：同樣是「多層＋預測何時再用」。但 Bidaw 的預測訊號（回答長度 → 人類閱讀時間）只適用於人類聊天；也沒有「丟掉重算」和「降精度」這兩個選項。它的 ShareGPT 結果正好可以拿來說明**為什麼不用 Poisson 補的時間戳**。

### FlexGen（ICML'23，Stanford 等）
**一句話**：只有一張小 GPU，也要跑 175B 大模型，於是把權重、KV、中間值都拆開放到 GPU、CPU、硬碟三層。

- **想解決什麼**：有些工作不在乎延遲、只在乎總量（例如批次處理一大堆文件）。能不能只用一張消費級 GPU 做到？
- **怎麼做**：
  1. 用**線性規劃**算出「權重、KV、activation 各有幾成放 GPU、CPU、disk」最省時。
  2. 改變計算順序（zig-zag block schedule），讓同一份權重被很多請求共用，減少搬運次數。
  3. 把權重和 KV **壓成 4 bit**。
- **實驗**：OPT-6.7B/30B/175B｜**合成資料，所有 prompt 補成同一長度**（512 或 1024），每個生成 32 token；吞吐量測試用假權重｜T4 16 GB、208 GB 記憶體、1.5 TB SSD
- **成果**：OPT-175B 在單張 16 GB GPU 上首次達到 **1 token/s**（有效 batch 144）；最大吞吐量比其他卸載系統高 **100 倍**。
- **短板**：為離線批次設計，不管延遲；配置事先算好，執行中不變；沒有跨請求重用；只測 OPT。
- **跟 Tiara 的關係**：最早把 KV 放進「三層」的工作之一。Tiara 的差別在於**線上、逐 block 地決定**，而且把重算和降精度也當成選項。

### CachedAttention（USENIX ATC'24，新加坡國立大學、上海交大、華為雲）
**一句話**：多輪對話不要每輪都重算歷史，把每個 session 的 KV 存進 HBM/DRAM/SSD，下一輪直接拿。

- **想解決什麼**：多輪對話每一輪都要重新 prefill 全部歷史，原文圖 4（Recomputation inefficiencies）顯示，隨著輪數增加，重算歷史占 prefill 時間的比例越來越高。
- **怎麼做**：
  1. **逐層預載、非同步存**：邊算第 i 層，邊把第 i+1 層的 KV 搬上來。
  2. **看排程器的佇列決定放哪**：排程器知道哪些 session 快輪到了，就先把它們的 KV 拉到快的一層。
  3. **把位置編碼和 KV 分開**：上下文超過長度上限要截斷時，存好的 KV 不會因此作廢。
- **實驗**：LLaMA-1 65B、LLaMA-2 13B/70B、Falcon-40B｜ShareGPT 9K 個 session（平均 5.75 輪），到達時間用 Poisson｜4× A100 80 GB、128 GB 記憶體、10 TB SSD
- **成果**：TTFT 最多降 **87%**，prefill 吞吐量最多升 **7.8 倍**，推論成本最多降 **70%**。
- **短板**：時間戳是模擬的；唯一的對照組是「全部重算」；模型多是 2K/4K 上下文的世代；預取只看得到已進佇列的請求（我們的判讀）。
- **跟 Tiara 的關係**：代表「session 級、按排程提示放置」這條路。Tiara 的預測器想看得更遠，猜的是**還沒抵達的請求**會用到誰。

### Mooncake（FAST'25 Best Paper，月之暗面 Kimi）
**一句話**：Kimi 的生產架構。prefill 和 decode 分開用不同機器，再把整個叢集閒置的 CPU、DRAM、SSD 池化成一個大的 KV 倉庫。

- **想解決什麼**：生產環境請求量常常超載，而且長上下文很多，GPU 放不下那麼多 KV。
- **怎麼做**：
  1. **prefill 叢集和 decode 叢集分開**（PD 分離），KV 在兩者之間傳。
  2. **全叢集共用的 KV 倉庫**：用每台機器閒著的 DRAM 和 SSD。
  3. **以 KV 為中心的排程器**：把請求派到「已經有它的 KV」的機器上。
  4. **超載時預測並提前拒絕**，不做白工。
- **實驗**：為保護商業機密，用**與 LLaMA2-70B 同架構的 dummy model**｜23,000 條真實 Kimi 請求（有真實到達時間）＋模擬資料｜每節點 8× A800、800 Gbps RDMA
- **成果**：模擬情境下吞吐量最多高 **525%**；真實負載下讓 Kimi 多處理 **75%** 的請求。
- **短板**：叢集級，需要 RDMA，單機用不上；dummy model 所以**沒有量品質**；原文自承在他們的負載裡「最多只有 50% 的 KV 能被重用」。
- **跟 Tiara 的關係**：**Tiara 的生產 trace 就取自 Mooncake 公開的資料**（toolagent 與 conversation）。注意它的 `hash_ids` 一個 block 是 **512 token**，不是 16（CLAUDE.md §1-6 記錄過踩雷經過）。

### KVDrive（preprint'26，港科大等）
**一句話**：單請求的超長上下文，把 KV 分放在 GPU、DRAM、SSD 三層，再把搬運和計算排成流水線。

- **想解決什麼**：既有的卸載系統把全部 KV 放在 CPU、每步挑重要的搬上來。但上下文和 batch 一變大，要搬的量就爆了，稀疏度又不能無限提高，否則會掉分。
- **怎麼做**：
  1. **prefill 時先剖析重要性**，決定每塊 KV 一開始放哪一層；全部 KV 另外寫一份到 SSD 當備份。
  2. **decode 時每步挑 top-K**，並在 GPU 裡快取「最近幾步常被挑中的」，減少重複搬運。
  3. **SSD 版面設計**：同一個 {layer, head} 的資料放在連續區段，讓 SSD 盡量做循序讀取。
  4. **流水線**：讓 I/O、CPU、GPU 三方重疊執行。
- **實驗**：Llama-3-8B-1048K、Qwen3-8B/14B、Phi-4-mini｜LongBench、RULER｜三種機器：L20 48 GB、H20 96 GB、RTX 4090 24 GB
- **成果**：吞吐量比既有最佳方法最多高 **1.74 倍**，準確度維持。
- **短板**：**只測單請求長文理解**，沒有多輪或跨請求重用；每塊 KV 都要寫進 SSD；量化列為 future work。
- **跟 Tiara 的關係**：少數有 SSD 階又用新模型（GQA）的工作。它的 future work 正好寫了「熱的放 HBM 用 FP16、冷的放 SSD 用 INT4」，這就是 Tiara 梯子的一部分。

### Tutti（preprint'26，廈門大學等）
**一句話**：從 SSD 讀 KV 回 GPU 時，連「發 I/O 指令」都讓 GPU 自己來，完全不經過 CPU。

- **想解決什麼**：vLLM 的 KV 在 GPU 上是很多零碎的小 block，從 SSD 讀回來就會變成**海量的小隨機 I/O**。每個 I/O 都要 CPU 發起，CPU 就成了瓶頸；就算用 NVIDIA 的 GPU Direct Storage（GDS）也一樣。
- **怎麼做**：
  1. **GPU 上的 KV 物件儲存**：把零碎 block 包成大物件整批搬。
  2. **GPU io_uring**：GPU 直接對 NVMe 發非同步 I/O，CPU 只要每層載入一次 I/O kernel。
  3. **看空檔排 I/O**：用 NVIDIA Green Contexts 切分 SM，避免 I/O kernel 霸佔計算資源。
- **實驗**：主要是 Llama3-8B 單卡（擴展性實驗用 GLM-4-9B-1M 兩卡）｜LEval、LooGLE 輪流抽樣，Poisson 到達｜H100、NVMe、PCIe 5.0
- **成果**：比起開了 GDS 的 LMCache，TTFT 降 **78.3%**，可承受的請求率高 **2 倍**，成本降 **27%**；效能接近「全放 DRAM」的 LMCache，容量卻幾乎無限。
- **短板**：只解決「怎麼搬」，不處理「該放哪、該丟誰」（我們的判讀）；依賴 GPU 檔案系統 GeminiFS 和 Green Contexts；時間戳是模擬的。
- **跟 Tiara 的關係**：它讓 SSD 這一階變快，**直接改變 Tiara 成本模型裡 SSD 的價格**。兩者可以疊加：Tutti 負責搬，Tiara 負責決定搬誰。

### LMCache（preprint'25，芝加哥大學 / Tensormesh）
**一句話**：業界最常用的開源 KV 快取層。把 vLLM、SGLang 產生的 KV 拿出 GPU，存到 CPU、磁碟、遠端，讓不同請求、不同引擎共用。

- **想解決什麼**：使用者存的 KV 總量增長得非常快，遠超過 GPU 記憶體，卻缺少一個又快又通用的搬運層。
- **怎麼做**：
  1. 批次化的搬運操作，加上計算和 I/O 的流水線。
  2. 模組化的 connector，跟推論引擎的版本變動解耦。
  3. **控制 API**：pin、lookup、cleanup、move、compress，讓上層可以自己寫策略。
- **實驗**：Llama-3.1-8B/70B、Qwen2.5-72B、Qwen2.5-Coder-32B、Qwen3-Coder-480B｜合成的多輪文件 QA（每個 query 10K token）、TriviaQA、兩家公司的真實 trace｜8× H100（雲端）
- **成果**：搭配 vLLM，吞吐量最多高 **15 倍**；另有業界觀察：上下文截斷會讓 prefix 命中率**掉一半**。
- **短板**：真實 trace 只用了輸入和輸出長度的分布，還把好幾天壓縮成 1 小時；它提供機制和 API，但放置策略不是它的貢獻；部分 baseline 是無法重現的商用服務。
- **跟 Tiara 的關係**：Tiara 的 **Tier 0 baseline 之一**（見 `main.tex` §6.4）。它預設的 `turboquant_k8v4`（K 用 8 bit、V 用 4 bit）是「K 比 V 敏感」的生產佐證。

### Strata（OSDI'26，Stanford、NVIDIA、SJTU、CMU 等）
**一句話**：SGLang 的分層快取（HiCache）。它把「KV 從 CPU 搬回 GPU」這件事做到又快又不卡，並讓排程器知道搬運要花時間。已經在生產環境部署。

- **想解決什麼**：長上下文時，系統常常卡在搬 KV 而不是計算。作者指出三個原因：
  1. KV 在 GPU 上被切成很多小 page，搬運時變成大量小傳輸，頻寬用不滿；
  2. 載入 KV 會卡住 prefill；
  3. **delay hit**：同一段上下文正在從下層載入時，又有別的請求要它，排程器卻不知道，造成嚴重的吞吐量下降。它引用 Mooncake 的 agent trace：38% 的請求會在 1 秒內和另一個請求共用至少 6K token 的前綴。
- **怎麼做**：
  1. **GPU 協助的 I/O**：GPU 端和主機端用不同的資料版面，讓傳輸可以整批進行。
  2. **cache-aware 排程**：避開容易發生 delay hit 的請求；組 batch 時讓「要載入的」和「要計算的」互相平衡，把載入時間藏起來；有空檔就插入互補的工作。
  3. **三種寫入策略**：write-back（快被逐出才寫）、write-through（一產生就寫）、預設的 **selective-write-through**（一個節點被存取的次數超過門檻，預設 2，才寫到下層）。
  4. **逐出策略：每一層都是 LRU**（原文：「For all memory layers, the Least Recently Used (LRU) algorithm serves as the default eviction policy」）。
- **實驗**：Llama-3.1-8B/70B、Qwen2.5-14B-1M、DeepSeek-V3｜LooGLE、NarrativeQA、ReviewMT、ShareGPT，**全部用 Poisson 到達**｜8× H200（PCIe 5.0）、H20 + NVMe 7 GB/s、GH200
- **成果**：吞吐量比 vLLM-LMCache 最多高 **5 倍**，比 TensorRT-LLM 最多高 **3.75 倍**，短上下文的效能不受影響。
- **短板**：
  - **放哪、丟誰用的都是 LRU**。它的貢獻在機制（怎麼搬、怎麼排），不在策略。
  - **不把重算當成選項**：原文認為上下文變長後重算「increasingly costly ... an unattractive alternative」。
  - 寫入的判斷依據是「被用過幾次」，不是「存起來划不划算」。
  - 原文自承的限制：I/O kernel 會搶 SM；排程不保證公平；從更慢的階預取只能靠佇列延遲來藏；只支援 dense attention。
- **跟 Tiara 的關係**：**這就是老師說的「Strata（機制）」**。它是很強的機制底座，但策略層是空的（LRU）。Tiara 要做的正是策略層，可以直接疊在這種機制上。要特別注意：它已經有「寫入時決定」，所以 Tiara 講「寫入時決定」時，必須說清楚我們的依據是**成本（位置）**，不是**次數**。

### MTDS（Complex & Intelligent Systems'26，中國電信研究院）
**一句話**：在只有 24 GB 小卡的邊緣伺服器上做 GPU → DRAM → SSD 三層 KV 儲存。每個請求進來先判斷「載入比較快還是重算比較快」；卸載時看佇列裡的請求決定誰優先；DRAM 滿了用自適應門檻逐出。

- **想解決什麼**：邊緣 GPU 算力弱、頻寬小、DRAM 也小。原文指出三個挑戰：
  1. 從 DRAM/SSD 載入 KV 的時間，**可能比直接重算還久**，這時重用反而讓 TTFT 變差；
  2. 多張 GPU 同時卸載會搶 PCIe；
  3. DRAM 放不下，固定的 LRU 門檻不是浪費空間，就是會爆掉。
- **怎麼做**：
  1. **選擇性載入**：用歷史延遲資料預測「載入時間」與「重算時間」，把請求分成三類：**全部載入**、**部分載入**（前綴載入、其餘重算）、**不載入**（全部重算）。歷史資料太少時，改用理論公式估計。
  2. **卸載排程**：把佇列裡累積的請求互相做前綴比對，被越多請求共用的 KV 越優先卸載。
  3. **自適應逐出**：DRAM 用兩級、會隨使用量調整的門檻，加上把「即將到來的請求」納入計算的滑動視窗 LRU。
- **實驗**：GPT-2 1.5B、LLaMa-2 7B、LLaMa-3 8B、Qwen-3 14B｜ShareGPT + Poisson（λ=1）、固定長度的合成資料｜4× A10 24 GB、64 GB DRAM、2 TB SSD、PCIe Gen4｜對照 vLLM、Mooncake、LMCache
- **成果**：TTFT 降 **超過 25%**，多層快取的命中率最多高 **20%**。
- **短板**：
  - **「算還是載」是整段一起決定**：要嘛整段前綴都載入，要嘛整段都重算，不會在同一段前綴裡依位置切開。
  - **「未來」只看得到已經進佇列的請求**，看不到還沒抵達的。
  - 決定發生在**讀取時**，不是寫入時。
  - **沒有精度階**，也沒有量品質。
  - 模型偏小（含 GPT-2），時間戳是 Poisson，只有一種硬體。
  - 原文的 future work：跨裝置協調、結合壓縮。
- **跟 Tiara 的關係**：**目前最接近 Tiara 的一篇**。它同時有多層、重算選項與「未來」預測，也同樣鎖定 24 GB 級的小卡（它用 A10，我們用 RTX 3090）。Tiara 必須正面回答跟它的差別：位置級的切分、寫入時決定、看得到還沒抵達的請求、加上精度階與品質約束。

---

## 第 2 類 可恢復驅逐：丟掉的還能撿回來嗎？

> **這一類的共同想法**：傳統驅逐是「丟了就沒了」，但一個 token 的重要性會變。**早期看起來不重要的，後來可能又很重要。**
> 所以不要真的丟，留一條退路。

### ArkVale（NeurIPS'24，北京大學）
**一句話**：GPU 只留重要的 KV page，其他的在 CPU 留備份；每個 page 附一張「摘要卡」，需要時看摘要卡決定要不要召回。

- **想解決什麼**：只保留重要 token 的方法（H2O 之類）假設重要性不會變，但作者觀察到 decode 過程中重要性會漂移：被丟掉的 token 後來可能又變重要。
- **怎麼做**：
  1. KV 以 page 為單位。每個 page 填滿後，**非同步複製一份到 CPU** 當備份。
  2. 替每個 page 做一個很小的**摘要（digest）**：把這個 page 所有 key 包起來的「外框盒子」（bounding volume）。
  3. 每次算 attention 前，用摘要估每個 page 的重要性：重要的**召回**，不重要的驅逐，再挑前幾名參與計算。
- **實驗**：**只有 LongChat-7b-v1.5-32k**｜LongBench 6 個資料集 + passkey｜1× A100 80 GB
- **成果**：cache 預算 2K–4K 時準確度幾乎不掉；decode 延遲最多快 **2.2 倍**（平均 1.7），batch 吞吐量最多高 **4.6 倍**（平均 3.5）。
- **短板**：只測一個模型；每個 page 都要在 CPU 備份，prefill 期間的備份延遲難以藏起來（原文 §7 自承）；前兩層不套用；單請求。後來 Yandex 發現它的摘要在需要大量抽取資訊的任務上會誤判。
- **跟 Tiara 的關係**：Tiara 的 CPU 階就是「無損的備份與召回」。差別在於 ArkVale 用**當下的 query** 決定召回誰，Tiara 則預測**未來的請求**。

### QEvict（preprint'26）
**一句話**：不是「留或丟」二選一，而是三階：重要的留全精度、中等的壓成 2 bit、最不重要的才丟；中等的後來變重要，就解壓升回全精度。

- **想解決什麼**：傳統驅逐是不可逆的。作者定義了兩個診斷指標（Future Missed Mass、Global LIR），證明被丟掉的狀態後來常常又拿到大量 attention。
- **怎麼做**：以「視窗」（一段連續 token）為單位，依累積 attention 分數分三階：全精度 / 可恢復的 INT2 / 刪除。decode 時持續更新分數，被量化的視窗變重要就**反量化升回全精度**。
- **實驗**：Llama-3.1-8B、Mistral-7B、Qwen2.5-7B｜LongBench、RULER（32K）、GSM8K｜A100
- **成果**：用 eager 模式時 TTFT 只多 0.5%、TPOT 降 9.3%；用 FlashAttention-2 時峰值記憶體從 29.54 GB 降到 20.78 GB（-29.7%）。
- **短板**：只在 GPU 裡做精度分層，沒有 CPU、SSD；要定期把 attention 分數算出來，用 FlashAttention-2 時會拉低 decode 吞吐量（原文自承）。
- **跟 Tiara 的關係**：它做**精度那幾階**，ArkVale 做**位置那幾階**，兩者互補但沒人合在一起。Tiara 的梯子就是把兩者合起來。

---

## 第 3 類 用「當下的 query」挑 KV：這一步要讀哪些？

> **這一類的共同想法**：生成每個 token 時，真正重要的 KV 只有一小部分。拿**當下的 query** 去估哪些 KV 重要，只讀那些。
> **這一類的兩個結構性限制**（`main.tex` §3.4 的論點）：
> ① query 只能回答「這一步要什麼」，回答不了「幾分鐘後的另一個請求要什麼」；
> ② 要拿 query 去跟 key 比，前提是 key 還在手邊。已經丟到 SSD 或刪掉的 KV，就沒辦法這樣估。

### InfiniGen（OSDI'24，首爾大學）
**一句話**：在算第 i 層時，先用很小的代價「預演」第 i+1 層的 attention，猜出哪些 KV 重要，只把那些從 CPU 預取上來。

- **想解決什麼**：KV 卸載到 CPU 之後，每步都把全部 KV 搬回 GPU 太慢。
- **怎麼做**：
  1. **離線修改模型權重**：對 query 和 key 矩陣做 SVD「扭轉」（skewing），讓少數幾個欄位就能代表大部分資訊。
  2. 執行時只用那幾個欄位的「部分權重」做 attention 預演，猜出重要 token。
  3. **只預取重要 token 的 KV**。
- **實驗**：OPT-6.7B/13B/30B、Llama-2-7B/13B｜**準確度用 COPA、OpenBookQA、WinoGrande、PIQA、RTE 等短任務** + WikiText-2/PTB；速度用 PG-19｜RTX A6000、PCIe 3.0
- **成果**：比既有 KV 管理方法最多快 **3.00 倍**，準確度也更好。
- **短板**：準確度在短任務上量（敏感度實驗的輸入只有 1,920 token）；要改模型權重；模型是舊世代；Yandex 發現它在需要大量抽取資訊的任務上會漏掉該讀的 key。
- **跟 Tiara 的關係**：Tiara 計畫中的學術 baseline（`main.tex` 附錄）。它的「預演分數」可以當成 Tiara 預測器的一個輸入特徵。

### Quest（ICML'24，MIT Han Lab）
**一句話**：每個 KV page 記下 key 的最小值和最大值；用當下的 query 算出每個 page「最多能拿到多少 attention」，只讀前 K 名。

- **想解決什麼**：長上下文 decode 時，每一步都要讀全部 KV，太慢。而且哪些 token 重要，**要看當下的 query 而定**。
- **怎麼做**：每個 page 存 key 的逐維最小和最大值。估計時，每一維取 `max(q×max, q×min)` 加總，就得到這個 page 的重要性**上界**，再取 top-K pages 做 attention。
- **實驗**：LongChat-7B-32k、Yarn-Llama-2-7B-128k｜PG19、passkey、LongBench 6 個資料集｜RTX 4090（kernel 測試）
- **成果**：self-attention 最多快 **7.03 倍**，整體延遲降 **2.23 倍**，準確度損失可忽略。
- **短板**：**只省「讀取量」，不省「容量」**：全部 KV 仍然放在 GPU（全文沒有提到 CPU）；前兩層不能用；模型是舊世代。
- **跟 Tiara 的關係**：代表「請求內的稀疏 attention」。Tiara 處理的是**請求之間**的放置，兩者可以疊加。

### ShadowKV（ICML'25，字節跳動 / CMU）
**一句話**：key 做低秩壓縮後留在 GPU，value 搬到 CPU；每步用壓縮過的 key 挑出少數重要的，再把對應的 value 抓回來。

- **想解決什麼**：稀疏 attention 方法要嘛沒省到 GPU 記憶體（例如 Quest），要嘛把 KV 卸載到 CPU 以後 decode 變很慢。
- **怎麼做**：
  1. **key 用 SVD 壓成低秩**（rank 160），存在 GPU。
  2. **value 卸載到 CPU**。
  3. 把連續 8 個 token 當一組，以平均 key 當作「地標」挑重要的組，並保留少數離群值；再**即時重建**被選中的 KV。
- **實驗**：Llama-3-8B-1M、GLM-4-9B-1M、Llama-3.1-8B、Yi-9B-200K 等｜RULER、LongBench、NIAH｜A100
- **成果**：batch 最多大 **6 倍**，吞吐量最多高 **3.04 倍**，準確度不掉。
- **短板**：Yandex 發現，rank 160 的 key 壓縮在需要大量抽取資訊的任務上「no longer reliably select」；只有 GPU、CPU 兩層；單請求。
- **跟 Tiara 的關係**：同上，屬於請求內的稀疏檢索。它也說明了**「在 NIAH 上表現好」不代表真的好**。

---

## 第 4 類 重算 vs 傳輸：要用時，算回來還是搬回來？

> **這一類的共同想法**：一段 KV 不在 GPU 上時，有兩條路：從原文**重算**（吃 GPU 算力），或從下層**傳回來**（吃 PCIe / SSD 頻寬）。與其二選一，不如同時做、各做一部分。
> **這一類的共同時序**：都是「**已經要用了**，怎麼最快拿到」。Tiara 問的是更早的問題：「**未來可能要用**，現在該放哪」。

### Cake（ICML'25，密西根大學）
**一句話**：GPU 從前面往後算，I/O 從後面往前載，兩邊在中間會合。

- **想解決什麼**：prefix caching 從儲存層載入 KV 很慢（頻寬有限），全部重算又很貴。
- **怎麼做**：同一段上下文，**GPU 從第 1 個 chunk 開始往後算，I/O 從最後一個 chunk 開始往前載**，兩邊碰頭就完成。為什麼這樣分？因為越後面的 token 重算越貴（要看的前文更長），所以後面的交給 I/O、前面的交給 GPU 最划算。
- **實驗**：LLaMA 3.1-8B/70B、Long-Alpaca-13B 等｜**合成 prompt**（4K 到 16K，每 2K 取一點；原文說「只有長度重要」）｜2× A100、1× H100
- **成果**：TTFT 平均比純計算或純載入降 **2.6 倍**。
- **短板**：**I/O 頻寬是用算的**，原文：「We simulate the chunk I/O loading process by calculating the appropriate delay time」；假設所有 KV 事先都已存好在儲存層；短序列時反而輸給純計算。
- **跟 Tiara 的關係**：它證明了「**重算成本隨位置增長**」，這正是 Tiara 式 (2) 的依據。

### KVPR（ACL Findings'25，南加大）
**一句話**：KV 放在 CPU 上時，先把一部分「重算需要的原料」傳給 GPU 讓它開始算，剩下的 KV 同時用 PCIe 傳，兩件事重疊。

- **想解決什麼**：KV 卸載到 CPU 以後，PCIe 頻寬成了瓶頸；GPU 越快，這個瓶頸越明顯。
- **怎麼做**：CPU 先傳**一部分 activation**，GPU 用它重算那部分 KV；同時 CPU 傳剩下的 KV。profiler 會依輸入長度和硬體算出最佳切分點。
- **實驗**：OPT-6.7B/13B/30B｜沿用 FlexGen 的補齊資料（256/512/1024）｜A100、PCIe 4.0
- **成果**：decode 延遲最多降 **35.8%**，吞吐量最多升 **46.2%**。
- **短板**（原文 Limitations 自承）：只支援單 GPU；**不處理從 disk 載入**；只在一開始 profile 一次，假設硬體條件不變。
- **跟 Tiara 的關係**：ACL'26 綜述整理「重算 ↔ 傳輸」重疊的表格時，只列了它一個例子。

### HCache（EuroSys'25，清華）
**一句話**：不存 KV，改存每層的 hidden state（約為 KV 的一半大），要用時花一步矩陣乘法轉回 KV。

- **想解決什麼**：GPU 放不下的 KV，要嘛重算（很貴）、要嘛從儲存層載回（I/O 很慢）。有沒有中間路線？
- **怎麼做**：
  1. 存每層的**中間 activation（hidden state）**。要用時只需要做 K、V 的投影，就能還原 KV，**不必跑完整的 attention 和 FFN**。
  2. **bubble-free 排程**：讓一部分層用 hidden state 還原、一部分層直接載 KV，把 GPU 和 I/O 都餵滿。
  3. **chunk 式儲存**：解決「存的時候逐層、還原的時候逐 token」的版面衝突。
- **實驗**：Llama2-7B/13B、OPT-30B（上下文擴到 16K）｜ShareGPT4、L-Eval｜4× A100-40 GB、4 顆 Samsung PM9A3 SSD
- **成果**：TTFT 比 KV 卸載快最多 **1.93 倍**，儲存空間省 **1.92–2.40 倍**；比重算快最多 **5.73 倍**。
- **短板**：**只適用 MHA**。原文 §7：「HCache can currently support LLMs using the MHA mechanisms」。GQA 模型的 KV 本來就比 hidden state 小。實驗時為了公平比較，還關掉了 GPU 上的 KV 重用。
- **跟 Tiara 的關係**：`main.tex` 用它論證「hidden state 可以一步重建 KV，切斷重算的依賴鏈」。但 Tiara 用的兩個模型都是 GQA，**hidden state 反而比 KV 大**（見附錄 B）。

### CacheBlend（EuroSys'25 Best Paper，芝加哥大學）
**一句話**：RAG 時把好幾段文件各自預先算好的 KV 直接拼起來，只挑一小部分 token 重算，補上段落之間的交互。

- **想解決什麼**：prefix caching 只能重用「開頭」那一段。RAG 的輸入是好幾段檢索回來的文件拼在一起，第二段以後的 KV 都用不上。直接拼接又會漏掉段落之間的 cross-attention，品質會掉。
- **怎麼做**：重用每段預先算好的 KV，**只挑少數 token 重算**，局部更新拼起來的 KV。重算的小延遲可以跟 KV 的載入重疊，所以 KV 可以放在比較慢、比較大的儲存裝置上。
- **實驗**：Mistral-7B、Yi-34B、Llama-70B（後兩者 8-bit 量化）｜2WikiMQA、Musique、SAMSum、MultiNews｜2× A40、NVMe 4.8 GB/s
- **成果**：TTFT 降 **2.2–3.3 倍**，吞吐量升 **2.8–5 倍**，品質不掉。
- **短板**：只針對 RAG 的多段拼接；選擇性重算屬於有損；原文自承只適用 Transformer，也沒測更多模型和量化設定。
- **跟 Tiara 的關係**：代表「以 vLLM 為基礎、做跨請求重用」的那條研究血脈。這條血脈把 KV 視為「精確、不可分割」的東西。

---

## 第 5 類 學習式驅逐：用模型預測誰未來有用

> **這一類的共同想法**：不用手寫規則（最近用過的留、attention 分數高的留），而是**訓練一個小模型**去預測「哪個 KV 未來還會被用到」。
> **這一類的共同限制**（Tiara 最鋒利的一句）：動作都只有「**留或丟**」兩種，也沒有退路。
> 「**最強的未來預測器沒有安全網，而最完善的安全網沒有未來預測器。**」

### KVP — Learning to Evict from Key-Value Cache（ICML'26，Apple）
**一句話**：替模型的每個 KV head 各訓練一個小型強化學習 agent，只看 K 和 V 就幫 token 排序，排後面的丟掉。

- **想解決什麼**：既有驅逐方法用「最近」或「過去的 attention 分數」當代理，但這些只是「未來是否有用」的間接指標。
- **怎麼做**：
  1. 事先跑模型、收集生成 trace，得到每個 token「未來到底有沒有被用到」。
  2. **每個 KV head 一個 agent**（Qwen2.5-7B 是 28 層 × 4 個 KV head = **112 個 agent**，每個約 65 萬參數）。
  3. reward 考慮「在所有可能的 cache 預算下，排序好不好」，所以一次訓練就能適用各種預算。
  4. **不改 LLM**，只讀 K、V 和位置。
- **實驗**：Qwen2.5-7B、Phi-4｜RULER（到 128K）、OASST2-4k 多輪對話，另有 BoolQ、LongBench passage retrieval、GovReport 零樣本測試｜訓練用 8× H100（不到 30 分鐘）
- **成果**：在 RULER 和 OASST2 上明顯勝過強 baseline；在 10K 上下文下，**單層**壓縮只要 0.71 ms，完整模型的 prefill 是 404 ms。
- **短板**：原文自承**沒有端到端延遲**（只報 policy 本身的時間）；每個 head、每一層用同樣的預算；reward 用「未來 attention」當代理；動作只有留或丟；換模型就要重收 trace、重新訓練。
- **跟 Tiara 的關係**：`main.tex` 引用最多次的對照對象。Tiara 表 7 的第一列：112 個 MLP、2 個動作、成本不對稱「否」。**注意** `main.tex:1017` 引「0.71 ms 對 404 ms」時沒寫出「單層」。

### ForesightKV（ICML'26，人大高瓴等）
**一句話**：先算出「如果知道未來，最佳的驅逐是什麼」，拿它當標準答案訓練一個評分模型，再用強化學習修正。

- **想解決什麼**：推理模型（如 DeepSeek-R1）會生成很長的思考鏈，KV 跟著暴漲；既有驅逐方法抓不到 KV 之間的複雜依賴。
- **怎麼做**：
  1. **Golden Eviction**：用未來的真實 attention 分數，算出每一步最該丟的 KV。
  2. 用 pairwise ranking loss 讓小模型學會模仿這個排序。
  3. 把驅逐寫成馬可夫決策過程，用 **GRPO** 強化學習修正「丟掉後，低熵 token 的 loss 暴增」的問題。
- **實驗**：Qwen3-1.7B/4B、DeepSeek-R1-Distill-Qwen-7B（全是推理模型）｜**AIME2024、AIME2025**（數學競賽）｜A800
- **成果**：只用**一半的 cache 預算**就勝過既有方法；驅逐機制只占總時間 **2.7%**；在 1,024 預算、32K 生成長度下吞吐量最多高 **9.79 倍**。
- **短板**：**只針對數學推理的長生成**；要兩階段訓練（監督學習 + RL）；動作只有留或丟。
- **跟 Tiara 的關係**：表 7 中唯一「部分（單邊）成本不對稱」的工作；它**蒸餾真實的未來 attention**，是「預測 KV 必須看 KV」的先例之一。

### LookaheadKV（ICLR'26，Samsung）
**一句話**：想知道「模型等一下會回答什麼」才能判斷哪些 KV 重要，但真的生成太貴。所以訓練幾個「偷看未來」的特殊 token 來代替。

- **想解決什麼**：有一派方法先用 draft 模型產生一段假的未來回答，再用它估重要性。準確，但 prefill 開銷很大。
- **怎麼做**：在模型裡加入**可學習的 lookahead token**，並加上**只在這些 token 上啟動的 LoRA 模組**。訓練目標是讓它們預測出的重要性，和真實回答導出的重要性一致。
- **實驗**：Llama3.1-8B、Llama3.2-1B/3B、Qwen3-1.7B/4B/8B｜LongBench、RULER、MT-Bench
- **成果**：驅逐成本最多降 **14.5 倍**，TTFT 明顯變快；額外參數不到 **0.5%**。
- **短板**：原文自承運算資源有限，**沒測更大的模型**；**只做 prefill 階段的驅逐**，decode 階段留給 future work；要改模型（加 LoRA）。
- **跟 Tiara 的關係**：表 7 的一列（LoRA r=8、2 個動作、成本不對稱「否」）。

### TRIM-KV — Cache What Lasts（ICLR'26，JPMorgan AI Research + Yale）
**一句話**：每個 token 一產生，就由一個小 gate 打一個「保留分數」，這個分數之後隨時間自動衰減；空間不夠時丟分數最低的。

- **想解決什麼**：量化、卸載、啟發式驅逐，不是管理成本高，就是用 attention 當代理不可靠。
- **怎麼做**：替每一層、每個 head 加一個輕量的 **retention gate**。token 建立時評分**一次**，之後以指數衰減自動重新排序，不必再評。gate 以「凍結的原模型」蒸餾加上容量 loss 來訓練。
- **實驗**：Qwen3-1.7B/4B/8B/14B、R1-Distill｜GSM8K、MATH-500、AIME24、LongProc、LongMemEval、LongBenchV2、SCBench｜訓練用 4× H100
- **成果**：在低記憶體預算下持續勝過驅逐和可學習檢索的 baseline，**某些設定甚至超過完整 cache**（作者解釋為一種正則化）。
- **短板**：以數學推理為主；要訓練 gate，等於改模型。
- **跟 Tiara 的關係**：Tiara **無法直接採用**「建立時評分一次」的做法，因為 Tiara 的特徵（存取間隔、衰減計數器）每次被存取都會更新，一次性評分會立刻過時。另外，`refs.bib` 把它標為「venue UNVERIFIED」，但 PDF 首頁寫的是 **ICLR 2026**。

### Marconi（MLSys'25 Outstanding Paper Honorable Mention，Princeton、AWS 等）
**一句話**：為「Attention + SSM 混合模型」量身做的 prefix cache。存不存、丟不丟，同時看重用機率、能省多少計算、佔多少記憶體。

- **想解決什麼**：混合模型（例如 Jamba）的 SSM 層是**就地更新狀態**，不能像 KV 那樣「退回到前面某個位置」。結果是只有完全相同的前綴才能命中，快取裡塞滿大而少用的項目。
- **怎麼做**：
  1. **准入**：依不同命中情境的分類，預測一個項目未來被重用的機率，機率低就不存。
  2. **驅逐**：除了最近使用時間，也看「命中一次能省多少 FLOP ÷ 佔多少記憶體」。
- **實驗**：自建 7B 混合模型、Jamba-1.5-Mini｜LMSys、ShareGPT、SWE-Bench（agent）｜8× A100-40 GB
- **成果**：token 命中率最多高 **34.4 倍**（TTFT 少 71.1%，也就是 617 ms）。
- **短板**：**只適用混合模型**，用在純 Transformer 上就退化成一般的 prefix cache（我們的判讀）；到達間隔是人工設定的；沒有自己的 CPU、SSD 階。
- **跟 Tiara 的關係**：Tiara 特徵矩陣中覆蓋率最高的既有工作（四項成本中的三項），也是最需要防守的對手。但它只在 GPU 上，結構上就不可能有「傳輸成本」這一項。

### SAECache（preprint'26，北京大學等）
**一句話**：不同「身分」的 token 重用率差很多（系統提示最常被重用、推理鏈最少），所以依 token 類型分佇列，用不同規則驅逐，而且規則的參數在線上自動學。

- **想解決什麼**：LRU 只看「最近何時用過」，把所有 token 一視同仁。
- **怎麼做**：
  1. **多佇列**：依前綴 block 的類型分到不同佇列，各有各的優先度公式。
  2. **語意權重**：依驅逐的回饋，在線上學習每種 token 類型值多少。
  3. **所有參數都在線上自動調**，不必手動設定。
- **實驗**：只用 Qwen2.5-1.5B 在 A40 上**驗證能整合進 vLLM**；主要結果是 **trace 驅動的模擬**｜ShareGPT、LMSys、Chatbot-Arena ＋ 4 個合成負載
- **成果**：TTFT 比生產等級的 baseline 快 **1.4–2.7 倍**。
- **短板**：原文自承，單一小模型時 vLLM「rarely fills the KV cache to capacity」，所以改用模擬；在 Chatbot-Arena（多為單輪對話）上**比最好的 baseline 慢 12–34%**；**自己的數字前後矛盾**（見附錄 B）。
- **跟 Tiara 的關係**：Tiara 預測器的 `token_type` 特徵就是依據它的觀察。但該引用的是 **42 倍**，不是 756 倍。

---

## 第 6 類 品質、壓縮與放置最佳化：有損的時候怎麼取捨？

> **這一類的共同想法**：只靠無損搬運不夠快，就壓縮 KV（量化、丟 token），但要控制品質損失。
> 也有工作把「放哪」直接寫成數學最佳化問題。
> **Tiara 在這一類的定位**：把品質寫成**硬性約束**（不能低於 1−ε），而不是可以用速度換掉的權重。

### KVServe（SIGCOMM'26，中科院計算所等）
**一句話**：KV 在 prefill 機和 decode 機之間走網路時要先壓縮，而且依當下的負載、頻寬、品質要求，自動挑最合適的壓縮組合。

- **想解決什麼**：PD 分離以後，KV 成了跨網路的大包裹。既有壓縮法都是固定設定，但不同任務的最佳壓縮法差很多（例如 KIVI 在 Qasper 最好，在 GSM8K 卻墊底）。
- **怎麼做**：
  1. 把各種壓縮手法拆成可以自由組合的模組。
  2. 離線用**貝氏最佳化**搜出「延遲 × 壓縮率 × 準確度」三維的 Pareto 候選（搜尋開銷降 50 倍）。
  3. 線上用延遲模型加上 bandit，在品質約束下挑選組合，並修正離線與線上的落差。
- **實驗**：Qwen2.5-7B/32B、Llama-3.1-8B｜訓練用 GSM8K、HumanEval、Multi-News、Qasper；另以 2WikiMQA、HotpotQA 測泛化｜RTX 5090、RTX 4090、Pro 6000、H100
- **成果**：PD 分離下 JCT 最多快 **9.13 倍**；KV 分離式服務下 TTFT 最多降 **32.8 倍**。
- **短板**：**只選壓縮組合，不決定 KV 放哪一層**；品質門檻固定為 97% 相對準確度。
- **跟 Tiara 的關係**：Tiara 所知**唯一**在頂級會議上把「模型品質」寫成硬性約束的 KV 工作。Tiara 參考它讓 ε 可以操作的做法（離線 profiling 加上 bandit）。

### EvicPress（preprint'25，芝加哥大學）
**一句話**：對每一段上下文的 KV，同時決定「壓多少」和「放哪一層」；對壓縮很敏感的少壓一點、放高一點。

- **想解決什麼**：既有工作不是只做驅逐，就是只做壓縮，沒有**聯合**決定。
- **怎麼做**：定義一個同時計入「品質」和「延遲」的**效用函數**；profiler 定期替每段上下文的每種「壓縮 × 層級」組合打分；再用快速的啟發式演算法重新安排各層的 KV。
- **實驗**：Llama-3.1-8B、Qwen2.5-14B、LongChat-7B、Mistral-7B、Qwen3-30B（MoE）｜LongBench 的 555 段上下文，**問題由 GPT-5 產生**（每段 100 題，一半用來訓練 profiler）｜1× H100、80 GB 記憶體、800 GB SSD
- **成果**：在同等品質下 TTFT 最多快 **2.19 倍**（12 個資料集、5 個模型）。
- **短板**（原文 §8 自承）：只在單節點測；假設各層頻寬已知且穩定；DRAM 夠大或各層頻寬相近時，優勢就消失。另外假設遠端儲存無限大；在 musique 上輸給 keydiff+LRU。
- **跟 Tiara 的關係**：特徵矩陣中「記憶體成本」的代表。它把品質當成**效用裡的一個權重**，Tiara 則把品質當作**約束**。

### AdaptCache（SOSP BigMem workshop'25，同一團隊）
**一句話**：EvicPress 的前身。對每筆 KV 決定壓縮演算法、壓縮率、放 DRAM 還是 SSD，目標是讓更多命中發生在 DRAM。

- **想解決什麼**：KV 太多，大多數命中都落在很慢的 SSD 上。
- **怎麼做**：效用 = **(預估的未來命中頻率) ×（α × 品質 − 大小 ÷ 頻寬）**，寫成多選背包問題，用 greedy 求解。**未來命中頻率用「過去的命中頻率」估計**。
- **實驗**：只有 Llama-3.1-8B｜LongBench 6 個資料集、1,100 段上下文，品質曲線用 GPT-4o 產生的問題，Poisson 到達｜1× A100、100 GB DRAM、400 GB SSD（1 GB/s）
- **成果**：同等品質下延遲省 **1.43–2.4 倍**；同等延遲下品質高 **6–55%**。
- **短板**：3 頁的 workshop 初步結果；只有一個模型；只看過去；品質是權重，不是約束。相對於 prefill，TTFT 降 56% 的代價是「within 15% quality drop」。**全文沒有列出實際的候選壓縮法和壓縮率**，只在概念上提到 token dropping 與 quantization，baseline 用 KIVI（2-bit）與 StreamingLLM。
- **跟 Tiara 的關係**：它的「未來頻率 = 過去頻率」正是 Tiara 預測器想要改進的地方。

### LeoAM（preprint'25）
**一句話**：在一台只有一張消費級 GPU 的個人電腦上跑長上下文。KV 依重要性切成大小不一的塊，分放 GPU、CPU、硬碟；硬碟上只讀「摘要」決定要不要搬。

- **想解決什麼**：因為隱私，想在本機跑長上下文，但 24 GB 的卡放不下 KV，只好落到硬碟。問題是評估重要性的開銷大，硬碟又慢。
- **怎麼做**：
  1. 依各層 attention 的偏斜程度，把 KV 切成**大小不一的 chunk**。
  2. 每個 chunk 在硬碟上附一份**很小的摘要**，只讀摘要來判斷要不要搬。
  3. 動態壓縮（INT4）加上流水線。
- **實驗**：LongChat-7B-32k、Yarn-Llama-2-13B-128k、OPT-6.7B｜**準確度用 COPA、RTE、PIQA、OpenBookQA** + PG-19；速度用 LongBench｜RTX 4090、120 GB 記憶體、SSD 7 GB/s
- **成果**：推論延遲平均快 **3.46 倍**，大 batch 時最多 **5.47 倍**；原文宣稱準確度下降「不到 1%」。
- **短板**：**「不到 1%」是在短的 few-shot 任務上量的**，不是長上下文任務；模型都是舊世代的 MHA；不同記憶體大小是用限制比例「模擬」出來的。
- **跟 Tiara 的關係**：和 Tiara 的平台 A（RTX 3090 24 GB）場景最接近。`main.tex` 說它是「唯一給出數值上界者」，引用時應註明是在短任務上量的。

### OrbitFlow（VLDB'26，POSTECH）
**一句話**：每個請求的 KV「哪幾層留在 GPU、哪幾層放 CPU」，用一個小型整數規劃即時求解，並在生成過程中持續修正。

- **想解決什麼**：長上下文服務的記憶體需求一直在變，固定的卸載策略會造成大量 CPU→GPU 搬運，延遲暴衝、違反 SLO。
- **怎麼做**：
  1. 用輕量的 **ILP** 決定每個請求哪些層的 KV 留在 GPU（受容量限制）。
  2. 生成中若計畫不再最佳，就依回饋重新調整。
  3. 負載太重時，暫緩記憶體用量特別大的請求。
- **實驗**：LLaMA3-8B（1× RTX A5000 24 GB、PCIe 3.0）、LLaMA3-70B（4× A6000）｜**從 ShareGPT 抽樣 + Poisson**｜batch 4、每批不超過 32K
- **成果**：TPOT 和 TBT 的 SLO 達成率分別提升 **62%** 和 **66%**，P95 延遲降 **38%**，吞吐量最多高 **3.3 倍**。
- **短板**：**無損設計**（原文：「KV offloading does not compromise accuracy」），所以 ILP 裡沒有品質變數；只有 GPU、CPU 兩層；原文承認公開的長上下文生產 trace 很少，所以負載是合成的。
- **跟 Tiara 的關係**：同樣是「最佳化求解放置」，但它的動作空間沒有有損的那幾階，也沒有 SSD 和丟掉重算。

### Het-Mem — Fang 等人（IEEE CAL'25，RPI / IBM）
**一句話**：HBM 加上 CPU 那側的 DRAM 組成異質記憶體；把「KV 放哪」寫成數學問題，推導出「如果完美放置，最多能多快」。

- **想解決什麼**：GH200 這類新硬體讓 GPU 能高速存取 CPU 記憶體，兩者的頻寬差距已縮小到一個數量級以內。怎麼放 KV，才能把兩邊的頻寬一起用滿？
- **怎麼做**：把放置問題形式化，用**模擬退火**配合「已知的未來存取」近似出**理論上界**。原文明說：「Rather than proposing a specific scheduling policy」。
- **實驗**：**純模擬**。GH200 的參數，但 HBM 刻意設成 24 GB｜只用 LLaMA-3.1-8B｜只有一種負載：NarrativeQA 約 30K 的 prompt，再 decode 10K
- **成果**：理論上界比靜態放置高 **5.87 倍**的吞吐量，代表「執行期還有很大的改善空間」。
- **短板**：整篇都是模擬；沒有提出策略；一個模型、一種負載；沒有 SSD。
- **跟 Tiara 的關係**：**Tiara 自我定位最直接的參照**。它留下的「策略空缺」就是 Tiara 要填的；Tiara 的 oracle 則把它延伸到 SSD 和丟掉重算。

---

## 第 7 類 KV 量化：一個數字要用幾個 bit？

> **這一類決定了 Tiara 梯子上「精度那三階」（BF16 / FP8 / INT4）長什麼樣。**

### KIVI（ICML'24，Rice 等）
**一句話**：KV 可以壓到 2 bit，關鍵在於 **K 要沿「通道」方向分組量化、V 要沿「token」方向分組量化**，因為兩者的離群值分布不一樣。

- **怎麼做**：研究熱門模型的 KV 數值分布後發現，K 的離群值集中在少數固定的通道，V 則沒有這種結構。於是 K 逐通道量化、V 逐 token 量化；還湊不滿一組的最新 token 先用全精度存。不必調參數。
- **實驗**：Llama-2-7B/13B、Falcon-7B、Mistral-7B、LongChat｜GSM8K、LongBench、NIAH 等｜A100
- **成果**：峰值記憶體（含權重）少 **2.6 倍**，batch 可大 **4 倍**，吞吐量高 **2.35–3.47 倍**，品質幾乎不變。
- **短板**：只在 GPU 裡；最新的 token 要保留全精度區。
- **跟 Tiara 的關係**：Tiara 附錄計畫的學術 baseline 之一；它說明了「梯子只是示意，量化器可以替換」。

### KVTuner（ICML'25，華為諾亞方舟等）
**一句話**：每一層對量化的敏感度不同，所以逐層決定 K、V 各用幾 bit，並在離線時先搜好。

- **怎麼做**：從理論分析 attention 模式和量化誤差的關係，以及為什麼 **K 通常比 V 重要**；離線以多目標最佳化搜出每一層的（K bit, V bit）組合，線上直接套用。
- **實驗**：Llama-3.1-8B、Qwen2.5-3B～32B、Mistral-7B｜以 GSM8K 等數學推理為主
- **成果**：Llama-3.1-8B 平均 **3.25 bit** 近乎無損；對量化敏感的 Qwen2.5-7B 要 **4.0 bit**；吞吐量比 KIVI-KV8 高 **21.25%**。
- **短板**：每個模型都要離線搜尋一次；近乎無損的 bit 數會隨模型改變；評測偏重數學。
- **跟 Tiara 的關係**：佐證 Tiara 在 §7 自承的限制：Tiara 把 K 和 V 綁在一起降精度，而文獻顯示分開配置更好。

---

## 第 8 類 量測研究與綜述：別人怎麼看這整個領域

### Bottlenecks — Understanding Bottlenecks for Efficiently Serving LLM Inference With KV Offloading（MLSys'26）
**一句話**：KV 卸載到 CPU 以後，prefill 反而會從「算力瓶頸」變成「PCIe 瓶頸」。作者推導出一個臨界值 **κ_crit** 來判斷什麼時候會發生。

- **怎麼做**：定義 κ_crit = (每 token 的 prefill FLOP ÷ 每 token 的 KV 大小) × (PCIe 頻寬 ÷ GPU 算力)。**「已快取 token ÷ 新 token」的比值一旦超過 κ_crit，就會卡在 PCIe**。這個式子還可以拆成模型因子 κ_M 和硬體因子 κ_HW。
- **實驗**：Llama-3.1-70B、Qwen3-235B-A22B｜用 ShareGPT、NarrativeQA、FinQA 統計真實的快取/新 token 比值｜H100、B200（prefill-only 微基準）
- **成果**：κ_crit 介於 **1 到 76**，但真實負載的比值中位數，ShareGPT 為 **100**、NarrativeQA 為 **5,000**、FinQA 為 **10,000**，遠超臨界值；**99% 的延遲花在傳輸**，GPU 平均只用到額定功耗的 **28%**。
- **短板**：只測 prefill；prompt 是用重複片段拼出來的；MLA 模型量不出來，留給 future work；「99% 在傳輸」的前提是容量壓力存在。
- **跟 Tiara 的關係**：⚠️ **它的 κ_crit 和 Tiara 的 κ 在代數上是同一個量**（見附錄 B）。`main.tex` 目前只引了「99% 延遲在傳輸」。

### Yandex / YAKV — KV Cache Offloading for Context-Intensive Tasks（preprint'26，Yandex + HSE）
**一句話**：現有 KV 卸載方法都在「大海撈針」這類簡單任務上測；換成「要從上下文抽出大量資訊」的任務，它們就壞了。

- **怎麼做**：做了一個新的評測 **Text2JSON**（從原文抽出結構化的 JSON，需要讀很多地方），再拿現有方法去測，並分析失敗原因：
  1. **key 的低秩壓縮不夠準**（例如 ShadowKV 的 SVD），選不出該讀的東西；
  2. **用「地標」代表一組 token 不可靠**（例如 ShadowKV 的平均 key、ArkVale 的外框盒子），誤判很多。
  據此提出一個更簡單的替代方案 YAKV。
- **實驗**：Llama-3.1-8B、Qwen3-4B/30B/32B 等｜Text2JSON 等需要大量抽取資訊的任務，另有 NIAH、RULER、LongBench｜H200、A100、B200
- **成果**：在 Llama 3 和 Qwen3 上都觀察到現有方法大幅掉分；YAKV 在多個模型家族上明顯改善準確度。
- **短板**（原文自承）：YAKV 只是最小可用的系統，沒有預取、也沒有自適應預算；只適用 full attention 的模型。
- **跟 Tiara 的關係**：Tiara 據此**把大海撈針降級成 sanity check**，品質約束改用「最差任務」的分數，不用跨任務平均。

### KV Survey — Towards Efficient LLM Serving: A Survey on System-Aware KV Cache Optimization（ACL Findings'26）
**一句話**：把 KV 系統研究整理成三個維度：**時間**（什麼時候執行、怎麼排程）、**空間**（放哪、怎麼搬）、**結構**（怎麼表示、要不要保留）。

- **對 Tiara 有用的三處**：
  1. **觀察 O7**：「KV cache 壓縮雖然熱門，卻跟其他行為各做各的，是 **a missed opportunity for co-design**」。
  2. **挑戰 C5**：呼籲在共享預算下**聯合決定驅逐、卸載、預取**。這份清單裡**沒有重算**。
  3. **附錄 G.2**：提出「中間狀態」的想法，例如「在 GPU 上壓縮」「在 CPU 上壓縮」「在 CPU/SSD 上只留摘要」「在 GPU 上可回收」，把策略寫成狀態之間的轉移。**這和 Tiara 的梯子非常接近**，而且它只是提出方向，沒有實作。
- 它整理「重算 ↔ 傳輸重疊」的表格時，只列了 KVPR 一個例子。
- **跟 Tiara 的關係**：Tiara 缺口論證的外部佐證。但附錄 G.2 代表「多階狀態」的想法已經被點名過了，Tiara 的新意要放在**實作、成本模型與品質約束**上。

---

## 附錄 A：一表總覽

| 論文 | 場所 | 類別 | 一句話 | 原文成果（擇要） | 最大短板 |
|---|---|---|---|---|---|
| Bidaw | FAST'26 | 1 多層 | 算和存互相知道；用上一輪回答長度預測；改存 tensor 6 | 延遲 ↓3.58×、吞吐 ↑1.83× | 靠真實時間戳；tensor 6 只對 MHA 有利 |
| FlexGen | ICML'23 | 1 多層 | LP 決定權重、KV 在三層各放多少 | 175B 單卡 1 token/s | 離線批次、合成資料 |
| CachedAttention | ATC'24 | 1 多層 | 多輪 session 的 KV 存三層，靠排程佇列預取 | TTFT ↓87% | 只跟全部重算比；時間戳是模擬的 |
| Mooncake | FAST'25 | 1 多層 | 叢集級 PD 分離＋池化 KV 倉庫 | Kimi 多處理 75% 請求 | dummy model、需要 RDMA |
| KVDrive | preprint'26 | 1 多層 | 三層加 SSD 版面最佳化加流水線 | 吞吐 ↑1.74× | 只有單請求，沒有跨請求重用 |
| Tutti | preprint'26 | 1 多層 | GPU 直接對 NVMe 發 I/O | TTFT ↓78.3% | 只管搬，不管放哪 |
| LMCache | preprint'25 | 1 多層 | 開源 KV 快取層加控制 API | 吞吐 ↑15× | trace 只用長度分布 |
| Strata | OSDI'26 | 1 多層 | GPU 協助 I/O 加 cache-aware 排程（SGLang HiCache） | 吞吐比 vLLM-LMCache ↑5× | 策略是 LRU、不考慮重算 |
| MTDS | C&IS'26 | 1 多層 | 邊緣三層；依預測時間選全載、部分載或不載 | TTFT ↓>25% | 整段一起決定；只看佇列 |
| ArkVale | NeurIPS'24 | 2 可恢復 | CPU 備份加 page 摘要召回 | decode ↑2.2× | 只測 1 個模型 |
| QEvict | preprint'26 | 2 可恢復 | 全精度 / INT2 / 丟，三階可升回 | 峰值記憶體 ↓29.7% | 只在 GPU 內 |
| InfiniGen | OSDI'24 | 3 query | 預演下一層 attention 再預取 | ↑3.00× | 準確度在短任務上量；要改權重 |
| Quest | ICML'24 | 3 query | page 的 min/max key 估上界取 top-K | attention ↑7.03× | 不省容量 |
| ShadowKV | ICML'25 | 3 query | 低秩 key 在 GPU、value 在 CPU | 吞吐 ↑3.04× | 在資訊密集任務上選不準 |
| Cake | ICML'25 | 4 重算傳輸 | 前面算、後面載，中間會合 | TTFT ↓2.6× | I/O 是模擬的 |
| KVPR | ACL F'25 | 4 重算傳輸 | 傳部分 activation 重算，同時傳其餘 KV | 延遲 ↓35.8% | 不處理 disk |
| HCache | EuroSys'25 | 4 重算傳輸 | 存 hidden state 一步還原 KV | TTFT ↑1.93×、空間 ↓1.92–2.40× | 只對 MHA 有利 |
| CacheBlend | EuroSys'25 | 4 重算傳輸 | RAG 多段 KV 拼接、少量重算 | TTFT ↓2.2–3.3× | 只適用 RAG |
| KVP | ICML'26 | 5 學習式 | 每個 KV head 一個 RL agent 排序 | 勝過強 baseline | 沒有端到端延遲 |
| ForesightKV | ICML'26 | 5 學習式 | 模仿最佳驅逐＋GRPO | 半預算勝出 | 只做數學推理 |
| LookaheadKV | ICLR'26 | 5 學習式 | 可學習的前瞻 token＋LoRA | 驅逐成本 ↓14.5× | 只做 prefill |
| TRIM-KV | ICLR'26 | 5 學習式 | 建立時打分、隨時間衰減 | 有時超過完整 cache | 以數學為主、要訓練 |
| Marconi | MLSys'25 | 5 學習式 | 混合模型的 prefix cache 准入與驅逐 | 命中率 ↑34.4× | 只適用混合模型 |
| SAECache | preprint'26 | 5 學習式 | 依 token 類型分佇列、線上學權重 | TTFT ↑1.4–2.7× | 主要是模擬；數字矛盾 |
| KVServe | SIGCOMM'26 | 6 品質 | 依服務狀態自動選壓縮組合 | JCT ↑9.13× | 不決定放哪 |
| EvicPress | preprint'25 | 6 品質 | 聯合決定壓縮與放置 | TTFT ↑2.19× | 假設頻寬穩定、單節點 |
| AdaptCache | BigMem'25 | 6 品質 | 壓縮法、壓縮率、層級一起選 | 延遲 ↓1.43–2.4× | workshop、只看過去 |
| LeoAM | preprint'25 | 6 品質 | 消費卡上的 GPU/CPU/硬碟三層 | 延遲 ↓3.46× | 準確度在短任務上量 |
| OrbitFlow | VLDB'26 | 6 品質 | ILP 決定每層 KV 的去留 | 吞吐 ↑3.3× | 無損，所以沒有品質變數 |
| Het-Mem | IEEE CAL'25 | 6 品質 | 形式化放置問題並給上界 | 上界 5.87× | 純模擬、沒有策略 |
| KIVI | ICML'24 | 7 量化 | K 逐通道、V 逐 token，2 bit | 吞吐 ↑2.35–3.47× | 只在 GPU |
| KVTuner | ICML'25 | 7 量化 | 逐層決定 K、V 的 bit 數 | 3.25 bit 近乎無損 | 每個模型都要離線搜尋 |
| Bottlenecks | MLSys'26 | 8 量測 | 推導 κ_crit，卸載後卡在 PCIe | 99% 延遲在傳輸 | 只測 prefill |
| Yandex/YAKV | preprint'26 | 8 量測 | 資訊密集任務讓卸載方法現形 | 現有方法大幅掉分 | 最小系統 |
| KV Survey | ACL F'26 | 8 綜述 | 時間 / 空間 / 結構三維整理 | O7、C5、G.2 | — |

---

## 附錄 B：讀原文時抓到、跟我們有關的事

這些是**待處理的問題**，還沒改進 `main.tex`。依 CLAUDE.md，論文修改要另開 paper-writing 工作階段處理。

1. **Bottlenecks 的 κ_crit 就是我們的 κ。** 它的 κ_crit = (F_pf / B_kv) × (BW / C_eff)，其中 F_pf = 2 × 參數量；我們 `main.tex:266` 的 κ = t_recompute / t_transfer，用 2N_params FLOP ÷ FLOPS 對上 KV 大小 ÷ 鏈路頻寬。展開後相同。它還拆了模型因子與硬體因子，也討論了 MoE。**我們仍然獨有的部分**：κ 隨 block 的絕對位置增長、把 κ 當作決策門檻 p* = 1/(1+κ)、MI300X 與 RTX 3090 的實測。
2. **SAECache 的 756 倍應改為 42 倍。** 它的 Table 1 是 92.3% ÷ 2.2% ≈ 42 倍，引言寫 44 倍，只有摘要和圖 4 的說明寫 756 倍。`main.tex:957` 還寫了「極冷的工具輸出」，但它的表上工具輸出是 23.0%，最冷的是推理鏈（2.2%）。
3. **HCache 的論證要加上「僅限 MHA」。** 依 `results/m1_capacity/model_configs.json` 的 config 計算（非量測）：Qwen2.5-7B 每層每 token 的 hidden state 是 3,584 維、KV 是 2×4×128 = 1,024 維，hidden state 是 KV 的 **3.5 倍**；Llama-3.1-8B 是 4,096 對 2,048，**2 倍**。
4. **LeoAM 的「不到 1%」是在短任務上量的。** 引用時要註明。
5. **KVP 的「0.71 ms 對 404 ms」是單層的時間。** 原文寫「a single-layer compression takes 0.71ms」，`main.tex:1017` 沒有寫出「單層」。
6. **KV Survey 附錄 G.2 已經提出「多階中間狀態」的想法。** 這是 Tiara 梯子的概念前身，Related Work 應該引用並說明差異（我們有實作、成本模型與品質約束）。
7. **Strata 已經有「寫入時決定」**（selective-write-through，被存取的次數超過門檻才寫）。Tiara 講寫入時決定，要強調依據是**位置成本**而不是**次數**。
8. **MTDS 已經有「算還是載」的選擇和「未來命中」的預測**，是最接近的對手。差別在於：它整段一起決定、只看得到佇列裡的請求、在讀取時決定、沒有精度階。
9. **我們自己也沒跑 ShareGPT。** 可以預先準備理由：它沒有時間戳，而 Bidaw §5.3 示範了「用 Poisson 補時間戳，會讓時間相關的策略失效」。
