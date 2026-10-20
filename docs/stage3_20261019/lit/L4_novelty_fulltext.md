# L4　查新：六篇（＋兩篇補充）全文核對，判定第三階段四個候選有沒有被做掉

> 寫於 2026-10-19（UTC）。只讀、只上網；沒跑 GPU、沒 import torch／vllm、沒改任何既有檔案。
> **怎麼讀的**：每篇都用 curl 打開 arXiv abs 頁（確認標題、作者、版本、Comments 欄），再下載 PDF，用 tiara venv 的 PyMuPDF（只 import `fitz`）抽成文字逐段讀。**不是** WebFetch 的小模型摘要。
> **頁碼**一律是 **PDF 實體頁**（抽字時的第幾頁），不是論文自己印的頁碼。
> **讀不到的**：圖裡的數值（只讀文字，沒看圖）。所以下面的數字都是**內文或表格**寫出來的；只出現在圖上的沒有引用。
> **規則**：不同硬體的數字只並列，不排名；〔判讀〕是我的解讀，不是原文。

---

## 0. 先講結論

| 候選 | 判定 | 最大威脅 | 一句話 |
|:--|:--|:--|:--|
| **C-I1** 多 session 同時還原 | **部分做過** | CacheFlow（批次 DP 聯合決定誰算誰載）＋ Cascade（依剩餘預算排序＋還原或重算） | 「聯合決定」已有；但**讀過的論文都假設計算與搬運互不干擾**（CacheFlow 附錄 A.1 p.14、InferCept p.6、Stony Brook 報告 p.22 明寫；Cascade 用保護係數 γ 吸收），而且 CacheFlow 最佳化的是**整批的完成時間**、不是每個 session 的 P95；Cascade 是模擬器。 |
| **C-I3** agent 暫停的四選一＋κ | **部分做過** | Cascade（HBM 留／從 DRAM 或 NVMe 還原／重算，依 SLO 預算）；Tokencake（預測載回）；補充的 Stony Brook 技術報告（κ 跨 GPU 變動、模型 1,764 組硬體） | 四個動作（含 SSD 與重算）在 Cascade 裡**都有**，但觸發是 SLO 預算、不是工具暫停，也不是 κ；「κ 依硬體而變」本身**已經不是新觀察**（Stony Brook、EfficientAgent、Bottlenecks）。 |
| **C-F** 復用不划算的邊界 | **部分做過**（只有單點） | EfficientAgent（容量讓卸載翻轉，repo 已有卡）；Cascade 圖 4（深層命中反而違反 SLO）；Stony Brook（命中率高時「全部還原」失敗） | 「復用比不復用差」已在多篇以**單點或模擬**出現；**前綴重疊 × 容量 × 併發三維、在真實引擎上畫邊界、並比較平均 vs goodput／P95 目標會不會翻轉**——這六篇都沒做。 |
| **C-C1** 選擇性還原 | **這六篇都沒做** | CacheSmith（query-aware 打分，但選的是「要重算哪些 token」，其餘**全部載入**）；L1 已列的 IMPRESS／ContiguousKV／CacheTune 仍是最大威脅 | Cascade 的「只還原 M 位元組」是依預算、不是依相關性，而且剩下的是**重算**、不是留在下層待補。 |

**順帶的重要發現**（不屬於四個候選，但影響 Tiara 主張）：Stony Brook 的技術報告 FSL-26-02（2026-08，非 arXiv、非同儕審查）在 H200 與 RTX A5000 上實測「還原 vs 重算」，並用模型掃 6 張 NVIDIA GPU × 6 代 PCIe × SATA～Gen5 NVMe 共 1,764 組，結論是「**固定長度下，硬體選擇可讓還原時間差 2–3 個數量級**」「幾乎每張 GPU 的重算曲線都和還原時間分布相交」〔原文 p.2〕。這和 Tiara「κ 跨硬體變動 32 倍」的主張高度重疊；它沒有 AMD、沒有爭用（〔原文 p.19〕明寫）。

---

## 1. 論文卡片

### 1.1　Cascade〔全文〕

- **出處**：M. Adnan（UBC，Microsoft Azure Research 實習）, R. Mahapatra, P. J. Nair, D. Berger, P. Zardoshti（NVIDIA）, R. Fonseca, E. Choukse, "Cascade: Exploiting SLO-Aware latency budget for fair and high goodput LLM inference serving," arXiv **2608.06557v1**（2026-08-06；abs 頁已開；Comments 欄空白 → venue NOT_FOUND）。內文沒有程式碼連結。
- **問題**：同一個 SLO 下的請求，扣掉自己的執行時間後，剩下的「延遲預算」差很多；排隊、搶佔、從深層搬 KV 都在花同一份預算。現有系統把排程和 KV 管理分開做〔p.1–2〕。
- **方法**：
  - 預算 B_r＝類別的 TTFT SLO − 估計 TTFT L_r；L_r＝(實際負載下估計＋最大負載下估計)／2 × γ（γ≥1，每個模型離線定一個值，「吸收估計誤差、**記憶體匯流排爭用**、排隊變異」）〔p.6 Eq.3–4〕。
  - 排程：剩餘預算 B_rem＝B_r −（現在 − 到達時間），**剩最少的先跑**；預算為負的放第二佇列，每批保留固定比例給它〔p.5–6 Eq.5〕。
  - **還原或重算**：從第 k 層搬 x 位元組的時間 T_k(x)＝Σ 每段鏈路固定成本＋x·Σ 1/B_eff（B_eff＝「負載下的有效頻寬」；NVMe 要走 NVMe→DRAM→HBM 兩段）〔p.7 Eq.6〕。反解出預算內最多能搬的位元組 M_{r,k}，**搬到 M 為止，剩下的前綴改用 GPU 重算**〔p.7 Eq.7、Alg.1〕。
  - 搶佔：HBM 不夠時搶**剩餘預算最多**的 prefill 請求；被搶的 KV「**塞得進它的剩餘預算就卸到 DRAM，否則重算**」〔p.8〕。
  - 實作：vLLM＋LMCache（HBM／DRAM／NVMe）；每層內 LRU 回收〔p.7–8〕。
- **決策變數**：請求順序；每個請求從第 k 層還原多少位元組（其餘重算）；搶誰；被搶者卸 DRAM 或重算。
- **硬體**：**評測在模擬器裡**：擴充 Vidur 模擬多副本叢集與多層 KV，參數用 **GB200 NVL72** 上實測的搬運 profile〔p.8；p.9「all policies and CASCADE are evaluated … within our extended Vidur simulator」〕。模型 Qwen-2.5-72B（刻意用 MHA 設定）、Llama-3-70B、Llama-3-405B，TP=4，權重 NVFP4、KV FP8〔p.8 Table I〕。叢集 9／6／12 個實例〔p.9〕。
- **是否實測干擾**：**否**。計算與搬運的干擾沒有量，只用 B_eff 和 γ 吸收〔p.6–7〕。
- **是否多層**：是（HBM／DRAM／NVMe；同一請求的命中可跨層，圖 6）〔p.5–7〕。但每個請求**各自**用 B_eff 算 M_{r,k}，沒有跨請求分配鏈路〔p.7〕。
- **是否含 SSD**：是（本地或網路 NVMe，經 GPU-Direct Storage）〔p.3 圖 3 說明、p.5〕。
- **指標**：goodput（TTFT 與 TPOT 都達標的請求之輸出 token/s，對 FCFS 正規化）、SLO 違反率、Jain 公平指數（依輸入長度分 6 組）、TTFT 分位數〔p.4、p.9〕。
- **主要數字（條件）**：
  - 負載：阿里雲百鍊公開 trace 2 小時（ChatBot／Tool&Agent／Coder／Reasoning）組成 10 種混合〔p.8 Table II〕；SLO＝單請求隔離執行的 TTFT p90 10 倍、TPOT p90 5 倍〔p.8〕。
  - 平均 goodput 2.4×（對 FCFS）、違反少 40%，每個 trace 都最高；Mixed 2.7×；輕載 trace 1.1–1.5×，原文說來自預取而非重排〔p.9–10〕。
  - Qwen-2.5-72B 違反數 1.5×10⁵ → 4×10³〔p.10〕。各長度組違反率 2–15%（SJF 在 p99+ 組 65–72%）〔p.10–11〕；Jain 指數 0.98–1.0〔p.11〕。
  - 48 QPS（Qwen-72B）：goodput 1.5×、違反 14%；FCFS／EDF 0.05×、>90%〔p.11〕；p95 TTFT 3.5 s〔p.11–12〕。7 個實例的 goodput 高於 FCFS 用 9 個實例〔p.11〕。
  - 動機圖 4：64K 前綴、Qwen-72B，從 NVMe 取會超過 TTFT SLO（即使用 LMCache 批次預取）〔p.4〕——〔判讀〕這是一個「命中反而違反 SLO」的單點。
- **限制**〔判讀，附依據〕：
  - 模擬器評測；對照只有 FCFS／EDF／SJF，**沒有 Cake、CacheFlow、LMCache 的還原策略當對照**〔p.9〕。
  - 「還原 M 位元組、其餘重算」**沒說哪一段載、哪一段算、兩者是否重疊**〔p.7〕；引言說會比較重算成本〔p.2〕，但 Eq.7／Alg.1 只用預算〔p.7〕。
  - Tool&Agent 類的輸入很短（p50 574、p99 6,294 token）；Mixed 的 p99 是 13,792〔p.8 Table II〕。長 context（≥32K）只出現在動機圖。
  - 只有一種硬體（GB200 profile），沒有跨硬體比較。
  - Cascade 的預取是對**已到達**的請求觸發（Alg.1 `TriggerPrefetch`）〔p.7〕，不是預測「何時回來」的提前載入。

### 1.2　CacheSmith〔全文〕

- **出處**：X. Ma, D. Lee（POSTECH）, A. Shah（Microsoft Research，現 Mirendil）, M. Jeon（POSTECH）, "CacheSmith: Accurate RAG Serving under Latency SLOs with Adaptive KV Reuse," arXiv **2610.18925v1**（2026-10-13；abs 頁已開；Comments 空白 → venue NOT_FOUND；分類 cs.AR）。
- **問題**：RAG 的 chunk 快取（每個 chunk 單獨 prefill、不是前綴）缺跨 chunk 的 attention；選擇性重算（CacheBlend、EPIC、ProphetKV、FusionRAG）用固定設定，不適應請求差異和動態延遲預算〔p.1–2〕。
- **方法**：
  - 把選擇性重算拆成 WarmUp（前 w 層完整 prefill，**不讀快取**）→ Probe（再 p 層累積選擇訊號：KV 偏差或 query–context 互動 QCI）→ Patch（重算前 r 比例的 token）；設定＝(S, w, p, r)〔p.5〕。
  - Planner：依 attention 集中度（normalized entropy）選訊號；用離線 profile 的「需求分布」選排名深度；**transfer-aware 成本模型**：逐層走，層要等它的 KV 到了才能算；I/O 卡住時用空檔多重算一些早期 token；執行中頻寬掉了（例：I/O 爭用）就重新規劃〔p.7〕。
  - Scheduler：把佇列依 **deadline 與 context 長度分桶**，一次規劃一批、給批次預算；突發時用「部分 KV 配置」（只留即將執行的層）提高併發〔p.7–8 Alg.1〕。
  - Runtime：依設定產生搬運計畫（WarmUp 不搬、Probe 搬該層 KV、Patch 只搬沿用的 token），**依 deadline 順序送出**〔p.8〕；實作上「同一 deadline 的搬運在裝置的 **4 條 I/O stream** 上並行送出」〔p.9 §6〕。
- **決策變數**：批次組成、批次預算、訊號 S、w、p、每請求覆蓋率 r_i、KV 視窗。
- **硬體**：2 張 NVIDIA RTX PRO 6000（96 GB）、AMD EPYC 9554、Samsung 9100 PRO NVMe（PCIe Gen5×4）；Llama-3.1-8B（1 卡）、Qwen3-32B（2 卡 TP）；vLLM 0.10.2〔p.9〕。
- **是否實測干擾**：**否**（計算對搬運的干擾）。它處理的「干擾」是**同批 lockstep 執行的請求互相拖累**〔p.2、p.4–5〕；I/O 爭用只用「頻寬掉了就重新規劃」處理〔p.7〕。
- **是否多層**：快取放主記憶體**或** SSD，分開當不同設定（10、6、3 GB/s，後兩者是對 SSD 限速）〔p.9〕；不是多層混在同一佇列。
- **是否含 SSD**：是（chunk KV 預先存到 SSD，39–707 GB）〔p.9〕。
- **指標**：quality-gated SLO attainment（QSA＝TTFT 達標**且**品質過門檻的比例）、準確率、P99 TTFT、吞吐、stall 比例〔p.6、p.10〕。
- **主要數字（條件）**：PassageRetrieval、MuSiQue（LongBench）＋RULER 式 VT、MV；context 最長約 15K（MuSiQue）；Poisson 0.2–0.4 req/s、Gamma CV 2–3；基準 SLO 0.75 s〔p.9–10〕。
  - 96 個設定中 93 個高於「事後挑的最佳固定設定」，QSA +2–27.5%〔p.10〕；P99 TTFT 最多 −86%〔p.2〕。
  - 頻寬 10→3 GB/s：基線 12–72% 時間卡在等搬運，CacheSmith 3–7%；PR 準確率 69%→92%（慢鏈路讓它多做 WarmUp 重算）〔p.10〕。
  - 訊號：10% 覆蓋率時，PR 上 KV 偏差比 QCI 高 20%、RULER-VT 上 QCI 高 17%——**沒有單一訊號通吃**〔p.4〕。
- **限制**〔判讀〕：只做 RAG prefill 的非前綴 chunk；不是多輪 session 還原；到達率很低（≤0.4 req/s）；只在 NVIDIA；QCI 用自寫未調校的 Triton kernel（原文自承延遲偏悲觀）〔p.9〕。
- **和候選的關係**：
  - C-I1：「依 deadline 分組、多 stream 送出」與「transfer-aware 逐層成本模型」是**排序／分組**的前作，但對象是 RAG 而不是 N 個回來的 session，也沒有「這個 session 全載、那個全算」的跨請求取捨。
  - C-C1：它打分是為了**選要重算的 token**；沒被重算的 token **全部載入**〔p.8〕——**沒有「不載入、留在下層待補」**。但「Probe 層要先把該層 KV 載進來才能打分」〔p.7〕這個依賴，C-C1 一樣會遇到。

### 1.3　CacheFlow（v2）〔全文〕

- **出處**：S. Nian, H. Shen, Z. Wu, J. Fang, Q. Feng, F. Lai（UIUC、CMU）, "CacheFlow: Efficient LLM Serving via Automated 3D-Parallel KV Cache Restoration," arXiv **2604.25080v2**（v1 2026-04-28，v2 2026-09-26；abs 頁已開；Comments「17 pages, 17 figures」；PDF 頁首「Preprint」→ venue NOT_FOUND）。程式碼「論文被接受後釋出」〔p.10〕。
- **問題**：還原被當成「整個請求算或載」的粗粒度選擇；忽略模型內部（token、層、GPU）的平行性，以及**批次內請求對 GPU 與 I/O 的爭用**〔p.1–3〕。
- **方法**：
  - 階梯分割：每個層區塊 i 重算前 j_i 個 chunk、載入其餘，j_1 ≥ … ≥ j_L；全部相等＝Cake（token 維），前幾區塊全算＝HCache 式（層維）〔p.5 Eq.1〕。
  - 成本模型：重算 T_comp＝Σ f(j_i)，載入 T_load＝Σ d(j_i)/**B（單一載入頻寬）**；「兩者並行，所以還原在較慢的那個完成時結束」，目標 min max(T_comp, T_load)〔p.5 Eq.2〕。
  - **批次 DP**：把每個（請求, 層區塊）當一個項目，D_m[x]＝前 m 項省下 x 位元組時的最小**總**重算時間，最後選 x* 平衡總重算與總載入〔p.6 Eq.5〕。
  - f(j) 從最近的服務紀錄**線上量**、內插〔p.6〕；GPU 維用快取的邊界 activation 讓各分片同時還原〔p.4–5〕。
- **「計算與載入互不干擾」的假設寫在哪**（SOTA_ANALYSIS §2.2 要補的）：
  - **附錄 A.1「Model and assumptions」，PDF p.14**：「One compute resource and one independent loading resource overlap **without interference**, with sufficient buffers.」同段還有「Costs are exact, nonnegative, and additive」。
  - 同頁 A.1 開頭：理論分析「does not yet cover … **non-additive batched costs, or estimated costs**」。
  - 主文 §3.2「Modeling recomputation and transfer」，**p.5**：T_load 只有一個 B，且「These two run concurrently」——主文的隱含版本。
  - 批次版 Eq.5（**p.6**）把各請求的重算時間**直接相加**，等於假設批次重算沒有額外交互作用。
- **決策變數**：每個（請求, 層區塊）的重算邊界 j；批次共用的重算／載入平衡點 x*。**沒有請求順序**（誰先）這個變數。
- **硬體**：NVIDIA A100、H100（單卡與多卡；Qwen3.5-122B-A10B 用 8×H100）；頻寬設定 80／40／10 Gbps（原文說對應 InfiniBand／Lambda Lab SSD 讀取／AWS 跨節點），**預設 10 Gbps**；原文沒說是實接裝置還是限速〔p.7〕。附錄算例 B＝1.25 GB/s〔p.16〕。vLLM v0.29.0、LMCache v0.5.5rc5、Cake 當對照〔p.7〕。
- **是否實測干擾**：**否**（假設不干擾，見上）。它報了 GPU SM 89%、I/O 89% 同時使用（Qwen3.5-9B，圖 5）〔p.8〕；〔判讀〕在 1.25 GB/s 的預設頻寬下，鏈路遠沒飽和，這不構成「互不干擾」的檢驗。
- **是否多層**：**否**，只有一條載入頻寬 B〔p.5〕。
- **是否含 SSD**：只以頻寬標籤（40 Gbps＝「SSD 讀取速度」）出現〔p.7〕，沒有獨立的 SSD 層。
- **指標**：TTFT（平均、p95、p99）、TTLT、GPU／I/O 使用率、規劃成本與預測誤差〔p.7–9〕。
- **主要數字（條件）**：
  - 平均 TTFT 快 2.24–3.00×（對現有方法）；p95 1.77–3.02×、p99 1.48–2.95×〔p.8〕。
  - 對 Cake：40 Gbps 2.63×、80 Gbps 2.40×〔p.9〕；H100 2.69×、A100 2.07×（Qwen3.5-35B-A3B、OSWorld 2.0、10 Gbps）〔p.9〕；batch 1／4／8 時 2.8–2.9×（H100、Llama-3.1-8B）〔p.9〕。
  - 預測：92% 重算、98% 載入的預測誤差在 10% 內；95% 的還原，重算與載入在 10% 內同時完成〔p.9〕。DP 成本中位 0.019%、p99 0.34% TTFT〔p.8〕。
  - GPU 數 1→2→4：平均 TTFT 1.59→0.96→0.67 s〔p.9〕。線上（SWE-Bench＋Mooncake 到達時間）p95 TTFT 與 TTLT 約 1.64×〔p.9〕。
  - 理論：與理想最佳解差 O(1/N)；Qwen3.5-9B、10 Gbps、32K／64K／128K 時上界 6.52%／3.26%／1.63%〔p.7、p.16〕。
- **限制**：所有設定都報加速，沒有失效區域；只有 NVIDIA；單一載入頻寬；目標是**整批的完成時間**，不是每個請求的尾延遲；batch 最大 8〔p.9〕。
- **和 SOTA_ANALYSIS §2.2 舊卡的核對**：舊卡的數字（2.24–3.00×、對 Cake 2.40–2.63×、92%／98%／95%、O(1/N)、不處理寫入、單一頻寬、所有設定都加速、程式碼未釋出）**都和 v2 一致**。補：互不干擾假設的位置（上）、批次大小只到 8、預設頻寬 1.25 GB/s。注意：CacheFlow 的 κ 是「每 token 重算時間的下界」〔p.7、p.16〕，和 Tiara 的 κ 意義不同（舊卡已註）。

### 1.4　Where Should the KV Cache Live?〔全文〕

- **出處**：S. D. Tumkur, J. Iyer, M. Simhadri, S. P. Kumar, S. K. Kumar, R. Nampelly（Vizuara）, arXiv **2609.16215v1**（2026-09-14；abs 頁已開；cs.AI；venue NOT_FOUND）。repo 卡 `docs/research_20261009_explore/cards/A_WhereKVLive.md`。
- **問題**：KV block 該放 HBM／DRAM／SSD 哪一層、何時升降、預取值不值得〔p.1–3〕。
- **方法**：離散事件模擬器，用 random-forest 執行時間預測器校準；策略＝recency、reuse-frequency、predicted-reuse（原文發現和 recency 逐位元相同）、EWMA 預測；預取 lookahead 掃描〔p.1、p.3–4〕。chat 用 6 個併發 session（單一 session 時所有淘汰策略結果相同）〔p.4〕。
- **決策變數**：每個 block 的層、升降時機、預取深度。
- **硬體**：**全模擬**；層形狀 1:8:64；batch 1、單 GPU；內文沒寫模型〔p.4–5〕。
- **是否實測干擾**：否。模型裡「投機抓取與需求抓取共用同一條被爭用的鏈路」〔p.6、p.8〕——這是鏈路爭用的**模型設定**，不是實測，也不含計算對搬運的干擾。
- **是否多層／含 SSD**：是／是（模擬）。
- **指標**：PCIe 搬移量（GB）、TTFT（P99）、TPOT 與 stall、每 GPU session 數、每 session 成本。
- **主要數字（條件，皆模擬）**：分層讓 session 數 ×73.02、成本 ÷62.04，＝層容量倍數 1+8+64〔p.1、p.5 Table III〕；chat：recency 27.38 GB vs reuse-frequency 62.99 GB（2.30×）；agent、文件 QA 由 reuse-frequency 勝（1.60×、1.63×）〔p.5〕；stall <1% TPOT〔p.6〕；tiered P99 TTFT 17.32／56.23／66.27 ms〔p.5〕。
- **「prefetch 不划算」的條件**（核對結果）：
  - **指標分兩個**：摘要說「連讀得到未來的 oracle 在**搬移量**上也沒有一格贏過不預取」〔p.1；arXiv 摘要原句「never beats no prefetch on migration traffic」〕；§V-E 說的是「因果預測器在**任何一格**都沒拿到 oracle 可得增益的正份額；oracle 本身在**超過一半**的格子比不預取**慢**」〔p.6〕。〔判讀〕所以「沒有一格贏」只對**搬移量**成立；就延遲而言，oracle 在少於一半的格子裡是贏的。repo 舊卡把兩者合寫成一句，應拆開。
  - **成立條件**：模擬器、batch 1、單 GPU、decode 算力瓶頸、合成工作負載（原文自承「依構造有利於對應策略」）、**投機抓取與需求抓取搶同一條鏈路**〔p.6–8〕。原文自己說：預取「只有在有空閒頻寬時才會移除 stall，而 batch 1 時大致沒有」，並把「有空閒頻寬可藏的預取模型」列為下一步〔p.8〕。
  - 〔判讀〕它的預取是 decode／恢復時的 **block 級 lookahead**，不是「工具暫停期間、依預測回來時間提前載入」。C-I3 的 warm-up 正是在**閒置期**用**空閒鏈路**搬，落在原文明說沒測的那一區，**不被這篇推翻**。
- **另一個相關結果**：緊 HBM（32 block）＋固定 DRAM（64 block）、SSD 從 8 加深到 1,024 block：chat 與 agent **在任何深度都達不到** GPU-only（遇 miss 重算）的 P99 TTFT；文件 QA 在深度 64 剛好打平（餘裕 0.000 ms）〔p.7–8〕——〔判讀〕模擬裡「深層 SSD 不如重算」的一個單點，和 C-F 有關。
- **限制**：全模擬、合成負載、batch 1、stall 的 DRAM／SSD 分攤是按命中比例估的〔p.8〕。

### 1.5　InferCept §3–4〔部分：p.1–7 全讀；評測 §5 只讀前段〕

- **出處**：R. Abhyankar, Z. He, V. Srivatsa, H. Zhang, Y. Zhang, "InferCept: Efficient Intercept Support for Augmented Large Language Model Inference," arXiv **2402.01869v2**（abs 頁已開；v1 2024-02-02、v2 2024-05-30；ICML 2024，L3 已核 PMLR）。
- **問題**：工具／外部互動造成的「攔截」：現有系統當成請求結束、回來重算；重算佔總 forward 時間 37–40%〔p.1、p.5〕。
- **方法（§3–4）**：
  - 三種做法的 GPU 記憶體浪費（記憶體 × 時間）：Discard＝T_fwd(C)·C·M ＋ T_fwd(C)·C_other·M〔p.5 Eq.1〕；Preserve＝T_INT·C·M〔p.5 Eq.2〕；Swap＝2·T_swap(C)·C_batch·M〔p.5 Eq.3〕。
  - Swap 逐層 pipeline、分多個 iteration；每步可「免費」搬的 token 數 N_i 由 **T_swap(N_i)＝T_fwd(B_i)** 決定（swap budget）〔p.6 §4.1〕。
  - 重算切成 chunk，每塊大小＝GPU 飽和點 S − 執行中的 token 數，和 decode 混跑〔p.6 §4.2；Eq.4〕。
  - 決策：每個被攔截的請求算 min(WastePreserve, WasteChunkDiscard)，**依浪費由大到小給 swap budget**，其餘依 Eq.5 選留或丟〔p.7 §4.3〕。恢復：swap 佇列與等待佇列都依原到達時間 FCFS〔p.7〕。
  - 攔截時長估計 T̂＝t_now − t_call，達 oracle 的 93%〔p.7 §4.4〕。
- **隱含的不干擾假設**：§4.2 開頭「Unlike Swap, which consumes GPU-CPU-link bandwidth and **can be hidden from foreground GPU tasks**」〔p.6〕；swap budget 的定義本身就假設搬運完全藏在 forward 後面、不拖慢它。
- **決策變數**：每個暫停請求留／丟／換；swap 頻寬怎麼分給換出與換入。
- **硬體**：A100（GPT-J-6B 1 卡；Vicuna-13B 1 卡與 2 卡 TP；Llama3-70B 4 卡 TP）〔p.7 §5〕。
- **是否實測干擾**：否。**是否多層**：否（GPU／CPU）。**含 SSD**：否。
- **主要數字**：攔截時長平均 9×10⁻⁵ s（數學）到 28.6 s（chatbot，估計值）；攔截時 context 平均 753–2,185 token〔p.3 Table 1〕；Discard 浪費 27% GPU 資源、Swap 26% 且 >25% 時間在等 swap；Preserve 時超過 60% 的時間有近一半 GPU 記憶體被暫停請求佔住〔p.5–6〕；比 vLLM 多撐 1.6–2× 負載、完成請求數 >2×〔p.2〕。
- **限制**〔判讀〕：context 約 1–2K（重算便宜、搬運量小）；只有 CPU 一層；成本離線 profile 一次；目標是記憶體浪費、不是延遲 SLO。

### 1.6　Tokencake（v5）的載回規則〔部分：§1–7 讀過；§5 空間排程只略讀〕

- **出處**：Z. Bian, F. Wu, **Z. Li**, T. Ma（Alibaba）, Y. Zhuo（PKU）, "TokenCake: A KV-Cache-centric Serving Framework for LLM-based Multi-Agent Applications," arXiv **2510.18586**（abs 頁已開；**v5 2026-09-28**；v1 2025-10-21）。**abs 頁 Comments 欄寫「Accepted at EuroSys '27」**——L3 說「venue 只見於 repo 描述」，現在可改為「arXiv Comments 欄自述」。作者名單 v5 比 L3 記的 v2 多一位（Zhuoran Li）。
- **問題**：多 agent 應用裡，等工具的 agent 把 KV 佔在 GPU（時間上閒置），而關鍵 agent 的 KV 被擠掉（空間爭用）〔p.1–2〕。
- **卸出規則（§4.2，p.5–6）**：T_transfer＝T_offload＋T_upload（隨 block 數線性、用目標平台實測的每 block 常數校準）；T_window＝T_fc − T_transfer ≤ 0 → 不卸；把視窗換成 token 容量，等待佇列裡要有 first-fit 塞得進的請求才卸（Alg.1）；另有四個硬拒絕（CPU 容量、暫停太短、沒人塞得進、GPU 壓力低於門檻）與加權分數〔p.6〕。**重算不是選項**。
- **載回規則（§4.3「Predictive Upload」，p.6）**：
  - 時長預測：每種工具的歷史 EWMA，與使用者給的估計以 β 混合〔p.5 Eq.1〕。
  - 接近預測完成時開始**漸進**載回；工具比預測早回來就立刻載回；觀測值回饋給預測器〔p.5〕。
  - 優先序 P_upload＝I（Spatial Scheduler 的重要度）＋U（離預測完成時間越近越急）〔p.6〕。
  - 載回預算 B_upload＝max(0, 空閒 GPU block − max(0, 關鍵等待請求需求 − 共享池空閒))〔p.6 Eq.3〕；每步最多預留剩餘缺額的一半〔p.6 Eq.4〕。
- **硬體（v5）**：Qwen2.5-14B／1×A100 80GB、Qwen2.5-32B／1×**H20** 96GB、Qwen2.5-72B／2×H20 TP2；CPU 卸載區 100 GB〔p.9〕。〔注意〕L3 依 v2 記的是「A100＋**H200**」，v5 已改。
- **是否實測干擾**：否。原文只說「高併發下不必要的搬移會吃 PCIe 頻寬、佔 GPU block」〔p.12 §7.6〕。**多層**：否（GPU／CPU；Mooncake 是遠端 KV 對照）。**SSD**：否。
- **主要數字**：A100 PCIe、4,096 token（256 block）：卸出 32.0 ms、載回 31.7 ms，重算 1,815 ms → **重算慢 28.5×**；1,024–5,120 token 間重算慢 26.8–37.5×〔p.12 §7.6〕。〔注意〕L3 依 v2 記「搬運約 60 ms、重算約 9,000 ms、150 倍」，v5 的數字不同。gpu_mem_utilization 0.5 時延遲最多 −47.06%〔p.10〕；Qwen2.5-72B Code-Writer D2 1.0 QPS −20.7%〔p.9〕；KV 使用率 85.7–87.0% vs vLLM 69.9–74.1%〔p.10〕。
- **預測誤差的敏感度**（和 C-I3 的 warm-up 直接相關）：工具時長乘性雜訊 0→0.25→0.5，相對 agent-only 排程：−14.8%→**+8.3%（變慢）**→−3.4%；原文說中等誤差最難（剛好通過閘門的錯誤時機最傷）〔p.11–12 §7.5〕。
- **原文承認依硬體而變**：「On hardware with higher memory bandwidth (e.g., H100), reduced transfer overhead causes the gate to reject most offload candidates」，要在目標平台重新驗證門檻〔p.12〕。
- **限制**：context 1–5K；工具時長是重播錄下的分布〔p.9〕；只單節點、≤2 卡〔p.13〕。

### 1.7　補充 A：Stony Brook 技術報告 FSL-26-02〔部分〕

- **出處**：A. H. Najafi Zadeh, "Understanding and Optimizing KV-cache Management for Long-Context LLM Inference," Research Proficiency Exam, Stony Brook University, **Technical Report FSL-26-02**, 2026-08。PDF：<https://www.fsl.cs.stonybrook.edu/docs/llm/amir26kvcache-rpe.pdf>（已下載開過）。**非 arXiv、非同儕審查**。模擬器 <https://github.com/sbu-fsl/KVC-TierSim>〔p.18 註〕。讀了 p.2、p.16–19、p.22、p.26、p.31。
- **做了什麼**：把「還原 vs 重算」當成 I/O 問題；在 **H200 與 RTX A5000**（都接 SATA SSD）上跑 ShareGPT chatbot 實測〔p.16–17〕；再建效能模型掃 **1,764 組硬體**（V100、RTX A5000、RTX 6000 Ada、A100、H100、H200 × 6 代 PCIe × SATA～Gen5 NVMe）〔p.21 Table 4.2，經 grep 定位〕。
- **結論**：固定長度下硬體選擇讓還原時間差 2–3 個數量級；幾乎每張 GPU 的重算曲線都和還原時間分布相交〔p.2〕。實測：H200 上除了最小的總 context 以外，重算都比還原快；A5000 上兩者幾乎可互換〔p.17〕。把一次還原**依比例分給儲存路徑與 GPU**（＝Cake 式切分），達標組態比例比「全還原」「全重算」高 2–10×〔p.2、p.26〕；命中率越高，「全還原」越容易失敗（因為搬得越多）〔p.26〕。
- **假設**：「Contention. Each configuration is evaluated in isolation, with no competing store traffic and no queueing across concurrent requests.」〔p.19〕；「Because restoration and recomputation consume **disjoint resources**, they can proceed concurrently」〔p.22〕。
- **未來工作**（原文）：把兩條路徑當**佇列**而不是固定速率資源、放進「排程中的請求群體」評估、接上 admission control；把多層（DRAM、NVMe、遠端）推廣成 max-flow〔p.31〕。
- 〔判讀〕對 Tiara 主張「κ 跨硬體變動」是**直接重疊**（只是沒有 AMD、沒用 κ 這個名字、沒有精度軸）。對 C-I1 則是**支持**：它把「併發、爭用、多層」明列為沒做的事。

### 1.8　補充 B：EfficientAgent（repo 已有卡，這裡只核 abs）

- arXiv **2609.33762v1**（2026-09-27；abs 頁已開）。repo 卡 `docs/research_20261009_explore/cards/A_EfficientAgent.md`〔部分，p.1–9〕。
- abs 原句：同一個 coding-agent 負載，卸載「speeds up one deployment, slows down another, and changes nothing on a third, even where loading a token back is several times cheaper than recomputing it」；「Across three GPU types and two models, offloading pays off when the GPU has little compute per byte of host bandwidth and the host tier holds the working set」。
- 〔判讀〕這是「**每 byte 主機頻寬的算力**（≈ κ 的倒數）決定卸載划不划算」的跨 GPU 實測版本，對 C-I3 與 C-F 都是威脅；它用 vLLM 0.13.0＋LMCache 真實重播（repo 卡），但只有 GPU／CPU 兩層、只看容量與寫入准入，沒有重算與 SSD 在同一決策裡。

---

## 2. 候選 × 論文判定表

符號：**已做**＝候選的核心問題與方法已被這篇涵蓋；**部分**＝有重疊的機制或結論，但缺候選的關鍵條件；**沒做**＝無關或只沾邊。

### C-I1　多 session 同時還原（各自 Cake vs 聯合決定；GPU 與 PCIe 共用；GPU 忙時 H2D 變慢）

| 論文 | 判定 | 證據位置 | 說明 |
|:--|:--|:--|:--|
| CacheFlow | **部分（最大威脅）** | §3.2 Eq.5 p.6；附錄 A.1 p.14；p.9 圖 15 | 已做「整批請求的（請求, 層區塊）聯合決定算或載、共用 GPU 與 I/O」。缺：干擾（明寫假設不干擾）、順序（沒有「誰先」）、每 session 的 P95（目標是批次 makespan）、多層、真實背景負載、AMD；batch ≤8；預設頻寬 1.25 GB/s。 |
| Cascade | **部分** | §IV-C Eq.5–7、Alg.1 p.6–7；p.9 | 已做「依剩餘預算排序（誰先）＋每請求決定還原多少、其餘重算」，多層 DRAM／NVMe。缺：模擬器；干擾只用 γ 與 B_eff 吸收；各請求獨立算 M，沒有跨請求分配鏈路；沒說載與算是否重疊。 |
| CacheSmith | 部分（排序面） | §5.2 Alg.1 p.7–8；§6 p.9 | 依 deadline 分組、同 deadline 搬運並行走 4 條 stream；逐層 compute–transfer 重疊模型。對象是 RAG chunk，不是 session 還原；沒有跨請求「誰算誰載」。 |
| InferCept | 部分（弱） | §4.1 p.6；§4.3 p.7 | 多個暫停請求共用 swap 頻寬（依浪費排序）；明寫 swap「can be hidden from foreground GPU tasks」。短 context、單層。 |
| Tokencake | 沒做 | §4.3 p.6 | 有載回優先序（重要度＋急迫度）與預算，但不涉及重算、不涉及 GPU 爭用。 |
| WhereKVLive | 沒做 | p.6、p.8 | 模擬中投機與需求抓取搶同一鏈路；batch 1、單 GPU。 |
| Stony Brook（補） | 沒做（且明列為未來工作） | p.19、p.22、p.31 | 單請求切分；明寫無爭用、兩路資源「disjoint」；把佇列化、admission、多層 max-flow 列為未來工作。 |

**總判定：部分做過。** 「聯合決定誰算、誰載」已被 CacheFlow（批次 DP）佔掉，「誰先」被 Cascade（預算排序）與 CacheSmith（deadline 分組）佔掉；**「計算與搬運互相拖慢」這個前提，所有讀過的論文都假設不存在**（CacheFlow p.14、Stony Brook p.22、InferCept p.6 明文；Cascade 用 γ 吸收）。

### C-I3　agent 工具暫停的四選一（GPU 留／卸 CPU／卸 SSD／丟掉重算）＋κ＋預測預取

| 論文 | 判定 | 證據位置 | 說明 |
|:--|:--|:--|:--|
| Cascade | **部分（最大威脅）** | p.2；Eq.7 p.7；p.8 搶佔段 | 動作空間其實很接近：留 HBM、從 DRAM 或 NVMe 還原／預取、重算；被搶佔者「塞得進預算就卸 DRAM，否則重算」。但觸發是 SLO 預算與 HBM 壓力，**不是工具暫停**；決策量是預算，**不是 κ**；只有 GB200 profile；模擬器；Tool&Agent 類 prompt 很短（p50 574）。 |
| InferCept | 部分 | Eq.1–5 p.5–7 | 三選一（留／丟重算／換 CPU）＋成本模型，專為工具攔截。缺 SSD、長 context、跨硬體、κ 實測。 |
| Tokencake | 部分 | §4.1–4.3 p.5–6；§7.5 p.11–12；§7.6 p.12 | 卸 CPU 與否＋**預測載回**（＝C-I3 的 warm-up）已做，含預測誤差敏感度；原文承認門檻依硬體而變（H100 會拒絕多數卸出）。重算不是選項；無 SSD；context ≤5K。 |
| WhereKVLive | 沒做（不構成反證） | §V-E p.6；§VI p.8 | 「預取不划算」的條件是 batch 1、投機與需求搶同一鏈路；閒置期用空閒頻寬的預取正是原文列為沒測的情況。 |
| CacheFlow | 沒做 | — | 不處理暫停期間的去留。 |
| CacheSmith | 沒做 | — | 不處理暫停。 |
| Stony Brook（補） | 部分（κ 跨硬體這半） | p.2、p.17、p.26 | 還原 vs 重算在 6 張 GPU 間翻轉（模型＋2 張實測）；無 agent、無暫停、無 AMD、無爭用。 |
| EfficientAgent（補，repo 卡） | 部分（κ 跨硬體這半） | abs；repo 卡 p.4–8 | 三種 GPU 上「每 byte 主機頻寬的算力」決定卸載划不划算；兩層、無重算選項。 |

**總判定：部分做過。** 四個動作都在某篇出現過（Cascade 幾乎全有），預測載回在 Tokencake 有、含誤差分析；「κ 依硬體而變」已被 Stony Brook、EfficientAgent（以及 repo 已知的 Bottlenecks κ_crit）以不同形式說過。**沒找到**：以「工具暫停」為觸發、把四個動作放進同一規則、用實測 κ（含長 context 下依位置的重算成本）做門檻、並證明**同一規則在 NVIDIA 與 MI300X 上選不同動作**的論文。

### C-F　復用不划算的邊界（前綴重疊 × KV 容量 × 併發；平均 vs goodput／P95）

| 論文 | 判定 | 證據位置 | 說明 |
|:--|:--|:--|:--|
| Cascade | 部分（單點） | §III-B 圖 4 p.4；Takeaway 2 p.4–5 | 64K、Qwen-72B 從 NVMe 取前綴會讓 TTFT 超過 SLO；「接受每個深層命中會把命中變成違反」。只是一個點，沒畫邊界；也沒比較目標函數。 |
| Tokencake | 部分（單點） | §7.5 p.12 | 某負載下把壓力門檻調到拒絕所有卸出，延遲反而 −32%（107.5 s）；原文強調不是「不卸最好」，而是卸出要有人用得上。 |
| WhereKVLive | 部分（模擬單點） | §V-G p.7–8 | 緊 HBM 下加深 SSD，chat／agent 的 P99 TTFT 始終達不到「遇 miss 重算」的基準。 |
| Stony Brook（補） | 部分（模型） | p.26 | 以 P95 門檻的「達標比例」為指標，命中率越高「全還原」越失敗——一條「復用量 × 硬體」的邊界，但是模型、單請求、無併發。 |
| EfficientAgent（補） | 部分（實機，最大威脅） | abs；repo 卡 p.7 | 真實 vLLM＋LMCache 重播：同一負載在三個部署上卸載加速／變慢／無效；容量決定翻轉。缺：前綴重疊與併發兩軸、goodput／P95 目標。 |
| CacheFlow | 沒做 | p.8–9 | 所有設定都報加速，沒有失效區域。 |
| CacheSmith | 沒做（反向的一點） | p.10 | 「事後最佳固定設定」在 8／96 個設定輸給 Full Prefill（排隊輕、SLO 寬時）——RAG chunk 復用不划算的一個點。 |
| InferCept | 沒做 | — | — |

**總判定：部分做過（只有零散單點）。** 「復用／卸載有時比重算差」已經是多篇都看到的現象；**三維邊界地圖（重疊 × 容量 × 併發）在真實引擎上**、以及**同一組設定在平均延遲與 goodput／P95 兩種目標下最佳解會不會翻轉**，這幾篇都沒有做。

### C-C1　選擇性還原（只載與新問題相關的 block，其餘留在下層、需要時補；與 Cake 結合）

| 論文 | 判定 | 證據位置 | 說明 |
|:--|:--|:--|:--|
| CacheSmith | 沒做（機制相關，最近） | §3.2 p.4；§5.1 p.6–7；§5.3 p.8 | query-aware（QCI）打分、在預算下調整；但打分是為了選**要重算**的 token，其餘**全部載入**；沒有「留在下層待補」。可借鏡：Probe 要先載入該層 KV 才能打分；訊號依任務而不同（PR vs VT 差 17–20%）。 |
| Cascade | 沒做 | Eq.7 p.7 | 部分還原依**預算**決定位元組數，剩下的**重算**；與內容相關性無關。 |
| CacheFlow | 沒做 | — | 全量還原的切分。 |
| WhereKVLive | 沒做 | p.3 | 依 recency／frequency，不看 query。 |
| InferCept／Tokencake | 沒做 | — | 整請求的去留。 |

**總判定：這六篇都沒做。** 最大威脅仍是 L1 §2 C27 的 IMPRESS／ContiguousKV／CacheTune（還原時依重要度只載一部分），但 L1 已記錄它們都沒和 Cake 結合、也沒量 GPU 忙。

---

## 3. 每個候選剩下的新意空間

**C-I1**：(1) **干擾感知的聯合還原**：把「GPU 忙時 H2D 慢 46–50%」（任務說明引用的第二階段結果，出處 `research_map.md` 第 3 點所指 08 §2.1、09 §6.1，本文未重新核對）放進成本模型，與 CacheFlow／Stony Brook 的「互不干擾」假設正面對照；(2) 目標換成**每 session 的 P95**並加入「誰先」（CacheFlow 沒有順序、只看批次 makespan）；(3) CPU／SSD／NFS 混在同一佇列、共用鏈路（CacheFlow 單一 B；Cascade 各自算）；(4) 在 MI300X 真機量（Cascade 是模擬）。最小證偽仍是 experiment_matrix 的 I1：聯合策略對各自 Cake 的 P95 好不到 5% → 停。

**C-I3**：動作空間與預測載回都已有人做，κ 跨硬體也已有人說。剩下的只有**交集**：以工具暫停為觸發、四動作含 SSD 與重算、**長 context（≥32K）下依位置的重算成本**、實測 κ，並證明**同一規則在 NVIDIA（引用公開數字）與 MI300X 上選不同動作**；外加 warm-up 在 MI300X 上會不會拖慢正在 decode 的請求（I2）。若做不出「決策翻轉」，這題只剩 AMD 上的量測。

**C-F**：邊界本身是新的（三維、真實引擎、併發）；**目標函數翻轉**（平均 vs goodput／P95）沒人做。但要先承認 EfficientAgent、py-kvcache、Cascade 圖 4、Stony Brook 已各有一維或單點；新意必須來自「併發」這一軸和「目標翻轉」，不能只是再找到一個變慢的點。

**C-C1**：六篇都沒做；L1 已知的前作也沒有「依相關性排載入順序＋與 Cake 前算後載結合＋其餘留在下層按需補」。CacheSmith 給的兩個警訊要放進設計：打分本身要先載部分 KV（Probe 依賴）、query-aware 訊號在分散型任務上會輸給 KV 偏差訊號（p.4）。新意風險主要在品質：若下一輪需要的 block 不可預測（research_map 缺口 3），整題不成立。

---

## 4. 對 repo 既有文件的更正與補充（只列，未改原檔）

| 文件 | 原寫法 | 全文核對後 |
|:--|:--|:--|
| `docs/SOTA_ANALYSIS_20261006.md` §2.2 CacheFlow | 「理論假設計算和載入兩種資源互不干擾」（沒給位置） | 位置：**附錄 A.1「Model and assumptions」p.14** 原句「overlap without interference, with sufficient buffers」；主文 §3.2 Eq.2 p.5 是隱含版；批次 DP Eq.5 p.6 假設重算時間可相加。另補：batch 只到 8；預設 10 Gbps（1.25 GB/s）。 |
| `docs/research_20261009_explore/cards/A_WhereKVLive.md` | 「連 oracle prefetch 都沒有一格贏過不 prefetch」 | 這句只對**搬移量**成立〔p.1〕；就延遲，oracle 在「超過一半」的格子比不預取慢〔p.6〕。條件：batch 1、投機與需求搶同一鏈路；原文說有空閒頻寬時另當別論〔p.8〕。 |
| `L3_reuse_serving_agentic.md` H6 Tokencake | v2：A100＋H200；搬運約 60 ms vs 重算約 9,000 ms（150×）；venue 只見 repo 描述 | v5（2026-09-28）：A100＋**H20**；4,096 token 時往返 63.7 ms vs 重算 1,815 ms（28.5×），1–5K token 間 26.8–37.5×〔p.12〕；**arXiv Comments 欄自述 EuroSys '27**。 |
| `L3_reuse_serving_agentic.md` §4.0 表 Cascade 列 | 有 SSD？「摘要提到 deeper tier」；成本比「摘要未說」 | 有 NVMe（DRAM、NVMe 兩個深層，多段路徑）；**沒有** κ 或跨硬體成本比，決策量是 SLO 預算；評測在 Vidur 模擬器、GB200 profile。 |
| `L2_compression_offload_prefetch.md` CacheSmith | 「SLO 感知排程、把同 deadline 的載入分組、跨多個 I/O stream」 | 屬實（p.8–9），但主題是 **RAG 非前綴 chunk 的選擇性重算**，4 條 I/O stream 是實作細節；不是 session 還原。 |

---

## 5. 參考（本次都實際打開過）

| # | 論文 | 打開了什麼 | 讀到 |
|:--|:--|:--|:--|
| 1 | Cascade, arXiv 2608.06557v1 | abs 頁＋PDF 全文 | 〔全文〕（文字，不含圖） |
| 2 | CacheSmith, arXiv 2610.18925v1 | abs 頁＋PDF 全文 | 〔全文〕（文字，不含圖） |
| 3 | CacheFlow, arXiv 2604.25080v2 | abs 頁＋PDF 全文（含附錄） | 〔全文〕（文字，不含圖） |
| 4 | Where Should the KV Cache Live?, arXiv 2609.16215v1 | abs 頁＋PDF 全文 | 〔全文〕（文字，不含圖） |
| 5 | InferCept, arXiv 2402.01869v2 | abs 頁＋PDF p.1–7 | 〔部分〕 |
| 6 | TokenCake, arXiv 2510.18586v5 | abs 頁＋PDF §1–7 | 〔部分〕 |
| 7 | Najafi Zadeh, Stony Brook FSL-26-02（技術報告） | PDF p.2、16–19、22、26、31 | 〔部分〕 |
| 8 | EfficientAgent, arXiv 2609.33762v1 | abs 頁（全文由 repo 卡讀過 p.1–9） | 〔摘要〕＋repo 卡 |
| — | SwiftCache, arXiv 2606.16135；Bottlenecks, arXiv 2601.19910 | abs 頁（搜尋時順手確認） | 〔摘要〕：SwiftCache 是跨模型借 HBM＋NVLink，與 C-C1 無關；Bottlenecks 已在 repo（κ_crit） |

**本次沒做的**：沒有看任何論文的圖（圖上的數值沒引用）；InferCept §5 評測與附錄、Tokencake §5–6 只略讀；Stony Brook 報告只讀上列頁；沒有找 Cascade／CacheFlow 的程式碼（前者內文沒給連結，後者說接受後才釋出）。另用 WebSearch 做了 4 次補漏搜尋（併發還原＋爭用、agent 暫停＋SSD／重算＋硬體、prefix caching 變慢＋併發、選擇性還原），除了上面第 7、8 篇外沒有找到新的直接威脅；**查不到不等於沒有**。
