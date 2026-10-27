# L1：Strata 全文與四個引擎——「單一引擎內、共享 KV 在下層 tier 時」已經被做掉的部分

> **verifier 更正（2026-10-26，[V_verifier.md](V_verifier.md)；本文其餘未改）**
> - §1.4：transient node 有兩種：**in-queue**（本輪掃描中已有請求要算、還沒開算）與 **in-flight**（正在計算）。延後的對象是命中這兩種之一、超過 100 token 的請求，不只是「正在被計算」（V #1）。
> - §1.5：Strata 沒有「以成本比較」的逐請求重算 vs 載入；但 SSD 層預設 best-effort 預取，請求被排上就中止、只用已到的部分，其餘只能 prefill（隱含的退回〔推論〕）；OSDI 版另可設 wait-complete／timeout（V #5）。
> - §1.4 的 38%：要加「不計 system prompt」；這句、§6 的 future work、§6 的公平性自承，都只在 **OSDI'26 版**，arXiv v1 沒有（V #6、#8、#9）。
> - §5.2 PEEK：排在 cluster 大小前面的 score 本身就是「等待中共用者的深度加權需求」（路徑上 pending_count×邊長），另有 group-major 讓同 cluster 連續送；所以 PEEK **確實依等待人數排序**，只是只有 GPU 快取（V #17）。

- **日期**：2026-10-26
- **執行者**：文獻與原始碼調查 agent（Claude）。**self-review，不是 cross-model review**（CLAUDE.md §5 的已知落差）。沒有跑 GPU、沒有安裝套件、沒有 git commit；除本檔外沒有改任何 repo 檔案；`gh` 只做唯讀查詢。
- **三向交集的定義（本檔通用）**：**共享**＝多個請求用同一份 KV；**延遲隱藏（hiding）**＝等待時讓別的有用工作重疊；**排程**＝誰先跑、誰等。另外三個概念分開記：**latency reduction**（操作本身變快，例如 GPU-assisted I/O）、**work avoidance**（快取重用、不做）、**speculation**（先做可能需要的，例如預取）。
- **查了什麼**
  - **全文**（PDF 用 `pdftotext -layout` 轉文字，逐頁讀）：
    1. Strata，USENIX OSDI'26 版 PDF（17 頁；封面算 PDF p.1）；另用 arXiv v1（13 頁）做關鍵字比對。
    2. PEEK（arXiv 2607.02525 v2；本文 p.1–10 與附錄表 4–5）。
    3. TempoKV（arXiv 2609.35065 v1；8 頁全讀）。
  - **原始碼**（淺層 clone 到 scratchpad，只讀不執行）：
    - vLLM `main` @ `0f6b2619`（2026-10-26）
    - SGLang `main` @ `66cac6b7`（2026-10-26）
    - LMCache `dev` @ `a2cd25df`（2026-10-25）
    - NVIDIA Dynamo `main` @ `ee012092`（2026-10-25）
    - llm-d-router `main` @ `b5cb5f88`（2026-10-25）；llm-d `main` @ `d5e53624`（2026-10-22，只讀 `guides/tiered-prefix-cache/README.md`）
  - **GitHub 唯讀**：vLLM issue／PR 約 10 次 `view`、約 19 次 `search`；SGLang PR 約 14 次 `search`。
  - **文獻搜尋**：
    - WebSearch 8 次；WebFetch 1 次（ACM 頁回 403）。
    - arXiv API 12 次：關鍵字查詢 8 次，ID 批次取摘要 4 次。
    - arXiv abs 頁實際打開 11 篇。
    - Semantic Scholar 2 次：引用 Strata 的論文 45 篇，全部掃過摘要；另 1 次被 429 限流。
    - Crossref 2 次。
- **證據等級**
  - 〔全文〕讀了原文相關段落。
  - 〔部分〕只讀部分章節。
  - 〔摘要〕只讀 arXiv／Crossref 摘要。
  - 〔標題〕只確認了書目。
  - 〔程式碼〕GitHub 原始碼（標 commit 與 `file:line`）。
  - 〔PR〕GitHub PR／issue 內文或留言：是工程聲明，不是同儕審查。
  - 〔文件〕官方文件。
  - 〔判讀〕我自己的推論。
- **頁碼**：一律用 PDF 頁。Strata 用 USENIX PDF，p.1 是封面，所以論文頁碼 n＝PDF 頁 n+1。
- **引用規則**：一律轉述。「查不到」寫 NOT_FOUND，未核實寫 NOT_VERIFIED；**查不到 ≠ 沒有**。
- **不重做的部分**：
  - Strata 的評估細節已在 `docs/research_20261006_eval/cards/E04_tiered_systems.md` 逐頁核對過。本檔只補「共享／排程」的機制細節，評估只列最小集合。
  - vLLM 0.19.1／LMCache／SGLang／Dynamo 的**寫入側**阻塞行為見 `docs/research_20261009_explore/H0_offload_blocking.md`。
  - Cascade、Bidaw、py-kvcache 見 `docs/stage3_design_20261024/lit/L8b_contention_aware_restore.md`。
  - 以上本檔只用一行指過去。

---

## §0 結論

**Strata 的 delay hit 處理到底是什麼**〔全文〕：

1. 它是「**在排程時把晚到的請求延後一輪，等第一個請求把 context 算完**」，屬於延後式合併，不是在載入層把傳輸合併。
2. 判準只看「與正在進行中的 context 重疊多少 token」（預設 >100 token）。**不看有幾個人在等**，也**不會為了湊更多共享者而刻意等待**（§4.3.1，p.7–8）。
3. 原文定義的 in-flight 是「快取**正在被計算**」。對「正在從 host 載入」的同一段 context，原文沒有另外的延後規則。它靠 **bundle hit** 處理：組 batch 時優先把共用同一 context 的請求放進同一批（Alg. 1，p.8）。

**vLLM main 仍保留 0.19.1 的延後邏輯，只是改名**〔程式碼〕：

1. `_blocks_being_loaded` 已改名 `_chunks_being_loaded`，語意不變（`offloading/scheduler.py:651-654, 1026-1050` @ `0f6b2619`）。
2. 載入是非同步的：請求進入 `WAITING_FOR_REMOTE_KVS`，排程器跳過它、繼續排別的請求。這是最基本的「一個等、另一個跑」（`v1/core/sched/scheduler.py:1032-1038, 1119-1124, 1472-1503`）。
3. #44294 沒有被修：issue 以 not planned 關閉。**修正 PR #44295 至 2026-10-26 仍是 OPEN**，帶 stale／needs-rebase 標籤。repo 的 `L3` 寫它「同樣關閉」是錯的，要改。
4. 真正的「同一段只載一次、後到者直接掛上同一份 GPU block」只存在於 **open PR #57418**（2026-09-17 開，尚未 review 通過）〔PR〕。

**SGLang HiCache**〔程式碼〕：

1. Strata 的 I/O 部分已進上游：`--hicache-io-backend` 預設 `kernel`，`--hicache-mem-layout` 預設 `page_first`。
2. 有 storage prefetch 三種停止策略；上游 CLI 預設是 `timeout`，論文預設是 best-effort。
3. LPM 排序把 **host 命中**算進匹配長度（`mem_cache/common.py:260`），但 L3（storage）命中不算。
4. 在上游沒有找到 Strata 的 delay-hit 延後、balanced batch、bubble filling（grep 與 PR 搜尋皆 NOT_FOUND）。
5. 「同一段正在算」的情況，SGLang 有較舊的 **in-batch prefix caching**：只排一個、其餘暫時降優先（`managers/schedule_policy.py:106-119, 346-383`）。

**LMCache、Dynamo KVBM、llm-d**：

- 都沒有找到「下層→GPU 的 in-flight 載入去重」。
- LMCache MP 的 L2→L1 prefetch 刻意讓不同請求可以同時 stage 同一個 key〔程式碼〕。
- Dynamo kvbm-engine v2 只在**卸載方向**有 in-flight 去重〔程式碼〕。
- llm-d 的 router 有依 tier 加權的前綴分數；有「正在載入中」成本項的只有 **LoRA**，KV 沒有〔程式碼〕。

**2025–2026 新文獻**：

- 引用 Strata 的 45 篇中，有 arXiv 摘要的 38 篇**沒有一篇**以 delayed hit 或 in-flight 載入共享為主題；其餘 7 篇只看了標題。
- 最接近的兩篇是 **PEEK** 與 **TempoKV**〔全文〕：
  - **PEEK**：用等待佇列的 radix tree 讓同前綴群的 pioneer 先跑、sibling 隨後命中，並依群大小排序。**只有 GPU 快取，沒有下層 tier**。
  - **TempoKV**：SSD→快速層的 staging 會依 KV 物件合併，並依「距離被取用的時間 vs 準備好所需時間」決定何時投入資源。**明確不改請求排程**。

**一句話**：單一引擎內：

- 「**同一段正在算時，叫後到者等**」（Strata、SGLang、PEEK）已被做掉。
- 「**同一段正在從 CPU 載入時，只載一次**」（vLLM 延後式、vLLM tiering promotion、TempoKV staging 合併、PR #57418）已被做掉。
- 「**載入時讓別的請求或 decode 先跑**」（vLLM 非同步、Strata bubble filling、balanced batch）已被做掉。
- **沒找到**的是：依**等待者人數**與**下層 tier 的成本**，在「等 in-flight／重複載入／重算」三者之間選，並決定 pioneer 與 followers 的先後（甚至刻意延後以湊更多共享者）的排程規則。
- 工程社群正在碰這一塊：#44294、#44295、#44882、#57418。它們的實測顯示「一律等」與「一律重算」在不同併發下互有勝負（§2.2）。

---

## §1 Strata 全文（OSDI'26）

### 1.1 書目與版本

| 項目 | 內容 |
|:--|:--|
| 標題 | Strata: Hierarchical Context Caching for Long Context Language Model Serving |
| 作者 | Zhiqiang Xie（第一作者，Stanford＆NVIDIA）等 8 人，含 C. Kozyrakis〔全文 PDF p.1–2〕 |
| venue | 20th USENIX OSDI，2026-07-13～15，Seattle〔全文 PDF p.1 封面；USENIX 頁 `usenix.org/conference/osdi26/presentation/xie-zhiqiang`〕 |
| arXiv | 2508.18572，只有 v1（2025-08-26），Comments 寫「under peer review」〔abs 頁已開〕 |
| 版本差異 | OSDI 版比 arXiv v1 多了：三種寫入策略、各層 LRU、DeepSeek-V3／H20 磁碟實驗。repo `E04` 卡 ②已記。本檔一律引 OSDI 版。 |
| 實作 | 做在 SGLang 裡，並在數家公司的生產環境部署〔全文 p.3〕 |

### 1.2 cache-aware scheduling 的決策規則

整體目標：最大化快取效益，避開 delay hit 與載入空泡（stall）〔全文 §4.3，p.7〕。排程分三段，每段的輸入、輸出與目標如下：

| 階段 | 輸入 | 輸出 | 目標／規則 | 出處 |
|:--|:--|:--|:--|:--|
| ① Delay-hit 延後 | 等待佇列中每個請求的 token、HiRadixTree 上的 **transient node** | 被延後到下一輪、但排在佇列**最前面**的請求 | 避免同一段 context 在被算出之前又被別人重算。只有在 transient node 上匹配超過門檻（預設 100 token）才延後 | §4.3.1，p.7–8 |
| ② Balanced batch | 每個請求的「要載入多少」與「要算多少」，都由 HiRadixTree 估出 | 下一個 prefill batch | batch 的「總載入量／總計算量」超過門檻（預設 100）就算 loading-bound；會造成 loading-bound 的請求移到降優先清單。加入一個請求後，**優先把與它 bundle hit 的請求一起放入** | §4.3.2、Alg. 1，p.8 |
| ③ Bubble filling | batch 仍是 loading-bound | 先送 decode batch，和 context 載入並行 | 用有用的計算填補載入空泡 | §4.3.3，p.8 |

- **門檻怎麼來**：loading-bound 門檻與硬體、模型相關，可以分開 profile。預設 100，對應 Fig. 1 開始出現空泡的位置〔全文 p.8〕。
- **防飢餓**：被降優先的請求保持原順序；每次組 batch 一律從佇列第一個請求開始〔全文 p.8〕。
- **實作**：SGLang 的排程器是 Python，Strata 另外用 C++ 實作這些策略以壓低開銷〔全文 p.7〕。

### 1.3 載入如何被當成資源；空泡怎麼填

**載入頻寬被當成一級資源**〔全文〕：

- 原文主張：排程器必須把 CPU→GPU 頻寬當一級資源，組 batch 時平衡計算與搬運（§3.2，p.5）。
- 做法是用「載入量／計算量」的比值限制每個 batch（見上表 ②）。
- 原文的問題設定：既有系統假設逐層重疊可以把第 N+1 層的載入藏在第 N 層的計算後面；但 cached context 很長、新 token 很少時，這個假設不成立（§3.2，p.5）。

**空泡怎麼填**〔全文〕：

- 例子：請求 G0 要載很長的 context。排程器先擱置已準備好的 prefill batch，改送 decode batch 和載入並行（§4.3.3，p.8；Fig. 7，p.7）。
- 為什麼兩者能重疊：decode 主要吃 HBM 頻寬，載入吃 PCIe 頻寬，所以爭用小（p.8）。
- 原文另說：在 PD 分離系統裡也可以插 prefill batch（p.8）。

**這不是「讓等待的請求做別的事」**〔判讀〕：bubble filling 填的是**別的請求**（running 中的 decode），對象是整個 batch 的空泡，不是針對某一段共享 KV 的等待者。

### 1.4 delay hit 的處理到底是什麼

**定義**〔全文〕：

- Strata 沿用網路領域的 delay hit：同一資料的第一個 miss 還沒解決時，又有請求到達。後到者實際上也變成 miss（§3.2，p.5；引用 Atre et al., SIGCOMM'20）。
- 在 LLM serving 中的後果是**重複的 prefill 計算**：同 context 的請求被排進同一 batch，或跨 data-parallel 實例（p.5）。
- 非同步排程器（先準備下一批）會把 miss 的解決窗口拉長到一整個 batch 的執行時間，使 delay hit 更容易發生（p.5）。

**機制**〔全文 §4.3.1，p.7–8〕：

- HiRadixTree 加入 **transient node**，有兩種標記：
  - in-queue：有請求引用一段新 context。
  - in-flight：這段 token 的快取**正在被計算**。
- 掃佇列時，必要時插入 in-queue 節點。命中既有 transient node 的請求延後到下一輪，但放在等待佇列**最前面**，以便享用「即將變熱」的快取。
- 請求開始執行時，它的節點改標 in-flight；完成後轉為一般節點，指向已就緒的快取。
- 原文列出兩種情境：(i) 共用同一 miss 的多個請求被排進同一 batch；(ii) 請求被非同步地準備，卻不知道對應 context 還在計算中（p.7）。

**逐項回答**：

| 問題 | 答案 | 證據 |
|:--|:--|:--|
| 是合併（coalesce：同一段只做一次、多人等）嗎？ | **是**，但用的是「延後到下一輪再命中」，不是把多個等待者掛在同一個 in-flight 物件上。合併的對象是 prefill **計算** | 〔全文〕p.7–8 |
| 對「正在從 host 載入」的 context 也延後嗎？ | 原文的 in-flight 只定義為「正在計算」。載入側沒有獨立描述；載入側的共享靠 bundle hit（下列） | 〔全文〕p.7–8；〔判讀〕 |
| 是排序嗎？ | 是：被延後者放到佇列**最前面**；bundle hit 讓同 context 的請求在 batch 形成時被優先拉在一起 | 〔全文〕p.7–8 |
| 會依「等待同一段的人數」調整優先序嗎？ | **不會**。延後門檻只看匹配 token 數（預設 100）。bundle hit 是「有就一起放」，也不依人數排序 | 〔全文〕p.8 |
| 會刻意等待以湊更多共享者嗎？ | **沒有**。只有已經在佇列中的請求才可能 bundle（Alg. 1 只掃當下的佇列 Q） | 〔全文〕p.8；〔判讀〕 |

**bundle hit**〔全文 p.8〕：

- 例子：D0 和 D1 共用同一 context，放在同一 batch 可以平衡載入，也省 GPU 記憶體與 on-device 頻寬。原文稱這是 delay hit 的反面。
- 〔判讀〕「省 GPU 記憶體」意味著同一份 context 只載一次、兩個請求共用。但原文沒有寫出載入層的去重規則。

**量測**〔全文〕：

- 影響 delay hit 嚴重度的兩個系統因素：快取解決時間（prefill 加排程延遲），以及系統吞吐，也就是到達率。兩者越大，delay hit 造成的 miss 越多（§5.3.4，Fig. 12，p.12–13）。
- 這部分是**模擬**：用 Mooncake Tool-Agent trace、等效無限的快取；吞吐 1／10／100 req/s 由縮放時間戳得到。原因是作者的機器撐不到夠高的吞吐（p.12–13）。
- 原文估 1K-token 輸入的解決時間，依模型與硬體從數百毫秒到數秒（p.12）。
- 實機消融（LooGLE、Qwen2.5-14B、H200；Fig. 11，p.12）：同 context 請求排在一起（min cache distance）時，delay-hit 緩解讓峰值吞吐 +42%；平均分散（max cache distance）時完全沒有效益。

**共享前綴有多常見**〔全文〕：Mooncake agent tool trace 中，38% 請求在 1 秒窗口內，和至少一個其他請求共用 ≥6K token 前綴（不含 system prompt）（§3.2，p.5）。這和 repo `D8` 卡一致。

### 1.5 recompute vs load、tier、公平性

| 問題 | 答案 | 證據 |
|:--|:--|:--|
| 有 recompute vs load 的選擇嗎？ | **沒有**。原文指出長 context 時重算越來越貴，是不吸引人的替代方案；只把「必要時改重算」歸給 Cake（引用 [23]） | 〔全文〕§3.2，p.5；參考文獻 p.16 |
| 只有 CPU tier 嗎？ | **GPU／CPU／外部儲存（SSD）三層** | 〔全文〕§4.2.1，p.7 |
| SSD 怎麼處理？ | storage 層命中時預取到 host，並與該請求的**排隊延遲**重疊。被排程時中止預取，用已到的部分。預設 best-effort；可改 wait-complete（等完）或 timeout | 〔全文〕§4.2.1，p.7 |
| SSD 有沒有做端到端排程實驗？ | 磁碟只在 §5.3.5 的子實驗出現（H200 的磁碟 <1 GiB/s，另在 H20 上測） | 〔全文〕p.10、p.13 |
| 有公平性處理嗎？ | 只有防飢餓（被降優先者保序、每批從佇列頭開始）。原文自承：排程以整體 I/O 與計算效率優先，仍可能不公平、違反個別 SLO，SLO／公平感知列為延伸 | 〔全文〕p.8；§6，p.14 |
| 寫入策略與淘汰？ | write-back、write-through、selective-write-through（預設，門檻 2）；各層預設 LRU | 〔全文〕§4.4，p.9 |

### 1.6 GPU-assisted I/O 是什麼

這一項屬於 **latency reduction**，不是 hiding〔全文 §4.2，p.6–7〕：

- **做法**：不再反覆呼叫小的 `cudaMemcpyAsync`，改成啟動一個 CUDA kernel，讓數千個 thread 各自從來源（GPU global memory 或 pinned CPU memory）搬一小塊到目的地。
- **好處**：
  - 併發高。
  - 粒度小：多數架構上 128 B 就有效率。
  - 搬運時可以順便做版面轉換：GPU 端用 layer-first，host 與磁碟用 page-first。
- **干擾控制**：用少量大的 CUDA block，讓硬體排程器把 I/O kernel 限制在少數 SM 上，並用指令繞過快取以減少污染。
- **量測**（H200，Fig. 5）：2 個 1024-thread block 達到 48 GB/s，prefill 降速 <5%、decode 降速 10%。
  - 預設：載入（關鍵路徑）用 2 個 block，備份（非關鍵）用 1 個 block。
  - 端到端影響 <5%（p.6–7）。
- **ROCm**：原文說 kernel 經 ROCm 後端也相容 AMD GPU（p.6）。〔判讀〕repo 的 MI300X 平台理論上可用；**NOT_MEASURED**。
- **後續討論**（§6，p.14）：`cudaMemcpyBatchAsync`（CUDA 12.8）走 DMA、不搶 SM，但同一微基準只有 38 GB/s（kernel 是 48 GB/s），適合用在非關鍵的 GPU→CPU 備份。

### 1.7 評估（只列原文數字與位置；更多細節見 repo `E04` 卡）

| 項目 | 內容 | 位置 |
|:--|:--|:--|
| 硬體 | H200 節點：8×H200、PCIe 5.0 x16（單向 64 GB/s）、1.6 TB DRAM。H20-storage：8×H20＋Intel P5510 NVMe（7 GB/s 讀）。GH200：464 GB LPDDR5X，到 CPU 384 GB/s | §5.1，p.9 |
| 模型 | Llama-3.1-8B-Instruct、Qwen2.5-14B-Instruct-1M、Llama-3.1-70B-Instruct（4 卡 TP）。磁碟子實驗另用 DeepSeek-V3（8×H20） | p.9、p.13 |
| context 長度 | 表 1 平均輸入：LooGLE 21,613、NarrativeQA 54,797、ReviewMT 17,708、ShareGPT 680.9。NarrativeQA 濾掉 >128K 的文件 | 表 1，p.9 |
| trace／到達 | 資料集沒有時間戳，用 Poisson。ShareGPT 插 60 s thinking time。in-flight 上限 128。delay-hit 分析用縮放後的 Mooncake Tool-Agent trace（模擬） | p.10、p.12–13 |
| baseline | vLLM v0.8.5＋LMCache v0.2.1（chunk 256、page 32）；TRT-LLM v0.17.0（page 32）；SGLang v0.4.5；SGLang-HiCache（page 32） | p.9 |
| 指標 | 平均 TTFT、輸出 token 吞吐 | p.10 |
| 主要數字 | LooGLE 同 TTFT 下的吞吐，依序對 SGLang-HiCache／vLLM-LMCache／TRT-LLM-HiCache：Llama-8B 最多 3.2×／2.6×／1.9×；Qwen-14B 3.9×／2.1×／1.9×；Llama-70B 5×／5×／3.75×。ReviewMT（Llama-8B）1.7×／2.3×／2.3×。warm NarrativeQA 對 vLLM-LMCache：2.3×／2.6×／2.5×（三個模型） | p.10 |
| 消融 | 排程與 I/O 各帶來最多 1.8× 與 2.3× 峰值吞吐（Fig. 9）。Strata-IO-LPM 在低請求率時有幫助，高請求率時因淘汰變多而失效 | p.11–12 |
| 動機數字 | Qwen2.5-14B＋LooGLE，SGLang 卸到 CPU：74% 的 prefill 時間卡在 KV 傳輸，吞吐最多掉 4×（p.2）；用了 Strata 的 I/O 後仍有最多 24% 卡住（p.3）。8192 token、page 32 只用到 PCIe 5.0 理論頻寬約 22%，GH200 上低到 5%（p.4–5） | p.2–5 |
| 磁碟 | DeepSeek-V3、8×H20、page 32、12 req/s：page-first 版面讓平均 TTFT 改善 2.1×、吞吐改善 1.3× | §5.3.5，p.13 |

### 1.8 原文自列限制與 future work（§6，p.14；§7，p.14）

1. **I/O kernel 干擾**：用少量 SM 爭用換傳輸效率；未來要降低 kernel 開銷，並主張更通用的晶片內 I/O 加速器。
2. **公平性**：排程以整體效率優先，可能不公平、違反 SLO；延伸方向是 SLO／公平感知的 batch 形成。
3. **更慢 tier 的預取**：目前的策略只藏 host→HBM 的延遲。從更慢的 storage 預取，只能靠排隊延遲來藏；為它創造排程餘裕是「同樣有前景」的方向。〔判讀〕這正是「共享 KV 在 SSD 時的排程」的空位。
4. **模型範圍**：只支援 dense attention；sparse／linear attention 待做。
5. **範圍**：只做**單一計算實例**內的記憶體管理與排程（§7，p.14）。

---

## §2 vLLM main（`0f6b26191064384aa35060c87f4ed570c7875e66`，2026-10-26）

### 2.1 0.19.1 的延後邏輯還在嗎？——**還在，改名**〔程式碼〕

- **改名**：PR #52615（2026-09-09，block→chunk 改名）把 `_blocks_being_loaded` 改成 `_chunks_being_loaded`。
- **啟用條件**：只有開 GPU prefix caching 時才啟用。註解寫明用途是避免重複載入（`offloading/scheduler.py:651-654`）。
- **記錄與清除**：
  - 發出載入工作時，把要載的 key 加進集合（`:1470-1471`）。
  - 載入完成時移除（`:2218-2220`）。
- **延後判斷**：`_lookup_complete_chunks` 最後一段，只要命中範圍內有任何 chunk 正在被載入，就回傳 None，也就是延後整個請求（`:1026-1050`）。
- **另一種延後**：同一請求自己還有 in-flight 傳輸時，也延後（`:1252-1257`）。
- **tiering 的 HIT_PENDING／RETRY**：後端回這兩個值（例如正在從 SSD promotion 到 CPU）時，也延後（`:786-808`）。

### 2.2 #44294 的行為有沒有被修？——**沒有合併任何修正**

**#44294**（issue）〔PR〕：

- 2026-06-02 開。2026-10-07 被機器人以 stale、not planned 關閉。
- 作者的量測（2×L40S、Qwen3-8B、50 併發共用 10K 前綴）：0.19.1 開卸載 TTFT 11.46 s，不開 5.95 s，0.17.1 開卸載 0.90 s。
- 作者把回歸追到 PR #29087。

**#29087**〔PR〕：

- 2025-11 開，2026-01-21 merged。標題是 Prevent redundant loads。
- 用意：多個請求同時命中同一批 CPU block 時，避免各發一次 CPU→GPU 傳輸，以及在 GPU prefix cache 裡產生重複副本。

**#44295**（修正 PR）〔PR〕：

- 2026-06-02 開，**至 2026-10-26 仍為 OPEN**。標籤：`bug`、`needs-rebase`、`stale`、`kv-connector`；review 狀態 REVIEW_REQUIRED。
- 2026-10-15 被標 stale：若 30 天沒有活動會自動關閉。
- **repo 的 `docs/stage3_20261019/lit/L3_reuse_serving_agentic.md` §F4 寫「修正 PR #44295 同樣關閉」與此不符，需要更正。**
- 它的修法：每一步記錄「GPU KV 是否有壓力」。
  - 有壓力：照舊延後。
  - 沒壓力：回 0，放棄這次 CPU 命中，改由 GPU 端處理。
- PR 內文的表（2×L40S、Qwen3-8B、10K 共享前綴；表頭沒寫指標，依上文推測是相對不卸載的吞吐變化，**指標定義 NOT_VERIFIED**）：

  | 併發 | GPU 狀態 | 未修 | 只拿掉延後（一律回 0） | 本 PR |
  |:--|:--|:--|:--|:--|
  | 1 | 不溢出 | −3.8% | −3.7% | −3.9% |
  | 20 | 溢出 2 倍 | −20.5% | −38.0% | −32.0% |
  | 50 | 溢出 5 倍 | −40.7% | −33.8% | −32.4% |

  - PR 作者的解釋：一律回 0 會拿掉自然的節流，讓 20 個請求同時重新 prefill（PR 內文）。
  - 〔判讀〕三種規則在不同併發下互有勝負：併發 20 時「延後」最好，併發 50 時「放棄命中」較好。**等 in-flight 載入 vs 自己重算，沒有一條固定規則在所有負載下都最好**。這是三向交集裡唯一看到的直接證據；它是工程量測，不是論文。

**同一問題的後續**〔PR〕：

| 編號 | 狀態 | 內容 |
|:--|:--|:--|
| #44882（feature request） | 2026-06-08 開；open，標 stale | 列出三種現有做法及各自代價：重複載入（浪費 I/O 與 GPU 空間）、在 connector 端擋住（convoy）、後到者回 miss（重算已知的外部命中）。要求在核心層讓後到者掛上同一份 GPU materialization |
| #48189 | 2026-07-09 開，2026-07-10 關閉，未合併 | 在核心排程器做同一外部前綴的合併：follower 停駐、不配 block，owner 完成後經 APC 發布 |
| **#57418** | 2026-09-17 開；**OPEN**，REVIEW_REQUIRED，2026-10-25 仍有更新 | **在 block pool 索引「正在載入中的確切前綴」**；後到且符合條件的請求直接以 refcount 掛上同一份目的 block，只發一次載入；處理 owner 或 follower 取消、失敗傳播 |
| #55724 | 2026-09-07 開；open，needs-rebase | 本步的 token budget 被 running 請求用完時，仍讓「只需非同步載入」的 waiting 請求以 0 token 進場，使載入能和本步的模型執行重疊 |
| #61961 | 2026-10-16 開；open，標 `ready` | 小的 CPU 命中改成在同一步同步載入；預設關（`sync_load_budget_bytes`） |

**#57418 的細節**：

- 只支援：內建 CPU offloading、單一 full-attention 群組、block 對齊的部分前綴。
- 作者的 GPU 驗證只有 Qwen3-0.6B、8 併發、觀察到 7 個 follower；**沒有效能主張**（PR 內文）。
- 留言中，另一位 contributor 從 RL（GRPO）角度跟進：PD＋NIXL 下同組 n 個請求會把前綴拉 n 次。他貼的 GB200 基線表前兩列，transfers／group 等於併發數（1→1、4→4）；其餘列未讀。

**#61961 的細節**：

- 動機：wide-EP MoE 的 prefill 一步 0.3–0.5 s，而非同步載入會讓請求停約 2 步，即使拷貝只要約 1 ms。
- PR 內文數字：
  - Qwen3-30B-A3B、H100，相對 GPU 命中對照組的額外 TTFT 中位數：單節點從 +219 ms 降到 +1 ms；雙節點從 +344 ms 降到 +8 ms（用的是較早的版本）。
  - GLM-5.3：約 1.9 s 降到約 0.85 s。
- 〔判讀〕**非同步載入的 latency hiding 不是免費的**：步長很長時，「等下一步才能被排」的固定代價會蓋過拷貝本身。

### 2.3 非同步載入時，排程器會不會先跑別人？——**會**〔程式碼〕

- **connector 端**：`get_num_new_matched_tokens` 對任何命中都回報「非同步載入」（`offloading/scheduler.py:1279`，回傳 `num_hit_tokens, bool(num_hit_tokens)`）。
- **核心排程器**（`v1/core/sched/scheduler.py`）：
  - connector 回 None（延後）時，`skip_request` 把請求移出，記入 `deferred_waiting`，然後 `continue` 去排**下一個** waiting 請求（`:1032-1038, 1119-1124`）。
  - 非同步載入時，配好 block，把狀態設為 `WAITING_FOR_REMOTE_KVS`；新 block 先不登記進 prefix cache（`delay_cache_blocks=load_kv_async`，`:1415`）。之後同樣 `skip_request` 並 `continue`（`:1472-1503`）。
  - 被跳過的請求，本輪結束時重新放回**佇列最前面**，保留原順位（`:1564-1568`）。
  - 持有 KV block 的 waiting 請求（含正在載入者）放在 `kv_holding_waiting`，每步先處理（`:1041-1046`）。
- **完成後**：
  - worker 回報 `finished_recving` 後，排程器記下，**下一步**才把請求升回 WAITING（`:3409-3426`，註解寫「schedule the request during the next step」；`:3371-3386`）。
- **〔判讀〕三個邊界**：
  1. 延後與載入完成都是**每步輪詢**，沒有「完成就喚醒」。
  2. waiting 迴圈是 `while token_budget > 0`（`:1040`）：budget 被 running 請求用完時，後面的 waiting 請求**連查詢都不會發**，下層載入無法提早開始。這是 #55724 要修的。
  3. 等待者之間只依 FCFS 或 priority 排序（`request_queue.py:16-17`）；沒有依載入量或共享程度排序，也沒有 Strata 式的「載入／計算」平衡。

### 2.4 多個請求命中同一批正在載入的 block：只載一次嗎？

| 情境 | 行為 | 證據 |
|:--|:--|:--|
| CPU→GPU（OffloadingConnector、開 prefix caching） | **只載一次，靠延後**：後到者被延後；owner 完成並把 block 登記進 GPU prefix cache 後，後到者下次查詢變成 GPU 命中 | 〔程式碼〕`offloading/scheduler.py:651-654, 1026-1050, 2218-2220`；〔判讀〕 |
| 不開 prefix caching | 集合是 None，不去重：各自載入 | 〔程式碼〕`:652-654` |
| SSD→CPU（TieringOffloadingSpec 的 promotion） | **只 promotion 一次**：發起時先在 CPU 層配好空間並標為 in-flight（ref_cnt −1）；之後的查詢回 HIT_PENDING，請求被延後。註解說明是為了避免重複 promotion | 〔程式碼〕`v1/kv_offload/tiering/manager.py:633-655, 679-705` |
| 後到者直接掛上同一份 GPU 目的 block（不用等到下一次查詢） | **尚未合併**，只在 PR #57418 | 〔PR〕 |

### 2.5 與排程相關的其他旋鈕（只列查到的）

- **每請求 `max_load_tokens`**（PR #55885，2026-09-16 merged）：限制一個請求最多載多少（`offloading/scheduler.py:370, 399-409, 872-876`）〔程式碼〕。
- **每請求 tier 過濾**（#48123）〔PR 標題〕。
- **OffloadingConnector 的 retention interval**（#51886）〔PR 標題〕。
- 以上都不是共享感知的排程。

---

## §3 SGLang HiCache（main `66cac6b79ccffa13c108297fad8a43ddc49d06f8`，2026-10-26）

### 3.1 Strata 的哪些部分已在上游〔程式碼〕

| 參數 | 上游預設 | 對應 Strata | 位置 |
|:--|:--|:--|:--|
| `--hicache-io-backend` | `kernel`（GPU-assisted I/O） | §4.2 | `arg_groups/fields/memory.py:128-134` |
| `--hicache-mem-layout` | `page_first` | §4.2.1 | `:135-147` |
| `--hicache-write-policy` | `write_through` | 論文預設是 selective，門檻 2 | `:121-127` |
| `--hicache-storage-prefetch-policy` | `timeout`；選項 `best_effort`／`wait_complete`／`timeout` | 論文預設 best-effort | `:170-176` |

### 3.2 prefetch 策略的確切行為

- **何時觸發**：L1／L2 比對之後，若 L3 命中長度超過門檻（預設 256 token），才發 prefetch〔程式碼 `mem_cache/unified_radix_cache.py:316, 2048, 2136`；文件 `docs/docs/advanced_features/hicache_design.mdx:92-95`〕。
- **何時可以中止**〔程式碼 `unified_radix_cache.py:2281-2299`〕：
  - `best_effort`：隨時可以中止。
  - `wait_complete`：一律不中止，直到完成。
  - `timeout`：由 PP rank 0 判斷是否超時，TP 間取 MAX。
- **超時預算**：
  - 程式碼：從第一次「被擋住的准入輪詢」開始計時，預算＝base＋頁數×每頁時間（`:2245-2257`）。
  - 程式碼裡的 fallback 預設：base 1.0 s、每 Ki token 0.25 s（`:482-483`）。
  - 文件寫的是：base 2 s、每 1024 token 0.1 s、上限 30 s（`hicache_design.mdx:105-115`）。
  - **兩者不一致，實際生效值 NOT_VERIFIED**。
- **限流**：依「預取佔用的 host token 數」限流；程式碼註解自承待改進成依後端效能（`managers/cache_controller.py:1269-1286`）〔程式碼〕。
- **〔判讀〕三種策略與三向交集的關係**：
  - `wait_complete` 就是「讓請求等下層載完，排程器先跑別人」。
  - `best_effort` 是「不等，缺的部分重算」。
  - `timeout` 介於兩者之間。
  - 三者都是**單一請求**的停止規則，不看有沒有別人也在等同一段。

### 3.3 有沒有 in-flight 載入去重？

| 方向 | 結果 | 證據 |
|:--|:--|:--|
| L3→L2（storage prefetch） | **NOT_FOUND**（找不到跨請求的去重）：<br>• 「是否已在預取」是以**請求**為鍵檢查（`request in self.ongoing_prefetch`）。<br>• 預取完成插入 host 樹時，若前綴已被別人插入，就釋放重複的 host 記憶體。<br>〔判讀〕這表示兩個請求可能同時讀同一段，事後才丟掉重複 | 〔程式碼〕`unified_radix_cache.py:2092-2096, 2491-2492`；grep「dedup／in-flight／coalesce」無相關結果 |
| L2→L1（host→GPU load-back） | **隱性共享**〔判讀〕：<br>• `load_back` 發出 H→D 拷貝後，立刻把 device indices 掛到樹節點上，並記一個 GPU 儲存事件。<br>• 之後比對到這個節點的請求，看到的是 device 命中。<br>• 第二個請求在**後續的步**是否正確等待逐層載入完成：NOT_VERIFIED | 〔程式碼〕`unified_radix_cache.py:1806-1925`；`unified_cache/unified_tree_core.py:2681-2711` |
| 正在**計算**中的同前綴 | **in-batch prefix caching**：<br>• 對快取匹配很短（≤32 token）的請求，另外在「等待佇列 radix tree」上比對。<br>• 若等待佇列中已有人有同前綴（≥32 token），就暫時降優先，只讓一個先跑。<br>• 這是 Strata delay-hit 延後的簡化版，**只看佇列，不看 in-flight** | 〔程式碼〕`managers/schedule_policy.py:106-119, 346-383` |

### 3.4 排程有沒有考慮下層命中？

- **LPM 算進 host 命中**〔程式碼〕：
  - LPM 依 `num_matched_prefix_tokens` 排序（`schedule_policy.py:441-451`）。
  - 這個值＝device 匹配＋**host 命中長度**（`mem_cache/common.py:260`）。所以 host 命中和 device 命中**等價計入**，沒有依載入成本打折。
- **L3（storage）命中不算進 LPM**〔判讀〕：它只在之後的 prefetch 路徑出現。
- **佇列太長時退化**：等待佇列超過 128 時，LPM 與 HRRN 退化成 FCFS（`schedule_policy.py:137, 304-315`）〔程式碼〕。
- **host 載入的准入只檢查記憶體預算**：沒有 Strata 式的載入／計算比（`schedule_policy.py:1486-1504`，`prepare_load_back` → `init_load_back`）〔程式碼〕。
- **Strata 的三段排程**（transient node 延後、balanced batch、bubble filling）：在上游 `managers/`、`mem_cache/` grep `delay_hit`／`bundle`／`loading_bound`／`bubble`，以及 `gh search prs` 搜「delay hit／bundle hit／Strata／balanced batch／bubble filling」：**NOT_FOUND**。〔判讀〕可能以別的名稱存在、或只在 NVIDIA 內部分支。

---

## §4 LMCache、NVIDIA Dynamo KVBM、llm-d

| 系統（版本） | 非同步載入與排程的互動 | in-flight 載入去重 | 證據 |
|:--|:--|:--|:--|
| **LMCache in-process connector**（dev `a2cd25df`） | 對 vLLM 一律回「同步」（`load_async=False`）：載入在 forward 內完成，排程器看不到等待 | NOT_FOUND | 〔程式碼〕`lmcache/inprocess/integration/vllm/lmcache_connector_v1.py:151-171` |
| **LMCache MP connector** | 命中時回非同步（`need_to_load > 0`）。lookup 還沒回來時回 None，請求延後。可開 `lmcache.mp.eager_prefetch`：請求一進 waiting queue 就提交 lookup／prefetch | **刻意不去重**：L2→L1 prefetch 的 L1 寫入保留以**請求**為 tag，註解說明同時載入同一 key 的多個請求彼此不搶保留；L1Manager 允許不同 tag 同時 stage 同一 key | 〔程式碼〕`lmcache/integration/vllm/lmcache_mp_connector.py:1390-1392, 1483, 642-644, 1485-1493`；`lmcache/v1/distributed/storage_controllers/prefetch_controller.py:85-97`；`lmcache/v1/distributed/l1_manager.py:476-510` |
| **Dynamo KVBM**（vLLM connector，v1 bindings；`ee012092`） | 一律以非同步 onboard 命中的 block（註解明寫） | onboarding 方向 NOT_FOUND：grep「inflight／dedup／coalesce」，命中的都是請求層記帳或事件去重 | 〔程式碼〕`lib/bindings/kvbm/src/block_manager/vllm/connector/leader.rs:289-303` |
| **Dynamo kvbm-engine v2** | 文件：本地 G2 全命中直接回；需要 G3→G2 或遠端時回 AsyncSession | **只在卸載方向**有：`PresenceFilter`＋`PendingTracker` 過濾目的層已有或正在傳輸中的 block，避免重複傳輸。G4 搜尋有每個 session 的 `pending_load`（不跨請求） | 〔程式碼〕`lib/kvbm-engine/docs/onboarding.md:180-185`；`lib/kvbm-engine/src/offload/policy.rs:259-277`；`src/leader/session/initiator.rs:55-70, 1213-1216` |
| **llm-d router**（`b5cb5f88`） | 叢集層。精確前綴打分依 tier 加權，預設 gpu 1.0、cpu 0.8、storage 0.6、shared_storage 0.4、object_store 0.2 | KV：NOT_FOUND。但 **LoRA** residency scorer 有 `PendingCost`：同一 adapter 已有較早請求正在載入時，後到者以較低成本計（預設 0.1），等那次載入，不付自己的載入成本 | 〔程式碼〕`pkg/kvcache/backend.go:34-42`；`pkg/epp/framework/plugins/scheduling/scorer/loraresidency/lorascore/score.go:30-55` |
| **llm-d approximate prefix** | 路由決定時就把 prompt hash 插進被選 pod 的索引。〔判讀〕等於路由層的 speculation：假設該 pod 即將擁有這段 KV，讓之後的同前綴請求跟過去 | 不適用 | 〔程式碼〕`pkg/epp/framework/plugins/requestcontrol/dataproducer/approximateprefix/plugin.go:423` |
| **llm-d tiered-prefix-cache 指南**（`d5e53624`） | 推薦 `native`，即 vLLM `OffloadingConnector`；router 另加一個追蹤 CPU tier 的前綴分數。〔判讀〕單一引擎內的行為就是 §2 的 vLLM 行為 | 同 vLLM | 〔文件〕`guides/tiered-prefix-cache/README.md:15-21` |

---

## §5 2025–2026 新文獻

### 5.1 怎麼查

- **arXiv 關鍵字**（依提交日期新到舊）：
  - 摘要含 "delayed hit(s)"／"delay hit"：只回 5 篇快取理論，0 篇 LLM。
  - "in-flight"＋"KV cache"＋load。
  - "shared prefix"＋offload＋schedul。
  - "KV cache"＋"loading bound"：只回 Strata。
  - "KV cache"＋coalesc。
  - "prefix cache"＋"concurrent requests"＋"same prefix"：0 篇。
- **WebSearch**：
  - delayed hit＋LLM。
  - in-flight dedup。
  - 前綴共享＋階層快取＋佇列感知（extended）。
  - thundering herd／request coalescing／single-flight＋prefix cache。
  - load bandwidth "first-class resource"。
  - HotPrefix、FlashGen 書目。
- **引用 Strata 的論文**：Semantic Scholar 列出 45 篇，用 arXiv API 取回其中有 arXiv ID 的 38 篇摘要，關鍵字掃描（delayed／delay hit、in-flight、coalesc、duplicate load、same prefix／context、concurrent requests、shared prefix）：**0 篇命中**。另外 7 篇（DualPath、REKV、CXL、MCP、Act While Thinking、RAGBoost、模擬平台）沒有 arXiv ID；只看標題，與主題無關。
- **讀全文**：PEEK、TempoKV，理由見下。

### 5.2 最接近的兩篇（全文）

#### PEEK〔全文〕

- **書目**：B. Xie *et al.*（4 人），"PEEK: Predictive Queue-Informed KV Cache Management for LLM Serving"。
  - arXiv 2607.02525，abs 頁顯示 v2；PDF 頁首 2026-09-12；Comments 寫 under review。
  - API 的 published 欄是 2026-05-10，與 2607 編號不符，**v1 日期 NOT_VERIFIED**。
  - venue NOT_VERIFIED；程式在 `github.com/xiexbing/peek`（原文 p.1）。
- **做什麼**：
  - 對**等待佇列**建增量 radix tree，看出哪些等待中的請求共用前綴（cluster）（§3.1，p.4）。
  - 排序鍵（Eq. 1，p.4）：區段（warm／pioneer／sibling）、−LPM 命中、−祖先需求分數、−cluster 大小、到達時間。
  - 任何 pioneer 都先於 sibling 進場，讓 sibling 隨後命中剛算出的前綴（p.4–5）。pioneer 判斷用前 θ＝32 個 token 是否在本輪已被認領。
  - **cluster 大小（等同「等同一段的人數」）是排序的 tie-break**。
- **淘汰**：只在引擎必須釋放記憶體時，依佇列需求覆蓋 LRU（Eq. 2，p.5）。
- **公平**：與到達順序 lane 交錯，α 預設 0.7；動態 lane 讓公平份額在 0.10–0.60 之間調整（p.5）。
- **評估**（表 4–5，PDF p.16；§4，p.5–9）：
  - 引擎：SGLang 0.5.9、vLLM 0.19.1。
  - 硬體：1–4×H100 80 GB。
  - 模型：Qwen2.5-32B、Llama-3.1-70B（TP2，DP1／2）、Mistral-Small-24B、Gemma-2-27B。
  - 負載：LooGLE、RepoBench、Mooncake conversation trace、LMSYS。
  - 最多 3.0×／2.6× 命中、7.9×／7.1× TTFT、6.7×／5.5× E2E、3.6×／4.5× 吞吐（SGLang／vLLM，摘要與 p.2）。
  - 只用 cLPM 排序，命中就到完整 PEEK 的 3 個百分點內（p.7）。
  - W1 的 TPOT 比 baseline 高 8–26%（p.7）。
- **與三向交集的關係**〔判讀〕：
  - 它做掉了「共享×排程（依佇列中的共享人數）」，但**只對 GPU 快取與計算 miss**。
  - 記憶體設定只有 GPU 比例（表 5），全文沒有 host／SSD tier、沒有載入、沒有 delay hit 字樣，也**沒有引用 Strata**（grep）。
  - 自列限制：只在 H100、排序仍在 Python、沒有共享的流量下攤銷變小（p.10）。

#### TempoKV〔全文〕

- **書目**：J. H. Park, H. Kim, D. Kim（Samsung Semiconductor），"TempoKV: Timely Staging of LLM KV Caches for Memory-Semantic Flash"。arXiv 2609.35065 v1（2026-09-28）；venue NOT_VERIFIED。
- **做什麼**：
  - 等待中的請求一旦被認出有可重用 KV，只記一筆 metadata「claim」，不發 I/O、不保留容量（§3，p.3）。
  - 執行端估「距離開始取用還有多久」（TTU），儲存端估「若現在投入，要多久才能放進快速層並受保護」（TTR）。
  - 兩者相減（laxity＝TTU−TTR）≤0 時才請求投入（§3.1–3.2，p.3）。
- **共享**：
  - claim 以請求為單位，但 staging 與保護可以共享。
  - 儲存端對**同一 KV 物件合併 staging**，只為尚未被別人保護或保留的 bytes 加保留容量，避免重複 SSD 讀與重複保留（§3.2，p.4）。
  - TTR 會考慮「正在進行的共享 staging」：一個 claim 即使不新增 SSD 讀，也可能因為在等別人的 staging 而 TTR 非零（p.3）。
- **排程**：**刻意不改請求排程**。投入順序可以和請求順序不同，但請求順序不變（p.2、p.4）。
- **評估**（§5，p.5–6）：
  - 平台：1×H100 PCIe（PCIe Gen5×16）；CXL 記憶體裝置（XCENA MX1P，128 GiB DRAM，後端 PM1753 NVMe）。
  - 軟體：修改版 vLLM v0.23.0＋LMCache v0.5.1；**關閉 GPU prefix caching**、FCFS、最多 4 個 active sequence。
  - 模型：Qwen2.5-14B（YaRN×4）、Llama-3.1-8B。
  - 負載：12 份 NarrativeQA 文件，16K–68K token，每 0.5 s 到達一個。
  - 保護容量成本比立即 staging 少 63–91%。
  - 100% 前綴比例時，相對 Demand：p95 TTFT 最多 −25.7%、吞吐最多 +20.4%。
  - 相對 LMCache-DAX：NarrativeQA p95 TTFT 最多 −48.0%、吞吐 +27.8%；LooGLE 為 −32.4%、+15.5%。
  - 動機實驗：64K token、8 GiB 的前綴，staging 提前量從 0 增到 1 s，中位延遲從 1.23 s 降到 0.51 s，之後持平；提前到 4 s 時保護成本從 8 增到 32 GiB·s（Fig. 1，p.2）。
  - 同樣排在佇列第一位，時間餘裕可以從提早 2.67 s 到延遲 6.02 s，所以佇列順位不是期限（Fig. 2，p.2）。
- **與三向交集的關係**〔判讀〕：
  - 做掉了「下層（SSD）staging 的 in-flight 合併」與「依佇列進度決定何時開始藏延遲」。
  - 但評估負載每份文件只有一個問題（p.5），共享 staging 幾乎沒有被壓測。
  - 不決定誰先跑，也不在等待／重算之間選。

### 5.3 只讀摘要或標題的相關工作

| 論文 | 讀到 | 與三向交集的關係 |
|:--|:--|:--|
| HotPrefix（Y. Li *et al.*, Proc. ACM Manag. Data 3(4), 2025-09；DOI 10.1145/3749168，Crossref 確認） | 〔摘要〕 | GPU／CPU 兩層：依熱度選擇性准入 CPU、定期 promotion 回 GPU、I/O 與計算 pipeline；做在 SGLang。看不到等待者人數或 in-flight 合併 |
| SARIA（arXiv 2610.31169，2026-10-22；H. Li *et al.*） | 〔摘要〕 | LMCache MP 上 DRAM＋SSD：選擇性寫 SSD；SSD 命中後提升到 DRAM；傳輸協調器在 SSD 傳輸進行時節制 DRAM 讀，共享 GPU 入口頻寬。是**載入頻寬排程**，不是共享 |
| PCR（arXiv 2603.23049，W. Wang *et al.*） | 〔摘要〕 | RAG：用排程器佇列裡的 pending 請求做 look-ahead LRU；逐層重疊；依佇列從 SSD 預取到 DRAM |
| ProServe（arXiv 2512.12928 v2，W. Huang *et al.*） | 〔摘要〕 | 多優先權排程；搶佔時非同步卸載、pipeline reload、可調 copy budget，讓計算和 host-device 搬運重疊 |
| Janus（arXiv 2609.36938，W. He *et al.*） | 〔摘要〕 | sparse attention＋SSD：提前預測要讀哪些 KV，讓 SSD 讀與計算重疊；合併相鄰讀取（同一請求內，不是跨請求） |
| Range（arXiv 2610.21241，D. Shi *et al.*） | 〔摘要〕 | 指出 SSD restore 的成本隨「邏輯物件數」增長（同 6 GiB：192 個物件 2.1 s、6,144 個物件 3.0 s）；屬 latency reduction |
| FlashGen（J. Jeong, J. Ahn, ASPLOS'25 Vol.2；DOI 10.1145/3676641.3716245，Crossref 確認） | 〔標題〕 | Strata 說它在逐層載入 pipeline 上加了「重排執行」的排程，已實作在 SGLang，並用在 Strata 的 baseline 裡（Strata p.14）。摘要 NOT_VERIFIED（ACM 頁 403） |
| Cascade（arXiv 2608.06557） | repo `L8b` 〔全文〕 | 依延遲預算決定還原／預取／留 HBM／重算；沒有提到共享或 in-flight |
| Delayed-hit 快取理論：Gurushankar *et al.*（arXiv 2501.16535，Comments 寫 INFOCOM 2025）；B. Jiang *et al.* VA-CDH（2504.20335）；B. Jiang *et al.*（2505.15531）；原始定義 Atre *et al.* SIGCOMM'20（Strata 參考文獻 [8]） | 〔標題〕 | 古典骨幹：miss 尚未解決時後到者的延遲模型。沒有一篇處理 LLM KV |

---

## §6 對三向交集的意義

### 6.1 在單一引擎內，已經被做掉的部分

| # | 已做掉的事 | 誰做的 | 證據 |
|:--|:--|:--|:--|
| A1 | **同一段 context 正在計算時，後到者延後、算完再命中**（共享×排程，計算 miss） | Strata transient node（門檻 100 token、延後者放佇列最前） | 〔全文〕§4.3.1，p.7–8 |
| | | SGLang in-batch prefix caching（等待佇列中同前綴只先放一個） | 〔程式碼〕`schedule_policy.py:106-119, 346-383` |
| | | PEEK pioneer-first | 〔全文〕§3.2，p.4–5 |
| A2 | **依等待同一前綴的人數排序** | PEEK：cluster 大小與祖先需求分數是排序鍵的 tie-break。**限 GPU 快取** | 〔全文〕Eq. 1，p.4 |
| A3 | **同一段正在從 CPU 載入時只載一次**（共享×work avoidance，載入 miss） | vLLM：延後式 | 〔程式碼〕`offloading/scheduler.py:651-654, 1026-1050` |
| | | Strata：bundle hit 同批 | 〔全文〕p.8 |
| | | SGLang：load-back 立即掛到樹節點，隱性共享 | 〔程式碼〕＋〔判讀〕 |
| A4 | **同一段正在從 SSD 搬上來時只搬一次** | vLLM tiering promotion（HIT_PENDING） | 〔程式碼〕`tiering/manager.py:679-705` |
| | | TempoKV：同一 KV 物件合併 staging 與保護 | 〔全文〕§3.2，p.4 |
| A5 | **後到者直接掛上同一份 in-flight 目的 block**（不用等到下一次查詢） | vLLM PR #57418（**open**）；先前的 #48189 已關閉 | 〔PR〕 |
| A6 | **一個請求等載入時，其他請求照跑**（hiding×排程） | vLLM `WAITING_FOR_REMOTE_KVS`＋`skip_request` | 〔程式碼〕`v1/core/sched/scheduler.py:1032-1038, 1472-1503` |
| | | Dynamo KVBM、LMCache MP 的非同步 onboard／載入 | 〔程式碼〕 |
| A7 | **載入空泡塞 decode**；**batch 依「載入／計算」比組成** | Strata bubble filling、balanced batch（門檻 100） | 〔全文〕§4.3.2–4.3.3，p.8 |
| A8 | **下層預取和排隊時間重疊**；可選「等完／不等／超時」 | Strata §4.2.1 | 〔全文〕p.7 |
| | | SGLang 三種 prefetch 策略 | 〔程式碼〕`memory.py:170-176`；`unified_radix_cache.py:2281-2299` |
| | | py-kvcache 的排程感知預載 | repo `L3`、`L8b` |
| | | TempoKV 以 TTU 對 TTR 定時 | 〔全文〕 |
| A9 | **排程把下層命中算進前綴匹配** | SGLang LPM（host 命中與 device 命中等價計入） | 〔程式碼〕`common.py:260` |
| | | llm-d 叢集路由依 tier 加權 | 〔程式碼〕`backend.go:34-42` |
| A10 | **「一律等 in-flight」與「一律放棄命中改算」在不同併發下互有勝負**的工程證據 | vLLM #44294、#44295 | 〔PR〕§2.2 表 |

### 6.2 明確沒有被做的部分（查不到 ≠ 沒有）

| # | 沒找到的事 | 怎麼查的 |
|:--|:--|:--|
| B1 | **依「等待同一段下層 KV 的人數」排序載入**（不是 GPU 前綴群）：例如一段 SSD／CPU 上的前綴被 N 個等待者需要，就先載它 | Strata 全文：延後門檻是 token 數，bundle hit 不依人數。PEEK 全文：人數只用在 GPU 計算 miss。TempoKV：不改排程。vLLM／SGLang 程式碼的排序只有 FCFS、priority、LPM／DFS／HRRN。45 篇引用 Strata 的摘要關鍵字掃描。arXiv 與 WebSearch 共 6 組查詢 |
| B2 | **為了湊更多共享者而刻意延後 pioneer 或一次載入**（批次窗口） | 同上；Strata Alg. 1 只掃當下佇列。thundering herd／request coalescing 搜尋只找到應用層（API 呼叫層）的部落格 |
| B3 | **每個 follower 在「等 in-flight／重複載入／自己重算」三者之間依成本選擇**，成本含 tier 頻寬、等待者數、GPU 是否忙 | vLLM #44882 只列出三種做法和各自代價；#44295 用「GPU 有沒有壓力」一個布林值切換；Strata 不做 recompute vs load（p.5）；Cascade（repo `L8b` 全文）有還原／重算的選擇，但沒有共享。**沒有任何論文**把這個選擇寫成規則 |
| B4 | **SSD tier 的共享感知排程** | Strata 把「為更慢 tier 創造排程餘裕」列為 future work（p.14）；TempoKV 明確不改排程；vLLM tiering 只有延後式去重 |
| B5 | **follower 等待時間要拿什麼工作填**（針對共享等待的 hiding，不是整個 batch 的空泡） | Strata 的 bubble filling 是 batch 層級（p.8）；vLLM 是「跳過、排下一個」。沒有找到依共享關係安排填補工作的設計 |
| B6 | **pioneer 與 followers 之間的公平性**，在下層載入 convoy 下 | Strata 自承不公平（p.14）；PEEK 的公平 lane 只對 GPU 群；#44294 的 convoy 是 TTFT 量測，不是公平分析 |
| B7 | **以上任何一項在 AMD 或跨硬體比較** | 所有論文都是 NVIDIA（H200、H20、GH200、H100、L40S、GB200）；Strata 只宣稱 ROCm 相容（p.6），沒有量測。**NOT_MEASURED** |

### 6.3 對新提案的威脅等級

| 提案的主張 | 威脅 | 理由 |
|:--|:--|:--|
| 「第一個處理多請求在 miss 未解決時要同一段 KV（delay hit）」 | **高** | Strata（OSDI'26，生產部署）明文處理，並引用 SIGCOMM'20 的 delay hit；SGLang 與 PEEK 也有類似機制 |
| 「第一個讓下層 KV 的 in-flight 載入只做一次、多人共用」 | **高** | vLLM 自 2026-01（#29087）就有延後式去重；tiering promotion 也有；TempoKV 合併 SSD staging；#57418 正在做真正的掛載共享，隨時可能合併 |
| 「第一個在等待載入時讓別的工作先跑」 | **高** | vLLM 非同步載入的基本行為、Strata bubble filling、balanced batch 都已存在 |
| 「依等待者人數與 tier 成本，在等／重複載／重算之間選，並決定 pioneer 與 follower 的先後（可含刻意延後湊共享者）」 | **中** | 論文層級沒有找到（B1–B4），但三面都有鄰居：<br>• Strata（延後但不數人）<br>• PEEK（數人但只有 GPU）<br>• TempoKV（合併但不排程）<br>• #44295 的表（等與重算互有勝負）<br>工程社群正在動這一塊（#44882、#57418、#55724、#61961），新穎性會隨上游合併而縮水。站得住的寫法必須給出**規則與它翻轉的條件**（例如何時重算勝過等待），而不是機制本身 |
| 「上述規則在 SSD tier、或跨硬體（κ）會翻轉」 | **低–中** | 沒有找到量測；Strata 把 SSD 排程餘裕列為 future work；所有工作都只在 NVIDIA。風險在於 py-kvcache 的 future work 包含持續估計佇列延遲（repo `L8b` T4） |

**〔判讀〕給下一步的一句話**：單一引擎內，「共享＋隱藏＋排程」的**機制**已經全被做掉。剩下的是**策略**：一段下層 KV 被 N 個請求等待時，誰先、等不等、要不要乾脆重算。PR #44295 的表已經在工程層面暗示這個策略會隨負載翻轉。要成立，最小證據是：在真實引擎上，「一律等」「一律重算」「依 N 與 tier 成本選」三者的 TTFT／goodput 有交叉，而且交叉點會隨 tier（CPU vs SSD）或硬體移動。**這是假設，NOT_MEASURED。**

---

## §7 證據索引（IEEE 格式；最後一欄為核實狀態）

1. Z. Xie *et al.*, "Strata: Hierarchical Context Caching for Long Context Language Model Serving," in *Proc. 20th USENIX OSDI*, 2026; arXiv:2508.18572. —〔全文〕（USENIX PDF＋arXiv v1）
2. B. Xie, Z. Wang, M. Tanaka, Z. Zhen, "PEEK: Predictive Queue-Informed KV Cache Management for LLM Serving," arXiv:2607.02525v2, 2026. —〔全文〕；venue NOT_VERIFIED
3. J. H. Park, H. Kim, D. Kim, "TempoKV: Timely Staging of LLM KV Caches for Memory-Semantic Flash," arXiv:2609.35065v1, 2026. —〔全文〕；venue NOT_VERIFIED
4. Y. Li *et al.*, "HotPrefix: Hotness-Aware KV Cache Scheduling for Efficient Prefix Sharing in LLM Inference Systems," *Proc. ACM Manag. Data*, vol. 3, no. 4, 2025, doi:10.1145/3749168. —〔摘要〕（Crossref）
5. H. Li *et al.*, "SARIA: Persistence-Aware KV Caching with Coordinated DRAM–SSD Reload for LLM Agents," arXiv:2610.31169v1, 2026. —〔摘要〕
6. W. Wang *et al.*, "PCR: A Prefetch-Enhanced Cache Reuse System for Low-Latency RAG Serving," arXiv:2603.23049v1, 2026. —〔摘要〕
7. W. Huang *et al.*, "ProServe: Unified Multi-Priority Request Scheduling for LLM Serving," arXiv:2512.12928v2, 2025. —〔摘要〕
8. W. He *et al.*, "Efficient Agentic LLM Serving over SSD-based Sparse KV Storage," arXiv:2609.36938v1, 2026. —〔摘要〕
9. D. Shi *et al.*, "Range: Decoupling KV-Cache Reuse Granularity from Storage Work," arXiv:2610.21241v1, 2026. —〔摘要〕
10. J. Jeong, J. Ahn, "Accelerating LLM Serving for Multi-turn Dialogues with Efficient Resource Management," in *Proc. ASPLOS'25*, vol. 2, 2025, doi:10.1145/3676641.3716245. —〔標題〕
11. N. Atre, J. Sherry, W. Wang, D. S. Berger, "Caching with Delayed Hits," in *Proc. ACM SIGCOMM*, 2020. —〔標題〕（取自 Strata 參考文獻 [8]）
12. K. Gurushankar *et al.*, "Latency Guarantees for Caching with Delayed Hits," arXiv:2501.16535 (Comments: INFOCOM 2025). —〔標題〕
13. vLLM, `OffloadingConnector` scheduler 與 v1 scheduler，commit `0f6b2619`. —〔程式碼〕
14. vLLM issue #44294；PR #29087、#44295、#48189、#55724、#57418、#61961；issue #44882. —〔PR〕（2026-10-26 查詢狀態）
15. SGLang，commit `66cac6b7`：`arg_groups/fields/memory.py`、`mem_cache/unified_radix_cache.py`、`managers/schedule_policy.py`、`mem_cache/common.py`、`docs/docs/advanced_features/hicache_design.mdx`. —〔程式碼〕〔文件〕
16. LMCache，commit `a2cd25df`（dev）；NVIDIA Dynamo，commit `ee012092`；llm-d-router，commit `b5cb5f88`；llm-d，commit `d5e53624`. —〔程式碼〕〔文件〕
