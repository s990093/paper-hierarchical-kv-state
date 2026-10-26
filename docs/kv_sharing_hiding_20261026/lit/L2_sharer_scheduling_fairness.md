# L2：排程器怎麼利用（或忽略）「共享」——共享者排序、刻意延後、攤提 kernel、公平性，外加稀疏選頁的聯集（Q5）

> **verifier 更正（2026-10-26，[V_verifier.md](V_verifier.md)；本文其餘未改）**
> - DLPM「跨 client 誰付」那段在 **§9** Limitation and Future Work（p.12），不是 §8（V #19）。
> - Strata 的 38%、§6 的 future work 與公平性自承只在 **OSDI'26 版**；arXiv v1 的頁碼與節號不同（V #6、#8、#9）。

- **日期**：2026-10-26
- **執行者**：文獻 agent（Claude）。**self-review，不是 cross-model review**（CLAUDE.md §5 的已知落差）。只做網路搜尋、下載 PDF 轉文字、`gh` 唯讀查詢；沒有跑 GPU、沒有安裝套件；除本檔外沒有改任何 repo 檔案。
- **情境（使用者給定，本檔不驗證）**：一份很長的共享 context（例如 128K token 文件，Llama-3.1-8B 約 16 GiB KV）平時放在 CPU 記憶體或 SSD，一群使用者陸續來問同一份文件。〔判讀〕以 32 層 × 8 KV head × 128 維 × BF16 算，每 token 128 KiB，128K token 正好 16 GiB，算術一致。
- **四個概念分開用**：latency hiding（等待時重疊別的有用工作）、latency reduction（讓等待本身變短）、work avoidance（快取重用、少做事）、speculation（猜測後先做）。下文提到某系統時會標它屬於哪一種。

## 查了什麼

- **搜尋**：WebSearch 14 次（standard／extended）；WebFetch 3 次（FlashInfer cascade 部落格、Triton batcher 文件；其中 1 次 404 後改抓存檔頁）。
- **GitHub 唯讀**（`gh api`／`gh search`／`gh pr view`／`gh issue view`）約 35 次：
  - SGLang `main` @ `66cac6b`（2026-10-26T01:58Z）
  - vLLM `main` @ `0f6b261`（2026-10-26T01:56Z）
  - llm-d-kv-cache `main` @ `822e6dd`（2026-10-13）
  - NVIDIA Dynamo `main` @ `ee01209`（2026-10-25）
  - AIBrix `main` @ `d459142`（2026-10-25）
- **arXiv**：42 個 abs 頁用 curl 打開，抽出標題、第一作者、日期、Comments、Journal-ref（§5 參考清單逐筆列出）。arXiv API 當時回 429（rate limit），所以改讀 abs 頁。
- **PDF**：34 份，只從 arxiv.org 和 proceedings.neurips.cc 下載到 scratchpad，每篇一個子目錄，用 `pdftotext -layout` 轉文字；另外用 `pdftotext`（不加 `-layout`）產生閱讀順序版本，方便讀雙欄排版。頁碼一律是 **PDF 頁**（以 form feed 計數）。沒有執行任何下載下來的檔案。
- **讀全文的**（正文從頭讀到尾；附錄只查關鍵字）：
  - DLPM／D²LPM（2501.14312，必讀）
  - PEEK（2607.02525 v2，新搜尋中最接近的第 1 篇）
  - TokenDance（2604.03143 v1，新搜尋中最接近的第 2 篇）
- **讀相關章節的〔部分〕**：Strata、SGLang 論文、Preble、BatchLLM、Hydragen、RelayAttention、ChunkAttention、FlashInfer、RAGCache、Cascade、KVFlow、Tokencake v5、Continuum v7、AIBrix、Feather、HiSparse、Fathom、OasisKV、FusionRAG、Shared RAG-DCache、PrefixShield、SCBench、PQCache、RetroInfer。
- **只讀摘要〔摘要〕的**：其餘列在 §5。

## 證據等級

| 標記 | 意思 |
|:--|:--|
| 〔全文〕 | 讀了原文的相關段落，位置標到 §／PDF 頁／表／圖 |
| 〔部分〕 | 只讀部分章節，或用關鍵字定位後讀前後文 |
| 〔摘要〕 | 只讀 arXiv 摘要或官方頁摘要 |
| 〔程式碼〕 | GitHub 原始碼，標 `file:line @ commit` |
| 〔PR〕 | GitHub PR 或 issue 內文（工程聲明，不是同儕審查） |
| 〔文件〕 | 官方文件或部落格；經 WebFetch 讀的會註明，因為工具內有小模型整理 |
| 〔判讀〕 | 本 agent 的推論，不是原文寫的 |

- 「NOT_FOUND」指本次搜尋沒找到，**不代表不存在**。「NOT_VERIFIED」指沒有核實。
- 所有數字都是各論文在**自己的硬體、模型、負載**下量的，不可互相排名。
- **不重做**（只補新事實）：
  - `docs/stage3_20261019/lit/L3_reuse_serving_agentic.md`：E1–E9、G8、H1–H10、F4 的 vLLM #44294 延後載入
  - `docs/stage3_design_20261024/lit/L8b_contention_aware_restore.md`：Cascade 的 restore 面向
  - `docs/research_20261010_directions/cards/D8_Strata_排程補充.md`

---

## §0 結論

**一句話**：「看有多少請求在等同一段 prefix 再決定順序」「讓後到的共享者等先到的那次載入或計算」「把共享者排在同一批」都已經有人做過。但都只在 **HBM 內**，或者只把 CPU／SSD 命中當成**靜態折扣**。本次沒有找到任何系統同時做到下面兩件事：

- 用「下層 tier 的取回成本 × 等待中的共享者人數」排序；
- 為了湊更多共享者而**刻意延後**開始取回。

**Q1（依共享者人數，或「取回一次、多人受益」排序）——部分做過，但都不看下層 tier 的取回成本。**

- 數「等待中有幾個請求共用這段 prefix」的：
  - PEEK 的 pending radix tree 與 cluster size〔全文 p4 §3.1–3.2〕
  - SGLang 的 DFS-weight，以子樹內等待請求數當權重〔程式碼 `base_prefix_cache.py:347-387`〕
  - Feather 的 RL state 裡的 *w*，即共用此 prefix 的等待請求數〔部分 p8 §4.3〕
  - Preble 的淘汰成本：被淘汰 token 的 prefill 時間 × 節點命中數〔部分 p5–6 §3.2〕
- 但它們有的只看 GPU 內 cache（PEEK、Feather、Preble），有的只認 device 節點（DFS-weight 用 `last_device_node`〔程式碼 `common.py:245-252`〕）。
- 會碰到 CPU／SSD 的系統則把下層命中當成**等同命中或固定折扣**：
  - SGLang LPM 把 host 命中長度直接加進 matched tokens〔程式碼 `common.py:260`〕
  - llm-d 的預設權重是 GPU 1.0、CPU 0.8〔程式碼 `pkg/kvcache/backend.go:26-31`〕
  - Dynamo router 的預設 credit 是 host 0.75、disk 0.25〔文件 configuration-and-tuning.md L75–76〕
- 最接近「載入一次、配多個共享者」的是 **Strata 的 bundle hit**：共用同一段 context 的請求優先放進同一個 prefill batch〔部分 p6–7 §4.3.2，Alg.1〕。但它的排序起點仍是 FIFO 的隊首，**不依共享者人數排序**。

**Q2（刻意延後，等更多共享者）——「等已經在飛的那次載入或計算」做過了；「等未來才會到的共享者」NOT_FOUND。**

- 讓後到者等先到者，避免重複算或重複載（這屬於 work avoidance，不是 anticipatory）的有：
  - Strata 的 delay-hit deferral，門檻 100 token〔部分 p6 §4.3.1〕
  - vLLM `OffloadingConnector`：命中的 chunk 正在被載入時，延後整個請求〔程式碼 `offloading/scheduler.py:1026-1050`〕
  - SGLang in-batch prefix caching：prefix 相同的等待請求只先排一個〔程式碼 `schedule_policy.py:106-119, 346-396`〕
  - KVFlow 的 status-aware scheduling〔部分 p6 §3.3〕
  - PEEK 的 pioneer-first〔全文 p4–5〕
- 「為了等**還沒到**的共享者而延後開始取回」的 batching window，在 LLM KV 文獻與 vLLM／SGLang 程式碼中都沒找到。
- 唯一的「刻意延遲」是 Triton dynamic batcher 的 `max_queue_delay_microseconds`。它只為湊 batch 大小，跟 prefix 或 KV 無關〔文件〕。
- Strata 的消融剛好劃出這個缺口：
  - 共享者排在一起（最小 cache distance）時，delay-hit 處理讓峰值吞吐 +42%；
  - 共享者均勻散開（最大 cache distance）時毫無幫助〔部分 p10 §5.3.3〕。

**Q3（共享前綴 attention 攤提 × tier 還原）——NOT_FOUND。**

- 這些 kernel 都假設共享前綴的 KV **已經在 HBM**：Hydragen、FlashInfer cascade／composable formats、RelayAttention、ChunkAttention、BatchLLM 的 fused kernel。
  - ChunkAttention：KV 只存正在 decode 的序列〔部分 p2〕
  - FlashInfer composable formats：不做資料搬移，只算索引〔部分 p5 §3.1.2〕
- vLLM 在 **2026-10-12 移除了 cascade attention**：它預設關閉、多數 backend 不支援、bug 不斷〔PR #61126〕。
- 最接近「還原一次 → 共享者一起算」的有兩個：
  - Strata 的 bundle hit（只在 prefill batch 層級）；
  - TokenDance 的 collective reuse：每一輪只做一次分析，**但還原仍逐個 Mirror 請求載入 Master**〔全文 p7 Alg.1〕。

**Q4（公平性、誰付還原成本）——餓死問題被承認、也有實測；「下層 tier 還原 I/O 由誰付」NOT_FOUND。**

- 承認會餓死：SGLang 論文自己寫 LPM 可能餓死請求，並列為未來工作〔部分 p6、p10〕。
- 生產環境量到：
  - SGLang issue #31954：cache-miss 請求形成 20–60 s 的尾巴，約 5% 的請求貢獻約 75% 的累積 TTFT〔PR〕；
  - SGLang PR #32911：開 HiCache 的生產流量下，LPM 的 p99 TTFT 比 FCFS 高 252.56%〔PR〕。
- 對策有很多種：DLPM 的 deficit、PEEK 的 fairness lane、Preble 的優先權分組配額、RAGCache 的等待窗、Strata 保留原順序、Cascade 的 Tier-2 保留額度、SGLang HRRN、Feather 的 RL 探索。
- 「誰付錢」目前只有三種做法：
  - DLPM：prefix 只在第一次算進 GPU 時計費一次〔全文 p4 §3〕
  - PrefixShield：誰新建 block 誰揹債〔部分 p5〕
  - Dynamo DRR：只收「未命中 token」，host／disk 命中依權重算成已快取〔文件〕
- **DLPM 原文把「不同 client 共用同一 prefix 時，誰被扣 quantum」明列為未來工作**〔全文 p12 §8〕。
- 沒有一篇處理「從 CPU／SSD 取回的 I/O 時間由哪個共享者付」。

**Q5（稀疏選頁 × 多請求共用同一份長 context：同頁只載一次、依頁熱度排序、把挑到的頁重疊多的請求排在一起）——NOT_FOUND。**

- 「挑到的頁取聯集」只出現在**單一請求內**：
  - 跨 GQA head：Fathom〔部分 p1〕、OasisKV〔部分 p8〕
  - 跨 speculative verify 的多個 query：SSV〔部分 p2〕、TokenSpeed PR #2373〔PR〕、vLLM issue #47763〔PR〕
- 跨請求的情形：
  - OasisKV 只是把同一 batch 內不同請求的 miss **依層打包傳輸**，不是去重；而且原型不支援 prefix caching〔部分 p8、p13〕。
  - HiSparse（已合進 SGLang）是每個請求、每層各一個 GPU cache；host 存每個 active 請求的完整 KV〔部分 p7 Fig.2〕。
- 評估方式：
  - Quest、ReadKV、Fathom 的 offload 實驗都是 batch 1〔部分〕；
  - ShadowKV、PQCache、MagicPIG 的多請求就是 batch 維度，各自不同 context〔部分〕；
  - 唯一系統性做「共享 context、多請求」評估的是 SCBench，**只量品質，不量取回位元組**〔部分〕。
- RetroInfer 觀察到：同一份 context 上不同 query 的稀疏度差異很大〔部分 p4 Fig.4(b)〕。這暗示聯集可能不小，但**聯集大小沒有人量過（NOT_MEASURED）**。

**威脅等級**（細節見 §3.3）：

| 提案 | 威脅等級 |
|:--|:--|
| 只做「共享者感知的准入／合併等待」 | 高（PEEK＋Strata＋vLLM／SGLang 已涵蓋） |
| tier 感知的「共享者人數 × 還原成本」排序＋公平性 | 中 |
| 為等未來共享者而刻意延後還原（anticipatory window） | 低到中（沒找到前例，但價值依賴到達型態） |
| Q5 跨請求選頁聯集／頁熱度排序 | 低（沒找到前例，但可行性未知） |

---

## §1 Q1–Q5 × 論文對照表

欄位說明：「下層 tier」指系統本身會不會從 CPU／SSD／遠端取回 KV。短語後面的括號是證據位置；詳細內容見 §2 卡片。

| # | 論文／系統（venue） | 讀到 | 下層 tier | Q1 依共享者或下層命中排序 | Q2 刻意延後等共享者 | Q3 攤提 kernel × 還原 | Q4 公平性／誰付 | Q5 |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| 1 | DLPM／D²LPM（arXiv 2501.14312；venue NOT_FOUND） | 〔全文〕 | **無**（全文找不到 offload／tier；只有 GPU radix cache） | LPM 排序＋deficit；只看已快取 prefix 長度（p4–5 Alg.1） | 無；work-conserving（p5） | 無 | quantum 有界（Thm 4.1 p4）；prefix 只計一次（p4 §3）；跨 client 誰付列為未來工作（p12 §8） | — |
| 2 | SGLang 論文（NeurIPS'24） | 〔部分〕 | 論文版無；未來工作列多層記憶體（p10） | LPM＝DFS 序，離線最佳（Thm 3.1 p6） | 無 | 無 | 承認 LPM 會餓死，留待未來（p6、p10） | — |
| 3 | SGLang `main` 程式碼與 PR | 〔程式碼〕〔PR〕 | HiCache L1／L2／L3 | LPM 把 host 命中算進 matched（`common.py:260`）；DFS-weight 用子樹內等待數，但只認 device 節點（`base_prefix_cache.py:347-387`、`common.py:245-252`）；佇列 >128 時改 FCFS（`schedule_policy.py:137, 304-314`） | in-batch：prefix 相同者只先排一個（`schedule_policy.py:346-396`）；HiCache 的 L3 預取策略 `wait_complete`／`timeout` 是逐請求設定（文件） | 無 | HRRN 已合併（PR #32911），LPM p99 +252.56%；aging 提案 #31954／#31723 皆關閉 | — |
| 4 | vLLM `OffloadingConnector`（`main`） | 〔程式碼〕〔PR〕 | CPU（另有 tiering 管理器） | 無排序（預設 FCFS） | 命中的 chunk 正在載入時延後請求（`scheduler.py:1026-1050`）；`HIT_PENDING` 沒有期限（issue #49829） | cascade attention 已移除（PR #61126） | 無 | — |
| 5 | Strata（OSDI'26） | 〔部分〕 | CPU＋SSD | bundle hit：同 context 優先放同批，但起點是 FIFO 隊首（p6–7 Alg.1） | delay-hit deferral，門檻 100 token（p6）；SSD→CPU 預取與排隊時間重疊（p5 §4.2.1） | 只到 prefill 同批（p7）；kernel 未提 | 被延後者保留原順序，每批從隊首開始（p7） | — |
| 6 | Preble（ICLR'25） | 〔部分〕 | 無 | 淘汰成本＝prefill 時間 × 節點命中數（p5–6）；依 cached % 分優先群（p7） | 無 | 無 | 優先群按比例配額，改善 p99 但不改平均（p7、p10） | — |
| 7 | **PEEK**（arXiv 2607.02525，審稿中） | 〔全文〕 | **無**（「KV pressure tier」指超賣倍數） | pending tree 的 cluster size 與 ancestor score（p4）；淘汰保護等待需求（p5） | pioneer 先跑、siblings 排在後面；GM 把整群連續排（p4–5） | 無 | fairness lane＋動態 lane；GM 讓 TTFT／E2E 變差 10–29%，DL 救回 2.4–15.1%（p2、p7） | — |
| 8 | **TokenDance**（arXiv 2604.03143，審稿中） | 〔全文〕 | LMCache 後端（p7–8） | 同一輪的請求一起處理（p5 §4.2） | 靠應用的「輪」自然聚集；低 QPS 時收益最大（p9） | 分析一次／輪，但還原逐個 Mirror 載 Master（p7 Alg.1） | 無 | — |
| 9 | Feather（SIGOPS ATC'26，依 PDF 頁首） | 〔部分〕 | 無 | RL state 含 *w*＝共用此 prefix 的等待請求數（p8 §4.3） | Add／Stop：決定何時停止加人並送出，不等未來到達（p8–9） | 同質 batch 用現有 kernel；與 PAT 整合列為未來（p12） | 同分時依到達序＋RL 探索（p12） | — |
| 10 | BatchLLM（MLSys'26 Industry） | 〔部分〕 | 無（工業負載無 swap，p9） | 以 prefix 群為排程單位，decode/prefill 比大者先（p4、p6 Eq.2–3） | 離線，整批已知 | 水平融合的 prefix-shared kernel（p11） | 不適合延遲敏感（p12） | — |
| 11 | RAGCache（arXiv 2404.12457；TOCS'25 NOT_VERIFIED） | 〔部分〕 | GPU＋host | PGDSF 優先權＝Clock＋頻率×成本／大小（p5–6）；重排優先權＝已快取長度／計算長度（p7） | 無 | 無 | 每請求一個等待窗，實驗設 32（p7、p11） | — |
| 12 | Cascade（arXiv 2608.06557） | 〔部分〕（共享面向） | HBM→DRAM→NVMe | 逐請求的預算；沒有跨請求合併（grep 無 dedup） | 無 | 無 | Tier-2 保留每批固定比例（p5）；各請求的還原從自己的預算出（p2、p6） | — |
| 13 | KVFlow（NeurIPS'25） | 〔部分〕 | CPU | 共享節點取子節點中最不該淘汰的優先權（p5） | status-aware：prefix 載入中的請求先跳過（p6） | 無 | 無 | — |
| 14 | Tokencake v5（EuroSys'27） | 〔部分〕 | CPU | 依 agent 型別彙總分數做保留（p8） | 無 | 無 | 有界 aging（Eq.5 p8） | — |
| 15 | Continuum v7 | 〔部分〕 | 看設定 | 無跨程式共享邏輯 | 無 | 無 | 程式層級 FCFS（p7） | — |
| 16 | FusionRAG（arXiv 2601.12904） | 〔部分〕 | disk／host | 熱門 chunk 跨 query 重用（p13） | 非同步載入，好了就併批（p14 Fig.11） | 無 | 無 | 共用 chunk KV，各 query 重算自己的關鍵 token（p13；細節未全核） |
| 17 | Shared RAG-DCache（arXiv 2504.11765） | 〔部分〕 | 共享 disk＋CPU | 熱門文件（p3） | 用排隊時間預取或生成 KV（p3） | 無 | 無 | — |
| 18 | Hydragen、FlashInfer cascade／composable、RelayAttention、ChunkAttention | 〔部分〕〔文件〕 | **無**，全部假設 HBM 常駐 | — | — | 攤提只在 HBM 內 | — | — |
| 19 | llm-d kv-cache scorer | 〔程式碼〕 | GPU／CPU | 最長連續前綴 × 每 tier 靜態權重（GPU 1.0／CPU 0.8） | 無 | — | — | — |
| 20 | NVIDIA Dynamo KV router | 〔文件〕 | GPU／host／disk | host 0.75、disk 0.25 的 credit；佇列可用 FCFS／WSPT | router 佇列只在 worker 忙時暫存（不是等共享者） | — | DRR 只收未命中 token | — |
| 21 | AIBrix（arXiv 2504.03648）＋router README | 〔部分〕〔文件〕 | 分散式 KV pool | prefix_match% 由高到低＋負載門檻；README 沒提 tier | 無 | — | 無 | — |
| 22 | PrefixShield（arXiv 2608.01657） | 〔部分〕 | 無（GPU；多層只在 related work） | — | — | — | 新建 block 者揹債；命中不收錢（p3、p5） | — |
| 23 | Quest／ShadowKV／InfiniGen／PQCache／RetroInfer／MagicPIG／ArkVale | 〔部分〕〔摘要〕 | 多數有 CPU | — | — | — | — | 逐請求或逐 query 選頁；多請求就是 batch 維度（PQCache p22）；同 context 不同 query 稀疏度差異大（RetroInfer p4） |
| 24 | HiSparse（arXiv 2608.07009，已合進 SGLang） | 〔部分〕 | host | — | — | — | — | 每請求、每層一個 GPU cache；併發請求的 miss 搶同一條 host link（p2、p7） |
| 25 | Fathom（2609.17652）、ReadKV（2610.11245） | 〔部分〕 | host（Fathom） | — | — | — | — | 聯集只跨 GQA head；offload 表格是 batch 1；多 session 情境只用推論、沒量（Fathom p12） |
| 26 | OasisKV（2608.08097） | 〔部分〕 | host／遠端 | — | — | — | — | 跨 head 取聯集；同 batch 的 miss 依層打包傳；不支援 prefix caching（p8、p13） |
| 27 | SSV（2605.19893）、TokenSpeed PR #2373、vLLM #47763 | 〔部分〕〔PR〕 | 無 | — | — | — | — | 只在單一請求內的多個 query 之間去重 |
| 28 | SCBench（ICLR'25） | 〔部分〕 | — | — | — | — | — | 共享 context 的 multi-request 模式，**只量品質** |

---

## §2 逐篇卡片（只寫 repo 沒有的新事實）

### 2.1 DLPM／D²LPM（必讀，全文）

**書目**：S. Cao, Y. Wang, Z. Mao, P.-L. Hsu, L. Yin, T. Xia, D. Li, S. Liu, Y. Zhang, Y. Zhou, Y. Sheng, J. Gonzalez, I. Stoica, "Locality-aware Fair Scheduling in LLM Serving," arXiv 2501.14312 v1（2025-01-24）。abs 頁沒有 Comments 也沒有 Journal-ref → **venue NOT_FOUND**。作者 13 人（abs 頁）。〔全文〕

**L3 已寫的不重複**：前綴只計費一次、DRR quantum、2.87× 等數字。以下是本次新讀到的。

**有沒有考慮多層 tier 或取回延遲？——沒有。**

- 全文 grep `offload`／`tier`／`swap`／`SSD`／`DRAM`／`host` 都沒有命中（只有「CPU scheduling overhead」的 CPU）。
- 快取只有 SGLang 的 GPU radix tree〔全文 p2 §2.1、p5 §5.1〕。
- 失去局部性的代價一律是**重算**：D²LPM 唯一輸給 Preble 的情況是 S2@Long-Context QA、D=8。原因是為了公平犧牲局部性，長文件得重算〔全文 p9〕。

**計費（Q4）**：

- 服務量＝w_e·(extend token 數)＋w_q·(輸出 token 數)，w_e＝1、w_q＝2。理由是 prefix 只在第一次算進並存進 GPU 記憶體時計一次〔全文 p4 §3〕。
- 〔判讀〕所以「先到的人付 prefix 錢、後到的人免費」；而且「付」的只是計算，不含取回。

**明列的未來工作（Q4 的直接證據）**：

- 本文只處理**同一 client 內部**的 prefix 共享，忽略跨 client 共享。
- 跨 client 共用同一 prefix 時「該由誰扣 quantum」被列為有趣的未來工作〔全文 p12 §8〕。

**演算法**：

- 每個 batching step：佇列依 matched prefix 長度排序。client 的 deficit counter q_i > 0 才能加入 batch，加入時扣 w_e·extend 長度，每個 decode step 再扣 w_q。所有 active client 都 ≤ 0 時，各補 Q_u〔全文 p4–5 §4.1，Alg.1〕。
- 公平界：兩個持續 backlogged 的 client，服務差 ≤ 2(U＋Q_u)，U＝w_e·L_input＋w_q·M〔全文 p4 Thm 4.1〕。
- 演算法 work-conserving，只改派送順序，不拒絕〔全文 p5〕。

**分散式（D²LPM）**：

- 全域排程器先找最長匹配的 worker 集合 G。G 中有正 deficit 的 worker，取佇列最短者；都沒有就退到所有有正 deficit 的 worker〔全文 p6–7 Alg.2〕。
- Q_w 設成無限大時，行為類似 Preble，此時不公平〔全文 p11 註腳、附錄 Thm A.5〕。

**實驗**：

- 實作在 SGLang 上；Llama-3.1-8B、Llama-3.2-3B；A100 80GB、A10G〔全文 p7 §6〕。
- 負載的平均 prefix／輸出長度：

  | 負載 | 平均 prefix | 平均輸出 |
  |:--|--:|--:|
  | Long-Context QA（LooGLE） | 21,449 | 15 |
  | Tree-of-Thoughts | 546 | 256 |
  | LLM-as-a-Judge | 2,701 | 256 |
  | Multi-Turn | 56 | 142 |

  〔全文 p7 Table 2〕
- 到達過程用合成的 Gamma process〔全文 p8〕。
- 良好 client 的延遲：Preble、RR＋LPM 最多是 D²LPM 的 7.18×、9.55×（平均 2.90×、4.06×）；VTC 最多 7.96×（平均 2.98×）〔全文 p10〕。
- Preble 在 Tree-of-Thoughts 上，把 client 0 的請求一直送到同一張 GPU。結果 client 1、2 在 100–700 s 之間幾乎拿不到服務，排隊最多約 600 s，叢集只發揮約 1/4〔全文 p11〕。

**相關工作**：

- 設計靈感來自 Deficit Round Robin 與 Delay Scheduling（Zaharia 等，EuroSys'10）〔全文 p12；參考文獻 [54] p15〕。
- 〔判讀〕DLPM 本身**沒有實作「為局部性而等」的 delay**，只借用了這個名詞。

### 2.2 SGLang：論文＋目前的程式碼＋PR

**論文**：L. Zheng *et al.*, NeurIPS'24, arXiv 2312.07104（讀 v2 PDF）〔部分〕

- cache-aware scheduling 依 matched prefix 長度排序。離線情況下，cache 容量 ≥ 最長請求時，DFS 序可達到最佳命中率，而 longest-shared-prefix-first 等價於 DFS〔部分 p6 Thm 3.1；證明 p14–16 §A.3〕。
- 原文自己寫：貪婪的 cache-aware scheduling 可能造成餓死，與公平排程的整合留待未來〔部分 p6〕。
- 未來方向列出：讓 RadixAttention 跨多層記憶體（DRAM、Disk），以及修正 cache-aware scheduling 的餓死問題〔部分 p10〕。

**目前 `main` 的行為**（`@66cac6b`，2026-10-26）〔程式碼〕

- **可選的政策**：
  - cache-aware：`lpm`、`dfs-weight`、`hrrn`、`shortest-prefill-first`
  - 不看 cache：`fcfs`、`lof`、`random`、`routing-key`
  - 〔`schedule_policy.py:172-187`〕
- **LPM 把 host 命中當成命中（Q1）**：
  - `num_matched_prefix_tokens = min(prefix_len + host_hit_length, max_len)`〔`common.py:260`〕，LPM 用它排序〔`schedule_policy.py:441-452`〕。
  - 〔判讀〕在 LPM 眼中，一段在 CPU、需要取回的 prefix 和一段在 GPU 的 prefix 完全等價；取回時間不進排序。
- **HRRN 也把 host 命中當作不用做事**：docstring 寫明 device＋host 都算 cached〔`schedule_policy.py:454-458`〕。
- **DFS-weight 確實在數等待人數（Q1）**：
  - 節點權重＝以它為最後匹配節點的等待請求數，加上子樹權重；兄弟節點依權重由大到小走〔`base_prefix_cache.py:347-387`〕。
  - 但傳進去的是 `req.last_node`，而 `last_node` 被設成 `match_result.last_device_node`〔`common.py:245-252`；`schedule_policy.py:496-502`〕。
  - 〔判讀〕只在 host 的節點不會成為分群依據，所以「一群人在等同一份 CPU 上的文件」不會被 DFS-weight 看成同一群（未實測）。
- **佇列長時退回 FCFS**：LPM 與 HRRN 在等待佇列 >128 時改用 FCFS，理由是 prefix 匹配與排序太貴〔`schedule_policy.py:137, 304-314`〕。
- **in-batch prefix caching（Q2 的「讓後到者等」）**：
  - 適用條件：device prefix ≤ 32 token（`IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD`）。
  - 用一棵「等待佇列的模擬 radix tree」比對；若與前面某個等待請求共用 ≥ 32 token（`..._DEPRIORITIZE_THRESHOLD`），就**暫時降級**，讓同 prefix 的只先排一個〔`schedule_policy.py:106-119, 346-396`〕。
  - 〔判讀〕一份文件整段在 host 時，device prefix 接近 0，會觸發這條路徑：第一個請求先去載，其他共享者被排到後面。這是合併（work avoidance），不是等未來的人。
- **HiCache 的 L3 預取策略**是逐請求的終止條件：`best_effort`／`wait_complete`／`timeout`。
  - timeout＝min(max, base＋per_ki_token × token 數／1024)，預設 base 2 s、0.1 s/1K token、max 30 s〔文件 `hicache_design.mdx` L98–117〕。
  - 本次沒看到「依共享者人數調整預取等待」的邏輯〔文件；NOT_FOUND〕。
  - 進行中預取的追蹤表以**請求**為 key〔程式碼 `unified_radix_cache.py:431, 2092`〕；兩個請求同時要同一段 L3 prefix 時會不會重複預取：NOT_VERIFIED。

**PR／issue**〔PR〕

- **#32911**（2026-09-08 合併）HRRN：
  - 測試環境：作者團隊的生產流量；sglang v0.5.15.post1、GLM-5.2 nvfp4、1P1D、2×8×B200；prefill 開 HiCache、chunk 65,536；混合 agent 負載，輸入最長 1M token。
  - conc＝100 時的 TTFT：

    | 政策 | 平均 | p99 |
    |:--|--:|--:|
    | FCFS | 24.2 s | 60.5 s |
    | LPM | 16.4 s | 213.3 s（+252.56%） |
    | DFS-weight | 19.0 s | 164.1 s（+171.24%） |
    | HRRN | 7.48 s | 55.48 s（−8.30%） |

  - 這是本次找到**最直接的「共享導向排序在 tier 化部署下餓死人」的數字**。工程聲明，沒有同儕審查。
- **#31954**（feature request，2026-09-20 因不活躍自動關閉）：
  - 生產環境：GLM-5.2-NVFP4、8×B300、agentic、平均輸入約 115K token、radix 命中率約 96%。
  - 中位數 TTFT 0.7 s，但 cache-miss 請求形成 20–60 s 的尾巴：約 5% 的請求貢獻約 75% 的累積 TTFT。
  - 提案 #31723 讓 LPM 的鍵加上 aging，2026-10-22 關閉、未合併。

### 2.3 vLLM：延後、移除 cascade、沒有期限的 pending

- **`OffloadingConnector` 的延後仍在 `main`**〔程式碼 `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:650-654, 1026-1050` @0f6b261〕：
  - 開 GPU prefix caching 時追蹤 `_chunks_being_loaded`；
  - 請求命中的 chunk 若正被載入，`get_num_new_matched_tokens` 回 None，整個請求延後；
  - backend 回 `HIT_PENDING` 時也延後（L793–852）。
- **PR #44295**（修 #44294 的 convoy；2026-10-26 查詢時 state＝OPEN，updatedAt 2026-10-15。L3 記為已關閉，以本次查詢為準，可能被重開）〔PR〕：
  - 作者在 2×L40S、Qwen3-8B、10K-token 共享 prefix 下：
    - 無條件取消延後，在 2× 溢出時吞吐損失更大（−38.0%，未修是 −20.5%）；
    - 作者稱延後在 GPU 有壓力時是「自然的節流」。
  - 〔判讀〕「讓共享者等先到者那次載入」不是純壞事：它同時是 work avoidance 與流量控制。
- **issue #49829**（OPEN）：`TieringOffloadingManager.lookup()` 對寫入中的 block 回 `HIT_PENDING` 沒有期限。寫入卡住時，請求一直延後到客戶端逾時；作者觀察到 aiohttp 預設 300 s 逾時〔PR〕。
- **PR #61126**（2026-10-12 合併，作者 WoosukKwon）移除 cascade attention〔PR〕：
  - 原因：預設關閉、FlashInfer 已硬關閉、多數 backend 不支援；
  - 代價：排程器每一步都要掃描 running 請求的 block 找共同前綴；
  - 一連串 bug（#55674、#47183、#39208、#55227、#53690）。
  - 〔判讀〕主流引擎的共享前綴 kernel 只看「正在跑的請求之間」的共同前綴，現在連這個都拿掉了。

### 2.4 Strata（共享面向）

**書目**：Z. Xie *et al.*, OSDI'26。USENIX 的 OSDI'26 議程頁由搜尋結果確認（presentation/xie-zhiqiang）。本次讀 arXiv 2508.18572 **v1**（13 頁），頁碼以它為準，OSDI 版頁碼可能不同。〔部分〕

- **delay hit 的兩種情境**〔部分 p6 §4.3.1〕：
  - 共用同一個 miss 的多個請求被排進同一 batch；
  - 非同步排程器在 context 還在算時就準備下一批。
  - 原文說非同步排程讓 miss 的解決窗口拉長到一整個 batch 的執行時間〔部分 p4 §3.2〕。
- **機制**〔部分 p6〕：
  - HiRadixTree 裡插入「transient node」，標記 in-queue 或 in-flight；
  - 請求匹配到 transient node 時，延到下一輪，但放在佇列最前面；
  - 只有匹配的 transient token 數超過門檻才延後，預設 100 token。
- **balanced batch＋bundle hit**〔部分 p6–7 §4.3.2，Alg.1〕：
  - 從隊首開始組批；加入請求會讓 batch 變成 loading-bound（load／compute 比 > 門檻，預設 100）時，把它移到降級清單；
  - 每加入一個請求，就掃整個佇列，把與它 bundle hit（同一 context）的請求優先加進來；
  - 原文說 bundle 同批可同時平衡載入與計算、減少 GPU 記憶體與 on-device 頻寬壓力。
- **防餓死**：降級的請求保留原順序，每次組批都從佇列第一個開始〔部分 p7〕。
- **SSD 那一層**：在 storage 命中時，預取到 host 與請求的排隊時間重疊；請求被派送時中止預取，用已在 host／GPU 的部分〔部分 p5 §4.2.1〕。屬於 latency hiding。
- **情境吻合**：LooGLE 與 NarrativeQA 被描述為「長 context 被多個使用者反覆查詢」的經典 RAG 用例〔部分 p8 §5.1〕。
- **cache distance 消融**〔部分 p10 §5.3.3〕：
  - 最小 cache distance（同 context 的請求排在一起）時不需要分層快取；delay-hit 處理讓峰值吞吐 +42%。
  - 最大 cache distance（均勻散開）時，delay-hit 處理**沒有幫助**，因為相似請求相距太遠，很少碰上同一次 miss。
- 〔判讀〕Strata 處理的是「剛好碰上」的共享者。它不會為了讓散開的共享者碰上而延後，也不依共享者人數排序。

### 2.5 Preble（只補 Q1–Q4）

- **淘汰成本 M_i＝Σ（被淘汰節點的 prefill 時間 × 該節點命中數 N_j）**：共用越多、越多人用，越貴〔部分 p5–6 §3.2〕。〔判讀〕這是「共享者人數加權」的最早形式之一，但只用在 GPU 內的淘汰與路由。
- **本地排程**：依「已快取 token 百分比」分 P 個優先群，組批時按比例從各群挑。例：P＝10、選 55 個時，第 10 群取 10 個、第 9 群取 9 個，依此類推。原文引 Wu 等（2023）說明「依共享量排序」會餓死〔部分 p7 §3.3〕。
- **消融**：優先權佇列改善 p99、不改平均〔部分 p10〕。
- 全文找不到 offload／host 相關機制（grep）。

### 2.6 PEEK（全文；新搜尋中最接近的第 1 篇）

**書目**：B. Xie, + 3 人, "PEEK: Predictive Queue-Informed KV Cache Management for LLM Serving," arXiv 2607.02525（v1 2026-05-10；讀 v2）。Comments 寫「Preprint, under review」→ venue NOT_VERIFIED。程式碼：p1 註腳給 `github.com/xiexbing/peek`，但 p10 結論寫 Code will be released，兩處不一致。〔全文〕

**核心觀察**：等待佇列本身就是有結構的負載。SGLang 的 LPM／DFS_WEIGHT 只看當下 cache 狀態，看不到等待請求彼此的共享關係〔全文 p1–3 §1–2，Table 1〕。

**機制**〔全文 p4–5 §3〕：

- 對等待佇列維護一棵**增量 pending radix tree**，每個內部節點記錄 `pending_count`（經過它的等待 prompt 數）；最深的 count ≥ 2 祖先就是 cluster。
- cLPM 的排序鍵：(section, −h, −score, −|cluster|, arrival)。
  - section：warm、pioneer、sibling 三種；
  - score：沿路徑的 pending_count × 邊長加總。
  - 這讓任何 pioneer 都排在任何 sibling 前面，同 cluster 緊鄰。
- **群組連續（GM，可選）**：一個 cluster 全部連續送出，等於「1 次 prefill＋size−1 次命中」。
- **淘汰鉤子**：沿祖先鏈的 pending_count × 深度，最低的先淘汰，零需求的 block 最先淘汰。
- **公平 lane**：以到達序為鍵的 Lane B，與 cLPM 的 Lane A 做 stride 交錯。
  - Lane B 的比例 1−α 由單身請求比例與最老單身請求的等待時間（除以 2 s）動態調整，夾在 [0.10, 0.60]；
  - 最老的請求至少每 ⌈1/(1−α)⌉ 輪前進一次。

**結果**〔全文 p2、p7〕：

- 最多 4×H100；SGLang 與 vLLM；Qwen2.5-32B、Llama-3.1-70B 等。
- 相對各引擎最好的原生基線：命中率 3.0×／2.6×、TTFT 7.9×／7.1×。
- cLPM 單獨就達到全套的 3 pp 以內。
- GM 讓 TTFT／E2E 變差 10–29%（延後了單身請求），DL 救回 2.4–15.1%。
- 原文明說 GM「延後單身請求」〔全文 p7 Takeaway 1〕。
- **TPOT 比基線高 8–26%**〔全文 p7〕。

**跟本題的關係**：

- **沒有下層 tier**。全文 grep 找不到 HiCache、offload、host memory；LMCache 只出現在 related work。「KV pressure tier」指 2–16× 的超賣〔全文 p5 §4.1，p9〕。
- W2「長文件 RAG」的意思是文件超出 KV arena，不是從 CPU 取回〔全文 p7 §4.2〕。
- **Q2**：pioneer-first 讓 siblings 等 pioneer 先 prefill（合併，不是等未來）。PEEK 沒有「等更多 sibling 到齊再開始」的窗口〔全文 p4–5〕。
- **限制**：單身請求為主時，N/C 攤提接近 1；cLPM 排序在 Python；只在 H100 評估〔全文 p10〕。
- 〔判讀〕PEEK 已經把「看佇列裡的共享者人數」做成主角。我們若主張「依共享者人數排序」，必須說明**下層還原成本**怎麼改變它的決策；否則審稿人會說是 PEEK。

### 2.7 TokenDance（全文；新搜尋中最接近的第 2 篇）

**書目**：Z. Bian, + 5 人（北大；與 Tokencake 同團隊），"TokenDance: Scaling Multi-Agent LLM Serving via Collective KV Cache Sharing," arXiv 2604.03143 v1（2026-04-03）。Comments 寫 preprint under review → venue NOT_VERIFIED。〔全文〕

- **情境**：multi-agent 的 All-Gather 輪次。每個 agent 的下一輪 prompt＝私有歷史＋所有 agent 上一輪的共同輸出區塊〔全文 p1、p3 §2〕。
- **collective reuse**〔全文 p5 §4.2〕：
  - KV Collector 把同一輪中「可相容」的請求（相同 active prompt 長度、相同可見快取區段、slot 不重疊）合成一組；
  - 每層一次 batched RoPE，在檢查層做一次 key-difference，找出重要位置；
  - 「分析成本每輪付一次」，只有最後的 cache 更新是各請求自己的。
- **Master-Mirror 儲存**：一份 dense Master；其餘存成 block-sparse diff；7B 壓縮 11.2×、14B 壓縮 17.5×〔全文 p6 §4.3，p10 §6.4〕。
- **還原（Q3 關鍵）**：
  - Alg.1 的標題是「Fused Diff Restore for **One Mirror Request**」：逐層把 Master chunk 載進 ping-pong buffer，套用 diff、RoPE 還原，再寫入 paged KV〔全文 p7 §4.4〕。
  - 實作擴充 LMCache GPU connector 的 layerwise 傳輸路徑〔全文 p8 §5〕。
  - 〔判讀〕Master 對每個 Mirror 請求各載一次，**I/O 並沒有「載一次給 N 個人」**；攤提的是 RoPE 與選位置的計算。
- **Q2**：
  - 分組靠應用層的同步輪次；vLLM V1 adapter 在每個排程 step 掃描請求並分組〔全文 p7–8 §5〕。
  - 原文說收益在低 QPS 最大，因為系統有時間完整利用分組；高 QPS 時 GPU 計算飽和，收益收斂到 1.2–1.6×〔全文 p9 §6.3〕。
  - 沒有刻意等待的機制。
- **實驗**：A100 80GB；Qwen2.5-7B／14B；GenerativeAgents、AgentSociety；SLO 1500 ms 下，可同時服務的 agent 數最多是 vLLM 的 2.7×（摘要）〔全文 p8–9〕。
- **Q4**：無。

### 2.8 Feather（"Requests of a Feather Must Flock Together"）

**書目**：S. Rathi, + 2 人, arXiv 2605.06046（讀 v3）。PDF 頁首印 SIGOPS ATC 2026（2026-11-15～18，香港），**只依 PDF 頁首，未開會議議程頁**。〔部分〕（§1–3、§4.3、§6 starvation 段）

- **主張**：在 prefix 共享負載下，較小但 prefix 同質的 decode batch，吞吐可能比較大的異質 batch 高，因為存取 KV 時的局部性較好〔摘要；部分 p1–2〕。
- **RL 排程（Q1）**〔部分 p8–9 §4.3〕：
  - state＝(b, Δ, w)：b＝目前 batch 大小，Δ＝共享 prefix 的損失，w＝**共用該 prefix 的等待請求數**；
  - 動作是 Add 或 Stop，reward 是 decode 吞吐。
  - 〔判讀〕w 就是「等待中的共享者人數」，用來判斷「現在犧牲局部性，之後會不會被補回」。但 Stop 是立刻送出，不是等待未來到達。
- **防餓死**：同分時依到達序；RL 探索期會週期性放入冷門 prefix〔部分 p12〕。
- 與 prefix-aware kernel（PAT）的深度整合列為未來工作〔部分 p12〕。
- 全文 grep 找不到 offload／host。

### 2.9 BatchLLM、BlendServe、AlignedServe（批次與合批）

- **BatchLLM**（Z. Zheng *et al.*, arXiv 2412.03594 v3；Comments 寫 MLSys 2026；PDF p1 頁尾是 MLSys'26 Industry Track）〔部分〕
  - 離線、整批已知：先在全域找出共同 prefix，把請求分組；**以群為排程單位**〔部分 p4 §3.1〕。
  - 群內依「decode 長度 ÷ prefill 長度」比例排序，比例大的先〔部分 p6 Eq.2–3〕。
  - 水平融合的 prefix-shared attention kernel，與 FlashInfer cascade 比較〔部分 p11 Fig.11〕。
  - 工業負載下沒有 KV swap 或重算〔部分 p9〕。
  - 自列限制：可能拉長單請求延遲，不適合延遲敏感；建議未來做串流與批次模式切換〔部分 p12〕。
- **BlendServe**（Y. Zhao *et al.*, arXiv 2411.16102）〔摘要〕：離線；用 resource-aware prefix tree，同時做資源重疊與 prefix 共享；最多 1.44×。
- **AlignedServe**（F. Bai *et al.*, arXiv 2605.23389）〔摘要〕
  - 這裡的「prefix-aware batching」指把 **KV 長度相近**的請求放同批，減少同一 iteration 內的空泡；不是共享前綴。
  - 用大量 CPU 記憶體保留在途請求以便湊批，並有「一張 GPU 幫另一張預取 KV」的架構。
  - venue：Continuum v7 參考文獻 [10] 寫 Proceedings of the ACM on Management of Data 4(3), 2026〔部分 Continuum p13〕，未開 ACM 頁。

### 2.10 RAG 系：RAGCache、Shared RAG-DCache、FusionRAG

- **RAGCache**（C. Jin *et al.*, arXiv 2404.12457 v2；venue：只有第三方彙整頁寫 ACM TOCS 2025 → NOT_VERIFIED）〔部分〕
  - 知識樹跨 GPU 與 host；PGDSF 優先權＝Clock＋Frequency × Cost／Size。Frequency 是時間窗內被檢索的次數，所以是**歷史頻率**，不是等待人數〔部分 p5–6 §5.1〕。
  - swap-out-only-once：host 保留一份，之後在 GPU 淘汰時直接釋放〔部分 p7〕。
  - 重排優先權＝已快取長度 ÷ 計算長度；為防餓死，每個請求設一個等待窗，保證不會晚於窗大小才被處理〔部分 p7 §5.2〕；實驗窗大小 32〔部分 p11〕。
  - A10G 24GB、PCIe 4.0×16〔部分 p9〕。
  - 〔判讀〕優先權沒有算進 host→GPU 的取回時間。
- **Shared RAG-DCache**（H. Lee *et al.*, arXiv 2504.11765）〔部分〕
  - 多個 LLM 實例共用 disk 上的文件 KV。
  - 對排隊超過門檻的請求，預先取回或生成它需要的文件 KV，用掉排隊時間〔部分 p3〕。屬於 latency hiding。
  - 2 GPU、1 CPU；吞吐 +15–71%〔摘要〕。
- **FusionRAG**（J. Wang *et al.*, arXiv 2601.12904 v1）〔部分〕
  - 熱門 chunk 被多個 query 用到時，標準 prefix cache 會在不同前綴下存多份；FusionRAG 改成可跨前綴重用〔部分 p13〕。
  - 非同步 KVCache 排程器：KV 在 disk／host 時發出非同步載入，其間 GPU 先跑別的；載好的請求再用 continuous batching 併入〔部分 p14 Fig.11〕。
  - 載入仍是**逐請求**。
  - 搜尋摘錄提到「各 query 的重算 token 寫到專屬頁」，但 v1 全文 grep 沒找到原句 → NOT_VERIFIED。

### 2.11 Cascade（只看共享面向；restore 見 L8b）

- 全部以**單一請求**的預算為核心：
  - 原文的切入點：現有 prefix cache 只判斷物件「整體上划不划算」，不判斷「要用它的那個請求等不等得起」〔部分 p2 §I〕。
  - 剩餘預算 B_rem＝B_r −（t_now − a_r）〔部分 p6 Eq.5〕。
- 全文 grep 找不到 dedup／同 prefix 多請求合併。
- 〔判讀〕多個共享者同時需要同一段深層 prefix 時，Cascade 會各自用自己的預算計算 restore 上限，不會合併。
- **公平性**：
  - 以 context 長度分類算 Jain index〔部分 p4 §III-A〕；
  - 負預算的請求降到 Tier-2 佇列，每批保留固定比例給 Tier-2，防止持續過載時餓死〔部分 p5 §IV〕。

### 2.12 KVFlow、Tokencake、Continuum（只補跨 agent 共享＋預取重疊）

- **KVFlow**〔部分〕
  - 共享節點取所有子節點中最低（最不該淘汰）的優先權，只要有任何 agent 很快會用到就留著〔部分 p5 §3.2〕。〔判讀〕看「有沒有人要」，不是看「幾個人要」。
  - 預取有併發上限〔部分 p5 §3.3〕。
  - status-aware scheduling：prefix 正在載入的請求先跳過，避免重複載入〔部分 p6〕。
- **Tokencake v5**（2026-09-28；Comments 寫 EuroSys'27）〔部分〕
  - 每請求優先權＝結構重要性＋同步壓力＋有界 aging〔部分 p8 Eq.5〕。
  - agent 型別分數彙總同型別所有 active 請求，決定要不要為整個型別保留記憶體〔部分 p8〕。
  - 參考文獻引用 TokenDance〔部分 p14〕。
- **Continuum v7**（2026-09-08）〔部分〕
  - 排序：被搶佔 > TTL 內釘住 > 程式層級到達序（FCFS 公平）〔部分 p7〕。
  - 全文 grep "prefix" 只有 3 處，沒有跨程式共享前綴的邏輯；related work 提到 AlignedServe〔部分 p3〕。

### 2.13 共享前綴 attention kernel（只問 Q3）

| kernel | 讀到 | 攤提方式 | 有沒有提到 KV 在 CPU／SSD？ |
|:--|:--|:--|:--|
| **Hydragen**（J. Juravsky *et al.*, arXiv 2402.05099 v2；venue NOT_VERIFIED） | 〔部分〕 | 把 prefix attention 與 suffix attention 分開；prefix 部分跨序列批次化成矩陣乘法。CodeLlama-13b 吞吐最多 32×（對 vLLM）〔p1–2〕。prefix 從 1K 增到 16K，吞吐降不到 15%〔摘要〕。 | 沒有。原型要使用者指定共享位置；整合進持續接收請求的系統留待未來〔p11〕。相關工作提到別的系統在 GPU／CPU／disk 間搬 KV，但 Hydragen 本身不處理〔p12〕。 |
| **FlashInfer cascade inference**（部落格，2024-02-02） | 〔文件〕經 WebFetch | 共享 prefix 用 multi-query kernel 讀一次，unique suffix 用 batch decode，再用 merge operator 合併。H100 SXM：32,768-token 共享 prefix、batch ≥128、unique ≤256 時，最多 31×（對 vLLM 0.2.6）。只有 kernel 層 benchmark。 | 沒有（只談 GPU 內的記憶體層級） |
| **FlashInfer 論文**（Z. Ye *et al.*, arXiv 2501.01005；MLSys 2025） | 〔部分〕 | composable formats：共享 prefix 對應的稠密子矩陣用較大 block 存，**不搬資料，只算索引**〔p5 §3.1.2〕。MLCEngine 平行生成 n＝4 時，ITL −13.73%／−17.42%、TTFT −16.41%／−22.86%（8B／70B）；n 太小或太大都沒好處〔p9–10〕。 | 沒有 |
| **RelayAttention**（L. Zhu *et al.*, arXiv 2402.14808；ACL 2024） | 〔部分〕 | 一個 batch 只從 HBM 讀一次 system prompt 的 KV〔p1〕 | 沒有（「DRAM」指 GPU 的 off-chip HBM） |
| **ChunkAttention**（L. Ye *et al.*, arXiv 2402.15220；ACL 2024） | 〔部分〕 | prefix-tree 分塊 KV＋two-phase partition；system prompt 1024–4096 時 kernel 快 3.2–4.8×〔p1〕 | 沒有。KV 只存正在 decode 的序列〔p2〕；放 CPU 的是 prefix tree 的 metadata，不是 KV〔p6 §3.3〕。限制：共享段必須在開頭〔p9〕。 |

- vLLM 的 cascade attention 已移除（§2.3）。

**〔判讀〕Q3 的答案**：

- 所有攤提 kernel 都把「共享 prefix 已在 HBM、共享者已同時 running」當成前提；
- 沒有任何一篇把「何時還原」與「共享者何時聚在一起 decode」一起考慮；
- 也沒有人量過「還原一次後，讓共享者用 cascade kernel 一起 decode」相對於「各自 decode」的收益與還原等待時間的取捨。NOT_FOUND。

### 2.14 KV 感知路由：分數有沒有算進下層 tier？

- **llm-d**（`llm-d-kv-cache` @822e6dd）〔程式碼〕
  - `LongestPrefixScorer`：對最長連續匹配的每個 block，取該 pod 在各 tier 的最大權重累加〔`pkg/kvcache/kvblock_scorer.go:77-153`〕。
  - 預設權重 GPU 1.0、CPU 0.8〔`pkg/kvcache/backend.go:26-31`〕。
  - 〔判讀〕下層命中是**靜態打折**，跟取回時間、在途載入、等待中的共享者人數都無關。
- **NVIDIA Dynamo KV router**（@ee01209）〔文件〕
  - `--router-host-cache-hit-weight` 預設 0.75、`--router-disk-cache-hit-weight` 預設 0.25〔`configuration-and-tuning.md` L75–76〕。這些權重也影響佇列排序用的「已快取 token 估計」〔同檔 L169〕。
  - tier 感知路由依框架而異：
    - vLLM：v0.24.0+、Dynamo v1.3.0+ 才有；
    - SGLang HiCache：有；
    - TensorRT-LLM：GPU＋RAM 合在一起看，權重不適用。
    - 〔`offloading-support-matrix.md` L16–18〕
  - DRR 佇列成本＝max(1, 輸入 token − cached token)，入隊時快照、之後不重算〔`deficit-round-robin.md` L25–30〕。〔判讀〕從 host／disk 取回的 I/O 不在成本裡。
  - 佇列只在所有 worker 都忙時暫存請求，用來在送出時拿到最新的負載資訊；不是為了等共享者〔同檔 L80〕。
- **AIBrix**（arXiv 2504.03648；作者「The AIBrix Team」；venue NOT_VERIFIED）〔部分〕〔文件〕
  - 論文的 prefix-cache-aware 政策：命中超過門檻者優先〔部分 p5 §3.2〕。
  - 目前的 router README（@d459142）：依 prefix_match% 由高到低，加上負載門檻（平均＋k·σ），沒提 tier〔文件 `prefix_cache_readme.md` L84–103〕。
  - repo 內有 `prefixcache/preble.go`（Preble 風格的實作；只確認檔案存在）〔程式碼〕。

### 2.15 其他公平性與「誰付」

- **PrefixShield**（Z. Wang, + 1 人, arXiv 2608.01657）〔部分〕
  - 「admission responsibility gap」：建立新 KV block 的群組會讓別的群組失去可重用狀態。
  - 做法：對新建立的 full block 計量；純命中不收錢；以預估債務選出要提供淘汰候選的群組。原文說「債決定誰付」〔部分 p3、p5〕。
  - 實作在 vLLM；**只有 GPU**（「multi-tier」只出現在 related work，p9）。
- **Resource-Fair Scheduling**（D. Yao, + 1 人, arXiv 2608.02244）〔摘要〕：同批 decode 的成本由最長的 KV 主導，短請求被「外部性」拖累；提出 ISJL，並討論與 token 計價的對齊。與共享無關。
- **FairServe**（R. I. S. Khan *et al.*, arXiv 2411.15997）〔摘要〕：MS CoPilot 的數百萬請求分析；節流＋加權服務計數器。不看 prefix。
- **TOPAS**（arXiv 2608.25523）〔摘要〕：task 層級的 aging 防餓死；決策會算進 prefix 搬移與搶佔的成本（L3 H10 已列摘要）。
- **以下未核實（NOT_VERIFIED）**：
  - CacheCast：只在 ResearchGate 看到，搜尋摘要說有跨 HBM／DRAM／SSD 的公平放置，沒找到 arXiv，未開原文。
  - GitHub `srujan20/prefix-cost-attribution`：搜尋摘要稱用 prefix trie 上的 Shapley 值分攤成本，單一作者 repo，未開。

### 2.16 Q5：稀疏選頁 × 多個請求共用同一份長 context

| 工作 | 讀到 | 選頁單位 | 評估是單請求還是多請求？共用前綴嗎？ | 跨請求去重或聯集？ |
|:--|:--|:--|:--|:--|
| Quest（arXiv 2406.10774；ICML'24） | 〔摘要〕＋grep | 每頁記 key 的 min／max，**每個 query** 選 Top-K 頁 | 系統效率在「single-batch」量〔p8〕 | 無 |
| ShadowKV（arXiv 2410.21465；venue NOT_VERIFIED） | 〔摘要〕＋grep | 每條序列一個低秩 K、V offload 到 CPU、chunk 層級選取 | batch 是不同序列；投影「每條序列」各算〔p5〕 | 無 |
| InfiniGen（arXiv 2406.19707；OSDI'24） | 〔摘要〕 | 用下一層的 rehearsal 推測重要 token，只從 host 預取那些 | 多 batch 大小，各自不同序列 | 無（NOT_FOUND） |
| PQCache（arXiv 2407.12820；venue NOT_VERIFIED） | 〔部分〕 | PQ 碼近似找重要 token，再取回 | 多請求＝在計算中加 batch 維度〔p22〕 | 無 |
| RetroInfer（arXiv 2505.02922；PVLDB 19(5) 2026） | 〔部分〕 | wave index＋GPU-CPU wave buffer | 原文觀察：**同一 context 上不同 query 的稀疏度差異很大**〔p4 Fig.4(b)〕 | 無 |
| MagicPIG（arXiv 2410.16179；venue NOT_VERIFIED） | 〔摘要〕 | LSH 抽樣，在 CPU 算 attention | batch 是不同序列 | 無 |
| ArkVale（NeurIPS'24 proceedings） | 〔部分〕grep | page digest；在 CPU 備份並可召回 | 一般 batch | 無（grep 無 prefix 或跨請求） |
| ReadKV（S. Bhandari *et al.*, arXiv 2610.11245） | 〔部分〕 | 依 query 決定每個 key／value 讀幾個 bit | A10G、8K token、**batch 1**、單層〔p13–14 Table 4〕 | 無 |
| Fathom（V. Kalyanarangan, arXiv 2609.17652 v3） | 〔部分〕 | 依 query 決定掃 K 的位元深度 | offload 表格是 batch 1，只有一個 batch 2 檢查；「多 session」情境用 index 大小推論，**沒實測**〔p12 Limitations〕 | 只跨同一 KV head 的 4 個 query head 取聯集〔p1〕 |
| HiSparse（Z. Xie *et al.*, arXiv 2608.07009；已合進 SGLang） | 〔部分〕 | 與 indexer 無關（DSA、NSA、Quest），每請求、每層一個 GPU LRU cache | 多併發，但每請求 32K 輸入，host 存「每個 active 請求」的完整 KV〔p7 Fig.2，p9〕 | 無；併發請求的 miss 搶同一條 host link〔p2〕 |
| OasisKV（C. Xiao *et al.*, arXiv 2608.08097） | 〔部分〕 | 用 speculative decoding 的草稿 token 預測未來要的 block，預取到 HBM | 多請求 | 跨 head 取聯集；同 batch 不同請求的 miss **依層打包傳輸**（不是去重）〔p8〕；原型不支援 prefix caching〔p13〕 |
| SSV（Z. Wang *et al.*, arXiv 2605.19893 v2；HTML v1 標題 SpecSA） | 〔部分〕 | NSA | 單一請求的多個 verify query | overlap-aware kernel 對重疊的選取 block 去重〔p2〕，**只在請求內** |
| TokenSpeed PR #2373（2026-10-15 合併） | 〔PR〕 | QSA | 單請求 | 同一請求連續 query 的 2048 個選取 slot 完全相同時共用 KV |
| vLLM issue #47763（OPEN） | 〔PR〕 | MiniMax-M3 sparse decode | verify-as-decode | 提議「每個 request-tile 載入選取 block 的聯集」，只在請求內 |
| SCBench（Y. Li *et al.*, arXiv 2412.10319；ICLR'25） | 〔部分〕 | 評估 benchmark | 有 multi-turn 與 multi-request 兩種共享 context 模式；結論之一：sub-O(n) 記憶體的方法到後續請求會掉分〔摘要、p3〕 | 只量品質，不量 I/O |
| KVDrive（arXiv 2605.18071）、ParisKV（arXiv 2602.07721；ICML'26） | 〔摘要〕 | 多 tier、依 query 取回 | 摘要沒提共享 context | NOT_FOUND |

**〔判讀〕對第三階段主線（GPU 留縮印版，使用者回來時用新問題逐層挑頁，只從 CPU／SSD 取回原版）的意義**：

1. 「多個使用者問同一份文件時，把挑到的頁取聯集、去重、依熱度排序取回」本次**沒找到前例**。查法：§「查了什麼」中的 Q5 相關搜尋 4 次，加上上表 16 項逐篇 grep `shared prefix`／`batch`／`multi-request`／`union`／`dedup`。
2. 但**可行性的關鍵數字，也就是聯集大小，沒有人量過**。RetroInfer 只說不同 query 的稀疏度差異大，沒說選中頁的重疊率。這是 NOT_MEASURED，也是最小實驗的第一題。
3. 最自然的落腳點是 HiSparse 的架構（per-request GPU cache＋host 全量），也就是在 SGLang 內把「每請求 cache」改成「每文件共用 cache＋每請求視圖」。審稿人一定會拿 HiSparse 當 baseline。

### 2.17 經典名詞（只當詞彙用，不當前作）

- **Delay Scheduling**（M. Zaharia *et al.*, EuroSys'10）：為了資料局部性，讓工作在佇列裡短暫等待。DLPM 引為靈感來源〔標題，依 DLPM 參考文獻 [54] p15〕。
- **Caching with Delayed Hits**（N. Atre *et al.*, SIGCOMM'20）：miss 還在解決時，同物件的後續請求也要等。Strata 引用並移植到 LLM 服務〔標題，依 Strata 參考文獻 [5] p12〕。
- **Triton dynamic batcher「Delayed Batching」**：`max_queue_delay_microseconds` 讓批次器最多等這麼久來湊批；湊滿就立刻送出，否則逾時送出較小的批；預設不延遲〔文件，WebFetch Triton 2.67 存檔頁〕。這是唯一找到的「刻意等待以湊批」的服務端機制，但跟內容無關。

---

## §3 對三向交集（Sharing × Latency Hiding × Scheduling）的意義

### 3.1 已經被做掉的部分

| 子問題 | 已有做法 | 出處 |
|:--|:--|:--|
| 看佇列裡「幾個人共用這段 prefix」來排序或分群 | PEEK cLPM＋GM；SGLang DFS-weight；Feather 的 w；BatchLLM 的群排程 | §2.6、§2.2、§2.8、§2.9 |
| 依共享程度保護快取不被淘汰 | Preble 的淘汰成本 × 命中數；PEEK 的淘汰鉤子；KVFlow 的共享節點優先權；RAGCache 的 PGDSF 頻率 | §2.5、§2.6、§2.12、§2.10 |
| 讓後到的共享者等先到者那次計算或載入（避免重複） | Strata delay-hit；vLLM `_chunks_being_loaded`；SGLang in-batch deprioritize；KVFlow status-aware；PEEK pioneer-first | §2.4、§2.3、§2.2、§2.12、§2.6 |
| 把共享者放同一批，讓載入配上計算（hiding） | Strata bundle hit＋balanced batch＋bubble filling；FusionRAG 非同步載入；Shared RAG-DCache 用排隊時間預取；Strata 的 SSD 預取 | §2.4、§2.10 |
| 共享前綴的 attention 攤提（HBM 內） | Hydragen、FlashInfer cascade／composable、RelayAttention、ChunkAttention、BatchLLM fused kernel | §2.13 |
| 共享導向排序的餓死與對策 | SGLang（承認＋PR 實測）；DLPM、PEEK lane、Preble 分群配額、RAGCache 等待窗、Strata 保序、Cascade Tier-2、Feather 探索、Tokencake／TOPAS aging、SGLang HRRN | §2.1–2.12、§2.15 |
| 共享成本誰付（GPU 內） | DLPM（第一次算的人付一次）；PrefixShield（建新 block 的人揹債）；Dynamo DRR（只收未命中） | §2.1、§2.15、§2.14 |
| 路由考慮下層 tier | llm-d（CPU 0.8）、Dynamo（host 0.75、disk 0.25）——靜態權重 | §2.14 |
| 依延遲預算決定還原多少（單請求） | Cascade | L8b、§2.11 |

### 3.2 明確沒被做的部分（查不到≠沒有）

**查法**：

- WebSearch 14 次，關鍵字包括：
  - prefix-aware scheduling starvation、cache-aware scheduling fairness
  - "shared prefix" KV offload scheduling、"batching window"／delay scheduling／anticipatory
  - multi-tenant shared document RAG serving、cost attribution
  - Q5 的 sparse／shared-context／union／dedup 共 4 組
- 讀了上面 28 列論文與系統的相關段落；`gh` 查 SGLang／vLLM 的 issue 與 PR（starvation、HRRN、LPM、being loaded、cascade attention）。
- 下面每一項都是「在這些來源中沒找到」。

1. **依「等待中的共享者人數 ×（下層 tier 的取回成本或可避免的重算成本）」排序或准入**
   - PEEK、Feather、DFS-weight 只看 GPU，或只認 device 節點。
   - LPM、llm-d、Dynamo 把下層命中當成等價或靜態折扣，不看取回時間、在途頻寬、人數。
2. **為了湊到更多共享者而刻意延後開始取回（anticipatory window）**
   - 現有的「等」都是等**已經在飛**的那一次（Strata、vLLM、SGLang、KVFlow、PEEK）。
   - Strata 的消融顯示：共享者散開時，delay-hit 處理沒有收益〔p10〕。這正是 anticipatory window 想處理的情境。〔判讀〕
3. **還原一次之後，讓共享者用共享前綴 kernel 一起 decode，並把「還原完成時間」與「共享者聚集程度」一起排程**
   - kernel 全部假設已在 HBM；Strata 的 bundle 只到 prefill 同批；TokenDance 的 I/O 仍逐請求；vLLM 已移除 cascade。
4. **下層 tier 取回的 I/O 成本在共享者之間怎麼分攤，以及非共享者會不會因為「一次取回多人受益」的優先權而被餓死**
   - DLPM 把跨 client 誰付列為未來工作；PrefixShield 只管 GPU；Dynamo 不收取回成本。
5. **Q5：跨請求的選頁聯集、去重、依頁熱度排序、把挑到的頁重疊多的請求排在一起**
   - 只有單一請求內的聯集或去重（GQA head、spec-verify query）。
   - 聯集大小 NOT_MEASURED。

### 3.3 對新提案的威脅等級

| 若提案主打… | 威脅 | 理由 |
|:--|:--|:--|
| 「共享者感知的准入／合併」：讓後到的等先到的、同 prefix 一起跑 | **高** | PEEK（全文）已把佇列 cluster 做成主角並有公平 lane；Strata（OSDI'26）有 delay-hit＋bundle；vLLM／SGLang 程式碼本身就有延後與降級。審稿人會說「PEEK＋Strata」。 |
| 「tier 感知的共享者排序」：優先權＝f(共享者人數, 還原時間, 頻寬爭用)＋公平 | **中** | 每個成分都有前例：人數（PEEK、Feather）、tier（Strata、Cascade）、公平（DLPM、HRRN）；但沒人組合。必須證明**決策會翻轉**：同一份佇列，在不同 κ、不同 tier 下最佳順序不同；而且只看 GPU 的 PEEK 或只看預算的 Cascade 會做錯。生產 PR #32911 已顯示 LPM 在 HiCache 下 p99 很差，這是可引用的動機。 |
| 「anticipatory window」：為等未來的共享者刻意延後還原 | **低到中** | 沒找到前例（威脅低）。但價值取決於到達型態：共享者成群到時 Strata 的 delay-hit 已經夠；均勻散開時要等很久，與 TTFT SLO 衝突。需要先量真實 trace 中「同一文件的到達間隔」分佈，否則審稿人會質疑情境。 |
| 「還原一次 → 共享者一起 decode（cascade）」 | **低到中** | 沒找到前例；但 vLLM 移除 cascade 是反證：實務上收益不足以抵消複雜度。必須顯示在 128K、共享者夠多時，decode 端的攤提是顯著的。FlashInfer 的 31× 是 kernel 層級、32K prefix、batch ≥128 的條件。 |
| 「Q5 跨請求選頁聯集或頁熱度排序」 | **低** | 沒找到前例；可行性未知（聯集大小沒人量）。HiSparse 是最近的系統，審稿人會要求以它為 baseline。 |

### 3.4 審稿人會要求的 baseline

1. **FCFS**（vLLM 預設），以及 **vLLM `OffloadingConnector` 的原生延後**（含 PR #44295 的變體）。
2. **SGLang LPM**、**DFS-weight**、**HRRN**（2026-09 合併，生產 PR 顯示 p99 最好），都開 HiCache。
3. **PEEK**（cLPM＋GM＋DL＋PE；有 SGLang 與 vLLM 的 shim）。
4. **Strata**（delay-hit deferral＋balanced batch＋bubble filling；建在 SGLang 上；開源狀態本次未查，NOT_VERIFIED）。
5. **DLPM／VTC**（公平基線）；分散式時加 **D²LPM** 與 **Preble**。
6. **Cascade**（LRBF＋預算切分；主要結果來自模擬器，可能只能重實作規則）。
7. **Feather**（若主張 decode 端的同質批）。
8. **天真的固定 batching window**（Triton 式 `max_queue_delay`）與**知道未來到達時刻的 oracle**：用來界定 anticipatory window 的上下限。
9. **Q3**：同樣還原後，各自 decode 對比 FlashInfer cascade／composable 一起 decode。
10. **Q5**：每請求的 Quest 或 HiSparse 對比「聯集取回」。
11. **路由層**（多實例時）：llm-d 的 tier 權重、Dynamo 的 host／disk 權重。

---

## §4 不確定性與沒做的事

- **沒做跨模型審查**：本檔是 self-review。
- **WebFetch 有小模型整理**：WebFetch 讀的頁（FlashInfer 部落格、Triton 文件）經工具內小模型整理，細節可能失真；PDF 與程式碼是本 agent 自己讀的。
- **版本差異**：Strata 讀的是 arXiv v1，不是 OSDI'26 版，頁碼會不同。Tokencake、Continuum 讀的是 v5、v7，與 L3 讀的版本不同，L3 的頁碼不適用。
- **程式碼判讀沒有實測**：SGLang 的「host 上的文件會觸發 in-batch 降級」「DFS-weight 不認 host 節點」，以及 vLLM 的延後行為，都是讀程式碼推得，**沒有跑**（NOT_MEASURED）。
- **venue 未核實的**：DLPM、Hydragen、ShadowKV、PQCache、MagicPIG、RAGCache、Cascade、AIBrix、PEEK、TokenDance、Feather（只有 PDF 頁首）、AlignedServe（只依他文引用）。
- **在 MI300X／H200 上的行為**：全部 NOT_MEASURED。

---

## §5 參考清單（arXiv 一律打開過 abs 頁）

格式：ID｜標題｜第一作者（+人數）｜v1 日期｜Comments／Journal-ref｜本檔讀到

| arXiv | 標題（縮寫） | 第一作者 | 日期 | venue（abs 頁） | 讀到 |
|:--|:--|:--|:--|:--|:--|
| 2501.14312 | Locality-aware Fair Scheduling in LLM Serving | Shiyi Cao (+12) | 2025-01-24 | 無 → NOT_FOUND | 全文 v1 |
| 2312.07104 | SGLang | Lianmin Zheng (+11) | 2023-12-12 | 無（NeurIPS'24 見 L3） | 部分 v2 |
| 2407.00023 | Preble | Vikranth Srivatsa (+4) | 2024-05-08 | 無（ICLR'25 見 L3） | 部分 v2 |
| 2508.18572 | Strata | Zhiqiang Xie (+7) | 2025-08-26 | under peer review；OSDI'26 由 USENIX 頁（搜尋）確認 | 部分 v1 |
| 2607.02525 | PEEK | Bing Xie (+3) | 2026-05-10 | Preprint, under review | 全文 v2 |
| 2604.03143 | TokenDance | Zhuohang Bian (+5) | 2026-04-03 | preprint under review | 全文 v1 |
| 2605.06046 | Requests of a Feather… (Feather) | Saksham Rathi (+2) | 2026-05-07 | 無；PDF 頁首 SIGOPS ATC 2026 | 部分 v3 |
| 2412.03594 | BatchLLM | Zhen Zheng (+5) | 2024-11-29 | Accepted at MLSys 2026 | 部分 v3 |
| 2411.16102 | BlendServe | Yilong Zhao (+7) | 2024-11-25 | 無 | 摘要 |
| 2605.23389 | AlignedServe | Fengyao Bai (+5) | 2026-05-22 | 無（他文引為 PACMMOD 4(3) 2026） | 摘要 |
| 2404.12457 | RAGCache | Chao Jin (+6) | 2024-04-18 | 無 → NOT_VERIFIED | 部分 v2 |
| 2504.11765 | Shared RAG-DCache | Hyungwoo Lee (+7) | 2025-04-16 | 無 | 部分 v1 |
| 2601.12904 | FusionRAG | Jiahao Wang (+12) | 2026-01-19 | 無 | 部分 v1 |
| 2608.06557 | Cascade | Muhammad Adnan (+6) | 2026-08-06 | 無 | 部分（共享面向） |
| 2507.07400 | KVFlow | Zaifeng Pan (+8) | 2025-07-10 | 無（NeurIPS'25 見 L3） | 部分 v1 |
| 2510.18586 | TokenCake | Zhuohang Bian (+4) | 2025-10-21 | Accepted at EuroSys '27 | 部分 v5 |
| 2511.02230 | Continuum | Hanchen Li (+10) | 2025-11-04 | 無（ICLR'26 workshop 見 L3） | 部分 v7 |
| 2402.05099 | Hydragen | Jordan Juravsky (+5) | 2024-02-07 | 無 | 部分 v2 |
| 2501.01005 | FlashInfer | Zihao Ye (+10) | 2025-01-02 | Accepted by MLSys 2025 | 部分 v2 |
| 2402.14808 | RelayAttention | Lei Zhu (+3) | 2024-02-22 | ACL 2024 main | 部分 v3 |
| 2402.15220 | ChunkAttention | Lu Ye (+3) | 2024-02-23 | ACL 2024 | 部分 v4 |
| 2504.03648 | AIBrix | The AIBrix Team (+26) | 2025-02-22 | 無 | 部分 v1 |
| 2608.01657 | PrefixShield（Admission Responsibility） | Zhiyu Wang (+1) | 2026-08-03 | 無 | 部分 v1 |
| 2608.02244 | Resource-Fair Scheduling | Dayi Yao (+1) | 2026-08-03 | 無 | 摘要 |
| 2411.15997 | FairServe | Redwan Ibne Seraj Khan (+14) | 2024-11-24 | 無 | 摘要 |
| 2608.25523 | TOPAS | Hongqiu Ni (+4) | 2026-08-26 | 8 pages | 摘要 |
| 2609.18112 | Token Latency Fairness | Dev Bali (+5) | 2026-09-16 | 無 | 摘要（L3 已列） |
| 2608.14561 | Biophysically-Inspired Cache Fairness | Matt R. Flax | 2026-05-26 | 無 | 摘要（L3 已列） |
| 2406.10774 | Quest | Jiaming Tang (+5) | 2024-06-16 | ICML 2024 | 摘要＋grep |
| 2410.21465 | ShadowKV | Hanshi Sun (+8) | 2024-10-28 | 無 | 摘要＋grep |
| 2406.19707 | InfiniGen | Wonbeom Lee (+3) | 2024-06-28 | OSDI 2024 | 摘要＋grep |
| 2407.12820 | PQCache | Hailin Zhang (+7) | 2024-07-01 | 無 | 部分 v2（p22） |
| 2505.02922 | RetroInfer | Yaoqi Chen (+18) | 2025-05-05 | VLDB 2026；PVLDB 19(5) | 部分（p4） |
| 2410.16179 | MagicPIG | Zhuoming Chen (+10) | 2024-10-21 | 無 | 摘要＋grep |
| 2610.11245 | Read What Matters（ReadKV） | Siddharth Bhandari (+3) | 2026-10-08 | 無 | 部分 v1 |
| 2609.17652 | Fathom | Vivek Kalyanarangan | 2026-09-15 | 無 | 部分 v3 |
| 2608.07009 | HiSparse | Zhiqiang Xie (+5) | 2026-08-07 | 無 | 部分 v1 |
| 2608.08097 | OasisKV | Can Xiao (+9) | 2026-08-08 | 無 | 部分 v1 |
| 2605.19893 | SSV（v1 HTML 名 SpecSA） | Zhibin Wang (+5) | 2026-05-19 | 無 | 部分 v2 |
| 2412.10319 | SCBench | Yucheng Li (+10) | 2024-12-13 | Accepted at ICLR 2025 | 部分 v2 |
| 2602.07721 | ParisKV | Yanlin Qi (+5) | 2026-02-07 | ICML 2026 | 摘要 |
| 2605.18071 | KVDrive | Jian Lin (+7) | 2026-05-18 | 無 | 摘要 |

**非 arXiv 來源**：

- **ArkVale**：R. Chen *et al.*, NeurIPS 2024，proceedings.neurips.cc PDF〔部分 grep〕
- **FlashInfer cascade 部落格**：flashinfer.ai/2024/02/02/cascade-inference.html〔文件〕
- **Triton batcher**：docs.nvidia.com 的 Triton 2.67 存檔頁〔文件〕
- **SOLA**（K. Hong *et al.*, MLSys 2025）：PEEK 說它「在 prefix 分群與逐請求 SLO 間仲裁」，本次只看到 MLSys 頁摘要，**沒看到 prefix 分群的描述 → NOT_VERIFIED**
- **GitHub**：
  - sgl-project/sglang：PR #32911、#31723、#31710；issue #31954
  - vllm-project/vllm：PR #44295、#61126；issue #49829、#47763
  - lightseekorg/tokenspeed：PR #2373
  - llm-d/llm-d-kv-cache、ai-dynamo/dynamo、vllm-project/aibrix 的原始碼與文件
  - 版本見檔頭
