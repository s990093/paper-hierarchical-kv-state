# 長 context KV Cache 研究的實驗基礎：大家怎麼做實驗、為什麼、我該怎麼做

> **一句話**：這個領域有四套各自為政的評測傳統：serving 系統、KV 壓縮與稀疏、跨請求重用與分層、快取策略。它們量的東西、用的資料、對手、長度都不同，結論幾乎不能互相搬用。PoC 要在同一套 harness 上同時放進四套的代表，並補上它們共同缺的五件事：16K–512K 的長度、寫入時的決策、真實的到達時間、重複與信賴區間、模擬對真機的校準。

| 項目 | 內容 |
|:--|:--|
| 日期 | 2026-10-07 |
| 證據 | [research_20261006_eval/cards/](research_20261006_eval/cards/) 的 11 份評測卡：54 篇論文、15 個品質 benchmark、12 個負載資料集、6 個 benchmark 工具、5 個 KV 層系統、8 份量測方法學文獻 |
| 查證方式 | 每篇讀原文全文或原始碼，每格附頁碼、章節或行號。之後由另一個沒參與抽取的 agent 回原文逐格複核：錯的直接改，找不到的改標〔未查證〕。統計見附錄 A |
| 標記 | 「〔E03〕Cake p5」＝E03 卡片裡 Cake 那張、原文第 5 頁。〔判讀〕＝我們的推論，不是原文的話。〔未查證〕＝查不到 |
| 不含 | 本專案的任何實測數字。本文只整理文獻與工具，實驗還沒有照本文的協定重跑 |

---

## 目錄

- [0. 這份文件回答什麼](#0-這份文件回答什麼)
- [1. 地圖：四套評測傳統](#1-地圖四套評測傳統)
- [2. 生態系：引擎、KV 層、量測工具](#2-生態系引擎kv-層量測工具)
- [3. 資料與負載](#3-資料與負載)
- [4. 指標](#4-指標)
- [5. 實驗參數總表](#5-實驗參數總表)
- [6. 對手怎麼選](#6-對手怎麼選)
- [7. 評測章節的結構與統計](#7-評測章節的結構與統計)
- [8. 評測陷阱總表](#8-評測陷阱總表)
- [9. 重現：怎麼確認設定與原論文一致](#9-重現怎麼確認設定與原論文一致)
- [10. 如果是我：PoC 怎麼設計](#10-如果是我poc-怎麼設計)
- [11. 接下來四週怎麼學](#11-接下來四週怎麼學)
- [12. 寄給老師的兩份 PDF：勘誤](#12-寄給老師的兩份-pdf勘誤)
- [附錄 A：評測卡索引與複核統計](#附錄-a評測卡索引與複核統計)
- [附錄 B：原論文本身的問題](#附錄-b原論文本身的問題)
- [附錄 C：參考來源](#附錄-c參考來源)

---

## 0. 這份文件回答什麼

| 你的問題 | 在哪一節 |
|:--|:--|
| 這個領域有哪些做法，彼此差在哪 | §1 |
| LMCache、vLLM、SGLang 這些元件是什麼，怎麼接在一起 | §2 |
| ShareGPT 等資料集是什麼、怎麼被用、為什麼不夠 | §3 |
| 指標怎麼定義，工具之間差在哪 | §4 |
| 實驗參數有哪些，典型值，為什麼這樣設 | §5 |
| 對手怎麼選 | §6 |
| systems 論文的評測章節怎麼組織，統計怎麼做 | §7 |
| 哪些做法會讓結果不可信 | §8 |
| 重現時怎麼確認自己的設定與原論文一致 | §9 |
| 如果是我，PoC 要怎麼設計才不會只對單一方法 | §10 |
| 接下來四週怎麼學 | §11 |
| 寄給老師的兩份 PDF 要改哪裡 | §12 |

§3–§6 的寫法相同：**大家怎麼設 → 為什麼這樣設 → 陷阱 → 我們的設定**。細節（逐篇的模型、硬體、版本、資料、頁碼）都在卡片裡，本文只放結論和比較。

---

## 1. 地圖：四套評測傳統

### 1.1 總表

| 面向 | A. Serving 系統 | B. KV 壓縮與稀疏（請求內） | C. 跨請求重用與分層 | D. 快取策略 |
|:--|:--|:--|:--|:--|
| 代表論文 | Orca、vLLM、SGLang、Sarathi-Serve、DistServe、Splitwise〔E01〕 | H2O、StreamingLLM、SnapKV、PyramidKV、KIVI、KVQuant、KVTuner〔E07〕；Quest、ShadowKV、InfiniGen、MInference、DuoAttention〔E08〕；學習式的 KVP、ForesightKV、LookaheadKV、TRIM-KV〔E06〕 | Cake、CacheFlow、Pensieve、HCache、KVPR、Bottlenecks、py-kvcache〔E03〕；Strata、Bidaw、MTDS、LMCache、Mooncake、CachedAttention、Tutti、KVDrive〔E04〕；CacheGen、CacheBlend〔E07〕 | LRB、HALP（CDN）；Fancy-eviction、AsymCache、Marconi、AdaptCache、EvicPress、KVCache in the wild〔E05〕；LARU、LPC、SAECache〔E06〕 |
| 主張的形式 | 在某個延遲條件下，系統可撐的負載是對手的幾倍〔E01 共同模式 2〕 | 在某個記憶體預算下，品質接近 full KV〔E07 P3〕 | 還原（載入或重算）比對手快，或同 TTFT 下吞吐更高〔E03、E04〕 | 命中率或成本加權命中率比對手高、離 Belady 更近〔E05〕 |
| 主負載 | 資料集的**長度分布**＋合成內容（ShareGPT、Alpaca、arXiv summarization 等）〔E01 共同模式 1〕 | 單請求的品質 benchmark（LongBench、RULER、NIAH、PG-19）〔E07、E08〕 | 長文件被重複查詢（LooGLE、NarrativeQA、LEval）、多輪對話、少數真實 trace〔E04 共同模式 4〕 | 生產 trace 重播（CDN trace、Mooncake、Bailian 等）〔E05〕 |
| 到達過程 | Poisson；七篇的主結果都不是用真實到達時間重播出來的〔E01〕 | 沒有（單請求或固定 batch）〔E07 P4、E08 共同模式 1〕 | Poisson 為主；主實驗用真實時間戳的只有 Mooncake、Bidaw〔E04 共同模式 4〕 | trace 原生順序，在模擬器上跑〔E05 共同模式 1〕 |
| 長度 | 上限 2K–16K，其中三篇被 2048 卡死〔E01 共同模式 5〕 | 有 full KV 基線的多 ≤32K；更長的多半只有 NIAH／passkey 類〔E07 P3〕；學習式與稀疏派用 RULER 評到 128K〔E06、E08〕 | 多數 ≤16K；Tutti 3K–200K、KVDrive 60K–360K（沒有跨請求重用）〔E03、E04〕 | 由 trace 決定，多數偏短〔E05、E06〕 |
| 主指標 | TTFT、TBT、normalized latency 對請求率；goodput、capacity〔E01〕 | 準確度對預算的曲線〔E07〕 | TTFT、同 TTFT 下的吞吐、命中率〔E03、E04〕 | 命中率（object／byte／token）對容量的曲線〔E05〕 |
| 對手 | 多半自己重做（Orca 三種變體、FasterTransformer）〔E01〕 | Full KV＋同族方法；對手常被改寫〔E08 共同模式 5〕 | 只算＝vLLM chunked prefill，只載＝LMCache；多半自己重做〔E03 共同模式 6、E04 共同模式 6〕 | LRB 比 14 種演算法＋Belady；Fancy 比 14 種＋Belady、BeladyCompute、ILP；其餘 LLM 論文多半只比 3–6 種、沒有上界〔E05 共同模式 4〕 |
| 品質 | 不量（無損） | 主軸 | 只有有損方法才量（CacheGen、CacheBlend、EvicPress）〔E07、E05〕 | 不量 |
| 重複與誤差 | 七篇都沒有誤差棒或信賴區間〔E01 共同模式 6〕 | 幾乎沒有〔E08 共同模式 6〕 | E04 的 8 篇與 Cake、CacheFlow、Pensieve、HCache 都沒寫；只有 KVPR（5 次平均）、py-kvcache（3 次平均）、Bottlenecks（mean±std）有報〔E03 共同模式 5、E04 共同模式 6〕 | HALP 有生產 A/A（實驗／no-op／對照三組）；LLM 這組都沒有〔E05 共同模式 5〕 |

### 1.2 為什麼會分成四套

以下依各卡「設計理由（原文）」一格歸納，歸納本身是〔判讀〕。

* **A 派研究的是排程與記憶體管理**，內容不影響時間。Splitwise 明說文字不影響效能指標〔E01 Splitwise〕，但這個前提只在沒有快取時成立。資料集沒有時間戳，所以只能補 Poisson（vLLM p10、DistServe p9）〔E01〕。
* **B 派的 KV 生命期就是一個請求**，到達過程不改變準確度，所以不需要多請求；系統收益靠「省記憶體 → 更大 batch → 更高吞吐」間接論證〔E07 P4〕。
* **C 派的收益取決於重用距離與頻寬**，所以一定要有重用結構。但長 context 的真實 trace 很少，只好用「少量長文件被多次查詢」來構造〔E04 共同模式 4〕。
* **D 派要在大量容量與 trace 上比策略**，所以用模擬器。CDN 傳統是「模擬器跑廣度、原型或生產環境量開銷與端到端」，LLM 這邊大多只做其中一半〔E05 共同模式 1〕。

### 1.3 為什麼不能只針對單一方法做實驗

1. **結論綁在工作點上。**
   * Fancy-eviction 的 Belady 空間只出現在 HBM 大小的容量。到 1 TiB 時，LRU 與 Belady 幾乎重合，讀圖約只差 1–2 點（Fig. 2 p5、§5 p9）〔E05〕。
   * Cake 只量單請求的 TTFT，沒有掃負載〔E11 §1.2〕。
   * 文獻中最大的倍數，多出現在 baseline 已經飽和的工作點〔E04 共同模式 5；workloads_eval §0 第 10 條〕。
2. **對手的版本與路徑本身就是變因。** Cake 用 vLLM v0.6.2＋LMCache v0.1.4，CacheFlow 用 vLLM v0.29.0＋LMCache v0.5.5rc5〔E03 共同模式 6〕。py-kvcache 量到，同一份負載下 LMCache 與 vLLM Offload 的 TTFT 差 1.61×（206 vs 128 ms，ShareGPT，p10）〔E03〕。
3. **同名的指標不同義。** vLLM 的 TPOT 等於 AIPerf 與 GuideLLM 的 ITL；GuideLLM 的 TPOT 把 TTFT 也算進去〔E02 A5〕。normalized latency 在 Orca 取中位數，在 vLLM 取平均〔E01 指標字典〕。
4. **品質數字取決於協定。**
   * SnapKV 自己的 Fig. 4 顯示，同一份文件換一個問題，選出的重要位置就不一樣（p5–6）〔E07 P2〕。
   * KIVI 與 KVQuant 在 prefill 時用的是精確 KV，量化誤差只影響 decode〔E07 P5〕。所以它們會低估「下一輪 prefill 讀到低精度 KV」的品質損失。
5. **對手幾乎不重疊。** 496 個論文配對中，72.6% 的 baseline 集合完全不交集〔workloads_eval §0 第 1 條〕。

所以 PoC 必須在**同一個引擎、同一批 prompt、同一條 pipeline** 上〔E08 T22〕，同時放進每一套傳統至少一個代表（§6、§10）。

---

## 2. 生態系：引擎、KV 層、量測工具

### 2.1 三者的關係

```
量測工具（client）           引擎（server）                      KV 層（GPU 以外的儲存）
vllm bench serve   ─HTTP─▶  vLLM 排程器 ─ KV connector API ─▶  LMCache / OffloadingConnector / Mooncake
SGLang bench_serving ─────▶ SGLang radix cache ─ HiCache ───▶  L2 主機記憶體 / L3 後端
AIPerf、GuideLLM、MLPerf LoadGen：只是另一種 client，算指標的方式不同（§4.2）
```

* **引擎**負責排程、計算與 GPU 上的 KV。
* **KV 層**負責把 KV 存到 GPU 以外的地方，之後再拿回來。
* **量測工具**負責送請求、記時間、算指標。
* 三者各有預設值；任何一個沒寫進 manifest，結果就不可比（§9.1）。

### 2.2 LMCache 是什麼〔E02 B1〕

| 面向 | 內容 |
|:--|:--|
| 層級 | GPU → CPU DRAM（pinned）→ 本地磁碟或 NVMe → 遠端（Redis、Mooncake 等） |
| 接法 | vLLM 內建 `LMCacheConnectorV1`（in-process）與 `LMCacheMPConnector`（另起 LMCache server）。in-process 模式的文件已標為 deprecated |
| 粒度 | chunk，預設 256 token，以 token 雜湊索引 |
| 寫入 | prefill 每一步之後寫；**只存完整 chunk**（未滿 256 的尾段預設丟掉）；**decode 產生的 KV 預設不存**；寫到所有啟用的層（寫穿） |
| 逐出 | in-process 預設 LRU（可選 LFU、FIFO、MRU）；MP 模式必須指定 |
| 非前綴重用 | 有，即 CacheBlend |
| 官方 benchmark | `long_doc_qa`（長文件、重複查詢）、`multi_round_qa`（多使用者多輪）等。**`multi_round_qa` 的每位使用者以固定間隔發問，不是 Poisson**；prompt 由重複的 "hi" 組成 |

〔判讀〕LMCache 的三個預設（不存 decode KV、丟掉未滿的尾段、寫到所有層）會直接影響「寫了什麼」。用它當 baseline 時，三個設定都要寫明。

### 2.3 vLLM 的 KV connector 與 `OffloadingConnector`〔E02 B2〕

* **介面語意**：排程端的 `get_num_new_matched_tokens` 回傳「已計算部分之後，還能從外部載入的**連續前綴**」，剩下的尾段由排程器算。
* **對你的下一步最重要的一點**〔E02 PoC 建議 7，判讀〕：Cake 的順序剛好相反，從前段重算、從後段載入，所以在現行 API 下無法直接表達。現成能用的只有「逐層載入與計算重疊」。要做 Cake 式還原，大概需要改排程器，或自訂 connector 加 model runner。**這是 PoC 架構要先決定的事**（§9.3、§10.2）。
* **`OffloadingConnector` 的預設**：`store_threshold=0`，也就是全部都存；設成 ≥2 時，一個 chunk 要被計數達 N 次才存。逐出可選 `lru`（預設）或 `arc`；另有 `TieringOffloadingSpec`，可以接 fs、obj 等次層。
* **版本差異**：`store_threshold` 的計數點，v0.28.0 是在 `lookup()` 時記被查詢的次數，main 是在 `prepare_store` 時記被提議儲存的次數。所以兩版的同一個 N 不等價。
* **清快取的陷阱**：
  * 不帶參數的 `POST /reset_prefix_cache` **只清 GPU prefix cache**，要加 `?reset_external=true` 才會清 connector。
  * `OffloadingConnector` 有實作清空。
  * LMCache 的 connector 沒有實作，API 卻仍回 `success: true`。
  * 所以「冷快取」的量測可能根本不冷。這正是專案 CLAUDE.md 規則 7「查不到不等於沒有」的情形。

### 2.4 其他 KV 層系統〔E02 B3–B4〕

| 系統 | 層級 | 粒度 | 寫入（預設） | 逐出（預設） | 備註 |
|:--|:--|:--|:--|:--|:--|
| LMCache | GPU→CPU→本地碟→遠端 | chunk 256 token | 寫穿到所有啟用層；decode 不存 | LRU | 有 CacheBlend |
| vLLM OffloadingConnector | GPU→CPU（Tiering 可再往下） | GPU block 或指定 chunk | 全存（`store_threshold=0`） | LRU（可 ARC） | — |
| SGLang HiCache | L1 GPU→L2 主機→L3 後端 | page | `write_through`（另有 selective：命中 2 次才寫；write_back） | radix LRU（可 LFU、SLRU 等） | L2 每個 instance 私有 |
| Mooncake Store | 叢集 DRAM 池→本地 SSD | 物件 | 寫入記憶體；記憶體逐出時才寫 SSD | 近似 LRU，用量 90% 時逐出 5% | 有 soft pin |
| Dynamo KVBM | GPU→主機→磁碟→物件儲存 | block | 依 policy；磁碟層預設只寫頻率 ≥2 的 block | 〔未查證〕 | **v1.5.0 起棄用**，目標 v1.6.0 移除 |

**共同點**〔E02 重點摘要 7〕：
* 機制上的「不存」開關是有的，例如 LMCache 每請求的 `skip_save`、`store_threshold≥2`、HiCache 的 `write_back`。
* 但查到的系統裡，**沒有一個依「重算成本 vs 傳輸成本」決定不存**。這就是本研究「寫入時決定狀態」的空間〔判讀〕。

### 2.5 量測工具的四個預設值陷阱〔E02 重點摘要〕

| 陷阱 | 內容 | 後果 |
|:--|:--|:--|
| `--random-range-ratio` 預設都是 0，意思卻相反 | vLLM：長度取 [L(1−r), L(1+r)]，r=0 就是固定 L。SGLang：取 [max(L·r,1), L]，r=0 就是 1 到 L 的均勻分布 | SGLang 預設下的平均長度只有約 L/2 |
| Mooncake trace 的 block 是 512 token | vLLM `timed_trace` 預設 16，hash 不夠時不補長度；SGLang 的 mooncake 模式不用 `input_length` | 用 vLLM 預設重播，prompt 會縮成約 1/32，**而且不會報錯**。和本專案 2026-08-31 踩過的錯一樣 |
| 暖機會重送第一個請求 | vLLM `--num-warmups` 與 SGLang 的暖機都重送第一個請求的 prompt；SGLang 預設不清快取 | 量 KV 重用時，第一筆量測請求會意外命中 |
| 到達過程的實作不同 | vLLM 的 gamma 到達會把總時長正規化成剛好 N/rate；SGLang 只有 Poisson；GuideLLM 的 sweep 內插預設是 constant | 同樣寫「Poisson、rate=r」，實際送法不同 |

另外，vLLM 在 PR #55508（2026-09-13）才修正計時。本專案用的 v0.28.0 不含這個修正：
* completions 端點的 TTFT 殘差約 −0.35 µs；
* chat 端點的 E2EL 會延到最後的 usage chunk，殘差約 +24 µs；usage chunk 若延遲 30 ms，殘差會變成約 +31,473 µs。
* 所以**一律用 completions 端點**〔E02 A1.5、PoC 建議 1〕。

---

## 3. 資料與負載

### 3.1 四類資料，各能回答什麼

| 類別 | 例子 | 能回答 | 不能回答 |
|:--|:--|:--|:--|
| 只有長度，內容合成 | Cake 的合成 token（p5）、vLLM `random`、LMCache 的 "hi" prompt | 無損路徑的 TTFT、頻寬、重算成本〔E09 原則查證〕 | 品質；資料相依方法的時間（例如 MInference 的稀疏 prefill，SCBench D.1 p22）〔E09〕 |
| 有內容、沒有時間戳的對話 | ShareGPT、LMSYS-Chat-1M | 長度分布、多輪結構 | 到達時間、重用間隔 |
| 有時間戳的 trace | Mooncake、Alibaba Bailian、Azure、BurstGPT、TraceLab | 到達過程、重用距離、時間相依的策略 | 品質（多半沒有內容，或只有 hash） |
| 品質 benchmark | LongBench、RULER、SCBench、∞Bench、HELMET | 有損動作的 ε | 時間結構（沒有到達過程） |

〔判讀〕本研究需要的是「真實 trace 的時間與 session 結構」配「長 context 的內容」：前者決定命中與時間，後者決定品質。沒有一份公開資料同時具備兩者，所以要拼接，並寫明哪一段是真實、哪一段是合成（§10.4）。

### 3.2 ShareGPT〔E10 C1〕

**它是什麼**：sharegpt.com 爬下的 ChatGPT 對話，原本是 Vicuna 訓練用的資料，後來被 serving 社群當成「真實聊天的長度分布」。最常用的檔是 `ShareGPT_V3_unfiltered_cleaned_split.json`，共 94,145 筆。

**為什麼大家用它**：
* 原文給的理由：
  * vLLM：它有真實 LLM 服務的輸入輸出文字，比 Alpaca 長、變異大（p10）。
  * Strata：拿來測短 context，因為先前的分層 KV 研究用過（p10）。
  * CachedAttention：73% 的對話是多輪（p2）。
* 〔判讀〕vLLM 論文與 benchmark 工具把它變成預設，下載只要一行，審稿人也熟悉。

**它的五個問題**（數字都由複核者用自己的腳本重算過）：

| 問題 | 內容 |
|:--|:--|
| 被切段 | 長對話依 FastChat 流程切成約 2K token 的段。94,145 筆中，43,459 筆是續段；36.7% 的條目第一則是 `gpt`。切段腳本只依累積 token 數切，沒有強制從 human 開始 |
| 太短 | 每筆所有則串起來（Qwen2.5 tokenizer）：中位 1,655、p99 2,873；≥16K 只占 0.018% |
| vLLM 工具讀成單輪 | `ShareGPTDataset` 只取每筆的前兩則當 prompt 與 completion，**不檢查說話者**；再丟掉 prompt <4、output <4、prompt >1024、總長 >2048 的樣本。通過過濾的 81,267 筆中，35.3% 是「拿 GPT 的回答當 prompt、拿使用者的話當輸出」（OPT tokenizer） |
| 沒有時間戳與使用者身分 | vLLM、DistServe、Bidaw 的原文都說沒有時間戳 |
| 版本混亂 | 至少有 V3 split、`_no_imsorry`、ShareGPT4／openchat、原始 90k 等版本。各篇報的統計因此對不上：Bidaw 平均 5.7 輪、Pensieve 48,159 段 5.56 輪、CachedAttention 9K 段 5.75 輪。vLLM 論文報的平均長度（輸入 161.31、輸出 337.99，Fig. 11）用三種組法都重現不出來 |

**各篇怎麼用它**〔E10 C1 表〕：

| 論文 | 用法 | 到達 |
|:--|:--|:--|
| vLLM | 只取輸入輸出長度來合成請求；chatbot 實驗把 prompt 截到最後 1024 token，**輪間不保留 KV** | Poisson |
| Sarathi-Serve | `openchat_sharegpt4`；每一輪當成一個請求；去掉總長 >8,192 | Poisson |
| CachedAttention | 9K 個 session、約 52K 輪；前 10K 輪暖機 | session 到達為 Poisson（λ=1.0/s） |
| Pensieve | 48,159 段；最大 context 16,384 | Poisson；下一輪等上一輪回應完成，再加指數分布的 think time（平均 60 s） |
| Strata | 短 context 的對照組；保留輪間依賴 | Poisson＋60 s thinking time |
| HCache | ShareGPT4；context 上限 16K | session 到達為 Poisson，輪間固定 30 s |
| SAECache | 74% 的請求屬於多輪 session | 固定注入間隔 0.02–0.08 s |

**結論**〔E10 判讀〕：ShareGPT 對 16K–512K 的研究只能當「短 context、無重用」的負對照，**不能當時間相關的負載**。若審稿人要求跑，要做四件事：
1. 從 human 開始重組 session。
2. 用完整歷史的多輪模式。
3. 時間從真實 trace 移植。
4. 寫明檔名、sha256、tokenizer 與被過濾掉的比例。

### 3.3 其他資料集速查〔E10 C2–C13〕

| 資料集 | 規模與長度 | 時間戳 | 多輪／身分 | 授權 | 對本研究的用途〔判讀〕 |
|:--|:--|:--|:--|:--|:--|
| Alpaca | 52,002 筆；prompt 中位 15 token、output 平均 58 | 無 | 單輪 | CC BY-NC 4.0 | 不用 |
| LMSYS-Chat-1M | 100 萬段；平均 2.0 輪；prompt 平均 69.5 token | **無** | 無 user id | gated，禁止再散布 | 不用；部分叢集疑似腳本批次送出，會製造人為的前綴重用 |
| Chatbot Arena | 33K 段；平均 1.2 輪 | 有 | 匿名 user id | prompt CC-BY-4.0、輸出 CC-BY-NC-4.0 | 不用；同一 prompt 送兩個模型 |
| WildChat-1M／4.8M | 837,989 段／3,199,860 段 | 有，但是 assistant 回應**完成**的時間 | `hashed_ip`＋header 可串使用者 | ODC-BY | 時間來源的候選；到達時間要反推 |
| OpenOrca | 約 4.2M 筆（GPT-4 約 1M、GPT-3.5 約 3.2M）；MLPerf 限 <1024 | 無 | 單輪 | MIT | 不用；`system_prompt` 種類少，會製造共同前綴 |
| BurstGPT | BurstGPT_3：5,344,021 列；request 中位 327、≥32K 只占 0.029% | 有 | BurstGPT_3 有 session | CC-BY-4.0 | 只有長度、沒有內容；可當突發與輪間時間的來源（輪間 p50 131 s） |
| Azure 2023／2024 | 只有長度與時間 | 有 | 無 | CC-BY-4.0 | 可當突發與日週期的來源；DynamoLLM 量到尖峰對谷底 34.6 倍（p4） |
| Mooncake FAST'25 | conversation 12,031 筆（中位 6,909、最大 126,195）；toolagent 23,608 筆；synthetic 3,993 筆 | 有（毫秒） | 無 session id，靠 hash 推 | repo Apache-2.0 | **生產重用的錨點**；hash block 512 token |
| Alibaba Bailian | 四條 2 小時 trace；輸入中位 574–4,540；≥32K 只占 0–0.18% | 有（秒） | `chat_id`、`parent_chat_id` | Apache-2.0 | 短 context 的負對照；hash block 16 token（論文寫 4 token） |
| TraceLab | v0.0.1：357,161 列、43 人、中位 124,018。**v0.0.2（2026-07-24）：665,453 列、52 人、中位 132,092、>262,144 占 14.1%** | 有（毫秒，逐事件） | session、round | 資料 CC BY 4.0 | **長 context＋真實時間的主骨架**；沒有內容與 hash；`prefix_tokens` 是 provider 的實際命中 |
| Bidaw trace | 1,268,346 輪、56,573 人；每人 22.42 輪；輪間 p50 43 s | 有（整數秒） | user_id＋round | **沒有 license 檔** | 與 Bidaw 正面比較時用 |

**長文件當負載**〔E10 C13〕：LongBench、arXiv summarization、L-Eval、LooGLE、NarrativeQA 都是「benchmark 加合成到達」。
* DistServe 用的是 LongBench 的 summarization（截到 2,048），不是 arXiv summarization。
* Sarathi-Serve 濾掉 >16,384。
* Strata 濾掉 >128K 的 NarrativeQA 文件，再抽 50 份。
* 到達幾乎都是 Poisson。HCache 的 L-Eval 主實驗是 batch size 1，沒有到達過程；Zipf 只用在 GPU 重用的子實驗，描述的是「哪一份 context 被請求」的熱度（p10、p13）。

**trace 的單位驗算**（CLAUDE.md 規則 6）：
* 複核者驗算 Mooncake 與 Bailian 共 8 個檔，`ceil(input/B)==len(hash_ids)` 全部 100% 成立：Mooncake 的 B＝512，Bailian 的 B＝16。
* 每個 loader 都要寫一條斷言〔E10 PoC 建議 6〕：
  * Mooncake：`ceil(input/512)==len(hash_ids)`
  * Bailian：`ceil(input/16)==len(hash_ids)`
  * TraceLab：`prefix+new==input`
  * Bidaw：`round_index` 在每位使用者內連續
  * ShareGPT：第一則必須是 human

### 3.4 沒有時間戳時，到達從哪來〔E10 C14〕

| 方式 | 誰用 | 已知問題 |
|:--|:--|:--|
| Poisson | vLLM、DistServe、Sarathi-Serve、Splitwise（連自家的 Azure trace 也只取長度，到達用 Poisson，p9）、CachedAttention、Pensieve、Strata、Tutti、HCache、MTDS、LMCache、MLPerf Server | Bidaw p12：換成 Poisson 後，依前一輪回答長度的逐出不再降低 miss rate。也沒有突發，沒有日週期 |
| Gamma（可調 CV） | Llumnix、vLLM `--burstiness`、BurstGPT 的擬合 | 仍是平穩的 renewal 過程；CV 的選擇通常沒有依據〔判讀〕 |
| 固定間隔 | SAECache（0.02–0.08 s，刻意讓 GPU 飽和）；vLLM `burstiness=inf` | 只能看飽和吞吐，看不到尾端延遲〔判讀〕 |
| 從別的 trace 移植時間 | EvicPress（Azure 的時間戳＋自己生成的內容） | 時間與內容的關聯被切斷；縮放倍數會改變重用距離，必須記錄〔判讀〕 |
| think time／閉迴路 | Pensieve（指數分布，平均 60 s）、Strata（60 s）、HCache（30 s） | think time 的平均直接決定 KV 在快層待多久，等於研究者在決定結果〔判讀〕 |
| 真實時間戳重播 | Mooncake FAST'25、Bidaw、Bailian 官方重播器、vLLM `timed_trace` | 只有少數 trace 有；長度短；1–2 小時的窗看不到長間隔的重用〔判讀〕 |

**真實的輪間時間**：

| 來源 | p50 | 尾端 |
|:--|:--|:--|
| Bidaw trace | 43 s | p99 142 s |
| BurstGPT_3 conversation | 131 s | p90 2,340 s |
| Bailian（SAECache 擬合，複核者重算） | 約 110 s | P99 約 2,207 s |
| CC-Bench agent（SAECache） | 約 8.5 s | P80 約 25 s |
| TraceLab | 8.8 s | p99 1,782 s（workloads_eval，未重算） |

〔判讀〕人類聊天的中位數是 40–130 s，agent 是 8–9 s，差一個數量級。Pensieve 與 Strata 用的 60 s 落在人類聊天的區間。**對 agent 型的長 context 負載，60 s 太長。**

### 3.5 品質 benchmark 怎麼選〔E09〕

**長度單位與截斷先對齊**〔E09 重點摘要 1–2〕：

| benchmark | 長度單位 | 截斷 | 原生超過 128K？ |
|:--|:--|:--|:--|
| LongBench v1／-E | 英文詞數、中文字數 | 從中間截，保留頭尾 | 否（各任務平均 1,235–22,337 詞或字） |
| LongBench v2 | 詞數；Short <32K、Medium 32K–128K、Long >128K **詞** | 從中間截 | Long 組是 |
| RULER | 受測模型的 token，**含模板與生成** | 不截，直接生成剛好的長度 | 任意長度 |
| HELMET | Llama-2 token，K＝1024 | 從尾端截，保留開頭 | README 寫 >128K 尚未支援 |
| ∞Bench | 依模型 | 從中間截 | 平均約 200K token |
| SCBench | 原始 context 很長 | 官方腳本預設截到 131,072 | 是，但預設會截 |
| LooGLE | token | 頭尾拼接 | 否（平均 21K–36K token） |

**用途 → 選哪個**〔E09 選擇準則表〕：

| 要量什麼 | 用哪些 | 理由 |
|:--|:--|:--|
| 有損壓縮或量化後的 ε | RULER 的多鍵、多值、變數追蹤；SCBench 字串檢索；HELMET 的 RAG＋Recall；∞Bench En.QA／En.MC、LongBench v2 | 不可壓縮的精確檢索最敏感；真實任務補足摘要與推理。Llama-3.1-8B 多輪下，KIVI 2-bit 讓 SCBench 字串檢索從 57.1 掉到 12.0，全域類只從 35.1 到 31.0（Table 4 p9） |
| 跨輪重用後的品質 | SCBench 的多輪與多請求模式 | 唯一把「同一 context 多次查詢」設計進題目的 benchmark |
| 壓縮時不知道 query | SCBench 多請求模式 | context 先 prefill，query 後到 |
| 長度外推、有效長度 | RULER、BABILong、NoLiMa、HELMET 8K–128K | 同一組任務掃長度 |
| 不建議單獨使用 | 原始 NIAH、困惑度 | 飽和、haystack 可壓縮、與下游相關低（HELMET p7 量到 Spearman ρ 全部 ≤0.8） |

**SCBench 的四個限制**〔E09〕：
* 多輪歷史放的是**標準答案**，不是模型自己的輸出（p8；維護者在 issue #154 確認）。
* 官方腳本預設截到 131,072 token。
* 不量時間，也沒有到達過程。
* 重用比例由輪數決定，約 1−1/T；論文 931 sessions／4,853 輪，換算約 80.8%〔計算〕。「80%」不是原文寫的數字。

**「只量時間可以用合成 token」何時成立**〔E09 原則查證，複核後〕：
* 就 TTFT 而言成立（Cake p5）。條件只有一個：prefill 的計算量與 KV 搬移量跟內容無關。輸出長度不進入 TTFT；固定位元寬的 FP8／INT4 格式，搬移與反量化的位元組數也與內容無關。
* 延伸到端到端延遲或吞吐時，才要加限定：有損方法會改變回應長度，必須用真實內容並記錄回應長度（Rethinking p5）。
* 資料相依的方法（例如 MInference）連 TTFT 都不能用合成 token。

---

## 4. 指標

### 4.1 系統指標字典（論文的定義）〔E01 指標字典〕

| 指標 | 定義的差異 | 什麼時候用〔判讀〕 |
|:--|:--|:--|
| TTFT | 是否含排隊不一致：Sarathi-Serve、Etalon 明說含排程延遲；DistServe 的文字說「prefill 階段的持續時間」，但公式與圖都含排隊 | 與還原或 prefill 有關的主張一定要報，並**拆成排隊、載入、重算、剩餘 prefill** |
| TPOT | 每個請求、不含第一個 token 的平均（DistServe、Etalon） | 表達單一請求的生成速度；會把 stall 攤平 |
| TBT | Sarathi-Serve：每個 token 一個值、報 P99；Splitwise 寫的是平均 | 有 prefill 或背景載入與 decode 搶 GPU 時用 |
| ITL | **七篇論文都沒用這個詞**，是工具的用語 | 論文裡用到時要附工具名與版本 |
| normalized latency | E2E ÷ 輸出長度；Orca 取中位數、vLLM 取平均；SGLang 的同名指標意思完全不同 | **長 context 不要用**：它由 prefill 主導，還會把排程延遲攤平 |
| goodput | DistServe：在 SLO attainment 目標（例如 90%）下，每張 GPU 能服務的最大請求率 | 一定要附三個參數：SLO 門檻、attainment 目標、是否以每張 GPU 正規化 |
| capacity | Sarathi-Serve：P99 TBT ≤ SLO 且排程延遲中位數 ≤2 s 下的最大 QPS；Splitwise：9 個 SLO 都滿足 | 主張「同品質下能服務更多」時用，並附可持續條件 |
| fluidity-index | Etalon：每個 token 有期限，達成的期限 ÷ 總期限 | 串流互動、decode 期間會插入還原工作時 |

### 4.2 工具的定義〔E02 A5〕

| 概念 | vLLM bench | SGLang | AIPerf | GuideLLM | LLMPerf | MLPerf LoadGen |
|:--|:--|:--|:--|:--|:--|:--|
| 每請求 (E2E−TTFT)/(n−1) | TPOT | TPOT | ITL | ITL | — | TPOT |
| chunk 間隔攤平後的分布 | ITL | ITL | ICL | — | — | — |
| 含 TTFT 的 E2E/n | — | — | — | TPOT（n＝token 數） | ITL（n＝SSE chunk 數） | — |
| 第一個 token 的判定 | 第一個帶 choices 的 chunk（可為空） | 第一個 text 非空的 chunk | 第一個內容非空的回應 | first token iteration | 第一個 content 非空的 delta | SUT 回報 |
| 輸出 token 數 | 伺服器 usage，沒有才重新 tokenize | usage，沒有就**用 max_tokens** | client tokenizer | 伺服器 usage | Llama tokenizer | SUT 回報 |

〔判讀〕跨工具的數字不要放在同一張表，除非先換算。

### 4.3 建議寫進論文 methodology 的命名約定〔E01 PoC 建議〕

* **TTFT**：從請求到達到第一個輸出 token，包含排隊；另外拆出載入、重算與 prefill。
* **TPOT**：每個請求、不含第一個 token 的平均（DistServe 定義）。
* **TBT**：每個 token 一個值，報 P50 與 P99，並註明母體是 token。
* **ITL**：不使用；必須用時附工具名稱與版本。
* **goodput**：寫成「在 {TTFT ≤ a·基準, TPOT ≤ b} 且 attainment ≥ 90% 下的最大請求率」，並註明是否以每張 GPU 正規化。
* **SLO 門檻**：用相對值，例如 Sarathi-Serve 取「無干擾 decode 時間」的 5×／25×〔E01 共同模式 4〕。KV 分層可以取「KV 全在 GPU」的 TTFT 為基準〔判讀〕。

### 4.4 快取指標

**命中率至少有六種粒度**（token 加權前綴、block、請求級、單層或任一層等），同一份 Mooncake trace 在無限容量下依定義可以是 1.1% 到 100%〔workloads_eval §2〕。快取策略派另外分「計數」與「成本加權」兩種〔E05 共同模式 3〕：

| 傳統 | 計數型 | 成本加權型 |
|:--|:--|:--|
| CDN | object miss ratio（對應延遲） | byte miss ratio（對應 WAN 成本）。HALP 指出兩者會衝突（proc. p1156） |
| LLM 前綴快取 | token 或 block 命中率 | Fancy 的 compute-savings ratio：依每個 block 的重算 FLOPs 給命中加權（p10）；Marconi 的 FLOP efficiency；AsymCache 的 f·ΔT，ΔT 隨位置線性增加 |

〔判讀〕在 CDN 裡，物件的「大小」決定成本；在 LLM 前綴快取裡，block 的「深度」決定成本。本研究的成本隨位置變，所以主指標應是成本加權型，並同時報計數型與各層命中分布。

**一個可以直接沿用的診斷**〔E05 共同模式 6〕：同一份 trace 上，hit-optimal 的 Belady 與 BeladyCompute 的差距，就是「成本感知值不值得做」的上限。Fancy 量到 FreeInference 在 24 GiB 時差 12.6 點，其他五條 trace 都 ≤1.1 點（p10）。

### 4.5 品質指標 ε〔E07、E09〕

* **成對比較**：同一批 prompt、greedy decoding，與 BF16 full KV 逐題比較。統計做在「樣本」上，不是重跑〔E07 PoC 建議 1〕。
* **兩個量**：任務分數差（附配對 bootstrap 信賴區間或 McNemar 檢定），以及與 BF16 輸出一致的比例（答案完全相同、第一個分歧 token 的位置）。
* **樣本數**：若 5% 的題目答案翻轉，n=500 的差值 95% CI 約 ±2 pp，n=2,000 約 ±1 pp〔E09 PoC 建議 4，計算〕。RULER 每個長度 13 個任務 × 500 題足夠；LongBench v2（503 題）與 SCBench（922 列）只夠看出幾個 pp 的差。
* **相對門檻**：以 BF16 在同長度的分數為 base，ε＝相對下降（借 NoLiMa 的做法），比 RULER 的絕對門檻更適合跨模型比較〔E09〕。
* **「近乎無損」沒有共同門檻**：KIVI 約 2% 準確度、KVQuant <0.1 PPL、CacheBlend ≤0.02 F1〔E07 P3〕。所以論文要自己定義，並寫清楚。

---

## 5. 實驗參數總表

| 參數 | 大家怎麼設（例） | 為什麼 | 陷阱 | 我們的設定〔判讀〕 |
|:--|:--|:--|:--|:--|
| 模型與注意力類型 | Cake：LongAlpaca-7B／13B（MHA，512／800 KiB per token）、Llama-3.1-8B（GQA，128 KiB）、70B（GQA，320 KiB）〔E03 Cake 規格〕；多數論文用 7–8B | 取最常見的開源模型；MHA 的 KV 大，I/O 壓力明顯 | LongAlpaca-7B（MHA）每 token 512 KiB，Llama-3.1-8B（GQA）128 KiB，差 4 倍，重算對傳輸的比值跟著變。KIVI 原文把 Mistral-7B 寫成 MHA，與其附表不一致〔E07〕 | 原生長 context 的 GQA 模型；每個 run 記錄每 token 的 KV 位元組（由 config 算） |
| context 長度與單位 | A 派 2K–16K；B 派 ≤32K；C 派多數 ≤16K〔§1.1〕 | 受模型視窗、資料集統計、前作設定限制；Cake 引 CacheGen 的資料集統計〔E03 共同模式 3〕 | 單位不一：詞、字、Llama-2 token、模型 token〔E09〕 | 16K–512K；一律用受測模型的 tokenizer 計，並註明 benchmark 原本的單位 |
| 輸出長度 | Orca 讓模型永不輸出 EOS；Splitwise 強制生成指定數量〔E01〕；Cake 只看 TTFT | 讓時間只受系統影響 | 有損方法會改變輸出長度，用固定長度量吞吐不恰當（Rethinking p5）〔E08 T4〕 | 時間實驗：固定輸出或只看 TTFT。有損實驗：greedy、保留 EOS，並報輸出長度分布 |
| 到達與併發 | Poisson（A、C 派多數）；固定間隔（LMCache multi_round_qa、HCache 同 session 30 s）；無（B 派） | 資料集沒有時間戳 | Bidaw p12：時間戳換成 Poisson 後，依時間的逐出就不再降低 miss rate〔E04〕 | 主結果用真實 trace 的時間；Poisson 只當敏感度分析 |
| 工作點 | 倍數常取在 baseline 已飽和的點〔E04 共同模式 5〕 | 凸顯差異 | 只報峰值比值會誇大 | 報整條負載曲線，並標出 baseline 的飽和點 |
| I/O 頻寬 | Cake：7、25、32、56、100 Gbps，以延遲注入模擬〔E03 Cake 規格〕；Bidaw 以「從 host 複製＋注入延遲」模擬 5 GB/s SSD；Mooncake 模擬 24–400 Gbps〔E04 共同模式 1〕 | 真實 I/O 波動大、難重現 | 用真實裝置的兩篇都量到規格頻寬遠高於有效頻寬：Bottlenecks 只用到 PCIe 5.0 單向峰值的 23%（p8）；py-kvcache 的 staging buffer 讓 4.4 GB 的資料搬了約 10 GB（p12–13）〔E03 共同模式 1〕。Gbps 是十進位還是二進位，原文沒寫 | 限速器加上「每次 I/O 的固定成本」；Gbps 的單位寫死在設定檔；在至少三個點對照真實裝置 |
| 快取容量 | CDN：絕對容量、對數尺度（LRB 64 GB–4 TB）；Fancy：24／48／96 GiB 與 1 TiB，附錄密掃 65 點；in-the-wild：HBM 的倍數〔E05 共同模式 2〕 | 依硬體取點 | 只取少數幾個容量，會藏住策略交叉與非單調（Fancy p17） | 對數尺度 15–20 點，同時報三種單位：GiB、HBM 倍數、佔 trace unique KV 的百分比 |
| 壓縮預算與位元數 | 固定 token 數、prompt 的百分比、每層平均 token 數、名目位元數〔E07 P1〕 | 各自方便 | 單位彼此不可比；多數不算 scale、zero point、residual。以 KVQuant 的口徑估，KIVI 的「2-bit」約 3.05–3.17 bit〔E07〕 | 報實測位元組，含 metadata |
| block 與 chunk 大小 | vLLM block 16；Cake 計算 chunk＝token budget 512、I/O chunk 128；LMCache chunk 256；Mooncake trace 的 hash 512 token | 各引擎預設 | trace 的 hash 粒度與引擎的 block 粒度不同時，重播會出錯（§2.5） | 載入 trace 時寫斷言驗算單位（CLAUDE.md 規則 6）；I/O chunk 掃 128／256／512 |
| GPU 忙碌程度 | Cake：Cake 拿到的 token budget ÷ 總 budget，取 12.5／50／87.5／100%〔E03 Cake 規格〕 | 模擬「別人在用 GPU」 | 背景請求的長度、型態沒寫；只算、只載基線是否承受同樣的背景負載也沒寫 | 背景負載的型態寫明，並一致套用在所有對照組 |
| SLO 門檻 | 作者自訂；較好的做法是相對化（Sarathi-Serve 5×／25×）〔E01 共同模式 4〕 | 沒有業界標準 | 門檻一改，結論可能翻轉 | 相對門檻＋DistServe 式的 SLO Scale 掃描 |
| 暖機與量測窗 | 多數沒說〔E05 共同模式 5〕 | — | 工具的暖機會重送第一個請求（§2.5） | 暖機用不重疊的 prompt；排除前 X 分鐘；寫進 manifest |
| 重複次數 | 七篇經典 serving、八篇分層系統都沒報〔E01、E04〕 | 效益多半 ≥2×，遠大於雜訊〔E11〕 | 效益在 5–15% 時，沒有變異量的結果很容易被質疑 | 真機每點 ≥6 次獨立 run（重啟 server），附無母數 CI（§7.2） |
| 軟體版本 | vLLM v0.6.2（Cake）到 v0.29.0（CacheFlow）〔E03〕 | 各自寫作時的版本 | 版本差會改變 baseline 的表現，也會改變工具的指標定義（§2.5） | 鎖版本並寫進 manifest；同版本內比較 |
| 取樣 | LongBench、RULER、SCBench、HELMET：greedy；LongBench v2：temperature 0.1；∞Bench 部分腳本 0.8〔E09〕 | — | 抽樣本身就會造成輸出長度大幅變化〔E08 T5〕 | greedy；偏離官方設定之處逐項列出 |
| chat template | LongBench 的 6 個補全式任務不套；其 `build_chat` 不認得新模型，等於不套〔E09〕 | — | 模板一改，分數就不可比 | `tokenizer.apply_chat_template`，補全式任務照原設計不套 |
| prefix cache 狀態 | Bottlenecks 關閉 prefix caching〔E03〕 | 隔離變因 | `/reset_prefix_cache` 預設不清 CPU 層，LMCache 不清卻回成功（§2.3） | 每輪前確認各層都已清空，並寫進 manifest |

---

## 6. 對手怎麼選

### 6.1 原則

1. **用「上界、實務、最差」三種對手來夾。** vLLM 對 Orca 做了 Oracle、Pow2、Max 三種變體〔E01〕。本研究對應的是 Oracle、簡單 heuristic、全部重算或 LRU〔判讀〕。
2. **上界要在同一個成本模型下算。**
   * Fancy 同時給 Belady、BeladyCompute 與 ILP 最佳解〔E05〕。
   * 學習式七篇沒有一篇在主結果報同一成本模型下的 OPT〔E06 共同模式 5〕，這一格可以佔。
3. **學習式方法要比「學習式＋保底」的對手**，不能只比 LRU。LARU 用同一個預測器跑 FPB、HF 與 λ 縮放〔E06 PoC 建議 4〕。LPC 自承只比 LRU 是限制（p10）。
4. **參數照原作者設定，並核對。** LRB 比 14 種演算法，其中 4 種的參數與原作者核對過（proc. p536–537），而且 metadata 從快取容量扣除〔E05〕。
5. **改寫要透明。**
   * DuoAttention 把 H2O 與 TOVA 改成 FlashAttention prefill 加 decode 時才逐出（p17）。
   * ShadowKV 的部分對照是外推與理論頻寬推算（p8 註 5）〔E08 T12–T13〕。
   * 改寫過的要列出，並做敏感度分析。

### 6.2 四套傳統各取代表

| 類別 | 對手 | 為什麼要有它 |
|:--|:--|:--|
| 最差與現狀 | 全部重算（vLLM chunked prefill）；全部載入（OffloadingConnector 或 LMCache，記版本） | Cake 的兩個基線〔E03〕 |
| 讀取時還原 | Cake 式（token 維度、前算後載）；layer 維度（HCache／CacheFlow 式） | 這是本研究延伸的對象；CacheFlow Fig. 2 顯示 36K 以下 token 維度輸給 layer 維度〔E03 PoC 建議 4〕 |
| 逐出與佈局 | Pensieve 式（從對話開頭逐出）；AsymCache 式 f·ΔT；Fancy 的 PartialNode RandomCompute | 位置感知的逐出已有前作〔E03、E05〕 |
| 經典快取 | LRU、ARC、S3-FIFO、W-TinyLFU；生產規則（OffloadingConnector `store_threshold≥2`、HiCache `write_through_selective`） | OS 基準；現成的「依次數才寫」對照組〔E02 PoC 建議 9、E05 PoC 建議 5〕 |
| 精度 | 全部降精度＋LRU（FP8；INT4 要同時測 KV4 與 K8V4） | 量化可能取代放置。KVTuner 顯示 Qwen2.5-7B 的 4-bit key 會崩（KIVI-HQQ 實作下 PPL 235.03），K8V4 卻幾乎無損（Table 2 p4）〔E07〕 |
| 寫入時放置的前例 | KVDrive（prefill 結束時依重要度決定放哪層）；Strata 的 selective 寫入；py-kvcache 的整段門檻；HCache 的逐層狀態（hidden、KV 或不存） | 查新與對照都要處理〔E03 共同模式 4、E04 PoC 建議 6〕 |
| 上界 | Belady、BeladyCompute、多層成本模型下的離線最佳（M4 Oracle） | 回答「還剩多少空間」；決定 go/no-go |

---

## 7. 評測章節的結構與統計

### 7.1 標準組件〔E11 §1.2〕

逐一拆解 vLLM、Sarathi-Serve、DistServe、CachedAttention、Mooncake、Strata、Cake 的評測章後：

| 組件 | 有的論文 | 回答的問題 |
|:--|:--|:--|
| 端到端（掃負載） | 七篇中六篇；Cake 只有單請求 TTFT | 整體上好多少 |
| 拆解 | DistServe（五段延遲）、Mooncake（五段延遲）、Strata（各優化的歸因）、CachedAttention（命中分 DRAM／disk） | 好處從哪裡來 |
| 消融 | 七篇都有（Cake 只有「兩種退化情形」） | 每個元件是否必要 |
| 敏感度 | 七篇都有（block size、SLO、容量、頻寬、chunk、長度……） | 結論對參數是否穩健 |
| 開銷 | vLLM（kernel 慢 20–26%）、Sarathi（最多約 25%）、Cake §5.9 等 | 付出的代價 |
| 與上界比較 | 只有 vLLM（Orca Oracle）、Mooncake（理論最高命中率）、Strata（Strata-Oracle） | 還剩多少空間 |
| 重複次數或 CI | **七篇都沒有** | — |

〔判讀〕頂級 systems 會議的評測，至少要有「端到端＋消融＋開銷」三件。Cake 是唯一只量單請求 TTFT、沒有掃負載的。ICML 的標準與 OSDI 不同；若要投 systems 場所，必須補「多請求、真實到達、端到端吞吐對延遲的曲線」〔E11〕。

### 7.2 統計

* **大家都沒做，不代表不用做。** 七篇沒有報 CI，是因為效益通常是 2× 以上，遠大於雜訊。本研究 go/no-go 的灰區是 5–15%，沒有變異量的結果很容易被質疑；Kalibera 引 Mytkowicz 指出，領域內效能進步的中位數只有 10%〔E11 §1.2 判讀〕。
* **無母數信賴區間需要的樣本數**（雙側、95%、i.i.d.、連續分布；複核者以精確二項分布重算）〔E11 重點摘要 4〕：

  | 估計對象 | 最少樣本數 |
  |:--|:--|
  | 中位數 | 6 |
  | 第 95 百分位 | 59 |
  | 第 99 百分位 | 299 |

  n=3 時，中位數雙側 CI 的最大覆蓋率只有 0.75。所以「每點至少 3 次＋信賴區間」給不出 95% 的無母數 CI〔E11 對使用者文件的指控〕。
* **MLPerf 的做法**：為了在 99% 信心、0.05% 誤差界內估 P99，建議至少 270,336 個 query，且每次跑至少 600 秒〔E11〕。
* **重複的單位要是「重啟 server」**，不只是同一次啟動裡多送請求（Kalibera）〔E11 檢查表 11〕。
* **品質用配對檢定**（McNemar 或配對 bootstrap），不要用兩組獨立的 CI〔E09 PoC 建議 4〕。
* **方法學底線**（Hoefler & Belli）：
  * 先判斷量測是否具決定性；非決定性的資料要給 CI；不要假設常態。
  * 比率不能只給相對值；要和上界或理想值比。
  * Hoefler 抽樣的 95 篇適用論文中，只有 2 篇給平均值的 CI（p4）〔E11 重點摘要 3〕。

### 7.3 GPU 量測的三個坑〔E11 重點摘要 5〕

1. **非同步執行**：不同步就只量到 launch 時間。
2. **第一次呼叫的初始化**：PyTorch 範例中，bmm 第一次 2775.5 µs、第二次 22.4 µs。
3. **時脈與熱節流**：nvidia-smi 的 HW Thermal Slowdown 會把時脈降一半以上；鎖頻需要 root。

### 7.4 模擬什麼時候被接受〔E11 重點摘要 6–7〕

* **常見模式**：真機做端到端的主結果，模擬只做真機做不到的消融或 what-if，而且**在同一篇論文裡用真機校驗模擬器**。
  * DistServe 的誤差 <2%（p13 Table 2）。
  * Vidur 靜態負載的 P95 誤差最多 3.33%，85% 容量下多數 <5%，7B 在 95% 容量下最高 12.65%。
  * Splitwise 的效能模型以 80:20 切分驗證，MAPE <3%。
* **Cake 是例外**：所有主結果的 I/O 都以延遲模擬，沒有一點對照真實裝置〔E11 §3.2、E03 共同模式 1〕。
* **trace 驅動模擬的根本前提**是 trace 外生：新策略不會改變被重播的 trace（CausalSim 稱為 exogenous trace assumption）。多輪對話的下一輪到達時間取決於上一輪多快回覆，違反這個前提〔E11，例子為判讀〕。Heiser 也點名「拿含同一組假設的模擬器評估自己的模型」沒有價值。

### 7.5 Artifact evaluation〔E11 重點摘要 8〕

* AE 都在論文錄取之後才做，與錄取與否無關，但時間很短。
  * SOSP'26：錄取通知 7/3，artifact 7/13 要交。
  * EuroSys'27：同樣 10 天。
  * ATC'26：只有 4 天。
* OSDI'26 只評 Artifacts Available 一個 badge，並接受 GitHub／GitLab。SOSP、EuroSys、ATC 不接受 GitHub。
* MLSys'26 評 Available、Functional、Reproduced。
* 〔判讀〕所以 artifact 要從現在開始準備：
  * 每個實驗一支腳本，加一支把結果轉成論文圖表的腳本，結果能追溯到 run_id；
  * 模擬層不需要 GPU，評審幾分鐘就能重現。

### 7.6 審稿人可能問的問題

OpenReview 的審稿意見這次讀不到。網頁要人機驗證，公開 API 回 403，我們沒有嘗試繞過〔E11 §5.1〕。下表是依公開的評測批判（Rethinking'25 的 Missing Pieces、Heiser、SIGPLAN checklist）歸納，**不是**引自真實的審稿意見〔E11 §5.3〕：

| 類別 | 可能的問題 |
|:--|:--|
| 對手不夠強 | 有沒有跟 Cake、Strata、LMCache 的最新版比？參數照官方建議嗎？ |
| 引擎不是生產級 | 是否在 vLLM／SGLang 上量，而不是 HF Transformers？ |
| 長度太短 | 跨請求重用的實驗最長多少？ |
| 只量單請求 | 有沒有多請求、真實到達、吞吐對延遲的曲線？ |
| 沒報開銷與退化 | 預測器的 CPU 時間、記憶體？短 context 是否變差？ |
| 沒有統計 | 重複幾次？CI？差異是否大於雜訊？ |
| 模擬不可信 | 模擬器誤差多少？trace 是否外生？ |
| 負載不真實 | Poisson 還是真實時間戳？open loop 嗎？ |
| 只測有利區段 | 頻寬、長度、容量是否掃到對自己不利的區段？ |
| 平均方式 | 加速比怎麼平均？有沒有排除點？（Cake 的 Table 3 平均排除了「對任一基線 >10×」的格，p6） |
| 沒有上界 | 離 Oracle 還差多少？ |
| 品質 | 有損動作對品質的影響？ |

---

## 8. 評測陷阱總表

從 E01–E11 合併、去重，依「對本研究的影響」排序。每一條都有原文出處（見對應卡片）。

| # | 陷阱 | 誰指出或示範 | 對策〔判讀〕 |
|:--|:--|:--|:--|
| 1 | 把問題放進被壓縮的內容（query-aware），結果看起來太好 | kvpress README；SCBench；SnapKV Fig. 4；Agrawal & Mayer 的 SnapKV 是在 `query_aware: true` 下跑的，論文沒寫〔E08 T1、E07 P2〕 | 放置在 prefill 結束、問題出現前決定；同一 context 配 2 個以上、位置分散的問題 |
| 2 | 品質在 prefill 用精確 KV 量，低估「讀取方 prefill 會看到低精度 KV」的 ε | KIVI p5、KVQuant p22；KVTuner 則刻意在 prefill 也反量化〔E07 P5〕 | 至少兩輪：先 prefill 並降精度，再送問題 |
| 3 | 只有 NIAH 或困惑度 | HELMET；SCBench（haystack 可壓縮）；KVQuant 的 passkey 與 RULER 結論不同〔E09、E07〕 | 用 RULER 多鍵多值、SCBench 檢索、真實任務 |
| 4 | 在 HF transformers 量吞吐 | Rethinking Missing Piece 1〔E08 T3〕 | 時間只在 vLLM 本體量；HF 只用來量品質 |
| 5 | 固定輸出長度量有損方法的吞吐 | Rethinking Missing Piece 2〔E08 T4〕 | greedy、保留 EOS，報輸出長度分布 |
| 6 | OOM 的樣本被靜默跳過（倖存者偏差） | Agrawal & Mayer 的 artifact：TurboQuant 跳過 21–23／200 筆；配對後壓縮率 4.43→3.76，TTFT 由「比 All KV 快」變成慢約 9%〔E08 T21〕 | OOM 當成結果報；平均只用所有方法都跑完的配對樣本 |
| 7 | 基線與方法走不同 harness、不同生成上限 | Agrawal & Mayer：All KV 用 KIVI 的腳本，SnapKV、CaM 用 kvpress；CaM 的生成上限一律 64〔E08 T22〕 | 所有動作（含 Full-GPU 基線）同一條 pipeline |
| 8 | 壓縮比是名目值，沒算 metadata | KVQuant App M；Quest、InfiniGen、ShadowKV〔E07 P1、E08 T9〕 | 每層報實測位元組 |
| 9 | 長度太短，「長 context」其實 ≤8K | Yuan、InfiniGen、kvpress leaderboard、Rethinking〔E08 T14〕 | 主軸 16K–512K，報每段樣本數 |
| 10 | 只測單請求或固定 batch | 幾乎全部〔E08 T15〕 | 真實 trace 的時間與 session；Poisson 只當敏感度 |
| 11 | I/O 用模擬，卻沒對照真實裝置 | Cake〔E03 共同模式 1〕 | 至少三個點對照，報限速器與真實的差距 |
| 12 | 規格頻寬當有效頻寬 | Bottlenecks（PCIe 5.0 只用到 23%）、py-kvcache（staging 讓搬移量變 2 倍以上）〔E03〕 | 用 fio、nvbandwidth 實測，並記錄 |
| 13 | 用 FLOPs 公式估長 context 的重算成本 | Bottlenecks 的 F_pf＝2N 刻意省略 context 項〔E03〕 | 長 context 的重算成本要實測，必要時抽點 |
| 14 | trace 單位或 hash 粒度搞錯 | vLLM `timed_trace` 預設 16 對上 Mooncake 的 512〔E02〕；本專案 2026-08-31 的事故 | 載入時寫「用 A 欄驗算 B 欄」的斷言 |
| 15 | 工具預設值改變負載 | SGLang 的 range ratio、暖機重送第一個請求〔E02〕 | 長度區間從 log 抄進 manifest；暖機用不重疊的 prompt |
| 16 | 「冷快取」其實不冷 | `/reset_prefix_cache` 預設不清 CPU 層；LMCache connector 不清卻回成功〔E02〕 | 用 `?reset_external=true`，並另外確認各層為空 |
| 17 | 跨論文引用時版本混用 | Mooncake arXiv v4 與 FAST'25 的數字差很多；Strata arXiv 與 OSDI 版不同〔E04 PoC 建議 7〕 | 引用時標明版本 |
| 18 | 只取少數容量點 | Fancy 的密掃顯示會藏住策略交叉與非單調（p17）〔E05〕 | 對數尺度 15–20 點 |
| 19 | 平均值變好、中位數變差 | Fancy 的 19.9% 是平均 TTFT，中位數反而變差（p11）〔E05〕 | 報平均、中位數、P99 |
| 20 | 「壓縮後比全量更好」被當成結論 | Agrawal & Mayer、MInference、ShadowKV〔E08 T20〕 | 沒有誤差棒時，小幅勝過全量視為雜訊；先確認基線與方法走同一條 pipeline |
| 21 | 學習式方法的資料洩漏 | LPC 的程式碼對逐輪樣本隨機切驗證集，同一對話會跨兩邊（只影響 checkpoint 選擇）〔E06〕 | 依時間切，同一 session 只在一邊 |
| 22 | 預測 wall-clock 回來時間，等於把硬體學進模型 | LPC 的到達模型是「上一回應完成時間＋think time」（p7）〔E06 PoC 建議 2(d)、9〕 | 預測 think time 或佇列 token 數，再用量測常數換成時間 |

---

## 9. 重現：怎麼確認設定與原論文一致

### 9.1 通用 checklist（manifest 必填）

合併 E02 的 PoC 建議 2、E11 的自我檢查表、E09 的統一協定、專案 CLAUDE.md §4 的記錄協定。缺一項，結果就不寫進 `results/`。

**模型與引擎**
1. 模型的 HF id 與 revision；注意力類型；每 token 的 KV 位元組；各層的 KV dtype。
2. 引擎與版本（commit）；connector 名稱與完整設定（kv-transfer-config、LMCache 設定檔全文）；block／page／chunk 大小。
3. prefix caching 開關；CUDA graph 是否啟用（比較的各組要一致）。

**負載**
4. 資料集與版本；取樣與過濾規則；長度區間（直接從 log 抄出實際區間）。
5. trace 的 hash 粒度與換算方式，以及驗算斷言的結果。
6. 到達過程：原生時間戳（寫明縮放倍數）、Poisson（寫明 rate 與 burstiness）或閉迴路併發 N。
7. 輸出長度的控制方式（ignore_eos、max_tokens）；取樣參數。
8. 暖機方式與暖機排除規則；請求順序（原始或打亂）。

**快取狀態**
9. 每輪前各層是否清空，以及用什麼方法確認。
10. 各層容量（位元組），以及「工作集 ÷ GPU 容量」的壓力。

**量測**
11. 量測工具與版本；endpoint 類型（completions 或 chat）；percentile 清單；goodput 字串。
12. 硬體與**實測**頻寬；時脈與節流計數器的前後差值；有無外來 PID（gpu_guard）。
13. 重複次數（重啟 server 的次數）與 CI 方法。

**品質**
14. benchmark 的 commit（例如 RULER 必須 ≥ `c3f5e3b4`，含 answer prefix 修正〔E09〕）；截斷策略；chat template；max_new_tokens；偏離官方設定之處。

### 9.2 怎麼判定「重現成功」

* **比趨勢，不比絕對值**〔E03 Cake 規格〕。硬體不同時，比的是：
  * 對只算的加速隨長度上升（Cake Table 5）；
  * 對只載的加速隨頻寬下降（Cake Table 3 逐列）。
* **先校準與硬體無關的量**：Cake 的會合點。用量到的逐 chunk 重算曲線與限速器的 I/O 時間，算出「前段累積重算時間＝後段累積載入時間」的位置，再和實測比。這一步最能檢查實作有沒有寫對〔E03 Cake 規格〕。
* **模擬器要自報誤差**：照 DistServe、Vidur、Splitwise 的格式，用另一批真機 run 驗證同一個指標〔E11〕。
* **原文自己有不一致時**，以表格為準，並把不一致記下來（附錄 B）。例如 Cake 內文寫 H100 在 100%、32 Gbps 是 2.23×，表上是 2.40；2.23 其實是 87.5% 那一格。

### 9.3 Cake 的重現規格（摘要）〔E03 卡 1b，已逐數字複核〕

| 項目 | 內容 |
|:--|:--|
| 軟體 | vLLM v0.6.2（V0 engine，chunked prefill）、LMCache v0.1.4；Cake 是在兩者上約 1,000 行的修改。PyTorch、CUDA、驅動版本沒寫 |
| 模型 | LongAlpaca-7B／13B（MHA）、Llama-3.1-8B／70B（GQA；70B 用 FP8 權重）；權重 FP16，KV BF16 |
| 硬體 | 2×A100 80GB NVLink；1×H100（SXM 或 PCIe 沒寫） |
| chunk | 計算端 chunk＝token budget，預設 512（Table 4 掃 64–2048）；I/O 端每個 chunk 128 token |
| 頻寬點 | 7、25、32、56、100 Gbps（只有 Table 3 用滿 5 點） |
| prompt | 合成 token；長度 4K–16K；token 內容與 tokenizer 沒寫 |
| 程式碼 | 沒有公開 |

**建議的校準順序**：
1. Fig. 4 的逐 chunk 重算曲線。不需要限速器，能驗證「成本隨 index 線性上升」的前提，也同時產出模擬器需要的 f(i)。
2. 限速器自檢。
3. 會合點的預測與實測對照。
4. 一個表格格子的趨勢，例如 Table 4 中 1×A100、7B、chunk 512、32 Gbps、100%、16K 這一格（對只算 2.20×、對只載 1.94×）。

**重現時必須自己決定的事**（原文沒寫；E03 列了 15 條，這裡列最關鍵的 6 條）：
1. 限速放在「儲存→CPU」還是「CPU→GPU」。原文三處說法不同。
2. 延遲公式有沒有每次 I/O 的固定成本、在途 I/O 有幾個。
3. Gbps 的單位。複核者以 800 dpi 數位化 Fig. 4，SSD 線為 62.4 ms，與 256 MiB ÷ 4 GiB/s＝62.5 ms 一致，偏向二進位，但仍是圖讀值。
4. 計算 chunk（512）與 I/O chunk（128）怎麼對齊。App. B.2 與 §5.1 的說法不一致。
5. 「其他使用者」的背景負載是什麼型態，只算與只載基線是否承受同樣的背景負載。
6. vLLM 版本落差。v0.6.2 是 V0 engine；現行 connector API 的語意是「連續前綴」，無法直接表達 Cake 的順序（§2.3）。

### 9.4 LMCache 官方 benchmark 的重現要點〔E02 B1〕

* `long_doc_qa`：文件長度 20,000（由 "hi" 字數構成）、文件數 8、輸出 100、每份重複 2 次。先跑暖機輪（每份送一次，都不命中），再跑 query 輪，比較兩輪的 TTFT。
* `multi_round_qa`：README 範例為 10 位使用者 × 5 輪、QPS 0.5、系統提示 1,000、歷史 2,000、回答 100。每位使用者以固定間隔 `num_users/qps` 發問；前一個請求未完成時不送（閉迴路加最小間隔）。
* 兩者的 prompt 內容都不是自然語言，所以只能量時間，不能量品質（§3.1）。

---

## 10. 如果是我：PoC 怎麼設計

本節全部是〔判讀〕，依據是前面各節與卡片。它延續 `RESEARCH_INTRO_20261006.md` §10–§14 的實驗順序（H1、E1–E4），補上這次查證後要改的地方（§10.7）。（RESEARCH_INTRO 與 NEXT_STEPS 目前在 worktree 分支 `claude/llm-long-context-inference-27746c` 的 `docs/`，尚未合併。）

### 10.1 三道關卡，先過關再往下

| 關卡 | 問題 | 怎麼做 | 不過關時 |
|:--|:--|:--|:--|
| G0 成本感知有沒有空間 | 在我們的 trace 與容量上，Belady 與 BeladyCompute 差多少 | trace 驅動模擬；容量取對數尺度 15–20 點；trace 用 Mooncake、Bailian、TraceLab〔E05 PoC 建議 4〕 | 若在相關容量下差距都很小（Fancy 的五條 trace ≤1.1 點那種情況），位置感知的放置不值得做，回頭檢討命題 |
| G1 牆在哪裡（H1） | 16K–512K 時，只算、只載、Cake 式各多快 | 先完成 §9.3 的四步校準，再掃長度 | 照 intro 表 19：256K 以上 Cake 式仍快 ≥1.5×，就重新檢討命題 |
| G2 精度層的 ε | FP8、INT4（KV4 與 K8V4）在各位置帶、各長度的品質損失 | §10.5 的 ε 矩陣 | 若 FP8 在所有位置都近乎無損，精度維度就退化成「全部 FP8」，放置要和它比 |

### 10.2 先決定的架構問題

vLLM 現行的 connector API 只表達「連續前綴」，Cake 的「前算後載」不能直接接上（§2.3）。動手寫程式之前，要先在三條路之間決定：

1. 改 vLLM 排程器，讓它接受「中間有洞」的命中。
2. 自訂 connector 加 model runner。
3. 先用模擬器做 E1，真機只量 H1 的成本曲線。

〔判讀〕第 3 條最快。它和 intro 表 19「先模擬、後真機」的順序一致，而且能先用 G0 判斷值不值得改引擎。

### 10.3 實驗矩陣

| 實驗 | 研究問題 | 自變數 | 固定 | 對手（§6.2） | 主要指標 |
|:--|:--|:--|:--|:--|:--|
| E0 校準 | 我們的平台與文獻對得上嗎 | — | — | Cake Fig. 4 曲線；限速器對照真實裝置；`vllm bench serve` 的基本曲線 | 誤差（與原文的趨勢、與真實裝置的差） |
| H1 牆 | 長度增加時，各還原方式多快 | 長度 16K–512K；頻寬；GPU 忙碌程度 | 模型、chunk、版本 | 只算、只載、Cake 式、layer 維度 | TTFT（拆成排隊、載入、重算、prefill）、會合點的位置、GPU 與 I/O 使用率 |
| E1 模擬 | 寫入時依位置決定，比「全存＋Cake」好多少 | 容量（對數 15–20 點）、trace、長度 | 量測常數 | P0–P4（intro 表 20）、經典快取、AsymCache 式、Pensieve 式、全部降精度＋LRU、Oracle | 成本加權節省率、各層命中分布、TTFT（以常數換算）、相對 Oracle 的差距 |
| Q 品質 | 精度層的 ε | 精度（BF16／FP8／KV4／K8V4）、位置帶、長度、被重用的輪數 | 模型、benchmark 子集 | BF16 full KV | 配對分數差＋CI、輸出一致率、回應長度 |
| E2 寫入干擾 | 寫入會不會拖慢別人的載入 | 寫入頻寬、併發 | — | 不寫入 | 別人的 TTFT 與 TBT |
| E3 多層 vs 單層 | 多一層值不值得 | 層數 | 總容量 | 只有 SSD；CPU＋SSD | 同 E1 |
| E4 穩健性 | 讀取時 GPU 變忙會不會變差 | 讀取時的 GPU 可用比例與頻寬 | 寫入決定 | 全存＋Cake | 同 H1 |

### 10.4 負載

| 用途 | 負載 | 理由 |
|:--|:--|:--|
| 生產重用的錨點 | Mooncake FAST'25 的 conversation 與 toolagent（用全量） | 有毫秒時間戳與前綴 hash；重用率中等；審稿人熟悉〔workloads_eval §6.1 W1〕 |
| 長 context、真實時間 | TraceLab（寫明 v0.0.1 或 v0.0.2 與 sha256） | 唯一公開、真實時間、多使用者、長 context 的 trace〔workloads_eval §6.1 W2、E10 C11〕。v0.0.2 有 14.1% 的請求 >262,144，要事先定好截斷、丟棄或縮放的規則，並記錄比例 |
| 品質 | RULER、SCBench（多輪與多請求）、∞Bench、LongBench v2 | §3.5 |
| 短 context 的負對照 | Alibaba Bailian | 展示「短 context 下位置感知沒有價值」，把可能的負面結果變成有依據的適用範圍 |

* **拼接規則**〔E10 PoC 建議 1〕：
  * 時間與 session 結構來自 trace。
  * 量時間時，依 TraceLab 每輪的 `prefix_tokens`／`newly_append_tokens` 合成 token，重用結構照原始 trace。
  * 量品質時，換成真實長文件：每個 session 對應一份文件，每一輪對應一個 query。
  * 哪一段是合成要寫明。
* **512K 那一端在公開資料中只能靠合成**，論文要明說〔E10 PoC 建議 5〕。
* **agent 型負載的 think time 不要沿用 60 s**（§3.4）。
* **ShareGPT 不當時間相關的負載**（§3.2）。

### 10.5 品質 ε 的協定

1. **query-agnostic**：放置、降精度、DROP 都在 prefill 結束、問題出現之前決定；每段 context 配 2 個以上的問題（SCBench 多請求模式）〔E08 PoC 建議 1〕。
2. **兩輪以上**：第 t 輪的 KV 以低精度存著，第 t+1 輪的 prefill 讀它〔E07 PoC 建議 2〕。
3. **ε 做成矩陣**：ε(位置帶, 精度, 長度, 被重用的輪數)。其他 chunk 保持 BF16，只把某一段降精度；開頭幾個 sink token 一律全精度〔E07 PoC 建議 3〕。
4. **K 與 V 分開**：INT4 同時測 KV4 與 K8V4〔E07 PoC 建議 4〕。
5. **FP8 要自己量**：九篇壓縮論文都沒有 FP8 KV 的實驗〔E07〕。用 vLLM 的 fp8 KV 路徑，並記下 scale 怎麼算。
6. **讀低精度 KV 的成本要量**：沒有人量過「prefill 讀低精度 KV」的成本〔E07 PoC 建議 9〕。GPU-FP8 與 GPU-INT4 的讀取成本要在 vLLM 的 paged 佈局下實測。
7. **benchmark 分四段**〔E09 PoC 建議 1〕：
   * A：RULER 13 任務，16K、32K、64K、128K，以及模型支援的 256K；每格 n≥100（正式版 500），配對比較。
   * B：SCBench 的檢索類與 RepoQA，`--max_seq_length` 設成模型上限，逐輪報分。
   * C：∞Bench En.QA／En.MC、LongBench v2 Medium＋Long。
   * 回歸：LongBench-E，確認短 context 沒退化。

### 10.6 指標與統計

* **時間**：TTFT 的平均、P50、P99，拆成四段；兩個加速比（對只算、對只載）；會合點的位置；有效頻寬。限速的結果一律標「模擬」〔E03 PoC 建議 7〕。
* **快取**：成本加權節省率；block 命中率；各層命中分布；Belady 與 BeladyCompute 的差距。
* **品質**：配對分數差＋CI；最差任務；輸出一致率。
* **統計**：
  * 真機每點 ≥6 次獨立 run（重啟 server），報中位數與無母數 95% CI。
  * 只有幾百個請求時不報 P99，改報 P90 或附 CI〔E11 檢查表 12〕。
  * 模擬是決定性的，所以改報不同 trace 時間窗或子樣本的變異。
* **學習式元件**：依時間切加依 session 分組；報校準（reliability diagram、Brier score），因為「成本變了只移動門檻」依賴機率校準，學習式七篇沒有一篇報〔E06 PoC 建議 7〕。

### 10.7 這次查證後，要改的地方（對照 RESEARCH_INTRO §10–§14）

| 原本的設定 | 改成 | 依據 |
|:--|:--|:--|
| 「每點至少 3 次，附信賴區間」 | 每點 ≥6 次獨立 run；P99 需要約 300 個樣本，否則報 P90 | §7.2 |
| 沒有「值不值得做」的前置檢查 | 加 G0：Belady 與 BeladyCompute 的差距 | §4.4、§10.1 |
| 在 vLLM 做 Cake 式還原 | 先決定架構：connector API 不能直接表達前算後載 | §2.3、§10.2 |
| 品質協定沒寫 query 何時可見 | query-agnostic、兩輪以上、同一 context 多問題 | §10.5 |
| INT4 層 | 同時測 KV4 與 K8V4；FP8 要自己量 | §10.5 |
| 讀取成本 | 加量「prefill 讀低精度 KV」的成本 | §10.5 |
| 冷快取 | 每輪用 `?reset_external=true` 清空，並另外確認各層為空 | §2.3 |
| Mooncake trace 重播 | chunk 設 512、時間單位換算，並驗算 `len(prompt)==input_length` | §2.5 |
| 量測端點 | v0.28.0 一律用 completions 端點 | §2.5 |
| 對手 | 加入 layer 維度（HCache／CacheFlow 式）、AsymCache 式、KVDrive 的寫入時放置、`store_threshold≥2` | §6.2 |
| 容量軸 | 對數尺度 15–20 點、三種單位 | §5 |
| 「Mooncake 的 128K 是模擬資料」 | 這句出自 arXiv v4，FAST'25 的真實 trace 最長 126,195 token | §12 |

---

## 11. 接下來四週怎麼學

時程對齊 `NEXT_STEPS_20261006.md` 表 1，這裡補上每週要讀的、要跑的、要交的。

| 週 | 讀（先讀卡片，再讀原文的指定章節） | 跑（練工具） | 交 |
|:--|:--|:--|:--|
| 1｜評測設定 v1 | E01（vLLM §6、Sarathi-Serve §5、DistServe §6）、E02、E11 §1–§2 | `vllm bench serve`：random 與 ShareGPT，掃請求率；練 manifest、暖機、清快取、6 次重複 | 一條「延遲對請求率」曲線＋完整 manifest＋CI；本文 §9.1 的 checklist 定稿 |
| 2｜Cake 還原與限速器 | E03（Cake 全文、CacheFlow、HCache、Pensieve）；E02 B2（connector API） | Cake Fig. 4 的逐 chunk 重算曲線；限速器自檢；會合點預測 | f(i) 曲線、限速器的誤差；§10.2 的架構決定 |
| 3｜H1 與品質試跑 | E07（KIVI、KVQuant、KVTuner）、E08（Rethinking、Agrawal & Mayer、陷阱表）、E09（RULER、SCBench） | 16K–512K 的只算、只載；RULER 子集在 16K／32K 跑 BF16 對 FP8 | 撞牆的位置；第一份配對 ε＋CI |
| 4｜E1 模擬與 G0 | E05（Fancy、LRB、AsymCache）、E06（LARU、LPC） | trace 驅動模擬：Mooncake 上的 LRU、Belady、BeladyCompute | G0 的結果；E1 的 P0–P4 與 Oracle；go/no-go 判定 |

**讀論文時每篇都問同樣的 20 格**（README 的評測卡格式）。特別注意三格：
* 「原文沒講清楚的地方」：這是重現時一定會卡住的地方；
* 「設計理由（原文）」與「設計理由〔判讀〕」要分開；
* 「與既有整理不一致」：讀到的跟我們先前寫的不同時，以原文為準。

---

## 12. 寄給老師的兩份 PDF：勘誤

以下每一條都經過兩道查證：抽取者提出、複核者回原文確認。**「不要改」那一類特別重要**：抽取者曾指控這些寫錯，複核後原文支持你原本的寫法。

### 12.1 確認要改

| # | 位置 | 原寫法（改寫） | 查證結果 | 卡片 |
|:--|:--|:--|:--|:--|
| 1 | intro 表 6 | KIVI／KVTuner 同一列，吞吐 2.35–3.47× | 這個數字只屬於 KIVI，而且是在短的 ShareGPT 合成負載上量的（平均輸入 161、輸出 338，p7–8）。KVTuner 的是 Table 8 的 +9.22%～+21.25%（p9）。同列的「2–4 bit」對 KVTuner 也不精確（它的精度對是 {2,4,8}²） | E07 |
| 2 | intro 表 5 | KIVI、KVTuner 列為 L1 代表，欄位是 BF16／FP8／INT4 | 兩篇都沒有 FP8 的實驗（KVTuner 只在 p2 有一句主張）。表 5 是把它們列為 L1 的代表，措辭要修，例如「INT2／INT4（無 FP8 實驗）」 | E07 |
| 3 | intro 表 8 | vLLM [40] 列在 LRU、ARC 那一列 | SOSP 論文裡沒有 LRU／ARC，只有序列級 all-or-nothing 的搶佔，再用 swap 或重算恢復（§4.5 p8）。放在「空間不夠時的逐出」沒錯，但 LRU／ARC 屬於現行的 OffloadingConnector 程式碼，要改引程式碼或部落格 [37] | E01 |
| 4 | intro L551、L853；表 17 | 「Mooncake 的 128K 是模擬資料 [11]」；「模擬 16K–128K、快取比例 50%」 | 這些內容出自 arXiv v4（Table 2 p15），但 [11] 引的是 FAST'25。FAST'25 的真實 trace 最長 126,195 token（p10）。要嘛改引 arXiv 版，要嘛改寫 | E04、E10、E11 |
| 5 | intro 表 16 | 每點至少 3 次，附信賴區間 | n=3 給不出 95% 的無母數 CI（中位數最大覆蓋率 0.75）。改成 ≥6 次 | E11 |
| 6 | sota §2.11；intro §5.2 | 「連 Belady 上界都還有明顯空間」「在各種容量下」 | 只在 HBM 大小的容量成立；到 1 TiB 時 LRU 與 Belady 幾乎重合（Fancy Fig. 2 p5、§5 p9）。intro 的措辭跟 Fancy 摘要一致，加「在 HBM 容量下」即可 | E05 |
| 7 | intro、sota | EvicPress「同品質下最多快 2.19 倍」 | 正文是對全 prefill、品質下降 <3% 的結果（p7、p9）；「同品質」是摘要的說法 | E05 |
| 8 | intro、sota | Fancy 的 19.9% | 只是平均 TTFT，中位數反而變差（p11） | E05 |
| 9 | intro [1] | Cake 的 72K／Llama2-70B／30 秒 | 只出現在 arXiv v1／v2 的 p1，ICML 正式版已刪除；[1] 引的是 ICML | E03 |
| 10 | intro | 「16K 左右是 Cake 自己說最划算的區域」 | ICML p6 只說兩種資源相當時最有利，沒有綁定 16K | E03 |
| 11 | intro、sota | 「CacheFlow 自己在 LMCache 上重做 Cake」 | CacheFlow 原文沒有這句（v1 p7、v2 p7、p10） | E03 |
| 12 | sota | HCache「最長 16K」 | 主實驗 ≤16K，但敏感度實驗到 32K（Fig. 11(i) p11） | E03 |
| 13 | sota | KVPR「只支援單 GPU」 | 單 GPU 加資料平行（p9 Limitations） | E03 |
| 14 | intro §2（表 1 下方，L137） | 依 AttentionStore 的評估，約 80% 的快取命中發生在磁碟層（Cake 引述）；大部分命中都要從最慢那層拿回來 | Cake 確實這樣寫（p2），但 CachedAttention 的 v1、v2、v3／ATC 版都找不到這個數字；原文報告超過 99.6%（v1 為 99.9%）的命中發生在 DRAM，因為預取先把 KV 拉進 DRAM。原文支持的是「多數 KV 只能放在磁碟，沒有 SSD 容量就沒有大部分命中」：由 Fig. 24 推算，靠 SSD 容量才得到的命中佔 78.8%–97.6%〔計算〕，Cake 的「約 80%」可能由此而來〔判讀〕。後半句與原文矛盾，應刪或加條件，並改引 CachedAttention 原文頁碼 | E04 |
| 14b | intro §7.1 | CachedAttention 的 session 最長 32K | 32K 是 Fig. 2 為了顯示而排除的範圍；評測模型的視窗是 2K–4K，溢出時截掉一半（p10、p12） | E04 |
| 15 | intro 表 9、§5.4；sota §3.1 | LPC 要訓練 118M 參數的嵌入模型；屬「訓練成本高、因換 LLM 而重訓」 | e5-small 是凍結的，只訓 3 層 MLP，不到 10 分鐘（p4–5）。118M 的說法來自 SAECache 的轉述（p11）。每天重訓是為了追使用者行為漂移，不是因為換 LLM（p5、p10） | E06 |
| 16 | intro 表 10；sota §3.2 | 「深度學習派訓練重、而且綁定 LLM」 | 對 L1 的四篇（KVP、ForesightKV、LookaheadKV、TRIM-KV），「綁定 LLM」成立。「訓練重」只能說到「要多張資料中心級 GPU」：KVP 作者自稱輕量，ForesightKV 單一設定約 64 H800 小時。對 LPC 不成立 | E06 |
| 17 | sota 表 S1 | LRB「保底：無」；HALP「訓練／更新：—」；LPC「保底：未查證」 | LRB 先用 LRU 頂著（proc. p533）；HALP 是線上訓練，每 1024 筆標記資料更新一次（proc. p1152）；LPC 可更正為「無」（p5–6） | E05、E06 |
| 18 | sota §2.3 | SGLang HiCache「被存取超過門檻（預設 2）次才備份（預設）」 | SGLang 現行的預設是 `write_through`；門檻 2 只用於 `write_through_selective` | E02 |
| 19 | intro §1.2 | 「NVIDIA Dynamo 有把 KV 卸載到 CPU／SSD 的機制」 | 目前不算錯，但 KVBM 在 v1.5.0 起棄用，v1.6.0 後就會過時。建議寫成「Dynamo 原有 KVBM（v1.5.0 起棄用），改走引擎原生卸載」 | E02 |
| 20 | intro 表 17 | RULER、LongBench v2 列在「負載合成方式」 | 兩者描述本身沒錯，但它們只提供內容，沒有到達時間或 session 結構。建議加註「只提供內容」 | E09 |
| 21 | sota §4.6 | SCBench「重用率是構造出來的」 | 正確。建議補：歷史用標準答案、預設截到 131,072、不量時間 | E09 |
| 22 | sota L395 | [31]（Agrawal & Mayer）已做 L1 跨方法基準 | 「L1」的範圍沒有錯；但系統效能是單請求、在 HF 上量的，要加這個限定 | E08 |

| 23 | intro 表 17 | HCache：真實的長文件，以 Zipf 分布合成到達 | ShareGPT4 是 Poisson 的 session 到達，輪間固定 30 s；L-Eval 主實驗是 batch size 1，沒有到達過程；Zipf 只用在 GPU 重用的子實驗（p10、p13） | E10 |
| 24 | sota 表 S4 註 | 「重用率為無限容量下的估計」 | 對 Mooncake、Bailian 成立；對 TraceLab（95.7%）與 Codex（94.2%）不成立，那兩個是 provider **實際**的快取命中 | E10 |
| 25 | intro §1.3、§11 | TraceLab 357,161 次呼叫、43 位開發者、中位約 12.4 萬 | 數字是 v0.0.1 的；2026-07-24 已發布 v0.0.2（665,453 列、52 人、中位 132,092）。要標版本 | E10 |
| 26 | intro §1.3 | 兩份生產 trace 的歷史區塊占 37.6%、貢獻 70.2% 的存取 [13] | 數字本身正確，但 37.6%／70.2% 只來自 FreeInference 一條 trace（Fancy p4–p5），不是兩份 trace 的共同結論 | E10 |

### 12.2 被質疑過、但原文支持你原本的寫法（不要改）

| 位置 | 你的寫法 | 原文依據 | 卡片 |
|:--|:--|:--|:--|
| intro 表 6 | AsymCache 1.90–2.03× | Fig. 11（8B＋LooGLE、低分散、QPS 0.04，p9）；§6.2 的 1.86×／1.91× 是 70B 的例子 | E05 |
| sota 表 S1、PAPERS_BY_LEVEL | Marconi「依命中情境估重用機率」 | 摘要 p1 與 §4.1 p5 就是這樣寫 | E05 |
| sota 表 S1 | LARU 預測「逐出優先順序」 | p5 §4 自己稱 predictor 的輸出為 eviction priorities | E06 |
| sota | Bidaw：時間戳換成 Poisson 後，依時間的逐出失效 | p12 | E04 |
| PAPERS_BY_LEVEL | CacheBlend 是 EuroSys'25 Best Paper | EuroSys 2025 官網 Awards 頁 | E07 |
| intro | py-kvcache「值不值得存」 | 原文主張本身包含「存」（p1、p2、p14），只是實作只擋載入 | E03 |
| intro L864–865 | 「只量時間（TTFT）時可以用合成 token」 | Cake p5。就 TTFT 而言成立，只需排除資料相依的方法 | E09 |
| intro L836 | [56] 已在 L1 做跨方法基準 | 沒有事實錯誤 | E08 |
| intro | 「換一台 server 就要重訓：視預測對象」 | 學習式七篇都沒做換硬體實驗；證據支持「視預測對象」這個結論 | E06 |

### 12.3 也要改的既有內部文件（`workloads_eval.md` 等）

| 檔案 | 問題 | 卡片 |
|:--|:--|:--|
| workloads_eval §1.1 Mooncake 列 | FAST 列的內容其實是 arXiv v4；525%、75%、「128K 是模擬資料」都只適用 arXiv 版 | E04 |
| workloads_eval §1.1 CachedAttention 列 | 「session ≤32K、≤40 輪」是 Fig. 2 為了顯示而排除的範圍，不是負載設定 | E04 |
| workloads_eval §1.1 CacheBlend 列 | 「皆 GQA」原文沒有標；「2×A40」應補 7B／34B 只用 1 張 | E07 |
| workloads_eval §1.2 KIVI 列 | Mistral-7B 標 GQA，但原文 §4.1 寫 MHA（原文自身不一致） | E07 |
| workloads_eval §3.3 | LooGLE「多數 >100K」：論文平均 20,887–36,412 token；源頭 Tutti p9 寫的是 "many" | E09 |
| workloads_eval §3.3 | SCBench「299K–3.17M 字元」只是英文 QA 的範圍；全體是 28,207–6,556,639 | E09 |
| workloads_eval §3.3 | SCBench「80% 重用」不是原文數字 | E09 |
| workloads_eval §4 LMCache 列 | multi_round_qa 是每使用者固定間隔，不是 Poisson | E02 |
| workloads_eval §4 SGLang 列 | `--use-trace-timestamps` 在現行 SGLang 對排程沒有作用（只影響輸出標籤） | E02 |
| workloads_eval §4 MLPerf 列 | 跨請求重用不是「未規範」：規則明文禁止跨 query 快取（L895–897） | E02 |
| workloads_eval §6.3 B.1–B.2 | 「TPOT／ITL」寫成一項；goodput 沒寫 attainment 目標 | E01 |
| workloads_eval §6.2 | 「SAECache 對 Bailian 聊天擬合的 log-normal：μ≈4.1、σ≈1.0（p17）」 | p17 確有這組數字，但它是線上更新規則收斂到的值，原文說它會低估 σ。對 Bailian 的離線 MLE 擬合是 μ=4.82、σ=1.25（p17 Table 3／4；複核者用 traceA 重算完全一致）。要用就用後者 | E10 |
| workloads_eval §3.2 | ShareGPT「多輪鏈可能不完整」標為判讀；現在可改成計算結果：43,459／94,145 筆是續段，36.7% 以 gpt 開頭 | E10 |
| PAPERS_BY_LEVEL | LMCache「在 AMD 上跑不起來」與原文自稱的 AMD 支援不符（實際能不能跑是平台實測的問題，要另外標明） | E04 |
| PAPERS_BY_LEVEL | KIVI 成果缺條件；「2-bit」實際約 3 bit | E07 |

---

## 附錄 A：評測卡索引與複核統計

| 卡片 | 內容 | 張數 | 複核檢查 | ❌ 改正 | ⚠️ 補註 |
|:--|:--|:--|:--|:--|:--|
| [E01](research_20261006_eval/cards/E01_serving_classic.md) | 經典 serving 論文＋指標字典 | 7 | 211 格 | 11 | 11 |
| [E02](research_20261006_eval/cards/E02_tools_ecosystem.md) | benchmark 工具與 KV 層系統 | 6 工具＋5 系統 | 179 項 | 5 | 6 |
| [E03](research_20261006_eval/cards/E03_restore_recompute.md) | 讀取時還原、重算與載入（含 Cake 重現規格） | 7＋1 | 237 格 | 5 | 15 |
| [E04](research_20261006_eval/cards/E04_tiered_systems.md) | 分層儲存系統 | 8 | 227 格 | 14 | 9 |
| [E05](research_20261006_eval/cards/E05_eviction_policy.md) | 逐出與快取策略（含 CDN） | 8 | 229 格 | 7 | 6 |
| [E06](research_20261006_eval/cards/E06_learned_eviction.md) | 學習式逐出 | 7 | 約 280 格 | 12 | 3 |
| [E07](research_20261006_eval/cards/E07_compression.md) | 壓縮與量化 | 9 | 285 格 | 13 | 10 |
| [E08](research_20261006_eval/cards/E08_sparse_and_critiques.md) | 稀疏注意力與評測批判 | 9 | 280 格 | 28 | 1 |
| [E09](research_20261006_eval/cards/E09_quality_benchmarks.md) | 長 context 品質 benchmark | 15 | 296 格 | 11 | 7 |
| [E10](research_20261006_eval/cards/E10_workload_datasets.md) | 負載資料集（含 ShareGPT 的逐筆重算） | 12＋2 表 | 299 格；約 190 個數字用獨立腳本重算 | 6 | 11 |
| [E11](research_20261006_eval/cards/E11_systems_eval_craft.md) | systems 評測方法學 | 7 篇拆解＋方法學 | 約 381 項 | 4 | 5 |
| **合計** | | | **約 2,904 格／項** | **116** | **84** |

* **E08 改正最多**（28 格，其中 7 格只是頁碼），表示壓縮與 benchmark 批判這一組最需要複核。
* **複核也推翻了抽取者的指控**：12.2 節列的九條，抽取者原本說你的文件寫錯，回原文後是你對。這是 double check 的主要價值：不只抓抽取者漏掉的錯，也防止我們去改原本正確的內容。
* **限制**：
  * 複核者與抽取者是同一個模型家族，不是 cross-model review（專案 CLAUDE.md §5 的落差）。
  * 圖上的數值多半是讀圖；只有 Cake Fig. 4 做了 800 dpi 數位化。
  * OpenReview 的審稿意見讀不到（§7.6）。

## 附錄 B：原論文本身的問題

複核時發現、原文自身前後不一或與自己引用的公式不符的地方。引用這些論文時要小心。

| 論文 | 問題 | 卡片 |
|:--|:--|:--|
| Cake | 內文說 H100 在 100%、32 Gbps 是 2.23×，Table 3 是 2.40（2.23 是 87.5% 那格）；App. B.2 的切 chunk 方式與 §5.1 的「I/O chunk 128」說法不一致 | E03 |
| Bottlenecks | 說省略的 context 項在 65K 時約讓 F_pf 增加 10%（p10）；用它自己引的公式算，Llama-3.1-70B 是 +61%、Qwen3-235B-A22B 是 +229% | E03 |
| Splitwise | 摘要的 2.35× 與正文的 2.15× 對不上；表中 Llama2-70B 的 head 數與 HF config 不符；公開 trace 的檔內時間與論文寫的日期、時長不一致 | E01 |
| Sarathi-Serve | 摘要與正文的倍數對不上 | E01 |
| Etalon | 把 vLLM 的 normalized latency 說成取中位數（vLLM 原文是平均） | E01 |
| KIVI | §4.1 把 Mistral-7B 寫成 MHA，附表是 GQA | E07 |
| CacheGen | 機率表的來源，§5.2 說每個 LLM 離線 profile 一次，§6 說由對應 context 統計 | E07 |
| CacheBlend | 2.2–3.3× 的比較對象在 p1、p2、p10、p12 寫法不一；作為對照的 prefix caching 假設載入零延遲（p11） | E07 |
| PyramidKV | 摘要的 12%／0.7% 與 Table 2 的分母對不上 | E07 |
| Agrawal & Mayer | TurboQuant 的離群值來自 OOM 樣本被跳過；SnapKV 在 query-aware 設定下跑；CaM 的生成上限一律 64；論文都沒寫 | E08 |
| SCBench | 論文 931 sessions，HF 實際 922 列 | E09 |
| RULER | 獨立腳本在 2025-01-22 到 2026-07-22 之間遺失 answer prefix（issue #107，#108 修正） | E09 |
| LongBench v2 | 失敗的請求會被丟掉，分母只算寫出的列（issue #144） | E09 |
| in-the-wild（Bailian） | 論文寫 hash 粒度 4 token，公開檔是 16 token | E05 |

## 附錄 C：參考來源

各篇的完整出處（版本、URL、讀了哪些頁）在各卡片的「來源清單」一節。下表只列本文直接引用的主要來源。寄給老師的版本若需要 IEEE 格式，已在 RESEARCH_INTRO 參考文獻中的條目沿用原格式，新增的條目要再補作者全名。

| 簡稱 | 標題 | 場所 | 連結 |
|:--|:--|:--|:--|
| Orca | Orca: A Distributed Serving System for Transformer-Based Generative Models | OSDI 2022 | https://www.usenix.org/system/files/osdi22-yu.pdf |
| vLLM | Efficient Memory Management for Large Language Model Serving with PagedAttention | SOSP 2023 | https://arxiv.org/abs/2309.06180 |
| SGLang | Efficient Execution of Structured Language Model Programs | NeurIPS 2024 | https://arxiv.org/abs/2312.07104 |
| Sarathi-Serve | Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve | OSDI 2024 | https://www.usenix.org/system/files/osdi24-agrawal.pdf |
| DistServe | DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving | OSDI 2024 | https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf |
| Splitwise | Splitwise: Efficient Generative LLM Inference Using Phase Splitting | ISCA 2024 | https://arxiv.org/abs/2311.18677 |
| Etalon | Etalon: Holistic Performance Evaluation Framework for LLM Inference Systems | arXiv 2024 | https://arxiv.org/abs/2407.07000 |
| Cake | Compute or Load KV Cache? Why Not Both? | ICML 2025 | https://proceedings.mlr.press/v267/jin25d.html |
| CacheFlow | CacheFlow: Efficient LLM Serving via Automated 3D-Parallel KV Cache Restoration | arXiv 2026 | https://arxiv.org/abs/2604.25080v2 |
| Pensieve | Stateful Large Language Model Serving with Pensieve | EuroSys 2025 | https://arxiv.org/abs/2312.05516v3 |
| HCache | Fast State Restoration in LLM Serving with HCache | EuroSys 2025 | https://arxiv.org/abs/2410.05004v1 |
| KVPR | KVPR: Efficient LLM Inference with I/O-Aware KV Cache Partial Recomputation | ACL Findings 2025 | https://aclanthology.org/2025.findings-acl.997.pdf |
| Bottlenecks | Understanding Bottlenecks for Efficiently Serving LLM Inference with KV Offloading | arXiv 2025（MLSys 2026 錄取未查證） | https://arxiv.org/abs/2601.19910v1 |
| py-kvcache | Building py-kvcache: A Performance Characterization of External KV Caching for vLLM with NVMe SSDs | arXiv 2026 | https://arxiv.org/abs/2609.11744v1 |
| Strata | Strata: Hierarchical Context Caching for Long Context Language Model Serving | OSDI 2026 | https://www.usenix.org/system/files/osdi26-xie-zhiqiang.pdf |
| Bidaw | Bidaw: Enhancing Key-Value Caching for Interactive LLM Serving via Bidirectional Computation–Storage Awareness | FAST 2026 | https://www.usenix.org/system/files/fast26-hu-shipeng.pdf |
| MTDS | Multi-tier dynamic storage of KV cache for LLM inference under resource-constrained conditions | Complex Intell. Syst. 2026 | https://doi.org/10.1007/s40747-025-02200-4 |
| LMCache | LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference | arXiv 2025 | https://arxiv.org/abs/2510.09665 |
| Mooncake | Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot | FAST 2025（另有 arXiv v4，內容不同） | https://www.usenix.org/system/files/fast25-qin.pdf |
| CachedAttention | Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention | ATC 2024 | https://www.usenix.org/system/files/atc24-gao-bin-cost.pdf |
| Tutti | Tutti: Making SSD-Backed KV Cache Practical for Long-Context LLM Serving | arXiv 2026 | https://arxiv.org/abs/2605.03375 |
| KVDrive | KVDrive: A Holistic Multi-Tier KV Cache Management System for Long-Context LLM Inference | arXiv 2026 | https://arxiv.org/abs/2605.18071 |
| Fancy-eviction | When Fancy Eviction Fails: Rethinking Cache Replacement for LLM Prefix Reuse | arXiv 2026 | https://arxiv.org/abs/2609.28870 |
| AsymCache | Multi-Segment Attention: Enabling Efficient KV-Cache Management for Faster LLM Serving | arXiv 2026 | https://arxiv.org/abs/2606.02964 |
| Marconi | Marconi: Prefix Caching for the Era of Hybrid LLMs | MLSys 2025 | https://arxiv.org/abs/2411.19379 |
| AdaptCache | AdaptCache: KV Cache Native Storage Hierarchy for Low-Delay and High-Quality Language Model Serving | SOSP 2025 BigMem Workshop | https://arxiv.org/abs/2509.00105 |
| EvicPress | EvicPress: Joint KV-Cache Compression and Eviction for Efficient LLM Serving | arXiv 2025 | https://arxiv.org/abs/2512.14946 |
| in-the-wild | KVCache Cache in the Wild: Characterizing and Optimizing KVCache Cache at a Large Cloud Provider | ATC 2025（讀的是 arXiv v5） | https://arxiv.org/abs/2506.02634 |
| LRB | Learning Relaxed Belady for Content Distribution Network Caching | NSDI 2020 | https://www.usenix.org/system/files/nsdi20-paper-song.pdf |
| HALP | HALP: Heuristic Aided Learned Preference Eviction Policy for YouTube Content Delivery Network | NSDI 2023 | https://www.usenix.org/system/files/nsdi23-song-zhenyu.pdf |
| LARU | Toward Robust and Efficient ML-Based GPU Caching for Modern Inference | arXiv 2025 | https://arxiv.org/abs/2509.20979 |
| LPC | Learned Prefix Caching for Efficient LLM Inference | NeurIPS 2025 | 見 E06 來源清單 |
| SAECache | Not All Tokens Are Worth Caching: Learning Semantic-Aware Eviction for LLM Prefix Caches | arXiv 2026 | https://arxiv.org/abs/2605.18825 |
| KVP | Learning to Evict from Key-Value Cache | ICML 2026 | https://arxiv.org/abs/2602.10238 |
| ForesightKV | ForesightKV: Optimizing KV Cache Eviction for Reasoning Models by Learning Long-Term Contribution | ICML 2026 | https://arxiv.org/abs/2602.03203 |
| LookaheadKV | LookaheadKV: Fast and Accurate KV Cache Eviction by Glimpsing into the Future without Generation | ICLR 2026 | https://arxiv.org/abs/2603.10899 |
| TRIM-KV | Cache What Lasts: Token Retention for Memory-Bounded KV Cache in LLMs | ICLR 2026 | https://arxiv.org/abs/2512.03324 |
| StreamingLLM | Efficient Streaming Language Models with Attention Sinks | ICLR 2024 | https://arxiv.org/abs/2309.17453 |
| H2O | H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models | NeurIPS 2023 | https://arxiv.org/abs/2306.14048 |
| SnapKV | SnapKV: LLM Knows What You are Looking for Before Generation | NeurIPS 2024 | https://arxiv.org/abs/2404.14469 |
| PyramidKV | PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling | arXiv | https://arxiv.org/abs/2406.02069 |
| KIVI | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache | ICML 2024 | https://arxiv.org/abs/2402.02750 |
| KVTuner | KVTuner: Sensitivity-Aware Layer-wise Mixed Precision KV Cache Quantization | ICML 2025 | https://arxiv.org/abs/2502.04420 |
| KVQuant | KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization | NeurIPS 2024 | https://arxiv.org/abs/2401.18079 |
| CacheGen | CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving | SIGCOMM 2024 | https://arxiv.org/abs/2310.07240 |
| CacheBlend | CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion | EuroSys 2025 | https://arxiv.org/abs/2405.16444 |
| MInference | MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention | NeurIPS 2024 | https://arxiv.org/abs/2407.02490 |
| DuoAttention | DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads | ICLR 2025（會議版未比對） | https://arxiv.org/abs/2410.10819 |
| Quest | Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference | ICML 2024 | https://arxiv.org/abs/2406.10774 |
| ShadowKV | ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference | ICML 2025 | https://arxiv.org/abs/2410.21465 |
| InfiniGen | InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management | OSDI 2024 | https://arxiv.org/abs/2406.19707 |
| Yuan'24 | KV Cache Compression, But What Must We Give in Return? | EMNLP 2024 Findings | https://arxiv.org/abs/2407.01527 |
| Rethinking'25 | Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving | MLSys 2025 | https://arxiv.org/abs/2503.24000 |
| Agrawal & Mayer | Benchmarking KV-Cache Optimizations across Task Quality and System Performance for Long-Context Serving | arXiv 2026（未正式出版） | https://arxiv.org/abs/2607.05399 |
| kvpress | NVIDIA kvpress | GitHub | https://github.com/NVIDIA/kvpress |
| LongBench | LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding | ACL 2024 | https://arxiv.org/abs/2308.14508 |
| LongBench v2 | LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks | ACL 2025 | https://arxiv.org/abs/2412.15204 |
| RULER | RULER: What's the Real Context Size of Your Long-Context Language Models? | COLM 2024 | https://arxiv.org/abs/2404.06654 |
| ∞Bench | ∞Bench: Extending Long Context Evaluation Beyond 100K Tokens | ACL 2024 | https://arxiv.org/abs/2402.13718 |
| SCBench | SCBench: A KV Cache-Centric Analysis of Long-Context Methods | ICLR 2025 | https://arxiv.org/abs/2412.10319 |
| HELMET | HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly | ICLR 2025 | https://arxiv.org/abs/2410.02694 |
| Vidur | Vidur: A Large-Scale Simulation Framework for LLM Inference | MLSys 2024 | https://arxiv.org/abs/2405.05465 |
| Hoefler & Belli | Scientific Benchmarking of Parallel Computing Systems | SC 2015 | https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf |
| Heiser | Systems Benchmarking Crimes | 網頁 | https://gernot-heiser.org/benchmarking-crimes.html |
| Kalibera & Jones | Rigorous Benchmarking in Reasonable Time | ISMM 2013 | 見 E11 來源清單 |
| Mytkowicz et al. | Producing Wrong Data Without Doing Anything Obviously Wrong! | ASPLOS 2009 | 見 E11 來源清單 |
| MLPerf | MLCommons Inference Rules | GitHub | https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc |
| ACM AE | Artifact Review and Badging, Version 1.1 | ACM | 見 E11 來源清單 |
| vLLM bench | `vllm bench serve` 原始碼 | GitHub | https://github.com/vllm-project/vllm （commit 見 E02） |
| LMCache code | LMCache 原始碼與文件 | GitHub | https://github.com/LMCache/LMCache （commit 見 E02） |
| SGLang HiCache | SGLang HiCache 設計文件 | GitHub | https://github.com/sgl-project/sglang （commit 見 E02） |
