# SOTA Matrix — 老師指定的五篇

**日期**：2026-09-19
**方法**：104 個 agent、22 份一手來源全文抓取、110 條 claim 抽取、**取前 25 條做 3 票對抗式驗證**。
**結果**：**12 條確認、13 條被推翻、0 條無法判定。**

> ⚠️ **本文件只寫「通過驗證」的內容。** 被推翻的 13 條列在 §6，**不可引用**——
> 其中好幾條聽起來對我們有利（例如「Strata 完全沒有 policy」「ShareGPT 零 headroom」），
> 但沒撐過驗證。寫進論文會被審稿人抓到。

---

## 1. 一句話總結

**這五篇沒有互相重疊，而是把 KV cache 的決策空間切成不相交的幾塊。**
每一篇最強的地方，正好是其他篇完全不談的地方。
**沒有任何一篇同時決定「放哪一階 + 幾位元 + 要不要重算」。**

---

## 2. 主表

| | **Cake** (ICML'25 / PMLR v267) | **AdaptCache** (SOSP'25 **BigMem workshop**) | **Strata** (OSDI'26) | **Bidaw** (FAST'26) | **MTDS** (Springer) |
|---|---|---|---|---|---|
| **解什麼問題** | 已經確定 cache 命中後，這段 KV 要用算的還是用載的 | 壓縮方法 + 壓縮率 + DRAM/SSD 放置 + 逐出，四者一起選 | 多階 KV 的 **I/O 機制**與 cache-aware 批次排程 | 從 DRAM performance layer 逐出誰 | **未知** |
| **核心機制** | **雙指標對撞**：`compute_ptr=0` 往前、`io_ptr=N-1` 往後，交會點就是切分點 | NP-hard MCKP，用 **greedy** 解（作者自承次佳） | page-first layout + GPU-assisted gather/scatter；**每一階都是 LRU** | 用**上一輪回答長度**預測 hit potential（Spearman 0.94–0.98），對 Belady ghost cache 線上校準 | **未知** |
| **決策訊號** | **無**。只有一個二元 residency check；chunk 大小是離線調好的常數（COMP 512 / FETCH 128） | **歷史命中頻率（LFU 式）**＋離線品質曲線 | 靜態設定 | **單一手挑特徵**（上一輪答案長度） | **未知** |
| **learned?** | ❌ | ❌ `learn/neural/regress/predict` 全文 = 0 | ❌ | 🟡 predictive 但非 ML | **未知** |
| **控 GPU HBM** | — | ❌ | ✅ | — | **未知** |
| **控 CPU DRAM** | — | ✅ | ✅ | ✅ | **未知** |
| **控 SSD** | — | ✅ | 🟡 **只在 §5.3.5 一個子實驗出現** | ✅ | **未知** |
| **控 remote/object** | ❌ | ❌ 全文 remote/network/multi-node = 0 | ❌ | — | **未知** |
| **控 recompute-vs-load** | ✅ **唯一一篇** | ❌ 重算只當靜態 baseline | — | — | **未知** |
| **控 bit-width** | ❌ 壓縮是**外生**的（「agnostic to its representation or compression scheme」） | ✅ **唯一一篇** | ❌ **零壓縮**，全文 compress\|quantiz\|fp8\|int8\|bit-width 只有 2 次命中，**都在參考文獻** | — | **未知** |
| **品質/精度軸** | ❌ 無任何 accuracy 指標 | ✅ | ❌ **lossless by design**，明文對比 CacheGen/CacheBlend | — | **未知** |
| **評測硬體** | 2×A100 80GB NVLink / 1×H100；**無任何儲存裝置** | **1×A100**，100 GB DRAM，400 GB SSD @1 GB/s | 8×H200（1.6 TB DRAM, PCIe 5.0）＋8×H20＋GH200 | 200 GB DRAM performance layer | **未知** |
| **serving engine** | 自建 | **未指名**（vLLM/SGLang/LMCache 全文 = 0） | **SGLang HiCache**（已 upstream） | — | **未知** |
| **trace** | 無真實 trace | LongBench 六個資料集，**Poisson 合成到達時間** | LooGLE/NarrativeQA/ReviewMT/ShareGPT，**Poisson 合成**；Mooncake 只當**模擬**子實驗 | **自有 >1M 輪真實 trace（未公開）**＋ShareGPT | **未知** |
| **頭條數字** | 對 I/O-only 的 speedup（見下方失敗點） | TTFT −56% vs prefill（品質掉 15% 內）；−69% vs KIVI | 吞吐量最高 **5×** vs vLLM-LMCache、**3.75×** vs TRT-LLM（皆在 70B） | miss rate −57.6% vs queue-enhanced、−69.9% vs LFU/LRU/FIFO | **未知** |
| **程式碼** | ❌ 無 | ❌ 無 | 🟡 無 artifact badge、論文無 repo URL，但 code 在 SGLang HiCache | ❌ 關鍵 trace 專有 | **未知** |
| **我這台跑得起來嗎** | 🟡 需重寫（它的儲存是模擬的） | 🟡 需自己實作（無 code） | ❌ 需 SGLang + 8 卡；我只有 1 卡 | ❌ trace 不公開 | **未知** |

---

## 3. 🔴 老師要的那一欄：「最強方法在什麼條件下會做錯決定」

這是整份研究最有價值的部分。**三篇有明確、可引用的失敗面。**

### 3.1 Cake — 有 **5 個實測的失敗點**，而且作者自己承認缺的正是成本模型

> **Cake 的 compute-vs-load 沒有成本模型、沒有估計器、沒有 fallback。**
> 機制是無條件的雙指標對撞，唯一的理由是一句位置不對稱性：
> 「Compute cost increases for later tokens, while I/O cost remains constant regardless of token position.」

**實測它比單一資源 baseline 還慢的 5 個設定**（Table 6，2×A100，16k seq，chunk 512）：

| 設定 | 相對 baseline |
|---|---|
| 0.99×、0.98×、0.91× | 略輸 |
| **Llama-3.1-70B @ GPU 利用率 12.5% / 32 Gbps** | **0.80× vs I/O-only** → **TTFT 比「單純去載」高 25%** |
| Table 5 另有一點 | 0.99× vs compute-only |

**作者自己寫的未來工作**（逐字）：
> "This can be mitigated by incorporating a **fallback mechanism** that dynamically adjusts resource allocation when one resource significantly outperforms the other."
> "we can optimize this scheduling strategy by incorporating an **estimation mechanism**."

**這就是老師要的那句話的答案：**
> **「當一種資源明顯強過另一種時（資源不平衡），Cake 仍會硬把一部分工作分給較慢的那邊，因此輸給什麼都不做的單一資源策略。Tiara 的 κ 成本模型正是在做 Cake 承認自己沒有的那個估計器。」**

⚠️ **精確度要求**（否則會被審稿人打）：
- 不可寫「Cake 沒有 scheduler」——它**有**一個獨立的 adaptive priority scheduler（decode > non-prefix-cache prefill > prefix-cache prefill），只是那個不決定 compute/load 切分。
- 要寫「demonstrably / measured」不是「provably」。
- 最糟的那一點發生在 **12.5% GPU 利用率**，是特定 regime，不是普遍失效。

### 3.2 Cake — **量測路徑裡沒有任何真實儲存裝置**

全部 5 種「I/O 頻寬」設定（7/25/32/56/100 Gbps）都是**用 chunk 大小和標稱頻寬算出延遲，然後暫停傳輸**來模擬的。逐字：

> "We **simulate** the chunk I/O loading process by calculating the appropriate delay time based on the chunk size and network bandwidth. The simulated storage backend is then set to **pause data transfer** until the specified delay has elapsed."

佐證：56 Gbps = 剛好 7,000 MB/s = Samsung 980 PRO 的**規格書**循序讀數值。
§5.1 硬體段落**沒有指名任何儲存裝置**，而這是一篇主題就是儲存 I/O 頻寬的論文。

→ **沒有 SATA vs NVMe 的差別、沒有裝置佇列、沒有 tail latency、沒有檔案系統效應。**
→ **我們有真實裝置量測（fio + vLLM 自己的計數器），這是一個實質差異。**

### 3.3 AdaptCache — 兩個決策訊號都**不具預測性**

> 「未來重用」= **純粹的歷史命中頻率**（LFU 式），無冷啟動初始化、無 aging/decay/recency 項。
> 品質-壓縮率曲線 = **離線**建立，每個資料集只抽 **10 筆**，用 **GPT-4o 產生的問題**。

全文 `predict=0`、`Belady=0`、`oracle=0`，唯一的 `learn` 命中在參考文獻標題。
效用函數對頻率是乘法的，所以**零歷史的新條目會被推向最大壓縮或直接逐出**。

→ **我們的 GBDT 預測器 AUC 0.917–0.922、ECE 0.003–0.004，是 predictive 的。這是可辯護的差異點。**
→ 但注意：AdaptCache 是 **5 頁 workshop 論文**，評測段落標題就叫 "Preliminary Results"。**別把它當成強 baseline 來打，那樣顯得我們挑軟柿子。**

### 3.4 Strata — 完全沒有品質/精度軸，SSD 幾乎沒評測

- **零壓縮、零量化、零 bit-width**：全文 `compress|quantiz|fp8|int8|bit-width|precision` 只有 2 次命中，**都在參考文獻**。明文自我定位為 lossless：「Unlike these approximate caching schemes, Strata does not impact the accuracy of requests.」
- **SSD 幾乎沒評測**：§5.1 逐字「Disk storage is **not used in most benchmarks except in §5.3.5** due to limited support in baseline systems.」主測試台的碟 **< 1 GiB/s**。唯一的碟結果只有 2.1× TTFT / 1.3× 吞吐。
- **remote/object storage 完全不存在**。
- §6 自承：「an equally promising direction is to create scheduling headroom for hiding the latency of **prefetching from slower storage tiers**」。

⚠️ **但四條更強的「Strata 沒有 X」全部被推翻**（見 §6）。**只能用上面這兩條。**

---

## 4. 🎯 The Gap — 沒有任何一篇佔領的位置

驗證結論（high confidence）：

> **一個明確的成本模型，同時仲裁「重算 vs 載入」、「放哪一階」、「幾位元」三個決策，
> 並在某一種資源明顯佔優時有 fallback，而且由 predictive（非歷史）訊號驅動。**
> **五篇裡有兩篇把這件事的一部分寫成自己的未來工作，沒有一篇實作。**

**為什麼這三個決策不能分開做**（這是 gap 的核心論證）：

```
改變 bit-width → 改變位元組數 → 改變載入時間 → 移動 Cake 的 compute/load 交叉點 → 改變該放哪一階
```

- **AdaptCache** 選 bit-width，但把交叉點當**固定**
- **Cake** 決定交叉點，但把 bit-width 當**外生**（「agnostic to its representation or compression scheme」）
- 另一篇獨立的模擬研究更直接寫道：「placement composes with KV quantization and eviction... **we hold those fixed to isolate placement**」

**這正是 Tiara 論文的六態動作空間 {GPU-BF16, GPU-FP8, GPU-INT4, CPU, SSD, DROP+重算} 的設計動機。**
**gap 是真的。**

### 4.1 但要對自己誠實：Tiara 目前還沒填上這個 gap

| gap 的組成 | Tiara 現況 |
|---|---|
| 明確的成本模型 | ✅ **有，而且是真機量的**（κ 表）——這是最強的一塊 |
| 仲裁 recompute-vs-load | ✅ 模擬器內已實作 |
| 仲裁 tier placement | ✅ 模擬器內已實作 |
| **仲裁 bit-width** | 🟡 **只有 INT8、只有平台 A**；FP8/INT4 從未進入策略；oracle 根本沒有精度階 |
| 資源不平衡時的 fallback | ❌ 未實作 |
| **predictive 訊號** | ✅ GBDT，AUC 0.917–0.922 |
| **在真實 serving engine 上跑** | ❌ **從未**。全部在模擬器裡 |
| 公開 trace + 回報變異 | 🟡 用公開 Mooncake trace ✅，但**從未回報變異** |

→ **gap 存在，方向站得住；但 Tiara 目前佔領的是 gap 的一半，而且是在模擬器裡。**

---

## 5. 三個對我們特別有利的方法學發現

### 5.1 ⭐ 回報變異數會**超越**這個領域的現行水準，不只是及格

> 「**None of the verified evaluations reports seeds, run-to-run variance, confidence intervals, or A/A testing**；
> 這個子領域的事實標準是單次執行的點估計，有時還報到小數點後四位。
> **一篇回報變異的新論文會超越、而不只是符合現行慣例。**」

**老師要求做 A/A test，不只是研究誠信，它本身就是一個 contribution。**
（注意：Strata 與 Bidaw 的變異回報情況**未被窮舉驗證**，不可斷言。）

### 5.2 ⭐ 模擬器在頂會是可接受的，但只限特定形式

- **ICML'25 接受了 Cake**，其**整條儲存 I/O 路徑都是解析式模擬**，無真實裝置、無 artifact evaluation、無 code。
- **OSDI'26 接受了 Strata** 的 Mooncake 模擬子實驗，理由是測試台吞吐不足。

**可接受的兩種形式：**
1. 真實 GPU 運算 + 模擬頻寬的 harness
2. 因測試台限制而做的子實驗（並明說理由）

**要避免的反例**：一篇 5 天前的 arXiv preprint，模擬器「calibrated to a random-forest execution-time predictor」，但全文對那個預測器**沒有任何 R²/MAPE/驗證**。
→ **我們的 `simulator_validation.json`（4 個模型中 3 個在 14–29% 內吻合）正是那篇缺的東西。要把它放進論文。**

### 5.3 ⭐ 「公開 trace 到底有沒有 headroom」是未解問題 — 這本身可以是貢獻

- **Bidaw 是這個子領域唯一量化的正 headroom 證據**，但建立在**未公開的專有 trace** 上（>1M 輪、平均 22.4 輪/使用者）。
- 唯一的「近零 headroom」證據是 preprint / GitHub 等級，而且是在 **block-hash prefix cache 粒度**上量的，與 Bidaw 的**每使用者對話粒度**不同 → 兩者不直接衝突。
- 公開 trace **普遍缺少 timestamp 與使用者身分**，所以大家都用 Poisson 合成到達時間。Azure / BurstGPT / LMSys 在這五篇裡**完全沒出現**。

> **「在一個可釋出的公開 trace 上重現 Belady-gap 量測，並判定這個 gap 在從對話粒度換到 prefix-cache 粒度後是否存活」
> ——研究報告明文說這「既是未解問題，也是一個可獨立成立的貢獻」。**

**這正好是我們已經在做的事**（Mooncake trace + Belady oracle + 7 個模型），而且**我們的負面結果（原始順序 headroom 只有端到端 4%）在這個框架下變成資料點，不是失敗。**

---

## 6. ⛔ 被推翻的 13 條 — 不可引用

這些聽起來很有用，但沒撐過 3 票驗證。**寫進論文會被抓。**

| 被推翻的 claim | 票數 |
|---|---|
| Strata 完全沒有 eviction/admission/placement policy，只有 LRU | 1-2 |
| Strata 在 §6 明確放棄三項能力 | 0-3 |
| Strata 提供了 Mooncake 的 headroom 量測（38% 請求共享 ≥6k prefix） | 0-3 |
| Strata 的 SSD 在**所有** benchmark 都被排除 | 0-3 |
| Strata 的每個決策都是靜態手調門檻 | 1-2 |
| Strata 沒有 compute-vs-load policy | 1-2 |
| AdaptCache 的範圍嚴格限於 DRAM↔SSD，**證明**無法決定 HBM 或 remote | 1-2 |
| Bidaw 證明 ShareGPT 零 headroom | 0-3 |
| Bidaw 的控制面刻意窄（明確排除量化為 orthogonal） | 1-2 |
| 73.02× 只是 1:8:64 階層形狀的記帳假象 | 0-3 |
| 跨階 prefetch 不划算（oracle prefetcher 有一半格子更慢） | 0-3 |
| Alibaba/Aliyun trace 的理想無限快取命中率上限 62%/54% | 0-3 |
| 放置策略的延遲 headroom（某條敘述） | 0-3 |

**特別注意**：Bidaw 的**排除邊界從未被建立**。安全的 Bidaw 那一格**只能寫**：trace 特性描述、答案長度預測器 + ghost cache、57.6%/69.9% 這兩個數字。

---

## 7. MTDS — 空白列

**零條 claim 通過驗證。沒有抽到任何一手內文。**

> 「**不可用標題或摘要推斷填補這一列。**」

矩陣目前是**四篇寬，不是五篇**。要補這一列，必須真的拿到 Springer 全文（付費牆）。
另外它的 venue 等級（Springer 期刊 vs OSDI/FAST/ICML）在並列時需要獨立評估其評測嚴謹度。

---

## 8. 對「Tiara 到底要打誰」的回答

| 系統 | 是不是我們的對手 | 怎麼定位 |
|---|---|---|
| **vLLM** | ❌ **不是** | 老師說得對——它是我們的**底座**，不是 SOTA。比贏 vLLM 內建的 LRU/ARC 不構成貢獻 |
| **Cake** | ✅ **主要對手** | 我們的 κ 成本模型 = Cake 承認自己缺的 estimator + fallback。**它有 5 個實測失敗點可以打** |
| **AdaptCache** | 🟡 **次要** | 唯一同樣做 bit-width 的。我們的 GBDT 是 predictive，它的是 LFU。**但它只是 5 頁 workshop 論文，不能當主要打擊對象** |
| **Strata** | ❌ **不是直接對手** | lossless + 無 bit-width，決策空間不重疊。但它設定了 **I/O 機制的標準**，我們要引用它、不要宣稱贏它 |
| **Bidaw** | ❌ **不是對手，是盟友** | 它證明了 headroom 在多輪對話 workload 上存在。**引用它來支持我們的題目值得做** |
| **MTDS** | ❓ | 資訊不足 |

### 最核心的一句 claim（候選）

> **「多階 KV cache 的三個決策——重算 vs 載入、放哪一階、幾位元——是耦合的：
> 改變位元寬會移動重算/載入的交叉點，進而改變最佳放置。
> 現有系統各自固定其中兩個來隔離第三個（Cake 固定位元寬、AdaptCache 固定交叉點），
> 因此在資源不平衡時會做出可量測的錯誤決策（Cake 實測 5 個設定輸給單一資源 baseline，最差 0.80×）。
> Tiara 用真機量測的成本模型 κ 同時仲裁三者。」**

⚠️ **但這句話現在還不能寫**，因為：
1. Tiara 的 bit-width 只有 INT8、只有平台 A
2. 從未在真實 serving engine 上跑過
3. 還沒證明「聯合最佳化贏過最佳的循序組合」（先選 bit-width，再跑 Cake）

**研究報告把第 3 點列為 open question**：
> 「Nobody has measured whether joint optimization beats the best sequential composition...
> **which determines whether the identified gap is worth a paper.**」

**→ 這就是下一個實驗。而且它只需要模擬器 + 現有的成本常數，設備完全足夠。**

---

## 附錄：來源與驗證方式

- 一手 PDF 全文抓取：OSDI'26 Strata、FAST'26 Bidaw、PMLR v267 Cake、arXiv AdaptCache
- 驗證方式：每條 claim 由 3 個獨立 agent 對抗式投票，需 2/3 反駁才推翻
- **來源不對稱**：Strata/Bidaw/Cake 是頂會同儕審查的一手 PDF；**AdaptCache 是 5 頁 workshop 論文**，評測段落自題為 "Preliminary Results"，數字要相應降權
- **版本漂移**：Strata arXiv v1 寫「5× lower TTFT」，OSDI camera-ready 寫「5× throughput」→ **引用 camera-ready**。兩版的節號也不同（碟實驗在一版是 §5.3.4、另一版 §5.3.5）
- **時效**：Strata 釘住的 baseline 版本（vLLM v0.8.5、LMCache v0.2.1、TRT-LLM v0.17.0、SGLang v0.4.5）到 2026-09-19 已經過時數月，任何重測都不會重現那些比值
