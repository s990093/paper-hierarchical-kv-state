# 老師指定的五篇 — 完整解說

**日期**：2026-09-20
**證據等級**：本文只寫通過 3 票對抗式驗證的內容（12 條確認／13 條推翻）。被推翻的列在 §7，**不可引用**。

---

## 0. 先回答：為什麼是這五篇？

老師不是隨機挑的。**他把你論文宣稱的貢獻拆成四個決策，然後每個決策各找一篇目前最強的。**

你的論文說 Tiara 的動作空間是六態：

```
{GPU-BF16, GPU-FP8, GPU-INT4, CPU, SSD, DROP+重算}
```

這六態其實是**三個獨立決策**的組合：

| 決策 | 白話 | 老師派誰來守 |
|---|---|---|
| **① 幾位元** | 這塊 KV 要用 16/8/4 bit 存？ | **AdaptCache** |
| **② 放哪裡** | GPU HBM / CPU DRAM / SSD / 丟掉？ | **Strata**（機制）、**Bidaw**（逐出）、**MTDS** |
| **③ 算還是載** | 丟掉重算，還是從慢速階搬回來？ | **Cake** |

**老師的訊息是：你六態全包，但每一格都已經有人在做，而且比你做得深。
你必須先說清楚「他們各自在哪裡做不到」，Tiara 才有存在的理由。**

至於 **vLLM**，老師說得很直白：它是你的**底座**不是對手。
比贏 vLLM 內建的 LRU/ARC 只證明「我的策略比最陽春的策略好」，那不是研究貢獻。

---

## 1. Cake（ICML 2025 / PMLR v267）— 決定「算還是載」

> **論文**：Compute or Load KV Cache? Why Not Both?
> **你 refs.bib 已經有了**：`cake2025`

### 1.1 它在解什麼問題（白話）

情境：使用者送來一個很長的 prompt，而這段 prompt 的 KV cache **之前算過、存在硬碟裡**。
現在有兩條路：

- **路 A（載）**：從硬碟把 KV 搬回 GPU。慢在 I/O 頻寬。
- **路 B（算）**：直接重新 prefill 一次。慢在 GPU 算力。

以前的系統是**二選一**。Cake 的洞見是：**這兩條路用的是不同的硬體資源**
（一個吃 PCIe/硬碟，一個吃 GPU 的 SM），所以**可以同時跑，把兩邊的頻寬都吃滿**。

### 1.2 機制：雙指標對撞（這是它最漂亮也最脆弱的地方）

```
compute_ptr = 0                  ← 一個執行緒從「最前面的 token」開始往後算
io_ptr = total_tokens - 1        ← 另一個執行緒從「最後面的 token」開始往前載
while compute_ptr < io_ptr:      ← 兩邊同時跑
    ...
兩個指標相遇或交錯時結束
```

**為什麼是「前面用算的、後面用載的」？** 論文只給了一句理由：

> *"Insight: Compute cost increases for later tokens, while I/O cost remains constant regardless of token position."*

翻譯：attention 是 causal 的，**第 n 個 token 要看前面 n 個**，所以越後面的 token 重算越貴；
但從硬碟搬一個 token 的 KV，不管它在第幾位，位元組數都一樣。
→ 所以「便宜的前段用算的、昂貴的後段用載的」。

這個直覺是對的，而且很優雅。**問題出在它是無條件的。**

### 1.3 🔴 它做不到什麼 —— 這是老師要你找的東西

**(a) 沒有成本模型、沒有估計器、沒有 fallback**

全文統計（PMLR 正式版 PDF）：
`placement=0`、`admission=0`、`hit rate=0`、`working set=0`、`LRU=0`、`replacement polic*=0`

chunk 大小是**離線調好的常數**（COMP 512、FETCH 128），
唯一的 runtime 訊號是一個**二元的 residency check**（「下一塊搬回來了沒」）。

**(b) 因此在資源不平衡時，它會輸給「什麼都不做」**

Table 6（2×A100，16k seq，chunk 512）有四個 sub-1.0× 的格子：**0.99、0.98、0.91、0.80**。
最糟的那個：

> **Llama-3.1-70B @ GPU 利用率 12.5% / I/O 32 Gbps → 0.80× vs I/O-only
> 也就是 TTFT 比「單純去載」還高 25%。**

Table 5 還有第五個（0.99× vs compute-only）。

**原因**：當 GPU 只剩 12.5% 可用時，「算」那一路超級慢，
但雙指標還是硬把一部分工作分給它 → 整體被拖累。

**(c) 作者自己承認缺的正是那個東西**（逐字引用）：

> *"This can be mitigated by incorporating a **fallback mechanism** that dynamically adjusts resource allocation when one resource significantly outperforms the other."*
> *"we can optimize this scheduling strategy by incorporating an **estimation mechanism**. If one resource significantly outperforms the other, Cake could adaptively fall back to a single-resource mode."*

**(d) 量測路徑裡沒有任何真實儲存裝置**

全部五種「I/O 頻寬」（7/25/32/56/100 Gbps）都是**算出延遲然後暫停傳輸**模擬的：

> *"We **simulate** the chunk I/O loading process by calculating the appropriate delay time based on the chunk size and network bandwidth. The simulated storage backend is then set to **pause data transfer** until the specified delay has elapsed."*

佐證：56 Gbps = 剛好 7,000 MB/s = Samsung 980 PRO 的**規格書**數值。
§5.1 硬體段落**沒有指名任何儲存裝置**——而這是一篇主題就是儲存 I/O 的論文。

→ 沒有 SATA vs NVMe 差別、沒有裝置佇列、沒有 tail latency、沒有檔案系統效應。

**(e) 它只解「已經確定命中」之後的事**

> *"we precompute and store all requests' KV cache in advance."*

全程 100% prefix-cache 命中率。沒有 hit rate、沒有 working set、沒有任何 accuracy 指標。
壓縮只是**外生**的縮放因子：

> *"Cake treats the KV cache as data and is **agnostic to its specific representation or compression scheme**."*

### ⚠️ 引用時的精確度要求（否則會被審稿人打）

- **不可寫「Cake 沒有 scheduler」**。它**有**一個獨立的 adaptive priority scheduler
  （decode > non-prefix-cache prefill > prefix-cache prefill），只是那個不決定 compute/load 切分。
- 要寫 **"demonstrably / measured"**，不要寫 "provably"。
- 最糟那點發生在 **12.5% GPU 利用率**，是特定 regime，不是普遍失效。

### 1.4 對你的意義

**Cake 是你的主要對手，而且是唯一一個有「實測失敗點」可以打的。**
你的 κ 成本模型，字面上就是 Cake 承認自己缺的 estimator + fallback。

---

## 2. AdaptCache（SOSP 2025 **BigMem workshop**，5 頁）— 決定「幾位元」

> **論文**：AdaptCache: KV Cache Native Storage Hierarchy for Low-Delay and High-Quality LM Serving
> **你 refs.bib 已經有了**：`adaptcache2025`
> ⚠️ **這是 5 頁的 workshop 論文**，評測段落標題直接叫 "**Preliminary Results**"。

### 2.1 它在解什麼問題

它是**唯一一篇把「壓幾位元」當成決策變數**的。對每一筆 KV cache entry，它同時決定四件事：

> *"we use marginal utility gain to evaluate the effect of each design decisions —
> **compression algorithm, compression rate, cache placement and eviction**"*

也就是：用哪種壓縮法（丟 token vs 量化）、壓多少、放 DRAM 還是 SSD、要不要直接逐出。

### 2.2 機制：MCKP + greedy

**什麼是 MCKP？**
Multi-Choice Knapsack Problem（多選背包問題）。想像每個 KV entry 是一件行李，
你可以選「原尺寸帶走 / 壓一半帶走 / 壓四分之一帶走 / 不帶」，
每個選擇有不同的重量（佔空間）和價值（省多少延遲、掉多少品質）。
背包容量固定，要最大化總價值。**這是 NP-hard 的。**

AdaptCache 的做法：

> *"We adopt a **greedy** approach based on the textbook solution [Kellerer et al., 2004] that achieves Linear Programming Optimality."*
> *"For each storage tier, we will greedily decide the compression choice (to evict, store in full quality or compress each entry) that results in the **minimal marginal utility drop**."*

效用函數是**乘法的**：

```
Utility(i) = Freq(i) × ( α × Quality(i, Mi, Ri) − size(i, Mi, Ri) / Bandwidth )
```

### 2.3 🔴 它做不到什麼

**(a) 作者自己承認最佳化是次佳的**

> *"**Acknowledging that this design is greedy and not necessarily optimal**, we argue that it suffices to deliver significant delay reduction... and optimal solution is not trackable [sic] since it is NP-hard."*

⚠️ 但它**只承認最佳化器**，從來沒有討論它的**估計器**——而估計器問題更大：

**(b) 兩個決策訊號都不具預測性，而且都不是學出來的**

全文關鍵字掃描：**`learn=0`、`neural=0`、`regress=0`、`predict=0`**（唯一的 learn 命中在參考文獻標題）。

- **未來重用** = 純粹的**歷史命中頻率**（LFU 式）：
  > *"We estimate the future cache hit frequency of one KV cache entry using its **historical hit frequency**."*

  沒有冷啟動初始化、沒有 aging/decay/recency 項。
  而效用對頻率是乘法的 → **零歷史的新條目會被推向最大壓縮或直接逐出**。

- **品質-壓縮率曲線** = **離線**建的，每個資料集只抽 **10 筆**，用 **GPT-4o 產生的問題**：
  > *"It also **samples ten entries from each dataset**, using questions generated by GPT 4o, to generate the quality-compression rate curve for each compression method."*

  → 要嘛假設能遷移到沒見過的 workload，要嘛**每個資料集都要重新離線校準**。
  而且它就在它校準的那六個 LongBench 資料集上評測。**論文沒有 limitations section。**

**(c) 評測規模很小，而且沒有任何真實 trace**

> 一張 A100（100 GB DRAM、400 GB SSD，讀 1 GB/s），只有 Llama-3.1-8B-Instruct，
> 1,100 個 context 來自六個 LongBench 資料集，
> **到達時間用 Poisson 合成**（"These datasets do not contain the timestamps for request arrival"）。

全文掃描：`vLLM=0`、`SGLang=0`、`LMCache=0`（**很諷刺，作者正是 LMCache/UChicago 那組**）、
`Mooncake=0`、`ShareGPT=0`、`PCIe=0`、`NVMe=0`、`github=0`、`artifact=0`。

**沒有指名任何 serving engine、沒有 code、沒有 artifact。**

**(d) 只有 DRAM + SSD 一台機器**

`remote/network/multi-node/cluster/distributed` 全部 **= 0 次命中**。

**頭條數字**：TTFT −56% vs prefill（品質掉 15% 內）；−69% vs KIVI（等品質）；
DRAM 命中率 81/56/44/11% vs KIVI-LRU 的 38%。
Baseline 只有三個：Without-Compression / KIVI-LRU / StreamingLLM-LRU。

### 2.4 對你的意義

- ✅ **可打的點**：它的訊號是 LFU（不具預測性），你的 GBDT 是 predictive（AUC 0.917–0.922、ECE 0.003–0.004）。
- ⚠️ **但不要拿它當主要打擊對象**。它是 5 頁 workshop 的 preliminary results，
  把它當成「最強 SOTA」然後宣稱打贏，審稿人會覺得你在挑軟柿子。

---

## 3. Strata（OSDI 2026）— 多階 KV 的 **I/O 機制**天花板

> **論文**：Xie Zhiqiang et al., OSDI'26
> ⚠️ **你的 refs.bib 沒有這篇。** 需要補。

### 3.1 它在解什麼問題

前面兩篇都在講「**決策**」（選幾位元、選算還是載）。
Strata 不管決策，它管**機制**：「既然決定要搬，怎麼搬得最快」。

這是目前**唯一一個進到 production serving engine 的**（SGLang HiCache 已 upstream）。

### 3.2 機制

- **page-first layout**：改 KV cache 在記憶體裡的擺法，讓搬運時是**大塊連續**而不是散落
  （👉 **這一點跟我們昨天量到的描述符粒度問題直接相關**，見 §6.3）
- **GPU-assisted gather/scatter**：用 GPU 的算力去做搬運時的重組
- **cache-aware batch scheduling**：排程時就考慮「哪些請求的 KV 已經在哪一階」
- **HiRadixTree**：擴充 SGLang 的 RadixTree，加上階層資訊

### 3.3 🔴 它做不到什麼

**(a) 零壓縮、零量化、零 bit-width，而且是刻意的**

全文掃描 `compress|quantiz|fp8|int8|bit-width|bitwidth|8-bit|4-bit|precision|dtype|lossless|lossy`
→ **只有 2 次命中，兩次都在參考文獻**（LMDeploy 標題、CacheGen 標題）。

§6 Related Work 明文自我定位：
> *"Unlike these approximate caching schemes, **Strata does not impact the accuracy of requests**."*

**它是 lossless by design。所有頭條數字都是 TTFT 和吞吐量，沒有任何 accuracy 指標。**

→ **「量化感知的分層」（4-bit 放 SSD / 8-bit 放 DRAM / 16-bit 放 HBM）
完全在這個子領域最強 I/O 系統的設計空間之外。**

**(b) SSD 階基本上沒評測**

§5.1 逐字：
> *"Disk storage is **not used in most benchmarks except in §5.3.5** due to limited support in baseline systems."*

主測試台（8×H200）的碟 **< 1 GiB/s**。唯一的碟結果換到另一台機器：
DeepSeek-V3 on 8×H20 + Intel P5510 NVMe（7 GB/s），page size 32，12 req/s
→ 只有 **2.1× TTFT、1.3× 吞吐**，而且那是**內部消融**（page-first layout vs 更大的 page size），
不是跨階放置的比較。

**remote/object storage 完全不存在。**

**(c) 每一個記憶體階的預設逐出策略都是 LRU**

§5.1：*"For all memory layers, the **Least Recently Used (LRU)** algorithm serves as the default eviction policy"*

**(d) 自己承認的未來工作，正好是慢速階的 prefetch**

§6 Discussion 逐字：
> *"an equally promising direction is to **create scheduling headroom for hiding the latency of prefetching from slower storage tiers**, going beyond Strata's current approach of overlapping prefetches with queuing delay"*

### 3.4 評測與可重現性（這張表你寫實驗表時要用）

| 項目 | 值 |
|---|---|
| 主測試台 | **8 × H200** NVLink，Intel Sapphire Rapids，**1.6 TB DRAM**，PCIe 5.0 ×16（64 GB/s peak），1 TB pinned |
| 第二台 | 8 × H20 + Intel P5510 NVMe（~7 GB/s）← 只為了碟實驗 |
| 第三台 | GH200（H100 + Grace，464 GB LPDDR5X） |
| Baseline（版本釘死） | vLLM **v0.8.5**、LMCache **v0.2.1**、TensorRT-LLM **v0.17.0**、SGLang **v0.4.5** |
| 模型 | Llama-3.1-8B (128k)、Qwen2.5-14B-1M、Llama-3.1-70B (4-GPU TP) |
| 資料集 | LooGLE / NarrativeQA / ReviewMT / ShareGPT，**Poisson 合成到達** |
| Mooncake | 只在 §5.3.4 當**模擬**子實驗（"our testbed cannot sustain sufficiently high throughput"） |
| 頭條 | 吞吐量最高 **5×** vs vLLM-LMCache、**3.75×** vs TRT-LLM（皆在 70B） |
| Artifact | **無 badge、論文內無 repo URL**，但 code 在 sgl-project/sglang HiCache |

⚠️ **版本漂移**：arXiv v1 寫 "5× lower **TTFT**"，OSDI camera-ready 寫 "5× **throughput**"。
**引用 camera-ready。** 兩版節號也不同（碟實驗一版 §5.3.4、另一版 §5.3.5）。

### 3.5 對你的意義

**Strata 不是你的對手，是你的天花板參照。**
它證明了「機制」這條路已經被 OSDI 級的團隊做到 production。
你**不應該宣稱贏它**，而應該說「機制屬於它們，我做的是它們刻意不碰的品質-延遲取捨」。

---

## 4. Bidaw（FAST 2026）— 唯一量到「headroom 真的存在」的那篇

> **論文**：Hu Shipeng et al., FAST'26
> ⚠️ **你的 refs.bib 沒有這篇。** 需要補。**而且它其實是你的盟友。**

### 4.1 它最重要的貢獻：一個量測

Bidaw 是**這整個子領域唯一一篇量化回報「Belady oracle 相對現有策略有多少空間」的論文**。

§2.2 標題就是 "Characterizing KV Access with **Million-round** Real-world Workload"：
> *"we collect and analyze a real-world workload from our industry partner... spanning **more than one million conversation rounds**"*
> 平均每個使用者 **22.4 輪**對話（平均/中位/P90 = 22/18/45）

§2.3 逐字：
> *"as the performance layer accommodates **40.1%** of KVs on average, the hit rates are **only around 20%** with existing eviction strategies"*

三個「現有策略」= queue-enhanced（CachedAttention）、FIFO、LRU。
performance layer = 本機記憶體（評測機 200 GB）。

§3.3.2 —— **這是關鍵句**：
> *"when the weighted reuse distance exceeds the performance layer size and continues to increase,
> **the hit rate of the optimal strategy remains above zero** and declines gradually over a wide range of distances.
> We refer to this range as the **promising reuse distance**."*

翻譯：**即使工作集遠大於快取容量，Belady 最佳策略的命中率仍然 > 0，
而現有策略掉到 20%。那個差距就是 headroom。**

### 4.2 機制：用「上一輪回答多長」預測，拿 Belady ghost cache 線上校準

- **預測訊號**：上一輪模型回答的長度。
  §3.3.1：*"Across the 12 groups, the Spearman coefficient ranges from **0.94 to 0.98**"*
  （與「最小加權重用距離」的相關性）
- **校準方式**：§3.3.3 *"we maintain a **ghost cache** adopting the optimal eviction strategy in background"*
  ——背景跑一個假想的 Belady 快取，拿它來校準真實策略。
  開銷：Bidaw 0.35 ms vs 直接在 ghost cache 跑最佳策略 2.86 ms。

**什麼是 ghost cache？** 一個不存實際資料、只存「如果我用策略 X，現在會命中什麼」的影子結構。
LRB / ARC 也用類似技巧。

**成果**：miss rate 相對 queue-enhanced 降 **57.6%**、相對 LFU/LRU/FIFO 降 **69.9%**。

### 4.3 🔴 它最大的弱點：trace 不公開

> **那份 >1M 輪的 trace 是專有的、未公開。**
> 唯一用到的公開 trace 是 ShareGPT（平均 5.7 輪，時間戳是 Poisson 合成的）。

而且論文**從未報告 Belady 在 40.1% 這個操作點的總命中率**，
所以那個 headroom 是被 Bidaw 自己的 57.6% 改進**下界**住的，不是被直接量出來的。

### ⚠️ 引用時的紅線

**Bidaw 的「做不到什麼」從來沒有被建立。**
兩條關於 Bidaw 排除範圍的 claim（說它不做量化、不做 remote、不做 recompute-vs-load）
**都被推翻了（0-3 和 1-2）**。

**安全的 Bidaw 那一格只能寫三件事**：
① trace 特性描述　② 答案長度預測器 + ghost cache　③ 57.6% / 69.9% 這兩個數字。

### 4.4 對你的意義

**引用它來支持「這個題目值得做」，不要把它當對手。**
它替你證明了 headroom 在多輪對話 workload 上是存在的——
而且它的 trace 不公開，正好留下「**在公開 trace 上重現 Belady-gap 量測**」這個空位給你。

---

## 5. MTDS（Complex & Intelligent Systems，Springer）— 🔴 空白列

> **你 refs.bib 已經有了**：`mtds2026`

**零條 claim 通過驗證。沒有抽到任何一手內文。**（付費牆）

驗證結論明文寫著：

> **"do not fill that row from the abstract or title."**

**矩陣目前是四篇寬，不是五篇。** 要補這一列必須真的拿到 Springer 全文。
另外它是 Springer 期刊（vs OSDI/FAST/ICML），並列時需要獨立評估評測嚴謹度。

---

## 6. 五篇合起來：決策空間是**切開的**，不是重疊的

### 6.1 一張圖看懂

```
決策 ①「幾位元」      決策 ②「放哪裡」              決策 ③「算還是載」
     │                      │                            │
 AdaptCache            Strata（機制）                   Cake
 （唯一）              Bidaw（逐出）                   （唯一）
     │                 MTDS（未知）                      │
     │                      │                            │
  greedy MCKP           LRU everywhere              雙指標對撞
  LFU 訊號              lossless                    無成本模型
  無 recompute          SSD 幾乎沒測                 無 fallback
  無 HBM                無 bit-width                 儲存全模擬
```

**沒有任何一篇同時決定三者。**

### 6.2 為什麼三者不能分開做（這是 gap 的核心論證）

```
改 bit-width → 改位元組數 → 改載入時間 → 移動 Cake 的算/載交叉點 → 改變該放哪一階
```

- **AdaptCache** 選 bit-width，但把交叉點當**固定**
- **Cake** 決定交叉點，但把 bit-width 當**外生**（"agnostic to its representation or compression scheme"）
- 另一篇獨立模擬研究更直接寫：*"placement composes with KV quantization and eviction... **we hold those fixed to isolate placement**"*

**兩篇裡有兩篇把這件事的一部分寫成自己的未來工作，沒有一篇實作。**

### 6.3 ⚠️ 但我們昨天的實驗，讓這個 gap 的價值打了折

昨天跑的 #1 實驗發現：**在純延遲層面，量化與放置是替代品不是互補品**。

| trace | 全 BF16 的 oracle headroom | 全 INT4 之後 |
|---|---|---|
| toolagent | 20.66% | **2.02%** |
| conversation | 20.91% | **3.52%** |

只把 `--kv-cache-dtype` 改成 int4、配原廠 LRU，就吃掉 oracle headroom 的 **91.8%／103.9%**。

**意思是：上面那個「三者耦合」的論證在延遲上成立，但方向跟直覺相反——
不是「一起最佳化會更好」，而是「選對 bit-width 之後，放置策略就沒剩多少可做」。**

→ **這反而是比原本的 gap 更強的 novelty**，因為它**反駁**了 AdaptCache 的前提。
→ 但也意味著：**Tiara 想有貢獻，只能靠品質約束**（不能全部丟 INT4），
而平台 B 的 ε 目前 **n=120、每個 dtype 都與零不可區分**。

另外 §3.2 提到 Strata 的 **page-first layout**——
那正是我們昨天量到的「描述符粒度決定頻寬」的同一件事。
**Strata 在 NVIDIA 上用佈局解決了它；我們量到 vLLM 在 ROCm 上沒解決（17.1× 跨模型差異）。**
這條線索需要先確認 Strata 有沒有報告過同類數字。

---

## 7. ⛔ 被推翻的 13 條 — 不可引用

這些聽起來很有用，但沒撐過 3 票驗證。寫進論文會被抓。

| claim | 票數 |
|---|---|
| Strata 完全沒有 eviction/admission/placement policy，只有 LRU | 1-2 |
| Strata 在 §6 明確放棄三項能力 | 0-3 |
| Strata 提供了 Mooncake 的 headroom 量測（38% 請求共享 ≥6k prefix） | 0-3 |
| Strata 的 SSD 在**所有** benchmark 都被排除 | 0-3 |
| Strata 的每個決策都是靜態手調門檻 | 1-2 |
| Strata 沒有 compute-vs-load policy | 1-2 |
| AdaptCache **證明**無法決定 HBM 或 remote | 1-2 |
| **Bidaw 證明 ShareGPT 零 headroom** | 0-3 |
| **Bidaw 的控制面刻意窄（明確排除量化）** | 1-2 |
| 73.02× 只是 1:8:64 階層形狀的記帳假象 | 0-3 |
| 跨階 prefetch 不划算 | 0-3 |
| Alibaba/Aliyun trace 命中率上限 62%/54% | 0-3 |
| 放置策略的延遲 headroom（某條敘述） | 0-3 |

---

## 8. 你原本選的 vs 老師選的 —— 差在哪

### 8.1 先說公道話：**你的引用覆蓋率其實很好**

`refs.bib` 有 **46 筆**，而且**已經包含 AdaptCache、Cake、MTDS**。
你也引了經典快取理論（Belady 1966、LRB、Parrot、Mockingjay、Sibyl、Pythia）——
這在 LLM serving 論文裡不常見，是加分項。

**你缺的只有兩篇：Strata（OSDI'26）和 Bidaw（FAST'26）**，兩篇都非常新。

### 8.2 真正的問題不在引用，在**baseline**

你的 `main.tex` §1084 寫的 baseline 是：

> `full_gpu`（vLLM 預設）、`cpu_lru`、`cpu_arc`（vLLM 內建兩個策略）、`tier_fs`、LMCache

然後：

> *「學術與學習式對照（InfiniGen、KIVI、KVP、ForesightKV、LookaheadKV、稀疏檢索一系）**尚未執行***」

**所以：你把 AdaptCache 和 Cake 寫進 related work，但從來沒有跟它們比過。**

老師的話翻成白話就是：

> 「你引了正確的論文，但你打的是 vLLM 內建的 LRU 和 ARC。
> 那只證明你比最陽春的策略好。**你 related work 裡那幾篇，才是你真正要打的人。**」

### 8.3 你排除學習式對照的理由：**站得住，但要改寫**

`main.tex` §1440：

> **學習式對照組未釋出權重。** ... 我們因此**不**以外部學習式系統作為核心主張的檢驗手段。
> 本文的主張是「對稱損失在異質動作成本下失準」，而檢驗該主張的正確方式是**在同一系統內替換損失函數**。

**這個論證本身是好的**（同系統內消融消除無關差異，方法學上正確）。
KVP / ForesightKV / LookaheadKV 確實沒釋出權重，單卡也跑不動——
這在 `EXPERIMENT_PLAN.md:486` 已經記錄。

**但它不能延伸到 AdaptCache 和 Cake。** 因為：

| 對照組 | 為什麼可以不跑 | 為什麼**不能**不跑 |
|---|---|---|
| KVP / ForesightKV / LookaheadKV | ✅ 權重未釋出、單卡不可行 | — |
| **Cake** | — | 🔴 **它是決策層的直接對手**。而且它的儲存是模擬的，你有真裝置量測，這反而是你的優勢 |
| **AdaptCache** | 🟡 無 code，要自己實作 | 🔴 但它是唯一同做 bit-width 的，至少要在紙上逐項對照 |
| **Strata** | ✅ 需要 SGLang + 8 卡，你只有 1 卡 | 🟡 可以誠實寫「設備不足，未測」——**這是允許的**，老師說過 |
| **Bidaw** | ✅ 關鍵 trace 專有 | 🟡 引用即可，不需要打 |

### 8.4 一句話總結

> **不是你選錯論文，是你選對了論文卻沒跟它們比。
> 而你排除學習式 baseline 的理由是對的，但那個理由被錯誤地延伸到了 Cake 和 AdaptCache。**

---

## 9. 你 refs.bib 裡還有 **12 篇高風險論文**，必須優先讀

這些是你**自己引了、但從沒細讀**的，而且標題直接撞上你的貢獻宣稱。
**在跟老師報告之前必須確認它們沒有已經做掉你的東西。**

| 論文 | 為什麼高風險 |
|---|---|
| 🔴 **EvicPress**（arXiv 2512.14946） | 標題直接是 "**Joint** KV-Cache Compression **and** Eviction" ——**正是 §6.2 那個 gap** |
| 🔴 **Understanding Bottlenecks...KV Offloading**（arXiv 2601.19910, **MLSys 2026**） | 標題直接是「KV offloading 的瓶頸分析」——**正是我們昨天 #2 的發現** |
| 🔴 **KVPR**（ACL Findings 2025） | "I/O-Aware KV Cache **Partial Recomputation**" ——**跟 Cake 同一格，可能更細緻** |
| 🟡 **KVDrive**（arXiv 2605.18071） | "**Holistic** Multi-Tier KV Cache Management" |
| 🟡 **Tutti**（arXiv 2605.03375） | "Making **SSD-Backed** KV Cache Practical" —— 你的 SSD 階結論 |
| 🟡 **HetMem**（IEEE CAL 2025） | "Dynamic KV Cache **Placement** in **Heterogeneous Memory**" |
| 🟡 **KVTuner**（**ICML 2025**） | "Sensitivity-Aware **Layer-Wise Mixed-Precision** KV Cache" —— bit-width 決策 |
| 🟡 **OrbitFlow**（**PVLDB 2026**） | "Fine-Grained KV Cache **Reconfiguration**" |
| 🟡 **HCache**（**EuroSys 2025**） | "Fast State Restoration" —— 重算 vs 載入的另一種解法 |
| 🟡 **KVServe**（**SIGCOMM 2026**） | "Service-Aware KV Cache **Compression**" |
| 🟢 **YaKV**（Yandex，arXiv 2604.08426） | **code 公開** —— 可能是你唯一跑得起來的外部 baseline |
| 🟢 **LeoAM**（arXiv 2506.20187） | "Adaptive KV Management on a **Single Commodity GPU**" —— 跟你設備條件相同 |

> ✅ **查證已完成（2026-09-20）。完整結果在 → [`PAPERS_EXPLAINED_20260920.md`](PAPERS_EXPLAINED_20260920.md)**
> （99 個 agent、18 份來源、25 條 claim 對抗式驗證）

### 9.1 查證結論摘要

| 發現 | 判決 |
|---|---|
| **(A) 量化與放置是替代品** | 🟢 **存活**，但要改寫成更窄的宣稱 |
| **(B) 有效傳輸頻寬是實作性質** | 🟡 **一半被搶走** |

**(A) 存活的理由**：**沒有任何一篇論文有 Belady／offline-optimal oracle**
（5 篇全文關鍵字掃描 `belady=0`），所以沒有人能量「headroom 隨 bit-width 怎麼變」。
而且整個領域把「正交」當成**未經檢驗的背景假設**寫出來：

- arXiv:2609.16215 §II-A：*"placement is **orthogonal and complementary**"*，Limitations：*"we hold those fixed to isolate placement"*
- KVTuner：*"KV cache quantization is **orthogonal** to most other KV cache management"*
- EvicPress 摘要：*"prior work misses an important opportunity: **jointly** optimizing the eviction and compression decisions"*

→ **它們是你的靶子（foils），不是先行工作（prior art）。**

**(B) 被搶走的部分**：**MLSys 2026**（arXiv:2601.19910）§6.2 逐字：
> *"We measure sustained PCIe bandwidth of **15 GB/s (23% of unidirectional 64 GB/s peak)**"*，
> 歸因於 *"CPU-GPU memory copy overheads, NUMA effects, and **transfer granularity**"*。
> 而且它還推導了 **κ_crit = (F_pf/B_kv) × (BW_PCIe/C_eff)** ← ⚠️ **與你論文的 κ 記號衝突，要改名**

### 9.2 三個新出現的高危來源（都不在你原本的清單裡）

| 來源 | 日期 | 對你的傷害 |
|---|---|---|
| **arXiv:2609.16215**「Where Should the KV Cache Live?」 | **6 天前** | 獨立發表了「placement policy 價值很低」，但歸因於階層容量比 1:8:64 與 batch-1 → **「放置是二階的」不再 novel，只剩「headroom 隨 bit-width 崩塌」** |
| **SGLang PR #40278** | **1 天前** | 同結構宣稱，但機制相反（那裡 DMA copy engine 是被追趕的**快**天花板） |
| **SGLang PR #37701/#37635**（ROCm, MI355X） | **17 天前** | 🔴 **已查證**：`block_quota` 2→8 讓 host→device 從 **24 → 55 GB/s**。**「AMD 上頻寬是軟體性質」這個論點已被搶走**，但機制是 workgroup residency 不是 DMA 描述符 → 你的機制、幅度（17.1× vs 2.2×）、跨模型變異、粒度曲線都還活著 |

### 9.3 兩個必須立刻做的事

1. 🔴 **讀 EvicPress 全文**（arXiv:2512.14946，AdaptCache 同一組人）。
   要確認三件事：(1) 它的 "compression" 是不是字面上的 bit-width；(2) 有沒有任何 optimal/上界 baseline；
   (3) **有沒有一個「只壓縮 + LRU」的對照臂，它離 EvicPress 本體多近**。
   **若那一臂很接近，(A) 等於已經被同一組人發表了。**
2. 🔴 **QEvict**（arXiv:2608.05326）**在你自己的 bib 裡但沒進這次的 12 篇清單**，
   標題字面就是「Recoverable **Quantized** KV **Eviction**」= 量化 × 逐出。**必讀。**

---

## 10. 目前 SOTA 的 gap，以及 Tiara 站在哪

### 10.1 驗證過的 gap（high confidence）

> **一個明確的成本模型，同時仲裁「重算 vs 載入」「放哪一階」「幾位元」，
> 在某資源明顯佔優時有 fallback，且由 predictive（非歷史）訊號驅動，
> 在公開的、有時間戳的多輪 trace 上驗證，並回報執行間變異。**

### 10.2 Tiara 目前佔了多少

| gap 的組成 | Tiara |
|---|---|
| **真機量測的成本模型** | ✅ **最強的一塊**（κ 表，兩平台實測） |
| 仲裁 recompute-vs-load | ✅ 模擬器內已實作 |
| 仲裁 tier placement | ✅ 模擬器內已實作 |
| **仲裁 bit-width** | 🟡 只有 INT8、只有平台 A；FP8/INT4 從未進入策略 |
| 資源不平衡時的 fallback | ❌ 未實作 |
| **predictive 訊號** | ✅ GBDT，AUC 0.917–0.922 |
| **在真實 serving engine 上跑** | ❌ **從未**，全部在模擬器裡 |
| 公開 trace | ✅ Mooncake |
| **回報變異** | ❌ → 但昨天做了（見下） |

### 10.3 三個對你特別有利的方法學發現

1. **⭐ 回報變異數會「超越」而非只是符合現行水準。**
   驗證結論：這五篇**沒有任何一篇**回報 seed、變異、CI 或 A/A。
   *"a new paper that reports variance would **exceed, not merely match**, current practice."*
   → 我們昨天已經做了：headroom = **20.6% (95% CI 20.2–23.8, n=60 bootstrap)**。

2. **⭐ 模擬器在頂會是可接受的。**
   ICML'25 接受 Cake **整條儲存路徑都是解析式模擬**；
   OSDI'26 接受 Strata 的 Mooncake 模擬子實驗。
   要避免的反例是「模擬器的校準模型本身沒有驗證」——
   **我們的 `simulator_validation.json`（4 個模型中 3 個在 14–29% 內吻合）正是那個缺口的答案。**

3. **⭐ 「公開 trace 有沒有 headroom」是公開的未解問題。**
   Bidaw 是唯一的正證據但 trace 專有；近零 headroom 的證據只有 preprint 等級，
   而且粒度不同（block-hash prefix vs 每使用者對話）。
   驗證報告明文說：在可釋出的公開 trace 上重現 Belady-gap 量測，**本身就是一個可獨立成立的貢獻**。

### 10.4 所以「Tiara 要打誰」

| 系統 | 定位 |
|---|---|
| **vLLM** | ❌ **不是對手，是底座**。老師說得對 |
| **Cake** | ✅ **主要對手**。5 個實測失敗點 + 作者自承缺 estimator/fallback |
| **AdaptCache** | 🟡 次要。唯一同做 bit-width，但只是 5 頁 workshop，不能當主要打擊對象 |
| **Strata** | ❌ 非直接對手（lossless、無 bit-width）。**引用它，不要宣稱贏它** |
| **Bidaw** | ❌ **是盟友**。引用它證明 headroom 存在 |
| **MTDS** | ❓ 資訊不足，空白列 |
| **EvicPress / bottlenecks2026 / KVPR** | ⏳ **查證中，可能是真正的對手** |

---

## 附錄：來源與驗證方式

- 一手 PDF 全文抓取：OSDI'26 Strata、FAST'26 Bidaw、PMLR v267 Cake、arXiv AdaptCache
- 驗證：每條 claim 由 3 個獨立 agent 對抗式投票，需 2/3 反駁才推翻
- **來源不對稱**：Strata / Bidaw / Cake 是頂會同儕審查的一手 PDF；
  **AdaptCache 是 5 頁 SOSP'25 BigMem workshop**，評測段落自題 "Preliminary Results"，數字要相應降權
- **未驗證**：LMCache 本體、CacheGen、CacheBlend、Mooncake、Pensieve、InfiniGen、FlexGen、
  AttentionStore、LayerKV、ALISA、KVPress、RadixAttention —— 本輪只看到 Strata 對它們的一句定位
- **時效**：Strata 釘住的 baseline 版本到 2026-09 已過時數月，任何重測都不會重現那些比值
