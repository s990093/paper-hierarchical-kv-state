# 相關論文更新：Cake／Pensieve 重做核對清單、寫入路徑比較、新論文查新（2026-10-08）

> **一句話**：Cake 和 Pensieve 原文都**完全沒有寫入策略**（Cake 假設全部事先存好；Pensieve 的寫入是 GPU 快滿時才換出）。我們的還原和逐出可以照原文重做，但有 6 個地方原文沒寫，或我們刻意改了，要在 RUNLOG 寫明。2026-06 到 10 月新出的論文裡，**「寫入時就決定放哪一層」已經有人做**（Lachesis，依壽命決定；EfficientAgent，依容量決定要不要寫）。但「依 token 位置／重算成本、無損、配合雙向還原」的組合，在本次查到的範圍內還沒有人做（不代表一定沒有，見 §c 末）。

**出處與證據等級**
- 本地 PDF：`/mlsteam/data/tiara/papers/`。用 PyMuPDF 抽文字（本機沒有 `pdftotext`），頁碼是 **PDF 的實體頁**。Cake 用 PMLR 版（`02_Cake_ICML25_PMLR_v267.pdf`）；arXiv 版 2410.03065 的內容與頁碼對照過，關鍵句一致。
- Pensieve 本地沒有。從 arXiv 2312.05516 下載 PDF，頁首是「EuroSys '25」，所以是 camera-ready 版，頁碼以那份 PDF 為準。
- §c 的論文都**抓過 arXiv abs 頁**確認存在（標 ✅）。一句話摘要和威脅評估**只根據 abstract**，沒讀全文的會註明。沒抓到的標 `NOT_VERIFIED`。
- 本檔**沒有任何實測數字**。引用的數字都是**原論文自己報的**，只用來說明對方的主張，不是我們的結果（CLAUDE.md §1-1）。

---

## (a) Cake／Pensieve 照原文重做的核對清單

### a1. Cake（ICML 2025，PMLR 267；作者 Jin, Liu, Zhang, Mao）

| 項目 | 原文怎麼寫（頁） | 我們（`03`、`04`、`05`） | 判定 |
|:--|:--|:--|:--|
| 適用情境 | 從**容量大、頻寬低**的層載入，例如本地 SSD、遠端儲存（p3 §4 開頭；p2 Fig.1） | CPU 層和 SSD 層都用 Cake 還原 | **部分偏離**：Cake 沒有評估「從 CPU 層」用雙向還原。我們把它套到 CPU 層是延伸，要註明 |
| 寫入／存法 | 「we precompute and store all requests' KV cache in advance」（p5 Datasets）。沒有寫入路徑、沒有逐出、只有一層 | S0＝全部寫 SSD；其他策略只存一部分 | S0 **符合**。其他策略是延伸 |
| 「沒存到的 chunk」 | 沒提到（全部都有存） | 演算法 2 第 9 行：載入線遇到沒存的 chunk 就結束 | **我們加的**。Cake 沒有這個情況，要寫明是延伸 |
| 會合點怎麼決定 | 不是事先算的。兩條線平行跑，「merging point … shifts accordingly」（p5 Part 3）；Fig.5 說會「automatically identifies the optimal merging point」（p8）。p7 說還沒有估計機制，列為 future work：「incorporating an estimation mechanism … fall back to a single-resource mode」 | 還原時動態決定（演算法 2）**符合**。寫入時另外**預估** b（演算法 1） | 還原端**符合**。演算法 1 是**我們加的**，Cake 自己說沒有估計器（p7）。這點對 S5 是支持證據 |
| 兩條線同步的細節 | 沒寫。只說每一步「lightweight check to determine whether the next chunk has been fetched」（p9 §5.9） | 兩個指標加鎖 | **原文沒寫**，我們自己決定 |
| 計算端 chunk | vLLM chunked prefill，token budget 預設 512（p5 Baselines）。Part 1 用 512 token（p3），Fig.4 每 chunk 512（p4） | 512 | **符合** |
| I/O 端 chunk | 「We choose the I/O loading part chunk size as 128 tokens」（p5–6） | 512 | **偏離（刻意）**。`04` §4 已經寫了。建議 A0 加跑一組 128 的 I/O chunk 對照 |
| 頻寬設定 | 7、25、32、56、100 Gbps（p6 Table 2）。波動實驗在 0–25 Gbps 間隨機取樣（p8） | 換算成 GiB/s，A0 掃頻寬 | **符合**，但單位有歧義（Gbps 是 bit 還是 byte，原文沒說清楚；Fig.3 又用 GB/s）。換算方式要寫在 RUNLOG |
| I/O 怎麼模擬 | 依 chunk 大小和頻寬算延遲，「pause data transfer to GPU memory until the specified delay has elapsed」（p5） | 演算法 4：延遲＋**每次 I/O 固定開銷 c**＋真實 PCIe 複製，取兩者較慢的；讀寫共用 t空閒 | **部分偏離**：c、讀寫互相干擾是我們加的。Cake 沒寫資料原本放在哪裡（CPU 記憶體？），也沒寫有沒有真的複製 → **原文沒寫** |
| 「GPU 忙」怎麼模擬 | 用 token budget 的佔比：512 裡 Cake 用 256 就是 50%（p6）。點：12.5／50／87.5／100% | 用背景負載，報實測的變慢倍數 | **偏離（刻意）**。Cake 用的是**分給它的 budget 比例**，不是真的有別人在用 GPU。建議至少一組照 Cake 的 budget 定義跑，才能對照 |
| 模型 | LongAlpaca-7B／13B（MHA）、Llama-3.1-8B、Llama-3.1-70B（FP8 權重）；KV 用 BF16（p5 Table 1） | 一個模型（Llama-3.1-8B），BF16 | **符合**（Llama-3.1-8B 在 Cake 的 Table 6 裡）。但 Cake 主要的表（Table 3／4／5／7）都是 LongAlpaca-13B |
| 長度 | 主實驗 4K–16K，每 2K 一點（p5）；Table 3 固定 16K；動機實驗 32K（p3） | 4K／8K／16K／32K | **符合**，32K 是延伸 |
| 資料 | 「only token length matters」，用合成 prompt（p5） | 只量時間 | **符合** |
| 硬體 | 2×A100（NVLink，TP）、1×A100、1×H100（p5） | 1×MI300X（AMD） | **偏離**：平台不同，原文的數字不能拿來直接比 |
| 實作基礎 | 建在 vLLM 上（p3、p8 §5.9）。基線：vLLM v0.6.2 只算、LMCache v0.1.4 只載（p5）。原文**沒有附自己的程式碼連結**（全文只有 LMCache 的 GitHub） | 自己寫 harness，不改 vLLM | **偏離**。`README` 要問老師的第 1 題 |
| 指標 | 只有 TTFT（p5）；報「對只載／對只算的加速倍數」 | TTFT 的中位數／平均／P90＋95% CI | 我們報得比較多 |
| 平均怎麼算 | 拿掉紅色極端值（對某一方超過 10 倍）再平均（p6） | 不拿掉 | 要註明。不要拿我們的平均去和 Cake 的「2.6×」比 |
| 重複次數、誤差棒 | 沒寫 | 每格 ≥6 次，附 CI | **原文沒寫** |
| 調適排程 | 優先順序：decode → 非前綴的 prefill → Cake 的 prefill（p5）。實驗：16K 加 22 個 32–448 長度的請求（p8） | 第一階段沒有做 | **沒做**。要註明 S* 都沒有 Cake 的調適排程 |
| 「新一輪的 query」 | TTFT＝「loading stored KV cache or computing new KV cache」（p5）。**沒寫**是否包含快取以外的新 token | 演算法 2 第 12 行：算 256 個新 token | **原文沒寫**，我們自己決定 |

**原論文自己報、跟我們最相關的一點**：Table 6（p8，2×A100、16K）裡，Llama-3.1-8B（GQA）在 12.5% util、32 Gbps 時，Cake 對只載的加速是 **0.98**（比只載還慢）；70B 是 0.80。原文的解釋是：GQA 讓 KV 變小，I/O 相對變快（p7 §5.5）。我們用 GQA 模型，而且 MI300X 算力很強，Cake 在「計算」那一側能拿到的好處可能很小，S5 的「前段給重算」空間也會跟著變小。**這是原文的數字，不是我們的。我們實際會怎樣，要等 A0／C1 量完（NOT_MEASURED）。**

### a2. Pensieve（EuroSys 2025；arXiv 2312.05516；作者 Yu, Lin, Li）

| 項目 | 原文怎麼寫（頁） | 我們（S2b、S4+） | 判定 |
|:--|:--|:--|:--|
| 層 | 只有 GPU＋CPU 兩層；CPU 滿了就**丟掉**，之後重算（p4、p6）。**沒有 SSD** | CPU＋SSD | **偏離**：我們把它對應到「CPU→SSD 降級」「SSD→刪掉」，是延伸 |
| 保留值 | V＝Cost(s,l)／T。Cost 是在長度 l 的 context 上重算一個大小 s 的 chunk 的成本；T 是這個對話**閒置多久**。依 V 由小到大逐出（p6 §4.3.1） | `03` §1：「最便宜＝位置最前面；同樣便宜時，選最久沒用的 session」 | **偏離（重要）**。我們是「先比位置、再比 LRU」的**字典序**；Pensieve 是**比值**，閒置很久的 session，後段也可能被逐出。**建議加一個照原文的變體 S2b-P／S4+-P（用 V＝f(i)／T）**，不然審稿人會說「你的成本感知基線不是 Pensieve」 |
| 成本模型 | Cost(l)＝Cost_attention(l)＋c。離線量 context 長度是 2 的次方的點，其他長度用內插（p6） | f(i) 由 C1 實測 | **符合精神**。內插法可以照抄 |
| 粒度 | chunk＝**32 token**（p6） | 512 token | **偏離**，要註明 |
| 寫入時機 | GPU 的空位低於門檻（例：25%）時，**提前**把選中的 KV 複製到 CPU；GPU 的空間延後才回收（p6 §4.3.2）。這是**由容量觸發的延後寫入**，不是寫穿 | S4／S4+＝只寫 CPU，滿了才降級 | 性質相近，Pensieve 是「GPU→CPU」，我們是「CPU→SSD」 |
| 還原 | 前面被丟的 → 重算；中間在 CPU 的 → 換入；最後在 GPU 的 → 直接用（p4 Fig.5、p7 §4.3.4）。換入和計算逐層 pipeline（p6–7） | Cake 雙向還原 | 不同。但 Pensieve 已經形成「**前段重算、後段載入**」的版面，**而且是逐出時做出來的**。這正是 S4+ 想代表的對手 |
| 讀寫干擾 | GPU↔CPU 兩個方向同時傳時，兩邊的吞吐都掉 18–20%。所以有換入在跑時，換出要等（p9） | 演算法 4 讓讀寫共用 t空閒 | **符合精神**。要不要加「讀優先」，在 RUNLOG 寫明 |
| 負載 | ShareGPT、UltraChat；請求到達用 Poisson；思考時間是指數分布，預設平均 60 s，另掃 30／60／120／300／600 s（p9、p12 Fig.15） | B 實驗先用事件順序 | `04` 的描述**正確**（已核對） |
| 原論文自己報的成本感知效果 | 和 LRU 比，CPU 命中率最多高 4.4 個百分點，重算的 token 最多少 14.6%（p12） | — | 只是對方的主張。代表「從前段逐出」在它的設定下好處有限，可以作為 S4+ 的先驗參考 |
| 程式碼 | 原文說約 7K 行 C++／CUDA（p9），沒附連結 | 照描述重做 | 和 `04` 卡片的查證一致 |

---

## (b) 最接近的系統：寫入路徑怎麼做

| 系統（出處） | 寫入時機 | 准入（要不要寫） | 逐出／降級 | 是否依位置 | 無損 | 頁 |
|:--|:--|:--|:--|:--|:--|:--|
| **Cake**（ICML'25） | 不處理，全部事先存好 | 無 | 無 | 還原時依位置（前算後載） | 是 | p5 |
| **Pensieve**（EuroSys'25） | GPU 空位 <門檻時才換出到 CPU（**延後寫**） | 無 | V＝Cost／T；CPU 滿了就丟 | **逐出時**依位置（從開頭丟） | 是 | p6–7 |
| **CachedAttention／AttentionStore**（ATC'24） | 每輪 HBM→DRAM **非同步存**，逐層和計算重疊，HBM 留 write buffer；全部請求都存（p1、p6 §3.2.2） | 無 | DRAM 滿了 → 依 job queue 的 look-ahead window **逐出到 SSD**（DRAM→SSD 是延後寫）；以 session 為單位（p7） | 否 | 是（另有去除位置編碼後的截斷） | p6–7 |
| **HCache**（EuroSys'25） | 生成時就存 hidden state，兩段式（先一次 cudaMemcpy 到 host，再由 daemon 寫到 SSD），存 SSD（p8 §4.2.2） | 無 | 原文沒寫 | **依層**（寫入時，用離線量的速度決定 L_H／L_O，p7）。**token-wise 切法有討論過，但因為 GEMM 形狀不規則會產生 bubble 而沒採用**（p6–7） | 是 | p6–8 |
| **Strata**（OSDI'26） | 可設定三種：write-back（快被逐出時才存）、write-through（每次都存）、**selective write-through（預設：存取次數超過門檻才存，預設 2）**（p9） | 有（存取次數門檻） | 各層都用 LRU（p9） | 否 | 是 | p9 |
| **Bidaw**（FAST'26） | 計算時就把 KV（MHA 模型存 storage-efficient tensor）寫進儲存；**inclusive caching**：容量層一定有一份，所以逐出時不用再寫（p6、p10）＝寫穿 | 無；GQA 模型直接存 KV（p10） | 用前一輪回答長度預測下次存取，記憶體→SSD（p8–9） | 否 | 是 | p6, p8–10 |
| **Tutti**（arXiv 2605.03375） | **GPU 逐出時**才經 P2P DMA 寫到 NVMe，寫完才登記 metadata（p8） | 無 | 交給 Mooncake | 否 | 是 | p8 |
| **EvicPress**（arXiv 2512.14946） | 新 KV 先存 CPU，接著 `manage` 依效用函數決定（裝置、壓縮法、壓縮率），**以整段 context 為單位，寫入時就決定**（p7 §5） | 隱含在效用函數裡 | 裝置滿了用貪婪的 multi-choice knapsack 降級或壓縮（p7） | 否 | **否**（有損） | p6–7 |
| **AdaptCache**（arXiv 2509.00105） | 存入時決定壓縮法、壓縮率、放哪個裝置（p1–2） | 同上 | 依邊際效用下降貪婪選擇（p2） | 否 | **否** | p1–2 |
| **LMCache**（arXiv 2510.09665） | store 只存還沒存過的新 token（p5）；有 `batched_admit`／`batched_evict` 介面（p9） | 有介面，論文沒有給策略 | — | 否 | 是 | p5, p9 |
| **vLLM tiered offloading**（官方部落格，2026-09-10，✅ 已抓） | 先 DMA 到 host，再**同時**送到所有次級層（行為上是寫穿）；host 滿了才 LRU／ARC 逐出，逐出不會觸發寫入 | 文章沒提 | LRU／ARC | 否 | 是 | 部落格 |
| **Mooncake Store**、**SGLang HiCache** | 依 `04` 和卡片 E02：Mooncake 記憶體逐出時才寫 SSD；HiCache 可選 write_through／selective／write_back | — | — | 否 | 是 | 卡片 E02（**本次沒有重新核對**） |

**怎麼讀這張表**：
- 三種寫入策略（寫穿、延後寫、選擇性寫）在業界和學界**都已經有完整的實作**。Strata 甚至把三種都做成可以設定。所以 S1、S4 不是假想的對手。
- 「依位置」只出現在**還原時**（Cake、CacheFlow）或**逐出時**（Pensieve、AsymCache）。
- 在寫入時決定「存什麼／放哪」的有 HCache（依層）、EvicPress／AdaptCache（依整段 context，有損）、Lachesis（依壽命，見 §c）。但**都不是依 token 位置**。

---

## (c) 新論文和 `03_paper_map.md` 漏掉的論文

威脅程度指的是對 S5 新穎性的威脅。S5 的新穎性＝「**寫入時**決定、**依位置／重算成本**、**無損**、**多層**」。

| 論文 | arXiv／出處 | 日期 | 核心做法（一句話） | 對 S5 的影響 | 威脅 | 查證 |
|:--|:--|:--|:--|:--|:--|:--|
| **Lachesis**: Lifetime-Aware KV Cache Placement for Agent Serving across HBM and HBF | 2610.08378 | 2026-10-06 | **寫入時**依每段 context 的**壽命**決定放 HBM 或 HBF，目的是減少寫到 flash 的量、延長 HBF 壽命 | 直接拿走「寫入時依段決定放哪一層」這個說法。但它依的是壽命，不是位置或重算成本，沒有重算，目標是 endurance。S5 必須把新穎點收窄到「**依重算成本／位置，配合雙向還原**」 | **中** | ✅ abs |
| **EfficientAgent**: What Makes KV Cache Offloading Work for Concurrent Agents? | 2609.33762 | 2026-09-27 | 用 stack-distance 模型估 host 層要多大。層太小時**停止寫入**大批被逐出的 context（寫入准入），只延長還在快取裡的前綴 | 是「寫入准入」的直接前例，也支持「寫了沒被讀就是浪費」這個動機。但它決定的是寫或不寫，不是依位置放哪一層；而且要求從前綴開始連續覆蓋（和 Cake 允許「後段有、前段沒有」相反） | **中** | ✅ abs |
| **CacheFlow**（v2 改名：…via Automated 3D-Parallel KV Cache Restoration） | 2604.25080 | v1 2026-04-28；**v2 2026-09-26** | token×layer 的階梯式切分，加上 batch 感知的 DP，跨 GPU 還原 | 已經在 map 裡。v2 是新版：還原端的 SOTA 已經不是 Cake。S5 用 Cake 推出來的 b，換成 CacheFlow 還原時可能不準。abstract 沒提寫入路徑 | 低（新穎性）／**中**（基線會過時） | ✅ abs |
| **Where Should the KV Cache Live?** | 2609.16215 | 2026-09-14 | 用模擬器比較 GPU／CPU／SSD 三層的 recency、frequency、predicted 放置。結論：好處主要來自容量，不是放置策略 | 不威脅新穎性（沒有寫入時決定、沒有重算）。但它威脅**重要性**：「策略影響不大」。已經在 DIRECTIONS_0924 裡，**但 03_paper_map 沒收** | 低（新穎）／中（重要性） | ✅ 本地 PDF p1–3 |
| Predictive Multi-Tier Memory Management for KV Cache（Ganjihal） | 2604.26968 | v2 2026-08-07 | 6 層；Bayesian 預測重用；value score「balances the cost of recomputation against the cost of storage at each tier」，超過各層的門檻才放進那一層 | 文字上最接近「依重算成本決定放哪一層」，但**沒有公式**、沒提位置、沒說是不是寫入時決定，效果多是分析推算。要引用並區分 | 低–中 | ✅ abs＋HTML |
| Enabling HBF for Generative Recommendation Serving with Write-Aware KV Cache Policy | 2609.07175 | 2026-09-07 | 用 admission-controlled LRU-K 減少寫進 flash 的量 | 准入的前例，以使用者為單位、推薦系統、和位置無關 | 低 | ✅ abs |
| Characterizing High Bandwidth Flash for LLM Serving | 2609.39131 | v2 2026-10-05 | HBM–HBF–host 階層加上 buffered cache-aware scheduling（trace 模擬） | 背景文獻。abstract 沒有寫入時放置的策略 | 低 | ✅ abs |
| LM-CXD: Bridging LLM Serving and CXL-SSDs with Chunk-Aware KV Cache Management | 2609.26828 | 2026-09-20 | CXL-SSD 以 KV chunk 為單位做 I/O、預取、逐層 pipeline | 讀取端和裝置端，沒有寫入策略 | 低 | ✅ abs |
| UNISON | 2609.09643 | 2026-09-09 | agent 的 session KV：依回訪間隔逐出，用工具等待時間做 DMA 預算（近記憶體硬體） | 以 session 為單位，在逐出／預取時決定 | 低 | ✅ abs |
| CacheTune: Adaptive KV Cache Reuse | 2605.24022 | v2 2026-10-04 | 非前綴重用：選關鍵 token 重算、其餘載入；依層的特性調重算比例 | 讀取端、**有損**（品質「接近」全部重算） | 低 | ✅ abs |
| PatchKV | 2609.26219 | v1 標 2026-08-18 | context 被編輯後，重算受影響的 block、從 CPU 還原後綴 | 讀取端、近似 | 低 | ✅ abs |
| KVBoost | 2608.21362 | 2026-05-21 | chunk 層級重用＋依偏差選擇性重算，有 int8／int4 | 有損、讀取端 | 低 | ✅ abs |
| IMPRESS（FAST'25） | usenix fast25 | 2025 | 多層前綴 KV，從 SSD 只載重要 token | 有損。在卡片 E05 裡，**03_paper_map 沒收** | 低 | 卡片 E05（本次沒有重抓） |
| Hypic（position-independent caching，hybrid attention） | 2607.01299 | — | 只在搜尋結果看到 | — | — | `NOT_VERIFIED` |

**查新的範圍**：做了 10 次 web 搜尋，題目包括寫入時／准入、依位置卸載、延後寫到 SSD、重算＋載入混合還原、Cake 的後續和原作者的新作。沒有找到 Cake 原作者在 2026 年發表、和這個題目相關的新作。**「查不到」不等於「沒有」**（CLAUDE.md §1-7）：投稿前要再用 Semantic Scholar 查「引用 Cake 的論文」清單（本次沒做）。

---

## (d) 第一階段報告的結論

1. **S5 的新穎點要收窄。** 「寫入時就決定」這件事本身已經有人做：HCache 依層、EvicPress／AdaptCache 依整段 context（有損）、Lachesis 依壽命（2610.08378，10/6 才出）、EfficientAgent 決定寫或不寫。還站得住的組合是：**依 token 位置／重算成本 × 無損 × 多層 × 和雙向還原的會合點對齊**。另外，HCache 討論過 token-wise 切法但沒有採用（p6–7），相關工作要正面回應這一點。
2. **S4+ 要照 Pensieve 的原文做，否則對比不公平。** 現在 `03` 的「最便宜」是「先比位置、再比 LRU」的字典序，Pensieve 用的是 V＝Cost／T 的比值（p6）。粒度也不同（32 對 512 token）。建議加 S4+-P（V＝f(i)／T），兩種都跑。Pensieve 的版面（前段丟、中段 CPU、後段 GPU，p7）其實就是「**逐出時做出的依位置分層**」。S4+ vs S5 這組比較，應該明確說成「同一個版面，逐出時做 vs 寫入時做」。
3. **Cake 原文有 6 處沒寫或我們偏離了，要在 RUNLOG 寫明**：I/O chunk 128→512、GPU 忙的定義（budget 比例 vs 背景負載）、每次 I/O 的固定開銷 c 和讀寫干擾、兩條線怎麼同步、沒存的 chunk 怎麼處理、TTFT 是否包含新 query。建議 A0 至少各跑一組「照 Cake 原設定」（I/O chunk 128、用 token budget 定義 util），確認我們的 harness 能重現 Cake 的**趨勢**。
4. **要注意的地方**：Cake 原文在 GQA 的 Llama-3.1-8B、低 util 時，對只載的加速 <1（Table 6，p8，原文數字）。MI300X 算力強，「前段給重算」的空間可能比 Cake 論文裡的更小，S5 和 S4+ 的差距可能主要來自寫入干擾，而不是會合點。這要等 C1／A0 量完才知道（NOT_MEASURED）。另外，2609.16215 說「好處主要來自容量，不是放置策略」，這是第一階段可能得到 <5% 的先例，報告時要把它當成合理的可能結果。
5. **還原端的基線會過時**：CacheFlow v2（2026-09-26）已經比 Cake 更強。第一階段固定用 Cake 還原沒問題，但報告要寫明「b 是用 Cake 的還原模型推出來的；還原端換成 CacheFlow 時，S5 的分界要重新推導」。這留到第二階段。

---

**附：本次用到的外部連結**
- Pensieve: https://arxiv.org/abs/2312.05516
- Lachesis: https://arxiv.org/abs/2610.08378
- EfficientAgent: https://arxiv.org/abs/2609.33762
- CacheFlow: https://arxiv.org/abs/2604.25080
- Where Should the KV Cache Live?: https://arxiv.org/abs/2609.16215
- Ganjihal（Predictive Multi-Tier）: https://arxiv.org/abs/2604.26968
- Write-Aware KV（HBF recsys）: https://arxiv.org/abs/2609.07175
- HBF characterization: https://arxiv.org/abs/2609.39131
- LM-CXD: https://arxiv.org/abs/2609.26828
- UNISON: https://arxiv.org/abs/2609.09643
- CacheTune: https://arxiv.org/abs/2605.24022
- PatchKV: https://arxiv.org/abs/2609.26219
- KVBoost: https://arxiv.org/abs/2608.21362
- vLLM tiered offloading blog: https://vllm.ai/blog/2026-09-10-tiered-kv-offloading
