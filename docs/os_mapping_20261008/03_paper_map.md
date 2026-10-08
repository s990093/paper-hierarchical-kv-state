# 03 論文地圖：每一篇在做 OS 的哪一件事

> **一句話**：把讀過的 61 篇論文與系統（評測卡 E01、E03–E08 的 55 篇，E02 的 4 個 KV 層系統（LMCache 已在 E04），以及影片的 ReKV、MuKV）依「OS 的哪個概念」重新分類。和本研究同一類（跨請求、無損、分層）的大多在做**替換**和**讀取還原**；在**寫入時**就決定的很少，而且多半是有損的或依層決定的（限評測卡涵蓋的範圍）。

**怎麼讀**
- 「一句話」照抄評測卡（已讀原文並獨立複核），出處欄是卡片代號。〔複核修正：原有 8 列（HCache、DistServe、Splitwise、Etalon、KVCache in the wild、LRB、kvpress、MuKV）是節錄、不是照抄，已改回卡片原句〕
- 「OS 概念」「何時決定」「無損或有損」是本文的分類〔判讀〕，由獨立 agent 依卡片核對過（見 README 複核紀錄）。
- 每篇只放在一個主要的組，它也碰到的其他概念寫在「也碰到」欄。

**分類用的 OS 概念**（`01` §2）：放置、替換、寫入策略（寫穿、延後寫、寫入配置、准入）、資料搬移（預取、需求分頁、逐出）、粒度，加上 KV 特有的**重算**。

**「何時決定」**：寫入時（KV 產生、存入當下）／逐出時（空間不夠時）／讀取時（請求來了、要用之前）／每一步（解碼的每一步）／離線（事先定好）。

---

## A. 跨請求、無損的 KV 快取（本研究這一類）

〔複核修正〕A 組以「做分層放置或快取管理」收錄，不是每一篇都同時「跨請求」又「無損」。依卡片，例外有：KVDrive 沒有跨請求重用、用的是有損的稀疏注意力（E04 KVDrive 卡「重用結構」「與既有整理不一致」③）；KVPR 是單一 batch 解碼時的卸載、沒有跨請求重用（E03 卡 5「重用結構」）；AdaptCache、EvicPress 靠有損壓縮（E05）；LRB、HALP 是 CDN 快取，不是 KV（E05）。

### A1. 寫入策略與准入：什麼時候寫、要不要寫

| 論文／系統 | 主要 OS 概念 | 也碰到 | 何時決定 | 一句話（評測卡） | 卡 |
|:--|:--|:--|:--|:--|:--|
| **HCache**（EuroSys'25） | 寫入策略：依層決定存 hidden state、存 KV 或不存 | 重算（用投影算回 KV） | 寫入時（依離線量的速度定死） | 不存 KV，改存每層的 hidden state（MHA 下是 KV 的一半），還原時用一次投影把 KV 算回來，並用其他方法補 pipeline 空泡 | E03 |
| **Marconi**（MLSys'25） | 准入 | 替換（FLOP-aware） | 寫入時（准入）＋逐出時 | 為 Attention＋SSM 混合模型設計前綴快取的准入與 FLOP-aware 逐出 | E05 |
| **SGLang HiCache** | 寫入策略：寫穿／選擇性／延後寫 | 替換（radix LRU 等） | 寫入時或逐出時（依設定） | （系統，見 EVAL §2.4） | E02 |
| **vLLM `OffloadingConnector`** | 准入（`store_threshold`） | 替換（LRU／ARC） | 寫入時 | （系統，見 EVAL §2.3） | E02 |
| **Dynamo KVBM** | 准入（磁碟層只寫頻率 ≥2） | — | 寫入時 | （系統，v1.5.0 起棄用；見 EVAL §2.4） | E02 |
| **Mooncake Store** | 寫入策略：延後寫（記憶體逐出時才寫 SSD） | 替換（近似 LRU） | 逐出時 | （系統，見 EVAL §2.4） | E02 |

### A2. 放置：放哪一層

| 論文／系統 | 主要 OS 概念 | 也碰到 | 何時決定 | 一句話（評測卡） | 卡 |
|:--|:--|:--|:--|:--|:--|
| **KVDrive**（arXiv 2605.18071） | 放置：依注意力重要度 | 替換（lookahead）；搬移（只抓稀疏注意力需要的） | 寫入時（prefill 結束時） | 長 context decode 把 KV 卸到 DRAM／SSD，只抓稀疏注意力需要的部分 | E04 |
| **MTDS**（CIS 2026） | 放置＋寫入（卸載優先順序） | 讀取還原（全載／部分載／不載）；替換（兩級門檻） | 寫入時、逐出時、讀取時 | 邊緣小卡上做 GPU→DRAM→SSD 三層 KV，載入前先判斷比重算快不快 | E04 |
| **AdaptCache**（SOSP'25 workshop） | 放置＋壓縮（有損） | — | 寫入時（新 entry 產生時） | 對每段 KV 決定壓縮法、壓縮率與放 DRAM 或 SSD，以提高 DRAM 命中 | E05 |
| **EvicPress**（arXiv 2512.14946） | 放置＋壓縮（有損） | 替換 | 寫入時（存入時），層滿時再調整 | 以效用函數為每段 context 同時決定壓縮法、壓縮率與存放層（GPU／CPU／SSD） | E05 |

### A3. 替換：空間不夠時誰出去

| 論文／系統 | 主要 OS 概念 | 也碰到 | 何時決定 | 一句話（評測卡） | 卡 |
|:--|:--|:--|:--|:--|:--|
| **Pensieve**（EuroSys'25） | 替換：成本感知（重算成本 ÷ 閒置時間），從對話開頭逐出 | 搬移（提前換出、換入優先）；重算 | 逐出時 | 多輪對話的 KV 存在 GPU＋CPU 兩層；空間不夠時，優先從閒置久、重算便宜的對話的開頭逐出；被丟的用重算補回 | E03 |
| **Fancy-eviction**（arXiv 2609.28870） | 替換：比較 14 種逐出法（含 LRU 本身〔複核補充，E05〕），提出成本感知與粒度原則 | — | 逐出時 | 用兩條生產 agent trace 檢驗傳統逐出法是否適用 LLM 前綴快取，並提出成本感知與粒度原則 | E05 |
| **AsymCache**（arXiv 2606.02964） | 替換：重用機率＋位置成本 | 粒度（不連續命中，MSA kernel） | 逐出時 | 在 GPU 內的逐出決策裡，同時考慮重用機率與「隨位置增加的重算成本」，並用 MSA kernel 支援不連續的命中 | E05 |
| **KVCache in the wild**（ATC'25） | 替換：workload-aware | 負載特性刻畫 | 逐出時 | 刻畫阿里雲百煉（Tongyi）兩種生產負載的 KV 重用特性，並提出 workload-aware 逐出 | E05 |
| **LARU**（arXiv 2509.20979） | 替換：學習式（GBDT），預測錯時退回 LRU | — | 逐出時 | 在 LRU 上接 GBDT 預測，偵測到預測錯就逐步縮回 LRU 的 GPU 快取 | E06 |
| **LPC**（NeurIPS'25） | 替換：學習式（預測對話會不會繼續） | — | 逐出時 | 用對話文字預測「這段對話會不會繼續」，取代 LRU 的時間戳排序 | E06 |
| **SAECache**（arXiv 2605.18825） | 替換：依 token 類型分佇列，參數線上學 | — | 逐出時 | 依 token 類型（系統提示、CoT…）與 session 結構把 block 分到不同佇列，參數線上學的前綴快取逐出 | E06 |
| **LRB**（NSDI'20，CDN） | 替換：學習式（模仿 relaxed Belady） | — | 逐出時 | 用 GBDT 模仿「relaxed Belady」（下一次存取超過 Belady 邊界就可逐出），降低 CDN 的 byte miss ratio | E05 |
| **HALP**（NSDI'23，CDN） | 替換：學習式（LRU 取候選＋NN 兩兩比較） | — | 逐出時 | 用 LRU 取 4 個候選，再用單隱藏層 NN 做兩兩比較，逐出 YouTube CDN 的 DRAM 物件 | E05 |
| **Bidaw**（FAST'26） | 替換：用上一輪回答長度預測重用距離 | 排程（I/O 感知）；寫入（包含式） | 逐出時 | 多輪聊天 KV 存 DRAM＋SSD；讓計算與儲存互相感知，減少載入阻塞與 miss | E04 |
| **CachedAttention**（ATC'24） | 替換＋預取：依工作佇列往前看 | 寫入（非同步保存） | 逐出時、讀取前 | 多輪對話的 KV 存 DRAM＋SSD 不丟，用重疊與佇列感知預取藏住載入 | E04 |

### A4. 讀取還原：需求分頁＋重算

| 論文／系統 | 主要 OS 概念 | 也碰到 | 何時決定 | 一句話（評測卡） | 卡 |
|:--|:--|:--|:--|:--|:--|
| **Cake**（ICML'25） | 需求分頁＋重算：前段重算、後段載入，動態會合 | — | 讀取時 | 長 context 的 KV 存在慢儲存層時，讓 GPU 從前往後重算、I/O 從後往前載，兩邊在中間會合 | E03 |
| **CacheFlow**（arXiv 2604.25080） | 需求分頁＋重算：token、層、GPU 三維規劃 | — | 讀取時 | 把還原拆成 token、層、GPU 三個維度並行，用批次感知的 DP 決定每個（請求, 層區塊）重算多少、載入多少 | E03 |
| **KVPR**（ACL Findings'25） | 重算＋搬移：前段傳 activation 重算、其餘傳 KV | — | 讀取時（解碼期的每一步都求一次切點；〔複核補充〕單一 batch 解碼時的卸載，沒有跨請求重用，E03 卡 5） | KV 卸載在 CPU 時，先傳前段的 activation 讓 GPU 投影出 KV，其餘 KV 同時經 PCIe 傳入 | E03 |
| **Bottlenecks**（arXiv 2601.19910） | 分析：何時從算力瓶頸變成 PCIe 瓶頸 | — | — | KV 卸載到 CPU 後，prefill 何時從算力瓶頸變成 PCIe 瓶頸；定義臨界比值 κ_crit 並實測 | E03 |
| **py-kvcache**（arXiv 2609.11744） | 讀取時的准入：載不載（損益平衡門檻，整段判斷、不看位置） | 預取（預載） | 讀取時〔複核修正：原放在 A1、寫「寫入時」。卡片：門檻在 lookup 時拒絕載入、改由 vLLM 重算；程式碼 `prepare_store` 照樣存所有新 block，門檻只用在 `_should_decline_load`（E03 卡 7「與既有整理不一致」①④、共同模式 4）。原文的主張是「也應避免儲存」（p.14），但實作只擋載入〕 | 量 vLLM 外接 KV 快取（CPU／NVMe）何時比重算快，並做一個帶預載與損益平衡門檻的 KV Offload connector | E03 |

### A5. 資料搬移：怎麼搬得快

| 論文／系統 | 主要 OS 概念 | 也碰到 | 何時決定 | 一句話（評測卡） | 卡 |
|:--|:--|:--|:--|:--|:--|
| **Strata**（OSDI'26） | 搬移：GPU 協助搬、載入感知排程 | 寫入策略可切換（WT／WB／選擇性）；粒度（page 大小、版面） | 讀取時 | 長 context 載回 KV 卡在 I/O；用 GPU 協助搬運與載入感知排程解決 | E04 |
| **Tutti**（arXiv 2605.03375） | 搬移：GPU 自己發 SSD I/O | 寫入（讀取優先、寫入等空檔） | 讀取時 | 從 NVMe 還原 KV 時把 I/O 控制交給 GPU，讓 SSD 層接近 DRAM 層 | E04 |
| **LMCache**（arXiv 2510.09665） | 搬移：chunk 批次、管線化、預取 | 寫入（寫穿）；粒度（chunk 256） | 讀取時 | 把引擎的 KV 抽出存到 CPU／磁碟／遠端，跨查詢與跨引擎共用的快取層 | E04 |
| **Mooncake**（FAST'25） | 放置與排程：全域 KV 池、以 KV 為中心排程 | 搬移（RDMA）；副本 | 讀取時（排程） | Kimi 的 P/D 分離架構，池化叢集記憶體成全域 KV 快取並以 KV 為中心排程 | E04 |

### A6. 粒度與結構：page、前綴樹、拼接

| 論文／系統 | 主要 OS 概念 | 也碰到 | 何時決定 | 一句話（評測卡） | 卡 |
|:--|:--|:--|:--|:--|:--|
| **vLLM／PagedAttention**（SOSP'23） | 分頁（KV block 就是 page） | 共享 | — | 仿照 OS 的分頁，用 PagedAttention 管理 KV，消除碎片並支援共享 | E01 |
| **SGLang／RadixAttention**（NeurIPS'24） | 索引結構（前綴樹）＋替換 | — | 逐出時 | 前端語言加上 runtime（RadixAttention、compressed FSM、API speculative execution），加速 LM 程式 | E01 |
| **CacheBlend**（EuroSys'25） | 非前綴重用（L4）＋部分重算 | 搬移（與載入 pipeline） | 讀取時 | 把多段預先算好的 KV 拼接，每層只重算一小部分高偏差 token 以補 cross-attention，並與載入 pipeline | E07 |

---

## B. 有損：把 KV 變少（單一請求內，或壓縮）

這一類丟掉或壓縮 KV，會改變模型輸出，所以要量品質。和本研究的第一階段不同類，但和第二階段（精度）有關。

| 論文 | 主要 OS 概念 | 何時決定 | 一句話（評測卡） | 卡 |
|:--|:--|:--|:--|:--|
| **StreamingLLM**（ICLR'24） | 替換：固定保留 sink＋最近視窗 | 每一步 | 只留開頭幾個 attention sink＋最近視窗，讓模型無限長串流生成 | E07 |
| **H2O**（NeurIPS'23） | 替換：累積注意力 | 每一步 | 以累積注意力保留 heavy hitter＋最近 token 的固定預算逐出策略 | E07 |
| **SnapKV**（NeurIPS'24） | 替換：prefill 後一次壓縮 | 寫入時（prefill 後） | 用 prompt 末端觀察視窗的注意力投票，prefill 後一次把 prompt KV 壓到固定大小 | E07 |
| **PyramidKV**（arXiv 2406.02069） | 替換：層間分配預算 | 寫入時（prefill 後） | 層間預算呈金字塔（低層多、高層少），層內沿用 SnapKV 式選擇 | E07 |
| **DuoAttention**（arXiv 2410.10819） | 替換：依 head 決定留全部或只留 sink＋最近 | 離線 | 只給少數 retrieval head 完整 KV，其他 head 只留 sink＋recent | E08 |
| **KVP**（ICML'26） | 替換：學習式（RL 排序） | 寫入時（prefill 後壓一次，decode 不再逐出）〔複核修正：原寫「每一步」。卡片：「只在 prefill 後壓一次，decode 零額外 FLOPs」（E06 KVP 學習設定「推論開銷」，[KVP] p17）〕 | 每個 KV head 一個小 RL agent，只看 K、V、位置就排出 token 的保留順序 | E06 |
| **ForesightKV**（ICML'26） | 替換：學習式（預測長期貢獻） | 解碼中每 L＝256 步一次（LongBench 泛化實驗是 prefill 後一次）〔複核修正：原寫「每一步」。卡片：依 R-KV 設定每 L 步壓一次、eviction length L＝256（E06 ForesightKV「對手」「掃描的自變數」「資料／負載」，[FKV] p6、p8）〕 | 長推理生成中，用小 MLP 預測每個 KV 的長期貢獻；先監督學習，再用 GRPO 強化學習 | E06 |
| **LookaheadKV**（ICLR'26） | 替換：學習式（預測回應會看哪些 prompt token） | 寫入時（prefill 時） | 加 32 個可學的 lookahead token，加上只對它們生效的 LoRA，在 prefill 時預測回應會看哪些 prompt token | E06 |
| **TRIM-KV**（ICLR'26） | 替換：學習式（產生時給保留分數） | 寫入時給分，之後每一步衰減 | token 生成時由小 gate 給一個隨時間指數衰減的保留分數，超出預算就丟最低者 | E06 |
| **KIVI**（ICML'24） | 壓縮：2-bit 量化 | 寫入時（token 離開最近 R 個的視窗時） | K 逐通道、V 逐 token 的非對稱 2-bit 量化，最近 R 個 token 保持 FP16 | E07 |
| **KVTuner**（ICML'25） | 壓縮：逐層混合精度 | 離線 | 離線用多目標最佳化搜尋逐層的 K/V 精度對（如 K8V4、K4V2） | E07 |
| **KVQuant**（NeurIPS'24） | 壓縮：3-bit 以下量化 | 寫入時 | pre-RoPE per-channel K、敏感度加權的非均勻資料型別、逐向量稠密＋稀疏離群值，達到 3-bit 以下 | E07 |
| **CacheGen**（SIGCOMM'24） | 壓縮＋搬移：編碼成位元流，依頻寬調整 | 寫入時編碼、讀取時選級別 | 把 KV 編成位元流（delta＋分層量化＋算術編碼），依頻寬逐 chunk 調整壓縮級別或改送文字重算 | E07 |
| **MuKV**（CVPR'26，影片） | 替換：寫入時多粒度剪枝 | 寫入時 | 在 ReKV 的框架上，把每段影片存成段、格、區塊三種粒度的 KV，存之前依注意力和頻率剪掉大部分，問答時兩階段檢索 | V01 |

---

## C. 稀疏注意力：每一步只取需要的 KV（像需求分頁）

KV 大多還是完整保存（在 GPU 或 CPU），但每一步只讀一部分。讀哪一部分由當下的 query 決定，像 OS 的需求分頁加預取。

| 論文 | 主要 OS 概念 | 何時決定 | 一句話（評測卡） | 卡 |
|:--|:--|:--|:--|:--|
| **Quest**（ICML'24） | 需求分頁：依 query 只讀前 K 個 page | 每一步 | 每個 KV page 記 key 的逐維 min／max，用當下 query 估注意力上界，只讀前 K 個 page | E08 |
| **InfiniGen**（OSDI'24） | 預取：用前一層預演，只從 CPU 預取重要的 KV | 每一步 | 在 layer i−1 用部分 query 權重與部分 key cache 預演 layer i 的注意力，只從 CPU 預取重要的 KV | E08 |
| **ShadowKV**（ICML'25） | 放置＋需求分頁：key 低秩留 GPU、value 卸到 CPU | 每一步 | pre-RoPE key 做低秩留在 GPU，value 卸到 CPU；decode 時用 chunk landmark 挑約 1.56% 的 KV 重建或抓回 | E08 |
| **MInference**（NeurIPS'24） | 計算：prefill 的稀疏注意力（不動 KV 大小） | 每一次 prefill | 用三種注意力稀疏型態加速長 prompt 的 prefill，不動 KV 大小 | E08 |
| **ReKV**（ICLR'25，影片） | 需求分頁：每題依相似度取回 64 格 | 讀取時 | 讓現成的 Video-LLM 不用訓練就能做串流影片問答：邊看邊存 KV，問題來時只取回相關的幾格 | V01 |

---

## D. Serving 的排程與架構（不是 KV 管理，但是背景）

| 論文 | 對應的 OS 概念 | 一句話（評測卡） | 卡 |
|:--|:--|:--|:--|
| **Orca**（OSDI'22） | 排程：以 iteration 為單位 | 以 iteration-level scheduling 加 selective batching，讓請求可以逐 iteration 進出 batch | E01 |
| **Sarathi-Serve**（OSDI'24） | 排程：切片（chunked prefill） | 用 chunked-prefills 加 stall-free batching，在不拖慢 decode 的前提下塞進 prefill | E01 |
| **DistServe**（OSDI'24） | 資源劃分：prefill 與 decode 分開放 | 把 prefill 與 decode 放到不同 GPU，並分別搜尋兩者的平行化與配置，以最大化每張 GPU 的 goodput | E01 |
| **Splitwise**（ISCA'24） | 資源劃分：兩個階段放不同機器 | 把 prompt 階段與 token 階段放到不同（可以是異質的）機器上，並用模擬器設計成本、功耗或吞吐最佳的叢集 | E01 |

---

## E. 評測方法與工具

| 論文 | 內容 | 卡 |
|:--|:--|:--|
| **Etalon**（arXiv 2407.07000） | 指出 TTFT、TBT、TPOT、normalized latency 各自的盲點，提出以 token 期限為基礎的 fluidity-index | E01 |
| **Yuan'24**（EMNLP'24 Findings） | 在對齊的壓縮比下比較五類長 context 方法的品質 | E08 |
| **Rethinking'25**（MLSys'25） | 從生產部署角度重量 KV 壓縮：吞吐、輸出長度分布、逐樣本失敗 | E08 |
| **Agrawal & Mayer'26**（arXiv 2607.05399） | 在同樣的模型與資料上比 KIVI、TurboQuant、SnapKV、CaM 的品質與單請求效能 | E08 |
| **kvpress**（NVIDIA） | 在 HF transformers 上用 forward hook 實作數十種 KV 壓縮（press），附統一評測 CLI 與 leaderboard | E08 |

---

## F. 統計：同一類（A）的論文在做什麼

把 A 組（共 33 項；多數跨請求、無損，例外見 A 組開頭〔複核修正〕）的「主要 OS 概念」和「何時決定」攤開。分析類的 Bottlenecks 和 vLLM 的分頁不列入；LRB、HALP 是 CDN 快取，不是 KV，放進來是因為它們是學習式替換的前例：

| | 寫入時 | 逐出時 | 讀取時 |
|:--|:--|:--|:--|
| 寫入策略／准入 | HCache、Marconi、HiCache、`OffloadingConnector`、Dynamo | Mooncake Store、HiCache（延後寫） | — |
| 放置 | KVDrive（依重要度，有損稀疏注意力）、MTDS、AdaptCache、EvicPress（有損壓縮）〔複核修正：原寫「（後兩者有損）」，會讀成 KVDrive 無損；KVDrive 在 RULER 上比 Full 低約 5–6 點（E04 KVDrive 卡③）〕 | EvicPress（層滿時調整） | MTDS |
| 替換 | — | Pensieve、Fancy-eviction、AsymCache、KVCache in the wild、LARU、LPC、SAECache、LRB、HALP、Bidaw、CachedAttention、SGLang | — |
| 讀取還原、搬移 | — | — | Cake、CacheFlow、KVPR、py-kvcache（載不載）、Strata、Tutti、LMCache、Mooncake、CacheBlend |

〔複核修正：py-kvcache 由「寫入策略／准入 × 寫入時」移到「讀取還原 × 讀取時」，理由見 A4 該列〕

**讀法**〔判讀〕：
- 跨請求、無損的研究，主力在**替換**（12 項）和**讀取還原、搬移**（9 項）〔複核修正：原寫 8 項，加入 py-kvcache〕。
- **寫入時就決定**的，要嘛是准入開關（依次數：HiCache 的 selective、`store_threshold`、Dynamo 磁碟層；或依前綴樹的分支點決定存哪些位置的 SSM 狀態：Marconi，單層），要嘛依層（HCache）、依重要度（KVDrive，有損）、依重用排卸載順序（MTDS），要嘛靠有損壓縮（AdaptCache、EvicPress）。〔複核修正：原寫「准入開關（存或不存，依次數或門檻）」；「門檻」原指 py-kvcache，它其實是讀取時決定；Marconi 的准入是規則式的命中情境分類（投機插入時存分支點的 state、另存最後一個 decode token 的 state），不是依次數（E05 Marconi 卡「與既有整理不一致」第 2 點，原文 p.5）〕
- **「寫入時、依 token 位置、無損地決定放哪一層」**在這張表裡是空的。這就是本研究的位置。這個結論只限於評測卡涵蓋的論文，動手前要再查新。
