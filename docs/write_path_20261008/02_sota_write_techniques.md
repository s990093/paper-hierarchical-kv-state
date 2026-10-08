# 02 現有系統怎麼寫 KV

> **一句話**：幾乎都是「產生了就全部寫」。差別在三個地方：什麼時候寫（寫穿、延後、命中兩次才寫）、怎麼寫才不擋路（非同步、讀取優先、批次、讓 GPU 發 I/O），以及少寫一點（只寫新的、壓縮、存較小的中間值）。依「位置」決定寫到哪的，在評測卡涵蓋的範圍內目前沒有看到〔複核修正：補上範圍限定〕；但「寫入時就決定」本身已有前例：KVDrive（依注意力重要度）、AdaptCache／EvicPress（依整段 context 的效用）、HCache（依層，比較重算與傳輸）〔複核補充，見 §3〕。

**出處**：評測卡 E02（工具與 KV 層系統）、E03（還原與重算）、E04（分層系統）、E05（逐出與快取策略；AdaptCache、EvicPress、Fancy-eviction）〔複核修正：原清單漏了 E05，但表中有引用〕、E07（壓縮），以及 `docs/EVAL_FOUNDATIONS_20261007.md` §2。這些都已讀過原文或程式碼，並由另一個 agent 獨立複核。卡內代號：[S] Strata、[B] Bidaw、[M] MTDS、[L] LMCache、[MC-F] Mooncake FAST 版、[CA] CachedAttention、[T] Tutti、[K] KVDrive。

---

## 1. 逐一看各系統

| 系統 | 什麼時候寫 | 寫到哪 | 寫什麼 | 怎麼寫才不擋路 | 有沒有量寫入的代價 | 出處 |
|:--|:--|:--|:--|:--|:--|:--|
| **Cake**（ICML'25） | 不管寫入：事先全部存好 | 模擬的單一慢層 | 全部 | — | 沒有 | E03；ICML p5 |
| **LMCache** | 寫穿：寫到所有啟用的層 | CPU、本地碟、遠端；論文實驗只用 CPU 和遠端 CPU | 只寫後端還沒有的新 token；只存滿 256 token 的 chunk；decode 的 KV：論文寫「累積滿一個 chunk 才寫」，程式碼預設不存（EVAL §2.2） | chunk 為單位批次搬；計算與 I/O 管線化 | 沒有寫入或逐出的消融 | E04 [L] p.5–8、p.11–15；EVAL §2.2 |
| **vLLM `OffloadingConnector`** | 預設全部存（`store_threshold=0`）；設成 ≥2 時，一個 chunk 要被計數到 N 次才存 | CPU；用 `TieringOffloadingSpec` 可再往下接檔案層 | 依門檻 | — | — | EVAL §2.3 |
| **SGLang HiCache** | 可選三種：寫穿、選擇性（命中 2 次才寫）、延後寫；**上游 CLI 預設是寫穿** | L2 主機記憶體、L3 後端 | 依策略 | — | — | E04 Strata 卡（程式碼 `542addad`）；EVAL §2.4 |
| **Strata**（OSDI'26） | 三種可切換：延後寫、寫穿、選擇性（**論文預設**，存取次數超過門檻才備份，寫入頻寬充足時門檻是 2） | GPU→CPU（pinned）→ 外部儲存 | 依策略 | 用 GPU kernel 搬運：CPU→GPU 用 2 個 CUDA block，GPU→CPU 備份用 1 個 | 量了搬運 kernel 對計算的干擾（prefill 慢不到 5%、decode 慢 10%）；原文認為「單一寫入策略不夠」 | E04 [S] p.6、p.9 §4.4 |
| **Mooncake**（FAST'25） | prefill 節點把新產生的增量 KV 寫回 CPU 記憶體（全部寫）；Store 先寫進記憶體，**記憶體逐出時才寫 SSD**〔複核補充：後半句出自開源 Mooncake Store 的設計文件（E02 B3），不是 FAST 論文〕 | 叢集的 DRAM 池 → SSD | 全部 | RDMA 傳輸引擎 | 論文沒有任何 SSD 實驗 | E04 [MC-F] p.5–6；EVAL §2.4 |
| **CachedAttention**（ATC'24） | 每一輪都寫（全部保存） | HBM → DRAM → SSD | 全部 | prefill、decode 都逐層**非同步**寫回 host；HBM 另設讀寫 buffer | 消融：非同步保存讓**整體執行時間**降 13–15%（prompt 1K–1.6K、decode 20 步；對照是整輪結束後一次寫完）〔複核修正：原寫「帶來 13–15% 的改善」，沒寫指標，[CA] p.11〕 | E04 [CA] p.6–8、p.11–14 |
| **Bidaw**（FAST'26） | 包含式快取：每一輪計算時就寫，慢層也保留一份，所以**逐出時不必大量寫** | DRAM → SSD | 省空間的中間 tensor（只對 MHA 有利） | 寫入不在關鍵路徑；轉換放在低優先 CUDA stream | — | E04 [B] p.6–10 |
| **MTDS** | 多張 GPU 同時卸載時排優先順序：被越多排隊請求共用的越先寫；資料庫已有的降優先；已完整存在的不寫；容量超過上限就停止新卸載 | GPU → DRAM → SSD | 依重用排序 | 專屬 I/O thread | 排程讓 PCIe 使用率從 32.09% 降到 13.24%，命中率從 0.875 小降到 0.841（ShareGPT、LLaMa-7B）〔複核補充：同一張表 2 裡，Random、LLaMa-13B 的命中率由 0.380 降到 0.215，降幅不小（[M] p.14 表 1–2）〕 | E04 [M] p.8–10、p.14 |
| **Tutti** | 全部寫，但**讀取優先**：寫入只在有空檔時發出，剩下的在 decode 期間盡量刷出；分散式模式下，從 GPU 被逐出時才寫 SSD | HBM → SSD（跳過 DRAM） | 全部，以 vLLM block 為物件，跨多顆 SSD 輪流放 | 讓 GPU 發 I/O（GPU io_uring，讀和寫都走這條路） | 量到讀寫並行時總頻寬降 60%；store 頻寬約 10 GB/s（Fig. 9b）；原文說寫入頻寬對端到端的影響小於讀取〔複核補充：這句是引用前作 [41]，不是 Tutti 自己量的（[T] p.10）〕 | E04 [T] p.2–8、p.10 |
| **KVDrive** | prefill 結束時**全部先寫 SSD**，再依注意力重要度把最重要的放 HBM、次重要的放 DRAM | HBM／DRAM／SSD | 全部；SSD 依語意連續性與 layer–head 打包成 extent | — | — | E04 [K] p.9–14 |
| **Pensieve**（EuroSys'25） | GPU 剩餘槽位不到 25% 時提前換出（原文寫「e.g. 25%」） | GPU → CPU | 全部 | **換入優先、換出等待** | 量到雙向同時傳輸，兩個方向各掉 18–20% | E03；p.6（25%）、p.9（18–20%、換入優先）〔複核修正：原寫 p.6–7；18–20% 與「換入優先」在 arXiv v3 p.9〕 |
| **HCache**（EuroSys'25） | prefill 與 decode 產生新的 hidden state 時就存（**兩段式**：先用一次 cudaMemcpy 把整個 batch 快照到主機記憶體，再由主機 daemon 整理成 chunk 寫 SSD）；每層存什麼（hidden state、KV 或不存、回來時從 token 重算）由離線 profile 的「傳輸 vs 計算」速度決定 | SSD（真實 NVMe） | 存每層的 hidden state，不存 KV；MHA 下大小是 KV 的一半，儲存省 1.92–2.40×；OPT-30B 排成 40 層 hidden＋8 層重算，那 8 層不存 | 兩段式存檔把寫入移出關鍵路徑，並把小寫入合併成大 chunk；以 64-token chunk 輪流分散到多顆 SSD，聚合頻寬、避免碎片；讀回時用 SPDK＋GDRCopy 直接進 GPU | 有：TBT 比理想最多高 4%；消融中直接寫 SSD（DirectIO）時 TBT 高 34%（7B、batch 16）、13%（13B、batch 32） | E03；HCache p.6–8 §4.1–4.2、p.10 表 3、p.12 Fig. 14〔複核修正：原「什麼時候寫」與「有沒有量寫入的代價」都寫「—」，漏了兩段式存檔與它的消融〕 |
| **CacheGen**（SIGCOMM'24） | 離線先編碼 | 儲存／網路 | 壓縮後的 KV：Mistral-7B（LongChat）上，8-bit 量化是 622 MB，CacheGen 是 176 MB〔複核修正：原寫「從 622 MB 降到 176 MB」，622 MB 是 8-bit 量化基線，不是原始 KV（p.2 Table 1）〕；每段存多個編碼級別，總儲存量和量化基線相近（p.11） | — | 編碼延遲約 200 ms（p.11） | E07 p.2 Table 1、p.9–11 |
| **AdaptCache**（SOSP'25 BigMem workshop） | **新 KV entry 產生時**觸發：對新 entry 和所有已存 entry 計算「每省一單位空間的效用損失」，對每個儲存層（DRAM、SSD）貪婪決定逐出、全精度存或壓縮〔複核修正：原寫「決定時機原文沒講清楚〔未查證〕」；已查 arXiv v2 p.2 §2〕 | DRAM 或 SSD | 每段 KV 決定壓縮法與壓縮率（有損），讓更多段留在 DRAM | — | 對 prefill 與卸載：TTFT −56%，品質下降 15% 以內 | E05 p.1–2 |
| **EvicPress**（arXiv 2512.14946） | 以整段 context 為單位，用效用函數同時決定壓縮與存放層。**新 KV 存入時**就依最高效用選壓縮與層；某層滿時，再對該層既有的 context 加重壓縮或往下層逐出（貪婪解多選擇背包）〔複核修正：原寫「從名稱看是逐出時決定〔判讀〕」；原文 p.6–7 §4.4、p.7 §5、p.16 Alg. 1 是存入時與層滿時都決定〕 | GPU／CPU／SSD | 壓縮法與壓縮率（有損），依每段 context 的品質敏感度 | — | 對壓縮＋LRU：同品質下 TTFT 快 1.43–3.77× | E05 p.4–9 |
| **Dynamo KVBM** | 磁碟層預設只寫存取頻率 ≥2 的 block | GPU → 主機 → 磁碟 → 物件儲存 | 依頻率 | — | — | EVAL §2.4（v1.5.0 起棄用） |

---

## 2. 寫入技巧分四類

![圖 2](figures/fig2_write_timing.svg)

### A. 什麼時候寫、要不要寫（時機與准入）

| 做法 | 意思 | 誰在用 | 好處 | 代價 |
|:--|:--|:--|:--|:--|
| **寫穿** | 一產生就寫到所有層 | LMCache、SGLang 上游預設、CachedAttention、Bidaw | 慢層一定有一份；被擠出時不用再寫 | 寫最多次；很多寫了用不到 |
| **延後寫** | 先放快的層，被擠出時才寫到慢層 | Mooncake Store、HiCache／Strata 的選項、Tutti 的分散式模式 | 沒被擠出就不用寫 | 擠出那一刻要一口氣寫一批，可能剛好碰上有人在讀 |
| **選擇性** | 命中（或被查到）夠多次才寫 | Strata 預設、HiCache selective、`store_threshold`、Dynamo 磁碟層 | 冷的資料不寫 | 第一次被重用時可能已經沒了 |
| **依重用排序** | 多個要寫時，先寫比較可能被用到的 | MTDS | 有限的頻寬先給重要的 | 要能預測重用 |

### B. 怎麼寫才不擋路（路徑）

| 做法 | 誰在用 |
|:--|:--|
| 非同步寫，不在關鍵路徑上 | CachedAttention（逐層）、LMCache（論文有逐層管線，[L] p.9；程式碼 `use_layerwise` 預設關閉，磁碟與遠端的 put 是非同步，E02 B1）、Bidaw（原文只說寫入不在關鍵路徑，[B] p.3）、HCache（兩段式存檔）〔複核修正：原寫「非同步、逐層寫」並列 Bidaw，Bidaw 原文沒有說逐層；補上 HCache〕 |
| 讀取優先，寫入等空檔 | Tutti、Pensieve |
| 用較大的單位批次寫 | LMCache（chunk 256）、KVDrive（extent）、Strata（page-first 版面）、HCache（把小寫入整理成 64-token chunk）〔複核補充〕 |
| 讓 GPU 自己搬、少經過 CPU | Strata（GPU→CPU 備份也用 GPU kernel）；Tutti（讀和寫都由 GPU io_uring 發出，寫入排在讀取之後）；HCache 只有讀回用 SPDK＋GDRCopy 直通，寫入是 cudaMemcpy 到主機、再由 CPU daemon 寫 SSD〔複核修正：原寫「Tutti、HCache 主要用在讀回的路徑」；Tutti 的寫入也走 GPU io_uring（[T] p.7 §3.3、p.8 的 P2P DMA 寫入），只有 HCache 是讀回限定（HCache p.8）〕 |

### C. 少寫一點（寫什麼）

| 做法 | 誰在用 |
|:--|:--|
| 只寫還沒有的新 token；只寫滿的 chunk；decode 的 KV 晚點寫（論文；程式碼預設是不存 decode 的 KV，EVAL §2.2）〔複核補充：與 §1 表一致〕 | LMCache |
| 存比較小的東西：hidden state、中間 tensor、壓縮後的 KV | HCache、Bidaw、CacheGen |

### D. 寫到哪（放置）

| 做法 | 誰在用 |
|:--|:--|
| 全部寫到同一層 | Cake（假設）、Mooncake（CPU）、Tutti（SSD） |
| 包含式：每一層都有一份 | Bidaw、寫穿的系統 |
| 全部先寫最慢層，再依**重要度**往上提 | KVDrive |
| 存入時依整段 context 的效用決定壓縮與層（有損） | AdaptCache、EvicPress〔複核補充〕 |
| 依**層**決定存什麼（hidden state、KV 或不存），依據是離線量的傳輸 vs 計算速度 | HCache〔複核補充〕 |
| 依**位置**決定放哪一層 | 評測卡涵蓋範圍內目前沒有看到（本研究的 S5）〔判讀〕〔複核修正：補範圍限定〕 |

---

## 3. 用「依什麼決定」來定位

| 決定的依據 | 系統 |
|:--|:--|
| 不決定，全部寫 | LMCache、Mooncake、CachedAttention、Bidaw、Tutti |
| 被用過幾次 | Strata、HiCache selective、`store_threshold`、Dynamo |
| 預測會不會被重用 | MTDS |
| 注意力重要度（有損稀疏注意力） | KVDrive |
| 每段 context 的品質敏感度（有損壓縮＋放哪一層） | AdaptCache、EvicPress（以整段 context 為單位，不看段內位置；存入時就決定） |
| 重算 vs 傳輸的速度（**依層**，離線 profile 一次），寫入時決定每層存 hidden state、KV 或不存 | HCache〔複核補充：E03 共同模式 4 的複核修正；HCache p.7 §4.1.2、§4.2，p.10 表 3〕 |
| 重算成本（從前段逐出） | Pensieve、Fancy-eviction 的成本感知逐出；但這是**逐出**時決定，不是寫入時 |
| **位置（重算成本 vs 搬運成本），在寫入時決定** | **目前沒有看到**〔判讀，限於評測卡涵蓋的系統〕 |

**AdaptCache、EvicPress** 也替每段決定放哪一層，而且是在存入時就決定〔複核修正：見 §1 表〕，但它們靠有損壓縮換空間，而且以整段 context 為單位，不分段內的前後位置。它們和第二階段（精度）比較相關。

最接近的是 **KVDrive**：它在寫入時就決定每段放哪一層。但有三點不同：
- 它依注意力重要度決定，不看位置；
- 它是每個請求各自一段長 context、彼此不共享的批次 decode（360K 時 batch 2），用的是有損的稀疏注意力〔複核修正：原寫「單一請求的 decode」；E04 KVDrive 卡 ② 指出主吞吐實驗是 batch 2–8 的離線批次，只有 Fig. 23 的成本分析有 bs=1〕；
- 我們是跨請求的精確重用。

查新時必須引用 KVDrive，並說清楚差別（E04 KVDrive 卡〔判讀〕）。〔複核補充：**HCache 也必須引用**。它是無損的、在寫入時依「重算 vs 傳輸」的成本決定每層存什麼（有些層不存、回來時重算），在「決定的依據」上比 KVDrive 更接近本研究；差別是它的單位是層、對同一套硬體離線定死一次，不看 token 位置，而且原文說目前只支援 MHA、GQA 屬範圍外（HCache p.13；E03 HCache 卡）〕
