# L7a　查新複查：候選 1（疊加壓縮的配比與分層搬運）與 K1b（跨層共用 token 子空間、全部層一組、Tucker）

> **日期**：2026-10-22（UTC）。**執行者**：文獻查新 agent（只讀 repo、只上網；沒有開 GPU 行程、沒有 import torch／vllm、沒有安裝套件；除本檔外沒有改任何檔案）。
> **範圍**：(1) 10/20 之後新出、或 L2／L5a／L5b／L5c／L5d 漏掉的相關工作；(2) TD-MoE 原文的作法能不能搬到 KV 上；(3)「跨層共用的是 token 子空間（左奇異向量）而不是向量」前人有沒有寫過、有沒有人量過。L2／L5a–d 已寫過的卡片**不重抄**，只引用它們的編號（例如「L5a B1」＝L5a 的 KVTC 卡）。
> **先讀過的本專案檔案**：`stage3_candidates.md`、`results_analysis.md`、`experiment_log.md`（K1b 一節與 10/22 補跑一節）、`lit/L5a_lowrank.md`、`lit/L5b_merge_crosslayer.md`、`lit/L5c_compress_on_offload.md`、`lit/L5d_stacking.md`、`lit/L2_compression_offload_prefetch.md`（目錄與相關卡）。
> **查證方式**：
> - 每篇都用 WebFetch 開過 arXiv abs 頁（或會議頁／PMLR 頁）核對標題、作者、日期；沒有 arXiv 版的 TD-MoE 用 ICLR 海報頁、ML Anthology 頁核對，正文 PDF 在 OpenReview 被人機驗證擋住，**打不開**（下面相關敘述標 NOT_VERIFIED），改讀官方程式碼（`github.com/ust-xu/TD-MoE`，用 curl 抓 4 個 .py 與 2 個設定檔到 scratchpad，只讀、沒有執行）。
> - 「讀 HTML」＝用 WebFetch 抓 `arxiv.org/html/<id>`，由工具內的摘要模型回答我指定的問題、照抄我指定的表格。**這不等於我自己逐段讀過**，所以這類一律標〔部分〕。關鍵數字（KVTC 表 16／18、JoLT 的分組規則、2609.28029 的 K／V 對比）都用第二次針對性的抓取再核一次。
> **閱讀深度標記**：〔全文〕本輪沒有；〔部分〕＝讀 HTML 並取回方法、指定表格、限制段；〔摘要〕＝只讀 abs 頁的摘要；〔標題〕＝本輪只核書目（內容沿用 L5x 的卡）；〔README〕／〔程式碼〕＝讀 repo。〔判讀〕＝我對原文的解讀；〔推論〕＝原文沒寫、我推的。
> **規則**：「沒查到」只代表這輪的搜尋沒找到（CLAUDE.md 規則 7），不等於沒有。各論文的數字是原作者在自己的模型、長度、預算下量的，**只並列、不排名**。本專案的數字一律引 `results_analysis.md`／`experiment_log.md`，本檔沒有產生任何新數字。
> **審查性質**：self-review，不是 cross-model review（CLAUDE.md §5）。

---

## 1. 結論（5 行）

1. **候選 1 的「方法」本身已被做掉**：KVTC（ICLR 2026）就是「全部層 × head 串起來的跨層 PCA（K 先去 RoPE）＋依倍數分配位元（0 bit＝降秩）＋DEFLATE 無損約 ×1.23＋放 HBM／DRAM／SSD」，16× 在 Llama-3.1-8B 的短上下文任務（≤8K）幾乎不掉；xKV 也已報過「跨層 SVD＋RTN 4-bit／3-bit 到 25.7×／32×」。
2. **候選 1 剩下的新意在「量測面」**：≥32K–128K 的生成答對率（獨立樣本、CI）、Qwen2.5-7B-1M、以及壓縮格式在 CPU／SSD／NFS 的**實測還原時間**與 GPU 忙時的翻轉——KVTC 只在 ≤8K（8B 模型）評品質、只量 H100 解壓與對重算的 TTFT。
3. **K1b 大部分已被做掉**：「逐 token 不像、但主奇異向量對齊」是 xKV 的核心論點（CKA＋附錄推導）；JoLT 直接用「相鄰層 token 基底的平均平方 cosine」分組；KVTC 表 16 已顯示串的層越多越好（1→32 層，倍數越高差越大）；Krishnan & Schulz（2609.28029）已用 Tucker／CP／TT／t-SVD 比過 KV 四階張量，結論是 head 與 layer 模態近乎滿秩——和我們「Tucker 多出來的維度沒幫助」同方向。
4. **TD-MoE** 是權重（專家×輸出×輸入）上的 Tucker＋校準白化，不訓練但要校準資料；它的三個零件在 KV 上都有人各自做過（張量化：JoLT／2609.28029；白化：AATC／SPECTRA／KQ-SVD 的單層 2D 版；秩分配：JoLT／KVTC），**「白化後的跨層（全部層）KV 基底」沒查到**，但 K1b 已顯示多的模態沒幫助，預期收益只會來自白化本身〔推論〕。
5. **最大威脅：KVTC**（https://arxiv.org/abs/2511.01815）；K1b 的 Tucker 部分則是 JoLT（https://arxiv.org/abs/2607.12550）與 2609.28029。

---

## 2. 對照表

「我們」欄的縮寫：**C1**＝候選 1（跨層低秩＋係數量化＋無損、配比隨倍數變、壓縮後放下層）；**K1b**＝跨層 token 子空間、全部層一組、Tucker。

| 論文 | 年份／出處 | 做了什麼 | 和我們重疊 | 理由（對到原文哪裡） | 深度 |
|:--|:--|:--|:--|:--|:--|
| **KVTC**（Staniszewski, Łańcucki） | 2025-11／ICLR 2026 | 每個 token 把**全部 l 層 × h head** 的 K（去 RoPE）或 V 串成 p＝l·h·d 維，離線 PCA（約 200K token、每個模型×倍數一次、所有請求共用）→ DP 分位元（尾端 0 bit）→ DEFLATE；目標是 HBM／DRAM／SSD 存放 | **C1：部分重疊（高）**；**K1b：部分重疊**（全部層一組） | 三個軸（跨層低秩、位元、無損）和我們的贏家家族相同；表 16 的「串幾層」消融；表 18 的「每 prompt vs 一次校準」。但 8B 模型的品質只到 8K（RULER-VT、NIAH 8K），沒有分層還原時間、沒有 GPU 忙 | 〔部分〕（L5a B1、L5d S2 已讀） |
| **xKV**（Chang *et al.*） | 2025-03／ICML 2026（PMLR 306） | 每 W 層（預設 4）串接 pre-RoPE K／V 做每 prompt SVD；附錄 D.5 疊 RTN 4-bit／3-bit | **C1：完全重疊**（低秩＋量化這個組合）；**K1b：完全重疊**（解釋） | 原文：「adjacent layers exhibit surprisingly low token-level similarity」＋高 CKA；附錄 A 推導「high CKA ⇒ dominant left singular vectors aligned」；表 9：Full 92.09、8.03× 88.85、25.70× 87.64、32.12× 84.64；表 8：W＝8 88.91 ≈ W＝4 88.50 | 〔部分〕（L5b X3 已讀全文） |
| **JoLT**（Krishnan, Schulz） | 2026-07（v4 2026-09-24）／投 ICLR 2027 | 依「相鄰層 token 模態前 8 個奇異向量的平均平方 cosine」把層分組（2–8 層），對每組四階張量（層×head×token×特徵）做**只截 token 與特徵模態**的 partial Tucker；殘差隨機旋轉後 {0,2,4,8} bit；一個 Lagrangian 在位元組預算下聯合分配秩與位元 | **K1b：部分重疊（高）**；**C1：部分重疊**（秩×位元聯合分配） | 分組準則就是我們 K1b Q1 的子空間重疊量（同一族量法）；layer／head 模態保持滿秩；自承優勢只在 2×–4×；不做卸載 | 〔部分〕（L5d S10 已讀） |
| **Tensor Decomposition of Transformer KV Caches**（Krishnan, Schulz；2609.28029） | 2026-09／投 SIMAX | KV 當四階張量（head, token, feature, Lg＝4 層），同儲存下比 Tucker、CP、TT、t-SVD、per-head SVD、Palu 式、xKV 式 | **K1b：部分重疊（高）** | 原文：「The head and layer modes are nearly full-rank and resist compression at any practical error level」；K：xKV 式贏 Tucker-4D，V：Tucker-4D 贏 xKV 式；只有 T＝1024、只有重建誤差、沒試 Lg＞4 | 〔部分〕 |
| **SPECTRA**（J. Zhang *et al.*） | 2026-08／arXiv | 每層由校準資料的 E[hᵀh] 得到基底（吸收進權重）、反向注水分位元（0 bit＝降秩）；K 在 RoPE 前快取 | **C1：部分重疊** | 同樣是「轉座標＋分位元」在 8–11×：Llama-3.1-8B LongBench 8.84× 51.89、11.12× 48.81（dense 53.24）；但每層、不跨層、無無損、無卸載、≤31.5K | 〔部分〕 |
| **AATC**（Laus *et al.*） | 2026-08／arXiv | 用校準輸入共變異的 Cholesky 白化 W_K／W_V 再 SVD（每層），attention 感知分位元 | **C1：相關但不同**；TD-MoE 搬遷問題的**白化零件已有 2D 版** | 原文 Σ_X＝LLᵀ、W̃＝LᵀW；5.82×；每層、不跨層（「per-layer transforms under a global budget」） | 〔部分〕（L5d S11） |
| **KV-Kaizen**（Monteiro *et al.*，Apple） | 2026-09／arXiv | 學一個 selector，逐層在「跨層共用／位元 {2,4,8,16}／MLA latent 截秩」裡挑，達總倍數 | **C1：相關但不同**（配比隨預算變的想法相同；要微調） | 要和 backbone 一起微調 270K 步；GQA 要先轉 MLA；原文觀察：只靠精度過不了 8×（2 bit 是地板）、在非 MLA 的 backbone 上超過 8× 要讓至少一半層改用跨層共用；沒有卸載 | 〔部分〕 |
| **TD-MoE**（Y. Xu *et al.*） | 2026／ICLR 2026（無 arXiv） | 同一層所有專家的**權重**疊成（專家×輸出×輸入）三階張量，校準白化後 HOSVD，依壓縮比分配三個秩 | **相關但不同**（權重、MoE） | 見 §3.7：程式碼要 wikitext2 256 條校準、不微調；Mixtral 設定專家模態不降秩 | 〔摘要＋程式碼〕；正文 NOT_VERIFIED |
| **Rethinking Tensor Decompositions…**（Zagitov *et al.*） | 2026-06／ICML 2026 workshop 海報 | 大規模檢驗張量分解壓縮 LLM 權重；Qwen3-30B-A3B 上 TD-MoE 明顯輸 MoBE | **相關但不同** | 摘要：張量分解假設的「共用子空間」和 LLM 的異質表示不合 | 〔部分〕 |
| **DynaCalKV**（Nguyen, Dang） | 2026-07／arXiv | K：依 CKA 動態把 **head** 分組＋秩分配；V：沿用 ReCalKV 的離線校準 | **K1b：相關但不同**（CKA 用在 head 間，不是層間） | 摘要自承 GQA、長上下文要謹慎 | 〔摘要〕 |
| **D-Quant**（Su *et al.*） | 2026-09／arXiv | 把逐 token 的熵編碼資料轉成固定長度位元流，讓 attention kernel 能平行反量化 | **C1：相關但不同**（無損層） | 熵編碼放進 attention 內，不是搬運前的無損 | 〔摘要〕 |
| **Lexico**（Kim *et al.*） | 2024-12／ICML 2025（PMLR 267） | 約 4K 個原子、與輸入無關的通用字典，OMP 稀疏編碼 KV | **C1：相關但不同**（離線共用基底的另一種形式） | 稀疏編碼不是低秩；短任務（GSM8K、LongBench） | 〔摘要〕 |
| **OjaKV**（Zhu *et al.*） | 2025-09／Findings of ACL 2026（L5a） | 線上 Oja 更新低秩基底；主張靜態離線基底在分布偏移時會變差 | **C1：部分重疊**（「事先校準基底會不會輸」的反方論點） | 摘要第一句即此主張 | 〔摘要〕（L5a A5） |
| **StiefAttention**（Benfenati *et al.*） | 2026-01（v2 2026-05）／arXiv | 在 Stiefel 流形上學正交投影基底（最小化 decoder 層輸出誤差），逐層依預算分配秩 | **C1：相關但不同**（離線基底） | 每層、不跨層；Llama3-8B 對 EigenAttention | 〔摘要〕 |
| **KQ-SVD**（Lesens *et al.*） | 2025-12／AISTATS 2026（PMLR 300） | 閉式求 attention 矩陣（QKᵀ）的最佳低秩，而不是只壓 K | **TD-MoE 搬遷：相關**（query 感知＝KV 版的「白化」方向） | 摘要 | 〔摘要〕（L5a #20） |
| **KV-COBRA**（Ha *et al.*） | 2026-09／arXiv | 每 head 聯合選截秩與量化、跨 head 重新分配預算 | **C1：部分重疊**（秩×位元聯合） | 每 head、不跨層 | 〔摘要〕（L5a B5） |
| **kvcompress**（社群，sachncs） | 2026／GitHub | JoLT 的獨立重寫；有 `vllm.v1.kv_offload` 的 `Offload` 子類別 | **C1：部分重疊**（「壓縮後接 vLLM 卸載路徑」的雛形） | README：GPU 端到端驗證「pending」、作者不是論文作者、沒有實測數字、沒提 ROCm | 〔README〕 |
| **DecoQuant**（Liu *et al.*） | 2024-05／ACL 2024 | 用張量分解把離群值移到小張量，大張量低位元量化（data-free） | **K1b：相關但不同** | 分解是為了量化，不是為了跨層共用 | 〔摘要〕 |
| **TPA**（Zhang *et al.*） | 2025-01／NeurIPS 2025 Spotlight | Q／K／V 用張量積的情境低秩表示，新架構 T6 | **相關但不同**（要訓練的架構） | 摘要 | 〔摘要〕 |
| **Tucker Attention**（Klein *et al.*） | 2026-03／arXiv | 對 attention **權重**做 Tucker，GQA／MLA 為特例 | **相關但不同** | 摘要明說分解的是 weight objects | 〔摘要〕 |
| MiniCache／KVSharer／CommonKV／SwiftKV／CLA／LCKV | 2024–2025 | 跨層共用（逐 token、整層、權重 latent、或要訓練） | **K1b：相關但不同**（我們 K1 已照條件停掉逐 token 共用） | 內容見 L5b X1、X2、X4 與 §3.3；本輪只核書目 | 〔標題〕 |
| CacheGen／KVServe／AdaptCache／EvicPress | 2023–2026 | 依頻寬／儲存層選壓縮格式 | **C1：部分重疊**（搬運面） | L5c：策略池都沒有低秩；GPU 負載只當殘差 | 〔標題〕（L5c 全文） |
| Eigen Attention／Loki／Palu／ShadowKV | 2024 | 離線基底或權重分解（單層） | **C1：相關但不同** | 見 L5a B2、B3、C1、A1 | 〔標題〕 |
| DepthWeave-KV | 2026-07／arXiv | 相鄰層共用低秩**通道**基底＋逐 token 殘差＋路由 | **K1b：相關但不同**（和我們「兩側分解」最像） | 摘要沒寫基底模型；L5b X8 已判可信度低 | 〔摘要〕 |
| GraceKV（Tian *et al.*，2608.07001） | 2026-08（v2 2026-10-10）／arXiv | 跨層、head、位置的原型樹，在同一預算下取捨覆蓋與解析度 | **相關但不同**（合併／丟 token 軸） | 摘要沒提位元或秩 | 〔摘要〕 |
| KV-Surgeon（Wagner *et al.*） | 2026-10-19／arXiv | 校準資料＋二次代理，離線刪 KV 並補償剩下的 | **相關但不同** | 摘要沒提低秩、量化、卸載 | 〔摘要〕 |
| Placement Policies Across GPU/CPU/SSD（Tumkur *et al.*，2609.16215） | 2026-09／arXiv | 模擬 HBM／DRAM／SSD 放置策略 | **相關但不同**（只放置、不壓縮） | 摘要 | 〔摘要〕 |

---

## 3. 卡片（重要的幾篇）

### 3.1　KVTC（arXiv 2511.01815；ICLR 2026）〔部分〕——候選 1 的最大威脅

L5a B1、L2 B3、L5d S2 已經有機制與時間的卡片，本輪只補「和候選 1／K1b 正面相撞」的四件事（都是本輪重新抓 HTML 核過的）：

1. **它就是「全部層一組」的跨層低秩**。原文把 l 層、h head 的 K（或 V）沿 hidden 維串起來（p＝l·h·d_head），對這個 p 維做 PCA；所以每個 token 只存一組係數、所有層共用——和 xKV 把 W 設成全部層是同一件事，只是基底在**特徵側**、而且是**離線校準、所有請求共用**〔判讀〕。K、V 各自一組基底；K 先去 RoPE（原文：「positional embeddings distort the apparent low-rank structure of keys」）。
2. **它就是「低秩×位元×無損」三層疊加，且依倍數自動改配比**。DP 對每個主成分組選位元（含 0 bit＝截掉），每個模型×每個倍數校準一次；再用 DEFLATE 無損約 ×1.23。Llama 3.1 8B kvtc16×（實際 CR 18–22）：GSM8K 56.9（不壓 56.8）、LITM 99.3（99.4）、RULER-VT 99.1（99.8）。→ 我們「8–20× 用 INT4、24–32× 改 K4V2」這種「配比隨倍數變」，在 KVTC 是 DP 自動做的事〔判讀〕。
3. **表 16（附錄 B.10，-DPQ＝只截主成分、不分位元）：串的層越多越好，倍數越高差越大**（Llama 3.1 8B，LITM／RULER-VT）：

   | 倍數 | 1 層 | 4 層 | 8 層 | 16 層 | 32 層 |
   |:--|:--|:--|:--|:--|:--|
   | 8× | 49.1／70.8 | 95.9／93.0 | 99.2／90.7 | 98.8／92.8 | 98.6／93.0 |
   | 16× | 0.0／0.5 | 32.0／62.7 | 60.7／85.8 | 72.7／89.0 | 69.2／87.0 |
   | 32× | 0.0／0.1 | 0.0／1.6 | 2.6／33.0 | 24.6／60.4 | 44.4／83.0 |

   表 17：32× 只壓 K 時 32 層 LITM 96.7、只壓 V 時 81.4（K 比 V 好壓）。→ 和我們 K1b「全部層一組在同位元組下誤差是 4 層一組的 0.78–0.85（K）／0.69（V）」**同方向**，而且它是答對率、我們只是重建誤差〔判讀〕。差別：它是校準 PCA、8K 以下、只有 Llama。
4. **表 18（附錄 B.11）：每 prompt 基底 vs 一次校準**。每 prompt 算基底時要存 p×r 的投影，8K 下壓縮比只剩 1.0–12.4；「模擬對話」（之後生成的 token 也用同一基底壓）時 GSM8K 掉到 31.4（一次校準 56.9）。→ 和我們 38c／38k（同位元組、32K、加 INT4，兩種基底分辨不了）**比的不是同一件事**：它沒有在同位元組＋量化下比，也沒有 ≥32K（基底的位元組在長上下文會被攤薄）〔判讀〕。

**它沒做的**（對候選 1 是缺口）：8B 模型的品質只評到 8K（RULER-VT、NIAH 8K；NIAH 100K 只有 70B）；沒有 Qwen2.5-7B-1M；時間只量 H100 上的壓縮／解壓（BS 8、8K：379／267 ms）與「解壓 vs 重算」的 TTFT（3,098 → 380 ms），以及 B.16 的 LMCache 多 client（L5a B1）；沒有分層（CPU／SSD／NFS）的還原時間對照，也沒有 GPU 忙。

### 3.2　JoLT（arXiv 2607.12550；投 ICLR 2027）〔部分〕——K1b Tucker 部分的主要前作

L5d S10 已寫它的同位元組結果（約 4× 時輸給單純量化）。本輪補三點：
- **分組準則＝我們的子空間重疊量**。原文：「We partition the L layers into G contiguous groups by clustering layers whose token-mode singular vectors are well-aligned」、「score consecutive layers by the mean squared cosine between their bases. A new group starts whenever the score falls below the median score across the network」（前 8 個奇異向量；組大小 2–8，尾端單層併入，最多 9）。我們 K1b Q1 的 ‖U_lᵀU_{l+1}‖²_F／64 也是平均平方 cosine，只是取前 64 個〔判讀〕。**它沒有報分數的數值、沒有隨機基準、沒有和逐 token cosine 對比、沒說有沒有去平均**；組數只給 Mistral（T＝1024）32 層分成 11 組。
- **只截 token 與特徵兩個模態，layer 與 head 模態保持滿秩（單位矩陣）**；token、特徵基底由組內所有層、所有 head 共用。沒有「分組 vs 不分組」的消融。
- **xKV 是它的基準**：Mistral、n＝1024 困惑度 2×／3×／4×：xKV +6.75%／+13.2%／+19.4%，JoLT +0.01%／+0.04%／+5.65%。RULER 64K（LLaMA-3.1-8B）：不壓 90.97、2× 91.20、3× 90.87、4× −0.90、5× 88.57（−2.40）。只到 5×，沒有卸載、沒有吞吐。

### 3.3　Krishnan & Schulz，Tensor Decomposition of Transformer KV Caches（arXiv 2609.28029；投 SIMAX）〔部分〕

- **設定**：K／V 每層是 nh×T×dh 的三階張量，相鄰 Lg＝4 層疊成四階；K 用 pre-RoPE；Mistral-7B-v0.3（GQA）、LLaMA-2-13B（MHA）；WikiText-2、**T＝1024**、3 次抽樣；只量重建誤差。
- **格式比較**：同參數量下 Tucker（ST-HOSVD；HOOI 不改變誤差）在 2–5× 都是最低，CP、t-SVD、TT 依序較差。
- **和 2D 展開比**（K、V 共同預算，表 5）：K 是 xKV 式（Lg＝4 串接）較低（Mistral 2×：0.099 vs Tucker-4D 0.134；LLaMA-2-13B 0.113 vs 0.145）；V 是 Tucker-4D 較低（Mistral 2×：0.143 vs 0.202；4×：0.308 vs 0.369）。
- **模態二分**：原文「The head and layer modes are nearly full-rank and resist compression at any practical error level」；強迫 layer 秩＝3 時 Mistral K 誤差 0.134 → 0.393。
- **RoPE**：「Post-RoPE keys lose 41%–64% of their pre-RoPE compressibility」。
- **沒做的**：Lg＞4 或全部層一組（原文的「all layers」是逐層誤差平均，不是聯合分解）；CKA、主角、跨層子空間；答對率；量化、卸載、校準基底。作者自己列為開放問題：「Whether the token-mode structure measured on a filled prompt persists during incremental generation」。
- **和 K1b 的關係**〔判讀〕：我們「4 維 Tucker 幾乎等於全部層一組（多出來的維度沒幫助）」＝它的「layer／head 模態近乎滿秩」在 32K、Qwen 上的重現；我們 V 的全部層一組比 4 層一組好（0.69）也和它「V 用 Tucker-4D 比 xKV-4 好」同方向。差別只在長度（32K vs 1K）、模型、和「全部層一組」這個操作點。

### 3.4　xKV 的解釋（arXiv 2503.18893；ICML 2026，PMLR 306:12758–12778）〔部分〕——回答指定問題 3

- **原文的解釋就是「共用子空間，不是共用向量」**：「adjacent layers exhibit surprisingly low token-level similarity」（圖 2a，V cache）；CKA 圖「prominent red blocks indicating remarkably high CKA scores」（圖 2b）；附錄 A 是推導（不是正式定理），結論「a high CKA score implies that the dominant left singular vectors … are strongly aligned」、「a large CKA value requires significant overlap」；圖 2c：串的層越多，達 95% 能量所需的秩比例越低。
- **沒做的**：沒有直接量主角或子空間重疊（只有 CKA 與 95% 能量秩）；窗口只試到 W＝8（表 8：45.71／75.15／88.50／88.91，原文說 8 時「saturates」），**沒有 W＝全部層**；沒有校準基底（每 prompt 線上 SVD）。
- **疊加量化（表 9）**：Full 92.09；xKV 8.03× 88.85；＋RTN 4-bit 25.70× 87.64；＋3-bit 32.12× 84.64（量化套在「壓縮後的 cache」上，group 大小、K／V 是否分開未說明；表 9 的 context 長度在擷取段落裡沒寫）。→ 和我們「32 倍掉 7–8」〔`results_analysis.md` 4.11〕量級相近〔判讀，任務與量法都不同，不排名〕。

### 3.5　SPECTRA（arXiv 2608.07915）〔部分〕

- 每個 K／V 投影一個基底，來自校準資料的二階動差 G＝E[hᵀh]（離線、吸收進權重）；K 在 RoPE 前快取、重建後再套 RoPE；反向注水 b_j＝b̄＋½log₂(v_j/GM(v))，0 bit 的組直接丟（＝降秩）；每 token 一個 fp16 scale／zero 算進位元組。
- Llama-3.1-8B-Instruct LongBench（8 個英文任務、每任務前 150 題、**中段截斷到 31,500 token**）：不壓 53.24；3.56× 53.56、7.66× 54.00、8.84× 51.89、11.12× 48.81；約 8× 時 TurboQuant 47.11、RotateKV 45.97。Mistral 8.93× 50.41（不壓 53.20）。NIAH 單針到 12× 不掉。Qwen2.5-7B 只和不壓比。
- 沒有 xKV、KVTC、Palu 當基準；沒有無損、沒有卸載、沒有時間（只有容量）；不跨層。
- **對候選 1**〔判讀〕：它在 ≤31.5K 給出了「單一轉換編碼 8–11×」的點，和我們「約 8 倍接近無損、12 倍掉 1–3」量級接近；我們要說新意，就得說「跨層」與「≥32K／分層搬運」，而不是「能到 8–12×」。

### 3.6　KV-Kaizen（arXiv 2609.37988）〔部分〕

- 動作空間：每層 17 個選項＝「沿用別層（depth）」＋4 種 latent 截秩比例 × 4 種位元（2／4／8／16）；selector 是兩層小 transformer，prefill 前跑一次（約 3 ms）。
- **一定要微調**：backbone 和 selector 在 Dolci-Instruct-SFT 上一起訓 270K 步；GQA 先用 CARE＋TransMLA 轉成 MLA；原文說把學到的配置套在凍結的 backbone 上「collapses accuracy」。模型：Qwen2.5-Instruct 1.5B–32B、Mistral-7B-v0.3、OLMo-3-7B-Think；訓練長度 4,096；RULER 16K（Qwen2.5-14B，4× QA 0.614、未壓縮的 control 0.543；＋SnapKV 16× 0.560、32× 0.531）。
- **和候選 1「配比隨倍數變」最像的觀察**：「Precision alone cannot pass 8× because two bits is its floor」；在非 MLA 的 backbone 上，超過 8× 要讓至少一半層改用 depth 共用。→ 想法相近，但它要訓練、latent 是權重式、跨層是整層沿用；我們是不訓練、活化值跨層 SVD〔判讀〕。沒有卸載。

### 3.7　TD-MoE（ICLR 2026；OpenReview D9cnZNZfxX；無 arXiv）〔摘要＋程式碼〕——回答指定問題 2

**能核到的**（ICLR 海報頁、ML Anthology 頁、官方 repo `ust-xu/TD-MoE`，MIT）：
- 作者 Yuebin Xu、Yanhong Wang、Xuemei Peng、Hui Zang、Minghao Chen、Pengfei Xia、Zeyi Wen。摘要：同一層的所有專家聯合成一個三維張量分解；三個零件＝cross-expert tensorization、multi-linear whitening（去相關輸入與輸出特徵）、3D rank allocation（在目標壓縮比下分配三個秩、最小化重建誤差）；Qwen2-57B-A14B、Mixtral-8×7B、7 個常識推理任務，20% 參數減少「almost lossless」。
- **三個模態**〔程式碼〕：`whiten_tensor` 的註解寫 `W: [n_expert, d_out, d_in]`＝（專家, 輸出, 輸入）；分解是逐模態 SVD 截斷的 HOSVD（`hosvd_decomposition`，用 tensorly 的 multi_mode_dot 算 core），不是 HOOI。argparse 描述字串是「Whitened Tucker-MoE」——**使用者問的「TuckerMoE」很可能就是指它**〔推論〕。
- **不訓練、但要校準資料**〔程式碼＋設定檔〕：流程只有 covariance → whitening → evaluate，沒有微調步驟；設定檔 `whitening_nsamples: 256`、`dataset: wikitext2`，序列長 2048（`run_covariances.py` 預設）。
- **白化怎麼算**〔程式碼〕：輸入側＝輸入活化共變異的 Cholesky 因子右乘（`W @ S2`）；輸出側＝反向傳播得到的梯度共變異的 Cholesky 逆左乘；梯度用 `torch.randn_like` 產生的隨機 dummy gradient 反傳。Mixtral 設定用 `whiten_type: output`。
- **秩分配**〔程式碼〕：`calculate_tucker_ranks_balanced` 固定專家模態秩 r0（預設＝專家數；設定檔 `k`：Mixtral 為 0＝「keep all experts」、Qwen2 為 8），對 (r1, r2) 做格點搜尋，目標是參數量誤差最小、並懲罰 r1/n1 與 r2/n2 不平衡。**和摘要說的「最小化重建誤差」是否同一個演算法：NOT_VERIFIED**（正文打不開）。只壓設定檔列出的部分層。
- **只作用在權重**；沒有任何 KV、attention、活化值張量的內容（程式碼與摘要都沒有）。
- **第三方評價**：Zagitov *et al.*（2606.03465，ICML 2026 workshop 海報）在 Qwen3-30B-A3B 上報 TD-MoE「performs substantially worse than MoBE」，並主張張量分解假設的共用子空間和 LLM 的異質表示不合〔部分〕。

**搬到 KV 上，前人做過哪些零件**〔判讀〕：

| TD-MoE 零件 | KV 上的對應 | 誰做過 | 結果 |
|:--|:--|:--|:--|
| 疊成張量、共用模態因子 | 層（或 head）當一個模態 | JoLT、2609.28029；矩陣版＝xKV、KVTC | **layer／head 模態近乎滿秩**（2609.28029），JoLT 乾脆不截；收益來自共用 token（或特徵）模態——TD-MoE 自己的 Mixtral 設定也不降專家模態，形狀一致 |
| 多線性白化 | KV 的「白化」應該以 query／attention 輸出為準 | AATC（輸入共變異 Cholesky 白化 W_K／W_V，每層）、SPECTRA（E[hᵀh]，每層）、KQ-SVD（QKᵀ 的閉式最佳低秩） | 都是**單層 2D**；**跨層（或全部層）張量＋白化：沒查到** |
| 3D 秩分配 | 秩 × 位元在位元組預算下聯合分配 | JoLT（Lagrangian，token／特徵秩＋殘差位元）、KVTC（DP）、KV-COBRA、RDKV（L5d） | 已有，而且比 TD-MoE 程式碼的格點搜尋更完整 |

→ 可以搬，但三個零件都已各自出現在 KV 文獻裡；唯一沒查到的組合是「query 感知白化後的跨層（全部層）基底」。K1b 已顯示多出來的模態沒幫助，所以就算做，預期收益只會來自白化本身〔推論，沒有我們自己的訊號〕。

### 3.8　其他（短卡）

- **AATC**（2608.14191）〔部分〕：A 每層全域、B 每 head-group（Llama G＝2、Qwen G＝1）；轉換用 wikitext2 校準，位元分配另用 FineWeb＋OpenR1-Math 256 條 × 2048 token；5.82× 在兩個模型都在雜訊內（Qwen RULER-32k 0.715 vs 不壓 0.720，KIVI 5× 0.233）；沒有 CUDA 實作、沒有時間、沒有無損與卸載。是 TD-MoE「白化」零件在 KV 上最直接的 2D 版本。
- **kvcompress**（GitHub sachncs/kvcompress，Apache-2.0）〔README〕：JoLT 的獨立重寫；vLLM 整合有三種形狀，其中 Shape B 是 `vllm.v1.kv_offload` 的 `Offload` 子類別（目標 vLLM ≥0.19,<1.0），**GPU 端到端驗證 pending**；README 自承結果資料夾是作者機器上的 CPU 數字、不是論文重現；沒提 ROCm、沒有無損。→ 是「壓縮後走 vLLM 卸載路徑」的現成雛形，但沒有任何量測。
- **OjaKV**（2509.21623）〔摘要〕：明確主張「static subspace learned offline … degrades when the data distribution shifts」，所以用 Oja 線上更新、首尾 token 保留全秩。這是我們「事先校準基底還沒證明不輸」的反方前作，正式檢驗時應該引用。
- **Lexico**（2412.08890）／**StiefAttention**（2601.21686）〔摘要〕：另外兩種「離線、所有請求共用」的表示（通用字典、Stiefel 流形上學的正交基底），都是單層、短任務。
- **D-Quant**（2609.19880）〔摘要〕：熵編碼做成固定長度位元流以便 kernel 內反量化——和我們「無損只在搬運時用」的位置不同。
- **DecoQuant**（2405.12591，ACL 2024）〔摘要〕：張量分解是為了把離群值移走再量化，data-free；不是跨層共用。
- **μMoE**（2402.12550，NeurIPS 2024）、**TuckA**（2511.06859）〔摘要〕：名字裡有 MoE／Tucker，但都是要訓練的專家參數化（視覺 MoE、PEFT adapter），和 KV 無關；列出來只是排除「TuckerMoE」的其他可能指涉。

---

## 4. 缺口清單與「被做掉的部分」

### 4.1　被做掉的（不能再當新意講）

| 我們的主張 | 前人 | 判定 |
|:--|:--|:--|
| 跨層低秩（pre-RoPE K）＋係數量化可以疊、而且到 25–32× | xKV 表 9（25.70× 87.64、32.12× 84.64，Full 92.09） | 完全重疊 |
| 低秩 × 位元 × 無損三層疊加、依預算聯合分配（0 bit＝降秩） | KVTC（PCA＋DP＋DEFLATE ×1.23）；JoLT、SPECTRA、AATC、KV-COBRA（秩×位元） | 完全重疊（原理）；我們的「配比地圖」只是這個原理在實務格式（KIVI INT4／K4V2）上的實測 |
| 配比隨目標倍數變 | KVTC 的 DP 每倍數校準一次；JoLT「優勢只在 2×–4×」；KV-Kaizen「只靠精度過不了 8×」 | 部分重疊（現象已知；我們的具體轉折點 20×／24× 是新的實測點） |
| 串進基底的層越多越好、倍數越高差越大 | KVTC 表 16、17（1→32 層，答對率） | 大部分重疊（它是校準 PCA、8K、Llama；我們是每 prompt、32K、重建誤差） |
| 跨層共用的是 token 子空間（主左奇異向量）不是逐 token 向量 | xKV（cosine 低＋CKA 高＋附錄 A 推導）；JoLT 用同一族量法分組 | 完全重疊（解釋）；我們只是用「子空間重疊＋去平均＋隨機基準」在 32K 重量一次 |
| Tucker 多出來的模態（層、head）沒幫助 | 2609.28029「head and layer modes are nearly full-rank」；JoLT 不截這兩個模態 | 大部分重疊（它們在 T＝1024） |
| 逐 token 跨層共用（MiniCache 式）在這兩個模型、長上下文不成立 | xKV 重測 MiniCache（L5b X1、X3） | 完全重疊（K1 是在我們資料上的重現） |
| 事先校準、所有請求共用的基底 | KVTC、SPECTRA、AATC、Eigen Attention、Lexico、StiefAttention | 完全重疊（作法）；「在我們設定下不輸」仍是開放問題 |
| 依重要性丟 token、合併不值得疊 | L5d §1.5（Q-Hitter、SelKV、GRKV 等） | 部分重疊（我們的版本是長上下文＋同位元組＋多輪） |

### 4.2　前人沒做（沒查到）、我們有訊號的

| # | 缺口 | 最近的前人與差距 | 我們的訊號（出處） | 強度 |
|:--|:--|:--|:--|:--|
| G1 | **疊加格式在分層（CPU／SSD／NFS）上的實測還原時間**，以及最快格式隨 GPU 忙碌翻轉 | KVTC 只量 H100 解壓與對重算的 TTFT；CacheGen／KVServe／AdaptCache／EvicPress 的策略池沒有低秩，GPU 負載只當殘差（L5c）；kvcompress 有卸載介面但沒有量測 | X3 端到端 8 格：慢層還原約壓到 1/3；GPU 很忙時低秩＋INT4 比不壓縮慢（`results_analysis.md` 4.9） | 中（只有 4 組翻轉實測） |
| G2 | **≥32K–128K、生成答對率、獨立樣本＋CI** 下的疊加極限（8–32×），含 Qwen2.5-7B-Instruct-1M | xKV：RULER（Llama；疊量化那張表沒寫長度）；JoLT：64K 只到 5×；SPECTRA：≤31.5K；KVTC：8B 模型 ≤8K | X2／X2b／X4b（`results_analysis.md` 4.7、4.11） | 中（題組間差到約 3 個百分點） |
| G3 | **同位元組＋係數量化**下，每 prompt 基底 vs 事先校準基底，在 ≥32K | KVTC 表 18 只在 -DPQ（不量化）、8K、且把每 prompt 基底的位元組算進去；OjaKV 主張靜態會變差 | 38c／38k：分辨不了，但還不能證明不輸（`results_analysis.md` 4.8） | 弱（需要更多題） |
| G4 | **每 prompt、全部層一組**在 ≥32K 的同位元組比較 | xKV 只到 W＝8；JoLT 最多 8–9 層；2609.28029 只有 Lg＝4；KVTC 的 32 層是校準 PCA、8K | K1b（Qwen，CPU 重建誤差 0.78–0.85／0.69）；GPU 檢驗 62b 在跑（`experiment_log.md` 10/22） | 弱（被 KVTC 表 16 大幅壓縮新意） |
| G5 | 「query 感知白化」＋跨層（全部層）基底（TD-MoE 搬遷的唯一空位） | AATC／SPECTRA／KQ-SVD 都是單層 | 無 | 無訊號（推測） |
| G6 | 跨層子空間重疊的**數值**（去平均、隨機基準、32K、兩個 GQA 模型） | xKV 只給 CKA 圖；JoLT 不報分數 | K1b Q1：去平均 0.63–0.67、隨機約 0.002 | 弱（只是量法上的補充，不足以單獨成為貢獻） |
| G7 | AMD MI300X／ROCm 上的上述全部 | 沒查到 | 本輪所有實測都在 MI300X | 弱（只是平台） |

### 4.3　對既有文件的更正（只列，沒有改原檔）

1. `stage3_candidates.md` 候選 1 的「前人」欄寫「xKV 做了跨層 SVD＋4-bit；兩兩組合很多；『三軸以上、長上下文、同位元組＋搬運』的系統性比較查不到」。應補上：**KVTC 已是「全部層跨層 PCA＋位元分配＋無損＋分層存放」的三層疊加**，和我們的贏家家族相同；查不到的只剩「長上下文品質＋分層還原時間＋GPU 忙」這一段。
2. `experiment_log.md` K1b 一節的查新寫「沒查到把 Tucker 用在 KV 活化值（層×head×token×維度）上、不訓練的作法」。實際上 **JoLT（2607.12550）與 2609.28029 都是**，而且 L5d 的參考清單裡就有（S10、O6）；那次是漏看。
3. L5d 對 JoLT 的卡沒記到「依 token 基底對齊度分組」這件事，L5d 對 2609.28029 只核到摘要；本檔 §3.2、§3.3 已補。

---

## 5. 建議的定位

**一句話**：「長上下文（32K–128K）下，把跨層低秩＋係數量化＋無損當成**存到下層的格式**時，它在 CPU／SSD／NFS 還原時省多少時間、什麼時候會因 GPU 忙而反輸——前人（KVTC、xKV、JoLT）只在短上下文給出格式與品質，沒有把長上下文品質、分層還原時間、GPU 爭用放在同一個決策裡。」

要避免的說法：
- 「第一個把低秩、量化、無損疊起來」——KVTC 已做。
- 「發現跨層共用的是子空間不是向量」——xKV 已寫。
- 「Tucker／多維分解沒幫助」當成獨立發現——2609.28029、JoLT 已有，只能說「在 32K、Qwen 上重現」。
- 「全部層一組比 4 層一組好」當成新發現——KVTC 表 16 已顯示，只能說「每 prompt、32K 下也成立（待 62b）」。

建議的下一步對照（只寫想法、不排實驗）：在我們的 32K 題組上加一組 **KVTC 式基準**（全部層、校準 PCA、去 RoPE、位元分配），因為它在短上下文報的 16× 幾乎無損，比我們 12× 掉 1–3 好看；不比它，審稿人一定會問〔判讀〕。

---

## 6. 參考文獻

「核」：A＝本輪用 WebFetch 開 arXiv abs 頁核過標題、作者、日期（與 comment）；H＝本輪讀了 arXiv HTML（〔部分〕）；P＝核過會議頁／proceedings 頁；C＝讀過官方程式碼或 README；NV＝NOT_VERIFIED。

### 主要威脅與直接相關
| # | 文獻 | URL | 核 |
|:--|:--|:--|:--|
| R1 | K. Staniszewski, A. Łańcucki, "KV Cache Transform Coding for Compact Storage in LLM Inference," ICLR 2026（arXiv comment：Accepted to ICLR 2026；v1 2025-11-03、v2 2026-03-11） | https://arxiv.org/abs/2511.01815 | A＋H |
| R2 | C.-C. Chang, W.-C. Lin, C.-Y. Lin, H.-Y. Chiang, Y. Akhauri, X. Dai, H. Jiang, Y. Li, L. Ceze, K.-C. Wu, M. S. Abdelfattah, "xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction," ICML 2026, PMLR 306:12758–12778（v1 2025-03-24、v2 2026-05-27） | https://arxiv.org/abs/2503.18893 ；https://proceedings.mlr.press/v306/chang26d.html | A＋H＋P |
| R3 | R. Krishnan, V. Schulz, "A JoLT for the KV cache: Near-Lossless KV Cache Compression via Joint Rank-bit Allocation," arXiv 2026（v1 2026-07-14 … v4 2026-09-24；comment：Under review at ICLR 2027）。搜尋結果顯示另一個題名「…via Joint Tucker and JL-Residual Allocation for LLMs」，應是較早版本，未逐版核對 | https://arxiv.org/abs/2607.12550 | A＋H |
| R4 | R. Krishnan, V. Schulz, "Tensor Decomposition of Transformer Key-Value Caches: Spectral Structure and Format Comparison," arXiv 2026（v1 2026-09-23；comment：Submitted to SIMAX） | https://arxiv.org/abs/2609.28029 | A＋H |
| R5 | J. Zhang, L. Wu, K. Wan, H. Chen, L. Hong, "SPECTRA: Pushing the KV Cache Beyond the 2-Bit Cliff via Spectral Transform Coding," arXiv 2026（v1 2026-08-08） | https://arxiv.org/abs/2608.07915 | A＋H |
| R6 | H. Laus, C. M. Verdun, H. Wang, F. du Pin Calmon, F. Krahmer, "KV Cache Compression Through the Lens of Transform Coding"（AATC），arXiv 2026（v1 2026-08-14） | https://arxiv.org/abs/2608.14191 | A＋H |
| R7 | J. Monteiro, L. Béthune, A. Filippova, S. Laguna, D. Grangier, M. Cuturi, "KV-Kaizen: Learning Context-Adaptive Cache Compression Choices," arXiv 2026（v1 2026-09-29、v2 2026-09-30） | https://arxiv.org/abs/2609.37988 | A＋H |
| R8 | Y. Xu, Y. Wang, X. Peng, H. Zang, M. Chen, P. Xia, Z. Wen, "TD-MoE: Tensor Decomposition for MoE Models," ICLR 2026（無 arXiv；OpenReview D9cnZNZfxX） | https://iclr.cc/virtual/2026/poster/10010804 ；https://mlanthology.org/iclr/2026/xu2026iclr-tdmoe/ ；https://github.com/ust-xu/TD-MoE | P＋C；**正文 NV**（OpenReview PDF 被人機驗證擋住） |
| R9 | A. Zagitov, A. Miasnikov, M. Krutikov, V. Aletov, G. Molodtsov, N. Bashirov, A. Tsedenov, A. Beznosikov, "Rethinking the Role of Tensor Decompositions in Post-Training LLM Compression," arXiv 2026（v1 2026-06-02）；ICML 2026 Workshop on Weight-Space Symmetries 海報 | https://arxiv.org/abs/2606.03465 ；https://icml.cc/virtual/2026/68106 | A＋H＋P |
| R10 | sachncs, kvcompress（JoLT 的獨立重寫，Apache-2.0） | https://github.com/sachncs/kvcompress | C（README） |

### 相關（摘要層級）
| # | 文獻 | URL | 核 |
|:--|:--|:--|:--|
| R11 | T. T. Nguyen, Q. V. Dang, "DynaCalKV: Key-Value Cache Compression via Head Grouping and Adaptive Rank Allocation," arXiv 2026（v1 2026-07-27） | https://arxiv.org/abs/2607.24331 | A |
| R12 | Y. Su, H. Liu, G. Yu, J. Zhu, "D-Quant: Driftable Entropy Coding for KV Cache Quantization," arXiv 2026（v1 2026-09-17） | https://arxiv.org/abs/2609.19880 | A |
| R13 | J. Kim, J. Park, J. Cho, D. Papailiopoulos, "Lexico: Extreme KV Cache Compression via Sparse Coding over Universal Dictionaries," arXiv 2024-12-12；ICML 2025, PMLR 267:30672–30687 | https://arxiv.org/abs/2412.08890 ；https://proceedings.mlr.press/v267/kim25ae.html | A＋P |
| R14 | Y. Zhu, D. H. Yang, M. M. Amiri, K. Murugesan, T. Pedapati, P.-Y. Chen, "OjaKV: Context-Aware Online Low-Rank KV Cache Compression," arXiv（v1 2025-09-25、v2 2026-04-16）；venue 見 L5a #15 | https://arxiv.org/abs/2509.21623 | A |
| R15 | L. Benfenati *et al.*, "Don't be so Stief! Learning KV Cache low-rank approximation over the Stiefel manifold," arXiv（v1 2026-01-29、v2 2026-05-29） | https://arxiv.org/abs/2601.21686 | A |
| R16 | D. Lesens, B. T. Rakhshan, G. Rabusseau, "KQ-SVD: Compressing the KV Cache with Provable Guarantees on Attention Fidelity," AISTATS 2026, PMLR 300:3556–3564（arXiv v1 2025-12-05） | https://arxiv.org/abs/2512.05916 ；https://proceedings.mlr.press/v300/lesens26a.html | A＋P |
| R17 | S. Ha, J. Lee, Y.-S. Jeon, "KV-COBRA: KV Cache Compression via Co-Optimized Bit-Rank Allocation," arXiv 2026（v1 2026-09-21） | https://arxiv.org/abs/2609.24298 | A |
| R18 | P. Liu, Z.-F. Gao, W. X. Zhao, Y. Ma, T. Wang, J.-R. Wen, "Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression"（DecoQuant），arXiv 2024-05-21；ACL 2024（Long Papers）pp. 2430–2440 | https://arxiv.org/abs/2405.12591 ；https://aclanthology.org/2024.acl-long.133/ | A＋P |
| R19 | Y. Zhang, Y. Liu, H. Yuan, Z. Qin, Y. Yuan, Q. Gu, A. C.-C. Yao, "Tensor Product Attention Is All You Need," NeurIPS 2025 Spotlight（arXiv comment；v1 2025-01-11 … v7 2026-01-11） | https://arxiv.org/abs/2501.06425 | A |
| R20 | T. Klein, J. Kusch, S. Sager, S. Schnake, S. Schotthöfer, "Tucker Attention: A generalization of approximate attention mechanisms," arXiv（v1 2026-03-31） | https://arxiv.org/abs/2603.30033 | A |
| R21 | J. Oldfield *et al.*, "Multilinear Mixture of Experts: Scalable Expert Specialization through Factorization," NeurIPS 2024（arXiv comment） | https://arxiv.org/abs/2402.12550 | A |
| R22 | Q. Lei, Z. Yang, Q. Xu, C. Hua, P. Wen, Q. Huang, "TuckA: Hierarchical Compact Tensor Experts for Efficient Fine-Tuning," arXiv（v1 2025-11-10） | https://arxiv.org/abs/2511.06859 | A |
| R23 | A. Cordoba *et al.*, "DepthWeave-KV: Token-Adaptive Cross-Layer Residual Factorization for Long-Context KV Cache Compression," arXiv（v1 2026-07-07） | https://arxiv.org/abs/2607.06523 | A（可信度見 L5b X8） |
| R24 | H. Tian, Y. Liu, T. Wang, "Every Cache Entry Earns Its Place: Global Allocation of Resolution and Coverage for KV Cache Compression"（GraceKV），arXiv（v1 2026-08-07、v2 2026-10-10） | https://arxiv.org/abs/2608.07001 | A |
| R25 | M. Wagner, C. Roux, M. Zimmer, S. Pokutta, "The KV-Surgeon: Compensation-Aware Offline KV-Cache Compaction," arXiv（v1 2026-10-19） | https://arxiv.org/abs/2610.25933 | A |
| R26 | S. D. Tumkur *et al.*, "Where Should the KV Cache Live? Placement Policies Across GPU, CPU, and SSD for Long-Lived Sessions," arXiv（v1 2026-09-14） | https://arxiv.org/abs/2609.16215 | A |

### 本輪只核書目、內容沿用 L5x 卡片的
| # | 文獻 | URL | 核 | 卡片 |
|:--|:--|:--|:--|:--|
| R27 | MiniCache（A. Liu *et al.*；v1 2024-05-23、v2 2024-09-07） | https://arxiv.org/abs/2405.14366 | A | L5b X1 |
| R28 | KVSharer（Y. Yang *et al.*；v1 2024-10-24；comment：Under Review by ICLR2025） | https://arxiv.org/abs/2410.18517 | A | L5b X2 |
| R29 | CommonKV（Y. Wang, H. Qiao, L. Li, Q. Zhu, W. Che；v1 2025-08-22） | https://arxiv.org/abs/2508.16134 | A | L5b X4 |
| R30 | SwiftKV（A. Qiao, Z. Yao, S. Rajbhandari, Y. He；v3 2025-06-02） | https://arxiv.org/abs/2410.03960 | A | L5b §3.3 |
| R31 | Cross-Layer Attention（W. Brandon *et al.*；v1 2024-05-21） | https://arxiv.org/abs/2405.12981 | A | L5b §3.3 |
| R32 | LCKV（H. Wu, K. Tu；comment：ACL2024 main） | https://arxiv.org/abs/2405.10637 | A | L5b §3.3 |
| R33 | XQuant（H. Yang *et al.*；comment：EMNLP 2025） | https://arxiv.org/abs/2510.11236 | A | L5b X5 |
| R34 | CacheGen（Y. Liu *et al.*；comment：SIGCOMM'24） | https://arxiv.org/abs/2310.07240 | A | L5c、L2 |
| R35 | KVServe（Z. Liu *et al.*；comment：Accepted by SIGCOMM 2026） | https://arxiv.org/abs/2605.13734 | A | L5c |
| R36 | AdaptCache（S. Feng *et al.*；comment：SOSP 2025 BigMem workshop） | https://arxiv.org/abs/2509.00105 | A | L5c |
| R37 | EvicPress（S. Feng *et al.*；v1 2025-12-16） | https://arxiv.org/abs/2512.14946 | A | L5c |
| R38 | Eigen Attention（U. Saxena *et al.*；v2 2024-11-08） | https://arxiv.org/abs/2408.05646 | A | L5a B3 |
| R39 | Loki（P. Singhania *et al.*；comment：NeurIPS 2024） | https://arxiv.org/abs/2406.02542 | A | L5a B2 |
| R40 | Palu（C.-C. Chang *et al.*；v2 2024-11-04） | https://arxiv.org/abs/2407.21118 | A | L5a C1 |
| R41 | ShadowKV（H. Sun *et al.*；v3 2025-04-25） | https://arxiv.org/abs/2410.21465 | A | L5a A1 |

### 查了沒找到（≠不存在）
- 「白化（query／attention 感知）＋跨層或全部層」的 KV 基底或張量分解。
- 每 prompt SVD 的「全部層一組」（W＝L）在 ≥32K 的同位元組比較（xKV 最多 W＝8；JoLT 最多 9 層；2609.28029 只有 Lg＝4）。
- 低秩組合格式在 CPU／SSD／NFS 的實測還原時間、以及 GPU 忙碌下的還原時間。
- 跨層子空間重疊（主角、平均平方 cosine）的**數值**報告，含隨機基準與去平均（JoLT 用它分組但不報值；xKV 只給 CKA）。
- 名為「TuckerMoE」的論文（最接近的是 TD-MoE 程式碼裡的「Whitened Tucker-MoE」字串）。
- TD-MoE 的 arXiv 版本。
