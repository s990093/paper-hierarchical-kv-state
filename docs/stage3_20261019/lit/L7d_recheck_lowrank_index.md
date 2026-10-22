# L7d　重新查新：低秩 K 當選擇性還原的索引（候選 4）＋C1a／C1b＋問題①

> **日期**：2026-10-22（UTC）。**執行者**：文獻 agent。只讀、只上網：沒碰 GPU、沒 import torch／vllm、沒裝套件、沒改任何既有檔案，只寫這一個檔。
> **範圍**：
> - 候選 4：用（RoPE 前）低秩 K 當選擇性還原的索引。
> - C1a：寫入時的分數能不能猜中第 2 輪要的 block。
> - C1b：選擇性還原，只把挑中的 block 從下層搬回。
> - 使用者的問題①：量化＋token 重要性（丟 vs 量化）。
>
> **不重抄**：L1（C1–C29、A10 KVzip、A11 SCBench）、L4（CacheSmith、Cascade）、L5a（ShadowKV、xKV、Loki、LRQK、SALS）、L5d（More Tokens、TailorKV）、`research_20261010_directions/cards/D5_SmartGen.md` 已寫過的，這裡只補**新的事實**和**對候選 4 的重疊判斷**。
>
> **怎麼查的**：
> - 每一篇都用 WebFetch 打開 arXiv abs 頁（或官方會議頁）核對標題、作者、日期、版本、Comments 欄。
> - 細節用 WebFetch 對 arXiv HTML 定向提問。注意：WebFetch 回的是小模型整理過的答案，引號裡的句子**可能被改寫**，數字照回傳抄。
> - 兩篇自己抽 PDF 文字讀了指定段落（tiara venv 的 PyMuPDF，`python -I`，檔案放 scratchpad）：
>   - LOCKS：p.2–3、App. A.2、App. C.2。
>   - IMPRESS：§3.1–3.2、§4.1、§4.3、§5.1。
> - WebSearch 約 18 次。
> - **查不到不等於沒有**（CLAUDE.md §1 第 7 條）：寫「沒查到」的地方都是這個意思。
>
> **閱讀深度**：
> - 〔部分〕＝abs＋HTML 定向問答，或 PDF 指定段落。
> - 〔摘要〕＝只讀 abs 頁或會議頁。
> - **本檔沒有〔全文〕**。
>
> **數字規則**：都是原論文在**自己的**模型、長度、預算下量的，不能跨列比較。〔判讀〕是我的解讀；〔推論：算術〕是我算的，不是任何人量的。
>
> **審查性質**：self-review，不是 cross-model（CLAUDE.md §5）。

---

## 1. 結論

1. **「還原時只把挑中的 block 搬回、其餘留在下層可補載、多輪第 2 輪」這個形狀，沒查到完全相同的前作**。最近的三篇：
   - IMPRESS（FAST'25）、ContiguousKV（2026）：還原時依新請求挑，但靠探針 head 或第一層 K；沒挑到的直接不用；單輪、prefix 只有 4–10K。
   - SmartGen（2026）：P/D 傳輸。重要 block 先送、按需抓、其餘背景補送。
2. **「用比 Quest 更準的 K 摘要挑 page」已經被做掉（decode 時）**：
   - LOCKS：每頁各自的低秩摘要。RULER-32K、b＝128 時 Quest 53.6、LOCKS 80–83。
   - FIER、YAKV：1–2 bit 的 K。
   - SAKI：量了 RoPE 前低秩 K 的召回，用的就是我們的兩個模型。
3. **文獻對「共用／校準基底的低秩 K 打分」給的是負面訊號**：
   - ShadowKV 表 1：Loki 35.52、Quest 76.29。
   - LOCKS：共用基底的 page 召回 58–63%，每頁各自 fit 的是 90.5%。
   - YAKV：同樣的 PCIe 預算下，低秩 K 輸給 1–2 bit 的 K。
   - SAKI：PCA r＝32 的 top-64 召回只有 0.75（Llama）、0.79（Qwen）。
   - → K1 量的「能量」不等於「召回」。
4. **剩下的新意**：多輪第 2 輪、512-token block、還原時一次選完（深層 q 拿不到）、索引的位元組要算進 CPU／SSD／NFS 的搬運、直接拿疊加格式裡的低秩 K 係數當索引（不另外存）、補載。而且實驗一定要同時對照「細 page 的 Quest」和「1–2 bit 的 K」。
5. **最大威脅**：LOCKS（<https://arxiv.org/abs/2607.24555>）。框架上的威脅是 ShadowKV（2410.21465）與 IMPRESS（FAST'25）。

---

## 2. 對照表

**沒有任何一篇判成「完全重疊」**：
- 有「還原時」的都不是低秩 K、也不是多輪。
- 有「低秩／更好的摘要」的都是 decode 時、16-token page。

### 2.1 還原時（從下層搬回）只搬一部分

| 論文 | 年份／出處 | 做了什麼 | 和我們重疊 | 理由（對到原文哪裡） | 深度 |
|:--|:--|:--|:--|:--|:--|
| **IMPRESS** | 2025／FAST'25 | prefix KV 存在 GPU、CPU、SSD 三層。新請求來時，先從磁碟只載 3 個 probe head 的 K，用請求的 query 找重要 token，**只載重要 KV**，其他不用 | **部分重疊** | 還原時、依新 query 挑，和 C1b 同。不同處：<br>• 索引是「少數 head 的完整 K」，不是低秩。<br>• 沒挑到的「discarded」（§3.1），沒有補載。<br>• OPT-6.7B／13B／30B，prefix 4–10K、平均 4.8–5.7K（§5.1）。<br>• 單輪 few-shot。<br>• §3.2：「recall ratios above 80%」時答對率仍掉最多 5%（RTE）、3.3%（SQuAD），和 C1b 的觀察同向 | 部分 |
| **ContiguousKV** | 2026／arXiv | Re-Prefill 時，每個 Period 的第一層載 prefix K、和 query 算出重要 ContiguousChunk（例：16 token），整個 Period 共用這組索引，並跨 Period 預取 | **部分重疊** | 還原時、依 query 挑、以 chunk 為 I/O 單位，和 C1b 同。不同處：<br>• 索引是第一層的 K，不是低秩。<br>• 沒挑到的沒明說會不會補載。<br>• Qwen2.5-7B／14B／32B，prefix 3.8–6K，few-shot 分類，沒有多輪。<br>• 預算 5／10／25／50% 時答對率掉 5.62／3.56／1.21／0.04% | 部分 |
| **SmartGen** | 2026／arXiv（repo 已有卡 D5_SmartGen） | P/D 分離。prefill 端依**離線 profiling** 的 block 重要度先推；decode 端用 InfiniGen／HATA 的索引按需抓；網路空閒時依重要度推剩下的，約 10 個 decode 步內推完 | **部分重疊**（結構） | 「先送重要的、按需補、背景補完」和 C1b 的 (b) 補載同一個形狀。不同處：<br>• 選法是離線位置先驗，不看當下問題。<br>• 情境是 P/D 單請求、LongBench 平均 3–10K，不是多輪還原。<br>• 沒有低秩索引 | 部分 |
| CacheTune | 2026／arXiv | 非前綴 chunk 復用：離線頻域分析挑要重算的 KV，其餘沿用並稀疏傳輸 | 相關但不同 | 選的是「要重算的」，不是「要搬的」；L1 C27 已判 | 摘要 |
| CacheSmith | 2026／arXiv | RAG chunk：query-aware 打分選要重算的 token，其餘全部載入 | 相關但不同 | L4 §1.2 已判：沒有「留在下層待補」 | 摘要（L4 全文） |
| SparseX | 2026／arXiv | 段落級 KV 共用；用 Sparse-Q 索引挑要修正的 key、只重算那些 | 相關但不同 | 選擇性**重算**，不是選擇性搬運 | 摘要 |
| RISK-KV（**10/21 新**） | 2026-10-21／arXiv | 請求遷移或中斷恢復時，挑一些「層組」不存，用跨層 ridge 迴歸重建；省 18.2–33.3% 的歷史 KV，困惑度變化 <0.5% | 相關但不同 | 選的是**層**、依品質風險，不是依 query 選 block；只有 512-token 的歷史 | 摘要 |
| Tutti、HyMCache（abs 標題「A CXL Memory Rack for Multi-Turn LLM Serving」）、LM-CXD、Persistent Q4、SparKV、Predictive Multi-Tier、TierKV、KVDrive | 2026／arXiv | SSD／CXL／裝置端的分層 KV。全量還原、預取，或 decode 時抓關鍵 entry | 相關但不同 | abs 都沒說「依下一輪問題只載一部分」；TierKV 在 prefill 時把 token 分到精確／低秩／flash 三層（行動裝置，細節 NOT_VERIFIED） | 摘要 |

### 2.2 用低秩、量化或雜湊的 K 摘要做稀疏取回（全部是 decode 時）

| 論文 | 年份／出處 | 做了什麼 | 和我們重疊 | 理由 | 深度 |
|:--|:--|:--|:--|:--|:--|
| **LOCKS** | 2026／arXiv（v3 9/29） | 每個 16-token page 存 rank-r 的**頁內**低秩摘要，用 LSE 估每頁的 attention 質量來選頁。KV 可以卸到 Grace 記憶體 | **部分重疊（最大威脅）** | 直接回答「比 Quest 準的低秩 K 摘要」與「共用基底 vs 每頁各自 fit」（表 1、表 C.2、Lemma 2）。不同處：decode 時、page＝16、單輪、post-RoPE | 部分（PDF 段落） |
| **SAKI** | 2026／arXiv（v3 10/12） | 每 head 校準、**RoPE 前**的低秩 K 索引，以「保留分數」為目標；量 top-64 token 召回 | **部分重疊** | 量的正是「RoPE 前低秩 K 當索引的召回」，模型是 Llama-3.1-8B、Qwen2.5-7B。不同處：token 級、4K–64K、沒有任務分數、沒有卸載、沒和 Quest 比 | 部分 |
| **ShadowKV** | 2024／ICML'25 | GPU 放 RoPE 前低秩 K（rank 160），V 放 CPU；用 **post-RoPE 的 chunk 平均（landmark）** 選 chunk，再重建 K、取 V | **部分重疊**（系統框架） | 「低秩 K 在快層、只搬挑中的」已經有了。但選擇用的是 landmark，**不是**拿低秩 K 打分。表 1 也給了 Loki 和 Quest 的對照（§3） | 部分 |
| **YAKV** | 2026／arXiv | 同樣的 PCIe 預算下比 ShadowKV、InfiniGen、ArkVale、LRQK；GPU 放 1–2 bit 的 K，token 級選 | **部分重疊**（索引比較） | 回答「低秩 vs 低位元 K 當索引」：低秩 K 與 landmark 在抽取型任務選錯（Takeaway A、B）。decode 時、單輪 | 部分 |
| **FIER** | 2025／EMNLP'25（abs comment） | 1-bit K（g32）token 級打分，對照 Quest 的 page 選法 | 部分重疊（索引比較） | 低位元 K 勝 Quest 的直接證據。decode 時、≤32K、沒有卸載描述 | 部分 |
| **Self-Indexing KVCache** | 2026／arXiv | 1-bit 符號 VQ 的 K「同時是儲存、也是索引」 | 部分重疊（自帶索引的想法） | 「壓縮後的 K 直接當索引」已有 1-bit 版；低秩版本沒查到 | 摘要 |
| SpeCache | 2025／arXiv | GPU 放 1–2 bit 的 KV 副本，全部 token 用低位元算；只有 top-k（預設 64）換成 16-bit 從 CPU 抓，並用投機 token 預取 | 相關但不同 | 「全部低精度＋挑中的升精度」的 decode 版；§5 的變體 B 要對照它 | 部分 |
| RocketKV（含 -MT） | 2025／ICML'25 | 第二階段 HSA＝Quest 式的 page min/max，加上 SparQ 式只用部分維度；MT 版多輪不刪 | 部分重疊（多輪下 Quest 的表現） | SCBench 多輪時 Quest 遠離 exact top-k（§3 數字）。GPU 常駐、decode 時 | 部分 |
| Loki | 2024／NeurIPS'24 | 每 head 校準 PCA，用前 d 維近似分數選 token | 相關但不同 | 沒和 Quest 比；只有 Jaccard 約 0.9（Llama2-7B、prompt 約 3K）；沒有 ≥32K | 部分 |
| HashAttention | 2024／ICML'25 | 學出來的 Hamming 碼，每 token 每 head 32 bit | 相關但不同 | 要訓練（CLAUDE.md §8 不做）；不卸載。表 1 有 Quest 加量化後的位元數 | 部分 |
| Double Sparsity | 2024／arXiv | 離線校準的重通道「label cache」（K 的 1/16、4-bit）；卸載版 GPU 只放 label cache | 相關但不同 | 低維 K 索引＋卸載。沒和 Quest 比；沒有 token 召回 | 部分 |
| SparQ | 2023／arXiv | 用 query 最大的 r 個分量讀 K 的部分通道來近似分數 | 相關但不同 | 低維打分的一種；多輪數字見 RocketKV | 摘要 |
| TailorKV | 2025／arXiv | 逐層二選一：量化（層 0）或卸載＋token 級 top-K（用 8／12 個關鍵通道打分） | 相關但不同 | L5d S25 已列；沒有同記憶體比較，也沒有多輪 | 部分 |
| ClusterKV | 2024／arXiv | 依語意群召回，KV 卸到 CPU | 相關但不同 | 主張「依位置切 page 會挑錯」；LongBench 上 Quest 43.23／45.59（1K／2K）、ClusterKV 48.34／48.70 | 部分 |
| Quest | 2024／ICML'24 | page（16）min/max 的上界打分 | 基準 | 我們 C1b 用的是 **512-token** 版本 | 摘要（L1） |
| ArkVale | 2024／NeurIPS'24 | page 的外框盒子摘要，CPU 留備份可召回 | 相關但不同 | YAKV 表 3 中挑錯很多（§3） | 摘要 |
| OmniKV | 2025／ICLR'25 | 少數 filter 層挑、其他層沿用 | 相關但不同 | 和 ContiguousKV 的 Period 共用索引同一想法；decode 時 | 摘要 |
| InfiniGen、LRQK、SALS、Fathom、PQCache、RetrievalAttention、MagicPIG、TokenSelect、HiSparse、ReTopK（2607.27692） | 2023–2026 | decode 時各種索引：SVD 扭轉、QK 聯合低秩、latent、位元平面、PQ、ANN、LSH、token 點積、分層快取、索引重用 | 相關但不同 | 都是每步取回；L1／L5a 已有卡的不重寫 | 摘要／L1、L5a |

### 2.3 多輪：第 2 輪要的 KV 能不能預測（C1a）

| 論文 | 年份／出處 | 做了什麼 | 和我們重疊 | 理由 | 深度 |
|:--|:--|:--|:--|:--|:--|
| SCBench | 2024／ICLR'25 | 多輪、多請求共用 context 的基準 | 部分重疊 | §5：「important KVs remain stable within a turn, they vary significantly between queries」。這是定性的 attention map，沒有 block 召回 | 部分 |
| KVzip | 2025／NeurIPS'25 Oral | 和 query 無關、用「重建原文」打分；多 query 下 query-aware 的方法保留 90% 就掉 | 部分重疊 | 寫入時分數的最強候選，**C1a 沒測**。沒有卸載、沒有選擇性載入（論文明說只縮 KV） | 部分 |
| IMPRESS §3.2 | 2025／FAST'25 | 「事先記下重要 token」會漏掉，因為重要度隨 query 變 | 部分重疊 | 和 C1a「寫入時只比隨機好 11–24 點」同向 | 部分 |
| Lookahead Q-Cache、LORE-KV（2610.07643） | 2025／EMNLP'25；2026-10 | 用假的或取樣的後續當 query 打分 | 相關但不同 | 預測的是**同一題**的回答，不是下一輪的新問題 | 摘要 |
| Make Each Token Count | 2026／arXiv | 學出來的保留閘，有多輪對話評測 | 相關但不同 | 要訓練；丟而不是排順序 | 摘要 |
| Rate–distortion view（2607.08032） | 2026／arXiv | 立場文：留不留的訊號多半是 attention 大小或最近程度，會丟掉之後的 query 要的；重複壓縮很少被量 | 相關但不同 | 支持「多輪要另外量」，沒有實驗數字（abs） | 部分（前 100K 字） |

### 2.4 丟 vs 量化：同位元組、多輪（問題①）

| 論文 | 年份／出處 | 做了什麼 | 和我們重疊 | 理由 | 深度 |
|:--|:--|:--|:--|:--|:--|
| More Tokens, Lower Precision | 2024／EMNLP Findings'25 | 同記憶體（token × bit 固定）：「4× token、4 bit」多半贏「1× token、16 bit」；2 bit 崩 | **部分重疊** | 同位元組的丟 vs 量化已做，但只有單輪、≤16K（RULER 4–16K）、Llama-3-8B／Mistral-7B；沒有多輪（L5d S8 已讀全文） | 部分 |
| SCBench 表 4 | 2024／ICLR'25 | 多輪模式：KIVI（2-bit、τ＝1/8）vs SnapKV、PyramidKV、StreamingLLM（τ＝1/32） | **部分重疊** | 多輪下量化比丟穩。但**預算不等**（KIVI 多 4 倍位元組），附錄沒有同壓縮率的掃描 | 部分 |
| KVzip §4.2 | 2025／NeurIPS'25 | KVzip（丟 70%）＋QServe 4-bit KV：124K 時 16.3 GB → 1.2 GB，「negligible」 | 相關但不同 | 是疊加，不是同位元組比較 | 部分 |
| Yuan et al. 基準（2407.01527）、Benchmarking KV-cache opt（2607.05399） | 2024／2026 arXiv | 量化、丟 token 等方法在長文任務上的基準；後者說 KIVI4 最穩、壓縮比不能預測端到端 | 相關但不同 | 單輪；abs 沒說是同位元組 | 摘要 |

---

## 3. 重要論文卡片（只寫對候選 4 有用的新事實）

### 3.1　LOCKS〔部分：abs、HTML、PDF p.2–3／App. A.2／App. C.2〕

- **出處**：Junsung Hwang（單一作者），「LOCKS: Page-Local Compact Key Summaries for Efficient Long-Context Decoding」，arXiv 2607.24555。v1 2026-07-27、v3 2026-09-29；Comments 欄沒有。
- **做法**：
  - 每個 page（B＝16）算 centroid μ_j 與頁內共變異 H_j，取前 r 個特徵向量當摘要。
  - 用摘要重建頁內 logit，估「頁 LSE」（頁的 attention 質量）來選頁，選擇時不讀候選頁的 K／V。
  - 頁完成時才建摘要，不在 decode 的關鍵路徑上。
- **共用基底 vs 每頁各自 fit**（表 1，PDF p.3）：
  - 條件：RULER-32K、Llama-3.1-8B-Instruct、B＝16、b＝128。每頁每 KV head 的選擇器狀態都是 488–490 B。

  | centroid | 基底 | 秩 | 質量 % | 召回 % | 分數 % |
  |:--|:--|--:|--:|--:|--:|
  | 每頁 | 每頁（LOCKS r4i4） | 4 | 85.8 | **90.5** | 80.1 |
  | 每頁 | 整段序列共用 | 20 | 84.4 | 81.3 | 77.0 |
  | 每頁 | 全域校準 4K | 20 | 84.6 | 81.9 | 76.3 |
  | 每頁 | 全域校準 32K | 20 | 84.5 | 81.9 | 75.2 |
  | 序列 | 序列 | 28 | 81.2 | **60.2** | 47.6 |
  | 全域 4K | 全域 4K | 28 | 81.3 | **58.0** | 63.1 |
  | 全域 32K | 全域 32K | 28 | 82.2 | **63.3** | 56.9 |

  - Proposition 1：固定的共用低秩投影會有「盲方向」——兩個頁的 centroid 與投影都相同，真實質量卻可以差到無限大。
- **對 Quest 的理論批評**（Lemma 2，App. A.2）：Quest 的 envelope 分數只看每一維的極值；極值相同的頁，對任何 query 都拿到相同分數。
- **RoPE 前 vs 後**（表 C.2；rank-8、RULER-32K、B＝16、b＝1024；質量／頁召回）：

  | 模型 | pre-RoPE | post-RoPE |
  |:--|:--|:--|
  | Llama-3.1-8B | 0.8936／0.9461 | 0.8935／0.9365 |
  | Qwen3-4B | 0.8770／0.9362 | 0.8767／0.9306 |

  頁內 fit 時，RoPE 前後幾乎一樣。
- **和 Quest 的答對率**（附錄，Llama-3.1-8B；Quest／LOCKS r8i4／r4i4）：

  | 任務 | b＝128 | b＝512 | b＝2048 |
  |:--|:--|:--|:--|
  | RULER-32K | 53.6／83.4／80.1 | 72.1／86.1／86.0 | 82.8／88.0／87.7 |
  | LongBench-v1 | 39.8／46.7／46.4 | 45.9／47.1／47.0 | 46.9／47.2／47.1 |

- **page 大小**：B＝16／32／64 都測了；頁越大召回越低，RULER-32K 上 B＝16 比大頁好 2.3–3.7 分。
- **卸載**：原始 KV 放 Grace 記憶體、GPU 放 hot cache。64K–256K 時吞吐是 dense 的 3.82–4.22 倍。摘要放在哪一層：沒查到明寫。
- **限制**：
  - decode 時、單輪、page＝16；沒有 Qwen2.5。
  - 摘要約佔完整 KV 的 4–10%（秩 2–8）。
  - 單一作者預印本，沒有同儕審查。
- **對候選 4 的意思**〔判讀〕：
  1. 「比 Quest 準的低秩摘要」在 decode 時已成立，這不能當新意。
  2. 我們想用的「校準或每題共用的 RoPE 前基底」，正是 LOCKS 表 1 裡召回最低的那一類（58–63%）；只有保留每頁 centroid 才回到約 81%。
  3. LOCKS 沒測 512-token 的大 block；頁越大越差的趨勢，說明 C1b 的 Quest-512 失敗有一部分可能來自 block 太大，不只是 min/max 本身。

### 3.2　SAKI〔部分〕

- **出處**：Lin Zhang（單一作者），「SAKI: Score-Aware Low-Rank Key Indexing with Random-Matrix Noise Correction for KV Retrieval」，arXiv 2608.03228。v1 2026-08-04、v3 2026-10-12；Comments「16 pages, 5 figures, 4 tables」。
- **設定**：
  - **RoPE 前**的碼，重建後再旋轉（§3）。
  - 每 head 用公開文字校準一次（§6）。
  - 指標：最後 512 個 query 的 **token 級 top-64 召回**，4K 主結果（§3）；§7 到 64K。
- **數字**（中位數，跨 head）：

  | 模型 | r | key PCA | SAKI |
  |:--|--:|--:|--:|
  | Llama-3.1-8B | 16 | 0.679 | 0.731 |
  | Llama-3.1-8B | 32 | **0.748** | 0.799 |
  | Llama-3.1-8B | 64 | 0.833 | 0.876 |
  | Qwen2.5-7B（Instruct 或 1M：NOT_FOUND） | 16 | 0.717 | 0.777 |
  | Qwen2.5-7B | 32 | **0.786** | 0.850 |
  | Qwen2.5-7B | 64 | 0.864 | 0.914 |

  - Llama r＝32 到 64K：PCA 0.696、SAKI 0.748。
  - 只用前 4K 校準、沿用到 64K，召回只少 0.01–0.03。
- **限制**（§9）：「task benchmarks (LongBench, RULER) and contexts beyond 64K are not measured」；沒和 Quest 比；沒有卸載。
- **對候選 4 的意思**〔判讀〕：
  - K1 說「32 維留住大部分**能量**」，但同樣 r＝32 的 PCA，token 級召回只有 0.75–0.79，到 64K 還會降。能量不等於召回。
  - 512-token block 的召回（block 內 attention 加總）會不會比 token 級寬容：NOT_MEASURED。
  - SAKI 的「保留分數」目標函數，可以直接當我們的第二個低秩變體。

### 3.3　ShadowKV（補 L5a 沒寫的兩點）〔部分〕

- 核對：arXiv 2410.21465，v1 2024-10-28、v3 2025-04-25；abs 頁**沒有** Comments 欄。ICML'25 出自 L1／L5a 的核對，本次沒重核。
- **選擇用的不是低秩 K**：原文「Post-RoPE key cache is segmented into chunks, with the mean of each chunk computed as landmarks」「For incoming queries, we first compute the approximate attention scores using the landmarks」。低秩 RoPE 前 K 只拿來**重建**選中的 chunk。設定：rank 160、chunk 8、預算 1.56%（§8.1）。
- **表 1：低秩打分 vs Quest**（RULER 128K 平均、Llama-3.1-8B、稀疏預算 1.56%）：

  | 方法 | 分數 |
  |:--|--:|
  | 完整 attention | 85.53 |
  | ShadowKV | 83.57 |
  | Quest (V) | 79.72 |
  | **Quest** | **76.29** |
  | InfiniGen (V) | 68.76 |
  | InfiniGen | 59.27 |
  | Loki (V) | 58.29 |
  | **Loki** | **35.52** |

  - 基準的超參數（Loki 的維度、Quest 的 page 大小）在我讀到的部分 NOT_FOUND。
  - 〔判讀〕在這個 decode 設定下，校準 PCA 打分（Loki）遠輸 Quest。這是「低秩 K 當索引」最直接的負面前例。
- **要更正的 repo 寫法**（只列，未改原檔，見 §4.3）：`stage3_candidates.md` 候選 4 的「前人」欄寫「ShadowKV 在 decode 時用低秩 K 做稀疏取回」。準確的說法：ShadowKV 把低秩 K 當儲存，用 landmark 做選擇。

### 3.4　YAKV（補 L1 C18 沒抄的表）〔部分〕

- 核對：Bocharnikov 等 5 人，「KV Cache Offloading for Context-Intensive Tasks」，arXiv 2604.08426。v1 2026-04-09、v5 2026-09-01；Comments「Preprint」。
- **Takeaway A**（§3.2、§4.1）：
  - ShadowKV 的截斷 SVD（rank 160）在抽取型任務上「the LLM can no longer reliably select which information it needs」。
  - 「default SVD setting is insufficient … even when we load 10× as many tokens」。
  - rank 512 好一些，但那時壓縮比已經不如 FP8。
  - 不用 SVD（完整 K）時，要載 5–10% 的 token 才追上完整 attention。
- **Takeaway B**：landmark（ShadowKV 的 chunk 平均、ArkVale 的外框盒子）有很多誤選；group size 2–4 也一樣。
- **同 PCIe 預算**（表 3，Llama-3.1-8B-Instruct；Text2JSON／MultiNeedle／LongProc／Loong）：

  | 方法 | Text2JSON | MultiNeedle | LongProc | Loong |
  |:--|--:|--:|--:|--:|
  | 原始 | 46.75 | 94.55 | 22.71 | 37.28 |
  | ShadowKV | 10.46 | 85.73 | 4.59 | 31.25 |
  | InfiniGen | 2.64 | 84.91 | 0.00 | — |
  | ArkVale | 1.19 | 90.09 | 0.06 | — |
  | LRQK | 3.30 | 93.18 | 0.85 | 30.49 |
  | YAKV（1–2 bit K） | 30.60 | 93.36 | 14.02 | 33.53 |

  Qwen3-4B 的 Text2JSON：原始 91.25、ShadowKV 69.38、ArkVale 22.10、LRQK 87.99、YAKV 90.83。
- **沒有的**：召回指標、多輪、prefill 時的選擇。
- 〔判讀〕在同搬運量下，「低位元 K」是比「低秩 K」更強的索引候選。候選 4 的實驗一定要放一組 1–2 bit K。

### 3.5　IMPRESS（補 L1 C27 沒寫的機制）〔部分：PDF §3.1–3.2、§4.3、§5.1〕

- 核對：Weijian Chen 等 9 人（浙大、華為雲），FAST'25，pp. 187–201（USENIX 頁）。
- **機制**：
  - 新請求＝prefix＋query（§4.1）。
  - 從磁碟只載 3 個 probe head 的 K（§4.3），用 query 算 attention，找出重要 token。
  - 依「probe head 之間的 Jaccard 相似度」決定該層要不要啟用；不夠像的層退回載全部 K。原文說這種層 <20%。
  - 然後只載重要 token 的 KV；「unimportant tokens' KVs are discarded」（§3.1）。
- **對 C1a 的旁證**（§3.2）：
  - 事先記下重要 token 不行，因為重要度隨 query 變。
  - 「even with recall ratios above 80%, the omission of vital KVs results in accuracy and F1 score declines of up to 5% for RTE and 3.3% for SQuAD」（OPT-6.7B、OPT-30B）。
- **設定**：OPT-6.7B／13B／30B；prefix 擴到 4K（30B）與 10K，平均 4.8–5.7K；保留率 50%→5%；system prompt 是 2–10 個 few-shot 例子（§5.1）。
- 〔判讀〕IMPRESS 是「還原時、依新 query、只載挑中的」最早的版本。它的索引是「少數 head 的完整 K」。GQA 的 8B 模型、32K 以上、多輪、補載，它都沒有。

### 3.6　ContiguousKV〔部分〕

- 核對：Jing Zou 等 5 人，arXiv 2601.13631，v1 2026-01-20；Comments 沒有。
- **機制**：每個 Period（例：4 或 8 層，原文前後不一致）的第一層載 prefix K、算 query，選出重要 ContiguousChunk（例 c＝16），整個 Period 共用；跨 Period 用上一個 Period 的索引投機預取。Period 之間的相似度平均 52–64%（Qwen2.5-7B）。
- **數字**：
  - 預算 5／10／25／50%：對完整 KV 掉 5.62／3.56／1.21／0.04%；對 IMPRESS 高 7.69／4.81／3.58／1.63%。
  - 5% 時 Re-Prefill 比 IMPRESS 快 3.85 倍。
- **沒有的**：多輪（引言提到同一份文件的後續問題，但實驗沒做）、≥10K 的 prefix、補載。
- 〔判讀〕給 c1b_howto §9「逐層的 Quest」一個現成的做法：還原時**逐段（Period）選**，深層的 q 就不用事先拿到。代價是逐段傳輸、每片更小，正好撞上 I4 量到的每片固定成本。

### 3.7　RocketKV 與 SCBench：多輪下 Quest 離 exact top-k 多遠〔部分〕

- **RocketKV**（Behnam 等 6 人，arXiv 2502.14051，ICML 2025 abs comment）：
  - 表 4（SCBench 多輪、Llama-3.1-8B-Instruct；預算 1024／16384）：

    | 方法 | 1024 | 16384 |
    |:--|--:|--:|
    | Full-KV | 49.9 | 49.9 |
    | Exact-TopK | 45.0 | 50.8 |
    | RocketKV-MT | 44.3 | 50.5 |
    | HSA | 22.1 | 50.0 |
    | **Quest** | **15.7** | 39.3 |
    | SparQ | 13.6 | 48.0 |

  - 表 2（預算 256、LongBench）：Quest 17.8、SparQ 25.7、Exact-TopK 52.6。
  - 附錄 A.3：Quest 改成以 GQA 的 group 為單位。
- **SCBench**（Li 等 11 人，arXiv 2412.10319，「Accepted at ICLR 2025」）：表 4 Llama-3.1-8B 多輪／多請求平均：
  - 完整 48.7／37.2
  - Quest（τ＝1/32；chunk 16、token budget 4096）22.1／19.6
  - RetrievalAttention（top 2000、IVF）29.4／25.8
  - SnapKV 20.9／15.2
  - **KIVI（2-bit、group 32、τ＝1/8）32.0／24.8**
- 〔判讀〕
  - 「query-aware 的 page 摘要在多輪下離 exact top-k 很遠、預算大才追上」已有兩組數字。C1b 的 STOP 和文獻一致，不是我們實作的問題。
  - 但這些都是 decode 時、每步重選；「還原時一次選完」的版本沒查到。

### 3.8　其他

- **FIER**〔部分〕：Dongwei Wang 等 8 人，arXiv 2508.08256，「EMNLP2025 Camera-ready」。
  - 1-bit g32 的 K、token 級。
  - 數字：
    - Mistral LongBench 平均 39.22 vs Quest 36.36；LongChat 30.16 vs 30.11。
    - passkey 10K、預算 32：87% vs 65%（LongChat-7B）。
    - HotpotQA、同載入比 0.125：15.46 vs 11.78（Quest-p16）。
  - 召回只畫在圖上、沒有數字。WebSearch 摘要裡的「44% vs 91%」在 HTML 文字中**沒找到**，不引用。
- **Self-Indexing KVCache**〔摘要〕：Xu Yang 等 5 人，arXiv 2603.14224，2026-03-15。1-bit 符號 VQ 的 K 同時是儲存和索引。abs 沒有數字。
- **SpeCache**〔部分〕：Shibo Jie 等 5 人，arXiv 2503.16163，2025-03-20。
  - GPU 放 2-bit 或 1-bit 的 KV（KIVI），**用在 attention 計算裡**；top-k（預設 64）換成 16-bit。
  - 只到 32K（Mistral）、單一 A6000、沒有多輪。
- **SmartGen**：repo 已有卡（`research_20261010_directions/cards/D5_SmartGen.md`）。本次補兩點：
  - 推送的重要度是 block 級、用 2WikiMultihopQA 離線 profiling 出來的。
  - 背景補送的比例設成「約 10 個 decode 步內送完全部」（§4.1、§4.3）。
- **KVzip §4.2**：和 QServe（W8A8KV4）疊用；**沒有**卸載、沒有選擇性載入、沒和 Quest 比（原文：「our method focuses on explicitly reducing the KV cache size」）。

---

## 4. 缺口清單與「被做掉的部分」

### 4.1 被做掉的（不能再當新意）

| 被做掉的 | 誰做的 | 條件 |
|:--|:--|:--|
| 比 Quest min/max 更準的 page 摘要 | LOCKS（每頁各自低秩）、FIER（1-bit K）、HashAttention（學出來的碼）、RocketKV HSA | decode 時、page 16 左右、單輪 |
| 共用 vs 每頁各自的低秩基底，對選頁的影響 | LOCKS 表 1、Prop. 1 | 同上 |
| RoPE 前低秩 K 索引的 token 召回，在**我們的兩個模型**上 | SAKI | 4K–64K、沒有任務分數 |
| 低秩 K vs 低位元 K 當索引（同搬運量） | YAKV 表 2、3 | decode 時、抽取型任務 |
| 低秩 K 放快層、V 放下層、只搬挑中的 | ShadowKV、xK-SR（L5a）、LRQK（L5a） | decode 時每步 |
| 還原時依新 query 只載重要 KV | IMPRESS、ContiguousKV | 單輪 few-shot、prefix ≤10K、沒挑到的就不用 |
| 先送重要的、按需補、背景補完 | SmartGen | P/D、離線位置先驗 |
| 多輪下 query-aware 稀疏取回 vs exact top-k | RocketKV-MT、SCBench | GPU 常駐、decode 時 |
| 同記憶體下「多 token、低精度」勝「少 token、高精度」 | More Tokens, Lower Precision（L5d S8） | 單輪、≤16K |
| 全部低精度＋挑中的升精度 | SpeCache、TailorKV | decode 時、單輪 |

### 4.2 缺口（沒查到）

| # | 缺口 | 最接近的前作 | 我們已有的 |
|:--|:--|:--|:--|
| G1 | **多輪第 2 輪、還原時一次選完、512-token block** 的索引比較：Quest-512／Quest-16 聚合到 512／低秩（PCA、SAKI 式）／1–2 bit K／探針 head／完整 K 上界，放在同一題組 | LOCKS、YAKV（decode）；IMPRESS（單輪、只有 probe head 一種） | C1b 有兩端：Quest-512（STOP）與 FULLK（Qwen 放回 1/4 不掉） |
| G2 | 還原時**深層 q 拿不到**（C1b 的 QUEST_MEAN vs QUEST_BLIND），以及「逐段選」能不能解 | ContiguousKV（逐 Period，單輪）、IMPRESS（逐層） | C1b 量了 MEAN 與 BLIND |
| G3 | **索引的位元組要算進下層搬運**（CPU／SSD／NFS），配合「搬運 ≤30%」這類預算 | 所有 decode 時的論文都假設索引常駐 GPU、不計搬運 | c1b_howto §2.5、§2.7 的位元組與還原時間公式 |
| G4 | **疊加格式裡的低秩 K 係數直接當索引**（不另外存） | Self-Indexing KVCache（1-bit、decode）；ShadowKV 存低秩 K 但用 landmark 選 | X 系列的 `xl:k22:v34+q:int4_kivi` 格式 |
| G5 | 還原情境下的**補載**：第 2 輪生成時才發現漏了，能補回多少 | ArkVale、SmartGen（品質沒和不補比） | C1b (b)：+1～+14 點但補不回 ε |
| G6 | 寫入時分數（KVzip 類）拿來**排還原或預取順序**，而不是丟 | KVzip（只丟）；L1 G4 | C1a 測了 KNORM／KEYDIFF／SNAP，**沒測 KVzip** |
| G7 | 丟 vs 量化：**同位元組、≥32K、多輪第 2 輪** | More Tokens（單輪 ≤16K）、SCBench（預算不等） | X1、X4d：丟一半 block，第 2 輪掉 17–19 點 |
| G8 | AMD／ROCm 上任何一個索引法 | 沒有（L1 已記） | 我們的 harness 是純 PyTorch |

### 4.3 對 repo 既有文件的更正（只列，未改原檔）

| 文件 | 原寫法 | 核對後 |
|:--|:--|:--|
| `stage3_candidates.md` 候選 4 的「前人」欄 | 「ShadowKV 在 decode 時用低秩 K 做稀疏取回；還原時選 block 的版本沒找到」 | ShadowKV 的**選擇**用 post-RoPE 的 chunk 平均（landmark），低秩 K 只用來重建（§3.3）。真正拿低秩 K **打分**的是 Loki、LRQK、SALS、SAKI，以及 LOCKS（每頁各自）。還原時選 block 的：IMPRESS、ContiguousKV（不是低秩）。 |
| L1 §0 第 2 點 | 「快層必須留一份 key 摘要：Quest 的 min/max、ShadowKV 的低秩 K…」 | 同上：ShadowKV 的摘要是 landmark＋outlier。 |
| `results_analysis.md` §4.10、`experiment_log.md` C1b 節 | 「用 RoPE 前的低秩 K（K1：32 維就留住大部分能量）當更準又便宜的挑選索引」 | 能量 ≠ 召回：同模型 PCA r＝32 的 token 召回 0.75／0.79（SAKI）；共用基底的 page 召回 58–63%（LOCKS）；ShadowKV 表 1 的 Loki 遠輸 Quest。「更準」要實驗證明，不能從 K1 推。 |
| L5a §7 G-L3 停止條件 | 「recall@k 比 Quest 的 min／max 摘要高不到 5 個百分點…→ 不用」 | 建議對照組加 Quest-16 聚合與 1–2 bit K。只贏 Quest-512 不夠：Quest-512 的失敗可能有一部分來自 block 太大（LOCKS 的 page 大小結果，以及 Lemma 2）。 |

---

## 5. 建議的定位

### 5.1 不要押的

- **「低秩 K 比 Quest 準」**：LOCKS 已在 decode 時做掉；而且文獻對「共用基底的低秩打分」偏負面（ShadowKV 表 1 的 Loki、LOCKS 表 1、YAKV）。
- **「token 重要度可以拿來丟」**：More Tokens、SCBench、我們的 X1／X4d 都指向量化支配丟（問題①）。

### 5.2 可以押的〔判讀〕

**形狀**：多輪對話**還原時**的選擇性還原，比較各種索引「每多花一個位元組，離完整 K 上界還差多少」。四個限制是 decode 時的論文沒有的：
- (a) 深層 q 拿不到（BLIND）；
- (b) I/O 單位是 512-token block；
- (c) 索引本身也要從下層搬、算進搬運；
- (d) 沒挑到的留在下層可補載。

低秩這一角只在 **G4「自帶索引」** 有獨特性：疊加格式本來就存了 RoPE 前、跨層的低秩 K 係數，拿它打分不用另外存。

### 5.3 位元組帳

〔推論：算術〕每 token、每（層, KV head），BF16 KV＝2×128×2＝512 B。比例和模型無關；MiB 是 32K token。

| 索引 | 每 token 每（層, head） | 佔 BF16 KV | Llama 32K | Qwen 32K |
|:--|--:|--:|--:|--:|
| Quest，512-token block（C1b 用的） | 1 B | 0.20% | 8 MiB | 3.5 MiB |
| Quest，16-token page | 32 B | 6.25% | 256 MiB | 112 MiB |
| RoPE 前低秩 r＝32，BF16 | 64 B | 12.5% | 512 MiB | 224 MiB |
| 同上，INT4（不含 scale） | 16 B | 3.1% | 128 MiB | 56 MiB |
| r＝16，INT4 | 8 B | 1.6% | 64 MiB | 28 MiB |
| 1-bit K（不含 scale） | 16 B | 3.1% | 128 MiB | 56 MiB |
| 2-bit K | 32 B | 6.25% | 256 MiB | 112 MiB |
| 疊加格式 `xl:k22` 的 K 係數，INT4（每 head 等效秩 22；不含 scale 與基底） | 11 B | 2.1% | 88 MiB | 38.5 MiB |
| 完整 K（FULLK 上界） | 256 B | 50% | 2 GiB | 896 MiB |

- C1b 的 k＝25%：不含摘要的初始搬運約 0.268（c1b_howto §2.5 的 0.2698 減掉 Quest-512 摘要）。
- 所以在「搬運 ≤30%」這條規則下，索引只剩約 3% 的空間：
  - r＝16 INT4、`xl` K 係數放得下；
  - r＝32 INT4、1-bit K 是邊界；
  - Quest-16、2-bit K、r＝32 BF16 放不下（除非索引常駐 GPU、不算搬運）。

### 5.4 最小實驗（建議，**沒跑**）

**做法**：
- 在 C1b 同一題組（DS2 兩輪、32K、seed 70000）上，先只做 C1a 式的答案 block recall@16，再對最好的兩種做 C1b 的 k＝25% (a)／(b)。
- 對照組：
  - Quest-512、FULLK（已有）；
  - Quest-16 聚合到 512；
  - PCA r＝16／32（INT4）；
  - SAKI 式的保留分數基底（選用）；
  - 1-bit、2-bit K；
  - `xl:k22` 的 K 係數；
  - 每組分 MEAN 與 BLIND 兩種 q。

**事先寫死的停止條件（建議）**：
- 最好的「可部署」索引（BLIND q、索引位元組計入）在 k＝25% (b) 時不在 ε 內，**或**要超過 30% 搬運才進 ε → 候選 4 停。
- 低秩組在同位元組下沒贏 1–2 bit K（配對差 CI 下界 ≤0）→ 低秩這一角只剩 G4「自帶索引」可講；索引改用低位元 K。
- 文獻給的先驗是「會輸」（YAKV、LOCKS），所以這一條要先寫死。

### 5.5 變體 B：還原時「全部低精度＋挑中的升精度」〔判讀〕

- 先把疊加壓縮的全部 KV 搬回（約 BF16 的 1/12），用它跑第 2 輪問題的 prefill，**拿到近乎真實的 q 與 attention**；再只把挑中的 block 升回 BF16。
- 這同時解掉 G2（BLIND q）和 G3（索引位元組）。decode 版已有 SpeCache、TailorKV；多輪還原版沒查到。
- 但 X4 顯示 8 倍疊加已接近無損、12 倍只掉 1–3 點，升精度能救的空間可能很小，要先算上界再決定。

### 5.6 問題①的定位

- 「同位元組下丟輸給量化」：More Tokens 已在單輪做過。
- 我們的增量是 ≥32K、多輪第 2 輪、兩個 GQA 模型（G7）。新意中等，**適合當支撐結果，不適合當主貢獻**。
- 重要度的正確用途寫成「還原時挑要搬或要升精度的」，和候選 4 綁在一起。

---

## 6. 參考文獻

- 「核」✓＝本次（2026-10-22）用 WebFetch 打開 abs 或官方頁，核過標題、作者、日期。
- venue 只寫 abs 的 Comments／journal_ref 或官方頁上有的，其餘寫「未列」。

| # | 論文 | URL | 日期／版本 | venue（來源） | 核 | 深度 |
|:--|:--|:--|:--|:--|:--|:--|
| 1 | LOCKS（J. Hwang） | https://arxiv.org/abs/2607.24555 | v1 2026-07-27；v3 2026-09-29 | 未列 | ✓ | 部分（PDF 段落） |
| 2 | SAKI（L. Zhang） | https://arxiv.org/abs/2608.03228 | v1 2026-08-04；v3 2026-10-12 | 未列 | ✓ | 部分 |
| 3 | ShadowKV（H. Sun 等 9 人） | https://arxiv.org/abs/2410.21465 | v1 2024-10-28；v3 2025-04-25 | abs 沒有 Comments；ICML'25 見 L1／L5a | ✓ | 部分 |
| 4 | KV Cache Offloading for Context-Intensive Tasks（YAKV；A. Bocharnikov 等 5 人） | https://arxiv.org/abs/2604.08426 | v1 2026-04-09；v5 2026-09-01 | 「Preprint」 | ✓ | 部分 |
| 5 | IMPRESS（W. Chen 等 9 人） | https://www.usenix.org/conference/fast25/presentation/chen-weijian-impress | 2025-02 | FAST'25（USENIX 頁） | ✓ | 部分（PDF 段落） |
| 6 | ContiguousKV（J. Zou 等 5 人） | https://arxiv.org/abs/2601.13631 | v1 2026-01-20 | 未列 | ✓ | 部分 |
| 7 | SmartGen（X. Luo 等 4 人） | https://arxiv.org/abs/2607.28150 | v1 2026-07-30 | 未列 | ✓ | 部分 |
| 8 | CacheTune（Adaptive KV Cache Reuse…；F. Li 等 7 人） | https://arxiv.org/abs/2605.24022 | v1 2026-05-20；v2 2026-10-04 | 未列 | ✓ | 摘要 |
| 9 | CacheSmith（X. Ma 等 4 人） | https://arxiv.org/abs/2610.18925 | v1 2026-10-13 | 未列 | ✓ | 摘要（L4 全文） |
| 10 | SparseX（Q. Zhang 等 8 人） | https://arxiv.org/abs/2606.01751 | v1 2026-06-01；v2 2026-06-07 | 未列（WebSearch 說 NSDI'26：NOT_VERIFIED） | ✓ | 摘要 |
| 11 | RISK-KV（Wang Yuekai） | https://arxiv.org/abs/2610.29544 | v1 2026-10-21 | 未列 | ✓ | 摘要 |
| 12 | Quest（J. Tang 等 6 人） | https://arxiv.org/abs/2406.10774 | v1 2024-06-16；v2 2024-08-26 | ICML 2024（Comments） | ✓ | 摘要 |
| 13 | Loki（P. Singhania 等 5 人） | https://arxiv.org/abs/2406.02542 | v1 2024-06-04；v2 2024-11-07 | NeurIPS 2024（Comments） | ✓ | 部分 |
| 14 | InfiniGen（W. Lee 等 4 人） | https://arxiv.org/abs/2406.19707 | v1 2024-06-28 | OSDI 2024（Comments） | ✓ | 摘要 |
| 15 | ArkVale（R. Chen 等 10 人） | https://neurips.cc/virtual/2024/poster/96635 ；https://mlanthology.org/neurips/2024/chen2024neurips-arkvale | 2024 | NeurIPS 2024（會議頁） | ✓ | 摘要 |
| 16 | OmniKV（J. Hao 等 8 人） | https://proceedings.iclr.cc/paper_files/paper/2025/hash/da1131a86ac3c70e0b7cae89c3d4df22-Abstract-Conference.html | 2025 | ICLR 2025（proceedings） | ✓ | 摘要 |
| 17 | RetrievalAttention（D. Liu 等 14 人） | https://arxiv.org/abs/2409.10516 | v1 2024-09-16；v3 2024-12-31 | 未列 | ✓ | 摘要 |
| 18 | PQCache（H. Zhang 等 8 人） | https://arxiv.org/abs/2407.12820 | v1 2024-07-01；v2 2025-03-30 | 未列 | ✓ | 摘要 |
| 19 | ClusterKV（G. Liu 等 5 人） | https://arxiv.org/abs/2412.03213 | v1 2024-12-04；v2 2025-06-14 | 未列 | ✓ | 部分 |
| 20 | MagicPIG（Z. Chen 等 11 人） | https://arxiv.org/abs/2410.16179 | v1 2024-10-21；v4 2024-12-18 | 未列 | ✓ | 摘要 |
| 21 | HashAttention（A. Desai 等 6 人） | https://arxiv.org/abs/2412.14468 | v1 2024-12-19；v2 2025-06-03 | ICML'2025（Comments） | ✓ | 部分 |
| 22 | RocketKV（P. Behnam 等 6 人） | https://arxiv.org/abs/2502.14051 | v1 2025-02-19；v3 2025-08-13 | ICML 2025（Comments） | ✓ | 部分 |
| 23 | SparQ Attention（L. Ribar 等 6 人） | https://arxiv.org/abs/2312.04985 | v1 2023-12-08；v6 2024-09-04 | 未列 | ✓ | 摘要 |
| 24 | Double Sparsity（S. Yang 等 5 人） | https://arxiv.org/abs/2408.07092 | v1 2024-08-11；v2 2024-08-18 | 未列 | ✓ | 部分 |
| 25 | TokenSelect（W. Wu 等 9 人） | https://arxiv.org/abs/2411.02886 | v1 2024-11-05；v4 2025-10-09 | EMNLP2025（Comments） | ✓ | 摘要 |
| 26 | LRQK（T. Li 等 5 人） | https://arxiv.org/abs/2510.23649 | v1 2025-10-25；v3 2025-12-23 | NeurIPS 2025（journal_ref） | ✓ | 摘要（L5a 全文） |
| 27 | SALS（J. Mu 等 6 人） | https://arxiv.org/abs/2510.24273 | v1 2025-10-28 | 未列 | ✓ | 摘要 |
| 28 | Fathom（V. Kalyanarangan） | https://arxiv.org/abs/2609.17652 | v1 2026-09-15；v3 2026-09-18 | 未列 | ✓ | 摘要 |
| 29 | SpeCache（S. Jie 等 5 人） | https://arxiv.org/abs/2503.16163 | v1 2025-03-20 | 未列 | ✓ | 部分 |
| 30 | FIER（D. Wang 等 8 人） | https://arxiv.org/abs/2508.08256 | v1 2025-05-28；v2 2025-09-16 | 「EMNLP2025 Camera-ready」 | ✓ | 部分 |
| 31 | Self-Indexing KVCache（X. Yang 等 5 人） | https://arxiv.org/abs/2603.14224 | v1 2026-03-15 | 未列 | ✓ | 摘要 |
| 32 | TailorKV（D. Yao 等 7 人） | https://arxiv.org/abs/2505.19586 | v1 2025-05-26；v2 2025-05-27 | 未列（WebSearch 說 Findings ACL'25：NOT_VERIFIED） | ✓ | 部分 |
| 33 | HiSparse（Z. Xie 等 6 人） | https://arxiv.org/abs/2608.07009 | v1 2026-08-07 | 未列 | ✓ | 摘要 |
| 34 | xKV（C.-C. Chang 等 11 人） | https://arxiv.org/abs/2503.18893 | v1 2025-03-24；v2 2026-05-27 | ICML 2026（Comments） | ✓ | 摘要（L5a 全文） |
| 35 | Recall Before You Rank／ReTopK（W. Yao 等 8 人） | https://arxiv.org/abs/2607.27692 | v1 2026-07-30 | 未列 | ✓ | 摘要 |
| 36 | SCBench（Y. Li 等 11 人） | https://arxiv.org/abs/2412.10319 | v1 2024-12-13；v2 2025-03-11 | 「Accepted at ICLR 2025」 | ✓ | 部分 |
| 37 | KVzip（J.-H. Kim 等 6 人） | https://arxiv.org/abs/2505.23416 | v1 2025-05-29；v2 2025-09-30 | 「NeurIPS 2025 Oral」 | ✓ | 部分 |
| 38 | More Tokens, Lower Precision（J. Zhang 等 9 人） | https://arxiv.org/abs/2412.12706 ；https://aclanthology.org/2025.findings-emnlp.429/ | v1 2024-12-17；v2 2025-02-20 | Findings of EMNLP 2025, pp. 8092–8105（ACL Anthology） | ✓ | 部分 |
| 39 | KV Cache Compression, But What Must We Give in Return?（J. Yuan 等 12 人） | https://arxiv.org/abs/2407.01527 | v1 2024-07-01；v2 2024-10-08 | 未列 | ✓ | 摘要 |
| 40 | Benchmarking KV-Cache Optimizations…（N. Agrawal、R. Mayer） | https://arxiv.org/abs/2607.05399 | v1（abs 頁寫 2026-05-03） | 未列 | ✓ | 摘要 |
| 41 | What to Keep, What to Forget（A. G. Colaco、N. Lahjouji） | https://arxiv.org/abs/2607.08032 | v1 2026-07-09 | 未列 | ✓ | 部分（HTML 前 100K 字） |
| 42 | Lookahead Q-Cache（Y. Wang 等 7 人） | https://arxiv.org/abs/2505.20334 | v1 2025-05-24；v2 2025-11-17 | 「Accepted by EMNLP 2025 Main」 | ✓ | 摘要 |
| 43 | Monte Carlo Estimation for KV Cache Eviction／LORE-KV（A. Bilal 等 8 人） | https://arxiv.org/abs/2610.07643 | v1 2026-10-06；v2 2026-10-14 | 未列 | ✓ | 摘要 |
| 44 | Make Each Token Count（N. Bui 等 4 人） | https://arxiv.org/abs/2605.09649 | v1 2026-05-10 | 未列 | ✓ | 摘要 |
| 45 | KVDrive（J. Lin 等 8 人） | https://arxiv.org/abs/2605.18071 | v1 2026-05-18 | 未列 | ✓ | 摘要 |
| 46 | TierKV（Z. Shu 等 7 人） | https://arxiv.org/abs/2609.21172 | v1 2026-09-18；v2 2026-09-29 | 未列 | ✓ | 摘要 |
| 47 | HyMCache（H. Jang 等 5 人；abs 標題「A CXL Memory Rack for Multi-Turn LLM Serving」） | https://arxiv.org/abs/2607.18141 | v1 2026-07-20；v3 2026-08-05 | 未列 | ✓ | 摘要 |
| 48 | Tutti（S. Qiu 等 9 人） | https://arxiv.org/abs/2605.03375 | v1 2026-05-05 | 未列 | ✓ | 摘要 |
| 49 | LM-CXD／Bridging LLM Serving and CXL-SSDs（H. Chung 等 6 人） | https://arxiv.org/abs/2609.26828 | v1 2026-09-20 | 未列 | ✓ | 摘要 |
| 50 | Persistent Q4 KV Cache（Y. P. Shkolnikov） | https://arxiv.org/abs/2603.04428 | v1 2026-02-17 | 未列 | ✓ | 摘要 |
| 51 | SparKV（H. Liu 等 6 人） | https://arxiv.org/abs/2604.21231 | v1 2026-04-23；v3 2026-10-08 | Comments「IoTJ」 | ✓ | 摘要 |
| 52 | Predictive Multi-Tier Memory Management（S. R. Ganjihal） | https://arxiv.org/abs/2604.26968 | v1 2026-04-19；v2 2026-08-07 | 未列 | ✓ | 摘要 |
| 53 | ScoutAttention（Q. Zhang 等 8 人） | https://arxiv.org/abs/2603.27138 | v1 2026-03-28 | DAC 2026（Comments） | ✓ | 摘要 |

**只在 WebSearch 結果看到、沒打開 abs 的（不引用、標 NOT_VERIFIED）**：
- LookaheadKV（arXiv 2603.10899，搜尋摘要說 ICLR 2026、要 LoRA 訓練）
- HATA（SmartGen 引用的雜湊索引）
- SGLang issue #45128、#27574

**本次沒做的**：
- 沒看任何論文的圖；只出現在圖上的數值（FIER 的召回、LOCKS 圖 5）沒有引用。
- LOCKS 附錄 C.3 只讀了表頭。
- SAKI 的 §7 只拿到 4K 與 64K 的絕對值。
- ShadowKV 基準的超參數 NOT_FOUND。
- TierKV、KVDrive、HyMCache 的選擇機制只有 abs，NOT_VERIFIED。
