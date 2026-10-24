# L8a　查新：漸進式／嵌入式讀回（progressive / embedded restore）＋新問題決定要精修哪些區塊

> **日期**：2026-10-24（UTC）
> **執行者**：文獻 agent。只上網、只讀檔；**沒有跑任何 GPU 程式、沒有安裝套件、沒有改 repo 裡任何既有檔案**，只寫了本檔。
> **怎麼查的**：
> - WebSearch 約 50 次，其中約 42 次是主題查詢、8 次是書目核對。
> - 用 WebFetch 讀 arXiv abs 頁。另外用 curl 抓 arXiv abs 頁，從 meta 欄位取標題、第一作者、作者數、v1 日期、Comments 欄。古典文獻用 Crossref API 核對。
> - 最接近的論文用 curl 從 arxiv.org 抓 PDF，再用 `pdftotext -layout` 轉文字閱讀，共 13 篇：2610.11245、2503.18869、2609.17652、2505.02922、2608.02901、2403.04643、2503.23294、2604.08426、2508.08256、2402.18096、2406.02542、2602.08005、2310.07240。另有 2606.08635 只讀了摘要與 grep 片段。
> - PDF 放在 session scratchpad：`/private/tmp/claude-501/-Users-hungwei-Desktop-Proj-paper-hierarchical-kv-state--claude-worktrees-kv-cache-tiering-compression-2ac2e6/8290b619-6d48-41a3-947c-454b66072a3f/scratchpad/l8a_pdfs/`。**暫存，不進 repo**。
> - 下載內容一律當資料讀，沒有執行裡面任何東西。
>
> **本檔是 self-review，不是 cross-model review**（CLAUDE.md §5）。
>
> **重要警告**：最接近的威脅 ReadKV（2610.11245，2026-10-08 上傳，距今 16 天）是在**第 50 次搜尋左右**才找到的。所以本檔的「沒找到」是弱證據（CLAUDE.md 規則 7）。**建議投稿前兩週再查一次**。

**證據等級**

| 標記 | 意思 |
|:--|:--|
| 〔全文〕 | PDF 抽字後讀過相關各節。頁碼是 **PDF 實體頁**（數 form feed），不是論文印的頁碼。只讀文字，**圖上的數值沒讀**。 |
| 〔部分〕 | 只讀了指定段落（grep 定位），或讀網頁。 |
| 〔摘要〕 | 只讀 arXiv abs 頁或官方摘要頁。 |
| 〔標題〕 | 只在搜尋結果或別篇論文的參考文獻裡看到，**NOT_VERIFIED**。 |
| 〔判讀〕 | 我對原文的解讀。 |
| 〔推論〕 | 我的推測。 |
| NOT_FOUND | 這輪沒找到，不等於不存在。 |

不同模型、硬體、context 的數字**只並列、不排名**。

---

## 0. 結論

1. **完全相同的組合沒找到（NOT_FOUND）。** 這個組合是：
   - (a) KV 只在慢速層（SSD／NFS）存**一份**巢狀表示；
   - (b) **讀回時**先把所有區塊的 base 載入，用**新問題在 base 上做 prefill** 得到的注意力挑區塊；
   - (c) 只為挑中的區塊抓精修層，可在 TTFT 期限內做 anytime 排程；
   - (d) 沒有區塊會遺失；
   - (e) 可以和 Cake 合用。

   但**概念核心「存一份漸進碼，由查詢決定每塊讀多深，沒讀的精修層保留給之後」已經被發表了**：ReadKV〔全文〕。所以本題的新意**不在表示法**，在**情境與機制**：
   - 讀回邊界：多輪續問時從慢速層讀回；
   - 區塊粒度；
   - **在 base 上做 prefill 取得選擇訊號**；
   - TTFT 期限；
   - 和重算合用。
2. **前三大威脅**：
   - **ReadKV（arXiv 2610.11245）**〔全文〕：每個 K／V 純量存成 8-bit 漸進碼。decode 時，先用 query 分配 key 各 channel 讀幾位，再用「重建的 K 算出的注意力」分配 value 各 token 讀幾位（p.1 摘要、§3、p.2 Fig.1）。Discussion 明說可延伸到「慢速大記憶體放少讀的精修層」（p.15）。
   - **Xie et al. 2025（arXiv 2503.18869）**〔全文〕：KV 以 bit-plane 存在 DRAM。用 Quest 排序 page，前 5 頁讀 BF16、接下來讀 FP8／FP4 前綴，由記憶體控制器只解需要的 plane（Table II p.3；§III p.4）。
   - **Fathom（arXiv 2609.17652）**〔全文〕：4-bit K 以 channel-major bit-plane 存在 **host memory**，每個 query 決定每個 channel 讀幾個 plane，用來做 top-k 掃描（摘要 p.1；§3.2 p.4；Table 1 p.3）。
3. **其次的威脅**：
   - **RetroInfer**（2505.02922，PVLDB 2026）：GPU 存每群的 centroid 與 value 總和，用來**估計**未取回群的注意力；重要群從 CPU 取精確 KV（p.6 Eq.2）。
   - **AnchorKV**（2608.02901）：每個 token 都有 anchor 投影當 base，只有部分 token 存量化殘差。選擇在 prefill 結束時用 observation window 決定（p.1、§3.3 p.3）。
   - **QAQ**（2403.04643）：CPU 留未量化副本，位元數需要升高時取回再量化（p.5）。
   - **SpeCache**（2503.16163，本 repo 已涵蓋）：decode 版的「全部低位元＋取回 top-k 高精度」。
4. **還剩的缺口**（一句話）：沒找到任何論文在**多輪讀回邊界**，用**新問題在粗表示上做 prefill 的注意力**，決定從**慢速層（SSD／NFS）**為哪些**區塊**補抓精修層，並在 **TTFT 期限**下和**重算（Cake）**一起排程。上面這些論文全部是 **decode-step**、**HBM／DRAM／host memory**；ReadKV 與 Fathom 的 query 都來自**精確的 dense prefill 或精確 decode**，不是粗表示上的 forward。
5. **Q4（粗表示上的注意力能不能可靠地排區塊）證據兩面都有**：
   - **正面**：
     - FIER：1-bit K 的 top-64 recall 91%，Quest page 16／8 是 44%／67%（Fig.6 p.6）；
     - Loki：PCA 25% 維度 Jaccard 約 0.9（Fig.6 p.9）；
     - Fathom：逐 token 低位元掃描在 RULER 上離 exact top-k oracle 0.008（32k）／0.025（128k）以內（§5.5 p.9）。
   - **反面**：
     - 2604.08426：**低秩 K（ShadowKV SVD rank 160）在高資訊密度任務上選錯**，多載 10 倍 token 仍追不上（§4.1 p.5）；2-bit 逐 token 量化反而優於低秩代理（§4.3 p.7）。
     - Fathom：**區塊 landmark** 在 128k 掉 0.130（Table 7 p.10）。
     - Fathom：真實 coding-agent session 上，最好的掃描與 exact top-k 的 step agreement 也只有 0.60（Table 8 p.10）。
   - 〔推論〕我們的 base 若只放低秩係數，**排序可能是最弱環節**。base 應保留逐 token 的 1–2-bit K（排序用），低秩改放在 V 或精修層。
6. **沒有任何論文量過「query 本身是在粗 KV 上算出來的」時的選擇 recall**。
   - 各篇都用精確 query：ReadKV 明寫 Dense prompt prefill（p.10）；FIER、Fathom、Loki 都用精確 decode query。
   - 在 base 上做 prefill，第 l 層的誤差會傳到第 l+1 層的 query。這是**必須自己量的 NOT_MEASURED**，也是本題最乾淨的新知識點。
7. **審稿人一定會要的對照**（細節見 §6）：
   - ReadKV 式「同一漸進碼、讀取預算相同」的**查詢無關**分配（同 bytes 的 uniform 前綴）；
   - **寫入時**依重要性分配（AnchorKV／MiKV／SpectrumKV 式）；
   - **便宜索引選擇**：Quest 512、1–2-bit K 掃描（FIER／YAKV／Fathom 式）、ShadowKV landmark；
   - **oracle 選擇**（用完整 K）；
   - 選擇性讀回、直接丟棄其餘區塊（我們已量）；
   - 全部 BF16 載入、Cake、uniform 壓縮後全部載入（xKV 8×）；
   - SpeCache 式 decode 時逐步取回。
   - 任務必須包含**高資訊密度任務**（Text2JSON／LongProc／Loong）與**多輪**（SCBench）或 **agent session**，不能只用 RULER／LongBench。
8. **漸進讀回的構想本身早有人提出**：CacheGen §9 寫了「像 SVC 一樣先送低品質 KV，再送差值」的 future work（p.12）〔全文；本 repo 已涵蓋 CacheGen，但這句是新事實〕。所以論文不能宣稱「首次提出漸進式 KV 讀回」。能宣稱的只有**查詢導向的區塊精修**與**讀回邊界的機制和量測**。

---

## 1. 對照表

重疊等級的定義：
- **完全重疊**：同一個情境（讀回時、慢速層、新問題）＋同一機制（全部讀 base、依新問題補精修）。
- **部分重疊**：機制核心相同，但情境不同（decode／HBM／寫入時）。
- **相關但不同**：只共享一個元素。

| # | 論文 | 年份／venue | 做什麼 | 重疊等級 | 理由（原文位置） | 深度 |
|:--|:--|:--|:--|:--|:--|:--|
| 1 | **ReadKV**（Bhandari 等 4 人） | 2026-10-08 v1；preprint（Comments 只寫頁數） | 8-bit 巢狀漸進碼；decode 時 query 分配 K channel 前綴深度，再用重建 K 的注意力分配 V token 前綴深度 | **部分重疊（最高）** | 表示法與「讀多少由查詢決定、沒讀的保留」相同（p.1、p.2 Fig.1、§3）。但：decode-step、HBM；所有方法都用 Dense prompt prefill（p.10）；粒度是純量／channel／token，不是區塊；慢速層只在 Discussion 提一句（p.15）；沒有 TTFT 期限、沒有重算 | 〔全文〕 |
| 2 | **Xie et al.**（Rui Xie 等 7 人） | 2025-03-24 v1；venue NOT_VERIFIED | 記憶體控制器把 KV 以 bit-plane 存放；Quest 排序的 page 依名次給 BF16／FP8／FP4，只抓需要的 plane | **部分重疊** | 「查詢導向的 page 排序＋從 bit-plane 存放讀不同深度」已有（Table II p.3；§III p.4）。但：排名外的 page 不讀；decode；DRAM；硬體模擬；沒有讀回與多輪 | 〔全文〕 |
| 3 | **Fathom**（Kalyanarangan，1 人） | 2026-09-15 v1；preprint | host memory 的 4-bit K 以 bit-plane 存；每個 query 用 reverse water-filling 決定每個 channel 讀幾個 plane，再做 top-k | **部分重疊** | 巢狀存放＋逐查詢讀取深度＋放在慢速層（摘要 p.1；§3.2–3.6 p.4–5）。但：只用在**選擇用的 K 索引**；選到的列讀精確 KV，沒選到的不參與；decode；沒有讀回／SSD（§8 p.12） | 〔全文〕 |
| 4 | **RetroInfer**（Yaoqi Chen 等 19 人） | 2025-05-05 v1；PVLDB 19(5):1016–1031, 2026 | GPU 存 centroid、value 總和、群大小；重要群從 CPU 取精確 KV；中等群用 centroid 估計注意力 | **部分重疊** | 「粗估多數＋精確取回少數」，而且精確資料在 CPU（p.2 §1；p.6 Eq.2–3）。但：粗表示是群摘要，**不是精確 KV 的前綴**（不能補精修）；三區只涵蓋約 25% context（p.9）；decode；索引在同一請求的 prefill 建 | 〔全文〕 |
| 5 | **AnchorKV**（Khalaf 等 5 人） | 2026-08-03 v1；Under review | 每個 token 都有 anchor 投影當 base；預算內為「影響輸出最大」的 token 存量化殘差 | **部分重疊** | 「每個 token 都留粗表示＋只為部分 token 精修」；強調**沒有位置離開 softmax**（p.1）。但：選擇在 **prefill 結束時**用 observation window 的 query（§3.3 p.3）；沒選的殘差丟掉〔判讀〕；HBM；沒有讀回／層級 | 〔全文〕 |
| 6 | **QAQ**（Dong 等 4 人） | 2024-03-07 v1；venue NOT_VERIFIED | 每個 token 依注意力決定位元數；CPU 留**未量化副本**；位元數要升高時從 CPU 取回再量化 | **部分重疊** | 「依注意力從 CPU 層升精度」（p.5 §4 末段）。但：不是巢狀（另存整份副本）；decode；依注意力歷史，不是新問題 | 〔全文（該段）〕 |
| 7 | SpeCache（2503.16163） | 本 repo 已涵蓋 | VRAM 放 1–2-bit 副本；每個 decode step 從 CPU 取 top-k 的 16-bit | 部分重疊 | decode 版的「全部 base＋取回精確」。這輪沒有新事實 | 已涵蓋 |
| 8 | Lynx（2607.01831） | 本 repo 已涵蓋 | 8-bit 拆成高 4 bit（Anchor）／低 4 bit（Residual）兩條流，P/D 傳輸時先用高位 speculative decode | 部分重疊 | 巢狀＋先粗後精，但**沒有選擇**，全部都補 | 已涵蓋 |
| 9 | **Cocktail**（Wei Tao 等 5 人） | 2025-03-30 v1；DATE 2025（arXiv Comments） | 用 Contriever 算 query 與各 chunk 的相似度，依兩個門檻給 FP16／INT4／INT2 | 部分重疊 | **查詢決定每塊精度**（§III p.3）。但：單輪，從剛 prefill 的 FP16 KV 量化下去；外部 encoder，不是模型注意力；不巢狀、沒有層級 | 〔全文〕 |
| 10 | CacheGen §9（2310.07240） | SIGCOMM'24（本 repo 已涵蓋） | Future work：像 SVC 一樣漸進串流 KV | 相關但不同 | 漸進構想的優先權（p.12 §9），但沒有查詢導向，也沒實作 | 〔全文（該段）〕 |
| 11 | MiKV（Yang 等 8 人） | 2024-02-28 v1；venue NOT_VERIFIED | 重要 token 高精度，原本要丟的 token 改存低精度 | 相關但不同 | 「沒有 token 被丟」的證據（Table 1 p.5）。重要性來自 H2O 式注意力歷史，HBM，不巢狀 | 〔全文（§2–3）〕 |
| 12 | KV Offloading for Context-Intensive Tasks／YAKV（Bocharnikov 等 5 人） | 2026-04-09 v1；Preprint | 低秩 K 與群 landmark 在高資訊密度任務選錯；改用 GPU 上 1–2-bit 量化 K 選擇，RAM 放 4-bit KV | 相關但不同（Q4 關鍵） | §4.1–4.3 p.5–7；Table 2–3 p.8。附錄 E 的殘差 landmark 量化是「先粗選再精排」（p.20） | 〔全文〕 |
| 13 | ResidualKV（Hao 等 5 人） | 2026-02-08 v1；preprint | 參考 token＋其餘 token 的量化殘差碼；配合稀疏注意力只重建被選中的 | 相關但不同 | 選擇沿用 OmniKV／SnapKV（p.2）；HBM；沒選的不參與注意力 | 〔部分〕 |
| 14 | SpectrumKV（Pengju Yang，1 人） | 2026-06-07 v1；preprint | P/D 傳輸時逐 token 給 fp16／int8／int4 | 相關但不同 | 重要性來自 prefill（p.2、p.7），不是後來的新問題；不巢狀。另報 Qwen2.5-7B 在 int4 KV 會崩（摘要 p.1） | 〔部分〕 |
| 15 | ResKV（Zhan 等 3 人） | 2026-07-31 v1 | 固定預算下，用殘差統計補回被省略 token 的注意力分子與分母 | 相關但不同 | 補沒選的部分；有 decode 時的逐 query gate；沒提層級 | 〔摘要〕 |
| 16 | vAttention（Desai 等 8 人） | 2025-10-07 v1；ICLR 2026（journal ref） | top-k＋隨機抽樣估計其餘，有 (ε,δ) 保證 | 相關但不同 | 「沒選的也有近似貢獻」的另一路線；decode | 〔摘要〕 |
| 17 | Self-Indexing KVCache（Xu Yang 等 5 人）／Self-Indexing Attention（7 人） | 2026-03-15／2026-08-16 v1；前者據搜尋為 AAAI 2026〔標題〕 | 壓縮 K 的符號位元同時當索引；後者把符號位元用在 **prefill 群選擇**與 decode 取回 | 相關但不同 | 索引是存放內容的子集〔摘要〕；HBM | 〔摘要〕 |
| 18 | FFD（Zhigeng Liu 等 8 人） | 2026-08-31 v1；ICML 2026（Comments） | 用低位元量化掃描 K 並融合 top-δ 選擇與計算 | 相關但不同 | Fathom 說 FFD 把 K 拆成 2-bit thumbnail＋8-bit residual（Fathom p.3）；我沒讀 FFD 原文 | 〔摘要〕 |
| 19 | FIER（Dongwei Wang 等 8 人） | 2025-05-28 v1；EMNLP 2025（Comments） | 1-bit K 估計重要性，逐 token top-k 精確注意力 | 相關但不同（Q4 正面） | §3 OB2 p.4；Fig.6 p.6 | 〔全文〕 |
| 20 | Loki（Singhania 等 5 人） | 2024-06-04 v1；NeurIPS 2024（Comments） | PCA 低維 K 估分數，選 top-k 回到全維 | 相關但不同（Q4） | Fig.6 p.9 | 〔部分〕 |
| 21 | PQCache（Hailin Zhang 等 8 人） | 2024-07-01 v1；PACMMOD 3(3)（SIGMOD 2025，Crossref） | prefill 時對 K 做 PQ，decode 用 PQ 碼找重要 token 再取 KV | 相關但不同 | 粗碼選擇＋取精確 | 〔摘要〕 |
| 22 | LRQK（Tenghui Li 等 5 人） | 2025-10-25 v1；NeurIPS 2025（journal ref） | prefill 分解出低秩 Q／K，decode 用代理分數選 top-k，混合 GPU-CPU 快取只傳缺的 | 相關但不同 | 2604.08426 指出它在高資訊密度任務上選錯（Table 2–3 p.8） | 〔摘要〕 |
| 23 | KVSwap（Huawei Zhang 等 3 人） | 2025-11-14 v1；venue NOT_VERIFIED（搜尋說 MobiSys 2026） | 全部 KV 在磁碟，記憶體放壓縮 K 預測要預載哪些 | 相關但不同 | 磁碟層＋壓縮 K 選擇；目標是 decode 吞吐，不是 TTFT〔摘要〕 | 〔摘要〕 |
| 24 | KVDrive（Jian Lin 等 8 人） | 2026-05-18 v1；venue NOT_VERIFIED | GPU／DRAM／SSD 多層的 decode 時稀疏取回系統 | 相關但不同 | 多層含 SSD，但沒有巢狀或精修 | 〔摘要〕 |
| 25 | TTKV（Dzikanyanga 等 7 人） | 2026-03-27 v1 | 依時間遠近分層：新的放快層高精度，舊的放慢層 | 相關但不同 | 依位置／時間決定，正是本專案發現 (1) 說沒用的那類 | 〔摘要〕 |
| 26 | OasisKV（Can Xiao 等 10 人） | 2026-08-08 v1 | 用 speculative decoding 的 lookahead 預測重要塊，從 host／remote 預取到 HBM | 相關但不同 | 預測選擇，不涉精度 | 〔摘要〕 |
| 27 | QCFuse（Jianxin Yan 等 11 人） | 2026-06-04 v1 | RAG cache fusion，用 chunk anchor 做查詢探測，選要重算的 token | 相關但不同 | 選擇是為了重算，不是精修 | 〔摘要〕 |
| 28 | SparKV（Hongyao Liu 等 6 人） | 2026-04-23 v1；IoTJ（Comments） | 裝置端逐 chunk 決定串流 KV 或本地重算 | 相關但不同 | Cake／CacheGen 類，不是查詢導向精度 | 〔摘要〕 |
| 29 | SpecKV／Draft-based Approx. Inference（Galim 等 6 人） | 2025-06-10 v1；ICLR 2026 | 用小 draft 模型的注意力估計 prompt token 重要性 | 相關但不同（Q4） | 用近似 forward 選 token 的證據 | 〔摘要〕 |
| 30 | How Query Visibility Changes KV-Cache Compression Rankings（Luo 等 3 人） | 2026-07-11 v1 | 審計：壓縮時看不看得到問題，會改變方法排名 | 相關但不同（方法論） | SnapKV 在查詢無關協定下輸給「保留開頭＋最近視窗」（摘要） | 〔摘要〕 |
| 31 | ElastKV | ICDCS 2026〔標題〕 | 重用不同精度模型產生的 KV，選擇性重算 | 相關但不同 | 只在搜尋結果看到 | 〔標題〕 |

---

## 2. 威脅卡片

### 2.1 ReadKV — Read What Matters: Query-Adaptive Quantization for KV Caches（arXiv 2610.11245）

- **書目**：
  - 作者：Siddharth Bhandari（第一作者）、Lucas Gretta、Krishna Balasubramanian、Shiva Kasiviswanathan，共 4 人（Amazon、UC Berkeley）。
  - v1：2026-10-08。Comments 只寫「50 pages, 3 figures, 12 tables」，**venue：preprint**。
  - 〔全文〕
- **做什麼**：
  - **寫入**：每個 K、V 純量存成一個 W-bit 漸進碼（巢狀非均勻純量碼，每個前綴深度都可解）。存的位元數 W 和每個 query 平均讀的位元數 R 分開計算（p.1 摘要；p.2 Fig.1；§3.1 p.4）。
  - **讀**：分兩階段，順序是：
    1. 用當前 query 決定 key 的哪些 channel 讀多深；
    2. 用重建出的 K 算注意力；
    3. 再依注意力決定 value 的哪些 token 讀多深（§3.2 p.4）。

    存的紀錄不改，沒讀的精修層留給之後的 query（p.2 Fig.1 圖說）。
  - **理論**：在有限維的反例族上，查詢導向讀取嚴格優於所有查詢無關讀取（Theorem 1–2，§4）。
- **設定**：
  - **模型**：Qwen2.5-3B／7B／14B、Yi-1.5-6B、DeepSeek-LLM-7B、Mistral-7B-v0.3（Table 2 p.11）。QA 用 Qwen2.5-7B-Instruct 與 Mistral-7B-Instruct-v0.3（p.10）。
  - **context**：C4 每篇 2,048 token；LongBench 的 prompt 上限 3,500 與 7,500 token（p.10）；GPU 計時是 8,192 token（p.13）。
  - **硬體**：NVIDIA A10G，**單層**、batch 1（§5.1 p.13；Table 4 p.14）。
  - **層級**：無，全部在 GPU 記憶體。
  - **基線**：Dense、Full Reader (4,4)、KIVI 2-bit、TurboQuant K4/V4、SparQ r32/k128（p.10）。
  - **指標**：ΔPPL、ΔF1、logical reads／retained capacity（% of Dense）。
- **關鍵數字**：

  | 項目 | 結果 | 出處 |
  |:--|:--|:--|
  | ReadKV (8,4) | 六個模型 PPL 增幅最多 0.66% | 摘要 p.1；p.10 |
  | Qwen2.5-7B：ReadKV (8,4) | +0.39%，reads／retained = 25.4%／50.1% | Table 2 p.11 |
  | Qwen2.5-7B：Full Reader (4,4) | +1.34% | Table 2 p.11 |
  | Qwen2.5-7B：KIVI 2-bit | +2.79% | Table 2 p.11 |
  | **查詢導向的價值**（Qwen 3B 配對控制組，Appendix F.5） | 改用「只用 calibration」分配 key，PPL 從 11.789 升到 12.011；32 篇裡 31 篇偏好用當前 query | p.10 |
  | 單層延遲（A10G） | ReadKV (8,2) 0.093 ms；TurboQuant 0.154 ms；Dense 0.051 ms | Table 4 p.14 |

- **原文自己說的限制**：
  - 還沒證明端到端 serving 有優勢（p.15）。
  - 適用情境是 cache 主導 bytes 的長 context、大 batch，**「possibly tiered memory」**：把少讀的精修層放在較慢、較大的記憶體（p.15）。
- **它沒做的**〔判讀〕：
  - (1) 沒有從 SSD／NFS **讀回**，也沒有多輪 session；
  - (2) **prefill 一律 dense**，沒有在粗表示上做 prefill 來決定精修（p.10）；
  - (3) 粒度是純量／channel／token，**不是 512-token 區塊**；
  - (4) 沒有 TTFT 期限或 anytime 排程；
  - (5) 沒有和重算合用；
  - (6) 每個 decode step 都重新分配，不是讀回時決定一次。
- **對我們的意義**：
  - 「存一份漸進碼、查詢決定讀多深、沒讀的保留」**不能再當成我們的貢獻**。
  - ReadKV 的「先 K 後 V」兩階段，正好可以當我們「先讀 base、用注意力選塊、再補精修」的**直接前例與對照**。
  - 它 Discussion 那句 tiered memory，等於預告了我們要做的情境，**審稿人一定會拿它比**。

### 2.2 Xie et al. — Reimagining Memory Access for LLM Inference: Compression-Aware Memory Controller Design（arXiv 2503.18869）

- **書目**：
  - 作者：Rui Xie（第一作者）等 7 人（RPI、Wiseburn Da Vinci Science）。
  - v1：2025-03-24；v3：2025-04-21。Comments：「9 pages, 11 figures」。**venue NOT_VERIFIED**。
  - 〔全文〕
- **做什麼**：
  - AI 加速器的記憶體控制器把權重與 KV **依位元位置分平面存放**（bit-plane disaggregation），再做跨 token 的 channel 聚集與 exponent delta，提高 LZ4／ZSTD 的無損壓縮率（摘要 p.1；§III-A p.4）。
  - 讀取時控制器**只抓並解壓需要的 bit-plane**，例如 FP8 只讀高 8 個 plane，讓 DRAM 流量和選的精度成正比（§III p.4）。
- **查詢導向那一段**：
  - Table II（p.3），LLaMA 3.1 8B、BookSum，一個 page 是 16 token：

    | 設定 | 困惑度 |
    |:--|:--|
    | Full KV | 10.49 |
    | Sliding Window 64 | 14.33 |
    | Quest（前 5 頁 BF16） | 12.49 |
    | 前 5 頁 BF16＋接下來 3 頁 FP8＋2 頁 FP4 | 11.87 |
    | 前 5 頁 BF16＋接下來 5 頁 FP8 | 11.60 |

  - 正文（p.3）把 11.60 寫成「next 5 FP8, next 3 FP4」，**和表格不一致**。以上照表格列出。
- **其他結果**：KV 無損壓縮後占用減少 46.9%，權重減少 25.2%；在 2 GHz、32 lanes 的模擬下吞吐可到 2 TB/s，面積低於 5.69 mm²（摘要 p.1）。
- **它沒做的**〔判讀〕：
  - (1) 排名之外的 page **不讀**，所以不是「全部 base＋選擇性精修」；
  - (2) 是 DRAM／控制器硬體設計，沒有 CPU／SSD 層，也沒有讀回；
  - (3) 用 Quest 頁摘要排序，不是在粗 KV 上算注意力；
  - (4) 沒有端到端系統、沒有 TTFT。
- **對我們的意義**：ReadKV 自己稱它是「最接近的系統先例」（ReadKV p.2、p.14、App.B.3 p.22）。「依查詢排名，從 bit-plane 存放讀不同精度」這個**機制**在 2025 年就有了。

### 2.3 Fathom — Per-Query Read Depth for Sparse Decoding over Offloaded KV Caches（arXiv 2609.17652）

- **書目**：
  - 作者：Vivek Kalyanarangan（單一作者）。
  - v1：2026-09-15；v3：2026-09-18。Comments：19 pages，附程式碼。**preprint**。
  - 〔全文〕
- **做什麼**：
  - K 量化成 4-bit uniform symmetric（逐 channel，每 64 token 一個 fp16 scale），以 **channel-major bit-plane** 存放；前 t 個 plane 正好就是 t-bit 量化器（§3.2 p.4）。
  - 每個 query 依各 channel 的分數變異，用 reverse water-filling 分配每個 channel 讀幾個 plane（§3.3 p.4），掃完全部 key 選 top-k，再取被選中的列做精確注意力（p.2 §2）。
  - 掃描索引和 KV 列都在 **pinned host memory**（§5.1 p.6）。
- **設定**：
  - **硬體**：A100（PCIe 4.0、2 TB host RAM）；Llama-3.1-8B 用 L4 上擷取的 activation（p.5）。
  - **模型**：Qwen3-8B、Qwen3-4B、Llama-3.1-8B、Qwen2.5-7B、Qwen2.5-7B-1M（Table 6 p.9）。
  - **context**：4k–1M。
  - **基線**：Loki、Double Sparsity、SparQ、2-bit thumbnail、block landmark（ShadowKV 式 8-token 平均）（p.2–3）。
  - **指標**：注意力輸出相對 L2 誤差、RULER 類任務分數、coding-agent session 的 step agreement。
- **關鍵數字**：

  | 項目 | 結果 | 出處 |
  |:--|:--|:--|
  | decode step GPU 時間（1M token，Qwen3-8B，k=512） | 比 136-bit 掃描快 1.67× | Table 1 p.3 |
  | RULER，Qwen3-8B 32k、k=128 | dense 0.884／exact top-k oracle 0.874／Fathom 56-bit 0.881／block landmark 0.831 | Table 7 p.10 |
  | RULER，Qwen2.5-7B-1M 128k、k=128 | dense 0.677／oracle 0.660／Fathom KLT 56-bit 0.671／landmark 0.531 | Table 7 p.10 |
  | 真實 agent session（Qwen2.5-7B-1M，約 95k，k=512），step agreement | Fathom 92-bit 0.60、SparQ r=32 0.60、landmark 0.42 | Table 8 p.10 |

- **原文自己說的限制**（§8 p.12）：
  - 端到端任務成功率沒量。
  - 索引若在 HBM 就不比較快。
  - 「從 channel-major 布局重建單一 token 不便宜」，而且用四個 plane 重建被選中 K 的路徑**沒有實作**。
- **它沒做的**〔判讀〕：
  - (1) 巢狀只用在**選擇用的索引**，V 不漸進；
  - (2) 沒選中的 token **完全不參與注意力**，沒有 base 回退；
  - (3) 只有 decode-step，沒有讀回、多輪續問的 prefill、SSD。
- **對我們的意義**：「巢狀存放放在慢速層＋每個 query 決定讀多深」已經在 host memory 上量過了。我們要往前推的是：(i) 讀回邊界、(ii) 區塊粒度、(iii) V 也要漸進、(iv) 沒選中的區塊仍以 base 參與注意力。

### 2.4 RetroInfer — A Vector Storage Engine for Scalable Long-Context LLM Inference（arXiv 2505.02922）

- **書目**：
  - 作者：Yaoqi Chen（第一作者）等 19 人（USTC、Microsoft Research 等）。
  - v1：2025-05-05。**PVLDB 19(5):1016–1031, 2026**（PDF p.1 與 arXiv journal ref 一致）。
  - 〔全文〕
- **做什麼**：
  - 把 token 分成三區：
    - steady：開頭與最近的 token，精確；
    - retrieval：用 centroid 選出的群，**從 CPU 取精確 KV**；
    - estimation：其餘群用 centroid 與群大小估注意力權重，用預存的 value 總和估輸出（p.2 §1；p.6 Eq.2）。
  - centroid 估計是群內指數和的下界，用 Jensen 不等式證明（p.6 Eq.3）。
  - meta index（centroid、value 總和、群大小）放 GPU，KV 放 CPU（p.6）。
- **設定**（§5.1 p.9）：
  - **硬體**：A100 80GB、PCIe 4.0 ×16（32 GB/s）、EPYC 7V12、1.7 TB RAM。
  - **模型**：Llama3.1-8B、Qwen2.5-7B、Llama3-8B-1048K、Qwen2.5-72B。
  - **任務**：RULER、NIAH 到 1M、AIME24、GPQA。
  - **基線**：Quest、MagicPIG、InfiniGen、PQCache、full attention、vLLM。
  - **參數**：retrieval 預算 1.8%；estimation zone 占 23.2% 的群，**三區合計約涵蓋 25% context**；GPU 快取是 KV 的 5%。
- **關鍵數字**：
  - 128K 時，比 full attention 少 0.66%／1.50%／1.97%／0.73%（四個模型）（§5.2 p.9）。
  - 吞吐：120K 時比 full attention 快 4.4×；1M 時比稀疏基線快 12.2×（摘要 p.1）。
- **它沒做的**〔判讀〕：
  - (1) 粗表示是**群摘要**，不是精確 KV 的可補前綴，所以不能「補精修」，只能「換成精確」；
  - (2) 約 75% 的 context 落在三區之外，沒有任何貢獻（由 p.9「~25%」推得）；
  - (3) decode-step；索引在同一請求的 prefill 建，沒有跨輪持久化、沒有 SSD；
  - (4) 沒有 TTFT 期限。
- **對我們的意義**：「粗估大多數＋精確取回少數＋精確資料在 CPU」是 decode 版的近親。如果我們的 base 足夠好，可以讓「其餘 75%」也有回退。這一點要用實驗區分。

### 2.5 AnchorKV — Anchor-Residual KV Cache Compression（arXiv 2608.02901）

- **書目**：
  - 作者：Malik Khalaf（第一作者）等 5 人（Technion）。
  - v1：2026-08-03。Comments：「Under review」。
  - 〔全文（§1–4）〕
- **做什麼**：
  - 每個 KV head 精確存一小組 anchor；其他 token 用「最相似 anchor 的縮放複本」表示（每個 token 只花幾個 byte）。
  - 剩下的預算拿來為「注意力輸出誤差最大」的 token 存 Hadamard 旋轉後的低位元殘差（p.1–2；§3.1 p.2–3）。
  - 殘差分配用 observation window 的 query 估計一階影響（§3.3 p.3）。
  - **只在 prefill 結束時壓縮一次**（p.1）。
  - 強調沒有任何位置離開 softmax。
- **設定**：
  - **模型**：Llama-3.1-8B／70B-Instruct、Mistral-Small-3.1-24B。
  - **任務**：RULER 32K／64K、LongBench（p.4–5）。
  - **硬體**：A100-80GB，batch 1（p.5）。
- **關鍵數字**：20× 時，RULER-32K 保留 FullKV 分數的 93.5%／95.3%／99.3%（8B／24B／70B）（§4.2 p.5）。
- **它沒做的**〔判讀〕：
  - (1) 精修對象在**寫入時**就決定，用的是當前 prompt 的 observation window；之後的新問題無法改選；
  - (2) 沒選的殘差丟掉，不是放到慢速層；
  - (3) 沒有層級、沒有讀回。
- **對我們的意義**：它正好是我們要打敗的「**寫入時決定精修**」對照組。「base 給全部＋殘差給部分」這個表示有效，它已經證明了；**我們的增量只有「延後到讀回時、用新問題決定」**。這必須用同 bytes 的實驗證明。這和本專案發現 (1)「寫入時依位置決定沒用」一脈相承。

### 2.6 其他部分重疊（短卡）

- **QAQ（2403.04643）**〔全文（p.5 該段）〕：
  - 每生成一個 token，就把**未量化的新 KV 複製到 CPU**。
  - 每步依推導的公式重算各 token 該用幾 bit：位元數下降就再量化；**上升就從 CPU 取未量化值重新量化**。
  - 原文承認這會多出 CPU-GPU 傳輸，用 attention window 減少（p.5 §4 末段）。
  - 摘要說最多約 10× 壓縮。
  - 和我們的差別：不巢狀（整份副本）、decode、依注意力歷史。
  - venue NOT_VERIFIED。
- **Cocktail（2503.23294，DATE 2025）**〔全文〕：
  - 用 Facebook Contriever 編碼 query 與每個 context chunk，算相似度。
  - 高於 T_high 給 FP16，低於 T_low 給 INT2，中間給 INT4；再把同精度的 chunk 重新排列成連續記憶體（§III p.3–4）。
  - 和我們的差別：單輪、從剛算好的 FP16 量化下去、用外部 encoder；不巢狀、沒有層級。
- **MiKV（2402.18096）**〔全文（§2–3）〕：
  - line retrieval，H2O 式重要性（Table 1 p.5）：

    | 重要 token 比例 | 被逐出的 token 改存成 | KV 占用 | 準確率 |
    |:--|:--|:--|:--|
    | 20% | INT4 | 41% | 100.0% |
    | 20% | INT2 | 32% | 64.0% |
    | 20% | 直接逐出 | — | 4.0% |
    | 50% | INT4 | — | 100.0% |
    | 50% | INT2 | — | 84.6% |
    | 50% | 直接逐出 | — | 43.2% |

  - 這是「全部保留粗表示」勝過丟棄的直接證據；但 INT2 會掉，所以 **base 精度有下限**。
  - 「More Tokens, Lower Precision」（2412.12706）〔摘要〕的結論同方向：低精度保留更多 token 勝過少 token 高精度。
- **CacheGen §9（2310.07240v6，p.12）**〔全文（該段）〕：
  - 「Incremental KV cache streaming」寫為 future work：像 SVC 一樣先送低品質 KV，再送差值逐步提升。
  - 沒有查詢導向，也沒實作。

---

## 3. Q2　嵌入式／巢狀／可逐步精修的表示

「巢狀」= 存一份，就能用不同保真度讀（前綴可解碼）。

| 表示 | 論文 | 巢狀？ | 對象 | 跨層級？ | 查詢導向？ | 位置／深度 |
|:--|:--|:--|:--|:--|:--|:--|
| 8-bit 漸進非均勻純量碼，每個前綴都可解 | ReadKV 2610.11245 | **是** | KV | 否（只在 Discussion 提，p.15） | **是**（K 依 query、V 依重建注意力） | §3.1 p.4〔全文〕 |
| 浮點 bit-plane（BF16→FP8→FP4 前綴） | Xie 2503.18869 | **是** | 權重＋KV | DRAM（控制器） | **是**（Quest 排 page） | §III p.4；Table II p.3〔全文〕 |
| 4-bit K channel-major bit-plane（前 t 個 plane = t-bit 量化器） | Fathom 2609.17652 | **是**（只有 K 索引） | KV（K） | **host memory** | **是**（逐 channel 深度） | §3.2 p.4〔全文〕 |
| 8-bit = 高 4 bit Anchor＋低 4 bit Residual | Lynx 2607.01831 | 是 | KV | P/D 網路 | 否 | 本 repo 已涵蓋 |
| 巢狀 INT8 = 高 4＋低 4 | QuantSpec 2502.10424 | 是 | KV | HBM | 否（draft／verify） | 本 repo 已涵蓋 |
| 2-bit thumbnail＋8-bit residual | FFD 2609.00097 | 是〔標題，據 Fathom p.3〕 | KV（K） | HBM〔推論〕 | 掃描依門檻 | 〔摘要〕 |
| 4-bit landmark（每 8 個 key 一個）＋每個 key 1-bit 殘差 | 2604.08426 附錄 E | 兩級 | KV（K 索引） | 索引在 GPU、KV 在 RAM | 是（先選 landmark，只對選中 chunk 算殘差） | p.20〔全文〕 |
| 符號位元 = 索引 ⊂ sign-magnitude 存放 | Self-Indexing 2603.14224／2609.13205 | 部分〔摘要〕 | KV（K） | HBM | 是 | 〔摘要〕 |
| anchor 投影（全部）＋量化殘差（部分 token） | AnchorKV 2608.02901 | 兩級；沒選的殘差丟掉 | KV | HBM | 寫入時用 observation window | p.1–3〔全文〕 |
| 參考 token＋殘差碼 | ResidualKV 2602.08005 | 否（重建用） | KV | HBM | 選擇沿用 OmniKV／SnapKV | 〔部分〕 |
| centroid＋value 總和（摘要）＋精確 KV（CPU） | RetroInfer 2505.02922 | 否（摘要不是前綴） | KV | GPU 摘要／CPU 精確 | 是 | p.6〔全文〕 |
| 殘差向量量化，深度 8 | RVQ-KV 2410.15704 | 〔推論〕RVQ 構造上前綴可解；原文是否評估不同深度只見摘要「深度 8 恢復大部分效能」 | KV | 否 | 否 | 〔摘要〕 |
| 加法量化，codebook 與 RoPE 可交換 | CommVQ 2506.18879（ICML 2025） | 〔推論〕不是為巢狀設計（多個 codebook 聯合學習） | KV | 否 | 否 | 〔摘要〕 |
| MSE 量化＋殘差 1-bit QJL | TurboQuant 2504.19874 | 兩級（第二級是為了無偏內積） | KV（及向量檢索） | 否 | 否 | 〔摘要〕；venue NOT_VERIFIED |
| 超低位元量化＋低秩殘差＋稀疏離群值 | GEAR 2403.05527 | 加法修正，不是為漸進設計 | KV | 否 | 否 | 〔摘要〕 |
| Matryoshka 訓練的正交投影（任意秩前綴） | MatryoshkaKV 2410.14731 | **是（秩方向）** | KV | 否 | 否（逐層／head 搜預算） | 〔摘要〕 |
| 逐 token 依重要性的混合精度 | MiKV、ZipCache 2405.14256、SpectrumKV、QAQ | 否（不同 token 不同精度；QAQ 另存全精度副本） | KV | QAQ 有 CPU 副本 | 依注意力歷史或 prefill | 見 §1 |
| 逐 chunk 依查詢相似度的混合精度 | Cocktail 2503.23294 | 否 | KV | 否 | 是（當前單輪 query） | §III p.3〔全文〕 |
| 隨記憶體漸滿降低存放位元數（16→8→4→2） | PM-KVQ 2505.18610（ICLR 2026） | 未查證 | KV | 否 | 否 | 〔摘要〕 |
| bit-plane 疊加 3…n bit 模型 | Any-Precision LLM 2402.10517（ICML 2024） | **是** | 權重 | 否 | 否（逐請求選精度） | 〔摘要〕 |
| int8 取高位得 int4／int2，多尺度共同訓練 | MatQuant 2502.06786（ICML 2025〔標題〕） | **是** | 權重 | 否 | 否 | 〔摘要〕 |
| 每輪約 1 bit/參數的殘差塊，排序堆在儲存，依可用記憶體載入不同數量 | BitStack 2410.23918（ICLR 2025） | **是** | 權重 | **是（儲存↔記憶體）** | 否 | 〔摘要〕 |
| decode 後段逐步降低權重精度 | PMPD 2410.13461（ICLR 2025〔標題〕） | 未查證 | 權重 | 否 | 依任務／prompt 排程 | 〔摘要〕 |
| 每 chunk 多個獨立編碼等級（不巢狀） | CacheGen 2310.07240 | 否；§9 提 SVC 式漸進為 future work | KV | 網路／儲存 | 否（依頻寬） | p.12〔全文（該段）〕 |

**整理**：

- **KV＋巢狀＋查詢導向**：ReadKV、Xie et al.、Fathom（只有 K）、2604.08426 附錄 E（只有 K 索引）。
- **KV＋巢狀＋跨層級**：Fathom（host memory）、Lynx（網路）。
- **四項都占（KV、巢狀、跨到 SSD 級層級、新問題在讀回時決定）**：**NOT_FOUND**。
- **權重＋巢狀＋跨儲存**：BitStack（依可用記憶體，不依查詢）。
- 「multi-resolution KV」這個詞沒找到對應論文（NOT_FOUND）。

---

## 4. Q3　古典錨點（只核對書目）

書目欄位（卷期頁）以 Crossref API 回傳為準；概念句是給我們的用法〔判讀〕。

1. **W. H. R. Equitz and T. M. Cover**, "Successive refinement of information," *IEEE Trans. Inf. Theory*, vol. 37, no. 2, pp. 269–275, Mar. 1991. doi:10.1109/18.75242。
   - 核對：Crossref；搜尋結果摘要。
   - 給我們的：先給粗描述再補精修，可以在每個速率都達到最佳。條件是各速率的最佳描述能排成 Markov 鏈；而且**不是所有來源都成立**（摘要提到反例）。這是「base＋精修不必付代價」的理論上界與警告。
2. **J. M. Shapiro**, "Embedded image coding using zerotrees of wavelet coefficients," *IEEE Trans. Signal Process.*, vol. 41, no. 12, pp. 3445–3462, Dec. 1993. doi:10.1109/78.258085。
   - 核對：Crossref。
   - 給我們的：位元流依重要性排序，解碼端可以在任何位置停下（embedded），就是「同一份存放，任意截斷都可用」。
3. **A. Said and W. A. Pearlman**, "A new, fast, and efficient image codec based on set partitioning in hierarchical trees," *IEEE Trans. Circuits Syst. Video Technol.*, vol. 6, no. 3, pp. 243–250, Jun. 1996. doi:10.1109/76.499834。
   - 核對：Crossref。
   - 給我們的：先傳「哪些係數重要」（排序／顯著性），再逐個 bit-plane 傳精修。對應「先排區塊，再按 plane 補」。
4. **G. K. Wallace**, "The JPEG still picture compression standard," *Commun. ACM*, vol. 34, no. 4, pp. 30–44, Apr. 1991. doi:10.1145/103085.103089。
   - 核對：書目用 Crossref；progressive 模式內容只見二手來源〔標題〕。
   - 給我們的：progressive 模式有兩條正交的漸進軸，一條是 spectral selection（先送哪些頻帶），一條是 successive approximation（先送高位元）。對應我們的「低秩成分（秩）× 量化位元（plane）」。
5. **H. Schwarz, D. Marpe, and T. Wiegand**, "Overview of the scalable video coding extension of the H.264/AVC standard," *IEEE Trans. Circuits Syst. Video Technol.*, vol. 17, no. 9, pp. 1103–1120, Sep. 2007. doi:10.1109/TCSVT.2007.905532。
   - 核對：Crossref；UCSB 課程網站的 PDF 首頁。
   - 給我們的：base layer 必須能獨立解碼；enhancement layer 可以丟掉；時間、空間、品質三種可分級性；可分級要付編碼效率的代價。CacheGen §9 引用的正是 SVC。
6. **T. Dean and M. Boddy**, "An analysis of time-dependent planning," in *Proc. AAAI-88*, 1988, pp. 49–54。
   - 核對：AAAI PDF 與 mlanthology 列表（搜尋結果）。Crossref 某筆引用把標題寫成「…planning problems」，以 AAAI 原檔為準。
   - 給我們的：anytime 演算法隨時可中斷並回傳結果，品質是計算時間的函數。對應「TTFT 期限到了就用目前的精修程度開跑」。
7. **S. Zilberstein**, "Using anytime algorithms in intelligent systems," *AI Mag.*, vol. 17, no. 3, pp. 73–83, 1996. doi:10.1609/aimag.v17i3.1232。
   - 核對：AAAI OJS 頁與 DOI；頁尾 83 只見於一個書目來源；出版日 OJS 寫 1996-03-15，與「Fall 1996」不一致，未解決。
   - 給我們的：performance profile 與 meta-level control，也就是「何時停止精修、把時間給誰」。對應我們的精修排程器。

---

## 5. Q4　粗表示上的注意力能不能可靠地排出重要區塊？

| 論文 | 粗表示 | query 來源 | 指標 | 結果 | 位置 | 和我們情境的差距 |
|:--|:--|:--|:--|:--|:--|:--|
| FIER 2508.08256 | 1-bit RTN K（group 32） | 精確 query（14k context 的 prefill 測試；decode） | 與完整注意力 top-k 的重疊（recall） | Top-64：Fier 91%；Quest page 16 是 44%、page 8 是 67%（LLaMA） | §3 OB2 p.4；Fig.6 p.6 | 逐 token；K 以外都精確；不是區塊 |
| Loki 2406.02542 | PCA 前 25% 維度（未量化） | 精確 decode query | 與 Exact-TopK 的 Jaccard | 兩組設定約 0.9（Llama2-7B） | Fig.6 p.9 | 壓縮只有 4×（維度）；逐 token |
| Fathom 2609.17652 | 4-bit bit-plane K，逐 query 讀 47–92 bit/token | 精確 decode query | RULER 與 oracle 的差；agent step agreement | 逐 token 掃描離 oracle 0.008（32k）／0.025（128k）以內；**8-token block landmark 掉 0.043／0.130**；agent 最好也只有 0.60（k=512） | §5.5 p.9；Table 7–8 p.10 | decode；逐 token；**真實 agent 任務對選擇誤差敏感** |
| 2604.08426（YAKV） | SVD rank-160 K（ShadowKV）vs HIGGS 1／2／4-bit 逐 token | decode | 任務準確率 | **低秩 K 在高資訊密度任務上選錯，多載 10 倍 token 仍不夠**（§4.1 p.5、Fig.2）。2-bit 逐 token 量化勝過 ShadowKV（同 GPU 記憶體）與 LRQK；4-bit 追平上界（§4.3 p.7）。Table 2 p.8，Qwen3-4B 的 Text2JSON：Original 91.25、ShadowKV 69.38、LRQK 87.99、YAKV 90.83 | 見左 | **最直接的警告**：我們的 base 若是低秩係數，排序可能失準 |
| RetroInfer 2505.02922 | 群 centroid（平均每 16 token 一個） | decode | centroid 排名 vs 群累積注意力 | 定性：排在前面的 centroid 累積注意力較高，估計值緊貼真值（Llama3-8B-1048K，128K） | Fig.8 p.7 | 只有一層一個 head 的圖 |
| Xie et al. 2503.18869 | Quest page 摘要排序＋精度分級 | decode | 困惑度 | 分級（BF16／FP8／FP4）11.60–11.87，優於只留前 5 頁 BF16 的 12.49 | Table II p.3 | 粗表示只用來排序 |
| MiKV 2402.18096 | 非重要 token 改存 INT4／INT3／INT2 | decode | line retrieval | INT4 回到 100%，INT2 掉到 64–85% | Table 1 p.5 | 這是「粗表示參與注意力」的品質，不是排序 |
| SpecKV 2506.08373 | 小 draft 模型的注意力 | draft 的 lookahead | RULER 等 | 摘要說優於 SnapKV 類基線 | 〔摘要〕 | 近似的是**整個模型**，不是 KV |
| Self-Indexing Attention 2609.13205 | K 的符號位元 | **prefill** 群選擇＋decode | LongBench／RULER | 5% 密度接近 dense | 〔摘要〕 | 唯一提到用壓縮 K 做 **prefill 選擇**的，但 query 本身仍精確〔推論〕 |
| 本專案（已量） | Quest min／max，512-token 區塊 | — | 任務分數 | 載入一半區塊掉 8–12 分；用完整 K 的 oracle 顯示 Qwen 只需 1/4 區塊不掉分 | 本專案 results | 區塊粒度＋便宜摘要失準 |

**判讀**：

1. **逐 token 的低位元 K（1–4 bit）排序相當可靠**：FIER、Fathom、YAKV 都是。**區塊摘要與低秩代理在高資訊密度任務上不可靠**：YAKV、Fathom 的 landmark、本專案的 Quest-512 都是。
2. **沒有論文量過「query 本身是在粗 KV 上 forward 出來的」時的排序 recall**（NOT_FOUND）。
   - 上表每一篇的 query 都來自精確路徑。
   - 我們的方案是新問題在 base 上 prefill。第 l 層的注意力誤差會改變第 l+1 層的 hidden state，進而改變 query。
   - 相近的間接證據：QuantSpec／VeriCache 在量化 KV 上 draft 的接受率（本 repo 已涵蓋）。
   - **這是本題最該先量的 NOT_MEASURED**。
3. 〔推論〕**base 的設計建議**：
   - base 裡保留**逐 token 的 1–2-bit K**（排序用，對 K 約 8–16× 壓縮），不要只放低秩係數；
   - 低秩（xKV 式）可以放在 V，或當「第一層精修」；
   - 排序可以用「base K × 精確新問題 query 的前幾層」，或用 ReadKV 式「先讀 K 前綴、再依注意力選 V」。
   - 這三種要做消融。

---

## 6. 缺口與審稿人會要的對照

### 6.1 還剩的缺口（可以主張的新意）

1. **讀回邊界**：多輪續問時，KV 從 SSD／NFS 讀回，TTFT 是主要指標。前面所有部分重疊的論文都是 decode-step（ReadKV、Fathom、RetroInfer、SpeCache、QAQ、Xie），或寫入時決定（AnchorKV、Cocktail、MiKV）。
2. **選擇訊號來自新問題在 base 上的 prefill**：
   - ReadKV 與 Fathom 都用精確 query（ReadKV p.10 明寫 Dense prefill）；
   - Cocktail 用外部 encoder；
   - AnchorKV 用寫入當時的 observation window。
3. **區塊粒度，而且沒選中的區塊仍以 base 參與注意力**：Xie、Fathom、RetroInfer 的排名外內容不參與或只剩約 25% 覆蓋；ReadKV 是純量粒度。
4. **期限內的 anytime 精修排程**：TTFT 預算內決定精修多少、先精修誰。和 **Cake 式「前段重算、後段載入」合用**（前段重算本來就精確，只需精修後段）。所有論文都 NOT_FOUND。
5. **慢速層的 I/O 形狀**：精修層要能連續讀取（Fathom §3.6 p.5 指出 bit-plane 讓查詢導向的讀取變成長的連續 run，這點可以借用）。

### 6.2 審稿人一定會要的對照（同 bytes 或同 TTFT 比較）

| # | 對照組 | 回答什麼問題 |
|:--|:--|:--|
| B1 | 全部 BF16 載入；Cake（重算＋載入） | 上界與現行做法 |
| B2 | 只載入 base、不精修；uniform 壓縮全部載入（INT4 3.2×、xKV 8×，本專案已量），**讀取 bytes 相同** | 精修值不值得 |
| B3 | **ReadKV 式查詢無關讀取**：同一個漸進碼，每塊讀相同前綴深度，讀取預算相同 | 「查詢導向」帶來多少（對應 ReadKV Full Reader 與 calibration-only 控制組，p.10） |
| B4 | **寫入時決定精修**（AnchorKV／MiKV／SpectrumKV 式，或 KVzip 分數），bytes 相同 | 延後到讀回時決定有沒有贏（本題核心主張） |
| B5 | 選擇性讀回並丟棄其餘（Quest-512、oracle，本專案已量） | 「沒有區塊遺失」帶來多少 |
| B6 | 便宜索引選擇再取精確：Quest、ShadowKV landmark、**1–2-bit K 逐 token 掃描**（FIER／YAKV／Fathom 式） | 「在 base 上 prefill 選塊」是否優於便宜索引 |
| B7 | **Oracle 精修選擇**（用完整 K 或完整注意力排區塊） | 選擇誤差的上界；配合 Q4 的 recall 量測 |
| B8 | SpeCache 式 decode 每步取回 | 只在讀回時精修一次夠不夠，還是 decode 也要持續取 |
| B9 | 不選擇、依序全部補精修（Lynx／CacheGen §9 式）＋期限截斷 | 選擇 vs 單純漸進 |
| B10 | RetroInfer 式「centroid 估計＋精確取回」 | base 用「摘要」還是「可補前綴」 |

**任務與量測**：

- 除了 RULER／LongBench，**必須包含高資訊密度任務**：Text2JSON、LongProc HTML→TSV、Loong（2604.08426 Table 1–3）。
- 必須包含多輪共享 context（SCBench）或真實 agent 軌跡（Fathom 用 OpenHands／SWE-rebench，§5.6 p.9）。
- 2607.11942〔摘要〕指出，壓縮時看不看得到問題，方法排名會變。所以**新問題必須在寫入之後才出現**（查詢無關的寫入協定），不能把問題放進 prompt 再壓縮。
- 必量的數字：
  - TTFT；從各層讀取的 bytes；品質（分數）；
  - **base 上 prefill 的區塊排序 recall**，對照 oracle 與精確 query 兩種；
  - 精修比例和品質的曲線。
- 每個數字都要能追到 run_id（CLAUDE.md §4）。

---

## 7. 參考文獻（IEEE 格式；最後一欄是本輪核對狀態）

**第一級：最接近的威脅**

[1] S. Bhandari, L. Gretta, K. Balasubramanian, and S. Kasiviswanathan, "Read what matters: Query-adaptive quantization for KV caches," arXiv:2610.11245, v1 Oct. 8, 2026. https://arxiv.org/abs/2610.11245 — 〔全文〕；preprint。

[2] R. Xie, A. U. Haq, L. Ma, Y. Fang, Z. B. Engineer, L. Liu, and T. Zhang, "Reimagining memory access for LLM inference: Compression-aware memory controller design," arXiv:2503.18869, v1 Mar. 24, 2025. https://arxiv.org/abs/2503.18869 — 〔全文〕；venue NOT_VERIFIED。

[3] V. Kalyanarangan, "Fathom: Per-query read depth for sparse decoding over offloaded KV caches," arXiv:2609.17652, v1 Sep. 15, 2026. https://arxiv.org/abs/2609.17652 — 〔全文〕；preprint。

[4] Y. Chen *et al.* (19 authors), "RetroInfer: A vector storage engine for scalable long-context LLM inference," *Proc. VLDB Endow.*, vol. 19, no. 5, pp. 1016–1031, 2026; arXiv:2505.02922, v1 May 5, 2025. https://arxiv.org/abs/2505.02922 — 〔全文〕；venue 已核（PDF p.1、arXiv journal ref）。

[5] M. Khalaf, Y. Shamshoum, N. Hodos, Y. Sieradzki, and A. Schuster, "AnchorKV: Anchor-residual KV cache compression," arXiv:2608.02901, v1 Aug. 3, 2026. https://arxiv.org/abs/2608.02901 — 〔全文 §1–4〕；Under review。

[6] S. Dong, W. Cheng, J. Qin, and W. Wang, "QAQ: Quality adaptive quantization for LLM KV cache," arXiv:2403.04643, v1 Mar. 7, 2024. https://arxiv.org/abs/2403.04643 — 〔全文（p.5 該段）〕；venue NOT_VERIFIED。

[7] W. Tao, B. Zhang, X. Qu, J. Wan, and J. Wang, "Cocktail: Chunk-adaptive mixed-precision quantization for long-context LLM inference," in *Proc. DATE*, 2025; arXiv:2503.23294, v1 Mar. 30, 2025. https://arxiv.org/abs/2503.23294 — 〔全文〕；venue 依 arXiv Comments。

[8] Y. Liu *et al.*, "CacheGen: KV cache compression and streaming for fast large language model serving," in *Proc. ACM SIGCOMM*, 2024; arXiv:2310.07240 (v6). https://arxiv.org/abs/2310.07240 — 本 repo 已涵蓋；本輪新讀 §9 p.12〔全文（該段）〕。

**第二級：Q4 證據**

[9] A. Bocharnikov, I. Ermakov, D. Kuznedelev, V. Zhdanovskiy, and Y. Yershov, "KV cache offloading for context-intensive tasks," arXiv:2604.08426, v1 Apr. 9, 2026 (v5 Sep. 1, 2026). https://arxiv.org/abs/2604.08426 — 〔全文〕；Preprint。

[10] D. Wang *et al.* (8 authors), "FIER: Fine-grained and efficient KV cache retrieval for long-context LLM inference," EMNLP 2025; arXiv:2508.08256, v1 May 28, 2025. https://arxiv.org/abs/2508.08256 — 〔全文〕。venue 依 arXiv Comments「EMNLP2025 Camera-ready」；搜尋結果的 ACL Anthology 編號是 2025.findings-emnlp.515（未開頁，屬 Findings 或主會未確認）。

[11] P. Singhania, S. Singh, S. He, S. Feizi, and A. Bhatele, "Loki: Low-rank keys for efficient sparse attention," in *Proc. NeurIPS*, 2024; arXiv:2406.02542. https://arxiv.org/abs/2406.02542 — 〔部分〕（p.9 Fig.6）；venue 依 Comments。

[12] J. Y. Yang *et al.* (8 authors), "No token left behind: Reliable KV cache compression via importance-aware mixed precision quantization," arXiv:2402.18096, v1 Feb. 28, 2024. https://arxiv.org/abs/2402.18096 — 〔全文 §2–3〕；venue NOT_VERIFIED。

[13] K. Galim *et al.* (6 authors), "Draft-based approximate inference for LLMs," in *Proc. ICLR*, 2026; arXiv:2506.08373. https://arxiv.org/abs/2506.08373 — 〔摘要〕；venue 依 Comments／journal ref。

[14] X. Yang *et al.* (7 authors), "Self-indexing attention for compression-compatible sparse long-context LLM inference," arXiv:2609.13205, v1 Aug. 16, 2026. https://arxiv.org/abs/2609.13205 — 〔摘要〕。

[15] X. Yang, J. Zhang, D. Zhao, G. Chen, and Z. Tang, "Self-indexing KVCache: Predicting sparse attention from compressed keys," arXiv:2603.14224, v1 Mar. 15, 2026. https://arxiv.org/abs/2603.14224 — 〔摘要〕；AAAI 2026〔標題〕（只在搜尋結果看到 ojs.aaai.org 頁）。

**第三級：相關但不同**

[16] J. Hao, Q. Huang, Y. Wang, M. Zhang, and J. Yu, "ResidualKV: Residual-based KV cache compression for efficient long-context inference," arXiv:2602.08005, v1 Feb. 8, 2026. https://arxiv.org/abs/2602.08005 — 〔部分〕。

[17] Y. Pengju, "SpectrumKV: Per-token mixed-precision KV cache transfer for prefill–decode disaggregated LLM serving," arXiv:2606.08635, v1 Jun. 7, 2026. https://arxiv.org/abs/2606.08635 — 〔部分〕。arXiv meta 的作者寫作「Pengju, Yang」（單一作者）；PDF 首頁寫 Pengju Yang。

[18] Y. Zhan, L. Chen, and S. Shang, "ResKV: Reconstructing omitted attention contributions for fixed-budget KV cache compression," arXiv:2607.29591, v1 Jul. 31, 2026. https://arxiv.org/abs/2607.29591 — 〔摘要〕。

[19] A. Desai *et al.* (8 authors), "vAttention: Verified sparse attention," in *Proc. ICLR*, 2026; arXiv:2510.05688. https://arxiv.org/abs/2510.05688 — 〔摘要〕。

[20] Z. Liu *et al.* (8 authors), "Faster than flash: Exploiting attention sparsity for efficient long-context decoding," in *Proc. ICML*, 2026; arXiv:2609.00097. https://arxiv.org/abs/2609.00097 — 〔摘要〕；venue 依 Comments。

[21] H. Zhang *et al.* (8 authors), "PQCache: Product quantization-based KVCache for long context LLM inference," *Proc. ACM Manag. Data*, vol. 3, no. 3, 2025 (SIGMOD), doi:10.1145/3725338; arXiv:2407.12820. — 〔摘要〕；venue 用 Crossref 核對。

[22] T. Li, G. Zhou, X. Zhao, Y. Qiu, and Q. Zhao, "Efficient low rank attention for long-context inference in large language models," in *Proc. NeurIPS*, 2025; arXiv:2510.23649. https://arxiv.org/abs/2510.23649 — 〔摘要〕。

[23] H. Zhang, C. Xia, and Z. Wang, "KVSwap: Disk-aware KV cache offloading for long-context on-device inference," arXiv:2511.11907, v1 Nov. 14, 2025. https://arxiv.org/abs/2511.11907 — 〔摘要〕；venue NOT_VERIFIED。

[24] J. Lin *et al.* (8 authors), "KVDrive: A holistic multi-tier KV cache management system for long-context LLM inference," arXiv:2605.18071, v1 May 18, 2026. https://arxiv.org/abs/2605.18071 — 〔摘要〕。

[25] G. Dzikanyanga *et al.* (7 authors), "TTKV: Temporal-tiered KV cache for long-context LLM inference," arXiv:2604.19769, v1 Mar. 27, 2026. https://arxiv.org/abs/2604.19769 — 〔摘要〕。

[26] C. Xiao *et al.* (10 authors), "OasisKV: Scaling in-decode KV cache beyond HBM with lookahead sparse prefetching," arXiv:2608.08097, v1 Aug. 8, 2026. https://arxiv.org/abs/2608.08097 — 〔摘要〕。

[27] J. Yan *et al.* (11 authors), "QCFuse: Query-aware cache fusion via compressed view for efficient RAG serving," arXiv:2606.05875, v1 Jun. 4, 2026. https://arxiv.org/abs/2606.05875 — 〔摘要〕。

[28] H. Liu *et al.* (6 authors), "SparKV: Overhead-aware KV cache loading for efficient on-device LLM inference," arXiv:2604.21231 (Comments: IoTJ). https://arxiv.org/abs/2604.21231 — 〔摘要〕。

[29] D. Luo *et al.* (3 authors), "How query visibility changes KV-cache compression rankings: A matched-budget audit," arXiv:2607.11942, v1 Jul. 11, 2026. https://arxiv.org/abs/2607.11942 — 〔摘要〕。

[30] J. Zhang *et al.* (9 authors), "More tokens, lower precision: Towards the optimal token-precision trade-off in KV cache compression," arXiv:2412.12706. https://arxiv.org/abs/2412.12706 — 〔摘要〕。

[31] T. Liu *et al.* (10 authors), "PM-KVQ: Progressive mixed-precision KV cache quantization for long-CoT LLMs," ICLR 2026 (arXiv Comments); arXiv:2505.18610. https://arxiv.org/abs/2505.18610 — 〔摘要〕。

[32] ElastKV, IEEE ICDCS 2026 — 〔標題〕（只看到 cis.temple.edu 的 PDF 連結）；NOT_VERIFIED。

**第四級：巢狀表示（Q2）**

[33] A. Kumar, "Residual vector quantization for KV cache compression in large language model," arXiv:2410.15704 (NeurIPS 2024 ENLSP workshop, PMLR 262〔標題〕). https://arxiv.org/abs/2410.15704 — 〔摘要〕。

[34] J. Li *et al.* (11 authors), "CommVQ: Commutative vector quantization for KV cache compression," in *Proc. ICML*, 2025; arXiv:2506.18879. — 〔摘要〕；venue 依 Comments「ICML 2025 poster」。

[35] B. Lin *et al.* (8 authors), "MatryoshkaKV: Adaptive KV compression via trainable orthogonal projection," ICLR 2025〔標題〕; arXiv:2410.14731. — 〔摘要〕。

[36] A. Zandieh, M. Daliri, M. Hadian, and V. Mirrokni, "TurboQuant: Online vector quantization with near-optimal distortion rate," arXiv:2504.19874. — 〔摘要〕；ICLR 2026 只見二手來源，NOT_VERIFIED。

[37] H. Kang *et al.* (7 authors), "GEAR: An efficient KV cache compression recipe for near-lossless generative inference of LLM," arXiv:2403.05527。據搜尋結果，正式版改名為 "GEAR: An efficient error reduction framework for KV cache compression in LLM inference"，收於 PMLR 262（NeurIPS 2024 ENLSP workshop）〔標題〕。 — 〔摘要〕。

[38] Y. He *et al.* (6 authors), "ZipCache: Accurate and efficient KV cache quantization with salient token identification," NeurIPS 2024〔標題〕; arXiv:2405.14256. — 〔摘要（搜尋）〕。

[39] Y. Park, J. Hyun, S. Cho, B. Sim, and J. W. Lee, "Any-precision LLM: Low-cost deployment of multiple, different-sized LLMs," in *Proc. ICML*, 2024; arXiv:2402.10517. — 〔摘要〕；venue 依 Comments。

[40] P. Nair, P. Datta, J. Dean, P. Jain, and A. Kusupati, "Matryoshka quantization," ICML 2025（PMLR 267〔標題〕）; arXiv:2502.06786. — 〔摘要〕。

[41] X. Wang *et al.* (6 authors), "BitStack: Any-size compression of large language models in variable memory environments," in *Proc. ICLR*, 2025; arXiv:2410.23918. — 〔摘要〕；venue 依 Comments。

[42] H. M. Chen, F. Tan, A. Kouris, R. Lee, H. Fan, and S. I. Venieris, "Progressive mixed-precision decoding for efficient LLM inference," ICLR 2025〔標題〕; arXiv:2410.13461. — 〔摘要（搜尋）〕。

**第五級：古典（Q3，見 §4）**

[43] W. H. R. Equitz and T. M. Cover, *IEEE Trans. Inf. Theory*, 37(2):269–275, 1991. — Crossref 已核。

[44] J. M. Shapiro, *IEEE Trans. Signal Process.*, 41(12):3445–3462, 1993. — Crossref 已核。

[45] A. Said and W. A. Pearlman, *IEEE Trans. Circuits Syst. Video Technol.*, 6(3):243–250, 1996. — Crossref 已核。

[46] G. K. Wallace, *Commun. ACM*, 34(4):30–44, 1991. — Crossref 已核。

[47] H. Schwarz, D. Marpe, and T. Wiegand, *IEEE Trans. Circuits Syst. Video Technol.*, 17(9):1103–1120, 2007. — Crossref 已核。

[48] T. Dean and M. Boddy, *Proc. AAAI-88*, pp. 49–54, 1988. — AAAI 原檔／mlanthology（搜尋）。

[49] S. Zilberstein, *AI Mag.*, 17(3):73–83, 1996. — OJS／DOI 已核；頁尾與出版日見 §4。

**本 repo 已涵蓋、本輪沒有新事實**：Lynx 2607.01831、QuantSpec 2502.10424、SpeCache 2503.16163、TailorKV、ShadowKV 2410.21465、KVTC 2511.01815、xKV 2503.18893、LOCKS 2607.24555、IMPRESS、ContiguousKV、SmartGen、VeriCache 2605.17613、KVzip 2505.23416、SCBench、RocketKV、Quest。
