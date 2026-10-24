# 03　競爭對手：每個方向最接近的前作、他們沒做的、我們必須跑的基準

> **日期**：2026-10-24。**來源**：這輪的 [L8a](lit/L8a_progressive_query_aware_restore.md)–[L8e](lit/L8e_novelty_criteria_venues.md)（全文閱讀、標頁碼），加上前幾輪的 `docs/stage3_20261019/lit/L1–L7d`。
> **深度標記**照 L 檔：〔全文〕讀了 PDF 文字；〔部分〕讀了指定章節；〔摘要〕只讀 abs 頁。所有判讀都是 self-review，**不是 cross-model**；最大威脅的幾篇另由獨立 verifier 抽查（結果見 §6）。
> 「沒查到」只代表這輪沒找到，不等於不存在（CLAUDE.md 規則 7）。回到核心頁：[README](README.md)。

---

## 0. 一頁結論

| 方向 | 最大威脅 | 他們已經做了 | 我們還剩的 | 新意〔判讀〕 |
|:--|:--|:--|:--|:--|
| **A 漸進式還原（副線）** | ReadKV（2610.11245）、Fathom（2609.17652）；Xie et al.（2503.18869）較弱 | 一份漸進碼、由查詢決定每塊讀多深、沒讀的保留；全部在 decode、HBM／DRAM／host memory，query 都是精確的 | 多輪**讀回邊界**、從 **SSD／NFS**、以**塊**為單位、用**新問題在粗 KV 上 prefill** 的注意力挑塊、**TTFT 期限**、和 **Cake** 合用；以及「query 不精確」造成多少損失（沒人量過） | 中 |
| **B 資源感知（主線）** | Cascade（2608.06557）、Bidaw（FAST'26）、vLLM PR #42212；KVServe、HCache、Tutti | 延遲預算決定搬多少、其餘重算（負載只看 decode 數、爭用是離線 γ）；依大小排讀取（假設 SM 閒）；依 payload 大小靜態選 copy engine 或 SM（28 KiB、12 SM）；依頻寬／SLO 選壓縮（不看 GPU 負載） | 用當下、按資源拆開的爭用**聯合**選切點、解碼路徑、搬法、SM 配額、順序；「算力型翻、頻寬型不翻」；12 個 GPU 共用系統都沒把 KV 還原放進 SM 分配 | 中高 |
| **C 外推預測** | Cake、CacheFlow、FSL-26-02 | Cake 會合點自然形成（沒有估計器）；CacheFlow 有 DP 但假設互不干擾；FSL-26-02 跨 1,764 組硬體但沒有負載 | 加入長度、負載、kernel 三項，並在沒看過的格子上驗證 | 低中 |
| **D 原生索引** | **Janus（2609.36938）** | 原生稀疏模型、KV 在 SSD、多輪 append prefill、用模型自己的 indexer 預測要讀的（recall >96%）、漏的補讀、無損 | 沒選中的改讀粗版（分級）；低於 exact union 的有損預算；dense 模型；512-token／NFS 粒度下 union 的大小 | 低（併進 A） |
| 業界實務 | TensorRT-LLM NVFP4 cold-page；vLLM TieringOffloadingSpec；DeepSeek-V4／V4.1 | 冷層才壓成 4-bit（只看記憶體壓力）；多層卸載；V4 的 prefix 命中時整段還原壓縮後的 KV、V4.1 還原時重算 SWA | 依 GPU 負載決定「壓、丟、算」；查詢導向的分級還原 | — |

---

## 1. 方向 A：漸進式、查詢導向的還原（副線）

（L8a §1 的 31 篇裡挑最相關的；完整表與威脅卡在 L8a。）

| 論文 | 出處 | 做了什麼 | 和我們的差別 | 重疊 | 深度 |
|:--|:--|:--|:--|:--|:--|
| **ReadKV**（Bhandari 等 4 人） | arXiv 2610.11245，2026-10-08，preprint | 每個 K／V 純量存成 8-bit 漸進碼；decode 時 query 決定 K 每個 channel 讀幾位，再用重建 K 的注意力決定 V 每個 token 讀幾位（p.1、§3、Fig.1） | decode、HBM；prefill 用 dense（p.10）；粒度是純量／channel／token；慢速層只在 Discussion 提一句（p.15）；沒有期限、沒有重算 | 部分（最高） | 〔全文〕 |
| Xie et al.（R. Xie 等 7 人） | arXiv 2503.18869，2025-03-24，venue NOT_VERIFIED | 記憶體控制器把 KV 存成 bit-plane（§III p.4）。BF16／FP8／FP4 分頁混用只出現在一個**動機用的困惑度實驗**（§II-C Table II p.3，Llama-3.1-8B／BookSum），結果 11.87／11.60 優於 Quest top-5 BF16 的 12.49 | **V8 更正**：原文沒說用 Quest 排名、也沒說排名外的頁不讀；§IV-B 的硬體動態量化評估對象是**權重**，不是 KV。所以它不是「查詢導向分級讀 KV」的系統，只是 bit-plane 存放的前例 | 相關 | 〔全文〕；V8 核對 |
| **Fathom**（Kalyanarangan） | arXiv 2609.17652，2026-09-15，preprint | host memory 的 4-bit K 存成 channel-major bit-plane，每個 query 決定每個 channel 讀幾個 plane，用來做 top-k（§3.2–3.6） | 只用在選擇用的 K 索引；沒選到的不參與；decode；沒有 SSD | 部分 | 〔全文〕 |
| RetroInfer（Y. Chen 等 19 人） | arXiv 2505.02922；PVLDB 19(5), 2026 | GPU 存群的 centroid 與 value 總和，估計沒取回群的注意力；重要群從 CPU 取精確 KV（p.6 Eq.2） | 粗表示是群摘要，不是精確 KV 的前綴，不能補精修；decode | 部分 | 〔全文〕 |
| AnchorKV（Khalaf 等 5 人） | arXiv 2608.02901，2026-08-03，under review | 每個 token 有 anchor 投影當 base；只為部分 token 存量化殘差；「沒有位置離開 softmax」（p.1） | 選擇在 prefill 結束時用觀察窗決定（＝寫入時）；HBM；沒有讀回 | 部分 | 〔全文〕 |
| QAQ（Dong 等 4 人） | arXiv 2403.04643 | 依注意力決定每個 token 的位元；CPU 留未量化副本，要升精度時取回（p.5） | 不巢狀；decode；依注意力歷史，不是新問題 | 部分 | 〔全文（該段）〕 |
| Cocktail（W. Tao 等 5 人） | arXiv 2503.23294；DATE 2025（arXiv Comments） | 用外部 encoder 算 query 與 chunk 的相似度，給 FP16／INT4／INT2（§III p.3） | 單輪；從剛 prefill 的 FP16 量化；不巢狀、沒有層級 | 部分 | 〔全文〕 |
| SpeCache | arXiv 2503.16163 | VRAM 留 1–2 bit 副本；每個 decode step 從 CPU 取 top-k 的 16 bit | decode、每步都挑、副本常駐 GPU | 部分 | L7d〔部分〕 |
| Lynx | arXiv 2607.01831 | 8-bit 拆成高 4 bit／低 4 bit 兩條流，P/D 傳輸時先用高位推測 decode | 沒有選擇，全部都補 | 部分 | L5c |
| IMPRESS | FAST'25 | 還原時依新請求挑 | 沒挑到的不用（＝丟）；單輪 | 部分 | L7d〔部分〕 |
| CacheGen §9 | SIGCOMM'24 | 未來工作：像 SVC 一樣漸進串流 KV（p.12） | 沒有查詢導向、沒實作 | 相關 | 〔全文（該段）〕 |
| KV Offloading for Context-Intensive Tasks（Bocharnikov 等 5 人） | arXiv 2604.08426 | §4.1：ShadowKV 的 rank-160 SVD **K 壓縮**在「context-intensive」任務上，即使多載 10 倍 token 也追不上 full attention；§4.3（v5）：逐 token 2-bit（HIGGS）量化 K 做選擇，勝過 ShadowKV 的 chunk-8 landmark 和 LRQK rank-32 | 不是對手，是設計上的警告：低秩 K 當壓縮會掉、當選擇用的代理也輸給逐 token 低位元 K；我們的 base 不能只放低秩係數。（V8 更正：§4.1 講的是壓縮，不是選擇；和 LRQK 的比較只在 v5） | 相關（關鍵） | 〔全文〕；V8 核對 |

**審稿人一定會要的對照**（L8a §6；已寫進 [05](05_poc_plan_h200.md) §4.2）：全部 BF16 載入與 Cake；同位元組只讀 base；同一漸進碼、同讀取預算但查詢無關（ReadKV 的對照）；寫入時決定精修（AnchorKV／MiKV 式）；丟塊與 oracle；便宜索引（Quest-512、1–2 bit K 掃描、ShadowKV landmark）；SpeCache 式逐步取回；全部依序精修到期限為止。任務要有高資訊密度的（Text2JSON、LongProc、Loong）和多輪或 agent session，而且問題要在 KV 存下之後才來。

---

## 2. 方向 B：資源感知的還原（主線）

（L7b 已查；[L8b](lit/L8b_contention_aware_restore.md) 補 SM 分割、copy engine 對 SM 搬運、干擾模型、排隊。）

**L8b 的判定**：整個組合沒查到；每一軸都有局部前例。主張只能寫成「還原成本是隨當下爭用類型而變的資源向量，最好的計畫會翻」，不能寫成「第一個做載入／重算切分」或「第一個做 SLO 感知的還原」。

### 2.1 三個最大威脅（L8b §0 二、§2）

| 論文 | 出處 | 做了什麼 | 沒做的 | 深度 |
|:--|:--|:--|:--|:--|
| **Cascade** | arXiv 2608.06557，2026-08-06，UBC＋Microsoft Azure Research＋NVIDIA，venue NOT_VERIFIED | 每個請求的延遲預算決定最多從第 k 層搬回多少位元組，其餘改成 GPU 重算（§IV-C Eq. 6–7 p.7）；最少剩餘預算優先（Alg. 1 p.7） | 成本只有「每條連線的固定開銷＋位元組／頻寬」；負載只看 active decode 請求數（§IV-B p.6）；爭用塞進離線固定的 γ（Eq. 3 p.6）；沒有格式、解碼路徑、SM 配額；主要結果來自擴充的 Vidur 模擬器（§V-A p.8） | 〔全文〕 |
| **Bidaw** | USENIX FAST'26 | 依「等待時間／KV 大小」（HRRN）排第二層的讀取（§3.2 Eq. 1 p.7）；把轉回 KV 的運算放到低優先權 stream，理由是觀察到超過三成 SM 閒置（§4 p.10） | 「SM 有空閒」是靜態假設，不是即時量測；我們量到忙時低秩重建慢 2.8 倍 | 〔全文〕 |
| **vLLM PR #42212** | 2026-06-03 合併 | CPU→GPU 載入依 payload 大小選 `cuMemcpyBatchAsync`（copy engine）或 Triton SM gather（descriptor ≥16 且最大 payload <28 KiB 時走 kernel）；kernel 固定 12 個 SM，理由是 12／16／20 SM 的端到端 req/s 相同，選最小的以少搶 decode | 門檻與 SM 數都是靜態常數 | 〔PR〕 |

**次要威脅**：py-kvcache（arXiv 2609.11744）把「持續估計 prefill 時間、層級頻寬、佇列延遲並更新決策」與「爭用」列為未來工作（§10.2 p.22）→ 有被搶先的風險。SGLang PR #45945（2026-10-23，open）有依連線暫停／恢復背景傳輸的機制，但只做暫停、不選計畫。

### 2.2 其他前作

| 論文 | 出處 | 做了什麼 | 和我們的差別 | 重疊 | 深度 |
|:--|:--|:--|:--|:--|:--|
| **KVServe**（Z. Liu 等 12 人） | arXiv 2605.13734；SIGCOMM 2026 | 離線用貝氏搜尋找 Transform×Quantizer×Codec 的 Pareto 組合；線上每個請求挑一次；EWMA 殘差＋ε-greedy 修正 | context 是（工作負載、頻寬、SLO、品質下限），**不含 GPU 使用率**（§3.1、§6）；沒有改變 GPU 負載的實驗；動作集沒有低秩 | 部分（最大威脅） | L5c〔全文〕 |
| **HCache**（S. Gao、Y. Chen、J. Shu） | arXiv 2410.05004；EuroSys 2025 | 每層決定從 hidden state 用 GEMM 投影回 KV，或傳 KV、或重算；依離線 profile 解切分（§4.1） | 離線定一次；沒有評估和其他 GPU 工作的競爭（§6.2–6.3 都是 batch 1）。GQA 模型的 hidden state 比 KV 大，好處變小〔判讀〕 | 部分（機制最像） | L7b〔部分〕 |
| **Tutti**（S. Qiu 等 9 人） | arXiv 2605.03375 | GPU 主導的 SSD KV 儲存；I/O kernel 放在獨立的 green context，依離線 slack 表排程 | 有干擾意識，但不選格式、不壓縮；slack 是離線查表 | 部分 | L7b〔部分〕 |
| ShadowServe、KVCodec | arXiv；SIGCOMM 2026 | 量到 GPU 解壓會和模型互相拖慢；用 SmartNIC、NVDEC 硬體隔離 | 問題一樣，解法相反（隔離，不是選擇） | 部分 | L5c〔全文〕／L7b〔部分〕 |
| Cake（S. Jin 等 4 人） | arXiv 2410.03065；ICML 2025 | 前段重算、後段載入，兩邊同時跑到會合 | 本身會隨 GPU 忙碌多載少算，但只選「算或載」、格式在寫入時就定 | 相關 | 〔摘要〕 |
| Nexus、DuetServe | arXiv | 同一張 GPU 上依預測把 SM 分給 prefill 和 decode | 把爭用放進 SM 分配，但對象不是 KV 還原 | 相關 | 〔摘要〕 |
| py-kvcache | arXiv 2609.11744 | 離線量各層的 break-even prefix 長度，查找時決定載入或重算 | 服務期間不量也不擬合（§6.6 p.17）；動態估計與爭用列為未來工作 | 相關（搶先風險） | L8b〔全文〕 |
| Krul、CacheTune | arXiv | 求重算比例（Krul：min\|T_C − T_L\|；CacheTune：離線校準＋golden-section search） | 都是靜態或離線校準，沒有 GPU 負載訊號 | 相關 | L8b〔部分〕 |
| When KV Is Slower Than GPU | arXiv，2026-10 | T_load＝T₀＋ns／B 對 T_recompute＝n／R | R 取 replay 中位數；多引擎爭用是估計的、沒實測（§8 p.8） | 相關 | L8b〔全文〕 |
| NanoFlow | OSDI'25 | 裝置內 nano-batch 重疊；有 KV offload／load | offload／load 不在資源模型裡；只量到慢 3.0%（§6.4） | 相關 | L8b〔部分〕 |
| Bullet、MuxWise、semi-PD | arXiv 2025 | prefill／decode 空間共用＋即時效能模型（Bullet：roofline × SM 比例） | 對象不含 KV 還原（Bullet §3.2.3 p.6）；Bullet 的模型形式可借來當我們的資源向量模型 | 相關 | L8b〔全文〕／〔摘要〕 |

### 2.3 空間共用：沒有一個把 KV 還原放進 SM 分配（L8b §0 三）

查過的 12 個系統：LithOS、Orion、REEF、Paella、MuxServe、semi-PD、Bullet、MuxWise（SGLang PDMux）、NanoFlow、POD-Attention、RAPID-Serve、Hummingbird。最接近的是 NanoFlow（有 offload 但不在資源模型裡）與 Tutti（I/O kernel 放獨立 green context，靜態）。SGLang PDMux 的 green context 只切 prefill／decode 兩條 stream；vLLM、TensorRT-LLM、Dynamo 這輪沒找到 green context 的 PR。

### 2.4 copy engine 對 SM 搬運（L8b §0 四）

- SM 搬運會搶 SM 與 L1／L2（vLLM #42212 討論、SGLang #44029、py-kvcache §5.1、ConCCL §VI-A、NCCL 2.28 CE collectives）。
- copy engine 不搶 SM，但仍搶 HBM 頻寬（ConCCL §VII-A），小傳輸有固定成本的懸崖（ConCCL §VI-C：<32 MB 時最多慢 4 倍），還有 copy engine 佇列的阻塞問題（SGLang #38449）。
- 現行引擎全是靜態選路：vLLM 28 KiB＋12 SM；SGLang `--hicache-io-backend` 預設 `kernel`；LMCache 的解壓位置在設定時決定（RFC #4498）。

### 2.5 審稿人一定會要的對照（L8b §0 八；已寫進 [05](05_poc_plan_h200.md) §10 X4）

每台硬體事後最好的固定格式；Cake；Cascade 式預算切分；py-kvcache／CacheTune 式離線 break-even；KVServe 式 EWMA 殘差；vLLM 靜態 28 KiB／12 SM 與 SGLang `kernel`／`direct`；靜態 green context／MPS 預留 SM（Tutti 式）；還原順序 FCFS vs Bidaw HRRN vs Cascade 最少剩餘預算優先；單一使用率純量 vs 資源向量。

---

## 3. 方向 C：還原方式的成本模型

| 論文 | 出處 | 做了什麼 | 我們還剩的 | 深度 |
|:--|:--|:--|:--|:--|
| Cake | ICML 2025 | 兩條線自然會合，沒有估計器 | 事前預測會合點與兩種做法的時間 | L4、D6 卡 |
| CacheFlow（S. Nian 等 6 人） | arXiv 2604.25080 v2 | 多 session 還原的 3D 平行，DP | 附錄 A.1 p.14 假設計算與載入互不干擾 | L4〔全文〕 |
| Stony Brook FSL-26-02 | 技術報告，2026-08，非同儕審查 | H200、A5000 實測＋1,764 組硬體的模型；還原時間差 2–3 個數量級 | p.19：各組態獨立評估、沒有爭用；p.22：認為還原與重算可以並行 | L4〔部分〕 |
| Where Should the KV Cache Live?（S. D. Tumkur 等） | arXiv 2609.16215 | GPU／CPU／SSD 的放置政策 | 長壽命 session 的放置，不是還原方式 | L4〔全文〕 |

---

## 4. 方向 D：模型自帶的稀疏注意力索引

| 論文／系統 | 出處 | 做了什麼 | 我們還剩的 | 深度 |
|:--|:--|:--|:--|:--|
| **Janus**（SJTU 等 8 人） | arXiv 2609.36938，2026-09-29，venue NOT_VERIFIED | 原生稀疏模型（DeepSeek-V4-Flash／Pro、GLM-5.2），KV 在 SSD，多輪 agent 的 append prefill；用較早層的 hidden state 預測 indexer 要讀的（recall >96%，p.7），漏的補讀，整體無損；TTFT 比 Tutti／LMCache 平均快 1.22–1.85 倍 | 分級（沒選中的讀粗版）、有損預算、dense 模型、512-token／NFS 粒度 | L8d〔全文〕 |
| HiSparse（SGLang）、ESS、SAC、FlashMemory、NOSA | 文件、arXiv | decode 時只在 GPU 留 index K、選擇性取回；NOSA 原生就為 CPU offload 設計 | 都在 decode | L8d |
| Native Top-k study（Meituan） | arXiv 2512.03494 | DeepSeek-V3.2-Exp 的 indexer top-k 和逐 head 精確 top-k 的重疊平均約 60%（HELMET 128K，§3 p.5） | 原生索引對「答案塊」的 recall 對完整 K 上界，沒人量過 | L8d〔全文〕 |
| DeepSeek-V4（§3.5.2）、V4.1（§3.2.1） | 技術報告 | prefix 命中時**整段還原**壓縮後的 KV，不做選擇性還原 | 說明工業界目前沒有在讀回邊界做選擇 | L8d〔全文〕 |

---

## 5. 業界實務（[L8c](lit/L8c_practice_and_trends.md)）

| 系統 | 做法 | 和我們的關係 |
|:--|:--|:--|
| TensorRT-LLM NVFP4 cold-page | KV 頁在 host 或磁碟時才存成 NVFP4，GPU 上維持原精度；只看記憶體壓力；只支援 Blackwell（SM100／103／107）[R11][R12] | **路 1 的「慢層全部壓」是業界作法**；論文要點名 |
| vLLM（v0.32）TieringOffloadingSpec | fs、p2p、obj 次層；I/O 背壓時丟掉寫入 [R36] | 「什麼條件下不寫」的現成對照 |
| vLLM KV dtype | `auto`（預設）、fp8、`nvfp4`（SM90 走軟體反量化）、`int4_per_token_head`、TurboQuant [R1][R4][R5][R6] | 現成基準，不用自己做 |
| DeepSeek-V4.1-Flash | FP4 主 KV 寫進模型、KV 持久化到 SSD、還原時重算 SWA（Bounded Replay）[R19] | 「重算 vs 載回」和「冷層低精度」都是活躍的工業方向；學界的貢獻要是**決策準則**，不是機制 |
| 各主流引擎 | 都有多層卸載；**這輪沒找到任何系統依 GPU 負載決定壓、丟、算** | 支持方向 B 的空白 |

---

## 6. 獨立抽查（verifier）

零前文 verifier（[V8](lit/V8_verifier.md)，self-review，不是 cross-model）回到原文核對 14 條關鍵說法：**11 條 OK、3 條更正、0 條查不到**。標題、第一作者、arXiv id、v1 日期全部相符。

| # | 說法 | 結果 |
|:--|:--|:--|
| 1 | Janus：原生稀疏模型、SSD、預測 indexer 讀取 recall >96%、無損、TTFT 平均快 1.22–1.85 倍 | OK（§4.2 p.7、§5.2 p.10） |
| 2 | ReadKV：8-bit 漸進碼、query 決定 K channel 位元、attention 決定 V token 位元、prefill dense、Discussion 提到慢速大記憶體 | OK（§3 p.4、§5 p.10、§7 p.14–15） |
| 3 | Xie et al.：Quest 排名、排名外不讀 | **更正**：只有 bit-plane 存放是真的；分頁混精度只是動機實驗；沒寫 Quest 排名或不讀；硬體評估對象是權重（已改 §1 表） |
| 4 | Fathom：host memory 的 bit-plane K、逐 query 決定讀幾個 plane、用來選 top-k | OK |
| 5 | Cascade：預算切分（Eq. 6–7）、負載只看 decode 數、離線 γ、Vidur 模擬、機構 | OK |
| 6 | Bidaw：HRRN 排序、重建放低優先權 stream、理由是 >30% SM 閒置 | OK |
| 7 | vLLM #42212：copy engine 或 Triton kernel、n≥16 且 <28 KiB、固定 12 SM | OK |
| 8 | TensorRT-LLM NVFP4 cold-page：只在 host／disk 編碼、SM100／103／107、記憶體壓力觸發 | OK（官方 docs、PR #18091） |
| 9 | KVServe：context 不含 GPU 使用率、EWMA＋ε-greedy、沒有改變 GPU 負載的實驗 | OK |
| 10 | NVIDIA 文件警告 Qwen2.5-7B 的 FP8 KV | **更正**：出處是 TensorRT **Edge-LLM** 的〈FP8 KV Cache〉使用手冊，原因寫的是**最後一層**的 KV 值很大，也點名 Qwen2.5-VL-7B-Instruct，並指向 TensorRT-LLM issue #4218（已改 07） |
| 11 | 2604.08426：低秩 K 選錯 | **更正**：§4.1 講的是 rank-160 SVD 的 K **壓縮**追不上 full attention；§4.3 的逐 token 2-bit 選擇勝過低秩代理成立，但和 LRQK 的比較只在 v5（已改 §1 表、04、05） |
| 12 | ICPE 2027 Research Track 與 Emerging Research Track 的條文與日期 | OK |
| 13 | ISPASS 2027 主題與日期 | OK |
| 14 | HotOS XXI 日期、頁數、地點 | OK |
