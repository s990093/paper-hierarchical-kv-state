# L7b　查新複查：爭用感知的還原格式選擇（候選 2）＋ I1 多 session 同時還原＋ I3 agent 暫停時 KV 的去留

> **日期**：2026-10-22（UTC）
> **執行者**：文獻 agent。只上網、只讀 repo。**沒開任何 GPU 行程，沒 import torch／vllm，沒裝套件，沒改任何既有檔案**；只寫了本檔。
> **怎麼查的**：
> - WebSearch 約 30 次，另外用 arXiv 站內搜尋「KV cache offloading」「KV cache interference」，依公告日期排序各看一頁，補 9–10 月的新稿。
> - 要引用的每一篇，都用 WebFetch 打開 arXiv abs 頁（或官方頁、GitHub issue），核對標題、作者、版本日期與 Comments 欄。
> - 需要細節的，用 WebFetch 讀 arXiv HTML 的指定章節。**WebFetch 是工具用小模型抽取，不是逐字核對**，所以凡是讀 HTML 的一律標〔部分〕。引號裡只放短語，並附原文節號。
> - 沒有下載 PDF，沒有看圖。
>
> **先讀過的本專案檔**：
> - 候選與結果：`stage3_candidates.md`；`results_analysis.md`（§4.3、§4.9、§4.12、§綜合）。
> - 模擬與設定：`i3_pause_sim.md`（結論段）；`x3_time_howto.md`（背景負載設定）；`code/m7_busy.py`（背景負載的實作）。
> - 前幾輪文獻：`lit/L5c`、`lit/L2`、`lit/L3`、`lit/L4`（結論段、總表、參考清單）。
>
> **L2／L3／L4／L5c 已寫過的卡片，本檔只引用編號、不重抄。**
> **本檔是 self-review，不是 cross-model review**（CLAUDE.md §5）。

**閱讀深度標記**
- 〔全文〕：這輪**沒有**任何一篇讀全文。寫「L5c〔全文〕」的，是引用前一輪的閱讀結果。
- 〔部分〕：用 WebFetch 讀 arXiv HTML 的指定章節，或 GitHub issue 本文＋留言、官方部落格；內容由工具抽取。
- 〔摘要〕：只開了 arXiv abs 頁，或論文的官方頁（會議頁、論文庫頁）。
- 〔標題〕：只在搜尋結果或 arXiv 列表上看到標題，沒有打開 → 核對狀態一律 NOT_VERIFIED。
- 〔判讀〕＝我對原文的解讀；〔推論〕＝我的推測；〔算術〕＝我自己算的，公式寫在旁邊。
- 「沒查到」只代表這輪的搜尋沒有找到，**不等於不存在**（CLAUDE.md 規則 7）。

---

## 0. 結論

1. **候選 2 的新意還在，但說法要改。**
   - 新意：沒查到任何論文或系統，把「執行期的 GPU 忙碌」當成選 KV 還原格式（或決定在哪裡解壓）的輸入。
   - 要改的說法：「前人都假設計算與搬運互不干擾」不成立。
     - ShadowServe、KVCodec、Tutti 都明寫：解壓或 I/O kernel 會和模型搶 GPU。
     - 它們的處理方式是靜態隔離（SmartNIC、NVDEC、green context 加離線 slack 表）。
     - KVServe 則把 GPU 負載當成誤差，用 EWMA 殘差吸收。
2. **真正剩下的新意：動作集裡要有「GEMM 型重建」的格式。**
   - 前作自適應的動作集裡，解碼都是逐元素型的。GPU 忙不會讓這類格式的排序翻轉——我們的 X3 STOP-0 也看到同樣的事。
   - 唯一有 GEMM 型還原表示的是 HCache（hidden state 投影回 KV），但它用離線 profile 定一次。
3. **最大威脅是 KVServe**（SIGCOMM'26，<https://arxiv.org/abs/2605.13734>）。
   - 題目一樣：每個請求在執行期挑壓縮組合。
   - 如果 GPU 負載持續偏高，它的殘差修正也可能自己學會改選 INT4。
   - 機制上最像的是 HCache（EuroSys'25）。
4. **I1：沒查到能推翻「收掉」的反證。**
   - 新看到 ObjectCache：在頻寬受限的遠端物件儲存上，多請求聯合分頻寬比平均分好 1.2–1.8 倍。
   - 但它的對照是「平均分」，不是各自用 Cake；場景也和我們不同。
5. **I3：「知道 session 結束」的機制已經有人在做，但沒人量過它值多少。**
   - 已有的機制：Continuum v7 在程式最後一個請求後就釋放 KV；vLLM RFC #57103、llm-d #2584 正在提介面。
   - 依下層裝置換規則：只有 CrossKV 做了 SSD 對重算的頻寬交叉點。
   - NFS 層的 agent KV 策略：沒查到。

---

## 1. 對照表

「重疊程度」只用三級：**完全重疊**、**部分重疊**、**相關但不同**。這輪**沒有找到完全重疊的**。

### 1.1 候選 2：依 GPU 忙碌與連線，在還原時選格式

| 論文 | 年份／出處 | 做了什麼 | 重疊程度 | 理由（對到原文哪裡） | 閱讀深度 |
|:--|:--|:--|:--|:--|:--|
| **KVServe** | 2026／SIGCOMM'26（arXiv Comments） | 離線用貝氏搜尋，從 Transform×Quantizer×Codec 找出 3D Pareto 組合；線上每個請求挑一次；用 EWMA 殘差＋ε-greedy 修正「離線估的」和「實際的」落差 | **部分重疊（最大威脅）** | context＝(工作負載, 頻寬, SLO, 品質下限)，**不含 GPU 使用率**（§3.1、§6）。GPU 負載只被當成漂移（§6.2）。沒有任何實驗改變 GPU 負載（§7.2、§7.4 只掃頻寬）。動作集**沒有低秩**（§5.1） | 〔部分〕＋L5c〔全文〕 |
| **HCache** | 2024–25／EuroSys'25 | 每層決定：從 hidden state 用 GEMM 投影回 KV，或改用傳 KV／重算。依離線 profile 的計算與傳輸速度解切分 | **部分重疊（機制最像）** | 「用計算換位元組」的還原表示，依計算／I/O 比來選。但原文說 "offline profiling of the hardware characteristics"（§4.1）；沒有評估和其他 GPU 工作的競爭（§6.2–6.3 都是 batch＝1） | 〔部分〕 |
| **ShadowServe** | 2025／arXiv（venue 沒查到） | KV 的擷取與解壓整個搬到 BlueField-3 SmartNIC | **部分重疊**（問題一樣，解法相反） | 量到 GPU 解壓與模型互相拖慢（L5c 引 p.3：L40S ≥25–30%）。用硬體隔離處理，沒有任何選擇 | 〔摘要〕＋L5c〔全文〕 |
| **KVCodec** | 2026／SIGCOMM'26 | KV 編成 H.265 無損視訊、用 NVDEC 解；fetcher 依上一個 chunk 的傳輸時間估頻寬、調解析度 | **部分重疊** | §2.2 量到 CacheGen 的 CUDA 解壓讓 prefill 多 50%、decode 多 20%（圖 5）。但執行期只看頻寬（§3.3.2）；品質固定，解碼固定走 NVDEC；沒有受控的背景負載實驗 | 〔部分〕 |
| **Tutti** | 2026／arXiv | 以 GPU 為主導的 SSD KV 物件儲存。I/O kernel 和模型分在不同 green context；I/O 依 slack 排程 | **部分重疊**（有干擾意識，但不選格式） | 原文："I/O kernels also compete with model execution for SM resources"（§3.3）。slack 是離線 profile 出來的查表（§3.3–3.4）。沒有壓縮，也不選格式 | 〔部分〕 |
| SplitZip | 2026／arXiv | BF16 指數碼本的無損 KV 壓縮 | 相關但不同 | §5 一句話承認："for short contexts or compute-bound regimes, codec overhead may outweigh the reduced transfer volume"。但沒有量，也沒有自適應；編解碼是在隔離環境量的（§4.1） | 〔部分〕＋L2 B1 |
| CacheGen | 2024／SIGCOMM'24 | 讀回時逐 chunk，依量到的吞吐選量化等級，或改送文字重算 | 部分重疊 | 「讀回時選格式」已做過，但只看頻寬。附錄 D 有一張「GPU 可用週期 × 頻寬」熱圖，那是**評測軸**，不是決策輸入（L5c §2.2） | L5c〔全文〕（本輪沒重讀） |
| AdaptCache／EvicPress | 2025／BigMem@SOSP'25；arXiv | 依層級、命中頻率、品質，選丟 token 或量化，以及壓縮率 | 相關但不同 | 目標是容量；解壓開銷只在離線量一次（L5c §2.2） | 本輪〔摘要〕核 abs；L5c〔全文〕 |
| Cake | 2024–25／ICML'25 | 前段重算、後段載入，兩邊同時跑到交會為止 | 相關但不同 | 本身就會隨 GPU 忙碌多載、少算（摘要："adapting to fluctuating resource availabilty"〔原文拼字〕）。但它選的是「算或載」，不是格式；格式在寫入時就定了 | 〔摘要〕 |
| HACK | 2025／arXiv | 2-bit KV 不反量化，直接拿來算 | 相關但不同 | 量到反量化的開銷（L2 B6），解法是把反量化去掉，沒有選擇 | 〔摘要〕＋L2 B6 |
| Lynx | 2026／arXiv（Comments：原投 SIGCOMM'26） | 高位元先送、低位元後送；先用高位元推測 decode，再驗證 | 相關但不同 | 讀回時漸進，但摘要沒提 GPU 負載 | 〔摘要〕 |
| ElasticKV（"Does Execution Require Target KV Fidelity?"） | 2026／arXiv | 執行期依記憶體壓力調 KV fidelity（多一個中間態、雙模式 attention） | 相關但不同 | 「執行期調 KV 表示」已經有人做；但觸發條件是記憶體壓力，作用在 GPU 內 | 〔摘要〕 |
| MorphServe | 2025–26／MLSys'26 | 高負載時把部分層換成量化版、依記憶體壓力調 KV 容量 | 相關但不同 | 「依負載換精度」已經有人做；但換的是權重 | 〔摘要〕＋MLSys 頁 |
| MetaKV | 2026／arXiv（venue NOT_VERIFIED） | 每個 prompt 依延遲與峰值記憶體預算，挑一組壓縮設定 | 相關但不同 | 輸入是使用者給的預算，不是 GPU 現況；也不搬到下層 | 〔摘要〕 |
| Understanding GPU Resource Interference One Level Deeper | 2025–26／arXiv | 把 GPU 干擾拆成 block scheduler、運算單元、快取、記憶體頻寬幾類，量各工作的敏感度 | 相關但不同（背景） | 支持「干擾依資源類型而不同」——這是 §4 補實驗 E1 的依據 | 〔摘要〕 |
| Nexus；DuetServe | 2025（DuetServe v2 2026）／arXiv | 同一張 GPU 上，依預測動態把 SM 分給 prefill 和 decode | 相關但不同（背景） | 已把爭用放進 **SM 分配**的決策。Nexus 摘要說記憶體頻寬爭用是主要瓶頸——〔推論〕真實 decode 背景下，被拖慢的可能是逐元素格式，而不是 GEMM 格式 | 〔摘要〕 |
| Splitwise | 2023–24／arXiv（venue 見 §5） | PD 分離，逐層非同步傳 KV、和 prompt 計算重疊 | 相關但不同 | §IV-C 承認可能有 "performance interference and increase the TTFT"；§VI-A 量到開銷 <7%。不壓縮，壓縮只列為未來工作（§VII） | 〔部分〕 |
| DistServe；Mooncake | 2024／OSDI'24；arXiv（v5 2026-10-08） | PD 分離；以 KVCache 為中心的分離架構 | 相關但不同 | 摘要都沒提 KV 壓縮，也沒提傳輸和計算的爭用（細節見 L3 G1、G3） | 〔摘要〕 |
| LMCache RFC #4498、MP serde、TRT-LLM cold-page NVFP4 | 2026 | 靜態設定的卸載壓縮 | 相關但不同 | 見 L5c §2.1。RFC 寫「CPU 壓可以避開 GPU 爭用」，但那是陳述，不是決策 | L5c |

### 1.2 I1：多 session 同時還原

| 論文 | 年份／出處 | 做了什麼 | 重疊程度 | 理由 | 閱讀深度 |
|:--|:--|:--|:--|:--|:--|
| CacheFlow（v2） | 2026／arXiv | 批次 DP，聯合決定每個（請求, 層區塊）要算還是要載 | 部分重疊 | L4 C-I1 的最大威脅；附錄 A.1 假設兩者不干擾 | L4〔全文〕；本輪核 abs（v2 已改名，見 §3.3） |
| Cascade | 2026／arXiv | 依剩餘 SLO 預算排序；每個請求決定還原多少、其餘重算 | 部分重疊 | 見 L4；評測在模擬器裡 | L4〔全文〕 |
| **ObjectCache** | 2026／arXiv | 從 S3 相容物件儲存逐層取回 KV。多請求共用頻寬上限時，依每層的計算窗分配頻寬（Stall-opt） | **部分重疊** | 「多請求同時還原的聯合分配」做了：對「平均分」少 1.2–1.8 倍額外 TTFT（摘要、§5.7）。但它不選重算、不選格式；計算窗當常數；不考慮 GPU 爭用（§3.6） | 〔部分〕 |
| Strata | 2025–26／OSDI'26 | GPU 輔助 I/O，加上快取感知排程（處理 delay hit） | 相關但不同 | 排程會看載入延遲；摘要沒提計算和 I/O 的爭用 | 〔摘要〕（USENIX 頁） |
| MTDS | 2026／Complex & Intelligent Systems | 多張 GPU 往 DRAM 卸出時，依預期重用排優先序，處理 PCIe 爭用 | 相關但不同 | 排的是**卸出**，不是還原；不管 GPU 計算爭用 | 〔部分〕（Springer 頁） |
| Unified KV Pooling | 2026／arXiv | 多個 DRAM 模組加 SSD 合成一個池、依頻寬分散；用 SPDK 繞過檔案系統 | 相關但不同 | 是單一請求的多裝置並行讀，不是多 session 排程 | 〔摘要〕 |

### 1.3 I3：agent 工具暫停時 KV 的去留

| 論文 | 年份／出處 | 做了什麼 | 重疊程度 | 理由 | 閱讀深度 |
|:--|:--|:--|:--|:--|:--|
| **Continuum v7** | 2025–26／arXiv（L3 記為 ICLR'26 workshop poster） | 工具呼叫時把 KV 釘在 GPU 一段 TTL；§6.3 加了 SSD 層 | **部分重疊** | Alg. 1 在 "the last request of its program" 時釋放；§5.2 解除同一 program_id 的所有釘 → 已經用了「session 結束」訊號。但沒有單獨量它的價值（§6.4 消融只比 FCFS 和靜態 TTL）。SSD 只換容量，沒有依裝置的規則（§4.1 的 reload 項只看有沒有開 CPU 卸載） | 〔部分〕 |
| **vLLM RFC #57103** | 2026-09-16 開，仍 open | 可程式化 KV：保護宣告（優先序、壽命、目標層級）、release、task completion | **部分重疊（機制）** | 任務完成時，結束整棵任務樹的宣告；工具暫停時保持熱。10/19 宣布參考實作 PR #62382。沒有評估；層級選擇不看裝置速度 | 〔部分〕 |
| **llm-d #2584** | 2026-09-25 開，仍 open | session 的 open／heartbeat／close；close 立刻釋放 | **部分重疊（機制）** | 只是介面提案：沒有實驗，也沒有 CPU／SSD 分層的邏輯 | 〔部分〕 |
| **CrossKV** | 2026／西江大學碩士論文 | SSD 冷命中比重算慢時改成重算，並把結果提升到 CPU；交叉點由磁碟頻寬決定 | **部分重疊** | 是「依下層裝置速度換做法」的單點版本；只有 GPU／DRAM／SSD，沒有 NFS，也沒有 agent 暫停 | 〔摘要〕 |
| Janus（"Efficient Agentic LLM Serving over SSD-based Sparse KV Storage"） | 2026／arXiv | agent 會話的稀疏 KV 放 SSD；預測要讀的、合併讀取；主動讀的時候限制背景寫 | 相關但不同 | 和 I3 模擬裡「寫入中讀取變慢」是同一個現象，他們用限寫處理；但沒有暫停期間的去留決策 | 〔摘要〕 |
| Ask the Tool, Don't Guess | 2026／arXiv | 工具執行中回報進度，用來決定 KV 留、走、何時回來 | 相關但不同 | 解的是「何時回來」，不是「已經結束」；只有 HBM 與 HBM＋DRAM 兩種配置 | 〔摘要〕（repo 已有卡 D8_AskTheTool） |
| UNISON | 2026／arXiv | 近記憶體硬體排程：依回訪間隔選逐出；用工具等待時間當 DMA 預算 | 相關但不同 | 是硬體，SRAM／HBM 兩層；沒有 SSD／NFS | 〔摘要〕 |
| DualPath | 2026／arXiv | agent 多輪的瓶頸在 KV 的儲存 I/O：改從 decode 端載入、再用 RDMA 送給 prefill | 相關但不同 | 說明 agent 多輪的瓶頸常在儲存網路；不處理暫停期間的去留 | 〔摘要〕 |
| Tool Waiting and Re-arrival（NPU） | 2026／arXiv | 編譯期固定的 NPU 服務上，分析工具等待與再到達的成本；用模擬器從 2,077 組設定挑一組 | 相關但不同 | 「用模擬挑設定」和 I3 的做法類似；但沒有下層儲存 | 〔摘要〕 |
| Agent Memory Below the Prompt | 2026／arXiv | 邊緣裝置把每個 agent 的 Q4 KV 存到磁碟、再載回 | 相關但不同 | 是「壓縮後落盤」的 agent 版本；只有一台 Apple M4 Pro，沒有依裝置選擇 | 〔摘要〕 |
| SAGA；Policy-Driven Runtime Layer（CacheScout） | 2026／arXiv | 以工作流程為單位排程、預測跨工具呼叫的 KV 重用；agent runtime 層 | 相關但不同 | 沒有依下層裝置的規則 | 〔摘要〕 |
| LMCache on MI300X（部落格） | 2026-05-12 | MI300X 上用 vLLM 0.19＋LMCache 的 CPU 層，重播 Claude Code trace | 相關但不同 | AMD 上**唯一**找到的「agent＋卸載」數據點；只有 CPU DRAM 一層 | 〔部分〕 |
| InferCept、Tokencake、KVFlow、MORI、Parrot、Autellix、EfficientAgent、Lachesis、Cascade | — | 見 L3 §4、L4 §2 | 部分重疊／相關但不同 | 本輪沒有重讀 | L3／L4 |

---

## 2. 重要論文卡片

### A1　KVServe（SIGCOMM'26）——最大威脅〔部分：§3.1、§5.1、§6、§6.2、§7.2、§7.4〕
- **出處**：Z. Liu, X. Ma, D. Luo, H. Zhao, B. Lu, W. Huang, Y. Gu, X. Liu, Z. Wei, J. Liu, D. Tao, G. Tan, "KVServe: Service-Aware KV Cache Compression for Communication-Efficient Disaggregated LLM Serving," arXiv 2605.13734（只有 v1，2026-05-13）。Comments 欄寫 "Accepted by SIGCOMM 2026"。
- **動作集（§5.1）**：
  - Transform：Delta、Hadamard、Affine。
  - Quantizer：KIVI 式的多維量化；層級與 head 級的混合精度。
  - Codec：nvCOMP（沒有列出具體演算法）。
  - **沒有 PCA、SVD 或任何低秩**，所以也沒有任何格式要付重建的 GEMM。
- **線上控制器（§3.1、§6.2）**：
  - context＝(w, B, T_SLO, q_min)，分別是工作負載類別、有效頻寬、延遲預算、品質下限。
  - 每個（品質桶, 頻寬區間）當成一個獨立的小環境；可以選的是 2–3 個候選 profile。
  - 學的東西是殘差 δ＝T_obs − T̂，用 EWMA 追；選擇用 ε-greedy；每執行一次就更新。
  - 原文明說不需要 LinUCB 一類的 contextual bandit。
- **和 GPU 忙碌的關係**：
  - §6.2 只有一句："GPU load and queue contention change the actual (de)compression throughput"。
  - 沒有把它當輸入，也沒有任何實驗改變 GPU 負載：§7.2 用流量整形掃 5–100 Gbps，§7.4 的頻寬隨時間變，硬體只換 GPU 型號。
  - 原文沒寫解壓在 GPU 還是 CPU 上做。
- **原文數字（摘要）**：PD 分離的 JCT 最多快 9.13 倍；KV 分離的 TTFT 最多少 32.8 倍。
- **為什麼是最大威脅**〔判讀〕：
  1. 問題的說法幾乎一樣：「最佳壓縮組合會隨服務狀況變，要在執行期選」。
  2. 如果 GPU 負載**持續**偏高（例如整段時間都是 90% duty），低秩那個選項量到的時間會一直偏慢，殘差變大，ε-greedy 就會改選 INT4——等於**自己學出我們的翻轉**。
  3. 所以「依情況選」對 KVServe 式控制器的優勢，只可能出現在這三種情況：
     - (a) 負載變得比 EWMA 收斂還快；
     - (b) 同一個頻寬區間裡，負載忽高忽低，殘差被平均掉；
     - (c) 新的（頻寬, 品質）環境還在冷啟動。
  4. 它的動作集沒有低秩，所以**在它自己的動作集裡本來就不會翻**——這和我們 X3 的 STOP-0（只在已驗證的量化格式之間挑時，永遠是 INT4 最快）一致。
- **對我們的意義**：要證明的不是「沒人做自適應」，而是兩件事：
  1. 動作集加進 GEMM 型格式之後，排序會隨背景負載翻轉；
  2. 這時「把負載當輸入」比「靠殘差慢慢學」好，而且要量出好多少（§4 的 E2）。

### A2　HCache（EuroSys'25）——機制最像〔部分：§4、§4.1.2、§5、§6.1.1、§6.2–6.3〕
- **出處**：S. Gao, Y. Chen, J. Shu, "Fast State Restoration in LLM Serving with HCache," arXiv 2410.05004（只有 v1，2024-10-07）。Comments 欄寫 "EuroSys 2025"。
- **決策（§4.1.2，State Partition Algorithm）**：
  - 輸入是離線 profile 的四個時間：傳 hidden state、傳 KV、重算 token、把 hidden state 算成 KV。
  - 解出 L_H 層用 hidden state，其餘層用互補的方法（傳 KV 或重算）。
  - §4 說請求到來時，由 bubble-free 排程器產生還原方案；**沒說執行期怎麼更新 profile**。
- **和 GPU 爭用**：
  - 沒有評估和其他 GPU 工作的競爭。
  - §6.1.1 用 Poisson 到達的多輪對話，TBT 最多比理想多 4%。
  - §6.2–6.3 的還原實驗都是 batch＝1。
- **硬體**：4×A100-40G。敏感度實驗另用 A30、4090、L20、H800（表 2）。
- **為什麼像**〔判讀〕：
  - hidden state → KV 是乘 W_K、W_V 的一次 GEMM。我們的「低秩係數 → KV」是乘重建矩陣，**同一類運算**。
  - 差別在依據：HCache 依「這台機器的計算 vs I/O 比」來選；我們依「**此刻** GPU 還剩多少計算」來選。
  - 審稿人很可能問：「這不就是 HCache 的切分，加上動態 profile？」要正面回答三點：
    1. HCache 的 profile 是在 GPU 沒有別的工作時量的；
    2. 我們在同一台 MI300X 上量到，排序會隨背景負載翻轉，靜態 profile 會選錯；
    3. 我們的動作集同時有逐元素型（INT4）和 GEMM 型，HCache 沒有量化。
- **可以借來用的**〔推論〕：hidden state 是**無損**的 GEMM 型格式，正好補上我們動作集裡「無損＋GEMM」那一格。這個格子 L5c 的模式 L 沒有考慮。

### A3　ShadowServe＋KVCodec：干擾早就量過了，解法是靜態隔離
- **ShadowServe**〔摘要〕（全文見 L5c §2.2）：
  - 出處：X. Xiang, R. Joshi, Y. Liu, J. Yao, C. Zhao, J. Jiang, Y. Zhou, E. Kohler, M. Yu；arXiv 2509.16857（只有 v1，2025-09-21）；沒有 Comments。
  - 做法：資料面整個放到 SmartNIC，讓 host 的 GPU 和 CPU 不受干擾。
  - 摘要數字：頻寬 ≤20 Gbps 時，有負載的 TPOT 最多少 2.2 倍，TTFT 最多少 1.38 倍，吞吐最多高 1.35 倍。
- **KVCodec**〔部分：§2.2、§3.3.2、§4、§5.2、§5.3〕：
  - 出處：L. Mi, W. Wang, J. Chen, T. Cao, H. Dai, Y. Liu, "Efficient Remote KV Cache Reuse with GPU-native Video Codec," arXiv 2602.09725（v1 2026-02-10、v2 2026-02-12、v3 2026-05-12）。Comments 欄寫 "Accepted by SIGCOMM 2026"。
  - 干擾數字（§2.2，圖 5）：CacheGen 的 CUDA 解壓讓 prefill 多 50%、decode 多 20%。§2.2 的基準實驗用 2 張 H20 跑 LWM-7B，但內文**沒指明圖 5 是哪張卡**。
  - 記憶體：4K token 時，CacheGen 解壓的峰值記憶體 5.5 GB，是原 KV 的 2.7 倍（圖 6）。
  - 自適應（§3.3.2）：只依上一個 chunk 的傳輸延遲估頻寬，調整「張量 → 影格」的解析度。品質固定為無損 H.265；chunk 固定 10K token；解碼固定走 NVDEC。
  - 併發（§4）：某個請求吃掉全部頻寬或 GPU 記憶體時用 FCFS，否則平均分頻寬。
  - 評測（§5.2）：只有一條 0.2 req/s 的混合 trace，沒有受控的背景負載。摘要數字：TTFT 最多少 3.51 倍。
- 〔判讀〕兩篇都承認「GPU 上解壓會拖慢模型」，解法都是**不在 GPU 上解**。
  - MI300X 沒有 SmartNIC；VCN 有沒有無損模式，L2 B5 記為 NOT_VERIFIED。所以在我們的平台上，GPU 端解碼避不開——這正好是「依負載選格式」的動機。
  - 但它也表示，審稿人會問：「為什麼不乾脆把解碼搬走？」回答要放在：AMD 沒有對應硬體，而且低秩重建本身就是 GEMM，搬不到 NVDEC 上。

### A4　Tutti——有干擾意識的還原排程，但不選格式〔部分：§2.2、§3.1–3.4、§4〕
- **出處**：S. Qiu, Y. Hu, X. Wang, W. Zhu, J. Yan, H. Chen, K. Xu, K. Chen, Y. Zhang, "Tutti: Making SSD-Backed KV Cache Practical for Long-Context LLM Serving," arXiv 2605.03375（只有 v1，2026-05-05）；沒有 Comments。
- **爭用**：
  - §3.3 原文："I/O kernels also compete with model execution for SM resources"。
  - §3.2 用 NVIDIA green context 把 GPU 分成 Compute Domain 和 I/O Control Domain。
  - §3.3 也處理 SSD 端讀寫同時進行時的頻寬爭用。
- **slack 怎麼來的**：
  - 定義是「有空閒 SM、又沒有有害讀寫爭用的執行窗」（§3.3）。
  - 離線把每一層 profile 成一張查表，以輸入長度與前綴長度為索引；暖機時做一次（§3.4）。
- **沒做的**：沒有 KV 壓縮，也沒有格式選擇（§3.1 的「格式」指的是 NVMe 描述子與檔案佈局）。沒有在 GPU 有負載時直接量 I/O 對計算的拖慢（§3.3 的讀寫干擾是用 FIO 量的）。
- **硬體**：2×H100；Solidigm D7-PS1010。原文對 SSD 數量前後不一（§2.2 寫兩顆、§4 寫四顆）。
- 〔判讀〕這是目前**最接近「干擾感知還原」**的系統：它把「還原工作和模型搶 SM」當成正式的排程限制。但它決定的是**什麼時候發、發多少** I/O，而且用離線表。
  - 我們的差別：決定的是**用哪個格式還原**；依據是**執行期**量到的負載。
  - 在 ROCm 上，green context 沒有直接對應；`code/m7_busy.py` 的註解寫的是 CU 遮罩（`HSA_CU_MASK`）〔程式碼〕。

### A5　SplitZip——只有一句「compute-bound 時不划算」〔部分：§4.1、§4.2、§5、附錄 A〕
- **出處**：Y. Guo, S. Joshi, arXiv 2605.01708（L2 B1 已核 abs）。硬體是 H200 加 ConnectX-7 等（§4.1）。
- §5 承認：短 context 或 compute-bound 時，編解碼開銷可能超過省下的傳輸。
- 附錄 A 用解析模型算出約 463.2 GB/s 的「可隱藏門檻」——只有 NVLink 級的連線才會超過。
- 沒有量它和並行計算的干擾；§4.2 說壓縮可以利用「傳輸期間閒著的 GPU 計算」，但沒有量。
- 〔判讀〕這是我們讀到最接近「盈虧會因 GPU 忙而翻轉」的**一句話**，但它講的是工作負載本身的 compute-bound，不是另一個工作在搶。可以在論文裡引用為「前人已預期、但沒量也沒用」。

### A6　執行期調精度的三篇：ElasticKV、MorphServe、MetaKV〔摘要〕
- **ElasticKV**（J. Jiang, Y. Yang, P. Yang, F. Liu；arXiv 2609.33536，v1 2026-09-27）：
  - 多一個精簡的中間 KV 態，依**記憶體壓力**調 fidelity。
  - 摘要數字：高併發下 TTFT 比 vLLM 低 3.8–4.0 倍，P90 TTFT 低 9.1 倍。
- **MorphServe**（Z. Su, Z. Zhang, T. Lan, Z. Wang, H. Shen, J. Yang, Y. Cheng；arXiv 2506.02006，v1 2025-05-24、v2 2026-01-07；MLSys 2026 oral，mlsys.org 頁已確認）：
  - 高負載時把影響小的層換成量化版。
  - 摘要數字：SLO 違約平均少 92.45%，P95 TTFT 好 2.2–3.9 倍。
- **MetaKV**（M. Wang, K. Li, R. Bostandoost；arXiv 2609.07966，v1 2026-09-07）：
  - 每個 prompt 依延遲與峰值記憶體預算，從 KVQuant、H2O、RocketKV 的 10 組設定裡挑。
  - 摘要數字：受限成功率比最好的靜態設定平均高約 0.07、最多 0.135。
  - 搜尋結果裡有一個 NeurIPS 2026 頁面，但打開是 404 → venue NOT_VERIFIED。
- 〔判讀〕「依負載或資源狀況，在執行期換精度或表示」作為**一般概念**已經不新。候選 2 不能用「執行期自適應」當賣點，要用「GPU 忙碌 × 格式的計算型態」。

### A7　干擾特徵化，以及我們背景負載的盲點〔摘要＋程式碼〕
- **三篇的說法**：
  - Understanding GPU Resource Interference One Level Deeper（P. Elvinger, F. Strati, N. Enright Jerger, A. Klimovic；arXiv 2501.16909，v3 2026-02-16）：干擾分好幾種資源（block scheduler、運算單元、快取、記憶體頻寬），每種工作的敏感度不同。
  - Nexus（arXiv 2507.06608）摘要：prefill 與 decode 共用 GPU 時，記憶體頻寬爭用是主要瓶頸。
  - DuetServe（arXiv 2511.04791）：預測到延遲會變差時，才切 SM。
- **我們的盲點**〔程式碼〕：`code/m7_busy.py` 的背景負載是 8192×8192 的 BF16 matmul，用 duty cycle 調強度——是**純計算型**。X3 的「很忙」是 duty 0.9（`x3_time_howto.md`）。
- **〔推論〕兩點**：
  1. 我們量到「低秩重建忙時慢 2.8 倍、INT4 幾乎不受影響」，很可能是因為背景和重建搶的是**同一種資源**（矩陣運算單元）。
  2. 真實服務的背景主要是 decode，偏記憶體頻寬型；那時被拖慢的可能換成 INT4 反量化與 H2D，翻轉方向可能相反。
- **風險也是機會**：如果方向真的隨背景型態改變，主張可以更尖銳——「格式選擇要看 GPU 被**什麼**佔用，不只是佔用多少」。
- 這也呼應 I4 的觀察：busy2 幾乎不影響單純的搬運（results_analysis §4.2），因為計算型背景本來就不搶 copy。

### A8　順帶：Range（2026-10-15，前一輪漏掉）〔摘要〕
- **出處**：D. Shi, J. Zhang, L. Lv, B. Huang, Y. Dai, Y. He, D. Chen, H. Fu, W. Feng, S. Fu, "Range: Decoupling KV-Cache Reuse Granularity from Storage Work," arXiv 2610.21241（只有 v1，2026-10-15）；Comments 欄只有頁數。
- **主張**：從 NVMe 還原時，瓶頸是儲存堆疊上「每個物件」的開銷，不是 SSD 頻寬。
- **摘要數字**：固定還原 6 GiB 時，物件數跨 32 倍，Range 的中位 TTFT 都在約 1.8 s；in-process LMCache 從 2.12 s 升到 3.01 s。
- 〔判讀〕和我們 I4 的「逐塊 copy 的固定成本才是主因」同一個方向，是**候選 3** 的旁證；和候選 2 無關。

### B1　ObjectCache——I1 的新前作〔部分：§3.4、§3.6、§5.5、§5.7、§6.2〕
- **出處**：Y. Zhu, A. Dhakal, Y. Xiao, D. Milojicic, G. Alonso, "ObjectCache: Layerwise Object-Storage Retrieval for KV Cache Reuse," arXiv 2605.22850（只有 v1，2026-05-16）；沒有 Comments。
- **分配規則（§3.6）**：
  - 每個請求的每層傳輸量 s_i、每層計算窗 c_i；stall＝max(0, s_i/r_i − c_i)。
  - 零 stall 的速率 r_i\*＝s_i/c_i。在總上限 B、而且 r_i ≤ r_i\* 下，最小化 Σ s_i/r_i。
  - 校準版給每個請求多 δ＝5 Gbps 的餘裕（§5.7，在同一個 testbed 上掃出來的）。
  - 以 epoch 為單位重新分配。
- **沒做的**：
  - 不在載入和重算之間選；§6.2 只建議計算窗太小時改用 chunkwise、DRAM 熱快取或重算。
  - 不選格式。
  - c_i 當常數，「approximately constant across layers」（§3.6）；不建模多個 prefill 之間的 GPU 爭用。
- **硬體**：100 Gbps RoCEv2、A100 80GB、vLLM v0.19.0、修改過的 LMCache＋NIXL、Ceph RGW、DAOS（4 顆 NVMe）；模型 Llama 3.1 8B。
- **摘要數字**：64K 時只比本地 DRAM 多 5.6% 延遲；4K 時比最佳的本地逐層基線多 56–75 ms；共用頻寬上限時，額外 TTFT 比「平均分」少 1.2–1.8 倍。
- 〔判讀〕這給了 I1 一個「聯合分配有用」的正例，但不推翻我們的結論。
  - 它的對照組是「平均分」，不是「各自用 Cake」。
  - 它的場景是頻寬上限 50–80 Gbps 的遠端物件儲存，鏈路真的是瓶頸；我們的 I1 是本機 MI300X。
  - 如果要重開 I1，只值得在「多 session 同時從頻寬受限的 NFS 或遠端讀」這個場景重開。

### C1　Continuum v7——已用「程式結束」訊號，但沒量它的價值〔部分：§4.1、§4.2、§5.2、§6.3、§6.4、Alg. 1〕
- **出處**：H. Li, R. He, Q. Mang, Q. Zhang, H. Mao, X. Chen, H. Zhou, H. Zhang, A. Cheung, J. Gonzalez, I. Stoica；arXiv 2511.02230。
  - 版本：v1 2025-11-04 … v7 2026-09-08。
  - arXiv 沒有 Comments；L3 依 iclr.cc 頁記為 ICLR'26 workshop poster。
- **結束訊號**：
  - Alg. 1 在請求是 "the last request of its program" 時釋放 KV。
  - §5.2：排程器會 "proactively unpins any remaining pins with the same program_id"。
  - → 它**假設**系統知道程式何時結束，但§6.4 的消融只比 Program-Level FCFS 和 Static TTL，**沒有單獨量這個訊號值多少**。
- **SSD**：§6.3 在 B200 上跑 Llama-8B 的 SWE-bench，用 LMCache 加 400G／800G 的 SSD 層；只有定性結論："consistently improves average delay"。
- **依裝置的規則**：沒有。§4.1 的 reload 項只分「有沒有開 CPU 卸載」，由離線 profile 得來（§5.2）。
- **GPU 爭用**：建模了記憶體佔用的爭用；**reload 和計算之間的干擾沒有建模**。
- 〔判讀〕I3 說「最有價值的是知道 session 結束（NFS 下中位快 26%）」，正好量了 Continuum 假設、卻沒量的那一塊。
  - 但我們的數字是**模擬**，而且 MI300X 192 GB 沒有容量壓力。
  - 依專案原則（只靠模擬判死的方向只算暫定），這個「收掉」不是定論。

### C2　vLLM RFC #57103 與 llm-d #2584——「session 結束」的介面正在標準化〔部分〕
- **vLLM #57103**「[RFC]: Programmable KV Cache: Composable Policies for Agentic Serving」：
  - zhhangBian 於 2026-09-16 開，仍 open。
  - 保護宣告有目標、優先序、壽命、目標層級等欄位。9/24 作者說 "Pin is dropped"，改成「逐出優先序偏置＋可選 TTL」。
  - release 會移除宣告；**task completion 會結束整棵任務樹的宣告**。
  - 工具暫停不算結束，session 維持熱。
  - 10/19 作者宣布參考實作 PR #62382。
  - 留言裡有人（yinli-systems，9/18）在 L20＋Qwen3-4B＋vLLM 0.29.0 上量到：CPU→GPU 重載約每 token 6.4 µs（約 23 GB/s）；在工具等待期間預取，TTFT 少 52–57%。
  - RFC **沒說**層級選擇要看裝置速度。
- **llm-d #2584**「[Feature]: Agent-Aware KV Cache Affinity Scheduling and Cache Management」：
  - daisyEliot 於 2026-09-25 開，仍 open。
  - 提案內容：open 時釘住；heartbeat 延長 TTL；**close 立即釋放**；TTL 內沒有 heartbeat 就回到一般逐出池。
  - 沒有實驗，也沒有 CPU／SSD 分層邏輯。
  - 10/6 維護者回覆：這些在 SIG Agentic Inference 工作組已經在討論。
- 〔判讀〕「應用層告訴引擎 session 結束了」這個**機制**，上游已經在做。我們能貢獻的不是機制，而是**價值隨下層裝置變化的量測**（SSD vs NFS）。這是一份小而具體、可以交給上游討論的數據，不是一篇論文的主軸。

### C3　CrossKV——依下層頻寬換做法的單點〔摘要〕
- **出處**：배재형（Bae Jae-hyeong），西江大學 AI.SW 研究所碩士論文，指導教授 김영재，2026-08 授予。repo 的 `research_20261010_directions/D3_kappa_map.md` §5 已記過摘要。
- **內容**：在 A100＋LMCache 上，SSD 冷命中可能比重算慢；磁碟頻寬低於交叉點時，每個 context 長度都比重算慢。CrossKV 這時跳過 SSD、改重算，並把結果放進 CPU 層；重用請求的 TTFT 最多少 90%。
- 〔判讀〕這是 I3「下層換成 NFS 之後，最好的現成規則就換人」的**單一規則版本**。沒有 NFS，沒有 agent 暫停，也沒有「在多個現成規則之間選」。

### C4　Janus（agentic＋SSD 稀疏 KV）〔摘要〕
- **出處**：W. He, P. Zhang, X. Hu, C. Wang, J. Hou, Y. Cheng, P. Sun, F. Fu, "Efficient Agentic LLM Serving over SSD-based Sparse KV Storage," arXiv 2609.36938（只有 v1，2026-09-29）。摘要裡的系統名叫 Janus。
- **內容**：
  - 針對 append prefill，預測要讀哪些 KV、讓讀取和計算重疊；猜錯的頁在 attention 前補抓。
  - 合併相鄰的讀；把零散頁打包成循序寫；主動讀的時候限制背景寫。
- **摘要數字**：TTFT 好 1.57–3.69 倍（平均 1.22–1.85 倍）。
- 〔判讀〕I3 模擬用了「寫入中讀取變慢 3.306×（本地）／1.673×（NFS）」，而且結論之一是「本地 SSD 時串寫最好、NFS 時串寫最差」。Janus 證明讀寫干擾在 agent＋SSD 上是真的、而且值得專門處理。但它沒有在「留／卸／丟」之間做決定。

---

## 3. 被做掉的部分與缺口

### 3.1 已經被做掉、不能再當新意講的

| 子主張 | 狀態 | 誰做了（出處） |
|:--|:--|:--|
| 依頻寬或儲存層級選壓縮等級／格式 | 已做 | CacheGen、KVServe、AdaptCache、EvicPress（L5c B1–B2）；KVCodec 依頻寬調解析度（§3.3.2） |
| 執行期自適應挑壓縮組合（含線上修正） | 已做 | KVServe（§6.2） |
| 「GPU 上的解壓或 I/O 會和模型互相拖慢」這個**現象** | 已做 | ShadowServe（L40S）；KVCodec（prefill +50%、decode +20%）；HACK（反量化佔 JCT 17.2–37.9%，L2 B6）；Tutti（SM 競爭，§3.3）；Splitwise（定性，§IV-C） |
| 用靜態方式避開這個干擾 | 已做 | SmartNIC（ShadowServe）；NVDEC（KVCodec）；green context＋離線 slack（Tutti）；CPU 壓縮（LMCache RFC #4498，L5c） |
| 「用計算換位元組」的還原表示，依計算／I/O 比切分 | 已做（靜態） | HCache（§4.1.2） |
| 執行期依現況在「算」與「載」之間調整 | 已做 | Cake（雙向）；Cascade（依預算）；CacheFlow（批次 DP） |
| 執行期依負載或記憶體壓力調精度／fidelity | 已做（作用在 GPU 內） | MorphServe（權重）；ElasticKV（KV fidelity，依記憶體壓力） |
| 爭用放進 GPU 資源分配的決策 | 已做（不是為了 KV 還原） | Nexus、DuetServe（SM 分給 prefill／decode） |
| 多請求共用頻寬時的聯合分配 | 已做 | ObjectCache（遠端物件儲存）；CacheFlow（批次） |
| session／程式結束就釋放 KV | 機制已有 | Continuum v7（Alg. 1、§5.2）；vLLM RFC #57103；llm-d #2584 |
| 下層太慢時改成重算（依裝置頻寬的交叉點） | 已做（單點） | CrossKV；LMCache 技術報告（D3 引 p.15）；py-kvcache（L3） |

### 3.2 缺口（這輪沒查到；不等於不存在）

| # | 缺口 | 和我們的關係 |
|:--|:--|:--|
| G1 | 把**執行期 GPU 忙碌**（或背景負載的型態）當成**還原格式**或**解壓位置**選擇的輸入 | 候選 2 的核心；L5c 的 B5、B6 結論不變 |
| G2 | 同一個動作集同時有 GEMM 型（跨層低秩、hidden state）和逐元素型（INT4）格式，並量它們在背景負載下的排序 | 我們 X3 端到端實測的 4 組是目前唯一的證據（results_analysis §4.9） |
| G3 | 背景是計算型（GEMM）還是記憶體型（decode）時，翻轉方向是否相同 | 我們的風險（A7）；也是把主張做尖的機會 |
| G4 | 負載快速變動時，「負載當輸入」和「殘差學習（KVServe 式）」差多少 | 對 KVServe 唯一硬的差異，**必須自己量** |
| G5 | AMD／ROCm 上任何「依現況選格式」的系統 | L5c 已記：只有固定格式（AMD ROCm 部落格、UltraQuant）與 hipCOMP 提案 |
| G6 | I3：「知道 session 結束」的價值在 SSD 和 NFS 下各是多少；NFS 層的 agent KV 規則 | 機制已有、量化沒查到；我們目前只有模擬 |
| G7 | I1：GPU 忙時 H2D 被拖慢的情況下，做聯合還原 | 我們自己量到效果小：互相拖慢 ≤6%、實測／模擬比 0.97–1.05（results_analysis §4.3），所以**不是值得追的缺口** |

### 3.3 對本專案既有文件的更正（只列，**沒改原檔**）

| 文件 | 原寫法 | 建議改成 |
|:--|:--|:--|
| `stage3_candidates.md` 候選 2「前人」欄；`results_analysis.md` 摘要第 3 點、§綜合「GPU 忙時」段（§3 文獻那句只講 CacheFlow、InferCept、FSL-26-02，對那一組是對的） | 「前作都假設互不干擾」；「前人都假設計算和搬運互不干擾」 | 「前作知道 GPU 上的解壓／I/O 會和模型互相拖慢（ShadowServe、KVCodec、Tutti），但用靜態隔離或誤差吸收處理；**沒有人把 GPU 忙碌當成還原格式選擇的輸入**」。L4 那句「讀過的論文都假設不干擾」只對 CacheFlow、InferCept、Stony Brook 報告、Cascade 那一組成立，不能推到壓縮傳輸那一組 |
| `L3` H5 Continuum | 作者 10 人，順序 H. Li, Q. Mang, R. He… | arXiv v7（2026-09-08）作者 11 人：Hanchen Li, Runyuan He, Qiuyang Mang, Qizheng Zhang, Huanzhi Mao, Xiaokun Chen, Hangrui Zhou, Huanchen Zhang, Alvin Cheung, Joseph Gonzalez, Ion Stoica。v7 有 §6.3 的 SSD 實驗，Alg. 1 有程式結束時釋放 |
| `L4` CacheFlow | （L4 讀的是 v2，沒寫標題） | v2（2026-09-26）abs 頁標題是 "CacheFlow: Efficient LLM Serving via Automated 3D-Parallel KV Cache Restoration"；v1 標題沒有 "Automated"（依搜尋結果的 HTML v1 標題，〔標題〕） |
| `L5c` 總表 KVCodec 列 | 「沒找到釋出連結」；深度引 L2 B5〔部分〕 | 已有 v3（2026-05-12），Comments "Accepted by SIGCOMM 2026"；fetcher 只依頻寬調解析度（§3.3.2）；干擾數字見本檔 A3。搜尋結果顯示 HTML v2 的標題是 "Efficient Remote Prefix Fetching with GPU-native Media ASICs"〔標題〕，不同版本換過標題 |

---

## 4. 建議的定位

### 4.1 主張的寫法（草稿）

> KV 從下層還原時，最佳的壓縮格式取決於兩件事的交互作用：**還原端解碼的計算型態**（GEMM 型重建、逐元素反量化、或只是搬位元組），和**此刻 GPU 被什麼、佔了多少**。
>
> - 跨層低秩加 INT4 在 GPU 閒、或鏈路慢時最省時間；但 GPU 被計算型工作佔用時，它反而比不壓縮還慢。
> - INT4 對同樣的背景幾乎不敏感。
>
> 所以還原格式應該在**讀回時**依現場狀況選，而不是寫入時依鏈路頻寬定一次，也不是靠事後的殘差慢慢學。

### 4.2 不要再講的話
- ✗「前人都假設計算和搬運互不干擾」→ 改成 §3.3 的說法。
- ✗「沒人把 GPU 忙碌放進決策」→ 加上限定「放進**還原格式**或**解壓位置**的選擇」。理由：Nexus、DuetServe 把爭用放進 SM 分配；Tutti 把 SM 預算放進 I/O 時機。
- ✗ 把「執行期自適應」當賣點 → KVServe、MorphServe、ElasticKV、MetaKV 都已經是執行期自適應。

### 4.3 每個實驗都要並列的對照（專案原則：同時對部署預設與對 oracle-best 報）
1. **部署預設**：vLLM OffloadingConnector 搬原始 BF16（L5c §0 第 6 點）。
2. **事後最好的固定格式**。
3. **只看鏈路的自適應（KVServe 式）**：context＝頻寬區間，EWMA 殘差＋ε-greedy，**用同一個動作集**（含低秩）。這是對最大威脅的正面比較。
4. **HCache 式靜態 profile**：在 GPU 閒時量一次，之後固定。
5. **Oracle**：每次還原都選對。

### 4.4 為了守住新意，要補的實驗（只是建議，沒有排程）

| # | 實驗 | 要回答的問題 | 對應的威脅或缺口 |
|:--|:--|:--|:--|
| E1 | 背景負載換型態：計算型 matmul（現有的 `m7_busy`）、記憶體型（真實的大 batch decode）、兩者混合 | 翻轉方向會不會隨背景型態改變？ | A7、G3 |
| E2 | 背景 duty 以 0.1／1／10 s 的週期切換；比較「負載當輸入」和「EWMA 殘差」的遺憾 | 比 KVServe 式控制器好多少？在哪種時間尺度才好？ | A1、G4 |
| E3 | 真實併發：多個請求同時 decode 時，量各格式的還原時間與 TTFT | 翻轉在真實服務裡會不會出現？（`stage3_candidates.md` 候選 2 原本的下一步） | G2 |
| E4 | 把 HCache 的 hidden state 加進動作集，作為「無損＋GEMM 型」那一格 | 無損模式下，有沒有同樣的翻轉？ | A2 |
| E5 | 儲存成本〔算術〕：讀回時才選，需要同時存兩種表示（像 CacheGen 的多版本，L5c §2.2），或在寫入時預測。用 X4 的倍數算：INT4（1/3.2≈0.313）＋低秩＋INT4（約 1/12≈0.083）≈ BF16 的 0.40，單存 INT4 是 0.31 | 多存約 0.08×BF16 的位元組值不值得？ | 設計限制 |

- **統計力**：目前「依情況選 +27.6%」只有 4 組（results_analysis §4.9），模擬又低估 5–26%。E1–E3 要事先寫死格子數與判定條件。
- **ε 的限制**：低秩＋INT4 的品質在 32K、約 12 倍時掉 1–3 個百分點（§4.11）。所以「允許有損」的翻轉，要和「只准無損」（E4）分開報。

### 4.5 I1 與 I3 的定位
- **I1**：維持收掉。
  - 寫論文時可以當負面結果：在 MI300X 上，聯合還原 ≈ 各自用 Cake（實測／模擬 0.97–1.05）。
  - 並列 CacheFlow 的正面結果（預設 1.25 GB/s 鏈路、batch ≤8，L4）和 ObjectCache 的正面結果（頻寬受限的遠端物件儲存、對照是平均分），說明差在**鏈路是不是瓶頸**。
  - 只有在「多 session 同時從頻寬受限的 NFS 讀」時才值得重開。
- **I3**：不當主題。
  - 可以變成一節「session 結束提示的價值，隨下層裝置而變」的量測。對話對象是 vLLM RFC #57103 和 llm-d #2584：他們有機制、沒有量化。
  - 但目前只有模擬，而且 MI300X 沒有容量壓力。要做，得在 HBM 被壓縮的設定下，用 GPU 驗證（i3_pause_sim §7 已有建議設定）。
  - 依專案原則（只靠模擬判死的方向只算暫定），現在的「收掉」是**暫定**的。

---

## 5. 參考文獻清單

「核」欄：
- ✓abs：這輪用 WebFetch 打開 arXiv abs 頁，核了標題、作者、版本日期、Comments。
- ✓官方：打開了會議或論文庫的官方頁。
- ✓issue：打開了 GitHub issue。
- ✓W：打開了部落格。
- L2／L4／L5c：前幾輪核過，本輪沒重開。
- NOT_VERIFIED：沒打開。

### 5.1 本輪打開過的

| # | 文獻 | URL | 核 | 深度 |
|:--|:--|:--|:--|:--|
| 1 | Z. Liu, X. Ma, D. Luo, H. Zhao, B. Lu, W. Huang, Y. Gu, X. Liu, Z. Wei, J. Liu, D. Tao, G. Tan, "KVServe: Service-Aware KV Cache Compression for Communication-Efficient Disaggregated LLM Serving," SIGCOMM 2026（v1 2026-05-13） | https://arxiv.org/abs/2605.13734 | ✓abs | 〔部分〕 |
| 2 | S. Gao, Y. Chen, J. Shu, "Fast State Restoration in LLM Serving with HCache," EuroSys 2025（v1 2024-10-07） | https://arxiv.org/abs/2410.05004 | ✓abs | 〔部分〕 |
| 3 | X. Xiang, R. Joshi, Y. Liu, J. Yao, C. Zhao, J. Jiang, Y. Zhou, E. Kohler, M. Yu, "ShadowServe: Interference-Free KV Cache Fetching for Distributed Prefix Caching," arXiv 2025（v1 2025-09-21；venue 沒查到） | https://arxiv.org/abs/2509.16857 | ✓abs | 〔摘要〕＋L5c〔全文〕 |
| 4 | L. Mi, W. Wang, J. Chen, T. Cao, H. Dai, Y. Liu, "Efficient Remote KV Cache Reuse with GPU-native Video Codec"（KVCodec）, SIGCOMM 2026（v3 2026-05-12） | https://arxiv.org/abs/2602.09725 | ✓abs | 〔部分〕 |
| 5 | S. Qiu, Y. Hu, X. Wang, W. Zhu, J. Yan, H. Chen, K. Xu, K. Chen, Y. Zhang, "Tutti: Making SSD-Backed KV Cache Practical for Long-Context LLM Serving," arXiv 2026（v1 2026-05-05） | https://arxiv.org/abs/2605.03375 | ✓abs | 〔部分〕 |
| 6 | Y. Guo, S. Joshi, "SplitZip: Ultra Fast Lossless KV Compression for Disaggregated LLM Serving," arXiv 2026 | https://arxiv.org/abs/2605.01708 | L2（本輪讀 HTML v1） | 〔部分〕 |
| 7 | S. Jin, X. Liu, Q. Zhang, Z. M. Mao, "Compute Or Load KV Cache? Why Not Both?"（Cake）, ICML 2025（v2 2025-02-20） | https://arxiv.org/abs/2410.03065 ；https://icml.cc/virtual/2025/poster/45020 | ✓abs ✓官方 | 〔摘要〕 |
| 8 | Z. Zhang, H. Shen, S. Vargaftik, R. Ben Basat, M. Mitzenmacher, M. Yu, "HACK: Homomorphic Acceleration via Compression of the Key-Value Cache for Disaggregated LLM Inference," arXiv 2025（v1 2025-02-05） | https://arxiv.org/abs/2502.03589 | ✓abs | 〔摘要〕＋L2 B6 |
| 9 | W. Han, G. M. Yeung, M. Barletta, W. Toner, A. Hoste, A. Barker, "Lynx: Progressive Speculative Quantization for accelerating KV Transfer in Long-Context Inference," arXiv 2026（v1 2026-07-02） | https://arxiv.org/abs/2607.01831 | ✓abs | 〔摘要〕 |
| 10 | J. Jiang, Y. Yang, P. Yang, F. Liu, "Does Execution Require Target KV Fidelity? A Mixed-Fidelity KV Runtime for LLM Serving"（ElasticKV）, arXiv 2026（v1 2026-09-27） | https://arxiv.org/abs/2609.33536 | ✓abs | 〔摘要〕 |
| 11 | Z. Su, Z. Zhang, T. Lan, Z. Wang, H. Shen, J. Yang, Y. Cheng, "MorphServe: Efficient and Workload-Aware LLM Serving via Runtime Quantized Layer Swapping and KV Cache Resizing," MLSys 2026（v2 2026-01-07） | https://arxiv.org/abs/2506.02006 ；https://mlsys.org/virtual/2026/oral/3816 | ✓abs ✓官方 | 〔摘要〕 |
| 12 | M. Wang, K. Li, R. Bostandoost, "MetaKV: Adaptive KV Cache Compression for Constrained LLM Inference," arXiv 2026（v1 2026-09-07；venue NOT_VERIFIED，NeurIPS 頁 404） | https://arxiv.org/abs/2609.07966 | ✓abs | 〔摘要〕 |
| 13 | P. Elvinger, F. Strati, N. Enright Jerger, A. Klimovic, "Understanding GPU Resource Interference One Level Deeper," arXiv（v3 2026-02-16） | https://arxiv.org/abs/2501.16909 | ✓abs | 〔摘要〕 |
| 14 | X. Shi, C. Cai, J. Du, Z. Jia, "Nexus: Proactive Intra-GPU Disaggregation of Prefill and Decode in LLM Serving," arXiv 2025（v5 2025-08-07） | https://arxiv.org/abs/2507.06608 | ✓abs | 〔摘要〕 |
| 15 | L. Gao, C. Jiang, H. Entezari Zarch, D. Wong, M. Hill, M. Annavaram, "DuetServe: Harmonizing Prefill and Decode for LLM Serving via Adaptive GPU Multiplexing," arXiv（v2 2026-05-30） | https://arxiv.org/abs/2511.04791 | ✓abs | 〔摘要〕 |
| 16 | P. Patel, E. Choukse, C. Zhang, A. Shah, Í. Goiri, S. Maleki, R. Bianchini, "Splitwise: Efficient generative LLM inference using phase splitting"（v2 2024-05-20；abs 的 Comments 沒寫 venue，常見引用為 ISCA'24 → 本輪 NOT_VERIFIED） | https://arxiv.org/abs/2311.18677 | ✓abs | 〔部分〕（HTML §IV-C、§VI-A、§VII） |
| 17 | Y. Zhong, S. Liu, J. Chen, J. Hu, Y. Zhu, X. Liu, X. Jin, H. Zhang, "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving," OSDI 2024 | https://arxiv.org/abs/2401.09670 | ✓abs | 〔摘要〕 |
| 18 | R. Qin, Z. Li, W. He, M. Zhang, Y. Wu, W. Zheng, X. Xu, "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving"（v5 2026-10-08；abs 的 Comments 沒寫 venue） | https://arxiv.org/abs/2407.00079 | ✓abs | 〔摘要〕 |
| 19 | D. Shi, J. Zhang, L. Lv, B. Huang, Y. Dai, Y. He, D. Chen, H. Fu, W. Feng, S. Fu, "Range: Decoupling KV-Cache Reuse Granularity from Storage Work," arXiv 2026（v1 2026-10-15） | https://arxiv.org/abs/2610.21241 | ✓abs | 〔摘要〕 |
| 20 | Y. Zhu, A. Dhakal, Y. Xiao, D. Milojicic, G. Alonso, "ObjectCache: Layerwise Object-Storage Retrieval for KV Cache Reuse," arXiv 2026（v1 2026-05-16） | https://arxiv.org/abs/2605.22850 | ✓abs | 〔部分〕 |
| 21 | S. Nian, H. Shen, Z. Wu, J. Fang, Q. Feng, F. Lai, "CacheFlow: Efficient LLM Serving via Automated 3D-Parallel KV Cache Restoration," arXiv（v2 2026-09-26） | https://arxiv.org/abs/2604.25080 | ✓abs | L4〔全文〕 |
| 22 | M. Adnan, R. Mahapatra, P. J. Nair, D. Berger, P. Zardoshti, R. Fonseca, E. Choukse, "Cascade: Exploiting SLO-Aware latency budget for fair and high goodput LLM inference serving," arXiv（v1 2026-08-06） | https://arxiv.org/abs/2608.06557 | ✓abs | L4〔全文〕 |
| 23 | Z. Xie, Z. Xu, M. Zhao, Y. An, V. S. Mailthody, S. Mahlke, M. Garland, C. Kozyrakis, "Strata: Hierarchical Context Caching for Long Context Language Model Serving," OSDI 2026 | https://www.usenix.org/conference/osdi26/presentation/xie-zhiqiang ；https://arxiv.org/abs/2508.18572 | ✓官方；arXiv 見 L2 | 〔摘要〕 |
| 24 | J. Wang, J. Hu, Q. Cao, Y. Zhu, X. Lin 等（頁面上的作者列表被截斷）, "Multi-tier dynamic storage of KV cache for LLM inference under resource-constrained conditions"（MTDS）, Complex & Intelligent Systems 12, 104（2026-01-27 上線） | https://link.springer.com/article/10.1007/s40747-025-02200-4 | ✓官方 | 〔部分〕 |
| 25 | M. Kang, C. Shin, J. Jeong, J. Park, W. Kim, B. Gu, D. Kang, G. Yang, C. Yoo, "Unified KV Pooling to Accelerate Long-Context LLM Serving," arXiv 2026（v1 2026-06-10） | https://arxiv.org/abs/2606.14779 | ✓abs | 〔摘要〕 |
| 26 | H. Li, R. He, Q. Mang, Q. Zhang, H. Mao, X. Chen, H. Zhou, H. Zhang, A. Cheung, J. Gonzalez, I. Stoica, "Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live," arXiv（v7 2026-09-08） | https://arxiv.org/abs/2511.02230 | ✓abs | 〔部分〕（HTML v7） |
| 27 | vLLM, "[RFC]: Programmable KV Cache: Composable Policies for Agentic Serving" #57103（2026-09-16 開，open） | https://github.com/vllm-project/vllm/issues/57103 | ✓issue | 〔部分〕 |
| 28 | llm-d, "[Feature]: Agent-Aware KV Cache Affinity Scheduling and Cache Management" #2584（2026-09-25 開，open） | https://github.com/llm-d/llm-d/issues/2584 | ✓issue | 〔部分〕 |
| 29 | Y. Liu, Y. Zhang, F. Li, H. Zhang, "Ask the Tool, Don't Guess: Agent Tool Calls Hold Their Progress, and the Serving System Should Read It," arXiv 2026（v1 2026-09-16） | https://arxiv.org/abs/2609.18849 | ✓abs | 〔摘要〕 |
| 30 | 배재형（Bae Jae-hyeong）, "CrossKV: A Bandwidth-Crossover-Aware Fallback Policy for Hierarchical KV Caches in LLM Serving," 西江大學碩士論文, 2026-08 | https://dcollection.sogang.ac.kr/dcollection/srch/srchDetail/000000083477 | ✓官方 | 〔摘要〕 |
| 31 | W. He, P. Zhang, X. Hu, C. Wang, J. Hou, Y. Cheng, P. Sun, F. Fu, "Efficient Agentic LLM Serving over SSD-based Sparse KV Storage"（Janus）, arXiv 2026（v1 2026-09-29） | https://arxiv.org/abs/2609.36938 | ✓abs | 〔摘要〕 |
| 32 | F. He, Y. Li, X. Zeng, "UNISON: A Co-Designed Near-Memory Scheduler of Session KV Residency for LLM Agents," arXiv 2026（v1 2026-09-09） | https://arxiv.org/abs/2609.09643 | ✓abs | 〔摘要〕 |
| 33 | Y. Wu, S. Chen, Y. Zhong, R. Huang, Y. Tan, W. Zhang, L. Zhang, S. Zhou, Y. Liu, S. Zhou, M. Zhang, X. Jin, P. Huang, "DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference," arXiv 2026（v2 2026-02-26） | https://arxiv.org/abs/2602.21548 | ✓abs | 〔摘要〕 |
| 34 | D. Jang, I.-N. Wang, J. Jeong, "Tool Waiting and Re-arrival in Compile-Time-Static LLM Serving: Cost Mechanisms and Configuration Selection," arXiv 2026（v1 2026-09-28） | https://arxiv.org/abs/2609.34663 | ✓abs | 〔摘要〕 |
| 35 | Y. P. Shkolnikov, "Agent Memory Below the Prompt: Persistent Q4 KV Cache for Multi-Agent LLM Inference on Edge Devices," arXiv 2026（v1 2026-02-17） | https://arxiv.org/abs/2603.04428 | ✓abs | 〔摘要〕 |
| 36 | D. Guo, J. Wu, S. M. Yiu, "SAGA: Workflow-Atomic Scheduling for AI Agent Inference on GPU Clusters," arXiv 2026（v2 2026-06-19） | https://arxiv.org/abs/2605.00528 | ✓abs | 〔摘要〕 |
| 37 | R. Zhang, C. Kim, L. Hu, "A Policy-Driven Runtime Layer for Agentic LLM Serving," arXiv 2026（v2 2026-07-30） | https://arxiv.org/abs/2605.27744 | ✓abs | 〔摘要〕 |
| 38 | A. Luo and the LMCache Team, "Benchmarking LMCache for Multi-Turn Agentic Workloads on AMD MI300X," LMCache Blog, 2026-05-12 | https://blog.lmcache.ai/en/2026/05/12/benchmarking-lmcache-for-multi-turn-agentic-workloads-on-amd-mi300x/ | ✓W | 〔部分〕 |
| 39 | Y. Liu, Y. Cheng, J. Yao, Y. An, X. Chen, S. Feng, Y. Huang, S. Shen, R. Zhang, K. Du, J. Jiang, "LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference," arXiv（v2 2025-12-05） | https://arxiv.org/abs/2510.09665 | ✓abs | 〔摘要〕 |
| 40 | S. Feng, H. Li *et al.*, "AdaptCache," BigMem@SOSP'25（v2 2026-01-15）；S. Feng, Y. Liu *et al.*, "EVICPRESS," arXiv（v1 2025-12-16，venue 沒查到） | https://arxiv.org/abs/2509.00105 ；https://arxiv.org/abs/2512.14946 | ✓abs | 〔摘要〕＋L5c〔全文〕 |

### 5.2 只引用前幾輪、本輪沒重開的
- CacheGen（L5c [1]）；LMCache RFC #4498、MP serde、TRT-LLM cold-page NVFP4（L5c [10]、[11]、[15]）。
- InferCept、Tokencake v5、Stony Brook FSL-26-02、EfficientAgent（L4）。
- KVFlow、Parrot、Autellix、MORI、Lachesis、Bidaw、py-kvcache、CacheScout（L3）。
- CrossKV 的前一次記錄（`research_20261010_directions/D3_kappa_map.md` §5）；LMCache 技術報告 p.15 的交叉點（同上，D3 引用）。

### 5.3 查過、判定無關（本輪〔摘要〕核過）
- KVFetch（arXiv 2610.08811，L. Dong，v1 2026-09-23，Comments "Submitted to ICLR 2027"）：冷層量化加循序預取，屬選擇性還原那一線（候選 4）。
- Approximation Algorithms for Dual-Memory Request Scheduling（arXiv 2610.15579，v1 2026-10-11）：PD 兩池的記憶體佔用排程，不涉及還原或爭用。
- Service-Induced Congestion in Memory-Constrained LLM Serving（arXiv 2606.15555，v3 2026-10-20）：連續批次的逐出動力學，不涉及下層。
- SMetric（arXiv 2607.08565，v2 2026-09-13）：以 session 為單位的路由，不涉及還原。
- Unified AI Gateway（arXiv 2609.06940，v1 2026-09-07）：模型路由加 KV 動作的聯合選擇，屬於分析模擬，不涉及 GPU 爭用。

### 5.4 只看到標題、沒打開（NOT_VERIFIED）
- FastCache（arXiv 2503.08461）、PackKV（2512.24449，L2 B8 有摘要）、BitDecoding（2503.18773）、VecInfer（2510.06175）。
- APEX（2506.03296）、HybridGen、TraCT（2512.18194）、Swarm（2603.17803）。
- "Bridging LLM Serving and CXL-SSDs with Chunk-Aware KV Cache Management"（2609.26828）、"The Serialized Bridge"（2606.23969）。
- RepairKV（"Cache You Later"，ICML 2026 頁出現在搜尋結果）、STAR-KV（ICML 2026）、AnchorKV（2608.02901）。
- 部落格 zolotukhin.ai（RDNA4 上 swap 對重算 177 倍）。

### 5.5 查了沒找到
- 把執行期 GPU 忙碌當成 KV 還原格式或解壓位置選擇輸入的論文、系統或上游 PR。
- 依背景負載**型態**（計算型或記憶體型）比較 KV 解碼 kernel 敏感度的量測。
- MI300X／ROCm 上量「KV 還原在背景負載下變慢多少」的公開數據（只找到 LMCache 部落格的端到端 TTFT）。
- NFS／共享檔案系統層上的 agent KV 去留規則。
- 量化「session 結束提示」價值的實驗。
- 2026-10-20 之後新上架、和本題直接相關的論文（arXiv 搜尋列表最新只看到 10/15 的 Range 與 2610.21249；索引可能有延遲）。
