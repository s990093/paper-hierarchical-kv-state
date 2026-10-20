# L5c　查新：卸載時順便壓縮，依連線速度與 GPU 忙碌動態選格式（方向 B）

> **日期**：2026-10-20（UTC）
> **執行者**：文獻 agent。只讀檔、只上網；**沒有開任何 GPU 行程，沒有 import torch／vllm**。PDF 用 tiara venv 的 PyMuPDF（只 import `pymupdf`）抽字，指令一律 `nice -n 19 taskset -c 0-3`。
> **新下載的 PDF**：`/mlsteam/data/tiara/papers/stage3_L5/shadowserve_arXiv2509.16857.pdf`（其他論文用 `/mlsteam/data/tiara/papers/` 既有的檔）。
> **先讀過的本專案檔案**：`research_map.md`、`experiment_matrix.md`、`stage2_factcheck_and_toolbox.md`、`lit/L0`、`lit/L2`（特別是 B1–B10、§1.5、G1、G4）、`lit/L4`、`docs/precision_20261013/08_report_mi300x.md` §2.5／§2.8、`09_sweep_plan.md` §6.1、`11_issues.md` §3a／§3d、`cards/P02_CacheGen_mechanism.md`、`results/RUNLOG_MI300X.md` 14.1、`results/m10_restore2_mi300x/tables_13.md`、`results/m16_stage3_mi300x/i3_pause_sim/costs_sources.csv`。**L2 已寫過的卡片只引用、不重寫**。
> **本檔是 self-review，不是 cross-model review**（CLAUDE.md §5）。

**證據等級**
- 〔全文〕＝整篇 PDF 抽字後逐段讀過，附 PDF 實體頁碼（不是論文印的頁碼）；只讀文字，**圖上的數值沒讀**。
- 〔部分〕＝只讀了指定的節或頁；或是用 WebFetch 讀網頁（由工具抽取，不是逐字核對）。
- 〔摘要〕＝只讀 arXiv abs 頁（用 curl 實際開過，§7 的「核」欄）。
- 〔README〕〔程式碼〕〔文件〕＝讀 GitHub README、原始碼、官方文件。
- 〔判讀〕＝我對原文的解讀；〔推論〕＝我的推測；〔算術〕＝我用公開或本專案的數字算的，公式寫在旁邊。
- **NOT_FOUND**＝這輪查了沒找到（≠不存在，CLAUDE.md 規則 7）；**NOT_MEASURED**＝本專案還沒量。
- 不同硬體、batch、context 的數字**只並列、不排名**。

---

## 0. 一頁結論

1. **B 還有新意，但只剩一小塊。** 「依連線速度或儲存層級選壓縮格式與位元數」已經有人做了：
   - CacheGen 在讀回時逐 chunk 依量到的吞吐量換等級，來不及就改送文字重算〔全文，SIGCOMM'24 §5.3 p.6–7〕。
   - KVServe 依有效頻寬分段挑最佳設定，有損益兩平公式〔全文，SIGCOMM'26 Thm 6.1–6.2 p.8–9〕。
   - AdaptCache／EvicPress 依 DRAM／SSD 層級、命中頻率、品質敏感度，選演算法、壓縮率與放哪一層〔全文〕。
   - 已上線的實作：LMCache MP 的 serde 可以**每個 L2 adapter 各自**設 fp8／TurboQuant（靜態）〔文件〕；TensorRT-LLM 的 cold-page NVFP4 在卸載時才壓〔文件〕。

   **沒找到**任何一篇把「GPU 忙不忙」當成決策輸入：
   - KVServe 承認「GPU load and queue contention change the actual (de)compression throughput」，但只把它當成漂移，交給 EWMA 殘差修正（p.8–9）。
   - ShadowServe、KVCodec 是把解壓**整個**搬離 GPU（SmartNIC、NVDEC），屬於靜態放置。
   - LMCache RFC #4498 是靜態設計：預設用 CPU 壓縮，避開 GPU 爭用。

   所以 B 剩下的新意只有四塊：
   - (i) 執行期的 GPU 忙碌（加上 CPU 空閒）當成輸入。
   - (ii) 最自然的形態是**讀回時決定在哪裡解**（CPU 多執行緒、GPU 一次拆、或根本不壓），而不是寫入時選格式。
   - (iii) 本機的 NFS 層。
   - (iv) AMD／ROCm。
2. **四個指定問題的答案**（細節見 §3）：
   - **(a) GPU 忙碌**：沒有人放進決策（NOT_FOUND）。干擾本身已有很多實測（ShadowServe、KVCodec、HACK，以及我們自己的第二階段）。
   - **(b) 本機分層**：部分做過。AdaptCache 做 DRAM＋SSD（磁碟 1 GB/s），EvicPress 做 CPU／本地 SSD／遠端磁碟。**NFS 沒找到**。EvicPress 自己把「頻寬變動」列為限制與未來工作（p.12）。
   - **(c) AMD／ROCm**：只有固定格式。AMD ROCm 部落格在 MI355X 上做 4-bit＋LMCache CPU 層；UltraQuant 是 EMNLP'26 Industry。另有一個提案：RFC #4498 的 hipCOMP backend。自適應版 NOT_FOUND。
   - **(d) 低秩、token 合併**：都沒有被放進任何卸載時的格式決策。
     - token 丟棄有被放進去：AdaptCache、EvicPress 的動作集就是丟棄法。
     - 低秩只以固定形式出現：KVTC 的 PCA、ShadowKV 的低秩 K。
     - token 合併只在 EvicPress 的分類裡被提到，沒有用（p.3）。
3. **我們自己第二階段的實測，已經把 B 的大半空間壓掉了**〔實測＋推論〕。
   - **CPU 層**（32K、全部載入、融合後一次拆 P3）：INT4 KIVI 在閒與忙都最快。

     | 模型 | BF16（閒／忙） | FP8（閒／忙） | INT4（閒／忙） |
     |:--|:--|:--|:--|
     | Llama | 84／124 ms | 46／49 ms | 33／36 ms |
     | Qwen | 39／55 ms | 21／22 ms | 18／19 ms |

   - **限速層**：INT4 快 36–57%。
   - **品質**：ε 24 格全過。
   - **一次拆幾乎不受背景影響**：Llama INT4 在 duty 0.4／0.7／0.9 是 35.2／36.1／36.7 ms。

   → **預測：允許有損時，決策圖整張都是 INT4，自適應對「事後最好的固定格式」約等於 0**。這一半很可能直接 NO-GO（§4）。
4. **剩下的機會在「只准無損」模式。** Tiara 本來就把 CPU／SSD 兩階定義成無損搬移（CLAUDE.md §0），VeriCache 也主張有損輸出會隨生成長度發散〔摘要〕。
   - **省多少的上界**〔算術，假設解壓免費而且完全重疊、壓縮比 r＝1.25–1.33〕：對不壓縮，NFS 級省 6–16%、SSD 級省 17–24%；CPU 層多半沒好處。
   - **在哪裡解有取捨**〔算術〕：
     - CPU 解：用 ShadowServe 量的單核速度推算，要跟上 SSD 級速度，LZ4 約要 5 個核、Deflate 要超過 32 個核。
     - GPU 解：會和忙碌的 GPU 搶資源。
   - **我們 KV 真正的 r：NOT_MEASURED**（S1 的 KV 匯出還沒跑）。
5. **最小實驗**（§5）分四步，GPU 合計〔估〕1.5–2.5 小時、加緩衝 ≤3 小時：

   | 步驟 | 做什麼 | GPU |
   |:--|:--|:--|
   | Step 0 | 用已量的常數先畫預測圖 | 0 |
   | Step 1 | KV 匯出＋CPU 端 codec 量測 | 〔估〕0.2 h |
   | Step 2 | 各格式在 duty 0／0.4／0.7／0.9 下的 GPU 解壓微測 | 〔估〕0.5–1 h |
   | Step 3 | 4–8 個格子的端到端驗證 | 〔估〕0.5–1 h |

6. **每一步都要同時報三種比較**：
   - 對**部署預設**：不壓縮，等於 vLLM OffloadingConnector 搬原始位元組〔程式碼〕。
   - 對**事後最好的固定格式**。
   - 對**只看連線速度的分層靜態**：等於前作已有的能力。
   
   **事先寫死的停止條件**（§5.5）：
   - 允許有損時每格最佳格式都一樣 → 有損那半停。
   - 我們 KV 的 BF16 無損比 <1.15 → 無損那半停。
   - 「連線＋GPU 忙」對「只看連線」在每一格都好不到 5% → GPU 忙這個軸沒價值，B 退回前作，停。
   - 對事後最好的固定格式（幾何平均）好不到 5% → NO-GO。
   - 5–15% → 停下來問使用者。
7. **對既有文件的更正**（§6）：
   - L2 說「ROCm 上沒有官方的 nvCOMP 對應品」要更新：ROCm 組織下有 **hipCOMP-core v2.3.0**。它是 early-access，README 列出在 gfx942 上測試過；LZ4／Snappy／Cascaded／RLE 能壓也能解，ZSTD／Deflate 只能解（實驗性）〔README〕。
   - LMCache MP 現在有 per-adapter 的 fp8／TurboQuant／AES-GCM serde〔文件〕。

---

## 1. 先把 B 拆開：一個「卸載時壓縮」的決策有哪些面向

| 面向 | 選項 | 備註 |
|:--|:--|:--|
| **輸入** | 連線速度 B（CPU pinned、本地 SSD、NFS；或網路）、**GPU 忙碌 u**、CPU 空閒、品質預算、容量壓力、命中頻率 | 前作用的是 B、容量、頻率、品質；**u 沒人用**（§3） |
| **動作：格式** | BF16 原樣；無損（位元組分組＋zstd／zlib、指數碼本）；FP8／INT8／INT4 KIVI；有損＋無損；低秩；丟 token；合併 | 第二階段已驗證品質的：FP8（只有 Llama）、INT8、INT4 KIVI（08 §2.3–2.5、15.9） |
| **動作：在哪裡解** | GPU 逐塊、GPU 一次拆（P3）、CPU 多執行緒、其他硬體（SmartNIC、NVDEC） | 我們 MI300X 沒有 SmartNIC；VCN 的無損模式 NOT_VERIFIED（L2 B5） |
| **時機** | 寫入（卸載）時選格式；讀回時選要讀哪個版本或在哪解 | 寫入時不知道讀回時 GPU 多忙 → 要嘛預測，要嘛把決策搬到讀回時 |
| **粒度** | 每段對話、每個 chunk／block、每層 | CacheGen 每 chunk；KVServe 每請求；EvicPress 每段 context |

**和 Tiara 定義的關係**〔判讀〕：CLAUDE.md §0 把 CPU、SSD 兩階定義為「無損的位元組搬移」，有損精度是另外的動作（GPU-FP8／INT4）。所以照 Tiara 的定義，B 可以分成兩個模式：
- **模式 L（只准無損）**：在 CPU／SSD／NFS 這幾層選無損 codec 與解壓位置。這才是 B 在 Tiara 動作空間裡的位置。
- **模式 Q（允許有損）**：把第二階段驗證過的低精度也放進來。這其實是把 GPU 上的精度動作延伸到卸載層。

---

## 2. 前作：誰做過「依頻寬／裝置動態選壓縮程度」

### 2.1 總表

欄位：決策輸入／動作／時機／是否考慮 GPU 計算爭用／硬體／連線／程式碼。「GPU 爭用」一欄的「量」＝有量干擾，「決策」＝把它當決策輸入。

| 作品（venue） | 決策輸入 | 動作 | 時機 | GPU 爭用 | 硬體 | 連線 | 程式碼 | 深度 |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| **CacheGen**（SIGCOMM'24） | 上一個 chunk 量到的吞吐量、TTFT SLO、估計的重算時間 | 每 chunk 選一個預先編好的量化等級（＋算術編碼），或送文字重算 | **讀回時**（寫入時先編好多個等級） | 決策：**否**（只有重算時間；附錄 D 的熱圖把「GPU 可用週期」當評測軸）；量：自稱解壓開銷很小 | NVIDIA A40 | 網路 0.1–400 Gbps | UChi-JCL/CacheGen；**現行 LMCache 只剩一個固定 codec**（P02 §33、§36） | 〔全文〕 |
| **AdaptCache**（BigMem@SOSP'25） | 每筆的品質—壓縮率曲線（離線）、裝置頻寬與解壓開銷（離線，用假題量）、歷史命中頻率 | 演算法（token 丟棄 vs 量化）、壓縮率、放 DRAM 或 SSD、或淘汰 | **寫入時**：新的一筆進來時，連同已存的重算邊際效用 | 決策：**否**；解壓開銷只在離線量一次 | 1×A100、100 GB DRAM、400 GB SSD | 本機 DRAM／SSD（磁碟讀 1 GB/s） | NOT_FOUND（論文沒附） | 〔全文〕 |
| **EvicPress**（arXiv 2512.14946，venue NOT_FOUND） | 每段 context 的品質敏感度（離線出題）、各層的取回延遲、存取頻率、α | 丟 token 的方法（keydiff／knorm／snapkv）× 比例 × 放哪一層 | **寫入時**＋某層滿時＋品質漂移時重新量 | 決策：**否**；重新量只在 GPU 有空時做，會拖慢正在跑的請求（p.10） | 1×H100、80 GB DRAM、800 GB SSD、遠端磁碟 | 本機 CPU／SSD＋遠端磁碟 | NOT_FOUND | 〔全文〕 |
| **KVServe**（SIGCOMM'26） | 工作負載類別 w、有效頻寬 B、SLO、品質下限 | 從離線 3D Pareto 集合選 Transform×Quantizer×Codec（nvCOMP）組合 | **每個請求在 KV 開始搬時選一次** | 決策：**否**；GPU 負載只當漂移，用 EWMA 殘差吸收（p.8–9） | A100（離線量）、H100（decode）、4090／5090／Pro 6000／H100（prefill） | PD 傳輸、遠端 KV pool，5–100 Gbps | hpdps-group/KVServe | 〔全文〕 |
| **ShadowServe**（arXiv 2509.16857，venue NOT_FOUND） | —（固定格式） | 固定：分箱量化＋Deflate；**解壓放 SmartNIC** | — | **量**：L40S 上雙向干擾 ≥25–30%（p.3）；處理方式是靜態搬離 GPU | L40S＋BlueField-3 | 遠端前綴快取 10–40 Gbps | NOT_FOUND（論文沒附） | 〔全文〕 |
| KVCodec（SIGCOMM'26） | — | 量化＋H.265 無損，NVDEC 解碼 | 寫入時編、讀回時解 | **量**：GPU 解壓讓 prefill 慢約 50%、decode 慢約 20%；處理方式是改用專用解碼器 | A100／H20／L20 | 1–40 Gbps | 沒找到釋出連結 | L2 B5〔部分〕 |
| HACK（arXiv 2502.03589） | — | 2-bit KV，不反量化直接算 | — | **量**：反量化佔 JCT 17.2–37.9% | A10G、V100、T4、L4、A100 | PD 10–400 Gbps | 匿名 repo | L2 B6〔部分〕 |
| KVTC（ICLR'26） | 離線校準 | PCA＋DP 分配位元＋DEFLATE（**低秩類**，固定壓縮比） | 卸載前 | 否 | H100 | GPU 內與卸載儲存 | 第三方 NOT_VERIFIED | L2 B3〔部分〕 |
| SplitZip（NeurIPS'26） | — | 無損 BF16 指數碼本 | 傳輸時 | 否（只量 GPU 閒時的吞吐） | H200 等 | PD RDMA | Intelligent-Microsystems-Lab/SplitZip（README：需要 CUDA GPU） | L2 B1〔部分〕 |
| HCache（EuroSys'25） | **離線量的**計算與 I/O 速度 | 每層存 hidden state 或 KV（或重算）的分配 | 部署時定一次 | 否（靜態硬體 profile） | A100、H800 | SSD、DRAM | — | 〔部分〕§4.1 p.6–7、§6.2 p.11 |
| FlexGen（ICML'23） | 手動開關 | 權重與 KV 4-bit 分組量化 | 卸載時 | 有一句：開量化時把 CPU 代算關掉，因為 CPU 上的壓縮／解壓太慢（p.7） | T4 | CPU／磁碟 | 有 | 〔部分〕§5 p.6–7 |
| TensorRT-LLM cold-page NVFP4 | 靜態設定 | GPU 上維持原 dtype，**搬到 Host／Disk 時才編成 NVFP4** | 卸載時編、載回時解 | 文件沒提；也沒說在 GPU 還是 CPU 編解碼 | SM100／SM103 | Host、Disk | TRT-LLM | 〔文件〕 |
| **LMCache MP serde** | 靜態設定，**每個 L2 adapter 各自設** | fp8（e4m3／e5m2）、TurboQuant 四種 preset、AES-GCM；「one `--l2-adapter` may use fp8 while another stores raw bytes」 | 存 L2 前壓、讀回後解 | 文件沒提 | — | L2（fs、nixl 等） | LMCache | 〔文件，WebFetch〕 |
| **LMCache RFC #4498**（2026-08-12 開，open） | 靜態設定 | 無損壓縮存 L2；讀回時用 nvCOMP／**hipCOMP** 在 GPU 上解 | 存時壓（預設**CPU 非同步**）、讀回時 GPU 解 | 設計上明寫：CPU 壓「avoids GPU compute contention」；GPU 解「is beneficial only when the saved transfer time exceeds those costs and any interference with model inference」——**是陳述，不是決策** | NVIDIA＋AMD | 本地 NVMe、遠端 L2 | 提案；作者 10/04 說要拆 7 個 PR | 〔全文 RFC＋留言〕 |
| LMCache RFC #4149／PR #4264（Intel QAT） | 靜態 | DEFLATE 無損（可選砍尾數） | 存時 | — | Intel QAT／IAA | — | 兩者 2026-10-20 仍 open | L2 B10〔全文〕 |
| Intel KV Shrink | 環境變數 | DEFLATE；只壓指數位元組；前幾層可略過壓縮 | 存時；**QAT、IAA、CPU 三個後端誰先做完誰接**（壓縮引擎的動態分派） | 否 | Intel CPU＋NVIDIA | — | intel/intel-accel-for-llm（沒有 HIP 後端） | L0 §1.2〔全文〕 |
| AMD ROCm 部落格〈4-bit KV Caching in LMCache〉（2026-08-28） | 固定 | TurboQuant 4-bit（前後各 2 層保留 bf16），**HBM 內就是 4-bit**，溢出時照原格式搬到 CPU DRAM | — | 文中寫搬運「fully overlapped with compute」，沒分析干擾 | **2×MI355X** | CPU DRAM（L2） | rocm/vllm-dev 映像、fork commit | 〔部分，WebFetch〕 |
| Metrum AI 部落格（2026-09） | 固定 preset | LMCache 0.5.1 MP 的 TurboQuant serde，**只在 SSD 邊界壓**（HBM 維持 BF16） | 存 SSD 時 | 觀察到 decode 慢 9%，作者說「likely」是卸載時的 GPU 爭用，但**沒量** | 4×RTX PRO 6000 Blackwell、Solidigm SSD | 本地 SSD | — | 〔部分，WebFetch〕 |
| KVDrive（arXiv 2605.18071） | — | 未來工作：熱 block 留 FP16、冷 block 進 SSD 時用 INT4（「Tiered Mixed-Precision Storage」） | — | — | — | HBM／DRAM／SSD | — | 〔部分〕p.23 |
| Lynx（arXiv 2607.01831） | — | 高位元先送（Anchor）、殘差後送；先用高位元推測 decode，再驗證 | 讀回時漸進 | 摘要沒提 | — | PD 網路 | — | 〔摘要〕 |
| SpectrumKV（arXiv 2606.08635） | token 重要度＋部署時的模型探測 | 每個 token FP16／INT8／INT4 | 傳輸前 | 摘要沒提 | — | PD 網路 | — | 〔摘要〕 |
| vLLM 0.28 OffloadingConnector | — | **不壓縮**（`compress_ratio` 是 MLA attention spec 的屬性，不是卸載壓縮） | — | — | — | CPU、檔案層 | `/mlsteam/workspace/src/vllm-v0.28.0` | 〔程式碼〕 |

**沒有壓縮、但同樣在做分層決策的**（只列出備查）：
- Kareto（arXiv 2603.08739）：儲存配置的 Pareto 搜尋〔摘要〕。
- Predictive Multi-Tier（arXiv 2604.26968）〔摘要〕。
- LMCache SSD 特性量測（arXiv 2610.21249）〔摘要〕。
- Where Should the KV Cache Live?（arXiv 2609.16215）：只在相關工作提到量化，見 p.2〔部分〕。
- Tutti、Strata、Bidaw、MTDS：抽字後搜 compress／quantiz，只出現在相關工作裡〔部分〕。

### 2.2 全文卡片（最接近 B 的五篇）

#### CacheGen（Y. Liu *et al.*, SIGCOMM'24；arXiv 2310.07240）〔全文；機制與程式碼落差另見 P02〕
- **決策輸入**：
  - 用上一個 chunk 量到的吞吐量估頻寬，假設之後都不變。
  - 各設定的預期延遲＝大小÷吞吐量（§5.3 p.7）。
  - 多個請求同時來時：預期延遲＝含 chunk c 的請求數 N_c × 單一請求延遲（p.7）。
  - Alg. 1 用「剩餘 SLO 時間」與 `time_recompute` 比較（App. C.1 p.19）；P02 指出 `time_recompute` 沒有定義，程式碼裡是常數。
- **動作**：
  - 每個 chunk（預設 1.5K token）事先編成多個量化等級（離線）。
  - 讀回時，每個 chunk 選「SLO 內損失最小」的等級，或改送文字讓 LLM 重算（§5.3 p.6–7）。
- **GPU 爭用**：
  - 解壓用 CUDA 寫，和傳輸重疊；自稱「decoding has minimal impact on the end-to-end delay」（§7.5 p.11）。
  - 唯一和 GPU 負載有關的是附錄 D 的 Fig. 19（p.19）：「GPU available cycles（1/n，n＝並行請求數）× 頻寬」下對最佳基線的改善熱圖。〔判讀〕那是**評測軸**，GPU 負載只透過「重算有多貴」進來，不是決策輸入，也沒有模型化解壓被拖慢。
- **硬體／連線**：NVIDIA A40（§7.1 p.9）；頻寬掃 0.4–15、15–400 Gbps（§7.3 p.10）；適應實驗每 chunk 的頻寬從 0.1–10 Gbps 抽（§7.4 p.10–11）。
- **自己寫的限制**：高階 GPU＋低頻寬時可能不比送文字好；未來工作是像 SVC 一樣先送低品質再補差值（§9 p.12）。
- **和 B 的差距**〔判讀〕：
  - 讀回時、依網路頻寬選等級，B 的「依連線速度」那半它做過了。
  - 沒有 GPU 忙碌、沒有本機分層、沒有無損 vs 有損的選擇（算術編碼總是開著）。
  - 多版本儲存的代價（附錄 E：8.5K token、Llama-13B 約 5 GB，p.19）。

#### AdaptCache（S. Feng, H. Li *et al.*, BigMem@SOSP'25；arXiv 2509.00105v2）〔全文，3 頁〕
- **venue**：arXiv Comments「Accepted at SOSP 2025 - The International Workshop on Big Memory (BigMem)」。正文 3 頁，是 EvicPress 的前身（同一團隊、同樣的效用函數形式）。
- **問題**：DRAM 放不下，大部分命中來自慢的 SSD。用有損壓縮讓 DRAM 多放，以提高 DRAM 命中（p.1）。
- **決策**：
  - Utility(i)＝Freq(i)·(α×Quality(i, M_i, R_i) − size(i, M_i, R_i)/Bandwidth)（p.2）。
  - 多選擇背包問題，用邊際效用降幅做貪婪：每一層各自決定「淘汰／全精度／壓縮」（p.2）。
  - 輸入：
    - 估計器**離線**用假題量「各裝置的傳輸延遲與解壓開銷」。
    - 抽 10 筆、用 GPT-4o 出題，畫品質—壓縮率曲線。
    - 命中頻率用歷史頻率估（p.2）。
- **動作**：演算法（token 丟棄 vs 量化）、壓縮率、裝置（DRAM／SSD）。動機說「有的文字適合丟 token、有的適合量化」（p.1）。
- **時機**：新的一筆產生時，對新的和已存的一起算邊際效用（p.2）→ 寫入時。
- **評測**：
  - 1,100 段 LongBench context（六個資料集）、Poisson 到達、Llama-3.1-8B。
  - 1×A100、100 GB DRAM、400 GB SSD，**磁碟讀 1 GB/s**（p.2）。
  - 對照：不壓縮、KIVI＋LRU、StreamingLLM＋LRU、prefill（p.2）。
  - 結果：同品質下快 1.43–2.4×，同延遲下品質高 6–55%（p.1–2）。
  - DRAM 命中：KIVI 2-bit 38% vs AdaptCache 81／56／44／11%（隨 α，p.2–3）。
- **GPU 爭用**：沒有；解壓開銷是離線常數。
- **程式碼**：論文沒附（NOT_FOUND）。
- **和 B 的差距**〔判讀〕：「依層級選壓縮」它做了。但頻寬是靜態常數、沒有 GPU 忙碌、沒有無損、沒有 NFS。它的目標是**容量**（讓 DRAM 多放），不是**還原速度**。

#### EvicPress（S. Feng, Y. Liu *et al.*；arXiv 2512.14946v1，2025-12-16）〔全文，16 頁〕
- **venue**：arXiv 沒有 Comments → NOT_FOUND。**名字不一致**：Fig. 1（p.1）與結論（p.12）寫成「CacheServe」。
- **決策**：
  - Util(method, ratio, device)＝(α·quality − TTFT)·frequency（§4.2 p.6）。
  - 以「整段 context」為單位，理由是 chunk 或 token 層級看不到全域重要度，而且讀不滿頻寬（p.6）。
  - 某一層滿時，用貪婪的多選擇背包，挑效用降幅最小的「再壓一點」或「降到下一層」，遞迴往下（§4.4 p.7、App. A Alg. 1 p.16）。
- **輸入**：
  - 品質：每段 context 用 GPT-5 出 100 題，50 題拿來 profiling（p.6、p.8）。
  - 延遲：從該層取回（壓縮後）KV 的估計延遲（p.6）。
  - 存取頻率。
  - 線上品質比 profiling 低 X% **而且 GPU 有空**時，重新 profiling（p.5、p.7）。
- **動作**：
  - 實際評測只用三種**丟 token** 的方法（keydiff、knorm、snapkv）× 比例 × 層級。
  - 層級：CPU DRAM、本地 SSD、遠端磁碟（p.5–6 Fig. 6；p.8）。
  - 量化「can be easily extended」但沒做（p.12）。
- **時機**：存入時（存在 CPU 上做，不擋推論，p.5）；層滿時；週期性重新 profiling。
- **評測**：
  - 實作：vLLM v0.11.2＋LMCache v0.3.9post2，約 3K 行（p.7）。
  - 硬體：1×H100、80 GB DRAM、800 GB SSD、遠端假設無限（p.8）。
  - 5 個模型（含 Qwen2.5-14B、Qwen3-30B-A3B），12 個 LongBench 資料集。
  - 結果：
    - 對「固定壓縮＋LRU」：同品質 TTFT 少 1.43–3.77×。
    - 對 IMPRESS：1.5–5.2×。
    - 對 prefill：品質掉 3% 以內時快 1.29–2.19×（p.7、p.9）。
  - 磁碟層上的 KV 平均壓得比 CPU 層狠（p.10 Fig. 11 的文字）。
- **GPU 爭用**：沒有建模；重新 profiling 會讓正在跑的請求變慢（Fig. 10 p.10）。
- **自己寫的限制**（p.12）：
  - 假設各層頻寬已知、而且在 profiling 間隔內穩定。
  - 頻寬變得比重新 profiling 快、或和突發到達相關時，效用分數會暫時偏離。
  - 「we plan to … react more quickly to bandwidth shifts」。
  - 快慢兩層頻寬接近時，好處會變小。
  - DRAM 夠大時好處消失。
- **程式碼**：論文沒附（NOT_FOUND）。
- **和 B 的差距**〔判讀〕：B 的「依層級選壓縮」在 EvicPress 裡最完整。它沒有的剛好是 B 剩下的：
  - 執行期的頻寬與 GPU 忙碌——它自己列為未來工作的前半。
  - 無損。
  - 量化——它的評測只有丟 token。
  - NFS。
  - AMD。

#### KVServe（Z. Liu, X. Ma *et al.*, SIGCOMM'26；arXiv 2605.13734）〔全文，16 頁〕
- **服務情境（§3.1 p.4）**：c＝(w, B, T_SLO, q_min)。B 是「application-level goodput under contention」的有效頻寬。每個**請求**在 KV 開始搬時選一次設定，之後固定。
- **模型**（Eq. 1 p.5）：T_p＝T_model＋V/s_p＋V/(B·cr_p)。s_p 是編碼與解碼吞吐的調和平均，是**每個設定的常數**。
- **兩個定理**：
  - Thm 6.1（p.8）：B\*_p＝(1−1/cr_p)·s_p，鏈路比 B\* 快就不要壓，和 V 無關。
  - Thm 6.2（p.8–9）：最佳設定在 x＝1/B 上是分段常數（下包絡）。
- **殘差修正的 bandit（§6.2 p.8–9）**：
  - 原文：「GPU load and queue contention change the actual (de)compression throughput, and system scheduling and concurrency introduce additional overhead」。
  - 做法：每個（品質桶 b, 頻寬區間 i）各自維護 2–3 個候選的 EWMA 殘差，用 ε-greedy 選；有 SLO 護欄與冷卻。
  - 〔判讀〕GPU 負載**不是** context 的一個維度：同一個頻寬區間下，GPU 閒和 GPU 忙會共用同一組殘差估計。所以它只能追蹤慢慢漂移的平均負載，不能依「此刻 GPU 多忙」換設定。
- **策略池（§5.1 p.6）**：
  - Transform：Delta、Hadamard、Affine。
  - Quantizer：多維量化、MixHQ（retrieval／streaming head 混精度）。
  - Codec：nvCOMP。
  - **沒有低秩、沒有 token 合併**；DuoAttention（剪枝）只當對照。
- **評測**：
  - 實作：vLLM 0.10.1（p.9）。
  - 測試平台（p.10）：離線量 4×A100、decode H100；prefill 有三級——4090／5090 @10 Gbps、Pro 6000 @50 Gbps、H100 @100 Gbps。
  - 頻寬用 tc／NIC 限速 5–100 Gbps（p.11）。
  - 結果：PD 的 JCT 最多 9.13×；遠端 KV pool 的 TTFT 最多 32.8×（對 CacheGen 退回重算的情況）；每次決策 <1 ms（p.10–11）。
  - 動機實驗：三種方法的門檻 50／55／110 Gbps（p.3）。
- **程式碼**：GitHub hpdps-group/KVServe（API 確認存在，最後 push 2026-10-12）。
- **和 B 的差距**〔判讀〕：
  - 「依頻寬選壓縮」最完整的前作，而且是**線上**選。
  - 沒有的：GPU 忙碌當輸入、本機分層（只有網路與遠端 KV pool）、AMD（codec 段綁 nvCOMP）。
  - 無損只出現在量化之後的 codec 段，不是「純無損 vs 有損」的選擇。

#### ShadowServe（X. Xiang *et al.*；arXiv 2509.16857v1）〔全文，PDF 已存 stage3_L5〕
- **干擾的實測（§2.2 p.3、Fig. 3）**：
  - L40S 上解壓和 LLM decode（30K、batch 1）放一起，試了 CUDA stream、MPS、MPS 切 SM、Green Context。
  - 算術解碼：做不到兩邊都慢 <30%。
  - 反量化：最好的組合兩邊仍各慢 >25%。
- **CPU 解壓（Table 1 p.3）**：單一主機核 Deflate 2.5 Gbps、LZ4 18.6 Gbps；BlueField-3 加速器 276.5／246.3 Gbps。作者也說主機 CPU 常被向量搜尋等工作占用（p.1）。
- **作法**：分箱量化＋Deflate（固定，p.8）；整條取回與解壓的資料路徑放在 SmartNIC 上（p.4）。
- **取捨（§6，p.10）**：
  - 頻寬 ≤20 Gbps 或輸出 ≥16 token 時 ShadowServe 贏；頻寬高且輸出短時，GPU 解壓（CacheGen-Async）贏。
  - GPU 解壓在干擾下最多約 32 Gbps。
  - 〔判讀〕**這是最接近「連線速度 × GPU 負載 → 在哪裡解」的一張取捨圖**，但它只報告取捨，**沒有依當下狀態切換**，格式也是固定的。
- **限制**：不支援 chunked prefill、不支援部分命中（p.6）；提到 AMD Pensando 理論上可以（p.7），沒有實作。

### 2.3 和 B 有關、但只到摘要或部分深度的
- **VeriCache**（arXiv 2605.17613）〔摘要〕：用壓縮過的 KV 先草擬 token，再用完整 KV 驗證，保證輸出相同。摘要寫有損方法的輸出「increasingly diverge … catastrophic failures in code generation and tool calling」。〔判讀〕這支持 B 需要一個「只准無損」的模式。
- **UltraQuant**（EMNLP'26 Industry；arXiv 2606.20474）〔摘要〕：AMD 團隊，4-bit KV，CDNA4 的 FP4 路徑，Claude Code trace。KV 在 HBM 內就是 4-bit，不是卸載時才壓。
- **RDKV**（arXiv 2605.08317）、**AATC**（arXiv 2608.14191）〔摘要〕：把丟 token 與量化統一成「每個 token／channel 分幾個 bit」的率失真問題，作用在 GPU 內，不看鏈路。〔判讀〕如果 B 要把「丟 token」也放進動作集，這是現成的語言。
- **KVSwap**（arXiv 2511.11907）〔摘要〕：裝置端，完整 KV 放磁碟，記憶體裡留精簡的中繼資料來預測要預載哪些。**ShadowKV**（arXiv 2410.21465）〔摘要〕：低秩 K 留在 GPU、V 卸到 CPU。兩者的低秩都是固定的，不是依鏈路選的。
- **系統化的 KV 綜述**（arXiv 2607.08057）〔部分〕：
  - p.5：多數系統處理頻寬時「without explicit modeling contention among concurrent KV transfers」。
  - p.8–9：把「KV generation, migration, and compression」之間的跨行為干擾導致 SLO 違約，列為開放挑戰 C1。

---

## 3. 查新判定

### 3.1 B 的各個子主張

| # | 子主張 | 判定 | 最接近的前作（出處） |
|:--|:--|:--|:--|
| B1 | 依連線速度選壓縮等級 | **已做** | CacheGen（讀回時逐 chunk，p.6–7）；KVServe（Thm 6.1–6.2，p.8–9） |
| B2 | 依儲存層級選格式與壓縮率 | **已做** | AdaptCache（p.2）；EvicPress（p.6–7）；LMCache 每個 adapter 的 serde（靜態）〔文件〕 |
| B3 | GPU 上保持原格式，卸載時才壓 | **已做**（固定格式） | TRT-LLM cold-page NVFP4〔文件〕；LMCache serde〔文件〕；Metrum（SSD 邊界）〔部分〕 |
| B4 | 「純無損 vs 有損」放在同一個動作集裡選 | **部分** | KVServe 的 codec 段（量化之後再做 nvCOMP）；RFC #4498 只有無損；**沒有人依連線＋負載在兩者之間選** |
| B5 | **GPU 忙碌當成決策輸入** | **NOT_FOUND** | KVServe 把它當漂移（p.8–9）；CacheGen 只透過重算時間（p.19）；HCache 用離線硬體 profile（p.6–7） |
| B6 | **讀回時依 GPU 忙碌選在哪裡解**（CPU／GPU／不壓） | **NOT_FOUND** | ShadowServe、KVCodec 靜態搬到別的硬體；LMCache RFC 靜態預設 CPU 壓；FlexGen 靜態關掉 CPU 代算（p.7）；Intel KV Shrink 的「三個引擎誰先做完誰接」是壓縮端的動態分派，不看 GPU |
| B7 | 本機 DRAM／SSD／NFS | **部分** | AdaptCache（DRAM／SSD 1 GB/s）；EvicPress（CPU／SSD／遠端磁碟）；**NFS NOT_FOUND** |
| B8 | AMD／ROCm | **部分**（只有固定格式或提案） | AMD ROCm 部落格（MI355X，固定 4-bit）〔部分〕；RFC #4498 的 HipcompBackend（提案）〔全文〕；hipCOMP-core（early access）〔README〕 |
| B9 | 低秩、token 合併進入卸載格式的決策 | **NOT_FOUND** | 丟 token 有（AdaptCache、EvicPress）；低秩只有固定形式（KVTC、ShadowKV）；合併只在 EvicPress 的分類出現（p.3） |

### 3.2 四個指定問題

**(a) 有沒有人把 GPU 忙碌放進決策？** 沒有（NOT_FOUND）。
- 有人量干擾：ShadowServe（L40S，≥25–30%，p.3）、KVCodec（prefill +50%、decode +20%，L2 B5）、HACK（反量化佔 JCT 17.2–37.9%，L2 B6）、我們的第二階段（09 §6.1、11 §3d）。
- 有人用靜態放置避開它：SmartNIC、NVDEC、CPU 預設（RFC #4498）。
- 有人把它當誤差吸收：KVServe 的 EWMA。
- 有一張接近「頻寬 × GPU 負載」的評測熱圖：CacheGen 附錄 D，p.19。但它是評測，不是決策。
- 〔推論〕「依此刻 GPU 多忙換格式或換解壓位置」這件事本身沒人做，**但它是不是值得做，要看解壓對 GPU 忙碌有多敏感**——見 §4：我們的一次拆在 MI300X 上幾乎不敏感。

**(b) 有沒有人在本機分層（CPU DRAM、本地 SSD、NFS）而不是網路上做？** 有，但只做到 DRAM＋SSD。
- AdaptCache：DRAM＋SSD，磁碟讀 1 GB/s（p.2）。
- EvicPress：CPU／本地 SSD／遠端磁碟（p.8）。
- 兩者都假設頻寬是已知常數。
- **NFS（共享檔案系統、頻寬會跳）沒找到**；EvicPress 把「頻寬變動」列為限制（p.12）。
- 我們已量到 NFS 會跳：第一階段 C2 讀 376.7–446.9 MiB/s，同一天稍後 500–830 MiB/s（L2 §0.1 轉述 07 的 C7 錨點段）。

**(c) 有沒有 AMD／ROCm 的版本？** 只有固定格式與提案。
- AMD ROCm 部落格：MI355X、TurboQuant 4-bit 經 LMCache 溢出到 CPU DRAM，格式固定，HBM 內就是 4-bit〔部分〕。
- UltraQuant（EMNLP'26 Industry）〔摘要〕。
- LMCache RFC #4498 要同時做 `NvcompBackend` 與 `HipcompBackend`，2026-10-04 作者說要拆 7 個 PR，目前沒有實作〔全文〕。
- hipCOMP-core v2.3.0 early access；README 列出在 gfx942、ROCm 7.1.1／7.2.0／7.2.3 測試過（**我們是 7.2.2，不在清單上**）〔README〕。
- Künas 等人在 MI300X 上量通用無損，浮點資料只有 1.03–1.04×（L2 B10）。
- 依連線或 GPU 忙碌自適應的 AMD 版：NOT_FOUND。

**(d) 低秩、token 合併有沒有被放進這個決策？**
- **token 丟棄**：有。AdaptCache 的動作集是 token 丟棄 vs 量化（p.1）；EvicPress 只有丟棄（p.8）；CacheGen 的「送文字重算」等於丟掉再算。
- **低秩**：只有固定形式。KVTC 用 PCA＋DP 位元分配，壓縮比在校準時定死，L2 B3 記錄它「沒有和鏈路頻寬掛鉤的盈虧分析」；ShadowKV 的低秩 K 是固定 rank。
- **token 合併**：EvicPress 在背景分類裡列了 HOMER、Look-M（p.3），沒有用；KVServe 的策略池沒有。依鏈路或負載選低秩 rank、選合併比例：NOT_FOUND。
- 〔推論〕低秩重建是一次 GEMM，**正好和背景 matmul 搶同一種資源**。如果 B 真的要一個「對 GPU 忙碌敏感」的格式，低秩比量化更可能是那個格式。但它要先過品質（ε），這不在最小實驗裡（§5.3 N4）。

### 3.3 新意還剩什麼形狀
- **最可能成立的版本**〔推論〕：「**寫入時依層級選格式，讀回時依 GPU 忙碌與 CPU 空閒選在哪裡解**」。後者對無損格式特別自然：同一份無損位元組，CPU 也能解、GPU 也能解，**不需要多版本儲存**（CacheGen 要），也**不需要在寫入時預測讀回時的 GPU 狀態**。
- **新意風險**：
  1. 第二階段已經看到，一次拆讓反量化幾乎不受背景影響（§4.1）。「GPU 忙」這個軸對量化格式可能沒有作用；只有比較重的解碼器（熵解碼、低秩 GEMM）才可能有。
  2. 「只看連線速度」的分層靜態已經是前作（KVServe、LMCache serde）。如果決策圖只隨連線速度變，B 的貢獻就只剩「AMD＋NFS 上的工程數字」。
  3. 無損的天花板很低：r 最多約 1.33（SplitZip 的格式上限，L0 §4.0），省下的最多是傳輸時間的 25%。

---

## 4. 用第二階段已量的數字，先畫一張「預測圖」（0 GPU）

### 4.1 CPU 層（真實 pinned H2D）：允許有損時，INT4 每格都贏

32K、全部載入（loadall）、方法 P3＝全部搬完一次拆；中位數，ms〔實測，`results/m10_restore2_mi300x/tables_13.md` Q4；run `20261018-093743-m10-rs2-q4-llama-fp8-P0` … `-112037-…-qwen-int4-P5`；P2 重量 `20261018-135548-m10-rs2-q4-llama-fp8-P2-remeasure`；gpu_guard 除被重量的那格以外全 CLEAN，RUNLOG 14.1〕：

| 模型 | GPU | BF16（E0） | FP8 | INT4 KIVI |
|:--|:--|--:|--:|--:|
| Llama | 閒 | 84 | 46 | 33 |
| Llama | 忙（duty 0.7） | 124 | 49 | 36 |
| Qwen | 閒 | 39 | 21 | 18 |
| Qwen | 忙 | 55 | 22 | 19 |

- **背景強度掃描**（INT4、P3、全載入，duty 0.4／0.7／0.9）〔實測，run `20261018-112419-m10-rs2-q4duty-llama-P0` … `-113436-…-qwen-P3`〕：
  - Llama 35.2／36.1／36.7 ms，Qwen 19.3／19.7／20.8 ms。
  - 對照：逐塊拆（P0）是 Llama 44.3／55.2／60.0、Qwen 28.6／42.8／52.9。
  - 背景 matmul 吞吐不掉（P3 367／372 vs P0 368／371）。
- 原寫法（逐塊、PyTorch 拼的）的 FP8 反量化在忙時慢 2.4–3.8 倍（09 §6.1），**換成一次拆之後這個差幾乎消失**。

### 4.2 慢層：INT4 一樣每格都贏
- b16／b32／b48（Llama 5.27／1.47／0.42、Qwen 2.77／0.76／0.22 GiB/s），閒與忙：INT4 對 BF16 快 36.1–56.9%〔實測，08 §2.5.1；run `20261016-034902-m10-grid-int4-llama`、`20261016-052143-m10-grid-int4-qwen`〕。
- 全部放在慢層（GPU 閒，原寫法的拆）〔實測，08 §2.8 B；run `20261018-154708-m10-tier2-llama`、`20261018-160950-m10-tier2-qwen`〕：

  | 模型 | 慢層 | BF16 | INT4 |
  |:--|:--|--:|--:|
  | Llama | NFS 級 | 2.64 s | 1.72 s |
  | Llama | SSD 級 | 0.47 s | 0.26 s |
  | Qwen | NFS 級 | 1.79 s | 0.98 s |
  | Qwen | SSD 級 | 0.23 s | 0.14 s |

- 品質：INT4 KIVI 在 32K 兩個模型 24 格 ε 全過（08 §2.5.3）；64K／128K 和 E0 差 ≤2（Llama）、≤2.5（Qwen）個百分點（`results/m10_longctx_mi300x/analysis/tables_longctx.md`，RUNLOG 15.9）。FP8 只有 Llama 過（Qwen 的 fp8_head −32.8，08 §2.3）。
- 〔推論〕**模式 Q 的預測圖是一整片 INT4**。自適應對「事後最好的固定格式（INT4）」約等於 0；對「不壓縮」則有 32–67%，但那是「改用 INT4」的好處，不是「自適應」的好處。這一半在 Step 0 就應該照停止條件收掉（§5.5 STOP-0）。
- **例外要查的一點**：INT2 KIVI 在 CPU 層閒時更快。Llama 的 E1s 融合後是 0.035 s，INT4 KIVI 的 E1 是 0.065 s（RUNLOG 14.3，43d Q3 段；11 §3a）。但品質沒過 ε：唯一的 INT2 選手 `SW-ERR-20@int2_kivi` 對 E0 是 −1.2 [−2.6, +0.1]（RUNLOG 11.2p）。而且太狠的格式在 64K／128K 有幾格會隨長度變差（RUNLOG 15.9）。所以它**不在安全的動作集裡**。

### 4.3 模式 L（只准無損）：省多少的上界〔算術〕

**公式**：
- Cake 的離散會合：T＝min_b max(Σ_{i<b} f_i, (64−b)·chunk_bytes·ρ／B)。
- ρ：位元組比例（無損 1/r；INT4 KIVI 是 20/64）。
- f_i：每個 chunk 的實測重算時間中位數，讀 `results/m10_precision_mi300x/speed_<m>/bw.json`（run `20261015-154558-m10-calibf-llama`、`20261015-154647-m10-calibf-qwen`）。
- GPU 忙：f 乘上 2.117（Llama）／2.114（Qwen）（`costs_sources.csv` 的 busy_x_compute(32768)，run `20261015-161226-m10-grid-fp8-llama`、`20261015-175050-m10-grid-fp8-qwen`）；慢層的載入時間不變（限速主導）。
- **假設**：解壓免費、完全重疊；不含新一輪；r 是假設值（我們的 r 是 NOT_MEASURED）。

**尺準**（全 BF16，公式 vs 實測）：

| 情境 | 公式 | 實測 | 出處 |
|:--|--:|--:|:--|
| Llama b16 閒 | 0.569 s | 0.578 s | 08 §2.5.1 |
| Llama NFS 級（0.331） | 2.599 s | 2.64 s | 08 §2.8 |
| Llama SSD 級（6.98） | 0.457 s | 0.47 s | 08 §2.8 |

**對不壓縮（BF16）的還原時間省多少**：

| 模型 | 鏈路（GiB/s） | GPU | BF16（s） | 無損 r＝1.25 | 無損 r＝1.33 | INT4 位元組（只算搬運） |
|:--|:--|:--|--:|--:|--:|--:|
| Llama | NFS 限速 0.331 | 閒 | 2.599 | −7.0% | −8.3% | −36.5% |
| Llama | NFS 限速 0.331 | 忙 | 4.181 | −9.7% | −11.9% | −46.4% |
| Llama | NFS 實測中位 0.383 | 閒 | 2.454 | −5.8% | −8.6% | −37.6% |
| Llama | NFS 實測中位 0.383 | 忙 | 3.921 | −10.0% | −12.5% | −47.9% |
| Llama | SSD 實測中位 7.87 | 閒 | 0.413 | −16.9% | −20.7% | −63.9% |
| Llama | SSD 實測中位 7.87 | 忙 | 0.461 | −18.6% | −23.7% | −66.6% |
| Qwen | NFS 限速 0.331 | 閒 | 1.734 | −8.6% | −10.7% | −44.9% |
| Qwen | NFS 限速 0.331 | 忙 | 2.560 | −12.3% | −15.3% | −53.6% |
| Qwen | SSD 實測中位 7.87 | 閒 | 0.198 | −18.6% | −22.8% | −66.0% |
| Qwen | SSD 實測中位 7.87 | 忙 | 0.209 | −18.8% | −23.8% | −67.2% |

鏈路的出處：
- NFS 實測中位 391.7 MiB/s＝0.383 GiB/s，SSD 實測中位 8,053.8 MiB/s＝7.87 GiB/s：`results/m7_write_policy_mi300x/calib_c2.csv`，run `20261008-130438-m7-c2`（經 `costs_sources.csv` 轉錄）。
- 限速參數：`tier_params.json`。

**讀法**〔推論〕：
1. 無損的上界在 NFS 級只有 6–16%，SSD 級 17–24%。前提是解壓真的「免費且重疊」。
2. **GPU 忙時，NFS 級的好處變大**：重算變貴，Cake 改成多載入，位元組變少就更值錢。這是「GPU 忙改變最佳選擇」唯一明確的機制，但方向是「更該壓」，**不會讓最佳格式翻轉**。
3. **真正可能翻轉的是解壓位置**（〔算術〕，用 ShadowServe Table 1 的單核速度，是別台 CPU、量化過的資料，只當量級參考）：
   - SSD 級、r＝1.33 時，解壓輸出要跟上約 7.87×1.33≈10.5 GiB/s。LZ4 單核約 2.17 GiB/s → 約 5 核；Deflate 單核約 0.29 GiB/s → 約 36 核，超過 32 核配額。
   - NFS 級只要約 0.51 GiB/s，Deflate 約 2 核就夠。
   - CPU 層（49.9 GiB/s）CPU 解壓幾乎不可能跟上 → 原樣搬最好。
   - 但 LZ4 直接壓 BF16 KV 幾乎壓不動：SplitZip 的 Table 2 量到 nvCOMP LZ4 1.019×（L2 B1）。實際要用的是「位元組分組＋zstd」或指數碼本，**它們在這台 EPYC 上的速度是 NOT_MEASURED**。
4. vLLM 0.28 的 CPU 層只有 4.02–4.11 GiB/s（run `20261008-141324-m7-c6-vllm`），但瓶頸是 copy **次數**（每次約 3.5 µs，run `20261010-112119-f1-s1-rocmfixcoal`，F1 §4.6）。〔推論〕壓縮減少的是位元組、不是次數，所以那條路徑上「連線速度」不是位元組比例的成本。決策圖要用「和位元組成比例的那部分成本」，不能直接用 4.1 GiB/s。

**預測圖**〔推論，待 Step 1–2 檢驗〕：
- **模式 Q**：整張 INT4 → 停。
- **模式 L**：CPU 層原樣；NFS 級用 CPU 解的無損；SSD 級要看兩件事——CPU 解的多核速度，以及 GPU 解在忙時被拖多少。**只有 SSD 級這一帶，最佳選擇可能隨 GPU 忙碌翻轉**，這就是 B 唯一可能的新意區。

---

## 5. 最小實驗設計

### 5.1 假說（先寫死）
- **H-B0（模式 Q）**：用第二階段驗證過品質的格式（Llama：BF16、FP8、INT8、INT4 KIVI；Qwen：BF16、INT8、INT4 KIVI），在（鏈路 × GPU 忙碌 × 模型）的每一格，最佳格式都是 INT4 KIVI。**預期成立**（§4.1–4.2）→ 有損自適應沒有空間。
- **H-B1（模式 L）**：我們 32K 真實 KV 的 BF16 無損比 r ≥1.2（位元組分組或只壓高位元組＋zstd；或指數碼本）。
- **H-B2（模式 L，B 的新意）**：在 SSD 級附近（約 3–12 GiB/s），最佳解壓位置（CPU 多執行緒、GPU 一次拆、不壓）會隨 GPU 忙碌（duty 0 → 0.7）翻轉。而且「連線＋GPU 忙」的選擇，比「只看連線」的分層靜態，在至少一部分格子好 ≥5%。

### 5.2 已有、直接用的常數（不重量）

| 量 | 值 | 檔案 | run_id |
|:--|:--|:--|:--|
| 真實 pinned H2D（單塊） | Llama 49.858、Qwen 48.644 GiB/s | `results/m10_precision_mi300x/speed_{llama,qwen}/bw.json` | `20261015-154558-m10-calibf-llama`、`20261015-154647-m10-calibf-qwen` |
| 每 chunk 重算時間 f(i)（64 塊） | 總和 3.631／3.077 s | 同上 | 同上 |
| H2D 線性擬合（交叉參考） | 48.17／45.95 GiB/s | `results/m16_stage3_mi300x/i3_pause_sim/costs_sources.csv` | 由 `…-grid-fp8-*`、`…-lc-grid-*` 擬合（見該檔） |
| CPU 層限速參數 | 讀 35.4、寫 35.5 GiB/s | `results/m7_write_policy_mi300x/tier_params.json` | `20261008-130257-m7-c0`、`20261008-130438-m7-c2` |
| 本地 SSD 實測 | 讀中位 8,053.8 MiB/s（7,943.0–9,152.2）、寫 2,178.9 MiB/s；限速參數 6.98 | `results/m7_write_policy_mi300x/calib_c2.csv`；`tier_params.json` | `20261008-130438-m7-c2` |
| NFS 實測 | 讀中位 391.7 MiB/s（376.7–446.9）、寫 689.2 MiB/s；限速參數 0.331 | 同上 | 同上 |
| NFS 同一天稍後的錨點 | 約 500–830 MiB/s | `docs/phase1_20261008/07_report.md` C7 段（L2 §0.1 轉述） | **NOT_FOUND**（本檔沒有核到 run_id） |
| D2H | 42.877 GiB/s | `results/m7_write_policy_mi300x/calib_c0.csv` | `20261008-130257-m7-c0` |
| vLLM 0.28 CPU 層 | 4.02／4.09／4.11 GiB/s（8K／16K／32K） | `docs/phase1_20261008/07_report.md` §7 | `20261008-141324-m7-c6-vllm` |
| 每次 copy 的送出成本 | 約 3.5 µs／次〔算術〕 | `docs/research_20261010_followup/F1_offload_slowdown.md` §4.6 | `20261010-112119-f1-s1-rocmfixcoal` |
| GPU 忙的定義 | duty 0.4 → 重算 ×1.48；0.7 → ×2.07 | `results/m7_write_policy_mi300x/busy_levels.json` | `20261008-130834-m7-c5`、`20261008-130956-m7-c5b` |
| 32K 忙／閒倍數 | 重算 ×2.117／×2.114；H2D ×1.50／×1.43（Llama／Qwen） | `speed_<m>/summary_fp8.csv`（經 `costs_sources.csv`） | `20261015-161226-m10-grid-fp8-llama`、`20261015-175050-m10-grid-fp8-qwen` |
| 原寫法 FP8 反量化（逐塊） | 只載：Llama 14.9 → 56.8 ms、Qwen 6.8 → 26.0 ms（閒 → 忙） | `09_sweep_plan.md` §6.1；`restores.csv.gz` | `20261015-161226-m10-grid-fp8-llama`、`20261015-194150-m10-regrid-fp8-qwen` |
| 融合、一次拆（P3）的 CPU 層還原 | 見 §4.1 | `results/m10_restore2_mi300x/tables_13.md` Q4 | `20261018-093743-…` 到 `-113436-…`（RUNLOG 14.1） |
| 全部在慢層的 BF16／INT4 | 見 §4.2 | `results/m10_tier_mi300x/tables_<m>.md` | `20261018-154708-m10-tier2-llama`、`20261018-160950-m10-tier2-qwen` |
| 每塊大小 | Llama BF16 64／FP8 32／INT4 KIVI 20 MiB；Qwen 28／14／8.75 MiB | `08_report_mi300x.md` §2.8 A1 | 〔算術〕 |
| 品質 ε | INT4 KIVI 兩個模型 24 格全過；FP8 Llama 過、Qwen 不過；Qwen INT8 過 | `analysis/int4/tables_int4_<m>.md`；08 §2.3–2.5 | 33b 的 12 個 run（08 §2.5.3 列出） |

**沒有、要新量的**：
- 我們 KV 的真實無損比：NOT_MEASURED。
- 這台 EPYC 上的 CPU 解壓吞吐：NOT_MEASURED。
- MI300X 上 GPU 無損解碼閒／忙的時間：NOT_MEASURED。
- 低秩重建在忙時的時間：NOT_MEASURED。
- 真實 NFS 讀取＋GPU 忙同時的還原：NOT_MEASURED（L2 X5）。
- hipCOMP 在 ROCm 7.2.2 上能不能編、能不能跑：NOT_MEASURED。

### 5.3 要新量的（最少）

| 編號 | 量什麼 | 怎麼量 | 資源〔估計〕 |
|:--|:--|:--|:--|
| **N1** | 真實 KV 的無損比：BF16 原樣、位元組分組（bytes2）、只壓高位元組（bytes2hi，＝Intel KV Shrink 的作法）、指數平面（planes／planes_e）× zstd（−3、1、3）／zlib（1、6）；外加 order-0 熵；FP8、INT4 碼也做一次；逐層、K／V 分開 | `code/m16_s1_lossless.py`（自測 ALL PASS，experiment_log S1），資料用 `code/m16_s1_kvdump.py` 匯出（Llama、Qwen 32K 全層） | 匯出 **0.2 GPU 時**（experiment_matrix S1）；壓縮統計純 CPU |
| **N2** | CPU 解壓吞吐與 CPU 成本：1／4／8／16 執行緒的輸出 GiB/s、每 GiB 花幾 CPU 秒、逐位元還原檢查 | 同上，只用已裝的 zlib／zstandard | CPU 2–4 小時（L2 X1 的估計）；**只在第二階段沒有量時間的時段跑**，否則綁 ≤4 核、nice 19 |
| **N3** | 各格式在 GPU 忙碌下的解碼時間，以及它拖慢背景多少（雙向，同 ShadowServe Fig. 3 的量法）。格式：FP8、INT4 融合（逐塊 vs 一次拆，**重新錨定**，和 §4.1 對照）；GPU 無損（指數碼本解碼的 Triton kernel——SplitZip README 寫要 CUDA，移植未驗證；或 hipCOMP 的 LZ4／Cascaded——**要安裝，需使用者同意**）；duty 0／0.4／0.7／0.9；兩個模型；3 launch × 10 rep；gpu_guard | 沿用 `m7_busy.py`、`m10_speed.Busy`、`m10_dequant_fast.py` 的量法 | **0.5–1 GPU 時**。依據：43b 的四個背景強度掃描約 15 分鐘（run `-112419-` 到 `-113436-`）；這裡格式更多、只量微測 |
| N4（選做） | 低秩重建（每層 K 的 rank r＝32、64，head_dim 128）的 GEMM 在忙碌下的時間——**只量成本，不量品質** | 同 N3 | +0.2 GPU 時。品質沒過 ε 前，**不能進決策圖**，只標成「如果品質過了，它會落在哪」 |
| **N5** | 驗證：只挑 §5.4 預測圖上「最佳選擇翻轉」或「差 <10%」的 4–8 個格子，做端到端還原（G5b 會合規則、新一輪 256 token）；鏈路用 SSD 限速 7.87、NFS 限速 0.383、真實 CPU 層；duty 0／0.7；2 launch × 5 rep。真實 NFS 讀取只做 1–2 格，而且只在沒有量時間的 stage 時做（experiment_matrix §6） | `m10_tier2.py` 式的 grid（`Restorer` 的 `host[i]` 可以是 `Enc`，限速器照實際位元組計時，`stage2_factcheck` §3.1） | **0.5–1 GPU 時**。依據：stage 44 兩層混合每個模型 1,000 次還原，兩個模型共約 37 分鐘（08 §2.8 B，15:46 → 16:23Z） |

**合計**〔估計，非量測〕：GPU 1.5–2.5 小時，加緩衝 ≤3 小時；CPU 3–5 小時。量時間的 job 照第二階段的規矩兩卡獨佔（除非 44cb 判出 `XGPU_OK`），**排在第二階段佇列之後**（使用者 10/19 的決定）。

### 5.4 格子、對照、指標

- **格子**：
  - 鏈路：0.25、0.33（NFS 限速）、0.383（NFS 實測中位）、0.83（NFS 稍後錨點）、1、3、6.98（SSD 限速）、7.87（SSD 實測中位）、12、20、49.9 GiB/s（真實 pinned）。其中 1、3、12、20 和 stage 44 一樣。
  - GPU duty：0、0.4、0.7、0.9。
  - 模型：Llama、Qwen。
  - 長度：32K（主）；128K 只用算術延伸。
  - 預測圖用 §4.3 的公式，加上 N1–N3 量到的 r、編解碼時間與忙碌倍數；N5 只驗證挑出來的格子。
- **動作集**：
  - 模式 L：{原樣、無損＋CPU 解（執行緒數 1／4／8／16）、無損＋GPU 解（一次拆）}。
  - 模式 Q：模式 L ∪ {FP8（只有 Llama）、INT8、INT4 KIVI}（融合、一次拆）。
  - 還原方法固定用 G5b（第二階段對每格最好只差 0.4%，RUNLOG 14.2），只讓格式與解壓位置變。
- **三種比較（每一步都要報，不准只報最強的那個）**：
  1. **D＝部署預設**：不壓縮。vLLM 0.28 OffloadingConnector 只搬原始位元組〔程式碼〕。
  2. **F＝事後最好的固定格式**：在同一組格子上，幾何平均最小的那一個格式（＋固定的解壓位置）；另報「每個模型各自最好的固定格式」。
  3. **S＝只看連線的分層靜態**：每個鏈路速度各自最好的一個選擇，不看 GPU 忙碌。等於 LMCache 每個 adapter 一個 serde、或 KVServe 的分段策略能做到的。
  4. **A＝連線＋GPU 忙（B）**：每格最好。**先用「還原時 GPU 忙碌完全已知」算上界**；上界都不過，就不用設計預測器。對無損格式，「在哪解」本來就是讀回時才決定，忙碌是讀回當下看得到的，不需要預測。
- **指標**：
  - 每格的還原時間＋新一輪 TTFT（中位數、P95）。
  - 聚合：A／D、A／F、A／S 的幾何平均；最差一格；「A 比 S 好 ≥5% 的格子比例」。
  - 背景吞吐的變化（對方被拖慢多少）。
  - CPU 秒／GiB。
- **品質**：動作集只用第二階段已驗證過 ε 的格式＋無損，**不跑新的品質**。低秩、合併、INT2 不進主判定。

### 5.5 停止條件（事先寫死）

| 代號 | 什麼時候判 | 條件 | 結果 |
|:--|:--|:--|:--|
| **STOP-0** | Step 0（0 GPU，現在就能做） | 模式 Q 用 §5.2 的實測常數，每一格的最佳格式都相同 | 有損自適應 NO-GO，只留模式 L。〔預測：成立〕 |
| **STOP-1** | N1／N2 之後 | 兩個模型的 BF16 無損比都 <1.15（沿用 L2 G1 的門檻）；或在 ≤16 執行緒下，CPU 解壓連 NFS 級需要的約 0.5 GiB/s 都跟不上，而且又沒有可用的 GPU 解碼器 | 模式 L NO-GO → **B 整條停**（寫成負面結果） |
| **STOP-2a** | N3 之後，用預測圖（上界：忙碌已知） | 每一格 A 對 S 都好不到 5% | **GPU 忙碌這個軸沒有價值**；B 退化成前作（KVServe、LMCache serde）→ 新意 NO-GO。D、S 的工程數字照報 |
| **STOP-2b** | 同上 | A 對 F 的幾何平均好不到 5% | NO-GO（規則 4） |
| **STOP-2c** | 同上 | A 對 F（或對 S）的好處在 5–15% | **停下來問使用者**（CLAUDE.md §7） |
| **GO 到 N5** | 同上 | A 對 S 在 ≥25% 的格子好 >15%，而且 A 對 F 的幾何平均 >15% | 做 N5 驗證 |
| **STOP-3** | N5 之後 | 驗證格子裡有 ≥2 格的實測和預測差 >10%；或實測的 A 對 F <5% | 預測圖不可信，或沒有好處 → 停 |

**一定要寫進結果的**：A 對 D 的數字。即使 STOP-2 停，「不壓縮 → 每層最好的固定做法」對部署是有用的工程結論，但它**不是 B 的新意**。

### 5.6 結果會長成什麼樣（決策圖）
- **主圖**：x＝鏈路速度（對數軸，0.25–50 GiB/s，標出 NFS 0.383、SSD 7.87、CPU 49.9 三條真實線，另外註記 vLLM 4.1 是 copy 次數限制、不是位元組限制）；y＝GPU duty（0、0.4、0.7、0.9）。每格顏色＝最佳的（格式, 解壓位置）；第一名和第二名差 <5% 的格子打斜線。
- **面板**：Llama／Qwen × 模式 L／模式 Q。
- **副圖**：每格 F、S 對 A 的悔恨（%），看 GPU 忙碌那個方向有沒有任何格子 ≥5%。
- 〔預測〕模式 Q 是一整片 INT4；模式 L 是「左邊（慢）CPU 解、右邊（快）原樣」兩塊，中間 SSD 帶的分界線**可能**隨 duty 移動。**如果那條線是垂直的（不隨 duty 移動），B 的新意就沒了。**

### 5.7 執行規則與風險
- 共用機器、第二階段優先：GPU 的 job 排在第二階段佇列之後，每個 job ≤2 小時，gpu_guard 包住；`contaminated` 的時間欄作廢重量（CLAUDE.md §3）。
- CPU 量測（N2）會污染別人的時間欄：只在沒有量時間的 stage 時跑；否則 `nice -n 19 taskset -c 0-3`，只量 1／4 執行緒，8／16 留到空檔。
- NFS 是整個叢集共用，gpu_guard 抓不到 NFS 上的別人（L2 X5）：真實 NFS 的格子要同時記一次基準讀取。
- hipCOMP 要從原始碼建，SplitZip 要移植 Triton kernel：**兩者都要使用者同意才做**；沒有 GPU 無損解碼器時，模式 L 的動作集就只剩「原樣 vs CPU 解」。這時 B2 的「GPU 忙碌翻轉」只能透過 H2D 在忙時變慢（×1.43–1.50）來體現。
- 128K 只做算術延伸；要實測就併進第二階段長 context 的 grid，不另開。

---

## 6. 對既有文件的更正與補充（只列，未改原檔）

1. **L2 B10／§1.4**：「ROCm 上沒有官方的 nvCOMP 對應品」要改成：ROCm 組織下有 **hipCOMP-core**（MIT；README 標 v2.3.0、early-access technology preview、「not yet performance-optimized」；以 nvCOMP branch-2.2 為基礎）。
   - LZ4／Snappy／Cascaded／RLE：能壓也能解。
   - ZSTD／Deflate：只能解，而且是實驗性。
   - Bitcomp／ANS／GDeflate：不支援。
   - 測過的平台：gfx942、gfx950 等；ROCm 7.1.1、7.2.0、7.2.3。
   - GitHub API：2024-09-02 建立，最後 push 2026-10-14〔README〕。
2. **L2 B10 的 LMCache 列**：v1 文件的 `remote_serde`（naive／cachegen）是舊路徑；**MP 模式**的 L2 serde 現在有 `fp8`、`turboquant`（四種 preset）、`aesgcm`，每個 `--l2-adapter` 各自設，預設 `max_workers=1`〔文件，WebFetch〕。
3. **L2 B10 的 Intel QAT 列**：RFC #4149、PR #4264 在 2026-10-20 仍是 open。另有新的 **RFC #4498**（rob-9，2026-08-12）：無損壓縮＋nvCOMP／hipCOMP 在 GPU 上解；預設 CPU 壓縮以避開 GPU 爭用；2026-10-04 作者規劃拆 7 個 PR〔全文〕。
4. **AdaptCache 的 venue**：arXiv Comments 是「SOSP 2025 – BigMem workshop」，正文 3 頁。L2 G4 把它和 EvicPress 並列為「壓縮與淘汰跨層聯合決定」，正確；但 AdaptCache 的動作集**含量化（KIVI）**，EvicPress 的評測**只有丟 token**——兩者不同，引用時要分開。
5. **EvicPress** 的 Fig. 1 與結論寫成「CacheServe」，引用時用 arXiv 標題 EVICPRESS。

---

## 7. 參考清單

「核」欄：✓＝這輪用 curl 開過 arXiv abs 頁（標題、作者、日期、Comments）；G＝用 GitHub API 確認存在；W＝WebFetch 讀過官方頁。

### 讀全文的
| # | 文獻 | 連結 | 核 | 深度 |
|:--|:--|:--|:--|:--|
| [1] | Y. Liu *et al.*, "CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving," SIGCOMM'24 | https://arxiv.org/abs/2310.07240 ；https://github.com/UChi-JCL/CacheGen | ✓ G | 〔全文〕（§5.3、§6、§7.3–7.5、§9、App. C–E）＋P02 |
| [2] | S. Feng, H. Li *et al.*, "AdaptCache: KV Cache Native Storage Hierarchy for Low-Delay and High-Quality Language Model Serving," BigMem workshop @ SOSP 2025 | https://arxiv.org/abs/2509.00105 （v2 2026-01-15） | ✓ | 〔全文〕 |
| [3] | S. Feng, Y. Liu *et al.*, "EVICPRESS: Joint KV-Cache Compression and Eviction for Efficient LLM Serving," arXiv 2025（venue NOT_FOUND） | https://arxiv.org/abs/2512.14946 | ✓ | 〔全文〕 |
| [4] | Z. Liu, X. Ma *et al.*, "KVServe: Service-Aware KV Cache Compression for Communication-Efficient Disaggregated LLM Serving," SIGCOMM 2026 | https://arxiv.org/abs/2605.13734 ；https://github.com/hpdps-group/KVServe | ✓ G | 〔全文〕 |
| [5] | X. Xiang *et al.*, "ShadowServe: Interference-Free KV Cache Fetching for Distributed Prefix Caching," arXiv 2025（venue NOT_FOUND） | https://arxiv.org/abs/2509.16857 | ✓（L2） | 〔全文〕 |

### 部分、文件、README、程式碼
| # | 文獻 | 連結 | 核 | 深度 |
|:--|:--|:--|:--|:--|
| [6] | S. Gao, Y. Chen, J. Shu, "Fast State Restoration in LLM Serving with HCache," EuroSys 2025 | https://arxiv.org/abs/2410.05004 | ✓ | 〔部分〕§4.1、§6.2 |
| [7] | Y. Sheng *et al.*, "FlexGen," ICML 2023 | https://arxiv.org/abs/2303.06865 | ✓ | 〔部分〕§5 |
| [8] | J. Lin *et al.*, "KVDrive: A Holistic Multi-Tier KV Cache Management System for Long-Context LLM Inference," arXiv 2026 | https://arxiv.org/abs/2605.18071 | ✓ | 〔部分〕未來工作 p.23 |
| [9] | J. Jiang *et al.*, "Towards Efficient Large Language Model Serving: A Survey on System-Aware KV Cache Optimization," arXiv 2026 | https://arxiv.org/abs/2607.08057 | 本地 PDF | 〔部分〕p.5、p.8–9 |
| [10] | LMCache, "[RFC][MP] Lossless KV Compression with GPU Decompression" #4498（open） | https://github.com/LMCache/LMCache/issues/4498 | G | 〔全文＋4 則留言〕 |
| [11] | LMCache 文件, "L2 Serde (Serialization / Deserialization)"（MP 模式） | https://docs.lmcache.ai/mp/serde.html | W | 〔文件〕 |
| [12] | ROCm/hipCOMP-core README（v2.3.0） | https://github.com/ROCm/hipCOMP-core | G | 〔README〕 |
| [13] | A. G. Rana, B. Bao, D. Limpus, I. Chakrabarti, S. Tiwari, T. Crepaldi, A. Sirasao, "4-bit KV Caching in LMCache: Offloading Quantized KV Beyond HBM for Context-Heavy Agents on AMD MI355X," ROCm Blogs, 2026-08-28 | https://rocm.blogs.amd.com/software-tools-optimization/4bit-KV-LMcache/README.html | W | 〔部分〕 |
| [14] | Metrum AI, "What happens when compression moves to the SSD?"（2026-09，沒列作者；作者自稱 preliminary） | https://www.metrum.ai/blog/turboquant-kv-cache-compression-solidigm-ssd | W | 〔部分〕 |
| [15] | NVIDIA TensorRT-LLM, "KV Cache Compression"（cold-page NVFP4） | https://nvidia.github.io/TensorRT-LLM/1.3.0rc27/features/kv-cache-compression.html | W | 〔文件〕 |
| [16] | vLLM 0.28.0 原始碼：`vllm/v1/kv_offload/`、`…/kv_connector/v1/offloading/`、`vllm/v1/kv_cache_interface.py:387` | `/mlsteam/workspace/src/vllm-v0.28.0` | — | 〔程式碼〕 |
| [17] | Intel, intel-accel-for-llm（KV Shrink） | https://github.com/intel/intel-accel-for-llm | 見 L0 | L0 §1.2 |

### 只讀摘要
| # | 文獻 | 連結 | 核 |
|:--|:--|:--|:--|
| [18] | W. Han *et al.*, "Lynx: Progressive Speculative Quantization for accelerating KV Transfer in Long-Context Inference," arXiv 2026（Comments：原投 SIGCOMM'26；venue NOT_FOUND） | https://arxiv.org/abs/2607.01831 | ✓ |
| [19] | Y. Pengju, "SpectrumKV: Per-Token Mixed-Precision KV Cache Transfer for Prefill-Decode Disaggregated LLM Serving," arXiv 2026 | https://arxiv.org/abs/2606.08635 | ✓ |
| [20] | X. Luo *et al.*, "SmartGen: Seamless Disaggregated LLM Inference with Selective KV Cache Transfer," arXiv 2026 | https://arxiv.org/abs/2607.28150 | ✓ |
| [21] | J. Yao *et al.*, "VeriCache: Turning Lossy KV Cache into Lossless LLM Inference," arXiv 2026 | https://arxiv.org/abs/2605.17613 | ✓ |
| [22] | I. Chakrabarti *et al.*, "UltraQuant: 4-bit KV Caching for Context-Heavy Agents," EMNLP 2026 Industry Track | https://arxiv.org/abs/2606.20474 | ✓ |
| [23] | J. Zhang *et al.*, "RDKV: Rate-Distortion Bit Allocation for Joint Eviction and Quantization of the KV Cache," arXiv 2026 | https://arxiv.org/abs/2605.08317 | ✓ |
| [24] | H. Laus *et al.*, "KV Cache Compression Through the Lens of Transform Coding"（AATC）, arXiv 2026 | https://arxiv.org/abs/2608.14191 | ✓ |
| [25] | X. Zheng *et al.*, "Adaptive Multi-Objective Tiered Storage Configuration for KV Cache in LLM Service"（Kareto）, arXiv 2026 | https://arxiv.org/abs/2603.08739 | ✓ |
| [26] | S. R. Ganjihal, "Predictive Multi-Tier Memory Management for KV Cache in Large-Scale GPU Inference," arXiv 2026 | https://arxiv.org/abs/2604.26968 | ✓ |
| [27] | Y. He *et al.*, "Characterizing Predictability-Latency Trade-offs of KV-Cache SSD Offloading in LMCache for LLM Serving Systems," arXiv 2026 | https://arxiv.org/abs/2610.21249 | ✓ |
| [28] | DeepSeek-AI, "DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression," arXiv 2026 | https://arxiv.org/abs/2609.19969 | ✓ |
| [29] | H. Zhang, C. Xia, Z. Wang, "KVSwap: Disk-aware KV Cache Offloading for Long-Context On-device Inference," arXiv 2025 | https://arxiv.org/abs/2511.11907 | ✓ |
| [30] | H. Sun *et al.*, "ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference," ICML 2025 | https://arxiv.org/abs/2410.21465 | ✓ |
| [31] | S. D. Tumkur *et al.*, "Where Should the KV Cache Live? Placement Policies Across GPU, CPU, and SSD for Long-Lived Sessions," arXiv 2026 | https://arxiv.org/abs/2609.16215 | ✓（全文見 L4 §1.4） |

### 引用 L2 卡片、本檔沒有重讀的
KVCodec（L2 B5，arXiv 2602.09725）、HACK（B6，2502.03589）、KVTC（B3，2511.01815）、SplitZip（B1，2605.01708；G 確認 repo 存在）、DFloat11／ZipNN（B7）、UCCL-Zip（B10，2604.17172）、Künas 等（B10）、LMCache RFC #4149／PR #4264（B10；G 確認仍 open）。

### 查了沒找到（NOT_FOUND）
- 把 GPU 忙碌當成卸載壓縮決策輸入的論文或系統。
- 讀回時依 GPU 負載切換解壓位置（CPU／GPU）的論文或系統。
- 在 NFS／共享檔案系統層做 KV 壓縮選擇的論文。
- 依鏈路或負載選低秩 rank、或 token 合併比例的卸載系統。
- AMD／ROCm 上自適應的卸載壓縮。
- AdaptCache、EvicPress、ShadowServe 的公開程式碼連結。
- NFS 稍後錨點（500–830 MiB/s）的 run_id（本檔沒有核到）。
