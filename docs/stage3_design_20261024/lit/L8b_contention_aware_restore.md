# L8b：有沒有人用「當下量到的 GPU 爭用」來選 KV restore 計畫？

- **日期**：2026-10-24
- **執行者**：文獻 agent（Claude；只做網路搜尋與閱讀，沒有跑 GPU、沒有安裝套件，除本檔外沒有改任何 repo 檔案）
- **怎麼查的**：
  - 約 34 次 WebSearch、約 14 次 WebFetch。
  - 用 `gh` CLI 唯讀搜尋 vLLM / SGLang / LMCache / TensorRT-LLM / NVIDIA Dynamo 的 PR 與 issue，並讀 PR 內文與討論串。
  - 書目資料用 arXiv API、Crossref API、Semantic Scholar API 交叉核對（標題、第一作者、作者數、arXiv 編號、v1 日期、venue）。
  - 原文 PDF 只從 arxiv.org / usenix.org / dl.acm.org 下載到 scratchpad 的 `l8b_pdfs/`，用 `pdftotext -layout` 轉成文字後閱讀；頁碼一律是 **PDF 頁**（form feed 計數）。
  - **讀了全文或相關章節的**：
    - Cascade（arXiv 2608.06557，全文 14 頁）
    - ConCCL（arXiv 2412.14335，全文）
    - py-kvcache（arXiv 2609.11744，§1、§4–§6.6、§8–§10）
    - When the KV Cache Is Slower Than the GPU（arXiv 2610.21500，全文）
    - LithOS（arXiv 2504.15465 v1，§1–§4.7、§6）
    - Bullet（arXiv 2504.19516 v4，§1–§3.3）
    - Bidaw（USENIX FAST'26，§1–§5）
    - Krul（arXiv 2507.08045，§6）
    - Paragon（ACM PDF，§1–§2.3）
  - **只讀 HTML 局部（WebFetch）的**：NanoFlow（§4.1、§4.2.2、§6.4）、ShadowServe（§2.2、§4、§5、§8）、CacheTune（§4.3、App. C、§5.5）。
- **審查方式**：**self-review，不是 cross-model review。** 數字都回原文或 PR 內文再對一次，但沒有找獨立 reviewer。
- **證據等級**：
  - 〔全文〕讀了原文的相關段落
  - 〔部分〕只讀了部分章節，或 WebFetch 的局部 HTML
  - 〔摘要〕只讀摘要、官方頁或搜尋摘要
  - 〔PR〕GitHub PR / issue 內文或討論串（是工程聲明，不是同儕審查）
  - 〔程式碼〕GitHub 原始碼
  - 〔標題〕只確認了標題、作者、venue、DOI
- 文中「未找到」一律指「本次搜尋沒有找到」，不代表不存在。`NOT_VERIFIED` 表示沒有核實。
- 本檔**不重做**已收錄的 KVServe、HCache、ShadowServe、KVCodec、Tutti、SplitZip、CacheGen、Cake、Strata、Nexus、DuetServe、Splitwise、DistServe、Mooncake、ObjectCache、CacheFlow、InferCept、Continuum、"One Level Deeper"。對它們只補**新查到的事實**（例如 ShadowServe 對 Green Context 的評語）。
- 為了避免轉載原文，本檔一律用轉述，不直接引用原句；位置都標到 §／頁／Table／Figure。

---

## 0. 結論

**一、新穎性判定：整個組合沒有被做過；但每一個維度都有局部前例。**

- 本次搜尋**沒有找到**任何工作同時做到下面兩件事：
  - 把**當下量到的 GPU 爭用**（而且是分資源維度的：SM／tensor core、HBM、copy engine／PCIe）當成輸入；
  - 據此**聯合選擇** KV restore 計畫：recompute／load 怎麼切、儲存格式走哪條解碼路徑、走 copy engine 還是 SM gather、restore 分到多少 SM、多個並行 restore 的先後順序。
- 也**沒有找到**任何工作報告我們量到的非單調現象：最快的 restore 格式在 **compute 型背景下翻轉**，在 **memory-bandwidth 型背景下不翻轉**。
- 但下列單點都**已經有人做過**，主張時必須避開：
  - 「依 SLO 預算決定 restore 多少、其餘 recompute」→ Cascade 做過。
  - 「依 KV 大小排 restore 讀取順序」→ Bidaw 做過。
  - 「依 payload 大小在 copy engine 與 SM kernel 之間選路」→ vLLM PR #42212 做過。
  - 「離線量 break-even，在 lookup 時決定 load 或 recompute」→ py-kvcache、CacheTune、Krul 都做過。
- 結論：主張必須寫成「**restore 成本不是硬體常數，而是隨當下爭用類型改變的資源向量；最佳計畫會翻轉**」。不能寫成「第一個做 load／recompute 切分」或「第一個做 SLO-aware restore」。

**二、三個最大威脅（ID 對應 §1 對照表與 §2 卡片）**

- **T1　Cascade**（arXiv 2608.06557，2026-08-06，UBC＋Microsoft Azure Research＋NVIDIA）——**部分重疊**
  - 重疊之處：
    - 用同一個 per-request latency budget 計算「最多可以從第 k 層 tier 搬回多少 bytes」，超過的部分改成 GPU recompute（§IV-C，Eq. 6–7，p7）。
    - 用 least-remaining-budget-first 排序（Algorithm 1，p7）。
  - 不重疊之處：
    - restore 成本只有「每條 link 的固定 overhead＋bytes／有效頻寬」兩項。
    - TTFT 預測只拿 active decode 請求數當負載（§IV-B，p6）。
    - 爭用被塞進一個**離線固定**的保護係數 γ（Eq. 3，p6）。
    - 沒有儲存格式、沒有解碼路徑、沒有 SM 配額。
    - 主要結果來自擴充後的 Vidur 模擬器（§V-A，p8）。
- **T2　Bidaw**（USENIX FAST'26）——**部分重疊**
  - 重疊之處：
    - 依「等待時間／KV 大小」的 HRRN 比值排序第二層 tier 的 KV 讀取（§3.2，Eq. 1，p7）。
    - 把 storage-efficient tensor 轉回 KV 的運算丟到 low-priority CUDA stream，理由是作者觀察到超過三成 SM 閒置（§4，p10）。
  - 不重疊之處：「SM 有空閒」是**靜態假設**，不是即時量測。這正是我們在 busy GPU 下量到會失效的情境（low-rank 重建約慢 2.8 倍）。
- **T3　vLLM PR #42212**（2026-06-03 merged）——**部分重疊（工程層）**
  - 重疊之處：CPU→GPU onload 依 payload 大小在 `cuMemcpyBatchAsync`（copy engine）與 Triton SM gather kernel 之間選路：descriptor 數 ≥16 且最大 payload <28 KiB 時走 kernel。
  - kernel 固定用 12 個 SM。PR 作者的理由是：12／16／20 SM 的端到端 req/s 相同，所以選最小的切片以減少對 decode 的爭用。
  - 討論串中另一位 contributor 指出，他之所以放棄 SM 型 copy，是因為會和並行 kernel 搶 SM 與 L1/L2（PR 討論）。
  - 不重疊之處：門檻與 SM 數都是靜態常數，不看當下爭用。
- **次要威脅**
  - **py-kvcache**（arXiv 2609.11744）把「持續估計 prefill 時間、tier 頻寬、佇列延遲並更新 load／defer 決策」以及「爭用」明列為 future work（§10.2，p22）→ **有被搶先的風險**。
  - **SGLang PR #45945**（2026-10-23，open）提供依 link 暫停／恢復背景傳輸的 runtime 機制，但只做 pause，不選計畫。
  - **KVServe**（已收錄）仍是「依 runtime 狀態選壓縮」最近的前例；GPU 負載只以 EWMA 殘差吸收。

**三、Q2（空間共享）：沒有任何一篇把 KV load／restore／dequant 放進 SM 分配決策。**

- 本次查到的空間共享工作都**沒有**這樣做：LithOS、Orion、REEF、Paella、MuxServe、semi-PD、Bullet、MuxWise（SGLang PDMux）、NanoFlow、POD-Attention、RAPID-Serve、Hummingbird。
- 最接近的兩個：
  - **NanoFlow**：有 KV offload／load，但不在它的資源／干擾模型裡；只量到 offload 讓 pipeline 慢 3.0%（§4.2.2、§6.4）〔部分〕。
  - **Tutti**（已收錄）：把 I/O kernel 放在獨立的 green context。
- SGLang PDMux 的 green context 只切 prefill／decode 兩條 stream〔程式碼〕。
- vLLM、TensorRT-LLM、NVIDIA Dynamo：本次未找到 green-context PR。

**四、Q3（copy engine vs SM copy）：證據方向一致，但沒有人依即時爭用選路。**

- **SM copy 會和計算搶資源**：搶 SM 與 L1/L2（vLLM #42212 討論、SGLang #44029、py-kvcache §5.1 p11、ConCCL §VI-A p7、NCCL 2.28 CE collectives）。
- **copy engine 不搶 SM，但有自己的問題**：
  - 仍會搶 HBM 頻寬（ConCCL §VII-A p8）。
  - 小 transfer 有 setup／descriptor 懸崖（vLLM #42212；ConCCL §VI-C p7：<32 MB 時最多慢 4×）。
  - 有 copy-engine queue 仲裁的阻塞問題（SGLang #38449）。
- **現行引擎全是靜態選路**：
  - vLLM：28 KiB 門檻＋12 SM。
  - SGLang：`--hicache-io-backend` 預設 `kernel`。
  - LMCache：解壓縮位置在設定時就決定（RFC #4498）。
- 這就是古典 **PIO vs DMA**（arXiv 2409.08141：約 1 KiB 以下 PIO posted write 勝過 DMA〔摘要〕），以及 **CPU memcpy vs I/OAT／DSA copy offload** 的 GPU 版本。

**五、Q4（古典骨幹）：建議用 Bubble-Up／Paragon 的「敏感度 × 壓力源」框架。**

- **Bubble-Up**（MICRO'11）：用可調壓力的「bubble」量 sensitivity。
- **Paragon**（ASPLOS'13）：10 個壓力源（SoI）各配一個 microbenchmark，再用 SVD＋PQ 協同過濾分類，§2.3。
- 兩者最適合拿來當「每種 restore 格式 × 每種資源壓力」的敏感度矩陣骨幹。
- GPU 端可直接比較的模型形式：
  - Bullet SRM：roofline × SM 比例，再用 memcpy kernel 共跑校準爭用。
  - ConCCL：roofline × CU-loss 查表。
  - HSM：白箱 DRAM 模型＋線性回歸。
  - NanoFlow：資源占比 → 效能的配對查表。
  - LithOS：線性 TPC 外插＋線上修正。
- 「格式在 compute 背景翻轉、在 memory 背景不翻轉」，正好對應 Paragon「不同 SoI 的敏感度不同」的概念。

**六、Q5（排程／降級）：有 size／slack-aware 的 restore 排序，但沒有依爭用改格式的 brownout。**

- 已有的 restore 排序：Bidaw disk-HRRN、Cascade least-remaining-budget-first、Nuska（近似 LLF）、TokenFlow（依 buffer 佔用排優先序）。
- 已有的負載觸發品質降級：MorphServe（高負載時換成量化層）、OrbitFlow（重負載時延後大請求）。
- **未找到**：依當下爭用改 restore 的格式或精度（Brownout 式降級）。

**七、剩下的缺口（一句話）**

沒有工作把 restore 成本建模成「隨當下 SM／tensor、HBM、copy-engine／PCIe 爭用而變的向量」，並據此在 recompute／load 切分、格式解碼路徑、copy-engine vs SM 路徑、SM 配額與多個 restore 的順序之間做**聯合**、**線上**選擇。

**八、審稿人會要求的 baseline（細節見 §7）**

1. 每台硬體事後最佳的固定格式
2. Cake（bidirectional，本身就會隨 runtime 速度移動交會點）
3. Cascade 式預算切分（Eq. 6–7）
4. py-kvcache／CacheTune 式離線 break-even 或校準
5. KVServe 式 EWMA 殘差修正
6. vLLM 靜態 28 KiB／12 SM 選路，以及 SGLang `kernel`／`direct`
7. 靜態 green-context／MPS 預留 SM（Tutti 式）
8. FCFS vs Bidaw HRRN vs Cascade LRBF 的 restore 排序
9. 純 utilization 純量模型 vs 我們的資源向量模型（Bullet SRM／ConCCL 查表形式）

---

## 1. 對照表（Q1：用當下爭用選 restore 計畫）

| ID | 論文／產物 | 年份／venue | 做了什麼 | 重疊 | 理由（附位置） | 深度 |
|---|---|---|---|---|---|---|
| T1 | Cascade [1] | 2026-08，arXiv（venue NOT_VERIFIED） | per-request latency budget 同時決定排程順序，以及 deep-tier KV 要 restore 還是 recompute | **部分重疊** | 有 recompute／load 切分（Eq. 7，§IV-C，p7）與 slack 排序（Alg. 1，p7）；但成本只看 link 頻寬（Eq. 6），負載只看 active decode 數（§IV-B，p6），爭用由離線固定的 γ 吸收（Eq. 3，p6）；沒有格式、路徑、SM 維度 | 〔全文〕 |
| T2 | Bidaw [2] | FAST'26 | 依 I/O 感知排 KV 讀取順序；把 history tensor 轉 KV 的工作丟到 low-priority stream | **部分重疊** | HRRN 排序（§3.2，Eq. 1，p7）；轉換假設 SM 有空閒（§4，p10）；沒有爭用量測 | 〔全文〕 |
| T3 | vLLM PR #42212 [3] | 2026-06 merged | onload 依 payload 大小選 copy engine 或 Triton SM kernel | **部分重疊（工程）** | 靜態門檻（28 KiB、n≥16、12 SM）；SM 數依端到端掃描與「少搶 decode」的理由選定 | 〔PR〕 |
| T4 | py-kvcache [4] | 2026-09，arXiv | 離線量各 tier 的 break-even prefix 長度，在 lookup 時決定 load 或 recompute | 相關但不同（有被搶先風險） | 服務期間不量也不擬合（§6.6，p17）；動態估計與爭用列為 future work（§10.2，p22） | 〔全文〕 |
| T5 | When KV Is Slower Than GPU [5] | 2026-10，arXiv | 兩項成本模型 T_load = T0 + ns/B 對比 T_recompute = n/R | 相關但不同 | R 取 replay 中 cold miss 速率的中位數（§4，p3）；多 engine 的爭用是估計的，沒有實測（§8，p8） | 〔全文〕 |
| — | KVServe（已收錄） | SIGCOMM'26 | 依 runtime 狀態選壓縮 | 部分重疊 | 已在 repo；GPU 負載只當 EWMA 殘差 | — |
| — | Krul [9] | 2025-07，arXiv | recompute／load pipeline；以 min\|T_C − T_L\| 求 recompute 比例 r_c | 相關但不同 | r_c 由 F_peak／B_peak 或校準資料集求得（§6.1，p8），靜態 | 〔部分〕 |
| — | CacheTune [10] | 2026-05（v2 10 月），arXiv | 為異質 tier 調 recompute 比例 | 相關但不同 | 部署時離線 profiling＋校準集上做 golden-section search（§4.3、App. C）；文中沒有 GPU 負載訊號 | 〔部分〕 |
| — | KVPR [11] | ACL Findings 2025 | partial recompute＋傳輸；profiler＋scheduler | 相關但不同 | 摘要提到 profiler 收集硬體資訊；是否線上 NOT_VERIFIED（ACL PDF 連線被拒） | 〔摘要〕 |
| — | CacheBlend [60] | 2024，arXiv（venue NOT_VERIFIED） | 選擇性重算，並與 KV 取回 pipeline 化 | 相關但不同 | 摘要：把重算延遲藏在取回延遲裡；沒有提到爭用 | 〔摘要〕 |
| — | NanoFlow [12] | OSDI'25 | 裝置內 nano-batch 重疊；有 KV offload／load | 相關但不同 | offload／load 不在資源模型中（§4.1.1、§4.2.2）；只量到 3.0% 減速（§6.4） | 〔部分〕 |
| — | ShadowServe（已收錄）新事實 [13] | arXiv 2509.16857 | SmartNIC 解壓 | 相關但不同 | 設計靜態：核心分配固定，FIFO 佇列（§4.1、§4.2、§5）；認為 Green Context 表現差、不能切記憶體頻寬（§2.2，Fig. 3 說明） | 〔部分〕 |
| — | SGLang PR #45945 [14] | 2026-10，open | 依 link（NVLink／PCIe H2D／D2H）暫停或恢復背景傳輸的 gate | 相關但不同（機制） | 只有 pause／resume，不選計畫；PR 說明 runtime 目前尚未使用 | 〔PR〕 |
| — | LMCache RFC #4498 [19] | 2026-08，open | 無損壓縮＋GPU 解壓（nvCOMP／hipCOMP） | 相關但不同 | 承認 GPU 解壓會干擾推論；選擇在設定時決定 | 〔PR〕 |
| — | Bullet [8]／MuxWise [47]／semi-PD [46] | 2025，arXiv | prefill／decode 空間共享＋即時效能模型 | 相關但不同 | 有爭用模型，但對象不含 KV restore（Bullet §3.2.3，p6） | 〔全文〕／〔摘要〕 |
| — | ConServe [59] | 2024，arXiv | online／offline 共同服務；layer-wise preemption＋incremental KV | 相關但不同 | restore 指的是被搶占的 offline 工作；不選格式或路徑 | 〔摘要〕 |

**判定說明**
- **完全重疊**：本次沒有找到任何一篇。
- **部分重疊**：做了「計畫選擇」的某一軸（切分、排序或選路），但輸入不是即時、多維的 GPU 爭用。

---

## 2. 主要威脅卡片

### T1　Cascade [1]〔全文〕

- **書目**
  - Adnan, Mahapatra, Nair, Berger, Zardoshti, Fonseca, Choukse，共 7 人。
  - arXiv 2608.06557 v1，2026-08-06；只有 v1；沒有 Comments 欄 → **venue NOT_VERIFIED**。
- **場景**：PD co-located serving（§II-B，p3），HBM → DRAM → NVMe 三層 tier（§IV-A，p5–6）。
- **硬體**
  - profile 來自 GB200 NVL72（MNNVL、NVLink-C2C、GDS over PCIe Gen5，Fig. 3，p3）。
  - TP=4；權重 NVFP4、KV FP8（Table I，p8）。
- **實作與評估方式**
  - 實作在 vLLM＋LMCache 上（§IV-D，p7）。
  - 但叢集規模的結果來自擴充後的 Vidur 模擬器，模擬器用 GB200 實測的資料搬移 profile 填參數（§V-A，p8）。
  - → **主要結果不是端到端實機**。
- **工作負載**
  - Aliyun Bailian trace：ChatBot、Tool&Agent、Coder、Reasoning，組成 10 種混合（Table II，p8）。
  - 模型：Qwen-2.5-72B（以 MHA 設定）、Llama-3-70B、Llama-3-405B。
- **Baselines**：FCFS（vLLM 預設）、EDF、SJF（§V-A d，p9）。**沒有** Cake、CacheBlend、HCache 這類 restore baseline。
- **指標**：goodput（TTFT p90 與 TPOT p90 都達標的 output tokens/s）、違規率、Jain fairness（Eq. 1，p4）。
- **關鍵數字**
  - 相對 FCFS：goodput 最高 2.4×，違規少 40%（Fig. 8，§V-B1，p9）。
  - Qwen 的違規數從 1.5×10⁵ 降到 4×10³（Fig. 11，§V-B2，p10）。
  - 7 個實例的表現就超過 FCFS 用 9 個實例，即少用 22% GPU（Fig. 13，§V-D1，p11）。
- **機制**
  - TTFT 估計器在四種情境下預測 prefill 時間：零負載、實際負載、實際負載加 tier-aware 命中、最大負載。輸入是 context 長度、各 tier 命中分布、active decode 數，預測模型離線訓練（§IV-B，p6）。
  - 取「實際負載」與「最大負載」兩個估計的平均再乘上 γ（Eq. 3）。γ 每個模型一個值，離線固定，不隨 trace 或負載改變；作者說它吸收估計誤差、記憶體匯流排爭用和佇列變異（§IV-B，p6）。
  - restore 時間 T_k(x)：路徑上每條 link 的固定 overhead 加總，再加 x 乘以各 link 有效頻寬倒數的總和（Eq. 6，p7）。
  - 可安全搬回的 bytes 上限 M_{r,k} 由剩餘預算反推（Eq. 7）；超過的部分 fallback 成 recompute（§IV-C，p7）。
  - HBM 不夠時，搶占剩餘預算最大的請求（Alg. 1）。
- **它沒有做的**（＝我們的空間）
  1. **recompute 成本不受背景負載影響**：只用 active decode 數，沒有量 SM／tensor 爭用。我們量到 busy 時 recompute 約慢 2×，Cascade 的切分點會因此偏掉。
  2. **沒有格式**：沒有壓縮格式、dequant、低秩重建，因此不可能出現「格式翻轉」。
  3. **只有頻寬**：沒有 copy engine vs SM 路徑，也沒有 SM 配額。
  4. **切分是「先 restore 到預算上限，剩下 recompute」**，不是兩邊並行的最佳交會點（沒有引用 Cake；grep 全文找不到 Cake、compress、dequant 字樣）。
  5. **評估以模擬為主**。

### T2　Bidaw [2]〔全文〕

- **書目**
  - Shipeng Hu, Guangyan Zhang, Yuqi Zhou, Yaya Wei, Ziyan Zhong, Jike Chen，共 6 人。
  - 收錄於 USENIX FAST'26（2026-02-24～26，Santa Clara；p1 封面）。
- **場景**：多輪互動；KV 放在 host memory（performance layer）與 SSD（capacity layer）兩層。
- **硬體**：1× A800 80 GB、PCIe Gen4、SSD 為 4 顆 SATA SSD 組成的 RAID-5，頻寬 1.5 GB/s（§5，p11）。
- **模型**：OPT-6.7B／13B／30B、Qwen-7B／14B（§5，p11）。
- **Baselines**：vLLM，以及作者在 vLLM 上重新實作的 CachedAttention 與 FlashGen（§5，p11）。
- **關鍵數字**：相對既有方法，回應延遲最多降 3.58×、吞吐最多升 1.83×（摘要與 §1，p2–3；§5.1，p11）。
- **機制**
  - **雙佇列**：KV 已在 performance layer 的請求進 ready queue，否則進 preparing queue（§3.2，p7）。
  - **disk-HRRN**：response ratio = 1 + 等待時間／KV 大小，依此決定向 capacity layer 發 read 的先後（Eq. 1，p7）。這是 size-aware 加 aging 的 restore 排序。
  - **storage-efficient tensor caching**：快取比 KV 小的中間 tensor，載回 GPU 後再轉成 KV。轉換放在獨立的 low-priority CUDA stream，與正常推論平行，理由是觀察到超過三成 SM 閒置（§4，p10）。作者也指出 GQA 模型不適用（§4，p10）。
- **它沒有做的**
  - 轉換放在 low-priority stream 的前提（SM 有空）是**固定觀察，不是即時量測**。GPU 忙時，這個轉換（性質類似我們的 low-rank 重建）會變慢；我們量到約 2.8×。
  - 排序只看 KV 大小與等待時間，不看各 restore 會消耗哪一種資源。
  - 沒有 recompute／load 切分，也沒有格式選擇。

### T3　vLLM PR #42212 [3]〔PR〕

- **基本資訊**：作者 Etelis；2026-05-10 開，2026-06-03 merged。
- **內容**
  - `OffloadingConnector` 原本用 `cuMemcpyBatchAsync` 做 host↔device copy。
  - PR 加了一個 Triton kernel（`_swap_blocks_kernel`），在 batch ≥16 且最大 payload <28 KiB 時接手 **CPU→GPU** 方向；**GPU→CPU** 方向維持 DMA。
- **PR 內文的量測**（H100；PR 自報，未經同儕審查）
  - DMA 對 4–24 KiB 小 descriptor 的吞吐平坦在約 5–7 GB/s，到 28 KiB 跳約 6×。
  - 28 KiB 是 DMA 與 Triton 的交叉點，SM 數 ≥12 時都一樣。
  - 12／16／20 SM 的端到端 req/s 相同，因此選最小切片以降低對 decode 的爭用。
  - gpt-oss-120b TP=2：相對 vanilla DMA，req/s +105–154%；onload 實際頻寬從 4.8 提升到 37 GB/s。
- **討論串**
  - 另一位 contributor 說，他放棄 SM 型 copy 是因為會和並行 kernel 搶 SM 與 L1/L2，而且 cudaMemcpy 的 CPU 開銷放到背景 thread 就可以接受。
  - 他也追問在沒有 cache hit 的 workload 下 SM 干擾會不會更明顯。PR 裡**沒有**對這個問題的量測回覆。
- **相關 PR**
  - ROCm 版：**#52700**（2026-08-18 開，open），用 mapped device memory 恢復 KV-offload 的 Triton fast path（只讀到標題與檔案清單）。
  - **#43018**（2026-08-21 merged）：ROCm 7.15 上 `hipMemcpyBatchAsync` 比逐筆 `hipMemcpyAsync` 快 6–19×（MI350X）；count >8192 時會 fault，因此加了分批的環境變數。
- **它沒有做的**：28 KiB、n≥16、12 SM 都是常數。選路**不看**當下 GPU 是不是在跑 compute-bound 或 memory-bound 的工作。
  - 我們的量測顯示：最佳 restore 路徑會隨背景類型翻轉。因此這是可以直接拿來當 baseline 的靜態規則。

### T4　py-kvcache [4]〔全文〕（被搶先風險）

- **書目與環境**
  - Kanichai, De Matteis, Trivedi，共 3 人；VU Amsterdam＋IBM Research Zurich；arXiv 2609.11744 v1，2026-09-10；venue NOT_VERIFIED。
  - 兩台機器：一台 RTX 4000 Ada 本地節點，一台 Snellius 超級電腦上的 H100（§8，p21）。
- **break-even gate**
  - 每個節點、每個模型跑一次離線 Pareto 程序，得到「每個 source tier 的 break-even prefix 長度」。
  - 服務期間不量測也不擬合；lookup 時若預測 load 比 recompute 貴，就放棄 load（§6.6，p17）。
- **作者自己點出的缺口**
  - 只看 prefix 長度與裝置頻寬的靜態門檻，分不出「需求讀取」和「可以被佇列藏住的預載」（§8，p21）。
  - future work 寫明要做：持續估計 prefill 時間、tier 頻寬、佇列延遲、promotion 成功率，並動態更新 store／load／defer 決策；也要在爭用下評估 tail TTFT（§10.2，p22）。
- **對我們的意義**：這兩點與我們的方向相鄰，必須**明確引用並區分**。他們的動態化是針對 tier／佇列；我們是針對 **GPU 資源向量**以及**格式／路徑翻轉**。
- **新事實（Q3 用）**：LMCache 用 CUDA kernel 做 copy，會吃 GPU 計算資源（§5.1，p11）。

### T5　When the KV Cache Is Slower Than the GPU [5]〔全文〕

- **書目**：Muhammad Ibrahim（Multiverse Computing），1 位作者；arXiv 2610.21500 v1，2026-10-15；Comments 只寫頁數 → venue NOT_VERIFIED。
- **場景**：GLM-5.3（FP8 MoE＋MLA）跑在 8×B300 上，用 PCP8；LMCache＋Valkey remote tier。
- **模型**（§4，p3）
  - T_recompute = n/R，T_load = T0 + ns/B。
  - break-even 頻寬 B* = sR ≈ 2.53 GB/s。
  - T0 ≈ 0.33 s，只用一個點擬合。
  - R 取 replay 中 cold miss 速率的中位數。
- **限制**（§8，p8）
  - 每個條件只跑一次。
  - 多 engine 的爭用是估計，沒有實測。
  - 文中提到 GPU 只有 7% busy 時佇列就已經不穩（§6.3）。
- **對我們的意義**
  - 這是「restore 成本當成硬體常數」的典型例子，可在 introduction 當**反例**引用。
  - 他們的 R 在 GPU 忙碌時會變，但模型沒有表達這件事。

---

## 3. Q2：空間共享／SM 切割，有沒有把 KV restore 放進決策？

| 工作 | venue（核實狀態） | 共享機制 | 決策中有沒有 KV load／restore／dequant？ | 依據 |
|---|---|---|---|---|
| LithOS [7] | SOSP'25（Crossref） | TPC 級排程、kernel atomization、TPC stealing；建在 MPS 上；線上 kernel 延遲預測 | **沒有**。全文找不到 memcpy／DMA／KV 字樣；只自行實作 cuLaunchKernel 等少數 API，其餘自動產生（§4.7 p8；§5 p9）；testbed 是 1×A100（§6，p9） | 〔全文〕 |
| Orion [39] | EuroSys'24（Crossref） | 攔截 kernel launch，依 operator 的 compute／memory 需求排程 | 摘要沒有 | 〔摘要〕 |
| REEF [40] | OSDI'22（USENIX 頁） | 微秒級 kernel preemption＋受控並行 | 摘要沒有 | 〔摘要〕 |
| Paella [41] | SOSP'23（Crossref） | software-defined GPU scheduling | 未查到相關內容 | 〔標題〕 |
| MuxServe [45] | arXiv 2404.02015（venue NOT_VERIFIED） | 多 LLM 空間-時間 multiplexing（統一資源管理） | 摘要沒有（KV 只談記憶體 multiplexing） | 〔摘要〕 |
| semi-PD [46] | arXiv 2504.19867（venue NOT_VERIFIED） | SM 級分離計算＋統一儲存；SLO-aware 動態切割 | 摘要沒有（目的是「避免」KV 傳輸） | 〔摘要〕 |
| Bullet [8] | arXiv 2504.19516（venue NOT_VERIFIED） | libsmctrl stream mask＋MPS；SM-scaling roofline 模型＋爭用校準 | **沒有**。爭用模型的對象是 prefill／decode kernel；用 memcpy kernel 只是為了量最壞情況的記憶體干擾（§3.2.3，p6） | 〔全文〕 |
| MuxWise／SGLang PDMux [47][18] | arXiv 2504.14489（venue NOT_VERIFIED） | green context 切 prefill／decode stream；contention-tolerant 估計器 | **沒有**。`pdmux_context.py` 只建 (prefill_sm, decode_sm) 的 stream group | 〔摘要〕〔程式碼〕 |
| NanoFlow [12] | OSDI'25（USENIX 頁，pp. 749–765 來自搜尋摘要） | nano-batch＋資源占比 R → 效能 P 的配對查表 | **有 offload／load，但不在模型內**（§4.1.1、§4.2.2）；offload 讓 pipeline 慢 3.0%（§6.4） | 〔部分〕 |
| POD-Attention [48] | ASPLOS'25（Crossref／arXiv Comments） | 把 prefill 與 decode attention 融合成單一 kernel | 未查到 | 〔標題〕 |
| RAPID-Serve [49] | arXiv 2601.11822（venue NOT_VERIFIED） | AMD CU masking 做 runtime 計算資源分配 | 摘要沒有 | 〔摘要〕 |
| Hummingbird [50] | arXiv 2601.04071 | 閉源 GPU 上的微秒級 preemption | 摘要沒有 | 〔摘要〕 |
| ConCCL [6] | ISPASS'25（Crossref） | CU 分割＋排程優先序；把通訊卸到 DMA | 對象是 collective，**不是 KV** | 〔全文〕 |
| Tutti（已收錄） | — | I/O kernel 放在獨立 green context | 有（已收錄，不重述） | — |
| SGLang PR #22827 [18] | closed PoC（2026-04） | 用 green context 讓 overlap schedule stream 不被 persistent GEMM 卡住 | 沒有 | 〔PR〕 |
| vLLM／TensorRT-LLM／Dynamo | — | — | 本次 `gh search` 用 "green context"、"greenctx"、"SM partition" 都沒找到相關 PR | 〔PR〕 |

**新事實**
- ShadowServe（已收錄）認為 Green Context 是實驗功能、表現差，而且不能切記憶體頻寬（§2.2，Fig. 3 說明）。
- 它也指出 MPS 與 Green Context 的效果不確定，且依 workload 而定，因為 GPU scheduler 是閉源的（§8）〔部分〕。
- → 我們若用 green context／MPS 當「SM 配額」這個控制旋鈕，必須自己量它的實際隔離度，不能假設它能隔離 HBM。

---

## 4. Q3：copy engine vs SM copy 在爭用下的證據

### 4.1 SM 型 copy 會和計算搶資源

| 證據 | 內容 | 位置／深度 |
|---|---|---|
| vLLM #42212 討論串 | contributor 放棄 SM 型 copy，原因是會搶 SM 與 L1/L2、拖慢計算；PR 作者因此選最小的 12 SM 以減少對 decode 的爭用 | 〔PR〕 |
| SGLang #44029（2026-10-13，open） | 和 attention／GEMM 重疊時，NCCL ring／LL allgather kernel 會搶 SM 與 HBM 頻寬；改用 `NCCL_CTA_POLICY_ZERO` 讓 copy engine 做 allgather | 〔PR〕 |
| 同上，PR 表格 | 2 GPU：0.0625 MiB 時 zero-SM 反而較慢（0.71×），64 MiB 時快 1.53×；與計算重疊時 64 MiB 快 1.75× | 〔PR〕 |
| py-kvcache §5.1 | LMCache 用 CUDA kernel 做 copy，會吃 GPU 計算資源 | p11〔全文〕 |
| ConCCL §I、§VI-A | MI300X 上計算與通訊並行只拿到理想加速的 21%，原因是共用 CU、cache、HBM；把通訊卸到 DMA 後 CU 全部留給 GEMM，而且 DMA 在 IOD、繞過 L1/L2 | p1–2、p7〔全文〕 |
| NCCL 2.28 [21][22] | CE collectives 把 NVLink 域內的 AlltoAll／Gather／Scatter 從 SM 卸到 copy engine，達到 zero-SM；為 opt-in，需要 symmetric window 與 `NCCL_CTA_POLICY_ZERO` | 〔摘要〕 |
| LMCache RFC #4498 | GPU 解壓吃計算與頻寬，只有在省下的傳輸時間大於成本**加上**對推論的干擾時才划算；store 路徑預設 CPU 壓縮以避開 GPU 爭用 | 〔PR〕 |

### 4.2 copy engine 也有自己的爭用與門檻

| 證據 | 內容 | 位置／深度 |
|---|---|---|
| ConCCL §VII-A | DMA 不碰 L1/L2，但 **HBM 頻寬爭用仍在**；作者把記憶體頻寬分割留作 future work | p8〔全文〕 |
| ConCCL §VI-C | DMA collective 在 <32 MB 時最多慢 4×，原因是 CPU 端發動與同步的成本無法攤平 | p7〔全文〕 |
| vLLM #42212 | 4–24 KiB 小 descriptor 的 DMA onload 平坦在約 5–7 GB/s；descriptor 越多，懸崖越深 | 〔PR〕 |
| SGLang #38449（2026-09-08，open） | 8×H100 TP=8：HiCache load burst 把同一 copy engine 上的 round-head H2D 小 copy 排在後面，80 個 prefill round 中有 27 個卡住；改成在 burst 之前先發 head copy 後，0／98 | 〔PR〕 |
| SGLang #45945（2026-10-23，open） | 背景大量傳輸會拖慢共用同一條 link 的延遲關鍵操作，而 GPU 上已在跑的工作無法被搶占 → 用 `cuStreamWaitValue32` 做 gate；H200 上每個 chunk 多約 2.0 µs | 〔PR〕 |
| DMA-Latte [23] | MICRO 2026；把 DMA offload 推到 KB 到低 MB 的延遲敏感區；相對 baseline DMA offload 最多 4.5× | 〔摘要〕 |

### 4.3 現行引擎的選擇方式：全是靜態

- **vLLM**：靠 size 門檻（#42212）。ROCm 的批次 API 直到 7.14／7.15 才真正有批次化效果（#43018）。
- **SGLang**：`--hicache-io-backend` 可選 `direct` / `kernel` / `kernel_ascend`，預設 `kernel`〔程式碼〕。官方 HiCache 文件只說 kernel 路徑傳輸較快（最多 3×），**沒有**討論對計算的干擾〔官方文件〕。
- **LMCache**：解壓位置經由 serde 設定決定（RFC #4498）。

### 4.4 古典類比

- **PIO vs DMA**
  - arXiv 2409.08141（Ruzhanskaia 等 4 人，venue NOT_VERIFIED）比較了三種方式：descriptor-ring DMA、PCIe 上的 PIO、cache-coherent PIO。
  - 約 1 KiB 以下，PIO posted write 勝過 DMA；原因是 descriptor 設計用小傳輸的延遲換大傳輸的吞吐〔摘要〕。
  - → 這和 vLLM 的 28 KiB 交叉點是同一個結構：SM gather 相當於 PIO，copy engine 相當於 DMA。
  - 差別在 GPU 上「PIO」吃的是和推論共用的 SM，所以交叉點**會隨爭用移動**。這正是我們可以補上的部分。
- **CPU memcpy vs 硬體 copy engine**
  - I/OAT：IPDPS'07，Vaidyanathan 等 4 人〔標題〕。
  - Intel DSA：ASPLOS'24 pp. 37–54〔摘要〕。arXiv 版 7 位作者、ACM 版 10 位作者，作者數以 DOI 版為準。
- **GPU thread 直接做 I/O 的路線**：EMOGI（PVLDB 2020）、PyTorch-Direct（PVLDB 2021）、BaM（ASPLOS'23）〔標題〕。本次沒有查證它們的爭用量測，只能列為「SM 驅動 I/O」的設計前例。

---

## 5. Q4：古典干擾／成本模型（書目已核實＋模型形式）

| 工作 | 書目（核實來源） | 模型形式（一行） | 深度 |
|---|---|---|---|
| Bubble-Up [31] | J. Mars 等 5 人，MICRO-44，2011，pp. 248–259，doi 10.1145/2155620.2155650（Crossref） | 用可調壓力的「bubble」施壓記憶體子系統，量應用的敏感度與造成的壓力，再預測共置後的劣化（自報誤差 1–2%） | 〔摘要〕（Google Research 頁） |
| Bubble-Flux [32] | H. Yang 等 4 人，ISCA'13，pp. 607–618，doi 10.1145/2485922.2485974（Crossref） | 模型形式 NOT_VERIFIED | 〔標題〕 |
| Paragon [33] | C. Delimitrou、C. Kozyrakis，ASPLOS'13，pp. 77–88，doi 10.1145/2451116.2451125（Crossref） | 10 個壓力源各配一個可調 microbenchmark，量「能容忍／會造成」的干擾分數；新應用只跑兩次 1 分鐘的 microbenchmark，再用 SVD＋PQ 重建補齊整列（§2.3，p4）；平均誤差 5.3%（Table 2） | 〔全文〕 |
| Quasar [34] | C. Delimitrou、C. Kozyrakis，ASPLOS'14，pp. 127–144，doi 10.1145/2541940.2541941（Crossref） | 使用者給效能目標而非資源保留；以過去應用的資訊做分類，再自動選資源量與種類 | 〔部分〕（作者投影片＋二手） |
| HSM [35] | X. Zhao、M. Jahre、L. Eeckhout，ASPLOS'20，pp. 1371–1385，doi 10.1145/3373376.3378457（Crossref） | 白箱（有效頻寬對 DRAM row-buffer hit rate）＋黑箱線性回歸；平均誤差 6.8% | 〔摘要〕（搜尋摘要） |
| Themis [36] | W. Zhao 等 9 人，IPDPS'19，pp. 653–663（Crossref） | 預測空間多工 GPU 上的應用級 slowdown；形式 NOT_VERIFIED | 〔標題〕 |
| Baymax [37] | Q. Chen 等 4 人，ASPLOS'16，pp. 681–696（Crossref） | 形式 NOT_VERIFIED | 〔標題〕 |
| Prophet [38] | Q. Chen 等 6 人，ASPLOS'17，pp. 17–32（Crossref） | 非搶占加速器上的 QoS 預測；形式 NOT_VERIFIED | 〔標題〕 |
| Orion [39] | F. Strati 等 3 人，EuroSys'24，pp. 1075–1092（Crossref） | 依 operator 的 compute／memory 需求避免干擾 | 〔摘要〕 |
| iGniter [43] | F. Xu 等 7 人，IEEE TPDS，2023，pp. 812–827（Crossref） | 輕量推論效能模型，用實務上取得到的系統與 workload 指標捕捉干擾 | 〔摘要〕 |
| USHER [42] | S. S. Shubha、H. Shen、A. Iyer，OSDI'24（USENIX 頁） | kernel 級資源（compute／memory）估計器＋干擾感知排程 | 〔摘要〕 |
| Abacus [44] | W. Cui 等 11 人，SC'21，pp. 1–15，doi 10.1145/3458817.3476143（Crossref） | 重疊 operator 的精確延遲預測；形式 NOT_VERIFIED | 〔標題〕 |
| LithOS [7] | P. H. Coppock 等 9 人，SOSP'25，pp. 1–17，doi 10.1145/3731569.3764818（Crossref） | 線上依 operator 記錄延遲；缺資料時保守假設對 TPC 數線性縮放（§4.7，p8–9） | 〔全文〕 |
| Bullet SRM [8] | arXiv 2504.19516（6 人；venue NOT_VERIFIED） | roofline 的峰值算力與頻寬按 SM 比例縮放，在飽和點封頂；再用「memcpy kernel 與 GEMM 共跑」量最壞記憶體干擾；兩個樣本就能校準（§3.2.2–3.2.4，p5–6） | 〔全文〕 |
| ConCCL 啟發式 [6] | ISPASS'25，pp. 1–11，doi 10.1109/ISPASS64960.2025.00018（Crossref） | roofline 時間乘上「失去 k 個 CU 的 slowdown 查表」，取 max(GEMM, comm) 最小的 CU 分配；30 個情境中預測對 24 個（§V-C，p7） | 〔全文〕 |
| NanoFlow [12] | OSDI'25 | 每個 kernel 的資源占比 R 透過離線配對 profiling 表映射成效能 P，並限制 ΣR ≤ 1（§4.1.1–4.1.3） | 〔部分〕 |

**對我們的建議**
- 把我們的設計寫成 **Bubble-Up／Paragon 的 GPU 版**：
  - 壓力源 SoI 至少分三種：tensor／SM 計算、HBM 頻寬、copy engine／PCIe。
  - 「被干擾者」是各種 restore 運算元：H2D copy、SM gather、FP8 dequant、INT4 dequant、low-rank GEMM、recompute prefill。
  - 每個格合起來就是一條「敏感度曲線」。
- 我們量到的「compute 背景翻轉、memory 背景不翻轉」，就是不同 SoI 下敏感度排序不同；這是 Paragon 式分類能表達、而純量 utilization 模型（例如只看 active decode 數的 Cascade）表達不了的。
- 線上部分的作法：
  - 學 Paragon「只跑兩次短 profiling＋矩陣補全」；
  - 或學 Bullet「兩點校準＋線性外插」。
  - 兩者都能回應審稿人「profiling 成本多大」的質疑。

---

## 6. Q5：restore 的排序、期限，以及降級

| 工作 | 書目（核實） | 和 restore 排序或降級的關係 | 深度 |
|---|---|---|---|
| Bidaw [2] | FAST'26 | disk-HRRN：依「等待時間／KV 大小」排 capacity-layer 的讀取（SRPT 加 aging）（§3.2，p7） | 〔全文〕 |
| Cascade [1] | arXiv 2608.06557 | least-remaining-budget-first，並以預算限制 restore 的 bytes（Alg. 1，p7） | 〔全文〕 |
| Nuska [57] | Y. Sun 等 8 人，arXiv 2603.17456（venue NOT_VERIFIED） | 在 PD 分離架構中，對 KV-block 取回、collective、P2D 等多階段 flow 近似做 LLF（Reverse MLQ，defer-and-promote）；整合進 vLLM | 〔摘要〕 |
| TokenFlow [58] | J. Chen 等 9 人，EuroSys'26（arXiv Comments） | 依即時 token buffer 佔用與消耗速率排優先序；KV 在背景 GPU↔CPU 搬移；硬體為 RTX 4090／A6000／H200 | 〔摘要〕 |
| OrbitFlow [56] | X. Ma 等 7 人，VLDB'26（arXiv Comments） | 用 ILP 決定每個請求保留哪些層的 KV；重負載時延後大請求（降級） | 〔摘要〕 |
| MorphServe [55] | Z. Su 等 7 人，arXiv 2506.02006（venue NOT_VERIFIED） | 高負載時把層換成量化版本；記憶體壓力下調整 KV 大小（負載觸發的品質降級，針對權重與 KV 容量） | 〔摘要〕 |
| ConServe [59] | Y. Qiao 等 13 人，arXiv 2410.01228 | 層級 preemption＋incremental KV，讓 offline 工作讓路給 online | 〔摘要〕 |
| SRPT [52] | L. E. Schrage、L. W. Miller，Oper. Res. 14(4)，pp. 670–684，1966，doi 10.1287/opre.14.4.670（Crossref） | M/G/1 下 SRPT 的分析；restore 排序的古典根據 | 〔標題〕 |
| Size-based web scheduling [53] | M. Harchol-Balter, B. Schroeder, N. Bansal, M. Agrawal，ACM TOCS 21(2)，pp. 207–233，2003（Crossref） | 系統實作中 SRPT 的公平性與效益 | 〔標題〕 |
| HRRN [54] | P. Brinch Hansen，SOSP'71，pp. 101–105，doi 10.1145/800212.806506（Crossref；Bidaw 引用的 HRRN 來源） | HRRN | 〔標題〕 |
| **Brownout** [51] | C. Klein, M. Maggio, K.-E. Årzén, F. Hernández-Rodriguez，**ICSE 2014**，pp. 700–711，doi 10.1145/2568225.2568227（Crossref 核實：4 位作者、頁碼、DOI） | 用控制理論在 runtime 關掉可選內容，在過載或故障時保持回應性（RUBiS／RUBBoS 原型） | 〔摘要〕（Lund 大學紀錄與二手摘要） |

**Q5 判定**
- 「restore 依大小或 slack 排序」**已有前例**（Bidaw、Cascade、Nuska）。我們的排序只有在把「每個 restore 會消耗哪種資源」也納入時，才有新意。例如兩個 restore 一個主要吃 copy engine、一個主要吃 SM，它們可以並行；兩個都吃 SM 時則應該排序。
- 「Brownout 式降級」在 KV restore 上，本次**未找到**依 GPU 爭用切換格式或精度的工作。MorphServe 是依負載降級權重精度，最接近。
- 我們量到 p99 約 98% 是佇列時間。這表示排序（SRPT／HRRN／LRBF）本身可能就是主要收益來源；**必須做消融**，證明「爭用感知的計畫選擇」在固定排序之外還有額外貢獻。

---

## 7. 缺口與審稿人會要求的 baseline

### 7.1 確定存在的缺口（本次搜尋範圍內）

1. **輸入**：沒有 restore 規劃器使用即時、多維的 GPU 爭用量測。
   - Cascade 只用 active decode 數＋離線 γ。
   - KVServe 用 EWMA 殘差。
   - py-kvcache、CacheTune、Krul 都是離線校準。
   - vLLM、SGLang、LMCache 都是靜態常數或設定。
2. **決策空間**：沒有工作把下面幾項放進同一個決策：
   - recompute／load 切分
   - 儲存格式解碼路徑：BF16 直搬、FP8／INT4 dequant、low-rank 重建
   - copy engine vs SM gather
   - SM 配額（green context／MPS／CU mask）
   - 多個 restore 的資源感知排序
3. **現象**：沒有人報告「最佳格式隨爭用**類型**翻轉」。
   - 最接近的反面證據都只說「會干擾」：ShadowServe 說解壓會互相干擾；Bidaw 假設 SM 有空；ConCCL 說 HBM 爭用仍在。

### 7.2 必要 baseline（依審稿人可能的質疑排序）

| # | baseline | 目的 | 來源 |
|---|---|---|---|
| B1 | 每台硬體事後最佳的單一固定格式／路徑 | 證明「依情境選」優於最佳固定選擇（我們量到 6–14%） | 自建 |
| B2 | **Cake**（bidirectional，交會點在 runtime 自然移動） | Cake 本身對執行速度變化就有隱含適應力。必須顯示在 busy GPU 下 Cake 仍然不夠：它不能換格式或路徑，也不能分 SM | 已收錄 |
| B3 | **Cascade 式預算切分**：Eq. 6–7，γ 離線固定，負載＝active decode 數 | 證明只看頻寬與佇列長度的切分在 compute 型背景下會偏 | [1] |
| B4 | **py-kvcache／CacheTune 式離線 break-even 或校準** | 證明離線常數在背景負載改變時會選錯（呼應我們 H200 上 12 格錯 10 格） | [4][10] |
| B5 | **KVServe 式 EWMA 殘差修正** | 證明純量殘差無法表達「依資源類型翻轉」 | 已收錄 |
| B6 | **vLLM 靜態選路**（28 KiB、n≥16、12 SM）與 SGLang `kernel`／`direct` 二選一 | 證明 copy engine vs SM 的交叉點會隨爭用移動 | [3][17] |
| B7 | **靜態 SM 預留**（Tutti 式 green context，或固定 MPS 百分比） | 證明動態配額優於固定預留，並量 green context 的實際隔離度（ShadowServe 認為它不能切 HBM） | 已收錄、[13] |
| B8 | **restore 排序**：FCFS vs Bidaw HRRN vs Cascade LRBF vs 我們的資源感知排序 | 把「排序」與「計畫選擇」的貢獻分開（p99 約 98% 是佇列） | [1][2] |
| B9 | **成本模型消融**：純 utilization 純量模型 vs Bullet-SRM 式「roofline × SM 比例」vs ConCCL 式 CU-loss 查表 vs 我們的資源向量 | 回應「為什麼需要向量」 | [6][8] |
| B10 | **把解碼搬離 GPU**：CPU 解碼，或 ShadowServe 式 SmartNIC | 作為「完全避開 GPU 爭用」的上界或對照 | 已收錄 |

### 7.3 審稿人會追問的其他點

- **量測成本與時間尺度**：爭用訊號多久更新一次？會不會震盪？可以參考 Paragon 的短 profiling、LithOS 的線上修正、Bullet 的兩點校準。
- **跨平台**：MI300X（CU masking、SDMA）與 H200（green context、copy engine）的旋鈕不同。ConCCL 與 RAPID-Serve 是 AMD 上的直接前例。
- **與 py-kvcache future work 的區隔**：必須在 related work 裡明講兩者的差別。

---

## 8. 參考文獻（IEEE 格式；最後一欄為核實狀態）

[1] M. Adnan, R. Mahapatra, P. J. Nair, D. Berger, P. Zardoshti, R. Fonseca, and E. Choukse, "Cascade: Exploiting SLO-aware latency budget for fair and high goodput LLM inference serving," arXiv:2608.06557, Aug. 2026. https://arxiv.org/abs/2608.06557 — arXiv API 核實（7 人，v1 2026-08-06）；venue NOT_VERIFIED；〔全文〕

[2] S. Hu, G. Zhang, Y. Zhou, Y. Wei, Z. Zhong, and J. Chen, "Bidaw: Enhancing key-value caching for interactive LLM serving via bidirectional computation–storage awareness," in *Proc. 24th USENIX FAST*, Santa Clara, CA, Feb. 2026. https://www.usenix.org/conference/fast26/presentation/hu-shipeng — PDF 封面核實；〔全文〕

[3] vLLM PR #42212, "[Perf] Triton fast path for small CPU→GPU `swap_blocks_batch` in the offloading connector," opened 2026-05-10, merged 2026-06-03. https://github.com/vllm-project/vllm/pull/42212 —〔PR〕

[4] J. Kanichai, T. De Matteis, and A. Trivedi, "Building py-kvcache: A performance characterization of external KV caching for vLLM with NVMe SSDs," arXiv:2609.11744, Sep. 2026. https://arxiv.org/abs/2609.11744 — arXiv API 核實（3 人）；venue NOT_VERIFIED；〔全文〕

[5] M. Ibrahim, "When the KV cache is slower than the GPU: Scaling remote KV tiers for context-parallel prefill," arXiv:2610.21500, Oct. 2026. https://arxiv.org/abs/2610.21500 — arXiv API 與 PDF 核實（1 人；有 LinkedIn 貼文把作者寫成他人，以 PDF 為準）；venue NOT_VERIFIED；〔全文〕

[6] A. Agrawal, S. Aga, S. Pati, and M. Islam, "ConCCL: Optimizing ML concurrent computation and communication with GPU DMA engines," in *Proc. IEEE ISPASS*, 2025, pp. 1–11, doi: 10.1109/ISPASS64960.2025.00018. arXiv:2412.14335. https://arxiv.org/abs/2412.14335 — Crossref＋arXiv 核實；〔全文〕

[7] P. H. Coppock *et al.*, "LithOS: An operating system for efficient machine learning on GPUs," in *Proc. ACM SOSP*, 2025, pp. 1–17, doi: 10.1145/3731569.3764818. arXiv:2504.15465. — Crossref 核實（9 人）；〔全文〕（arXiv v1）

[8] Z. Lin, H. Xu, G. Chen, Z. Chen, Y. Lu, and X. Zhang, "Boosting LLM serving through spatial-temporal GPU resource sharing," arXiv:2504.19516 v4, 2025（v1 標題 "Bullet: Boosting GPU utilization for LLM serving via dynamic spatial-temporal orchestration"）. https://arxiv.org/abs/2504.19516 — arXiv API 核實（6 人）；venue NOT_VERIFIED；〔全文〕

[9] J. Wen, J. Liang, Z. Hong, W. Chen, T. Cai, and Z. Zheng, "Krul: Efficient state restoration for multi-turn conversations with dynamic cross-layer KV sharing," arXiv:2507.08045, Jul. 2025. — arXiv API 核實（6 人）；venue NOT_VERIFIED；〔部分〕（§6）

[10] F. Li *et al.*, "Adaptive KV cache reuse for fast long-context LLM serving," arXiv:2605.24022, May 2026（v2 Oct. 2026）. — arXiv 核實（7 人）；venue NOT_VERIFIED；〔部分〕

[11] C. Jiang, L. Gao, H. E. Zarch, and M. Annavaram, "KVPR: Efficient LLM inference with I/O-aware KV cache partial recomputation," in *Findings of ACL*, 2025. arXiv:2411.17089. — arXiv Comments 核實；〔摘要〕

[12] K. Zhu *et al.*, "NanoFlow: Towards optimal large language model serving throughput," in *Proc. USENIX OSDI*, 2025（pp. 749–765 來自搜尋摘要）. arXiv:2408.12757. https://www.usenix.org/conference/osdi25/presentation/zhu-kan — 〔部分〕

[13] X. Xiang *et al.*, "ShadowServe: Interference-free KV cache fetching for distributed prefix caching," arXiv:2509.16857, Sep. 2025. — 已收錄；本檔只補新事實；〔部分〕

[14] SGLang PR #45945, "[Feat] Comm: Gated background transfers," opened 2026-10-23（open）. https://github.com/sgl-project/sglang/pull/45945 —〔PR〕

[15] SGLang PR #44029, "[Perf] KVShard: 0SM KV cache allgather for KV cache sharding," opened 2026-10-13（open）. https://github.com/sgl-project/sglang/pull/44029 —〔PR〕

[16] SGLang PR #38449, "[Perf] hicache: Issue round-head H2D copies before the load burst hand-over," opened 2026-09-08（open）. https://github.com/sgl-project/sglang/pull/38449 —〔PR〕

[17] SGLang, `python/sglang/srt/arg_groups/fields/memory.py`（`hicache_io_backend`，main，讀取日 2026-10-24），以及 HiCache design doc（https://docs.sglang.io/advanced_features/hicache_design.html）. —〔程式碼〕〔官方文件〕

[18] SGLang, `python/sglang/srt/multiplex/pdmux_context.py`（main，2026-10-24），以及 PR #22827, "[PoC] Use Green Context SM partitioning for overlap schedule stream"（closed，2026-04）. —〔程式碼〕〔PR〕

[19] LMCache issue #4498, "[RFC][MP] Lossless KV compression with GPU decompression," opened 2026-08-12（open）. https://github.com/LMCache/LMCache/issues/4498 —〔PR〕

[20] vLLM PR #43018（ROCm `hipMemcpyBatchAsync`，merged 2026-08-21）與 PR #52700（ROCm 恢復 KV-offload Triton fast path，open，2026-08-18）. —〔PR〕

[21] NVIDIA, "NCCL Release 2.28.3," release notes. https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-28-3.html —〔摘要〕

[22] S. Jeaugey *et al.*, "Fusing communication and compute with new device API and copy engine collectives in NVIDIA NCCL 2.28," NVIDIA Technical Blog, Nov. 10, 2025. — 〔部分〕

[23] S. Pati *et al.*, "DMA-Latte: Expanding the reach of DMA offloads to latency-bound ML communication," in *Proc. IEEE/ACM MICRO*, 2026（arXiv Comments：accepted）. arXiv:2511.06605. — 6 人；〔摘要〕

[24] S. Pati *et al.*, "T3: Transparent tracking & triggering for fine-grained overlap of compute & collectives," in *Proc. ACM ASPLOS*, 2024, pp. 1146–1164, doi: 10.1145/3620665.3640410. — Crossref；〔標題〕

[25] A. Ruzhanskaia *et al.*, "Rethinking programmed I/O for fast devices, cheap cores, and coherent interconnects," arXiv:2409.08141, Sep. 2024. — arXiv API（4 人）；venue NOT_VERIFIED；〔摘要〕

[26] K. Vaidyanathan *et al.*, "Designing efficient asynchronous memory operations using hardware copy engine: A case study with I/OAT," in *Proc. IEEE IPDPS*, 2007, pp. 1–8, doi: 10.1109/IPDPS.2007.370479. — Crossref；〔標題〕

[27] R. Kuper *et al.*, "A quantitative analysis and guidelines of data streaming accelerator in modern Intel Xeon scalable processors," in *Proc. ACM ASPLOS*, 2024, pp. 37–54, doi: 10.1145/3620665.3640401. — Crossref（10 人）vs arXiv 2305.02480（7 人），作者數不一致；〔摘要〕

[28] S. W. Min *et al.*, "EMOGI: Efficient memory-access for out-of-memory graph-traversal in GPUs," *PVLDB*, 2020, pp. 114–127, doi: 10.14778/3425879.3425883. — Crossref；〔標題〕

[29] S. W. Min *et al.*, "Large graph convolutional network training with GPU-oriented data communication architecture," *PVLDB*, 2021, pp. 2087–2100, doi: 10.14778/3476249.3476264. — Crossref；〔標題〕

[30] Z. Qureshi *et al.*, "GPU-initiated on-demand high-throughput storage access in the BaM system architecture," in *Proc. ACM ASPLOS*, 2023, pp. 325–339, doi: 10.1145/3575693.3575748. — Crossref；〔標題〕

[31] J. Mars, L. Tang, R. Hundt, K. Skadron, and M. L. Soffa, "Bubble-Up: Increasing utilization in modern warehouse scale computers via sensible co-locations," in *Proc. IEEE/ACM MICRO-44*, 2011, pp. 248–259, doi: 10.1145/2155620.2155650. — Crossref＋Google Research 頁；〔摘要〕

[32] H. Yang, A. Breslow, J. Mars, and L. Tang, "Bubble-Flux: Precise online QoS management for increased utilization in warehouse scale computers," in *Proc. ISCA*, 2013, pp. 607–618, doi: 10.1145/2485922.2485974. — Crossref 核實完整作者名單（4 人）；〔標題〕

[33] C. Delimitrou and C. Kozyrakis, "Paragon: QoS-aware scheduling for heterogeneous datacenters," in *Proc. ACM ASPLOS*, 2013, pp. 77–88, doi: 10.1145/2451116.2451125. — Crossref；〔全文〕（§2.3）

[34] C. Delimitrou and C. Kozyrakis, "Quasar: Resource-efficient and QoS-aware cluster management," in *Proc. ACM ASPLOS*, 2014, pp. 127–144, doi: 10.1145/2541940.2541941. — Crossref；〔部分〕

[35] X. Zhao, M. Jahre, and L. Eeckhout, "HSM: A hybrid slowdown model for multitasking GPUs," in *Proc. ACM ASPLOS*, 2020, pp. 1371–1385, doi: 10.1145/3373376.3378457. — Crossref；〔摘要〕

[36] W. Zhao *et al.*, "Themis: Predicting and reining in application-level slowdown on spatial multitasking GPUs," in *Proc. IEEE IPDPS*, 2019, pp. 653–663, doi: 10.1109/IPDPS.2019.00074. — Crossref；〔標題〕

[37] Q. Chen, H. Yang, J. Mars, and L. Tang, "Baymax: QoS awareness and increased utilization for non-preemptive accelerators in warehouse scale computers," in *Proc. ACM ASPLOS*, 2016, pp. 681–696. — Crossref 核實完整作者名單（4 人）；〔標題〕

[38] Q. Chen *et al.*, "Prophet: Precise QoS prediction on non-preemptive accelerators to improve utilization in warehouse-scale computers," in *Proc. ACM ASPLOS*, 2017, pp. 17–32. — Crossref（6 人）；〔標題〕

[39] F. Strati, X. Ma, and A. Klimovic, "Orion: Interference-aware, fine-grained GPU sharing for ML applications," in *Proc. ACM EuroSys*, 2024, pp. 1075–1092, doi: 10.1145/3627703.3629578. — Crossref 核實完整作者名單（3 人）；〔摘要〕

[40] M. Han, H. Zhang, R. Chen, and H. Chen, "Microsecond-scale preemption for concurrent GPU-accelerated DNN inferences," in *Proc. USENIX OSDI*, 2022. https://www.usenix.org/conference/osdi22/presentation/han — USENIX 頁；〔摘要〕

[41] K. K. W. Ng, H. M. Demoulin, and V. Liu, "Paella: Low-latency model serving with software-defined GPU scheduling," in *Proc. ACM SOSP*, 2023, pp. 595–610, doi: 10.1145/3600006.3613163. — Crossref 核實完整作者名單（3 人）；〔標題〕

[42] S. S. Shubha, H. Shen, and A. Iyer, "USHER: Holistic interference avoidance for resource optimized ML inference," in *Proc. USENIX OSDI*, 2024. https://www.usenix.org/conference/osdi24/presentation/shubha — USENIX 頁；〔摘要〕

[43] F. Xu *et al.*, "iGniter: Interference-aware GPU resource provisioning for predictable DNN inference in the cloud," *IEEE Trans. Parallel Distrib. Syst.*, 2023, pp. 812–827, doi: 10.1109/TPDS.2022.3232715. — Crossref＋Semantic Scholar（7 人）；〔摘要〕

[44] W. Cui *et al.*, "Enable simultaneous DNN services based on deterministic operator overlap and precise latency prediction," in *Proc. SC*, 2021, pp. 1–15, doi: 10.1145/3458817.3476143. — Crossref（11 人）；〔標題〕

[45] J. Duan *et al.*, "MuxServe: Flexible spatial-temporal multiplexing for multiple LLM serving," arXiv:2404.02015, Apr. 2024. — arXiv API（8 人）；venue NOT_VERIFIED；〔摘要〕

[46] K. Hong *et al.*, "semi-PD: Towards efficient LLM serving via phase-wise disaggregated computation and unified storage," arXiv:2504.19867, Apr. 2025. — arXiv API（12 人）；venue NOT_VERIFIED；〔摘要〕

[47] Y. Chen *et al.*, "Towards high-goodput LLM serving with prefill-decode multiplexing," arXiv:2504.14489 v3, 2025. — arXiv API（10 人）；venue NOT_VERIFIED；〔摘要〕

[48] A. K. Kamath *et al.*, "POD-Attention: Unlocking full prefill-decode overlap for faster LLM inference," in *Proc. ACM ASPLOS*, 2025, pp. 897–912, doi: 10.1145/3676641.3715996. — Crossref＋arXiv Comments（6 人）；〔標題〕

[49] A. Masood *et al.*, "RAPID-Serve: Resource-efficient and accelerated P/D intra-GPU disaggregation," arXiv:2601.11822, Jan. 2026. — arXiv API（3 人）；venue NOT_VERIFIED；〔摘要〕

[50] T. Hu *et al.*, "Hummingbird: SLO-oriented GPU preemption at microsecond-scale," arXiv:2601.04071, Jan. 2026. — arXiv API（12 人）；venue NOT_VERIFIED；〔摘要〕

[51] C. Klein, M. Maggio, K.-E. Årzén, and F. Hernández-Rodriguez, "Brownout: Building more robust cloud applications," in *Proc. 36th Int. Conf. Software Engineering (ICSE)*, 2014, pp. 700–711, doi: 10.1145/2568225.2568227. — Crossref 核實（4 人、頁碼、DOI）；〔摘要〕

[52] L. E. Schrage and L. W. Miller, "The queue M/G/1 with the shortest remaining processing time discipline," *Oper. Res.*, vol. 14, no. 4, pp. 670–684, 1966, doi: 10.1287/opre.14.4.670. — Crossref；〔標題〕

[53] M. Harchol-Balter, B. Schroeder, N. Bansal, and M. Agrawal, "Size-based scheduling to improve web performance," *ACM Trans. Comput. Syst.*, vol. 21, no. 2, pp. 207–233, 2003, doi: 10.1145/762483.762486. — Crossref；〔標題〕

[54] P. Brinch Hansen, "Short-term scheduling in multiprogramming systems," in *Proc. 3rd ACM SOSP*, 1971, pp. 101–105, doi: 10.1145/800212.806506. — Crossref；〔標題〕

[55] Z. Su *et al.*, "MorphServe: Efficient and workload-aware LLM serving via runtime quantized layer swapping and KV cache resizing," arXiv:2506.02006, May 2025. — arXiv（7 人）；venue NOT_VERIFIED；〔摘要〕

[56] X. Ma *et al.*, "OrbitFlow: SLO-aware long-context LLM serving with fine-grained KV cache reconfiguration," in *Proc. VLDB*, 2026（arXiv Comments：accepted）. arXiv:2601.10729. — 7 人；〔摘要〕

[57] Y. Sun *et al.*, "Stage-aware communication scheduling for disaggregated LLM serving," arXiv:2603.17456, Mar. 2026. — arXiv API（8 人）；venue NOT_VERIFIED；〔摘要〕

[58] J. Chen *et al.*, "TokenFlow: Responsive LLM text streaming serving under request burst via preemptive scheduling," in *Proc. ACM EuroSys*, 2026（arXiv Comments：accepted）. arXiv:2510.02758. — 9 人；〔摘要〕

[59] Y. Qiao *et al.*, "ConServe: Fine-grained GPU harvesting for LLM online and offline co-serving," arXiv:2410.01228, Oct. 2024. — arXiv API（13 人）；venue NOT_VERIFIED；〔摘要〕

[60] J. Yao *et al.*, "CacheBlend: Fast large language model serving for RAG with cached knowledge fusion," arXiv:2405.16444, May 2024. — arXiv API（9 人）；venue NOT_VERIFIED（本次沒有核實 EuroSys 收錄）；〔摘要〕

**作者姓名的核實狀態**：[32][37][38][39][41] 的完整作者名單已用 Crossref 逐 DOI 核對。標 *et al.* 的條目只核實了第一作者與作者數。
