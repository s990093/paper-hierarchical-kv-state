# L2　壓縮（B）× 卸載與分層（C）× 預取與重疊（D）：第三階段文獻地圖

> **日期**：2026-10-19（UTC）
> **執行者**：文獻 agent（只讀、只上網；**沒有碰 GPU、沒有跑任何量測**，機器正在計時）。
> **範圍**：B＝KV 壓縮裡第二階段**沒涵蓋**的部分（無損、傳輸用壓縮、盈虧分析）；C＝卸載／分層／SSD／NFS 的**新作與缺口**（已整理過的只列指標）；D＝預取、非同步搬運、計算與搬運重疊，特別是 **GPU 忙時**。
> **先讀過的本專案檔案**：`docs/precision_20261013/02_papers.md`、`cards/P01–P08`（特別是 P02、P07）、`09_sweep_plan.md`、`08_report_mi300x.md` §2.1／§2.8、`11_issues.md` §3c／§3d、`12_tier_plan.md`、`docs/SOTA_ANALYSIS_20261006.md`、`docs/research_20261009_explore/`（lit_A、cards/A_*）、`docs/research_20261006_eval/cards/E04_tiered_systems.md`、`docs/phase1_20261008/07_report.md`。

**標記**
- 閱讀深度：〔全文〕＝整頁讀過（多為官方部落格／文件）；〔部分〕＝用 WebFetch 讀 arXiv HTML 版（前 10 萬字元），由工具依我給的欄位抽取，**不是逐頁人工核對**；〔摘要〕＝只讀 arXiv 摘要頁或官方摘要。
- 句子層級：〔原文〕＝論文自己寫的（經工具抽取）；〔實測 run〕＝本專案量到的，附 run_id；〔算術〕＝我用公開數字或本專案數字算的；〔推論〕＝我的判讀；**NOT_FOUND**＝查了沒找到（≠不存在）；**NOT_VERIFIED**＝有線索但沒核到原始來源。
- **所有 arXiv ID 都用 WebFetch 實際開過 abs 頁**（§6 的「核」欄）。不同硬體、batch、context 的數字**只並列，不排名**。
- 本檔是 **self-review**，不是 cross-model review（CLAUDE.md §5）。

---

## 0. 一頁結論

1. **無損 KV 壓縮已經有「專門給 GPU 用」的 codec，但壓縮比天花板很低**。SplitZip（NeurIPS 2026）只動 BF16 的指數位元，6 個模型的 KV 都是 **1.31–1.33×**，H200 上編碼 613、解碼 2,182 GB/s〔原文〕。通用 GPU codec 在同一份 KV 上幾乎壓不動（nvCOMP LZ4 1.019×、Cascaded 1.000×）〔原文〕；AMD GPU 上的通用無損（LZ4／Snappy／Cascaded 的 HIP 移植）在真實浮點資料（地震模擬）上也只有 1.03–1.04×〔摘要〕。
2. **「多慢的鏈路才值得壓縮」已經有人寫成公式，也有實測門檻**：KVServe（SIGCOMM'26）B\*＝(1−1/cr)·s（加總模型、不重疊）；SplitZip 的管線版 B_hide＝min(G_enc, G_dec)/ρ；ShadowServe 實測 ≤20 Gbps 才贏、>20 Gbps 慢 11–24%；KVServe 量到 3 種方法的門檻 50／55／110 Gbps〔原文〕。本專案 stage 44 也量過 INT4 對 BF16 的損益兩平點 17.6–29.9 GiB/s〔實測，08 §2.8〕。**沒找到**任何一篇把「解壓／反量化和計算搶 GPU」放進盈虧公式；而我們在 MI300X 上量到的最大項正是它（GPU 忙時反量化慢 2.4–3.8 倍，瓶頸是 kernel 排隊，11 §3d）。
3. **壓縮救不了 NFS**〔算術＋推論，§1.5〕：NFS 級（0.331 GiB/s 限速）全 BF16 還原比全在 CPU 慢 28 倍（Llama 2.64 vs 0.094 s）、34 倍（Qwen 1.79 vs 0.052 s）〔實測 stage 44〕；INT4（位元組剩 1/3.2）只拉到 18–19 倍。無損 1.3× 更不可能接近 CPU。**壓縮在 NFS 上真正的用途是縮短「預取到 DRAM」需要的提前量**：Llama 32K BF16 要 12.1 s，INT4 3.8 s〔算術〕。這把問題從「怎麼壓」轉成「預取來不來得及」。
4. **C：分層系統在 2026 年全部上線，但卸載層幾乎都不壓縮**。vLLM OffloadingConnector、llm-d FS backend、Dynamo KVBM、AMD Infinity Context（hipFile，依 AMD 部落格 ROCm 7.14 GA）、NVIDIA ICMSP 都只搬原始位元組；**唯一在卸載時壓縮的是 TensorRT-LLM 的 cold-page NVFP4**（只有 SM100／SM103）；LMCache 的 Intel QAT 無損壓縮是**未合併的草稿 PR**，沒有任何數字〔全文／原文〕。我們的容器是 ROCm 7.2.2，**hipFile 不在期限內的可用範圍**〔推論〕。
5. **D：預取／重疊的論文幾乎都假設 GPU 只跑自己**（InfiniGen、ShadowKV、KVPR、FreeKV、HCache 都沒量干擾）。干擾的證據來自別處：訓練通訊的重疊平均拖慢計算 18.9%（最多 40%，含 MI250／MI210）〔摘要〕；MI300X 上計算與通訊並行只拿到理想加速的 21%，改用 14 個 SDMA 引擎後到 72%（但 HBM 爭用仍在）〔部分〕；vLLM 部落格量到「用 GPU 核心拷貝」比 DMA 在併發時差、甚至比不卸載還差 6%〔全文〕。**我們自己的 13 號實測已經是這個缺口最強的證據**：GPU 忙時 Cake 的重疊最慢，「全部搬完一次拆」最穩（11 §3d）。
6. **最值得投入的三個缺口**（§4）：G1「壓縮×預取提前量」在 NFS 級的成本模型；G2「GPU 忙時，搬運走 SDMA 還是 kernel、重疊還划不划算」（MI300X 特有、便宜可測）；G3「多請求、多層混合時誰先還原」。G4（壓縮＋丟棄＋卸載聯合模型）有前作（EvicPress、InferCept、ALISA、KVServe），而且很可能被第二階段的結論（位置不重要、INT4 全用就夠）壓成一條簡單規則，**新意風險最高**。

### 0.1 委託裡的三個歷史數字（核實結果）

| 委託裡的說法 | 實際出處 | 正確讀法 |
|:--|:--|:--|
| CPU 還原 32K 約 202 ms | 第一階段 A1，run `20261008-134526-m7-a1`（Llama-3.1-8B、MI300X、GPU 閒、寫完才回來、TTFT 中位數）：「全放 CPU」202 ms 是 **NFS 參數那一列**；本地 SSD 那一列是 189 ms（`docs/phase1_20261008/07_report.md` §5） | 這是第一階段 harness 的 TTFT。第二階段的 E0（全部在 CPU、Cake、真實 H2D）是 **Llama 0.095 s、Qwen 0.052 s**（GPU 閒；08 §2.1；run `20261015-161226-m10-grid-fp8-llama`、`20261015-194150-m10-regrid-fp8-qwen`）。兩個 harness 不同，**不要混用** |
| NFS 約 2720 ms | 同一個 run：NFS 參數、S0 純 Cake、32K，會合點 50 | 「NFS」是**限速器模擬**（0.331 GiB/s＝實測最慢 ×0.9），不是真的 NFS；真實 NFS 讀取在第一階段量到 377 MiB/s（C2），同一天稍後的錨點約 500–830 MiB/s（07 的 C7 錨點段）。第二階段 stage 44：全部在 NFS 級，BF16 **Llama 2.64 s、Qwen 1.79 s**；INT4 1.72／0.98 s（08 §2.8） |
| GPU 忙讓還原慢 13–66% | 第一階段 Q2：**S3（一開始不寫前段）相對 S0（每段都存）**在 GPU 忙時慢 13–66%（16K 以上；忙 1＝1.5×、忙 2＝2.1×） | 這是「沒有備份」的代價，**不是**「GPU 忙」本身讓還原變慢的幅度。後者〔算術，用 08 §2.1〕：CPU 層 E0 Llama 0.095→0.165 s（+74%）、Qwen 0.052→0.066 s（+27%）；b48（Llama 0.42 GiB/s）2.421→3.951 s（+63%）。拆開看：H2D Llama 80.0→120.1 ms（+50%）、Qwen 35.7→52.2 ms（+46%）；FP8 反量化慢 2.4–3.8 倍（09 §6.1） |

---

## 1. B：KV 壓縮——第二階段沒涵蓋的部分

### 1.1 第二階段已經涵蓋、這裡不重做的

- **有損量化的格式與位置**：FP8 e4m3／e5m2、INT8、INT4（每 token、KIVI、Hadamard）、MXFP4、NVFP4、INT2、K8V4、K4V2（09 §2.1；P07 §1）。結論：位置沒幫助，INT4 KIVI 在 32K 兩個模型都過 ε（08 §0）。
- **反量化加速**：QServe、BitDecoding、FLUTE、Petit、ShadowServe、KVFetcher、PackKV 的 C 級轉述（P07 §4），以及我們自己的融合 kernel（11 §3a–3d）。
- **CacheGen 的機制與程式碼落差**（P02，〔全文〕＋程式碼逐行）。
- **更正 P07 的一處**：P07 §4 寫的「KVFetcher（2026）」，arXiv 2602.09725 的論文標題是 *Efficient Remote KV Cache Reuse with GPU-native Video Codec*，系統名稱是 **KVCodec**，「KV fetcher」只是它的一個元件〔摘要，§6 [B5]〕。

### 1.2 總表

| # | 論文（年，venue） | 無損？ | 目標鏈路 | 壓縮比（原文） | 編／解碼速度（原文，硬體） | MI300X／ROCm | 深度 |
|:--|:--|:--|:--|:--|:--|:--|:--|
| B1 | SplitZip（2026，NeurIPS'26） | **位元級無損** | PD 分離的 RDMA | BF16 KV 1.31–1.33×；FP8 E5M2 再 1.14× | 613.3／2181.8 GB/s（H200） | Triton kernel，〔推論〕可移植，未驗證 | 〔部分〕 |
| B2 | KVServe（2026，SIGCOMM'26） | 有損＋nvCOMP | PD 分離、遠端 KV 池 | 依配置 | 以參數 s 表示，原文沒給 nvCOMP 數字 | codec 段依賴 nvCOMP（CUDA） | 〔部分〕 |
| B3 | KVTC（2025，ICLR'26） | 有損（PCA＋量化）＋DEFLATE | GPU 內與卸載儲存 | 最多 20×（特定 40×+）；DEFLATE 約再 1.23× | H100、8K×8：壓 379 ms、解 267 ms | PyTorch 部分可移植；DEFLATE 用 nvCOMP | 〔部分〕 |
| B4 | ShadowServe（2025，arXiv） | 有損（量化）＋DEFLATE | 分散式前綴快取，10–40 Gbps | 量化正好 1/2；整體比沒報 | BlueField-3 DEFLATE 解壓 276.5 Gbps；單一 CPU 核 2.5 Gbps | 需要 BlueField-3，**不可重現** | 〔部分〕 |
| B5 | KVCodec（2026，SIGCOMM'26） | 量化（有損）＋H.265 無損模式 | 遠端 KV，1–40 Gbps | 11.9×（基準 dtype 沒寫） | NVDEC：2×H20 約 67K token/s | MI300X 有 VCN 解碼器，但 H.265 無損模式與編碼端 **NOT_VERIFIED** | 〔部分〕 |
| B6 | HACK（2025，arXiv） | 有損 2-bit，**不反量化** | PD 分離，10–400 Gbps | 原文稱 CacheGen／KVQuant 約 86% 壓縮 | 反量化佔 JCT 17.2–37.9%（他們量別人的） | Triton＋vLLM＋FA2，〔推論〕部分可移植 | 〔部分〕 |
| B7 | 權重無損家族：DFloat11、ZipNN、ZipServ、Huff-LLM、Ecco | 無損（Ecco 有損） | 權重 | 約 30%（DFloat11）、常見 33%（ZipNN） | ZipNN 單執行緒 BF16 解壓 1,660 MB/s（README） | ZipNN 是 CPU 函式庫 → **可直接在主機用** | 〔摘要〕 |
| B8 | KVComp／PackKV（2025，arXiv） | 有損＋無損打包 | GPU 內 | 比既有方法多 47%（平均）／K 153.2%、V 179.6% | A100、RTX Pro 6000 | CUDA（未查證） | 〔摘要〕 |
| B9 | CacheGen（2023，SIGCOMM'24） | 有損＋算術編碼 | 網路 | 見 P02 | GPU 解碼會拖慢計算（B4、B5 的批評） | LMCache CUDA 擴充，ROCm **NOT_VERIFIED** | 見 P02 |
| B10 | 系統實作：LMCache serde／QAT PR、TRT-LLM cold-page NVFP4、NIXL／Mooncake、UCCL-Zip、ROCm 壓縮函式庫 | 混合 | — | — | — | 見 §1.4 | 〔全文／摘要〕 |

### 1.3 卡片

#### B1　SplitZip：BF16 KV 的超快無損壓縮〔部分〕
- **出處**：Yipin Guo、Siddharth Joshi。arXiv [2605.01708](https://arxiv.org/abs/2605.01708)（v1 2026-05-03，v3 2026-06-23）。NeurIPS 2026 poster（[neurips.cc](https://neurips.cc/virtual/2026/poster/148973) 已核）。
- **問題**：PD 分離時 prefill→decode 的 KV 傳輸是瓶頸；現有 codec 不是在 CPU 上跑，就是編碼太慢、跟不上 prefill 產生 KV 的速度〔原文〕。
- **方法與假設**：只壓 BF16 的指數位元。離線校準一份 top-16 指數碼本，常見的 16 種指數用定長碼，罕見的放進稀疏 escape 串；符號與尾數原樣保留，**位元級無損**。刻意不用可變長熵編碼（理由：編碼太慢）〔原文〕。假設指數分布能離線校準、跨請求穩定；改成逐 token／逐 channel 校準時吞吐掉到 <1 GB/s〔原文 §4.3.3〕。
- **改善（原文）**：
  - 指數熵 2.89–3.59 bit；top-16 覆蓋 99.3–99.9%；實得壓縮比 1.31–1.33×（Qwen3-30B-A3B、Qwen3-32B、Qwen3-Next-80B-A3B、Llama-3.1-70B、Llama-3-8B、Phi-2；Table 1）。
  - 編／解碼（GB/s）：H200 613.3／2181.8；A100 352.5／842.6；H100 579.8／1486.3；B200 637.9／2607.4（Table 2、10）。
  - 同一份資料（H200，Table 2）的對照：nvCOMP LZ4 1.019×、Cascaded 1.000×、Bitcomp 1.056×；DFloat11 1.423× 但編碼 0.004 GB/s；ZipServ 1.236×；ZipNN 1.515×（吞吐引自 ZipNN 原論文）。
  - 端到端（SGLang PD 分離＋Mooncake 傳輸，Qwen3-32B）：TTFT 最多 1.303×（batch 1）、1.274×（batch 16）；吞吐 1.233×／1.206×。傳輸時間：約 22K token 時只從 56.5 ms 到 53.1 ms；16K、batch 16 從 441.4 到 353.8 ms；64K 從 1749.3 到 1397.0 ms（Fig. 4）。
  - FP8 E5M2 上再壓 1.14×；FP8 E4M3 用 top-8 反而變大（App. C）。
- **成本、限制、失敗條件（原文）**：短 context、計算受限時 codec 開銷可能大於省下的；小 payload 有固定開銷；只有 NVLink 級鏈路（B_hide＝613.3/1.324≈463 GB/s 以上）才看得到 codec 時間（App. B）；傳輸拆解是「加總」而不是實測的管線（§4.1）；V cache 最差一層的 top-16 覆蓋 98.77%。
- **實作與 ROCm**：[GitHub Intelligent-Microsystems-Lab/SplitZip](https://github.com/Intelligent-Microsystems-Lab/SplitZip)；README：「GPU encode and decode kernels implemented in Triton」「requires a CUDA-capable GPU」。〔推論〕Triton 在 ROCm 上能跑（本專案的 `code/m10_dequant_fast.py` 已在 MI300X 跑 Triton），移植機會高，但**未驗證**；碼本要用我們的模型重新校準。
- **已研究 vs 缺證據**：已研究 RDMA／NVLink 級的 PD 傳輸。**缺**：CPU／SSD／NFS 卸載層；GPU 忙時解碼的干擾；和 Cake 會合點的耦合；**K 有固定大偏移的模型**（Qwen2.5，08 §2.4 的 K3）的指數分布會不會更集中或更分散。
- **期限前可測的最小版本**：X1（只用 CPU 算我們真實 KV 的指數熵與 top-16 覆蓋率，**不用寫 codec 就能預測壓縮比**）；X4（Triton 解碼在 MI300X 閒／忙）。

#### B2　KVServe：依服務狀態自動選壓縮組合〔部分〕
- **出處**：Zedong Liu 等 12 人。arXiv [2605.13734](https://arxiv.org/abs/2605.13734)（2026-05-13）。arXiv 註記 **SIGCOMM 2026**。
- **問題**：PD 分離與 KV 分離式服務裡，KV 跨網路與儲存搬運是端到端瓶頸；固定的壓縮設定在負載、頻寬、SLO 變動時不是最好〔原文〕。
- **方法與假設**：把壓縮拆成 Codec(Quantizer(Transform(X)))：Transform＝Delta／Hadamard／Affine；Quantizer＝KIVI 式多維量化、MixHQ（retrieval／streaming head 分開混精度）；Codec＝nvCOMP。離線用貝氏搜尋產生 3D Pareto 候選（搜尋量省 50×，約 20 小時離線 profiling），線上用解析延遲模型＋bandit 選設定〔原文〕。
- **盈虧公式（原文 Eq. 1、Eq. 5、Thm 6.1）**：T_p＝T_model＋V/s_p＋V/(B·cr_p)，對照 T_0＝T_model＋V/B。損益兩平頻寬 **B\*_p＝(1−1/cr_p)·s_p，和 KV 量 V 無關**。這是**加總**模型（原文沒說編解碼和傳輸是否重疊）。
- **改善（原文）**：PD 分離 JCT 最多 9.13×；KV 分離式 TTFT 最多 32.8×（對 CacheGen，5–15 Gbps，Qwen2.5-32B，Pro 6000）。動機實驗量到三種方法的門檻 50／55／110 Gbps，超過就比不壓縮慢（§2.2；原文列的方法依序是 CacheGen、MixHQ、KIVI，**對應關係原文沒明寫**）。硬體：prefill 2×4090／2×5090 @10 Gbps、2×RTX Pro 6000 @50 Gbps、2×H100 @100 Gbps；decode H100；模型 Qwen2.5-7B／32B、Llama-3.1-8B。
- **限制**：加總模型；品質分桶離線固定；線上探索可能違反 SLO（有護欄）；**GPU 爭用沒有建模**，只說 bandit 的殘差修正會吸收（§6.2），而且 §7.2 報 codec 開銷「可忽略」〔原文〕。
- **一個和本專案對得上的旁證**：原文提到靜態基線在 **Qwen2.5-7B 上因為 KV projection 的 bias 失敗**〔原文，經工具抽取、未逐頁核〕——和我們的 K3（Qwen 的 FP8 問題在 K 的固定偏移，08 §2.4）是同一件事。
- **實作與 ROCm**：[GitHub hpdps-group/KVServe](https://github.com/hpdps-group/KVServe)，整合 vLLM。codec 段依賴 nvCOMP（CUDA-only）；Transform／Quantizer 是張量運算〔推論：可移植〕。
- **已研究 vs 缺證據**：已研究「依頻寬選方法」與損益兩平；**缺**：解壓搶 GPU 的成本、卸載層（SSD／NFS）、和重算（Cake）混用。
- **最小版本**：直接拿它的公式，代入本專案的層頻寬與 X1／X4 量到的 s，算各層的 B\*（X2，純算術）。

#### B3　KVTC：PCA＋量化＋熵編碼的 KV 轉換編碼〔部分〕
- **出處**：Konrad Staniszewski、Adrian Łańcucki（NVIDIA）。arXiv [2511.01815](https://arxiv.org/abs/2511.01815)（v2 2026-03-11）。arXiv 註記 **ICLR 2026**。
- **問題**：讓 KV 能長期留在 GPU 內或卸載儲存，用壓縮取代重算〔原文〕。
- **方法與假設**：離線 SVD／PCA 去相關（每個模型、每個壓縮比一次，約 200K token 校準、一張 H100）；動態規劃分配每個主成分的位元（很多尾端成分給 0 bit）；打包後用 **nvCOMP 的 DEFLATE**（GPU）做無損熵編碼。**最近 128 個 token 和最前 4 個 sink 不壓**（壓它們傷準確率，Fig. 4、App. B.3／B.7）〔原文〕。
- **改善（原文）**：最多 20×，特定情境 40× 以上；**DEFLATE 平均在量化之上再給約 1.23×**（§4.1）。延遲（H100、Mistral NeMo 12B、簡單的 Transformers 實作，Table 5）：BS 8、8K：壓縮 379 ms、解壓 267 ms；TTFT 重算 3,098 ms vs KVTC 解壓 380 ms；BS 2、16K：1,780 vs 208 ms。
- **限制（原文）**：70B 在 MATH-500 掉 1.2 個百分點（10×）、3.0 個百分點（20×）；只測到 70B 的 dense 模型；多輪是用 benchmark 模擬；Frobenius 誤差只是準確率的代理；和丟 token 的組合沒評估。**沒有和鏈路頻寬掛鉤的盈虧分析**（§2 只有定性說明）。
- **實作與 ROCm**：論文沒有官方程式碼連結；搜尋到第三方實作 OnlyTerp/kvtc（**NOT_VERIFIED**）。PCA／量化是 PyTorch 運算〔推論：可移植〕；DEFLATE 在 ROCm 上沒有成熟的 GPU 版（§1.4），卸載層可改在 CPU 上做。
- **對本專案的意思**〔算術＋推論〕：KVTC 是「有損的上限」那一端。20× 套在 NFS 級（0.331 GiB/s）等於 6.6 GiB/s 的有效頻寬，接近我們的 SSD 級（6.98 GiB/s）——SSD 級全 BF16 的實測還原是 Llama 0.47 s，仍是 CPU 層（0.094 s）的 5 倍〔實測 stage 44〕。所以**即使 20× 也只能讓 NFS 變成 SSD**。
- **最小版本**：不建議期限前做（要 PCA 校準＋品質重量）。只把它放進 §1.5 的算術當上界。

#### B4　ShadowServe：把 KV 解壓搬到 SmartNIC〔部分〕
- **出處**：Xingyu Xiang、…、Junchen Jiang、Minlan Yu 等 9 人。arXiv [2509.16857](https://arxiv.org/abs/2509.16857)（2025-09-21）。arXiv 頁沒有 venue。
- **問題**：頻寬有限時 KV 抓取是瓶頸；壓縮能省頻寬，但**解壓和模型計算互相干擾**〔原文〕。
- **方法**：控制面留在主機，資料面（抓取、DEFLATE 解壓、反量化）全部卸到 BlueField-3；分塊管線、最少拷貝的記憶體管理。編碼是向量分箱量化（剛好砍一半）＋DEFLATE，**不是** CacheGen 的算術編碼〔原文 §4.3、§5〕。
- **干擾的量測（原文 §2.2、Fig. 3）**：L40S 上解壓和 decode（30K context、batch 1）放在一起，算術解碼沒辦法讓兩邊的拖慢都低於 30%；最好的反量化設定兩邊仍慢 25% 以上。
- **改善與失敗條件（原文 §6.2）**：TTFT 在 ≤20 Gbps 快 1.20–1.38×，**>20 Gbps 慢 11–24%**（SmartNIC 管線只到約 20.6 Gbps，單獨跑 37.3 Gbps，受記憶體爭用限制）。loaded TPOT 好 1.06–2.19×。>40 Gbps 不壓縮的基線會贏（**推算，不是實測**）。BF3 DEFLATE 解壓 276.5 Gbps vs 單一主機 CPU 核 2.5 Gbps（Table 1）。
- **限制**：不支援 chunked prefill、不支援部分命中；單機、儲存伺服器走 loopback；CacheGen 基線是作者自己改的非同步版。
- **ROCm**：需要 BlueField-3，**不可重現**。原文提到 AMD Pensando 理論上可以（§5），沒有實作。
- **對本專案的意思**〔推論〕：它是「解壓搶 GPU」最直接的量測，也是「壓縮只在慢鏈路贏」的實測證據（20 Gbps≈2.3 GiB/s，介於我們的 NFS 級與 SSD 級之間）。

#### B5　KVCodec：用 GPU 內建的影片編解碼器壓 KV〔部分〕
- **出處**：Liang Mi、Weijun Wang、…、Yunxin Liu。arXiv [2602.09725](https://arxiv.org/abs/2602.09725)（v3 2026-05-12）。arXiv 註記 **SIGCOMM 2026**。
- **問題**：遠端 KV 重用在低頻寬時變差；CacheGen 這類壓縮的解壓太重，抵銷了好處〔原文〕。
- **方法**：KV 先量化成整數（和 CacheGen／ShadowServe 同一套量化），排成適合影片編碼的版面，再用 **H.265 無損模式**（lossless=1，跳過 DCT 與量化）；NVDEC 解碼、和傳輸與還原做成管線〔原文 §3–4〕。所以「lossless accuracy」只指影片編碼那一段，前面的整數量化仍是有損〔原文，工具判讀〕。
- **改善（原文）**：壓縮比 11.9×（基準 dtype 沒寫）；比 CacheGen 多壓 2.17×、比 ShadowServe 1.93×；TTFT 最多快 3.51×；<40 Gbps 時對 CacheGen 快 1.29–3.50×（§5.2）。**干擾量測（§2.2、Fig. 5）**：GPU 上同時解壓讓 prefill 慢約 50%、decode 慢約 20%。硬體：A100（5 個 NVDEC）、H20（7 個）、L20（3 個）；模型 LWM-7B、Yi-34B、Llama3-70B；20K–200K。
- **限制（原文）**：NVENC 不夠用，所以**線上壓縮**（PD 分離、遷移）不行；優勢隨頻寬升高而縮小（NVDEC 變瓶頸）；GQA 模型的 KV 小、好處變小；解碼吞吐依 NVDEC 數量，L20 上只有 CacheGen 的 0.3×。
- **ROCm**：rocDecode 文件把 MI3xx（gfx942）列為 VCN 4.0、3–4 個 VCN，支援 H.265 8／10 bit 解碼〔摘要，[rocDecode 文件](https://rocm.docs.amd.com/projects/rocDecode/en/latest/reference/rocDecode-formats-and-architectures.html)〕；但 **H.265 無損模式（RExt）能不能在 VCN 解碼、MI300X 有沒有編碼器：NOT_VERIFIED**。程式碼：LMCache v0.3.7＋vLLM v0.10.2 的外掛，約 5K 行，**沒找到釋出連結**。〔推論〕期限內不可重現。

#### B6　HACK：不反量化、直接在量化 KV 上算〔部分〕
- **出處**：Zeyu Zhang、Haiying Shen、…、Minlan Yu。arXiv [2502.03589](https://arxiv.org/abs/2502.03589)（2025-02-05）。arXiv 頁沒有 venue。
- **問題**：PD 分離時 KV 傳輸是瓶頸；量化省了傳輸，但**反量化本身很貴**〔原文〕。
- **方法**：KV 2-bit、Q 8-bit、attention 機率 INT8，矩陣乘法直接在量化資料上做（同態），跳過反量化〔原文 §5.1〕。
- **他們量到的反量化成本（原文 §2.2、§7.2）**：CacheGen、KVQuant 的反量化最多佔 JCT 37.9%；A10G prefill＋Llama-3.1-70B＋Cocktail 時 17.2–30.4%；不同 GPU 26.4–37.9%。
- **改善（原文）**：JCT 比 PD 分離基線少最多 70.9%、比 KV 量化法少最多 52.3%；準確率比基線少 0.76–1.56%（Π＝64）。硬體：AWS 上 4×A10G（40 Gbps）、4×V100（10 Gbps）、4×T4（50 Gbps）、4×L4（40 Gbps）做 prefill，8×A100（400 Gbps）做 decode；模型 Mistral-7B 到 Falcon-180B；Cocktail 平均 16.2K token。
- **限制**：2-bit 要小分區才準，又會拉長 JCT；Triton 只支援 INT8 運算、有執行期開銷（原文 §8）。
- **ROCm**：vLLM＋FlashAttention-2＋Triton，程式碼是匿名 repo（anonymous.4open.science/r/HKVQ）〔原文〕。〔推論〕Triton INT8 在 MI300X 可行，但要換 attention 路徑，**違反本專案「反量化後走同一條 attention」的規則（07）**，只能當延伸。
- **對本專案的意思**：它把「反量化太貴」量化成 JCT 佔比，和我們 11 §3a–3d 的結論同方向；但我們已經用融合 kernel＋一次拆，把 CPU 層的反量化壓到 1.4–4.3 ms（11 §3d）。

#### B7　權重無損壓縮家族（DFloat11、ZipNN、ZipServ、Huff-LLM、Ecco）〔摘要〕
| 論文 | 出處 | 重點（原文摘要） | 和 KV 的關係〔推論〕 |
|:--|:--|:--|:--|
| DFloat11 | arXiv [2504.11651](https://arxiv.org/abs/2504.11651)，NeurIPS 2025 | BF16 權重熵編碼，約 30% 變小、輸出逐位元相同；GPU 解壓 kernel（SRAM 查表、兩階段執行緒協調）；比部分卸載到 CPU 的吞吐高 2.3–46.2× | 指數位元的冗餘在 KV 也存在（SplitZip 表 2：1.423×），但**編碼 0.004 GB/s**，不能用在線上寫入 |
| ZipNN | arXiv [2411.05239](https://arxiv.org/abs/2411.05239)，IEEE Cloud（arXiv 註記） | 常見省 33%、有時 >50%；比 vanilla 壓縮多省 17%、壓／解快 62% | **CPU 函式庫**（`pip install zipnn`；zstd／lz4／huffman；README：「Currently … implemented on CPUs」）。README：16 個 worker 解壓最多 80 GB/s（224 核 Xeon 8480+），單執行緒 BF16 解壓 1,660 MB/s、比 1.51（**權重**）→ **主機端可直接用來壓卸載層的 KV** |
| ZipServ | arXiv [2603.17435](https://arxiv.org/abs/2603.17435)，ASPLOS'26 | 定長編碼 TCA-TBE＋解壓融合進 GEMM；模型小最多 30%、kernel 對 cuBLAS 最多 2.21×、端到端平均 1.22× | 論證「可變長位元流和 SIMT 衝突」——這正是 SplitZip 選定長碼的理由 |
| Huff-LLM | arXiv [2502.00922](https://arxiv.org/abs/2502.00922) | 權重在雲端、磁碟、主記憶體、晶片內都保持壓縮；摘要沒有數字 | 硬體提案，和 KV 無直接關係 |
| Ecco | arXiv [2505.06901](https://arxiv.org/abs/2505.06901)，ISCA 2025 | 分組非均勻量化＋共用 k-means 樣板＋Huffman；平行 Huffman 解碼延遲降兩個數量級；對 AWQ 最多 2.9× | 有損＋硬體（L2 旁），不可在 MI300X 重現 |

#### B8　KVComp／PackKV：有損量化＋無損打包〔摘要〕
- KVComp：Bo Jiang 等，arXiv [2509.00579](https://arxiv.org/abs/2509.00579)（2025-08-30）。**有損**，記憶體縮減比既有方法平均多 47%、最多 83%；部分情況矩陣向量乘法比 cuBLAS 快〔原文〕。
- PackKV：同一團隊，arXiv [2512.24449](https://arxiv.org/abs/2512.24449)（v2 2026-01-07）。K、V 的記憶體縮減率比既有量化法平均高 153.2%、179.6%；在 A100 與 RTX Pro 6000 上，對 cuBLAS 矩陣向量乘法吞吐高 75.7%（K）、171.7%（V）〔原文〕。
- 和本專案〔推論〕：都是 GPU 內的有損格式，屬於第二階段的範圍；只列出備查。

#### B9　CacheGen 的傳輸面（指標）
- 機制、程式碼落差、現行 LMCache 只剩固定 codec：見 [P02](../../precision_20261013/cards/P02_CacheGen_mechanism.md)。arXiv [2310.07240](https://arxiv.org/abs/2310.07240)（SIGCOMM'24，abs 頁已核）。
- 這輪新增的外部批評：ShadowServe（B4）、KVCodec（B5）都量到 **CacheGen 的 GPU 解碼會拖慢同機的計算**（≥25–30%；prefill +50%、decode +20%）；KVCodec 還量到 CacheGen 為 4K token 預先配置 5.5 GB（原始 KV 的 2.7 倍）〔原文〕。

#### B10　系統實作的壓縮現況（2026-10）
| 系統 | 有沒有壓縮 | 出處 | 深度 |
|:--|:--|:--|:--|
| **LMCache** | v1 設定的 `remote_serde` 選項是 cachegen／torch／safetensor／fast；另一頁寫 naive（預設）或 cachegen——**兩頁版本不一致**；沒有 KIVI | [LMCache v1 設定](https://docs.lmcache.ai/configuration/v1/v1_config.html)、[CacheGen 頁](https://docs.lmcache.ai/kv_cache_optimizations/compression/cachegen.html) | 〔摘要〕 |
| **LMCache＋Intel QAT** | RFC [#4149](https://github.com/LMCache/LMCache/issues/4149)（2026-07-18，open）：主機端用 QAT／IAA 做 DEFLATE，NVIDIA Decompression Engine 可選；`truncate_bits=0` 時位元級無損，>0 時砍尾數低位（有損）。PR [#4264](https://github.com/LMCache/LMCache/pull/4264)（Intel，2026-07-27，**draft、未合併**；serde 名 `accel_kv_compress`）。**兩者都沒有任何壓縮比、吞吐、延遲數字**；作者說要等 LMCache MP 的 layerwise 支援之後再繼續 | GitHub | 〔全文〕 |
| **TensorRT-LLM** | **cold-page NVFP4**：GPU 上的 page 保持原 dtype，**卸載到 host／disk 時才編成 NVFP4，載回時解回**；只支援 SM100／SM103；文件沒給數字、也沒說編解碼在 GPU 還是 CPU | [TRT-LLM KV cache compression](https://nvidia.github.io/TensorRT-LLM/1.3.0rc27/features/kv-cache-compression.html) | 〔全文〕 |
| **NIXL** | 後端有 GDS、GDS_MTGPU、POSIX（io_uring 等）、OBJ（S3）；**壓縮：NOT_FOUND** | [NIXL backend selection](https://docs.nvidia.com/nixl/user-guide/backend-selection) | 〔摘要〕 |
| **Mooncake** | 傳輸引擎內建壓縮：**NOT_FOUND**；SplitZip 是在 Mooncake 傳輸上另外加 codec（研究原型） | B1 | 〔摘要〕 |
| **Dynamo KVBM**、**llm-d FS**、**vLLM OffloadingConnector**、**AMD AIC**、**NVIDIA ICMSP** | 文件都沒提到壓縮（見 §2） | §2 | 〔全文／摘要〕 |
| **UCCL-Zip** | 把無損壓縮融合進 GPU 通訊（P2P 的 split-send 管線、NCCL persistent kernel）；RL 權重同步最多快 47.5%、vLLM 端到端延遲最多少 10%；**沒提 AMD、沒提 KV** | arXiv [2604.17172](https://arxiv.org/abs/2604.17172) | 〔摘要〕 |
| **ROCm 上的 GPU 無損函式庫** | 沒有官方的 nvCOMP 對應品〔摘要〕。學術移植：Künas 等把 LZ4、Snappy、Cascaded 移植到 HIP，在 MI50／MI210／MI300X／RX 7900 XT 上量：MI300X 解壓吞吐最多是 MI50 的 11×，但**真實浮點資料（地震模擬，不是 KV）的無損比只有 1.03–1.04×**、端到端被搬運主導（Springer 2027，DOI 10.1007/978-3-032-30491-9_23；[eScholarship](https://escholarship.org/uc/item/5hs138d9)）。第三方 DietGPU 的 HIP 版 hipANS（PAA-NCIC）：**NOT_VERIFIED** | 〔摘要〕 |

### 1.4 小結：ROCm 上能直接用的
1. **主機端（CPU）無損**：ZipNN（zstd／lz4／huffman，BF16／FP8）〔README〕；或標準 zstd／lz4。本機 CPU 配額 32 核（`docs/MI300X_MLSTEAM.md` §7.5.1）。**要裝套件，需使用者同意**。
2. **GPU 端無損**：SplitZip 的 Triton kernel（移植未驗證）；或照它的格式自己寫一個 top-16 指數查表解碼（和我們已有的 FP8 查表反量化 kernel 同一個形狀）〔推論〕。
3. **不可用**：nvCOMP（KVServe、KVTC 的熵編碼段）、NVDEC（KVCodec）、BlueField（ShadowServe）、Intel QAT（LMCache PR；主機 CPU 型號沒查，但 PR 本身沒合併）。

### 1.5 盈虧分析：誰做過、代入本機數字會怎樣

**已經有的公式與實測門檻**

| 來源 | 形式 | 條件 | 深度 |
|:--|:--|:--|:--|
| KVServe Eq. 5／Thm 6.1 | B\*＝(1−1/cr)·s；鏈路比 B\* 快就不要壓 | 編解碼和傳輸**加總**、不重疊 | 〔部分〕 |
| SplitZip App. B | B_hide＝min(G_enc, G_dec)/ρ；鏈路比它慢時 codec 時間被管線藏住 | 編解碼和傳輸**重疊** | 〔部分〕 |
| ShadowServe §6.2 | ≤20 Gbps 贏 1.20–1.38×；>20 Gbps 輸 11–24% | 實測，L40S＋BF3 | 〔部分〕 |
| KVCodec §5.2 | <40 Gbps 對 CacheGen 1.29–3.50×，頻寬越高優勢越小 | 實測，A100／H20／L20 | 〔部分〕 |
| KVServe §2.2 | 3 種方法的門檻 50／55／110 Gbps | 實測 | 〔部分〕 |
| py-kvcache | 「存 SSD 比重算划算」的損益平衡長度 | 實測，vLLM＋NVMe（`SOTA_ANALYSIS` §2.9） | 〔摘要〕 |
| **本專案 stage 44** | INT4 對 BF16 的損益兩平：Llama・忙 18.1、Qwen・閒 29.9、Qwen・忙 17.6 GiB/s；Llama・閒到 CPU 層都還在省 | 實測（08 §2.8；run `20261018-154708-m10-tier2-llama`、`20261018-160950-…-qwen`） | 〔實測〕 |
| **本專案 11 §3c** | Cake 的增益隨「只載÷只算」的比而變；壓縮會移動這個比（b32：BF16 增益 1.98× → INT4 1.35×，但 INT4 絕對時間仍快 1.38→0.68 s） | 實測格子 | 〔實測〕 |

**代入本機的層頻寬**〔算術〕（層頻寬取 `results/m7_write_policy_mi300x/tier_params.json`：CPU 35.4、SSD 級 6.98、NFS 級 0.331 GiB/s；無損比 cr 暫用 SplitZip 在 Llama-3-8B 量到的 1.32，**不是我們的實測**）：

- KVServe 的 B\*＝(1−1/1.32)·s＝0.242·s。要讓壓縮在某一層「不虧」，codec 吞吐 s 至少要：CPU 層 146 GiB/s、SSD 級 28.8 GiB/s、NFS 級 1.37 GiB/s。
  - GPU 端：SplitZip 在 H200 解碼 2,182 GB/s，遠超過；**但這是 GPU 閒時的數字**。我們的反量化在 GPU 忙時慢 2.4–3.8 倍，而且主要成本是 kernel 排隊，不是吞吐（11 §3d）。〔推論〕CPU 層是否划算，取決於忙時的排隊，**公式算不出來，要量（X4）**。
  - CPU 端：ShadowServe 量到單一 CPU 核 DEFLATE 解壓 2.5 Gbps（≈0.29 GiB/s）。在加總模型下 NFS 級要約 5 個核；在重疊模型下解碼只要跟上 0.331×1.32≈0.44 GiB/s 的輸出，約 2 個核。ZipNN README 的單執行緒 BF16 解壓 1,660 MB/s（權重、不同 CPU）。〔推論〕**NFS／SSD 級用 CPU 解壓是「免費」的：不搶 GPU，32 核配額足夠**。
- **但「不虧」≠「有用」**。NFS 級 32K 的還原（stage 44 實測，GPU 閒）：

  | 模型 | 全在 CPU（E0） | 全在 NFS 級 BF16 | 全在 NFS 級 INT4（位元組 1/3.2） | NFS÷CPU（BF16→INT4） |
  |:--|--:|--:|--:|:--|
  | Llama | 0.094 s | 2.64 s | 1.72 s | 28× → 18× |
  | Qwen | 0.052 s | 1.79 s | 0.98 s | 34× → 19× |

  位元組少 3.2 倍，還原只快 1.5–1.8 倍，因為 Cake 在慢層本來就重算大部分 block（b48 限速層的會合點 37–48，08 §2.1）。〔推論〕1.32× 的無損改變更小；**沒有任何壓縮能把 NFS 拉到 CPU 附近**。最多只能像 B3 說的，20× 有損把 NFS 變成 SSD 級（而 SSD 級全 BF16 仍是 CPU 的 5 倍）。
- **壓縮在 NFS 上真正的用途：縮短預取提前量**〔算術〕。把 32K 的整段從 NFS 級搬到 DRAM 要的時間：

  | 模型 | BF16 | 無損 ×1.32（假設） | FP8 | INT4 KIVI |
  |:--|--:|--:|--:|--:|
  | Llama（4 GiB） | 12.1 s | 9.2 s | 6.0 s | 3.8 s |
  | Qwen（1.75 GiB） | 5.3 s | 4.0 s | 2.6 s | 1.7 s |

  128K 時 Llama BF16 15.75 GiB → 47.6 s、Qwen 6.89 GiB → 20.8 s〔算術，大小見 08 §2.8 A1〕。如果使用者／工具的「回來間隔」比這些長，預取就能把 NFS 完全藏起來，還原時間＝CPU 層。**這把 G1 變成「間隔分布 × 提前量」的問題**（§4 G1）。

---

## 2. C：卸載、分層記憶體、SSD／NFS——只補新的與缺的

### 2.1 已經整理過、這裡不重做的（指標）

| 論文／系統 | 在哪裡 |
|:--|:--|
| Cake、CacheFlow、Strata／HiCache、Pensieve、Bidaw、MTDS、AdaptCache、EvicPress、LMCache、Mooncake、py-kvcache、AsymCache、When Fancy Eviction Fails、HCache、KVPR | `docs/SOTA_ANALYSIS_20261006.md` §2 |
| CachedAttention（ATC'24）、Tutti、其他分層系統的評測設定 | `docs/research_20261006_eval/cards/E04_tiered_systems.md` |
| DualBlade、HyMCache、IMPRESS、Krul、Lachesis、TierKV、Where KV Live、MPIC、Marconi、Jenga、MORI、EfficientAgent、HBF LRU-K 等 20 張 | `docs/research_20261009_explore/cards/A_*.md` |
| KVDrive、KVServe、Het-Mem、LeoAM、OrbitFlow | `docs/RELATED_WORK_WEAKNESSES.md`、`docs/PAPER_GUIDE.md` |
| AMD Infinity Context（擷取） | `docs/research_20261009_explore/lit_B_summary.md` 等 |
| vLLM OffloadingConnector、KVBM、llm-d（提及） | 多處；本節補官方數字與機制 |

### 2.2 新補的總表

| # | 論文／系統（年，venue） | 分層 | 壓縮？ | 重點數字（原文） | MI300X／ROCm | 深度 |
|:--|:--|:--|:--|:--|:--|:--|
| C1 | vLLM OffloadingConnector（部落格 2026-01-08） | GPU→CPU | 否 | 0.12.0 起 TTFT 最多少 4×、吞吐 5×；CPU 載回 TTFT 2–22× | 部落格寫支援 NVIDIA 與 AMD | 〔全文〕 |
| C2 | llm-d FS backend（IBM，2026-02-10） | GPU→（CPU）→共享檔案系統 | 否 | 長 prompt 對 prefill 最多 16.8× | 沒提 AMD | 〔全文〕 |
| C3 | AMD ROCm Infinity Context＋hipFile（2026-07-22） | HBM→DRAM→RDMA 網路儲存 | 否 | **沒有效能數字** | 原生（MI300X 支援）；hipFile 依 AMD 部落格 ROCm 7.14 GA | 〔全文〕 |
| C4 | NVIDIA Dynamo KVBM（文件）、ICMSP（新聞稿 2026-01-05） | G1–G4／HBM→DRAM→乙太網 flash | 否 | ICMSP：tokens/s 與能效最多 5×（廠商） | CUDA／BlueField | 〔摘要〕 |
| C5 | TensorRT-LLM cold-page NVFP4 | 卸載到 host／disk 才壓 | **是（有損）** | 文件沒數字 | SM100／103 only | 〔全文〕 |
| C6 | DirectKV（OSDI'26） | GPU 直接讀 CPU 記憶體（NVLink-C2C） | 否 | 傳輸量最多少 50%、GPU 記憶體少 43%、端到端 1.2× | GH200／GB200 | 〔摘要〕 |
| C7 | Beluga（SIGMOD'26）、CXL-SpecKV（FPGA'26） | CXL 記憶體池 | Beluga 否；CXL-SpecKV 有（FPGA） | Beluga：TTFT 少 89.6%、吞吐 7.35×；CXL-SpecKV：吞吐最多 3.2× | 需 CXL 交換器／FPGA | 〔摘要〕 |
| C8 | KVSwap（arXiv 2025） | 裝置端：全部 KV 放磁碟 | 否 | 摘要沒數字 | 手機／嵌入式 | 〔摘要〕 |
| C9 | FlexGen（ICML'23）、LayerKV（arXiv 2024） | GPU／CPU／磁碟；逐層 | FlexGen 4-bit | FlexGen：16 GB GPU 跑 OPT-175B 1 token/s；LayerKV：TTFT 最多 69× | 純 PyTorch（FlexGen）〔推論〕 | 〔摘要〕 |

### 2.3 卡片

#### C1　vLLM OffloadingConnector（官方部落格）〔全文〕
- **出處**：[vLLM blog, 2026-01-08](https://vllm.ai/blog/2026-01-08-kv-offloading-connector)。
- **問題**：GPU KV 不夠時卸載到 CPU，讓 CPU 命中能省重算。
- **機制（原文）**：CPU 後端用 `cudaMemcpyAsync`，走 GPU 的 DMA 引擎，「minimal overhead on the CPU and GPU cores」；另一種是用 GPU 核心逐 16 位元組拷貝的自訂 kernel，「creates greater interference」。0.9.0 加入非同步 load／store；卸載是非同步的，請求不用等 GPU→CPU 拷貝完成。
- **關鍵數字（原文）**：
  - 0.12.0 把每層分散的 KV 改成跨所有層連續的一塊（PR #27743），實體 block 大小 ×(2×層數)（Llama-3.1-8B：32 KB → 2 MB），吞吐「提升一個數量級」。
  - 2 MB block 雙向：DMA 83.4 GB/s、自訂 kernel 68.5 GB/s；單向兩者約 50 GB/s。小 block 時自訂 kernel 較快但變異大。
  - **併發時的干擾**（H100、Llama-3.2-1B，10,000 個 512 token 請求）：DMA 比自訂 kernel 吞吐好約 5.5%（0% 命中）到 15%（80% 命中）；**自訂 kernel 在 0% 命中時比不卸載還差 6%**，原文歸因於和模型計算搶資源。Llama-3.1-8B：DMA 吞吐最多多 32%，TTFT 相同。
  - Llama-3.1-8B、H100：CPU 載回讓單請求 TTFT 少 2–22×（依 prompt 長度），吞吐隨 CPU 命中率最多 9×。
- **限制（原文）**：hybrid 模型沒最佳化；磁碟層是「下一個里程碑」；搶占後從 CPU 重載預計 0.14.0；有一個卸載與計算的 race 在 PR #31341 修。
- **MI300X**：原文說支援「CUDA-compatible GPUs (NVIDIA and AMD)」。〔推論〕ROCm 上 `hipMemcpyAsync` 是否一定走 SDMA，取決於 ROCr 的 `HSA_ENABLE_SDMA`（預設 1）與 HIP 的實作細節（blit kernel 的回退條件 **NOT_VERIFIED**）→ X3。
- **對本專案**：這是「搬運走 DMA 還是 kernel」在 KV 卸載上**唯一找到的併發量測**，而且是 NVIDIA。MI300X 版本沒人量（G2）。

#### C2　llm-d FS backend：卸載到任何共享檔案系統〔全文〕
- **出處**：Kfir Toledo、Danny Harnik、Effi Ofer、Or Ozeri、Guy Margalit（IBM）。[llm-d blog, 2026-02-10](https://llm-d.ai/blog/native-kv-cache-offloading-to-any-file-system-with-llm-d)。
- **機制（原文）**：接在 vLLM 原生 OffloadingConnector 上，KV block 以檔案存在共享檔案系統，目錄結構就是索引；非同步讀寫、多執行緒；預設用 GPU DMA。**GDS 列為未來工作**；儲存→GPU「通常多一跳 CPU DRAM」。block 大小以 token 計、可調。
- **數字（原文）**：Llama-3.1-70B、4×H100、IBM Storage Scale：長 prompt 對 prefill 最多 16.8×；單請求比 GPU／CPU 快取慢。另有一組雲端儲存約 10 GB/s I/O 的混合負載（只有圖）。測過 Storage Scale、本地 NVMe、CephFS；**沒測 NFS、Lustre**。
- **限制（原文）**：單請求贏不了 DRAM；是 CPU 卸載的補充；分層（CPU 之後再到儲存）、NIXL、物件儲存、GDS 都是未來工作。
- **MI300X**：沒提 AMD。〔推論〕走 OffloadingConnector＋POSIX，理論上可在 ROCm 跑，NOT_VERIFIED。
- **對本專案**：llm-d 的 v0.7 文件與 well-lit-paths 頁對「GPU→CPU→Storage 分層是否已上線」說法不一致〔摘要〕，引用時要寫版本。

#### C3　AMD ROCm Infinity Context（AIC）＋hipFile〔全文〕
- **出處**：Juergen Frick、Stephen Bates，[ROCm blog, 2026-07-22](https://rocm.blogs.amd.com/software-tools-optimization/amd-infinity-context/README.html)；[hipFile 文件](https://rocm.docs.amd.com/projects/hipFile/en/latest)、[GitHub ROCm/hipFile](https://github.com/ROCm/hipFile)。
- **內容（原文）**：在 HBM、DRAM 之下加一層 RDMA 網路儲存（NVMe 後端）；**hipFile** 讓資料在 HBM 與儲存之間直接搬、不經主機記憶體匯流排；堆疊：vLLM（SGLang 計畫中）、LMCache（HiCache 計畫中）、llm-d、NIXL（Mooncake 計畫中）。支援 MI300X、MI350（前端乙太網）。參考實作 [github.com/ROCm/rocm-aic](https://github.com/ROCm/rocm-aic)。**沒有任何效能數字、沒提壓縮**。
- **hipFile 的成熟度：來源互相矛盾**〔摘要〕：AMD 部落格說 ROCm 7.14（2026-07）GA；GitHub README 寫「early-access software technology preview」、不建議生產使用；ROCm 儲存函式庫頁列 hipFile 0.4.0；第三方新聞稱 ROCm 10.1 加了非同步 fast path 與 batch API。API 對應 cuFile（HIPIFY 有 cuFile→hipFile 對照）。
- **MI300X 可重現性**〔推論〕：我們的容器是 ROCm 7.2.2（`docs/MI300X_MLSTEAM.md` §1），**沒有 hipFile**；要升級 ROCm 才能試，期限內不建議。本地碟是 MegaRAID volume、持久掛載是 NFS（`DIRECTIONS_20260924.md`），GPU 直通儲存的前提（NVMe＋P2P DMA）也不確定。
- **對本專案**：「ROCm 有沒有 GDS 等價物」的答案是**有（hipFile），但在我們的環境不可用**；寫論文時要說「NFS／SSD 路徑都經過主機 DRAM」，這和 CachedAttention、llm-d、KVBM（無 GDS 時）的路徑一致。

#### C4　NVIDIA Dynamo KVBM 與 ICMSP〔摘要〕
- **KVBM**（[Dynamo 設計文件](https://docs.nvidia.com/dynamo/v1.0.0/design-docs/component-design/kvbm-design)，多版本一致）：G1 GPU、G2 CPU（pinned，可從 GPU 卸載、載回、再卸到磁碟）、G3 本地／池化 SSD（經 NIXL，POSIX 寫，有 GDS 時用 GDS）、G4 遠端（不透明 blob；v1.1.1 有 NFS／Lustre／GPFS 的 POSIX 變體）。**壓縮與卸載過濾：這輪搜尋 NOT_FOUND**（lit_A 曾記錄「磁碟過濾預設開」，本輪未重核）。
- **ICMSP**（Inference Context Memory Storage Platform；NVIDIA 新聞稿 2026-01-05，[investor.nvidia.com](https://investor.nvidia.com/news/press-release-details/2026/NVIDIA-BlueField-4-Powers-New-Class-of-AI-Native-Storage-Infrastructure-for-the-Next-Frontier-of-AI/default.aspx)）：BlueField-4 驅動的 KV 專用儲存層；廠商宣稱 tokens/s 與能效最多 5×。第三方描述為「G3.5」乙太網 flash 層。**沒有獨立量測**，第三方部落格的 13×、11× TTFT 是部落格自己估的〔摘要〕。
- **對本專案**：業界把「HBM→DRAM→網路 flash」標準化了，**但都是搬原始位元組**；這是「卸載層壓縮」在系統面仍是空白的旁證（B10）。

#### C5　TensorRT-LLM 的 cold-page NVFP4〔全文〕
- 見 B10。重點：**延後量化**（卸載到冷層時才壓）已經進了一個主流引擎，正是我們 12 號「分層：CPU 存 BF16、降到慢層才轉 INT4」的設計（12 §0）。差別：它是 NVFP4、限 Blackwell、沒有數字；我們有 MI300X 上的實測（08 §2.8 B）。寫論文時這是**必須引用的系統前作**。

#### C6　DirectKV：零拷貝 KV 卸載（OSDI'26）〔摘要〕
- **出處**：Shutian Luo、Haiying Shen，"No Buffer, No Bottleneck: Efficient Zero-Copy KV Cache Offloading for Long-Context LLMs"，OSDI 2026（[papers.cool 列表](https://papers.cool/venue/luo@osdi26@USENIX)；USENIX PDF 存在但本機無法解析文字）。**沒找到 arXiv 版**。
- **方法（摘要）**：在 GH200／GB200 上，GPU kernel 透過 NVLink-C2C 直接讀 CPU 上的 KV，拿掉 GPU 暫存緩衝；KV 生成和 attention 融合成一個 CUDA kernel，warp 級管線重疊讀取、計算、寫回。
- **數字（摘要，GH200）**：CPU–GPU 傳輸量最多少 50%、GPU 記憶體少 43%、端到端最多 1.2×。
- **MI300X**〔推論〕：MI300X 和主機之間是 PCIe，沒有 NVLink-C2C 級的一致性頻寬；零拷貝（GPU 直接讀 pinned 主機記憶體）在 ROCm 上做得到，但頻寬受 PCIe 限制。**不建議期限內做**。

#### C7　CXL 記憶體池：Beluga、CXL-SpecKV〔摘要〕
- **Beluga**：Xinjun Yang 等 12 人，arXiv [2511.20172](https://arxiv.org/abs/2511.20172)，**SIGMOD'26**。GPU 與 CPU 經 CXL 交換器共用大記憶體池，原生 load/store；對 vLLM 裡 RDMA 方案 TTFT 少 89.6%、吞吐 7.35×〔原文〕。
- **CXL-SpecKV**：Dong Liu、Yanxuan Yu，arXiv [2512.11920](https://arxiv.org/abs/2512.11920)，**FPGA'26 Oral**。KV 卸到遠端 FPGA 記憶體、**推測式預取**、FPGA 壓縮／解壓；吞吐最多 3.2×、記憶體成本少 2.8×、壓縮讓頻寬需求最多少 4×；摘要沒說是實機還是模擬〔原文〕。
- **對本專案**：硬體不可得；但 CXL-SpecKV 是少數**同時有「壓縮＋預取＋卸載」**的設計（G4 的近親），而且壓縮放在 GPU 以外——和 ShadowServe（SmartNIC）、LMCache QAT（主機加速器）同一個思路：**解壓不要搶 GPU**。

#### C8　KVSwap：裝置端的磁碟 KV 卸載〔摘要〕
- Huawei Zhang、Chunwei Xia、Zheng Wang，arXiv [2511.11907](https://arxiv.org/abs/2511.11907)（v2 2025-12-11）。全部 KV 放磁碟，記憶體留精簡的 metadata 預測要預載哪些；計算和磁碟存取重疊；讀取版面配合儲存特性。摘要沒有數字〔原文〕。和本專案只在「慢儲存＋預測預載」有交集。

#### C9　經典：FlexGen、LayerKV〔摘要〕
- **FlexGen**：Ying Sheng 等，arXiv [2303.06865](https://arxiv.org/abs/2303.06865)，ICML 2023（PMLR v202 已核）。GPU／CPU／磁碟一起用，線性規劃決定 tensor 放哪、怎麼存取；權重與 attention cache 壓到 4-bit；單張 16 GB GPU 跑 OPT-175B 首次到 1 token/s（有效 batch 144）〔原文〕。它是「放置＋壓縮＋重疊」三合一的最早範本，但目標是**延遲不敏感的批次吞吐**。
- **LayerKV**：Yi Xiong 等，arXiv [2410.00428](https://arxiv.org/abs/2410.00428)（v3 2024-10-09）。逐層的 KV block 配置、管理、卸載＋SLO 感知排程；7B–70B，TTFT 最多 69×、SLO 違反率少 28.7%〔原文〕。
- 已收的 HCache、KVPR、Tutti、CacheFlow 在 D 節從「重疊」的角度再看一次。

### 2.4 ROCm 上「GPU 直通儲存」與相關旋鈕的結論
| 問題 | 答案 | 出處 |
|:--|:--|:--|
| ROCm 有沒有 GDS 等價物？ | 有：hipFile（cuFile 相容）；GA 時間來源矛盾（ROCm 7.14 vs early-access） | C3 |
| 我們能不能用？ | 容器 ROCm 7.2.2，**不能**（要升級）〔推論〕 | `docs/MI300X_MLSTEAM.md` |
| 卸載搬運走 DMA 還是 kernel？ | ROCr `HSA_ENABLE_SDMA`（預設 1）控制 hsa_memory_copy、hsa_amd_memory_async_copy 等 API 是否用 DMA 引擎；HIP `hipMemcpyAsync` 何時回退到 blit kernel：**NOT_VERIFIED** | [ROCr 環境變數](https://rocm.docs.amd.com/projects/ROCR-Runtime/en/master/api-reference/environment_variables.html) |
| MI300X 有幾個 copy engine？ | 14 個 SDMA（IOD 上），在 L2 之外 | D9（ConCCL §II-A） |

---

## 3. D：預取、非同步搬運、計算與搬運重疊

### 3.1 總表

| # | 論文（年，venue） | 預取什麼、從哪裡 | 提前多少 | 有沒有量 GPU 忙／干擾 | MI300X／ROCm | 深度 |
|:--|:--|:--|:--|:--|:--|:--|
| D1 | InfiniGen（OSDI'24） | decode 時重要的 KV，CPU→GPU | 一層 | 否 | 建在 FlexGen 上 | 〔部分〕 |
| D2 | ShadowKV（ICML'25 spotlight） | V 在 CPU、低秩 K 在 GPU，稀疏取回 | 每步 | 否 | **CUDA／CUTLASS／FlashInfer** | 〔摘要＋README〕 |
| D3 | FreeKV（arXiv 2025）、SpeCache（ICML'25）、DualDecoder（arXiv 2026） | 推測式取回 | 下一步 | FreeKV：沒有干擾分析 | FreeKV 另有昇騰 NPU 版 | 〔部分／摘要〕 |
| D4 | KVPR（ACL Findings'25） | 部分 activation＋其餘 KV，CPU→GPU，邊傳邊重算 | 每層 | 否（假設硬體不變） | CUDA streams | 〔部分〕 |
| D5 | HCache（EuroSys'25） | hidden state 從 SSD（SPDK＋GDRCopy），層切分 | 整段還原 | 否（只報 TBT +4%） | 只 MHA；CUDA | 〔部分〕 |
| D6 | LMCache layerwise、CacheBlend、CachedAttention | 逐層載入與計算重疊；排程感知的磁碟→DRAM 預取 | 一層／佇列前瞻 | 否 | LMCache 的 ROCm：NOT_VERIFIED | 〔文件／指標〕 |
| D7 | KVFlow（NeurIPS'25） | 下一步要跑的 agent 的 KV，CPU→GPU | 一步 | **主張不干擾**（PCIe 全雙工）；預取比執行久時會卡 | SGLang 0.4.4 | 〔部分〕 |
| D8 | Tokencake（EuroSys'27）、Continuum（arXiv） | 工具呼叫期間卸出、預測回來時間再載回；或 TTL 保留 | 依工具時間預測 | Tokencake：**沒量** | vLLM | 〔部分／摘要〕 |
| D9 | ConCCL、DMA-Latte（MICRO'26）、FiCCO | 通訊卸到 **MI300X 的 SDMA** | — | **有**：C3 只得理想的 21%；DMA 後 72% | **MI300X 實機** | 〔部分／摘要〕 |
| D10 | Lee et al.（arXiv 2025） | 訓練的計算／通訊重疊 | — | **有**：重疊平均拖慢計算 18.9%、最多 40% | 含 MI250、MI210 | 〔摘要〕 |
| D11 | vLLM DMA vs kernel 拷貝 | 見 C1 | — | **有**：kernel 拷貝比不卸載差 6% | 部落格稱支援 AMD | 〔全文〕 |
| D12 | PRESERVE（arXiv 2025） | 權重與 KV **從 HBM 到晶片內快取**（不是主機卸載） | 通訊期間 | — | 昇騰 | 〔摘要〕 |

**「KV Infinity」**：搜尋 NOT_FOUND。最接近的是 InfiniGen（D1）與 DualDecoder（D3）。

### 3.2 卡片

#### D1　InfiniGen：推測下一層要哪些 KV，只預取那些〔部分〕
- **出處**：Wonbeom Lee、Jungi Lee、Junghwan Seo、Jaewoong Sim。arXiv [2406.19707](https://arxiv.org/abs/2406.19707)，**OSDI 2024**。
- **方法與假設（原文）**：在第 i−1 層用該層的輸入、部分 query 權重與部分 key cache「預演」第 i 層的 attention，**提前一層**預取重要的 KV；離線用 SVD 對 Q／K 權重做斜交轉換（數學上不改輸出）。平均只用到 <10% 的 KV，每層最多送 20%。
- **數字（原文，RTX A6000 48 GB、PCIe 3.0 x16、96 GB DDR4）**：對既有 KV 管理法最多 3.00×；OPT-13B、batch 20：1.63–32.93×（32.93× 含 UVM，不公平）；序列長度掃描對 FlexGen 最多 5.28×。
- **限制**：只最佳化 decode，prefill 沒處理；每個模型要離線 SVD 與調參；只有一個 PCIe 3.0 平台；推測會漏 token。**沒有 GPU 忙時的量測**。
- **ROCm**：建在 FlexGen（PyTorch）上〔原文〕；程式碼釋出狀態在讀到的部分沒寫（NOT_VERIFIED）。
- **對本專案**：它是「decode 時從 CPU 稀疏取回」，和我們「prefill 前整段還原」不是同一個問題；只在「提前一層」的時間尺度上可借。

#### D2　ShadowKV：低秩 K 留 GPU、V 卸到 CPU〔摘要＋README〕
- **出處**：Hanshi Sun 等 9 人（ByteDance Seed、CMU）。arXiv [2410.21465](https://arxiv.org/abs/2410.21465)（v3 2025-04-25）；**ICML 2025 Spotlight**（ByteDance Seed 部落格與 OpenReview 列表）。
- **數字（原文）**：A100 上 batch 最多大 6×、吞吐最多 3.04×，準確率不掉；Llama-3.1-8B、Llama-3-8B-1M、GLM-4-9B-1M、Yi-9B-200K、Phi-3-Mini-128K、Qwen2-7B-128K。
- **實作（README）**：需要 flash-attn、FlashInfer（cu121／torch2.3 wheel）、CUTLASS、自建 kernel；只寫 A100。**ROCm 沒提，CUDA 專用** → 期限內不可重現。
- **旁證**：FreeKV（D3）報告 ShadowKV 在 Qwen-2.5 上有重複輸出，歸因於重建的 K 有誤差〔原文〕——又一個「Qwen 的 K 不好壓」的獨立觀察。

#### D3　推測式取回：FreeKV、SpeCache、DualDecoder
- **FreeKV**〔部分〕：Guangda Liu 等，arXiv [2505.13109](https://arxiv.org/abs/2505.13109)（v5 2026-03-09）。第 i 步沿用第 i−1 步選出的 KV（相鄰 query 餘弦相似度 >0.84），把選擇與取回移出關鍵路徑；相似度低於門檻才修正（修正率 0.04–0.52）；GPU 用 NHD、CPU 用 HND 的混合版面（傳輸單位從 256 B 變 8 KB，最多 10.5×），雙緩衝串流取回。A100 40 GB＋PCIe Gen4；對 ArkVale 最多 13.7×、ShadowKV 8.4×、InfiniGen 3.2–8.5×。**沒有干擾分析**（原文自己列為缺）。另有**昇騰 910B 版**（AscendC），對 ArkVale 最多 4.1×——**證明這類設計能搬到非 CUDA 平台**〔原文 App. D〕。程式碼 github.com/sjtu-zhao-lab/FreeKV。
- **SpeCache**〔摘要〕：Shibo Jie 等，arXiv [2503.16163](https://arxiv.org/abs/2503.16163)，**ICML 2025**（PMLR v267）。完整 KV 在 CPU，GPU 留低位元副本估重要度，**推測下一個 token 會看誰並在計算時預取**；LongBench、NIAH 上 10× 壓縮比仍不失資訊〔原文〕。
- **DualDecoder**〔摘要〕：Zuning Liang 等，arXiv [2607.26475](https://arxiv.org/abs/2607.26475)（2026-07-29）。用推測出的前一個 token 預測下一步要的 KV 並預取，雙 token 解碼管線、逐層傳輸排程；decode 吞吐最多 2.62×〔原文〕。
- **對本專案**〔推論〕：三篇都是 decode 時的稀疏取回，都假設 GPU 只跑自己；**和「整段還原」的關係是：它們證明了「提前一步」的預測在 decode 尺度上準**，但我們要的是「提前幾秒到幾十秒」（§1.5 的提前量），預測對象是「使用者／工具何時回來」，不是「下一個 token 看誰」。

#### D4　KVPR：傳一部分 activation，GPU 邊重算邊收其餘 KV〔部分〕
- **出處**：Chaoyi Jiang、Lei Gao、Hossein Entezari Zarch、Murali Annavaram。arXiv [2411.17089](https://arxiv.org/abs/2411.17089)（v2 2025-06-04），**ACL Findings 2025**。
- **方法（原文）**：每層先傳一部分 activation 讓 GPU 重算前段 KV，其餘 KV 同時傳；profiler 量 PCIe 與 GPU 速度，解一個整數變數的最佳化定切點 ℓ；CUDA streams、雙緩衝、pinned 記憶體。每層時間＝activation 傳輸＋max(重算, KV 傳輸)，**假設兩者並行**。
- **數字（原文，A100 40 GB、PCIe 4.0 x16 32 GB/s）**：decode 延遲比 HF Accelerate 少最多 35.8%；吞吐對 FlexGen 最多 15.1%（OPT-6.7B）、46.2%（13B）、29.0%（30B）；GPU 使用率 85%→99%。batch 1、KV 最小時 FlexGen 反而贏（Table 2）。
- **限制（原文 §7）**：只在一開始量一次、假設硬體不變；**不處理磁碟或網路儲存**；不支援模型／張量平行。程式碼 [github.com/chaoyij/KVPR](https://github.com/chaoyij/KVPR)。
- **對本專案**：這是「Cake 的 decode 版」；它的「重算與傳輸並行」假設在 GPU 忙時會壞，和我們 13 號觀察到的 Cake 變慢是同一件事〔推論〕。

#### D5　HCache：從 hidden state 還原、層切分的無氣泡排程〔部分〕
- **出處**：Shiwei Gao、Youmin Chen、Jiwu Shu。arXiv [2410.05004](https://arxiv.org/abs/2410.05004)，**EuroSys 2025**。已收（`SOTA_ANALYSIS` §2.12）；這裡只補重疊與干擾。
- **排程（原文 §4.1.2）**：**逐層**切：前 L_O 層用互補方法（算力強的平台用 token 重算、I/O 強的用 KV 載入），其餘 L_H 層用 hidden state；有封閉解；離線依平台 profile。逐 token 切比逐層切慢 12%（取整後 7%），因為 GEMM 時間對 token 數不是線性（§6.3.2）。
- **I/O（原文）**：64 token 一塊、輪流分散到多顆 SSD；寫入兩段式（cudaMemcpy 快照到主機、8 個執行緒組塊寫 NVMe）；還原用 SPDK＋GDRCopy 直接進 GPU。4×A100-40G、4×PM9A3（單顆讀 6.9 GB/s）。
- **干擾**：**沒有專門的爭用實驗**；只報 TBT 最多比不還原多 4%〔原文 §6.1.1〕。
- **限制**：只 MHA（GQA／MQA 要低秩投影，原文說超出範圍）；A30 上對 KV 卸載只 1.33×；沒排程時可能比 KV 卸載慢 13%。
- **ROCm**：DeepSpeed-MII v0.2.0＋CUDA／C++，5,731 行；沒找到釋出連結。我們的模型都是 GQA → 不適用〔推論〕。

#### D6　逐層載入與排程感知預取：LMCache layerwise、CacheBlend、CachedAttention
- **LMCache layerwise**〔文件〕：`use_layerwise: true`；先載第 0 層，之後每層「等待」推進下一層；**計算與搬運用不同 CUDA stream**，GPU 只需要一層大小的緩衝（[LMCache 文件](https://docs.lmcache.ai/kv_cache_optimizations/layerwise.html)；技術報告 arXiv [2510.09665](https://arxiv.org/abs/2510.09665)）。vLLM 端開 layerwise 時要用 PIECEWISE CUDA graph〔摘要〕。
- **CacheBlend**：arXiv [2405.16444](https://arxiv.org/abs/2405.16444)（abs 已核；EuroSys'25 依本專案既有卡片）。選擇性重算與載入做成管線（本專案多處已收）。
- **CachedAttention**（ATC'24，arXiv [2403.19708](https://arxiv.org/abs/2403.19708)）：**依 job queue 的前瞻視窗把即將執行的 session 從磁碟預取到 DRAM**；逐層預載（PL-B15 −61%）。已收於 E04 卡。
- **對本專案**〔推論〕：「逐層載入＋計算重疊」是業界預設，但**沒人量它在 GPU 被別人佔用時的表現**；而我們的 13 號顯示「多開 kernel、和背景排隊」才是忙時的主成本——逐層載入每層都要同步一次，可能比 Cake 更怕忙（G2，X3 的延伸）。

#### D7　KVFlow：依工作流程預測下一個 agent 並預取〔部分〕
- **出處**：Zaifeng Pan 等 9 人（UCSD、AWS）。arXiv [2507.07400](https://arxiv.org/abs/2507.07400)，**NeurIPS 2025**（neurips.cc poster 119883）。
- **方法（原文）**：Agent Step Graph＋「距離執行的步數」做逐節點的淘汰；下一步要跑的 agent 的 KV 由背景執行緒從 CPU 預取到 GPU；分支時全部預取（有上限）。
- **數字（原文）**：單一工作流程（A10G、PCIe Gen1 2 GB/s，Llama-3.1-8B；H100、PCIe Gen5 64 GB/s，Qwen2.5-32B）對 SGLang＋HiCache 最多 1.83×；高併發最多 1.25×（對 LRU 版 HiCache 2.19×）；真實 PEER 負載最多 1.12×。
- **干擾與 stall（原文）**：主張模型前向和 KV 載入用不同硬體資源、PCIe 全雙工、**可以不干擾地重疊**；但**agent 的執行比它的預取短時會卡住**，高併發時工作流程互搶 PCIe；用「跳過還在載入的請求」緩解。**浪費的預取沒量**。沒有消融區分「淘汰」和「預取」各貢獻多少。
- **ROCm**：SGLang v0.4.4 原型；沒找到程式碼連結。
- **對本專案**：「PCIe 全雙工所以不干擾」這個主張，和我們量到的「GPU 忙時 H2D 慢 46–50%」（§0.1）**直接衝突**——值得在 MI300X 上確認是哪一邊（X3）。

#### D8　工具呼叫期間的卸出與預測載回：Tokencake、Continuum
- **Tokencake**〔部分〕：Zhuohang Bian 等，arXiv [2510.18586](https://arxiv.org/abs/2510.18586)（v5 2026-09-28），arXiv 註記 **EuroSys '27**。
  - 時間排程器：工具呼叫開始時，若「預測的停頓 > 來回搬運時間」且 GPU 使用率 >0.60 等條件成立才卸出；工具時間用 EWMA（α＝0.5）預測，和使用者給的估計各半混合；**提早回來就立刻載回**〔原文 §4.1–4.2〕。
  - **載回的排序**：P_upload＝重要度 I＋急迫度 U（離預測完成越近越急），每步最多保留剩餘缺口的一半，有上傳預算防止搶走關鍵請求的 block〔原文 §4.3〕——**這是「誰先還原」的一個具體規則**（G3）。
  - 數字：A100 PCIe、Qwen2.5-14B、4,096 token（256 block × 3 MiB）：卸出 32.0 ms、載回 31.7 ms，重算 1,815 ms；記憶體受限時端到端延遲比 vLLM 少最多 47.06%〔原文〕。
  - 限制：要宣告 DAG 與工具型別；單機；**在 H100 這種高頻寬硬體上，閘門會拒絕大部分卸出**，門檻要依平台驗證；搬運對計算的干擾**沒量**。程式碼 [github.com/pku-lemonade/TokenCake](https://github.com/pku-lemonade/TokenCake)。
- **Continuum**〔摘要〕：Hanchen Li 等 11 人（含 Ion Stoica），arXiv [2511.02230](https://arxiv.org/abs/2511.02230)（v7 2026-09-08）。工具呼叫期間把 KV **釘在 GPU**，保留時間（TTL）依「重載成本」與「淘汰造成的排隊延遲」決定；SWE-Bench、BFCL、OpenHands，平均 JCT 改善 8× 以上〔原文〕。
- **對本專案**〔推論〕：這兩篇的「預測回來時間」是 GPU↔CPU 的尺度（幾十 ms 的搬運）；**沒有一篇把同樣的想法用在 NFS／SSD→DRAM（秒到幾十秒）並把壓縮當成縮短提前量的手段**（G1）。

#### D9　MI300X 的 DMA 引擎：ConCCL、DMA-Latte、FiCCO〔部分／摘要〕
- **ConCCL**〔部分〕：Anirudha Agrawal、Shaizeen Aga、Suchita Pati、Mahzabeen Islam（AMD）。arXiv [2412.14335](https://arxiv.org/abs/2412.14335)（v2 2025-04-25），arXiv 頁無 venue。
  - 平台：**MI300X**；IOD 上有 **14 個 SDMA 複製引擎**；HBM 5.3 TB/s（§II）。
  - 結論（原文）：計算與通訊並行（C3）平均只拿到理想加速的 **21%**（kernel 搶 CU、快取、HBM）；排程優先權與 CU 切分到 42%；把通訊卸給 DMA 引擎（ConCCL）到 **72%**、最多 1.67×（摘要的平均；§VI-D 的端到端約 66%）。DMA 去掉了 CU 與 L1／L2 的干擾，但「**HBM 頻寬的爭用仍在**」（§VII-A1）。
  - 小傳輸：<32 MB 時比 RCCL 慢最多約 4×，原因是 CPU 端發動與同步的成本沒有攤掉（§VI-C）。用 `hsa_amd_memory_async_copy_on_engine` 指定引擎。
- **DMA-Latte**〔摘要〕：Suchita Pati 等，arXiv [2511.06605](https://arxiv.org/abs/2511.06605)（v3 2026-09-14），**MICRO 2026**。把 DMA 卸載推到 KB～低 MB 的延遲受限區；運算子層級最多 4.5×、幾何平均 3.2×；省電 3–10%；端到端 LLM 推論延遲最多低 1.65×、吞吐 1.9×（對 vLLM）〔原文〕。
- **FiCCO**〔摘要〕：Shagnik Pal 等，arXiv [2512.10236](https://arxiv.org/abs/2512.10236)（v3 2026-06-03）。比分片更細的計算／通訊重疊；主要限制是拆解與爭用造成的變慢；用 DMA 引擎降爭用；最多 1.6×〔原文〕。
- **對本專案**〔推論〕：這組 AMD 自己的論文是「MI300X 上搬運要走 SDMA 才不搶 CU」的最強證據，但對象是**GPU 之間的集合通訊**，不是主機→GPU 的 KV 還原。**把它搬到 KV 還原、並且在有背景負載時量，沒人做過**（G2）。

#### D10　計算／通訊重疊的拖慢：Lee et al.〔摘要〕
- Seonho Lee、…、Divya Mahajan，arXiv [2507.03114](https://arxiv.org/abs/2507.03114)（2025-07-03）。分散式訓練：重疊讓計算平均慢 **18.9%、最多 40.0%**；但完全不重疊（依序）平均只慢 **10.2%**（最多 26.6%）；重疊時峰值功耗最多高 25%（搜尋摘要）〔原文〕。評測 GPU 含 H100、A100、MI250、MI210（搜尋摘要；abs 頁沒列）。
- **對本專案**〔推論〕：「重疊一定比較好」的假設在訓練就不成立；和我們 13 號「GPU 忙時全部搬完再一次拆比 Cake 重疊快」是同一種現象。

#### D11　vLLM：DMA vs GPU 核心拷貝（見 C1）〔全文〕
- 併發時自訂拷貝 kernel 比不卸載差 6%（0% 命中），DMA 吞吐好 5.5–15%（Llama-3.2-1B、H100）；Llama-3.1-8B 最多 32%〔原文〕。這是 KV 卸載領域唯一的「搬運方式 × 干擾」量測。

#### D12　PRESERVE（名稱相近、問題不同）〔摘要〕
- Ahmet Caner Yüzügüler、Jiawei Zhuang、Lukas Cavigelli（華為蘇黎世），arXiv [2501.08192](https://arxiv.org/abs/2501.08192)（v2 2025-05-26）。在**通訊期間**把權重與 KV 從 **HBM 預取到晶片內快取**；端到端最多 1.6×、最佳 L2 大小再 1.25× 性價比〔原文〕。**不是主機卸載**，委託裡把它列在「預取」是對的，但它和我們的還原問題只在「利用閒置窗口」有交集。

### 3.3 「GPU 忙時」的證據彙整

| 證據 | 平台 | 量到什麼 | 出處 |
|:--|:--|:--|:--|
| 計算／通訊重疊拖慢計算 | H100、A100、MI250、MI210 | 平均 18.9%、最多 40%；依序只慢 10.2% | D10〔摘要〕 |
| 並行只拿到理想的一小部分 | **MI300X** | 21% → DMA 卸載後 72%；HBM 爭用仍在 | D9〔部分〕 |
| GPU 上解壓拖慢 decode／prefill | L40S；A100／H20／L20 | ≥25–30%；prefill +50%、decode +20% | B4、B5〔部分〕 |
| 反量化佔 JCT | A10G 等 | 17.2–37.9% | B6〔部分〕 |
| 拷貝 kernel 搶計算 | H100 | 比不卸載差 6%；DMA 好 5.5–32% | C1〔全文〕 |
| 「PCIe 全雙工所以不干擾」 | A10G、H100 | 主張，沒量；預取比執行久時會卡 | D7〔部分〕 |
| Cake 用 token budget 模擬 GPU 忙 | A100（延遲注入） | 有退化區（Llama-3.1-70B、12.5% GPU、32 Gbps 時 0.80×） | `SOTA_ANALYSIS` §2.1 |
| **本專案**：H2D 在 GPU 忙時變慢 | MI300X | Llama 80.0→120.1 ms、Qwen 35.7→52.2 ms | 09 §6.1〔實測〕 |
| **本專案**：反量化在 GPU 忙時變慢 | MI300X | 2.4–3.8 倍 | 09 §6.1〔實測〕 |
| **本專案**：忙時 Cake 重疊最慢、一次拆最穩 | MI300X | INT4、duty 0.4→0.9：S4＋全載入 35→37 ms（Llama）；Cake 57→134 ms；留 CU 給還原讓背景慢 12–20% | 11 §3d〔實測，stage 43b／43ca〕 |
| **本專案**：沒存前段時 GPU 忙的代價 | MI300X | S3 比 S0 慢 13–66%（16K 以上） | 第一階段 Q2〔實測，run `20261008-134526-m7-a1`〕 |

**讀法**〔推論〕：外部文獻在「重疊會互相拖慢」上是一致的，但**沒有一篇在 KV 還原、主機→GPU、有真實背景負載的條件下比較「DMA 引擎 vs kernel 拷貝」與「重疊 vs 一次做完」**。我們手上已經有一半的答案（11 §3d），缺的是「為什麼 H2D 本身也慢 46–50%」——如果是因為 HIP 的拷貝走了 blit kernel，改走 SDMA 就可能消掉（X3）；如果是 HBM／PCIe 爭用，就消不掉。

---

## 4. 研究缺口

> 每個缺口：有沒有人做過（附出處）→ 可能的新意 → 最小證偽實驗（§5 的編號）→ 何時停止投入。「沒找到」只代表這輪搜尋沒找到。

### G1　壓縮能不能把 NFS 拉近 CPU？——改問「壓縮 × 預取提前量」

- **有沒有人做過**：
  - 損益兩平公式：KVServe（加總）、SplitZip（管線）；實測門檻：ShadowServe、KVCodec、KVServe（§1.5）。
  - 「預取到 DRAM」：CachedAttention 依 job queue 前瞻把 session 從磁碟拉到 DRAM（ATC'24）；HyMCache 用請求層級的前綴預取，把延遲關鍵的讀取先放進 CXL 裝置的 DRAM（後面是 SSD）〔摘要，arXiv [2607.18141](https://arxiv.org/abs/2607.18141)；`A_HyMCache` 卡〕；搜尋摘要提到預取讓多輪測試少 14%（**NOT_VERIFIED**）；Tokencake 依工具時間預測載回（GPU↔CPU）；Bidaw 依上一輪回答長度預測回來時間來決定**降到 SSD**（不是預取回來，`SOTA_ANALYSIS` §2.5）。
  - 卸載層壓縮：TensorRT-LLM cold-page NVFP4（延後量化，沒數字）；LMCache QAT（草稿）；本專案 stage 44（NFS／SSD 級 INT4 對 BF16，實測）。
- **沒找到的**：(a) 把**壓縮比當成縮短預取提前量的手段**、和**間隔（使用者思考時間、工具執行時間）分布**放在同一個模型；(b) 在慢層用 **CPU 端無損解壓**（不搶 GPU）的實測；(c) 真實 NFS（頻寬隨時間跳動，07 的 C7 錨點段：377 MiB/s 到約 830 MiB/s）下的提前量預測誤差。
- **可能的新意**〔推論〕：「慢層的價值不是還原速度，而是『預取來不來得及』；壓縮的價值是把來不及的比例變小」——一個可以畫成「提前量 vs 間隔 CDF」的簡單模型，帶真實 NFS 的變異與 GPU 忙時的不干擾驗證。
- **最小證偽實驗**：X1（我們 KV 的真實無損比）→ X2（提前量 × 間隔分布的算術）→ X5（真實 NFS 預取 1–2 個點）。
- **何時停止**：
  - X1 的 BF16 無損比 <1.15× → 放棄無損這條，只留有損（INT4）縮短提前量；
  - X2 顯示在可取得的間隔分布上，「藏得住的比例」BF16 vs INT4 差 <5 個百分點（例如間隔中位數遠大於 12 s，或遠小於 1.7 s）→ 壓縮對預取沒有邊際價值，停止；
  - 找不到有時間戳的長 context 間隔資料（`SOTA_ANALYSIS` §4.4：長 context＋真實時間＋中低重用的公開 trace 找不到）→ 只能用合成分布，結論降級成敏感度分析。

### G2　預取與重疊在 GPU 忙時還有沒有效？會不會反而造成 stall？

- **有沒有人做過**：KV 預取／重疊的論文（D1–D8）**都沒有量真實背景負載**；Cake 用 token budget 模擬；HCache 只報 TBT +4%；KVFlow 主張不干擾。干擾的量測來自訓練通訊（D9、D10）、壓縮解碼（B4、B5）、vLLM 的拷貝方式（C1）。本專案已量：忙時 Cake 最慢、一次拆最穩，留 CU 有代價（11 §3d）；H2D 本身慢 46–50%（09 §6.1）。
- **沒找到的**：MI300X 上主機→GPU 的 KV 還原，**SDMA vs blit 拷貝**在背景負載下的差；**逐層載入（LMCache layerwise）**在背景負載下和「全部搬完再一次拆」的比較；「重疊的淨收益變成負的」那條界線（背景 duty × 頻寬）。
- **可能的新意**〔推論〕：一條實用規則「GPU 忙到多少以上，還原就不要重疊、改走 SDMA 一次搬完」，加上 MI300X 的機制解釋（kernel 排隊 vs HBM 爭用）。這和 13 號的 G5b 會合規則（離每格最好只差 0.4%）是同一條線的延伸。
- **最小證偽實驗**：X3（`HSA_ENABLE_SDMA`＝0／1 × 背景 duty × block 大小，量 H2D 與背景速度）；延伸：逐層載入 vs S4。
- **何時停止**：X3 的 SDMA 開／關在每個 duty 下差 <5% → 拷貝方式不是原因，H2D 變慢來自 HBM／PCIe 爭用，這條沒有可改的旋鈕，停止；逐層載入和 S4 差 <5% → 已被 13 號的規則涵蓋，停止。

### G3　多請求併發時，搬運佇列誰先還原？

- **有沒有人做過**（不少，但都只做了一部分）：
  - CacheFlow：整批請求的（請求, 層區塊）用動態規劃決定重算或載入，共享 I/O（arXiv 2604.25080 v2；abs 頁明寫處理「batched requests 之間的資源爭用」）；但只有一個載入頻寬、假設計算和載入互不干擾（`SOTA_ANALYSIS` §2.2）。
  - Strata：排程器同時看要載入和要計算的，處理 delayed hit。
  - Tutti：逐層 slack 感知的 SSD I/O 排程。
  - CachedAttention：job queue 前瞻的磁碟→DRAM 預取。
  - Tokencake：載回優先序＝重要度＋急迫度，有預算（D8）。
  - KVFlow：依「距離執行的步數」預取；跳過還在載入的請求（D7）。
  - CacheSmith（arXiv [2610.18925](https://arxiv.org/abs/2610.18925)，2026-10-13）：SLO 感知排程、把同 deadline 的載入分組、跨多個 I/O stream〔摘要＋搜尋摘要〕。
  - FastSwitch（arXiv [2411.18424](https://arxiv.org/abs/2411.18424)）：公平性搶占下的 context switch I/O；對 vLLM 1.4–11.2×〔摘要〕。
  - Continuum：TTL＋程式層級 FCFS（D8）。
- **沒找到的**：(a) 同一個佇列裡請求分別在**不同層**（CPU／SSD／NFS），而且每個還原本身是 Cake 雙路（重算吃共享 GPU、載入吃各自的鏈路）；(b) **壓縮等級當成排程旋鈕**（截止時間緊的請求在降級時就存更小的格式）；(c) 有真實背景負載時的量測。
- **可能的新意**〔推論〕：慢層請求應該「先開始預取、晚一點才搶 GPU」，快層請求「後開始、先搶 GPU」——兩種資源的排序不同；SRPT 類規則在雙資源下怎麼改。
- **最小證偽實驗**：X6（模擬器＋4 個請求的小實測）。
- **何時停止**：最好的排序和 FCFS 的平均／p95 TTFT 差 <5%（CLAUDE.md 規則 4 的 NO-GO 門檻）；或差異全部來自「還原方法」（Cake vs 全載入）而不是「順序」。

### G4　「壓縮＋丟棄（重算）＋卸載」三者聯合的成本模型

- **有沒有人做過**（這是四個缺口裡前作最多的）：
  - EvicPress／AdaptCache：壓縮與淘汰跨層聯合決定，效用＝頻率 ×（品質權重 − 大小／頻寬）（`SOTA_ANALYSIS` §2.7；EvicPress arXiv [2512.14946](https://arxiv.org/abs/2512.14946)）。
  - InferCept（ICML 2024，arXiv [2402.01869](https://arxiv.org/abs/2402.01869)）：工具呼叫時 preserve／discard（重算）／swap 依浪費量自適應組合；重算佔前向時間 37–40%；吞吐 1.6–2×〔摘要〕。
  - ALISA（ISCA 2024，arXiv [2403.17312](https://arxiv.org/abs/2403.17312)）：稀疏視窗 attention（丟）＋三階段逐 token 排程在「快取 vs 重算」之間平衡，單 GPU-CPU 對 FlexGen 3×、vLLM 1.9×〔摘要〕；量化部分只見第三方評述（NOT_VERIFIED）。
  - KVServe（壓縮方法依頻寬／SLO 選）、CacheGen（壓縮等級或送文字重算）、KVTC（壓縮 vs 重算的 TTFT）、TensorRT-LLM cold-page、py-kvcache（存 vs 算的損益長度）、Krul（每層前段不存）。
  - 本專案：stage 44 的 IO 牆與容量（08 §2.8）、13 號 G5b 會合規則（11 問題三）、第一階段的「依位置放置」NO-GO。
- **沒找到的**：一個同時含 **(i) 位置相依的重算成本（Cake）、(ii) 每層鏈路、(iii) 有損等級與無損、(iv) GPU 忙時的解壓排隊、(v) 品質約束**，而且**在真實的層上驗證**的模型。
- **新意風險（要先講）**〔推論〕：第二階段已經得到「位置沒幫助、INT4 KIVI 全用就過 ε、G5b 自動選 Cake 或全載入離最好 0.4%」。聯合模型很可能退化成「全部 INT4＋G5b」這條簡單規則，**能贏它的空間可能 <5%**。
- **最小證偽實驗**：用已量的零件（每塊重算時間、各層頻寬、反量化／解壓時間閒與忙、INT4 品質）組一個模型，在（層 × 忙 × 長度 × 容量）格子上比「模型選的配置」與「全 INT4＋G5b」的推算還原時間；只有差 ≥5% 的格子才拿去實測。純 CPU 推算約半天（沿用 `m10_sweep_sim`、`m10_iowall`）。
- **何時停止**：所有格子的推算差 <5% → NO-GO，寫成負面結果（「在 MI300X＋GQA 模型上，聯合模型等於一條規則」）；推算差 ≥5% 但實測差 <5% → 停。

### G5（附帶）　無損當成「對敏感模型安全」的預設
- 證據：Qwen2.5 的 FP8 掉 33 個百分點（08 §2.3），KVServe 也觀察到 Qwen2.5-7B 的 KV bias 讓靜態基線失敗（B2），FreeKV 觀察到 ShadowKV 在 Qwen-2.5 重複輸出（D2）。〔推論〕有損格式的風險依模型而異、事前難知；無損 1.3× 雖小但**品質零風險**。可以和 G1 一起量（X1），不另開實驗。停止條件同 G1。

---

## 5. 期限前（約 3 天、2×MI300X、2.2 TB DRAM、本地 SSD 級 ~7 GiB/s、NFS ~0.33 GiB/s）可測的最小實驗候選

> **前提**：機器正在計時量測。下面所有會用 CPU 或 GPU 的項目，都要**等計時的 stage 結束、gpu_guard CLEAN** 才能跑；CPU 工作（壓縮、模擬）也會污染計時（CPU 配額只有 32 核）。要裝的套件（zstandard、zipnn）需要使用者同意。所有數字照 CLAUDE.md §4 記 run_id。

| 編號 | 對應缺口 | 假說（先寫死） | 對照 | 指標 | 資源／時間 | 停損 |
|:--|:--|:--|:--|:--|:--|:--|
| **X1** 真實 KV 的無損壓縮率 | G1、G5 | H1a：BF16 KV 用指數分離（top-16 定長碼）可到 ≥1.25×（SplitZip 在其他模型 1.31–1.33×）；H1b：通用 zstd／lz4（不做位元組分組）≤1.10×；H1c：FP8 ≤1.15×、INT4 KIVI 打包後 ≈1.0×；H1d：Qwen 的 K（固定大偏移）指數熵比 Llama 低 | 同一批 tensor：(i) 隨機打亂指數位元組（負對照）；(ii) 權重（正對照，ZipNN 報 1.51×） | 每層、K／V、每個 block 的指數熵、top-16 覆蓋率、實得比；CPU 單執行緒編／解 GB/s；逐位元還原檢查 | 32K、兩個模型各 64 block（從既有 harness 存一份 KV；需要 1 次 prefill）；**CPU 約 2–4 小時** | BF16 比 <1.15× → 無損整條停 |
| **X2** 盈虧＋預取提前量算術 | G1、G4 | H2a：無損在 NFS 級讓還原變快 <1.3×，永遠到不了 CPU 的 5 倍以內；H2b：INT4 把 Llama 32K 的提前量從 12.1 s 縮到 3.8 s，在合理的間隔分布上「藏得住的比例」多 ≥10 個百分點 | BF16、無損（X1 的比）、FP8、INT4；KVServe 加總 vs SplitZip 管線兩種界 | 每層的 B\*、推算還原時間（沿用 `m10_sweep_sim`／`m10_iowall`，09 §5 已驗證 1–2% 誤差，但**慢層＋GPU 忙會偏快 23–24%**）、提前量、P(間隔 > 提前量) | **0 GPU、CPU <1 小時**；間隔分布：Mooncake 時間戳（512-token block，CLAUDE.md 規則 6）、Bidaw trace（若在機器上）、合成 | 「藏得住的比例」BF16 vs INT4 差 <5 個百分點 → G1 停 |
| **X3** SDMA vs blit：H2D 為什麼在 GPU 忙時慢 46–50% | G2 | H3：若 HIP 的 pinned H2D 有一部分走 blit kernel，`HSA_ENABLE_SDMA=1` 時忙／閒差 <15%，`=0` 時 ≥40%；若兩者一樣慢 → 原因是 HBM／PCIe 爭用 | `HSA_ENABLE_SDMA`＝0／1 × 背景 duty 0／0.4／0.9 × block 2 MiB／64 MiB × 方法 S0／S4／Cake（CPU 層） | H2D GB/s、還原時間、**背景工作的速度**（13 號同一套）、gpu_guard | **1 張 GPU、2–3 小時**；不需要新程式，只改環境變數〔推論；env 對 HIP 路徑的效果 NOT_VERIFIED，先用 rocprof 看拷貝是 SDMA 還是 kernel〕 | 開／關在每個 duty 差 <5% → 拷貝方式不是原因，G2 的這半條停 |
| **X4** GPU 端無損解碼在忙時的成本 | G1、G2 | H4：逐塊解碼在忙時會像反量化一樣慢 2–4 倍；「全部搬完一次解」維持平穩（同 11 §3d）；CPU 層上無損＋一次解 ≤ BF16 全載入 | BF16 全載入、INT4 KIVI＋融合拆（現有最佳） | 還原時間（閒／忙）、背景速度 | 寫一個 top-16 指數查表解碼的 Triton kernel（形狀和 `m10_dequant_fast.py` 相同），或試移植 SplitZip；**0.5–1 天** | CPU 層忙時無損還原 ≥ BF16 → 無損只留給慢層（用 CPU 解），GPU 解碼停 |
| **X5** 真實 NFS→DRAM 預取，GPU 忙時的干擾 | G1、G2 | H5a：背景 NFS 讀取（CPU 執行緒）對同時進行的 CPU 層還原與背景 GPU 工作影響 <5%；H5b：真實 NFS 的吞吐變異係數 <30%（否則提前量不可預測） | BF16、INT4（可加 X1 的無損），每格 ≥10 次，分散在不同時段 | NFS 讀取 GiB/s 的分布、預取完成時間、同時的還原時間與背景速度 | **0.5 天**；注意 NFS 是整個叢集共用（`MI300X_MLSTEAM.md`），**gpu_guard 抓不到 NFS 的別人**——要另記同時段的 NFS 基準讀取 | H5b 不成立（CV ≥50%）→ 提前量模型只能給上界，G1 降級 |
| **X6** 多請求還原的排序 | G3 | H6：請求的層與大小混雜時，「慢層先開始預取＋最短還原先搶 GPU」比 FCFS 的平均 TTFT 少 ≥15%；GPU 忙時排序的影響小於還原方法的影響 | FCFS、SRPT（估計還原時間最短先）、EDF（依預測回來時間）、Tokencake 式（重要度＋急迫度） | 平均／p95 TTFT、總還原時間 | 先用模擬器（每塊成本、各層頻寬都是已量的）**半天**；最好的兩個規則在 1 張 GPU 上做 4 個併發還原的小實測**半天** | 最好的規則對 FCFS 差 <5% → NO-GO（規則 4） |

**建議順序**〔推論〕：X2（0 GPU，現在就能寫程式，不用等）→ X1（計時結束後的第一個 CPU 空窗）→ X3（最便宜的 GPU 實驗，結果直接決定 G2 要不要做）→ 依 X1–X3 的停損結果再決定 X4／X5／X6。三天內實際做得完的大概是 X1＋X2＋X3，外加 X4 或 X6 其中一個。

---

## 6. 參考清單

「核」欄：✓＝這輪用 WebFetch 開過 arXiv abs 頁（或官方頁）確認存在與標題；P＝會議 proceedings 頁已開；— ＝非 arXiv 文件，直接讀官方頁。

### B（壓縮）
| # | 文獻 | 連結 | 核 |
|:--|:--|:--|:--|
| [B1] | Y. Guo, S. Joshi, "SplitZip: Ultra Fast Lossless KV Compression for Disaggregated LLM Serving," NeurIPS 2026 (poster) | https://arxiv.org/abs/2605.01708 ；https://github.com/Intelligent-Microsystems-Lab/SplitZip | ✓＋neurips.cc |
| [B2] | Z. Liu *et al.*, "KVServe: Service-Aware KV Cache Compression for Communication-Efficient Disaggregated LLM Serving," SIGCOMM 2026 | https://arxiv.org/abs/2605.13734 | ✓ |
| [B3] | K. Staniszewski, A. Łańcucki, "KV Cache Transform Coding for Compact Storage in LLM Inference," ICLR 2026 | https://arxiv.org/abs/2511.01815 | ✓ |
| [B4] | X. Xiang *et al.*, "ShadowServe: Interference-Free KV Cache Fetching for Distributed Prefix Caching," arXiv 2025 | https://arxiv.org/abs/2509.16857 | ✓ |
| [B5] | L. Mi *et al.*, "Efficient Remote KV Cache Reuse with GPU-native Video Codec"（KVCodec）, SIGCOMM 2026 | https://arxiv.org/abs/2602.09725 | ✓ |
| [B6] | Z. Zhang *et al.*, "HACK: Homomorphic Acceleration via Compression of the Key-Value Cache for Disaggregated LLM Inference," arXiv 2025 | https://arxiv.org/abs/2502.03589 | ✓ |
| [B7a] | T. Zhang *et al.*, "70% Size, 100% Accuracy: Lossless LLM Compression … (DFloat11)," NeurIPS 2025 | https://arxiv.org/abs/2504.11651 | ✓ |
| [B7b] | M. Hershcovitch *et al.*, "ZipNN: Lossless Compression for AI Models," IEEE Cloud | https://arxiv.org/abs/2411.05239 ；https://github.com/zipnn/zipnn | ✓ |
| [B7c] | R. Fan *et al.*, "ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression," ASPLOS'26 | https://arxiv.org/abs/2603.17435 | ✓ |
| [B7d] | P. Yubeaton *et al.*, "Huff-LLM: End-to-End Lossless Compression for Efficient LLM Inference," arXiv 2025 | https://arxiv.org/abs/2502.00922 | ✓ |
| [B7e] | F. Cheng *et al.*, "Ecco: Improving Memory Bandwidth and Capacity for LLMs via Entropy-aware Cache Compression," ISCA 2025 | https://arxiv.org/abs/2505.06901 | ✓ |
| [B8a] | B. Jiang *et al.*, "KVComp: A High-Performance, LLM-Aware, Lossy Compression Framework for KV Cache," arXiv 2025 | https://arxiv.org/abs/2509.00579 | ✓ |
| [B8b] | B. Jiang *et al.*, "PackKV: Reducing KV Cache Memory Footprint through LLM-Aware Lossy Compression," arXiv 2025 | https://arxiv.org/abs/2512.24449 | ✓ |
| [B9] | Y. Liu *et al.*, "CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving," SIGCOMM'24 | https://arxiv.org/abs/2310.07240 | ✓ |
| [B10a] | LMCache RFC #4149（Hardware-Accelerated KV Compress Adapter）、PR #4264（Intel QAT，draft） | https://github.com/LMCache/LMCache/issues/4149 ；https://github.com/LMCache/LMCache/pull/4264 | — |
| [B10b] | TensorRT-LLM, "KV Cache Compression"（cold-page NVFP4） | https://nvidia.github.io/TensorRT-LLM/1.3.0rc27/features/kv-cache-compression.html | — |
| [B10c] | S. Ma *et al.*, "UCCL-Zip: Lossless Compression Supercharged GPU Communication," arXiv 2026 | https://arxiv.org/abs/2604.17172 | ✓ |
| [B10d] | C. Künas *et al.*, "Characterizing Lossless GPU Data Compression Across AMD CDNA and RDNA Architectures," Springer 2027, DOI 10.1007/978-3-032-30491-9_23 | https://escholarship.org/uc/item/5hs138d9 | 摘要（二手） |
| [B10e] | LMCache 文件（v1 設定、CacheGen、layerwise） | https://docs.lmcache.ai/configuration/v1/v1_config.html | — |
| [B10f] | NIXL backend selection | https://docs.nvidia.com/nixl/user-guide/backend-selection | — |

### C（卸載與分層）
| # | 文獻 | 連結 | 核 |
|:--|:--|:--|:--|
| [C1] | vLLM blog, "Inside vLLM's New KV Offloading Connector," 2026-01-08 | https://vllm.ai/blog/2026-01-08-kv-offloading-connector | — |
| [C2] | K. Toledo *et al.*（IBM）, "Native KV Cache Offloading to Any File System with llm-d," 2026-02-10 | https://llm-d.ai/blog/native-kv-cache-offloading-to-any-file-system-with-llm-d | — |
| [C3a] | J. Frick, S. Bates, "Introducing ROCm AMD Infinity Context," ROCm blog 2026-07-22 | https://rocm.blogs.amd.com/software-tools-optimization/amd-infinity-context/README.html | — |
| [C3b] | hipFile 文件／GitHub | https://rocm.docs.amd.com/projects/hipFile/en/latest ；https://github.com/ROCm/hipFile | — |
| [C3c] | ROCr Runtime 環境變數（HSA_ENABLE_SDMA） | https://rocm.docs.amd.com/projects/ROCR-Runtime/en/master/api-reference/environment_variables.html | — |
| [C3d] | rocDecode 支援的格式與硬體 | https://rocm.docs.amd.com/projects/rocDecode/en/latest/reference/rocDecode-formats-and-architectures.html | 摘要 |
| [C4a] | NVIDIA Dynamo KVBM design | https://docs.nvidia.com/dynamo/v1.0.0/design-docs/component-design/kvbm-design | 摘要 |
| [C4b] | NVIDIA 新聞稿, "NVIDIA BlueField-4 Powers New Class of AI-Native Storage Infrastructure," 2026-01-05 | https://investor.nvidia.com/news/press-release-details/2026/NVIDIA-BlueField-4-Powers-New-Class-of-AI-Native-Storage-Infrastructure-for-the-Next-Frontier-of-AI/default.aspx | 摘要（經轉載） |
| [C6] | S. Luo, H. Shen, "No Buffer, No Bottleneck: Efficient Zero-Copy KV Cache Offloading for Long-Context LLMs"（DirectKV）, OSDI 2026 | https://papers.cool/venue/luo@osdi26@USENIX ；https://www.usenix.org/system/files/osdi26-luo.pdf | 列表頁 |
| [C7a] | X. Yang *et al.*, "Beluga: A CXL-Based Memory Architecture for Scalable and Efficient LLM KVCache Management," SIGMOD'26 | https://arxiv.org/abs/2511.20172 | ✓ |
| [C7b] | D. Liu, Y. Yu, "CXL-SpecKV: A Disaggregated FPGA Speculative KV-Cache for Datacenter LLM Serving," FPGA'26 | https://arxiv.org/abs/2512.11920 | ✓ |
| [C8] | H. Zhang, C. Xia, Z. Wang, "KVSwap: Disk-aware KV Cache Offloading for Long-Context On-device Inference," arXiv 2025 | https://arxiv.org/abs/2511.11907 | ✓ |
| [C9a] | Y. Sheng *et al.*, "FlexGen: High-Throughput Generative Inference of LLMs with a Single GPU," ICML 2023 | https://arxiv.org/abs/2303.06865 | ✓＋P |
| [C9b] | Y. Xiong *et al.*, "LayerKV: Optimizing LLM Serving with Layer-wise KV Cache Management," arXiv 2024 | https://arxiv.org/abs/2410.00428 | ✓ |
| [C10a] | S. Qiu *et al.*, "Tutti: Making SSD-Backed KV Cache Practical for Long-Context LLM Serving," arXiv 2026 | https://arxiv.org/abs/2605.03375 | ✓ |
| [C10b] | Z. Xie *et al.*, "Strata: Hierarchical Context Caching for Long Context Language Model Serving"（arXiv 版標 under review；OSDI'26 依本專案卡片） | https://arxiv.org/abs/2508.18572 | ✓ |
| [C10c] | S. Nian *et al.*, "CacheFlow: Efficient LLM Serving via Automated 3D-Parallel KV Cache Restoration," arXiv 2026 (v2) | https://arxiv.org/abs/2604.25080 | ✓ |
| [C10d] | B. Gao *et al.*, "Cost-Efficient LLM Serving for Multi-turn Conversations with CachedAttention," ATC 2024 | https://arxiv.org/abs/2403.19708 | ✓ |
| [C10e] | Y. Liu *et al.*, "LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference," arXiv 2025 | https://arxiv.org/abs/2510.09665 | ✓ |
| [C10f] | S. Feng *et al.*, "EVICPRESS: Joint KV-Cache Compression and Eviction for Efficient LLM Serving," arXiv 2025 | https://arxiv.org/abs/2512.14946 | ✓ |
| [C10g] | J. Kanichai *et al.*, "Building py-kvcache …," arXiv 2026 | https://arxiv.org/abs/2609.11744 | ✓ |
| [C10h] | S. Jin *et al.*, "Compute Or Load KV Cache? Why Not Both?"（Cake；ICML 2025 依 P01） | https://arxiv.org/abs/2410.03065 | ✓ |

### D（預取與重疊）
| # | 文獻 | 連結 | 核 |
|:--|:--|:--|:--|
| [D1] | W. Lee *et al.*, "InfiniGen: Efficient Generative Inference of LLMs with Dynamic KV Cache Management," OSDI 2024 | https://arxiv.org/abs/2406.19707 | ✓ |
| [D2] | H. Sun *et al.*, "ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference," ICML 2025 Spotlight | https://arxiv.org/abs/2410.21465 ；https://github.com/ByteDance-Seed/ShadowKV | ✓ |
| [D3a] | G. Liu *et al.*, "FreeKV: Boosting KV Cache Retrieval for Efficient LLM Inference," arXiv 2025 | https://arxiv.org/abs/2505.13109 | ✓ |
| [D3b] | S. Jie *et al.*, "SpeCache: Speculative Key-Value Caching for Efficient Generation of LLMs," ICML 2025 | https://arxiv.org/abs/2503.16163 | ✓（PMLR 經搜尋） |
| [D3c] | Z. Liang *et al.*, "DualDecoder: Accelerate Long Context LLM Inference by Predictive Prefetch," arXiv 2026 | https://arxiv.org/abs/2607.26475 | ✓ |
| [D4] | C. Jiang *et al.*, "KVPR: Efficient LLM Inference with I/O-Aware KV Cache Partial Recomputation," ACL Findings 2025 | https://arxiv.org/abs/2411.17089 | ✓ |
| [D5] | S. Gao, Y. Chen, J. Shu, "Fast State Restoration in LLM Serving with HCache," EuroSys 2025 | https://arxiv.org/abs/2410.05004 | ✓ |
| [D6] | J. Yao *et al.*, "CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion" | https://arxiv.org/abs/2405.16444 | ✓ |
| [D7] | Z. Pan *et al.*, "KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows," NeurIPS 2025 | https://arxiv.org/abs/2507.07400 | ✓（neurips 經搜尋） |
| [D8a] | Z. Bian *et al.*, "TokenCake: A KV-Cache-centric Serving Framework for LLM-based Multi-Agent Applications," EuroSys '27 | https://arxiv.org/abs/2510.18586 | ✓ |
| [D8b] | H. Li *et al.*, "Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live," arXiv 2025 | https://arxiv.org/abs/2511.02230 | ✓ |
| [D9a] | A. Agrawal *et al.*, "Optimizing ML Concurrent Computation and Communication with GPU DMA Engines," arXiv 2024 | https://arxiv.org/abs/2412.14335 | ✓ |
| [D9b] | S. Pati *et al.*, "DMA-Latte: Expanding the Reach of DMA Offloads to Latency-bound ML Communication," MICRO 2026 | https://arxiv.org/abs/2511.06605 | ✓ |
| [D9c] | S. Pal *et al.*, "Design Space Exploration of DMA based Finer-Grain Compute Communication Overlap," arXiv 2025 | https://arxiv.org/abs/2512.10236 | ✓ |
| [D10] | S. Lee *et al.*, "Characterizing Compute-Communication Overlap in GPU-Accelerated Distributed Deep Learning: Performance and Power Implications," arXiv 2025 | https://arxiv.org/abs/2507.03114 | ✓ |
| [D12] | A. C. Yüzügüler *et al.*, "PRESERVE: Prefetching Model Weights and KV-Cache in Distributed LLM Serving," arXiv 2025 | https://arxiv.org/abs/2501.08192 | ✓ |

### 缺口節另外引用的
| # | 文獻 | 連結 | 核 |
|:--|:--|:--|:--|
| [G1] | R. Abhyankar *et al.*, "InferCept: Efficient Intercept Support for Augmented LLM Inference," ICML 2024 (PMLR v235) | https://arxiv.org/abs/2402.01869 | ✓＋P |
| [G2] | Y. Zhao *et al.*, "ALISA: Accelerating LLM Inference via Sparsity-Aware KV Caching," ISCA 2024 | https://arxiv.org/abs/2403.17312 | ✓ |
| [G3] | A. Shen *et al.*, "FastSwitch: Optimizing Context Switching Efficiency in Fairness-aware LLM Serving," arXiv 2024 | https://arxiv.org/abs/2411.18424 | ✓ |
| [G4] | X. Ma *et al.*, "CacheSmith: Accurate RAG Serving under Latency SLOs with Adaptive KV Reuse," arXiv 2026-10 | https://arxiv.org/abs/2610.18925 | ✓ |
| [G5] | H. Jang *et al.*, "A CXL Memory Rack for Multi-Turn LLM Serving"（HyMCache）, arXiv 2026 | https://arxiv.org/abs/2607.18141 | ✓ |

### 查了沒找到（NOT_FOUND）
- 「KV Infinity」這個名稱的論文或系統。
- Intel／LMCache 的 QAT KV 壓縮的任何量測數字（只有設計與草稿 PR）。
- NIXL、Mooncake 傳輸引擎、Dynamo KVBM 內建的 KV 壓縮。
- 把「GPU 忙時的解壓排隊」放進盈虧公式的論文。
- 把壓縮比和「預取提前量 × 間隔分布」放在一起的論文。
- KV 還原（主機→GPU）在背景負載下 SDMA vs blit 拷貝的量測。
- ROCm 上官方的 GPU 無損壓縮函式庫（nvCOMP 對應品）。
