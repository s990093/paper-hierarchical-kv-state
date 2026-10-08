# E11：systems 論文的評測手藝（章節結構、量測統計、模擬可信度、artifact evaluation、審稿人常問）

- **抽取者**：子 agent E11，查證日期 2026-10-06～10-07
- **範圍**：(1) 七篇 serving／KV 快取論文的評測章節結構；(2) 量測與統計的規則（Hoefler & Belli、Heiser、Mytkowicz、Kalibera & Jones、SIGPLAN checklist，以及 GPU／LLM 特有的陷阱）；(3) 模擬與真機：模擬器怎麼驗證、審稿人在什麼條件下接受；(4) artifact evaluation：ACM badge 定義與 OSDI／SOSP／EuroSys／ATC／MLSys 的做法；(5) 審稿人常問的評測問題。
- **頁碼慣例**：「p*N*」一律指**下載的 PDF 第 N 頁**（USENIX 會議版第 1 頁是封面，所以比論文印的頁碼多 1 左右）。章節編號照原文。
- **引用**：以改寫為主；直接引用一次不超過 15 個英文字。

## 來源清單（全部於 2026-10-06／07 讀取）

| 代號 | 來源 | 讀的版本 | URL | 讀了什麼 |
|:--|:--|:--|:--|:--|
| vLLM | Kwon et al., Efficient Memory Management for LLM Serving with PagedAttention（SOSP'23） | arXiv 2309.06180v1（PDF 內有 SOSP '23 版權列） | https://arxiv.org/pdf/2309.06180 | §6–§7 全文＋圖 |
| Sarathi | Agrawal et al., Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve（OSDI'24） | USENIX 會議版 | https://www.usenix.org/system/files/osdi24-agrawal.pdf | §5 全文＋圖 |
| DistServe | Zhong et al., DistServe（OSDI'24） | USENIX 會議版 | https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf | §6 全文＋圖 |
| CA | Gao et al., Cost-Efficient LLM Serving for Multi-turn Conversations with CachedAttention（ATC'24） | USENIX 會議版 | https://www.usenix.org/system/files/atc24-gao-bin-cost.pdf | §4 全文＋圖 |
| Mooncake | Qin et al., Mooncake: Trading More Storage for Less Computation（FAST'25） | USENIX 會議版 | https://www.usenix.org/system/files/fast25-qin.pdf | §5 全文＋圖 |
| Strata | Xie et al., Strata: Hierarchical Context Caching for Long Context LM Serving（OSDI'26） | USENIX 會議版 | https://www.usenix.org/system/files/osdi26-xie-zhiqiang.pdf | §5–§6 全文＋圖 |
| Cake | Jin et al., Compute or Load KV Cache? Why Not Both?（ICML'25） | PMLR v267 版（主）；另比對 arXiv 2410.03065v2 | https://raw.githubusercontent.com/mlresearch/v267/main/assets/jin25d/jin25d.pdf ；https://arxiv.org/pdf/2410.03065 | §5–§6 全文 |
| Hoefler | Hoefler & Belli, Scientific Benchmarking of Parallel Computing Systems（SC'15） | 作者網站 PDF | https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf | 全文 |
| Heiser | Gernot Heiser, Systems Benchmarking Crimes（網頁） | 2026-10-06 抓取 | https://gernot-heiser.org/benchmarking-crimes.html | 全文 |
| Mytkowicz | Mytkowicz et al., Producing Wrong Data Without Doing Anything Obviously Wrong!（ASPLOS'09） | 課程網站鏡像 PDF | https://users.cs.northwestern.edu/~robby/courses/322-2013-spring/mytkowicz-wrong-data.pdf | 摘要、§1、§6–§7 |
| Kalibera | Kalibera & Jones, Rigorous Benchmarking in Reasonable Time（ISMM'13） | Kent 典藏「Updated Version」（修正過 ISMM 版），經鏡像取得 | http://petertsehsun.github.io/soen691/current/papers/reasonable_benchmarking.pdf （原典藏頁 https://kar.kent.ac.uk/33611/） | 全文 |
| SIGPLAN | Berger, Blackburn, Hauswirth, Hicks, SIGPLAN Empirical Evaluation Checklist（2018-10） | 官方 PDF | https://www.sigplan.org/Resources/EmpiricalEvaluation/ ；https://github.com/SIGPLAN/empirical-evaluation/raw/master/checklist/checklist.pdf | 全文 |
| LeBoudec | J.-Y. Le Boudec, Performance Evaluation of Computer and Communication Systems（書） | 作者網站 PDF（〔複核補充〕封面標 Version 2.3.2 of July 26, 2026；書 p33＝PDF p55，書 p313–317＝PDF p335–339） | https://leboudec.github.io/perfeval/book/perf.pdf | Thm 2.1（書 p33）、附錄 A 表（書 p313–317） |
| PT-CUDA | PyTorch 文件 CUDA semantics | docs 2.14（stable 轉址） | https://docs.pytorch.org/docs/2.14/notes/cuda.html | Asynchronous execution、Memory management、CUDA Graphs |
| PT-bench | PyTorch Benchmark recipe | 2026-10-06 抓取 | https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html | 第 3–4 步 |
| NCU | NVIDIA Nsight Compute Profiling Guide | 2026-10-06 抓取 | https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html | §2.6.2–2.6.4 |
| SMI | nvidia-smi 文件 | 2026-10-06 抓取 | https://docs.nvidia.com/deploy/nvidia-smi/index.html | -lgc、Clocks Event Reasons、Compute Mode、Utilization |
| MPS | NVIDIA Multi-Process Service 文件 | 2026-10-06 抓取 | https://docs.nvidia.com/deploy/mps/index.html | 概述 |
| MLPerf | MLCommons Inference Rules | commit d3eba2f（2026-10-06 的 HEAD） | https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc | Scenarios 表、最小 query 數表、Appendix Early Stopping |
| Vidur | Agrawal et al., Vidur: A Large-Scale Simulation Framework for LLM Inference（MLSys'24） | arXiv 2405.05465v2（PDF 內標 Proceedings of the 7th MLSys） | https://arxiv.org/pdf/2405.05465 | §5、§7.1–7.2、附錄 A |
| LLMServingSim | Cho et al.（IISWC'24〔二手：搜尋結果與 dblp〕） | arXiv 2408.05499v1 | https://arxiv.org/pdf/2408.05499 | §V-B 驗證 |
| Splitwise | Patel et al.（ISCA'24〔二手：IEEE Xplore 搜尋結果〕） | arXiv 2311.18677v2 | https://arxiv.org/pdf/2311.18677 | §V-B 模擬器 |
| CausalSim | Alomar et al., CausalSim（NSDI'23〔未在 PDF 內確認 venue〕） | arXiv 2201.01811v4 | https://arxiv.org/pdf/2201.01811 | 摘要、§1 |
| Rethinking | Gao et al., Rethinking KV Cache Compression Techniques for LLM Serving（MLSys'25） | arXiv 2503.24000v1（PDF 內標 8th MLSys） | https://arxiv.org/pdf/2503.24000 | §3 Missing Pieces、§4 Observations |
| ACM | ACM Artifact Review and Badging, Version 1.1（2020-08-24） | acm.org 直連回 403；讀 Wayback 2025-05-26 快照 | https://web.archive.org/web/20250526121626/https://www.acm.org/publications/policies/artifact-review-and-badging-current | 全文 |
| SA | sysartifacts.github.io：SOSP 2026、EuroSys 2027、ATC 2026 的 Badges 與 AE 頁 | 2026-10-06 抓取 | https://sysartifacts.github.io/sosp2026/badges 等 | 全文 |
| OSDI-CFA | OSDI '26 Call for Artifacts | 2026-10-06 抓取 | https://www.usenix.org/conference/osdi26/call-for-artifacts | 全文 |
| OSDI-CFP | OSDI '26 Call for Papers | 2026-10-06 抓取 | https://www.usenix.org/conference/osdi26/call-for-papers | 評審標準段 |
| MLSys-AE | MLSys'26 Call for Artifact Evaluations | 2026-10-06 抓取 | https://mlsys.org/Conferences/2026/CallForAEs | 全文 |
| cTuning | cTuning AE reviewing guide（MLSys AE 頁連過去） | GitHub commit dfb0228（2026-10-06 HEAD） | https://github.com/ctuning/artifact-evaluation/blob/master/docs/reviewing.md | Results reproduced 段 |

**沒讀到的**：OpenReview 上所有審稿意見（含 Cake 的 forum `WOyOtaO6lQ`、SCBench 的 `gkUyYcY1W9`）。openreview.net 與 api2.openreview.net 都回「Verifying your browser」人機驗證頁（curl、WebFetch、內建瀏覽器、Chrome 擴充都一樣）；依規則不代為通過驗證。Wayback 的快照只有論文本體，審稿內容是前端另外載入的，沒被存到。Hugging Face 上的 OpenReview 鏡像資料集，datasets-server 一直回「index is loading」。詳見 §5 與「未查證清單」。E03、E06 卡也記錄了同樣的阻擋。

---

## 重點摘要

1. **七篇的評測章節骨架相近**〔判讀〕：setup（硬體、模型、負載、對手、指標）→ 端到端（掃請求率或負載）→ 拆解／消融 → 敏感度 → 開銷。章首交代「這節要回答什麼」的只有一部分：Sarathi、Mooncake、Cake 把問題清單直接寫在章首（Sarathi p11、Mooncake p9、Cake p5）〔原文〕；Strata 用問句當小節標題（p10–p13）〔原文〕；DistServe 章首是結果摘要＋節次（p10）〔原文〕。〔複核修正〕原寫「七篇…先一段『這節要回答哪些問題』」並標〔原文〕，與原文不符：vLLM §6 章首只有一句「在多種負載下評估效能」（vLLM p9），CachedAttention §4 直接進入 4.1 Experimental Setup（CA p9）。
2. **七篇都沒有報重複次數、誤差棒或信賴區間**（對全文做關鍵字搜尋：repeat、standard deviation、confidence interval、variance、error bar 等，只找到與統計無關的用法）。DistServe 改用「SLO 達成率 90%／99%」，Sarathi 用 P99 TBT 與中位數排程延遲上限 2 s 定義容量，等於把變異放進指標定義裡，但沒有跨次重複。〔原文，關鍵字搜尋〕
3. 方法學文獻的共同底線：**先判斷量測是否具決定性；非決定性的資料要給信賴區間；不假設常態；比率不能只給相對值；要和上界或理想值比**（Hoefler Rule 1、4–7、11）。Hoefler 抽樣 120 篇 HPC 論文、其中 95 篇適用〔複核修正：原寫「抽樣 95 篇」；Hoefler p2：25/120 不適用〕，只有 2 篇給平均值的信賴區間（p4）；Kalibera 抽樣 2011 年 PLDI／ASPLOS／ISMM 等 90 篇量執行時間的論文，71 篇沒有任何變異量（p2）。〔原文〕
4. **量尾端延遲需要的樣本數遠比直覺多**：用 Le Boudec 的無母數法，中位數的 95% 信賴區間至少要 6 個樣本，第 95 百分位至少 59 個（書 p313–317）；同法算第 99 百分位至少 299 個〔計算〕。MLPerf 為了在 99% 信心、0.05% 誤差界內估 P99，建議至少 270,336 個 query，且每次跑至少 600 秒（MLPerf rules 表）。〔原文〕〔文件〕
5. **GPU 計時三大坑都有官方文件可引**：非同步執行（不同步就只量到 launch 時間）、第一次呼叫的初始化（PyTorch 範例：bmm 第一次 2775.5 µs，第二次 22.4 µs）、時脈與熱節流（nvidia-smi 的 HW Thermal Slowdown 會把時脈降一半以上；鎖頻要 root）。〔文件〕
6. **模擬被接受的模式很固定**：真機做端到端主結果，模擬只拿來做真機做不到的消融或 what-if（DistServe、Mooncake、Strata；〔複核修正〕Cake 是例外：所有主結果的 I/O 都以延遲模擬，見 §3.2，Cake PMLR p5），而且**同一份論文裡用真機校驗模擬器**——DistServe 誤差 < 2%（p13 Table 2），Vidur 靜態負載 P95 誤差最多 3.33%、85% 容量下多數 < 5%、7B 在 95% 容量下最高 12.65%（p9、p15），Splitwise 的效能模型用 80:20 切分驗證，MAPE < 3%（p9）。〔原文〕
7. **trace 驅動模擬的根本前提**是「新策略不會改變被重播的 trace」（CausalSim 稱為 exogenous trace assumption，p1）。多輪對話的下一輪到達時間取決於上一輪多快回覆，就違反這個前提。Heiser 也點名「拿含同一組假設的模擬器評估自己的模型」毫無價值。〔原文〕〔判讀：多輪對話的例子〕
8. **Artifact evaluation 都在論文錄取之後、與錄取無關**，但時間很短：SOSP'26 錄取通知 7/3，artifact 7/13 就要交（SA）；〔複核補充〕EuroSys'27 春季梯次 8/21 → 8/31、秋季 2027-01-29 → 02-08，同樣 10 天；ATC'26 9/18 → 9/22 只有 4 天（SA eurosys2027、atc2026 首頁「Important Dates」）。OSDI'26 今年只評「Artifacts Available」一個 badge（OSDI-CFA）；SOSP'26、EuroSys'27、ATC'26 只能申請三種固定組合，不含 Reusable（SA）；MLSys'26 評 Available、Functional、Reproduced，並鼓勵拿 Results Reproduced（MLSys-AE）。〔文件〕
9. **OpenReview 審稿意見本次讀不到**（人機驗證）。改用可讀的公開批判（Rethinking'25 的 Missing Pieces、Heiser、SIGPLAN checklist）整理出 15 條有出處的評測批評；它們**不是**審稿意見，§5 分開標示。〔原文〕〔文件〕

---

## 1. 評測章節的結構拆解

### 1.1 逐篇

#### vLLM（SOSP'23；arXiv 2309.06180v1）

| 小節（原文） | 頁 | 回答的問題 | 圖表（形狀） | 出處 |
|:--|:--|:--|:--|:--|
| 6 Evaluation（章首一句） | p9 | 在多種負載下評估效能 | — | 〔原文〕p9 |
| 6.1 Experimental Setup | p10 | 模型 OPT-13B/66B/175B、LLaMA-13B；GCP A2（A100）；ShareGPT、Alpaca 的長度＋Poisson 到達；指標 normalized latency（端到端延遲 ÷ 輸出長度）；多數 trace 1 小時，175B 因成本只跑 15 分鐘；對手 FasterTransformer、Orca（Oracle／Pow2／Max 三種自己重做的變體） | Table 1（設定）、Fig. 11（長度分布直方圖） | 〔原文〕p9–p10 |
| 6.2 Basic Sampling | p11 | 一般取樣時能撐多高請求率 | Fig. 12：2×3 折線格（x＝請求率，y＝normalized latency）；Fig. 13：長條（平均同時 batch 的請求數） | 〔原文〕p10–p11 |
| 6.3 Parallel Sampling and Beam Search | p11 | 共享 KV 的效益隨 parallel size／beam width 怎麼變 | Fig. 14：2×3 折線格；Fig. 15：長條（記憶體節省 %） | 〔原文〕p11 |
| 6.4 Shared prefix | p11–p12 | 共享前綴（1-shot／5-shot 翻譯提示）時的吞吐 | Fig. 16：兩張折線 | 〔原文〕p12 |
| 6.5 Chatbot | p12 | 多輪聊天負載（截到最後 1024 token） | Fig. 17：折線 | 〔原文〕p12 |
| 7 Ablation Studies／7.1 Kernel Microbenchmark | p12 | PagedAttention kernel 比 FasterTransformer 慢多少（20–26%） | Fig. 18a：kernel 延遲 vs context 長度 | 〔原文〕p12 |
| 7.2 Impact of Block Size | p12 | block size 怎麼選（預設 16 的理由） | Fig. 18b：長條（不同 block size 的延遲） | 〔原文〕p12 |
| 7.3 Comparing Recomputation and Swapping | p13 | 重算與換出在不同 block size 下誰划算 | Fig. 19a：微基準長條；19b：端到端 | 〔原文〕p13 |

#### Sarathi-Serve（OSDI'24，USENIX 版）

| 小節（原文） | 頁 | 回答的問題 | 圖表（形狀） | 出處 |
|:--|:--|:--|:--|:--|
| 5 Evaluation（章首列 4 個問題） | p11 | 見下列四節 | Table 1–3（模型／資料集／SLO） | 〔原文〕p11 |
| 5.1 Capacity Evaluation | p11–p12 | 嚴格與寬鬆 SLO 下一個副本最多能撐多少 QPS；SLO 定為無干擾 decode 迭代時間的 5×／25×；「可持續」定義為中位數排程延遲 ≤ 2 s | Fig. 10–11：分組長條（capacity，SLO-S／SLO-R × 模型） | 〔原文〕p11–p12 |
| 5.2 Throughput-Latency Tradeoff | p12–p13 | 改變 P99 TBT SLO 時容量怎麼變 | Fig. 12：折線（x＝P99 TBT SLO，y＝容量） | 〔原文〕p12–p13 |
| 5.3 Making Pipeline Parallel Viable | p13 | 跨節點時 TP 與 PP 的比較 | Fig. 13：長條（a 中位數 TBT，b 容量） | 〔原文〕p13 |
| 5.4 Ablation Study／5.4.1 Overhead of chunked-prefills | p13–p14 | 切塊對 prefill 的額外開銷（chunk 512 最多約 25%） | Fig. 14：長條（正規化開銷 vs prompt 長度） | 〔原文〕p14 |
| 5.4.2 Impact of individual techniques | p14 | 兩個技術單獨用與合用的 TTFT／TBT（128 個請求） | Table 4 | 〔原文〕p14 |

指標：TTFT 取中位數、TBT 取 P99，理由是 TTFT 每個請求只有一個值、TBT 每個 token 都有一個值（p11）。〔原文〕

#### DistServe（OSDI'24，USENIX 版）

| 小節（原文） | 頁 | 回答的問題 | 圖表（形狀） | 出處 |
|:--|:--|:--|:--|:--|
| 6 Evaluation（章首摘要結果與節次） | p10 | — | — | 〔原文〕p10 |
| 6.1 Experiments Setup | p10–p11 | 4 節點 32×A100、跨節點 25 Gbps；OPT（故意選 MHA 以加重傳輸壓力）；三種應用各定 TTFT／TPOT SLO；Poisson 到達；主指標 SLO 達成率（看 90%，附錄 C 看 99%）；對手 vLLM、DeepSpeed-MII | Table 1、Fig. 7（長度分布） | 〔原文〕p10–p11 |
| 6.2 End-to-end Experiments | p11–p12 | 每 GPU 能撐多高請求率、能撐多嚴的 SLO | Fig. 8–9：折線（x＝每 GPU 請求率或 SLO Scale，y＝SLO 達成率；90% 垂直線） | 〔原文〕p11–p12 |
| 6.3 Latency Breakdown | p12 | KV 傳輸佔多少時間（OPT-175B 低於 0.1%） | Fig. 10：左為五段堆疊長條，右為傳輸時間 CDF | 〔原文〕p12 |
| 6.4 Ablation Studies | p13 | 分離與放置演算法各自的貢獻；**此節用模擬器**，並先報模擬器誤差 < 2% | Table 2（模擬 vs 真機的 SLO 達成率）、Fig. 11（折線） | 〔原文〕p13 |
| 6.5 Algorithm Running Time | p13 | 放置演算法隨 GPU 數的執行時間 | Fig. 12：長條 | 〔原文〕p13 |

#### CachedAttention（ATC'24，USENIX 版）

| 小節（原文） | 頁 | 回答的問題 | 圖表（形狀） | 出處 |
|:--|:--|:--|:--|:--|
| 4 Performance Evaluation／4.1 Experimental Setup | p9–p10 | 4×A100、128 GB DRAM、10 TB SSD；LLaMA／Falcon；ShareGPT 9K 對話、Poisson（λ=1.0 session/s）；**前 10K 輪拿來暖機**；對手只有重算（RE） | — | 〔原文〕p9–p10 |
| 4.2 End-to-end Performance | p10–p11 | 命中率、TTFT、prefill 吞吐、GPU 時間、雲端成本 | Fig. 13–17：每個模型一組長條 | 〔原文〕p10–p11 |
| 4.3 Ablation Studies：4.3.1 Recomputation v.s. CachedAttention | p11 | 不同「歷史／新 token」比例下 prefill 時間 | Fig. 18：分組長條 | 〔原文〕p11 |
| 4.3.2 Overlapped KV cache Access | p11 | 逐層預載與非同步寫回各省多少 | Fig. 19–20：長條 | 〔原文〕p11 |
| 4.3.3 Scheduler-aware Fetching and Eviction | p12 | 與 LRU、FIFO 比命中率（DRAM／disk 分開） | Fig. 21：堆疊長條 | 〔原文〕p12 |
| 4.3.4 Performance of Decoupled KV Cache Truncation | p12 | context 溢出時不失效的效益 | Fig. 22：長條 | 〔原文〕p12 |
| 4.3.5 Accuracy of Decoupled Positional Encoding | p12–p13 | 品質是否保持（PPL、MMLU／LongEval／PIQA） | Table 1–2 | 〔原文〕p12–p13 |
| 4.3.6 The Cache Capacity Requirement | p13 | 要多少容量才夠 | Fig. 23：折線 | 〔原文〕p13 |
| 4.3.7 Impact of Caching Storage Mediums | p13–p14 | 只用 HBM、HBM+DRAM、加 SSD 的差別 | Fig. 24：長條 | 〔原文〕p13–p14 |
| 4.3.8 Impact of Session Arrival Rates | p14 | 到達率 0.5–2.0/s 的影響 | Fig. 25：長條 | 〔原文〕p14 |

#### Mooncake（FAST'25，USENIX 版）

| 小節（原文） | 頁 | 回答的問題 | 圖表（形狀） | 出處 |
|:--|:--|:--|:--|:--|
| §4.2 末段（評測前的排程實驗） | p9 | 隨機、負載平衡、本地與全域快取感知排程的 TTFT | Fig. 5（圖在 p8）〔複核修正：原只寫 p9；文字在 p9，Fig. 5 圖與圖說在 PDF p8〕 | 〔原文〕p8–p9 |
| 5 Evaluation（章首兩個問題；用 dummy LLaMA3-70B） | p9 | (1) 真實場景是否勝過現有系統 (2) Mooncake Store 是否比傳統前綴快取好 | — | 〔原文〕p9 |
| 5.1 Setup | p9 | 8×A800＋4×200 Gbps RDMA；TBT 定義為 token 間隔最長 10% 的平均；TTFT 門檻 30 s，TBT 100／200／300 ms；有效請求比例；對手 vLLM v0.5.1（prefix caching 與 chunked prefill 分開測） | — | 〔原文〕p9 |
| 5.2 End-to-end Performance：5.2.1 Workload | p10 | 兩份 Kimi 真實 trace（依時間戳重播）＋一份公開資料合成（Poisson） | Table 2 | 〔原文〕p10 |
| 5.2.2 Effective Request Capacity | p10–p11 | 不同 TBT 門檻下的有效請求比例 | Fig. 1（p2）、6、7：曲線（x＝TBT，y＝有效請求比例） | 〔原文〕p2、p10–p11 |
| 5.2.3 Prefill GPU Time | p11 | 每請求 prefill 的 GPU 時間 | Fig. 8：分組長條 | 〔原文〕p11 |
| 5.3 Mooncake Store：5.3.1 Quantitative Analysis of Cache Capacity | p11–p12 | 只看請求序列的理論最高命中率 vs 容量（**上界分析**） | Fig. 9：折線（x＝容量，對數軸） | 〔原文〕p11–p12 |
| 5.3.2 Practical Workload Experiment | p12 | 全域 vs 本地快取（輸出限 1 token 以隔離 decode） | Fig. 10：長條 | 〔原文〕p12 |
| 5.3.3 Cache Replica | p12 | 熱門 key 的副本數隨時間 | Fig. 11：折線 | 〔原文〕p12 |
| 5.4 KVCache Transfer Performance：5.4.1 Transfer Engine | p12–p13 | 與 Gloo、TCP 比傳輸延遲 | Fig. 12：折線 | 〔原文〕p13 |
| 5.4.2 Bandwidth Demand by Mooncake | p13 | **模擬** 24–400 Gbps 頻寬下的 TTFT | Fig. 13：折線 | 〔原文〕p13 |
| 5.4.3 E2E Latency Breakdown | p13 | 五段延遲（前綴快取 0% vs 95%） | Fig. 14：堆疊長條 | 〔原文〕p13 |
| 5.5 P/D Ratio | p13–p14 | prefill／decode 節點比例 | Fig. 15：長條＋折線 | 〔原文〕p13–p14 |

#### Strata（OSDI'26，USENIX 版）

| 小節（原文） | 頁 | 回答的問題 | 圖表（形狀） | 出處 |
|:--|:--|:--|:--|:--|
| 5.1 Methodology | p9–p10 | 三個平台（8×H200、8×H20＋NVMe、GH200）；對手 vLLM v0.8.5、vLLM-LMCache v0.2.1、TRT-LLM v0.17.0（＋HiCache）、SGLang v0.4.5、自己做的 SGLang-HiCache；Poisson；in-flight 上限 128；ShareGPT 插 60 s 思考時間；主指標平均 TTFT 與輸出 token 吞吐 | Table 1 | 〔原文〕p9–p10 |
| 5.2.1 How does the performance of Strata compare to state-of-the-art…? | p10 | 長 context 負載下的吞吐–TTFT | Fig. 8：3 模型 × 4 資料集的曲線格（x＝吞吐，y＝平均 TTFT） | 〔原文〕p10–p11 |
| 5.2.2 How does Strata perform with a warm cache? | p10 | 先把 CPU 記憶體填滿再跑的穩態 | Fig. 8 第二列 | 〔原文〕p10 |
| 5.2.3 How does Strata perform with short-context? | p10–p11 | 短 context 下是否退化 | Fig. 8 末列 | 〔原文〕p10–p11 |
| 5.3.1 How much do efficient I/O and scheduling benefit Strata? | p11–p12 | 兩個元件各自的貢獻 | Fig. 9：吞吐–延遲曲線 | 〔原文〕p11–p12 |
| 5.3.2 Can Strata alleviate the burden of choosing a page size? | p12 | 對 page size 的敏感度 | Fig. 10：正規化峰值吞吐 vs page size | 〔原文〕p12 |
| 5.3.3 Can Strata adapt to varying cache distances? | p12 | 請求排列（min／shuffle／max 距離）下各優化的歸因 | Fig. 11：堆疊長條 | 〔原文〕p12 |
| 5.3.4 What causes delay hit? | p12–p13 | delay hit 與解析時間、吞吐的關係；**用模擬執行 Mooncake Tool-Agent trace**，因測試平台撐不到那麼高的吞吐 | Fig. 12：折線 | 〔原文〕p12–p13 |
| 5.3.5 Does the decoupled memory layout benefit disk caching? | p13 | 換到 H20＋NVMe 平台、DeepSeek-V3 | Fig. 13：長條 | 〔原文〕p13 |
| 5.4 Benchmark on GH200 machine | p13 | 新硬體上的表現；**Strata-Oracle 模擬 CPU–GPU 無限頻寬當上界** | Fig. 14：曲線；Fig. 15：長條（持續頻寬） | 〔原文〕p13 |
| 6 Discussion | p14 | 限制：I/O kernel 與計算搶 SM、排程公平性、模型涵蓋範圍 | — | 〔原文〕p14 |

#### Cake（ICML'25，PMLR 版）

| 小節（原文） | 頁 | 回答的問題 | 圖表（形狀） | 出處 |
|:--|:--|:--|:--|:--|
| 5 Evaluation（章首列 7 個因素） | p5 | 頻寬與算力、長度、架構、壓縮、資源波動、調適排程、開銷 | — | 〔原文〕p5 |
| 5.1 Experiment Setup | p5–p6 | 對手 vLLM v0.6.2 chunked prefill（compute-only）、LMCache v0.1.4（I/O-only）；2×A100 與 1×H100；**I/O 頻寬用延遲模擬**；「GPU 利用率」用 token budget 佔比模擬；主指標 TTFT | Table 1–2 | 〔原文〕p5–p6 |
| 5.2 Evaluation Across Compute and I/O | p6 | 頻寬 × 硬體 × 利用率的加速比 | Table 3（加速比表，以顏色標示） | 〔原文〕p6 |
| 5.3 Evaluation Across Varying Chunk Sizes | p6–p7 | chunk 大小的影響（**PMLR 版新增**，arXiv v2 沒有） | Table 4 | 〔原文〕p6–p7 |
| 5.4 Evaluation Across Sequence Lengths | p7 | 長度的影響 | Table 5 | 〔原文〕p7 |
| 5.5 Evaluation Across Model Architectures | p7 | MHA vs GQA、模型大小 | Table 6 | 〔原文〕p7–p8 |
| 5.6 Incorporating KV cache Compression with Cake | p8 | 8-bit、3-bit 量化下的效益 | Table 7 | 〔原文〕p8 |
| 5.7 Handling Fluctuations in Resources | p8 | 頻寬 0–25 Gbps、算力 0–512 token 隨機波動 | Fig. 5：時間軸 trace | 〔原文〕p8 |
| 5.8 Evaluation on Adaptive Scheduling | p8 | 一個 16K 請求＋22 個短請求的例子 | Fig. 6：token batch size 隨時間 | 〔原文〕p8 |
| 5.9 Overheads of Cake | p8–p9 | 每個 engine step 的時間是否變慢 | Fig. 7：逐 step 時間 | 〔原文〕p8–p9 |
| 6 Discussion | p9 | 與其他加速法相容；多節點分散式（**以論述帶過，沒有實驗**）（PMLR 版新增） | — | 〔原文〕p9 |

Cake 的平均加速比**排除了紅色標示的點**（對某一個對手超過 10×），理由是與極弱的對手比會誤導（p6）。〔原文〕

### 1.2 標準評測組件表

✓＝有；位置為原文節次與 PDF 頁。分類由本卡判斷〔判讀〕，各格內容的事實出處見 1.1。

| 組件 | vLLM | Sarathi | DistServe | CachedAttention | Mooncake | Strata | Cake |
|:--|:--|:--|:--|:--|:--|:--|:--|
| 端到端（掃負載） | ✓ §6.2–6.5 p11–12 | ✓ §5.1 p11–12 | ✓ §6.2 p11–12 | ✓ §4.2 p10–11（固定到達率） | ✓ §5.2 p10–11 | ✓ §5.2 p10–11 | 單請求 TTFT（§5.2 p6），沒有掃負載 |
| 拆解（時間或來源分項） | 機制指標：Fig. 13 batch 數、Fig. 15 記憶體節省 | — | ✓ §6.3 p12 五段延遲＋傳輸 CDF | 命中率分 DRAM／disk §4.3.3 p12 | ✓ §5.4.3 p13 五段延遲 | ✓ §5.3.3 p12 各優化的歸因 | — |
| 消融（拿掉某元件） | ✓ §7 p12–13 | ✓ §5.4.2 p14 | ✓ §6.4 p13（模擬） | ✓ §4.3.1–4.3.4 p11–12 | ✓ §5.3.2 p12；排程 Fig. 5（圖 p8，文 p9）〔複核修正〕 | ✓ §5.3.1 p11–12 | 兩個對手即兩種退化情形（§5.2）；無元件消融 |
| 敏感度（掃參數） | block size §7.2；parallel／beam §6.3 | SLO §5.2 p12–13 | SLO Scale Fig. 8 第二列；99% 附錄 C | 容量 §4.3.6、媒介 §4.3.7、到達率 §4.3.8 | 頻寬 §5.4.2；P/D §5.5 | page size §5.3.2；cache distance §5.3.3 | chunk §5.3、長度 §5.4、架構 §5.5、壓縮 §5.6 |
| 開銷 | ✓ §7.1 kernel 慢 20–26% | ✓ §5.4.1 最多約 25% | 演算法執行時間 §6.5 | 寫回開銷 §4.3.2 | 傳輸開銷含在 §5.4.3 | §6 只在討論中談 SM 爭用 | ✓ §5.9 p8–9 |
| 不退化（保守準則） | — | — | — | 品質 §4.3.5（PPL、準確度） | — | ✓ §5.2.3 短 context | — |
| 擴展性（規模、模型、硬體） | 13B–175B、1／4／8 GPU（Fig. 12） | 7B–180B；跨節點 §5.3 | 13B–175B；GPU 數 §6.5 | 13B–70B | 16 節點叢集 | 8B–70B、三平台、§5.4 GH200 | 7B–70B；多節點只論述（§6） |
| 與上界／理想比較 | Orca (Oracle)：假設事先知道輸出長度（p10） | — | — | — | 理論最高命中率 §5.3.1 | Strata-Oracle §5.4 | — |
| 案例或部署證據 | — | — | 選出的放置設定（p11、附錄 B） | 雲端成本換算（§4.2 p11） | Kimi 歷史統計（p9）；副本行為 §5.3.3 | — | 調適排程範例 §5.8 |
| 模擬的使用 | 無 | 無 | 消融（已校驗） | 無 | 頻寬掃描 | delay-hit 分析、Oracle | I/O 頻寬全部以延遲模擬 |
| 重複次數／CI | 無 | 無 | 無 | 無 | 無 | 無 | 無 |

**判讀**：
- 頂級 systems 會議的評測，最少要有「端到端＋消融＋開銷」三件；有上界比較的只有 vLLM、Mooncake、Strata 三篇，但這正是審稿人最容易接受的「還剩多少空間」論證（Hoefler Rule 11、Heiser 的 proper baseline 一節都要求）。
- Cake 是唯一只量單請求 TTFT、沒有掃負載的；ICML 是 ML 會議，標準與 OSDI 不同。若要投 systems 場所，必須補「多請求、真實到達、端到端吞吐–延遲曲線」。
- 七篇都沒有統計重複，代表「沒有 CI 也能上」；但這不代表審稿人不會問，而是這些系統的效益通常是 2× 以上，遠大於雜訊。效益只有 5–15% 時（本研究 go/no-go 的灰區），沒有變異量的結果很容易被質疑（Kalibera p2 引 Mytkowicz：領域內效能進步的中位數只有 10%）。

---

## 2. 量測與統計的嚴謹性

### 2.1 Hoefler & Belli（SC'15）十二條規則

背景：抽樣 2011–2014 年 HPDC、SC、PPoPP 共 120 篇，95 篇適用；只有 15 篇提到任何變異量，只有 2 篇給平均值的信賴區間（p2、p4）。〔原文〕

| # | 規則（改寫） | 出處 | 對本研究〔判讀〕 |
|:--|:--|:--|:--|
| 1 | 報加速比要說明基準是什麼，並給基準的絕對值；推廣：**不要只給比率** | p3 | 每張加速比圖都要附「全 GPU」的絕對 TTFT 或吞吐 |
| 2 | 只報部分 benchmark 或沒用滿資源時要說明原因；報全部結果，不只報最好的 | p3 | 選哪些 trace、哪些長度要寫理由 |
| 3 | 算術平均只用在成本（時間）；速率用調和平均 | p3 | tokens/s 不能直接算術平均 |
| 4 | 不要對比率做平均；改對成本或速率做平均；不得已才用幾何平均 | p4 | 跨 trace 的平均加速比要從時間算回來 |
| 5 | 說明量測是否具決定性；非決定性的資料要給信賴區間（例：收集到 99% CI 落在平均的 5% 內為止） | p4 | 模擬器重播是決定性的，要明說；真機量測給 CI |
| 6 | 沒有診斷前不要假設常態 | p5 | 延遲分布常右偏，用無母數 CI |
| 7 | 非決定性的資料要用統計上站得住的方式比較（不重疊的 CI 或 ANOVA） | p6 | 兩策略差 5% 時要有 CI |
| 8 | 想清楚平均或中位數是否有意義；最差延遲之類要看其他百分位 | p6 | TTFT 報中位數＋P99 |
| 9 | 記錄所有變動因子與其水準，以及完整的軟硬體設定 | p8 | 對應 CLAUDE.md §4 的 run 記錄 |
| 10 | 平行時間量測要說明量測、同步、彙總方式 | p8 | 多請求時「誰的時間」要定義 |
| 11 | 可以的話，畫出效能上界 | p9 | Oracle 上界（Belady／offline 最佳） |
| 12 | 畫出解讀所需的資訊；只有表示趨勢且內插有效時才連線 | p10 | 離散設定（如層級組合）不要連線 |

其他重點：無母數中位數 CI 的做法引用 Le Boudec（p5）；**量測數要超過 5 個（n > 5，即至少 6 個）才能用無母數法估 CI**（原文「n > 5」，p8）〔複核修正：原寫「至少要 5 個以上」，中文「5 個以上」可讀成含 5，與 n > 5 不符〕；建議每做 k 次就重算 CI、達到要求就停（p8）；第一次迭代要排除（warmup），並考慮 cache 是冷是熱（p7）；不建議刪除離群值，改用百分位等穩健統計（p5）。〔原文〕

### 2.2 Heiser：Systems Benchmarking Crimes

Heiser 寫明：審稿時看到這些錯誤，論文就已「大半進了拒絕區」（網頁開頭段）。〔文件〕

| 類別 | 罪狀（改寫） | 對本研究〔判讀〕 |
|:--|:--|:--|
| 選擇性評測 | **沒評估可能的退化**：要同時證明「關心的地方變好」與「別處沒變差」 | 短 context、低負載下 Tiara 不能比 LRU 差，要量 |
| | 沒有充分理由就只用 benchmark 子集，還報整體平均 | 只挑對自己有利的 trace |
| | 輸入範圍挑在系統好看的區段（例：32 核機器只測到 32 個 client） | context 長度只測到 κ 對自己有利的區段 |
| 結果處理不當 | 拿微基準代表整體效能 | 只報搬運微基準、不報端到端 |
| | 吞吐只降 x% 就說開銷是 x%；要報 CPU 負載、每單位成本 | 預測器的 CPU 開銷要單獨報 |
| | 淡化開銷：百分點當百分比、分母挑對自己有利的、算錯 | 「改善 x%」的分母一律是基準 |
| | **沒有顯著性**：至少給標準差；系統常很決定性，可寫「所有標準差 < 1%」；有疑慮就做 t 檢定 | 每個數字附標準差或 CI |
| | 對正規化後的分數用算術平均；應用幾何平均 | 同 Hoefler Rule 4 |
| 用錯 benchmark | **評估簡化的模擬系統**：模擬是模型，不能含會影響所測效能面向的簡化假設 | 見 §3 |
| | 用不能說明問題的 benchmark | — |
| | **校準與驗證用同一份資料**：兩者必須完全不交集 | 成本常數的量測集與驗證集要分開 |
| 比較不當 | 沒有適當基準；基準常是最先進方法、最佳解或硬體極限 | 加 Oracle 上界與「全 GPU」 |
| | 只跟自己比；或用含同樣假設的模擬器評估自己的模型 | 見 §3 |
| | **不公平地測對手**：要寫清楚對手的所有設定；結果與對手發表的數字不符時要特別小心 | LMCache、vLLM offload 要用建議設定，並寫出參數 |
| | 不跟最新方法比，以膨脹改善幅度 | 基準要是 Cake／Strata 等最新者 |
| 缺資訊 | 平台規格不全（CPU 型號、核心數、記憶體、網路、OS 與版本） | 照 Hoefler Rule 9 |
| | 只報套件總分、不報子項 | 每個 trace 分開報 |
| | **只給相對數字** | 同 Hoefler Rule 1 |

網頁末的「最佳實務」：開始前確認系統靜止、驗證讀寫的資料正確、同一點連續跑與隔開跑各兩次、反轉量測順序、不要只用 2 的冪次（2ⁿ±1 常是病態點）、多跑幾次看標準差（他們領域通常 < 0.1%，> 1% 要警覺）、跑足暖機迭代。〔文件〕

### 2.3 Mytkowicz et al.（ASPLOS'09）

- **主張**：看似無關的實驗設定會造成「量測偏差」，足以讓結論相反。改變 UNIX 環境變數的大小，執行時間常變約 33%、一次接近 300%（p2 Fig. 1b；每點為 5 次平均＋95% CI）。〔原文〕
- **普遍**：Pentium 4、Core 2、m5 模擬器，gcc 與 icc，多數 SPEC CPU2006 C 程式都有（p1）。〔原文〕
- **文獻調查**：ASPLOS、PACT、PLDI、CGO 共 133 篇，88 篇有評測方法段落，沒有一篇妥善處理量測偏差（p1、p8）。〔原文〕
- **對策 1：setup randomization**——在多種設定下各量一次，得到分布再做統計；例：22 種連結順序 × 22 種環境大小＝484 種設定，每種跑 3 次，用 t 檢定得 O3 加速比 1.007 ± 0.003（p9–p10）。〔原文〕
- **對策 2：causal analysis**——介入（只動被懷疑的因子）→ 量測 → 確認結果是否如預期改變（p10）。〔原文〕
- **對本研究〔判讀〕**：GPU 上類似的「無關」因子有：CUDA graph 是否啟用、記憶體配置器狀態、同機其他行程、時脈狀態、請求順序（Strata 證明排列會改變哪個機制重要，p12）。trace 的時間窗或子樣本也該隨機化幾次。

### 2.4 Kalibera & Jones（ISMM'13）

- **調查**：2011 年 PLDI、ASPLOS、ISMM 等 122 篇中 90 篇量執行時間，71 篇沒有任何變異量；65 篇報執行時間比，只有 3 篇試著給比率的 CI（p2、p10）。〔原文〕
- **不推薦單純做顯著性檢定**：p 值不回答「快多少」，樣本一多任何微小差異都會「顯著」（p4）。**建議報效果量的信賴區間**，例如「95% 信心 A 比 B 快 5.5% ± 2.5%」（p4）。〔原文〕
- **量測要在「獨立狀態」**：迭代時間要 i.i.d. 才能重複取樣，否則變異與 CI 估計會有偏差（p5）；先人工檢查一次每個 benchmark／平台要多少迭代才到這個狀態（p6 建議）。〔原文〕
- **多層重複**：變異可能來自編譯、執行、迭代不同層；先做一次「定規模實驗」決定每層的最佳重複數，平台不變就不用重做（p10 建議）。最高層至少 5 次才能估變異（p11）。〔原文〕
- **比率的 CI**：用 Fieller 法（p10）。〔原文〕
- **對本研究〔判讀〕**：vLLM 一次 server 啟動＝「執行」，每個請求＝「迭代」。啟動之間的變異（CUDA graph 捕捉、記憶體配置）可能大於請求之間，應該重啟 server 做多次，而不是只在一次啟動內多送請求。

### 2.5 SIGPLAN Empirical Evaluation Checklist（2018-10）

七類：Clearly Stated Claims、Suitable Comparison、Principled Benchmark Choice、Adequate Data Analysis、Relevant Metrics、Appropriate and Clear Experimental Design、Appropriate Presentation of Results（checklist PDF）。與本研究最相關的項目〔文件〕：
- 只用（不實際的）模擬卻宣稱「在真實硬體上可行」＝主張範圍過大。
- 只在「開發時參考過的例子」上評估（tested on training set），要做交叉驗證。
- **負載產生器要 open loop**，產生的工作不能取決於受測系統的效能（gated workload generator）。
- 關鍵參數要掃一個範圍；從暖機到穩態都要考慮。
- 要量所有重要效果（例：編譯變慢也要報）。
- 試驗次數不足會把雜訊當訊號；沒報分布（只報平均或中位數）會誤導。
- 「4%、6%、7%、49% 寫成最多 49%」是誤導的摘要；比率要用對數軸畫。
- 精度要與誤差相稱（誤差 ±1% 卻寫 49.9%）。

### 2.6 GPU／LLM 特有的量測陷阱

| # | 陷阱 | 官方說法（改寫） | 出處 | 做法〔判讀〕 |
|:--|:--|:--|:--|:--|
| G1 | 非同步執行 | GPU 操作預設非同步；不同步的計時不準；要嘛先 `torch.cuda.synchronize()`，要嘛用 `torch.cuda.Event(enable_timing=True)` 記錄並在讀取前同步 | PT-CUDA「Asynchronous execution」 | kernel 級用 Event；端到端（HTTP 請求）用 client 端牆鐘，兩者不要混用 |
| G2 | 沒同步的 timeit 只量到 launch | 範例中 timeit 量 bmm 得 22.4 µs，`torch.utils.benchmark` 得 181.04 µs，差異來自 timeit 沒有同步 | PT-bench 第 3 步 | — |
| G3 | 第一次呼叫的初始化 | 同範例 bmm 第一次 2775.5 µs、第二次 22.4 µs，因 cuBLAS 第一次要載入；要先暖機 | PT-bench 第 3 步 | 每次 server 啟動先送一批不計分的請求 |
| G4 | 自動化的重複量測 | `blocked_autorange` 先增加每回合次數直到遠大於量測開銷（兼暖機），再取多次量測以估可靠度 | PT-bench 第 4 步 | 微基準直接用 `torch.utils.benchmark` |
| G5 | CUDA graph | graph 重播省掉 Python／C++／driver 的 launch 開銷，kernel 也略快；捕捉前要在 side stream 暖機；形狀與控制流要固定 | PT-CUDA「CUDA Graphs」 | 是否啟用 graph（vLLM 的 eager 模式）會改變每步延遲，必須記錄並在比較時一致 |
| G6 | 記憶體讀數 | 快取配置器保留的記憶體在 nvidia-smi 上仍顯示為已用；要用 `memory_allocated()`、`max_memory_reserved()` 等 | PT-CUDA「Memory management」 | 容量實驗同時記 nvidia-smi 與 PyTorch 計數器 |
| G7 | 時脈狀態 | 前面有沒有跑過 kernel 會改變時脈；Nsight Compute 預設把時脈限制在 boost 值以減少不確定性；**硬體或驅動的熱節流無法控制、一律優先** | NCU §2.6.2 | 長時間量測要同時記錄時脈與節流原因 |
| G8 | 熱與功耗節流 | nvidia-smi「Clocks Event Reasons」：SW Power Cap、HW Thermal Slowdown（時脈降到一半以下）、SW Thermal Slowdown 等；另有以 µs 計的累計計數器 | SMI「Clocks Event Reasons」「Clock Event Reasons Counters」 | 每個 run 前後各抓一次計數器，差值 > 0 就標記 |
| G9 | 鎖頻 | `--lock-gpu-clocks` 要 root、Volta 以後支援 | SMI「-lgc」 | 本機沒有 sudo（CLAUDE.md §3），只能記錄不能鎖；論文要寫明 |
| G10 | Profiler 的冷快取 | Nsight Compute 預設每次 replay 前清空 GPU 快取，結果等於隔離執行；要看應用內行為用 `--cache-control none` | NCU §2.6.3 | ncu 的數字不能直接當端到端成本 |
| G11 | 驅動初始化 | 沒開 persistence mode 時第一個程式要付初始化成本；建議開啟 | NCU §2.6.4；SMI「-pm」 | 本機無 root，記錄狀態即可 |
| G12 | 共用機器 | nvidia-smi 的 GPU 使用率＝取樣期間（1/6～1 秒）內「至少有一個 kernel 在跑」的時間比例；不代表 SM 用滿，也不能證明沒有別人 | SMI「Utilization」 | 以「有無外來 PID」判斷，不以使用率判斷（專案 `gpu_guard` 已這樣做） |
| G13 | 多行程共用 GPU | Compute Mode 可設為只允許一個 context（Exclusive Process）；MPS 文件說啟用 MPS 可減少 GPU context switching | SMI「Compute Mode」；MPS 概述 | 不啟用 MPS 時多行程會互相切換〔判讀〕；本機不能改 compute mode |
| G14 | 尾端延遲的樣本數 | 無母數 CI：中位數 95% 至少 6 個樣本（99% 至少 8 個）；第 95 百分位 95% 至少 59 個（99% 至少 90 個） | LeBoudec Thm 2.1（書 p33）、附錄 A 表 A.1、A.3（書 p315、317） | — |
| G15 | P99 要多少樣本 | 同一公式：CI 上界最多到最大值時，條件為 1 − 0.99ⁿ ≥ 0.95，得 n ≥ 299（99% 信心則 n ≥ 459），且此時 CI 上界就是樣本最大值 | 〔計算〕依 LeBoudec Thm 2.1 | 只有幾百個請求的 run 報 P99 意義很弱，改報 P90 或給 CI |
| | 〔複核補充〕〔計算〕重算 G14–G15 | **假設**：樣本 i.i.d.、分布連續（Thm 2.1 的前提）、雙側次序統計量 CI［X₍ⱼ₎, X₍ₖ₎］。**算式**：p 分位數的覆蓋率＝B_{n,p}(k−1) − B_{n,p}(j−1)（B＝二項 CDF）；n 固定時最大覆蓋率在 j=1、k=n，為 1 − pⁿ − (1−p)ⁿ。要求 ≥ γ 的最小 n：中位數 γ=0.95 → 1 − 2·0.5ⁿ，n=5 得 0.9375、n=6 得 0.9688，故 **6**（γ=0.99 → 8）；P95 γ=0.95 → n=58 得 0.9490、n=59 得 0.9515，故 **59**（γ=0.99 → 90）；P99 γ=0.95 → n=298 得 0.94996、n=299 得 0.95049，故 **299**（γ=0.99 → 459）。n=299 時可取的 CI 為［X₍₂₈₉₎, X₍₂₉₉₎］，覆蓋率 0.9502。中位數、P95 的結果與 LeBoudec 表 A.1、A.3 一致（n=6 的 p=0.969、n=59 的 p=0.951、n=90 的 p=0.990；書 p315、p317）。原卡「1 − 0.99ⁿ ≥ 0.95」略去了 0.01ⁿ 與 B(j−1) 兩項，數值結論不變 | Python 精確二項計算（V11，2026-10-07） | 結論成立 |
| G16 | 業界基準的規定 | MLPerf：每個情境至少跑 600 秒；在 99% 信心下估 P90／P95／P97／P99，建議最少 24,576／57,344／90,112／270,336 個 query；有 early stopping 準則，樣本少時以較高的有效百分位作為懲罰 | MLPerf Scenarios 表、最小 query 數表、Appendix Early Stopping | 這是「認證級」標準，研究論文不必照做，但可以引用來說明自己的取樣夠不夠 |
| G17 | 到達過程 | MLPerf 的 Server 情境用 Poisson 到達；SIGPLAN 要求 open-loop 負載 | MLPerf Scenarios 表；SIGPLAN | 固定併發數（closed loop）會讓慢的系統自動少收請求，低估排隊 |

---

## 3. 模擬 vs 真機

### 3.1 LLM serving 模擬器怎麼驗證

| 模擬器 | 怎麼建 | 怎麼驗證 | 報的誤差 | 出處 |
|:--|:--|:--|:--|:--|
| **Vidur**（MLSys'24） | 把模型拆成 token 級、序列級、通訊運算子，在單 GPU 做少量剖析，用隨機森林內插執行時間 | 四個模型（LLaMA2-7B/70B、InternLM-20B、Qwen-72B）× 三個 trace，在 A100／H100 的 vLLM fork 上比對；靜態負載比單請求執行時間，動態負載（Poisson）比 normalized 端到端延遲，負載設在容量的 85% | 摘要：< 9%；靜態 P95 最多 3.33%；動態多數 < 5%；附錄：7B 在 95% 容量時最高 12.65%（CPU 開銷造成誤差累積） | 〔原文〕p1、p5–p6、p8–p9、p15 |
| Vidur 的自述限制 | — | 逼近容量時系統處於臨界點，微小誤差會被排隊放大，兩邊延遲很難對上 | — | 〔原文〕p9 |
| **DistServe** 內建模擬器 | 離散事件模擬，依 DNN 執行可預測性 | 對 vLLM 與 DistServe-Low，在不同請求率下比較模擬與真機的 SLO 達成率 | 所有情況 < 2% | 〔原文〕p13 Table 2 |
| **Splitwise**（ISCA'24） | 分段線性效能模型（依 batch、輸入、輸出大小與平行設定剖析），餵給事件驅動叢集模擬器 | 效能模型用 80:20 train:test 切分；模擬器用超過 50K 次迭代的生產負載做端到端驗證 | 效能模型 MAPE < 3% | 〔原文〕p9 |
| **LLMServingSim**（IISWC'24） | 硬體／軟體協同模擬（NPU 模擬器＋系統模擬） | 與 GPU 上的 vLLM 比吞吐**隨時間的趨勢**；與 NeuPIMs 比吞吐 | 平均 14.7%；對 NeuPIMs 誤差 < 20%、幾何平均 8.88% | 〔原文〕p2、p8–p9 |

**判讀**：可信度的排序大致是「同一指標、同一負載區間、與真機逐點比對，且校準資料與驗證資料分開」（Splitwise、DistServe、Vidur）＞「比趨勢」（LLMServingSim）。

### 3.2 系統論文裡「部分模擬」的做法

| 論文 | 模擬了什麼 | 原文給的理由 | 有沒有校驗 | 出處 |
|:--|:--|:--|:--|:--|
| Cake | **I/O 頻寬**：依 chunk 大小與網路頻寬算出延遲，模擬的儲存後端在延遲到期前不把資料交給 GPU；頻寬選 7–100 Gbps，各對應一個實體設定（Table 2） | 「準確控制 I/O 頻寬並確保可重現」 | 沒有拿真實 SSD／網路比對 | 〔原文〕p5–p6 |
| Cake | **多使用者負載**：用 vLLM token budget 中 Cake 請求所佔比例代表 GPU 利用率 | 線上伺服器常同時處理多個請求 | 無 | 〔原文〕p6 |
| Mooncake | 網路頻寬 24–400 Gbps | 評估頻寬對效能的影響 | 圖 13b 同時畫「實際」與「理論」傳輸時間，顯示壅塞造成的偏離 | 〔原文〕p13 |
| Strata | delay hit：用 Mooncake trace 做模擬執行，快取設為近乎無限 | 測試平台撐不到足夠的吞吐 | 無 | 〔原文〕p12–p13 |
| Strata | Strata-Oracle：CPU–GPU 頻寬無限 | 當上界 | 不適用 | 〔原文〕p13 |
| DistServe | 消融實驗 | vLLM 不支援 inter-op 平行，且測試平台沒有高跨節點頻寬 | 有（< 2%） | 〔原文〕p13 |

### 3.3 審稿人接受 trace 驅動模擬的條件

找不到任何會議 CFP 或官方審稿指南明文規定「模擬要滿足什麼條件」：OSDI'26 CFP 只說好論文要「證明方案的實用性與效益」（原文 practicality and benefits；OSDI-CFP「Submitting a Paper」段）〔複核修正：原譯「可行性」，原文是 practicality〕。以下條件是從有出處的方法學文獻與已錄取論文的做法歸納，**歸納本身是〔判讀〕**，每條附依據：

| # | 條件 | 依據 |
|:--|:--|:--|
| S1 | 主要結論至少有一個真機端到端實驗支撐；模擬只用在真機做不到的消融或 what-if，並寫明為什麼做不到 | §1.2「模擬的使用」一列：七篇中有用模擬的是四篇（DistServe、Mooncake、Strata、Cake），其中前三篇是這樣；Cake 例外，主結果的 I/O 全部以延遲模擬（Cake PMLR p5）；DistServe p13、Strata p12–13 有寫理由。〔複核修正：原寫「七篇中有用模擬的五篇都是這樣」；§1.2 該列只有四篇用模擬，且 Cake 不符合此模式〕 |
| S2 | 在同一份論文裡，用真機在**同一個指標、同一個負載區間**驗證模擬器，報誤差 | DistServe p13；Vidur p9（並指出逼近容量時誤差會放大） |
| S3 | 校準資料與驗證資料不重疊 | Heiser「Same dataset for calibration and validation」；Splitwise 80:20（p9）；SIGPLAN「Tested on training set」 |
| S4 | 模擬器不能含「會影響所測效能面向」的簡化假設，也不能用含同樣假設的模擬器評估同一個模型 | Heiser「Benchmarking of simplified simulated system」「Only evaluate against yourself」 |
| S5 | trace 必須是外生的：新策略不能改變被重播的 trace，否則重播有偏差 | CausalSim p1（exogenous trace assumption） |
| S6 | 負載要 open loop | SIGPLAN「Gated workload generator」 |
| S7 | 主張的範圍要與證據相稱：只做模擬就不能說「在真硬體上有效」 | SIGPLAN「Claims not appropriately scoped」 |
| S8 | 模擬是決定性的就明說；變異來自哪裡（trace 時間窗、子樣本）要另外報 | Hoefler Rule 5；Mytkowicz setup randomization（p9） |

**對本研究〔判讀〕**：
- Tiara 的成本常數由真機量測，再拿去算模擬結果，所以 S3 的「校準集」就是量 κ 的那批 run；驗證要用另一批（不同長度、不同 trace）真機 run，比較模擬的 TTFT 與實測。
- S5 是 Mooncake／多輪 trace 最大的風險：重播時保留原始時間戳，等於假設 Tiara 讓回覆變快或變慢都不會改變使用者下一輪何時送出。真實多輪對話是 closed loop（Strata 給 ShareGPT 插 60 s 思考時間，p10，就是一種處理）。要在論文裡明說這個假設，並做「時間戳縮放」的敏感度（Strata 用縮放時間戳改變吞吐，p13 Fig. 12 圖說）。
- Cake 式的延遲注入只模擬「大小 ÷ 頻寬」，不含 I/O 變異、排隊、PCIe 爭用與反序列化的 CPU 成本；使用者文件已把這點列為 Cake 的前提之一（intro.txt §7.2）。若本研究也注入延遲，就必須至少在一個層級（CPU DRAM）用真實搬運對照。

---

## 4. Artifact evaluation

### 4.1 ACM 官方定義（Artifact Review and Badging v1.1，2020-08-24）

讀的是 Wayback 2025-05-26 的快照（acm.org 直連回 403）。〔文件〕〔複核補充〕2026-10-07 再試 acm.org 仍回 403；Wayback 有較新的 2026-07-03 快照（http://web.archive.org/web/20260703121942/https://www.acm.org/publications/policies/artifact-review-and-badging-current），仍標 Version 1.1，從術語定義到 Review Procedures 的文字與 2025-05-26 版逐字相同（diff 無差異）。

- **三個術語**：Repeatability＝同團隊、同設定；Reproducibility＝不同團隊、**用作者的 artifact**；Replicability＝不同團隊、**自己獨立開發的 artifact**。ACM 依 NISO 建議把 reproducibility 與 replication 的用法對調過，並更新了舊 badge。
- **Artifacts Evaluated – Functional**：documented（有清單與足夠說明）、consistent（與論文相關且對主要結果有貢獻）、complete（盡可能包含全部元件；專有部分要說明取得方式並提供替代品）、exercisable（腳本能跑、資料能讀）。
- **Artifacts Evaluated – Reusable**：具備 Functional 的全部條件，而且文件與結構好到便於重用與改作，嚴格遵守該領域的規範。Functional 與 Reusable 只能給一個。
- **Artifacts Available**：放在公開、有永久保存計畫的典藏庫，附 DOI 或連結；**個人網頁不算**；不需要經過評估，也不必完整。
- **Results Reproduced**：他人在後續研究中，部分使用作者的 artifact 得到主要結果。**Results Replicated**：他人不用作者的 artifact 獨立得到主要結果。兩者都不要求完全一致，只要在該類實驗可接受的容差內，且**差異不改變論文的主要主張**。
- 三類 badge 彼此獨立，可以任意組合；ACM 不規定審查流程細節。

### 4.2 Systems 會議（sysartifacts 與 USENIX）

| 會議 | 評哪些 badge | 流程與時程 | 出處 |
|:--|:--|:--|:--|
| SOSP 2026 | 只能申請三種組合：Available＋Functional＋Reproduced（多數軟體）、Available＋Functional（資料集或需特殊環境）、Functional＋Reproduced（無法公開的軟硬體）。申請第一種失敗時可退而給另兩種 | 錄取通知 2026-07-03 → artifact 截止 07-13 → kick-the-tires 到 07-27 → badge 決定 08-25 → 永久典藏 08-27 → camera-ready 08-28；single blind；過程是合作式，作者可在合理時間內修正 | SA sosp2026/badges、/index |
| EuroSys 2027 | 與 SOSP 2026 相同的三種組合與檢查表 | 〔複核補充〕兩個梯次：春季錄取 2026-08-21 → artifact 08-31 → kick-the-tires 09-08 → Functional／Reproduced 決定 09-21 → 典藏 09-23 → camera-ready 09-25；秋季錄取 2027-01-29 → artifact 02-08 → 02-16 → 03-01 → 03-03 → 03-05 | SA eurosys2027/badges、/index（Important Dates） |
| ATC 2026 | 同上（該頁寫 ATC 使用 ACM badge） | 〔複核補充〕錄取 2026-09-18 → artifact 09-22（只有 4 天）→ kick-the-tires 09-29 → 決定 10-14 → camera-ready 10-16 | SA atc2026/badges、/index（Important Dates） |
| OSDI '26 | **今年只評 Artifacts Available**；建議 Zenodo，GitHub、GitLab、機構典藏也可，個人網頁不行；要附可重現全部結果的說明（如 README），但不另外要求功能或正確性 | 通知 2026-03-26 → 截止 05-08 → 結果 06-01 → 定稿 06-09；可加最多 2 頁 Artifact Appendix；README 要有約 30 分鐘內可完成的「Getting Started」與完整重現的「Detailed Instructions」；AE 在錄取後才開始，不影響錄取 | OSDI-CFA |

**SOSP／EuroSys／ATC 的檢查表重點**（SA badges 頁）〔文件〕：
- Available：放在有不可撤銷版本與長期保存的典藏（例：Zenodo，**GitHub 不算**）；授權允許比較與延伸（CC-BY、MIT）；README 指向論文；「之後會公開」不算。
- Functional：README 說明每個元件與論文的對應、作者用的確切環境（OS、硬體）；只放與論文相關的程式與資料，改過的既有程式要與原版分開；**論文中的可量化主張要有簡單腳本輸出**；資料的修改（匿名化、丟棄）要記錄；要有最小可執行範例、每類實驗的預期資源用量（例如「5 分鐘、10 GB」）、可預期的異常訊息；相依套件要精確列出，OS 層相依用 VM／container 並附產生腳本，專有相依要附 mock；每類實驗至少一組範例輸入與設定。
- Reproduced：**每個實驗一個腳本**輸出結果，腳本要有文件以便確認對應哪個主張；處理常見錯誤（忘了參數、重跑兩次）；有把結果轉成接近論文呈現形式的腳本；評審在沒有除錯的情況下，主動操作時間不應超過幾分鐘。

### 4.3 MLSys

- MLSys'26：錄取論文自願參加，**不影響錄取**；依 ACM 政策評 Availability、Functional、Reproducible，並鼓勵盡量拿 Results Reproduced；要交 artifact 摘要（最低軟硬體需求、怎麼驗證、預期結果）、論文 PDF 與 Artifact Appendix；截止 2026-03-08，結果 04-16；通過任一 badge 要把 badge 與附錄加進 camera-ready；設 Distinguished Artifact Award。〔文件〕MLSys-AE
- 該頁連到 cTuning 的評審指南與附錄模板（ae-20190108.tex）；指南寫明經驗與數值結果允許一定變異，並指向 SIGPLAN 的評測指南（cTuning reviewing.md「Results reproduced」段）。〔文件〕
- EuroMLSys（本研究的備案場所）是否有 AE：〔未查證〕。

### 4.4 從現在開始怎麼準備〔判讀〕

依據上列官方要求，對照本專案已有的做法（CLAUDE.md §2、§4）：

1. **每個實驗一支腳本，再加一支把結果轉成論文圖表的腳本**（SOSP Reproduced 檢查表）〔複核修正：原寫「每個圖表一支腳本」；檢查表原文是每個實驗一支執行腳本，另有轉換成接近論文呈現形式的腳本，簡單的表格可合併〕：`code/` 裡每個 Fig／Table 對應一個入口，輸出 `results/` 的 CSV，再由畫圖腳本產生圖。現有 `run_id`／`ts` 欄位正好是 AE 要的「結果能追溯」。
2. **分兩層 artifact**：模擬層（只吃 CSV 與 trace，CPU 就能跑，評審幾分鐘可重現）與真機層（需要 GPU）。前者可拿 Reproduced，後者在評審沒有 3090／MI300X 時退成 Functional（SOSP 允許的組合）。
3. **README 兩段式**：30 分鐘內的 Getting Started（OSDI'26 要求），加完整重現說明；寫明每個實驗的時間與磁碟用量。
4. **環境固定**：vLLM 版本、CUDA／ROCm、驅動、模型權重 hash；用 container 或鎖定的 requirements；專案已有 `env_fingerprint.py`，延伸成 artifact 的一部分。
5. **trace 授權**：Mooncake trace 是 Apache-2.0 repo 的一部分（workloads_eval.md 已查；〔複核補充〕GitHub API 2026-10-07：kvcache-ai/Mooncake 的授權檔 LICENSE-APACHE 為 Apache-2.0，`FAST25-release/traces/` 下有 conversation／synthetic／toolagent 三個 trace）；artifact 裡附下載腳本而非直接打包，並附校驗碼。
6. **典藏**：最後上傳 Zenodo 拿 DOI（SOSP／EuroSys／ATC 的檢查表明說 GitHub 不算 Available）。〔複核修正：原寫「GitHub 不算 Available」未分會議；OSDI'26 CFA 把 GitHub、GitLab 列為有效的典藏選項（只排除個人網頁），見 §4.2 OSDI 列〕
7. **結果容差**：在 artifact 裡寫明「時間類數字在不同 GPU 上會差多少、哪些主張不受影響」（ACM：差異不能改變主要主張）。
8. **時程**：SOSP 從錄取到交 artifact 只有 10 天（〔複核補充〕EuroSys'27 也是 10 天，ATC'26 只有 4 天，見 §4.2）；投稿時就要讓 artifact 可以跑。

---

## 5. 審稿人常問的評測問題

### 5.1 OpenReview 審稿意見：本次未能讀取

目標論文的 forum（本卡只確認了 ID，沒有讀到內容）：
- Cake（ICML'25）：https://openreview.net/forum?id=WOyOtaO6lQ 〔二手：搜尋結果〕
- SCBench（ICLR'25）：https://openreview.net/forum?id=gkUyYcY1W9 〔二手：搜尋結果〕
- DuoAttention（ICLR'25）、SnapKV（NeurIPS'24）、LookaheadKV、TRIM-KV、KVP：forum ID 〔未查證〕

所有 OpenReview 頁面都先跳到「Verifying your browser」驗證頁；依安全規則不代為通過。**本節沒有任何一條來自 OpenReview 審稿意見。**

〔複核補充〕複核者 2026-10-07 再試公開 API：`https://api2.openreview.net/notes?forum=WOyOtaO6lQ` 與 `https://api.openreview.net/notes?forum=WOyOtaO6lQ` 都回 HTTP 403，JSON 內容為 `ChallengeRequiredError`（「Challenge verification required」，附 openreview.net/challenge 的轉址）。API 同樣要求人機驗證，依規則停止，未嘗試繞過；審稿意見維持〔未查證〕。

**觀察（與審稿無關，但有出處）**：Cake 的 PMLR 定稿比 arXiv v2 多了 §5.3「Evaluation Across Varying Chunk Sizes」與 §6 Discussion（與其他加速法的相容性、多節點部署）（比對兩版 PDF：PMLR p6–p7、p9；arXiv v2 只到 §5.8）。這些增補**可能**是回應審稿意見，但沒讀到審稿內容，無法確認〔判讀〕。

### 5.2 可讀到的公開評測批評（非審稿意見）

以下 15 條都有出處，但來源是**批判論文或方法學文件**，不是審稿意見。

| # | 批評（改寫） | 出處 | 類別 |
|:--|:--|:--|:--|
| R1 | 壓縮研究很少量吞吐；有量的多在 HF Transformers（TRL）上，沒結合 FlashAttention、PagedAttention；也沒量多 GPU（tensor parallel） | Rethinking'25 Missing Piece 1，p5 | 對手或引擎不是生產級 |
| R2 | 吞吐評測常固定回應長度 | Rethinking'25 p5（Missing Piece 1 前後文） | 負載不真實 |
| R3 | 壓縮改變回應長度，進而改變端到端延遲，卻普遍被忽略 | Rethinking'25 Missing Piece 2，p5 | 指標不完整 |
| R4 | 只報整體分數，沒分析個別樣本的品質（負樣本） | Rethinking'25 Missing Piece 3，p6 | 只報平均 |
| R5 | 在某些 batch size、序列長度、TP 設定下，壓縮反而使效率變差 | Rethinking'25 Observation 2，p8 | 只測有利區段 |
| R6 | 用 TRL 量的效率結果不可靠，應在成熟 serving 框架上量 | Rethinking'25 Observation 1，p7 | 對手或引擎不是生產級 |
| H1 | 沒評估可能的退化（只證明變好、沒證明別處沒變差） | Heiser | 沒報開銷或退化 |
| H2 | 只給相對數字 | Heiser | 呈現 |
| H3 | 不公平地測對手（例：用了開除錯選項的預設設定） | Heiser | 對手設定 |
| H4 | 不跟最新方法比以膨脹改善 | Heiser | 對手不夠強 |
| H5 | 沒有顯著性（至少給標準差） | Heiser | 統計 |
| H6 | 校準與驗證用同一份資料 | Heiser | 模擬可信度 |
| P1 | 閉迴路負載產生器會誤導 | SIGPLAN checklist | 負載不真實 |
| P2 | 「最多 49%」式的摘要 | SIGPLAN checklist | 呈現 |
| P3 | 只做模擬卻宣稱在真實硬體上可行 | SIGPLAN checklist | 模擬可信度 |

補充背景數字：Hoefler 調查 95 篇只有 2 篇給平均值 CI（p4）；Kalibera 調查 90 篇有 71 篇沒有任何變異量（p2）；Mytkowicz 調查 88 篇沒有一篇妥善處理量測偏差（p8）。〔原文〕

### 5.3 歸納：審稿人可能問的評測問題〔判讀〕

以下是依 5.2 與 §1–§4 的歸納，**不是**引自審稿意見。

| 類別 | 可能的問題 | 依據 |
|:--|:--|:--|
| 對手不夠強 | 有沒有跟 Cake、Strata、LMCache 的最新版比？對手參數是否照官方建議？ | H3、H4；Strata 列出所有對手版本與參數（p9） |
| 引擎不是生產級 | 是否在 vLLM／SGLang 上實作與量測，而非 HF Transformers？ | R1、R6 |
| 長度太短 | 跨請求重用的論文最長多少？使用者文件已整理：Cake 4K–16K、CachedAttention 32K 等 | intro.txt L548–551（使用者文件，圖 6 說明）；Cake p5 |
| 只量單請求 | 有沒有多請求、真實到達、吞吐–延遲曲線？ | §1.2：Cake 是唯一沒有掃負載的 |
| 沒報開銷與退化 | 預測器的 CPU 時間、記憶體？短 context 是否變差？ | H1；Strata §5.2.3；vLLM §7.1 |
| 沒有統計 | 重複幾次？CI？差異是否大於雜訊？ | H5；Hoefler Rule 5、7；Kalibera |
| 模擬不可信 | 模擬器誤差多少？校準與驗證是否分開？trace 是否外生？ | §3.3 S2–S5 |
| 負載不真實 | 到達是 Poisson 還是真實時間戳？是否 open loop？回應長度固定嗎？ | P1、R2；MLPerf Server 情境 |
| 只測有利區段 | 頻寬、長度、容量是否掃到不利的區段？ | R5；Heiser「Selective data set」 |
| 平均方式 | 加速比怎麼平均？有沒有排除點？ | Hoefler Rule 3–4；Cake 排除紅色點（p6） |
| 沒有上界 | 離 Oracle 還差多少？ | Hoefler Rule 11；Mooncake §5.3.1、Strata §5.4 |
| 品質 | 有損動作（量化、丟棄）對品質的影響？ | CachedAttention §4.3.5；R3、R4 |

---

## 給使用者的評測自我檢查表〔判讀〕

| # | 檢查項 | 依據 |
|:--|:--|:--|
| 1 | 評測章開頭列出要回答的 3–5 個問題，每個問題對應一節 | §1.1（Sarathi、Mooncake、Cake、Strata） |
| 2 | 至少有：端到端掃負載、消融、敏感度、開銷、不退化 | §1.2 |
| 3 | 有一個上界（Oracle）或理想值，說明還剩多少空間 | §1.2；Hoefler Rule 11 |
| 4 | 對手是最新方法，版本與所有參數都寫出來 | §2.2；Strata p9 |
| 5 | 在 vLLM／SGLang 等生產級引擎上量，不在 HF Transformers 上量 | §5.2 R1、R6 |
| 6 | 每張加速比圖附基準的絕對值 | Hoefler Rule 1；Heiser |
| 7 | 跨設定的平均從時間或速率算回來，不直接平均比率 | Hoefler Rule 3–4 |
| 8 | 摘要數字反映全部分布，不寫「最多 x 倍」而不給分布 | SIGPLAN |
| 9 | 有排除的點要寫規則與理由，並附上未排除的結果 | Hoefler Rule 2；§1.1 Cake |
| 10 | 真機數字附 CI；至少 6 次獨立 run 才能用無母數法估中位數的 95% CI | Hoefler p8；LeBoudec |
| 11 | 重複是在「重啟 server」層級，不只是同一次啟動裡多送請求 | Kalibera |
| 12 | 只有幾百個請求時不報 P99，改報 P90 或附 CI | §2.6 G14–G16 |
| 13 | 計時用同步過的 Event 或 client 端牆鐘，不混用 | §2.6 G1–G2 |
| 14 | 每次啟動先暖機，暖機請求不計分 | §2.6 G3；Heiser |
| 15 | 記錄 CUDA graph 是否啟用，比較的各組一致 | §2.6 G5 |
| 16 | 每個 run 記錄時脈與節流計數器的前後差值 | §2.6 G7–G8 |
| 17 | 記憶體同時記 nvidia-smi 與 PyTorch 計數器 | §2.6 G6 |
| 18 | 以「有無外來 PID」判斷污染，不以使用率判斷 | §2.6 G12 |
| 19 | 負載是 open loop；用 Poisson 時說明理由，用真實時間戳時說明是否縮放 | §2.6 G17；SIGPLAN |
| 20 | 掃描範圍包含對自己不利的區段（短 context、高頻寬） | Heiser；R5 |
| 21 | 量測順序要反轉或隨機化一次，檢查順序效應 | Heiser 最佳實務；Mytkowicz |
| 22 | 模擬器用另一批真機 run 驗證，報同一指標的誤差 | §3.3 S2–S3 |
| 23 | 寫明 trace 外生性假設，並做時間戳縮放的敏感度 | §3.3 S5 |
| 24 | 寫明模擬是決定性的，另報 trace 時間窗或子樣本的變異 | Hoefler Rule 5；Mytkowicz |
| 25 | 主張範圍與證據相稱：模擬得到的結論不寫成真機結論 | SIGPLAN；§3.3 S7 |
| 26 | 有損動作同時報品質，並看個別樣本的失敗 | CachedAttention §4.3.5；R3、R4 |
| 27 | 報預測器與排程本身的開銷（CPU 時間、記憶體） | Heiser；vLLM §7.1 |
| 28 | 每個實驗一支腳本＋一支轉成論文圖表的腳本，結果能追溯到 run_id〔複核修正：原寫「每個圖表一支腳本」〕 | §4.2 SOSP 檢查表 |
| 29 | 模擬層 artifact 不需要 GPU，評審幾分鐘可重現 | §4.4 |
| 30 | 最後上傳 Zenodo 拿 DOI（〔複核補充〕SOSP／EuroSys／ATC 不接受 GitHub；OSDI'26 接受 GitHub／GitLab；Zenodo 在 SA 檢查表與 OSDI-CFA 都被列為範例） | §4.1–4.2 |

---

## 與既有整理不一致

| 檔案與位置 | 既有寫法 | 原文 | 判定 |
|:--|:--|:--|:--|
| intro.txt「評測協定」表（L830） | 重複「每點至少 3 次，附信賴區間」 | Hoefler：無母數 CI 需要 n > 5（p8）；Le Boudec：中位數 95% CI 在 n ≤ 5 時不存在（附錄 A 表 A.1）；Kalibera：最高層重複少於 5 次就估不準變異（p11） | **不一致**：3 次不足以給無母數 CI；若假設常態可用 t 分布，但 Hoefler Rule 6 要求先診斷常態。〔複核〕✅ 指控成立：n=3 時雙側次序統計量 CI 對中位數最大覆蓋率＝1 − 2·0.5³＝0.75，到不了 0.95（算式見 §2.6 G15 下的複核補充）；Kalibera p11 也說最高層少於 5 次「難以估對變異」。但使用者文件沒寫要用哪種 CI，用 t 區間（n=3、自由度 2）在數學上算得出來，只是很寬且需常態假設，所以精確說法是「3 次給不出無母數 CI，參數 CI 也站不住」 |
| workloads_eval.md L513 | 真機層 A/A 至少 5 次 | 同上（Hoefler「n > 5」） | 接近但差一點：建議改成至少 6 次 |
| workloads_eval.md L431 | 模擬層 A/A（換 seed）無意義，因 trace 重播是決定性的 | Hoefler Rule 5 要求明說決定性；Mytkowicz 的 setup randomization（p9）說明變異也來自設定 | 部分同意：換 seed 確實沒意義，但 trace 時間窗或子樣本的變異仍應報〔判讀〕 |
| intro.txt L551 | 「Mooncake 的 128K 是模擬資料 [11]」，[11] 引的是 FAST'25 版 | FAST'25 §5.2.1：從線上叢集取樣的 1 小時 conversation trace 含長達 128k、平均約 12k token 的請求（p10；Table 2 平均輸入 12,035） | **不一致**：FAST'25 版的 128K 出現在真實 trace，不是模擬資料。〔複核〕✅ 指控成立（對所引的 FAST'25 版）：FAST'25 p10 §5.2.1 寫 conversation trace 是從線上叢集取樣 1 小時、請求最長到 128k、平均約 12k；Table 2 平均輸入 12,035、依時間戳重播。〔複核補充〕「模擬 16K–128K」的來源是 arXiv 版：arXiv 2407.00079v4 PDF p15 Table 2 有「Simulated Data：16k、32k、64k、128k，輸出 512，cache ratio 50%，Poisson」。所以使用者文件是把 arXiv 版的內容掛在 FAST'25 版的引用 [11] 上；同樣問題也出現在 intro.txt L853（「Mooncake [11] 模擬 16K–128K、快取比例 50% 的請求」）。FAST'25 版另有 Fig. 14（p13）在 0% 與 95% 前綴快取下比較 16k–128k 輸入，是受控實驗，不是 trace |
| workloads_eval.md L58（Mooncake 列） | dummy「LLaMA2-70B 同架構」；指標 P90 TTFT／TBT；頁碼 p7–p8、p15–p17 | FAST'25 版：dummy LLaMA3-70B（p9）；TBT＝token 間隔最長 10% 的平均，TTFT 門檻 30 s（p9） | **可能是版本差異**：既有檔的頁碼像 arXiv 版。本卡只讀 FAST'25 版，arXiv 版未核對。〔複核補充〕已核對 arXiv 2407.00079v4（https://arxiv.org/pdf/2407.00079，2025-09-03 版）：p3 與 p15 寫 dummy model 與 LLaMA2-70B 同架構；p15 Table 2 有模擬資料 16k–128k、cache 50%；p15 用 P90 TTFT／TBT 當最終指標；testbed 寫 RDMA 最高 800 Gbps。L58 的內容與 arXiv v4 一致，**確認是版本混用**：列名標 FAST'25，內容卻是 arXiv v4 |
| 〔複核補充〕intro.txt「評測協定」表（L826） | 指標寫「TTFT p50／p99」 | LeBoudec Thm 2.1＋本卡 G15 的計算：P99 的 95% 無母數 CI 至少要 299 個請求，且上界就是最大值 | 不是錯誤，但每個點要有 ≥ 299 個請求才給得出 P99 的 CI；請求數不夠時改報 P90 或附註 |
| intro.txt L611 | 「重點幾篇都沒有報重複次數或信賴區間」 | 本卡對七篇做關鍵字搜尋，結果一致 | ✅ 一致 |
| intro.txt L37、L564 | Cake 用「大小 ÷ 頻寬」模擬 I/O | Cake p5：依 chunk 大小與頻寬算延遲，延遲到期前不把資料交給 GPU | ✅ 一致 |

---

## 未查證清單

| 項目 | 原因 | 下一步 |
|:--|:--|:--|
| 所有 OpenReview 審稿意見（Cake、SCBench、DuoAttention、SnapKV、LookaheadKV、TRIM-KV、KVP 等） | openreview.net 與 API 都要求人機驗證；Wayback 只存論文本體；HF 鏡像資料集的 datasets-server 一直回「index is loading」。〔複核補充〕2026-10-07 複核者再試 api2／api 兩個公開 API，皆回 403 ChallengeRequiredError，已停止 | 使用者用自己的瀏覽器登入 OpenReview 後手動開啟；或改由下載 HF 上的 OpenReview 鏡像資料集（需使用者同意下載） |
| DuoAttention、SnapKV、LookaheadKV、TRIM-KV、KVP 的 forum ID | 沒能查到 | 同上 |
| Cake PMLR 版新增的 §5.3、§6 是否為回應審稿 | 讀不到審稿 | 同上 |
| ACM badge 頁的最新版 | acm.org 直連回 403，讀的是 2025-05-26 的 Wayback 快照。〔複核補充〕2026-07-03 的 Wayback 快照仍是 Version 1.1，相關段落逐字相同；live 頁仍 403 | 用瀏覽器直接開 |
| Splitwise（ISCA'24）、LLMServingSim（IISWC'24）、CausalSim（NSDI'23）的會議版 | 只讀 arXiv 版；venue 來自搜尋結果。〔複核補充〕複核者查 dblp API 沒有取得回應，venue 仍是〔二手〕 | 讀會議版確認驗證段落是否相同 |
| Vidur 的 MLSys 會議版是否與 arXiv v2 相同 | arXiv v2 內文標示 MLSys，但未逐頁比對 | — |
| Mooncake arXiv 版與 FAST'25 版的差異 | 只讀 FAST'25 版。〔複核補充〕複核者已讀 arXiv 2407.00079v4 的 §8 評測段（p15）：dummy LLaMA2-70B、模擬 16k–128k、P90 指標都只在 arXiv 版，見「與既有整理不一致」表；其餘章節未逐頁比對 | 若要引用模擬資料或 P90 指標，引 arXiv v4，不要引 FAST'25 |
| EuroMLSys 是否有 AE | 沒查 | 查 EuroMLSys 2027 CFP |
| MLSys 2027 的 AE 要求 | 2027 年的頁面未查 | 公告後再查 |
| vLLM（本機 0.28.0）預設的 CUDA graph 捕捉大小與 eager 開關 | 不在本卡範圍；E02 可能有 | 讀 vLLM 文件或原始碼 |
| Jain《The Art of Computer Systems Performance Analysis》中模擬驗證的章節 | 沒有可讀的公開版本 | 圖書館 |
| GPU 時間切片（無 MPS 時多行程如何共用 SM）的官方說明 | MPS 文件只說 MPS 可減少 context switching | 讀 CUDA Programming Guide 或 NVIDIA GPU sharing 文件 |

---

**狀態**：抽取完成（E11，2026-10-07）；§5.1 的 OpenReview 審稿意見未取得（複核者再試公開 API 也被人機驗證擋下）；已完成獨立複核（V11，2026-10-07），見下方「複核紀錄」。

---

## 複核紀錄

- **複核者**：V11（未參與抽取；沒有讀抽取者的推理或筆記，只讀卡片與原始來源）
- **日期**：2026-10-07
- **用的來源**：抽取者下載的原檔（`scratchpad/E11/` 的 PDF、HTML、adoc、md），先確認版本：vLLM＝arXiv 2309.06180v1（PDF 內有 SOSP '23 版權列）；Sarathi、DistServe＝OSDI '24 USENIX 版（頁尾 18th OSDI）；CachedAttention＝ATC '24 USENIX 版；Mooncake＝FAST '25 USENIX 版（頁尾 23rd FAST）；Strata＝OSDI '26 USENIX 版（頁尾 20th OSDI）；Cake＝PMLR 267 版＋arXiv 2410.03065v2（2025-02-20）；Vidur＝arXiv 2405.05465v2（內標 7th MLSys）；Rethinking＝arXiv 2503.24000v1（內標 8th MLSys）；Splitwise＝arXiv 2311.18677v2；LLMServingSim＝arXiv 2408.05499v1；CausalSim＝arXiv 2201.01811v4；Hoefler＝SC '15 作者版；Kalibera＝KAR「Updated Version」；Mytkowicz＝ASPLOS '09 PDF；SIGPLAN checklist＝2018-10 PDF；LeBoudec＝Version 2.3.2。PDF 頁碼用自己寫的分頁工具（`scratchpad/V11/pg.sh`）逐頁確認，必要時用 pdftoppm 看圖（Strata p13、Mooncake p12）。另外自己下載：Mooncake arXiv 2407.00079v4（`scratchpad/V11/mooncake_arxiv.pdf`）、ACM badge 頁 2026-07-03 Wayback 快照；用 GitHub API 查 MLPerf inference_policies HEAD＝d3eba2f（2026-08-20）、cTuning artifact-evaluation HEAD＝dfb0228（2026-07-24），與卡片一致。
- **檢查了幾項**：約 381 項。來源清單 30 列（版本與身分）、重點摘要 9 條、§1.1 七篇評測章 71 列（小節標題、頁碼、圖表、設定）、§1.2 組件表 77 格＋3 條判讀、§2.1 Hoefler 12 條規則＋6 條背景／其他重點、§2.2 Heiser 19 列＋2 段、§2.3 Mytkowicz 5 條、§2.4 Kalibera 5 條、§2.5 SIGPLAN 9 條、§2.6 GPU 陷阱 17 條、§3.1 6 列、§3.2 6 列、§3.3 12 條、§4.1 6 條、§4.2 7 項、§4.3 3 條、§4.4 8 條、§5.1 3 條、§5.2 16 條、§5.3 12 列、自我檢查表 30 條、與既有整理不一致 7 列。
- **結果**：✅ 370、❌ 4、⚠️ 5（另有 2 項維持〔未查證〕：OpenReview 審稿意見、Splitwise／LLMServingSim／CausalSim 的會議 venue）。七篇的小節標題與頁碼、組件表的有／無、Hoefler 12 條、Heiser 各罪狀、Mytkowicz、Kalibera、SIGPLAN、GPU 陷阱 17 條的出處、模擬器誤差數字（DistServe < 2%、Vidur < 9%／3.33%／< 5%／12.65%、Splitwise MAPE < 3%、LLMServingSim 14.7%／8.88%）、ACM badge 定義、OSDI'26 只評 Available、SOSP'26 錄取後 10 天交 artifact、MLSys'26 三個 badge，都與原文一致。
- **〔計算〕是否成立**：成立。P95 至少 59、P99 至少 299、中位數至少 6（γ＝0.95，雙側、i.i.d.、連續分布），算式與逐點數值寫在 §2.6 G15 下方的複核補充列；中位數與 P95 的值和 LeBoudec 表 A.1、A.3 一致。
- **對使用者文件的指控**：
  - intro.txt L830「每點至少 3 次，附信賴區間」不足以算無母數 CI：✅ 指控成立（n=3 時中位數雙側 CI 最大覆蓋率 0.75）；補充：t 區間算得出來，但需常態假設，3 個點無法診斷。
  - intro.txt L551「Mooncake 的 128K 是模擬資料 [11]」與 FAST'25 p10 不符：✅ 指控成立；補充：這句的內容來自 arXiv v4（p15 Table 2），不是 FAST'25；L853 同樣問題。
  - intro.txt L611、L37／L564 兩列「✅ 一致」：複核同意。
  - workloads_eval.md L58「可能是版本差異」：已確認是版本混用（內容＝arXiv v4）。

### 逐條修改（原內容 → 新內容＋出處）

| # | 判定 | 位置 | 原內容 | 新內容 | 出處 |
|:--|:--|:--|:--|:--|:--|
| 1 | ❌ | §3.3 S1 依據 | 「七篇中有用模擬的五篇都是這樣」 | 用模擬的是四篇（DistServe、Mooncake、Strata、Cake）；前三篇符合「模擬只做消融或 what-if」，Cake 例外，主結果的 I/O 全部以延遲模擬 | 本卡 §1.2「模擬的使用」列；Cake PMLR p5「I/O Bandwidth Control」段 |
| 2 | ❌ | 重點摘要 1 | 「七篇…先一段『這節要回答哪些問題』」標〔原文〕 | 改標〔判讀〕；只列實際有寫的四篇＋DistServe 的摘要式章首，並註明 vLLM 只有一句、CachedAttention 直接進 setup | vLLM p9 §6 首句；CA p9 §4→4.1；DistServe p10 |
| 3 | ❌ | §4.4 第 6 點 | 「（GitHub 不算 Available）」 | 限定為 SOSP／EuroSys／ATC；註明 OSDI'26 CFA 把 GitHub、GitLab 列為有效選項 | OSDI-CFA「Process」段 Artifacts Available；SA sosp2026/badges「Available」檢查表 |
| 4 | ❌ | §1.1 Mooncake 第一列；§1.2 消融格 | Fig. 5 在 p9 | 文字 p9，Fig. 5 圖與圖說在 PDF p8 | Mooncake FAST'25 PDF p8 |
| 5 | ⚠️ | §2.1 其他重點 | 「至少要 5 個以上的量測」 | 「超過 5 個（n > 5，即至少 6 個）」 | Hoefler p8 |
| 6 | ⚠️ | 重點摘要 3 | 「Hoefler 抽樣 95 篇」 | 「抽樣 120 篇、其中 95 篇適用」 | Hoefler p2 |
| 7 | ⚠️ | §3.3 前言 | OSDI'26 CFP「可行性與效益」 | 「實用性與效益（practicality and benefits）」 | OSDI-CFP「Submitting a Paper」 |
| 8 | ⚠️ | §4.4 第 1 點；自我檢查表 28 | 「每個圖表一支腳本」 | 「每個實驗一支腳本＋一支轉成論文圖表的腳本」 | SA sosp2026/badges「Reproduced」檢查表 |
| 9 | ⚠️ | 重點摘要 6 | 「真機做端到端主結果，模擬只拿來做…消融或 what-if」未註例外 | 加註 Cake 例外 | Cake PMLR p5 |
| 10 | 補充 | §2.6 G15 下方 | — | 新增〔計算〕列：假設、精確算式、n=5/6、58/59、298/299 的覆蓋率 | LeBoudec Thm 2.1（書 p33）、表 A.1／A.3（書 p315、317）；Python 精確二項計算 |
| 11 | 補充 | 與既有整理不一致 | L551 列只寫「不一致」；L58 列「可能是版本差異」 | 標 ✅ 指控成立；補 arXiv v4 p15 Table 2 為來源、intro.txt L853 同問題；L58 確認為版本混用 | FAST'25 p10；arXiv 2407.00079v4 p3、p15 |
| 12 | 補充 | 與既有整理不一致 | — | 新增 intro.txt L826「TTFT p50／p99」列：每點需 ≥ 299 個請求才給得出 P99 的 CI | 本卡 G15 |
| 13 | 補充 | 重點摘要 8；§4.2；§4.4 第 8 點 | EuroSys／ATC 時程「—」 | EuroSys'27 兩梯次各 10 天、ATC'26 4 天 | SA eurosys2027、atc2026 首頁 Important Dates |
| 14 | 補充 | §4.1；未查證清單 | 只讀 2025-05-26 快照 | 2026-07-03 快照仍為 v1.1，相關段落逐字相同 | Wayback 20260703121942 |
| 15 | 補充 | §4.4 第 5 點 | Mooncake 授權只引 workloads_eval | GitHub API：LICENSE-APACHE＝Apache-2.0，`FAST25-release/traces/` 有三個 trace | api.github.com/repos/kvcache-ai/Mooncake（2026-10-07） |
| 16 | 補充 | §5.1；未查證清單 | 只寫網頁被擋 | 公開 API（api2、api）也回 403 ChallengeRequiredError；已停止，未繞過 | 2026-10-07 實測 |
| 17 | 補充 | 來源清單 LeBoudec 列 | — | 版本 2.3.2（2026-07-26）與書頁／PDF 頁對照 | LeBoudec PDF 封面 |

### 因時間或條件沒有檢查的

- Splitwise（ISCA'24）、LLMServingSim（IISWC'24）、CausalSim（NSDI'23）的會議 venue：dblp API 沒有回應，未另找會議版，維持〔二手〕。
- §1.1 各圖的「形狀」描述（折線、長條、堆疊等）：只對 Strata p13、Mooncake p12 看過圖，其他依圖說與軸標文字判斷，沒有逐張開圖。
- 使用者文件 intro.txt L548–551 中 Cake 以外各論文的長度（HCache、LMCache、Strata 55K、CacheFlow）：不在本卡範圍，未核對。
- sysartifacts 與 OSDI-CFA、MLSys-AE 頁面：用的是抽取者 2026-10-06 下載的副本（已確認是正確頁面），沒有重新抓 live 版。
- Vidur MLSys 會議版、Mooncake arXiv v4 §8 以外的章節：未逐頁比對。
