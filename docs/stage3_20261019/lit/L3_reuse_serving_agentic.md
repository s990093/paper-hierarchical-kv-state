# L3：復用、多請求服務與 agent 工作負載的文獻地圖（第三階段選題用）

**日期**：2026-10-19（UTC 13:20 開始整理）
**範圍**：E（prefix caching／shared KV／復用）、F（paging／配置／記憶體壓力）、G（多請求吞吐／goodput／尾延遲／准入／路由）、H（agent 多輪、重複前綴、KV 存活時間）
**作者**：文獻調查 agent（self-review，**不是 cross-model review**；CLAUDE.md §5 的已知落差）
**沒有做的事**：沒有跑任何 GPU 或耗 CPU 的工作（機器正在計時量測）；沒有安裝套件；沒有下載大型資料；沒有改 repo 其他檔案。唯一的本機運算是用 `venv/tiara` 的 PyMuPDF 把 13 篇 PDF 抽成純文字（放在 scratchpad，不進 repo），和 `grep` 本機 vLLM 0.19.1 原始碼。

## 讀法與標註

- **閱讀深度**：〔全文〕讀過全文；〔部分〕讀過方法與評測段落（PDF 抽字後逐段讀，或官方 HTML）；〔摘要〕只讀 arXiv 摘要頁或官方頁面摘要。〔摘要〕級的數字**引用前要回原文**。
- **〔原文〕**＝原文數字（都是**各論文在自己的硬體、模型、負載**下量的；**不同硬體／batch／QPS 的數字不准互相排名**，本文也沒有排名）。**〔原始碼〕**＝我讀了本機 `/mlsteam/workspace/src/vllm`（tag `v0.19.1`，commit `b1388b1f`）。**〔repo〕**＝本 repo 既有文件或實測。**〔判讀〕**＝我的推論。
- arXiv ID 都用 WebFetch 打開 `arxiv.org/abs/<id>` 確認過存在（§8 每筆標「已開」）。查不到的寫 **NOT_FOUND**；沒量的寫 **NOT_MEASURED**。
- **已在 repo 的文獻不重做**，只補新的與缺的。已有卡片的：Cake、CacheFlow、Strata、Pensieve、Bidaw、MTDS、AdaptCache／EvicPress、LMCache、Mooncake、py-kvcache（部分）、AsymCache、When Fancy Eviction Fails、HCache、KVPR（`docs/SOTA_ANALYSIS_20261006.md` §2）；LPC、Alibaba「KVCache Cache in the Wild」、CachedAttention、Marconi、KVFlow、Continuum、SGLang HiCache、Dynamo KVBM（`docs/research_20261010_directions/D4_workload.md` §5）；EfficientAgent、Lachesis、MORI、WhereKVLive、Jenga、MPIC、HyMCache、SparsePrefixCaching、SuffixReplay（`docs/research_20261009_explore/cards/`）；Continuum、InferCept、Llumnix、Sarathi-Serve、DistServe、TraceLab、llm-d P2P、BuyKV、Strata 排程、NEO、Five-Minute Rule（`docs/research_20261010_directions/cards/`）；公開 trace 盤點（`docs/research_20260924/workloads_eval.md` §3）；vLLM／LMCache／SGLang／Dynamo 的卸載阻塞行為（`docs/research_20261009_explore/H0_offload_blocking.md`）。這些在本文只用一行指過去，或補它們沒寫到的細節。

---

## §0 先講結論

1. **「復用不划算」的區域是存在而且有人量過的，但幾乎都是「單請求」量的。** py-kvcache 在 H100 上量到：8K prompt 要 77.8% 的前綴命中才打平重算，80K 只要 7.8%；Qwen3-4B 的 SSD 打平點是 6,203 token〔原文 2609.11744 p.11、p.20〕。llm-d 量到 P2P 抓取與重算約在 8K 打平〔repo 卡片〕。**有併發、有搶頻寬、有「同一塊正在被載入」時的邊界，沒有人系統性量過**；py-kvcache 自己把「加上預期排隊時間」列為下一步〔原文 p.21〕。
2. **我們用的 vLLM 0.19.1 本身就有一個「命中但從慢層搬反而慢」的機制**：`OffloadingConnector` 的排程端在「命中的 block 正在被別的請求載入」時把請求**延後**〔原始碼 `offloading/scheduler.py:126-140`〕。上游 issue #44294 在 L40S、0.19.1、50 個併發共用 10K 前綴時量到開卸載 TTFT 11.46 s，不開 5.95 s（0.17.1 開卸載是 0.90 s）；issue 2026-10-07 以 not planned 自動關閉〔原文 issue〕。**在 MI300X 上沒有人量過**（NOT_MEASURED）。
3. **agent 工具呼叫暫停時「KV 該留 GPU／卸 CPU／卸 SSD／丟掉重算」**：已經有一串系統在做二選一或三選一（InferCept：丟／留／換到 CPU；Continuum：留多久；Tokencake、MORI：何時卸到 CPU；Lachesis：HBM／HBF；Cascade：依延遲預算決定從深層還原、留著、或重算）。**沒有找到**「用實測、依硬體而變的重算÷搬運成本比（κ），在 GPU／CPU／SSD／丟棄四個動作之間選，並跨硬體比較決策會不會翻轉」的論文〔查不到不等於沒有；Cascade 只讀了摘要，是最大的未知數〕。
4. **以 goodput／P95 為目標的 KV 放置**：Mooncake 明講「等慢層的 KV 可能違反 TTFT SLO」並做早期拒絕〔原文〕；Cascade（2026-08）宣稱用延遲預算決定還原／保留／重算，goodput 最多 2.4 倍〔摘要〕。兩者都沒有公開「換目標函數（平均 vs P95／goodput）會不會讓放置決定翻轉」的分析〔判讀〕。
5. **多請求的容量公平性**：公平性文獻（VTC、DLPM、FairInference）量的是**計算服務量或 token 延遲**，不是「誰佔了 HBM、誰被趕到 CPU、誰被丟掉」。`cache_salt`／workspace 隔離解決的是**安全**（時序側通道），不是**配額**〔判讀〕。

**最重要的 3 個缺口**、**3 個最小實驗**見 §5、§7；**vLLM 0.19.1（ROCm）上可直接用的功能**見 §7.0。

---

## §1 E：Prefix caching、共享 KV、跨請求復用

### 1.0 總表

| # | 論文 | 年份／venue | 讀到 | 一句話 | 和 Tiara 的關係 |
|:--|:--|:--|:--|:--|:--|
| E1 | vLLM PagedAttention＋APC＋`cache_salt` | SOSP'23；APC 為系統功能 | 〔部分〕＋〔原始碼〕 | 以 block hash 鏈做前綴命中；LRU 從請求尾端先丟 | 我們的引擎本身 |
| E2 | SGLang／RadixAttention | NeurIPS'24 | 〔摘要〕 | radix tree 前綴快取＋最長前綴優先排程 | 對照系統；MI300X 上 NOT_MEASURED |
| E3 | Preble | ICLR'25 | 〔部分〕 | 分散式排程：共享 token > 非共享 token 才去命中那台 | **最早的「復用 vs 負載」門檻規則** |
| E4 | Marconi | MLSys'25 | 〔部分〕（repo 已有卡） | hybrid 模型：准入／淘汰加「每 byte 省多少 FLOP」 | 純 attention 模型上這項近乎常數 |
| E5 | CacheBlend | EuroSys'25（best paper） | 〔部分〕 | 非前綴 chunk 復用＋約 15% 選擇性重算；**依 T_load vs T_recompute 選儲存裝置** | **單請求版的「κ 決定放哪」** |
| E6 | EPIC | ICML'25 | 〔摘要〕 | 位置無關快取（PIC），LegoLink 只重算每段開頭 | 非前綴命中 |
| E7 | KVLink／APE／Block-Attention／TurboRAG／MEPIC／MiniPIC／KV-Repurpose／BCR | 2025–2026 | 〔摘要〕 | PIC 家族與 2026 新作 | 品質代價、可共享性 |
| E8 | py-kvcache | arXiv 2609.11744（2026-09） | 〔部分〕 | **單請求的「復用打平前沿」**（前綴比例 × 長度） | 缺口 2 的直接前作 |
| E9 | 隔離與側通道：`cache_salt`、Early Bird、Auditing Prompt Caching、KVGov | 2024–2026 | 〔摘要〕 | 共享快取的時序側通道與 salt 隔離 | 缺口 4 的安全面 |

### E1 vLLM PagedAttention＋Automatic Prefix Caching（APC）＋`cache_salt`

- **論文**：W. Kwon *et al.*, "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023. arXiv 2309.06180（已開；Comments 欄寫 SOSP 2023）。APC 是系統功能，設計文件 <https://docs.vllm.ai/en/latest/design/prefix_caching.html>。〔部分〕＋〔原始碼〕
- **問題**：KV 連續配置造成碎片與浪費；跨請求共享前綴時重複計算。
- **方法與假設**：
  - 每個 block 的 hash＝父 block hash＋本 block token＋額外鍵（LoRA ID、多模態 hash、`cache_salt`）〔設計文件〕。0.19.1 預設 `enable_prefix_caching=True`、`prefix_caching_hash_algo="sha256"`〔原始碼 `config/cache.py:66-68`〕。
  - 淘汰：free queue 當 LRU；請求釋放的 block 以**反序**放進佇尾，所以**請求的最後一個 block 最先被丟**，理由是它的 hash 涵蓋最多 token、最不可能被別人用到〔設計文件〕。
  - `cache_salt` 只混進**第一個 block** 的 hash，同 salt 的請求才能互相命中，用來擋「看延遲推測別人 prompt」的時序攻擊〔設計文件；原始碼 `v1/core/kv_cache_utils.py:518-526`〕。
  - 設計文件說 prefix caching「almost a free lunch」，沒有給額外開銷的量測〔設計文件〕。
- **改善**：論文摘要：同延遲下吞吐比 FasterTransformer、Orca 高 2–4 倍〔原文摘要〕。
- **限制／失敗條件**：只認**完全相同的前綴**；中間插入一個 token（例如剝掉 thinking）後面全部失效（repo `DIRECTIONS_20260927.md` 方向 4、19 已記）。
- **MI300X／ROCm**：APC 與硬體無關；0.19.1 有 Prometheus 指標 `vllm:prefix_cache_queries`、`vllm:prefix_cache_hits`、`vllm:external_prefix_cache_queries`（connector 命中）〔原始碼 `v1/metrics/loggers.py:505-538`〕。
- **已研究 vs 缺證據**：命中率的定義在文獻中有至少 6 種（repo `SOTA_ANALYSIS` §4.3）；**APC 本身在高併發、低重疊時的開銷**沒看到量測。
- **期限前最小版本**：X1（§7）的「APC 關 vs 開」兩欄就是這個。

### E2 SGLang／RadixAttention

- **論文**：L. Zheng *et al.*, "SGLang: Efficient Execution of Structured Language Model Programs," NeurIPS 2024（proceedings 已確認）. arXiv 2312.07104（已開）。〔摘要〕
- **方法**：KV 存在 radix tree；排程依「最長前綴匹配」排序以提高命中（DLPM 論文對 LPM 的描述，見 G8）。
- **改善**：摘要：最多 6.4 倍吞吐〔原文摘要；硬體、模型摘要未列〕。
- **MI300X**：本環境沒有 SGLang；ROCm 版可用性 NOT_MEASURED。它的 `bench_serving` 有可控前綴重疊的產生器（§6）。
- **缺證據**：LPM 會讓共享前綴多的租戶佔便宜——這正是 DLPM（G8）要修的公平性問題。

### E3 Preble：分散式的「復用 vs 負載」門檻

- **論文**：V. Srivatsa, Z. He, R. Abhyankar, D. Li, Y. Zhang, "Preble: Efficient Distributed Prompt Scheduling for LLM Serving," ICLR 2025（proceedings 與 poster 頁已確認；PDF 頁首誤印為 ICLR 2024）. arXiv 2407.00023（已開）。程式 <https://github.com/WukLab/preble>（Apache-2.0）。〔部分〕
- **問題**：單機前綴快取不管多 GPU；把同前綴都送同一台會負載不均，平均分散又浪費快取。
- **方法與假設**：
  - 研究 5 種負載（工具呼叫、embodied agent、程式生成、影片 QA、長文件 QA）：prompt 比輸出長 37–2494 倍，prompt 中 85–97% 的 token 和別的請求共享〔原文 p.2〕。
  - **E2 規則**：「省下的重算 > 新計算」時（即共享前綴 token 數 > 剩下非共享 token 數）才 exploit，送到快取最長前綴的 GPU；否則 explore，送到「prompt-aware 負載」最低的 GPU（近期 prefill 量＋為騰空間要淘汰的成本＋本請求成本，可重用部分記 0）〔原文 p.2〕。
  - 熱門前綴超過一台能扛時自動複製〔原文 p.2〕。
- **改善**：Mistral-7B 與 Llama-3-70B；4×A6000 叢集與 8×H100 伺服器（原文沒有逐一對應哪個模型在哪台）；五種負載加 Azure 到達模式；平均延遲比 SGLang 快 1.5–14.5 倍、p99 快 2–10 倍〔原文 p.2–3〕。
- **限制**：門檻是固定的 token 數比較，沒有把「從哪一層載入、載入多快」放進去〔判讀〕。
- **MI300X**：程式基於舊版 SGLang；移植 NOT_MEASURED。
- **和缺口 2 的關係**：這是文獻中**最早寫成明確規則的「什麼時候不值得去追命中」**，但只管路由，不管層級。

### E4 Marconi（repo 已有卡：`cards/A_Marconi_補充.md`）

- **論文**：R. Pan *et al.*, "Marconi: Prefix Caching for the Era of Hybrid LLMs," MLSys 2025. arXiv 2411.19379。〔部分〕
- **補充本文要的兩點**：
  - 淘汰分數 S(n)＝recency(n)＋α·FLOP efficiency(n)；最多 34.4 倍 token 命中率（TTFT 低 71.1% 或 617 ms）〔原文摘要、p.7〕。
  - 原文明說純 Transformer 的 KV「FLOP efficiency 近乎常數」，所以傳統系統只看 recency〔原文 p.6〕。〔判讀〕對 Llama-3.1-8B／Qwen2.5-7B 這兩個純 attention 模型，Marconi 的 FLOP 項幾乎沒有分辨力；但「依位置的重算成本」（越後面越貴）**不是**常數——這是 Tiara 的 κ(位置) 與 Marconi 的差別。

### E5 CacheBlend：單請求版的「依 κ 選儲存裝置」

- **論文**：J. Yao *et al.*, "CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion," EuroSys 2025（Best Paper，依 UChicago 新聞與 LMCache 頁；未開 ACM DL）. arXiv 2405.16444（已開，v3 2025-04）。程式在 LMCache。〔部分〕（本機 PDF 抽字讀 §5、§7、§9）
- **問題**：RAG 的多個 chunk 不在前綴位置，前綴快取用不上；直接拼接各 chunk 的 KV 會失去 chunk 間的 cross-attention，品質掉。
- **方法與假設**：
  - 每層只重算偏差最大的約 15% token（r\*＝15%，由實驗挑）〔原文 p.8〕。
  - **載入控制器**：第 i 層的選擇性重算可以和第 i+1 層的 KV 載入並行。先估 T_recompute(r%)＝r%×Prefill(LLM, L)（離線 profile）與 T_load(LLM, L, 裝置)（離線量裝置速度），挑 r 使兩者相等，再取 max(r, 15%)；**在固定 15% 時，從開發者給的裝置清單中選「T_recompute ≥ T_load 的最便宜裝置」**〔原文 p.8〕。
  - 例子：Llama-7B、4K，重算 15% 每層 3 ms，NVMe 載一層 16 ms → 載入蓋得住重算；Llama-70B 重算 15% 每層 7 ms，NVMe 載一層 4 ms → 蓋不住〔原文 p.8〕。
- **改善**：Runpod 2×A40、128 GB RAM、1 TB NVMe（實測 4.8 GB/s）；Mistral-7B、Yi-34B、Llama-70B；2WikiMQA（200 題）、Musique（150 題）、SAMSum、MultiNews；每請求 6 個 512-token chunk；TTFT 比完全重算快 2.2–3.3 倍、F1／Rouge-L 差 ≤0.02；吞吐 2.8–5 倍〔原文 p.10–12、摘要〕。慢磁碟（4 Gbps）時和「完全復用」的差距縮小，因為延遲被載入主導〔原文 p.13〕。
- **限制**：原文自列：只適用 Transformer；只整合 vLLM、沒測 DistServe 等；沒研究跨節點共享〔原文 p.13 §9〕。〔判讀〕載入控制器只看**單一請求**的 T_load；沒有考慮多個請求同時搶同一條 PCIe／SSD，也沒考慮 GPU 被別的請求佔用時 T_recompute 變大。
- **MI300X／ROCm**：CacheBlend 現在在 LMCache 裡；LMCache 的 ROCm 安裝頁沒有提到 blending 是否支援〔LMCache 文件〕。NOT_MEASURED。
- **和 Tiara 的關係**：**這是「用 T_recompute/T_load（≈κ）決定放哪一層」最早的明文規則**，必須引用並劃清界線：它是 RAG chunk、單請求、離線 profile、固定 15% 品質底線；Tiara 的 κ 是跨硬體實測、依位置、多層、多請求。

### E6 EPIC

- **論文**：J. Hu *et al.*, "EPIC: Efficient Position-Independent Caching for Serving Large Language Models," ICML 2025（PMLR v267，proceedings 頁已確認）. arXiv 2410.15332（已開）。〔摘要〕
- **方法**：位置無關快取（PIC）；LegoLink 處理每個文件開頭的 attention sink，只重算少量 token〔摘要、ICML 頁〕。
- **改善**：TTFT 最多 8 倍、吞吐最多 7 倍，準確度「negligible or no」損失〔摘要；模型與硬體摘要未列〕。
- **MI300X**：官方程式 NOT_FOUND（未查）。
- **同作者群**的 MPIC（多模態 PIC）已在 repo 卡片。

### E7 PIC 家族與 2025H2–2026 新作（速查表，全部〔摘要〕）

| 論文 | 年份／venue | arXiv（已開） | 原文數字（摘要） | 需要改模型？ | 本文的判讀 |
|:--|:--|:--|:--|:--|:--|
| KVLink | NeurIPS 2025（proceedings 已確認） | 2502.16002 | 7 個資料集平均 QA 準確度 +4%；TTFT 最多 −96% | 要（學習的特殊 token） | 品質靠訓練換 |
| APE | ICLR 2025（Comments 欄） | 2502.05431 | RAG 保留循序編碼 98%、ICL 93%；128K 時 prefill 時間少 28 倍、端到端 4.5 倍 | 摘要未說 | — |
| Block-Attention | ICLR 2025（Comments 欄） | 2409.15355 | 11 個 benchmark；32K 輸入 TTFT 約 45 ms；比 full attention 少 98.7% TTFT、99.8% FLOPs | 要（block fine-tuning） | 換模型 |
| TurboRAG | EMNLP 2025（ACL Anthology 2025.emnlp-main.334） | 2410.07590 | TTFT 最多快 9.4 倍、平均 8.6 倍 | 要（fine-tune） | 換模型 |
| MEPIC | 未查證（Huawei） | 2512.16822 | 比 SOTA PIC 少 2 倍 HBM、長 prompt 最多 5 倍 | 不用 | MiniPIC 說它沒公開程式碼 |
| MiniPIC | 未查證（IBM） | 2606.13126 | <100 行核心改動；2WikiMultihopQA prefill 吞吐 +49%（對 vLLM）；最壞開銷 5.7% | 不用 | 程式連結 NOT_FOUND（搜到的 `IBM/v11m` 在 GitHub API 回 Not Found） |
| KV-Repurpose | 未查證 | 2610.13060（2026-10-09） | Zipf RAG 負載、20% 請求共享固定前綴：30 GB 時 token 命中 63.9% vs 前綴快取 3.3%；前綴快取 1 TB 時也只有 17.9%；跨文件推理品質有限制 | 不用 | **有負載產生器**（Apache-2.0，§6） |
| Budgeted Cache Repair | 未查證 | 2610.02233 | 跨 context 復用在 MMLU、GSM8K 上「substantial accuracy」損失；挑重算位置在 64-token chunk 粒度只比隨機多消除 10.6% 誤差 | 不用 | **近似復用的品質代價是真的** |

- 〔判讀〕對 Tiara 的意義：非前綴命中能把命中率從個位數拉到 60% 以上（KV-Repurpose），但都要付「選擇性重算＋品質風險」。這把「重算」從「全有全無」變成連續變數——和 Cake 式的「部分重算」同一軸，但**目前沒有一篇把 PIC 的修補重算和跨層搬運成本一起最佳化**（CacheBlend 的載入控制器是唯一接近的）。

### E8 py-kvcache：單請求的「復用打平前沿」（repo `SOTA_ANALYSIS` §2.9 只讀摘要，這裡補正文）

- **論文**：J. Kanichai, T. De Matteis, A. Trivedi, "Building py-kvcache: A Performance Characterization of External KV Caching for vLLM with NVMe SSDs," arXiv 2609.11744（2026-09；已開）。程式 <https://github.com/atlarge-research/py-kvcache>（Apache-2.0）。〔部分〕（本機 PDF 抽字讀 §1、§5、§7–9）
- **問題**：外部 KV 快取（CPU、NVMe）什麼時候比重算快？
- **方法**：vLLM KV Offload connector，非同步 direct I/O、有上限的共用 staging、**排程感知的預載（請求還在排隊就開始讀磁碟）**〔原文 p.1〕。
- **改善／量到的邊界**：
  - 80K token 從磁碟載入比 LMCache 快 2.0 倍（預載貢獻 1.34 倍）；GPU＋CPU＋磁碟都開時比 LMCache 快 1.23 倍、和 vLLM 原生 KV Offload 差約 4%〔原文摘要〕。
  - **打平前沿**：Snellius（4×H100 PCIe5、Samsung PM1743）、Llama 3.2 3B、llm-d 檔案系統快取：**8K prompt 要 77.8% 的前綴命中才打平重算；80K 只要 7.8%**；「a cache hit is not sufficient for caching to be beneficial」〔原文 p.11〕。
  - Qwen3-4B 的 SSD 打平點 **6,203 token**；Bailian trace 在 H100 上平均請求低於打平點，而且 H100 的大 HBM 就留得住大部分工作集 → 外部快取幾乎沒好處；同樣 trace 在 RTX 4000 Ada 上有改善〔原文 p.20〕。
  - 結論原文：「外部 KV 快取應該被當成依每套硬體設定而定的准入決策」；「CPU 或 SSD 只在預測的搬運成本低於重算時才用，**重算是正當的退路，不是 cache miss**」〔原文 p.21〕。
- **限制（原文自己說）**：只以前綴大小與裝置頻寬設的靜態門檻「無法分辨需求讀與會被排隊遮住的預載」；更好的策略要結合前綴大小、來源層、staging 容量、各媒體實測速度、**預期排隊時間**〔原文 p.21〕。
- **MI300X**：CUDA 環境開發（vLLM v0.16–v0.22 fork）；0.19.1＋ROCm NOT_MEASURED。
- **和 Tiara 的關係**：這是缺口 2 的**單請求基線**。它已經把「依硬體而變的打平點」講出來了（和 Tiara 的「κ 跨硬體變 32 倍」同一個方向）。Tiara 要有新意，必須在它沒做的**併發／排隊／爭用**維度上找到前沿移動。

### E9 隔離與側通道（缺口 4 的安全面）

| 來源 | 讀到 | 重點 |
|:--|:--|:--|
| vLLM `cache_salt`（設計文件、0.19.1 原始碼） | 〔原始碼〕 | salt 混進第一個 block 的 hash；只有同 salt 才共享；文件沒有引用任何論文〔設計文件〕 |
| L. Song *et al.*, "The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems," IEEE TIFS（Comments 欄：已接受）. arXiv 2409.20002（已開） | 〔摘要〕 | 以時序推測快取命中、逐 token 還原共享前綴；黑箱測了線上服務 |
| C. Gu, X. L. Li, R. Kuditipudi, P. Liang, T. Hashimoto, "Auditing Prompt Caching in Language Model APIs," ICML 2025. arXiv 2502.07776（已開） | 〔摘要〕 | 在 **7 家 API 供應商（含 OpenAI）**偵測到跨使用者的全域快取共享 |
| T. C. Addagada, "Governing the KV Cache: Preventing Timing Side-Channel Leakage in Multi-Tenant LLM Inference," arXiv 2608.09225（2026-08；已開） | 〔摘要〕 | 3 種公開攻擊對沒保護的 vLLM／SGLang 最高 100% 成功；A100、vLLM 0.26.0、Qwen2.5-7B 冷／熱 TTFT 比 0.22；**salt 放在 prompt 分岔點而不是鏈頭，估計保留 93% 的快取好處**（防禦只做了模擬） |
| Anthropic、OpenAI 官方文件（2026-10-19 抓） | 〔部分〕 | Anthropic：組織間絕不共享；Claude API 上再以 **workspace** 隔離（Bedrock、Google Cloud 只到組織層）。OpenAI：不跨組織、不跨區域處理邊界 |

- 〔判讀〕隔離把一份可共享的前綴變成 N 份，**直接增加容量壓力**；KVGov 估的「保留 93%」是模擬。「隔離 × 容量 × 尾延遲」沒人實測——併入缺口 4。

### 1.10 E 方向的核心問題：復用效益會不會隨條件改變？有沒有「復用不划算」的區域？

| 條件 | 有證據的變化 | 來源 |
|:--|:--|:--|
| **前綴重疊程度** | 打平所需命中比例隨長度下降：8K 要 77.8%、80K 要 7.8%（H100、Llama 3.2 3B） | py-kvcache〔原文 p.11〕 |
| | 路由層：共享 token ≤ 非共享 token 時不追命中 | Preble〔原文 p.2〕 |
| | 短請求（<2,048 token）在任何引擎都有 ≥75% 命中（共用 system prompt），追 locality 沒好處；長請求才要黏住原引擎 | Autellix〔原文 §4.3〕 |
| **併發** | 50 併發共用 10K 前綴：0.19.1 開 CPU 卸載 TTFT 11.46 s vs 不開 5.95 s（L40S） | vLLM #44294〔原文 issue〕 |
| | 同一段 context 在 miss 解決前被多個請求要（delayed hit）是主要問題之一 | Strata〔repo 卡片〕 |
| | 16 個 agent 併發把同一 agent 的重用距離拉到中位 9.0 s（5 GiB host） | EfficientAgent〔repo 卡片〕 |
| | 即使 KV 瞬間載回，被趕出的請求仍要排隊等 GPU 記憶體 | Continuum〔原文 p.5〕 |
| **KV 容量** | host 5 GiB/rank 時「不寫」准入少 36% 重算；40 GiB/rank 時同一規則讓重算變 4.3 倍 | EfficientAgent〔repo 卡片〕 |
| | H100 的 HBM 留得住 Bailian 工作集 → 外部快取沒好處 | py-kvcache〔原文 p.20〕 |
| | 生產 trace 上 SSD 卸載對多數負載沒必要 | When Fancy Eviction Fails〔repo `SOTA_ANALYSIS` §2.11〕 |
| **GPU 爭用／搬運開銷** | GPU 只剩 12.5%、32 Gbps、70B 時 Cake 比只載還慢（0.80 倍） | Cake〔repo `SOTA_ANALYSIS` §2.1〕 |
| | 多個 workflow 搶 PCIe 時預取會擋住生成 | KVFlow〔原文 §3〕 |
| | **MI300X、vLLM 0.28：16K 從 CPU 載回 1.678 s，比 ROCM_ATTN 重算 1.460 s 還慢 15%**；ROCm 上每次 copy 有固定成本 | repo `F1_offload_slowdown.md` §0〔repo 實測〕 |

**「復用不划算」的區域（文獻＋本 repo 已看到的）**：
1. **短前綴／低命中比例**（py-kvcache；llm-d P2P 約 8K 打平、抓取有 1.2–1.3 s 固定下限〔repo 卡片 `D8_llmd_P2P.md`〕）。
2. **命中但在慢層**：搬運固定成本或低頻寬（llm-d P2P；repo D2 的 NFS 層等了 83 s 不重算〔repo `F1` §4.7〕；MTDS「不載反而快」〔repo `SOTA_ANALYSIS` §2.6〕）。
3. **併發＋慢層＋同一塊正在載**（vLLM 0.19.1 的延後機制；#44294）。
4. **容量大於工作集時還做寫入過濾**（EfficientAgent 40 GiB）。
5. **快 GPU＋大 HBM**（py-kvcache H100；NEO 的「GPU 記憶體越大好處越小」〔repo 卡片〕——MI300X 192 GB 很可能落在這裡〔判讀〕）。
6. **近似（非前綴）復用的品質代價**（Budgeted Cache Repair；KV-Repurpose 的跨文件推理）。

---

## §2 F：KV paging、配置、碎片、記憶體壓力

### 2.0 總表

| # | 論文 | 年份／venue | 讀到 | 一句話 |
|:--|:--|:--|:--|:--|
| F1 | PagedAttention：swap vs 重算 | SOSP'23 | 〔部分〕 | 小 block 時 swap 開銷爆炸；16–64 的 block 兩者相當 |
| F2 | vAttention | ASPLOS'25 | 〔摘要〕 | 不用 PagedAttention，虛擬連續 KV；吞吐最多 1.23 倍 |
| F3 | Jenga | SOSP'25 | 〔部分〕（repo 卡） | 異質層型的兩層配置器 |
| F4 | vLLM V1 搶佔＝只有重算；0.19.1 卸載的延後載入 | 系統 | 〔原始碼〕 | 被搶佔的請求 `num_computed_tokens=0` |
| F5 | Llumnix | OSDI'24 | 〔摘要〕（repo 卡） | 請求與 KV 即時遷移，尾延遲改善一個數量級 |
| F6 | FastSwitch | arXiv 2411.18424 | 〔摘要〕 | 公平排程下 context switch 的 swap 開銷 |
| F7 | KunServe | arXiv 2412.18169 | 〔摘要〕 | 過載時丟參數副本騰 KV 空間 |
| F8 | Oneiros（原名 MIRAGE） | arXiv 2507.11507 | 〔摘要〕 | 參數重映射給 KV，避免 swap |
| F9 | 輸出長度不確定下的 KV 保留 | arXiv 2607.16892 | 〔摘要〕 | 預留不足 → 搶佔＝終止＋重算 |

### F1 PagedAttention 的 swap vs 重算（§4.5、§7.3）

- 出處同 E1。〔部分〕（PDF 抽字讀 §4.5、§7.3）
- **方法**：GPU block 用完時選序列整組搶佔；兩種還原：**swap** 到 CPU（CPU swap 空間上限＝GPU 的 KV 空間）、**重算**（把已生成的 token 接回 prompt，一次 prefill 算回來）〔原文 §4.5〕。
- **結果**：「swap 在小 block 時開銷過大，因為很多小傳輸讓 PCIe 有效頻寬變低；重算的開銷不隨 block 大小變……recomputation overhead is never higher than 20% of swapping's latency；16–64 的中等 block 兩者端到端相當」〔原文 §7.3；OPT-13B、ShareGPT〕。
- 〔判讀〕這就是「每次搬運有固定成本」（repo 故事 2、F1 的 ROCm 每次 copy 約 6 µs）在 2023 年的版本：**搬運單位越碎，κ 越偏向重算**。
- **期限前**：vLLM V1（含 0.19.1）只剩重算（F4），swap 對照組無法直接跑。

### F2 vAttention

- **論文**：R. Prabhu, A. Nayak, J. Mohan, R. Ramjee, A. Panwar, "vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention," ASPLOS 2025（Comments 欄）. arXiv 2405.04437（已開）。〔摘要〕
- **改善**：吞吐比 FlashAttention／FlashInfer 的 PagedAttention kernel 最多高 1.23 倍〔摘要〕。
- **MI300X**：依賴 GPU 驅動的虛擬記憶體介面；ROCm 可移植性 **未查證**。期限內不建議碰。

### F3 Jenga（repo 卡 `cards/A_Jenga.md`）

- SOSP 2025（依清華頁；未開 ACM DL）。arXiv 2503.18292。以各 embedding 大小的最小公倍數為大 page，再切給各層型；每層型自訂命中與淘汰規則。只管 GPU 記憶體，沒做卸載〔repo 卡片〕。
- 對 Llama-3.1-8B／Qwen2.5-7B（單一層型）幾乎無關；對 hybrid 模型才重要。

### F4 vLLM V1 的搶佔與 0.19.1 卸載的「延後載入」（〔原始碼〕＋上游 issue）

- **搶佔**：V1 預設也是唯一的方式是重算；`_preempt_request` 把 `num_computed_tokens` 歸零〔repo 卡 `D8_vLLM_Optimization_Preemption.md`；原始碼 `v1/core/sched/scheduler.py:956-968`〕。指標有 `num_preemptions`〔原始碼 `v1/metrics/loggers.py:133-144`〕。
- **0.19.1 `OffloadingConnector` 的延後載入**：開 GPU prefix caching 時，排程端用 `_blocks_being_loaded` 追蹤正在從 CPU 載回的 block；**新請求命中的 block 如果正被別人載入，`get_num_new_matched_tokens` 回 `None`，請求被延後**〔原始碼 `distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:47-51、126-140`〕。
- **vLLM issue #44294**（2026-06-02 開，〔原文〕）：2×L40S、TP=2、Qwen3-8B、`cpu_bytes_to_use`≈57.6 GB、50 併發共用 10,000-token 前綴、輸出 128：TTFT 不卸載 5.95 s、0.19.1 卸載 11.46 s、0.17.1 卸載 0.90 s；吞吐開銷 0.19.1 −39.9%、0.17.1 −6.5%。維護者沒回應；2026-10-07 被機器人以 not planned 關閉，修正 PR #44295 同樣關閉。**〔2026-10-26 更正〕PR #44295 沒有關閉，至 2026-10-26 仍是 OPEN（stale、needs-rebase）；見 `docs/kv_sharing_hiding_20261026/lit/L1_strata_and_engines.md` §2.2、`lit/V_verifier.md` #11。**
- 〔判讀〕這是 Strata 所說 delayed hit 的工程版本，而且**正好在我們的版本裡**。在 MI300X 上是否同樣發生：NOT_MEASURED（repo F1 的卸載測試用的是 0.28）。
- **期限前最小版本**：X2（§7），半天。

### F5 Llumnix（repo 卡 `D8_Llumnix.md`）

- B. Sun *et al.*, OSDI'24（Comments 欄）. arXiv 2406.03243。多實例間即時遷移請求與 KV，處理負載不均、碎片、優先權；尾延遲改善一個數量級、高優先權請求快 1.5 倍、同尾延遲下省 36% 成本〔摘要〕。
- 單機 2 卡、TP 不用的情境下，「遷移」退化成「兩張卡各自排程」；期限內不測。

### F6 FastSwitch

- A. Shen, Z. Li, M. Gao, "FastSwitch: Optimizing Context Switching Efficiency in Fairness-aware Large Language Model Serving," arXiv 2411.18424（2024-11，只有 v1；venue NOT_FOUND）。〔摘要〕
- 公平排程需要頻繁搶佔與換入換出；在不同尾端 TTFT／TBT 指標上比 vLLM 快 1.4–11.2 倍〔摘要〕。〔判讀〕把公平性（G8）和 swap 粒度（F1）接起來；V1 引擎已無 swap。

### F7 KunServe

- R. Cheng, Y. Lai, X. Wei, R. Chen, H. Chen, "KunServe: Parameter-centric Memory Management for Efficient Memory Overloading Handling in LLM Serving," arXiv 2412.18169（v5 2025-10；venue NOT_FOUND）。〔摘要〕
- 過載時丟掉複製的模型參數騰出 KV 空間，而不是搶佔請求；限流下的尾 TTFT 比 Llumnix、vLLM、InferCept 最多低 72.2 倍〔v5 摘要〕。〔判讀〕需要多副本（多卡）才有東西可丟，單卡 8B 不適用。

### F8 Oneiros（arXiv 2507.11507，原名 MIRAGE）

- R. Li *et al.*；v2 2025-10-29 改名 Oneiros；venue NOT_FOUND。〔摘要〕把參數記憶體重映射給 KV，避免 swap；多租戶下尾 TBT −44.8–82.5%、尾 TTFT −20.7–99.3%、吞吐 +6.6–86.7%（對 vLLM）；以 Grace Hopper 的高 CPU-GPU 頻寬為例〔摘要〕。

### F9 輸出長度不確定下的 KV 預留

- J. Cheng, D. T. Do, D. T. Nguyen, "Robust KV Cache Management for LLM Serving under Output Token Length Uncertainty," arXiv 2607.16892（2026-07；venue NOT_FOUND）。〔摘要〕
- 預留不足 → 搶佔（終止＋重算）；用 BurstGPT、Azure、ShareGPT，成本比固定分位數預留最多低 56%；P99、goodput、SLO 違反只說「competitive」〔摘要〕。

### 2.10 F 方向小結

- **已研究**：配置與碎片（PagedAttention、vAttention、Jenga）已解；過載時的處理有三派：搶佔重算（vLLM V1）、遷移（Llumnix）、騰參數（KunServe、Oneiros）。
- **缺證據**：「容量壓力下的 thrash」——KV 被趕到 CPU 又被拉回、或反覆被搶佔重算——**多數用吞吐或平均延遲報告，沒有報「重複搬運／重算的量」**。EfficientAgent 是例外（量了 host 寫入 token 數，79.2% 的寫入在下一次呼叫前就被淘汰〔repo 卡片〕）。repo D4 也量到 Mooncake trace 75–78% 的 block 寫了沒人讀〔repo RUNLOG_MI300X〕。

---

## §3 G：多請求吞吐、goodput、尾延遲、准入、KV 感知路由

### 3.0 goodput 的定義（同一個字，至少三種）

| 來源 | 定義 | 備註 |
|:--|:--|:--|
| DistServe（OSDI'24） | 同時滿足 TTFT 與 TPOT 限制下每 GPU 能服務的最大請求率 | 摘要：最多多服務 7.4 倍請求或 SLO 緊 12.6 倍；>90% 請求在限制內〔摘要〕 |
| Mooncake（FAST'25） | 滿足 SLO 的吞吐，但**只算完整做完的請求**（被拒絕或中途失敗的不算） | 端到端實驗設 TTFT_P90＝單一請求無干擾時的 10 倍、TBT_P90＝5 倍；P90 超過門檻的資源算浪費〔原文 p.4、p.15〕 |
| vLLM 0.19.1 `bench serve --goodput`、SGLang `bench_serving --goodput` | 請求同時滿足列出的 ttft／tpot／e2el（ms）才算「good」；輸出 request goodput 與 SLO 達成率 | 0.19.1〔原始碼 `benchmarks/serve.py:398-475、1455`〕；SGLang〔官方文件〕 |

〔判讀〕Tiara 若以 goodput 為目標，必須先選定義並寫死 SLO（建議沿用 vLLM `--goodput ttft:<ms>`，因為可以直接在 0.19.1 跑）。

### G1 DistServe（repo 卡 `D5_DistServe.md`）

- Y. Zhong *et al.*, OSDI 2024. arXiv 2401.09670（已開）。prefill／decode 分離、各自配資源以最大化 goodput〔摘要〕。單節點 2 卡上 P/D 分離有意義，但和 Tiara 的「單請求 × context 遞增」主軸無關；只取它的 goodput 定義。

### G2 Sarathi-Serve（repo 卡 `D8_SarathiServe.md`）

- A. Agrawal *et al.*, OSDI 2024（USENIX 頁已確認）. arXiv 2403.02310（已開）。chunked prefill＋stall-free 排程；尾延遲限制下服務容量：Mistral-7B 單張 A100 2.6 倍、Yi-34B 兩張 A100 最多 3.7 倍、Falcon-180B（PP）最多 5.6 倍（對 vLLM）〔摘要〕。
- 〔判讀〕vLLM 0.19.1 預設已有 chunked prefill；Cake／重算線可以搭 decode 便車（repo 方向 4）。

### G3 Mooncake 的 Conductor 與過載排程（repo 卡 `A_Mooncake_補充.md` 補充）

- R. Qin *et al.*, FAST 2025. arXiv 2407.00079。〔部分〕（本機 PDF 抽字讀 §1–3、§6–7）
- **和 KV 放置直接相關的三句**：
  1. 「prefill 階段主要目標是盡量復用 KV……但**等存在較低層的 KV 可能違反 TTFT SLO**」；Conductor 要預測 block 未來用量、做 swap 與複製〔原文 p.2〕。
  2. 若最佳遠端前綴長度不超過本地可重用前綴乘以一個門檻，**偏好直接計算**；門檻「目前手調」〔原文 p.11，註腳〕。
  3. 過載時做**預測式早期拒絕**；直接早拒會造成負載震盪，加上負載預測可緩解〔原文 p.2、§7〕。
- **改善**：模擬情境最多 525% 吞吐（滿足 SLO）；真實負載讓 Kimi 多處理 75% 請求〔原文摘要〕。
- **限制**：多節點、RDMA；門檻手調〔原文〕。
- 〔判讀〕Mooncake 是「以 SLO 為目標的 KV 放置」最早的生產系統，但決策粒度是「選哪台 prefill 節點、要不要拉遠端 KV」，沒有跨 GPU／CPU／SSD／重算的 κ 模型。

### G4 llm-d：精確前綴快取感知路由（官方部落格，2025-09-24）

- M. Ayoub *et al.*（IBM、Red Hat、Google、Alibaba Cloud、DaoCloud），"KV-cache wins you can see," llm-d blog, <https://llm-d.ai/blog/kvcache-wins-you-can-see>。〔部分〕（WebFetch 取得全文摘要）
- **設定**：8 個 vLLM pod × 2×H100（共 16 張），每 pod KV 307,328 token；150 個企業客戶各有 6,000-token 共享前綴、每客戶 5 個併發使用者、1,200-token 問題；Poisson 到達 3→60 QPS；KV 需求約為叢集容量的 73%〔原文〕。
- **結果**（輸出 tok/s；TTFT P90）：precise 8,730／0.542 s；approximate 6,944／31.083 s；load-only 4,429／94.865 s；random 4,429／92.551 s〔原文表〕。原文沒有 P99。
- **何時沒用**：原文說隨機或 round-robin 對「對稱、重用很少」的負載就夠了；RAG 因檢索文件順序常變而難快取；**沒有量低重疊或低負載**〔原文〕。
- **MI300X**：靠 vLLM 的 KV 事件（ZMQ）；0.19.1 有 `KVEventsConfig(enable_kv_cache_events, publisher="zmq")`〔原始碼 `config/kv_events.py:11-20`〕。單機 2 卡可以做「兩個 vLLM 實例＋簡單路由器」的縮小版，但期限內不建議。
- NVIDIA Dynamo 的 KV router：本次沒開文件，NOT_VERIFIED。

### G5 Autellix 的負載平衡（細節在 H3）＋ DualMap

- Autellix：短請求（<2,048 token）在任何引擎都有 ≥75% 命中，所以送最閒的引擎；長請求黏住它所屬程式的引擎；負載平衡器最多 1.5 倍吞吐〔原文 §4.3、p.4〕。
- Y. Yuan *et al.*, "DualMap: Enabling Both Cache Affinity and Load Balancing for Distributed LLM Serving," arXiv 2602.06502（2026-02；venue NOT_FOUND）。同 TTFT SLO 下有效請求容量最多 2.25 倍〔摘要〕。〔摘要〕

### G6 Cascade：以 SLO 延遲預算決定「從深層還原／留在 HBM／重算」

- M. Adnan, R. Mahapatra, P. J. Nair, D. Berger, P. Zardoshti, R. Fonseca, E. Choukse, "Cascade: Exploiting SLO-Aware latency budget for fair and high goodput LLM inference serving," arXiv 2608.06557（2026-08-06；venue NOT_FOUND）。〔摘要〕
- 摘要：記憶體管理器用延遲預算決定不在 HBM 的 KV 是「從較深層還原或預取、留在 HBM、或重算」；生產 trace、三個 LLM；goodput 最多 2.4 倍、SLO 違反少 40%（對 vLLM 預設 FCFS）〔摘要〕。
- **這是缺口 1、3 最大的新穎性威脅**。決策規則、門檻、硬體、是否有 SSD、是否跨硬體比較：摘要都沒寫。**必須讀全文再決定第三階段題目**（§5）。

### G7 SLOs-Serve

- S. Chen, Z. Jia, S. Khan, A. Krishnamurthy, P. B. Gibbons, "SLOs-Serve: Optimized Serving of Multi-SLO LLMs," arXiv 2504.08784（2025-04；venue NOT_FOUND）。6 種應用情境（摘要、程式、聊天、工具呼叫、推理…）；每 GPU 服務容量平均 2.2 倍〔摘要〕。〔摘要〕不涉及 KV 分層。

### G8 公平性：VTC、DLPM、FairInference

| 論文 | 讀到 | 公平性量的是什麼 | 原文數字 | 和 KV 容量的關係 |
|:--|:--|:--|:--|:--|
| Y. Sheng *et al.*, "Fairness in Serving Large Language Models," OSDI'24（USENIX 頁已確認）. arXiv 2401.00588；程式 `Ying1123/VTC-artifact`（Apache-2.0） | 〔摘要〕 | 輸入＋輸出 token 的加權服務量（Virtual Token Counter） | 兩個 backlogged 客戶的服務差有 2 倍的緊上界〔摘要〕 | 不看命中，也不看誰佔 KV |
| S. Cao *et al.*, "Locality-aware Fair Scheduling in LLM Serving," arXiv 2501.14312（2025-01；venue NOT_FOUND） | 〔部分〕 | **前綴只在第一次算進 GPU 時計費一次**的服務量；DRR 式 quantum Q 換取 LPM 局部性 | 比 VTC 吞吐最多 2.87 倍；比 SOTA 分散式系統每客戶延遲最多低 7.18 倍；Llama-3.1-8B、Llama-3.2-3B；A100 80GB、A10G〔原文〕 | 公平的是**計算**；命中多的客戶付得少 |
| D. Bali *et al.*, "Token Latency Fairness: Performance Isolation for Multi-Tenant LLM Serving," arXiv 2609.18112（2026-09；venue NOT_FOUND） | 〔摘要〕 | token 延遲：隔離時要 d 的 token，多租戶下要在 d＋δ 內 | 摘要沒有數字 | 摘要說排程有考慮「GPU 記憶體中共享 KV 快取造成的額外延遲」 |
| M. R. Flax, "A Biophysically-Inspired Feedback Controller for Multi-Class Cache Fairness," arXiv 2608.14561（2026-05） | 〔摘要〕 | 最差類別的 miss ratio | 合成多類別負載：補回 LRU→Belady 差距的 27–72%〔摘要〕 | 類別像 system prompt／文件／程式／對話歷史共用一個淘汰池；**沒用真實引擎** |

- 〔判讀〕沒有一篇把公平性定義在「**各租戶佔用哪一層、佔多久**」上，也沒有一篇量「共享 LRU 池讓高重用租戶擠掉低重用租戶」對後者 P95 的影響。這是缺口 4。

---

## §4 H：agent 多輪、重複前綴、KV 存活時間

### 4.0 工具暫停時 KV 怎麼辦：現有做法一覽

| 系統 | 動作空間 | 用什麼決定 | 有 SSD？ | 有跨硬體的成本比？ | 讀到 |
|:--|:--|:--|:--|:--|:--|
| vLLM／SGLang 預設 | 請求結束就可被淘汰（等於「丟，靠 APC 運氣」） | LRU | 看是否開卸載 | 無 | 〔原始碼〕 |
| InferCept（ICML'24） | 丟（重算）／留在 GPU／換到 CPU | **GPU 記憶體 × 時間的浪費**最小 | 無 | 無（離線 profile 一次） | 〔部分〕 |
| Continuum（ICLR'26 workshop） | 留在 GPU 多久（TTL） | 工具時長經驗 CDF × （排隊成本＋Prefill-Reload）− 佔用 | 無 | Prefill-Reload **二選一**：開卸載就用 reload，否則用 prefill；離線 profile | 〔部分〕 |
| Tokencake（arXiv 2510.18586） | 卸到 CPU 與否、何時載回 | T_fc > T_offload＋T_upload 且空出的時間塞得下一個等待請求 | 無 | 無（假設搬運遠小於重算） | 〔部分〕 |
| KVFlow（NeurIPS'25） | 淘汰順序＋從 CPU 預取 | Agent Step Graph 的「幾步後執行」 | 無 | 無（假設 PCIe 比重算快很多） | 〔部分〕 |
| MORI（arXiv 2606.00866） | HBM／DRAM 依閒置程度排名 | 程式閒置程度；兩層都有准入 | 無 | 原文承認短空檔的搬運可能超過空檔本身 | repo 卡〔部分〕 |
| EfficientAgent（arXiv 2609.33762） | host 層寫不寫 | 工作集 vs host 容量的壓力訊號 | 無 | 無 | repo 卡〔部分〕 |
| Lachesis（arXiv 2610.08378） | HBM／HBF | harness 給的 segment 壽命 | HBF（模擬） | 無 | repo 卡〔全文〕 |
| Bidaw（FAST'26） | DRAM／SSD | 上一輪回答長度預測多久回來 | 有 | 無；只對真人聊天成立 | repo 卡〔全文〕 |
| Cascade（arXiv 2608.06557） | 從深層還原／留 HBM／重算 | SLO 延遲預算 | 摘要提到「deeper tier」 | 摘要未說 | 〔摘要〕 |
| Copilot 特徵化（arXiv 2608.00101） | 建議：輪內留 GPU、輪邊界卸到 DRAM／磁碟、預測閒置時間決定何時卸 | 閒置時間預測（ROC-AUC 0.73） | 建議有 | 無 | 〔部分〕 |

**一句話**：所有人都在決定「留或卸」或「留多久」，**重算這個選項的成本都用離線 profile 一次、而且多半只在 CPU 一層上比**；沒有人把 GPU／CPU／SSD／丟棄四個動作放在同一個決策裡，用依硬體實測的成本比做門檻〔判讀；查不到不等於沒有〕。

### H1 InferCept

- **論文**：R. Abhyankar, Z. He, V. Srivatsa, H. Zhang, Y. Zhang, "InferCept: Efficient Intercept Support for Augmented Large Language Model Inference," ICML 2024（PMLR 235:81–95，已確認）. arXiv 2402.01869（已開）。程式 <https://github.com/WukLab/InferCept>（Apache-2.0，2024 版 vLLM）。〔部分〕（PDF 抽字讀 §2–4）
- **問題**：工具／外部互動（interception）時，現有系統把請求結束、回來重算；重算占總 forward 時間 37–40%〔摘要〕。
- **方法與假設**：
  - 三種做法的「GPU 記憶體浪費」（記憶體 × 時間）：Discard＝重算時間 ×（自己的 context＋其他請求的 context）× M；Preserve＝攔截時長 × 自己的 context × M；Swap＝2 × 搬運時間 × batch context × M〔原文 Eq.1–3〕。
  - Swap 改成逐層 pipeline、分多個 iteration 做，每步可「免費」搬的 token 數 N_i 由 T_swap(N_i)＝T_fwd(B_i) 決定（swap budget）；重算也切 chunk 和 decode 混跑〔原文 §4.1–4.2〕。
  - 排程：依潛在浪費排序，swap budget 給浪費最大的，其餘比較 Discard 與 Preserve 的浪費〔原文 §1、§4.3〕。
  - 攔截時長估計：T̂_INT＝t_now − t_call（攔越久估越長），達到知道真實時長的 oracle 的 93%〔原文 §4.4〕。
- **原文數字**：A100；GPT-J-6B、Vicuna-13B、Llama3-70B；六種攔截（數學、QA、虛擬環境、聊天、圖片生成、TTS），平均攔截時長 9×10⁻⁵ s 到 28.6 s，攔截時的 context 平均 753–2,185 token〔原文 Table 1〕。Discard 浪費 27% GPU 資源；Swap 浪費 26%、超過 25% 時間在等 swap；Preserve 時超過 60% 的時間有將近一半 GPU 記憶體被暫停的請求佔住〔原文 §3〕。結果：比 vLLM 多撐 1.6–2 倍負載、完成的請求數多 2 倍以上〔原文 p.2〕。
- **限制**：〔判讀〕context 很短（約 1–2K），重算便宜；只有 GPU／CPU；搬運速度與重算速度離線量一次。**長 context（≥32K）下三者的取捨會完全不同**。
- **MI300X**：舊 vLLM fork，移植 NOT_MEASURED；期限內只能把它的浪費公式放進模擬（X3）。
- **和 Tiara 的關係**：**最接近 Tiara 動作空間的前作**（三選一＋成本模型）。Tiara 要多的是：SSD、長 context 下依位置的重算成本、跨硬體的 κ。

### H2 Parrot

- C. Lin *et al.*, "Parrot: Efficient Serving of LLM-based Applications with Semantic Variable," OSDI 2024（Comments 欄）. arXiv 2405.19888（已開）。〔摘要〕
- 用「語意變數」把多次 LLM 呼叫的相依關係暴露給服務端，做應用層排程與共用前綴；常見應用最多改善一個數量級〔摘要〕。〔判讀〕證明「應用層提示」有用；但不處理暫停期間 KV 放哪。

### H3 Autellix（NSDI'26 版改名 Agentix）

- M. Luo *et al.*, "Autellix: An Efficient Serving Engine for LLM Agents as General Programs," arXiv 2502.13965（已開，只有 v1）。USENIX 網站有 NSDI'26 spring 的預印 PDF，標題 "Agentix"，同作者——**接受與否只由搜尋看到 PDF 檔名推得，未開 USENIX 議程頁確認**。官方程式 NOT_FOUND（GitHub 只有第三方重現）。〔部分〕
- **方法**：程式層級排程 PLAS（單執行緒程式，依程式已獲服務量給優先權）與 ATLAS（DAG 程式，依關鍵路徑）；離散化優先權減少搶佔造成的 GPU-CPU swap；負載平衡器依請求長度決定要不要黏 locality（G5）〔原文 §4〕。
- **原文數字**：同延遲下程式吞吐 4–15 倍（對 vLLM 等）〔摘要〕；分析用 LLaMA-3.1-8B、單張 A100-80GB 的一小時 trace〔原文 §3〕；負載 ShareGPT、BFCL、LATS、MCTS〔原文圖〕。
- **限制**：Continuum 指出 Autellix 在 SWE-Bench 上表現差，因為它假設「跑得越久的程式剩下越多」〔Continuum 原文 §6.2〕。

### H4 KVFlow（repo D4 表內已有一行，這裡補評測細節）

- Z. Pan *et al.*, "KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows," NeurIPS 2025（proceedings 已確認）. arXiv 2507.07400（已開）。官方程式 NOT_FOUND。〔部分〕
- **方法**：Agent Step Graph 給每個 agent「幾步後會執行」；據此做 radix tree 節點層級淘汰；背景執行緒從 CPU 預取下一步會用的 KV；預取來不及時改排已在 GPU 的 agent〔原文 §3〕。
- **假設**：原文 Fig. 2 說 PCIe 傳 KV 遠比重算快，所以卸到 CPU 是記憶體壓力下的有效策略〔原文 §2〕。
- **原文數字**：(1) Llama-3.1-8B、A10G 24GB、原文寫「2GB/s PCIe Gen1」；(2) Qwen2.5-32B、H100 80GB、64 GB/s PCIe Gen5；10-agent 循序 workflow，合成 prompt（固定 4,096／8,192 + 動態後綴）；batch 1；對 SGLang（含 HiCache）單 workflow 最多 1.83 倍、多 workflow 併發最多 2.19 倍〔原文 §4、摘要〕。
- **限制**：〔判讀〕合成 prompt、固定前綴；κ 被假設成「搬運永遠便宜」；併發時 PCIe 爭用被原文承認會擋住生成〔原文 §3〕。

### H5 Continuum（repo 卡 `D8_Continuum.md` 只讀到 p.5，這裡補 §4–6）

- **論文**：H. Li, Q. Mang, R. He, Q. Zhang, H. Mao, X. Chen, H. Zhou, A. Cheung, J. E. Gonzalez, I. Stoica, "Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live," arXiv 2511.02230（已開；v4 曾改名 CacheTTL）。**ICLR 2026 Workshop "Lifelong Agents" poster**（iclr.cc 頁已開，不是主會議）。程式 <https://github.com/Hanchenli/vllm-continuum>（Apache-2.0，描述「Preview Code」）。〔部分〕（本機 PDF 抽字讀 §3–6）
- **方法與假設**：
  - 每個帶工具呼叫的請求選 TTL：τ\*＝argmax_τ P(τ, f)·(T̄·η＋Prefill-Reload(r)) − τ。P(τ, f) 是該工具名稱歷史時長的經驗 CDF；T̄ 是被趕出的請求平均排隊時間；η＝−Corr(k, N−k) 是「記憶性」（程式剩下幾步是否可預測）〔原文 §4.1–4.2 Eq.1–2〕。
  - **Prefill-Reload(r)「依有沒有開 CPU 卸載，取 prefill 或 reload 的時間」**；離線 profile（GPU-CPU 頻寬取平均卸載吞吐；prefill 對 context 長度擬合二次曲線；每對硬體＋模型 <10 分鐘）〔原文 §4.1、§5.2〕。
  - 冷啟動：歷史記錄 ≤K（K＝100）時用全域 CDF 或用 Exp(1)、η＝1 推出的固定 TTL〔原文 §4.2〕。
  - 排程優先序：被搶佔的 > TTL 內被釘住的 > 依程式到達順序 FCFS；空間不夠時從最晚到達的程式解釘，避免死鎖〔原文 §4.3、§5.2〕。
- **原文數字**：A100-SXM（Runpod）、H100（AWS、某公司）、B200；Llama-3.1-8B／70B、Gemma-3-12B，摘要另提 GLM-4.5 355B；SWE-Bench（mini-swe-agent，用 GPT-5 產生 trace）、BFCL v4、OpenHands；Poisson 到達；對照 vLLM 0.10.2、LMCache 0.3.7（DRAM 100／200 GB）、Autellix、InferCept、SGLang 0.5.5、Dynamo 0.7.0〔原文 §6.1〕。平均工作完成時間改善「over 8x」（內文 up to 8.18x）；Llama-3.1-8B 對 vanilla vLLM 的平均回應時間最多降約 2 倍；P90／P95 也較好〔原文 §6.2〕。原文提醒 agentic 負載的 jobs/s 遠低於以往 serving 論文（單機圖的橫軸約 0.01–0.125 JPS；分散式實驗約 0.6–3.0 JPS）〔原文圖 9、11、12〕。
- **限制**：〔判讀〕(1) 只有「留在 GPU 多久」一個旋鈕，**沒有在 CPU／SSD／重算之間選**；(2) Prefill-Reload 的「重算 vs 載入」不是被比較出來的，是看設定；(3) 合成到達（Poisson）；(4) 原文自己說：即使載回瞬間完成，被趕出的請求仍要排隊——**排隊才是大頭**（repo 方向 9 因此判死）。
- **MI300X**：vLLM 0.10.2 fork；移植到 0.19.1 期限內不可行。它的 TTL 公式可以直接放進 X3 模擬當對手。

### H6 Tokencake

- Z. Bian, F. Wu, T. Ma, Y. Zhuo, "Tokencake: A KV-Cache-centric Serving Framework for LLM-based Multi-Agent Applications," arXiv 2510.18586（v2 2025-10-31；已用搜尋與 HTML 確認存在）。官方程式 <https://github.com/pku-lemonade/TokenCake>（Apache-2.0）；repo 描述寫「[EuroSys27]」——**venue 只見於 repo 描述，未確認**。〔部分〕（WebFetch 讀 HTML v2）
- **方法**：Space Scheduler 動態分割記憶體保護關鍵路徑 agent；Time Scheduler 在函式呼叫期間卸到 CPU、預測完成前主動載回。規則：估 T_transfer＝T_offload＋T_upload（隨 block 數線性）；預測 T_fc；**T_fc ≤ T_transfer 不卸**；否則把 T_fc − T_transfer 換成 token 容量，**有等待請求塞得進去才卸**；時長預測 t＝α·t_req＋(1−α)·t_hist（EWMA）〔原文 Alg.1〕。
- **原文數字**：A100 80GB（Qwen2.5-14B）、H200 140GB（Qwen2.5-32B）；CPU swap 100 GB；Code-Writer 與 Deep Research（由 ShareGPT、AgentCode 合成，Poisson）；**工具時長用 Poisson 模擬，不是真工具**；4,096 個 block 搬運約 60 ms、重算約 9,000 ms；1.0 QPS 時端到端延遲比 vLLM 低 47.06% 以上；KV 利用率高 16.9%〔原文〕。原文自列限制：時長預測模型簡單、只測單 GPU、工具是模擬〔原文〕。
- 〔判讀〕Tokencake 的「卸不卸」門檻是 T_fc vs T_transfer——**重算從來不是選項**，因為它在 NVIDIA 上量到搬運比重算便宜 150 倍。若 κ 在某硬體或某長度翻轉（repo F1 在 MI300X／vLLM 0.28 已看到 16K 時 CPU 載回比重算慢），這條規則就會做錯決定。

### H7–H9 MORI、EfficientAgent、Lachesis（repo 已有卡片，這裡只補對缺口的用途）

- **MORI**（arXiv 2606.00866，〔部分〕，`cards/A_MORI.md`）：依程式閒置程度把 HBM／DRAM 分層，H200（限 80 GB）／B200、SGLang v0.5.10＋HiCache、Claude Code 真實 trace；比最好的卸載基線吞吐高 20–71%、TTFT 低 18–43%。搜尋摘要另提到原文寫「短工具空檔的搬運成本可能超過空檔本身，讓卸載適得其反」〔搜尋摘錄，未回原文核對頁碼〕。
- **EfficientAgent**（arXiv 2609.33762，〔部分〕，`cards/A_EfficientAgent.md`）：容量條件的 host 寫入准入；同一規則在 5 GiB 有益、40 GiB 有害。8×H20 為主，另有 8×RTX 3090；vLLM 0.13.0＋LMCache 0.3.12；OpenHands 重播。
- **Lachesis**（arXiv 2610.08378，〔全文〕，`cards/A_Lachesis.md`）：harness 給 segment 壽命 → 寫入時選 HBM／HBF；全模擬。
- 〔判讀〕三篇加起來說明：**2026 年的 agent KV 論文已經把「何時卸／要不要寫／寫到哪」三個點都佔了**，但都沒有「丟掉、需要時重算」作為一個依硬體而變的正式動作。

### H10 2026 新出、只讀摘要的 agent KV 系統

| 論文 | arXiv（已開） | 讀到 | 原文數字（摘要） |
|:--|:--|:--|:--|
| R. Zhang *et al.*（含 J. Jiang），"Learning Agent Execution for KV-Cache Management in Agentic Serving"（系統名 CacheScout） | 2608.14624（2026-07-16） | 〔摘要〕 | vLLM；KV 命中 +10–18 個百分點、平均 TTFT −18–45%、每輪延遲 −29–38%、峰值吞吐 +57%；學習的執行模型指導淘汰與主動預取 |
| H. Ni *et al.*, "TOPAS: Workflow-Aware Prefix-State Scheduling for Multi-Agent LLM Serving" | 2608.25523（2026-08-26） | 〔摘要〕 | SGLang；合成 DAG 平均 JCT 最多 −39.8%、p99 −49.4%；MetaGPT-TL 平均 −22.0%、p99 −26.6% |
| H. Zheng *et al.*, "Efficient Serving for Dynamic Agent Workflows with Prediction-based KV-Cache Management" | 2605.06472（2026-05-07） | 〔摘要〕 | 動態 workflow 比 LRU 最多 1.85 倍；靜態 workflow 比 KVFlow 最多 1.26 倍 |
| B. Ma, J. Eitzinger, H. Koestler, "Leyline: KV Cache Directives for Agentic Inference" | 2606.01065（2026-05-31） | 〔摘要〕 | 宣告式 KV 編輯指令（就地拼接或截斷重 prefill）；重播命中 +11.2 pp、延遲最多 −241 ms；debug-gym 解題率 +14.3 pp |

### H11 工作負載特徵化（2026 新增）

- **TraceLab**（arXiv 2606.30560；repo `workloads_eval.md` §3.1 已算過：輸入中位 124,018 token、輪間隔 p50 8.8 s、p99 1,782 s、provider 命中 95.7%）。
- **B. Liu, H. Qiu, Í. Goiri, R. Fonseca, R. Bianchini, E. Choukse, "Agentic Coding in the Wild: Characterizing GitHub Copilot Traces at Production Scale," arXiv 2608.00101（2026-07-30；已開）**。〔部分〕（WebFetch 讀 HTML v1 前 100K 字元）
  - 規模：2026 年 6 月抽樣，320 萬使用者、1,300 萬 session、7.61 億次 LLM 呼叫、95T token〔摘要〕。
  - **KV 閒置時間**（同一 session 兩次 LLM 呼叫之間）：輪內 P50 1.2 s／P95 37 s；跨輪 P50 172 s／P95 4,500 s；使用者兩輪之間中位閒置 1,512 s（25.2 分），尾巴超過一天〔原文 Table 8、§9.1；原文內文另寫過不同的中位數，作者表與文不一致〕。
  - 每次呼叫 prompt 中位 68K token、其中命中 63K；輸出中位 247 token〔原文〕。
  - 命中率：輪內平均約 90%、跨輪邊界掉到 55%、換模型掉到 8%；**閒置 <約 2 分鐘命中率穩定，2–10 分鐘急降**——作者稱這是時間型淘汰（TTL）的特徵〔原文〕。
  - 作者建議：輪內留 GPU；輪邊界降級並卸到 DRAM 或磁碟；用預測的閒置時間決定何時卸；考慮在 300 s 保留期前送 keep-alive；閒置預測器對 >60 s 的 ROC-AUC 0.73、抓到 86–90% 的總閒置時間〔原文〕。
  - **資料**：作者寫「plan to release sanitized traces … soon」；截至 2026-10-19 **NOT_FOUND**。
- **C. Chang *et al.*, "From LLM Inference to Agentic Workloads: Characterization and Implications for Serving Systems," arXiv 2608.15127（2026-08-15；已開）**。〔摘要〕10 個 agent 應用（AgentSysBench）；5／10 個應用由非 LLM 元件主導延遲；生產 session 在活動步驟之間閒置數分鐘到數小時；狀態卸載省 4.6 倍記憶體。資料是否釋出：NOT_FOUND。

### H12 商業 prompt caching 的 TTL 與定價（2026-10-19 抓官方文件）

| 供應商 | 存活時間 | 命中是否續命 | 價格倍率（相對基本輸入價） | 最短可快取長度 | 隔離 |
|:--|:--|:--|:--|:--|:--|
| **Anthropic**（<https://platform.claude.com/docs/en/build-with-claude/prompt-caching>） | 預設 5 分鐘；可選 1 小時（加價） | **命中即免費續命**；壽命從**請求開始**算，生成時間也算 | 寫入：5 分鐘 1.25 倍、1 小時 2 倍；讀取 0.1 倍（Claude Opus 5.5、Sonnet 5.5 為 0.05 倍） | 依模型 512–4,096 token（例：Opus 5.5／Sonnet 5.5 等 512；Opus 4.6／Haiku 4.5 為 4,096） | 組織間絕不共享；Claude API 上再以 workspace 隔離（Bedrock、Google Cloud 只到組織） |
| **OpenAI**（<https://developers.openai.com/api/docs/guides/prompt-caching>） | GPT-5.6 以後：`prompt_cache_options.ttl` 只支援也預設 `30m`（最後一次寫入或重用後至少 30 分鐘）；更早的模型：`in_memory` 約閒置 5–10 分鐘、最長 1 小時；`24h` 延長保留（gpt-5.5、5.4、5.2、5.1、5、gpt-4.1 等；非 ZDR 組織預設 24h） | 以「最後一次寫入或重用」起算 | GPT-5.6 以後：寫入 1.25 倍、讀取 0.1 倍；更早的模型：快取輸入另有折扣價、沒有寫入加價；折扣最多 95% | GPT-5.6 以後 1,024 token | 不跨組織、不跨區域處理邊界；以前綴 hash＋機器負載路由；超過約 15 RPM 可能溢出到別台 |

- 〔判讀〕兩家都把「保留多久」做成**明碼標價的契約**：Anthropic 的 1 小時寫入是 5 分鐘的 1.6 倍（2 ÷ 1.25）；Anthropic 文件自己算出 5 分鐘 TTL 只要 2 次請求就回本、1 小時要 3 次〔claude-api skill 內附文件快照〕。Copilot 論文看到的「2–10 分鐘命中急降」與 5 分鐘 TTL 吻合。這給缺口 1 一個**外部錨點**：生產端的保留決策是「TTL × 價格」，而不是 κ。repo 方向 20（TTL 契約與 keepalive）已經記了這條線。

---

## §5 研究缺口

每個缺口寫：有沒有人做過（附出處）、可能的新意、最小證偽實驗、何時應停止投入。

### 缺口 1：agent 工具暫停（秒到分鐘）時，在 GPU／CPU／SSD／丟棄之間，用**實測、依硬體而變的 κ** 做決定

- **有沒有人做過**
  - 三選一但沒有 SSD、κ 離線量一次、短 context：InferCept（ICML'24）〔原文 §3–4〕。
  - 只決定留 GPU 多久，重算 vs 載入是設定不是決策：Continuum〔原文 §4.1〕。
  - 卸不卸只比 T_fc 和搬運時間，重算不是選項：Tokencake〔原文 Alg.1〕；KVFlow 明文假設搬運遠比重算快〔原文 §2〕。
  - DRAM／SSD 但對象是真人聊天：Bidaw〔repo 卡〕；HBM／HBF 依壽命：Lachesis（模擬）〔repo 卡〕。
  - 「依延遲預算決定從深層還原／留／重算」：**Cascade**（arXiv 2608.06557）〔摘要〕——規則、硬體、有沒有 SSD、κ 是否實測**都未知**。
  - 「先量常數、再決定」的單請求版本：py-kvcache（打平前沿）〔原文 p.11、p.21〕、CacheBlend 載入控制器〔原文 p.8〕。
  - 「存 vs 算」的經濟面：Five-Minute Rule 重推（只比 DRAM 與 flash、不把重算當選項）〔repo 卡〕、BuyKV（N\*≈1）〔repo 卡〕。
  - **查不到的**：同時有 (a) 四個動作含 SSD 與丟棄、(b) 依位置的重算成本、(c) 跨硬體實測 κ、(d) agent 的工具時長分布、(e) 多請求併發的論文。
- **可能的新意**〔判讀〕：「同一條暫停規則在 NVIDIA 上選卸載、在 MI300X 上選重算（或反之）」——如果能用實測 κ 與公開 trace 的間隔分布證明**決策翻轉**，這是 Tokencake／KVFlow 的隱含假設（搬運永遠便宜）的反例，也延續 Tiara「κ 跨硬體變 32 倍」的主張。
- **最小證偽實驗**：X3（§7）。用 repo 已量的 MI300X 與 3090 還原成本（`results/m10_*`、M2 成本模型），加 TraceLab／Copilot（表 8 分位數）／Exgentic 的間隔分布，模擬 7 個策略（vLLM 預設、Continuum TTL、Tokencake 規則、InferCept 浪費最小、MORI 閒置排名、κ 感知四選一、知道未來的 oracle），指標是 P95 JCT 與每輪 TTFT goodput。
  - **證偽條件**：在兩個平台、兩個模型、所有容量設定下，κ 感知策略比最好的現成規則（Continuum 或 Tokencake 或 InferCept）**P95 JCT 好不到 5%** → 缺口不存在（或不值得）。
  - **翻轉條件**：至少一組（平台 × 長度帶 × 間隔帶）中，現成規則與 κ 感知策略選了不同動作，而且差 ≥10%。
- **何時停止**：(1) 讀完 Cascade 全文，發現它已經做了實測、多層、含重算的版本 → 改做「AMD 上的量測＋負面結果」；(2) X3 證偽條件成立；(3) 依 CLAUDE.md §1 規則 4：<5% 停、5–15% 問人。另外 repo 方向 9 已經因「排隊是大頭」判死過一次（`D8_wild_ideas.md`；Continuum 原文 p.5）——這次若排隊依然主導、κ 只影響載回那一小段，也應停。

### 缺口 2：前綴重疊 × KV 容量 × 併發的「復用不划算」邊界

- **有沒有人做過**
  - 單請求的二維前沿（前綴比例 × 長度）：py-kvcache〔原文 p.11、p.20〕。
  - 路由層的一維門檻：Preble（共享 > 非共享）〔原文 p.2〕、Autellix（<2,048 token 不追）〔原文 §4.3〕、Mooncake（遠端前綴不夠長就算）〔原文 p.11〕、llm-d P2P（約 8K）〔repo 卡〕。
  - 容量使准入翻轉：EfficientAgent〔repo 卡〕。
  - 併發使卸載變慢的**單點觀察**：vLLM #44294（L40S）〔原文 issue〕、Strata 的 delayed hit〔repo 卡〕。
  - **查不到的**：在真實引擎上，同時掃重疊、容量、併發，畫出「APC／卸載比不復用還差」的三維邊界；特別是 AMD 平台。py-kvcache 自己把「加入排隊時間」列為下一步〔原文 p.21〕。
- **可能的新意**〔判讀〕：併發把單請求前沿往哪裡推？兩個方向都可能：併發讓 GPU 更忙 → 重算變貴 → 復用更划算；或併發讓 PCIe／同一 block 的載入排隊 → 搬運變貴 → 復用更不划算。哪個主導、在 MI300X（192 GB HBM、ROCm 每次 copy 固定成本）上是否不同，沒有人量過。
- **最小證偽實驗**：X1＋X2（§7）。
  - **證偽條件**：所有格子裡「APC＋CPU 卸載」都不比「不復用」差超過 5%（P95 TTFT），**而且**併發造成的邊界移動 <10%（和 C＝1 的單請求邊界比）→ 邊界就是 py-kvcache 的單請求前沿，沒有新東西。
- **何時停止**：X2 在 MI300X 上重現不了 #44294，**且** X1 的邊界在 C＝1 與 C＝64 之間沒移動 → 停。若只重現 #44294 而沒有更一般的邊界，結果只值一個工程 issue／PR，不是論文貢獻（和 repo F1 的判定同性質）。

### 缺口 3：以 goodput／P95 而不是單請求延遲為目標的 KV 放置

- **有沒有人做過**
  - 以 SLO 為目標、會拉遠端 KV 或改算：Mooncake（手調門檻＋早期拒絕）〔原文 p.2、p.11〕。
  - 以延遲預算決定還原／保留／重算、報 goodput：Cascade〔摘要〕。
  - 報 P90／P95 JCT 但目標是平均：Continuum〔原文 §6〕；TOPAS 報 p99 JCT〔摘要〕。
  - 以 goodput 定義做配置但不涉及 KV 分層：DistServe、SLOs-Serve〔摘要〕。
  - 多層放置報 P99 TTFT：Where Should the KV Cache Live?（arXiv 2609.16215，repo 卡；結論「prefetch 不划算」）。
  - **查不到的**：「目標函數換成 P95／goodput 之後，最佳的層級放置會不會和以平均延遲為目標時**不同**」的系統性比較。
- **可能的新意**〔判讀〕：平均延遲獎勵「多數請求快一點」，P95 懲罰「少數請求等很久」。卸載到慢層通常讓多數請求變快（命中）但製造長尾（delayed hit、排隊）——兩個目標可能選出相反的設定。若能在真實引擎上量到「平均 TTFT 最佳的設定 ≠ goodput 最佳的設定」，就說明單請求延遲的放置研究（包括 Tiara 第一、二階段）需要重新評估。
- **最小證偽實驗**：X1 的每一格同時算平均 TTFT 與 `--goodput ttft:<SLO>`；比較三個設定（不復用、APC、APC＋卸載）在兩個目標下的排名。
  - **證偽條件**：所有格子兩個目標的最佳設定都相同（或差 <3%）→ 缺口 3 是虛的，goodput 只是換個報法。
- **何時停止**：X1 證偽條件成立；或 Cascade 全文已經做了這個比較。

### 缺口 4：多請求容量競爭的公平性（誰佔 HBM、誰被趕到 CPU、誰被丟）

- **有沒有人做過**
  - 計算服務量的公平：VTC（OSDI'24）、DLPM（前綴只計一次）〔原文〕；token 延遲的公平：FairInference〔摘要〕；公平與 swap 開銷：FastSwitch〔摘要〕。
  - 快取類別公平（合成負載、無真實引擎）：arXiv 2608.14561〔摘要〕。
  - 隔離（安全）：`cache_salt`、KVGov、供應商的組織／workspace 隔離〔原始碼、摘要、官方文件〕。
  - **查不到的**：以「各租戶在各層的 KV 佔用 × 時間」定義公平，並量共享 LRU 池對低重用租戶 P95 的影響；以及 salt 隔離造成的容量重複對尾延遲的影響。
- **可能的新意**〔判讀〕：LRU＋前綴命中天生偏袒高重用租戶（它們的 block 一直被「最近使用」）；在 192 GB 的 MI300X 上這可能根本不成問題（容量夠），在 24 GB 的 3090 上可能很嚴重——又是一個「依硬體而變」的結論。
- **最小證偽實驗**：X4（§7）。
  - **證偽條件**：兩租戶情境下，低重用租戶在共享池中的 P95 TTFT 和「它獨佔同容量」相比，比值 <1.2 → 在本規模下沒有公平問題。
- **何時停止**：證偽條件在 MI300X 與（若有時間）3090 都成立；或 FairInference 全文已經涵蓋 KV 佔用。注意 CLAUDE.md §8：「不要在平台 A 做多租戶／機會成本結論」——**這個缺口只能在 MI300X 上做**。

---

## §6 可以在本機直接拿來產生「可控前綴重疊」與「多輪 agent」工作負載的公開資料／產生器

**單位驗證提醒（CLAUDE.md §1 規則 6）**：每個 trace 匯入時要寫一條「用 A 欄驗算 B 欄」的斷言。下表「單位風險」欄列出需要驗的地方。

### 6.1 產生器（不用下載大資料）

| 名稱 | 位置／連結 | 授權 | 能控制什麼 | 單位風險／備註 |
|:--|:--|:--|:--|:--|
| vLLM `bench serve --dataset-name prefix_repetition` | 本機 0.19.1 `vllm/benchmarks/datasets.py`（`PrefixRepetitionRandomDataset`，預設 prefix 256、suffix 256、10 個前綴、輸出 128） | Apache-2.0（vLLM） | `--prefix-repetition-prefix-len`、`-suffix-len`、`-num-prefixes`、`-output-len` → 重疊比例與工作集大小 | 長度單位是 token（隨機 token） |
| vLLM `bench serve --dataset-name random --random-prefix-len` | 同上 | Apache-2.0 | 所有請求共享同一段前綴＋隨機後綴 | — |
| vLLM `bench serve` 到達與指標 | 同上 `benchmarks/serve.py` | Apache-2.0 | `--request-rate`、`--burstiness`、`--max-concurrency`、`--ramp-up-strategy`、`--goodput`、`--percentile-metrics`、`--metric-percentiles` | 0.19.1 已有〔原始碼〕 |
| vLLM `benchmarks/multi_turn/benchmark_serving_multi_turn.py` | 本機 `src/vllm/benchmarks/multi_turn/`；設定檔 `generate_multi_turn.json` | Apache-2.0 | 對話數、輪數分布、**共同前綴長度**（`common_prefix_num_tokens`）、每段對話前綴分布、每輪輸入／輸出長度；`--num-clients`、`--max-active-conversations`、每 client 的 Poisson `--request-rate`（等於思考時間） | 要一個文字檔（README 用 Gutenberg pg1184，小檔）；README 本身就叫「Benchmark KV Cache Offloading with Multi-Turn Conversations」 |
| SGLang `bench_serving --dataset-name generated-shared-prefix` | <https://docs.sglang.io/developer_guide/bench_serving.html> | Apache-2.0 | `--gsp-num-groups`、`--gsp-prompts-per-group`、`--gsp-system-prompt-len`、`--gsp-question-len`、`--gsp-output-len`、`--gsp-group-distribution {uniform,zipf}`、`--gsp-zipf-alpha` | 本機沒裝 SGLang；能否對 vLLM 後端跑：未驗證 |
| SGLang `bench_serving --dataset-name agentic-trace`／`mooncake` | 同上 | Apache-2.0 | agentic-trace：逐輪重播、把伺服器回覆接回歷史（需 `sglang-oai-chat` 後端）；mooncake：`--mooncake-workload conversation|mooncake|agent|synthetic`、`--use-trace-timestamps` | 文件寫 mooncake 只支援 sglang 後端 |
| LMCache `benchmarks/multi_round_qa/multi_round_qa.py` | <https://github.com/LMCache/LMCache/tree/dev/benchmarks/multi_round_qa> | Apache-2.0（repo） | `--num-users`、`--num-rounds`、`--qps`、`--shared-system-prompt`、`--user-history-prompt`、`--answer-len`、`--time` | 打任何 OpenAI 相容端點（README 用 `vllm serve`） |
| KV-Repurpose 的負載產生器 | <https://github.com/taiga333/eval-kv-repurpose> | Apache-2.0 | Zipf RAG 式、固定比例共享前綴 | vLLM 版本未查；2026-10 新 |
| SCBench | repo `workloads_eval.md` §3.3 | MIT | multi-turn／multi-request 兩種共享模式 | 重用是構造的；repo D4 判定不適合做容量模擬 |

### 6.2 有時間戳或間隔的 trace（2026 新增的標 ★；其餘 repo 已盤點過）

| 名稱 | 連結 | 授權 | 大小 | 有什麼 | 單位風險 |
|:--|:--|:--|:--|:--|:--|
| ★ SemiAnalysis `cc-traces-weka-062126`（AgentX／InferenceX） | <https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126> | apache-2.0（HF 標籤） | 393 列（session）、98,827 次請求（56,798 主 agent、1,697 sub-agent 群）、1.85 GB | 每請求 `t`（時間）、`in`、`out`、`hash_ids`、`api_time`；`block_size`＝64；從 Claude Code proxy 匿名化，保留前綴重用型態 | **`in` 是由 hash block 數 ×64 推出來的**（HF 頁說會高估，>990,016 的已刪）——`in` 與 `hash_ids` 不是獨立欄位，**不能互相驗算**；要找第三個欄位（例如 `api_time` 與長度的關係）做斷言 |
| ★ Exgentic `agent-llm-traces` | <https://huggingface.co/datasets/Exgentic/agent-llm-traces> | cdla-permissive-2.0 | 1,781 列、984 MB | OpenTelemetry span：開始／結束時間、輸入／輸出 token、完整訊息與工具定義、session_id；harness 含 claude_code、smolagents 等；benchmark 含 SWE-bench、AppWorld、BrowseCompPlus、TAU-Bench；模型含 DeepSeek-V3.2、Kimi-K2.5、claude-opus-4-5、gemini-3-pro-preview、gpt-5.2；collected_at 約 2026-05 | 沒有 hash；重用要自己用內容算；工具時長要由 span 時間差推（**驗：span 長度 ≥ 0、同 session 內不重疊**） |
| ★ GitHub Copilot 生產 trace（arXiv 2608.00101） | — | — | — | — | **NOT_FOUND**（作者「plan to release」） |
| TraceLab | <https://github.com/uw-syfi/TraceLab> | 資料 CC BY 4.0、程式 Apache-2.0 | 357,161 次呼叫、4,265 session | 逐事件 ms 時間戳、prefix／new 切分；沒有內容 | repo 已驗：`prefix+new=input` 100% 成立 |
| Inferact `codex_swebenchpro_traces` | <https://huggingface.co/datasets/Inferact/codex_swebenchpro_traces> | MIT | 218 MB；610 trial、20,230 次呼叫 | 有內容；README 給呼叫間隔統計（平均 10.5 s、p50 5.2 s、p99 81.4 s） | 檔內有無逐呼叫時間戳：repo 記「未查證」 |
| MLPerf Agentic Inference | <https://github.com/mlcommons/endpoints/tree/main/examples/10_Agentic_Inference> | Apache-2.0 等 | 613 條軌跡 | 每則訊息 `delay_seconds`；有內容 | 長度是 repo 以字元÷4 估的 |
| Mooncake FAST'25（含 toolagent） | <https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release> | Apache-2.0（repo） | conversation 12,031、toolagent 23,608 列 | ms 時間戳、**512-token** block hash | repo 已有斷言 `ceil(input/512)=len(hash_ids)`（規則 6 的起源） |
| Alibaba Qwen-Bailian（ATC'25） | <https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon> | Apache-2.0 | 四條 trace（含 coder、thinking） | 秒時間戳、**16-token** SipHash、`chat_id`／`parent_chat_id`／`turn` | 最後一個 block 可能有 padding |
| BurstGPT | <https://github.com/HPMLL/BurstGPT> | CC-BY-4.0 | BurstGPT_3：534 萬列 | 只有長度；v2 有 Session ID 與經過時間 | 無法算重用 |
| Azure LLM Inference 2023／2024 | <https://github.com/Azure/AzurePublicDataset> | CC-BY-4.0 | — | 只有長度＋時間戳 | 2024 的 conversation trace 已於 2024-09-28 從 repo 撤下（搜尋所得的 commit 訊息，未開 commit 核對） |
| CC-Bench trajectories、SWE-bench 系軌跡（nebius、SWE-smith、OpenHands-SFT） | repo `workloads_eval.md` §3.4 | MIT／CC-BY-4.0 | 1 萬–10 萬筆級 | 有內容；多數沒有時間 | 時間要另外假設 |
| WildChat-1M／4.8M | repo `workloads_eval.md` §3.2 | ODC-BY | 84 萬段 | 逐則 assistant 時間、`hashed_ip` 串使用者 | 時間是回應完成時間 |

- **Qwen／Kimi**：Qwen 系的就是 Bailian trace；Kimi 系的就是 Mooncake trace。**2026 年 Kimi 或 Qwen 官方的 agent trace（帶 hash）NOT_FOUND**（搜尋只找到 SemiAnalysis 的 proxy 資料集與 AMD×Moonshot 的部落格，後者沒說釋出資料）。
- 〔判讀〕要做「工具暫停期間 KV 怎麼辦」，最實用的組合是：**Exgentic（有真實工具時長、有內容、小）＋ SemiAnalysis（有時間戳與 hash、Claude Code、長 context）＋ TraceLab（規模大、有間隔）**；Copilot 的表 8 分位數可以當參數化分布的錨點。

---

## §7 期限前（約 3 天、2×MI300X、Llama-3.1-8B／Qwen2.5-7B、vLLM 0.19.1+rocm）可測的最小實驗候選

**前提**：
- 機器目前在計時量測；以下實驗要排在佇列之後，**每個時間數字都要包 `GpuWatcher`／`gpu_guard`**（CLAUDE.md §3），走 `runsh` 記錄殼（§4.1）。
- 判準（下面的「證偽條件」與「停止」）要在跑之前寫進對應的設計檔並記 `date -u`，照 repo D1／D4 的做法。
- 我沒有跑任何一個；所有「預估時間」是**算術估計**，不是量測。

### 7.0 vLLM 0.19.1（ROCm）上能直接用的功能（全部由本機原始碼確認存在；ROCm 上的行為除註明者外 NOT_MEASURED）

| 功能 | 怎麼開 | 證據 | ROCm／MI300X 注意事項 |
|:--|:--|:--|:--|
| APC（GPU 前綴快取） | 預設開；關：`--no-enable-prefix-caching` | `config/cache.py:66` | 硬體無關 |
| `cache_salt` | Chat Completions 請求欄位 `cache_salt` | `entrypoints/openai/chat_completion/protocol.py:323` | 硬體無關 |
| 前綴命中指標 | `/metrics`：`vllm:prefix_cache_queries`／`_hits`、`vllm:external_prefix_cache_queries`、`vllm:kv_cache_usage_perc`、搶佔數 | `v1/metrics/loggers.py:133-538` | — |
| 原生 CPU 卸載 | `--kv-offloading-size <GiB>`（`--kv-offloading-backend native|lmcache`），或 `OffloadingConnector` | `config/cache.py:144-153`、`config/vllm.py:645-670` | **只有 GPU↔CPU，沒有 SSD 層**（0.28 才有 fs 層）〔repo `H0` §1〕；**命中正在載入的 block 會延後請求**（F4）；repo F1 在 0.28 看到 ROCm 每次 copy 有固定成本；F1 說 0.19.1 **沒有**「有 connector 就拿掉 ROCM_ATTN」的邏輯 |
| `SimpleCPUOffloadConnector` | `--kv-transfer-config` | `kv_connector/factory.py` | lazy 模式有 GPU 端水位線〔repo `H0`〕 |
| `LMCacheConnectorV1`／`LMCacheMPConnector` | `--kv-transfer-config` | 同上 | 需另裝 LMCache：官方 ROCm wheel v0.5.3rc2-rocm 是 **torch 2.11**、cp312、gfx942，ABI 對應上游 `vllm/vllm-openai-rocm` image；我們是 **torch 2.10.0+rocm7.0** → 要從原始碼建（`BUILD_WITH_HIP=1`、`PYTORCH_ROCM_ARCH=gfx942`）；能否與 0.19.1 相容：NOT_MEASURED。**期限內不建議** |
| 容量控制 | `--num-gpu-blocks-override`、`--kv-cache-memory-bytes`、`--gpu-memory-utilization` | `config/cache.py:60、135` | 用來把 192 GB 壓到「放不下工作集」 |
| 排程 | `--scheduling-policy fcfs|priority`＋請求欄位 `priority` | `config/scheduler.py:108`、`protocol.py:285` | X4 的對照組 |
| 搶佔 | 只有重算（V1） | `v1/core/sched/scheduler.py:956-968` | swap 對照組無法跑 |
| KV 事件 | `KVEventsConfig(enable_kv_cache_events=True, publisher="zmq")` | `config/kv_events.py:11-20` | 可拿來算各租戶佔用（X4） |
| 負載產生 | `vllm bench serve`（`prefix_repetition`、`random --random-prefix-len`、`--goodput`、`--max-concurrency`）；`benchmarks/multi_turn` | §6.1 | multi_turn 需要額外 Python 套件（`requirements.txt`）；**不要裝進 vLLM venv**，比照 CLAUDE.md 的 `pylibs --target` 做法；是否已有：未查 |
| P/D connector（Nixl、Mooncake、MoRIIO、P2pNccl） | — | `factory.py` | 不在本題範圍 |

### X1：復用邊界圖（重疊 × 容量 × 併發），兼測缺口 3 的「目標翻轉」

- **假說**：存在格子使「APC」或「APC＋CPU 卸載」的 P95 TTFT 比「不復用」差 ≥10%；預期在**短 L、低重疊、高併發、開卸載**的角落。且至少一格「平均 TTFT 最佳的設定 ≠ goodput 最佳的設定」。
- **對照**：(A) `--no-enable-prefix-caching`；(B) APC；(C) APC＋`--kv-offloading-size`（例如工作集的 50%）。
- **負載**：`vllm bench serve --dataset-name prefix_repetition`；總長 L＝prefix＋suffix ∈ {4K, 16K, 64K}；重疊 r＝prefix/L ∈ {0.25, 0.75, 0.95}；不同前綴數 K ∈ {1, 64}（控制工作集 vs 容量）；`--max-concurrency` C ∈ {1, 16, 64}；容量用 `--num-gpu-blocks-override` 壓到工作集的 25%／100%（兩檔）。
- **指標**：TTFT P50／P95、`--goodput ttft:<SLO>`（SLO 事先寫死，例如 C＝1、不復用時 P50 的 3 倍）、`vllm:prefix_cache_hits/queries`、外部命中、搶佔數；每格 3 個 seed。
- **規模與時間（算術估計）**：先跑子集：L 3 × r 3 × C 3 × K 1（K＝64、容量 25%）× 3 設定 = 81 runs；每 run 約 2 分鐘（含暖機）＋每個設定重啟伺服器 → 約 3 小時／張卡，兩張卡平行約 1.5–2 小時；完整版（加 K＝1、容量 100%）再 ×2–4。
- **證偽**：所有格子 (C) 與 (B) 都不比 (A) 差 >5%，而且兩個目標的最佳設定全部相同 → 缺口 2、3 在本規模不成立。
- **風險**：`prefix_repetition` 用隨機 token，前綴內容不影響計算量，但會讓 attention 分佈不真實（只影響品質，不影響時間）。

### X2：MI300X 上重現「命中但正在載入 → 延後」（vLLM #44294）

- **假說**：在 0.19.1＋ROCm，多個併發請求共用一段已被卸到 CPU 的長前綴時，開卸載的 TTFT 比不開還差（#44294 在 L40S 是 11.46 s vs 5.95 s）。
- **對照**：(A) APC、不卸載；(B) APC＋原生 CPU 卸載；(C) 不復用。
- **負載**：先用一個請求寫入前綴 P ∈ {10K, 32K}，再用填充請求把它擠出 GPU（或用 `--num-gpu-blocks-override` 讓 GPU 放不下），然後同時送 N ∈ {1, 8, 32, 64} 個共用該前綴的請求（`--random-prefix-len P --max-concurrency N --request-rate inf`）。
- **指標**：TTFT P50／P95、吞吐、debug log 裡「Delaying request … being loaded」的次數（需把 connector 的 logger 調到 DEBUG）。
- **時間**：約半天（含寫腳本）。
- **證偽**：所有 N 下 (B) ≤ (A) → 0.19.1 在 MI300X 上不受影響，此路停。
- **若成立**：最小改法是「命中的 block 正在載入時，改成重算或等同一個載入完成，而不是延後到下一輪排程」——這正是 κ 決策在**併發**下的版本（等載入 vs 重算）。但要注意：這是工程修正，論文價值要看它能不能推廣成缺口 2 的邊界。

### X3：工具暫停的 κ 感知四選一（先模擬，再一個 GPU 驗證點）

- **假說**：在「間隔介於搬運時間與重算時間之間」且「HBM 有壓力」的帶，κ 感知（GPU／CPU／SSD／丟棄）比最好的現成規則（Continuum TTL、Tokencake、InferCept）P95 JCT 好 ≥10%；在 MI300X 與 3090 上被選中的動作不同（決策翻轉）。
- **輸入**：
  - 成本：repo 已量的還原／重算成本（`results/m10_*_mi300x`、平台 A 的 M2 成本模型；MI300X H2D 50.8／D2H 48.6 GB/s〔repo `MI300X_MLSTEAM.md`，`gpu_smoke.py` 實測〕）。**不新增量測**。
  - 間隔：Exgentic span 時間差（真實工具）、SemiAnalysis 的 `t`（Claude Code）、TraceLab 的輪間隔、Copilot 表 8 分位數（參數化）。
  - 併發：每平台 C ∈ {1, 8, 32} 個同時活著的 agent。
- **對手**：vLLM 預設（APC LRU）、Continuum（τ\* 公式，K＝100 冷啟動）、Tokencake（T_fc vs T_transfer）、InferCept（三式浪費最小）、MORI（閒置排名）、κ 感知四選一、oracle（知道下一次回來的時間）。
- **指標**：JCT 平均與 P95、每輪 TTFT 的 goodput、HBM 佔用秒數、SSD 寫入量。
- **GPU 驗證點**：用 `benchmarks/multi_turn`（每 client Poisson `--request-rate` 當思考時間）跑一組 C 與間隔，比較「keep（APC）」「CPU 卸載」「discard（APC 關）」三個能原生跑的動作，檢查模擬器的排名是否一致（偏差容忍：排名一致、差距 ±30%）。
- **時間**：模擬 1 天（可重用 repo `m8_conc_sim.py` 的併發骨架與 `restore_v2`）；GPU 驗證半天。
- **證偽**：所有設定下 κ 感知比最好的現成規則好 <5%，或兩平台選的動作完全相同 → 缺口 1 不成立（照 CLAUDE.md 規則 4 停）。
- **主要風險**：repo D1／D4 已經在相近模型上得出「延後版追得上」的負面結果；若排隊主導（Continuum 的觀察），κ 的影響會被稀釋。這是要**先寫進判準**的已知風險。

### X4：兩租戶容量競爭的公平性（只在 MI300X 做）

- **假說**：共享 LRU 池中，高重用租戶 A（一段 32K 共享前綴）會擠掉低重用租戶 B（各自獨立的 32K context）的 KV，使 B 的 P95 TTFT 比「B 獨佔同容量」高 ≥1.5 倍；`cache_salt` 不能修（它是隔離不是配額）。
- **對照**：(1) 共享池 APC LRU；(2) A、B 各用不同 `cache_salt`；(3) B 設較高 `priority`；(4) B 單獨跑（基線）。
- **設定**：容量用 `--num-gpu-blocks-override` 壓到兩租戶工作集的約 50%；A、B 各自用 `--request-rate` 控制到達；`KVEventsConfig` 開 ZMQ 收 block 事件，算各租戶佔用。
- **指標**：各租戶 P95 TTFT、命中率、佔用 block 數 × 時間。
- **時間**：半天（含寫一個極小的 ZMQ 收集器）。
- **證偽**：B 的 P95 比值 <1.2 → 本規模沒有公平問題，缺口 4 停。

### 3 天排程建議（先做訊號最強、成本最低的）

| 天 | 卡 0 | 卡 1 | 決策點 |
|:--|:--|:--|:--|
| 1 | X2（半天）→ X1 子集（L＝4K、16K） | X1 子集（L＝64K） | X2 不成立且 X1 子集無「不划算」格 → 缺口 2 停 |
| 2 | X1 補格（容量 100%、K＝1） | X4 | 缺口 3 看 X1 的目標翻轉；缺口 4 看 X4 |
| 3 | X3 GPU 驗證點 | （CPU）X3 模擬 | 讀 Cascade 全文；依 X3 決定缺口 1 去留 |

---

## §8 參考清單

「已開」＝我用 WebFetch 打開 arXiv abs 頁（或官方頁面）確認存在與標題；venue 標「已確認」者另開了會議頁或 proceedings 頁，否則照 arXiv Comments 欄或寫「未查證」。

**E：prefix caching／復用**
1. W. Kwon *et al.*, "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023. arXiv:2309.06180（已開）. https://arxiv.org/abs/2309.06180
2. vLLM, "Automatic Prefix Caching" 設計文件. https://docs.vllm.ai/en/latest/design/prefix_caching.html（已開）
3. L. Zheng *et al.*, "SGLang: Efficient Execution of Structured Language Model Programs," NeurIPS 2024（已確認）. arXiv:2312.07104（已開）
4. V. Srivatsa, Z. He, R. Abhyankar, D. Li, Y. Zhang, "Preble: Efficient Distributed Prompt Scheduling for LLM Serving," ICLR 2025（已確認）. arXiv:2407.00023（已開）
5. R. Pan *et al.*, "Marconi: Prefix Caching for the Era of Hybrid LLMs," MLSys 2025. arXiv:2411.19379（本機 PDF）
6. J. Yao *et al.*, "CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion," EuroSys 2025（Best Paper，依大學新聞）. arXiv:2405.16444（已開）
7. J. Hu *et al.*, "EPIC: Efficient Position-Independent Caching for Serving Large Language Models," ICML 2025（PMLR v267，已確認）. arXiv:2410.15332（已開）
8. J. Yang, B. Hou, W. Wei, Y. Bao, S. Chang, "KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse," NeurIPS 2025（已確認）. arXiv:2502.16002（已開）
9. X. Yang, T. Chen, B. Chen, "APE: Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding," ICLR 2025. arXiv:2502.05431（已開）
10. D. Ma, Y. Wang, L. Tian, "Block-Attention for Efficient Prefilling," ICLR 2025. arXiv:2409.15355（已開）
11. S. Lu, H. Wang, Y. Rong, Z. Chen, Y. Tang, "TurboRAG: Accelerating Retrieval-Augmented Generation with Precomputed KV Caches for Chunked Text," EMNLP 2025（ACL Anthology 2025.emnlp-main.334）. arXiv:2410.07590（已開）
12. "MEPIC: Memory Efficient Position Independent Caching for LLM Serving," arXiv:2512.16822（搜尋結果確認 ID 與標題；作者未逐一核對）
13. N. Ordonez, T. Parnell, "MiniPIC: Flexible Position-Independent Caching in <100LOC," arXiv:2606.13126（已開）
14. T. Ikeda, D. Miyashita, J. Deguchi, "KV-Repurpose: A Testbed for Evaluating Non-Prefix KV-Cache Reuse," arXiv:2610.13060（已開）
15. H. Kang, C. D. Yoo, "Budgeted Cache Repair for Cross-Context KV-Cache Reuse," arXiv:2610.02233（已開）
16. J. Kanichai, T. De Matteis, A. Trivedi, "Building py-kvcache: A Performance Characterization of External KV Caching for vLLM with NVMe SSDs," arXiv:2609.11744（本機 PDF）
17. L. Song *et al.*, "The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems," IEEE TIFS（Comments 欄）. arXiv:2409.20002（已開）
18. C. Gu, X. L. Li, R. Kuditipudi, P. Liang, T. Hashimoto, "Auditing Prompt Caching in Language Model APIs," ICML 2025. arXiv:2502.07776（已開）
19. T. C. Addagada, "Governing the KV Cache: Preventing Timing Side-Channel Leakage in Multi-Tenant LLM Inference," arXiv:2608.09225（已開）

**F：paging／記憶體壓力**
20. R. Prabhu, A. Nayak, J. Mohan, R. Ramjee, A. Panwar, "vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention," ASPLOS 2025. arXiv:2405.04437（已開）
21. C. Zhang *et al.*, "JENGA: Effective Memory Management for Serving LLM with Heterogeneity," SOSP 2025（依 repo 卡片）. arXiv:2503.18292
22. vLLM issue #44294, "[Bug][OffloadingConnector] _blocks_being_loaded serialises concurrent requests through a single load …," 2026-06-02（已開）. https://github.com/vllm-project/vllm/issues/44294
23. B. Sun *et al.*, "Llumnix: Dynamic Scheduling for Large Language Model Serving," OSDI 2024. arXiv:2406.03243（已開）
24. A. Shen, Z. Li, M. Gao, "FastSwitch: Optimizing Context Switching Efficiency in Fairness-aware Large Language Model Serving," arXiv:2411.18424（已開）
25. R. Cheng, Y. Lai, X. Wei, R. Chen, H. Chen, "KunServe: Parameter-centric Memory Management for Efficient Memory Overloading Handling in LLM Serving," arXiv:2412.18169（已開）
26. R. Li *et al.*, "Oneiros: KV Cache Optimization through Parameter Remapping for Multi-tenant LLM Serving," arXiv:2507.11507（已開）
27. J. Cheng, D. T. Do, D. T. Nguyen, "Robust KV Cache Management for LLM Serving under Output Token Length Uncertainty," arXiv:2607.16892（已開）

**G：goodput／尾延遲／路由／公平**
28. Y. Zhong *et al.*, "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving," OSDI 2024. arXiv:2401.09670（已開）
29. A. Agrawal *et al.*, "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve," OSDI 2024（已確認）. arXiv:2403.02310（已開）
30. R. Qin *et al.*, "Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot," FAST 2025. arXiv:2407.00079（本機 PDF）
31. M. Ayoub *et al.*, "KV-cache wins you can see," llm-d blog, 2025-09-24（已開）. https://llm-d.ai/blog/kvcache-wins-you-can-see
32. Y. Yuan *et al.*, "DualMap: Enabling Both Cache Affinity and Load Balancing for Distributed LLM Serving," arXiv:2602.06502（已開）
33. M. Adnan *et al.*, "Cascade: Exploiting SLO-Aware latency budget for fair and high goodput LLM inference serving," arXiv:2608.06557（已開）
34. S. Chen, Z. Jia, S. Khan, A. Krishnamurthy, P. B. Gibbons, "SLOs-Serve: Optimized Serving of Multi-SLO LLMs," arXiv:2504.08784（已開）
35. Y. Sheng *et al.*, "Fairness in Serving Large Language Models," OSDI 2024（已確認）. arXiv:2401.00588（已開）
36. S. Cao *et al.*, "Locality-aware Fair Scheduling in LLM Serving," arXiv:2501.14312（已開）
37. D. Bali *et al.*, "Token Latency Fairness: Performance Isolation for Multi-Tenant LLM Serving," arXiv:2609.18112（已開）
38. M. R. Flax, "A Biophysically-Inspired Feedback Controller for Multi-Class Cache Fairness," arXiv:2608.14561（已開）

**H：agent**
39. R. Abhyankar, Z. He, V. Srivatsa, H. Zhang, Y. Zhang, "InferCept: Efficient Intercept Support for Augmented Large Language Model Inference," ICML 2024（PMLR 235，已確認）. arXiv:2402.01869（已開）
40. C. Lin *et al.*, "Parrot: Efficient Serving of LLM-based Applications with Semantic Variable," OSDI 2024. arXiv:2405.19888（已開）
41. M. Luo *et al.*, "Autellix: An Efficient Serving Engine for LLM Agents as General Programs," arXiv:2502.13965（已開）；NSDI'26 預印版題為 "Agentix"（只見 PDF 檔名，未確認議程）
42. Z. Pan *et al.*, "KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows," NeurIPS 2025（已確認）. arXiv:2507.07400（已開）
43. H. Li *et al.*, "Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live," ICLR 2026 Workshop on Lifelong Agents（poster，已確認）. arXiv:2511.02230（本機 PDF）；程式 https://github.com/Hanchenli/vllm-continuum
44. Z. Bian, F. Wu, T. Ma, Y. Zhuo, "Tokencake: A KV-Cache-centric Serving Framework for LLM-based Multi-Agent Applications," arXiv:2510.18586（HTML v2 已開）；程式 https://github.com/pku-lemonade/TokenCake
45. T. Xia *et al.*, "Idleness is Relative: Exploiting Tool-Call Idle Windows for Offloading in Agentic Systems with MORI," arXiv:2606.00866（repo 卡）
46. K. Shao *et al.*, "EfficientAgent: What Makes KV Cache Offloading Work for Concurrent Agents?," arXiv:2609.33762（repo 卡）
47. J. Yang *et al.*, "Lachesis: Lifetime-Aware KV Cache Placement for Agent Serving across HBM and High-Bandwidth Flash," arXiv:2610.08378（repo 卡）
48. R. Zhang *et al.*, "Learning Agent Execution for KV-Cache Management in Agentic Serving," arXiv:2608.14624（已開）
49. H. Ni *et al.*, "TOPAS: Workflow-Aware Prefix-State Scheduling for Multi-Agent LLM Serving," arXiv:2608.25523（已開）
50. H. Zheng *et al.*, "Efficient Serving for Dynamic Agent Workflows with Prediction-based KV-Cache Management," arXiv:2605.06472（已開）
51. B. Ma, J. Eitzinger, H. Koestler, "Leyline: KV Cache Directives for Agentic Inference," arXiv:2606.01065（已開）
52. B. Liu, H. Qiu, Í. Goiri, R. Fonseca, R. Bianchini, E. Choukse, "Agentic Coding in the Wild: Characterizing GitHub Copilot Traces at Production Scale," arXiv:2608.00101（已開）
53. C. Chang *et al.*, "From LLM Inference to Agentic Workloads: Characterization and Implications for Serving Systems," arXiv:2608.15127（已開）
54. UW SyFI Lab, "TraceLab: Characterizing Coding Agent Workloads for LLM Serving," arXiv:2606.30560（repo 卡）
55. Anthropic, "Prompt caching" 官方文件（2026-10-19 抓）. https://platform.claude.com/docs/en/build-with-claude/prompt-caching
56. OpenAI, "Prompt caching" 官方文件（2026-10-19 抓）. https://developers.openai.com/api/docs/guides/prompt-caching

**資料與工具**
57. SemiAnalysis, `cc-traces-weka-062126`. https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126（已開）
58. Exgentic, `agent-llm-traces`. https://huggingface.co/datasets/Exgentic/agent-llm-traces（已開）
59. SGLang, "Bench Serving." https://docs.sglang.io/developer_guide/bench_serving.html（已開）
60. LMCache, "multi_round_qa" benchmark. https://github.com/LMCache/LMCache/tree/dev/benchmarks/multi_round_qa（已開）
61. LMCache 安裝（ROCm 節）. https://docs.lmcache.ai/dev/getting_started/installation.html（已開）；ROCm wheel v0.5.3rc2-rocm（newreleases.io 轉載頁已開，GitHub release 頁未直接開）

---

## 附錄：本次的不確定性與沒做的事

- **Cascade（#33）只讀摘要**，卻是缺口 1、3 最大的威脅。第三階段定題前必須讀全文。
- MORI 原文「短空檔搬運超過空檔」那句是搜尋摘錄，未回原文核對頁碼。
- Autellix 的 NSDI'26 接受只由檔名推得；Tokencake 的 EuroSys'27 只見於 GitHub repo 描述；Jenga 的 SOSP'25 依 repo 卡片（未開 ACM DL）。
- WebFetch 讀 HTML 頁（Tokencake、Copilot、llm-d、官方文件）時，內容經過工具內的小模型整理；本機 PDF（CacheBlend、Continuum、InferCept、KVFlow、Autellix、Preble、DLPM、Marconi、Mooncake、PagedAttention、py-kvcache、LMCache、CachedAttention）是我自己抽字後逐段讀的，可信度較高。
- NVIDIA Dynamo KV router、vAttention 的 ROCm 可移植性、LMCache 與 vLLM 0.19.1 的相容性、SGLang 在本機 ROCm 的可用性：都沒查（NOT_VERIFIED／NOT_MEASURED）。
- 本文是 self-review，不是 cross-model review。
