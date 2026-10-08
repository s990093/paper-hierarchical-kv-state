# E04　分層儲存系統的評測卡（Strata、Bidaw、MTDS、LMCache、Mooncake、CachedAttention、Tutti、KVDrive）

> **狀態**：抽取完成（E04，2026-10-06）；已獨立複核（V04，2026-10-07），修正見檔尾「複核紀錄」。
> **範圍**：多層 KV 儲存系統（GPU HBM／CPU DRAM／SSD／遠端）怎麼做實驗：分層怎麼實作、負載怎麼構造、在哪個工作點報主結果。
> **頁碼**：一律為 PDF 頁。USENIX 版（Strata、Bidaw、Mooncake FAST、CachedAttention ATC）的 PDF p.1 是封面，正文從 p.2 起；MTDS 的 PDF 頁＝期刊「Page X of 17」。
> **標記**：〔原文〕〔程式碼〕〔文件〕〔摘要〕〔二手〕〔計算〕〔判讀〕〔未查證〕，規則見 `../README.md`。

---

## 來源清單

| 代號 | 論文 | 讀的版本與 URL | 讀法 |
|:--|:--|:--|:--|
| [S] | Strata（OSDI'26） | USENIX 版 PDF，17 頁：https://www.usenix.org/system/files/osdi26-xie-zhiqiang.pdf | 全文 |
| [S-arX] | Strata | arXiv 2508.18572v1（2025-08-26，13 頁）：https://arxiv.org/abs/2508.18572 | 部分（與 OSDI 版逐節比對差異） |
| [B] | Bidaw（FAST'26） | USENIX 版 PDF，17 頁：https://www.usenix.org/system/files/fast26-hu-shipeng.pdf | 全文 |
| [M] | MTDS（Complex & Intelligent Systems 12:104, 2026） | 出版社 PDF，17 頁，CC BY-NC-ND，DOI https://doi.org/10.1007/s40747-025-02200-4 。Springer 對自動下載回應 JS challenge，改讀使用者已存的出版社 PDF（`<repo 根>/papers/mtds2026.pdf`；PDF metadata 為 Springer、DOI 相符；與 `~/Downloads/s40747-025-02200-4.pdf` MD5 相同） | 全文 |
| [L] | LMCache | arXiv 2510.09665v2（2025-12-05，19 頁）：https://arxiv.org/abs/2510.09665 | 全文 |
| [MC-F] | Mooncake（FAST'25） | USENIX 版 PDF，17 頁：https://www.usenix.org/system/files/fast25-qin.pdf | 全文 |
| [MC-A] | Mooncake | arXiv 2407.00079v4（2025-09-03，23 頁）：https://arxiv.org/abs/2407.00079 | 部分（§1–4、§6–8；§5 prefill pool 略讀） |
| [CA] | CachedAttention（ATC'24） | USENIX 版 PDF，17 頁：https://www.usenix.org/system/files/atc24-gao-bin-cost.pdf | 全文 |
| [CA-v1/v2] | 同上，arXiv 舊版（題名 AttentionStore） | arXiv 2403.19708v1、v2：https://arxiv.org/abs/2403.19708v1 | 只全文檢索「磁碟命中比例」相關句 |
| [T] | Tutti | arXiv 2605.03375v1（2026-05-05，14 頁）：https://arxiv.org/abs/2605.03375 | 全文 |
| [K] | KVDrive | arXiv 2605.18071v1（2026-05-18，25 頁）：https://arxiv.org/abs/2605.18071 | 全文 |
| [code-SGL] | SGLang | commit `542addad`（2026-10-06）：`python/sglang/srt/arg_groups/fields/memory.py` L106–143、`python/sglang/srt/mem_cache/unified_radix_cache.py` L501–503 | 只查 HiCache 參數與預設值 |
| [code-LMC] | LMCache | dev 分支 commit `8c77a6f7`（2026-10-05）：`lmcache/v1/config.py` L90、L96、L401–403 | 只查預設值 |
| [gh-T] | Tutti repo | https://github.com/xPU-IO/Tutti （GitHub API：2026-05-30 建立、Apache-2.0、最後 push 2026-10-06） | 只查 metadata |
| [HF] | 模型 config | `https://huggingface.co/<model>/resolve/main/config.json`（2026-10-06 取得；gated 的 Llama 用 NousResearch 鏡像） | 只看注意力頭數與視窗 |
| [Cake] | Cake（ICML'25） | `<repo 根>/papers/cake2025.pdf` | 只查它對 AttentionStore 的引用句（p.2） |

所有論文查證日期：2026-10-06。

---

## 本組的共同模式

1. **慢層常常沒被真的量到。** 端到端實驗真的讀 SSD 的只有 Bidaw（4 顆 SATA SSD RAID-5，1.5 GB/s）、CachedAttention（10 TB SSD）、Tutti（NVMe）、KVDrive（NVMe；主吞吐實驗是否用到 SSD 原文沒講）。Strata 的 SSD 只在一個 DeepSeek-V3 子實驗；Mooncake 兩個版本、LMCache 都沒有 SSD 實驗；MTDS 有 2 TB SSD，但只報 DRAM 命中率。頻寬掃描常是模擬：Bidaw 以「從 host 複製＋注入延遲」模擬 5 GB/s SSD；Mooncake 模擬 24–400 Gbps 網路；Strata-Oracle 模擬 CPU–GPU 無限頻寬；LMCache 的 32／64／128 Gbps 怎麼產生沒寫。〔原文；歸納為〔判讀〕〕
2. **寫入幾乎都是「全部寫」，只有 Strata 把寫入策略當成可切換的設定。** Strata 有 write-back／write-through／selective（存取次數超過門檻 2 才備份）三種；MTDS 依預測重用排卸載優先度並延後低優先者；Tutti 全寫但把寫入延到計算空檔；KVDrive 全寫 SSD 再依注意力重要度往上層放；CachedAttention、Bidaw（inclusive）、LMCache、Mooncake 都是全寫。**沒有一篇做「寫入策略 × 容量壓力」的消融。**〔判讀，依各卡〕
3. **逐出以 LRU 為預設，新策略只跟 LRU／FIFO／LFU 比，彼此不比。** Strata（各層預設）、Mooncake、LMCache（程式碼預設）用 LRU；Bidaw（回答長度＋Belady ghost cache）、CachedAttention（佇列 look-ahead）、MTDS（兩級門檻＋滑動視窗 LRU）、KVDrive（注意力 lookahead）各自發明，對照組都是 LRU／FIFO／LFU（Bidaw 另比 CachedAttention 的 queue-enhanced）。〔判讀，依各卡〕
4. **長 context 的跨請求重用是靠抽樣構造的。** 長 context 重用負載都是「少量長文件被多次查詢」：Strata 的 LooGLE／NarrativeQA、Tutti 的 LEval／LooGLE、LMCache 的多輪文件 QA、Mooncake FAST 的 Synthetic，以隨機抽樣或 round-robin 加 Poisson 排列（LMCache 多輪 QA 是「先 40 位使用者、再依 QPS 加入」，到達分布原文沒寫〔複核補充，[L] p.11〕）。主實驗有真實時間戳的只有 Mooncake（Conversation／Tool&Agent，平均 input 8.6K–12K）與 Bidaw（短對話）；另外 Strata 只在 delay-hit 模擬（Fig 12）縮放 Mooncake 時間戳，LMCache 的公司 F／G trace 把數天壓成 1 小時〔複核補充，[S] p.12–13、[L] p.12〕。Strata 是唯一把「重用距離」（min／shuffle／max cache distance）當自變數的。評測實際長度：CachedAttention 2K–4K 視窗；Bidaw 推定 ≤2K（OPT 與 Qwen-14B 視窗 2K；Qwen-7B 為 8K〔複核修正〕）；MTDS ≤24K；LMCache 端到端 10K–20K（微基準約 400K）；Mooncake 平均約 12K、最長 128K；Strata 平均最長約 55K；Tutti 3K–200K（擴展性到 640K）；KVDrive 60K–360K（無跨請求重用）。〔原文；歸納為〔判讀〕〕
5. **倍數的定義各不相同，最大值都在基線最吃力的工作點。**「同 TTFT 下的吞吐」（Strata、LMCache）；「相近延遲下的 users/min」（Bidaw）；「固定重播負載下 SLO 達成比例之比」（Mooncake FAST 59%–498%，隨 TBT 門檻變；arXiv 版的 75% 是 100%／57%）；「1 s TTFT SLO 下的有效請求率」（Tutti）；「單一負載點的 TTFT 降幅」（CachedAttention、MTDS）；「離線生成吞吐」（KVDrive）。〔原文；〔計算〕見 Mooncake 卡〕
6. **沒有重複與誤差棒；基線多是自己重做。** 8 篇都沒寫重複次數、誤差棒或信賴區間。Bidaw 在 vLLM 上重做 CachedAttention 與 FlashGen；Strata 的「SGLang-HiCache」基線是作者自建（page 32、cudaMemcpyAsync；原文稱之為 "state-of-the-art layer-wise ... implementation"，「弱化版」是〔判讀〕〔複核修正：原未標〕，[S] p.9）；KVDrive 在自己的框架重做全部 8 個基線；MTDS 以 5 GB buffer 跑 Mooncake／LMCache，部署方式沒寫。成本模型的單價也互相沿用：CachedAttention 與 Tutti 用同一組 AWS 價格（GPU $5/h、DRAM $0.0088/GB/h、SSD $0.000082/GB/h；CA 的 GPU 是 A100、Tutti 是 H100，單價相同〔複核補充，[CA] p.11、[T] p.12〕）。〔原文〕

---

## 逐篇評測卡

### Strata：Hierarchical Context Caching for Long Context Language Model Serving（OSDI'26；arXiv 2508.18572）

- **讀了什麼**：〔全文〕OSDI'26 USENIX 版；〔部分〕arXiv v1 比對差異；〔程式碼〕SGLang `542addad` 只查 HiCache 參數。查證 2026-10-06。
- **一句話**：長 context 載回 KV 卡在 I/O；用 GPU 協助搬運與載入感知排程解決。
- **評測要證明的主張**：(1) 小 page 讓 CPU→GPU 搬運只用到一小部分 PCIe 頻寬，排程器又忽略載入時間與 delay hit，長 context 因此變成 I/O-bound；(2) GPU 協助搬運＋GPU／host 版面解耦＋cache-aware 排程，能在長 context 吞吐上勝過 vLLM-LMCache（最高 5×）與 TRT-LLM（3.75×），且不傷短 context。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | Llama-3.1-8B-Instruct（128K）、Qwen2.5-14B-Instruct-1M（1M）、Llama-3.1-70B-Instruct（128K，4 卡 TP）〔原文〕。動機圖用 Qwen2.5-14B（Fig 1）、Mistral-24B（Fig 2）；磁碟子實驗用 DeepSeek-V3。注意力：三個主模型皆 GQA（8 個 KV head）〔文件〕；DeepSeek-V3 為 MLA〔文件〕〔複核補充：V04 重查 `deepseek-ai/DeepSeek-V3` config.json，`kv_lora_rank`=512，2026-10-07；原為〔二手〕〕。權重／KV dtype：未說明〔未查證〕 | [S] p.9 §5.1 Models；p.3 Fig 1；p.4 Fig 2；p.13 §5.3.5；[HF] |
| 硬體 | H200 平台：8×H200（NVLink）、Sapphire Rapids、1.6 TB DRAM，每卡 PCIe 5.0 x16（單向峰值 64 GB/s）；H20-storage：8×H20＋Intel P5510 NVMe（讀 7 GB/s）；GH200：1×H100＋Grace 64 核、464 GB LPDDR5X、CPU↔GPU 單向 384 GB/s。H200 平台的磁碟 <1 GiB/s，所以磁碟實驗改在 H20 上做〔原文〕 | [S] p.9 §5.1 Testbed；p.13 §5.3.5 |
| 軟體與版本 | SGLang v0.4.5（SGLang、SGLang-HiCache、Strata 三者共用）；vLLM v0.8.5＋LMCache v0.2.1（chunk 256、vLLM page 32）；TensorRT-LLM v0.17.0（page 32）。page：SGLang 與 Strata 為 1，SGLang-HiCache 為 32。SGLang 的 Python 排程器以 C++ 補強〔原文〕 | [S] p.9 Baselines；p.7 §4.3 |
| 資料／負載 | 表 1：LooGLE（只用 Wikipedia 部分）105 contexts／2,410 queries；NarrativeQA（濾掉 >128K 後抽 50 份文件）50／1,461；ReviewMT（多代理審稿對話）100／1,092；ShareGPT 200,869 queries。delay hit 分析另用 Mooncake Tool-Agent trace〔原文〕 | [S] p.9 Table 1、Datasets；p.12–13 §5.3.4 |
| 長度 | 平均 input／output（token）：LooGLE 21,613／15.60；NarrativeQA 54,797／13.00；ReviewMT 17,708／208.3；ShareGPT 680.9／260.9。單筆上限受 128K 視窗限制〔原文〕 | [S] p.9 Table 1 |
| 到達與併發 | 資料集沒有時間戳，以 Poisson 掃 request rate。ReviewMT／ShareGPT 保留輪次相依；ShareGPT 在回覆與下一問之間插 60 s thinking time（沿用 Pensieve）。長 context 資料集隨機抽 query。in-flight 上限 128（原文強調是併發上限，不是 batch 大小）。delay hit 模擬以縮放 Mooncake 時間戳得到 1／10／100 req/s〔原文〕 | [S] p.10 §5.1；p.13 Fig 12 圖說 |
| 重用結構 | 長文件被多個 query 反覆查（LooGLE、NarrativeQA，原文比作 RAG 問答）；多代理對話（ReviewMT）；多輪聊天（ShareGPT）。另由 LooGLE 生成三種排列：同一文件的請求排在一起（min cache distance）、原始隨機（shuffle）、均勻分散（max）〔原文〕 | [S] p.9–10；p.12 §5.3.3 |
| 分層實作 | **層**：GPU HBM → CPU DRAM（pinned，配置 1 TB；GH200 為 400 GB）→ 外部儲存（多數實驗不用）。GPU KV 池依各引擎預設；ShareGPT 時把 GPU 限制在約 500K token。<br>**頻寬**：全部是實機；只有 Strata-Oracle 模擬「CPU–GPU 無限頻寬」的 TTFT。<br>**寫入**：write-back（要被逐出時才備份）、write-through（每次產生都備份）、selective-write-through（**預設**；HiRadixTree 節點存取計數超過門檻才備份，寫入頻寬充足時門檻 2）。CPU→GPU 載入用 2 個 CUDA block，GPU→CPU 備份用 1 個。<br>**外部儲存**：命中時 best-effort 預取到 host（與排隊重疊），可改 wait-complete 或 timeout。<br>**逐出**：各層預設 LRU〔原文〕 | [S] p.10 §5.1；p.9 §4.4；p.6 §4.2；p.7 §4.2.1；p.13 §5.4 |
| 工作集／容量 | 原文沒有直接給比例；錨點是「40 GB HBM 只放得下 Llama-8B 約 0.3M token」〔原文〕。〔計算〕以 Llama-3.1-8B 每 token 128 KiB（32 層×8 KV head×128 維×K、V×2 bytes），並以平均 input 近似文件長度：LooGLE 105×21,613≈2.27M token≈277 GiB；NarrativeQA 50×54,797≈2.74M token≈334 GiB。都超過單張 H200（141 GB）而小於 1 TB pinned DRAM，與原文「靠 CPU 記憶體達到約 95% 命中」一致〔複核：V04 重算 277.0／334.5 GiB 無誤；「H200 141 GB」是廠商規格，原文沒寫，屬〔判讀〕；「與 95% 命中一致」也是〔判讀〕〕 | [S] p.2；p.10 §5.2.1；〔計算〕 |
| 主結果的工作點 | Fig 8 為 output token 吞吐 vs 平均 TTFT 曲線（掃 request rate）；倍數定義為「同 TTFT 下的吞吐」並取最大值〔原文〕。消融（Fig 9）：低 request rate 時排程的貢獻較大，高 rate 時 I/O 機制主導〔原文〕 | [S] p.10 §5.2.1；p.11 §5.3.1 |
| 掃描的自變數 | request rate；模型 3 種；資料集 4 種；page size（Fig 2：1–512；Fig 10：32–1024）；cache distance（min／shuffle／max）；cache resolve time 與吞吐（模擬，1／10／100 req/s）；平台（H200 vs GH200）；I/O kernel 的 CUDA block 數（Fig 5：0–128）〔複核修正：原寫 1–128；Fig 5 橫軸從 0 起，[S] p.6〕；版面 layer-first vs page-first（Fig 13）；冷／暖快取（NarrativeQA 預熱）〔原文〕 | [S] p.4–6；p.10–13 |
| 對手 | vLLM v0.8.5；vLLM-LMCache（LMCache v0.2.1）；TRT-LLM v0.17.0；TRT-LLM-HiCache（TRT 的自動 CPU 卸載）；SGLang v0.4.5；**SGLang-HiCache 是作者自己實作的基線**（layer-wise cudaMemcpyAsync、page 32，對應 CachedAttention／Pensieve／FlashGen 的做法）。消融變體：Strata-IO、Strata-Schedule-Only、Strata-IO-LPM、Strata-Oracle〔原文〕 | [S] p.9 Baselines；p.11 §5.3.1；p.13 §5.4 |
| 系統指標 | 主指標：平均 TTFT、output token 吞吐。另有：cache hit rate（以 page 為單位比對；Fig 10、12 正規化呈現，Fig 2 為原始值〔複核修正：原寫一律正規化，[S] p.4 Fig 2、p.11 Fig 10、p.13 Fig 12〕）、prefill 中被 I/O 卡住的時間比例、持續頻寬（GB/s）。沒有 SLO 門檻；percentile 只在動機圖（Fig 2 的 P90 TTFT）〔原文〕 | [S] p.10 §5.2；p.3 Fig 1；p.4 Fig 2；p.13 Fig 15 |
| 品質指標 | 無。原文說 Strata 做精確快取，不影響準確度〔原文〕 | [S] p.14 §7 |
| 主要結果 | (1) LooGLE「同 TTFT 吞吐」：Llama-8B 對 SGLang-HiCache／vLLM-LMCache／TRT-LLM-HiCache 為 3.2×／2.6×／1.9×；Qwen-14B 3.9×／2.1×／1.9×；Llama-70B 5×／5×／3.75×。ReviewMT（Llama-8B）1.7×／2.3×／2.3×。(2) 暖快取 NarrativeQA 對 vLLM-LMCache：2.3×／2.6×／2.5×（8B／14B／70B）。(3) 圖 15：Strata-IO 的持續 host–GPU 頻寬由 40 GB/s 到 150 GB/s（原文文字；對應 PCIe 平台與 GH200）。(4) 磁碟子實驗（DeepSeek-V3、8×H20、page 32、12 req/s）：page-first 版面讓平均 TTFT 好 2.1×、吞吐好 1.3×〔原文〕 | [S] p.10 §5.2.1–5.2.2；p.13 §5.3.5、§5.4 |
| 消融／敏感度／開銷 | 元件分解（Fig 9）：排程與 I/O 各帶來最高 1.8× 與 2.3× 的峰值吞吐。page size（Fig 10）：SGLang-HiCache 最佳設定（page 512）只達 Strata-IO 的 93%，主因命中率低 2.4%。cache distance（Fig 11）：min 時 delay hit 緩解 +42%；shuffle／max 時 I/O 機制 +76%／+95%，balance batch 再 +11%／+12%，stall hiding 再 +8%／+3%。delay hit 敏感度（Fig 12，模擬、無限容量）。I/O kernel 干擾（Fig 5）：2 個 1024-thread block 達 48 GB/s，prefill 降 <5%、decode 降 10%。§6：cudaMemcpyBatchAsync 38 GB/s vs GPU kernel 48 GB/s〔原文〕 | [S] p.6；p.11–14 |
| 重複與統計 | 未說明（無重複次數、誤差棒、信賴區間）〔原文〕 | [S] 全文 |
| 程式碼／資料 | 原文說已整合進 SGLang 並部署於數家 AI 公司的生產環境，沒給專屬 repo〔原文〕。〔程式碼〕SGLang `542addad` 的 HiCache 參數：`hicache_write_policy` 可選 write_back／write_through／write_through_selective，**CLI 預設 write_through**；`hicache_io_backend` 預設 kernel；`hicache_mem_layout` 預設 page_first；`hicache_ratio` 在 cache 模式預設 2.0（host 池為 device 池的 2 倍）；selective 的門檻寫死為 2（write_through 為 1）。資料集皆公開 | [S] p.2–3；[code-SGL] |
| 設計理由（原文） | Little's Law：吞吐 X＝C·S／L，加大單次傳輸量 S 是最實際的槓桿，但大 page 會降低命中率，所以要讓小 page 也能高效搬（§3.1）；context 越長，重算越貴，是「不具吸引力的替代方案」，所以載入延遲必須藏住（§3.2）；單一寫入策略不夠，不同部署在寫入頻寬、容量、持久性、未來重用之間的取捨不同（§4.4）；磁碟延遲高又難預測，所以預取採 best-effort（§4.2.1）；ShareGPT 限制 GPU 記憶體是為了凸顯分層基線的行為；多數實驗不用磁碟是因為基線對磁碟支援有限（§5.1） | [S] p.4、p.5、p.7、p.9、p.10 |
| 設計理由〔判讀〕 | 主實驗只用 CPU 層，讓比較集中在 PCIe 搬運與排程，避開 SSD 的變異；代價是 SSD 結論只來自一個子實驗。「少量長文件 × 多 query」讓命中率約 95%，主實驗的工作點是「I/O-bound、幾乎全命中」，不是容量受限、需要逐出的情境 | — |
| 原文沒講清楚的地方 | GPU KV 池實際大小（只說依各引擎預設）；KV dtype；Fig 8 各點的 request rate 值與 in-flight 上限是否被觸發；每個 rate 的請求數與時長；冷快取實驗有沒有暖機；`loading_bound` 門檻 100 與 delay-hit 門檻 100 token 是否隨模型調整；selective 門檻的邊界語意（原文寫「超過」門檻，又說門檻設 1 等於 write-through；〔程式碼〕顯示 write_through 對應門檻 1，〔判讀〕實際是「達到」）；SGLang-HiCache 基線的 CPU 容量與逐出是否與 Strata 相同 | [S] p.8–10；[code-SGL] |
| 與既有整理不一致 | ① ✅ workloads_eval §1.1：已核對一致（模型、表 1 統計、Poisson＋60 s、in-flight 128、基線、三個平台）。可補：Llama-70B 用 4 卡 TP；ShareGPT 限 GPU 約 500K token；CPU 1 TB pinned。〔複核：V04 對照 workloads_eval L59 與 [S] p.9–10，一致〕<br>② ✅ **版本**：arXiv v1 只有 H200 與 GH200 兩個平台；沒有三種寫入策略與「各層 LRU」的段落；磁碟只有「8192 token 從本地磁碟載到 CPU」的微基準（page-first 最多快 4×），沒有 DeepSeek-V3／H20 實驗。凡引「預設門檻 2」「各層 LRU」都要引 OSDI 版。〔複核：V04 全文檢索 [S-arX]，無 write-back／write-through／LRU 字樣；"two platforms" 見 v1 p.7；4× 見 v1 p.10 §5.3.4 與 Fig 12〕<br>③ ✅ intro.txt 表 7 腳註 e、表 8「准入控制」、sota.txt §2.3、PAPERS_BY_LEVEL 的「寫入門檻預設 2」與論文一致，但 **SGLang 上游目前 CLI 預設是 write_through（每次都備份）**〔程式碼〕；寫「生產做法」時要說明是論文預設還是上游預設。〔複核修正：原列「表 6、§8.3」；intro 表 6 與 §8.3 都沒有提門檻 2，改列表 7 腳註 e 與表 8。上游預設已由 V04 於 `542addad` 重抓 `memory.py` L118–124、`unified_radix_cache.py` L501–503 確認〕<br>④ ✅（措辭補正，非事實錯誤）「逐出只用 LRU」（intro 表 6）：原文是「各層預設 LRU」（[S] p.9）。「已在 SGLang 的生產環境使用」（sota §2.3）：原文是「整合進 SGLang，並部署在數家 AI 公司的生產環境」（[S] p.3）。<br>⑤ ✅（一致，無指控）intro.txt 表 7「最長評測 55K」：是各資料集平均 input 的最大值（NarrativeQA 54,797），單筆可到 128K（腳註 f 已說明，一致）。<br>⑥ ✅（部分）PAPERS_BY_LEVEL「全部用 Poisson 到達」：端到端實驗確實都是 Poisson（[S] p.10），但 Fig 12 用的是縮放後的 Mooncake 時間戳（模擬執行，[S] p.12–13），ShareGPT 另在 Poisson 之上插固定 60 s；§5.3.5 的 12 req/s 未說明到達分布。<br>⑦ ✅ 其餘（吞吐最高 5×、Little's Law、delay hit、SSD 只在一個子實驗、原文認為長 context 重算不划算）已核對一致 | [S]、[S-arX]、[code-SGL] |
| 對本研究的意義〔判讀〕 | **可沿用**：(a) min／shuffle／max cache distance 的構造法，把「重用距離」變成自變數；(b) 縮放真實 trace 時間戳做負載掃描（Fig 12）；(c) 三種寫入策略可直接當 PoC 的寫入基線（SGLang 參數可切換）。**不可比**：主實驗幾乎全命中 CPU 層、沒有容量壓力，與「寫入時決定存不存」需要的容量受限情境不同。**要小心**：倍數是曲線上的最大值；「SGLang-HiCache」是作者自建的弱化基線（page 32），不是上游 HiCache | — |

---

### Bidaw：Enhancing Key-Value Caching for Interactive LLM Serving via Bidirectional Computation–Storage Awareness（FAST'26）

- **讀了什麼**：〔全文〕FAST'26 USENIX 版。查證 2026-10-06。
- **一句話**：多輪聊天 KV 存 DRAM＋SSD；讓計算與儲存互相感知，減少載入阻塞與 miss。
- **評測要證明的主張**：既有的兩層 KV 快取（CachedAttention、FlashGen）比理想的「全部在 DRAM」延遲高最多 3.8×、吞吐低最多 2.0×；I/O 感知的雙佇列排程＋以上一輪回答長度輔助的逐出＋改存較省空間的中間 tensor，能把延遲降最多 3.58×、吞吐提高最多 1.83×，接近全 DRAM 上界。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | OPT-6.7B、Qwen-7B、OPT-13B、Qwen-14B、OPT-30B（Qwen 引的是 2023 年的 Qwen Technical Report，即 Qwen1）〔原文〕。注意力：原文說「存中間 tensor」只對 MHA 有利，GQA 應改存 KV〔原文〕；所用模型的 config 都沒有 KV head 分組，為 MHA〔文件〕。HF config 的視窗：OPT 2,048；Qwen-7B seq_length 8,192；Qwen-14B seq_length 2,048〔文件〕。dtype 未說明 | [B] p.11 §5 Environment；p.10–11 §4；[HF] |
| 硬體 | 1×A800 80 GB；200 GB host memory（performance layer，等於 GPU 記憶體的 2.5×）；4 顆 SATA SSD 組 RAID-5，1.5 GB/s（capacity layer）；PCIe Gen4（約 30 GB/s）。另以「從 host memory 複製＋注入延遲」模擬 5 GB/s SSD〔原文〕 | [B] p.11 §5；p.5；p.12 §5.1 |
| 軟體與版本 | CachedAttention 與 FlashGen 在 vLLM 上重做（[B] p.11）；Bidaw 本身建在哪個引擎上原文沒寫〔未查證〕〔複核修正：原寫「原型建在 vLLM 上」，全文只說 "we implement CachedAttention and FlashGen based on vLLM" 與 "We design and implement the Bidaw system"（p.3），沒有說 Bidaw 本身基於 vLLM〕；vLLM 版本未說明。混粒度 GPU block（歷史與 query 用 256-token 大 block，回答用 16-token 小 block）；tensor 轉換放在低優先 CUDA stream〔原文〕 | [B] p.10–11 §4、§5 |
| 資料／負載 | (1) 產業夥伴的 interactive conversation workload：超過 100 萬輪；query 平均 36、response 平均 45；每位使用者輪數平均／中位／P90＝22／18／45（另一處寫平均 22.4）；trace 公開於 GitHub。(2) ShareGPT 多輪（只跑 OPT-13B）。逐出規律的分析用 8:00–20:00 不同時段的 10 分鐘 trace，共 12 組〔原文〕 | [B] p.4 §2.2；p.5；p.8 §3.3.1；p.11 腳註 1、§5；p.12 §5.3 |
| 長度 | 自家負載只給平均 query 36、response 45（**原文未標單位**）；動機實驗最多展示 2,048 個歷史 token（Fig 3）；成本效率分析用 2,048-token 歷史（Fig 14）。正式評測的 context 分布，以及歷史超過模型視窗時怎麼處理，原文沒寫〔原文〕 | [B] p.4；p.10 |
| 到達與併發 | 自變數是「平均每分鐘新到的使用者數」（users/min）；使用者開始對話後會間歇送出多輪請求。ShareGPT 沒有時間戳，用 Poisson 模擬〔原文〕；自家負載的輪間時間應來自 trace〔判讀：原文以「ShareGPT 沒有時間戳」說明為何改用 Poisson〕。只評單張 A800 在去除冗餘計算後算得完的到達率。開跑前暖機，把 performance layer 填滿〔原文〕 | [B] p.5；p.11 §5 Workload；p.12 §5.3 |
| 重用結構 | 同一使用者的多輪對話歷史；每一輪都要載入之前所有輪的 KV；不跨使用者共享〔原文〕 | [B] p.2 Fig 1、§1；p.14 §6（「不跨使用者」出自與 MeanCache 的對照句）〔複核補充〕 |
| 分層實作 | **層**：performance layer＝host DRAM 200 GB（敏感度掃 120–200 GB）；capacity layer＝SSD RAID-5 1.5 GB/s（實測；5 GB/s 為模擬）。<br>**寫入**：inclusive caching，每輪計算時就把（省空間的）tensor 寫入，capacity layer 也保留一份，逐出時不必大量寫；寫入不在關鍵路徑。<br>**逐出**：host 空閒記憶體低於 5% 時觸發；以「上一輪回答長度」預測下次存取的重用距離下界，剪掉不可能的分桶，再用背景 Belady ghost cache 估各分桶的命中機率，算出「命中潛力」，逐出最低者；不逐出等待中請求的資料。<br>**讀取**：ready／preparing 雙佇列；preparing 佇列以 disk-HRRN（1＋等待時間／KV 大小）排 SSD 讀取〔原文〕 | [B] p.6 §3.1；p.7 §3.2；p.8–9 §3.3；p.10 §4；p.13 §5.6 |
| 工作集／容量 | 原文有：同時快取的 KV 總量隨到達率快速超過 200 GB，最多達 performance layer 的 3.91×（Fig 5）；80% 的存取重用距離超過 200 GB（OPT-13B、30 users/min，Fig 6a）；performance layer 平均容納 40.1% 的 KV 時，既有逐出策略命中率只有約 20%（Fig 6b）；現有 GPU 伺服器的 host memory 通常是 GPU 記憶體的 1.6–3.2×〔原文〕 | [B] p.5–6 |
| 主結果的工作點 | Fig 15 是平均延遲 vs users/min 曲線，各模型 x 軸範圍不同。「延遲最多降 3.58×」出自 OPT-13B（原文只說 "For OPT-13B ... up to 3.58×"；「在基線延遲暴增的點」是〔判讀〕〔複核修正：原未標〕）；吞吐定義為「相近延遲下可撐的 users/min」，為 1.43–1.83×。5 GB/s 模擬時 FlashGen 15.18→20.23、Bidaw 27.81→30.35 users/min。ShareGPT 上只剩 1.40×（FlashGen 9.09 s vs Bidaw 8.70 s 的點）〔原文〕 | [B] p.11–12 §5.1、§5.3 |
| 掃描的自變數 | users/min；模型 5 種；host memory 120–200 GB（GPU 記憶體的 1.5–2.5×）；SSD 頻寬 1.5 vs 模擬 5 GB/s；負載（自家 vs ShareGPT）；逐出策略；逐步加入技術的消融〔原文〕 | [B] p.11–14 |
| 對手 | vLLM（每輪全重算）；CachedAttention、FlashGen（皆閉源，作者在 vLLM 上重做）；理想上界（重複從 host 載入同一份 KV，模擬「全在 DRAM」）。逐出比較：queue-enhanced（CachedAttention 的策略）、LFU、LRU、FIFO；Belady 只用在 ghost cache 內〔原文〕 | [B] p.11 §5 Comparing systems；p.12 §5.4 |
| 系統指標 | 每請求平均端到端回應延遲；吞吐＝相近延遲下的 users/min；P90／P95／P99 延遲；performance layer miss rate（每次請求載入一次歷史＝一次存取）；排隊時間 CDF；開銷（排程、逐出、轉換時間）〔原文〕 | [B] p.11–14 |
| 品質指標 | 無；原文說 Bidaw 無損（只重排請求，tensor 轉換可還原 KV）〔原文〕 | [B] p.12 |
| 主要結果 | (1) OPT-13B 延遲最多降 3.58×，平均延遲最多降 83.9%；吞吐 1.43–1.83×。(2) miss rate 比 queue-enhanced 最多低 57.6%，比一般策略最多低 69.9%。(3) 排隊時間 5.76 s→2.45 s（−57.5%，OPT-13B、30 users/min）。(4) ShareGPT 只有 1.40×；原文說用 Poisson 模擬時間後，回答長度逐出不再降低 miss rate〔原文〕 | [B] p.11–13 |
| 消融／敏感度／開銷 | 逐步消融（OPT-30B）：I/O 感知排程延遲降 1.58×，加逐出吞吐再 1.25×，加省空間 tensor 再 1.10×；host memory 敏感度（1.75–2.19×）；SSD 頻寬（5 GB/s 模擬）；尾延遲（對 CachedAttention／FlashGen：P90 −52.96%／−66.63%，P95 −49.30%／−62.64%，P99 −47.03%／−56.81%）；開銷：排程 0.62 ms、逐出 0.35 ms（Belady ghost 2.86 ms）、tensor 轉換數十 ms〔原文〕 | [B] p.12–14 |
| 重複與統計 | 未說明。Spearman 相關 0.94–0.98（12 組）只用來支持「回答長度與重用距離下界相關」〔原文〕 | [B] p.8 |
| 程式碼／資料 | trace 公開：https://github.com/ShipengHu-777/Interactive-conversation-workload（論文腳註）。系統程式碼原文沒給連結〔未查證是否另有釋出〕 | [B] p.11 腳註 1 |
| 設計理由（原文） | KV 位置與大小差異大、SSD 頻寬低；layer-wise 重疊只能蓋住第一個 iteration（數十 ms），SSD I/O 卻要數百 ms，所以要在請求層級排程（§3.2）。人要讀完回答才問下一題，回答越長、中間插入的其他使用者越多，重用距離下界越大，這是人機互動的本質，所以可推廣（§3.3.1）。最佳策略的命中率會隨容量與負載改變，固定估計不準，所以用 ghost cache 持續估（§3.3.3）。用 PCIe 4.0 是因為主流伺服器用它，CPU–GPU 頻寬很少是瓶頸，SSD 才是（§5）。只評單卡算得完的到達率，避免引入計算瓶頸（§2.2） | [B] p.5、p.7–9、p.11 |
| 設計理由〔判讀〕 | 用 SATA RAID-5（1.5 GB/s）讓 SSD 成為明顯瓶頸，倍數因此放大；改用 5 GB/s 模擬時，FlashGen 提升 33%、Bidaw 只提升 9%，差距縮小。用 OPT（2K 視窗、MHA）讓「存中間 tensor」有利；換成 GQA 長 context 模型時，第三個技術不再適用 | — |
| 原文沒講清楚的地方 | users/min 怎麼從 trace 產生（抽樣使用者？壓縮時間？）；每位使用者的輪間時間是否用真實值；歷史超過模型視窗（OPT 2K）時怎麼截斷；正式評測的 context 長度分布；vLLM 版本；KV dtype；ghost cache 用多長的過去 trace；promising 分桶數 m；ShareGPT 的輪數與時間設定 | [B] p.5、p.9、p.11 |
| 與既有整理不一致 | ① ✅ workloads_eval §1.1（L56）「真實時間戳，**以抽樣使用者**調整 users/min」：原文沒有說怎麼調整，應改為〔未查證〕。〔複核：V04 全文檢索 "sampl" 無相關句；p.10–11 只定義 users/min 為每分鐘新到使用者數〕其餘（模型、基線、指標、硬體、ShareGPT 只跑 OPT-13B）已核對一致；指標可補 P90／P95／P99。<br>② （無指控）workloads_eval 的 trace 統計（含歷史 prompt 中位 1,066、p99 8,168）是〔計算〕，本卡未重算。<br>③ ⚠️ 部分成立：sota.txt §2.5、PAPERS_BY_LEVEL「用上一輪回答多長預測使用者多久後回來：回答越長，KV 越早移到 SSD」。〔複核修正：原寫「**過度簡化**……不是單調的『回答越長越早移』」，說得太重〕原文自己的概述就是這樣講的：p.6 "uses its length to predict the user future access timing"、p.3「回答越長，使用者讀得越久，下一次存取越晚」。機制層面，原文實際預測的是「加權重用距離下界」（兩次存取之間其他使用者存取的 KV 總量，單位是位元組，[B] p.8），再用 Belady ghost cache 的分桶命中率算「命中潛力」，逐出最低者（p.9 式 2）。對同一位使用者，下界越大會剪掉命中率較高的分桶，命中潛力不升反降，所以「回答越長越早被逐出」的方向成立〔判讀〕；跨使用者時還取決於各自的重用距離分布。建議只補一句「以位元組計的重用距離下界＋影子快取」，不必改成「錯誤」。<br>④ ❌ 指控不成立：sota.txt §2.5 寫的是「『回答越長、越晚回來』只對真人聊天成立，原文承認換成 Poisson 到達就失效」，主語本來就是這個訊號（即逐出元件），與原文 p.12 "the previous-answer-based eviction strategy no longer reduces miss rates" 一致；PAPERS_BY_LEVEL 也寫「第 2 招……失效」。原卡「失效的是逐出元件，整體仍有 1.40×」是補充，不是更正。〔複核修正〕<br>⑤ ✅ intro.txt 表 7「Bidaw 最長評測 8.2K（腳註 k：公開 trace 的 p99）」：這是 trace 的計算值，不是論文評測的長度（腳註已揭露來源，但欄名是「最長評測」）；論文展示的歷史只到 2,048 token（p.4 Fig 3、p.10 Fig 14）。OPT 與 Qwen-14B 的視窗是 2K，但 Qwen-7B 的 `seq_length` 是 8,192〔文件〕，所以「實際評測 context 很可能 ≤2K」只對 OPT／Qwen-14B 成立〔判讀〕〔複核補充：原卡只寫 OPT 2K〕。<br>⑥ 一致：延遲 ↓3.58×、既有方法比全 DRAM 慢最多 3.8×、問題平均 36（原文未標單位；sota 與 PAPERS_BY_LEVEL 寫「36 token」，單位是使用者文件自加，原文同段以 token 描述 Mooncake，推定為 token〔判讀〕〔複核補充〕）、存中間 tensor 只對 MHA 划算、Belady 影子快取、trace 公開、只適用真人聊天、PAPERS_BY_LEVEL 各點 | [B]；workloads_eval；intro.txt；sota.txt |
| 對本研究的意義〔判讀〕 | **可沿用**：(a)「性能層容量／同時快取的 KV 總量」與「重用距離分布」兩個工作集指標；(b) Belady ghost cache 估「命中潛力」，可當 PoC 的上界或預測器校準；(c) 用延遲注入模擬更快 SSD 的作法（要註明是模擬）。**不可比**：短對話（每輪約 81 個單位）、MHA、2K–8K 視窗〔複核修正：原寫 2K；Qwen-7B 為 8K〕，與 16K–512K 的長 context 不在同一量級。**要小心**：倍數依賴慢 SSD（1.5 GB/s） | — |

---

### MTDS：Multi-tier dynamic storage of KV cache for LLM inference under resource-constrained conditions（Complex & Intelligent Systems 2026）

- **讀了什麼**：〔全文〕出版社 PDF（Springer，12:104，17 頁），DOI https://doi.org/10.1007/s40747-025-02200-4 ；取得方式見來源清單。查證 2026-10-06。
- **一句話**：邊緣小卡上做 GPU→DRAM→SSD 三層 KV，載入前先判斷比重算快不快。
- **評測要證明的主張**：在頻寬與 DRAM 都小的邊緣伺服器上，選擇性重用（全載／部分載／不載）＋多 GPU 卸載優先排程＋兩級自適應 DRAM 逐出，比雲端式卸載降 TTFT 超過 25%，DRAM（active）命中率提高超過 20%。〔複核補充：原文自身不一致，摘要寫 "by up to 20%"（p.1），引言與 §Evaluation 寫 "more than 20%"（p.2、p.12），結論寫 "by up to 20%"（p.15）〕

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | GPT-2 1.5B、LLaMa-2 7B、LLaMa-3 8B、Qwen-3 14B（2 卡 TP，其餘單卡）；消融另用 LLaMa-7B／13B（表 1–2）、LLaMa2-7B（Fig 14）、LLaMa-1B（Fig 15）〔原文〕。注意力〔文件〕：GPT-2 MHA（n_positions 1,024）、Llama-2-7B MHA、Llama-3-8B GQA（原生視窗 8,192）、Qwen3-14B GQA。dtype：理論式假設 KV 為 Float16〔原文〕 | [M] p.10 Models；p.7 式 (8)；p.12–14；[HF] |
| 硬體 | 4×NVIDIA A10（每張 24 GB）、64 GB DRAM、2 TB SSD、PCIe Gen4〔原文〕 | [M] p.10 Testbed |
| 軟體與版本 | PyTorch／Python 實作並整合進 vLLM（**版本未說明**）；GPU↔DRAM 傳輸由專屬 I/O thread 經 shared memory 處理；前綴比對用 Trie、排序用 TimSort〔原文〕 | [M] p.8；p.10 |
| 資料／負載 | ShareGPT（Hugging Face `shibing624/sharegpt_gpt4`）；合成 Random 資料集（固定 input／output 長度，來源 GitHub `theripnono/get-random-wikipedia-content`）〔原文〕 | [M] p.10 Workloads；p.15 Data Availability |
| 長度 | ShareGPT 的長度沒報；Random 在 LLaMa-3 8B 上把 input 從 16K 增到 24K 做壓力測試（Fig 9b）；消融固定 1K input〔原文〕 | [M] p.11；p.12 |
| 到達與併發 | ShareGPT 以 Poisson（λ=1）到達；Random 的到達方式沒寫；預設 batch size 20；消融 batch 50〔原文〕 | [M] p.10；p.12 |
| 重用結構 | 前綴比對：新請求對 DRAM 資料庫，以及排隊請求彼此之間；ShareGPT 的多輪〔判讀〕；Random 的重用怎麼來沒寫；消融以「匹配比例 α∈[0,1]」人為控制〔原文〕 | [M] p.4–5；p.8；p.12 |
| 分層實作 | **層**：GPU VRAM（每個方法的 KV buffer 統一設 5 GB）→ DRAM 64 GB（命中稱 active hit）→ SSD 2 TB（命中稱 inactive hit）。頻寬皆實機，無模擬。<br>**載入**：Selective KV Cache Loader 依歷史延遲（資料不足時用理論式）把請求分三類：全載、部分載（載入匹配的前綴，其餘重算）、不載；有多個匹配長度時，選最接近最佳點 NoT2 的那一個。<br>**寫入（卸載）**：多 GPU 同時卸載時排優先度：與其他排隊請求共享前綴越多優先度越高，資料庫已有其前綴者優先度下降，已完整存在者不再卸載；低優先度延後。<br>**逐出**：兩級門檻（容量 V0／V1／V2；利用率門檻 U1＜U2，例：0.01／0.1）；利用率＝ξ×未來視窗內的匹配頻率＋(1−ξ)×歷史 LRU 分數；超過 V2 就停止新卸載〔原文〕 | [M] p.5–10；p.14 |
| 工作集／容量 | 原文沒給工作集對容量的比例；只給 DRAM 64 GB、VRAM KV buffer 5 GB，並說在邊緣峰值請求率下兩級逐出不會觸發截斷〔原文〕 | [M] p.9–10 |
| 主結果的工作點 | 單一負載點（Poisson λ=1、batch 20），沒有負載掃描；TTFT 降 25–31% 是 ShareGPT 上各模型的數字。Random 以 input 長度為自變數，效益先上升後持平〔原文〕 | [M] p.10–11 |
| 掃描的自變數 | 模型 4 種；資料集 2 種；input 長度 16K–24K（Random）；匹配比例 α；卸載排程器開／關；逐出模組原版 vs 自適應（ξ=0.25）〔原文〕 | [M] p.11–14 |
| 對手 | 原版 vLLM（開啟 KV 卸載與重用；只有全重用或全重算）、Mooncake、LMCache；三者都設 5 GB VRAM KV buffer。版本與部署方式沒寫〔原文〕 | [M] p.10 Baseline |
| 系統指標 | TTFT、prefill 吞吐（兩個資料集）；DRAM（active）cache hit rate、GPU time（只在 ShareGPT，原文說只有長時間運行才有意義）；GPU 記憶體使用（每模型每方法隨機取 20 個時間點，畫 box plot）；PCIe 頻寬利用率（5 分鐘視窗）〔原文〕 | [M] p.3；p.10–14 |
| 品質指標 | 無（精確前綴重用）〔原文〕 | [M] 全文 |
| 主要結果 | (1) ShareGPT 上 TTFT 比原版 vLLM 降 25–31%，也低於 Mooncake／LMCache。(2) DRAM 命中率 91.4%／89.2%／85.1%／73.6%（GPT-2／LLaMa-2 7B／LLaMa-3 8B／Qwen-3 14B），比 vLLM 高 20% 以上；模型越大，優勢越小。(3) 排程器降低 PCIe 利用率（例：ShareGPT LLaMa-7B 32.09%→13.24%），命中率小幅下降（0.875→0.841）；但 Random LLaMa-13B 從 0.380 降到 0.215〔原文〕 | [M] p.11；p.12；p.14 表 1–2 |
| 消融／敏感度／開銷 | 選擇性重用 vs vLLM 全重用／全重算（α 掃描，LLaMa2-7B、batch 50、1K input，Fig 14）；排程器開／關（表 1–2）；自適應逐出 vs 原版（LLaMa-1B，依插入時間分 49 組看平均命中頻率，Fig 15）。沒有開銷量測，只說長輸入時策略選擇耗時會限制效益〔原文〕 | [M] p.11–14 |
| 重複與統計 | 未說明；GPU 記憶體以 20 個隨機取樣點畫 box plot〔原文〕 | [M] p.12 |
| 程式碼／資料 | 原文沒有提供程式碼連結或程式碼可得性聲明〔未查證是否另有釋出〕；Data Availability 只給兩個資料集的公開連結〔原文〕〔複核修正：原寫「程式碼未釋出」，原文沒有這樣的陳述〕 | [M] p.15 |
| 設計理由（原文） | 邊緣多用 PCIe 4.0（×16 單向 ≤32 GB/s），DRAM 常少於 512 GB，載入 KV 可能比重算還慢（C1）；多 GPU 同時卸載搶 PCIe，低重用的 KV 擋住高重用的，命中率下降（C2）；固定 LRU 門檻在小 DRAM 上很難設（C3）。實測載入時間隨可重用 token 數「先慢後快」非線性成長，重算時間線性成長，所以只有中段值得載（Fig 6）。兩級逐出是精細度與 CPU 開銷的折衷。理論式忽略初始化與封包開銷，只在歷史資料少時用 | [M] p.2–4；p.6–7；p.9 |
| 設計理由〔判讀〕 | 「載入時間超線性、重算線性」與一般預期（傳輸與長度成正比、注意力計算隨長度超線性）相反，比較可能來自其實作（shared memory I/O thread、封包化）與 5 GB VRAM buffer，而不是硬體本質；所以「命中很長時不載反而快」高度依賴實作 | — |
| 原文沒講清楚的地方 | NoT1／NoT2／NoT3 的數值；vLLM 版本，以及「原版多層卸載」指的是哪個功能；Mooncake（設計上需 RDMA 多節點）怎麼在單機 4×A10 上部署；Random 的到達過程與重用構造；ShareGPT 的長度與輪數；V0／V1／V2 與 ξ 的實際值；「LLaMa-3 8B」是 Llama-3（原生 8K 視窗）還是 3.1（128K）——若是前者，16K–24K 輸入已超出原生視窗；主實驗中 SSD 實際被讀到的比例 | [M] p.6；p.10–11；[HF] |
| 與既有整理不一致 | ① ✅ workloads_eval §1.1（L60）「Random 資料集為固定長度（數值未抄錄）」：原文有 16K→24K（LLaMa-3 8B，Fig 9b，[M] p.11），可補。其餘（模型、ShareGPT＋Poisson λ=1、基線、指標、4×A10／64 GB／2 TB、batch 20）已核對一致。<br>② ⚠️ 部分成立：sota.txt §2.6「邊緣伺服器……頻寬也小，**從 SSD 載入**有時候比重算還慢」。〔複核修正：原卡說「SSD 只出現在背景敘述，應改成從 DRAM 載入」，說得太重〕原文的動機句確實寫 "KV caches must be reloaded from DRAM or SSD before reuse, incurring transfer latency that may exceed recomputation time"（[M] p.1），p.2 也寫 "from SSD or DRAM to GPU VRAM"，所以 sota 的句子有原文依據。但 MTDS 的機制（C1 "limited DRAM-to-GPU bandwidth"、Fig 5 的 DRAM 資料庫、Policy #1–#3）與 Fig 6 的量測都是 DRAM→GPU（[M] p.4–6）。建議改成「從 DRAM 或 SSD 載入」，並註明實驗量的是 DRAM→GPU。<br>③ ✅（補充，非更正）sota.txt §2.6 與 intro.txt 表 6「整段一起決定，不依位置切開」：大致成立。原文在多個匹配長度中挑最接近 NoT2 的那個（[M] p.6 式 3），等於「**載入前段、重算後段**」〔判讀〕〔複核修正：原未標判讀〕；方向與 Cake（重算前段、載入後段）相反，值得寫出來。注意 intro.txt 表 7 已給 MTDS「位置感知 △，腳註 l：可以只載入前段」，使用者文件其實已部分寫到。<br>④ ✅ intro.txt 表 7「MTDS 最長評測 —」：可填 24K（Random 壓力測試，[M] p.11）。表 7 的「—」定義為「原文未涉及或未查證」，屬漏填而非錯誤。<br>⑤ ✅（一致，附條件）intro.txt §7.5「MTDS 量到命中很長時不載入反而快」：與原文一致（Fig 6 第三區、p.6；長輸入時更常選重算、p.12），但要註明限於其 A10／PCIe 4／5 GB buffer 的實作，且原文沒給 NoT 數值。<br>⑥ 一致：TTFT ↓ 超過 25%、模型最大 14B、ShareGPT＋Poisson、依歷史延遲預測、被越多排隊請求共用的 KV 越優先卸載（補：資料庫已有其前綴者優先度下降）、PAPERS_BY_LEVEL 各點 | [M]；workloads_eval；intro.txt；sota.txt |
| 對本研究的意義〔判讀〕 | **可沿用**：「載入 vs 重算的交叉點要每套硬體實測」的立場，以及 α（匹配比例）掃描的消融設計。**不可比**：輸入短、單一負載點、沒有重複、基線部署方式不明。**要小心**：MTDS 是「載前段、算後段」，與 Cake 相反，PoC 的對照組應兩種方向都放。它也是 24 GB 級卡（A10）的前例，與平台 A（3090）規模相近 | — |

---

### LMCache：An Efficient KV Cache Layer for Enterprise-Scale LLM Inference（preprint 2025；arXiv 2510.09665）

- **讀了什麼**：〔全文〕arXiv v2；〔程式碼〕dev `8c77a6f7` 只查預設值。查證 2026-10-06。
- **一句話**：把引擎的 KV 抽出存到 CPU／磁碟／遠端，跨查詢與跨引擎共用的快取層。
- **評測要證明的主張**：以 chunk 為單位批次搬運、計算與 I/O 管線化、模組化 connector 與控制 API，讓 LMCache＋vLLM 在多輪 QA、文件分析等負載上吞吐最高 15×、延遲至少低 2×。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | Llama-3.1-8B-Instruct、Sao10K-L3-8B、Llama-3.1-70B-Instruct、Qwen2.5-Coder-32B-Instruct、Qwen3-Coder-480B-A35B-Instruct-FP8、Qwen2.5-72B-Instruct；SGLang 實驗另用 Qwen3-32B（TP=2）〔原文〕。注意力全為 GQA（8 個 KV head）〔文件〕〔複核補充：V04 重查 HF config 確認；唯 "Sao10K-L3-8B" 原文沒給確切 repo，查到的 Sao10K L3-8B 系列（Stheno-v3.2、Lunaris-v1）皆為 8 KV head，屬推定〕。dtype 未說明（480B 為 FP8 權重） | [L] p.10–11 §8.1；p.15 §8.8；[HF] |
| 硬體 | 單節點：GMI Cloud 的 8×H100，各模型用能啟動的最少 GPU 數。多節點：同樣 GPU 數，外加以 CPU 記憶體存 KV 的集中遠端伺服器（15 Gbps）。PD：prefiller 與 decoder 各用相同 GPU 數，以 NVLink 相連。§8.7 的 prefill 延遲在 B200 上量〔原文〕 | [L] p.11 Hardware；p.12 §8.4；p.14 §8.7 |
| 軟體與版本 | LMCache v0.3.6；basic vLLM v0.10.2（只在 GPU 做 prefix caching）；vLLM v0.11.0 原生 CPU offloading；兩家商用端點（9 月 10 日存取）；PD 比 vLLM 原生 PD（NIXL）。預設 chunk 256 token〔原文〕 | [L] p.11 Baselines；p.6 §5.1；p.13 §8.5 |
| 資料／負載 | (1) 模擬多輪文件 QA：每個 query＝一份約 12 頁 PDF 的文件＋一個獨特短問題；(2) 公司 F、G 的 input／output 長度分布驅動的 trace；(3) LongBench TriviaQA（集中儲存實驗）；(4) vLLM 官方 benchmark 的 random（PD 實驗）〔原文〕 | [L] p.11 §8.2；p.12 §8.3–8.4；p.13 §8.5 |
| 長度 | 多輪 QA 每 query 10K token（Llama-3.1-8B 為 20K），輸出最多 100；PD 為 8K input／200 output；§8.7 的單請求延遲敏感度把 context 掃到約 400K（Fig 15 橫軸 0–400K）〔原文〕 | [L] p.11；p.13；p.14–15 |
| 到達與併發 | 多輪 QA：一開始 40 位使用者，之後依指定 QPS 加入新使用者。公司 trace：原始 trace 跨數天，為了在 1 小時內跑完而壓縮時間（原文用 stretch 一詞）。TriviaQA：用 vLLM benchmark 腳本以 Poisson 產生到達。PD：官方腳本〔原文〕 | [L] p.11–12 |
| 重用結構 | 多輪文件 QA（同一文件被多次問）；公司 trace 的前綴重用；集中儲存跨實例共享〔原文〕 | [L] p.11–12 |
| 分層實作 | **層（設計）**：GPU／CPU DRAM／本地磁碟／遠端（Redis、Mooncake、S3 等）。**層（實驗）**：只用 CPU（上限 500 GB）與遠端 CPU 伺服器（15 Gbps）；**任何端到端實驗都沒有 SSD**。<br>**寫入**：token processor 找出後端還沒有的新 token 並存入〔原文〕，即「全部寫」〔判讀〕；decode 產生的 KV 累積滿一個 chunk 才寫（delayed decode storing）；GPU 空閒 page 只預先複製一部分到 CPU（dynamic offloading，三指標）。<br>**讀取**：layer-wise pipelining；排隊期間預取到較快層。<br>**逐出**：論文沒寫 LMCache 後端的逐出策略（Fig 7 的 "Least recently used" 標籤指的是 vLLM GPU 空閒 page 池的排列，不是 LMCache 後端逐出〔複核補充，[L] p.8〕）；〔程式碼〕dev 分支 `cache_policy` 預設 "LRU"、`max_local_cpu_size` 預設 5.0 GB、`chunk_size` 預設 256。<br>**頻寬**：§8.7 的 32／64／128 Gbps 怎麼產生沒寫 | [L] p.5–8；p.11–12；p.14；[code-LMC] |
| 工作集／容量 | 原文沒給比例；只給 CPU 上限 500 GB，並說遠端能放比 CPU 多得多的 KV。§2.2 的使用統計（只有相對值）顯示超出 GPU 的 KV 量逐週增加〔原文〕 | [L] p.3；p.11–12 |
| 主結果的工作點 | Fig 8 掃 QPS；「同 TTFT 下的吞吐」對最強基線為 2.3–14×；低 QPS（QPS=1）時 TTFT 小 1.9–8.1×、ITL 小 7–92%。公司 trace 在高 QPS 時 TTFT 至少小 3.7–6.8×、ITL 小 19–58%（Fig 9 圖說為 4.4–6.6×）。集中儲存 1.3–3×。**摘要寫最高 15×，正文是 14×**〔原文〕 | [L] p.1；p.11–12 |
| 掃描的自變數 | QPS；模型 6 種；情境（CPU offload、集中儲存、PD）；網路頻寬 32／64／128 Gbps × context 長度（§8.7）；引擎（vLLM／SGLang）〔原文〕 | [L] p.10–15 |
| 對手 | basic vLLM v0.10.2；vLLM v0.11.0 CPU offloading（在 Qwen3-Coder-480B 上跑不起來）；商用 #1／#2（黑盒，原文推測 #1 沒有二級儲存卸載）；vLLM 原生 PD；SGLang（有／無原生 CPU offloading）〔原文〕 | [L] p.11–13；p.15 |
| 系統指標 | TTFT（prefill 延遲）、ITL（相鄰 token 平均間隔）；PD 另報 P95 TTFT；同 TTFT 下的 QPS；元件延遲分解；載入頻寬（Gbps）〔原文〕 | [L] p.11；p.13–14 |
| 品質指標 | 無（精確前綴重用）〔原文〕 | [L] 全文 |
| 主要結果 | (1) CPU offload：同 TTFT 下 QPS 為最強基線的 2.3–14×。(2) 從 CPU 載入頻寬 400 vs 88 Gbps（vLLM 原生）。(3) PD：平均 TTFT 好 1.53–1.84×、ITL 好 1.12–1.66×。(4) B200 上，32 Gbps 時只有 context 超過 256K 載入才比 prefill 快，64／128 Gbps 時所有長度都較快〔原文〕 | [L] p.11；p.13–15 |
| 消融／敏感度／開銷 | 元件分解：PD 傳輸 3.68 s vs 4.47 s（Fig 14）、CPU 載入頻寬（表 5）、非同步計算使端到端延遲降 1.46×（Fig 13）；context 長度 × 網路頻寬（Fig 15）；SGLang 整合（Fig 16）。**沒有寫入策略或逐出策略的消融**〔原文〕 | [L] p.13–15 |
| 重複與統計 | 未說明〔原文〕 | [L] 全文 |
| 程式碼／資料 | 開源：https://github.com/LMCache/LMCache ；公司 F／G 的 trace 不公開（只用其分布）；LongBench 公開〔原文〕 | [L] p.1；p.12 |
| 設計理由（原文） | 引擎原生 page（20–63 KB）太小，吃不滿頻寬，所以以 chunk 為單位搬；傳輸要與推論重疊，並避免 CPU 發 memcpy 的開銷；避免多餘複製；connector API 讓 LMCache 跟得上快速演進的引擎；控制 API 讓路由器等上層能做 KV 感知的決策；遠端載入只有在 context 夠長或頻寬夠大時才比 prefill 快，所以應依情況決定（§8.4、§8.7） | [L] p.4；p.6；p.12–15 |
| 設計理由〔判讀〕 | 評測重心是「搬得快不快」與「整合廣不廣」，策略（放哪、丟誰）不是貢獻，所以沒有逐出／寫入的消融；大倍數主要來自 GPU-only 基線在高 QPS 時的重算排隊 | — |
| 原文沒講清楚的地方 | 逐出策略與各層容量（CPU 500 GB 之外）；多輪 QA 每位使用者的輪數、輪間時間、文件數；公司 trace「壓成 1 小時」的方法（等比縮放？）；§8.7 的頻寬是實際網路還是限速；GPU KV 池大小；「同 TTFT」取哪個 TTFT 值；商用服務的設定 | [L] p.11–15 |
| 與既有整理不一致 | ① ✅ workloads_eval §1.1（L62）：模型列表漏了 Qwen2.5-72B-Instruct（[L] p.11 列在模型清單，也出現在 Fig 8、9、11）；其餘（10K／20K、40 位使用者＋QPS、公司 trace 壓成 1 小時、TriviaQA Poisson、基線版本、8×H100）已核對一致。<br>② （無指控）workloads_eval §0／§5.3 的「14×」與正文一致（[L] p.11）；摘要寫 15×（p.1），是原文自身前後不一。<br>③ ✅（補註建議）intro.txt §7.1「LMCache 每次查詢 10K–20K」：端到端實驗一致；但 §8.7 的 context 長度敏感度（Fig 15 橫軸 0–400K，[L] p.14–15）掃到約 400K，在「評測 context 太短」的論述中可補註；「單請求」是〔判讀〕，原文沒寫併發設定。表 6「吞吐 ↑ 15×、策略不是重點」一致（15× 為摘要值）。<br>④ ✅ sota.txt §2.8、PAPERS_BY_LEVEL「業界最常用的開源 KV 快取層」：這是原文的自我描述（摘要 "the first and so far the most efficient"、結論 "most widely adopted production-ready KV caching layer"，[L] p.1、p.16），應標為原文主張。「被 vLLM、SGLang 整合」一致。<br>⑤ ✅ PAPERS_BY_LEVEL「相依套件是 NVIDIA 專屬，在 AMD 上很可能跑不起來」：原文 §9 說社群已加入對 NVIDIA、AMD、Ascend、TPU 四種處理器的支援（[L] p.15–16）；此句至少應改成「原文稱支援 AMD，平台 B 上未驗證」。PAPERS_BY_LEVEL 其他各點一致 | [L]；workloads_eval；intro.txt；sota.txt；PAPERS_BY_LEVEL |
| 對本研究的意義〔判讀〕 | **可沿用**：作為「全部寫＋LRU」的生產基線（vLLM connector 可直接接）；§8.7 的「context 長度 × 頻寬」交叉圖正是 κ 的另一種量法，可直接對照。**不可比**：沒有 SSD 端到端、沒有容量壓力掃描。**要小心**：基線是 GPU-only vLLM 時倍數會很大，PoC 應拿 LMCache 本身當基線 | — |

---

### Mooncake：Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot（FAST'25；arXiv 2407.00079）

- **讀了什麼**：〔全文〕FAST'25 USENIX 版；〔部分〕arXiv v4（題名改為 *A KVCache-centric Disaggregated Architecture for LLM Serving*）。**兩版的實驗內容差異很大，本卡分開標註 [MC-F] 與 [MC-A]**。查證 2026-10-06。
- **一句話**：Kimi 的 P/D 分離架構，池化叢集記憶體成全域 KV 快取並以 KV 為中心排程。
- **評測要證明的主張**：FAST 版：全域 KV 池＋KV 為中心的排程，在長 context 下讓「有效請求容量」比基線方法高 59%–498%〔複核修正：原寫「比 vLLM」；摘要原文是 "compared to baseline methods"，Fig 1 有三個 vLLM 設定，百分比對哪一個沒寫，[MC-F] p.2〕；生產上讓 Kimi 在 A800／H800 叢集多處理 115%／107% 請求；全域快取命中率最高是本地快取的 2.36×。arXiv v4：模擬情境吞吐最高 +525%，真實負載多處理約 75% 請求，並加上過載時的提前拒絕。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | FAST：dummy 模型，架構同 LLaMA3-70B（80 層、d=8192、GQA 8、BF16；每 token KV 320 KB）。arXiv v4：dummy，架構同 LLaMA2-70B。兩版都不量品質〔原文〕 | [MC-F] p.4 Table 1、p.11 §5.3.1；[MC-A] p.3、p.15 |
| 硬體 | FAST：每節點 8×A800-SXM4-80GB＋4×200 Gbps RDMA NIC；每個系統 16 節點（排程實驗也是 16 節點；快取實驗用 10 個 prefill 節點）；每節點約 1 TB DRAM 可當本地快取。arXiv：8×A800，節點間 RDMA 最高 800 Gbps〔原文〕 | [MC-F] p.9 §5.1；p.10；p.12；[MC-A] p.15 |
| 軟體與版本 | FAST：vLLM v0.5.1（prefix caching 與 chunked prefill 因版本限制分開測）；Mooncake Store 的 block 設 256 token；Transfer Engine（GPUDirect RDMA、拓樸感知選路、16 KB 切片、端點池用 SIEVE 逐出）〔原文〕 | [MC-F] p.6–7；p.9 |
| 資料／負載 | **FAST 表 2**：Conversation（線上 1 小時取樣，12,031 筆，平均 input 12,035／output 343，cache ratio 40%，時間戳）；Tool&Agent（1 小時，23,608 筆，8,596／182，59%，時間戳）；Synthetic（ShareGPT、LEval、LooGLE 以 1:1:1 混合，3,993 筆，15,325／149，66%，Poisson）。**arXiv 表 2**：ArXiv Summarization（8,088／229，~0%）、L-Eval（19,019／72，>80%）、Simulated（16k／32k／64k／128k、output 512、50%）、Real（7,955／194，~50%，時間戳）；公開 trace 23,608 筆，hash block 512 token〔原文〕 | [MC-F] p.10 Table 2、§5.2.1；[MC-A] p.6–7 §4、p.15 Table 2 |
| 長度 | FAST：Conversation 最長到 128K、平均約 12K（真實資料）；延遲分解掃 8k–128k prompt。arXiv：Simulated 16k–128k〔原文〕 | [MC-F] p.10；p.13 Fig 14；[MC-A] p.15–16 |
| 到達與併發 | FAST：兩份真實 trace 依時間戳送出，輸出達預定長度即終止；Synthetic 以 Poisson 送出。arXiv：真實 trace 依時間戳重播；其他以 Poisson 並用 RPS 控制；過載實驗把重播速度加到 2×〔原文〕 | [MC-F] p.10；[MC-A] p.15；p.17 |
| 重用結構 | 前綴 block 鏈式雜湊；Conversation 為多輪歷史；Tool&Agent 為又長又完全重複的系統提示；Synthetic 中多輪保留先後、同一長 prompt 的多題各成一請求，再整體隨機打亂〔原文〕 | [MC-F] p.6；p.10 |
| 分層實作 | **層（設計）**：池化每節點的 CPU DRAM 與 SSD，以 RDMA 跨節點傳。Store 以 paged block 存（典型 16–512 token，實驗 256），滿了用 LRU 逐出（正在被存取的除外）；熱 block 由排程器複製多份。<br>**寫入**：prefill 節點把新產生的增量 KV 寫回 CPU 記憶體（全部寫）。<br>**層（實驗）**：兩版的快取都在 DRAM，**沒有任何 SSD 實驗**。<br>**頻寬**：Fig 13 的 24–400 Gbps 為模擬；Transfer Engine 為實測（40 GB 傳輸達 87／190 GB/s）〔原文〕 | [MC-F] p.5–6；p.12–13；[MC-A] p.5 |
| 工作集／容量 | FAST：LLaMA3-70B 每 token 320 KB，1 TB DRAM 約 3M token；本地 3M token 在多數負載達不到理論最大命中率的 50%；約 50M token 才接近理論最大，需至少 20 個節點的 DRAM（Fig 9，只看請求序列的模擬）。arXiv：單一全域池 1,000→50,000 blocks 時，LRU 命中率由 0.30 升到 0.50〔原文〕 | [MC-F] p.4；p.11–12；[MC-A] p.7–8 Table 1 |
| 主結果的工作點 | FAST 的「有效請求容量」＝固定重播負載下，TTFT <30 s 且 TBT <門檻的請求比例〔原文，[MC-F] p.9〕；倍數是兩系統比例之比〔判讀〕〔複核修正：原標〔原文〕；原文只定義比例並在圖上標 +X%，沒寫倍數怎麼算；§5.2 另寫 "we measure the maximum throughput that remains within the defined SLO thresholds"，與比例定義並存〕，所以在最嚴的 TBT 門檻（100 ms）最大〔判讀〕：Conversation 圖上依門檻 I／II／III 標 +498%／+157%／+59%（對應關係依標籤位置判讀）；Tool&Agent 在 200 ms 對 vLLM prefix caching +42%；Synthetic 在 200 ms 對 vLLM +40%。arXiv 的「約 75% 更多請求」：同一重播下 Mooncake 約 100%、vLLM 只有 57% 滿足 TBT SLO〔原文〕，〔計算〕1／0.57≈1.75（V04 重算無誤），**是 SLO 達成比例之比，不是量到的最大負載差**〔判讀：原文只寫 "In this experiment, Mooncake can process approximately 75% more requests"，沒寫算法；與 1／0.57 吻合〕〔複核修正：原未標判讀〕 | [MC-F] p.2 Fig 1；p.9–11；[MC-A] p.17 |
| 掃描的自變數 | FAST：TBT 門檻（100／200／300 ms）；負載 3 種；排程演算法 4 種；本地 vs 全域快取；快取容量（Fig 9 模擬 1e4–1e9 token）；網路頻寬 24–400 Gbps（模擬）；prompt 長度 8k–128k × prefix ratio 0%／95%；P/D 比例（16 節點）；NIC 組態。arXiv：RPS、P/D 組態（3P+1D／2P+2D）、拒絕策略〔原文〕 | [MC-F] p.9–14；[MC-A] p.15–17 |
| 對手 | FAST：vLLM v0.5.1、vLLM＋prefix caching、vLLM＋chunked prefill（各 16 節點，每節點一個實例）；排程比 random、load-balancing、local cache-aware；傳輸比 TCP 與 torch.distributed（Gloo）。arXiv：vLLM-[4M]、vLLM-[20M]；過載實驗比「兩階段開始前就依負載拒絕」〔原文〕 | [MC-F] p.9–10；p.13；[MC-A] p.15–17 |
| 系統指標 | FAST：TTFT、TBT（取最長 10% token 間隔的平均）、有效請求容量、prefill GPU time、每請求 cache hit rate；TTFT 門檻 30 s；P/D 實驗用 10 s／100 ms。arXiv：P90 TTFT／TBT，門檻＝最低 RPS 觀測值的 10× 與 5×，並正規化為 1.0；拒絕數〔原文〕 | [MC-F] p.9；p.14；[MC-A] p.15 |
| 品質指標 | 無（dummy 模型）〔原文〕 | [MC-F] p.4 |
| 主要結果 | (1) FAST：Conversation 有效請求容量 +59%–498%。(2) prefill GPU time 比 vLLM 少 36%／53%／64%（三種負載）。(3) 全域 vs 本地快取：命中率最高 +136%，prefill 時間最多 −48%。(4) 128k 輸入時 prefix caching 讓 prefill 時間降 92%，含開銷後 TTFT 降 86%〔原文〕 | [MC-F] p.2；p.10–12；p.13〔複核修正：(4) 在 PDF p.13 §5.4.3，原寫 p.14〕 |
| 消融／敏感度／開銷 | 排程演算法（平均 TTFT：global 3.07 s、local cache-aware 3.58 s、load balancing 5.27 s、random 19.65 s）；快取容量理論分析；本地 vs 全域；副本數隨時間（Fig 11）；傳輸引擎 vs TCP／Gloo（2.4×／4.6×）；網路頻寬敏感度（建議至少 100 Gbps）；延遲分解；P/D 比例（約 1:1 最佳）〔原文〕 | [MC-F] p.8（Fig 5 排程 TTFT）〔複核補充〕；p.9–14 |
| 重複與統計 | 未說明〔原文〕 | [MC-F]、[MC-A] 全文 |
| 程式碼／資料 | 開源 https://github.com/kvcache-ai/Mooncake （trace 與傳輸引擎）〔原文〕；README 稱獲 FAST'25 Erik Riedel Best Paper〔文件〕 | [MC-F] p.4；GitHub README |
| 設計理由（原文） | 重用 KV 是否划算取決於頻寬 B 與算力 G 的比，模型越大越容易滿足（式 2：8×A800、prefix 8192 需約 6 GB/s，8×H800 需 19 GB/s）；本地 DRAM 只拿得到理論命中率約一半，所以要全域池；用 dummy 模型加重播 trace，是為了保護商業資訊又能重現；負載變化太快無法準確預測，所以副本改用啟發式熱點遷移 | [MC-F] p.3–5；p.8；[MC-A] p.3 |
| 設計理由〔判讀〕 | 以「SLO 達成比例」定義容量，讓倍數在嚴格 TBT 門檻下（vLLM 的長 prefill 打斷 decode）特別大；這主要量到的是 P/D 分離的好處，不是快取分層本身 | — |
| 原文沒講清楚的地方 | FAST 版 trace 的 hash block 大小（只有 arXiv 與 README 說 512，而實驗的 Store block 是 256）；Fig 1 的百分比是對哪個基線；Store 實際用的 DRAM 容量（快取實驗每節點 3M token）；SSD 是否參與任何實驗；模擬頻寬的方法；arXiv「7,590」與 FAST「8,596」的平均長度差異 | [MC-F] p.6；p.9–12；[MC-A] p.7 |
| 與既有整理不一致 | ① ✅〔複核：V04 逐項對照 workloads_eval L58 與 [MC-A] p.3、p.7–8、p.15–17，全部可在 arXiv v4 找到，FAST 版（共 17 頁，p.15–17 為參考文獻）都沒有〕**workloads_eval §1.1 的 Mooncake 列標為 FAST'25，但內容全部是 arXiv v4**（LLaMA2-70B dummy、ArXiv／L-Eval／模擬 16K–128K／23,000 條、vLLM-[4M]／[20M]、P90 正規化、Table 1、800 Gbps、p7–8／p15–17）。FAST 版是 LLaMA3-70B dummy、三種負載（12,031／23,608／3,993 筆）、vLLM v0.5.1 三種設定、TTFT 30 s＋TBT 100／200／300 ms 的有效請求容量、4×200 Gbps NIC。兩版要分開寫。<br>② ✅ workloads_eval §0 第 10 條（L40）、§5.2（L377）、§5.3（L399）的「525%」「75%」是 arXiv 數字（[MC-A] p.1、p.3、p.16–17）；FAST 版對應的是 59%–498% 與生產上的 115%／107%（[MC-F] p.2–3、p.9）。另：L377 標的 "p1、p16"，75% 實際在 arXiv p.17〔複核補充〕。<br>③ ✅（支持既有判讀）workloads_eval §2.2「原文 p7 寫 7,590，公開檔算出 8,590，判讀為筆誤」：FAST 表 2 的 Tool&Agent 平均 input 是 8,596，與公開檔計算一致，支持筆誤判讀，也確認公開的 toolagent trace 就是 FAST 的 Tool&Agent 負載（同為 23,608 筆；Conversation 12,031 筆、Synthetic 3,993 筆也與公開檔筆數相同）。<br>④ ✅ intro.txt §7.1「Mooncake 的 128K 是模擬資料」、表 17「模擬 16K–128K、快取比例 50%」：只適用 arXiv 版（[MC-A] p.15 Table 2）；使用者引用的 [11] 是 FAST 版（intro.txt 參考文獻 [11] 題名為 "Trading more storage for less computation"），其中 Conversation 真實 trace 本身就到 128K、平均 12K（[MC-F] p.10），模擬負載是 ShareGPT＋LEval＋LooGLE 混合。<br>⑤ ✅ intro.txt 表 6「請求量 ↑ 75%」、sota.txt §2.8「讓 Kimi 多處理 75% 的請求」、PAPERS_BY_LEVEL「真實負載下讓 Kimi 多處理 75% 的請求」：都是 arXiv 數字；arXiv 的摘要自己就寫 "enables Kimi to handle 75% more requests"（[MC-A] p.1、p.3），所以使用者的措辭忠於 arXiv，錯在版本混用。它在 §8.1.3 的來源是同一重播下 SLO 達成比例（約 100% 對 57%，p.17），不是量到的最大吞吐差〔判讀，見本卡主結果〕；引 FAST 版應寫 115%／107%（生產）或 59%–498%（有效請求容量）。〔複核補充：原卡沒提 arXiv 摘要本身的措辭〕<br>⑥ ⚠️ 非錯誤，僅補充：sota.txt §2.8、intro.txt §1「池化 CPU 記憶體和 SSD」與原文架構敘述一致（[MC-F] p.2 "pooling CPU, DRAM, SSD and RDMA resources"）；只是兩版實驗都沒有 SSD，引用時別當成「SSD 層已被驗證」。〔複核修正：補上判定〕<br>⑦ ✅ 一致：每個 hash block 512 token（arXiv §4.1、p.7；FAST 版本文沒寫 512，FAST 實驗的 Store block 是 256〔複核補充〕）、需要多節點與 RDMA、dummy 模型不量品質 | [MC-F]、[MC-A]；workloads_eval；intro.txt；sota.txt；PAPERS_BY_LEVEL |
| 對本研究的意義〔判讀〕 | **可沿用**：公開 trace（Conversation、Tool&Agent、Synthetic）可直接當負載；Fig 9「容量 vs 理論最大命中率」的模擬法可當工作集分析的範本。**不可比**：多節點 RDMA、P/D 分離、dummy 模型。**要小心**：引用數字一定標版本；「有效請求容量」型的倍數取決於 SLO 門檻，不能當吞吐用 | — |

---

### CachedAttention：Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention（ATC'24；arXiv 2403.19708）

- **讀了什麼**：〔全文〕ATC'24 USENIX 版；arXiv v1、v2（當時題名 AttentionStore）只全文檢索「命中發生在哪一層」相關句。查證 2026-10-06。
- **一句話**：多輪對話的 KV 存 DRAM＋SSD 不丟，用重疊與佇列感知預取藏住載入。
- **評測要證明的主張**：分層 KV 快取（AttentionStore）＋layer-wise 預載與非同步保存＋排程感知的預取與逐出＋去掉位置編碼後再截斷，可讓 TTFT 降最多 87%、prefill 吞吐最多 7.8×、端到端成本降最多 70%。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | LLaMA-1 65B、LLaMA-2 13B／70B、Falcon-40B（活化 FP16）；另實作 Mistral-7B（32K 視窗），但只出現在動機圖 Fig 4b；PPL／準確度實驗用 LLaMA-7B／13B〔原文〕。注意力：原文說 70B 與 Falcon-40B 用 GQA（factor 8 與 16）〔原文〕；65B、13B 為 MHA〔文件〕。每 token KV：65B 2.5 MB、13B 0.78 MB、70B 0.31 MB、Falcon-40B 0.12 MB〔原文〕 | [CA] p.9–10 §4.1–4.2；p.5 Fig 4；[HF] |
| 硬體 | 4×A100 80GB、128 GB DRAM、10 TB SSD、PCIe Gen4（原文估有效頻寬約 26 GB/s）；SSD 型號與頻寬沒給（只泛稱磁碟 <5 GB/s）。13B 用 2 卡、24 batch；65B／70B／40B 用 4 卡、各 24 batch；部分消融用單卡〔原文〕 | [CA] p.5；p.9–10；p.11 |
| 軟體與版本 | PyTorch／Python 自建，模型實作基於 Transformers；host 與磁碟以 block 管理；專屬 CUDA stream 做 GPU↔host，獨立 I/O thread 做 host↔disk；continuous batching〔原文〕 | [CA] p.9 §4.1 |
| 資料／負載 | ShareGPT 9K 個 session（平均 5.75 輪，共約 52K 輪）；前 10K 輪暖機，評估後面 42K 輪〔原文〕 | [CA] p.10 §4.1–4.2 |
| 長度 | 受模型視窗限制：LLaMA-2 為 4K、LLaMA-65B 為 2K；溢出時截掉最早的一半（ratio 0.5）。ShareGPT 統計：47% 的 session 超過 2K、30% 超過 4K。**Fig 2 的 CDF 為了顯示而排除超過 40 輪或超過 32K 的 session**〔原文〕 | [CA] p.4–5；p.10；p.12 |
| 到達與併發 | 新 session 以 Poisson λ=1.0／s 到達；同一 session 內各輪的間隔沒寫；§4.3.8 在 LLaMA-13B 上掃 0.5–2.0 session/s〔原文〕 | [CA] p.10；p.14 |
| 重用結構 | 同一 session 的多輪歷史；session 是最小的快取與逐出單位（要嘛全用、要嘛全不用）〔原文〕 | [CA] p.8 §3.3.2 |
| 分層實作 | **層**：HBM → DRAM（128 GB）→ SSD（2／5／10 TB），HBM 另設 read／write buffer。頻寬為實機。<br>**寫入**：每輪的 KV 都非同步寫回 host（prefill 逐層、decode 逐層），全部保存。<br>**預取**：依 job queue 的 look-ahead 視窗（長度＝可用 DRAM／平均 session KV），把即將執行的 session 從磁碟拉到 DRAM。<br>**逐出**：look-ahead eviction 視窗（長度＝(DRAM＋磁碟)／平均 session KV），視窗內的不逐出，優先逐出視窗尾端的；磁碟滿時，把最晚到的 session 移出系統。另有 TTL（§4.3.6 設 1 小時）〔原文〕 | [CA] p.6–8 §3.2–3.3；p.13 §4.3.6 |
| 工作集／容量 | 原文有：容量需求 CCpUT＝每單位時間的不同 session 數 × 每 session 的最大 KV；RCC／CCpUT＝0.1 時命中 51%、0.25 時 98%（TTL 1 h）。只用 HBM（10 GB）命中近 0%；HBM＋DRAM 為 3.4%／1.7%／7.7%／19.1%；再加 SSD 為 86%／71%／89%／90%（13B／65B／70B／40B）〔原文〕 | [CA] p.13–14 §4.3.6–4.3.7 |
| 主結果的工作點 | 單一負載點（λ=1.0 session/s）的整體比較，不是曲線。改變到達率時命中率 82%→77%、TTFT 0.122→0.154 s，原文結論是影響很小〔原文〕 | [CA] p.10；p.14 |
| 掃描的自變數 | 模型 4 種；歷史／新 token 比（500/500–900/100）；預載 buffer 大小（0–15 層）；prompt 長度 1K–1.6K（保存）；DRAM／SSD 容量（128G／2T、5T、10T）；溢出處理（OF vs CA）；RCC／CCpUT；儲存媒介組合；session 到達率 0.5–2.0／s〔原文〕 | [CA] p.11–14 |
| 對手 | RE（每輪全重算，溢出時截斷 token）；逐出消融比 LRU、FIFO；截斷比 TT（截 token 後重算）與 NKVT（直接截含位置編碼的 KV）；儲存媒介比 HBM-only、HBM＋DRAM〔原文〕 | [CA] p.10–14 |
| 系統指標 | cache hit rate（DRAM＋磁碟合計，並分層報）；TTFT；prefill 吞吐；GPU time；推論成本（AWS：A100 $5/h、DRAM $0.0088/GB/h、SSD $0.000082/GB/h）〔原文〕 | [CA] p.10–11 |
| 品質指標 | 只用於截斷方案：PPL（WikiText-2、C4、PTB）與準確度（MMLU、LongEval、PIQA），比較 CA、TT、NKVT〔原文〕 | [CA] p.12–13 表 1–2 |
| 主要結果 | (1) 命中率 86%／71%／89%／90%，TTFT 降 85%／61%／87%／86%，prefill 吞吐 6.8×／2.6×／7.8×／7.2×（13B／65B／70B／40B）。(2) 成本降 70%／43%／66%／68%，儲存成本佔 16.4%／9.0%／9.0%／9.0%。(3) 128G／10T 時 CA 命中 86%、LRU 58%、FIFO 48%；**CA 超過 99.6% 的命中發生在 DRAM**，LRU／FIFO 的 DRAM 命中只有約 0.5–0.6%〔原文〕 | [CA] p.10–12 |
| 消融／敏感度／開銷 | RE vs CA（歷史／新 token 比）；layer-wise 預載（PL-B0 −35%、PL-B15 −61%）；非同步保存（−13% 至 −15%）；排程感知預取／逐出 vs LRU／FIFO（GPU time 最多 2.7×）；截斷（OF 命中率降 17.6%–41.5%）；PPL／準確度；容量需求；儲存媒介；到達率〔原文〕 | [CA] p.11–14 |
| 重複與統計 | 未說明〔原文〕 | [CA] 全文 |
| 程式碼／資料 | 論文沒給程式碼連結〔未查證是否另有釋出〕；ShareGPT 公開 | [CA] 全文 |
| 設計理由（原文） | 以 LLaMA-65B 為例，2K token 的 KV 有 5 GB，190 GB 空閒 HBM 14 秒就滿、512 GB host 不到 1 分鐘就滿，所以要用磁碟；磁碟容量遠大於 DRAM，多數 KV 會在磁碟，請求隨機到達時就得讀磁碟，所以要靠 job queue 的未來資訊預取；同一 session 的 KV 要嘛全用要嘛全不用，所以以 session 為單位；RPE 讓 KV 能在存的時候去掉位置資訊 | [CA] p.3；p.5；p.8–9 |
| 設計理由〔判讀〕 | 預取靠的是「排程器已知的未來」（佇列中的等待請求），所以在 λ 固定、佇列有一定長度時最有效；若佇列很短（Bidaw 指出線上常少於 50 個等待請求），效果會變差 | — |
| 原文沒講清楚的地方 | session 內的輪間間隔；SSD 型號與頻寬；GPU KV 池大小；暖機以外是否有其他預熱；Mistral-7B-32K 是否進入任何評測；成本計算用的 DRAM／SSD 配置量；DSpUT 的實際值 | [CA] p.9–14 |
| 與既有整理不一致 | ① ✅〔複核：[CA] p.4 Fig 2 圖說 "For better display effect, the statistics exclude conversations with over 40 turns or sessions that exceed a length of 32K"；Mistral-7B 只在 p.5 Fig 4b〕workloads_eval §1.1（L57）「session ≤32K、≤40 輪」：**這是 Fig 2 為了顯示而排除的範圍，不是負載設定或上限**；實際可用 context 受 2K／4K 視窗限制。「Mistral-7B-32K」只在動機圖，不在評測。其餘（ShareGPT 9K／52K／5.75、Poisson λ=1.0、RE、LRU／FIFO、4×A100／128 GB／10 TB）已核對一致。<br>② ✅ intro.txt §7.1「CachedAttention 的 session 最長 32K」：同上，錯誤；應寫「評測模型視窗 2K–4K，溢出時截掉一半」（[CA] p.10 truncation ratio 0.5；p.12 LLaMA-65B 2K）。<br>③ ✅〔複核：V04 重讀 [Cake] p.2 原句屬實；ATC 版 p.12、v3 p.11、v2 皆為 "over 99.6% of the hits occurring in DRAM"，v1 為 99.9%；四個版本都沒有「80% 在磁碟」；Fig 24 推算 96.0／97.6／91.3／78.8% 重算無誤〕**intro.txt §2（表 1 下方）「依 AttentionStore 的評估，約 80% 的快取命中發生在磁碟層（Cake 引述），也就是說大部分的命中都要從最慢的那一層拿回來」**：Cake 確實這樣寫（[Cake] p.2），但 CachedAttention 的 v1、v2、v3／ATC 版全文都找不到這個數字；相反地，原文報告 CA 有超過 99.6%（v1 為 99.9%）的命中發生在 DRAM，因為排程感知預取先把要用的 KV 拉進 DRAM。原文支持的是「多數 KV 存放在磁碟」與「沒有 SSD 容量就沒有大部分命中」：〔計算〕由 Fig 24（HBM＋DRAM 的命中 vs 加 SSD 後的命中），靠 SSD 容量才得到的命中佔 96.0%／97.6%／91.3%／78.8%（13B／65B／70B／40B）。Cake 的「約 80%」可能由此而來〔判讀〕。建議改寫為「多數 KV 只能放在磁碟；不做預取時（LRU／FIFO），命中幾乎都發生在磁碟層」，並改引 CachedAttention 原文頁碼。後半句「大部分命中都要從最慢那層拿回來」與原文的 CA 結果矛盾，應刪或加條件。<br>④ ✅（加註建議）workloads_eval §2.1 Cake 列（L165）「引用 AttentionStore『約 80% 的命中發生在磁碟層』」：引用屬實，但被引的原文不支持，需加註。<br>⑤ ✅ 一致：intro.txt 表 8「預取：依排程佇列提前載入（AttentionStore）」 | [CA]、[CA-v1/v2]、[Cake]；workloads_eval；intro.txt |
| 對本研究的意義〔判讀〕 | **可沿用**：RCC／CCpUT（容量需求比）是把容量壓力參數化的好方法；分層命中率要分開報（DRAM vs 磁碟），並說明有沒有預取。**不可比**：2K–4K 視窗、單一負載點。**要小心**：引用「命中在磁碟」時要分清「KV 放在磁碟」與「從磁碟讀」 | — |

---

### Tutti：Making SSD-Backed KV Cache Practical for Long-Context LLM Serving（preprint 2026；arXiv 2605.03375）

- **讀了什麼**：〔全文〕arXiv v1；〔文件〕GitHub repo metadata。查證 2026-10-06。
- **一句話**：從 NVMe 還原 KV 時把 I/O 控制交給 GPU，讓 SSD 層接近 DRAM 層。
- **評測要證明的主張**：分頁 KV 造成大量小隨機 I/O，CPU 主導的路徑（即使用 GDS）是瓶頸；GPU 原生物件抽象＋GPU io_uring＋slack-aware 排程，讓 TTFT 比 GDS 版 LMCache 低 78.3%、可達請求率 2×、成本低 27%，接近 DRAM 版 LMCache。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | Llama3-8B（單卡，主要）；GLM-4-9B-Chat-1M（2 卡 TP，擴展性實驗）〔原文〕。注意力：Llama3-8B GQA（8 個 KV head）、GLM-4-9B-1M GQA（multi_query_group_num 4）〔文件〕。dtype 未說明 | [T] p.8–9 Models；[HF] |
| 硬體 | 64 核 Xeon 6530、512 GB 記憶體、2×H100 80GB、4×Solidigm D7-PS1010 7.68 TB；分層設定配 256 GB pinned DRAM、每 GPU 14 TB SSD。動機實驗：DRAM–HBM 50 GB/s，兩顆 SSD 峰值讀 29 GB/s、寫 12 GB/s。頻寬消融用兩顆 RAID-0〔原文〕 | [T] p.3 §2.2；p.8 Environments；p.10 |
| 軟體與版本 | vLLM 0.12.0 與 0.17.0 兩代；LMCache（圖例為 0.3.9 搭 vLLM 0.12.0、0.4.1 搭 0.17.0；§2.2 文字寫最新版 v0.4.2）；Tutti 約 8,000 行 C++＋1,500 行 Python，接 vLLM KVConnector；建在 GeminiFS 上；以 NVIDIA green context 切分 SM〔原文〕 | [T] p.3；p.6–9 |
| 資料／負載 | LEval（20 個子任務，3k–200k token）、LooGLE（4 個子任務，許多超過 100k）〔原文〕 | [T] p.9 Workloads |
| 長度 | LEval 3k–200k；LooGLE 多數 >100k；頻寬消融 1K–128K；TTFT 消融固定總長 128K、prefix 16K–128K；GLM 擴展性 prefix 128K–640K；bubble 消融固定 32K〔原文〕 | [T] p.9–12 |
| 到達與併發 | 資料集沒有原生時間戳，以 Poisson 模擬；以 round-robin 從各子資料集抽請求，模擬多 session。request rate：LEval 0.25–1.5、LooGLE 0.2–0.6 req/s（Fig 8 橫軸）〔原文〕 | [T] p.9 |
| 重用結構 | 長文件被多次查詢〔判讀：LEval／LooGLE 同一文件對應多題〕；原文沒描述重用怎麼構造〔原文〕 | [T] p.9 |
| 分層實作 | **層**：Tutti 本身是 HBM–SSD 兩層（不經 DRAM）；基線為 HBM-only、LMCache-DRAM-LW、LMCache-SSD、LMCache-GDS。頻寬為實機量測。<br>**放置**：以 vLLM block 為物件，每個 GPU file＝2×層數個物件；跨多顆 SSD round-robin。<br>**寫入**：讀取優先；寫入只在有 slack 時發出，剩下的在 decode 期間 best-effort 刷出；分散式模式下，KV 從 GPU 被逐出時才寫 SSD，並向 Mooncake 註冊副本。<br>**逐出**：SSD 層的逐出沒寫〔原文〕 | [T] p.2；p.5–8 |
| 工作集／容量 | 原文用表 1 呈現各儲存層的命中率：HBM 8%／4%、DRAM 53%／24%、SSD 84%／86%（LEval／LooGLE）；沒給工作集大小〔原文〕。「每列代表該層為最低層時的總命中率」是〔判讀〕（原文只說 HBM 容量不足、DRAM 提高到 53%／24%、SSD 維持 84%／86%，[T] p.9）〔複核修正：原把這個解讀寫成原文〕。〔計算〕Llama3-8B 每 token 128 KiB，128K token 的 KV 約 16 GiB；256 GB DRAM 約可放 2M token | [T] p.9 Table 1；〔計算〕 |
| 主結果的工作點 | Fig 8 為平均 TTFT／ITL vs request rate，違反 SLO 的點不畫。78.3% 是 vLLM 0.17.0、LEval「高負載點」對 GDS 的 TTFT 降幅（同點對 DRAM 69.1%）〔複核修正：原寫「最高負載點」；新版段落原文是 "at high load"，明寫 "at the highest load point" 的是舊版 vLLM 0.12.0 對 GDS 的 71.8%，[T] p.10〕；2× 是「1 s TTFT SLO 下的有效請求率」（對 DRAM +50%、對 GDS +100%）；LooGLE 0.6 RPS 時 TTFT 對 DRAM −93.2%、對 GDS −62.0%〔原文〕 | [T] p.10 §4.1 |
| 掃描的自變數 | request rate；vLLM 版本；資料集；context 長度（頻寬）；prefix 長度（TTFT）；命中率 50–100%（bubble）；PRP vs SGL；GPU 數與磁碟數（擴展性）〔原文〕 | [T] p.9–12 |
| 對手 | HBM-only vLLM；LMCache-DRAM-LW（端到端結果中的「DRAM」）；LMCache-DRAM（無 layer-wise）；LMCache-SSD（memcpy＋標準非同步 I/O）；LMCache-GDS。全部自己架設；vLLM 為 64-token page、LMCache 為 256-token chunk〔原文〕 | [T] p.8–10 |
| 系統指標 | 平均 TTFT、平均 ITL；各層命中率；儲存頻寬（retrieve／store）；GPU bubble time；每百萬 token 成本（H100 $5/h、DRAM $0.0088/GB/h、SSD $0.000082/GB/h）〔原文〕 | [T] p.9；p.12 |
| 品質指標 | 無（精確重用）〔原文〕 | [T] 全文 |
| 主要結果 | (1) TTFT 對 GDS −78.3%、對 DRAM −69.1%（LEval，新版 vLLM，高負載點〔複核修正：原寫最高負載點〕；舊版 vLLM 在最高負載點對 GDS −71.8%〔複核補充〕）。(2) 1 s TTFT SLO 下有效請求率對 DRAM +50%、對 GDS +100%。(3) retrieve 頻寬最高 25.9 GB/s，比 GDS（約 11.9 GB/s 飽和）高 2.08×。(4) GLM-4-9B-1M：GDS 在 512K／640K 時 OOM，Tutti 完成並在 640K 得到 1.2 s TTFT〔原文〕 | [T] p.10–11 |
| 消融／敏感度／開銷 | retrieve／store 頻寬（1K–128K）；PRP vs SGL（31.0×／91.3×）；prefix 長度 16K–128K（對 GDS 好 5.8%–61.4%；>96K 時落後 DRAM 至多 20.6%）；多 GPU 擴展；layer-wise 非同步管線（bubble 平均 25 ms，計算轉 I/O 的交叉點推到 98.3% 命中）；成本（LooGLE 0.5 QPS 對 LMCache-SSD −66.2%、對 GDS 約 −27%）。動機另量：GPU 雜湊表比 CPU 慢（insert 9.0–24.2×、lookup 25.6–50.0×）；讀寫並行時總頻寬降 60%（Fig 6 標 60.1%）〔原文〕 | [T] p.4–5；p.6（60% 在 §3.3 與 Fig 6）〔複核補充〕；p.7；p.10–12 |
| 重複與統計 | 未說明〔原文〕 | [T] 全文 |
| 程式碼／資料 | 原文自稱開源但沒附連結；GitHub `xPU-IO/Tutti`（Apache-2.0，2026-05-30 建立）〔文件〕。資料集公開 | [T] p.2；[gh-T] |
| 設計理由（原文） | SSD 的瓶頸不是原始頻寬，而是分頁版面造成的大量小隨機 I/O 與 CPU 介入；GDS 仍由 CPU 發每一個 I/O；layer-wise 管線在 SSD 上會把傳輸切得更碎；讀寫並行會搶 NVMe 內部快取，所以讀寫要分開排；寫入頻寬對端到端的影響小於讀取 | [T] p.1–4；p.6–7；p.10 |
| 設計理由〔判讀〕 | 選 LEval／LooGLE 讓長前綴、高命中成為常態，凸顯讀取路徑；比較兩代 vLLM 是為了論證「計算越快，I/O 路徑越關鍵」 | — |
| 原文沒講清楚的地方 | SLO 門檻（圖說說違反 SLO 的點省略，但只在文字提到 1 s TTFT）；每個 rate 的請求數與時長；重用怎麼構造（每份文件被問幾次）；SSD 層的逐出；128K prefix 的 TTFT 155.743 s 與 640K 的 1.2 s 數量級落差沒有解釋；LMCache 版本 0.4.1 與 0.4.2 前後不一；GPU KV 池大小 | [T] p.3；p.9–11 |
| 與既有整理不一致 | ① ✅ workloads_eval §1.1（L61）：已核對一致（模型、LEval 3K–200K、LooGLE >100K、round-robin、Poisson、基線、指標、2×H100、4×Solidigm）。可補：擴展性到 640K prefix；LMCache 版本 0.3.9／0.4.1。<br>② ✅ intro.txt §7.1「Tutti 到 200K，但只處理 SSD→GPU 的搬運」：端到端到 200K 正確，但 GLM-4-9B-1M 的擴展性實驗到 640K prefix（[T] p.11 Fig 12）；Tutti 也處理寫入（store_layer）路徑與讀寫排程（p.7、p.10 Fig 9b），不只讀取。「只處理搬運、不處理放置策略」這層意思則成立〔判讀〕。<br>③ ✅（一致，無指控）sota.txt 開源表「Tutti 有」：論文自稱 "first open-source"（p.2）但沒附連結；repo `xPU-IO/Tutti` 存在（V04 以 GitHub API 重查：2026-05-30 建立、Apache-2.0）。<br>④ ✅ 一致：intro.txt §7.3「Tutti 把寫入排在關鍵路徑的讀取之後」（p.7）；表 6「TTFT ↓78.3%」（條件：vLLM 0.17.0、LEval 高負載點、對 GDS〔複核修正：原寫最高負載點〕）；PAPERS_BY_LEVEL「比開了 GDS 的 LMCache TTFT 降 78.3%」「需要 NVMe 和特殊的 GPU 檔案系統」 | [T]；workloads_eval；intro.txt；sota.txt；PAPERS_BY_LEVEL |
| 對本研究的意義〔判讀〕 | **可沿用**：表 1「同一負載、不同最低層的命中率」是量化「多一層值多少」的簡潔方式；「讀寫並行使頻寬降 60%」可直接當「寫入有成本」的證據。**不可比**：只有 HBM–SSD 兩層，DRAM 只當基線。**要小心**：78.3% 是單一高負載點〔複核修正：原寫最高負載點〕；平台需要 NVMe＋GeminiFS | — |

---

### KVDrive：A Holistic Multi-Tier KV Cache Management System for Long-Context LLM Inference（preprint 2026；arXiv 2605.18071）

- **讀了什麼**：〔全文〕arXiv v1。查證 2026-10-06。
- **一句話**：長 context decode 把 KV 卸到 DRAM／SSD，只抓稀疏注意力需要的部分。
- **評測要證明的主張**：稀疏度已難再壓，長 context＋大 batch 時 KV 搬運主導 decode 延遲；以注意力為依據的 GPU 快取（滑動視窗＋lookahead 逐出）、彈性管線、協調的 HBM–DRAM–SSD 儲存，能在維持準確度下把吞吐提高最多 1.74×。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | Llama-3-8B-1048K（gradientai）、Qwen3-8B、Qwen3-14B（128K，用 YaRN）、Phi-4-mini-instruct（3.8B，128K，用 LongRoPE）〔原文〕。注意力全為 GQA（8 個 KV head）〔文件〕。dtype 未說明 | [K] p.15 §9.1；p.14–15 §8；[HF] |
| 硬體 | L20 48 GB＋Xeon 8457C＋100 GB DDR5；H20 96 GB＋EPYC 9K84（96 核）＋200 GB DDR5；RTX 4090 24 GB＋Xeon Gold 6430＋120 GB DDR5；磁碟為 NVMe U.2（型號、數量、頻寬沒給）〔原文〕 | [K] p.15 |
| 軟體與版本 | 約 9,000 行 Python、1,000 行 C++、3,000 行 CUDA；PyTorch 2.3.0、Python 3.12、CUDA 12.1；SSD 以 numpy.memmap 存取；聚類用 RetroInfer 的 Triton kernel，gather／copy 取自 ShadowKV，另用 FlashInfer；**所有基線都在作者的統一框架中重做**〔原文〕 | [K] p.14–15 §8 |
| 資料／負載 | LongBench、RULER（準確度）；吞吐實驗用的輸入內容沒寫〔原文〕 | [K] p.15；p.17 |
| 長度 | 吞吐：Llama-3-8B-1048K 為 60k／120k／240k／360k；Qwen3-8B 與 Phi-4-mini 為 60k／90k／120k。準確度最長到 128K〔原文〕 | [K] p.16 Fig 13；p.17 |
| 到達與併發 | 沒有到達過程；離線批次 decode，batch 與 context 對應：60k→8、90k→4、120k→4、240k→2、360k→2；另有 batch 掃描（Fig 24）。原文說支援 continuous batching〔原文〕 | [K] p.16 Fig 13 圖說；p.22；p.15 |
| 重用結構 | 沒有跨請求重用；所謂重用是 decode 步之間「關鍵 entries」的時間局部性〔原文〕 | [K] p.5 Finding 1 |
| 分層實作 | **層**：HBM（滑動視窗的關鍵 entries 快取＋索引）→ DRAM（pinned buffer＋memmap 被動快取）→ SSD（完整 KV 的持久備份）。頻寬實機，無模擬。<br>**寫入**：prefill 結束時，用最後 16–64 個 token 的注意力（SnapKV 式）算重要度；全部 KV 先寫 SSD，最高分的放 HBM，次高放 DRAM（importance-guided warm-up）。SSD 版面依語意連續性與 layer–head 分段打包成 extent。<br>**逐出**：GPU 快取用 lookahead（當步注意力最低者先丟）；每個 layer–head 的視窗大小以多選擇背包問題（MCKP）離線求解〔原文〕 | [K] p.9–14 §5–7 |
| 工作集／容量 | 原文：batch 8、100K context 的 Llama-3-8B KV 約 100 GB；資料中心節點的 host memory 約 100 GB〔原文〕。〔計算〕128 KiB／token × 8 × 100K≈98 GiB，與原文一致；L20 上 360K×2≈88 GiB，貼近 100 GB DRAM，這就是需要 SSD 的工作點 | [K] p.5 §3；p.7；〔計算〕 |
| 主結果的工作點 | Fig 13：L20 上各 context／batch 組合的生成吞吐，對最強基線（ShadowKV）最高 +70%，圖上標的倍數 1.25–1.74×；H20／RTX 4090 為 1.23–1.53×（Fig 14）。許多基線在長 context 下 OOM 或 <1 token/s〔原文〕 | [K] p.16–17 §9.2 |
| 掃描的自變數 | context 長度；batch size；模型；硬體 3 種；視窗大小 {2,3,4}；chunk 大小 {1,4,8}；centroid 數 {2048,4096,8192}；稀疏預算 1.56%／6.25%；逐出 LRU vs LA；2D 視窗開／關；DRAM-only vs DRAM＋SSD〔原文〕 | [K] p.17–22 |
| 對手 | Original（全 GPU）、FlexGen、Quest、ShadowKV（chunk 8、outlier 48）、PQCache、MagicPIG、RetroInfer(E)（原版）、RetroInfer（關掉注意力估計）；稀疏基線一律保留 4 個 sink＋64 個 local token〔原文〕 | [K] p.15 §9.1 |
| 系統指標 | 生成吞吐（tokens/s）；GPU 快取命中率（每步關鍵 entries 由 GPU 快取供應的比例）；每層延遲分解；GPU 記憶體；prefill 延遲；抓取時間〔原文〕 | [K] p.16–22 |
| 品質指標 | RULER（13 項）與 LongBench（9 項）的任務分數，與 Full 及各稀疏基線比較〔原文〕 | [K] p.17 表 2 |
| 主要結果 | (1) 吞吐最高 1.74×（對最佳基線）。(2) GPU 快取約 80% 命中；LA 比 LRU 高 0.9–3.9 點，但 Phi-4-mini 上 Quest、ShadowKV 反而下降。(3) DRAM＋SSD 只比 DRAM-only 少約 40% 吞吐，但能支援更大 batch。(4) RTX 4090＋KVDrive 吞吐最高為 H20 全 GPU 的 3×，記憶體用量約 1/4〔原文〕 | [K] p.12；p.16–17 表 3；p.21–23 |
| 消融／敏感度／開銷 | LA vs LRU（表 3）；2D 視窗（Fig 15）；視窗／chunk／centroid 掃描（Fig 16–19）；記憶體版面（Fig 20）；DRAM-only vs SSD（Fig 21）；prefill 延遲（Fig 22）；成本效益（Fig 23）；batch（Fig 24）〔原文〕 | [K] p.17–22 |
| 重複與統計 | 未說明〔原文〕 | [K] 全文 |
| 程式碼／資料 | 論文沒附程式碼連結〔未查證〕；LongBench、RULER 公開 | [K] 全文 |
| 設計理由（原文） | 鄰近 decode 步的關鍵 entries 高度重疊，保留多步的滑動視窗比每步重抓省（Finding 1：6.25% 預算下每步傳輸由 >500 MB 降到 <12.5 MB）；選取與抓取佔近一半時間且循序執行，造成 stall（Finding 2）；GPU–SSD 頻寬遠低於 GPU–DRAM，現有系統無法有效使用 SSD（Finding 3）；prefill 最後的注意力可以預估長期重要度，所以用來決定初始放哪層 | [K] p.5–7；p.13 |
| 設計理由〔判讀〕 | 吞吐實驗刻意把 batch × context 推到 DRAM 邊緣（100 GB 級），SSD 層才有存在價值；倍數主要來自對稀疏卸載基線的管線改進，不是分層本身 | — |
| 原文沒講清楚的地方 | 主吞吐實驗（Fig 13–14）有沒有用到 SSD；吞吐實驗的稀疏預算與輸入資料；SSD 頻寬與數量；240K／360K 的準確度（只量到 128K）；§3 說會在 §9.1 列出基線「關掉哪些輔助最佳化」，但 §9.1 沒有完整列出 | [K] p.5；p.15–17 |
| 與既有整理不一致 | ① ✅ workloads_eval §1.1（L75）：已核對一致（模型、60K–360K、LongBench／RULER、無到達、8 個基線、LRU vs LA、L20／H20／4090、§2.1 的命中率 70.0–91.0%，即 [K] p.17 表 3 全部方法的範圍）。<br>② ⚠️ 部分成立：intro.txt §7.1「KVDrive 到 360K，但是**單一請求**、沒有跨請求重用」：「沒有跨請求重用」正確；「單一請求」不精確，主吞吐實驗（Fig 13）在 360K 是 batch 2、其他長度 batch 4–8 的離線批次（每個請求有自己的 context，[K] p.16）。但 Fig 23 的成本分析確實有 360K、bs=1 的點（p.22），所以「360K 單一請求」也有原文對應〔複核修正：原卡沒提 Fig 23〕。建議改寫成「每個請求各自一段長 context、彼此不共享」。<br>③ ✅（加註建議，非針對使用者文件）準確度：原文稱 "preserving accuracy"（p.1、p.23），且 §9.3 的比較對象是其他卸載系統而非 Full；表 2 的 RULER 平均在 Qwen3-8B（73.94→68.07，−5.9）與 Phi-4-mini（64.69→59.91，−4.8）比 Full 低約 5–6 點〔複核修正：V04 由表 2 重算為 4.8 與 5.9 點〕；Llama-3-8B-1048K 為 77.74→76.15（−1.6）；LongBench 差距小（39.85→39.00、31.89→32.39、46.34→45.87）。引用「無損」要加註 | [K]；workloads_eval；intro.txt |
| 對本研究的意義〔判讀〕 | **最相關的一點**：prefill 結束時依注意力重要度決定每段 KV 放 HBM／DRAM／SSD（全部先寫 SSD），是「寫入時決定狀態」的直接前例（雖然是單請求 decode、有損稀疏注意力、依重要度而非位置）；查新時必須引用並劃清界線（有損稀疏 vs 精確前綴重用、單請求 vs 跨請求）。它的未來工作也明說要做分層混合精度（HBM FP16、SSD INT4），與本研究的精度維度重疊。**可沿用**：「同一 GPU 預算下 LRU vs 新逐出」的表格形式。**要小心**：24 GB（4090）上的吞吐比較（Fig 14b）用的是 Phi-4-mini；Fig 23 的成本分析則在 4090 上跑 Llama-3-8B-1048K（60K–360K），對照的是 H20 全 GPU 而非其他卸載系統〔複核修正：原寫「4090 上的結果是 Phi-4-mini 等小模型」，漏了 Fig 23，[K] p.16、p.22〕 | — |

---

## 本組對 PoC 設計的建議〔判讀〕

1. **把「工作集／各層容量」做成第一個自變數，並放在報告第一張表。** 範本：Bidaw 的「同時快取 KV 達性能層 3.91×、性能層容納 40.1%」，CachedAttention 的 RCC／CCpUT，Mooncake 的「本地 3M token vs 理論飽和 50M token」。沒有容量壓力時（Strata 主實驗約 95% 命中），寫入決策沒有作用空間。
2. **寫入策略當成第二個自變數。** 至少包含 write-back、write-through、selective（門檻 2）。SGLang HiCache 的上游預設（write_through、kernel I/O、page_first、host 池為 device 的 2 倍）當「生產預設」基線；LMCache（全寫＋LRU）當第二個生產基線。Tutti 的「讀寫並行頻寬降 60%」說明寫入成本要計入。
3. **負載分兩類並明講哪段是真實、哪段是合成。** (a) Mooncake FAST25 的 Conversation／Tool&Agent（真實時間，可仿 Strata Fig 12 縮放時間戳做負載掃描）；(b) 長文件重查詢（LooGLE／NarrativeQA／LEval），用 Strata 的 min／shuffle／max 控制重用距離。到 128K 以上就只剩合成或抽樣構造，必須標明。
4. **SSD 要實測。** 若用延遲注入模擬更快的裝置（Bidaw 的做法），要與實機交叉驗證並分開報；頻寬掃描若是模擬，寫在主結果旁邊，不要埋在附錄。
5. **倍數的定義寫死，報整條曲線。** 建議同時報「同 TTFT 下的吞吐」與「固定負載下的 SLO 達成率」，標出基線飽和點與 SLO 門檻；每個點至少 3 次重複並附信賴區間。本組 8 篇沒有一篇報重複次數，這是最容易拉開差距的地方。
6. **對照組要包含兩種載入方向與寫入時放置的前例。** Cake（重算前段、載入後段）與 MTDS（載入前段、重算後段）方向相反，兩者都放；KVDrive 的「prefill 結束時依重要度決定放哪層」是寫入時放置的直接前例，查新與對照都要處理。
7. **跨論文引用時標版本。** Mooncake（arXiv v4 vs FAST'25）與 Strata（arXiv v1 vs OSDI'26）的數字與設定差很多；使用者文件與 workloads_eval 已出現混用。

---

## 未查證清單

- Bidaw 公開 trace 的「含歷史 prompt 中位 1,066、p99 8,168」：沿用 workloads_eval 的〔計算〕，本卡未重算。
- Bidaw 的 users/min 產生方式與截斷方式；Bidaw、CachedAttention、KVDrive 是否另有程式碼釋出。
- Mooncake FAST Fig 1 的 +498%／+157%／+59% 與門檻 I／II／III 的對應：依圖上標籤位置判讀；百分比是對哪個基線，原文沒寫。
- LMCache 論文所用 v0.3.6 的逐出策略：只查了 2026-10-05 dev 分支的預設（LRU），沒查 v0.3.6 tag。
- SGLang HiCache 與論文 Strata 的完整對應：只查了參數、預設值與 selective 門檻。
- Tutti repo 的內容與可重現性（只查 metadata）；Tutti 128K 時 155.743 s 與 640K 時 1.2 s 的落差原因。
- MTDS 的 vLLM 版本、Mooncake 部署方式、NoT 數值、「LLaMa-3 8B」是 Llama-3 還是 3.1。
- Strata 工作集估計以平均 input 近似文件長度，只是量級估計；GPU KV 池大小原文沒給。
- Cake「約 80% 命中在磁碟」的由來：CachedAttention 三個版本都沒有這句；「靠 SSD 容量才得到的命中佔 79%–98%」是本卡由 Fig 24 推算〔計算〕，Cake 是否如此推得為〔判讀〕。
- ~~DeepSeek-V3 為 MLA：沿用 workloads_eval（HF config），本卡未重查。~~〔複核：V04 已重查 HF config，`kv_lora_rank`=512，移出清單〕
- Bidaw 本身建在哪個推論引擎上（原文只說兩個對手在 vLLM 上重做）〔複核補充〕。
- MTDS 是否另有程式碼釋出（原文無聲明）〔複核補充〕。
- Mooncake 獲 FAST'25 Best Paper：來自 GitHub README〔文件〕，未查 USENIX 官方頁。
- Bidaw「query 平均 36、response 平均 45」的單位：原文沒標，未以公開 trace 交叉驗證。

---

## 抽取紀錄

- 抽取者：E04（子 agent），2026-10-06。
- 原文取得：USENIX 版以 `curl` 下載官方 PDF；arXiv 以 `arxiv.org/pdf/<id>` 下載最新版（Strata v1、LMCache v2、Mooncake v4、CachedAttention v3 並另取 v1／v2、Tutti v1、KVDrive v1）；MTDS 見來源清單。全部以 `pdftotext -layout` 轉文字，依換頁符號標 PDF 頁碼後通讀。
- 〔計算〕方法：每 token KV＝2（K、V）×層數×KV head 數×head 維度×2 bytes（BF16），層數與頭數取自 HF config；其餘計算在各格內寫明算式。
- 核對的使用者文件：`scratchpad/intro.txt`、`scratchpad/sota.txt`；既有整理：`docs/research_20260924/workloads_eval.md` §0、§1.1、§1.3、§2.1–2.2、§5.2–5.3，`docs/PAPERS_BY_LEVEL.md` 的 L2／L3 各條。

---

## 複核紀錄

- **複核者**：V04（子 agent，未參與抽取，沒讀抽取者的推理或筆記）。**日期**：2026-10-07。
- **原文來源與版本確認**：使用 `scratchpad/E04/` 的原始 PDF，自行以 `pdftotext -layout` 逐頁重轉到 `scratchpad/V04/`（不用抽取者的 txt）。arXiv 版本由頁側浮水印確認：Strata v1（2025-08-26）、Mooncake v4（2025-09-03）、LMCache v2（2025-12-05）、CachedAttention v3（2024-06-30，另 v1、v2）、Tutti v1、KVDrive v1；USENIX 版以封面確認會議與年份；MTDS PDF 的 MD5 與 `papers/mtds2026.pdf`、`~/Downloads/s40747-025-02200-4.pdf` 相同。Cake 引用句以 `papers/cake2025.pdf` p.2 重轉確認。程式碼：以 `raw.githubusercontent.com/<repo>/<commit>/…` 重抓 SGLang `542addad` 的 `memory.py`、`unified_radix_cache.py` 與 LMCache `8c77a6f7` 的 `config.py`，行號與內容一致。HF config 重抓 27 個模型。GitHub API 重查 `xPU-IO/Tutti` 與 Bidaw trace repo 的 metadata。
- **判定標準**：✅ 與原文一致（含只補頁碼或補充說明）；❌ 有錯，已改並標〔複核修正〕；⚠️ 原文找不到或把〔判讀〕寫成原文，已改成〔未查證〕或補標〔判讀〕。〔判讀〕格只檢查有沒有把推論寫成事實；〔計算〕格全部重算。

### 每張卡的檢查結果（每卡 26 格＝3 個開頭欄位＋23 列）

| 卡 | 檢查格數 | ✅ | ❌ | ⚠️ |
|:--|--:|--:|--:|--:|
| Strata | 26 | 22 | 3 | 1 |
| Bidaw | 26 | 22 | 2 | 2 |
| MTDS | 26 | 24 | 1 | 1 |
| LMCache | 26 | 25 | 0 | 1 |
| Mooncake | 26 | 23 | 2 | 1 |
| CachedAttention | 26 | 26 | 0 | 0 |
| Tutti | 26 | 21 | 4 | 1 |
| KVDrive | 26 | 24 | 2 | 0 |
| 本組的共同模式（6 條） | 6 | 4 | 0 | 2 |
| 來源清單（13 列） | 13 | 13 | 0 | 0 |
| **合計** | **227** | **204** | **14** | **9** |

〔計算〕重算全部吻合：Strata 工作集 277.0／334.5 GiB；Bidaw 5 GB/s 模擬下 +33%／+9%；Mooncake 1／0.57≈1.75；CachedAttention Fig 24 推算 96.0／97.6／91.3／78.8%；Tutti 128K≈16 GiB、256 GB≈2M token；KVDrive 8×100K≈98 GiB、360K×2≈88 GiB；各模型每 token KV 由 HF config 重算一致。

### 逐條修改（原內容 → 新內容；出處）

**Strata**
1. 模型：DeepSeek-V3 為 MLA〔二手〕→〔文件〕，V04 重查 HF config `kv_lora_rank`=512。
2. 工作集／容量：補標「H200 141 GB」與「與 95% 命中一致」為〔判讀〕（原文沒寫 141 GB）。
3. 掃描的自變數：Fig 5 CUDA block 數「1–128」→「0–128」（[S] p.6 橫軸）。
4. 系統指標：命中率「正規化呈現」→「Fig 10、12 正規化，Fig 2 為原始值」（[S] p.4、p.11、p.13）。
5. 與既有整理不一致 ③：「intro.txt 表 6／表 7、§8.3」→「表 7 腳註 e、表 8」；intro 表 6 與 §8.3 都沒提門檻 2。其餘各點補 ✅ 判定與頁碼；⑥ 改為「部分成立」（端到端確實都是 Poisson，[S] p.10）。

**Bidaw**
6. 軟體與版本：「原型建在 vLLM 上」→「Bidaw 本身建在哪個引擎上原文沒寫〔未查證〕」；原文只說對手在 vLLM 上重做（[B] p.11）。
7. 重用結構：「不跨使用者共享」補出處 p.14 §6。
8. 主結果的工作點：「出自基線延遲暴增的點」補標〔判讀〕（原文只說 "For OPT-13B ... up to 3.58×"）。
9. 與既有整理不一致 ③：「sota §2.5 **過度簡化**、不是單調的」→「⚠️ 部分成立」。原文 p.6 自己寫 "predict the user future access timing"；對同一使用者，下界越大，命中潛力越低（p.9 式 2），所以方向成立。
10. 與既有整理不一致 ④：原指控 sota「Poisson 就失效」不準 →「❌ 指控不成立」。sota 的主語本來就是「回答越長、越晚回來」這個訊號，與原文 p.12 一致。
11. 與既有整理不一致 ⑤：「OPT 視窗 2K，實際 ≤2K」→ 補上 Qwen-7B 的 `seq_length` 為 8,192，≤2K 只對 OPT／Qwen-14B 成立；⑥ 補註「36 token」的單位是使用者文件自加。
12. 對本研究的意義：「2K 視窗」→「2K–8K 視窗」。

**MTDS**
13. 主張：補上原文自身不一致：摘要與結論寫 "up to 20%"（p.1、p.15），引言與實驗寫 "more than 20%"（p.2、p.12）。
14. 程式碼／資料：「程式碼未釋出」→「原文沒有提供程式碼連結或可得性聲明〔未查證〕」（p.15 只有 Data Availability）。
15. 與既有整理不一致 ②：原指控「sota『從 SSD 載入比重算慢』應改成從 DRAM」→「⚠️ 部分成立」。原文 p.1 寫 "reloaded from DRAM or SSD ... may exceed recomputation time"；但機制與 Fig 6 是 DRAM→GPU。
16. 與既有整理不一致 ③：「等於載入前段、重算後段」補標〔判讀〕，並註明 intro 表 7 腳註 l 已寫「可以只載入前段」。

**LMCache**
17. 模型：補註 "Sao10K-L3-8B" 原文沒給確切 repo，8 KV head 是推定。
18. 分層實作：補註 Fig 7 的 "Least recently used" 是 vLLM 空閒 page 池，不是 LMCache 後端逐出（[L] p.8）。
19. 與既有整理不一致：各點補 ✅ 判定與頁碼（④ 的 "most widely adopted" 出自結論 p.16；⑤ 的四種處理器出自 p.15–16）。

**Mooncake**
20. 主張：「比 vLLM 高 59%–498%」→「比基線方法高」（摘要原文 "compared to baseline methods"，[MC-F] p.2）。
21. 主結果的工作點：「倍數是兩系統比例之比〔原文〕」→〔判讀〕；「不是量到的最大負載差」補標〔判讀〕。
22. 主要結果：(4) 的頁碼 p.14 → p.13（§5.4.3）。消融格補 Fig 5 在 p.8。
23. 與既有整理不一致：① 到 ⑤ 補 ✅ 與頁碼；⑤ 補「arXiv 摘要自己就寫 'enables Kimi to handle 75% more requests'，使用者措辭忠於 arXiv，錯在版本混用」；⑥ 判定為「⚠️ 非錯誤，僅補充」（與原文 p.2 的架構敘述一致）；⑦ 補「FAST 版本文沒寫 512」；② 補「workloads_eval L377 標 p16，75% 實際在 arXiv p.17」。

**CachedAttention**
24. 無錯誤。與既有整理不一致 ① 到 ⑤ 補 ✅ 與複核出處（Fig 2 圖說 p.4；99.6%／99.9% 在 v1、v2、v3 的 p.11 與 ATC p.12）。

**Tutti**
25. 主結果的工作點、主要結果 (1)、與既有整理不一致 ④、對本研究的意義：「最高負載點」→「高負載點」。新版 vLLM 段落原文是 "at high load"；明寫 "at the highest load point" 的是舊版對 GDS 的 71.8%（[T] p.10）。同一錯誤出現在 4 格。
26. 工作集／容量：「若某層是最低層」補標〔判讀〕。
27. 消融：「讀寫並行頻寬降 60%」補出處 p.6（Fig 6 為 60.1%）。
28. 與既有整理不一致 ②、③：補 ✅ 判定與出處；② 加註「只處理搬運、不處理放置」這層意思成立。

**KVDrive**
29. 與既有整理不一致 ②：「單一請求不準確」→「⚠️ 部分成立」。Fig 23 確有 360K、bs=1 的點（[K] p.22）。
30. 與既有整理不一致 ③：「約 5–6 點」→ 重算為 −5.9（Qwen3-8B）與 −4.8（Phi-4-mini），並補 Llama −1.6 與 LongBench 三組數字。
31. 對本研究的意義：「4090 上的結果是 Phi-4-mini 等小模型」→ 補上 Fig 23 在 4090 上跑 Llama-3-8B-1048K（[K] p.22）。

**本組的共同模式**
32. 第 4 條：「Bidaw 推定 ≤2K（OPT 視窗）」→ 補 Qwen-7B 8K；補註 LMCache 多輪 QA 的到達分布沒寫；補註 Strata（Fig 12 模擬）與 LMCache（壓縮時間）也用到真實 trace 時間。
33. 第 6 條：「SGLang-HiCache 是作者自建的弱化版」→「弱化版」標〔判讀〕（原文稱 "state-of-the-art layer-wise ... implementation"）；補註 CA 與 Tutti 的 GPU 單價相同，但 GPU 型號不同。

**未查證清單**：移除「DeepSeek-V3 為 MLA」（已查證）；新增「Bidaw 本身的推論引擎」與「MTDS 是否釋出程式碼」。

### 對使用者文件與 workloads_eval 的指控：判定總表

| 卡 | 指控 | 判定 |
|:--|:--|:--|
| Strata | ② 引「門檻 2」「各層 LRU」須引 OSDI 版 | ✅ |
| Strata | ③ SGLang 上游預設是 write_through，「生產做法」要區分 | ✅（引用位置已更正） |
| Strata | ④ 「只用 LRU」→「各層預設 LRU」；「SGLang 生產環境」措辭 | ✅（措辭補正） |
| Strata | ⑥ PAPERS_BY_LEVEL「全部用 Poisson」 | ✅ 部分成立 |
| Bidaw | ① workloads_eval「以抽樣使用者調整 users/min」 | ✅ |
| Bidaw | ③ sota／PAPERS_BY_LEVEL「回答越長越早移 SSD」過度簡化 | ⚠️ 部分成立（原卡說得太重） |
| Bidaw | ④ sota「Poisson 就失效」不準 | ❌ 不成立 |
| Bidaw | ⑤ intro 表 7「8.2K」不是評測長度 | ✅ |
| MTDS | ① workloads_eval 漏 16K→24K | ✅ |
| MTDS | ② sota「從 SSD 載入比重算慢」應改 DRAM | ⚠️ 部分成立（原卡說得太重） |
| MTDS | ④ intro 表 7 可填 24K | ✅ |
| LMCache | ① workloads_eval 漏 Qwen2.5-72B | ✅ |
| LMCache | ④ sota「業界最常用」應標為原文自稱 | ✅ |
| LMCache | ⑤ PAPERS_BY_LEVEL「NVIDIA 專屬、AMD 跑不起來」 | ✅ |
| Mooncake | ① workloads_eval 的 FAST 列內容其實是 arXiv v4 | ✅ |
| Mooncake | ② workloads_eval 的 525%、75% 是 arXiv 數字 | ✅ |
| Mooncake | ④ intro「128K 是模擬資料」只適用 arXiv | ✅ |
| Mooncake | ⑤ intro／sota／PAPERS_BY_LEVEL「多處理 75%」版本混用 | ✅ |
| Mooncake | ⑥ 「池化 CPU 與 SSD」 | ⚠️ 非錯誤，僅補充 |
| CachedAttention | ① workloads_eval「session ≤32K、≤40 輪」是圖的顯示範圍 | ✅ |
| CachedAttention | ② intro「session 最長 32K」 | ✅ |
| CachedAttention | ③ intro「約 80% 命中在磁碟層」原文不支持（原文是 >99.6% 在 DRAM） | ✅ |
| CachedAttention | ④ workloads_eval Cake 列需加註 | ✅ |
| Tutti | ② intro「到 200K、只處理 SSD→GPU」 | ✅（「只處理搬運」這層意思成立） |
| KVDrive | ② intro「360K 單一請求」 | ⚠️ 部分成立（Fig 23 有 360K、bs=1） |

### 因時間沒有檢查或只部分檢查的

- 沒有重新上網下載 USENIX／arXiv PDF。只用 `scratchpad/E04/` 的原始 PDF，以浮水印、封面與 MD5（MTDS）確認版本；URL 本身是否仍有效未重測。
- Mooncake arXiv v4 只用關鍵字檢索並讀了 §4、§8 與 Table 1–2，沒有通讀 §5–§7（prefill pool、排程、拒絕策略的細節）。
- Strata arXiv v1 只讀了平台、§5.1、§5.3.4，並全文檢索寫入策略與 LRU，沒有通讀。
- KVDrive Fig 15–22 的數值只能從圖讀，pdftotext 取不到，只核對了內文敘述；Fig 13／14 的倍數標籤以 `pdftotext -raw` 取出（1.25–1.74×、1.23–1.53×）。
- SGLang `542addad` 與 LMCache `8c77a6f7` 的 commit 日期（2026-10-06、2026-10-05）沒有查，只確認檔案內容與行號。
- 沒有交叉驗證 Bidaw 公開 trace 的「36／45」單位，也沒有重算 workloads_eval 的 1,066／8,168（維持〔未查證〕）。
- 「本組對 PoC 設計的建議」整節是〔判讀〕，只核對了其中引用的數字（3.91×、40.1%、3M／50M token、95%、上游預設、60%），全部一致。
