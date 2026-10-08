# E03｜讀取時的還原：重算 vs 載入（評測卡）

- **抽取者**：子 agent E03，2026-10-06
- **範圍**：使用者回來時，KV 不在 GPU 上，怎麼「拿回來」：全部重算、全部載入，或兩者並行。7 篇：Cake、CacheFlow、Pensieve、HCache、KVPR、Bottlenecks、py-kvcache。每張卡除了 README 的固定欄位，另加三列：**I/O 怎麼實現**、**重算與載入成本怎麼量**、**長度範圍與停止理由**。
- **本組特別產物**：Cake 評測卡之後附「Cake 重現規格卡」，供下一步在 vLLM 上重現 Cake 式還原＋限速器。
- **頁碼**：一律是 PDF 頁（第 1 頁＝封面頁），不是期刊印刷頁；KVPR 另附印刷頁。

## 來源清單（全部於 2026-10-06 下載並讀過，暫存於 scratchpad/E03）

| 簡稱 | 讀的版本 | URL | 備註 |
|:--|:--|:--|:--|
| Cake | **ICML'25 正式版**（PMLR 267:28031–28043，13 頁）；**arXiv v2**（2025-02-20，12 頁）；arXiv v1（2024-10-04，略讀做版本比對） | https://proceedings.mlr.press/v267/jin25d.html ；PDF：https://raw.githubusercontent.com/mlresearch/v267/main/assets/jin25d/jin25d.pdf ；https://arxiv.org/abs/2410.03065v2 ；https://arxiv.org/abs/2410.03065v1 | 版本差異見 Cake 卡 |
| CacheFlow | **arXiv v2**（2026-09-26，17 頁）全文；v1（2026-04-28，11 頁）比對關鍵段落 | https://arxiv.org/abs/2604.25080v2 ；https://arxiv.org/abs/2604.25080v1 | v1 與 v2 方法不同（見卡） |
| Pensieve | arXiv v3（2024-10-07，15 頁，含 EuroSys'25 ACM 版權格式與 DOI 10.1145/3689031.3696086，判讀為定稿版） | https://arxiv.org/abs/2312.05516v3 | |
| HCache | arXiv v1（2024-10-07，16 頁，含 EuroSys'25 ACM 版權格式與 DOI 10.1145/3689031.3696072，判讀為定稿版） | https://arxiv.org/abs/2410.05004v1 | arXiv 只有 v1 |
| KVPR | ACL Findings 2025 正式版（PDF 15 頁，印刷頁 19474–19488） | https://aclanthology.org/2025.findings-acl.997.pdf ；程式碼 https://github.com/chaoyij/KVPR | arXiv 2411.17089 未比對 |
| Bottlenecks | arXiv v1（2025-12-16，14 頁）；arXiv 註記「Submitted to MLSys 2026」 | https://arxiv.org/abs/2601.19910v1 | **MLSys'26 錄取未查證**，見卡 |
| py-kvcache | arXiv v1（2026-09-10，24 頁）；程式碼 commit `3abba7a`（2026-07-26） | https://arxiv.org/abs/2609.11744v1 ；https://github.com/atlarge-research/py-kvcache | |

---

## 本組的共同模式（跨論文比較）

1. **I/O 的真實度是一條光譜，而且沒有人校準模擬**〔原文＋判讀〕。
   - 延遲注入：Cake（依 chunk 大小 ÷ 頻寬算延遲，到時才放行；ICML p5 §5.1）。
   - 沒說：CacheFlow（只說「10/40/80 Gbps bandwidth conditions」，Fig. 5 用「% of cap」；v2 p7–8）。
   - 真實 CPU DRAM＋PCIe：Pensieve（p9）、KVPR（PCIe 4.0 ×16，p6）、Bottlenecks（PCIe 5.0，p7）。
   - 真實 NVMe：HCache（4×PM9A3＋SPDK/GDRCopy 直通，p8–9）、py-kvcache（O_DIRECT＋io_uring，p9–16）。
   - 用真實裝置的兩篇都量到「規格頻寬≠有效頻寬」：Bottlenecks 只用到 PCIe 5.0 單向峰值的 23%（15 GB/s，p8）；py-kvcache 的 llm-d 讀 9–12 GB/s、寫約 2 GB/s，staging buffer 讓 4.4 GB 的有效資料搬了約 10 GB（p12–13）。**用延遲模擬的 Cake 沒有任何一點對照真實裝置。**
2. **重算成本有五種量法，只有三篇把「位置」放進去**〔原文〕。
   - 逐 chunk 實測：Cake Fig. 4（每個 512-token chunk 的 step time 隨 index 線性上升，p4）。
   - 離線 profile＋內插：Pensieve（2 的冪次 context，32-token chunk 的 attention 成本，p6）。
   - 線上 profile＋內插：CacheFlow（從近期服務紀錄更新 f(j)，p6）。
   - 解析式 FLOPs：HCache（p5）、KVPR（p5）、Bottlenecks（Fpf＝2N，**刻意省略 context 項**，p4、p10）。
   - 端到端 TTFT 掃描（Pareto capture）：py-kvcache（p8、p11）。
   - 依位置：Cake、Pensieve、CacheFlow。HCache、KVPR 的「重算」是從存下的 activation 做投影，沒有注意力項，成本跟位置無關；Bottlenecks、py-kvcache 把整段前綴當一個量。
3. **評測長度都 ≤128K，多數 ≤16K**〔原文〕〔複核修正：原寫「停在 128K 以下」，CacheFlow Fig. 7 正好到 128K（v2 p8），改為 ≤128K〕：KVPR ≤1K＋128 生成；Cake 主表 4K–16K（動機圖 32K）；Pensieve 上限 16,384；HCache 主實驗 ≤16K（OPT-30B 敏感度到 32K）；Bottlenecks 實驗 K≤65,536（只有統計用到 450K）；py-kvcache ≤80k（81,920）；CacheFlow ≤128K。停止理由多半沒明說；有說的是：資料集統計（Cake 引 CacheGen）、模型 context 上限（Cake v1：LongAlpaca 32K；HCache 把模型擴到 16K）、沿用前作設定（KVPR 沿用 FlexGen）。
4. **切分的維度不同，但決策都在讀取時或逐出時**〔原文＋判讀〕：token 維度（Cake、KVPR）、layer 維度（HCache）、token×layer 階梯＋跨 GPU（CacheFlow）、逐出順序（Pensieve 從開頭丟）、整段門檻（py-kvcache 只管「載不載」）。**沒有一篇依「位置」在寫入時決定。**〔複核修正：原寫「沒有一篇在寫入時決定」，與 HCache 不符。HCache 的 state partition（離線 profile 求 L_H、L_O）決定每一層**存成什麼**：hidden state、KV，或不存、回來時從 token 重算；原文明寫「每層的狀態存成 hidden、KV 或原始 token 之一」，且「有些層甚至不需要存」（HCache p7 §4.2、p10 Table 3：OPT-30B＝40 H＋8 RE）。所以 HCache 有寫入時的決定，只是以「層」為單位、依硬體離線定死，與位置無關〕 py-kvcache 程式碼的 `prepare_store` 照樣存所有新 block（`py_kvcache/vllm.py` L338–371），門檻只用在讀取（`reactor.py` L701、L737）〔程式碼 3abba7a〕。
5. **統計與重複普遍薄弱**〔原文〕：KVPR 報 5 次平均（p6）；py-kvcache 每設定 3 次取平均（p7）；Bottlenecks 每設定 200 個請求，報 mean±std（p7）；Cake、CacheFlow、Pensieve、HCache 都沒寫重複次數或誤差。Cake 的 +26% 只有一個例子（ICML §5.8，p8）。
6. **「只算」「只載」兩個基線的實作會漂**〔原文〕：只算＝vLLM chunked prefill（Cake 用 v0.6.2；CacheFlow 用 v0.29.0）；只載＝LMCache（Cake 用 v0.1.4；CacheFlow 用 v0.5.5rc5；Bottlenecks 用 v0.3.5；py-kvcache 用 v0.4.7）。HCache 不用 vLLM，而是在 DeepSpeed-MII v0.2.0 上重做 AttentionStore。py-kvcache 量到同一份 LMCache 與 vLLM Offload 在 ShareGPT 上 TTFT 差 1.61×（206 vs 128 ms，p10），差異主要來自 store 路徑的位置。**基線的版本與路徑本身就是一個大變因。**

---

## 卡 1｜Cake：Compute or Load KV Cache? Why Not Both?（ICML 2025；arXiv 2410.03065）

- **讀了什麼**：〔全文〕ICML 正式版 PDF（PMLR v267）＋ arXiv v2 全文；arXiv v1 只比對關鍵段落。查證 2026-10-06。
- **一句話**：長 context 的 KV 存在慢儲存層時，讓 GPU 從前往後重算、I/O 從後往前載，兩邊在中間會合。
- **評測要證明的主張**：在不同 GPU、GPU 可用比例、I/O 頻寬、長度、模型與量化下，雙向並行的 TTFT 都比只算或只載短，平均 2.6×；遇到資源波動時會合點會自動調整；開銷可忽略（ICML p1 摘要、p9 結論）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | LongAlpaca-7B、LongAlpaca-13B（LLaMA2 架構，MHA）；LLaMA 3.1-8B、LLaMA 3.1-70B（GQA）。權重 FP16，只有 70B 因硬體限制用 FP8 權重；KV 預設 BF16。Table 1 的 KV／token：7B 512 kB、13B 800 kB、8B 128 kB、70B 320 kB。〔計算〕Table 1 把 13B 寫成 32 個 KV head，但 800 KiB 要 40 個 head 才算得出來（2×40 層×40×128×2 B＝819,200 B），判讀 32 是誤植 | 〔原文〕ICML p5 §5.1、Table 1 |
| 硬體 | 伺服器 1：2×A100 80GB（NVLink）、64 核 AMD EPYC 7763、2.0 TB 記憶體；伺服器 2：1×H100、26 核 vCPU、200 GB 記憶體。表中另有「1×A100」設定，判讀為伺服器 1 只用一張卡。沒有真實儲存裝置參與主實驗 | 〔原文〕ICML p5 §5.1、p6 Table 3 |
| 軟體與版本 | 擴充 LMCache 並整合進 vLLM，約 1,000 行：非同步 get、預先配置 chunk buffer、每個 engine step 後檢查下一個 chunk 是否已在 resident dictionary。只算基線：vLLM v0.6.2 chunked prefill；只載基線：LMCache v0.1.4。PyTorch、CUDA 版本**未說明** | 〔原文〕ICML p5、App. B p12–13 |
| 資料／負載 | 用 LongChat（多輪）、TriviaQA、NarrativeQA 只為了取長度範圍：依 CacheGen 的統計，多數查詢落在 4K–16K。prompt 是合成的，原文理由是「only token length matters」。token 內容怎麼生成沒寫；動機實驗（Fig. 3）寫的是「random context of 32k tokens」 | 〔原文〕ICML p5 §5.1、p3 Part 1 |
| 長度 | 主表 4K–16K。文字寫「每 2K 取一點」，但 Table 5 只列 4K/8K/12K/16K，其餘表固定 16K。動機的 Fig. 3、Fig. 4 用 32K。arXiv v1：文字說在 5K–16K 間均勻取 20 點、另加 32K 壓力測試（v1 p7）；實際圖上是 Fig. 5a 的 32K、Fig. 5b 的 2K–32K（v1 p7），§5.3–5.4 的 Fig. 6 用 14K（v1 p8）〔複核修正：原寫「v1 的主實驗是 2K–32K 與 32K」，漏了 v1 文字的 5K–16K 與 14K 的實驗〕 | 〔原文〕ICML p5、p7 Table 5、p3–4；v1 p7–8 |
| I/O 怎麼實現 | **延遲注入**：依 chunk 大小與「網路頻寬」算出延遲，模擬的儲存後端暫停資料傳到 GPU，直到延遲到期（ICML 措辭）。v1 的說法是讓 LMCache sleep 到期後才把資料放進 CPU 記憶體。Appendix A 的演算法是「從儲存讀到 CPU 記憶體」。**CPU→GPU 這一段是否真實搬運、有沒有算進 TTFT，原文沒講清楚。** I/O chunk＝128 token。頻寬點 7、25、32、56、100 Gbps，各對應一種實體設定（Table 2）。**沒有和真實裝置校準** | 〔原文〕ICML p5 §5.1、p6 Table 2、p12 App. A；v1 p7 |
| 重算與載入成本怎麼量 | 重算：vLLM chunked prefill（chunk＝token budget，預設 512）。Fig. 4 量了每個 512-token chunk 的 step time 隨 chunk index 的變化：A100 約 40→110 ms、H100 約 27→93 ms，跨約 64 個 chunk（≈32K）；每 chunk 的 KV 記憶體固定（約 256 MB）〔原文圖讀值〕。〔複核補充：以 800 dpi 數位化 Fig. 4（座標軸刻度校準），A100 為 38.8→112.0 ms、H100 為 27.4→93.5 ms（chunk 0→63）、SSD 線 62.4 ms、記憶體柱 255 MB；Fig. 4 本身沒寫模型與 GPU 數，256 MB／512 token＝512 KiB／token 只對得上 LongAlpaca-7B（Table 1），判讀為 7B 單卡〕載入：Fig. 4 的「SSD 32 Gbps」是一條水平線（約 62 ms／chunk）。Fig. 3 的「等效吞吐」＝KV 檔案大小 ÷ 處理時間：HDD ~200 MB/s、A100 ~3.3 GB/s、H100 ~4 GB/s、NVMe SSD ~4 GB/s、Network ~1 GB/s（LongAlpaca-7B、32K、chunk 512）。Fig. 3 的儲存欄是量測還是規格值，沒寫 | 〔原文〕ICML p3–4 Fig. 3–4 |
| 到達與併發 | 主實驗是單一請求、沒有到達過程。「GPU 使用率」用 vLLM token budget 的佔比模擬：budget 512 中 Cake 拿 256、其餘給別的使用者＝50%（12.5%／50%／87.5%／100%）。§5.7：計算 budget 在 0–512 間均勻隨機、頻寬在 0–25 Gbps 間隨機，取樣週期沒寫。§5.8：1 個 16K 的 prefix 請求＋22 個 32–448 token 的請求（「spiked distribution」沒有定義） | 〔原文〕ICML p6、p8 |
| 重用結構 | 單一請求的完整前綴命中；KV 全部事先算好、存好（「we precompute and store all requests' KV cache in advance」，p5） | 〔原文〕ICML p5 |
| 掃描的自變數 | 硬體（2×A100 TP、1×A100、H100）× GPU 使用率（4 級）× 頻寬（5 點）（Table 3）；chunk／token budget 64–2048（Table 4，**只在 ICML 版**）；長度 4K–16K（Table 5）；模型 4 種（Table 6）；KV 16/8/3-bit（KVQuant，Table 7）；資源波動（Fig. 5）；adaptive scheduling（Fig. 6）；開銷（Fig. 7） | 〔原文〕ICML p5–9 |
| 對手 | 只算（vLLM v0.6.2 chunked prefill，budget 512）、只載（LMCache v0.1.4）。沒有和 CacheGen、AttentionStore 等其他快取系統比 | 〔原文〕ICML p5 |
| 系統指標 | TTFT（請求到達到第一個 token）；表中是「對只載的加速 \ 對只算的加速」。加速比的定義（平均 TTFT 的比值？單次？）沒寫。§5.8 用完工時間與吞吐 | 〔原文〕ICML p5–6、p8 |
| 品質指標 | 無（無損載入，不量品質） | 〔原文〕全文無品質實驗 |
| 主要結果 | ① 摘要與結論：對兩個基線平均 2.6×，計算方式沒寫（p1、p9）。② Table 3（LongAlpaca-13B、16K、chunk 512）：2×A100、100%、32 Gbps 時為 2.63×\1.59×；排除紅色極端值（對某一基線 >10×）後平均 2.23×\3.76×（p6）。③ Table 6：Llama-3.1-70B、12.5%、32 Gbps 時對只載為 **0.80×**，是四個沒贏過基線的點之一（p8）；Table 5：2×A100、7 Gbps、100%、4K 時對只算 0.99×（p7）。④ §5.8：完工時間 1.5→1.19 s，吞吐 +26%，只有一個例子（p8） | 〔原文〕ICML p1、p6–9 |
| 消融／敏感度／開銷 | chunk 大小（Table 4，平均 1.96×\2.25×）；長度（Table 5，平均 3.34×\2.24×）；架構 MHA vs GQA（Table 6，平均 2.68×\3.01×）；量化（Table 7）；波動追蹤（Fig. 5）；排程（Fig. 6）；per-step 開銷（Fig. 7：A100/H100 的 step time 有無 Cake 幾乎重疊） | 〔原文〕ICML p7–9 |
| 重複與統計 | 未說明；沒有誤差棒、沒有重複次數 | 〔原文〕全文 |
| 程式碼／資料 | **沒有找到公開程式碼**。查證：arXiv 頁、PMLR 頁、ICML poster 頁都沒有 code 連結；一作頁 shuoweijin.com/publication/cake 與二作頁 xenshinu.github.io 的 Cake 項都只連到 arXiv；GitHub 搜尋「compute or load kv cache」無結果。二作的 LMCache fork（xenshinu/LMCache）有 2024-09 的 `fast_serde.py` 修改與 2025-01 的 `local_backend.py` 修改（patch-1 分支），**無法確認是否為 Cake 實作**。OpenReview（WOyOtaO6lQ）被人機驗證擋下，未讀到〔複核補充 2026-10-07：GitHub API 確認 xenshinu/LMCache 是 LMCache/LMCache 的 fork，xenshinu 本人的 commit 只有 2024-09-20～22 的 `fast_serde.py`／`__init__.py`（dev 分支）與 2025-01-21 的 `local_backend.py`「thread pool instead of process pool」（patch-1 分支），沒有非同步 get、resident dictionary 等 App. B 描述的元件；`gh search repos "compute or load kv cache"` 無結果〕 | 〔文件〕〔未查證〕 |
| 長度範圍與停止理由 | 原文理由：資料集查詢多在 4K–16K（引 CacheGen 統計）。v1 另說 32K 是 LongAlpaca 的 context 上限。LLaMA 3.1 支援 128K，卻沒有跑超過 16K，原文沒解釋 | 〔原文〕ICML p5；v1 p7 |
| 設計理由（原文） | chunk 級排程比 token 級更能用 GPU 平行度，又比序列級細（p3）；前算後載是因為「後面的 token 計算更貴、I/O 成本與位置無關」（p4 Insight）；I/O chunk 128 是「頻寬利用與處理粒度的經驗平衡」（p5）；延遲模擬是「為了精確控制頻寬、確保可重現」（p5）；用 token budget 佔比代表多使用者共用（p6）；排除 >10× 的點，因為對極弱基線的加速會誤導（p6）；最有利的部署是兩種資源相當、對兩個基線都約 2× 時（p6） | 〔原文〕ICML p3–6 |
| 設計理由〔判讀〕 | 4K–16K 剛好是「計算≈I/O」的區間：依 Fig. 4，32 Gbps 的 I/O 線大約在 A100 第 18–19 個 chunk（≈9.2K–9.7K）、H100 第 33 個 chunk（≈16.9K）和重算曲線相交〔複核修正：原寫 A100 第 17 個（≈8.7K）、H100 第 22 個（≈11K）。800 dpi 數位化 Fig. 4（ICML p4）：A100 在 chunk 18 為 62.2 ms、chunk 20 為 63.7 ms；H100 在 chunk 32 為 61.4 ms、chunk 33 為 62.6 ms；I/O 線 62.4 ms。H100 原值錯得最多〕。注意 Fig. 4 是 7B（見上一列），主表是 13B，交點位置不能直接搬。所以 Cake 的主表大致落在雙向並行最有利的區域。到更長的 context，重算曲線遠高於 I/O 線，會合點會往前移，加速會趨近「只載」 | 〔判讀〕依 Fig. 4 讀值 |
| 原文沒講清楚的地方 | 見下方「重現規格卡」逐條列出 | — |
| 與既有整理不一致 | ① intro §2.1 與表 3 用的「72K token、Llama2-70B、A100 約 30 秒」**只在 arXiv v1/v2 的 p1，ICML 版已刪除**；參考文獻 [1] 引的是 ICML，應改引 arXiv v2 p1。原文也沒說這 30 秒怎麼量的（v1 只寫「70B parameter model」）。② intro §7.1「16K 左右時重算和載入相近，**Cake 自己說這是它最划算的區域**」：Cake 說的是「計算與 I/O 能力相當時最有利」（p6），沒有綁 16K；它量到兩者相當的 Fig. 3 是在 32K。③ intro §6.1「4K–16K、7–100 Gbps、2.6×」**已核對一致**，但要加條件：7–100 Gbps 只出現在 Table 3（13B、16K），其他表只用 7/32 或 7–56 Gbps；動機實驗到 32K。④ intro §6.2「CacheFlow 是自己在 LMCache 上重做 Cake 的」：**CacheFlow v1、v2 都沒寫怎麼實作 Cake**〔原文〕；在 LMCache＋vLLM 上實作的是 Cake 自己（App. B）。⑤ intro §6.2 第 5 點「限速器的做法簡單公開，能在同樣設定下重做它的表格」：限速器的位置、單位、併發、GPU 使用率的模擬方式都沒寫清楚（見重現卡），**過度樂觀**。〔複核修正：GPU 使用率的「定義」其實有寫（budget 佔比，ICML p6），沒寫的是吃掉其餘 budget 的背景請求長什麼樣（v1 p7 只說「其他合成請求」）；結論「過度樂觀」維持〕⑥ sota §2.1 列的掃描維度**已核對一致**，但漏了 chunk 大小（Table 4，只在 ICML 版）。⑦ workloads_eval 的 Cake 列：「budget 512／1024」是 v2 的寫法，ICML 是預設 512，並在 §5.3 掃 64–2048；「§5.7」是 v2 的節號，ICML 是 §5.8；硬體漏了 1×A100 這組設定。⑧ PAPERS_BY_LEVEL「GPU 只剩 12.5% 時比單純載入慢 25%」**算術一致**（1/0.80＝1.25），但只限 Llama-3.1-70B、32 Gbps 這一格（Table 6），沒寫條件。⑨ intro 表 1（2 TB/s、25 GB/s、0.5–4 GB/s）與「約 80% 命中在磁碟層」**已核對一致**（ICML p2 Fig. 1；80% 是 Cake 轉述 AttentionStore，屬〔二手〕）。⑩ intro §7.3 的引句「we precompute and store all requests' KV cache in advance」**逐字一致**（p5）。⑪ intro、sota 都沒提 +26%；workloads_eval §5.1 對 +26% 的描述正確，只是節號用的是 v2 | 〔原文〕〔計算〕 |
| 對本研究的意義〔判讀〕 | **可沿用**：I/O chunk 128 與 compute chunk＝budget 的分離設計、頻寬點、token budget 佔比模擬 GPU 忙碌、「對只載＼對只算」雙比值的報法。**不可比**：絕對加速倍數與 GPU 世代、TP 綁在一起，而且 I/O 是模擬的。**要小心**：(a) 13B 主表在 16K 下的 KV 是 13.4 GB〔計算〕，只載在 32 Gbps 要 3.36 s，所以只載基線本身就很慢，倍數容易好看；(b) 單次執行、沒有誤差；(c) 「沒有退路」在 GQA 大模型、GPU 很忙時會輸（0.80×），這正是本研究 H1 的檢驗點 | 〔判讀〕 |

### 版本差異：ICML 正式版 vs arXiv v2〔原文，逐段比對〕

1. ICML 新增 §5.3「不同 chunk 大小」與 Table 4（64–2048 token；LongAlpaca-7B/13B；1×A100；100%；32 Gbps；16K）。所以之後的節號、表號都加一：v2 的 §5.3–5.8 → ICML 的 §5.4–5.9；v2 的 Table 4–6 → ICML 的 Table 5–7。
2. ICML 新增 §6 Discussion（和 speculative decoding、KV 量化、P/D 分離相容；分散式環境下各節點各自跑 Cake），p9。
3. ICML 刪掉 v1/v2 p1 引言裡「72K token、Llama2-70B、A100 約 30 秒」的例子。
4. 只算基線：v2 寫 token budget「512 and 1024」（v2 p5）；ICML 寫「預設 512」，並在 §5.3 另外掃（p5）。
5. 其他數字（Table 2/3/5/6/7、2.23×/3.76×、3.34×/2.24×、2.68×/3.01×、+26%、2.6×）兩版相同。
6. v1（2024-10）和 v2 差很多：只有 LongAlpaca-7B/13B；頻寬是 2000/5000/10000 Mbps；長度 5K–16K（20 點）再加 32K 壓力測試；比較對象多了 CacheGen 壓縮；限速是讓 LMCache sleep 到期後才放進 CPU 記憶體（v1 p7）。

---

## 卡 1b｜Cake 重現規格卡（給「在 vLLM 上重現 Cake 式還原＋限速器」用）

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 軟體版本 | vLLM **v0.6.2**（V0 engine，chunked prefill 模式）；LMCache **v0.1.4**；Cake 是在兩者上約 1,000 行的修改。**PyTorch、CUDA、驅動版本都沒寫** | 〔原文〕ICML p5、p12 App. B |
| 模型與 dtype | LongAlpaca-7B（MHA，32 層，512 KiB／token）、LongAlpaca-13B（MHA，40 層，800 KiB／token）、LLaMA 3.1-8B（GQA，128 KiB）、LLaMA 3.1-70B（GQA，320 KiB，FP8 權重）；權重 FP16，KV BF16。70B 在 A100 上用 FP8 權重，用的是哪個量化 kernel 沒寫 | 〔原文〕ICML p5 Table 1；kernel〔未查證〕 |
| TP | 「2×A100」用 tensor parallel（判讀 TP=2）；其餘單卡 | 〔原文〕ICML p6 §5.2 |
| 硬體 | 2×A100 80GB NVLink＋EPYC 7763＋2.0 TB；1×H100（SXM 還是 PCIe 沒寫）＋26 vCPU＋200 GB | 〔原文〕ICML p5 |
| chunk 與 budget | 計算端：chunk＝vLLM token budget，預設 512；Table 4 掃 64、128、256、512、1024、2048。I/O 端：每個 chunk 128 token。GPU 使用率＝Cake 拿到的 budget ÷ 總 budget（12.5/50/87.5/100%） | 〔原文〕ICML p5–7 |
| 頻寬點 | 7 Gbps（Google Cloud 標準 egress）、25（Google Cloud tier-1 egress）、32（Lambda Lab SSD 讀）、56（Samsung 980 Pro 讀）、100（InfiniBand/RoCE）。Table 3：5 點都有；Table 5、6：7、32；Table 7：7、25、32、56；Table 4：32；Fig. 5：0–25 隨機 | 〔原文〕ICML p6 Table 2 |
| prompt 怎麼生成 | 合成 token，長度在 4K–16K 間每 2K 取一點（表上只看到 4K 步距）；長度範圍取自 CacheGen 對 LongChat、TriviaQA、NarrativeQA 的統計；**token 內容、tokenizer、是否隨機都沒寫**。動機實驗用「random context of 32k tokens」 | 〔原文〕ICML p5、p3 |
| 量了哪些圖表 | **Fig. 1**（p2）：三層儲存的頻寬／容量設定（取自 LambdaLab 伺服器規格）。**Fig. 2**（p4）：雙向示意。**Fig. 3**（p4）：等效載入頻寬；LongAlpaca-7B、32K、chunk 512；HDD ~200 MB/s、A100 ~3.3 GB/s、H100 ~4 GB/s、NVMe ~4 GB/s、Network ~1 GB/s。**Fig. 4**（p4）：每個 512-token chunk 的 step time vs chunk index（A100 約 40→110 ms、H100 約 27→93 ms、SSD 32 Gbps 約 62 ms 水平線；圖讀值）〔複核補充：數位化值 A100 38.8→112.0 ms、H100 27.4→93.5 ms、SSD 62.4 ms、chunk 0–63；I/O 線與 A100 交於 chunk 18–19、與 H100 交於 chunk 33；模型未寫，依 256 MB／chunk 判讀為 LongAlpaca-7B〕。**Table 3**（p6）：硬體×使用率×頻寬；13B、16K、chunk 512；例：2×A100、100%、32 Gbps＝2.63\1.59；H100、100%、32 Gbps＝2.40\1.75；A100、100%、32 Gbps＝2.00\1.91〔複核補充：ICML p6 內文說「100%、32 Gbps 時 A100 2×、H100 2.23×、2×A100 2.63×」，H100 的 2.23 與表上 2.40 不符（2.23 是 H100、87.5% 那格）；對照時以表為準，並知道原文有此不一致〕。**Table 4**（p7）：chunk 64–2048；1×A100、100%、32 Gbps、16K；例：7B、512＝2.20\1.94；13B、512＝2.03\1.91。**Table 5**（p7）：長度 4K–16K；2×A100 與 1×A100；7／32 Gbps；13B、chunk 512；例：2×A100、32 Gbps、100%：4K 2.67\1.25 → 16K 2.63\1.59。**Table 6**（p8）：四個模型；2×A100、16K、chunk 512；例：Llama3.1-8B、100%、32 Gbps＝1.37\3.18；70B、12.5%、32 Gbps＝0.80\44.26。**Table 7**（p8）：16/8/3-bit；2×A100、100%、13B、16K；例：32 Gbps 時 2.63\1.59、1.85\2.12、1.37\3.73。**Fig. 5**（p8）：波動下的兩條指標軌跡；A100、7B、0–25 Gbps、budget 0–512、16K。**Fig. 6**（p8）：adaptive vs 預設排程的 token batch size；完工 1.5→1.19 s（硬體沒寫）。**Fig. 7**（p9）：A100/H100 的 per-step 時間，有無 Cake | 〔原文〕〔原文圖讀值〕 |
| 建議先重現哪一張當校準點〔判讀〕 | **第一步：Fig. 4（逐 chunk 重算成本曲線）**。它不需要限速器，只量 chunked prefill 每一步的時間，能驗證「成本隨 index 線性上升」這個前提在我們的硬體與 vLLM 版本上成立，也同時產出模擬器需要的 f(i)。**第二步：限速器自檢**。固定頻寬，量每個 128-token I/O chunk 的實際放行間隔，對照「大小 ÷ 頻寬」（例：7B 的 128-token chunk＝64 MiB，32 Gbps 下 16.8 ms〔計算，以 10⁹ bit/s〕；若照 Fig. 4 數位化結果採二進位解讀〔32 Gbps＝4 GiB/s〕則為 15.6 ms〔複核補充，計算〕）。**第三步：會合點預測 vs 實測**。用第一步的 f(i) 和第二步的 I/O 時間，算出「前段累積重算時間＝後段累積載入時間」的位置，再和 Cake 實測的會合點比；這一步跟硬體無關，最能檢查實作有沒有寫對。**第四步：一張表格的格子**。單卡、7B、chunk 512 的唯一格是 Table 4（7B、512、32 Gbps、100%、16K：2.20\1.94）〔複核修正：原寫「單卡、7B 的唯一格」；Table 4 有 6 個單卡 7B 格（chunk 64–2048，p7），只有 chunk 512 是唯一〕；GQA 的是 Table 6 的 Llama3.1-8B 欄，但那是 2×A100。硬體不同時，比的是**趨勢**：對只算的加速隨長度上升（Table 5）、對只載的加速隨頻寬下降（Table 3 逐列），而不是絕對倍數 | 〔判讀〕〔計算〕 |
| 程式碼是否公開 | 否（查證過程見卡 1「程式碼／資料」） | 〔文件〕〔未查證〕 |

**原文沒講清楚、重現時必須自己決定的地方**〔原文缺漏，逐條〕

1. **限速放在哪一段**：ICML 說「暫停資料傳到 GPU 記憶體」（p5）；v1 說「LMCache sleep 到期後才放進 CPU 記憶體」（v1 p7）；App. A 說 I/O 執行緒把 KV「從儲存讀到 CPU 記憶體」（p12）。CPU→GPU 是真實搬運還是也被吸收進延遲、有沒有算進 TTFT，都要自己定。
2. **延遲公式**：只有「依 chunk 大小與網路頻寬算出延遲」。是否有每次 I/O 的固定延遲、是一次一個 chunk 還是多個在途、延遲從何時起算，都沒寫。
3. **Gbps 的單位**：Fig. 4 的「SSD 32 Gbps」線約 62 ms／512-token chunk（7B 的 256 MiB）。用 4 GiB/s 算是 62.5 ms，用 4×10⁹ B/s 算是 67.1 ms〔計算〕；判讀較接近二進位單位，但讀圖精度不足以定論。必須在我們的實作裡寫明。〔複核補充：以 800 dpi 數位化 Fig. 4（y 軸 40/60/80/100 ms 刻度校準，線寬約 ±0.5 ms），SSD 線中心為 62.4 ms，可排除 67.1 ms（256 MiB÷4×10⁹ B/s）與 64.0 ms（256×10⁶ B÷4×10⁹ B/s），與 256 MiB÷4 GiB/s＝62.5 ms 一致〔計算〕。仍是圖讀值，且原文沒說這條線是量的還是算的〕
4. **每個 chunk 的位元組數**：KV 的實體佈局（逐層、逐 head）與 dtype 決定傳輸量；8-bit、3-bit 實驗是只縮小位元組數，還是也算了反量化的計算時間，沒寫（p8 §5.6）。
5. **兩種 chunk 怎麼對齊**：計算 chunk＝budget（512），I/O chunk＝128。「下一個 chunk 已到」的判斷（App. B.2 第 4–6 步）要等 4 個 I/O chunk 都到嗎？會合處被部分載入的計算 chunk 怎麼處理？〔複核補充：原文本身有兩種說法。App. A 的 Algorithm 1 用兩個不同常數 COMP_CHUNK_SIZE 與 FETCH_CHUNK_SIZE（p12）；但 App. B.2 第 1–2 步寫 Cake「依排定的 token budget 切 chunk」，再把這些 chunk 的 hash 倒序推進 task queue（p13），讀起來像 I/O 也用 budget 大小的 chunk。§5.1 的「I/O chunk 128」與 B.2 的關係原文沒交代〕
6. **停止條件**：計算端每個 engine step 後檢查（App. B.2）；I/O 端在指標交錯時停止（App. A）。在途的 I/O 要不要取消、會合時有沒有重複工作，沒寫。〔複核補充：Algorithm 1 第 3–5 行有明確的停止訊號：計算執行緒發現下一個計算 chunk 已在 CPU 記憶體時「Signal I/O worker to stop」再 break（p12）；App. B.2 第 5 步則是 chunk 已 resident 就中斷 chunk prefill、直接開始生成（p13）。沒寫的是「停止」是否取消在途的讀取〕
7. **prompt 內容**：隨機 token？哪個 tokenizer？Table 5 為何只有 4K 步距而文字說 2K？
8. **「其他使用者」怎麼模擬**：ICML 只給 budget 佔比的定義（p6）；v1 說「用其他合成請求吃掉剩下的 budget」（v1 p7），但這些請求的長度、是 prefill 還是 decode 沒寫。也沒寫只算、只載基線是否承受同樣的背景負載。
9. **TTFT 的邊界**：是否包含最後幾個未命中 token 的 prefill 與第一個 decode step；只載基線的 TTFT 是否包含 LMCache 讀完後算最後一個 token。
10. **加速比怎麼算**：單次執行還是平均？2.6× 是哪些表、哪些格的平均（p1、p9 沒交代）？Table 3 的平均排除了「對任一基線 >10×」的紅色格（p6），其他表有沒有排除？
11. **LMCache 的 chunk 與 hash**：Cake 用 128-token 的 I/O chunk，LMCache 本身的 chunk 與 prefix hash 粒度要和它對齊；GPU 上的 prefix cache 是否關閉，沒寫。
12. **vLLM 版本落差**：v0.6.2 是 V0 engine；現在的 V1 engine 預設就是 chunked prefill，token budget 對應 `max_num_batched_tokens`，KV 外接要走 KV connector（例如 OffloadingConnector，見 py-kvcache p4 的 API 說明）。〔複核標註：「V0／V1 engine、V1 預設 chunked prefill」不在 Cake 或本組任何論文中，屬〔判讀，未附出處〕，動手前要查 vLLM 文件確認；py-kvcache p4–5 有寫 KV Offload API 自 v0.11.0、`offloading_connector` 把 Offload 操作轉成 Transfer 介面〕重現時要決定 Cake 的「每步檢查」掛在哪個 hook。
13. **Fig. 5、Fig. 6 的設定**：波動軌跡的取樣週期；「spiked distribution」的定義；Fig. 6 的硬體。
14. **H100 的型號**（SXM／PCIe）、CPU 記憶體是否 pinned、NUMA 綁定，都沒寫，會影響 CPU→GPU 段。
15. **長度上限**：LLaMA 3.1 原生支援 128K，但原文沒跑；LongAlpaca 只到 32K（v1 p7）。要延伸到 128K–512K，必須換成原生長 context 的模型（依本專案規則不開 YaRN）。

---

## 卡 2｜CacheFlow：Efficient LLM Serving via Automated 3D-Parallel KV Cache Restoration（arXiv 2026；arXiv 2604.25080）

- **讀了什麼**：〔全文〕arXiv v2（2026-09-26，含附錄 A、B）；v1（2026-04-28）比對方法與實驗設定。查證 2026-10-06。
- **一句話**：把還原拆成 token、層、GPU 三個維度並行，用批次感知的 DP 決定每個（請求, 層區塊）重算多少、載入多少。
- **評測要證明的主張**：在 dense 與 hybrid（GDN）模型、對話與 agent 負載、不同硬體與頻寬下，平均 TTFT 比既有方法快 2.24–3.00×，TTLT 快 1.64×；planner 開銷小、成本預測準（v2 p1–2、p8–9）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | Llama-3.1-8B（full attention）、Qwen3.5-9B（hybrid）、Qwen3.5-35B-A3B（MoE＋hybrid）、Qwen3.5-122B-A10B。Qwen3.5-9B：8 個 block，每個 3 層 Gated DeltaNet＋1 層 full attention（4 個 BF16 KV head、head 維度 256）。其他模型的 KV dtype 沒寫 | 〔原文〕v2 p7 §4.1、p16 App. A.1 |
| 硬體 | A100 與 H100，單卡與多卡；122B 用 8×H100（TP=8/PP=1 與 TP=4/PP=2，EP 開／關）。CPU、DRAM、儲存裝置、網路**都沒寫** | 〔原文〕v2 p7–8 |
| 軟體與版本 | 實作在 vLLM 與 LMCache 上（版本沒寫）。基線：vLLM **v0.29.0**（只算）、LMCache **v0.5.5rc5**（只載）、Cake（沒寫怎麼實作）。v1 的基線還有 SGLang（HiCache）、LMCache v0.3.1 | 〔原文〕v2 p7；v1 p7 |
| 資料／負載 | LMSYS-Chat（真實多輪）、SWE-Bench（agent 寫程式）、OSWorld 2.0（電腦操作 agent）。取樣幾筆、怎麼前處理、prompt 是否真實內容，**都沒寫**。v1 用的是 LMSYS-Chat、WildChat、SWE-Bench | 〔原文〕v2 p7；v1 p7 |
| 長度 | Fig. 7 的 cached length 到 128K（Qwen3.5-9B、OSWorld 2.0）；Fig. 2 的 cached prefix 到約 96K（Llama-3.1-8B）。附錄 A.1 的理論表算到 32K/64K/128K | 〔原文〕v2 p4 Fig. 2、p8 Fig. 7、p16 |
| I/O 怎麼實現 | **沒寫**。只說評估「80、40、10 Gbps 的頻寬條件」（預設 10 Gbps；多卡時是「per worker」），分別對應 InfiniBand/RoCE、Lambda Lab SSD、Amazon 節點間頻寬。Fig. 5 的 y 軸是「Cache load (% of cap)」，判讀有一個頻寬上限，但是限速器還是真實網路，原文沒寫。planner 的模型是載入時間＝位元組 ÷ B（Eq. 2） | 〔原文〕v2 p5、p7–8；上限〔判讀〕 |
| 重算與載入成本怎麼量 | 重算：f(j)＝「前 j 個 chunk 通過一個層 block」的時間，從近期服務的實際執行**線上收集**，沒觀察到的 chunk 數用內插。原文理由：固定模型與硬體時，成本主要由 token 長度決定、和內容無關（p6）。載入：d(j)＝(K−j)·S_attn＋（未全算時）S_rec 位元組，除以 B。Fig. 10：92% 的重算預測、98% 的載入預測誤差在 10% 內；95% 的還原，重算與載入在 10% 內同時完成（p9）。理論：Prop. 1 的差距隨 O(1/N_c) 消失；Qwen3.5-9B、10 Gbps、C=512 時最多 19.5 ms（32K 的 6.52%、128K 的 1.63%）（p7、p16） | 〔原文〕v2 p5–7、p9、p16 |
| 位置與層的成本 | token 維度：full attention 下越後面越貴，累積成本隨長度二次成長，載入每 token 約固定（p4）。層維度：每層攤提 kernel launch、同步、權重搬移等固定開銷（p4）。Fig. 2（Llama-3.1-8B）：cached prefix 約 36K 以下 layer-wise 比 token-wise 快（最多約 20%），以上反過來（圖讀值）〔複核修正：圖上最左點（約 11K）的中位數差約 −22%，陰影帶下緣到約 −34%；96K 時 token-wise 快約 11%；y 軸是「paired per-request TTFT 差的中位數」（v2 p4 Fig. 2）〕。128K 時 full-attention 層的狀態是 recurrent checkpoint 的 125×（35B-A3B）／250×（9B），H100 上每層重算時間差 2.6–4.5×（p5） | 〔原文〕〔原文圖讀值〕v2 p4–5 |
| 到達與併發 | batch 大小 1、4、8（Fig. 15）；線上服務實驗用「Mooncake arrival timing」重播 SWE-Bench（Fig. 14），用哪一份 Mooncake trace 沒寫；其餘實驗的到達方式沒寫 | 〔原文〕v2 p7、p9 |
| 重用結構 | 多輪與 agent 的前綴重用；還原的是已快取的前綴 N_c，後面接未快取的 suffix | 〔原文〕v2 p3、p6–7 |
| 掃描的自變數 | 模型（4）、負載（3）、頻寬（10/40/80）、GPU 世代（A100/H100）、GPU 數（1/2/4）、batch（1/4/8）、長度分組、元件消融（layer-wise → 多卡 → token-wise） | 〔原文〕v2 p8–9 |
| 對手 | vLLM v0.29.0、LMCache v0.5.5rc5、Cake。Cake 怎麼實作（重做？哪個版本？）**沒寫**；Fig. 5 中 Cake 的 SM active 只有 36%（cache load 90%）〔原文圖讀值〕，判讀這是 CacheFlow 自己的 Cake 實作在它的設定下的結果 | 〔原文〕v2 p7–8；〔判讀〕 |
| 系統指標 | TTFT（平均、p95、p99、CDF）、TTLT（p95）、還原期間 GPU 與 I/O 使用率、planner 時間佔 TTFT 比 | 〔原文〕v2 p7–9 |
| 品質指標 | 無（重算與載入都無損） | 〔原文〕 |
| 主要結果 | ① 平均 TTFT 2.24–3.00×；p95 1.77–3.02×、p99 1.48–2.95×（p8）。② OSWorld 2.0、Qwen3.5-9B：對 Cake、LMCache、vLLM 為 3.00×、3.96×、4.82×（p8）。③ 對 Cake：40 Gbps 2.63×、80 Gbps 2.40×（p9）；H100 2.69×、A100 2.07×（p9）。④ 使用率：CacheFlow 89% GPU／89% I/O；LMCache 1% GPU；vLLM 98% GPU（p8） | 〔原文〕v2 p8–9 |
| 消融／敏感度／開銷 | 元件分解（Fig. 9：10.46×／4.46×／3.73× vs 單卡只載）；DP 開銷（中位數 0.019%、p99 0.34% 的 TTFT）；預測準確度（Fig. 10）；GPU 數（1.59→0.96→0.67 s）；頻寬；GPU 世代；batch；122B 大模型（1.21–1.43×）；線上服務 | 〔原文〕v2 p8–9、p17 |
| 重複與統計 | 未說明重複次數；只報 CDF 與百分位 | 〔原文〕 |
| 程式碼／資料 | 未公開。Reproducibility Statement：「接受後釋出」程式碼、設定與腳本（p10）。GitHub 搜尋未找到 | 〔原文〕〔未查證〕 |
| 長度範圍與停止理由 | 到 128K；原文沒說為什麼停在 128K（判讀：受負載資料本身長度與單卡記憶體限制） | 〔原文〕〔判讀〕 |
| 設計理由（原文） | chunk C=512 對齊 FlashAttention 與 Cake 的粒度，維持重算效率（p4）；相反方向走訪是為了避開昂貴的尾巴（p4）；用 block 當規劃單位，因為重算一個 block 需要前面層的 activation（p5）；DP 的狀態量化到 chunk，降低狀態空間（p6）；線上 profile，因為成本由長度決定、與內容無關（p6）；三個頻寬點對應三種實體情境，預設 10 Gbps（p7） | 〔原文〕v2 p4–7 |
| 設計理由〔判讀〕 | 預設 10 Gbps 是對只載最不利、對重算最有利的點，正好凸顯「兩邊都要用」；把 Cake 當主要對手並在 40/80 Gbps 比，是要證明高頻寬下 token 維度不夠 | 〔判讀〕 |
| 原文沒講清楚的地方 | I/O 是限速還是真實網路、從哪一層載入（CPU？遠端？）；Cake 基線怎麼實作；負載取樣筆數與 prompt 內容；除了線上實驗外的到達方式；Mooncake 用哪份 trace；重複次數 | 〔原文缺漏〕 |
| 與既有整理不一致 | ① PAPERS_BY_LEVEL 的「最佳切點＝L×T_io÷(T_comp＋T_io)」來自 **v1 §3.2 Eq. (1)**（v1 p5），**v2 已改成 DP，沒有這條式子**；而且它假設每個單位的成本一樣（ℓ/L·T_comp），和 token 維度的二次成本不符，所以「等於 1/(1+κ)」只在均勻成本下成立。PAPERS_BY_LEVEL 標〔部分〕卻沒標版本。② intro §7.2「CacheFlow 的載入時間也是位元組÷頻寬」：**planner 的模型是如此（Eq. 2）**，但實驗的 I/O 怎麼做原文沒寫，不能據此說它的實驗是模擬的。〔複核：這句字面上對 planner 成立，問題在它放在「I/O 是用大小÷頻寬模擬的」這個標題下，容易讀成實驗也是模擬；判為部分成立〕③ intro §6.2、sota §6「CacheFlow 是自己在 LMCache 上重做 Cake」：**原文沒寫**，改標〔未查證〕。〔複核補充：v2 的 Reproducibility Statement 說 §4.1 描述了「baseline implementations」（p10），但 §4.1 對 Cake 只有一句定義（p7）；v1 p7 同樣只有一句。兩版都只說「CacheFlow 本身」建在 vLLM＋LMCache 上〕④ sota「比 Cake 快 2.40–2.63 倍（40／80 Gbps）」：數字一致，對應是 40 Gbps＝2.63×、80 Gbps＝2.40×。⑤ intro §7.3「假設所有狀態都可以載入，被逐出的只能重算」**一致**（p6：不可用的狀態設為無限成本；p14：目標快取「可從裝置外載入」）。⑥ intro §8.6「92%／98% 在 10% 內」**一致**（p9）。⑦ intro 表 7「最長評測 128K」**一致**。⑧ sota「理論假設兩種資源互不干擾」**一致**（p14：一個計算資源、一個獨立載入資源，重疊無干擾）。⑨ CacheFlow 自己把 40 Gbps 對應到「Lambda Lab SSD」、80 Gbps 對應到「Infiniband/RoCE」並引 Cake，但 Cake Table 2 寫的是 32 Gbps 與 100 Gbps（Cake p6）；我們的頻寬點若要和兩篇對照，要知道這個落差 | 〔原文〕〔計算〕 |
| 對本研究的意義〔判讀〕 | 讀取端的最強對手，而且**成本線上量測**＝我們「硬體成本用量的」設計原則的前例。**要小心**：(a) Fig. 2 顯示 36K 以下 token 維度（Cake 式）比 layer 維度慢，代表每個 chunk 的固定開銷不能忽略，我們的限速器與成本模型都要有「每筆固定成本」；(b) 它只有一個載入頻寬、一個來源層，多層放置沒有涵蓋；(c) 程式碼未釋出，只能在同平台重做 | 〔判讀〕 |

---

## 卡 3｜Pensieve：Stateful Large Language Model Serving with Pensieve（EuroSys 2025；arXiv 2312.05516）

- **讀了什麼**：〔全文〕arXiv v3（2024-10-07，EuroSys'25 定稿格式）。查證 2026-10-06。
- **一句話**：多輪對話的 KV 存在 GPU＋CPU 兩層；空間不夠時，優先從閒置久、重算便宜的對話的**開頭**逐出；被丟的用重算補回。
- **評測要證明的主張**：保留跨請求的對話狀態，吞吐是 vLLM、TensorRT-LLM 的 1.14–3.0×，中等負載下延遲顯著降低（p1–2）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | OPT-13B、OPT-66B（MHA）；Llama 2-13B（**作者把 KV head 從 40 改成 10** 來模擬 GQA）、Llama 2-70B（GQA，8 個 KV head）。權重與中間值都是 FP16 | 〔原文〕p9 Table 1、§6.1 |
| 硬體 | Azure NC A100 v4：最多 4×A100-80GB、24 核 AMD EPYC 7003、每張 GPU 配 220 GB CPU 記憶體；每張 GPU 給 KV 40 GB（為了公平）。小模型單卡、大模型 4 卡 TP | 〔原文〕p9 §6.1 |
| 軟體與版本 | 自寫約 7K 行 C++/CUDA，用 PyTorch v2.0.0（CUDA 11.8）的 C++ 前端；以 Cutlass 寫支援非連續 KV 的 multi-token attention kernel。基線：vLLM v0.2.0（作者補寫了驅動迴圈）、TensorRT-LLM v0.12.0 | 〔原文〕p8–9 |
| 資料／負載 | ShareGPT（48,159 段對話，平均 5.56 輪，平均輸入 37.77、輸出 204.58 token）、UltraChat（1,468,352 段，3.86 輪）。context 上限 16,384 token，丟掉 0.57% 超長的 ShareGPT 對話 | 〔原文〕p9 Table 2 |
| 長度 | 累積 context ≤16,384 token | 〔原文〕p9 |
| I/O 怎麼實現 | **真實 GPU↔CPU 搬運**（PCIe 世代沒寫）。提前換出（GPU 剩餘槽 <25% 時）；換入逐層 pipeline，第 i 層的 KV 到了才開始第 i 層的 attention（CUDA event）。實測雙向同時傳輸會讓兩個方向的吞吐都掉 18–20%，所以換入優先、換出等待。沒有 SSD | 〔原文〕p6–7、p9 |
| 重算與載入成本怎麼量 | 重算：Cost(l)＝Cost_attention(l)＋c，以 32-token chunk 為單位；**離線 profile** 2 的冪次 context 長度，其餘內插（p6）。Fig. 4：32-token chunk 的 attention 時間 ÷ 非 attention 時間，隨 context 線性上升，約 0.4（4K）、0.77（8K）、1.53（16K）（圖讀值；模型沒寫）。載入：沒有獨立的成本模型，靠 pipeline 隱藏 | 〔原文〕〔原文圖讀值〕p4、p6 |
| 到達與併發 | 對話開始時間 Poisson（多種請求率）；同一對話內，上一輪回應收到後，經過「使用者思考時間」才送下一輪；思考時間取指數分布，預設平均 60 s，另掃 30–600 s | 〔原文〕p9、p12 |
| 重用結構 | 同一對話的多輪追加；不跨使用者共享（系統提示可另外標為可重用，p13 註 3） | 〔原文〕p9、p13 |
| 掃描的自變數 | 請求率、模型、資料集、思考時間、逐出策略（vs LRU）、統一排程與否、kernel 的 past-KV 數 | 〔原文〕p10–12 |
| 對手 | vLLM v0.2.0、TensorRT-LLM v0.12.0（都是無狀態）；自己的 Pensieve (GPU cache) 變體（不用 CPU 層） | 〔原文〕p9–10 |
| 系統指標 | 吞吐（req/s）與 normalized latency（端到端延遲 ÷ 輸出 token 數）。§6.1 說用 90 百分位，§6.2 說是平均，**原文前後不一致** | 〔原文〕p10 |
| 品質指標 | 無 | 〔原文〕 |
| 主要結果 | 單卡 ShareGPT：OPT-13B 在 120 ms/token 時為 vLLM 的 1.36×、TRT-LLM 的 1.14×；Llama 2-13B 在 180 ms/token 時 1.70×／1.58×（p10）。4 卡：Llama 2-70B 在 400 ms/token 時 3.0×／2.47×（p11）。逐出策略 vs LRU：約 3 req/s 以上才有差，CPU 命中率最多高 4.4 個百分點，重算的 KV token 最多少 14.6%（p12） | 〔原文〕p10–12 |
| 消融／敏感度／開銷 | kernel 微基準（Fig. 12）、統一排程（Fig. 13）、逐出策略（Fig. 14）、思考時間（Fig. 15） | 〔原文〕p11–12 |
| 重複與統計 | 未說明 | 〔原文〕 |
| 程式碼／資料 | 沒有找到（GitHub 搜尋 pensieve 相關 repo、一作 lingfanyu 的 repo 列表，皆無） | 〔未查證〕 |
| 長度範圍與停止理由 | 16,384 是作者設的上限，理由沒寫（判讀：當時模型的 context 上限） | 〔原文〕〔判讀〕 |
| 設計理由（原文） | 從開頭逐出，因為前面的 token 重算便宜（Fig. 4，p4）；保留值 V＝Cost(s,l)／T，同時表達 LRU 與「從前面丟」兩種偏好（p6）；32-token chunk 是「實驗發現好用」，也減少決策與小塊 PCIe 傳輸的開銷（p6）；沒有 timestamp 所以用 Poisson＋指數思考時間（p9）；改 KV head 數是為了展示 GQA 的效果（p9） | 〔原文〕p4、p6、p9 |
| 設計理由〔判讀〕 | Fig. 5 的請求佈局（開頭被丟要重算｜中段在 CPU 要載入｜後段仍在 GPU｜新 prompt 要計算）其實就是「前段重算、中段載入、後段留在快層」的靜態版，**只是由逐出順序造成，不是規劃出來的**。重算段與新 prompt 在同一個 prefill 裡算，載入段逐層 pipeline，兩者實際上也是並行的 | 〔判讀〕依 p4 Fig. 5、p7 §4.3.4 |
| 原文沒講清楚的地方 | PCIe 世代；Fig. 4 用的模型；normalized latency 是平均還是 p90；Llama 2-13B 改成 10 個 KV head 後權重怎麼處理（判讀：只量效能，輸出無意義）；重複次數 | 〔原文缺漏〕 |
| 與既有整理不一致 | ① intro 表 6、sota §2.4「從對話開頭逐出，缺的再重算；吞吐 1.14–3.0×；只有 GPU/CPU；沒有精度」**已核對一致**。② intro 表 7 的「最長評測 —」可以補成 **16K（作者設的上限）**。③ PAPERS_BY_LEVEL「最早把『位置越前面越便宜重算』用在逐出上」：原文沒有這個宣稱，「最早」無法驗證，改標〔判讀〕。〔複核補充：sota §2.4 也有同一句「最早把位置不同、重算成本就不同用在逐出上」，同樣要改；原文最接近的是 p13 Table 3，只說它在比較的 5 個系統中唯一「從開頭」逐出〕④ sota 標〔部分〕，本次已讀全文。⑤ intro 表 8 把 Pensieve 歸為 GreedyDual-Size 式的成本感知替換：保留值＝重算成本÷閒置時間，**形式一致**〔判讀〕。⑥「沒有開源」：本次搜尋也沒找到 | 〔原文〕〔判讀〕 |
| 對本研究的意義〔判讀〕 | 是 E1 的 P2（全存＋從前段逐出）的直接前例，而且它的讀取路徑已經有「前段重算＋中段載入」的形狀。**要小心**：Pensieve 的長度只到 16K、層只有 GPU/CPU、成本模型只看 attention；若 P2 就拿走大部分好處，本研究的新穎性要改寫。Fig. 4 也提供一個獨立的「位置成本」數據點：16K 時，attention 已經是非 attention 的約 1.5 倍 | 〔判讀〕 |

---

## 卡 4｜HCache：Fast State Restoration in LLM Serving with HCache（EuroSys 2025；arXiv 2410.05004）

- **讀了什麼**：〔全文〕arXiv v1（2024-10-07，EuroSys'25 定稿格式）。查證 2026-10-06。
- **一句話**：不存 KV，改存每層的 hidden state（MHA 下是 KV 的一半），還原時用一次投影把 KV 算回來，並用其他方法補 pipeline 空泡。
- **評測要證明的主張**：TTFT 比 KV offload 最多快 1.93×、比重算最多快 5.73×，儲存省 1.92–2.40×，TBT 開銷 <4%（p1–2）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | Llama2-7B、Llama2-13B、OPT-30B（MHA）。作者把三個模型的最大 context 擴到 16K（怎麼擴沒寫）；FP16〔複核修正：原文沒有明寫模型或 KV 的 dtype，只有 Table 2 註明 FLOPS 是 FP16 運算（p9），改為〔判讀〕FP16〕。7B／13B 單張 A100，OPT-30B 4 張 A100 TP（p9） | 〔原文〕p9 |
| 硬體 | 預設：4×A100-40G SXM4（NVLink）、2×EPYC 7642、256 GB DDR4、**4×Samsung PM9A3 4TB**。敏感度：雲端 A30、L20、H800（用 host DRAM 當儲存後端；Table 2 寫傳輸速度 A100/A30/L20 為 32 GB/s、H800 為 64 GB/s）〔複核修正：漏了 RTX 4090。Table 2 有五列：A100 40G／312T、A30 24G／165T、**4090 24G／330T**、L20 48G／120T、H800 80G／990T；傳輸速度 A100／A30／4090／L20 都是 32 GB/s、H800 64 GB/s（p9）；Fig. 11(a) 也有 4090（p11）〕 | 〔原文〕p9 Table 2、p11 Fig. 11 |
| 軟體與版本 | 在 DeepSpeed-MII v0.2.0 上加約 5,731 行（CUDA、C++、Python）；cuBLAS 做投影、自寫 RoPE kernel；SPDK＋GDRCopy 實作 GPU 直讀 SSD | 〔原文〕p8 |
| 資料／負載 | ShareGPT4（每輪平均輸入 66.8、輸出 358.8 token；歷史截在 16K）；L-Eval（20 個子任務，平均 context 16,340.2；代表子任務 Paper Assistant、GSM-100、QuALITY；另取 200 個混合請求） | 〔原文〕p3–4 Table 1、p10 |
| 長度 | 主實驗歷史 4K–16K（L-Eval）；敏感度 Fig. 11(g–h) 1K–16K、**Fig. 11(i) OPT-30B 到 32K**；Fig. 11(d–f) 固定 1,024 | 〔原文〕p10–11 |
| I/O 怎麼實現 | **真實 NVMe**：hidden state 以 64-token chunk 依 round-robin 分散到多顆 SSD，用 SPDK＋GDRCopy 直接 P2P 寫進 GPU BAR。1 顆 PM9A3 讀 6.9 GB/s，4 顆可塞滿 A100 的上行 PCIe。**OPT-30B 需要 8–16 顆時用 DRAM 模擬**（插槽不夠）。雲端 GPU 實驗用 host DRAM | 〔原文〕p7–8、p11 |
| 重算與載入成本怎麼量 | 解析式（每層，MHA）：IO_hidden＝N·D/BW；C_hidden＝4·N·D²/FLOPS；token 重算 T_rec＝(24·N·D²＋N²·D)/FLOPS；所以 hidden 的計算至少省 6×、傳輸省 2×（p5）。實際排程用**離線 profile** 平台的傳輸與計算速度，解 min-max 求 L_H、L_O（p7）。從 hidden 算 KV 沒有注意力項，**成本與位置無關、隨長度線性**；token 重算才有 N² 項 | 〔原文〕p5、p7 |
| 到達與併發 | ShareGPT4：session 到達 Poisson，同 session 各輪間隔固定 30 s，每輪結束就逐出 KV；L-Eval：batch size 1；GPU 重用實驗（§6.4）用 Zipf α 合成 context 的到達 | 〔原文〕p10、p13 |
| 重用結構 | 多輪對話歷史、長文件問答的 context 重用 | 〔原文〕p3–4 |
| 掃描的自變數 | 負載率（sessions/s）、GPU 平台、SSD 數（1–4，DRAM 模擬 8–16）、歷史長度、Zipf α、解碼 batch（存檔消融） | 〔原文〕p10–13 |
| 對手 | Recomputation（DeepSpeed-MII）、KV offload（**在 DeepSpeed-MII 上重做 AttentionStore**，沒做其 decoupled position embedding）、Ideal（GPU 上放佔位 KV、不需還原）；消融：HCache-O、Naive Hybrid（同一個 bubble-free 排程器混「token 重算＋KV offload」，不用 hidden state） | 〔原文〕p9、p12 |
| 系統指標 | TTFT（還原＋prefill）、TBT、還原速度（還原的歷史 token 數 ÷ 還原時間）、GPU cache hit ratio | 〔原文〕p9、p13 |
| 品質指標 | 無（宣稱無損） | 〔原文〕p13 |
| 主要結果 | ShareGPT4：對 KV offload 1.27–1.90×、對重算 2.21–3.57×（p10）。L-Eval：1.62–1.93×、2.66–5.73×（p10）。消融：比 Naive Hybrid 快 1.28–1.42×（p12）。Zipf α=2.0、GPU 命中 94% 時仍比 KV offload 快 1.15×（p13） | 〔原文〕p10–13 |
| 消融／敏感度／開銷 | GPU 平台、SSD 數、長度（Fig. 11）；排程器（Fig. 12）；token-wise vs layer-wise 切分（Fig. 13：naive token-wise 慢 12%，取整後仍慢 7%）；兩段式存檔（Fig. 14：DirectIO 在 batch 16 時 TBT 高 34%）；GPU 重用（Fig. 15） | 〔原文〕p10–13 |
| 重複與統計 | 未說明 | 〔原文〕 |
| 程式碼／資料 | 沒有找到（GitHub 搜尋 HCache；一作 gswxp2 的公開 repo 中沒有） | 〔未查證〕 |
| 長度範圍與停止理由 | 原文：為了容納 L-Eval 與長對話把模型擴到 16K（p9）；單張 A100-40G 只放得下 1–3 個長 context（p4、p10）〔複核修正：「1–3 個長 context」在 p1 引言與 p10，不在 p4〕 | 〔原文〕p1、p9、p10 |
| 設計理由（原文） | 用 layer-wise 而不用 token-wise 切分，因為 token-wise 會產生 cuBLAS 不擅長的不規則矩陣（p7、Fig. 13）；64-token chunk 依 round-robin 分到多顆 SSD 以聚合頻寬、避免依最長長度預留空間的碎片（p7–8）；預設用 SSD 因為便宜、大、頻寬高（p6）；L-Eval 用 batch 1 因為 GPU 只放得下 1–3 個（p10）；沿用前作的 Poisson 與 Zipf（p10、p13） | 〔原文〕 |
| 設計理由〔判讀〕 | HCache 的排程器其實是「layer 維度的 Cake」：前 L_O 層從 token 重算（或後 L_O 層直接載 KV），其餘層載 hidden 再投影；平衡點由離線 profile 決定，不是執行時自然會合 | 〔判讀〕依 p7 §4.1.2 |
| 原文沒講清楚的地方 | 模型怎麼擴到 16K（RoPE 縮放？）；重複次數；DRAM 模擬 8–16 顆 SSD 時的延遲模型〔複核補充：模型與 KV 的 dtype 沒寫；Table 3 的每 token 儲存量剛好都是 FP16 算出值的一半（例：7B 的 KV offload 寫 256 KiB，但 2×32 層×4096×2 B＝512 KiB；13B 寫 400 KiB，算出 800 KiB；OPT-30B 寫 672 KiB，算出 1,344 KiB）〔計算〕，原文沒解釋，引用這些數字前要先弄清單位〕 | 〔原文缺漏〕p10 Table 3 |
| 與既有整理不一致 | ① intro §7.1「HCache 最長 16K」：主實驗是，但 **Fig. 11(i) 的 OPT-30B 到 32K**（p11）。② sota §2.12、PAPERS_BY_LEVEL「hidden state 約 KV 一半、最多快 1.93×、只適用 MHA」**已核對一致**；原文的說法是「目前不改模型即支援 MHA」，GQA 要先投影到低秩表示、屬範圍外（p13）。③ PAPERS_BY_LEVEL「GQA 的 KV 本來就比 hidden state 小」〔計算〕一致：例如 Llama-3.1-8B 每層 KV＝2×8×128＝2,048 個值，hidden＝4,096。④ sota「在層的維度實作了『會被重算的就不存』」**一致**：OPT-30B 排程為 40 層 hidden＋8 層從 token 重算，那 8 層不需要存（p10 Table 3）。⑤ workloads_eval 的 HCache 列**已核對一致**（Zipf 只用在 §6.4 的 GPU 重用實驗） | 〔原文〕〔計算〕 |
| 對本研究的意義〔判讀〕 | 提供「真實 SSD＋GPU 直讀」的 I/O 實作參考，以及「每層存什麼」的第三種狀態（hidden state）。對 GQA 模型（本研究的主力）不划算，所以不必納入狀態空間，但可以在相關工作裡說明。layer-wise 與 token-wise 的 GEMM 形狀差異，提醒我們 chunk 大小會影響重算效率 | 〔判讀〕 |

---

## 卡 5｜KVPR：Efficient LLM Inference with I/O-Aware KV Cache Partial Recomputation（ACL Findings 2025）

- **讀了什麼**：〔全文〕ACL Anthology 正式版 PDF（印刷頁 19474–19488）。查證 2026-10-06。
- **一句話**：KV 卸載在 CPU 時，先傳前段的 activation 讓 GPU 投影出 KV，其餘 KV 同時經 PCIe 傳入。
- **評測要證明的主張**：**解碼階段**的延遲最多降 35.8%、吞吐最多升 46.2%（p1）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | OPT-6.7B、13B、30B（MHA），FP16；附錄另有 LLaMA2-7B、13B | 〔原文〕p2 Table 1、p6、p13–14 App. A.6 |
| 硬體 | 1×A100 40GB，PCIe 4.0 ×16（32 GB/s），AMD EPYC 64 核 2.6 GHz；附錄 A.7 用 8×A100＋單顆 CPU；A.5 另有低階 GPU | 〔原文〕p6（印刷頁 19479）、p14 |
| 軟體與版本 | 建在 Hugging Face Transformers v4.46.1 與 FlexGen 上；CUDA stream 非同步重疊；pinned memory | 〔原文〕p6 |
| 資料／負載 | 沿用 FlexGen 的資料，prompt 統一補齊到同長度 | 〔原文〕p6 |
| 長度 | prompt 256、512、1024（附錄另有 128）；生成 32 或 128〔複核修正：p6 文字寫 256／512／1024，但正文的延遲實驗 Fig. 7（batch 64）用的是 prompt **128／256／512**（各配生成 32／128），代表結果 35.8% 就是 prompt 128＋生成 128（p7）；吞吐實驗 Fig. 6 才是 256／512／1024，batch 掃描固定 1024＋32（p7）。所以 128 不是「附錄另有」，延遲實驗也沒有 1024〕 | 〔原文〕p6–7、p12 |
| I/O 怎麼實現 | **真實 CPU DRAM→GPU（PCIe 4.0）**；不處理磁碟或網路 | 〔原文〕p6、p9 Limitations |
| 重算與載入成本怎麼量 | 解析式：傳 activation X[0:l]（b·l·h·p 位元組）、傳 KV[l:s']（2·b·(s'−l)·h·p）；重算 FLOPs＝4·b·l·h²，時間＝FLOPs÷v_gpu；以單一整數變數的 LP 求最佳 l（Eq. 6–11，p5）。v_gpu、v_com 由 profiler 在**推論開始時量一次**。重算只是投影、沒有注意力項，**與位置無關** | 〔原文〕p5、p9 |
| 到達與併發 | 無到達過程；延遲實驗 batch 64，吞吐實驗有效 batch 32×8；另掃 batch 1–48 | 〔原文〕p6–7 |
| 重用結構 | 無跨請求重用；是單一 batch 解碼時的 KV 卸載 | 〔原文〕p3 |
| 掃描的自變數 | 模型大小、prompt 長度、生成長度、batch、4-bit KV 量化（group-wise） | 〔原文〕p6–8 |
| 對手 | 延遲：DeepSpeed Inference、HF Accelerate；吞吐：FlexGen；附錄 A.7：FastDecode | 〔原文〕p6、p14 |
| 系統指標 | 解碼延遲、解碼吞吐（tokens/s）、GPU 使用率；**明說不影響 prefill** | 〔原文〕p6–7 |
| 品質指標 | 無（精確計算，無近似） | 〔原文〕p2 |
| 主要結果 | OPT-6.7B、prompt 128、生成 128 時延遲比 HF Accelerate 低約 35.8%（p7）；吞吐比 FlexGen 高最多 15.1%／46.2%／29.0%（6.7B／13B／30B，p7）；解碼期 GPU 使用率 85%→99%（p7） | 〔原文〕p7 |
| 消融／敏感度／開銷 | 隱藏重算於權重載入（Table 2）；KV 壓縮（Fig. 9）；runtime 分解（Fig. 10：KV 傳輸 58%→38%，GPU 計算 2.3%→13.3%） | 〔原文〕p8 |
| 重複與統計 | 5 次平均（沒有誤差） | 〔原文〕p6 |
| 程式碼／資料 | 公開：https://github.com/chaoyij/KVPR（最後 commit `1712a52`，2025-05-28；未讀內容） | 〔原文〕p1；〔文件〕GitHub API |
| 長度範圍與停止理由 | 沿用 FlexGen 的設定，原文沒有另外說明 | 〔原文〕p6 |
| 設計理由（原文） | 用 FlexGen 的資料是為了公平比較（p6）；延遲導向保留權重在 GPU、吞吐導向卸載權重（p6）；先載 W_K、W_V 讓重算提早開始（p6） | 〔原文〕 |
| 設計理由〔判讀〕 | 切點只有一個整數變數，所以 LP 很便宜；但它求的是「每一步解碼」的切點，跟 Cake 的「還原一次」是不同問題 | 〔判讀〕 |
| 原文沒講清楚的地方 | 「activation」具體是哪一層的輸入（判讀：每層的 X^i，等同 hidden state）；profiler 量測的方法與時長；附錄 A.4 說最佳切點從生成長度 1 的 182「增加」到 32 的 128，數字方向與文字矛盾〔複核補充：而且該設定是 prompt 128（p12），Eq. 11 的限制是 0 ≤ l ≤ s（p5），l＝182 本身就超出範圍〔計算〕〕 | 〔原文〕p5、p12 |
| 與既有整理不一致 | ① sota §2.12「只支援單 GPU」：原文是「單 GPU 與 data-parallel 多 GPU」，不支援 TP（p9）。② sota、intro 都沒提 **KVPR 只優化解碼、不影響 prefill**（p6），而且 prompt ≤1K；把它和 Cake 放在同一類「讀取時還原」要加註。③ intro 表 10「KVPR 只量一次並假設硬體不變」**一致**（p9 Limitations）。④ workloads_eval 的 KVPR 列**已核對一致** | 〔原文〕 |
| 對本研究的意義〔判讀〕 | 「傳 activation 取代 KV」的手法對 GQA 的長 context 還原不划算（同 HCache 的理由），而且它的場景是解碼，與 TTFT 還原不同。可沿用的是「profiler＋單變數最佳化」的輕量設計，以及它承認「只量一次」是限制 | 〔判讀〕 |

---

## 卡 6｜Bottlenecks：Understanding Bottlenecks for Efficiently Serving LLM Inference with KV Offloading（arXiv 2601.19910；**MLSys'26 錄取未查證**）

- **讀了什麼**：〔全文〕arXiv v1（2025-12-16）。arXiv 的 Comments 欄寫「Submitted to MLSys 2026」；PDF 頁首是「Submission and Formatting Instructions for MLSys 2025」（樣板未改）。mlsys.org/virtual/2026/papers.html 抓到 173 篇，其中沒有這篇（2026-10-06）。查證 2026-10-06。
- **一句話**：KV 卸載到 CPU 後，prefill 何時從算力瓶頸變成 PCIe 瓶頸；定義臨界比值 κ_crit 並實測。
- **評測要證明的主張**：真實負載的 κ_ratio（已快取 ÷ 新 token）遠高於硬體的 κ_crit；延遲 99% 花在傳輸、GPU 只用到 28% TDP（p1）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | 實驗：Llama-3.1-70B（dense，GQA，B_kv＝328 KB）、Qwen3-235B-A22B（MoE，GQA，192 KB），FP16。分析表另含 405B、Qwen3-30B-A3B、DeepSeek-V3（MLA）；MLA 因實作開銷無法隔離 PCIe，延後 | 〔原文〕p7、p6–7 Table 3 |
| 硬體 | 實驗：8×H100 SXM5（NVLink 4.0）、AMD EPYC 7R13（48 核）、2 TB DDR4-3200、PCIe 5.0 ×16。**B200、A100 只出現在解析表與 roofline，沒有實測**〔複核補充：原文自己的引言（p1）寫「在 H100 與 B200 系統上套用框架，量測端到端延遲、GPU 使用率與排程」，但實驗設定（p7）只有 8×H100 這台，§6 的量測也都沒標 B200。原文前後不一致；以實驗設定為準〕 | 〔原文〕p1、p7 |
| 軟體與版本 | vLLM v0.10.1＋LMCache v0.3.5；**關閉 prefix caching** 以隔離 PCIe；輸出限 1 token 模擬 prefill-only 伺服器；改寫 vLLM scheduler 記錄每次 iteration；TP 大小**沒寫** | 〔原文〕p7 |
| 資料／負載 | 微基準：prompt＝K 個「Hi」＋T 個隨機單 token 字。負載量測：依經驗 κ_ratio 分布，從 ShareGPT 與 NarrativeQA 取 1,000 個請求。κ_ratio 的統計：ShareGPT（90k 段對話）、NarrativeQA（文件中位 57K，15K–190K）、FinQA（文件中位 167K，60K–450K） | 〔原文〕p5–7 |
| 長度 | 實驗：K＝1,024–65,536（2 的冪）、T＝64–2,048（Fig. 6，圖讀值）。FinQA 的 60K–450K **只用在 κ_ratio 統計** | 〔原文〕〔原文圖讀值〕p6、p9 |
| I/O 怎麼實現 | **真實 CPU DRAM→GPU（PCIe 5.0）**，經 LMCache。實測持續頻寬 15 GB/s＝單向峰值 64 GB/s 的 23%；原文列出的可能原因是 CPU-GPU memcpy 開銷、NUMA、傳輸粒度，**沒有逐一追證** | 〔原文〕p8 |
| 重算與載入成本怎麼量 | 解析：TTFT＝K·B_kv/BW＋T·F_pf/C_eff；F_pf＝2N（每 token 的 FLOPs，**省略 context 項**）；κ_crit＝(F_pf/B_kv)×(BW/C_eff)。實測：先在關閉卸載時量 t_GPU，再量卸載請求，差值即 PCIe 開銷；每個設定 200 個請求、30 s 暖機、nvidia-smi 每 200 ms 取樣 | 〔原文〕p4、p7 |
| 到達與併發 | 微基準：每設定 200 個請求（vLLM benchmark tool）；負載量測：**以 70 與 130 RPS 重播**、穩態後跑 5 分鐘（到達分布沒寫） | 〔原文〕p7 |
| 重用結構 | 合成的「K 已快取＋T 新」；多輪的 K＝累積歷史、文件問答的 K＝文件長 | 〔原文〕p5 |
| 掃描的自變數 | K、T、模型、RPS；解析部分另掃 GPU（A100/H100/B200）與 PCIe 世代 | 〔原文〕p6–9 |
| 對手 | 無（量測研究） | 〔原文〕 |
| 系統指標 | mean TTFT±std、PCIe 開銷 P_OH＝t_PCIe/t_GPU、每 iteration 排入的 token 數、GPU 平均／最大功耗 | 〔原文〕p5、p8 |
| 品質指標 | 無 | 〔原文〕 |
| 主要結果 | κ_crit 解析範圍 1–76（多數 <15，p1）；實測 κ_crit：Llama 2、Qwen 1（解析用峰值頻寬估 14.3、7.8；用 15 GB/s 重估 3.3、1.8）（p8）；Qwen、K≈65K、T=64：P_OH＝86，即 99% 時間在傳輸（p8–9）；K=8K、T=128 時 88%（p8）；GPU 平均只用 28%（ShareGPT）、22%（NarrativeQA）的 TDP（p8） | 〔原文〕p1、p8–9 |
| 消融／敏感度／開銷 | Fig. 5（κ_crit 實測）、Fig. 6（K×T 網格）、Table 4（排程 token 數）、Table 5（功耗）；系統建議：NVLink C2C 使 κ_crit ×5.3、統一 HBM 再 ×9、MLA、KV 量化、利用率感知排程（p9–10） | 〔原文〕p8–10 |
| 重複與統計 | 每設定 200 個請求，報平均與標準差（誤差棒＝TTFT 標準差） | 〔原文〕p7–8 |
| 程式碼／資料 | 原文沒有程式碼連結 | 〔原文〕 |
| 長度範圍與停止理由 | 微基準最大 K＝65,536，原文沒說理由 | 〔原文〕 |
| 設計理由（原文） | 關閉 prefix caching 以隔離 PCIe；輸出 1 token 模擬 prefill-only；用「Hi」重複精確控制 K 並觸發卸載（p7）；把模型因子 κ_M 與硬體因子 κ_HW 分開，讓軟硬體分開討論（p1、p4） | 〔原文〕 |
| 設計理由〔判讀〕 | F_pf＝2N 讓 κ_crit 與位置無關、可以查表；代價是長 context 時低估了重算成本 | 〔判讀〕 |
| 原文沒講清楚的地方 | TP 大小；LMCache 的 chunk 大小與設定；RPS 重播的到達分布；DeepSeek 用的是 V2 還是 V3（實驗段寫 V2、分析寫 V3） | 〔原文〕p7 |
| 與既有整理不一致 | ① 會議別：workloads_eval、PAPERS_BY_LEVEL 與本次任務都寫 MLSys'26，**未查證**（見「讀了什麼」）；intro 的參考文獻 [53] 只寫 arXiv 2026，較穩妥。② intro §7.2「實測只用到 23%（15 GB/s），99% 的延遲花在傳輸上」數字一致，但 **99% 只限 Qwen3-235B、K≈65K、T=64 這一格**；K=8K、T=128 是 88%。〔複核補充：原文摘要（p1）也是不帶條件地寫「99% of latency spent on transfers」，intro 是照摘要轉述；條件只出現在 p8 與 Fig. 6 圖說（p9）〕③ workloads_eval 的 Bottlenecks 列：硬體「H100、B200」→ 實測只有 H100〔複核：workloads_eval 這樣寫有原文依據（p1 引言宣稱 H100 與 B200 都量），錯在原文自己前後不一致，不是轉述錯；建議改成「實測 H100（引言稱含 B200，實驗段未見）」〕；「FinQA 文件 60K–450K」→ 只用在統計，實驗的 K 最大 65,536；到達過程分類為「無」→ 負載量測有以 70／130 RPS 重播。④ PAPERS_BY_LEVEL「把頻寬當成硬體給定，建議換更快的硬體」：原文也提了 MLA、KV 量化、利用率感知排程，並列出 memcpy、NUMA、粒度三個軟體因素，只是沒追證，屬過度簡化。⑤ PAPERS_BY_LEVEL「κ_crit 和我們的 κ 代數上是同一個量」〔計算〕一致：κ_crit＝(F_pf/C_eff)÷(B_kv/BW)＝每 token 重算時間÷每 token 傳輸時間，前提是 F_pf 與位置無關。⑥ **新發現〔計算〕**：原文說省略的 context 項在 65K 時約讓 F_pf 增加 10%（p10）。用原文自己引用的 Kaplan 式（2·n_layer·n_ctx·d_attn），在 n_ctx＝65,536 時：Llama-3.1-70B（80 層、d_attn＝64×128）為 2N 的 +61%；Qwen3-235B-A22B（94 層、d_attn＝64×128，HF config.json）為 +229%；連 405B（126 層、d_attn＝128×128）都是 +33%。所以「約 10%」與它自己的公式不符，κ_crit 在長 context 下被低估 | 〔原文〕〔計算〕 |
| 對本研究的意義〔判讀〕 | 提供「CPU 層真實搬運」的基準數字（15 GB/s，PCIe 5.0、LMCache v0.3.5）與實驗手法（關 prefix cache、1 token 輸出、K×T 網格）。它的 κ 與位置無關，正好是我們要補的：在 128K–512K，attention 項主導重算，κ 必須隨位置變 | 〔判讀〕 |

---

## 卡 7｜py-kvcache：Building py-kvcache: A Performance Characterization of External KV Caching for vLLM with NVMe SSDs（arXiv 2609.11744）

- **讀了什麼**：〔全文〕arXiv v1（2026-09-10）；〔程式碼〕GitHub `atlarge-research/py-kvcache` commit `3abba7a`（2026-07-26）的 `break_even.py`、`reactor.py`、`vllm.py`、`load_planner.py`。查證 2026-10-06。
- **一句話**：量 vLLM 外接 KV 快取（CPU／NVMe）何時比重算快，並做一個帶預載與損益平衡門檻的 KV Offload connector。
- **評測要證明的主張**：外接快取的效益取決於傳輸粒度、中介記憶體與傳輸進入排程的時機，不只是裝置頻寬；py-kvcache 在 80k 時磁碟載入比 LMCache 快 2.0×，三層時比 LMCache 快 1.23×、與原生 vLLM Offload 差約 4%；外接快取應視為依設定而定的准入決策（p1）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | Llama 3.2 3B Instruct（114,688 B／token）、Qwen3 4B Instruct 2507；KV 一律 FP16（「最壞情況」） | 〔原文〕p7、p12 |
| 硬體 | 本地節點：2×RTX 4000 Ada（20 GiB、PCIe 4.0）、Xeon Silver 4514Y、256 GiB DDR5、Kioxia CM7-R（PCIe 5.0）與 Samsung PM9A3。Snellius 節點：4×H100（94 GiB、PCIe 5.0）、雙 EPYC 9334、768 GiB、Samsung PM1743。**所有實驗只用單卡**；節點獨占 | 〔原文〕p9 Tables 2–3 |
| 軟體與版本 | vLLM v0.16–v0.22（隨上游演進，評估用自己的 v0.22 fork）；LMCache v0.4.7；llm-d（自己加 direct I/O 的 fork）；py-kvcache 用 io_uring（liburing Python 綁定）＋O_DIRECT；本地 CUDA 13.2／PyTorch 2.11，Snellius CUDA 12.8／PyTorch 2.11 | 〔原文〕p6–9、p14 |
| 資料／負載 | 合成 long-document 基準（可控文件長、重用比例、前綴長、併發、到達率）；ShareGPT（vllm bench，>10,000 個 prompt、32 req/s）；LongBench（Multi-Doc QA、Code Repo）；SCBench（KV 任務，round-robin 重播）；Alibaba Bailian（Coder、A、B 三份 trace，合成 prompt 重建） | 〔原文〕p7–8、p10、p18–20 |
| 長度 | 1k–80k，**二進位單位**（80k＝81,920 token） | 〔原文〕p7 |
| I/O 怎麼實現 | **全部真實**：NVMe 以 O_DIRECT 避開 page cache；fio 量到 Kioxia 13.5 GB/s、PM9A3 約 6 GB/s；nvbandwidth 量到 CPU↔GPU 本地 25–26 GB/s（PCIe 4.0）、Snellius 54–56 GB/s（PCIe 5.0）。用 256-token block、每 block 一個檔案（3B 模型約 28 MiB）。GPUDirect Storage（KvikIO）原型反而最慢 | 〔原文〕p10、p13–14、p16 |
| 重算與載入成本怎麼量 | **端到端量測**：關閉快取量 cold TTFT f(x)，再量 cache-hit TTFT g(x)，內插得到 Pareto 邊界 g(P)＋f(D−P)＜f(D)（p8）。由此算「打平所需頻寬」：max_io(D)＝f(D)−(g(D)−t_copy(D))（p12）。門檻是每個節點、每個模型**離線校準一次**，寫成檔案，服務時不再量（p17）。只看前綴長度，不看位置 | 〔原文〕p8、p11–12、p17 |
| 到達與併發 | long-document 基準：先送 N 個不同文件，再隨機順序重送 R 次；磁碟實驗 50 個併發請求；混合負載 50 個 80k 請求、50% 重用、最大併發 8；ShareGPT 32 req/s；Bailian 依 trace 重播 | 〔原文〕p10、p17–18 |
| 重用結構 | 重複長文件、前綴鏈（LongBench）、多輪共享 context（SCBench）、生產 trace 的前綴重用 | 〔原文〕p7–8 |
| 掃描的自變數 | 文件長、重用比例、`max_num_batched_tokens`、併發、預載開關、層（GPU/CPU/disk）、I/O 引擎與請求大小、檔案數（metadata） | 〔原文〕p10–19 |
| 對手 | LMCache v0.4.7、原生 vLLM KV Offload（CPU＋檔案系統層）、llm-d（只在特性分析）、GPU prefix caching、Full compute | 〔原文〕p6、p17–18 |
| 系統指標 | TTFT（query TTFT 平均）、總執行時間、各段時間（lookup、CPU↔GPU、disk）、頻寬（GB/s）、命中率、break-even 點 | 〔原文〕p4 |
| 品質指標 | 無（只用 LongBench、SCBench 的資料，不評分） | 〔原文〕p8 |
| 主要結果 | ① break-even（Snellius、Llama 3.2 3B、llm-d）：8k 需 77.8% 重用、80k 只需 7.8%（p11）。② 打平所需頻寬：1k 23.2 GB/s、8k 10.4 GB/s、80k 3.5 GB/s；約 8k 以下沒有任何量到的 SSD 能比 GPU 重算快，中階 SSD 的門檻到 32k（p12）。③ 磁碟：比 LMCache 快 1.5×（不預載）、2.0–2.5×（預載）；預載本身 1.66×（40k）、1.34×（80k）（p17–18）。④ Bailian 在 H100 上：平均請求低於 Qwen3 4B 的 SSD 門檻 6,203 token，外接快取與只用 GPU prefix cache 同 TTFT（p20） | 〔原文〕p11–20 |
| 消融／敏感度／開銷 | 介面比較（Transfer vs Offload API）、傳輸粒度、I/O 引擎、metadata 擴展性、預載、staging 上限（SCBench：原生 Offload 讀了 3.4 TB、py-kvcache 85 GB） | 〔原文〕p10–19 |
| 重複與統計 | 每設定重複 3 次取平均；暖機前先送小請求（首請求 TTFT 高 2–5×） | 〔原文〕p7 |
| 程式碼／資料 | 公開：py-kvcache（Apache-2.0）、實驗腳本 `atlarge-research/kvcache-experiments`（後者未讀） | 〔原文〕p1、p9；〔程式碼〕 |
| 長度範圍與停止理由 | 80k 為上限，原文沒說理由（判讀：本地節點 20 GiB 顯存，80k 的 KV 約 8.75 GiB，見 p3） | 〔原文〕〔判讀〕 |
| 設計理由（原文） | FP16 KV 測最壞情況；1 個輸出 token 隔離 prefill；O_DIRECT 避免 page cache 扭曲重讀；兩台機器的 GPU 速度與顯存落在 break-even 的兩側；預載只允許一個、需求來時讓位，因為無上限的預載會吃光 CPU 記憶體（p7、p9、p16、p20） | 〔原文〕 |
| 設計理由〔判讀〕 | 它把「值不值得載」從頻寬問題改成關鍵路徑問題：同樣的位元組，早一點開始讀就能藏進排隊時間 | 〔判讀〕 |
| 原文沒講清楚的地方 | 為何停在 80k；混合負載以外的到達分布；Pareto 邊界的內插方法（程式碼有 `_pchip.py`，判讀為 PCHIP，未讀） | 〔原文缺漏〕〔程式碼〕 |
| 與既有整理不一致 | ① sota §2.9、PAPERS_BY_LEVEL「短於這個長度的前綴**就不存也不載**」：**「不存」不成立**。原文的門檻在 lookup 時拒絕載入、改由 vLLM 重算（p17）；程式碼的 `prepare_store` 會存所有尚未存在的 block（`vllm.py` L338–371），門檻只在 `reactor.py` 的 `_should_decline_load`（L701、L737）〔程式碼 3abba7a〕。原文的「Characterization Takeaway」確實說「應避免載入或儲存」低於門檻的前綴（p14），但實作只做了前者。〔複核：判為部分成立。「不存」若當成 py-kvcache **做了什麼**是錯的（程式碼已在 commit 3abba7a 重新下載確認 L338 `prepare_store`、L701／L737 `_should_decline_load`）；但若當成原文的**主張**，p2（「cache admission and bypass policies instead of unconditional loading and storing」）與 p14 都支持。sota §2.9 放在「怎麼做」底下，所以應改成「主張不存也不載；實作只擋載入」〕② intro §7.5、表 7、表 14「py-kvcache 認為**值不值得存**是准入決策」：原文用的詞是「admission decision」（p1），實作是**載入准入**，應改成「值不值得載」。〔複核修正：此指控過強，不成立。intro §7.5 說的是 py-kvcache「認為」什麼，也就是原文的主張；原文 p1 的「setup specific admission decision」、p2 的「admission and bypass policies instead of unconditional loading and storing」、p14 的「avoid loading or storing」都包含「存」。「值不值得存」是對原文主張的合理轉述，不需要改成「值不值得載」；只需在描述**實作**時（表 14「整段准入」）加註「實作只擋載入」〕③「整段判斷，不看位置」**一致**（`should_load(prefix_tokens, ...)`）。④ PAPERS_BY_LEVEL「決定發生在讀取時」**一致**。⑤ sota 標〔摘要＋部分〕，本次已讀全文 | 〔原文〕〔程式碼〕 |
| 對本研究的意義〔判讀〕 | 直接對應我們要用的 vLLM KV Offload／OffloadingConnector 路徑：Offload API 自 v0.11.0、多層自 v0.22.0（p4、p6），每次搬運大小跟著 `max_num_batched_tokens`（p11）。可沿用：真實裝置的量法（fio、nvbandwidth）、Pareto 打平邊界、3 次重複、二進位長度單位要寫明。break-even 門檻是我們「整段准入」對照組的現成實作。**要小心**：它的門檻只看整段長度，而且只決定載不載，寫入照存，這正是本研究要補的寫入端 | 〔判讀〕 |

---

## 本組對 PoC 設計的建議〔判讀〕

對象：下一步「在 vLLM 上重現 Cake 式還原＋限速器，量 16K–512K 的只算、只載、Cake 式」（intro 表 19 的 H1）。

1. **先校準成本，再重現加速**。依重現規格卡的四步：Cake Fig. 4 式的逐 chunk 重算曲線 → 限速器自檢 → 會合點預測 vs 實測 → 一個 Cake 表格格子的趨勢。第三步不受硬體影響，最能抓實作錯誤。
2. **限速器放在「儲存→CPU」，CPU→GPU 保持真實**。這最接近 Cake 的實際做法（v1 p7、App. A），也和 py-kvcache、vLLM Offload 的路徑一致（先進 CPU staging，再 DMA 到 GPU）。延遲公式在 Cake 的「大小÷頻寬」之外加上「每次 I/O 的固定成本」，並把 Gbps 的單位寫死在設定檔裡（Cake 圖讀值偏向二進位）。
3. **在至少三個點對照真實裝置**。Cake 一點都沒做。可比照 py-kvcache（fio、nvbandwidth、O_DIRECT）與 Bottlenecks（關 prefix cache、1 token 輸出）量 CPU 層與 SSD 的有效頻寬，報出「限速器 vs 真實」的差距。
4. **I/O chunk 大小要掃**。Cake 的 128 是經驗值（p5）；CacheFlow Fig. 2 顯示 36K 以下 token 維度輸給 layer 維度，HCache 顯示不規則 GEMM 會慢 7–12%。至少掃 128／256／512，並記錄每個 chunk 的固定開銷。
5. **對手至少五個**：只算（同 budget 的 chunked prefill）、只載（OffloadingConnector 或 LMCache，記版本）、Cake 式 token 維度、layer 維度（HCache／CacheFlow 式，作為「另一種切法」）、Pensieve 式佈局（前段丟、中段 CPU、後段 GPU，即 E1 的 P2）。py-kvcache 的整段門檻當「整段准入」對照。CacheFlow 釋出程式碼後再加。
6. **GPU 忙碌的模擬要寫清楚並一致套用**。沿用 Cake 的 token budget 佔比，但要明確背景請求的長度與型態，而且只算、只載基線也承受同樣的背景負載。
7. **報法**：TTFT 的平均、p50、p99，每點至少 3 次並附信賴區間；同時報兩個加速比（對只算、對只載）、會合點的位置（佔總長的比例）、GPU 與 I/O 使用率（比照 CacheFlow Fig. 5）、實際有效頻寬。限速的結果一律標「模擬」。
8. **長 context 的成本要用量的，不用 FLOPs 公式**。Bottlenecks 的 F_pf＝2N 在 65K 就低估 61–229%〔計算〕〔複核修正措辭：省略的注意力項相當於 2N 的 +61%（Llama-3.1-70B）到 +229%（Qwen3-235B-A22B），亦即 2N 只有真值的 62%／30%；算式依原文 p10 引用的 2·n_layer·n_ctx·d_attn，Qwen3 的 94 層、64×128 已用 HF config.json 重新確認〕；Pensieve Fig. 4 在 16K 時 attention 已是非 attention 的 1.5 倍。128K–512K 的 f(i) 要實測，必要時抽點。
9. **模型選原生長 context 的 GQA 模型**，依專案規則不開 YaRN；並記錄每 token 的 KV 大小。例：Llama-3.1-8B（128 KiB／token）在 512K 時 KV 為 68.7 GB，32 Gbps 只載要 17.2 s〔計算，10⁹ bit/s〕。
10. **把「Cake 輸給只載」的格子當成必測點**（Cake Table 6 的 0.80×：GQA 大模型、GPU 很忙、32 Gbps），這正是 H1 的否證條件所在。

---

## 未查證清單

| 項目 | 狀態 | 下一步 |
|:--|:--|:--|
| Cake 的 OpenReview 審稿意見、是否附補充材料或程式碼 | OpenReview（WOyOtaO6lQ、cK0kUzocJW）被人機驗證擋下 | 用瀏覽器手動開啟 |
| Cake 是否有任何公開實作 | 作者頁、PMLR、ICML、GitHub 搜尋皆無；二作的 LMCache fork 關聯不明 | 寫信問作者，或讀 xenshinu/LMCache 的 patch-1 分支 diff |
| Cake 的 H100 型號、PyTorch／CUDA 版本、Fig. 6 硬體、70B FP8 kernel | 原文未寫 | 無法查證，重現時自行決定並記錄 |
| CacheFlow 的 I/O 實作（限速或真實）、Cake 基線怎麼做 | 原文未寫 | 等程式碼釋出（接受後） |
| Bottlenecks 是否被 MLSys'26 錄取 | arXiv 只寫「Submitted」；mlsys.org 2026 列表 173 篇中沒有 | 查 MLSys 2026 proceedings 或作者頁 |
| Bottlenecks 的 TP 大小、LMCache 設定 | 原文未寫 | — |
| HCache EuroSys 定稿與 arXiv v1 是否有差 | 判讀為同版（arXiv v1 已含 ACM 版權頁） | 比對 ACM DL 版 |
| HCache、Pensieve 程式碼 | GitHub 搜尋未找到 | 寫信問作者 |
| KVPR arXiv 版與 ACL 版差異、程式碼內容 | 只讀 ACL 版；repo 未讀 | 需要時再讀 |
| Pensieve Fig. 4 用的模型 | 原文未寫 | — |
| py-kvcache 實驗腳本 repo（kvcache-experiments）、Pareto 內插方法（`_pchip.py`） | 未讀 | 重現 break-even 時再讀 |
| Cake Fig. 4 的 Gbps 單位判讀 | 讀圖精度不足〔複核更新：800 dpi 數位化得 62.4 ms，支持 256 MiB÷4 GiB/s；仍屬圖讀值〕 | 重現時以我們自己的設定為準 |
| Cake 原文內部不一致（複核新增） | Table 3 H100 100% 32 Gbps＝2.40，內文寫 2.23（p6）；App. B.2 的 chunk＝budget 與 §5.1 的 I/O chunk 128 | 以表與 §5.1 為準，重現時記錄選擇 |
| Bottlenecks 引言稱 B200 有量測（複核新增） | p1 有、p7 實驗設定沒有 | 以 p7 為準 |

---

## 複核紀錄

- **複核者**：V03（未參與抽取；未讀抽取者的推理或筆記，只用原文）
- **日期**：2026-10-07
- **原文來源**：scratchpad/E03 的 PDF 先用 `pdfinfo` 與首頁文字確認身分與版本（Cake ICML＝PMLR 267 首頁版權列、13 頁；v1＝2410.03065v1 4 Oct 2024、16 頁；v2＝20 Feb 2025、12 頁；CacheFlow v1 28 Apr 2026／v2 26 Sep 2026；Pensieve v3 7 Oct 2024 含 DOI …3696086；HCache v1 含 DOI …3696072；KVPR ACL Findings 印刷頁 19474–19488；Bottlenecks v1 16 Dec 2025；py-kvcache v1 10 Sep 2026），再由複核者自己重新 `pdftotext` 到 scratchpad/V03。圖表數字用 `pdftoppm` 轉 PNG 看圖；Cake Fig. 4 另以 800 dpi 依座標軸刻度做像素數位化。py-kvcache 程式碼在 commit `3abba7a`（GitHub API 確認 2026-07-26）重新下載比對行號；KVPR repo 最新 commit `1712a52`（2025-05-28）以 GitHub API 確認；Qwen3-235B-A22B 的層數與 head 數以 HF `config.json` 確認。
- **方法**：Cake 卡與 Cake 重現規格卡**逐格、逐數字**核對（Table 2–7 每個被引用的格子、Fig. 3／4 數值、所有版本號、頻寬點、chunk 大小、節號頁碼）；Cake v2 與 ICML 的所有「x\y」倍數用程式逐一比對（ICML 172 組、v2 160 組，差異只有 ICML 新增的 Table 4 的 12 組）。其他六張卡逐格回原文確認。

### 每張卡的檢查格數與判定

| 區塊 | 檢查格數 | ✅ | ❌ | ⚠️ |
|:--|--:|--:|--:|--:|
| 來源清單 | 7 | 7 | 0 | 0 |
| 本組的共同模式 | 6 | 4 | 1 | 1 |
| 卡 1 Cake（標頭 3＋表 23） | 26 | 23 | 1 | 2 |
| Cake 版本差異 | 6 | 6 | 0 | 0 |
| 卡 1b Cake 重現規格卡（表 10＋缺漏清單 15） | 25 | 21 | 0 | 4 |
| 卡 2 CacheFlow（3＋24） | 27 | 25 | 0 | 2 |
| 卡 3 Pensieve（3＋23） | 26 | 26 | 0 | 0 |
| 卡 4 HCache（3＋23） | 26 | 23 | 1 | 2 |
| 卡 5 KVPR（3＋23） | 26 | 25 | 1 | 0 |
| 卡 6 Bottlenecks（3＋23） | 26 | 24 | 0 | 2 |
| 卡 7 py-kvcache（3＋23） | 26 | 24 | 1 | 1 |
| 本組對 PoC 設計的建議 | 10 | 9 | 0 | 1 |
| **合計** | **237** | **217** | **5** | **15** |

（「與既有整理不一致」一格內有多條指控，該格只要有一條需修正就計為 ❌ 或 ⚠️；逐條判定見下方指控表。純補充、原內容無誤的格子計為 ✅。）

### 逐條修改（原內容 → 新內容＋出處）

**❌ 錯誤（5）**
1. 共同模式 #4：「沒有一篇在寫入時決定」→「沒有一篇依『位置』在寫入時決定」，並註明 HCache 的 state partition 決定每層存 hidden／KV／不存（HCache p7 §4.2、p10 Table 3）。
2. Cake「設計理由〔判讀〕」：I/O 線與重算曲線交點「A100 第 17 個 chunk（≈8.7K）、H100 第 22 個（≈11K）」→「A100 第 18–19 個（≈9.2–9.7K）、H100 第 33 個（≈16.9K）」（ICML p4 Fig. 4，800 dpi 數位化：A100 chunk 18＝62.2 ms、20＝63.7 ms；H100 chunk 32＝61.4 ms、33＝62.6 ms；I/O 線 62.4 ms）；另註明 Fig. 4 是 7B、主表是 13B。
3. HCache「硬體」：敏感度平台漏了 RTX 4090；Table 2 實為 A100／A30／4090／L20／H800，傳輸 32 GB/s 的是 A100／A30／4090／L20（HCache p9 Table 2、p11 Fig. 11a）。
4. KVPR「長度」：「prompt 256／512／1024（附錄另有 128）」→ 延遲實驗 Fig. 7 是 prompt 128／256／512（35.8% 即 prompt 128＋生成 128），吞吐實驗 Fig. 6 才是 256／512／1024（KVPR p7）。
5. py-kvcache「與既有整理不一致」②：「intro 的『值不值得存』應改成『值不值得載』」→ 指控不成立；原文主張本身包含「存」（p1 admission decision、p2「instead of unconditional loading and storing」、p14「avoid loading or storing」），只需在描述實作時註明只擋載入。

**⚠️ 不精確或需補條件（15）**
1. 共同模式 #3：「停在 128K 以下」→「≤128K」（CacheFlow v2 p8 Fig. 7 到 128K）。
2. Cake「長度」：v1 的描述補上文字的 5K–16K 取 20 點、Fig. 6 的 14K（v1 p7–8）。
3. Cake「與既有整理不一致」⑤：GPU 使用率的定義其實有寫（budget 佔比，ICML p6），沒寫的是背景請求的組成；「過度樂觀」結論維持。
4. 重現卡「建議先重現哪一張」：「單卡、7B 的唯一格」→「單卡、7B、chunk 512 的唯一格」（Table 4 有 6 個單卡 7B 格，ICML p7）；另補二進位解讀下 128-token chunk 為 15.6 ms〔計算〕。
5. 重現卡缺漏 #5：補 App. A Algorithm 1 用兩個常數 COMP_CHUNK_SIZE／FETCH_CHUNK_SIZE（p12），而 App. B.2 寫依 token budget 切 chunk 再倒序推進佇列（p13），與 §5.1 的 128 關係未交代。
6. 重現卡缺漏 #6：原寫停止方式「沒寫」不完整；Algorithm 1 第 3–5 行有「Signal I/O worker to stop」（p12），B.2 第 5 步中斷 prefill 開始生成（p13）；沒寫的只剩是否取消在途讀取。
7. 重現卡缺漏 #12：「V0／V1 engine、V1 預設 chunked prefill」不在任何原文中，標〔判讀，未附出處〕。
8. CacheFlow「位置與層的成本」：Fig. 2「最多約 20%」→ 最左點（約 11K）中位數約 −22%，陰影帶到約 −34%（v2 p4）。
9. CacheFlow「與既有整理不一致」②：判為部分成立（字面對 planner 成立，問題在上下文標題）。
10. HCache「模型」：「FP16」原文未明寫，只有 Table 2 的 FLOPS 註明 FP16（p9），改標〔判讀〕。
11. HCache「長度範圍與停止理由」：「1–3 個長 context」出處 p4 → p1、p10。
12. Bottlenecks「硬體」：補原文 p1 引言自稱量了 H100 與 B200，與 p7 實驗設定只有 H100 不一致。
13. Bottlenecks「與既有整理不一致」③：workloads_eval 寫「H100、B200」有原文 p1 依據，是原文自相矛盾，不是轉述錯。
14. py-kvcache「與既有整理不一致」①：判為部分成立（實作不擋存，但原文主張 p2、p14 包含不存）；行號已在 commit 3abba7a 重新確認。
15. PoC 建議 #8：「低估 61–229%」措辭改為「省略項相當於 2N 的 +61%～+229%（2N 只有真值的 62%／30%）」。

**〔複核補充〕（原內容無誤，補上重現會用到的事實）**：Cake Fig. 4 數位化數值與 SSD 線 62.4 ms（支持二進位單位）；Cake Table 3 內文 2.23 與表 2.40 不一致（ICML p6）；xenshinu/LMCache fork 的 commit 內容（GitHub API）；CacheFlow Reproducibility Statement 宣稱 §4.1 有 baseline 實作但實際沒有（v2 p10）；Pensieve「最早」一說 sota §2.4 也有；HCache Table 3 每 token 儲存量恰為 FP16 計算值的一半〔計算〕；KVPR A.4 的 l＝182 超出 0 ≤ l ≤ s＝128〔計算〕；Bottlenecks 摘要本身就不帶條件寫 99%；未查證清單新增兩列。

### 對使用者文件（intro／sota／workloads_eval／PAPERS_BY_LEVEL）的指控逐條判定

| 卡 | 條目 | 判定 | 依據 |
|:--|:--|:--|:--|
| Cake | ① 72K／Llama2-70B／30 秒只在 v1／v2 p1，ICML 已刪，[1] 引 ICML | ✅ 指控成立 | v1 p1、v2 p1 有；ICML 全文無；intro [1] 標 ICML |
| Cake | ② 「16K 左右是 Cake 自己說最划算的區域」 | ✅ 指控成立 | ICML p6 只說兩種資源相當時最有利，未綁 16K |
| Cake | ③ 4K–16K、7–100 Gbps、2.6× 一致但要加條件 | ✅（一致性確認＋條件成立） | ICML p1、p5、p6 Table 3；只有 Table 3 用滿 5 個頻寬點 |
| Cake | ④ 「CacheFlow 自己在 LMCache 上重做 Cake」原文沒寫 | ✅ 指控成立 | CacheFlow v1 p7、v2 p7、p10 |
| Cake | ⑤ 限速器「簡單公開、可重做表格」過度樂觀 | ⚠️ 部分成立 | GPU 使用率定義有寫（p6），其餘缺漏屬實 |
| Cake | ⑥ sota 掃描維度漏 chunk 大小 | ✅ | ICML p7 Table 4；v2 無此表 |
| Cake | ⑦ workloads_eval：budget 512／1024 是 v2 寫法、§5.7 是 v2 節號、漏 1×A100 | ✅ | v2 p5、ICML p5、p8、p6 Table 3 |
| Cake | ⑧ PAPERS_BY_LEVEL「12.5% 時慢 25%」算術一致但缺條件 | ✅ | ICML p8 Table 6 0.80 |
| Cake | ⑨ intro 表 1 與「80% 在磁碟層」一致 | ✅（確認） | ICML p2 Fig. 1 |
| Cake | ⑩ 「precompute and store … in advance」逐字一致 | ✅（確認） | ICML p5 |
| Cake | ⑪ intro／sota 沒提 +26%；workloads_eval 正確但節號是 v2 | ✅ | ICML p8 §5.8；v2 §5.7 |
| CacheFlow | ① PAPERS_BY_LEVEL 的切點公式來自 v1 Eq. (1)，v2 已改 DP | ✅ | v1 p5；v2 p5–6 |
| CacheFlow | ② 「載入時間也是位元組÷頻寬」不能推論實驗是模擬 | ⚠️ 部分成立 | v2 p5 Eq. 2；實驗 I/O 未寫 |
| CacheFlow | ③ 「自己在 LMCache 上重做 Cake」原文沒寫 | ✅ 指控成立 | v2 p7、p10；v1 p7 |
| CacheFlow | ④–⑧ 2.40–2.63、假設全可載、92%／98%、128K、資源不干擾 | ✅（一致性確認） | v2 p6、p8–9、p14 |
| CacheFlow | ⑨ 頻寬點對應與 Cake Table 2 不同 | ✅ | v2 p7 vs Cake ICML p6 |
| Pensieve | ①–⑥ | ✅ 全部成立／一致 | p9–13 |
| HCache | ① 「最長 16K」漏了 Fig. 11(i) 到 32K | ✅ 指控成立（32K 只在敏感度實驗） | p11 Fig. 11(i) |
| HCache | ②–⑤ | ✅（一致性確認） | p5、p10、p13 |
| KVPR | ① sota「只支援單 GPU」→ 單 GPU＋資料平行 | ✅ 指控成立 | p9 Limitations |
| KVPR | ② 沒提「只優化解碼」 | ✅ | p6 |
| KVPR | ③④ | ✅（確認） | p9；workloads_eval |
| Bottlenecks | ① MLSys'26 錄取未查證 | ✅ | arXiv Comments「Submitted to MLSys 2026」；mlsys.org 173 篇無此篇 |
| Bottlenecks | ② 99% 要加條件 | ✅（但原文摘要本身沒加） | p1、p8、p9 |
| Bottlenecks | ③ workloads_eval：B200／FinQA／到達過程 | ⚠️ 部分成立：FinQA 與 70／130 RPS 成立；「B200」有原文 p1 依據 | p1、p6–7 |
| Bottlenecks | ④ PAPERS_BY_LEVEL 過度簡化 | ✅ | p8–10 |
| Bottlenecks | ⑤ κ_crit 與 κ 同一個量 | ✅（代數確認） | p4 |
| Bottlenecks | ⑥ 「約 10%」與原文自引公式不符 | ✅（計算重現：+61%／+229%／+33%） | p10；HF config |
| py-kvcache | ① 「不存也不載」的「不存」不成立 | ⚠️ 部分成立：對實作成立，對原文主張不成立 | 程式碼 3abba7a；p2、p14、p17 |
| py-kvcache | ② 「值不值得存」應改「值不值得載」 | ❌ 指控不成立 | p1、p2、p14 |
| py-kvcache | ③④⑤ | ✅ | p17；程式碼；sota 標記 |

### 沒有檢查或只部分檢查的地方（如實記錄）

- **Cake**：OpenReview 審稿與補充材料仍未讀到（未嘗試繞過人機驗證）。Fig. 5、6、7 只核對圖說與內文數字，沒有數位化曲線。
- **其他六篇**：逐格核對時以「卡片引用到的頁」為主，用全文 grep 加讀相關頁；未逐頁通讀的有 Pensieve p1–3、p14–15，HCache p2、p14–16（參考文獻），KVPR p1–4、p10–15（參考文獻與附錄圖），Bottlenecks p2–6（只 grep 公式與資料統計），CacheFlow v1 p1–3、p8–11，py-kvcache p5、p15–16、p18–19、p21–24。這些頁上的事實若卡片沒有引用，就沒有被檢查。
- **圖讀值**：CacheFlow Fig. 5（Cake SM 36%、cache load 90%）與 Fig. 2、Pensieve Fig. 4、HCache Fig. 11、Bottlenecks Fig. 5–6 看過圖；CacheFlow Fig. 9–15、HCache Fig. 12–15、py-kvcache 各圖只核對內文數字，沒看圖。
- **程式碼**：py-kvcache 只確認卡片引用的行號（`vllm.py` L338、`reactor.py` L701／L731／L737）；`_pchip.py`、`kvcache-experiments` repo 未讀。KVPR repo 只確認 commit，未讀內容。
- **使用者文件**：intro.txt、sota.txt、workloads_eval.md、PAPERS_BY_LEVEL.md 只讀了被指控的段落與表格，未通讀。

### 結論：Cake 重現規格卡可否直接使用

可以使用，但要搭配本次修正。版本（vLLM v0.6.2、LMCache v0.1.4）、頻寬點（7／25／32／56／100 Gbps 與各表用到的子集）、chunk 大小（計算 512 預設、Table 4 掃 64–2048、I/O 128）、硬體、模型的 KV 大小、Table 3–7 被引用的每個倍數都和 ICML 原文一致。要改的是：Fig. 4 交點（H100 是第 33 個 chunk，不是第 22 個）、「單卡 7B 唯一格」限定 chunk 512、停止訊號其實有寫（Alg. 1）、App. B.2 與 §5.1 的 chunk 大小說法互相矛盾、Table 3 內文 2.23 與表上 2.40 不一致、V1 engine 的敘述沒有出處。
