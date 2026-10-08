# E08 稀疏注意力／卸載檢索，以及「批判既有評測」的 benchmark 論文

> 狀態：抽取完成（E08，2026-10-06）；**已複核（V08，2026-10-07）**，修正與補充見各格〔複核修正〕〔複核補充〕與檔尾「複核紀錄」。
> 頁碼一律是 PDF 頁（`pdftotext` 分頁）；程式碼附 commit 與行號。
> 引用原文一次不超過 15 個英文字，其餘皆為改寫。

## 範圍

1. 稀疏注意力與卸載檢索（逐篇評測卡）：MInference、DuoAttention、Quest、ShadowKV、InfiniGen。
2. 批判既有評測與統一 benchmark（評測卡＋逐條「評測陷阱」）：Yuan et al.（EMNLP'24 Findings）、Rethinking KV Cache Compression（MLSys'25）、Agrawal & Mayer（arXiv 2607.05399）、NVIDIA kvpress（repo＋leaderboard）。
3. 核對使用者文件（`intro.txt`、`sota.txt`）對 [56]／[31] Agrawal & Mayer 與 SCBench 的描述。

## 來源清單（全部於 2026-10-06 取得）

| # | 項目 | 讀的版本 | URL |
|:--|:--|:--|:--|
| 1 | MInference 1.0 | arXiv v2（2024-10-30，PDF 版頭為 NeurIPS 2024；arXiv comment「Accepted at NeurIPS 2024 (Spotlight)」），全文 28 頁 | https://arxiv.org/abs/2407.02490v2 ；code https://github.com/microsoft/MInference |
| 2 | DuoAttention | arXiv v1（2024-10-14），全文 21 頁。ICLR'25 會議版**未比對**，arXiv 與 repo README 都沒寫 venue〔未查證〕 | https://arxiv.org/abs/2410.10819v1 ；code https://github.com/mit-han-lab/duo-attention |
| 3 | Quest | arXiv v2（2024-08-26，PDF 版頭 ICML 2024／PMLR 235），全文 12 頁 | https://arxiv.org/abs/2406.10774v2 ；code https://github.com/mit-han-lab/Quest |
| 4 | ShadowKV | arXiv v3（2025-04-25），全文 22 頁；ICML'25 Spotlight 是 repo README 的說法〔文件〕，會議版未比對 | https://arxiv.org/abs/2410.21465v3 ；code https://github.com/ByteDance-Seed/ShadowKV |
| 5 | InfiniGen | arXiv v1（2024-06-28，arXiv comment「OSDI 2024」），全文 19 頁；USENIX 正式版未比對 | https://arxiv.org/abs/2406.19707v1 ；code https://github.com/snu-comparch/InfiniGen |
| 6 | Yuan et al., *KV Cache Compression, But What Must We Give in Return?* | arXiv v2（2024-10-08，含附錄 28 頁）；repo README 稱 EMNLP 2024 Findings | https://arxiv.org/abs/2407.01527v2 ；code https://github.com/henryzhongsc/longctx_bench |
| 7 | Gao et al., *Rethinking KV Cache Compression Techniques for LLM Serving* | arXiv v1（2025-03-31，PDF 版頭 MLSys 2025），全文 22 頁 | https://arxiv.org/abs/2503.24000v1 ；code https://github.com/LLMkvsys/rethink-kv-compression |
| 8 | Agrawal & Mayer, *Benchmarking KV-Cache Optimizations across Task Quality and System Performance for Long-Context Serving* | arXiv v1（**2026-05-03 提交**，雖然編號是 2607）；PDF 套 PVLDB 樣板但卷期為佔位字「XXX」，**不是已出版論文**；全文 13 頁＋artifact repo（commit `925fadf`，2026-05-09） | https://arxiv.org/abs/2607.05399v1 ；artifact https://github.com/agrawal-nikita/Benchmarking-KV-Cache-Optimizations-across-Task-Quality-and-System-Performance （原文寫 `nikagrwal/...`，現已轉址） |
| 9 | NVIDIA kvpress | repo commit `7136c22`（2026-10-05）：README、evaluation/ 全部腳本、pipeline.py、attention_patch.py、數個 press、speed_and_memory.ipynb；leaderboard Space commit `e81fc40`（2026-10-05）的檔案清單與 `src/textual_content.py`。相關論文 arXiv 2510.00636 **未讀** | https://github.com/NVIDIA/kvpress ；https://huggingface.co/spaces/nvidia/kvpress-leaderboard |
| — | 核對用：SCBench | arXiv v2（2025-03-11，ICLR 2025 版頭）p1–9 | https://arxiv.org/abs/2412.10319v2 |
| — | 核對用：LongBench 生成上限 | `LongBench/config/dataset2maxlen.json`（THUDM/LongBench main `2e00731`） | https://github.com/THUDM/LongBench |

〔複核修正〕上表的頁數是抽取者分頁檔多算一頁（最後一頁為空白分頁）。以 `pdfinfo` 為準：MInference **27** 頁、DuoAttention **20**、Quest **11**、ShadowKV **21**、InfiniGen **18**、Yuan **27**、Rethinking **21**、Agrawal 13、SCBench 31。各卡引用的頁碼（p 幾）經抽查與 PDF 頁一致，不受影響。版本皆已核對 PDF 版頭：MInference v2（2024-10-30）、DuoAttention v1（2024-10-14）、Quest v2（2024-08-26）、ShadowKV v3（2025-04-25）、InfiniGen v1（2024-06-28）、Yuan v2（2024-10-08）、Rethinking v1（2025-03-31）、Agrawal v1（2026-05-03）、SCBench v2（2025-03-11）。venue 補查（〔文件〕，2026-10-07）：ShadowKV README 寫「accepted by ICML 2025 as Spotlight」、longctx_bench README 寫「Accepted at EMNLP 2024 Findings」、duo-attention README 沒有寫 venue。

## 本組的共同模式

1. **幾乎都是單請求或固定 batch，沒有到達過程、沒有跨請求重用。** Quest（p8「single-batch」）與 Agrawal & Mayer（程式碼逐筆 batch 1）明確是一次一個請求；〔複核修正〕MInference（報的是「每個 prompt」的 prefill 延遲，p2、p9）與 DuoAttention（p9）**沒有寫 batch**，只能說沒有到達過程，不能斷定是 batch 1（兩卡的「到達與併發」格本來就標〔未查證〕）。ShadowKV、InfiniGen 有 batch，但只是同長度序列一起 decode。本組唯一有 Poisson 到達的是 Rethinking 的 request router（p11）。〔原文〕〔程式碼〕
2. **品質和效率在不同的堆疊、不同的設定上量。** 品質多在 HF transformers／PyTorch 參考實作上量，效率在自寫 kernel（FlashInfer、Triton、CUDA）或另一個引擎（LMDeploy）上量，而且模型、長度常常不同（Quest 品質用 LongChat-7B、效率用 Llama2-7B 組態，p5、p7；Agrawal 品質用 6 個資料集、系統用 3 個，p5——〔複核修正〕系統的 3 集中 Qasper 也在品質集內，只有 NarrativeQA、GovReport 是品質集沒有的）。〔原文〕〔複核補充〕Agrawal 的 artifact 顯示連品質評測本身也是兩套 harness：All KV／KIVI 用 KIVI 的 LongBench 腳本，SnapKV／CaM 用 kvpress 的 `evaluate.py`（見 Agrawal 卡）。〔程式碼〕
3. **「問題放在哪裡」決定誰贏。** 稀疏讀取法（Quest、ShadowKV）保留全部 KV、看到問題才挑；逐出法（H2O、SnapKV）在問題出現前就丟。各篇用不同協定處理：把問題逐 token 模擬成 decode（Quest p6）、把輸入最後 50 token 當 decode（DuoAttention p17）、把問題排除在壓縮外（kvpress README L277）。同一方法換協定，結論會反轉（Yuan p9 的 7-digit vs 64-digit passkey）。〔原文〕〔文件〕〔複核補充〕Agrawal 的 SnapKV 品質數字是在 kvpress 的 `query_aware: true` 下跑的（問題被接到 context 後一起壓），論文沒有寫；這正是 kvpress README L277 說會「偏袒 SnapKV」的設定。〔程式碼〕artifact `925fadf` 的 `benchmarking_accuracy/snapkv_0.75/*/*/*/config.yaml`
4. **壓縮比多半是名目值，不是量到的位元組。** token budget、設定的 ratio、理論公式（ShadowKV 7.08×，p16）各說各話；metadata 即使有提到（Quest p5 的讀取量公式含每 page 的 min/max；InfiniGen p13 說 partial key 佔全部 KV 的 15%），也沒有併入報告的壓縮比或容量〔複核修正：原寫「常不計入」，兩篇其實都有提，只是沒併入〕；kvpress 的 head-wise press 甚至只是遮蔽、不省記憶體（attention_patch.py L53–55）。〔原文〕〔程式碼〕〔複核補充〕例外是 Agrawal 的 Table 4：用 repo 的 `results.json` 可重現為「各桶 FP16 的 `cache_size_mb` 平均 ÷ 方法的 `cache_size_mb` 平均」，是實測位元組；但量測時點不同（CaM 在生成後量、其他在 prefill 後量），且有 OOM 跳過的樣本（見 Agrawal 卡）。〔計算〕
5. **對手常被改寫或推算。** H2O／TOVA 被改成 FlashAttention prefill＋decode 才逐出（DuoAttention p17 兩者都改；Quest p6 只交代改了 H2O〔複核補充〕）；ShadowKV 的「同 batch」與「無限 batch」對照是外推與理論頻寬推算（p8 註 5）；Quest 對各對手的效率比較是「定性」估計（p8–9）。〔原文〕
6. **幾乎沒有重複與誤差棒。** 只有 Rethinking 說吞吐取 3 次平均（p16）、Yuan 的 needle 每格 3 個隨機 key（p15）、Agrawal 的 repo 品質評測有 3 個 run 目錄、系統評測有 2 個（論文沒寫；〔複核修正〕原寫「3 個 run 目錄」只對品質評測成立）。其餘「未說明」。〔原文〕〔程式碼〕

---

## 一、稀疏注意力與卸載檢索

### MInference：MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention（NeurIPS'24；arXiv 2407.02490）

- **讀了什麼**：〔全文〕arXiv v2，https://arxiv.org/pdf/2407.02490v2 ，查證 2026-10-06
- **一句話**：用三種注意力稀疏型態加速長 prompt 的 prefill，不動 KV 大小。
- **評測要證明的主張**：在 100K–1M 的 prefill，以動態稀疏注意力在單張 A100 上最多加速 10×，同時維持（甚至略升）長 context 任務的準確度（p1、p9）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | LLaMA-3-8B-Instruct-262k、LLaMA-3-8B-Instruct-1048k（Gradient）、GLM-4-9B-1M、Yi-9B-200K；NIAH 另測 Phi-3-Mini-128K、Qwen2-7B-128K；〔複核修正〕LLaMA-3-70B 在**正文** Table 6（p9），不是附錄；正文稱「LLaMA-3-70B-1M」，但註腳連結與表頭都是 Gradient 的 Llama-3-70B-Instruct-**262k**。原文**沒有**標注意力類型〔未查證〕。延遲實驗用 bfloat16；KV dtype 未另說明 | §4 p6、p9、Table 6 p9、App C.2 p19〔原文〕 |
| 硬體 | 延遲：單張 A100（附錄寫明 80G）。另說用 tensor parallel＋context parallel 在 8×A100 可把 1M prefill 降到 22 秒（無細節） | p6、p9、p19〔原文〕 |
| 軟體與版本 | 自寫 PyTorch 實作，建在 FlashAttention、Triton 與 PIT 動態稀疏編譯器上；改寫 HF LLaMA 以避開 >50K 的 OOM：attention 依 head 切、MLP 依序列切、移除 attention mask、只算最後一個 token 的 LM head。版本號未給 | p6、App C.2–C.4 p19–20〔原文〕 |
| 資料／負載 | InfiniteBench 10 任務、3,992 例、平均約 214K token；RULER 13 任務、4K–128K 每長度 2,600 例；NIAH 到 1M（〔複核補充〕共 750 例，p18）；PG-19 隨機 1,000 個長於 100K 的樣本、prompt 到 100K | p6–7、App C.1 p18〔原文〕 |
| 長度 | 4K–1M token；延遲曲線 10K–1M | Fig 1b、Fig 10 p22〔原文〕 |
| 到達與併發 | 無到達過程；單一 prompt 的 prefill 延遲。batch 大小未說明 | p9〔原文〕；batch〔未查證〕 |
| 重用結構 | 無 | 〔原文〕 |
| 掃描的自變數 | context 長度、模型、稀疏型態（消融） | p7–9、p22〔原文〕 |
| 對手 | StreamingLLM（1k global＋4k local）、StreamingLLM w/ dilated、w/ strided、InfLLM（128 global＋8k local）、Ours w/ static。**所有對手只在 prefill 稀疏、decode 用 dense** | §4 Baselines p7〔原文〕 |
| 系統指標 | prefill 延遲與相對 FlashAttention 的加速；延遲拆解（建 index 佔 5%–20%）；單一 kernel micro-benchmark | p9、App D p21–22〔原文〕 |
| 品質指標 | 各 benchmark 官方腳本；RULER 以 85% 為「effective」門檻；PG-19 perplexity | p6、p8〔原文〕 |
| 主要結果 | ① 100K／300K／500K／1M 分別加速 1.8×／4.1×／6.8×／10×，1M 由 30 分鐘降到 3 分鐘（p9）。② InfiniteBench，LLaMA-3-8B-262K 平均 38.2 → 38.8；StreamingLLM 23.8、InfLLM 34.8（Table 2 p7）。③ RULER，LLaMA-3-8B-262K 平均 84.4 → 87.0，effective 長度 16K → 32K（Table 3 p7）。④ 10K 時建 index 的佔比由 5% 升到 30%，端到端接近 FlashAttention（Limitations p18） | 〔原文〕 |
| 消融／敏感度／開銷 | static vs dynamic index；只用單一型態；只用 vertical／只用 slash（Table 4 p9、Table 8 p22）；與 SnapKV 結合（Table 5 p9）；70B（Table 6 p9）；各型態在各層的分布（Fig 11 p23）；kernel 實際稀疏度（Fig 12 p23）；index 記憶體 1M 時 <160MB（p22） | 〔原文〕 |
| 重複與統計 | 全部 greedy decoding（為了穩定）；延遲重複次數、誤差棒：未說明 | p6、p19〔原文〕 |
| 程式碼／資料 | 公開（aka.ms/MInference → microsoft/MInference） | p1〔原文〕；repo 存在〔文件〕 |
| 設計理由（原文） | 長 prompt 時 attention 佔 prefill 延遲 90% 以上（p2）；稀疏位置隨輸入變動，換 prompt 重用 top-k 索引 recall 會掉（Fig 2c p3），所以要線上建 index；搜尋用 kernel 的實際 FLOPs 而非概念估算（p5）；型態搜尋只用 1 個 30K token 的 KV retrieval 樣本，約 15 分鐘（p19） | 〔原文〕 |
| 設計理由〔判讀〕 | 對手全是「只稀疏 prefill」的方法，所以它比的是 prefill 近似品質；decode 不變、KV 不縮小，因此它**不是** KV 容量方法。只用一個樣本校準，對任務分布的泛化是靠下游 benchmark 間接證明 | 〔判讀〕 |
| 原文沒講清楚的地方 | 延遲的 batch 與重複次數；InfiniteBench 樣本超過模型 window 時怎麼截斷；8×A100 的平行切法 | 〔判讀〕 |
| 與既有整理不一致 | `workloads_eval.md`、`PAPERS_BY_LEVEL.md` 都沒有 MInference 的評測列，無可比對。補充：SCBench（同一團隊）指出 MInference 用 prompt 最後一段（通常是問題）估稀疏型態，所以要在「拿不到問題」的 multi-request 模式下另測（SCBench p8） | 〔原文〕 |
| 對本研究的意義〔判讀〕 | ① **DROP＋重算的成本會被稀疏 prefill 改寫**：≥100K 時重算可能便宜 1.8–10×，16–32K 幾乎沒好處。量 κ（重算／傳輸成本比）時必須寫明「dense prefill」，並把稀疏 prefill 當敏感度軸。② 它是有損的（RULER 分數會變），若當作重算路徑，品質約束要重新量。③ KV 容量不變，不能當 L2 容量節省的對手 | 〔判讀〕 |

### DuoAttention：DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads（ICLR'25〔未查證〕；arXiv 2410.10819）

- **讀了什麼**：〔全文〕arXiv v1，https://arxiv.org/pdf/2410.10819v1 ，查證 2026-10-06；會議版未比對
- **一句話**：只給少數 retrieval head 完整 KV，其他 head 只留 sink＋recent。
- **評測要證明的主張**：在幾乎不掉準確度下，MHA／GQA 的長 context 記憶體最多省 2.55×／1.67×、decode 最多快 2.18×／1.50×、prefill 最多快 1.73×／1.63×；配合 8-bit 權重＋4-bit KV，Llama-3-8B 可在單張 A100-80G 放 3.3M token（p1、p9–10）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | Llama-2-7B-chat、Llama-2-7B-32K-Instruct（MHA，每層 32 head）；Llama-3-8B-Instruct、Llama-3-8B-Instruct-Gradient-1048k、Llama-3-70B-Instruct、Mistral-7B-Instruct-v0.2（GQA，每層 8 KV head）。權重與 activation 預設 BF16 | Fig 4 註 p4、p6–7、p9〔原文〕 |
| 硬體 | 找 retrieval head 的訓練：8×A100；效率量測：單張 A100；3.3M 實驗：A100-80G；FastGen 對手：8×A100-80G | p3、p5、p9、p10、p18〔原文〕 |
| 軟體與版本 | PyTorch；FlashInfer 的 RoPE／RMSNorm kernel；FlashAttention-2 做 prefill；訓練用 FSDP2＋DeepSpeed Ulysses；量化 kernel 用 QServe。版本號未給 | p6–7、p10、App A.1 p17〔原文〕 |
| 資料／負載 | 找 head：BookSum 中隨機插入十組 32-word passkey，長度取 1,000 到模型上限之間 50 個區間，2,000 steps。評測：NIAH、LongBench（Fig 7 為 14 任務，Table 3／4 為全部 21 任務）、MMLU（1-shot）、MBPP 與 MT-Bench（0-shot） | p5、p7–8〔原文〕 |
| 長度 | NIAH：Llama-2-7B-32K 到 32K、Llama-3-8B-1048K 到 1048K；prefill 效率：100K（MHA）、320K（GQA）；找 head 的訓練序列最長只有 32,000（Llama-3-8B-1048K 亦同） | Fig 6 p6、Fig 10 p9、Table 2 p17〔原文〕 |
| 到達與併發 | 無；效率實驗的 batch 未說明，只說設計「適合大 batch」 | p6、p9〔原文〕；batch〔未查證〕 |
| 重用結構 | 無 | 〔原文〕 |
| 掃描的自變數 | retrieval head 比例（＝KV budget）、context 長度、prefill chunk size、sink／recent 大小 | Fig 7、9–11、13〔原文〕 |
| 對手 | H2O、TOVA、StreamingLLM、FastGen（社群實作，以 recovery ratio 0.7／0.87 調到平均 >25%／>50% budget；只跑得到 24K／32K）。H2O／TOVA 被改成 FlashAttention prefill＋decode 時逐出，並把**輸入最後 50 token 模擬成 decode** | p7、App A.3 p17、A.5 p18〔原文〕 |
| 系統指標 | 每 token decode 延遲與記憶體（**預先配置整段 KV** 以排除動態配置開銷）；prefill 延遲與 peak memory | p9〔原文〕 |
| 品質指標 | NIAH 正確率；LongBench 官方指標；MMLU／MBPP／MT-Bench | p7–8〔原文〕 |
| 主要結果 | ① LongBench 平均：Llama-3-8B-1048K Full 40.08 vs Duo（50%）40.21，H2O 35.76、StreamingLLM 32.26、TOVA 35.55（Table 3 p18）；Llama-2-7B-32K Full 37.52 vs Duo（25%）34.49，H2O 26.84（Table 4 p19）。② 記憶體 2.55×（MHA）／1.67×（GQA）、decode 延遲 2.18×／1.50×（Fig 11 p9）。③ 3.30M token、相對 BF16 full attention 容量 6.4×（Fig 12 p10） | 〔原文〕 |
| 消融／敏感度／開銷 | 找 head 的方法（最佳化 vs 注意力 profiling vs language modeling loss）、sink＋recent 的必要性、部署時 sink／recent 大小（16／64 後持平）；以 Mistral-7B、30K-word passkey、100 個深度做 | Fig 13 p10〔原文〕 |
| 重複與統計 | 未說明；Fig 9 的 OOM 點是**由量測外插**的 | Fig 9 註 p8〔原文〕 |
| 程式碼／資料 | 公開 | p1〔原文〕；repo 存在〔文件〕 |
| 設計理由（原文） | 自然語言中需要長距離推論的監督訊號太稀疏，所以用合成 passkey 找 head（p4–5）；以「限制成 sink＋recent 後輸出偏多少」定義 retrieval head，而不是看注意力分數（p4）；H2O／TOVA 依賴注意力分數、與 FlashAttention 不相容，所以改寫；若答案只有一個 token，改寫版就等於 full attention，所以模擬最後 50 token（p17） | 〔原文〕 |
| 設計理由〔判讀〕 | 「最後 50 token 當 decode」讓逐出法有機會犯錯，但只有 50 步；長生成、多輪都沒測。KV 預配置＋單請求，避開了 paged 管理與 batch 間的干擾 | 〔判讀〕 |
| 原文沒講清楚的地方 | 效率量測的 batch、KV dtype；LongBench 的截斷 | 〔判讀〕 |
| 與既有整理不一致 | `workloads_eval.md` 沒有 DuoAttention 評測列（只在 KVServe 那列當對手），無衝突。補充：kvpress 的 `DuoAttentionPress` 用**遮蔽**實作（duo_attention_press.py L112–115＋attention_patch.py L53–55），不會省記憶體；用 kvpress 重現 DuoAttention 的記憶體收益會得到錯的結論 | 〔程式碼〕kvpress `7136c22` |
| 對本研究的意義〔判讀〕 | ① **同一個 token 的 KV，不同 head 的價值不同**：streaming head 的中段 KV 根本不會被讀，屬於零成本 DROP。以 token／block 為單位的放置決策會錯過這一層。② 不要學它的 OOM 外插與 KV 預配置報法。③ head 只用到 32K 的資料找，卻部署到 1M，長度外推要自己驗證 | 〔判讀〕 |

### Quest：Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference（ICML'24；arXiv 2406.10774）

- **讀了什麼**：〔全文〕arXiv v2，https://arxiv.org/pdf/2406.10774v2 ，查證 2026-10-06
- **一句話**：每個 KV page 記 key 的逐維 min／max，用當下 query 估注意力上界，只讀前 K 個 page。
- **評測要證明的主張**：self-attention 最多快 7.03×、端到端 decode 最多快 2.23×，在長依賴任務上準確度幾乎不掉（p1–2、p8）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | 品質：LongChat-v1.5-7b-32k、Yarn-Llama-2-7b-128k；效率：Llama2-7B 組態。原文以 32 head 計算 KV 大小（p3），`workloads_eval` 記為 MHA〔二手〕。FP16 | p1 註、p3、p5、p7〔原文〕 |
| 硬體 | kernel：RTX 4090、CUDA 12.2；端到端：**Ada 6000**（為了跑較長的 context） | p7〔原文〕 |
| 軟體與版本 | 以 FlashInfer 為基礎的 CUDA kernel；Top-K 用 RAFT；page size 16（kernel 實驗） | p7〔原文〕 |
| 資料／負載 | PG19 test（100 本、平均 70K token），LongChat 測到 32K；passkey：LongChat 10K、Yarn 100K；LongBench 6 集（NarrativeQA、HotpotQA、Qasper、TriviaQA、GovReport、MultifieldQA），**只用 LongChat** | p5–6〔原文〕 |
| 長度 | 最長 100K（passkey） | Table 1 p5〔原文〕 |
| 到達與併發 | 無；端到端是 single-batch | p8〔原文〕 |
| 重用結構 | 無 | 〔原文〕 |
| 掃描的自變數 | token budget（32–4,096）、序列長度、page size | Table 1、Fig 7–10〔原文〕 |
| 對手 | H2O、TOVA、StreamingLLM。H2O 因需完整注意力矩陣，context 階段改用 FlashAttention，從 decode 才開始累計分數。效率對手：FlashInfer dense attention | p5–6〔原文〕 |
| 系統指標 | kernel 延遲（NVBench）、self-attention 時間拆解（PyTorch profiler）、decode 每 token 平均延遲（不含 sampling） | p7–8〔原文〕 |
| 品質指標 | PG19 perplexity、passkey 正確率、LongBench 官方指標 | p5–6〔原文〕 |
| 主要結果 | ① passkey 10K：budget 64 時 Quest 99%，H2O／TOVA／StreamingLLM ≤1%（Table 1 p5）。② 32K、budget 2048：self-attention 7.03×；端到端 1.74×（FP16 權重）、2.23×（4-bit 權重）（p8）。③ 加上前兩層的全量 cache，LongBench 6 集達到 lossless 的稀疏度為 Qasper 1/6、HotpotQA 1/6、GovReport 1/5、TriviaQA 1/10、NarrativeQA 1/5、MultifieldQA 1/6（p6；〔複核修正〕原文的資料集順序與上方「資料」格不同，補上對應名稱以免錯配）。④ 同準確度下對 TOVA：GovReport 3.82×、TriviaQA 4.54×（p9） | 〔原文〕 |
| 消融／敏感度／開銷 | kernel micro-benchmark（估計、Top-K、近似注意力，Fig 8）；Top-K 只要 5–10 µs（p5 註、p7） | 〔原文〕 |
| 重複與統計 | 未說明 | 〔原文〕 |
| 程式碼／資料 | 公開 | p1〔原文〕 |
| 設計理由（原文） | decode 佔推論時間大宗（16K prompt＋512 response 時 >86%，p3）；為了測長依賴，把問題與指令「一個 token 一個 token」當 decode 餵入（p6）；前兩層稀疏度低，不套用 Quest 與對手（p4–5）；對手沒有 kernel，所以只用 FlashInfer 在對應 budget 下的延遲做**定性**比較，忽略對手的額外開銷（p8–9） | 〔原文〕 |
| 設計理由〔判讀〕 | 這個協定讓逐出法「在看到問題前就要丟」，而 Quest 保留全部 KV、看到問題才挑——比的是讀取稀疏 vs 容量逐出，兩者記憶體占用不同，**同 token budget 比準確度並不對等** | 〔判讀〕 |
| 原文沒講清楚的地方 | 端到端實驗的輸出長度；LongBench 的截斷；page metadata 的實際記憶體（只說可用量化或更大的 page 降低，p7） | 〔判讀〕 |
| 與既有整理不一致 | `workloads_eval.md` 寫硬體「RTX 4090（kernel）」，**端到端其實在 Ada 6000**（p7），應補；LongBench 只用 LongChat 一個模型（p6）。`PAPERS_BY_LEVEL.md`「只省讀取量，不省容量」與原文一致（p2：保留全部 KV），可補一句「另需每 page 兩個 key 向量的 metadata，讀取量約為 1/S＋K·S/L」（p5） | 〔原文〕 |
| 對本研究的意義〔判讀〕 | ① 它是「讀取時選擇」，不是放置；但 page 的 min／max 摘要很便宜，可當作「之後可能被讀」的寫入時訊號。② 拿它當對手時要同時報容量（它不省）與 metadata 位元組。③ 卸載到 CPU 後 Quest 在 1M、batch 3 只有 9.34 tok/s（ShadowKV Table 14 p18〔二手〕），說明「讀取稀疏」一旦跨 PCIe，瓶頸就換了 | 〔判讀〕 |

### ShadowKV：ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference（ICML'25〔文件〕；arXiv 2410.21465）

- **讀了什麼**：〔全文〕arXiv v3，https://arxiv.org/pdf/2410.21465v3 ，查證 2026-10-06
- **一句話**：pre-RoPE key 做低秩留在 GPU，value 卸到 CPU；decode 時用 chunk landmark 挑約 1.56% 的 KV 重建或抓回。
- **評測要證明的主張**：GPU 上的 KV 記憶體省 6× 以上而不掉準確度；batch 可以大 6×、吞吐最高 3.04×，甚至超過「GPU 記憶體無限」時的 full attention（p1、p6、p8）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | 主實驗：Llama-3-8B-1M、GLM-4-9B-1M、Llama-3.1-8B、Yi-9B-200K；NIAH 另測 Phi-3-Mini-128K、Qwen2-7B-128K；附錄 Llama-3-70B-1M。KV head：Llama 8、GLM／Yi 4。權重與 KV 皆 BF16，另做 FP8（e5m2）敏感度 | p7、Table 3 p8、App A p16–17、p20〔原文〕 |
| 硬體 | A100；理論推算用 PCIe 31.5 GB/s、HBM 2 TB/s。GPU 記憶體容量、CPU 型號、DRAM 大小**未說明** | p1、p6、p8〔原文〕；其餘〔未查證〕 |
| 軟體與版本 | PyTorch＋自寫 CUDA kernel；FlashAttention；FlashInfer 與 vLLM 的 fused kernel；multi-stream 讓 K 重建與 V 抓取重疊。rank 160、chunk 8、outlier 48、sparse budget 1.56% | p7、App B.1 p20〔原文〕 |
| 資料／負載 | RULER 128K；LongBench **只取 >4K 的樣本**、budget 256；NIAH 16K–1M；InfiniteBench（Table 16）；multi-turn NIAH；與 MInference 合用時 RULER 8K–256K | p7–8、p20〔原文〕 |
| 長度 | 主實驗 128K；吞吐 60K／122K／244K／488K；附錄 1M（Llama-3-8B）、512K（70B） | Table 3–4 p8、Table 11 p17〔原文〕 |
| 到達與併發 | 無到達過程；固定 batch，最大 48–50 | Table 3–4 p8〔原文〕 |
| 重用結構 | 只有同一 context 的多輪 NIAH（session 內），沒有跨請求重用 | Fig 7 p8〔原文〕 |
| 掃描的自變數 | batch × context、sparse budget、chunk size、rank、outlier 數 | Table 4、Fig 8–9、Table 15〔原文〕 |
| 對手 | Quest、Loki、InfiniGen，各有「整個 KV 卸載」與「只卸 V（標 V）」兩版；全部 exact prefill，decode 時「挑選 KV」的計算量設為 full attention 的 1/16。吞吐對手：能整個放進 GPU 的最大 batch 的 full attention；另列「同 batch」與「Inf」——**同 batch 是量單一 Transformer block 後外推到整個模型，Inf 是用 A100 理論頻寬推算** | p7、Table 3 註 5 p8〔原文〕 |
| 系統指標 | decode generation throughput（tokens/s）；單一 block 的延遲拆解；GPU 記憶體節省（理論式，舉例 7.08×） | p8、App A.2 p16、A.6 p18〔原文〕 |
| 品質指標 | RULER、LongBench、NIAH 官方指標 | p7〔原文〕 |
| 主要結果 | ① RULER 128K，Llama-3-8B-1M 平均：Full 86.68、ShadowKV 86.88、Quest 82.03、InfiniGen 70.13（Table 1 p7）。② Llama-3.1-8B、122K：80.78 tok/s（batch 4）→ 245.90（batch 24），3.04×；「Inf」欄 134.30（Table 3 p8）。③ 1M、batch 3：Full Attention（CPU）0.21、Quest（CPU）9.34、ShadowKV 45.32 tok/s（Table 14 p18）。④ prefill 額外開銷（SVD 等）佔比 64K 6.65% → 512K 0.97%（Table 12 p18） | 〔原文〕 |
| 消融／敏感度／開銷 | budget、chunk size、rank、outlier、FP8、ShadowKV+（生成 token 也存低秩）、與 MInference 合用、與 InfiniGen 的詳細比較 | §5.3、App A〔原文〕 |
| 重複與統計 | 未說明 | 〔原文〕 |
| 程式碼／資料 | 公開 | p1〔原文〕 |
| 設計理由（原文） | pre-RoPE key 最低秩、value 不是，所以只卸 V（p2、p4）；chunk 內 post-RoPE key 相似度高，均值可當 landmark（p5）；相鄰 decode 步選到的 chunk 約 60% 重疊，可加 cache（p4–5、p9）；LongBench 輸入比 RULER 短，所以只測 >4K 的樣本、budget 改設 256（p7；〔複核修正〕原文用同一句話交代這兩個設定，「since it has shorter inputs」可同時是兩者的理由，原卡只寫成「所以只取 >4K」不完整）；SVD 也可在 CPU 非同步做，或預先算好存成 prefix cache 的一部分（p2 註 1） | 〔原文〕 |
| 設計理由〔判讀〕 | 「超過無限記憶體的 full attention」這個主張的分母是**理論推算**，不是量測。吞吐只算 decode，prefill（含 SVD）沒有併入 | 〔判讀〕 |
| 原文沒講清楚的地方 | GPU 記憶體容量、CPU／DRAM、輸出長度、吞吐的量測時間窗；multi-turn NIAH 的前幾輪是否放入模型自己的回答 | 〔判讀〕 |
| 與既有整理不一致 | `workloads_eval.md` 大致相符，應補「吞吐表的同 batch 與 Inf 兩欄是推算值（註 5）」與附錄 1M／512K、InfiniteBench。`PAPERS_BY_LEVEL.md` 說「單一請求」不精確：它量的是固定 batch（同長度序列一起 decode），只是沒有到達過程、沒有跨請求重用；同條目的「Yandex 的發現」不在 ShadowKV 原文 | 〔原文〕；Yandex 說法〔未查證〕 |
| 對本研究的意義〔判讀〕 | ① 它把「GPU／CPU 兩層＋讀取時選擇」做到 1M，是 GPU／CPU 分層最強的對手之一；但它的 CPU 層只放 V、K 以低秩表示，屬於**改變表示法**，和本專案「CPU、SSD 兩層是無損位元組搬移」的定義不同，比較時要分開報品質。② 寫入時（prefill）的一次性處理成本隨長度攤銷（Table 12），支持「寫入時狀態管理」的可行性。③ 不能沿用它推算出來的 baseline | 〔判讀〕 |

### InfiniGen：InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management（OSDI'24；arXiv 2406.19707）

- **讀了什麼**：〔全文〕arXiv v1，https://arxiv.org/pdf/2406.19707v1 ，查證 2026-10-06；與 USENIX 正式版未比對
- **一句話**：在 layer i−1 用部分 query 權重與部分 key cache 預演 layer i 的注意力，只從 CPU 預取重要的 KV。
- **評測要證明的主張**：在卸載式推論系統上比既有 KV 管理方法快最多 3.00×、準確度最多高 32.6 個百分點（p2）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | OPT-6.7B／13B／30B、Llama-2-7B／13B；長序列 perplexity 用 Llama-2-7B-32K；Llama-3-8B-1048K 只做注意力分析。注意力類型原文沒直接寫（p2 以 H 個 head 的 MHA 敘述），`workloads_eval` 記為 MHA〔二手〕。FlexGen 基線以 FP16 載入 KV | p9、p11、p13–14〔原文〕 |
| 硬體 | RTX A6000 48GB；Xeon Gold 6136；96GB DDR4-2666；PCIe 3.0 ×16 | p9〔原文〕 |
| 軟體與版本 | 實作在 FlexGen 上；離線用 SVD 把 Wq、Wk 乘上正交矩陣（skew，計算結果不變）。partial weight ratio 0.3；alpha 4（OPT）／5（Llama-2），平均用不到 10% KV；每層最多送 20% KV 上 GPU | p2、p8–9〔原文〕 |
| 資料／負載 | lm-evaluation-harness 5-shot：COPA、OpenBookQA、WinoGrande、PIQA、RTE；WikiText-2、PTB perplexity；速度用 PG-19 隨機句子 | p9〔原文〕 |
| 長度 | 延遲：1,920 輸入＋128 輸出；序列掃描 512–2,048；perplexity 到 32K（Llama-2-7B-32K） | p11–13〔原文〕 |
| 到達與併發 | 無；固定 batch 4–20 | p11–12〔原文〕 |
| 重用結構 | 無 | 〔原文〕 |
| 掃描的自變數 | batch、序列長度、模型大小、alpha、partial ratio、相對 KV 大小、CPU 端 KV pool 上限（80%） | Fig 11–19、Table 2〔原文〕 |
| 對手 | 執行環境：CUDA UVM、FlexGen（KV 全放 CPU）；方法：H2O（20% budget）、INT4 group-wise 非對稱量化，兩者都建在 FlexGen 上 | p9、p11〔原文〕 |
| 系統指標 | 含 prefill＋decode 的 wall-clock 推論延遲；tokens/s；單一 block 延遲拆解 | p9、p11–13〔原文〕 |
| 品質指標 | accuracy（lm-eval）、perplexity | p9〔原文〕 |
| 主要結果 | ① OPT-13B、1,920＋128、batch 20：對各基線快 1.63×–32.93×（Fig 14 p11）。② batch 4→20：InfiniGen 27.36→41.99 tok/s，INT4 12.22→14.02、H2O 21.31→25.70（p12）。③ 只比全在 GPU 的 Ideal 慢 1.52×；FlexGen 有 96.9% 時間在傳輸（Fig 18 p13）。④ partial query 權重與 partial key cache 分別佔全部參數 2.5%、全部 KV 15%（p13） | 〔原文〕 |
| 消融／敏感度／開銷 | skewing 有無（Fig 13）；KV pool 80% 上限下 FIFO／LRU／Counter（Table 2，只看 perplexity）；alpha 與 partial ratio（Fig 17） | p11–13〔原文〕 |
| 重複與統計 | 未說明 | 〔原文〕 |
| 程式碼／資料 | 公開（snu-comparch/InfiniGen） | 〔文件〕 |
| 設計理由（原文） | 卸載系統中 KV 傳輸是新瓶頸（p1）；重要 token 數不隨長度線性成長（OPT-13B 在 2,048 時平均約 73 個，H2O 的 20% 卻要載 409 個，p12）；選 Counter 而非 LRU，避免 atomic 更新（p9、p11） | 〔原文〕 |
| 設計理由〔判讀〕 | 「長 context」的證據主要是 perplexity 與注意力分析，需要長距離檢索的任務沒測。ShadowKV 在 RULER 128K 上量到 InfiniGen 70.13 vs Full 86.68（ShadowKV Table 1 p7〔二手〕） | 〔判讀〕 |
| 原文沒講清楚的地方 | 重複次數；30B 有 30% 權重在 CPU（p12）時延遲如何歸屬；KV pool 上限實驗只看 perplexity | 〔判讀〕 |
| 與既有整理不一致 | `workloads_eval.md` InfiniGen 列相符；「敏感度實驗 1,920」宜改寫成「延遲實驗 1,920 輸入＋128 輸出」。`PAPERS_BY_LEVEL.md`「要修改模型權重」需加註：是離線的**等價變換**（輸出不變，p8），不是重新訓練；但要多存 partial 權重與 partial key（p13）。「準確度只在短任務上量」與原文一致 | 〔原文〕 |
| 對本研究的意義〔判讀〕 | ① CPU 端 KV pool＋使用者設定上限＋victim policy，是「CPU 層容量有限時逐出誰」的早期版本，但只用 perplexity 評。② PCIe 3.0、A6000、≤2K 的結論不能外推到 16K–512K | 〔判讀〕 |

---

## 二、批判既有評測與統一 benchmark

### Yuan'24：KV Cache Compression, But What Must We Give in Return? A Comprehensive Benchmark of Long Context Capable Approaches（EMNLP'24 Findings；arXiv 2407.01527）

- **讀了什麼**：〔全文〕arXiv v2（含附錄），https://arxiv.org/pdf/2407.01527v2 ，查證 2026-10-06
- **一句話**：在對齊的壓縮比下，比較五類長 context 方法的品質。
- **評測要證明的主張**：在「合理對齊」的環境下評 10+ 種方法、65 種設定、7 類任務，整理出先前未知的現象（p2、p7–9）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | Llama-3-8B-Instruct（原生 8K；needle 測試時 RoPE θ 放大 32×）、Mistral-7B-Instruct-v0.2、LongChat-7B-v1.5-32K；Mamba-2.8B、Mamba2-2.7B、Mamba-Chat-2.8B、RWKV-5-World-7B、RecurrentGemma-2B／9B-it。dtype 未說明 | p5–6、p15〔原文〕；dtype〔未查證〕 |
| 硬體 | 一張或多張 A100-80G（DGX A100） | p4〔原文〕 |
| 軟體與版本 | 各方法官方或作者背書的實作（KIVI、FlexGen 的 group-wise 量化、H2O、InfLLM repo 內的 StreamingLLM、LLMLingua-2）；HF Transformers 版本會讓記憶體差最多 2×，所以承諾的實測記憶體延後發表 | App B.3 p16、p17–18〔原文〕 |
| 資料／負載 | LongBench 英文 15 任務（刪掉 PassageCount）＋自建 passkey needle（Paul Graham essays 當填充、7-digit key）；LongBench 多數任務平均 5K–15K、每任務 200 筆（MultiFieldQA 150、LCC 500、〔複核補充〕RepoBench-P 500） | p4–5、App A p14–15〔原文〕 |
| 長度 | LongBench 依官方設定**中段截斷**：Llama-3 **7,500**、Mistral／LongChat 31,500 token；needle 512–20,480 **words**（約 27.2K／30.6K token），10 長度 × 10 深度 × 3 個隨機 key。〔複核補充〕截斷只套用在 baseline 與 prefill 後才壓的方法（KIVI、FlexGen、H2O）；**prefill 時就壓的 InfLLM、StreamingLLM 收到未截斷的完整輸入**，只是 KV 預算上限設成 max_length ÷ 壓縮比（例：Mistral 2× 為 31,500/2＝15,750） | Table 3 p15、App B.1 p15〔原文〕 |
| 到達與併發 | 無 | 〔原文〕 |
| 重用結構 | 無 | 〔原文〕 |
| 掃描的自變數 | 壓縮比 2×／4×／6×／8×；量化 2／4 bit | Table 2 p5〔原文〕 |
| 對手 | 自身即比較：KIVI-2／4bit、FlexGen-4bit、InfLLM、StreamingLLM、H2O、LLMLingua-2、線性時間與 hybrid 模型。StreamingLLM 與 H2O 改成「依輸入長度比例」的預算以對齊壓縮比 | p6、App B.3 p16〔原文〕 |
| 系統指標 | **無**。原文明說效率難以做 apple-to-apple 比較 | p9〔原文〕 |
| 品質指標 | LongBench 官方指標（F1、ROUGE-L、accuracy）；needle exact match | p14〔原文〕 |
| 主要結果 | ① Llama-3-8B 的 LongBench 平均：baseline 45.2、KIVI-2bit 44.3（5.05×）、H2O-8x 43.4、StreamingLLM-8x 30.3、LLMLingua2-8x 26.9（Table 2 p5）。② 7-digit passkey 下 H2O 4× 的 needle 為 100%，改成 64-digit 後掉到 35.0%；KIVI-2bit 100→91.0%（p9）。③ 不在 prefill 壓縮的方法（KIVI、FlexGen、H2O）普遍優於在 prefill 之中或之前壓縮的方法（OB❶ p7–8） | 〔原文〕 |
| 消融／敏感度／開銷 | 壓縮比掃描；64-digit needle（Fig 21–23）；修正 InfLLM 對 condensing RoPE 的處理後 LongChat 結果明顯變好（p17） | 〔原文〕 |
| 重複與統計 | needle 每格 3 個隨機 key；LongBench 未說明 | p15〔原文〕 |
| 程式碼／資料 | 公開（p1）；「含完整 log」是 `workloads_eval` 的說法，本次未進 repo 確認 | p1〔原文〕；log〔二手〕workloads_eval |
| 設計理由（原文） | 不同流派無法全域對齊，只能以壓縮比對齊（p6–7）；token dropping 改成比例式，才能和量化比較（p6）；PassageCount 所有模型都 <10%，平均會被拖壞，所以刪掉（p14）；needle 以 words 計，跨 tokenizer 才能給相同輸入（p15）；用雷達圖是沿用 LongBench，但承認會誇大差異（p18） | 〔原文〕 |
| 設計理由〔判讀〕 | 只對齊品質面、系統面完全不量；它自己也說理論效率相同不代表實際效率相同 | 〔判讀〕 |
| 原文沒講清楚的地方 | dtype、LongBench 的重複次數、各方法是否用 FlashAttention | 〔判讀〕 |
| 與既有整理不一致 | `workloads_eval.md`「15 個 LongBench 資料集＋passkey」相符；**應補「Llama-3-8B 的 LongBench 輸入被截到 7,500 token」**。〔複核修正〕原寫「Llama-3 那半的結論其實是 ≤7.5K 的短 context」過頭：baseline、KIVI、FlexGen、H2O 確實只看到 ≤7,500 token，但 InfLLM、StreamingLLM 看到完整輸入、KV 預算 ≤7,500÷壓縮比（App B.1 p15），兩組的輸入條件不同 | Table 3、App B.1 p15〔原文〕 |
| 對本研究的意義〔判讀〕 | 見下方陷阱 Y1–Y8 | 〔判讀〕 |

**它指出的評測陷阱**（每條附原文位置；「意義」為〔判讀〕）

- **Y1 壓縮比很難對齊。** 固定預算的 token dropping，壓縮比會隨輸入長度變；KIVI 有全精度的 residual，壓縮比必須指定參考長度（它用 10K）（Table 2 註 p5、p6）。→ 意義：分層 PoC 的「各層位元組」要用實測值，並附上長度。
- **Y2 在 prefill 內壓縮，與在 prefill 後壓縮，差很多；而且所有任務都是長輸入、短輸出。** 結論不一定適用長生成（OB❶ p7–8；Limitations p10）。→ 意義：寫入時決策屬於「prefill 後」，但仍要另測長輸出。
- **Y3 第一個 token 是免費的。** decode 時才逐出的方法，第一個輸出 token 在未壓縮的狀態下產生；7-digit passkey 的答案前幾位因此「白送」，讓 H2O 看起來 100%（p9）。→ 意義：答案要夠長（64-digit）或多 token，否則高估。
- **Y4 與 FlashAttention 不相容。** 依賴完整注意力矩陣的方法很難整合（p9；Table 1 註 p2：H2O 沒有 FA 相容 kernel，不適合直接線上使用）。
- **Y5 理論效率 ≠ 實際效率。** 取決於好不好優化、與既有框架是否相容（p9）。
- **Y6 記憶體量測隨軟體版本變。** HF Transformers 4.42 前後最多 2× 差異（p17–18）。→ 意義：記憶體要用引擎計數，並固定版本。
- **Y7 預訓練配方不同，架構比較不公平**（p6–7、p17）。
- **Y8 單位。** needle 長度用 words，換成 token 依 tokenizer 而異（20,480 words ≈ 30.6K 或 27.2K token，p15）。

### Rethinking'25：Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving（MLSys'25；arXiv 2503.24000）

- **讀了什麼**：〔全文〕arXiv v1（含附錄），https://arxiv.org/pdf/2503.24000v1 ，查證 2026-10-06
- **一句話**：從生產部署角度重量 KV 壓縮：吞吐、輸出長度分布、逐樣本失敗。
- **評測要證明的主張**：既有研究漏掉三個維度——吞吐、長度分布、negative samples——它們會左右能否上線；並提供吞吐預測、長度預測、負樣本 benchmark 三個工具（p1–2）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | 吞吐：LLaMA-2-7B／13B／70B（正文多寫 LLaMA-7B）、Mistral-7B；長度與負樣本：LLaMA-3.1-8B-Instruct、Mistral-7B-v0.1 | p6、App A.2 p15〔原文〕 |
| 硬體 | 4×A6000（NVLink）、Xeon Gold 6326；LLaMA-70B 在 H800 | App A.5 p16、Fig 2 p7〔原文〕 |
| 軟體與版本 | Torch 2.1.2、Transformers 4.43.1（文中稱 TRL）；FlashAttention 2.5.6（TRL+FA）；各方法實作在 LMDeploy「v6.0.1」（原文寫法）；吞吐預測沿用 Vidur | App A.4 p15–16、App E p17〔原文〕 |
| 資料／負載 | 吞吐用合成輸入；長度與 E2E 用 ShareGPT 1,000 筆（以 vLLM benchmark 程式取樣，最多生成 1,024 token，過長則截斷）；verbosity 200 筆；負樣本用 LongBench（官方設定） | p6、p8–9、App A.1 p15〔原文〕 |
| 長度 | prompt／KV 長度到 8K | Fig 1、Fig 3〔原文〕 |
| 到達與併發 | Fig 5 的 E2E 以 batch 1 逐筆量；request router 實驗：〔複核補充〕LLaMA-7B、4×A6000（Baseline 是 4 張都跑同一設定；三種路由策略是 1 張跑 FP16、3 張跑壓縮法）、LMDeploy、1,000 筆 ShareGPT、**Poisson 10 req/s** | p9、§5.4 p11〔原文〕 |
| 重用結構 | 無 | 〔原文〕 |
| 掃描的自變數 | batch 1–16、prompt／KV 長度、TP 1／2／4、壓縮設定（bits、cache size）、temperature | Fig 1、Table 3、Fig 4〔原文〕 |
| 對手 | KIVI（G=32、R=128）、GEAR（s=2%、r=2%）、StreamingLLM（64＋448＝512）、H2O（64＋448＝512）；附錄另測 SnapKV | App A.3 p15、Fig 9〔原文〕 |
| 系統指標 | prefill／decode 吞吐（tokens/s）、相對 FP16 的加速、E2E 延遲 CDF、router 的平均 E2E | §4.2–4.3、§5.4〔原文〕 |
| 品質指標 | LongBench 指標；semantic score（與 ChatGPT 回答的相似度）；negative sample＝原本是 benign（分數 ≥ 平均）且壓縮後相對損失超過門檻 θ | p9–10、p2 註 1、p9 註 2（〔複核修正〕「benign＝分數 ≥ 平均」在 p9 註 2，不在 p2）〔原文〕 |
| 主要結果 | ① Table 3：decode 相對 FP16，KIVI-4 為 0.98×／0.88×／0.9×（TP 1／2／4）、GEAR-4 1.02×／0.97×／0.97×、H2O 1.34×／0.69×／0.85×；prefill 的 H2O 只有 0.51–0.58×（p8）。② >20% 樣本輸出長度 ≥1.5×（Table 5 p9）。③ 平均分數幾乎不變（LLaMA-3.1-8B：41.2 → KIVI 41.3、GEAR 40.9），但 10% 門檻下仍有大量 negative samples（Fig 6 p10；平均分數見 App D p17）。④ router 同時用兩個預測器，平均 E2E 快 1.45–1.80×（Table 8 p11） | 〔原文〕 |
| 消融／敏感度／開銷 | 換模型（Mistral、13B、70B）、TP、SnapKV；長度分布隨壓縮比變平（Fig 4） | App B–C〔原文〕 |
| 重複與統計 | 吞吐排除初始化、取 3 次平均；長度實驗以 temperature 1.0 抽樣，每筆是否多次抽樣未說明 | p16、p8〔原文〕；多次抽樣〔未查證〕 |
| 程式碼／資料 | 公開 | p1〔原文〕 |
| 設計理由（原文） | 選 LMDeploy：量化 kernel 比 vLLM 有效率、開發較快；KIVI 作者 2024 年 4 月就說整合 vLLM 有困難；除 Observation 2 外，結論不依賴特定引擎（p15–16）。用 T＝0.9、1.1 當對照，區分「溫度造成的長度變化」與「壓縮造成的」（p8） | 〔原文〕 |
| 設計理由〔判讀〕 | 它自己的對照組顯示，光是抽樣就有 27.5%（T=0.9）／31.4%（T=1.1）的樣本變長 ≥1.5×，比四種壓縮法的 21.3%–27.1% 還多；壓縮法的差別在於「變短 ≥50% 的樣本較少」（6.8%–16.5% vs 20.8%–21.3%）（Table 5 p9）〔計算〕。所以「壓縮讓輸出變長」的證據強度有限，較穩的說法是「長度分布偏移」，且應以 greedy 重驗 | 〔判讀〕 |
| 原文沒講清楚的地方 | Table 3 用哪個模型、batch、長度；Fig 5 的輸出長度上限；Fig 6 的絕對數值需讀圖 | 〔判讀〕 |
| 與既有整理不一致 | ① `PAPERS_BY_LEVEL.md` 標〔摘要＋擷取〕，「4-bit KV 的 decode 吞吐只有 FP16 的 0.88–1.02 倍」與 Table 3 相符（合併 KIVI-4 與 GEAR-4、TP 1／2／4），但要補條件：這是 TP 比較表，模型／batch／長度未說明；「輸出偏長」見上方判讀；「很多樣本從答對變答錯」應改成「相對損失超過 θ（10%）的 benign 樣本」。② `workloads_eval.md` §4 寫「負載／到達：未涵蓋」**有誤**：§5.4 有 Poisson 10 req/s 的 router 實驗（p11）；「CPU/SSD 分層：摘要未提」→ 全文也沒有 | 〔原文〕 |
| 對本研究的意義〔判讀〕 | 見下方陷阱 R1–R10 | 〔判讀〕 |

**它指出的評測陷阱**

- **R1 在 HF transformers（TRL）上量吞吐不可靠**，要在有 PagedAttention／FlashAttention 的服務框架量（Missing Piece 1 p5；Observation 1 p7；Table 1 的 Frw 欄 p4 顯示多數論文只用 T）。→ 意義：Tiara 的時間數字必須來自 vLLM 本體。
- **R2 只在單卡量，沒看 TP**；TP 會稀釋壓縮帶來的頻寬收益，甚至變負（Missing Piece 1 p5；Table 3 p8）。→ 意義：CLAUDE.md 已規定容量懸崖不用 TP，但要在論文寫明「結論只適用單卡」。
- **R3 固定輸出長度量吞吐是錯的**；有損壓縮會改變輸出長度分布，E2E 要連長度一起算（Missing Piece 2 p5；Observation 3–4 p9）。→ 意義：品質與時間要在同一批請求上量，報 E2E，不只 TTFT。
- **R4 只報平均準確度會掩蓋個別樣本的失敗**（Missing Piece 3 p6；Observation 5 p10；Table 7 p11）。→ 意義：報 paired、逐樣本的退化分布，不只平均。
- **R5 任務類型敏感度不同**：摘要與 QA 最脆弱（Observation 6 p10；Fig 7）。
- **R6 稀疏法與 FlashAttention／PagedAttention 不相容**：重要度依賴注意力分數，但 FlashAttention 不存分數，要多兩趟讀取；PagedAttention 假設 KV 單調增長，定期逐出讓長度忽大忽小（§3.1.2 p5）。
- **R7 量化法的全精度 recent window 讓 paging 要管兩種 tensor**，計算變得不規則（§3.1.1 p4）。
- **R8 加速比依 batch、長度、TP 而定，有負加速的區域**（Observation 2 p8）；量化法在 KV 8,192 甚至 OOM（p8）。→ 意義：報整條曲線，標出「哪裡變慢」。
- **R9 越細的粒度（token、channel）準確度越好，但計算越不規則，GPU 效率越差**（§3.1.1–3.1.2 p3–4）。
- **R10 既有 benchmark 研究多只報準確度**（Table 2 p6：包括 Yuan'24 的 LongCTX-Bench 只有 Acc），只有 LLM-QBench 量吞吐（p5）。

### Agrawal & Mayer'26：Benchmarking KV-Cache Optimizations across Task Quality and System Performance for Long-Context Serving（arXiv 2607.05399）

- **讀了什麼**：〔全文〕arXiv v1，https://arxiv.org/pdf/2607.05399v1 （2026-05-03 提交）；〔程式碼〕artifact repo commit `925fadf`（`kvpress/benchmarking_snapkv_kvpress.py`、`kvpress/benchmarking_cam_kvpress.py`、vendored `kvpress/kvpress/presses/cam_press.py`、`benchmarking_accuracy/` 目錄結構），查證 2026-10-06。〔複核補充〕V08 另行 clone 同一 commit（`925fadf`，2026-05-09），讀了 `KIVI/benchmarking_kivi.py`、`turboquant_445/benchmarking_turboquant.py`、`benchmarking_system_performance/*/*/{0,1}/results.json`、`benchmarking_accuracy/*/*/*/*/{config.yaml,metrics.json,predictions.csv}`，2026-10-07
- **一句話**：在同樣的模型與資料上比 KIVI、TurboQuant、SnapKV、CaM 的品質與單請求效能。
- **評測要證明的主張**：壓縮比本身無法預測端到端表現；KIVI4 品質最穩、SnapKV 吞吐最好、CaM 隨負載差異很大（p1–2）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | Llama-3.1-8B-Instruct、Mistral-7B-Instruct-v0.3（兩者皆 GQA；原文說 Mistral 用 sliding window、32K window）；FP16＋FlashAttention2 | §3 p6〔原文〕 |
| 硬體 | 單張 A100 **40GB** | p6、致謝 p12〔原文〕 |
| 軟體與版本 | HF transformers 系；KIVI 官方實作；TurboQuant 為社群實作（相容 Transformers 4.45）；SnapKV、CaM 用 kvpress 的 SnapKVPress、CAMPress。量測迴圈：DynamicCache、逐 token argmax。〔複核補充〕系統評測的 `results.json` 記錄 torch 2.4.1+cu121、A100-SXM4-40GB；FP16（All KV）基線是 KIVI 腳本設 `k_bits=v_bits=16`。品質評測是**兩套 harness**：All KV／KIVI 用 KIVI 的 LongBench 腳本（`benchmarking_accuracy/{FP16,kivi*}/*/*.jsonl`），SnapKV／CaM 用 kvpress `evaluate.py`（`config.yaml`），兩者的 prompt 組法不同 | p6〔原文〕；`benchmarking_snapkv_kvpress.py` L273–390〔程式碼〕；artifact `925fadf` 的 `benchmarking_system_performance/*/*/0/results.json`、`benchmarking_accuracy/`〔程式碼〕 |
| 資料／負載 | 品質：LongBench 6 集（HotpotQA、2WikiMQA、Qasper、MultiFieldQA_en、TriviaQA、MultiNews）；系統：3 集 NarrativeQA、GovReport、Qasper（〔複核修正〕原寫「另外三集」：Qasper 兩邊都有，只有 NarrativeQA、GovReport 是品質集沒有的）。論文說用 LongBench 的 prompt 模板與各集生成上限（p6）；〔複核補充〕但 CaM 的品質 run 實際把 `max_new_tokens` 一律設 64（含 MultiNews，LongBench 官方上限 512），SnapKV 則用資料集預設 | §3 p5–6〔原文〕；`benchmarking_accuracy/CaM/*/*/*/config.yaml`〔程式碼〕 |
| 長度 | Table 1 標 1K–64K；系統分 0–4K、4–8K、8K+ 三桶（以 context 的 token 數、門檻 4,000／8,000 分）；實際最長到 81,496 token（Mistral NarrativeQA 8K+）。〔複核補充〕各桶樣本數（`results.json` run 0）：Llama——NarrativeQA 0／8／192、GovReport 20／73／107、Qasper 79／107／14；Mistral——NarrativeQA 0／0／200、GovReport 15／52／133、Qasper 62／117／21。系統評測 600 筆中 8K+ 佔 313（Llama）／354（Mistral），**過半在 8K 以上**，但 8K+ 一桶涵蓋 8K–81K | Table 1 p5、Table 4 p9〔原文〕；分桶 `get_context_length_buckets()` L410–411〔程式碼〕；樣本數〔計算〕 |
| 到達與併發 | **無**；每筆 batch 1、依序跑 | 〔程式碼〕`benchmark_one()` |
| 重用結構 | 無 | 〔原文〕 |
| 掃描的自變數 | 方法 × 資料集 × 長度桶 | Table 2–4〔原文〕 |
| 對手 | All KV（KIVI 的 FP16 版）；KIVI2／4（group 32、residual 32）；TurboQuant3（32 個 outlier channel 存 4 bit，等效 3.25 bit）／4；SnapKV（ratio 0.75、window 32、kernel 7）；CaM（interval 32、target 1,024、hidden buffer 64、merge budget 32、內部 SnapKV window 16） | p6〔原文〕 |
| 系統指標 | TTFT（原文：從送出到第一個 output token；**程式碼裡是 prefill 時間＋第一個 decode step**）；output throughput（decode 步數 ÷ decode 時間，單請求）；prefill KV memory（prefill 前後 `memory_allocated` 的差）；realized compression ratio（計算方式原文未說明；§4.2 小標題寫「Compression rate on KV memory」）。〔複核補充〕用 repo 的 `results.json` 可重現 Table 4：每桶「FP16 的 `cache_size_mb` 平均 ÷ 方法的 `cache_size_mb` 平均」（例：run 0 算得 Llama CaM GovReport 3.10／6.04／13.82、Qasper 1.26／1.17／1.23，表為 3.09／6.04／13.82、1.26／1.17／1.22；KIVI2、KIVI4、SnapKV 也差 ≤0.01），即**實測位元組之比**；但 `cache_size_mb` 的量測時點不同：KIVI／FP16（`benchmarking_kivi.py` L339）、SnapKV（L305）、TurboQuant（L314）在 prefill 後量，CaM 在生成結束後量（`benchmarking_cam_kvpress.py` L156） | p7、p9〔原文〕；L289–377〔程式碼〕；重現〔計算〕 |
| 品質指標 | LongBench 指標：QA 用 F1、TriviaQA 用 EM、MultiNews 用 ROUGE | p6–7〔原文〕 |
| 主要結果 | ① Llama HotpotQA：All KV 55.72、SnapKV 59.2、CaM 59.8；MultiNews：27.15／23.04／17.56（Table 2 p7）。② TTFT 各法與 All KV 差距小，TurboQuant 最大（Table 3 p8）。③ 壓縮率：KIVI2 5.16–5.32、KIVI4 3.15–3.19、SnapKV 4.00、CaM 1.00–14.20（Table 4 p9）。〔複核補充〕repo 對照：Llama SnapKV MultiNews 三個 run 的 `metrics.json` 都是 **23.4**，論文表為 23.04（疑為轉錄誤植，未能確認）；CaM MultiNews 17.64／17.47／17.57，平均 17.56 與表一致；All KV 三個 run 完全相同（greedy） | 〔原文〕；repo〔程式碼〕〔計算〕 |
| 消融／敏感度／開銷 | 無 | 〔原文〕 |
| 重複與統計 | 原文只說 TurboQuant 的摘要任務「只跑一次」（Table 2 註）；repo 的 `benchmarking_accuracy/` 每個方法×模型有 `0/1/2` 三個 run 目錄〔複核修正：原寫「TurboQuant Mistral 只有 `0/1`」不對——TurboQuant 兩個模型都有 `0/1/2`，只是 `multi_news` 只出現在 run 1，與「摘要只跑一次」一致〕；`benchmarking_system_performance/` 每個方法×模型只有 `0/1` 兩個 run；誤差棒未報 | p7〔原文〕；目錄〔程式碼〕`925fadf` |
| 程式碼／資料 | 公開 | p1〔原文〕 |
| 設計理由（原文） | 只選不用改架構、不用重訓的 drop-in 方法，方便資料系統的管理者整合（§2.3 p4）；A100 40GB 只能跑 7–8B（p6）；系統評測挑 context 較長的三集（p5）；可預測的壓縮率比極端壓縮率重要（Lessons p10） | 〔原文〕 |
| 設計理由〔判讀〕——CaM 的壓縮率很可能是量測產物 | CAMPress 是 decoding press：prefill 不壓，decode 每 32 步才壓一次，壓到 1,024 token（原文 p6；vendored `cam_press.py` L240、L281）；cache 大小在生成結束後才量（`benchmarking_cam_kvpress.py` L156）。NarrativeQA／Qasper 是短答案（LongBench 生成上限 128，實際多半在 EOS 前就停）→ 很少觸發 → 1.00–1.26；GovReport 生成上限 512 → 一定觸發。〔複核修正〕原估算式「（輸入＋輸出）÷1,024」的分子不對：FP16 基線的 cache 是在 prefill 後量的（`benchmarking_kivi.py` L339），不含輸出；正確的關係是「FP16 prefill 時的 cache ÷ CaM 生成後的 cache（約 1,040 token＝target 1,024＋未滿 32 步的新 token）」。〔複核補充〕repo `results.json`（run 0）直接證實：CaM 在 GovReport 每一桶的平均 `compressed_cache_tokens` 都是 1,038–1,042（兩個模型皆然），Llama 三桶平均輸入 3,144／6,219／14,312 token，÷1,040 ≈ 3.02／5.98／13.76，表為 3.09／6.04／13.82；NarrativeQA 的平均 cache 長度約等於輸入長度（沒壓）〔計算〕。也就是說 Table 4 的 CaM 數字幾乎完全由「輸入長度 ÷ target_size」與「生成是否超過 32 步」決定。原文把差異歸因於「GovReport 冗餘多、NarrativeQA 相關內容多」（p10），這個解釋沒有證據支持 | 〔判讀〕〔計算〕；artifact `925fadf` |
| 〔複核補充〕TurboQuant 在 Mistral NarrativeQA 的「離群值」是 OOM 跳過造成的 | 量測腳本遇到 CUDA OOM 會跳過該筆、只計數（`benchmarking_snapkv_kvpress.py` 的 `skipped_oom`；TurboQuant 腳本同樣記 `num_skipped_oom`）。Mistral NarrativeQA 的 TurboQuant3／4 在兩個 run 各有 21–23 筆（共 200 筆）OOM 被跳過，正好是輸入最長的那些（69,232–81,496 token），而 All KV 200 筆全跑完。用 run 0 重算：TurboQuant4「All KV 全 200 筆的平均 cache ÷ TQ 存活 178 筆的平均 cache」＝ **4.43**（＝Table 4 的值），只用同一批 178 筆配對則是 **3.76**（與 Llama 及其他資料集的常數 3.76 相同）；TTFT 方面，TQ4 平均 4,173 ms（＝Table 3 的 4,172.87）看起來比 All KV 的 4,932 快，但同一批 178 筆的 All KV 只有 3,827 ms，TQ4 其實慢約 9%。原文把 TurboQuant 在 NarrativeQA 壓縮率偏高解釋成「旋轉量化在較長、較多樣的 context 上分布更均勻」（p10），這是倖存者偏差，不是方法特性 | 〔計算〕〔判讀〕；`benchmarking_system_performance/turboquant{3,4}/mistral-7b-instruct-v0.3/{0,1}/results.json`（`num_skipped_oom`＝23／21、22／22） |
| 〔複核補充〕QA 任務上「壓縮勝過 All KV」大概不是壓縮造成的 | ① CaM 只在 decode 每 32 步壓一次，而它在 HotpotQA 的答案平均 2.9 個英文字、最長 30 字，TriviaQA 最長 14 字（`predictions.csv`），所以這兩集幾乎從未觸發壓縮，CaM 仍比 All KV 高 4.1 分（59.8 vs 55.72）。② SnapKV 的品質 run 設 `query_aware: true`（問題接在 context 後一起壓），論文沒寫。③ All KV 與 SnapKV／CaM 用不同的評測 harness（見「軟體與版本」）。三者合起來，Table 2 中 SnapKV／CaM 在 QA 上的領先，可能主要來自 prompt 組法與 query-aware 設定，而非壓縮本身；Lessons「壓縮不必犧牲準確度」（p10）因此證據不足。④ CaM 的 MultiNews 只給 64 個新 token（官方 512），預測平均 48.9 字、SnapKV 383 字，所以 CaM 的摘要分數低（17.56）至少部分是長度上限造成的，不只是原文說的「合併的近似誤差」（p8） | 〔程式碼〕〔計算〕〔判讀〕；`benchmarking_accuracy/{CaM,snapkv_0.75}/llama-3.1-8b-instruct/0/*/{config.yaml,predictions.csv}` |
| 原文沒講清楚的地方 | ① 壓縮率用 bytes 還是 tokens 算（〔複核補充〕repo 可重現為 bytes 之比，見「系統指標」）。② 各桶樣本數（〔複核修正〕原寫「NarrativeQA 4–8K 桶 min＝max＝7,964，可能只有 1 筆」：`results.json` 顯示是 **8 筆**、長度都是 7,964，推測是同一份文件的 8 個問題〔計算〕；完整樣本數見「長度」格）。③ 是否用 `--max_prompt_tokens` 截斷：程式預設不截（L483）；〔複核補充〕所有系統評測 run 的 `results.json` 都記錄 `max_prompt_tokens: null`，品質 run 的 kvpress `config.yaml` 也是 `max_context_length: null`，所以實際**沒有截斷**，Mistral（32K window）確實收到最長 81,496 token 的輸入〔程式碼〕。④ Fig 3 的「準確度」與「吞吐」是否來自不同資料集。⑤ Lessons 說 KV 開銷相對「decode 與 detokenization」可忽略（p11），與它自己 TTFT 的定義（只含 prefill＋第一步）不一致。⑥〔複核補充〕論文沒提 OOM 樣本被跳過、SnapKV 用 query-aware、CaM 生成上限 64（見上三列） | 〔判讀〕；〔程式碼〕`925fadf` |
| 與既有整理不一致（**使用者文件**） | intro 表 16 註：「已經有人在 L1（壓縮）做跨方法的基準測試 [56]」；sota §4.6：「有人做了 L1（壓縮）的跨方法基準測試，同時看任務品質和系統效能 [31]」。① **「L1」太窄**：使用者自己把 L1 定義成「存多細：BF16／FP8／INT4」（intro 表 5），但 [56] 的四個方法中 SnapKV（逐出）與 CaM（合併）不是精度，屬於使用者另一類「單一請求內丟 token」（sota L19）。建議改成「在單請求內的 KV 壓縮（量化、逐出、合併）」。〔複核修正〕這一點**不是**對 [56] 的描述錯誤：使用者原句寫的是「L1（**壓縮**）」，而且 intro 表 6（txt L329）也把 KVP（學習式挑選保留 token）歸在 L1，可見使用者實際把 L1 當「單請求內的壓縮」用。問題在使用者自己的表 5 定義比用法窄，屬術語一致性，建議修表 5 的 L1 定義或此句措辭（擇一即可）。② **「系統效能」要加限定**：單張 A100-40GB、HF transformers、batch 1、無到達過程、只分三個長度桶（0–4K／4–8K／8K+，8K+ 一桶涵蓋 8K–81K），沒有 serving engine；標題的 serving 指的是部署考量，不是負載量測〔判讀〕。〔複核修正〕原寫「長度主體 ≤8K」不對：系統評測 600 筆中過半在 8K+（Llama 313、Mistral 354，`results.json`〔計算〕）；品質集的範圍則是 1K–32K（Table 1 p5）。③ 只有 4 個方法、6 個設定，「跨方法」成立但規模小。④ 引用寫「arXiv:2607.05399, May 2026」正確（提交日 2026-05-03）；PDF 的 PVLDB 樣板卷期是佔位字，不能寫成已發表。⑤ 「但 L2 的分層還沒有」與原文一致：沒有 CPU／SSD 層、沒有跨請求重用 | 〔原文〕 |
| 對本研究的意義〔判讀〕 | 見下方陷阱 A1–A6 | 〔判讀〕 |

**它指出（或它本身示範）的評測陷阱**

- **A1 只看壓縮比預測不了端到端**：壓縮比相近的方法，吞吐與品質差很多（貢獻 3 p2；§4.2 p9–10）。〔原文〕
- **A2 壓縮率不穩定的方法在通用服務難以營運**：CaM 1.00–14.2（p9–10、Lessons p10）。〔原文〕——但見上方判讀：不穩定可能來自「輸出長度 vs 壓縮間隔」，不是任務冗餘（〔複核補充〕repo 的 `compressed_cache_tokens` 已證實 GovReport 每筆都被壓到約 1,040 token，見上方判讀列）。→ 意義：任何 decode-time 或「依觸發條件」的策略，實際容量要在**生成結束時與生成中**都量，並報輸出長度。
- **A3 品質與系統在大致不同的資料集上量**（p5；〔複核修正〕只有 Qasper 兩邊都有）。〔原文〕→ 意義：Tiara 的品質與時間要來自同一批請求。
- **A4 單請求 throughput 不是 serving throughput**：output throughput 定義為單請求 decode 速度（p7；程式碼 L375–377）。〔判讀〕→ 意義：要在併發與到達過程下量。
- **A5 長度分桶粗、上界開放**：8K+ 一桶含 8K–81K，平均值混在一起（Table 3–4）；〔複核補充〕而且 8K+ 正是樣本最多的一桶（過半），論文也沒報每桶樣本數。〔判讀〕→ 意義：16K–512K 要細分且報每桶樣本數。
- **A6 TTFT 定義與程式不一致**（原文 p7 vs 程式 L342）。〔判讀〕→ 意義：TTFT 要明定起訖點（引擎 timestamp）。
- **A7〔複核補充〕OOM 樣本被靜默跳過，桶平均的樣本集合因方法而異**：Mistral NarrativeQA 的 TurboQuant 有 21–23／200 筆（最長的 69K–81K）OOM 被跳過，造成 Table 4 的 4.43／5.02 與 Table 3 的「TTFT 比 All KV 低」都是倖存者偏差（同批配對後壓縮率回到 3.76、TTFT 反而慢約 9%）。〔程式碼〕〔計算〕→ 意義：OOM 本身就是結果，要報「失敗率」並只用配對樣本比較平均；Tiara 的 M1 容量懸崖量測也一樣，不能把 OOM 的點從平均裡默默拿掉。
- **A8〔複核補充〕不同方法用不同 harness、不同生成上限、不同 query 可見性**：All KV／KIVI 走 KIVI 的 LongBench 腳本；SnapKV 走 kvpress 且 `query_aware: true`；CaM 走 kvpress、`query_aware: false` 但 `max_new_tokens` 一律 64。論文 p6 寫「用 LongBench 的模板與各集生成上限」，與 CaM 的設定不符。〔程式碼〕→ 意義：所有動作（含 Full KV 基線）必須跑在同一條 pipeline、同一份 prompt 與生成上限，差異只能來自被比較的那個變因。

### kvpress：NVIDIA kvpress（repo）與 KVPress Leaderboard（HF Space）

- **讀了什麼**：〔文件〕〔程式碼〕repo commit `7136c22`（2026-10-05）：`README.md`、`evaluation/README.md`、`evaluation/evaluate.py`、`evaluate_config.yaml`、`evaluate.sh`、`leaderboard.sh`、`evaluate_registry.py`、`kvpress/pipeline.py`、`attention_patch.py`、`presses/{adakv,duo_attention,cam,scorer,snapkv,decoding,base}_press.py`、`evaluation/benchmarks/ruler/{README.md,create_huggingface_dataset.py,calculate_metrics.py}`、`notebooks/speed_and_memory.ipynb`；leaderboard Space commit `e81fc40`（2026-10-05）的檔案清單與 `src/textual_content.py`；RULER 資料集大小（HF datasets-server）。查證 2026-10-06。kvpress 的論文（arXiv 2510.00636）**未讀**。
- **一句話**：在 HF transformers 上用 forward hook 實作數十種 KV 壓縮（press），附統一評測 CLI 與 leaderboard。
- **要證明的主張**：簡化新方法的開發，提供標準化 benchmark 做公平比較（README L13；leaderboard `ABOUT_TEXT`）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | 預設 Llama-3.1-8B-Instruct（`evaluate_config.yaml` L6）；`leaderboard.sh` 用 Qwen3-8B（L7）。Leaderboard 現有結果目錄：Qwen3-8B 131 個、Llama-3.1-8B-Instruct 121＋14 個（兩種命名）。FAQ 列出測過的架構：Llama、Mistral、Phi3、Qwen2、Qwen3、Gemma3 | 〔程式碼〕〔計算〕Space 檔案清單；README L211 |
| 硬體 | 不規定；`evaluate.sh` 每個 press 綁一張 GPU（L7–24）；`leaderboard.sh` 註明 4 GPU（L4）；速度 notebook 在 A100-80GB（cell 0） | 〔程式碼〕 |
| 軟體與版本 | HF transformers 的 `DynamicCache`；可用時自動開 FlashAttention-2（`evaluate.py` L380–388），ObservedAttentionPress 強制 eager；量化走 transformers 的 `QuantizedCache`（README L169–181） | 〔程式碼〕〔文件〕 |
| 壓縮時機 | 預設只在 prefill 壓 **context**（README L39、L55）；`DecodingPress` 為實驗功能、只支援 ScorerPress（L60–95）；CAMPress 屬 decoding press（L153） | 〔文件〕 |
| 資料／負載 | Loogle、RULER、ZeroSCROLLS、InfiniteBench、LongBench、LongBench-E、LongBench-v2、NIAH、AIME25、MATH500（`evaluate_registry.py` L54–66）。kvpress 的 RULER 資料集只有 4096／8192／16384 三種長度，各 6,500 列 | 〔程式碼〕；列數〔計算〕datasets-server |
| 長度 | Leaderboard **全部是 `ruler__4096`**（266 個結果目錄）；RULER 的 4096 是用 Llama-3.1-8B tokenizer 算的，與原版 RULER 不可直接比 | 〔計算〕Space 清單；ruler README〔文件〕 |
| 到達與併發 | 無；依 context 分組，**壓一次、答多題**；每題答完把答案從 cache 移除 | `evaluate.py` L434–461；`pipeline.py` L237–269〔程式碼〕 |
| 重用結構 | 同一 context 的多個問題（壓縮後共用），沒有跨 session | 同上 |
| 掃描的自變數 | 壓縮比；Leaderboard 現有 0.10–0.97，加上 DMS／KVzap 的閾值 −3 到 −9；`leaderboard.sh` 本身用 0.25／0.5／0.75／0.875 | 〔計算〕；`leaderboard.sh` L15–21〔程式碼〕 |
| query-aware 開關 | `query_aware: true` 會把問題接到 context 後一起壓（`evaluate_config.yaml` L19；`evaluate.py` L353–356；`evaluation/README.md` L35）。Leaderboard 把 SnapKV、AdaKV-SnapKV、Finch、ChunkKV 另外跑 query-aware 版（`leaderboard.sh` L41–49） | 〔程式碼〕〔文件〕 |
| 截斷 | 超過 `max_context_length` 時**保留開頭**（`pipeline.py` L165–169），與 LongBench 官方的中段截斷不同 | 〔程式碼〕 |
| 系統指標 | Leaderboard **沒有**；速度與記憶體只在 notebook：KnormPress、單序列、8K–128K、生成 100 token、關閉 EOS、greedy（`speed_and_memory.ipynb` cells 6–10）；README 說記憶體「應減少約 compression_ratio × KV 大小」（L233） | 〔程式碼〕〔文件〕 |
| 品質指標 | RULER 用 string match（QA 用 part、其餘用 all）（`ruler/calculate_metrics.py`） | 〔程式碼〕 |
| 記錄的壓縮比 | 結果檔寫入的是 press 的**設定值**（`evaluate.py` L457–460），不是量到的位元組 | 〔程式碼〕 |
| 重複與統計 | seed 固定 42（L76）＋greedy（`pipeline.py` L306、L318），等於每點跑 1 次；沒有誤差棒 | 〔程式碼〕 |
| 設計理由（原文） | 用 `model.generate` 無法把問題排除在壓縮外，那樣會不公平地偏袒 SnapKV 這類方法；理想的壓縮應與後面接什麼無關（README L277）。RULER 統一用 Llama-3.1-8B tokenizer，是為了比較同模型不同壓縮比，作者認為可接受（ruler README） | 〔文件〕 |
| 設計理由〔判讀〕 | 這是目前「query-agnostic 品質評測」最乾淨的公共工具，但完全不碰 serving：沒有 paged KV、沒有 batch、沒有時間 | 〔判讀〕 |
| 原文沒講清楚的地方 | Leaderboard 每個點的樣本數是否為全部 6,500；不同 press 的實際記憶體 | 〔判讀〕 |
| 與既有整理不一致 | `workloads_eval.md` kvpress 列：「壓縮比 0.25–0.94」**已過時**，現為 0.10–0.97＋閾值；「每個模型」實為 Qwen3-8B 與 Llama-3.1-8B-Instruct 兩個；其餘相符。應補：head-wise press 只遮蔽、不省記憶體；截斷保留開頭；RULER 只到 16K | 〔計算〕〔程式碼〕 |
| 對本研究的意義〔判讀〕 | 見下方陷阱 K1–K6 | 〔判讀〕 |

**它指出（或它的程式碼揭露）的評測陷阱**

- **K1 把問題一起壓會偏袒 query-aware 方法**（README L277；`evaluation/README.md` L35）。〔文件〕→ 意義：寫入時的放置決策看不到之後的問題，評測必須 query-agnostic。
- **K2 head-wise 壓縮在參考實作裡只是遮蔽**：AdaKV、DuoAttention 用「假 key 讓注意力權重為 0」，不降低 peak memory、略增時間（`attention_patch.py` L53–55；`adakv_press.py` L76–80；`duo_attention_press.py` L112–115）。〔程式碼〕→ 意義：「壓縮比」與「省下的記憶體」是兩件事，一定要量。
- **K3 部分 press 會暫時加大記憶體**：KVComposePress 在 prefill 時建出約 2× context 的 KV（README L135–136）；BlockPress 不是真正的 chunked prefill（L148）。〔文件〕
- **K4 截斷方式不同**：kvpress 保留開頭（`pipeline.py` L165–169），LongBench 官方保留頭尾。〔程式碼〕→ 意義：跨工具比分數前先統一截斷。
- **K5 tokenizer 統一造成長度不對齊原版**（ruler README）。〔文件〕
- **K6 Leaderboard 只有 4K 的 RULER、只有準確度**。〔計算〕→ 意義：它的排名對 16K–512K、對時間都沒有資訊。

---

## 三、核對使用者文件（[56]／[31] 與 SCBench）

| 使用者文件的描述 | 位置 | 原文怎麼說 | 判定 |
|:--|:--|:--|:--|
| 「已經有人在 L1（壓縮）做跨方法的基準測試 [56]，但 L2 的分層還沒有」 | `intro.txt` L836（表 16 註）；md L554 | [56] 比 KIVI、TurboQuant（量化）＋SnapKV（逐出）＋CaM（合併），單張 A100-40GB、HF transformers、batch 1、無到達過程（p4–9；程式碼）；〔複核修正〕長度不是「≤8K 為主」：Table 1 為 1K–64K，系統評測實測到 81K、600 筆中過半在 8K+（repo `results.json`〔計算〕） | 〔複核修正〕原判「⚠️ 過度簡化」改為：**這句沒有事實錯誤**。指控「L1 太窄」❌ 不成立——原句寫「L1（壓縮）」，且 intro 表 6（txt L329）也把 KVP（逐出類）歸 L1，屬使用者表 5 定義過窄的術語問題，不是對 [56] 的誤述；指控「系統效能只是單請求」❌ 不適用——此句根本沒提系統效能。後半句「L2 分層還沒有」✅（就 [56] 而言：無 CPU／SSD 層、無跨請求重用；「全領域都沒有」不在本卡查證範圍） |
| 「有人做了 L1（壓縮）的跨方法基準測試，同時看任務品質和系統效能 [31]」 | `sota.txt` L395；md L244 | 同上；品質與系統大致用不同資料集（p5；只有 Qasper 重疊） | 「L1 太窄」❌ 不成立（理由同上）。「系統效能要加限定」✅ 指控成立——原句字面正確（[56] 確實量 TTFT、output throughput、prefill KV memory，p7），但這些都是 HF 迴圈中的單請求數字，沒有 serving engine、沒有併發；建議加「單請求、HF transformers、無 serving engine」。〔複核修正〕原建議中的「≤8K 為主」要刪（與 repo 數據不符）。〔複核補充〕若要更進一步引用 [56] 的結論（例如「壓縮不必犧牲準確度」），要注意其品質評測的 harness 不一致與 OOM 跳過問題（見 Agrawal 卡 A7、A8） |
| 參考文獻「arXiv:2607.05399, May 2026」 | `intro.txt` L1116–1117；`sota.txt` L520–521 | arXiv 提交日 2026-05-03（PDF 側邊戳記「3 May 2026」） | ✅（複核確認） |
| 「SCBench 量共享 context 下的品質，但重用率是構造出來的 [32]」 | `sota.txt` L396；md L245 | SCBench 的每個 session 由一段共享 context 加多個後續查詢組成，12 任務、兩種共享模式（multi-turn、multi-request）、931 session／4,853 query、平均 5 輪（p1–5）；multi-turn 模式用**標準答案**而非模型輸出當前幾輪內容（p8）；沒有量吞吐或延遲（p8 只寫 greedy、BF16、4×A100；〔複核補充〕全文 31 頁搜尋 TTFT／latency／throughput 等詞，只出現在方法分類的描述（p5 提到 KV retrieval 可降低 TTFT）與參考文獻，沒有系統指標的量測結果） | ✅ 方向正確（複核確認：931 session／4,853 query／平均 5 輪 p5，multi-turn 用標準答案 p8，兩種模式 p2、p8）。「重用率」不是 SCBench 的用語，嚴格說是「session 內 100% 共用 context，由構造決定」〔判讀〕。可補兩點：只量品質；multi-turn 用標準答案接續 |
| 「SCBench [58]：多輪、共用 context 的品質評測」（表 17「負載合成方式」） | `intro.txt` L857；md L569 | 同上；另有 multi-request 模式（跨 session 重用同一 context）（p3、p8） | ✅ 但漏了 multi-request 模式；且它沒有到達時間，放在「負載合成」表中容易被讀成負載來源〔判讀〕 |

補充（給 E09 協調）：`workloads_eval.md` 記 SCBench HF 資料「922 列」，論文寫 931 session（p5）；兩者是否同一計數單位未查證。

---

## 四、評測陷阱總表

| # | 陷阱 | 哪篇指出／示範 | 原文位置 | 對使用者 PoC 的對策〔判讀〕 |
|:--|:--|:--|:--|:--|
| T1 | 把問題放進被壓縮的內容（query-aware），結果看起來太好 | kvpress；SCBench；Quest 的協定（反向示範）；ShadowKV 的多輪 NIAH；〔複核補充〕Agrawal（未揭露的實例） | kvpress README L277、eval README L35；SCBench p2、p8；Quest p6；ShadowKV Fig 7 p8（SnapKV 從第二輪起掉）；Agrawal artifact `benchmarking_accuracy/snapkv_0.75/*/*/*/config.yaml`（`query_aware: true`，論文未寫） | 放置決策在 prefill 結束時做，問題在之後才出現；同一 context 至少 2 個以上不同問題（multi-request） |
| T2 | decode 才逐出＋短答案，第一個 token 免費 | Yuan；DuoAttention；〔複核補充〕Agrawal（CaM，示範） | Yuan p9（7→64-digit：100%→35%）；DuoAttention App A.3 p17；Agrawal：CaM 每 32 個 decode 步才壓，HotpotQA 答案平均 2.9 字、TriviaQA 最長 14 字，幾乎從未觸發（`predictions.csv`〔計算〕） | 用多 token 答案（64-digit、長摘要）；DROP 動作要在答案生成前就生效 |
| T3 | 在 HF transformers 量吞吐，與真實引擎不符 | Rethinking | Missing Piece 1 p5；Obs 1 p7 | 時間只在 vLLM（FlashAttention＋paged KV）量；HF 只用來量品質 |
| T4 | 固定輸出長度量吞吐；有損法改變輸出長度 | Rethinking | Missing Piece 2 p5；Obs 3–4 p9 | greedy、不設固定長度（保留 EOS），報 E2E 與輸出長度分布 |
| T5 | 抽樣溫度本身就造成大幅長度變化 | Rethinking（自身對照組；〔複核修正〕這是本卡對其 Table 5 的重讀〔計算〕，原文的結論是「溫度造成的長短變化大致對稱，壓縮則偏向變長」p8，並沒有把它當成陷阱指出） | Table 5 p9（T=0.9／1.1 變長 ≥1.5× 的比例 27.5%／31.4%，四種壓縮法 21.3%–27.1%） | 用 greedy；若要抽樣，每筆多次並與「同設定重跑」比較 |
| T6 | 只報平均準確度 | Rethinking | Missing Piece 3 p6；Obs 5 p10 | 報逐樣本 paired 差、退化比例、最差分位數 |
| T7 | 與 FlashAttention／PagedAttention 不相容，實際變慢或跑不動 | Rethinking；Yuan；DuoAttention；Quest | Rethinking §3.1.2 p5；Yuan p2 註、p9；DuoAttention p17；Quest p6 | 只納入能在 vLLM 跑的動作；不相容的對手明寫「改寫版」 |
| T8 | 量化的全精度 window 讓壓縮比隨長度變、paging 變複雜 | Rethinking；Yuan；Agrawal | Rethinking §3.1.1 p4；Yuan Table 2 註 p5；Agrawal Table 4 p9 | INT4／FP8 層的位元組要含 scale、zero point、residual |
| T9 | 壓縮比是名目值；metadata 沒併入；遮蔽不省記憶體 | Quest；InfiniGen；ShadowKV；kvpress | Quest p5、p7；InfiniGen p13；ShadowKV App A.2 p16；kvpress attention_patch L53–55、evaluate.py L457–460。〔複核修正〕原列 Agrawal Table 4 為「名目值」的例子不對：它是實測位元組之比（repo 可重現），問題在量測時點與樣本集合（見 T10、T21） | 每層報實測位元組（含 metadata、索引）與引擎的 peak GPU memory |
| T10 | 觸發式壓縮的實際容量取決於輸出長度（與量測時點） | Agrawal（CaM，示範） | p6、Table 4 p9；`cam_press.py` L240、L281；〔複核補充〕`benchmarking_cam_kvpress.py` L156（生成後才量）vs FP16 在 prefill 後量（`benchmarking_kivi.py` L339）；GovReport 每筆都壓到約 1,040 token（`results.json`〔計算〕） | 容量在生成中與生成後都量；報輸出長度；所有方法在同一時點量 |
| T11 | 加速比依 batch／長度／TP 而定，有負區域 | Rethinking；MInference；Quest | Rethinking Obs 2 p8、Table 3；MInference p18（10K 時建 index 佔 30%）；Quest p7（短序列估計階段頻寬用不滿） | 掃 16K–512K 全段與多個併發度，畫出「哪裡變慢」 |
| T12 | 對手的數字是推算或定性估計 | ShadowKV；Quest | ShadowKV Table 3 註 5 p8；Quest §4.3.3 p8–9 | 所有對手都實測；推算值另欄標「推算」 |
| T13 | 對手被改寫（prefill 換 FlashAttention、預算改比例式） | DuoAttention；Quest；Yuan；ShadowKV | DuoAttention p7、p17–18；Quest p6；Yuan p6、p16；ShadowKV p7 | 改寫要列出並做敏感度；優先用官方實作 |
| T14 | 長度太短（「長 context」其實 ≤8K） | Yuan；InfiniGen；kvpress；Rethinking | Yuan Table 3 p15（Llama-3 7,500，僅 baseline 與 prefill 後才壓的方法）；InfiniGen p11（延遲 ≤2,048；perplexity 到 32K）；kvpress Leaderboard 4K；Rethinking Fig 1 ≤8K。〔複核修正〕刪去 Agrawal：其系統評測過半樣本在 8K+、最長 81K（repo `results.json`〔計算〕），它的問題是分桶太粗（見 A5），不是太短 | 主軸 16K–512K，並報每段樣本數 |
| T15 | 只測單請求或固定 batch，沒有到達過程 | 幾乎全部 | 見「共同模式 1」 | 用真實 trace 的時間與 session；Poisson 只當敏感度 |
| T16 | prefill 與 decode 分開評、只優化一邊 | MInference；Quest；ShadowKV；Agrawal | MInference p7（decode dense）；Quest p6（prefill exact）；ShadowKV p8（只算 decode 吞吐）；Agrawal p7 | 同一請求報 TTFT＋decode＋E2E；重算成本要含 prefill |
| T17 | 截斷方式與單位不一致 | Yuan；kvpress；Agrawal | Yuan p15（中段截斷、words；且 prefill 時壓縮的方法不截斷，App B.1）；kvpress pipeline L165–169（保留開頭）；Agrawal 程式預設不截斷（L483），〔複核補充〕實際 run 的 `results.json`／`config.yaml` 也都是 `max_prompt_tokens: null`／`max_context_length: null`，Mistral（32K window）收到最長 81K 的輸入 | 截斷策略寫進協定；長度一律用目標模型 tokenizer 驗算；超過模型 window 的樣本要排除或明確截斷 |
| T18 | 記憶體數字隨軟體版本變 | Yuan | p17–18 | 固定 vLLM／torch 版本並記錄；同版本內比較 |
| T19 | 校準資料單一或太短 | MInference；DuoAttention | MInference p19（1 個 30K 樣本）；DuoAttention Table 2 p17（只到 32K） | 若有學習式預測，校準與測試分開，並測長度外推 |
| T20 | 「壓縮後比全量更好」被當成結論 | Agrawal；MInference；ShadowKV | Agrawal Lessons p10；MInference p8；ShadowKV p7。〔複核補充〕Agrawal 的 CaM 在 HotpotQA 幾乎沒觸發壓縮卻高 All KV 4.1 分，差距更可能來自 harness 不同（見 T22），不是雜訊也不是壓縮 | 沒有誤差棒時，小幅「勝過全量」視為雜訊；至少 3 次重複並附信賴區間；先確認基線與方法走同一條 pipeline |
| T21〔複核補充〕 | OOM 的樣本被靜默跳過，平均值的樣本集合因方法而異（倖存者偏差） | Agrawal（示範，論文未揭露） | `benchmarking_system_performance/turboquant{3,4}/mistral-7b-instruct-v0.3/{0,1}/results.json` 的 `num_skipped_oom`＝21–23／200；同批配對後 TQ4 壓縮率 4.43→3.76、TTFT 由「比 All KV 快」變成「慢約 9%」〔計算〕 | OOM 當成結果報（失敗率、在哪個長度開始）；平均只用所有方法都跑完的配對樣本；容量懸崖量測不得把 OOM 點默默剔除 |
| T22〔複核補充〕 | 基線與各方法用不同 harness、不同生成上限、不同 query 可見性 | Agrawal（示範，論文未揭露） | All KV／KIVI：KIVI 的 LongBench 腳本；SnapKV：kvpress、`query_aware: true`；CaM：kvpress、`max_new_tokens: 64`（MultiNews 官方 512，CaM 預測平均 48.9 字 vs SnapKV 383 字）；論文 p6 卻寫用 LongBench 各集生成上限 | 所有動作（含 Full-GPU 基線）同一條 pipeline、同一份 prompt、同一生成上限；設定逐項寫進 run 紀錄 |

---

## 五、本組對 PoC 設計的建議〔判讀〕

1. **協定必須是 query-agnostic**：放置／降精度／DROP 在 prefill 結束、問題出現之前決定；每段 context 配 2 個以上、位置分散的問題（kvpress 的「壓一次答多題」＋SCBench 的 multi-request）。這也是寫入時決策的真實情境。
2. **容量用量的，不用算的**：每一層（GPU-BF16／FP8／INT4、CPU、SSD）報實測位元組，含 scale、索引、metadata；GPU 端看 vLLM 的 block 數與 peak memory。不要報名目壓縮比（T8–T10）。
3. **時間只在 vLLM 本體量**：TTFT 與 E2E 用引擎 timestamp；輸出不固定長度、保留 EOS、greedy；同時報輸出長度分布（T3–T5）。
4. **品質與時間來自同一批請求**；報逐樣本 paired 差與退化比例，不只平均（T6、A3）。
5. **長度與負載**：16K–512K 細分成多段並報每段樣本數；掃併發度；到達用真實 trace，Poisson 只當敏感度（T14–T15）。
6. **重算成本分兩版**：dense prefill（主）與稀疏 prefill（MInference 類，敏感度）。κ 在 ≥100K 時可能被稀疏 prefill 改變 1.8–10×（MInference p9）。
7. **對手實測、改寫透明**：Quest／ShadowKV／InfiniGen 這類「讀取時選擇」與 Tiara 的「寫入時放置」不同類；若納入，要同時報容量與時間，不能只比同 token budget 的準確度（Quest 卡的判讀）。
8. **答案長度要夠**：用多 token 答案任務；DROP 的影響要在足夠的 decode 步數下觀察（T2）。
9. **head 層級的異質性**：在決定粒度前，先量「streaming 類 head 的中段 KV 是否從未被讀」（DuoAttention），這可能是最便宜的 DROP 來源。
10. **重複與版本**：每點至少 3 次、附信賴區間；固定 vLLM／torch／driver 版本並記錄（T18、T20）。
11. 〔複核補充〕**OOM 與 pipeline 一致性**：OOM 當成結果報，不從平均裡剔除；所有方法的平均只用配對樣本；Full-GPU 基線與每個動作走同一條 pipeline、同一份 prompt 與生成上限，設定逐項寫進 run 紀錄（T21、T22）。

---

## 六、未查證清單

- DuoAttention 的 ICLR'25 會議版與 arXiv v1 是否有實驗差異；venue 本身（arXiv 與 repo README 都沒寫）。
- ShadowKV（ICML'25）與 InfiniGen（OSDI'24）的會議正式版與 arXiv 版差異。
- MInference：延遲實驗的 batch、重複次數；InfiniteBench 超過模型 window 時的截斷方式。
- ShadowKV：A100 記憶體容量（40／80GB）、CPU 與 DRAM、吞吐量測的輸出長度。
- Quest：端到端實驗的輸出長度；LongBench 精度實驗的 page size。
- InfiniGen、DuoAttention、Quest、ShadowKV、MInference 的重複次數（原文都未說明）。
- Rethinking：Table 3 的模型、batch、長度；長度實驗每筆是否多次抽樣。
- Yuan：模型 dtype；LongBench 是否重複。
- Agrawal & Mayer：Fig 2–3 的內容（只讀了圖說）。〔複核更新〕以下已由 V08 用 artifact `925fadf` 的 `results.json`／`config.yaml` 解決：壓縮率＝bytes 之比（可重現）；各桶樣本數；實際 run 沒有截斷（`max_prompt_tokens: null`）。CaM 與 TurboQuant 的解釋是用 repo 已存的量測紀錄重算，**沒有重跑模型**；SnapKV MultiNews 23.04（論文）vs 23.4（repo）的落差原因未查證。
- kvpress：相關論文（arXiv 2510.00636）未讀；Leaderboard 每點的樣本數；Space 的 app.py 未讀。
- `PAPERS_BY_LEVEL.md` 的「ShadowKV 在需要大量抽取資訊的任務上會挑錯（Yandex 的發現）」：來源不在 ShadowKV 原文，未追查。
- SCBench HF 資料列數（922）與論文 session 數（931）的關係。
- 各論文中模型的注意力類型，凡標〔二手〕或〔未查證〕者，原文未明寫。

---

## 複核紀錄

- **複核者**：V08（未參與抽取；沒有讀抽取者的推理或筆記，只用原文）
- **日期**：2026-10-07
- **用了哪些原始檔**：scratchpad/E08 的 9 份 PDF，先以 `pdfinfo` 與 PDF 版頭確認論文與版本（見來源清單下的〔複核修正〕），再自行 `pdftotext` 逐頁轉檔到 scratchpad/V08 重讀；Agrawal & Mayer artifact 自行 clone（commit `925fadf`，2026-05-09）；kvpress 自行 clone（`7136c22`，commit date 2026-10-05）；kvpress leaderboard Space 用 HF API 讀 `e81fc40` 的檔案清單與 `src/textual_content.py`；RULER 資料集列數用 HF datasets-server；LongBench `dataset2maxlen.json` 用 `2e00731`；ShadowKV／longctx_bench／duo-attention 的 README（2026-10-07）。

### 判定統計

| 區塊 | 檢查格數 | ✅ | ❌（已改正） | ⚠️（原文找不到） | 〔複核補充〕 |
|:--|--:|--:|--:|--:|--:|
| 來源清單 | 11 | 4 | 7（7 篇的頁數多算 1 頁） | 0 | 1（版本與 venue 核對） |
| 本組的共同模式 | 6 | 2 | 4（第 1、2、4、6 條） | 0 | 2（第 3、5 條） |
| MInference | 23 | 22 | 1 | 0 | 2 |
| DuoAttention | 23 | 23 | 0 | 0 | 0 |
| Quest | 23 | 22 | 1 | 0 | 0 |
| ShadowKV | 23 | 22 | 1 | 0 | 0 |
| InfiniGen | 23 | 23 | 0 | 0 | 0 |
| Yuan'24（23 格＋Y1–Y8） | 31 | 29 | 1 | 1 | 2 |
| Rethinking'25（23 格＋R1–R10） | 33 | 32 | 1 | 0 | 1 |
| Agrawal & Mayer'26（23 格＋A1–A6） | 29 | 22 | 7 | 0 | 新增 3 列判讀＋A7、A8，另有 8 格補充 |
| kvpress（24 格＋K1–K6） | 30 | 30 | 0 | 0 | 0 |
| 三、核對使用者文件 | 5 | 3 | 2（判定改寫） | 0 | 2 |
| 四、評測陷阱總表 | 20 | 17 | 3（T5、T9、T14） | 0 | 5 列補充（T1、T2、T10、T17、T20）＋新增 T21、T22 |
| **合計** | **280** | **251** | **28** | **1** | — |

（「✅」含只補充、內容本身正確的格；判讀格只檢查有沒有被寫成事實，沒有逐一同意或否定其推論。）

### 逐條修改（原內容 → 新內容；出處）

**來源清單**
1. 頁數：MInference 28→27、DuoAttention 21→20、Quest 12→11、ShadowKV 22→21、InfiniGen 19→18、Yuan 28→27、Rethinking 22→21（`pdfinfo`；抽取者的分頁檔最後多一個空頁）。各卡 p 幾的引用抽查與 PDF 頁一致。

**共同模式**
2. 第 1 條：「MInference、Quest、DuoAttention、Agrawal 都是一次一個請求」→ 只有 Quest（p8）與 Agrawal（程式碼）能確定；MInference、DuoAttention 沒寫 batch（全文搜尋 batch：MInference 0 次；DuoAttention 只有 p6「適合大 batch」與 p7 找 head 時 batch 1）。
3. 第 2 條：「Agrawal 系統用另外 3 個」→ Qasper 兩邊都有（p5–6）；補充兩套品質 harness。
4. 第 4 條：「metadata（Quest、InfiniGen）常不計入」→ 兩篇都有提（Quest p5、InfiniGen p13），只是沒併入壓縮比；補充 Agrawal Table 4 其實是實測位元組。
5. 第 6 條：「Agrawal repo 有 3 個 run 目錄」→ 品質 3 個、系統 2 個（repo 目錄）。

**各卡**
6. MInference 模型格：「附錄 LLaMA-3-70B-Instruct-262K」→ 正文 Table 6 p9，正文稱 70B-1M、註腳與表頭為 262k。
7. Quest 主要結果③：稀疏度 1/6、1/6、1/5、1/10、1/5、1/6 補上原文順序 Qasper、HotpotQA、GovReport、TriviaQA、NarrativeQA、MultifieldQA（p6），避免依「資料」格順序錯配。
8. ShadowKV 設計理由：「LongBench 輸入較短，所以只取 >4K」→ 同一句同時交代「只測 >4K」與「budget 256」（p7）。
9. Yuan 與既有整理不一致：「Llama-3 那半的結論其實是 ≤7.5K」→ 只對 baseline／KIVI／FlexGen／H2O 成立；InfLLM、StreamingLLM 收到完整輸入、預算 ≤7,500÷壓縮比（App B.1 p15）。同卡「長度」格補上此點；「資料」格補 RepoBench-P 500（p14）；「程式碼」格的「含完整 log」改標〔二手〕（⚠️，未進 repo 確認）。
10. Rethinking 品質指標：「p2 註」→「p2 註 1、p9 註 2」（benign＝分數 ≥ 平均在 p9 註 2）。
11. Agrawal 資料格：「系統：另外三集」→ Qasper 重疊；補 CaM 品質 run `max_new_tokens: 64`（`config.yaml`）。
12. Agrawal 重複與統計：「TurboQuant Mistral 只有 0/1」→ 兩模型都有 0/1/2，multi_news 只在 run 1；系統評測只有 0/1。
13. Agrawal CaM 判讀：「約（輸入＋輸出）÷1,024」→ FP16 在 prefill 後量（`benchmarking_kivi.py` L339），關係是「輸入 ÷ 約 1,040」；`results.json` 證實 GovReport 每桶平均 `compressed_cache_tokens` 1,038–1,042、算得 3.02／5.98／13.76 對表 3.09／6.04／13.82。
14. Agrawal 原文沒講清楚②：「NarrativeQA 4–8K 可能只有 1 筆」→ 8 筆、長度皆 7,964（`results.json`）。③ 由「是否截斷未知」→ 實際 run 皆 `max_prompt_tokens: null`／`max_context_length: null`，沒有截斷。
15. Agrawal 使用者文件格②：「長度主體 ≤8K」→ 系統評測 600 筆中過半在 8K+（Llama 313、Mistral 354）。①「L1 太窄」降級為使用者術語一致性問題（intro 表 6 txt L329 把 KVP 歸 L1）。
16. Agrawal A3：「不同資料集」→「大致不同（只有 Qasper 重疊）」。

**三、核對使用者文件**
17. intro L836：原判「⚠️ 過度簡化」→ 此句沒有事實錯誤；「L1 太窄」❌ 不成立；「系統效能只是單請求」❌ 不適用（此句沒提系統效能）；「L2 分層還沒有」✅（就 [56] 而言）。
18. sota L395：「L1 太窄」❌ 不成立；「系統效能要加限定」✅ 成立；原建議的「≤8K 為主」刪除。
19. 參考文獻、SCBench 兩列：✅ 維持，補上全文搜尋結果與頁碼。

**四、評測陷阱總表**
20. T5：補註這是本卡對 Rethinking Table 5 的重讀〔計算〕，原文結論相反（p8），不是原文「指出」的陷阱。
21. T9：刪去 Agrawal 為「名目壓縮比」的例子（實測位元組，可重現）。
22. T14：刪去 Agrawal 為「長度太短」的例子（過半樣本 8K+、最長 81K）；Yuan 的 7,500 加註適用範圍。
23. T17：Agrawal 由「程式預設不截斷」更新為「實際 run 設定也是不截斷」。
24. 補充 T1（Agrawal SnapKV `query_aware: true`）、T2（CaM 32 步才壓，短答案從未觸發）、T10（量測時點不同）、T20（HotpotQA 無壓縮仍高 4.1 分）。
25. 新增 T21（OOM 樣本被靜默跳過的倖存者偏差）、T22（基線與方法 harness／生成上限／query 可見性不一致）。

### 抽取者漏掉、對評測設定重要的事實（〔複核補充〕，皆來自 artifact `925fadf` 已存的量測紀錄，沒有重跑模型）

- TurboQuant 在 Mistral NarrativeQA 有 21–23／200 筆最長樣本 OOM 被跳過，Table 4 的 4.43（TQ4）與 Table 3 的「TTFT 比 All KV 低」都是倖存者偏差；配對後壓縮率 3.76、TTFT 慢約 9%。論文 p10 的解釋（旋轉量化在長 context 分布更均勻）因此不成立。
- SnapKV 的品質數字在 `query_aware: true` 下產生；CaM 的品質 run 生成上限一律 64（MultiNews 官方 512）；All KV／KIVI 與 SnapKV／CaM 走不同 harness。CaM 在 HotpotQA 幾乎沒觸發壓縮仍高 All KV 4.1 分，論文「壓縮不必犧牲準確度」（p10）證據不足。
- SnapKV Llama MultiNews：論文 23.04，repo 三個 run 都是 23.4。

### 因時間沒有逐一檢查的部分（如實列出）

- 只出現在圖上的數值（DuoAttention Fig 7、9–11 的曲線、Quest Fig 7–11、Rethinking Fig 1、4、6、Agrawal Fig 2–3、MInference Fig 10–12）沒有讀圖驗證；卡片引用的倍數都已在正文文字中找到。
- 表格只核對卡片引用的那幾格：ShadowKV Table 2、5–11、15–16，Yuan Table 5–9（App D），Rethinking Table 9–10、App E–G，MInference Table 8 以外的附錄，沒有逐格看。
- SCBench 只細讀 p1–9，p10–31 僅用關鍵字搜尋（TTFT／latency／throughput 等）確認沒有系統指標結果。
- kvpress：`presses/{scorer,snapkv,decoding,base}_press.py` 只確認存在，未逐行讀；Space 的 `app.py`、kvpress 論文 arXiv 2510.00636 都沒讀（與抽取者相同）。Leaderboard 每點的樣本數仍未查證。
- Agrawal：Fig 2–3 的內容；品質 harness 差異造成的分數差多大，沒有重跑實驗分離；只看了 Llama 的 CaM／SnapKV 預測長度，Mistral 未看。
- `workloads_eval.md`、`PAPERS_BY_LEVEL.md` 只核對卡片引用到的那幾列。
