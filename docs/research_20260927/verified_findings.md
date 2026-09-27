# 2026-09-27 workflow 的綜合結論（原文照錄）

> `deep-research` workflow（run `wf_00cb5dd9-097`）最後一個 synthesize agent 的輸出，未改寫。它只用通過三票驗證的 10 條 claim；【算術】【判讀】是綜合者自己的推論，沒有經過驗證。三票驗證的投票者都是 Claude subagent，**不是 cross-model review**。
> 統計：{"angles": 5, "sourcesFetched": 28, "claimsExtracted": 140, "claimsVerified": 25, "confirmed": 10, "killed": 15, "unverified": 0, "afterSynthesis": 9, "urlDupes": 0, "budgetDropped": 2, "agentCalls": 110}

## 摘要

這一輪有 10 條 claim 通過三票對抗式驗證。它們共同指向一個改題候選：把 Tiara 的放置單位從同質的 KV block，擴大成「異質的推論狀態物件」。這些物件包括 hybrid 模型的 recurrent-state checkpoint、跨 LoRA adapter 或 fine-tune 借用的 KV、position-free 的 chunk KV，以及同一 block 的多精度副本；每一類都推翻 main.tex 形式化裡的一條前提（單一 ℓ_i／π_i、只有 KV、前綴命中、KV 只屬於一個模型）。證據最強、也最適合在一個月內用 7×3090＋1×MI300X 做的是方向 8–10（Gated DeltaNet／KDA 類 hybrid 模型）：Sparse Prefix Caching 只放 CPU DRAM，DASC 只在 HBM，Tail-Replay 只做近似重建，三篇都沒有把 state 和 KV 一起放進 HBM/DRAM/SSD＋精度＋重算的決策；依 HF config 換算（非實測），一份 checkpoint 等於 3,144 到約 5,384 個 token 的 KV，佔用和 KV 同量級，SGLang 的一個重現也顯示 state 池滿時 host KV 命中歸零。方向 11–13（跨 adapter 借用 KV 的 ε、品質損失在 agent 迴圈中拉長軌跡、非前綴命中需要可共享的 canonical chunk KV）的機制端已經擁擠，剩下的空間在長上下文與分層量測；方向 14（VeriCache 的多精度副本加無損 verify）主要是論文必須正面回應的威脅，但可以和故事 2、3 合成「何時值得保留多副本」的判準。angle B（P/D 分離、RDMA／CXL／GDS）與 angle D 的經濟和能耗面，這一輪沒有 claim 通過驗證，所以沒有從這兩處提方向；所有「還沒人做」的判斷都只相對於本輪讀過的論文，動手前要先做 novelty-check。

## F0. （信心 medium；票數 綜合：[0][3][4][5][6][7][8][9] 為 3-0，[1][2] 為 2-1）

改題候選（總綱）：把 Tiara 的決策單位從「同質的 KV block」擴成「異質的推論狀態物件」，κ 和故事 6 的事前判準也改成按（硬體, 物件類型）各算一次。2026 年的服務系統已經同時快取四類性質不同的物件：hybrid 模型的 recurrent-state checkpoint（單一大物件，只在邊界有效）、跨 adapter 或 fine-tune 借用的 KV（帶 ε）、position-free 的 canonical chunk KV（跨位置共享），以及同一 block 的多精度副本（有損版放 HBM、無損版放 host）。

【推翻的假設】main.tex §Problem Formulation 的 s_t 給每個 block 一個所在層 ℓ_i 與一個精度 π_i；動作空間只有 6 種 KV 動作；命中採前綴語意；KV 只屬於一個模型。本輪證據分別打掉其中一條：
- hybrid 模型需要 KV 以外的 state checkpoint（DASC、Sparse Prefix Caching、SGLang PR #39436，見方向 8–10）。
- VeriCache 讓 INT4@HBM 和完整 KV@DRAM 同時存在（方向 14）。
- LRAgent 和 DroidSpeak 讓 KV 跨 adapter 或 fine-tune 借用（方向 11）。
- MEPIC 指出，每個請求各自修補的非前綴 KV 無法 page 共享（方向 13）。
【最接近前作】每一類物件都有人做。但在本輪讀到的論文裡，沒有一篇跨物件類型做 HBM/DRAM/SSD＋精度＋重算的聯合放置：
- Marconi 做 hybrid 的 admission/eviction，依 9/24 筆記沒有自己的 CPU/SSD 階。
- Sparse Prefix Caching 只放 CPU DRAM。
- DASC 只在 HBM 內。
- VeriCache 做多精度副本，但物件只有 KV。
- LRAgent 只做共享，不做分層。
【新穎性風險】中。總綱本身容易被審稿人看成「把幾個機制拼在一起」，要有統一的成本模型（把 5 常數＋MRC 判準推廣到多物件）才站得住。hybrid 狀態快取這條線在 2026-08-31 到 09-14 之間就冒出 DASC、Tail-Replay 和 SGLang PR，撞車機率高。
【可行性】一個月內做不完全部。建議：
- 方向 8＋9＋10（hybrid 模型的狀態分層）當實作載體。
- 方向 11–12 當第二條線。
- 方向 13–14 只進 related work 與形式化的修正。

來源：https://arxiv.org/abs/2605.05219、https://arxiv.org/abs/2608.30386、https://arxiv.org/abs/2608.30310、https://github.com/sgl-project/sglang/pull/39436、https://arxiv.org/abs/2605.17613、https://arxiv.org/abs/2602.01053、https://arxiv.org/abs/2411.02820、https://arxiv.org/abs/2512.16822

## F1. （信心 high；票數 [3] 3-0、[4] 3-0、[6] 3-0、[7] 3-0）

方向 8：hybrid linear-attention 模型（Gated DeltaNet／KDA，例如 Qwen3-Next、Kimi-Linear、Qwen3.5／3.6、OLMo-Hybrid）的 recurrent-state checkpoint，是性質和 KV block 不同的新快取物件：會被就地覆寫、只在邊界有效、單份很大，SGLang 預設以 FP32 存 temporal state。把它和 full-attention KV 一起放進 HBM/DRAM/SSD＋精度＋重算的動作空間，是本輪最清楚的空白。已讀的三篇各只處理其中一面：Sparse Prefix Caching 只放 CPU DRAM，而且 DP 目標不含載入成本；DASC 只在 HBM 內壓縮 checkpoint；Tail-Replay 只做近似重建。

【證據】
(1) 物件性質（DASC，arXiv 2608.30386，3-0）
- recurrent state 會隨 token 就地覆寫，不保留較早的 prefix 邊界。所以 SGLang 每隔固定 token 數，就要存一份涵蓋所有 linear-attention 層的完整 checkpoint，和 KV block 並存。
- 沒快取到的 checkpoint 只能靠 prefill 重算。
- per-rank 預算是 694,681,600 B（Kimi-KDA）和 1,236,271,104 B（Qwen-GDN），各等於 128 個 dense slot。換算下來，TP8 時每個 rank 的一份 checkpoint 約 5.43 MB 和 9.66 MB。驗證者用 HF config 逐位元組核對過。
(2) 大小換算【算術，非實測】
- 乘上 8 個 rank，一份完整 checkpoint 約 77.3 MB（Qwen3-Next-80B-A3B）和 43.4 MB（Kimi-Linear-48B-A3B）。
- 依 HF config：Qwen3-Next 的 48 層中有 12 層 full attention（2 個 KV head、head_dim 256）；Kimi-Linear 的 27 層中有 7 層 MLA（latent 512+64）。BF16 KV 每 token 分別是 24,576 B 和 8,064 B。
- 所以一份 checkpoint 等於 3,144 個和約 5,384 個 token 的 KV（以整個模型的唯一位元組計）。
- 若照 SGLang PR 的設定，每 4,096-token chunk 存一份，checkpoint 約是同一段 KV 的 0.77 倍和 1.31 倍。
(3) 命中成本（Sparse Prefix Caching，arXiv 2605.05219，3-0）
- 分三項：checkpoint 從 CPU 載入 GPU、recurrent 層重播 suffix、attention 層載入 cached KV。
- wall-clock 約為待重算 token 數的線性函數，加上一個會隨快取狀態量變大的常數。
- checkpoint 放置有精確的 O(NM) DP；overlap 均勻分布時，最佳期望重算量為 N/(2(M+1))+O(1)。
- 限制：DP 目標只算重算 token 數；全文搜尋 SSD、NVMe、FP8、INT4、BF16 都是 0 次命中；硬體只有一張 RTX 2080 Super 加 16 GB DDR4。
(4) DASC 的範圍（3-0）
- 所有實驗都在 HBM 內，環境是 Hopper、TP8、SGLang、LRU。
- 它把 CachedAttention、RAGCache 列為「互補」；全文搜尋 offload、CPU、SSD、DRAM 都是 0 次命中。
【推翻的假設】
- 快取單位是 per-token KV block，各自獨立逐出。
- κ 和 5 常數判準只需要 KV 的常數。
- DROP 的重算成本隨位置增加。對 recurrent 層而言，重算成本只取決於從最近的 checkpoint 要重播幾個 token，與絕對位置無關；full-attention 層仍與位置相關【判讀】。
【最接近前作】
- 論文：Marconi（MLSys'25，已讀）、Sparse Prefix Caching（2026-04 preprint）、DASC（2026-08-31 preprint）、Tail-Replay（標註 NeurIPS'26 ML for Systems workshop）、HeadWiseKV 2609.02029（只讀摘要）、ReplaySSM（Dao 2026）、Kimi K3 的 KDA state 持久化。
- 引擎：vLLM v0.11.1 加入 mamba_block_size；v0.15.0 起，開啟 prefix caching 時預設 mamba_cache_mode=all，在 block 邊界存 state。所以「稀疏地存 checkpoint」本身不新；新的是依重疊分布調整放置，以及跨階層的成本。
- 9/24 自查（self-review 等級）：MLPerf v6.1 Agentic 用了 Kimi K3（KDA）和 Qwen3.6-35B-A3B（GDN hybrid）；YAKV 自承只適用 full-attention 模型。
【新穎性風險】中偏高。空白只在這幾篇裡確認過，驗證者明講「the gap exists in this paper, not in the field」；主題正在快速冒出。
【與既有故事的連結】
- 故事 1 可以直接移植：把 FP32 checkpoint 降到 BF16 或 FP8，是否也會吃掉放置的空間？
- 故事 2 預測：state 是少量大物件，KV 是大量小 descriptor，兩者的有效頻寬和 κ 會不同【判讀，待量】。
- 故事 4 的「load front, recompute tail」在 hybrid 模型上是結構性的（見方向 9）。
【可行性（1×MI300X＋7×3090，約 1 個月）】
- 3090 可以跑這些論文用過的小型 hybrid 模型：Qwen-3.5-0.8B、Qwen3.5-4B、OLMo-Hybrid-7B。
- MI300X 的 192 GB 放得下 BF16 的 Kimi-Linear-48B（約 96 GB【算術】）；Qwen3-Next-80B（約 160 GB【算術】）放得下，但剩下的空間很少。
- 第 1 週：確認 vLLM 0.28 是否支援這些模型的 prefix caching，以及 OffloadingConnector 會不會把 mamba／GDN state 一起卸載（都未查證）。
- 第 2–3 週：量 state 的 κ，以及「checkpoint 間隔 × 階層」的成本曲面。
- 第 4 週：在 Sparse Prefix Caching 的 DP 裡加上各階的載入成本。
- ROCm 對這些模型的支援未查證，是主要風險。

來源：https://arxiv.org/abs/2608.30386、https://arxiv.org/abs/2605.05219、https://arxiv.org/abs/2608.30310、https://github.com/sgl-project/sglang/pull/39436、https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/config.json、https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/raw/main/config.json、https://arxiv.org/abs/2609.02029、https://mlcommons.org/2026/07/agentic-inference-for-mlperf-inference/

## F2. （信心 low；票數 [5] 3-0（另一條把它解讀成 dead memory 的 claim 為 1-2，被否決））

方向 9：hybrid 模型的 host tier 其實是兩個互相牽制的池，一個放 KV，一個放 state checkpoint；容量切分與逐出必須聯合決定。SGLang HiCache 有一個重現：state 池滿了以後，被逐出的 prompt 重放時 host 命中是 0/8，TTFT 0.82 s，等於 cold，但當時 KV 池還有 35% 是空的。把 state 池加大到 734 個 slot 後，命中變成 8/8，TTFT 降到 0.15 s。

【證據】sgl-project/sglang PR #39436，3-0。這個 PR 仍是 open、未經審查、CI 失敗；證據只有單一設定、8 次重放。
- 模型與硬體：GLM-5.3-Flash（sparse DSA 加 linear attention 的 hybrid，320B 參數，18B active），TP4，4× RTX PRO 6000 Blackwell。
- 設定：--hicache-size 32、write_through、chunked prefill 4096。
- 預設切分：只有 565 個 host Mamba slot，對應 2,680,896-token 的 KV host tier，coverage 82%。
- 負載：灌入 200 個 8,448-token 的 prompt。每個有 3 個 checkpoint，共 600 個，超過 565。
- 改用 --hicache-mamba-size-gb 14 後，從 host 還原了 8,192 token。
- 每個 slot 約 19.06 MB，是驗證者從 PR 裡的兩組設定推出來的。
【附帶觀察】還原停在 8,192，也就是和 chunk 對齊的最深 checkpoint；剩下 256 token 要重算。在 hybrid 模型上，「load front, recompute tail」不是策略選擇，而是 checkpoint 粒度造成的結構性結果。驗證者用 PR 的 memory.py 和文件核對過這個推導。
【推翻的假設】
- host tier 的容量等於 KV 的容量，命中率只受 KV 池大小影響。
- MRC 只對一種物件畫就夠。實際上兩池要聯合看，其中一池多出來的 byte 可能完全沒用【判讀】。
- 另一條把這點強化成「dead memory／單池規劃會系統性失效」的 claim 以 1-2 被否決。所以目前只能說有一個重現，不能說是普遍現象。
【最接近前作】這支 PR 本身（它的 'auto' 模式會讓 checkpoint slot 覆蓋 KV host tier）、Marconi（已讀）、vLLM 的 mamba_cache_mode。
【新穎性風險】單獨成文的價值偏低：這是工程修正，而且 PR 已經提出自動切分。比較適合當方向 8 的 motivating measurement，或用來實證「兩種物件的聯合 MRC／聯合容量切分」這個理論小節。
【可行性】高，約 1–2 週。
- 在 3090 上用小型 hybrid 模型，於 vLLM（或套用這支 PR 的 SGLang）重現「state 池滿、KV 池有空卻 0 命中」。
- 掃 state 和 KV 的切分比例。
- 這支 PR 的 AMD ROCm CI 目前失敗，所以不要先在 MI300X 上走這條。

來源：https://github.com/sgl-project/sglang/pull/39436

## F3. （信心 medium；票數 [8] 3-0（單一來源））

方向 10：recurrent state 的「有損重建」可以成為一個帶自己 ε 的新動作，但直接 DROP state 的代價不小。Tail-Replay 用 replay 比例 r 近似重建 hybrid 模型的 linear-attention state。r=5% 時，品質保留在 full prefill 的 92.8–98.9%（LongBench）與 93.1–99.9%（RULER）。完全不重建時（只重用 FA KV，state 歸零），RULER 平均掉到 0.215–0.415，full prefill 則是 0.812–0.971。品質損失依模型與任務而異，OLMo-Hybrid-7B 最差，和故事 3 一致。

【證據】arXiv 2608.30310，TeleAI 與上海交大，標註 ML for Systems @ NeurIPS 2026 workshop，3-0。
- 模型與平台：OLMo-Hybrid-7B、Qwen3.5-4B、Qwen3.6-27B，H100，PyTorch 2.9.1。
- r=10% 時：LongBench 保留 93.9–98.1%，RULER 保留 96.7–99.9%（依逐格資料，上限是 100.0%）。
- r 加大不一定比較好：Qwen3.5-4B 的 qasper 從 99.0% 掉到 94.7%，OLMo 的 musique 從 92.5% 掉到 89.5%。
- 最差單格：OLMo 的 LongBench qasper@16K，從 .394 掉到 .308（78.2%）。
【限制】
- RULER 只測 4 個任務，長度只有 8K 和 16K。LongBench 測 5 個任務，長度 16K；narrativeqa 另外測到 32K 和 64K。
- 每個設定只跑一次，沒有變異數，也沒有整合進 vLLM 或 SGLang。
- Qwen3.6-27B 的 RULER 數字前後不一致：Table 1 寫 0.987，逐格平均是 0.971。
- 另一條描述 Tail-Replay 機制細節的 claim 以 0-3 被否決。引用機制前要回原文重讀。
【推翻的假設】
- 命中只有「精確命中」和「未命中」兩種。
- state 可以像 token KV 那樣丟掉。zero-only 的崩潰說明 state 不能直接 DROP。
【對 Tiara 的用法】動作空間多一個「state 近似重建（預算 r）」。它的成本是 r·m 個 token 的 replay，品質代價是 ε(model, task, r)，可以直接和「把精確 checkpoint 存在 DRAM 或 SSD」比較。故事 1 說放置的價值被量化吃掉了，只剩「品質不允許全面降精度」這個情境；這個動作可能讓品質約束下的放置在 hybrid 模型上重新有意義【判讀】。
【最接近前作】
- Tail-Replay 本身。
- DASC 的 dasc-wr：在 HBM 內從有界的 suffix 補回被省略的 state 單元，屬於靜態設定。
- Sparse Prefix Caching：精確、無損的做法。
【新穎性風險】近似重建這個機制已有人做，風險高。在品質約束下，決定「精確 checkpoint 放哪一階」還是「做近似重建」，本輪沒看到有人做，風險中等。
【可行性】高，約 2–3 週。Qwen3.5-4B 和 OLMo-Hybrid-7B 在 3090 上可以跑，既有的 m5_quality harness 可以直接用。要補的是 64K–128K 的長上下文，以及我們既有的模型 × 任務矩陣。

來源：https://arxiv.org/abs/2608.30310、https://arxiv.org/abs/2608.30386、https://arxiv.org/abs/2605.05219

## F4. （信心 high；票數 [0] 3-0、[1] 2-1）

方向 11：在 multi-LoRA agent 之間，或在同一底座、不同 fine-tune 的模型之間，「借用別人的 KV」是一個帶自己 ε 的新動作，應該和 BF16/FP8/INT4、DROP 放在同一個動作空間裡比較。已驗證的品質代價：直接共用完整 KV、完全不重算（FullShared）時，平均準確率最多掉 5.3%；只共用 base cache（BaseShared）最多掉 0.7%；DroidSpeak 式的部分層重算，在 LRAgent 的重現中最多掉 2.6%。這些 ε 都是在 1.0–1.5K token 的 context 上量的，長上下文仍是空白。

【證據】
1. LRAgent（arXiv 2602.01053，ICML'26 poster，3-0）
- 設定：AutoAct 式的 plan、action、reflect 三個 LoRA agent，跑在 HotpotQA 和 ScienceQA；模型是 LLaMA-3.1-8B 和 Ministral-8B。
- 平均掉分的最大值：FullShared 5.28、DroidSpeak 2.63、BaseShared 0.67、BaseLRShared 1.43。
- 每個難度跑 20 次，std 0.16–0.45%，只用單一 seed。
2. DroidSpeak（arXiv 2411.02820，NSDI'26，2-1）
- 規模：8 組同架構、不同 fine-tune 的模型對；共 6 個資料集，每組報 3 個。
- 硬體：兩台 8×A100 VM，以 200 Gbps InfiniBand 相連；70B 模型用 4-bit AWQ。
- 效果：prefill 延遲比 full prefill 降 1.7–3.1×，平均 2.1×。這是離線量的 prefill 延遲，不是負載下的 TTFT。
- 吞吐最多 4×，但只來自 HotpotQA，而且只有 8 組中的 4 組。
- 「可忽略」的定義：profiling 時品質下降不超過 5%；§5.3 改用 1%。
3. LRAgent 用官方設定重跑 DroidSpeak
- TTFT 最多快 1.56×，吞吐最多 1.36×。
- Ministral 在 66.4K context 時 OOM，原因是 GQA 下 hidden-state cache 比單層 KV 還大。
- 但 LRAgent 重算的是分散的層。NSDI'26 原文重算的是連續的層組，並警告只重算分散的關鍵層會造成大誤差，所以 2.6% 可能偏悲觀。
4. 驗證者另外引用、未單獨驗證的來源
- ForkKV（2604.06370）：跨 adapter 完整重用平均掉 5.40 分，APIGen 上最多掉 21.95 分；只共用 base cache 最多掉 1.60 分。
- 2609.17109：完整前綴重用在 HotpotQA 掉 0.8–4.6 EM。
- aLoRA：adapter 若為此設計，跨 adapter 重用可以無損。
- 由此可知 5.3% 不是上界。
【推翻的假設】
- 一份 KV 只屬於一個模型，因此 multi-adapter 服務的工作集等於「adapter 數 × 單份 KV」。
- 要分層的物件只有 KV。DroidSpeak 式做法還要另存 hidden state，而 GQA 下 hidden state 比單層 KV 大。凡是以「存 hidden state 比存 KV 省」為前提的設計（例如 HCache），在 GQA 模型上都要重驗【判讀，需回原文確認】。
【最接近前作】LRAgent、DroidSpeak、ForkKV、aLoRA。共享機制這塊已經很擠。
【新穎性風險】
- 共享機制本身：高。
- 以下兩點本輪沒看到有人做，風險中等：把 base KV（大、共享、重用多）和各 adapter 的差量（小、私有）當成不同物件來分層；在 32K–128K 下，借用 KV 的 ε 和 INT4 的 ε 疊加時是相加還是互相放大。
【可行性】高，約 2–3 週。
- Llama-3.1-8B 加 LoRA 放得進 3090。
- 借用 KV 的 ε 不用改引擎就能量：用 HF transformers，以 adapter A 做 prefill、adapter B 做 decode 來模擬。
- 既有的品質 harness 可以擴到長上下文。
- 要先確認 vLLM 的 prefix cache 是否把 LoRA 身分放進 block hash（未查證）。如果有，跨 adapter 的命中預設就是 0。

來源：https://arxiv.org/abs/2602.01053、https://arxiv.org/abs/2411.02820、https://arxiv.org/abs/2604.06370、https://arxiv.org/abs/2609.17109

## F5. （信心 low；票數 [0] 3-0（本方向用的是該 claim 中帶 hedge 的部分））

方向 12：品質損失會回頭改變負載。在 agent 迴圈裡，準確率較低的快取方法會讓 agent 多走幾步，累積更長的 context。這對故事 1「全域 INT4 加 LRU 約等於 oracle」是一個還沒檢驗過的威脅：在閉迴路的 agentic workload 下，INT4 的 4 倍容量優勢可能被變長的軌跡吃回去。因此評估放置策略時，應該用端到端的 agent 任務，而不是只看單一請求的品質。

【證據】LRAgent（arXiv 2602.01053，3-0）原文：「latency in agent systems depends on both the cache sharing method's efficiency and the system accuracy, since lower-accuracy methods tend to take more steps and accumulate longer contexts.」附錄 Table 17：LLaMA 上的平均序列長度，Non-Shared 是 1,093 token，FullShared 是 1,514 token。
【反例與限制】（驗證者）
- 作者用的是「tend to」，不是定律。
- Ministral 上，FullShared 的平均 E2E 延遲 13.83 s，反而比 Non-Shared 的 14.18 s 低。
- BaseLRShared 與 FullShared 的準確率差很多，序列長度卻幾乎一樣（1,412 vs 1,417）。
- 作者沒有量 KV 佔用；「KV 佔用增加」是推論。
- context 只有 1.0–1.5K token。
【相關前作】Rethinking KV Cache Compression（MLSys'25，arXiv 2503.24000）在單一請求的層級觀察到：壓縮讓輸出變長，端到端延遲反而增加。這筆出自 9/24 自查，未經對抗式驗證。
【推翻的假設】
- ε 是每個請求的靜態約束，和未來的負載無關。
- 我們的 oracle 與 headroom 是在固定 trace 上算的，隱含假設軌跡長度不會隨放置決策改變。
【最接近前作】LRAgent（共享 KV 的閉迴路效應）、Rethinking KV Cache Compression（單一請求的輸出變長）。「精度階 × agent 閉迴路 × 分層容量」這個組合，本輪沒看到有人做。
【新穎性風險】中。這個效應本身已有人觀察到；我們的角度是把它量化進放置的成本模型，並用來檢驗故事 1。
【可行性】中，約 2 週。
- 做法：在 3090 上用 Llama-3.1-8B，分別以 BF16、FP8、INT4 KV 跑 AutoAct 式的 HotpotQA 三角色 agent，量步數、總 token、KV 峰值與成功率。
- 不要選 Qwen2.5：它在 FP8 和 INT4 下會崩（故事 3），量到的會是整體失敗，而不是「微小 ε 拉長軌跡」。
- 要多跑幾個 seed。這個效應可能很小，單一 seed 容易只量到雜訊。

來源：https://arxiv.org/abs/2602.01053、https://arxiv.org/abs/2503.24000

## F6. （信心 medium；票數 [2] 2-1）

方向 13：非前綴命中要真的省到容量，每一階存的 chunk KV 就必須是 canonical、可以 page 共享的。CacheBlend、EPIC 這類 position-independent caching，會對每個請求各自做 selective recomputation 和位置調整。結果同一個 chunk 的 KV 在不同請求之間分歧，在 paged KV cache 中無法對齊或共享，HBM 只省下有限的量。這是故事 4（區段命中加雙向重疊）必須遵守的設計約束：修補結果不能寫進共享頁。

【證據】MEPIC（arXiv 2512.16822，Huawei，2025-12-18，未經同儕審查，2-1）§2.1.3 原文：「because recomputation and positional adjustment are performed independently per request, the resulting KV representations diverge across requests and cannot be page-aligned or shared in the paged KV cache.」摘要說，這樣只換來「modest HBM savings even when many requests reuse the same content」。
【範圍】（驗證者）
- 重複只發生在 HBM。MEPIC §2.2 說 chunk 放在 CPU 或 disk 階，要用時才搬進 HBM，所以 CPU／disk 階每個 chunk 只有一份 canonical copy。
- MEPIC 只量了一個模型（Mistral-7B-Instruct-v0.3）、一種硬體（Ascend 910B）。
【已有的解法】（驗證者的描述；另一條更細的 MEPIC 機制 claim 以 0-3 被否決，引用前要回原文確認）
MEPIC 和 IBM 的 MiniPIC（arXiv 2606.13126，2026-06，在 IBM 的 vLLM fork 內）都做到讓同一組實體 block 服務所有位置的並行存取，做法是：
- K 以未旋轉的形式存，到 attention kernel 內才套 RoPE。
- chunk 對齊 block。
- 只重算第一個 block。
所以「可共享的 PIC」本身不是空白。
【推翻的假設】
- 非前綴命中和前綴命中省下一樣多的記憶體。
- 放置可以不管 KV 的表示方式。旋轉過的 K 綁死了位置，只有 canonical 表示能跨位置去重，而這會改變每一階存的位元組數與重建成本。
【最接近前作】CacheBlend（已讀）、EPIC、MEPIC、MiniPIC。KVShareArena（arXiv 2609.10266）自稱是非前綴重用的 benchmark，但本輪關於它的 3 條 claim 全部以 0-3 被否決，不能引用。
【新穎性風險】中偏高，因為 MEPIC 和 MiniPIC 已經做到共享。可能還剩的角度：
- 32K 以上長上下文、NVIDIA 或 AMD 硬體上，canonical chunk KV 在 HBM/DRAM/SSD 之間的放置與重建成本。
- 把故事 2 的軟體路徑成本套進來。
- MEPIC 是否量過 CPU/disk 階的載入延遲，本輪無法確認（那條 claim 被否決）。
【可行性】中。MiniPIC 的程式碼是否公開、能否在 vLLM 0.28 上跑，都未查證；若不行，就得自己改 attention kernel 裡 RoPE 的位置處理，一個月內偏緊。比較穩的做法是先用我們的 trace，以算術或模擬求出 canonical 與逐請求修補兩種做法的 HBM 佔用差，放進故事 4。

來源：https://arxiv.org/abs/2512.16822、https://arxiv.org/abs/2606.13126

## F7. （信心 high；票數 [9] 3-0）

方向 14（主要是威脅，其次才是機會）：同一個 KV block 可以同時以多種精度存在多個階層。VeriCache 把壓縮過的 KV 放在 GPU HBM 做 draft，完整 KV 放在 CPU DRAM（或 storage）做 verify；greedy decoding 下，輸出和用完整 KV 解碼相同。壓縮器可以是 token-dropping（KVzip、KVzap、ExpectedAttention、SnapKV），也可以是量化（KIVI、KVQuant、RotateKV，8／4／2 bits）。這直接推翻 main.tex「每個 block 只有一個 ℓ_i 與一個 π_i」的前提，也讓「把精度階當 speculative draft」這個方向的新穎性風險很高。

【證據】VeriCache（arXiv 2605.17613，UChicago／Tensormesh／Samsung Semiconductor／Microsoft Research，2026-05-17，3-0）
- 實作：建在 vLLM 和 LMCache 上，約 8K 行 Python 與 C++。
- 運作：壓縮的 cache 留在 HBM 做 draft，每次 verify 都從 CPU 重新載入完整 KV。
- 「相同」的範圍：greedy 下相同，但排除硬體不確定性；sampling 時 KL < 0.01 nats。
- 遠端 prefix caching 情境：完整 KV 放在 verifier GPU 旁的 storage node（40 GB/s），drafting GPU 在遠端（1.2 GB/s）。這是本輪唯一碰到 angle B 的證據。
- 全文沒有提到 SSD、NVMe 或 AMD 硬體。
【更早的前作】（驗證者）
- QuantSpec（arXiv 2502.10424，2025-02）已經用階層式 4-bit 量化 KV 做 self-speculative decoding。
- TriForce（COLM 2024）有一個設定：剩下的 KV 放 CPU，GPU 上放 retrieval draft cache。
- MagicDec、SparseSpec 從稀疏或壓縮的 cache 做 draft，但完整 KV 固定放在 HBM。
- VeriCache 新在三點：完整 KV 不在 HBM、任何壓縮器都能接、支援遠端 prefix caching。它正是這條路線的分層版本。
【推翻的假設】main.tex §Problem Formulation 的 s_t 讓每個 block 只有一個 ℓ_i 和一個 π_i，而且「精度維度只存在於 GPU 常駐的三階」。審稿人可以直接問：為什麼不同時保留 INT4@HBM 和 BF16@DRAM？
【我們還能做的角度】【判讀，待量】VeriCache 的收益取決於兩件事：
- (a) draft 的接受率。這就是故事 3 的 ε(model, task)；Qwen2.5 在 FP8／INT4 下崩潰，表示它的 draft 很差。
- (b) 每次 verify 重載完整 KV 的成本。故事 2 顯示這由軟體路徑決定：MI300X 上每筆 descriptor 約 13 µs 固定成本，跨模型差 17 倍。
兩者合起來可以預測：最需要無損保證的模型，正好是 draft 最差、verify 最貴的模型。這可以寫成「何時值得用多副本做無損分層」的事前判準，是故事 6 的自然延伸。
【新穎性風險】機制本身已被 VeriCache、QuantSpec、TriForce 做掉，風險高。「接受率 × 軟體路徑傳輸成本」的判準和 AMD 上的量測，風險中等。
【可行性】中，約 2–3 週。VeriCache 的程式碼是否公開，未查證。不依賴它也能做：
1. 先建解析模型：由接受率 α、每次 verify 的 descriptor 數與位元組數，推出有效 decode 延遲。
2. 再用 HF transformers 做小原型：INT4 KV 負責 draft、BF16 KV 負責 verify，實際量出 α。
不管做不做，這篇都要放進 related work，形式化也要承認多副本狀態。

來源：https://arxiv.org/abs/2605.17613、https://arxiv.org/abs/2502.10424

## F8. （信心 medium；票數 綜合 [0] 3-0、[8] 3-0、[1] 2-1）

橫切佐證（延伸故事 3，不是全新方向）：本輪三組獨立來源顯示，品質代價隨模型 × 任務改變的不只是量化，而是每一種有損動作：跨 adapter 借用 KV、近似重建 recurrent state、跨 fine-tune 借用 KV 都一樣。這支持把 ε 從單一常數改成依 (model, task, action) 查的校準表，並把 ε profiling 當成系統元件。

【證據】
- 跨 adapter（LRAgent，3-0）：同樣是 FullShared，在 LLaMA／Ministral × HotpotQA／ScienceQA 四格中掉 2.48–5.28 分。驗證者引用的 ForkKV：完整重用平均掉 5.40 分，APIGen 上最多掉 21.95 分。
- 近似 state（Tail-Replay，3-0）：r=5% 時的 RULER 保留率，OLMo-Hybrid-7B 是 93.1%，兩個 Qwen 模型是 99.8–99.9%。3 個模型中有 2 個，LongBench 最差的任務都是 qasper。驗證者提醒：只有 3 個模型、其中 2 個同族，沒有信賴區間，而且 OLMo 的 baseline 本身較弱，可能是干擾因素。
- 跨 fine-tune（DroidSpeak，2-1）：「可忽略」是逐一 profiling 每組模型對後，以 5% 品質下降為門檻定義的；§5.3 改用 1%。
- 我們自己的故事 3：Qwen2.5 族在 FP8／INT4 下崩潰，其他族到 129K 都沒問題（既有結果，這次沒有重新研究）。
【推翻的假設】ε 是某個動作（例如 INT4）的固定成本，用一個全域門檻就能處理。
【最接近前作】KVTuner（已讀，逐層精度敏感度）、DroidSpeak 的逐模型對 profiling。
【新穎性風險】中。逐動作校準本身不新；新在把不同物件類型（KV 精度、借用 KV、近似 state）的 ε 放進同一套量測協定和同一個放置決策。
【可行性】高。這是 m5_quality harness 的擴充，不需要新硬體。但故事 3 目前的 CI 很寬（GSM8K n=120，±10pp），擴充前要先加大樣本數，否則跨動作比較沒有統計檢定力。

來源：https://arxiv.org/abs/2602.01053、https://arxiv.org/abs/2608.30310、https://arxiv.org/abs/2411.02820、https://arxiv.org/abs/2604.06370

## 限制與注意事項（caveats）

(1) 「還沒人做」只相對於本輪讀過的論文。驗證者多次明講，gap 只存在於那一篇，不代表整個領域都沒有。hybrid 狀態快取在 2026-08-31 到 09-14 之間就出現了 DASC、Tail-Replay 和 SGLang PR #39436，撞車風險高。動手前要先用 novelty-check 查「hybrid／Mamba／GDN state offload」「state checkpoint CPU/SSD tiering」等關鍵字。

(2) 來源品質不一。
- 經同儕審查的只有 LRAgent（ICML'26 poster）和 DroidSpeak（NSDI'26）。
- Tail-Replay 標註為 NeurIPS'26 workshop 論文。
- Sparse Prefix Caching、DASC、MEPIC、MiniPIC、VeriCache、ForkKV、2609.17109 都是 preprint。
- SGLang #39436 是未合併、未審查、CI 失敗的 PR，只有一個設定、8 次重放。

(3) 以下被否決的 claim 不得當作證據：
- 2604.03143 的 multi-agent 冗餘與 Diff-Aware Storage（1-2）
- CacheTune 2605.24022 的兩條（0-3、1-2）
- KVShareArena 2609.10266 的三條（皆 0-3）
- LRAgent 的 base＋LR 分解細節（1-2）與 K/V 不對稱（0-3）
- DroidSpeak「平均只有 11% 關鍵層」（0-3）
- MEPIC 的 NoPE 機制細節與評測範圍（0-3）
- DASC 的 state／KV 敏感度不對稱（1-2）
- Tail-Replay 的機制細節（0-3）
- SGLang PR 的 dead-memory 解讀（1-2）
如果 CacheTune 真的用 κ 決定重算比，它會是故事 6 最接近的前作。否決只代表細節不可信，不代表論文不存在，必須親自讀原文。

(4) 標【判讀】和【算術】的內容是綜合者的推論，沒有經過三票驗證。「一份 checkpoint 等於 3,144 或約 5,384 個 token 的 KV」這個換算的依據與前提：
- 依據：已驗證的 per-rank 預算乘以 8，加上本次用 curl 取得的 HF config。
- 前提：SGLang 預設的 dtype；KV 以 BF16 計（Kimi-Linear 用 MLA latent）；以整個模型的唯一位元組計算，不計 TP 下的 KV head 複本。
這不是量測。

(5) 文中的數字都是別人在別的硬體上量的：H100、Hopper TP8、A100＋InfiniBand、RTX PRO 6000 Blackwell、Ascend 910B、RTX 2080 Super。依 CLAUDE.md 規則 1，這些數字只能當我們實驗的假設，不能寫進 results/，也不能當成我們的量測。可行性的週數是規劃估計，也不是量測。

(6) 覆蓋缺口。以下兩個方向本輪沒有 claim 通過驗證，所以本報告沒有從這裡提方向。這是證據缺口，不代表那裡沒有機會。
- angle B：P/D 分離、RDMA／CXL／NVLink-C2C、GDS、CacheGen 類壓縮。
- angle D 的經濟與能耗面：cached-token 定價、每租戶精度、碳排。
另外，angle A 的 ε 全部是在 1.0–1.5K token 的短 context 上量的。

(7) 時間敏感。
- vLLM 和 SGLang 對 hybrid 模型的快取行為每一版都在變；vLLM 從 v0.11.1 到 v0.15.0 就改過預設。
- PR #39436 也可能被改寫或關閉。
- ROCm 和 vLLM 0.28 對這些 hybrid 模型的支援都還沒查證。
- 一個月的時程和 MLSys 2027 截止日（2026-10-30）重疊，要先決定投哪裡。

## 還沒回答的問題

1. 我們的 vLLM 0.28 OffloadingConnector 遇到 hybrid 模型（Qwen3.5-4B、OLMo-Hybrid-7B）時，會不會把 recurrent state 和 KV 一起卸載？如果只卸載 KV，host 命中會不會像 SGLang PR #39436 那樣，在 state 缺席時歸零？這是方向 8–10 的第一個 go/no-go，在 3090 上幾天內就能回答。
2. 故事 1（量化取代放置）能不能移植到 state？把 FP32 checkpoint 降到 BF16 或 FP8 時，ε 是多少？這個 ε 是否也隨模型 × 任務而變？反過來說，在 agent 閉迴路裡，INT4 KV 的微小 ε 會不會拉長軌跡，把 4 倍容量吃回去（方向 12）？
3. angle B 和 angle D 這一輪沒有任何 claim 通過驗證。P/D 分離與遠端傳輸的路徑（NIXL、Mooncake Transfer Engine、LMCache P2P）上，有沒有人量過每筆 descriptor 的固定軟體成本？如果沒有，故事 2 可以直接延伸到 P/D 分離與 CXL 階，值得另開一輪專門查。
4. 跨 adapter 借用 KV 的 ε，在 32K–128K context 下是多少？和 INT4 疊在一起時，兩者是相加還是互相放大？LRAgent 量的 context 只有 1.0–1.5K token。

## 被否決的 claim（不可引用）

- **1-2**，https://arxiv.org/abs/2604.03143：Multi-agent All-Gather 回合會造成大量跨 agent 的 KV 冗餘。8-agent GenerativeAgents 回合中，pairwise block similarity 為 91%–97%。在 Qwen2.5-14B、單張 A100 80GB 上，multi-agent workload 的 KV 佔 41.5 GiB（pool 的 99.3%），相同條件下的獨立請求只佔 24.8 GiB（59.2%）。這些共享輸出區塊在各 agent prompt 裡不是共同 prefix，要靠 RoPE re-rotation 對齊位置，所以 prefix-only 命中語意沒辦法去重。→ 對本研究：這是 agentic workload 上的直接量化證據，支持 non-prefix／段級重用（我方 finding 4）。

- **1-2**，https://arxiv.org/abs/2604.03143：Diff-Aware Storage 只保留一份 dense master copy。其餘 sibling agent 的 KV 存成 block-sparse K/V diff（block index 加上 block 級修正值），在 layerwise 傳輸路徑上做 fused restore，不會實體化 dense 副本。在代表性 workload 上壓縮率為 11–17×，per-agent KV 最多降 17.5×。作者自己承認壓縮率取決於 workload 結構：各 agent 收到的共享輸出子集差異越大，correction 就越大，收益也隨之遞減。

- **0-3**，https://arxiv.org/abs/2605.24022：CacheTune 用逐層成本模型決定非前綴（non-prefix）KV 重用時「載入多少、重算多少」：T(r)=max(rN·tc,(1−r)N·ti)+to，解析最佳點 r0=ti/(tc+ti)，等同 r0=1/(1+κ)，其中 κ=tc/ti 是重算對傳輸的成本比。之後再用 10 筆 SAMSum 樣本做 golden-section search 微調，每個硬體或儲存設定約需 3–4 分鐘。在「用 κ 驅動 DROP+重算決策、並隨硬體調整」這件事上，這是目前最接近的先前工作，對 Tiara 的 κ 主張構成 novelty risk。與 Tiara 的差異有三：傳輸成本只是每 token 線性項加每層固定開銷 to，沒有 per-descriptor 的軟體開銷項；決策只是單一比例 r，不是在多個 tier 之間放置；依擷取到的全文，沒有精度維度（BF16/FP8/INT4）。

- **1-2**，https://arxiv.org/abs/2605.24022：最佳重算比取決於 KV 放在哪一層儲存，而且儲存越慢，重算比應越高。固定重算比 15% 時，HDD 與 SSD 相對 Full Recompute 的 TTFT 加速平均只有 1.92× 與 2.05×。改用硬體感知校準後，HDD 選出 36.4%、SSD 選出 30.9%，加速提升到 2.36× 與 2.34×。論文明言慢速儲存上「不應盲目最大化 KV 重用」，這和 Tiara 動作空間的 DROP+重算、以及「load front, recompute tail」的分界直接重疊。

- **1-2**，https://arxiv.org/abs/2602.01053：LRAgent（ICML 2026 poster）把 multi-LoRA agent 的 KV cache 拆成兩部分：一份所有 agent 共用的 base cache（由 pretrained W0 產生），加上每個 agent 各一份 low-rank LR cache（X·A_i，rank r=8，只套在 Q/V projection）。理論上總 KV 大小降為 non-shared 的 1/N + r/d_out ≈ 1/N。N=3 agents、Ministral-8B、66.4k tokens 時實測約為 Non-Shared 的 1/3，與 FullShared 相差 <1 GB。附錄 Table 22（總記憶體，含 14.95 GB 權重）在 66.4k 時：Non-Shared 39.84 GB，BaseLRShared 23.74 GB。【我方觀察】同表 132.0k 列的 Non-Shared 是 64.34 GB，超過 48 GB A6000 的容量，所以這張表至少有一部分是推估值，不是實測。對分層 KV 的意涵：要放置的物件從「N 份完整 KV」變成「1 份共享 base + N 份極小的 low-rank delta」。

- **0-3**，https://arxiv.org/abs/2602.01053：同一 context 下，不同 LoRA agent 之間的 KV 差異主要來自 adapter output。Table 1 用 128 個 2k-token HotpotQA 樣本、3 組 agent pair 量測：base cache 的跨 agent 平均 cosine similarity 是 0.9726（LLaMA-3.1-8B）/ 0.9530（Ministral-8B），adapter output 只有 0.0538 / 0.0225，接近正交。key cache 的平均相似度是 0.9922 / 0.9840，所以作者讓所有 agent 直接共用整個 key cache，只有 value cache 需要各 agent 分開處理。這代表 K 和 V 的可共享性不對稱，可以作為 K/V 分開決定 tier 或精度的依據。

- **0-3**，https://arxiv.org/abs/2411.02820：Across LLMs that share an architecture but have different fine-tuned weights, reusing another model's entire KV cache causes a large accuracy loss. On average only 11% of layers are 'critical'. DroidSpeak recomputes just those layers and reuses the other model's KV for the rest, with negligible quality loss. This is the closest prior work for cross-fine-tune / cross-agent KV reuse.

- **0-3**，https://arxiv.org/abs/2512.16822：MEPIC 以 NoPE（positional-encoding-free）格式存 chunk KV，materialize 時不套 RoPE，改在 fused RoPE-attention kernel 裡即時套上旋轉偏移。已快取的 chunk 只重算第一個 KV block，其餘 block 當作 canonical KV，可跨位置、請求與批次共享。因此非前綴命中的重算成本從 token 級、與位置成線性，變成每個 chunk 固定一個 block。這對「load front, recompute tail」只適用前綴命中的前提構成直接反例，也是 segment／非前綴命中方向最接近的前作。

- **0-3**，https://arxiv.org/abs/2512.16822：評估範圍很窄：只有一個模型（Mistral-7B-Instruct-v0.3），在 Ascend 910B NPU（64 GB HBM）上跑，用四個 QA 資料集（SQuAD、NewsQA、NarrativeQA、emrQA）各 300 個請求，平均請求長度只有 1,435 到 2,224 tokens。系統雖然透過 LMCache 做 CPU／磁碟持久層（lazy LRU donation），但從 HTML 全文萃取時沒找到 CPU／磁碟層的載入延遲量測，也沒找到 32K 以上長上下文或 NVIDIA/AMD GPU 的結果。所以「NoPE canonical chunk KV 在多層儲存與長上下文下的成本」這塊還是空的。

- **0-3**，https://arxiv.org/abs/2609.10266：KVShareArena（Xi Shi & Qian Lou, University of Central Florida；arXiv 2609.10266 v1, 2026-09-09, cs.CL, 未經同儕審查）指出，現有 KV-cache benchmark 只測 exact-prefix reuse。它提出一個 benchmark，測 non-prefix reuse 與跨 model checkpoint reuse。non-prefix 情境有兩種：RAG 每個 query 重新組合的 retrieved chunks，以及 multi-agent coordinator 讀取其他 agent 寫出的 reports。品質指標是 PGR = (S_method − S_floor)/(S_ceiling − S_floor)，其中 floor = 不用 cache，ceiling = full recomputation。成本分三軸：compute（PrefillWorkSaved = 1−R/D）、memory（KVBytesSaved）、cache 已在手上時的 per-request TTFT。一次性建 cache 的成本另外報告。harness 以 pip 套件 kvsharearena 發佈，並附 public leaderboard。對本研究的意義：它是 finding (4)（segment/non-prefix hit）最接近的 prior work，也可直接拿來當品質評測框架。

- **0-3**，https://arxiv.org/abs/2609.10266：不需重算的 position correction（free position alignment）在只依賴單一來源的問題上就夠用。一旦問題要同時結合多個來源（multi-hop QA），只有付出代價的方法能收回約 1/2 到 2/3 的 gap。代價有兩種：一是部分 re-encode，例如 CacheBlend 實測重算 17%，把 multi-hop 拉回 .39–.46 的 gap；二是訓練，例如 KVPacket。沒修復的 cache 可能比完全不用 cache 還差：在 agent reports 軌上，naive assembly 的 PGR 是 −.82。對本研究的意義：non-prefix hit 不是免費命中，放置與命中決策必須把 repair 重算量算進預算。這推翻了「hit = 省下整段 prefill」的假設，也表示 DROP+recompute 與 load 應該能在同一個 segment 上部分共存。

- **0-3**，https://arxiv.org/abs/2609.10266：在記憶體軸上，只有 compression 類方法能省位元組，而且要付出品質。在實際 tensor 上量，4-bit quantization 只省 71.9%，不是名目的 75%，因為 scale factors 也要存。所有修復品質的方法都保留完整 cache，KVPacket 與 MiniPIC 還額外多用約 3%。這些 compression 方法在單一 prompt 上近乎無害，但在新寫出的 agent reports 上，每一列 compression 都顯著低於 free position alignment，排在最前面的則是以重算為基礎的 repair。benchmark 的 compression 家族包含 SnapKV、TOVA、StreamingLLM、Knorm 與 4-bit quantization；是否每一種都在 agent reports 軌上評測過，沒有逐一核對。對本研究的意義：這與 finding (1)「INT4 取代 placement」有張力——在 cross-agent / non-prefix reuse 下，INT4 可能不再免費，precision tier 與 reuse 模式之間有交互作用。71.9% 這個數字也可以用來校正容量算術。

- **1-2**，https://github.com/sgl-project/sglang/pull/39436：SGLang HiCache 對 hybrid（attention + Mamba/KDA）模型，每個 prefill chunk 與每個完成請求各存一個 Mamba state checkpoint，作為 host KV 的還原錨點。預設依 device 端 bytes 比例把 `--hicache-size` 切給 KV 與 Mamba host pool；device Mamba pool 只按執行中的請求配置，所以很小，導致 host checkpoint slot 遠少於 KV tier 所需的錨點。某 prefix 的最後一個 checkpoint 被逐出後，它在 host 上的 KV rows 就無法還原，成為 dead memory。意涵：hybrid 模型的 KV 可重用性受另一個 pool（recurrent-state checkpoint）是否存活所約束，只看單一 pool 的容量規劃或 LRU 會系統性失效。

- **1-2**，https://arxiv.org/abs/2608.30386：state 與 KV 的品質敏感度很不對稱。在 RULER NIAH-S1 上，把 full-attention KV 歸零後準確率變成 0；把 recurrent-state checkpoint 歸零，準確率最多只差 1 點（GDN 0.990、KDA 1.000）。不過這只是單一 needle 的檢索。在較難的任務上，zero-fill（dasc-nr）隨 W_max 增大會掉分，例如 MMLU-Pro 從 66.94% 掉到 60.17%。「只丟部分 state、再從有界 suffix 重算」的 dasc-wr 則維持在 66.59–67.19%（Dense 為 67.50%）。W_max=128 時，它用 +23.0 ms 的 TTFT 換回 4.43 點。附錄 Table 8 給出每次 hit 的攤提重算成本：重放 16 token 約 67.0 ms，重放 841 token 約 426.0 ms。這個數字包含 concurrency 96 下的排隊。這些結果支持兩件事：state 和 KV 應各有不同的品質預算 ε，而且「部分丟棄＋有界重算」可以成為動作空間裡的新動作。

- **0-3**，https://arxiv.org/abs/2608.30310：Tail-Replay（TeleAI／上海交大，ML for Systems @ NeurIPS 2026 workshop 論文）處理 hybrid 模型（full-attention 層與 Gated DeltaNet linear-attention 層交錯）的 prefix cache 時完全不存 recurrent-state checkpoint。它只保留精確的 FA KV。命中 m 個 token 時，它把 linear-attention 狀態設為零，再重播 matched prefix 最後 k=⌈r·m⌉ 個 token，近似重建該狀態。因此重用邊界由共享 token 決定，不必對齊 checkpoint 位置（對照 Marconi 與 Sparse Prefix Caching）。對方向 C 的意涵：hybrid 模型的 recurrent state 多了一個「DROP 加上 replay 預算 r 可調的有損近似重算」動作，選項不再只有「存或不存精確 checkpoint」。
