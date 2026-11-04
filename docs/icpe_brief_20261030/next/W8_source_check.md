# W8：對手論文的數字回原文核對

> **性質**：核對紀錄（NEXT_PLAN W8）。**此次核對為 self-review，非 cross-model**（CLAUDE.md §5）。
> **為什麼**：11/4 讀 8 篇時用的抓取工具，後半段是模型摘要；寫進論文或登記之前要對原文逐字核對。
> **怎麼查**：用 `curl` 直接下載 arXiv HTML 全文（不經模型摘要），以標準庫 `html.parser` 轉成純文字（`python -I`，只讀），再用 `grep` 找原句。下載的檔案視為不受信任，放在 session 的 scratchpad，不進 git（版權、大小）；這裡只記 URL、版本、sha256、原句。
> **下載時間**：2026-11-04T22:56:29+08:00（§0 表的前七篇）。章節號是用轉出的純文字往前找最近的標題核過的；HTML 沒有頁碼，計畫裡寫的頁碼都**沒有核**。

## 0. 來源檔

| 論文 | URL | 版本（HTML 內的 arXiv id） | sha256（HTML） |
|:--|:--|:--|:--|
| PBKV | https://arxiv.org/html/2605.06472 | 2605.06472v1 | `8d7796ca3409ebd939aea83f50ce5204e0fdff463de9ea0aa32126626e001492` |
| CacheScout | https://arxiv.org/html/2608.14624 | 2608.14624v1 | `df2332819a3ec36c5b168eb08b5f2635a04b6ad28751bbf1f4a8e32e041b65ad` |
| When Fancy Eviction Fails | https://arxiv.org/html/2609.28870 | 2609.28870v2 | `88cacb51e5dd4a919e600d2abc1f3339e190c1a94c63e243106267c0fa41bc52` |
| Continuum | https://arxiv.org/html/2511.02230 | 2511.02230v7 | `c3ac6d3b182765f79863dc8b232ff5df0d89e59a1f20cd41ae548ec3ebdefd3e` |
| SAGA | https://arxiv.org/html/2605.00528 | 2605.00528v2 | `86105b6e070adbff2d9e4d938655ba669cf3c2a4d47435f8cdf03cc14e2b4e17` |
| CacheWise | https://arxiv.org/html/2606.16824 | 2606.16824v1 | `6d7f0e261b8b918346ab4c55f0d0c19cb074062575398cc0c147a38e8c17173c` |
| KVFlow | https://arxiv.org/html/2507.07400 | 2507.07400v1 | `9d0603809be462d89146ea449220227152d60d222cc4c6e7e8ebe9d7a30d3577` |
| CacheWise（摘要頁） | https://arxiv.org/abs/2606.16824 | 2606.16824v1 | `cc3970b90ac43e7bdf6ade44097f9630701894bfc9b334b91fc04bce0ebb456f` |
| Cake | https://arxiv.org/html/2410.03065 | 2410.03065v2 | `d06d29822cf64e8767668cd6c55f803190c25cf76c3b8b34a8aa40885d003115` |
| Pensieve | https://arxiv.org/html/2312.05516 | 2312.05516v3 | `6dd60a612bf1f09dae404a0bc0a919bc7675515bf2ad95289c0b0e914df01589` |
| CachedAttention | https://arxiv.org/html/2403.19708 | 2403.19708v3 | `aca60d242b8f1b50df48391c56dd5f5a6da80e7da49257bdc311c6978c3f2d9a` |
| InferCept | https://arxiv.org/html/2402.01869 | 2402.01869v2 | `6be291d2b5f0092609594277b26ecd3eeda0dd8e5d5bfcfa9a0df1cd3f92053a` |
| Predictive Multi-Tier | https://arxiv.org/html/2604.26968 | 2604.26968v2 | `2b6d272560c9554d1f8d85cd49bf0f277c4678bb497e194ef66aa310c2153774` |
| Ask the Tool, Don't Guess | https://arxiv.org/html/2609.18849 | 2609.18849v1 | `b00257bf3d208356e979b0b350a1bf03d5744cd4f8fc95f5b9222a92b9767b32` |

後六篇與 CacheWise 摘要頁的下載時間：2026-11-04T23:26:24+08:00；標題用摘要頁的 `citation_title` 對過（arXiv id 沒有記錯）。

## 1. 影響 PREREG §6b 的數字（先核，2026-11-04）

| 計畫裡寫的 | 原文（逐字） | 出處 | 判定 |
|:--|:--|:--|:--|
| CacheScout 的 TTL 用 0.3 s | 「Continuum represents cross-session KV reuse through TTL-based pinning around tool-call boundaries. Following the observed inter-turn latency in AutoGen, we set the TTL to 0.3 s.」 | CacheScout §5.1 Experimental Setup | **對**；但 0.3 s 是照 AutoGen 自己的兩輪間隔定的（只對那個工作負載），而且是 CacheScout 替 Continuum 這個對手設的值，不是 Continuum 自己的設定 |
| Continuum 用 TTL 釘住（「固定 TTL」） | 「τ\*=argmax_τ P(τ,f)×Benefit(r)−Cost(τ,r)」；「we estimate P(τ,f) using the empirical CDF derived from historical tool-call records S[f]」；冷啟動時「a fixed TTL threshold T_default, derived from the same cost model by assuming that the tool-call duration follows an exponential distribution with unit mean」 | Continuum §4.2 Setting the TTL Value | **部分對**：Continuum 的 TTL 是依歷史工具時間逐次算的，只有冷啟動（及消融「Static TTL」，Fig. 16）用固定值。PREREG 的固定 TTL 網格是「不預測」的簡化版，論文要這樣寫 |
| Continuum GPU 滿時怎麼辦 | 「we iteratively selects victims from pinned_requests with the latest program arrival time to unpin and free the space until the first request can be scheduled to run」 | Continuum §5.2 Efficient Pin with TTL in Scheduler | 對；PREREG 第 3 項照用 |
| SAGA 的 TTL | 「We maintain per-tool-type latency distributions using exponential moving averages and set TTL to the p-th percentile of expected duration, where p is configurable (default 95%). Under memory pressure, TTL is scaled down proportionally.」；Algorithm 1：`TTL_max=300s` | SAGA §4.2 Tool-Call-Aware TTL | 對；300 s 當 PREREG TTL 網格的上端 |
| PBKV 雜訊 λ＝0／10／20／30%，「回來時間乘上誤差」 | 「P̃_w^(k)=(1−λ)P_w^(k)+λU」，「U denote the uniform distribution over agents」，「λ=1 degenerates to a uniform (i.e., random) prediction. The perturbation is applied to the predictor outputs consumed by both eviction and prefetching」；文中提到 λ＝0、10%、20%、30%、40%；「Error bars denote standard deviation over 10 trials」 | PBKV 附錄 E.1 Motivation and Setup（定義）、E.3 Sweeping the Prediction Noise（結果） | **計畫寫錯**：PBKV 的雜訊是「預測的下一個 agent 分佈」和均勻分佈的**混合**，不是時間的乘法誤差；等級到 **40%**；**10 次**。PREREG 第 6 項改成兩種模型都跑，只有混合式能寫「照 PBKV」 |
| PBKV 並行 72／24／48 | 「we set the concurrency limit to 72/24/48, respectively」；「Following prior work [9, 24], we set the number of concurrent workflows to induce memory pressure」 | PBKV §6.1 Experimental Settings、Table 1 說明 | 對（HoVer＋LangChain／SWE-bench＋AutoGen／FinanceBench＋CrewAI） |
| PBKV PCIe 20 GB/s | 「8× NVIDIA A6000 (48 GB) GPUs interconnected via NVLink, 128 virtual CPU cores, 512 GB of memory, and 20 GB/s PCIe bandwidth」 | PBKV §6.1 | 對 |
| Fancy Eviction：SemiAnalysis 並行固定 16 | 「AgentX replays 393 Claude Code sessions from the SemiAnalysis corpus [53]. Because the corpus lacks absolute timestamps, we artificially schedule sessions to maintain a steady concurrency of 16, while preserving the exact timing between requests within each session.」 | Fancy Eviction §4.6 Findings Generalize Across Workloads | 對；另有附錄 B 在合成多輪紀錄上掃並行（HBM 8–32、pool 到 1,024），FreeInference 本身「averaging about 12 concurrent sessions」 |

## 2. NEXT_PLAN W8 點名的四項（2026-11-04 核完）

| 計畫裡寫的 | 原文（逐字） | 出處 | 判定 |
|:--|:--|:--|:--|
| CacheWise 開 CPU offload 後約 1.19 倍 | 「In this configuration, evicted blocks are transferred from GPU memory to CPU DRAM and swapped back on demand when the corresponding session resumes … Figure 18 shows the KVCache offloading latencies and session completion time for N=30 concurrent sessions. CacheWise achieves ~1.19x lower session completion times than vLLM. The absolute gains under offloading are smaller than under recomputation.」 | CacheWise §6.4 KVCache Movement、Fig. 18 | **對**；條件要寫全：N＝30 個並行對話、對 vLLM、指標是對話完成時間（不是 TTFT） |
| KVFlow 在 PEER 上約 1.08 倍 | 「KVFlow achieves clear improvements over both SGLang and HiCache, resulting in up to 1.12× and 1.08× speedups.」（工作負載：「we simulate agentic workflows based on the PEER [5] framework … We use the Financial QA dataset from PEER」） | KVFlow §4.2 的「Realistic Workflow Simulation」段、Fig. 8 | **對**；1.08× 是對 **SGLang＋HiCache**（GPU＋CPU 階層），對純 SGLang 是 1.12×；兩者都是「up to」 |
| CacheScout 預取最多 +1 個百分點命中率 | 「Enabling predictive eviction alone increases KV-cache hit rate by 18–22 percentage points over vanilla vLLM, whereas background prefetch alone adds at most one percentage point.」；同節：「Combined with predictive eviction, however, … the complete runtime reduces per-turn latency by a further 28% over eviction alone on GAIA (251 vs. 347 ms)」 | CacheScout §5.3 Design Breakdown、Fig. 12 | **對，但要加條件**：「最多 +1 個百分點」是**只開預取**（沒有預測淘汰）時；和預測淘汰一起開時，GAIA 上每輪延遲再降 28% |
| CacheWise 摘要的「22–2.6×」是排版錯誤（應為 2–2.6×） | arXiv 摘要頁（作者提交的純文字）：「CacheWise reduces KVCache evictions by up to 2-2.6x and improves total agent session completion time by up to 3.5x.」HTML 全文的摘要是數學元件 `$2$–$2.6\times$`，顯示成「2–2.6×」 | CacheWise 摘要（abs 頁與 HTML）、§1、§7 | **計畫寫錯**：原文就是 **2–2.6×**，**不是作者的排版錯誤**。「22」是 11/4 抓取工具把數學元件的顯示文字「2」和它的 LaTeX 原始碼「2」接在一起造成的（HTML 原始碼裡兩者並排）。**論文不能寫 CacheWise 有排版錯誤** |

## 3. NEXT_PLAN §5.4／§5.9 的其他對手數字（2026-11-04 核完）

| 計畫裡寫的 | 原文（逐字） | 出處 | 判定 |
|:--|:--|:--|:--|
| SAGA：Poisson λ≈8 tasks/min；作者說「task definitions rather than arrival traces」 | 「SWE-bench and WebArena are task definitions rather than arrival traces; we replay them under the same Poisson schedule (single-tenant, λ ≈ 8 tasks/min) for the § 9.2 end-to-end measurements.」；BurstGPT 衍生負載：「Tasks arrive per tenant as a Poisson process with approximate mean rates of 16 / 8 / 4 tasks/min/tenant」 | SAGA §9.1 Experimental Setup | 對 |
| SAGA：CPU↔GPU 25 GB/s | 「≈ 430 ms one-way over PCIe Gen4 × 16 at the 25 GB/s practical sustained bandwidth typical of A100 servers [7]」 | SAGA §9.1.2 CPU Swap as an Alternative Architecture | **部分對**：25 GB/s 是 SAGA 在「為什麼不用 CPU swap」的論證裡**假設的典型值**（引用文獻），不是它實驗平台量的、也不是它系統用的設定。§5.9 表裡要改寫成「SAGA 論證時假設 25 GB/s」 |
| SAGA：上下文 2–8K；70B | 「Each step: 2-4K prompt tokens」（SWE-bench）、「Each step: 4-8K prompt」（WebArena）；「Models: Llama-3-70B-Instruct」；8 節點 × 8 張 A100-80GB | SAGA §9.1 | 對（模型是 Llama-**3**-70B，不是 3.1） |
| KVFlow：64 GB/s（H100）、2 GB/s（A10G） | 「(1) Llama-3.1-8B on an NVIDIA A10G GPU with 24GB memory and 2GB/s PCIe Gen1 bandwidth; and (2) Qwen2.5-32B on an NVIDIA H100 GPU with 80GB memory and 64 GB/s PCIe Gen5 bandwidth」 | KVFlow §4.1 的「Models and testbeds」段 | 對 |
| CacheScout：到達速率 0.2–50 sessions/s；指標平均 TTFT | 「across the full arrival-rate sweep of Figure 11 (0.2–50 sessions/s)」；Metrics：「Time-to-first-token (TTFT)」（另報 P99 TTFT） | CacheScout §5.4、§5.1 | 對 |
| CacheScout：Llama-3.1-8B；大模型用 H200 | 「We use Llama-3.1-8B-Instruct as the main model」，平台「eight NVIDIA RTX PRO 6000 Blackwell GPUs」；「Qwen3-235B-A22B-FP8 … tensor parallelism across four H200 GPUs」 | CacheScout §5.1、§5.4 | 對（主實驗**不是** H200，是 RTX PRO 6000） |
| PBKV：主機層＝GPU 容量（HiCache 預設） | 「LRU on SGLang+HiCache (HICACHE_RATIO=1, allocating equal capacity on GPU and host memory), the default policy of SGLang」 | PBKV §6.1 | 對 |
| PBKV：Qwen3-14B／32B、A6000、512 GB | 「8× NVIDIA A6000 (48 GB) GPUs …, 128 virtual CPU cores, 512 GB of memory, and 20 GB/s PCIe bandwidth. We use Qwen3-14B and Qwen3-32B」 | PBKV §6.1 | 對 |
| CacheWise：Qwen2.5-Coder-32B、H200 | 「… H200 GPUs and AMD EPYC 9534 64-core CPU with Tensor Parallelism … (TP=2) enabled, with a combined GPU memory of 282GB」；「Qwen2.5-Coder-32B-Instruct」 | CacheWise §6.1 | 對（2 張 H200、TP＝2） |
| Fancy Eviction：Qwen3-Coder-30B、H200、16-token 塊、上限 256K | 「all evaluations use a 16-token block size and the Qwen3-Coder-30B [59] tokenizer, capping sequences at its 256k-token maximum context window (4.3% skipped in FreeInference Trace …)」；「measured on NVIDIA H200」 | Fancy Eviction §3.1 Traces and Simulator、§4.4 | 對 |
| Continuum：Llama-3.1-8B／70B；Poisson | 「We evaluate Continnum with Llama-3.1-8B, Llama-3.1-70B, and Gemma-3-12B. We use A100-SXM GPU from Runpod, H100 from AWS and Company A, and B200 GPU」；「using poisson distribution for the arrival pattern of agent programs」 | Continuum §6.1 Setup | 對 |
| Cake：5 個頻寬點，每點對應真實裝置；0–25 Gbps 只用在 Fig. 5 | Table 2：「Google Cloud standard egress limit 7 Gbps；Google Cloud tier-1 egress limit 25 Gbps；Lambda Lab SSD read 32 Gbps；Samsung 980 pro SSD read 56 Gbps；Inifiniband (RoCE) 100 Gbps」；Fig. 5：「I/O Bandwidth: 0-25Gbps」 | Cake §5.1 Experiment Setup、Table 2；§5.6 Fig. 5 | 對；**頁碼（計畫寫的「ICML p6」）HTML 看不到，沒有核** |
| Cake：重算≈載入是它最有利的條件 | 「the most favorable deployment scenario for Cake occurs when its speedup is approximately 2× compared to both compute-only and I/O-only approaches. This scenario arises when computation and I/O capabilities are well-balanced」 | Cake §5.2 | 對 |
| Cake：上下文 4K–16K；模型 | 「most dataset queries range between 4k and 16k tokens in length」；模型 LongAlpaca-7B／13B、LLaMA 3.1-8B／70B（Table 1） | Cake §5.1 | 對 |
| Pensieve：Poisson、思考時間指數分佈平均 60 s、掃到 600 s、上下文 16K | 「we simulate a request’s arrival time by sampling from a Poisson distribution under different request rates」；「we also simulate user think time … by sampling from an exponential distribution with varying mean」；「Our experiments so far use the average user think time of 60 seconds」；「we also show vLLM with 600-second think time as a comparison point」；「we limited the maximum context size to 16384」 | Pensieve §6.1 Experimental Setup、§6.7 Impact of User Think Time | 對 |
| CachedAttention：ShareGPT 9K、Poisson λ＝1 session/s | 「We set the number of different sessions arriving per second according to a Poisson distribution (with λ = 1.0). 9K conversation sessions are used in the experiments.」；另掃「from 0.5/s to 2.0/s」 | CachedAttention §4.1、§4.3.8 | 對 |
| InferCept：上下文上限 2K | 全文（HTML）**沒找到** 2K／2048 的上限句；模型是 GPT-J-6B、Vicuna-13B、Llama3-70B | InferCept §5 Evaluation Results | **NOT_VERIFIED**：來源要另查（可能來自它公開的 `merged_exp_uniform.json`，沒下載）；查到之前論文不寫「2K」 |
| Predictive Multi-Tier：NVMe 12 GB/s、評估是分析推算 | 「NVMe with GPUDirect Storage [28] (12 GB/s, …)」；章節「V-B Analytical Projection Methodology」；「Analytical projections combining validated component behavior with published hardware specifications」 | Predictive Multi-Tier §I Introduction、§V-B | 對（12 GB/s 是規格值，不是量的） |
| Ask the Tool：p90 TTFT；有 oracle 對照 | 「We include two more baselines: the engine’s own LRU, and an oracle that knows every exact return time.」；「Figure 9 shows the change in post-tool TTFT p90 and in throughput against LRU.」（H100 SXM） | Ask the Tool §5.1 Setup and baselines | 對 |

## 4. 對計畫的影響（要改的寫法）

1. **CacheWise「22–2.6×」不是排版錯誤**：NEXT_PLAN W8 的這一條要拿掉；論文照原文寫「2–2.6×」。
2. **CacheScout「預取 +1 個百分點」要寫條件**：只開預取時；和預測淘汰一起開時不是這樣。
3. **SAGA 的 25 GB/s 不是它的平台設定**：§5.9 的頻寬列要改寫。
4. **InferCept 的 2K 上限沒核到**：查到原文之前不寫。
5. PBKV 的雜訊不是時間的乘法誤差（§1，已反映在 PREREG 第 6 項）。
6. 其餘數字和原文一致。
