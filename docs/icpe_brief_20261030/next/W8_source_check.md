# W8：對手論文的數字回原文核對

> **性質**：核對紀錄（NEXT_PLAN W8）。**此次核對為 self-review，非 cross-model**（CLAUDE.md §5）。
> **為什麼**：11/4 讀 8 篇時用的抓取工具，後半段是模型摘要；寫進論文或登記之前要對原文逐字核對。
> **怎麼查**：用 `curl` 直接下載 arXiv HTML 全文（不經模型摘要），以標準庫 `html.parser` 轉成純文字（`python -I`，只讀），再用 `grep` 找原句。下載的檔案視為不受信任，放在 session 的 scratchpad，不進 git（版權、大小）；這裡只記 URL、版本、sha256、原句。
> **下載時間**：2026-11-04T22:56:29+08:00。

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

## 2. NEXT_PLAN W8 點名的四項（待核）

| 計畫裡寫的 | 狀態 |
|:--|:--|
| CacheWise 開 CPU offload 後約 1.19 倍 | 待核（PREREG 之後做） |
| KVFlow 在 PEER 上約 1.08 倍 | 待核 |
| CacheScout 預取最多 +1 個百分點命中率 | 待核 |
| CacheWise 摘要的「22–2.6×」是排版錯誤（應為 2–2.6×） | 待核 |

## 3. 其他 NEXT_PLAN §5.9 的對手數字（待核）

SAGA 25 GB/s 與 Poisson λ≈8 tasks/min；KVFlow 64 GB/s（H100）、2 GB/s（A10G）；Cake 的 7／25／32／56／100 Gbps；Pensieve 思考時間平均 60 s（掃到 600 s）；CacheScout 到達速率 0.2–50 sessions/s。
