# ICPE 摘要第 4 版：直覺、只留關鍵數字、最後有結論（草稿；icpe-adv，2026-11-08）

> 使用者 11/8 對第 3 版的意見：「寫不夠具體，最後也沒寫結論」；對帶很多數字的版本：「我不認爲要放很多數字……摘要必須要直覺……如果要數字就只放最關鍵的」。本版取代 `ABSTRACT_v3.md`。

## 0　故事：這一版怎麼改

- **每一句講清楚「發生什麼、為什麼」**，用白話，不用內部名詞（變體數、門檻、元件名都拿掉）。
- **數字只留一組對比**：一般 agent 服務「不到 1%」對工具類 agent「最多 62%」。這就是整個故事。
- **最後一句直接回答研究問題**：還原變便宜之後，怎麼還原、放在哪用固定規則就夠；值得蓋預測器的，主要是記憶體不夠時決定留下哪些對話。
- 細的數字（11.0–40.7%、3／9、14／14、OpenHands 0.2%→56%……）放到前言與評估。帶數字的長版放在第 4 節，給前言第 1 段用。

## 1　英文版（259 字，`wc -w`）

> LLM agents and multi-turn chat pause between requests, so serving systems move a session's KV cache out of GPU memory and must bring it back before the next first token. Recent agent-serving systems keep evicted KV in host memory and predict which sessions will return, and when, to decide what to evict and what to fetch early. But once bringing KV back is cheap, how much can knowing the future still save, and in which decision? We measure KV restoration on an NVIDIA H200 and replay production and public agent and chat traces in simulation, comparing strong rules that do not predict (LRU, time-to-live, prefix caching) with versions of the same rules that know exactly when each session returns. Restoration needs no prediction: one fixed choice per storage tier is already fastest, and deciding placement in advance gains nothing, because what it would use is still known at restore time. Knowing return times barely helps an ordinary agent-serving workload (under 1% lower time to first token) but helps tool-calling agents a lot (up to **62%【早上填】**). The gain appears only when the sessions about to return no longer fit in GPU and CPU memory, so that keeping the wrong one forces a slow reload; it comes mostly from choosing which sessions stay. On the trace that records tool calls, a simple tool-call signal does not capture it. Our conclusion: once restoration is cheap, how to restore and where to place KV need only fixed rules; prediction is worth building mainly for deciding which sessions to keep when memory runs out.

## 2　中文對照

> LLM agent 和多輪對話的請求之間會停頓，服務系統因此把對話的 KV cache 移出 GPU，下一個字出來之前要先搬回來。近來的 agent 服務系統把逐出的 KV 留在主機記憶體，並預測哪些對話、什麼時候會回來，用來決定該淘汰誰、該先搬誰。但如果搬回來本身已經很便宜，知道未來還能省多少？省在哪個決策上？我們在 NVIDIA H200 上實測 KV 的還原，並用正式服務與公開的 agent、聊天紀錄做模擬，把不做預測的強規則（LRU、存活時間 TTL、前綴快取）和「同一條規則、但知道每個對話何時回來」的版本相比。還原不需要預測：每一層儲存用一個固定的做法就已經最快；事先決定放哪也沒有收益，因為它要用的資訊到還原時一樣看得到。知道回來時間，對一般的 agent 服務幾乎沒幫助（首字延遲少不到 1%），對會呼叫工具的 agent 卻幫助很大（最多少 62%）。差別在於：只有當快要回來的對話已經放不進 GPU 和 CPU 記憶體時，留錯了才要再慢慢搬一次，知道未來才有用；省下的主要來自決定誰留下。在記了工具呼叫的那份紀錄上，「有沒有呼叫工具」這種簡單線索拿不到這些收益。結論：還原變便宜之後，怎麼還原、放在哪用固定規則就夠；值得蓋預測器的，主要是記憶體不夠時決定留下哪些對話。

## 3　每一句站得住嗎

| 句子 | 種類 | 依據 |
|:--|:--|:--|
| 前作把 KV 留在主機記憶體並預測 | 前作 | PBKV §6.1（HiCache）、KVFlow §3.2（老師第 4 點） |
| 每層一個固定做法就最快 | 〔實測〕 | 慢層每一格 INT4＋Cake 最快（比次佳快 11.0–40.7%）；CPU 層要看解壓寫法（固定表分兩種實作）。「每層一個」是簡化 |
| 事先決定放哪沒有收益 | 〔實測〕 | 16 組比較 0 組贏 |
| 「因為它要用的資訊到還原時一樣看得到」 | 〔判讀〕 | 舊稿的解釋（位置在還原時看得到），沒有單獨驗證 |
| 不到 1%／最多 62% | 〔模擬〕 | FreeInference 6 週最大 1.0%；工具類最大 62.4%（Mooncake toolagent R2）。**62% 早上填** |
| 「只有當快要回來的對話放不進記憶體時」 | 〔判讀〕 | 支持：UltraChat／LMSYS GPU 不用逐出時剛好 0；OpenHands 並行 16→32 從 0.2% 到 55.8%；CPU 容量中間最大（`ADVISOR.md:82`）。也解釋 FreeInference：CPU 很小時也放不下全部對話，但回來的集中在少數熱門對話，LRU 已命中（舊稿 99.29%），所以仍 <1%。**特意寫「快要回來的對話」而不是「全部對話」** |
| 主要來自決定誰留下 | 〔模擬〕 | 淘汰類佔 ≥15% 的（格, 元件）51.8%（原始工具）、94.5%（合成）；版本 B 修正版。**版本 C 修正版早上確認**；若原始到達 <50%，改「much of it」 |
| 簡單線索拿不到 | 〔模擬〕 | 只在 TraceLab 測過（`SIGNALS_BY_TRACE.md`），所以寫「記了工具呼叫的那份紀錄」 |
| 結論 | 〔判讀〕 | 刻意不寫「其他都用固定規則」：Mooncake 上預取也有 ≥15% 的格子 |

## 4　附錄：帶數字的長版（給前言第 1 段，不是摘要；283 字）

> LLM agents and multi-turn chat pause between requests, so serving systems move a session's KV cache out of GPU memory and must restore it before the next first token. Recent agent-serving systems keep evicted KV in host memory and predict which sessions return, and when, to drive eviction and prefetching. Yet once restoration itself is cheap, it is unclear which of these decisions still need prediction. On an NVIDIA H200 (Llama-3.1-8B and Qwen2.5-7B, 32K context), we measure how parked KV is restored. In trace-driven simulation of 28 variants of six production and public traces, we compare the per-setting best of strong non-predictive rules (ten CPU-tier rules; LRU, TTLs from 0.3 to 300 s, and block-level prefix caching at the GPU tier) with clairvoyant versions that know every return time, under a 15% threshold preregistered before any result. Restoration needs no prediction: a fixed table suffices, INT4 with Cake is fastest on slow tiers (11.0-40.7% over the runner-up), and placing KV at write time wins none of 16 comparisons. On a production agent-serving trace, knowing return times lowers time to first token (TTFT) by under 1%; on tool-calling traces, by up to 62%, passing 15% in 3 of 9 real-arrival and all 14 synthetic-concurrency variants, mostly by choosing which sessions stay in GPU and CPU memory. The gain follows pressure: on OpenHands trajectories it rises from 0.2% at 16 concurrent sessions to 56% at 32, and overall it peaks at mid-sized CPU memory. On the trace that records tool calls, a tool-call cue captures none of the gap. In short, once restoration is cheap, prediction pays mainly for eviction, and only where memory is tight; restoration and write-time placement need only a fixed policy.
