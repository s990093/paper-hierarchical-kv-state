# ICPE 摘要第 2 版（草稿；icpe-adv，2026-11-08）

> **已被 `ABSTRACT_v3.md` 取代**（2026-11-08 00:37，照同類論文的寫法改短；本檔留作紀錄）。

> 使用者 11/8：要投 ICPE；不再花時間在模擬；現在專心想摘要。以 `advisor_20261107/ADVISOR.md` 第 5 節的摘要為底，照老師 11/8 的意見改。
> 標 **【早上填】** 的數字，等 icpe-main 11/8 早上的版本 C 修正版總判定確認後才定（現在寫的是 ADVISOR.md 版本 C 的值，偏離 14 修正後的版本 B 判定字沒變）。
> 登記截止：11/10（一）19:59（台灣）；全文 11/17。

## 0　故事：摘要要講的一件事

- **舊摘要的主軸**：「預測能多省多少？」答案是「看工作負載」。
- **老師要的主軸**：「加了量化、多層、還原成本後，**哪一種決策**的價值變了、為什麼」，而且不寫成贏過前作。
- **新摘要因此改成三句話的故事**：
  1. 還原變便宜之後（INT4＋Cake），「怎麼還原」與「寫入時放哪」不用預測〔實測〕。
  2. 知道未來，在一般 agent 服務上幾乎沒用；在工具類 agent、記憶體吃緊時仍然有用，主要在「誰留在 GPU／CPU」〔模擬〕。
  3. 量法誠實：每條規則和它的預知版比，門檻事先登記；不比前作系統。

## 1　英文版（266 字）

> LLM agents and multi-turn chat leave gaps between requests, during which serving systems move a session's KV cache out of GPU memory and must restore it before the next first token. Recent agent-serving systems already keep evicted KV in host memory and predict which sessions return, and when, to drive eviction and prefetching. We ask which of these decisions still benefit from knowing the future once restoration is cheap. On an NVIDIA H200, we measure how to restore parked KV (load, recompute, Cake, INT4): a fixed lookup table suffices, with INT4 plus Cake fastest on slow tiers (11.0–40.7% over the runner-up), and deciding placement at write time gains nothing. In trace-driven simulation over 28 variants of six traces, we pair strong non-predictive rules (10 CPU-tier rules; LRU, TTLs and block-level prefix caching at the GPU tier) with clairvoyant versions that know every return time, under thresholds preregistered before any result. Knowing the future barely helps on a production agent-serving trace (under 1% lower time to first token, TTFT), but on tool-calling traces under memory pressure it cuts TTFT by up to **62%【早上填】**, exceeding 15% in **3 of 9【早上填】** real-arrival and **all 14【早上填】** synthetic-concurrency variants, mostly by deciding what stays in GPU and CPU memory. Public SWE-bench trajectories split the same way: up to **58%【早上填】** for OpenHands, under 11% for mini-SWE-agent. The tool-call flag present in these traces captures none of this gap; the traces carry no tool-progress signal. Cheap restoration thus removes the case for predicting how to restore or where to write, and leaves eviction under memory pressure as the decision where prediction still pays.

## 2　中文對照

> LLM agent 和多輪對話的請求之間常有空檔，服務系統會把對話的 KV cache 移出 GPU，下一個字出來之前要先搬回。近來的 agent 服務系統已經把逐出的 KV 留在主機記憶體，並預測哪些對話、什麼時候回來，用來決定淘汰與預取。我們問：還原變便宜之後，這些決策裡還有哪些需要知道未來？我們在 NVIDIA H200 上實測停放的 KV 怎麼還原（載入、重算、Cake、INT4）：一張固定的表就夠，慢層上 INT4 加 Cake 最快（比次佳快 11.0–40.7%），寫入時就決定放哪也沒有收益。在 6 份紀錄、28 個變體的紀錄驅動模擬裡，我們把一組強的不預測規則（10 個 CPU 層規則；GPU 層有 LRU、TTL 與 block 粒度前綴快取）和它們「知道每個對話何時回來」的預知版配對比較，門檻在看任何結果之前登記。知道未來，在真實的 agent 服務紀錄上幾乎沒用（首字延遲少不到 1%）；在記憶體吃緊的工具呼叫紀錄上，首字延遲最多少 62%，原始到達的 9 個變體裡 3 個、合成並行的 14 個全部超過 15%，主要靠決定誰留在 GPU 與 CPU 記憶體。公開的 SWE-bench 軌跡也一樣分兩邊：OpenHands 最多 58%，mini-SWE-agent 不到 11%。紀錄裡現有的「有沒有呼叫工具」旗標一點都拿不到；這些紀錄沒有工具進度的訊號。所以還原變便宜之後，「怎麼還原」「寫到哪」都不必預測，預測仍然有用的是記憶體吃緊時的淘汰。

## 3　每一處改動對應老師哪一點

| 改了什麼 | 舊（ADVISOR.md 第 5 節） | 新 | 老師的點 |
|:--|:--|:--|:--|
| 問題 | how much prediction can still save over **the strongest** non-predictive policy | **which decisions** still benefit from knowing the future once restoration is cheap | 5（哪種決策的價值變了）；不自稱「最強」 |
| 前作 | report large gains over LRU or prefix caching | **already keep evicted KV in host memory** and predict … to drive eviction and prefetching | 4（PBKV、KVFlow 都有 CPU 備份與預取） |
| oracle | a perfect oracle against the per-cell best | **clairvoyant versions** of the same rules | 2–3（它是「規則的預知版」，不是所有做法的上限；正文列出比對手慢的格數） |
| 結論 | Prediction pays only on some workloads | Knowing the future barely helps … but on tool-calling traces **under memory pressure** … | 1（兩邊並排、用記憶體壓力解釋） |
| 預取 | mostly by choosing what stays … **rather than by prefetching** | mostly by deciding what stays in GPU and CPU memory（**拿掉 rather than by prefetching**） | 2–3：完美預取是「用 LRU 騰空間」的版本，會低估預取；比較它和淘汰時不能下這麼強的句子 |
| 工具線索 | A free tool-call cue captures none of this gap | **The tool-call flag present in these traces** captures none; **the traces carry no tool-progress signal** | 8（Ask the Tool：工具其實知道進度，只是紀錄沒有） |
| 雜訊 | the gap survives ±30% error | **拿掉** | 偏離 14 的修正版雜訊結果還沒全部出；摘要不寫還在變的 |
| 收尾 | Measure the oracle against a strong baseline before building a predictor | Cheap restoration removes the case for predicting how to restore or where to write, and leaves eviction under memory pressure … | 5、14（接成別人能用的判斷） |
| 不寫 | — | 「贏過」任何預測型系統；打字線索（SYMPHONY 式，條件是合成的，放正文） | 5、6、7 |

## 4　早上要填、要確認的

- **【早上填】**：62%、3 of 9、all 14、58%。
  - 來源：icpe-main 的版本 C 修正版總判定（TraceLab ×3 跑完後）。
  - 現在的值來自 ADVISOR.md（版本 C，偏離 14 之前）；偏離 14 修正後，版本 B 的判定字 3／9、14／14 沒變（STATUS_MAIN 11/7 17:54）。
- **「mostly by deciding what stays in GPU and CPU memory」**：
  - 原始到達的淘汰類比例，版本 B 修正版是 51.8%（剛過一半），合成是 94.5%。
  - 版本 C 修正版的比例出來後：
    - 若原始到達 <50%，改成「largely」或拆成「synthetic: mostly; real-arrival: about half」。
- **不受早上結果影響的**：11.0–40.7%（實測）、under 1%（FreeInference）、under 11%（mini-SWE-agent）、tool-call flag（W3 判定，ADVISOR.md 3.4 節）。
- **字數**：英文版 266 字（`wc -w`，不含【早上填】標記），ICPE 的上限要查 CFP（NOT_VERIFIED）。

## 5　題目

- **建議維持**：*When the Session Returns: Restoring Parked KV Cache on H200, and Where Prediction Still Pays*
  - 「Where Prediction Still Pays」正好是老師要的「哪裡不用、哪裡仍然值得」。
- 備選：*Where Does Knowing the Future Still Pay? Measuring KV-Cache Restoration and Eviction for LLM Agents on an H200*

## 6　出處

- 摘要裡每個數字的出處同 `REPLY_TEACHER.md` 第 8 節；11.0–40.7%、寫入時放置 0 收益＝〔實測〕（PAPER.md v4.2，經 ADVISOR.md 第 8 節引用，本文沒重新核對實測值）。
- 投稿前照 `ADVISOR.md:184`，用 `paper-claim-audit` 逐句核對。
