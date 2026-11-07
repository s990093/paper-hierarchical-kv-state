# ICPE 摘要第 3 版：照同類論文的寫法改（草稿；icpe-adv，2026-11-08）

> 使用者 11/8：「要看其他 paper 怎麼寫比較好，摘要等等，確保寫法類似。」第 2 版（`ABSTRACT_v2.md`）保留當紀錄；本版取代它。
> 讀了什麼：13 篇相關論文和 2 篇 ICPE 論文的摘要（arXiv API，2026-11-08 00:35 下載，原檔不進 git，sha256 見第 6 節）。

## 0　故事：看完別人怎麼寫，我們要改什麼

- **最像我們的是兩篇量測論文**：
  - *When Fancy Eviction Fails*：用 FreeInference、Bailian、SemiAnalysis，結論是「花俏的淘汰贏不了 LRU」。
  - *KVCache Cache in the Wild*：Bailian 紀錄的出處。
- **它們的摘要有三個特點**：
  - **短**：155、164 字。
  - **幾乎沒有數字**。
  - **一條線**：缺口（「行為還不清楚」）→ 我們量了什麼 → 發現 → **原因**（「The reason is structural: …」）→ **給別人的建議**（「effective prefix-cache management should …」）。
  - 這正好就是老師第 14 點要的「把現象接成別人能用的判斷」。
- **我們第 2 版的問題**：
  - 266 字、十幾個數字、內部用語（28 variants、per-cell best）。
  - 讀者看完記不住一句話。
- **所以第 3 版**：
  - 照量測論文的那條線，221 字。
  - 數字只留 3 個：<1% 對最多 62% 的對比，以及 H200。
  - 收在一句建議：「怎麼還原、寫到哪不用預測；預測留給記憶體吃緊時的淘汰。」

## 1　英文版（221 字，`wc -w`）

> LLM agents and multi-turn chat pause between requests, so serving systems move a session's KV cache out of GPU memory and must restore it before the next first token. Recent agent-serving systems keep evicted KV in host memory and predict which sessions return, and when, to drive eviction and prefetching. Yet it remains unclear which of these decisions still need prediction once restoration itself is cheap. We measure KV restoration on an NVIDIA H200 and simulate six production and public traces, pairing strong non-predictive rules (LRU, TTL, prefix caching, and others) with clairvoyant versions that know every return time, under thresholds preregistered before any result. Restoration needs no prediction: a fixed table suffices, with INT4 plus Cake fastest on slow tiers, and choosing placement at write time gains nothing. Return times matter only when memory is tight. Knowing them cuts time to first token (TTFT) by under 1% on a production agent-serving trace, but by up to **62%【早上填】** on tool-calling traces, mostly by deciding which sessions stay in GPU and CPU memory; public SWE-bench trajectories split the same way. The tool-call flag in these traces captures none of this gap. Our results suggest that serving systems need not predict how to restore or where to write, and should reserve prediction for eviction under memory pressure, where these traces offer no cue.

## 2　中文對照

> LLM agent 和多輪對話的請求之間會停頓，服務系統因此把對話的 KV cache 移出 GPU，下一個字出來之前要先搬回。近來的 agent 服務系統把逐出的 KV 留在主機記憶體，並預測哪些對話、什麼時候回來，用來決定淘汰與預取。但還原本身變便宜之後，這些決策裡哪些還需要預測，目前還不清楚。我們在 NVIDIA H200 上實測 KV 的還原，並模擬 6 份正式服務與公開的紀錄，把一組強的不預測規則（LRU、TTL、前綴快取等）和它們「知道每個對話何時回來」的預知版配對比較，門檻在看任何結果之前登記。還原不需要預測：一張固定的表就夠，慢層上 INT4 加 Cake 最快，寫入時就決定放哪也沒有收益。回來時間只在記憶體吃緊時才重要。知道它，在正式的 agent 服務紀錄上首字延遲少不到 1%，在工具呼叫的紀錄上最多少 62%，主要靠決定哪些對話留在 GPU 與 CPU 記憶體；公開的 SWE-bench 軌跡也一樣分兩邊。紀錄裡的「有沒有呼叫工具」旗標一點都拿不到。我們的結果顯示：服務系統不必預測怎麼還原、寫到哪；預測應該留給記憶體吃緊時的淘汰，而這些紀錄在那裡沒有現成的線索。

## 3　逐句對照：我們每一句學誰

| 句子的角色 | 同類論文的寫法（原句） | 我們的句子 |
|:--|:--|:--|
| 情境（1 句） | Fancy Eviction：「Long-running LLM applications repeatedly send growing context, making prefix caching critical …」 | LLM agents and multi-turn chat pause between requests, so … must restore it before the next first token. |
| 既有做法 | PBKV：「Existing approaches either …」；SYMPHONY：「Existing systems either recompute caches or offload them to host memory」 | Recent agent-serving systems keep evicted KV in host memory and predict … （老師第 4 點：承認前作已有 CPU 備份與預取） |
| 缺口 | Fancy Eviction：「Yet prefix-cache behavior under agentic workloads remains poorly understood.」；KVCache in the Wild：「there is limited understanding of …」 | Yet it remains unclear which of these decisions still need prediction once restoration itself is cheap. （老師第 5 點：哪種決策） |
| 我們量了什麼 | Fancy Eviction：「We study production traces from two companies and evaluate 14 eviction algorithms …」；「We introduce … two offline oracles」 | We measure … on an NVIDIA H200 and simulate six … traces, pairing … with clairvoyant versions … （老師第 2–3 點：寫成「預知版」，不寫「上限」） |
| 發現一 | Fancy Eviction：「Despite a large gap to Belady, sophisticated policies … provide little benefit over LRU.」 | Restoration needs no prediction: a fixed table suffices … |
| 原因 | Fancy Eviction：「The reason is structural: …」 | Return times matter only when memory is tight. 〔判讀，見第 4 節〕 |
| 發現二（帶數字的對比） | Ask the Tool：「cuts the p90 time to first token (TTFT) … by 20.7% … against LRU」 | … cuts TTFT by under 1% on … but by up to 62% on tool-calling traces … |
| 線索 | Ask the Tool：「the running tool already holds the answer, but the agent stack … silences it」 | The tool-call flag in these traces captures none of this gap. （老師第 8 點：只講「這些紀錄」） |
| 建議（收尾） | Fancy Eviction：「Our results show that effective prefix-cache management should retain recency as its foundation while …」；Ask the Tool：「A serving system should not guess what its tools can tell it.」 | Our results suggest that serving systems need not predict how to restore or where to write, and should reserve prediction for eviction under memory pressure … （老師第 14 點） |

**同類論文共同的規則，我們照做的**：
- 第一人稱、主動語態（We study／We measure）。
- 缺口用「Yet … remains unclear」。
- 發現寫成斷言，不寫「we find that X may …」。
- 原因獨立一句。
- 收在一句給別人的建議。
- 不寫內部名詞（變體數、per-cell best、元件名）。
- 「oracle」在摘要裡可以出現（Fancy Eviction 寫「two offline oracles」），但我們用「clairvoyant versions」，更清楚地表示它是「規則的預知版」。

**刻意不同的地方**：
- 量測論文不寫數字；我們留一組「<1% 對 62%」，因為這個對比就是故事。ICPE 2026 的 SweetSpot 摘要也有數字（MAPE 1.79%、33.41×）。
- 不寫「We will release …」：要不要公開模擬器，使用者還沒決定。CacheWise 沒授權，不能公開。

## 4　要確認的三件事

1. **「Return times matter only when memory is tight.」是判讀，不是直接量到的**。
   - 支持的證據〔模擬〕：
     - CPU 容量在中間時上限最大，合成工具類在 1,024 GiB 時是 0（`ADVISOR.md:91`）。
     - UltraChat、LMSYS 的最強對手 GPU 一次都不用逐出，上限剛好 0（STATUS_MAIN 11/7 20:55）。
     - OpenHands 並行 16 是 0.2%，並行 32 是 55.8%（`ADVISOR.md:115`）。
   - 反例要注意：Mooncake 對話（非工具類、真實到達）的上限也高（`ADVISOR.md:71-72`），所以「只有工具類才有」不能寫；本版寫的是「記憶體吃緊」，不是「工具類」。
   - **較保守的寫法**（如果不想下這句）：刪掉這句，第 7 句改成「… but by up to 62% on tool-calling traces when GPU and CPU memory are tight, …」。
2. **【早上填】62%**：等 icpe-main 的版本 C 修正版總判定（同 `ABSTRACT_v2.md` 第 4 節）。
   - 「mostly by deciding which sessions stay」要對照版本 C 修正版的淘汰類比例：版本 B 修正版原始到達是 51.8%，剛過一半。
   - 若版本 C 修正版 <50%，改成「largely」或拿掉「mostly」。
3. **ICPE 的摘要字數上限**：NOT_VERIFIED，要看 CFP。

## 5　題目：照同類論文的格式

| 論文 | 題目格式 |
|:--|:--|
| When Fancy Eviction Fails: Rethinking Cache Replacement For LLM Prefix Reuse | 「何時／發現」＋冒號＋做了什麼 |
| KVCache Cache in the Wild: Characterizing and Optimizing KVCache Cache at a Large Cloud Provider | 鉤子＋冒號＋做了什麼 |
| Compute Or Load KV Cache? Why Not Both? | 問句 |
| Ask the Tool, Don't Guess: … | 主張＋冒號＋解釋 |
| SweetSpot: An Analytical Model for Predicting Energy Efficiency of LLM Inference（ICPE 2026） | 名字＋冒號＋做了什麼 |

- 共同點：**一個主題**、**題目裡不放 GPU 型號**、冒號後面說做了什麼。
- 我們現在的題目 *When the Session Returns: Restoring Parked KV Cache on H200, and Where Prediction Still Pays* 有兩個主題，又放了 H200。
- **建議**：*When the Session Returns: Where Prediction Still Pays for KV-Cache Restoration*
  - 保留老師看過的開頭；一個主題；冒號後面就是答案的方向（哪裡仍然值得）。
- 備選：
  - *Restore Cheaply, Predict Selectively: A Measurement Study of KV-Cache Management for LLM Agents*：主張式，像 Ask the Tool。
  - *Where Does Knowing the Future Still Pay? Rethinking Prediction in KV-Cache Restoration*：問句加「Rethinking」，像 Fancy Eviction。

## 6　出處

- 同類論文的摘要：
  - arXiv API `id_list` 13 篇（2609.28870v2、2506.02634v5、2410.03065v2、2511.02230v7、2608.14624v1、2605.00528v2、2605.06472v1、2507.07400v1、2606.16824v1、2609.18849v1、2312.05516v3、2403.19708v3、2407.00079v5）；回應 sha256 `3e9c516c…`。
  - ICPE 的兩篇：API 搜尋 `co:ICPE`：SweetSpot 2602.05695v2（ICPE 2026）、2502.00046v1（ICPE 2025 short paper）；回應 sha256 `14635f86…`。
  - SYMPHONY：NSDI'26 PDF 的摘要（sha256 `aa19a0fe…`）。
  - 字數用各自摘要文字的空白分詞計算。
  - 原檔在 session scratchpad，不進 git；本檔只引短句。
- 摘要裡的數字：同 `REPLY_TEACHER.md` 第 8 節；62% 待版本 C 修正版。
