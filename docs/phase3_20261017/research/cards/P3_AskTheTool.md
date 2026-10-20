# P3 卡：Ask the Tool, Don't Guess（讀工具進度，不預測時長）

- **出處**：Yipeng Liu（清華、阿里雲）, Yingqiang Zhang（浙大、阿里雲）, Feifei Li（阿里雲）, Huanchen Zhang（清華）。arXiv 2609.18849v1（2026-09-16），venue〔未查證〕。<https://arxiv.org/abs/2609.18849>
- **讀了哪裡**：PDF p.1–5、p.10、p.12–13。

## 主張與證據
- **呼叫前的估計排不出長呼叫的順序**：同一批指令，錄製時（生產負載）和閒置重播時的長呼叫時長 Kendall τ 只有 −0.04；錄製 vs 超賣 −0.07；閒置 vs 減資源 0.84、有爭用 0.88、超賣 0.59〔原文 p.4 Fig.3c〕。
- 長呼叫占了大部分等待：mini-SWE-agent 的 leaderboard 軌跡中，不到 3% 的工具呼叫占了將近三分之二的工具時間〔原文 p.3〕。
- 歷史只在「同一個環境」收集時才有用；Continuum 需要每個工具約 100 筆歷史才信任分布，他們的語料中只有少數工具達到〔原文 p.4〕。
- 時間訊號的階梯（Table 1）：固定 TTL（CachedAttention、SGLang、Anthropic）→ 已經過時間（InferCept、LAMPS）→ 每工具統計（Continuum）→ 學習模型（CacheWise）→ 引擎占用（MORI、ConServe、ThunderAgent、Adaptive Retention）→ 宣告時長（TokenCake、SGLang、Mooncake、vLLM、TensorRT-LLM）→ 回報進度（本文）〔原文 p.4〕。

## 剩餘時間誤差（median |ETA 誤差|／T）
- 呼叫進行到 50%／90% 時：條件歷史 32.6%／82.0%；Continuum CDF 50.0%／96.8%；CacheWise C100 57.4%／113.3%；TokenCake、InferCept 50.0%／10.0%（兩者都預測「已完成」）；回報進度 21.8%／9.0%〔原文 p.10 Fig.7〕。

## 端到端（4×H100，重播同一批 agent session）
| 方法 | 只有 HBM：p90 TTFT | 吞吐 | HBM＋DRAM：p90 TTFT | 吞吐 | DRAM 重載 |
|:--|:--|:--|:--|:--|:--|
| Continuum（TTL） | +7.4% | +0.1% | −4.8% | +1.1% | 31.1 GiB |
| CacheWise | +22.7% | −9.2% | −14.9% | +2.1% | 77.7 GiB |
| TokenCake（宣告） | +22.4% | −6.2% | −16.0% | +3.0% | 53.8 GiB |
| InferCept（剖析） | +7.9% | −6.0% | −18.2% | +3.4% | 51.7 GiB |
| 回報進度 | −20.7% | +2.2% | −20.8% | +2.1% | 6.0 GiB |
| Oracle（知道回來時間） | −27.7% | +3.3% | −22.6% | +2.5% | 6.5 GiB |

（相對 LRU；〔原文 p.13 Fig.9〕）
- 原文解釋：只有 HBM 時，放錯的代價是別人或自己重算，決策是純排序；有 host 層時，被逐出的 context 只是複製到 DRAM，放錯「只多付一次重載」，決策變成何時放、何時拿回〔原文 p.12〕。
- 有 host 層時，進度方法加遲滯（估計遠超門檻才放），並提前一個 lead time 拿回〔原文 p.12〕。

## 和第三階段的關係〔判讀〕
- **Q1b 的最好旁證**：加上 DRAM 層後，oracle 空間從 27.7% 縮到 22.6%，同時已發表的預測器從「比 LRU 差」變成「比 LRU 好」。還原便宜會同時縮小上限和下限的距離。
- 如果主 trace 是 agent，工具呼叫的時長本來就猜不準；人為停頓（回合邊界）才是第三階段該預測的東西。
- 進度回報需要改 agent harness，不在我們的範圍；但它的 oracle 設計（知道所有回來時間、有 host 層時提前一個 lead time 拿回）可以直接當 P3-3 的 oracle 參考。
