# 預測型 KV 管理相關研究：全文核對（2026-11-03）

**為什麼做**：圖 2 (a) 原本寫「Prediction-first（common practice）：先建預測器再看收益」。`docs/NOVELTY_CHECK_20261030.md` 記的是這幾篇「只讀了摘要」，而且「對照組是不是只有 GPU／miss 就重算」列在待辦。使用者問：這是真的主流做法嗎？是哪些？

**怎麼查**：把每篇的 arXiv HTML 全文抓下來（`https://arxiv.org/html/<id>`），抽出 ①標題與章節目錄 ②含 "baseline" 的句子 ③關鍵字在全文出現的次數（Cake、SSD、DRAM／host memory／offload、recompute、oracle／Bélády、LRU、prefetch）。**這不是逐字精讀**：下表只寫抓得到的事實，解讀另標〔判讀〕。

| 論文 | 章節目錄看到的做法 | 對照組（全文的 baseline 句） | 關鍵字次數：Cake／SSD／DRAM·offload／recompute／oracle·Bélády |
|:--|:--|:--|:--|
| PBKV（2605.06472） | Workflow Prediction；Lookahead KV-Cache Eviction；Conservative KV-Cache Prefetching；附錄 E.3「Sweeping the Prediction Noise」 | LRU；SOTA baseline KVFlow（最多比 LRU 快 1.85×、比 KVFlow 快 1.26×） | 0／0／5／0／5 |
| SAGA（2605.00528） | Pattern-Based AEG Inference；Workflow-Aware Eviction；Tool-Call-Aware TTL；Speculative Prefetching；§7.2 WA-LRU 對 Bélády 的實證競爭比；§9.1.2 CPU Swap 當另一種架構 | vLLM v0.15.1 含 Automatic Prefix Caching | 0／0／6／2／2 |
| CacheScout（2608.14624） | Transition Learner；Survival Scorer；Background Prefetch Coordinator | vLLM「reactive prefix caching with LRU eviction」 | 0／0／2／8／1 |
| CacheWise（2606.16824） | Prefix-aware Request Scheduling；Predictive KVCache Eviction；§6.3 消融；§6.5 Predictor Accuracy | vLLM（block-level LRU）；InferCept（用最近歷史預測工具時間）；**CacheWise\*：給真值的工具延遲**（消融） | 0／2／14／19／4 |
| Continuum（2511.02230） | Utility Model；Setting the TTL；Tool Call Handler | vanilla vLLM（0.10.2） | 0／5／47／4／0 |
| KVFlow（2507.07400） | 依 agent 步驟圖做 prefix 快取，含預取 | SGLang；**SGLang＋HiCache（GPU＋CPU 階層）** | 0／0／10／10／0 |
| Predictive Multi-Tier（2604.26968） | Six-Tier Memory Hierarchy；Bayesian Reuse Prediction | 評估是「analytical projection」，數字取自各系統已發表的論文 | 0／2／24／4／1 |
| Ask the Tool, Don't Guess（2609.18849） | 讀工具回報的進度（不是預測） | 引擎自己的 LRU；**一個知道每次確切回來時間的 oracle** | 0／0／14／5／13 |

## 能說什麼、不能說什麼

1. **查得到的事實**：這 8 篇裡沒有一篇的全文出現「Cake」這個字（次數都是 0）。
2. **「預測型」是一條近期的研究線，不是被證明的「主流做法」**。這批多半是 2025–2026 年、針對 agent 工作流的論文；我沒有調查整個領域，所以不能說「主流」。
3. **「先建預測器、對 LRU／vLLM 報收益」大致成立**，但不是全部：PBKV、SAGA、CacheScout、CacheWise 的對照組是 LRU 或 vLLM／SGLang 的前綴快取；KVFlow 的對照組還包含 GPU＋CPU 階層（HiCache）；Predictive Multi-Tier 是階層系統但用分析推算。
4. **「它們都不看上限」不成立**：PBKV 有預測雜訊的掃描（附錄 E.3）；CacheWise 有給真值的對照（CacheWise\*）；SAGA 有對 Bélády 的比較；Ask the Tool 有 oracle 對照，而且主張「不要猜、讀工具的進度」，和我們的免費訊號論點同向。
5. 〔判讀〕對本文的新穎性：我們的差別不在「有沒有 oracle 對照」，而在 **在分層還原（Cake）之上、以 TTFT 量上限、拆成四種一次只改一件事、並跨 trace 看它取決於什麼**。「先量上限」這個做法本身，近期文獻裡已有人用到。
6. 〔判讀〕圖 2 (a) 因此要改成「預測型 KV 管理（例：PBKV、CacheScout、CacheWise、SAGA）」，不再寫成「常見做法」。

## 限制

- 只用關鍵字次數、章節目錄與 baseline 句核對，沒有逐篇精讀；「次數 0」只代表這個字沒出現，不代表概念上完全沒有類似做法。
- 預測型論文還有很多（例如 EfficientAgent、SARIA、Gandalf 等見 NOVELTY_CHECK），這份只核對上面 8 篇。
- HTML 版本號：PBKV、CacheScout、CacheWise、Continuum（v7）、Predictive Multi-Tier（v2）、Ask the Tool、KVFlow 取當下最新；日後可能變動。
