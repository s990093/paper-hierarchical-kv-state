# 第三階段（預測器）的公開 trace 與文獻查核

- **寫於**：2026-10-17（UTC 09:50 左右完成），文獻與資料查核 agent。沒有跑任何第三階段的模擬或量測。
- **範圍**：(A) 公開 trace 能不能給「session 會不會回來、多久回來」的真實訊號；(B) 預測目標與 loss 在機制層級怎麼做；(C) 一個新穎性問題。最後一節 (D) 是對設計的含意，全部標〔判讀〕。
- **不重做的部分**：E06（LARU、LPC、SAECache、KVP、ForesightKV、LookaheadKV、TRIM-KV）、E05（LRB、HALP、KVCache-in-the-wild、Fancy-eviction）、E10（ShareGPT、LMSYS、Arena、WildChat、BurstGPT、Azure、Mooncake、Bailian、ServeGen、TraceLab、Bidaw trace）、D4 §5、D8 卡（Continuum、InferCept），以及 `research_20261009_explore/cards/` 的 MORI、EfficientAgent、CachedAttention、WhereKVLive。本文只補任務問的點；引用它們時寫卡片名稱。
- **標記**：
  - 〔原文 p.X〕：自己讀到的原文。頁碼一律是 **PDF 頁序**（USENIX 會議版含封面頁，所以比印刷頁碼多 1 左右）。
  - 〔摘要〕：只讀摘要。〔文件〕：README、資料卡、schema、HF API。〔程式碼〕：讀了程式碼。
  - 〔計算〕：我用小腳本算的，腳本與輸出在 `calc/`（見附錄 A）。〔二手〕：別篇論文的轉述。
  - 〔未查證〕：沒查到或拿不到原文。〔判讀〕：我的推論。
- **下載**：PDF 與資料樣本都放在 scratchpad（`…/scratchpad/papers/<名稱>/`），沒有進 repo。單檔都 <50 MB；沒有下載任何 >50 MB 的檔案。
- **卡片**（`cards/`）：`P3_FreeInference_trace`、`P3_Copilot2026`、`P3_CacheWise`、`P3_PBKV_CacheScout`、`P3_AskTheTool`、`P3_Serverless_KeepAlive`、`P3_LAVA`。我自己算的數字在 `calc/`（附錄 A）。

---

## 0. 一頁摘要

**(1) trace：有 session＋真實間隔、可以直接用的**

| trace | 為什麼能用 | 主要限制 |
|:--|:--|:--|
| **FreeInference agentic trace**（HF `harvardMadsys/freeinference_agentic_trace`，2026，**新**） | 12,002 個 session、267 個帳號、16 週；`session_id`＋`user_id`＋毫秒時間＋`turn_index`＋16-token 前綴 block＋工具時長〔文件〕。我抽第 1 週：prompt 中位 89K token，≥32K 占 88%；使用者發起的下一輪間隔中位 175 s〔計算〕 | 只有 agent（coding／assistant）；block 用 `o200k_base` 切，不是 Qwen／Llama tokenizer；session 不跨週 |
| **GitHub Copilot agent trace 2026**（Azure 公開資料，**新**） | 301,026 個 session、119 萬個使用者回合，一週；每次 LLM 呼叫有時間、時長、prompt／cached token；有工具時長〔文件〕 | 沒有 user id、沒有 block hash（只有 token 數）；7 個 tarball 共約 3.2 GB〔計算〕 |
| **Qwen-Bailian**（ATC'25） | 確認有 `chat_id`、`parent_chat_id`、`turn`、`type`，16-token hash〔文件〕；回來＝有子請求 | 每條只有 2 小時；長度短；**沒有跨 session 的使用者連結**（README 明說）〔文件〕 |
| WildChat-4.8M | 每個 assistant 回合有時間戳（秒）、每個 user 回合有 `hashed_ip`＋`header`，**不需同意條款**（ODC-BY）〔文件〕 | 時間是回應完成時間；短；聊天，不是長 context |
| BurstGPT_3 | `Session ID`＋`Elapsed time`（可以把服務時間從間隔裡扣掉）〔文件〕 | 沒有內容與 hash；請求很短 |
| TraceLab、ServeGen 對話檔 | 見 E10 C11、C10 | 見 E10 |

**需要使用者動手的（gated）**：LMSYS-Chat-1M、Chatbot Arena（`gated: auto`，要在 HF 同意條款）〔文件〕。但兩者都**沒有逐輪時間戳**，對第三階段幾乎沒用。

**沒有 session 或沒有真實間隔的**：Mooncake（沒有 session id；前綴鏈重建**有歧義**，「會回來」的比例依規則是 20% 或 44%〔計算〕）、Azure 2023／2024（Splitwise 用的就是 2023 版）、Chutes metadata（有 user id、沒有 session）、ShareGPT、LMSYS、多數 agent 軌跡資料集（只有訊息、沒有時間）。Continuum 說會釋出 trace，但 repo 裡找不到〔程式碼〕。

**(2) loss：各類代表作怎麼處理「不回來」與排序**

| 類別 | 代表作 | 「W 之後才回來」vs「永不回來」 | 排序怎麼來 |
|:--|:--|:--|:--|
| 回歸＋補值 | LRB（log 下次時間、L2）、3L-Cache、GL-Cache | LRB：視窗外一律標 2×視窗，**刻意同分**〔原文 p.8〕；3L-Cache：等待時間＋最長等待，刻意拉開〔原文 p.6〕 | 預測值排序 |
| 成對／排序 | HALP（成對 BCE）、Parrot（可微 NDCG＋DAgger） | HALP：只要其中一個先回來就有標籤，另一個不必回來〔原文 p.5〕 | 直接學順序 |
| 模仿 OPT | Glider／Hawkeye（OPTgen 標籤）、Parrot、LFO | 標籤＝OPT 的決策，不需要時間 | — |
| 分布／存活 | Raven（MDN，只讀到摘要）、LHD（年齡條件機率）、LAVA（以已存活時間為條件的 GBDT）、Copilot（12 個分位數迴歸→S(t)）、CacheWise（經驗存活曲線→E[剩餘時間]）、Continuum（經驗 CDF→TTL） | 存活曲線天然處理右設限；LHD 用「被逐出」當競爭事件〔原文 p.6〕 | 由 S(t) 或條件期望推出 |
| 分類 | LPC（加權 BCE：會不會續聊）、PBKV（CE：下 K 個 agent） | LPC：二元，不管多久 | 機率×時間衰減 |
| 線上專家 | LeCaR／CACHEUS（regret 調 LRU／LFU 權重） | 不預測時間 | — |
| 決策導向 | SPO+（線性目標、凸代理）〔原文 p.1〕 | — | **查不到用在快取或 KV 逐出的例子**（範圍見 B.5） |

**退路**的做法：LRB 冷啟動先用 LRU〔原文 p.6〕；HALP 用 LRU 先篩 4 個候選，「提供決策品質下限」〔原文 p.5〕；LARU 逐步縮候選集〔E06〕；CacheScout 分數乘上近期衰減、預測變平時自動退回 LRU〔原文 p.7〕；Continuum 用 TTL 到期當保險，冷啟動用固定 TTL〔原文 p.7〕；理論上「跟隨預測或穩健演算法、誰差就換」只要很小的代價〔原文 Chłędowski et al. p.1〕。

**(3) C 的答案**：**查不到**直接研究「讀取端自適應（Cake 類重算＋載入）會不會縮小預測價值」的論文，也查不到在多層 KV（含重算）裡比較「寫入時／淘汰時／預取時」預測的論文（範圍見 §C）。最接近的間接證據有三條：
- CacheWise：KV 搬到 CPU 而不是重算時，增益從 2.7–3.5 倍降到 1.19 倍，原文自己說「每次逐出的懲罰變小，空間也跟著變小」〔原文 p.11〕。
- Ask-the-Tool：加上 DRAM 層後，oracle 的 p90 改善從 27.7% 降到 22.6%；而 CacheWise、TokenCake 兩個已發表的預測器在只有 HBM 時**比 LRU 差** 22–23%，加上 DRAM 層後反而比 LRU 好 15–16%〔原文 p.13〕。
- Continuum：有 LMCache 快速卸載時，重新載入很便宜，InferCept 的 preserve 幾乎不再觸發〔原文 p.5〕。

**(4) 最重要的 5 條設計含意〔判讀〕**（詳見 §D）
1. **主 trace 換成 FreeInference agentic trace**（長 context、session、user、block、工具時長都有），Bailian／WildChat 當短 context 對照，Mooncake 只當「重建規則的敏感度」。
2. **「W 之後才回來」和「永不回來」在我們的動作空間裡不能同分**：LRB 刻意同分是因為單層快取只有「留／踢」，但我們有「留 CPU／降慢層／丟掉重算」，差別剛好就在這兩者。用存活 loss（處理右設限）或 3L-Cache 式拉開，而不是 LRB 式補值。
3. **預測做成「以已閒置時間為條件」的分布**（LAVA、Copilot、CacheWise 都這樣做，S(t|t₀)＝S(t)/S(t₀)），讓「寫入時一次 vs 決定當下重算」可以用同一個模型比。
4. **預取要保守**：只用閒置頻寬與空位，不為預取趕走有用的東西。PBKV 的激進預取在預測雜訊 ≥30% 時比不預取還差〔原文 p.18〕；WhereKVLive 與 CacheScout 的預取幾乎沒增益。
5. **先量「便宜還原」把空間吃掉多少**：Cake＋CPU 層讓逐出錯誤的代價變小，Ask-the-Tool 的數字顯示這同時縮小 oracle 空間、也讓爛預測器變無害。P3-3 的 Q1b 是這個問題的直接測試，前作沒人做過。

---

## A. 公開 trace：能不能提供「會不會回來／多久回來」的真實訊號

### A.1 總表

「真實間隔」＝同一個 session 相鄰兩輪的時間差來自真實紀錄，不是 Poisson 或固定 think time。「≥8K」指輸入長度。

| trace（版本） | 大小、license、gated | session／user id | 真實間隔 | 內容文字 | 時間跨度、規模 | 長度（≥8K） | 出處 |
|:--|:--|:--|:--|:--|:--|:--|:--|
| **FreeInference agentic trace**（HF sha `16d3c04d`，2026-10-14） | datasets-server 報 8.50 GB（16 週；第 1 週的 requests.parquet 22 MB）、CC-BY-4.0、**不 gated** | `session_id`、`user_id`（267 個帳號）、`harness`；子 agent 巢狀 | 有：每個請求 `start_ms`／`end_ms`、`step_trigger`（user／tool）、工具 `latency_ms` | 無（只有 token 數、工具名、去識別參數） | 2026-05-17 → 09-05、12,002 session、1,186,582 請求 | 第 1 週 prompt 中位 89,242，≥8K 99.2%、≥32K 88.2%、≥128K 13.9%〔計算〕 | README〔文件〕；第 1 週〔計算〕 |
| **GitHub Copilot coding agent 2026**（Azure，md commit `6ada3303`；release `ghcp-coding-agent-2026`） | 7 個 tar.gz 共約 3.22 GB、CC-BY、不 gated | `session_id`（雜湊）；**沒有 user id** | 有：LLM 呼叫 `timestamp`、`duration_ms`、`initiator_type`（user／agent）；工具 batch 時長 | 無 | 2026-06-01 → 06-07、301,026 session、1,189,581 回合 | 只有 token 數；全體 prompt 631.4B／9.31M 呼叫〔文件〕，分布未算 | md、DATASET_CARD、schema.json〔文件〕 |
| **Qwen-Bailian**（commit `5f7439c5`，沒有比 E10 新） | 4 檔 27.9–132.1 MB、Apache-2.0、不 gated | `chat_id`、`parent_chat_id`（−1＝根）、`turn`、`type`；**README：不保留跨 session 或使用者裝置的關聯** | 有（相對秒） | 無；16-token SipHash | 每檔 2 小時；thinking 檔 9,612 個 session〔計算〕；traceA 9,012 個有間隔的 session〔E10〕 | thinking：中位 3,680、≥8K 19.6%、≥32K 0.18%〔計算〕；四條的 ≥32K 都只有 0–0.18%〔E10〕 | README〔文件〕；〔計算〕 |
| **Mooncake**（`FAST25-release`，commit `1d0e4c75`；**沒有新 trace**） | 3.0／4.4 MB、repo Apache-2.0 | **無** | 時間戳真實（毫秒；70% 是整秒的倍數）〔計算〕；session 要用前綴鏈猜 | 無；512-token hash | 各約 1 小時、12,031／23,608 請求 | 中位 6,909／6,346〔E10〕 | README〔文件〕；〔計算〕 |
| **CacheWise CATraces**（commit `181c435a`，2026-06-14） | 188 個 events.json 合計約 28 MB（含工具預測用的副本）、**沒有 license 檔**（GitHub API `null`） | `session_id`、`project_id`（11 個專案） | 有：每個事件 ISO 時間、工具 `execution_time_ms`；日期看起來被平移過（session_0001 是 2020-07-07，但模型是 2025-08 版）〔判讀〕 | 已遮蔽（`<redacted>`） | 94 session、20,634 次 LLM 呼叫、11,617 次工具呼叫〔文件〕 | 只有 token 數（`input_tokens`、`cache_read_input_tokens`…） | manifest、events.json〔文件〕 |
| **WildChat-4.8M**（HF sha `c827c6df`）／1M（`7d6490e4`） | ODC-BY、**不 gated** | `conversation_hash`；user 回合有 `hashed_ip`＋`header`，可啟發式串同一使用者 | 有：assistant 回合 `timestamp`（秒）；user 回合的 timestamp 是 None | **有** | 2023-04 → 2025-08、3,199,860 段〔E10〕 | 未計算〔E10〕 | first-rows〔文件〕 |
| LMSYS-Chat-1M（sha `200748d9`） | **gated: auto** | `conversation_id`；**沒有 user id** | **無時間戳** | 有 | 100 萬段 | 短 | HF cardData〔文件〕 |
| Chatbot Arena（sha `1b6335d4`） | **gated: auto** | `judge`（匿名使用者） | 只有**每段對話一個** `tstamp`；回合沒有時間 | 有 | 33K 段 | 短 | HF cardData〔文件〕 |
| ShareGPT | 不 gated | 無 | **無** | 有 | — | 短 | E10 C1 |
| BurstGPT（release v2.0，2026-01-15） | `BurstGPT_3.csv` 231.7 MB、CC-BY-4.0 | `Session ID`（**只有 conversation 模式有**） | 有；另有 `Elapsed time`＝送出到完整回應的秒數 | 無 | BurstGPT_3：55,920 session〔E10〕 | request 中位 327〔E10〕 | README〔文件〕 |
| Azure LLM 2023（Splitwise）／2024（DynamoLLM）／LMM 2025 | CC-BY | **無** | 只有到達時間 | 無 | 1 小時／一週／一週 | 2023 版最大 14,050〔E10〕 | E10 C8；README〔文件〕 |
| Chutes metadata（HF `harvardMadsys/chutes_metadata`，sha `85722e69`） | 94.3 GB、CC-BY-4.0、不 gated | `user_id`＋`rehash_round`（hub 頁說 user id 每三個月輪換）；**沒有 session** | 有 `started_at`／`completed_at`（從 0 起算） | 無 | 6,122,418,314 列、一年 | 有 `it`／`ot`／`ct`（輸入、輸出、快取 token） | first-rows、hub 頁〔文件〕 |
| TraceLab v0.0.2、ServeGen 對話檔 | 見 E10 | TraceLab 有 session、round、匿名 user | 有 | 無 | 見 E10 | TraceLab 中位 132K；ServeGen 中位 7,745 | E10 C10、C11 |
| OpenHands feedback（HF `OpenHands/openhands-feedback`） | MIT、不 gated | 無 | 每個事件有微秒時間戳 | 有 | **275 列**（有重複分享） | — | rows API〔文件〕 |
| Agent 軌跡（`nebius/SWE-agent-trajectories`、`SWE-Gym/OpenHands-*`、`nvidia/SWE-Zero-*`、`SWE-bench/SWE-smith-trajectories`） | 不 gated | 無 | **沒有時間欄位**（first-rows 掃描）；`nebius/SWE-rebench-openhands-trajectories` 的 first-rows 被截斷，〔未查證〕 | 有 | — | — | first-rows〔文件〕 |
| Continuum 的 SWE-bench／BFCL 工具時長 | — | — | 論文 p.2 說會釋出；repo `Hanchenli/vllm-continuum`（HEAD `316a5879`）只有 job 完成時間的 `results.json`，**找不到逐工具時長** | — | 原文 Table 2（格式為「平均, 標準差」）：工具時間 SWE-bench (925, 3,550) ms、BFCL (1,923, 2,133) ms；各約 100 條軌跡〔原文 p.4〕 | — | 〔程式碼〕 |
| SWE-bench leaderboard 的 mini-SWE-agent 軌跡、OpenHands 評測輸出 | — | — | Ask-the-Tool 說這些軌跡「記錄了時長」，約 3,000 條〔二手：Ask-the-Tool p.3〕；我查的那個 submission 目錄裡沒有軌跡檔〔未查證〕 | 有 | — | — | — |

### A.2 逐項補充（只寫 E10 沒有的）

**Qwen-Bailian：欄位確認，以及「回來」標籤的形狀**
- 欄位就是 `chat_id, parent_chat_id, timestamp, input_length, output_length, type, turn, hash_ids`〔文件 README〕。README 的匿名化第 3 點：chat id 換成連續整數、「No linkage to user accounts or device identifiers」；隱私段：「No cross-session or user-device associations preserved」〔文件〕。所以**只有 session 內的「下一輪」**，沒有「同一個人隔天回來」。
- thinking 檔（27.9 MB，sha256 `41ac36d9…`）〔計算，`calc/bailian_thinking.txt`〕：
  - 10,812 列，`ceil(input/16)==len(hash_ids)` 100%（規則 6 的單位斷言）。
  - 9,612 個根（＝session），1,200 個子請求；子的 `turn` 全部等於父＋1；沒有任何父請求有兩個子。
  - 1,156／1,200（96.3%）的子請求在「父請求最後一塊以外」完全共用父的前綴（最後一塊會變，FAQ 有說明）。
  - **右設限很明顯**：前半段的請求有 16.2% 之後有下一輪，最後 600 秒的只有 4.3%。
  - 子減父的時間差（含父請求的服務時間）p10／50／90／99＝5.1／28.6／255.3／1,533.3 s。
- 〔判讀〕Bailian 的「回來」要當成**有設限的事件**處理；2 小時視窗下，晚回來的全部被截掉。

**Mooncake：沒有 session id，用前綴鏈重建的歧義**
- README 沒有新增 trace；欄位仍是 `timestamp`（ms）、`input_length`、`output_length`、`hash_ids`（512 token 一塊）〔文件〕。repo 附的 PDF 附錄說取樣時優先收同一使用者的 session〔E10 C9〕。
- 我試了兩條重建規則〔計算，`calc/mooncake_*.txt`〕：
  - **寬鬆**：之後有請求的前綴完整延伸它（取最長的共同前綴）。conversation 有 43.9% 的請求「會回來」（和 D4 的 43.9% 一致）、toolagent 44.1%。但這樣的子請求中，conversation 有 8,643／12,014、toolagent 有 20,123／23,581 有**不止一個**同樣長度的候選父請求（共用系統提示或文件）；toolagent 的間隔中位是 0 s（同一秒的分叉，不是下一輪）。
  - **嚴格**：共用前綴最多被 3 個請求共用、父請求唯一、時間差 >0。conversation 只剩 20.2% 會回來（2,436 條鏈），toolagent 10.6%（2,515 條）；間隔中位 132 s／123 s。
  - 寬鬆規則下，conversation 前半段的請求 45.9% 會回來，最後 600 秒 34.5%（設限）。
- 〔判讀〕Mooncake 的「會不會回來」標籤**由重建規則決定**，差 2 倍以上。可以用來測「標籤雜訊」的敏感度，不適合當主要學習資料。

**CacheWise CATraces：欄位與兩個不一致**
- 事件格式〔文件 events.json〕：
  - `llm_call`：`timestamp`、`model`、`input_tokens`、`output_tokens`、`cache_creation_input_tokens`、`cache_read_input_tokens`、`has_tool_use`、`num_tools_called`。
  - `tool_call`：`tool_name`、`tool_input`（`<redacted>`）、`execution_time_ms`、`result_timestamp`。
- 工具時長預測程式（`tool_duration_prediction/`）〔程式碼〕：每個工具名一組 TF-IDF＋MiniBatchKMeans 叢集（每叢集至少 30 筆、最多 1,000 個叢集），每叢集存一條經驗「仍在執行」曲線 1−F(t)；推論時算 E[T−t | T>t]（梯形積分 ÷ S(t)）；相似度 <0.3 就退回該工具的整體曲線。使用者停頓另外插成 `HumanPause` 假工具（≥2 s、<1 天、中間沒有工具呼叫）。
- 不一致 1：README 說原始 trace 在 `release/raw_traces/`，repo 實際在根目錄的 `raw_traces/`〔文件〕。
- 不一致 2：論文說「隨機取 80% 的 session 離線訓練」〔原文 p.8〕；repo 的 `train_sessions.json` 是 34 個 session（94 個的 36%）〔程式碼〕。
- 只有 94 個 session，低於 explore/README 的 1,000 session 門檻；只能當 agent 情境的補充。

**WildChat：逐輪時間與跨對話使用者**
- first-rows（4.8M 版）：對話層有 `timestamp`、`turn`、`hashed_ip`、`header`、`country`、`state`；`conversation` 裡 user 回合有 `hashed_ip`、`header`，`timestamp` 為 None；assistant 回合有 `timestamp`（到秒）與 `turn_identifier`〔文件〕。
- 〔判讀〕回合間隔＝相鄰 assistant 時間差，包含「下一輪的生成時間」。要估使用者停頓，得扣掉生成時間（沒有欄位，只能用輸出長度估）。跨對話的「同一個人回來」只能靠 `hashed_ip＋header` 啟發式（E10 也這樣說）。

**LMSYS／Arena／ShareGPT：LPC「公開資料沒有時間戳」查證**
- LPC 原文用 LMSys、ShareGPT、Chatbot-Arena，並說公開資料沒有時間戳，所以自己合成兩階段到達〔E06，LPC 原文 p.7〕。
- 查證結果：LMSYS-Chat-1M 的 features 沒有任何時間欄位，也沒有 user id；ShareGPT 沒有；Arena 有 `tstamp`（float64），但**每段對話只有一個**，回合本身只有 `content`／`role`〔文件 HF cardData〕。
- 〔判讀〕LPC 的說法對它用的三份資料成立（Arena 的單一時間戳給不出回合間隔）；但 WildChat（LPC 沒用）有逐輪時間。

**BurstGPT：可以把服務時間扣掉**
- README：`Session ID`「只有 conversation 模式有」；`Elapsed time`＝「從送出請求到拿到完整回應（含全部 token）」，單位秒，不是 TTFT〔文件〕。release v2.0 才有 `BurstGPT_3.csv`。
- 〔判讀〕使用者停頓＝下一輪 `Timestamp` −（這一輪 `Timestamp`＋`Elapsed time`），是少數能直接分離服務時間的公開 trace。

**GitHub Copilot 2026（新）**
- 欄位：每個 session 有 `turns`；每回合有 `llm_calls`（`timestamp`、`duration_ms`、`initiator_type`＝user／agent、`model`（匿名）、`result`、`tokens`{total, prompt, completion, cached, cached_pct}、`message_metadata`（各段類型與 token 數））與 `tool_batches`（`timestamp`、`duration_ms`、`auto_continue`、`function_calls`{name, status, duration_ms}）〔文件 schema.json〕。
- 明說沒有：程式碼、prompt、回應、檔案路徑、使用者／帳號 id、真實模型名〔文件 DATASET_CARD〕。取樣是「所有日期固定取樣率」〔文件〕。
- 論文（arXiv 2608.00101v1）的特徵化用的是全量（320 萬使用者、1,300 萬 session）〔摘要〕；公開的是取樣版。論文 §9 有回來時間預測器，見 B.3。

**FreeInference agentic trace（新）**
- README〔文件〕：
  - 每列一個 session；請求有 `step`、`turn_index`、`step_trigger`、`link_reason`、`start_ms`、`end_ms`、`ttft_ms`、`provider_tokens`、`role_tokens`、`mutation`、`block_ids`、`tool_calls`；工具有 `latency_ms`（「到下一次 LLM 請求的間隔」）與 `subagent`。
  - `block_ids`＝`o200k_base` 切成 16-token 的鏈式前綴 block，最後不滿一塊的丟掉；block id 每週各自編號。
  - session 是用重複的 tool-call id 與 assistant 訊息**重建**的；session 不跨週。
- 我抽第 1 週（2026-05-17 那週；requests.parquet sha256 `6772d28d…`）〔計算，`calc/freeinference_week1.txt`〕：
  - 125 個 session（其中 19 個是子 agent）、8 個使用者、10,738 個請求（785 個使用者發起、9,953 個工具觸發）。
  - 單位斷言 `len(block_ids)==floor(sum(role_tokens)/16)` 100% 成立（用 ceil 只有 6.5%，和「丟掉最後不滿一塊」一致）。
  - 使用者發起的下一輪：距上一個請求結束 p10／50／90／99＝25／175／3,587／34,687 s；工具觸發的：0.5／3.4／30.5／294.7 s。
  - 每個 session 的使用者回合數中位 3、p90 11。
- 〔判讀〕這是目前唯一同時有「長 context＋session＋使用者＋前綴 block＋真實間隔＋≥1,000 session」的公開 trace。第 1 週只有 8 個使用者，跨週合併後才有 267 個帳號；切分要依使用者或依時間，不能只依 session。

**沒有 session 的 trace 能不能拿來用**：Azure、Chutes 只能提供到達率與日週期；Chutes 的 `user_id` 可以看「同一使用者多久再來」，但沒有前綴，無法知道是不是同一份 context〔判讀〕。

### A.3 每個 loader 要寫的單位斷言（規則 6）

| trace | 斷言 | 狀態 |
|:--|:--|:--|
| Bailian | `ceil(input_length/16)==len(hash_ids)`；子 `turn`＝父＋1 | 兩條在 thinking 檔都 100%〔計算〕 |
| Mooncake | `ceil(input_length/512)==len(hash_ids)` | 兩檔都 100%〔計算〕 |
| FreeInference | `len(block_ids)==floor(Σrole_tokens/16)`；同一 session 的 `step` 連續 | 第一條第 1 週 100%〔計算〕；第二條未查 |
| Copilot | 回合內 `llm_calls` 時間遞增；`tokens.total==prompt+completion` | 未查（沒下載） |
| BurstGPT_3 | `Total tokens==Request+Response`；`Session ID` 只出現在 Conversation log | 未查 |
| WildChat | 相鄰 assistant `timestamp` 非遞減 | 未查 |

---

## B. 預測目標與 loss（機制層級）

### B.1 經典學習式快取

| 論文 | 預測什麼 | label 怎麼來 | loss | 「永不回來」怎麼處理 | 怎麼變成決策 | 和 Belady／oracle 比 | 退路 | 出處 |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| **LRB**（NSDI'20） | log(下次請求時間) | 等到再被請求；或超出滑動記憶窗 | **L2**；LightGBM 的 8 種 loss 都試過，以 good decision ratio 選 | 視窗內沒再請求的標成 **2×視窗**；理由：超過 Belady 邊界的物件「都是同樣好的逐出決定」 | 隨機抽 64 個候選，逐出預測最遠的 | relaxed Belady 當目標；縮小 SOTA→Belady 差距約 1/4〔E05〕 | 訓練資料不夠前用 LRU | 〔原文 p.6–8〕 |
| 3L-Cache（FAST'25） | log(下次到達間隔) | 再被請求的間隔 | MSE（6 個特徵） | 離開視窗的標成「已等待時間＋樣本中最長等待時間」，「讓最終會回來和不會回來的明確分開」 | 依預測排序 | — | 〔未查證〕 | 〔原文 p.6〕 |
| GL-Cache（FAST'23） | 物件**群組**的效用 Σ 1/(T×size) | 取樣群組後線上累加；被逐出的用 ghost entry 補 | L2 迴歸；「試過 ranking objective，沒有明顯差別」 | 沒回來的物件對效用貢獻趨近 0（T 很大） | 逐出效用最低的群組 | 有 oracle 輔助版做對照 | 〔未查證〕 | 〔原文 p.6–7、p.9〕 |
| **HALP**（NSDI'23） | 一對候選中誰先被再存取 | 任一個先被存取就有二元標籤；ghost cache 被踢掉時刪掉未決的比較 | **二元 cross entropy**，一層隱藏層 | 不需要另一個回來 | LRU 尾端 4 個候選，3 次成對比較（淘汰賽） | 和 LRB 比，沒有 Belady | LRU 篩候選「提供決策品質的下限」 | 〔原文 p.5–6〕 |
| **Parrot**（ICML'20） | 模仿 Belady 的逐出分布 | Belady（看得到未來）的動作與重用距離 | **可微 NDCG**（重用距離當相關度）＋輔助的 log 重用距離 MSE | Belady 視窗外設 ∞ | 選機率最大的行 | 直接模仿 Belady；on-policy（DAgger）比 off-policy 平均多 9.8% | 無 | 〔原文 p.4–5、p.8〕 |
| Glider（MICRO'19） | 這一行對 OPT 是 cache-friendly 還是 averse | OPTgen 模擬 OPT 在過去存取上的決定 | 離線 LSTM（attention），線上 ISVM（等同感知器） | 標籤本身就是 OPT 的決定 | 插入優先權 | 標籤即 OPT | — | 〔原文 p.2–3〕 |
| LHD（NSDI'18） | 命中密度＝期望命中 ÷（大小×期望剩餘壽命） | 線上統計命中年齡與逐出年齡的分布 | 無（條件機率估計） | 「被逐出」是壽命 L 的一部分：P[命中|年齡 a]＝P[H>a]/P[L>a]；1% 空間留給不逐出的 explorer 物件，避免自我強化 | 抽 64 個，逐出密度最低的 | 和 GDSF、AdaptSize、Hyperbolic 比 | — | 〔原文 p.6–7〕 |
| LeCaR／CACHEUS（FAST'21） | 不預測時間；學 LRU 與 LFU 兩個專家的權重 | 被誤逐出又被請求 → 懲罰該專家 | regret 最小化；CACHEUS 拿掉 discount、學習率自適應 | — | 依權重隨機選專家 | — | 專家本身就是啟發式 | 〔原文 p.5〕 |
| Raven（CoNEXT'22） | 下次到達時間的**分布**（Mixture Density Network） | — | 〔未查證〕 | 〔未查證：是否用設限似然〕 | 算「比快取裡其他物件更晚到」的機率，逐出最大者 | 以 Belady 為指引 | — | 〔摘要〕；全文（NSF PAR、ACM）都連不上 |
| LFO（HotNets'18） | OPT 的准入決定 | OPT | 決策樹分類 | — | — | — | — | 〔摘要〕（搜尋結果） |

〔判讀〕LRB 與 3L-Cache 對同一件事做了**相反**的選擇：LRB 讓「很晚才回來」和「不回來」同分，3L-Cache 刻意分開。單層快取只有「留／踢」，所以 LRB 的同分沒有損失；我們的動作空間有「降到慢層」和「丟掉重算」，差別剛好在這兩群之間。

### B.2 learning-augmented caching 的理論保證與實務退路

| 論文 | 預測什麼 | 錯誤怎麼量 | 保證 | 出處 |
|:--|:--|:--|:--|:--|
| Lykouris & Vassilvitskii（ICML'18；arXiv 1802.05399 v2020） | 每個元素下次出現的位置；**不再出現的設為 n+1** | 總絕對誤差 η＝Σ\|y−h\|（也推廣到平方誤差） | Predictive Marker：2＋O(min(√(η/Opt), log k))；預測再爛也不超過 O(log k) | 〔原文 p.2、p.8、p.10〕 |
| Rohatgi（SODA'20）、Wei（APPROX'20）、Antoniadis et al.（ICML'20） | 重用距離或 OPT 的動作 | η_reuse 或 η_cache | 表中各式 min(log k, f(η/OPT)) | 〔二手：Chłędowski et al. Table 1，p.3〕 |
| **Chłędowski et al.**（ICML'21，實驗研究） | 用 Parrot 當預測器 | — | 「盲從預測器或穩健演算法，誰變差就切換」在預測器好時開銷很小，預測器壞時仍和經典方法競爭，是「便宜的最壞情況保險」 | 〔原文 p.1〕 |
| LARU（arXiv 2509.20979） | 下次請求時間 | — | 1-consistent、O(k)-robust；偵測到誤判就縮候選集 | E06 卡 |

〔判讀〕理論的錯誤量是**絕對時間誤差**，遠在未來的元素誤差很大但對決策無關；LRB 改用 log 目標就是在修這件事。

### B.3 KV 專屬：每篇預測什麼、用什麼 loss

（LARU、LPC、SAECache 的完整卡在 E06，這裡只放比較欄位。）

| 論文 | 預測什麼 | label／訓練 | loss | 「不回來」 | 決策 | 和 oracle 比 | 退路 | 出處 |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| LARU | 節點的下次請求時間 | 滑動窗內再存取 | LightGBM（loss 原文未寫） | 原文沒講清楚 | LRU 前 l 個候選中逐出最遠的 | 只在動機圖有 OPT | 有（縮候選集） | E06 |
| LPC（NeurIPS'25） | 對話會不會有下一輪 | 對話結構 | **加權 BCE**（少數類加權） | 二元，不管多久 | 機率×指數時間衰減 | 自訂 oracle（完美知道會不會續聊）：補回 22–64% | 無，只有衰減 | E06 |
| SAECache | block 存活機率（log-normal 間隔）＋首輪會不會變多輪 | 線上統計＋MLP | 統計更新；MLP loss 原文未寫 | log-normal 存活 | 多佇列 | 無 | 權重上下限 | E06 |
| KVCache-in-the-wild WA（ATC'25） | 各類別（請求類型×輪數）的重用時間分布 | 背景取樣最近一小時，**擬合指數分布** | 不訓練（擬合） | 隱含在 F(t) | Priority＝(ReuseProb(t, life)＝F(t+life)−F(t), −offset) 依字典序 | 無 Belady〔E05〕 | 〔未查證〕 | 〔原文 p.11〕 |
| Bidaw（FAST'26） | 下次存取的「加權重用距離」落在哪一桶 | 每使用者的歷史分布＋上一輪回答長度給下界 | 不訓練 | 「極端」桶命中率 0 | 命中潛力＝Σ P(桶)×（Belady ghost cache 在該桶的命中率），逐出最低者 | 背景跑 Belady ghost cache 取命中率 | 〔未查證〕 | 〔原文 p.8〕 |
| KVFlow（arXiv 2507.07400） | 不學：應用給的 Agent Step Graph 算 steps-to-execution | — | — | — | steps-to-execution 大的先逐出；預取 | 比 SGLang（含 HiCache）快 1.83×／2.19× | — | 〔原文 p.1–2〕 |
| **Continuum**（arXiv 2511.02230v7） | 這個工具 f 在 τ 秒內回來的機率 | 同一 program_id 的請求間隔 | 不訓練：**經驗 CDF** P(τ,f)＝(1/\|S[f]\|)Σ1[t≤τ] | TTL 到期就放掉（有界損失） | τ\*＝argmax P(τ,f)·(T·η＋Prefill-Reload)−τ，在歷史時長中列舉 | 無 oracle | \|S\|≤K（K＝100）用固定 T_default（假設 Exp(1)、η＝1 推出）；\|S[f]\|≤K 用全體工具分布 | 〔原文 p.5–7〕 |
| **CacheWise**（arXiv 2606.16824v1） | 工具呼叫的**剩餘**時間 E[τ\|τ−T_i > t−T_i] | 依工具名＋TF-IDF 叢集的歷史時長 | 不訓練：KMeans 叢集＋經驗存活曲線 | 不適用（工具一定會回來）；人的停頓當 session 邊界 | 依 E[τ] 建逐出堆，每 3 個引擎迭代重建一次（因為預測會過期） | CacheWise\*（真實工具時長）：表現相近，有時還略好 | 無；漂移「留給未來」 | 〔原文 p.6–8〕；〔程式碼〕 |
| InferCept（arXiv 2402.01869） | 中斷（工具）時長 | 離線剖析；沒剖析過的用**已經過時間** T̂＝t_now−t_call | 不訓練 | — | 選 preserve／swap／discard 中浪費最小者 | 用這個估計達到 oracle 時長的 93% | — | 〔原文 p.7〕 |
| CachedAttention（ATC'24） | 不學：排隊中的 job | — | — | — | look-ahead 視窗預取、淘汰 | 見 A_CachedAttention 卡 | — | 卡 |
| MORI（arXiv 2606.00866） | 程式的「閒置程度」排名 | 工具呼叫的歷史型態 | 不訓練 | — | 最閒的放 CPU | 見 A_MORI 卡 | — | 卡 |
| **PBKV**（arXiv 2605.06472v1） | 接下來 K 步會呼叫哪些 agent | 離線呼叫 trace（prefix＋prefill embedding → 下 K 個 agent） | **cross entropy**（⟨END⟩ 之後 mask） | 預測終止之後的步數不計入分數 | 先回收「已退休」的 cache，再依前瞻重用分數排序，否則退回 LRU | 比 LRU 快 1.85×、比 KVFlow 快 1.26×；全文沒有 oracle 線（檢索 oracle／ground truth 無結果） | 證明效能對預測誤差 Lipschitz 連續 | 〔原文 p.1–6、p.19〕 |
| **CacheScout**（arXiv 2608.14624） | K 步內會不會再用到這個 agent anchor（存活機率） | **線上轉移矩陣計數**，不離線訓練 | 不訓練 | 修剪低信心邊後用 BFS 跳數近似 | Score＝(p_surv＋δ)·e^{−λ·age}＋δ·\|b\| | 和 vLLM、Continuum 比 | 預測變平時近期項主導，「自然退回 LRU」 | 〔原文 p.2、p.7〕 |
| **Copilot 2026**（arXiv 2608.00101v1） | 回合邊界後的使用者閒置時間 | 一週 150K session 訓練、**下一週** 50K 評估 | **LightGBM 分位數迴歸**（log 閒置時間，12 個分位數、各 400 棵樹）→ 組成 S(t) | 原文沒有提設限〔原文未見〕 | 依 S(t) 選操作點（例：Pr(idle>60 s) 低於門檻就逐出）；過了 t₀ 用 S(t)/S(t₀) 更新，不必重訓 | AUC 0.73（>60 s），「只看上一次閒置」的啟發式 0.58；沒有端到端策略評估 | — | 〔原文 p.17〕 |
| **Ask-the-Tool**（arXiv 2609.18849v1） | 不預測：讀執行中工具的進度 | — | — | — | 剩餘時間低於門檻就保留；有 host 層時加遲滯、提前一個 lead time 拿回 | 有 oracle（知道所有回來時間），見 B.4 | session credit 限制說謊的工具 | 〔原文 p.1–4、p.13〕 |

**預測目標的實測可預測度**（同一個問題在不同 trace 上的數字，**不能互相比較**，只給量級）：
- Copilot：>60 s 閒置的 AUC 0.73〔原文 p.17〕。
- Ask-the-Tool：同一指令在不同負載下，長呼叫時長的 Kendall τ：錄製→閒置重播 −0.04、錄製→超賣 −0.07；閒置→減資源 0.84、→有爭用 0.88、→超賣 0.59〔原文 p.4 Fig.3c〕。「呼叫前的估計排不出長呼叫的順序」。
- 我們自己 9 月的 GBDT：AUC 0.92–0.999，但策略輸給規則（任務背景）。

### B.4 誰做了預取：觸發依據、對手、增益

| 論文 | 預取方向 | 觸發依據 | 對手 | 增益 | 出處 |
|:--|:--|:--|:--|:--|:--|
| CachedAttention | 磁碟 → host | 排隊中 job 的 look-ahead 視窗 | LRU、FIFO | 整體命中 86% vs LRU 58%、FIFO 48% | A_CachedAttention 卡〔原文 p.7、p.11〕 |
| KVFlow | CPU → GPU，完全重疊 | Agent Step Graph | SGLang＋HiCache | 1.83×／2.19×（含逐出） | 〔原文 p.1–2〕 |
| **PBKV** | host → GPU | 一步前瞻；**只用閒置的 GPU 空位與 PCIe**、只在純 decode 批次（>90%）時 | 不預取（PBKV-HE）、激進版（可趕走 active cache） | 激進版只在雜訊 λ≤10% 時勝過保守版；λ≥30% 時**比不預取還差**；保守版在所有雜訊下都 ≥ 不預取 | 〔原文 p.6、p.17–18〕 |
| CacheScout | 重算暖機（送一個只有 agent anchor 的請求） | 轉移矩陣預測的下一個 agent | vLLM、Continuum | **單獨預取最多 +1 個百分點**；增益主要來自逐出（+18–22 點）；但和預測式逐出一起用時，GAIA 每回合延遲再降 28% | 〔原文 p.9–10〕 |
| Ask-the-Tool | DRAM → GPU | 工具回報的剩餘時間，提前一個 lead time | LRU、四個已發表預測器、oracle | 有 DRAM 層：p90 −20.8%（oracle −22.6%）；DRAM 重載 6.0 GiB vs CacheWise 77.7 GiB | 〔原文 p.13〕 |
| WhereKVLive | 下層 → 上層 | 預測重用、lookahead 參數 | 不預取 | 在爭用的連結上，**連 oracle 預取都沒有一格贏過不預取** | A_WhereKVLive 卡 |
| Serverless in the Wild | pre-warm（冷 → 熱） | 閒置時間直方圖的頭部（5 百分位） | 固定 keep-alive | 見 B.7 | 〔原文 p.8〕 |
| TraceLab | 只列為機會：「間隔預測＋預取可以讓逐出更積極」 | — | — | 未評估 | 〔原文 p.15〕 |

### B.5 決策導向學習（predict-then-optimize）

- **SPO／SPO+**（Elmachtoub & Grigas，Management Science；arXiv 1710.08005v5）：SPO loss＝預測造成的**決策誤差**（regret）；SPO+ 是用對偶推出的凸代理，在溫和條件下和 SPO 統計一致；適用於**線性目標**的多面體、凸或混合整數問題〔原文 p.1〕。
- **用在快取或 KV 逐出**：**查不到**（範圍：WebSearch「decision-focused learning／predict-then-optimize＋caching／eviction／cache admission／job scheduling」，只出現一般 DFL 方法論，例如 Mandi et al. 的 contrastive loss＋solution caching、Schutte et al. IJCAI'24 的 robust regret loss〔摘要〕；以及本文 B.1–B.3 讀過的 20 篇快取／KV 論文，沒有一篇用 regret 當訓練 loss）。
- 最接近「決策導向」的是**用 OPT 的決定當標籤**：Glider／Hawkeye（OPTgen）、Parrot（模仿 Belady＋NDCG）、LFO（學 OPT 的准入）、LRB（目標是 relaxed Belady，用 good decision ratio 選 loss）〔原文／摘要，見 B.1〕。
- 〔判讀〕逐出是線上、序列、組合的決策，不是 SPO+ 要求的「一次性線性目標」；直接套 SPO+ 不自然。比較可行的是「模仿 oracle 的決定」（Parrot 式 DAgger，或用模擬器的 oracle 動作當標籤）。

### B.6 存活分析與排序 loss 在系統預測上的用法

| 做法 | 系統裡的實例 | 怎麼處理設限 | 出處 |
|:--|:--|:--|:--|
| 以已存活時間為條件的 GBDT（資料擴增） | **LAVA**（Google VM 壽命，已上線）：每筆訓練樣本複製成 Tu＝12.5%、25%…的多筆，預測 E(Tr\|Tu)；壽命取 log10、封頂 7 天 | 用擴增把迴歸變成存活模型 | 〔原文 p.5、p.16〕 |
| 同上，模型比較 | LAVA Table 4：GBDT 迴歸 C-index 0.84、tree-based Cox（XGBoost）0.78、NN 迴歸 0.73、分層 Kaplan–Meier 0.73、線性 Cox 0.52 | — | 〔原文 p.16〕 |
| 重新預測 vs 一次預測 | LAVA：「重新預測比一次預測有根本性的提升」；上線後空主機 +2.3–9.2 pp | — | 〔原文 p.1〕 |
| 分位數迴歸 → S(t) | Copilot：12 個 pinball loss 的 LightGBM 組成存活曲線，條件更新 S(t)/S(t₀) | 原文未提 | 〔原文 p.17〕 |
| 經驗存活曲線 → 條件期望 | CacheWise（E[剩餘]）、Continuum（經驗 CDF→TTL）、SAECache（log-normal） | 工具一定會結束，沒有設限問題 | 〔原文〕見 B.3 |
| 競爭風險（命中 vs 被逐出） | LHD：P[H>a]/P[L>a] | 被自己的策略逐出＝競爭事件；explorer 物件減少選擇偏差 | 〔原文 p.6–7〕 |
| 存活分析的一般 loss | Cox partial likelihood（DeepSurv：Cox 比例風險的深度網路）、離散時間 hazard 的最大似然（Nnet-survival）、DeepHit（直接學存活時間分布、處理競爭風險）、Weibull 時間到事件（WTTE-RNN 系列） | 都把右設限放進似然 | 〔摘要〕（arXiv 1606.00931、1805.00917；AAAI'18 DOI 10.1609/aaai.v32i1.11842；WTTE 只查到後續論文）|
| 成對排序 loss（RankNet 型） | HALP（成對 BCE）、ForesightKV（成對排序，E06） | 任一方先發生就有標籤 | 〔原文 p.6〕、E06 |
| 列表排序 loss（LambdaRank／NDCG 型） | Parrot（可微 NDCG，相關度＝重用距離；比 log-likelihood 好） | Belady 視窗外 ∞ | 〔原文 p.5、p.8〕 |
| 排序 vs 迴歸 | GL-Cache：ranking objective 和 L2「沒有明顯差別」；LRB：二元分類、未取 log 的時間都比 log 時間差 | — | 〔原文 p.7〕、〔原文 p.8〕 |

〔判讀〕直接用 Cox／DeepHit 的系統論文我只找到 LAVA 的比較，而且輸給「以已存活時間為條件的 GBDT 迴歸」。這和我們 9 月「GBDT 很強」的經驗一致；第三階段的 loss 比較應該把「LAVA 式擴增＋L2」列為一個獨立候選。

### B.7 serverless 的 keep-alive／pre-warm（和「KV 留 CPU 多久、何時從 SSD 預取」同構）

| | Serverless in the Wild（ATC'20） | FaasCache（ASPLOS'21） |
|:--|:--|:--|
| 決定什麼 | 每個 app 的 **pre-warm 視窗**（執行完多久後重新載入）與 **keep-alive 視窗**（載入後留多久） | 容量滿時逐出哪個 warm container |
| 用什麼 | 每個 app 一個 **1 分鐘一格、最長 4 小時**的閒置時間直方圖 | Greedy-Dual：Priority＝Clock＋Freq×Cost／Size |
| 怎麼從分布變成決策 | 頭＝**第 5 百分位**→ pre-warm；尾＝**第 99 百分位**→ keep-alive；再各留 10% 餘裕 | 逐出優先權最低的；逐出時把 Clock 設成它的優先權 |
| 退路 | 直方圖不具代表性（樣本太少、bin 計數的 CV 低於門檻）→ 標準 keep-alive（pre-warm＝0、keep-alive＝直方圖範圍）；超出 4 小時的很多 → ARIMA 預測下次閒置，留 15% 餘裕 | 不是預測型，沒有退路的問題 |
| 為什麼是「急著放」 | FaaS 按配置時間計費，所以主動卸載，不等別人要空間；作者說最接近「新存取會重設 TTL 的 TTL 快取」，pre-warm 相當於「時間上的預取」 | **work-conserving**：容量滿了才逐出 |
| 評估 | 模擬（Azure Functions 一週）：**第 3 四分位 app 的冷啟動 %** vs **正規化的浪費記憶體時間**，畫 Pareto；固定 keep-alive 10 分鐘→1 小時，75 百分位 app 冷啟動 50.3%→25%；另在 OpenWhisk 實作 | 和 OpenWhisk 10 分鐘 TTL、重現的 HIST（作者稱為「TTL＋Prefetching」）比；冷啟動開銷降 3 倍以上 |
| 出處 | 〔原文 p.8–10、p.13〕 | 〔原文 p.1、p.5、p.10〕 |

〔判讀〕三個對應：
- CPU 層像 FaasCache：容量滿了才降級（work-conserving），不是按時間付費；所以 Shahrad 式「時間到就放」只適用在**慢層的預取時機**，不適用在 CPU 的降級。
- 「5 百分位 pre-warm」對應「預取提前量取回來時間分布的低分位數」，需要的就是 S(t)，不是點估計。
- 兩者評估都是**兩個目標的 Pareto**（冷啟動 vs 浪費的記憶體時間）；我們對應的是「回來的 TTFT vs 預取／慢層位元組」，應該畫 Pareto，而不是只報一個數字。

---

## C. 新穎性問題

**問題 1**：有沒有人研究「讀取端自適應（Cake 類重算＋載入）會不會縮小回來時間預測的價值」？

**答案：查不到。** 範圍：
- 檢索：WebSearch 3 次（預測價值×重算×載入×預取、多層 KV×預測×還原、2026 年 agent trace×預取×預測器）；arXiv 與 HF 的關鍵字查詢；本文讀過的 KV 論文的相關工作節（CacheWise §7、Continuum §7、PBKV §7、Ask-the-Tool §2.1 Table 1、Copilot §10、TraceLab §8、CacheScout Table）。
- 沒有一篇把 Cake 式「兩頭會合」的還原當成變數，去量 oracle／預測器的空間怎麼變。Cake 本身（D6_Cake 卡）沒有預測。

**最接近的間接證據**（都是「還原變便宜 → 預測的價值變小」的同一個方向）：

| 論文 | 比較了什麼 | 結果 | 出處 |
|:--|:--|:--|:--|
| CacheWise | 逐出後重算 vs 逐出後搬到 CPU 再搬回 | 前者 session 完成時間快 2.7–3.5 倍；後者只快 1.19 倍。原文：「PCIe 比重算快得多，每次逐出的懲罰較小，改善空間也相應變小」 | 〔原文 p.9、p.11〕 |
| Ask-the-Tool | 只有 HBM vs HBM＋DRAM | oracle：−27.7% → −22.6%（p90 TTFT）。CacheWise（叢集學習）／TokenCake（宣告時長）：**+22.7%／+22.4%（比 LRU 差）→ −14.9%／−16.0%（比 LRU 好）**。原文：有 host 層時，放錯「只多付一次重載」 | 〔原文 p.12–13〕 |
| Continuum | 有／沒有 LMCache 快速卸載 | 重載便宜時 InferCept 的 preserve 很少觸發；但「就算瞬間重載，回來的請求還是要等 GPU 記憶體」，排隊才是主要成本 | 〔原文 p.5〕 |
| WhereKVLive | 預取（含 oracle）vs 不預取 | 頻寬被爭用時 oracle 預取都沒贏 | A_WhereKVLive 卡 |

**問題 2**：在多層 KV（含重算）裡，有沒有人比較「預測放在寫入時 vs 淘汰時 vs 預取時」？

**答案：查不到（多層 KV 範圍內）。** 範圍同上，加上 D4 §5 前作表與 E05、E06 的 10 篇。已知的部分：
- 第一階段的「延後測試」是我們自己的（D4 §4.6：同一份知識寫入時用和淘汰時用結果相同）。
- **跨領域的「一次預測 vs 重新預測」**：LAVA 在 VM 排程上明確比了，重新預測勝出〔原文 p.1〕。Copilot 與 CacheWise 都把預測做成「過了 t₀ 就更新」（S(t)/S(t₀)、每 3 個迭代重建堆），但**都沒有做「只預測一次」的對照**〔原文 Copilot p.17、CacheWise p.8〕。
- EfficientAgent 決定在「第一次查 prefix 時」（接近寫入時），沒有和淘汰時版本比（A_EfficientAgent 卡）。
- 〔判讀〕所以 P3-5 的 Q3（寫入時預測一次 vs 決定當下重新預測）在 KV 領域沒有前作；LAVA 是可以引用的跨領域先例。

---

## D. 對第三階段設計的含意〔判讀〕

以下全部是判讀。依據是 A–C 的事實，事實的出處在各節。

1. **trace 的分工**：
   - **主 trace：FreeInference agentic trace**。滿足 explore/README 門檻（≥1,000 session、真實間隔）、長度落在 8K–128K 的公開資料還有 TraceLab 與 Copilot，但只有它同時有 16-token 前綴 block，可以直接接 D4 的 block 模擬。第一步是規則 6 的斷言（我在第 1 週驗過一條）和「session 重建」的驗算（README 說是用 tool-call id 串起來的）。
   - **第二主 trace：Copilot 2026**。量大（30 萬 session），但沒有 block，只能用 token 數模擬，沒有前綴共用。適合 X2（訊號在不在）與跨 trace 測試。
   - **短 context 對照：Bailian、WildChat、BurstGPT_3**。Bailian 的「回來」有乾淨的父子鏈；WildChat 有文字，是 P3-7 唯一的內容來源，而且**不需要使用者在 HF 同意條款**（README §4 寫的「可能要同意條款」不成立）。
   - **Mooncake 只當敏感度**：會不會回來的比例依重建規則差 2 倍（20% vs 44%），用它當主訓練資料等於把重建規則學進去。
   - 不要用：LMSYS、Arena、ShareGPT（沒有逐輪時間）、Azure、Chutes（沒有 session）。

2. **label 要分三類，不是兩類**：「H 內回來」「H 之後才回來」「視窗結束前沒回來（右設限）」。LRB 刻意讓後兩者同分，因為單層快取只有「留／踢」；我們有「留 CPU／降慢層／丟掉」，降慢層和丟掉的差別正好是這兩群。對應的 loss：存活似然（Cox、AFT、離散 hazard，正確處理設限），或 3L-Cache 式補值把兩群拉開；**不要用 LRB 式「補成 2×視窗」**，那正是 9 月 B5 的失敗模式。Bailian thinking 檔的設限有多嚴重可以直接看：前半段 16.2% 會回來、最後 600 秒只有 4.3%。

3. **把「以已閒置時間為條件」做進模型**：LAVA（擴增 Tu）、Copilot（S(t)/S(t₀)）、CacheWise（E[τ|已過 t]）三種做法都讓同一個模型在任何時刻重新回答。這有兩個好處：(a) P3-5 的 Q3 不用訓練兩個模型，只要比「t₀＝回合結束」和「t₀＝決定當下」；(b) 降級（要順序）、丟掉（要「不回來」的機率）、預取（要提前量＝低分位數）都從同一條 S(t) 讀出來。LAVA 的表顯示「條件 GBDT＋L2」贏過 Cox 類，所以 P3-4 的候選裡應該加一個「LAVA 式擴增＋L2」，和 L3（存活）並列。

4. **預取要保守，而且要先證明它有用**：三篇（PBKV、CacheScout、WhereKVLive）都發現預取的增益很小或會反噬；只有在「預測很準」（PBKV 雜訊 ≤10%）或「資訊確定」（CachedAttention 的排隊佇列、Ask-the-Tool 的進度回報）時才明顯。D3 預取的設計：只用閒置頻寬與空位、不為預取趕走東西（PBKV 的保守原則），提前量用 S(t) 的低分位數（Shahrad 的 5 百分位），而且在 P3-3 裡單獨開關，看它自己值多少。

5. **Cake 可能同時縮小 oracle 空間和「預測錯的代價」**：Ask-the-Tool 的數字（加 DRAM 層後 oracle 從 −27.7% 變 −22.6%，CacheWise 式預測器從比 LRU 差 22.7% 變成比 LRU 好 14.9%）和 CacheWise 的數字（重算 2.7–3.5 倍 vs 搬 CPU 1.19 倍）方向一致：還原越便宜，猜對的好處越小、猜錯的壞處也越小。Cake 讓「放錯層」更便宜，所以 Q1b 很可能得到「Cake 縮小 headroom」。這不是壞消息：它可能就是論文的結論（讀取時自適應讓預測變得不重要），而且前作沒人量過。P3-3 的報告應該同時列 (i) oracle headroom 和 (ii) 「加雜訊的 oracle」比 LRU 差多少，兩者在 Cake 與只載入下各一組。

**其他次要含意**
- **退路**：最便宜的保險是 Chłędowski 式「跟隨預測或 LRU、誰差就換」；CacheScout 式「分數乘近期衰減、預測變平就自然退回 LRU」不需要偵測器。兩者都比 LARU 簡單，可以先用。
- **評估要畫 Pareto**：serverless 兩篇都用「冷啟動 vs 浪費的記憶體時間」；我們對應的是「回來 TTFT vs 慢層寫入／預取位元組」。
- **切分**：Copilot 用「這一週訓練、下一週評估」；FreeInference 也是逐週釋出，正好可以照做（session 不跨週）。同一使用者會出現在多週，跨使用者的泛化要另外切。
- **工具時長預測有天花板**：Ask-the-Tool 顯示長工具呼叫的時長由執行環境決定，呼叫前的估計排不出順序（Kendall τ 約 0）。如果主 trace 是 agent，回來時間的主要不確定性來自工具與人，**人的停頓**（回合邊界，Copilot 中位 25 分鐘）才是降級／丟掉決策的主戰場；回合內的工具間隔（中位幾秒）多半用不到慢層。

---

## 附錄 A：我自己算的數字（可重現）

腳本與輸出都在 `docs/phase3_20261017/research/calc/`；輸入檔的 sha256 在 `calc/input_sha256.txt`。輸入檔放在 scratchpad，沒有進 repo。

| 輸出 | 指令 | 輸入（來源、版本） |
|:--|:--|:--|
| `calc/bailian_thinking.txt` | `python3 -I calc/bailian_sessions.py qwen_thinking_blksz_16.jsonl` | `https://media.githubusercontent.com/media/alibaba-edu/qwen-bailian-usagetraces-anon/main/qwen_thinking_blksz_16.jsonl`（commit `5f7439c5`，LFS oid `41ac36d9…`） |
| `calc/mooncake_conversation.txt`、`calc/mooncake_toolagent.txt` | `python3 -I calc/mooncake_chain.py <檔>`；`python3 -I calc/mooncake_chain2.py <檔>` | `https://raw.githubusercontent.com/kvcache-ai/Mooncake/1d0e4c75/FAST25-release/traces/<conversation|toolagent>_trace.jsonl` |
| `calc/freeinference_week1.txt` | `/ssd8/hungwei/paper-hkv/venv/tiara/bin/python -I calc/fi_stats.py <目錄>` | `https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace/resolve/16d3c04d/release/2026-05-17_2026-05-23/processed/{requests,sessions,tool_calls}.parquet` |

百分位數用線性內插（Bailian、Mooncake 是自己寫的 `pct`，FreeInference 用 numpy 預設）。Mooncake「嚴格」規則的門檻（共用前綴最多 3 個請求）是我隨手定的，沒有調過；它只用來說明歧義的大小。

## 附錄 B：讀了什麼（來源清單）

| 代號 | 版本 | URL | 讀的範圍 |
|:--|:--|:--|:--|
| CacheWise | arXiv 2606.16824v1 | https://arxiv.org/abs/2606.16824 ；資料 https://github.com/cachewise-project/cachewise-coding-traces （`181c435a`） | 全文 p.1–12；repo 的 manifest、一個 events.json、`tool_duration_prediction/` 全部程式 |
| Continuum | arXiv 2511.02230v7（2026-09-08） | https://arxiv.org/abs/2511.02230 ；https://github.com/Hanchenli/vllm-continuum | p.4–7、p.11–12；repo 檔案樹 |
| Copilot 2026 | arXiv 2608.00101v1 | https://arxiv.org/abs/2608.00101 ；https://github.com/Azure/AzurePublicDataset/blob/master/GitHubCopilotCodingAgentDataset2026.md | p.16–18；md、DATASET_CARD、schema.json、release 資產 |
| FreeInference | HF sha `16d3c04d` | https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace ；hub https://data.freeinference.org/ | README；第 1 週三個 parquet；論文「From Requests to Sessions」**沒讀**〔未查證〕 |
| Chutes | HF sha `85722e69` | https://huggingface.co/datasets/harvardMadsys/chutes_metadata | first-rows、HF API；論文 arXiv 2608.13573 沒讀 |
| PBKV | arXiv 2605.06472v1 | https://arxiv.org/abs/2605.06472 | p.1–6、附錄 E（p.17–18）、p.19 |
| CacheScout | arXiv 2608.14624 | https://arxiv.org/abs/2608.14624 | p.2、p.6–9 |
| Ask-the-Tool | arXiv 2609.18849v1 | https://arxiv.org/abs/2609.18849 | p.1–5、p.10、p.12–13 |
| TraceLab（論文） | arXiv 2606.30560v2 | https://arxiv.org/abs/2606.30560 | p.13–15 |
| MORI | arXiv 2606.00866v1 | https://arxiv.org/abs/2606.00866 | 只用 A_MORI 卡 |
| KVCache-in-the-wild | arXiv 2506.02634v5 | https://arxiv.org/abs/2506.02634 | p.10–11（WA 機制） |
| Bidaw | FAST'26 USENIX 版 | https://www.usenix.org/system/files/fast26-hu-shipeng.pdf | p.7–8 |
| KVFlow | arXiv 2507.07400 | https://arxiv.org/abs/2507.07400 | p.1–2、p.6 |
| InferCept | arXiv 2402.01869 | https://arxiv.org/abs/2402.01869 | p.7 §4.4 |
| LRB | NSDI'20 | https://www.usenix.org/system/files/nsdi20-paper-song.pdf | p.6–8 |
| LHD | NSDI'18 | https://www.usenix.org/system/files/conference/nsdi18/nsdi18-beckmann.pdf | p.6–7 |
| GL-Cache | FAST'23 | https://www.usenix.org/system/files/fast23-yang.pdf | p.6–7、p.9 |
| 3L-Cache | FAST'25 | https://www.usenix.org/system/files/fast25-zhou-wenbin.pdf | p.6 |
| HALP | NSDI'23 | https://www.usenix.org/system/files/nsdi23-song-zhenyu.pdf | p.5–6 |
| CACHEUS | FAST'21 | https://www.usenix.org/system/files/fast21-rodriguez.pdf | p.5 |
| Parrot | ICML'20；arXiv 2006.16239 | https://arxiv.org/abs/2006.16239 | p.4–5、p.8 |
| Glider | MICRO'19 | https://www.cs.utexas.edu/~lin/papers/micro19c.pdf | p.2–3 |
| Raven | CoNEXT'22，DOI 10.1145/3555050.3569134 | https://par.nsf.gov/servlets/purl/10424505 （連不上） | **只有摘要**（Semantic Scholar API） |
| Lykouris & Vassilvitskii | arXiv 1802.05399（2020 版） | https://arxiv.org/abs/1802.05399 | p.2、p.8、p.10 |
| Chłędowski et al. | ICML'21（PMLR 139）；arXiv 2106.14693 | https://arxiv.org/abs/2106.14693 | p.1、p.3 |
| SPO+ | arXiv 1710.08005v5 | https://arxiv.org/abs/1710.08005 | p.1 |
| LAVA | arXiv 2412.09840v2 | https://arxiv.org/abs/2412.09840 | p.1、p.5、p.16；venue〔未查證〕 |
| Serverless in the Wild | ATC'20 | https://www.usenix.org/system/files/atc20-shahrad.pdf | p.8–10、p.13 |
| FaasCache | ASPLOS'21 | https://homes.luddy.indiana.edu/prateeks/papers/faascache-asplos21.pdf | p.1、p.5、p.10 |
| DeepSurv、Nnet-survival、DeepHit | arXiv 1606.00931、1805.00917；AAAI'18 | — | 只有摘要 |

## 附錄 C：未查證清單

- Raven 的 loss 與是否處理右設限（全文拿不到）。
- FreeInference 論文「From Requests to Sessions」本文；session 重建規則的錯誤率；第 2–16 週的統計（只算了第 1 週）。
- Copilot trace 的長度分布、設限比例、單位斷言（沒下載）；論文的預測器是否處理設限（原文未見）。
- Chutes 論文（arXiv 2608.13573）；`user_id` 輪換的確切規則（只看到 hub 頁的一句話與 `rehash_round` 欄位）。
- `nebius/SWE-rebench-openhands-trajectories` 是否有時間欄位（first-rows 截斷）；SWE-bench leaderboard 軌跡的實際時間格式。
- LARU、SAECache 的 MLP、3L-Cache、GL-Cache、WA 的退路細節。
- LFO 的原文（只看到搜尋摘要）。
- WTTE-RNN 原始出處（只查到後續論文）。
