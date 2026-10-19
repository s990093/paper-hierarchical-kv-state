# X6：真實系統怎麼做降級／丟／預取、在哪裡出事（讀程式碼＋文件＋issue）

> 完成於 2026-10-17（`date -u` 見 §3）。§2 判準寫於 **2026-10-17T09:38:24Z**，在任何 clone、任何 issue 搜尋之前；之後沒有改判準。
> 方法：只讀原始碼、官方文件、GitHub issue／PR／討論。沒有跑任何 clone 下來的程式，沒有用 GPU，沒有在 GitHub 上留言或按任何東西（全部 `gh search`／`gh api` 唯讀）。
> 標記：〔程式碼〕＝我打開原始碼核過的行；〔文件〕＝官方文件；〔回報〕＝issue／PR／留言裡別人寫的東西（**不是我們量的**，數字照抄、不背書）；〔判讀〕＝我的推論。
> 本文**沒有任何我們自己的量測數字**；所有數字都是別人回報的，或是程式碼裡的預設值。

---

## 0. 一句話結論＋判定

**判定：有看頭**（照 §2 字面）——但〔判讀〕**有看頭的是「介面」問題，不是「預測」問題**。

- 照 §2.2 的定義，符合「需要回來資訊才能解、而且該系統現有旋鈕解不了」的獨立回報有 **7 組**（門檻 3）。每一組都有具體症狀或情境，例：
  - 生產環境、4×RTX 3090、10–20 個 coding agent：「1–5 s 的空檔在壓力下也會被 LRU 趕掉」「在佇列裡等超過一分鐘的那一輪幾乎全部 miss（56 個裡 1 個命中）」；proxy 沒辦法告訴引擎要留哪些 block〔回報 vllm#37003 mgabor3141, 2026-10-09〕。
  - 模擬（llm-d 模擬器）：只有在「回來間隔 ≥ LRU 自然留存時間」的區域，router 的 retention 才贏 LRU（間隔/留存 ≥ 4 時 LRU 0/480、retention 約 25%）；**SLRU 這種較聰明的本地策略拿到 0%**；而且 TTL 是設成「預測間隔 × k」〔回報 vllm#37003 k-cross, 2026-07-27〕。
  - 引擎只在請求到達時才開始從 L3 還原；orchestrator 在工具呼叫期間其實已經知道下一輪的前綴。提前還原後，空檔 ≥ 500 ms 時續接 TTFT 中位數 577 → 166 ms（L4、Qwen2.5-1.5B、4K token，n＝3）〔回報 sglang#42434, 2026-10-03〕。
- **但幾乎全部是 P-hint**：需要的資訊（工具還在跑、session 還沒結束、下一輪前綴）是 **agent 框架／orchestrator 自己就知道的**，缺的是把它傳給快取層的介面。真正「沒有人知道、只能猜」（P-pred）的只有 **1 組**（k-cross：TTL＝用工具時長平均與變異數預測的間隔 × k），而且是「TTL 要設多長」這種**設參數**的問題；歸 P-hint 的 mgabor3141 也指出 duration 要蓋過（未知的）佇列等待。
- **判定對嚴格度敏感**：只算有量測／觀察的仍有 4 組（有看頭）；只算生產環境觀察只剩 1 組、只算 P-pred 也只有 1 組（都會變「可能」）。見 §4.3。
- **沒有找到任何一條「人類使用者的多輪聊天（沒有 agent 框架）因為不知道誰會回來而出事」的回報**（查詢範圍見 §3.2；查不到不等於沒有）。
- 業界正在做的方向就是介面：vLLM 的 KV hints envelope 已合併（#53423，2026-09-21，只有傳遞、沒有動作）、retention API（#37003／PR #38514）還沒合併；SGLang 已有 session 感知淘汰（`--enable-session-radix-cache`，2026-08-02 合併）、streaming session（KV 不被淘汰，直到 close 或閒置 timeout）、priority 淘汰、T-LRU；Mooncake 有 soft pin（預設 30 分）／hard pin；TRT-LLM 有逐 token 區段的 priority＋`duration_ms`；Dynamo 有 `agent_hints`、`speculative_prefill`、session control，KVBM v1 正在被 KVCR 取代。
- **機制面最重要的共同點**〔程式碼〕：七個系統的「回來時」都是**請求到了才查、才搬**（vLLM：排程迴圈在 token budget 內才 lookup；SGLang：進等待佇列才發 L3 預取；LMCache：lookup 時才 async prefetch；Mooncake：engine 來 get 才讀）。沒有一個系統在請求到達前主動預取（只有未合併的 PR／實驗端點）。而且全部是「從頭開始的最長前綴命中」＋「同一序列內尾端先淘汰」（vLLM、LMCache 明寫反序 touch；SGLang／TRT-LLM／Dynamo lineage 只淘汰葉節點）。

---

## 1. 問題與要證明什麼

- **問題**：第三階段要找「延後設計下，還剩什麼值得預測」。X6 從真實系統這邊看：(1) 現在的系統在降級、丟、預取時依什麼決定、當下拿得到什麼資訊、有哪些旋鈕；(2) 2025–2026 有沒有人回報「多輪／agent 的 KV 太早被趕、預取太晚、回來 TTFT 暴增、想要 TTL／pin」，而且**需要知道未來回來時間才能解**、現有旋鈕解不了。
- **要證明的**：若真實系統的痛點都是「旋鈕就能解」或「是 bug／設定」，第三階段的預測器就沒有現實動機（死路）。若痛點需要回來資訊，要再分清楚：資訊是「框架本來就知道」（介面問題）還是「沒人知道、得猜」（預測問題）。
- **和前作的分工**：H0（10/09）讀過五個系統的**寫入／阻塞**路徑；D2、F1 量過 vLLM 0.28 OffloadingConnector 的丟資料與變慢。本文不重做，只補「淘汰順序、預取時機、決定當下的資訊、旋鈕」與 issue。學術文獻由 `docs/phase3_20261017/research/` 的 agent 負責。

---

## 2. 判準（開跑前寫）

**寫於 2026-10-17T09:38:24Z（`date -u`）**。寫的時候還沒 clone 任何 repo、還沒跑任何 issue 搜尋；只讀過 repo 內的 H0、D2、F1、D4、RELATED_WORK_WEAKNESSES。README §2 X6 的判準照抄，下面只把字詞定義寫細（只能更嚴，不能更鬆）。

### 2.1 README 的判準（照抄）
- **有看頭**：至少 3 個不同系統或不同的獨立回報，指出一個需要「預測回來」才能解的問題，而且現有旋鈕解不了。
- **死路**：找不到這種回報，或現有旋鈕（例如 TTL、pin）就解掉了。
- 其他＝**可能**。

### 2.2 字詞定義（更細、更嚴）
1. **回報**＝GitHub issue、discussion、PR 或 RFC 的描述，或官方文件／部落格裡寫明的已知問題。必須描述一個**觀察到的症狀或具體使用情境**（例：多輪對話回來時 TTFT 暴增、agent 的 KV 在工具呼叫期間被趕掉、預取來不及）。只有功能清單、沒有症狀或情境的不算。時間範圍：2025-01-01 到 2026-10-17。
2. **獨立**＝不同的作者（GitHub 帳號），而且不是同一件事的重複回報（被標 duplicate 或互相連結成同一件事的，算一件）。同一個作者在不同 repo 開的同一件事算一件。
3. **需要「預測回來」才能解**（代號 **P**）＝症狀的根因是「系統做降級／丟／預取決定時，不知道這個 session（或這段 prefix）會不會回來、什麼時候回來」，而且修好它需要回來時間或回不回來的資訊。再分兩種，分開計數：
   - **P-hint**：這份資訊應用端／agent 框架**自己知道**（例如工具呼叫時長、對話還沒結束），只要有 API 傳給快取層就能解。
   - **P-pred**：沒有人明確知道，只能靠系統預測（例如人類使用者何時回來）。
4. **現有旋鈕解得了**（代號 **K**）＝在回報當下或現在（2026-10-17 的 main／最新 release），該系統或同一部署裡可直接使用的設定或 API（TTL、pin、priority、容量、write policy、prefetch policy、eviction policy 選項）能讓回報者的情境不再發生，而且回報者或維護者沒有指出這個旋鈕不夠用。
   - 例外（仍算 P）：旋鈕存在，但它要求使用者事先給一個「回來時間」的數值（例如 TTL 秒數、pin 多久），而回報或討論指出這個值**隨 session／工具而變**、固定值會浪費容量或趕太早。這種標 **P（旋鈕要使用者自己預測）**，並另外註明。
   - 「固定 TTL／固定 pin 就夠」的回報算 K，不算 P。
5. **其他類型**（不算 P）：
   - **B（bug／實作缺陷）**：例如淘汰邏輯寫錯、漏記 hash、競爭條件、ref count 沒還。
   - **C（容量／設定）**：例如記憶體給太小、沒開 offload、選錯 backend。
   - **S（排程／頻寬）**：例如預取和需求讀取搶頻寬、預取太慢但根因是 I/O 路徑而非時機。
   - **U（看不出根因）**。
6. 一條回報只給一個主類別（P-hint／P-pred／K／B／C／S／U），由我照上面定義判，判斷理由寫在 `issues.csv` 的 `reason` 欄。看不出來就標 U，不硬分。
7. **判定的計數**：只數主類別為 P-hint 或 P-pred、且不是 K 的回報；要 ≥3 條獨立回報（跨系統或同系統都可，照 README「不同系統或不同的獨立回報」）。P-hint 和 P-pred 各自的數量也報告：**如果全部都是 P-hint，判定照字面仍可能是「有看頭」，但我會另外標〔判讀〕：這是「介面」問題，不是「預測」問題。**
8. 「找不到」的定義：照 §3 列出的查詢都跑過，仍沒有 P 類回報。查詢字串、命中數、看過哪些，全部寫進 `issues.csv` 與 run 目錄；**查不到不等於沒有**（CLAUDE.md 規則 7），所以「死路」要附上查詢範圍。

### 2.3 機制表要回答的欄位（每個系統一列）
- 降級（GPU→CPU、CPU→慢層）依什麼決定；丟（從最後一層淘汰）依什麼決定；預取（慢層→CPU／GPU）什麼時候觸發、依什麼決定。
- 決定當下拿得到哪些資訊：session／conversation id、回合結束、佇列中的請求、請求優先級、應用端提示。
- 旋鈕：TTL、pin、priority、prefetch timeout／policy、write policy、eviction policy、容量／水位。
- 每一項附 permalink（commit hash＋行號）或官方文件 URL；沒找到寫「沒找到（查了什麼）」。

### 2.4 不做的事
- 不執行任何 clone 下來的程式；不在 GitHub 上留言、開 issue、按任何東西。
- 不重做 H0／D2／F1 已經讀過的 vLLM 寫入路徑與阻塞行為，只補「淘汰、預取、資訊、旋鈕」。

---

## 3. 做了什麼

### 3.1 原始碼（run `20261017-173905-p3x6-src`）

`git clone --depth 1` 到 `/ssd8/hungwei/paper-hkv/tmp/p3x6/<repo>/`（每個 repo 自己的子目錄，不在 repo 根目錄），只讀不執行。指令在 run 目錄的 `cmd.sh`，commit 在 `stdout.log`：

| 系統 | repo | commit（日期） |
|:--|:--|:--|
| vLLM 0.28 | vllm-project/vllm tag `v0.28.0` | `2cf0a6915ce544dc493a0990f2ea38d81601128a`（2026-08-24） |
| vLLM main | vllm-project/vllm | `026f6ab2689397e64c1f53a5ac57d37c4dc46b67`（2026-10-17） |
| LMCache | LMCache/LMCache | `1567562908006129b48761ede368b59d4f3341e5`（2026-10-17） |
| SGLang | sgl-project/sglang | `80f40522cdc53a000577c57a9b0285c204e4c74b`（2026-10-17） |
| Mooncake | kvcache-ai/Mooncake | `d750cddafa4b62387ca135949595008c51e05815`（2026-10-17） |
| Dynamo | ai-dynamo/dynamo | `f24c59b936ba23c933cc35d2f8c7f9ff71f54c3d`（2026-10-17） |
| llm-d | llm-d/llm-d；llm-d/llm-d-kv-cache | `39b14fa06ad48ff91853d902cabaa4c8b91b8db7`；`822e6dd9c459632f34f872b4a5094766fbce2663` |
| AIBrix | vllm-project/aibrix | `065e50f7a25a60bba251dfe97c75c8de5aab72e0`（2026-10-17） |
| 額外：TRT-LLM（Dynamo KVBM 吃它的 retention priority） | NVIDIA/TensorRT-LLM 文件（`gh api` 讀單檔，沒 clone） | `e88d19ca3bda5bb761cf582e0379ed5397a7e60b` |
| 額外：KVCR（Dynamo 取代 KVBM 的新元件） | ai-dynamo/kvcr 文件（`gh api` 讀單檔） | `2a7ed995935f08b03150d250ac4bdd5a79cac395` |

### 3.2 issue 搜尋（run `20261017-175148-p3x6-issues`，第二批 `run `20261017-181123-p3x6-issues2``）

- 批次 1：11 個 repo × 28 個查詢字串（`gh search issues "<q>" --repo <r> --include-prs --created ">=2025-01-01" --limit 100`），每次間隔 2.6 s。repo：vllm、LMCache、sglang、Mooncake、dynamo、llm-d、llm-d-kv-cache、llm-d-router、aibrix、TensorRT-LLM、production-stack。查詢字串（節錄）：`multi-turn evict`、`agent kv cache`、`kv cache ttl`、`pin prefix`、`cache retention`、`session kv cache`、`evicted too early`、`prefetch timeout`、`prefetch ttft`、`offload ttft`、`keep kv cache`（全部在 `search_issues.sh`）。
- 批次 2：9 個 repo × 12 個補充字串（`tool call pause eviction`、`re-prefill`、`resume session ttft`、`cache thrashing`、`kv hints`、`retention directive`、`cache miss recompute multi-turn`、`hicache eviction`、`session_id kv cache` 等）。
- 每個查詢的命中數在 `results/p3_explore/x6/search_queries.csv`；合併去重後的所有命中在 `search_hits.csv`（1,335 條；去掉只被 `re-prefill` 命中的雜訊後 563 條 條）。
- 我先看標題篩出可能相關的，再用 `gh api` 抓本文與留言（原始 JSON 存在 run 目錄 `threads/`），逐條分類，結果在 `results/p3_explore/x6/issues.csv`（每列有 `run_id`、`ts`）。
- 追引用：從 RFC／PR 內的連結再追相關的 issue（例如 vllm#53421 → #51428 → #37003 → PR #38514 → sglang#27574）。追到的也寫進 `issues.csv`，`found_via` 欄寫來源。
- 指令與產出：`code/p3x/x6_collect.py`（只讀 run 目錄、寫三個 CSV），run ``20261017-213330-p3x6-collect``。

---

## 4. 結果

### 4.1 機制表（一個系統一列）

縮寫：「回來」＝同一個 session 的下一個請求到達。permalink 一律指到 §3.1 的 commit。

| 系統 | 降級／寫下層 | 丟（淘汰） | 預取（慢層→上層） | 決定當下拿得到的資訊 | 旋鈕 |
|:--|:--|:--|:--|:--|:--|
| **vLLM 0.28** OffloadingConnector＋TieringOffloadingSpec | GPU→CPU：每個新算好的 block 都寫（H0）；CPU→下層：**寫進 CPU 就複製到所有下層**〔程式碼 [V6]〕。單層時可設 `store_threshold`（≥N 次 lookup 才存）〔[V3]〕，**多層時不允許**〔[V8]〕 | CPU：`lru`（預設）或 `arc`，或自訂 `CachePolicy`〔[V1][V3]〕；只在要空間時淘汰 `ref_cnt==0` 的 block〔[V4]〕；touch 時**反序**，同一請求內尾端先被趕〔[V2]〕。fs 層**不淘汰、沒有容量上限**〔H0；文件 [LD1]〕。GPU prefix cache 也是反序釋放、尾端先趕〔[V11]〕 | **沒有到達前的預取**。下層→CPU 的 promotion 只在 `lookup` 時發起〔[V7]〕，而 lookup 只對「排程迴圈在 token budget 內處理到的等待請求」做〔[V10]〕；CPU 滿就 promotion 失敗＝MISS＝重算〔[V7]〕。命中＝從頭連續命中的 block 數〔[V9]〕 | `evict(n, protected)` 時只有 block 的新舊（ARC 多 ghost list）；touch／lookup 時有 `ReqContext`＝`req_id`＋`kv_transfer_params`（API 帶進來的 dict）〔[V5]〕。沒有 session id、沒有「回合結束後多久回來」 | `cpu_bytes_to_use`、`eviction_policy`、`cache_policy_module_path`（**外掛自訂淘汰策略，不用改 vLLM**）、`store_threshold`、`secondary_tiers`〔[V3][V15]〕。**沒有 TTL、pin、priority 淘汰** |
| **vLLM main**（2026-10-17） | 同上；多了下層 backpressure（下層有壓力就丟 store）〔[V17]〕 | 同上 | 同上 | `ReqContext` 多了 `kv_hints`（orchestrator 帶的版本化 envelope）〔[V12][V13]〕，**目前只有外部 KVCR 層會讀**〔[V14]〕；session_id 已能從 API 帶進 request（#48048） | KV hints 只有「傳遞」沒有「動作」（#53423）；retention API（priority＋duration）是 PR #38514，**未合併** |
| **LMCache**（in-process，已標 deprecated） | store 時同時寫所有後端（寫穿，H0） | CPU：`cache_policy`＝LRU（預設）／LFU／FIFO／MRU〔[L1][L2]〕，請求結束時反序 touch、尾端先趕〔[L3]〕；disk：LRU（H0） | `enable_async_loading`（預設關）時，lookup 時 async prefetch〔[L1][L9]〕；仍是請求到了才開始 | chunk key、recency；請求可帶 `lmcache.*`（`skip_save`、`skip_load`、`max_offload_tokens`、`tag`）〔[L4][L5]〕；request priority > `priority_limit` 就不存〔[L5]〕 | controller `pin`／`move`／`clear`／`compress` API，但 pin 會被 PinMonitor 在 `pin_timeout_sec`＝**300 s** 後強制解除〔[L1][L10]〕 |
| **LMCache MP 模式**（新架構） | L1→L2 store policy（預設全部寫到所有 adapter）〔[L7] 同目錄 store_policy.py〕 | L1 背景淘汰：用量 ≥ 0.8 就淘汰 20%，策略 LRU／ARC／IsolatedLRU（每個 cache_salt 一條 LRU）／noop〔[L6]〕 | prefetch policy：預設**從 L2 載入的資料用完就刪、不留在 L1**；`RetainPrefetchPolicy` 才留〔[L7]〕 | 同上＋cache_salt | L2 pin：`POST/DELETE /cache/pins`，**直到 unpin 為止、沒有 TTL**〔[L8]〕；PR #4456（未合併）要加每請求 retention TTL |
| **SGLang HiCache**（UnifiedRadixCache） | GPU→host：`write_through`（預設）／`write_through_selective`（命中 2 次才寫）／`write_back`〔[S1]〕；host→L3：L1→L2 完成後備份（H0） | `--radix-eviction-policy`：`lru`（預設）、`lfu`、`slru`、`priority`（請求 priority 低的先趕）、`tlru`（T-LRU：超出「下一輪 prefill 會超過 TTFT 預算」的尾端先趕，2026-09-17 合併 #34012）〔[S1][S2][S3]〕；**同一策略用在 GPU 和 host 的淘汰**〔[S4]〕；只淘汰葉節點（尾端先） | L3→host：**請求進等待佇列時**發起〔[S5]〕；停止條件 `best_effort`／`wait_complete`／`timeout`（預設 timeout：程式碼預設 1 s＋0.25 s／千 token，門檻 256 token〔[S6]〕；**文件寫 2 s＋0.1 s／千 token、上限 30 s，和程式碼不一致**〔[S9]〕）；host→GPU 在排程時載入 | 節點的 last_access、hit_count、priority（插入它的請求的 max）、T-LRU 的對話長度；開 `--enable-session-radix-cache` 時有 session 參照數〔[S4]〕；streaming session 有閒置時間〔[S8]〕 | `--enable-session-radix-cache`（應用每個請求帶 `session_id`、結束時 `/close_session`；有 session 參照的 KV 最後才趕，**軟保護、不是 pin**，2026-08-02 合併 #29173）〔[S1][S4]〕；`--enable-streaming-session`＋`open_session(timeout=…)`（KV 放在 session slot、**不參與淘汰**，直到 close 或閒置 timeout）〔[S8]〕；`priority`；`tlru` 的 `threshold`、`next_prompt_estimate`。HiRadixCache 的 TTL pin（#18941，2026-03 合併）已隨 HiRadixCache 移除（#40787，2026-09-22） |
| **Mooncake Store** | 引擎（vLLM／SGLang／LMCache connector）put 進來；`offload_on_evict`（預設關）可把寫 SSD 延到淘汰時〔[M2]〕 | 用量 ≥ **90%** 或配置失敗時淘汰 **5%**；近似 LRU：以 lease 到期時間排序（ExistKey／GetReplicaList 會給 10 s lease），有 lease 的不趕；soft pin 的最後才趕（`allow_evict_soft_pinned_objects` 預設 true）；hard pin 永遠不趕；group 一起趕〔[M1][M3][M4][M5]〕 | Store 本身沒有預取；`promotion_on_hit`（SSD→DRAM，預設關）〔[M6]〕；「exist 時就預取」是 PR #4272（未合併） | key、tenant、group id、soft／hard pin 旗標與期限；**沒有 session 語意**（RFC #2098、#2819 想加，#2819 已因閒置關閉） | `default_kv_lease_ttl`（10 s）、`default_kv_soft_pin_ttl`（**30 分**，讀取**不延長**）、`max_kv_soft_pin_ttl`（24 h）、請求級 `soft_pin_ttl_ms`、`with_hard_pin`、`eviction_high_watermark_ratio`、`eviction_ratio`〔[M1]–[M3]〕 |
| **NVIDIA Dynamo KVBM**（v1 正在 sunset，由 KVCR 取代〔[D10]〕） | G1→G2、G2→G3 offload policy：`pass_all`／`presence`／`presence_lfu`（`min_lfu_count`＝8）〔[D1]〕；v1 預設 G2→G3 頻率過濾（H0）；`DYN_KVBM_HOST_OFFLOAD_PREFIX_MIN_PRIORITY`〔[D1][D3]〕 | inactive pool：LRU、multi-LRU（TinyLFU 四級）、lineage（只趕葉）、FIFO〔[D2]〕；TRT-LLM 的 retention priority 用來過濾 offload〔[D4]〕 | onboard 在請求 match 時（`inter` async／`intra` layer-wise）〔[D1]〕；沒有到達前預取 | block hash、頻率、priority、lineage | 見 Dynamo 前端那列 |
| **Dynamo 前端／router**（給引擎的提示） | — | `nvext.agent_hints.priority` → SGLang `priority` 淘汰；vLLM、TRT-LLM「planned」〔[D5][D6][D9]〕 | `speculative_prefill`：回合結束後**馬上**把下一輪前綴（去掉 thinking）用 `max_tokens=1` 跑一次暖 KV〔[D5][D7]〕；session control `open/close`＝SGLang streaming session（不被淘汰、close 或閒置 timeout 釋放）〔[D6]〕；ThunderAgent：工具執行（ACTING）中的程式在記憶體壓力下暫停、放手讓引擎淘汰，之後小的先恢復〔[D8]〕 | session id、priority、osl、程式狀態（REASONING／ACTING） | router session affinity TTL；KVCR：`deposit(no_evict=True)` 硬保留、`no_retain` 提示、可插拔 `KVCachePolicy`（預設 LRU、同時間時後面的 key 先趕）；**`submit_hint` 不會主動預取**（「may be added later」）〔[D11]〕 |
| **llm-d** | 交給引擎（vLLM native／SGLang HiCache／LMCache／Mooncake）〔[LD1][LD2]〕 | 交給引擎；共享 fs 層**不淘汰**，要另外跑 PVC Evictor：用 atime 分冷熱（60 分內算熱）、用量 85% 開始刪、70% 停，**冷檔依發現順序刪（不是 LRU）**；`relatime` 會讓熱檔看起來是冷的〔[LD1][LD4]〕 | 交給引擎 | router 從 KV events 知道每個 pod／層有哪些 block；或用近似 LRU 索引估計〔[LD5]〕；turn-priority：依「這個 session 已經第幾輪」排佇列，讓前綴不在 session 中途被趕〔[LD3]〕 | `lruCapacityPerServer`、turn-priority 的權重×TTL（Beta）〔[LD2][LD3]〕 |
| **AIBrix KV cache** | L1（CPU）→L2：`HOT`（預設：L1 裡**第一次被命中**才寫 L2）／`ALL`／`EVICTED`；in-flight 上限 0＝同步寫〔[A1][A2][A3]〕 | L1：`S3FIFO`（預設；small FIFO 30%、命中 1 次升 main）、LRU、FIFO，預設容量 10 GB〔[A1]〕；RocksDB L2 TTL 600 s〔[A1]〕 | 載入第 i 個 chunk 時預取第 i+1 個（管線化，不是到達前）〔[A4]〕 | chunk key、頻率 | 上述 env；L2 每 token 逾時 20 ms〔[A1]〕 |
| **TRT-LLM**（額外） | GPU 淘汰前可複製到 host（`host_cache_size`，預設 0）；priority < `secondary_offload_min_priority`（35）的不搬、直接丟〔[T1]〕 | **prioritized LRU**（0–100，預設 35，低的全部趕完才趕下一級）；只趕葉〔[T1]〕 | 被重用時才搬回 GPU〔[T1]〕 | 每個 block 的 priority 與到期 | `KvCacheRetentionConfig`：逐 token 區段的 priority＋`duration_ms`（到期回到 35）、`decode_retention_priority`＋`decode_duration_ms`〔[T1]〕 |

**機制表的四個共同點**〔程式碼／文件，整理是我的〕：
1. **回來時才動作**：沒有一個系統（在已合併的程式碼裡）會在請求到達前把 KV 從慢層搬上來。最早的觸發點是 SGLang 的「進等待佇列」與 vLLM 的「排程迴圈處理到它」。到達前預取只存在於未合併的 PR／實驗端點（sglang#42434、Mooncake#4272）或 Dynamo 的 `speculative_prefill`（回合一結束就暖，但之後的空檔照樣可能被趕）。KVCR 文件明寫目前不做主動預取〔[D11]〕。
2. **淘汰只看過去**：預設全是 LRU（或 LRU 變體：ARC、S3FIFO、TinyLFU、lease 到期時間）。唯一用到「未來」的是 T-LRU 的 `next_prompt_estimate`（一個全域固定值）與各種「應用給的」priority／pin／TTL。
3. **最長前綴命中＋尾端先趕**：vLLM 的 offload lookup 從頭數連續命中〔[V9]〕；vLLM、LMCache 都刻意反序 touch，讓同一序列的尾端先被趕〔[V2][L3]〕（vllm#39021 寫得最清楚：「earlier prefix blocks end up as MRU … preserving KV cache prefix dependency」）；SGLang、TRT-LLM、Dynamo lineage 只趕葉節點。vllm PR #38514 甚至**拒絕**「後面 token 的 priority 比前面高」的 retention 指令，因為前面被趕掉後面就沒用〔回報〕。〔判讀〕這和 Cake 的「前段重算、後段載入」正好相反：Cake 想要的是後段還在；真實系統在容量不夠時先丟後段，而且 lookup 不允許「頭不在、尾在」的命中。
4. **資訊通道正在開，但大多只是管線**：vLLM `kv_hints`（只有傳遞）、`session_id`（#48048）；SGLang `session_id`＋`/close_session`、`priority`；Dynamo `agent_hints`、session control；Mooncake 的 soft pin TTL；TRT-LLM retention。**決定「TTL 要多長、priority 給多少」的都是應用端**，引擎不猜。

### 4.2 問題清單（issue／PR／RFC）

完整表在 `results/p3_explore/x6/issues.csv`（41 列）。下面只列分類為 P 的，與幾條最典型的非 P。日期＝建立日。

**P 類（需要回來資訊，現有旋鈕在該系統解不了）**：

| # | 群組（同一群只算一次） | 系統 | 連結（建立日） | 類別 | 證據 | 症狀（回報者自己的話／數字，非我們量的） | 現有旋鈕 |
|:--|:--|:--|:--|:--|:--|:--|:--|
| R01 | G-llmd：vMaroon（含 PR #38514、#48501、llm-d-kv-cache#395） | vLLM | [vllm#37003](https://github.com/vllm-project/vllm/issues/37003)（2026-03-13） | P-hint，duration 由 orchestrator 給 | 設計＋引用；PR #38514 有作者的 e2e | 工具呼叫暫停時 block 沒有引用，並發下被別的 agent 用 LRU 趕掉，回來時整段重算（RFC 說到 session 結束時 70K–200K token）。PR #38514 作者量到 server TTFT 比 LRU 少 43%／12%／12%（in-flight 1／4／8，gpt-oss-120b） | vLLM 沒有；#38514 未合併 |
| R04 | G-kcross | vLLM／llm-d 模擬器 | [#37003 留言](https://github.com/vllm-project/vllm/issues/37003#issuecomment-5087831234)（2026-07-27） | **P-pred** | 模擬 | LRU 自然留存 6.8 s（512 block／pod、qps 5）；間隔/留存 ≥ 4 時 LRU 0/480、router pin 約 25%；SLRU 與「升級提示」0%；pin 的 TTL＝預測間隔（工具時長平均與變異）× k | vLLM 沒有 |
| R05 | G-mgabor | vLLM 0.30（生產） | [#37003 留言](https://github.com/vllm-project/vllm/issues/37003#issuecomment-6090931030)（2026-10-09） | P-hint，duration 要蓋過佇列等待 | **生產流量** | 4×RTX 3090、10–20 個 coding agent、50–250K context；agent loop 91% 的下一輪 1 s 內到；池子有壓力時 1–5 s 的空檔也會被 LRU 趕掉；佇列等超過 1 分鐘的那一輪 56 個裡只有 1 個命中；「proxy has no way to tell the engine which blocks to keep」 | vLLM 沒有 |
| R08 | G-dynamo：karen-sy、ishandhanani（含 sglang#27574、#21045、vllm#53421／#59708、dynamo#13279、ThunderAgent 文件） | vLLM／SGLang | [vllm#51428](https://github.com/vllm-project/vllm/issues/51428)（2026-08-07） | P-hint | 設計 | 「local policy cannot tell a short tool gap from a long one」；「the workload knows a request will resume soon」；sglang#27574：引擎不知道「how long an external tool/subagent gap will last」 | vLLM：kv_hints 只有傳遞；SGLang：曾有 TTL pin（#18941），同一位維護者在 #21045 說 hard pin「wastes host memory」，之後隨 HiRadixCache 移除 |
| R12 | G-fermat | vLLM | [vllm#52113](https://github.com/vllm-project/vllm/issues/52113)（2026-08-13） | P-hint | 設計＋情境表 | 「vLLM cannot distinguish a temporarily idle but valuable agent context from a prefix that will never be used again」；等工具→短 TTL 保護；恢復→先預取 | vLLM 沒有 |
| R15 | G-mks | SGLang | [sglang#42434](https://github.com/sgl-project/sglang/pull/42434)（2026-10-03） | P-hint（**預取太晚**） | 作者的量測 | 請求到了才開始從 L3 還原，雖然 orchestrator 在工具呼叫時就知道前綴；提前還原後空檔 ≥ 500 ms 時續接 TTFT 中位數 577 → 166 ms（L4、Qwen2.5-1.5B、4K token、每格 n＝3） | 沒有（prefetch policy 只管「到達後」的預取何時停） |
| R38 | G-zhh | vLLM | [vllm#57103](https://github.com/vllm-project/vllm/issues/57103)（2026-09-16） | P-hint | 設計＋引用 AgentX 流量 | 「During a pause, the KV is often the most valuable prefix for the next request, yet it may have no active request reference and enter an ordinary LRU eviction order」 | vLLM 沒有 |

**典型的非 P（不算進判定）**：

| # | 連結 | 類別 | 為什麼不算 |
|:--|:--|:--|:--|
| R13／R14 | [sglang#24656](https://github.com/sgl-project/sglang/issues/24656)、[Mooncake#2098](https://github.com/kvcache-ai/Mooncake/issues/2098)（stmatengss） | P-hint | 「upcoming steps may be evicted too early」，但 SGLang 之後合併了 session 感知淘汰（#29173），討論裡沒人說它不夠 → 嚴格起見**不算** |
| R28 | [sglang#29173](https://github.com/sgl-project/sglang/pull/29173) | K | 「may evict an active session's prefix while unrelated, unreferenced KV remains cached」→ 用「session 還開著」（應用呼叫 `/close_session`）就解了，不需要回來時間 |
| R35 | [llm-d-router#2088](https://github.com/llm-d/llm-d-router/issues/2088)＋llm-d turn-priority 指南 | K | 指南的量測（回報者的）：並發 128、200 個深的 agent session，FCFS 0 個做完、p50 586.7 s；turn-priority 17 個做完、p50 37.8 s——「each one holds a growing prefix that keeps getting evicted」。解法是**用輪數排佇列**（固定參數），不是預測 |
| R27 | [sglang#34012](https://github.com/sgl-project/sglang/pull/34012)（T-LRU） | K | agentic trace 上 LRU 保護尾端、讓別的對話尾延遲多最多 43%（回報者的）；固定參數規則解掉 |
| R21／R36 | [LMCache#4456](https://github.com/LMCache/LMCache/pull/4456)、[TRT-LLM#16251](https://github.com/NVIDIA/TensorRT-LLM/issues/16251)（同作者） | K | 「this prompt's KV survives the next hour」＝付費固定 TTL 產品 |
| R17／R18／R20 | [vllm#47802](https://github.com/vllm-project/vllm/issues/47802)、[vllm#23083](https://github.com/vllm-project/vllm/issues/23083)、[sglang#36208](https://github.com/sgl-project/sglang/issues/36208) | K-wish | 靜態 priority／固定 pin（system prompt）就夠 |
| R39 | [vllm#50422](https://github.com/vllm-project/vllm/pull/50422) | K-wish | 「evicting one block in the middle forces recomputation of everything after it」→ 鏈狀結構的淘汰，與回來時間無關 |
| R22／R24／R25／R33 | sglang#32724、Mooncake#4272、Mooncake#2769、vllm#44294 | S | 預取完成要等 FCFS 掃到、exist→get 的空窗、promotion 排程、並發載入被序列化（vllm#44294 回報最多 12× TTFT） |
| R31／R32 | sglang#43738、vllm#48435 | B | 特定模型（DSv4、Gemma-4 SWA）的 prefix cache bug |

### 4.3 計數與判定

- `issues.csv` 共 41 列：P-hint 13、P-pred 1、K 5、K-wish 6、S 7、B 3、C 2、U 2、feature 2（`x6_collect.py` 的輸出，run `20261017-213330-p3x6-collect`）。
- **算進判定的獨立群組＝7**（每群一列，見上表）：P-hint 6（G-llmd、G-mgabor、G-dynamo、G-fermat、G-mks、G-zhh）、P-pred 1（G-kcross）。門檻 3 → **有看頭**。
- 其中 4 組（R01、R04、R05、R08）要的旋鈕是「priority＋duration／TTL」，**duration 由使用者給**——照 §2.2 第 4 條的例外仍算 P，並標「旋鈕要使用者自己預測」。支持「固定值不夠」的回報：R04（TTL 設成預測間隔 × k）、R05（要蓋過佇列等待）、R10／sglang#21045（hard TTL pin 浪費 host 記憶體）。
- 依系統：vLLM 6 組（R01、R04、R05、R08、R12、R38）、SGLang 2 組（R08 的 sglang#27574、R15）。Mooncake、LMCache、AIBrix、TRT-LLM、llm-d **沒有**符合的回報（那邊的回報是 K、K-wish 或 S）。
- **敏感度**（只報告，不改判定）：
  - 只算有量測／觀察的（去掉純設計 RFC）：R01（PR #38514 的 e2e）、R04（模擬）、R05（生產）、R15（量測）＝4 組 ≥ 3，仍是「有看頭」。
  - 只算生產環境觀察：只有 R05＝1 組 → 會變「可能」。
  - 只算 P-pred：1 組 → 會變「可能」。
- **沒找到**：人類使用者的一般多輪聊天（沒有 agent 框架）因「不知道誰會回來」而出事的回報（查了 `multi-turn evict`、`multi-turn eviction`、`multi-turn prefix cache`、`multi turn cache hit`、`conversation cache evicted`、`cache miss recompute multi-turn`、`resume session ttft`，11／9 個 repo；查不到不等於沒有）。多輪聊天相關的命中都是 B（特定模型 bug）或 C。

---

## 5. 前作

- H0（`docs/research_20261009_explore/H0_offload_blocking.md`）：五個系統的寫入／阻塞路徑。本文的「降級／寫下層」欄沿用它的結論，沒有重讀。
- D2、F1：vLLM 0.28 的丟資料率（0／133,504）、開 connector 換 attention backend、fs 層「等 83–89 s 不重算」。本文 §4.1 的「回來時才動作」和 F1 §4.7 的「等慢層而不是重算」是同一件事的兩面。
- D4 §5：KVFlow、Continuum、CachedAttention、LPC、Bidaw、WA 的機制（文獻）。本文找到的工業界提案大量引用 Continuum（TTL＝工具時長）與 KVFlow（workflow 感知）作為動機〔回報 vllm#37003、#38514〕。
- T-LRU（arXiv:2510.15152，NeurIPS'25）已經進 SGLang 主線（#34012）。它是「位置感知、固定參數、不預測回來時間」的淘汰規則；和我們第一階段「位置決策可以延後」的方向相關，交給 research agent 評估。〔我只讀了 SGLang 的 PR 與文件，沒讀論文原文〕

---

## 6. 偏離與問題

1. **被 API 用量上限中斷**：本機 21:33（13:33Z）才跑完收集。主 session 告知中斷發生在 18:2x（本機），恢復後從「讀 vLLM 四個 RFC／PR」接著做，沒有重讀已讀過的東西。總時間超過預算的 3 小時（含中斷）。
2. **查詢的限制**：`gh search issues` 每個查詢最多 100 筆。`vllm cache retention` 剛好 100（被截斷）。`re-prefill` 被拆成 `prefill`，7 個 repo 全部 100 筆、幾乎都是雜訊（772 條只被這個字串命中，§3.2 的 563 條不含它們）。GitHub Discussions、Slack、Discord、部落格留言都**沒有搜**。
3. **只讀了一部分**：大部分 thread 我讀了本文前 1,500–2,500 字＋留言；長留言串（vllm#37003、sglang#27574、dynamo#11673）讀了全部留言。原始 JSON 在 `20261017-175148-p3x6-issues/threads/`，`issues.csv` 的 `thread_file` 欄可以對回去。
4. **獨立性的判斷**：以 GitHub 帳號與互相引用判斷；看不到雇主。Dynamo 團隊（karen-sy、ishandhanani）的 vLLM／SGLang／Dynamo 提案算 1 組；vMaroon 與 PR #38514 共同作者算 1 組。k-cross 用的是 llm-d 模擬器，可能和 llm-d 團隊有關，但帳號與內容都不同，我算獨立——**這一條若合併，計數變 6，判定不變**。
5. **很多「回報」是廠商的 RFC**，不是使用者的 bug report；這是本題的主要弱點，敏感度見 §4.3。
6. **數字都不是我們量的**：§0、§4.2 的數字全部照抄回報者，沒有重現；R04 是模擬、R15 每格只有 n＝3。
7. **超出 README 列表**：加了 TRT-LLM（Dynamo KVBM 吃它的 retention priority）、KVCR（取代 KVBM 的新元件）、vLLM production-stack（只在搜尋裡）。
8. **SGLang 文件與程式碼不一致**：HiCache 預取 timeout 預設值（文件 2 s＋0.1 s／千 token、上限 30 s；程式碼 1 s＋0.25 s／千 token，搜不到 `prefetch_timeout_max`）。我以程式碼為準。
9. KVBM 的 v1 細節沿用 H0（10/09 的 commit），沒有對今天的 commit 重核；KVBM v1 已宣布 sunset。
10. vLLM 0.28 的行號是 tag 版本；main 的行號另標。permalink 的「#Lx–Ly」是我讀時記下的範圍，個別幾條（[V10]、[S5]）是兩個點，不是連續範圍。

---

## 7. 如果要繼續

〔判讀〕下面是我建議的順序，最值得追的是第 1 點。
1. **最值得追：「工具呼叫中」的 session，KV 該留在哪一層、留多久、何時開始搬回來。** 真實系統的回報幾乎都落在這裡：orchestrator **知道**「工具還在跑、session 還沒結束、下一輪前綴是什麼」，但**不知道還要多久**。現有／提案中的旋鈕都把「多久」丟給使用者（priority＋duration、TTL、soft pin 30 分）。k-cross 的模擬顯示只有「間隔 > LRU 自然留存」的區域才有價值，而且 TTL 要依工具時長調；sglang#42434 顯示空檔 ≥ 500 ms 時提前還原就能把續接 TTFT 從 577 ms 壓到 166 ms。這正好是第三階段的動作空間（GPU 留著／降到 CPU／降到慢層並在回來前預取／丟）加上一個條件存活問題：P(回來時間 > t | 工具名稱、歷史)。要接 X3（回來前的現成提示）與 X8（agent 工作負載）。需要有工具時長的 trace；Mooncake toolagent 有沒有工具時長要 X8 確認。
2. **模擬器要加「尾端先丟、只認最長前綴」**（X1、X5）：七個系統都是這樣。如果模擬器讓 Cake 在「頭還在慢層、尾被丟」時也能重疊，會高估延後設計。
3. **GPU 驗證（P3-6）可以不改 vLLM**：`eviction_policy` + `cache_policy_module_path` 可外掛自訂 `CachePolicy`，`ReqContext.kv_transfer_params`（main 還有 `kv_hints`）可帶每個請求的提示〔[V3][V5][V13]〕。限制：`evict()` 當下拿不到 request context，要在 touch 時自己記。
4. **追蹤**：vllm#38514（retention API）、#59714（G1 Retain）、#51428；SGLang #42434（到達前還原）；Mooncake #4272（exist 時預取）；KVCR 的主動預取。若其中任一合併，「現有旋鈕」的判斷要重做。

---

## 附：permalink 索引

- [V1] https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/v1/kv_offload/cpu/policies/factory.py#L85-L90 （內建 `lru`、`arc`）
- [V2] https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/v1/kv_offload/cpu/policies/lru.py#L43-L78 （touch 反序；evict 只挑 ref_cnt==0）
- [V3] https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/v1/kv_offload/cpu/spec.py#L118-L139 （`eviction_policy` 預設 lru、`cache_policy_module_path`、`store_threshold`）
- [V4] https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/v1/kv_offload/cpu/manager.py#L186-L201 （要空間時才淘汰）
- [V5] https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/v1/kv_offload/base.py#L91-L98 （`ReqContext`＝req_id＋kv_transfer_params）
- [V6] https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/v1/kv_offload/tiering/manager.py#L10-L20 （「Always offload to all tiers」「Staged promotion」）
- [V7] https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/v1/kv_offload/tiering/manager.py#L336-L452 （lookup 才發起 promotion；CPU 滿就 MISS）
- [V8] https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/v1/kv_offload/tiering/spec.py#L290-L293 （多層不支援 store_threshold）
- [V9] https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py#L599-L631 （從頭數連續命中）
- [V10] https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/v1/core/sched/scheduler.py#L751 、#L830 （等待佇列迴圈在 token budget 內才呼叫 connector lookup）
- [V11] https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/v1/core/kv_cache_manager.py#L570-L573 （「free the blocks in reverse order so that the tail blocks are evicted first」）
- [V12] https://github.com/vllm-project/vllm/blob/026f6ab2689397e64c1f53a5ac57d37c4dc46b67/vllm/v1/kv_hints/protocol.py#L9-L26 （KvHintsEnvelope）
- [V13] https://github.com/vllm-project/vllm/blob/026f6ab2689397e64c1f53a5ac57d37c4dc46b67/vllm/v1/kv_offload/base.py#L92-L106 （ReqContext 加 kv_hints）
- [V14] https://github.com/vllm-project/vllm/blob/026f6ab2689397e64c1f53a5ac57d37c4dc46b67/vllm/v1/kv_offload/tiering/kvcr/manager.py#L603-L613 （只有 KVCR 層 submit_hint）
- [V15] https://github.com/vllm-project/vllm/blob/026f6ab2689397e64c1f53a5ac57d37c4dc46b67/docs/features/kv_offloading_usage.md#L106-L110 （官方文件的旋鈕表）
- [V17] https://github.com/vllm-project/vllm/blob/026f6ab2689397e64c1f53a5ac57d37c4dc46b67/vllm/v1/kv_offload/tiering/backpressure.py#L62 （ThrottledDropPolicy）
- [L1] https://github.com/LMCache/LMCache/blob/1567562908006129b48761ede368b59d4f3341e5/lmcache/v1/config.py#L401-L430 （cache_policy LRU、enable_async_loading False、priority_limit）、#L594-L612 （pin_timeout_sec 300、pin_check_interval_sec 30）
- [L2] https://github.com/LMCache/LMCache/blob/1567562908006129b48761ede368b59d4f3341e5/lmcache/v1/storage_backend/cache_policy/ （lru／lfu／fifo／mru）
- [L3] https://github.com/LMCache/LMCache/blob/1567562908006129b48761ede368b59d4f3341e5/lmcache/v1/storage_backend/local_cpu_backend.py#L137-L142 （反序 touch）
- [L4] https://github.com/LMCache/LMCache/blob/1567562908006129b48761ede368b59d4f3341e5/lmcache/integration/vllm/utils.py#L131-L152 （`lmcache.*` request configs）
- [L5] https://github.com/LMCache/LMCache/blob/1567562908006129b48761ede368b59d4f3341e5/lmcache/integration/vllm/vllm_v1_adapter.py#L332-L339 、#L1658-L1663 （skip_save、priority_limit）
- [L6] https://github.com/LMCache/LMCache/blob/1567562908006129b48761ede368b59d4f3341e5/lmcache/v1/distributed/config.py#L484-L496 （MP 淘汰：LRU／ARC／IsolatedLRU／noop、0.8／0.2）
- [L7] https://github.com/LMCache/LMCache/blob/1567562908006129b48761ede368b59d4f3341e5/lmcache/v1/distributed/storage_controllers/prefetch_policy.py#L194-L330 （Default 不保留、Retain 保留）
- [L8] https://github.com/LMCache/LMCache/blob/1567562908006129b48761ede368b59d4f3341e5/lmcache/v1/mp_coordinator/schemas.py#L599-L616 （L2 pin 直到 unpin）
- [L9] https://github.com/LMCache/LMCache/blob/1567562908006129b48761ede368b59d4f3341e5/lmcache/v1/cache_engine.py#L1351-L1368 （async_lookup_and_prefetch）
- [L10] https://github.com/LMCache/LMCache/blob/1567562908006129b48761ede368b59d4f3341e5/lmcache/v1/pin_monitor.py#L28-L60 （PinMonitor 逾時強制解除）
- [S1] https://github.com/sgl-project/sglang/blob/80f40522cdc53a000577c57a9b0285c204e4c74b/python/sglang/srt/arg_groups/fields/memory.py#L25-L60 （radix eviction policy＋config）、#L86-L89 （session radix cache）、#L121-L127 （write policy）、#L170-L176 （prefetch policy）
- [S2] https://github.com/sgl-project/sglang/blob/80f40522cdc53a000577c57a9b0285c204e4c74b/python/sglang/srt/arg_groups/choices.py#L187 （lru／lfu／slru／priority／tlru）
- [S3] https://github.com/sgl-project/sglang/blob/80f40522cdc53a000577c57a9b0285c204e4c74b/python/sglang/srt/mem_cache/evict_policy.py#L41-L81 （Priority、TLRU）
- [S4] https://github.com/sgl-project/sglang/blob/80f40522cdc53a000577c57a9b0285c204e4c74b/python/sglang/srt/mem_cache/unified_cache/components/full.py#L201-L267 （session 參照排序，device 與 host 淘汰共用）
- [S5] https://github.com/sgl-project/sglang/blob/80f40522cdc53a000577c57a9b0285c204e4c74b/python/sglang/srt/managers/scheduler.py#L3269-L3317 、#L3407-L3420 （進佇列時 `_prefetch_kvcache`）
- [S6] https://github.com/sgl-project/sglang/blob/80f40522cdc53a000577c57a9b0285c204e4c74b/python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py#L394-L398 （預設 256／1 s／0.25 s per Ki）
- [S7] https://github.com/sgl-project/sglang/blob/80f40522cdc53a000577c57a9b0285c204e4c74b/python/sglang/srt/mem_cache/unified_radix_cache.py#L2297-L2326 （check_prefetch_progress）
- [S8] https://github.com/sgl-project/sglang/blob/80f40522cdc53a000577c57a9b0285c204e4c74b/python/sglang/srt/session/session_controller.py#L84-L110 （Session timeout）、https://github.com/sgl-project/sglang/blob/80f40522cdc53a000577c57a9b0285c204e4c74b/python/sglang/srt/session/streaming_session.py#L29-L45 （session slot）
- [S9] https://github.com/sgl-project/sglang/blob/80f40522cdc53a000577c57a9b0285c204e4c74b/docs/docs/advanced_features/hicache_design.mdx#L99-L116 （文件的 timeout 公式）、https://github.com/sgl-project/sglang/blob/80f40522cdc53a000577c57a9b0285c204e4c74b/docs/docs/advanced_features/radix_eviction_policy.mdx 、https://github.com/sgl-project/sglang/blob/80f40522cdc53a000577c57a9b0285c204e4c74b/docs/docs/advanced_features/session_radix_cache.mdx
- [M1] https://github.com/kvcache-ai/Mooncake/blob/d750cddafa4b62387ca135949595008c51e05815/mooncake-store/include/types.h#L93-L101 （lease 10 s、soft pin 30 min、max 24 h、0.05／0.90）
- [M2] https://github.com/kvcache-ai/Mooncake/blob/d750cddafa4b62387ca135949595008c51e05815/mooncake-store/src/master.cpp#L140-L161 、#L235-L238 、#L416 （旗標）
- [M3] https://github.com/kvcache-ai/Mooncake/blob/d750cddafa4b62387ca135949595008c51e05815/mooncake-store/include/replica.h#L104-L126 （ReplicateConfig：soft_pin_ttl_ms、with_hard_pin、group_ids）
- [M4] https://github.com/kvcache-ai/Mooncake/blob/d750cddafa4b62387ca135949595008c51e05815/mooncake-store/src/master_service.cpp#L11428-L11450 （候選：跳過 hard pin、依 deadline、soft pin 另列）
- [M5] https://github.com/kvcache-ai/Mooncake/blob/d750cddafa4b62387ca135949595008c51e05815/docs/source/design/store/mooncake-store.md#L352-L401 （Eviction／Lease／Soft Pin／Hard Pin）
- [M6] https://github.com/kvcache-ai/Mooncake/blob/d750cddafa4b62387ca135949595008c51e05815/mooncake-store/include/master_config.h#L190 （promotion_on_hit = false）
- [D1] https://github.com/ai-dynamo/dynamo/blob/f24c59b936ba23c933cc35d2f8c7f9ff71f54c3d/docs/fern/pages/reference/components/kvbm-configuration.mdx#L15-L51 、#L218-L219
- [D2] https://github.com/ai-dynamo/dynamo/blob/f24c59b936ba23c933cc35d2f8c7f9ff71f54c3d/lib/kvbm-logical/src/pools/inactive/backends/ （lru／multi_lru（TinyLFU）／lineage／fifo）
- [D3] https://github.com/ai-dynamo/dynamo/blob/f24c59b936ba23c933cc35d2f8c7f9ff71f54c3d/lib/bindings/kvbm/src/block_manager/vllm/connector/leader/slot.rs#L202-L206
- [D4] https://github.com/ai-dynamo/dynamo/blob/f24c59b936ba23c933cc35d2f8c7f9ff71f54c3d/lib/bindings/kvbm/python/kvbm/trtllm_integration/connector/kvbm_connector_leader.py#L137
- [D5] https://github.com/ai-dynamo/dynamo/blob/f24c59b936ba23c933cc35d2f8c7f9ff71f54c3d/docs/fern/pages/developer-guide/additional-resources/nvidia-request-extensions-nvext.md#L146-L240
- [D6] https://github.com/ai-dynamo/dynamo/blob/f24c59b936ba23c933cc35d2f8c7f9ff71f54c3d/docs/fern/pages/developer-guide/knowledge-base/modular-components/backends/sglang/agents-on-sglang.md#L50-L87 、#L131-L187
- [D7] https://github.com/ai-dynamo/dynamo/blob/f24c59b936ba23c933cc35d2f8c7f9ff71f54c3d/lib/llm/src/preprocessor/speculative_prefill.rs#L4-L9
- [D8] https://github.com/ai-dynamo/dynamo/blob/f24c59b936ba23c933cc35d2f8c7f9ff71f54c3d/docs/fern/pages/use-cases/agents/thunderagent-program-scheduler.md#L95-L105
- [D9] https://github.com/ai-dynamo/dynamo/blob/f24c59b936ba23c933cc35d2f8c7f9ff71f54c3d/docs/fern/pages/use-cases/agents/agent-hints.md#L50-L56
- [D10] https://github.com/ai-dynamo/dynamo/issues/11673#issuecomment-4996613234 （2026-07-16：「KVBM v1 is being sunset」）
- [D11] https://github.com/ai-dynamo/kvcr/blob/2a7ed995935f08b03150d250ac4bdd5a79cac395/docs/design_overview.md#L195 （no_evict）、#L260 （submit_hint 不主動預取）、#L306-L317 （Policy API、預設 LRU）
- [LD1] https://github.com/llm-d/llm-d/blob/39b14fa06ad48ff91853d902cabaa4c8b91b8db7/docs/architecture/kv-management/kv-offloader.md#L122-L134 、#L215
- [LD2] https://github.com/llm-d/llm-d/blob/39b14fa06ad48ff91853d902cabaa4c8b91b8db7/guides/tiered-prefix-cache/README.md
- [LD3] https://github.com/llm-d/llm-d/blob/39b14fa06ad48ff91853d902cabaa4c8b91b8db7/guides/turn-priority-fairness/README.md#L23-L35 、#L604-L635
- [LD4] https://github.com/llm-d/llm-d-kv-cache/blob/822e6dd9c459632f34f872b4a5094766fbce2663/kv_connectors/pvc_evictor/README.md#L88-L134
- [LD5] https://github.com/llm-d/llm-d/blob/39b14fa06ad48ff91853d902cabaa4c8b91b8db7/docs/architecture/kv-management/prefix-cache-aware-routing.md#L23-L30
- [A1] https://github.com/vllm-project/aibrix/blob/065e50f7a25a60bba251dfe97c75c8de5aab72e0/python/aibrix_kvcache/aibrix_kvcache/envs.py#L50-L105
- [A2] https://github.com/vllm-project/aibrix/blob/065e50f7a25a60bba251dfe97c75c8de5aab72e0/python/aibrix_kvcache/aibrix_kvcache/cache_manager.py#L824-L831
- [A3] https://github.com/vllm-project/aibrix/blob/065e50f7a25a60bba251dfe97c75c8de5aab72e0/python/aibrix_kvcache/aibrix_kvcache/l1/eviction_policy/fifo.py#L91-L94
- [A4] https://github.com/vllm-project/aibrix/blob/065e50f7a25a60bba251dfe97c75c8de5aab72e0/python/aibrix_kvcache/aibrix_kvcache/integration/vllm/kv_connector/aibrix_offloading_connector_type1.py#L1031-L1035
- [T1] https://github.com/NVIDIA/TensorRT-LLM/blob/e88d19ca3bda5bb761cf582e0379ed5397a7e60b/docs/source/features/kvcache.md#L17-L29 、#L248-L252
