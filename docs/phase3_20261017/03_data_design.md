# 第三階段的資料與載入設計（2026-10-17，第一版）

> **為什麼寫這份**：使用者 10/17：「資料集、loading 等等看怎麼設計，跟別人怎麼做，要合成也可以」。
> **現況**：X1–X9 各自寫了一份 trace 載入程式（`code/p3x/x*_trace.py`、`x2_load.py`、`x*_fi.py`…），同一份資料有好幾種讀法。這份規格把它們收斂成一個統一的載入層（`code/p3_data/`），之後的 P3-1～P3-6 都只讀這一層。
> **來源標記**同 repo 慣例。別人的做法出自已讀過的卡片（E05、E06、`RELATED_WORK_WEAKNESSES.md`、D4 §5、`research/LIT_AND_TRACES.md`），頁碼見各卡。

---

## 0. 一句話

**資料分三級，各管各的主張**：
- 真實 trace（有 session、真實間隔）才能撐「預測學得到」的主張；
- 半合成（真實的回來結構＋換算過的長度或併發）只報告；
- 全合成（已知答案的產生器）只用來測程式和機制。

三級共用一個統一格式，每個來源的載入函式裡都有「用 A 欄驗算 B 欄」的斷言（CLAUDE.md 規則 6）。

---

## 1. 別人怎麼做

| 論文 | 資料 | 時間戳 | 到達／回來怎麼產生 | 重播方式 | 教訓 |
|:--|:--|:--|:--|:--|:--|
| LPC（NeurIPS'25） | LMSys、ShareGPT、Chatbot-Arena（平均輸入 36–113 token） | 公開資料**沒有**逐輪時間戳 | 會話內間隔取指數分布（平均 λ_chat＝100 s）；會話開始時間錯開，同時活躍 ≤200 | **閉迴路**：下一輪＝上一輪回應實際完成時間＋取樣的間隔 | 原文說 Poisson 會讓請求早於上一輪回應完成，所以用閉迴路〔E06，原文 p.7〕 |
| SAECache（2605.18825） | ShareGPT、LMSys、Arena；4 個合成負載 | 用 Bailian（9,012 session）、CC-Bench 擬合輪間間隔 | log-normal 間隔（離線 MLE μ＝4.82、σ＝1.25） | — | 用真實 trace **擬合分布**，再用分布產生〔E06，原文 p.17〕 |
| LARU（2509.20979） | Aibrix-Synthetic（500 對話、平均 prompt 2,761 token）、內部 Online-QA | 原文沒說用不用原始時間戳 | 用併發數控制 | 〔未查證〕 | Qwen-Bailian 只出現在動機圖〔E06〕 |
| CachedAttention（ATC'24） | ShareGPT 9K session | **Poisson 模擬** | — | — | 時間戳不是真的〔RELATED_WORK_WEAKNESSES〕 |
| Strata（OSDI'26） | LooGLE、NarrativeQA、ReviewMT、ShareGPT | **全部 Poisson**（原文：「individual query timestamps are not available」） | — | — | 同上〔RELATED_WORK_WEAKNESSES〕 |
| Bidaw（FAST'26） | 自己的 trace | 真實 | — | — | **換成 Poisson 時間戳後效果消失** → 學習的主張不能用合成時間〔D4 §5，原文 p.12〕 |
| Mooncake（FAST'25） | conversation、toolagent | 真實（毫秒、解析度約 1 s） | trace 原樣 | **開迴路**（照時間戳送） | 沒有 session id〔D4 §4.1〕 |
| KVCache-in-the-wild（ATC'25） | Bailian 真實 trace | 真實 | trace 原樣 | — | 每條 2 小時、右設限明顯〔LIT §0〕 |
| Copilot 2026（2608.00101） | 一週、301,026 session | 真實 | trace 原樣 | — | **這週訓練、下週評估**〔LIT B.3〕 |
| CacheWise（2606.16824） | 真實 Claude Code trace（94 session） | 真實 | trace 原樣 | — | 論文說隨機 80% session 訓練，repo 的切分卻是 34／94〔LIT cards/P3_CacheWise〕 |
| 我們 9 月的 M5 | Mooncake＋自己合成的長文件 | 真實／合成 | — | — | **120 個獨立事件**撐不起學習的主張（B11）；CacheWise×16 併發是重排出來的（密度真、併發假） |
| 我們第一階段 | chat、doc（8 個 session） | seed 決定回來順序 | — | 一次一個請求 | 回來順序依構造就不可預測（審查指出） |

**整理出四條規則〔判讀〕**：
1. 「預測學得到」只能用真實間隔；Poisson 或擬合分布產生的時間，會把特徵和間隔的關聯洗掉（Bidaw）。
2. 重播用閉迴路：下一輪＝上一輪結束＋使用者思考時間。標籤用思考時間，不用「到達間隔」，因為到達間隔含服務時間，會隨硬體與策略改變（LPC 的理由；E06 的 LARU 卡也指出這點）。
3. 切分依時間（前段訓練、後段測試），且同一 session 不跨切分（Copilot 的做法）。
4. 報**獨立 session 數**，不報 block 數（B11）。

---

## 2. 三級資料

| 級別 | 是什麼 | 能撐什麼主張 | 來源 |
|:--|:--|:--|:--|
| **R：真實** | 有 session、真實間隔，原樣使用 | 全部，包括「預測學得到」 | FreeInference（12,002 session，長 context）、Bailian（有 parent_chat_id，2 小時）、WildChat-1M（有文字、時間戳、hashed_ip，短）、BurstGPT（若有 session 欄）、Mooncake（重建 session，**兩種規則都要報**）、CacheWise（94 session，只報告） |
| **H：半合成** | 真實的 session 結構與間隔，換掉一個量 | 只報告，要和 R 分開列 | (a) 長度換算：把 token 數換成某個模型的 KV 位元組，或把短 trace 的長度映射到 8K–32K；(b) 併發重排：把真實 session 重新錯開，提高同時活躍數（M5 的 CacheWise×16 做法）；(c) tokenizer 換算：FreeInference 是 o200k_base，換成 Llama／Qwen 的 token 數 |
| **S：全合成** | 參數式產生器，**答案已知** | 程式測試、機制、「預測要多準才有用」的曲線 | §6 的產生器 |

---

## 3. 統一格式（每個來源都轉成這張表）

一列＝一個 session 的一輪（turn）。存成 parquet：`/ssd8/hungwei/paper-hkv/datasets/p3_unified/<source>/<variant>/turns.parquet`，旁邊放 `manifest.json`。

| 欄位 | 型別 | 說明 |
|:--|:--|:--|
| `source`、`variant` | str | 例如 `freeinference`／`week01`、`mooncake`／`conv-rule-loose` |
| `session_id`、`user_id` | str | 沒有 user 就留空；重建的 session 加 `session_rule` |
| `turn_idx` | int | 從 0 開始 |
| `trigger` | str | `user`／`tool`／`unknown` |
| `t_start_ms`、`t_end_ms` | int64 | 請求開始、回應結束；沒有結束時間就留空，另記 `t_end_source`（`trace`／`estimated`） |
| `input_tokens`、`output_tokens`、`cached_tokens` | int | 來源自己的 tokenizer；`tokenizer` 欄寫是哪一個 |
| `new_user_tokens` | int | 這一輪新加的使用者輸入（打字量的上界）；算法寫在來源的 loader 裡 |
| `context_tokens` | int | 這一輪開始時的累積 context |
| `block_ids` | list[int] 或空 | 前綴 block（有的來源才有），加 `block_tokens` 欄（16、512…） |
| `tool_name`、`tool_category`、`tool_latency_ms` | str／int | agent trace 才有 |
| `req_type` | str | Bailian 的 type、WildChat 的 model 等 |
| `text_ref` | str | 文字在原始檔的位置（只有 WildChat 等有文字的來源）；**文字本身不進統一表** |
| `next_t_start_ms` | int64 | 同一 session 的下一輪開始；沒有就空 |
| `think_ms` | int64 | `next_t_start_ms − t_end_ms`（使用者思考時間；第三階段的主標籤） |
| `censored`、`obs_end_ms` | bool／int64 | 到觀察窗結束都沒有下一輪＝右設限；`obs_end_ms` 是這一列的觀察窗終點（trace 結尾或週邊界） |
| `split` | str | `train`／`valid`／`test`，依 §7 |

`manifest.json`：輸入檔的 sha256、loader 的 git hash、每條斷言的結果（通過率）、列數、session 數、user 數、設限比例、思考時間分位數。

---

## 4. 各來源怎麼讀、怎麼驗算

| 來源 | session 怎麼來 | 規則 6 的斷言（用 A 欄驗算 B 欄） | 思考時間 | 設限 |
|:--|:--|:--|:--|:--|
| FreeInference | `session_id`（官方重建，session 不跨週）；子 agent 巢狀 | `len(block_ids)==floor(Σrole_tokens/16)`（文獻 agent 第 1 週驗過 100%）；同 session 的 `step` 連續 | `start_ms`／`end_ms` 都有 | 週邊界 |
| Bailian | `chat_id`／`parent_chat_id` 串鏈 | 16-token 單位（環境 agent 驗過吻合率 1.0）；`turn` 和鏈的深度一致 | 只有時間戳的話要估回應結束（標 `estimated`） | 2 小時的 trace 結尾 |
| WildChat-1M | `conversation_hash`；`hashed_ip` 跨對話串同一人 | 時間戳在對話內單調；輪數＝訊息數÷2 | 時間戳是回應完成時間 → 思考時間＝下一個 user 訊息的時間（若有）− 上一個回應完成；欄位不夠就照實記 NOT_AVAILABLE | 資料集結尾 |
| BurstGPT | 若有 `Session ID`／`Elapsed time` 就用；沒有就**不當成 session 資料** | 時間戳單位用總長度驗算 | 用 `Elapsed time` 扣服務時間 | trace 結尾 |
| Mooncake | 前綴鏈重建，**至少兩種規則**（寬鬆／嚴格；文獻 agent 量到「會回來」20% vs 44%），各存一個 variant | `len(hash_ids)==ceil(input_length/512)`；前綴鏈一致（`m8_trace_oracle` 已有） | 只有到達時間 → 用輸出 token 數估服務時間（標 `estimated`，寫公式） | trace 結尾（58.9 分鐘） |
| CacheWise | repo 的 session 欄 | 依 README 的欄位交叉驗算 | 工具時長＋使用者停頓 | — |

- 所有斷言寫在 loader 函式裡，不在外部腳本；任何一條低於門檻（例如吻合率 <0.95）就丟錯，不靜默繼續（規則 6、規則 7）。
- 下載的資料是不受信任的：只讀不執行，讀它的 Python 用 `python -I`。

---

## 5. 怎麼重播（模擬器與 GPU harness 共用）

- **閉迴路（預設）**：第 k＋1 輪的到達＝第 k 輪在**模擬裡**的完成時間＋`think_ms`。這樣不同策略的服務時間差不會污染「使用者什麼時候回來」。
- **開迴路（敏感度）**：照 trace 的 `t_start_ms` 送。和閉迴路差很多的格子要標出來。
- **工具觸發的輪**（`trigger=tool`）：間隔是工具時長，不是人的思考時間，分開統計。降級、丟掉的決策主要看使用者觸發的那類。
- **併發**：預設照 trace 原本的重疊；提高併發是 H 級（半合成），只報告。

---

## 6. 全合成產生器（S 級，`code/p3_data/synth.py`）

用途：測程式（oracle 不能輸給任何策略、LRU 逐位元一致等）、測機制（Cake 會不會縮小預測的價值）、畫「預測要多準才有用」的曲線。**不能撐「學得到」的主張。**

| 參數 | 意思 | 預設（可改，寫進 manifest） |
|:--|:--|:--|
| `n_sessions`、`seed` | 規模與隨機種子 | 10,000、0 |
| 三群混合 | 工具型（秒級）、人（分鐘級）、不回來；各自的比例 | 依 FreeInference 第 1 週的分位數擬一個起點，**標成「擬合值」** |
| 每群的間隔分布 | log-normal（SAECache 的做法）或經驗分布重抽 | log-normal |
| `signal_strength` ρ ∈ [0, 1] | 一個可觀測特徵和真實間隔的相關程度：ρ＝0 時特徵沒用（只剩 LRU），ρ＝1 時特徵＝答案 | 掃 0、0.25、0.5、0.75、1 |
| context 成長 | 每輪加多少 token（可以從真實 trace 重抽） | 從 FreeInference 重抽 |
| 模型 | KV 每 token 的位元組（Llama 128 KiB、Qwen 56 KiB，BF16） | 兩個都產 |

- `signal_strength` 讓我們在**不訓練任何模型**的情況下，回答「特徵要多有用，預測才值得做」：直接把 oracle 的資訊加雜訊、或把 ρ 調低，看 headroom 剩多少（X1 的 Q1c）。
- 每次產生都存 manifest（參數、seed、程式 hash），同參數必須逐位元重現。

---

## 7. 切分與洩漏

| 來源 | 訓練／驗證／測試 | 理由 |
|:--|:--|:--|
| FreeInference | 依週：前 10 週訓練、2 週驗證、後 4 週測試；另做「依 user」的切分（某些 user 完全不在訓練集）當泛化測試 | Copilot 的「這週訓練、下週評估」；session 不跨週 |
| Bailian、Mooncake | 每條 trace 依時間前 60%／中 20%／後 20% | 只有 1–2 小時，週切不了 |
| WildChat | 依時間（資料集有時間戳）；同一 `hashed_ip` 的人另做依 user 切 | — |

- 同一 session 一定在同一個切分；跨切分邊界的 session 歸到它**開始**的那一段，邊界之後的輪當設限。
- 驗證集只用來選門檻與早停；測試集在 P3-4 之前不看（凍結時記雜湊）。

---

## 8. 實作（交給資料 agent，現在就可以做，不訓練任何模型）

- `code/p3_data/schema.py`（欄位、型別、manifest）、`code/p3_data/load_<source>.py`（每個來源一支）、`code/p3_data/synth.py`、`code/p3_data/build.py`（全部轉出統一表）、`code/p3_data/test_p3_data.py`（斷言與小樣本對拍）。
- 參考 X1–X9 已寫的 loader（`code/p3x/`），**但不改它們**；兩邊讀出的 session 數、間隔分位數要能對上，對不上就寫出差在哪。
- 輸出：`/ssd8/hungwei/paper-hkv/datasets/p3_unified/`；摘要表（每個來源一列：session 數、user 數、輪數、設限比例、思考時間分位數、斷言結果）寫在 `results/p3_data/summary.csv`（有 `run_id`、`ts`）。
- 文件：`docs/phase3_20261017/03b_data_report.md`（實作與數字）。
