# 第三階段資料管線：實作與數字（2026-10-17，資料管線 agent）

> 規格：[`03_data_design.md`](03_data_design.md)。這份只寫**實際做了什麼、實際量到什麼**；沒跑的寫 NOT_MEASURED。
> 每個數字附 run_id（`/ssd8/hungwei/paper-hkv/runs/<run_id>/`）。摘要表 `results/p3_data/summary.csv`（每列有 run_id、ts），
> 對拍表 `results/p3_data/crosscheck.csv`。沒有訓練任何模型、沒有調任何策略參數（CLAUDE.md §8）。
> 程式在 `code/p3_data/`，**沒有 commit**（照指示）；統一表在 `/ssd8/hungwei/paper-hkv/datasets/p3_unified/`（7.9 GB，不進 git）。

---

## 0. 一句話與狀態

7 個來源（FreeInference、Bailian、Mooncake、BurstGPT、WildChat-1M、CacheWise、全合成）全部轉成同一張表；
**hard 斷言全過**，只有 Bailian traceB 照預期不過（沒有任何 parent 連結 → 不當成 session 資料）。
和 X 系列 loader 對拍 114 項：102 項逐位相同，12 項不同，都已追到原因（§4；都是規則定義不同，不是讀錯）。
測試 48 項全過（run `20261017-221058-p3data-test`）。合成產生器同參數同 seed 逐位元重現（三次獨立執行的 parquet sha256 相同）。

### 0.1 最終 run（程式凍結後全部重跑一次；數字和第一次完全相同，第一次的 run 見 §6.3）

| 步驟 | run_id |
|:--|:--|
| 服務時間公式常數 | `20261017-220333-p3data-ficonst` |
| FreeInference week01–12／13／14–15／16 | `20261017-220329-p3data-fi2-w01`、`20261017-220330-p3data-fi2-w13`、`20261017-220331-p3data-fi2-w14`、`20261017-220332-p3data-fi2-w16` |
| Bailian（traceA、thinking、coder、traceB） | `20261017-220343-p3data-bailian`（exit 5＝traceB 的斷言不過，預期中） |
| Mooncake（6 個 variant） | `20261017-220402-p3data-mooncake` |
| BurstGPT v3 | `20261017-220414-p3data-burstgpt` |
| CacheWise | `20261017-220427-p3data-cachewise` |
| WildChat-1M | `20261017-220431-p3data-wildchat`（token 數用第一次 run `20261017-214522-p3data-wildchat` 切好的快取，鍵＝檔案 sha256＋tokenizer sha256） |
| 合成：擬合／產生 | `20261017-220343-p3data-synthfit`、`20261017-220349-p3data-synth` |
| summary.csv／crosscheck.csv／測試 | `20261017-221049-p3data-summary`、`20261017-221053-p3data-crosscheck`、`20261017-221058-p3data-test` |

---

## 1. 做了什麼

| 檔 | 內容 |
|:--|:--|
| `code/p3_data/schema.py` | 欄位與型別、`Checks`（規則 6 的斷言；hard 不過 → `DataAssertionError`）、`finalize()`（turn_idx、下一輪、think_ms、設限）、切分、寫檔與 manifest |
| `load_freeinference.py` | 一週一個 variant；block_ids 依 row group 串流寫入（不進 pandas；全部 8 GB） |
| `load_bailian.py`、`load_mooncake.py`、`load_burstgpt.py`、`load_wildchat.py`、`load_cachewise.py` | 各來源的 loader，斷言寫在 loader 裡，**loader 自己丟錯**（規格 §4） |
| `synth.py` | 全合成產生器（`fit` 由 FreeInference week01 擬合起點；`generate` 產生） |
| `build.py` | `build.py <source> [--variants …]`、`fi-constants`、`summarize`；斷言不過 → 不寫 turns.parquet，寫 `FAILED.json`（完整 checks），舊輸出改名 `.stale` |
| `crosscheck.py` | 和 X1–X9 的輸出 CSV 對拍（只讀，不改、不執行 `code/p3x/`） |
| `test_p3_data.py` | 單元、斷言真的會丟錯（造 16 vs 512 token 單位錯的假檔）、合成逐位元重現、已轉好的表（manifest 斷言、validate、FreeInference block 對齊、和文獻 agent 的小樣本對拍） |

輸出：`p3_unified/<source>/<variant>/{turns.parquet, manifest.json}`。manifest 有：輸入檔 sha256（和各資料目錄 README.txt／tables.json 比對，
全部相同）、程式的 git HEAD＋`code/p3_data/*.py` 各自的 sha256（還沒 commit，光 git hash 不夠）、每條斷言的值與門檻、統計、
公式（估的服務時間）、切分邊界、**test 列的雜湊**（§7「凍結時記雜湊」：session_id、turn_idx、t_start_ms 排序後的 sha256）。

用法：`schema.read_unified(source, variant, columns=[...])`；FreeInference 讀很多週時記得不要讀 `block_ids`（7.8 GB）。

---

## 2. 統一格式：規格的欄位全有，另外加的欄位與約定

規格 §3 的 28 個欄位全部照做。另外加了 16 個（schema.py 的 `EXTRA`），理由：

| 欄位 | 為什麼加 |
|:--|:--|
| `parent_turn_idx`、`n_children` | Bailian、Mooncake 是**樹**（父可以有多個子）；只靠 turn_idx 表達不了 |
| `next_trigger` | think_ms 是人的思考（下一輪 user）還是工具時長（下一輪 tool）——§5 要求分開統計，標籤要知道下一輪是哪種 |
| `gap_ms` | next_t_start − t_start（到達間隔，含服務時間）；X 系列多半用這個，對拍要用 |
| `t_start_source`、`think_source` | WildChat 缺的是**開始**時間（不是結束）；think_ms 是實測、估的還是合成的要能分 |
| `service_est_ms` | 估的服務時間（estimated 的來源才有），讓下游可以換公式重算 |
| `split_user` | §7 的「依 user 切」（FreeInference、WildChat） |
| `split_drop`、`split_censored`、`split_obs_end_ms` | §7「邊界之後的輪當設限」：主欄位保留全 trace 的真實標籤（模擬器重播要用），切分視窗內的標籤另存 |
| `harness`、`is_subagent`、`parent_session_id`、`next_waited_for`、`provider_prompt_tokens` | FreeInference 專有，下游會用到（子 agent、ask_user_tool） |
| `x_*` | 來源自己的額外欄位（只有 synth 用：真實答案、訊號特徵、KV 位元組） |

約定（全部寫在 schema.py）：
- **下一輪**＝同一 session 裡以這一列為父的請求中**最早**的一個（鏈狀來源＝下一列）。這樣 think_ms 衡量的是「這一列的 context 何時被延伸」。
- **設限**：觀察窗內沒有下一輪＝`censored=True`（spec 原文）。所以 WildChat／BurstGPT 對話的最後一輪也是設限（觀察窗＝到資料集結尾，所以它們實際上幾乎等於「不回來」）；
  summary.csv 另外給 `censored_short_frac`＝設限且剩下的觀察窗 < think p90 的比例（真正「看不清楚」的設限）。
- **think_ms 分位數**（summary.csv）：只算 next_trigger ∈ {user, unknown}；工具觸發的另列 `tool_gap_ms_p50/p90`。
- **實測的 think_ms 不改**（可能 <0，例如上一個回應完成前就送下一個；計數寫在 manifest）；**估的** t_end 若超過下一輪開始或觀察窗終點才截斷（計數）。
- **估的服務時間**（Bailian、Mooncake、WildChat）：`service_est_ms = max(0, A + B·output_tokens + C·input_tokens)`，
  A＝1509.1 ms、B＝14.213 ms/token、C＝0.01484 ms/token。由 FreeInference **訓練週**（week01–10，399,463 個請求）的 (end_ms − start_ms)
  做中位數迴歸（statsmodels QuantReg q=0.5，seed 0 抽 200,000 列）；全部列上預測 ≤ 實際的比例 0.4998，|殘差| p50 2,542 ms、p90 9,386 ms
  （run `20261017-220333-p3data-ficonst`，`results/p3_data/service_constants.json`）。**跨來源的粗估**（模型、硬體、批次、prefix cache 都不同），
  只用來把服務時間從間隔裡扣掉；think_ms 的低分位數對這個公式敏感。
- 切分邊界（Bailian、Mooncake、BurstGPT、WildChat、synth）：觀察窗跨度的 60%／80% 處，依 session **開始**時刻分。

---

## 3. 各來源的數字

單位：秒（除非另註）。「think」＝使用者思考時間分位數 p10／50／90／99。完整斷言在各 manifest.json。

### 3.1 FreeInference（R；runs 見 §0.1；`harvardMadsys/freeinference_agentic_trace @ 16d3c04d`）

16 週合計（`build.py summarize`，run `20261017-221049-p3data-summary`）：1,222,199 列、14,036 session（子 agent 2,034）、267 user、
使用者觸發 87,508 輪、工具觸發 1,134,691 輪、設限 0.0115；**think（next_trigger＝user，73,472 個）＝3.1／96.2／1,040.5／29,404.3**；
工具間隔 p50／p90＝1.69／16.94。
- 和 README 對得上：非子 agent 的列＝1,186,582（README「1,186,582 LLM requests」）、根 session＝12,002（README「12,002 agent sessions」）、
  user＝267（README「267 accounts」）。多的 35,617 列是巢狀子 agent 的請求。
- 每週 18 條 hard 斷言全過（16 週 288/288）：sha256 與列數＝tables.json；requests 依 (session_id, step) 排序（串流寫 block_ids 的前提）；
  **len(block_ids)＝floor(Σrole_tokens/16)：每週 1.0**（ceil 只有 ~6%，和 README「最後不滿一塊丟掉」一致）；step 連續；session 集合＝sessions.parquet；
  start≤end；ttft≤end−start；**工具 latency_ms＝下一個 start − 這個 end（±1 s）**（README 的定義，A 欄驗算 B 欄）；
  start_ms 落在目錄名的日期內（毫秒 epoch）；(end−start)/completion 的中位數 ∈ 0.5..500 ms/token；turn_idx＝step。
- 和文獻 agent 的獨立計算逐位一致（test_p3_data.py 有測）：第 1 週 125 session、使用者觸發 785、工具觸發 9,953、think＝25.0／175.0／3,586.5／34,686.8
  （文獻 agent：25／175／3,587／34,687）。
- 工具觸發、但其實在等人或等子 agent：`next_waited_for`＝ask_user_tool 4,417 列、subagent 3,594 列（其餘 tool 1,126,680）。
  think 分位數不含它們；要把 ask_user_tool 當成人的思考，用 `next_waited_for`。
- 切分（§7，依週）：train（week01–10）399,463 列、valid（11–12）164,070、**test（13–16）658,666＝54%**（後面幾週大很多）。
  test 週的 198 個 user 中 **133 個從沒出現在 train 週** → 依週切同時是很強的 user 漂移。
  split_user（sha256 雜湊 80/10/10）：user 217／34／16，但列數 950,927／227,609／43,663（test 只有 3.6%：少數 user 占大部分請求）。
- 每週的數字：

| 週 | session | user | 列 | 使用者觸發 | 設限 | think（s） | 工具 p50／p90 |
|:--|--:|--:|--:|--:|--:|:--|:--|
| 01 | 125 | 8 | 10,738 | 785 | 0.0116 | 25.0／175.0／3,586.5／34,686.8 | 3.41／30.48 |
| 02 | 170 | 9 | 12,827 | 968 | 0.0133 | 31.3／150.5／1,013.2／32,102.7 | 1.20／9.95 |
| 03 | 76 | 10 | 7,324 | 496 | 0.0104 | 20.7／173.0／1,443.9／20,866.1 | 1.64／11.65 |
| 04 | 80 | 7 | 6,859 | 437 | 0.0117 | 49.9／218.8／1,626.6／17,786.8 | 1.06／14.89 |
| 05 | 479 | 23 | 35,659 | 1,485 | 0.0134 | 15.8／123.8／1,125.0／14,848.9 | 1.16／9.31 |
| 06 | 715 | 50 | 56,310 | 2,716 | 0.0127 | 15.4／115.5／1,415.2／41,050.2 | 1.00／10.06 |
| 07 | 628 | 49 | 50,615 | 2,838 | 0.0124 | 14.9／114.0／1,152.6／46,946.8 | 1.40／17.04 |
| 08 | 978 | 49 | 68,302 | 4,153 | 0.0143 | 11.6／103.2／745.4／27,316.3 | 1.42／14.16 |
| 09 | 969 | 54 | 72,025 | 5,900 | 0.0135 | 6.8／88.0／699.1／23,885.4 | 1.56／15.51 |
| 10 | 927 | 66 | 78,804 | 6,121 | 0.0118 | 5.6／98.5／1,125.2／28,871.6 | 1.67／15.71 |
| 11 | 1,198 | 71 | 84,524 | 6,550 | 0.0142 | 4.7／107.9／1,239.3／30,540.7 | 1.33／17.45 |
| 12 | 973 | 74 | 79,546 | 6,130 | 0.0122 | 4.7／112.8／1,072.7／22,365.2 | 1.74／21.20 |
| 13 | 1,558 | 104 | 163,970 | 11,341 | 0.0095 | 2.4／105.7／1,164.5／32,700.3 | 1.90／17.60 |
| 14 | 1,505 | 106 | 144,227 | 10,600 | 0.0104 | 2.4／91.9／1,257.2／30,768.0 | 1.97／17.97 |
| 15 | 1,608 | 113 | 159,925 | 13,134 | 0.0101 | 2.6／65.0／818.7／25,373.3 | 1.77／19.20 |
| 16 | 2,047 | 141 | 190,544 | 13,854 | 0.0107 | 5.9／92.4／944.6／29,152.1 | 2.02／18.79 |

〔判讀〕設限只有 1.1%：FreeInference 的 session 是 agent 的長 session，「最後一輪」很少；這裡的設限只表示「這個 session 裡沒有下一個請求」，
session 結束後同一個 user 會不會開新 session 要用 user_id 另外算（沒做）。

### 3.2 Bailian（R；run `20261017-220343-p3data-bailian`；`alibaba-edu/qwen-bailian-usagetraces-anon @ 5f7439c5`）

| variant | 列 | session | ≥2 輪 | 設限 | think（estimated） | 斷言 | 估的 t_end 截到子請求的列 |
|:--|--:|--:|--:|--:|:--|:--|--:|
| traceA | 43,058 | 23,101 | 9,012 | 0.5365 | 21.6／102.0／712.1／2,196.1 | 13/13 | 7 |
| thinking | 10,812 | 9,612 | 509 | 0.8890 | 1.3／13.6／217.8／1,521.5 | 13/13 | 55 |
| coder | 43,011 | 26,406 | 6,693 | 0.6139 | 2.7／102.1／928.7／3,475.4 | 13/13 | 828 |
| traceB | — | — | — | — | — | **6/7**：`n_child_requests (has session structure)`＝0 | — |

- 斷言：sha256＝README.txt、len(hash_ids)＝ceil(input_length/16)（四份 1.0）、timestamp 非遞減、ts_max ∈ [7100, 7200] s（README 2 小時）、
  chat_id 唯一、父都找得到、子 turn＝父 turn＋1（1.0）、子 hash 前綴＝父 hash[:-1]（traceA 1.0、thinking 0.963、coder 0.996）、子 ts ≥ 父 ts、
  median(間隔／父 output) ∈ 0.01..10 s/token、turn − 根 turn＝鏈深度。
- user_id 全是 null（README：不保留跨 session 的使用者關聯），trigger 全是 unknown。thinking 的 session 只有 509 個 ≥2 輪（低於 1,000 門檻）。

### 3.3 Mooncake（R，但 session 是重建的；run `20261017-220402-p3data-mooncake`；`kvcache-ai/Mooncake @ d750cdda`）

三種重建規則（規則全文在 load_mooncake.py 與 manifest 的 `rule_doc`）：
**loose**＝照抄文獻 agent 的 `research/calc/mooncake_chain.py`；**strict**＝照抄 `mooncake_chain2.py`（前綴 ≤3 個請求共用、候選父唯一、時間嚴格較早）；
**d4**＝X2／X3 用的規則（為了對拍才存）。

| variant | session | ≥2 輪 | 有子請求的比例 | 最大 session（請求數） | 分叉父的比例 | 子 input ≥ 父 input＋0.9 父 output | 設限 | think（estimated） |
|:--|--:|--:|--:|--:|--:|--:|--:|:--|
| conv-rule-loose | 17 | 7 | 0.4394 | 12,007 | 0.218 | 0.849 | 0.5606 | 0.0／72.1／404.4／1,233.3 |
| conv-rule-strict | 9,595 | 1,959 | 0.2020 | 3 | 0.0025 | 0.940 | 0.7980 | 40.4／123.6／519.1／1,546.2 |
| conv-rule-d4 | 8,056 | 2,012 | 0.3268 | 43 | 0.0071 | 0.939 | 0.6732 | 35.2／115.5／466.2／1,454.4 |
| tool-rule-loose | 27 | 11 | 0.4408 | 10,917 | 0.344 | 0.794 | 0.5592 | 0.0／0.0／227.4／607.1 |
| tool-rule-strict | 21,093 | 2,061 | 0.1062 | 3 | 0.0028 | 0.927 | 0.8938 | 39.6／114.5／452.1／919.9 |
| tool-rule-d4 | 7,208 | 1,854 | 0.3883 | 9,202 | 0.280 | 0.795 | 0.6117 | 0.0／1.0／249.8／617.4 |

- 「有子請求的比例」和文獻 agent 逐位相同（loose 0.4394／0.4408、strict 0.2020／0.1062；test_p3_data.py 有測到 1e−12）。
- **loose 規則把幾乎所有請求併成少數幾棵樹**（conversation 17 個 session，最大 12,007 個請求）：它容許長度 1 的前綴（共用系統提示）當父子依據。
  所以 loose 的「session」沒有意義，只有每一列的「下一輪」有意義；整棵樹都從 train 段開始，valid／test 是空的。
- 同秒分叉：loose 的連結有 26.5%（conv）／42.5%（tool）間隔＝0；tool-rule-d4 的 think 中位數 1.0 s、loose 0.0 s ——toolagent 用寬鬆規則接起來的大多是同一秒的分叉，不是下一輪。
- hard 斷言 7/7（每個 variant）：sha256、512-token 單位（1.0）、timestamp 非遞減、跨度 0.98 小時（∈[0.5, 2]）、前綴鏈（同 hash 同位置、同前一個 hash：0 例外）、
  連結的 median(間隔／父 output) ∈ 0.01..10 s/token。
- ⚠ X2 把「子 input ≥ 父 input＋0.9 父 output ≥ 0.90」當 hard，toolagent 的 d4 因 0.7949 被 BLOCKED。這裡把它改成**只報告**（理由見 §7.4）；
  照 X2 的標準，conv-rule-loose（0.849）、tool-rule-loose（0.794）、tool-rule-d4（0.795）都會被擋，只有 strict 兩個與 conv-rule-d4 過。

### 3.4 BurstGPT（R；run `20261017-220414-p3data-burstgpt`；release v2.0）

- **只有 BurstGPT_3.csv（與 _without_fails_3）有 `Session ID`、`Elapsed time`**：header 讀過，BurstGPT_3＝`Timestamp,Session ID,Elapsed time,Model,Request tokens,Response tokens,Total tokens,Log Type`，
  BurstGPT_1／_2＝`Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type` → 照規格不轉。
- v3：233,617 列（Conversation log；API log 5,110,404 列沒有 session，不進表）、55,920 session（≥2 輪 36,086；和 E10 的 55,920 相同）、
  設限 0.2394；**think（trace：下一個 Timestamp −（Timestamp＋Elapsed））＝17.0／124.0／2,331.0／78,786.6**；think<0 的 95 列保留原值；
  Response tokens＝0 的失敗請求 1,856 列保留。
- hard 8/8：Total＝Request＋Response（1.0）、Session ID 非空 ⇔ Conversation log（1.0）、時間跨度 109.9985 天（README 110 天；Timestamp 是秒）、
  Timestamp 非遞減、Elapsed ≥ 0、median(Elapsed／Response) 0.0167 s/token。解析度 1 s（Timestamp、Elapsed 都是整數秒）。
- new_user_tokens 是有雜訊的上界：只有 58.7% 的下一輪 Request ≥ 上一輪 Request＋Response（請求不一定帶完整歷史）。

### 3.5 WildChat-1M（R；run `20261017-220431-p3data-wildchat`；`allenai/WildChat-1M @ 7d6490e4`）

- 1,960,074 輪、837,989 session（≥2 輪 331,483）、178,724 個 hashed_ip（user）、設限 0.4275；
  **think（estimated：t_start 是估的）＝24.3／121.3／1,246.0／27,506.7**（1,122,085 個）。
- README：assistant 的 timestamp＝「backend 收到 ChatGPT 完整回應的時刻」→ t_end（trace）；user 訊息**沒有** timestamp（斷言：1.0 是 null）
  → t_start＝t_end − service_est（不早於上一輪 t_end；被截的 213 輪）。
- token：WildChat 沒有 token 數 → 用 Llama-3.1-8B-Instruct 的 tokenizer.json 逐訊息切（不加 chat template）；全部訊息 3.857 字元／token。
- session_id＝conversation_hash＋第一則的 turn_identifier（README：conversation_hash「不是唯一鍵」；第 0 個檔 59,857 個對話只有 59,363 個不同的 hash）。
- hard 24/24：14 個檔的 sha256、turn＝user 訊息數＝訊息數÷2（1.0）、對話層 timestamp＝最後一則 assistant（1.0）、assistant timestamp 非空、user timestamp 全 null、
  對話內 assistant 時間非遞減（1.0）、user／assistant 交替（1.0）、字元／token ∈ [1, 8]。沒有丟掉任何對話。
- 切分：train／valid／test 1,324,417／272,399／363,258 列；split_user 依 hashed_ip。

### 3.6 CacheWise（只報告；run `20261017-220427-p3data-cachewise`；`cachewise-coding-traces @ 181c435a`）

- 11,135 個主線請求、81 個 session（94 個檔中 13 個沒有主線請求）、trigger user 1,535／tool 9,553／unknown 47、設限 0.0073；
  think（下一輪是文字 user 觸發）＝−0.001／28.5／156.4／6,749.0（1,501 個；think<0 的 155 列保留）；工具間隔 p50／p90＝0.08／10.16。
- t_start＝觸發這個請求的 user／tool_result 紀錄的時間；t_end＝這個請求最後一則 assistant 紀錄的時間（兩個都是 trace）。
  sidechain（子 agent）1,013 則紀錄沒轉；split 照 repo 的 `train_sessions.json`（34 個）→ train，其餘 test。
- hard 4/4：parsed events 的 request_id 集合＝raw 的 requestId 集合（94/94）、execution_time_ms＝result_ts − ts（±1 ms，11,614 個工具 1.0）、
  每個請求都有 usage、manifest session 數＝94。
- 各 session 的時間戳被各自平移過（session_0001 在 2020-07，模型是 2025-08 版）〔判讀，文獻 agent〕→ 沒有共同的觀察窗，每個 session 的觀察窗＝它最後一則紀錄。

### 3.7 全合成（S；run `20261017-220349-p3data-synth`）——見 §5

---

## 4. 和 X1–X9 的 loader 對拍（run `20261017-221053-p3data-crosscheck`，`results/p3_data/crosscheck.csv`）

114 項：**102 項相同**（Bailian 27/27、WildChat 8/8、FreeInference 17/18、Mooncake 48/54、CacheWise 2/7）。X 的值一律從它們的 CSV 讀。

**相同的（摘要）**
- Bailian traceA／thinking／coder vs X2、X3：請求數、session 數、≥2 輪 session 數、設限比例、間隔 p10／50／90／99（到小數第 4 位）、連結數，全部相同。
- Mooncake conv-rule-d4、tool-rule-d4 vs X2／X3／X4：session 數（8,056、7,208）、連結數（3,975、16,400）、≥2 輪的 session（2,012、1,854）、
  連結間隔分位數（X3）、「依時間的下一個請求」的比例（X4 0.330396）。strict vs X2「lit」、X3 strict、X4：session 9,595／21,093、連結 2,436／2,515、
  ≥2 輪 1,959／2,061、有子請求的比例 0.20198／0.10624。
- FreeInference vs X9、X3：請求 1,222,199、session 14,036、使用者回來 73,472（含負值）／73,458（X3 去掉負值）、工具回來 1,134,688、
  負的間隔 17、使用者 think 與工具間隔的 p10／50／90／99（相對誤差 < 5e−4）、有工具回來的 session 13,953。
- WildChat vs X2、X9：輪數 1,960,074、session 837,989、設限 0.427529、assistant→assistant 間隔 p10／50／90／99＝28／128／1,253／27,511.3、回來數 1,122,085。

**不同的 12 項，原因**
1. Mooncake conv-rule-d4、conv-rule-strict 的設限比例（統一表 0.67318／0.79802 vs X2 0.669604／0.797523）：**定義不同**。X2 的「下一輪」＝同 session 依時間的下一個請求；
   統一表＝最早的子請求（分叉時兩者不同）。用 X2 的定義在統一表上重算＝0.669604／0.797523，逐位相同（同一個 csv 的另一列，MATCH）。
2. Mooncake d4 的連結數與 session 數 vs X9 census（3,975 vs 3,974、8,056 vs 8,057；toolagent 16,400 vs 16,399、7,208 vs 7,209）：
   **X9 的父規則不是 D4**——X9 用「父的 hash_ids[:-1]」當鍵（`code/p3x/x9_census.py` 第 339–350 行），D4 用「父的完整 block 數（input_length//512）」。
   最後一塊剛好滿的請求兩者不同，差 1 條連結。
3. FreeInference「有使用者回來的 session」11,117 vs X3 11,116：X3 丟掉負的間隔；有 1 個 session 唯一的使用者回來是負的（只算 think ≥ 0 時＝11,116）。
4. CacheWise 請求數 11,135 vs X2 11,498：統一表只取主線；sidechain 有 363 個請求，11,135＋363＝11,498（逐位相同）。
5. CacheWise 使用者回來 1,464 vs X3 2,587，以及 think 分位數（28.5 vs 17.1 等）：**事件定義不同**。X3 數 raw 裡每一則文字 user 紀錄；
   這 2,587 則中只有 1,467 則直接觸發主線請求，其餘 1,120 則沒有觸發任何請求（內容已遮蔽，看不出是什麼）。統一表的「回來」是「請求」，所以只有前者；
   X3 的思考時間減的是「前一則 assistant 紀錄（含 sidechain）」，統一表減的是上一個主線請求的最後一則。CacheWise 只報告，兩種都不是錯的，但不能混用。

---

## 5. 全合成產生器（`code/p3_data/synth.py`）

- 模型：session 開始 ~ Uniform(0, horizon)；每一輪之後三選一（tool／human／end，每輪獨立）；間隔 ~ LogNormal；context 從 FreeInference week01 重抽（第 0 輪＋每輪增量，依 trigger）；
  輸出 token 從 week01 重抽；服務時間用 §2 的公式；觀察窗 [0, horizon]；訊號特徵 `x_feat_rho_<ρ>`＝ρ·s＋√(1−ρ²)·ε（s＝到下一輪真實時間的常態分數，end 排最後），
  ρ＝0、0.25、0.5、0.75、1 共用同一個 ε；KV 位元組 `x_kv_bytes_llama`（×131,072）、`x_kv_bytes_qwen`（×57,344）由架構算出。真實答案在 `x_true_next_class`、`x_true_think_ms`。
- **起點參數是擬合值**（run `20261017-220343-p3data-synthfit`，`results/p3_data/synth_fit_fi_week01.json`）：**只用 FreeInference week01（2026-05-17_2026-05-23）**
  （照規格 §6；它在 train 段）。p_tool＝0.9269、p_human＝0.0615、p_end＝0.0116；
  tool：μ＝8.1342、σ＝1.5684（中位 3.41 s）；human：μ＝12.0724、σ＝1.9383（中位 175.0 s）。log-normal 用分位數擬合：μ＝ln p50、σ＝(ln p90 − ln p10)/(2×1.2816)。
  week01 只有 125 個 session、8 個 user——起點很窄。
- 產出：`default-seed0`（10,000 session、7 天）834,991 列、設限 0.0120（其中 123 列是被觀察窗截掉、真的會回來）、human think 中位 172.5 s、工具間隔 p50／p90＝3.41／25.40 s；
  `small-seed1`（500 session、1 天）38,687 列。
- 一致性斷言 4/4：觀察到的 think＝真實 think、觀察到的 next_trigger＝真實類別、沒設限 ⇒ 不是 end、設限且會回來 ⇒ 真實下一輪 ≥ horizon。
- 訊號：Spearman(特徵, 真實到下一輪的時間)＝−0.0006／0.239／0.482／0.734／0.99999917（ρ＝0／0.25／0.5／0.75／1）。
- **逐位元重現**：default-seed0 的 parquet sha256＝`1bba37a0…8959`、small-seed1＝`79b42d97…99a2`，在 scratch 試跑、第一次 run `20261017-215100-p3data-synth`、
  最終 run `20261017-220349-p3data-synth` 三次相同；`test_synth_bitwise_reproducible` 測同 seed 位元組相同、不同 seed 不同。

---

## 6. 失敗、偏離與異常（完整錯誤）

### 6.1 FreeInference 的解碼速度斷言公式錯了
scratch 試跑（2026-10-17T13:38Z，week01；這次沒有走 run3090.sh，沒有 run_id，FAILED.json 在 scratchpad）：

```
build freeinference/week01 ASSERTION FAILED: freeinference/week01: 1 條斷言不過：[{'check': 'median((end-start-ttft)/completion_tokens) ms/token', 'value': 0.0547945205479452, 'threshold': '0.5..500', 'passed': False, 'hard': True, 'note': '解碼速度：時間是毫秒（秒會是 ~0.02、微秒 ~2e4）'}]
```

查原因（同一份資料）：provider B 的 `ttft_ms` 幾乎等於整段時長——completion ≥ 50 的請求中 20.6% 的 end−start−ttft ≤ 1 ms；
provider 別的中位數 (end−start−ttft)/completion：B 0.020、C 5.34、G 6.58、H 34.7 ms/token（week01），所以 (end−start−ttft) 不是解碼時間。
時間單位本身沒問題：同一週 start_ms 全部落在目錄名的日期內，(end−start)/completion 中位數 44.9 ms/token。
**處理（偏離，請主 session 確認）**：這條改成只報告（值仍寫在每週 manifest），另加 hard 斷言 `median((end-start)/completion_tokens) ∈ 0.5..500 ms/token`。
同一原因讓原本打算的「TTFT 中位數＋解碼 ms/token 中位數」服務時間公式不能用 → 改成整段時長的中位數迴歸（§2）。
（X2 的 `load_freeinference` 有同一條斷言，門檻也是 0.5..500；X2 的 FreeInference 沒有輸出列，可能沒跑到。）

### 6.2 Bailian traceB（預期中）

```
bailian/traceB: 1 條斷言不過：[{'check': 'n_child_requests (has session structure)', 'value': 0, 'threshold': '>=1', 'passed': False, 'hard': True, 'note': 'traceB 全部 parent_chat_id=-1 → 不當成 session 資料'}]
```
`p3_unified/bailian/traceB/FAILED.json` 有完整 checks；summary.csv 的 status＝ASSERT_FAIL。

### 6.3 程式改過之後全部重跑
第一次的 build（runs `20261017-213931-p3data-ficonst`、`…-213951/213952/213953/213954-p3data-fi-w*`、`…-214254-p3data-bailian`、`…-214255-p3data-mooncake`、
`…-214256-p3data-burstgpt`、`…-214522-p3data-wildchat`、`…-214718-p3data-cachewise`、`…-214943-p3data-synthfit`、`…-215100-p3data-synth`）之後，
我改了程式（每個 loader 自己 enforce、text_ref 指回原檔、schema 加 x_ 欄、synth 一致性斷言修正），所以在凍結後全部重跑（§0.1）。
兩次的統計、session 數、分位數完全相同；synth 的 parquet 位元組相同。現在的 manifest 記的是最終程式的 sha256。

### 6.4 其他異常（計數，不是錯）
- 估的服務時間比觀察到的間隔還長（截斷）：Bailian traceA 7、thinking 55、coder 828；Mooncake conv-loose 1,304、tool-loose 5,620、tool-d4 4,420、conv-d4 5；WildChat 213。
- 實測 think_ms < 0：FreeInference 17（下一個請求的開始早於這個的結束）、BurstGPT 95、CacheWise 155。

---

## 7. 和規格不同的地方與理由

1. **Bailian traceB、BurstGPT_1／_2 沒有轉**：沒有 session 結構（規格 §4 的 BurstGPT 規則；traceB 同理）。
2. **Mooncake 存了三種規則**（規格要求至少兩種）：多存 d4 是為了和 X2／X3／X4 對拍；session 數、連結數逐位相同（§4）。
3. **FreeInference 的 obs_end_ms＝該週最大的 end_ms**，不是名目的週邊界：README 沒寫週邊界的時區。設限比例只有 1.1%，影響很小。
4. **Mooncake 的「子 input ≥ 父 input＋0.9×父 output」改成只報告**（X2 是 hard）：規格 §4 列的 Mooncake 斷言是單位與前綴鏈（hard 都過）；
   這一條量的是「重建規則」把不同 session 接在一起的程度，不是資料單位。照它擋掉的話，toolagent 只剩 strict 能用。數字照列（§3.3），**要不要擋請主 session 決定**。
5. **think_ms 分位數只算 next_trigger ∈ {user, unknown}**：§5「工具觸發的分開統計」；工具的另列。
6. **WildChat 估的是 t_start（不是 t_end）**：資料有回應完成時間、沒有使用者送出時間；所以加 `t_start_source`。
7. **CacheWise 的切分照 repo 的 train_sessions.json**（34/94）：規格 §7 沒列 CacheWise；它只報告。BurstGPT 規格也沒列，比照 Bailian 時間 60/20/20。
8. **split_user 用 sha256 雜湊 80/10/10**：規格沒定比例；我選的，寫在 schema.user_split。
9. **主欄位保留全 trace 的標籤**，§7 的「邊界之後當設限」放在 `split_drop／split_censored／split_obs_end_ms`：模擬器重播要完整的 trace。
   用切分訓練／評估時：丟掉 `split_drop`，標籤用 `split_censored`、`split_obs_end_ms`。
10. **H 級（半合成）沒有做**：規格 §2 有，§8 的實作清單沒有；長度換算、併發重排、tokenizer 換算都還沒寫（§9）。
11. summary.csv 的統計**包含 test 週**（FreeInference week13–16）：那是管線檢查（斷言、session 數），沒有拿來調任何東西；
    test 的雜湊記在各 manifest（例如 week13：`63cb4045…d900`，163,970 列）。

---

## 8. 使用者要注意的事

1. **CacheWise 沒有 LICENSE**（GitHub API license＝null）→ 預設著作權保留。用在論文前要向作者確認（NEEDS_USER）。這裡只讀 JSON，沒執行 repo 裡的任何 .py。
2. **WildChat-1M 是 ODC-BY，內容是真實使用者對話**（可能有不當內容）。統一表**不含文字**，只有 token 數與 `text_ref`（指回原 parquet 的列與訊息）；
   但 `user_id` 欄存的是資料集本身的 `hashed_ip`（已雜湊）——統一表在 /ssd8，不進 git。引用時照 LICENSE.md 標出處。
3. BurstGPT CC-BY-4.0、Mooncake／Bailian Apache-2.0、FreeInference CC-BY-4.0：引用時要標出處。
4. **Bailian／Mooncake／WildChat 的 think_ms 是估的**（跨來源的服務時間公式，§2）；p10 一類的低分位數對公式很敏感，不要拿來下精確結論。
5. **Mooncake 的 session 是重建的**，「會不會回來」由規則決定（loose 0.44 vs strict 0.20／0.11）；loose 的 session 根本不是 session。只適合當標籤雜訊的敏感度。
6. FreeInference 的 test 週占 54% 的列，且 2/3 的 test 週 user 沒出現在 train 週。
7. 只有 FreeInference、Bailian traceA／coder、BurstGPT、WildChat、Mooncake（d4／strict）有 ≥1,000 個 ≥2 輪的 session；
   Bailian thinking（509）、CacheWise（80）低於門檻，只能報告。

---

## 9. 沒做的、需要決定的

- 沒做：H 級半合成產生器；§5 的閉迴路重播（那是模擬器的事，這裡只提供 think_ms 與 t_end）；session 之後同一 user 的「再回來」（FreeInference 有 user_id，可以算）；
  Copilot 2026 trace（沒下載）；LMSYS（gated，NEEDS_USER）。
- 要決定：(a) §6.1 的斷言替換是否接受；(b) Mooncake 的一致性檢查要不要當 hard（§7.4）；(c) FreeInference 的 ask_user_tool（4,417）要不要算成使用者的思考；
  (d) 服務時間公式要不要換（例如依來源各自的模型）——換了要重跑 Bailian／Mooncake／WildChat（WildChat 有 token 快取，幾分鐘）；
  (e) p3_data 的 commit（我沒 commit）。
- 重跑：`TIARA_NO_GPU=1 bash code/run3090.sh <短名> python -I code/p3_data/build.py <source>`；測試：`python -I -m pytest -q code/p3_data/test_p3_data.py`。

---

## 主 session 對待決事項的暫定處理（2026-10-17 22:2x，+08:00；使用者可以推翻）

| # | 事項 | 暫定 | 理由 |
|:--|:--|:--|:--|
| (a) | FreeInference 的解碼速度斷言在第 1 週失敗後，改成「(end−start)/completion」的硬檢查 | **接受，但標成「失敗後才改的斷言」** | 原本那條的目的是驗證時間單位是毫秒；它失敗的原因是 provider B 的 `ttft_ms` 幾乎等於整個請求時長，不是單位錯。替代的檢查量到 44.9 ms／token，時間戳也落在該週日期內，兩條獨立證據都支持毫秒。CLAUDE.md 規則 2 要求的「記錄、回報」已經做了；這裡記下是誰、何時接受的 |
| (b) | Mooncake 的一致性檢查（子的輸入 ≥ 父的輸入＋0.9×父的輸出）要不要當硬檢查 | **維持只報告**，但 conv-loose、tool-loose、tool-d4 三個 variant 在 manifest 標 `low_confidence_reconstruction` | 照 X2 的標準它們會被擋下；loose 規則本來就把整份 trace 併成 17／27 棵大樹，沒有意義。下游分析一律以 strict 為主、d4 為敏感度，loose 不用 |
| (c) | FreeInference 裡等 `ask_user_tool` 的工具間隔，算不算使用者思考時間 | **主分析算「使用者」**，另報不算的版本 | 這段時間由人決定，對「人什麼時候回來」的預測來說是同一類事件；欄位 `next_waited_for` 保留區別 |
| (d) | 跨來源共用的服務時間公式（用 FreeInference 第 1–10 週擬合） | **暫時沿用**，之後加 ±50% 的敏感度 | Bailian、Mooncake、WildChat 都沒有回應結束時間，沒有更好的來源；它們的 `think_ms` 一律標 `estimated` |
| (e) | 依週切分讓 54% 的 FreeInference 列落在測試段；測試週 198 個使用者裡有 133 個沒在訓練週出現過 | **照規格不改** | 這正是「這週訓練、下週測」加上「沒見過的使用者」的泛化測試；P3-4 要另報「只看見過的使用者」的子集 |
| (f) | 授權 | **需要使用者決定**：CacheWise 沒有 LICENSE，進論文前要問作者；WildChat 是 ODC-BY 的真實對話，統一表不存文字，但 `user_id` 是資料集本身的 `hashed_ip` | — |
| (g) | commit `code/p3_data` | 已 commit | — |
| — | H 級（半合成）產生器還沒做 | **留到 P3-1 正式版** | 規格 §8 沒列這項；目前沒有方向需要它 |
