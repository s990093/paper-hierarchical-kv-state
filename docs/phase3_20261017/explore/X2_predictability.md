# X2　回來時間猜得到嗎？LRU 以外有沒有訊號

（§2 在算任何 C-index、hazard 之前寫好。寫的時候只看過：各資料的欄位名稱、每份資料前幾列的結構、環境 agent 的下載紀錄與單位檢查（`datasets/*/README.txt`）。§0、§3 以後是結果出來才寫的。）

## 0. 一句話結論＋判定

**一句話**：真實 trace 上「誰會回來」有 LRU 以外的訊號，但那是**頻率**（這個 session 已經來過幾次），不是**時間**；而且一條不用學的 2Q 式規則就拿到它。「何時回來」（上一次間隔、上一輪輸出長度、預期剩餘時間）沒有一份 trace 穩定地贏 LRU。

**判定（原登記版，總結以此為準）：有看頭**（照 §2.4 字面）。
- 過關的只有 S2（目前為止的請求數）：Bailian coder ΔC **+0.123**（95% CI 0.116–0.130，主集合 19,297 個 session）、thinking **+0.148**（0.122–0.175，7,593 個）〔實測 20261017-175046-p3x2-bl-coder、20261017-175045-p3x2-bl-thinking〕。Bailian 的 session 是資料自帶的 `parent_chat_id`，不靠重建。
- Mooncake conversation 照主 session 的要求取三種重建規則中最小的 ΔC：S1 −0.039、S2 −0.012、S3 −0.042、S4 −0.039（D4 規則下 S2 +0.115、S1／S4 +0.052，strict 下 S1／S4 降到 +0.045；文獻規則對 S2 有構造上的偏差，§4.5）。Bailian traceA 最大 +0.008；WildChat 全部 <0（LRU 0.752 最好）。toolagent BLOCKED（重建驗算不過），traceB 沒有 session。
- **三個但書**（決定這個「有看頭」值多少）：
  1. 訊號是頻率：先比請求數、再比閒置時間的 2Q 式固定規則（S2lex，沒有參數）拿到 S2 的全部（coder +0.127、thinking +0.191、traceA +0.035）〔事後，§4.7〕。也就是說 LRU 以外的訊號**不需要學**。
  2. 只在「CPU 只裝得下最近 1–3 分鐘」時出現（流量大的叢集）：那一段正好是 hazard 上升到峰值的地方，LRU 幾乎是隨機（C 0.51–0.62）。WildChat 的 256 GiB 裝得下 17.5 小時，LRU 0.75，沒有任何 S1–S4 或 S2lex 贏它〔§4.6〕。
  3. 降級挑選：256 GiB 集合裡挑「預測最晚回來」的前 10%，300 s 內就回來的比例——隨機 0.235–0.262、LRU 0.169–0.240、S2lex 0.126–0.181、oracle 0（coder、traceA、Mooncake）〔事後，§4.7b〕。2Q 式規則把挑錯減少 25–40%，離 oracle 還很遠。
- **hazard 形狀**：不是單調下降。對話類先升後降（0–5 s 到 30–60 s 的峰值升 15–20 倍，之後降到 1/130–1/240）；coder／thinking／CacheWise 在 0–10 s 有尖峰再下降〔§4.3〕。「單調下降所以 LRU 接近最好」只在閒置超過約 1 分鐘後成立。

**判定（含 FreeInference 版）：有看頭**（不變，仍然只靠 Bailian coder、thinking 的 S2）。FreeInference 本身（「等使用者」組、256 GiB）S1–S4 的 ΔC 全部 <0（S2 −0.159、S1 −0.296），單看它是**死路**：agent session 99.4% 都有 ≥2 個請求，頻率分不出誰；256 GiB 只裝 17 個 session（每個 ~14.5 GiB）、涵蓋 2.8 小時，LRU 0.751 最好；降級挑錯率 LRU 4.0% vs oracle 1.0%〔實測 20261017-181823-p3x2-fi3、-181825-p3x2-fi-extra3〕。「等工具」那一組回來的中位是 1.7 s、99.7% 會回來，LRU 0.832。

## 1. 問題與要證明什麼

- 第一階段說「贏家由回來順序決定」（D3）；審查（`../review_opencode_20261017.md` §1）指出：那是 seed 隨機的工作負載，**順序重要不等於順序可預測**。
- 這裡只問一件事：**在真實 trace 上，降級當下「誰先回來」的排序，有沒有比 LRU（閒置越久越晚回來）更好的無參數訊號？**
- 只用無參數統計（CLAUDE.md §8）：Kaplan–Meier、life-table hazard、分箱條件分布、Harrell C-index。不擬合任何模型（包括 Cox），不從資料估查表。

## 2. 判準（開跑前寫）

**寫於**：2026-10-17T09:42:37Z（`date -u`）。

### 2.1 資料與 session 定義

| trace | session 怎麼來 | 「回來」的時刻 | 單位斷言（用 A 欄驗算 B 欄） |
|:--|:--|:--|:--|
| Mooncake conversation、toolagent | 前綴鏈重建：請求 r 的父＝「最後一個完整 block（位置 ≥1，不算共用的 block 0）出現在 r 的前綴裡」的最近一個請求（D4 `cmd_char` 的規則）。session＝父鏈的根。驗算：(a) 子的 `input_length` ≥ 父的 `input_length`＋父的 `output_length`×(1−容差) 的比例；(b) 父的扇出分布（一個父有幾個子）；(c) 父子間隔 ÷ 父的輸出 token（D4 §4.1 的單位檢查）。驗算不過（(a) <90%）就停下來回報，不換規則硬幹 | 下一個請求的 timestamp（毫秒→秒） | `len(hash_ids)==ceil(input_length/512)`（m8 `load_trace`）；timestamp 毫秒：父子間隔 ÷ 父輸出 token 的中位數要落在 0.01–10 s/token |
| Qwen-Bailian traceA、traceB、thinking、coder | `parent_chat_id` 串起來，session＝根。驗算：子的 `hash_ids` 前綴含父的 `hash_ids[:-1]`（最後一個 block 可能因 padding 改變，見上游 `docs/qa-context-growth-pattern.md`）；子的 `turn`＝父的 `turn`＋1 | 下一個請求的 timestamp（秒） | `len(hash_ids)==ceil(input_length/16)`；timestamp 秒：父子間隔 ÷ 父輸出 token 的中位數落在 0.01–10 s/token；間隔 < 父輸出×1 ms（≥1000 token/s 的解碼）的比例 <5% |
| WildChat-1M | `conversation_hash`（一個對話＝一個 session） | 下一輪 assistant 訊息的 timestamp（**資料只有 assistant 有時間戳**，所以「回來」＝下一輪回應完成的時刻，含生成時間；這是近似，照實標） | 頂層 `timestamp`＝最後一個 assistant 訊息的 timestamp；頂層 `turn`＝user 訊息數；同一對話內 assistant timestamp 不遞減 |
| CacheWise（只報告） | `session_id`；同一 `request_id` 的多筆 `llm_call` 只取第一筆 | 下一個 request 的第一筆 timestamp | tool 的 `execution_time_ms`＝`result_timestamp − timestamp`（毫秒） |

- **一次回合**＝session 裡的一個請求；它的**間隔**＝到同 session 下一個請求的時間。trace 結束前沒有下一個請求＝**右設限**（設限時間＝trace 結束 − 這個請求的時刻），不是「不回來」。
- session 的 KV 大小（定義快照集合用）＝最新一個請求的 (input＋output) token × 128 KiB（Llama-3.1-8B BF16，和 D4／X1 同單位）。WildChat 沒有 token 數：用 Llama-3.1 tokenizer 量一個隨機樣本的「字元／token」比，換算對話累積字元數；比值是實測，附 run_id。

### 2.2 分數（全部固定規則，方向事先定死；分數越大＝預測越晚回來）

| 分數 | 定義（在快照時刻 t） | 方向的依據 |
|:--|:--|:--|
| S0（LRU） | a＝t − 最後一個請求的時刻（已閒置多久） | LRU |
| S1 | 上一次間隔（最後兩個請求的時間差） | 上次隔得久，這次也久 |
| S2 | −（目前為止的請求數） | 輪數越多越快回來（WA／KVCache-in-the-wild 用「第幾輪」查重用機率；方向是我事先的假設〔判讀〕） |
| S3 | 最後一個請求的輸出長度（token；WildChat 用字元數） | Bidaw：回答越長，下次重用越晚 |
| S4 | 上一次間隔 − a | 預期剩餘時間 |

- S1、S4 在只有一個請求的 session 上沒有定義。**主比較**：S1、S4 只在「已有 ≥2 個請求」的子集上算，S0 也在同一個子集上重算，ΔC＝C(Sk)−C(S0) 用同一個子集。另外報告「補值版」（沒有上一次間隔時 S1＝a、S4＝0，全集合），不進判定。
- 反方向（1−C）只報告，不進判定。

### 2.3 快照（「同一時刻同時閒置的 session」）

- **時刻格點**：Mooncake 每 30 s，600 s 到（結束 − 600 s）；Bailian 每 60 s，900 s 到（結束 − 900 s）；WildChat 每 6 h，第一筆 + 7 天到最後一筆 − 7 天。CacheWise 的 session 時間戳各自平移過（看起來是 2020 年），不做快照。
- **閒置**：最後一個請求的時刻 ≤ t、下一個請求（如果有）> t。
- **主集合（判定用）**：LRU 在 CPU **256 GiB** 時會留在 CPU 的閒置 session——依 a 由小到大累加 KV 大小，加到 256 GiB 為止。這就是「降級當下要比的對象」的近似（假設 CPU 由 LRU 管）。
- **報告用集合**：128、512、1024 GiB；不設限（Mooncake、Bailian 是 trace 開始到 t 的全部閒置 session；WildChat 是過去 7 天內）。
- **剩餘時間**：R＝下一個請求 − t；沒有的話設限在 trace 結束 − t。
- **C-index**：Harrell，處理右設限：可比的一對＝較早的那個有回來（R 小且是事件），或兩者 R 相同而另一個是設限；分數同分算 0.5；兩個事件同時間不可比。**每個 trace 把所有快照的可比對數加總後算一個 C**（主值），另報各快照 C 的中位數、前半／後半段、快照 bootstrap（1,000 次，成對重抽快照）的 95% CI。
- **全體版本（只報告）**：每個回合結束當下（a＝0）一個樣本，T＝間隔，設限同上；S0 在這裡全部同分（C＝0.5，沒意義，照列），S4＝S1。
- **條件版本（只報告）**：只比「已閒置時間同一個箱子」的 session（箱：0–10、10–30、30–60、60–120、120–300、300–600、600–1200、1200–3600、>3600 s），回答「控制住 a 之後，Sk 還有沒有訊號」。

### 2.4 判定（照 README §2 X2，加嚴一條）

- **有看頭**：至少一個真實 trace，主集合的 session 數 ≥1,000（＝至少出現在一個主快照裡的不同 session），某個 S1–S4 的 ΔC ≥ 0.05，**而且**（加嚴）快照 bootstrap 的 ΔC 95% CI 下界 > 0。
- **死路**：所有真實 trace（不含 CacheWise）的主集合上，S1–S4 沒有一個 ΔC ≥ 0.02。
- 其他＝**可能**。
- hazard 形狀、KM、分箱、全體版本、條件版本、其他容量的集合：只報告，不改判定。
- 真實 trace＝Mooncake conversation、toolagent，Bailian 四份，WildChat。session 數 <1,000 的 trace 不能讓判定變「有看頭」，但仍算進「死路」的檢查。

### 2.4a 補充（2026-10-17T09:44:51Z，看過 session 重建的驗算、還沒算任何 C-index／hazard）

- **加嚴**：S1、S4 的比較在「≥2 個請求」的子集上做，所以「有看頭」另外要求**這個子集**的 session 數也 ≥1,000。
- **Mooncake 重建驗算的容差**：§2.1 (a) 寫了「×(1−容差)」但沒寫數字。看到診斷後才定為 0.1（理由：上游 Bailian 文件說下一輪會去掉特殊 token；Mooncake 也可能截斷）。conversation：容差 0 時 89.3%（差 0.7 個百分點不過）、0.1 時 93.9%；toolagent：0 時 78.3%、0.1 時 79.5%、0.5 時 80.9%。所以 **toolagent 不管容差取多少都不過 → toolagent 停在這裡（BLOCKED），不換重建規則硬幹**；conversation 在容差 0.1 下過，但這個數字是事後定的，照實列在 §6。
- **Bailian traceB**：172,800 筆全部 `parent_chat_id=−1`、`turn=1`（To-B API，沒有 session 結構）→ 不適用 X2，只列描述。

### 2.5 hazard 與描述統計（只報告）

- 每個 trace：session 數、請求數、有 ≥2 個請求的 session 比例、回合「有回來」的比例、間隔分位數（只算有回來的；另報 KM 中位數）、設限比例（回合中有多少是 trace 結束時還沒回來），以及「設限時間 < 有回來的間隔 p90」的比例（尾端看不清楚的部分）。
- KM：S(a) 在 10、30、60、120、300、600、1800、3600 s（WildChat 加 6 h、1 d、7 d）。
- life-table hazard：箱邊 0、5、10、20、30、45、60、90、120、180、240、300、450、600、900、1200、1800、2700、3600、5400、7200 s（WildChat 再加 3 h、6 h、12 h、1 d、3 d、7 d、30 d）；hazard＝d ÷ (n − c/2) ÷ 箱寬（actuarial）。形狀：列出每一箱，並報「相鄰兩箱都有 ≥30 個事件時，下降的比例」。
- 分箱條件分布：依輪數（1、2、3、4–5、6–10、≥11）、上一次間隔（無、<30、30–60、60–120、120–300、300–600、≥600 s）、上一輪輸出長度（<64、64–256、256–512、512–1024、≥1024 token），各報 n、KM 中位數、P(60 s 內回來)、P(300 s 內回來)、設限比例。

### 2.6 登記之後的補充（2026-10-17T10:07:00Z；照 README §4〔主 session 10:01Z〕與主 session 的訊息）

**寫的時候已經看過**：Mooncake conversation（D4 規則、本文的 strict 規則）、Bailian 四份、WildChat、CacheWise 的全部 X2 結果（§4）；文獻 agent 規則的 Mooncake 結果也已經跑完、看過（20261017-180543）。**FreeInference 的任何結果都還沒看過**（loader 在除錯時只印出進度）。

1. **Mooncake 至少兩種重建規則，判定照較保守（ΔC 較小）的那個**：(i) 登記的 D4 規則；(ii) 本文的 strict（在 D4 上切掉驗算 (a) 不過的連結、扇出 >1 只留最早的子；看過主結果才加）；(iii) 文獻 agent 的嚴格規則（`research/calc/mooncake_chain2.py` 照抄：共用前綴 ≤3 個請求、父唯一、時間嚴格較早）。每個分數取三者中最小的 ΔC 進判定。
2. **FreeInference agentic trace（登記之後才加）**：判定列兩個版本——**原登記版**（不含 FreeInference，總結以它為準）與**含 FreeInference 版**。FreeInference 的定義（寫在看結果之前）：
   - session＝`session_id`（資料自己重建、不跨週；子 agent 也算）。閒置從請求的 `end_ms` 起算；回來＝同 session 下一個請求的 `start_ms`；設限在該週最後一個 `end_ms`。上一次間隔＝這次 `start_ms` − 上次 `end_ms`（和其他 trace 的「到達到到達」不同，因為這份有結束時間）。
   - 快照：每週內每小時一個，避開每週頭尾 6 h；只看同一週的 session。集合、分數、C-index 同 §2.3（S2＝−目前為止的請求數，含工具觸發的請求；另報 S2u＝−使用者發起的回合數，只報告）。
   - **兩種尺度分開報**：閒置當下就知道的分組——這個請求的 `finish_reason` 是 `tool_calls` → 「等工具」，否則「等使用者」。LRU 集合是兩組一起算（同一個 CPU），C-index 在每組內各自算。**含 FreeInference 版的判定只用「等使用者」那組**（降級決策主要落在這裡）；「等工具」與全部只報告。
   - 單位斷言：`len(block_ids)==floor(Σrole_tokens/16)`；同 session 的 `step` 連續；`start_ms ≤ end_ms`；`ttft_ms ≤ end_ms − start_ms`；(end−start−ttft)÷completion_tokens 的中位數落在 0.5–500 ms/token（毫秒）；工具 `latency_ms` ≈ 下一個請求 start − 這個請求 end（只報告）。
   - session 大小用 `prompt_tokens＋completion_tokens`（o200k token）× 128 KiB（Llama 的位元組，和其他 trace 同單位）。

## 3. 做了什麼

全部只用 CPU（`TIARA_NO_GPU=1`），走記錄殼 `m7run`（`code/run3090.sh`），每個 run 目錄在 `/ssd8/hungwei/paper-hkv/runs/<run_id>/`（`cmd.sh`、`context.txt`、`stdout.log`、`stderr.log`、`exit_code`、逐快照計數 `snap_counts*.csv.gz`）。每一列結果的有效 run 列在 `results/p3_explore/x2/x2_runs.csv`。

| 步驟 | 指令（`source code/env_3090.sh; TIARA_NO_GPU=1 m7run <短名> python -I ...`） | run_id | 產出 |
|:--|:--|:--|:--|
| C-index／KM 單元測試 | `code/p3x/x2_stats.py` | 20261017-175022-p3x2-selftest | stdout：Harrell C＝暴力 O(n²) 300 次、＝`lifelines.utils.concordance_index` 50 次 |
| WildChat 字元／token | `code/p3x/x2_main.py --measure-wildchat-cpt` | 20261017-175040-p3x2-wccpt | 3.8040（Llama-3.1 tokenizer，2,002 個隨機對話，seed 0） |
| 主分析（每個 trace 一個 run） | `code/p3x/x2_main.py --trace <trace>` | mc-conversation 20261017-175040；mc-toolagent 20261017-175041（**exit 5，驗算不過**）；bl-traceA 175042；bl-traceB 175043；bl-thinking 175045；bl-coder 175046；cachewise2 175316；wc 175255（皆 20261017-） | `x2_validation.csv`、`x2_summary.csv`、`x2_km.csv`、`x2_hazard.csv`、`x2_bins.csv`、`x2_cindex.csv`、`x2_cindex_global.csv` |
| Mooncake 第三種重建規則（文獻 agent 的） | `x2_main.py --trace mooncake_conversation_lit`；`x2_posthoc.py --trace mooncake_conversation_lit` | 20261017-180543-p3x2-mc-lit；20261017-180559-p3x2-ph-mc-lit | 同上 |
| 事後：2Q 式固定規則 S2lex | `code/p3x/x2_posthoc.py --trace <trace>` | 20261017-175443／175444／175445／175446／175447-p3x2-ph-* | `x2_cindex_posthoc.csv` |
| 事後：Mooncake strict 重建 | `x2_posthoc.py --trace mooncake_conversation_strict --full-variants` | 20261017-175941-p3x2-ph-mc-strict2 | `x2_cindex_posthoc.csv` 與其他 CSV 的 `mooncake_conversation_strict` 列 |
| 事後：集合涵蓋的閒置時間 | `code/p3x/x2_setspan.py --trace <trace>` | 20261017-175643／175644／175645／175646／175647-p3x2-span-* | `x2_setspan.csv` |
| FreeInference block 斷言（逐週平行） | `x2_main.py --fi-block-check` | 20261017-181642-p3x2-fi-blocks | `x2_validation.csv` |
| FreeInference 主分析（登記後加入） | `x2_main.py --trace freeinference` | 20261017-181823-p3x2-fi3（前兩次失敗：181030 我停掉、181644-fi2 exit 1，見 §6） | 同主分析 |
| FreeInference 事後（S2lex、涵蓋時間、降級挑選） | `code/p3x/x2_fi_extra.py` | 20261017-181825-p3x2-fi-extra3（前兩次 exit 1，見 §6） | `x2_cindex_posthoc.csv`、`x2_setspan.csv`、`x2_victims.csv` |
| 事後：降級挑選（其他 trace） | `code/p3x/x2_victims.py --trace <trace>` | 20261017-181151～181157-p3x2-vic-* | `x2_victims.csv` |

- 程式：`code/p3x/x2_load.py`（載入、session 重建、單位斷言）、`x2_stats.py`（Harrell C〔numba Fenwick tree〕、KM、life table）、`x2_main.py`（描述、hazard、分箱、快照 C-index、bootstrap）、`x2_posthoc.py`、`x2_setspan.py`。`m8_trace_oracle.load_trace` 只 import 不改（在 process 內把 `TRACE_DIR` 指到 `/ssd8/.../datasets/mooncake/traces`）。
- 資料都是環境 agent／主 session 下載、驗過 sha256 的副本（`/ssd8/hungwei/paper-hkv/datasets/*/README.txt`）：Mooncake（git blob 驗過）、Bailian commit `5f7439c5`、WildChat-1M revision `7d6490e4`（**不 gated**）、CacheWise commit `181c435a`、FreeInference revision `16d3c04d`。

## 4. 結果

### 4.1 資料狀態與單位斷言（規則 6）

| trace | 狀態 | 斷言（值） | 來源 |
|:--|:--|:--|:--|
| Mooncake conversation | 可用 | `len(hash_ids)==ceil(input/512)` 1.000；前綴鏈 0 違反；子 input ≥ 父 input＋0.9×父 output 0.939（容差 0 時 0.893）；父扇出 >1 0.7%（最多 8）；父子間隔÷父輸出 中位 0.344 s/token（→毫秒） | 〔實測 20261017-175040-p3x2-mc-conversation，x2_validation.csv〕 |
| Mooncake toolagent | **BLOCKED** | 子 input 一致比例 0.795 < 0.90（容差 0 時 0.783）；父扇出 >1 28.0%（最多 46）；最大的「session」有 9,202 個請求＝共用前綴把不同 session 併在一起 | 〔實測 20261017-175041-p3x2-mc-toolagent，exit 5〕 |
| Bailian traceA | 可用 | `ceil(input/16)` 1.000；子 hash 前綴＝父 [:-1] 1.000；子 turn＝父＋1 1.000；間隔÷父輸出 中位 0.288 s/token（→秒）；<1 ms/token 0% | 〔實測 20261017-175042-p3x2-bl-traceA〕 |
| Bailian traceB | 不適用 | 172,800 筆全是根請求（`parent_chat_id=−1`、`turn=1`） | 〔實測 20261017-175043-p3x2-bl-traceB〕 |
| Bailian thinking | 可用（≥2 輪的子集太小） | 前綴 0.963；turn 1.000；間隔÷父輸出 中位 0.036 s/token；<1 ms/token 0.43% | 〔實測 20261017-175045-p3x2-bl-thinking〕 |
| Bailian coder | 可用 | 前綴 0.996；turn 1.000；中位 0.199 s/token；<1 ms/token 0% | 〔實測 20261017-175046-p3x2-bl-coder〕 |
| WildChat-1M | 可用 | 頂層 timestamp＝最後一則 assistant 1.000；turn＝user 訊息數 1.000；assistant 時間不遞減 1.000；交替 1.000；間隔 < 下一則回應字元/2000 的比例 0.00001 | 〔實測 20261017-175255-p3x2-wc〕 |
| CacheWise（只報告） | 可用 | 工具 `execution_time_ms`＝result−ts（±1 ms）1.000（n＝11,614）；manifest 94 個 session，其中 13 個沒有任何 llm_call → 81 個 | 〔實測 20261017-175316-p3x2-cachewise2〕 |

### 4.2 描述統計（session、回來比例、間隔、設限）

| trace | session | 請求 | ≥2 個請求的 session | 回合有回來 | 有回來的間隔 p10／50／90／99（s） | 設限的回合 | 設限中「設限時間 < 間隔 p90」 |
|:--|--:|--:|--:|--:|:--|--:|--:|
| Mooncake conversation（D4 規則） | 8,056 | 12,031 | 2,012（25.0%） | 33.0% | 39／123／474／1,461 | 67.0% | 17.6% |
| 同上，strict | 8,335 | 12,031 | 1,970 | 30.7% | — | — | — |
| 同上，文獻規則 | 9,595 | 12,031 | 1,959（20.4%）；**每個 session 最多 3 個請求** | 20.2% | 45／132／526／1,549 | 79.8% | 17.3% |
| Bailian traceA | 23,101 | 43,058 | 9,012（39.0%） | 46.3% | 27／111／719／2,207 | 53.7% | 15.4% |
| Bailian thinking | 9,612 | 10,812 | 509（5.3%） | 11.1% | 5／29／255／1,533 | 88.9% | 2.1% |
| Bailian coder | 26,406 | 43,011 | 6,693（25.3%） | 38.6% | 6／118／942／3,489 | 61.4% | 16.9% |
| Bailian traceB | 172,800 | 172,800 | 0 | 0% | — | 100% | — |
| WildChat-1M | 837,989 | 1,960,074 | 331,483（39.6%） | 57.2% | 28／128／1,253／27,511 | 42.8%（≈對話結束） | 0.002% |
| CacheWise（只報告） | 81 | 11,498 | 80 | 99.3% | 0／7.0／48／399 | 0.7% | — |

來源：〔實測，x2_summary.csv；run_id 同 §4.1，strict＝20261017-175941-p3x2-ph-mc-strict2，文獻規則＝20261017-180543-p3x2-mc-lit〕。Mooncake／Bailian 的間隔是「到達到到達」（含伺服器生成時間）；WildChat 是「回應完成到回應完成」（含下一則的生成時間）。

- **trace 尾端設限**：Bailian、Mooncake 只有 1–2 小時，設限的回合裡 15–18% 的設限時間比「有回來的間隔 p90」還短（尾端看不清楚）；thinking 只有 2.1%（它的間隔短）。WildChat 的設限幾乎都是對話真的結束。

### 4.3 hazard 形狀（不是單調下降）

life-table hazard（每秒，×10⁻³；全部回合）：

| 箱（s） | Mooncake conv | traceA | coder | thinking | WildChat | CacheWise |
|:--|--:|--:|--:|--:|--:|--:|
| 0–5 | 0.017 | 0.209 | **7.104** | 2.214 | 0.184 | 74.0 |
| 5–10 | 0.401 | 0.929 | 2.298 | **3.091** | 1.143 | **131.9** |
| 20–30 | 1.020 | 3.123 | 1.606 | 1.716 | 3.273 | 31.1 |
| 30–45 | 1.409 | 3.208 | 1.915 | 0.818 | **3.595** | 23.4 |
| 45–60 | **2.173** | **3.245** | 1.821 | 0.380 | 3.444 | 20.0 |
| 60–90 | 2.123 | 2.457 | 1.604 | 0.275 | 2.880 | 14.5 |
| 120–180 | 1.020 | 1.216 | 0.866 | 0.118 | 1.558 | 7.6 |
| 300–450 | 0.356 | 0.403 | 0.287 | 0.021 | 0.483 | 2.7 |
| 900–1200 | 0.022 | 0.095 | 0.072 | 0.003 | 0.101 | 0.44 |
| 1800–2700 | 0.009 | 0.023 | 0.025 | 0.000 | 0.027 | 0.27 |

來源：〔實測，x2_hazard.csv，stratum＝all〕。完整的箱、第 1 輪／≥2 輪分層都在 CSV。

- **形狀**：對話類（Mooncake、traceA、WildChat）是**先升後降**——從 0–5 s 到峰值上升 15–20 倍（traceA 0.209→3.245、WildChat 0.184→3.595；Mooncake 第一箱只有 1 個事件）——使用者在讀、在打字，Mooncake／Bailian 的間隔還含生成時間；峰值在 30–60 s，之後一路下降，到 30–45 分鐘只剩峰值的 1/130–1/240。coder、thinking、CacheWise 在 0–10 s 有一個尖峰（很快的自動後續請求），之後下降（coder 在 30–60 s 還有一個小峰）。
- 「相鄰兩箱（各 ≥30 事件）中下一箱較小的比例」：Mooncake 0.77、traceA 0.71、coder 0.89、thinking 0.82、WildChat 0.85、CacheWise 0.91〔實測，x2_summary.csv〕——**都不是單調**。
- 第 1 輪與 ≥2 輪的 hazard 形狀相同，但 ≥2 輪在每一箱都高約 1.5–3 倍（例：traceA 45–60 s 3.78 vs 2.79；Mooncake 45–60 s 3.35 vs 1.62）〔實測，x2_hazard.csv〕。
- 〔判讀〕「hazard 單調下降時 LRU 本來就接近最好」只在閒置超過峰值（約 1 分鐘）之後成立；閒置不到 1 分鐘的 session，越近期的反而越不會馬上回來。

### 4.4 KM 與分箱條件分布

KM S(a)（還沒回來的比例）：

| a | Mooncake | traceA | coder | thinking | WildChat |
|:--|--:|--:|--:|--:|--:|
| 60 s | 0.925 | 0.857 | 0.875 | 0.924 | 0.842 |
| 300 s | 0.731 | 0.643 | 0.717 | 0.897 | 0.593 |
| 3600 s | 0.641 | 0.510 | 0.594 | 0.888 | 0.453 |
| 7 d | — | — | — | — | 0.428 |

P(300 s 內回來)，依目前輪數：

| 輪數 | Mooncake | traceA | coder | thinking | WildChat |
|:--|--:|--:|--:|--:|--:|
| 1 | 0.194 | 0.301 | 0.175 | 0.049 | 0.283 |
| 2 | 0.325 | 0.380 | 0.378 | 0.439 | 0.433 |
| 3 | 0.414 | 0.406 | 0.415 | 0.575 | 0.471 |
| 4–5 | 0.497 | 0.435 | 0.482 | 0.647 | 0.502 |
| 6–10 | 0.636 | 0.540 | 0.592 | 0.611 | 0.548 |
| ≥11 | 0.815 | 0.730 | 0.769 | 0.586 | 0.646 |

依上一次間隔（s）：0–30 → ≥600 的 P(300 s 內)：Mooncake 0.644 → 0.113；traceA 0.499 → 0.306；coder 0.682 → 0.290；WildChat 0.702 → 0.266（中間各箱單調下降）。依上一輪輸出長度：Mooncake 0.219（<64）→ 0.348（≥1024），大致上升、不單調，**和 Bidaw 的方向相反**；traceA 0.29–0.37 幾乎平；WildChat（字元）0.583 → 0.312，和 Bidaw 同向。來源：〔實測，x2_km.csv、x2_bins.csv〕。

- 〔判讀〕輪數是最乾淨的單調訊號，五份都一樣；上一次間隔也單調，但它和「已閒置多久」高度重疊（見 4.6）。

### 4.5 主結果：C-index（主集合＝LRU 在 CPU 256 GiB 時留在 CPU 的閒置 session）

ΔC＝C(Sk) − C(S0)（S1、S4 在 ≥2 個請求的子集上和同子集的 S0 比；S2、S3 在全集合上比）；括號是快照 bootstrap 95% CI。

| trace | 集合裡的 session（≥2 請求） | C(S0) 全／子集 | ΔS1 | ΔS2 | ΔS3 | ΔS4 | run_id |
|:--|:--|:--|:--|:--|:--|:--|:--|
| Mooncake conv，D4 規則 | 5,512（1,451） | 0.507／0.509 | **+0.052**（0.037, 0.068） | **+0.115**（0.103, 0.126） | −0.015 | **+0.052**（0.035, 0.068） | 20261017-175040-p3x2-mc-conversation |
| Mooncake conv，strict | 5,685（1,419） | 0.505／0.509 | +0.045（0.029, 0.062） | **+0.113**（0.101, 0.124） | −0.022 | +0.044（0.027, 0.062） | 20261017-175941-p3x2-ph-mc-strict2 |
| Mooncake conv，文獻規則 | 6,597（1,342） | 0.506／0.508 | −0.039 | −0.012 | −0.042 | −0.039 | 20261017-180543-p3x2-mc-lit |
| **Mooncake，三者取最小（進判定）** | — | — | −0.039 | −0.012 | −0.042 | −0.039 | — |
| Bailian traceA | 18,596（7,300） | 0.573／0.555 | −0.052 | +0.008（−0.005, 0.019） | −0.133 | −0.071 | 20261017-175042-p3x2-bl-traceA |
| Bailian coder | 19,297（5,175） | 0.521／0.524 | −0.000 | **+0.123**（0.116, 0.130） | −0.115 | −0.004 | 20261017-175046-p3x2-bl-coder |
| Bailian thinking | 7,593（**205**） | 0.621／0.627 | −0.113† | **+0.148**（0.122, 0.175） | −0.028 | −0.158† | 20261017-175045-p3x2-bl-thinking |
| WildChat-1M | 808,093（319,525） | 0.752／0.746 | −0.394 | −0.083 | −0.234 | −0.514 | 20261017-175255-p3x2-wc |

† 子集只有 205 個 session（<1,000），不能進「有看頭」。來源：〔實測，x2_cindex.csv、x2_cindex_posthoc.csv，set＝lru256GiB、strat＝none〕。

- **前半／後半段**：coder S2 0.647／0.640 vs S0 0.525／0.516；thinking S2 0.793／0.742 vs S0 0.618／0.623；Mooncake（D4）S2 0.615／0.626 vs S0 0.505／0.508，S1 的 ΔC 前半 0.046、後半 0.057〔實測，同上〕。
- **文獻規則對 S2 是構造上的偏差**〔判讀，有數字支持〕：它只在「共用前綴 ≤3 個請求」時才連，一個真實 session 只有最後兩條連結會被保留（第 k 個請求最後一個完整 block 被第 k…n 個請求共用）。所以「已經 ≥2 輪」在這個規則下＝「接近這個 session 的結尾」，而且它用到了未來的請求數。實測：文獻規則下第 3 輪的請求 **100% 設限、沒有一個回來**（D4 規則下第 3 輪 300 s 內回來 41%）〔實測 x2_bins.csv〕。照主 session 的要求判定仍取它，但這個 −0.012 不代表「輪數沒有訊號」。

### 4.6 其他集合、條件版本、全體版本（只報告）

C(S0) 與 C(S2) 隨集合變大：

| trace | 128 GiB | 256 | 512 | 1024 | 全部閒置 |
|:--|:--|:--|:--|:--|:--|
| Mooncake S0／S2 | 0.494／0.621 | 0.507／0.621 | 0.542／0.626 | 0.593／0.628 | 0.820／0.633 |
| traceA | 0.530／0.572 | 0.573／0.581 | 0.623／0.590 | 0.677／0.594 | 0.867／0.603 |
| coder | 0.516／0.646 | 0.521／0.644 | 0.548／0.646 | 0.584／0.650 | 0.753／0.661 |
| thinking | 0.605／0.776 | 0.621／0.769 | 0.644／0.765 | 0.703／0.765 | 0.801／0.692 |
| WildChat | 0.746／0.664 | 0.752／0.670 | 0.785／0.673 | 0.829／0.674 | 0.892／0.679 |

**集合涵蓋多少閒置時間**（各快照中位數；事後加的描述）：

| trace | 256 GiB 集合裡最久的閒置 | 集合大小 | 集合裡 trace 結束前會回來的比例 |
|:--|--:|--:|--:|
| Mooncake | 51 s | 176 | 34% |
| coder | 57 s | 322 | 36% |
| traceA | 150 s | 757 | 39% |
| thinking | 172 s | 245 | 3.2% |
| WildChat | 63,120 s（17.5 h） | 1,762 | 2.3% |

來源：〔實測，x2_cindex.csv；x2_setspan.csv，20261017-175643～175647-p3x2-span-*〕。

- 〔判讀〕**LRU 好不好，看 CPU 裝得下多久的歷史**。流量大的叢集（Mooncake、Bailian）256 GiB 只裝得下最近 1–3 分鐘——正好落在 hazard 上升到峰值的那一段，所以「越近期越快回來」幾乎是隨機（C 0.51–0.62）。WildChat 的流量小，256 GiB 裝得下 17 小時，大部分是已經結束的對話，LRU 就好（0.75），沒有任何 S1–S4 贏它。集合越大，S0 越好，S2 幾乎不變。
- **條件版本**（只比同一個閒置時間箱子裡的 session；256 GiB）：S0 0.505（Mooncake）、0.520（traceA）、0.512（coder）、0.540（thinking）、0.652（WildChat）；S2 0.619、0.585、0.642、0.756、0.678；S1 0.562、0.494、0.517、0.513、0.333〔實測，x2_cindex.csv strat＝agebin〕。**控制住閒置時間後，輪數在五份 trace 都還有訊號（含 WildChat）**；上一次間隔只有 Mooncake 有。
- **全體版本**（每個回合結束當下，a＝0；S0 全同分＝0.5）：S1（≥2 輪子集）0.579（Mooncake）、0.560（traceA）、0.636（coder）、0.614（thinking）、0.625（WildChat）、0.720（CacheWise）；S2 0.618、0.562、0.657、0.753、0.612、0.464；S3 0.503、0.513、0.498、0.755、0.599、0.498〔實測，x2_cindex_global.csv〕。回合剛結束時，上一次間隔確實有點訊號，但一到快照（同時閒置、a 不同）就被 a 蓋過。
- **反方向**（只報告）：WildChat 的 S4 反過來（a − 上一次間隔，「逾期越久越晚回來」）0.768 vs S0 0.746；coder 的 S3 反過來（輸出越長越早回來）0.594 vs 0.521〔實測，x2_cindex.csv 的 C_flipped〕。方向是事先定的，這些不進判定。

### 4.7 事後：不用學的「頻率＋近期」能拿到多少（S2lex，只報告）

S2lex＝先比請求數（多的預測先回來），同請求數再用 LRU——就是 2Q／SLRU 式「看過兩次的放保護區」的 session 版，沒有參數。256 GiB 集合、全集合：

| trace | C(S0) | C(S2) | C(S2lex) | ΔS2lex（95% CI） |
|:--|--:|--:|--:|:--|
| Mooncake（D4） | 0.507 | 0.621 | 0.623 | +0.116（0.106, 0.126） |
| Mooncake（文獻規則） | 0.506 | 0.495 | 0.499 | −0.008 |
| traceA | 0.573 | 0.581 | 0.608 | +0.035（0.026, 0.043） |
| coder | 0.521 | 0.644 | 0.648 | +0.127（0.122, 0.133） |
| thinking | 0.621 | 0.769 | 0.812 | +0.191（0.172, 0.211） |
| WildChat | 0.752 | 0.670 | 0.727 | −0.025（−0.030, −0.020） |

來源：〔實測，x2_cindex_posthoc.csv，20261017-17544x-p3x2-ph-*、20261017-180559-p3x2-ph-mc-lit〕。

- 〔判讀〕S2 的好處幾乎全部被這條不用學的規則拿到。WildChat 上它輸給 LRU，因為 17 小時的集合裡「近期」比「輪數」重要，而字典序讓輪數完全蓋過近期（但同一個閒置箱子裡 S2lex 0.712 > S0 0.652）。要把兩者加權合在一起就需要參數，這一輪不准做（CLAUDE.md §8）。

### 4.7b 事後：降級時挑得準不準（只報告）

C-index 看整個排序；降級只要挑出集合裡最晚回來的一小撮。每個快照在 LRU 集合裡挑「預測最晚回來」的前 10%（至少 1 個），看它們之中 300 s 內就回來（挑錯、很快又要還原）與 trace 結束前回來的比例；「隨機」＝集合的基準率；oracle 照真實剩餘時間挑。

| trace（256 GiB） | 隨機 300 s／結束前 | LRU | S2lex | oracle |
|:--|:--|:--|:--|:--|
| Mooncake（D4） | 0.262／0.341 | 0.240／0.317 | 0.181／0.251 | 0／0 |
| Mooncake（strict） | 0.244／0.320 | 0.226／0.301 | 0.167／0.234 | 0／0 |
| Mooncake（文獻規則） | 0.150／0.204 | 0.132／0.187 | 0.148／0.201 | 0／0 |
| traceA | 0.259／0.383 | 0.169／0.295 | 0.126／0.224 | 0／0 |
| coder | 0.235／0.355 | 0.211／0.327 | 0.131／0.210 | 0／0 |
| thinking | 0.024／0.033 | 0.013／0.020 | 0.007／0.010 | 0／0 |
| WildChat | 0.003／0.024 | 0.000／0.008 | 0.000／0.003 | 0／0 |

來源：〔實測，x2_victims.csv，20261017-181151～181157-p3x2-vic-*〕。1024 GiB 時 LRU 的挑錯率降到 0.035–0.111（300 s 內），S2lex 0.027–0.079。

- 〔判讀〕流量大的 trace 上，LRU 挑出來的「最該降級」的 session 有 17–24% 在 5 分鐘內就回來，和隨機（24–26%）差不多（traceA 例外，好一些）；2Q 式規則降到 13–18%；oracle 是 0。所以「挑錯」的空間很大，但能用的無參數訊號只拿到其中一小部分。WildChat 上 LRU 幾乎不挑錯（0.8%），沒有空間。這些比例要乘上「挑錯一次多花多少還原時間」才是 TTFT，那是 X1 的事。

### 4.8 CacheWise（只報告；81 個 session）

- 間隔中位 7.0 s、p90 48 s；hazard 峰值在 5–10 s；回合結束當下 S1（≥2 輪）C 0.720、S2 0.464、S3 0.498〔實測 20261017-175316-p3x2-cachewise2〕。agent 的請求大多是工具觸發，秒級回來；session 數太少，不能判定。

### 4.9 FreeInference（登記之後加入；含 FreeInference 版的判定）

**資料**：16 週、14,036 個 session（12,002 個主 session＋2,034 個子 agent）、267 個使用者、1,222,199 個請求〔實測 20261017-181823-p3x2-fi3〕（README 寫 1,186,582 個請求，差 3.0%，原因沒查）。斷言：`len(block_ids)==floor(Σrole_tokens/16)` 1.000（逐週全部 1.000）〔實測 20261017-181642-p3x2-fi-blocks〕；step 連續 1.000；start ≤ end 1.000；ttft ≤ end−start 1.000；解碼 3.98 ms/token（→毫秒）；工具 `latency_ms`＝下一個 start − 這個 end 1.000（n＝1,134,643）；17 個請求在上一個請求結束前就開始（閒置開始改用累積最晚結束，§6）〔實測 fi3〕。

**兩種尺度**（閒置當下就知道的分組；和下一個請求的 `step_trigger` 一致的比例：等使用者 98.1%、等工具 99.98%）：

| | 回合數 | 有回來 | 有回來的間隔 p10／50／90／99（s） | KM 中位 |
|:--|--:|--:|:--|--:|
| 等使用者 | 85,011 | 87.9% | 2.8／93／1,015／29,020 | 123 s |
| 等工具 | 1,137,188 | 99.7% | 0.41／1.7／17／221 | 1.7 s |

hazard（×10⁻³/s）：等使用者 0–5 s 25.6、5–60 s 約 4.8–5.6（平）、120–180 s 3.0、600–900 s 0.47、1800–2700 s 0.090；等工具 0–5 s 242、單調下降到 1800–2700 s 0.075。相鄰箱下降比例：等使用者 0.89、等工具 0.95、全部 1.00〔實測 fi3，x2_hazard.csv〕。**和對話類不同，這裡沒有「先升」那一段**（agent 的使用者回合沒有讀長回答的延遲，或被 0–5 s 的快速後續蓋過〔判讀〕）。

**C-index（256 GiB 集合）**：

| 集合 | session（≥2 請求） | C(S0) | ΔS1 | ΔS2 | ΔS3 | ΔS4 | ΔS2u | ΔS2lex（事後） |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| **等使用者（進判定）** | 7,195（7,117） | 0.751 | −0.296 | −0.159 | −0.276 | −0.499 | −0.160 | −0.157 |
| 等工具 | 3,930（3,889） | 0.832 | −0.383 | −0.245 | −0.346 | −0.633 | −0.255 | −0.243 |
| 全部 | 9,707（9,618） | 0.778 | −0.338 | −0.189 | −0.288 | −0.549 | −0.188 | −0.187 |

來源：〔實測 x2_cindex.csv（fi3）、x2_cindex_posthoc.csv（20261017-181825-p3x2-fi-extra3）〕。CI 全部在 0 以下（例：等使用者 ΔS2 −0.172 到 −0.146）。

- **256 GiB 只裝得下 17 個 session、最久閒置 2.8 小時**（中位數；每個 session 平均 ~14.5 GiB，因為 prompt 中位 ~89K token）〔實測 x2_setspan.csv，fi-extra3〕。和 WildChat 一樣是「CPU 涵蓋很長的歷史」那一型，LRU 好（0.75–0.83）。
- **輪數沒有訊號**：99.4% 的 session 有 ≥2 個請求，P(300 s 內回來) 依輪數 0.949–0.969 幾乎平〔實測 x2_bins.csv〕。agent session 幾乎都「回來過」，頻率分不出誰。條件版本（同一閒置箱子）S2 0.605 vs S0 0.674（等使用者）〔實測〕。
- **降級挑選**（256 GiB、等使用者、前 10%）：300 s 內回來的比例 LRU 0.040、S2lex 0.115、隨機 0.115、oracle 0.010；結束前回來 LRU 0.222、oracle 0.060〔實測 x2_victims.csv，fi-extra3〕。LRU 已經接近 oracle（300 s 尺度），2Q 式規則反而等於隨機。
- 回合結束當下（a＝0）的 S1 C 0.705（≥2 請求、全部）主要是「等工具／等使用者」混在一起造成的：分組後等使用者只有 0.505〔實測 x2_cindex_global.csv〕。

## 5. 前作

| 前作 | 說了什麼 | 和本文的關係 | 讀到哪裡 |
|:--|:--|:--|:--|
| **KVCache Cache in the Wild**（Wang 等，ATC'25；arXiv 2506.02634；就是 Bailian trace 的論文） | 「single-turn requests demonstrate significantly lower rates of issuing next-turn requests (mean=0.5)」，多輪 0.8（§3.3、Fig. 11）；「the probability distribution of KV$ reuse time follows an exponential distribution」（給定請求類別，§3.4）；WA 在背景「fit an exponential distribution to the sampled data」再查表，「We don't consider frequency explicitly」（§4.2）；WA 比最好的基準命中率高 1.5–3.9%（§4.3）；Trace B「has little workload information」，WA 等於 LRU（§4.3） | 輪數有訊號：一致（本文 S2）。**指數分布（常數 hazard）和本文不一致**：traceA 的 hazard 從 0–5 s 到 45–60 s 升 15 倍、之後降到 1/140（§4.3）；不過他們依「類型＋輪數」分層、本文只依輪數分層，不是直接矛盾〔判讀〕。traceB 沒有 session：一致（本文 §4.1） | 〔原文 §3.3、§3.4、§4.2、§4.3，經 WebFetch 讀 arXiv HTML 的前 100k 字元〕 |
| Bidaw（Hu 等，FAST'26） | 上一輪回答長度預測下次重用距離；換成 Poisson 時間戳就沒效 | 本文 S3（回答越長越晚回來）：在 Mooncake、Bailian 方向相反或沒有訊號，只有 WildChat 的分箱同向，但 C-index 仍輸 LRU | 〔D4 §5 轉述（D4 讀原文 p.12），本次沒讀原文〕 |
| LPC（Yang 等，NeurIPS'25） | 用對話文字預測會不會續聊，只補回 LRU→oracle 差距的 22–66% | 內容型預測；本文只用負載訊號 | 〔D4 §5 轉述（原文 p.23），本次沒讀〕 |
| When Fancy Eviction Fails（Liu 等，arXiv 2609.28870） | 第 2 次命中後再被重用的機率 69.4% → 88.7%，所以頻率適合當准入 | 和本文「輪數＝頻率」的訊號同一件事；2Q 式規則拿到大部分（§4.7） | 〔D4 §5 轉述（原文 p.8），本次沒讀〕 |
| Mooncake（Qin 等，FAST'25） | CPU pool 用 LRU；「不可能準確預測未來使用量」 | 本文：時間（S1、S3、S4）確實沒有穩定的訊號；頻率有 | 〔D4 §5 轉述，本次沒讀〕 |
| 2Q、SLRU、ARC 等經典快取 | 「看過兩次」的放保護段 | 本文 S2lex 就是它們的 session 版 | 〔判讀，本次沒查原文〕 |

## 6. 偏離與問題

1. **Mooncake 重建驗算的容差是看到診斷後才定的**（§2.4a）：容差 0 時 conversation 89.3%（差 0.7 個百分點不過）。容差只決定「過不過」，不影響重建本身；S2 在三種規則下的結果見 §4.5。
2. **Mooncake toolagent：BLOCKED**。驗算 (a) 0.795 < 0.90；父扇出 >1 28%；一個「session」9,202 個請求。完整錯誤（stderr，run 20261017-175041-p3x2-mc-toolagent，exit 5）：`x2_load.ValidationError: mooncake_toolagent: 驗算不過 {'check': 'child_input>=parent_input+(1-0.1)*parent_output', 'value': 0.7948780487804878, 'threshold': '>=0.9', 'passed': False, 'note': 'n_pairs=16400；容差 0 時 0.7828'}`。沒有換規則再跑。
3. **Mooncake 的第二、三種重建規則**（strict、文獻規則）都是看過 D4 規則的結果才加的；文獻規則是主 session 10:01Z 要求的。判定照主 session 的要求取三者最小，但文獻規則對 S2 有構造上的偏差（§4.5 最後一條）。
4. **Bailian traceB 不適用**（沒有 session）。**thinking 的 ≥2 輪子集只有 205 個 session**，S1、S4 不能進「有看頭」。
5. **CacheWise 第一次跑有 bug**：208 筆 `request_id=None` 的 llm_call 被我併成一筆（20261017-175047-p3x2-cachewise，作廢）；修正後重跑 20261017-175316-p3x2-cachewise2。兩次的列都還在 CSV 裡，用 `x2_runs.csv` 分辨。另外 manifest 的 94 個 session 裡有 13 個沒有 llm_call。CacheWise 的 session 時間戳各自平移，不做快照。repo 沒有 LICENSE（NEEDS_USER：用在論文前要問作者，見資料 README.txt）。
6. **「回來」的時刻**：Mooncake、Bailian 是請求到達（間隔含伺服器生成時間，pitfall #14 說標籤應該用使用者思考時間）；WildChat 只有 assistant 的時間戳，回來＝下一則回應完成（含生成時間）。WildChat 的「上一輪輸出長度」用字元數，但分箱沿用 token 的邊界（64／256／512／1024），字元比 token 長約 3.8 倍。
7. **WildChat 的 KV 大小**用量到的 3.804 字元／token 換算；「全部閒置」集合只看過去 7 天。
8. **快照 bootstrap 低估變異**：同一個 session 會出現在很多快照裡，重抽快照沒有處理這個相依。另報了前半／後半段（§4.5）。
9. **快照集合假設 CPU 由 LRU 管**：換別的策略，CPU 裡的 session 會不同。
10. **trace 開頭的左截斷**：trace 開始前就在的 session，輪數被少算。前後半段的 S2 差不多（§4.5），影響看起來不大〔判讀〕。
11. **事後加的分析**（S2lex、集合涵蓋時間、strict 重建、文獻規則）都在看過主結果後才加，只報告。
12. **記錄殼被別的 agent 改寫**：20261017-175914-p3x2-ph-mc-strict 執行到一半，`code/run3090.sh` 被改（我沒有改它），殼在第 111 行報 `syntax error near unexpected token 'then'`、沒寫 exit_code；Python 本身跑完了（stdout 有 `posthoc done`）。照樣重跑成 20261017-175941-p3x2-ph-mc-strict2（exit 0），第一次的列不要用。
13. 所有 trace 的 KV 大小都用 Llama-3.1-8B 的 128 KiB／token（和 D4、X1 同單位），不管實際服務的模型。
14. **FreeInference 是登記之後才加的**（README §4，主 session 10:01Z）。我在 2026-10-17T10:07:00Z 寫 §2.6 的定義；那時已經看過 Mooncake（三種規則）、Bailian、WildChat、CacheWise 的全部 X2 結果，還沒看過任何 FreeInference 結果。
15. **FreeInference 跑了四次才成**：
    - 20261017-181030-p3x2-fi：我手動停掉（exit 143，留 INTERRUPTED）；讀 8 GB block_ids 太慢，而且同一份程式已確定會失敗（下一條）。
    - 20261017-181359-p3x2-fi-extra（exit 1）：`AssertionError: session 內時間倒退`——同一 session 有 17 個請求在上一個請求結束前就開始（重疊）。改成「閒置從到目前為止所有請求的最晚結束起算」，上一次間隔 <0 時記 0。這是定義的變更，不是單位錯誤（start 時間本身單調）。
    - 20261017-181644-p3x2-fi2、-181645-p3x2-fi-extra2（exit 1）：`AssertionError: 剩餘時間 < 0：設限時刻早於快照`——各週的時間在邊界重疊（一週最後的請求結束時間晚於下一週的開始），快照沒限定同一週。改成每個快照只看同一週的請求。fi2 在失敗前已經寫了描述統計與全體版本的列，不要用（`x2_runs.csv`）。
    - block 斷言拆成另一個 run（20261017-181642-p3x2-fi-blocks，4 個 process 逐週平行），主分析的 loader 不再讀 block_ids。
16. FreeInference 的 n_req 1,222,199 和 README 的 1,186,582 不同（差 3.0%），沒有查原因。session 大小用 o200k token × Llama 的 128 KiB。
17. 這次是 self-review，沒有零前文審查。

## 7. 如果要繼續

1. **X1 的對手要加 session 版的 2Q／SLRU**（「已經回來過的 session 放保護段」）。本文顯示 LRU 以外的訊號主要是頻率，而這條規則不用學就拿到（§4.7）；不加的話，oracle 或預測器對 LRU 的 headroom 會有一大塊其實是免費的。D4 的 block 層 ARC／S3-FIFO 可能已經拿到一部分，要在 session 層確認。
2. **要問的不是 C-index，是「降級時挑錯的代價」**：降級只要挑出最晚回來的那幾個。可以算「LRU 集合裡被挑出的最後 k 個中，trace 結束前回來的比例」（S0、S2lex、oracle），直接接到 X1 的 TTFT。
3. **流量／容量比才是自變數**：同樣 256 GiB，Mooncake 只裝 51 s、WildChat 裝 17 h。X1 的 CPU 大小軸換成「裝得下幾分鐘的歷史」來講，比 GiB 或工作集百分比更能解釋 LRU 什麼時候夠好。
4. **toolagent 要另一種 session 來源**：前綴鏈重建在 agent trace 上不成立。agent 用 FreeInference 代替——它上面 LRU 已經接近 oracle（等使用者組挑錯率 4.0% vs 1.0%），而且 256 GiB 只裝 17 個 session，agent 長 context 的問題比較像「容量」而不是「預測」〔判讀〕。
5. 如果 oracle（X1）判 GO，才輪到「輪數＋閒置時間＋上一次間隔」的學習式組合（要參數，這一輪不准）。條件版本（§4.6）顯示輪數在控制閒置時間後還有訊號，是唯一值得學的輸入。
