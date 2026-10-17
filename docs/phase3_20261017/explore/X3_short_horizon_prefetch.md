# X3 回來前幾秒的現成提示：不用長期預測就能預取嗎？

> 第三階段探索方向 X3（[`README.md`](README.md) §2 X3）。〔算術＋模擬，CPU〕。self-review，不是零前文審查。
> 標記：〔實測〕＝本次從資料或常數檔算出來、附 run_id；〔算術〕；〔模擬〕＝兩條線還原模型；〔原文 p.X〕／〔摘要〕；〔假設〕；〔判讀〕。

## 0. 一句話結論＋判定

**判定（照 §2.2／§2.4 字面）：原登記版＝有看頭；含 FreeInference 版＝有看頭**（FreeInference 自己的格子一格都沒過 0.5，靠的是同一批原登記的 trace）。

- **人打字的聊天**（Bailian traceA、WildChat 英文、Mooncake conversation 兩種重建規則）：主設定的 36／36 個帶內格子，回來事件 h 的中位數都是 1.0。打字時間中位 11–22 s，要搬的量中位只有 2–13 個 block，所以幾乎全部藏得住〔模擬，run 20261017-213319-p3x3-hide〕。
- **但這是便宜的有看頭**〔判讀〕：這些 trace 的 context 很短，要藏的慢層還原時間本來就小。以 NFS、INT4 為例，中位只有 30–270 ms。
- **真正難的是長 context 的 agent**。FreeInference 的工具呼叫中位 1.7 s，要還原的 context 中位約 119K token。NFS、INT4 只能藏中位 20%；超過 128K 的事件算 0 的話只剩 4.5%。CacheWise 的工具呼叫中位 82 ms，藏中位 0.5–4.5%。
- 只看閒置 ≥10 s 的工具事件（也就是真的可能被降級的），藏得住 100%。所以 agent 需要的是**事先知道工具會跑多久**，不是知道「回來」這件事。

## 0.1 給主 session 的摘要（每個數字的出處在 §4）

| 項目 | 數字 | 來源 |
|:--|:--|:--|
| 需要的提前量（Llama、GPU 閒、32K＝64 block，整份預取） | INT4：NFS 3.78 s、b48 2.98 s、b32 0.85 s、4 GiB/s 0.31 s；BF16：12.1／9.5／2.7／1.0 s | 〔模擬〕20261017-175005-p3x3-tables |
| 同上，只預取會合點之後的後段 | INT4 NFS 1.59 s（0.53 GiB，整份 1.25 GiB），但最多只藏 49%（BF16 29%） | 同上 |
| 打字速度換成 token/s | 桌機 51.56 WPM → 1.03 token/s（Qwen）／1.01（Llama）；手機 36.17 WPM → 0.73／0.71 | 文獻＋〔實測〕20261017-175409-p3x3-consts |
| 拿得到的提前量 (a) 打字，中位 | Bailian traceA 10.9 s、WildChat 15.4 s、Mooncake conv 15.9 s（strict 21.9 s）、FreeInference 使用者 16.9 s | 〔實測〕hide run |
| (c) 工具，中位 | FreeInference 1.69 s（p90 16.9 s）；CacheWise 0.082 s（p90 14.6 s） | 同上 |
| 判定格（主設定、帶內、(a)） | 36／36 格 h 中位 ≥0.5；全部 =1.0；≥0.5 的事件比例最低 0.70（Mooncake conv）、0.98（Bailian A）、0.99（WildChat） | `x3_verdict.json` |
| (c) FreeInference，NFS／b48／b32／4 GiB/s，INT4 | h 中位 0.20／0.25／0.82／1.0（>128K 截在 128K）；0.045／0.056／0.19／0.50（>128K 算 0） | 同上 |
| 整份 vs 後段（Bailian A、NFS、INT4） | h 中位 1.0 vs 0.47；時間加權 0.99 vs 0.45 | `x3_hide_summary.csv` |

## 1. 問題與要證明什麼

延後設計（BF16 進 CPU、滿了降級、降級時 INT4 進慢層、回來時 Cake 還原）下，被降到慢層的 session 回來時要多付「慢層還原時間」。
oracle 預取要準到秒級的回來時間（審查 §1：學不到）。這裡問的是**不需要長期預測**的訊號：

- (a) **打字**：使用者開始打新一輪訊息時，系統就知道「這個 session 幾秒後會回來」；提前量＝打字時間。
- (b) **思考時間**：上一輪回應結束到下一輪請求的整段閒置；任何預取的上界（不是訊號）。
- (c) **工具呼叫**：agent 發出工具呼叫時就知道它會在工具結束後回來；提前量＝工具執行時間。

要回答：這些提前量，和「把這個 session 從慢層搬回 CPU」需要的時間比，夠不夠？能藏起多少慢層還原時間？
整份預取 vs 只預取 Cake 會合點之後的後段，差多少？

## 2. 判準（開跑前寫，`date -u`＝2026-10-17T09:43Z；寫的時候還沒有任何 X3 的計算結果）

### 2.1 定義

- **回來事件**：同一 session 的第 k≥2 輪請求。它要還原的舊 context＝上一輪的輸入＋輸出（trace 裡用前綴共用驗算，§3）。block＝512 token（main.tex §2.1），n＝ceil(舊 context token 數 ÷ 512)。
- **還原模型**：`code/m7_sim.py` 的 `sim_restore`（Cake 兩條線：重算從前、載入從後）。f(i) 用 MI300X 實測（超出實測長度的 block 用線性外插，標〔外插〕）；CPU 層 35.4 GiB/s；慢層＝限速層。INT4（KIVI）每個 block 的位元組數、H2D、反量化時間用第二階段實測（`results/m10_sweep_mi300x/parts/`）。
  - T_none＝整個 session 都在慢層、回來才還原；T_full＝整個 session 都已在 CPU（預取完）；**慢層還原時間（要藏的）＝T_none − T_full**。
  - 預取：從慢層以該層讀頻寬、從最後一個 block 往前搬進 CPU（和 Cake 載入線同方向）。提前量 L 秒能搬 k(L)＝floor(L × 頻寬 ÷ 每 block 位元組) 個。回來時用 k 個在 CPU、其餘在慢層的配置跑同一個 Cake 模型，得到 T(L)。
  - **可藏起來的比例** h＝(T_none − T(L)) ÷ (T_none − T_full)，介於 0 與 1。
  - **整份預取**：一直搬到 n 個；**只預取後段**：只搬 T_none 的會合點 b 之後的 n−b 個（Cake 本來就要從慢層讀的部分），搬完就停。
- **拿得到的提前量**：
  - (a) L_a＝min(打字時間, 估計的思考時間)。打字時間＝新一輪使用者輸入 token 數 ÷ 打字速度（token/s）；打字速度＝HCI 文獻的每分鐘字數 × 官方或實測的「每 token 幾個字元」換算（§5 附出處）。新一輪 token 數含貼上的內容，所以 L_a 是上界。
  - (b) L_b＝估計的思考時間＝下一輪到達 − 上一輪回應結束；回應結束＝上一輪到達＋估計的 TTFT＋上一輪輸出 token 數 × TPOT（TPOT 是〔假設〕，主值 50 ms，敏感度 20／100 ms）；若資料有回應結束的時間戳就用實測。只當上界報告。
  - (c) L_c＝工具執行時間（CacheWise 的 `execution_time_ms`）。CacheWise 只有 94 個 session，只報告。
- **真實 trace**：有真實時間戳、≥1,000 個「至少回來一次」的獨立 session（README §1）。

### 2.2 判定（照字面）

- **主設定**：MI300X 常數、Llama-3.1-8B、GPU 閒、慢層存 INT4（論文設計）、原始長度、打字速度用桌機文獻值、新 token 數扣掉 chat template 的固定開銷、TPOT 50 ms。
- **有看頭**：主設定下，在至少一個真實 trace、至少一個 Cake 帶內的慢層（0.33–4 GiB/s）上，用 (a) 或 (c)，回來事件的 h **中位數 ≥ 0.5**。（(c) 只有 CacheWise，達不到 1,000 session，所以實際上只有 (a) 能讓判定變有看頭。）
- **死路**：所有真實 trace、所有層（含帶外的層）、BF16 與 INT4、兩個模型、GPU 閒與忙、所有打字速度與 TPOT 變體，用 (a) 與 (c)，h 中位數都 **< 0.2**。
- 其他＝**可能**。
- 只報告、不改判定：(b) 上界；映射到 8K／16K／32K 的結果；整份 vs 只預取後段；h 的完整分布（p10–p90、≥0.5 與 ≥0.99 的比例）；依時間加權的藏起比例（Σ藏起的時間 ÷ Σ慢層還原時間，對應平均 TTFT）；回來請求的完整 TTFT 平均（含新一輪 prefill）在不預取／(a) 預取／完美預取下的值。
- 比主設定更嚴的地方：有看頭只看主設定；死路要所有變體都 <0.2。這兩邊都沒有比 README 鬆。

### 2.3 事先知道的限制（寫在跑之前）

- 單一 session 的算術：預取獨佔慢層頻寬、慢層沒有同時寫入（寫入干擾 3.31× 不算）、其他 session 的需求讀取不搶頻寬。多人同時回來是 X4 的範圍。所以 h 偏樂觀。
- 打字訊號要客戶端回報「開始打字」（聊天 UI 才有；API 流量沒有）。
- 佇列訊號（CachedAttention）的提前量取決於負載（X4），這裡不算。

### 2.4 補充定義（`date -u`＝2026-10-17T09:56Z；已跑完還原模型的驗證與各 trace 的事件抽取，**還沒有算任何 h**）

只把 §2.1–2.2 沒寫清楚的地方寫死，全部是更嚴或等嚴，不放鬆：
1. **沒東西可藏的事件不算**：T_none − T_full < 1 ms 的回來事件（例如只有 1 個 block：Cake 的重算線在時間 0 先拿 block 0，所以不管放哪一層都是 f(0)）不進 h 的分布，另外報它們的比例。若算成 h＝1 會把中位數往上推，所以排除是比較嚴的做法。
2. **判定用整份預取的 h**（同一個提前量下，整份 ≥ 只預取後段，除了 Cake 貪婪分配的少數例外）；只預取後段另外報告。
3. **(a) 只在「人在打字」的 trace 上判定**：Bailian traceA（To-C 聊天）、Mooncake conversation、WildChat 英文。Bailian coder、Mooncake toolagent 的新 token 大多不是打字（程式碼脈絡、工具輸出），只報告；但「死路」要把它們也算進去（所有 trace 都 <0.2 才算死路）。
4. **新 token 數**：Bailian／Mooncake＝子請求輸入 − 父請求輸入 − 父請求輸出 − 10（chat template 每輪開銷，用 Qwen2.5／Llama-3.1 tokenizer 實測都是 10，run 20261017-175409-p3x3-consts），負的當 0（沒有提前量）。WildChat＝使用者訊息本身的字元數（不經 token 換算）。
5. **打字速度**：桌機 51.56 WPM（主）、手機 36.17 WPM（敏感度）；1 word＝5 個字元。token 換算用實測的 WildChat 英文使用者訊息每 token 字元數（Qwen 4.158、Llama 4.269，同一個 consts run）。
6. **估計的思考時間**＝到達間隔 − 上一輪的 prefill（f 模型）− 上一輪輸出 × TPOT；WildChat 沒有 user 時間戳，用「上一輪 assistant 時間戳 → 這一輪 assistant 時間戳」扣掉這一輪的 prefill 與生成。負的當 0。
7. 要還原的 n 超過 256 個 block（128K）就截在 256，報截掉的比例。

## 3. 做了什麼

全部只用 CPU（`CUDA_VISIBLE_DEVICES=""`），最多 8 個 process。記錄殼照 CLAUDE.md §4.1，每個 run 目錄在 `/ssd8/hungwei/paper-hkv/runs/<RUN_ID>/`，裡面有 cmd.sh、context.txt、stdout.log、stderr.log、exit_code。全部 run 的 exit_code 都是 0。

| run_id | 指令（摘要） | 產出 |
|:--|:--|:--|
| 20261017-174414-p3x3-lit | curl 下載兩篇打字速度論文的 PDF＋pdftotext（sha256 在 stdout.log） | run 目錄 `data/` |
| 20261017-174929-p3x3-validate | `code/p3x/x3_restore.py validate` | `x3_restore_validate*.csv` |
| 20261017-175005-p3x3-tables | `x3_restore.py tables --procs 8`：2 模型 × 12 層 × 2 精度 × 2 GPU＝96 組，每組算 T(n,k)，n≤256 | run 目錄 `tables/*.npz`、`x3_needed_lead.csv` |
| 20261017-175409-p3x3-consts | `code/p3x/x3_leads.py consts`：chat template 開銷、WildChat 每 token 字元數 | `x3_typing_consts.json`、`x3_checks_consts.csv` |
| 20261017-175445-p3x3-bailian | `x3_leads.py bailian` | events_bailian_*.csv.gz、`x3_checks_bailian.csv` |
| 20261017-180334-p3x3-mooncake-loose／20261017-180336-p3x3-mooncake-strict | `x3_leads.py mooncake --rule loose｜strict` | events_mooncake_*、`x3_checks_mooncake_{loose,strict}.csv` |
| 20261017-175521-p3x3-wildchat | `x3_leads.py wildchat --files 2`（前 2 個 parquet） | events_wildchat_en、`x3_checks_wildchat.csv` |
| 20261017-175502-p3x3-cachewise | `x3_leads.py cachewise` | events_cachewise_{tool,human}、`x3_checks_cachewise.csv` |
| 20261017-180405-p3x3-fi | `x3_leads.py freeinference`（16 週 processed parquet） | events_freeinference_{tool,user}、`x3_checks_freeinference.csv` |
| **20261017-213319-p3x3-hide** | `code/p3x/x3_hide.py --tables … --events …`（12 個事件檔） | `x3_hide_summary.csv`（Llama）、`x3_hide_summary_qwen.csv`、`x3_hide_mapped.csv`、`x3_hide_sens.csv`（桌機）、`x3_hide_sens_mobile.csv`、`x3_lead_dist.csv`、`x3_verdict.json` |
| 20261017-213321-p3x3-extra | `code/p3x/x3_extra.py`：依閒置時間條件化、依長度分箱、WildChat 時間戳檢查 | `x3_cond_idle.csv`、`x3_bylen.csv`、`x3_wildchat_tscheck.csv` |

作廢的 run：20261017-175800-p3x3-hide（第一次，還沒有 FreeInference 與嚴格規則，被後面的 run 覆蓋）；20261017-181448-p3x3-hide（判定的篩選把 `DataFrame.clip` 方法當成欄位，main_cells＝0；CSV 沒問題；run 目錄留了 `VERDICT_BUG`）。

**trace 匯入的驗算（規則 6）**〔實測，`x3_checks_*.csv`〕：
- Bailian：`len(hash_ids)==ceil(input_length/16)` 四個檔都是 100%；子請求 turn＝父 turn＋1，100%；間隔 ÷ 父輸出 token 的中位 0.29 s（traceA），所以 timestamp 是秒。traceB 沒有任何 parent 連結，不能當 session trace。
- Mooncake：`len(hash_ids)==ceil(input_length/512)`，100%；間隔 ÷ 父輸出 token 的中位 0.34 s，所以是毫秒；70% 的 timestamp 是 1000 的倍數，解析度約 1 秒。
- CacheWise：`execution_time_ms` 和 `result_timestamp − timestamp` 差超過 1 ms 的有 0 筆（11,614 筆工具呼叫）。
- FreeInference：前兩週 `len(block_ids)==floor(Σrole_tokens/16)`，100%；transition＝append 時，block 共同前綴 ×16 ÷（上一個輸入＋輸出）的中位 0.998、p1 0.958，所以「要還原的長度＝上一個輸入＋輸出」的定義對得上。
- WildChat：user 訊息沒有時間戳，只有 assistant 有。assistant 時間戳是不是回應結束，**驗不出來**：間隔和這一輪、上一輪輸出長度的相關係數差不多（0.354 vs 0.370）；長度 ≥300 token 的回應裡，沒有一筆的間隔小於 20 ms × 輸出 token 數。它只影響 (a) 的上限截斷；截斷前後的中位是 16.3 s 和 15.4 s。

## 4. 結果

### 4.1 還原模型的誤差〔模擬 vs 第二階段實測，20261017-174929-p3x3-validate〕

拿第二階段 MI300X 實測的 E0（全 BF16）、E1（全 INT4）Cake 還原中位數來比：

| 範圍 | 格數 | 中位相對誤差 | 最大 | 會合點在 ±1 個 block 內 |
|:--|:--|:--|:--|:--|
| 慢層（b16／b32／b48 × 8K／16K／32K × 兩模型；32K 另有 GPU 忙） | 48 | 3.5% | 38.8% | 91.7% |
| CPU 層 | 16 | 19.7% | 66.5% | 100% |

- 慢層 BF16 的誤差在 −6.4% 到 +2.1% 之間。
- INT4 系統性**低估** 3–39%，短 context、快的層最嚴重（例如 Llama 8K b16 是 59 vs 93 ms）。這看起來是每次還原有一個約 20–40 ms 的固定開銷沒有模型化。這個開銷同時出現在 T_none 和 T_full 裡，對 T_none − T_full 的影響比較小，但沒有另外驗證。
- 敏感度變體 plus_h2d（慢層 block 再加一次 H2D）的慢層誤差中位 4.8%，比主模型差，所以不用。

### 4.2 需要的提前量〔模擬，`x3_needed_lead.csv`，20261017-175005-p3x3-tables〕

Llama、GPU 閒。「要藏的」＝T_none − T_full。「整份」＝把 n 個 block 全部搬回 CPU 要多久；「後段」＝只搬 T_none 會合點 b 之後的 n−b 個；「後段最多藏」＝只預取後段的 h。

| 層（GiB/s） | 精度 | n（token） | 要藏的 | 整份 | 後段 | 後段 GiB／整份 GiB | 後段最多藏 |
|:--|:--|:--|:--|:--|:--|:--|:--|
| NFS 0.331 | INT4 | 16（8K） | 0.33 s | 0.94 s | 0.35 s | 0.12／0.31 | 34% |
| NFS 0.331 | INT4 | 64（32K） | 1.57 s | 3.78 s | 1.59 s | 0.53／1.25 | 49% |
| NFS 0.331 | INT4 | 256（128K，f 外插） | 8.33 s | 15.1 s | 8.55 s | 2.83／5.0 | 65% |
| NFS 0.331 | BF16 | 64 | 2.52 s | 12.1 s | 2.64 s | 0.88／4.0 | 29% |
| b48 0.420 | INT4 | 64 | 1.37 s | 2.98 s | 1.44 s | 0.61／1.25 | 55% |
| b48 0.420 | BF16 | 64 | 2.26 s | 9.53 s | 2.38 s | 1.0／4.0 | 32% |
| b32 1.467 | INT4 | 64 | 0.57 s | 0.85 s | 0.61 s | 0.90／1.25 | 77% |
| b32 1.467 | BF16 | 64 | 1.24 s | 2.73 s | 1.36 s | 2.0／4.0 | 57% |
| 4 GiB/s | INT4 | 64 | 0.21 s | 0.31 s | 0.27 s | 1.07／1.25 | 99.5% |
| 4 GiB/s | BF16 | 64 | 0.58 s | 1.0 s | 0.70 s | 2.81／4.0 | 77% |

- **後段為什麼藏不完**：後段搬進 CPU 之後，前段還在慢層。Cake 的載入線很快吃完 CPU 裡的後段，接著又從慢層讀前段，和重算線一起把前段做完；所以還原時間降不到「全部在 CPU」的 T_full。
- n＝1 時，不管放哪一層都沒東西可藏：重算線在時間 0 先拿 block 0，T＝f(0)＝29 ms。
- Cake 是貪婪分配，所以 T(n,k) 對 k 不是嚴格單調，少數格的 h 是小的負數（例如 Llama b48、BF16、n＝16，後段的 h＝−0.013）。

### 4.3 拿得到的提前量〔實測＋文獻常數，`x3_lead_dist.csv`，20261017-213319-p3x3-hide〕

| trace | session（有回來的） | 回來事件 | 要還原 n 的中位（p90） | (a) 打字 p25／p50／p75 | (b) 思考 p50 | (c) 工具 p50（p90） |
|:--|:--|:--|:--|:--|:--|:--|
| Bailian traceA（To-C 聊天） | 9,012 | 19,957 | 3（13） | 5.8／10.9／27.1 s | 86 s | — |
| WildChat 英文（前 2 個 parquet） | 26,166 | 91,029 | 2（6） | 8.1／15.4／32.8 s | 94 s | — |
| Mooncake conversation（loose） | 2,012 | 3,975 | 13（52） | 0／15.9／78.5 s | 103 s | — |
| Mooncake conversation（strict） | 1,959 | 2,436 | 12（51） | 0／21.9／101 s | 108 s | — |
| Bailian coder（只報告 (a)） | 6,693 | 16,605 | 7（22） | 4.5／15.5／61 s | 74 s | — |
| Mooncake toolagent（loose，只報告 (a)） | 1,854 | 16,400 | 12（15） | 0／0／5.1 s | 1.8 s | — |
| Bailian thinking（509 session，只報告） | 509 | 1,200 | 10（34） | 0／0／0 | 0 | — |
| FreeInference 使用者（登記後加） | 11,116 | 73,458 | 243（截在 256；48% >128K） | 6.0／16.9／46.7 s | 96 s（實測） | — |
| FreeInference 工具（登記後加） | 13,953 | 1,134,688 | 232（45% >128K） | — | — | 1.69 s（16.9 s） |
| CacheWise（94 session，只報告） | 80 | 10,576 | 157（16% >128K） | — | 20 s（人） | 0.082 s（14.6 s） |

- **新 token 數的算法**：Bailian 和 Mooncake 用「子請求輸入 − 父請求輸入 − 父請求輸出 − 10」，10 是 Qwen2.5 和 Llama-3.1 的 chat template 每輪開銷（兩個 tokenizer 都實測到 10，含 assistant 結尾的 EOS）。Bailian traceA 增量的 p1 是 12，也就是 template 加 1–2 個字，對得上。
  - 增量是負的就當 0：Mooncake conversation 有 10.7%，toolagent 21.7%，thinking 26.6%（推理 token 被拿掉，Bailian 文件 Q1）。
  - Mooncake conversation 有 28.7% 的打字提前量是 0（增量 ≤10）。Kimi 的 template 開銷未知，不扣的版本見 §4.6。
- WildChat 用使用者訊息本身的字元數，除以 51.56×5/60＝4.30 字元/s。
- **(a) 一定截在思考時間以內**：Bailian thinking 的思考時間估計大多是 0，因為 TPOT 50 ms × 很長的推理輸出 > 實際間隔，所以 TPOT 假設在那裡不成立。

### 4.4 判定格：可藏起來的比例〔模擬，`x3_hide_summary.csv`、`x3_verdict.json`〕

**主設定**：Llama、GPU 閒、慢層 INT4、原始長度、整份預取、桌機打字速度、TPOT 50 ms。h 的中位數：

| trace | NFS 0.331 | b48 0.420 | b32 1.467 | 4 GiB/s | ≥0.5 的事件比例（NFS） | ≥0.99（NFS） | 時間加權（NFS） | 要藏的中位（NFS） |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| Bailian traceA (a) | 1.0 | 1.0 | 1.0 | 1.0 | 0.984 | 0.984 | 0.988 | 60 ms |
| WildChat 英文 (a) | 1.0 | 1.0 | 1.0 | 1.0 | 0.992 | 0.992 | 0.993 | 30 ms |
| Mooncake conv loose (a) | 1.0 | 1.0 | 1.0 | 1.0 | 0.704 | 0.687 | 0.692 | 268 ms |
| Mooncake conv strict (a) | 1.0 | 1.0 | 1.0 | 1.0 | 0.717 | 0.704 | 0.687 | 226 ms |
| FreeInference 使用者 (a)，>128K 截斷 | 1.0 | 1.0 | 1.0 | 1.0 | 0.791 | 0.647 | 0.765 | 7.85 s |
| FreeInference 使用者 (a)，>128K 算 0 | 0.043 | 0.053 | 0.185 | 0.430 | 0.445 | 0.388 | 0.274 | 7.85 s |
| FreeInference 工具 (c)，截斷 | 0.205 | 0.255 | 0.816 | 1.0 | 0.284 | 0.165 | 0.333 | 7.42 s |
| FreeInference 工具 (c)，>128K 算 0 | 0.045 | 0.056 | 0.189 | 0.499 | 0.171 | 0.102 | 0.123 | 7.42 s |
| CacheWise 工具 (c)（只報告） | 0.005 | — | 0.045 | — | 0.175 | 0.125 | 0.197 | 4.63 s |

- 主設定下，帶內 9 個層 × 4 個人打字的 trace（Mooncake 兩種規則各算一個）＝36 格，36 格 h 中位都是 1.0 → **原登記版：有看頭**。
- **含 FreeInference 版**：FreeInference 用「>128K 算 0」的下界（較嚴），18 格裡 0 格 ≥0.5，最大 0.499（工具、4 GiB/s）；加上原登記的 36 格 → **有看頭**（36／54）。
- 死路的條件（全部 <0.2）不成立：(a)、(c) 在所有變體中的最大 h 中位是 1.0。
- **回來請求完整 TTFT 的平均**（還原＋新一輪 prefill；不預取 → 用提示預取 → 完美預取；NFS、INT4、截斷版）：
  - Bailian traceA 132 → 38 → 37 ms；WildChat 78 → 34 → 34 ms。
  - Mooncake conversation 699 → 329 → 164 ms。
  - FreeInference 使用者 6,422 → 1,840 → 431 ms；FreeInference 工具 6,188 → 4,199 → 216 ms。
  - CacheWise 工具 5,044 → 4,088 → 181 ms。

### 4.5 整份 vs 只預取後段〔模擬，`x3_hide_summary.csv` 的 scope 欄〕

| trace（Llama、GPU 閒） | 層、精度 | h 中位：整份／後段 | 時間加權：整份／後段 |
|:--|:--|:--|:--|
| Bailian traceA (a) | NFS INT4 | 1.0／0.47 | 0.99／0.45 |
| Bailian traceA (a) | b32 INT4 | 1.0／0.94 | 0.99／0.79 |
| WildChat (a) | NFS INT4 | 1.0／0.47 | 0.99／0.51 |
| Mooncake conv (a) | NFS INT4 | 1.0／0.40 | 0.69／0.35 |
| FreeInference 使用者 (a) | NFS INT4 | 1.0／0.58 | 0.77／0.52 |
| FreeInference 工具 (c) | NFS INT4 | 0.205／0.204 | 0.33／0.27 |

- 打字提前量夠長時，後段只少搬一些位元組，卻少藏一半。
- 提前量很短時（工具），整份和後段差不多，因為兩者在後段還沒搬完時就都用完了提前量。
- 〔判讀〕「只預取後段」只在 CPU 空間不夠放整份、或慢層頻寬要讓給別人時才划算；就藏時間而言，它永遠不比整份好。

### 4.6 敏感度（只報告；Llama、GPU 閒，`x3_hide_sens*.csv`、`x3_hide_summary_qwen.csv`、`x3_hide_mapped.csv`）

- **模型、GPU 狀態、精度**：四個人打字的 trace，在 Llama／Qwen × 閒／忙 × BF16／INT4 × 帶內各層，h 中位全部是 1.0。≥0.5 的事件比例最低：Bailian A 0.98、WildChat 0.99、Mooncake 0.67。
- **打字速度**：手機 36.17 WPM 比桌機慢，提前量更長；任何格的 h 中位都沒有變小。
- **TPOT**：100 ms 時思考時間變短。Bailian A 的 ≥0.5 比例最低降到 0.85（BF16）／0.87（INT4），h 中位仍是 1.0。Bailian thinking 在 TPOT 20 ms 時 h 中位變成 1.0，50／100 ms 時是 0。
- **新 token 數**：
  - 不扣 template 開銷：Mooncake conversation 的 ≥0.5 比例從 0.70 升到 0.88。
  - 非 text 類型（search／file／image）一律算 0：Bailian A 在快的帶內層（b32、2、2.77、4 GiB/s）h 中位變 0，NFS、b48、0.5、0.76、1 GiB/s 仍是 1.0。原因是 Bailian A 裡長的 context 大多是 search／file 回合：n 在 9–16 個 block 的事件只有 25% 是 text。WildChat 全是 text，不受影響。
- **映射長度**（全部 session 改成 8K／16K／32K，提前量不變）：32K 下，NFS INT4 的 h 中位，Bailian A、WildChat、Mooncake conv 都是 1.0，≥0.99 的比例 0.90／0.92／0.64。NFS BF16（要 12 s）時，Bailian A 的 h 中位 0.93、≥0.99 只剩 0.50。
- **長度分箱**（`x3_bylen.csv`，NFS INT4）：FreeInference 工具在 129–256 個 block（占 74% 事件）時 h 中位 0.17，17–32 個 block 時 0.64。Bailian A 每一箱都是 1.0（最長那箱只有 8 個事件）。

### 4.7 選擇效應：只看「閒置夠久、真的可能被降級」的事件（只報告，`x3_cond_idle.csv`）

LRU 降級的是閒置最久的 session，所以要條件在閒置時間上看（INT4，NFS／b32）：

| trace | 條件 | 占事件 | h 中位（NFS／b32） | ≥0.5 比例（NFS） |
|:--|:--|:--|:--|:--|
| FreeInference 工具 (c) | 工具 ≥1 s | 64.8% | 0.38／1.0 | 0.42 |
| FreeInference 工具 (c) | 工具 ≥10 s | 15.1% | 1.0／1.0 | 1.0 |
| CacheWise 工具 (c) | 工具 ≥1 s | 26.8% | 0.87／1.0 | 0.65 |
| CacheWise 工具 (c) | 工具 ≥10 s | 12.0% | 1.0／1.0 | 1.0 |
| FreeInference 使用者 (a) | 思考 ≥60 s | 61.8% | 1.0／1.0 | 0.90 |
| Bailian traceA (a) | 思考 ≥60 s | 59.4% | 1.0／1.0 | 1.0 |

- 〔判讀〕工具呼叫的提前量就是工具本身的時長，所以久到會被降級的工具呼叫，自己就帶著夠長的提前量。
- 但要用到這段提前量，系統得在**工具結束之前**就開始搬，也就是要知道工具大概會跑多久（Continuum／CacheWise 的工具時長預測）；不然就只能乾脆不降級有工具在跑的 session。所以 (c) 的 h 是「知道工具時長」的上界，不是「只知道有工具在跑」就拿得到的。
- 打字不同：打字開始本身就是「幾秒後會回來」的訊號，系統只要收到就能開始搬，不用預測。

## 5. 前作與文獻常數

| 來源 | 用在哪 | 標記 |
|:--|:--|:--|
| Dhakal, Feit, Kristensson, Oulasvirta, *Observations on Typing from 136 Million Keystrokes*, CHI 2018. https://www.pokristensson.com/pubs/DhakalEtAlCHI2018.pdf（DOI 10.1145/3173574.3174220；PDF sha256 d21bd145…，run 20261017-174414-p3x3-lit） | 桌機 51.56 WPM（SD 20.2）〔原文 p.5〕；1 word＝5 個字元、從第一個到最後一個按鍵〔原文 p.4〕。是**照抄句子**的速度，自己想著打通常更慢，所以用它算的打字時間偏短（偏嚴） | 〔原文 p.4、p.5〕 |
| Palin, Feit, Kim, Kristensson, Oulasvirta, *How do People Type on Mobile Devices?*, MobileHCI 2019. https://www.pokristensson.com/pubs/PalinEtAlMobileHCI2019.pdf（DOI 10.1145/3338286.3340120；sha256 9e388641…） | 手機 36.17 WPM（SD 13.22），37,370 人〔原文 p.6；摘要 p.1 寫 36.2〕 | 〔原文 p.1、p.6〕 |
| OpenAI Help, *What are tokens and how to count them?* https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them | 「1 token ≈ 4 個英文字元 ≈ ¾ 個字」只當交叉檢查（頁面回 403，內容來自搜尋摘錄）；實際換算用本地 tokenizer 在 WildChat 英文訊息上的實測值 4.16（Qwen）／4.27（Llama） | 〔摘要〕＋〔實測〕 |
| Qwen2.5-7B-Instruct-1M／Llama-3.1-8B-Instruct 的 chat template（本地 tokenizer） | 每輪開銷 10 token | 〔實測〕 |
| Mooncake FAST'25 trace README（datasets/mooncake/FAST25-README.md） | timestamp 毫秒、512-token block | 〔文件〕 |
| Qwen-Bailian README（datasets/qwen-bailian-usagetraces-anon/README.md、docs/qa-context-growth-pattern.md） | parent_chat_id、16-token block；推理 token 被拿掉 | 〔文件〕 |
| FreeInference dataset card（datasets/freeinference_agentic_trace/README.md，rev 16d3c04d） | step_trigger、start_ms／end_ms、latency_ms＝「到下一個 LLM 請求的間隔」、o200k 16-token block | 〔文件〕 |
| CacheWise（arXiv 2606.16824）、Continuum（arXiv 2511.02230） | 工具時長預測／TTL 釘住；我**沒讀原文**，機制描述引自 `../research/LIT_AND_TRACES.md` | 〔未查證〕 |
| CachedAttention（佇列看得到才預取） | 沒讀；本方向沒算佇列訊號（§2.3） | 〔未查證〕 |
| 中文打字速度 | 搜尋只找到小型研究（手機拼音鍵盤，約 18–25 字/分），而且只看了搜尋摘要；沒有大規模資料，所以**沒有用**。Bailian 的打字時間也用英文的 token/s 算（NOT_MEASURED：中文實際打字速度） | 〔摘要〕 |

## 6. 偏離與問題（照實列）

1. **FreeInference 是登記之後才加的**。主 session 在 README §4（2026-10-17T10:01Z）加入，我在約 10:03Z 收到通知，開始寫 loader。
   - 收到時我**已經看過** X3 原登記版的全部主要結果：第一次 hide run（20261017-175800）的判定「有看頭」、h 的分布、CacheWise (c) 的中位 ≤0.05、映射長度、敏感度，以及 x3_extra（條件化、分箱）。
   - FreeInference 的處理規則（>128K 算 0 當判定用的下界、使用者增量用 role_tokens.user、n_old＝上一個輸入＋輸出）是看到 FreeInference 結果之前寫進 x3_hide.py 的，但那時我已經知道其他 trace 的結果。
2. **Mooncake 的嚴格重建規則也是事後加的**（同一則通知）。兩種規則下 h 中位都是 1.0；≥0.5 的比例 loose 0.704、strict 0.717，較保守的是 loose。
3. **Llama 的 128K 上限**：FreeInference 有 45–48% 的事件、CacheWise 有 16–20% 的事件超過 256 個 block。主表把它們截在 256（偏樂觀）；「算 0」版是下界。
   - f(i) 在 64 個 block 以後是線性外插〔外插〕：用 block 16–63 的直線，斜率 Llama 0.886 ms/block、Qwen 0.767。
4. **還原模型的誤差**（§4.1）：INT4 系統性低估到 39%；CPU 層誤差中位 20%。模擬只能篩，不能進論文。
5. **單一 session 的算術**：預取獨佔慢層讀頻寬、沒有寫入干擾、沒有別的 session 搶頻寬。所以 h 偏樂觀，多人同時回來的情況是 X4 的範圍。
6. **(a) 的提前量是上界**：新 token 數含貼上的內容、搜尋結果與檔案。用照抄的打字速度則偏短。實際上要客戶端回報「開始打字」，API 流量沒有這個訊號。
7. **思考時間要估回應結束**（Bailian、Mooncake）：用 TPOT 50 ms〔假設〕。Bailian thinking 在這個假設下多半變成 0。FreeInference 和 CacheWise 有實測的回應結束時間。CacheWise 有 164 筆時間戳亂序的人類事件（思考時間是負的），已排除。
8. **WildChat** 只用前 2 個 parquet（共 14 個），也只用英文；user 沒有時間戳，assistant 時間戳的語意驗不出來（§3）。
9. **Bailian 是中文服務**，打字速度卻用英文的 token/s，因為沒找到可靠的中文來源；方向不確定。
10. **n 的取法**：用 ceil(token÷512)，不滿一個 block 也算一個 block，短 context 會多算位元組（偏嚴）。Bailian 的 n_old 用共同前綴（16-token block）：traceA 有 10% 的子請求和父請求的共同前綴不到父 context 的 25%。
11. **流程問題**：第二次 hide run（20261017-181448）的判定有 bug（`DataFrame.clip`），已修正後重跑；session 在約 18:2x（本地時間）因 API 用量上限中斷過一次，但所有 run 的 exit_code 都是 0、沒有半途中斷的 run，所以沒有 INTERRUPTED 檔。
12. **CSV 大小**：為了讓每個檔 <1 MB，`x3_hide_summary.csv` 依模型拆成兩個、`x3_hide_sens.csv` 依打字速度拆成兩個，原始完整檔在 run 目錄 `full/`。
13. 沒有用到 3090 常數（X7 還沒給）；全部是〔MI300X 常數〕。佇列訊號沒算。

## 7. 如果要繼續

1. **把 agent 的工具時長和還原時間放在一起看**（最值得追）。FreeInference 有 1.13M 個工具事件、13,953 個 session。可以問：用固定規則（例如每個工具名的歷史中位數，在前幾週定好、後幾週測）預估工具時長，再「降級，並在預估結束前 L 秒預取」，能拿到 §4.7 上界的多少？這是 Continuum／CacheWise 的設定加上慢層與 Cake，目前查到的前作沒有和還原時間一起算（待確認）。注意 CLAUDE.md §8：oracle 出來之前，這只能用事先固定的規則，不能訓練。
2. **併發**（給 X4）：很多 session 同時打字時，預取搶慢層頻寬，h 會掉多少。
3. **長 context 的人打字**：FreeInference 的使用者回合 h 中位，截斷版是 1.0，算 0 版是 0.04。差別全部來自 >128K 的事件，要用 Qwen-1M 的 f(i) 實測（不外插）才能定。
4. 3090 常數出來之後重跑 `x3_restore.py tables` 與 `x3_hide.py`（只要換常數）。
