# F4 資料完整性稽核（DATA_AUDIT）

**判準寫於**：2026-10-10T07:42:33Z（開始查資料之前；只看過 SUMMARY §4、D3、D6 §4.5 與 main.tex 原文）
**性質**：self-review。執行和稽核都是同一個 Claude，**不是** cross-model 或 zero-context 審查，因為這台機器沒有 Codex MCP。照 `paper-claim-audit` skill 的步驟手動做。main.tex 數字核對分三塊做：主 agent、fork A（平台 A）、fork B（平台 B）。fork 是同一個模型、帶著同樣的前文，所以也不是獨立審查。
**不改** main.tex、不改任何既有檔、不 commit、不用 GPU（runsh 的 context.txt 會跑一次唯讀的 `rocm-smi`）。

## 一頁摘要

**問題是什麼**：main.tex 的數字，到底有沒有來源？來源乾不乾淨？投稿前哪些一定要改？
**要證明什麼**：逐一核對 main.tex 的數字，範圍包括摘要、引言、背景與評估各節、§平台 B、結論、附錄 A，以及全部表格。每一個都找到 run_id 和檔案，或寫明找不到。

**main.tex 數字核對：241 個（逐列見 `docs/audit_20261010/claims_table.csv`）**

| 一致 | 不一致 | 找不到來源 | 來源有疑慮 |
|:--|:--|:--|:--|
| **95** | **59** | **17** | **70** |

- 只看「實測」類的 141 個：一致 59、不一致 31、找不到來源 6、來源有疑慮 45。
- 「一致」的 95 個裡，有 28 個只靠 run 的 stdout，或靠本次從 stdout 重建的檔，論文引用的 CSV 已經不在。照判準字面改判「來源有疑慮」的話，是一致 67、來源有疑慮 98。
- 摘要的 14 個數字：一致 3、不一致 5、來源有疑慮 6。

**投稿前一定要修的 5 件事**
1. **κ 的頭條數字**（摘要 L100–106、引言 L132、表 kappa、結論 L1608）
   - 3090 的 8.9／9.5 是在整機 HEAVY 下量的。整機爭用只拖慢 CPU 取回，所以 κ 被壓低；同一個量在 QUIET 時約 **11.9**（§1）。
   - 「全距 0.7–11.7」把 κ_ssd（0.72）和 κ_cpu 混在一起。
   - 「6–21 倍／54–190 倍」、L335「6–190 倍」、表 models 的 κ＝108／54 是舊的 100% MFU 算術估計，和同一篇的實測表矛盾。
   - 表 kappa 標題說「末兩列 7.6 倍」，實際是第 3 列對第 5 列。
2. **headroom 和寫入頻寬是 oracle 修正前的舊值**（摘要 L108、表 verdict、表 writebw、§sweet-spot）
   - 舊 RUNLOG 自己記錄：修正後 qwen-awq 的 8.42／9.23% 變成 12.3／12.0%，寫入比值從 1/4–1/7 變成 1/48.8–1/59.5（qwen-awq），而且「舊版 M4 CSV 全部要重跑」。論文還在用舊值。
   - 成本常數全在 HEAVY 下量。
   - 峰值位置的兩種說法互相矛盾：L1306 說 3.5 P*，hw_sweep 說 262K；四條線裡有兩條不在 2–3.5 P*。
3. **平台 B 的品質主張不成立**（L1527–1546）
   - 「64K–129K 下 LongBench／RULER 差 ±1 分內」：實際量在 32K 視窗、16K 長度，四個模型有 6 格超過 ±1。
   - 「GSM8K 七個模型 ±5pp 內」：Qwen2.5-7B-1M 掉 90.8pp、Nemo INT4 掉 7.5pp。
   - 「推理看不出、檢索掉 100pp」的對比在平台 B 不存在：同一個模型兩類一起崩。
   - 摘要的「於 sm_86 只有 INT8 保住品質（95／5／0%）」其實是 Qwen 模型族的性質，MI300X 上的 Qwen 也一樣。
4. **表 eps-task 混了兩個模型**（L1229–1242）
   - GSM8K 那欄是 Llama-BF16，其他欄是 qwen-awq，表題卻全寫 qwen-awq。
   - 所以「同設定換任務 ε 差 21–36 倍」是跨模型比。
   - 另外：「FWE 是 14 個任務中唯一」不成立；數值誤差那段（2.64%、0.65%）其實是高斯合成資料，論文沒揭露。
5. **來源檔不在、或找不到產生它的 run**
   - MI300X 的 `cost_constants`、`m5_quality/*`、`recompute_position_*`、`oracle_*`、`policy_sim`、`attn_importance_*` 從沒進 git，52803bc 時一起被刪了。前四類本次已從 stdout 重建（§2）。
   - 負面結果表的 Llama toolagent 列（3.76／−33.2／−35.0）找不到產生它的 run。
   - 「decode 佔 78–82%」「48.8／53.8%」只有文字，沒有數據檔；decode 擬合 18.158 重算不出來。
   - 「自檢 0 個 FAIL」，實際唯一的 verify run 是 5 FAIL。

**其他四個問題的結論**
- **3090 κ_cpu＝8.9：有疑慮**。
  - 自己那張卡**很可能乾淨**：每次重開 server 的 KV pool 都一樣，重算時間穩定。但沒有直接紀錄：guard JSON 被覆寫了，CSV 也沒有 own_gpu_intruders 欄。
  - HEAVY 指的是其他 6 張卡在忙。它讓 CPU 取回慢 49%，κ 因此偏低。
  - git 裡有一組 QUIET 重複（20260831-211745 第 3 輪），κ≈11.9，比 8.9 高 34%，但樣本只有 4 筆。
- **MI300X CSV：已重建**。
  - m2 取回、重算是逐請求、逐 rep 從 stdout 抄回來的。cost_constants 和擬合重算後，和當時的 analyze 輸出 **26/26、28/28 格相同**。
  - 品質只能重建到摘要層級，因為逐題輸出在任何地方都不存在。
- **κ 定義**：1.5 和 16 都是真的，量的是不同軟體路徑。差距主要來自傳輸路徑：vLLM 的 CPU 卸載比 memcpy 慢 8.9 倍。建議統一成「同一引擎、同一後端，重算÷取回，並附平台／模型／L／路徑」（§4）。
- **FP8 書面檢查**：兩次 run 都確實用了 `--kv-cache-dtype fp8`，KV 容量剛好兩倍，log 也寫「用 fp8 存 KV」。FP8 和 BF16 用同一個 attention 後端（ROCM_ATTN），其他低精度都換成 TRITON_ATTN。所以「逐字相同數」不能跨精度比。FP8 的值有沒有真的進到 attention，要等 F3 的 GPU 驗證（§5）。

**新發現（任務書沒問到）**：MI300X 上，卸載組（cpu_lru、tier_fs）和 INT4／INT8 都換成了 TRITON_ATTN，基準是 ROCM_ATTN。所以下面三件事都混進了換 kernel 的效果：
- 「卸載和不卸載 13–25/60 輸出不同，原因是分塊邊界」；
- 「INT4 反量化 1.59 µs/token」；
- MI300X 的 κ_cpu。


---

## 事先寫好的判準（07:42:33Z，沒改過）

**每個 main.tex 數字的狀態（四選一）**
- **一致**：找到帶 run_id 的來源，數值在顯示精度內相符，而且來源本身沒有已知問題。來源可以是 git 歷史的 CSV、run 目錄的 stdout／CSV，或可重算的算術。
- **不一致**：找到來源，但數值、範圍或口徑對不上，差距超過四捨五入。範圍包括模型、平台、ctx、n。
- **找不到來源**：git 全部分支和這台機器的 run 目錄都找不到能產生這個數字的檔案或 log。
- **來源有疑慮**：數值對得上，但來源本身有問題。例如：
  - 時間欄被標污染或 HEAVY；
  - 論文標的來源檔不存在，只剩 stdout 摘要；
  - 平台或模型和文字不符；
  - 推算值被寫成實測。

**問題 1（3090 κ_cpu＝8.9）的判定**
- **可用**：自己那張卡沒有外來 process，而且有 QUIET 的重複量測、數值在 ±10% 內。
- **有疑慮**：符合下面任一條：
  - 自己那張卡乾淨，但整機 HEAVY，又沒有 QUIET 重複；
  - 偏差的方向會讓結論變強。
- **不可用**：自己那張卡被污染（contaminated＝True 或 own_gpu_intruders>0），或來源找不到。

**問題 2（MI300X 品質 CSV）**：run 目錄裡找得到逐題原始輸出，才重建逐題 CSV。只剩 stdout 摘要時，重建成摘要層級的 CSV，並在欄位標明 `granularity`，不假裝是原始資料。

**執行中補充的口徑**（看到資料之後才定的，照實列出）
- 「論文標的來源檔不存在，但 run 的 stdout 有完整數字」：數字本身判**一致**，引用那個檔的那一句另判**找不到來源**。fork B 先這樣做，主 agent 跟著統一。
- 外部規格（廠商頻寬、HF config）：只驗算術。規格值本身標〔未查證〕。

---

## 1. 3090 的 κ_cpu＝8.9 能不能用

**問題是什麼**：摘要第 100 行寫 RTX 3090 的 κ_cpu＝8.9。它的來源每一列都標 `host_contention=HEAVY`。
**要證明什麼**：
- 自己那張卡有沒有被污染；
- 整機爭用讓 κ 偏向哪一邊；
- 有沒有乾淨的重複量測。

### 1.1 來源（逐列找到）

| main.tex | 來源檔 | run_id | 算出來 |
|:--|:--|:--|:--|
| 表 kappa 列 1（SATA）325.5／36.8／346.0／**8.9** | `git 52803bc^:results/m2_harness/retrieval_cost_sata.csv`（26 列） | `20260830-222852-m2-retrieval` | 325.50／36.77／346.00 µs/token，κ_cpu＝**8.852**〔算術〕 |
| 表 kappa 列 2（NVMe）323.6／33.9／392.9／**9.5** | `git 52803bc^:results/m2_harness/retrieval_cost_nvme.csv`（26 列） | `20260830-223536-m2-retrieval` | 323.63／33.94／392.89，κ_cpu＝**9.535**〔算術〕 |

- 算法和 main.tex 表的說明一樣：(warm TTFT 中位數 − gpu_resident warm) ÷ 16,384〔檔案 code/m9_f4_kappa.py〕。兩列的數字都重現了〔run 20261010-075352-f4-kappa〕。
- 這兩個 CSV 在 commit `08702db`（2026-08-30 22:44）加進 git，在 `52803bc` 從工作目錄刪掉。main.tex 第 312 行引用的路徑現在不存在，要用 `git show 52803bc^:…` 取。

### 1.2 自己那張卡有沒有被污染

| 證據 | 看到什麼 | 判讀 |
|:--|:--|:--|
| CSV 欄位 | 只有 `host_contention／foreign_gpu_count／foreign_max_util`，沒有 `own_gpu_intruders`、`contaminated` | 自己那張卡的狀態**沒有逐列記錄** |
| `host_contention` 的定義 | `host_contention(exclude_gpu=gpu)`：**排除自己那張卡**，只看其他卡〔git 52803bc^:code/gpu_guard.py:374–414〕 | HEAVY＝其他 6 張卡在忙，**不是**自己那張卡被佔 |
| 程式開跑前 | `GpuWatcher` 的 `started_clean` 為 False 就 `return 2`，不量〔git 59ca82f:code/m2_cost_model.py:513–517〕 | CSV 存在，代表開跑時自己那張卡是乾淨的 |
| 程式跑到一半 | CSV 在 `with GpuWatcher` **裡面**就寫出，之後才檢查 `g.contaminated`。被插隊也只是 `return 3`，**不會刪 CSV**〔同檔 :520–527〕 | 中途有沒有人插隊，**CSV 本身看不出來** |
| `gpu_guard_m2.json` | 每次 run 都覆寫。git 裡只有 19:58（d20932a，CLEAN）和隔天 04:31（4124a8e，被污染，不是這兩次）的版本；08702db 沒有提交它 | 這兩次 run 的 guard 結果**沒有留下**〔git log -- results/m2_harness/gpu_guard_m2.json〕 |
| run 目錄 | `/ssd7/hungwei/paper-hkv/runs/20260830-22*`，在 3090 機器上，這台看不到 | 無法查 stdout 的「有人插隊」訊息〔未查證〕 |
| 間接證據 1 | 兩次 run、每個 tier 各自重開 server，`gpu_kv_cache_tokens` 全部是 48,128〔CSV〕 | 每次啟動時，卡上都沒有別人佔著記憶體。RUNLOG 記過：被污染的那次，這一欄逐列不同〔git 52803bc^:results/RUNLOG.md:2371〕 |
| 間接證據 2 | 重算（drop warm）只吃自己那張卡的算力，四次是 5,426–5,518 ms，差 1.7% | 沒有算力被搶的跡象 |

**結論**：自己那張卡**很可能是乾淨的**〔判讀〕，但**沒有直接紀錄**。能確定的只有「整機 HEAVY」。

### 1.3 整機爭用讓 κ 偏向哪一邊

git 裡有一個天然對照組：`git 52803bc^:results/m2_harness/retrieval_cost_nvme_interleaved.csv`〔run 20260831-211745-m2-retrieval〕。
- 同一個量：3090、Llama-3.1-8B BF16、16K、NVMe、同一支程式。
- 交錯量三輪。前兩輪整機 HEAVY，**第三輪剛好碰到整機 QUIET**（21:34:58 起，其他卡全空）。

| 子集 | CPU warm 中位數 | 重算 warm 中位數 | 基準 | κ_cpu |
|:--|:--|:--|:--|:--|
| HEAVY 列（CPU 8 筆、重算 10 筆） | 843.4 ms（727.6–905.4） | 5,150.6 ms | 140.0 | **7.1** |
| QUIET 列（CPU 4 筆、重算 1 筆、基準 1 筆） | 565.1 ms（558.9–571.4） | 5,166.9 ms | 144.3 | **11.9** |
| QUIET 的 CPU＋全部重算與基準 | 565.1 | 5,150.6 | 143.4 | **11.9** |
| 全部混在一起 | 790.9 | 5,150.6 | 143.4 | 7.7 |

〔run 20261010-075352-f4-kappa，`results/audit_20261010/kappa_3090_contention.csv`〕

- **偏向**：整機 HEAVY 讓 CPU 取回（走 PCIe 與主機記憶體）慢了約 49%。重算幾乎不受影響（5,151 對 5,167 ms），基準也一樣（140 對 144 ms）。所以 **HEAVY 讓 κ_cpu 偏低**〔算術〕。
- 舊 RUNLOG 在 08-31 21:00 寫「這台機器 24 小時都是 HEAVY，5,900 多個樣本裡 QUIET 是 0」〔git 52803bc^:results/RUNLOG.md:1912–1926〕。fork A 照這句判「沒有 QUIET 重複」。但同一晚 21:17 開跑的交錯重量，第三輪有 11 列標 QUIET，foreign_gpu_count 是 0〔CSV 本身〕。所以 QUIET 重複**存在**，只是很少。
- 08-30 那兩次 run 的 CPU warm 是**雙峰**：SATA 是 556、594、880、888 ms，NVMe 是 548、553、885、898 ms。低的那一半剛好落在 QUIET 的 559–571 ms。只用低峰算，κ_cpu 是 12.6（SATA）和 13.8（NVMe）；只用高峰算，是 7.1 和 7.2〔算術〕。
- 8.9 落在「一半被拖慢」的中間值。QUIET 下的值大約 **12**，比 8.9 高 34%。

### 1.4 判定：**有疑慮**

- 照事先的判準：自己那張卡應該是乾淨的（間接證據）。但整機 HEAVY，而 QUIET 重複差了 +34%，超過 ±10%。所以不是「可用」。
- 也不到「不可用」：沒有證據顯示自己那張卡被污染，來源也找得到。
- **偏差的方向讓結論變強**：
  - 用 QUIET 的值，3090 的 κ_cpu 約 12，跨平台比值會從 5.8 倍變成約 7.8 倍〔算術：11.9÷1.531〕。
  - 所以「κ 跨平台差很多」這個方向不受影響，但 **8.9 和 5.8 這兩個數字本身不可靠**。
- QUIET 樣本很少（CPU 4 筆、重算 1 筆、基準 1 筆，都在同一輪）。要寫進論文，應該重量。這台 MI300X 量不了 3090，要回 3090 機器。
- **連帶影響**〔判讀，未逐一查證〕：平台 A 所有計時常數都在 HEAVY 下量的（git 裡每一個帶 `host_contention` 的計時 CSV 都是 HEAVY 或 UNKNOWN；只有 `retrieval_cost_precision_tiers_quiet.csv` 是 QUIET，但它沒有 CPU 階）。包括主設定 qwen-awq 的 CPU 0.298、SSD 10.245 ms/block。舊 RUNLOG 自己寫過「CPU 成本被高估 → headroom 很可能是高估的」〔git 52803bc^:results/RUNLOG.md:1228–1231〕。

---

## 2. MI300X 遺失的 CSV：找了哪裡、重建了什麼

**問題是什麼**：main.tex 引用 `results/m5_quality_mi300x/*.csv`、`results/m2_harness_mi300x/cost_constants_mi300x.csv` 等檔，但這些檔不在 git，也不在硬碟上。
**要證明什麼**：
- 它們是不是真的不見了；
- 剩下的東西能不能重建；
- 重建出來的和當時的分析一不一致。

### 2.1 找過的地方（全部找不到）

| 找法 | 結果 |
|:--|:--|
| `git log --all --name-only`、逐一 commit `ls-tree`（5 個分支、153 個 commit） | 從來沒有任何 `m5_quality_mi300x/*.csv`、`cost_constants_mi300x.csv`、`retrieval_cost_b-*.csv`、`recompute_position_b-*.csv`、`oracle_*.csv`、`policy_sim.csv`（MI300X）被提交 |
| `git fsck --unreachable --no-reflogs`、`git stash list` | 沒有懸空物件、沒有 stash |
| `find / -xdev` 和 `find /mlsteam /root /home /tmp` 找上述檔名 | 沒有 |
| commit `12aac34` 的訊息 | 「原始 CSV 留在 results/ 目錄但不進 git」 |
| commit `52803bc`（推倒重來） | 刪掉整個舊 results/，連同沒進 git 的 CSV 一起消失〔判讀〕 |

### 2.2 run 目錄裡剩什麼

- 外層 run：`cmd.sh`、`context.txt`、`stdout.log`、`stderr.log`、`exit_code`。
- 內層 run（程式自己開的，例如 `20260916-062212-m5-needle/`）：每個精度一個子目錄，只有 `cmd.txt` 和 vLLM 的 `server.log`。
- server.log 只有 `POST /v1/completions 200 OK`，**沒有任何模型輸出**。
- 所以：
  - m2 的取回和重算：stdout 有**逐請求**的 TTFT，可以逐列重建。
  - m5 品質：stdout 只有**每個設定的總分**和每個任務的分數，**逐題輸出不存在**，只能重建摘要。

### 2.3 重建了什麼〔run 20261010-075351-f4-rebuild，程式 `code/m9_f4_rebuild.py`〕

| 新檔（`results/audit_20261010/`） | 列數 | 來源 run | 粒度 | 自我檢查 |
|:--|:--|:--|:--|:--|
| `m2_retrieval_rows_mi300x_rebuilt.csv` | 1,057 | 18 個 m2 retrieval run 的 stdout | 每列一個請求（tier × cold/warm × prefix × 輪） | — |
| `m2_recompute_rows_mi300x_rebuilt.csv` | 277 | 9 個 m2 recompute run | 每列一個 rep（位置 P × rep） | — |
| `cost_constants_mi300x_rebuilt.csv` | 56 | 上一檔；照 `git 52803bc^:code/m2_analyze_b.py` 的算法重算 | 每模型 × 每階 | 和 `20260919-150345-m2-analyze` 的 stdout **26／26 格相同**（warm 中位數、µs/token、n、κ） |
| `recompute_fit_mi300x_rebuilt.csv` | 28 | 同上 | 每模型 × 中位數／最小值 × 範圍 | 和同一份 analyze stdout **28／28 列相同**（C0、斜率、R²） |
| `m5_quality_mi300x_summary_rebuilt.csv` | 1,317 | 52 個 m5／m5c run 的 stdout | 摘要：每設定的正確率、每任務分數、巨觀平均、和 BF16 逐字相同數、prompt 長度 | — |
| `m5_server_paperwork_mi300x.csv` | 950 | 42 個內層 run 的 `cmd.txt`、`server.log` | 每 run × 每精度：實際的 `--kv-cache-dtype`、attention 後端、KV 容量、max-model-len | — |
| `REBUILD_MANIFEST.csv` | 6 | — | 上面每個檔「是什麼、缺什麼」 | — |

每列都有這幾欄：
- `run_id`：外層 run。
- `inner_run_id`：內層 run。
- `ts`：外層 run 的開始時間。stdout 沒有逐請求的時間，`ts_source=run_start` 寫明這一點。
- `source_file:source_line`：抄自哪個檔的哪一行。
- `granularity`。

**沒辦法重建的**
- 逐題的模型輸出、`out_sha1`、逐題分數。所以「CPU 階對磁碟階 0/60」「重跑 0/60」這類逐題比對無法重做（見 claims B-070、B-071）。
- m2 的 `req_read_bytes`、`chunk_queries`。原 analyze stdout 沒有「未完整讀回」的註記，所以當時 SSD 列都通過了這個檢查〔20260919-150345-m2-analyze〕。
- `attn_importance_*.csv`（逐位置的注意力形狀）、`oracle_*.csv`、`policy_sim.csv`。fork B 從 stdout 追到大部分數字，見 claims_B。

**重建時發現的兩件事**
- 原 CSV 是 **append 模式**〔git 52803bc^:code/m2_cost_model.py:447–464 `write_rows`〕。同一路徑被跑過兩次的，原檔其實混了兩次 run 的列。例如：
  - `recompute_position_b-seedoss36b.csv` 混了失敗的 170743 和重跑的 221352；
  - `retrieval_cost_b-qwen14b-1m.csv` 混了 rc=1 的 213028 和 122356。
  - 重建檔照實保留兩個 run_id。
- 原 `cost_constants_mi300x.csv` 只有 4 個模型（Llama、Qwen-7B-1M、Qwen3-30B、Seed）。main.tex 圖 b-tiers 說「七個模型」，引用的卻是這個檔。另外 3 個模型我從它們的 run 補算了，標 `in_original_cost_constants=False`：

  | 模型 | ctx | κ_cpu | κ_ssd |
  |:--|:--|:--|:--|
  | Qwen2.5-14B-1M | 96K | 6.05 | 2.89 |
  | Mistral-Nemo-12B | 96K | 4.69 | 2.10 |
  | UltraLong-8B | 96K | 4.50 | 2.12 |

  〔cost_constants_mi300x_rebuilt.csv〕

---

## 3. main.tex 數字核對（claim table）

**問題是什麼**：論文的每個數字找不找得到來源？
**要證明什麼**：逐一核對，每個標上 run_id 和四種狀態之一。

- 範圍：摘要、引言、背景（含全部表格）、§Evaluation、§Limitations、§平台 B、結論、附錄 A，以及附錄「尚未執行」裡出現的實測數字（表 models 等）。
- 分工：
  - 摘要、引言、背景表格、結論和跨段發現：主 agent，claim_id 開頭 M；
  - 平台 A 各節：fork A，開頭 A，run `20261010-080228-f4-platA`；
  - 平台 B 各節：fork B，開頭 B，run `20261010-075307-f4-platBc`。
- 合併：run `20261010-080301-f4-claims`。

### 3.1 各區段計數〔results/audit_20261010/claims_counts.csv〕

| 區段 | 一致 | 不一致 | 找不到來源 | 來源有疑慮 |
|:--|:--|:--|:--|:--|
| 摘要（L95–109） | 3 | 5 | 0 | 6 |
| 引言（L115–142） | 4 | 4 | 0 | 1 |
| 背景（L144–459，含表 kvsize／fit／ratio／kappa） | 12 | 9 | 1 | 8 |
| §Evaluation＋Limitations（平台 A） | 25 | 12 | 6 | 24 |
| §平台 B 實測 | 38 | 16 | 8 | 14 |
| 結論 | 0 | 3 | 0 | 1 |
| 附錄 A（平台 A 明細） | 9 | 2 | 0 | 14 |
| 附錄「尚未執行」裡的數字 | 4 | 8 | 2 | 2 |
| **合計 241** | **95** | **59** | **17** | **70** |

- 「來源有疑慮」裡最大的一群，是平台 A 的計時數字。
  - 它們全在整機 HEAVY 下量，CSV 也沒有自己那張卡的污染欄位。數值本身都對得上。
  - 照 CLAUDE.md §3，HEAVY 不等於被污染，但 §1 顯示它會讓走 PCIe 的量偏大。
- 兩條獨立的抽取路徑互相驗算過。fork B 的 `platB_evidence.csv` 和主 agent 的 `m5_quality_mi300x_summary_rebuilt.csv` 抽出的平台 B 品質數字完全一致，包括 ±1、±5pp 那幾格的超出值。

### 3.2 已知嫌疑的核對結果

| 嫌疑 | 行 | 結果 | 依據 |
|:--|:--|:--|:--|
| 「64K–129K 下 LongBench／RULER 差 ±1 分內」 | 1527–1528 | **不一致** | 實際量在 LongBench 32K 視窗（prompt 最長約 18.7K token）、RULER 16K。四個模型的巨觀平均有 6 格超過 ±1：Llama RULER INT4 +1.26；UltraLong LB INT4 −1.24、RULER INT4 −1.16、RULER ptk +1.07；Qwen3 LB INT4 −1.03；Seed RULER ptk −1.18。Seed 的 BF16 本身只有 20.1／46.2，「無損」沒有鑑別力〔B-038；m5_quality_mi300x_summary_rebuilt.csv〕 |
| 「GSM8K 七個模型差 ±5pp 內」 | 1537 | **不一致** | Qwen2.5-7B-1M 在 FP8、ptk、INT4 都掉 90.8pp；Nemo INT4 −7.5；UltraLong ptk +5.84；Llama FP8、INT4 剛好 −5.00〔B-043；run 20260917-012817、010336、20260917-181725、20260916-192213〕 |
| 「INT8 95%／FP8 5%／INT4 0%」 | 106 | **數字一致，歸因不一致** | 來源是 qwen-awq、32K、n=20〔20260831-181930-m5-needle；它也在 main 的 52803bc^，不只在 claude/ 分支，A-038〕。但 MI300X 的 Qwen2.5-7B-1M 也是「只有 INT8 活下來」，Llama 全部 100%，所以是模型族的性質，不是 sm_86 的〔M-008〕 |
| κ 全距 0.7–11.7 | 102、132、1428、1608 | **不一致** | κ_cpu 實測是 1.53–11.65；0.72 是 κ_ssd〔M-005〕 |
| 41,648／120,320–547,744 | 104、1358–1371 | 數字**一致** | 量的是 KV pool 的 measure 階段〔A-071〕。41,648 是 pool 擺動的低模式（5.08 GiB），同設定的高模式是 48,128〔A-069〕；後四個的 verify 階段都 UNEXPECTED_FAIL，是 pool 大小，不是可用的上下文 |
| 13.2× | 104、1377 | **不一致** | 547,744/41,648＝13.15 算術對，但同時換了模型、權重和 KV 精度，卻寫成「由權重精度決定」。只換權重（Llama）是 2.89×〔M-007、A-072〕 |
| P* 3.5× | 106、1147、1612 | **來源有疑慮** | 37,717/10,851＝3.48。兩組常數都在 HEAVY 下量；Drop 斜率是兩端點連線〔A-019、M-009〕 |
| 「6–21 倍／54–190 倍」 | 106、334 | **不一致** | 舊的 100% MFU 算術估計，和同一篇的實測 κ（1.5–11.7；8.9–9.5）矛盾〔M-010、M-011、A-002〕 |

### 3.3 其他重要的「不一致」（完整清單在 CSV）

| 行 | main.tex 寫的 | 實際 | 依據 |
|:--|:--|:--|:--|
| 128 | Llama 在 MI300X 的 KV 預算 157.8 GiB | 155.2 GiB（−1.7%）；RUNLOG_MI300X 早已記錄 | M-020 |
| 297 | 表 kappa「末兩列只換模型，7.6 倍」 | 末兩列是 2.6 倍；7.6 是第 3 列對第 5 列 | M-036 |
| 411 | 3090 的 PCIe 是 Gen3（15.75 GB/s） | 同文 L1076、表 ratio 寫 4.0×16、31.5 GB/s；0.0024 ms 也算錯（應為 0.0039） | A-005、A-007 |
| 1084 | 重算 225 µs 對 CPU 4.2 µs | 實測 325.5 對 36.8 µs/token（舊算術殘留） | A-016 |
| 1197 | qwen-awq decode 擬合 18.158＋0.001267N | 用現存資料重算是 4.878＋0.001579N；581 GB/s 也算不出來 | A-031、A-032 |
| 1229–1242 | 表 eps-task 全是 qwen-awq；ε 差 21–36 倍 | GSM8K 欄是 Llama-BF16，倍數跨模型 | A-037、A-042 |
| 1306、1311 | 峰值在 3.5 P*；四條線都在 2–3.5 P* | hw_sweep 平台 A 線峰在 262K（6.95 P*）；兩條線峰在 1.74 P* | A-059、A-060 |
| 1326 | MI300X 的 P* 346K–693K、甜蜜點 692K–2.4M | 平台 B 已實測 P* 是幾千到一萬多 token（m2-analyze：Llama 3,396；A-062 用另一組常數得約 14K），外推被否定 | A-062；20260919-150345-m2-analyze |
| 1377 | MLA 把壓力軸推後 4.2× | 實測容量比 3.32× | A-074 |
| 1459 | 自檢 0 個 FAIL | 唯一的 verify run：5 FAIL、2 WARN | B-006；20260919-150350-verify |
| 1473–1474 | 五個 R²≥0.999；斜率 9.75「µs/千 token」 | 六個；單位差 1000 倍 | B-014、B-017 |
| 1500 | 2–8× 壓力 21–43%，四個模型都過 15% | 跑了 7 個模型，實際 5.9–43.0%，Qwen3 在 2× 只有 5.9% | B-027、B-028 |
| 1586–1588 | 預測器 AUC 0.917–0.922、ECE 0.003；四個 baseline 差 1.6% | 只有 conversation 是這樣；合成長上下文 AUC 0.26–0.49；1.6% 只在 Seed toolagent | B-061～B-067 |
| 1616 | 「κ 只有單平台實證」 | 和摘要自相矛盾；L1076、L1411、L1438 也還寫「沒有機器」 | M-045 |
| 1948–1953 | 表 models 的 κ 108／54／8／3、懸崖 315K、≈132K | 舊估計，或根本沒跑過；實測懸崖 273,872、120,320 | B-081～B-088 |


---

## 4. κ 的兩種定義：誰用哪一種、為什麼 1.5 和 16 差這麼多

**問題是什麼**：同一台 MI300X、同一個 Llama-3.1-8B，main.tex 寫 κ_cpu＝1.5，harness 第一個 chunk 卻是 16 倍。
**要證明什麼**：差距從哪來；論文該用哪一個定義。

### 4.1 現在有四種「κ」在流通

| 名稱 | 定義 | 量法 | 用在哪 |
|:--|:--|:--|:--|
| κ_tex（main.tex 表 kappa） | 重算 ÷ 傳輸，>1 代表搬比較便宜 | vLLM 0.28 整段 16K（或 96K）warm TTFT，減掉「block 還在 GPU」的基準，除以 token 數 | 摘要 8.9／1.5／11.7、表 kappa、引言、結論 |
| κ_ssd（同一張表的隱藏欄） | 重算 ÷ SSD 取回 | 同上 | 摘要的「全距 0.7」其實是 MI300X Llama 的 κ_ssd＝0.72 |
| κ_est（舊版算術） | 重算 ÷ 傳輸 | 簡化 prefill 模型（2N FLOP/token、100% MFU）÷ PCIe 理論頻寬 | 摘要的「6–21 倍／54–190 倍」、第 335 行「6–190 倍」、表 models 的 108／54／8／3 |
| harness 的 ℓ/f（D3、第一階段） | 載入 ÷ 重算（**方向相反**），每 512-token chunk 一個值 | f(i)：HF transformers SDPA 逐 chunk 量〔20261008-130316-m7-c1〕；ℓ：35.4 GiB/s×0.9 的 memcpy 頻寬〔tier_params.json〕 | D3「第一個 chunk 16 倍」是 f(0)/ℓ |

### 4.2 把 16 一步步換成 1.5〔run 20261010-075352-f4-kappa，`results/audit_20261010/kappa_reconcile.csv`〕

| 步驟 | f/ℓ（＝重算÷傳輸） | 換了什麼 |
|:--|:--|:--|
| H0 harness 第一個 chunk | **16.36** | f(0)＝28.88 ms（HF SDPA）÷ ℓ＝1.765 ms |
| H1 改成 16K 前綴的平均 f | 24.09 | 位置效應：越後面的 chunk 重算越貴 |
| H2 改用 vLLM 的逐 chunk f（扣固定開銷） | 8.71 | 引擎：vLLM 的 kernel 比 HF 快（前 32 個 chunk 平均 15.4 ms） |
| H3 改用 vLLM 整段 prefill 的平均（含每步固定開銷） | 13.56 | 46.74 µs/token × 512＝23.9 ms |
| H4 再把傳輸換成 vLLM OffloadingConnector 實測 | **1.53** | CPU 路徑 30.53 µs/token（有效約 4.3 GB/s），等於 main.tex 的值 |

- **最大的一項是傳輸路徑**：vLLM 的 CPU 卸載路徑比 harness 的 memcpy 慢 **8.86 倍**。同一台機器的 vLLM CPU 路徑在 C6 也量到 4.02–4.11 GiB/s〔results/m7_write_policy_mi300x/summary_c6_vllm_bw.csv〕。
- 重算這一側，HF 的 f(0) 對 vLLM 整段平均只差 1.21 倍。
- 所以 1.5 和 16 都是真的。它們量的是**不同的軟體路徑**：一個是 vLLM 今天的 CPU 卸載實作，一個是「如果搬運跑到 memcpy 頻寬」的上限〔判讀〕。
- 同一條 vLLM 路徑，不同模型的 CPU 有效頻寬就差 16 倍：Qwen3-30B 37.8 GB/s、Qwen-7B-1M 2.3 GB/s〔git 52803bc^:results/RUNLOG_MI300X.md:440–463〕。所以 κ_tex 有一大部分在量「軟體」，不只是「硬體」。

### 4.3 量法本身的兩個問題（新發現）

1. **MI300X 上，各 tier 的 attention 後端不一樣**。看 server.log：
   - gpu_resident（基準）、drop（重算）、gpu_fp8 用 `ROCM_ATTN`；
   - cpu、ssd、gpu_int4（以及所有 per-token-head 精度）用 `TRITON_ATTN`。
   - 〔檔案 /mlsteam/data/tiara/runs/20260915-124109-m2-retrieval/*_r0/server.log〕〔results/audit_20261010/m5_server_paperwork_mi300x.csv〕
   - 所以 κ_cpu 的分母是「TRITON 的 warm TTFT − ROCM 的基準」，INT4 的「反量化 1.59 µs/token」也一樣，混進了換後端的差。
   - 偏差多大：NOT_MEASURED。
2. **3090 和 MI300X 不是完全同一套量法**：
   - 3090：1 輪，每階 n=4，不交錯；
   - MI300X：3 輪交錯，n=12。
   - CPU 階容量也不同。表 kappa 標題寫「同一套 harness」，只在程式碼血緣上成立。

### 4.4 建議的單一定義（給寫作階段參考，不改 main.tex）

> **κ_tier(平台, 模型, L, 路徑) ＝ 重算一段長度 L 的前綴所多花的時間 ÷ 從該 tier 取回同一段前綴所多花的時間。**
> 兩者都在**同一個 serving engine、同一個 attention 後端**下，以 warm TTFT 減「前綴仍在 GPU」的基準量。報告時一定要附四樣東西：平台、模型、L、軟體路徑（例如 vLLM 0.28 OffloadingConnector）。
> 需要逐 chunk 的決策時，另報 κ(i)＝f(i)/ℓ。整段的 κ＝Σf/Σℓ，兩者用同一個方向（重算÷傳輸）。

- 這樣做的效果：
  - 摘要只能引用 κ_cpu（1.5–11.7）。
  - κ_ssd 另列。
  - κ_est（6–21／54–190、108／54）全部刪掉，或明寫「算術估計」。
  - harness 的值要寫成「若搬運達 memcpy 頻寬」的另一條路徑。
- 3090 的值要等 QUIET 重量。整機爭用會讓 κ_cpu 偏低，見 §1。

---

## 5. FP8 有沒有生效：只看書面紀錄（不重跑；GPU 重現是 F3 的工作）

**問題是什麼**：Llama 在 129K 時，FP8 的輸出和 BF16 逐字完全相同（20/20）。
**要證明什麼**：從 cmd、log 能不能看出 FP8 真的有開。

| 項目 | `20260916-062139`（65K） | `20260917-103133`（129K） | 來源 |
|:--|:--|:--|:--|
| 外層指令 | `m5_quality.py --mode needle --needle-ctx 65536` | `… --needle-ctx 129024` | 各 run 的 `cmd.sh` |
| 內層 run | `20260916-062212-m5-needle` | `20260917-103206-m5-needle` | 時間配對；stdout 印出的 CSV 路徑 |
| FP8 server 指令 | `vllm serve … --max-model-len 66048 --kv-cache-dtype fp8` | `… --max-model-len 129536 --kv-cache-dtype fp8` | `fp8/cmd.txt`（venv tiara-v028） |
| server 收到的參數 | `'kv_cache_dtype': 'fp8'`；engine config `kv_cache_dtype=fp8` | 同左 | `fp8/server.log` |
| vLLM 自己的訊息 | 「Using fp8 data type to store kv cache… may cause accuracy drop without a proper scaling factor」 | 同左 | 同上（沒有校正 scale，用預設值） |
| KV 容量 | BF16 1,267,504 → FP8 **2,535,008**（恰好 2.000 倍） | 同左 | server.log `GPU KV cache size` |
| attention 後端 | BF16 和 FP8 都是 `ROCM_ATTN`；fp8_ptk、int8、int4 是 `TRITON_ATTN` | 同左 | server.log `Overriding with …` |
| 請求參數 | `temperature 0.0, seed 12345, max_tokens 400` | 同左 | 〔git 2702df6:code/m5_quality.py:266〕（context.txt 記的 repo commit） |
| prompt 真的有那麼長 | — | BF16、FP8 兩邊的 prompt throughput 累計都約 1.39M token，prefix cache 命中率同為 45.2% | server.log loggers |
| 和 BF16 逐字相同 | FP8 17/20、fp8_ptk 10/20、int8 15/20、int4 12/20 | FP8 **20/20**、fp8_ptk 5/20、int8 18/20、int4 6/20 | 各 run 的 stdout 末段 |
| 整機爭用 | `UNOBSERVABLE_HOST`；沒有 CONTAMINATED 檔 | 同左 | stdout、內層目錄 |

**書面上能說的**
- FP8 **確實被設定、也確實改變了 KV 的儲存格式**：容量剛好兩倍，vLLM 印出「用 fp8 存 KV」。不是旗標沒傳到。
- FP8 是唯一和 BF16 **同一個 attention 後端**的低精度設定，其他三個都換成 TRITON。所以各精度的「和 BF16 逐字相同」數**不能互相比**：ptk、int8、int4 的差異裡還混了換 kernel 的效果〔判讀〕。
- 書面上**看不出**的是：attention 讀 KV 時到底是不是真的讀了反量化的 FP8 值（例如 prefill 時是不是直接用還沒寫進 cache 的 BF16）。這要 F3 在 GPU 上驗證。NOT_MEASURED。
- 同一套 FP8 設定在 Qwen2.5-7B-1M 上會把撈針打到 15／20／40／10%（4K／16K／32K／131K），而且和 BF16 逐字相同只有 0–3/20〔20260917-095033、100045、101331、20260916-065920〕。所以 FP8 在這個 ROCm 堆疊上**不是**整個沒作用。Llama 20/20 相同，比較可能是誤差小，而不是 FP8 沒開〔判讀，未查證〕。

---

## 6. 其他跨段發現

1. **卸載和不卸載用了不同的 attention 後端**。所有 MI300X lossless run 都是這樣：full_gpu 用 ROCM_ATTN，cpu_lru、tier_fs 都用 TRITON_ATTN（10／10）〔m5_server_paperwork_mi300x.csv〕。
   - main.tex 第 1601 行把「13–25/60 輸出不同」解釋成分塊邊界造成浮點順序不同。這個原因沒有量過，而換 kernel 是更直接的另一個解釋。
2. **過時的文字**。摘要和 §平台 B 寫「兩平台實測」，但下面這幾處還是舊說法，自相矛盾：
   - 第 1076 行「平台 B 尚無機器」；
   - 第 1411 行「平台 B 的常數需要機器」；
   - 第 1438 行「我們尚無該機器」；
   - 第 1616 行「κ 跨硬體主張目前只有單平台實證」。
3. **GB 和 GiB 混用**。例如「1M 需要 122 GB」實際是 122 GiB（131 GB），「192 GB HBM」在 amd-smi 是 192 GiB。數值通常剛好差 7%。
4. **舊算術值還留在文中**。這些在 2026-09-19 的 `CLAIM_EVIDENCE_MI300X.md` B1 就被標成「必須修改」，main.tex 沒改：
   - 摘要的「6–21／54–190 倍」；
   - 第 335 行的「6–190 倍」；
   - 表 models 的 κ＝108／54／8／3。

---

## 7. 做了什麼、檔案、run_id

| run_id | 做什麼 | 程式 |
|:--|:--|:--|
| `20261010-075351-f4-rebuild` | 從 stdout 重建 MI300X 的 m2 與 m5 CSV，並和原 analyze stdout 比對 | `code/m9_f4_rebuild.py` |
| `20261010-075352-f4-kappa` | 3090 κ 的整機爭用對照；κ 定義的換算 | `code/m9_f4_kappa.py` |
| `20261010-080228-f4-platA` | fork A：平台 A 數字核對（從 git 52803bc^ 抽 CSV） | `code/m9_f4_platA_check.py` |
| `20261010-074654-f4-platB`、`20261010-075307-f4-platBc` | fork B：平台 B 證據抽取與核對 | `code/m9_f4_platB_check.py`、`code/m9_f4_platB_claims.py` |
| `20261010-080301-f4-claims` | 合併三份核對成 claims_table、計數 | `code/m9_f4_claims.py` |

- 報告：`docs/audit_20261010/DATA_AUDIT.md`（本檔）、`docs/audit_20261010/claims_table.csv`。
- 資料：`results/audit_20261010/*.csv`，片段在 `results/audit_20261010/fragments/`。
- 所有新檔都小於 1 MB。沒有改任何既有檔，沒有 commit。

## 8. 限制與偏離（照實列）

- **不是獨立審查**：主 agent 和兩個 fork 是同一個模型、看過同樣的前文（D3、D6 的結論），有確認偏誤的風險。skill 要求的 zero-context reviewer 沒有做。
- **判準的補充**：「引用檔不存在但 stdout 有數字」怎麼判，是看到資料後才定的。照字面讀判準，這類應判「來源有疑慮」，現在判「一致」，引用句另判「找不到來源」。這樣「一致」的數量偏多。照字面重算的版本見摘要（一致 67、來源有疑慮 98）。
- **fork 之間的不同意見**：fork A 認為 3090 沒有任何 QUIET 重複（根據 RUNLOG 的文字）。主 agent 在 CSV 裡找到 11 列 QUIET，以 CSV 為準（§1.3）。
- **QUIET 樣本很少**：§1 的 11.9 只靠 4 筆 CPU、1 筆重算、1 筆基準。它只能說明偏差的**方向**，不能當新的 κ 值。
- **沒查**：
  - 3090 機器上的 run 目錄（這台看不到）；
  - 外部規格值（廠商頻寬、HF config 的 L／H／d，本機沒有的三個模型）；
  - 引言第 2 項「15 篇近期工作」這類文獻計數；
  - 圖本身的內容（PDF 只有成品，來源 CSV 不在）。
