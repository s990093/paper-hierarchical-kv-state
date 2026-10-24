# X8　agent 工作負載與長 context：猜對回來時間的價值會不會變大？

> 方向登記：[`README.md`](README.md) §2 X8。共同規則：同檔 §1。
> 標記：〔模擬〕〔算術〕〔外插〕〔MI300X 常數〕〔原文 p.X〕〔摘要〕〔判讀〕。
> 這份文件是 self-review，不是零前文審查。

---

## 0. 一句話結論＋判定

**判定（原登記版 A，照 §2.3 字面）：有看頭。** 含 FreeInference 的 B 版也是有看頭；H＝1,800／3,600 s 的觀察窗穩健性版本也都是有看頭（但 H≥1,800 時只剩 Bailian coder 撐住，Mooncake 不到 1,000 個 session）；Mooncake 兩種重建規則各自判也都是有看頭。來源：`results/p3_explore/x8/x8_verdict.json`（run `20261017-220831-p3x8-verdict2`，資料 run `20261017-213609-p3x8-grid3`、`20261017-214055-p3x8-grid3-fi`）。

**但 X8 要問的「agent／長 context 讓猜對的價值比 Mooncake 大很多」不成立**〔模擬〕：
- 成立的格子都來自**分鐘級回來、短 context** 的 trace（Bailian coder、Mooncake）。本地 SSD、GPU 閒、G＝16 GiB 時 headroom 13.7–28.9%，但絕對只差 20–105 ms（回來請求的平均 TTFT 0.15–0.40 s）。而且中位 79–97% 來自「知道會不會回來」（P-Ret），不是「何時回來」；對手只有 LRU／TTL（照 README 的清單，沒有抗掃描對手）。
- **真的 agent trace（有工具時長）在本地 SSD 上都很小**：FreeInference（10,714 個 session，登記後加的）≤3.0%，CacheWise（只報告）≤2.1%。FreeInference 只有在「沒有慢層、CPU ≤256 GiB」的格子過門檻（22–28%），截到 131,072 token 後就只剩「可能」。
- toolagent 和 conversation 的 headroom 差不多（R-carry 下 G16 本地閒 20.7–25.3% vs 17.5–26.2%），而且兩份 trace 有 90.9% 的請求（時間、輸入長度、輸出長度）完全一樣，**不是兩份獨立證據**（run `20261017-220905-p3x8-overlap-check`）。
- **長 context 讓相對 headroom 變小**：toolagent 從原始長度的 15.1–25.3% 降到 128K 的 6.7–14.0%；絕對差距變大（37–44 ms → 220–456 ms），因為新一輪 prefill 也跟著變大。
- **GPU 釘住（TTL）**：1,656 格裡固定 TTL 與 Continuum 式 TTL 沒有一格贏過「GPU 區當 LRU」。agent 的秒級回來，LRU 就接住了。猜對回來時間在 agent 上省的是 **HBM，不是 TTFT**：CacheWise 原始時間下，oracle 的 TTFT 和 LRU 一樣，但平均只釘 0.41 GiB，LRU 釘 8.31 GiB。對別的請求的延遲影響＝NOT_MEASURED（模型沒有排隊）。

---

## 1. 問題與要證明什麼

agent 的工具呼叫讓 session 幾秒到幾分鐘後一定回來（Continuum 用工具時長設 TTL，把 KV 釘在 GPU）；長 context（64K–128K）讓放錯層的代價變大、能同時放的 session 變少。
要回答：**在這兩種情況下，完美知道回來時間（oracle）比最好的固定規則快多少？是不是比 Mooncake（第一階段的參考工作負載）大很多？**

只回答「猜對值多少」，不回答「猜不猜得到」（那是 X2、X3）。

---

## 2. 判準（開跑前寫；`date -u` = 2026-10-17T09:43:22Z；寫的時候沒有任何 X8 的模擬結果，也還沒載入任何 trace）

### 2.1 模擬設定（事先寫死）

- **單位**：session 層級。每個 session 在閒置時整份 KV 在一個地方：**GPU 釘住**（還原 0）／**CPU**／**慢層**／**丟**（回來全部重算）。回來時用 Cake 兩條線還原（`m7_sim.sim_restore` 的同一演算法，CPU／慢層每個 block 固定載入時間、無爭用）。
- **負載**：trace 原始時間，一次一個請求、沒有排隊（和 D4 `cmd_sim`、第一階段相同）。所以 **GPU 釘住對別的請求的代價只用容量算**：釘住的 GiB（時間平均）、換成「少了幾個 32K Llama 序列的 batch 容量」（每個 4 GiB）。對別的請求的延遲影響＝**NOT_MEASURED**（沒有排隊模型）。
- **成本〔MI300X 常數〕**：Llama-3.1-8B BF16，一個 block＝512 token＝64 MiB。f(i)＝`results/m7_write_policy_mi300x/calib_c1.csv`（run `20261008-130316-m7-c1`）每個 block 的中位數；block ≥80 用最小平方直線外插〔外插〕。載入：CPU 1.7647 ms／block、本地 SSD 8.9527、NFS 188.774（`tier_params.json`，run `20261008-130257-m7-c0`、`20261008-130438-m7-c2`）。GPU 忙＝重算（含新一輪 prefill）×2.07（main.tex §3.4）。
- **格子**：GPU 釘住預算 G ∈ {0, 16, 64} GiB × CPU C ∈ {128, 256, 512, 1024} GiB × 慢層 ∈ {本地 SSD、NFS、沒有慢層（CPU 滿了就丟）}（慢層容量不限）× GPU ∈ {閒、忙}。
- **策略（參數事先寫死）**：
  - 固定規則：**F-LRU**（GPU 釘住區當 LRU 前綴快取 → CPU LRU → 慢層；vLLM 預設行為的類比）；**F-TTL-T**，T ∈ {0, 2, 10, 60, 300} s（釘在 GPU T 秒，到期或擠不下就降到 CPU，CPU 以下 LRU；T＝0＝論文設計）；**F-Cont**（Continuum 式：T＝**trace 前半段**算出的工具時長中位數；CacheWise 依工具名、沒看過的工具用全體中位數、人類回合 T＝0；Mooncake／Bailian 沒有工具名，用前半段全體回來間隔的中位數）。
  - oracle 族（知道每個 session 下次回來的確切時間；trace 結束前不回來＝永遠不回來）：**O-Belady**（每一層滿了就把「下次回來最晚」的往下推，永不回來的直接丟）；**O-GDS**（優先度＝距下次回來的時間 × 大小 ÷ 留在這層比下一層省的還原時間）。每格取兩者中較快的（B7：oracle 取一族最好）。
  - 對稱性：所有策略都能用 GPU 釘住、CPU、慢層、丟這四個去處；oracle 只多了資訊。**都沒有預取**（預取是 X1／X3）。
- **評估**：模擬整條 trace（前半段暖機），只算**後半段**到達的請求（CacheWise：依 session 開始時間的後半）。主指標＝回來請求（有可還原的前綴）**完整 TTFT 的平均**（還原＋新一輪 prefill）；同時報中位數、p90、只算還原的平均、後半段全部請求的總和。
- **headroom**＝(最好固定規則的平均 − 最好 oracle 的平均) ÷ 最好固定規則的平均，逐格算。另報對 F-TTL-0（論文設計）與 F-LRU 各自的 headroom。
- **CI**：依 session 重抽（1,000 次）的 95% CI。
- **長度**：原始長度＋映射（所有 token 長度 × k，k 讓該 trace 請求輸入長度的中位數變成 64K 或 128K，超過 131,072 截斷，截斷比例要報）。**判準只用原始長度**，映射只報告。

### 2.2 工作負載

| 代號 | 資料 | session 怎麼來 | 判定資格 |
|:--|:--|:--|:--|
| W0 | Mooncake conversation | 前綴鏈重建（D4 規則） | 參考（「比 Mooncake 大很多嗎」的對照）；≥1,000 session 時也算 |
| W1 | Mooncake toolagent | 前綴鏈重建 | ≥1,000 session 時算 |
| W2 | Qwen-Bailian coder | 明確的 parent_chat_id | ≥1,000 session 時算 |
| W3 | Qwen-Bailian traceB（To-B API 自動化） | 明確的 parent_chat_id | ≥1,000 session 時算 |
| W4 | CacheWise（Claude Code，94 session，有工具時長） | 原生 session | **只報告** |

「session 數」＝後半段有至少一次回來的獨立 session 數。

### 2.3 判定（照字面）

- **有看頭**：至少一個 ≥1,000 session 的工作負載（原始長度），存在一組 (G, 慢層, GPU 狀態)，在 **≥2 種 CPU 大小** headroom ≥15%，且這些格子的 session 重抽 95% CI 下界 ≥10%。（比 README 嚴：多了「≥2 種 CPU 大小」與 CI 下界。）
- **死路**：所有真實工作負載（W0–W4，原始長度）的所有格子 headroom 都 <5%。
- 其他＝**可能**。
- 另外報告（不改判定）：成立的格子是不是只在 NFS／沒有慢層；G＝0 與 G>0 的差；映射到 64K／128K 後的 headroom；oracle 是否不慢於所有固定規則（違反要列出）；各策略釘住的 GiB 與慢層寫入量。

### 2.4 執行時補的細節（09:5xZ，跑 grid 之前；看過 trace 的描述統計，還沒有任何 headroom）

- **CacheWise 時間對齊**：94 個 session 的（去識別）日期散在 359 天，同時存在的最多 3 個。所以跑兩種：`cw_orig`（原始時間）與 `cw_stack`（每個 session 都從 0 開始，最大重疊）。兩者都只報告。
- **Mooncake session 重建**：D4 的規則會把 toolagent 9,202 個共用 12 個 block 模板的請求併成一個 session（README §9 坑 #2）。改成「子請求要含父請求全部完整 block、而且帶了父請求的輸出（child.input ≥ parent.input＋parent.output−512）、每個父請求只能被接續一次」；並加上規則 6 斷言「同 session 前後兩個請求，後一個 ≥ 前一個留下的 KV−512」≥95%。下文叫它 **R-carry**。
- **Bailian traceB**：172,800 個請求全部 turn＝1、parent＝−1，沒有 session 結構 → W3 拿掉（不是結果，是資料沒有這個欄位）。

### 2.5 事後補充（2026-10-17T10:06:07Z，主 session 10:01Z 的 README §4）

**這時我已經看過**：登記的 grid（run `20261017-175416-p3x8-grid`）全部格子的 headroom 與照字面的判定（「有看頭」，來自 W0／W1／W2 的本地 SSD 與 NFS 格子）；長度映射的格子；`P-Ret` 的拆解；CacheWise 的格子。登記後才加的 `P-Ret`（09:58Z）、`--dedup`（10:00Z）、`H-Turn2`（10:03Z）都在看過 grid 之後，**只報告**。
**新加**（都在看過上面的結果之後，照 README §4 列兩個版本，總結以原登記版為準）：
1. **W5 FreeInference agentic trace**（HF `harvardMadsys/freeinference_agentic_trace` @ `16d3c04d`）：session＝`session_id`（子 agent 是自己的 session）；前綴＝同 session 前後請求 `block_ids`（16-token 鏈式）的共同前綴；到達＝`start_ms`、閒置開始＝`end_ms`；回來種類＝`step_trigger`（tool／user）。Continuum 式 TTL＝**前 8 週**（2026-05-17 → 07-11）的工具時長（`latency_ms`）中位數，依工具名組合，沒看過的用全體中位數，使用者回合 T＝0；**評估＝後 8 週**的 session（session 不跨週）。只跑原始長度（原生就是長 context；token 用 o200k_base 計，直接當 Llama token，不換算）。規則 6：`len(block_ids)==floor(Σrole_tokens/16)`、工具觸發的請求「下一個 start − 上一個 end ≈ latency_ms」、`transition=append` 時共同前綴＝上一個請求的全部 block。
2. **Mooncake 第二種重建規則 R-strict**：照 `docs/phase3_20261017/research/calc/mooncake_chain2.py`（共用前綴最多 3 個請求、唯一候選父請求、時間嚴格較晚）。W0、W1 各報 R-carry 與 R-strict；**判定照較保守（headroom 較小）的那個**。
3. 判定列兩版：**原登記版**（W0–W4；Mooncake 取兩種重建裡較保守的）與 **含 FreeInference 版**。
4. FreeInference 最長 2,460 個 block（1.26M token），45% 的請求 >131,072 token（Llama-3.1-8B 的上限）。照登記跑原始長度（f(i) 直線外插到 2,560 個 block〔外插〕），另加 `fi_clip`（截到 131,072）只報告。

### 2.6 事後補充（13:4xZ，主 session 轉來 X9 的提醒；這時已看過 grid2 全部結果與 dedup、H-Turn2）

- **觀察窗穩健性**：短觀察窗會把「晚點回來」當成「不回來」，oracle 的「丟」會偏樂觀。加報只評估「決定時間（同 session 前一個請求的閒置開始）離觀察窗結束 ≥H 秒」的回來請求，H＝1,800、3,600 s。觀察窗結束＝trace 最後一個到達；FreeInference＝該週最後一個到達（session 不跨週）。**不改原登記的判定**；和 H＝0 不同就在 §0 寫明，總結以較保守的為準。
- 2026-10-17T13:33Z 恢復工作時檢查：`20261017-181007-p3x8-fi` 有 `exit_code`＝0（16 週全部跑完、斷言全過），不需要 INTERRUPTED 檔、沒有重跑。

---

## 3. 做了什麼

全部只用 CPU（`CUDA_VISIBLE_DEVICES=""`），每個指令都走 `code/p3x/x8_run.sh`（CLAUDE.md §4.1 的殼），run 目錄在 `/ssd8/hungwei/paper-hkv/runs/<run_id>/`（`cmd.sh`、`context.txt`、`stdout.log`、`stderr.log`、`exit_code`，全部 exit 0）。

| 步驟 | 程式 | run_id | 產出 |
|:--|:--|:--|:--|
| trace → session 請求表（第一版，D4 規則；發現錯誤合併） | `code/p3x/x8_traces.py` | `20261017-174818-p3x8-traces` | 作廢（見 §6.1） |
| trace（R-carry） | 同上 | `20261017-174953-p3x8-traces` | grid 用的 npz |
| trace（加 `shared_tok`；基本欄位逐位元相同） | 同上 | `20261017-175626-p3x8-traces2` | dedup 用的 npz |
| trace（加 R-strict） | 同上 | `20261017-180838-p3x8-traces3` | 最終 npz；`results/p3_explore/x8/x8_traces.csv` |
| FreeInference → 請求表 | `code/p3x/x8_fi.py` | `20261017-181007-p3x8-fi` | `freeinference.npz`（放在 traces3 目錄）；`x8_traces.csv` 追加 |
| 容量算術、長度×層的還原時間、重複計算倍率 | `code/p3x/x8_capacity.py` | `20261017-175508-p3x8-capacity` | `x8_capacity.csv`、`x8_restore_len.csv`、`x8_dedup.csv` |
| 登記的 grid（W0–W4，R-carry） | `code/p3x/x8_sim.py` | `20261017-175416-p3x8-grid` | 第一版結果（存在 run 目錄 `x8_*_grid1.csv`） |
| 加 P-Ret／H-Turn2 欄位重跑（登記的策略逐格相同，差 ≤4×10⁻⁷） | 同上 | `20261017-175950-p3x8-grid2` | 中間版 |
| dedup 敏感度（只報告） | 同上 `--dedup` | `20261017-175733-p3x8-grid-dedup`、`20261017-180303-p3x8-grid2-dedup` | `x8_headroom_dedup.csv`、`x8_cells_dedup.csv.gz` |
| **最終 grid**：W0、W1（R-carry＋R-strict）、W2、W4 × 三種長度 × H∈{0,1800,3600} | 同上 | **`20261017-213609-p3x8-grid3`** | `x8_headroom.csv.gz`、`x8_cells.csv.gz` |
| **FreeInference grid**（fi、fi_clip；原始長度；`X8_NMAX=2560`） | 同上 | **`20261017-214055-p3x8-grid3-fi`** | `x8_headroom_fi.csv`、`x8_cells_fi.csv.gz` |
| 判定 | `code/p3x/x8_verdict.py` | **`20261017-220831-p3x8-verdict2`**（從 .gz 重跑 `20261017-220937-p3x8-verdict3`，除 ts 外逐位元相同） | `x8_verdict.json`、`x8_summary.csv` |
| X5 提醒的驗算（Mooncake 兩檔重疊、FreeInference 間隔／大小／併發） | 單行指令（見 run 的 `cmd.sh`） | `20261017-220905-p3x8-overlap-check` | 只有 stdout |

- 自我檢查：`x8_sim.cake()` 和第一階段的 `m7_sim.sim_restore()` 在 3 層 × 8 種長度逐格相同（`selftest`，max |Δt|＝0）。
- 格子數：每個工作負載 × 長度 3 G × 4 C × 3 慢層 × 2 GPU＝72 格；每格 7 個登記的固定規則＋2 個 oracle＋P-Ret＋H-Turn2。

指令（可複製）：
```bash
T3=/ssd8/hungwei/paper-hkv/runs/20261017-180838-p3x8-traces3/traces
code/p3x/x8_run.sh grid3 "python -I code/p3x/x8_sim.py --traces $T3 --outdir results/p3_explore/x8 --rawdir \$RUN --procs 8 --wl mc_conv mc_tool mc_conv_s mc_tool_s bl_coder cw_orig cw_stack"
code/p3x/x8_run.sh grid3-fi "X8_NMAX=2560 python -I code/p3x/x8_sim.py --traces $T3 --outdir results/p3_explore/x8 --rawdir \$RUN --procs 8 --wl fi fi_clip --lens orig --tag _fi"
code/p3x/x8_run.sh verdict 'python -I code/p3x/x8_verdict.py'
```
（python＝`/ssd8/hungwei/paper-hkv/venv/tiara/bin/python`）

## 4. 結果

### 4.1 資料（規則 6 斷言全過）〔`x8_traces.csv`，run `20261017-180838-p3x8-traces3`、`20261017-181007-p3x8-fi`〕

| 代號 | 請求 | session | 回來 | eval 有回來的 session（H＝0／1800／3600） | ctx 中位／p90（token） | 回來間隔 p10／50／90（s） | 斷言 |
|:--|--:|--:|--:|:--|:--|:--|:--|
| W0 conv R-carry | 12,031 | 8,117 | 3,914 | 1,209／251／0 | 6,909／27,367 | 42／123／474 | 單位 1.0、成長 1.0、ms 1.0 |
| W0 conv R-strict | 12,031 | 9,595 | 2,436 | 1,157／172／0 | 同上 | 45／132／526 | 成長 0.996 |
| W1 tool R-carry | 23,608 | 14,459 | 9,149 | 2,530／215／0 | 6,346／16,806 | 0／3／258 | 單位 1.0、成長 1.0、ms 0.993 |
| W1 tool R-strict | 23,608 | 21,093 | 2,515 | 1,180／141／0 | 同上 | 45／120／459 | 成長 0.996 |
| W2 Bailian coder | 43,011 | 26,406 | 16,605 | 4,173／2,759／1,064 | 4,540／12,776 | 6.1／118／942 | hash 前綴 0.996、turn＋1 1.0、成長 0.996 |
| W3 Bailian traceB | 172,800 | — | 0 | — | — | — | 沒有 session（全部 turn＝1） |
| W4 CacheWise | 11,453 | 81（另 13 個空） | 11,372 | 39／38／38 | 80,306／141,499 | 0／0.085／27.7 | 工具時長 1.0、cache_read≈上一輪 0.932、成長 0.969 |
| W5 FreeInference | 1,222,199 | 14,036（12,002＋子 agent 2,034） | 1,208,163（工具 1,134,691、使用者 73,472） | 10,714／10,690／10,662 | 118,093／380,339 | 0.42／1.88／33.7 | 單位 1.0、append 前綴 1.0、工具時長 1.0 |

- W5 有 45% 的請求 >131,072 token、最長 1,259,654；W4 有 16% >131,072。
- **W0 和 W1 不獨立**：conversation 12,031 個請求裡有 10,938 個（90.9%）在 toolagent 有一模一樣的（timestamp, input_length, output_length）；hash 清單相同的只有 25 個（兩檔各自編號）〔run `20261017-220905-p3x8-overlap-check`〕。

### 4.2 判定（`x8_verdict.json`）

| 版本 | H＝0 | H＝1,800 s | H＝3,600 s | 誰撐住 |
|:--|:--|:--|:--|:--|
| **A 原登記**（W0–W4；Mooncake 每格取兩種重建中較小的） | **有看頭**（48 組） | 有看頭（14 組） | 有看頭（13 組） | H＝0：W0、W1、W2；H≥1,800：只剩 W2 |
| A，只用 R-carry／只用 R-strict | 有看頭／有看頭 | 有看頭／有看頭 | 有看頭／有看頭 | |
| B＝A＋FreeInference | 有看頭（54 組） | 有看頭（20 組） | 有看頭（19 組） | |
| 只看 FreeInference（fi） | 有看頭（6 組） | 有看頭（6 組） | 有看頭（6 組） | 全部是「沒有慢層」、C＝128／256 |
| 只看 fi_clip（截到 131,072） | 可能 | 可能 | 可能 | |
| dedup 敏感度（只報告） | 有看頭 | — | — | |

「組」＝(工作負載, G, 慢層, GPU)，在 ≥2 種 CPU 大小 headroom ≥15% 且 CI 下界 ≥10%。**只看本地 SSD 的組**：A＠H0 有 12 組（W2 3 組、W0 5 組、W1 4 組）；A＠H1800 只剩 W2 的 (G16, 閒)、(G64, 閒)；A＠H3600 只剩 W2 的 (G64, 閒)。**死路**的條件（全部 <5%）在任何版本都不成立。oracle 沒有一格慢於任何固定規則／P-Ret／H-Turn2（違反 0 格）。

### 4.3 主要格子：本地 SSD、GPU 閒、G＝16 GiB、原始長度、H＝0〔模擬，MI300X 常數〕

| 工作負載 | headroom C＝128／256／512／1024 GiB | CI 下界 | 絕對差（ms） | 最好固定規則的平均 TTFT（s） |
|:--|:--|:--|:--|:--|
| W0 conv R-carry | 22.0／26.2／24.6／17.5% | 19.7／23.7／21.9／15.3% | 77／90／77／50 | 0.351／0.342／0.312／0.286 |
| W0 conv R-strict | 22.3／25.8／22.7／17.4% | 20.0／23.1／20.2／15.0% | 89／100／82／59 | 0.389（C256） |
| W1 tool R-carry | 20.7／25.3／22.0／15.1% | 18.8／22.9／19.8／13.3% | 37／44／36／22 | 0.176／0.173／0.162／0.148 |
| W1 tool R-strict | 25.2／28.9／25.0／18.2% | 22.7／26.0／22.4／15.9% | 92／105／86／57 | 0.363（C256） |
| W2 Bailian coder | 17.3／18.0／16.8／13.7% | 16.6／17.5／16.3／13.2% | 28／29／26／20 | 0.163／0.159／0.154／0.148 |
| W4 cw_orig | 0／0／0／0% | — | 0 | 1.017 |
| W4 cw_stack | 2.0／0.8／0.4／0.4% | — | 22／9／5／4 | 1.103／1.073／1.067／1.067 |
| W5 fi | 2.5／1.0／0.4／0.3% | 2.3／1.0／0.4／0.2% | 76／28／11／7 | 2.983／2.736／2.679／2.672 |
| W5 fi_clip | 5.0／4.3／4.1／4.0% | 4.7／4.1／3.9／3.8% | 37／31／30／29 | 0.740／0.728／0.726／0.725 |

- 全部格子（原始長度）的最大 headroom：W0 74.1%、W1 75.5%、W2 51.9%、cw_stack 22.7%、cw_orig 0%、fi 27.9%。≥15% 的格子比例：W0 91.7%、W1 87.5%、W2 79.2%、cw_stack 8.3%、fi 16.7%；<5% 的比例：cw_orig 100%、cw_stack 76.4%、fi 50%。大的數字都在 NFS 或「沒有慢層」。
- 觀察窗穩健性（本地、閒、G16，C＝128／256／512／1024）：W2 H＝1,800 為 15.1／16.4／16.8／14.1%，H＝3,600 為 4.2／7.9／15.3／14.9%（CI 下界 3.5／7.0／14.3／13.9%）。fi 三個 H 幾乎一樣（2.6／1.0／0.4／0.3%）。Mooncake 在 H＝1,800 只剩 141–251 個 session。

### 4.4 headroom 從哪裡來

- **「會不會回來」占大部分**：P-Ret（LRU＋只知道會不會回來）拿到的比例（中位）W0 80.0%（strict 94.7%）、W1 85.0%（strict 97.0%）、W2 78.6%、cw_stack 21.8%。FreeInference 不同：本地、閒、G16 只有 7–42%，G64 9–16%，大部分來自「何時回來」——但那裡的 headroom 本身只有 0.3–3.0%。
- **無參數的事後規則拿不到**：H-Turn2（session 第一個請求的 KV 先趕）在 W0–W4 的 40% 格子贏 LRU，增益中位 W2 +4.3%、W0 +5.1%、W0 strict −1.9%、W1 −4.6%、W1 strict +0.6%；FreeInference 72 格全輸（中位 −0.9%）。oracle 對 H-Turn2 的差距在本地、閒、G16 仍有 W2 12.6–15.0%、W1 22.8–27.2%。
- **只算還原的部分**（本地、閒、G16）：W1 53.6–67.1%、W2 52.5–58.4%、fi 1.3–9.0%。完整 TTFT 的 headroom 小，是因為新一輪 prefill 所有策略都一樣。
- **dedup 敏感度**（共用前綴只存一次；本地、閒、G16）：W2 15.2／15.1／13.6／10.6%、W0 21.7／25.2／22.3／15.8%、W1 20.1／22.7／17.2／10.8%，比沒 dedup 小 0.3–4.8 個百分點。照判準算仍是有看頭〔`x8_headroom_dedup.csv`，run `20261017-180303-p3x8-grid2-dedup`〕。

### 4.5 agent vs Mooncake；和 X5 的 FreeInference 結果對照

- toolagent 的 headroom 沒有比 conversation 大（§4.3），而且兩者 91% 是同一批請求（§4.1）。
- 有工具時長的兩份真實 agent trace（CacheWise、FreeInference）：本地 SSD ≤3.0%。**秒級的工具回來，LRU 本來就接得住**：FreeInference 的 F-LRU 在 G16 時 39% 的回來直接在 GPU，C＝1,024 時只有 0.14% 落在慢層。
- **X5 說 FreeInference 96 格都 <1%、128 GiB 放得下活躍的部分、0% 落在慢層**。我這邊本地 SSD 是 0.1–3.0%、C＝128 時 F-LRU 有 8.2% 的回來落在慢層，設定差在這裡（X5 的設定我沒有讀，只寫自己的）：
  - **每個 session 的大小**：Llama-3.1-8B BF16（128 KiB／token），不截斷 → session 中位 14.5 GiB、p90 46.5、p99 88.9、最大 153.8 GiB。截到 131,072 的 fi_clip 在 C＝128 只有 1.4% 落在慢層。
  - **併發**：後 8 週每小時取樣，同時存在的 session 中位 20、p90 35、最多 48 → 活躍部分約 20 × 14.5 GiB，超過 128 GiB。
  - **間隔**：閒置間隔＝下一個 `start_ms` − 上一個 `end_ms`，中位 1.88 s（工具 1.69 s、使用者 96 s）。X5 的 9.5 s 我不知道怎麼定義的。
  - 方向一致：在 FreeInference 上，本地 SSD 的猜對價值很小。

### 4.6 GPU 釘住（TTL）與它的代價

- **F-LRU 在全部 1,656 格（W0–W4 1,512＋W5 144，H＝0）都是最好的固定規則**；TTL＝0／2／10／60／300 s 與 F-Cont 沒有一格嚴格比它好。F-Cont 的 TTL：W0 120 s、W1 0 s（R-strict 120 s）、W2 102.6 s；CacheWise 39 個鍵（全體 0.081 s）、FreeInference 786 個鍵（全體 1.30 s）。
- 釘住 vs 不釘（對論文設計 F-TTL-0 的 headroom）：cw_orig G16 17.4%、G64 22.3%；fi_clip G16 23.6–24.4%、G64 36–37%。但這些 F-LRU 全部拿到了，oracle 對 F-LRU 只剩 0–5%。
- **釘住的 HBM**（平均 GiB，本地、閒、G16、C256）：W1 LRU 14.5 vs oracle 14.6；W2 15.4 vs 15.3；cw_orig 8.31 vs 0.41；cw_stack 2.58 vs 0.68；fi 7.20 vs 5.66（G64：34.1 vs 28.7）。
- **代價怎麼算**〔算術〕：MI300X 192 GiB − Llama 權重 14.96 GiB（safetensors 檔案大小）＝177.0 GiB 當 KV 池上界。釘住 16 GiB＝上界的 9.0%＝4 個 32K Llama 序列（每個 4 GiB）的 batch 容量；64 GiB＝36%＝16 個。cw_orig 裡 LRU 多釘的 7.9 GiB 約等於 2 個 32K 序列。**對別的請求 TTFT／吞吐的影響＝NOT_MEASURED**（沒有排隊模型；Continuum 的主要收益正在這裡）。

### 4.7 長 context

**映射到 64K／128K**（只報告；本地、閒、G16，C＝128／256／512／1024）：

| 工作負載 | 原始 | 64K | 128K | 128K 絕對差（ms） | 128K 最好固定規則的 TTFT（C＝128，s） |
|:--|:--|:--|:--|:--|:--|
| W0 conv | 22.0／26.2／24.6／17.5% | 4.4／6.8／10.3／13.6% | 4.2／6.7／10.3／14.2% | 198／315／481／659 | 4.68 |
| W1 tool | 20.7／25.3／22.0／15.1% | 4.4／6.1／8.5／10.9% | 6.7／8.3／10.9／14.0% | 220／270／353／456 | 3.26 |
| W2 coder | 17.3／18.0／16.8／13.7% | 4.8／4.8／5.9／7.3% | 4.5／4.6／5.6／7.3% | 254／258／312／405 | 5.67 |
| cw_stack | 2.0／0.8／0.4／0.4% | 1.9／0.9／0.7／0.7% | 2.9／1.5／0.8／0.6% | 36／19／9／8 | 1.28 |

放大倍數 k：W0 9.5／19.0、W1 10.3／20.7、W2 14.4／28.9；128K 版有 50% 的請求被截斷。相對 headroom 變小，是因為新 prefill 也跟著放大；只算還原的部分在 128K 仍有 W1 27–58%。

**容量算術**〔算術，`x8_capacity.csv`〕：Llama BF16 一個 session 32K＝4 GiB、64K＝8 GiB、128K＝16 GiB（INT4 是 ×0.3125，main.tex §6.3）。

| 層 | 32K | 64K | 128K（INT4） |
|:--|--:|--:|--:|
| HBM 上界 177 GiB | 44 | 22 | 11（35） |
| 釘住 16／64 GiB | 4／16 | 2／8 | 1／4（3／12） |
| CPU 128／256／512／1024 GiB | 32／64／128／256 | 16／32／64／128 | 8／16／32／64（25／51／102／204） |

**還原時間**〔模擬；block ≥80 是外插〕，`x8_restore_len.csv`：

| context | CPU | 本地 SSD | NFS | 全部重算 |
|:--|:--|:--|:--|:--|
| 32K（GPU 閒） | 0.120 s | 0.484 s | 2.643 s | 3.626 s |
| 64K（閒／忙） | 0.220／0.249 s | 0.936／1.087 s | 6.607／9.816 s | 10.89／22.55 s |
| 128K（閒／忙） | 0.443／0.455 s | 1.971／2.147 s | 16.99／23.60 s | 36.33／75.20 s |

128K 的會合點：CPU 13、本地 42、NFS 166（GPU 閒）。128K 冒煙只有一次、而且是 Qwen：全部重算 30.25 s、CPU 層 E0 0.166 s（RUNLOG_MI300X 12.2）。只當量級參考。

## 5. 前作

- **Continuum**（arXiv 2511.02230）：工具呼叫時把 KV 釘在 GPU 一段 TTL，TTL 由重載／重算成本與被趕之後的排隊延遲決定；原文說有快速卸載時，回來的請求「還是要排隊等 GPU 記憶體」〔卡片 `D8_Continuum.md`，前人讀原文 p.1–5；我沒讀原文〕。我們的模擬沒有排隊，所以 **Continuum 真正賺的那一塊（排隊）不在 X8 裡**；X8 的 F-Cont 只是「用工具時長中位數設 TTL」這一條規則。
- **InferCept**（arXiv 2402.01869）：工具中斷時保留／換出／丟掉的選擇只看重載成本〔卡片 `D8_InferCept.md`，摘要級〕。
- **CacheWise**（arXiv 2606.16824）：以「工具剩餘時間的條件期望」排逐出順序；換成 CPU 卸載（不重算）時只快 1.19 倍，重算時 2.7–3.5 倍〔卡片 `P3_CacheWise.md`，前人讀原文 p.1–12；我沒讀原文〕。和 X8 的方向一致：還原越便宜，猜對的價值越小。
- **FreeInference trace 論文**（Nixon et al. 2026）：沒讀〔未查證〕；只用它的資料與 HF README。
- **KVCache in the wild**（ATC'25，Bailian trace）：沒讀〔未查證〕；只用資料。
- 本 repo 的 **D4**（`docs/research_20261010_directions/`）：Mooncake 上 oracle 對最好的延後版，總 TTFT 差距中位 0.015%／0.002%（指標是全部請求的總和、block 層級、對手含 ARC／S3-FIFO／2-hit）。X8 的指標（只算回來請求）、層級（session）、對手（只有 LRU／TTL）都不同，數字不能直接比（見 §6）。

## 6. 偏離與問題

1. **第一版 session 重建是錯的**（run `20261017-174818-p3x8-traces`，作廢）：D4 的前綴鏈規則把 toolagent 9,202 個共用 12 個 block 模板的請求併成一個 session（README §9 坑 #2）。我當時的規則 6 斷言只驗「子 vs 父」，抓不到；是看最大 session 才發現。改成 R-carry、加「同 session 前後請求的長度成長」斷言（≥95%）後重跑；這發生在任何模擬之前。
2. **Mooncake 有兩種重建**：R-carry（我的，toolagent 9,149 個回來、eval 2,530 個 session）與 R-strict（research agent 的規則，2,515 個回來、eval 1,180 個 session）。兩者 headroom 相近（§4.2）；判定每格取較小的。R-strict 共用前綴最多 3 個請求是隨手定的門檻（README §4）。
3. **Bailian traceB 沒有 session**（全部 turn＝1），W3 拿掉。
4. **CacheWise 的前綴假設「只會往後長」**：`nold＝min(上一輪 ctx＋out, 這一輪 ctx)`，只要 context 變長就假設整段前綴可重用；變短（3.6%）才當成重設。用 `cache_read≈上一輪 ctx`（93%）驗過，但 X9 在 FreeInference 看到的「中途改 system prompt／工具定義、整段前綴失效」在 CacheWise 這種寫法下抓不到（長度照樣變長）→ CacheWise 的可還原前綴可能高估。**FreeInference、Mooncake、Bailian 沒有這個假設**：前綴是同 session 前後請求的 block hash 逐一比出來的（FreeInference 有 3.7% 的 mutation、0.7% 的 switch，都照實算）。但所有工作負載的 session 都只留最新一份 KV：分叉或 switch 之後舊分支就沒了，切回去要重算。
5. **CacheWise 時間**：13 個 session 是空的（剩 81 個）；日期散在 359 天、同時最多 3 個 → `cw_orig` 幾乎沒有容量壓力（headroom 全 0），`cw_stack` 是人為對齊。兩者都只報告。
6. **Continuum 式 TTL 退化**：toolagent（R-carry）前半段回來間隔的中位數＝0 s（timestamp 解析度 1 s、同一秒回來的很多），F-Cont＝F-TTL-0。CacheWise／FreeInference 的鍵是「工具名組合」，不是單一工具。**全部 1,656 格（H＝0）裡，固定 TTL 與 F-Cont 從來沒贏過 F-LRU**：沒有排隊時，釘住的唯一代價是預算 G，而 LRU 把 G 用滿；TTL 主動讓出 G 只會更慢。Continuum 真正賺的排隊延遲不在這個模型裡 → 這不是「TTL 沒用」的證據。
7. **沒有排隊、t_idle＝t_arr**：Mooncake／Bailian 沒有完成時間，閒置從到達算起（高估閒置時間一個服務時間）。GPU 釘住對別人的代價只用容量（釘住的 GiB；每 4 GiB＝一個 32K Llama 序列的 batch 容量）；**對別的請求的延遲影響＝NOT_MEASURED**。MI300X 的 KV 池用「192 GiB − 權重 14.96 GiB（safetensors 檔案大小）＝177.0 GiB」當上界，vLLM 實際的池＝NOT_MEASURED。
8. **session 層級會重複計算共用前綴**：每個 session 存自己整份 context。重複倍率（session 層級 ÷ 不同 block）conversation 1.10×、toolagent 1.42×、coder 2.30×〔`x8_dedup.csv`〕，LRU 的容量壓力因此偏高。事後的 `--dedup` 敏感度（共用前綴只存一次、永遠在 GPU）見 §4.4；FreeInference 沒做 dedup（NOT_MEASURED）。
9. **對手只有 LRU／TTL**（照 README X8 的清單）：沒有 ARC、S3-FIFO、2-hit 這類抗掃描的對手（那是 X1）。headroom 有 79–97%（中位）來自「知道會不會回來」（P-Ret），抗掃描或准入規則可能拿走其中一部分；我事後加的 H-Turn2（無參數）拿不到（§4.3）。
10. **oracle 的右設限**：trace 結束前沒回來＝永遠不回來。Mooncake 只有 58.9 分鐘：H＝1,800 s 時只剩 141–251 個 session、H＝3,600 s 時 0 個，Mooncake 在穩健性版本裡不夠 1,000 session、不能判。
11. **長度映射**：規則是「請求長度中位數 → 64K／128K」，所以 128K 版有 50% 的請求被截到 131,072（toolagent 64K 版 14%）；新一輪的輸入也一起放大，新 prefill 變成 TTFT 的大頭。
12. **f(i) 外插**：實測只有 block 0–79（40,960 token）。64K／128K、CacheWise（最長 384 個 block）、FreeInference（最長 2,460 個 block）都是直線外插〔外插〕，而且 >131,072 token 已超過 Llama-3.1-8B 的上限；FreeInference 的 token 是 o200k_base，直接當 Llama token。128K 冒煙（RUNLOG_MI300X 12.2，Qwen、只有一次、scratch）：全部重算 30.25 s；我們的 Llama 外插 256 個 block 全部重算 36.3 s〔`x8_restore_len.csv`〕——不同模型，只當量級參考。
13. **harness 比 vLLM 慢 1.7–2.0×**（main.tex §3.1）：沒有做「f×0.5」的敏感度（NOT_MEASURED）。
14. **模擬器誤差**：新模組，沒有 GPU 驗證；第一階段模擬器在策略之間的差距誤差 5–6 個百分點（main.tex §3.3）。所以這裡的「有看頭」最多讓方向進下一輪。
15. **CI 是條件在「哪個策略最好」已經選定之後**；session 重抽 1,000 次。
16. **事後加的東西（時間）**：P-Ret 09:58Z、`--dedup` 10:00Z、H-Turn2 10:03Z（都在看過 grid1 之後）；FreeInference、R-strict 10:06Z；觀察窗 H 13:4xZ。全部只報告或另列，原登記的判定只用 §2.1–2.3 的設定。
17. **中斷**：10:2xZ 因 API 用量上限中斷，13:33Z 恢復；FreeInference 的 loader 當時已經跑完（exit 0）。
18. `results/p3_explore/x8/x8_traces.csv` 是最後一次 loader（traces3）＋FreeInference 追加的版本；之前 loader 的 meta 在各自 run 的 `stdout.log`。超過 1 MB 的 CSV 以 `.csv.gz` 存（同 `results/m7_write_policy_mi300x/b_share.csv.gz` 的做法）。

## 7. 如果要繼續

- **最值得追的一點〔判讀〕**：X8 留下來的訊號是「**分鐘級回來的工作負載上，這個 session 還會不會回來**」，不是 agent 的工具回來，也不是「何時回來」。秒級的工具回來，LRU 就接住了。這個價值在 Bailian coder（≥1,000 session、2 小時窗、H＝3,600 仍有 2 組）最穩，但對手只有 LRU。下一步要先換上抗掃描或准入類的對手（2-hit、S3-FIFO、ARC、LRU＋不寫慢層），在同一個 session 層級模擬器、W2 上重算。headroom 還 ≥15%，才值得去問 X2「會不會回來猜不猜得到」（W2 與 FreeInference 的使用者回合，看 C-index）。
- **agent 與長 context 這條線可以先停**：真的 agent trace 在本地 SSD 上 ≤3%；長 context 讓相對價值變小。GPU 釘住的價值要有排隊模型才看得到（X4 的範圍）；在目前的模型裡 TTL 一律輸 LRU。
- 如果要把 FreeInference 當主 trace：先決定模型與截斷（Llama 131K 上限 vs 原生 1M），再做 dedup（工具定義＋system prompt 是 block 開頭，跨 session 共用應該很多）。
- 所有「有看頭」都只是模擬篩選（策略差距誤差 5–6 個百分點），進論文前要 GPU 驗證。
