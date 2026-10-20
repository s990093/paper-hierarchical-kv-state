# X5　精度×容量：每個 session 該留 BF16、在 CPU 壓成 INT4、搬去慢層、還是丟？

> 第三階段探索方向 X5（[README §2 X5](README.md)）。〔模擬，CPU〕。self-review，非 cross-model、非零前文審查。
> 不改 `main.tex`、不 commit、不改既有程式（只新增 `code/p3x/x5_*.py`）。

## 0. 一句話結論＋判定

**判定（照 §2 事先登記，兩種 session 重建規則取較保守的）：有看頭**〔模擬〕。但有看頭的**不是**精度×容量本身，而是「知道誰不會再回來」。

- 主設定（Llama、原寫法反量化 K0、reg 校正、子樹重建規則）48 格裡 11 格 headroom ≥ 15%，最大 39.8%（toolagent、128 GiB、b48、GPU 閒）〔模擬 run `20261017-175507-p3x5-grid`，判定 `20261017-181716-p3x5-analyze`〕。11 格全部在 **CPU ≤ 256 GiB（≤ 2.2% 不重複 block 工作集）且慢層 ≤ 1.47 GiB/s**；512 GiB 以上沒有一格 ≥ 15%，1024 GiB 全部 < 5%；SSD 級慢層（b16，5.27 GiB/s）全部 < 8%。
- 嚴格重建規則（README §4 第 2 點）headroom 反而更大（17 格 ≥ 15%、最大 51.3%；48 格裡 47 格比子樹規則大）〔run `20261017-181708-p3x5-strict`、`20261017-213311-p3x5-analyze-strict`〕→ 較保守的是子樹規則，判定以它為準，兩者都是「有看頭」。
- **拆解**：只知道「這個 session 之後會不會回來」（二元，不知道何時）的部分 oracle（R2b／R4b／R6b），在那 11 格拿到 oracle 好處的 77–99.7%（中位數 96.6%；嚴格規則 ≥ 99.6%）。扣掉這部分之後，「知道何時回來」再多給的（含依回來時間選精度）中位數 1.4%、最大 11.6%（只有 conversation、128 GiB、b48、閒這一格）〔同一 run，`x5_headroom.csv` 的 `share_of_headroom_from_binary_return_info`〕。
- **精度本身**：那 11 格裡最好的固定規則 8 格是 **R2（CPU 全 INT4）**，連原寫法反量化也是。oracle 會依情況選：128 GiB 時回來請求 41–84% 讀 CPU-INT4，256 GiB、GPU 閒時 60–77% 讀 CPU-BF16（因為不回來的丟掉之後放得下 BF16）。〔判讀〕對固定規則來說，精度的答案是「容量緊就全 INT4」；oracle 依情況選精度的那部分值很小（見拆解）。
- **品質未驗證**：每格 4.8–5.5% 的回來請求 context > 32K 而且讀到 INT4。只算 ≤ 32K 的回來請求，那 11 格的 headroom 變成 10.9–31.3%，8 格仍 ≥ 15%〔模擬，run `20261017-220204-p3x5-le32k`，§4.5〕。
- **兩個 Mooncake trace 不獨立**：conversation 12,031 個請求裡 10,938 個在 toolagent 有相同的（時間、輸入、輸出長度）〔實測，§4.1〕，所以「兩個 trace 都成立」只算一份證據。
- **兩個判定版本**（README §4）：**原登記（只有 Mooncake）＝有看頭**（總結以這個為準）；**含 FreeInference＝有看頭**（Mooncake 的格子照樣成立），但 **FreeInference 本身＝死路**：96 格 headroom 全部 < 1%（最大 0.69%）〔run `20261017-213331-p3x5-fi`、`20261017-220031-p3x5-analyze-fi`〕。原因〔判讀〕：一週裡同時活躍的 session 很少、回來很快（間隔中位數 9.5 s），連 128 GiB（0.76% 工作集）都放得下活躍的 session，86–98% 的回來直接命中 CPU-INT4、0% 到慢層，沒有東西可以決定。FreeInference 88% 的回來請求 context > 32K 而且讀 INT4 → **品質未驗證**；34.8% 超過 128K（Llama 放不下），全部是外插。
- **設限穩健性**（X9 提醒，事後加，不改判定）：只評估距離 trace 結束 ≥ 1,800 s 的決定，子樹規則 12 格 ≥ 15%（最大 39.4%）、嚴格規則 18 格 → 仍是有看頭；**沒設限的版本較保守（11 格），判定照它**。但設限後「只知道會不會回來」的比例降到中位數 80%（最低 62%），「知道何時」再多的最大到 18.0%（conversation、128 GiB、b48、閒）。→ 「好處幾乎都是二元資訊」這句在設限後要改成「大部分」。H＝3,600 s：Mooncake 只有 3,537 s，**沒有決定可以評估**；FreeInference 兩個 H 都是死路。〔run `20261017-213557-p3x5-cens1800`、`-214751-p3x5-strict-cens1800`、`-215650-p3x5-fi-cens1800`、`-215735-p3x5-fi-cens3600` 與對應的 analyze〕
- **只載入還原**（X6 提醒，只報告）：不用 Cake、有存的就整段載入，主設定 17 格 ≥ 15%（最大 60.7%）→ 一樣有看頭，二元資訊佔中位數 99.4%〔run `20261017-215824-p3x5-loadonly`、`20261017-220137-p3x5-analyze-loadonly`〕。
- 〔判讀〕X5 的「值得」主要和 X1 的「O-丟掉」、X2 的「會不會回來」是同一件事。精度要不要依回來時間決定：這裡看到的值很小（主設定中位數 0.2%，≥ 15% 的格子中位數 1.4%；設限後最大 18%、8 格 ≥ 5%，都在 CPU ≤ 256 GiB、慢層 ≤ 1.47 GiB/s）。

## 1. 問題與要證明什麼

第二階段量到：INT4 在慢層省 36–57% 還原時間，在真實 CPU 層（原寫法反量化）反而可能變慢，但 INT4 讓容量變 3.2 倍（main.tex §5.1 表 V、表 VI、§6.3）。所以「CPU 裡放 INT4」是用還原速度換容量。

要回答：**這個取捨會不會隨 session 的回來時間而不同，讓「知道回來時間」變得值錢？** 具體：知道每個 session 下一次回來時間（與回來時要讀多少 block）的 oracle，在四個去處（CPU-BF16／CPU-INT4／慢層 INT4／丟掉）之間選，比最好的固定規則快多少（回來請求的完整 TTFT 平均）。

- 如果 oracle 幾乎等於最好的固定規則 → 精度×容量的取捨不需要預測（固定規則就夠）。
- 如果 oracle 快很多 → 再看那個差距來自哪裡（丟掉不回來的、精度選擇、還是順序），交給 X1／X2 判「猜不猜得到」。

## 2. 判準（開跑前寫，2026-10-17T09:45:46Z `date -u`；寫的時候沒有任何 X5 的模擬結果）

### 2.1 工作負載與 session 重建（事先寫死）

- Mooncake FAST'25 `conversation`、`toolagent`（`/ssd8/hungwei/paper-hkv/datasets/mooncake/traces/`，commit d750cdd，sha256 見該目錄 README.txt），**原始長度**、原始時間戳（毫秒），依時間順序一次服務一個請求（和第一階段相同，沒有排隊）。
- 單位斷言（規則 6）：`len(hash_ids)==ceil(input_length/512)` 吻合率 ≥ 0.95；前綴鏈（同一個 hash 永遠在同一位置、前一個 hash 相同）0 違反。
- **共用前綴**：前綴樹上「出度（不同的下一個 hash，含請求結束）≥ 64」的節點與它的所有祖先＝共用前綴（system／tool prompt）。探索時看到出度分布在 16–1023 之間是空的（conversation 只有 block 0；toolagent 3 個節點），所以 16–1023 之間任何門檻結果都一樣；程式裡斷言這一段是空的，不空就停。共用前綴的 block 一律常駐 CPU BF16、計入容量、所有策略相同。
- **session**＝請求的第一個非共用 block（前綴樹在共用前綴下的子樹）。請求 r 的匹配深度 k＝r 開頭連續「之前出現過」的 block 數；共用深度 s。k > s 的請求是**回來請求**（讀到 session 自己的 block）。
- **session 存的東西**＝該 session 最近一次請求的私有 block（`len(hash_ids) − s` 個）。回來請求實際能從 session 讀到的 block 數＝r 和 session 目前路徑的共同前綴（從舊分支分叉出去、不在目前路徑上的 block 要重算，次數另外報告）。
- **A 欄驗算 B 欄**（規則 6）：對「接在上一輪最後一個完整 block 後面」的回來請求，`input_length(r) ≥ input_length(上一輪) + output_length(上一輪) − 512` 的比例要 ≥ 0.8（hash 鏈說它是續寫，長度欄要看得到上一輪的輸出被接進來）；不到就停下來回報。

### 2.2 成本（每個 block；第二階段實測，〔MI300X 常數〕）

- 模型：**Llama-3.1-8B 為主設定**，Qwen2.5-7B 另列。每個 block 的位元組取 `restores.csv` 的 `bytes_stored ÷ n_blocks`（不從名稱推）。
- 還原用 Cake 兩條線（`m10_sweep_sim.simulate`，import 不改）：重算線 f(i) 取 `speed_<m>/bw.json`（run `20261015-154558-m10-calibf-llama`／`-154647-…-qwen`）的每 block 中位數，i ≥ 64 線性外插〔推算〕；GPU 忙＝閒 × 同一個 INT4 grid 的 `compute_only` 忙÷閒比值。
- 載入線每個 block：CPU-BF16＝E0 只載的 `t_h2d ÷ 64`；CPU-INT4＝E1 只載的 `t_h2d ÷ 64`＋Cake 列的 `t_dequant ÷ n_load_int4`；慢層＝`max(位元組 ÷ B, h)＋d`（同 `m10_iowall`）。慢層頻寬用主線量過的三個：Llama b16／b32／b48＝5.27／1.47／0.42 GiB/s（Qwen 2.77／0.76／0.22）。慢層容量無限。
- **校正**：兩條線的推算和實測 Cake 中位數比（8K／16K／32K 閒、32K 忙）。CPU 層的時間一律乘「實測÷推算」比值（在 16／32／64 個 block 之間線性內插、範圍外取最近的值，〔推算〕）；慢層推算誤差中位數 ≤ 5% 才用推算，不過就停下來回報。
- 反量化兩種寫法各算一次：**原寫法 K0**（主線凍結的，`results/m10_precision_mi300x/speed_*`）與**融合版**（stage 42，`results/m10_sweep_mi300x/speed_dqfast_int4_*`，只在 32K CPU 層量過；其他長度、慢層用它的每 block 反量化時間〔推算〕）。
- 新一輪的 prefill（r 的 `input_length − 512k` 個 token）用同一個 f(i) 依比例算，所有策略相同。丟掉＝共用前綴從 CPU 載入＋私有 block 全部重算。
- 不模擬：寫入慢層的頻寬與干擾、CPU 裡壓縮的時間、排隊、GPU 層（都假設在背景、免費；所有策略一樣受惠，但壓縮免費偏袒有「CPU 內壓縮」的 R3–R6 與 oracle）。

### 2.3 策略（參數事先寫死）

非 oracle（固定規則，決定時只看已知資訊：大小、上次存取時間、狀態、成本表）：
- **R1** 論文設計：寫入 CPU-BF16；CPU 滿了把 LRU session 降到慢層（INT4）。
- **R2** CPU 全 INT4：寫入就是 CPU-INT4；滿了把 LRU 降到慢層。
- **R3** 寫入 BF16；滿了先在 CPU 裡把**最大的** BF16 session 壓成 INT4（同大小取較舊的）；沒有 BF16 可壓了才把 LRU 的 INT4 降到慢層。
- **R4** 同 R3，但先壓 **LRU** 的 BF16 session。
- **R5** 兩段 LRU：BF16 最多佔 CPU 容量的 50%；滿了時 BF16 超過 50% 就壓 LRU 的 BF16，否則把 LRU 的 INT4 降到慢層。
- **R6** 成本感知（Pensieve／GDSF 式，無參數）：每個在 CPU 的 session、每個可用動作（壓縮／降級），分數＝(動作後的還原時間 − 現在的還原時間，用目前大小算) ÷ 釋放的位元組 ÷ 閒置時間，取最小的做。
- 丟掉：慢層容量無限時，Cake 從慢層還原不會比全部重算慢，所以「丟」只對不回來的 session 有用；非 oracle 不知道誰不回來，所以 R1–R6 不丟。

oracle（知道每個 session 下一次回來的時間與要讀的 block 數；一族取最好，B7 教訓）：
- **OA**＝R1 的動作，順序改成 Belady（最晚回來的先降；不回來的直接丟）。
- **OB**＝R4 的動作（先壓再降），順序改成 Belady；不回來的直接丟。
- **OC**＝R2（全 INT4），順序改成 Belady；不回來的直接丟。
- **OD**＝R6 的分數，把「閒置時間」換成「距離下次回來的時間」、還原時間用下次真的要讀的 block 數算；不回來的分數＝0（先丟）。
- **oracle 值**＝每格 OA–OD 裡最快的；**最好的固定規則**＝每格 R1–R6 裡最快的。

### 2.4 指標與判定

- **主指標**：回來請求的完整 TTFT 平均（還原＋新一輪 prefill）。同時報中位數、p90、只算還原的部分、全部請求的總和。
- **headroom**＝(T_最好固定規則 − T_oracle) ÷ T_最好固定規則，逐格算。
- **格子**＝trace × 模型 × CPU {128, 256, 512, 1024} GiB（同時報占工作集的百分比）× 慢層 {b16, b32, b48} × GPU {閒, 忙} × 反量化 {原寫法, 融合}。
- **判定**（只用原始長度）：
  - **有看頭**：主設定（Llama、原寫法反量化 K0）下，至少一個 trace、至少一種 CPU 大小、至少一個（慢層, GPU）組合 headroom ≥ 15%。（比 README 嚴：只在主設定判；Qwen、融合版只報告。）
  - **死路**：**所有**格子（兩個模型、兩種反量化、全部 CPU／慢層／GPU）headroom < 5%。（比 README 嚴：Qwen、融合版也要 < 5%。）
  - 其他＝**可能**。
- **品質**：INT4 只在 32K 驗證過 2 分界線（main.tex §5.4）。回來時讀到 INT4 block、而且 context > 32K 的回來請求，標「品質未驗證」，每個策略每格報比例。≥15% 的格子若主要靠這些請求，照實寫。
- **只報告**（不改判定）：長度映射到 8K–32K 的結果；oracle 是否不慢於所有規則（違反要列）；oracle 的動作按「距離回來的時間」分箱的比例（精度選擇是否真的隨回來時間變）；headroom 裡「丟掉不回來的」佔多少（OD 關掉丟掉的版本）。
- 模擬只能篩（main.tex §3.3：策略差距誤差 5–6 個百分點），「有看頭」最多讓方向進下一輪。

## 3. 做了什麼

全部只用 CPU（`CUDA_VISIBLE_DEVICES=""`、`HIP_VISIBLE_DEVICES=""`），最多 8 個 process；Python＝`/ssd8/hungwei/paper-hkv/venv/tiara/bin/python -I`。每個 run 目錄 `/ssd8/hungwei/paper-hkv/runs/<RUN_ID>/` 都有 `cmd.sh`、`context.txt`（`date -Is`、`nvidia-smi`、`git rev-parse HEAD`＝6d4f729、`x5_*.py` 的 sha256）、`stdout.log`、`stderr.log`、`exit_code`。

### 3.1 程式（只新增，`code/p3x/`）

| 檔 | 內容 |
|:--|:--|
| `x5_trace.py` | Mooncake 載入、規則 6 斷言、共用前綴、兩種 session 重建（`subtree`＝事先登記；`strict`＝README §4 事後補充）、長度映射（只報告，沒跑） |
| `x5_fi.py` | FreeInference 載入（事後補充；選週規則寫在檔頭） |
| `x5_costs.py` | 第二階段實測 → 每 block 成本 → Cake 兩條線（import `m10_sweep_sim.simulate`，不改）→ 還原時間表；校正 variant `reg`（登記）／`excess`（敏感度）／`loadonly`（X6 敏感度） |
| `x5_sim.py` | 策略 R1–R6、oracle OA–OD、ODn、部分 oracle R2b／R4b／R6b；評估回來請求 TTFT |
| `x5_run.py` | 格子 driver（multiprocessing）；`--rule`、`--fi`、`--censor-H`、`--main-only`、`--variants` |
| `x5_analyze.py` | headroom、判定、session-cluster bootstrap（2,000 次）、表 |

### 3.2 run

| run_id | 做什麼 | 產出（`results/p3_explore/x5/`） |
|:--|:--|:--|
| `20261017-175507-p3x5-grid` | 主格子：2 trace × 2 模型 × 4 CPU × 24 成本設定（reg／excess × K0／融合 × 閒／忙 × b16／b32／b48）× 14 策略，子樹規則；5,376 列、1,283 s、exit 0 | `x5_cells.csv.gz`、`x5_trace_meta.csv`、`x5_cost_constants.json`、`x5_cost_validation.csv`、`x5_oracle_actions.csv`；raw `ttft.npz` |
| `20261017-181716-p3x5-analyze` | 主判定 | `x5_headroom.csv`、`x5_verdict.json`、`x5_bootstrap.csv`、`x5_actions_summary.csv`、`tables_x5.md` |
| `20261017-181708-p3x5-strict` | 嚴格重建規則，只跑 reg；2,688 列、exit 0 | `*_strict.*` |
| `20261017-213311-p3x5-analyze-strict` | 嚴格規則的判定 | `x5_verdict_strict.json` 等 |
| `20261017-181924-p3x5-fi` | **中斷**（我砍掉：同時跑嚴格規則，超過 8 個 process）；目錄有 `INTERRUPTED`、`KILLED_BY_AGENT.txt`，數字不用 | — |
| `20261017-213331-p3x5-fi` | FreeInference 重跑，reg；1,344 列、exit 0 | `*_fi.*` |
| `20261017-213557-p3x5-cens1800`、`-214751-p3x5-strict-cens1800`、`-215650-p3x5-fi-cens1800`、`-215735-p3x5-fi-cens3600` | 設限穩健性（事後加），主設定（Llama、K0、reg）；exit 全部 0 | `*_cens1800.*`、`*_strict_cens1800.*`、`*_fi_cens1800.*`、`*_fi_cens3600.*` |
| `20261017-215824-p3x5-loadonly`（raw npz 在 `…-loadonly-raw/`） | 只載入還原（X6 事後加），主設定；exit 0 | `*_loadonly.*` |
| `20261017-220031-p3x5-analyze-fi`、`-220051-…-cens1800`、`-220055-…-strict-cens1800`、`-220105-…-fi-cens1800`、`-220120-…-fi-cens3600`、`-220137-…-loadonly` | 各版本的判定（exit 全部 0） | `x5_verdict_*.json`、`x5_headroom_*.csv`、`tables_x5_*.md` |
| `20261017-220204-p3x5-le32k` | §4.5：只算 ≤ 32K 的回來請求 | `x5_headroom_le32k.csv` |

### 3.3 用到的第二階段實測檔（全部 gpu_guard CLEAN；程式斷言沒有被標污染的列、`kv_bad`＝0）

| 用途 | 檔案 | run_id |
|:--|:--|:--|
| f(i)（GPU 閒，64 個 block 的中位數） | `results/m10_precision_mi300x/speed_llama/bw.json`、`speed_qwen/bw.json` | `20261015-154558-m10-calibf-llama`、`20261015-154647-m10-calibf-qwen` |
| 原寫法 K0：E0／E1 的 cake、load_only、compute_only（8K／16K／32K 閒，32K 忙；CPU 層＋b16／b32／b48） | `results/m10_precision_mi300x/speed_llama/restores.csv.gz`、`speed_qwen/restores.csv.gz`（只取該 run 的列） | `20261016-034902-m10-grid-int4-llama`、`20261016-052143-m10-grid-int4-qwen` |
| 融合反量化（stage 42，只有 32K CPU 層，閒／忙） | `results/m10_sweep_mi300x/speed_dqfast_int4_llama/restores.csv`、`speed_dqfast_int4_qwen/restores.csv` | `20261016-235027-m10-dqfast-grid-int4-llama`、`20261017-000209-m10-dqfast-grid-int4-qwen` |

由這些算出的常數（`x5_cost_constants.json`）〔MI300X 常數〕：

| | Llama | Qwen |
|:--|:--|:--|
| 每 block 位元組 BF16／INT4（`bytes_stored`÷64） | 64／20 MiB | 28／8.75 MiB |
| f(0)、f(63)、外插斜率 | 29.0、84.8 ms、0.885 ms/block | 23.6、72.5 ms、0.770 |
| GPU 忙÷閒（compute_only 32K） | 2.107 | 2.122 |
| CPU 層 H2D 每 block BF16 閒／忙 | 1.25／1.88 ms | 0.56／0.81 |
| CPU 層 H2D 每 block INT4 閒／忙 | 0.43／0.47 ms | 0.22／0.24 |
| 反量化每 block（Cake 列）K0 閒／忙 | 0.71／1.82 ms | 0.41／1.04 |
| 反量化每 block 融合 閒／忙 | 0.16／0.47 ms | 0.08／0.35 |

**沒有實測、用推算的組合**〔推算〕：GPU 忙的 8K／16K（用 32K 的比值）；融合版的 8K／16K 與所有慢層（用融合版 32K CPU 層的每 block 反量化代入）；> 64 個 block（f 線性外插、比值取 64 的）；慢層一律是兩條線推算、沒有校正（登記的 gate）。注意：CPU 層 INT4 在 8K／16K **有實測**（GPU 閒），不是推算。

**推算驗證**（`x5_cost_validation.csv`）：慢層推算對實測 cake 中位數，誤差中位數 Llama 3.5%、Qwen 3.4%（24 格），**過 5% gate**；但慢層 INT4 在 b16 一律偏快（Llama 閒 16 個 block −34%、32 個 −24%；忙 −23%），也就是推算讓慢層 INT4 看起來比實際便宜。CPU 層原始推算誤差中位數 Llama 12%、Qwen 21%，乘比值後在實測點上等於實測。

## 4. 結果

### 4.1 trace 與 session 重建〔實測，`x5_trace_meta.csv`、`x5_trace_meta_strict.csv`〕

| | conversation 子樹 | toolagent 子樹 | conversation 嚴格 | toolagent 嚴格 |
|:--|--:|--:|--:|--:|
| 請求 | 12,031 | 23,608 | 12,031 | 23,608 |
| 單位吻合率、前綴鏈違反 | 1.000、0 | 1.000、0 | 同左 | 同左 |
| 共用前綴節點（出度 ≥ 64）／block | 1／1 | 3／18 | 同左 | 同左 |
| session／有回來的 session | 7,373／2,259 | 18,781／2,415 | 9,595／1,959 | 21,093／2,061 |
| 回來請求 | 4,658 | 4,827 | 2,436 | 2,515 |
| A/B 驗算（續寫的回來請求 `L(r) ≥ L(q)+O(q)−512`） | 99.90%（n＝3,833） | 99.90%（3,861） | 99.96%（2,391） | 99.96%（2,443） |
| 回來間隔中位數／p90 | 123／480 s | 114／420 s | 132 s | 120 s |
| 回來能讀的 block 數 k_eff 中位數／p90／最大 | 12／49／240 | 12／47／240 | 12 | 13 |
| 回來請求 context > 32K（k_eff > 64） | 5.5% | 4.8% | 6.0% | 5.0% |

- 嚴格規則的 conversation 2,436 條連結、20.2% 的請求有子請求，和文獻 agent 的 `mooncake_chain2.py`（2,436、20.2%）一樣。
- 子樹規則的「有子請求」比例＝4,658 ÷ 12,031＝38.7%，介於寬鬆（44%）和嚴格（20%）之間〔算術〕。
- **兩個 trace 不獨立**：conversation 12,031 個請求裡有 10,938 個，在 toolagent 有一模一樣的 (timestamp, input_length, output_length)（hash 編號不同）〔實測，同一個 run 的 `overlap_rows_same_ts_in_out_with_other_trace`〕。所以「兩個 trace 都成立」不是兩份獨立的證據。
- 容量的分母（X9 提醒）：128 GiB＝不重複 block 工作集（Llama 11.2 TiB）的 1.12%，＝「每個 session 最大私有 block 數的總和」（10.3 TiB）的 1.24%〔算術〕。本文的百分比都用不重複 block；換成 session 加總只差 1.1 倍（這裡的 session 加總是每個 session 取最大的一次，不是每個請求加總，所以和 X9 的 1.27–2.60 倍不同定義）。

### 4.2 主判定（事先登記：子樹規則、Llama、K0、reg）〔模擬，`tables_x5.md`〕

headroom（括號＝session-cluster bootstrap 95% CI），回來請求完整 TTFT 平均：

| CPU（占工作集） | 慢層 | conversation 閒 | conversation 忙 | toolagent 閒 | toolagent 忙 |
|:--|:--|--:|--:|--:|--:|
| 128 GiB（1.1%） | b16 5.27 GiB/s | 7.9% | 4.0% | 7.3% | 3.7% |
| | b32 1.47 | **21.8%** [19.7, 24.2] | 13.3% | **21.7%** [19.4, 24.1] | 12.8% |
| | b48 0.42 | **39.4%** [36.2, 42.6] | **29.6%** [27.0, 32.4] | **39.8%** [36.3, 43.2] | **29.7%** [26.8, 32.6] |
| 256 GiB（2.2%） | b16 | 5.2% | 2.6% | 3.8% | 1.9% |
| | b32 | **15.1%** [13.4, 17.0] | 8.7% | 13.0% | 6.9% |
| | b48 | **29.6%** [26.6, 32.9] | **20.7%** [18.5, 23.3] | **26.3%** [23.2, 29.4] | **18.0%** [15.7, 20.4] |
| 512 GiB（4.5%） | b16／b32／b48 | 2.2／6.9／14.6% | 1.1／3.7／9.4% | 0.8／4.7／10.0% | 0.5／2.0／5.9% |
| 1024 GiB（9.0%） | b16／b32／b48 | 0.5／1.7／3.7% | 0.4／0.9／2.0% | 0.0／0.8／1.8% | 0.2／0.3／0.6% |

- 判定：**有看頭**（11 格 ≥ 15%；其中 10 格 CI 下界也 ≥ 15%）。主設定 headroom 中位數 4.9%。
- 絕對值（例：conversation、128 GiB、b48、閒）：最好規則 R2 0.681 s → oracle OD 0.413 s；只看還原那一段，headroom 是 81.2%。
- 死路的條件（reg 下 192 格全部 < 5%）不成立：85 格 ≥ 5%、30 格 ≥ 15%（Llama K0 11、Llama 融合 10、Qwen K0 4、Qwen 融合 5）。
- **oracle 不是上界**：7 格 oracle 比 R6 慢，最多 1.3%（全部是融合反量化，多在 b16、GPU 忙）〔`tables_x5.md` 最後一節〕。這些格子的 headroom 本來就 < 5%，不影響判定。

### 4.3 兩種重建規則（README §4 第 2 點；判定照較保守的）

| | 子樹（登記） | 嚴格 |
|:--|--:|--:|
| 主設定 ≥ 15% 的格子 | 11／48 | 17／48 |
| 主設定 headroom 中位數／最大 | 4.9%／39.8% | 8.7%／51.3% |
| 主設定逐格「子樹 ≤ 嚴格」 | 47／48 格 | |
| 逐格取兩者較小的值，≥ 15% 的格子 | 11（和子樹相同） | |
| 判定 | 有看頭 | 有看頭 |

- 〔判讀〕嚴格規則只留「父請求唯一、前綴很少人共用」的連結，被當成「不回來」的 session 變多（有回來的 session 1,959 vs 2,259），所以「知道誰不回來」更值錢。

### 4.4 好處從哪裡來（只報告）

| 主設定 ≥ 15% 的 11 格 | 數字 |
|:--|:--|
| 部分 oracle（只知道會不會回來，R2b／R4b／R6b 取最快）拿到 oracle 好處的比例 | 77.4%–99.7%，中位數 96.6%（嚴格規則：99.6%–99.9%） |
| 扣掉二元資訊後，「知道何時」再多的 headroom（T_部分 − T_oracle）÷ T_部分 | 中位數 1.4%，最大 11.6%（conversation、128 GiB、b48、閒） |
| reg 下 192 格，這個「剩下的」≥ 5% 的格子 | 6 格（最大 11.6%）；excess 校正也是 6 格、最大 11.8% |
| OD 和 ODn（不准丟）的 TTFT | 全部格子相同（慢層無限時，丟和降到慢層對不回來的 session 效果一樣） |
| 最好的固定規則 | R2（CPU 全 INT4）8 格、R6 2 格、R4 1 格 |
| oracle OD 的動作（conversation、128 GiB、b48、閒） | 丟 7,257 次（全部是不回來的）；壓縮／降級，依距離下次回來：60–300 s 1,369／178、300–1,200 s 639／240、≥ 1,200 s 46／56（`x5_actions_summary.csv`） |

- 〔判讀〕oracle 的確依回來時間選去處：越晚回來越常降級而不是壓縮（降級比例 11% → 27% → 55%）。但這部分只值中位數 1.4%。X5 問的「精度×容量的取捨會不會隨回來時間變、讓預測有用」，在 Mooncake 上的答案是：**有變，但值很小；大的部分是「會不會回來」**。

### 4.5 品質未驗證的部分（INT4 在 > 32K）

- 主設定 ≥ 15% 的 11 格，最好規則的回來請求有 5.5%（conversation）／4.8%（toolagent）是「context > 32K 而且讀到 INT4」；oracle 是 4.0–5.5%。這些請求的品質**未驗證**（main.tex §5.4 只驗到 32K）。
- 只算 k_eff ≤ 64（≤ 32K）的回來請求重算 headroom（同一批 TTFT，`ttft.npz`；run `20261017-220204-p3x5-le32k`，`x5_headroom_le32k.csv`）：子樹規則 11 格變成 10.9%–31.3%，8 格仍 ≥ 15%；嚴格規則 17 格變成 12.6%–41.3%，12 格仍 ≥ 15%。→ 判定不靠 > 32K 的請求。

### 4.6 其他設定（只報告，`tables_x5.md` 第二節）

- **融合反量化**：Llama 主格子 ≥ 15% 的格子數：conversation 5、toolagent 5（K0 是 6、5）；最大 40.2%／40.6%。〔判讀〕融合讓 CPU 層 INT4 不再比 BF16 慢，但 headroom 幾乎不變，再次說明好處不是來自精度。
- **Qwen**（每 block 小 2.3 倍，同樣 GiB 放得下更多）：最大 headroom 23.8–28.5%，≥ 15% 的格子 2–3 個／trace。
- **excess 校正**（我在開跑前看驗證表後加的敏感度，§6）：判定一樣是有看頭；主設定 11 格 ≥ 15%、最大 40.9%。

### 4.7 FreeInference（事後補充，登記之後才加；`x5_trace_meta_fi.csv`、`tables_x5_fi.md`）

- 週＝2026-07-26_2026-08-01（第一個 ≥ 1,000 session 的週，看結果前寫死）：84,524 個請求、1,198 個 session（1,193 個有回來）、83,326 個回來請求。單位斷言 `len(block_ids)==floor(L/16)` 100%；A/B 驗算（續寫的回來請求 `L(r) ≥ L(q)+completion(q)−512`）99.06%（n＝80,433）。
- 回來間隔中位數 9.5 s、p90 58.9 s；k_eff 中位數 198 個 block（約 101K token）、p90 539、最大 1,904；88.2% 的回來 > 32K、34.8% > 128K；最長的請求 1,249,133 token〔實測，資料本身〕。f(i) 從 64 個 block 外插到 2,500 個〔外插〕。
- 128／256／512／1024 GiB＝session 工作集（Llama 16.4 TiB）的 0.76／1.5／3.1／6.1%。
- 結果：96 格（2 模型 × 2 反量化 × 4 CPU × 6）headroom 0.0–0.69%，只算還原最多 7.3%；最好的規則幾乎都是 R6；oracle 違反 0 格。
- 判定：**死路**（FreeInference 單獨）。兩個判定版本：原登記（Mooncake）＝有看頭；含 FreeInference＝有看頭（因為 Mooncake 那幾格）。

### 4.8 設限穩健性（事後補充，X9 提醒；不改判定）

**做法（跑之前定好）**：t_c＝最後一個請求的時間 − H。(a) 只評估上一輪在 t_c 之前的回來請求；(b) oracle 與部分 oracle 只能把「最後一次出現在 t_c 之前」的 session 當成不回來，之後才最後出現的當作在 T_end＋H 回來（不准丟）。規則（不知道未來）不受影響，只換評估的請求。

| 版本（主設定 48 格；FI 24 格） | 評估的回來請求 | ≥ 15% | 最大 | 二元資訊佔比（≥ 15% 的格子，中位數／最低） | 「知道何時」再多的最大 |
|:--|:--|--:|--:|:--|--:|
| 子樹、沒設限（登記） | 4,658／4,827 | 11 | 39.8% | 96.6%／77.4% | 11.6% |
| 子樹、H＝1,800 s | 2,289／2,342 | 12 | 39.4% | 80.0%／62.1% | 18.0% |
| 嚴格、沒設限 | 2,436／2,515 | 17 | 51.3% | 99.9%／99.6% | 0.2% |
| 嚴格、H＝1,800 s | — | 18 | 52.4% | 80.0%／71.2% | 18.0% |
| Mooncake、H＝3,600 s | **0**（trace 只有 3,537 s） | 不可評估 | | | |
| FreeInference、H＝1,800／3,600 s | — | 0／0 | 0.64%／0.65% | — | — |

- 判定在每個版本都不變；**較保守的是沒設限的子樹版本**。
- 〔判讀〕設限之後「知道何時」變得比較重要（最大 18%，conversation、128 GiB、b48、閒；主設定有 8 格 ≥ 5%），因為一部分「不回來」的標籤被拿掉了。所以 §0 的「好處幾乎都是二元資訊」在設限下要說成「大部分（中位數 80%）」。

### 4.9 只載入還原（事後補充，X6 提醒；只報告）

- 做法：有存的 block 整段載入（CPU 層乘 32K 實測只載÷推算只載的比值，慢層不校正），丟掉的照舊重算；決策同主設定。驗證：32K CPU 層 BF16 82.4 ms（實測 82.36）、INT4 60.0 ms（實測 59.96）。
- 主設定 17 格 ≥ 15%、最大 60.7%；二元資訊佔中位數 99.4%（最低 81.3%），「知道何時」最大 21.1%。
- 〔判讀〕沒有 Cake 時，放錯層的代價更大（慢層要整段讀），headroom 變大；但結構相同：大部分來自「會不會回來」。

## 5. 前作

- **main.tex §5.1 表 V、表 VI、§5.4、§6.3**（本 repo，第二階段）：INT4 在慢層省 36–57%、在真實 CPU 層（原寫法）變慢；INT4 容量 3.2 倍；INT4 品質只驗到 32K。本方向的成本全部取自它們背後的 run（§3.3）。
- **D4**（`docs/research_20261010_directions/D4_workload.md`）：Mooncake 上「知道重用」對總 TTFT 只差 0.015%／0.002%，只有 CPU ≤ 10% 工作集才有差。本方向把指標換成回來請求、單位換成 session、加上精度選擇之後，差距出現在 CPU ≤ 2.2%、慢層 ≤ 1.47 GiB/s，方向一致。
- **Pensieve**（EuroSys'25，`refs.bib` pensieve2025）：成本÷閒置時間的淘汰；R6 照這個形式寫〔本次未讀原文，用 main.tex 表 I 的描述〕。
- **CachedAttention**（ATC'24）：session 級 KV 放在 DRAM／SSD 兩層、用排隊中的請求預取〔卡片轉引，`research/LIT_AND_TRACES.md` 的 A_CachedAttention 卡；本次未讀原文〕。
- **KIVI**（ICML'24）：INT4 格式〔main.tex 引用；本次未讀原文〕。
- 我沒有找到「依 session 的回來時間決定 KV 精度與層」的前作；這次沒有做文獻搜尋，只是沒找，不是沒有。

## 6. 偏離與問題（照實列）

1. **成本模型在開跑前加了 excess 校正**（2026-10-17 約 09:55Z，看了 `x5_costs.py` 的驗證表、還沒跑任何策略模擬）：登記的比值校正在 k < 16 會讓 CPU-INT4（42 ms）比 CPU-BF16（34.5 ms）慢、甚至比慢層 b16（29–62 ms，沒有校正）還慢，因為 CPU 層乘了 1.19–1.45 而慢層沒乘。所以加了「差值內插（k＝1 時差＝0）、慢層也校正」的 excess 版本當敏感度。判定用登記的 reg；excess 也是有看頭。
2. **主指標的 prefill 依 token 比例算**：新一輪很短時，實測 `t_new_turn`（32K 約 75 ms）比比例算的大。這個共同項偏小 → headroom 偏大。只算還原的 headroom 另列（§4.2 的 81.2% 那種）。
3. **沒模擬的**：寫慢層的頻寬與干擾、CPU 內壓縮的時間、排隊、GPU 層、慢層容量（無限）。壓縮免費偏袒 R3–R6 與 oracle；慢層無限讓「丟」等於「降級」（OD＝ODn）。
4. **慢層推算偏快**：b16 的 INT4 偏快 23–36%（§3.3）。登記的 gate 是看誤差中位數（3.5%），過了，但偏差有方向。excess 版本有校正慢層，判定不變。
5. **session 存的是最近一次請求的路徑**：從舊分支分叉、不在目前路徑上的 block 當成重算。回來請求有 0.43%（conversation）／0.44%（toolagent）有這種情況（共 158／161 個 block）。
6. **只模擬整個 session 一起搬**（X6 提醒）：真實系統（vLLM、LMCache、SGLang、TRT-LLM、Dynamo）先丟尾端、只認最長前綴，頭不在、尾在的 block 不算命中，和 Cake 的方向相反。我的模擬器的假設：一個 session 的私有 block 永遠在同一個狀態（CPU-BF16／CPU-INT4／慢層／丟），沒有「一部分在、一部分被丟」；還原一律 Cake（前段重算＋後段載入）。只報告的「只載入」敏感度見 §4.9（判定不變）；「先丟尾端」的部分驅逐要改 session 狀態，不便宜，沒做（§7 第 3 點）。
7. **兩個 trace 不獨立**（§4.1）。
8. **oracle 不是上界**：7 格比 R6 慢 ≤ 1.3%（§4.2）。
9. **FreeInference 是登記之後才加的**（README §4，10:01Z）。我加入它的時間：約 10:2xZ 寫載入程式（第一次 run 10:19Z 開跑後被我砍掉，見第 10 點）、13:33Z 重跑（`20261017-213331-p3x5-fi`）。當時已經看過 Mooncake 子樹規則的全部結果（主判定「有看頭」、拆解）和嚴格規則的判定。選週規則（第一個 ≥ 1,000 session 的週）是看 FreeInference 任何結果之前寫在 `x5_fi.py` 檔頭的。f 外插到 2,500 個 block 用的是向量化的兩條線（和 `m10_sweep_sim.simulate` 在 k ≤ 260、192 張表上最大差 7e-15）；先試的 min–max 簡化版比貪婪的兩條線樂觀（中位數 6%、最大 49%），沒用。FreeInference 的共用前綴用「每個 session 第一個請求的前 2,048 個 16-token block、被 ≥ 64 個 session 共用」，這個門檻沒有像 Mooncake 那樣檢查出度分布有沒有空隙。
10. **FreeInference 第一次 run 中斷**：`20261017-181924-p3x5-fi` 我自己砍掉（同時有嚴格規則的 8 個 process，超過上限），留了 `INTERRUPTED`；數字不用，重跑成 `20261017-213331-p3x5-fi`。
11. **設限穩健性是事後加的**（X9 提醒，13:4xZ）：做法在跑之前定好（§4.8 開頭），不改登記的判定。當時已經看過所有沒設限的結果。Mooncake 只有 3,537 s，H＝3,600 s 沒有任何決定可以評估。設限版的 OB oracle 比較慢（同一個 heap 裡壓縮過的 INT4 越來越多，找 BF16 要掃過去），子樹版跑了 710 s，結果不受影響。
12. **工作階段中斷**：10:2xZ（+08:00 的 18:2x）因 API 用量上限中斷，13:33Z（21:33）恢復；中斷時嚴格規則的 run 已經 exit 0，FreeInference 第一次 run 已經被我砍掉（第 10 點）。
15. **只載入還原（X6）是事後加的敏感度**：`x5_costs.py` 加了 `loadonly` variant（只新增、不改 reg），13:5xZ 跑；X6 的「先丟尾端」部分驅逐沒有做（§7 第 3 點）。
13. **長度映射（8K–32K，只報告）沒有跑**：程式（`x5_trace.map_lengths`）寫好了，時間不夠。
14. **計時**：從開始到寫完約 4 小時（不含中斷），超過預算的 3 小時。

## 7. 如果要繼續，下一步

1. **把「會不會回來」交給 X2 判可預測性**：這裡的好處 77–99.7% 來自二元資訊。下一輪應該問的是「session 結束時，能不能用負載訊號預測它之後不會回來」（分類，有設限），不是「何時回來」。
2. **同一個問題換成會讓「何時」變值錢的設定**：慢層容量有限（丟掉才真的有代價）、寫入干擾、併發。這些都可能讓精度×時間的那 1.4% 變大，要在登記時寫好再跑。
3. **部分驅逐**（X6）：模擬器加「先丟尾端、只認最長前綴」的 block 級狀態，和 Cake 的「頭重算、尾載入」比；這需要改 `x5_sim` 的 session 狀態，不便宜，這次沒做。
4. **INT4 > 32K 的品質**：等 MI300X 的 64K／128K 結果；FreeInference 這種長 session 全部落在未驗證區。
5. GPU 驗證（照 README §1：模擬只能篩）：挑 conversation、128 GiB、b48 的 R2 對 R2b 在 harness 上重播。
