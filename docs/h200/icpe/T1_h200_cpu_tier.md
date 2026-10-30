# T1：ICPE 只用 H200——CPU 層換成 H200 自己量的常數，重跑受影響的模擬

> **性質**：判準登記（事先寫死）。登記時間見文末「登記」一節（`date -u`）。登記之後這一節以上不改；和登記不同的做法寫成文末的「偏離 N」。
> **來源**：使用者 2026-10-28：「這個新主張是給 ICPE 只會有一個平台 都h200」「好登記 ok」；層級常數的範圍選「只換 CPU 層」（SSD、NFS 維持限速模擬的裝置層）。
> **為什麼有這一關**：ICPE_EVIDENCE B、C 節標「H200 常數」的模擬，重算 f(i)、忙倍數、反量化都是 H200 量的，但 **CPU／SSD／NFS 三層的頻寬來自 `results/m7_write_policy_mi300x/tier_params.json`**（MI300X 上量的；`code/env_h200.sh` 的 03 §R2「只換 GPU、不換裝置」）。論文改成只有 H200 一個平台之後，CPU 層（host 到 GPU 的 PCIe，是平台本身的性質）要換成 H200 自己量的；SSD、NFS 是儲存裝置，照使用者的選擇維持原本的限速值，論文寫明數值與出處。
> **範圍**：只有 H200。只重跑模擬（CPU），不上 GPU。

---

## 1. 新常數（登記前已算出；只是把既有的 H200 實測套用既有的規則，不是實驗結果）

- **規則**＝`code/m7_params.py`（MI300X 那份 tier_params 的規則）：每個方向取實測最慢的一次（C0 排除每組第 0 次暖機）再乘 0.9。k_read＝duplex 的「單獨」中位數 ÷「同時有 D2H」中位數（MI300X 檔的 `k_read_source` 同一個定義），取到小數 3 位（同 MI300X 檔的寫法）。
- **資料**（都已經在 git）：H200 的 C0 `20261016-061254-m7h-c0`（`results/m7_write_policy_h200/calib_c0.csv`；`H2D_pinned_chunk_into_kv_view`、`D2H_pinned_chunk_from_kv_view` 去掉第 0 次後各 14 列）；duplex `20261016-061303-m7h-duplex`（`calib_c0_duplex.csv`，兩種各 5 列）。
- **值**：

| | H200（新） | MI300X（舊，H200 模擬原本用的） |
|:--|--:|--:|
| cpu 讀（GiB/s） | 0.9 × 48.432 ＝ **43.5888** | 35.4159 |
| cpu 寫（GiB/s） | 0.9 × 45.743 ＝ **41.1687** | 35.5482 |
| cpu k_read | 53.528 ÷ 53.401 ＝ **1.002** | 1.161 |
| cpu ell_chunk_ms（64 MiB） | 照新的讀頻寬重算 | 1.7647 |

- `local`（SSD 級）、`nfs`（NFS 級）：**逐位元照抄** MI300X 的檔（使用者選擇）。
- 新檔：`results/icpe_h200/t1/tier_params_h200cpu.json`（`code/icpe/t1.py build` 產生；記 `rule`、`source_runs`、`derived_from`、每個值的來源）。

## 2. 開關與輸出

- 環境變數 **`P3_TIER=h200cpu`**（沒設＝原本的行為，逐位元不變）：
  - `p3_platform.TIER_PARAMS` 指向 §1 的新檔；
  - `p3_platform.explore_out(x)` 改成 `results/icpe_h200/t1/explore/<x>`（不碰 `results/p3_explore_h200`）；
  - `code/s3/e2e.py` 的輸出改成 `results/icpe_h200/t1/e2e/`（不碰 `results/s3_h200/e2e`）。
  - 其他輸出（`p3_fvar.py --out-root`、`x1_gpu.py fullsim --out`、`p12_queue.py --out`）一律指到 `results/icpe_h200/t1/` 底下。
- **自檢**（`code/icpe/t1.py selftest`；不過就不跑）：
  1. 沒設開關時：`x1_sim.constants()`、X3 用的 CPU／慢層常數、X4 的 `load_params` 和 git 版逐值相同。
  2. 設了開關時：只有 cpu 的讀、寫、k_read（與 ell_chunk_ms）改變；其他鍵逐值相同。
  3. X5 的成本常數（`x5_costs` 算出來的）設不設開關逐值相同 → X5 不讀 CPU 層常數，不重跑（下面 §4）。
  4. 設了開關時，所有輸出路徑都在 `results/icpe_h200/t1/` 底下。

## 3. 重跑什麼（ICPE_EVIDENCE B、C 節所有「H200 常數」〔模擬〕而且會讀 CPU 層常數的列——全部重跑，不挑）

| 列 | 重跑的步驟（和原本同一支程式、同樣的參數，只加開關、換輸出目錄） | 原本的 run |
|:--|:--|:--|
| B1 | P0-1 (c) V0 的 X1：sweep、analyze | `20261021-010140-p3h-x1-sweep`→`…-011301-p3h-x1-analyze`（V0 和它逐值相同，RUNLOG ICPE-P01-C） |
| B2 | V0 的 X1 analyze `--dead-h 1800`／`3600`；p12_compare 的 X1 部分 | `20261025-194549-…-x1-h1800-V0`、`…-194554-…-x1-h3600-V0`；`20261025-215527-icpe-p12-compare` |
| B3（X1 部分） | P1-2 A2：X1 FreeInference（前 6 週、Cake）sweep、analyze、只算使用者發起 | `20261025-203058-icpe-p12-x1fi-an`、`…-203104-…-x1fiuser-an` |
| B4 | X4 主掃描（sweep、analyze）＋`p12_queue.py` | `20261021-012002-p3h-x4-sweep`→`20261022-121515-icpe-p12-queue` |
| B7（X1 部分） | P0-1 (c) V1、V2 的 X1：sweep、analyze、`--dead-h 1800`／`3600`（V0＝B1）；p01c 的判定字 | ICPE-P01-C 的 V1／V2；`20261025-202422-icpe-p01c-analyze` |
| C1 | X3：validate、tables、leads（7 個來源）、hide | `20261021-011321-p3h-x3-validate`～`…-011752-p3h-x3-hide` |
| C2（全尺寸） | P0-2 fullsim | `20261025-202314-icpe-p02-fullsim` |
| E2E-C | `e2e.py run`（llama、qwen × conversation、freeinference）＋`report` | E2E-C 第一段（舊常數那份，本登記時還在跑） |

## 4. 不重跑（照實列為限制）

- **X5**（B3 的 X5 部分、B5、B7 的 X5 部分）：成本常數來自 H200 實測的還原時間（`m10_precision_h200` 的格子，CPU 層是真的 H2D）與各平台自己定義的 b\* 慢層，不讀 tier_params → 不受影響（§2 自檢 3 確認）。
- **CPU 層在 GPU 上被限速成 35.4159 GiB/s 的 GPU 實測 run**：A1'、A1''、A2、A3（m7 的 `mk_tier("cpu", P["cpu"])`）；B6 與 C2 的 GPU 縮小重播（`x1_gpu.ChanTier` 用 X1 的常數排程）。這次不重跑；論文寫「這些 run 的 CPU 層以 35.4 GiB/s 限速」。B6 的「實測÷模擬」兩邊本來就用同一組常數，不重算。
- SSD 級、NFS 級（使用者選擇）；D 節（實測，CPU 層是真的 H2D）；X8、地圖（不在 ICPE_EVIDENCE）。

## 5. 判準與報告

- 每一列的判定字照它**原本的登記**：X1＝`docs/phase3_20261017/explore/` X1 §3 與 P0-1 §3.3；X3＝X3 §3；B4＝P1-2 §3「D」（D1、D2）；C2 只報告；E2E-C＝`docs/stage3_design_20261024/runs/E2E_combo.md` §4–§5。程式與門檻一律不改。
- 報告：`results/icpe_h200/t1/t1_compare.{csv,json}`（`code/icpe/t1.py report`）：每一列的舊值（MI300X 的 CPU 層）與新值（H200 的 CPU 層）並排、判定字有沒有變、數字差；跨門檻的格子逐格列出（README 規則 6）。
- **ICPE（單一平台 H200）用新值當主結果**；舊值寫成「CPU 層以 35.4 GiB/s 限速時」的敏感度。這一條在登記時決定，不看結果改。
- E2E-C：第二段（GPU 驗證）做不做，仍照 `E2E_combo.md` 的登記（舊常數那份）判；T1 版只報告。**兩份判定不同 → 停下來問使用者**，不自己挑。
- 任何一步失敗 → 停下、回報（CLAUDE.md 規則 2）；自檢不過 → 不跑。

## 6. 執行

- **compute node**（hgx-cn01；X3 與 X1 FreeInference 的資料在本地 `/tmp/hungweilai-p3/datasets`）：§3 的 X1（B1、B2、B7）、X1 FreeInference（B3）、X3（C1）、X4 掃描（B4）、fullsim（C2）→ 一個不量時間的 Slurm job（同 ICPE 的 `cpu`／`p12` part）；runner 是新檔 `code/r_h200_t1.sh`，**不改** `r_h200_icpe.sh`／`r_h200_p3.sh`。〔估計約 2 小時：原本的 cpu part 49 分、P3 explore 65 分〕
- **head node**：判定字、並排、`p12_queue.py`、`t1.py report`；E2E-C 的 T1 版等現在那份 E2E-C 跑完再跑（FreeInference 約 7–11 小時〔估計〕）。
- 送 job 前照 stage-3 的規則：`quota -s -g ChihChungHsu`、`sched.py next-icpe`、`squeue`；sbatch 用 `env -u TMPDIR -u CLAUDE_CODE_TMPDIR`。

## 7. 已經看過的數字（登記前）

- §1 的新常數本身。
- 舊版的全部結果（ICPE_EVIDENCE B、C）。
- E2E-C（舊常數）：**還沒有看過任何配置的數字**（FreeInference 還在跑；conversation 的 CSV 已在 `results/s3_h200/e2e/raw/`，沒有打開過）。
- **沒有任何 T1 的模擬數字**。

---

## 登記

- 登記時間（`date -u`）：2026-10-27T22:50:34Z
- 登記時 repo HEAD：`903d8da`（本檔隨下一個 commit 進 git）

## 偏離（登記之後）

### 偏離 1（2026-10-28T00:17:34Z；看過 X4 validate 的 V1 驗證數字，沒有看過任何 T1 的結果數字）：X4-VALIDATE 的 V1 在 T1 下不適用，不當成停止條件

- **發生什麼**：runner 照原本 explore 的順序，在 X4 sweep 前多跑了 trace 與 validate（登記 §3 的 B4 只列 sweep、analyze＋`p12_queue.py`）。job 22772 的 `T1-X4-VALIDATE`（run `20261028-081514-icpe-t1-x4-validate`）exit 1：V2 10/10 完全相同、V3 最大差 0、V4 最大差 0，但 **V1 54/160 格在 1% 內**（要 ≥95%；|err| 最大 11.2%）。
- **原因**（診斷，只報告、不進 results；scratch 腳本照抄 `x4_sim.cmd_validate` 的 V1 迴圈）：V1 拿 X4 的模擬器去比 `results/m7_explore_h200/r1_sim_llama31_8b.csv`，那份參考是用**舊的** CPU 層常數算的。沒設開關時 V1 **160/160**（10/21 原本也過；程式沒改）；設了開關時 54/160，各參考 CPU 頻寬組都差 6.7–11.2%——即使參考格指定了 CPU 讀頻寬，寫入頻寬與 k_read 仍換成新值。所以 V1 檢查的是「兩個模擬器的程式一致」，在舊常數下已成立，T1 下參考值本身就不同，不適用。
- **改什麼**：T1 不把 X4-VALIDATE 當停止條件；從 `T1-X4-SWEEP` 接著跑（sweep、analyze，同登記 §3）。validate 的輸出（`results/icpe_h200/t1/explore/x4/x4_validate_v*.csv`）照實保留、在報告裡列出。
- **影響**：判準不變；B4 的數字照登記算。

---

## 結果（2026-10-28；E2E-C 的 T1 版還沒跑——等登記的那份 E2E-C 跑完，登記 §6）

- **job**：22762（06:58–07:43，到 T1-X3-LEADS 失敗：tokenizer 路徑，RUNLOG ICPE-T1-FAIL-1）→ 22772（08:08–08:15，從 T1-X3-EVENTS；到 T1-X4-VALIDATE 停，偏離 1）→ 22788（08:17–08:19，從 T1-X4-SWEEP）。head node：`20261028-082005-icpe-t1-p12-queue`、報告 `20261028-082016-icpe-t1-report`。
- **run_id**（都 exit 0）：自檢 `20261028-065820-icpe-t1-selftest`（compute node 上再跑一次，PASS）；X1 V0 `…-065829-…-x1-sweep-V0`、`…-070955-…-an-V0`、`…-070959-…-h1800-V0`、`…-071003-…-h3600-V0`；V1 `…-071007`～`…-072136`；V2 `…-072140`～`…-073308`；fullsim `…-073312-icpe-t1-p02-fullsim`；X1 FreeInference `…-073351-…-p12-x1fi-sweep`、`…-074217-…-x1fi-an`、`…-074221-…-x1fiuser-an`；X3 `…-074241-…-x3-validate`、`…-074247-…-x3-tables`、`…-080851`～`…-080952-…-x3-leads-*`（7 個）、`…-081248-…-x3-hide`；X4 `…-081502-…-x4-trace`、`…-081741-…-x4-sweep`、`…-081922-…-x4-analyze`。失敗的 `…-074309-…-x3-leads-consts`、`…-081514-…-x4-validate` 見上。
- **產出**：`results/icpe_h200/t1/`（`p01/sim_V*/x1/`、`p12/x1/`、`p12/x4/`、`p02/`、`explore/x3/`、`explore/x4/`）；並排 `t1_compare.{csv,json}`。>1 MB 的 CSV 照舊 gzip。

### 判定字：全部沒有變

| 列 | 判準（原本的登記） | 舊（MI300X 的 CPU 層） | 新（H200 的 CPU 層） |
|:--|:--|:--|:--|
| B1／B2／B7 | X1 原登記、截尾 H＝1,800／3,600（rule＝min），V0／V1／V2 | 9 個都「有看頭」 | 9 個都「有看頭」 |
| B7 | P0-1 §3.3 (c)：V1、V2 的 X1 原登記、H3600、X5 的字和 V0 相同 | 成立 | **成立**（X5 不重跑，字沿用） |
| B3（X1） | X1 FreeInference（前 6 週、Cake） | 死路 | 死路（x1fi／x1fiuser 的 verdict 欄沒有變） |
| B4 | P1-2 §3「D」：D1（排隊占 p99 ≥0.90）、D2（多數格「還原變快」主導） | D1 成立、D2「平均值的空間主要來自還原變快」 | **同** |
| X4（B4 的來源） | X4 §3 的 30 個判定（metric × 群組） | 死路／有看頭／可能 | 30 個全部相同 |
| C1 | X3 §2.2／§2.4 | 原登記「有看頭」、加 FreeInference「有看頭」 | **同** |

### 數字（〔模擬〕；舊 → 新）

| 列 | 項目 | 舊 | 新 |
|:--|:--|--:|--:|
| B1 | conversation、SSD、GPU 閒、Cake，R1／R2 取小：O-PF（CPU 128／256 GiB） | 48.9／44.6% | **50.5／46.5%** |
| | O-ORD | 25.4／27.0% | **27.65／31.0%** |
| | O-DROP | 27.0／32.8% | **27.6／34.9%** |
| B2 | V0、rule＝min、384 格：不截尾 O-DROP 最大（≥15% 的格數） | 45.9%（48） | **47.7%（58）** |
| | 截尾 H＝3,600：O-DROP 最大（格數） | 0.0%（0） | 0.0%（0） |
| | 截尾 H＝3,600：O-PF、O-ORD 最大 | 91.8%、99.9% | 92.4%、99.9% |
| B3（X1） | FreeInference X1 最大（只算使用者發起） | 0.28%（1.48%） | **0.30%（1.52%）** |
| B4 | D1：排隊占 NP p99 的中位（最小；可判的格） | 0.983（0.846；262） | 0.983（0.837；264） |
| | D2：「還原變快」主導的格／平均 headroom ≥5% 的格 | 43／63 | 44／65 |
| | 只報告：QP 對 NP 的 p99 改善中位（最大） | 3.1%（25.9%） | 3.5%（28.1%） |
| B7 | X1（rule＝min，384 格）≥15% 的格數 V0／V1／V2 | 167／168／169 | **190／192／189** |
| C1 | 原登記：有看頭的格／主格子；NFS 級最好的格（bailian traceA、打字提示）時間加權藏住 | 36／36；98.74% | 36／36；98.74% |
| | 加 FreeInference：有看頭的格／主格子 | 37／54 | 37／54 |
| C2 | 全尺寸 24 格：HINT-PF 對 LRU；O-PF 對 LRU | 5.6–47.8%；5.5–52.4% | **6.9–49.1%；7.0–54.2%** |
| | HINT-PF÷O-PF 改善比例的中位 SSD／b32／NFS；>1 的格數 | 0.939／0.949／1.117；6 | **0.933／0.941／1.118；4** |

- **跨門檻的格子**（rule＝min，門檻 0.05、0.15；README 規則 6）：9 個 X1 檔共 **271 列，全部是往上跨**（新的 headroom 比舊的大）；逐格在 `t1_compare.json` 的 `crossings`。V0 原登記那份：34 列（O-DROP 14、O-ORD 15、O-PF 5）。
- **其他變化**（只記錄）：最好的對手換人（`best_rival`）X1 每檔 8–16 格、X4 cfg 26–43 格；X4 可判的格 262→264（`eligible` 2 格）、`pass5_mean` 4 格、`pass5_p90` 5 格；`p12_queue_cells` 63→65 列（1 列只在舊、3 列只在新）。
- **X4 validate**（偏離 1）：V1 54/160（參考值是舊常數算的，不適用）、V2 10/10、V3 0、V4 0；輸出在 `explore/x4/x4_validate_v*.csv`。
- **沒有重跑、照登記 §4 列為限制**：X5（B3 的 X5 部分 2.44%、B5、B7 的 X5 部分）；CPU 層在 GPU 上被限速成 35.4 GiB/s 的 GPU 實測 run（A1'、A1''、A2、A3、B6、C2 的 GPU 縮小重播）；SSD、NFS 級。
- **意思**：CPU 層換成 H200 自己的常數之後，ICPE B、C 節的模擬**判定字一個都沒有變**；X1 的 oracle headroom 普遍變大一點（跨門檻的 271 列全部往上），FreeInference 上 X1 仍然是 0.30%（使用者發起 1.52%）。照登記 §5，ICPE（只用 H200）改用新值；舊值是「CPU 層以 35.4 GiB/s 限速時」的敏感度。

### E2E-C 的 T1 版（2026-10-29；登記 §5：只報告）

- **run_id**（都 exit 0）：dcheck `20261028-160754-icpe-t1-e2e-dcheck`（可用組合 0 個，同登記版）；`20261028-160904-icpe-t1-e2e-llama-conv`、`…-160905-…-qwen-conv`（16:09 結束）、`…-160906-…-llama-fi`（10/29 02:16 結束）、`…-160907-…-qwen-fi`（10/29 01:05 結束）；報告 `20261029-021758-icpe-t1-e2e-report`；曲線 `20261029-021836-icpe-t1-e2e-curve`；並排 `20261029-021851-icpe-t1-report`。第一次送出的 `20261028-1608{22,23,24,26}-*` 是我在 zsh 下拆字串的錯（exit 127，沒有跑任何東西）。
- **產出**：`results/icpe_h200/t1/e2e/`（`raw/` 四個 CSV 的列數和登記版相同：177／145／2,817／2,305 行，和登記版一樣不進 git〔`.gitignore` 的 `results/**/raw/`〕；`e2e_report.json`；`mps_tradeoff.{csv,png}`）；`t1_compare.{csv,json}` 補上 E2E。
- **判定：和登記版相同**：E2E-NOGO（0／32 格過、ASK 0、NOT_MEASURED 8＝Qwen × cprefill），四組（模型, 背景）的 m\* 都是空的 → 兩份判定相同，不用問使用者（登記 §5）。
- 並排：除了 E2E 那幾項從 NOT_MEASURED 變成有值，`t1_compare` 其他列和 10/28 那份相同（跨門檻 271、字的變化 317，都是 B、C 節原本就有的）。`e2e_report.json` 1,381 個值裡 699 個數字有變（字沒有變）。
- **MPS 取捨〔模擬〕（只報告；測試集，每組 4 格的中位）**：不設上限的 MPS 對 K2（stream）的 TTFT 改善 30.7–62.9%（登記版 32.9–63.0%），背景 B 比 stream 少 0～6.5 個點（登記版 0～6.7）；28 個點裡 TTFT 改善最多差 4.0 個百分點、B 最多差 0.002。
