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

（無）
