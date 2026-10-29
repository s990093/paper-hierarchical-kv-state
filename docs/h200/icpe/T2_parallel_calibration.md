# T2：平行爭用校準——我們自己的單卡量時間 job 在不同卡上同時跑，時間會不會變

> **性質**：判準登記（事先寫死）。登記時間見文末「登記」一節（`date -u`）。登記之後這一節以上不改；和登記不同的做法寫成文末的「偏離 N」。
> **來源**：使用者 2026-10-29「要不要跑平行爭用的校準？……這個計劃優先做」「改一張卡大量送出」「任務能送就送」。
> **為什麼有這一關**：目前的規則是量時間的 job 不能和我們自己的任何 job 同時在節點上（memory `h200-no-self-cotenancy-for-timing`；10/16 F2 的 H2D 讓 P1 r36 的 S5 慢 8.3%）。這條規則讓 GPU 工作只能一個接一個跑。這一關量「同時跑」到底會不會改變我們會用到的兩種量測；會的話照舊依序跑，不會的話之後那兩種 job 可以平行送。
> **範圍**：只有 H200（hgx-cn01）、只有 Llama-3.1-8B。只校準下面兩種量測；**MPS（有背景 process）的 job 不在範圍內**，照舊依序跑。

---

## 1. 探針（每個 job 跑同一份，都是既有的程式，不改）

每個 job 是一張卡（`--gres=gpu:1`、16 核、170G，和 T1G 相同），依序跑兩個部件：

| 部件 | 指令 | 代表什麼 | 指標（每個 job 一個值） |
|:--|:--|:--|:--|
| **mb**（還原微基準） | `b0.py run --model llama --setup b0 --modes stream --bgs idle --prims h2d_perblock,h2d_batch,recompute1,e2e_cake_bf16_cpu,e2e_cake_int4_cpu,e2e_cake_int4_b32,e2e_load_int4_b32 --reps 6 --seed 1` | 吃 PCIe／主機記憶體的還原、重算（之後長上下文的組合驗證就是這一類） | 每個原語：6 次 `t_med_s` 的中位數（`own_gpu_intruders`＞0 的列不用） |
| **rp**（GPU 重播） | `x1_gpu.py gpu --k 8 --cpu 128 --policy HINT-PF --max-req 600`（環境同 T1G 的 P02：T1 常數） | 縮小版的 GPU 重播（B6／C2、之後 E2E 的 GPU 驗證就是這一類） | 回來請求（`kind=1`）的平均 TTFT；`contaminated`＝1 → 那個 job 的 rp 不用 |

## 2. 條件與順序

- **solo**：一次只有一個我們的 job。開跑前等節點上沒有我們自己的別的 job（`t1g.py wait-clear`）。
- **par**：一「波」W＝3 個 job 同時送；每個 job 開跑前等同一波的其他 job 也在跑（`calib.py barrier`，最多等 30 分鐘，逾時照跑、記下來），再確認節點上沒有這一波以外的我們的 job。W＝3：節點 8 張卡，登記時別人佔 5 張；lab 帳號上限 4 張。
- **順序**（ABAB，抵銷時間漂移）：S1→S2 → 波 1（3 個）→ S3→S4 → 波 2（3 個）→ S5→S6 → 波 3（3 個）。solo 6 個、par 9 個，共 15 個 job。用 Slurm `afterok` 串起來：任何一個失敗，後面不跑（規則 2）。
- kv-cliffs 的 kc-* job 在這段時間暫停（已通知，對方 10:21 確認）。

## 3. 每個部件屬於哪個條件（用實際時間判，不用送出時的標籤）

- 每個部件跑完寫一列 `calib_windows.csv`：job、波、條件標籤、部件、開始／結束時間、卡的 index。
- 另用 `sacct` 查這個時間窗裡節點上**不屬於校準**的我們自己的 job；有 → 這個部件**不用**（照實列出）。sacct 查不到 → 那一列 NOT_MEASURED（規則 7）。
- 對每個部件窗 w：每個別的校準 job j 的覆蓋率 c_j＝|w ∩（j 的 mb 窗 ∪ j 的 rp 窗）|／|w|。
  - **solo**：所有 c_j＝0。
  - **par(k)**：k＝c_j ≥ 0.9 的 job 數 ≥ 1，而且沒有 0＜c_j＜0.9 的 job。
  - 其他（部分重疊）→ **partial**，不進判定，只報告。

## 4. 判準（每個指標分開判）

- S＝solo 的值，P＝par（k ≥ 1）的值。n_S ≥ 6 且 n_P ≥ 6，否則那個指標 NOT_MEASURED。
- σ＝S 的標準差 ÷ S 的中位數；**δ\*＝max(5%, 2σ)**（05 §2.1 的規則）。**2σ ＞ 5% → 「太晃」，只報告、不判**。
- 變慢 d＝median(P)／median(S) − 1；95% CI：S、P 各自對 job 重抽 10,000 次（seed 20261029），取 d 的 2.5／97.5 百分位。
- **OK（不變慢）**：d ≤ δ\* 且 CI 上界 ≤ δ\*。**SLOW**：d ＞ δ\* 且 CI 下界 ＞ 0。其他 → **INCONCLUSIVE**。
- **一種探針 OK**：mb 的 7 個原語裡可判的都 OK（太晃的不算，但可判的至少 4 個）；rp 的指標 OK。
- **結論**：
  - **CAL-OK**：mb、rp 都 OK → 之後這兩類量時間的 job 可以平行送，同時最多 1＋floor(median k of P) 個（k 是 P 裡實際的同時數）。
  - **CAL-PARTIAL**：只有一種 OK → 只有那一種可以平行。
  - **CAL-NO**：都不 OK（任何 SLOW 或 INCONCLUSIVE）→ 照舊依序跑。
  - 任何一種 NOT_MEASURED → 那一種照舊依序跑。
- **只報告**（不判定）：按 k 分開的 d；共跑的卡和自己在同一個／不同 NUMA 節點（用 mb 列的 `gpu_numa`）的 d；solo 每張卡的中位數；partial 的部件；窗裡別的使用者的 job 開始／結束次數（`sacct -a`，窗前後各 120 s，只記數目、不記名字）；`host_level`。

## 5. 程式與輸出

- `code/icpe/calib.py`（`barrier`、`window`、`analyze`、`self-test`）；runner `code/r_h200_calib.sh`；送出 `code/icpe/calib_submit.sh`。
- 輸出 `results/icpe_h200/calib/`：`raw/mb_<job>.csv`、`raw/rp_<job>.csv`、`raw/gpu_guard_*.json`、`calib_windows.csv`、`calib_report.json`、`calib_metrics.csv`。
- 分析：`source code/env_h200.sh; m7run icpe-cal-analyze python code/icpe/calib.py analyze`（head node）。

## 6. 已經看過的數字（登記前）

- 10/16 F2 的 H2D 讓 P1 r36 的 S5 慢 8.3%（memory）；T1G 附加 1 的「別的使用者的 job」結果（A2 chat 那一步暴露的重複慢 8.0%，其他 ±0.1%）。
- B0、P02、T1G 的全部結果（solo 的情況）。
- **沒有看過任何「我們自己的 job 同時跑」的時間數字**。

---

## 登記

- 登記時間（`date -u`）：2026-10-29T02:28:14Z
- 登記時 repo HEAD：`599da04`（本檔隨下一個 commit 進 git）

## 偏離（登記之後）
