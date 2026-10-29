# T1G-2：A3（第 1 輪模擬存活設定的 GPU 確認）在 H200 的 CPU 層常數下重做

> **性質**：判準登記（事先寫死）。登記時間見文末「登記」一節（`date -u`）。登記之後這一節以上不改；和登記不同的做法寫成文末的「偏離 N」。
> **來源**：使用者 2026-10-28「兩批都排，T1 先」。T1G（[`T1G_gpu_rerun.md`](T1G_gpu_rerun.md)）§6 把 A3 排到這一份：A3 的格子是用舊常數（CPU 層 35.4 GiB/s）的第 1 輪模擬選出來的，換常數要先重選。
> **範圍**：只有 H200、只有 Llama-3.1-8B（A3 的 H200 列；LongAlpaca 照原本的追加 4 不上 GPU）。先 CPU（head node）重選，再 GPU 量時間。

---

## 1. 步驟

1. **重跑第 1 輪模擬（CPU，head node）**：和原本同一支程式、同樣的參數，只換 CPU 層常數與輸出目錄：
   `M7_TIER_PARAMS=results/icpe_h200/t1/tier_params_h200cpu.json M7_EXPLORE=results/icpe_h200/t1/gpu/a3 python code/m7_round1_sim.py --model llama31_8b`
   （原本 run `20261016-095639-m7h-r1-sim-llama31-8b`；f(i)＝H200 的 C1 `results/m7_write_policy_h200/calib_c1.csv`，同原本）。
   - 格子裡 `cpu_gibps=None` 的那一欄＝tier_params 的 CPU 層，從 35.42 變成 **43.59**；11.6、3.69、1.0 三欄是覆寫值，只有 k_read（1.161 → 1.002）不同。
2. **選存活設定**：`M7_EXPLORE=…/t1/gpu/a3 python code/m7_round1_analyze.py sim`（原本 run `20261016-173455-m7h-r1-sim-an`）→ `r1_sim_configs.csv` 的 `survive`（≥3/5 seed 同時贏「最好的不看位置」、S4B、S4W、S4C 各 ≥5%）。規則、程式都和原本相同。
3. **每個存活設定選一個寫入時決定的策略**：`r1_sim_verdict.csv` 的 `best_wt` 在 5 個 seed 裡出現最多的；平手取原本 A／B／C 用過的（S5L、S5c），再平手取字母序第一個。
4. **GPU 確認（量時間）**：每個存活設定照原本 `code/r_h200_r1h.sh` 的做法：對照組 S1、S2b、S4、S4+、S4+P、S4L、S4B、S4W、S4C＋第 3 步選的策略；workload doc、seed 0–4、每個 3 次、`--verify`、`M7_IO_MODEL=share`、f(i) 同上；CPU 頻寬是覆寫值的設定設 `M7_CPU_GIBPS`，43.59 的設定不設（讀 T1 的 tier_params）；`M7_TIER_PARAMS`＝T1 的檔。輸出 `results/icpe_h200/t1/gpu/a3/r1h_gpu_llama_<設定>.csv`。runner 由第 2、3 步的輸出機械地產生（`code/icpe/t1g.py a3-plan`），排程規則同 T1G §4（等空檔、查重疊、重疊就重跑最多 3 次）。job 接在 T1G 第一批（job 22882）之後（`afterany`）。

## 2. 判準

- **每個存活設定**（同原本）：寫入時決定策略的回來請求 TTFT 中位數，同時贏「最好的不看位置」、S4B、S4W、S4C 各 ≥5% 的 seed 數 ≥3/5 → 存活（≥15% 算強）。程式＝`code/m7_round1_analyze.py gpu --csv … --cpu … --release … --ssd … --frac … --wt … --tag …`。
- **A3 的判定字**：H200 的 Llama 存活設定在 GPU 上存活的個數（原本：3 格 0/3）。
- **模擬沒有存活設定**：A3 寫成「T1 常數下第 1 輪模擬沒有存活設定」，不上 GPU。
- 判定字和原本不同（GPU 上有存活）→ 先停下來回報使用者。
- MI300X 選出來的那一格（11.6／free／doc／local／0.5，原本 P3 0/5）不重量：ICPE 只用 H200，那一格不是 H200 的存活設定（原本在 H200 模擬也是 0/5）。如果它在第 2 步變成 H200 的存活設定，就照第 4 步一起量。

## 3. 已經看過的數字（登記前）

- 原本 A3 的全部結果（H200 3 格 0/3、MI300X 那格 0/5）、原本 r1_sim 的存活設定。
- T1、T1G 登記前看過的（T1G §7）。
- **沒有看過 T1 常數下的第 1 輪模擬**（第 1 步還沒跑）。

---

## 登記

- 登記時間（`date -u`）：2026-10-28T08:30:41Z
- 登記時 repo HEAD：`8ad833d`（本檔隨下一個 commit 進 git）

## 偏離（登記之後）

- 2026-10-28T15:08:47Z：別的使用者的 job 照 T1G 附加 1 處理（`docs/h200/icpe/T1G_gpu_rerun.md`），只報告，不改判定。

---

## 結果（2026-10-29）

- **第 1–3 步（CPU，2026-10-28）**：見 RUNLOG_H200_ICPE「ICPE-T1G2-PREP」。T1 常數下的存活設定〔模擬〕是 2 個：11.6／hold／doc／nfs／0.5（best_wt S5L）、43.59／hold／doc／nfs／0.25（S5c）。原本的 35.42／free／local／0.5 不再存活。MI300X 那格沒有變成 H200 的存活設定，所以不量（§2）。
- **第 4 步（GPU）**：job 22886，2026-10-29 02:56:50 → 09:15:29，COMPLETED。
  - 10 步（2 格 × seed 0–4，每步 10 個策略 × 3 次）全部 exit 0。
  - 每一步開跑時節點上都沒有我們自己的別的 job（`wait-clear` 都等 0 次）；`overlap.csv` 10 列全部「重疊：無」；沒有重跑；gpu_guard 都沒有判污染。
- **分析**：`20261029-091732-icpe-t1g-analyze-a3`（`t1g.py analyze --part a3` → `m7_round1_analyze.py gpu`）。

### 判定字：沒有變（GPU 上沒有存活設定）

| 格 | 原本（舊常數，H200） | 新（T1 常數） |
|:--|:--|:--|
| 11.6／hold／doc／nfs／0.5（S5L） | 原本 A：2/5 seed 過（seed 0、2）→ 不存活 | **2/5**（seed 0、2）→ **不存活** |
| 35.42 → 43.59／hold／doc／nfs／0.25（S5c） | 原本 C：2/5（seed 1、3）→ 不存活 | **2/5**（seed 1、3）→ **不存活** |
| 35.42／free／doc／local／0.5 | 原本 B：0/5 → 不存活 | T1 常數下模擬就不存活，不量 |
| **A3 的判定字** | **0/3** | **0/2** |

### 數字（實測；回來請求 TTFT 中位數，寫入時決定的策略對「最好的不看位置」，正＝寫入時決定快）

- 11.6／0.5（S5L）：seed 0–4 是 +15.5／−26.8／+24.5／−27.2／−29.3%（原本 +17.1／−23.8／+24.5／−19.2／−22.1%）。輸的三個 seed，最好的不看位置都是 S1。
- 43.59／0.25（S5c）：−4.7／+7.8／−4.5／+7.2／−102.0%（原本 35.42：−4.4／+7.8／−4.5／+7.2／−98.8%）。
- 每個 seed 過不過、最好的不看位置是哪個策略，新舊幾乎都一樣（只有 11.6 的 seed 1 從 S4 變 S1、43.59 的 seed 1 從 S2b 變 S4+）。
- 模擬對 GPU：|誤差| ≤10% 的格 46/50（11.6）、44/50（43.59），中位誤差 0.010、0.005。

### 附加 1（別的使用者的 job；只報告）

- scan `20261029-091805-icpe-t1g-exposure-scan2`：28 步（第一批 18 步＋A3 10 步），sacct 都查得到。第一批 18 步的列和上次的 scan 逐值相同（只差 run_id、ts）。
- A3 的 10 步裡只有 43.59 的 s0（2 個事件）、s2（7 個事件）有別的 job 開始或結束；都不是同 lab 帳號的 job。
- 暴露單位 11/300，全在那兩步。
- 影子目錄的回歸（什麼都不拿掉）：逐值相同（`20261029-091816-icpe-t1g-exposure-a3-regress`）。
- 拿掉暴露單位（`20261029-091830-icpe-t1g-exposure-a3`）：沒有任何判定字不同。
  - A3 每點只有 3 次，有拿掉的點都不到 6 次 → NOT_MEASURED。
  - 所以 43.59 的 seed 0、2 是 NM，剩 3 個 seed 裡 2 個過。這一格的存活字在附加 1 下是 NM（缺的兩個 seed 都過的話會到 4/5）。11.6 那格沒有暴露單位，仍然 2/5。
- 同一點裡暴露的重複對沒暴露的重複（偏離 1）：
  - s0：2 點，中位 −1.6%。
  - s2：6 點，中位 −0.3%（四分位 −4.4～−0.1%）。
  - 8 點裡 0 點變慢，也就是暴露的重複沒有比較慢。
- **產出**：`results/icpe_h200/t1/gpu/a3/`（原始列 `r1h_gpu_llama_doc_<格>.csv` 都 < 1 MB，不用壓；`r1_gpu_{summary,verdict}_<格>.csv`；`analysis/verdict.json`）、`overlap.csv`、`t1g_compare.json`、`exposure/`。
  - `*_valid.csv` 是 `t1g.py analyze` 篩過的副本，不進 git。
  - `b_workload_seed<k>.json` 被第二格覆寫。兩格同一個 seed 的工作負載在各自的 stdout 裡逐字相同，所以沒有差別。
- **意思**：CPU 層換成 H200 自己的常數之後，第 1 輪模擬的存活設定從 3 個變 2 個。GPU 上兩格都只有 2/5 seed 過，和原本同樣的 seed 過、同樣的 seed 不過。**A3 的判定字（GPU 上沒有存活設定）不變**。T1G 兩批全部的判定字都和原本相同。
