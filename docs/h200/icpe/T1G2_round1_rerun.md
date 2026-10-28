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
