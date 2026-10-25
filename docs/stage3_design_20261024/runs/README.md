# 第三階段的 run 文件（H200）

> 每個閘一個 md：**最上面是事先登記的判準與量法（commit＋`date -u` 之後不改）**，結果與偏離附在後面。
> 規則照 `docs/h200/icpe/README.md`：登記在看到任何正式數字之前 commit；冒煙（smoke）只驗程式、不進 `results/`、不看分數；
> 和登記不同的地方一律寫成「偏離 N」附在文件後面，說明為什麼、在看到哪些數字之前或之後。

| 閘 | 文件 | 程式 | 結果 | 狀態 |
|:--|:--|:--|:--|:--|
| A0（＝G-M01/02） | [A0_progressive_restore.md](A0_progressive_restore.md) | `code/s3/a0.py`、`code/s3/a0_gpu.py` | `results/s3_h200/a0/` | 已登記，待冒煙 |
| B0＋G-M08（量時間） | [B0_interference.md](B0_interference.md) | `code/s3/b0.py`、`code/s3/bgengine.py` | `results/s3_h200/b0/` | 已登記，待冒煙 |
| G-M04（品質，併在 A0 TF） | [G_M04_prefix_select.md](G_M04_prefix_select.md) | `code/s3/a0_gpu.py --gm04` | `results/s3_h200/a0/raw/gm04_*` | 已登記 |
| C0（CPU） | [C0_cost_model.md](C0_cost_model.md) | `code/s3/c0.py` | `results/s3_h200/c0/` | 已登記 |
| W0（量時間） | [W0_write_cost.md](W0_write_cost.md) | `code/s3/w0.py` | `results/s3_h200/w0/` | 已登記 |
| G-fio（量時間） | [G_fio.md](G_fio.md) | `code/s3/gfio.py` | `results/s3_h200/gfio/` | 已登記 |

runner：`code/r_h200_s3.sh`；RUNLOG：`results/RUNLOG_H200_S3.md`；GPU 排程規則見 memory `stage3-gpu-scheduling`。

---

## 共同規則：速度判準的「雜訊」（2026-10-25 補；prereg deviation）

操作定義寫在 [05 §2.1](../05_poc_plan_h200.md)。所有**量時間**的 run 的判準段都引用它，摘要如下：

- σ_noise＝baseline 配置在同一個 Slurm allocation、同一張卡、同一輪隨機順序裡重複 ≥6 次的相對標準差（σ÷中位數），每個量測點各自算、不跨點共用。
- δ\*＝max(5%, 2×σ_noise)。「有改進」：中位數差 > δ\* 且差的 95% CI（對 rep 重抽 10,000 次）下界 > 0。「沒有變慢」：變慢 ≤ δ\* 且變慢的 95% CI 上界 ≤ δ\*。點估計過、CI 沒過 → INCONCLUSIVE（要加重複次數）。P95 同規則，σ 用 P95 的 bootstrap 標準誤。
- CSV 每列有 `n_rep`、`sigma_rel`、`delta_star`（P95：`p95_sigma_rel`、`p95_delta_star`）；量不到寫 NOT_MEASURED。
- 2σ > 5% 的點＝量測太晃：先回報，不拿來判定。
- 品質判準（A0、G-M04）不變，不套用 σ 規則。

**為什麼是 deviation**：05 §2 原本只寫「δ\*＝max(5%, H200 上量到的雜訊)」，沒有定義雜訊怎麼量；B0 的登記（第 66 行）直接取 δ\*＝5%。

**受影響的 run（逐一）**

| run | 有沒有看過時間數字 | 處理 |
|:--|:--|:--|
| B0＋G-M08 | **沒有**（冒煙 22238 在量任何還原之前就失敗；冒煙 22242 進行中，輸出在 run 目錄、不進 results、不判定）。看過的只有 P0b 背景自檢的單獨速度（bgcheck：每步毫秒），判準不用它 | **改用新規則**（B0 檔的偏離 1） |
| W0 | 沒有（還沒跑） | **改用新規則**（W0 檔的偏離 1） |
| G-fio | 沒有（還沒跑） | 沒有判定，只換常數；加 6 次重複與三個欄位（G-fio 檔的偏離 1） |
| C0-v1 | **有**（10/25 已出 GC-FAIL） | **維持原判準**；另寫一行「事後以 2σ 重算的結果」只供參考（C0 檔） |
| C0-v2（之後同程式重跑） | 判準在 v1 已經看過 | 維持原判準（和 v1 相同）；同樣附 2σ 參考行 |
| G-M13 | **有**（10/25 已出 GM13-PASS；MI300X 的 I3g 重播） | **維持原判準**；另寫一行「事後以 2σ 重算的結果」只供參考（G-M13 檔） |
| A0、G-M04 | 品質 | 不變 |
| P0e | 沒有時間判定（能／不能） | 不適用 |
| 之後的量時間的閘（G-M06、G-M11、G-M12 速度、A2…） | 還沒登記 | 登記時直接照 05 §2.1 |
