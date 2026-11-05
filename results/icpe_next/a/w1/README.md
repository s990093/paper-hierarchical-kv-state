# W1：Mooncake 的 CPU 層命中率（定義照 T4 的 HR）

> NEXT_PLAN_20261104 W1。**只用現有輸出、只報告**，不跑新模擬（唯一的計算是紀錄前處理，算每個請求的 U）。
> **run_id**：`20261104-231951-icpe-next-a-w1-hitrate`（head node；指令、環境、紀錄檔 sha256 在 `/work/ChihChungHsu/hungweilai/tiara-runs/<run_id>/`）。第一次執行 `20261104-231847-icpe-next-a-w1-hitrate` 還沒有摘要表，兩次的命中率 CSV 除了 `run_id`／`ts` 逐字相同。
> **程式**：`code/icpe/next_a_w1_hitrate.py`（定義與自檢先 commit 在 `e260598`，之後才執行；之後只加了摘要表）。

## 定義

- HR＝Σ_ret C ÷ Σ_ret U（T4 §2）。U＝前面連續幾個 block 在這個請求之前被讀過（只由紀錄決定）；C＝到達時有存的前綴裡在 CPU 層的 block 數。ret＝回來的請求（kind＝cont）。
- **C 只能從 `restore=load` 的列拿**：load 模式逐塊載入整段有存的前綴，`ret_cpu_loads`＝C；cake 模式只載入後半段，`ret_cpu_loads` ≤ C。ΣU 不在 X1 的任何欄位，要從紀錄算（`t4.reusable`）。
- 預取策略（O-PF、LRU+RPF、O-ALL*）一個 job 只跑一種時間設定，表裡是 load 那一輪的 HR；cake 那一輪＝NOT_MEASURED（`hr_scope=load_run_only`）。
- 來源：Mooncake＝`results/icpe_h200/t1/p01/sim_V1/x1/x1_sweep.csv.gz`（run `20261028-071007-icpe-t1-x1-sweep-V1`）；FreeInference＝`results/icpe_h200/t4/t4_sweep.csv`（run `20261030-100335-icpe-t4-sweep`，6 週合併 ΣC÷ΣU，同 T4）。

## 自檢（全過）

- K1：T4 的 720 個 load 列，`C_ret`＝`ret_cpu_loads`、`S_ret`＝`ret_slow_loads` 全部相同；720 個 cake 列 `ret_cpu_loads` < `C_ret` 全部成立 → 「load 列的 `ret_cpu_loads`＝C」在 T4 自己的資料上成立。
- K2：紀錄前處理的回來請求數、總請求數和 sweep 相同（4 個 trace×rule）。
- K3：2,208 個 load 列，`ret_cpu_loads`＋`ret_slow_loads`＋`ret_recompute`＝Σ_ret n_all 逐列相同，且有存的 ≤ ΣU。
- 只報告：非預取策略的 C 在 12 種時間設定間只有 GDSF 會變（GDSF 的成本用慢層等級，按慢層分 job）；T4 的 C 全部不變。

## 結果（主設定：SSD 級、GPU 閒、回來的請求；〔模擬〕，全表見 `w1_summary_ssd_idle.csv`、`w1_hitrate_mooncake.csv`、`w1_hitrate_fi_t4.csv`）

**LRU 的 CPU 層命中率**（LRU 和 LRU+NONE 的 CPU 層命中率逐格相同）：

| 紀錄 | 8 GiB | 16 | 32 | 64 | 128 | 256 | 512 | 1,024 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Mooncake conversation R1 | — | — | — | — | 7.86% | 17.03% | 45.63% | 70.53% |
| Mooncake conversation R2 | — | — | — | — | 5.87% | 13.40% | 40.45% | 65.79% |
| Mooncake toolagent R1 | — | — | — | — | 41.49% | 49.33% | 69.25% | 86.30% |
| Mooncake toolagent R2 | — | — | — | — | 8.86% | 19.63% | 48.19% | 74.87% |
| FreeInference（6 週） | 13.19% | 46.61% | 82.02% | 97.59% | **99.29%** | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |

（FreeInference 256–1,024 GiB：T1 的 `x1_sweep_fi` 只有 cake 列，拿不到 C。Mooncake 8–64 GiB 沒有跑過。）

**O-BEL（oracle 順序）的 CPU 層命中率**：Mooncake 128 GiB 時 conversation 61.65%／56.62%（R1／R2）、toolagent 78.24%／62.34%；FreeInference 128 GiB 99.36%。→ 128 GiB 時 oracle 順序比 LRU 多 **36.7–53.8 個百分點**（Mooncake），FreeInference 只多 0.07 個百分點〔算術〕。

**固定規則裡命中率最高的**（LRU、LRU2、ARC、S3FIFO、GDSF、PENS、FREQ2Q）：Mooncake 128 GiB 時 20.19–51.48%（conversation R1 是 LRU2 24.27%）；仍比 O-BEL 低 26.8–38.9 個百分點〔算術〕。

**CPU 容量佔工作集（全部不同 block，BF16）的比例**：Mooncake 128 GiB＝1.20%（conversation）、1.28%（toolagent）；FreeInference 各週的工作集 1.6–16.3 TiB，所以 128 GiB＝0.78–8.06%、16 GiB＝0.10–1.01%〔算術〕。

**有慢層時「CPU＋慢層」的命中率**：Mooncake LRU 每格 100%；FreeInference 99.42%（同 T4）。

## 對照 W1 的事先登記

- 登記（NEXT_PLAN W1）：「預期 Mooncake 的 LRU 命中率明顯低於 FreeInference（假設，未量）。若不低，『差距取決於固定規則命中多少』這個解釋就要拿掉。」
- **結果**：同為 CPU 128 GiB，Mooncake 的 LRU 命中率 5.87–41.49%，FreeInference 99.29% → **比 FreeInference 低**，這個解釋**不用拿掉**。
- 〔判讀〕「容量佔全部工作集的比例」解釋不了差距：FreeInference 在 16 GiB（工作集的 0.10–1.01%）時 LRU 還有 46.61%，Mooncake conversation 在 128 GiB（1.20%）只有 7.86%。差距比較像來自重用的型態（FreeInference 兩輪間隔中位數 1.9 s，`trace_stats.csv`），但這裡**沒有驗證因果**，只是兩組數字並排。
- 限制：Mooncake 和 FreeInference 的 session 是不同方式重建的（R1／R2 vs 資料自帶的 `session_id`）；兩份紀錄的 CPU 容量網格只在 128 GiB 重疊。
