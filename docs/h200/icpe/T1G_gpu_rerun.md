# T1G：ICPE 只用 H200——CPU 層換成 H200 的常數，重量 GPU 實測的 run（第一批：A1''、A1'、A2、B6／C2）

> **性質**：判準登記（事先寫死）。登記時間見文末「登記」一節（`date -u`）。登記之後這一節以上不改；和登記不同的做法寫成文末的「偏離 N」。
> **來源**：使用者 2026-10-28：GPU 不要空著，「兩批都排，T1 先」（第一批＝T1 的 GPU 補量，第二批＝A4）；和 kv-cliffs（kc-\*）的 job 怎麼共存：「照規則，等 kc-\* 空檔」。
> **為什麼有這一關**：T1（[`T1_h200_cpu_tier.md`](T1_h200_cpu_tier.md)）把模擬的 CPU 層換成 H200 自己量的常數，但 §4 把「CPU 層在 GPU 上被限速成 35.4159 GiB/s 的 GPU 實測 run」列為限制、沒有重量。ICPE 只用 H200，這些 run 的 CPU 層應該用 H200 的 43.5888 GiB/s。
> **範圍**：只有 H200（hgx-cn01）。量時間。A3 和 A4 不在這一批（§6）。

---

## 1. 重量什麼

| 列 | 原本的 run | 這次的步驟（runner `code/r_h200_t1g.sh`） | 輸出 | 原本花的時間 |
|:--|:--|:--|:--|--:|
| A1''（B 主設定、fa3） | `20261022-210344-icpe-p01a-b-nfs-gap0`（job 21654） | `T1G-A1PP`：`m7_write_policy.py b --reps 6 --ssd-dev nfs --gap 0 --wl-seed 0 --verify`，`M7_ATTN=fa3`、`M7_F_CSV=results/icpe_h200/p01/calib/calib_c1.csv`、`M7_IO_MODEL=share` | `results/icpe_h200/t1/gpu/a1pp/` | 98 分 |
| A1'（同上、sdpa） | `20261016-063929-m7h-b-nfs-gap0`（rep 0–2）＋`20261016-172445-m7h-b-nfs-gap0-r36-rr`（rep 3–5） | `T1G-A1P`：同一條指令、不設 `M7_ATTN`（sdpa）、`M7_F_CSV=results/m7_write_policy_h200/calib_c1.csv` | `…/t1/gpu/a1p/` | 67＋20 分 |
| A2（08 消融） | `20261016-202603-m7h-b2-g1-doc-hold-cpu3.69-s0-rr`、`…-132601-…-s1`、`…-142019-m7h-b2-g2-chat-hold-s0` | `T1G-A2-G1-s0`、`-s1`、`T1G-A2-G2-s0`：原本的三條指令（策略、cpu_frac、`--reps 3 --verify`、`M7_CPU_GIBPS=3.69`〔G1〕都相同） | `…/t1/gpu/a2/` | 83＋54＋38 分 |
| B6、C2 的 GPU 縮小重播 | job 21655（冒煙＋12 個 run） | `T1G-P02-SMOKE`＋12 個 `T1G-P02-<cpu>-<做法>-r<rep>`：`x1_gpu.py gpu --k 8`，順序和原本相同（seed 20261023） | `…/t1/gpu/p02/` | 3 時 18 分 |

合計照原本的時間約 10 小時（不含等空檔）。一個 job 依序跑完（`--part all`；順序 A1'' → A1' → A2 → P02）。

## 2. 改了什麼（其他全部和原本相同）

- **CPU 層常數**：`M7_TIER_PARAMS=results/icpe_h200/t1/tier_params_h200cpu.json`（T1 §1：讀 43.5888、寫 41.1687 GiB/s、k_read 1.002；SSD／NFS 級逐位元照抄 MI300X 的檔）。P02 另設 `P3_TIER=h200cpu`（`x1_sim.constants()` 的 CPU 層，`x1_gpu.ChanTier` 用它排程）。
- **輸出目錄**：`M7_RES` 改到上表的目錄；f(i) 用 `M7_F_CSV` 明確指回原本那份（`m7_write_policy.py b` 只從 `M7_RES` 讀 f(i)，不讀 busy_levels）。原本的結果檔一律不碰。
- **A2 的 G1（`M7_CPU_GIBPS=3.69`）**：CPU 層的讀寫頻寬原本就被覆寫成 3.69，這次只有 `k_read`（share 模式的同時讀寫因子）從 1.161 變成 1.002；G2（chat）三個值都換。
- **A1' 改成一次 6 rep**：原本是 rep 0–2 一個 run、rep 3–5 另一個 run（後者因為重疊重跑，只跑 cpu 0.25／0.5 與 7 個策略）。這次和 A1'' 同一個寫法（`--reps 6`，全部 cpu_frac 與策略）。
- **B6／C2 的 k 固定 8**：不在 T1 常數下重選 k（`x1_gpu.py select`），因為重選可能換掉抽樣的 session、就不是同一個縮小設定。CPU 容量 16／32 GiB（128／256 ÷ 8）不變。
- **P02 的 CPU 前置**（head node，在 GPU 之前）：
  - selftest（T1 常數下）：run `20261028-162428-icpe-t1g-p02-selftest`，9/9 PASS。
  - S_orig：run `20261028-162505-icpe-t1g-p02-sim-orig` → `…/t1/gpu/p02/sim_requests.csv`。
  - S_real 在 GPU 之後，用 GPU run 的實際開始時間（同原本）。

## 3. 判準與報告

- 每一列的判定字照它**原本的登記**，用原本的分析程式（只換輸入、輸出目錄）：
  - A1''、A1'：P0-1 §3.1——每個 cpu_frac，S5 對 S4+（回來請求 TTFT 中位數，rep 0–5 合併）≤ +δ\*，δ\*＝max(5%, C4 的雜訊)；C4 用原本那份（A1''：`results/icpe_h200/p01/calib/c4_noise.csv`；A1'：`results/m7_write_policy_h200/c4_noise.csv`）；三個 cpu_frac 都過、污染列 0 → 成立。程式＝`m7_repro_compare.analyze`（同 `p01_tools.py analyze-a`）。
  - A2：08 消融的判準（`m7_b2_analyze.py`：寫入時決定同時贏最好的不看位置與 S4B 各 ≥5% 的 GPU 列數）。
  - B6：P0-2 §8 的三條（回來請求平均 TTFT 實測÷S_orig 全在 [0.9, 1.1]；S_orig 差 ≥5% 的配對排序相同；headroom 正負號相同）。程式＝`code/icpe/p02_analyze.py --dir results/icpe_h200/t1/gpu/p02`。
  - C2 的 GPU 部分：只報告（HINT-PF 拿到 O-PF 改善的比例）。
- **報告**：每列的判定字（舊 → 新）、關鍵數字（舊 → 新）。照 T1 §5：ICPE 以新值為主，舊值寫成「CPU 層以 35.4 GiB/s 限速時」的敏感度。
- **任何一個判定字變了 → 先停下來回報使用者**，再改 `ICPE_EVIDENCE.md`。

## 4. 排程：量時間不和我們自己的 job 同時跑

- 我們自己的 job＝同一個 Slurm 使用者的所有 job（含 kv-cliffs 的 kc-\*、DL_Lab4、stage-3、ICPE）。
- **每一步開跑前**：等節點上沒有我們自己的別的 job 在跑（`code/icpe/t1g.py wait-clear`，每 60 s 查 squeue；查不到就停〔規則 7〕；等超過 6 小時就停、回報）。
- **每一步跑完**：用 sacct 查這一步的時間窗裡，節點上有沒有我們自己的別的 job 開始或還在跑（`t1g.py overlap`，每次都記一列到 `results/icpe_h200/t1/gpu/overlap.csv`）。
  - 有重疊，或 gpu_guard 判污染（`x1_gpu` exit 3）→ **照重疊重跑**（不看結果），同一步最多 3 次。
  - 3 次都重疊 → 那一步記成「重疊」，**不進判定**（那一列 NOT_MEASURED），runner 接著下一步；最後照實回報。重疊那幾次的資料保留，只能當參考。
- 分析只用每一步「最後一次、沒有重疊、沒有污染」的那個 run_id（照 `overlap.csv` 篩；同一個 CSV 裡別的 run_id 的列不用）。
- 其他：sbatch 一律 `env -u TMPDIR -u CLAUDE_CODE_TMPDIR`；ICPE hold 中的 job 不動；這個 job 不加進任何 ICPE job 的依賴。

## 5. 分析程式

`code/icpe/t1g.py analyze-*`（只是把 §3 的原本分析程式指到 T1G 的目錄，外加 `overlap.csv` 的篩選與舊→新對照）在**資料收齊之後、打開任何 T1G 輸出之前**寫好並 commit。資料收齊之前不看任何 T1G 的 GPU 數字。

## 6. 不在這一批

- **A3**（第 1 輪模擬存活設定的 GPU 確認）：那幾格（H200 自己 3 格＋MI300X 那格）是用舊常數的第 1 輪模擬選出來的；換常數後要先在 CPU 上用同一條規則重選格子，才知道要量哪幾格 → 另寫登記（T1G-2）。
- **A4**（Llama INT4 第二階段 K2 搬到 H200）：使用者排在 T1 之後，另寫登記。
- SSD 級、NFS 級的限速層、X5、D 節：同 T1 §4，不動。

## 7. 已經看過的數字（登記前）

- T1 的全部模擬結果（`T1_h200_cpu_tier.md` 結果節），以及 E2E-C 第一段的結果（`docs/stage3_design_20261024/runs/E2E_combo.md`）。
- P02 在 T1 常數下的 selftest 輸出（縮小 trace 上 5 個做法的模擬平均 TTFT，例 LRU 0.3048 s；是模擬，不是 GPU）。
- **沒有看過任何 T1 常數下的 GPU 時間**（一個 T1G 的 GPU run 都還沒跑）。

---

## 登記

- 登記時間（`date -u`）：2026-10-28T08:25:20Z
- 登記時 repo HEAD：`64a25bc`（本檔、`code/r_h200_t1g.sh`、`code/icpe/t1g.py` 隨下一個 commit 進 git）

## 偏離（登記之後）
