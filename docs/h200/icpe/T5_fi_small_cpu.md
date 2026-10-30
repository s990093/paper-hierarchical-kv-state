# T5：FreeInference 在小 CPU 層（16–64 GiB）的 X1，對最好的對稱對手

> **性質**：判準登記（事先寫死）。登記時間見文末。登記之後這一節以上不改；和登記不同的做法寫成文末的「偏離 N」。
> **來源**：使用者 2026-10-30「(b) 另外登記一個補跑：在 FreeInference 上，用 X1 完整的對手清單掃 CPU 16–64 GiB，算出對最好對手的上界，再決定判準怎麼改。只用 head node 的 CPU 跑。現在跑」。
> **為什麼**：T4（[`T4_hit_vs_ttft_bridge.md`](T4_hit_vs_ttft_bridge.md)）只報告的一列：NFS 級慢層、CPU 16／32 GiB 時，agent trace 上 oracle 的順序對 **LRU** 值 17.7–29.2%。B3（X1 FreeInference 最大 0.30%）只掃了 CPU 128–1,024 GiB。這一關用 X1 原本的方法（每個元件對**最好的對稱對手**、原本的判定規則）把 FreeInference 擴到 16–64 GiB，決定 B3 的範圍與「預知何時值得做」的判準怎麼寫。
> **範圍**：只有 H200 的常數（CPU 層 T1，`P3_TIER=h200cpu`）；Cake 規劃器（不設 `X1_PLANNER`）；V0；只在模擬器裡（head node，CPU），不上 GPU。

---

## 1. 跑什麼（和 T1 的 P12-X1FI 同一支程式、同樣的參數，只換 CPU 容量與輸出目錄）

- `python code/icpe/p3_fvar.py --variant V0 --out-root results/icpe_h200/t5/p12 x1 sweep --workers 6 --traces <6 週> --rules session_id --modes cake --cpu 16 32 64 128 --out x1_sweep_fi.csv`
  - 6 週＝`fi:2026-05-17_2026-05-23` … `fi:2026-06-21_2026-06-27`（同 T1／T3／T4）；策略＝X1 的預設清單（`POLICIES` 去掉 PRED，同 T1）；慢層 ssd／b32／nfs × GPU 閒／忙；還原只有 Cake（同 T1 的 FI）。
  - 資料讀 `/work/ChihChungHsu/hungweilai/datasets/freeinference_agentic_trace`（T4 已驗證 sha256，並重現 T1 18/18）；開跑前再驗一次 sha256。
- `x1 analyze --sweep x1_sweep_fi.csv --traces freeinference --out-prefix x1fi`（全部回來的請求）＋`--metric mean_ret_user --out-prefix x1fiuser`（只算使用者發起）。
- **CPU 128 GiB 是對照**：它和 T1 的同一格必須逐值相同（§2 檢查 1）。

## 2. 檢查（`code/icpe/t5.py check`；不過就不判）

1. **重現 T1**：CPU 128 GiB 的每一列（週 × 策略 × 慢層 × GPU）的 `mean_ret`、`mean_ret_user`、`n_demote`、`n_drop`、`ret_cpu_loads`、`ret_slow_loads` 和 `results/icpe_h200/t1/p12/x1/x1_sweep_fi.csv.gz`（run `20261028-073351-icpe-t1-p12-x1fi-sweep`）相同（整數逐值；浮點差 ≤1e-6）。
2. 每個（週, 容量）都有完整的策略清單（和 T1 的 128 GiB 同一組策略名）。

## 3. 判準

- **判定字**：照 X1 原本的登記（`docs/phase3_20261017/explore/X1_oracle_decomposition.md` §3，程式 `x1_sim.cmd_analyze`，不改）：某個（元件, 慢層, GPU）在 ≥2 個 CPU 容量上 headroom ≥15%（Cake）→「有看頭」；全部 <5% →「死路」；其他 →「可能」。全部回來的請求為主，使用者發起的只報告。
- **論文怎麼寫**（登記時決定，不看結果改）：
  - **死路** → B3 擴成「CPU 16–1,024 GiB 都 <5%」；T4 的 NFS 那一列是因為 LRU 是弱對手（最好的對稱對手拿走了那部分）。「預知何時值得做」的判準不變。
  - **有看頭** → B3 要限定在「CPU ≥128 GiB」；判準加上 agent trace 的條件：照熱組（`x1fi_hot_groups.csv`）寫出元件、慢層、GPU、CPU 容量。原本「聊天型回來」那個限定要拿掉或改寫。
  - **可能** → 照實報告數字，先停下來回報使用者。
- **只報告**：每個元件、每個 CPU 容量的最大 headroom 與最好的對手；O-ALL(report)；違反（oracle 比某個對手慢）的格數；和 T1（128–1,024 GiB）接起來的 16–1,024 GiB 曲線；使用者發起的判定字。
- 某一項跑不出來 → NOT_MEASURED。

## 4. 限制（照實寫進報告）

- 16–64 GiB 的 CPU 層對 H200 主機偏小（每張卡分到的 DRAM 通常比這大〔判讀，沒有量〕）；這一關回答的是「容量相對工作集小的時候」，不是「典型主機」。
- 只有 FreeInference 前 6 週、Llama-3.1-8B 的 f(i)（V0）、Cake；不跑 STAIR、X5、8 GiB。
- X1 的模型：一次一個請求、不排隊；session 粒度；512 token 的 block。

## 5. 執行

- `code/r_h200_t5.sh`（head node，不用 Slurm、不用 GPU）：資料驗證 → sweep（workers 6）→ `t5.py check` → analyze（全部、使用者發起）→ `t5.py report`。
- 環境同 `code/r_h200_t4.sh`。原始的逐請求檔（RUN_DIR）放 `/work/ChihChungHsu/hungweilai/t5-raw`（不進 git）。
- 輸出：`results/icpe_h200/t5/p12/x1/`、`results/icpe_h200/t5/t5_check.json`、`t5_report.json`。

## 6. 已經看過的數字（登記前）

- T4 的全部結果（BRIDGE-NO-GAP；NFS 級、16／32 GiB 時 O-BEL 對 LRU 17.7–29.2%；ssd 時 ≤2.9%）。
- T1、T3 的 X1 FreeInference（128–1,024 GiB，最大 0.30%／使用者發起 1.52%）。
- **沒有看過**：16–64 GiB 時任何「對最好的對稱對手」的 headroom、O-PF／O-DROP／O-NOW 在小容量的任何數字。

---

## 登記

- 登記時間（`date -u`）：2026-10-30T02:22:45Z
- 登記時 repo HEAD：`ce7d324`（本檔隨下一個 commit 進 git）

## 偏離（登記之後）

無。

---

## 結果（2026-10-30）：「可能」→ 照 §3 先停下來回報使用者

- **執行**（head node，全部 exit 0）：資料驗證 `20261030-102302-icpe-t5-data-verify`；sweep `20261030-102447-icpe-t5-x1fi-sweep`（1,272 個 job，約 22 分鐘）；檢查 `20261030-104713-icpe-t5-check`（**PASS**：CPU 128 GiB 的 828 列和 T1 逐值相同、0 列對不上；24 個（週, 容量）的策略清單完整）；analyze `20261030-104731-icpe-t5-x1fi-an`、`20261030-104745-icpe-t5-x1fiuser-an`；整理 `20261030-104800-icpe-t5-report`。NOT_MEASURED：無。
- **判定字**（X1 原登記的規則）：全部回來的請求 **「可能」**（有 ≥5% 的格，但沒有任何一組在 ≥2 個容量 ≥15%）；只算使用者發起的 **「死路」**。熱組：0。

每個元件的最大 headroom（Cake、對最好的對稱對手；〔模擬〕；全部回來的請求）：

| 元件 | 16 GiB | 32 GiB | 64 GiB | 128 GiB（＝T1） |
|:--|--:|--:|--:|--:|
| O-ORD | 5.92%（nfs、忙；對 S3-FIFO） | 11.81%（nfs、忙；S3-FIFO） | 3.64%（nfs、忙；PENS） | 0.30% |
| O-PF | ≤0（全部格都是負的或 <0；最大 −4.38%） | 4.89%（b32、閒） | 0.47% | 0.22% |
| O-DROP | 0.21% | 0.60% | 0.65% | 0.15% |
| O-NOW | 0.03% | 0.03% | 0.06% | 0.00% |

- **≥5% 的格**：96 格裡 4 格，全部是 O-ORD、NFS 級、CPU 16／32 GiB（5.59／5.92%、11.62／11.81%，閒／忙），對手都是 S3-FIFO，oracle 是 O-BELS，沒有寫入飽和。ssd 級最大 1.66%、b32 級 3.60%（O-ORD）。
- **T4 的 18–29% 大部分是 LRU 太弱**：同一個 NFS 級、16／32 GiB，O-BEL 對 LRU 17.7–29.2%（T4），對最好的對稱對手（S3-FIFO）只剩 5.6–11.8%。
- **只報告**：小容量時 oracle 預取（O-PF）常比對手慢（16 GiB 每格都是負的；最差是 32 GiB、nfs、忙的 −29.5%）——X1 的 O-PF 只預知「何時回來」、騰空間照 LRU，CPU 層小時預取會擠掉別人；oracle 比某個對手慢的格 53 個（O-PF 18、O-NOW 14、O-ALL(report) 21）。使用者發起的最大：O-ORD 3.92%、O-PF 1.44%。
- **意思**（照 §3「可能」：先回報，再改論文）：FreeInference 上，CPU 16–1,024 GiB 的每一格 oracle 都 <15%；5–12% 只出現在「CPU 16–32 GiB＋NFS 級慢層」的淘汰順序。建議的寫法（待使用者決定）：B3 改成「agent trace 上，CPU 16–1,024 GiB 都沒有任何預知元件 ≥15%；CPU ≥128 GiB 時 ≤0.30%」；「預知何時值得做」的判準不加 agent trace 的條件。
- **產出**：`results/icpe_h200/t5/`（`p12/x1/x1_sweep_fi.csv`、`x1fi_*`、`x1fiuser_*`、`t5_check.json`、`t5_report.json`、`p12/fvar_runs.jsonl`）。
