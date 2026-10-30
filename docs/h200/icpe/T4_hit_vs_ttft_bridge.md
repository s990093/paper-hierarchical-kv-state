# T4：同一段 FreeInference，命中率的差距 vs TTFT 的差距（橋接 Liu 等人）

> **性質**：判準登記（事先寫死）。登記時間見文末。登記之後這一節以上不改；和登記不同的做法寫成文末的「偏離 N」。
> **來源**：查新（[`docs/NOVELTY_CHECK_20261030.md`](../../NOVELTY_CHECK_20261030.md)「什麼能讓差別站得住」第 1 點）；使用者 2026-10-30「1. 要不要登記並跑那個「命中率 vs TTFT」的小模擬？ 其他先不動」（我解讀成「做這一項」）。
> **為什麼**：Liu 等人（*When Fancy Eviction Fails*，arXiv 2609.28870）在同一條 FreeInference 上說「LRU 和 Belady 的命中率差距大」，他們的 miss 成本只算重算。ICPE 的 B3 說「Cake 之上，oracle 的 TTFT 上界 ≤0.30%（X1）」。這一關回答：兩者怎麼兜在一起——差距是被「分層＋Cake」吸收了，還是我們的設定裡本來就沒有差距。
> **範圍**：只有 H200 的常數（CPU 層用 T1 的，`P3_TIER=h200cpu`；規劃器是原本的 Cake）；只在模擬器裡（head node，CPU），不上 GPU。**這不是 Liu 等人的重現**（見 §5）。

---

## 1. 設定（全部沿用 B3 的 X1 FreeInference，只多下面標「新」的部分）

- **trace**：FreeInference 前 6 週（`fi:2026-05-17_2026-05-23` … `fi:2026-06-21_2026-06-27`，同 T1／T3），session＝資料自帶的 `session_id`。資料讀 `/work/ChihChungHsu/hungweilai/datasets/freeinference_agentic_trace`（**新**：T1／T3 讀的是 compute node 本地 /tmp 的同一份；同一個 HF revision `16d3c04d`）。開跑前用 `code/h200_p3_verify_data.py` 對 README.txt 的 sha256，再由 §3 的自檢 1 確認重現 T1 的數字。
- **模擬器**：`code/p3x/x1_sim.py`，不改。每個請求一次還原、不排隊；回合結束整個 session 的 block 以 BF16 進 CPU；CPU 滿就依策略選一個 session **降級**（INT4 寫到慢層）或**丟**。
- **CPU 容量**（**新**）：8、16、32、64、128 GiB。128 是 B3 的最小格，用來接上 B3；8–64 是新的。
- **策略**（4 個）：
  - `LRU`：LRU 順序、降級到慢層（＝B3 的對手）。
  - `O-BEL`：X1 的 oracle 順序（降級「下一次被讀最晚」的 session）、降級到慢層（＝B3 O-ORD 的元件之一）。
  - `LRU+NONE`：LRU 順序、**丟**（沒有慢層，沒命中就重算）。
  - `O-BEL+NONE`（**新**）：oracle 順序、丟。`t4.py` 在同一個 process 裡把這個名字加進 `x1_sim.POLICIES`（＝`("bel", "none", None, None)`），不改 `x1_sim.py`。
- **時間設定**：慢層 ssd／b32／nfs × GPU 閒／忙 × 還原 cake／load（只載入），12 種。**主設定**：ssd、GPU 閒（同 B1）。

## 2. 量什麼

每個請求 r 在到達時（`t4.py` 包一層 `View.restore` 記錄，不改它的計算）：
- U＝「可重用的前綴」：前面連續幾個 block 在 r 之前被讀過（`acc[b][0] < r`）。
- C＝在 CPU 層的 block 數（lay＝1、2）；S＝在慢層的（lay＝3）；U − C − S＝不在任何層（被丟過）→ 要重算。

指標（回來的請求，kind＝cont；6 週合併＝Σ÷Σ）：
- **命中率** HR＝ΣC ÷ ΣU（CPU 層的命中，和 Liu 等人的「快取命中」對應）。
- **TTFT**＝回來請求的平均 TTFT（同 B3 的 `mean_ret`；6 週合併＝Σ sum_ret ÷ Σ n_ret）。

差距（對同一個容量 c、主設定）：
- **G_hit**＝HR(`O-BEL+NONE`) − HR(`LRU+NONE`)，單位百分點。
- **G_rec**＝1 − TTFT(`O-BEL+NONE`, load) ÷ TTFT(`LRU+NONE`, load)：沒有慢層、沒命中就重算時，oracle 對 LRU 的 TTFT 改善（最接近 Liu 等人的成本模型）。
- **G_load**＝1 − TTFT(`O-BEL`, load) ÷ TTFT(`LRU`, load)：有慢層、只載入（只報告：分出「慢層」的作用）。
- **G_cake**＝1 − TTFT(`O-BEL`, cake) ÷ TTFT(`LRU`, cake)：有慢層、Cake（＝B3 的設定）。

## 3. 自檢（`code/icpe/t4.py selftest`；任一條不過就不跑）

1. **重現 T1**：`fi:2026-05-17_2026-05-23` 和 `fi:2026-06-21_2026-06-27`、CPU 128 GiB、`LRU`／`O-BEL`／`LRU+NONE`、三個 cake 的 GPU 閒設定：`mean_ret`、`mean_ret_user`、`ret_cpu_loads`、`ret_slow_loads`、`ret_recompute`、`n_demote`、`n_drop` 和 `results/icpe_h200/t1/p12/x1/x1_sweep_fi.csv.gz`（run `20261028-073351-icpe-t1-p12-x1fi-sweep`）相同（整數逐值；浮點差 ≤1e-6，CSV 取到小數 6 位）。→ 確認 /work 的資料、包的那一層都沒有改變模擬。
2. **記帳**：每個請求 0 ≤ C＋S ≤ U ≤ len(full[r])；`*+NONE` 策略 ΣS＝0；記錄的請求數＝trace 的請求數。
3. `O-BEL+NONE` 在 `fi:2026-05-17_2026-05-23`、CPU 8 GiB 跑完，引擎的 audit（每 2,000 步）全過。

## 4. 判準（主設定、回來的請求、6 週合併；每個容量 c 各判一次）

- c 是 **GAP**（有 Liu 等人那種差距）：G_hit ≥ 5 個百分點 **且** G_rec ≥ 15%。
- GAP 的 c 再分：**ABSORBED**（G_cake < 5%）、**PARTLY**（5% ≤ G_cake < 15%）、**NOT-ABSORBED**（G_cake ≥ 15%）。
- 總判定：
  - **BRIDGE-ABSORBED**：至少一個 c 是 GAP，而且每個 GAP 的 c 都是 ABSORBED。論文寫：「同一段 FreeInference 上，oracle 在重算時的 TTFT 差距（≥15%）到了分層＋Cake 之下 <5%」。
  - **BRIDGE-NO-GAP**：沒有任何 c 是 GAP。論文**不能**寫「差距被分層＋Cake 吸收」；B3 的低上界要寫成「容量夠，本來就沒差距（在這個模型裡）」。**先停下來回報使用者**，再改故事。
  - **BRIDGE-MIXED**：其他情況。**先停下來回報使用者**。
- **只報告**：G_load（慢層 vs Cake 各吸收多少）；其他 5 種（慢層 × GPU）設定的同一組判定；只算使用者發起的回來請求（`user_trigger`）；每週各自的差距；所有請求（含新請求）的命中率；`LRU`／`O-BEL`（有慢層）的 CPU 命中率與慢層命中率；oracle 比 LRU 差（負的差距）的格子照實列出（session 粒度、大小不一的 Belady 不保證最好）。
- 某一項跑不出來 → 那一項 NOT_MEASURED。

## 5. 這不是什麼（限制，照實寫進報告）

- **不是 Liu 等人的重現**：他們是 block 粒度（16 token）、HBM 24–120 GiB 加上併發、DRAM／SSD 池、oracle 是 Belady 與 BeladyCompute；我們是 X1 的模型：session 粒度、512 token 的 block、一次一個請求不排隊、CPU 層當快取、oracle 是 X1 的 `bel` 順序。只比**方向**（命中率有差距時，TTFT 的差距在分層＋Cake 之下剩多少），不比數字。
- 只有 FreeInference 前 6 週、Llama-3.1-8B 的 f(i)（V0）、Cake 規劃器；不跑 STAIR、不跑 conversation。

## 6. 執行

- `code/r_h200_t4.sh`（head node，不用 Slurm、不用 GPU）：資料驗證 → `t4.py selftest` → `t4.py sweep --workers 4`（6 週 × 5 容量 × 4 策略＝120 個 job，每個 job 12 種時間設定）→ `t4.py analyze`。
- 環境同 `code/r_h200_t3.sh --part head`，但不設 `X1_PLANNER`（＝Cake），`TIARA_DATASETS=/work/ChihChungHsu/hungweilai/datasets`。
- 輸出：`results/icpe_h200/t4/`（`t4_sweep.csv`、`t4_cells.csv`、`t4_weeks.csv`、`t4_verdict.json`、`t4_selftest.json`）。

## 7. 已經看過的數字（登記前）

- T1 的 X1 FreeInference sweep（`x1_sweep_fi.csv.gz`）：CPU 128–1,024 GiB 時 `LRU`、`O-BEL`、`LRU+NONE` 在 ssd、GPU 閒、Cake 的 `mean_ret`，以及 `LRU` 的 `n_demote`、`ret_cpu_loads`、`ret_slow_loads`、`ret_recompute`。看到的是：128 GiB 以上，`LRU+NONE` 比 `LRU` 最多慢約 3%，降級次數一週 65–798 次——**容量壓力很小**。這是 §1 加上 8–64 GiB 的原因。
- ICPE_EVIDENCE、T1、T3 的全部結果；Liu 等人論文裡的數字（查新報告）。
- **沒有看過**：8–64 GiB 的任何數字、`O-BEL+NONE` 的任何數字、任何命中率。

---

## 登記

- 登記時間（`date -u`）：2026-10-30T01:53:11Z
- 登記時 repo HEAD：`aca1f1b`（本檔隨下一個 commit 進 git）
