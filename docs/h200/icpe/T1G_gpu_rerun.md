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

### 附加 1：別的使用者的 job（只報告，不改判定）

- **時間**：2026-10-28T15:08:47Z（repo HEAD `a6ce4c4`）。登記之後、看任何 T1G GPU 時間之前寫的。寫的時候只看過 `b_share.csv` 的欄名與 `ts` 欄的格式（到秒），沒有看任何時間數字。
- **為什麼**：§4 只擋我們自己的 job。同一個節點上別的使用者的 job（含同 lab 帳號的人）擋不了，原本只在每個 run 開始（`context.txt`）和結束（`cotenants_end.txt`）各記一次 squeue。使用者 10/28 問「同 lab 其他人一起用不就也有影響」。我用 `sacct -a -N hgx-cn01` 查了 T1G 開跑（17:18）到 23:00 這段：別的 lab 有 6 個長 job 一直在跑；另外有 7 個短 job（同 lab 帳號 4 個、別的 lab 3 個）在兩次快照之間開始又結束，快照完全沒記到。
- **記什麼**：分析每一步（§4 篩出來的那個 run_id）時，用 `sacct -a -N <node> -S <t_start> -E <t_end> -X` 列出這一步時間窗裡 `$USER` 以外的 job。
  - 窗開始前就在跑、窗結束後還在跑 →「常駐鄰居」：只記數量與 job 名，不算暴露（各臂、各次重複都一樣在它旁邊，量測順序又是隨機的）。
  - 在窗裡開始或結束 →「事件」（負載改變的時刻：開始時載權重，結束時釋放）。
- **暴露單位**：一個臂、一個量測點的一次重複＝同一個 run_id、同一組設定欄（檔案裡有的才用：strategy／policy、cpu_frac、cpu_cap_chunks、ssd_frac、ssd_dev、gap_s、wl_seed、workload、cpu_gibps、release、k、cpu_GiB）、同一個 rep 的那些列。它的時間＝那些列 `ts` 的〔最早 − 120 s，最晚 + 120 s〕（`ts` 只到秒；120 s 涵蓋鄰居開跑時載權重那段）。有任何事件落在裡面，這個單位就算「暴露」。
- **報告**：每一步列出常駐鄰居數、事件數、暴露單位數／總單位數。再把暴露的單位拿掉，用同一支分析程式（§5）重算一次，列出拿掉前後的判定字，以及每個量測點剩下幾次重複。
- **怎麼用**：
  - 判定照 §3，用全部單位。
  - 拿掉暴露單位後判定字不同 → 停下來回報使用者，不寫進 ICPE_EVIDENCE。
  - 某個量測點拿掉後剩不到 6 次重複 → 那一點的敏感度結果寫 NOT_MEASURED。
  - sacct 查不到 → 那一步寫 NOT_MEASURED，不是「沒有暴露」（CLAUDE.md §1 規則 7）。
- **舊的 run**（T1G 拿來對照的原本 ICPE run）：用同樣的定義只報暴露比例，不重算判定。
- T1G-2（A3）照同樣的做法。
- **程式**：另寫 `code/icpe/t1g_exposure.py`，不改正在被 runner 呼叫的 `t1g.py`。在 22882 跑完、打開任何 T1G 輸出之前寫好並 commit。

#### 附加 1 的偏離 1（2026-10-28T18:27:31Z；看過暴露單位的「數量」之後、看任何 T1G 時間數字之前）

- **看到什麼**（只有數量，沒有時間）：scan `20261029-022626-icpe-t1g-exposure-scan`（16 步，sacct 都查得到）。
  - A1''：180 個單位裡 29 個暴露，拿掉後 30 個量測點有 22 個不到 6 次。
  - A1'：26／180 個單位暴露，30 個點有 20 個不到 6 次。
  - A2：45／132 個單位暴露，44 個點有 23 個不到 6 次（A2 每點本來就只有 3 次，有一次暴露就不到 6 次）。
  - 照附加 1，這三步的「拿掉重算」大多會是 NOT_MEASURED，回答不了「鄰居的 job 有沒有影響」。
- **加一項（只報告；附加 1 原本的規則不改）**：在同一個量測點裡，直接比暴露的重複和沒暴露的重複。
  - 每個單位取回來請求（round > 1）TTFT 的平均。
  - 一個點同時有暴露和沒暴露的重複時，算〔暴露的中位 ÷ 沒暴露的中位 − 1〕。
  - 每一步報點數、中位數、四分位，以及變慢（> 0）的點數。
  - P02 每個點只有 1–2 次重複，不算這項。
- 這項不進判定，也不觸發「停下來回報」；判定字的比較仍照附加 1。

---

## 結果（第一批；2026-10-29）

- **job 22882**：2026-10-28 17:18:14 → 2026-10-29 02:56:49，COMPLETED。每一步開跑時節點上都沒有我們自己的別的 job（`wait-clear` 都等 0 次）；`overlap.csv` 18 列全部「重疊：無」；沒有重跑；gpu_guard 都沒有判污染。
- **分析**（§5 的 `t1g.py analyze`）：`20261029-023018-icpe-t1g-analyze-a1a2`（A1''、A1'、A2）；P02 `20261029-030016-icpe-t1g-analyze-p02`。第一次 `20261029-025847-icpe-t1g-analyze-p02` exit 1：`FileNotFoundError: '/work/ChihChungHsu/hungweilai/datasets/mooncake/traces/conversation_trace.jsonl'`。原因是 head node 的 shell 沒設 `X1_TRACE_DIR`，和 10/26 原本的 P02 分析同一個問題（RUNLOG_H200_ICPE 該節）。在 `load_trace` 就失敗，沒有寫出任何東西。設 `X1_TRACE_DIR=/work/ChihChungHsu/hungweilai/datasets/traces` 重跑（沒有改 `t1g.py`，22886 的 runner 每一步都會呼叫它）。

### 判定字：全部沒有變

| 列 | 判準（原本的登記） | 舊（CPU 層 35.4 GiB/s） | 新（CPU 層 H200 的常數） |
|:--|:--|:--|:--|
| A1''（fa3） | P0-1 §3.1：每個 cpu_frac，S5 對 S4+（回來請求 TTFT 中位數，正＝S5 快）≤ +δ\*，三格都過、污染列 0 | +0.07／−0.47／+0.03%，δ\* 5% → 成立 | **−0.37／−0.37／+0.01%**（rep 6，C4 雜訊 0.03% → δ\* 5%）→ **成立** |
| A1'（sdpa） | 同上 | 合併組 cpu 0.25／0.5／1.0 都在 ±δ\*（−0.03／−0.12／+0.03%） | **+1.16／+0.09／−0.02%**（rep 6，C4 雜訊 1.76% → δ\* 5%）→ **成立** |
| A2 | 08 消融：寫入時決定同時贏最好的不看位置與 S4B 各 ≥5%（GPU 列） | 0/16 | **0/16** |
| B6／C2（P02） | P0-2：實測÷模擬在 [0.9, 1.1]、S_orig 差 ≥5% 的對排序相同、headroom 正負號相同 | 成立（10 格 0.911–0.975；15 對；8 個正負號） | **成立**（10 格 **0.911–0.987**；排序、8 個正負號全同；12 個乾淨 run） |

### 數字（實測；縮小設定 k＝8，Mooncake conversation、SSD、CPU 16／32 GiB〔cpu_GiB 128／256 縮放〕、Llama-3.1-8B）

| 項目 | 舊 | 新 |
|:--|--:|--:|
| 回來請求平均 TTFT，GPU：LRU／HINT-PF／O-PF（128） | 0.278／0.238／0.229 s | 0.278／0.237／0.228 s |
| 同上（256） | 0.273／0.237／0.229 s | 0.272／0.237／0.228 s |
| HINT-PF 對 LRU；O-PF 對 LRU（GPU，128／256） | 14.3／12.9%；17.7／16.0% | **14.5／12.9%；17.9／16.2%** |
| HINT-PF 拿到 O-PF 改善的比例（GPU；同設定的模擬 S_orig） | 0.81／0.81（0.86／0.85） | **0.81／0.79**（0.86／0.85） |
| O-PF 對 LRU，模擬 S_orig（128／256） | 23.1／19.9% | 24.2／21.1% |
| 實測÷S_real（換成實際開始時間的模擬） | 0.966–1.013 | 0.967–1.021 |
| 含排隊時 O-PF 對 LRU（只報告；回來請求平均排隊 1.19–1.26 s） | 7.9／4.6% | 8.1／4.7% |
| O-PF 比 HINT-PF 快（GPU；S_orig） | 3.9／3.6%（3.9／3.6%） | 4.0／3.8%（4.2／3.9%），<5% 不判 |

- **附加 1（別的使用者的 job；只報告）**：scan `20261029-030108-icpe-t1g-exposure-p02`（18 步，sacct 都查得到）。
  - 暴露單位：A1'' 29／180，A1' 26／180，A2 45／132，P02 1／12（256-LRU-r0；那一步窗裡有 2 個別的 lab 的 job 結束）。
  - 影子目錄的回歸（什麼都不拿掉）：四步都逐值相同（`20261029-023222-icpe-t1g-exposure-regress`、`…-030108-…-exposure-p02`）。
  - 拿掉暴露單位（`20261029-023333-icpe-t1g-exposure-a1a2`、`…-030108-…-exposure-p02`）：**沒有任何判定字不同**。照附加 1 剩不到 6 次的點寫 NOT_MEASURED，所以 A1''、A1'、P02 的字全部 NM；A2 還能比的 6 列都仍然「不過」。
  - 同一點裡暴露的重複對沒暴露的重複（偏離 1）：
    - A1''：22 點，中位 +0.01%（四分位 −0.07～+0.06%），12 點變慢。
    - A1'：20 點，中位 +0.08%（−0.13～+0.17%），13 點變慢。
    - A2 doc g1-s0：11 點，中位 −0.02%（−0.54～+0.22%）。
    - **A2 chat g2-s0：8 點，中位 +8.0%（−0.6～+22.0%），6 點變慢**。這一步窗裡有 13 個事件，10 個是同 lab 帳號的 job。A2 chat 那幾列本來就輸最好的對手 10–217%（要贏 5% 才算過），所以不會翻。
  - 原本的 run（`exposure_old.csv`）也有一樣的鄰居：暴露單位 A1'' 25／180、A1' 45／132、A2 22／132、P02 0／12、A3 104／450。
- 自己的程式錯（`t1g_exposure.py`，第一次 regress `20261029-023149-icpe-t1g-exposure-regress`）：單位表已經有 `run_id` 欄（來源 run），再插入這次的 run_id 時撞名，exit 1。單位表的欄改名成 `src_run` 後重跑。
- **產出**：`results/icpe_h200/t1/gpu/{a1pp,a1p,a2,p02}/`（>1 MB 的原始列壓縮：`b_share.csv.gz`、`b2_share.csv.gz`、`gpu_requests.csv.xz`；`analysis/` 底下篩過的副本由 `t1g.py analyze` 重建，不進 git）、`overlap.csv`、`t1g_compare.json`、`exposure/`。
- **意思**：CPU 層換成 H200 自己的常數之後，GPU 實測的第一批（A1''、A1'、A2、B6／C2）**判定字一個都沒有變**，數字的變化都在 1.2 個百分點以內。照 T1 §5，ICPE（只用 H200）改用新值。
- **第二批 A3（T1G-2，2026-10-29）**：判定字也不變（GPU 上存活 0/2，原本 0/3）。結果在 [`T1G2_round1_rerun.md`](T1G2_round1_rerun.md) 的結果節。
