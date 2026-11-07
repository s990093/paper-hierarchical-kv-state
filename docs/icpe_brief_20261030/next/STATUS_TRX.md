# STATUS_TRX（icpe-trx，TRX40）

> 只由 `icpe-trx` 寫。格式照 STATUS_DATA：日期時間／做完什麼（commit、run_id）／卡住什麼（完整錯誤）／需要別的 session 做什麼／需要使用者決定什麼。最新的一節在最上面。
> 工作＝STATUS_DATA 2026-11-05T09:00+08:00 節「給 `icpe-trx`（TRX40）的步驟」0–7：SemiAnalysis 的 `sa_main` → `sa_main_ttl7` → `sens_rep --only 'sa_*'`。不解讀上限、不下判定。

## 2026-11-07T17:53+08:00 — 偏離 14 第一段：**Mooncake 的 main＋P1 完成並收進 git**；檢查表 mc_p1 `D14_TABLE_RESULT IDENTICAL`；第一段的 sa 全部（main、P1–P3）都過了檢查

- **block**：
  - d14_old_main/mooncake（3,456 job）。
  - d14_old_p1/mooncake_ttl（1,728 job，17:47:30 → 17:47:38）、mooncake_trunc（6,912 job，17:47:38 → 17:51:10）。
  - 我的檢查 `ok=2 bad=0`。d14_sa_p3 的 24 個也 `ok=24 bad=0`。
  - icpe-data 的檢查（run `20261107-175120-icpe-trx-d14-blockcheck-mc`，plans 到 d14_old_p1 為止）：`D14_BLOCKCHECK PASS n=111 fail=0`。`results/icpe_next/b/d14_blockcheck_trx.csv` 換成這次的 111 列。
- **collect**：`collect --plans d14_old_main d14_old_p1`，collected 3，輸出在 `results/icpe_next/b/d14_old_main/mooncake`、`d14_old_p1/mooncake_{ttl,trunc}`。
- **檢查表**：run `20261107-175127-icpe-trx-d14-verb-mc`，rc 0。
  - 參考表：`origin/icpe-next-main` 58252b4f 的 `verB/mc_p1_headroom_components.csv.xz`，sha256 80a8b4d0e36e…。
  - patch：`D14_PATCH_OK blocks=2`。
  - rowcheck：`mc_main_d14` 29376 rows（54／17／8）PASS；`mc_p1_ttl_d14` 25920 rows（54／15／8）PASS。
  - 照抄：
    - O-DROP、O-GPU、O-NOW、O-ORD：x=324 y=324 matched=324，7 欄 ndiff 全部 0。
    - O-PF（skipped）：x=324 matched=324；T_oracle 234、T_rival 7、headroom 234、best_rival_gpu_policy 5、oracle_gpu_policy 113，其他 0。
    - O-ALL(report)（skipped）：x=216 y=216 matched=216；T_oracle 165、headroom 165，其他 0。
    - 結果 `D14_TABLE_RESULT IDENTICAL`。
  - 收在 `results/icpe_next/b/analysis_d14/mc_p1/`。table_sha256_uncompressed 322593827ac6…。
- **之後**：Mooncake 的第二段（mooncake_conversation、mooncake_toolagent）另外開一個 `next_d14_verc_queue.sh` 跑，和 sa/uc/lm 那個佇列的變體不重疊。

## 2026-11-07T17:47+08:00 — **偏離 14 第二段（修正版版本 C）開跑**：sa 12＋uc 6＋lm 6 個變體，和第一段的後半同時跑；第一段 sa_p2 24／24 OK

- **GO 的來源**：
  - icpe-data 17:4x：icpe-main 給了 GO。
  - icpe-main 17:4x 轉達使用者在 icpe-main 說的「开始跑」（約 17:45）。
  - 這在使用者 15:4x 在本 session 選的「接，等引擎定了再開」範圍內。
  - 引擎關卡：icpe-sim 對拍 v8_dev14 36780534，41,282 列和 x1fast 63b8e32 相同（STATUS_SIM 17:41）；icpe-rux 的 D3 通過。
- **merge**：`origin/icpe-next-data` cdb9f0b1 → 9f16f756。新增 `next_d14_verc_queue.sh`、`next_d14_store_table.py`，以及 acvlab 第一段的結果（FI、sg/wc、ms/oh）。沒有改到既有的程式。
- **引擎 worktree**：`$ROOT/rux-d14-wt`，detached 在 icpe-next-rux 36780534，唯讀，不在那裡 commit。
  - `v8_dev14/src/main.rs` sha256 d2048fa38a42c78b（等於 next_d14_verc.sh 要求的值）。
  - origin/icpe-next-rux 之後的 commit 沒有改 v8_dev14、v8_features、p3x（`git diff --stat` 是空的）。
- **佇列**：外層 run `20261107-174218-icpe-trx-d14-stage2-verc-queue`（`trx_run.sh` 加三個守門），17:42:18 開始。
  - 指令：`next_d14_verc_queue.sh 16 semianalysis_N{16,32,64,72}_s{1,2,3} ultrachat_T{60,600}_s{1,2,3} lmsys_T{60,600}_s{1,2,3}`。
  - 環境：`D14REPO=$ROOT/rux-d14-wt`、`DATAREPO`＝本 repo、`NEXT_B_TAG=icpe-trx`；NEXT_B_PY、X1_NB_DIR、X1_TRACE_DIR 和 Rust 工具鏈都用 TRX40 的路徑。
  - workers 16：第一段還在用 46 個 worker，兩段同時跑，CPU 會超過 48 個執行緒。
  - 每個變體在 collect 之前一定要過三關，任何一步失敗就停：
    1. `D14_SWEEP_RESULT IDENTICAL`：opf 以外的列和舊的 apc sweep 逐位元相同。
    2. `EXPLORE_RESULT REPRODUCED`：版本 B 的 16 個 GPU 規則 × 9 個容量，對第一段的檢查表 `analysis_d14/<fam>_p1`。
    3. `py_fallback_jobs=0` 且 `src_sha256=4eca5bda`。
  - 這就是我說過的「v8 的版本在 TRX40 上要另外過一次重現檢查」，每個變體都做。
  - 舊的 sweep：sa 從 `$ROOT/d14_old_apc/`（sha256 已對過 MANIFEST），uc/lm 從本 repo 的 `results/icpe_next/apc/`。
  - 第一個變體 semianalysis_N16_s1 有 14,184 個 job，前 200 s 跑了 3,087 個。
- **第一段**：d14_sa_p2 24 個 block 都 OK（我的檢查）；17:41 跑到 d14_sa_p3，99／180 個 block 完成，0 失敗。

## 2026-11-07T17:30+08:00 — 偏離 14 第一段：**SemiAnalysis 的 main＋P1 完成並收進 git**；檢查表 sa_p1 `D14_TABLE_RESULT IDENTICAL`

- **block**：d14_sa_main 12 個（每個 864 job）、d14_sa_p1 24 個（`_ttl` 12 個，每個 432 job；`_trunc` 12 個，每個 1,728 job）。
  - 我的檢查：d14_sa_p1 `ok=24 bad=0`。
  - icpe-data 的檢查（run `20261107-172812-icpe-trx-d14-blockcheck-sa`）：`D14_BLOCKCHECK PASS n=61 fail=0`。`results/icpe_next/b/d14_blockcheck_trx.csv` 換成這次的 61 列。
- **collect**：`collect --plans d14_sa_main d14_sa_p1`，collected 36（第一個 commit 前的 sa_main 12 個加上 sa_p1 24 個），輸出在 `results/icpe_next/b/d14_sa_main/`、`d14_sa_p1/`。
- **檢查表**：run `20261107-172843-icpe-trx-d14-verb-sa`，rc 0。
  - 參考表：`origin/icpe-next-main` 58252b4f 的 `results/icpe_next/main/verB/sa_p1_headroom_components.csv.xz`，sha256 a24f5fa9ed1e…，用 `git show` 取到 `$ROOT/runs/d14-verb-refs/`。
  - patch：`D14_PATCH_OK blocks=24`。
  - rowcheck：`sa_main_d14` 88128 rows（54／17／8）PASS；`sa_p1_ttl_d14` 77760 rows（54／15／8）PASS。
  - 照抄（x=648 y=648 matched=648）：
    - O-DROP、O-GPU、O-NOW、O-ORD：7 欄 ndiff 全部 0。
    - O-PF（skipped）：T_oracle 564、headroom 564、oracle_gpu_policy 83，其他 0。
    - O-ALL(report)（skipped）：T_oracle 575、headroom 575，其他 0。
    - 結果 `D14_TABLE_RESULT IDENTICAL`。
  - 收在 `results/icpe_next/b/analysis_d14/sa_p1/`（格式同 bl_p1）。table_sha256_uncompressed 92c16c8d8f17…。
- **佇列**：到 17:29:55 為止 67／180 個 block 跑完，0 失敗；正在跑 d14_sa_p2。Mooncake 要等 d14_old_p1:mooncake_*。

## 2026-11-07T17:19+08:00 — 偏離 14 第一段：**UltraChat、LMSYS 的 main＋P1 `_ttl` 完成並收進 git**；檢查表 uc_p1／lm_p1 `D14_TABLE_RESULT IDENTICAL`；佇列繼續（sa_p1 起）

- **已跑完並通過檢查的 block**（佇列 run `20261107-171019-icpe-trx-d14-stage1-queue`）：

  | plan | block | 每個 block 的 job |
  |:--|--:|--:|
  | d14_sa_main | 12 | 864 |
  | d14_old_main:mooncake | 1 | 3,456 |
  | d14_chat_main uc/lm | 12 | 864 |
  | d14_chat_p1 uc/lm | 12 | 432 |

  - 我的檢查（`d14_blockcheck.py`）：`D14_BLOCKCHECK ok=37 bad=0`。
  - icpe-data 的 `code/icpe/next_d14_blockcheck.py`（run `20261107-171749-icpe-trx-d14-blockcheck-uc-lm`）：`D14_BLOCKCHECK PASS n=37 fail=0`。輸出放在 `results/icpe_next/b/d14_blockcheck_trx.csv`（37 列；之後每收一個家族就更新）。
- **collect**：`next_b_queue.py collect --plans d14_chat_main d14_chat_p1`，collected 24。
  - 輸出在 `results/icpe_next/b/d14_chat_main/{uc,lm}_T{60,600}_s{1,2,3}` 和 `d14_chat_p1/*_ttl`，每個 block 只含清單 rerun_policies（O-PF、O-ALL）的列。
  - LMSYS 和以前的 chat_main／chat_p1 一樣只放彙總的 sweep 列，沒有逐對話的數字。
- **第一段的版本 B 檢查表**（`next_d14_verb.sh`；DATAREPO＝本 repo；NEXT_B_PY、MAIN_ROWCHECK 用 TRX40 的路徑；run `20261107-171807-icpe-trx-d14-verb-{uc,lm}`，rc 0）：
  - patch：`D14_PATCH_OK blocks=12`。chat_p1 每個 block 是 `base=6480 d14=432 replaced=432 kept=6048`。
  - rowcheck：`uc_main_d14`、`lm_main_d14` 都是 44064 rows、cells 54、cpu 17、gpu 8，PASS；`uc_p1_ttl_d14`、`lm_p1_ttl_d14` 都是 38880 rows、cells 54、cpu 15、gpu 8，PASS。
  - 和原本的版本 B 表 `results/icpe_next/b/analysis/{uc,lm}_p1` 比，照抄：
    - 兩個家族的 6 個元件都是 `x=324 y=324 matched=324`，7 欄 ndiff 全部 0。
    - O-PF、O-ALL(report) 是 `skipped=True`（不要求相同），ndiff 也是 0。
    - 結果 `D14_TABLE_RESULT IDENTICAL`。
  - 照 icpe-data 的格式（bl_p1，246d3fed）收在 `results/icpe_next/b/analysis_d14/{uc,lm}_p1/`：表 .csv.xz、violations .csv.xz、table_vs_verB.csv、rowcheck.csv、patch_summary.csv、manifest.json（含 run_id、cmd、context、各檔 sha256）。
  - table_sha256_uncompressed：uc 1b5745c94382…、lm 794899a08cc8…。
- **佇列**：17:17:31 開始 d14_sa_p1。資源：MemAvailable 最低 55.6 GiB；本組 swap 0 MiB；/ssd4 剩 490 GiB。

## 2026-11-07T17:12+08:00 — **xref_dev14 在 TRX40 重現：Python 6/6、Rust 6/6 PASS**（1,184 列，0 列不同）→ **第一段 180 個 d14 block 開跑**（17:10:19）

- **merge**：
  - `origin/icpe-next-sim` b7d2ee57 → 553a3cee（xref_dev14 的 6 個 CSV、README、env.sh、`next_sim_dev14_check.py`；`code/p3x` 沒變）。
  - `origin/icpe-next-data` a3e848fd → 86e30c34（只有 STATUS_DATA；17:01 那節＝TRX40 的第一段 GO，條件是先重現 xref）。
- **xref_dev14 重現**：
  - 腳本 `$ROOT/tools/xref_dev14_trx.sh <py|rust> <workers>`：README 的 6 個 sweep，env.sh 的路徑換成 TRX40 的；`ON="--pf-expire --noise-overdue resample --pf-gpu-recheck"`，`RUN_DIR=""`。
  - 比對：`code/icpe/next_sim_dev14_check.py <ref> <new> <out> <run_id> <tag> all`。
  - **第一次啟動失敗（我的腳本錯）**：run `20261107-165931-icpe-trx-xref-dev14-{py,rust}`，rc 3。
    - 我用 `:` 切 sweep 參數，把 `nb:semianalysis_N16_s1` 切斷成 `--traces nb`。
    - 錯誤：`FileNotFoundError: [Errno 2] No such file or directory: '/ssd4/hungwei/icpe-next/datasets/mooncake/traces/nb_trace.jsonl'`。
    - 引擎沒跑到任何 job。兩個目錄留了 `FAILED.txt`，不當證據。改用 `|` 當分隔符後重跑。
  - **重跑**：run `20261107-165948-icpe-trx-xref-dev14-rust`（6 workers，16:59:48 → 17:01:19）、`20261107-165948-icpe-trx-xref-dev14-py`（40 workers，16:59:48 → 17:10:00），都是 rc 0，沒有 `*KILLED`。
    - Python 3.12.15。Rust：`src_sha256=0bfacb59…`、`py_fallback_jobs=0`。
    - 照抄：

      | sweep | job | 列 | Rust | Python |
      |:--|--:|--:|:--|:--|
      | sa_qb | 178 | 208 | `DEV14 trx-rust-sa_qb PASS` | `DEV14 trx-py-sa_qb PASS` |
      | sa_qt | 108 | 128 | PASS | PASS |
      | mc_qb | 178 | 208 | PASS | PASS |
      | mc_qt | 108 | 128 | PASS | PASS |
      | sa_n | 232 | 256 | PASS | PASS |
      | mc_n | 232 | 256 | PASS | PASS |

- **第一段開跑**：外層 run `20261107-171019-icpe-trx-d14-stage1-queue`（`trx_run.sh`：nice 10；memwatch 6,144 MiB、ownswapguard 本組 256 MiB、diskguard 50 GiB）。
  - 指令：`next_b_queue.py run --engine rust --workers 46 --plans <$ROOT/tools/d14_trx_plans.txt 的 15 項，照 icpe-data 的順序>`，repo 86e30c34。
  - 開跑時整機 swap 3,090 MiB（舊的 2,933 上限在這裡就會停）；本組 swap 0 MiB；MemAvailable 56–58 GiB。
  - 第一個 block `d14_sa_main/sa_N16_s1` 用了 16 s，864 個 job。
- **逐 block 檢查**：腳本 `$ROOT/tools/d14_blockcheck.py <count 的 stdout> <runs>`，照 icpe-data 17:01 的要求：
  - 全部 block：rc 0、`jobs=` 等於 count、`rust_timing.jsonl` 行數等於 jobs、`py_fallback_jobs=0`、`code_sha=3180504e8d57+rs0bfacb59`、沒有 `*KILLED`。
  - P4 另外要求：`kept_jobs` 等於 count、`unmatched_cells=0`。
  - 第一個 block：`OK d14_sa_main/sa_N16_s1 jobs=864 timing=864`。
- **之後**：每個家族的 main＋P1 `_ttl` 跑完就 collect、push，通知 icpe-data 和 icpe-main。家族指 sa、mc、uc、lm。

## 2026-11-07T16:03+08:00 — merge `origin/icpe-next-data` 4bae90be（含 icpe-sim f5b2960c，引擎 63b8e32）；**旗標全關三項對拍 0 差**；count **MATCH_RERUN 300／WHOLE_BLOCK 20／MISMATCH 0／ALL TOTAL 2,193,997**；TRX40 的 180 個 d14 block dry-run OK；**還沒跑任何 d14 block**（等 xref_dev14）

- **merge**：9a00c00d。PREREG 的衝突 icpe-data 已在 c68abaad 解掉；我這邊沒有衝突。
  - 引擎檔和 40a4c87f 相同：x1_sim 8b53588b85c3、x1_rust 0d89c4ba33c5、x1_nb 8251e2587464，main.rs 逐字相同。
  - 唯讀 worktree `$ROOT/sim-wt-40a4c87f` 留著，不再用。
- **旗標全關三項對拍**（主 repo 9a00c00d；外層 run `20261107-160107-icpe-trx-verify-dev14-off-9a00c00d`，rc 0，16:01:07 → 16:03:02；新守門 ownswapguard 本組 swap 0 MiB；MemAvailable 最低 56,090 MiB）：
  - `code_sha 3180504e8d57+rs0bfacb59`，`py_fallback_jobs=0`。
  - (a) G0：4,416 列，128,064 格，不同 0，`G0_RESULT IDENTICAL`。
  - (b1) xc_sa 對 icpe-data：1,280 格，不同 0。
  - (b2) xc_sa 對 TRX40 的 Python：`CMP PASS`。
  - (c) `sa_N16_s1` 整個 block 對 git：293,760 格，不同 0。
- **count**（run `20261107-160107-icpe-trx-count-d14-all`，全部 25 個 d14_ plan）：rc 0。照 icpe-data 要的檢查：
  - `MATCH_RERUN` 300、`WHOLE_BLOCK` 20、`MISMATCH` 0。
  - `ALL TOTAL jobs=2193997`。
- **dry-run TRX40 這份**（run `20261107-160131-icpe-trx-dryrun-d14-trx`）：
  - 指令：`run --dry-run --engine rust --workers 46`。plan 照 icpe-data 的順序，清單在 `$ROOT/tools/d14_trx_plans.txt`。
  - rc 0，180 個 block：

    | plan | block 數 |
    |:--|--:|
    | d14_sa_main | 12 |
    | d14_old_main | 1 |
    | d14_chat_main | 12 |
    | d14_chat_p1 | 12 |
    | d14_sa_p1、d14_sa_p2、d14_sa_p3 | 各 24 |
    | d14_old_p1、d14_old_p2、d14_old_p3 | 各 2 |
    | d14_sens_rep | 9 |
    | d14_sa_p4_b1、d14_sa_p4_b2 | 各 24 |
    | d14_p4_b1、d14_p4_b2 | 各 4 |

  - 逐 block 檢查（腳本讀 dry-run 的 stdout）：
    - 非 P4 的 124 個都有 `--pf-expire --pf-gpu-recheck`。
    - P4 模型 P 的 28 個另外有 `--noise-overdue resample`，模型 M 的 28 個沒有。
    - P4 的 56 個都有 `--jobs-from`。
    - 沒有 wt × 預取到 GPU 的組合。
    - 25 個 nb 紀錄都在 `$ROOT/datasets/x1_nb`，另有 Mooncake 的 conversation、toolagent 兩個原始檔。
    - 沒有 CacheWise。
  - 磁碟：非 P4 的 block 照程式存 npz，但只重跑 O-PF／O-ALL。以前完整的 sa_main N72 block 的 req 是 6.2 GiB。/ssd4 剩 498 GiB，diskguard 50 GiB。
- **還沒做**：旗標開的那一關，要用 Python 和 Rust 各比一次 icpe-sim 的 xref_dev14，它在 git 上還沒有。之後 icpe-data 說可以，才開正式 block。

## 2026-11-07T15:58+08:00 — 偏離 14 引擎 40a4c87f 在 TRX40 上**旗標全關的三項對拍都 0 差**；旗標開的參考失敗（icpe-sim），**HOLD**；**使用者決定 swap 守門改看本組自己的 swap**；分工加上 Mooncake

- **同步方式**：
  - 試 `git merge --no-commit origin/icpe-next-sim`（40a4c87f），PREREG.md 是 add/add 衝突。那份不是我的檔案，所以 `git merge --abort`，分支沒有變（仍是 5acbc46e）。
  - icpe-data 會在 icpe-next-data 合好，我之後只 merge icpe-next-data。
  - 為了先驗引擎，開了唯讀的 detached worktree `$ROOT/sim-wt-40a4c87f`，不在那裡 commit。
  - 引擎檔的 sha256 和 icpe-sim 給的相同：x1_sim.py 8b53588b85c3、x1_rust.py 0d89c4ba33c5、x1_nb.py 8251e2587464。
- **對拍**：外層 run `20261107-155247-icpe-trx-verify-dev14-off-40a4c87f`（`trx_run.sh`；三個守門），rc 0，15:52:47 → 15:55:31。
  - 用的是新腳本 `$ROOT/tools/verify_dev14_off.sh`，比對方法同 `verify_rust2.sh`：sweep 在 worktree 跑，參考檔和比對程式用主 repo。
  - 三個新旗標都沒開。
  - Rust 在 TRX40 重編：src_sha256 0bfacb59…（等於 icpe-sim 給的值）、bin_sha256 1d024543…。三個 run 的 `code_sha` 都是 `3180504e8d57+rs0bfacb59`，`py_fallback_jobs=0`。
  - 比對 run `20261107-155530-icpe-trx-compare-40a4c87f`，照抄：

| 項目 | sweep run | 參考 | 結果 |
|:--|:--|:--|:--|
| (a) G0 | `20261107-155247-icpe-trx-g0-rust-40a4c87f`（26.6 s） | H200 `results/icpe_h200/t1/p01/sim_V1/x1/x1_sweep.csv.gz` | 4,416 列，128,064 格，不同 0，`G0_RESULT IDENTICAL` |
| (b1) xc_sa | `20261107-155247-icpe-trx-xc_sa-rust-40a4c87f`（44.8 s） | `results/icpe_next/b/xcheck/xc_sa`（icpe-data 的 Python） | 32 列，1,280 格，不同 0 |
| (b2) xc_sa | 同上 | TRX40 的 Python xc_sa（run `20261105-105555-…-54fa2d8`），`next_sim_cmp.py` | 32 列 × 40 欄，不同 0，`CMP PASS` |
| (c) sa_N16_s1 | `20261107-155247-icpe-trx-sa_N16_s1-rust-40a4c87f`（1:30.6） | `results/icpe_next/b/sa_main/sa_N16_s1/x1_sweep.csv.xz` | 7,344 列，293,760 格，不同 0，`G0_RESULT IDENTICAL` |

- **HOLD**：
  - icpe-sim 15:5x 通知：旗標開的參考（xref_dev14）在 SemiAnalysis N16 s1、CPU 8、O-ALL、wt＋預取到 GPU、三個旗標全開時失敗。
    - Python：`AssertionError: (12, 277)`（`_g_unprefetch` ← `drop` ← `_turn_end_wt`）。
    - Rust：`gspf 12 277`（main.rs:1099）。
  - 所以 flags-on 的引擎 commit 會改，40a4c87f 不是最後版。icpe-data 也說先不要 merge、不要跑。
  - 我沒有用過任何一個新旗標。
- **swap**（需要使用者決定，見下）：
  - 這次跑的時候 swap 從 1,431 漲到 2,825 MiB，在 15:53:18 → 15:53:48 之間（xc_sa 那一步）。MemAvailable 一直在 56 GiB 以上，swapguard 沒有觸發。
  - 現在換出去的頁幾乎都是別的使用者的閒置行程（例：另一位使用者的 `MainThread` 兩個各 793／710 MiB，跑了將近 2 天；gnome-shell 250 MiB）。我自己的行程合計 VmSwap 17 MiB。
  - 上限 2,933 MiB 是 11/5 09:4x 使用者定的「比基準多 2 GiB」，基準是當時的值。現在離上限只剩 108 MiB，正式 block 很可能一開就被 swapguard 停掉。
  - **使用者決定（15:5x，在本 session 用 AskUserQuestion）**：「改看我自己的 swap (Recommended)」。這個選項的說明是：只算這次 run 那一組行程自己的 VmSwap，超過 256 MiB 就停；可用記憶體低於 6 GiB 就停的規則照舊；別人被換出的頁不算，但只要我的模擬開始用到 swap 就停。
  - **實作**：新守門 `$ROOT/tools/ownswapguard.sh <pgid> <log> 256 5`。
    - 每 5 s 把 `pgrep -g` 那一組行程的 VmSwap 加總，超過 256 MiB → SIGTERM 整組，寫 `OWNSWAP_KILLED`。
    - 每 30 s 記一筆：本組 VmSwap、行程數、整機 swap（只供參考）。
  - `trx_run.sh`、`rux_run.sh`、`relaunch_at_boundary.sh` 改成掛這個守門，取代 `swapguard.sh … 2933`。改之前的版本留在 `*.bak-20261107-swap`；`swapguard.sh` 本身沒刪。
  - 模擬的子行程（Pool worker、常駐的 x1fast）都沒有 setsid，留在同一個行程群組裡，所以會算進去。
  - 試跑 `20261107-155720-icpe-trx-guardtest-ownswap`（只有 `sleep 8`，不是結果）：swap.log 第一筆 `0 2 2825`。
- **分工更新**（icpe-data，STATUS_DATA 15:56，c0da1c65）：TRX40 另外接 Mooncake。理由：只有 SemiAnalysis＋uc/lm 的話，TRX40 只分到約 1/3 的 CPU 時間，而使用者要 TRX40 接重的部分。
  - plan 名稱：
    - SemiAnalysis：`d14_sa_main`、`d14_sa_p1..p3`、`d14_sens_rep:sa_*`、`d14_sa_p4_b1/b2`。
    - Mooncake：`d14_old_main:mooncake`、`d14_old_p1..p3:mooncake_*`、`d14_sens_rep:mc_*`、`d14_p4_b1/b2:conversation_*,toolagent_*`。
    - UltraChat/LMSYS：`d14_chat_main:uc_*,lm_*`、`d14_chat_p1:uc_*,lm_*`。
  - 規模是 icpe-data 的估計：180 個 block，1,697,648 個 job，約 300 執行緒小時，46 workers 約 7 小時。
  - Mooncake 原始檔兩台的 sha256 前 12 碼相同（conversation b8cbb061a852、toolagent 48a2db1a13d3）。
  - 還是 HOLD：icpe-data 要等 icpe-sim 說 xref 過了，才給正式 merge 的 commit。

## 2026-11-07T15:45+08:00 — **使用者決定 TRX40 接偏離 14 的重跑**，以及 P4 版本 C 的 5 組；現在只做準備，還沒跑任何東西

- **來源**：icpe-main 15:3x 轉達使用者在 icpe-main 說的「把 TRX40 把任务交给他，或者是把重一點任务给他」。依規則，我在這裡用 AskUserQuestion 直接問了使用者（15:4x）。答案照抄：
  - 偏離 14 的重跑：「接，照這個流程 (Recommended)」。流程是：
    - icpe-sim push 驗證過的引擎後，同步並編 Rust。
    - 先證明 TRX40 能逐位元重現，才開正式 block。
    - block 由 icpe-data 分配，CacheWise 照舊不放 TRX40。
  - P4 加跑版本 C 的 5 組（偏離 10 第 5 點，清單在 `origin/icpe-next-main` 58252b4f 的 `results/icpe_next/main/p4/verC/p4_cells_verC_add.csv`）：「接，等引擎定了再開 (Recommended)」。要用哪個引擎（雜訊＋偏離 14 規則＋APC/BELP）icpe-sim 還沒定。
- **git 上的狀態（15:3x fetch）**：
  - `origin/icpe-next-sim` 在 0786f56f（STATUS_SIM 14:05：`--pf-expire` 在修好、重新驗證之前不要用）。icpe-main 說的 63b8e32（使用者選 A：`--pf-gpu-recheck` 放在旗標後面）還沒 push。
  - 偏離 14 的正文在 PREREG（fe345222）。重跑清單由 icpe-sim 寫進 STATUS_SIM、icpe-data 排程，現在還沒有。
- **TRX40 上現有的資料**：SemiAnalysis（N16／N32／N64／N72 的 nb）、Mooncake traces、TraceLab、FreeInference、UltraChat、LMSYS-Chat-1M。沒有 CacheWise（不准放），也沒有 Bailian。
- **重現檢查的提案**（開正式 block 前；細節等 icpe-sim 指定）：
  - 旗標關：照以前的三項，(a) G0 對 H200 的 4,416 列；(b) xc_sa 對 icpe-data 的 Python、對 TRX40 的 Python；(c) 整個 `sa_N16_s1` block 對 git 上的正式結果。
  - 旗標開：在 TRX40 有的資料上，Python 對 Rust 逐格比，再加跨機器的參考（請 icpe-sim／icpe-data 提供 acvlab 上同樣參數的輸出）。
- **資源（15:40）**：MemAvailable 55 GiB、/ssd4 剩 509 GiB、load 0.29。

## 2026-11-07T11:40+08:00 — **LMSYS-Chat-1M 版本 B、C 跑完**：下載（使用者登入 HF、同意條款）→ 轉換 → chat_main＋chat_p1 12 個 block → 版本 B 上限表（rowcheck 12/12 PASS）→ 版本 C 6 個變體，重現 324/324、0 差

- **下載**：
  - 第一次 run `20261107-113215-icpe-trx-dl-lmsys` 失敗，rc 1。使用者已登入（`hf auth whoami` → 使用者的 HF 帳號），但帳號還沒取得存取權：`Error: Access denied. Dataset 'lmsys/lmsys-chat-1m' requires approval.` 沒有下載到任何 parquet，我刪掉建出來的目錄（只有 README.md 和 .cache），留 `FAILED.txt`。
  - 使用者在 HF 同意條款之後，run `20261107-113321-icpe-trx-dl-lmsys` 成功，rc 0。6 個 parquet 的 sha256 和位元組數都等於 HF 的 LFS 記錄（`LMSYS_SHA_OK`）。
  - `$ROOT/datasets/lmsys-chat-1m/README.txt` 是 `sha256  data/…parquet` 格式，資料檔設成唯讀。授權禁止轉發，所以原始資料、nb、逐對話的數字都不進 git。
- **轉換**：run `20261107-113408-icpe-trx-conv-chat-lmsys`，repo a65347cb，rc 0，30 s，最大 RSS 1.8 GB。6 個 nb 各 n_req 36,371、n_sess 9,000、chunks 41,085。
  - stdout 照抄：
    - `對話 1000000，合格 334319，抽 9000；{'lt2_user_turns': 665681, 'ok': 334319}`
    - `規則 6：27371/27371 = 1.0000`
    - `規則 6（整段切 vs 逐則切）：{'n': 300, 'median': 0.0, 'p90': 0.0, 'max': 0.004761904761904762, 'frac_exact': 0.99}`
  - span_h：T60 是 4.2／3.2／4.6，T600 是 29.4／27.2／30.0。seed 1 的其他統計已傳給 icpe-data。
- **版本 B 模擬**：`next_b_queue.py run --plans 'chat_main:lm_*' 'chat_p1:lm_*' --engine rust --workers 40`，佇列 log `$ROOT/runs/queue-chat-lm-rust-w40-20261107-113454.log`，11:34:54 → 11:36:46。
  - 12 個 block 都 rc 0，`py_fallback_jobs=0`，`code_sha=3280b1c8799b+rs62b0932b`，沒有 `*KILLED`。npz 照程式存在 `req/`。
  - 資源：MemAvailable 最低 57,312 MiB，swap 1,429 MiB。
  - collect → `results/icpe_next/b/chat_main/lm_*`（各 7,344 列）、`chat_p1/lm_*_ttl`（各 6,480 列）。
- **版本 B 上限表**：run `20261107-113711-icpe-trx-verB-lm_p1`，rc 0。
  - `ROWCHECK lm_main: … 44064 rows, cells 54 cpu 17 gpu 8: PASS`、`ROWCHECK lm_p1_ttl: … 38880 rows, cells 54 cpu 15 gpu 8: PASS`。
  - 1,944 列、violations 168 列，收在 `results/icpe_next/b/analysis/lm_p1/`。
  - 照抄（max／median／n_ge15／n_ge5）：
    - T60 三個 seed：全部 0。
    - T600_s1：O-ALL(report) 0.044064／0.021591／0／0；O-DROP 0.046114；O-GPU 0.044064；O-ORD 0.018689；O-PF 0.027114。
    - T600_s2：O-ALL(report) 0.034217／0.019395／0／0；O-PF 0.019334。
    - T600_s3：O-ALL(report) 0.037636／0.020285／0／0；O-PF 0.022427。
- **版本 C**：rux worktree 4a713386，`rux_verc_chat.sh`（步驟同 UltraChat），參考表＝上面的 `lm_p1_headroom_components.csv`。
  - **使用者 11:37 要求盡量讓 CPU 跑滿**（CPU 上下抖）。
    - 量的原因：一個 block 6,174 個 job 的 Rust 計算合計 89 s（每個 job 約 7 ms）。整個 block 8.8 s × 40 workers＝352 worker 秒，只用了約 25%；其餘是每個 job 的 Python 派送（約 30 ms）、開頭的載入和尾巴的空檔。
    - 所以版本 C 改成 6 個變體同時跑，每個 8 workers，合計 48。`mpstat` 在跑的時候：%nice 86.7、%sys 9.9、%idle 3.1。
  - 6 個 run 是 `20261107-113722`／`113724`／`113725`／`113726`／`113727`／`113728` `-icpe-trx-rux-verC-lmsys_{T60_s1,T600_s1,T60_s2,T600_s2,T60_s3,T600_s3}`，11:37:22 → 11:38:53。每個 14,184 個 job，rc 0，`py_fallback_jobs=0`，`code_sha=3280b1c8799b+rse7f6ee8f`。MemAvailable 最低 56,893 MiB。
  - **重現檢查 6 個都是**（照抄）`版本 B 表上 324 格×元件，對上 324；各欄不同數 {… 全部 0}` → `EXPLORE_RESULT REPRODUCED cells=324`。
  - 版本 C 表（analyze_v8 照抄，max／median）：
    - T60：全部 0.0。
    - T600_s1：O-ALL(report) 0.043990／0.021591，O-DROP 0.046114，O-GPU 0.043990，O-PF 0.026963。
    - T600_s2：O-ALL(report) 0.034217／0.019395，O-PF 0.019180。
    - T600_s3：O-ALL(report) 0.037636／0.020285，O-PF 0.022427。
    - n_ge15 全部 0、n_ge5 全部 0；belp_wins 0。
  - 收在 `results/icpe_next/apc/lmsys_*`，collect_v8 原樣。sha256 對過 MANIFEST；sweep 16,524 列；`code_sha` 只有一個。sweep CSV 沒有任何內容欄，只有模擬結果。
- **TRX40 現在沒有交辦中的工作**。CPU 之後會閒置，要不要接別的由使用者決定。

## 2026-11-07T11:20+08:00 — **UltraChat 版本 B、C 跑完**：轉換（e56c2be0 修好一行兩物件）→ chat_main＋chat_p1 12 個 block → 版本 B 上限表（rowcheck 12/12 PASS）→ 版本 C 6 個變體，重現 324/324、0 差

- **使用者決定（在這邊，AskUserQuestion）**：icpe-data 轉達了使用者在 h200 的決定（PREREG 偏離 13；STATUS_DATA 11:05）：對話資料集的版本 B、C 都由同一台機器跑。使用者在這邊確認 TRX40 接 UltraChat 和 LMSYS 的 B 和 C（選「接，B 和 C 都跑」）。
- **merge**：origin/icpe-next-data e56c2be0 → icpe-next-trx 6667a4ee。
  - code 改兩處：`next_b_convert.py`，UltraChat 一行多個物件用 `raw_decode` 拆開，記在 `filter.load_stats`；`next_b_queue.py`，新 plan `chat_p1`。
  - 模擬器和 Rust src 沒動。
- **轉換**：run `20261107-110557-icpe-trx-conv-chat-ultrachat`，rc 0，57 s，最大 RSS 1.9 GB。寫出 6 個 nb：`ultrachat_T{60,600}_s{1,2,3}`，各 n_req 34,595、n_sess 9,000、chunks 26,349。
  - stdout 照抄：`對話 1468352，合格 1468294，抽 9000；{'ok': 1468294, 'lt2_user_turns': 58}`
  - `規則 6：25595/25595 = 1.0000`
  - `規則 6（整段切 vs 逐則切）：{'n': 300, 'median': 0.0, 'p90': 0.0, 'max': 0.0, 'frac_exact': 1.0}`
  - meta 的 `load_stats.ultrachat_multi_obj_lines`＝3。其他 seed 1 的統計已傳給 icpe-data。
- **版本 B 模擬**：`next_b_queue.py run --plans 'chat_main:uc_*' 'chat_p1:uc_*' --engine rust --workers 40`，佇列 log `$ROOT/runs/queue-chat-uc-rust-w40-20261107-110736.log`，11:07:36 → 11:09:46。
  - 12 個 block 都 rc=0。chat_main 每個 6,174 個 job，chat_p1 `_ttl` 每個 6,480 個。
  - 每個 block 都是 `py_fallback_jobs=0`、`code_sha=3280b1c8799b+rs62b0932b`，沒有 `*KILLED`。資源：MemAvailable 最低 57,240 MiB，swap 1,420 MiB。
  - 逐請求 npz 存在各 run 的 `req/`（每個 block 約 171 MB，不進 git）。icpe-data 說它原本寫「沒有 npz」是寫錯了，照已 commit 的程式存 npz 是對的。
  - collect：`next_b_queue.py collect --plans chat_main chat_p1`，結果在 `results/icpe_next/b/chat_main/uc_*`（每個 7,344 列）和 `chat_p1/uc_*_ttl`（每個 6,480 列）。
- **版本 B 上限表**：run `20261107-111029-icpe-trx-verB-uc_p1`，rc 0。
  - 先跑 `main_rowcheck.py`（icpe-next-main b54d3cff，sha256 cde7fb3f…，解到 `$ROOT/code-archive/main-b54d3cff/`）：
    - `ROWCHECK uc_main: 6 (file, trace, rule), 44064 rows, cells 54 cpu 17 gpu 8: PASS`
    - `ROWCHECK uc_p1_ttl: 6 (file, trace, rule), 38880 rows, cells 54 cpu 15 gpu 8: PASS`
  - 再跑 `next_sim_analyze.py --sweep chat_main/uc_*/… chat_p1/uc_*_ttl/… --out-prefix uc_p1 --pf-target cpu`：headroom_components 1,944 列、violations 192 列。收在 `results/icpe_next/b/analysis/uc_p1/`。
  - 照抄（max／median／n_ge15／n_ge5）：
    - T60 的 3 個 seed：6 個元件全部是 0.000000／0.000000／0／0。
    - T600_s1：O-ALL(report)＝O-DROP＝O-GPU 0.065979／0.051566／0／54；O-ORD 0.026561／0／0／0；O-PF 0.031703／0／0／0；O-NOW 0.000904。
    - T600_s2：O-ALL(report)＝O-DROP＝O-GPU 0.065719／0.051617／0／54；O-ORD 0.026508；O-PF 0.030541；O-NOW 0.000781。
    - T600_s3：O-ALL(report)＝O-DROP＝O-GPU 0.066789／0.053059／0／54；O-ORD 0.025455；O-PF 0.029834；O-NOW 0.000709。
- **版本 C**：rux worktree 移到 4a713386（sweep_v8／explore_apc／analyze_v8／src／Cargo.toml 和現在的 origin/icpe-next-rux 逐位元相同；main.rs 48e4efce）。
  - 機器工具 `$ROOT/tools/rux_verc_chat.sh`：步驟同 run_apc_formal.sh，只是參考表用上面的 `uc_p1_headroom_components.csv`，並查 worktree 是 4a713386、`code_sha` 是 3280b1c8799b+rse7f6ee8f。不存 npz。
  - 6 個 run：`20261107-111046`／`111106`／`111127`／`111147`／`111209`／`111228` `-icpe-trx-rux-verC-ultrachat_{T60_s1,T600_s1,T60_s2,T600_s2,T60_s3,T600_s3}`，11:10:46 → 11:12:50，每個 14,184 個 job，rc=0。
  - **重現檢查 6 個變體都是**（照抄）`版本 B 表上 324 格×元件，對上 324；各欄不同數 {… 全部 0}` → `EXPLORE_RESULT REPRODUCED cells=324`。
  - 版本 C 表（analyze_v8 照抄）：T60 的 3 個 seed 全部 0.0；T600 三個 seed 的每個元件的 max／median／n_ge15／n_ge5 都跟上面的版本 B 相同。
  - O-GPU（④ 含 BELP）：T600 的 max_old＝max_new（0.066／0.0657／0.0668），belp_wins 0；T60 全部 0。
  - 收在 `results/icpe_next/apc/ultrachat_*`，每個 7 個檔，是 collect_v8（4a713386）收的原樣。sha256 對過 MANIFEST；sweep 16,524 列，`code_sha` 只有 3280b1c8799b+rse7f6ee8f。
- **LMSYS-Chat-1M**：還在等使用者在 HF 同意條款並登入 TRX40。登入之後照同樣的步驟做。

## 2026-11-07T11:15+08:00 — 對話資料集準備：tokenizer、tokenizers、UltraChat 好了；**UltraChat 轉換失敗（JSONDecodeError: Extra data）→ 停**，等 icpe-data

- **merge**：origin/icpe-next-data 688ad0bc（icpe-data 的登記，STATUS_DATA「11:00 對話資料集補登記」）→ icpe-next-trx 0142cdb6。
  - code 只改 `next_b_convert.py`（新子命令 `chat`）和 `next_b_queue.py`（plan `chat_main`）。模擬器和 Rust src 沒動。
  - 一起進來的還有 icpe-next-data 已有的 P4 acvlab 結果（b0d84c45）和 STATUS_DATA。
- **tokenizer**：`NousResearch/Meta-Llama-3.1-8B-Instruct` @ d10aef79 的 tokenizer.json，在 `$ROOT/hf-home/hub/models--NousResearch--Meta-Llama-3.1-8B-Instruct/snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77/`。
  - sha256 79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4，跟 icpe-data 給的一樣（run `20261107-105712-icpe-trx-dl-llama31-tokenizer`）。
- **tokenizers 0.23.3**：`$ROOT/tools/env/bin/python -m pip install --no-deps --only-binary=:all: --no-compile --target $ROOT/pylibs/tokenizers-0.23 "tokenizers>=0.23,<0.24"`（run `20261107-105737-icpe-trx-pip-tokenizers`）。共用 venv 沒動（它也沒有 pip）。試切過，vocab 128,256。
- **UltraChat 下載**：
  - 第一次（run `20261107-105811-icpe-trx-dl-ultrachat`）rc=1，是我的指令錯：`hf download … --include 'train_*.jsonl' README.md`。CLI 印 `UserWarning: Ignoring --include since filenames have been explicitly set.`，只下了 README.md，接著 `sha256sum: train_0.jsonl: No such file or directory`。我刪掉它建的目錄（只有 .cache 和 README），失敗的 run 留著（`FAILED.txt`）。
  - 第二次（run `20261107-105843-icpe-trx-dl-ultrachat`）明確列 11 個檔名，rc=0。10 個 `train_*.jsonl` 的 sha256 與位元組數都等於 HF 的 LFS 記錄（`ULTRACHAT_SHA_OK`）。
  - `$ROOT/datasets/ultrachat/README.txt` 照 icpe-data 要的格式（`sha256  相對路徑`，說明行以 `#` 開頭）；資料檔設成唯讀。
- **UltraChat 轉換失敗 → 停**：run `20261107-110038-icpe-trx-conv-chat-ultrachat`（`trx_run.sh`＝§4.1 的殼＋三個守門；repo 0142cdb6），exit 1，16.6 s，最大 RSS 228 MB。
  - 指令：`python -I code/icpe/next_b_convert.py chat --source ultrachat --src-dir $ROOT/datasets --tokenizer <上面的 tokenizer.json> --pylibs $ROOT/pylibs/tokenizers-0.23 --out-dir $ROOT/datasets/x1_nb --seeds 1 2 3`。
  - 完整錯誤（stderr）：
    ```
    File "code/icpe/next_b_convert.py", line 497, in conv_chat
      for i, (cid, msgs) in enumerate(_iter_chat(src, d, files)):
    File "code/icpe/next_b_convert.py", line 451, in _iter_chat
      r = json.loads(ln)
    json.decoder.JSONDecodeError: Extra data: line 1 column 13052 (char 13051)
    ```
  - stdout 是空的（還在第一輪，沒印計數）。沒有寫出任何 nb（`x1_nb` 仍是 27 個檔）。
  - **原因**（唯讀檢查，scratchpad 的 `find_bad_jsonl.py` 逐行 `json.loads`）：1,468,349 行裡有 3 行各有兩個 JSON 物件接在一起（`…"]}{"id"…`），後面沒有別的字元：
    - train_3.jsonl 第 127,819 行（第一個物件 id 578253，錯在第 13,052 欄，就是這次的錯誤）
    - train_4.jsonl 第 82,900 行（id 683335，第 9,289 欄）
    - train_7.jsonl 第 90,760 行（id 1141196，第 15,658 欄）
    - 其他 7 個檔 0 行。檔案本身跟 HF 的 sha256 一樣，所以是上游資料就這樣，不是下載壞掉。
  - 要怎麼處理（例如一行拆成兩個物件，或算不合格）是轉換規則的問題，等 icpe-data 決定、改程式、commit 之後再跑。我沒改程式，也沒有繞過去。
- **LMSYS-Chat-1M**：還沒下載，等使用者在 HF 同意條款並登入 TRX40。
- 機器工具新增 `$ROOT/tools/trx_run.sh`（主 repo 版的 rux_run.sh：§4.1 的殼、nice 10、memwatch 6,144／swapguard 2,933／diskguard 50 GiB）。

## 2026-11-07T11:05+08:00 — W5（PREREG 偏離 11 A）tracelab_c1、FreeInference 05-17 跑完；base 跟版本 C 在 3 個容量逐值相同；使用者決定 TRX40 接 LMSYS-Chat-1M＋UltraChat（W5 之後準備，登記前不跑模擬）

- **W5 run**（rux worktree d35aed35，`run_w5_formal.sh … 40`，各 4,728 個 job，`engine=rust`、`py_fallback_jobs=0`、`code_sha=3280b1c8799b+rse7f6ee8f`）：
  - `20261107-104717-icpe-trx-rux-tl_c1-w5`：10:47:18 → 10:55:10，rc=0，sweep 465 s。BF16 那一半＝`20261107-100449-icpe-trx-rux-tl_c1-apcformal` 的 sweep（5,508 列）。
  - `20261107-105510-icpe-trx-rux-fi_w1-w5`：10:55:10 → 10:55:38，rc=0，sweep 5.4 s。BF16＝`20261107-095548-icpe-trx-rux-fi_w1-apcformal` 的 sweep（5,508 列）。
  - 兩個都印 `W5_FORMAL_DONE`，沒有 `*KILLED`。資源：MemAvailable 最低 46,978 MiB，swap 最高 1,394 MiB，PSS 最高 9,255 MiB。
- **核對 base＝版本 C**（scratchpad 的 `cmp_w5_base.py`，先用 icpe-rux 的 semianalysis_N16_s1 試過 108／108 相同）：W5 的 `a/base_headroom_components.csv` 對 `results/icpe_next/apc/<變體>/verC_headroom_components.csv` 在 111.0625／128／256 GiB，以 13 個鍵配對，13 個值欄逐字串比：
  - tracelab_c1：108 列全部配上，各欄不同數全部 0 → `W5_BASE_VS_VERC IDENTICAL rows=108`。
  - fi_2026-05-17_2026-05-23：同上，108 列，全部 0 → `IDENTICAL`。
- **結果**（explore_w5 和 w5_mechanism 的輸出照抄，不解讀）：
  - tracelab_c1，各元件 18 格的 max（base／forced／allowed）與 ≥15% 格數（base／forced／allowed）：
    - O-ALL(report)：0.0905／0.0076／0.0076，0／0／0
    - O-GPU：0.1704／0.0151／0.0151，2／0／0
    - O-ORD：0.2188／0.0449／0.0449，2／0／0
    - O-PF：0.0220／−0.0026／−0.0026，0／0／0
    - O-DROP：0.0236／0.0245／0.0245，0／0／0
    - O-NOW：0.0006／0.0004／0.0004，0／0／0
    - `W5 較快的組合：5508／5508`。
    - w5_mechanism（allowed 比 base 升高 > 5 個百分點）：5 個（格, 元件）。O-ALL(report) 中位升高 59.42 個百分點（b32 1 格、nfs 2 格），O-PF 50.29 個百分點（nfs 2 格）。
  - fi_2026-05-17_2026-05-23：每個元件的 max 都 ≤ 0.0001，≥15% 都是 0；`W5 較快的組合：5113／5508`；w5_mechanism 0 列。
- **收進 repo**：`results/icpe_next/w5/{tracelab_c1,fi_2026-05-17_2026-05-23}/`，各 8 個檔，是 run_w5_formal.sh 自己用 collect_v8（d35aed35）收的，原樣複製。
  - sha256 都對過 MANIFEST。
  - `x1_sweep_bf16.csv.xz` 的 run_id 是 apcformal 的 run，不是 W5 的 run，不在 MANIFEST 的 runs 裡，但寫在 MANIFEST 的 note（BF16 sweep 的路徑）。icpe-rux 的 semianalysis_N16_s1 也是這樣。
- **使用者決定（在這邊，AskUserQuestion）**：icpe-data 轉達，11/9 前補跑公開對話資料集，TRX40 負責 **LMSYS-Chat-1M** 和 **UltraChat**。
  - 使用者選「接，但等 W5 跑完再準備」，兩個都是。
  - 範圍：只跑 `chat_main` 主網格，時間是合成的，結果單獨一節。
  - icpe-data 登記、commit 並 push、傳完整指令之前，不跑模擬。
  - LMSYS-Chat-1M 是 gated，要使用者在 HF 同意條款並在 TRX40 登入（`HF_HOME=$ROOT/hf-home`）。目前 TRX40 沒有 HF token。
  - 原始資料、轉換後的紀錄都不進 git。

## 2026-11-07T10:55+08:00 — 流程跑完（PIPE_ALL_DONE 10:46:26）：**9 個變體的 9 容量版本 B 重現全部 324／324 格×元件、7 欄 0 差**；W3 再判 8 個變體（照抄）；偏離 10 正式 APC 9 個變體收進 repo；W5（偏離 11）開跑

- **流程**：背景殼 09:06 → 10:46:26 印 `PIPE_ALL_DONE`，exit 0。19 個 run 的 `exit_code` 全是 0，沒有 `*KILLED`。
  - 資源（全部 run 的 mem.log）：MemAvailable 最低 42,786 MiB，swap 最高 1,394 MiB，行程群組 PSS 最高 14,003 MiB；/ssd4 剩 524 GiB。
  - 引擎：每個 sweep 都是 `engine=rust`、`py_fallback_jobs=0`、`code_sha=3280b1c8799b+rse7f6ee8f`（v8 main.rs 48e4efce）。
- **重現檢查**（explore_apc.py 對版本 B 表；TraceLab 對 `tl_p1_headroom_components` blob 48da5cd4，FreeInference 對 `fi_p1_headroom_components` blob 8322cf03）。每一行都是照抄 `重現檢查：版本 B 表上 324 格×元件，對上 324；各欄不同數 {… 全部 0}` → `EXPLORE_RESULT REPRODUCED cells=324`：

  | 變體 | run | worktree | 備註 |
  |:--|:--|:--|:--|
  | tracelab_c2 | `20261107-090206-icpe-trx-rux-tl_c2-reproB`（13,518 job） | 8f04a91c | W3 的先決條件 |
  | tracelab_c4 | `20261107-092415-icpe-trx-rux-tl_c4-apcformal`（14,184 job） | 7a89a9b5 | 正式 APC 第 2 步，兼 W3 先決條件 |
  | fi 第 1–6 週 | `20261107-095548`／`095646`／`095752`／`095837`／`095919`／`100116` `-icpe-trx-rux-fi_w{1..6}-apcformal`（各 14,184 job） | 7a89a9b5 | 同上 |
  | tracelab_c1 | `20261107-100449-icpe-trx-rux-tl_c1-apcformal` | 7a89a9b5 | 第二次核對（第一次＝108＋216，09:10 節） |
  | tracelab_c2 | `20261107-102412-icpe-trx-rux-tl_c2-apcformal` | 7a89a9b5 | 第二次核對 |

- **W3**（偏離 9；analyze_w3 的判定照抄，不解讀）。cue：TraceLab 用 08:30 節做好的 `tracelab_c{2,4}.cue.u8`；FreeInference 用 `make_cue.py fi`（run `20261107-095233-icpe-trx-rux-fi-cue`，6 週的 n_req／n_cue＝10,738／9,953、12,827／11,859、7,324／6,828、6,859／6,422、35,659／34,174、56,310／53,594）。每個 W3 是 360 個 job。

  | 變體 | W3 run | n_ge15 | cue 齊全 | f ≤ 0.5 | 其中飽和 | f 中位數 | f 最小 | 判定（照抄） |
  |:--|:--|--:|--:|--:|--:|--:|--:|:--|
  | tracelab_c2 | `20261107-092256-…-tl_c2-w3` | 36 | 36 | 36 | 0 | −5.061049 | −9.376651 | 後半成立：沒有簡單線索能拿走大部分 |
  | tracelab_c4 | `20261107-095115-…-tl_c4-w3` | 61 | 61 | 61 | 0 | −2.603243 | −5.792954 | 後半成立：沒有簡單線索能拿走大部分 |
  | fi 第 1–6 週 | `…-fi_w{1..6}-w3`（095625／095727／095822／095906／100031／100326） | 0 | 0 | 0 | 0 | NaN | NaN | 前半不成立（沒有上限 ≥15% 的格） |

  - 各元件（≥15% 的格）的 f 中位數與 f > 0.5 的個數（照抄 analyze_w3 的表）：
    - c2：O-GPU −5.003（24 格，0 個 > 0.5）、O-ORD −5.112（10，0）、O-PF −6.126（2，0）；O-ALL(report) −4.005（30，0）。③ 選到 CUEO（54 格）。
    - c4：O-GPU −2.460（36，0）、O-ORD −2.623（20，0）、O-PF −3.930（5，0）；O-ALL(report) −1.971（37，0）。③ 選到 CUEO（54 格）。
    - FreeInference 6 週：每個元件 n_ge15＝0；③ 都選到 CUEP（54 格）。
  - 先決條件都滿足：c2 的 W3 在 9 容量重現通過之後才跑；c4 和 FreeInference 的 W3 在各自正式 APC 的重現通過之後才跑。
- **偏離 10 正式 APC（版本 C）**：analyze_v8 的表照抄（每個變體 54 格；欄＝max、median、n_ge15、n_ge5）：

  | 變體 | O-ALL(report) | O-GPU | O-ORD | O-PF |
  |:--|:--|:--|:--|:--|
  | tracelab_c1 | 0.153024／0.034788／1／22 | 0.392314／0.074359／16／33 | 0.218800／0.023014／4／21 | 0.137512／0.016512／0／16 |
  | tracelab_c2 | 0.323644／0.144924／24／33 | 0.462917／0.130289／24／44 | 0.272358／0.055978／8／28 | 0.218775／0.027792／3／24 |
  | tracelab_c4 | 0.513531／0.278803／36／44 | 0.487513／0.243290／36／54 | 0.349862／0.095379／18／35 | 0.221250／0.019288／5／20 |
  | fi 05-17 | 0.000768／0.000032／0／0 | 0.000915／0.000079／0／0 | 0.000014／0／0／0 | 0.000682／0／0／0 |
  | fi 05-24 | 0.002221／0.000314／0／0 | 0.003457／0.000520／0／0 | 0.001541／0／0／0 | 0.000339／0／0／0 |
  | fi 05-31 | 0.000117／0.000038／0／0 | 0.000117／0.000038／0／0 | 0.000054／0／0／0 | 0／0／0／0 |
  | fi 06-07 | 0.002192／0.000214／0／0 | 0.002457／0.000286／0／0 | 0.000595／0／0／0 | 0／0／0／0 |
  | fi 06-14 | 0.007109／0.000666／0／0 | 0.007257／0.000820／0／0 | 0.002063／0.000206／0／0 | 0.004935／0.000366／0／0 |
  | fi 06-21 | 0.010521／0.001505／0／0 | 0.009889／0.001575／0／0 | 0.007495／0.000758／0／0 | 0.008460／0.000748／0／0 |

  - O-DROP、O-NOW 的列見各變體的 `verC_headroom_components.csv`。
- **收進 repo 的檔**（這個 commit）：
  - `results/icpe_next/apc/<變體>/`，9 個變體 × 7 個檔，是 run_apc_formal.sh 自己用 collect_v8（7a89a9b5）收的，原樣複製；`x1_sweep.csv.xz` 每個 16,524 列。
    - TraceLab 三個的 `x1_sweep.csv.xz` 是 1.4–1.5 MB，超過 CLAUDE.md 的 1 MB 判準；照 icpe-rux 在 d35aed35 進 git 的 `apc/semianalysis_N16_s1/x1_sweep.csv.xz`（1.2 MB）的做法放進來。
  - `results/icpe_next/v8_explore/w3/<變體>/`，8 個新變體（c2、c4、FreeInference 6 週），用 c0e06332 的 collect_v8 收：x1_sweep.csv.xz（1,080 列）、w3_rows.csv（324 列）、w3_verdict.csv、cue.json。MANIFEST 的 runs 列了重現的 run（和 FreeInference 的 cue run）。
  - 核對：每個檔的 sha256 等於 MANIFEST 的 `sha256_source`（xz 解壓後比）；每個 CSV 有 run_id、ts（分析 CSV 另有 host），run_id 都在 MANIFEST 的 runs 裡；列數等於 MANIFEST。
- **W5（PREREG 偏離 11，icpe-rux d35aed35）**：使用者核可的 TraceLab／FreeInference 範圍內，所以沒再問。
  - 10:47 worktree 從 7a89a9b5 移到 d35aed35。這兩版之間 `code/` 只改 5 個 W5／排隊的檔（explore_queue.py、explore_w5.py、run_queue_formal.sh、run_w5_formal.sh、w5_mechanism.py）；sweep_v8、run_apc_formal 等 0 差。
  - 依序跑：`20261107-104717-icpe-trx-rux-tl_c1-w5`（`run_w5_formal.sh nb:tracelab_c1 nb tracelab_c1 <tl_c1-apcformal 的 x1_sweep.csv> 40`，4,728 個 job），通過 `W5_FORMAL_DONE` 才接 fi 第 1 週（BF16 那一半用 `fi_w1-apcformal` 的 sweep）。
  - 跑完要核對：W5 的 `base_headroom_components.csv` 在 111.0625／128／256 GiB 跟 `apc/<變體>/verC_headroom_components.csv` 逐值相同。

## 2026-11-07T09:15+08:00 — 偏離 10 的不存 npz 開關好了（7a89a9b5）；流程改成「正式 APC（兼 9 容量重現）→ W3」

- **icpe-rux 的修改**：origin/icpe-next-rux 7a89a9b5，只改 `run_apc_formal.sh`。
  - 預設不存逐請求 npz（`RUN_DIR` 空字串）；`APC_NPZ=1` 才存。
  - 從 8f04a91c 到 7a89a9b5，以及到更新的 199b0610：sweep_v8、analyze_w3、explore_apc、analyze_v8、make_cue、v8 `src`／`Cargo.toml`、v6_slowcap、`code/p3x`、next_sim_analyze 的 diff 都是 0 行。
  - 只多了 `collect_v8.py`（engine_used、`--base`）和 `run_apc_formal.sh`。
- **流程調整**（09:05）：
  - `rux_formal_w3.sh`（機器工具）：每個變體先跑 `run_apc_formal.sh`。它的第 2 步是 9 個容量的版本 B 重現（16 個規則，7 欄逐值相同），同時算版本 C、收進 `results/icpe_next/apc/<變體>/`。印 `EXPLORE_RESULT REPRODUCED` 和 `APC_FORMAL_DONE` 才接 W3。
  - 這樣 c4 和 FreeInference 不用另跑一次 16 個規則的重現。
  - 09:05:28 停掉舊流程的外層殼（pid 2432467，SIGTERM）。那時它正在等 c2，不是失敗。
  - c2 的 `rux_repro_w3.sh`（pid 2435770）沒停，繼續跑：9 容量重現 run `20261107-090206-…-tl_c2-reproB` → 通過才接 W3，worktree 仍在 8f04a91c。
- **新流程**（背景，09:06 開始）依序：
  1. 等 c2 跑完；確認 reproB 是 `REPRODUCED`、W3 的 rc 是 0，否則停。
  2. worktree 移到 7a89a9b5。
  3. tl_c4：正式 APC → W3。
  4. `make_cue.py fi`（6 週）。
  5. FreeInference 第 1–6 週：正式 APC → W3。
  6. tl_c1、tl_c2：正式 APC（W3 已做；c2 會再做一次 16 個規則的重現，當二次核對）。
- 任何一步失敗或沒重現就停，回報 icpe-main 和 icpe-rux。

## 2026-11-07T09:10+08:00 — **tracelab_c1 的 9 容量重現檢查補完：324／324 格×元件 0 差** → c1 的 W3 先決條件滿足；**使用者決定 TRX40 接偏離 10 的正式 APC 重算，但不存 npz**

- **偏離 9 補記**：origin/icpe-next-rux aec99d11（icpe-rux 寫）。W3 前的版本 B 重現要涵蓋全部 9 個容量。
  - `git diff 8f04a91c aec99d11` 在 sweep_v8、explore_apc、analyze_w3、make_cue、v8 src、`code/p3x`、next_sim_analyze 上是 0 行。
- **c1 補 6 個容量**：run `20261107-084950-icpe-trx-rux-tl_c1-reproB`，rc=0，08:49:50 開始。
  - worktree 8f04a91c；版本 B 的規則組合（17 × 16），CPU 16／32／64／128／256／512。
  - 9,012 個 job，732 s，40 workers，`py_fallback_jobs=0`，`code_sha=3280b1c8799b+rse7f6ee8f`。
  - 資源：MemAvailable 最低 47,191 MiB，PSS 最高 9,446 MiB，swap 最高 1,385 MiB。
  - 重現檢查（照抄）：`重現檢查：版本 B 表上 216 格×元件，對上 216；各欄不同數 {'T_oracle': 0, 'T_rival': 0, 'headroom': 0, 'best_rival': 0, 'best_rival_gpu_policy': 0, 'oracle_policy': 0, 'oracle_gpu_policy': 0}` → `EXPLORE_RESULT REPRODUCED cells=216`。
  - **c1 合計**：108（CPU 8／111.0625／1024，run `…-083210-…-tl_c1-apc`）＋216＝**324 格×元件，7 欄全部 0 差**，涵蓋 9 個容量 × ssd／b32／nfs × 閒／忙 × Cake × 6 個元件。
  - **所以 c1 的 W3 先決條件（偏離 9＋補記）滿足。** 08:55 節照抄的 analyze_w3 判定字（「後半成立：沒有簡單線索能拿走大部分」）不再是「待確認」。W3 的 run 和檔都沒有重跑或改動。
  - `w3/tracelab_c1/MANIFEST.json` 用 c0e06332 的 collect_v8 重收，runs 列了三個 run（3 容量重現、6 容量重現、W3）。其他 4 個檔逐位元相同。
- **使用者決定（在這邊，AskUserQuestion）**：TRX40 接 **PREREG 偏離 10**（origin/icpe-next-rux 935dcd75，腳本 82af61df `run_apc_formal.sh`）的正式 APC 重算。
  - 對象：TraceLab c1／c2／c4、FreeInference 6 週；版本 C，結果放 `results/icpe_next/apc/<變體>/`。
  - **但不存逐請求 npz**：腳本寫死 `RUN_DIR=$RUN/req`，x1_sim 會對每個 job `np.savez_compressed`（每個時間設定存 ttft、rest 兩個陣列）。
    - 〔推算〕每列約 2 × 665,453 × 8 B ≈ 10.6 MB；c1 在 9 個容量約 16,500 列，壓縮前約 175 GB。每個 TraceLab 變體約 120–175 GB，3 個就超過 /ssd4 剩的 541 GiB。
  - 已請 icpe-rux 加不存 npz 的開關。加好、我核對 code diff 之後才跑。
  - 在那之前照目前流程跑（9 容量版本 B 重現 → W3）。
- **流程**：c2 的 9 容量重現 run `20261107-090206-icpe-trx-rux-tl_c2-reproB` 09:02:07 開始，`jobs=13518`（stdout）。

## 2026-11-07T09:00+08:00 — **使用者決定：W3 的前置重現檢查要涵蓋全部 9 個 CPU 容量** → **c1 的 W3 判定改為「待確認」**；c2 的 3 容量檢查停掉；改用新流程

- **使用者的決定**（11/7 約 08:47，在 icpe-main，icpe-main 轉達）：
  - 偏離 9 第 3 點要「同一個變體重算版本 B 的表、逐值相同」，W3 才能判。
  - c1 只檢查了 CPU 8／111.0625／1024（108 格×元件）。但 c1 判定用的 22 個 ≥15% (格, 元件) 裡，有 14 個在 16／32／64／128 GiB，不在檢查範圍內。
  - 要做的：
    1. c1 補 16／32／64／128／256／512 GiB，版本 B 的規則組合（17 × 16），對 `tl_p1_headroom_components`（blob 48da5cd4）。合計應是 324 格×元件、0 差。**W3 不用重跑。**
    2. c2、c4 和 FreeInference 6 週：重現檢查涵蓋全部 9 個容量（FreeInference 對 `fi_p1_headroom_components`）。
    3. 補完之前，`analyze_w3` 的結果只能當「待確認」。
    4. 任何一格不同就停，回報 icpe-main 和 icpe-rux。
- **所以**：
  - **08:55 節的 tracelab_c1 W3 判定字（「後半成立」）目前只算「待確認」，不是判定**，要等 c1 的 6 個容量補完、324 個格×元件 0 差。
  - 已 commit 的檔（9e1064f0、16ba40f6）不改。
- **停掉的 run**：`20261107-084318-icpe-trx-rux-tl_c2-repro`（舊流程的 c2，3 個容量）。
  - 08:49:29 我送 SIGTERM 給 pgid 2430140，當時做到 1,843／4,506 個 job，`exit_code` 143。這是我刻意停的，不是失敗。
  - 原因：舊流程（`rux_variant.sh`）在 3 容量通過後會直接接 W3，不符合新決定。
  - 這個 run 的結果不用。
- **新流程**（`$ROOT/tools/rux_repro_w3.sh`，機器工具；08:50 開始，背景執行）：
  - 每個變體：版本 B 規則組合在指定容量 sweep → `explore_apc.py` → 印 `EXPLORE_RESULT REPRODUCED` 才接 W3；任何一步失敗或沒重現就停。
  - 依序：
    1. `tl_c1`：只補 6 個容量（16 32 64 128 256 512），不跑 W3。
    2. `tl_c2`、`tl_c4`：9 個容量重現 → W3。
    3. `make_cue.py fi`（6 週）。
    4. FreeInference 第 1–6 週：9 個容量重現 → W3（`--traces fi:<週> --rules session_id`）。
  - FreeInference 第一週的完整 APC（含 BELP）和 W5 排在這之後，照 icpe-rux 的優先順序。

## 2026-11-07T08:55+08:00 — **W3 tracelab_c1 完成**（PREREG 偏離 9）；`analyze_w3` 的判定字：「後半成立：沒有簡單線索能拿走大部分」；收進 `v8_explore/w3/tracelab_c1/`；c2、c4 開跑

- **run** `20261107-084104-icpe-trx-rux-tl_c1-w3`：rc=0，08:41:04 → 08:42:19。
  - worktree 8f04a91c。偏離 9 的 commit 時間是 08:37:10，這個 run 在它之後開始。
  - sweep：360 個 job，71.9 s，40 workers，`py_fallback_jobs=0`，`code_sha=3280b1c8799b+rse7f6ee8f`；`rust_timing.jsonl` 360 行。
  - `RUX_CUE_FILE=$ROOT/rux-trx/cue/tracelab_c1.cue.u8`（`CUE_CHECK PASS`，08:40 節）。
  - 資源：MemAvailable 最低 46,263 MiB，PSS 最高 11,009 MiB，swap 1,381 MiB。
- **先決條件**：同一變體的版本 B 重現檢查 108／108 0 差（run `20261107-083210-…-tl_c1-apc`，08:50 節）。
- **`analyze_w3.py` 的輸出**（照抄；③ 選到的 GPU 規則：CUEO 54 格）：

  | 元件 | n | n_ge15 | cue_beats_rival | f_c_med | f_c_max | f_c_med_ge15 | n_f_gt_half_ge15 |
  |:--|--:|--:|--:|--:|--:|--:|--:|
  | O-ALL(report) | 54 | 15 | 0 | −8.273 | 21.238 | −6.209 | 0 |
  | O-DROP | 54 | 0 | 0 | −244.121 | −32.184 | NaN | 0 |
  | O-GPU | 54 | 18 | 0 | −10.929 | −3.356 | −7.413 | 0 |
  | O-NOW | 54 | 0 | 0 | −1085.825 | 8189.312 | NaN | 0 |
  | O-ORD | 54 | 4 | 0 | −38.622 | −2.623 | −5.131 | 0 |
  | O-PF | 54 | 0 | 0 | −29.344 | 327.587 | NaN | 0 |

  - f_NP43 的中位數 −5.96。
  - **判定（偏離 9）**：
    - n_cells_judged 270；≥15% 的 (格, 元件) 22 個，③ 齊全 22 個；
    - 其中 f ≤ 0.5 的 22 個，寫入飽和的 0 個；
    - f_c 中位數 −7.4087、最小 −8.9094。
    - **verdict＝「後半成立：沒有簡單線索能拿走大部分」**。
  - `W3_RESULT cells=54 rows=324`。
  - 這是登記程式算出的判定字，照抄。推翻條件整體怎麼判、原始／合成怎麼合，由 icpe-main／icpe-rux 依偏離 9 做，我不解讀。
- **收集**：`collect_v8.py w3 tracelab_c1`（worktree 8f04a91c），檔案在 `results/icpe_next/v8_explore/w3/tracelab_c1/`：
  - `x1_sweep.csv.xz`、`w3_rows.csv`（324 列）、`w3_verdict.csv`（1 列），CSV 都有 run_id、ts、host；
  - `tracelab_c1.cue.json`；
  - `MANIFEST.json`，runs 列了重現檢查的 run 和 W3 的 run。
- **接著**：c2、c4 各自先跑版本 B 設定的重現檢查（17 × 16 規則，CPU 8／111.0625／1024，`explore_apc.py`），印 `EXPLORE_RESULT REPRODUCED` 才接 W3；沒通過就停。
  - 用 `$ROOT/tools/rux_variant.sh`（機器工具）依序跑。
  - `rux_run.sh` 加了 `X1_FI_DIR=$ROOT/datasets/freeinference_agentic_trace/release`，給之後的 FreeInference 用。

## 2026-11-07T08:50+08:00 — **TraceLab c1 的版本 B 重現檢查通過（108／108 格×元件，各欄 0 差）**；APC／BELP 探索收進 `v8_explore/apc/tracelab_c1/`；c1 的 W3 開跑

- **run** `20261107-083210-icpe-trx-rux-tl_c1-apc`：rc=0，08:32:10 → 08:39:59，worktree 2b859912。
  - sweep：4,728 個 job，384 s，40 workers，`py_fallback_jobs=0`；`rust_timing.jsonl` 4,728 行；`x1_sweep.csv` 5,508 列。
  - `code_sha=3280b1c8799b+rse7f6ee8f`：`rs` 後面是 v8 的 `Cargo.toml`＋`src/main.rs` 的 sha256（照 `x1_rust_v6.src_sha()` 重算，同值）。`main.rs` sha256 是 `48e4efce…`。
  - 資源：MemAvailable 最低 43,198 MiB，PSS 最高 13,528 MiB，swap 一直 1,381 MiB（沒有再漲），load1 最高 41.84。
- **重現檢查**（`explore_apc.py` 的第一行，照抄）：
  `重現檢查：版本 B 表上 108 格×元件，對上 108；各欄不同數 {'T_oracle': 0, 'T_rival': 0, 'headroom': 0, 'best_rival': 0, 'best_rival_gpu_policy': 0, 'oracle_policy': 0, 'oracle_gpu_policy': 0}` → `EXPLORE_RESULT REPRODUCED cells=108`。
  - 對照表是版本 B 的 `tl_p1_headroom_components`（blob 48da5cd4），CPU 8／111.0625／1024 三個容量。
  - **這表示 TRX40 的 TraceLab 轉換＋v8 引擎算出的版本 B 表，和 acvlab 的逐值相同。偏離 9 對 tracelab_c1 的先決條件滿足。**
- **APC／BELP 的摘要表**（`explore_apc.py` 印的，restore＝cake，每個元件 18 格；探索，不是正式結果，我不解讀）：

  | 元件 | max_base | max_apc | n15_base | n15_apc | med_d_rival | med_d_hr_pt | rival_apc | oracle_apc |
  |:--|--:|--:|--:|--:|--:|--:|--:|--:|
  | O-ALL(report) | 0.4137 | 0.0905 | 4 | 0 | −0.0419 | −3.9279 | 17 | 0 |
  | O-DROP | 0.0130 | 0.0136 | 0 | 0 | −0.0679 | 0.1447 | 18 | 18 |
  | O-GPU | 0.3895 | 0.0514 | 6 | 0 | −0.0419 | −4.0439 | 17 | 0 |
  | O-NOW | 0.0007 | 0.0007 | 0 | 0 | −0.0679 | 0.0048 | 18 | 18 |
  | O-ORD | 0.2329 | 0.2188 | 2 | 2 | −0.0666 | 0.0007 | 18 | 18 |
  | O-PF | 0.0601 | 0.0780 | 0 | 0 | −0.0419 | 0.9114 | 17 | 18 |

  - O-GPU 的 ④ 加上 BELP：18 格，max_old 0.3895 → max_new 0.3923，med 0.0754 → 0.0615，BELP 比 BEL 快的有 18 格。
- **收集**：用 `collect_v8.py apc tracelab_c1`（worktree 2b859912 時執行），檔案在 `results/icpe_next/v8_explore/apc/tracelab_c1/`，共 596K：
  - `x1_sweep.csv.xz`（5,508 列）；
  - `explore_apc_rows.csv`（108）、`explore_ogpu_belp_rows.csv`（18）、`base_headroom_components.csv`（108）、`apc_headroom_components.csv`（108），每個 CSV 都有 run_id、ts、host；
  - `MANIFEST.json`（repo_commit 2b859912，v8_main_rs_sha256 48e4efce…，sweep_code_sha `3280b1c8799b+rse7f6ee8f`）。
- **icpe-rux 的 c2、c4、FreeInference 第 2–6 週**：重現檢查只跑版本 B 的設定（17 × 16 個規則，CPU 8／111.0625／1024），然後跑 `explore_apc.py`。c1 和 FreeInference 第一週跑完整 APC（含 BELP）。
- **偏離 9 更正**：origin/icpe-next-rux e7e0c778，只改 PREREG 的文字，`git diff 8f04a91c e7e0c778 -- code` 是 0 行。worktree 留在 8f04a91c。
- **W3 tracelab_c1 開跑**：run `…-icpe-trx-rux-tl_c1-w3`。
  - worktree 8f04a91c；`main.rs` sha256 仍是 48e4efce…。
  - `RUX_CUE_FILE=$ROOT/rux-trx/cue/tracelab_c1.cue.u8`；CPU 9 個容量 × NON_ORACLE 10 × {CUEP, CUEO} × ssd／b32／nfs，40 workers。
  - 跑完接 `analyze_w3.py --headroom tl_p1_verB.csv`。

## 2026-11-07T08:45+08:00 — W3 的規則已登記（PREREG 偏離 9，origin/icpe-next-rux 8f04a91c）；**更正 08:40 節的時間**；W3 等 c1 的重現檢查通過才開始

- **更正**（icpe-main 來訊）：08:40 節寫「使用者 11/7 約 08:45 決定 W3 先登記才跑」，時間是照抄 icpe-main 訊息的錯。實際是 acvlab 本地時間約 08:28–08:30。舊節不改。
  - 先後順序不變：使用者決定 → icpe-rux 登記偏離 9 → TRX40 才開始 W3。
- **登記已確認**：
  - commit `8f04a91ccc022d9aceacec52f6691b0a370fe134`，commit 時間 2026-11-07T08:37:10+08:00。
  - `git branch -r --contains 8f04a91c` → `origin/icpe-next-rux`。
  - 內容：PREREG.md 檔尾的「偏離 9：W3 線索 (a) 的規則」，以及 `analyze_w3.py`（輸出 `w3_rows.csv`、`w3_verdict.csv`）。
  - 2b859912 → 8f04a91c 在 `code/` 下只改了 `analyze_w3.py`。v8 `src/main.rs` 的 sha256 仍是 `48e4efcee98cb5e5…`，和 worktree 現在的相同。
- **到目前還沒跑任何 W3 sweep 或 analyze_w3。**
- **偏離 9 的先決條件**：每個變體跑 ③ 之前，要先用 `explore_apc.py` 對版本 B 重現，0 個不同；不同就停，不判。
  - tracelab_c1 的 APC（run `20261107-083210-icpe-trx-rux-tl_c1-apc`）還在跑：4,728 個 job，08:38 時做完 3,570 個。
  - 跑完、重現檢查 0 差之後，worktree 才移到 8f04a91c，跑 c1 的 W3。
  - c2、c4、FreeInference 的重現檢查怎麼跑（只跑版本 B 的設定，或完整 APC），已問 icpe-rux；沒回的話，c2、c4 先只跑版本 B 的設定。

## 2026-11-07T08:40+08:00 — 轉換、cue 都完成（CUE_CHECK 3／3 PASS，c1 和 icpe-rux 的值相同）；**W3 照使用者決定暫停，等 PREREG 登記**；TraceLab c1 的 APC／BELP 開跑

- **使用者的決定（第 2 次）**：在這邊用 AskUserQuestion 問。
  - 結果改照 icpe-rux 的 `results/icpe_next/v8_explore/<實驗>/<變體>/`（origin/icpe-next-rux 2b859912 的 README）。
  - sweep 的 `x1_sweep.csv.xz` 也進 git，另寫 `MANIFEST.json`，用 `collect_v8.py` 收。
  - 取代 08:30 寫的 `rux_trx/`。
- **使用者的決定（11/7 約 08:45，在 icpe-main，icpe-main 轉達）**：**W3 先登記才跑**。
  - TraceLab c1／c2／c4 和 FreeInference 的 W3 sweep（CUEP／CUEO）、`analyze_w3` 都**不跑**，等 icpe-rux 把 W3 的規則登記進 PREREG，並通知 commit 號碼。
  - 下載、轉換、make_cue、CUE_CHECK、APC／BELP、W5 照常。
- **0b. 轉換**：run `20261107-082523-icpe-trx-rux-tl-convert`，rc=0，08:25:23 → 08:31:09。
  - c1、c2、c4 都是 n_req 665,453、n_sess 8,058、chunks 222,667,849；span_h 分別是 7,282.8／4,258.9／2,994.8。輸出在 `$ROOT/datasets/x1_nb_rux/`。
  - 記憶體：MemAvailable 最低 46,428 MiB，PSS 最高 11,838 MiB。
  - **swap 從 790 漲到 1,381 MiB**，還在 2,933 MiB 上限內。可用記憶體一直有 46 GiB 以上，應該是核心換出閒置頁〔推論〕。之後繼續盯。
- **worktree 移到 2b859912**（08:31，轉換之後、cue 之前）。
  - `git diff --stat b7444163 2b859912 -- code`：只有 `collect_v8.py`（新檔）和 `explore_queue.py`（我不用）。
  - 轉換用的 `next_b_convert.py` 兩版相同。
- **0c. cue**：run `20261107-083128-icpe-trx-rux-tl-cue`，rc=0。
  - `make_cue.py tracelab --compress 1 2 4`，然後 `check` 三次：**`CUE_CHECK PASS`** tracelab_c1／c2／c4，n_req nb=665,453＝cue。
  - tracelab_c1 的 `.cue.json`：n_req 665,453、n_cue 574,485、sha_t `a761221af4812de6…`、sha_sess `461381d11e93ef0f…`。**和 icpe-rux 在 RUX3DPRO 算的相同。**
  - c2、c4 的 n_cue 也是 574,485；sha_t／sha_sess 不同，因為時間被壓縮了。
  - 檔在 `$ROOT/rux-trx/cue/`；`cue.u8` 不進 git。
- **0d. FreeInference 下載**：run `20261107-082812-icpe-trx-rux-fi-download`，rc=0，提前做，只用網路。
  - `hf_verify harvardMadsys/freeinference_agentic_trace sha=16d3c04dae6ccaa0c48a9064e204e3a8f1eb9ade files=81 bad=0`，7.8 GiB。
  - 放在 `$ROOT/datasets/freeinference_agentic_trace/`，README.txt 由腳本寫。
- **1. TraceLab c1 的 APC／BELP**：run `20261107-083210-icpe-trx-rux-tl_c1-apc`，08:32:10 開始，40 workers。
  - 跑完接 `explore_apc.py`。它的第一行是重現檢查，不是全部 0 差就停下來回報，後面都不跑。

## 2026-11-07T08:30+08:00 — **使用者決定：TRX40 跑 icpe-rux 交辦的 TraceLab 與 FreeInference 探索**（APC／BELP、W3 線索、W5）；TraceLab 已下載，轉換中

- **來源**：icpe-rux（rux-proj）兩則訊息，轉述使用者的話：
  - 「icpe-next-trx 目前空的 叫他模擬」「把 TraceLab 給他」「CacheWise 都先不跑」；
  - 「FreeInference 也是」「分工一下」。
- **使用者在這邊直接回答**（AskUserQuestion）：
  - TraceLab：**照做，三個都跑**。任何一步失敗或對不上就停下來回報。
  - FreeInference：**也跑**，排在 TraceLab c1 之後。
  - 結果放**新目錄 `results/icpe_next/rux_trx/`**：只放分析程式的彙總 CSV（有 run_id、ts、host）。原始 sweep CSV 留在 /ssd4，STATUS_TRX 寫路徑。
- **範圍變更**：這改掉了 11/6「TRX40 只做 SemiAnalysis」。STATUS_DATA 的授權清單沒有 FreeInference，這次是使用者明確交辦。
  - FreeInference：HF `harvardMadsys/freeinference_agentic_trace` @ 16d3c04d，CC-BY-4.0。
  - git 不放原始資料或轉換後的紀錄。CacheWise 仍然不碰。
- **性質**：探索，不是正式 block。語意還沒登記（PR:57／95），由 icpe-rux 找 icpe-main 協調。不動任何佇列。
- **程式**：唯讀 worktree `$ROOT/rux-wt` = `origin/icpe-next-rux` b7444163（detached）。不在那裡 commit。
  - 每一步用 `$ROOT/tools/rux_run.sh <name> '<指令>'` 跑（機器工具，不在 git），照 CLAUDE.md §4.1 留 `cmd.sh`／`context.txt`／stdout／stderr／`exit_code`。
  - 每一步都是 `nice -n 10`，掛 memwatch 6144 MiB、swapguard 2933 MiB、diskguard 50 GiB。
- **0a. 下載 TraceLab**：run `20261107-082511-icpe-trx-rux-tl-download`，rc=0。
  - `syfi_coding_trace.jsonl.gz` sha256 `11ce51ec0a25e3d1…`，等於指定值（`sha256sum -c` OK）。
  - 放在 `$ROOT/datasets/tracelab/`，`README.txt` 由腳本寫。
  - 前兩次失敗，都是我呼叫時漏設東西，資料沒有動：
    - `20261107-082442-…`：`setup_3090_data.sh: line 27: TIARA_VENV: unbound variable`。我用 `TIARA_PLATFORM=C_H200` 跳過 env_3090.sh，漏設了 `TIARA_VENV`。
    - `20261107-082450-…`：`setup_3090_data.sh: line 30: curl: command not found`。這台沒有 curl。
  - 改照 11/5 下載 SemiAnalysis 的同一條指令重跑，只把 `$ROOT/tools/env/bin`（工具環境的 curl）加到 PATH 後面：
    `PATH=$PATH:$ROOT/tools/env/bin TIARA_SKIP_VENV=1 TIARA_HOME=$ROOT TIARA_DATASETS=$ROOT/datasets HF_HOME=$ROOT/hf-home TIARA_PY=$ROOT/venv/bin/python TIARA_HF_CLI=$ROOT/tools/env/bin/hf bash $REPO/code/setup_3090_data.sh tracelab`
- **對照表**（版本 B，`origin/icpe-next-main` c868ef2d），放在 `$ROOT/rux-trx/ref/`：
  - `tl_p1_verB.csv`：blob 48da5cd4，973 行。
  - `fi_p1_verB.csv`：blob 8322cf03，1,945 行。
- **0b. 轉換**：run `20261107-082523-icpe-trx-rux-tl-convert` 進行中，輸出到 `$ROOT/datasets/x1_nb_rux/`，和 SA 的 nb 目錄分開。規則 6 的單位驗算：665,453／665,453＝1.0000。
- **接下來**（icpe-rux 的優先順序）：
  1. TraceLab：cue 與 `CUE_CHECK`（c1、c2、c4），然後 APC／BELP 的重現檢查。重現檢查不是 0 差就停。之後跑 c1 的 W3。
  2. FreeInference：下載，然後重現檢查（先一週），再跑 6 週的 W3。
  3. 兩份的 APC／W5。

## 2026-11-07T06:30+08:00 — icpe-data 已收進 icpe-next-data（8e7fc13c）；TRX40 這次的工作結束

- icpe-data 傳訊息說，已從 `origin/icpe-next-trx` a9769257 checkout 這 6 個目錄，並在 acvlab 重新核對過。
- 我在這邊比過：`origin/icpe-next-data` 上這 6 個目錄的 12 個檔，blob 和 a9769257 的完全相同（12／12）。
- icpe-data 說 TRX40 不需要再做什麼。TRX40 沒有佇列在跑，也沒有守門程式留著。

## 2026-11-07T06:20+08:00 — **Mooncake P4 的 6 個 block 全部完成並進 git**（a9769257；1,040,346 個 job、1,243,186 列）；TRX40 目前沒有佇列在跑

- **佇列**：21:03:52 → 06:07:24。6 個 block 都 rc=0，沒有 FAILED／KILLED，三個守門程式都沒觸發。
  - 36 workers：log `$ROOT/runs/queue-p4mc-rust-w36-20261106-210352.log`。
  - 01:08 起 46 workers：log `$ROOT/runs/queue-p4mc-rust-w46-20261107-010805.log`。

  | block | run | workers | 牆鐘 | job 數（＝count） | 列數 | `run_s` 合計 | 每 job `run_s` | 每 job 牆鐘 |
  |:--|:--|--:|--:|--:|--:|--:|--:|--:|
  | `p4_b1/toolagent_P_B1` | `20261106-210352-…` | 36 | 5,092 s | 196,718 | 236,488 | 181,817 | 0.924 | 0.0259 s |
  | `p4_b1/toolagent_M_B1` | `20261106-222844-…` | 36 | 7,325 s | 148,738 | 178,808 | 262,866 | 1.767 | 0.0492 s |
  | `p4_b2/conversation_P_B2` | `20261107-003049-…` | 36 | 2,235 s | 205,160 | 242,360 | 78,914 | 0.385 | 0.0109 s |
  | `p4_b2/conversation_M_B2` | `20261107-010806-…` | 46 | 2,246 s | 153,870 | 181,770 | 101,491 | 0.660 | 0.0146 s |
  | `p4_b2/toolagent_P_B2` | `20261107-014532-…` | 46 | 6,241 s | 191,920 | 230,720 | 284,493 | 1.482 | 0.0325 s |
  | `p4_b2/toolagent_M_B2` | `20261107-032933-…` | 46 | 9,471 s | 143,940 | 173,040 | 434,257 | 3.017 | 0.0658 s |

- **逐 block 核對**，6／6 通過：
  - `exit_code` 0；`kept_jobs` 等於 count；`unmatched_cells=0`；`py_fallback_jobs=0`。
  - `RUN_DIR` 是空的，`req/` 0 個檔；`rust_timing.jsonl` 行數等於 kept。
  - stderr 只有 `session groups…`／`x1fast serve…` 兩種行。
- **程式**：
  - block 的 context 記的 git HEAD 有 96477494（2 個）、b0a43121（1 個）、e9f6c7f0（1 個）、31a28798（1 個），被停掉的半截 run 是 b0a43121。
  - `git diff 313a9aff 31a28798 -- code` 是 0 行，中間只改了 STATUS 檔。
  - 只有第一個 block 啟動時工作目錄不乾淨：當時 `sa_p4_b1/`、`sa_p4_b2/` 兩個結果目錄還沒 commit（未追蹤），不是程式。
- **collect**（06:08:10 → 06:10:42）：只寫出這 6 個目錄、12 個檔（各一個 `x1_sweep.csv.xz`＋`manifest.json`）。
  - 每個檔：
    - `run_id` 單一值，且是該 block 的 run；`host` 都是 `acvlab-TRX40-AORUS-MASTER`；`code_sha` 都是 `3280b1c8799b+rs62b0932b`；`ts` 沒有空值。
    - `noise_model`／`noise_draw` 和 block 名稱一致；`trace` 和 block 名稱一致；`rule` 是 R1、R2；`noise_seed` 0–9。
  - `manifest.json` 的 rows 等於原始 `out/x1/x1_sweep.csv` 的列數（6／6）。
  - `noise_lambda` 的集合和 sa_p4 同類的檔一樣：P_B1 0.0–0.4、M_B1 0.0–0.3、P_B2 0.1–0.4、M_B2 0.1–0.3。
- **大小**：
  - xz 合計 51 MB：`p4_b1` 17 MB、`p4_b2` 34 MB。
  - 6 個 xz 都超過 1 MiB，最大 11.9 MB（`conversation_P_B2`）。
  - 照使用者對 sa_p4 的決定（commit as is）同樣處理。
- **資源**：
  - 36 workers 那段：MemAvailable 最低 52,342 MiB，PSS 最高 1,977 MiB，load1 最高 42.56。
  - 46 workers 那段：MemAvailable 最低 55,105 MiB，PSS 最高 2,474 MiB，load1 最高 47.05。
  - swap 一直 790 MiB；/ssd4 剩 541–542 GiB。
- **B2 對 B1 的速度**（只有 toolagent 兩種都有；B2 是 46 workers、B1 是 36 workers，B2 本身的倍數和 SMT 影響分不開）：
  - 每 job 牆鐘：P 是 1.26 倍，M 是 1.34 倍。
  - 每 job `run_s`：P 是 1.60 倍，M 是 1.71 倍。
  - 02:25 時用重區段的 2.15–2.40 倍推算 06:55～08:00 結束，實際 06:07:24，推算偏晚。
- **沒有收的**：被停掉的半截 run `20261107-010804-icpe-trx-p4_b2-conversation_M_B2`（01:15 節）。它不在 DONE 裡，collect 沒有收；留在 `$ROOT/runs/`。
- **需要別的 session 做的**：icpe-data 從 `icpe-next-trx` 把 `results/icpe_next/b/p4_b1/toolagent_{P,M}_B1/` 和 `p4_b2/{conversation,toolagent}_{P,M}_B2/` checkout 進 icpe-next-data（照 19:20 的約定）。
- **TRX40 現在**：沒有佇列在跑。STATUS_DATA 19:20 的 6 個 Mooncake block 都做完了，沒有別的已分配工作。

## 2026-11-07T02:25+08:00 — Mooncake P4：**4／6 完成**；`toolagent_P_B2` 過半；toolagent 的 B2 比 B1 慢約 2 倍以上（含 SMT 影響）→ **ETA 約 07:10～08:15**〔推算〕

- **佇列**：pgid 2301987（46 workers），沒有 FAILED／KILLED，三個守門程式都沒觸發。

  | block | workers | 起 → 迄 | 牆鐘 | job 數 | `run_s` 合計 | 每 job | 中位數／p90／max |
  |:--|--:|:--|--:|--:|--:|--:|:--|
  | `p4_b2/conversation_M_B2` | 46 | 01:08:06 → 01:45:32 | 2,246 s | 153,870 | 101,491 | 0.660 | 0.279／1.450／2.427 |
  | `p4_b2/toolagent_P_B2` | 46 | 01:45:32 → 跑中 | — | 96,315／191,920（02:20:08） | 94,125（到目前） | 0.977（到目前） | — |
  | `p4_b2/toolagent_M_B2` | 46 | 未開始 | | 143,940 | | | |

- **`conversation_M_B2` 核對**都通過：
  - rc=0；kept 153,870 等於 count；unmatched 0；py_fallback 0。
  - `req/` 0 個檔；`rust_timing` 153,870 行；stderr 只有 `session groups…`／`x1fast serve… threads 46`。
- **`toolagent_P_B2` 開頭**正常：`kept_jobs=191920` 等於 count；`unmatched_cells=0`；`py_fallback_jobs=0`；`req/` 0 個檔。
- **資源**（46 workers，01:08:06 → 02:20）：
  - MemAvailable 最低 55,176 MiB；PSS 最高 2,272 MiB。
  - swap 一直 790 MiB；load1 最高 46.83；/ssd4 剩 541 GiB。
  - 有效並行度（`run_s` 合計 ÷ 牆鐘）：`conversation_M_B2` 是 45.2，36 workers 時約 35.8。但 46 workers 時每個 job 的 `run_s` 會被 SMT 拉高，所以這不等於快了 26%。實際加速 NOT_MEASURED。
- **toolagent B2 對 B1**：
  - 比較方法：`toolagent_P_B2`（46 workers）和 `toolagent_P_B1`（36 workers），依 job 序位比例切 5% 一段，比每段平均 `run_s`。
  - 結果：
    - 前 30% 的輕 job，比例 0.40–1.08；
    - 30–50% 重 job 開始的區段，1.79–2.40；
    - 累積到 50%：1.92 倍（94,125 對 48,909）。
  - 這是「B2 倍數 × SMT 膨脹」合在一起的倍數，兩者分不開。兩個 block 的 job 清單也不完全一樣（191,920 對 196,718），序位只能大致對齊。
- **ETA**〔推算〕：
  - `toolagent_P_B2`：剩下的 `run_s` 用 B1 後半段 × 1.92～2.37 估，總計約 350,000–409,000 執行緒秒。用並行度 45.2，約 **03:55～04:20** 結束。
  - `toolagent_M_B2`：用 `toolagent_M_B1` 的 262,866 × 工作量比（143,940／148,738）× 1.92～2.37，約 488,000–603,000 執行緒秒、3.0–3.7 h。約 **06:55～08:00** 結束。
  - 之後 collect＋核對＋commit 約 10–20 分鐘 → **約 07:10～08:15** 進 git。
- **接下來**：同上，等佇列結束（背景等待 `wait_plan.sh … 2301987`），逐 block 核對、collect、commit／push、通知 icpe-data。

## 2026-11-07T01:15+08:00 — **使用者決定：TRX40 改 46 workers、跑到完不降回**；01:08 在 block 邊界重開；`conversation_P_B2` 完成（3／6）；ETA 改為約 05:00～10:00〔推算〕

- **使用者決定**（00:52 在 icpe-data 說「TRX40 也是 吃更多 核心」，由 icpe-data 轉達。我在這邊用 AskUserQuestion 直接問，使用者 00:55 前後選了）：
  - **46 workers**，留約 2 個執行緒給這台共用機器的其他人。
  - **跑到完，不在 08:00 降回 36**。
- **依據**：
  - TRX40 是 Threadripper 3960X，24 核 48 執行緒（`lscpu`）。
  - 36 workers 時 load 約 36，`top` 閒置 24.2%（00:52）。
  - 之前說「load 90 不會更快」是 Python 引擎時的情況。Rust 是 1 個 worker 約 1 個執行緒。
  - 加上去的 worker 落在 SMT 的第二個執行緒。〔推算〕只快約 5–15%，NOT_MEASURED。
- **重開方式**：腳本 `$ROOT/tools/relaunch_at_boundary.sh`（機器工具，不在 git）。
  - 等舊 log 出現 `end   p4_b2/conversation_P_B2 rc=0`，SIGTERM 舊行程群組 2231622。
  - 用同樣的 `--plans 'p4_b1:toolagent_*' 'p4_b2:conversation_*,toolagent_*' --engine rust`、改 `--workers 46` 重開。
  - memwatch 6144、swapguard 2933、diskguard 50 GiB 重新掛上。
- **實際經過**：
  - 01:08:04 `conversation_P_B2` rc=0 → SIGTERM → 01:08:05 舊佇列結束。
  - 01:08:06 新佇列起來：pgid 2301987，log `$ROOT/runs/queue-p4mc-rust-w46-20261107-010805.log`。
  - 新佇列 `[skip]` 了 3 個已完成的 block，從 `conversation_M_B2` 開始。
  - 被停掉的半截 run `20261107-010804-icpe-trx-p4_b2-conversation_M_B2` 只活了約 1 秒：沒有 `exit_code`，stdout／stderr 都是 0 bytes。它不在 DONE 裡，collect 不會收；我留著沒刪。
  - 新的 `conversation_M_B2`（run `20261107-010806-…`）開頭核對正常：
    - `kept_jobs=153870` 等於 count；`unmatched_cells=0`；`py_fallback_jobs=0`。
    - `RUN_DIR` 是空的；stderr 寫 `threads 46`。
    - context 記的 git HEAD 是 b0a43121，工作目錄是乾淨的。
  - 01:09 `top` 三次取樣：nice 81–96%、閒置 4–16%。
- **`conversation_P_B2` 完成**：00:30:49 → 01:08:04，2,235 s。
  - 核對都通過：rc=0；kept 205,160 等於 count；unmatched 0；py_fallback 0；`req/` 0 個檔；`rust_timing` 205,160 行。
  - `run_s` 合計 78,914 s；每 job 0.385；中位數／p90／max 0.056／1.507／2.482。
  - 00:50 我外推的 110,500–113,900 高估了約 40%。原因是用 toolagent 的曲線套 conversation，不適用。
- **ETA**〔推算〕：
  - 換算基準：36 workers 時的有效並行度 35.8。46 workers 時假設快 5–15%。
  - `conversation_M_B2`：用 `conversation_P_B2` 的 0.385 × toolagent 的 M／P 比 1.91 估，約 113,000 執行緒秒，約 01:55–02:00 結束。
  - toolagent 兩個 B2 block 的 B2 倍數 f 仍是 NOT_MEASURED：
    - f＝1 → 約 05:00；
    - f＝1.5 → 約 06:30；
    - f＝2.5 → 約 09:10–10:00。
  - 之後 collect＋核對＋commit 另外約 10–20 分鐘。
- **之後怎麼量**：
  - `toolagent_P_B2` 現在跑在 46 workers 下，SMT 會把每個 job 的 `run_s` 拉高，所以和 B1（36 workers）比出來的是「f × SMT 膨脹」的合併倍數。
  - 不過推算 `toolagent_M_B2` 的牆鐘時間要用的正是這個合併倍數。`toolagent_P_B2` 跑到一半時會有背景等待叫醒我，到時再更新這裡。

## 2026-11-07T00:50+08:00 — Mooncake P4：**2／6 完成**（rc=0、核對通過），第 3 個 `conversation_P_B2` 跑到約 53%；toolagent 比原本估的慢，**完成時間改為 11/7 約 06:00～11:00**〔推算〕

回覆 icpe-data 的詢問（訊息，見 STATUS_DATA 00:50）。

- **佇列**：pgid 2231622，log `$ROOT/runs/queue-p4mc-rust-w36-20261106-210352.log`，沒有 FAILED／KILLED，三個守門程式都沒觸發。

  | block | 起 → 迄 | 牆鐘 | job 數 | `run_s` 合計（執行緒秒） | 每 job 平均 | 中位數／p90／max |
  |:--|:--|--:|--:|--:|--:|:--|
  | `p4_b1/toolagent_P_B1` | 21:03:52 → 22:28:44 | 5,092 s | 196,718 | 181,817 | 0.924 | 0.083／2.856／7.561 |
  | `p4_b1/toolagent_M_B1` | 22:28:44 → 00:30:49 | 7,325 s | 148,738 | 262,866 | 1.767 | 0.082／5.570／8.238 |
  | `p4_b2/conversation_P_B2` | 00:30:49 → 跑中 | — | 108,557／205,160（00:48:43） | 36,615（107,856 個 job 時） | 0.339（到目前） | — |
  | `p4_b2/conversation_M_B2` | 未開始 | | 153,870 | | | |
  | `p4_b2/toolagent_P_B2` | 未開始 | | 191,920 | | | |
  | `p4_b2/toolagent_M_B2` | 未開始 | | 143,940 | | | |

  - `run_s` 取自各 run 的 `rust_timing.jsonl`（單一執行緒秒數）。兩個完成的 block 有效並行度約 35.7、35.9（`run_s` 合計 ÷ 牆鐘）。
- **完成的 2 個 block 逐項核對**都通過：
  - `exit_code` 0；`kept_jobs` 等於 count（196,718、148,738）。
  - `unmatched_cells=0`；`py_fallback_jobs=0`；`req/` 0 個檔；`rust_timing.jsonl` 行數等於 kept。
  - stderr 只有 `session groups…` 與 `x1fast serve: trace_load_s…` 兩種行。
  - `conversation_P_B2` 開頭的同樣檢查也正常：`kept_jobs=205160`，`unmatched_cells=0`，`py_fallback_jobs=0`。
- **資源**（memwatch 21:03:54 → 00:46）：
  - MemAvailable 最低 52,342 MiB；行程群組 PSS 最高 1,977 MiB。
  - swap 一直 790 MiB；load1 最高 42.56；/ssd4 剩 541 GiB。
- **為什麼比 3 h 久**：
  - 21:15 的 3 h 是用 acvlab 的 0.374 執行緒秒／job 算的。TRX40 的 toolagent B1 實測是 0.924（P）和 1.767（M）。
  - 重的 job 集中在 job 順序的 40%–80%：`toolagent_P_B1` 跑完一半的 job 時，`run_s` 只累積了 26%。所以不能用「跑完幾成 job」線性外推。
- **ETA**〔推算〕：
  - `conversation_P_B2`：用 toolagent B1 的「job 序位 → 累積 `run_s`」曲線外推，總共約 110,500–113,900 執行緒秒。約 **01:25** 結束。
  - `conversation_M_B2`：假設 M／P 每 job 比例和 toolagent B1 一樣（1.91 倍），約 161,000 執行緒秒、約 75 分鐘。
  - toolagent 兩個 B2 block：用 B1 的每 job 速率 × B2 倍數 f。f 在 TRX40 的 Mooncake 上 NOT_MEASURED。
  - 剩下約 668,000 ＋ 431,800 ×（f − 1）執行緒秒，用 35.8 的並行度：
    - f＝1.0 → 約 **06:00**；
    - f＝1.5 → 約 **07:40**；
    - f＝2.5（acvlab Bailian 的倍數）→ 約 **11:00**。
  - 之後 collect＋核對＋commit 另外要約 10–20 分鐘〔推算〕。
  - `toolagent_P_B2` 跑到一部分時（〔推算〕約 03:00 之後），可以用同樣的曲線和 `toolagent_P_B1` 比，量出 f，再更新這裡。
- **接下來**：照 21:15 的計畫，逐 block 核對；全部完成後 `collect --plans p4_b1 p4_b2`，commit／push，通知 icpe-data。

## 2026-11-06T21:15+08:00 — **sa_p4 完成**（48 個 block、1,004,366 個 job、1,434,342 列，已進 git）；Mooncake 對拍 0 差；**6 個 Mooncake P4 block 21:03:52 開跑**

**sa_p4**
- **佇列**：15:16:47 → 20:58:03，48 個 block 都 rc=0，沒有 FAILED／KILLED，三個守門程式都沒觸發。log `$ROOT/runs/queue-sa_p4-rust-w36-20261106-151647.log`。
- **逐 block 核對**，48／48 通過：
  - `kept_jobs`＝`count`（run `20261106-150924-icpe-trx-count-sa_p4`），合計 1,004,366。
  - `unmatched_cells=0`、`py_fallback_jobs=0`、exit 0、`req/` 0 個檔。
  - `rust_timing.jsonl` 的筆數＝`kept_jobs`。
- **`collect --plans sa_p4_b1 sa_p4_b2`**：
  - `sa_p4_b1` 24 個 block、727,272 列；`sa_p4_b2` 24 個 block、707,070 列。
  - 合計 1,434,342 列，和 15:15 那節估 npz 時數的 view 數相同。
  - 每個檔都檢查過：
    - `run_id` 單一值，且是該 block 的 run。
    - `host`＝`acvlab-TRX40-AORUS-MASTER`，`code_sha`＝`3280b1c8799b+rs62b0932b`。
    - `ts` 沒有空值。
    - `noise_model`／`noise_draw` 都和 block 名稱一致。
- **程式**：block 的 context 記的 git HEAD 有 4f5c831（1 個）、cdea0d4（38 個）、afc6e76（1 個）、313a9af（8 個）。`git diff 4f5c831 313a9af -- code` 是空的，中間只改了 STATUS 檔。
- **時間**：sweep 時間每個 block 20–1,059 s；`sa_p4_b1` 合計 10,701 s，`sa_p4_b2` 合計 9,758 s。
  - `rust_timing` 1,004,366 筆：`run_s` 中位數 0.726 s、p90 0.985 s、最長 2.270 s。
- **記憶體**（677 筆）：MemAvailable 最低 47.3 GiB（48,435 MiB）；PSS 合計最高 3.3 GiB；swap 790–947 MiB，沒有超過開跑時的量；load1 最高 43.8。/ssd4 剩 547–548 GiB。
- **檔案大小**：48 個 xz（preset 9）合計 101 MB，36 個超過 1 MiB，最大 4.5 MB（`sa_p4_b1/semianalysis_N72_s3_P_B1`）。
  - CLAUDE.md 寫進 git 的檔要 < 1 MB。git 是資料到 acvlab 和分析 session 的唯一管道。
  - **使用者決定**（本 session，11/6 約 21:10，AskUserQuestion）：照原樣 commit。

**Mooncake 的雜訊對拍**（`$ROOT/tools/mcheck.sh 4f5c831`，19:15:35–19:20:39）
- 參數：`--traces toolagent --rules R1 R2 --cpu 8 --policies LRU O-BEL O-PF O-ALL --slows ssd --modes cake --gpu-gib 111.0625 --gpu-write wb --gpu-policies LRU TTLB:10 --pf-target cpu --noise-models P M --noise-lambdas 0.3 --noise-draws B1 B2 --noise-seeds 0`，`--workers 8`。
- Rust：run `20261106-191535-icpe-trx-mc_noise-rust-4f5c831`，19 s，`py_fallback_jobs=0`。
- Python：run `20261106-191535-icpe-trx-mc_noise-py-4f5c831`，285 s。
- `next_sim_cmp.py`（compare run `20261106-192039-icpe-trx-compare-mc_noise-4f5c831`）：96 列 × 41 欄，鍵含 4 個雜訊欄，不同的格子 0，**CMP PASS**。

**merge 和對拍**
- merge `origin/icpe-next-data` @ 58fdfae（STATUS_DATA 19:20 的最終清單）→ **313a9af**。只改了文件；`code/` 和 4f5c831 相同。
- 三項對拍（`verify_rust2.sh 313a9af`，21:01:05–21:03:00；compare run `20261106-210300-icpe-trx-compare-313a9af`）都是 0 個不同：
  - (a) G0：128,064 格，IDENTICAL。
  - (b1) xc_sa 對 icpe-data：1,280 格。
  - (b2) xc_sa 對 TRX40 的 Python：CMP PASS。
  - (c) `sa_N16_s1` 整個 block：293,760 格。
  - `code_sha`＝`3280b1c8799b+rs62b0932b`。

**Mooncake P4 開跑**
- 指令：`run --plans 'p4_b1:toolagent_*' 'p4_b2:conversation_*,toolagent_*' --engine rust --workers 36`，21:03:52，pgid 2231622。
  - log `$ROOT/runs/queue-p4mc-rust-w36-20261106-210352.log`。
  - 守門程式：memwatch 6,144、swapguard 2,933、diskguard 50 GiB。
  - 開跑前：MemAvailable 53.8 GiB，swap 790 MiB，/ssd4 剩 542 GiB。
- **第一個 block**（`p4_b1/toolagent_P_B1`，run `20261106-210352-icpe-trx-p4_b1-toolagent_P_B1`）：
  - `kept_jobs=196718`（＝`count`），`unmatched_cells=0`，`py_fallback_jobs=0`。
  - `export RUN_DIR=`（空的），`req/` 0 個檔，`code_sha=3280b1c8799b+rs62b0932b`。
- 〔推算〕約 3 h（acvlab 的 0.374 執行緒秒／job）。TRX40 跑 Mooncake 的速度 NOT_MEASURED。

**接下來**：每個 block 照上面核對；全部跑完後 `collect --plans p4_b1 p4_b2`，只會有這 6 個目錄；檢查後 commit／push，並通知 icpe-data。

**需要別的 session 做什麼**：無。
**需要使用者決定什麼**：無。

## 2026-11-06T19:20+08:00 — sa_p4 跑到 38／48；**使用者決定：sa_p4 跑完後，TRX40 接 acvlab P4 裡還沒開始的 Mooncake block**（清單等 icpe-data 定）

**依據**
- icpe-main 的訊息（11/6 約 19:10）轉述使用者在 icpe-main 的話：「那帮我在 RTX 上面继续跑 P4 的任务，可以直接交给他。」
  - 哪些 block 搬由 icpe-data 定；CacheWise 一律留在 acvlab；同一個 block 不能兩台都跑。
- **使用者在本 session 確認**（AskUserQuestion，11/6 約 19:15）：**只接 Mooncake**。
  - 這改了 11/6 早上「TRX40 只做 SemiAnalysis」的決定，也把我寫 results 的範圍擴大到這幾個 Mooncake block（`results/icpe_next/b/p4_b1|p4_b2/<block>/`）。
  - Bailian、TraceLab 不搬：TRX40 上沒有這兩份資料。

**sa_p4 進度**
- 19:05 時 38／48 個 block rc=0，沒有 FAILED／KILLED。swap 947 MiB 沒變，/ssd4 剩 547 GiB。
- 剩 10 個 B2 block（`N64_s2`–`N72_s3`）。〔推算〕用同一批變體的 B1 時間（519–1,060 s），約 21:00 跑完。

**TRX40 上的 Mooncake 資料**
- `$ROOT/datasets/mooncake/traces` 的 sha256：
  - `conversation_trace.jsonl` `b8cbb061…`、`toolagent_trace.jsonl` `48a2db1a…`
  - 和 STATUS_DATA 11/4 23:55 的值相同。
- 不用搬任何資料。

**job 數**（run `20261106-191245-icpe-trx-count-p4-acvlab`，`count --plans p4_b1 p4_b2`）
- 合計 2,297,276，和 STATUS_DATA 相同。
- 依 STATUS_DATA 18:20，acvlab 還沒開始的 Mooncake block 有 6 個，共 **1,040,346** 個 job：

  | block | job |
  |:--|--:|
  | `p4_b1/toolagent_P_B1` | 196,718 |
  | `p4_b1/toolagent_M_B1` | 148,738 |
  | `p4_b2/conversation_P_B2` | 205,160 |
  | `p4_b2/conversation_M_B2` | 153,870 |
  | `p4_b2/toolagent_P_B2` | 191,920 |
  | `p4_b2/toolagent_M_B2` | 143,940 |

- 〔推算〕用 acvlab 實測的 0.374 執行緒秒／job（STATUS_DATA 18:40），36 workers 約 3 h。TRX40 跑 Mooncake 的速度 NOT_MEASURED。
- 〔推算〕acvlab 照 18:40 的速率，約 22:00 會跑到 `toolagent_P_B1`。所以跳過的設定要在那之前做好。
- **dry-run**：
  - 指令：`run --plans p4_b1:toolagent_P_B1,toolagent_M_B1 p4_b2:conversation_P_B2,conversation_M_B2,toolagent_P_B2,toolagent_M_B2 --engine rust --workers 36 --dry-run`
  - 剛好這 6 個 block。都是 `export RUN_DIR=`（空的），都有 `--jobs-from`，`X1_TRACE_DIR` 是上面的目錄。沒有建立 run 目錄。

**計畫**（清單以 icpe-data 在 STATUS_DATA 寫的為準）
1. sa_p4 跑完 → `collect`、核對、commit／push（照 15:20 那節）。
2. 等 icpe-data 在 STATUS_DATA 寫好 TRX40 的 block 清單，以及 acvlab 佇列（pid 3531487）怎麼跳過這些 block。
3. `git merge --no-ff origin/icpe-next-data`。
   - `git diff 4f5c831 HEAD -- code/p3x code/icpe/p3_fvar.py code/p3_platform.py` 要是空的。不是空的就重做三項對拍。
   - `code_sha` 要等於 acvlab P4 的 `3280b1c8799b+rs62b0932b`。
4. **Mooncake 的雜訊對拍**：G0 本身就是 Mooncake conversation，15:05 在這份程式上已經 0 差，但它不含雜訊。所以另外比一個 Mooncake toolagent 的 B1＋B2 子集：同樣參數在 TRX40 用 Python 和 Rust 各跑一次，要 0 個不同的格子。
5. 照清單用 `--plans p4_b1:<…> p4_b2:<…>` 指定 block，`--engine rust --workers 36`。
   - 守門程式照舊：memwatch 6,144、swapguard 2,933、diskguard 50 GiB。
   - 每個 block 核對：`kept_jobs`＝`count`、`unmatched_cells=0`、`py_fallback_jobs=0`、`req/` 是空的。
6. 跑完 `collect --plans p4_b1 p4_b2`。
   - 只收 TRX40 跑的這幾個 block，`host` 要是 `acvlab-TRX40-AORUS-MASTER`。
   - commit／push 前在這裡寫清單，避免和 acvlab 收的檔重疊。
7. 任何一步失敗或有差，就停下來，把完整錯誤寫在這裡，並回報使用者。

**舊計畫的兩步不重跑**（使用者問的，run `20261106-163455-icpe-trx-oldplan-coverage`）
- 舊計畫的 `sa_main_ttl7`（20 個 block）和 `sens_rep sa_*`（4 個 block）：dry-run 後，逐一拿參數去和已完成的 P1–P3 比。
- 24 個都是某個已完成 block 的子集，0 個沒涵蓋：
  - `sa_main_ttl7` → `sa_p1 *_ttl`
  - nvme807、load → `sa_p2`
  - wt、pfgpu → `sa_p3`
- 唯一沒涵蓋的 `sa_t128k` 已經在 11:22 跑完。
- 注意：比的是參數。「參數是子集，所以 job 也是子集」是〔推論〕。

**需要別的 session 做什麼**：**icpe-data**：在 STATUS_DATA 寫 TRX40 的 block 清單，以及 acvlab 佇列跳過這些 block 的做法（附 commit）。已傳訊息。
**需要使用者決定什麼**：無（已決定，見上）。

## 2026-11-06T15:20+08:00 — merge icpe-data 的「P4 不存 npz」（f828e79）；**sa_p4 開跑**（48 個 block、1,004,366 個 job、36 workers）

**依據**：STATUS_DATA 15:25（@ f828e79，git 上讀過）。PREREG 的 P4 只要 seed 間的中位數、最小–最大、平均±標準差，`next_sim_analyze.py` 只讀 sweep CSV，所以不需要逐請求 npz。P4 的 block 帶 `no_npz=True` → `RUN_DIR` 是空字串，x1_sim 不存 npz。

**做完什麼**
- `git merge --no-ff origin/icpe-next-data` → **4f5c831**。這次只改了 `code/icpe/next_b_queue.py`（6 行）。
  - `git diff 424d390 HEAD -- code/p3x code/icpe/p3_fvar.py code/p3_platform.py` 是空的：模擬程式和 15:15 那節對拍過的相同，所以沒有重做對拍。
  - `git diff origin/icpe-next-data HEAD -- code/` 也是空的。
  - 我查過 `RUN_DIR` 在 `x1_rust.py`、`p3_fvar.py`、`p3_platform.py` 都沒有用到，x1_sim 只用在 `_job_rows` 存 npz（`if rd:`）。
- **dry-run**：48 個 block 都是 `export RUN_DIR=`（空的），都有 `--jobs-from` 和 `--workers 36 --engine rust`。
- **開跑**：`run --plans sa_p4_b1 sa_p4_b2 --engine rust --workers 36`，15:16:47，pgid 2121436，log `$ROOT/runs/queue-sa_p4-rust-w36-20261106-151647.log`。
  - 守門程式：memwatch（6,144 MiB）、swapguard（2,933 MiB），另加 **diskguard**（新的機器工具 `$ROOT/tools/diskguard.sh`，/ssd4 剩不到 50 GiB 就停）。
  - 開跑前：swap 947 MiB、MemAvailable 57.8 GiB、/ssd4 剩 548 GiB。
- **第一個 block**（`sa_p4_b1/semianalysis_N16_s1_P_B1`，run `20261106-151647-icpe-trx-sa_p4_b1-semianalysis_N16_s1_P_B1`）：
  - `jobs_from=` 那行：rows 287、cells 287、`kept_jobs=8323`（＝`count`）、`dropped_jobs=39893`、`unmatched_cells=0`。
  - `py_fallback_jobs=0`，`code_sha=3280b1c8799b+rs62b0932b`。
  - `req/` 是空的（0 個檔），run 目錄 1.6 MB。
- 〔推算〕時間：1,004,366 job × 0.578 s（`sa_p3 _wt` 的 `run_s` 中位數）÷ 36 ≈ 4.5 h，約 19:45 跑完。B2 的速度還沒量。

**接下來**：每個 block 跑完核對 `kept_jobs`＝`count`、`unmatched_cells=0`、`py_fallback_jobs=0`。全部跑完後 `collect`，檢查每個檔的 `run_id`／`host`／`code_sha`，再 commit／push。任何 block 失敗就停下來，把完整錯誤寫在這裡。

**需要別的 session 做什麼**：無。
**需要使用者決定什麼**：無。

## 2026-11-06T15:15+08:00 — merge icpe-next-data（v5＋`--jobs-from`＋P4 plan）、對拍全部 0 差、count／dry-run 正確；**sa_p4 沒有開跑：逐請求 npz 估計約 1,164 GiB，/ssd4 只剩 548 GiB** → 使用者選「請 icpe-data 加不寫 npz 的開關」

**做完什麼**
- 依據 STATUS_DATA 15:10（@ 7947e95）：「icpe-trx 可以開始 sa_p4」。
- `git merge --no-ff origin/icpe-next-data` → **424d390**（無衝突）。
  - `git diff origin/icpe-next-data HEAD -- code/` 是空的；`git diff 1bf35bd HEAD -- code/p3x code/icpe/p3_fvar.py code/p3_platform.py` 也是空的。
  - main.rs 和 9167f11 相同，Rust 沒有重編（src `62b0932b…`）。`code_sha`＝`3280b1c8799b+rs62b0932b`，和 icpe-data 的 G0 相同。
- **對拍**（`verify_rust2.sh 424d390`＋`b2check.sh 424d390`，15:05:11–15:08:54），全部不同的格子 0，`py_fallback_jobs=0`：
  - G0：128,064 格。
  - xc_sa：對 icpe-data 的 Python 1,280 格；對 TRX40 的 Python，CMP PASS。
  - `sa_N16_s1` 整個 block：293,760 格。
  - B2 子集 Python 對 Rust：24 列 × 44 欄，CMP PASS。
- **count**：run `20261106-150924-icpe-trx-count-sa_p4`。48 個 block；`sa_p4_b1` 509,256、`sa_p4_b2` 495,110，**合計 1,004,366**，和 STATUS_DATA 相同。
- **dry-run**：`run --plans sa_p4_b1 sa_p4_b2 --engine rust --workers 36 --dry-run`。48 個 block 都有 `--jobs-from <run>/jobs.csv` 和 `--engine rust`，沒有建立 run 目錄。

**為什麼沒開跑：磁碟**
- 每個 job 跑完，`x1_sim._job_rows` 會在 `$RUN_DIR`（佇列設成 `<run>/req`）存一個逐請求 TTFT 的 npz。檔名含雜訊參數，所以每個 job 一個檔，不會互相覆蓋。
- 實測：`sa_p1` 的 `_ttl` block 每個 job 1 個 view，**每個 view 700–922 kB**，依變體而定。`sa_p3` 的 `sa_N72_s3_wt` 的 run 目錄 6.3 GiB 幾乎全是 `req/`。`$ROOT/runs` 目前共 400 GiB。
- 〔推算〕把 P4 每個 job 的 view 數（用同一個 `jobs_for`＋`--jobs-from` 的過濾算）乘上同一變體的每 view 位元組。
  - run `20261106-151238-icpe-trx-p4-npz-est`，程式 `$ROOT/tools/p4_npz_est.py`，只讀程式、不跑模擬。
  - 結果：1,434,342 個 view，**約 1,250 GB（1,164 GiB）**。
  - `/ssd4` 剩 587.8 GB（548 GiB，`df -B1`）→ **放不下**，約跑到 4 成就會滿。/ssd4 是共用磁碟，滿了也會影響別人。
- 使用者決定（本 session，11/6 約 15:12，AskUserQuestion）：**請 icpe-data 加一個讓 P4 不寫 npz 的開關**。npz 是 `x1_boot.py` 成對 bootstrap 用的，所以先請 icpe-data 確認 P4 的分析要不要用；不用的話，sa_p4 只寫 sweep CSV。TRX40 在那之前不跑 P4。

**需要別的 session 做什麼**
- **icpe-data**：
  1. 確認 PREREG 的 P4 分析需不需要逐請求 npz。必要時問 icpe-main／icpe-ana。
  2. 不需要的話，在 `next_b_queue.py` 加開關，讓 P4 的 block 不設 `RUN_DIR`、或不寫 npz；需要改 `x1_sim` 的話請 icpe-sim 改。
  3. 好了在 STATUS_DATA 寫 commit。
- **acvlab 的 P4 也要看一下**：`p4_b1`＋`p4_b2` 有 2,297,276 個 job。TraceLab 每個 job 約 4.4 MB（STATUS_DATA 11/4 的數字），應該也要先算磁碟。

**需要使用者決定什麼**：無（已決定，見上）。

## 2026-11-06T14:30+08:00 — 先 merge v5（B2 進 Rust，icpe-next-sim 4356484，程式 4a4baa4）；對拍全部 0 差，B2 在 TRX40 走 Rust

**依據**：STATUS_SIM 14:25（@ 4356484，git 上讀過）「需要 icpe-trx 做什麼：merge 後重做 G0；venv 不是 3.12 的話 B2 會退回 Python；用 `--engine rust` 跑 xc_sa 和 Python 版逐格比」；icpe-sim 也直接傳了訊息。sa_p4 還要等 `--jobs-from`。TRX40 現在是閒的，所以先把 v5 併進來驗證，提早發現 TRX40 特有的問題（例如編譯失敗、B2 退回 Python）。

**做完什麼**
- TRX40 的 venv 是 **Python 3.12.15**（`$ROOT/venv/bin/python -I -c 'import sys;print(sys.version)'`），所以 B2 會交給 Rust。
- `git merge --no-ff origin/icpe-next-sim` → **9167f11**（無衝突）。`git diff 4356484 HEAD -- code/p3x code/icpe/p3_fvar.py code/p3_platform.py` 是空的。
- Rust 自動重新編譯：src_sha256 `62b0932b…`，bin `x1fast-work/build-62b0932bb6cd6d9d/fast/x1fast3`（bin_sha256 `576b78a0…`）。新的 `code_sha`＝`fd0dccbf7093+rs62b0932b`。
- **三項對拍**（`verify_rust2.sh 9167f11`，14:21:50–14:24:05；compare run `20261106-142405-icpe-trx-compare-9167f11`）：
  - (a) G0：128,064 格，不同 0。
  - (b1) xc_sa 對 icpe-data 的 Python：1,280 格，不同 0。
  - (b2) xc_sa 對 TRX40 的 Python：32 列 × 40 欄，不同 0，CMP PASS。
  - (c) `sa_N16_s1` 整個 block：293,760 格，不同 0。
  - 都是 `py_fallback_jobs=0`。
- **B2 對拍**（新的機器工具 `$ROOT/tools/b2check.sh`）：
  - 參數照 SIM_CHANGES 20 的 SemiAnalysis B2 子集：`--traces nb:semianalysis_N16_s1 --rules nb --cpu 8 --policies LRU O-BEL O-PF O-ALL --slows ssd --modes cake --gpu-gib 0 111.0625 --gpu-write wt --gpu-policies TTLB:10 --pf-target cpu --noise-models P M --noise-lambdas 0.3 --noise-draws B2 --noise-seeds 0`，`--workers 8`。
  - Rust：run `20261106-142406-icpe-trx-sa_b2-rust-9167f11`，3 s，`py_fallback_jobs=0`，stdout 沒有 `noise_B2_py`。
  - Python：run `20261106-142406-icpe-trx-sa_b2-py-9167f11`，91 s。
  - `next_sim_cmp.py`（compare run `20261106-142540-icpe-trx-compare-b2-9167f11`）：24 列 × 44 欄，鍵含 4 個 noise 欄，不同的格子 0，**CMP PASS**。
- 記憶體：MemAvailable 最低 53.8 GiB，swap 959 MiB 沒變。

**還沒做**：sa_p4 仍然等 `--jobs-from`。icpe-sim 預計 15:30–16:15 push；之後 icpe-data 要 merge 進 icpe-next-data、做 G0。到時我 merge icpe-next-data、重做上面的對拍，再 dry-run／count，然後開跑（見 14:20 那節）。

**需要別的 session 做什麼**：無。
**需要使用者決定什麼**：無。

## 2026-11-06T14:20+08:00 — 接下 P4 的 SemiAnalysis（`sa_p4_b1`、`sa_p4_b2`），36 workers；等 icpe-data 通知

**依據**
- STATUS_DATA 14:05（@ 8f1ef80，git 上讀過）：使用者決定 P4 照 PREREG 全跑；`sa_p4_b1`（509,256 job）和 `sa_p4_b2`（495,110 job）交給 TRX40，共 48 個 block、1,004,366 個 job。icpe-data 也直接傳了訊息。
- **使用者決定**（本 session，11/6 約 14:15，AskUserQuestion）：TRX40 接 sa_p4；用 **36 workers**，和 P1–P3 相同（icpe-data 建議的是 32）。

**開跑的條件**（都還沒到）
1. icpe-sim push v5（B2 進 Rust）和 `x1 sweep --jobs-from`。
2. icpe-data merge 進 `icpe-next-data`、做完 G0，然後在 STATUS_DATA 寫好。

**我的步驟**
1. `git merge --no-ff origin/icpe-next-data`。
2. Rust 在 TRX40 重新編譯，重做三項對拍（G0、xc_sa 對 Python、`sa_N16_s1`）。如果 STATUS_SIM 對 v5／`--jobs-from` 指定了 TRX40 要做的驗證，照它加做。
3. `run --plans sa_p4_b1 sa_p4_b2 --engine rust --workers 36`：先 `--dry-run` 和 `count`，確認 48 個 block、1,004,366 個 job；memwatch（6,144 MiB）和 swapguard（2,933 MiB）都開。
4. 任何一步失敗或對拍有差，就停下來，把完整錯誤寫在這裡，回報使用者。

**需要別的 session 做什麼**：icpe-data 在 STATUS_DATA 寫「icpe-trx 可以開始 sa_p4」時，請附上 icpe-next-data 的 commit。
**需要使用者決定什麼**：無。

## 2026-11-06T13:45+08:00 — **`sa_p3` 的 12 個 `_wt` 完成 → TRX40 的 SemiAnalysis 偏離 6（P1–P3）＋`sa_t128k` 全部完成**；目前 TRX40 沒有佇列在跑

**做完什麼**
- 佇列 `run --plans sa_p3 --only 'sa_N*_wt' --engine rust --workers 36`：12 個 block 都 rc=0，13:12:59 → 13:37:22（24 分鐘）。log `$ROOT/runs/queue-sa_p3_wt-rust-w36-20261106-131259.log`；沒有 `*_KILLED`，swapguard log 是空的。
- `collect --plans sa_p3` → `results/icpe_next/b/sa_p3/<變體>_wt/`：12 個 block，各 7,344 列（job 6,174），合計 74,088 job、88,128 列。之前收的 12 個 `_pfgpu` 沒有變動。
  - 12 個檔都 `run_id` 單一值，等於各自的 run（`20261106-131259…sa_N16_s1_wt` 到 `20261106-133506…sa_N72_s3_wt`）。
  - `host` 都是 `acvlab-TRX40-AORUS-MASTER`，`code_sha` 都是 `fd0dccbf7093+rse9497446`，`gpu_write` 都是 `wt`，`ts` 沒有空值。
  - 12 個 run 都 `py_fallback_jobs=0`，`rust_timing.jsonl` 各 6,174 筆。
- 程式：第一個 block 的 context 記 git HEAD 2b73df1，其餘 11 個記 a1608f2。中間的 commit 只改了 STATUS_TRX（`git diff --stat 2b73df1 a1608f2` 只有這一個檔），所以 12 個 block 的程式相同，也和三項對拍相同。
- sweep 每個 block 88.8–121.8 s，合計 1,296 s。`rust_timing` 74,088 筆：`run_s` 中位數 0.578 s、p90 0.902 s、最長 1.851 s。
- 記憶體（49 筆，每 30 s）：PSS 最高 2.8 GiB，MemAvailable 最低 53.5 GiB；swap 959 MiB，整段沒變；load1 最高 39.4。
- 早上失敗的 run `20261106-093803-icpe-trx-sa_p3-sa_N16_s1_wt`（舊程式）不收；同一個 block 這次用新程式跑完。

**TRX40 的 SemiAnalysis 總表**（全部在 git 裡）

| 目錄 | block | 列 | code_sha |
|:--|--:|--:|:--|
| `sa_main` | 12 | 88,128 | （Python，見 11/5 各節） |
| `sa_p1`（`_ttl`、`_trunc`） | 24 | 139,968 | `a27559138492+rs43dbe828` |
| `sa_p2`（`_nvme807`、`_load`） | 24 | 146,880 | `a27559138492+rs43dbe828` |
| `sa_p3` `_pfgpu` | 12 | 15,552 | `a27559138492+rs43dbe828` |
| `sa_p3` `_wt` | 12 | 88,128 | `fd0dccbf7093+rse9497446` |
| `sens_rep/sa_t128k` | 1 | 1,632 | `a27559138492+rs43dbe828` |

- `_wt` 用的是修正後的程式，其他用修正前的程式。照 STATUS_SIM 10:10／13:15，wb＋GPU 的格子在兩版之間不變；本機的三項對拍（13:15 那節）也顯示 wb 的 xc_sa 和 `sa_N16_s1` 在新程式下 0 差。
- 不解讀上限。P4（雜訊）不在 TRX40 的工作範圍。

**需要別的 session 做什麼**：無。
**需要使用者決定什麼**：無。

## 2026-11-06T13:15+08:00 — merge wt 修正（icpe-next-sim 776486c，程式＝7193a2d）；三項對拍全部 0 差；12 個 `_wt` 開跑

**依據**：STATUS_SIM 13:15（@ 776486c，git 上讀過）「需要 icpe-trx 做什麼：merge 後重做 G0；main.rs 變了會自動重編；用 `--engine rust` 跑 xc_sa 和 Python 版逐格比對，之後再跑 12 個 `_wt` block」；icpe-sim 也直接傳訊息請我 merge 776486c。icpe-data 還沒把它併進 `icpe-next-data`（13:09 查過），所以直接 merge icpe-next-sim。

**做完什麼**
- `git merge --no-ff origin/icpe-next-sim` → **2b73df1**（無衝突）。`git diff 776486c HEAD -- code/p3x code/icpe/p3_fvar.py code/p3_platform.py` 是空的。
  - 一起進來的還有準備快取 c7ef740（`X1_PREP_CACHE` 預設 1，快取寫在 `X1_RUST_WORK`＝`$ROOT/x1fast-work`）。`code/icpe/next_b_queue.py` 沒變（icpe-next-data 的 23865ee 沒併）。
- Rust 自動重新編譯：src_sha256 `e9497446…`，bin `x1fast-work/build-e94974467f6f608c/fast/x1fast3`（bin_sha256 `f4c5e7d7…`）；新的 `code_sha`＝`fd0dccbf7093+rse9497446`。
- **三項對拍**（`verify_rust2.sh 2b73df1`，`--engine rust --workers 32`，13:10:30–13:12:47，memwatch／swapguard 有開；compare run `20261106-131244-icpe-trx-compare-2b73df1`）：

| 項目 | run | 對照 | 結果 |
|:--|:--|:--|:--|
| (a) G0 | `20261106-131030-icpe-trx-g0-rust-2b73df1`（26.5 s） | H200 `results/icpe_h200/t1/p01/sim_V1/x1/x1_sweep.csv.gz` | 4,416 列，128,064 格，不同 0 |
| (b1) xc_sa | `20261106-131030-icpe-trx-xc_sa-rust-2b73df1`（17.6 s） | `results/icpe_next/b/xcheck/xc_sa`（icpe-data 的 Python） | 32 列，1,280 格，不同 0 |
| (b2) xc_sa | 同上 | TRX40 的 Python xc_sa（run `20261105-105555-icpe-trx-xcheck-xc_sa-54fa2d8`），`next_sim_cmp.py` | 32 列 × 40 欄，不同 0，CMP PASS |
| (c) sa_N16_s1 整個 block | `20261106-131030-icpe-trx-sa_N16_s1-rust-2b73df1`（89.9 s） | `results/icpe_next/b/sa_main/sa_N16_s1` | 7,344 列，293,760 格，不同 0 |

  - 三個 run 都 `py_fallback_jobs=0`。(b)(c) 都是 wb＋GPU，照 STATUS_SIM 不受這次修正影響，結果也的確沒變。
- **12 個 `_wt` 開跑**：`run --plans sa_p3 --only 'sa_N*_wt' --engine rust --workers 36`。先 `--dry-run` 確認剛好選到 12 個 `_wt` block，沒有建立任何 run 目錄。13:12:59 開始，pgid 2091475，log `$ROOT/runs/queue-sa_p3_wt-rust-w36-20261106-131259.log`。memwatch（6,144 MiB）和 swapguard（2,933 MiB）都有開。
  - 開跑前 swap 959 MiB、MemAvailable 57.7 GiB。
  - 舊的失敗 run `20261106-093803-icpe-trx-sa_p3-sa_N16_s1_wt` 不收。

**需要別的 session 做什麼**：無。
**需要使用者決定什麼**：無。

## 2026-11-06T11:22+08:00 — `sa_p3` 的 12 個 `_pfgpu`＋`sens_rep sa_t128k` 完成；`_wt` 等 icpe-sim 的修正（7193a2d）

**使用者決定（本 session，11/6 約 11:05，AskUserQuestion）**：等 icpe-sim 修，期間先用現在這版 Rust 跑 `_pfgpu` 和 `sa_t128k`，出錯就停；修好後 merge → 重做三項對拍 → 跑 12 個 `_wt`。已直接通知 icpe-sim（SendMessage）。

**icpe-sim 的回覆**（STATUS_SIM 10:10 @ 88a81e3，已在 git 上核對）：
- Python 在同一個 job 也出錯（O-PF、CPU 8、wt、TTLB:10；`demote()` 的 `self.sb[s][-1]` → IndexError）→ 模型的邊界情況，不是 Rust 獨有。
- 原因：新請求沒有完整 block 時，回合結束只做 `cpu_set.discard(s)`，順序結構（LRU／ARC／S3FIFO）留下過期項目。SemiAnalysis 每個變體 198 個這種請求；`semianalysis_N16_s1_t128k` 125 個。
- **wb＋GPU 不會發生**（GPU 回合開始時已 `_order_remove`）→ `sa_main`、`sa_p1`、`sa_p2`、`_pfgpu`、`sa_t128k`、三項對拍不受影響。我查過本機每個 SA run 的 cmd.sh：除了失敗的 `sa_p3/sa_N16_s1_wt`，全部是 `--gpu-gib 111.0625 --gpu-write wb`。
- 修法：使用者在 icpe-sim 選 A（兩個回合結束都 `order_remove`，兩個引擎都改），commit 7193a2d；驗證 run `20261106-111012-icpe-sim-g1-7193a2d`（最後跑整個 `sa_N16_s1_wt` Rust block），預計 12:30，過了寫 STATUS_SIM。`main.rs` 有改 → TRX40 merge 後會重新編譯。

**做完什麼**
- 佇列：`run --plans sa_p3 sens_rep --only 'sa_N*_pfgpu' 'sa_t128k' --engine rust --workers 36`（先 `--dry-run` 確認只選到這 13 個 block；`'*_pfgpu'` 會選到 sens_rep 其他資料集的格子，所以不用）。11:06:08 → 11:19:19，log `$ROOT/runs/queue-sa_pfgpu_t128k-rust-w36-20261106-110608.log`；程式同三項對拍（git HEAD 98b0d2f，`code/` 和 eeaecef 相同）。
- `results/icpe_next/b/sa_p3/<變體>_pfgpu/`：12 個 block，各 1,296 列（job 1,296），sweep 45.9–53.8 s，合計 610 s。
- `results/icpe_next/b/sens_rep/sa_t128k/`：run `20261106-111913-icpe-trx-sens_rep-sa_t128k`，1,372 job、1,632 列，sweep 4.1 s（`--traces nb:semianalysis_N16_s1_t128k --gpu-gib 111.0625 --gpu-write wb --pf-target cpu`）。
- 13 個檔都 `run_id` 單一值、`host`＝`acvlab-TRX40-AORUS-MASTER`、`code_sha`＝`a27559138492+rs43dbe828`，13 個 run 都 `py_fallback_jobs=0`。
- 記憶體（27 筆）：PSS 最高 3.2 GiB，MemAvailable 最低 53.2 GiB；swap 959 MiB；load1 最高 32.5。

**還沒跑**：`sa_p3` 的 12 個 `_wt`（74,088 job）。失敗的 run `20261106-093803-icpe-trx-sa_p3-sa_N16_s1_wt` 不收。

## 2026-11-06T09:45+08:00 — `sa_p2` 完成；**`sa_p3` 第一個 block（`sa_N16_s1_wt`）失敗：Rust 引擎 panic（`src/main.rs:667`）→ 佇列停了**

**`sa_p2` 完成**
- 24 個 block 都 rc=0，08:53:56 → 09:38:03（44 分鐘）→ `results/icpe_next/b/sa_p2/`（commit dc8892c，12 MB）：12 個 `_nvme807` 各 2,448 列、12 個 `_load` 各 9,792 列；合計 124,416 job、146,880 列。每個檔 `run_id` 單一值、`host`＝`acvlab-TRX40-AORUS-MASTER`、`code_sha`＝`a27559138492+rs43dbe828`；24 個 run 都 `py_fallback_jobs=0`。
- sweep 每個 block 30.3–171.0 s，合計 2,251 s。`rust_timing.jsonl` 124,416 筆：`run_s` 中位數 0.619 s、p90 0.916 s、最長 1.919 s。
- 記憶體（88 筆）：PSS 最高 3.0 GiB，MemAvailable 最低 53.3 GiB；swap 959 MiB；load1 最高 40.8。

**`sa_p3` 失敗**（完整錯誤；沒有重試）
- run `20261106-093803-icpe-trx-sa_p3-sa_N16_s1_wt`，exit 1。git HEAD b68d034（程式＝eeaecef，sim 1304345，`bin_sha256=ceaf3065…`，和三項對拍相同）。
- 指令（cmd.sh 最後一行的尾段）：`… x1 sweep --traces nb:semianalysis_N16_s1 --rules nb --cpu 8 16 32 64 128 256 512 1024 111.0625 --policies LRU LRU2 ARC S3FIFO GDSF PENS FREQ2Q LRU+NONE LRU+2HW LRU+2HA LRU+RPF O-BEL O-BELS O-DROP O-NOW O-PF O-ALL --slows ssd b32 nfs --modes cake --gpu-gib 111.0625 --gpu-write wt --gpu-policies LRU BEL TTLA:0.3 TTLA:10 TTLA:300 TTLB:0.3 TTLB:10 TTLB:300 --pf-target cpu --workers 36 --engine rust`
- 跑到 400／6,174 個 job（stdout 最後一行 `[400/6174 8s] … O-ALL …`；`rust_timing.jsonl` 400 筆）。
- stderr.log 全文：
  ```
  session groups: 550 groups for 2090 sessions, largest 153
  x1fast serve: trace_load_s 0.649105179 threads 36

  thread '<unnamed>' (2052850) panicked at src/main.rs:667:41:
  index out of bounds: the len is 0 but the index is 18446744073709551615
  note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace

  thread '<unnamed>' (2052865) panicked at src/main.rs:667:41:
  index out of bounds: the len is 0 but the index is 18446744073709551615
  Traceback (most recent call last):
    File ".../code/icpe/p3_fvar.py", line 163, in <module>
      main()
    File ".../code/icpe/p3_fvar.py", line 137, in main
      XS.main()
    File ".../code/p3x/x1_sim.py", line 2740, in main
      {"selftest": cmd_selftest, "sweep": cmd_sweep, "analyze": cmd_analyze}[a.cmd](a)
    File ".../code/p3x/x1_sim.py", line 2347, in cmd_sweep
      _rust_sweep(JR, a.workers, put)
    File ".../code/p3x/x1_sim.py", line 2401, in _rust_sweep
      raise RuntimeError(f"x1fast（常駐，{ev[1].tdir}）提早結束，exit {ev[1].p.wait()}（錯誤訊息在 stderr）")
  RuntimeError: x1fast（常駐，/ssd4/hungwei/icpe-next/x1fast-work/trace-7cfa6556228f0469c7dc5113）提早結束，exit -6（錯誤訊息在 stderr）
  ```
- 佇列 log：`BLOCK FAILED sa_p3/sa_N16_s1_wt rc=1` → 佇列 09:38 結束（`sa_p3` 其餘 23 個 block 沒跑）。保護沒觸發，沒有殘留行程。
- `src/main.rs:667` 是 `demote()` 裡開預取時的 `let t = self.tr.next_access(bl[bl.len() - 1], self.cur);`：`bl`（`self.sbl(s)`）是空的，`len()-1` 溢位。只看到這一行，沒有找出是哪個 job（哪個策略／TTL／CPU 大小）；Python 引擎在同一個 job 會不會出錯 NOT_MEASURED。
- **需要 icpe-sim 做什麼**：看 `--gpu-write wt` 時 `demote()` 為什麼會拿到沒有 block 的 session（重現：上面的指令、`nb:semianalysis_N16_s1`）。程式歸 icpe-sim，我不改。
- **需要使用者決定什麼**：`sa_p3` 怎麼繼續（已問）。

## 2026-11-06T08:58+08:00 — **`sa_p1` 完成**（24 個 block，Rust，36 workers）；`sa_p2` 在跑

- 佇列 `queue-sa_p123-rust-w36-20261106-080648.log`：`sa_p1` 24 個 block 都 rc=0，08:06:48 → 08:53:56（47 分鐘）。run_id＝`20261106-HHMMSS-icpe-trx-sa_p1-<block>`（第一個 `…-080648-…-sa_N16_s1_ttl`，最後一個 `…-085152-…-sa_N72_s3_trunc`）。
- collect → `results/icpe_next/b/sa_p1/<block>/`（commit c800ea8，9.2 MB）：
  - 12 個 `_ttl`：各 6,480 列（job 6,480）；12 個 `_trunc`：各 5,184 列（job 4,824）。合計 135,648 job、139,968 列。
  - 每個檔：`run_id` 單一值、`host`＝`acvlab-TRX40-AORUS-MASTER`、`code_sha`＝`a27559138492+rs43dbe828`；24 個 run 都 `py_fallback_jobs=0`。
- 時間：sweep 每個 block 69.0–132.4 s，合計 2,467 s。`rust_timing.jsonl`（135,648 筆＝job 數）：Rust 引擎每 job `run_s` 中位數 0.629 s、p90 0.940 s、最長 1.924 s；從派出到讀回中位數 0.637 s、最長 2.288 s。
- 記憶體（mem.log 08:06:48–08:53:56，93 筆）：行程群組 PSS 最高 2.9 GiB，MemAvailable 最低 53.4 GiB；swap 958–959 MiB；load1 最高 40.2。保護都沒觸發。
- 下一個：`sa_p2/sa_N16_s1_nvme807` 08:53:56 開跑。〔算術〕照 `sa_p1` 的速度（約 48 job/s，含每個 block 的準備時間），`sa_p2`＋`sa_p3` 共 214,056 job 約 75 分鐘 → 約 10:10 跑完；之後 merge 23865ee、跑 `sens_rep --only sa_t128k`。
- 不解讀上限（分析由新的分析 session 做）。

## 2026-11-06T08:12+08:00 — **三項對拍都 0 個不同**；Rust 佇列 `sa_p1 sa_p2 sa_p3` 開跑（36 workers，使用者決定）

**對拍**（STATUS_DATA 08:10 第 3 步；`$ROOT/tools/verify_rust2.sh eeaecef`；`--engine rust --workers 32`；比對 run `20261106-080609-icpe-trx-compare-eeaecef`）
- 執行檔：`src_sha256=43dbe828…`、`bin_sha256=ceaf3065f96e46e1fefb834ca3586806771cace604a75456ca46a5c3922c197b`（TRX40 用 rustc 1.99.1 自己編）；code_sha `a27559138492+rs43dbe828`（和 acvlab 的 G0 rust run `20261106-074940-icpe-data-g0-rust-merged-1304345` 相同）；三個 run 的 `py_fallback_jobs=0`。
- (a) G0：run `20261106-080343-icpe-trx-g0-rust-eeaecef`，sweep 18.4 s（`time -v` 牆鐘 25.8 s）→ H200 參考：4,416 列 × 29 欄＝**128,064 格，0 個不同**，`G0_RESULT IDENTICAL`。
- (b) xc_sa：run `20261106-080343-icpe-trx-xc_sa-rust-eeaecef`，sweep 1.9 s → `results/icpe_next/b/xcheck/xc_sa`：32 列 × 40 欄＝**1,280 格，0 個不同**；→ TRX40 的 Python xc_sa（run `20261105-105555-icpe-trx-xcheck-xc_sa-54fa2d8`）：`next_sim_cmp.py` **CMP PASS**（欄位、列數相同，不同的格子 0）。
- (c) 整個 `sa_N16_s1` block：run `20261106-080343-icpe-trx-sa_N16_s1-rust-eeaecef`，6,174 job，sweep 90.1 s（`time -v` 牆鐘 1:43.5，最大 RSS 2.9 GB）→ `results/icpe_next/b/sa_main/sa_N16_s1/x1_sweep.csv.xz`：7,344 列 × 40 欄＝**293,760 格，0 個不同**，`G0_RESULT IDENTICAL`。〔算術〕同一個 block 的 Python（40 workers）sweep 3,903 s。
- 記憶體（mem.log，10 筆）：行程群組 PSS 最高 2.7 GiB，MemAvailable 最低 53.4 GiB，swap 958 MiB。

**使用者決定（本 session，11/6 08:0x，AskUserQuestion）**：使用者說「用更多核心，最後 load 90 上下」；我說明 TRX40 只有 48 個硬體執行緒、load 90 不會更快後，使用者選 **36 workers（約 45 CPU）**。所以不是 STATUS_DATA 08:10 的 32。

**開跑**：08:06:48，`env … NEXT_B_TAG=icpe-trx RUSTC=… X1_CARGO=… CARGO_HOME=… RUSTUP_HOME=… X1_RUST_WORK=$ROOT/x1fast-work python -I code/icpe/next_b_queue.py run --plans sa_p1 sa_p2 sa_p3 --engine rust --workers 36`（git HEAD 9a78df5；log `$ROOT/runs/queue-sa_p123-rust-w36-20261106-080648.log`；memwatch 6 GiB、swapguard 2,933 MiB）。第一個 block `sa_p1/sa_N16_s1_ttl`（run `20261106-080648-icpe-trx-sa_p1-sa_N16_s1_ttl`）：stdout 印 `jobs=6480 workers=36 code_sha=a27559138492+rs43dbe828`。
- 每個 plan 跑完：`collect` → commit → push，記 run_id、時間、記憶體、`rust_timing.jsonl`。
- **`sa_t128k`**：icpe-data 在 STATUS_DATA 08:20（23865ee）更正，要跑。佇列已照舊指令開跑，不中斷；`sa_p3` 跑完後 merge 23865ee（只改 `next_b_queue.py`），再跑 `run --plans sens_rep --only sa_t128k --engine rust --workers 36`。`sens_rep` 其餘 4 個 `sa_*` 不跑。

## 2026-11-06T08:06+08:00 — 使用者決定：Rust 跑 SemiAnalysis 的 P1–P3；裝好 rustc 1.99.1、merge 到 9be4925；三項對拍在跑

**使用者決定**
- 在本 session（AskUserQuestion，11/6 07:5x）：「改用 Rust 跑偏離 6」。
- 在 icpe-data（STATUS_DATA 08:10 @ 9be4925，以這節為準）：TRX40 裝 Rust 自己跑 `sa_p1` → `sa_p2` → `sa_p3`；`sa_main_ttl7`、`sens_rep sa_*` **不跑**（理由寫在該節：P2／P3 已涵蓋全部格子）。
- 注意（事實，不是建議）：`sens_rep` 的 5 個 `sa_*` 格裡，`sa_t128k`（截斷 128K 的紀錄 `semianalysis_N16_s1_t128k`）在 `sa_p1`–`sa_p3` 裡沒有對應的 block；另外 4 個（nvme807、load、wt、pfgpu）有。

**做完什麼**
- **Rust 工具鏈**：`micromamba create -p $ROOT/tools/rust -c conda-forge rust=1.99.1`（使用者權限；log `$ROOT/tools/rust-install-*.log`）→ `rustc 1.99.1 (58cf8f9c3 2026-10-13)`、`cargo 1.99.1 (5f94df478 2026-08-27)`。版本和 STATUS_DATA 08:10 要的相同；裝法是 conda-forge 不是 rustup（使用者在本 session 選的選項寫明 conda-forge）。cargo 要 `RUSTC=$ROOT/tools/rust/bin/rustc`（不然找不到 rustc）。環境：`X1_CARGO=$ROOT/tools/rust/bin/cargo CARGO_HOME=$ROOT/tools/cargo-home RUSTUP_HOME=$ROOT/tools/rustup-unused X1_RUST_WORK=$ROOT/x1fast-work`。試編（scratchpad 目標目錄）4.9 s 成功。
- **merge**：
  - 741bc2c：`origin/icpe-next-data` @ bc1e040（Rust 0b0ef1e、`run --engine rust`、偏離 6 的 `sa_p1`–`sa_p3`）。
  - **eeaecef**：`origin/icpe-next-data` @ 9be4925（含 icpe-sim 47a4617）。`git diff 1304345 HEAD -- code/p3x code/icpe/p3_fvar.py code/p3_platform.py` **是空的**。
- **job 數**：run `20261106-075249-icpe-trx-count-sa-p1p3`（`count --plans sa_p1 sa_p2 sa_p3 sens_rep`，在 741bc2c；`code/icpe/next_b_queue.py` 在 eeaecef 沒變）：`sa_p1` 135,648、`sa_p2` 124,416、`sa_p3` 89,640（合計 349,704，和 STATUS_DATA 的數相同）；`sens_rep` 的 `sa_*` 5 格合計 5,336（不跑）。
- **作廢**：07:52 在 741bc2c 上開的 G0／xc_sa（py＋rust）四個 run `20261106-075222-icpe-trx-{g0,xc_sa}-{py,rust}-741bc2c`：08:02 手動 SIGTERM（py 兩個還在跑；rust 兩個 exit 0 但沒比）。理由：STATUS_DATA 08:10 要用 47a4617／1304345 重做。目錄裡留了 `SUPERSEDED`，不當驗證證據。

**現在在跑**：STATUS_DATA 08:10 第 3 步的三項對拍（`$ROOT/tools/verify_rust2.sh eeaecef`，08:03:43 開始，log `$ROOT/runs/verify-rust2-eeaecef-20261106-080343.log`；memwatch 6 GiB、swapguard 2,933 MiB）。都是 `--engine rust --workers 32`，依序：
- (a) G0：run `20261106-080343-icpe-trx-g0-rust-eeaecef` → H200 參考（`next_b_g0.py`）。
- (b) xc_sa：run `20261106-080343-icpe-trx-xc_sa-rust-eeaecef` → `results/icpe_next/b/xcheck/xc_sa`（`next_b_g0.py`）＋TRX40 的 Python xc_sa（run `20261105-105555-icpe-trx-xcheck-xc_sa-54fa2d8`；`next_sim_cmp.py`）。
- (c) 整個 `sa_N16_s1` block：run `20261106-080343-icpe-trx-sa_N16_s1-rust-eeaecef`（manifest 的 cmd＋`--engine rust --workers 32`）→ `results/icpe_next/b/sa_main/sa_N16_s1/x1_sweep.csv.xz`（`next_b_g0.py`）。
- 三項都 0 個不同 → `run --plans sa_p1 sa_p2 sa_p3 --engine rust --workers 32`。有任何不同 → 停、寫這裡。

## 2026-11-06T07:55+08:00 — **`sa_main` 12／12 完成**；3 個 TTL 的分析＋`ttl7cells` 完成；下一步等使用者決定

**做完什麼**
- 最後一個 block：run `20261106-030157-icpe-trx-sa_main-sa_N72_s3`，exit 0，sweep 5,805 s（queue 03:01:57 → 04:38:54），7,344 列 → `results/icpe_next/b/sa_main/sa_N72_s3/`（commit de566f7）。記憶體（183 筆）：PSS 合計最高 10.4 GiB，MemAvailable 最低 45.6 GiB；swap 980 MiB。佇列 04:38:54 結束，兩個保護跟著結束、都沒觸發。
- `sa_main` 12 個 block 都在 `results/icpe_next/b/sa_main/`：每個 7,344 列、`run_id` 單一值、`host`＝`acvlab-TRX40-AORUS-MASTER`；程式都是 8ef0ada（＝sim 54fa2d8，maxtasksperchild 4），40 worker。sweep 時間 3,789–5,888 s，12 個合計 60,634 s。
- **分析**（STATUS_DATA 09:00 第 6 步之 1）：run `20261106-074822-icpe-trx-analyze-sa_main-ttl3`，exit 0，stderr 空。`next_sim_analyze.py --sweep results/icpe_next/b/sa_main/*/x1_sweep.csv.xz --out-prefix sa_main_ttl3 --pf-target cpu`（`export RUN_ID`、`nice -n 10`；git HEAD de566f7，`code/` 和 8ef0ada 相同）。headroom_components 3,888 列、violations 5,322 列 → `results/icpe_next/b/analysis/sa_main_ttl3/`（兩個都 xz，解壓後和 run 目錄的原檔 `cmp` 相同；`manifest.json` 照 `analysis/cw_main_ttl3/` 的格式）。
- **`ttl7cells`**（第 6 步之 2）：run `20261106-074827-icpe-trx-ttl7cells-sa_main`，exit 0：`cells=466／648  (trace,rule,cpu,slow)=253  jobs=40480` → `results/icpe_next/b/sa_main_ttl7/cells.csv`（每列有 run_id、ts、host、src_analysis_run_id）。
- 上兩項 commit f0b13bb。不解讀、不下判定。

**使用者決定（11/6，AskUserQuestion）**：TRX40 **只做 SemiAnalysis**。FreeInference、Bailian 的 P1–P3 不在 TRX40 跑。

**還沒決定**：`sa_main` 之後 TRX40 跑什麼（Python 照 09:00 跑 `sa_main_ttl7`＋`sens_rep sa_*`，或改 Rust 跑偏離 6 的 `sa_p1`→`sens_rep sa_*`→`sa_p2`→`sa_p3`）。已再問使用者。在那之前 TRX40 不跑模擬、不 merge。
- git 上（fetch 07:5x）：`origin/icpe-next-data` @ bc1e040 已 merge Rust 引擎（875fb2b → 304fd72）、G0（rust）PASS、佇列加 `run --engine rust`；`origin/icpe-next-sim` @ 47a4617 再加 W4 雜訊 B1（G1 PASS，STATUS_SIM 06:14），icpe-data 還沒 merge 這個。

## 2026-11-06T03:25+08:00 — Rust 引擎 G1 PASS（STATUS_SIM 03:15）＋偏離 6 已登記 → `sa_main` 之後跑什麼，問使用者

- **git 上看到的**（fetch 03:2x）：
  - `origin/icpe-next-sim` @ 875fb2b，STATUS_SIM 03:15：`x1 sweep --engine rust`（程式 0b0ef1e），G1 py／rust 都 4,416×29、0 格不同；xcheck 14 個 block（含 xc_sa）0 格不同。預設仍是 py。code/p3x 有改 → merge 後要重做 G0、xc_sa。
  - `origin/icpe-next-data` @ 3102df4：PREREG 偏離 6（1546086；P1＝全部格子 7 個 TTL、② ④ 同一組＋截尾規則；P2 nvme807＋只載入；P3 wt＋預取進 GPU）；`next_b_queue.py` 加了 `sa_p1`／`sa_p2`／`sa_p3`（3385802）；使用者 11/6：Rust 的佇列 CPU 約 1.25×workers。**icpe-data 還沒 merge 0b0ef1e。**
  - TRX40 沒有 Rust 工具鏈（`cargo`、`rustc` 都不在）。
- 〔算術〕SemiAnalysis 的 P1–P3＝每個變體 11,304＋10,368＋7,470＝29,142 job × 12＝349,704 job。照 `sa_main` 在 TRX40 的 Python 速度（40 worker 約 1.2 job/s）約 80 h，11/9 前跑不完；Rust 在 TRX40 的速度 NOT_MEASURED。
- **問使用者**：`sa_main` 跑完（約 04:40）之後，TRX40 照 STATUS_DATA 09:00 用 Python 跑 `sa_main_ttl7`＋`sens_rep sa_*`，還是改成 Rust 跑偏離 6 的 `sa_p1`（取代 `sa_main_ttl7`）→ `sens_rep sa_*` → `sa_p2` → `sa_p3`（要在 `$ROOT` 裝 Rust 工具鏈、merge、G0＋xc_sa py／rust 都比）。不管選哪個，3 個 TTL 的分析＋`ttl7cells` 照做（幾秒，偏離 6：沒做完的優先度照偏離 4／5 報告）。

## 2026-11-06T03:03+08:00 — `sa_main` 第 11／12 個 block 完成（`sa_N64_s3`）

- run `20261106-012556-icpe-trx-sa_main-sa_N64_s3`：exit 0，sweep 5,750 s（queue 01:25:56 → 03:01:57），6,174 job，7,344 列 → collect 進 `results/icpe_next/b/sa_main/sa_N64_s3/`（`x1_sweep.csv.xz` 601 KiB＋`manifest.json`；`run_id` 單一值、`host`＝`acvlab-TRX40-AORUS-MASTER`）。
- 記憶體（mem.log 01:25:56–03:01:57，181 筆）：本組 PSS 合計最高 10.4 GiB，MemAvailable 最低 45.8 GiB；swap 980–981 MiB。兩個保護都沒觸發。
- 最後一個 `sa_N72_s3` 03:01:57 開跑。〔算術〕照 s1／s2 的 N72 約 5,850–5,890 s → 約 11/6 04:40 跑完。

## 2026-11-06T01:27+08:00 — `sa_main` 第 10／12 個 block 完成（`sa_N32_s3`）

- run `20261106-000537-icpe-trx-sa_main-sa_N32_s3`：exit 0，sweep 4,807 s（queue 00:05:37 → 01:25:56），6,174 job，7,344 列 → collect 進 `results/icpe_next/b/sa_main/sa_N32_s3/`（`x1_sweep.csv.xz` 571 KiB＋`manifest.json`；`run_id` 單一值、`host`＝`acvlab-TRX40-AORUS-MASTER`）。
- 記憶體（mem.log 00:05:37–01:25:56，151 筆）：本組 PSS 合計最高 10.0 GiB，MemAvailable 最低 45.9 GiB；swap 981 MiB。兩個保護都沒觸發。
- 下一個 `sa_N64_s3` 01:25:56 開跑。〔算術〕剩 2 個 block，照 s1／s2 同 N 的時間約 3.2 h → `sa_main` 約 11/6 04:40 跑完。

## 2026-11-06T00:08+08:00 — `sa_main` 第 9／12 個 block 完成（`sa_N16_s3`）；icpe-ana 的 Rust 引擎與補跑草案（記錄，現在不動）

- run `20261105-230217-icpe-trx-sa_main-sa_N16_s3`：exit 0，sweep 3,789 s（queue 23:02:17 → 00:05:37），6,174 job，7,344 列 → collect 進 `results/icpe_next/b/sa_main/sa_N16_s3/`（`x1_sweep.csv.xz` 537 KiB＋`manifest.json`；`run_id` 單一值、`host`＝`acvlab-TRX40-AORUS-MASTER`）。
- 記憶體（mem.log 23:02:17–00:05:37，120 筆）：本組 PSS 合計最高 9.8 GiB，MemAvailable 最低 46.5 GiB；swap 981 MiB。兩個保護都沒觸發。
- 下一個 `sa_N32_s3` 00:05:37 開跑。〔算術〕剩 3 個 block，照 s1／s2 同 N 的時間約 4.6 h → `sa_main` 約 11/6 04:40 跑完。
- **sa_main 跑完後的步驟已備好**（本機 `$ROOT/tools/sa_post_main.sh`，不進 repo，只做過 `bash -n`）：§4.1 的殼跑 `next_sim_analyze.py … --out-prefix sa_main_ttl3 --pf-target cpu`（`export RUN_ID`、`nice -n 10`）→ 另一個殼跑 `ttl7cells --plan sa_main`。表的放法照 icpe-data 的 `analysis/cw_main_ttl3/`（headroom_components xz＋violations＋manifest.json）。檢查內容後才 commit。
- **icpe-ana 的訊息**（資訊；以 git 為準，已核對 `origin/icpe-next-ana` 的 STATUS_ANA）：
  - aee1527（23:58 節）：Rust 版 x1 引擎，和 `x1_sim.py` 逐位元相同；用本分支 `sa_main/sa_N16_s1` 對過 7,344／7,344 列相同。要等 icpe-sim 在 STATUS_SIM 寫 G1／xcheck PASS 才用；只用在還沒開始的 block。**`sa_main` 照舊用 Python 跑完。**
  - 2e32946（00:20 節）：使用者 11/6 決定補回偏離 4 砍掉的格子（P1–P4），PREREG 偏離 6 由 icpe-data 寫（目前是草稿）。TRX40 的分工**草案**：SemiAnalysis 的補跑＋`sens_rep sa_*`，以及 SemiAnalysis、FreeInference、Bailian 的 P1–P3（Rust 要在 TRX40 另外編、另外對拍）。
  - icpe-trx 的立場：等偏離 6 正式登記＋STATUS_SIM 的驗證；在那之前照 STATUS_DATA 09:00 的步驟走（`sa_main_ttl7`、`sens_rep sa_*` 用目前的 Python 佇列，除非 STATUS 改了）。
  - **草案裡超出 TRX40 目前授權的兩點**（要使用者決定，輪到時問）：
    1. FreeInference 的資料能不能放 TRX40：STATUS_DATA 09:00 的授權清單只有 SemiAnalysis、Bailian（Apache-2.0）、TraceLab（CC BY 4.0）；FreeInference 不在清單上。
    2. 寫入範圍：使用者給 icpe-trx 的只有 `results/icpe_next/b/` 底下的 `sa_*`（＋`analysis/sa_*`、`sens_rep/sa_*`）。FreeInference、Bailian 的下載、轉換、結果目錄都在範圍外。

## 2026-11-05T23:03+08:00 — `sa_main` 第 8／12 個 block 完成（`sa_N72_s2`）；s2 四個都進 results

- run `20261105-212443-icpe-trx-sa_main-sa_N72_s2`：exit 0，sweep 5,842 s（queue 21:24:43 → 23:02:17），6,174 job，7,344 列 → collect 進 `results/icpe_next/b/sa_main/sa_N72_s2/`（`x1_sweep.csv.xz` 607 KiB＋`manifest.json`；`run_id` 單一值、`host`＝`acvlab-TRX40-AORUS-MASTER`）。
- 記憶體（mem.log 21:24:43–23:02:17，184 筆）：本組 PSS 合計最高 10.8 GiB，MemAvailable 最低 45.5 GiB；swap 981 MiB。兩個保護都沒觸發。
- 下一個 `sa_N16_s3` 23:02:17 開跑。
- 〔算術〕s2 四個 block 合計 20,207 s（s1：20,276 s）。剩 s3 的 4 個 → 約 5.6 h → `sa_main` 約 11/6 04:40 跑完。

## 2026-11-05T21:26+08:00 — `sa_main` 第 7／12 個 block 完成（`sa_N64_s2`）

- run `20261105-194923-icpe-trx-sa_main-sa_N64_s2`：exit 0，sweep 5,708 s（queue 19:49:23 → 21:24:43），6,174 job，7,344 列 → collect 進 `results/icpe_next/b/sa_main/sa_N64_s2/`（`x1_sweep.csv.xz` 601 KiB＋`manifest.json`；`run_id` 單一值、`host`＝`acvlab-TRX40-AORUS-MASTER`）。
- 記憶體（mem.log 19:49:23–21:24:43，180 筆）：本組 PSS 合計最高 10.2 GiB，MemAvailable 最低 46.0 GiB；swap 977–981 MiB。兩個保護都沒觸發。
- 下一個 `sa_N72_s2` 21:24:43 開跑。
- 〔算術〕剩 5 個 block；照 s1 同 N 的時間 → 約 7.3 h → `sa_main` 約 11/6 04:40 跑完。

## 2026-11-05T19:50+08:00 — `sa_main` 第 6／12 個 block 完成（`sa_N32_s2`）

- run `20261105-182807-icpe-trx-sa_main-sa_N32_s2`：exit 0，sweep 4,865 s（queue 18:28:07 → 19:49:23），6,174 job，7,344 列 → collect 進 `results/icpe_next/b/sa_main/sa_N32_s2/`（`x1_sweep.csv.xz` 576 KiB＋`manifest.json`；`run_id` 單一值、`host`＝`acvlab-TRX40-AORUS-MASTER`）。
- 記憶體（mem.log 18:28:07–19:49:23，153 筆）：本組 PSS 合計最高 10.1 GiB，MemAvailable 最低 45.8 GiB；swap 959–977 MiB。兩個保護都沒觸發。
- 下一個 `sa_N64_s2` 19:49:23 開跑。
- 〔算術〕剩 6 個 block；照 s1 同 N 的時間 → 約 8.9 h → `sa_main` 約 11/6 04:40 跑完。

## 2026-11-05T18:29+08:00 — `sa_main` 第 5／12 個 block 完成（`sa_N16_s2`）

- run `20261105-172446-icpe-trx-sa_main-sa_N16_s2`：exit 0，sweep 3,792 s（queue 17:24:46 → 18:28:07），6,174 job，7,344 列 → collect 進 `results/icpe_next/b/sa_main/sa_N16_s2/`（`x1_sweep.csv.xz` 536 KiB＋`manifest.json`；`run_id` 單一值、`host`＝`acvlab-TRX40-AORUS-MASTER`）。
- 記憶體（mem.log 17:24:46–18:28:07，119 筆）：本組 PSS 合計最高 9.5 GiB，MemAvailable 最低 46.6 GiB；swap 958–959 MiB。兩個保護都沒觸發。
- 下一個 `sa_N32_s2` 18:28:07 開跑。
- 〔算術〕剩 7 個 block；照 s1 同 N 的時間 → 約 10.2 h → `sa_main` 約 11/6 04:40 跑完（和 17:26 的估計相同）。

## 2026-11-05T17:26+08:00 — `sa_main` 第 4／12 個 block 完成（`sa_N72_s1`）；s1 四個都進 results

- run `20261105-154628-icpe-trx-sa_main-sa_N72_s1`：exit 0，sweep 5,888 s（queue 15:46:28 → 17:24:46），6,174 job，7,344 列 → collect 進 `results/icpe_next/b/sa_main/sa_N72_s1/`（`x1_sweep.csv.xz` 606 KiB＋`manifest.json`；`run_id` 單一值、`host`＝`acvlab-TRX40-AORUS-MASTER`）。
- 記憶體（mem.log 15:46:28–17:24:46，186 筆）：本組 PSS 合計最高 10.4 GiB，MemAvailable 最低 44.1 GiB；swap 959–978 MiB。兩個保護都沒觸發。
- 下一個 `sa_N16_s2` 17:24:46 開跑。
- 〔算術〕s1 四個 block 合計 20,276 s（N16 3,903／N32 4,771／N64 5,714／N72 5,888）。剩 8 個＝s2、s3 各 4 個；假設和 s1 同 N 一樣快 → 約 11.3 h → `sa_main` 約 11/6 04:40 跑完。

## 2026-11-05T15:48+08:00 — `sa_main` 第 3／12 個 block 完成（`sa_N64_s1`）

- run `20261105-141103-icpe-trx-sa_main-sa_N64_s1`：exit 0，sweep 5,714 s（queue 14:11:03 → 15:46:28），6,174 job，7,344 列 → collect 進 `results/icpe_next/b/sa_main/sa_N64_s1/`（`x1_sweep.csv.xz` 601 KiB＋`manifest.json`；`run_id` 單一值、`host`＝`acvlab-TRX40-AORUS-MASTER`）。
- 記憶體（mem.log 14:11:03–15:46:28，180 筆）：本組 PSS 合計最高 10.3 GiB，MemAvailable 最低 46.0 GiB；swap 2,357 → 978 MiB（往下掉：別人的頁換回記憶體）。兩個保護都沒觸發。
- 下一個 `sa_N72_s1` 15:46:28 開跑。
- 〔算術〕s1 三個 block：N16 3,903 s、N32 4,771 s、N64 5,714 s。剩 9 個 block（N72_s1＋s2、s3 各 4 個）；假設 s2、s3 和 s1 同 N 的時間一樣、N72 每個 5,700–6,500 s → 合計約 12.7–13.4 h → `sa_main` 約 11/6 04:30–05:30 跑完。

## 2026-11-05T14:13+08:00 — `sa_main` 第 2／12 個 block 完成（`sa_N32_s1`）

- run `20261105-125122-icpe-trx-sa_main-sa_N32_s1`：exit 0，sweep 4,771 s（queue 12:51:22 → 14:11:03），6,174 job，7,344 列 → collect 進 `results/icpe_next/b/sa_main/sa_N32_s1/`（`x1_sweep.csv.xz` 577 KiB＋`manifest.json`；`run_id` 單一值、`host`＝`acvlab-TRX40-AORUS-MASTER`）。commit 8523791。
- 記憶體（同一個 mem.log，12:51:22–14:11:03，150 筆）：本組 PSS 合計最高 10.2 GiB，MemAvailable 最低 47.8 GiB；swap 2,357–2,401 MiB（沒有往上漲）。兩個保護都還在跑、都沒觸發。
- 下一個 `sa_N64_s1` 14:11:03 開跑。
- 〔算術〕N32 比 N16 多 22%（4,771 vs 3,903 s）。剩 10 個 block：N16×2 約 130 分、N32×2 約 160 分，N64／N72 共 6 個還沒量到——每個抓 80–110 分 → 合計約 13–16 h → `sa_main` 約 11/6 03:00–06:00 跑完。

## 2026-11-05T12:53+08:00 — `sa_main` 第 1／12 個 block 完成（`sa_N16_s1`）

- run `20261105-114609-icpe-trx-sa_main-sa_N16_s1`：exit 0，sweep 3,903 s（queue 記 3,912.7 s），6,174 job，7,344 列 → collect 進 `results/icpe_next/b/sa_main/sa_N16_s1/`（`x1_sweep.csv.xz` 548 KiB＋`manifest.json`；`host`＝`acvlab-TRX40-AORUS-MASTER`）。
- 記憶體（`queue-sa_main-w40-20261105-114609.mem.log`）：本組 PSS 合計最高 9.7 GiB（11:55:10），MemAvailable 最低 49.6 GiB；swap 2,442 → 2,406 MiB（沒有增加）。兩個保護都沒觸發。
- 下一個 `sa_N32_s1` 12:51:22 開跑（同一條佇列）。
- 〔算術〕剩 11 個 block × 65 分鐘 ≈ 12 h；CacheWise 的 block 隨 N 變大最多慢約 40%，所以約 12–17 h → `sa_main` 約 11/6 01:00–06:00 跑完。之後是分析＋`ttl7cells`、`sa_main_ttl7`、`sens_rep 'sa_*'`。

## 2026-11-05T11:47+08:00 — G1 PASS＋merge；**`sa_main` 開跑（40 worker、maxtasks 4）**

- icpe-sim：54fa2d8 的 **G1 PASS**（STATUS_SIM 984dc4f 11:45，run `20261105-100635-icpe-sim-g1-54fa2d8`，4,416 列 × 29 欄 0 格不同、自檢 997／997；我在 git 上看過那一節）。原因：開 GPU 層時 Engine 有循環參照、fork 出來的 worker 幾乎不做 gen2 gc（SIM_CHANGES 第 15 項）。
- icpe-data：merge 進 `icpe-next-data` @ 2dc6c8a。
- 我：merge `origin/icpe-next-data` @ 2dc6c8a → **`8ef0ada`**。`git diff 54fa2d8 HEAD -- code/p3x code/icpe/p3_fvar.py code/p3_platform.py` **是空的** → 11:08 節在 54fa2d8 上做的 G0（128,064 格 0 不同）、xc_sa（1,280 格 0 不同）適用。`next_b_queue.py`、`next_b_g0.py` 和 828e1d3 相同。
- **開跑**：11:46:09，`python -I code/icpe/next_b_queue.py run --plans sa_main --workers 40`（程式 8ef0ada；`X1_MAXTASKS` 沒設 → stdout `maxtasksperchild=4`；log `$ROOT/runs/queue-sa_main-w40-20261105-114609.log`）。保護：memwatch（MemAvailable < 6 GiB 停）、swapguard（swap > 2,933 MiB 停，使用者 09:4x 的上限）。開跑時 MemAvailable 58.3 GiB、swap 2,442 MiB（09:57 那次留下的別人的頁）、load 0.5。
- 第一個 block `sa_N16_s1`（run `20261105-114609-icpe-trx-sa_main-sa_N16_s1`）。每個 block 跑完：`collect --plans sa_main` → commit → push。

## 2026-11-05T11:08+08:00 — 54fa2d8 上提前做 G0、xc_sa：都 IDENTICAL；仍等 G1 PASS＋icpe-data merge

- 為了 G1／merge 一到就能開跑，先在 detached worktree `$ROOT/wt-sim-54fa2d8`（icpe-sim 54fa2d8，`--maxtasks` 預設 4）做我這邊要重做的兩個對拍：
  - **G0**：run `20261105-105553-icpe-trx-g0-54fa2d8`（16 worker、新功能全關；stdout 印 `maxtasksperchild=4`；sweep 679 s、`/usr/bin/time` 11:21）。
  - **xc_sa**：run `20261105-105555-icpe-trx-xcheck-xc_sa-54fa2d8`（同 queue 的 xc_sa 參數、16 worker）。不 collect。
  - 兩個 run 的 cmd.sh 最後一步的比對失敗（`next_b_g0.py` 是 icpe-data 的檔，不在 icpe-sim 的 worktree：`can't open file '.../wt-sim-54fa2d8/code/icpe/next_b_g0.py'`，exit 2）；sweep 本身都跑完。兩個目錄留了 `COMPARE_FAILED_SEE_NEXT_RUN`。
  - **比對**：run `20261105-110736-icpe-trx-compare-54fa2d8`（repo 828e1d3 的 `next_b_g0.py`）：
    - G0 對 `results/icpe_h200/t1/p01/sim_V1/x1/x1_sweep.csv.gz`：**4,416 列 × 29 欄＝128,064 格，0 個不同**。
    - xc_sa 對 `results/icpe_next/b/xcheck/xc_sa/x1_sweep.csv.xz`：**32 列 × 40 欄＝1,280 格，0 個不同**。
- 開跑條件：STATUS_SIM 寫 54fa2d8 的 G1 PASS ＋ icpe-data merge ＋ 我 merge 後 `git diff 54fa2d8 HEAD -- code/p3x code/icpe/p3_fvar.py code/p3_platform.py` 是空的（是空的，上面兩個對拍就適用；不是空的就重做）。然後 `sa_main` 40 worker。

## 2026-11-05T10:25+08:00 — icpe-sim 的修正 54fa2d8（`--maxtasks`，預設 4）在 TRX40 試跑：40 worker 記憶體持平；等 G1＋merge

- icpe-sim 54fa2d8（`origin/icpe-next-sim`）：`x1 sweep --maxtasks`（預設 4、`$X1_MAXTASKS`；0＝原本）→ `Pool(maxtasksperchild)`。G1 跑中（icpe-sim 10:1x 的訊息；以 STATUS_SIM 為準）。
- **試跑**（不進 results/；程式＝detached worktree `$ROOT/wt-sim-54fa2d8` @ 54fa2d8，G1 未過）：run `20261105-100804-icpe-trx-trial-maxtasks4-w40`——sa_N16_s1、和 `sa_main` 相同的 sweep 參數、**40 worker**、`maxtasksperchild=4`（stdout 有印）。跑到 1,329／6,174 job（968 s，涵蓋最重的 O-ALL、O-PF、LRU+RPF 段）後手動停（`STOPPED` 檔）。
  - 本組 PSS 合計：最高 **9.4 GiB**（10:09:46），之後在 7.3–8.8 GiB 之間，沒有往上漲（修正前同一段：889 job 時 39.4 GiB）。
  - MemAvailable 最低 50.2 GiB；swap 2,495 → 2,486 MiB（沒有增加）。
  - 速度：1,329 job／968 s＝每秒 1.37 個（修正前 40 worker 前 540 個：1.37）。
- **計畫**：G1 PASS（STATUS_SIM）→ icpe-data merge 進 `icpe-next-data` → 我 merge、G0＋xc_sa（新程式）→ `sa_main` 用 **40 worker**（使用者先前決定的值；上面的量測顯示記憶體夠）。
- 〔算術〕照每秒 1.37 個 job：一個 block 約 75 分鐘，`sa_main` 12 個約 15 h（N 大的 block 可能更久，CacheWise 多約 40%）。

## 2026-11-05T10:05+08:00 — **卡住**：worker 的記憶體每個 job 都在累積、不釋放 → `sa_main` 在 62 GiB 上跑不完；20 worker 的 block 被 swap 保護停掉

**發生什麼**
- 20 worker 的 run `20261105-092452-icpe-trx-sa_main-sa_N16_s1`：**09:57:27 swapguard 觸發**（swap 3,125 MiB > 使用者允許的 2,933 MiB），停在 1,958／6,174 job（1,946 s）。留了 `ABORTED`、`SWAPGUARD_KILLED`，不 collect。停掉前本組 PSS 49.3 GiB、MemAvailable 10.4 GiB。停掉後 MemAvailable 59.5 GiB、swap 2,503 MiB（別人的頁，慢慢換回）。
- 兩次 run 的本組 PSS 合計 vs 已跑的 job 數（`mem.log`＋stdout 的進度列）：

| run | 已跑 job | 本組 PSS 合計 | 合計 ÷ job 數 |
|:--|:--|:--|:--|
| 40 worker | 889 | 39.4 GiB | 44 MiB |
| 20 worker | 約 950（09:41:54） | 36.5 GiB | 約 38 MiB |
| 20 worker | 約 1,410（09:49:12） | 43.0 GiB | 約 30 MiB |
| 20 worker | 1,958（09:57） | 49.3 GiB | 約 25 MiB |

  合計大致跟「已跑的 job 數」走，和 worker 數關係不大。重的 job（O-ALL、O-PF、LRU+RPF）漲得快，後面的（O-BEL、PENS）慢一些，但**沒有持平**。

**診斷試跑**（不進 results/）：run `20261105-095815-icpe-trx-trial-memgrowth-w1`——1 worker、SA N16 s1、CPU 128、PENS／LRU2／GDSF／ARC／S3FIFO／LRU × GPU LRU／BEL、ssd、cake，14 個 job（142 s）；每 2 s 記 worker 的 `smaps_rollup`（`worker_rss.log`）。
- worker 的 Private_Dirty：開始 148 MiB → 14 個 job 後 388 MiB，**只升不降**；有的 job 一次 +45 MiB，有的持平（GDSF 4 個 job 只在第一個 +44）。RSS 958 → 1,198 MiB。
- 結論（這台量到的）：worker 每跑一個 job 都留下一些記憶體、之後不還。`sa_main` 一個 block 6,174 個 job，20 worker 時每個 worker 約 309 個 → 照上面的量推〔算術〕每個 worker 5–14 GiB，這台 62 GiB 不管幾個 worker 都裝不下。
- 是程式裡的洩漏還是 malloc 碎片，我分不出來；沒看程式改什麼，也沒有改。

**需要 icpe-sim 做什麼**（擋 TRX40 的全部 SA 工作）
- `x1_sim.py` 2278 行的 `mp.get_context("fork").Pool(a.workers)` 加 `maxtasksperchild`（例如 8–16），或找出跨 job 留下的東西。前者不改任何 job 的計算，worker 只是跑完 k 個 job 就換一個新的（從父行程重新 fork）。要做 G1。
- acvlab 也一樣會累積，只是 440 GiB 看不出來；TraceLab（記憶體較大）40 worker 可能也值得看一下 mem-watch。

**需要 icpe-data 做什麼**：上面的修正 G1 過了以後 merge 進 `icpe-next-data`；我 merge、重做 G0＋xc_sa、再開 `sa_main`。

**使用者決定（11/5 10:0x，在 icpe-trx）**：選「請 icpe-sim 加 maxtasksperchild」。已用 SendMessage 通知 icpe-sim（請求）與 icpe-data（告知）；以本檔為準。在修正 G1 過、merge 進 `icpe-next-data` 之前，TRX40 不跑任何 SA 的模擬。

## 2026-11-05T09:41+08:00 — 20 worker 的記憶體也在漲；**使用者：允許 swap 小幅增加**

- 20 worker（run `20261105-092452-icpe-trx-sa_main-sa_N16_s1`）09:40:20：本組 PSS 34.4 GiB（09:36 是 25.8 GiB），MemAvailable 23.5 GiB，swap 885 MiB（還沒增加）。09:38 時每個 worker PSS 中位數 1.54 GiB、最大 1.87 GiB；RSS 中位數 2.29 GiB、最大 2.62 GiB。
- 父行程（p3_fvar）RSS 0.95 GiB，其中和 worker 共用約 0.76 GiB；worker 的 RSS 已經比父行程大，所以不只是 copy-on-write。
- `x1_sim.py` 把重的 job（預取、scan）排在最前面先跑（`J.sort(...)`，2266 行），所以現在這段是記憶體最高的一段；每個 worker 的峰值**沒量到**。程式裡除了 `_PCACHE`（每份紀錄一個，父行程 fork 前就載好）沒有跨 job 的快取。
- 09:39 加了 swap 保護 `$ROOT/tools/swapguard.sh`（swap 用量 > 門檻就 SIGTERM 整組），門檻先設 1,141 MiB（基準＋256）。
- **使用者決定（11/5 09:4x，在 icpe-trx）**：選「允許 swap 小幅增加」＝swap 最多比基準多 2 GiB（門檻 **2,933 MiB**），MemAvailable < 6 GiB 才停。被換出去的是別的使用者閒置的桌面程式（gnome-shell 等）的頁，不影響模擬結果。這是對「不准用到 swap」的放寬，使用者明講。
  - 另兩個選項（沒選）：繼續 20、被停就降 12；現在就停、改 12 重跑。

## 2026-11-05T09:26+08:00 — 40 worker 記憶體不夠、swap 增加 → 停掉作廢；**`sa_main` 改 20 worker 重跑**

**發生什麼**
- run `20261105-091241-icpe-trx-sa_main-sa_N16_s1`（40 worker）跑到 889／6174 job（681 s），**09:24 我手動停掉**（`kill -TERM` 整個行程群組），留了 `ABORTED` 檔，不 collect。
- 記憶體（`$ROOT/runs/queue-sa_main-w40-20261105-091241.{mem.log,pss-5min.txt,pss-abort.txt}`）：

| 時間 | 本組 PSS 合計 | 每個 worker PSS | MemAvailable | swap 用量 |
|:--|:--|:--|:--|:--|
| 09:13:14 | 5.5 GiB | — | 50.2 GiB | 245 MiB |
| 09:17:49（開跑 5 分鐘） | 20.0 GiB | 中位數 501 MiB、p90 552、最大 637 | 36 GiB | 244 MiB |
| 09:21:06 | 28.9 GiB | — | 26.9 GiB | 264 MiB（開始增加） |
| 09:23:4x（停掉前） | 39.4 GiB | 中位數 968 MiB、最大 1,248；私有中位數 949 | 19.4 GiB | 885 MiB |

- swap 的增加不是本組被換出（本組每個行程 `VmSwap` 都是 0），是記憶體壓力把別的使用者閒置的頁（gnome-shell、update-manager、uid 1010 的行程）擠進 swap。照規則「不准用到 swap」，仍然算本組造成的，所以停。
- 漲法：09:17:49 → 09:19:00 只多 1.5 GiB；09:19 以後每 30 s 多約 2 GiB，時間上和 job 從 O-ALL（前 432 個）換到 O-PF／LRU+RPF 一致。所以看起來是這類 job 比較吃記憶體，不像單純越跑越漏；每個 worker 的上限**沒量到**。
- 停掉後 swap 留在 885 MiB（被換出的頁要等那些行程用到才換回）。之後的基準改成 885 MiB。

**處理**：照使用者 09:08 的決定（被停就作廢、降 worker 重跑）。09:24:52 重開 `run --plans sa_main --workers 20`（程式 828e1d3；log `$ROOT/runs/queue-sa_main-w20-20261105-092452.log`；記憶體監看同前）。20 是我依上面的量測選的：每個 worker 以 1.25–1.5 GiB 估，合計 25–30 GiB；本組約 29 GiB 時 swap 開始增加。之後在 block 之間依量到的峰值調整。

**對時間的影響**〔算術〕：40 worker 時每秒約 1.37 個 job（前 540 個）。20 worker 約一半，一個 block 約 2.5 h，`sa_main` 12 個約 30–40 h。實際時間等第一個 block 跑完再寫。

**需要 icpe-sim 知道**（不擋，只是告知）：SemiAnalysis（98,827 請求、4,200 萬 chunk）每個 worker 在 O-PF／LRU+RPF 的 job 上 PSS 約 1–1.25 GiB 而且還在漲（PSS 接近 RSS 的 2 GiB）。如果是 fork 後的 copy-on-write（refcount 碰到父行程的物件），共用的部分會慢慢變私有。這台 62 GiB，worker 數因此受限。

## 2026-11-05T09:14+08:00 — 指紋（不含 meta_json）13／13 相同；G0 PASS；xc_sa IDENTICAL；**`sa_main` 開跑（40 worker）**

**做完什麼**
- **merge** `origin/icpe-next-data` @ ec74563 → `828e1d3`（`--no-ff`）。模擬程式 `code/p3x`、`code/icpe/p3_fvar.py`、`code/p3_platform.py` 和 f4f5668 逐位元相同（`git diff` 空）；改到的只有 `next_b_queue.py`（digest 跳過 `meta_json`）、`next_sim_analyze.py`（`--rival-only-gpu`）、`next_a_*` 與文件。
- **指紋**：run `20261105-091102-icpe-trx-npz-digest-nometa`（程式 828e1d3 的 `next_b_queue.py digest`）對 `npz_digest.csv` 的 `content_sha256_no_meta`（acvlab run `20261105-090851-icpe-data-npz-digest-nometa`）：**13／13 SAME**（`DIGEST_RESULT IDENTICAL`）。和本檔 09:07 節的表也相同。icpe-data 的訊息說它比過也是 13／13；這裡是我自己重算、重比的結果。
- **4. G0 PASS**：run `20261105-090000-icpe-trx-g0-f4f5668`（程式 f4f5668、新功能全關、16 worker、nice 10）：
  - 指令：`P3_CONST=h200 P3_TIER=h200cpu M7_TIER_PARAMS=results/icpe_h200/t1/tier_params_h200cpu.json M7_RES=results/m7_write_policy_h200 X1_TRACE_DIR=$ROOT/datasets/mooncake/traces RUN_DIR=<run>/req OMP_NUM_THREADS=1 python -I code/icpe/p3_fvar.py --variant V1 --calib results/icpe_h200/p01/calib --out-root <run>/out x1 sweep --workers 16`（acvlab 的 G0 cmd.sh 在 acvlab 上、這裡看不到；照 STATUS_SIM G1 的指令＋queue 的環境變數組的）。
  - `next_b_g0.py results/icpe_h200/t1/p01/sim_V1/x1/x1_sweep.csv.gz <run>/out/x1/x1_sweep.csv`：**4,416 列 × 29 欄＝128,064 格，0 個不同**（`G0_RESULT IDENTICAL`）。
  - sweep 1,328 個 job 687 s（16 worker）；最大 RSS（單一行程）385 MiB；整組 PSS 最多 2.96 GiB。
  - G0 跑在 merge 前的 f4f5668；merge 沒有改模擬程式（上面），所以沒有重做。
- **5. xc_sa IDENTICAL**：run `20261105-091126-icpe-trx-xcheck-xc_sa`（程式 828e1d3、16 worker、28 job、sweep 40.6 s），比對 run `20261105-091229-icpe-trx-xcheck-compare-xc_sa`：`next_b_g0.py results/icpe_next/b/xcheck/xc_sa/x1_sweep.csv.xz <run>/out/x1/x1_sweep.csv` → **32 列 × 40 欄＝1,280 格，0 個不同**。沒有 collect。
  - 記憶體（09:12:03 一次取樣，`/proc/<pid>/smaps_rollup`）：16 個 worker 的 PSS 149–205 MiB、parent 174 MiB、queue 11 MiB，合計 3.2 GiB。這是 CPU 64 GiB 的小格子；`sa_main` 有到 1,024 GiB 的容量，worker 可能更大，開跑後再量。
- **6. `sa_main` 開跑**：09:12:41，`python -I code/icpe/next_b_queue.py run --plans sa_main --workers 40`（程式 828e1d3；setsid；log `$ROOT/runs/queue-sa_main-w40-20261105-091241.log`；記憶體監看 `…091241.mem.log`，每 30 s，MemAvailable < 6 GiB 就停整組）。開跑時 MemAvailable 55 GiB、swap 245 MiB（基準）、load 6.1。第一個 block `sa_N16_s1`。

**卡住什麼**：無。

**需要別的 session 做什麼**：無。

**需要使用者決定什麼**：無。

## 2026-11-05T09:07+08:00 — 轉換完成；**npz 內容指紋 13／13 和 acvlab 不同 → 停**（原因：指紋含 `meta_json`）；G0 跑中；**等 icpe-data 的新指紋**

**做完什麼**
- **3. 轉換**：run `20261105-085858-icpe-trx-convert-semianalysis`（exit 0）。p3x＝`git archive 6b17191 code/p3x` 解到 `$ROOT/code-archive/p3x-6b17191`。
  - `TIARA_DATASETS=$ROOT/datasets python -I code/icpe/next_b_convert.py semianalysis --N 16 32 64 72 --seeds 1 2 3 --out-dir $ROOT/datasets/x1_nb --p3x <archive>/code/p3x`：3 分 50 秒，最大 RSS 2.03 GiB（`/usr/bin/time -v`）。
  - 截斷版 `semianalysis --N 16 --seeds 1 --trunc-128k ...`：48 秒，最大 RSS 1.66 GiB。
  - 規則 6：`len(hash_ids)×64=in` 98827／98827。
- **指紋比對**：run `20261105-090355-icpe-trx-npz-digest`（exit 1）：`next_b_queue.py digest <13 個名字>` 和 `results/icpe_next/b/traces/npz_digest.csv` **13 個全部不同**（`DIGEST_RESULT DIFFER n=13 bad=13`）。照步驟 3「不同就停」：**SA 的 xcheck／正式模擬都沒開始。**

**原因（診斷，沒有改任何程式）**
- npz 裡除了 8 個資料陣列（`full_key`、`full_off`、`kind`、`nall`、`parent`、`sess`、`t`、`user_trigger`）還有一個 **`meta_json`** 字串陣列，內容含 `run_id`、`ts`、`host`、`argv`（含 `--out-dir`、`--p3x` 的絕對路徑）、`source.file`（絕對路徑）。
- `cmd_digest` 把 `meta_json` 也算進去 → 換一台機器、甚至同一台重轉一次，指紋必然不同。所以 `npz_digest.csv` 的值在別台**照定義就不可能相同**。
- 診斷 run `20261105-090538-icpe-trx-digest-diag`（腳本在該 run 目錄的 `digest_diag.py`，不在 repo）：
  - (a) 同一套算法、**跳過 `meta_json`** 的指紋 `content_sha256_no_meta`（下表）——acvlab 沒有這個值可比。
  - (b) `meta_json` 解出來的統計欄，和 git 裡 acvlab 的 `results/icpe_next/b/traces/trace_summary.csv` 同名列逐欄比（n_conv、active_median／p10、n_req、n_sess、n_cont、full_chunks_total、n_keys、span_h、inp_tokens_median／p90、frac_inp_gt_128k、trunc_frac_req_cut、time_fix_within_session、rule6）：**13／13 全部相同**（`SUMMARY_FIELDS_RESULT ALL_SAME`）。這是彙總量相同，不是逐位元相同。
  - 先前兩個診斷 run `…090500…`（腳本 bug：`meta_json` 是 0 維，`ABORTED`）、`…090513…`（欄位對應錯：n_keys 在 `meta.check`，`SUPERSEDED`）作廢。

| name | content_sha256_no_meta（TRX40） |
|:--|:--|
| semianalysis_N16_s1 | 3d6192cae774cdb66036527301ded814692b0b457d1f8c926907a9cb4c962b04 |
| semianalysis_N16_s1_t128k | f51da2d56ca961a47110ea13061ae2ee28b01879ad42b18956b25a461e81608f |
| semianalysis_N16_s2 | 460bb9572953b039ba3d589a97afa80549fd6ec309aee8fc62f09f93890b52fd |
| semianalysis_N16_s3 | 8ab19616850d09c155f58b9d53838dab92ebf0b4321cd9133d05c7068120eefe |
| semianalysis_N32_s1 | 517b8bbe9ce428ba7bb76eb76aba54ddf9404bb6c5162fb2f6680efa743a8b7a |
| semianalysis_N32_s2 | 0fb05f6661031aa0b8b8b93bee3a4a4484c9d02875d66f4598904ff05656ff4c |
| semianalysis_N32_s3 | 6ceac31f0c802142e7ba0f8ac9fe6d8f9ba8a6699bc0604743b3258773481584 |
| semianalysis_N64_s1 | f844e1a3b954bfccbff9814c25d4cddb8da50e69030aae5675f48fddc3b0c038 |
| semianalysis_N64_s2 | f5fd5d2534b099cf57e7f63261992c3c4444df73da48bde2378ace2aa18e54b4 |
| semianalysis_N64_s3 | 2074092728cf504f15fcf7337bff9114ada9008b11bec727b91ac7a8fa2d09b6 |
| semianalysis_N72_s1 | ef0f0a9fbd1832f7025886376353475a1320328ba2bc7235144352867d5553a1 |
| semianalysis_N72_s2 | d12b3468afc7c667bea791332e1f7a879bf9e748cd1733cce8ce4ef6049872e6 |
| semianalysis_N72_s3 | af8752b4bf015d9b2227be6d51c6d71594d7cdf36edc8bf829958de1ff190686 |

**跑中**：G0 run `20261105-090000-icpe-trx-g0-f4f5668`（Mooncake，16 worker；和 SA 的紀錄無關，所以沒停）。

**卡住什麼**：SA 紀錄的跨機器相等性沒有逐位元的證據（上面的原因）。

**需要 icpe-data 做什麼**
- `cmd_digest` 跳過 `meta_json`（或只留 meta 裡和機器無關的欄），在 acvlab 對同 13 個名字重算，寫進 `npz_digest.csv`（或另一個欄）。我會和上表逐名比。上表的算法＝`cmd_digest` 原樣、只在 `k == "meta_json"` 時 `continue`。
- 同一個問題也影響 h200 的 Bailian／FI 指紋比對。

**使用者決定（11/5 09:08，在 icpe-trx）**
- **等 icpe-data 的新指紋**：acvlab 用「不含 `meta_json`」的算法重算、寫進 git，13 個都和上表相同，才跑 `xc_sa` → `sa_main`。在那之前 SA 什麼都不跑（xcheck 也不跑）。
- **`sa_main` 直接 40 worker**（使用者：「CPU：48 執行緒 改吃40?」→ 選「sa_main 直接 40」）：第一個 block 就開 40，不先用 16 量。保險＝記憶體監看（MemAvailable < 6 GiB 就停整組；被停的 block 作廢、降 worker 重跑）。swap 以 245 MiB 為基準，增加就記錄並回報。

## 2026-11-05T09:00+08:00 — repo、venv、下載完成；轉換＋G0 跑中

**機器**：`acvlab-TRX40-AORUS-MASTER`，Threadripper 3960X 48 執行緒，62 GiB RAM，swap 8 GiB（開跑前別人已用約 245 MiB，以這個為基準，不准再增加）。Titan RTX 不碰。

**做完什麼**
- **0. Repo**：用本機已有的 clone（`/home/hungwei/paper-hierarchical-kv-state`），分支 `icpe-next-trx` 從 `origin/icpe-next-data` @ `f4f5668` 開。
  - `$ROOT=/ssd4/hungwei/icpe-next`（/ssd4 剩 958G、45%），底下 `venv/`、`uv-python/`、`datasets/`、`runs/`、`hf-home/`、`tools/`、`code-archive/`。程式在 repo 裡，和資料分開。
- **1. venv**：`$ROOT/venv`，Python **3.12.15**、numpy **2.3.5**、pandas **3.0.5**、pyarrow **25.0.1**、python-dateutil **4.0.0**、tzdata **2026.6**（`uv pip freeze` 只有這 5 個）。
  - 這台沒有 `uv`、`curl`、`pip`。另外用 micromamba（conda-forge）建工具環境 `$ROOT/tools/env`（uv 0.13.7、curl 8.23.0、huggingface_hub 的 `hf` 2.4.0），只用來裝 Python 與下載，和模擬用的 venv 分開。
  - Python：`uv python install 3.12.15 --install-dir $ROOT/uv-python --no-bin`（cpython-3.12.15-linux-x86_64-gnu）。
  - repo 的 `.claude/settings.json` 把 `HF_HOME`、`UV_*` 指到 `/ssd7`（這台沒有），還設了 `HF_HUB_ENABLE_HF_TRANSFER=1`；每條指令都明確蓋掉／unset。
- **2. 下載**：run `20261105-085632-icpe-trx-dl-semianalysis-mooncake`（exit 0）。`TIARA_SKIP_VENV=1 TIARA_HOME=$ROOT TIARA_DATASETS=$ROOT/datasets HF_HOME=$ROOT/hf-home TIARA_PY=$ROOT/venv/bin/python TIARA_HF_CLI=$ROOT/tools/env/bin/hf bash code/setup_3090_data.sh semianalysis mooncake`。
  - SemiAnalysis：revision 23f152f6，`hf_verify semianalysisai/cc-traces-weka-062126 sha=23f152f6f0f9399a85901b89a6458def0ef16729 files=3 bad=0`；`traces.jsonl` sha256 `29b6a19e…`。
  - Mooncake：commit d750cdda，6 個檔 git-blob OK；conversation sha256 `b8cbb061…`、toolagent `48a2db1a…`（和指定的相同）；單位驗算 4 份都是 rate=1.000000。

**卡住什麼**：無。

**需要別的 session 做什麼**：無。

**需要使用者決定什麼**：無。
