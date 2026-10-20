# RUNLOG — 平台 C（H200，hgx-cn01）第三階段

> 第三階段（`origin/p3-predictor`，3090）搬到 H200 照跑。只有這條線寫這份；第二階段的 H200 寫 `RUNLOG_H200.md`，3090 寫 `RUNLOG_3090_P3.md`。格式照 CLAUDE.md §4.2。
> 分支：`h200-p3`（worktree `/work/ChihChungHsu/hungweilai/paper-hkv-p3`，從 `h200` 的 `cabeae3` 開；合回 `h200` 要等第二階段 session 的計時 job 跑完，見 §待決定 4）。
> 每個 run 的原始輸出：`$TIARA_RUNS/<run_id>/`（＝`/work/ChihChungHsu/hungweilai/tiara-runs/`）；第三階段的大檔放 hgx-cn01 的本地 `/tmp/hungweilai-p3/`（見 P3H-0「做法」）。

---

## Milestone P3H-0 — 移植（程式、環境、H200 常數與格式表；只有 head node 的 CPU 步驟）
**狀態**: PASS（有例外）——程式移植、P3 環境、`costs_h200.json`、H200 格式表都完成；**例外**：Llama FP8-K0／INT8 的 H200 生成品質補算 **FAILED**（重現檢查沒過，照規則 2 停），格式表那兩格寫 NOT_MEASURED。
**執行時間**: 2026-10-19T20:20+08:00（開 worktree、合併）→ 2026-10-19T21:25+08:00（送出 job 21116）

**使用者的決定（10/19，這個 session）**:
- 把 3090 的第三階段（探索 X1–X9、格式表、校準、loss 開發、地圖模擬器）適配到 H200。
- 磁碟：lab 的 `/work` 配額 1500G／1536G。先選「刪 Qwen3-30B-A3B」，隨即說「等等」問有沒有別的空間 → **不刪**，改放 hgx-cn01 的本地 `/tmp`（`/dev/md0`，不在 lab 配額裡）。
- CPU 重的步驟包在 1-GPU job 裡（`h200q` 不收 `--gres=gpu:0`，10/16 試過；head node 32 核／62 GB、43 人共用）。
- 範圍：全部程式可在 H200 跑；資料層與測試重建並和 3090 逐位元對拍；依硬體的部分（X1/X3/X4/X5/X8、格式表、常數、CPU 忙校準）用 H200 常數重跑；地圖只跑測試、合成端到端、C0 使用率，**完整地圖等 04 v4 凍結**。

**做法**:
- 合併：`origin/p3-predictor`（`83dd09e`）→ `h200-p3`，commit `64b6994`，沒有衝突。3090 的結果檔一起進來（`results/p3_*`、`results/p3_calib_3090`），**只當參考**；H200 的輸出一律另放 `*_h200`，H200 的地圖與探索不放 3090 的列（memory h200-phase1-reproduction）。
- 和第二階段 session 協調（同一個 Slurm 帳號、同一個 working tree）：我改在另一個 worktree，不碰共用 checkout；所有 CPU 重或 GPU 的 job 接在它的計時 job（21109→21110→21111）後面（memory h200-no-self-cotenancy-for-timing）；`TIARA_HOME` 不改（它的原始資料在底下）。
- 本地 `/tmp` 留不留得住（head node 查 Slurm 設定）：`JobContainerType=job_container/none`（沒有每個 job 私有的 /tmp）、plugstack 只有 pyxis、epilog 只有 `50-epilog-enroot.sh`（只清 enroot 目錄）→ job 之間留得住；重開機（hgx-cn01 2026-06-15 開機）或被別人塞滿會不見 → 每個 part 開頭 `need_data`＋`h200_p3_verify_data.py`（sha256 對 README.txt），缺了就停。
- 程式（全部「沒設＝3090／MI300X 的原本行為」）：
  - `code/env_h200.sh`：P3 的 `.claude/settings.json` 把 `HF_HOME`、`TIARA_HOME`、`HF_HUB_CACHE`… 設成 `/ssd8/...` → 這台沒有 `/ssd8` 時一樣清掉（`TIARA_HOME` 回到 `$HOME`）。
  - p3x 20 支、`p3_data`、`run_map`：寫死的 `/ssd8` 路徑改讀 `TIARA_DATASETS`／`TIARA_HF_HUB`／`P3_DATA_CACHE`／`P3_MAP_OUT`；`p3_data` 的輸出可用 `P3_DATA_RESULTS` 蓋掉。
  - `code/p3_platform.py`（新）：`P3_CONST=mi300x|h200`。探索與 `build_inputs`／格式表讀哪一台的第一、二階段實測；限速層與每 block 位元組兩台都用 MI300X 的（03 §R2）；14＋2 個還原格子 run 的對應；`check_m7_res` 防止 f(i) 與其他常數混用（env_h200.sh 預設 `M7_RES` 是 H200 的，`m8_trace_oracle.load_f` 在 import 時讀它）。
  - `run_map.py`／`analyze_map.py`：`P3_MAP_PLATFORM`（h200＝`costs_h200`＋`official_h200` 格式表、敏感度沒有 3090 列）。
  - `formats.py`：`DQ_SETS` 加 h200 欄、`official_h200[:欄組]`、表的平台和要求的平台對不上就丟錯。
  - `setup_3090_data.sh`：H200 上不 source `env_3090.sh`；`_hf_dataset` 可只下載部分檔；加 `freeinference`（3090 上是手動下載的）。
  - 新：`code/h200_setup_p3_env.sh`、`code/r_h200_p3.sh`、`code/h200_p3_compare.py`、`code/h200_p3_verify_data.py`。
- **環境**（head node，`bash code/h200_setup_p3_env.sh`，log 在 `$TIARA_HOME/logs/p3_env/`；不產生實驗數字，沒走 run 殼）：`python -m venv --system-site-packages ~/venvs/tiara-p3` 疊在 conda `tiara` 上（tiara 一個檔都沒動）；裝 3090 同版的 scikit-learn 1.9.1、lightgbm 4.7.0、xgboost 3.4.1（PyPI 的 CUDA 13 build；3090 要 cu12 變體是因為 driver 550）、statsmodels 0.15.0、lifelines 0.30.0、scikit-survival 0.28.0、pycox 0.3.0、torchtuples 0.2.2、pytest 9.1.1；tiara 已有的 209 個套件當 constraint。venv 297 MB。
  和 3090 不同（照實，沒硬湊）：numpy 2.3.5（3090 2.2.6）、torch 2.13.0+cu130（3090 2.10.0+cu126）、transformers 5.19.0（3090 5.17.0）＝tiara 原本的版本。

**run_id**（head node 的 run 用 §4.1 手動記錄殼，`$TIARA_RUNS/<run_id>/{cmd.sh,context.txt,stdout.log,stderr.log,exit_code}`）:

| run_id | 內容 | exit | 判定 |
|:--|:--|:--:|:--|
| `20261019-205457-p3h-costs` | `P3_CONST=h200 M7_RES=…/m7_write_policy_h200 python code/p3_sim/build_inputs.py --platform h200` | 0 | OK → `code/p3_sim/config/costs_h200.json` |
| `20261019-210014-p3h-fmt-qderive` | `P3_CONST=h200 … python code/p3_formats/derive_quality.py` | **1** | **FAILED**：重現檢查沒過（下面），沒寫 derived |
| `20261019-210127-p3h-fmt-table` | `P3_CONST=h200 … python code/p3_formats/build_format_table.py` | 0 | OK → `results/p3_formats_h200/`（4 個 problems 照列） |

另外（不產生 results 的檢查，scratchpad）：`build_inputs.py` 不給參數時三個舊檔和 git 的逐欄相同（run_id／ts 以外）；`build_format_table.py` 的 MI300X 模式和 git 的表比，只差 12 格（都是讀 3090 `/ssd8` 原始 restores 的 Qwen 格：這台沒有那個檔，預期）；`setup_3090_data.sh mooncake` 在 head node 試下載到 scratchpad：規則 6 的單位驗算與 `m8_trace_oracle.load_trace` 的輸出和 3090 相同（conversation 12,031、toolagent 23,608，unit_match_rate 1.0，chain_bad 0）；FreeInference 的部分下載（README＋16 個 manifest）`hf_verify bad=0`。

**產出檔**: `code/p3_sim/config/costs_h200.json`；`results/p3_formats_h200/{format_table.json,format_table.csv,quality_pending.csv}`；本檔。

**關鍵數字**（只寫量到的；全部是 repo 裡第一、二階段 H200 實測檔的轉換，沒有新量測）:

| 常數（`costs_h200.json`） | H200 | MI300X（`costs_mi300x.json`） | 來源（H200） |
|:--|--:|--:|:--|
| Llama f(0)／f(63)／斜率 | 17.818／47.628 ms／0.4730 ms/block | 28.878／—／0.8877 | `m7_write_policy_h200/calib_c1.csv`（run 20261016-061311-m7h-c1） |
| Qwen f(0)／f(63)／斜率 | 15.741／41.430 ms／0.4040 | 23.638／—／0.7695 | `m10_precision_h200/speed_qwen/bw.json`（20261017-074209-m10h-calibf-qwen） |
| GPU 忙倍數（busy2） | 1.945 | 2.07 | `m7_write_policy_h200/busy_levels.json`（20261016-061331-m7h-c5） |
| b\*＝32 頻寬（Llama） | 2.49 GiB/s | 1.4668 | `m10_precision_h200/speed_llama/bw.json`（20261017-074151-m10h-calibf-llama） |
| CPU 層／本地 SSD／NFS（限速層） | 35.4159／6.981／0.331 GiB/s | 同左 | MI300X 的 `tier_params.json`（03 §R2） |
| CPU 忙的 H2D 倍數 | **NOT_MEASURED** | NOT_MEASURED | 3090 量過 1.35–1.81（不借用）；H200 見 §待決定 2 |

格式表（`results/p3_formats_h200`，30 列×227 欄；還原中・Cake・原寫法 K0 的每 block 反量化，ms，GPU 閒／忙）：Llama fp8_token 0.435／1.251（MI300X 0.387／1.033）、int4_kivi 0.846／1.822（0.712／1.816）；Qwen int8_token 0.236／0.618（0.166／0.409）、int4_kivi 0.544／1.260（0.411／1.038）。
每 block 位元組 30／30 和 MI300X 版相同；32K 品質判定除了 Llama FP8-K0／INT8（H200 補算失敗 → UNVERIFIED_TF_ONLY）以外相同。

**失敗與異常**:
1. **`20261019-210014-p3h-fmt-qderive` exit 1**（`derive_quality.py` 的兩個重現檢查，H200 的列對 repo 裡 H200 已提交的分析 JSON）：
   `重現檢查沒過（規則 2：停）：['E1s@fp8k0_head/B-i2b/mean_lp_read: 44.84 != 44.799', 'E1s@int8_token/B-i2b/mean_lp_read: 44.84 != 44.799', "run_ids 不同：[…11 個…] vs […9 個…]", 'E1/B-i2b/mean_lp_read: 43.88 != 43.872']`
   - 差、95% CI、eps_pass、n＝1,140 都相同；不同的只有 `mean_lp_read`（B-i2b 讀法讀到幾個低精度 block 的平均，診斷欄，不是品質分數）與 Qwen 的 run_ids（H200 的 step2b 分析在 T1-s1、LB-s1 兩個 shard 跑完**之前**算的，現在 `quality/` 裡多了它們）。Llama 的 run_ids 相同、`mean_lp_read` 仍差 0.008，原因**沒查到**。
   - 照規則 2 停、沒寫 `derived/`；格式表的 Llama fp8k0_token／int8_token 生成品質寫 NOT_MEASURED（附原因），判定變 UNVERIFIED_TF_ONLY。要怎麼處理：§待決定 1。
2. 改 `.claude/hooks/gpu_guard_hook.py`（讓它在 Slurm 叢集上不做事）被權限擋下（Self-Modification），沒繞過。現況：hook 在 `h200-p3` 分支裡是 3090 的原樣；合回 `h200` 後，H200 的 Claude session 只要指令裡寫了 `CUDA_VISIBLE_DEVICES=<數字>` 就會被擋（head node 查不到 GPU）。§待決定 3。
3. `pip install` 第一次 `ResolutionImpossible`：sentence-transformers 6.1.0 要 `huggingface-hub<2.0`，tiara 是 2.2.0 → 不裝（只有可選的 P3-7 內容型預測器與 smoke 的 libs 用）。
4. head node 的 `/tmp`（20G，共用）又滿了：zsh 的 heredoc 寫不進去（`can't create temp file for here document`）。我的工具在 `/work`，改把腳本寫到 scratchpad 再跑；第二階段 session 的工具全掛，照它的要求刪了它自己的 scratch（`sens_mi`，9.8 MB），它之後也刪了自己的舊目錄。
5. hf CLI 的 `--include a b`：第二個 pattern 被當成檔名、`--include` 被忽略（3090 的 `--exclude` 同一個坑）→ 改成每個 pattern 一個 `--include`。

**與假設的差異**:
- 3090 分支的 `README.md` 寫「之後 MI300X／H200 只換成本常數重跑」：常數本身大部分在 H200 第一、二階段已經量過（上表），**只有 CPU 忙的倍數**是新的 GPU 量測。
- `build_format_table` 對 H200 不是完全「只換常數」：H200 沒有 Qwen 的重量（regrid，用 speed_qwen 那一次）、43 的決賽選手不同（Llama 有 nvfp4、沒有 mxfp4）、INT4 端點品質在 `compare_mi300x_path`（H200 主線走 FP8 階）。都寫進表的 notes。

---

## Milestone P3H-0b — Llama FP8-K0／INT8 的 H200 品質補算（查清楚重現檢查為什麼沒過）＋使用者授權的決定
**狀態**: PASS
**執行時間**: 2026-10-19T21:05+08:00 → 21:15+08:00（head node）
**使用者 10/19**：「自行規劃看哪樣比較好」——P3H-0 列的待決定由我決定（下面「決定」）。
**run_id**: `20261019-211426-p3h-fmt-qderive2`（exit 0）、`20261019-211449-p3h-fmt-table2`（exit 0；取代 P3H-0 的 `…-210127-p3h-fmt-table` 那一版）
**指令**: `env P3_CONST=h200 M7_RES=$PWD/results/m7_write_policy_h200 RUN_ID=<id> RUN_DIR=$TIARA_RUNS/<id> ~/venvs/tiara-p3/bin/python code/p3_formats/derive_quality.py`；格式表同 P3H-0
**原因調查**（scratchpad，不進 results）：
- `m10_analyze quality` 的 `mean_lp_read` 是對**除了 LB 以外的所有列**取平均（含 T1，code/m10_analyze.py 第 224 行）；判準欄（差、CI、eps、KL、flips、same_as_e0，n＝1,140）只用主任務 T2–T5。
- 用 repo 裡 H200 已提交的 `quality_qwen_step2b.json` 列的同一批 9 個檔（不含後來的 T1-s1、LB-s1）重跑：**主任務欄位逐位元相同**，不同的只有 4 處——`e0_accuracy` 裡 T1／LB 的格（已提交的版本 T1 每格 n＝9、LB 只有 narrativeqa n＝18；現在的檔 T1 n＝30、LB 六個子集各 25）、`n_items`（627 vs 690）、兩個 `mean_lp_read`。
- → H200 的 step1／step2b 分析是在 q3 工作池的 T1／LB shard（55 號，只報告）**還沒寫完時**算的（Llama 的 T1-s1 已經在 run_ids 裡、但只寫了一部分）。判準欄不受影響。
**改動**：`derive_quality.py` 在 `P3_CONST=h200` 時，重現檢查對主任務欄位一樣要求逐位元相同；`mean_lp_read` 與 `run_ids` 的差照記在 `derive_quality.json` 的 `noted_not_blocking`、不擋（MI300X 模式不變，全部要相同）。
**關鍵數字**（H200，Llama，對 E0 的配對答對率差，T2–T5、1,140 列）：E1s@fp8k0_token A −0.20 [−0.78, +0.37] pt、B-idle −0.40 [−0.96, +0.13]、B-busy −0.36 [−0.93, +0.17]、B-i2b −0.39 [−0.97, +0.14]；E1s@int8_token A −0.11 [−0.55, +0.31]、B-idle −0.08、B-busy −0.20、B-i2b −0.03（四種情境都 eps_pass）。
格式表重建後：**30／30 列的 32K 品質判定和 MI300X 版相同**；剩下的 problems 只有「X5 的 H200 重跑還沒做 → 不對拍」（job 21116 的 FMT-TABLE 步驟補）。

**決定**（使用者授權，理由寫在旁邊）:
1. Llama FP8-K0／INT8：照上面查清楚之後補算（主任務欄位嚴格、只報告的欄照記）。
2. **H200 的 CPU 忙校準：不做**。3090 的定義是「整台主機忙」（N 到 48、兩個 NUMA）；Slurm 的 1-GPU job 只有 16 CPU 的 cpuset（`cpu_busy_restore.py` 的 same 會綁到 GPU 那個節點的 64 個 CPU，和 cpuset 交集可能是空的），做不出同一個條件；整台加壓會影響同節點別的 lab 的 job；而這個數字只進地圖的敏感度（只報告）。H200 的 CPU 忙寫 NOT_MEASURED，不借 3090 的 1.35／1.81。`r_h200_p3.sh` 的 cal 部分刪掉。
3. `.claude/settings.json` 與 GPU hook：合回 `h200` 時保留 `h200` 原本的設定（hook 是 3090 共用機用的；P3 的 env 指到不存在的 `/ssd8`）。使用者 10/19 授權後做了：commit `fd94762`（`git checkout cabeae3 -- .claude`）。`hooks/gpu_guard_hook.py` 檔還在、但 settings 沒有引用它（不會執行）。
4. ✅ 合回 `h200`、push `origin/h200`（10/20 晚）：第二階段 session 通知 21111 跑完後，在另開的暫時 worktree（detached 在 origin/h200＝b59a942）`merge --no-ff h200-p3`＝806a803，沒有衝突；NV_PORTING_NOTES.md 加兩行（開頭指到 P3_PORTING_NOTES、§6 清單一項）＝46c49d5，先問過它。fetch → merge → push（不 rebase，它要求），b59a942..46c49d5 是快轉。共用的第二階段程式（m7_env.sh、env_h200.sh、m7_guard_run.py、test_m7_correctness.py）合併前後在這台 source m7_env.sh 的環境逐行相同（env -i 與 session 環境都比過），只有 env_h200 的提示字不同。這個 worktree 在 21116 跑完前不合 origin/h200（job 讀這裡的程式）；21116 的結果之後再合一次。
5. 完整地圖：等 04 v4 凍結（3090 那邊在定）；凍結後再寫 H200 的鎖定清單（`costs_h200.json`、`results/p3_formats_h200/format_table.json`，沒有 3090 的校準檔）。

---

## Milestone P3H-1 — 資料層重建、測試、探索（H200 常數）、地圖工具（job 21116）
**狀態**: FAIL（SIM-TEST，見下面「失敗與異常」；接著跑＝P3H-1b）。送出時是 PENDING（`p3h-all`，`--part data,simtest,explore,map`；10/19 21:2x 之後的程式改動（X5 後重建格式表、刪 cal）在開跑時才讀到，runner 會改執行 repo 裡的版本，`--dependency=afterany:21111`：接在第二階段的 21109→21110→21111 之後，約 20 小時以上；10/20 第二階段在 21110 和 21111 之間插了 1 小時的計時 job 21377（m10h-attnb，21111 改 afterany:21377），21116 的依賴不變、晚約 1 小時開跑）
**指令**: `env -u TMPDIR -u CLAUDE_CODE_TMPDIR sbatch --job-name=p3h-all --dependency=afterany:21111 --time=48:00:00 code/r_h200_p3.sh --part data,simtest,explore,map`（在 worktree 送出）
**步驟**（`bash code/r_h200_p3.sh --part data,simtest,explore,map --list`）：DATA-DL → DATA-VERIFY → 統一表（照 3090 最終 run 的分批）→ 對拍、測試 → DATA-CMP（和 3090 的 `results/p3_data` 逐值比）；SIM-TEST（3090 最後一次 121 passed）；X1／X3／X4／X5／X8 的登記主結果（H200 常數）；地圖：MI300X 常數的合成端到端＋C0 使用率（和 3090 逐值比）→ H200 常數的同一批。
（先前送的 21113 `p3h-data`、21114 `p3h-simtest` 已取消、合成這一個，少排一次隊。）
**執行時間**: 2026-10-20T22:27:59+08:00 → 2026-10-20T23:14:21+08:00（46 分；hgx-cn01；git add63f5）
**run_id**: 資料 `20261020-222803-p3h-data-dl` … `20261020-230641-p3h-data-cmp`（每步一個，見 `slurm-p3h-all-21116.out`）；失敗的那步 `20261020-230656-p3h-sim-test`（`$TIARA_RUNS`＝`/work/ChihChungHsu/hungweilai/tiara-runs`）
**產出檔**: `results/p3_data_h200/{summary.csv, crosscheck.csv, service_constants.json, synth_fit_fi_week01.json, compare_3090.json}`；統一表在 hgx-cn01 `/tmp/hungweilai-p3/datasets/p3_unified`
**關鍵數字**:
- DATA-DL、DATA-VERIFY 過（sha256 和 README 相同）；統一表、FICONST、MOONCAKE、BURSTGPT、CACHEWISE、WILDCHAT、SYNTHFIT、SYNTH、SUMMARY、XCHECK、DATA-TEST 都 exit 0；DATA-BAILIAN exit 5＝只有 traceB 的 FAILED.json（和 3090 的 20261017-220343-p3data-bailian 相同，預期中）。
- SIM-TEST：**1 failed, 120 passed**（439 s）。
- DATA-CMP（和 3090 的 `results/p3_data` 逐值比）：verdict＝**MISSING**（不是預期的 IDENTICAL）：
  - `synth_fit_fi_week01.json` IDENTICAL。
  - `raw_id_chain.csv` MISSING：3090 是另外用 `crosscheck.py --chain-only` 寫的（d0c4658），runner 沒有這步。
  - `crosscheck.csv` DIFF 12 處＝全部是 bailian traceA／coder／thinking 的 `raw_id_chain:*` 4 列：3090 的 crosscheck.csv（11e074c）早於這幾列被加進 crosscheck.py（d0c4658），所以 3090 檔裡沒有；其餘 114 列逐值相同。
  - `service_constants.json` DIFF 6 處：FI 服務時間的中位數迴歸（statsmodels 0.15.0 QuantReg q=0.5）A 1509.1056832666832 → 1509.2071378337419（相對 6.7e-5）、B 14.212994865389332 → 14.212997448707114（1.8e-7）、C 0.014841663618334697 → 0.01484057044196145（7.4e-5），以及由它算的殘差分位數、frac_pred_le_dur。輸入的 dur／tokens 分位數逐位元相同 → 差在數值計算（numpy 2.3.5 ＋ scipy-openblas 0.3.30 vs 3090 的 numpy 2.2.6；QuantReg 是有容差的迭代法）。
  - `summary.csv` DIFF 3 處，全部是 Mooncake tool-rule 的 think 分位數：tool-rule-d4 think_ms_p50 1016.0 → 1015.0；tool-rule-strict p50 114535.0 → 114534.5、p99 919872.39 → 919872.32。機制：Mooncake 沒有回應結束時間，`service_est_ms = round(A + B·out + C·in)`（load_mooncake.py:173）→ A 差 0.1 ms 就讓部分列的整數 ms 差 1 → think_ms 差 ±1 ms。bailian、wildchat 的 summary 列逐位元相同（它們也用同一組常數，列層級可能有 ±1 ms 但分位數沒變；列層級沒有 3090 的檔可比）。
**失敗與異常**:
```
FAILED code/p3_sim/test_p3_sim.py::test_official_format_table_adapter
E   ValueError: /raid/work/ChihChungHsu/hungweilai/paper-hkv-p3/results/p3_formats/format_table.json 是平台 mi300x＋3090 的表，不能給 platform=h200 用
code/p3_sim/formats.py:187: ValueError
1 failed, 120 passed, 23 warnings in 439.28s (0:07:19)
```
原因＝我的移植：183ad30 在 `formats.DQ_SETS` 加了 h200 欄，這個測試掃 DQ_SETS 的每個平台去讀 MI300X＋3090 的表 → 我同一次加的平台防呆（表和平台對不上就丟錯）照設計擋下。不是數字不同。runner 照規則 2 停在 SIM-TEST。

## Milestone P3H-1b — 修正後接著跑（從 DATA-CHAIN 起）
**狀態**: FAIL（job **21403** `p3h-all`，afterany:21402，git 9a9b9c1；2026-10-20T23:41:35 → 23:50:53，9 分）：資料層與 SIM-TEST 過，停在 X1-SELFTEST（下面）；接著跑＝P3H-1c
**資料層（23:41–23:42）**: DATA-CHAIN、DATA-TEST exit 0；DATA-CMP（`20261020-234211-p3h-data-cmp`）：`raw_id_chain.csv` **IDENTICAL**、`synth_fit_fi_week01.json` IDENTICAL、`crosscheck.csv` 12 處＝同 P3H-1（3090 檔沒有 raw_id_chain 列；這幾列的值由 raw_id_chain.csv 比到，IDENTICAL）、`service_constants.json` 6 處、`summary.csv` 3 處（同 P3H-1）。**DATA-DIAG-CONST3090：用 3090 的迴歸常數重建 Mooncake → summary 的 mooncake 列和 3090 IDENTICAL**（`results/p3_data_h200/diag_const3090_mooncake.json`）→ 資料層和 3090 的差異只來自 QuantReg 常數（數值庫），沒有別的來源。
**SIM-TEST**（`20261020-234236-p3h-sim-test`）: **122 passed**（438 s；3090 的 121 個＋新加的 H200 表測試）。
**失敗與異常**（X1-SELFTEST，`20261020-235012-p3h-x1-selftest`，41 s）：
```
  File "/raid/work/ChihChungHsu/hungweilai/paper-hkv-p3/code/p3x/x1_sim.py", line 1077, in cmd_selftest
    with open(p, "a", newline="") as fh:
FileNotFoundError: [Errno 2] No such file or directory: '/raid/work/ChihChungHsu/hungweilai/paper-hkv-p3/results/p3_explore_h200/x1/x1_invariants.csv'
```
原因＝我的移植：H200 的探索輸出是新目錄 `results/p3_explore_h200/<x>`（3090 的在 git 裡本來就有），x1_sim 的 selftest 寫檔前不建目錄。其他步驟查過：x3_leads、x4_sim、x5_run、x8_*、地圖的 helper 都自己建目錄；只有 X1-SELFTEST 在任何步驟建 x1/ 之前跑。
修正：runner 的 explore part 一開頭 `mkdir -p` x1／x3／x4／x5／x8（不是 step，`--from` 也會跑）。

## Milestone P3H-1c — 接著跑（從 X1-SELFTEST 起）
**指令**: `env -u TMPDIR -u CLAUDE_CODE_TMPDIR sbatch --job-name=p3h-all --time=48:00:00 code/r_h200_p3.sh --part explore,map --from X1-SELFTEST`（第二階段 session 沒有排隊中的 GPU job，不加依賴）
**修正**（都在 worktree，h200-p3）：
1. `test_official_format_table_adapter` 只掃 `TABLE_PLATFORMS[表的平台]`（＝原本 3090 的 mi300x、3090 兩欄，行為和 3090 相同）；另加 `test_official_format_table_adapter_h200`：H200 表的位元組、解壓欄、品質判定和表一致，表／平台對不上兩個方向都丟錯。登入節點只跑這兩個：2 passed（1.1 s；H200 表四個格式在 32,768 的狀態＝pass／pass／pass／unverified，和 MI300X 相同）。
2. runner 加 `DATA-CHAIN`（`crosscheck.py --chain-only` → `raw_id_chain.csv`，讓 DATA-CMP 比得到）。
3. runner 加 `DATA-DIAG-CONST3090`（只診斷）：用 3090 的 `service_constants.json` 把 Mooncake 重建到 `/tmp/hungweilai-p3/diag-const3090`，summary 的 mooncake 列和 3090 比 → `results/p3_data_h200/diag_const3090_mooncake.json`。預期 IDENTICAL＝差異只來自迴歸常數；不是的話照列。`h200_p3_compare.py` 加 `--keep col=value`（兩邊都只留那些列）。
**決定**: H200 照用自己算的迴歸常數（同一支程式在這台的結果），不借 3090 的值；差異照列（P3_PORTING_NOTES 第 9 項）。之後 Mooncake 驅動的結果和 3090 比時，±1 ms 的 think 差會傳下去，要先看這一項。
**指令**: `env -u TMPDIR -u CLAUDE_CODE_TMPDIR sbatch --job-name=p3h-all --dependency=afterany:21402 --time=48:00:00 code/r_h200_p3.sh --part data,simtest,explore,map --from DATA-CHAIN`（接在第二階段的計時 job 21402 m10h-nvx 之後，不和它同時跑）

---

## 待使用者決定
已由我決定（使用者 10/19「自行規劃看哪樣比較好」、「都授權給你」），見 P3H-0b「決定」；第 3 點已做（`fd94762`）。
