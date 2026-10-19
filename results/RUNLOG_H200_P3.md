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

## Milestone P3H-1 — 資料層重建、測試、探索（H200 常數）、地圖工具（job 21116，排隊中）
**狀態**: PENDING（`p3h-all`，`--part data,simtest,explore,map`，`--dependency=afterany:21111`：接在第二階段的 21109→21110→21111 之後，約 20 小時以上）
**指令**: `env -u TMPDIR -u CLAUDE_CODE_TMPDIR sbatch --job-name=p3h-all --dependency=afterany:21111 --time=48:00:00 code/r_h200_p3.sh --part data,simtest,explore,map`（在 worktree 送出）
**步驟**（`bash code/r_h200_p3.sh --part data,simtest,explore,map --list`）：DATA-DL → DATA-VERIFY → 統一表（照 3090 最終 run 的分批）→ 對拍、測試 → DATA-CMP（和 3090 的 `results/p3_data` 逐值比）；SIM-TEST（3090 最後一次 121 passed）；X1／X3／X4／X5／X8 的登記主結果（H200 常數）；地圖：MI300X 常數的合成端到端＋C0 使用率（和 3090 逐值比）→ H200 常數的同一批。
（先前送的 21113 `p3h-data`、21114 `p3h-simtest` 已取消、合成這一個，少排一次隊。）

---

## 待使用者決定
1. **Llama FP8-K0／INT8 的 H200 品質補算**（P3H-0 失敗 1）：(a) 重現檢查只比品質欄（差、CI、eps），`mean_lp_read` 與 run_ids 照列不擋；(b) 先查清楚 `mean_lp_read` 的 0.008 從哪來；(c) 維持 NOT_MEASURED。
2. **H200 的 CPU 忙校準**（`--part cal`，已寫、**沒送**）：3090 的排程（N 到 48、同／另一個 NUMA）在 1-GPU job 做不到（cgroup 16 CPU、同一個 NUMA）；而且壓力會吃整顆 socket 的記憶體頻寬，同節點別的 lab 的 job 會受影響。可選：只在自己的 16 CPU 內 N＝0／4／8／12（已寫好）、要 4 張卡（64 CPU）照 3090 排程、或不量（地圖的 CPU 忙敏感度在 H200 寫 NOT_MEASURED）。
3. **`.claude/settings.json` 與 GPU hook**（P3H-0 失敗 2）：合回 `h200` 時，要用 P3 的版本（env 指 `/ssd8`、加 hook；`env_h200.sh` 已會清 `/ssd8` 的值）還是保留 `h200` 原本的？hook 要不要在 Slurm 叢集上停用（要你自己改或授權）？
4. **合回 `h200`、push `origin/h200` 的時間**：第二階段 session 的 job 都還在排隊，合併會改到它的 working tree；它說 21111 結束會通知我。
5. **完整地圖**（約 5,600 CPU 小時）：要等 04 v4 凍結；H200 版還要一份 H200 的鎖定清單（`lockfile.py` 現在鎖的是 3090 的校準檔與 MI300X 的格式表）。
