---
name: stage3-h200
description: Runbook for the Tiara stage-3 experiments (docs/stage3_design_20261024/, gates A0/B0/C0/W0/G-M*) on the H200 Slurm node hgx-cn01 - preregistration and deviation protocol, the speed-noise rule (05 §2.1), model weights on compute-node /tmp, GPU scheduling with the ICPE session (排滿 + yield rule), the r_h200_s3.sh runner parts, and pitfalls already hit. Use when continuing, submitting, monitoring or reporting any stage-3 H200 job, or when user says "stage3", "第三階段", "A0", "B0", "跑 stage3 的實驗".
argument-hint: "[gate or task, e.g. 'A0 gen' / 'B0 timing' / 'status']"
allowed-tools: Bash(*), Read, Grep, Glob, Edit, Write, Monitor, SendMessage
---

# Stage-3 on H200：怎麼接手、怎麼跑、怎麼報

> 計畫：`docs/stage3_design_20261024/`（核心頁 README、05_poc_plan_h200.md、06_tracker.md、mechanisms/）。
> **每個閘的判準登記**：`docs/stage3_design_20261024/runs/<閘>.md`（狀態總表＝`runs/README.md`）。
> **紀錄**：`results/RUNLOG_H200_S3.md`；結果：`results/s3_h200/<閘>/`；runner：`code/r_h200_s3.sh`；程式：`code/s3/`。
> 這份 skill 是 2026-10-25 第一個 stage-3 session 留下的；數字一律以 results／RUNLOG 為準，這裡不放結果。

---

## 0. 開工先做

1. `git pull --ff-only`（worktree `paper-hkv-stage3` 在 main；推送一律 `git fetch && (git merge-base --is-ancestor origin/main HEAD || git merge --no-edit origin/main) && git push origin HEAD:main`，**不要 rebase**）。
2. 讀 `results/RUNLOG_H200_S3.md` 最後幾節、`runs/README.md` 的狀態表。
3. Slurm：`export PATH=/cm/local/apps/slurm/current/bin:$PATH`；看 `squeue -A chihchunghsu`（lab 帳號上限 **4 張卡**，和 hao、chihchung 共用）與 `squeue -u $USER`。
4. 有沒有 ICPE 的 session 在線：`ListAgents`（名字 `h200`）；排程上的事要通知它（SendMessage）。

## 1. 不可協商（CLAUDE.md §1，這裡只列 stage-3 最常踩的）

- **判準先登記再看數字**：`runs/<閘>.md` 最上面是登記，commit＋`date -u` 之後不改；之後的修改一律寫成文末的「偏離 N」（日期、改什麼、為什麼、看過哪些數字之前或之後、影響哪些 run）。冒煙（smoke）只驗程式、不進 results、**不看分數**。
- 沒量到寫 `NOT_MEASURED`；失敗照實記、停下回報（runner 的 `step` 會停在失敗那一步，用 `--from <步驟>` 接著跑）。
- 品質實驗（A0、G-M04）不量時間，可以和別人的 job 同時跑；**量時間的 job 獨佔**：節點上不能有我們自己的任何別的 job（含 DL_Lab4、ICPE、stage-3 的品質 job）——runner 會檢查。
- FP8 的 Triton dequant 不修（memory `fp8-code-freeze-until-amd`），FP8 走 m10_codec 的 PyTorch 路徑。

## 2. 速度判準的「雜訊」（05 §2.1；2026-10-25 補）

- σ_noise＝**baseline 那一臂**在同一個 allocation、同一張卡、同一輪隨機順序裡 ≥6 次的相對標準差，每個量測點各自算。
- δ*＝max(5%, 2σ)。「有改進」：中位數差 > δ* **且**差的 95% CI（對 rep 重抽 10,000 次）下界 > 0；「沒有變慢」：變慢 ≤ δ* **且** CI 上界 ≤ δ*。點估計過、CI 沒過 → **INCONCLUSIVE**（加重複次數再判，不算過也不算變慢）。P95 用 P95 的 bootstrap 標準誤。
- 每個量時間的 CSV 列都要有 `n_rep`、`sigma_rel`、`delta_star`（不到 6 次寫 NOT_MEASURED）。**2σ > 5% 的點先回報、不判定。**
- 只套用到還沒看過時間數字的 run；看過的維持原判準，只附「事後以 2σ 重算（只供參考）」。品質判準（配對 bootstrap、ε＝2 pp、≥288 題）不套用。
- **NUMA 位置（只記錄，不進判定；10/25）**：`code/s3/numa_probe.py`。B0、W0 每列多 `gpu_numa`、`cpuset_numa`、`buf_numa`、`numa_local`，CSV 旁邊有 `<名稱>_numa.json`（start／bench／end）；`b0.py env` 有 `numa`；G-fio 記在 `_fs.json`；G-M12 速度只記 sidecar；G-M06 每列 `src_numa`＋sidecar。`numa_local≠yes` 的 run 先照實回報，要不要重量由使用者決定（還沒定規則）。**cpuset 對了不代表 pinned 在同一邊**：本地節點沒空間、`zone_reclaim_mode=0` 時頁會配到另一邊（lab-3090 實例見 RUNLOG S3-NUMA）。

## 3. 模型與大檔：全部在 hgx-cn01 本地 /tmp

- `/work` 的 hf-cache 自 10/23 起是空的；群組配額很緊。權重在 `/tmp/hungweilai-models/hf-cache`（HF_HOME 版面），由 `code/s3/fetch_models.py` 管：`fetch`（固定 revision、逐檔對 HF 的 sha256／git blob sha1）、`check`（大小＋inode）、`refresh`（touch）、`env`（印 export）。**只能在 job 裡用**（head node 看不到 compute node 的 /tmp）。
- **/tmp 是 `noatime` 掛載**，systemd-tmpfiles 30 天清理只看 mtime／ctime → 每個 job 開跑時 `ensure_models` 會 refresh；校準基底（`/tmp/hungweilai-s3/xstack/calib_xl_<model>.pt`）也會 touch。被清掉就自動重抓／重算。
- 新增模型：在 `fetch_models.py` 的 `MODELS` 加固定 revision（從 HF API 讀），先 `fetch`。

## 4. GPU 排程：和 ICPE 共用（使用者 10/25「排滿：stage3 品質先填空檔」）

- stage-3 的品質 job 切成 ≤4 小時、可續跑的分片（`--shard i/n`、`--budget-min`），一有空卡就跑。
- **送出新 job 時**：`for t in $(python3 code/s3/sched.py next-icpe); do python3 code/s3/sched.py add-dep --to $t --job <新 job>; done`（保留原本的依賴；ICPE session 也會改同一份清單，留著它的）。目標已經準備好（只剩在等 s3 的 job 或什麼都不等）時 `add-dep` 會自己跳過（新 job 開跑會讓位；加了只會讓 ICPE 白等）。**改 ICPE job 的依賴（含移除我們加的）會被權限擋下**，要使用者同意。
- **讓位規則**（`code/s3/sched.py yield-target`，runner 開跑時自動做）：ICPE 量時間的 job 只剩在等 s3-* 的 job（或什麼都不等），或 hold 中但前一個 ICPE 量時間的 job 剛結束 ≤60 分鐘 → 這個 stage-3 job 原樣重送一份（`--dependency=afterany:<ICPE job>`）然後 exit 0。重送出來的 job **不要**再加進 ICPE 的依賴（會死結）。
- **hold 中的 ICPE job 一律不動**（只有 ICPE session 能 release）。DL_Lab4 的 job 算「我們自己的」。
- stage-3 量時間的 job 排在 ICPE 量時間的鏈之後；送出前 `squeue -u $USER -w hgx-cn01` 確認。
- sbatch 一律 `env -u TMPDIR -u CLAUDE_CODE_TMPDIR sbatch ...`（head node 的 /tmp 會滿；不要把 TMPDIR 帶進 job）。

## 5. runner：`code/r_h200_s3.sh --part ...`

| part | 量時間 | 做什麼 |
|:--|:--|:--|
| `fetch` | 否 | 三個模型下載＋逐檔驗證＋manifest（`results/s3_h200/env/`） |
| `a0-smoke --cell C` | 否 | 校準（沒有才算）→ gpu-selftest → 2 題 TF 冒煙（輸出在 run 目錄） |
| `a0-tf --cell C --shard i/n` | 否 | A0 TF 篩選（含 G-M04 `--gm04`）→ `results/s3_h200/a0/raw/` |
| `a0-gen --cell C --shard i/n` | 否 | A0 生成（要先 `a0.py select-gen --cell C`；同長度另一格的 TF 也要做完，G4） |
| `b0-prep` | 否 | B0 冒煙：P0e、背景自檢、少量格子、MPS 冒煙 |
| `b0-run --model M --setup b0\|m08 --modes ...` | **是** | B0／G-M08；modes 含 mps 會自動開 MPS daemon |
| `w0gfio` | **是** | W0（兩個模型）＋G-fio |

- 判定格（A0）：`L128`、`Q128`、`Q256`、`Q14-256`（判定）、`L32`、`Q32`（對照）。CPU 端：`python code/s3/a0.py plan|pending|select-gen|report|--self-test`。
- CPU 的閘直接在 head node 跑：`c0.py cells|eval|ref2sigma`、`gm13.py run|ref2sigma`、`w0.py report`、`b0.py report`。numpy 只在 conda 環境裡：先 `source code/env_h200.sh`。
- 監看：Monitor 的 until-loop 或 `sacct` 狀態變化；不要 sleep 輪詢。

## 6. 已經踩過的坑（別再踩）

- **粗底分段編碼不是逐位元相同**：cuBLAS 依列數挑 GEMM 演算法，fp32 捨入不同 → A0 固定 `--seg-blocks 16`，自檢只要求位元組相同、重建誤差差 ≤1%（A0 偏離 1）。
- **ORACLE 的全域 attention 加總會被「近期」的 head 主導**：A0 加了只報告的 ORACLE-NEEDLE（A0 偏離 2）；判準不改。
- **b0 的 64 塊 session**：動作要作用在 block 0..63，不是 1..64（RoPE cos/sin 越界）。
- **heredoc 一定要加引號**（`<<'EOF'`），不然 Markdown 裡的反引號會被 shell 執行掉（10/25 B0／W0 的偏離段被吃過一次）。
- 既有還原資料：`contaminated_gpu=True` 的列、`results/icpe_h200/p01/grid/EXCLUDED_RUNS.csv` 的 run 一律排除；G5／G5b 目錄的 `cake` 列是規則自己的選擇，不是 Cake。
- 微基準（parts）的忙時 H2D 是單塊的 CUDA event 時間，**量不到時間片共用下的等待**（C0-v1 失敗的原因）。
- 不要原地改正在跑的 runner：寫 `.new` 再 `mv`（bash 邊跑邊讀）。
- 改 memory 前先讀（ICPE session 也會寫同一個 memory 目錄）。

## 7. 回報與 commit

- 每個閘：結果附在 `runs/<閘>.md` 的「結果」節＋`results/RUNLOG_H200_S3.md` 一節（狀態、run_id、指令、產出檔、關鍵數字〔實測〕、失敗與異常）。
- commit 前綴：結果 `results(s3-<閘>):`、登記與文件 `docs(s3-<閘>):`／`docs(s3):`、工具 `infra(s3):`、修 bug `fix(s3-<閘>):`、skill `skills:`；結尾照 attribution 規則加 `Co-Authored-By`。
- 11/17 的決策表：05 §1.1（A0／B0）與 mechanisms/06 §4（A0／G-M08／G-M06 → P1–P4）。
