# RUNLOG — 平台 C（NVIDIA H200 @ 學校 Slurm 叢集）

> 格式照 CLAUDE.md §4.2。原始 log 在 `/work/ChihChungHsu/hungweilai/tiara-runs/<run_id>/`（`$TIARA_RUNS`，不進 git）。沒量到就寫 `NOT_MEASURED`。
> 操作手冊：`docs/H200_SLURM.md`；計劃：`docs/precision_20261013/03_plan_h200.md`。H200 的數字不和 3090、MI300X 混用。

---

## Milestone R-env — 環境建置（head node，無 GPU）

**狀態**: PASS（環境、模型、資料、CPU 自測）；GPU 部分在 job 20417（R0＋R1–R3），見下一節
**執行時間**: 2026-10-15 20:51 → 21:25（+08:00）
**run_id**:
- `20261015-205432-h200-env-install`（**FAIL**，見下）
- `20261015-205636-h200-env-install-retry`（PASS）
- `20261015-211032-h200-setup-data`（模型、資料、checksum PASS；最後的 CPU 自測 **FAIL**，見下）
- `20261015-212040-h200-selftest-cpu`（修正後重跑 `h200_setup_data.sh --no-download`：PASS）

**指令**: 各 run 目錄的 `cmd.sh`（環境：`conda create -n tiara -c conda-forge --override-channels python=3.12`、`conda install pip`、`python -m pip install --no-cache-dir vllm==0.28.0`、`pandas==3.0.5 pyarrow==25.0.1 scipy==1.18.1 matplotlib==3.11.2`；資料：`bash code/h200_setup_data.sh`）
**產出檔**: `~/.conda/envs/tiara`（8.2 GiB）、`~/hf-cache`、`~/datasets/{hotpotqa,longbench}`、`~/pylibs/wonderwords`；pip freeze 在 retry 與 setup-data 的 run 目錄

**關鍵數字**（都是 head node 上的指令輸出）:
- 帳號：`$HOME`＝`/work/ChihChungHsu/hungweilai`（NFS `hgx-cn01:/work`）；lab 配額開始時 1377G／1536G（`quota -gs`），裝完 1415G
- head node 對外網路：huggingface.co、pypi.org、github.com 都回 HTTP 200（`curl -sI`）。compute node：NOT_MEASURED（R0 量）
- 版本：Python 3.12.15（conda-forge）、torch 2.13.0+cu130（CUDA 13.0）、vLLM 0.28.0、transformers 5.19.0（MI300X 的 tiara-v028 是 transformers 5.17.0：差異照錄）
- checksum：HotpotQA parquet、LongBench `data.zip` 都 OK（sha256 與 MI300X 相同）；模型用固定 revision（Llama `4699cc75…`、Qwen `e28526f7…`）
- CPU 自測：`m10_tasks --self-test --seeds 1` 94／0；`m10_longbench --self-test` 9,006／0（與 MI300X 相同）
- 叢集設定（`scontrol show node hgx-cn01`、`/cm/shared/apps/slurm/etc/slurm/{gres,cgroup,slurm}.conf`）：h200q 只有 1 個節點 `hgx-cn01`，`Gres=gpu:h200:8`、128 核（2×64）、RealMemory 1,393,029 MB；`ConstrainDevices=yes`、`ConstrainCores=yes`、**`ConstrainRAMSpace=no`**、`SelectTypeParameters=CR_Core`（記憶體不是排程資源，節點 `AllocMem=0`）；GPU 沒有 MPS／shard。→ GPU 與 16 核獨佔；主記憶體、/dev/shm、PCIe、/work 共用
- 排程：`PriorityType=priority/multifactor`（FairShare 100、QOS 100、Age 30、JobSize 10）、`PriorityDecayHalfLife=7-00:00:00`、`PriorityUsageResetPeriod=MONTHLY`；lab 上限 `gres/gpu=4`、`cpu=64`、`mem=680G`

**失敗與異常**:
1. **第一次裝環境失敗**（`20261015-205432-h200-env-install`，exit 1）：`error: externally-managed-environment`（PEP 668）。
   原因：conda-forge 建出的 env 沒有 pip，`pip` 落到系統的 `/usr/bin/pip`。處置：env 裡 `conda install pip`、一律 `python -m pip`，另開 run 重裝。
2. **CPU 自測找不到模型**（`20261015-211032-h200-setup-data`，exit 1）：
   `FileNotFoundError: /mlsteam/data/tiara/hf-cache/hub/models--unsloth--Llama-3.1-8B-Instruct/snapshots/*`（`m10_tasks.py` line 227, in load_tokenizer）。
   原因：`m10_tasks.MODELS` 寫死 MI300X 路徑；`h200_setup_data.sh` 在 MI300X 上只用 `--no-models` 試過，自測那段沒跑到。
   處置：`MODELS` 先讀 `M7_LLAMA_GLOB`／`M7_QWEN_GLOB`（commit `2955b35`，只改路徑）；`--no-download` 重跑通過。模型、資料、checksum 在失敗前已完成，不受影響。
3. **3090 的設定被帶進來**：repo 的 `.claude/settings.json` 讓 Claude Code 的 shell 帶著 `HF_HOME=/ssd7/...` 等 3090 路徑，sbatch 預設會把它們帶進 job；`module load slurm` 在這個 shell 裡不改 PATH。
   處置：`env_h200.sh` 偵測沒有 `/ssd7` 就清掉並印出、補 Slurm 的 PATH（commit `9778d96`）。
4. **`git push` 失敗**：`fatal: could not read Username for 'https://github.com'`——這台沒有 GitHub 憑證。commit 留在本機，待設定憑證後補 push。
5. **佇列**：送出前 `sbatch --test-only` 預估 2026-10-27 才開始（h200q 8 張全滿、前面有 5 個 2 天的 job；lab 本月用量全叢集最高，FairShare 被壓低）。
   對照 sacct（2026-05 起，h200q，eligible→start）：每月排隊中位數 0–0.1 h、p90 0–5.3 h、最長 67 h。這是排隊，不是量測，不進任何結論。

**與論文假設的差異**: 不適用（沒有量測）

---

## Milestone R0–R3 — 硬體報告＋第一階段復現（job 20417）

**狀態**: PENDING（2026-10-15 21:25:56 送出，`sbatch code/r_h200_all.sh`，priority 61，Reason=Priority）
**指令**: `sbatch code/r_h200_all.sh`（在 repo 根目錄；R0 沒過就停，M7_DISK 用 R0 的建議）
**判準**: `03_plan_h200.md` §R0–§R3（事先寫死）

---

## Milestone R-rest — 第一階段其餘的 GPU 實驗（`code/r_h200_rest.sh`）

**狀態**: 待送出（接在 R0–R3 的 job 後面，afterok；R3 不重現就自動撤回）
**決定**: 2026-10-15 使用者：「先把所有東西都復現，第二階段不動」。MI300X 到 10/23 到期。
**指令**: `make submit-rest`（＝`sbatch --dependency=afterok:<r-h200-all> --kill-on-invalid-dep=yes code/r_h200_rest.sh`）；每一步的指令：`make dry-run-rest`（28 步）
**範圍**（MI300X 上付費跑過、只用 Llama-3.1-8B、R3 沒涵蓋的；指令照 `docs/migration_20261013/inventory/run_commands.tsv`，只換路徑）:
- P1 B 的 8 個敏感度設定（rep36、seed1、cpuvllm、local、fifo、seed2、gap2、ssd50）→ `results/m7_write_policy_h200/b_*.csv`
- P2 08 消融的 GPU 確認（G1 seed 0／1、G2 chat hold）→ `b2_share.csv`
- P3 破解第 1 輪的 Llama 部分（C1 72 chunks、vLLM f(i)、存活設定 seed 0–4）→ `results/m7_explore_h200/`
- P4 D2 真實 vLLM 卸載（9 個 run＋collect）→ `results/m8_directions_h200/`
- MI300X 上這些 run 的量測時間（各 run 結果 CSV 第一列到最後一列的 `ts`，不含載入模型）：P1 約 245 分、P2 約 249 分、P3 的存活設定約 171 分、P4 約 62 分；C1、vLLM f(i) 從 run_id 間隔看各幾分鐘。H200：NOT_MEASURED。`--time=30:00:00`（分區上限 2 天）。

**和 MI300X 指令的差別**:
- cmd.sh 沒記到的環境變數照當時的腳本補：`M7_IO_MODEL`（share／fifo）、`M7_CPU_GIBPS`（3.69＝MI300X 的 C6 實測 × 0.9；11.6）、`M7_B2_OUT`。和 R3 一樣沿用 MI300X 的值，只換 GPU。
- D2 的 9 條指令在 run_commands.tsv 裡被 PATH 截斷：參數由 `results/m8_directions/d2_cells.csv`（cfg、workload、conc、n_sess）與 `D2_vllm_drop.md` §3.2 還原，其餘用 argparse 預設值。`m8_vllm_drop.py` 的模型路徑改讀 `M7_LLAMA_GLOB`（只改路徑）。
- vLLM f(i) 只跑現在的 `m7_vllm_f.py`（＝MI300X 的 vllmf2；第一版 vllmf 的程式沒有留下）。

**沒有放進來的（原因）**:
- 第 1 輪的 LongAlpaca-7B（13G）、Qwen3-30B-A3B（約 61G）：要先下載；lab 的 /work 配額是共用的，現在 1418G／1536G。等使用者決定。
- F2 能耗 κ：`m9_f2_energy.py` 用 amdsmi 的能量計數器，要先改成 NVML 並重做計數器驗證。
- F3（反量化、vLLM 原生 FP8 撈針）：會看到低精度的答對率；第二階段的事先登記凍結前不跑（03 開頭）。
- F1：ROCm 專屬。C2、C7-anchor、A2：量的是 MLSteam 的真實儲存，H200 的裝置不同（03 §R2）。
- CPU 就能跑的模擬與分析：之後在 head node 用 H200 的常數另跑。
