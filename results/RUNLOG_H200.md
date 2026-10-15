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

**狀態**: **STOPPED（R2-C6 失敗）**。2026-10-15 21:25:56 送出；2026-10-16 06:10:10 開始（hgx-cn01，git `bad42c7`）；**07:48:33 停在 R2-C6**（sacct：FAILED、exit 1:0、跑了 01:38:23）。R0 PASS、R1 PASS；R2 的 C0–C7-self、R3 的 A0／A1／B 主設定都 exit 0；C6、C6-backend、C4-t2、R-compare 沒跑 → **R3 還沒有判定**。細節見下面「R2-C6 失敗」
**指令**: `sbatch code/r_h200_all.sh`（在 repo 根目錄；R0 沒過就停，M7_DISK 用 R0 的建議）
**判準**: `03_plan_h200.md` §R0–§R3（事先寫死）

### R0 硬體報告 — PASS
**run_id**: `20261016-061012-h200-probe`（exit 0，24 s）
**產出檔**: `results/hw_h200.json`、`docs/h200/REPORT_H200.md`（腳本產生，不手改）；原始明細在 run 目錄的 `probe/`
**判準（03 §R0）**: GPU 是 H200 ✔、matmul 正確 ✔（max|I@Y−Y| = 0.0）、可見性已判定 ✔（`UNOBSERVABLE_HOST`）→ PASS
**關鍵數字**（只記錄；GPU smoke 不可寫進論文）:
- NVIDIA H200、143,771 MiB、compute capability 9.0、driver 610.43.02、PCIe Gen5 ×16（量的當下 Gen5）；job 裡看得到 1 張、節點 8 張
- CPU：AMD EPYC 9554 ×2（2 NUMA）。job 的 cpuset 112–127（16 核）；/dev/shm 755.8 GiB；cgroup 沒有記憶體上限（`memory_max` None，和 `ConstrainRAMSpace=no` 一致）
- 鄰居：量的當下同節點 8 個 job（含自己）；`gpu_guard.host_contention` = `UNOBSERVABLE_HOST`
- compute node 對外網路：huggingface.co、pypi.org、github.com 都是 HTTP/2 200
- `$TIARA_RUNS` 在 compute node 上是本地的 `/dev/md1`（ext4；head node 看到的是 NFS `hgx-cn01:/work`）→ **M7_DISK = md1**（在 /proc/diskstats）。job 結束後回填進 `env_h200.sh`
- GPU smoke（各 10 次，median）：BF16 matmul 788.1 TFLOPS、HBM 裝置內複製 4,173.9 GB/s、pinned H2D 53.55 GiB/s、D2H 50.04 GiB/s（1 GiB）
- 軟體：Ubuntu 24.04.3、kernel 6.8.0-90、torch 2.13.0+cu130、vLLM 0.28.0、transformers 5.19.0、triton 3.7.1、flash_attn 沒裝

**失敗與異常**:
1. **報告有 3 欄寫 NOT_MEASURED，但原始檔裡有值**：報告產生器沒讀懂（報告不手改，值照原始檔記在這裡）
   - 「CUDA（driver 支援的版本）」：driver 610 把 nvidia-smi 的欄名從 `CUDA Version` 改成 `CUDA UMD Version`。原始值 **13.3**（`probe/nvidia-smi.txt`、`probe/nvidia-smi-q.txt`）
   - 「NUMA affinity」「CPU affinity」：`nvidia-smi topo -m` 的表頭帶 ANSI 底線控制碼。原始值 **NUMA 1、CPU 112–127**（`probe/nvidia-smi-topo.txt`），和 job 的 cpuset 112–127 同一個 NUMA 節點
   - 產生器（`code/h200_probe.sh` 的解析）之後再修；修完要重跑 R0 才會更新報告。不影響 R0 判定
2. RAM 那欄 NOT_MEASURED：cgroup 本來就沒有記憶體上限，不是沒量到。報告的寫法之後一起修
3. fio：節點上沒有 fio（沒有 sudo，事先決定不自己裝）→ 儲存那節 NOT_MEASURED。03 §5.1：這節只記錄、不進判定

### R1 正確性 — PASS
**run_id**: `20261016-061036-m7h-verify`（`m10-verify`：通過 11、失敗 0、略過 2〔`--quick`〕）、`…-061208-…corr-llama-8k`、`…-061221-…corr-qwen-8k`、`…-061233-m7h-corr-qwen-32k`、R1-check，全部 exit 0
**產出檔**: `results/m7_write_policy_h200/correctness_{llama_8k,qwen_8k,qwen_32k}.json`
**關鍵數字**: 三個檔的 HF 跑兩次位元相同、重算 KV 逐位元相同、GPU↔pinned 往返相同、argmax 相同 → 03 §R1 的逐位元項目全過。對 HF 的 logit 最大差：Llama 8K 0.125、Qwen 8K 0.078125、Qwen 32K 0.125（MI300X Llama 是 0.09375；03 §R1：照實報告，不要求相等）

### R2、R3 已完成的步驟（job 20417，全部 exit 0；數字等 R-compare 跑完一起寫）
| 步驟 | run_id | 產出檔（`results/m7_write_policy_h200/`） |
|:--|:--|:--|
| R2-C0、C0-duplex、C1 | `20261016-061254-m7h-c0`、`…-061303-m7h-duplex`、`…-061311-m7h-c1` | `calib_c0.csv`、`calib_c0_duplex.csv`、`calib_c1.csv` |
| R2-C5、busy-levels | `…-061331-m7h-c5`、`…-061427-m7h-busy-levels` | `calib_c5.csv`、`busy_levels.json` |
| R2-C4 t1（same ×1、restart ×10） | `…-061430-m7h-c4-same-t1`、`…-061447`～`…-061655-m7h-c4-restart-t1` | `c4_noise.csv` |
| R2-C7-self | `…-061709-m7h-c7self` | `calib_c7_self.csv` |
| R3-A0、A1、B 主設定 | `…-061749-m7h-a0`、`…-062450-m7h-a1`、`…-063929-m7h-b-nfs-gap0` | `a0.csv`、`a1_share.csv`、`b_share.csv`、`b_workload_seed0.json` |

沒有任何 run 目錄有 `CONTAMINATED`。

### R2-C6 失敗 — job 20417 停下（2026-10-16 07:48）
**run_id**: `20261016-074652-m7h-c6-vllm`（exit 1）
**指令**: `python code/m7_c6_vllm.py --out results/m7_write_policy_h200/c6_vllm.csv --reps 6`（vLLM 0.28.0 serve Llama-3.1-8B＋OffloadingConnector；`vllm_cmd.txt`）
**錯誤**（`vllm_server.log`；vLLM 在 warmup 的取樣步驟 JIT 編譯 FlashInfer 的 top-k／top-p 取樣器）:
```
(EngineCore) [topk_topp_sampler.py:62] Using FlashInfer for top-p & top-k sampling.
(EngineCore) ... warmup_kernels → sample_tokens → flashinfer_sample(processed_logits, top_k, top_p)
(EngineCore)   File ".../flashinfer/jit/cpp_ext.py", line 61, in get_cuda_path
(EngineCore) RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
(APIServer) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```
`m7_c6_vllm.py` 的 stderr 最後是 `ProcessLookupError: [Errno 3] No such process`（收尾時 `os.killpg` 一個已經結束的 vLLM，蓋掉了上面的 `RuntimeError: vllm exited`）。
同一個 log 另有 `vllm.third_party.deep_gemm ... failed to import`（也是找不到 CUDA home；只是警告，BF16 不用 deep_gemm）。
c6_vllm.csv 沒有寫出任何列（vLLM 沒起來就停了）。

**連帶**: 20454 `r-h200-rest`、20544 `r-h200-la-f2`、20545 `r-h200-qwen3` 的 afterok 條件不成立 → 07:48:33 被 Slurm 撤回（CANCELLED，沒有開始）。

**原因調查**（head node，不用 GPU；暫存目錄，結果不進 results/）:
- conda 環境裡有 pip 裝的 nvcc（`nvidia-cuda-nvcc` 13.4.92，`site-packages/nvidia/cu13/bin/nvcc`），但不在 PATH、CUDA_HOME 也沒設，所以 FlashInfer 找不到。
- 試過把 `CUDA_HOME` 指過去、只編譯取樣器（`FLASHINFER_CUDA_ARCH_LIST=9.0a`，`gen_sampling_module().build()`）：**編不過**，
  `flashinfer/data/cccl/libcudacxx/include/cuda/std/__cccl/cuda_toolkit.h:41: #error "CUDA compiler and CUDA toolkit headers are incompatible"`。
  環境裡的 CUDA 套件版本不一致：nvcc 13.4、執行期 `nvidia-cuda-runtime` 13.0.96（torch 2.13.0+cu130）、`nvidia-cuda-cccl` 13.3。要讓 JIT 能用得改套件版本，風險比下面的做法大。
- R0（`h200_probe.sh`）和送出前的檢查都沒有在 compute node 上真的啟動過 vLLM，所以沒抓到。

**修正（2026-10-16，還沒送出；等使用者同意）**:
1. `code/env_h200.sh`：`VLLM_USE_FLASHINFER_SAMPLER=0`（vLLM 0.28 的開關；改用 PyTorch 的 top-k／top-p 取樣器）。理由：
   量測用的請求都是 `temperature=0`（`m7_c6_vllm.py`、`m7_vllm_f.py`、`m8_vllm_drop.py`），貪婪取樣不經過 top-k／top-p，只有暖機請求會用到；
   MI300X（ROCm）本來就不走 FlashInfer 取樣器。不設 CUDA_HOME，deep_gemm 等其他路徑維持原狀。
   head node 上確認 vLLM 讀到的值：`vllm.envs.VLLM_USE_FLASHINFER_SAMPLER` = False。
2. `code/m7_c6_vllm.py`：收尾的 `os.killpg` 遇到 `ProcessLookupError` 不再丟錯（只改收尾，量測不變）。
3. `code/r_h200.sh` 的 R2-C6-backend：grep 多抓 `sampl`，把實際用的取樣器記進 `c6_vllm_backend.txt`。
4. `code/env_h200.sh`：回填 `M7_DISK=md1`（R0 的建議，`probe/m7_disk.tsv`）。接著跑的 `r_h200.sh --from` 需要它。
5. Makefile：`make resume FROM=<步驟>`（＝`sbatch --job-name=r-h200-all --time=01:00:00 code/r_h200.sh --from <步驟>`；用同一個 job 名，submit-rest／submit-r1x 才接得上）。
- 其他會起 vLLM 的步驟（r-h200-rest 的 P3 vllmf2、P4 D2；r1x 的 LA／QM vllmf2）都 source `env_h200.sh`，一起套用。
  Qwen3-30B-A3B（MoE）：vLLM 0.28 在 SM90 上未量化的 MoE 優先用 Triton（`fused_moe/oracle/unquantized.py`：FlashInfer 的 MoE 在 Hopper 排到後面），不會走 FlashInfer JIT。
- **還沒在 GPU 上驗證**：接著跑的第一步就是 R2-C6，沒起來就會再停。

---

## Milestone R-rest — 第一階段其餘的 GPU 實驗（`code/r_h200_rest.sh`）

**狀態**: **CANCELLED，要重送**。job 20454（2026-10-15 23:51 `make submit-rest` 送出，`--dependency=afterok:20417 --kill-on-invalid-dep=yes`、`--time=30:00:00`）在 2026-10-16 07:48:33 因 20417 失敗被 Slurm 撤回，沒有開始
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
- vLLM 版本：MI300X 的 `m7_vllm_f.py` 是在 tiara venv 跑的，那是 **vLLM 0.19.1**（`12_round1_report.md` §4.2、`claude_memory/mi300x-env.md`）；H200 只有 0.28.0。D2 在 MI300X 上就是 0.28（tiara-v028），兩邊同版。

**沒有放進來的（原因）**:
- 第 1 輪的 LongAlpaca-7B、Qwen3-30B-A3B，以及 F2 能耗 κ：2026-10-16 使用者同意 → 見下一節 R-r1x。
- F3（反量化、vLLM 原生 FP8 撈針）：會看到低精度的答對率；第二階段的事先登記凍結前不跑（03 開頭）。
- F1：ROCm 專屬。C2、C7-anchor、A2：量的是 MLSteam 的真實儲存，H200 的裝置不同（03 §R2）。
- CPU 就能跑的模擬與分析：之後在 head node 用 H200 的常數另跑。

---

## Milestone R-r1x — 第 1 輪的 LongAlpaca／Qwen3-MoE＋F2 能耗 κ（`code/r_h200_r1x.sh`）

**狀態**: **CANCELLED，要重送**。2026-10-16 06:3x `make submit-r1x` 送出（git `53f22a0`）的 job 20544 `r-h200-la-f2`（afterok:20417）、job 20545 `r-h200-qwen3`（afterok:20417,afterany:20454），07:48:33 因 20417 失敗被撤回，都沒有開始
**注意**: 這兩個 commit（`b1c74cb`、`53f22a0`）是在 20417 執行中做的；20417 之後的 run 的 context.txt 會記成新的 HEAD，但它用到的檔案和 `bad42c7` 相同（這兩個 commit 只改 m9_f2_*、Makefile、新檔、RUNLOG、R0 的產出）
**決定**: 2026-10-16 06:1x 使用者對兩項都說「好」：下載 LongAlpaca-7B 和 Qwen3-30B-A3B（約 71 GiB）、F2 改成 NVML 再跑
**指令**: `make submit-r1x`，送出兩個 job：
- `r-h200-la-f2`＝`sbatch --job-name=r-h200-la-f2 --time=04:00:00 --dependency=afterok:<r-h200-all> --kill-on-invalid-dep=yes code/r_h200_r1x.sh --part la`（和 r-h200-rest 平行，另一張卡）
- `r-h200-qwen3`＝`… --time=05:00:00 --dependency=afterok:<r-h200-all>,afterany:<r-h200-rest> … --part qwen3`
- 每一步的指令：`make dry-run-r1x`（la 12 步、qwen3 5 步）

**範圍**（照 MI300X 的 `m7_run_round1.sh`、`round1b`、`round1d` 與 run_commands.tsv 的 F2 指令，只換路徑）:
- `--part la`：LongAlpaca-7B 的正確性、C1 f(i)（72 chunks × 3）、vLLM f(i)（`--max-chunks 62`）→ `results/m7_explore_h200/`；
  F2：probe → **gate（新）** → cakee → sweep（Llama，5 輪）→ final；LongAlpaca 補充 sweep（3 輪、切點 {8,16,32,48,56}、`--verify 0`）→ final → `results/m9_followup_h200/`
- `--part qwen3`：下載（沒有才下）→ 正確性、C1 grouped_mm、C1 eager、vLLM f(i) → `results/m7_explore_h200/`

**下載**:
- LongAlpaca-7B：head node，run `20261016-061951-h200-dl-longalpaca7b`（exit 0，約 3 分）。revision `14c462dc…`、allow_patterns 同 MI300X 的 `m7_dl_model.py`；snapshot 13G（兩個 safetensors：9,976,587,336＋3,500,305,616 B）。配額 1419G → 1431G／1536G
- **Qwen3-30B-A3B 還沒下載，故意延後**：r-h200-rest 的 D2 tier50 會把 fs 層暫存寫進 run 目錄（/work），MI300X 實測 35.5 GB（`FS_TIER_BYTES`＝35,458,647,792，`D2_vllm_drop.md`），跑完才刪。現在就下載的話，lab 剩約 46G，tier50 寫到最多時只剩約 13G；/work 的配額是 lab 共用的，寫滿時 lab 其他人的 job 也會失敗。所以 `r-h200-qwen3` 等 r-h200-rest 結束（afterany）才開始，第一步自己下載（`code/h200_dl_model.py`：revision `0d7cf239…`；下載後剩不到 15 GiB 就停）

**和 MI300X 的差別**:
- F2 的計數器：amdsmi → NVML（`code/m9_f2_energy.py` 的 `NvSmi`；MI300X 的 `Smi` 不動，由 `gpu_guard.vendor()` 選）。
  能量＝`nvmlDeviceGetTotalEnergyConsumption`（mJ）；瞬時功率＝`NVML_FI_DEV_POWER_INSTANT`（不支援時記 None，不拿別的值頂替）；
  1 s 平均＝`nvmlDeviceGetPowerUsage`；使用率＝`nvmlDeviceGetUtilizationRates`。NVML 的計數器沒有自己的時間戳（MI300X 有）。
  probe 另外記 torch 的 PCI id 和結尾的時脈／降頻原因。
- F2 §2.3 事先寫的兩條（更新週期 ≤ 最短還原的 5%；≥10 s 區段 ΔE/Δt 和功率讀值差 ≤5%）在 MI300X 上是看完 probe 手動判的；
  這裡寫成 `m9_f2_analyze.py gate`，sweep 之前跑，沒過就停。最短還原用 f(i) 和頻寬算的 Cake 時間（MI300X 用同樣算法給 cpu35 107.6 ms，報告寫「約 110 ms」）。
  更新週期太慢時，預先登記要改成「K 次一起量」，sweep 沒有實作 → 停下來回報，不自己加。沒有瞬時功率時用 1 s 平均比，並在輸出標明。
- Cake-E 切點由 cakee 依 H200 的 probe 和 H200 的 f(i) 重算（MI300X：Llama nfs=33、LongAlpaca nfs=60），用新的 `--cake-e-from` 帶進 sweep。
  Llama 的 f(i) 用 R2 的 `$M7_RES/calib_c1.csv`；LongAlpaca 用這個 job 剛量的 C1、chunk 256 MiB（新的 `--chunk-mib`）。
- 回歸測試（head node，不用 GPU）：用 git 裡 MI300X 的 `f2_probe_raw.csv`、`f2_sweep_raw.csv` 重跑 cakee、final：
  `f2_cakee`、`f2_verdict`、`f2_sweep_summary`、`f2_pareto`、`f2_cost`、`f2_kappa_e` 除了 run_id／ts 和 git 裡的完全相同；
  LongAlpaca 的 cakee 也重現 nfs=60。NVML 那段沒有 GPU 試不了：用到的 pynvml 名稱都存在（nvidia-ml-py 13.615.71），第一次真的跑是 F2-probe。
- M7_MODEL_GLOB 寫在每個 run 的 cmd.sh 裡（MI300X 的 cmd.sh 沒記到，當時是從腳本補）。
- vLLM 0.28.0（MI300X 的 `m7_vllm_f.py` 是 0.19.1）；transformers 5.19.0（MI300X 5.17.0）。
- la 和 r-h200-rest 同時跑在同一個節點的不同卡上：兩邊共用主機記憶體、/work；和別的 lab 的 job 一樣是看不到的鄰居（UNOBSERVABLE_HOST），每個 run 的 `cotenants_end.txt` 有記。

**不做的**: MI300X 第一版 vllmf（相減法；程式沒有留下）。CPU 的模擬（r1-sim、`m7_round1_analyze`）之後在 head node 做。

