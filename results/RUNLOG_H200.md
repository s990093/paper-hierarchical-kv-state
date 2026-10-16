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

**狀態**: **PASS — R3 判定 REPRODUCED**（(0)(1)(2) 都過；(3) 9/11 同方向，只記錄）。第一次 job 20417（2026-10-16 06:10:10 開始，hgx-cn01，git `bad42c7`）在 07:48:33 停在 R2-C6（見下面「R2-C6 失敗」）；修正後 job 20553 從 R2-C6 接著跑（08:56:30→09:00:20，COMPLETED，git `c4daf94`）
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

**修正（2026-10-16，使用者 08:5x 同意「四個一起送」）**:
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
- GPU 上驗證：job 20553 的 R2-C6（run `20261016-085632-m7h-c6-vllm`）exit 0。`vllm_server.log`：`FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0.`、`Using FLASH_ATTN attention backend`、`Using FlashAttention version 3`（`c6_vllm_backend.txt`）。
  同一個 log 也記到 `Default vLLM sampling parameters have been overridden by the model's generation_config.json: {'temperature': 0.6, 'top_p': 0.9}`：暖機請求沒給 temperature，所以用這組；量測請求都明確給 `temperature: 0`。

### R2-C6 起接著跑 — job 20553（`make resume FROM=R2-C6`）
**執行時間**: 2026-10-16 08:56:30 → 09:00:20（COMPLETED，exit 0；送出時 Slurm 預估 17:56 才開始，實際上 lab 空出一張卡就開始了）
**run_id**: `20261016-085632-m7h-c6-vllm`、`…-085838-m7h-c6-backend`、`…-085841-m7h-c4-same-t2`、`…-085857`～`…-085955-m7h-c4-restart-t2`（5 個）、`…-090009-m7h-repro-compare`
**產出檔**: `results/m7_write_policy_h200/` 的 `c6_vllm.csv`、`c6_vllm_backend.txt`、`c4_noise.csv`（加上 t2）、`repro_verdict.json`、`repro_vs_mi300x.csv`、`summary_*.csv`、`verdict.json`

### R3 判定 — REPRODUCED
**run_id**: `20261016-090009-m7h-repro-compare`（job 20553 的 R-compare；`repro_verdict.json`、`repro_vs_mi300x.csv`）
**判準**: `m7_repro_compare.py` 檔頭＝03 §R3（開跑前寫死）。δ* ＝ max(5%, H200 的 C4 δ)
**關鍵數字**（兩台都用 `m7_analyze.py` 同一套算法；**只比方向**，不進精度這條線的任何圖或判定）:
- C4 雜訊 δ：H200 **1.76%**（t1＋t2）、MI300X 6.17% → δ* = **5.0%**
- (0) kv_bad：A0＋A1＋B 合計 **0** → PASS
- (1) 讀取時 Cake 有用（A0）：28 格裡 **25 格**比「只算」「只載」較快者再快超過 δ*，最大 **93.3%**；最長 L 時有效的頻寬組和 MI300X 相同（100/56/32/25/7 Gbps、local、nfs）→ REPRODUCED
- (2) 寫入時依位置決定沒有淨價值（B 主設定，S5 對 S4+，正＝S5 較快）：cpu_frac 0.25／0.5／1.0 = **−0.07%／−0.24%／+0.03%**（各 3 rep），都在 ±5% 內 → REPRODUCED。
  MI300X 同一格（rep 0–5 合併）：+0.07%／+0.26%／−0.30%（來自下面的 gzfix 重跑）
- (3) GPU 忙時沒有備份的代價（A1 Q2，S3 對 S0，gap＝inf；MI300X 上顯著的 11 格）：**9/11 同方向**（S3 較慢）→ PARTIAL（只記錄，不擋 D0）。
  方向相反的兩格：local／busy2／4096（MI300X +45.6%、H200 −2.5%）、nfs／busy1／16384（MI300X +13.2%、H200 −3.9%）
- (4) C6（vLLM 0.28＋OffloadingConnector，只報告）：

| L | vLLM 冷（重算） | vLLM 熱（CPU 載入） | vLLM CPU→GPU | harness 只算 | harness ÷ vLLM 冷 |
|--:|--:|--:|--:|--:|--:|
| 8K | 0.209 s（MI300X 0.443） | 0.046 s（0.292） | 42.0 GiB/s（4.02） | 0.360 s（0.645） | 1.72×（1.46×） |
| 16K | 0.482 s（1.304） | 0.073 s（0.553） | 48.0 GiB/s（4.09） | 0.831 s（1.439） | 1.72×（1.10×） |
| 32K | 1.224 s（4.584） | 0.128 s（1.080） | 49.9 GiB/s（4.11） | 2.131 s（3.759） | 1.74×（0.82×） |

**R2 校準**（H200 vs MI300X；`repro_vs_mi300x_gzfix.csv`）:
- C0 pinned 1 GiB（中位數，rep>0）：H2D 53.54 vs 53.59 GiB/s、D2H 50.04 vs 45.25；duplex：單向 H2D 53.53 vs 53.53、同時做 D2H 時 H2D 53.40 vs 46.12
- C1 f(i)（Llama-3.1-8B，chunk 0／79）：17.8／55.2 ms vs 28.9／98.8 ms；斜率 0.473 vs 0.888 ms/chunk
- C5 GPU 忙的程度（busy-levels 依規則選）：busy1 = duty 0.6 → 1.51×（MI300X 0.4 → 1.48×）；busy2 = duty **1.0** → **1.945×**（MI300X 0.7 → 2.07×）。
  H200 上 duty 掃到 1.0 才接近 2×，離目標 0.055×（< 0.10 的警告線）
- C7-self 限速器誤差：−0.41%～0.0%（MI300X −0.3%～0.0%）

**失敗與異常**:
1. R2-C6 第一次失敗（上面）。
2. **比較程式沒讀到 MI300X 的 `b_share.csv.gz`**：`m7_repro_compare.py` 的 `analyze()` 只複製 `.csv`／`.json`，MI300X 的 B 主設定在 git 裡是 `.csv.gz` →
   job 20553 的 `repro_vs_mi300x.csv` 裡 B 主設定三格的 MI300X 欄是 `NOT_MEASURED`，自我檢查寫 `summary_b_verdict.csv: rows 17 vs 4`。
   判定 (2) 只看 H200 的值和 δ*，不受影響。修正：`analyze()` 也解壓 `.csv.gz`。
   head node 重跑（run **`20261016-090548-h200-repro-compare-gzfix`**，§4.1 手動記錄殼，只用 CPU；不給 `--write-summaries`）→ `repro_vs_mi300x_gzfix.csv`、`repro_verdict_gzfix.json`。
   H200 那邊用快照＝job 20553 判定時的輸入（`b_share.csv` 只留 `20261016-063929-m7h-b-nfs-gap0`：20554 的 P1-B-rep36 09:00 起在往同一個檔追加）。
   結果：`criteria` 和 job 的 `repro_verdict.json` **逐字相同**；並排表 103 列只有 B 主設定三格的 MI300X 欄從 NOT_MEASURED 補上值；
   MI300X 重算 vs 已提交 summary 四個檔的最大差都是 **0.0**（`summary_b_verdict.csv` 17 列全對上）。job 的原檔不覆寫。
   作廢的兩次（run 目錄留著，放了 `SUPERSEDED`）：`20261016-090254-…`（直接讀 `$M7_RES`，讀到 rep36 寫到一半的 210 列；判定碰巧相同）、
   `20261016-090446-…`（快照漏了 `c6_vllm_backend.txt` → C6 的 backend 欄 NOT_MEASURED）。
3. 圖的字型：head node 和 compute node 的 matplotlib 都找不到 Noto Sans CJK TC（`findfont` 警告）；圖在 run 目錄、不進 git，不影響數字。

**與 MI300X 的差異**: 方向（判準 (1)(2)）重現。值得注意的差：
- vLLM 從 CPU 載入：H200 42–50 GiB/s，接近 C0 的 pinned H2D（53.5）；MI300X 只有約 4.1 GiB/s。
  r-h200-rest 的 P1-cpuvllm 照原決定沿用 MI300X 的 `M7_CPU_GIBPS=3.69`（＝MI300X C6 × 0.9），所以它重現的是「MI300X 的頻寬＋H200 的 GPU」，不是 H200 上 vLLM 的實際頻寬。
- harness 只算比 vLLM 冷啟動慢 1.7×（MI300X 32K 時 harness 反而快：0.82×；MI300X 的 vLLM 開 connector 後退回 TRITON_ATTN，F1）。H200 的 vLLM 用 FlashAttention 3。

---

## Milestone R-rest — 第一階段其餘的 GPU 實驗（`code/r_h200_rest.sh`）

**狀態**: **RUNNING** — job 20554（2026-10-16 08:5x `make submit-rest` 送出，afterok:20553；**09:00:20 開始**）。第一次送的 job 20454 在 07:48:33 因 20417 失敗被 Slurm 撤回，沒有開始
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

**狀態**: **RUNNING／PENDING** — 2026-10-16 08:5x `make submit-r1x` 重送：job 20555 `r-h200-la-f2`（afterok:20553；**09:00:20 開始**）、job 20556 `r-h200-qwen3`（afterok:20553,afterany:20554；等 20554）。第一次送的 20544、20545（06:3x，git `53f22a0`）在 07:48:33 因 20417 失敗被撤回，沒有開始
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

### job 20555 `r-h200-la-f2` — 停在 F2-gate（2026-10-16 09:06:34，FAILED exit 1:0，跑了 6 分 14 秒）
**開跑時的配額**（`quota -gw`，hgx-cn01）：`/dev/md1` 1,500,651,816 KiB／上限 1,610,612,736 KiB（1431.1／1536 GiB）→ **compute node 上讀得到 lab 配額**，20556 可以自己在下載前檢查。
**完成的步驟**（都 exit 0）:
| 步驟 | run_id | 產出檔 |
|:--|:--|:--|
| LA-ok | `20261016-090022-m7h-r1-ok-longalpaca7b` | `results/m7_explore_h200/correct_longalpaca7b.json`：HF 兩次、重算、主機往返都逐位元相同；argmax 相同；對 HF logit 最大差 0.09375 |
| LA-c1 | `20261016-090036-m7h-r1-c1-longalpaca7b` | `results/m7_explore_h200/calib_c1_longalpaca7b.csv`（228 列） |
| LA-vllmf2 | `20261016-090056-m7h-r1-vllmf2-longalpaca7b` | `results/m7_explore_h200/vllmf2_longalpaca7b.csv`（1116 列） |
| F2-probe | `20261016-090154-m7h-f2-probe` | run 目錄的 `f2_probe.csv`、`f2_counter_raw.csv`、`f2_probe_power_samples.csv`、`f2_probe_meta.json` |
LA 的數字和 MI300X 的比較留到全部跑完、在 head node 一起做。

**F2-probe 的關鍵數字**（NVML，H200 PCI `00000000:51:00.0`＝torch 的卡；driver 610.43.02、power limit 700 W）:
- 能量計數器（`nvmlDeviceGetTotalEnergyConsumption`）：解析度 1 mJ；**更新週期中位數 99.87 ms**（p10 95.77、p90 104.25、最大 105.14 ms；5 s 內讀 660 次、變了 51 次）；
  **讀一次 6.67 ms**（中位數；最大 19.67 ms）。MI300X（amdsmi）：更新週期 1.98 ms、讀一次 0.27 ms。
- 功率（ΔE/Δt）：閒置（沒載模型）79.7–80.7 W；閒置（載了模型）123.2–123.3 W（結尾 125.7 W）；不限速 pinned H2D 153.3–154.2 W；
  NFS 速度只載入（P_wait）123.5–123.8 W；只重算 64 chunk（P_b，約 2.1 s）636.3–649.1 W。
- ≥10 s 區段，ΔE/Δt 和瞬時功率（`NVML_FI_DEV_POWER_INSTANT`，有支援）平均的差：−0.12%～+0.84%（13 段）。

**F2-gate**（run `20261016-090631-m7h-f2-gate`；`results/m9_followup_h200/f2_gate.csv`）:
- 沒過的一條：**更新週期 ≤ 最短還原的 5%**：99.87 ms vs 門檻 5.21 ms（最短還原＝cpu35 的 Cake，用 H200 的 C1 f(i) 和頻寬算出 104.1 ms）。
- 過的：計數器有變（51 次）、≥10 s 區段的差 ≤5%（13/13）、NVML 和 torch 是同一張卡。
- 其他層的預測還原時間：cpu11 274.8 ms、cpu369 643.6 ms、local 411.8 ms、nfs 1699.0 ms（都比 2 s 短，所以每一層都過不了 5%）。
- probe 裡的 e_rec(i)（每次 20 個 chunk 背靠背，0.36–0.95 s）也只有 4–10 個計數器更新週期，同樣受影響。

**停下的原因與要決定的事**: 預先登記（`docs/research_20261010_followup/F2_energy_kappa.md` §2 第 42 行）：
「更新週期 > 最短一次還原時間的 5% → 改成 K 次還原連續跑、一起量、再除以 K；K 在看任何還原結果之前，只依步驟 (a) 的更新週期決定」。
`m9_f2_energy.py sweep` 沒有實作 K 次一起量 → 照 R-r1x 開頭寫好的：停下來回報，不自己加。**等使用者決定**（實作 K、或 H200 的 F2 記 NOT_MEASURED）。
沒跑的步驟：F2-cakee、F2-sweep、F2-final、F2-cakee-la7b、F2-sweep-la7b、F2-final-la7b、F2-la7b-clean。20556（qwen3）只等 20553（afterok，已滿足）和 20554（afterany），不受影響。

### F2 的決定與修改（2026-10-16 09:1x）
**使用者的決定**: 「K=21，只做 Llama」（LongAlpaca 的補充 sweep 不做）；「重跑 probe，k_rec=119」。
**K 怎麼來的**（只看 probe 的計數器更新週期，不看任何還原結果；F2 §2 第 42 行）:
- sweep：K ＝ ceil(更新週期 ÷ (5% × 最短還原))，最短還原＝cpu35 的 Cake 預測 104.1 ms。中位數 99.87 ms → 20；**最大值 105.14 ms → 21**（採用）。門檻變成 5% × 21 × 104.1 ＝ 109.3 ms。
- probe 的 e_rec（每個位置背靠背重算 k_rec 個 chunk 一起量；κ_E 用）：同一條規則用在最短的窗（chunk 0，H200 C1 f(0)＝17.818 ms）：
  窗要 ≥ 105.14 ÷ 5% ＝ 2.103 s → **k_rec ＝ ceil(2.103 ÷ 0.017818) ＝ 119**（MI300X 20）。第一次 probe 的 e_rec 窗只有 0.36–0.95 s（4–10 個更新週期）。
  預先登記的 5% 規則字面上只講「還原」；e_rec 這條是比照，使用者同意。
- 新 probe 會重量更新週期；gate 用 K=21 再判一次，沒過一樣停。
**程式的修改**:
- `m9_f2_energy.py sweep --k-batch K`：每個量測在同一個量測窗裡連續跑 K 次還原，寫出的 `secs`、`E_J`、`t_*` 是每次的平均，`secs_batch`、`E_J_batch` 是整段，另記 `k_batch`、`meet_min`、`meet_max`。
  K 次之間不清零：每次還原都會重寫全部 64 個 chunk，做的事一樣；如果把清零（4 GiB memset）放進窗，會多算到它的能量。
  第 0 輪的逐 chunk 驗證在第 K 次之後做（第一次是從全零開始，固定會漏寫的錯一樣抓得到）。動態 Cake 每次的會合點取中位數。暖機維持 K=1。K=1 時寫法和 MI300X 相同。
- `m9_f2_analyze.py gate --k-batch K`：門檻＝5% × K × 最短還原；另記照中位數、最大值各自需要的 K。`final` 的摘要 JSON 多記 `k_batch`。
- `r_h200_r1x.sh`：`F2_K=21`、`F2_KREC=119`，加 `--until <步驟>`。Makefile：`make submit-f2`（＝`sbatch --job-name=r-h200-f2 --time=10:00:00 code/r_h200_r1x.sh --part la --from F2-probe --until F2-final`）。
**測試**（head node，不用 GPU；暫存目錄）:
- 用 git 裡 MI300X 的 `f2_probe_raw.csv`、`f2_sweep_raw.csv`、`f2_la7b_sweep_raw.csv` 重跑 cakee、final（含 la7b）：cakee、verdict、sweep_summary、pareto、cost、kappa_e、probe_summary、la7b 的 verdict／sweep_summary／pareto／cost 除了 run_id、ts 都和 git 相同；
  counter_validation 只差 10/16 加的 `poll_src`、`secs` 兩欄。
- gate 用第一次 H200 probe：K=1 不過（和 job 一樣）、K=20 過（門檻 104.12 ms）、K=21 過（109.33 ms）；需要的 K：中位數 20、最大值 21。
- `mean_info`（K 次合成一列）的單元測試通過。sweep 的 K 迴圈要 GPU，沒辦法在 head node 試。第一次真的跑就是這個 job。
**時間**: 用 H200 的 C1 f(i) 和各層頻寬算，每輪約 110 s 的還原 × 21＋62 個量測各等 0.5 s → 5 輪約 3.2 小時（算術估計，不是量的；MI300X 實際比這種算法多花一些時間）。`--time=10:00:00`。


---

## Milestone R-cpu — 第一階段的 CPU 模擬，用 H200 的常數重跑（`code/r_h200_cpu.sh --part a`）

**狀態**: **67/67 步 PASS**（F6-summary 第一次停在 (c)，使用者決定後重跑；見下）
**決定**: 2026-10-16 使用者「gogo」（在 head node 先跑只需要 R2 常數的那批）
**執行時間**: 2026-10-16 09:53:21 → 10:44:50（F6-summary 失敗停下）；F7 用 `--from F7-run` 補跑 10:45:33 → 10:46:11
**在哪裡跑**: head node `bcm11-headnode01`（32 核、62 GB、58 個使用者）。h200q 不收純 CPU 的 job：`sbatch --test-only -p h200q --gres=gpu:0` →
`Assigned QOS: h200q_0g ... allocation failure: Invalid qos specification`。每一步 `nice -n 19`、`--procs 8`（MI300X 24；各設定用自己的 seed，process 數不影響數字）。
`H200_SLURM.md` §5 寫「head node 不跑重的 CPU 工作」〔推論〕：使用者同意這批在 head node 跑。
**run_id**: 66 個，`results/m8_directions_h200/r_h200_cpu_run_ids.tsv`（標籤 → run_id）；每個 run 目錄有 cmd.sh、context.txt（含 M7_*／M8_OUT／M9_OUT）、stdout、stderr、exit_code
**指令**: `bash code/r_h200_cpu.sh --part a`（`--dry-run` 看全部 67 條）。照 MI300X 留下結果的 run 的指令（run_commands.tsv），
逐條自動比對：正規化路徑與 `--procs` 後 66 條和 MI300X 相同；D1-analyze 的 `--inp` 寫完整路徑（MI300X 是 `d1_sweep.csv`），讀的是同一個主掃描。
冒煙測試（d1-smoke、f6-smoke）、中途改過的版本（sim-sweep 第一次、r1-sim 第一次）不跑。

**常數**（03 §R2：只換 GPU、不換「裝置」）:
- f(i)：H200 的 C1（`results/m7_write_policy_h200/calib_c1.csv`，run `20261016-061311-m7h-c1`）；LongAlpaca：`results/m7_explore_h200/calib_c1_longalpaca7b.csv`（run `20261016-090036-…`），`M7_CHUNK_BYTES`＝256 MiB（MI300X 的 r1-sim 輸出也記 `chunk_MiB`＝256，當時是 shell 設的、cmd.sh 沒記到）
- 限速層：MI300X 的 `tier_params.json`（`M7_TIER_PARAMS`，和 GPU 端一樣）
- capture 的 D2H：H200 C0 的 `D2H_pinned` 64 MiB 中位數取一位小數 → **49.5 GiB/s**（同一個算法在 MI300X 給 44.756 → 程式寫的 44.8）
- 第 1 輪 GRID 的 vLLM CPU 頻寬 3.69 GiB/s：沿用 MI300X（和 r-h200-rest 的 P1-cpuvllm 同一個決定；H200 的 C6 是 42–50 GiB/s）

**程式的修改**（預設＝MI300X 不變）: `m7_sim`（`M7_D2H_GIBPS`）、`m7_round1_sim`／`m7_round1_analyze`（`M7_EXPLORE`、`M7_RES`）、`m8_conc_sim`（`M8_OUT`、`M7_EXPLORE`）、`m9_f6_analyze`／`m9_f7_fscale_sim`（`M9_OUT`）。commit `e62f044`。
**回歸**（head node，MI300X 的輸入、不設任何新變數；暫存目錄）: r1_sim 兩個模型、d1_validate、f7 五個檔和 git **完全相同**；d1_verdict_cfg、f6_main_S5T_cfg 只差 `src` 欄（來源 run 名）；
sim_sweep 共同的 5070 列（13 個策略）數值和順序完全相同，現在的程式多 S4C、S4W 兩個策略（MI300X 10/09 跑 sweep2 之後才加，`REDTEAM_round1.md`）→ H200 的 sim_sweep／sim_probe 有 15 個策略，比方向只看共同的。

**產出檔**: `results/m7_write_policy_h200/sim_sweep.csv.gz`、`sim_probe.csv`；`results/m7_explore_h200/r1_sim_llama31_8b.csv`、`r1_sim_longalpaca7b.csv.gz`；
`results/m8_directions_h200/d1_*.csv`（18 個每格表＋seed 表、validate、overflow）；`results/m9_followup_h200/f6_*_cfg.csv`（6 個）、`f7_*.csv`（5 個）。
>1 MB 的兩個檔進 git 的是 `.gz`（原檔留在原處給之後的分析讀，加進 `.gitignore`）。

**關鍵數字**（只比方向；兩台都是模擬）:
- D1-validate：6 格的模擬中位數和 H200 的 r1-sim（`20261016-095639-m7h-r1-sim-llama31-8b`）相同（誤差 ≤1e-13%）
- D1 每格表（pass10＝≥6/10 seed 贏所有對手 ≥10%）MI300X → H200：主判定 `d1_verdict_cfg` **5 → 10**、`d1_verdict_mean` 5 → 9、`d1_afq16_S1D` 5 → 3、`d1_local8_S1D` 6 → 6、`d1_qmax16` 10 → 10、`d1_posthoc` 3 → 2、`d1_x3_S1D` 3 → 2；
  加進所有延後版對手後（x4 的 6 個表）兩台都是 **0**；S5T 的各表兩台都是 0
- F6 (a)+(b)（ρ_KV<1 的格、中位數、k=6/10）：S5T、S1D 都是 **0 格通過** → CONFIRMED（MI300X 也是 0）。主網格 ρ<1 的格數 MI300X 192、H200 178
- F6 l8q16 S1D（中位數）pass10：MI300X 2、H200 **0**
- F7（判準在 `research_20261010_followup/README.md`「追加 F7」）：free 在 f×0.5、f×0.6 都是 **0 個 ≥10% 的存活者** → 「重算變快後結論仍成立」（MI300X 同）。
  hold：f×0.6 1 個、f×1.0 1 個 ≥10% 存活者（11.6 GiB/s、hold、doc、nfs、cpu_frac 0.5，g_min 中位數 0.205）。只列出，不改 free 的判定（同 MI300X 的寫法）

**失敗與異常**:
1. **F6-summary（run `20261016-104426-m7h-f6-summary`）exit 1**：
   ```
   File ".../code/m9_f6_analyze.py", line 265, in summary
       raise RuntimeError(f"D1 過載格應為 5 格，讀到 {len(five)}")
   RuntimeError: D1 過載格應為 5 格，讀到 3
   ```
   (c) 的檢查把 MI300X 的「D1 有 5 個過載格」寫死。H200 的 `d1_afq16_S1D_cfg.csv` 只有 3 格 pass10（2 格和 MI300X 同座標：local8／doc／λ0.4／poisson／cpu 3.69 與 11.6；1 格是 H200 才有的：λ0.4／bursty／11.6）。
   (a)+(b) 在例外之前已算完並寫出 `f6_summary.csv`，但這個 run 是失敗的 → **這次不 commit f6_summary.csv**，等 (c) 決定後重跑。
   不管用哪一組格子，H200 的 F6 l8q16 S1D 表在那些格子的 n10 都 ≤3/10（沒有一格 ≥6/10）；MI300X 的 (c) 是 1/5 重現（λ0.2／bursty／3.69：7/10）。
2. 圖的字型、其他：無。

**F6-summary 的 (c)：使用者 10/16 決定「兩組都報」→ 重跑**（run `20261016-105240-m7h-f6-summary`，exit 0）
- `m9_f6_analyze.py`：寫死的 5 改成 `--d1-overload-expect`（預設 5＝MI300X）；加 `--extra-overload-ref`／`--extra-overload-out`（別的平台的 D1 過載格當對照，不進判定）。
  回歸：MI300X 的輸入、預設參數 → `f6_summary.csv`、`f6_c_overload.csv` 和 git 完全相同（1/5，CONFIRMED）。
- H200：`--d1-overload-expect 3 --extra-overload-ref results/m8_directions/d1_afq16_S1D_cfg.csv --extra-overload-out f6_c_overload_mi300x_cells.csv`
  - (a)+(b)：S5T、S1D 都是 0 格 → CONFIRMED（ρ<1 的格：H200 202、MI300X 216）
  - **(c) 判定（H200 自己的 3 格）：0/3 → 「corner is NOISE」**；對照（MI300X 的 5 格）：0/5。**MI300X 是 1/5 → 「只在持續過載時贏」**：這一項兩台方向不同
- 產出：`results/m9_followup_h200/f6_summary.csv`、`f6_c_overload.csv`、`f6_c_overload_mi300x_cells.csv`

**還沒做**（`--part b`，之後）: m7_sim validate（要 P1）、08 消融的分析（要 P2）、D3（要 P1、P2、A0；C7-anchor 在 H200 沒做 → 那部分 NOT_MEASURED）、第 1 輪的分析（要 P3、Qwen3）。
D4（要 Mooncake trace、SCBench）、D5（要模型 config）要先下載並核對和 MI300X 同一份。D6 用到 3090 量的反量化成本（H200 沒量，F3 屬第二階段）→ 先不做，等使用者決定。
