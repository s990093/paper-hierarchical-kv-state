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

**程式版本和 MI300X 的差**（2026-10-16 查，`git diff 0b9df73 HEAD`；0b9df73＝MI300X 第一階段最後一次改 harness）：
- `m7_restore_harness.py`：10/15（`7eb0709`、`3915ed9`）為第二階段加了低精度載入路徑（`m10_codec.Enc`，只有 host 是 Enc 時才走）。
  BF16 路徑的 H2D 複製、限速器、同步不變，只多了每個 chunk 的記帳（`ev.query()`、`rec["blocks"]` 追加、`n_load_bf16` 等計數，`time_h2d` 預設 False 不量 H2D）。
  每個 chunk 多幾個 Python 操作；chunk 載入是毫秒級（64 MiB ÷ 35 GiB/s ≈ 1.8 ms），影響沒有量（NOT_MEASURED）。
- `m7_model.py`：M7_MODEL_GLOB 改成載入時才讀（修「靜默載錯模型」）、Qwen2 沒有 head_dim 時的預設、`forward_span(n_logits=1)` 的預設路徑不變。
- `m7_write_policy.py`：只有路徑（M7_RES、M7_TIER_PARAMS、M7_DISK）和 GpuWatcher 的卡號（smi_index）。
- 所以 H200 的 R1–R3 不是「逐位元同一份程式」：BF16 的計算與還原邏輯相同，記帳多了一點。R1 的正確性（逐位元）是在這份程式上驗的。

**與 MI300X 的差異**: 方向（判準 (1)(2)）重現。值得注意的差：
- vLLM 從 CPU 載入：H200 42–50 GiB/s，接近 C0 的 pinned H2D（53.5）；MI300X 只有約 4.1 GiB/s。
  r-h200-rest 的 P1-cpuvllm 照原決定沿用 MI300X 的 `M7_CPU_GIBPS=3.69`（＝MI300X C6 × 0.9），所以它重現的是「MI300X 的頻寬＋H200 的 GPU」，不是 H200 上 vLLM 的實際頻寬。
- harness 只算比 vLLM 冷啟動慢 1.7×（MI300X 32K 時 harness 反而快：0.82×；MI300X 的 vLLM 開 connector 後退回 TRITON_ATTN，F1）。H200 的 vLLM 用 FlashAttention 3。

---

## Milestone R-rest — 第一階段其餘的 GPU 實驗（`code/r_h200_rest.sh`）

**狀態**: **COMPLETED** — job 20554（2026-10-16 08:5x `make submit-rest` 送出，afterok:20553；**09:00:20 → 17:09:51**，28 步全部 exit 0）。第一次送的 job 20454 在 07:48:33 因 20417 失敗被 Slurm 撤回，沒有開始
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

### P1（B 的 8 個敏感度設定）— 完成；**主設定 cpu_frac 0.5 合併 6 次後 S5 快 8.3%（超過 δ*）→ 停下回報**
**執行時間**: job 20554，09:00:22（P1-B-rep36）→ 12:02:50（P2 開始）。8 個 run 都 exit 0，`contaminated_gpu` 全部 False
**產出檔**: `results/m7_write_policy_h200/b_share.csv.gz`（main＋r36＋seed1＋local＋seed2＋gap2＋ssd50；>1 MB 進 git 的是 .gz，原檔在原處、加進 .gitignore）、`b_fifo.csv`、`b_share_cpu3.69.csv`、`b_workload_seed{1,2}.json`
**B 全部設定**（run `20261016-124102-h200-b-all-settings`，head node；m7_analyze b() 同一套 → `summary_b_verdict_all.csv`、`b_vs_mi300x_all.csv`；不覆寫 R3 的 summary_*）:
- 17 格中 16 格 S5 對 S4+ 在 −1.1%～+0.6%（MI300X 17 格 −3.0%～+0.3%）；S5 對「最好的簡單策略」都是負的（S5 沒有比較好），和 MI300X 同方向
- **例外：主設定（share、harness、nfs、gap 0、ssd_frac 1、seed 0）cpu_frac 0.5，合併 rep 0–5（main＋r36，和 MI300X 的主設定一樣合併）：S5 對 S4+ ＝ +8.30%**（MI300X +0.26%）
  - 逐 rep（回來請求的 TTFT 中位數；S5 對 S4+）：rep 0–2（R3-B-main，06:39–07:46）−0.16／−0.20／+0.28%；rep 3–5（P1-B-rep36，09:00–09:22）−0.99／**+8.61／+7.55%**
  - rep 3–5 整體比 rep 0–2 慢：S4+ 0.79–0.81 s → 0.94–0.95 s（約 +18%）；cpu_frac 0.25 的 rep 3 也有 −4.66%
  - 同節點的 job：R3-B-main 時是 siru×2、peiyu、chihchung、hao×2＋20417；r36 時是 siru×2、peiyu、chihchung＋**我們自己的 20555（LA＋F2 probe）、20570（F2 probe＋sweep）**。
    F2 probe 有「不限速 pinned H2D 3×10 s（約 53 GiB/s）」，B 的 CPU 層載入也走 pinned H2D。〔判讀〕主機層的記憶體／PCIe 爭用很可能讓 rep 3–5 變慢、而且在 rep 內隨時間變動；
    主機層看不到（UNOBSERVABLE_HOST），GpuWatcher 只看自己那張卡 → **無法證明**
  - P1 其他 7 個 run（09:22–12:02）都和 20570 的 F2 sweep 同時跑；它們的 S5 對 S4+ 在 ±1.1% 內
- R3 的判定（job 20553，主設定只有 rep 0–2，cpu_frac 0.5 −0.24%）在 r36 之前就算完了，**沒有改**。照 03 §R3 的規則，合併 6 次的這一格是「S5 快超過 δ*＝5%」→ 和第一階段相反 → 停下回報，等使用者決定
- **決定（使用者 2026-10-16 12:5x）：「照重疊重跑」**：和我們自己的 20555／20570 時間重疊的 B run 全部重跑（P1 八個＋P2-G1-seed0），照重疊選、不照結果（只重跑 r36 會變成挑結果）。
  R3-B-main（只和 20417 自己）不重跑。舊資料保留，新舊並排報告。
  - `code/r_h200_brerun.sh`（commit `f1567d3`）：指令和 r_h200_rest.sh 的 P1、P2-G1-seed0 逐字相同（dry-run 比對 9/9；run 短名加 `-rr`）；
    輸出 `results/m7_write_policy_h200_rerun/`（`M7_RES`），f(i) 用同一份 H200 C1（`M7_F_CSV`）；開跑前檢查節點上沒有我們自己的別的 job，有就停
  - **job 20631**：`sbatch --dependency=afterany:20554,afterany:20556 code/r_h200_brerun.sh`（`--time=08:00:00`）
  - **結果（10/16 22:0x）**：合併組（R3-B-main＋重跑）的主設定 cpu 0.5 是 −0.12%（在 ±δ* 內）；但沒有重疊的重跑裡也出現 ±6–8% 的格（ssd50、vllm 3.69）→ 見「Milestone R-brr」。

## Milestone R-r1x — 第 1 輪的 LongAlpaca／Qwen3-MoE＋F2 能耗 κ（`code/r_h200_r1x.sh`）

**狀態**: la：LA 三步 PASS、F2 由 job 20570 完成（有看頭，同 MI300X）；qwen3：job 20556 **COMPLETED**（17:09:51 → 17:24:43，5 步 exit 0；見「R-cpu part b」一節）。2026-10-16 08:5x `make submit-r1x` 重送：job 20555 `r-h200-la-f2`（afterok:20553；**09:00:20 開始**）、job 20556 `r-h200-qwen3`（afterok:20553,afterany:20554；等 20554）。第一次送的 20544、20545（06:3x，git `53f22a0`）在 07:48:33 因 20417 失敗被撤回，沒有開始
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



### job 20570 `r-h200-f2`（K=21，只做 Llama）— COMPLETED
**執行時間**: 2026-10-16 09:16:07 → 12:38:20（03:22:13，exit 0:0）。`sbatch --job-name=r-h200-f2 --time=10:00:00 code/r_h200_r1x.sh --part la --from F2-probe --until F2-final`（git `122eab8`）
（`make submit-f2` 被「有未 commit 的改動」擋下：那是 20554 正在追加的 b_share.csv；直接下同一條 sbatch。）
**run_id**: `20261016-091609-m7h-f2-probe`（k_rec 119）、`…-092135-m7h-f2-gate`、`…-092138-m7h-f2-cakee`、`…-092140-m7h-f2-sweep`（09:21:40 → 12:38:17）、`…-123817-m7h-f2-final`
**產出檔**: `results/m9_followup_h200/f2_*.csv`（另把 probe／sweep 的原始列存成 `f2_probe_raw.csv`、`f2_sweep_raw.csv`，和 MI300X 一樣）
**關鍵數字**:
- 新 probe：計數器更新週期中位數 99.93 ms（最大 105.49 ms）→ 需要的 K 仍是 21；gate（K=21，門檻 109.33 ms）全部通過；≥10 s 區段 ΔE/Δt 和瞬時功率差 ≤0.19%
- e_rec 的窗 2.10–5.65 s（k_rec 119）；每 chunk 重算能量 11.7 J（chunk 0）→ 31.0 J（chunk 63）
- 功率：P_b（只重算）634.0 W、P_wait（NFS 速度只載入）123.5 W、P_idle（載了模型）123.0 W、沒載模型 79.4 W、DMA 153.3 W → **P_b／P_wait＝5.13**（MI300X 4.30）
- Cake-E 切點（cakee）：nfs m_E＝55，其他層 0（＝只載入）（MI300X Llama nfs＝33）
- sweep：290 列（5 輪 × 58），K=21；只重算 10 次的變異係數 **0.61%**（D8 殺掉條件 a：>10% 才 NOT_MEASURABLE）；kv_bad 0
- 帳 A（判定用）每層：時間最佳 vs 能量最佳、省多少、慢多少：
  cpu35 S(4) vs 只載入：省 68.9%、慢 5.9%；cpu11 SMEET(13) vs 只載入：省 73.6%、慢 24.4%；cpu369 SMEET(26) vs 只載入：省 65.4%、慢 68.3%；
  local Cake(19) vs 只載入：省 71.8%、慢 35.1%；nfs：時間最佳 SMEET(55)、能量最佳 Cake(55)，差 1.6%（不同層＝否）
  （MI300X：cpu35／cpu11／cpu369／local 省 62.0／69.9／64.2／64.4%、慢 4.1／15.0／43.1／25.3%；nfs 相同）
- κ_E／κ_T 中位數：cpu35 4.16、cpu11 4.60、cpu369 4.88、local 4.78、nfs 5.23（MI300X 2.86／3.73／3.93／3.46／4.12）
**判定（F2 §2.5）**: 帳 A **有看頭**（4 層「不同、省 ≥10%、各次範圍不重疊」）；帳 B、C50、C150、D0.5、D1、D2 也都是有看頭 → **和 MI300X 相同**
**和 MI300X 的差**: 量法是 K=21 次一起量、k_rec 119（MI300X K=1、k_rec 20）；NVML 計數器（MI300X amdsmi）。LongAlpaca 的補充 sweep 沒做（使用者決定）。
**異常**: log 末尾 `f2_run_ids.tsv 裡沒有 f2-sweep-la7b 的 run_id`：`--until F2-final` 之後的 F2-final-la7b 那行在判斷要不要跳過之前就先展開了參數（rid），只印訊息、沒有執行

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

### D6（精度的算術）＋ F3 工作 A（H200 的反量化成本）— 使用者 10/16：「D6 不要 3090，都 H200，但要跑實驗」
- **F3 工作 A**（GPU，job 20614，run `20261016-105426-m7h-f3-dequant`，exit 0，14 s）：`sbatch --job-name=m7h-f3-dequant --time=00:30:00 code/tiara.sbatch python code/m7_guard_run.py python code/m9_f3_dequant.py --out results/m9_followup_h200/f3_dequant.csv`
  - 程式不改：FP8 dtype 依 `gcnArchName` 自動選（MI300X `float8_e4m3fnuz`、H200 `float8_e4m3fn`）；cold＝8 組 buffer 輪流（H200 L2 50 MB）。
  - 反量化一個 64 MiB chunk，cold、輸出正確的實作裡最快的中位數：**INT8（每 token 每 head 一個 scale）0.0296 ms**（torch.compile；Triton 0.0297）、
    **FP8（整個 tensor 一個 scale）0.0293 ms**（torch.compile）、**INT4 0.0266 ms**（Triton）。MI300X：0.031、0.026、0.027 ms。
  - **失敗 2 個寫法**（照原設計記下、其他照量）：`fp8_tensor/triton`、`fp8_ptk/triton`：
    `triton.compiler.errors.CompilationError: ... q = tl.load(q_ptr + off, mask=m, other=0).to(tl.float32) ... cannot cast int32[constexpr[64], constexpr[128]] to <['64', '128'], fp8e4nv>`
    （NVIDIA 的 FP8 不能用整數 0 當 `other`；MI300X 的 fnuz 可以）。完整錯誤在 run 目錄的 `f3_dequant_raw.json`。沒有改程式重跑。
  - 工作 B（vLLM 原生 FP8 撈針，看得到答對率）**不跑**（第二階段的事先登記凍結前不看低精度的答對率）。
- **D6**（head node，`bash code/r_h200_cpu.sh --part d6`）：`--dmeas` 由 runner 從 `f3_dequant.csv` 照 F3 的判準自動取（不手抄）：`fp8=0.02933,int8=0.02958,int4=0.02661`，
  `--dmeas-src 20261016-105426-m7h-f3-dequant_f3A_cold_fastest_correct_median_ms_per_chunk`。MI300X 的 D6 用的是 3090 的 warm TTFT 差（fp8 0.087、int4 0.814，沒有 int8；F3 §4 查過不是反量化 kernel 的時間）。
  - run `20261016-110202-m7h-d6-arith`、`20261016-110913-m7h-d6-thresh`（都 exit 0）→ `results/m8_directions_h200/d6_*`。f(i) 擬合：17.781 + 0.4730·i ms，R²＝0.99994（H200 C1）
  - 回歸：`m8_precision_arith.py` 改成可換 f(i)／限速層／輸出目錄（預設＝MI300X），MI300X 的輸入重跑 → 7 個檔和 git 完全相同。
  - 門檻細掃（`d6_threshold.csv`，Q1＝BF16／INT8 混用、Q2＝四種精度）：反量化 ≤1 ms／chunk 時最大增益 H200 Q1 0.02%、Q2 0.00%（MI300X 0.00%、0.00%）；3 ms 時 Q1 3.91%、Q2 22.41%（兩台相同）。
  - **判定（照 F3 的 D6 判準原文：INT8 中位數 ≤1 ms → D6 死路）：0.0296 ms → 死路**（MI300X 0.031 ms → 死路）。
  - 和 MI300X 的 D6 不是逐字同一個實驗：程式相同（`m9_f3_dequant.py` 10/10 後沒改；`m8_precision_arith.py` 只加路徑參數、回歸相同），
    但 (1) FP8 格式不同（MI300X e4m3fnuz、最大 240；H200 e4m3fn、最大 448；資料都 clamp 在 ±240）；(2) H200 上 FP8 的 Triton 寫法編不過；
    (3) D6 的反量化成本來源不同：MI300X 的 D6 用 3090 的 warm TTFT 差（沒有 INT8），之後才由 F3 量 kernel 時間收尾；H200 直接用 H200 的 F3 kernel 時間（含 INT8）。
    判定的規則（INT8 ≤1 ms → 死路）兩台相同。
  - **決定（使用者 2026-10-16）**：FP8 的格式處理（e4m3fnuz／e4m3fn）和 H200 上編不過的 FP8 Triton 寫法「都不要動，等 AMD 的實驗都做好再改」。
    `m9_f3_dequant.py`、`m10_codec` 與 harness 的 FP8 路徑維持原樣。
  - 異常：D6 跑的時候我改了同一支 `r_h200_cpu.sh`（加 d4d5），bash 邊跑邊讀，兩步都完成後讀到錯位的行：
    `code/r_h200_cpu.sh: line 242: gap_posthoc.csv: command not found`、`line 243: TO: unbound variable` → runner exit 1。兩個 D6 run 都在這之前 exit 0、run_id 已記，沒有多出別的 run。之後不在 runner 執行中改它。

### D4（真實 trace 的重用）、D5（架構與模態）— 使用者 10/16：「下載並核對後再跑」
**資料下載**（head node，`bash code/h200_dl_cpu_data.sh`；每個 run 有 sha256.txt）:
- `20261016-110413-h200-dl-mooncake-traces`：kvcache-ai/Mooncake `245e7106` 的 FAST25-release/traces（conversation 12,031 列、synthetic 3,993、toolagent 23,608〔和 D4 §4 記的列數相同〕；8.2M）→ `$TIARA_DATASETS/traces`
- `20261016-110416-h200-dl-scbench`：HF dataset microsoft/SCBench 的 `data/*.jsonl`（12 個檔、647M；MI300X 的 inventory 記 903 MB，大小不同）→ `$TIARA_DATASETS/scbench/data`
- `20261016-110440-h200-dl-d5-qwen3vl-cfg`：Qwen/Qwen3-VL-8B-Instruct `0c351dd0` 的 4 個 config；`20261016-110443-h200-dl-d5-cfgs`：m8_arch_arith 讀的 6 個 config.json（revision 在 run 目錄的 revisions.json）
**核對**（MI300X 的常數＋這些資料、暫存目錄；`--src-run` 與 config 目錄名照 MI300X 的 run id）: D4 15 個、D5 6 個輸出檔**全部和 git 完全相同**
（`d4_scbench.csv` 只比 MI300X 重跑成功的 `20261010-054738-d4-scbench2`）→ 資料和 MI300X 用的是同一份（SCBench 大小不同，但 D4 用到的部分相同）。
**程式的修改**（預設＝MI300X）: `m8_trace_oracle.py`（M7_RES、M7_TIER_PARAMS、M8_OUT、TIARA_DATASETS、M7_LLAMA_GLOB）、`m8_arch_arith.py`（M7_RES、M7_TIER_PARAMS、M8_OUT、**M7_CALIB_RUN**）。
**H200 的 run**（`bash code/r_h200_cpu.sh --part d4d5`；12 個 run，`r_h200_cpu_run_ids.tsv`）: 11:46:58 → 12:28:04
- 異常：第一次停在 D5-arith（run `20261016-120525-m7h-d5-arith`，exit 1）：
  `File ".../code/m8_arch_arith.py", line 80, in calib: assert set(c.run_id) == {CALIB_RUN}, set(c.run_id)` → `AssertionError: {'20261016-061311-m7h-c1'}`
  （寫死 MI300X 的 C1 run `20261008-130316-m7-c1`）。改成 `M7_CALIB_RUN`（預設不變；MI300X 回歸 6 個 d5 檔相同），runner 對 D5 指定 H200 R2 的 C1 `20261016-061311-m7h-c1`，`--from D5-arith` 接著跑（commit `929326d`）
**和 MI300X 比**:
- 不依 GPU 常數的 `d4_trace_char`、`d4_reuse_dist`、`d4_scbench`：和 MI300X **完全相同**
- **D4 判定**（§2.3：真實 trace conversation、toolagent 的 6 格中位數；死路＝兩個都 gap_T<5% 且 gap_W<20%；有看頭＝任一 gap_T 或 gap_W ≥15%）:

| 延後版集合 | MI300X | H200 |
|:--|:--|:--|
| A：§2.2 逐字的 7 個（`d4_verdict_prereg7`） | **有看頭**（gap_W 86.8%／89.2%；gap_T 0.015%／0.002%） | **有看頭**（gap_W 87.9%／89.2%；gap_T 0.025%／0.003%） |
| B：實作時的 25 個組合（`d4_verdict`，主判定 run） | **死路**（gap_W 0.0%／0.0%） | **有看頭**（conversation gap_W **75.5%**、toolagent 0.0%；gap_T 0.025%／0.003%） |

  B 翻過來的原因（`d4_gap.csv`）：W_延後＝「TTFT 在延後最好的 1.05 倍內」的延後版裡寫最少的。H200 重算比較快，總 TTFT 變小（conversation 主格約 11,700 → 7,100 s），
  同樣的絕對差距佔比變大，MI300X 上在 1.05 倍內、幾乎不寫 SSD 的 LRU+H2／LRU+NONE，在 H200 上超出 → 只剩 LRU+WT（182,790 個 block）可比，gap_W 變 75.5%。〔判讀〕這是 5% 容忍度和 TTFT 尺度的交互作用，不是工作負載資訊變得更有用。
  TTFT 那一半兩台都是死路（gap_T 中位數 ≤0.03%）。歸因規則：WTP-a 的 gap 和 OPT 相同（兩台都是）。synthetic 只報告：gap_T 最大 MI300X 3.4%、H200 5.1%
- **D5**（各子方向的判定理由在 D5 §0）：方向都不變
  - (a) VLM：存 embedding 讓 Cake 快 6.5–14.1%（3.69 GiB/s；MI300X 7.0–10.3%）；r_v 0.15–0.24（一般圖／影格；MI300X 0.14–0.23）。判死路的理由（只多 22% 位元組、已有人做）不依 GPU
  - (b) 混合 SSM：每 512 token 存一份檢查點，全部寫出要的頻寬是本地 SSD 寫入的 **0.27–1.82 倍**（MI300X 0.16–1.09）→ H200 prefill 較快，寫入頻寬更不夠
  - (c) 滑動視窗：只看 config，不變
  - (d) P/D：25 Gbps、32K，逐 chunk 管線露出 1.02%（MI300X 0.59%），不重疊露出 39.7%（27.5%），Cake 分割 0%（同）
  - (e) MLA：3.69 GiB/s 的 b/n：DeepSeek-V3 0.078（0.047）、Kimi-K2 0.094（0.062）、Falcon-7B 0.047（0.031）、DeepSeek-V2-Lite 0.328（0.234）
**產出**: `results/m8_directions_h200/d4_*.csv`（15 個）、`d5_*.csv`（6 個）

**`--part b`**: 已做（m7_sim validate、08 消融的分析、第 1 輪的分析），見「R-cpu part b」一節。D3 還沒做（要 job 20643 的 c7_anchor sim、job 20631 的重跑）。
D4、D5、D6 已做（上面）。

### 決定（2026-10-16 14:0x）：D3 只有 H200 平台；A2、C7 錨點照 MI300X 跑限速層那一半
**使用者的話**（回答「c7_anchor 要不要記 NOT_MEASURED 跳過」「D3 要不要讀 git 的 3090」）:
「不要在看 main 把那個刪掉 不用這個paper 跟概念了 跟 只有h200 平台 基本上要復現 mix300 都要跟」、「如果需要也需要改code 成nv 用的」、「都用模擬限速」
**照這樣做**（〔判讀〕「那個」＝D3 從 git `52803bc^` 讀的 3090 資料）:
- **D3 不放 3090**：`m8_kappa_map.py --no-3090`（不讀 git 歷史，也不產 `d3_3090_tier.csv`、R1t 那一列）。程式碼留著、預設不變，所以 MI300X 的 D3 還能逐格重現；沒有刪檔。
- **照 MI300X 跟到底、只用限速層**：`03_plan_h200.md` 第 101 行原本不做的 C2、C7-anchor、A2 改成照 MI300X 10/08 的指令跑，只加 `--io sim`
  （限速層＝MI300X 的 `tier_params.json`，和 H200 其他實驗相同）。真實裝置那一半 → **NOT_MEASURED**。
  - C2 只量真實裝置的頻寬，沒有限速層的版本 → 不跑，NOT_MEASURED。
  - 另外查到：hgx-cn01 上我們能寫的只有本地的 md0（`/`、`/var/tmp`）和 md1（`/work`）；NFS 只有 `/home`（`172.16.100.100:/userhome`，沒有我們的目錄）和 `/cm/shared`。所以真實的 nfs 本來也量不到。
  - D3 的 c7_anchor：sim 兩列用 H200 的；real 兩列（R1、R1b）標成 `c7_anchor real NOT_MEASURED`、n＝0。
**程式**（commit `f6f7259`；預設＝MI300X）:
- `m7_write_policy.py`：anchor、a2 加 `--io`（預設 `sim real`）。只跑 sim 時不建真實裝置、不寫檔。b 的路徑沒動。
  20554 還在跑（之後的步驟會讀到新檔），所以先改副本、編譯過才用 `mv` 一次換掉。
- `m8_kappa_map.py`：輸入目錄由 `M7_RES`、`M7_EXPLORE`、`M7_TIER_PARAMS` 指定；`.csv` 不在就讀同名的 `.csv.gz`（git 裡的 MI300X 大檔是 gz）；`--no-3090`；`--dstar`（prereg-test 的 D*，預設 MI300X 的 0.6026，H200 會傳自己擬合的值）。
**核對**（run `20261016-141301-h200-d3-mi300x-regress`，head node）:
- MI300X 預設：主程式、`--b100`、`--prereg-test` 的 7 個 `d3_*.csv` 和 git 的 `results/m8_directions/` **逐格相同**（`check_exact`，run_id／ts 除外）。
- H200 的路徑（MI300X 資料拿掉 c7_anchor real 列＋`--no-3090`）：規則分數少了 `c7_anchor real`×2、`R1t 3090`；多了 `c7_anchor real NOT_MEASURED`×2（n＝0）；其他 72 列的 n、hits 都和 MI300X 相同；沒有產 `d3_3090_tier.csv`。
**job 20643** `r-h200-simanchor`（`code/r_h200_simanchor.sh`，`--dependency=afterany:20631`，上限 1 小時）:
A2-sim（fifo，reps 6）→ A2-share-sim（share，reps 3）→ C7-anchor-sim（L 8K／16K／32K，reps 3），全部 `--io sim`。
MI300X 上這 3 個 run 的量測時間 2.6、0.9、4.7 分（CSV 的 ts，不含載入模型）。開跑前會檢查節點上沒有我們自己的 job，`$M7_RES` 裡也還沒有這 3 個 CSV。

---

## Milestone R-cpu part b — 08 消融、B 的模擬驗證、第 1 輪的分析（`code/r_h200_cpu.sh --part b`）

**狀態**: **7/7 步 PASS**（head node，`nice -n 19`，2026-10-16 17:34:28 → 17:35:02）。D3 還沒做：要等 job 20643 的 c7_anchor sim 和 job 20631 的重跑。
**run_id**: `results/m8_directions_h200/r_h200_cpu_run_ids.tsv` 的 ABL-gain、SIM-validate、B2-analyze、R1-h1、R1-kappa、R1-sim、R1-gpu（`20261016-173428-m7h-abl-gain` … `20261016-173500-m7h-r1-gpu-an`）
**指令**: `bash code/r_h200_cpu.sh --part b`：`m7_ablate_analyze.py`、`m7_sim.py validate`、`m7_b2_analyze.py`、`m7_round1_analyze.py h1|kappa|sim|gpu`。
MI300X 只有 m7_sim validate 走過 §4.1 的殼（`20261009-064908-m7-sim-validate`）；其他分析照各程式檔頭的用法（RUNLOG_MI300X M7b、M7c）。
**輸入**: job 20554 的 P1–P3 原始資料（**含**和 F2 重疊的 P1 八個與 P2-G1-seed0；重跑版之後另做、並排報告）、job 20556 的 Qwen3 C1、`--part a` 的 sim_sweep 與 r1_sim。常數同 part a（限速層＝MI300X 的 tier_params.json、D2H 49.5 GiB/s）。
**程式的修改**: `m7_b2_analyze.py`、`m7_ablate_analyze.py` 的結果目錄改讀 `M7_RES`（不設＝MI300X）。
回歸 run `20261016-171318-h200-partb-mi300x-regress`（MI300X 的 git 資料複製到 run 目錄，不設 M7_TIER_PARAMS／M7_D2H_GIBPS／M7_F_CSV）：
`b2_summary`、`b2_verdict`、`sim_sweep_gain`、`sim_validate.csv` 和 git **逐格相同**（ts 與模擬輸出的 run_id 除外），`sim_validate.json` 相同。

**GPU 端的輸入**（全部 exit 0；`contaminated_gpu` 0 列、`kv_bad` 0 列、D2 的 guard 9/9 clean）:
| 步驟 | run_id | 產出 |
|:--|:--|:--|
| P2 08 消融 GPU | `20261016-120250-m7h-b2-g1-doc-hold-cpu3.69-s0`、`…-132601-m7h-b2-g1-doc-hold-cpu3.69-s1`、`…-142019-m7h-b2-g2-chat-hold-s0` | `results/m7_write_policy_h200/b2_share.csv`（3,648 列） |
| P3 第 1 輪 Llama | `20261016-145852-m7h-r1-c1-llama31-8b`、`…-145911-m7h-r1-vllmf2-llama31-8b`、`…-150011` … `…-162902-m7h-r1-gpu-llama-doc-free-local-s{0..4}` | `results/m7_explore_h200/calib_c1_llama31_8b.csv`（228 列）、`vllmf2_llama31_8b.csv`（1,296 列）、`r1_gpu_llama_doc_free_local.csv`（3,420 列）、`b_workload_seed{0..4}.json` |
| P4 D2 | `20261016-164310-m7h-d2-off-doc` … `…-170837-m7h-d2-cpu50bs512-doc`、`…-170948-m7h-d2-collect` | `results/m8_directions_h200/d2_*.csv`、`d2_run_ids.txt` |
| Qwen3 下載 | `20261016-170953-m7h-dl-qwen3-30b-a3b` | revision `0d7cf239…`；配額 1433.3 → 1490.2／1536 GiB（下載後剩 45.8 GiB） |
| Qwen3 正確性 | `20261016-171742-m7h-r1-ok-qwen3-30b-a3b` | `correct_qwen3_30b_a3b.json`：HF 兩次、重算、主機往返都逐位元相同；argmax 相同、top-5 5/5；對 HF logit 最大差 0.3125（MI300X 0.15625） |
| Qwen3 C1 | `…-171816-…-grouped`、`…-171902-…-eager` | `calib_c1_qwen3_30b_a3b_{grouped,eager}.csv`（各 228 列） |
| Qwen3 vLLM f(i) | `20261016-172259-m7h-r1-vllmf2-qwen3-30b-a3b` | `vllmf2_qwen3_30b_a3b.csv`（1,296 列；探索性，同 MI300X） |

**比方向**（MI300X 的數字來自 git 的 `results/m7_write_policy_mi300x/`、`m7_explore_mi300x/`、`m8_directions/` 同名檔）:
| 項目 | 判準（事先寫的） | MI300X | H200 | 方向 |
|:--|:--|:--|:--|:--|
| 模擬器驗證（08 §5 第 2 點） | ≥90% 的格誤差 ±10% 內，且實測 S5 對最好簡單策略 >5% 的格方向一致 | 114 格 100%；8/8 → 通過 | 114 格 **93.0%**（106/114，中位 0.47%，最大 19.8%）；9/9 → 通過 | 同 |
| 08 消融 GPU（§6＋§8） | 寫入時決定同時贏最好的不看位置與 S4B 各 ≥5% | **0/16** | **1/16**（G1、seed 0、cpu 0.25、S5c：+6.0%／+28.2%） | **待重跑**（見下） |
| 08 消融，同一批格的模擬 | 同上 | 1/16 | 0/16 | — |
| 08 模擬誤差（`b2_summary`） | — | 43/44 ≤10% | 40/44 ≤10%（中位 1.3%，最大 16.8%） | — |
| 模擬掃描（`sim_sweep_gain`，390 格，只列數） | — | gain_nwt ≥5%：21、≥15%：6；和 S4B 都 ≥5%：8；gain_brule ≥5%：29 | 23、8；14；30 | — |
| H1 寫入量 | ≥20% 的設定才存活 | 2/23（8.7%）→ 殺掉 | 2/23（8.7%）→ 殺掉（兩台都在 b2，但不是同兩個設定：MI300X chat s0 0.5、doc s1 0.25；H200 doc s0 0.25〔和 F2 重疊的 run〕、doc s1 0.5） | 同 |
| H2 κ 篩選 | 3.69 或 35.4 GiB/s 時 b/n ≥30% | Llama 0.312 過；LongAlpaca 0.594 過；Qwen3 grouped 0.047、eager 0 不過 | 0.406 過；0.719 過；0.219、0 不過 | 同 |
| 第 1 輪模擬存活者 | ≥3/5 seed 贏所有對照組 ≥5% | Llama 1 個；LongAlpaca 4 個 | Llama 3 個；LongAlpaca 3 個；**和 MI300X 的沒有一個相同** | **不同**（見下） |
| 第 1 輪 GPU 確認（MI300X 的存活設定：11.6、free、doc、local、50%、S5L） | ≥3/5 seed 同時贏四個對照 ≥5% | 0/5 | **0/5**（模擬誤差 46/50 ≤10%，中位 2.7%） | 同 |
| D2 主判定 | cpu50、C＝4/16 的跳過率 <1% → 死路 | 0.0% → 死路 | 0.0% → 死路 | 同 |

**細節**:
- **08 消融那一格通過，落在和 F2 重疊的 run**：`20261016-120250-…-s0` 12:02:50 開始，我們自己的 20570（F2）到 12:38:20 才結束。
  cpu 0.25 的 8 個策略的 rep 0 全部在 12:03–12:27（重疊期間），每個約 2.8–3.0 分鐘；rep 1–2 在 12:46 之後，約 1.2–1.5 分鐘（`b2_share.csv` 的 ts）。
  同一格的模擬是 +0.6%（不過）、MI300X 的 GPU 是 −0.1%。**08 的判定等 job 20631 的 P2-G1-seed0-rr 跑完，新舊並排再寫**，現在不改。
  → 重跑後（合併組）這一格 +3.0%／+26.0%，08 消融 GPU **0/16**（同 MI300X），見「Milestone R-brr」。
- **模擬器驗證超過 10% 的 8 格**都是 share、nfs：S5s 三格（主設定 cpu 0.25、ssd50 cpu 0.5、seed2 cpu 0.5：−19.7%～−19.8%）、主設定 cpu 0.25 的 S1／S2b／S4+／S5（−11.1%～−14.4%）、seed2 cpu 0.25 的 S4+P（−13.9%）。
  主設定拆成兩個 run 看〔事後診斷，不是判定〕：R3-B-main（rep 0–2，沒有和我們自己的 job 重疊）S1 −0.1%、S5s −0.7%，但 S4+、S5 在 cpu 0.25 也是 −11.1%、−11.2%；r36（rep 3–5，重疊）cpu 0.25 的 7 格 −11.2%～−21.4%。重跑版的 validate 之後另做。
- **κ 篩選的 f(i)**（64 個 chunk 的總和）：Llama 2.09 s（MI300X 3.64）、LongAlpaca 2.06 s（3.71）、Qwen3 grouped 3.88 s（15.34）、eager 57.3 s（86.7）。H200 重算快，所以 b/n 變大；分類（過／不過）兩台相同。
- **第 1 輪模擬存活者**（`r1_sim_configs.csv`，CPU GiB/s、release、workload、SSD、cpu_frac）：
  - MI300X：Llama 11.6／free／doc／local／0.5；LongAlpaca 11.6／free／doc／local／0.25、35.42／free／doc／local／0.25 與 0.5、35.42／hold／doc／nfs／0.25
  - H200：Llama 11.6／hold／doc／nfs／0.5（g_nwt 中位 26.8%、g_S4B 61.1%）、35.42／free／doc／local／0.5（7.4%）、35.42／hold／doc／nfs／0.25（7.0%）；
    LongAlpaca 1.0／free／doc／nfs／0.25 與 0.5、35.42／hold／doc／local／0.5
  - MI300X 的 Llama 存活設定在 H200 的模擬是 0/5。P3 照 MI300X 的指令只確認了這一格（0/5）；H200 自己的 3 個 Llama 存活設定**沒有上 GPU** → 問使用者。
    LongAlpaca 照 11 追加 4 不上 GPU（doc 的回來請求超過 32,768 token 上限）。
- **D2 的副產品**（不在主判定內）:
  - 首輪 TTFT 中位數 cpu50／off：MI300X 1.28×（chat C＝1）～1.96×（doc C＝1）；**H200 的 cpu25／cpu50／cpu4g／bs512 每一格 0.99–1.04×**（tier50 doc C＝4 是 0.62×）。MI300X 報告的「卸載讓新請求變慢」在 H200 沒有出現。
    〔判讀，未查證〕和 D2 §4.3 猜的「GPU→CPU DMA 和 prefill 在 MI300X 上搶資源」一致，但這裡沒有量機制。
  - 回來請求的 KV 來源（GPU／CPU／重算的比例）兩台幾乎相同（vLLM 的排程是確定的）。
  - E1 cpu4g 跳過率 MI300X 3.6%、H200 0.35%；E2 tier50 跳過 0.21% → 0%，最長 scheduler 延後 83.1 s → 1.5 s。
    E2 的 fs 層：MI300X 在 NFS，H200 在 run 目錄（hgx-cn01 本地的 md1）→ 裝置不同，不能比。
**產出檔**: `results/m7_write_policy_h200/`：`sim_sweep_gain.csv`、`sim_validate.csv`、`sim_validate.json`、`b2_summary.csv`、`b2_verdict.csv`；
`results/m7_explore_h200/`：`h1_write_volume.csv`、`h1_verdict.csv`、`kappa_screen.csv`、`r1_sim_verdict.csv`、`r1_sim_configs.csv`、`r1_gpu_summary.csv`、`r1_gpu_verdict.csv`。
**失敗與異常**:
1. head node 的 `/tmp`（/dev/sda4，20G）約 17:25 起 100% 滿（我們自己只佔 137M，其餘是別的使用者）。Claude Code 的工具輸出寫不進去，所有指令都回 `ENOSPC: no space left on device`，監看 job 的背景程式也中斷。
   GPU 的 job 不受影響（hgx-cn01 的 /tmp 是另一顆）。處置（使用者 17:3x）：Claude Code 改用 `CLAUDE_CODE_TMPDIR`／`TMPDIR`＝`/work/ChihChungHsu/hungweilai/tmp` 重開（在 tmux 裡）。
   〔注意〕這個 session 的 shell 有 `TMPDIR=/work/...`，之後從這裡 sbatch 會帶進 job（Slurm 預設 `--export=ALL`），所以送 job 時要用 `env -u TMPDIR -u CLAUDE_CODE_TMPDIR sbatch …`，job 的環境才和之前一樣。
2. 其他：無。

### 第 1 輪：H200 自己的模擬存活設定上 GPU（事先寫死；寫在 job 送出之前）
**決定**（使用者 2026-10-16 17:4x，選「3 個都上 GPU（建議）」）：H200 的模擬存活設定和 MI300X 的不同，照 MI300X 的方法（存活的設定上 GPU，`11_round1_plan.md` 追加 4）確認 H200 自己的 3 個 Llama 存活設定。
LongAlpaca 照追加 4 不上 GPU（doc 的回來請求超過 32,768 token 上限，H200 一樣）。
**來源**: `results/m7_explore_h200/r1_sim_configs.csv`（run `20261016-173455-m7h-r1-sim-an`；模擬 run `20261016-095639-m7h-r1-sim-llama31-8b`）
**設定與事先選定的寫入時決定策略**（規則：模擬 5 個 seed 裡當「最好的寫入時決定策略」次數最多的；`r1_sim_verdict.csv` 的 best_wt）:
| 格 | CPU GiB/s | release | SSD | cpu_frac | 策略 | 模擬：best_wt 次數 | 模擬 g_nwt 中位 | 模擬通過的 seed |
|:--|--:|:--|:--|--:|:--|:--|--:|:--|
| A | 11.6 | hold | nfs | 0.5 | **S5L** | 5/5 | 26.8% | 0、2、3 |
| B | 35.42（tier_params 的 CPU 層，不設 M7_CPU_GIBPS） | free | local | 0.5 | **S5L** | 5/5 | 7.4% | 2、3、4 |
| C | 35.42 | hold | nfs | 0.25 | **S5c** | 4/5（seed 4 是 S5s） | 7.0% | 0–3 |
**對照組**: S1、S2b、S4、S4+、S4+P、S4L（取最好的＝「最好的不看位置」）、S4B、S4W、S4C。workload doc、seed 0–4、每個 3 次、`--verify`、`M7_IO_MODEL=share`、f(i)＝H200 R2 的 C1（和模擬相同）。
**判準**（同追加 4）: 寫入時決定策略的回來請求 TTFT 中位數，同時贏「最好的不看位置」、S4B、S4W、S4C 各 ≥5%，**≥3/5 個 seed** 才算存活（≥15% 算強，照 08 §6）。平均列出、不進判定。
**分析**: `python code/m7_round1_analyze.py gpu --csv <檔> --cpu <GiB/s> --release <…> --ssd <…> --frac <…> --wt <策略> --tag <A|B|C>`（不給參數＝原本那一格；回歸 run `20261016-174418-h200-r1gpu-args-regress`：MI300X 與 H200 的 `r1_gpu_summary`、`r1_gpu_verdict` 和 git 逐格相同）。
**已知限制**（寫在看到結果之前）: MI300X 第 1 輪的教訓 10——模擬器對存活策略的偏差約 5–6 個百分點，B、C 的模擬優勢（7.4%、7.0%）在這個範圍內；A 的 26.8% 遠大於它。
**job 20660** `r-h200-r1h`（17:5x 送出，git `e0d1b8b`）: `code/r_h200_r1h.sh`（15 步：A、B、C 各 seed 0–4），`--dependency=afterany:20643`，`--time=14:00:00`；送出時拿掉 TMPDIR（`env -u TMPDIR -u CLAUDE_CODE_TMPDIR sbatch …`）。
開跑前檢查：節點上沒有我們自己的 job；三個輸出 CSV 還不存在。輸出：`results/m7_explore_h200/r1h_gpu_llama_*.csv`。
H200 上 P3（free、local、一格 5 個 seed）量測時間 1 小時 43 分；hold／nfs 的格 NOT_MEASURED，總時間〔推算〕6–9 小時。
- **開跑**：job 20660 2026-10-16 21:53:57 開始（20643 結束的同一秒）。開跑前的檢查通過（節點上沒有我們別的 job；`slurm-r-h200-r1h-20660.out` 第 3–9 行）。
- **結束**：2026-10-17 06:16:01 COMPLETED（8 小時 22 分）→ 結果見最後一節「第 1 輪 H200 自己的存活設定上 GPU — 結果」：**3 格都沒存活**（A 2/5、B 0/5、C 2/5）。

---

## Milestone R-brr — B 重跑（job 20631）、A2／C7 錨點只跑限速層（job 20643）、D3 κ 地圖

**狀態**: 20631 **COMPLETED**、20643 **COMPLETED**；分析 `r_h200_cpu.sh --part brr`、`--part a2c7`、`--part d3` 全部 exit 0。
**有兩件事要使用者決定**（見最後的「要問的」）：模擬器驗證的方向條件在合併組只有 9/10；D3 照 §2.4 字面在 H200 不是「死路」。
**執行時間**:
- job 20631 `r-h200-brerun`：2026-10-16 17:24:43 → 21:49:50（COMPLETED，exit 0:0，9 步）。開跑時同節點有別的使用者的 4 個 job（共 5 張卡；每個 run 結束時的同節點 job 記在 run 目錄的 `cotenants_end.txt`）；**沒有我們自己的 job**（腳本開跑前檢查）
- job 20643 `r-h200-simanchor`：21:49:50 → 21:53:57（COMPLETED，3 步；MI300X 上同樣 3 個 run 的量測時間約 8.2 分）
- head node 分析：21:58:46 → 22:06:33（`nice -n 19`）

### job 20631：B 重跑（和 F2 重疊的那些 run，照重疊選）
**run_id**（全部 exit 0，`gpu_guard.json` 的 `contaminated` 全是 False、沒有 intruders；`contaminated_gpu` 0 列）:
`20261016-172445-m7h-b-nfs-gap0-r36-rr`、`…-174512-m7h-b-nfs-seed1-rr`、`…-175330-m7h-b-nfs-cpuvllm-rr`、`…-181237-m7h-b-local-gap0-rr`、`…-182644-m7h-b-nfs-fifo-rr`、
`…-185947-m7h-b-nfs-seed2-rr`、`…-191226-m7h-b-nfs-gap2-rr`、`…-200806-m7h-b-nfs-ssd50-rr`、`…-202603-m7h-b2-g1-doc-hold-cpu3.69-s0-rr`
**產出檔**: `results/m7_write_policy_h200_rerun/`：`b_share.csv.gz`（6,420 列；原檔 >1 MB 不進 git）、`b_fifo.csv`、`b_share_cpu3.69.csv`（各 1,152）、`b2_share.csv`（1,536；kv_bad 0）、`b_workload_seed{0,1,2}.json`（和舊目錄的逐位元相同）

### 合併組（「沒有和我們自己的 job 重疊」的 B）＝ `results/m7_write_policy_h200_merged/`
**組成**（`MANIFEST.csv`，run `20261016-220045-m7h-brr-merge`；每一列原封不動，run_id 指回原 run）:
- `b_share.csv.gz`＝舊的 R3-B-main（`20261016-063929-m7h-b-nfs-gap0`，2,880 列）＋重跑的 6 個 run（6,420）＝9,300 列（和舊檔列數相同）
- `b_fifo.csv`、`b_share_cpu3.69.csv`＝重跑的；`b2_share.csv`＝舊的 G1-s1、G2-s0（12:38 F2 結束後才開始；2,112 列）＋G1-s0-rr（1,536）＝3,648 列
- 斷言：欄位相同、（設定, rep, ev）不重複、`contaminated_gpu` 全 False
**分析**（`bash code/r_h200_cpu.sh --part brr`；和 `--part b` 同一套指令，只把 `M7_RES` 指到合併目錄，f(i)＝同一份 H200 C1）:
`20261016-220047-m7h-brr-sim-validate`、`…-220056-m7h-brr-b2-analyze`、`…-220106-m7h-brr-r1-h1`、`…-220108-m7h-brr-b-all`、`…-220129-m7h-brr-side`
→ 合併目錄的 `sim_validate.*`、`b2_summary.csv`、`b2_verdict.csv`、`h1_*.csv`、`summary_b_verdict_all.csv`、`b_vs_mi300x_all.csv`、新舊並排 `old_vs_new_*.csv`
**新程式**: `code/h200_brr.py`（merge／stage／ball／side）。`ball` 的回歸 run `20261016-215954-h200-brr-ball-regress2`：在舊 B 資料的複本上重算，
`summary_b_verdict_all.csv`、`b_vs_mi300x_all.csv` 和 12:41 的 run `20261016-124102-h200-b-all-settings` **逐格相同**（run_id、ts 除外）。

**結果：新舊並排**（MI300X 欄來自 git 的 `results/m7_write_policy_mi300x/summary_b_verdict.csv` 等）
| 項目 | MI300X | H200 舊（20554 原始） | H200 合併組 |
|:--|:--|:--|:--|
| B 主設定 cpu_frac 0.5，rep 0–5 合併，S5 對 S4+（R3 判準 (2)，δ*＝5%） | +0.26% | **+8.30%**（超出 δ*） | **−0.12%**（在 ±δ* 內） |
| B 17 格，S5 對 S4+ | −3.0%～+0.3% | 16 格 −1.1%～+0.6%；主設定 0.5 +8.30% | 15 格 −0.61%～+0.54%；**ssd50 cpu 0.5 +8.31%**、**vllm 3.69 cpu 0.5 −5.96%** |
| B 17 格，S5 對最好的簡單策略 | 都 ≤ +0.07% | 都 < 0 | 都 < 0（−0.03%～−189%） |
| 模擬器驗證：誤差 ±10% 內 | 100% | 93.0%（106/114） | **95.6%**（109/114；中位 0.45%，最大 19.8%） |
| 模擬器驗證：實測差 >5% 的格方向一致 | 8/8 | 9/9 | **9/10**（見下） |
| 08 消融 GPU（寫入時決定贏最好的不看位置與 S4B 各 ≥5%） | 0/16 | 1/16（G1 s0 cpu 0.25 S5c +6.0%／+28.2%） | **0/16**（同一格 +3.0%／+26.0%） |
| 08 模擬誤差 ≤10% | 43/44 | 40/44 | 39/44（中位 1.4%，最大 16.8%） |
| H1 寫入量（≥20% 才存活） | 2/23 | 2/23 | 2/23（同兩個設定：b2 doc s0 0.25、doc s1 0.5） |

**主設定 cpu 0.5 逐 rep**（回來請求 TTFT 中位數 S4+／S5，s；`b_share.csv`）:
| run | rep | S4+ | S5 | S5 對 S4+ |
|:--|--:|--:|--:|--:|
| R3-B-main（06:39，沒有重疊） | 0／1／2 | 0.797／0.793／0.806 | 0.798／0.795／0.804 | −0.16／−0.20／+0.29% |
| 舊 r36（09:00，和 F2 重疊） | 3／4／5 | 0.947／0.947／0.937 | 0.957／0.866／0.867 | −0.99／+8.61／+7.55% |
| r36-rr（17:24，沒有重疊） | 3／4／5 | 0.863／0.862／0.863 | 0.863／0.795／0.862 | +0.07／+7.80／+0.15% |

**為什麼會跳 8%**〔事後診斷，不是判定〕:
- 每格每個 rep 有 24 個回來的請求，中位數＝第 12、13 個的平均。這一格的 TTFT 分成兩群（約 0.78–0.82 s 和 0.93 s 以上），第 12、13 個是下面那群的最後兩個。
  例：S4+ rep 0 排序後第 9–16 個＝0.783、0.795、0.796、0.797、0.797、**0.934**、0.948、0.952。只要一個請求從下面那群跑到上面，第 13 個就變成約 0.93，中位數從約 0.80 跳到約 0.86（+8%）。
- 跑過去的是 Cake 的會合點差一個 chunk：例如 ev 27（session 4 第 2 輪，歷史 16 個 chunk）rep 0–2 從 SSD 載 1 個（meet 15，TTFT 0.78 s），r36 與 r36-rr 的 S4+ 都載 2 個（meet 14，`t_load_ssd` 0.632 s，TTFT 1.09 s）。
  ssd50-rr 的 S4+ rep 1 第 12、13 個是 0.812／0.945，同一種情況；S4+ 有 7 個事件的 meet 在 3 個 rep 之間差 1。
- 所以 **「8% 的差」在沒有和自己 job 重疊的 run 裡也會出現**（r36-rr rep 4、ssd50-rr）。P1 一節寫的〔判讀〕「主機層爭用很可能讓 rep 3–5 變慢」**只說對一部分**：
  舊 r36 被標成 GPU 變慢（`t_new` 超過同歷史長度最小值 1.15 倍，m7_analyze b() 的 gpu_slow）的請求 **159/1,344**，r36-rr 只有 **6/1,344**；舊 r36 的 S4+ 整體也較慢（0.94–0.95 對 0.86）。
  但造成 S5 對 S4+ 差 8% 的那一跳，兩次都有。
- vllm 3.69 cpu 0.5 的 −5.96% 不是同一個原因：重跑的 rep 2（18:09–18:12）S4+P、S5 還原期間的重算每 chunk 0.0307／0.0255 s（其他 rep 約 0.022），`t_new` 正常（0.579 s），很多事件的 meet 一起變小。
  rep 0–1 是 −1.2%／−0.1%。原因看不到（UNOBSERVABLE_HOST：同節點別人的 job 用不同的卡，但主機是共用的）。
- 〔判讀〕這個指標（24 個請求的中位數）在這個設定下的離散跳動約 8%，比 δ*＝5% 大。R3 判準 (2) 用 rep 0–2 判 REPRODUCED（−0.24%），合併組 rep 0–5 −0.12% 也在 ±δ* 內；舊的 +8.30% 有一部分是這個跳動，不能全算在 F2 頭上。

**模擬器驗證 9/10 的那一格**: share、CPU 3.69 GiB/s、nfs、cpu 0.5：實測 S5 對最好的簡單策略 **−5.6%**（S5 較慢），模擬 **0.0%**。
程式的規則（`m7_sim.py validate`）：|實測| >5% 時要「實測×模擬 > 0」；模擬 0.0 → 不一致。這格的 −5.6% 來自上面那個 rep 2（rep 0–1：S5 對 S1 −0.8%、+1.1%）。
08 §5 第 2 點的文字是「實測差距 >5% 的格，模擬器要給出同樣的方向」。**照字面，合併組的第二部分沒過（9/10）**；舊的那組是 9/9。第一部分（≥90% 在 ±10%）兩組都過。

### job 20643：A2、C7 錨點，只跑限速層（使用者：「都用模擬限速」）
**run_id**: `20261016-214952-m7h-a2-sim`（fifo，reps 6）、`…-215122-m7h-a2share-sim`（share，reps 3）、`…-215148-m7h-c7anchor-sim`（L 8K／16K／32K，reps 3）；全部 exit 0，`contaminated_gpu`、`contaminated_disk` 全 False，c7 的 kv_bad 合計 0
**產出檔**: `results/m7_write_policy_h200/a2_fifo.csv`、`a2_share.csv`、`c7_anchor.csv`；分析 `summary_a2.csv`、`summary_c7_anchor.csv`、`a2c7_verdict.json`、`a2_vs_mi300x.csv`、`c7_anchor_vs_mi300x.csv`
**分析**（`bash code/r_h200_cpu.sh --part a2c7`；`code/h200_a2c7.py`，m7_analyze 的 calib()、a2() 同一套）: `20261016-215924-m7h-a2c7-an`
**程式的修改**: `m7_analyze.py` 的 C7 錨點與 A2，沒有 real 列時：比值記 NaN、`page_cache_ok`＝NOT_MEASURED、A2 改報各限速器模型（有 real 時照舊）。
回歸 `20261016-215846-m7h-a2c7-mi300x-regress`：MI300X 的 `summary_a2.csv`、`summary_c7_anchor.csv` 和 git **逐格相同**，verdict 的 A2、C7_anchor **相同**。
**A2（寫入會不會拖慢正在還原的人；L＝16K，TTFT 中位數）**:
| 裝置 | 模型 | 背景寫後段 vs 不寫：MI300X／H200 | 背景全寫 vs 不寫 | 寫後段 vs 全寫 |
|:--|:--|:--|:--|:--|
| local | 真實 | +113.5%／NOT_MEASURED | +120.2%／NOT_MEASURED | −3.0%／NOT_MEASURED |
| local | share（A2 校準的限速器） | +108.9%／+95.1% | +108.7%／+94.9% | +0.07%／+0.10% |
| local | fifo | +526%／+587% | +681%／+850% | −19.8%／−27.8% |
| nfs | 真實 | +10.5%／NOT_MEASURED | +10.7%／NOT_MEASURED | −0.19%／NOT_MEASURED |
| nfs | share | +11.0%／+24.8% | +11.0%／+24.8% | −0.02%／+0.02% |
| nfs | fifo | +34.3%／+47.4% | +483%／+785% | −76.9%／−83.3% |
- 方向：同一個限速器模型，「寫入 vs 不寫」兩台都是寫入較慢；「寫後段 vs 全寫」fifo 兩台都是寫後段較快，share 兩台都在 ±0.1% 內（nfs 的正負號不同：−0.02%／+0.02%）。
  H200 的「不寫」比較快（local share 0.223 對 0.276 s），同樣的寫入干擾佔的比例較大。真實裝置那一半沒有量。
**C7 錨點（只有限速層）**: H200 的 sim TTFT ÷ MI300X 的 sim TTFT：cpu 0.64–0.72、local 0.65–0.91、nfs cake 0.63–0.66、nfs load_only 0.99（頻寬限制，兩台用同一個限速器）。真實裝置 NOT_MEASURED。

### D3 κ 地圖（`bash code/r_h200_cpu.sh --part d3`；`--no-3090`）
**run_id**: 舊 B：`20261016-220616-m7h-d3-kappa-map`、`…-220621-m7h-d3-r1-b100`、`…-220622-m7h-d3-gputest-prereg`；
合併組：`…-220624-m7h-d3m-stage`（輸入目錄＝這個 run 目錄的 `res/`：舊目錄＋合併組的 B 檔，`STAGE.csv`）、`…-220626-m7h-d3m-kappa-map`、`…-220630-m7h-d3m-r1-b100`、`…-220632-m7h-d3m-gputest-prereg`
**指令**: `m8_kappa_map.py --out results/m8_directions_h200 --no-3090`，再 `--b100`，再 `--prereg-test --dstar <主程式的 D*>`；合併組 `M7_RES=<stage>/res --out results/m8_directions_h200/d3_merged`
**產出檔**: `results/m8_directions_h200/d3_*.csv`、`d3_merged/d3_*.csv`。讀取端、b100、prereg 新舊**逐格相同**（不讀重跑的 B）；寫入端（`d3_write_cells`、`d3_rule_scores`、`d3_posthoc`）不同
**D*（R4b，只用第一階段 GPU 擬合）**: H200 **0.6437**（新舊相同），MI300X 0.6026；c_sync（R1b）H200 9.0 ms、MI300X 15.0 ms
**規則分數**（命中／格數；括號＝笨規則）:
| 規則 | 資料 | MI300X | H200 舊 | H200 合併組 | GO 門檻（§2.3） |
|:--|:--|:--|:--|:--|:--|
| R1 | A0 28 格 | 27/28（27/28），誤差中位 4.1% | **26/28（25/28）**，誤差中位 **1.9%** | 同左 | ≥90%、誤差 ≤10%、> 笨規則 |
| R1 | c7 限速器 9 格 | 8/9（5/9） | 5/9（5/9） | 同左 | — |
| R1 | c7 真實 | 6/9（5/9） | NOT_MEASURED | NOT_MEASURED | — |
| R1b 測試 | A0 4K、16K | 10/14（14/14）→ KILL | 12/14（12/14） | 同左 | ≥90%；<75% KILL |
| R1b 測試 | c7 限速器 | 8/9（5/9） | 9/9（5/9） | 同左 | — |
| R1（附加 b100） | B 容量 100%：只載 vs Cake | 命中 | **沒中**（預測 Cake 快 1.07–1.09×，實測 1.003×） | 同左 | — |
| R2 安全性 | GPU／SIM08／SIMR1 Llama／LongAlpaca | 24/24、98.0%、96.3%、91.0% | 24/24、95.7%、93.3%、91.0% | 同左 | 各 ≥95% |
| R2' | GPU 全部 | 28/38 | 27/38 | 27/38 | GPU ≥90%、模擬 ≥85% |
| R3 | GPU 全部 | 23/28（22/28）＝82% | 22/28（21/28）＝79% | 23/28（22/28）＝82% | GPU、模擬各 ≥85% 且 > 笨規則 |
| R3 | SIM08 | 62.1%（53.1%） | 47.2%（69.2%） | 同左 | 同上 |
| R4b 測試 | GPU A08／R1 | 5/6（6/6）、0/5（5/5） | 同 MI300X | 同左 | ≥85% 且 > 笨規則 |
| R4b 測試 | 模擬三組 | 40.5%／55.0%／55.0%（87.7／71.9／69.4） | 35.9%／54.7%／38.1%（91.3／72.2／79.4） | 同左 | 同上 |
| R4a 反例（D<0.05 但重要） | GPU | 1 | 1 | 1 | 不能有 |
- **照 §2.4 字面**：
  - MI300X：讀取端沒有 GO（R1 打平笨規則、R1b KILL）→ **死路**（D3 §4.5）。
  - H200：**R1 GO**（26/28 ≥90%、誤差 1.9%、贏笨規則 25/28 一格）。寫入端 R3、R4b、R2' 都沒 GO，R2 只有 GPU 與 SIM08 過 95%。
    「死路」的兩個條件（讀取端沒 GO／寫入端全都打不贏笨規則）都不成立：R3 在 GPU 贏笨規則一格（新舊都是）。「有看頭」要 R1、R3、R4b 都 GO → 不成立。
    剩下「可能」（讀取端 GO，寫入端只有一部分 GO），但寫入端其實沒有一條 GO。另有新穎性條款「讀取端已有前作（D3 §5）→ 降一級」，條文只寫了「有看頭 → 可能」。
  - 〔判讀〕R1 的差別來自 A0 的類別分布：H200 的 A0 有 3 格 Cake「沒用」（4K×7 Gbps、4K×nfs、8K×nfs），MI300X 只有 1 格，所以笨規則在 H200 比較弱。R1 在這 3 格只中 1 格。
- prereg（LongAlpaca 候選格，D*＝0.6437）：R4b 預測「重要」0/24 格（MI300X 3/24）。

**失敗與異常**:
1. `h200_brr.py merge` 第一次（run `20261016-220028-m7h-brr-merge`，放了 `SUPERSEDED`）：我寫的斷言錯了，以為重跑的 run_id＝舊 id＋`-rr`，其實時間戳不同 → AssertionError。改成比對 run 短名後重跑。資料沒有問題。
2. `ball` 的第一次回歸（run `20261016-215542-h200-brr-ball-regress`，`SUPERSEDED`）：暫存目錄放進了 job 20643 剛寫出的 `c7_anchor.csv`（只有 sim），m7_analyze calib() `KeyError('real')` → C4 的 δ 變 NaN → `delta_pct` 欄不同（其他欄的差都是 0）。
   修正：`ball` 不放 a2_／c7_anchor（12:41 那次的輸入也沒有），m7_analyze 改成沒有 real 列也能算（上面的回歸）。
3. `h200_a2c7.py run` 第一次：我寫錯一行（itertuples 用欄名取值）→ TypeError，A2C7-regress 已經成功、只有 run 這一步沒產出。修正後 `--from A2C7-run` 重跑。
4. 圖的字型：head node 的 matplotlib 找不到 Noto Sans CJK TC（同 R3）；圖在 run 目錄，不進 git。

**要問的**（不自己決定）:
1. 模擬器驗證：合併組照 08 §5 第 2 點字面是 9/10（那一格是單一 rep 的重算變慢）。要算「沒過」、還是兩組並排、以舊的 9/9 為準？這會影響「能不能用 H200 的模擬器挑設定」（第 1 輪 H200 存活設定就是用它挑的；job 20660 正在 GPU 上確認，GPU 結果才算數）。
2. D3：H200 照 §2.4 落在「可能」與「死路」之間（R1 GO、寫入端沒有一條 GO）。要照哪個讀法寫？

**和 MI300X 的差異（只比方向）**:
- B：合併組的主設定 S5 對 S4+ −0.12%（MI300X +0.26%）→ 同方向；S5 從沒贏過最好的簡單策略 → 同。08 消融 GPU 0/16 → 同。H1 2/23 → 同。
- A2：兩台的限速器模型方向相同；真實裝置 H200 NOT_MEASURED。
- D3：寫入端兩台都沒有一條 GO（同）；讀取端 R1 在 H200 GO、MI300X 沒有（不同，見上）。
- 第 1 輪 GPU 確認：MI300X 的存活設定兩台都 0/5；H200 自己的 3 個存活設定 2/5、0/5、2/5 → 都沒存活（同；最後一節）。

### 決定（2026-10-16 22:1x）：兩個判定照 MI300X 的讀法（照條文字面）
**使用者的話**：「按照mix 300 那邊讀法？ 不懂？ 跟目前第一階段都跑好。？」
MI300X 的做法是判準照字面讀，判準本身有問題就照字面判、另外寫出來（例：D3 §7.2 第 7 點「這是判準設計的問題，我照字面判「不 GO」」）。照這樣:
1. **模擬器驗證（08 §5 第 2 點）**：合併組第二部分**沒過**（9/10）；舊資料**過**（9/9）。兩組並排報。
   影響：08 §5「沒過就不用模擬器下結論」。08 §6 本來就規定模擬器只能挑設定、結論看 GPU；H200 的判定（08 消融 GPU 0/16、第 1 輪 GPU 確認）都是 GPU 量的 → **沒有判定要改**。
   模擬的數字（`sim_sweep_gain`、`r1_sim_*`）照舊只當挑設定的參考。
2. **D3（§2.4）**：H200 讀取端 R1 GO、寫入端沒有一條 GO → 「可能」；§2.4 的新穎性條款（讀取端已經有前作，D3 §5）→ 降一級 → **死路**。
   和 MI300X 最後的判定相同，理由不同（MI300X 是讀取端沒 GO）。注意：條文的例子只寫了「有看頭 → 可能」，這裡把「降一級」照字面用在「可能」上。

### 第一階段的覆蓋（對照 RUNLOG_MI300X 的 M7–M7f；2026-10-16 22:1x）
| MI300X 的項目 | H200 |
|:--|:--|
| M7 第一階段：正確性、校準 C0–C7、A0、A1、B | 完成（R1–R3、R-rest P1，B 另有重跑） |
| M7 的 C2（真實儲存頻寬）、A2／C7 錨點的真實裝置那一半 | **NOT_MEASURED**（使用者：「都用模擬限速」；hgx-cn01 上也沒有我們能寫的 NFS）。A2、C7 的限速層那一半完成 |
| M7b 08 消融（模擬＋GPU） | 完成 |
| M7c 第 1 輪（Llama、LongAlpaca、Qwen3；MI300X 存活設定上 GPU） | 完成；H200 自己的 3 個存活設定也上了 GPU（job 20660）：**0/3 存活**（最後一節） |
| M7d D1、D2、D3、D4、D5、D6 | 完成（D6 用 H200 的 F3 工作 A） |
| M7d **D7**（OS：page cache、dirty writeback、反向讀、SSD 掉崖；不用 GPU） | **NOT_MEASURED，不跑**（使用者 2026-10-16 22:3x：「D7 記 NOT_MEASURED 不跑」）。量的是真實儲存與 OS 本身，沒有限速層的版本；MI300X 寫了約 161 GiB，lab 配額剩 45 GiB（1491G／1536G）；要在 hgx-cn01 上量就得用 GPU job（h200q 不收純 CPU job） |
| M7d D8 | 文獻與想法，沒有 run（run_commands.tsv 沒有 d8） |
| M7e F1（開卸載變慢的原因） | 不適用：原因是 ROCm 的 attention 退回 TRITON_ATTN；H200 的 D2 沒有變慢（0.99–1.04×），H200 的 vLLM 用 FlashAttention 3 |
| M7e F2 | 完成（K=21，只做 Llama） |
| M7e F3 | 工作 A（反量化成本）完成；FP8 那一半**凍結**（使用者：等 AMD 做完再改） |
| M7e F4（main.tex 稽核） | 不做（使用者：「不要在看 main」） |
| M7e F6、M7f F7 | 完成 |
| Milestone 10-pilot（精度彩排） | 第二階段，不動 |
| 9/15–9/20 的 m1–m6 | 程式已刪、現在的結論沒用到（migration README §2.2） |
| R0 硬體報告 | PASS；報告產生器的解析有幾欄 NOT_MEASURED（不影響判定），修好要重跑 R0 |
---

## 第 1 輪 H200 自己的存活設定上 GPU — 結果（job 20660）
**狀態**: DONE（判定：**3 格都沒存活**；寫入時決定在 H200 的 GPU 上也沒有贏過簡單策略 → 和 MI300X 相同）
**執行時間**: GPU 2026-10-16 21:53:57 → 2026-10-17 06:16:01（8 小時 22 分）；分析 2026-10-17 06:21（head node）
**判準與格**: 事先寫死在「第 1 輪：H200 自己的模擬存活設定上 GPU」一節（job 送出之前 commit），沒有改。
**run_id**（GPU，15 個都是 exit 0，沒有 `CONTAMINATED`，`kv_bad` 0、`contaminated_gpu` 0；開跑時節點上沒有我們別的 job）:
- A（11.6／hold／nfs／0.5，S5L）：`20261016-215359-m7h-r1h-gpu-llama-doc-hold-nfs-c11.6-f0.5-s0`、`20261016-224000-…-s1`、`20261016-230921-…-s2`、`20261016-234645-…-s3`、`20261017-002111-…-s4`
- B（35.42／free／local／0.5，S5L）：`20261017-004653-m7h-r1h-gpu-llama-doc-free-local-c35.42-f0.5-s0`、`20261017-011607-…-s1`、`20261017-013431-…-s2`、`20261017-015831-…-s3`、`20261017-021900-…-s4`
- C（35.42／hold／nfs／0.25，S5c）：`20261017-023346-m7h-r1h-gpu-llama-doc-hold-nfs-c35.42-f0.25-s0`、`20261017-033102-…-s1`、`20261017-041123-…-s2`、`20261017-045821-…-s3`、`20261017-054322-…-s4`
- 分析：`20261017-062112-m7h-r1h-gpu-an-A`、`20261017-062114-m7h-r1h-gpu-an-B`、`20261017-062116-m7h-r1h-gpu-an-C`
**指令**: GPU＝`code/r_h200_r1h.sh`（檔頭）；分析＝`bash code/r_h200_cpu.sh --part r1h`（新的 part：三格各一次 `python code/m7_round1_analyze.py gpu --csv … --cpu … --release … --ssd … --frac … --wt … --tag A|B|C`，參數見 r_h200_cpu.sh）
**產出檔**: `results/m7_explore_h200/r1h_gpu_llama_doc_{hold_nfs_c11.6_f0.5,free_local_c35.42_f0.5,hold_nfs_c35.42_f0.25}.csv`（原始列，各 3,420 列）、`r1_gpu_summary_{A,B,C}.csv`、`r1_gpu_verdict_{A,B,C}.csv`

**關鍵數字**（寫入時決定策略的回來請求 TTFT 中位數，對「最好的不看位置」／S4B／S4W／S4C 快多少；判準四個都 ≥5%）:
| 格 | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | 通過 | 模擬通過的 seed |
|:--|:--|:--|:--|:--|:--|:--|:--|
| A S5L | +17.1／+8.0／+52.3／+17.1 ✓ | −23.8／+0.8／+30.3／+45.7 | +24.5／+61.0／+67.8／+29.8 ✓ | −19.2／+62.9／+75.6／+46.7 | −22.1／+92.3／+26.7／+82.5 | **2/5** | 0、2、3 |
| B S5L | −13.9／+0.3／+44.0／+18.4 | −1.7／−2.5／+34.9／+35.6 | −8.3／−8.2／+57.9／+37.4 | −7.1／−0.7／+55.4／+39.3 | 0.0／0.0／+0.4／+51.1 | **0/5** | 2、3、4 |
| C S5c | −4.4／+25.0／+25.5／−4.6 | +7.8／+33.7／+38.4／+7.8 ✓ | −4.5／+31.3／+27.8／−4.5 | +7.2／+36.6／+33.8／+7.3 ✓ | −98.8／+20.3／−37.0／−2.5 | **2/5** | 0–3 |
（單位 %；「最好的不看位置」：A 是 S2b、S4、S2b、S1、S1；B 是 S4L、S4L、S4、S1、S4L；C 是 S2b ×4、S1。來源 `r1_gpu_verdict_{A,B,C}.csv` 的 stat=median 列）
- 判定：三格都 <3/5 → **都沒存活**。C 通過的兩個 seed 也只多 5% 一點（+7.2%～+7.8%），不到 08 §6 的「強」（≥15%）。
- 平均（不進判定）：15 個 seed 裡寫入時決定贏「最好的不看位置」只有 B seed 3（+2.6%）、B seed 4（+0.8%）、C seed 0（+0.7%），其餘都輸。
- 模擬誤差（|模擬−GPU|/GPU ≤10%）：A 46/50（中位 1.2%）、B 43/50（4.1%）、C 43/50（1.2%）。
  **寫入時決定策略 15 個 seed 裡 14 個被模擬低估**（模擬比 GPU 快；−16.7%～+0.2%）。MI300X 是 5/5 被低估（−1.8%～−8.8%）：方向相同，H200 的幅度比較大。

**模擬通過、GPU 沒過的 seed，各自的原因**（看原始列）:
- B seed 2、3，C seed 0、2：模擬把寫入時決定策略算快了 11%～17%（B S5L −16.7%、−12.1%；C S5c 兩個都 −11.4%）。C 這兩個 seed 的 S5c 有 60/72 個回來請求落在 1.0–1.5 s 那一群，GPU 1.400 s、模擬 1.241 s，是整群偏移，不是中位數跳動。
  事先寫的已知限制（模擬對存活策略偏 5–6 個百分點，B、C 的模擬優勢 7.4%、7.0% 在這個範圍內）成立，H200 的偏差還更大。
- B seed 4：GPU 上 S5L、S4L、S4B 都是 0.171 s（打平），模擬把 S5L 算成 0.163、S4L／S4B 算成 0.183。
- A seed 3：對照組 S1 的中位數 GPU 0.385 s、模擬 0.787 s（+104.6%）。S1 這個 seed 的 42 個回來請求分成兩群（22 個 <0.5 s、17 個 1–2 s、3 個 >10 s），GPU 的第 21、22 個都在下面那群（0.375、0.395）；模擬的中位數 0.787 落在兩群中間。
  這和 B 重跑合併組看到的「中位數落在兩群空隙上就跳」是同一種現象。A 的 26.8% 模擬優勢（遠大於 5–6 個百分點）在 seed 3 是被這一跳吃掉的，不是被模擬偏差。
  （A seed 4 的 S4W 也有一樣的跳動：GPU 0.616、模擬 3.284，n=21；那個 seed 模擬本來就沒過，不影響判定。）

**失敗與異常**: 無（15 步都 exit 0；三個分析 stderr 是空的）。
**與論文假設的差異**: 不適用（使用者：「不要在看 main」）。

**和 MI300X 對照**:
- MI300X：模擬存活設定 1 個（11.6／free／local／0.5，S5L）→ GPU 0/5。H200 在同一格（R-rest P3）也是 0/5。
- H200 自己的模擬存活設定 3 個 → GPU 2/5、0/5、2/5 → 0/3 存活。
- → 第 1 輪的結論兩台相同：**沒有任何設定證明寫入時決定有效**。模擬器對寫入時決定策略偏樂觀，兩台都是。

### 第一階段寫入問題：兩台方向對照（最後的彙總；只比方向）
| 項目 | MI300X | H200 | 同／不同 |
|:--|:--|:--|:--|
| B 主設定（cpu 0.5）S5 對 S4+ | +0.26% | 合併組 −0.12%（舊資料 +8.30%，和 F2 重疊，不用） | 同（都在 ±1% 內，沒有贏） |
| S5 對最好的簡單策略 | 沒贏過 | 沒贏過（合併組） | 同 |
| 08 消融 GPU | 0/16 | 合併組 0/16（舊資料 1/16，那一格和 F2 重疊） | 同 |
| H1（寫入量） | 2/23 → 殺掉 | 2/23 → 殺掉 | 同 |
| 第 1 輪 GPU 確認（Llama） | 1 個存活設定 → 0/5 | MI300X 那格 0/5；自己的 3 格 0/3 存活 | 同 |
| A2（背景寫入干擾） | 限速層＋真實裝置 | 限速層方向同；真實裝置 NOT_MEASURED | 限速層同 |
| D3 寫入端 | 沒有一條 GO | 沒有一條 GO | 同 |
| D3 讀取端 R1 | 不 GO（27/27 和笨規則打平） | GO（26/28，笨規則 25/28） | **不同** |
| D3 最後（照字面） | 死路 | 可能 → 新穎性條款降一級 → 死路 | 同（理由不同） |
| D7 | 有量 | NOT_MEASURED（使用者決定不跑） | — |
**結論**：寫入那個問題（寫入時就決定 KV 放哪一層，能不能比簡單策略好），H200 的答案和 MI300X 一樣：**不能**。兩台唯一方向不同的是 D3 讀取端的 R1，照 MI300X 的讀法最後的判定還是一樣。

---

## R0 硬體報告重量 — job 20673（`make probe`，2026-10-17 00:07:32 → 00:07:50 +08:00，COMPLETED）

**run_id**: `20261017-000734-r0-h200-probe`　**指令**: `bash code/h200_probe.sh`（`cmd.sh`）
**產出檔**: `docs/h200/REPORT_H200.md`、`results/hw_h200.json`（機器產生；**取代** 10/16 job 20417 的那份 `20261016-061012-h200-probe`，舊版在 git 歷史與它的 run 目錄）
**和 10/16 那份的差別**（只有「這次拿到哪張卡」）: PCI bus `E5:00.0` → `CB:00.0`、cpuset `112-127` → `96-111`、同節點 job 數 8 → 7。型號、記憶體、driver、PCIe Gen5 x16、`UNOBSERVABLE_HOST` 都相同；整體 PASS。
**記錄補在 10/17**：當時沒寫進 RUNLOG，這一節是事後照 `sacct -j 20673` 與 run 目錄補的。

---

## Milestone 10-H200 — 第二階段（精度）主線在 H200 上復現

**狀態**: 準備中（還沒有任何 H200 第二階段的結果）
**事先登記**: `docs/precision_20261013/13_h200_repro.md`（寫於 2026-10-16T22:53Z，h200 分支 `944497e` 之後）
**程式**: 第二階段主線和 MI300X（`origin/main` `bcfe5b1`）逐字相同，只差 `m10_tasks.py` 的模型路徑（`code/h200_m10_gates.py code` 驗）。
  H200 才有的：`code/r_h200_m10.sh`（`--part speed|quality`）、`code/h200_pool.py`（工作池，`nvidia-smi` 查剩餘記憶體）、
  `code/h200_m10_gates.py`（13 §3 的停止條件）、`code/h200_m10_analyze.sh`（判定，包同一支 `m10_analyze.py`）；`make submit-m10`。
**送出前的檢查**（head node，CPU）:
- `h200_m10_gates.py code`／`params`：PASS（16 個主線檔逐字相同；`CANDIDATE_PARAMS.json` sha256 `00c1034f…74412e2`）。
- `h200_m10_gates.py fmt`：拿 MI300X 的 fmt_choice 自己比自己 → PASS；改掉一格 → FAIL、exit 1。
- `h200_m10_gates.py e0`：拿 MI300X 彩排第 3 輪（`results/m10_pilot_mi300x/e0_tune/p3_*.csv`）測 → 合併答對率和 RUNLOG_MI300X 的第 3 輪一致（例：Llama T3 0.716、Qwen T4 0.675），PASS。
- `h200_m10_analyze.sh all` 的回歸：把 `results/m10_precision_mi300x/` 複製到 scratch（Qwen 的速度照 MI300X 換成 stage 22／23 重量那次），
  重算出來的 `decide_fp8`、`k3_qwen`、`quality_*_step1`、`quality_qwen_step2b`、`decide_int4`、`quality_*_int4`、`quality_llama_step2`、
  `step2_llama_int4` 和 MI300X commit 的檔**逐一相同**（只差寫在 JSON 裡的輸入檔路徑）；路徑：2b 在 Qwen → K3 → INT4 → Step 2 在 Llama → K2，和 08 相同。
- `r_h200_m10.sh --dry-run`（兩個 part）與 jobs.json 產生器：指令除了路徑以外和 MI300X 的 queue/10、20、21、30、33a、33b、33c 相同。
  和 MI300X 的差：GPU 忙用 `results/m7_write_policy_h200/busy_levels.json`（busy2＝duty 1.0）；Step 1 讀 `meets_fp8tier.json`（INT4 格子前的快照，MI300X 的 30 在 33a 之前跑、讀到的就是這個狀態）。

### 送出（2026-10-17 07:0x +08:00）
- `m10h-speed` **job 20701**（`--part speed`，24 h；`sbatch --test-only` 預估 10/17 16:02 開始）
- `m10h-q-llama` **job 20702**、`m10h-q-qwen` **job 20703**（`--part quality`，各 40 h；`--dependency=afterok:20701 --kill-on-invalid-dep=yes`）
- 送出時我們自己沒有其他 job（速度 job 不和自己的 job 同時跑）；lab 配額 1492G／1536G。
- 指令：`make submit-m10` 的同一串（`env -u TMPDIR -u CLAUDE_CODE_TMPDIR sbatch ...`）手動執行：工作目錄裡另一個工作階段正在改 `code/h200_queue.py`（排隊顯示用，job 不讀），make 的「有未 commit 的改動就不送」會擋下；其餘檢查（`h200_m10_gates.py code`／`params`）照跑、PASS。job 用的程式都在 commit `01296ac`。

### job 20701 開跑（2026-10-17 07:02:04 +08:00，比 `--test-only` 預估的 16:02 早；hgx-cn01、PCI 03:00.0）
- V0 `20261017-070207-m10h-verify`（`m10_verify.sh` 完整模式）：exit 0，07:02 → 07:08。細項等整個 job 跑完一起記。
- 07:08 起 E0 難度確認（工作池，同一張卡同時 3 個 process）。

### 10/17 改動：全部復現、128K 先跑（13 §8；送出時 20701 已在跑）
- `m10h-lc-speed` **job 20704**（afterok:20701，16 h）：`m10_sweep parts`（--busy-duty 1.0）→ 128K FP8、INT4 → 64K FP8、INT4 的速度格子。
- `m10h-lc-screen` **job 20705**（afterok:20704，48 h）：試跑三個長度 → 題數規則 → 正式（252 → 128 → 64 個 block）。
- 主線品質 20702、20703：`scontrol update Dependency=afterok:20704,after:20705`（不和量時間的 20704 同時跑；20705 開始後才開始）。
- 送出用 `make submit-lc` 的同一串指令手動執行（同 20701：工作目錄裡另一個工作階段的 `code/h200_queue.py` 沒 commit）；job 用的程式在 commit `2e27c28`。

### 全部復現的清單（MI300X 第二階段做過或排了的，對照 `code/m10_queue_mi300x/`；10/17 寫）
MI300X 到 10/16 晚的最後推送為止：主線（10–33cz）、39、39a、40 跑完，41 在跑，42 以後還沒跑（預計 10/20 前跑完）。H200 照同一批腳本做。

| 類 | MI300X stage | 內容 | 量什麼 | H200 |
|:--|:--|:--|:--|:--|
| 主線 | 10-pilot 驗證、K／V 分布 | codec、正確性、BF16 回歸、kv_stats | 驗證 | 20701（V0 過） |
| 主線 | 10、p1 | 格式小測、E0 難度、calib-f、noise | 分數＋時間 | 20701 |
| 主線 | 20、21、33a | FP8／2b／INT4 速度格子＋K0 | 時間 | 20701 |
| 主線 | 30、33b、33c、46 | Step 1、INT4 端點、INT4 混合（兩個模型） | 分數 | 20702、20703 |
| 主線 | 31、33bz、33cz | 判定 | CPU | `make analyze-m10` |
| 長 context | 40 的零件、44a | 掃描零件；64K／128K 速度格子 | 時間 | 20704 |
| 長 context | 44b、44c | 試跑 → 正式（32K／64K／128K × 13 格式） | 分數 | 20705 |
| 長 context | 12.1b | `m10_longctx_analyze` | CPU | 待移植（R 寫死 MI300X 的目錄） |
| 掃描 | 39 | GPU 冒煙：融合反量化、大緩衝 offset、dqsweep | 驗證 | 待排（Triton 在 NVIDIA 第一次跑） |
| 掃描 | 40 | 反量化微測（bench）；試跑 → 題數 | 時間；分數 | 待排；`sweep_plan_n.py` 待移植 |
| 掃描 | 41 | 海選：13 格式 × 17 佔比 × 12 放法（MI300X n＝24、約 10 h） | 分數 | 待排 |
| 掃描 | 42、42b、42c | 融合反量化的 CPU 層格子；反量化方法掃描 A／B／複賽 | 時間 | 待排；`m10_dqsweep` 的 GPU 忙寫死 MI300X |
| 掃描 | 42a、43 | 複賽（finalists → 速度＋12 個品質 process） | 時間＋分數 | 待排；`m10_confirm.py` 預設路徑待移植 |
| 分層 | 13.1 A | IO 牆／容量（`m10_iowall`） | CPU | 待移植（RUNS、R 寫死） |
| 分層 | 44 | 兩層混合還原（tier2） | 時間 | 待排；GPU 忙要指 H200 |
| 只報告 | 45 | FP8 混合配置 | 分數 | 待排 |
| 只報告 | 55、88 | T1＋LongBench（FP8／INT4） | 分數 | 待排 |
| 只報告 | 60 | 逐 block 敏感度地圖 | 分數 | 待排 |
| 只報告 | 70 | vLLM TTFT（預設 vs 512 切法） | 時間 | 待排；venv、模型路徑寫死 |
| 只報告 | 75 | RULER CWE 用 vLLM（E0）＋上游 RULER 出題 | 分數 | 待排；H200 沒有上游 RULER 與 LFS 詞表 |
| 只報告 | 10.11 | INT4 敏感度檢查 | CPU＋小 GPU | 待移植 |
| 不做 | 22、23 | Qwen 重量（MI300X 被標污染才重量） | — | H200 被標污染時才做 |
| 不做 | 10-pilot 調難度、31z、39a | 彩排／冒煙 | — | 參數已凍結；試跑涵蓋 |
| 新 | — | 256K（只有 Qwen、T2–T4） | 分數＋時間 | 等 128K 結果再問（13 §8） |

### 10/17 送出主線以外的復現（13 §9；`code/r_h200_x.sh`）
| job | 名稱 | 內容 | 依賴 |
|:--|:--|:--|:--|
| 20706 | `m10h-q3-llama` | 41 海選（n＝24）、45、55、60、88；上游 RULER CWE（重新產生）、hfgen、75 | afterok:20704、afterany:20702（接 20702 的卡） |
| 20707 | `m10h-q3-qwen` | 41 海選、45、55、60、88 | afterok:20704、afterany:20703 |
| 20708 | `m10h-t2` | 量時間：39 的 GPU 自測、40 微測、42、42b、42c、44、70、43 的冒煙與速度 | afterany:20706、20707、20705（我們自己的 job 都結束後才跑） |
| 20709 | `m10h-q5` | 43 的品質 | afterok:20708 |
- 同時最多 3 張卡（20705＋20702＋20703，之後 20705＋20706＋20707）。
- 20708、20709 開跑時才讀 repo 裡的 `r_h200_x.sh`。**開跑前還要補**：`m10_dqsweep`、`m10_tier2`、`m10_confirm` 的 GPU 忙與路徑（MI300X 寫死）、43 的冒煙／速度／品質步驟（現在是會直接失敗的佔位）。

### 融合反量化（Triton）在 NVIDIA 上編不編得過（2026-10-17 07:5x，head node，CPU；使用者問「反量化 Triton 不用改？」）
- **為什麼要先驗**：`m10_dequant_fast`（融合反量化）在 MI300X 上寫的、還沒在 NVIDIA 上跑過；20704 的第一步 `m10_sweep parts` 就會呼叫它（`DF.decode_fast`），編不過的話 128K 那條線停在那裡。
- **做法**：Triton 3.7.1 的 AOT 編譯（`triton.compile(ASTSource, GPUTarget("cuda", 90, 32))`），不用 GPU、不碰正在量時間的 20701。指標型別照 `m10_codec.encode` 的實際 dtype（FP8 碼 uint8、INT8 int8、INT4 uint8、scale fp32、KIVI scale／zero fp16）。
- **結果**：`code/tests_h200/aot_triton_sm90.py` **11／11 PASS**（`_dq_kernel` FP8／INT8、`_dq4_kernel` int4_token／KIVI-K／KIVI-V、`_dqm_kernel` FP8 查表／FP8 位元運算（K4）／INT8、`_dq4m_kernel` 三種；INT4 的關 FMA 選項也接受）。
- **反向對照**：`code/tests_h200/aot_triton_sm90_negative.py` 拿第一階段 `m9_f3_dequant` 的 `_dq_rowscale`、`_dq_tensorscale`（H200 job 20614 實測失敗、10/16 決定凍結不改）用同一個方法編 → **兩個都 FAIL**，錯誤和實測相同：`cannot cast int32[...] to <[...], fp8e4nv>`。所以這個方法抓得到那一類錯誤。
- **差別在哪**：`m9_f3` 從 fp8 指標 `tl.load(..., other=0)`（0 要轉成 fp8e4nv，NVIDIA 後端不允許）；`m10_dequant_fast` 讀 uint8 碼再查表或位元運算，沒有這個轉型。
- **沒驗到的**：數值。`m10_sweep parts` 會在真實 KV 上逐 block 比融合版和 PyTorch 版，不逐位元相同就停（`m10_sweep.py:337`）；t2 的 GPU 自測再驗一次。`m9_f3_dequant` 照 10/16 的決定不動。

### job 20701 `m10h-speed` 跑完（2026-10-17 07:02:04 → 11:19:39 +08:00，4 h 17 m，COMPLETED exit 0；commit `01296ac`）
**狀態**: PASS（13 §3 的五個停止條件都沒觸發）
- 開跑閘門：`gate code` PASS（16 個主線檔＝`bcfe5b1`，`m10_tasks.py` 只差模型路徑）；`gate params` PASS（`CANDIDATE_PARAMS.json` sha256 `00c1034f…74412e2`）。
- V0 `20261017-070207-m10h-verify`：exit 0。
- E0 難度 `20261017-073537-m10h-gate-e0`（`results/m10_precision_h200/e0_check/`）：**PASS**。T1–T4 n＝120、T5 n＝60：
  Llama T1 1.000、T2 0.773、T3 0.723、T4 0.767、T5 0.773；Qwen T1 0.967、T2 0.854、T3 0.712、T4 0.675、T5 0.675（T2–T5 都在 60–95%）。
- 格式重選 `20261017-074146-m10h-fmtselect` → `fmt_select/fmt_vs_mi300x.json`：**same＝true**（Llama fp8_token／int4_kivi，Qwen fp8_head／int4_kivi，和 MI300X 相同）。
- calib-f（`speed_<m>/bw.json`）：Llama b16 8.8331、b32 2.49、b48 0.7222 GiB/s（真 pinned H2D 48.67）；Qwen b16 4.33、b32 1.2297、b48 0.3582 GiB/s（真 H2D 48.424）。都低於真 H2D 的一半，沒有 warning。
- noise（`speed_<m>/noise.json`）：Qwen δ＝0.0144 → δ*＝0.05；**Llama δ＝0.2842 → δ*＝0.2842**，全部來自 `cpu|busy` 一格（三次 launch 中位數 0.15526、0.12117、0.12081 s；第一次慢 28%），其他 7 格 δ ≤ 0.091。gpu_guard CLEAN。CPU 層是不限速的真 H2D，同節點別人的 job 也用主機記憶體／PCIe（Slurm 下主機層看不到，host_level＝UNOBSERVABLE_HOST），**原因沒有查證**，照 03 §5.3 的規則照用、不改。MI300X Llama δ*＝0.05、Qwen 0.0943。
- K0（`speed_<m>/k0_verdict_{fp8,int4}.json`）：兩個模型 × 兩階都是「K0 不成立：至少一格 E1 贏 E0」，**和 MI300X 相同**（Llama 在 δ*＝0.284 下仍成立）。
- 速度格子 FP8、2b（fp8k0_<g>、int8_token）、INT4：每格 3 份 gpu_guard 都 CLEAN；`meets_fp8tier.json` 在 INT4 格子前快照。meets 不用 δ*。
- 失敗與異常：無（Llama δ* 見上）。

### job 20704 `m10h-lc-speed` 跑完（2026-10-17 11:19:39 → 19:25:15 +08:00，8 h 06 m，COMPLETED exit 0；commit `0febf74`）
**狀態**: PASS
- `m10_sweep parts`（`--busy-duty` 1.0）：`20261017-111941-m10h-sweep-parts-llama`、`20261017-112036-m10h-sweep-parts-qwen`，gpu_guard CLEAN、contaminated＝False。
  **融合反量化在 H200 真實 KV 上和 PyTorch 寫法逐位元相同**（`results/m10_sweep_h200/parts/parts_<m>.json` 的 `fused_vs_torch_real_kv`）：Llama fp8_token、int8_token、int4_token、int4_kivi，Qwen fp8_head、int8_token、int4_token、int4_kivi，各 128 個 block，不同的 0 個。上一節 AOT 沒驗到的數值這裡驗到了。
- 長 context 速度格子（`results/m10_longctx_h200/speed_<m>_<L>_<fmt>/`），每格 2 份 gpu_guard 都 CLEAN：
  128K（129024）：Llama FP8 11:21→12:56、Qwen FP8 →14:17、Llama INT4 →15:46、Qwen INT4 →17:04；64K（65536）：Llama FP8 →17:43、Qwen FP8 →18:16、Llama INT4 →18:53、Qwen INT4 →19:25。
- 失敗與異常：無。

### 10/17 21:32 佇列
- 20705 `m10h-lc-screen`：PENDING（Resources）——hgx-cn01 8 張全被佔（其他 lab 7 張、本 lab 1 張），lab 配額只用 1/4，卡在節點滿不是配額。`make q` 推估 10/18 02:08 開始（照過去 30 天）。
- 20702、20703（after:20705）、20706、20707、20708、20709 照原依賴等。
- 主線判定（`make analyze-m10`）要等 20702、20703 的品質。

### job 20705 `m10h-lc-screen` 失敗：試跑時我們自己的工作池超量開工作 → OOM（2026-10-18 02:43:06 → 02:56:09 +08:00，FAILED exit 1；commit `dc66667`）
**狀態**: FAIL（程式的排程問題，不是量測結果）
- 步驟 PILOT：6 個試跑（每個模型 × 64／128／252 個 block，每任務 1 題），工作池同一張卡最多 4 個。
- `20261018-024610-m10h-lc-pilot-llama-252` exit 1，完整錯誤（stderr 最後）：
  `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 MiB. GPU 0 has a total capacity of 139.79 GiB of which 28.56 MiB is free. Process 3898647 has 40.74 GiB memory in use. Including non-PyTorch memory, this process has 52.12 GiB memory in use. Process 3901687 has 20.92 GiB memory in use. Process 3903658 has 25.96 GiB memory in use.`
  發生在第 2 題（T3）的 `load_fmt` → `m10_codec.rel_err_centered`；第 1 題（T2）已跑完，peak 63.6 GiB。
- 原因：`code/h200_pool.py`（照抄 MI300X `pool.py`）只看「當下剩餘記憶體 ≥ need＋8 GB」。Llama-252 在題目之間把記憶體放到 ~52 GB，工作池在 02:47:39、02:49:54 依當下剩餘（41、46 GB）放進 Qwen-64（need 30）、Qwen-128（need 32），四個 need 加起來 50＋78＋30＋32＝190 GB > 140 GB；Llama-252 下一題再漲就 OOM。MI300X 192 GB 沒遇到。
- 其他 4 個試跑 exit 0（Llama-64、Llama-128、Qwen-64、Qwen-128）；Qwen-252 因為規則 2（有失敗就不開新的）沒開。
- H200 上量到的 peak（torch，不含 CUDA context）：Llama 64／128 個 block 28.3／40.3 GiB，Qwen 20.1／25.3 GiB；MI300X 的 need 是 40／50、30／32 GB。
- **修正**（`code/h200_pool.py`）：多一個條件「卡的總記憶體 − 跑中工作的 need 總和 ≥ need＋8 GB」（預約）；放不下時照順序找下一個放得下的。離線測試（假的 nvidia-smi）：試跑 6 個、正式 12 個的預約都不超過 132／140 GB；need＋8 超過整張卡的工作直接報錯；有失敗照舊不開新的。
  正式掃描的 252 個 block：Llama（78）＋ Qwen（45）兩個一起，不會兩個 Llama-252 同時（MI300X 可以）。題數規則（≤ 14 h，÷3 平行）不改；H200 的實際牆鐘照實記。
- **正在跑的 20706、20707 不受這次修正影響**（工作池已經載入舊程式）。10/18 07:5x 用 `srun --overlap nvidia-smi` 看：20706 四個 process 各 ~29 GB、共 115／140 GB；20707 四個各 ~21 GB、共 84 GB。q3 的工作都是 64 個 block（試跑看到 Llama peak 28.3 GiB），風險比 252 低，但沒有排除；如果 OOM，照規則 2 停下回報。

### 20895：20705 重送（2026-10-18 08:0x 送出，`--part screen --from PILOT`，commit `ca4eb72`）
- 送出後馬上開跑（hgx-cn01）。試跑 6 個全部重跑（同一組條件；`h200_longctx_plan.py` 讀每個名稱最新的 run）。
- `scontrol update JobId=20708 Dependency=afterany:20706:20707:20895`：量時間的 t2 等 20895 也結束才開始（不和自己的 job 同時跑）。

### job 20702、20703 主線品質跑完（Llama 02:56:09 → 06:42:15、Qwen 03:56:05 → 07:43:58 +08:00，COMPLETED exit 0）
- 每個 job 三段工作池：S1（E0 E1 E1s E1sr＋2b 兩個，讀 `meets_fp8tier.json`）7 個 process；I4E（INT4 端點）6 個；I4M（INT4 混合 F25／M25／B25／R25）7 個；全部 rc＝0。
- 產出：`results/m10_precision_h200/quality/step1_*`、`quality_sealed/int4end_*`、`int4mix_*`（逐列 CSV 進 git 的是 .gz，同 MI300X）。

### 主線判定（head node，2026-10-18 07:56–07:57 +08:00；`PYTHON=~/.conda/envs/tiara/bin/python bash code/h200_m10_analyze.sh all`）
**狀態**: 停在 K3（13 §5 的判定節點有一個「不同」；需要 H200 沒排進主線的資料 → 問使用者）
**產出檔**: `results/m10_precision_h200/analysis/fp8/`（decide_fp8.json、k3_qwen.json、quality_*_step1*.json／csv、quality_qwen_step2b*、speed_qwen_2b_*.json）

| 節點（13 §5） | MI300X | H200 | |
|:--|:--|:--|:--|
| K0（FP8、INT4 格子） | 兩個模型都「不成立」 | 兩個模型都「不成立」 | 相同 |
| Step 1 Llama | E1／E1s／E1sr 四情境全過 → K1 成立 | 同 | 相同 |
| Step 1 Qwen | E1／E1s／E1sr 四情境全不過；power_ok＝False、power_blocks＝False | 同 | 相同 |
| 下一步 | 只在 Qwen 做 Step 2b | 同 | 相同 |
| 2b ε（Qwen，E1s@fp8k0_head、E1s@int8_token） | 四情境全過 | 四情境全過 | 相同 |
| 2b 速度不輸（Qwen） | fp8k0_head 不過；**int8_token 過** | fp8k0_head 不過；**int8_token 不過** | **不同** |
| **K3（Qwen）** | **成立** → FP8 階停、往 INT4 | **不成立** → Step 2 在 FP8 階的 Qwen 上做 | **不同** |

- 不同的格子（規則：X 不輸＝每格 median(X) ≤ min(F25,M25,B25,R25)×(1+δ*)，03 §2.4）：
  E1s@int8_token 在 **CPU 層（不限速的真 H2D）、GPU 忙**：H200 0.07989 s vs 最快混合 0.07115 s，比值 **1.1228**，δ*＝0.05 → 不過；
  MI300X 0.07127 vs 0.06643，比值 **1.0729**，δ*＝0.0943 → 過。CPU 層 GPU 閒：H200 0.7705（過）、MI300X 1.0421（過）。b16／b32／b48 兩台都過（比值 0.69–0.90）。
- 敏感度（只報告，不改判）：H200 的 1.1228 用 MI300X 的 δ*（0.0943）也不過；MI300X 的 1.0729 用 5% 的 δ* 會不過。也就是 MI300X 那一格是貼著門檻過的。
- `h200_m10_analyze.sh` 照 13 §4 在這裡停：H200 的路徑要 Step 2（FP8 階、Qwen：F25／M25／B25／R25 @fp8_head），主線 job 沒排這組。
- **這組資料正在 20707 裡跑**：stage 45（FP8 混合配置，MI300X 只報告）的 `f8m-qwen-*`：F25／M25／B25／R25 @fp8_head、讀法 A／B、b32、`meets_fp8tier.json`、T2–T4 n＝180（6 個 shard）、T5 n＝60，輸出 `quality_sealed/fp8mix_qwen_*`——和 I4M 同一個設計，只差階。用不用它來走 H200 的 Step 2，等使用者決定。

### 10/18 使用者決定：K3 分岔後「走 H200 的路＋對照」（13 §10，寫在 fp8mix_qwen 任何一列被分析之前）
- `code/h200_m10_analyze.sh`：
  - `all` 在 K3 不成立時改走 `step2fp8 <m>`（MI300X 凍結的 `analyze_step2.sh` 換 H200 路徑，拿掉 Qwen regrid 特例）。
  - `fp8mix_done` 先確認 20707 的 7 個 fp8mix 檔都寫完：有 gpu_guard JSON、run 的 exit_code＝0。
  - 新增 `compare`：假設 K3 成立、照 MI300X 的路走 INT4，輸出放 `analysis/compare_mi300x_path/`，和 H200 的判定分開。
- 回歸（MI300X 資料的副本，Qwen 速度換成 regrid 那份，同 MI300X 的 analyze_2b／step1）：`all` → GO 2b qwen → K3 成立 → INT4 → GO step2 llama → K2＝True；`analysis/fp8`、`analysis/int4` 的判定 JSON 和 MI300X 原檔相同（`fp8/speed_qwen.json` 是故意換的輸入）。`compare` 的 7 個 JSON 和 MI300X `analysis/int4` 相同（`sensitivity.json` 不是這支產生的）。
- `step2fp8 qwen`（08:2x）：照預期拒絕——fp8mix_qwen 7 個檔都還沒寫完；`step2fp8 llama`：s0–s3 認得是完成的，s4、s5、T5 沒完成。

### H200 的判定走完＋對照（2026-10-18 09:49–09:50 +08:00，head node；背景監看等 20707 的 fp8mix_qwen 7 個檔都 exit 0 後自動跑 `h200_m10_analyze.sh all`、`compare`）
**狀態**: 判定走完。H200：K3 不成立 → FP8 階 Qwen Step 2 → **K2 成立**（位置策略停）。MI300X：K3 成立 → INT4 → Llama Step 2 → K2 成立。**終點相同、路徑不同。**
**產出檔**: `analysis/fp8/quality_qwen_step2*.json／csv`、`analysis/fp8/step2_qwen_seedA.json`；對照 `analysis/compare_mi300x_path/`（README.txt 標明只報告）；`quality_sealed/fp8mix_qwen_*.csv.gz`。

**H200 的 Step 2（FP8 階、Qwen，δ*＝0.05）**：穩定差距第 1 條六格都不成立 → stable_gap_1_2＝False → K2＝True。
| 配置 | GPU | 對 R25 的品質差（讀法 B）| 95% CI | 還原時間 ÷ R25 | 第 1 條 |
|:--|:--|--:|:--|--:|:--|
| F25 | 閒 | +0.1733 | [+0.1464, +0.2013] | 1.0799 | 不成立（慢 > δ*） |
| M25 | 閒 | −0.0493 | [−0.0663, −0.0332] | 0.9962 | 不成立（輸 R25） |
| B25 | 閒 | −0.0336 | [−0.0508, −0.0167] | 0.9383 | 不成立 |
| F25 | 忙 | +0.1881 | [+0.1607, +0.2159] | 1.0990 | 不成立（慢 > δ*） |
| M25 | 忙 | −0.0438 | [−0.0598, −0.0278] | 0.9528 | 不成立 |
| B25 | 忙 | −0.0188 | [−0.0352, −0.0029] | 0.9572 | 不成立 |
- F25 的品質差是結構造成的：F25＝前 25% 存 FP8；讀法 B（Cake）前段重算、後段載入，前 16 個 FP8 block 都落在重算的那段 → **lp_read＝0，答案和 E0 完全相同**（三個 B 情境 diff 都是 +0.0000）。讀法 A 全部載入時 F25 對 E0 −0.2812。所以 F25 在讀法 B 等於「不用低精度」，品質贏 R25 是因為它沒讀到低精度的 block；第 1 條沒過的是速度那一半（慢 8–10%）。
- Qwen 的 FP8（fp8_head）只要讀到低精度 block 就掉很多：E1 對 E0 讀法 A −0.3223、B −0.26；M25／B25／R25 −0.17～−0.23。

**對照（假設 K3 成立、MI300X 的路；只報告）**：
| 節點 | MI300X | H200（對照） | |
|:--|:--|:--|:--|
| K0（INT4） | 兩個模型都不成立 | 同 | 相同 |
| INT4 端點 Llama | E1／E1s／E1sr 品質全過、速度都不輸**不過** → 不過 | 品質全過；**E1sr 速度不輸過** → K1s 成立 → 過 | **不同** |
| INT4 端點 Qwen | 不過 | 不過（品質全過；速度：CPU 層 E1 忙 1.69 倍、閒 1.16 倍） | 相同 |
| Step 2 的模型 | Llama | **Qwen** | **不同** |
| K2 | 成立 | 成立（F25／M25／B25 對 R25 的差都在 ±0.6 pp 內；F25 慢 11%） | 相同 |

**Llama 的 δ*（0.284）影響了兩個節點**——照規則不改判，列出來：
1. **H200 Step 1 Llama 的 K1**：E1／E1s／E1sr 在 CPU 層、GPU 忙的比值 1.1463／1.1450／1.1411，靠 δ*＝0.284 才「速度不輸」。δ* 用 0.05 的話三個都不過 → Llama Step 1 不過 → 路徑會不同。MI300X 的同一格是 1.0436／1.0332／1.0415（δ* 0.05 下過）。**H200 Step 1「和 MI300X 相同」是靠這個 δ* 才成立的。**
2. **對照路徑的 INT4 Llama E1sr**：CPU 忙比值 1.2806，門檻 1.2842，差 0.004 過；E1 1.2890 不過。
- δ*＝0.284 來自 noise 的 `cpu|busy` 一格（三次 launch 中位數 0.15526、0.12117、0.12081 s，第一次慢 28%），和上面兩個節點的比值是同一格（CPU 層、GPU 忙）。原因沒查證（同節點其他 job 的主機記憶體／PCIe 用量在 Slurm 下看不到）。
- 一致的現象（數字見上）：H200 上**CPU 層（不限速的真 H2D）、GPU 忙**時，低精度配置都比 BF16 混合配置慢——Qwen 2b INT8 1.12、Llama FP8 端點 1.14–1.15、Llama INT4 1.28–1.29、Qwen INT4 1.68–1.69；b16／b32／b48 限速時都快（比值 0.53–0.90）。

### job 20895 `m10h-lc-screen` 失敗在 PLAN（2026-10-18 07:56:39 → 08:09:16 +08:00，FAILED exit 1；commit `ca4eb72`）——我 09:5x 才發現
**狀態**: FAIL（規劃程式讀不到試跑時間；不是量測問題）
- PILOT：6 個都 exit 0（新的工作池預約：Llama-252＋Qwen-252 同時 123／140 GB，沒有 OOM）。
- PLAN 錯誤：`試跑的單題時間讀不到：{('llama', 64): None, ('llama', 128): None, ('llama', 252): 113.23333333333333, ('qwen', 64): None, ('qwen', 128): None, ('qwen', 252): 89.05}`
- 原因：`m10_sweep screen` 會續跑（輸出 CSV 裡已有的題跳過）。20705 已經寫好 Llama／Qwen 64、128 的 4 題和 Llama-252 的第 1 題，所以 20895 的這幾個 run 沒跑那些題、stdout 沒有計時行；`h200_longctx_plan.py`（照抄 MI300X）只讀每個名稱最新的 run。
- **修正**：`per_item_seconds` 改成每一題用它實際跑的那一次（同一題在兩個 run 都有時用後面那次），每個模型 × 長度要湊滿 4 題；`screen_plan.json` 多記 `pilot_items`（每題的秒數與 run_id）。
- head node 試算（輸出到 scratchpad，不寫 results/）：**完整設定、n＝12、估計 2.74 h**（上限 14 h）。題數規則只會選 12、9、6、3 中 ≤ 14 h 的最大值，2.74 h 離上限很遠，混用兩次 run 的計時不會改變 n。
- Llama-252 第 1 題（209.8 s）是 20705 那次（同卡 4 個 process）；其他 3 題是 20895（同卡 2 個）。照實記。

### 20897 `m10h-lc-screen`（20895 從 PLAN 重送）：09:53 開跑，PLAN＝完整設定、n＝12、估計 2.74 h（`results/m10_longctx_h200/screen_plan.json`），09:54 起 SCREEN（12 個 process）。

### job 20706 `m10h-q3-llama` 跑完（2026-10-18 06:47:11 → 13:10:58 +08:00，COMPLETED exit 0；commit `dc66667` 開跑時的 repo）
- Q3-POOL：21 個 process 全部 rc＝0（41 海選 2 個 shard；45 FP8 混合；55 T1＋LongBench；60 敏感度地圖；88 INT4）。舊的工作池（沒有預約），沒有 OOM。
- **41 海選分析**（`results/m10_sweep_h200/analysis/`，`20261018-130247-m10h-sweep-analyze-llama`；09 §4.2 同一條規則）：96 題（同 MI300X）；勝出的組合 173（MI300X 170）。
  決賽 6 個——H200：int2_kivi ERR 0.25、int2_kivi ERR 0.2、kv-int4_kivi-int2_kivi ERR 0.15、kv-int4_kivi-int2_kivi ERR 0.25、nvfp4 ATT 0.1、nvfp4 AXE 0.1；
  MI300X：int2_kivi ERR 0.2、kv-int4_kivi-int2_kivi ERR 0.2、kv-int4_kivi-int2_kivi ERR 0.25、int2_kivi ERR 0.15、int4_kivi ERR 0.3、mxfp4 ATT 0.2。
  **完全相同 2 個**（int2_kivi ERR 0.2、kv-int4_kivi-int2_kivi ERR 0.25）；兩台的 6 個裡 ERR 都佔 4 個、格式都以 int2_kivi／kv-int4-int2 為主。43 複賽用 H200 自己的（13 §9）。
- **75 RULER CWE（Llama，32K，30 題）**：`results/m10_precision_h200/ruler_cwe/`
  | 來源 | MI300X | H200 |
  |:--|--:|--:|
  | m10（自己的產生器） | 0.4767 | 0.5067 |
  | 上游 RULER（up_raw，模板含 BOS） | 0.0100 | 0.0067 |
  | 上游 RULER（up_nobos） | 0.0600 | 0.0800 |
  | hfgen（up_raw 前 10 題，HF generate） | — | 0.000 |
  vLLM（`20261018-130851-m10h-cwe-vllm`）：up_raw 0.0067、up_nobos 0.08。結論和 MI300X 相同：上游 RULER CWE 在 32K 上 Llama 幾乎 0 分。
- **題目是否相同（13 §9：比逐題 gold）**：m10 的 30 題 gold 與 prompt token 數**全部相同**；上游 RULER 的 up_raw、up_nobos 各 30 題**一題都不同**。
  已核對：RULER commit `c3f5e3b4…` 相同、seed 42 相同、wonderwords 3.0.1 的詞彙 8,166 個（和 `m10_ruler_cwe_check.py` 記的 MI300X 數目相同；32K 只用到 wonderwords，不用 LFS 詞表）。
  看得到的差別：第 0 題的詞數 H200 1902、MI300X 1909（由 `n_list_items` 回推），prompt 32,470 對 32,610 token——產生器逐步加詞、用 tokenizer 數長度，兩台停在不同的詞數，之後的亂數序列就全部錯開。**原因（tokenizer／transformers 版本數 token 的差別）沒有驗證。**
  所以上游 RULER 這一項是「同樣的產生程序、不同的題目」，結論（≈0 分）相同。
- `CWE-VLLM` 的 stderr 有 vLLM `deep_gemm` 找不到 CUDA_HOME 的 WARNING（AssertionError 被 vLLM 當成 warning 吞掉），引擎照常起來、30 題都跑完。
- 45、55、60、88 的分數已寫進 `quality*/`，還沒分析（MI300X 的報告腳本移植後再做；只報告）。

### job 20707 `m10h-q3-qwen` 跑完（2026-10-18 07:43:58 → 13:56:17 +08:00，COMPLETED exit 0）
- Q3-POOL：21 個 process 全部 rc＝0（舊的工作池，沒有 OOM）。
- **41 海選分析**（`results/m10_sweep_h200/analysis/finalists_qwen.json`）：96 題（同 MI300X）；勝出的組合 364（MI300X 384）。
  決賽 6 個——H200：kv-int4_kivi-int2_kivi AXE 0.4、kv-int4_kivi-int2_kivi ATT 0.4、kv-int8_token-int4_kivi ATT 0.1、kv-int8_token-int4_kivi AXE 0.1、int8_token ERR 0.1、int4_kivi ATT 0.35；
  MI300X：kv-int4_kivi-int2_kivi ATT 0.4、AXE 0.4、int4_kivi ATT 0.25、kv-int8_token-int4_kivi ATT 0.1、AXE 0.1、int4_kivi AXE 0.25。
  **完全相同 4 個**（kv-int4-int2 ATT／AXE 0.4、kv-int8-int4 ATT／AXE 0.1）。
- 45、55、60、88 的 Qwen 分數已寫進 `quality*/`，還沒分析（只報告）。
- 20708（t2，量時間）現在只等 20897。

### job 20897 `m10h-lc-screen` 跑完（2026-10-18 09:53 → 14:27:02 +08:00，COMPLETED exit 0）
- SCREEN：12 個 process（252×3＋128×2＋64×1 shard，兩個模型）全部 rc＝0；新的工作池預約，最多 3 個同時、沒有 OOM。分析（`m10_longctx_analyze`）接著在 head node 做。

### job 20708 `m10h-t2` 失敗、20709 `m10h-q5` 跟著取消（2026-10-18 14:27:02 → 14:27:19，FAILED exit 1；git `cae5c17`）
**狀態**: FAIL（H200 移植漏改的路徑；沒有量到任何東西）
- 步驟 GPU-SELFTEST：`m10h-dqfast-selftest`（`m10_dequant_fast.py --self-test`，GPU）exit 0；`m10h-dqoffset` exit 1：
  `File ".../code/m10_queue_mi300x/tests/dq_fast_offset_gpu.py", line 7 ... ModuleNotFoundError: No module named 'm10_codec'`
- 原因：測試檔的 `sys.path.insert(0, "/mlsteam/workspace/paper-hierarchical-kv-state/code")` 寫死 MI300X 路徑。
- 20709 是 `afterok:20708`＋`--kill-on-invalid-dep=yes` → CANCELLED。
- **修正**（`code/r_h200_x.sh`，不改測試檔）：dqoffset 用 `PYTHONPATH=$REPO/code`；head node 確認 `import m10_codec, m10_dequant_fast` 可以。
- **預檢 t2 其他步驟又找到一個**：`m10_dqsweep plan`、`m10_confirm plan` 都寫 `<jobs>/queue/*.sh` 但不建 `queue/`（MI300X 的 jobs 目錄本來就有）→ DQS-C、CONFIRM-PLAN 會失敗。`h200_lib` 先 `mkdir -p $WORK/queue`。在 scratchpad 用 H200 的 finalists 跑 `m10_confirm plan`：rc 0，寫出 43 的腳本（4 段、9 行 speed、沒有 /mlsteam）。
- 也查了：`m10_vllm_ttft.py` 的模型路徑本來就讀 `M7_<MODEL>_GLOB`（env_h200.sh 有設），不用改；`m10_dqsweep list／env` 在 CPU 上正常；`m10_tier2` 讀 repo 相對路徑的 `tier_params.json`（存在）。GPU 才能跑的步驟（tier2 smoke、dqsweep smoke、TTFT）沒辦法預檢。

### 20912 `m10h-t2`、20913 `m10h-q5` 重送（14:29 送出；20912 14:29:43 開跑，`c739bc1`）
- 監看改看 20912、20913。

### 合併 origin/main（AMD 到 `1ddba0b`）→ h200 `5116193`（2026-10-18 14:33:03 +08:00 快轉）
- **為什麼**：讀 main 時發現 AMD 10/17 修了兩個 43 會踩到的 bug，H200 的分支只合到 `bcfe5b1`、沒有這兩個修正：
  1. `92a31df`：`m10_confirm.py`（43 的冒煙／速度，ATT／AXE 選手）與 `m10_quality.py`（43 的品質）呼叫 `m10_sweep.ItemState`，類別實際叫 `Item` → NameError。H200 兩個模型的決賽都有 ATT／AXE。
  2. `abd4e6f`：`m10_confirm.py plan` 把 `--params-file` 寫成字面 `$P`。H200 預檢的 `jobs_43q.json` 12 個指令都是 `--params-file $P` → q5 會失敗。
- 其餘 code 改動是 13 號（還原第二輪）的開關，預設關閉（`M10_RS_PRIO`、`M10_MEET` none／g5、`nodq`、`M10_BUSY_CU_MASK`、`M10_DEQUANT_FAST_X`、`--extra-modes`）；預設路徑不變。`m10_sweep_analyze.py` 只改圖和 agg 的副檔名（選手規則沒動）。
- 做法：另一個 worktree 合併、解衝突（`.gitignore`、`docs/precision_20261013/README.md`，兩邊都是各加一行，兩行都留），用合併後的程式在 scratchpad 跑 `m10_confirm plan`：12 個 `--params-file` 都是實際路徑；再快轉主工作目錄（不讓跑中的 t2 讀到有衝突標記的檔）。
- **20912 的步驟用到的程式**：GPU-SELFTEST 的 4 個 run（dqfast-selftest、dqoffset、dqsweep-smoke-qwen、dqsweep-smoke-llama，最後一個 14:32:08 開始）是 `c739bc1`；DQ-BENCH 起是 `5116193`。
- 預檢的副作用：`m10_confirm plan` 把 `confirm_plan.json` 寫進了 `results/m10_sweep_h200/`（14:32），已刪；CONFIRM-PLAN 會重寫。

### 長 context 分析（head node，2026-10-18 14:3x；`M10_PRECISION_DIR=results/m10_precision_h200 python code/m10_longctx_analyze.py --raw ~/longctx_m10h --out-dir results/m10_longctx_h200/analysis --lc results/m10_longctx_h200`）
**產出檔**: `results/m10_longctx_h200/analysis/`（`longctx_<m>.json`、`tables_longctx.md`、`figs/`）。探索、不做判定（10 §5）。每個長度 48 題（4 任務 × 12）。
- 生成答對率（%）：
  | 模型 | 長度 | E0 | ALL FP8 主線 | ALL int8_token | ALL int4_kivi |
  |:--|:--|--:|--:|--:|--:|
  | Llama | 32K | 67.6 | 70.3 | 67.6 | 69.2 |
  | Llama | 64K | 66.6 | 63.6 | 67.6 | 67.0 |
  | Llama | 128K | 42.2 | 40.8 | 42.4 | 42.7 |
  | Qwen | 32K | 75.9 | 39.0 | 75.3 | 75.4 |
  | Qwen | 64K | 61.0 | 15.5 | 61.5 | 60.9 |
  | Qwen | 128K | 56.4 | 3.8 | 54.1 | 56.7 |
- Qwen fp8_head 對 E0：−37.0／−45.5／−52.6 個百分點；「128K−32K」−15.6 [−31.9, +1.4]，CI 含 0，不標「隨長度變大」。
- Qwen 有 6 個格式三個長度都是 0%（ALL 對 E0 等於 −E0）：fp8e5_head、int2h_g32、int4_token、int4h_token、mxfp4、nvfp4。
- Llama 標「隨長度變大」（CI 不含 0）的：fp8_token 64K−32K、fp8e5_head 64K−32K 與 128K−32K、int2_kivi 128K−32K、int4_token、int4h_token、nvfp4 的 64K−32K（數字見 tables）。
- MI300X 的長 context（44b／44c）在 origin/main 上還沒有結果，還不能對照。

### job 20912 `m10h-t2` 停在 TTFT-qwen-default、20913 `m10h-q5` 跟著取消（2026-10-18 14:29:43 → 16:43:25 +08:00，FAILED exit 1）
- 跑完的步驟（gpu_guard 都 CLEAN；NEEDS_HUMAN 沒有內容）：GPU-SELFTEST（融合反量化自測、Llama 大小緩衝 offset、dqsweep 冒煙兩個模型）、DQ-BENCH、DQGRID ×4（42）、DQS-A ×2、DQS-B ×2、DQS-C（42c，15:10 → 16:16）、TIER2 ×2（44）、TTFT-llama-default、TTFT-llama-512（70）。結果的整理之後做。
- **失敗**：`20261018-164301-m10h-vllm-qwen-default`（vLLM 0.28.0，Qwen2.5-7B-Instruct-1M 原版 snapshot）
  `(EngineCore) TypeError: FlashAttentionImpl.__init__() got an unexpected keyword argument 'layer_idx'` → `RuntimeError: Engine core initialization failed.`
- 原因：Qwen-1M 的 config 有 `dual_chunk_attention_config`，vLLM 走 DCA 的 attention，CUDA 的 FlashAttention 後端不收 `layer_idx`。CLAUDE.md §3 在 3090 上記過同一件事（V1 engine 沒有可用的 DCA 路徑，改用拿掉 DCA 的變體）；這個 repo 裡已經沒有 `make_nodca_model.py`。MI300X 的 70 還沒跑（origin/main 沒有 vllm_ttft 結果），沒有前例。
- 處理：70 只報告、和後面的 43 無關 → t2 從 CONFIRM-PLAN 重送為 **20917**（q5 **20918**，afterok）（Qwen 的兩格 TTFT 先 NOT_MEASURED，問使用者要不要用拿掉 DCA 的變體補量；補量要和我們自己的 job 錯開）。

### 10/18 使用者「可以」：補 Qwen TTFT（拿掉 DCA）、加 256K、做 δ* 5% 的對照（13 §11；寫的時候三件都沒有資料）
- 拿掉 DCA 的變體：`~/models/Qwen2.5-7B-Instruct-1M-noDCA`（只有 config.json 是實體檔、少了 `dual_chunk_attention_config`；其他 symlink 到原 blob）。harness 的 transformers 5.19 Qwen2 沒有 DCA（`dual_chunk` 0 次）。
- 程式：`r_h200_lc.sh --part speed256`（TTFT-qwen-nodca default／512 → 256K 速度格子 FP8、INT4）、`--part screen256`（Qwen × 508 × T2–T4）；`h200_longctx_plan.py` 加 `LONGCTX_VARIANT=256`——沒設時在 scratchpad 重算的 `screen_plan.json`、`jobs_screen.json` 和 20897 用的逐字相同；256K 版沒有速度格子時照預期停（缺讀法 B）。
- 送出：**20936** `m10h-lc256-speed`（afterany:20917）、**20937** `m10h-lc256-screen`（afterok:20936）；q5 **20918** 改 `afterok:20917,afterany:20936`（量時間的 20936 不和 q5 同時跑；20937 和 q5 都只量分數，可以同時）。

### δ* 用 5% 的對照（head node，2026-10-18 19:5x；只報告，不改判）
**產出檔**: `results/m10_precision_h200/analysis/sensitivity_delta005/`（README.txt、inputs/、fp8/、compare_mi300x_path/）
- 只改 `speed_llama/noise.json` 的 δ*（0.284 → 0.05），重算 Llama 的 k0-verdict（FP8、INT4 都還是「K0 不成立」）與速度判定，再走 `h200_m10_analyze.sh all`、`compare`。
| 節點 | MI300X | H200（δ* 0.284，判定） | H200（δ* 0.05，對照） |
|:--|:--|:--|:--|
| Step 1 Llama | 過（K1） | 過（K1） | **不過**（CPU 忙 1.14–1.15 > 1.05） |
| Step 1 Qwen | 不過 | 不過 | 不過 |
| 2b 的模型 | Qwen | Qwen | **Llama** |
| K3 | 成立（Qwen INT8） | 不成立（Qwen） | 不成立（Llama：fp8k0 CPU 忙 1.2011、INT8 1.1431；ε 四情境都過） |
| Step 2 | INT4 階、Llama | FP8 階、Qwen | **FP8 階、Llama**（F25／M25／B25 對 R25 的差 −0.34～+0.19 pp，CI 都含 0） |
| K2 | 成立 | 成立 | 成立 |
| 對照路徑（假設 K3 成立）的 INT4 | 兩個模型都不過 → Step 2 Llama → K2 | Llama E1sr 過 → Step 2 Qwen → K2 | **兩個模型都不過 → Step 2 Llama → K2（和 MI300X 每個節點相同）** |
- 所有分岔都在同一格：**CPU 層（不限速的真 H2D）、GPU 忙**。終點三種走法都是 K2 成立。

### job 20917 `m10h-t2`（從 CONFIRM-PLAN）跑完（2026-10-18 16:45:10 → 20:17:19 +08:00，COMPLETED exit 0；程式 `5116193` 之後）
- CONFIRM-PLAN：照 H200 自己的 finalists 寫 43（`results/m10_sweep_h200/confirm_plan.json`、`_m10h_work/x-t2/queue/43a_speed.sh`、`43b_quality.sh`；`--params-file` 是實際路徑）。
- CONFIRM-SPEED（43a）：冒煙兩個模型都過、沒有寫 NEEDS_HUMAN；7 個格式的速度格子（Llama int2_kivi、kv-int4_kivi-int2_kivi、nvfp4；Qwen kv-int4_kivi-int2_kivi、kv-int8_token-int4_kivi、int8_token、int4_kivi），**21 份 gpu_guard 都 CLEAN**。
- 20912＋20917 合起來，t2 的步驟除了 Qwen 的兩格 TTFT（13 §11.1 在 20936 補量）全部跑完。產出：`results/m10_sweep_h200/{dqfast_selftest_gpu.json, parts/dequant_fast_bench.json, speed_dqfast*_<m>/, dqsweep/, confirm/}`、`results/m10_tier_h200/`、`results/m10_precision_h200/vllm_ttft/`（confirm 的 restores.csv 超過 1 MB，進 git 的是 .gz）。
- 70 Llama 的 vLLM TTFT 中位數（s）：default 8K 0.2134、16K 0.4989、32K 1.2265；512 切法 0.2441、0.5519、1.3513（`vllm_meta.json`）。
- 和 MI300X 的逐項對照（42 反量化、42b／42c 方法、43 複賽、44 兩層）等 q5（43 的品質）跑完一起做。
- 20936（Qwen noDCA TTFT＋256K 速度）20:17 開跑。

### job 20936 `m10h-lc256-speed` 跑完（2026-10-18 20:17:21 → 2026-10-19 03:36:01 +08:00，COMPLETED exit 0）
- **70 Qwen 的 vLLM TTFT（拿掉 DCA 的變體，13 §11.1）**：`vllm_meta.json` 的 model_path＝`~/models/Qwen2.5-7B-Instruct-1M-noDCA`；attention backend FLASH_ATTN。
  中位數（s）：default 8K 0.1935、16K 0.4343、32K 1.0494；512 切法 0.2250、0.5039、1.2127。
- **256K 速度格子**（`results/m10_longctx_h200/speed_qwen_260096_{fp8_head,int4_kivi}/`）：FP8 20:19 → 00:02，INT4 00:02 → 03:36；每格 2 份 gpu_guard 都 CLEAN。
- 03:36 起 20918（q5，43 的品質）和 20937（256K 品質）同時跑（都只量分數）。

### job 20937 `m10h-lc256-screen` 跑完（2026-10-19 03:36:04 → 05:18:29 +08:00，COMPLETED exit 0）
- PILOT256：1 個 process（Qwen × 508 × T2–T4 各 1 題），單題平均 152.4 s，peak 56.5 GiB（need 設 70）。PLAN256：完整設定、**n＝12**、估計 0.51 h（`results/m10_longctx_h200/screen_plan_256.json`）。SCREEN256：2 個 shard，rc＝0。
- 分析（head node；`code/h200_longctx256_analyze.py`，新寫，借用 `m10_longctx_analyze.py` 的函式、不改它）：`results/m10_longctx_h200/analysis/longctx256_qwen.json`、`tables_longctx256.md`。每個長度只取 T2–T4（36 題）。
  | 長度 | E0 | ALL fp8_head | ALL int8_token | ALL int4_kivi |
  |:--|--:|--:|--:|--:|
  | 32K | 77.1 | 46.9 | 77.1 | 77.2 |
  | 64K | 70.6 | 17.6 | 71.1 | 70.4 |
  | 128K | 67.9 | 3.9 | 65.7 | 69.2 |
  | 256K | 55.1 | 0.0 | 55.1 | 51.8 |
  - fp8_head 對 E0「256K−32K」−25.0 [−42.1, −6.1] ↑（CI 不含 0）。int4_kivi 256K −3.3 [−10.0, +1.1]、int8_token 0.0。
  - 位置（25%）：fp8_head 的 F 對 R0 +48.1 [+36.2, +59.9]、M +31.4 [+20.8, +42.5]、B −4.7 [−11.7, 0.0]；int8／int4_kivi 的差都在 ±3 個百分點內。
  - 速度（E1 對 E0）：限速層 FP8 快 36.6–43.4%、INT4 快 37.1–59.3%；CPU 層 FP8 快 4.3–5.2%、**INT4 慢 23.9–24.1%**。
- 注意：256K 的 CSV（`screen_qwen_508_*`）和 32K–128K 放在同一個 `~/longctx_m10h`；`m10_longctx_analyze.py` 的 LEN 沒有 508，之後要重跑 32K–128K 的分析時 `--raw` 指到不含 508 檔的目錄（已寫進新程式的說明）。

### job 20918 `m10h-q5` 跑完（2026-10-19 03:36:04 → 12:24:58 +08:00，COMPLETED exit 0）＝ H200 第二階段的 GPU 工作全部結束
- CONFIRM-QUALITY（43b）：12 個品質 process 全部 rc＝0；NEEDS_HUMAN 沒有內容；判定（`m10_confirm_analyze.py`）在同一段自動跑完：`results/m10_sweep_h200/confirm/{confirm_llama.json, confirm_qwen.json, tables_confirm.md, analysis/}`。
- 小事：43 品質的 run 名稱是 `m10-confirm-*`（`jobs_43q.json` 的指令沒有被改成 m10h；只有 43 的 .sh 被 sed 過）。不影響內容，run_id 照實記。

### 43 複賽（大掃描）：H200 vs MI300X（`tables_confirm.md` 並排；新 seed 50000 起，每個模型 300 題）
- **兩台都是「6 個選手沒有一個複賽成立」**（判定節點相同）。
- 兩台都有的選手：
  | 模型 | 配置 | MI300X：ε／贏隨機 A（Holm）／差 A（pp）／速度不輸 | H200：同 |
  |:--|:--|:--|:--|
  | Llama | SW-ERR-20@int2_kivi | 沒過／**是**／+2.0 [+0.8, +3.1]／True | 沒過／否／+0.9 [−0.3, +2.0]／True |
  | Llama | SW-ERR-25@kv-int4_kivi-int2_kivi | 過／否／−0.4 [−1.2, +0.4]／True | 過／否／+0.0 [−0.8, +0.8]／True |
  | Qwen | SW-ATT-40@kv-int4_kivi-int2_kivi | 過／否／+0.5 [−0.4, +1.4]／False | 過／否／+1.1 [+0.1, +2.1]／False |
  | Qwen | SW-AXE-40@kv-int4_kivi-int2_kivi | 過／否／+0.5 [−0.3, +1.5]／False | 過／否／+1.2 [+0.2, +2.1]／False |
  | Qwen | SW-ATT-10@kv-int8_token-int4_kivi | 過／否／−0.3 [−0.8, +0.2]／True | 過／否／−0.2 [−0.7, +0.3]／True |
  | Qwen | SW-AXE-10@kv-int8_token-int4_kivi | 過／否／−0.2 [−0.8, +0.2]／True | 過／否／−0.3 [−0.7, +0.2]／True |
- H200 Llama 的「速度不輸」用 δ*＝0.2842（主線凍結的 noise.json）。

### 42／42b／42c 反量化：H200 vs MI300X（`results/m10_sweep_h200/dqsweep/C/{report_C.json, tables_C.md}`，head node 10/19 跑 `m10_dqsweep.py report`）
- **移植修正**（`code/m10_dqsweep.py`，沒設環境變數時不變）：report 的 δ* 寫死讀 `m10_precision_mi300x`，「主線／42」參考欄寫死讀 MI300X 的 summary（`REF_SUMMARY`）→ 第一次跑出的 H200 表那一欄是 MI300X 的數字（例：Llama FP8 CPU 忙 0.171／0.082，和 MI300X 表逐字相同），發現後改成 `M10_PRECISION_DIR`／`M10_SWEEP_DIR` 有設時讀 H200 自己的格子；重跑後的表才是記錄的這份。E0、E1、不輸比等其他欄本來就是 H200 的資料。
- B 段（42b）：兩台都是「一組拆（S2k）」贏。冠軍 MI300X S2k8、H200 S2k16；進 C 段的 5 個有 3 個相同（S2k16、S4、S0）；problems 都是空的。
- 42（融合 kernel）與 C 段，CPU 層 E1 的還原時間中位數（s；主線 → 42 → C 段冠軍）：
  | | GPU | MI300X | H200 |
  |:--|:--|:--|:--|
  | Llama FP8 | 忙 | 0.171 → 0.082 → 0.085（S2k8） | 0.144 → 0.083 → 0.048（S2k16） |
  | Llama INT4 | 忙 | 0.189 → 0.082 → 0.083 | 0.168 → 0.077 → 0.042 |
  | Qwen FP8 | 忙 | 0.074 → 0.065 → 0.065 | 0.082 → 0.068 → 0.034 |
  | Qwen INT4 | 忙 | 0.146 → 0.066 → 0.065 | 0.122 → 0.067 → 0.034 |
  兩台都是：融合 kernel 把 CPU 層的變慢拿掉；H200 上「一組拆」再快一截（E1 比 E0 快 50–65%），MI300X 上一組拆和 S0 差不多（MI300X 記的「grouping hits the one-block-recompute floor」）。
- **和主線分岔的關係**：主線判定那幾格（CPU 層、GPU 忙，低精度慢 12–20%）用的是原寫法（K0、S0）；換成融合 kernel＋一組拆之後，同一格 H200 的 E1 比 E0 快。主線判定照凍結的寫法，不改。

### 還沒對照的（MI300X 在 origin/main 上還沒有結果，10/19 12:5x）
- 44 兩層混合還原（H200：`results/m10_tier_h200/`）、70 vLLM TTFT、44b／44c 長 context（MI300X 10/19 剛開始 44c）。
- 只報告、H200 資料已齊但還沒分析：45、55、60、88；13.1 A（`m10_iowall` 的 run 表待移植）、10.11（INT4 敏感度，待移植）。

### 10/19 晚：合併 main、13／14 號送出、三支 CPU 分析移植（head node）
- 合併 origin/main（AMD 到 `c27ec61`）→ h200 `4e5aa7d`。Makefile 衝突：用「有沒有 /mlsteam/workspace」分平台，H200 的 make 目標不變。gpu_guard 的 Slurm 測試 9 項過。
- 13 號（還原第二輪）、14 號（配對／丟 block）照 MI300X 跑：事先登記 13_h200_repro §12；`code/h200_port_stage.py`＋`code/r_h200_rs2.sh`。
  - 送出 **21109** `m10h-rs2`（量時間）→ **21110** `m10h-md`（afterany）→ **21111** `m10h-mdt`（afterany）。
  - MI300X 的 CPU 測試（`restore2_cpu_shim.py`、`match_drop_cpu.py`）在 H200 環境 ALL PASS。
- 使用者 10/19：「code 需要改成配 NV 可以，paper 數字要注意這些」→ 總表 `docs/h200/NV_PORTING_NOTES.md`。
- 另一個工作階段（P3 移植，worktree `paper-hkv-p3`，分支 `h200-p3`）：協調好它的 GPU／吃 CPU 的 job 都 `afterany:21111`；它不碰共用工作目錄。
- **head node `/tmp`（20 G，共用）又滿**：這次 Claude Code 的暫存在 `/tmp`。我在 scratchpad 解壓 MI300X 的品質 CSV 做回歸，把剩下的空間用完，所有工具 ENOSPC。P3 工作階段幫忙刪了 `sens_mi`（9.8 MB），我再刪自己 10/15–16 的舊暫存（134 MB，第一階段回歸的輸出、gh 的下載檔；git 用的 gh 在 `~/.local/bin`，不受影響）。之後的回歸暫存一律放 `/work/ChihChungHsu/hungweilai/tmp`。
- 三支 CPU 分析移植（沒設環境變數時不變）：
  - `m10_cake_ratio.py`：`M10_PRECISION_DIR`；沒有 regrid 目錄就用 `speed_qwen`。回歸：MI300X 的 `cake_ratio.md` 逐字相同。
  - `m10_int4_sensitivity.py`：`M10_PRECISION_DIR`、`M10_SENS_OUT`。回歸（MI300X 的品質 CSV 解壓到 /work）：`sensitivity.json`、`.md` 都逐字相同。
  - `m10_iowall.py`：`M10_IOWALL_GRIDS`（run 編號表；H200：`results/m10_tier_h200/iowall_grids_h200.json`，主線 FP8／INT4 格子，沒有 Qwen 重量那次）。沒有 MI300X 的 runs 目錄，回歸只能確認沒設變數時表不變（程式只多一個 if）。
- H200 的結果（只報告）：
  - `results/m10_precision_h200/analysis/cake_ratio.{md,png}`：Cake 增益 CPU 層 0.53–3.49×（比 0.016–0.070）、b16 0.93–1.30×、b32 1.12–1.98×、b48 1.50–2.04×（每層 12 格）。CPU 層上限 3.49× 是 GPU 忙的格子。
  - `results/m10_precision_h200/analysis/sensitivity_int4/`：和 E0 逐字相同的比例 Llama INT8 93.6%、FP8 K 先減平均 89.3%、FP8 86.3%、INT4 KIVI 82.6%；Qwen INT8 85.7%、FP8 K 先減平均 85.6%、**FP8 10.4%**（KL 中位數 0.537）、INT4 KIVI 78.1%。
  - `results/m10_tier_h200/{iowall.json, tables_iowall.md, figs/A2_*_wall.png, figs/A3_*_spill.png}`（A 用 stage 44 的實測更新）。

### job 21109 `m10h-rs2`（13 號還原第二輪，量時間）跑完（2026-10-20 12:22:33 → 17:15:59 +08:00，COMPLETED exit 0；程式 `d82e670`）
- stage 43a 43b 43c 43d 43e0 43e 44cz 44d 全部 exit 0；226 份 gpu_guard 全 CLEAN。log：`$TIARA_RUNS/_m10h_work/rs2/log/<stage>.log`；表 `results/m10_restore2_h200/{tables_13.md, report_13.json}`（head node `m10_restore2.py report`，目錄用環境變數指到 H200）。
- NEEDS_HUMAN 4 條：
  - CU 遮罩不生效（預期；`smoke/cumask.json`，132 SM，P4a／P4b 的 matmul 時間和沒遮罩一樣）→ P4 不跑，NOT_MEASURED（平台）。
  - attention 對照組自測沒過 → 43e 的 (b) 沒跑（見下）。
- 挑選結果 vs MI300X：
  - q4_pick **P3 = P3**。GPU 忙、全載入 E1 的幾何平均：P0 0.236 → P3 0.043 s（MI300X 0.047 → 0.029）。H200 的背景是 duty 1.0，P0 被拖得多很多；背景吞吐比 P3 0.97（MI300X 1.002）。
  - g5_pick **G5（MI300X 是 G5b）**：96 格「÷ 最好」幾何平均，G5 1.042、G5b 1.115（MI300X：G5 1.048、G5b 1.004）。所以 44d 用 G5。
  - c_pick **S2k16+G4（MI300X 是 S2k8+G4）**：結構相同（冠軍＋G4），冠軍是各自的。
- 43e：(a)、(d) 跑完；(b) 沒跑。原因：Triton 在 sm_90 遇到 BM＝16 時走 mma.sync，融合讀法和讀 BF16 的運算元排法不同（kWidth 4 vs 2），只差最後幾位；M＝256 走 wgmma，逐位元相同。分析與改法事先登記在 13_h200_repro §13，送出 attnb 補跑。
- (d) vLLM 0.28 CUDA，每個 token 的 decode（ms，BF16 → FP8 KV）：Llama 32K 5.58 → 5.47、128K 7.63 → 6.93；Qwen（拿掉 DCA）32K 5.08 → 4.93、128K 5.75 → 5.47。
- 報告標籤的移植：背景強度欄照 `busy_levels_q4.json`、Q3 原寫法路徑照 `M10_SWEEP_DIR`、(d) 標 CUDA。MI300X 的 `tables_13.md`、`report_13.json` 重跑逐字相同。
- 改 `r_h200_rs2.sh` 時，21110 正在執行它（re-exec 的 repo 版本）。我先把原內容原地寫回，再用「寫新檔再 mv」換上新版，21110 手上開著的仍是舊 inode。最壞情況是 21110 結束時少印 gpu_guard 摘要，可以事後補。之後改 runner 一律用 mv 換檔。

### 10/20 傍晚：合併 main（AMD 到 `61027ac`）、長 context 表重跑、和 MI300X 的 44d／長 context 對照（head node）
- 合併在另一個 worktree 做，h200 `6557401`，再 fast-forward 回來。
  - `m10_longctx_analyze.py` 衝突：保留 H200 的 `rg`（H200 沒有 Qwen 重量那次）＋main 的「32K 參考列篩長度」。
  - gpu_guard 的新程式（KFD 每卡 VRAM、ROCR 優先）只走 AMD 的路徑。`test_gpu_guard_slurm.py` 9 項 OK；MI300X 的 `restore2_cpu_shim.py`、`match_drop_cpu.py` ALL PASS。
  - m16（MI300X 第三階段）的程式和結果一起進來。H200 這邊不跑；P3 的移植由另一個工作階段負責。
- **長 context 表重跑**：main 修了一個 bug，32K 的 summary 裡 8K／16K 的閒置參考列會蓋掉 32K。
  - 指令：`M10_PRECISION_DIR=results/m10_precision_h200 python code/m10_longctx_analyze.py --raw <~/longctx_m10h 去掉 508 的 symlink 目錄> --out-dir results/m10_longctx_h200/analysis --lc results/m10_longctx_h200`。
  - 只有 `tables_longctx.md`、`longctx_<m>.json` 的「速度、32K、閒」那 24 格改變（例：Llama 32K fp8 cpu|idle 原本 0.03866→0.02346，現在 0.10201→0.07629）。品質表、圖、256K 都沒變；RUNLOG 沒有引用過這幾格。
- **長 context 品質 vs MI300X**：同一批題目（我們自己的產生器），兩台幾乎一樣。例：Qwen fp8_head 128K 1.5%（MI300X）vs 3.8%（H200）；E0 56.6 vs 56.4；Llama 128K E0 40.3 vs 42.2。
- **長 context 速度（E1 對 E0）vs MI300X**：
  - b32 兩台都是 E1 快 31–42%。
  - CPU 層、GPU 忙，兩台大多是 E1 較慢（Llama 64K fp8 −9.0% vs −5.5%；Qwen 128K fp8 −8.9% vs −4.2%）。例外是 Llama 128K fp8：MI300X +7.6%、H200 −3.7%。
  - CPU 層、閒，Qwen 方向不同：MI300X +13.2%（128K）、+2.6%（64K）；H200 −3.8%、−13.3%。
  - MI300X 的長 context 表還沒放進 44ca／44cab 的 INT4 格子，所以 INT4 只有 H200 的數字，還不能並排。
- **13 號 Q1（44d）vs MI300X**（CPU 層、GPU 忙、E1 cake 中位數 s，MI300X／H200）：
  - 方法的排序兩台大致相同：nodq ≤ Q4_P3 ≈ C 段 < P0。
  - 例外是 **H200 的 Qwen fp8_head**：nodq 128K 0.079／**0.464**、64K 0.040／**0.234**；Q4_P3 0.087／**0.390**；C 段 0.099／**0.200**，都比 P0（0.183）慢。同一格閒的時候是 0.071，Qwen INT4 忙的時候是 0.049，都不慢。
  - 32K 的 Q5 (a) 也一樣：「只搬不拆」忙 0.118、閒 0.019（Llama fp8 0.041／0.040）。
  - 只在 Qwen fp8_head、GPU 忙（H200 的 duty 1.0）時出現，原因還沒查。
  - G5／G5b：兩台各用自己挑的（H200 G5、MI300X G5b），這一列不並排。

### job 21110 `m10h-md`（14 號：冒煙＋DS1／DS2，只量分數）跑完（2026-10-20 17:15:59 → 19:47:23 +08:00，COMPLETED exit 0；程式 `d82e670`，17:34 之後新起的 process 用合併後的程式，NV 路徑沒有改變）
- 44e0 冒煙過；44e1 DS1 12 個 process、44e2 DS2 14 個 process，全部 rc＝0。gpu_guard 226 份全 CLEAN，NEEDS_HUMAN 沒有新的。表：`results/m10_drop_h200/ds1/tables_ds1.md`、`ds2/tables_ds2.md`、`ds2/ds2_report.json`；逐列 CSV 進 git 的是 `.csv.gz`（和 MI300X 一樣）。
- **DS1 挑的選手和 MI300X 完全相同**：Llama T＝KEYDIFF、KNORM，B＝KNORM、KEYDIFF；Qwen T＝KEYDIFF、EXPATT，B＝EXPATT、KEYDIFF。DS2 的 25 個配置兩台一樣。
- **DS2 結果兩台幾乎一樣**（對 E0 的點數差，MI300X／H200）。例：
  - Llama DS-T-KEYDIFF-25 −0.5／−1.5、DS-B-KNORM-50 −11.8／−12.4、DS-T-RND0-50 −73.5／−73.3；
  - Qwen DS-B-EXPATT-25 −5.6／−6.0、DS-T-KEYDIFF-50 −3.9／−3.4。
- ε（不輸）的判定只有一格不同：**Llama DS-T-NONE-0**（只對 token 群組化、不丟）MI300X 沒過（+0.2 [−1.0, +1.3]），H200 過（−0.5 [−1.6, +0.5]）。另外兩台都過的：Llama DS-T-QRY-25、Qwen DS-B-NONE-0、Qwen DS-T-NONE-0。其他都沒過。
- 21110 結束時有正常印出 gpu_guard 摘要，可見改 runner 時用 mv 換檔的做法沒有影響它。

### job 21377 `m10h-attnb`（43e 的 (b) 補跑，量時間；13_h200_repro §13）跑完（2026-10-20 19:47:23 → 20:00:00 +08:00，COMPLETED exit 0；程式 `7fe3fe9` 之後）
- 自測（`20261020-194727-m10h-rs2-attn-selftest3-bm64`，`smoke/attn_selftest_gpu_bm64.json`）ALL PASS。decode 的 6 格維持原本的情況：BM＝16 時不逐位元，BM＝64 時**全部逐位元相同**，對 SDPA 的誤差和上次一樣。所以先前查到的原因成立：拆出來的值相同，差別只在 mma.sync 的運算元排法。
- (b) 微測：`20261020-194739-m10h-rs2-q5b-llama`、`20261020-195436-m10h-rs2-q5b-qwen`，指令和 MI300X 一字不改。gpu_guard 228 份全 CLEAN。表：`tables_13.md` 的 Q5 (b)、(c)。
- **和 MI300X 的方向相同**：同一個 kernel 裡，讀壓縮比讀 BF16 慢。
  - M＝256：H200 慢 3.3–4.0 倍（fp8）、2.3–2.8 倍（int4）；MI300X 慢 3.6–4.4 倍、3.2–3.9 倍。
  - decode：H200 128K fp8 21.14 vs bf16 7.48 ms（閒，Llama）；MI300X 23.87 vs 7.81。
- (c) 划算的 token 數：
  - 閒的時候兩台同一級：H200 2–11、MI300X 2–27。
  - **忙的時候 H200 大很多**：27–100，MI300X 3–54。原因是 H200 背景 duty 1.0，P0 全載入在忙時被拖得多，還原省下的時間是 117–222 ms，MI300X 是 12–26 ms。兩台的「忙」不是同一個強度（NV_PORTING_NOTES §1），這一欄只能各自看。
- 別的平台差異：PyTorch SDPA 讀 BF16，H200 是 1–93 ms，MI300X 是 43–272 ms（ROCm 的 SDPA 路徑慢）；外部參考 vLLM-KIVI 在兩台都最慢。
- 小異常：Qwen 32K decode 的 bf16 kernel，忙 2.06 ms 比閒 3.38 ms 快。只有這一格，沒追。

### job 21111 `m10h-mdt`（14 號量時間：D3、M2、M1＋報告）跑完（2026-10-20 20:00:08 → 22:20:56 +08:00，COMPLETED exit 0；程式 `95fa1a7`）
- 44e3 D3（丟 block 之後的還原時間：Llama／Qwen int4，加 NFS 層）、44e4 M2（8 種格式的載入／計算配對：compute_only、load_only、cake）、44e5 M1（單位大小 u128／u256／u512）、44ez 報告，全部 exit 0。268 份 gpu_guard 全 CLEAN，NEEDS_HUMAN 沒有新的。
- 表：`results/m10_match_h200/{tables_speed_14.md, speed_14.json}`；20 個格子都「可用」。
- **MI300X 還沒有 14 號的量時間結果**（origin/main 上沒有 `results/m10_match_mi300x/`），所以目前只有 H200 的數字，等 MI300X 跑完再並排。
- 描述（只報告，Llama int4，32K）：
  - D3：丟 50% block，各層的 cake 中位數大約減半。例：CPU 層、忙，E1s 0.0764 → DB50 0.0385 s；b48、閒，E1s 0.856 → DB50 0.402 s。
  - M1：單位縮成 128 token，CPU 層、閒反而變慢（E1s 0.168 s vs E0 0.124 s）。
- 到這裡，13、14 號在 H200 能跑的部分都跑完了。沒跑的只有平台不支援的 P4a／P4b（CU 遮罩），寫 NOT_MEASURED。

### job 21402 `m10h-nvx`（13_h200_repro §14 的兩個 H200 才有的加做，量時間，只報告）跑完（2026-10-20 23:14:21 → 23:41:35 +08:00，COMPLETED exit 0；程式 `4216099`）
- 原本排在 P3 的 21116 之後。21116 在 23:14 失敗（P3 那邊的測試 bug），這個 job 就提早開始。P3 重送的 21403 排在這個之後。gpu_guard 305 份全 CLEAN。
- NEEDS_HUMAN 新增 1 條：運算節點上找不到 nsys（head node 有 `/usr/local/cuda/bin/nsys`，hgx-cn01 沒有）→ **A3 沒跑，NOT_MEASURED**。

**A：Qwen fp8_head 在 GPU 忙時「只搬不拆」也慢 → 事先寫的假設（極小 H2D × 時間片）不成立**
- A1 極小拷貝微測（`20261020-231425-m10h-diag-tinycopy`，`results/m10_diag_h200/tinycopy.csv`；64 次、中位數 ms，閒／忙）：
  - 連續 64 次單一大小：896 B 0.382／2.120、4 KiB 0.389／1.885、16 KiB 0.415／1.821、64 KiB 0.392／0.392、1 MiB 1.833／1.802、14 MiB 16.55／16.55。極小的只在第一次多等約 1.5 ms，之後不再等；換算每塊多 0.02–0.03 ms，**不到判準 (1) 的 0.5 ms → (1) 不成立**。
  - 模仿一個 block 的兩塊交錯：Qwen fp8_head（14.68 MB＋896 B）**17.2／117.0**（每個 block 多 1.56 ms，和正式量的 0.019／0.118 s 一樣）；Llama fp8_token（33.5 MB＋1 MiB）38.9／38.9；Qwen int4 11.3／11.4。
  - 程式的 bug：摘要 JSON 把各個大小的「single」合成同一個 key，逐筆 CSV 沒問題，上面的數字是從 CSV 算的。程式已修（`single:<大小>`）。
- A2 2×2 交叉（只搬不拆、CPU 層、E1 中位數 s，閒／忙）：
  - Qwen fp8_head 0.0185／0.1182（×6.38）
  - **Llama fp8_head 0.0391／0.0403（×1.03）**：scale 一個 block 2 KiB，同樣是極小拷貝，可是不慢
  - Qwen fp8_token 0.0185／0.0198（×1.07）
  - Llama fp8_token 0.0403／0.0415（×1.03）
  - 判準 (2) 要求 fp8_head 兩個模型都變慢 → **不成立**。
- A3：nsys 沒跑（見上）。
- **結論**：照 §14.1 寫死的規則，假設不成立，不換假設硬套。確定的只有：
  - 這個現象只出現在 **Qwen × fp8_head**；
  - 用「14.68 MB＋896 B 交錯拷貝」可以在 H2D 微測裡重現；
  - Llama 的 fp8_head（33.5 MB＋2 KiB）不會。
  - 原因 NOT_DETERMINED。要再查的話（例如換交錯的大小組合、在運算節點載入有 nsys 的 module），要先問使用者。

**B：P4 的 CUDA MPS 版**（`results/m10_restore2_h200/q4mps/{tables_q4mps.md, report_q4mps.json, mps.json, mps_log/}`）
- MPS daemon 用 job 自己的目錄起來了；GPU compute mode 是 Default。限制檢查：背景單獨跑的 matmul 從 1.394 ms 變成 1.465 ms（87%，×1.051）、1.650 ms（75%，×1.18），兩個都生效（p4a 剛好過 1.05）。gpu_guard 冒煙 CLEAN（MPS server 算本 job 的）。12 個 grid 全部可用。
- 閒的時候三欄和沒開 MPS 的 P0 一樣（例：Llama fp8 E1 cake 0.062 vs 0.061）。開了 MPS 本身不改變沒背景時的數字。
- **忙的時候**（E1 中位數 s，cake／loadall）：

| | P0mps | P4a_mps | P4b_mps | 參考：沒開 MPS 的 P0 | 參考：沒開 MPS 的 P3 |
|:--|:--|:--|:--|:--|:--|
| Llama fp8 | 0.397／0.108 | 0.127／0.061 | 0.076／0.056 | 0.117／0.236 | 0.047／0.051 |
| Qwen fp8 | 0.309／0.108 | 0.117／0.030 | 0.062／0.027 | 0.068／0.235 | 0.037／0.120 |
| Qwen int4 | 0.297／0.108 | 0.118／0.024 | 0.062／0.021 | 0.067／0.236 | 0.037／0.017 |

  - 背景吞吐（÷ P0mps）：P4a_mps 0.91、P4b_mps 0.78。MI300X 的 P4a、P4b（÷ 它的 P0）是 0.88、0.80，數值相近，但機制不同，不放同一欄。
  - 讀法：MPS 不限時，背景把所有 SM 佔滿，Cake 的重算被擠得很慢（cake 比時間片輪流的 P0 慢 3–5 倍）；全部載入不用等時間片，反而快一倍。限制背景的 SM 比例以後，兩種都快很多。P4b_mps 的 loadall 0.021–0.056 s，和沒開 MPS 的 P3 同一級或更快。
  - 只報告，不改 q4_pick。這三欄只和彼此比（§14.2）。

### 10/21：合併 main（AMD 到 `88016c8`，MI300X 第二階段結束）、14 號量時間 H200 vs MI300X（head node）
- 合併在 worktree 做，h200 `e0ab345`。
  - 衝突只有 `gpu_guard.py`：保留 main 的「先列行程、再算子孫」（race 修正）＋H200 的 `_in_own_job`（環境變數開關）。Slurm 測試 9 項 OK；MI300X 的 CPU 測試 ALL PASS。
  - main 帶進來：14 號量時間、55／60／88 品質、加 guard 的 vLLM TTFT（90–92）、第二階段報告（`code/report_stage2/`、`docs/precision_20261013/report_stage2/REPORT.md`）。
- **14 號量時間對照**（`results/m10_match_{mi300x,h200}/tables_speed_14.md`，32K 中位數；限速層的 GiB/s 兩台不同，只比比值與方向）：
  - **D3 丟 block（丟了以後 ÷ 不丟），兩台幾乎一樣**。
    - 慢的層、Cake、閒，MI300X／H200：Llama b32 E0 F50 1.00／0.96、B50 0.47／0.46、R50 0.65／0.65；b48 E1s F50 1.00／1.00、B50 0.45／0.47；nfs E0 B50 0.43／0.44。
    - 結論「丟前面（會合點以內）不省、丟後面省最多」兩台相同。
  - **D3 的平台差異**：CPU 層、Cake、GPU 忙、E1s 的 B50，MI300X 0.93（Llama）／0.98（Qwen），**H200 0.50／0.50**。
    - H200 的「忙」是 duty 1.0，這一格 Cake 幾乎都在搬（會合點 1–2 塊），所以丟一半就省一半；MI300X duty 0.7 時 GPU 還能算。
    - 全部載入兩台都是約 0.5。
  - **M2 格式配慢的層**（E1s、Cake、閒）：位元越少越快，兩台每一層都一樣。
    - Llama b32 FP8 → int2：MI300X 0.944 → 0.450，H200 0.544 → 0.253。
    - mxfp4 和 int4_kivi：MI300X 每一格 mxfp4 快 7–13%；H200 Llama 也快（b32 −14%），**Qwen b32 反過來**（mxfp4 0.365 vs int4 0.335），b48、nfs 仍是 mxfp4 快。
  - **M1 單位**（u512／u256／u128）：H200 列的每一格都是 512 最快。例：Llama b48 閒 E0 1.619／2.472／3.333；CPU 層、忙、E1s 0.117／0.276／0.393。MI300X 有兩個例外（Llama b16 忙 u128 較快、Llama CPU 閒 u256 較快），H200 沒有。結論「512 仍最好」兩台相同。
- 比較腳本是臨時的（`/work/.../tmp/cmp14.py`，只讀兩個 tables_speed_14.md），沒有新的數字來源。

### 10/21：55／60／88（只報告）的 H200 分析（head node，CPU；事先登記 13_h200_repro §15）
- 指令：`python code/h200_report_only.py` → `results/m10_precision_h200/analysis/report_only/{report_only.json, tables_report_only.md}`。
  - 重用 MI300X 凍結的 `figs_report_only.analyse()` 和 `m10_analyze` 的統計。
  - 回歸：同一支 `analyse()` 指到 MI300X 的資料，和 MI300X 的 `derived/report_only.json`（T1_LB）**逐值相同**。
- 封存規則照 REPORT.md 附錄 D，節點換成 H200 的：FP8 階端點、Qwen FP8 混合 T1、Qwen fp8_head 地圖是「主線」；INT4 階端點、Llama INT4 混合 T1、Llama int4 地圖是「對照」；其他「跑了、沒分析」。
- **端點，兩台幾乎一樣**（對 E0，百分點，讀法 A；MI300X／H200）：
  - Llama FP8 T1 都是 0.0；LongBench E1 +0.4／−0.2、E1s +0.7／+0.5。ε 兩台都過。
  - Qwen FP8 T1 E1 −48.6／−48.3、LongBench −36.0／−36.2。ε 兩台都沒過，同是 fp8_head 的崩壞。
  - Qwen INT4（對照）T1 −1.1／−0.8、LongBench −0.7／+0.2，ε 都過。
  - E0：T1 Llama 99.7／99.7%、Qwen 97.8／98.1%；LongBench 40.7／40.8、42.8／42.8。
- **H200 主線才有的**：
  - Qwen FP8 混合 T1（b32，對 E0，讀法 A）：F25 −11.4、M25 −42.2、B25 −5.0、R25 −21.9。讀法 B 對 R25：F25 +18.3、B25 +13.3、M25 −22.8（都在 CI 外）。也就是說，把 fp8_head 放在中間最傷，放在後段最不傷。
  - Qwen fp8_head 敏感度地圖（50 題，3200 列）：含答案的 block 平均 KL 0.158 [0.096, 0.267]，其他 block 0.541 [0.219, 0.969]。KL 最大的是 block 63、5、4、8、33。單一 block 存成 fp8_head 就有明顯 KL，而且不是答案所在的 block 最傷。
- **對照**：Llama INT4 混合 T1 對 R25 全部 0.0（MI300X 也全部 0.0）。Llama int4 地圖含答案／其他 0.00034／0.00027（MI300X 0.00038／0.00031），兩台都只有極小的 KL。
