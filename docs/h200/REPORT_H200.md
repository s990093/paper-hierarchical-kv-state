# 平台 C（NVIDIA H200 @ 學校 Slurm 叢集）硬體實測報告

產生時間：2026-10-16T06:10:34+08:00　產生者：`code/h200_probe.sh`　節點：hgx-cn01　run：`/work/ChihChungHsu/hungweilai/tiara-runs/20261016-061012-h200-probe`

> **機器產生，不要手改。** 每一節標來源檔（原始明細在 run 目錄的 `probe/`，不進 git）。
> `NOT_MEASURED`＝沒量到，**不是 0 也不是估計值**。比照平台 B 的 `REPORT_MI300X.md`（03_plan_h200 §R0）。

## 0. §R0 通過條件

| 條件 | 結果 |
|:--|:--|
| GPU 是 H200 | True |
| matmul 算得對 | True |
| Slurm 可見性已判定 | True |
| **整體** | **PASS** |

## 1. 硬體（來源：`/work/ChihChungHsu/hungweilai/tiara-runs/20261016-061012-h200-probe/probe/gpu_fields.tsv`、`/work/ChihChungHsu/hungweilai/tiara-runs/20261016-061012-h200-probe/probe/nvidia-smi.txt`、`/work/ChihChungHsu/hungweilai/tiara-runs/20261016-061012-h200-probe/probe/nvidia-smi-topo.txt`、`/work/ChihChungHsu/hungweilai/tiara-runs/20261016-061012-h200-probe/probe/lscpu.txt`）

| 項目 | 值 |
|:--|:--|
| GPU | NVIDIA H200 |
| 記憶體 | 143771 MiB |
| compute capability | 9.0 |
| driver | 610.43.02 |
| CUDA（driver 支援的版本） | NOT_MEASURED |
| PCIe 世代（max） | 5 |
| PCIe 寬度（max） | 16 |
| PCIe 世代（量的當下） | 5 |
| PCI bus | 00000000:E5:00.0 |
| 這張卡的 NUMA affinity | NOT_MEASURED |
| 這張卡的 CPU affinity | NOT_MEASURED |
| job 裡 nvidia-smi 看得到幾張卡 | 1 |
| NVLink（只記錄） | GPU 0: NVIDIA H200 (UUID: GPU-f7a2875d-7c3f-fb2c-fc93-779412a6c48e) | Link 0: 26.562 GB/s | Link 1: 26.562 GB/s | Link 2: 26.562 GB/s |
| 節點上的卡數（scontrol Gres） | 8 |
| CPU | AMD EPYC 9554 64-Core Processor（2 socket、2 NUMA node）|

> pinned 記憶體要配在 GPU 所在的 NUMA 節點（03 §4.1）：上表的 NUMA affinity。

## 2. 配額（來源：`/work/ChihChungHsu/hungweilai/tiara-runs/20261016-061012-h200-probe/probe/cgroup_limits.txt`、`/work/ChihChungHsu/hungweilai/tiara-runs/20261016-061012-h200-probe/probe/affinity.txt`、`/work/ChihChungHsu/hungweilai/tiara-runs/20261016-061012-h200-probe/probe/shm.txt`）

| 項目 | **job 真的能用的** | 機器上看到的（會騙人） |
|:--|:--|:--|
| CPU | **16**（sched_getaffinity；cpuset 112-127） | lscpu CPU(s) = 128 |
| RAM | **NOT_MEASURED GiB**（NOT_MEASURED） | MemTotal = 1511.5 GiB |
| /dev/shm | 755.8 GiB | — |

## 3. Slurm 可見性（來源：`/work/ChihChungHsu/hungweilai/tiara-runs/20261016-061012-h200-probe/probe/gpu_guard_host.json`、`/work/ChihChungHsu/hungweilai/tiara-runs/20261016-061012-h200-probe/probe/nvidia-smi-L.txt`、`/work/ChihChungHsu/hungweilai/tiara-runs/20261016-061012-h200-probe/probe/scontrol_node.txt`、`/work/ChihChungHsu/hungweilai/tiara-runs/20261016-061012-h200-probe/probe/squeue_all.txt`）

| 項目 | 值 |
|:--|:--|
| SLURM_JOB_ID／節點／CUDA_VISIBLE_DEVICES | 20417／hgx-cn01／0 |
| job 看得到的卡／節點上的卡 | 1／8 |
| **gpu_guard.host_contention 的 level** | **UNOBSERVABLE_HOST** |
| squeue 看得到別人的 job | True |
| 量的當下，同節點的 job 數（含自己） | 8 |

> `UNOBSERVABLE_HOST`：同節點別人的負載看不到 → 時間數字的鄰居負載只能部分觀測（H200_SLURM §5.3），
> 靠配置順序隨機打散＋配對比較＋每個 run 的 `squeue -w` 快照。`QUIET` 只在 job 看得到整個節點時才可能出現。

## 4. 軟體（來源：`/work/ChihChungHsu/hungweilai/tiara-runs/20261016-061012-h200-probe/probe/python_stack.txt`、`/work/ChihChungHsu/hungweilai/tiara-runs/20261016-061012-h200-probe/probe/os.txt`、`/work/ChihChungHsu/hungweilai/tiara-runs/20261016-061012-h200-probe/probe/pip_freeze.txt`、`/work/ChihChungHsu/hungweilai/tiara-runs/20261016-061012-h200-probe/probe/network.txt`）

| 項目 | 值 |
|:--|:--|
| os | Ubuntu 24.04.3 LTS |
| kernel | Linux hgx-cn01 6.8.0-90-generic #91-Ubuntu SMP PREEMPT_DYNAMIC Tue Nov 18 14:14:30 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux |
| python | 3.12.15 /work/ChihChungHsu/hungweilai/.conda/envs/tiara/bin/python |
| torch | 2.13.0+cu130 cuda 13.0 cudnn 92000 |
| vllm | 0.28.0 |
| transformers | 5.19.0 |
| flash_attn | NOT_INSTALLED ModuleNotFoundError |
| triton | 3.7.1 |
| repo_head | bad42c7dce91e90758a0e4ce0f3effb0d832cbbb |
| pip_freeze_lines | 208 |
| 對外網路 https://huggingface.co | HTTP/2 200  |
| 對外網路 https://pypi.org | HTTP/2 200  |
| 對外網路 https://github.com | HTTP/2 200  |

## 5. 儲存（來源：`/work/ChihChungHsu/hungweilai/tiara-runs/20261016-061012-h200-probe/probe/storage_bench.csv`、`/work/ChihChungHsu/hungweilai/tiara-runs/20261016-061012-h200-probe/probe/mounts.txt`、`/work/ChihChungHsu/hungweilai/tiara-runs/20261016-061012-h200-probe/probe/m7_disk.tsv`）

只記錄，這週不用真實裝置（03 §5.1：SSD／NFS 一律是限速層）。

fio：NOT_MEASURED（fio 不在（沒有 sudo，不自己裝）→ 儲存 NOT_MEASURED）

- `$TIARA_RUNS` 在：/dev/md1（ext4）
- **建議的 `M7_DISK`**：`md1`（在 /proc/diskstats：yes）。TIARA_RUNS 所在的區塊裝置
  → 填進 `code/env_h200.sh` 的 `M7_DISK` 預設值，或在 job 裡 `export M7_DISK=...`。
- tmpfs（shm）那一列是對照組（純記憶體速度的上界）；任何位置的讀取 ≥ tmpfs，必然是量到快取。

## 6. GPU smoke（來源：`/work/ChihChungHsu/hungweilai/tiara-runs/20261016-061012-h200-probe/probe/gpu_smoke.json`）

| 項目 | median | min | max |
|:--|--:|--:|--:|
| BF16 matmul TFLOPS | 788.124 | 754.661 | 790.458 |
| HBM 裝置內複製 GB/s（讀＋寫） | 4173.948 | 4154.927 | 4180.372 |
| pinned H2D GiB/s（1 GiB） | 53.551 | 53.519 | 53.559 |
| pinned D2H GiB/s（1 GiB） | 50.041 | 50.02 | 50.048 |
| 正確性 max\|I@Y−Y\| | 0.0（OK） | | |

torch 2.13.0+cu130、CUDA 13.0、NVIDIA H200、sm_90、torch 看得到 1 張

> ⚠️ smoke test，不是 benchmark：單一 stream、沒調 BLAS、每項 10 次。**不可寫進論文。**
> 正式的 H2D／D2H 是 R2 的 C0（`code/m7_calib.py c0`）。限速層的三個頻寬都要明顯低於這裡的 pinned H2D（03 §4.1）。

