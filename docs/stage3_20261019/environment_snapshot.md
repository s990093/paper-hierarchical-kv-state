# 環境快照（第三階段前置探索，MI300X @ MLSteam）

> **取樣時間**：2026-10-19 13:01–13:34Z（台灣 21:01–21:34）。只用唯讀指令（`amd-smi`、`lscpu`、`numactl`、`df`、`mount`、`pip list`）；沒有開任何 GPU 行程。
> 舊的指紋檔：`results/env_mi300x.json`、`results/hw_mi300x.json`（10/19 02:26Z，第二張卡掛上後）。

## 1. 時間與租約

| 項目 | 值 |
|:--|:--|
| 伺服器時區 | UTC（`/etc/timezone`＝Etc/UTC） |
| 租約結束 | **2026-10-23 17:00 台灣＝09:00Z**（使用者 10/19 確認） |
| 第三階段收尾 | 10/23 02:00Z 後不開新的長實驗；02–09Z 整理、commit、push、備份 |
| 第二階段佇列 | 另一個 Claude session 的兩條 runner（main 卡 1、helper 卡 0）；預估台灣 10/20 晚上跑完（該 session 10/19 13:2xZ 的訊息） |

## 2. 硬體

| 項目 | 值 | 出處 |
|:--|:--|:--|
| GPU | 2 × AMD Instinct MI300X（gfx942），192 GB HBM3，SPX／NPS1，304 CU | `amd-smi list`、`hw_mi300x.json` |
| GPU 編號 | amd-smi 0＝`0000:86:00.0`（10/19 新掛）、1＝`0000:a6:00.0`（10/18 以前所有計時用的那張）；HIP 編號與 amd-smi 一一對應 | RUNLOG_MI300X 15.1 |
| GPU 連接 | PCIe Gen5 x16（32 GT/s）；GPU 在 NUMA node 1 | `hw_mi300x.json` |
| CPU | AMD EPYC 9684X 96 核 × 2 socket（主機 192 執行緒）；**容器配額 32 核**（`cpu.max` 3200000/100000） | `lscpu`、cgroup |
| NUMA | 2 個 node（0：CPU 0–95；1：CPU 96–191），距離 10／32 | `numactl --hardware` |
| 記憶體 | 主機 2,267 GiB；**容器上限 434.8 GiB**；`/dev/shm` 178.8 GiB | `free -g`、`hw_mi300x.json` |
| 本地磁碟 | overlay 3.0 T（背後 sda，GBT3916-MR-32PD 3.5 T，非旋轉）；**看不到 /dev/nvme\***；第一階段實測讀 7,943–9,152 MiB/s（`/var/tmp`） | `lsblk`、`df`、`stage2_factcheck_and_toolbox.md` §1 第 10 條 |
| NFS | `/mlsteam/workspace`、`/mlsteam/data/tiara`：nfs4.1，rsize／wsize 64 KiB，192.168.11.202；第一階段實測讀 377–447 MiB/s | `mount`、同上 |
| 沒有的硬體 | QAT、IAA、DSA、AMX 的對應物（EPYC）；GPUDirect Storage／hipFile 的條件（NVMe、ROCm ≥7.14） | `lit/L0` §3 |

## 3. 軟體

| 項目 | 值 |
|:--|:--|
| OS／kernel | Ubuntu 24.04.4 LTS；6.8.0-138-generic |
| ROCm | 7.2.2（`/opt/rocm-7.2.2`）；amd-smi 26.2.2；amdgpu 6.16.13 |
| venv `tiara`（我們自己的 harness） | Python 3.12；torch 2.10.0+rocm7.0；transformers 5.17.0；triton 3.8.0（另有 triton-rocm 3.6.0 的記錄）；numpy 2.2.6；zstandard 可用；**lz4、blosc2、zipnn 沒有** |
| venv `tiara-v028`（所有 vLLM 實驗） | vLLM 0.28.0+rocm722（editable：`/mlsteam/workspace/src/vllm-v0.28.0`）；torch 2.12.1+rocm7.2；transformers 5.17.0；aiohttp 3.14.3；httpx 0.28.1；zstandard 0.25.0 |
| vLLM 0.28 的 KV connector／offload（原始碼目錄） | `offloading_connector.py`、`simple_cpu_offload_connector.py`、`lmcache_connector.py`（LMCache 本身沒裝）、`mooncake/`、`moriio/`、`nixl/`、`flexkv_connector.py`；`vllm/v1/kv_offload/{cpu, tiering, file_mapper.py}` |
| 已知的 ROCm 問題 | 接 KV connector 時 attention 從 ROCM_ATTN 退回 TRITON_ATTN（33K attention 1.367→3.277 s，RUNLOG_MI300X:176；`code/m9_f1_run.py:74` 有繞過的 patch）；vLLM CPU→GPU 實得約 4.1 GiB/s |
| 模型 | Llama-3.1-8B-Instruct（`/mlsteam/data/tiara/hf-cache/hub/models--unsloth--Llama-3.1-8B-Instruct`）、Qwen2.5-7B-Instruct-1M（vLLM 用 noDCA：`/mlsteam/data/tiara/models/Qwen2.5-7B-Instruct-1M-noDCA`） |
| 資料 | `/mlsteam/data/tiara/datasets/{hotpotqa, longbench, ruler_ref, scbench, traces（Mooncake conversation／toolagent／synthetic）, gsm8k, yakv_ref}` |

## 4. 取樣當下的負載（13:01–13:34Z）

| 項目 | 值 |
|:--|:--|
| GPU 0 | 0% 使用率、0.3 GB（被第二階段的 main 為量時間 stage 44ca 保留：`WANT_BOTH`） |
| GPU 1 | 71–100% 使用率、30.9–50.8 GB：第二階段 44ca（`m10_speed.py _worker`，Llama 128K INT4 KIVI 速度格子）＋背景負載 `m7_busy.py --duty 0.7` |
| CPU | loadavg 2.6–3.2（容器 32 核）；第二階段 worker 約 1.6 核、busy 約 0.7 核 |
| 其他使用者 | `amd-smi process` 只看得到本容器的行程（多租戶容器的監控限制，`docs/MI300X_MLSTEAM.md` §7.5.8）；沒有看到外來行程 |
| git | branch `main`，HEAD `1b9d8aa`（第二階段 session 持續在推）；第三階段的檔案都還沒 commit |

## 5. 和第二階段共用機器的協調（10/19 已約定）

- 第二階段先跑完（使用者決定）；第三階段的 runner（`code/m16_queue/runner.sh`）只在第二階段 `WANT_BOTH` 不在、兩個佇列都沒有 `*.sh`／`*.running` 時開 GPU job，拿同一組 `/mlsteam/data/tiara/gpu{0,1}.lock`，登記共用顯存帳 `vram_reserved.json`，每個 job 設 `HIP_VISIBLE_DEVICES`／`TIARA_GPU`。
- 第二階段計時期間：**任何行程不准開 /dev/kfd**（AMD 的 gpu_guard 用它判斷入侵者）→ 第三階段的 CPU 工作不 import torch；綁 0–3 號核心、`nice 19`（第二階段自己用 28–31）。
- git：不用 `git stash`／`pull --rebase --autostash`／`checkout --`（10/19 13:19Z 第二階段一次 stash 讓正在寫的 CSV 少了 176 列，已重量）；推送用 `code/m10_queue_mi300x/gitpush.sh`（fetch → merge → push）；只 add 第三階段自己的檔。
