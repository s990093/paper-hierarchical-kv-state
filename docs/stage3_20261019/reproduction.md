# 第三階段探索：怎麼重跑（指令與設定）

> 隨實驗進度補。每一步都照 CLAUDE.md §4.1 的記錄殼（`m7run`）：`/mlsteam/data/tiara/runs/<run_id>/` 有 cmd.sh、context.txt、stdout、stderr、exit_code。
> 平台：MI300X @ MLSteam（`environment_snapshot.md`）。其他平台要改的地方見最後一節。

## 0. 環境

```bash
cd /mlsteam/workspace/paper-hierarchical-kv-state
source code/m7_env.sh            # venv tiara（torch 2.10+rocm7.0）、HF_HOME、TIARA_RUNS、m7run；卡號＝${TIARA_GPU:-0}（有 /mlsteam/data/tiara/USE_ROCR 時設 ROCR_VISIBLE_DEVICES 並 unset HIP／CUDA，否則設 HIP／CUDA_VISIBLE_DEVICES；行程內一律 cuda:0）
# vLLM 實驗用另一個 venv：/mlsteam/workspace/venv/tiara-v028（vLLM 0.28.0+rocm722）
```

## 1. 只用 CPU 的部分（任何時候都能跑；第二階段計時期間要 `nice -n 19 taskset -c 0-3`，而且**不能 import torch**）

```bash
# 自測
python -I code/m16_s1_lossless.py --self-test          # S1 分析：25 項 ALL PASS
python3 -I code/m16_mcake.py                            # I1 排程核心：ALL PASS
# I1 事先預測〔模擬〕（run 20261019-134046-m16-i1pred）
m7run m16-i1pred nice -n 19 taskset -c 0-3 python -I code/m16_i1_predict.py --model llama \
  --out results/m16_stage3_mi300x/i1_concurrent_restore/prediction
```

## 2. GPU 部分（第三階段 runner）

### 2.1 runner

```bash
# 量時間的佇列 queue/（GPU-NEED: both；在卡 1；第二階段全空時獨佔兩張卡，或 XGPU 條件下用 WANT_CARD1 清空卡 1）
S3_ROLE=timing  setsid nohup bash code/m16_queue/runner.sh > /mlsteam/data/tiara/stage3/jobs/runner_timing.out 2>&1 &
# 品質的佇列 queue_q/（GPU-NEED: one；第二階段 main 空、沒有 WANT_CARD1 時在卡 1；helper 還在跑時 ≤ 96 GB）
S3_ROLE=quality setsid nohup bash code/m16_queue/runner.sh > /mlsteam/data/tiara/stage3/jobs/runner_quality.out 2>&1 &
# 放 job：把 code/m16_queue/jobs/<NN_name>.sh 複製到 /mlsteam/data/tiara/stage3/jobs/<queue>/（名稱照位元組排序執行）
# 停：touch /mlsteam/data/tiara/stage3/jobs/STOP
```

- 和第二階段的協調（RUNLOG_MI300X 15.13；`code/m16_queue/runner.sh` 檔頭）：量時間的 job 在第二階段兩個佇列都空時獨佔兩張卡；有 `XGPU_OK`、第二階段 main 空時改用 XGPU 條件（`WANT_CARD1` → 等卡 1 清空 → `gpu1.lock`，結果標「卡 0 同時有品質工作」，環境變數 `S3_XGPU_COND=1`）。品質 job 在第二階段 main 空時用卡 1。拿 `/mlsteam/data/tiara/gpu{0,1}.lock`；登記共用顯存帳 `vram_reserved.json`；`MAX-H` 超過 10/23 02:00Z 的不開。測試：`S2_JOBS=<假目錄> S3_JOBS=<假目錄> S3_LOCKDIR=<假目錄> bash code/m16_queue/runner.sh`（見 `experiment_log.md`）。
- 卡號變數（RUNLOG_MI300X 15.16）：第二階段的 `44d_xgpu_check2` 通過後會放 `/mlsteam/data/tiara/USE_ROCR`。之後 runner 與 `m16_vllm_bench.py serve` 都改設 `ROCR_VISIBLE_DEVICES`＝實體卡號、unset HIP／CUDA（只用 HIP 時行程會開兩張卡的 render 節點，amd-smi 兩張卡都列，對方的 guard 會當成入侵者）。測試：`M16_USE_ROCR_FLAG=<假檔> …/tiara-v028/bin/python code/tests/m16_bench_cpu.py`（有／沒有旗子都 ALL PASS）。**10/20 更新（RUNLOG_MI300X 15.17）**：ROCR 驗證也沒過（設了 ROCR 的行程一樣開兩張卡的 render 節點），所以 `USE_ROCR` 不會出現，切換程式留著但不會觸發，照 HIP 做法。第二階段改讓 `code/gpu_guard.py` 從 KFD sysfs（`/sys/class/kfd/kfd/proc/<pid>/vram_<gpu_id>`）讀每張卡分開的 VRAM；第三階段的 guard 全走 `gpu_guard`（`GpuWatcher`、`smi_index`、`host_contention`、`foreign_on`，介面沒變），自動用到新版。`ledger.py card_clear` 看的是整張卡的已用量（`amd-smi metric --mem-usage`），不受行程列表的問題影響。
- **10/20 起預設 `S3_AFTER_S2=1`**（使用者：「基本上是要排在第二階段後」）：任何 GPU job 都要第二階段兩個佇列都空、沒有 WANT_BOTH 才開；XGPU 條件與「helper 還在跑時用卡 1 跑品質」關掉。`S3_AFTER_S2=0` 才回到上面的寬鬆規則。假目錄測試 5 種情況都照預期（helper 忙時嚴格模式不開品質、不立 WANT_CARD1；全空時 timing／quality 都開；寬鬆模式 helper 忙時開品質）。
- job 檔頭：`# GPU-NEED: none|one|both`、`# GPU: 0|1`、`# NEED-GB: N`、`# MAX-H: h`。

### 2.1b 10/20 起：單一順序佇列＋cpu 佇列

```bash
# GPU 佇列（照 v0.4 時間表的順序；品質 job 用 CARDS: 0,1 在 job 內兩卡並行）
S3_ROLE=timing setsid nohup bash code/m16_queue/runner.sh > /mlsteam/data/tiara/stage3/jobs/runner_timing.out 2>&1 &
# CPU 分析佇列 queue_c/（nice 19、0–3 號核心；看到 CPU_QUIET 就先不開新的）
S3_ROLE=cpu    setsid nohup bash code/m16_queue/runner.sh > /mlsteam/data/tiara/stage3/jobs/runner_cpu.out 2>&1 &
```
- 檔頭新增 `CARDS: 0,1`（runner 兩張卡各登記 NEED-GB）與 `AFTER: <job 名>`（等那些 job 在任一佇列 .done）。`quality` 角色停用；舊的 `queue_q/` 兩個 job 移到 `retired_20261020/`，改名為 `05_s1_kvdump`（兩卡並行）與 `35_s1_cpubench`（排他、立 CPU_QUIET、綁 8–23 核）。
- 假目錄測試（10/20）：雙卡切片的登記與釋放、依賴（沒依賴的先跑、cpu job 完成後 GPU job 才開）、cpu 佇列收到 GPU job → failed、CPU_QUIET 擋住新 job，都照預期。

### 2.2 job 清單（`code/m16_queue/jobs/`）

| job | 做什麼 | 輸出 |
|:--|:--|:--|
| `00_smoke.sh` | 每支 GPU 程式最小設定跑一次 | `/mlsteam/data/tiara/stage3/smoke/`（不進 results） |
| `10_s1_kvdump.sh`（quality） | S1：存真的 KV＋各格式位元組（Llama／Qwen × T4／T2 × 32K；128K 選 5 層） | `/var/tmp/tiara_s3/kv_dumps/`（本地盤，可重生） |
| `11_s1_analyze.sh`（quality，拿卡 1 的鎖確保沒有量時間的 job 同時跑） | S1：壓縮比、熵、速度（1／4／8／16 thread）、盈虧點 | `results/m16_stage3_mi300x/s1_lossless/` |
| `20_i4_sdma0.sh`、`21_i4_sdma1.sh` | I4：搬法 × GPU 閒／busy2，SDMA 關／開 | `results/m16_stage3_mi300x/i4_copy_batching/` |
| `30_i1_interf.sh` | I1(a)＋I2：計算／搬運／decode 互相拖慢多少 | `results/m16_stage3_mi300x/i1_concurrent_restore/interf_llama.*` |
| `31_i1_grid.sh` | I1(b)：N 個 session 同時還原，實測 vs 預測 | `results/m16_stage3_mi300x/i1_concurrent_restore/grid_llama.*` |
| （待補） | C1a、S2（agent 寫中）；V0、F2、F1（vLLM，agent 寫中）；I3s（CPU 模擬，agent 寫中） | — |

## 3. 換平台要改的

- GPU 廠商相依：`code/gpu_guard.py`（amd-smi／nvidia-smi）、`HIP_VISIBLE_DEVICES`↔`CUDA_VISIBLE_DEVICES`；`HSA_ENABLE_SDMA`（I4）只在 ROCm 有意義。
- 限速層頻寬（b16／b32／b48）是**每個模型、每台機器**用 `m10_speed.py calib-f` 量出來的（`results/m10_precision_mi300x/speed_<m>/bw.json`），換機器要重量。
- runner 的第二階段路徑（`S2_JOBS`）只在這台 MI300X 的這次租約有意義；別的機器設成一個空目錄即可。
