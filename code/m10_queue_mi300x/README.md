# m10 GPU 佇列（MI300X 主實驗）

佇列的工作目錄在 scratch（job 在 scratch 執行，避免 abort 時 21 GB 的 gpucore 落進 repo）；這裡是腳本副本，跑完會再同步一次。
- `runner.sh`：依名稱順序跑 `queue/NN_*.sh`，每段 `flock gpu.lock`，失敗就停。
- `lib.sh`：共用環境（`m7run`、`$OUT`、`fmt_of`、`lastrun`、`pool`）。
- `pool.py`：同一段裡開多個品質 process，看 amd-smi 剩餘 VRAM、間隔 75 s 才開下一個。
- `analyze_*.sh`：只用 CPU 的判定（跑完一段後手動執行，不拿 GPU 鎖）。

**10/15 19:2xZ 起**（RUNLOG Milestone 10 §10.4）：
- 加 22（Qwen FP8 格子重量）、35（FP8 混合配置，封存）、88（從 55 拆出的 INT4 T1／LB）。
- 原 40／50／51 改名 80／85／86，排到最後。
- `pool.py` 讀 `pool_skip.txt`（名稱前綴）略過判定流程走不到的封存工作。
- `lib.sh` 打開 `M10_BUSY_TEARDOWN_PAUSE=1`。

**10/15 21:5xZ 起**：31（Step 1 判完 → `route_after_step1.py` 只調順序）、31z（大掃描冒煙）、40／42（大掃描，09）。

**10/16 03:4x–03:5xZ**（RUNLOG §10.8）：
- Step 1 判完：Llama K1、Qwen K3 → 往 INT4。80／85／86 → 33a／33b／33c，35 → 45（`route_after_step1.py` 原本 K3 分支寫成「整個停」，已改）。
- 33bz：INT4 端點判完 → `route_after_int4.py` 只調順序。
- 39：修好 `m10_codec_x.install()` 的遞迴後重跑冒煙；失敗 → 40–42 搬到 `hold/`。
- 45、55 改讀 `speed_<m>/meets_fp8tier.json`（33a 寫入前的會合點快照）。
- `queue/31z_sweep_smoke.sh`、`33a_grid_int4.sh` 是跑時的版本（`.done`／`.running` 複製出來）。
- 04:0x–04:2xZ 長 context（10_longctx_plan）：39a 冒煙（大掃描前）；90 速度、91 試跑 → `longctx_plan_n.py` 寫 92（所有 32K 工作之後）。

**10/16 10:29Z**（RUNLOG §10.10）：33bz 判「INT4 階：Llama 均勻不夠 → Step 2 在 Llama 上做」→ `route_after_int4.py` 把 33c 改成只跑 llama（`queue/33c_quality_int4_mixed.sh` 是改過、跑時的版本），寫出 `46_quality_int4_mixed_qwen.sh`（只報告）。
