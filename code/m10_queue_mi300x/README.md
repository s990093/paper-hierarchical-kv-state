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
