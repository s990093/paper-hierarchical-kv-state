#!/bin/bash
# smoke_p01.sh — ICPE 的 GPU 冒煙（不量時間、不進 results/）：P0-1 的 FA2／FA3／FA4／cuDNN 接線、m10_speed grid 在 fa3＋S4＋nfs；
# 10/22 加：P0-2 的 GPU 重播（LRU、k＝8、前 50 個請求）、P1-1 的 loadpath（--smoke，4 個 block）。輸出在 $TIARA_RUNS/$RUN_ID/smoke/。
# 用法（在 job 裡，經 tiara.sbatch 包成一個 run）：sbatch --job-name=icpe-p01-smoke --time=0:45:00 code/tiara.sbatch bash code/icpe/smoke_p01.sh
set -uo pipefail
cd "$REPO"
S=$TIARA_RUNS/$RUN_ID/smoke; mkdir -p "$S"
export M10_SWEEP_DIR=$REPO/results/m10_sweep_h200 M10_PRECISION_DIR=$REPO/results/m10_precision_h200
export M10_BUSY_LEVELS=$REPO/results/m7_write_policy_h200/busy_levels.json
rc=0
for m in llama qwen; do
  python code/icpe/attn_bench.py run --model $m --out-dir "$S/attn" --lens 2048 8192 --n-speed 8 --rounds 1 || { echo "attn $m exit $?"; rc=1; }
done
python code/icpe/attn_bench.py choose --out-dir "$S/attn" || { echo "choose exit $?"; rc=1; }
python code/icpe/p01_tools.py bw-nfs --out-dir "$S/grid" || rc=1
K4=$(python code/m10_dqsweep.py env S4 --model llama --fmt int4_kivi)
env $K4 M7_ATTN=fa3 python code/m10_speed.py grid --model llama --fmt int4_kivi --base-fmt fp8_token --bw-json "$S/grid/bw_llama.json" \
  --configs E0 E1 --bw-names cpu nfs --gpu-states idle --extra-modes loadall --refs --lengths --launches 1 --reps 1 --launch0 900 \
  --out-dir "$S/grid/llama_s4_fa3" || { echo "grid fa3 exit $?"; rc=1; }
K0=$(python code/m10_dqsweep.py env S0 --model llama --fmt int4_kivi)
env $K0 M7_ATTN=fa3 M10_MEET=g5b python code/m10_speed.py grid --model llama --fmt int4_kivi --base-fmt fp8_token --bw-json "$S/grid/bw_llama.json" \
  --configs E0 E1 --bw-names cpu --gpu-states busy --extra-modes loadall claim --refs --lengths --launches 1 --reps 1 --launch0 900 \
  --out-dir "$S/grid/llama_g5b_fa3" || { echo "grid g5b fa3 exit $?"; rc=1; }
# P0-2：x1_gpu 的 GPU 重播（前 50 個請求；選擇檔 k＝8）
P3_CONST=h200 X1_TRACE_DIR=$TIARA_HOME/datasets/traces python code/icpe/x1_gpu.py gpu --out "$S/p02" --k 8 --cpu 128 --policy HINT-PF \
  --max-req 50 --csv gpu_smoke.csv || { echo "x1_gpu smoke exit $?"; rc=1; }
# P1-1：loadpath 冒煙（兩個模型）
for m in llama qwen; do
  python code/icpe/p11_time.py loadpath --model $m --out "$S/p11" --reps 1 --smoke || { echo "p11 loadpath $m exit $?"; rc=1; }
done
exit $rc
