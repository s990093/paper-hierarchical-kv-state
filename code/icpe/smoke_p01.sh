#!/bin/bash
# smoke_p01.sh — P0-1 的 GPU 冒煙（不量時間、不進 results/）：FA2／FA3／FA4／cuDNN 的接線在真模型上通不通、
# m10_speed grid 在 M7_ATTN=fa3＋S4＋nfs 層能不能跑完。輸出在 $TIARA_RUNS/$RUN_ID/smoke/。
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
exit $rc
