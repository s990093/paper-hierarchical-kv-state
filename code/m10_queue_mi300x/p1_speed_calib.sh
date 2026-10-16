#!/usr/bin/env bash
# Phase 1（獨佔 GPU）：f(i)、頻寬選擇、雜訊 δ。兩個模型。
set -uo pipefail
REPO=/mlsteam/workspace/paper-hierarchical-kv-state
cd /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs
source $REPO/code/m7_env.sh
OUT=$REPO/results/m10_precision_mi300x
exec 9>/mlsteam/data/tiara/gpu.lock
flock 9
echo "[p1] lock acquired $(date -u +%FT%TZ)"
for m in llama qwen; do
  mkdir -p $OUT/speed_$m
  m7run m10-calibf-$m python $REPO/code/m10_speed.py calib-f --model $m --out-dir $OUT/speed_$m || { echo "calib-f $m FAILED"; exit 1; }
done
for m in llama qwen; do
  m7run m10-noise-$m bash -c "python $REPO/code/m10_speed.py noise --model $m --out-dir $OUT/speed_$m --bw-json $OUT/speed_$m/bw.json --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" || { echo "noise $m FAILED"; exit 1; }
done
echo "[p1] done $(date -u +%FT%TZ)"
