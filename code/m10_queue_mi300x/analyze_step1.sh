#!/usr/bin/env bash
# Step 1（FP8 端點）的判定：照 07 凍結版、m10_analyze.py；CPU 就好，不拿 GPU 鎖
set -euo pipefail
REPO=/mlsteam/workspace/paper-hierarchical-kv-state; OUT=$REPO/results/m10_precision_mi300x; A=$OUT/analysis/fp8
source $REPO/code/m7_env.sh >/dev/null
mkdir -p $A
for m in llama qwen; do
  python $REPO/code/m10_analyze.py quality --model $m --csv $OUT/quality/step1_${m}_*.csv --configs E1 E1s E1sr --tag step1 --out-dir $A
done
python $REPO/code/m10_analyze.py decide --tier fp8 --llama $A/quality_llama_step1.json --qwen $A/quality_qwen_step1.json \
  --speed-llama $A/speed_llama.json --speed-qwen $A/speed_qwen.json --out-dir $A
