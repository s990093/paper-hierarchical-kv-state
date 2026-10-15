#!/usr/bin/env bash
# Step 1（FP8 端點）的判定：照 07 凍結版、m10_analyze.py；CPU 就好，不拿 GPU 鎖
set -euo pipefail
REPO=/mlsteam/workspace/paper-hierarchical-kv-state; OUT=$REPO/results/m10_precision_mi300x; A=$OUT/analysis/fp8
source $REPO/code/m7_env.sh >/dev/null
mkdir -p $A
for m in llama qwen; do
  python $REPO/code/m10_analyze.py quality --model $m --csv $OUT/quality/step1_${m}_*.csv --configs E1 E1s E1sr --tag step1 --out-dir $A
done
# Qwen 的速度以重量那次為準（stage 22；原 run 175050 的 L0 被 gpu_guard 標污染，CLAUDE.md §3 整批重量）。原 run 的版本另存對照。
SQ=$OUT/analysis/fp8_regrid/speed_qwen.json
[ -f $SQ ] || { echo "Qwen 速度重量（stage 22）還沒有結果：$SQ"; exit 1; }
python $REPO/code/m10_analyze.py decide --tier fp8 --llama $A/quality_llama_step1.json --qwen $A/quality_qwen_step1.json \
  --speed-llama $A/speed_llama.json --speed-qwen $SQ --out-dir $A
python $REPO/code/m10_analyze.py decide --tier fp8 --llama $A/quality_llama_step1.json --qwen $A/quality_qwen_step1.json \
  --speed-llama $A/speed_llama.json --speed-qwen $A/speed_qwen.json --out-dir $OUT/analysis/fp8_origspeed > /dev/null
