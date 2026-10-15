#!/usr/bin/env bash
# 第 2 階（INT4）端點的判定：只有 FP8 階判定「不需要位置策略」（K1、K1s 或 K3）時才跑（07 §6 封存規則）
set -euo pipefail
REPO=/mlsteam/workspace/paper-hierarchical-kv-state; OUT=$REPO/results/m10_precision_mi300x; A=$OUT/analysis/int4
source $REPO/code/m7_env.sh >/dev/null
mkdir -p $A
for m in llama qwen; do
  I=$(python3 -c "import json;print(json.load(open('$OUT/fmt_select/fmt_choice.json'))['choice']['$m']['int4'])")
  python $REPO/code/m10_analyze.py quality --model $m --csv $OUT/quality/step1_${m}_*.csv $OUT/quality_sealed/int4end_${m}_*.csv \
    --strip-fmt $I --configs E1 E1s E1sr --tag int4 --out-dir $A
  python $REPO/code/m10_analyze.py speed --model $m --summary $OUT/speed_$m/summary_int4.csv --noise $OUT/speed_$m/noise.json --out-dir $A
done
python $REPO/code/m10_analyze.py decide --tier int4 --llama $A/quality_llama_int4.json --qwen $A/quality_qwen_int4.json \
  --speed-llama $A/speed_llama.json --speed-qwen $A/speed_qwen.json --out-dir $A
