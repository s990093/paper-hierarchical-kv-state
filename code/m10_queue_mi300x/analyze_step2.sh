#!/usr/bin/env bash
# Step 2（FP8 階的混合配置；穩定差距第 1、2 條、K2）：只在 decide／k3 說要做 Step 2 的模型上跑（03 §1.1）。用法：analyze_step2.sh <model>
# 品質：混合配置對 E0 的 ε（E0 列取 Step 1 同一批題，題號、seed 相同）；穩定差距：F25／M25／B25 對 R25。CPU 就好。
set -euo pipefail
m=$1
REPO=/mlsteam/workspace/paper-hierarchical-kv-state; OUT=$REPO/results/m10_precision_mi300x; A=$OUT/analysis/fp8
source $REPO/code/m7_env.sh >/dev/null
SP=$A/speed_$m.json; SB=$OUT/speed_$m/summary_fp8.csv
[ $m = qwen ] && { SP=$OUT/analysis/fp8_regrid/speed_qwen.json; SB=$OUT/speed_qwen_regrid/summary_fp8.csv; }   # Qwen 的速度用重量那次（stage 22）
python $REPO/code/m10_analyze.py quality --model $m --csv $OUT/quality/step1_${m}_T24_s*.csv $OUT/quality/step1_${m}_T5.csv \
  $OUT/quality_sealed/fp8mix_${m}_T24_s*.csv $OUT/quality_sealed/fp8mix_${m}_T5.csv --configs F25 M25 B25 R25 --tag step2 --out-dir $A
python $REPO/code/m10_analyze.py step2 --model $m --csv $OUT/quality_sealed/fp8mix_${m}_T24_s*.csv $OUT/quality_sealed/fp8mix_${m}_T5.csv \
  --speed $SP --summary $SB --tag seedA --out-dir $A
