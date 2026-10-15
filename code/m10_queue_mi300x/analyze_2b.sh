#!/usr/bin/env bash
# Step 2b（K3）：只在 decide 說要做 Step 2b 的模型上跑。用法：analyze_2b.sh <model>
set -euo pipefail
m=$1
REPO=/mlsteam/workspace/paper-hierarchical-kv-state; OUT=$REPO/results/m10_precision_mi300x; A=$OUT/analysis/fp8
source $REPO/code/m7_env.sh >/dev/null
G=$(python3 -c "import json;print(json.load(open('$OUT/fmt_select/fmt_choice.json'))['choice']['$m']['fp8'])"); g=${G#fp8_}
python $REPO/code/m10_analyze.py quality --model $m --csv $OUT/quality/step1_${m}_*.csv --configs E1s@fp8k0_$g E1s@int8_token --tag step2b --out-dir $A
SB=$OUT/speed_$m/summary_fp8.csv
[ $m = qwen ] && SB=$OUT/speed_qwen_regrid/summary_fp8.csv     # Qwen 的混合配置中位數用重量那次（stage 22）
for F in fp8k0_$g int8_token; do
  XS=$OUT/speed_$m/summary_2b_$F.csv
  [ $m = qwen ] && [ $F = int8_token ] && XS=$OUT/speed_qwen_regrid/summary_2b_$F.csv   # 原 run L1 監控報污染 → stage 23 重量
  python $REPO/code/m10_analyze.py speed --model $m --summary $SB --noise $OUT/speed_$m/noise.json \
    --configs E1s@$F --x-summary $XS --x-from E1s --tag _2b_$F --out-dir $A
done
python $REPO/code/m10_analyze.py k3 --quality $A/quality_${m}_step2b.json --speed $A/speed_${m}_2b_fp8k0_$g.json $A/speed_${m}_2b_int8_token.json --out-dir $A
