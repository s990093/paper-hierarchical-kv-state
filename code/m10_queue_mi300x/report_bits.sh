#!/usr/bin/env bash
# 報告用的表與圖（只排版、不重算判準；CPU）。用法：report_bits.sh speed <model> <fp8|int4> | step1 | k3 <model> | int4
set -euo pipefail
REPO=/mlsteam/workspace/paper-hierarchical-kv-state; OUT=$REPO/results/m10_precision_mi300x; FIG=$OUT/figs
source $REPO/code/m7_env.sh >/dev/null
T="python $REPO/code/m10_report_tables.py"; F="python $REPO/code/m10_figs.py"
mkdir -p $FIG
lastgrid() { python3 - "$1" "$2" "$3" <<'PY'
import json,sys
m,tier,suf=sys.argv[1],sys.argv[2],sys.argv[3]
d=json.load(open(f"/mlsteam/workspace/paper-hierarchical-kv-state/results/m10_precision_mi300x/speed_{m}{suf}/k0_verdict_{tier}.json"))
print(d["run_id"])
PY
}
case $1 in
speed)
  # speed <m> <tier> [後綴，例 _regrid：讀 speed_<m>_regrid/、寫 analysis/<tier>_regrid/、圖檔名加後綴]
  m=$2; tier=$3; SUF=${4:-}; S=$OUT/speed_$m$SUF; A=$OUT/analysis/$tier$SUF; mkdir -p $A
  RID=$(lastgrid $m $tier "$SUF"); RS=$S/restores.csv
  [ -f $RS ] || RS=$RS.gz
  { echo "## $m $tier 速度（run $RID）"; echo; $T k0 --json $S/k0_verdict_$tier.json; echo; $T speed --summary $S/summary_$tier.csv; echo;
    $T sources --restores $RS --run $RID; } > $A/tables_speed_$m.md
  $F speed --model $m --restores $RS --run $RID --out $FIG/speed_${m}_$tier$SUF.png
  for bw in b16 b32 b48 cpu; do $F sources --model $m --restores $RS --run $RID --bw $bw --out $FIG/sources_${m}_${tier}${SUF}_$bw.png; done
  echo "wrote $A/tables_speed_$m.md" ;;
step1)
  A=$OUT/analysis/fp8
  for m in llama qwen; do
    d=$(python3 -c "import json;print(json.load(open('$OUT/speed_$m/noise.json'))['delta_star'])")
    { echo "## $m Step 1（FP8 端點）"; echo; $T quality --json $A/quality_${m}_step1.json; echo; echo "### E0 各格"; echo; $T e0 --json $A/quality_${m}_step1.json; } > $A/tables_step1_$m.md
    SB=$OUT/speed_$m/summary_fp8.csv; [ $m = qwen ] && SB=$OUT/speed_qwen_regrid/summary_fp8.csv   # Qwen 的速度用重量那次（stage 22）
    $F A --quality $A/quality_${m}_step1.json --summary $SB --delta-star $d --out $FIG/A_${m}_fp8_step1.png
    $F C --model $m --csv $OUT/quality/step1_${m}_*.csv --configs E1 E1s E1sr --out $FIG/C_${m}_fp8_step1.png
    $F D --model $m --csv $OUT/quality/step1_${m}_*.csv --configs E0 E1 E1s E1sr --out $FIG/D_${m}_fp8_step1.png
  done
  $T decide --json $A/decide_fp8.json > $A/tables_decide_fp8.md; cat $A/tables_decide_fp8.md ;;
k3)
  m=$2; A=$OUT/analysis/fp8
  { echo "## $m Step 2b（換格式）"; echo; $T quality --json $A/quality_${m}_step2b.json; echo; $T k3 --json $A/k3_$m.json; } > $A/tables_k3_$m.md
  cat $A/tables_k3_$m.md ;;
int4)
  A=$OUT/analysis/int4
  for m in llama qwen; do
    { echo "## $m INT4 端點"; echo; $T quality --json $A/quality_${m}_int4.json; } > $A/tables_int4_$m.md
  done
  $T decide --json $A/decide_int4.json > $A/tables_decide_int4.md; cat $A/tables_decide_int4.md ;;
*) echo "用法見檔頭"; exit 2 ;;
esac
