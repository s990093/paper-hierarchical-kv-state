# 第 2 階（INT4）的速度格子（先跑；判定流程走到才分析）
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
for m in llama qwen; do
  G=$(fmt_of $m fp8); I=$(fmt_of $m int4)
  m7run m10-grid-int4-$m bash -c "python $REPO/code/m10_speed.py grid --model $m --fmt $I --base-fmt $G --out-dir $OUT/speed_$m --bw-json $OUT/speed_$m/bw.json --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" || exit 1
  RID=$(lastrun m10-grid-int4-$m)
  cp $OUT/speed_$m/summary.csv $OUT/speed_$m/summary_int4.csv
  m7run m10-k0v-int4-$m python $REPO/code/m10_speed.py k0-verdict --model $m --out-dir $OUT/speed_$m --from-run $RID --delta-json $OUT/speed_$m/noise.json || exit 1
  cp $OUT/speed_$m/k0_verdict.json $OUT/speed_$m/k0_verdict_int4.json
done
