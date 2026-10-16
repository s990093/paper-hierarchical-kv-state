# FP8 速度格子（8 配置 × 4 頻寬 × 閒／忙，3 launches × 10 reps，＋8K／16K）→ meets.json、summary、K0
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
for m in llama qwen; do
  G=$(fmt_of $m fp8)
  m7run m10-grid-fp8-$m bash -c "python $REPO/code/m10_speed.py grid --model $m --fmt $G --base-fmt $G --out-dir $OUT/speed_$m --bw-json $OUT/speed_$m/bw.json --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" || exit 1
  RID=$(lastrun m10-grid-fp8-$m)
  cp $OUT/speed_$m/summary.csv $OUT/speed_$m/summary_fp8.csv
  m7run m10-k0v-fp8-$m python $REPO/code/m10_speed.py k0-verdict --model $m --out-dir $OUT/speed_$m --from-run $RID --delta-json $OUT/speed_$m/noise.json || exit 1
  cp $OUT/speed_$m/k0_verdict.json $OUT/speed_$m/k0_verdict_fp8.json
  python3 $REPO/code/m10_analyze.py speed --model $m --summary $OUT/speed_$m/summary_fp8.csv --noise $OUT/speed_$m/noise.json --out-dir $OUT/analysis/fp8 || exit 1
done
