# Step 2b 的速度（E1-K0＝E1s@fp8k0_<粒度>、E1-INT8＝E1s@int8_token），只有 E1s 一個配置；meets 鍵是 E1s@<格式>
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
for m in llama qwen; do
  G=$(fmt_of $m fp8); g=${G#fp8_}
  for F in fp8k0_$g int8_token; do
    m7run m10-grid-2b-$F-$m bash -c "python $REPO/code/m10_speed.py grid --model $m --fmt $F --base-fmt $G --configs E1s --refs --lengths --out-dir $OUT/speed_$m --bw-json $OUT/speed_$m/bw.json --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" || exit 1
    cp $OUT/speed_$m/summary.csv $OUT/speed_$m/summary_2b_$F.csv
  done
done
