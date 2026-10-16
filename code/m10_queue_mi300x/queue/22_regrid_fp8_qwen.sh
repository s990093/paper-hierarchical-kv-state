# Qwen FP8 速度格子重量（CLAUDE.md §3：gpu_guard 報污染的 run 整批重量）。原 run 20261015-175050-m10-grid-fp8-qwen 的 L0
# 在 18:16:32 報 1 筆 pid -1（m7_busy 收尾那一秒；判讀見 RUNLOG Milestone 10）。輸出放另一個資料夾，不碰 speed_qwen/meets.json
# （stage 30 讀法 B 用那份——07 第 54 列：會合點只用沒被標污染的列；這次重量的會合點只拿來對照）。排在 30 前面：Qwen 的 K0／速度不輸以這次為準，
# 判定（decide_fp8）要等它。這次 M10_BUSY_TEARDOWN_PAUSE=1。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
m=qwen; G=$(fmt_of $m fp8); D=$OUT/speed_qwen_regrid; mkdir -p $D
m7run m10-regrid-fp8-$m bash -c "python $REPO/code/m10_speed.py grid --model $m --fmt $G --base-fmt $G --out-dir $D --bw-json $OUT/speed_$m/bw.json --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" || exit 1
RID=$(lastrun m10-regrid-fp8-$m)
cp $D/summary.csv $D/summary_fp8.csv; cp $D/spec_grid.json $D/spec_grid_fp8.json
m7run m10-k0v-regrid-fp8-$m python $REPO/code/m10_speed.py k0-verdict --model $m --out-dir $D --from-run $RID --delta-json $OUT/speed_$m/noise.json || exit 1
cp $D/k0_verdict.json $D/k0_verdict_fp8.json
python3 $REPO/code/m10_analyze.py speed --model $m --summary $D/summary_fp8.csv --noise $OUT/speed_$m/noise.json --out-dir $OUT/analysis/fp8_regrid || exit 1
