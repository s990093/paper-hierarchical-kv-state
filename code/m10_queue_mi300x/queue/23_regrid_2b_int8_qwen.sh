# Step 2b 速度 E1s@int8_token（Qwen）重量：原 run 20261015-193516-m10-grid-2b-int8_token-qwen 的 L1 在 19:39:35.79 報 1 筆 pid −1
# （L1 最後一列「忙」19:39:35 之後、m7_busy 收尾那一刻；同 stage 20 Qwen L0 的機制；stage 21 刻意沒開修正）。CLAUDE.md §3 → 重量。
# 輸出到 speed_qwen_regrid/（meets.json 只多 E1s@int8_token 鍵；speed_qwen/meets.json 不動，stage 30 讀法 B 用那份，07 第 54 列）。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
m=qwen; G=$(fmt_of $m fp8); D=$OUT/speed_qwen_regrid; F=int8_token
m7run m10-regrid-2b-$F-$m bash -c "python $REPO/code/m10_speed.py grid --model $m --fmt $F --base-fmt $G --configs E1s --refs --lengths --out-dir $D --bw-json $OUT/speed_$m/bw.json --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" || exit 1
cp $D/summary.csv $D/summary_2b_$F.csv; cp $D/spec_grid.json $D/spec_grid_2b_$F.json
