# 10/19 13:3xZ：重量 44ca 的 Llama 128K INT4 格子（`20261019-104723-m10-lc-grid-llama-129024-int4_kivi`）。量時間 → 兩張卡獨佔。
# 原因：我推 git 時用了 `git stash -u`。worker 每個 launch 開著 restores.csv 一直寫，那個檔案被 git 追蹤，stash 換掉檔案後的列寫進了孤兒檔、關檔時消失：
#   L0 的 GPU 忙少了 154 列（11:35–12:05Z）、L1 的 GPU 忙少了 22 列（13:19–13:21Z）。資料本身沒錯，但少了 reps → 照規則整格重量（RUNLOG 15.12）。
# 指令和 44ca 完全相同（同一個 out-dir；latest_rows 只讀最新的 run）。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
LC=$REPO/results/m10_longctx_mi300x
m=llama; L=129024; G=$(fmt_of $m fp8); F=$(fmt_of $m int4); D=$LC/speed_${m}_${L}_$F
m7run m10-lc-grid-$m-$L-$F-rerun bash -c "python $REPO/code/m10_speed.py grid --model $m --fmt $F --base-fmt $G --out-dir $D --bw-json $OUT/speed_$m/bw.json --main-length $L --lengths --launches 2 --reps 5 --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" \
  || echo "44cab：Llama 128K INT4 重量失敗（見 runs/*-m10-lc-grid-$m-$L-$F-rerun）" >> $JOBS/NEEDS_HUMAN
exit 0
