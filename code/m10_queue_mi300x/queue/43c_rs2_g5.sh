# 13 §4（Q2）：會合規則 G5（開始前估計，全部載完比重算第 0 塊快就不重算）。K*、S0。
# 同一個 process 裡交錯跑 g5（cake 列）、claim（照 Cake）、CPU 層另加 loadall。CPU 層 3 launch × 10 rep；限速層 b16 b32 b48 2 launch × 5 rep。
# 計時的 stage：跑的時候不在 GPU 上插任何東西。失敗寫 NEEDS_HUMAN、跑下一個；exit 0。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
R=$REPO/results/m10_restore2_mi300x/q2; mkdir -p $R
for m in llama qwen; do
  G=$(fmt_of $m fp8)
  for tag in fp8 int4; do
    F=$(fmt_of $m $tag); K=$(python3 $REPO/code/m10_dqsweep.py env S0 --model $m --fmt $F)
    BASE="python $REPO/code/m10_speed.py grid --model $m --fmt $F --base-fmt $G --bw-json $OUT/speed_$m/bw.json --configs E0 E1 R25 --refs --lengths --launch0 300 --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv"
    m7run m10-rs2-q2-$m-$tag-cpu env $K M10_MEET=g5 bash -c "$BASE --out-dir $R/${m}_${tag}_cpu --bw-names cpu --extra-modes loadall claim --launches 3 --reps 10" \
      || echo "43c：G5 $m $tag CPU 層失敗" >> $JOBS/NEEDS_HUMAN
    m7run m10-rs2-q2-$m-$tag-slow env $K M10_MEET=g5 bash -c "$BASE --out-dir $R/${m}_${tag}_slow --bw-names b16 b32 b48 --extra-modes claim --launches 2 --reps 5" \
      || echo "43c：G5 $m $tag 限速層失敗" >> $JOBS/NEEDS_HUMAN
  done
done
exit 0
