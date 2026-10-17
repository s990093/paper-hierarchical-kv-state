# 13 §6（Q5，拆融合進 attention 的對照組）：(a) 只搬不拆的還原（meet none＋nodq；CPU 層、b32；閒／忙；E0 E1；3 × 10）；
# (b) attention 微測（m10_attn_q.py bench：32K／64K／128K × 256／1 個 query × 閒／忙；43a 自測沒過就不跑 (b)）。只報告。計時的 stage。exit 0。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
R=$REPO/results/m10_restore2_mi300x; Q=$R/q5; mkdir -p $Q
for m in llama qwen; do
  G=$(fmt_of $m fp8)
  for tag in fp8 int4; do
    F=$(fmt_of $m $tag); K=$(python3 $REPO/code/m10_dqsweep.py env S0 --model $m --fmt $F)
    m7run m10-rs2-q5a-$m-$tag env $K M10_MEET=none M10_DQ_SCHED=nodq bash -c "python $REPO/code/m10_speed.py grid --model $m --fmt $F --base-fmt $G --out-dir $Q/nodq_${m}_$tag --bw-json $OUT/speed_$m/bw.json --bw-names cpu b32 --configs E0 E1 --refs --lengths --launches 3 --reps 10 --launch0 300 --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" \
      || echo "43e：Q5 (a) $m $tag 失敗" >> $JOBS/NEEDS_HUMAN
  done
done
if [ -f $R/smoke/OK_attn ]; then
  for m in llama qwen; do
    m7run m10-rs2-q5b-$m python $REPO/code/m10_attn_q.py bench --model $m --lengths 32768 65536 129024 --queries 256 1 --out-dir $Q/attn \
      || echo "43e：Q5 (b) attention 微測 $m 失敗" >> $JOBS/NEEDS_HUMAN
  done
else
  echo "43e：43a 的 attention 自測沒過 → 不跑 (b)" | tee -a $JOBS/NEEDS_HUMAN
fi
exit 0
