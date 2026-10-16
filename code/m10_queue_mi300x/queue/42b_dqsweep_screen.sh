# 09 §6.4 反量化的方法掃描（只報告）：A 段 kernel 微測（挑 K*）→ B 段還原方法海選（int4_kivi、CPU 層、閒／忙、E0 E1 R25、1 launch × 10）
# → m10_dqsweep.py plan 照事先寫的分數挑複賽選手、寫 42c。任何一個方法失敗只寫 NEEDS_HUMAN，不擋其他方法；exit 0。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
SW=$REPO/results/m10_sweep_mi300x/dqsweep; mkdir -p $SW/A $SW/B
for m in llama qwen; do
  m7run m10-dqs-A-$m python $REPO/code/m10_dqsweep.py kernels --model $m --out-dir $SW/A \
    || echo "dqsweep A $m 失敗或監控報污染（K* 退回 k1；見 runs/*-m10-dqs-A-$m）" >> $JOBS/NEEDS_HUMAN
done
for m in llama qwen; do
  G=$(fmt_of $m fp8)
  for meth in $(python3 $REPO/code/m10_dqsweep.py list B); do
    ( export $(python3 $REPO/code/m10_dqsweep.py env $meth --model $m --fmt int4_kivi)
      m7run m10-dqs-B-$m-$meth bash -c "python $REPO/code/m10_speed.py grid --model $m --fmt int4_kivi --base-fmt $G --out-dir $SW/B/${m}_$meth --bw-json $OUT/speed_$m/bw.json --bw-names cpu --configs E0 E1 R25 --refs --lengths --launches 1 --reps 10 --launch0 100 --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" ) \
      || echo "dqsweep B $m $meth 失敗（見 runs/*-m10-dqs-B-$m-$meth）" >> $JOBS/NEEDS_HUMAN
  done
done
python3 $REPO/code/m10_dqsweep.py plan --sw $SW --jobs $JOBS || echo "dqsweep plan 失敗（42c 沒寫）" >> $JOBS/NEEDS_HUMAN
exit 0
