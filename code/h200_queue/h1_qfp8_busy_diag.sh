# H200 才有的 stage（13_h200_repro §14 A）：查「Qwen fp8_head 在 GPU 忙時，只搬不拆也慢」。只報告。量時間的 stage。exit 0。
#   A1 極小 H2D 拷貝的微測（h200_tinycopy.py；閒／忙交替兩輪）
#   A2 2×2 交叉：模型（Llama／Qwen）× FP8 粒度（fp8_token／fp8_head），只搬不拆（指令同 43e (a)），CPU 層、2 launch × 5 rep
#   A3 nsys：Qwen fp8_head 只搬不拆，閒、忙各一次（1 launch × 3 rep，只有 E1；nsys 有負擔，時間只看相對）
source __JOBS__/lib.sh
set -uo pipefail
D=$REPO/results/m10_diag_h200; mkdir -p $D
m7run m10h-diag-tinycopy python $REPO/code/h200_tinycopy.py --out-dir $D \
  || echo "h1：A1 極小拷貝微測失敗（runs/*-m10h-diag-tinycopy）" >> $JOBS/NEEDS_HUMAN
for mf in "qwen fp8_head" "qwen fp8_token" "llama fp8_token" "llama fp8_head"; do
  set -- $mf; m=$1; F=$2; G=$(fmt_of $m fp8)
  K=$(python3 $REPO/code/m10_dqsweep.py env S0 --model $m --fmt $F)
  m7run m10h-diag-nodq-$m-$F env $K M10_MEET=none M10_DQ_SCHED=nodq bash -c "python $REPO/code/m10_speed.py grid --model $m --fmt $F --base-fmt $G --out-dir $D/nodq_${m}_$F --bw-json $OUT/speed_$m/bw.json --bw-names cpu --configs E0 E1 --refs --lengths --launches 2 --reps 5 --launch0 300 --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" \
    || echo "h1：A2 只搬不拆 $m $F 失敗" >> $JOBS/NEEDS_HUMAN
done
NSYS=$(command -v nsys || ls /usr/local/cuda/bin/nsys 2>/dev/null || true)
if [ -z "$NSYS" ]; then
  echo "h1：找不到 nsys → A3 不跑" | tee -a $JOBS/NEEDS_HUMAN
else
  K=$(python3 $REPO/code/m10_dqsweep.py env S0 --model qwen --fmt fp8_head)
  for st in idle busy; do
    m7run m10h-diag-nsys-qwen-fp8_head-$st env $K M10_MEET=none M10_DQ_SCHED=nodq bash -c "$NSYS profile -t cuda -s none --cpuctxsw none -f true -o \$TIARA_RUNS/\$RUN_ID/trace python $REPO/code/m10_speed.py grid --model qwen --fmt fp8_head --base-fmt fp8_head --out-dir \$TIARA_RUNS/\$RUN_ID/grid --bw-json $OUT/speed_qwen/bw.json --bw-names cpu --configs E1 --refs --lengths --gpu-states $st --launches 1 --reps 3 --launch0 300 && $NSYS stats -r cuda_gpu_trace -f csv -o \$TIARA_RUNS/\$RUN_ID/trace \$TIARA_RUNS/\$RUN_ID/trace.nsys-rep && python $REPO/code/h200_nsys_memcpy.py \$TIARA_RUNS/\$RUN_ID/trace_cuda_gpu_trace.csv --label $st --out $D/nsys_qwen_fp8_head_$st.json" \
      || echo "h1：A3 nsys $st 失敗（runs/*-m10h-diag-nsys-qwen-fp8_head-$st）" >> $JOBS/NEEDS_HUMAN
  done
fi
exit 0
