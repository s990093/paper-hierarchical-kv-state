# 09 §6.3-2：融合反量化（M10_DEQUANT_FAST=1，和原寫法逐位元相同）之後，主線 8 個 FP8 配置在真實 CPU 層的還原時間。
# 只報告，不改主線判定（主線照 07 用原寫法）。32K、閒／忙、3 launch × 10 rep；gpu_guard 照常。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
SW=$REPO/results/m10_sweep_mi300x
for m in llama qwen; do
  G=$(fmt_of $m fp8); D=$SW/speed_dqfast_$m; mkdir -p $D
  M10_DEQUANT_FAST=1 m7run m10-dqfast-grid-$m bash -c "python $REPO/code/m10_speed.py grid --model $m --fmt $G --base-fmt $G --out-dir $D --bw-json $OUT/speed_$m/bw.json --bw-names cpu --lengths --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" || exit 1
done
# 10/16 04:1xZ 加（09 §6.3-2b）：INT4（int4_kivi，主線第 2 階的格式）同樣做一次，對照 33a 的 CPU 層（原寫法）。融合 kernel 先在 GPU 上驗逐位元相同。
for m in llama qwen; do
  G=$(fmt_of $m fp8); I=$(fmt_of $m int4); D=$SW/speed_dqfast_int4_$m; mkdir -p $D
  M10_DEQUANT_FAST=1 m7run m10-dqfast-grid-int4-$m bash -c "python $REPO/code/m10_speed.py grid --model $m --fmt $I --base-fmt $G --out-dir $D --bw-json $OUT/speed_$m/bw.json --bw-names cpu --lengths --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" || exit 1
done
