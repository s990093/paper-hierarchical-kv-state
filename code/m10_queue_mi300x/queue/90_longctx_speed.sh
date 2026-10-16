# 10_longctx_plan §3.4：64K、128K 的速度格子（主線 8 個配置；FP8 主線粒度＋INT4 KIVI；主線的 bw.json；2 launch × 5 rep；參考線 E0、E1）。
# FP8 先跑（91／92 的讀法 B 要用它的參考線）。gpu_guard 不是 CLEAN 的格子不能用、要重量（CLAUDE.md §3）。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
LC=$REPO/results/m10_longctx_mi300x; mkdir -p $LC
for tier in fp8 int4; do
  for L in 129024 65536; do
    for m in llama qwen; do
      G=$(fmt_of $m fp8); F=$(fmt_of $m $tier); D=$LC/speed_${m}_${L}_$F; mkdir -p $D
      m7run m10-lc-grid-$m-$L-$F bash -c "python $REPO/code/m10_speed.py grid --model $m --fmt $F --base-fmt $G --out-dir $D --bw-json $OUT/speed_$m/bw.json --main-length $L --lengths --launches 2 --reps 5 --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" || exit 1
    done
  done
done
