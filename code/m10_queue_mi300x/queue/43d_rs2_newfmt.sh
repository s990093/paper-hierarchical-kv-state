# 13 §5（Q3）：大掃描新格式的融合 kernel（m10_dequant_fast_x，M10_DEQUANT_FAST_X=1；k1，新格式沒有調參）。
# 43 的速度配置照抄（queue/43_sweep_confirm.sh 的 grid），只跑真實 CPU 層、閒／忙、3 launch × 10 rep（launch 0–2，和 43 同 seed → R 的位置相同）。
# 43 的判定照原寫法不動；這裡只報告 CPU 層的絕對時間。計時的 stage。失敗寫 NEEDS_HUMAN；exit 0。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
R=$REPO/results/m10_restore2_mi300x/q3; mkdir -p $R
SRC=$REPO/code/m10_queue_mi300x/queue/43_sweep_confirm.sh
grep -v smoke $SRC | sed -n 's/.*m10_confirm.py speed --model \([a-z]*\) --fmt \([^ ]*\) --configs \(.*\) --out-dir.*/\1 \2 \3/p' | while read -r m f cfgs; do
  M10_CODEC_X=1 M10_DEQUANT_FAST=1 M10_DEQUANT_FAST_X=1 m7run m10-rs2-q3-$m-$f bash -c "python $REPO/code/m10_confirm.py speed --model $m --fmt $f --configs $cfgs --out-dir $R/speed_${m}_$f --bw-json $OUT/speed_$m/bw.json --bw-names cpu --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" \
    || echo "43d：Q3 $m $f 失敗" >> $JOBS/NEEDS_HUMAN
done
exit 0
