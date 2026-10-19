# 14 §2 M1：還原單位 512／256／128（M10_UNIT）。量時間 → 兩張卡獨佔；a6:00.0。INT4 KIVI；E0、E1s；CPU、b16、b32、b48；S0；2 launch × 5 rep，launch 從 600。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
[ -e $JOBS/smoke_md/NO_M1 ] && { echo "44e0 冒煙沒過 → 跳過 M1" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
MA=$REPO/results/m10_match_mi300x; mkdir -p $MA/m1
for m in llama qwen; do
  G=$(fmt_of $m fp8); E=$(python3 $REPO/code/m10_restore2.py env P0 --model $m --fmt int4_kivi)
  for U in 512 256 128; do
    m7run m10-md-m1-$m-u$U env $E M10_UNIT=$U bash -c "python $REPO/code/m10_speed.py grid --model $m --fmt int4_kivi --base-fmt $G --out-dir $MA/m1/${m}_u$U --bw-json $MA/bw_$m.json --bw-names cpu b16 b32 b48 --configs E0 E1s --refs --lengths --launches 2 --reps 5 --launch0 600 --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" \
      || echo "44e5：M1 $m 單位 $U 失敗（見 runs/*-m10-md-m1-$m-u$U）" >> $JOBS/NEEDS_HUMAN
  done
done
exit 0
