# 14 §4.5 D3：丟掉 block 的還原速度。量時間 → 兩張卡獨佔（沒寫 GPU-NEED＝both）；在 a6:00.0（TIARA_GPU=1）。
# 底＝E0（BF16）／E1s（INT4 KIVI）；丟 F／B／R 的 25%／50%；CPU 層（Cake＋全部載入）、b16、b32、b48；另外 nfs 只跑 BF16 底。S0（K*），2 launch × 5 rep，launch 從 800。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
[ -e $JOBS/smoke_md/NO_D3 ] && { echo "44e0 冒煙沒過 → 跳過 D3" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
MA=$REPO/results/m10_match_mi300x; mkdir -p $MA/d3
DC=""; for b in E0 E1s; do for p in F B R; do for x in 25 50; do DC="$DC $b-D$p$x"; done; done; done
DN=""; for p in F B R; do for x in 25 50; do DN="$DN E0-D$p$x"; done; done
for m in llama qwen; do
  G=$(fmt_of $m fp8); E=$(python3 $REPO/code/m10_restore2.py env P0 --model $m --fmt int4_kivi)
  m7run m10-md-d3-$m env $E bash -c "python $REPO/code/m10_speed.py grid --model $m --fmt int4_kivi --base-fmt $G --out-dir $MA/d3/${m}_int4 --bw-json $MA/bw_$m.json --bw-names cpu b16 b32 b48 --configs E0 E1s $DC --refs --lengths --extra-modes loadall --extra-tiers cpu --launches 2 --reps 5 --launch0 800 --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" \
    || echo "44e3：D3 $m 失敗（見 runs/*-m10-md-d3-$m）" >> $JOBS/NEEDS_HUMAN
  m7run m10-md-d3nfs-$m env $E bash -c "python $REPO/code/m10_speed.py grid --model $m --fmt int4_kivi --base-fmt $G --out-dir $MA/d3/${m}_nfs --bw-json $MA/bw_$m.json --bw-names nfs --configs E0 $DN --refs --lengths --launches 2 --reps 5 --launch0 800 --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" \
    || echo "44e3：D3 nfs $m 失敗（見 runs/*-m10-md-d3nfs-$m）" >> $JOBS/NEEDS_HUMAN
done
exit 0
