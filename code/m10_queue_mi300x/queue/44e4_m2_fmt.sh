# 14 §3 M2：用格式去匹配慢的層。量時間 → 兩張卡獨佔；a6:00.0。
# 每個模型 × 格式（主線 FP8、int4_kivi、K4V2、int2_kivi、mxfp4）一個 grid；b16／b32／b48（模型自己的 bw）＋nfs；E1s（第一個 grid 另加 E0）；
# 參考線只算＋只載 E1s（第一個另加只載 E0）；新格式用融合 X kernel（M10_CODEC_X=1 M10_DEQUANT_FAST_X=1）；S0；2 launch × 5 rep，launch 從 700。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
[ -e $JOBS/smoke_md/NO_M2 ] && { echo "44e0 冒煙沒過 → 跳過 M2" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
MA=$REPO/results/m10_match_mi300x; mkdir -p $MA/m2
for m in llama qwen; do
  G=$(fmt_of $m fp8); first=1
  for F in $G int4_kivi kv-int4_kivi-int2_kivi int2_kivi mxfp4; do
    if [ $first = 1 ]; then C="E0 E1s"; R="E0 E1s"; first=0; else C="E1s"; R="E1s"; fi
    E=$(python3 $REPO/code/m10_restore2.py env P0 --model $m --fmt $F)
    m7run m10-md-m2-$m-$F env $E M10_CODEC_X=1 M10_DEQUANT_FAST_X=1 bash -c "python $REPO/code/m10_speed.py grid --model $m --fmt $F --base-fmt $G --out-dir $MA/m2/${m}_$F --bw-json $MA/bw_$m.json --bw-names b16 b32 b48 nfs --configs $C --refs $R --lengths --launches 2 --reps 5 --launch0 700 --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" \
      || echo "44e4：M2 $m $F 失敗（見 runs/*-m10-md-m2-$m-$F）" >> $JOBS/NEEDS_HUMAN
  done
done
exit 0
