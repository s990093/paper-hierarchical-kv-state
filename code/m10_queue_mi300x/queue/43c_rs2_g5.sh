# 13 §4（Q2）：會合規則 G5（開始前估計，全部載完比重算第 0 塊快就不重算）。K*、S0。
# 同一個 process 裡交錯跑 g5（cake 列）、claim（照 Cake）、CPU 層另加 loadall。CPU 層 3 launch × 10 rep；限速層 b16 b32 b48 2 launch × 5 rep。
# 計時的 stage：跑的時候不在 GPU 上插任何東西。失敗寫 NEEDS_HUMAN、跑下一個；exit 0。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
R=$REPO/results/m10_restore2_mi300x/q2; mkdir -p $R
# 13 §8c：G5b（模擬 Cake 的兩條線再和全載入比）先在真模型冒煙（Qwen int4_kivi、1 × 1、每格檢查 KV）；沒過就只跑 G5
S=$REPO/results/m10_restore2_mi300x/smoke; G5B=1
m7run m10-rs2-smoke-g5b env $(python3 $REPO/code/m10_dqsweep.py env S0 --model qwen --fmt int4_kivi) M10_MEET=g5b bash -c "python $REPO/code/m10_speed.py grid --model qwen --fmt int4_kivi --base-fmt $(fmt_of qwen fp8) --bw-json $OUT/speed_qwen/bw.json --refs --lengths --launches 1 --reps 1 --launch0 391 --out-dir $S/g5b --bw-names cpu b32 --gpu-states idle --configs E0 E1 R25 --extra-modes loadall claim --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" \
  && python3 -c "
import csv,sys
rows=list(csv.DictReader(open('$S/g5b/restores.csv')))
ok=rows and all(r['kv_bad'] in ('','0') for r in rows) and all(r['meet_rule_used'] in ('none','claim') for r in rows if r['mode']=='cake')
sys.exit(0 if ok else 1)" || { G5B=0; echo "43c：G5b 冒煙沒過（$S/g5b）→ 只跑 G5" | tee -a $JOBS/NEEDS_HUMAN; }
RULES="g5"; [ $G5B = 1 ] && RULES="g5 g5b"
for rule in $RULES; do
sfx=""; [ $rule = g5b ] && sfx="_g5b"
for m in llama qwen; do
  G=$(fmt_of $m fp8)
  for tag in fp8 int4; do
    F=$(fmt_of $m $tag); K=$(python3 $REPO/code/m10_dqsweep.py env S0 --model $m --fmt $F)
    BASE="python $REPO/code/m10_speed.py grid --model $m --fmt $F --base-fmt $G --bw-json $OUT/speed_$m/bw.json --configs E0 E1 R25 --refs --lengths --launch0 300 --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv"
    m7run m10-rs2-q2-$m-$tag-cpu$sfx env $K M10_MEET=$rule bash -c "$BASE --out-dir $R/${m}_${tag}_cpu$sfx --bw-names cpu --extra-modes loadall claim --launches 3 --reps 10" \
      || echo "43c：$rule $m $tag CPU 層失敗" >> $JOBS/NEEDS_HUMAN
    m7run m10-rs2-q2-$m-$tag-slow$sfx env $K M10_MEET=$rule bash -c "$BASE --out-dir $R/${m}_${tag}_slow$sfx --bw-names b16 b32 b48 --extra-modes claim --launches 2 --reps 5" \
      || echo "43c：$rule $m $tag 限速層失敗" >> $JOBS/NEEDS_HUMAN
  done
done
done
python3 $REPO/code/m10_restore2.py pick-g5 || echo "43c：pick-g5 失敗 → 44d 用 g5" >> $JOBS/NEEDS_HUMAN
exit 0
