# 12_tier_plan §3：兩層混合還原（CPU 層 BF16 ＋ 限速慢層 6.98／0.331 GiB/s 存 BF16 或 INT4；32K；探索，只報告）。
# 先真模型冒煙（每個模型 1 launch × 1 rep、cpu＋ssd、閒，每次都檢查 KV）；冒煙失敗 → NEEDS_HUMAN、跳過那個模型的正式部分。
# 正式：2 launch × 5 rep × GPU 閒／忙 × 50 格。失敗 → NEEDS_HUMAN（含錯誤尾巴）、換下一個模型；exit 0，不擋後面的工作。
# 計時的 stage：跑的時候不在 GPU 上插任何東西。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
TO=$REPO/results/m10_tier_mi300x; S=$JOBS/smoke_tier2; rm -rf $S; mkdir -p $TO $S
for m in llama qwen; do
  m7run m10-tier2-smoke-$m python $REPO/code/m10_tier2.py smoke --model $m --out-dir $S/$m > $S/smoke_$m.log 2>&1 \
    || { echo "44 tier2 smoke $m 失敗（$S/smoke_$m.log）→ 跳過 $m 的正式部分" | tee -a $JOBS/NEEDS_HUMAN; tail -20 $S/smoke_$m.log; continue; }
  grep "^smoke" $S/smoke_$m.log
  D=$TO/$m; mkdir -p $D
  m7run m10-tier2-$m bash -c "python $REPO/code/m10_tier2.py grid --model $m --out-dir $D --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" \
    || { echo "44 tier2 grid $m 失敗（看 \$TIARA_RUNS 裡最新的 m10-tier2-$m）" | tee -a $JOBS/NEEDS_HUMAN; continue; }
  python $REPO/code/m10_tier2.py summary --restores $D/restores.csv --noise $OUT/speed_$m/noise.json --out-dir $TO --fig \
    > $TO/summary_$m.log 2>&1 || { echo "44 tier2 summary $m 失敗（$TO/summary_$m.log）" | tee -a $JOBS/NEEDS_HUMAN; tail -20 $TO/summary_$m.log; }
  gzip -kf $D/restores.csv
done
exit 0
