# 大掃描程式的冒煙測試（約 10 分鐘；GPU 被主線佔滿時沒辦法先試）：每個模型 1 題、少數格式與放法，把 parts／screen 的每條路徑跑過。
# 數字不用（輸出放 scratch）。失敗不擋主線：寫 NEEDS_HUMAN、留完整錯誤，佇列照常往下（40 開跑前修好）。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
S=$JOBS/smoke_sweep; rm -rf $S; mkdir -p $S
G_llama=/mlsteam/data/tiara/runs/20261015-161226-m10-grid-fp8-llama/blocks.csv
G_qwen=/mlsteam/data/tiara/runs/20261015-194150-m10-regrid-fp8-qwen/blocks.csv
fail=0
for m in qwen llama; do
  F=$(fmt_of $m fp8); GB=G_$m
  m7run m10-sweep-smoke-parts-$m python $REPO/code/m10_sweep.py parts --model $m --formats $F mxfp4 int2h_g32 kv-int8_token-int4_kivi --reps 1 --out-dir $S \
    > $S/parts_$m.log 2>&1 || { echo "smoke parts $m 失敗（$S/parts_$m.log）" | tee -a $JOBS/NEEDS_HUMAN; tail -20 $S/parts_$m.log; fail=1; continue; }
  m7run m10-sweep-smoke-screen-$m python $REPO/code/m10_sweep.py screen --model $m --params-file $P --tasks T2 T5 --n 1 \
    --formats $F mxfp4 int2h_g32 kv-int8_token-int4_kivi --policies F R0 ERR ATT AXE ORC --fracs 0.25 0.5 --parts $S/parts_$m.json --grid-blocks ${!GB} \
    --out $S/screen_$m.csv.gz > $S/screen_$m.log 2>&1 || { echo "smoke screen $m 失敗（$S/screen_$m.log）" | tee -a $JOBS/NEEDS_HUMAN; tail -20 $S/screen_$m.log; fail=1; }
done
m7run m10-dqfast-smoke python $REPO/code/m10_dequant_fast.py --self-test > $S/dqfast.log 2>&1 || { echo "smoke dqfast 失敗" | tee -a $JOBS/NEEDS_HUMAN; fail=1; }
echo "smoke fail=$fail"
exit 0
