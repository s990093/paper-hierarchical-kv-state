# 31z 冒煙抓到 m10_codec_x.install() 之後舊格式無限遞迴（10/16 03:47Z，screen 兩個模型都失敗）；修好後重跑同一個冒煙。
# 這次失敗 → 大掃描（40、42）搬到 hold/、寫 NEEDS_HUMAN，佇列照常往下跑只報告的工作（GPU 不空著）；exit 0。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
S=$JOBS/smoke_sweep2; rm -rf $S; mkdir -p $S
G_llama=/mlsteam/data/tiara/runs/20261015-161226-m10-grid-fp8-llama/blocks.csv
G_qwen=/mlsteam/data/tiara/runs/20261015-194150-m10-regrid-fp8-qwen/blocks.csv
fail=0
for m in qwen llama; do
  F=$(fmt_of $m fp8); GB=G_$m
  m7run m10-sweep-smoke2-parts-$m python $REPO/code/m10_sweep.py parts --model $m --formats $F int4_kivi mxfp4 int2h_g32 kv-int8_token-int4_kivi --reps 1 --out-dir $S \
    > $S/parts_$m.log 2>&1 || { echo "smoke2 parts $m 失敗（$S/parts_$m.log）" | tee -a $JOBS/NEEDS_HUMAN; tail -20 $S/parts_$m.log; fail=1; continue; }
  m7run m10-sweep-smoke2-screen-$m python $REPO/code/m10_sweep.py screen --model $m --params-file $P --tasks T2 T5 --n 1 \
    --formats $F int4_kivi mxfp4 int2h_g32 kv-int8_token-int4_kivi --policies F R0 ERR ATT AXE ORC --fracs 0.25 0.5 --parts $S/parts_$m.json --grid-blocks ${!GB} \
    --out $S/screen_$m.csv.gz > $S/screen_$m.log 2>&1 || { echo "smoke2 screen $m 失敗（$S/screen_$m.log）" | tee -a $JOBS/NEEDS_HUMAN; tail -20 $S/screen_$m.log; fail=1; }
done
# 10/16 04:1xZ 加：融合反量化（含新加的 INT4）在 GPU 上的逐位元自測，和 42 會用的「M10_DEQUANT_FAST=1 的速度格子」路徑（CPU 層、1 launch × 1 rep）
m7run m10-sweep-smoke2-dqfast python $REPO/code/m10_dequant_fast.py --self-test --out $S/dqfast_selftest.json > $S/dqfast.log 2>&1   || { echo "smoke2 dqfast 自測失敗（$S/dqfast.log）" | tee -a $JOBS/NEEDS_HUMAN; tail -20 $S/dqfast.log; fail=1; }
M10_DEQUANT_FAST=1 m7run m10-sweep-smoke2-dqgrid-qwen python $REPO/code/m10_speed.py grid --model qwen --fmt int4_kivi --base-fmt fp8_head   --out-dir $S/dqgrid_qwen --bw-json $OUT/speed_qwen/bw.json --bw-names cpu --gpu-states idle --launches 1 --reps 1 --configs E0 E1 --refs E0 E1 --lengths   > $S/dqgrid_qwen.log 2>&1 || { echo "smoke2 dqgrid 失敗（$S/dqgrid_qwen.log）" | tee -a $JOBS/NEEDS_HUMAN; tail -20 $S/dqgrid_qwen.log; fail=1; }
if [ $fail = 0 ]; then
  python3 - <<PY || fail=1
import csv, gzip
for m in ("qwen", "llama"):
    rows = list(csv.DictReader(gzip.open("$S/screen_%s.csv.gz" % m, "rt")))
    fm = sorted({r["fmt"] for r in rows}); ro = sorted({r["readout"] for r in rows})
    print(m, "rows", len(rows), "fmts", fm, "readouts", ro)
    assert len(rows) > 0 and {"A", "B_idle", "B_busy"} <= set(ro), (m, ro)
PY
fi
echo "smoke2 fail=$fail"
if [ $fail != 0 ]; then
  for s in 40_sweep_pilot 41_sweep_screen 42_dqfast_cpu_grid; do [ -f $JOBS/queue/$s.sh ] && mv $JOBS/queue/$s.sh $JOBS/hold/$s.sh && echo "hold $s"; done
  echo "smoke2 失敗 → 40／41／42 搬到 hold/（修好再搬回）" >> $JOBS/NEEDS_HUMAN
fi
exit 0
