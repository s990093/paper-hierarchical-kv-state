# 10_longctx_plan §6：長 context 的程式第一次在 GPU 上跑（128K＝252 個 block）。冒煙，數字不進結果（scratch）。
# 失敗 → NEEDS_HUMAN、把 90／91 搬到 hold/（修好再搬回），exit 0：不擋大掃描和其他工作。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
S=$JOBS/smoke_longctx; rm -rf $S; mkdir -p $S
fail=0
for m in qwen llama; do
  F=$(fmt_of $m fp8)
  m7run m10-lc-smoke-screen-$m python $REPO/code/m10_sweep.py screen --model $m --params-file $P --n-blocks 252 --tasks T2 --n 1 \
    --formats $F int4_kivi --policies F M B R0 ATT --fracs 0.25 0.5 --gen-all --gen-pos $F --out $S/screen_$m.csv.gz \
    > $S/screen_$m.log 2>&1 || { echo "longctx smoke screen $m 失敗（$S/screen_$m.log）" | tee -a $JOBS/NEEDS_HUMAN; tail -20 $S/screen_$m.log; fail=1; }
done
m7run m10-lc-smoke-grid-qwen python $REPO/code/m10_speed.py grid --model qwen --fmt fp8_head --base-fmt fp8_head --out-dir $S/grid_qwen \
  --bw-json $OUT/speed_qwen/bw.json --bw-names b32 cpu --gpu-states idle --launches 1 --reps 1 --configs E0 E1 --refs E0 E1 \
  --main-length 129024 --lengths --blocks-out $S/grid_qwen_blocks.csv > $S/grid_qwen.log 2>&1 \
  || { echo "longctx smoke grid 失敗（$S/grid_qwen.log）" | tee -a $JOBS/NEEDS_HUMAN; tail -20 $S/grid_qwen.log; fail=1; }
if [ $fail = 0 ]; then
  python3 - <<PY || fail=1
import csv, gzip, sys
sys.path.insert(0, "$REPO/code")
for m in ("qwen", "llama"):
    rows = list(csv.DictReader(gzip.open("$S/screen_%s.csv.gz" % m, "rt")))
    nb = {r["n_blocks"] for r in rows}; g = [r for r in rows if r["gen_score"] != ""]
    pol = sorted({(r["fmt"], r["policy"], r["frac"]) for r in g})
    print(m, "rows", len(rows), "n_blocks", nb, "gen rows", len(g), pol)
    assert nb == {"252"} and len(g) == 1 + 2 + 4, (nb, len(g))      # E0 + 2 個格式的 ALL + 主線 FP8 的 F/M/B/R0
import m10_sweep_sim as S
p = S.load_parts("$S/grid_qwen_blocks.csv", length=129024)
f, l = p["f"]["idle"], p["l"][("b32", "idle", "bf16")]
assert len(f) == 252 and len(l) == 252, (len(f), len(l))
print("sim 128K E0 b32 idle: b =", S.simulate(f, l)[0], "（只檢查推算跑得動；實測會合點見 grid_qwen.log）")
PY
fi
echo "longctx smoke fail=$fail"
if [ $fail != 0 ]; then
  for s in 90_longctx_speed 91_longctx_pilot 92_longctx_screen; do [ -f $JOBS/queue/$s.sh ] && mv $JOBS/queue/$s.sh $JOBS/hold/$s.sh && echo "hold $s"; done
  echo "longctx smoke 失敗 → 90／91／92 搬到 hold/" >> $JOBS/NEEDS_HUMAN
fi
exit 0
