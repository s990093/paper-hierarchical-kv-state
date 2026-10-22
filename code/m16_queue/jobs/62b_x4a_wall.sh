# GPU-NEED: one
# CARDS: 0,1
# NEED-GB: 170
# MAX-H: 1.5
# X4a 補跑：xKV「全部層一組」（Llama 32、Qwen 28；10/22 16:0xZ 看過 K1b 的 Qwen CPU 結果之後加，跑之前登記：experiment_log.md「X4a「全部層一組」」）。
# 配置 code/m16_xstack_combos_x4wall.json（m16_x4w.py grid --variant wall：T 20／24／32 × INT4／K4V2，W＝L 與同 T 的 W4 各一；Llama 9 個、Qwen 13 個含 E0）；GPU 路徑＝m16_x4.py tf
# （沒改，sha 同 X4a 的冒煙）。題目＝X4a 同一批 96 題（seed 64000 起）。事先寫死的判定：experiment_log.md「10/22 15:3xZ 起：補跑」。
# 流程：冒煙（卡 0 Llama、卡 1 Qwen 各 1 題；W＝L 的 SVD 是 32768／14336 維，32K 第一次跑）→ 兩卡共用工作池。判定規則同 W 大那組。
# GPU 時間〔估〕：Llama 每題約 60 卡秒（32768 維的 svd_lowrank）、Qwen 約 30 → 約 2.4 卡小時 → 兩卡約 0.7–1 小時；冒煙約 5–10 分鐘。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_common.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_pool.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x4_common.sh
JOB=62b_x4a_wall
X4W=code/m16_xstack_combos_x4wall.json
X4_GB_LLAMA=80
mkdir -p $X4D/tfwall
T0=$(date +%s)
m7run m16-x4w-selftest python -I code/m16_x4w.py --self-test || { nh "$JOB：m16_x4w self-test 沒過（$(lastrun m16-x4w-selftest)）"; exit 1; }
x4w_smoke() {
  local out=$X4D/smoke/x4awall_smoke_$1_s0.csv
  rm -f $out
  m7run m16-x4awall-smoke-$1 $X4PY tf --model $1 --combos $X4W --tasks T2 --n 1 --seed0 $X4A_SEED0 --shard 0/1 --out $out \
    || { nh "$JOB：$1 冒煙失敗（$(lastrun m16-x4awall-smoke-$1)）"; return 1; }
  x4_rows_ok $out $(( $(x4_nsel $X4W $1) + 1 )) || { nh "$JOB：$1 冒煙的列不對（$out）"; return 1; }
  grep -h "peak" $TIARA_RUNS/$(lastrun m16-x4awall-smoke-$1)/stdout.log | tail -1 || true
}
oncard 0 x4w_smoke llama & p0=$!
oncard 1 x4w_smoke qwen & p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
[ $r0 = 0 ] && [ $r1 = 0 ] || exit 1
x4aw_fill() { x4_units $1 x4awall tfwall 32K $X4A_SHARDS_LLAMA $X4A_SHARDS_QWEN "T2 T3 T4 T5" $X4A_N $X4A_SEED0 tf --combos $X4W; }
x4_run_pool $(( 75 - ($(date +%s) - T0) / 60 )) x4aw_fill || exit 1
echo "$JOB 結束 $(date -u +%FT%TZ)"
