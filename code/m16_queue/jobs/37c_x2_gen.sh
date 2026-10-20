# GPU-NEED: one
# CARDS: 0,1
# NEED-GB: 150
# MAX-H: 2
# AFTER: 36z_x1_report
# X2 32K 第 3 段（code/m16_xstack.py gen；docs/stage3_20261019/x_stack_howto.md §5–§6）：讀 X1 的 x2_selection.json，量生成答對率
# （m10_sweep.Item.gen()，第二階段同一個計分）。題目＝新題 seed 62000 起、T2–T5 各 $X2_N 題（避開 X1 的挑選偏差）。
# 32K 的層照順序 core → x3（給 X3 的 x2_feasible.csv）→ ext，四段 job 共用同一套逐題續跑：每段 100 分鐘預算，做完的層／題會跳過。
# x2_selection.json 的 go＝false（X1 的 S1-b 成立或 X1 不完整）→ 不佔 GPU、exit 0。
# Llama 在卡 0（$X2_SHARDS_LLAMA 個 shard）、Qwen 在卡 1（$X2_SHARDS_QWEN 個）；任一失敗 → NEEDS_HUMAN、exit 1。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_common.sh
x2_gate 37c_x2_gen
oncard 0 x2_seq llama $X2_SHARDS_LLAMA 100 core x3 ext & p0=$!
oncard 1 x2_seq qwen $X2_SHARDS_QWEN 100 core x3 ext & p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
[ $r0 = 0 ] && [ $r1 = 0 ]
