# GPU-NEED: one
# CARDS: 0,1
# NEED-GB: 160
# MAX-H: 2
# AFTER: 02_smoke_x
# X1（code/m16_xstack.py screen；docs/stage3_20261019/x_stack_howto.md）第 3 段（保險）：照〔估〕的上緣才需要；36a＋36b 已經做完就幾分鐘內結束。
# Llama 在卡 0（$X_SHARDS_LLAMA 個 shard）、Qwen 在卡 1（$X_SHARDS_QWEN 個）；兩個背景行程、等兩個都結束，任一失敗 → NEEDS_HUMAN、exit 1。
# 每個 process 開跑超過 100 分鐘就不再開新的題（--budget-min），所以這個 job ≤ 2 小時〔估：最後一題約 5 分鐘＋校準＋載模型〕。
# 題目：DS1／S2 同一套（seed 60000 起、T2–T5 各 $X1_N 題、任務輪流排）；每題：S2 的原本各組（約 68 列）＋組合清單 161 個＋E0。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_common.sh
oncard 0 x1_model llama $X_SHARDS_LLAMA 100 & p0=$!
oncard 1 x1_model qwen $X_SHARDS_QWEN 100 & p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
[ $r0 = 0 ] && [ $r1 = 0 ]
