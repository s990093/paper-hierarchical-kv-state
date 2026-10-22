# GPU-NEED: one
# CARDS: 0,1
# NEED-GB: 170
# MAX-H: 2
# AFTER: 37z_x2_report
# X2b 第 1 段（10/21 延伸；docs/stage3_20261019/x_stack_howto.md §14）：加題數解決統計力。新題 seed 63000 起、T2–T5 各 48 題（192 題）、32K、
# 配置＝code/m16_xstack_combos_v2.json 的 x2b（協調者的 10 個＋池子缺口的 q:int2_kivi、Qwen 的 xl:k22:v34；Llama 11、Qwen 12；
# roles／budgets 照第一版 x2_selection.json），gen --xl-cache（同題同秩的每題 SVD 只算一次）。
# 流程：plan（CPU，寫 results/…/x_stack/v2/x2b_selection.json）→ 冒煙（卡 0 Llama、卡 1 Qwen 並行：gpu-selftest --new-checks x2b＋X2b 的第一題，
#   輸出在 $XD/smoke/v2，不進 results；任一失敗 → NEEDS_HUMAN、exit 1）→ 兩卡共用工作池（x_pool.sh：Llama 12 份、Qwen 8 份交錯，
#   每卡 ≤4 個行程／≤165 GiB；預算＝100 分鐘扣掉冒煙）。逐題續跑：沒做完的留給 38b2–38b4。
# GPU 時間〔估〕：冒煙 10–20 分鐘；X2b 全部約 6.7–9.0 卡小時（Llama 4.0–6.0、Qwen 2.7–3.0；依據 howto §14.6：X2 逐列的實測吞吐＋X1 的 t_pipe_s）
# → 兩卡共用約 3.3–4.5 小時牆鐘；本段 ≤ 約 1.8 小時（工作池約 85 分鐘＋最後一題的尾巴）。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_common.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_pool.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_ext.sh
JOB=38b1_x2b_gen
T0=$(date +%s)
x2b_plan || { nh "$JOB：plan 失敗（$(lastrun m16-x2b-plan)）"; exit 1; }
oncard 0 x2b_smoke llama & p0=$!
oncard 1 x2b_smoke qwen & p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
[ $r0 = 0 ] && [ $r1 = 0 ] || exit 1
x_mark_smoke x2b
x2b_pool $(( X_EXT_BUDGET - ($(date +%s) - T0) / 60 )) || exit 1
echo "$JOB 結束 $(date -u +%FT%TZ)"
