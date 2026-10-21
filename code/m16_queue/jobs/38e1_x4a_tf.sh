# GPU-NEED: one
# CARDS: 0,1
# NEED-GB: 170
# MAX-H: 2
# AFTER: 37z_x2_report
# X4a 第 1 段（10/21；docs/stage3_20261019/x4_limit_plan.md、x_stack_howto.md §15）：teacher forcing 篩選 code/m16_xstack_combos_x4.json 的
# 每個模型 71 個配置（配比格點：每個目標倍數 8／12／16／20／24／32 × 7–8 種分法、輕度丟掃描、秩細格點、對照），每題共用 SVD。
# 題目：新種子 64000 起、T2–T5 各 24 題（96 題）。流程：冒煙（卡 0 Llama、卡 1 Qwen 並行：m16_x4 gpu-selftest＋X4a 1 題；任一失敗 →
# NEEDS_HUMAN、exit 1）→ 兩卡共用工作池（Llama 8 份、Qwen 4 份交錯；每卡 ≤4 行程／≤165 GiB）；預算 100 分鐘扣掉冒煙。
# GPU 時間〔估〕：冒煙 10–20 分鐘；X4a 全部約 2.8 卡小時（Llama 1.8、Qwen 1.0；howto §15.6 的成本模型）→ 兩卡約 1.4 小時；本段 ≤ 約 1.9 小時。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_common.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_pool.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x4_common.sh
JOB=38e1_x4a_tf
T0=$(date +%s)
prep() { x4_selftest $1 && x4_smoke_tf $1; }
oncard 0 prep llama & p0=$!
oncard 1 prep qwen & p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
[ $r0 = 0 ] && [ $r1 = 0 ] || exit 1
x4_mark_smoke x4a
x4_run_pool $(( X4_BUDGET - ($(date +%s) - T0) / 60 )) x4a_fill || exit 1
echo "$JOB 結束 $(date -u +%FT%TZ)"
