# GPU-NEED: one
# CARDS: 0,1
# NEED-GB: 170
# MAX-H: 2
# AFTER: 37z_x2_report
# X4b 第 3 段（續跑；做完就幾分鐘內結束；38f4 是保險）。設計見 38f1_x4b_gen.sh、x_stack_howto.md §15。冒煙沒過或之後程式改過 → NEEDS_HUMAN、exit 1。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_common.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_pool.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x4_common.sh
JOB=38f3_x4b_gen
x4_check_smoke x4b || exit 1
x4_run_pool $X4_BUDGET x4b_fill || exit 1
echo "$JOB 結束 $(date -u +%FT%TZ)"
