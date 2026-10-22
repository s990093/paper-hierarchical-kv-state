# GPU-NEED: one
# CARDS: 0,1
# NEED-GB: 170
# MAX-H: 2
# AFTER: 37z_x2_report
# X2b 第 2 段（續跑；做完就幾分鐘內結束：每個 shard 先問 pending，0 題就不開行程）。設計、冒煙、工作池見 38b1_x2b_gen.sh 與
# docs/stage3_20261019/x_stack_howto.md §14。第 1 段的冒煙沒過、或冒煙之後程式改過 → NEEDS_HUMAN、exit 1（不在沒驗過的程式上跑）。
# 每段工作池預算 100 分鐘（最後一題的尾巴〔估〕≤10 分鐘）。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_common.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_pool.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_ext.sh
JOB=38b2_x2b_gen
x_check_smoke x2b || exit 1
x2b_pool $X_EXT_BUDGET || exit 1
echo "$JOB 結束 $(date -u +%FT%TZ)"
