# GPU-NEED: one
# CARDS: 0,1
# NEED-GB: 170
# MAX-H: 2
# AFTER: 37z_x2_report
# xKV 事先校準的基底 第 2 段（續跑；做完就幾分鐘內結束）。設計、冒煙、工作池見 38c1_xkv_cal.sh 與 docs/stage3_20261019/x_stack_howto.md §14。
# 第 1 段的冒煙沒過、或冒煙之後程式改過 → NEEDS_HUMAN、exit 1。校準檔照理已在（calib-xl 有同設定就跳過；不在 → 這裡重算，〔估〕3–15 分鐘）。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_common.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_pool.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_ext.sh
JOB=38c2_xkv_cal
T0=$(date +%s)
x_check_smoke xlcal || exit 1
oncard 0 xlcal_calib llama & p0=$!
oncard 1 xlcal_calib qwen & p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
[ $r0 = 0 ] && [ $r1 = 0 ] || exit 1
xlcal_pool $(( X_EXT_BUDGET - ($(date +%s) - T0) / 60 )) || exit 1
echo "$JOB 結束 $(date -u +%FT%TZ)"
