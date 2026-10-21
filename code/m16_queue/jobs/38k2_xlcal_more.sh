# GPU-NEED: one
# CARDS: 0,1
# NEED-GB: 170
# MAX-H: 2
# AFTER: 37z_x2_report
# xKV 校準基底 vs 每題 SVD 加題數 第 2 段（續跑；做完就幾分鐘內結束）。設計見 38k1_xlcal_more.sh、x_stack_howto.md §16。
# 38k1 的冒煙沒過、或之後程式改過 → NEEDS_HUMAN、exit 1；校準檔的 sha256 再核對一次（被換過 → 停）。
# 〔估〕兩段的容量約 3.2 小時、需要約 2.6 小時；跑不完的話（38kz 會寫題數不齊）可以照抄這個檔成 38k3 再排一段。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_common.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_pool.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/xlcal3_common.sh
JOB=38k2_xlcal_more
xl3_check_smoke || exit 1
xl3_check_calib || exit 1
xl3_pool $XL3_BUDGET || exit 1
echo "$JOB 結束 $(date -u +%FT%TZ)"
