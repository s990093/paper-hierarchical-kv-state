# GPU-NEED: one
# CARDS: 0,1
# NEED-GB: 170
# MAX-H: 2
# AFTER: 37z_x2_report
# xKV 事先校準的基底 第 1 段（10/21 延伸；docs/stage3_20261019/x_stack_howto.md §14.4）：基底用校準題（seed 61000 起 8 題）事先算好、所有題共用，
# 和每題 SVD 的同名配置比（xl:k11:v17、xl:k22:v34+q:int4_kivi、xl:k28:v42+q:int4_kivi）。
# 流程：plan（CPU，寫 v2/xlcal_selection.json）→ 每個模型一張卡並行：calib-xl（已有同設定的就跳過）→ 冒煙（gpu-selftest --new-checks xlcal＋
#   TF 1 題＋生成 1 題，輸出在 $XD/smoke/v2；任一失敗 → NEEDS_HUMAN、exit 1）→ 兩卡共用工作池：先 TF（X1 的 48 題，校準版 3 個＋每題 SVD 錨點
#   xl:k11:v17）、再生成（X2 的 96 題，6 個配置＋E0）。預算＝100 分鐘扣掉前面。沒做完的留給 38c2。
# GPU 時間〔估〕：calib-xl 每個模型 3–15 分鐘（8 題 prefill＋4096 維 FP64 的 ΣZᵀZ 與 eigh；ROCm 上 4096 維 eigh 的速度 NOT_MEASURED）、
#   冒煙 10–15 分鐘；TF 約 0.4 卡小時、生成約 3.0 卡小時（howto §14.6）→ 兩卡合計約 2.0–2.3 小時牆鐘，本段做約 1.5 小時。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_common.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_pool.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_ext.sh
JOB=38c1_xkv_cal
T0=$(date +%s)
xlcal_plan || { nh "$JOB：plan 失敗（$(lastrun m16-xlcal-plan)）"; exit 1; }
prep() { xlcal_calib $1 && xlcal_smoke $1; }
oncard 0 prep llama & p0=$!
oncard 1 prep qwen & p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
[ $r0 = 0 ] && [ $r1 = 0 ] || exit 1
x_mark_smoke xlcal
xlcal_pool $(( X_EXT_BUDGET - ($(date +%s) - T0) / 60 )) || exit 1
echo "$JOB 結束 $(date -u +%FT%TZ)"
