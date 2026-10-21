# GPU-NEED: one
# CARDS: 0,1
# NEED-GB: 170
# MAX-H: 2
# AFTER: 37z_x2_report
# X4d 多輪 第 1 段（x_stack_howto.md §15）：C1a 的兩輪題組（seed 70000 起、T2–T4 各 32 題＝96 題）。壓縮整段 context（第 1 輪問題之前）→
# 生成第 1 輪 → 接第 2 輪（歷史＝第 1 輪的標準答案，照 m10_quality.run_turns）→ 生成；兩輪各自計分。配置＝X4b 前緣的前 2 個＋q:int4_kivi＋
# drop:KEYDIFF:0.5+q:int4_kivi＋E0（計畫檔由 38g1 的 select-cd 寫；38g1 沒跑的話這裡照同一個規則寫）。冒煙：每卡一個模型、兩輪生成 1 題。
# GPU 時間〔估〕：冒煙 5–10 分鐘；X4d 約 1.9 卡小時（Llama 1.1、Qwen 0.8）→ 兩卡約 0.95 小時。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_common.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_pool.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x4_common.sh
JOB=38h1_x4d_turns
T0=$(date +%s)
x4_check_smoke x4b || exit 1
x4_select_cd || exit 1
oncard 0 x4_smoke_gen llama $X4D_SEL 64 2 T2 $X4D_SEED0 x4d & p0=$!
oncard 1 x4_smoke_gen qwen $X4D_SEL 64 2 T2 $X4D_SEED0 x4d & p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
[ $r0 = 0 ] && [ $r1 = 0 ] || exit 1
x4_mark_smoke x4d
x4_run_pool $(( X4_BUDGET - ($(date +%s) - T0) / 60 )) x4d_fill || exit 1
echo "$JOB 結束 $(date -u +%FT%TZ)"
