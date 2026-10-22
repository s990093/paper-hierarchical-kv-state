# GPU-NEED: one
# CARDS: 0,1
# NEED-GB: 170
# MAX-H: 2
# AFTER: 37z_x2_report
# X4c 128K 第 1 段（x_stack_howto.md §15）：照事先寫死的規則從 X4b 挑前緣（m16_x4 select-cd：T* 的最佳配比、下一個較大 T 的、T＝8 的；
# X4b < 48 題 → NEEDS_HUMAN、exit 1；同時寫 X4d 的計畫檔）→ 冒煙（每卡一個模型：128K 生成 1 題）→ 兩卡共用工作池（44c 的 128K 題組：252 block、
# seed 40000 起、T2–T5 各 12 題＝48 題；前緣 3 個＋q:int4_kivi＋E0；Llama 4 份（100 GiB）、Qwen 4 份（50 GiB））。
# GPU 時間〔估〕：冒煙 5–15 分鐘；X4c 約 2.6 卡小時（Llama 1.7、Qwen 0.9；128K 的 SVD 約 32K 的 4 倍〔算術〕、prefill 與生成照 02 冒煙的 128K 實測）
# → 兩卡約 1.3 小時。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_common.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_pool.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x4_common.sh
JOB=38g1_x4c_128k
T0=$(date +%s)
x4_check_smoke x4b || exit 1
x4_select_cd || exit 1
oncard 0 x4_smoke_gen llama $X4C_SEL 252 1 T2 $X4C_SEED0 x4c & p0=$!
oncard 1 x4_smoke_gen qwen $X4C_SEL 252 1 T2 $X4C_SEED0 x4c & p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
[ $r0 = 0 ] && [ $r1 = 0 ] || exit 1
x4_mark_smoke x4c
x4_run_pool $(( X4_BUDGET - ($(date +%s) - T0) / 60 )) x4c_fill || exit 1
echo "$JOB 結束 $(date -u +%FT%TZ)"
