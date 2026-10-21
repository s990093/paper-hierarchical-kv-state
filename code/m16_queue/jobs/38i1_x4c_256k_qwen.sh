# GPU-NEED: one
# CARDS: 0,1
# NEED-GB: 170
# MAX-H: 2
# AFTER: 37z_x2_report
# X4c 的 Qwen 256K 切片（排最後；時間不夠就不排，報告會寫「沒有資料」）：504 block＝258,048 token（Qwen noDCA 上限 262,144，留問題＋生成），
# seed 40000 起、T2–T5 各 6 題（24 題），配置＝X4c 的計畫檔（前緣 3 個＋q:int4_kivi＋E0）。256K 的題組第二階段沒有跑過（m10_tasks.make_item 的
# n_blocks＝504 第一次用）→ 冒煙 1 題先過才跑。每卡 1 個行程（登記 95 GiB〔估〕：Qwen 128K 實測 42.1 GiB 的 2 倍＋SVD）。
# GPU 時間〔估〕：每題約 3–4 分鐘（prefill 1–2 分鐘＋SVD＋5 次生成）→ 24 題、兩個行程約 40–50 分鐘；冒煙 5–10 分鐘。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_common.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_pool.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x4_common.sh
JOB=38i1_x4c_256k_qwen
T0=$(date +%s)
x4_check_smoke x4c || exit 1
oncard 1 x4_smoke_gen qwen $X4C_SEL 504 1 T2 $X4C_SEED0 x4q256 || exit 1
x4_run_pool $(( X4_BUDGET - ($(date +%s) - T0) / 60 )) q256_fill || exit 1
echo "$JOB 結束 $(date -u +%FT%TZ)"
