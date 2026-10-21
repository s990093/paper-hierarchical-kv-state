# GPU-NEED: one
# CARDS: 0,1
# NEED-GB: 170
# MAX-H: 2
# AFTER: 37z_x2_report
# xKV 校準基底 vs 每題 SVD 加題數 第 1 段（10/21 第三輪；code/m16_xlcal3.py、x_stack_howto.md §16）：38c 兩個模型都 UNDETERMINED（96 題）→
# 只比有 INT4 的兩對（xl:k22:v34／xl:k28:v42，校準基底 vs 每題 SVD）＋E0，新題 seed 66000 起、T2–T5 各 48 題（192 題）、32K、同一題同一行程配對；
# m16_xstack.py gen＋--calib-xl＋--xl-cache（和 38c 同一條路徑）。校準基底沿用 38c 的 calib_xl_<m>.pt（不重算；sha256 不等於 38c 記的 → 停）。
# 流程：plan（CPU）→ check-calib（CPU，sha256）→ 冒煙（卡 0 Llama、卡 1 Qwen 並行：新題的第 1 題、5 列；失敗 → NEEDS_HUMAN、exit 1）→
# 兩卡共用工作池（Llama 12 份、Qwen 8 份；每卡 ≤4 行程／≤165 GiB）；預算 100 分鐘扣掉前面。沒做完的留給 38k2。
# GPU 時間〔估〕：全部約 5.1 卡小時（Llama 3.7、Qwen 1.4；依據＝38c 的實測，howto §16.4）→ 兩卡約 2.6 小時；本段 ≤ 約 1.9 小時（冒煙約 5–10 分鐘）。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_common.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_pool.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/xlcal3_common.sh
JOB=38k1_xlcal_more
T0=$(date +%s)
xl3_plan || exit 1
xl3_check_calib || exit 1
oncard 0 xl3_smoke llama & p0=$!
oncard 1 xl3_smoke qwen & p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
[ $r0 = 0 ] && [ $r1 = 0 ] || exit 1
xl3_mark_smoke
xl3_pool $(( XL3_BUDGET - ($(date +%s) - T0) / 60 )) || exit 1
echo "$JOB 結束 $(date -u +%FT%TZ)"
