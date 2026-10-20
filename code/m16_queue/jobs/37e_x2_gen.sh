# GPU-NEED: one
# CARDS: 0,1
# NEED-GB: 110
# MAX-H: 2
# AFTER: 36z_x1_report
# X2 128K 第 1 段：第二階段 44c 的題組（m10_sweep screen 的 --n-blocks 252、T2–T5 各 12 題、seed 40000 起；run 20261019-033004-m10-lc-screen-llama-252-s0
# 的 cmd.sh），只跑 x2_selection.json 裡標 128k 的配置（每個預算的最好單一方法＋前緣第 1 名＋INT4 KIVI＋部署對照）。探索，不進 M1／M2 判定。
# E0 的生成分數逐題對 44c 的逐列檔（/mlsteam/data/tiara/longctx_m10/screen_<m>_252_s*.csv.gz）比，記在 .checks.jsonl。
# Llama 一個 process 約 80–100 GiB〔估〕→ 卡 0 開 $X2L_SHARDS_LLAMA 個；Qwen 卡 1 開 $X2L_SHARDS_QWEN 個。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_common.sh
x2_gate 37e_x2_gen
oncard 0 x2_model llama 128K 128k $X2L_SHARDS_LLAMA 100 & p0=$!
oncard 1 x2_model qwen 128K 128k $X2L_SHARDS_QWEN 100 & p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
[ $r0 = 0 ] && [ $r1 = 0 ]
