# GPU-NEED: none
# AFTER: 38i1_x4c_256k_qwen 38hz_x4_report
# 加上 Qwen 256K 再出一次（38i1 沒排就不要排這個）（只用 CPU、不 import torch；cpu 佇列：nice 19、0–3 號核心）→ results/m16_stage3_mi300x/x_stack/x4/{x4_report.json, tables_x4.md,
# x4_best_allocation.csv, x4a_configs_<m>.csv, pareto_x4a_<m>.png, pareto_x4b_<m>.png}（只寫 x4/，不覆蓋第一版與 v2 的檔；有什麼算什麼）。
# 沒排的 job 要從上面的 AFTER 拿掉（runner 要所有 AFTER 都有 .done 才開）。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_common.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_pool.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x4_common.sh
JOB=38iz_x4_report
export OMP_NUM_THREADS=4
x4_report || exit 1
echo "$JOB 結束 $(date -u +%FT%TZ)"
