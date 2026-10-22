# GPU-NEED: none
# AFTER: 38fz_x4b_report 38g1_x4c_128k 38g2_x4c_128k 38h1_x4d_turns 38h2_x4d_turns
# X4 的完整報告（加 128K、多輪）（只用 CPU、不 import torch；cpu 佇列：nice 19、0–3 號核心）→ results/m16_stage3_mi300x/x_stack/x4/{x4_report.json, tables_x4.md,
# x4_best_allocation.csv, x4a_configs_<m>.csv, pareto_x4a_<m>.png, pareto_x4b_<m>.png}（只寫 x4/，不覆蓋第一版與 v2 的檔；有什麼算什麼）。
# 沒排的 job 要從上面的 AFTER 拿掉（runner 要所有 AFTER 都有 .done 才開）。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_common.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_pool.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x4_common.sh
JOB=38hz_x4_report
export OMP_NUM_THREADS=4
x4_report || exit 1
echo "$JOB 結束 $(date -u +%FT%TZ)"
