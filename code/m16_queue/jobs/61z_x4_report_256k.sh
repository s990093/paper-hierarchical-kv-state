# GPU-NEED: none
# AFTER: 61_x4c_256k_qwen_noT5
# X4 完整報告重出一次（加上 Qwen 256K 的 T2–T4；只用 CPU、不 import torch；cpu 佇列：nice 19、0–3 號核心）。
# 輸出照 x4_report：results/m16_stage3_mi300x/x_stack/x4/（上一版在 git 裡，commit 89dff4b 之前就有）。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_common.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x4_common.sh
JOB=61z_x4_report_256k
x4_report || exit 1
echo "$JOB 結束 $(date -u +%FT%TZ)"
