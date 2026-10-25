# GPU-NEED: none
# AFTER: 62_x4a_wbig
# X4a W 大補跑的報告（只用 CPU、不 import torch）→ results/m16_stage3_mi300x/x_stack/x4/wbig/{x4w_report.json, x4w_report.md}
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_common.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x4_common.sh
JOB=62z_x4w_report
m7run m16-x4w-report python -I code/m16_x4w.py report --raw "$X4D/tfw/x4aw_*_32K_s*.csv.gz" --out-dir $X4R/wbig \
  || { nh "$JOB：報告失敗（$(lastrun m16-x4w-report)）"; exit 1; }
echo "$JOB 結束 $(date -u +%FT%TZ)"
