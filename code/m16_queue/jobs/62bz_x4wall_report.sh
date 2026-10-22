# GPU-NEED: none
# AFTER: 62b_x4a_wall
# X4a「全部層一組」的報告（只用 CPU、不 import torch）→ results/m16_stage3_mi300x/x_stack/x4/wall/{x4w_report.json, x4w_report.md}
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_common.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x4_common.sh
JOB=62bz_x4wall_report
m7run m16-x4wall-report python -I code/m16_x4w.py report --combos code/m16_xstack_combos_x4wall.json --raw "$X4D/tfwall/x4awall_*_32K_s*.csv.gz" --out-dir $X4R/wall \
  || { nh "$JOB：報告失敗（$(lastrun m16-x4wall-report)）"; exit 1; }
echo "$JOB 結束 $(date -u +%FT%TZ)"
