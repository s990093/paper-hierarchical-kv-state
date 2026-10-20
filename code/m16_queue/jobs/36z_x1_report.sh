# GPU-NEED: none
# AFTER: 36a_x1_screen 36b_x1_screen 36c_x1_screen
# X1 的報告（只用 CPU、不 import torch；cpu 佇列：nice 19、0–3 號核心）：Pareto 前緣、各家族最好的單一方法、代理的 Spearman／崩潰 AUROC、
# L5d 的停止條件（S1-a／S1-b／S1-c）、MLPerf 式門檻（只報）→ results/m16_stage3_mi300x/x_stack/{x1_report.json, tables_x1.md, configs_*.csv,
# pareto_*.png, x2_selection.json}。S2 子集另用 S2 自己的 report（results/m16_stage3_mi300x/x_stack/s2_subset/）。
# X1 每個模型做完的題 < 48 → x2_selection.json 的 go＝false（37* 不佔 GPU），寫 NEEDS_HUMAN。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_common.sh
export OMP_NUM_THREADS=4
m7run m16-x1-report $XPY report --raw "$XD/x1/x1_*_s*.csv.gz" --out-dir $XR \
  || { nh "36z_x1_report：report 失敗（$(lastrun m16-x1-report)）"; exit 1; }
m7run m16-x1-s2report python code/m16_s2_equalbytes.py report --raw "$XD/x1/s2part_*_s*.csv.gz" --out-dir $XR/s2_subset \
  || { nh "36z_x1_report：S2 子集的 report 失敗（$(lastrun m16-x1-s2report)）"; exit 1; }
x2_go; g=$?
[ $g = 3 ] && nh "36z_x1_report：X1 判定不跑 X2（$XR/x2_selection.json 的 reason）——照 L5d 的停止條件，37* 會直接結束"
[ $g = 0 ] || [ $g = 3 ]
