# GPU-NEED: none
# AFTER: 37z_x2_report 37d_x2_gen 37e_x2_gen 37f_x2_gen
# X2 的最終報告：和 37z 同一支 gen-report，把 ext 層與 128K 的列也算進去（覆蓋 37z 的輸出；判定規則不變）。
# 37d／37e／37f 沒有排的話，把它們從上面的 AFTER 拿掉（runner 要所有 AFTER 都有 .done 才開）。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_common.sh
export OMP_NUM_THREADS=4
m7run m16-x2-report-final $XPY gen-report --raw "$XD/x2/x2_*_s*.csv.gz" --selection $XR/x2_selection.json --out-dir $XR \
  || { nh "37zz_x2_report_final：gen-report 失敗（$(lastrun m16-x2-report-final)）"; exit 1; }
