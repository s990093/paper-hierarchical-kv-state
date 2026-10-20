# GPU-NEED: none
# AFTER: 37a_x2_gen 37b_x2_gen 37c_x2_gen
# X2 的報告（只用 CPU；cpu 佇列）：L5d §6.3 的 M1／M2／完整性判定（32K），MLPerf 式 99%／99.9% 門檻只報；
# 另寫 X3（code/m16_xtime.py load_x2；39z_x3_map.sh 等這個 job）要的 results/m16_stage3_mi300x/x_stack/x2_feasible.csv
# （model、x3_segment、feasible＝ε 判定、run_id）與對不到 X3 解碼段的 x2_unmapped.csv。
# 37a–37c 是 32K 的 core＋x3 兩層〔估〕夠做完的段數；ext／128K 的結果由 37zz 重出一次（同一支 gen-report，會覆蓋這裡的輸出、只多列）。
# X2 被 X1 判定不跑時：沒有逐列檔 → 兩個 CSV 都不寫（X3 就是「只看時間」），exit 0。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_common.sh
export OMP_NUM_THREADS=4
m7run m16-x2-report $XPY gen-report --raw "$XD/x2/x2_*_s*.csv.gz" --selection $XR/x2_selection.json --out-dir $XR \
  || { nh "37z_x2_report：gen-report 失敗（$(lastrun m16-x2-report)）"; exit 1; }
