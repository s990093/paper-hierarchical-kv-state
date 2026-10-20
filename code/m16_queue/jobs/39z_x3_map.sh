# GPU-NEED: none
# AFTER: 39_x3_micro 37z_x2_report 35_s1_cpubench 21_i4_sdma1
# X3 map（只用 CPU；docs/stage3_20261019/x3_time_howto.md §3–§4）：用 micro 的實測＋第二階段的常數（每個都讀檔、記 run_id）＋S1 的無損比＋
#   X2 的品質可行集，算每一層 × duty 的 32K 還原時間、D／F／S／A、決策圖，照事先寫死的停止條件判定，挑 validate 的格子。
# cpu 佇列（runner 已經 nice 19、綁 0–3 號核心）；python -I、不 import torch。
# ⚠ AFTER 的 37z_x2_report 是 X2 report job 的暫定名字（由寫 X2 的 agent 決定）——名字對不上，這個 job 永遠不會開，要人核對。
#   X2 的輸出約定：results/m16_stage3_mi300x/x_stack/x2_feasible.csv（欄位 model、x3_segment、feasible、run_id；howto §3.4）。
#   沒有這個檔時 map 照跑，模式 X 標「只看時間」、不下結論。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
OUT=$RES/x3_time; mkdir -p $OUT
export OMP_NUM_THREADS=4
m7run m16-x3-map python -I code/m16_xtime.py map --micro-dir $OUT --out $OUT \
  || { nh "39z_x3_map 失敗（$(lastrun m16-x3-map)）"; exit 1; }
