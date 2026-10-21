# GPU-NEED: none
# AFTER: 38a_x3_micro2 38bz_x2b_report
# X3 map v2（10/21 補缺口；只用 CPU；docs/stage3_20261019/x3_time_howto.md §10）：
#   micro＝第一版（results/m16_stage3_mi300x/x3_time，為主）＋10/21 的新段（x3_time/v2，只補第一版沒有的段；重疊的段當漂移錨點）
#   ＋寫入成本（x3_time/v2/wcost；只有 bf16 錨點與 wcost_<m>.csv）；
#   X2＝38bz_x2b_report 寫的 results/m16_stage3_mi300x/x_stack/v2/x2_feasible.csv（--x2 明講；檔不在 → 程式停，不當成「沒有 X2」）。
#   判定門檻照 howto §4 原文（程式的 TH 沒改）。輸出到 x3_time/v2/（程式擋「合併多個 micro 卻寫進第一版目錄」），另寫 write_cost.csv。
# cpu 佇列（runner 已經 nice 19、綁 0–3 號核心）；python -I、不 import torch。
# map_cells.csv 第一版是 4.8 MB：>1 MB 就照 CLAUDE.md §2 移到 run 目錄（x3_full/，附 SHA256SUMS），results 留 BIG_FILES.md 指過去。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
OUT=$RES/x3_time/v2; mkdir -p $OUT
X2=$RES/x_stack/v2/x2_feasible.csv
[ -e "$X2" ] || { nh "38z_x3_map2：$X2 不存在（38bz_x2b_report 沒寫出來？）"; exit 1; }
for m in llama qwen; do
  [ -e $OUT/micro_meta_$m.json ] || nh "38z_x3_map2：$OUT/micro_meta_$m.json 不存在（38a 的 $m 被污染兩次或沒跑）→ map 照跑，但 $m 的新段是 NOT_MEASURED"
  [ -e $OUT/wcost/wcost_$m.csv ] || nh "38z_x3_map2：$OUT/wcost/wcost_$m.csv 不存在 → $m 的寫入成本表是 NOT_MEASURED"
done
export OMP_NUM_THREADS=4
m7run m16-x3-map2 python -I code/m16_xtime.py map --micro-dir $RES/x3_time $OUT $OUT/wcost --x2 $X2 --out $OUT \
  || { nh "38z_x3_map2 失敗（$(lastrun m16-x3-map2)）"; exit 1; }
r=$(lastrun m16-x3-map2)
if [ -e $OUT/map_cells.csv ] && [ "$(stat -c %s $OUT/map_cells.csv)" -gt 1000000 ]; then
  mkdir -p $TIARA_RUNS/$r/x3_full
  mv $OUT/map_cells.csv $TIARA_RUNS/$r/x3_full/
  (cd $TIARA_RUNS/$r/x3_full && sha256sum map_cells.csv > SHA256SUMS)
  cat > $OUT/BIG_FILES.md <<EOF
# X3 map v2 的大檔（>1 MB，照 CLAUDE.md §2 不進 git）

位置：\`$TIARA_RUNS/$r/x3_full/\`（map_cells.csv；SHA256SUMS 同目錄）。由 38z_x3_map2 自動移出（$(date -u +%FT%TZ)）。
摘要在 map_report.md、map_summary.json、map_decision.csv、write_cost.csv。
EOF
  echo "38z_x3_map2：map_cells.csv 移到 $TIARA_RUNS/$r/x3_full/"
fi
grep -q "模式 X 不完整" $OUT/map_report.md && nh "38z_x3_map2：X2 判可行、但 micro 沒有時間的段（見 $OUT/map_report.md 的判定）→ 模式 X 不完整"
exit 0
