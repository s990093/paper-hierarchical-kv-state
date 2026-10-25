# GPU-NEED: both
# GPU: 1
# NEED-GB: 80
# MAX-H: 2
# X3 翻轉格子（10/22 使用者要求補跑）：2 模型 × 3 層 × 4 個 GPU 忙碌度 × 6 個動作＝144 個端到端還原量測，看「最快的格式會不會隨 GPU 忙碌翻轉」
# （38j 只有 4 對）。量法＝m16_xtime.py validate（和 38j 同一條路徑），格子＝code/m16_x3grid.py build；事先寫死的判定：
# experiment_log.md「X3 翻轉格子」。GPU 時間〔估〕：38j 的 25 列 13 分鐘 → 144 列約 45–75 分鐘。HSA_ENABLE_SDMA 照第二階段：不設。
# 大的中間檔在 /mlsteam/data/tiara/stage3/x3grid/；results 只放 validate_cells.csv、validate_summary.json、x3grid_report.*。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
unset HSA_ENABLE_SDMA
export M10_BUSY_TEARDOWN_PAUSE=1
JOB=63_x3_flipgrid
MAP=/mlsteam/data/tiara/stage3/x3grid/map
OUTV=/mlsteam/data/tiara/stage3/x3grid/validate
OUTR=$RES/x3_time/v3grid
mkdir -p $OUTR
m7run m16-x3grid-build python -I code/m16_x3grid.py build --out $MAP || { nh "$JOB：build 失敗（$(lastrun m16-x3grid-build)）"; exit 1; }
m7run m16-x3grid-validate python code/m16_xtime.py validate --map-dir $MAP --out $OUTV --max-cells 24 \
  || { nh "$JOB：validate 失敗（$(lastrun m16-x3grid-validate)）"; exit 1; }
cp $OUTV/validate_cells.csv $OUTV/validate_summary.json $OUTR/
m7run m16-x3grid-report python -I code/m16_x3grid.py report --val $OUTV --out-dir $OUTR || { nh "$JOB：report 失敗（$(lastrun m16-x3grid-report)）"; exit 1; }
N=$(python3 -I -c "import json;print(json.load(open('$OUTR/x3grid_report.json')).get('n_contaminated', 0))")
[ "$N" = 0 ] || nh "$JOB：$N 列被污染（$(lastrun m16-x3grid-validate)）→ 判定沒算它們；要不要重量由人決定"
echo "$JOB 結束 $(date -u +%FT%TZ)"
exit 0
