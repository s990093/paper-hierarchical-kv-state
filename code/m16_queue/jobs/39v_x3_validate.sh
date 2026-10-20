# GPU-NEED: both
# GPU: 1
# NEED-GB: 80
# MAX-H: 2
# AFTER: 39z_x3_map
# X3 validate（docs/stage3_20261019/x3_time_howto.md §6）：map 判「有翻轉、值得驗證」（某個模式 GO_N5，或 X 有 X2 且 T-d 成立）時，
#   對 map 挑的 4–8 格做端到端還原實測，和預測比 → STOP-3。沒判值得驗證 → 印出原因、exit 0（不碰 GPU）。
#   第二階段還原器支援的動作走 m10_speed.py grid（2 launch × 5 rep、Cake＋全部載入）；其他動作走同一個 process 的載入線實測。
# GPU 時間〔估〕0.5–1.5 小時（依據 howto §7：每組 grid 約 4–6 分鐘 × 最多約 12 組；慢層的全部載入一次十幾秒、每項最多 60 秒）。
# HSA_ENABLE_SDMA 照第二階段：不設（ROCr 預設 1）。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
unset HSA_ENABLE_SDMA
export M10_BUSY_TEARDOWN_PAUSE=1
OUT=$RES/x3_time
W=$(python3 -I -c "import json;d=json.load(open('$OUT/flip_cells.json'));print('yes' if d.get('worth_validate') else 'no')") \
  || { nh "39v_x3_validate：讀不到 $OUT/flip_cells.json（map 沒跑完？）"; exit 1; }
if [ "$W" != yes ]; then
  echo "39v_x3_validate 不跑：$(python3 -I -c "import json;print(json.load(open('$OUT/flip_cells.json')).get('reason'))")"
  exit 0
fi
m7run m16-x3-validate python code/m16_xtime.py validate --map-dir $OUT --out $OUT/validate \
  || { nh "39v_x3_validate 失敗（$(lastrun m16-x3-validate)）"; exit 1; }
N=$(python3 -I -c "import json;print(json.load(open('$OUT/validate/validate_summary.json')).get('n_contaminated', 0))")
[ "$N" = 0 ] || nh "39v_x3_validate：$N 列被污染（$(lastrun m16-x3-validate)）→ STOP-3 沒算它們；要不要重量由人決定"
exit 0
