# GPU-NEED: both
# GPU: 1
# NEED-GB: 80
# MAX-H: 2
# AFTER: 38z_x3_map2
# X3 validate v2（使用者 10/21 13:0xZ：「要，排在 X4 之後」）：決策圖 v2（38z，run 20261021-123143-m16-x3-map2）判 validate 要跑（模式 X 的 T-d 成立、挑 8 格）。
#   對那 8 格做端到端還原實測、和預測比 → STOP-3（門檻照 x3_time_howto.md §4，沒改）。作法同 39v_x3_validate.sh，只是讀 v2。
#   map_cells.csv（>1 MB）在 run 目錄（v2/BIG_FILES.md）→ 先把 v2 的 map 檔和它一起複製到工作目錄（不進 git），--map-dir 指到那裡。
# GPU 時間〔估〕0.5–1.5 小時（同 39v 的依據）。HSA_ENABLE_SDMA 照第二階段：不設。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
unset HSA_ENABLE_SDMA
export M10_BUSY_TEARDOWN_PAUSE=1
V2=$RES/x3_time/v2
BIG=/mlsteam/data/tiara/runs/20261021-123143-m16-x3-map2/x3_full
WK=/mlsteam/data/tiara/stage3/x3_map2_work
mkdir -p $WK && cp $V2/map_decision.csv $V2/map_summary.json $V2/flip_cells.json $V2/constants_used.csv $WK/ \
  && (cd $BIG && sha256sum -c SHA256SUMS) && cp $BIG/map_cells.csv $WK/ \
  || { nh "38j_x3_validate2：準備 map 工作目錄失敗（$WK、$BIG）"; exit 1; }
W=$(python3 -I -c "import json;d=json.load(open('$WK/flip_cells.json'));print('yes' if d.get('worth_validate') else 'no')") \
  || { nh "38j_x3_validate2：讀不到 $WK/flip_cells.json"; exit 1; }
if [ "$W" != yes ]; then
  echo "38j_x3_validate2 不跑：$(python3 -I -c "import json;print(json.load(open('$WK/flip_cells.json')).get('reason'))")"
  exit 0
fi
m7run m16-x3-validate2 python code/m16_xtime.py validate --map-dir $WK --out $V2/validate \
  || { nh "38j_x3_validate2 失敗（$(lastrun m16-x3-validate2)）"; exit 1; }
N=$(python3 -I -c "import json;print(json.load(open('$V2/validate/validate_summary.json')).get('n_contaminated', 0))")
[ "$N" = 0 ] || nh "38j_x3_validate2：$N 列被污染（$(lastrun m16-x3-validate2)）→ STOP-3 沒算它們；要不要重量由人決定"
exit 0
