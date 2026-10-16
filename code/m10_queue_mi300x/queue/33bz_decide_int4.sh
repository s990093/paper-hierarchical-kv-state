# 第 2 階（INT4）端點跑完 → 跑凍結的判定（analyze_int4.sh），然後只調順序（route_after_int4.py）。10/16 03:5xZ 加的。
# 分析出錯或資料不全 → NEEDS_HUMAN、exit 0，佇列照預設順序繼續（33c 照跑，GPU 不空著）。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
bash $JOBS/analyze_int4.sh > $JOBS/queue/33bz_analyze.out 2>&1 || { echo "analyze_int4 失敗（見 33bz_analyze.out）" | tee -a $JOBS/NEEDS_HUMAN; tail -30 $JOBS/queue/33bz_analyze.out; exit 0; }
python3 $JOBS/route_after_int4.py || { echo "route_after_int4 失敗" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
