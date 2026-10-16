# Step 1 的判定（CPU；照凍結的 m10_analyze.py）＋ 依判定只調後面 GPU 工作的順序（route_after_step1.py）。
# 分析出錯不讓佇列停下（後面的 GPU 工作不依賴這裡）：寫 NEEDS_HUMAN、留完整錯誤，35 照預設順序跑。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
bash $JOBS/analyze_step1.sh > $JOBS/queue/31_analyze.out 2>&1 || { echo "analyze_step1 失敗（見 queue/31_analyze.out）" | tee -a $JOBS/NEEDS_HUMAN; tail -30 $JOBS/queue/31_analyze.out; exit 0; }
python3 $JOBS/route_after_step1.py || { echo "route_after_step1 失敗 rc=$?" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
