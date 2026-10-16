# 10/16 13:5xZ 使用者：長 context 移到只報告的工作前面（原名 91_longctx_pilot.sh；10 §6 補充）
# 10_longctx_plan §3.3：試跑（每個模型 × 長度 × 任務 1 題，正式的設定）→ longctx_plan_n.py 照規則決定題數，寫 queue/44c_longctx_screen.sh。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
RAW=/mlsteam/data/tiara/longctx_m10; mkdir -p $RAW
python3 $JOBS/longctx_plan_n.py jobs-pilot > $JOBS/queue/91_jobs.out 2>&1 || { cat $JOBS/queue/91_jobs.out; exit 1; }
pool $JOBS/jobs_91.json 4 || exit 1
python3 $JOBS/longctx_plan_n.py plan || exit 1
