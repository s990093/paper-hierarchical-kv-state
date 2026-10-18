# 10/16 13:5xZ 使用者：長 context 移到只報告的工作前面（原名 91_longctx_pilot.sh；10 §6 補充）
# 10_longctx_plan §3.3：試跑（每個模型 × 長度 × 任務 1 題，正式的設定）→ longctx_plan_n.py 照規則決定題數，寫 queue/44c_longctx_screen.sh。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
RAW=/mlsteam/data/tiara/longctx_m10; mkdir -p $RAW
python3 $JOBS/longctx_plan_n.py jobs-pilot > $JOBS/queue/91_jobs.out 2>&1 || { cat $JOBS/queue/91_jobs.out; echo "44b：longctx_plan_n.py jobs-pilot 失敗（$JOBS/queue/91_jobs.out）→ 跳過" >> $JOBS/NEEDS_HUMAN; exit 0; }
pool $JOBS/jobs_91.json 4 || { echo "44b：pool jobs_91.json 有 job 失敗（$JOBS/pool_*.log）→ 跳過這段的其餘部分" >> $JOBS/NEEDS_HUMAN; exit 0; }
python3 $JOBS/longctx_plan_n.py plan || { echo "44b：longctx_plan_n.py plan 失敗 → 沒有 44c" >> $JOBS/NEEDS_HUMAN; exit 0; }
