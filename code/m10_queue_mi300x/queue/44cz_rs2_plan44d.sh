# GPU-NEED: one　（10/19 雙卡：只用 CPU 寫 44d）
# 13 §7（CPU）：42c 的冠軍（pick-c）＋ Q4 帶過來的方法（43b 的 q4_pick.json）→ 寫 queue/44d_rs_longctx.sh。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
python3 $REPO/code/m10_dqsweep.py report > /dev/null || echo "44cz：dqsweep report 失敗" >> $JOBS/NEEDS_HUMAN
python3 $REPO/code/m10_restore2.py pick-c || echo "44cz：pick-c 失敗（44d 不跑冠軍）" >> $JOBS/NEEDS_HUMAN
[ -f $REPO/results/m10_restore2_mi300x/q4_pick.json ] || echo '{"pick": "P0", "note": "43b 沒有產生 q4_pick.json"}' > $REPO/results/m10_restore2_mi300x/q4_pick.json
[ -f $REPO/results/m10_restore2_mi300x/c_pick.json ] || echo '{"pick": "S0"}' > $REPO/results/m10_restore2_mi300x/c_pick.json
python3 $REPO/code/m10_restore2.py plan-44d --jobs $JOBS || { echo "44cz：plan-44d 失敗（44d 沒寫）" | tee -a $JOBS/NEEDS_HUMAN; }
exit 0
