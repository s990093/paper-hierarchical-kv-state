#!/usr/bin/env bash
# 容器重啟後把 GPU 佇列接回來（10/19 加第二張卡時寫）。用法：bash code/m10_queue_mi300x/restart_restore.sh
# 1) 從 NFS 備份把 scratchpad/jobs 放回同一個絕對路徑（所有 stage、lib.sh、pool.py 都寫死這個路徑）
# 2) 重啟時正在跑的 stage（*.running）：44a 換成會跳過已完成格子的 RESUME 版；其他 *.running 改回 *.sh 重跑
# 3) 不在這裡自動開 runner：先讓 Claude 檢查 GPU 數、卡的 BDF，再決定單／雙 runner
set -euo pipefail
J=/tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs
B=/mlsteam/data/tiara/jobs_backup/latest
if [ -d "$J/queue" ]; then echo "已經有 $J（沒有重啟？）不動"; exit 1; fi
mkdir -p "$(dirname $J)"
cp -a "$(readlink -f $B)" "$J"
for QD in "$J/queue" "$J/queue_g0"; do
[ -d "$QD" ] || continue
cd "$QD"
for f in *.running; do
  [ -e "$f" ] || continue
  n=${f%.running}
  if [ "$n" = 44a_longctx_speed ] && [ -f "$J/44a_longctx_speed.RESUME.sh" ]; then
    cp "$J/44a_longctx_speed.RESUME.sh" "$n.sh"; rm "$f"
  else
    mv "$f" "$n.sh"
  fi
  echo "接回：$n（重啟時正在跑；$(basename $QD)）"
done
done
rm -f "$J/WANT_BOTH"
echo "[runner] RESTART-RESTORE $(date -u +%FT%TZ)（容器重啟；從 $(readlink -f $B) 放回）" >> "$J/runner.log"
ls "$J/queue" "$J/queue_g0" 2>/dev/null | grep -v '\.log$\|\.out$\|\.done$'
cat <<EOF
10/19 起雙卡（RUNLOG 15.3）：先確認卡號（amd-smi list），再開兩條 runner：
  cd $J && ROLE=main TIARA_GPU=1 setsid nohup ./runner.sh >> runner.log 2>&1 < /dev/null &
  cd $J && ROLE=helper TIARA_GPU=0 setsid nohup ./runner.sh >> runner_g0.log 2>&1 < /dev/null &
（TIARA_GPU=1 要是 a6:00.0；不是的話照 amd-smi 的編號換）
EOF
amd-smi list 2>/dev/null | grep -E "GPU:|BDF" || true
