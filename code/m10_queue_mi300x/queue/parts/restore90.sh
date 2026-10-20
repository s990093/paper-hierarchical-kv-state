#!/usr/bin/env bash
# 10/20 13:0xZ：i4e-qwen-T1 從卡 1 搬到卡 0 續跑（pool jobs_88r.json）。它跑完前 90（量時間、兩張卡獨佔）先放在 parts/，跑完才放回 queue/。
J=/tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs
while pgrep -f "pool.py jobs_88r.json" > /dev/null; do sleep 30; done
if grep -q "\[pool\] done i4e-qwen-T1-r rc=0" $J/pool_88r.out; then
  mv $J/parts/90_vllm_ttft_guarded.hold $J/queue/90_vllm_ttft_guarded.sh && echo "restore90: 90 放回 queue/ $(date -u +%FT%TZ)"
else
  echo "88r：i4e-qwen-T1 在卡 0 續跑失敗（$J/pool_88r.out）→ 90 還在 parts/ 沒放回" >> $J/NEEDS_HUMAN
fi
