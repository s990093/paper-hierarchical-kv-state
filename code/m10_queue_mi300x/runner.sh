#!/usr/bin/env bash
# 佇列：queue/NN_*.sh 依名稱順序一個接一個跑（每個 stage 拿 GPU 鎖）。失敗 → 停（規則 2）。佇列空了等新的，空 3 小時才結束。
J=/tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs; Q=$J/queue; cd $J
idle=0
while :; do
  next=$(ls $Q/*.sh 2>/dev/null | sort | head -1)
  if [ -z "$next" ]; then idle=$((idle+1)); [ $idle -gt 360 ] && { echo "[runner] EXIT idle 3h $(date -u +%FT%TZ)"; exit 0; }; sleep 30; continue; fi
  idle=0; name=$(basename $next .sh); mv $next $Q/$name.running
  echo "[runner] START $name $(date -u +%FT%TZ)"
  flock /mlsteam/data/tiara/gpu.lock bash $Q/$name.running > $Q/$name.log 2>&1; rc=$?
  if [ $rc -eq 0 ]; then mv $Q/$name.running $Q/$name.done; echo "[runner] DONE $name $(date -u +%FT%TZ)";
  else mv $Q/$name.running $Q/$name.failed; echo "[runner] FAILED $name rc=$rc $(date -u +%FT%TZ)"; exit 1; fi
done
