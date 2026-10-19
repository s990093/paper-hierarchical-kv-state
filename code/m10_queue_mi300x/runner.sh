#!/usr/bin/env bash
# 雙卡佇列 runner（10/19 加第二張卡後重寫；舊版只有一張卡、一把鎖 gpu.lock）。
#   ROLE=main（預設）：queue/，TIARA_GPU 預設 1（a6:00.0＝10/18 以前所有計時用的那張）。
#       stage 開頭有「# GPU-NEED: one」→ 只拿自己那張卡的鎖；沒寫（預設 both）→ 量時間，兩張卡都要：
#       先立 WANT_BOTH 旗子（helper 看到就讓出卡），再依序拿 gpu0、gpu1 的鎖（固定順序，不會互卡）。
#   ROLE=helper：queue_g0/，TIARA_GPU 預設 0（86:00.0），只跑只量品質的 stage（一律 one）。
#       WANT_BOTH 在時不開新的 stage；pool 看到 WANT_BOTH 會停掉自己的 job、exit 75 → 這裡把 stage 放回佇列，
#       之後重跑（品質 job 照題號續跑，已寫的題不重做）。
# 名稱一律照位元組排序（LC_ALL=C；重啟後 locale 改了，不能讓排序跟著變）。失敗 → 寫 log、這條 runner 停（規則 2）。
export LC_ALL=C
J=/tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs
export ROLE=${ROLE:-main}
if [ "$ROLE" = main ]; then Q=$J/queue; export TIARA_GPU=${TIARA_GPU:-1}; else Q=$J/queue_g0; export TIARA_GPU=${TIARA_GPU:-0}; fi
L0=/mlsteam/data/tiara/gpu0.lock; L1=/mlsteam/data/tiara/gpu1.lock; MYL=/mlsteam/data/tiara/gpu${TIARA_GPU}.lock
WANT=$J/WANT_BOTH
touch $L0 $L1
cd $J
idle=0
tag="runner:$ROLE gpu$TIARA_GPU"
while :; do
  if [ "$ROLE" = helper ] && [ -e "$WANT" ]; then sleep 20; continue; fi
  next=$(ls $Q/*.sh 2>/dev/null | sort | head -1)
  if [ -z "$next" ]; then idle=$((idle+1)); [ $idle -gt 360 ] && { echo "[$tag] EXIT idle 3h $(date -u +%FT%TZ)"; exit 0; }; sleep 30; continue; fi
  idle=0; name=$(basename $next .sh); mv $next $Q/$name.running
  need=$(grep -m1 -oE "^# GPU-NEED: *(one|both)" $Q/$name.running | awk '{print $3}'); [ -z "$need" ] && need=both
  [ "$ROLE" = helper ] && need=one
  echo "[$tag] START $name need=$need $(date -u +%FT%TZ)"
  if [ "$need" = both ]; then
    touch "$WANT"
    flock $L0 flock $L1 bash $Q/$name.running >> $Q/$name.log 2>&1; rc=$?
    rm -f "$WANT"
  else
    flock $MYL bash $Q/$name.running >> $Q/$name.log 2>&1; rc=$?
  fi
  if [ $rc -eq 0 ]; then mv $Q/$name.running $Q/$name.done; echo "[$tag] DONE $name $(date -u +%FT%TZ)";
  elif [ $rc -eq 75 ] && [ "$ROLE" = helper ]; then mv $Q/$name.running $Q/$name.sh; echo "[$tag] PREEMPTED $name（讓卡給 main 的量時間 stage；之後續跑） $(date -u +%FT%TZ)";
  else mv $Q/$name.running $Q/$name.failed; echo "[$tag] FAILED $name rc=$rc $(date -u +%FT%TZ)"; exit 1; fi
done
