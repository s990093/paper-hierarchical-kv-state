#!/usr/bin/env bash
# 雙卡佇列 runner（v2，10/19 11:xxZ）。v1 見 RUNLOG 15.3；v2 加：
#   (1) XGPU_OK 旗子在（44cb 的干擾實驗過了）→ 量時間的 stage 只拿自己那張卡的鎖、不立 WANT_BOTH，helper 照跑；
#       沒有旗子 → 照 v1：兩張卡獨佔（WANT_BOTH＋依序拿 gpu0、gpu1 的鎖）。
#   (2) RELOAD_<role> 旗子 → 下一個 stage 開始前 exec 新版的自己（換 runner 不用中斷正在跑的 stage）。
#   (3) 啟動時接手上一條 runner 留下、還在跑的 *.running（等它結束、標成 .done），順序不會亂。
#   ROLE=main：queue/，TIARA_GPU 預設 1（a6:00.0）。ROLE=helper：queue_g0/，TIARA_GPU 預設 0（86:00.0），只跑品質。
# 名稱照位元組排序（LC_ALL=C）。失敗 → 記 log、這條 runner 停（規則 2）。
export LC_ALL=C
J=/tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs
export ROLE=${ROLE:-main}
if [ "$ROLE" = main ]; then Q=$J/queue; export TIARA_GPU=${TIARA_GPU:-1}; else Q=$J/queue_g0; export TIARA_GPU=${TIARA_GPU:-0}; fi
L0=/mlsteam/data/tiara/gpu0.lock; L1=/mlsteam/data/tiara/gpu1.lock; MYL=/mlsteam/data/tiara/gpu${TIARA_GPU}.lock
WANT=$J/WANT_BOTH
touch $L0 $L1
cd $J
tag="runner:$ROLE gpu$TIARA_GPU"
echo "[$tag] v2 啟動 $(date -u +%FT%TZ)（XGPU_OK=$([ -e $J/XGPU_OK ] && echo 有 || echo 沒有)）"
# (3) 接手還在跑的 stage
for f in $(ls $Q/*.running 2>/dev/null); do
  name=$(basename $f .running)
  if pgrep -f "bash $f" > /dev/null; then
    echo "[$tag] ADOPT $name（上一條 runner 開的，等它結束）$(date -u +%FT%TZ)"
    while pgrep -f "bash $f" > /dev/null; do sleep 30; done
    mv $f $Q/$name.done; echo "[$tag] DONE $name（接手；結束碼不明，看 $name.log）$(date -u +%FT%TZ)"
  fi
done
idle=0
while :; do
  if [ -e "$J/RELOAD_$ROLE" ]; then rm -f "$J/RELOAD_$ROLE"; echo "[$tag] RELOAD $(date -u +%FT%TZ)"; exec "$0"; fi
  if [ "$ROLE" = helper ] && [ -e "$WANT" ]; then sleep 20; continue; fi
  next=$(ls $Q/*.sh 2>/dev/null | sort | head -1)
  if [ -z "$next" ]; then idle=$((idle+1)); [ $idle -gt 360 ] && { echo "[$tag] EXIT idle 3h $(date -u +%FT%TZ)"; exit 0; }; sleep 30; continue; fi
  idle=0; name=$(basename $next .sh); mv $next $Q/$name.running
  need=$(grep -m1 -oE "^# GPU-NEED: *(one|both)" $Q/$name.running | awk '{print $3}'); [ -z "$need" ] && need=both
  [ "$ROLE" = helper ] && need=one
  shared=""; [ "$need" = both ] && [ -e "$J/XGPU_OK" ] && shared="(xgpu)"
  echo "[$tag] START $name need=$need$shared $(date -u +%FT%TZ)"
  if [ "$need" = both ] && [ -z "$shared" ]; then
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
