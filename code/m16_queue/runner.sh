#!/usr/bin/env bash
# 第三階段（m16）的佇列 runner。計畫：docs/stage3_20261019/experiment_matrix.md §6。
# 和第二階段共用機器的規則（使用者 10/19：第二階段先跑完；第二階段 session 10/19 13:2xZ 的協調）：
#   - 不動第二階段的 queue/、queue_g0/、hold/、WANT_BOTH、XGPU_OK。
#   - 要 GPU 的 job：第二階段 WANT_BOTH 不在、兩個佇列都沒有 *.sh／*.running 才開；拿同一組 gpu{0,1}.lock，
#     拿到鎖之後再查一次（拿鎖前後第二階段可能剛排了新的）；登記共用顯存帳 vram_reserved.json。
#   - job 一律設 HIP_VISIBLE_DEVICES／CUDA_VISIBLE_DEVICES／TIARA_GPU（分卡的 gpu_guard 要看）。
#   - CPU job（GPU-NEED: none）不能開 /dev/kfd（不准 import torch）；綁 0–3 號核心、nice 19。
#   - 10/23 02:00Z（台灣 10:00）之後不開任何預計跑不完的 job（檔頭 MAX-H）。
#   - 失敗 → .failed＋NEEDS_HUMAN，這條 runner 停（CLAUDE.md 規則 2）。
# job 檔頭：
#   # GPU-NEED: none|one|both     （both＝量時間：兩張卡的鎖都拿，job 本身只用 GPU: 那張）
#   # GPU: 0|1                    （one／both 時 job 用哪張；預設 1＝第二階段所有計時用的那張 a6:00.0）
#   # NEED-GB: 60                 （顯存峰值估計，登記用）
#   # MAX-H: 2                    （上限小時數；超過截止就不開）
export LC_ALL=C
S2=${S2_JOBS:-/tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs}
S3=${S3_JOBS:-/mlsteam/data/tiara/stage3/jobs}
Q=$S3/${S3_QNAME:-queue}                         # 兩條 runner：queue（量時間）、queue_g0（品質）
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LD=${S3_LOCKDIR:-/mlsteam/data/tiara}          # 測試時指到別處，不碰真的鎖
L0=$LD/gpu0.lock; L1=$LD/gpu1.lock
CUTOFF=${S3_CUTOFF:-$(date -u -d "2026-10-23T02:00:00Z" +%s)}
mkdir -p $Q
tag="runner:s3:${S3_QNAME:-queue}"
log() { echo "[$tag] $* $(date -u +%FT%TZ)" | tee -a $S3/runner_${S3_QNAME:-queue}.log; }

stage2_idle() {
  [ -d "$S2/queue" ] || return 1                     # 目錄不在（重啟沒還原）→ 當成不空，不開 GPU job
  [ -e "$S2/WANT_BOTH" ] && return 1
  # 不能用 ls 多個 glob：只要有一個沒對到 ls 就回非 0，會被當成「空」（10/19 自測抓到）
  [ -n "$(find "$S2/queue" "$S2/queue_g0" -maxdepth 1 \( -name '*.sh' -o -name '*.running' \) 2>/dev/null | head -1)" ] && return 1
  return 0
}
hdr() { grep -m1 -oE "^# $2: *[^ ]+" "$1" | awk '{print $3}'; }

log "啟動（S2=$S2，S3=$S3，截止 $(date -u -d @$CUTOFF +%FT%TZ)）"
while :; do
  [ -e $S3/STOP ] && { log "看到 STOP，結束"; exit 0; }
  next=$(ls $Q/*.sh 2>/dev/null | sort | head -1)
  [ -z "$next" ] && { sleep 30; continue; }
  name=$(basename $next .sh)
  need=$(hdr $next GPU-NEED); need=${need:-both}
  g=$(hdr $next GPU); g=${g:-1}
  gb=$(hdr $next NEED-GB); gb=${gb:-0}
  maxh=$(hdr $next MAX-H); maxh=${maxh:-2}
  now=$(date -u +%s)
  if [ $((now + ${maxh%.*} * 3600 + 3600)) -gt $CUTOFF ] && [ "$need" != none ]; then
    mv $next $Q/$name.cutoff; log "SKIP $name（MAX-H=$maxh 會超過截止）"; continue
  fi
  if [ "$need" = none ]; then
    mv $next $Q/$name.running; log "START $name need=none"
    nice -n 19 taskset -c 0-3 bash $Q/$name.running >> $Q/$name.log 2>&1; rc=$?
  else
    if ! stage2_idle; then sleep 60; continue; fi
    if [ "$need" = both ]; then LOCKS="flock $L0 flock $L1"; else LOCKS="flock $LD/gpu$g.lock"; fi
    mv $next $Q/$name.running; log "START $name need=$need gpu=$g need_gb=$gb"
    # 拿到鎖之後：再查一次第二階段、登記顯存帳，再跑；第二階段在這中間排了東西 → exit 99，放回佇列
    $LOCKS bash -c '
      '"$(declare -f stage2_idle)"'
      S2='"$S2"'
      stage2_idle || exit 99
      S2_JOBS='"$S2"' python3 '"$HERE"'/ledger.py reserve s3-'"$name"' '"$g"' '"$gb"' '"$$"' || exit 98
      export TIARA_GPU='"$g"' HIP_VISIBLE_DEVICES='"$g"' CUDA_VISIBLE_DEVICES='"$g"'
      bash '"$Q/$name.running"'; rc=$?
      S2_JOBS='"$S2"' python3 '"$HERE"'/ledger.py release s3-'"$name"'
      exit $rc' >> $Q/$name.log 2>&1; rc=$?
    if [ $rc -eq 99 ] || [ $rc -eq 98 ]; then
      mv $Q/$name.running $Q/$name.sh; log "DEFER $name（rc=$rc：拿鎖後第二階段不空／顯存帳放不下）"; sleep 120; continue
    fi
  fi
  if [ $rc -eq 0 ]; then mv $Q/$name.running $Q/$name.done; log "DONE $name"
  else mv $Q/$name.running $Q/$name.failed; log "FAILED $name rc=$rc"
       echo "$(date -u +%FT%TZ) $name rc=$rc（看 $Q/$name.log）" >> $S3/NEEDS_HUMAN; exit 1; fi
done
