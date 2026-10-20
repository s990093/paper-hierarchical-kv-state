#!/usr/bin/env bash
# 第三階段（m16）的佇列 runner。計畫：docs/stage3_20261019/experiment_matrix.md §6。
# 和第二階段共用機器的規則（使用者 10/19：第二階段先跑完；第二階段 session 10/19 13:2xZ 與 14:0xZ 的協調，RUNLOG_MI300X 15.13）：
#   - 不動第二階段的 queue/、queue_g0/、hold/、WANT_BOTH、XGPU_OK。唯一會碰的旗子是 WANT_CARD1（協調好的，見下）。
#   - 兩種角色（S3_ROLE）：
#     timing：量時間的 job（檔頭 GPU-NEED: both），一律在卡 1（a6:00.0，和第二階段所有計時同一張）。
#       (A) 第二階段兩個佇列都空、沒有 WANT_BOTH → 拿 gpu0、gpu1 兩把鎖（獨佔兩張卡）。
#       (B) 有 XGPU_OK、第二階段 main 佇列空、沒有 WANT_BOTH（helper 還在跑品質）→ touch WANT_CARD1（helper 的 pool 不再放 job 到卡 1、
#           已在卡 1 的 90 秒內移回卡 0）→ 等帳上沒有卡 1 的登記、amd-smi 卡 1 沒有 VRAM 行程 → 拿 gpu1.lock 量 → rm WANT_CARD1。
#           這種 run 設 S3_XGPU_COND=1，結果要註明「卡 0 同時有品質工作（XGPU 條件）」。
#     quality：只量品質的 job（GPU-NEED: one），在卡 1；第二階段 main 佇列空、沒有 WANT_BOTH、沒有 WANT_CARD1 才開；拿 gpu1.lock；
#       helper 還有 stage 時，登記需求 ≤ 96 GB（第二階段的上限：卡 1 上我們最多 2 個 job、合計 ≤ 96 GB；這條 runner 一次只跑 1 個）。
#       GPU: 0 的 job 只在第二階段兩個佇列都空時才開。
#   - 每個 GPU job：拿到鎖之後再查一次條件；登記共用顯存帳 vram_reserved.json；設 TIARA_GPU，以及 HIP／CUDA_VISIBLE_DEVICES（有 /mlsteam/data/tiara/USE_ROCR 時改設 ROCR_VISIBLE_DEVICES、unset HIP／CUDA）。
#   - CPU job（GPU-NEED: none）不能開 /dev/kfd（不准 import torch）；綁 0–3 號核心、nice 19；任何角色的 runner 都可以跑。
#   - 10/23 02:00Z（台灣 10:00）之後不開預計跑不完的 job（檔頭 MAX-H）。
#   - 失敗 → .failed＋NEEDS_HUMAN，這條 runner 停（CLAUDE.md 規則 2）。STOP 檔在 → 下一個 job 前停。
#   - 10/20 使用者：「基本上是要排在第二階段後」→ S3_AFTER_S2=1（預設）：任何 GPU job 都要第二階段兩個佇列都空、沒有 WANT_BOTH 才開；
#     上面的 (B) XGPU 條件與「helper 還在跑時用卡 1 跑品質」都關掉。S3_AFTER_S2=0 才回到 10/19 和第二階段協調的寬鬆規則。
# job 檔頭：# GPU-NEED: none|one|both   # GPU: 0|1（預設 1）   # NEED-GB: N   # MAX-H: h
export LC_ALL=C
S2=${S2_JOBS:-/tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs}
S3=${S3_JOBS:-/mlsteam/data/tiara/stage3/jobs}
ROLE=${S3_ROLE:-timing}
case $ROLE in timing) Q=$S3/queue ;; quality) Q=$S3/queue_q ;; *) echo "S3_ROLE 只能是 timing／quality"; exit 2 ;; esac
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LD=${S3_LOCKDIR:-/mlsteam/data/tiara}          # 測試時指到別處，不碰真的鎖
L0=$LD/gpu0.lock; L1=$LD/gpu1.lock
WC1=$S2/WANT_CARD1
CUTOFF=${S3_CUTOFF:-$(date -u -d "2026-10-23T02:00:00Z" +%s)}
WAIT_CARD1_S=${S3_WAIT_CARD1_S:-1800}
AFTER=${S3_AFTER_S2:-1}
mkdir -p $Q
tag="runner:s3:$ROLE"
log() { echo "[$tag] $* $(date -u +%FT%TZ)" | tee -a $S3/runner_$ROLE.log; }

has_stage() {      # $1＝queue 或 queue_g0：有 *.sh／*.running 就回 0。不能用 ls 多個 glob（有一個沒對到就回非 0，10/19 抓到）
  [ -n "$(find "$S2/$1" -maxdepth 1 \( -name '*.sh' -o -name '*.running' \) 2>/dev/null | head -1)" ]
}
s2_ok() { [ -d "$S2/queue" ]; }                  # 目錄不在（重啟沒還原）→ 一律當成不空
main_idle() { s2_ok && [ ! -e "$S2/WANT_BOTH" ] && ! has_stage queue; }
all_idle() { main_idle && ! has_stage queue_g0; }
helper_busy() { has_stage queue_g0; }
hdr() { grep -m1 -oE "^# $2: *[^ ]+" "$1" | awk '{print $3}'; }

run_locked() {     # $1 鎖指令 $2 檢查（shell 運算式）$3 gpu $4 need_gb $5 xgpu(0/1)；回傳 job 的 rc；99＝拿鎖後條件不符、98＝帳放不下
  local locks=$1 chk=$2 g=$3 gb=$4 xg=$5
  $locks bash -c '
    '"$(declare -f has_stage s2_ok main_idle all_idle helper_busy)"'
    S2='"$S2"'
    '"$chk"' || exit 99
    S2_JOBS='"$S2"' python3 '"$HERE"'/ledger.py reserve s3-'"$name"' '"$g"' '"$gb"' '"$$"' || exit 98
    export TIARA_GPU='"$g"' S3_XGPU_COND='"$xg"'
    if [ -e /mlsteam/data/tiara/USE_ROCR ]; then   # 同 code/m7_env.sh（RUNLOG_MI300X 15.16）：ROCr 層只開這張卡，amd-smi 只列在這張；行程內用 cuda:0
      export ROCR_VISIBLE_DEVICES='"$g"'; unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
    else
      unset ROCR_VISIBLE_DEVICES; export HIP_VISIBLE_DEVICES='"$g"' CUDA_VISIBLE_DEVICES='"$g"'
    fi
    bash '"$Q/$name.running"'; rc=$?
    S2_JOBS='"$S2"' python3 '"$HERE"'/ledger.py release s3-'"$name"'
    exit $rc' >> $Q/$name.log 2>&1
}

log "啟動（S2=$S2，S3=$S3，佇列 $Q，AFTER_S2=$AFTER，截止 $(date -u -d @$CUTOFF +%FT%TZ)）"
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
  if [ "$need" != none ] && [ $((now + ${maxh%.*} * 3600 + 3600)) -gt $CUTOFF ]; then
    mv $next $Q/$name.cutoff; log "SKIP $name（MAX-H=$maxh 會超過截止）"; continue
  fi
  rc=0
  if [ "$need" = none ]; then
    mv $next $Q/$name.running; log "START $name need=none"
    nice -n 19 taskset -c 0-3 bash $Q/$name.running >> $Q/$name.log 2>&1; rc=$?
  elif [ "$ROLE" = timing ]; then
    [ "$need" = both ] || { mv $next $Q/$name.failed; log "FAILED $name：timing 佇列只收 GPU-NEED: both"; exit 1; }
    if all_idle; then
      mv $next $Q/$name.running; log "START $name 獨佔兩張卡 gpu=1 need_gb=$gb"
      run_locked "flock $L0 flock $L1" all_idle 1 $gb 0; rc=$?
    elif [ "$AFTER" != 1 ] && [ -e "$S2/XGPU_OK" ] && main_idle; then
      touch $WC1; log "WANT_CARD1 立起（$name，XGPU 條件：卡 0 有第二階段的品質工作）"
      t0=$(date +%s); clear=0
      while [ $(( $(date +%s) - t0 )) -lt $WAIT_CARD1_S ]; do
        if S2_JOBS=$S2 python3 $HERE/ledger.py card_clear 1; then clear=1; break; fi
        sleep 15
      done
      if [ $clear = 0 ]; then rm -f $WC1; log "DEFER $name（等 $WAIT_CARD1_S s 卡 1 還沒清空）"; sleep 120; continue; fi
      mv $next $Q/$name.running; log "START $name XGPU 條件 gpu=1 need_gb=$gb"
      run_locked "flock $L1" "[ -e $S2/XGPU_OK ] && main_idle" 1 $gb 1; rc=$?
      rm -f $WC1; log "WANT_CARD1 拿掉（$name）"
    else
      sleep 60; continue
    fi
  else    # quality
    if [ "$g" = 0 ]; then
      all_idle || { sleep 60; continue; }
      mv $next $Q/$name.running; log "START $name quality gpu=0 need_gb=$gb"
      run_locked "flock $L0" all_idle 0 $gb 0; rc=$?
    else
      if [ "$AFTER" = 1 ]; then qchk="all_idle && [ ! -e $WC1 ]"; else qchk="main_idle && [ ! -e $WC1 ]"; fi
      eval "$qchk" || { sleep 60; continue; }
      if helper_busy && [ "${gb%.*}" -gt 96 ]; then
        mv $next $Q/$name.failed; log "FAILED $name：helper 還在跑時卡 1 上限 96 GB（NEED-GB=$gb）"; exit 1
      fi
      mv $next $Q/$name.running; log "START $name quality gpu=1 need_gb=$gb"
      run_locked "flock $L1" "$qchk" 1 $gb 0; rc=$?
    fi
  fi
  if [ $rc -eq 99 ] || [ $rc -eq 98 ]; then
    mv $Q/$name.running $Q/$name.sh; log "DEFER $name（rc=$rc：拿鎖後條件不符／顯存帳放不下）"; sleep 120; continue
  fi
  if [ $rc -eq 0 ]; then mv $Q/$name.running $Q/$name.done; log "DONE $name"
  else mv $Q/$name.running $Q/$name.failed; log "FAILED $name rc=$rc"
       echo "$(date -u +%FT%TZ) $name rc=$rc（看 $Q/$name.log）" >> $S3/NEEDS_HUMAN; exit 1; fi
done
