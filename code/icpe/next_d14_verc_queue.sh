#!/usr/bin/env bash
# next_d14_verc_queue.sh — icpe-data：第二段（修正版版本 C）的佇列。每個變體一個 §4.1 的 run 目錄，依序呼叫 next_d14_verc.sh。
# 任何一個變體失敗 → 整個佇列停下來（CLAUDE.md 規則 2），印 run 目錄；跑完（D14_VERC_DONE）的變體記在 $ROOT/runs/next_d14_verc_done.txt，重跑會跳過。
# 用法：D14REPO=… DATAREPO=… ROOT=… bash next_d14_verc_queue.sh <workers> <變體> [<變體> ...]
# 變體名＝results/icpe_next/apc/ 的目錄名；家族、--traces、--rules、第一段檢查表、舊的 apc sweep 由名稱推出：
#   bailian_* → bl、fi_<週> → fi（fi:<週> session_id）、tracelab_* → tl、sharegpt_* → sg、wildchat_* → wc、miniswe_* → ms、openhands_* → oh、
#   semianalysis_* → sa、ultrachat_* → uc、lmsys_* → lm、mooncake_<conversation|toolagent> → mc（"R1 R2"）
# 第一段檢查表＝$DATAREPO/results/icpe_next/b/analysis_d14/<家族>_p1/（manifest 的 run_dir 裡的 table_vs_verB.log 要 IDENTICAL）。
# 舊的 apc sweep：$DATAREPO/results/icpe_next/apc/<變體>/x1_sweep.csv.xz，沒有就用 $ROOT/d14_old_apc/<變體>/（sha256 已對過 MANIFEST）。
set -euo pipefail
W=$1; shift
: "${D14REPO:?}" "${DATAREPO:?}" "${ROOT:?}"
TAG=${NEXT_B_TAG:-icpe-data}
DONE=$ROOT/runs/next_d14_verc_done.txt
touch $DONE
PY="${NEXT_B_PY:-$ROOT/venv/bin/python} -I"
for V in "$@"; do
  if grep -qx "$V" $DONE; then echo "[skip] $V 已完成"; continue; fi
  case $V in
    bailian_*) F=bl; TR=nb:$V; RU=nb;;
    fi_*) F=fi; TR=fi:${V#fi_}; RU=session_id;;
    tracelab_*) F=tl; TR=nb:$V; RU=nb;;
    sharegpt_*) F=sg; TR=nb:$V; RU=nb;;
    wildchat_*) F=wc; TR=nb:$V; RU=nb;;
    miniswe_*) F=ms; TR=nb:$V; RU=nb;;
    openhands_*) F=oh; TR=nb:$V; RU=nb;;
    semianalysis_*) F=sa; TR=nb:$V; RU=nb;;
    ultrachat_*) F=uc; TR=nb:$V; RU=nb;;
    lmsys_*) F=lm; TR=nb:$V; RU=nb;;
    mooncake_*) F=mc; TR=${V#mooncake_}; RU="R1 R2";;
    *) echo "不認得的變體 $V"; exit 1;;
  esac
  T=$DATAREPO/results/icpe_next/b/analysis_d14/${F}_p1
  [ -f $T/manifest.json ] || { echo "$V：第一段檢查表 $T 還沒有 → 停"; exit 1; }
  VR=$($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['run_dir'])" $T/manifest.json)
  OLD=$DATAREPO/results/icpe_next/apc/$V/x1_sweep.csv.xz
  [ -f $OLD ] || OLD=$ROOT/d14_old_apc/$V/x1_sweep.csv.xz
  [ -f $OLD ] || { echo "$V：找不到舊的 apc sweep → 停"; exit 1; }
  RUN_ID=$(date +%Y%m%d-%H%M%S)-$TAG-d14verc-$V; RUN=$ROOT/runs/$RUN_ID; mkdir -p $RUN
  CMD="RUN_ID=$RUN_ID RUN=$RUN D14REPO=$D14REPO DATAREPO=$DATAREPO ROOT=$ROOT bash $DATAREPO/code/icpe/next_d14_verc.sh '$TR' '$RU' $V $T/${F}_p1_headroom_components.csv.xz $VR/table_vs_verB.log $OLD $W"
  printf '%s\n' "$CMD" > $RUN/cmd.sh
  { date -Is; hostname; echo "data HEAD $(git -C $DATAREPO rev-parse HEAD)"; echo "d14 HEAD $(git -C $D14REPO rev-parse HEAD)";
    git -C $D14REPO status --porcelain | head -5; uptime; free -g | head -2; } > $RUN/context.txt
  echo "[$(date '+%m-%d %H:%M:%S')] start $V run=$RUN_ID"
  set +e
  RUN_ID=$RUN_ID RUN=$RUN D14REPO=$D14REPO DATAREPO=$DATAREPO ROOT=$ROOT bash $DATAREPO/code/icpe/next_d14_verc.sh "$TR" "$RU" $V \
    $T/${F}_p1_headroom_components.csv.xz $VR/table_vs_verB.log $OLD $W > $RUN/stdout.log 2> $RUN/stderr.log
  rc=$?; set -e
  echo $rc > $RUN/exit_code
  echo "[$(date '+%m-%d %H:%M:%S')] end   $V rc=$rc"
  if [ $rc -ne 0 ] || ! grep -q "D14_VERC_DONE $V" $RUN/stdout.log; then
    echo "VARIANT FAILED $V rc=$rc run=$RUN"; tail -15 $RUN/stdout.log; tail -15 $RUN/stderr.log; exit 1
  fi
  echo $V >> $DONE
done
echo "D14_VERC_QUEUE_DONE"
