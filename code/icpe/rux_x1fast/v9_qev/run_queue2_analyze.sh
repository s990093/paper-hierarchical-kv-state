#!/usr/bin/env bash
# PREREG 偏離 19 的分析與收集（一個變體）：讀 run_queue2_sweep.sh 的 run 目錄，和修正版版本 C（icpe-data 第二段，v8_dev14 全部重跑）比。
# 用法：bash run_queue2_analyze.sh <變體目錄名> <run_queue2_sweep.sh 的 run 目錄> <修正版版本 C 的上限表 *_headroom_components.csv>
#                                <修正版版本 C 的 x1_sweep.csv(.xz)> <偏離 11 B 的 queue_rows.csv>
#   A：explore_queue.py（--queue-tag closed_post2；端到端：只有 TTFT 衍生的欄位可以和修正版版本 C 不同）→ results/icpe_next/queue1m/<變體>/
#   B：explore_queue2.py（結構檢查；和修正版版本 C、A、偏離 11 B 比）→ results/icpe_next/queue2/<變體>/
# 環境變數（只給端到端試跑用）：Q2_BASE（collect 的上層目錄，預設 results/icpe_next；可給絕對路徑）。任何一步失敗就停。
set -euo pipefail
VAR=$1; SR=$2; VERC=$3; NOQ=$4; QPR=$5
: "${RUN:?要在 rux_run.sh 的 run 目錄裡跑}"
BASE=${Q2_BASE:-results/icpe_next}
V9=$REPO/code/icpe/rux_x1fast/v9_qev
V8=$REPO/code/icpe/rux_x1fast/v8_features
cd $REPO
grep -q "QUEUE2_SWEEPS_DONE" $SR/stdout.log || { echo "$SR 的 sweep 沒有跑完 → 停"; exit 1; }
mkdir -p $RUN/m/a $RUN/q/a
echo "== A. 一階 qpost=2 的分析 $(date -Is)"
( cd $RUN/m && $ROOT/venv/bin/python -I $V8/explore_queue.py --qsweep $SR/m/out/x1/x1_sweep.csv --ref $VERC --noq-sweep $NOQ \
  --queue-tag closed_post2 --out $RUN/m/a ) > $RUN/m/explore_queue.log 2>&1
grep -q "QUEUE_RESULT E2E_OK" $RUN/m/explore_queue.log || { echo "A：端到端檢查沒過 → 停"; tail -5 $RUN/m/explore_queue.log; exit 1; }
grep "QUEUE_RESULT" $RUN/m/explore_queue.log
$ROOT/venv/bin/python -I $V8/collect_v8.py queue1m $VAR --base $BASE/queue1m --sweep $SR/m/out/x1/x1_sweep.csv \
  --files $RUN/m/a/queue_rows.csv $RUN/m/a/queue_headroom_components.csv --runs $SR $RUN \
  --note "PREREG 偏離 19 的對照：一階近似、到達不早於上一個完成（v9 qpost=2）、偏離 14 pfx=1；對照＝修正版版本 C：$VERC；重負載（(LRU,LRU) 平均等待 ≥ 20 s）只報告"
echo "== B. 二階的分析 $(date -Is)"
( cd $RUN/q && $ROOT/venv/bin/python -I $V9/explore_queue2.py --qsweep $SR/q/out/x1/x1_sweep.csv --ref $VERC --noq-sweep $NOQ \
  --qpost-rows m=$RUN/m/a/queue_rows.csv orig=$QPR --out $RUN/q/a ) > $RUN/q/explore_queue2.log 2>&1
grep -q "QUEUE2_RESULT E2E_OK" $RUN/q/explore_queue2.log || { echo "B：結構檢查沒過 → 停"; tail -5 $RUN/q/explore_queue2.log; exit 1; }
grep "QUEUE2_RESULT" $RUN/q/explore_queue2.log
$ROOT/venv/bin/python -I $V8/collect_v8.py queue2 $VAR --base $BASE/queue2 --sweep $SR/q/out/x1/x1_sweep.csv \
  --files $RUN/q/a/queue2_rows.csv $RUN/q/a/queue2_headroom_components.csv --runs $SR $RUN \
  --note "PREREG 偏離 19 的二階排隊（事件驅動 qev，v9＝v8_dev14＋qev）、偏離 14 pfx=1；比較 1 的基準＝修正版版本 C：$VERC；一階：m＝qpost=2（queue1m/）、orig＝$QPR（偏離 11 B，沒有 pfx）；重負載（二階 (LRU,LRU) 平均等待 ≥ 20 s）只報告"
echo "QUEUE2_ANALYZE_DONE $VAR"
