#!/usr/bin/env bash
# PREREG 偏離 19 的收尾：12 個變體的 sweep 都完成後，(1) 分析還沒收集的變體（run_queue2_chain_analyze.sh），
# (2) 12 個變體的加總表（agg_queue2.py，一個 run）→ results/icpe_next/queue2/aggregate_SA12/（agg_rows.csv 不放 git，留在 run 目錄）。
# 不 commit、不 push。任何一步失敗就停。
# 用法：bash run_queue2_finalize.sh <icpe-next-trx 的 commit（修正版版本 C）>
set -euo pipefail
C=${1:-}
[ -n "$C" ] || { echo "要給 icpe-next-trx 的 commit"; exit 1; }
ROOT=/root/icpe-next
REPO=/root/paper-hierarchical-kv-state
V9=$REPO/code/icpe/rux_x1fast/v9_qev
R=$ROOT/tools/rux_run.sh
ALL=$(for N in 16 32 64 72; do for S in 1 2 3; do echo -n "semianalysis_N${N}_s${S} "; done; done)
TODO=""
for V in $ALL; do [ -f $REPO/results/icpe_next/queue2/$V/queue2_rows.csv ] || TODO="$TODO $V"; done
echo "== 分析：${TODO:-（沒有）} $(date -Is)"
[ -z "$TODO" ] || bash $V9/run_queue2_chain_analyze.sh $C $TODO
for V in $ALL; do
  for f in queue1m/$V/queue_rows.csv queue2/$V/queue2_rows.csv queue/$V/queue_rows.csv; do
    [ -f $REPO/results/icpe_next/$f ] || { echo "FINALIZE_FAIL 少 $f"; exit 1; }
  done
done
echo "== 加總表 $(date -Is)"
out=$($R q2-agg-SA12 "cd \$REPO && \$ROOT/venv/bin/python -I code/icpe/rux_x1fast/v9_qev/agg_queue2.py --base results/icpe_next --variants $ALL --out \$RUN/a")
echo "$out"
echo "$out" | grep -q 'exit=0$' || { echo "FINALIZE_FAIL 加總表"; exit 1; }
RUN=$(echo "$out" | sed -n 's/^RUN=\([^ ]*\) exit=0$/\1/p')
grep -q "AGG_RESULT OK variants=12" $RUN/stdout.log || { echo "FINALIZE_FAIL AGG_RESULT"; tail -2 $RUN/stdout.log; exit 1; }
D=$REPO/results/icpe_next/queue2/aggregate_SA12
mkdir -p $D
for f in agg_summary.md agg_scen.csv agg_feedback.csv agg_arrival.csv agg_crosstab.csv agg_wait.csv AGG_MANIFEST.json; do cp $RUN/a/$f $D/; done
echo "FINALIZE_OK $RUN → $D $(date -Is)"
