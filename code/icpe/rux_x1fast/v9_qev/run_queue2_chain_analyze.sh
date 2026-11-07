#!/usr/bin/env bash
# PREREG 偏離 19 的分析：基準＝TRX40 第二段的修正版 SemiAnalysis 版本 C（origin/icpe-next-trx 的 results/icpe_next/apc_d14/<變體>/）。
# 基準檔用 git show 取到 $ROOT/datasets/apc_d14_<commit>/（不改本分支），一個變體一個 run（rux_run.sh）跑 run_queue2_analyze.sh。
# sweep 用該變體最新的完成 run（exit_code=0、有 QUEUE2_SWEEPS_DONE）。任何一個失敗就停。
# 用法：bash run_queue2_chain_analyze.sh <icpe-next-trx 的 commit> [變體 ...（預設 12 個）]
set -euo pipefail
C=${1:-}; shift || true
[ -n "$C" ] || { echo "要給 icpe-next-trx 的 commit"; exit 1; }
ROOT=/root/icpe-next
REPO=/root/paper-hierarchical-kv-state
R=$ROOT/tools/rux_run.sh
BD=$ROOT/datasets/apc_d14_$C
VS="$*"
[ -n "$VS" ] || VS=$(for N in 16 32 64 72; do for S in 1 2 3; do echo -n "semianalysis_N${N}_s${S} "; done; done)
mkdir -p $BD
git -C $REPO fetch -q origin icpe-next-trx
for V in $VS; do
  mkdir -p $BD/$V
  for f in verC_headroom_components.csv x1_sweep.csv.xz MANIFEST.json; do
    git -C $REPO show $C:results/icpe_next/apc_d14/$V/$f > $BD/$V/$f
  done
  SR=""
  for d in $(ls -td $ROOT/runs/*-icpe-rux-q2-sweep-$V 2>/dev/null); do
    [ "$(cat $d/exit_code 2>/dev/null)" = 0 ] && grep -q "QUEUE2_SWEEPS_DONE" $d/stdout.log && { SR=$d; break; }
  done
  [ -n "$SR" ] || { echo "$V 沒有完成的 sweep run"; exit 1; }
  out=$($R q2-analyze-$V "bash \$REPO/code/icpe/rux_x1fast/v9_qev/run_queue2_analyze.sh $V $SR $BD/$V/verC_headroom_components.csv $BD/$V/x1_sweep.csv.xz \$REPO/results/icpe_next/queue/$V/queue_rows.csv")
  echo "$out"
  echo "$out" | grep -q 'exit=0$' || { echo "CHAIN_Q2_ANALYZE_FAIL $V"; exit 1; }
done
echo "CHAIN_Q2_ANALYZE_OK $VS"
