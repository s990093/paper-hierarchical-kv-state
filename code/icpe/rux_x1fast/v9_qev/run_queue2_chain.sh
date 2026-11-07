#!/usr/bin/env bash
# PREREG 偏離 19 的正式模擬：SemiAnalysis 12 個變體，一個變體一個 run（rux_run.sh）跑 run_queue2_sweep.sh（A：qpost=2、B：qev；pfx=1、pgr=1）。
# 已經完成的變體跳過（有 run 目錄 exit_code=0、stdout 有 QUEUE2_SWEEPS_DONE、A／B 兩個 x1_sweep.csv.xz 都在）；
# 沒跑完的變體用新的 run 整個重跑（PREREG 偏離 19 §5「中斷就用新的 run 接著跑沒做完的變體」）。
# 每個變體跑完檢查 A、B 的 code_sha＝$Q2_CODE_SHA（和已完成的變體同一個引擎），不同就停。任何一個失敗就停。
# 用法：bash run_queue2_chain.sh [變體 ...（預設 12 個，照 N16→N72、s1→s3）]
set -euo pipefail
ROOT=/root/icpe-next
R=$ROOT/tools/rux_run.sh
SHA=${Q2_CODE_SHA:-3280b1c8799b+rs4d1cc219}
VS="$*"
[ -n "$VS" ] || VS=$(for N in 16 32 64 72; do for S in 1 2 3; do echo -n "semianalysis_N${N}_s${S} "; done; done)
done_run() {  # 印出這個變體最新的完成 run（沒有就空）
  local d
  for d in $(ls -td $ROOT/runs/*-icpe-rux-q2-sweep-$1 2>/dev/null); do
    [ "$(cat $d/exit_code 2>/dev/null)" = 0 ] && grep -q "QUEUE2_SWEEPS_DONE" $d/stdout.log \
      && [ -f $d/m/out/x1/x1_sweep.csv.xz ] && [ -f $d/q/out/x1/x1_sweep.csv.xz ] && { echo $d; return 0; }
  done
  return 0
}
for V in $VS; do
  d=$(done_run $V)
  if [ -n "$d" ]; then echo "SKIP $V（已完成：$d）"; continue; fi
  echo "== $V $(date -Is)"
  out=$($R q2-sweep-$V "bash \$REPO/code/icpe/rux_x1fast/v9_qev/run_queue2_sweep.sh nb:$V nb 16")
  echo "$out"
  echo "$out" | grep -q 'exit=0$' || { echo "CHAIN_Q2_FAIL $V"; exit 1; }
  d=$(done_run $V)
  [ -n "$d" ] || { echo "CHAIN_Q2_FAIL $V：找不到完成的 run"; exit 1; }
  for s in m q; do
    grep -q "code_sha=$SHA\$" $d/$s/sweep.log || { echo "CHAIN_Q2_FAIL $V：$s 段 code_sha 不是 $SHA"; grep code_sha $d/$s/sweep.log; exit 1; }
  done
done
echo "CHAIN_Q2_SWEEPS_OK $(date -Is)"
