#!/usr/bin/env bash
# icpe-rux v6：功能打開時的正式對拍（Python 參考 x1 sweep 對 Rust v6 sweep_v6.py --engine rust），一份紀錄一次。
# 用法（在 rux_run.sh 裡）：validate.sh <trace> <rule> <輸出目錄> [子集…]（預設 main wt pfgpu noise）
# REUSE_PY=<前一次的輸出目錄>：Python 參考已經跑過就沿用（硬連結），只重跑 Rust。
# 每個子集：先跑 Rust、再跑 Python（各 12 worker，不同時跑）；CSV 逐格比、npz 逐陣列比；任何一個不同就停（set -e）。
set -euo pipefail
source "$REPO/code/icpe/rux_x1fast/v6_slowcap/env_sim.sh"
cd "$REPO"
export X1_NB_DIR=$ROOT/datasets/x1_nb X1_TRACE_DIR=$ROOT/datasets/mooncake/traces
T=$1; RU=$2; B=$3; shift 3
SUBS=${*:-main wt pfgpu noise}
SP7="LRU LRU2 ARC S3FIFO GDSF PENS BEL"
COMMON="--traces $T --rules $RU --modes cake --gpu-gib 111.0625 --slow-cap-frac 0.5 --workers 12"
declare -A SUB
SUB[main]="--cpu 8 111.0625 --policies LRU LRU2 PENS GDSF ARC S3FIFO LRU+RPF LRU+2HA O-BEL O-NOW --slows ssd nfs --gpu-write wb --gpu-policies LRU BEL TTLA:10 --pf-target cpu --slow-policies $SP7"
SUB[wt]="--cpu 111.0625 --policies LRU O-BEL --slows ssd nfs --gpu-write wt --gpu-policies LRU --pf-target cpu --slow-policies $SP7"
SUB[pfgpu]="--cpu 111.0625 --policies LRU+RPF O-PF --slows ssd nfs --gpu-write wb --gpu-policies LRU --pf-target gpu --slow-policies $SP7"
SUB[noise]="--cpu 111.0625 --policies LRU O-BEL --slows ssd nfs --gpu-write wb --gpu-policies LRU BEL --pf-target cpu --slow-policies BEL LRU --noise-models P M --noise-lambdas 0.3 --noise-draws B1 B2 --noise-seeds 0"
for s in $SUBS; do
  for eng in rust py; do
    d=$B/$s/$eng
    if [ $eng = py ] && [ -n "${REUSE_PY:-}" ] && [ -f "$REUSE_PY/$s/py/out/x1/x1_sweep.csv" ]; then
      echo "== $s py：沿用 $REUSE_PY/$s/py（Python 參考不受 Rust 修改影響；硬連結）"
      mkdir -p "$B/$s" && cp -al "$REUSE_PY/$s/py" "$d"
      continue
    fi
    mkdir -p "$d/req"
    if [ $eng = rust ]; then cmd=$RSV6; ex="--engine rust"; else cmd=$PYREF; ex=""; fi
    echo "== $s $eng $(date -Is) load=$(cut -d' ' -f1-3 /proc/loadavg)"
    RUN_DIR=$d/req X1_RUST_TIMING_LOG=$d/rust_timing.jsonl /usr/bin/time -v nice -n 10 $cmd --out-root "$d/out" x1 sweep $COMMON ${SUB[$s]} $ex \
      > "$d/sweep.log" 2> "$d/sweep.err"
    grep -E '^(engine=|jobs=|sweep done)' "$d/sweep.log" || true
    grep -E 'Elapsed|Maximum resident' "$d/sweep.err" || true
  done
  echo "== compare $s $(date -Is)"
  $CMP csv "$B/$s/py/out/x1/x1_sweep.csv" "$B/$s/rust/out/x1/x1_sweep.csv"
  $CMP npz "$B/$s/py/req" "$B/$s/rust/req"
done
echo "VALIDATE_DONE $T"
