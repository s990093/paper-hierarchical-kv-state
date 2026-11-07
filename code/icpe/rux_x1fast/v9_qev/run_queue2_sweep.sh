#!/usr/bin/env bash
# PREREG 偏離 19 的模擬（一個固定並行的變體）：主結果的格（CPU 9 個容量）× CPU 17 × GPU 18（版本 C 的規則）× 6 個時間設定。
# 全部開偏離 14 的 pfx=1、pgr=1（修正版取代版本 C 的 O-PF／O-ALL 列；pgr 只在預取進 GPU 時有作用，這裡預取到 CPU）。兩段：
#   A. 一階、到達不早於上一個完成（v9 qpost=2；使用者 11/7 約 15:40「二階改，一階也重跑」）→ $RUN/m/out/x1/x1_sweep.csv
#   B. 二階（事件驅動 qev；每個時間設定各跑一次模擬）→ $RUN/q/out/x1/x1_sweep.csv
# 分析和收集在 run_queue2_analyze.sh（比較的基準＝icpe-data 第二段的修正版版本 C；使用者 11/7 約 16:3x：版本 C 全部用 v8_dev14 重跑）。
# 用法：bash run_queue2_sweep.sh <--traces> <--rules> [workers] [--cpu 的容量（預設主結果的 9 個）]。任何一步失敗就停。逐請求 npz 不存。
set -euo pipefail
TR=$1; RU=$2; W=${3:-16}; CPUS=${4:-"8 16 32 64 128 256 512 1024 111.0625"}
: "${RUN:?要在 rux_run.sh 的 run 目錄裡跑}"
V9=$REPO/code/icpe/rux_x1fast/v9_qev
source $REPO/code/icpe/rux_x1fast/v6_slowcap/env_sim.sh
cd $REPO
export RUX_PFX=1 RUX_PGR=1
P17="LRU LRU2 ARC S3FIFO GDSF PENS FREQ2Q LRU+NONE LRU+2HW LRU+2HA LRU+RPF O-BEL O-BELS O-DROP O-NOW O-PF O-ALL"
G18="LRU BEL TTLA:0.3 TTLA:1 TTLA:3 TTLA:10 TTLA:30 TTLA:100 TTLA:300 TTLB:0.3 TTLB:1 TTLB:3 TTLB:10 TTLB:30 TTLB:100 TTLB:300 APC BELP"
GRID="--traces $TR --rules $RU --cpu $CPUS --policies $P17 --slows ssd b32 nfs --modes cake --gpu-gib 111.0625 --gpu-write wb --gpu-policies $G18 --pf-target cpu --workers $W --engine rust"
mkdir -p $RUN/m $RUN/q
echo "== A. qpost=2 sweep $(date -Is) load=$(cut -d' ' -f1-3 /proc/loadavg) cpus=$CPUS"
RUX_QPOST2=1 RUN_DIR="" /usr/bin/time -v nice -n 5 $ROOT/venv/bin/python -I $V9/sweep_v9.py --variant V1 --calib $CALIB --out-root $RUN/m/out x1 sweep $GRID \
  > $RUN/m/sweep.log 2> $RUN/m/sweep.err
grep -E '^(sweep_v9|engine=|jobs=|sweep done)' $RUN/m/sweep.log
grep -q "py_fallback_jobs=0" $RUN/m/sweep.log || { echo "A：有 job 退回 Python → 停"; exit 1; }
grep -q "qpost2=True" $RUN/m/sweep.log && grep -q "pf_expire=True" $RUN/m/sweep.log || { echo "A：sweep 沒有開 qpost2／pfx → 停"; exit 1; }
echo "== B. qev sweep $(date -Is) load=$(cut -d' ' -f1-3 /proc/loadavg)"
RUX_QEV=1 RUN_DIR="" /usr/bin/time -v nice -n 5 $ROOT/venv/bin/python -I $V9/sweep_v9.py --variant V1 --calib $CALIB --out-root $RUN/q/out x1 sweep $GRID \
  > $RUN/q/sweep.log 2> $RUN/q/sweep.err
grep -E '^(sweep_v9|engine=|jobs=|sweep done)' $RUN/q/sweep.log
grep -q "py_fallback_jobs=0" $RUN/q/sweep.log || { echo "B：有 job 退回 Python → 停"; exit 1; }
grep -q "qev=True" $RUN/q/sweep.log && grep -q "pf_expire=True" $RUN/q/sweep.log || { echo "B：sweep 沒有開 qev／pfx → 停"; exit 1; }
xz -T4 -k $RUN/m/out/x1/x1_sweep.csv $RUN/q/out/x1/x1_sweep.csv
echo "QUEUE2_SWEEPS_DONE $TR $(date -Is)"
