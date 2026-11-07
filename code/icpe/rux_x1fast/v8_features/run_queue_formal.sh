#!/usr/bin/env bash
# PREREG 偏離 11 B 的排隊敏感度（一階近似 qpost）：一個固定並行的變體 × 主結果的格（CPU 9 個容量）× CPU 17 × GPU 18（版本 C 的規則）。
# 用法：bash run_queue_formal.sh <--traces> <--rules> <變體目錄名> <版本 C 的 verC_headroom_components.csv> <偏離 10 的 x1_sweep.csv(.xz)> [workers]
# 步驟：1. sweep（RUX_QPOST=1）；2. explore_queue.py：端到端檢查（和偏離 10 的 sweep 比：只有 TTFT 衍生的欄位不同）＋上限表＋負載分級；
#       3. collect → results/icpe_next/queue/<變體>/。逐請求 npz 不存。
set -euo pipefail
TR=$1; RU=$2; VAR=$3; VERC=$4; NOQ=$5; W=${6:-16}
: "${RUN:?要在 rux_run.sh 的 run 目錄裡跑}"
V8=$REPO/code/icpe/rux_x1fast/v8_features
source $REPO/code/icpe/rux_x1fast/v6_slowcap/env_sim.sh
cd $REPO
P17="LRU LRU2 ARC S3FIFO GDSF PENS FREQ2Q LRU+NONE LRU+2HW LRU+2HA LRU+RPF O-BEL O-BELS O-DROP O-NOW O-PF O-ALL"
G18="LRU BEL TTLA:0.3 TTLA:1 TTLA:3 TTLA:10 TTLA:30 TTLA:100 TTLA:300 TTLB:0.3 TTLB:1 TTLB:3 TTLB:10 TTLB:30 TTLB:100 TTLB:300 APC BELP"
mkdir -p $RUN/a
echo "== queue sweep $(date -Is) load=$(cut -d' ' -f1-3 /proc/loadavg)"
RUX_QPOST=1 RUN_DIR="" /usr/bin/time -v nice -n 5 $ROOT/venv/bin/python -I $V8/sweep_v8.py --variant V1 --calib $CALIB --out-root $RUN/out x1 sweep \
  --traces $TR --rules $RU --cpu 8 16 32 64 128 256 512 1024 111.0625 --policies $P17 --slows ssd b32 nfs --modes cake --gpu-gib 111.0625 \
  --gpu-write wb --gpu-policies $G18 --pf-target cpu --workers $W --engine rust > $RUN/sweep.log 2> $RUN/sweep.err
grep -E '^(engine=|jobs=|sweep done)' $RUN/sweep.log
grep -q "py_fallback_jobs=0" $RUN/sweep.log || { echo "有 job 退回 Python → 停"; exit 1; }
( cd $RUN && $ROOT/venv/bin/python -I $V8/explore_queue.py --qsweep $RUN/out/x1/x1_sweep.csv --ref $VERC --noq-sweep $NOQ --out $RUN/a ) | tee $RUN/explore_queue.log
grep -q "QUEUE_RESULT E2E_OK" $RUN/explore_queue.log || { echo "端到端檢查沒過 → 停"; exit 1; }
$ROOT/venv/bin/python -I $V8/collect_v8.py queue $VAR --base results/icpe_next/queue --sweep $RUN/out/x1/x1_sweep.csv \
  --files $RUN/a/queue_rows.csv $RUN/a/queue_headroom_components.csv --runs $RUN \
  --note "PREREG 偏離 11 B 的排隊敏感度（qpost）；對照＝版本 C：$VERC；重負載（(LRU,LRU) 平均等待 ≥ 20 s）只報告"
echo "QUEUE_FORMAL_DONE $VAR"
