#!/usr/bin/env bash
# PREREG 偏離 10 的正式 APC 重算：一個紀錄變體的主結果格 × CPU 17 × GPU 18（版本 B 的 16＋APC＋BELP）。
# 用法（在 run 目錄裡；REPO、ROOT、CALIB 等照 env_sim.sh；X1_NB_DIR／X1_TRACE_DIR／X1_FI_DIR 照紀錄）：
#   bash run_apc_formal.sh <--traces 的值> <--rules 的值（多個用空白，整個加引號）> <版本 B 的表名：mc|sa|bl|tl|fi|cw> <變體目錄名> [workers]
# 例：bash run_apc_formal.sh nb:semianalysis_N16_s1 nb sa semianalysis_N16_s1 16
#     bash run_apc_formal.sh toolagent "R1 R2" mc mooncake_toolagent 16
# 步驟（任何一步失敗就停，exit ≠ 0）：
#   1. sweep_v8：CPU 9 個容量 × 17 個 CPU 規則 × ssd／b32／nfs × Cake × GPU 111.0625 GiB wb × 18 個 GPU 規則、預取到 CPU。
#   2. explore_apc.py：只用版本 B 的 16 個規則重算上限表，和版本 B 的表全部格、全部元件、7 欄逐值相同（偏離 10 第 4 點）→ 必須 REPRODUCED。
#   3. analyze_v8.py：全部 18 個規則 → 版本 C 的上限表 verC_headroom_components.csv（APC 是不預測、BELP 只進 O-GPU 的 ④）。
#   4. collect_v8.py → results/icpe_next/apc/<變體>/（x1_sweep.csv.xz、verC／base 表、explore_apc 的列、MANIFEST.json）。
set -euo pipefail
TR=$1; RULES=$2; FAM=$3; VAR=$4; W=${5:-16}
: "${RUN:?要在 rux_run.sh 的 run 目錄裡跑}"
V8=$REPO/code/icpe/rux_x1fast/v8_features
source $REPO/code/icpe/rux_x1fast/v6_slowcap/env_sim.sh
cd $REPO
P17="LRU LRU2 ARC S3FIFO GDSF PENS FREQ2Q LRU+NONE LRU+2HW LRU+2HA LRU+RPF O-BEL O-BELS O-DROP O-NOW O-PF O-ALL"
G18="LRU BEL TTLA:0.3 TTLA:1 TTLA:3 TTLA:10 TTLA:30 TTLA:100 TTLA:300 TTLB:0.3 TTLB:1 TTLB:3 TTLB:10 TTLB:30 TTLB:100 TTLB:300 APC BELP"
mkdir -p $RUN/req $RUN/a
echo "== sweep $(date -Is) load=$(cut -d' ' -f1-3 /proc/loadavg)"
RUN_DIR=$RUN/req X1_RUST_TIMING_LOG=$RUN/rust_timing.jsonl /usr/bin/time -v nice -n 5 $ROOT/venv/bin/python -I $V8/sweep_v8.py --variant V1 --calib $CALIB \
  --out-root $RUN/out x1 sweep --traces $TR --rules $RULES --cpu 8 16 32 64 128 256 512 1024 111.0625 --policies $P17 \
  --slows ssd b32 nfs --modes cake --gpu-gib 111.0625 --gpu-write wb --gpu-policies $G18 --pf-target cpu --workers $W --engine rust \
  > $RUN/sweep.log 2> $RUN/sweep.err
grep -E '^(engine=|jobs=|sweep done)' $RUN/sweep.log
grep -q "py_fallback_jobs=0" $RUN/sweep.log || { echo "有 job 退回 Python → 停"; exit 1; }
git -C $REPO show origin/icpe-next-main:results/icpe_next/main/verB/${FAM}_p1_headroom_components.csv.xz | xz -dc > $RUN/verB_${FAM}_p1.csv
echo "== 重現檢查＋探索表 $(date -Is)"
( cd $RUN && $ROOT/venv/bin/python -I $V8/explore_apc.py $RUN/out/x1/x1_sweep.csv $RUN/verB_${FAM}_p1.csv $RUN/a ) | tee $RUN/explore_apc.log
grep -q "EXPLORE_RESULT REPRODUCED" $RUN/explore_apc.log || { echo "版本 B 沒有重現 → 停（偏離 10 第 4 點：這個變體不重算）"; exit 1; }
echo "== 版本 C $(date -Is)"
( cd $RUN && $ROOT/venv/bin/python -I $V8/analyze_v8.py --sweep $RUN/out/x1/x1_sweep.csv --out-dir $RUN/a --out-prefix verC ) | tee $RUN/analyze_verC.log
$ROOT/venv/bin/python -I $V8/collect_v8.py apc $VAR --base results/icpe_next/apc --sweep $RUN/out/x1/x1_sweep.csv \
  --files $RUN/a/verC_headroom_components.csv $RUN/a/verC_violations.csv $RUN/a/base_headroom_components.csv $RUN/a/explore_apc_rows.csv $RUN/a/explore_ogpu_belp_rows.csv \
  --runs $RUN --note "PREREG 偏離 10 的正式 APC 重算（版本 C）；重現檢查＝版本 B 的 16 個 GPU 規則，全部格 7 欄逐值相同"
echo "APC_FORMAL_DONE $VAR"
