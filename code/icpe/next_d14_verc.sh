#!/usr/bin/env bash
# next_d14_verc.sh — icpe-data：PREREG 偏離 14 第二段＝修正版版本 C（STATUS_DATA 2026-11-07 16:34，使用者在 icpe-main 的決定：整個變體重跑）。
# 步驟照 icpe-rux 的 run_apc_formal.sh（偏離 10），差別：引擎＝v8_dev14（icpe-next-rux 36780534）＋ RUX_PFX=1 RUX_PGR=1；
# 重現檢查的參考表＝第一段的版本 B 檢查表（next_d14_verb.sh 的輸出，O-PF／O-ALL 換上 d14 列；其他元件已核對和原本的版本 B 表相同）；
# 另外核對 opf 以外規則的列和原本的版本 C sweep 逐位元相同；收到 results/icpe_next/apc_d14/<變體>/（apc/ 不動）。
# 用法（在 §4.1 的 run 目錄裡，RUN 已設）：
#   D14REPO=<icpe-next-rux 36780534 的 worktree> DATAREPO=<icpe-next-data> ROOT=/ssd6/hungwei/icpe-next \
#   bash next_d14_verc.sh <--traces 的值> <--rules 的值（多個整個加引號）> <變體目錄名> <第一段的版本 B 檢查表> \
#        <該檢查表的 table_vs_verB.log> <原本的 apc/<變體>/x1_sweep.csv.xz> [workers]
# 例：bash next_d14_verc.sh nb:miniswe_N16_s1 nb miniswe_N16_s1 <run>/out/ms_p1_headroom_components.csv <run>/table_vs_verB.log \
#        $DATAREPO/results/icpe_next/apc/miniswe_N16_s1/x1_sweep.csv.xz 35
# X1_NB_DIR／X1_TRACE_DIR／X1_FI_DIR 照紀錄（預設＝acvlab 的位置）。任何一步失敗就停（exit ≠ 0），那個變體不用。
set -euo pipefail
TR=$1; RULES=$2; VAR=$3; VERB14=$4; VERBLOG=$5; OLDSW=$6; W=${7:-35}
: "${RUN:?要在 run 目錄裡跑}" "${D14REPO:?}" "${DATAREPO:?}" "${ROOT:?}"
grep -q "D14_TABLE_RESULT IDENTICAL" $VERBLOG || { echo "第一段的檢查表沒有通過「其他元件不變」→ 停"; exit 1; }
export REPO=$D14REPO
D14=$REPO/code/icpe/rux_x1fast/v8_dev14
V8=$REPO/code/icpe/rux_x1fast/v8_features
[ "$(sha256sum $D14/src/main.rs | cut -c1-16)" = d2048fa38a42c78b ] || { echo "v8_dev14 main.rs 不是 d2048fa3 → 停"; exit 1; }
source $REPO/code/icpe/rux_x1fast/v6_slowcap/env_sim.sh
export X1_NB_DIR=${X1_NB_DIR:-$ROOT/datasets/x1_nb}
export X1_TRACE_DIR=${X1_TRACE_DIR:-/ssd8/hungwei/paper-hkv/datasets/mooncake/traces}
export X1_FI_DIR=${X1_FI_DIR:-/ssd8/hungwei/paper-hkv/datasets/freeinference_agentic_trace/release}
cd $REPO
P17="LRU LRU2 ARC S3FIFO GDSF PENS FREQ2Q LRU+NONE LRU+2HW LRU+2HA LRU+RPF O-BEL O-BELS O-DROP O-NOW O-PF O-ALL"
G18="LRU BEL TTLA:0.3 TTLA:1 TTLA:3 TTLA:10 TTLA:30 TTLA:100 TTLA:300 TTLB:0.3 TTLB:1 TTLB:3 TTLB:10 TTLB:30 TTLB:100 TTLB:300 APC BELP"
mkdir -p $RUN/a
echo "== sweep $(date -Is) load=$(cut -d' ' -f1-3 /proc/loadavg)"
RUN_DIR="" RUX_PFX=1 RUX_PGR=1 X1_RUST_TIMING_LOG=$RUN/rust_timing.jsonl nice -n 10 $ROOT/venv/bin/python -I $D14/sweep_v8_dev14.py --variant V1 --calib $CALIB \
  --out-root $RUN/out x1 sweep --traces $TR --rules $RULES --cpu 8 16 32 64 128 256 512 1024 111.0625 --policies $P17 \
  --slows ssd b32 nfs --modes cake --gpu-gib 111.0625 --gpu-write wb --gpu-policies $G18 --pf-target cpu --workers $W --engine rust \
  > $RUN/sweep.log 2> $RUN/sweep.err
grep -E '^(sweep_v8_dev14|engine=|jobs=|sweep done)' $RUN/sweep.log | cut -c1-250
grep -q "py_fallback_jobs=0" $RUN/sweep.log || { echo "有 job 退回 Python → 停"; exit 1; }
grep -q "src_sha256=4eca5bda" $RUN/sweep.log || { echo "引擎不是 v8_dev14（src 4eca5bda）→ 停"; exit 1; }
echo "== opf 以外的列和原本的版本 C sweep 比 $(date -Is)"
$ROOT/venv/bin/python -I $DATAREPO/code/icpe/next_d14_cmp.py sweep $RUN/out/x1/x1_sweep.csv $OLDSW --skip-policies O-PF O-ALL \
  --out $RUN/sweep_vs_old.csv | tee $RUN/sweep_vs_old.log
grep -q "D14_SWEEP_RESULT IDENTICAL" $RUN/sweep_vs_old.log || { echo "沿用的列和原本的版本 C 不同 → 停"; exit 1; }
echo "== 重現檢查（16 個版本 B GPU 規則 vs 第一段的版本 B 檢查表）$(date -Is)"
( cd $RUN && $ROOT/venv/bin/python -I $V8/explore_apc.py $RUN/out/x1/x1_sweep.csv $VERB14 $RUN/a ) | tee $RUN/explore_apc.log
grep -q "EXPLORE_RESULT REPRODUCED" $RUN/explore_apc.log || { echo "第一段的版本 B 沒有重現 → 停（這個變體不用）"; exit 1; }
echo "== 修正版版本 C $(date -Is)"
( cd $RUN && $ROOT/venv/bin/python -I $V8/analyze_v8.py --sweep $RUN/out/x1/x1_sweep.csv --out-dir $RUN/a --out-prefix verC ) | tee $RUN/analyze_verC.log
$ROOT/venv/bin/python -I $V8/collect_v8.py apc $VAR --base $DATAREPO/results/icpe_next/apc_d14 --sweep $RUN/out/x1/x1_sweep.csv \
  --files $RUN/a/verC_headroom_components.csv $RUN/a/verC_violations.csv $RUN/a/base_headroom_components.csv $RUN/a/explore_apc_rows.csv \
    $RUN/a/explore_ogpu_belp_rows.csv $RUN/sweep_vs_old.csv \
  --runs $RUN --note "PREREG 偏離 14 第二段（修正版版本 C）：v8_dev14 d2048fa3（icpe-next-rux 36780534），RUX_PFX=1 RUX_PGR=1；重現檢查＝第一段的版本 B 檢查表（$VERB14），全部格 7 欄逐值相同；opf 以外的列和原本的 apc sweep 逐位元相同"
echo "D14_VERC_DONE $VAR"
