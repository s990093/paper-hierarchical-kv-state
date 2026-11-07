#!/usr/bin/env bash
# next_chat_verc.sh — icpe-data：對話資料集的版本 C（PREREG 偏離 13；STATUS_DATA 2026-11-07 11:05 補寫第 3 點）。
# 步驟照 icpe-rux 的 run_apc_formal.sh（偏離 10），只差兩點：紀錄是 nb:<對話資料集變體>；重現檢查的參考表是本機算的版本 B 表
# （chat_main＋chat_p1 _ttl 經 next_sim_analyze），不是 icpe-next-main 的 verB。
# 用法（在 §4.1 的 run 目錄裡，RUN 已設）：
#   V8REPO=<icpe-next-rux 4a713386 的 worktree> DATAREPO=<icpe-next-data> ROOT=/ssd6/hungwei/icpe-next \
#   bash next_chat_verc.sh <nb 名稱，例 sharegpt_T60_s1> <版本 B 表 *_headroom_components.csv(.xz)> [workers]
# 任何一步失敗就停（exit ≠ 0）。
set -euo pipefail
NB=$1; VERB=$2; W=${3:-35}
: "${RUN:?要在 run 目錄裡跑}" "${V8REPO:?}" "${DATAREPO:?}" "${ROOT:?}"
export REPO=$V8REPO
V8=$REPO/code/icpe/rux_x1fast/v8_features
[ "$(sha256sum $V8/src/main.rs | cut -c1-16)" = 48e4efcee98cb5e5 ] || { echo "v8 main.rs 不是 48e4efce → 停"; exit 1; }
source $REPO/code/icpe/rux_x1fast/v6_slowcap/env_sim.sh
export X1_NB_DIR=${X1_NB_DIR:-$ROOT/datasets/x1_nb}
cd $REPO
P17="LRU LRU2 ARC S3FIFO GDSF PENS FREQ2Q LRU+NONE LRU+2HW LRU+2HA LRU+RPF O-BEL O-BELS O-DROP O-NOW O-PF O-ALL"
G18="LRU BEL TTLA:0.3 TTLA:1 TTLA:3 TTLA:10 TTLA:30 TTLA:100 TTLA:300 TTLB:0.3 TTLB:1 TTLB:3 TTLB:10 TTLB:30 TTLB:100 TTLB:300 APC BELP"
mkdir -p $RUN/a
echo "== sweep $(date -Is) load=$(cut -d' ' -f1-3 /proc/loadavg)"
RUN_DIR="" X1_RUST_TIMING_LOG=$RUN/rust_timing.jsonl nice -n 10 $ROOT/venv/bin/python -I $V8/sweep_v8.py --variant V1 --calib $CALIB \
  --out-root $RUN/out x1 sweep --traces nb:$NB --rules nb --cpu 8 16 32 64 128 256 512 1024 111.0625 --policies $P17 \
  --slows ssd b32 nfs --modes cake --gpu-gib 111.0625 --gpu-write wb --gpu-policies $G18 --pf-target cpu --workers $W --engine rust \
  > $RUN/sweep.log 2> $RUN/sweep.err
grep -E '^(engine=|jobs=|sweep done)' $RUN/sweep.log | cut -c1-200
grep -q "py_fallback_jobs=0" $RUN/sweep.log || { echo "有 job 退回 Python → 停"; exit 1; }
grep -q "src_sha256=e7f6ee8f" $RUN/sweep.log || { echo "引擎不是 v8（Rust src e7f6ee8f）→ 停"; exit 1; }
echo "== 重現檢查（版本 B 的 16 個 GPU 規則）$(date -Is)"
( cd $RUN && $ROOT/venv/bin/python -I $V8/explore_apc.py $RUN/out/x1/x1_sweep.csv $VERB $RUN/a ) | tee $RUN/explore_apc.log
grep -q "EXPLORE_RESULT REPRODUCED" $RUN/explore_apc.log || { echo "版本 B 沒有重現 → 停（這個變體不報版本 C）"; exit 1; }
echo "== 版本 C $(date -Is)"
( cd $RUN && $ROOT/venv/bin/python -I $V8/analyze_v8.py --sweep $RUN/out/x1/x1_sweep.csv --out-dir $RUN/a --out-prefix verC ) | tee $RUN/analyze_verC.log
$ROOT/venv/bin/python -I $V8/collect_v8.py apc chat_$NB --base $DATAREPO/results/icpe_next/apc --sweep $RUN/out/x1/x1_sweep.csv \
  --files $RUN/a/verC_headroom_components.csv $RUN/a/verC_violations.csv $RUN/a/base_headroom_components.csv $RUN/a/explore_apc_rows.csv $RUN/a/explore_ogpu_belp_rows.csv \
  --runs $RUN --note "PREREG 偏離 13（對話資料集）的版本 C：v8 48e4efce（icpe-next-rux 4a713386）；重現檢查＝本機版本 B 表（chat_main＋chat_p1 _ttl），全部格 7 欄逐值相同"
echo "CHAT_VERC_DONE $NB"
