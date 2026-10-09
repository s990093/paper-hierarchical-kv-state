#!/bin/bash
# 11 第 1 輪補跑：LongAlpaca-7B（詞表越界修正後）。等 m7_run_round1.sh 跑完才開始
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m7_env.sh
cd /mlsteam/workspace/paper-hierarchical-kv-state/code
X=/mlsteam/workspace/paper-hierarchical-kv-state/results/m7_explore_mi300x
L=$(ls -t /mlsteam/data/tiara/runs/m7_round1_chain_*.log | head -1)
until grep -q "DONE" $L; do sleep 30; done
step() { echo "=== $(date -Is) $1"; shift; "$@"; echo "exit=$?"; }
LA="/mlsteam/data/tiara/hf-cache/hub/models--Yukang--LongAlpaca-7B/snapshots/*"
rm -f $X/calib_c1_longalpaca7b.csv     # 被 kill 的那次只寫了 0 列（連表頭都沒有）
M7_MODEL_GLOB="$LA" step ok-longalpaca m7run r1-ok-longalpaca7b-fix python m7_guard_run.py python test_m7_correctness.py $X/correct_longalpaca7b.json
M7_MODEL_GLOB="$LA" step c1-longalpaca m7run r1-c1-longalpaca7b-fix python m7_guard_run.py python m7_calib.py c1 --out $X/calib_c1_longalpaca7b.csv --max-chunks 72 --reps 3
echo "=== $(date -Is) DONE"
