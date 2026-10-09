#!/bin/bash
# 11 破解計劃第 1 輪（11_round1_plan.md）：H2 的 f(i) 校準。LongAlpaca-7B（MHA）、Qwen3-30B-A3B（MoE），Llama-3.1-8B 當同一天的參考
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m7_env.sh
cd /mlsteam/workspace/paper-hierarchical-kv-state/code
X=/mlsteam/workspace/paper-hierarchical-kv-state/results/m7_explore_mi300x
HUB=/mlsteam/data/tiara/hf-cache/hub
step() { echo "=== $(date -Is) $1"; shift; "$@"; echo "exit=$?"; }
LA="$HUB/models--Yukang--LongAlpaca-7B/snapshots/*"
QM="$HUB/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/*"
# 1. 正確性（自管前向 vs HF 整段 prefill，8K）
M7_MODEL_GLOB="$LA" step ok-longalpaca m7run r1-ok-longalpaca7b python m7_guard_run.py python test_m7_correctness.py $X/correct_longalpaca7b.json
M7_MODEL_GLOB="$QM" step ok-qwen3moe m7run r1-ok-qwen3-30b-a3b python m7_guard_run.py python test_m7_correctness.py $X/correct_qwen3_30b_a3b.json
# 2. C1：f(i)，512 token 一個 chunk，72 個 chunk（36,864 token；doc 負載最多用到 67 個），3 次
step c1-llama m7run r1-c1-llama31-8b python m7_guard_run.py python m7_calib.py c1 --out $X/calib_c1_llama31_8b.csv --max-chunks 72 --reps 3
M7_MODEL_GLOB="$LA" step c1-longalpaca m7run r1-c1-longalpaca7b python m7_guard_run.py python m7_calib.py c1 --out $X/calib_c1_longalpaca7b.csv --max-chunks 72 --reps 3
M7_MODEL_GLOB="$QM" M7_EXPERTS_IMPL=grouped_mm step c1-qwen3moe-grouped m7run r1-c1-qwen3-30b-a3b-grouped python m7_guard_run.py python m7_calib.py c1 --out $X/calib_c1_qwen3_30b_a3b_grouped.csv --max-chunks 72 --reps 3
M7_MODEL_GLOB="$QM" M7_EXPERTS_IMPL=eager step c1-qwen3moe-eager m7run r1-c1-qwen3-30b-a3b-eager python m7_guard_run.py python m7_calib.py c1 --out $X/calib_c1_qwen3_30b_a3b_eager.csv --max-chunks 72 --reps 3
echo "=== $(date -Is) DONE"
