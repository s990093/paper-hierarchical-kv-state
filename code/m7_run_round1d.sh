#!/bin/bash
# 11 追加 3：vLLM 引擎內部時間戳量 f(i)（探索性）。等 round1c 跑完才開始
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m7_env.sh
cd /mlsteam/workspace/paper-hierarchical-kv-state/code
X=/mlsteam/workspace/paper-hierarchical-kv-state/results/m7_explore_mi300x
HUB=/mlsteam/data/tiara/hf-cache/hub
L=$(ls -t /mlsteam/data/tiara/runs/m7_round1c_chain_*.log | head -1)
until grep -q "DONE" $L; do sleep 30; done
step() { echo "=== $(date -Is) $1"; shift; "$@"; echo "exit=$?"; }
step vllm2-llama m7run r1-vllmf2-llama31-8b python m7_guard_run.py python m7_vllm_f.py "$HUB/models--unsloth--Llama-3.1-8B-Instruct/snapshots/*" $X/vllmf2_llama31_8b.csv
step vllm2-longalpaca m7run r1-vllmf2-longalpaca7b python m7_guard_run.py python m7_vllm_f.py "$HUB/models--Yukang--LongAlpaca-7B/snapshots/*" $X/vllmf2_longalpaca7b.csv --max-chunks 62
step vllm2-qwen3moe m7run r1-vllmf2-qwen3-30b-a3b python m7_guard_run.py python m7_vllm_f.py "$HUB/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/*" $X/vllmf2_qwen3_30b_a3b.csv
echo "=== $(date -Is) DONE"
