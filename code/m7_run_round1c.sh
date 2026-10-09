#!/bin/bash
# 11 第 1 輪追加：vLLM 量 f(i)（Llama-3.1-8B 驗證方法、Qwen3-30B-A3B）。等 round1b 跑完才開始
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m7_env.sh
cd /mlsteam/workspace/paper-hierarchical-kv-state/code
X=/mlsteam/workspace/paper-hierarchical-kv-state/results/m7_explore_mi300x
HUB=/mlsteam/data/tiara/hf-cache/hub
L=$(ls -t /mlsteam/data/tiara/runs/m7_round1b_chain_*.log | head -1)
until grep -q "DONE" $L; do sleep 30; done
step() { echo "=== $(date -Is) $1"; shift; "$@"; echo "exit=$?"; }
step vllm-llama m7run r1-vllmf-llama31-8b python m7_guard_run.py python m7_vllm_f.py "$HUB/models--unsloth--Llama-3.1-8B-Instruct/snapshots/*" $X/vllmf_llama31_8b.csv
step vllm-qwen3moe m7run r1-vllmf-qwen3-30b-a3b python m7_guard_run.py python m7_vllm_f.py "$HUB/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/*" $X/vllmf_qwen3_30b_a3b.csv
echo "=== $(date -Is) DONE"
