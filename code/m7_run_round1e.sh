#!/bin/bash
# 11 追加 4：Llama 存活設定的 GPU 確認（CPU 11.6、free、doc、本地 SSD、50%；S5L vs 所有對照組；seed 0–4）
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m7_env.sh
cd /mlsteam/workspace/paper-hierarchical-kv-state/code
export M7_IO_MODEL=share M7_CPU_GIBPS=11.6
export M7_B2_OUT=/mlsteam/workspace/paper-hierarchical-kv-state/results/m7_explore_mi300x/r1_gpu_llama_doc_free_local.csv
S="S1 S2b S4 S4+ S4+P S4L S4B S4W S4C S5L"
step() { echo "=== $(date -Is) $1"; shift; "$@"; echo "exit=$?"; }
for seed in 0 1 2 3 4; do
  step seed$seed m7run r1-gpu-llama-doc-free-local-s$seed python m7_write_policy.py b --b2 --workload doc --release free --ssd-dev local --wl-seed $seed --strategies $S --cpu-fracs 0.5 --reps 3 --verify
done
echo "=== $(date -Is) DONE"
