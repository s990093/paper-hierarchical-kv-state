#!/usr/bin/env bash
# 第五段：等 D 段 DONE；CPU 層改用 vLLM OffloadingConnector 實測速度（C6：4.1 GiB/s × 0.9）
set -u
cd /mlsteam/workspace/paper-hierarchical-kv-state
source code/m7_env.sh
P=code/m7_write_policy.py
until grep -q "=== .* DONE" "$1"; do sleep 30; done
step() { echo "=== $(date -Is) $*"; }
S="--strategies S0 S1 S2b S3 S4 S4+ S4+P S5 S5s"
step B-cpuvllm; M7_IO_MODEL=share M7_CPU_GIBPS=3.69 m7run m7-b-nfs-cpuvllm python $P b --reps 3 --ssd-dev nfs --gap 0 --wl-seed 0 --cpu-fracs 0.25 0.5 $S
step DONE
