#!/usr/bin/env bash
# 第一階段第二段：等主鏈 DONE 之後才開始（同一張卡不並行）
set -u
cd /mlsteam/workspace/paper-hierarchical-kv-state
source code/m7_env.sh
P=code/m7_write_policy.py
MAIN=$1
until grep -q "=== .* DONE" "$MAIN"; do sleep 30; done
step() { echo "=== $(date -Is) $*"; }
step C6-vllm;      m7run m7-c6-vllm python code/m7_c6_vllm.py --out $R/c6_vllm.csv --reps 6
step B-rep36;      M7_IO_MODEL=share m7run m7-b-nfs-gap0-r36 python $P b --reps 3 --rep-start 3 --ssd-dev nfs --gap 0 --wl-seed 0 --strategies S0 S1 S2b S3 S4 S4+ S4+P S5 S5s
step B-local;      M7_IO_MODEL=share m7run m7-b-local-gap0 python $P b --reps 3 --ssd-dev local --gap 0 --wl-seed 0 --cpu-fracs 0.25 0.5 --strategies S0 S1 S2b S3 S4 S4+ S4+P S5 S5s
step B-fifo;       M7_IO_MODEL=fifo  m7run m7-b-nfs-fifo python $P b --reps 3 --ssd-dev nfs --gap 0 --wl-seed 0 --cpu-fracs 0.25 0.5 --strategies S0 S1 S2b S3 S4 S4+ S4+P S5 S5s
step B-gap2;       M7_IO_MODEL=share m7run m7-b-nfs-gap2 python $P b --reps 3 --ssd-dev nfs --gap 2 --wl-seed 0 --cpu-fracs 0.25 0.5 --strategies S0 S1 S2b S3 S4 S4+ S4+P S5 S5s
step B-seed1;      M7_IO_MODEL=share m7run m7-b-nfs-seed1 python $P b --reps 3 --ssd-dev nfs --gap 0 --wl-seed 1 --cpu-fracs 0.25 0.5 --strategies S0 S1 S2b S3 S4 S4+ S4+P S5 S5s
step B-ssd50;      M7_IO_MODEL=share m7run m7-b-nfs-ssd50 python $P b --reps 3 --ssd-dev nfs --gap 0 --wl-seed 0 --cpu-fracs 0.25 0.5 --ssd-frac 0.5 --strategies S0 S1 S2b S3 S4 S4+ S4+P S5 S5s
step C4-t2-same;   m7run m7-c4-same-t2 python $P noise --reps 10 --slot t2 --proc-mode same
step C4-t2-restart; for i in $(seq 1 5); do m7run m7-c4-restart-t2 python $P noise --reps 1 --slot t2 --proc-mode restart; done
step DONE
