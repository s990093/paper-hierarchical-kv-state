#!/usr/bin/env bash
# 重排（2026-10-08 15:25）：等 B 主設定（run 20261008-142647）結束；依優先順序跑敏感度
set -u
cd /mlsteam/workspace/paper-hierarchical-kv-state
source code/m7_env.sh
P=code/m7_write_policy.py
until [ -f /mlsteam/data/tiara/runs/20261008-142647-m7-b-nfs-gap0/exit_code ]; do sleep 20; done
step() { echo "=== $(date -Is) $*"; }
K="--strategies S1 S2b S4 S4+ S4+P S5 S5s"
S6="--strategies S1 S4 S4+ S4+P S5 S5s"
C="--cpu-fracs 0.25 0.5"
step B-rep36;   M7_IO_MODEL=share m7run m7-b-nfs-gap0-r36 python $P b --reps 3 --rep-start 3 --ssd-dev nfs --gap 0 --wl-seed 0 $C $K
step B-seed1;   M7_IO_MODEL=share m7run m7-b-nfs-seed1 python $P b --reps 3 --ssd-dev nfs --gap 0 --wl-seed 1 $C $S6
step B-cpuvllm; M7_IO_MODEL=share M7_CPU_GIBPS=3.69 m7run m7-b-nfs-cpuvllm python $P b --reps 3 --ssd-dev nfs --gap 0 --wl-seed 0 $C $S6
step B-local;   M7_IO_MODEL=share m7run m7-b-local-gap0 python $P b --reps 3 --ssd-dev local --gap 0 --wl-seed 0 $C $S6
step B-fifo;    M7_IO_MODEL=fifo  m7run m7-b-nfs-fifo python $P b --reps 3 --ssd-dev nfs --gap 0 --wl-seed 0 $C $S6
step C7-self-fixed; m7run m7-c7self-fixed python code/m7_calib.py c7 --out $R/calib_c7_self_fixed.csv --reps 3
step C4-t2-same;    m7run m7-c4-same-t2 python $P noise --reps 10 --slot t2 --proc-mode same
step C4-t2-restart; for i in $(seq 1 5); do m7run m7-c4-restart-t2 python $P noise --reps 1 --slot t2 --proc-mode restart; done
step CORE-DONE
step B-seed2;   M7_IO_MODEL=share m7run m7-b-nfs-seed2 python $P b --reps 3 --ssd-dev nfs --gap 0 --wl-seed 2 $C $S6
step B-gap2;    M7_IO_MODEL=share m7run m7-b-nfs-gap2 python $P b --reps 3 --ssd-dev nfs --gap 2 --wl-seed 0 $C $S6
step B-ssd50;   M7_IO_MODEL=share m7run m7-b-nfs-ssd50 python $P b --reps 3 --ssd-dev nfs --gap 0 --wl-seed 0 $C --ssd-frac 0.5 $S6
step DONE
