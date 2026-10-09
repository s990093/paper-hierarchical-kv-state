#!/usr/bin/env bash
# 第一階段主鏈（依序；同一張卡上不並行，避免互相干擾）
set -u
cd /mlsteam/workspace/paper-hierarchical-kv-state
source code/m7_env.sh
P=code/m7_write_policy.py
step() { echo "=== $(date -Is) $*"; }
step C4-same;    m7run m7-c4-same-t1 python $P noise --reps 10 --slot t1 --proc-mode same
step C4-restart; for i in $(seq 1 10); do m7run m7-c4-restart-t1 python $P noise --reps 1 --slot t1 --proc-mode restart; done
step C7-self;    m7run m7-c7self python code/m7_calib.py c7 --out $R/calib_c7_self.csv --reps 3
step C7-anchor;  m7run m7-c7anchor python $P anchor --L 8192 16384 32768 --reps 3
step A0;         m7run m7-a0 python $P a0 --reps 6
step A1;         M7_IO_MODEL=share m7run m7-a1 python $P a1 --reps 6
step B-main;     M7_IO_MODEL=share m7run m7-b-nfs-gap0 python $P b --reps 3 --ssd-dev nfs --gap 0 --wl-seed 0 --verify
step DONE
