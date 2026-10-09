#!/usr/bin/env bash
# 第三段：等第二段 DONE；C7 自檢重做（限速器修正後）、B 第三個工作負載種子
set -u
cd /mlsteam/workspace/paper-hierarchical-kv-state
source code/m7_env.sh
P=code/m7_write_policy.py
until grep -q "=== .* DONE" "$1"; do sleep 30; done
step() { echo "=== $(date -Is) $*"; }
step C7-self-fixed; m7run m7-c7self-fixed python code/m7_calib.py c7 --out $R/calib_c7_self_fixed.csv --reps 3
step B-seed2;       M7_IO_MODEL=share m7run m7-b-nfs-seed2 python $P b --reps 3 --ssd-dev nfs --gap 0 --wl-seed 2 --cpu-fracs 0.25 0.5 --strategies S0 S1 S2b S3 S4 S4+ S4+P S5 S5s
step DONE
