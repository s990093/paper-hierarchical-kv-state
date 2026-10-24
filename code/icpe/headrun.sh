#!/bin/bash
# headrun.sh — head node 上只用 CPU、幾秒到幾分鐘的步驟用的 §4.1 記錄殼（和 tiara.sbatch 同樣的 run 目錄格式；不在 Slurm job 裡）。
# 用法：bash code/icpe/headrun.sh <短名> <指令...>   → $TIARA_RUNS/<時間>-<短名>/{cmd.sh,context.txt,stdout.log,stderr.log,exit_code}
# 重的（多核、幾十分鐘以上）一律包在 Slurm job 裡（r_h200_icpe.sh --part cpu）。
set -uo pipefail
short=$1; shift
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TIARA_RUNS=${TIARA_RUNS:-$HOME/tiara-runs}
RUN_ID="$(date +%Y%m%d-%H%M%S)-$short"; RUN=$TIARA_RUNS/$RUN_ID
mkdir -p "$RUN"
{ printf '%q ' "$@"; echo; } > "$RUN/cmd.sh"
{ date -Is; echo "host: $(hostname)（head node，CPU）"; git -C "$REPO" rev-parse HEAD; echo "dirty: $(git -C "$REPO" status --porcelain | wc -l)"
  git -C "$REPO" status --porcelain | head -20; env | grep -E '^(P3_|M7_|X1_|X5_|TIARA_)' | sort; } > "$RUN/context.txt" 2>&1
cd "$REPO"
RUN_ID=$RUN_ID RUN_DIR=$RUN nice -n 19 bash "$RUN/cmd.sh" > "$RUN/stdout.log" 2> "$RUN/stderr.log"; rc=$?
echo "$rc" > "$RUN/exit_code"
echo "[headrun] $RUN_ID exit=$rc"; [ $rc -ne 0 ] && tail -20 "$RUN/stderr.log" >&2
exit $rc
