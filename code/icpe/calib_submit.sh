#!/bin/bash
# calib_submit.sh — ICPE T2（docs/h200/icpe/T2_parallel_calibration.md §2）：送出 15 個單卡 job，順序 S1→S2 → 波 1（3 個）→ S3→S4 → 波 2 → S5→S6 → 波 3，
# 用 afterok 串起來（任何一個失敗，後面不跑）。在 head node 跑；印出每個 job 的編號，也寫進 results/icpe_h200/calib/jobs.tsv。
#   bash code/icpe/calib_submit.sh [--dry-run]
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
export PATH=/cm/local/apps/slurm/current/bin:$PATH
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
W=3
OUT=$REPO/results/icpe_h200/calib/jobs.tsv
mkdir -p "$(dirname "$OUT")"
n=0
sub() {   # sub <job 名> <依賴（可空）> <參數...> → 印出 job 編號
  local name=$1 dep=$2; shift 2
  n=$((n + 1))
  if [ "$DRY" = 1 ]; then echo "dry$n"; echo "[dry] sbatch --job-name=$name ${dep:+--dependency=afterok:$dep} code/r_h200_calib.sh $*" >&2; return 0; fi
  local id
  id=$(env -u TMPDIR -u CLAUDE_CODE_TMPDIR sbatch --parsable --job-name="$name" ${dep:+--dependency=afterok:$dep} code/r_h200_calib.sh "$@")
  printf '%s\t%s\t%s\t%s\n' "$id" "$name" "${dep:-}" "$*" >> "$OUT"
  echo "$id"
}
[ "$DRY" = 1 ] || printf 'job\tname\tdepends_on\targs\n' > "$OUT"
prev=""
s=0
for wave in w1 w2 w3; do
  for _ in 1 2; do
    s=$((s + 1))
    prev=$(sub "icpe-cal-s$s-1" "$prev" --cond solo --wave "s$s" --n 1)
  done
  ids=()
  for i in $(seq 1 $W); do ids+=("$(sub "icpe-cal-$wave-$i" "$prev" --cond par --wave "$wave" --n $W)"); done
  prev=$(IFS=:; echo "${ids[*]}")
done
echo "[calib_submit] 送出 $n 個 job；最後一波：$prev"
