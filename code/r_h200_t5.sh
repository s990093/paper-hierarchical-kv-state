#!/bin/bash
# =============================================================================
# r_h200_t5.sh — ICPE T5（docs/h200/icpe/T5_fi_small_cpu.md）：FreeInference 在 CPU 16–64 GiB 的 X1（對最好的對稱對手）。
#
#   bash code/r_h200_t5.sh            # head node（不用 Slurm、不用 GPU）：資料驗證 → sweep（workers 6）→ check → analyze（全部／使用者發起）→ report
#   ... --list | --dry-run ；--from <步驟>
#
# 環境同 code/r_h200_t3.sh --part head，但不設 X1_PLANNER（＝Cake）；FreeInference 讀 /work 的那份。不量時間。
# =============================================================================
set -uo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO=$(cd "$_here/.." && pwd)
export REPO
DRY=0; LIST=0; FROM=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --list) LIST=1; DRY=1; shift ;;
    --from) FROM="$2"; shift 2 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "不認得的參數：$1" >&2; exit 2 ;;
  esac
done
[ "$DRY" = 1 ] && export TIARA_SKIP_CONDA=1
# shellcheck disable=SC1091
source "$REPO/code/env_h200.sh"
cd "$REPO"
ulimit -c 0
T1=$REPO/results/icpe_h200/t1
T5=$REPO/results/icpe_h200/t5
unset X1_PLANNER
export TIARA_ALLOW_NO_SLURM=1
export P3_CONST=h200 P3_TIER=h200cpu M7_TIER_PARAMS=$T1/tier_params_h200cpu.json
export M7_RES=$REPO/results/m7_write_policy_h200 P3_MAP_PLATFORM=h200
export X1_TRACE_DIR=$TIARA_HOME/datasets/traces X5_TRACE_DIR=$TIARA_HOME/datasets/traces
export M10_SWEEP_DIR=$REPO/results/m10_sweep_h200 M10_PRECISION_DIR=$REPO/results/m10_precision_h200
export M10_BUSY_LEVELS=$REPO/results/m7_write_policy_h200/busy_levels.json
export TIARA_DATASETS=/work/ChihChungHsu/hungweilai/datasets
WORKERS=6
RAW=/work/ChihChungHsu/hungweilai/t5-raw
WEEKS="fi:2026-05-17_2026-05-23 fi:2026-05-24_2026-05-30 fi:2026-05-31_2026-06-06 fi:2026-06-07_2026-06-13 fi:2026-06-14_2026-06-20 fi:2026-06-21_2026-06-27"
FV="python code/icpe/p3_fvar.py --variant V0 --out-root $T5/p12"

if [ "$DRY" = 0 ]; then
  [ -f "$M7_TIER_PARAMS" ] || { echo "[r_h200_t5] 找不到 $M7_TIER_PARAMS" >&2; exit 2; }
  mkdir -p "$T5" "$RAW"
  echo "[r_h200_t5] $(date -Is) host=$(hostname -s) git=$(git rev-parse --short HEAD) X1_PLANNER=${X1_PLANNER:-<unset>} TIARA_DATASETS=$TIARA_DATASETS"
fi

STARTED=1; [ -n "$FROM" ] && STARTED=0
step() {
  local name=$1; shift
  if [ "$STARTED" = 0 ]; then if [ "$name" = "$FROM" ]; then STARTED=1; else return 0; fi; fi
  if [ "$LIST" = 1 ]; then echo "$name"; return 0; fi
  if [ "$DRY" = 1 ]; then printf '[%s] ' "$name"; printf '%q ' "$@"; echo; return 0; fi
  echo "=== $(date -Is) $name"
  "$@"; local rc=$?
  if [ $rc -ne 0 ]; then echo "=== $(date -Is) $name 失敗（exit $rc）→ 停下、回報（規則 2）。接著跑：--from $name" >&2; exit $rc; fi
}

step T5-VERIFY   m7run icpe-t5-data-verify python -I code/h200_p3_verify_data.py "$TIARA_DATASETS" freeinference
step T5-SWEEP    m7run icpe-t5-x1fi-sweep env RUN_DIR="$RAW" $FV x1 sweep --workers $WORKERS --traces $WEEKS --rules session_id --modes cake --cpu 16 32 64 128 --out x1_sweep_fi.csv
step T5-CHECK    m7run icpe-t5-check python code/icpe/t5.py check
step T5-AN       m7run icpe-t5-x1fi-an $FV x1 analyze --sweep x1_sweep_fi.csv --traces freeinference --out-prefix x1fi
step T5-AN-USER  m7run icpe-t5-x1fiuser-an $FV x1 analyze --sweep x1_sweep_fi.csv --traces freeinference --metric mean_ret_user --out-prefix x1fiuser
step T5-REPORT   m7run icpe-t5-report python code/icpe/t5.py report
echo "[r_h200_t5] $(date -Is) 完成"
