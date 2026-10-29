#!/bin/bash
#SBATCH --job-name=icpe-t3-fi
#SBATCH --partition=h200q
#SBATCH --account=ChihChungHsu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=6:00:00
#SBATCH -o slurm-%x-%j.out
# =============================================================================
# r_h200_t3.sh — ICPE T3（docs/h200/icpe/T3_planner_sensitivity.md）：還原規劃器換成 STAIR，重跑 T1 的 X1（B1／B2／B7、B3）與 fullsim（C2）。
#
#   bash code/r_h200_t3.sh --part head          # head node（不用 Slurm）：V0／V1／V2 的 X1 sweep＋analyze、fullsim、ratio；workers 8
#   sbatch code/r_h200_t3.sh --part fi          # compute node（要本地 /tmp 的 p3_unified 資料）：X1 FreeInference sweep＋analyze；workers 14
#   ... --list | --dry-run ；--from <步驟>
#
# 指令和 code/r_h200_t1.sh 相同，只多 X1_PLANNER=stair，輸出改到 results/icpe_h200/t3/（登記 §2）。不量時間。
# =============================================================================
set -uo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
if [ -n "${REPO:-}" ] && [ -f "$REPO/code/env_h200.sh" ]; then :
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/code/env_h200.sh" ]; then REPO=$SLURM_SUBMIT_DIR
elif [ -f "$_here/env_h200.sh" ]; then REPO=$(cd "$_here/.." && pwd)
else echo "[r_h200_t3] 找不到 repo（code/env_h200.sh）" >&2; exit 2
fi
export REPO
PART=""; DRY=0; LIST=0; FROM=""
while [ $# -gt 0 ]; do
  case "$1" in
    --part) PART="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --list) LIST=1; DRY=1; shift ;;
    --from) FROM="$2"; shift 2 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "不認得的參數：$1" >&2; exit 2 ;;
  esac
done
case "$PART" in head|fi) ;; *) echo "要 --part head|fi" >&2; exit 2 ;; esac
[ "$DRY" = 1 ] && export TIARA_SKIP_CONDA=1
export ICPE_MODEL_ROOT="${ICPE_MODEL_ROOT:-/tmp/hungweilai-models/hf-cache}"
export HF_HOME="$ICPE_MODEL_ROOT"
# shellcheck disable=SC1091
source "$REPO/code/env_h200.sh"
cd "$REPO"
ulimit -c 0
T1=$REPO/results/icpe_h200/t1
T3=$REPO/results/icpe_h200/t3
export P3_CONST=h200 P3_TIER=h200cpu M7_TIER_PARAMS=$T1/tier_params_h200cpu.json X1_PLANNER=stair
export M7_RES=$REPO/results/m7_write_policy_h200 P3_MAP_PLATFORM=h200
P01=$REPO/results/icpe_h200/p01
export X1_TRACE_DIR=$TIARA_HOME/datasets/traces X5_TRACE_DIR=$TIARA_HOME/datasets/traces
export M10_SWEEP_DIR=$REPO/results/m10_sweep_h200 M10_PRECISION_DIR=$REPO/results/m10_precision_h200
export M10_BUSY_LEVELS=$REPO/results/m7_write_policy_h200/busy_levels.json
if [ "$PART" = head ]; then
  export TIARA_ALLOW_NO_SLURM=1
  RAWROOT=/work/ChihChungHsu/hungweilai/t3-raw          # head node 的 /tmp 會滿（memory h200-headnode-tmp-full）
  WORKERS=8
else
  export TIARA_P3_VENV="${TIARA_P3_VENV:-$TIARA_HOME/venvs/tiara-p3}" P3_LOCAL="${P3_LOCAL:-/tmp/hungweilai-p3}"
  P3DATA="$P3_LOCAL/datasets"
  export P3SIM_CACHE="$P3_LOCAL/cache/p3_sim" P3_DATA_CACHE="$P3_LOCAL/cache/p3_data" TIARA_HF_HUB="$HF_HOME/hub"
  export P3PY="$TIARA_P3_VENV/bin/python"
  RAWROOT=$P3_LOCAL/icpe-t3
  WORKERS=14
fi

if [ "$DRY" = 0 ]; then
  [ -f "$M7_TIER_PARAMS" ] || { echo "[r_h200_t3] 找不到 $M7_TIER_PARAMS" >&2; exit 2; }
  if [ "$PART" = fi ]; then
    [ -n "${SLURM_JOB_ID:-}" ] || { echo "[r_h200_t3] --part fi 要在 Slurm job 裡（本地 /tmp 的資料）" >&2; exit 2; }
    [ -x "$P3PY" ] || { echo "[r_h200_t3] 找不到 $P3PY" >&2; exit 2; }
  fi
  for f in calib_c1.csv calib_c5.csv; do [ -f "$P01/calib/$f" ] || { echo "[r_h200_t3] 缺 $P01/calib/$f（V1 要用）" >&2; exit 2; }; done
  python3 - "$TIARA_HOME/datasets/traces" <<'PY' || { echo "[r_h200_t3] Mooncake trace 的 sha256 不對 → 停" >&2; exit 2; }
import hashlib, sys
want = {"conversation_trace.jsonl": "b8cbb061a85206d7", "toolagent_trace.jsonl": "48a2db1a13d3bc05"}
for f, h in want.items():
    got = hashlib.sha256(open(f"{sys.argv[1]}/{f}", "rb").read()).hexdigest()
    print("[r_h200_t3]", f, got[:16]); assert got.startswith(h), (f, got)
PY
  mkdir -p "$T3" "$RAWROOT"
  echo "[r_h200_t3] $(date -Is) part=$PART job=${SLURM_JOB_ID:-head} host=$(hostname -s) git=$(git rev-parse --short HEAD) X1_PLANNER=$X1_PLANNER"
fi

STARTED=1; [ -n "$FROM" ] && STARTED=0
step() {
  local name=$1; shift
  if [ "$STARTED" = 0 ]; then if [ "$name" = "$FROM" ]; then STARTED=1; else return 0; fi; fi
  if [ "$LIST" = 1 ]; then echo "$name"; return 0; fi
  if [ "$DRY" = 1 ]; then printf '[%s] ' "$name"; printf '%q ' "$@"; echo; return 0; fi
  echo "=== $(date -Is) $name"
  "$@"; local rc=$?
  if [ $rc -ne 0 ]; then echo "=== $(date -Is) $name 失敗（exit $rc）→ 停下、回報（規則 2）。接著跑：--part $PART --from $name" >&2; exit $rc; fi
}
fresh() {
  local d
  for d in "$@"; do [ -e "$d" ] && { echo "[r_h200_t3] $d 已存在 → 改名 $d.stale-$(date +%Y%m%d-%H%M%S)"; mv "$d" "$d.stale-$(date +%Y%m%d-%H%M%S)" || return 1; }; done
  return 0
}
x1_sweep() {      # 同 r_h200_t1.sh
  local nm=$1 V=$2 root=$3 raw=$4; shift 4
  local ev=(); while [ $# -gt 0 ] && [ "$1" != -- ]; do ev+=("$1"); shift; done; shift
  fresh "$raw" "$root/x1/x1_sweep.csv" "$root/x1/x1_sweep_fi.csv" && mkdir -p "$raw" || return 1
  m7run "$nm" env RUN_DIR="$raw" ${ev[@]+"${ev[@]}"} python code/icpe/p3_fvar.py --variant "$V" --calib "$P01/calib" --out-root "$root" x1 sweep "$@"
}

if [ "$PART" = head ]; then
  step T3-SELFTEST m7run icpe-t3-selftest env -u X1_PLANNER python code/icpe/t3.py selftest
  for V in V0 V1 V2; do
    FV="python code/icpe/p3_fvar.py --variant $V --calib $P01/calib --out-root $T3/p01/sim_$V"
    step "T3-X1-$V-SWEEP" x1_sweep "icpe-t3-x1-sweep-$V" "$V" "$T3/p01/sim_$V" "$RAWROOT/x1sweep_$V" -- --workers $WORKERS
    step "T3-X1-$V-AN"    m7run "icpe-t3-x1-an-$V" $FV x1 analyze --sweep x1_sweep.csv --traces conversation toolagent
    step "T3-X1-$V-H1800" m7run "icpe-t3-x1-h1800-$V" $FV x1 analyze --sweep x1_sweep.csv --traces conversation toolagent --dead-h 1800 --out-prefix x1h1800
    step "T3-X1-$V-H3600" m7run "icpe-t3-x1-h3600-$V" $FV x1 analyze --sweep x1_sweep.csv --traces conversation toolagent --dead-h 3600 --out-prefix x1h3600
  done
  step T3-FULLSIM m7run icpe-t3-p02-fullsim python code/icpe/x1_gpu.py fullsim --out "$T3/p02"
  step T3-RATIO   m7run icpe-t3-ratio python code/icpe/t3.py ratio
else
  WEEKS="fi:2026-05-17_2026-05-23 fi:2026-05-24_2026-05-30 fi:2026-05-31_2026-06-06 fi:2026-06-07_2026-06-13 fi:2026-06-14_2026-06-20 fi:2026-06-21_2026-06-27"
  FV12="python code/icpe/p3_fvar.py --variant V0 --out-root $T3/p12"
  step T3-P12-VERIFY m7run icpe-t3-p12-data-verify "$P3PY" -I code/h200_p3_verify_data.py "$P3DATA" mooncake freeinference
  step T3-P12-X1FI-SWEEP  x1_sweep icpe-t3-p12-x1fi-sweep V0 "$T3/p12" "$RAWROOT/p12_x1fi" TIARA_DATASETS="$P3DATA" X4_TRACE_DIR="$TIARA_HOME/datasets/traces" -- --workers $WORKERS --traces $WEEKS --rules session_id --modes cake --out x1_sweep_fi.csv
  step T3-P12-X1FI-AN     m7run icpe-t3-p12-x1fi-an env TIARA_DATASETS="$P3DATA" $FV12 x1 analyze --sweep x1_sweep_fi.csv --traces freeinference --out-prefix x1fi
  step T3-P12-X1FI-AN-USER m7run icpe-t3-p12-x1fiuser-an env TIARA_DATASETS="$P3DATA" $FV12 x1 analyze --sweep x1_sweep_fi.csv --traces freeinference --metric mean_ret_user --out-prefix x1fiuser
fi
echo "[r_h200_t3] $(date -Is) part=$PART 完成"
