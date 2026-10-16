#!/bin/bash
#SBATCH --job-name=m10h-lc
#SBATCH --partition=h200q
#SBATCH --account=ChihChungHsu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=16:00:00
#SBATCH -o slurm-%x-%j.out
# =============================================================================
# r_h200_lc.sh — 第二階段長 context（32K／64K／128K）在 H200 上復現（MI300X：10_longctx_plan.md、queue/44a、44b、44c）
# 事先登記：docs/precision_20261013/13_h200_repro.md §8（10/17 補：全部復現、128K 先跑）
#
#   sbatch --job-name=m10h-lc-speed  --time=16:00:00 --dependency=afterok:<m10h-speed> code/r_h200_lc.sh --part speed
#   sbatch --job-name=m10h-lc-screen --time=48:00:00 --dependency=afterok:<lc-speed>  code/r_h200_lc.sh --part screen
#   bash code/r_h200_lc.sh --part speed|screen --dry-run | --list ;  --from <步驟> 接著跑
#
# --part speed（1 張卡，量時間；不和我們自己的其他 GPU job 同時跑）：
#   大掃描的零件（m10_sweep parts：記憶體、載入、反量化；GPU 忙用 H200 的 busy2）＋ 長 context 速度格子
#   （128K 先，再 64K；FP8 主線粒度、INT4 KIVI；主線 8 配置；主線的 bw.json；2 launch × 5 rep；同 MI300X 44a）
# --part screen（1 張卡，只量分數）：試跑（每個模型 × 長度 1 題）→ 照 10 §3.3 的規則決定題數 → 正式（252 → 128 → 64 個 block）
# 停止：任何一步失敗、速度格子或零件的 gpu_guard 不是 CLEAN（CLAUDE.md §3）、試跑缺讀法 B 的輸入。
# =============================================================================
set -uo pipefail

PART=""; DRY=0; LIST=0; FROM=""
while [ $# -gt 0 ]; do
  case "$1" in
    --part) PART="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --list) LIST=1; DRY=1; shift ;;
    --from) FROM="$2"; shift 2 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "不認得的參數：$1（--help 看用法）" >&2; exit 2 ;;
  esac
done
case "$PART" in speed|screen) ;; *) echo "要 --part speed|screen" >&2; exit 2 ;; esac

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
if [ -n "${REPO:-}" ] && [ -f "$REPO/code/env_h200.sh" ]; then :
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/code/env_h200.sh" ]; then REPO=$SLURM_SUBMIT_DIR
elif [ -f "$_here/env_h200.sh" ]; then REPO=$(cd "$_here/.." && pwd)
elif [ -f "$HOME/paper-hierarchical-kv-state/code/env_h200.sh" ]; then REPO=$HOME/paper-hierarchical-kv-state
else echo "[r_h200_lc] 找不到 repo（code/env_h200.sh）" >&2; exit 2
fi
export REPO
[ "$DRY" = 1 ] && export TIARA_SKIP_CONDA=1
# shellcheck disable=SC1091
source "$REPO/code/env_h200.sh"
cd "$REPO"
ulimit -c 0

OUT=$REPO/results/m10_precision_h200
SW=$REPO/results/m10_sweep_h200
LC=$REPO/results/m10_longctx_h200
BUSY=$REPO/results/m7_write_policy_h200/busy_levels.json
export LONGCTX_RAW=${LONGCTX_RAW:-$TIARA_HOME/longctx_m10h}          # 逐列 CSV（大，不進 git）
export LONGCTX_LC=$LC
WORK=$TIARA_RUNS/_m10h_work/lc-$PART          # 固定路徑：--from SCREEN 接著跑時要讀前一個 job 寫的 jobs_screen.json
export LONGCTX_JOBS=$WORK

if [ "$DRY" = 0 ]; then
  [ -n "${SLURM_JOB_ID:-}" ] || { echo "[r_h200_lc] 不在 Slurm job 裡。用 sbatch 送" >&2; exit 2; }
  python - "$OUT/fmt_select/fmt_vs_mi300x.json" <<'PY' || { echo "[r_h200_lc] 格式還沒選或和 MI300X 不同（13 §3-4）→ 停" >&2; exit 2; }
import json, sys
sys.exit(0 if json.load(open(sys.argv[1]))["same"] else 1)
PY
  for m in llama qwen; do
    [ -f "$OUT/speed_$m/bw.json" ] || { echo "[r_h200_lc] 找不到 $OUT/speed_$m/bw.json（m10h-speed 還沒跑完）" >&2; exit 2; }
  done
  [ -f "$BUSY" ] || { echo "[r_h200_lc] 找不到 $BUSY" >&2; exit 2; }
  # 13 §8：使用者 10/17 接受程式和 MI300X 不同 → 只記錄差異，不擋
  echo "[r_h200_lc] 和 MI300X（bcfe5b1）不同的程式："; git diff --stat bcfe5b1 -- code | tail -15
  mkdir -p "$WORK" "$LONGCTX_RAW"
  echo "[r_h200_lc] $(date -Is) part=$PART job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-?} git=$(git rev-parse --short HEAD)"
  nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.used --format=csv
fi

STARTED=1; [ -n "$FROM" ] && STARTED=0
SEEN=""
_gate() {
  local name=$1
  SEEN="$SEEN $name"
  if [ "$STARTED" = 0 ]; then
    if [ "$name" = "$FROM" ]; then STARTED=1; else [ "$LIST" = 0 ] && echo "[skip] $name"; return 1; fi
  fi
  return 0
}
step() {
  local name=$1; shift
  _gate "$name" || return 0
  if [ "$LIST" = 1 ]; then echo "$name"; return 0; fi
  if [ "$DRY" = 1 ]; then printf '[%s] ' "$name"; printf '%q ' "$@"; echo; return 0; fi
  echo "=== $(date -Is) $name"
  "$@"; local rc=$?
  if [ $rc -ne 0 ]; then
    echo "=== $(date -Is) $name 失敗（exit $rc）→ 停下、回報（CLAUDE.md 規則 2）。接著跑：sbatch ... code/r_h200_lc.sh --part $PART --from $name" >&2
    exit $rc
  fi
}
lastrun() { ls -d "$TIARA_RUNS"/*-"$1" | sort | tail -1 | xargs basename; }
fmt_of() { python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['choice'][sys.argv[2]][sys.argv[3]])" "$OUT/fmt_select/fmt_choice.json" "$1" "$2"; }
pool() { (cd "$WORK" && python "$REPO/code/h200_pool.py" "$1" "$2"); }
guard_files_clean() {   # 給定的 gpu_guard JSON 都要 CLEAN（CLAUDE.md §3）
  python - "$@" <<'PY'
import json, sys
fs = sys.argv[1:]
bad = [f for f in fs if json.load(open(f)).get("verdict") != "CLEAN" or json.load(open(f)).get("contaminated")]
print(f"[guard] {len(fs)} 份 gpu_guard JSON，不是 CLEAN 的：{bad or '無'}")
sys.exit(0 if fs and not bad else 1)
PY
}

parts() {         # MI300X queue/40 的零件（m10_sweep parts）；GPU 忙用 H200 的 busy2（MI300X 寫死 0.7）
  local m=$1 duty rid
  duty=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['busy2'])" "$BUSY") || return 1
  m7run m10h-sweep-parts-$m python code/m10_sweep.py parts --model $m --out-dir "$SW/parts" --busy-duty "$duty" || return 1
  rid=$(lastrun m10h-sweep-parts-$m)
  guard_files_clean "$SW/parts/gpu_guard_${rid}_parts_$m.json"
}
lc_grid() {       # MI300X queue/44a（＋GPU 忙用 H200 的 busy_levels）
  local m=$1 L=$2 tier=$3 g f d rid
  g=$(fmt_of $m fp8) || return 1; f=$(fmt_of $m $tier) || return 1
  d=$LC/speed_${m}_${L}_$f; mkdir -p "$d"
  m7run m10h-lc-grid-$m-$L-$f bash -c "python code/m10_speed.py grid --model $m --fmt $f --base-fmt $g --out-dir $d --bw-json $OUT/speed_$m/bw.json --busy-levels $BUSY --main-length $L --lengths --launches 2 --reps 5 --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" || return 1
  rid=$(lastrun m10h-lc-grid-$m-$L-$f)
  guard_files_clean "$d"/gpu_guard_${rid}_L*.json
}

if [ "$PART" = speed ]; then
  for m in llama qwen; do step PARTS-$m parts $m; done
  # 使用者 10/17「先跑 128」：128K 的兩階先，再 64K（MI300X 44a 的順序是 fp8 128K、fp8 64K、int4 128K、int4 64K）
  for L in 129024 65536; do
    for tier in fp8 int4; do
      for m in llama qwen; do step LCGRID-$m-$L-$tier lc_grid $m $L $tier; done
    done
  done
else
  step PILOT  bash -c 'python code/h200_longctx_plan.py jobs-pilot && cd "$LONGCTX_JOBS" && python "$REPO/code/h200_pool.py" jobs_pilot.json 4'
  step PLAN   python code/h200_longctx_plan.py plan
  step SCREEN pool "$WORK/jobs_screen.json" 4
fi

if [ "$STARTED" = 0 ]; then
  echo "[r_h200_lc] 沒有叫做 '$FROM' 的步驟。可用的：$SEEN" >&2; exit 2
fi
[ "$DRY" = 0 ] && echo "[r_h200_lc] $(date -Is) part=$PART 全部完成；記進 RUNLOG_H200"
exit 0
