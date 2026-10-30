#!/bin/bash
#SBATCH --job-name=icpe-cal
#SBATCH --partition=h200q
#SBATCH --account=ChihChungHsu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=3:00:00
#SBATCH -o slurm-%x-%j.out
# =============================================================================
# r_h200_calib.sh — ICPE T2（docs/h200/icpe/T2_parallel_calibration.md）：平行爭用校準的一個 job（一張卡）。
#
#   由 code/icpe/calib_submit.sh 送出：sbatch --job-name=icpe-cal-<波>-<i> code/r_h200_calib.sh --cond solo|par --wave <波> --n <這一波的 job 數>
#   bash code/r_h200_calib.sh --cond solo --wave s1 --n 1 --dry-run
#
# 兩個部件（登記 §1）：mb＝b0.py 還原微基準（idle、stream、7 個原語 × 6 次）；rp＝x1_gpu.py GPU 重播（k=8、CPU 128、HINT-PF、前 600 個請求，T1 常數）。
# solo：開跑前等節點上沒有我們自己的別的 job；par：等同一波的 job 都在跑，再等節點上沒有這一波以外的我們的 job（登記 §2）。
# 每個部件跑完用 calib.py window 記時間窗（登記 §3）；任何部件失敗（exit 不是 0 或 3）→ 停下、回報（規則 2）。
# =============================================================================
set -uo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
if [ -n "${REPO:-}" ] && [ -f "$REPO/code/env_h200.sh" ]; then :
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/code/env_h200.sh" ]; then REPO=$SLURM_SUBMIT_DIR
elif [ -f "$_here/env_h200.sh" ]; then REPO=$(cd "$_here/.." && pwd)
else echo "[r_h200_calib] 找不到 repo（code/env_h200.sh）" >&2; exit 2
fi
export REPO
COND=""; WAVE=""; N=1; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --cond) COND="$2"; shift 2 ;;
    --wave) WAVE="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "不認得的參數：$1" >&2; exit 2 ;;
  esac
done
case "$COND" in solo|par) ;; *) echo "要 --cond solo|par" >&2; exit 2 ;; esac
[ -n "$WAVE" ] || { echo "要 --wave" >&2; exit 2; }
[ "$DRY" = 1 ] && export TIARA_SKIP_CONDA=1
export ICPE_MODEL_ROOT="${ICPE_MODEL_ROOT:-/tmp/hungweilai-models/hf-cache}"
export HF_HOME="$ICPE_MODEL_ROOT" S3_LOCAL=/tmp/hungweilai-s3
unset M7_LLAMA_GLOB M7_QWEN_GLOB M7_MODEL_GLOB
# shellcheck disable=SC1091
source "$REPO/code/env_h200.sh"
cd "$REPO"
ulimit -c 0
TP=$REPO/results/icpe_h200/t1/tier_params_h200cpu.json
CAL=$REPO/results/icpe_h200/calib
export P3_CONST=h200 P3_TIER=h200cpu M7_TIER_PARAMS="$TP" X1_TRACE_DIR=$TIARA_HOME/datasets/traces
JOB=${SLURM_JOB_ID:-dry}
PRIMS=h2d_perblock,h2d_batch,recompute1,e2e_cake_bf16_cpu,e2e_cake_int4_cpu,e2e_cake_int4_b32,e2e_load_int4_b32

if [ "$DRY" = 0 ]; then
  [ -n "${SLURM_JOB_ID:-}" ] || { echo "[r_h200_calib] 不在 Slurm job 裡" >&2; exit 2; }
  [ -f "$TP" ] || { echo "[r_h200_calib] 找不到 $TP" >&2; exit 2; }
  python -c 'import sys; sys.path.insert(0, "code"); from gpu_guard import smi_index; print("[r_h200_calib] smi_index(0) =", smi_index(0))' \
    || { echo "[r_h200_calib] cuda:0 對不到 nvidia-smi 的卡 → 停" >&2; exit 2; }
  python3 "$REPO/code/s3/fetch_models.py" check --root "$ICPE_MODEL_ROOT" --models llama qwen \
    || { echo "[r_h200_calib] 模型權重不在 $ICPE_MODEL_ROOT → 停（exit 3）" >&2; exit 3; }
  want=$(python3 "$REPO/code/s3/fetch_models.py" env --root "$ICPE_MODEL_ROOT" --models llama qwen | sort)
  have=$(printf 'export HF_HOME=%s\nexport M7_LLAMA_GLOB=%s\nexport M7_QWEN_GLOB=%s\n' "$HF_HOME" "$M7_LLAMA_GLOB" "$M7_QWEN_GLOB" | sort)
  [ "$want" = "$have" ] || { echo "[r_h200_calib] env_h200.sh 的模型路徑和 fetch_models 的不同 → 停" >&2; exit 2; }
  python3 - "$TIARA_HOME/datasets/traces" <<'PY' || { echo "[r_h200_calib] Mooncake trace 的 sha256 不對 → 停" >&2; exit 2; }
import hashlib, sys
got = hashlib.sha256(open(f"{sys.argv[1]}/conversation_trace.jsonl", "rb").read()).hexdigest()
print("[r_h200_calib] conversation_trace.jsonl", got[:16]); assert got.startswith("b8cbb061a85206d7"), got
PY
  echo "[r_h200_calib] $(date -Is) cond=$COND wave=$WAVE n=$N job=$JOB node=${SLURMD_NODENAME:-?} git=$(git rev-parse --short HEAD)"
  echo "[r_h200_calib] 同節點的 job（開跑時）："; squeue -h -w "${SLURMD_NODENAME:-$(hostname -s)}" -o '%i %u %a %b %T %M'
  nvidia-smi --query-gpu=index,name,pci.bus_id,memory.used --format=csv
  mkdir -p "$CAL/raw"
  if [ "$COND" = solo ]; then
    python3 code/icpe/t1g.py wait-clear --job "$JOB" --max-min 360 || { echo "[r_h200_calib] 等不到節點空出來 → 停" >&2; exit 2; }
  else
    python3 code/icpe/calib.py barrier --wave "$WAVE" --n "$N" --max-min 30 || exit 2
    python3 code/icpe/calib.py wait-others --job "$JOB" --wave "$WAVE" --max-min 360 || exit 2
  fi
fi

comp() {   # comp <mb|rp> <m7run 短名> <指令...>
  local c=$1 short=$2 t0 t1 rc; shift 2
  if [ "$DRY" = 1 ]; then printf '[%s] ' "$c"; printf '%q ' "$@"; echo; return 0; fi
  t0=$(date -Is); echo "=== $t0 $c"
  "$@"; rc=$?
  t1=$(date -Is)
  python3 code/icpe/calib.py window --job "$JOB" --cond "$COND" --wave "$WAVE" --comp "$c" --start "$t0" --end "$t1" --short "$short" --rc "$rc" \
    || { echo "[r_h200_calib] sacct 查不到 → 停（規則 7）" >&2; exit 2; }
  if [ $rc -ne 0 ] && [ $rc -ne 3 ]; then echo "=== $t1 $c 失敗（exit $rc）→ 停下、回報（規則 2）" >&2; exit $rc; fi
  return 0
}

comp mb "icpe-cal-$WAVE-$JOB-mb" m7run "icpe-cal-$WAVE-$JOB-mb" python code/s3/b0.py run --model llama --setup b0 --modes stream --bgs idle \
  --prims "$PRIMS" --reps 6 --seed 1 --out "$CAL/raw/mb_$JOB.csv"
comp rp "icpe-cal-$WAVE-$JOB-rp" m7run "icpe-cal-$WAVE-$JOB-rp" python code/icpe/x1_gpu.py gpu --out "$CAL/raw" --k 8 --cpu 128 --policy HINT-PF \
  --max-req 600 --csv "rp_$JOB.csv"
echo "[r_h200_calib] $(date -Is) 完成"
