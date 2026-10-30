#!/bin/bash
#SBATCH --job-name=s3-e2e-gpu
#SBATCH --partition=h200q
#SBATCH --account=ChihChungHsu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=12:00:00
#SBATCH -o slurm-%x-%j.out
# =============================================================================
# r_h200_e2e_gpu.sh — E2E-C 第二段（docs/stage3_design_20261024/runs/E2E_gpu_mps.md）：MPS 取捨在 GPU 上驗證（量時間）。
#
#   sbatch code/r_h200_e2e_gpu.sh            # 冒煙 → stream 那一段 → 開 MPS → 冒煙 → mps 那一段
#   bash code/r_h200_e2e_gpu.sh --list | --dry-run ；--from <步驟> 接著跑
#
# 量時間：T2（平行爭用校準）判之前一律獨佔——每一步開跑前等節點上沒有我們自己的別的 job，跑完用 sacct 查重疊，
# 重疊或 gpu_guard 污染就重跑（最多 3 次），同 T1G。冒煙只驗程式：輸出在 run 目錄、不進 results、不看數字。
# 這支在 job 跑的時候不要原地改（bash 邊跑邊讀）：寫新檔再 mv 過去。
# =============================================================================
set -uo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
if [ -n "${REPO:-}" ] && [ -f "$REPO/code/env_h200.sh" ]; then :
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/code/env_h200.sh" ]; then REPO=$SLURM_SUBMIT_DIR
elif [ -f "$_here/env_h200.sh" ]; then REPO=$(cd "$_here/.." && pwd)
else echo "[r_h200_e2e_gpu] 找不到 repo（code/env_h200.sh）" >&2; exit 2
fi
export REPO
DRY=0; LIST=0; FROM=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --list) LIST=1; DRY=1; shift ;;
    --from) FROM="$2"; shift 2 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "不認得的參數：$1" >&2; exit 2 ;;
  esac
done
[ "$DRY" = 1 ] && export TIARA_SKIP_CONDA=1
export ICPE_MODEL_ROOT="${ICPE_MODEL_ROOT:-/tmp/hungweilai-models/hf-cache}"
export HF_HOME="$ICPE_MODEL_ROOT" S3_LOCAL=/tmp/hungweilai-s3
unset M7_LLAMA_GLOB M7_QWEN_GLOB M7_MODEL_GLOB
# shellcheck disable=SC1091
source "$REPO/code/env_h200.sh"
cd "$REPO"
ulimit -c 0
TP=$REPO/results/icpe_h200/t1/tier_params_h200cpu.json
OUTD=$REPO/results/s3_h200/e2e_gpu
OVL=$OUTD/overlap.csv
export P3_CONST=h200 P3_TIER=h200cpu M7_TIER_PARAMS="$TP" X1_TRACE_DIR=$TIARA_HOME/datasets/traces
JOB=${SLURM_JOB_ID:-dry}

if [ "$DRY" = 0 ]; then
  [ -n "${SLURM_JOB_ID:-}" ] || { echo "[r_h200_e2e_gpu] 不在 Slurm job 裡" >&2; exit 2; }
  [ -f "$TP" ] || { echo "[r_h200_e2e_gpu] 找不到 $TP" >&2; exit 2; }
  python -c 'import sys; sys.path.insert(0, "code"); from gpu_guard import smi_index; print("[r_h200_e2e_gpu] smi_index(0) =", smi_index(0))' \
    || { echo "[r_h200_e2e_gpu] cuda:0 對不到 nvidia-smi 的卡 → 停" >&2; exit 2; }
  python3 "$REPO/code/s3/fetch_models.py" check --root "$ICPE_MODEL_ROOT" --models llama qwen \
    || { echo "[r_h200_e2e_gpu] 模型權重不在 $ICPE_MODEL_ROOT → 停（exit 3）" >&2; exit 3; }
  want=$(python3 "$REPO/code/s3/fetch_models.py" env --root "$ICPE_MODEL_ROOT" --models llama qwen | sort)
  have=$(printf 'export HF_HOME=%s\nexport M7_LLAMA_GLOB=%s\nexport M7_QWEN_GLOB=%s\n' "$HF_HOME" "$M7_LLAMA_GLOB" "$M7_QWEN_GLOB" | sort)
  [ "$want" = "$have" ] || { echo "[r_h200_e2e_gpu] env_h200.sh 的模型路徑和 fetch_models 的不同 → 停" >&2; exit 2; }
  python3 - "$TIARA_HOME/datasets/traces" <<'PY' || { echo "[r_h200_e2e_gpu] Mooncake trace 的 sha256 不對 → 停" >&2; exit 2; }
import hashlib, sys
got = hashlib.sha256(open(f"{sys.argv[1]}/conversation_trace.jsonl", "rb").read()).hexdigest()
print("[r_h200_e2e_gpu] conversation_trace.jsonl", got[:16]); assert got.startswith("b8cbb061a85206d7"), got
PY
  python code/s3/e2e_gpu.py --self-test || { echo "[r_h200_e2e_gpu] 自檢失敗 → 停" >&2; exit 2; }
  echo "[r_h200_e2e_gpu] $(date -Is) job=$JOB node=${SLURMD_NODENAME:-?} git=$(git rev-parse --short HEAD)"
  echo "[r_h200_e2e_gpu] 同節點的 job（開跑時）："; squeue -h -w "${SLURMD_NODENAME:-$(hostname -s)}" -o '%i %u %a %b %T %M'
  nvidia-smi --query-gpu=index,name,pci.bus_id,memory.used --format=csv
  mkdir -p "$OUTD"
fi

STARTED=1; [ -n "$FROM" ] && STARTED=0
OVL_STEPS=""
step() {   # step <名稱> <m7run 短名> <指令...>：等節點沒有自己人 → 跑 → 查重疊；重疊或污染就重跑（最多 3 次）
  local name=$1 short=$2; shift 2
  if [ "$STARTED" = 0 ]; then if [ "$name" = "$FROM" ]; then STARTED=1; else return 0; fi; fi
  if [ "$LIST" = 1 ]; then echo "$name"; return 0; fi
  if [ "$DRY" = 1 ]; then printf '[%s] ' "$name"; printf '%q ' "$@"; echo; return 0; fi
  local k rc orc t0 t1
  for k in 1 2 3; do
    python3 code/icpe/t1g.py wait-clear --job "$JOB" --max-min 360 || { echo "=== $name：等不到節點空出來 → 停、回報" >&2; exit 2; }
    t0=$(date -Is); echo "=== $t0 $name（第 $k 次）"
    "$@"; rc=$?
    t1=$(date -Is)
    if [ $rc -ne 0 ] && [ $rc -ne 3 ]; then
      echo "=== $t1 $name 失敗（exit $rc）→ 停下、回報（規則 2）。接著跑：sbatch code/r_h200_e2e_gpu.sh --from $name" >&2; exit $rc
    fi
    python3 code/icpe/t1g.py overlap --job "$JOB" --start "$t0" --end "$t1" --step "$name" --short "$short" --log "$OVL"; orc=$?
    [ $orc -eq 2 ] && { echo "=== $name：sacct 查不到重疊 → 停（規則 7）" >&2; exit 2; }
    if [ $orc -eq 0 ] && [ $rc -eq 0 ]; then return 0; fi
    echo "=== $(date -Is) $name：$([ $rc -eq 3 ] && echo 'gpu_guard 污染' || echo '和我們自己的 job 重疊') → 重跑"
  done
  echo "=== $(date -Is) $name：3 次都重疊或污染 → 記成重疊、不進判定，接著下一步" >&2
  OVL_STEPS="$OVL_STEPS $name"
  return 0
}
smoke() {   # 冒煙：只驗程式（輸出在 run 目錄）；失敗 → 停
  local name=$1; shift
  if [ "$STARTED" = 0 ]; then if [ "$name" = "$FROM" ]; then STARTED=1; else return 0; fi; fi
  if [ "$LIST" = 1 ]; then echo "$name"; return 0; fi
  if [ "$DRY" = 1 ]; then printf '[%s] ' "$name"; printf '%q ' "$@"; echo; return 0; fi
  echo "=== $(date -Is) $name（冒煙）"
  "$@" || { echo "=== $name 冒煙失敗 → 停、回報（規則 2）" >&2; exit 2; }
}
mps_on() {
  [ "$DRY" = 1 ] && { echo "[mps_on]"; return 0; }
  command -v nvidia-cuda-mps-control >/dev/null || { echo "[r_h200_e2e_gpu] 找不到 nvidia-cuda-mps-control" >&2; exit 2; }
  export CUDA_MPS_PIPE_DIRECTORY=$S3_LOCAL/mps_pipe_${SLURM_JOB_ID} CUDA_MPS_LOG_DIRECTORY=$S3_LOCAL/mps_log_${SLURM_JOB_ID}
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  nvidia-cuda-mps-control -d || { echo "[r_h200_e2e_gpu] MPS daemon 開不起來 → 停" >&2; exit 2; }
  export GPU_GUARD_JOB_CGROUP_OWN=1
  trap 'echo quit | nvidia-cuda-mps-control; sleep 2; mkdir -p $OUTD/mps_log_$SLURM_JOB_ID; cp $CUDA_MPS_LOG_DIRECTORY/* $OUTD/mps_log_$SLURM_JOB_ID/ 2>/dev/null; rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY' EXIT
  echo "[r_h200_e2e_gpu] MPS daemon 開了（pipe $CUDA_MPS_PIPE_DIRECTORY）"
}
order() {   # order <seed> <行...>：固定 seed 打亂
  python3 - "$@" <<'PY'
import random, sys
xs = sys.argv[2:]
random.Random(int(sys.argv[1])).shuffle(xs)
print("\n".join(xs))
PY
}
G() {   # G <bg> <mode> <rep>：一個 GPU run
  local bg=$1 m=$2 r=$3 tag
  tag="$(echo "$bg" | cut -d: -f1)-$m-r$r"
  step "E2EG-$tag" "s3-e2eg-$tag" m7run "s3-e2eg-$tag" python code/s3/e2e_gpu.py gpu --bg "$bg" --mode "$m" --rep "$r" --out "$OUTD"
}

CP=cprefill:2048:65536; MX=mixed:512:32:8192
smoke E2EG-SMOKE-STREAM m7run s3-e2eg-smoke-stream bash -c "python code/s3/e2e_gpu.py gpu --bg $CP --mode stream --max-req 40 --out \"\$RUN_DIR/smoke\""
G idle stream 0
mapfile -t L1 < <(order 20261029 "$CP stream 0" "$MX stream 0" "$CP stream 1" "$MX stream 1")
for x in "${L1[@]}"; do read -r bg m r <<< "$x"; G "$bg" "$m" "$r"; done
[ "$LIST" = 1 ] || mps_on
smoke E2EG-SMOKE-MPS m7run s3-e2eg-smoke-mps bash -c "python code/s3/e2e_gpu.py gpu --bg $CP --mode mps --max-req 40 --out \"\$RUN_DIR/smoke\""
MPSL=()
for bg in $CP $MX; do for m in mps mps87 mps75 mps50; do MPSL+=("$bg $m 0"); done; MPSL+=("$bg mps 1"); done
mapfile -t L2 < <(order 20261030 "${MPSL[@]}")
for x in "${L2[@]}"; do read -r bg m r <<< "$x"; G "$bg" "$m" "$r"; done
echo "[r_h200_e2e_gpu] $(date -Is) 完成；3 次都重疊的步驟：${OVL_STEPS:-無}"
