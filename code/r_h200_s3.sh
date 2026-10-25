#!/bin/bash
#SBATCH --job-name=s3
#SBATCH --partition=h200q
#SBATCH --account=ChihChungHsu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=4:00:00
#SBATCH -o slurm-%x-%j.out
# =============================================================================
# r_h200_s3.sh — 第三階段（docs/stage3_design_20261024/）在 H200 的 runner。每一步一個 run 目錄（m7run）。
#
#   sbatch --job-name=s3-fetch --time=1:00:00 code/r_h200_s3.sh --part fetch     # 模型權重 → hgx-cn01 /tmp（不量時間）
#   bash code/r_h200_s3.sh --part <p> --list | --dry-run ；--from <步驟> 接著跑
#
# 排程（使用者 2026-10-25「排滿：stage3 品質先填空檔」，memory stage3-gpu-scheduling）：
#   品質／不量時間的 part（TIMING=0）：節點上有 ICPE 的量時間 job 在跑 → 不開跑（exit 3）。反方向（我們在跑時 ICPE 量時間的 job 開跑）
#     由送出端處理：送出時把下一個要跑的 ICPE 量時間 job 加上 afterany:<這個 job>（不動 hold 中的 ICPE job）。
#   量時間的 part（TIMING=1）：節點上有我們自己的任何別的 job（含 DL_Lab4、ICPE）→ 不開跑（memory h200-no-self-cotenancy-for-timing）。
# 模型權重放 hgx-cn01 本地 /tmp（使用者 2026-10-25；/work 的 hf-cache 自 10/23 起是空的）：要 GPU 的 part 開跑前 ensure_models。
# sbatch 送出時會複製一份；開跑時改執行 repo 裡的版本（R_H200_S3_REEXEC）。跑的時候不要原地改這支：寫新檔再 mv 過去。
# =============================================================================
set -uo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
if [ -n "${REPO:-}" ] && [ -f "$REPO/code/env_h200.sh" ]; then :
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/code/env_h200.sh" ]; then REPO=$SLURM_SUBMIT_DIR
elif [ -f "$_here/env_h200.sh" ]; then REPO=$(cd "$_here/.." && pwd)
else echo "[r_h200_s3] 找不到 repo（code/env_h200.sh）" >&2; exit 2
fi
export REPO
if [ -z "${R_H200_S3_REEXEC:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  export R_H200_S3_REEXEC=1
  echo "[r_h200_s3] 改執行 repo 裡的版本：$REPO/code/r_h200_s3.sh（git $(git -C "$REPO" rev-parse --short HEAD)）"
  exec bash "$REPO/code/r_h200_s3.sh" "$@"
fi
PART=""; DRY=0; LIST=0; FROM=""
while [ $# -gt 0 ]; do
  case "$1" in
    --part) PART="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --list) LIST=1; DRY=1; shift ;;
    --from) FROM="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "不認得的參數：$1" >&2; exit 2 ;;
  esac
done
case "$PART" in
  fetch) TIMING=0 ;;
  *) echo "要 --part fetch" >&2; exit 2 ;;
esac
ICPE_TIMING_RE='^icpe-(p01s1|p01ab|p01brr|p02|p2|p11t)$'     # r_h200_icpe.sh 裡 TIMING=1 的 part

# 模型權重：本地 /tmp（code/s3/fetch_models.py）。env_h200.sh 的 M7_*_GLOB 用 ${HF_HOME}，所以先設 HF_HOME
export S3_MODEL_ROOT=/tmp/hungweilai-models/hf-cache
export HF_HOME=$S3_MODEL_ROOT
[ "$DRY" = 1 ] && export TIARA_SKIP_CONDA=1
# shellcheck disable=SC1091
source "$REPO/code/env_h200.sh"
eval "$(python3 "$REPO/code/s3/fetch_models.py" env --root "$S3_MODEL_ROOT")"       # M7_LLAMA_GLOB、M7_QWEN_GLOB、M7_QWEN14_GLOB
cd "$REPO"
ulimit -c 0
S3=$REPO/results/s3_h200

if [ "$DRY" = 0 ]; then
  [ -n "${SLURM_JOB_ID:-}" ] || { echo "[r_h200_s3] 不在 Slurm job 裡" >&2; exit 2; }
  NODE="${SLURMD_NODENAME:-$(hostname -s)}"
  mine=$(squeue -h -w "$NODE" -u "$USER" -t R -o '%i %j' | awk -v me="$SLURM_JOB_ID" '$1!=me')
  icpe_t=$(printf '%s\n' "$mine" | awk '{print $2}' | grep -E "$ICPE_TIMING_RE" || true)
  if [ -n "$icpe_t" ]; then
    echo "[r_h200_s3] 節點上有 ICPE 量時間的 job 在跑（$icpe_t）→ 不開跑（exit 3；stage3 不和 ICPE 量時間重疊）" >&2; exit 3
  fi
  if [ "$TIMING" = 1 ] && [ -n "$mine" ]; then
    echo "[r_h200_s3] 量時間：節點上還有我們自己的 job 在跑 → 停：" >&2; echo "$mine" >&2; exit 2
  fi
  echo "[r_h200_s3] $(date -Is) part=$PART job=$SLURM_JOB_ID node=$NODE git=$(git rev-parse --short HEAD) timing=$TIMING"
  echo "[r_h200_s3] 同節點的 job（開跑時）："; squeue -h -w "$NODE" -o '%i %u %a %j %b %T %M'
  nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.used --format=csv
  mkdir -p "$S3/env"
fi

STARTED=1; [ -n "$FROM" ] && STARTED=0
SEEN=""
step() {
  local name=$1; shift
  SEEN="$SEEN $name"
  if [ "$STARTED" = 0 ]; then if [ "$name" = "$FROM" ]; then STARTED=1; else return 0; fi; fi
  if [ "$LIST" = 1 ]; then echo "$name"; return 0; fi
  if [ "$DRY" = 1 ]; then printf '[%s] ' "$name"; printf '%q ' "$@"; echo; return 0; fi
  echo "=== $(date -Is) $name"
  "$@"; local rc=$?
  if [ $rc -ne 0 ]; then echo "=== $(date -Is) $name 失敗（exit $rc）→ 停下、回報（規則 2）。接著跑：sbatch ... code/r_h200_s3.sh --part $PART --from $name" >&2; exit $rc; fi
}
# 要 GPU 的 part 開跑前呼叫：戳記對不上 → 重抓一次 → 再核對（重抓後還不對就停）
ensure_models() {
  [ "$DRY" = 1 ] && { echo "[ensure_models $*]"; return 0; }
  python3 code/s3/fetch_models.py check --root "$S3_MODEL_ROOT" --models "$@" && return 0
  echo "[r_h200_s3] 權重不在或對不上 → 重抓：$*"
  m7run s3-fetch python code/s3/fetch_models.py fetch --root "$S3_MODEL_ROOT" --models "$@" || return 2
  python3 code/s3/fetch_models.py check --root "$S3_MODEL_ROOT" --models "$@"
}

if [ "$PART" = fetch ]; then
  step FETCH m7run s3-fetch python code/s3/fetch_models.py fetch --root "$S3_MODEL_ROOT" --models llama qwen qwen14
  step CHECK-FULL python3 code/s3/fetch_models.py check --root "$S3_MODEL_ROOT" --models llama qwen qwen14 --full
fi

if [ "$STARTED" = 0 ]; then echo "[r_h200_s3] 沒有叫做 '$FROM' 的步驟。可用的：$SEEN" >&2; exit 2; fi
exit 0
