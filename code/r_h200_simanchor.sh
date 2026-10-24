#!/bin/bash
#SBATCH --job-name=r-h200-simanchor
#SBATCH --partition=h200q
#SBATCH --account=ChihChungHsu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=01:00:00
#SBATCH -o slurm-%x-%j.out
# =============================================================================
# r_h200_simanchor.sh — 第一階段的 A2 與 C7 錨點，只跑限速層那一半（2026-10-16）
#
#   sbatch --dependency=afterany:<r-h200-brerun> code/r_h200_simanchor.sh
#   bash code/r_h200_simanchor.sh --dry-run | --list ；--from <步驟>
#
# 為什麼：03_plan_h200.md 第 101 行原本「不做 C2、C7-anchor、A2」（量的是 MLSteam 的真實本地 SSD 和 NFS）。
# 使用者 10/16：「基本上要復現 MI300 都要跟」「都用模擬限速」「如果需要也需要改 code 成 NV 用的」。
# 所以照 MI300X 10/08 的指令（run_commands.tsv：20261008-131619-m7-a2、132038-m7-a2share、133016-m7-c7anchor）重跑，
# 只加 --io sim：限速層（MI300X 的 tier_params.json，env_h200.sh 的 M7_TIER_PARAMS）那一半照跑，真實裝置那一半不跑 → NOT_MEASURED。
#   - C2（真實裝置的頻寬校準）沒有限速層的對應，不跑 → NOT_MEASURED（限速層照舊用 MI300X 的參數）。
#   - hgx-cn01 上我們能寫的只有本地的 md0／md1，沒有自己的 NFS 目錄（/home 是別人的），所以真實的 nfs 本來也量不到。
# D3（m8_kappa_map.py）的 c7_anchor sim 兩列要用這裡的 c7_anchor.csv。
# 開跑前檢查：節點上沒有我們（$USER）別的 job（記憶 h200-no-self-cotenancy-for-timing）。任何一步失敗 → 停（規則 2）。
# =============================================================================
set -uo pipefail
DRY=0; LIST=0; FROM=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --list) LIST=1; DRY=1; shift ;;
    --from) FROM="$2"; shift 2 ;;
    *) echo "不認得的參數：$1" >&2; exit 2 ;;
  esac
done
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
if [ -n "${REPO:-}" ] && [ -f "$REPO/code/env_h200.sh" ]; then :
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/code/env_h200.sh" ]; then REPO=$SLURM_SUBMIT_DIR
elif [ -f "$_here/env_h200.sh" ]; then REPO=$(cd "$_here/.." && pwd)
else echo "[r_h200_simanchor] 找不到 repo" >&2; exit 2
fi
export REPO
[ "$DRY" = 1 ] && export TIARA_SKIP_CONDA=1
# shellcheck disable=SC1091
source "$REPO/code/env_h200.sh"
cd "$REPO"
P=code/m7_write_policy.py

if [ "$DRY" = 0 ]; then
  [ -n "${SLURM_JOB_ID:-}" ] || { echo "[r_h200_simanchor] 不在 Slurm job 裡" >&2; exit 2; }
  awk -v d="$M7_DISK" '$3==d{f=1} END{exit !f}' /proc/diskstats || { echo "[r_h200_simanchor] M7_DISK=$M7_DISK 不在 /proc/diskstats → 停" >&2; exit 2; }
  python -c 'import sys; sys.path.insert(0, "code"); from gpu_guard import smi_index; print("[r_h200_simanchor] smi_index(0) =", smi_index(0))' \
    || { echo "[r_h200_simanchor] cuda:0 對不到 nvidia-smi 的卡 → 停" >&2; exit 2; }
  others=$(squeue -h -w "${SLURMD_NODENAME:-$(hostname -s)}" -u "$USER" -t R -o '%i %j' | awk -v me="$SLURM_JOB_ID" '$1!=me')
  if [ -n "$others" ]; then
    echo "[r_h200_simanchor] 節點上還有我們自己的 job 在跑 → 停：" >&2; echo "$others" >&2; exit 2
  fi
  for f in a2_fifo.csv a2_share.csv c7_anchor.csv; do
    [ ! -e "$M7_RES/$f" ] || { echo "[r_h200_simanchor] $M7_RES/$f 已經存在（附加寫入會混在一起）→ 停" >&2; exit 2; }
  done
  echo "[r_h200_simanchor] 同節點的 job（開跑時）："; squeue -h -w "${SLURMD_NODENAME:-$(hostname -s)}" -o '%i %u %a %b %T %M'
  echo "[r_h200_simanchor] $(date -Is) job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-?} M7_RES=$M7_RES M7_TIER_PARAMS=$M7_TIER_PARAMS git=$(git rev-parse --short HEAD)"
fi

STARTED=1; [ -n "$FROM" ] && STARTED=0
step() {
  local name=$1; shift
  if [ "$STARTED" = 0 ]; then if [ "$name" = "$FROM" ]; then STARTED=1; else return 0; fi; fi
  if [ "$LIST" = 1 ]; then echo "$name"; return 0; fi
  if [ "$DRY" = 1 ]; then printf '[%s] ' "$name"; printf '%q ' "$@"; echo; return 0; fi
  echo "=== $(date -Is) $name"
  "$@"; local rc=$?
  if [ $rc -ne 0 ]; then echo "=== $(date -Is) $name 失敗（exit $rc）→ 停下、回報。接著跑：sbatch code/r_h200_simanchor.sh --from $name" >&2; exit $rc; fi
}
# ===== MI300X 10/08 的順序與參數（C2 不跑），只加 --io sim =====
step A2-sim        m7run m7h-a2-sim        env M7_IO_MODEL=fifo  python $P a2 --L 16384 --reps 6 --devs local nfs --io sim
step A2-share-sim  m7run m7h-a2share-sim   env M7_IO_MODEL=share python $P a2 --L 16384 --reps 3 --devs local nfs --io sim
step C7-anchor-sim m7run m7h-c7anchor-sim  env M7_IO_MODEL=fifo  python $P anchor --L 8192 16384 32768 --reps 3 --io sim
[ "$DRY" = 0 ] && echo "[r_h200_simanchor] $(date -Is) 全部完成 → $M7_RES"
exit 0
