#!/bin/bash
#SBATCH --job-name=r-h200-all
#SBATCH --partition=h200q
#SBATCH --account=ChihChungHsu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=12:00:00
#SBATCH -o slurm-%x-%j.out
# =============================================================================
# r_h200_all.sh — R0（h200_probe.sh）＋ R1–R3（r_h200.sh）放進同一個 job（2026-10-15）
#
#   sbatch code/r_h200_all.sh            # 在 repo 根目錄送出
#
# 為什麼合併：10/15 送出前 `sbatch --test-only` 預估 10/27 才開始（h200q 一個節點 8 張卡全滿、前面有 5 個 2 天的 job）。
#   分兩個 job，R0 結束後卡會先給排在前面的人，R1–R3 要再排一次。
# 閘門不變（03_plan_h200 §R0）：
#   - R0 沒過（h200_probe.sh exit≠0：不是 H200、matmul 不對、可見性沒判定）→ 停，不跑 R1。
#   - M7_DISK：只有在 R0 過、而且 probe 建議的裝置真的在 /proc/diskstats 時才用；否則停（規則 7）。
#     用的值記在 slurm-*.out 與 R0 的 run 目錄（m7_disk.tsv）；job 結束後要回填進 env_h200.sh。
#   - 之後的閘門全是 r_h200.sh 原本的（R1-check 不過 → 停；任何一步失敗 → 停）。
# 被砍時：R0 已經過了 → `M7_DISK=<值> sbatch code/r_h200.sh --from <步驟名>`（r_h200.sh 檔頭）。
# =============================================================================
set -uo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
if [ -n "${REPO:-}" ] && [ -f "$REPO/code/env_h200.sh" ]; then :
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/code/env_h200.sh" ]; then REPO=$SLURM_SUBMIT_DIR
elif [ -f "$_here/env_h200.sh" ]; then REPO=$(cd "$_here/.." && pwd)
elif [ -f "$HOME/paper-hierarchical-kv-state/code/env_h200.sh" ]; then REPO=$HOME/paper-hierarchical-kv-state
else echo "[r_h200_all] 找不到 repo（code/env_h200.sh）。請在 repo 根目錄 sbatch，或先 export REPO" >&2; exit 2
fi
export REPO
# shellcheck disable=SC1091
source "$REPO/code/env_h200.sh"
cd "$REPO"
[ -n "${SLURM_JOB_ID:-}" ] || { echo "[r_h200_all] 不在 Slurm job 裡。用 sbatch 送（H200_SLURM §5.1）" >&2; exit 2; }
echo "[r_h200_all] $(date -Is) job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-?} git=$(git rev-parse --short HEAD)"

# ===== R0 =====
echo "=== $(date -Is) R0-probe"
m7run h200-probe bash code/h200_probe.sh
rc=$?
if [ $rc -ne 0 ]; then
  echo "=== $(date -Is) R0 沒過（exit $rc）→ 停，不跑 R1–R3。看 results/hw_h200.json、docs/h200/REPORT_H200.md（CLAUDE.md 規則 2）" >&2
  exit $rc
fi

# ===== M7_DISK：只接受 probe 的建議、而且要在 /proc/diskstats 裡 =====
if [ "$M7_DISK" = R0_UNSET ]; then
  M7_DISK=$(python - "$REPO/results/hw_h200.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["storage"]
print(s["suggested_M7_DISK"] if s.get("M7_DISK_in_diskstats") == "yes" and s.get("suggested_M7_DISK") not in (None, "", "NONE", "NOT_MEASURED") else "")
PY
)
  if [ -z "$M7_DISK" ]; then
    echo "[r_h200_all] probe 沒有給出在 /proc/diskstats 裡的 M7_DISK → 停（規則 7）。看 R0 run 目錄的 probe/m7_disk.tsv" >&2
    exit 2
  fi
  export M7_DISK
  echo "[r_h200_all] M7_DISK=$M7_DISK（R0 的建議；job 結束後回填進 code/env_h200.sh）"
fi

# ===== R1–R3 =====
bash "$REPO/code/r_h200.sh"
