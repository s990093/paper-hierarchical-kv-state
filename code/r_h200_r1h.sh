#!/bin/bash
#SBATCH --job-name=r-h200-r1h
#SBATCH --partition=h200q
#SBATCH --account=ChihChungHsu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=14:00:00
#SBATCH -o slurm-%x-%j.out
# =============================================================================
# r_h200_r1h.sh — 第 1 輪：H200 自己的模擬存活設定上 GPU 確認（2026-10-16，使用者：「3 個都上 GPU」）
#
#   env -u TMPDIR -u CLAUDE_CODE_TMPDIR sbatch --dependency=afterany:<r-h200-simanchor> code/r_h200_r1h.sh
#   bash code/r_h200_r1h.sh --dry-run | --list ；--from <步驟>
#
# 為什麼：H200 的 r1_sim（run 20261016-095639-m7h-r1-sim-llama31-8b；r1_sim_configs.csv，run 20261016-173455-m7h-r1-sim-an）
# 篩出的 Llama 存活設定（≥3/5 seed 贏所有對照組 ≥5%）有 3 個，和 MI300X 的那一個不同。
# P3（r_h200_rest.sh）照 MI300X 的指令只確認了 MI300X 的存活設定（11.6、free、doc、local、0.5）：0/5。
# 方法照 11_round1_plan.md 追加 4（MI300X 確認存活設定的做法），事先寫死在 RUNLOG_H200「第 1 輪：H200 自己的存活設定」一節（這個 job 送出之前 commit）：
#   - 每格只測一個寫入時決定的策略：模擬 5 個 seed 裡當「最好的寫入時決定」次數最多的
#     A 11.6／hold／nfs／0.5 → S5L（5/5）；B 35.42／free／local／0.5 → S5L（5/5）；C 35.42／hold／nfs／0.25 → S5c（4/5，seed 4 是 S5s）
#   - 對照組：S1、S2b、S4、S4+、S4+P、S4L（取最好的）、S4B、S4W、S4C；seed 0–4，每個 3 次，--verify
#   - 判準：寫入時決定的回來請求 TTFT 中位數同時贏「最好的不看位置」、S4B、S4W、S4C 各 ≥5%，≥3/5 個 seed → 存活
# 指令和 P3（r_h200_rest.sh 的 P3-gpu-s*）相同，只換 release、ssd-dev、cpu-fracs、寫入時決定的策略、CPU 頻寬與輸出檔：
#   35.42 GiB/s＝tier_params.json 的 CPU 層（35.4159），所以不設 M7_CPU_GIBPS（和模擬的 cpu_gibps=None 相同）；11.6 照 P3 設 M7_CPU_GIBPS=11.6。
#   f(i)：同 P3（$M7_RES/calib_c1.csv，H200 R2 的 C1；模擬用的也是這份）；M7_IO_MODEL=share（模擬也是 share）。
# 開跑前檢查：節點上沒有我們（$USER）別的 job（記憶 h200-no-self-cotenancy-for-timing）；輸出 CSV 還不存在（附加寫入）。任何一步失敗 → 停（規則 2）。
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
else echo "[r_h200_r1h] 找不到 repo" >&2; exit 2
fi
export REPO
[ "$DRY" = 1 ] && export TIARA_SKIP_CONDA=1
# shellcheck disable=SC1091
source "$REPO/code/env_h200.sh"
cd "$REPO"
P=code/m7_write_policy.py
X=$REPO/results/m7_explore_h200
CTRL="S1 S2b S4 S4+ S4+P S4L S4B S4W S4C"
OUT_A=$X/r1h_gpu_llama_doc_hold_nfs_c11.6_f0.5.csv
OUT_B=$X/r1h_gpu_llama_doc_free_local_c35.42_f0.5.csv
OUT_C=$X/r1h_gpu_llama_doc_hold_nfs_c35.42_f0.25.csv

if [ "$DRY" = 0 ]; then
  [ -n "${SLURM_JOB_ID:-}" ] || { echo "[r_h200_r1h] 不在 Slurm job 裡" >&2; exit 2; }
  [ -z "${TMPDIR:-}" ] || [ "$TMPDIR" = /tmp ] || echo "[r_h200_r1h] 注意：TMPDIR=$TMPDIR（之前的 H200 job 沒有設）" >&2
  awk -v d="$M7_DISK" '$3==d{f=1} END{exit !f}' /proc/diskstats || { echo "[r_h200_r1h] M7_DISK=$M7_DISK 不在 /proc/diskstats → 停" >&2; exit 2; }
  python -c 'import sys; sys.path.insert(0, "code"); from gpu_guard import smi_index; print("[r_h200_r1h] smi_index(0) =", smi_index(0))' \
    || { echo "[r_h200_r1h] cuda:0 對不到 nvidia-smi 的卡 → 停" >&2; exit 2; }
  others=$(squeue -h -w "${SLURMD_NODENAME:-$(hostname -s)}" -u "$USER" -t R -o '%i %j' | awk -v me="$SLURM_JOB_ID" '$1!=me')
  if [ -n "$others" ]; then
    echo "[r_h200_r1h] 節點上還有我們自己的 job 在跑 → 停：" >&2; echo "$others" >&2; exit 2
  fi
  if [ -z "$FROM" ]; then
    for f in "$OUT_A" "$OUT_B" "$OUT_C"; do
      [ ! -e "$f" ] || { echo "[r_h200_r1h] $f 已經存在（附加寫入會混在一起）→ 停" >&2; exit 2; }
    done
  fi
  echo "[r_h200_r1h] 同節點的 job（開跑時）："; squeue -h -w "${SLURMD_NODENAME:-$(hostname -s)}" -o '%i %u %a %b %T %M'
  echo "[r_h200_r1h] $(date -Is) job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-?} M7_RES=$M7_RES M7_TIER_PARAMS=$M7_TIER_PARAMS git=$(git rev-parse --short HEAD)"
fi

STARTED=1; [ -n "$FROM" ] && STARTED=0
step() {
  local name=$1; shift
  if [ "$STARTED" = 0 ]; then if [ "$name" = "$FROM" ]; then STARTED=1; else return 0; fi; fi
  if [ "$LIST" = 1 ]; then echo "$name"; return 0; fi
  if [ "$DRY" = 1 ]; then printf '[%s] ' "$name"; printf '%q ' "$@"; echo; return 0; fi
  echo "=== $(date -Is) $name"
  "$@"; local rc=$?
  if [ $rc -ne 0 ]; then echo "=== $(date -Is) $name 失敗（exit $rc）→ 停下、回報。接著跑：sbatch code/r_h200_r1h.sh --from $name" >&2; exit $rc; fi
}
# ===== A：11.6 GiB/s、hold、doc、nfs、0.5，寫入時決定＝S5L =====
for seed in 0 1 2 3 4; do
  step A-s$seed m7run m7h-r1h-gpu-llama-doc-hold-nfs-c11.6-f0.5-s$seed \
    env M7_IO_MODEL=share M7_CPU_GIBPS=11.6 M7_B2_OUT="$OUT_A" \
    python $P b --b2 --workload doc --release hold --ssd-dev nfs --wl-seed $seed --strategies $CTRL S5L --cpu-fracs 0.5 --reps 3 --verify
done
# ===== B：35.42 GiB/s（tier_params 的 CPU 層）、free、doc、local、0.5，寫入時決定＝S5L =====
for seed in 0 1 2 3 4; do
  step B-s$seed m7run m7h-r1h-gpu-llama-doc-free-local-c35.42-f0.5-s$seed \
    env M7_IO_MODEL=share M7_B2_OUT="$OUT_B" \
    python $P b --b2 --workload doc --release free --ssd-dev local --wl-seed $seed --strategies $CTRL S5L --cpu-fracs 0.5 --reps 3 --verify
done
# ===== C：35.42 GiB/s、hold、doc、nfs、0.25，寫入時決定＝S5c =====
for seed in 0 1 2 3 4; do
  step C-s$seed m7run m7h-r1h-gpu-llama-doc-hold-nfs-c35.42-f0.25-s$seed \
    env M7_IO_MODEL=share M7_B2_OUT="$OUT_C" \
    python $P b --b2 --workload doc --release hold --ssd-dev nfs --wl-seed $seed --strategies $CTRL S5c --cpu-fracs 0.25 --reps 3 --verify
done
[ "$DRY" = 0 ] && echo "[r_h200_r1h] $(date -Is) 全部完成 → $X/r1h_gpu_*.csv；分析：python code/m7_round1_analyze.py gpu --csv … --tag …（head node）"
exit 0
