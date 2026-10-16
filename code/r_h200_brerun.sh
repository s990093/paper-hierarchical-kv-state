#!/bin/bash
#SBATCH --job-name=r-h200-brerun
#SBATCH --partition=h200q
#SBATCH --account=ChihChungHsu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=08:00:00
#SBATCH -o slurm-%x-%j.out
# =============================================================================
# r_h200_brerun.sh — 重跑「和我們自己的 F2 job 同時跑」的 B 實驗（2026-10-16，使用者：「照重疊重跑」）
#
#   sbatch --dependency=afterany:<r-h200-rest>,afterany:<r-h200-qwen3> code/r_h200_brerun.sh
#   bash code/r_h200_brerun.sh --dry-run | --list ；--from <步驟>
#
# 為什麼：r-h200-rest（job 20554）的 P1 八個 run（09:00–12:02）與 P2-G1-seed0（12:02 起）和我們自己的
# 20555（LongAlpaca＋F2 probe）、20570（F2 probe＋sweep，到 12:38）跑在同一個節點；F2 probe 有不限速的 pinned H2D（約 53 GiB/s）。
# 主設定 cpu_frac 0.5 合併 rep 0–5 後 S5 快 8.3%，全部來自 r36 的 rep 4–5（RUNLOG_H200 的 P1 一節）。
# 選哪些重跑**照重疊、不照結果**：時間上和 20555／20570 重疊的 B run 全部重跑（只重跑 r36 會變成挑結果）。
# R3-B-main（06:39–07:46，只和 20417 自己）不重跑。
#
# 指令和 r_h200_rest.sh 的 P1、P2-G1-seed0 逐字相同（那支正在 20554 裡跑，不能改，所以另寫）。只換輸出目錄：
#   M7_RES＝results/m7_write_policy_h200_rerun（B 的 CSV 是附加寫入，寫回原檔會和舊的 rep 混在一起）
#   M7_F_CSV＝results/m7_write_policy_h200/calib_c1.csv（同一份 H200 R2 的 C1）；限速層照 env_h200.sh（MI300X 的 tier_params.json）
# 開跑前檢查：節點上除了自己之外，沒有我們（$USER）的別的 job 在跑；有就停（這次要量的就是「沒有自己人干擾」）。
# 別的使用者的 job 照樣看不到（UNOBSERVABLE_HOST），和 R3-B-main 時一樣。任何一步失敗 → 停（規則 2）。
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
else echo "[r_h200_brerun] 找不到 repo" >&2; exit 2
fi
export REPO
[ "$DRY" = 1 ] && export TIARA_SKIP_CONDA=1
# shellcheck disable=SC1091
source "$REPO/code/env_h200.sh"
cd "$REPO"
export M7_RES=$REPO/results/m7_write_policy_h200_rerun
export M7_F_CSV=$REPO/results/m7_write_policy_h200/calib_c1.csv
P=code/m7_write_policy.py

if [ "$DRY" = 0 ]; then
  [ -n "${SLURM_JOB_ID:-}" ] || { echo "[r_h200_brerun] 不在 Slurm job 裡" >&2; exit 2; }
  awk -v d="$M7_DISK" '$3==d{f=1} END{exit !f}' /proc/diskstats || { echo "[r_h200_brerun] M7_DISK=$M7_DISK 不在 /proc/diskstats → 停" >&2; exit 2; }
  [ -f "$M7_F_CSV" ] || { echo "[r_h200_brerun] 找不到 $M7_F_CSV" >&2; exit 2; }
  python -c 'import sys; sys.path.insert(0, "code"); from gpu_guard import smi_index; print("[r_h200_brerun] smi_index(0) =", smi_index(0))' \
    || { echo "[r_h200_brerun] cuda:0 對不到 nvidia-smi 的卡 → 停" >&2; exit 2; }
  others=$(squeue -h -w "${SLURMD_NODENAME:-$(hostname -s)}" -u "$USER" -t R -o '%i %j' | awk -v me="$SLURM_JOB_ID" '$1!=me')
  if [ -n "$others" ]; then
    echo "[r_h200_brerun] 節點上還有我們自己的 job 在跑 → 停（這次要量的是沒有自己人干擾）：" >&2; echo "$others" >&2; exit 2
  fi
  echo "[r_h200_brerun] 同節點的 job（開跑時）："; squeue -h -w "${SLURMD_NODENAME:-$(hostname -s)}" -o '%i %u %a %b %T %M'
  mkdir -p "$M7_RES"
  echo "[r_h200_brerun] $(date -Is) job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-?} M7_RES=$M7_RES M7_F_CSV=$M7_F_CSV git=$(git rev-parse --short HEAD)"
fi

STARTED=1; [ -n "$FROM" ] && STARTED=0
step() {
  local name=$1; shift
  if [ "$STARTED" = 0 ]; then if [ "$name" = "$FROM" ]; then STARTED=1; else return 0; fi; fi
  if [ "$LIST" = 1 ]; then echo "$name"; return 0; fi
  if [ "$DRY" = 1 ]; then printf '[%s] ' "$name"; printf '%q ' "$@"; echo; return 0; fi
  echo "=== $(date -Is) $name"
  "$@"; local rc=$?
  if [ $rc -ne 0 ]; then echo "=== $(date -Is) $name 失敗（exit $rc）→ 停下、回報。接著跑：sbatch code/r_h200_brerun.sh --from $name" >&2; exit $rc; fi
}
# ===== 和 r_h200_rest.sh 的 P1、P2-G1-seed0 逐字相同（run 的短名加 -rr）=====
step P1-B-rep36   m7run m7h-b-nfs-gap0-r36-rr env M7_IO_MODEL=share python $P b --reps 3 --rep-start 3 --ssd-dev nfs --gap 0 --wl-seed 0 --cpu-fracs 0.25 0.5 --strategies S1 S2b S4 S4+ S4+P S5 S5s
step P1-B-seed1   m7run m7h-b-nfs-seed1-rr    env M7_IO_MODEL=share python $P b --reps 3 --ssd-dev nfs --gap 0 --wl-seed 1 --cpu-fracs 0.25 0.5 --strategies S1 S4 S4+ S4+P S5 S5s
step P1-B-cpuvllm m7run m7h-b-nfs-cpuvllm-rr  env M7_IO_MODEL=share M7_CPU_GIBPS=3.69 python $P b --reps 3 --ssd-dev nfs --gap 0 --wl-seed 0 --cpu-fracs 0.25 0.5 --strategies S1 S4 S4+ S4+P S5 S5s
step P1-B-local   m7run m7h-b-local-gap0-rr   env M7_IO_MODEL=share python $P b --reps 3 --ssd-dev local --gap 0 --wl-seed 0 --cpu-fracs 0.25 0.5 --strategies S1 S4 S4+ S4+P S5 S5s
step P1-B-fifo    m7run m7h-b-nfs-fifo-rr     env M7_IO_MODEL=fifo python $P b --reps 3 --ssd-dev nfs --gap 0 --wl-seed 0 --cpu-fracs 0.25 0.5 --strategies S1 S4 S4+ S4+P S5 S5s
step P1-B-seed2   m7run m7h-b-nfs-seed2-rr    env M7_IO_MODEL=share python $P b --reps 3 --ssd-dev nfs --gap 0 --wl-seed 2 --cpu-fracs 0.25 0.5 --strategies S1 S4 S4+ S4+P S5 S5s
step P1-B-gap2    m7run m7h-b-nfs-gap2-rr     env M7_IO_MODEL=share python $P b --reps 3 --ssd-dev nfs --gap 2 --wl-seed 0 --cpu-fracs 0.25 0.5 --strategies S1 S4 S4+ S4+P S5 S5s
step P1-B-ssd50   m7run m7h-b-nfs-ssd50-rr    env M7_IO_MODEL=share python $P b --reps 3 --ssd-dev nfs --gap 0 --wl-seed 0 --cpu-fracs 0.25 0.5 --ssd-frac 0.5 --strategies S1 S4 S4+ S4+P S5 S5s
step P2-G1-seed0  m7run m7h-b2-g1-doc-hold-cpu3.69-s0-rr env M7_IO_MODEL=share M7_CPU_GIBPS=3.69 python $P b --b2 --workload doc --release hold --ssd-dev nfs --wl-seed 0 --strategies S1 S2b S4+ S4L S4B S5 S5L S5c --cpu-fracs 0.25 0.5 --reps 3 --verify
[ "$DRY" = 0 ] && echo "[r_h200_brerun] $(date -Is) 全部完成 → $M7_RES"
exit 0
