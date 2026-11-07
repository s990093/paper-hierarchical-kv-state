#!/bin/bash
#SBATCH --job-name=icpe-next-w6q
#SBATCH --partition=h200q
#SBATCH --account=ChihChungHsu
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-socket=2
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-task=16
#SBATCH --time=1-12:00:00
#SBATCH -o slurm-%x-%j.out
# =============================================================================
# next_a_w6q.sh — W6a、W6b 在同一個 job 的 4 張卡上同時跑（PREREG 偏離 21；使用者 2026-11-07 22:2x：「11/8 當天要把 GPU 任務跑掉、
# 一定要並行」，AskUserQuestion 選「4 張（每個 NUMA 節點 2 張）」）。超出 T2 量過的範圍（T2 只有：兩個 job、卡在不同 NUMA 節點），
# 每個 run 當時幾條 lane 同時在跑由 next_a_corun.py classify 事後算，只報告。
#
#   sbatch [--dependency=afterok:<s3-fetch>] code/icpe/next_a_w6q.sh
#   bash code/icpe/next_a_w6q.sh --dry-run
#
# 一個 job 4 張卡：--gres=gpu:4 --ntasks-per-socket=2 --gres-flags=enforce-binding（h200q 的 job_submit 改成 4 task × 16 CPU、4×170G）。
# Slurm 的 GPU 只記 S:0-1（不知道每張卡在哪個 socket），--gpus-per-socket 沒有作用（24747 拿到 NUMA 0 三張、1 一張）；
# 改成每個 socket 2 個 task＋enforce-binding 之後，srun 試兩次都拿到兩個 NUMA 節點各 2 張（偏離 21 補記）。
# 開跑時 next_a_corun.py plan --n 4 核對：兩個 NUMA 節點各 2 張；不是、或查不到 → exit 2、一個 run 都不跑。
# 卡照 (NUMA, PCI) 排序，lane 0..3：
#   0 w6a      next_a_w6.sh  --part w6a --from W6A-256-LRU-r0                  → results/icpe_next/a/w6a/{gpu_requests,overlap}.csv
#   1 w6b-r01  next_a_w6b.sh --reps 6 --rep-from 0 --rep-to 1 --tag r01（含冒煙） → results/icpe_next/a/w6b/*_r01.csv
#   2 w6b-r23  next_a_w6b.sh --reps 6 --rep-from 2 --rep-to 3 --tag r23 --no-smoke
#   3 w6b-r45  next_a_w6b.sh --reps 6 --rep-from 4 --rep-to 5 --tag r45 --no-smoke
# 每條 lane taskset 到自己那張卡 NUMA 節點上的 16 個 CPU。兩支 runner 照舊：每個 run 前等節點上沒有我們自己的「別的 job」（同一個 job
# 的其他 lane 不算）、跑完 sacct 查重疊、GpuWatcher 看自己那張卡、污染或重疊重跑最多 3 次。W6b 三份的 csv 跑完後合併（偏離 21）。
# =============================================================================
set -uo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
if [ -n "${REPO:-}" ] && [ -f "$REPO/code/env_h200.sh" ]; then :
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/code/env_h200.sh" ]; then REPO=$SLURM_SUBMIT_DIR
elif [ -f "$_here/../env_h200.sh" ]; then REPO=$(cd "$_here/../.." && pwd)
else echo "[next_a_w6q] 找不到 repo（code/env_h200.sh）" >&2; exit 2
fi
export REPO
if [ -z "${NEXT_A_W6Q_REEXEC:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  export NEXT_A_W6Q_REEXEC=1
  echo "[next_a_w6q] 改執行 repo 裡的版本：$REPO/code/icpe/next_a_w6q.sh（git $(git -C "$REPO" rev-parse --short HEAD)）"
  exec bash "$REPO/code/icpe/next_a_w6q.sh" "$@"
fi
DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "不認得的參數：$1" >&2; exit 2 ;;
  esac
done
cd "$REPO"
D=$REPO/results/icpe_next/a/w6ab
SUBS=(w6a w6b-r01 w6b-r23 w6b-r45)
RUNNERS=(next_a_w6.sh next_a_w6b.sh next_a_w6b.sh next_a_w6b.sh)
ARGS=("--part w6a --from W6A-256-LRU-r0"
      "--reps 6 --rep-from 0 --rep-to 1 --tag r01"
      "--reps 6 --rep-from 2 --rep-to 3 --tag r23 --no-smoke"
      "--reps 6 --rep-from 4 --rep-to 5 --tag r45 --no-smoke")
OVLS=(results/icpe_next/a/w6a/overlap.csv results/icpe_next/a/w6b/overlap_r01.csv results/icpe_next/a/w6b/overlap_r23.csv results/icpe_next/a/w6b/overlap_r45.csv)

if [ "$DRY" = 1 ]; then
  for i in 0 1 2 3; do echo "[next_a_w6q] dry-run lane $i ${SUBS[$i]}：${RUNNERS[$i]} ${ARGS[$i]}（overlap → ${OVLS[$i]}）"; done
  python3 code/icpe/next_a_corun.py selftest || exit 1
  exit 0
fi
[ -n "${SLURM_JOB_ID:-}" ] || { echo "[next_a_w6q] 不在 Slurm job 裡（head node 不量時間）" >&2; exit 2; }
mkdir -p "$D"
echo "[next_a_w6q] $(date -Is) job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-?} git=$(git rev-parse --short HEAD) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-UNSET} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
PLAN=$(python3 code/icpe/next_a_corun.py plan --n 4 --out-dir "$D") \
  || { echo "[next_a_w6q] 4 張卡不是兩個 NUMA 節點各 2 張（或查不到）→ 不平行、一個 run 都不跑，停下回報（exit 2）。看 $D/corun_plan_$SLURM_JOB_ID.json" >&2; exit 2; }
eval "$PLAN"
GS=("$G0" "$G1" "$G2" "$G3"); BS=("$B0" "$B1" "$B2" "$B3"); NS=("$N0" "$N1" "$N2" "$N3"); CS=("$C0" "$C1" "$C2" "$C3")
for i in 0 1 2 3; do echo "[next_a_w6q] lane $i ${SUBS[$i]}：CUDA_VISIBLE_DEVICES=${GS[$i]} PCI ${BS[$i]} NUMA ${NS[$i]} CPU ${CS[$i]}"; done
echo "[next_a_w6q] CPU 分法 $SPLIT"
CR="$D/corun.csv"
[ -f "$CR" ] || echo "job,sub,cvd,bus,numa,cpus,ovl,t_start,t_end,exit" > "$CR"
run_lane() {   # run_lane <i>
  local i=$1 t0 t1 rc log="$REPO/slurm-icpe-next-w6q-$SLURM_JOB_ID-${SUBS[$1]}.out"
  t0=$(date -Is)
  # shellcheck disable=SC2086
  ( export CUDA_VISIBLE_DEVICES="${GS[$i]}" NEXT_A_W6_REEXEC=1 NEXT_A_W6B_REEXEC=1
    exec taskset -c "${CS[$i]}" bash "$REPO/code/icpe/${RUNNERS[$i]}" ${ARGS[$i]} ) > "$log" 2>&1
  rc=$?
  t1=$(date -Is)
  echo "$SLURM_JOB_ID,${SUBS[$i]},${GS[$i]},${BS[$i]},${NS[$i]},\"${CS[$i]}\",${OVLS[$i]},$t0,$t1,$rc" >> "$CR"
  echo "[next_a_w6q] $t1 lane $i ${SUBS[$i]} 結束（exit $rc），log：$log"
  return $rc
}
PIDS=()
for i in 0 1 2 3; do run_lane $i & PIDS+=($!); done
RC=0
for i in 0 1 2 3; do wait "${PIDS[$i]}"; r=$?; [ $r -ne 0 ] && RC=$r; done
echo "[next_a_w6q] $(date -Is) 全部結束（最後一個非 0 的 exit：$RC）。合併 W6b 的 csv、par 分類在 head node 做（偏離 21）"
exit $RC
