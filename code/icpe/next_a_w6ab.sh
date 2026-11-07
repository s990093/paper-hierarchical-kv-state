#!/bin/bash
#SBATCH --job-name=icpe-next-w6ab
#SBATCH --partition=h200q
#SBATCH --account=ChihChungHsu
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-socket=1
#SBATCH --sockets-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH -o slurm-%x-%j.out
# =============================================================================
# next_a_w6ab.sh — W6a 和 W6b 在同一個 job 的兩張卡上同時跑（PREREG 偏離 20；使用者 2026-11-07 22:0x「好可以」）。
# 依據：T2 的 CAL-OK（docs/h200/icpe/T2_parallel_calibration.md）：兩個縮小版 GPU 重播 job、卡在不同 NUMA 節點，同時跑和單獨跑
# 的回來請求平均 TTFT 差 +0.34%（95% CI −0.18～+0.77%）。同一個 NUMA、三個以上：沒量過 → 不平行。
#
#   sbatch --dependency=afterok:<s3-fetch> code/icpe/next_a_w6ab.sh --w6b-reps 6 [--w6a-from W6A-256-LRU-r0] [--w6b-from <步驟>]
#   bash code/icpe/next_a_w6ab.sh --w6b-reps 6 --dry-run
#
# 一個 job 兩張卡：--gres=gpu:2 --gpus-per-socket=1 --sockets-per-node=2（h200q 的 job_submit 會改成 32 CPU、2×170G）。
# 開跑時 next_a_corun.py plan 核對兩張卡的 NUMA 節點（sysfs）；不是兩個不同的 NUMA 節點、或查不到 → exit 2、一個 run 都不跑。
# W6a（next_a_w6.sh --part w6a）在第一張卡、W6b（next_a_w6b.sh）在第二張卡，各自 taskset 到自己那張卡 NUMA 節點上的 16 個 CPU
# （Slurm 給的 CPU 不是兩邊各一半時照編號對半分，照實記在 corun_plan_<job>.json）。兩支 runner 不改：每個 run 前照舊等節點上沒有
# 我們自己的「別的 job」（t1g.py wait-clear 用同一個 job id，所以同 job 的另一邊不算），跑完照舊用 sacct 查重疊，GpuWatcher 照舊。
# 兩邊的開始／結束寫 results/icpe_next/a/w6ab/corun.csv；每個 run 是 par／solo／partial 由 next_a_corun.py classify 事後在 head node 算（只報告）。
# =============================================================================
set -uo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
if [ -n "${REPO:-}" ] && [ -f "$REPO/code/env_h200.sh" ]; then :
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/code/env_h200.sh" ]; then REPO=$SLURM_SUBMIT_DIR
elif [ -f "$_here/../env_h200.sh" ]; then REPO=$(cd "$_here/../.." && pwd)
else echo "[next_a_w6ab] 找不到 repo（code/env_h200.sh）" >&2; exit 2
fi
export REPO
if [ -z "${NEXT_A_W6AB_REEXEC:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  export NEXT_A_W6AB_REEXEC=1
  echo "[next_a_w6ab] 改執行 repo 裡的版本：$REPO/code/icpe/next_a_w6ab.sh（git $(git -C "$REPO" rev-parse --short HEAD)）"
  exec bash "$REPO/code/icpe/next_a_w6ab.sh" "$@"
fi
W6A_FROM="W6A-256-LRU-r0"; W6B_REPS=""; W6B_FROM=""; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --w6a-from) W6A_FROM="$2"; shift 2 ;;
    --w6b-reps) W6B_REPS="$2"; shift 2 ;;
    --w6b-from) W6B_FROM="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "不認得的參數：$1" >&2; exit 2 ;;
  esac
done
[[ "$W6B_REPS" =~ ^[1-9][0-9]*$ ]] || { echo "要 --w6b-reps <正整數>（PREREG 偏離 16：6）" >&2; exit 2; }
cd "$REPO"
D=$REPO/results/icpe_next/a/w6ab
A_ARGS=(--part w6a); [ -n "$W6A_FROM" ] && A_ARGS+=(--from "$W6A_FROM")
B_ARGS=(--reps "$W6B_REPS"); [ -n "$W6B_FROM" ] && B_ARGS+=(--from "$W6B_FROM")

if [ "$DRY" = 1 ]; then
  echo "[next_a_w6ab] dry-run：卡 A → next_a_w6.sh ${A_ARGS[*]}；卡 B → next_a_w6b.sh ${B_ARGS[*]}"
  python3 code/icpe/next_a_corun.py selftest || exit 1
  exit 0
fi
[ -n "${SLURM_JOB_ID:-}" ] || { echo "[next_a_w6ab] 不在 Slurm job 裡（head node 不量時間）" >&2; exit 2; }
mkdir -p "$D"
echo "[next_a_w6ab] $(date -Is) job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-?} git=$(git rev-parse --short HEAD) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-UNSET} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
PLAN=$(python3 code/icpe/next_a_corun.py plan --out-dir "$D") \
  || { echo "[next_a_w6ab] 兩張卡不在兩個不同的 NUMA 節點（或查不到）→ 不平行、一個 run 都不跑，停下回報（exit 2）。看 $D/corun_plan_$SLURM_JOB_ID.json" >&2; exit 2; }
eval "$PLAN"
echo "[next_a_w6ab] 卡 A（W6a）：CUDA_VISIBLE_DEVICES=$GA PCI $BA NUMA $NA CPU $CA"
echo "[next_a_w6ab] 卡 B（W6b）：CUDA_VISIBLE_DEVICES=$GB PCI $BB NUMA $NB CPU $CB（CPU 分法 $SPLIT）"
CR="$D/corun.csv"
[ -f "$CR" ] || echo "job,sub,cvd,bus,numa,cpus,t_start,t_end,exit" > "$CR"
LOGA="$REPO/slurm-icpe-next-w6ab-$SLURM_JOB_ID-w6a.out"
LOGB="$REPO/slurm-icpe-next-w6ab-$SLURM_JOB_ID-w6b.out"
run_sub() {   # run_sub <sub> <cvd> <bus> <numa> <cpus> <log> <runner> <args...>
  local sub=$1 cvd=$2 bus=$3 numa=$4 cpus=$5 log=$6 runner=$7; shift 7
  local t0 t1 rc
  t0=$(date -Is)
  ( export CUDA_VISIBLE_DEVICES="$cvd" NEXT_A_W6_REEXEC=1 NEXT_A_W6B_REEXEC=1
    exec taskset -c "$cpus" bash "$REPO/code/icpe/$runner" "$@" ) > "$log" 2>&1
  rc=$?
  t1=$(date -Is)
  echo "$SLURM_JOB_ID,$sub,$cvd,$bus,$numa,\"$cpus\",$t0,$t1,$rc" >> "$CR"
  echo "[next_a_w6ab] $t1 $sub 結束（exit $rc），log：$log"
  return $rc
}
run_sub w6a "$GA" "$BA" "$NA" "$CA" "$LOGA" next_a_w6.sh "${A_ARGS[@]}" & PA=$!
run_sub w6b "$GB" "$BB" "$NB" "$CB" "$LOGB" next_a_w6b.sh "${B_ARGS[@]}" & PB=$!
wait $PA; RA=$?
wait $PB; RB=$?
echo "[next_a_w6ab] $(date -Is) W6a exit $RA、W6b exit $RB（par／solo 的分類在 head node 跑 next_a_corun.py classify，只報告）"
[ $RA -ne 0 ] && exit $RA
exit $RB
