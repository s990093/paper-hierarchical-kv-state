#!/bin/bash
#SBATCH --partition=h200q
#SBATCH --account=ChihChungHsu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=12:00:00
#SBATCH -o slurm-%x-%j.out
# =============================================================================
# tiara.sbatch — CLAUDE.md §4.1 記錄殼的 Slurm 版（平台 C，H200；H200_SLURM.md §5.2）
#
#   正式：  sbatch --job-name=<短名> code/tiara.sbatch <指令...>
#           例：sbatch --job-name=r1-correct-llama code/tiara.sbatch python code/test_m7_correctness.py results/m7_write_policy_h200/correctness.json
#   一個 job 裡依序跑多步（§R2/§R3 一個 job）：在 job 裡 `source code/env_h200.sh` 後用 m7run <短名> <指令...>，
#           m7run 會用 `bash code/tiara.sbatch` 把每一步各包成一個 run 目錄（#SBATCH 那幾行此時只是註解）。
#
# 產出：$TIARA_RUNS/<RUN_ID>/{cmd.sh,context.txt,stdout.log,stderr.log,exit_code,cotenants_end.txt}
# 沒有 run_id 的數字視同不存在。Slurm 自己的輸出（含 env_h200.sh 的錯誤）在送出目錄的 slurm-<短名>-<job>.out（*.out 已 gitignore）。
# 48 小時是硬上限（--time 依工作改）；長的工作要能從中斷處接著跑。
# =============================================================================
set -uo pipefail
[ $# -ge 1 ] || { echo "用法: sbatch --job-name=<短名> $0 <指令...>" >&2; exit 2; }

# ---- 找 repo：sbatch 會把這支複製到 spool，$0／BASH_SOURCE 不是 repo 裡的路徑 ----------------
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
if [ -n "${REPO:-}" ] && [ -f "$REPO/code/env_h200.sh" ]; then :
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/code/env_h200.sh" ]; then REPO=$SLURM_SUBMIT_DIR
elif [ -f "$_here/env_h200.sh" ]; then REPO=$(cd "$_here/.." && pwd)
elif [ -f "$HOME/paper-hierarchical-kv-state/code/env_h200.sh" ]; then REPO=$HOME/paper-hierarchical-kv-state
else echo "[tiara] 找不到 repo（code/env_h200.sh）。請在 repo 根目錄 sbatch，或先 export REPO" >&2; exit 2
fi
export REPO
# shellcheck disable=SC1091
source "$REPO/code/env_h200.sh"

# head node 不跑會產生數字的指令（H200_SLURM §5.1）。TIARA_ALLOW_NO_SLURM=1 只給在別的機器上測這支殼用
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${TIARA_ALLOW_NO_SLURM:-0}" != 1 ]; then
  echo "[tiara] 不在 Slurm job 裡（沒有 SLURM_JOB_ID）。請用 sbatch，或在 srun 的 job 裡跑" >&2; exit 2
fi

SHORT="${TIARA_SHORT:-${SLURM_JOB_NAME:-adhoc}}"
[ "$SHORT" = "tiara.sbatch" ] && { echo "[tiara] 沒給 --job-name，短名用 adhoc" >&2; SHORT=adhoc; }
# m7run 先算好 RUN_ID 傳進來：只在它屬於這個短名、目錄還不存在時沿用（runsh 2026-10-10 的修正）
if [[ "${RUN_ID:-}" == *"-${SHORT}" && ! -e "$TIARA_RUNS/${RUN_ID}" ]]; then :; else
  RUN_ID="$(date +%Y%m%d-%H%M%S)-${SHORT}"
fi
RUN="$TIARA_RUNS/$RUN_ID"
mkdir -p "$RUN" || { echo "[tiara] 建不了 $RUN" >&2; exit 2; }

# 1. 指令（%q 保留每個參數的邊界：含空白、引號的參數不會被拆壞）
{ printf '%q ' "$@"; echo; } > "$RUN/cmd.sh"

# 2. 環境
_cotenants() {   # 同節點上有哪些 job（H200_SLURM §5.3：鄰居只能部分觀測，至少記下來）
  if [ -n "${SLURMD_NODENAME:-}" ]; then squeue -w "$SLURMD_NODENAME" -h -o '%i %u %a %b %C %m %T %M' 2>&1
  else echo "NOT_IN_SLURM"; fi
}
{
  date -Is
  echo "platform: C_H200_Slurm"
  echo "host: $(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID:-NONE} job_name=${SLURM_JOB_NAME:-} node=${SLURMD_NODENAME:-} cuda_visible=${CUDA_VISIBLE_DEVICES:-UNSET} job_gpus=${SLURM_JOB_GPUS:-} step_id=${SLURM_STEP_ID:-}"
  echo "--- nvidia-smi ---"
  nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.used,memory.total,utilization.gpu,driver_version,pcie.link.gen.current,pcie.link.width.current --format=csv 2>&1 || echo "nvidia-smi FAILED"
  echo "--- compute apps ---"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv 2>&1 || echo "nvidia-smi FAILED"
  echo "--- repo ---"
  git -C "$REPO" rev-parse HEAD 2>&1
  echo "dirty files: $(git -C "$REPO" status --porcelain 2>/dev/null | wc -l)"
  git -C "$REPO" status --porcelain 2>/dev/null | head -20
  echo "--- python ---"
  echo "conda_env=${CONDA_DEFAULT_ENV:-NONE} python=$(command -v python || echo NONE)"
  python -c "import sys, torch; print(sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda)" 2>&1 | tail -1
  echo "--- env (TIARA/M7/HF/CUDA) ---"
  env | grep -E '^(TIARA_|M7_|HF_HOME|CUDA_|REPO=|SLURM_JOB_)' | sort
  echo "--- scontrol show job ---"
  if [ -n "${SLURM_JOB_ID:-}" ]; then scontrol show job "$SLURM_JOB_ID" 2>&1; else echo "NOT_IN_SLURM"; fi
  echo "--- 同節點的 job（squeue -w）---"
  _cotenants
} > "$RUN/context.txt" 2>&1

# 3. 跑，全部導進去
echo "[tiara] RUN_ID=$RUN_ID"
echo "[tiara] dir=$RUN"
cd "$REPO"
RUN_ID=$RUN_ID RUN_DIR=$RUN bash "$RUN/cmd.sh" > "$RUN/stdout.log" 2> "$RUN/stderr.log"
rc=$?
echo "$rc" > "$RUN/exit_code"
{ date -Is; _cotenants; } > "$RUN/cotenants_end.txt" 2>&1
echo "[tiara] exit=$rc"
if [ "$rc" -ne 0 ]; then
  echo "[tiara] ---- stderr 末 40 行 ----" >&2
  tail -40 "$RUN/stderr.log" >&2
  echo "[tiara] 指令失敗。照 CLAUDE.md §1 規則 2：記錄、停下、回報。不要換方法硬幹。" >&2
fi
exit "$rc"
