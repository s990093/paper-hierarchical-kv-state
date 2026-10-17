#!/usr/bin/env bash
# =============================================================================
# run3090.sh — CLAUDE.md §4.1 記錄殼的平台 A 版（RTX 3090，acvlab 共用機，沒有 Slurm）（2026-10-17）
#
#   用法：  CUDA_VISIBLE_DEVICES=<i> bash code/run3090.sh <短名> <指令...>
#           TIARA_NO_GPU=1            bash code/run3090.sh <短名> <指令...>    # 只用 CPU（模擬、分析）
#   或先 `source code/env_3090.sh`，再 `CUDA_VISIBLE_DEVICES=<i> m7run <短名> <指令...>`（和平台 B／C 同介面）
#
# 為什麼不沿用 tiara.sbatch：它 source env_h200.sh（會試 conda、把 TIARA_PLATFORM 寫成 C_H200），
#   context.txt 寫 H200/Slurm，且沒有 SLURM_JOB_ID 就拒跑。這裡照它的格式，換成 3090 的環境與防插隊。
#
# 產出：$TIARA_RUNS/<RUN_ID>/{cmd.sh,context.txt,stdout.log,stderr.log,exit_code,gpu_guard.json,context_end.txt}
#       被污染：另有 CONTAMINATED 檔，exit 3（m7_guard_run.py）。開跑前卡就不乾淨：NOT_CLEAN_AT_START，exit 4，不跑。
#
# 防插隊（CLAUDE.md §3「每次量測都要防插隊」）：
#   - 必須明確給 CUDA_VISIBLE_DEVICES=<單一 index>（共用機不設預設；先 python code/gpu_guard.py --idle-gpus）。
#   - 開跑前 gpu_guard --check；不乾淨就不跑（TIARA_ALLOW_DIRTY_START=1 才照跑，給品質量測用：分數不受影響，
#     但時間欄作廢，仍會留 CONTAMINATED）。
#   - 整段指令由 m7_guard_run.py 的 GpuWatcher 包住（TIARA_GUARD=0 可關，例如指令自己已經包了 GpuWatcher）。
#   - TIARA_NO_GPU=1：把 CUDA_VISIBLE_DEVICES 設成空字串（看不到任何卡），不做 GPU 檢查。
#   - TIARA_TWO_GPUS=1：准許 CUDA_VISIBLE_DEVICES=<i>,<j>（2026-10-17，使用者要求「OOM 改雙卡」）。只給**不計時**的
#     檢查用，例如 test_m7_correctness 把 HF 對照模型放第二張卡（M7_TEST_REF_DEVICE=cuda:1）。不是 tensor parallel：
#     受測模型整個在第一張卡。兩張卡開跑前都要乾淨，跑的時候兩張都監看。
# 沒有 run_id 的數字視同不存在。
# =============================================================================
set -uo pipefail
[ $# -ge 2 ] || { echo "用法: CUDA_VISIBLE_DEVICES=<i> $0 <短名> <指令...>   （或 TIARA_NO_GPU=1）" >&2; exit 2; }
SHORT=$1; shift

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO="${REPO:-$(cd "$_here/.." && pwd)}"
# shellcheck disable=SC1091
source "$REPO/code/env_3090.sh"

# ---- 卡 ----------------------------------------------------------------------
GPU_IDX=""
if [ "${TIARA_NO_GPU:-0}" = 1 ]; then
  export CUDA_VISIBLE_DEVICES=""
else
  if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "[run3090] 沒設 CUDA_VISIBLE_DEVICES。共用機：先 python code/gpu_guard.py --idle-gpus 挑一張空卡；只用 CPU 就 TIARA_NO_GPU=1" >&2; exit 2
  fi
  if [ "${TIARA_TWO_GPUS:-0}" = 1 ]; then
    if ! [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+,[0-9]+$ ]]; then
      echo "[run3090] TIARA_TWO_GPUS=1 要給兩個 index（例如 2,4），現在是 '$CUDA_VISIBLE_DEVICES'" >&2; exit 2
    fi
  elif ! [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]; then
    echo "[run3090] CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'：一個 job 綁一張卡（不准 tensor parallel，CLAUDE.md §3），請給單一 index（不計時的雙卡檢查用 TIARA_TWO_GPUS=1）" >&2; exit 2
  fi
  GPU_IDX=$CUDA_VISIBLE_DEVICES
fi

# m7run 先算好 RUN_ID 傳進來：只在它屬於這個短名、目錄還不存在時沿用（runsh 2026-10-10 的修正）
if [[ "${RUN_ID:-}" == *"-${SHORT}" && ! -e "$TIARA_RUNS/${RUN_ID}" ]]; then :; else
  RUN_ID="$(date +%Y%m%d-%H%M%S)-${SHORT}"
fi
export RUN_ID
RUN="$TIARA_RUNS/$RUN_ID"
mkdir -p "$RUN" || { echo "[run3090] 建不了 $RUN" >&2; exit 2; }

# 1. 指令（%q 保留每個參數的邊界）
{ printf '%q ' "$@"; echo; } > "$RUN/cmd.sh"

# 2. 環境
_smi_state() {
  echo "--- nvidia-smi ---"
  nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.used,memory.total,utilization.gpu,temperature.gpu,driver_version,pcie.link.gen.current,pcie.link.width.current --format=csv 2>&1 || echo "nvidia-smi FAILED"
  echo "--- compute apps ---"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv 2>&1 || echo "nvidia-smi FAILED"
}
{
  date -Is
  echo "platform: $TIARA_PLATFORM"
  echo "host: $(hostname)"
  echo "short=$SHORT run_id=$RUN_ID cuda_visible='${CUDA_VISIBLE_DEVICES}' gpu_idx=${GPU_IDX:-NONE} no_gpu=${TIARA_NO_GPU:-0} two_gpus=${TIARA_TWO_GPUS:-0} guard=${TIARA_GUARD:-1}"
  _smi_state
  echo "--- loadavg / mem ---"
  cat /proc/loadavg; free -g | head -2
  echo "--- repo ---"
  git -C "$REPO" rev-parse HEAD 2>&1
  git -C "$REPO" rev-parse --abbrev-ref HEAD 2>&1
  echo "dirty files: $(git -C "$REPO" status --porcelain 2>/dev/null | wc -l)"
  git -C "$REPO" status --porcelain 2>/dev/null | head -20
  echo "--- python ---"
  echo "venv=${VIRTUAL_ENV:-NONE} python=$(command -v python || echo NONE)"
  python -c "import sys, torch; print(sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda)" 2>&1 | tail -1
  echo "--- env (TIARA/M7/M10/HF/CUDA/caches) ---"
  env | grep -E '^(TIARA_|M7_|M10_|HF_|HUGGINGFACE_|TRANSFORMERS_CACHE|CUDA_|REPO=|PAPER_HKV|TRITON_CACHE_DIR|VLLM_CACHE_ROOT|TORCH_HOME|UV_)' | sort
} > "$RUN/context.txt" 2>&1

# 3. 開跑前的卡
if [ -n "$GPU_IDX" ]; then
  pre=0
  : > "$RUN/gpu_check_start.txt"
  for _g in ${GPU_IDX//,/ }; do
    echo "--- gpu $_g ---" >> "$RUN/gpu_check_start.txt"
    python "$REPO/code/gpu_guard.py" --check "$_g" >> "$RUN/gpu_check_start.txt" 2>&1 || pre=$?
  done
  if [ "$pre" -ne 0 ]; then
    if [ "${TIARA_ALLOW_DIRTY_START:-0}" = 1 ]; then
      echo "[run3090] GPU $GPU_IDX 開跑前不乾淨，TIARA_ALLOW_DIRTY_START=1 照跑（時間欄作廢）" >&2
    else
      cp "$RUN/gpu_check_start.txt" "$RUN/NOT_CLEAN_AT_START"
      echo "4" > "$RUN/exit_code"
      echo "[run3090] GPU $GPU_IDX 開跑前不乾淨（或查不到，rc=$pre），不跑：" >&2
      cat "$RUN/gpu_check_start.txt" >&2
      echo "[run3090] RUN_ID=$RUN_ID dir=$RUN" >&2
      exit 4
    fi
  fi
fi

# 4. 跑，全部導進去
echo "[run3090] RUN_ID=$RUN_ID"
echo "[run3090] dir=$RUN"
cd "$REPO" || exit 2
if [ -n "$GPU_IDX" ] && [ "${TIARA_GUARD:-1}" != 0 ]; then
  RUN_ID=$RUN_ID RUN_DIR=$RUN python "$REPO/code/m7_guard_run.py" bash "$RUN/cmd.sh" > "$RUN/stdout.log" 2> "$RUN/stderr.log"
else
  RUN_ID=$RUN_ID RUN_DIR=$RUN bash "$RUN/cmd.sh" > "$RUN/stdout.log" 2> "$RUN/stderr.log"
fi
rc=$?
echo "$rc" > "$RUN/exit_code"
{ date -Is; _smi_state; } > "$RUN/context_end.txt" 2>&1
echo "[run3090] exit=$rc"
[ -e "$RUN/CONTAMINATED" ] && echo "[run3090] CONTAMINATED：GPU $GPU_IDX 期間有外來 process（見 gpu_guard.json）。時間／VRAM 數字不得寫進 results/，重量。" >&2
if [ "$rc" -ne 0 ]; then
  echo "[run3090] ---- stderr 末 40 行 ----" >&2
  tail -40 "$RUN/stderr.log" >&2
  echo "[run3090] 指令失敗。照 CLAUDE.md §1 規則 2：記錄、停下、回報。不要換方法硬幹。" >&2
fi
exit "$rc"
