# source 這支：平台 C（NVIDIA H200，學校 Slurm 叢集）的環境。比照 m7_env.sh（平台 B）。
#
#   source code/env_h200.sh            # 安靜
#   source code/env_h200.sh --print    # 另外印出所有路徑、檢查存不存在（R0 時用）
#
# 〔推論，未在叢集上試過〕標「R0 確認」的值，要在 03_plan_h200 §R0／H200_SLURM §3 量完後回填。
# 任何值都可以在 source 之前先 export 蓋掉（全部用 ${VAR:-預設}）。
# 不設 CUDA_VISIBLE_DEVICES：交給 Slurm 依 --gres 設。
# 這支會被 source，所以不用 set -e、不 exit。

export TIARA_PLATFORM=C_H200

# ---- repo（這支檔案的上一層）------------------------------------------------
export REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# ---- conda 環境 tiara（Python 3.12、vLLM 0.28.0；H200_SLURM §4.1）----------------
# batch job 的 shell 不一定有 module 指令：先試 module，再試 wiki 上的 conda 位置（/cm/shared/apps/miniconda3）。
# TIARA_SKIP_CONDA=1 只給沒有 conda 的機器測這支檔案用（例如在 MI300X 上 bash -n／--print）。
if [ "${TIARA_SKIP_CONDA:-0}" != 1 ]; then
  if ! command -v conda >/dev/null 2>&1; then
    if ! type module >/dev/null 2>&1; then
      for f in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash; do
        [ -f "$f" ] && . "$f" && break
      done
    fi
    type module >/dev/null 2>&1 && module load miniconda3 >/dev/null 2>&1
    if ! command -v conda >/dev/null 2>&1 && [ -f /cm/shared/apps/miniconda3/etc/profile.d/conda.sh ]; then
      . /cm/shared/apps/miniconda3/etc/profile.d/conda.sh
    fi
  fi
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"          # batch job 裡 conda activate 需要這行
    conda activate "${TIARA_CONDA_ENV:-tiara}" || echo "[env_h200] conda activate ${TIARA_CONDA_ENV:-tiara} 失敗" >&2
  else
    echo "[env_h200] 找不到 conda（module load miniconda3 失敗）。照錄錯誤、停下來回報（CLAUDE.md 規則 2）" >&2
  fi
fi

# ---- 路徑 ----------------------------------------------------------------------
export TIARA_HOME="${TIARA_HOME:-$HOME}"                      # R0 確認：$HOME 是不是 /work/ChihChungHsu/<username>
export HF_HOME="${HF_HOME:-$TIARA_HOME/hf-cache}"
export TIARA_RUNS="${TIARA_RUNS:-$TIARA_HOME/tiara-runs}"     # 每個 run 的原始輸出（不進 git；每天備份，H200_SLURM §5.4）
export TIARA_DATASETS="${TIARA_DATASETS:-$TIARA_HOME/datasets}"
export TIARA_FONTS="${TIARA_FONTS:-$TIARA_HOME/fonts}"        # Noto Sans CJK TC（畫圖用）
export PAPER_HKV_PYLIBS="${PAPER_HKV_PYLIBS:-$TIARA_HOME/pylibs}"   # ruler_tasks 的 wonderwords 等側裝套件
# m10_tasks 的 T4：HotpotQA dev distractor（HF hotpotqa/hotpot_qa，sha256 c20b638c…，見 MI300X 的 datasets/hotpotqa/README.txt）
export M10_HOTPOT="${M10_HOTPOT:-$TIARA_DATASETS/hotpotqa/hotpot_qa_distractor_validation.parquet}"

# 模型：固定 revision（migration_20261013 README §1.1，和 MI300X 同一版）。直接指到 snapshot，不用 *，換版會找不到而報錯
export M7_LLAMA_GLOB="${M7_LLAMA_GLOB:-$HF_HOME/hub/models--unsloth--Llama-3.1-8B-Instruct/snapshots/4699cc75b550f9c6f3173fb80f4703b62d946aa5}"
export M7_QWEN_GLOB="${M7_QWEN_GLOB:-$HF_HOME/hub/models--Qwen--Qwen2.5-7B-Instruct-1M/snapshots/e28526f7bb80e2a9c8af03b831a9af3812f18fba}"
# m7_model.py 讀的是 M7_MODEL_GLOB。預設 Llama（第一階段、§R）；Qwen：M7_MODEL_GLOB=$M7_QWEN_GLOB python ...
export M7_MODEL_GLOB="${M7_MODEL_GLOB:-$M7_LLAMA_GLOB}"

# 第一階段 harness（m7_write_policy.py、m7_c6_vllm.py）
export M7_RES="${M7_RES:-$REPO/results/m7_write_policy_h200}"
# 03 §R2：限速層參數沿用 MI300X 的 tier_params.json（R3 只換 GPU、不換「裝置」）
export M7_TIER_PARAMS="${M7_TIER_PARAMS:-$REPO/results/m7_write_policy_mi300x/tier_params.json}"
# Guard 記錄磁碟 I/O 用的 /proc/diskstats 裝置名。R0 確認（h200_probe.sh 會印建議值）。
# 沒設時故意給一個不存在的名字：Guard 會丟錯，不會靜默記成 0（CLAUDE.md 規則 7）
export M7_DISK="${M7_DISK:-R0_UNSET}"
export M7_VLLM_VENV="${M7_VLLM_VENV:-${CONDA_PREFIX:-}}"       # vLLM 和 harness 同一個 conda 環境
export M7_EXTRA_PATH="${M7_EXTRA_PATH-}"                       # 平台 B 是 /opt/rocm/bin；這裡不需要（空字串）

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID      # 多卡時 torch 和 nvidia-smi 的編號順序一致（gpu_guard.smi_index）

R="$M7_RES"                              # m7_run_phase1*.sh 用 $R 當結果目錄
mkdir -p "$TIARA_RUNS" "$M7_RES" 2>/dev/null

# 記錄殼：和平台 B 的 m7run 同介面。在 job 裡逐步呼叫，每一步一個 run 目錄（tiara.sbatch 當一般 bash 腳本跑）
m7run() { local short=$1; shift; local id="$(date +%Y%m%d-%H%M%S)-$short"; RUN_ID=$id TIARA_SHORT=$short bash "$REPO/code/tiara.sbatch" env RUN_ID=$id "$@"; }

if [ "${1:-}" = "--print" ]; then
  _p() { printf '  %-16s %-90s %s\n' "$1" "$2" "$( [ -e "$2" ] && echo ok || echo MISSING )"; }
  echo "[env_h200] platform=$TIARA_PLATFORM host=$(hostname) slurm_job=${SLURM_JOB_ID:-NONE} cuda_visible=${CUDA_VISIBLE_DEVICES:-UNSET}"
  echo "[env_h200] conda_env=${CONDA_DEFAULT_ENV:-NONE} python=$(command -v python || echo NONE)"
  _p REPO "$REPO"; _p TIARA_HOME "$TIARA_HOME"; _p HF_HOME "$HF_HOME"; _p TIARA_RUNS "$TIARA_RUNS"
  _p TIARA_DATASETS "$TIARA_DATASETS"; _p TIARA_FONTS "$TIARA_FONTS"; _p PAPER_HKV_PYLIBS "$PAPER_HKV_PYLIBS"; _p M10_HOTPOT "$M10_HOTPOT"
  _p M7_LLAMA_GLOB "$M7_LLAMA_GLOB"; _p M7_QWEN_GLOB "$M7_QWEN_GLOB"; _p M7_MODEL_GLOB "$M7_MODEL_GLOB"
  _p M7_RES "$M7_RES"; _p M7_TIER_PARAMS "$M7_TIER_PARAMS"; _p M7_VLLM_VENV "${M7_VLLM_VENV:-<empty>}"
  printf '  %-16s %-90s %s\n' M7_DISK "$M7_DISK" "$(awk -v d="$M7_DISK" '$3==d{f=1} END{print (f?"ok":"MISSING in /proc/diskstats")}' /proc/diskstats)"
  printf '  %-16s %s\n' M7_EXTRA_PATH "'$M7_EXTRA_PATH'"
fi
