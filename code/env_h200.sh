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

# ---- 3090 帶過來的路徑（10/15 在 H200 上踩到）--------------------------------------
# repo 的 .claude/settings.json 是 3090 用的：Claude Code 的 session 會把 HF_HOME、VLLM_CACHE_ROOT、TMPDIR…
# 設到 /ssd7/hungwei/paper-hkv，而 sbatch 預設把送出時的環境帶進 job。這台沒有 /ssd7 → 模型找不到、vLLM 建不了快取。
# 偵測到就清掉、印出清了哪些（不靜默）。真的要指到別處，source 前 export 成不在 /ssd7 的路徑即可。
# 10/19：第三階段（p3-predictor）的 settings.json 改指 /ssd8/hungwei/paper-hkv，另外明設 TIARA_HOME、HF_HUB_CACHE 等
# （3090 的 ~/.bashrc 會蓋過 HF_HOME）。這台也沒有 /ssd8 → 一樣清掉。TIARA_HOME 清掉後照下面原本的預設（$HOME），H200 的路徑不變。
if [ ! -d /ssd7 ] || [ ! -d /ssd8 ]; then
  _cleared=""
  for _v in HF_HOME VLLM_CACHE_ROOT TORCH_HOME TRITON_CACHE_DIR XDG_CACHE_HOME TMPDIR UV_CACHE_DIR UV_PYTHON_INSTALL_DIR PAPER_HKV_BIG \
            TIARA_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE; do
    case "${!_v-}" in
      /ssd7/*) [ -d /ssd7 ] || { _cleared="$_cleared $_v=${!_v}"; unset "$_v"; } ;;
      /ssd8/*) [ -d /ssd8 ] || { _cleared="$_cleared $_v=${!_v}"; unset "$_v"; } ;;
    esac
  done
  # 3090 的 settings 同時開了 hf_transfer；這台的環境沒有裝那個套件（huggingface_hub 會因此報錯）
  if [ -n "$_cleared" ] && [ -n "${HF_HUB_ENABLE_HF_TRANSFER:-}" ]; then
    _cleared="$_cleared HF_HUB_ENABLE_HF_TRANSFER=$HF_HUB_ENABLE_HF_TRANSFER"; unset HF_HUB_ENABLE_HF_TRANSFER
  fi
  [ -n "$_cleared" ] && echo "[env_h200] 清掉 3090 的設定（這台沒有 /ssd7、/ssd8）:$_cleared" >&2
  unset _v _cleared
fi
# Slurm 指令（squeue、scontrol）：tiara.sbatch 記鄰居、gpu_guard 查節點卡數要用。
# Claude Code 的 session 帶著「已 module load slurm」的 LOADEDMODULES，但 PATH 被 settings.json 換掉 → module load 不會補
if ! command -v squeue >/dev/null 2>&1 && [ -x /cm/local/apps/slurm/current/bin/squeue ]; then
  export PATH="/cm/local/apps/slurm/current/bin:$PATH"
  export SLURM_CONF="${SLURM_CONF:-/cm/shared/apps/slurm/etc/slurm/slurm.conf}"
fi

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
export M10_LONGBENCH="${M10_LONGBENCH:-$TIARA_DATASETS/longbench}"   # m10_longbench：data/*.jsonl＋config/（code/h200_setup_data.sh 準備）

# 模型：固定 revision（migration_20261013 README §1.1，和 MI300X 同一版）。直接指到 snapshot，不用 *，換版會找不到而報錯
export M7_LLAMA_GLOB="${M7_LLAMA_GLOB:-$HF_HOME/hub/models--unsloth--Llama-3.1-8B-Instruct/snapshots/4699cc75b550f9c6f3173fb80f4703b62d946aa5}"
export M7_QWEN_GLOB="${M7_QWEN_GLOB:-$HF_HOME/hub/models--Qwen--Qwen2.5-7B-Instruct-1M/snapshots/e28526f7bb80e2a9c8af03b831a9af3812f18fba}"
# m7_model.py 讀的是 M7_MODEL_GLOB。預設 Llama（第一階段、§R）；Qwen：M7_MODEL_GLOB=$M7_QWEN_GLOB python ...
export M7_MODEL_GLOB="${M7_MODEL_GLOB:-$M7_LLAMA_GLOB}"

# 第一階段 harness（m7_write_policy.py、m7_c6_vllm.py）
export M7_RES="${M7_RES:-$REPO/results/m7_write_policy_h200}"
# 03 §R2：限速層參數沿用 MI300X 的 tier_params.json（R3 只換 GPU、不換「裝置」）
export M7_TIER_PARAMS="${M7_TIER_PARAMS:-$REPO/results/m7_write_policy_mi300x/tier_params.json}"
# Guard 記錄磁碟 I/O 用的 /proc/diskstats 裝置名。
# 10/16 R0 確認（run 20261016-061012-h200-probe 的 probe/m7_disk.tsv）：hgx-cn01 上 TIARA_RUNS 在 /dev/md1（ext4）→ md1。
# 換節點時要重做 R0；r_h200.sh 開跑前會檢查它在 /proc/diskstats 裡，不在就停（CLAUDE.md 規則 7）
export M7_DISK="${M7_DISK:-md1}"
export M7_VLLM_VENV="${M7_VLLM_VENV:-${CONDA_PREFIX:-}}"       # vLLM 和 harness 同一個 conda 環境
export M7_EXTRA_PATH="${M7_EXTRA_PATH-}"                       # 平台 B 是 /opt/rocm/bin；這裡不需要（空字串）

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID      # 多卡時 torch 和 nvidia-smi 的編號順序一致（gpu_guard.smi_index）

# vLLM 0.28 的 top-k／top-p 取樣器預設用 FlashInfer，第一次用時要 JIT 編譯（要 nvcc）。
# 10/16 job 20417 的 R2-C6 在這裡失敗：「Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist」
# （run 20261016-074652-m7h-c6-vllm）。環境裡有 pip 的 nvcc 13.4（nvidia-cuda-nvcc），但 CUDA_HOME 指過去也編不過：
# 執行期標頭是 13.0（nvidia-cuda-runtime 13.0.96、torch cu130），cccl 報「CUDA compiler and CUDA toolkit headers are incompatible」。
# 所以改用 vLLM 的 PyTorch 取樣器。量測的請求都是 temperature=0（貪婪，不經過 top-k／top-p），
# 只有暖機請求會用到；MI300X（ROCm）本來就不走 FlashInfer。不設 CUDA_HOME：deep_gemm 等其他 JIT 路徑維持原狀。
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

R="$M7_RES"                              # m7_run_phase1*.sh 用 $R 當結果目錄
# 不在這裡 mkdir（10/15 改：每次 source 都建空目錄，dry-run 會留下 results/m7_write_policy_h200）。
# 目錄由真的要寫的人建：tiara.sbatch 建 run 目錄，r_h200.sh 建 $M7_RES。

# 記錄殼：和平台 B 的 m7run 同介面。在 job 裡逐步呼叫，每一步一個 run 目錄（tiara.sbatch 當一般 bash 腳本跑）
m7run() { local short=$1; shift; local id="$(date +%Y%m%d-%H%M%S)-$short"; RUN_ID=$id TIARA_SHORT=$short bash "$REPO/code/tiara.sbatch" env RUN_ID=$id "$@"; }

if [ "${1:-}" = "--print" ]; then
  _p() { printf '  %-16s %-90s %s\n' "$1" "$2" "$( [ -e "$2" ] && echo ok || echo MISSING )"; }
  echo "[env_h200] platform=$TIARA_PLATFORM host=$(hostname) slurm_job=${SLURM_JOB_ID:-NONE} cuda_visible=${CUDA_VISIBLE_DEVICES:-UNSET}"
  echo "[env_h200] conda_env=${CONDA_DEFAULT_ENV:-NONE} python=$(command -v python || echo NONE)"
  _p REPO "$REPO"; _p TIARA_HOME "$TIARA_HOME"; _p HF_HOME "$HF_HOME"; _p TIARA_RUNS "$TIARA_RUNS"
  _p TIARA_DATASETS "$TIARA_DATASETS"; _p TIARA_FONTS "$TIARA_FONTS"; _p PAPER_HKV_PYLIBS "$PAPER_HKV_PYLIBS"; _p M10_HOTPOT "$M10_HOTPOT"; _p M10_LONGBENCH "$M10_LONGBENCH"
  _p M7_LLAMA_GLOB "$M7_LLAMA_GLOB"; _p M7_QWEN_GLOB "$M7_QWEN_GLOB"; _p M7_MODEL_GLOB "$M7_MODEL_GLOB"
  _p M7_RES "$M7_RES"; _p M7_TIER_PARAMS "$M7_TIER_PARAMS"; _p M7_VLLM_VENV "${M7_VLLM_VENV:-<empty>}"
  printf '  %-16s %-90s %s\n' M7_DISK "$M7_DISK" "$(awk -v d="$M7_DISK" '$3==d{f=1} END{print (f?"ok":"MISSING in /proc/diskstats")}' /proc/diskstats)"
  printf '  %-16s %s\n' M7_EXTRA_PATH "'$M7_EXTRA_PATH'"
fi
