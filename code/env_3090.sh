# source 這支：平台 A（RTX 3090，acvlab 共用機）的環境。比照 env_h200.sh（平台 C）的結構（2026-10-17）。
#
#   source code/env_3090.sh            # 安靜
#   source code/env_3090.sh --print    # 另外印出所有路徑、檢查存不存在
#
# 任何值都可以在 source 之前先 export 蓋掉（全部用 ${VAR:-預設}）。
# 例外（_t3_dir）：快取類變數若指到已經不用的舊根目錄，一律改成新根目錄——
#   ~/.bashrc 把 HF_HOME／HF_HUB_CACHE／TRANSFORMERS_CACHE 等設成 /ssd2/hungwei/hf_cache（/ssd2 100% 滿、沒有本專案的模型），
#   舊的 .claude/settings.json 把 UV／TRITON／VLLM 快取設在 /ssd7/hungwei/paper-hkv（已刪，/ssd7 99% 滿）。
#   照 ${VAR:-} 沿用的話，HF 會靜默去 /ssd2 找模型、Triton 會往 /ssd7 寫快取。
# 不設 CUDA_VISIBLE_DEVICES：共用機，呼叫者先 `python code/gpu_guard.py --idle-gpus` 挑空閒卡（CLAUDE.md §3）。
# 這支會被 source，所以不用 set -e、不 exit。

export TIARA_PLATFORM=A_3090

# ---- repo（這支檔案的上一層）------------------------------------------------
export REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# ---- 大檔案根目錄（2026-10-17 從 /ssd7/hungwei/paper-hkv 搬來；docs/ENV_3090.md）----------
export TIARA_HOME="${TIARA_HOME:-/ssd8/hungwei/paper-hkv}"
export PAPER_HKV_BIG="${PAPER_HKV_BIG:-$TIARA_HOME}"

_t3_dir() {   # $1=變數名 $2=預設。沒設、或指到舊根目錄時用預設；其他值（呼叫者刻意蓋的）沿用
  local cur="${!1:-}"
  case "$cur" in
    ""|/ssd7/hungwei/paper-hkv|/ssd7/hungwei/paper-hkv/*|/ssd2/hungwei/hf_cache|/ssd2/hungwei/hf_cache/*|/ssd2/hungwei/datasets|/ssd2/hungwei/datasets/*)
      export "$1=$2" ;;
    *) export "$1=$cur" ;;
  esac
}

# ---- 快取（全部在 TIARA_HOME 底下）-------------------------------------------
_t3_dir HF_HOME "$TIARA_HOME/hf-cache"
_t3_dir HF_HUB_CACHE "$HF_HOME/hub"                 # 優先於 HF_HOME/hub：~/.bashrc 有設，不能不管
_t3_dir HUGGINGFACE_HUB_CACHE "$HF_HUB_CACHE"       # 舊名，huggingface_hub 仍會讀
_t3_dir HF_DATASETS_CACHE "$HF_HOME/datasets"
_t3_dir TRANSFORMERS_CACHE "$HF_HUB_CACHE"          # 已棄用，但 ~/.bashrc 有設；設成同一處避免分岔
_t3_dir UV_CACHE_DIR "$TIARA_HOME/uv-cache"
_t3_dir UV_PYTHON_INSTALL_DIR "$TIARA_HOME/uv-python"
_t3_dir TORCH_HOME "$TIARA_HOME/torch-home"
_t3_dir TRITON_CACHE_DIR "$TIARA_HOME/triton-cache"
_t3_dir VLLM_CACHE_ROOT "$TIARA_HOME/vllm-cache"
_t3_dir XDG_CACHE_HOME "$TIARA_HOME/xdg-cache"
_t3_dir TMPDIR "$TIARA_HOME/tmp"
unset -f _t3_dir
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"          # uv 快取和 venv 同一顆碟也用 copy：避免硬連結讓「刪快取」波及 venv

# ---- TeX（TinyTeX，user-level；論文用 pdflatex）--------------------------------
export TIARA_TEXBIN="${TIARA_TEXBIN:-$TIARA_HOME/texlive/.TinyTeX/bin/x86_64-linux}"
case ":$PATH:" in *":$TIARA_TEXBIN:"*) ;; *) [ -d "$TIARA_TEXBIN" ] && export PATH="$TIARA_TEXBIN:$PATH" ;; esac

# ---- venv tiara（Python 3.12、torch 2.10.0+cu126、transformers 5.17.0；harness＋訓練）----------
# TIARA_SKIP_VENV=1 只給沒有這個 venv 的機器測這支檔案用
export TIARA_VENV="${TIARA_VENV:-$TIARA_HOME/venv/tiara}"
if [ "${TIARA_SKIP_VENV:-0}" != 1 ]; then
  if [ -f "$TIARA_VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$TIARA_VENV/bin/activate"
  else
    echo "[env_3090] 找不到 $TIARA_VENV（docs/ENV_3090.md §2）。照錄、停下來回報（CLAUDE.md 規則 2）" >&2
  fi
fi

# ---- 路徑 ----------------------------------------------------------------------
export TIARA_RUNS="${TIARA_RUNS:-$TIARA_HOME/runs}"           # 每個 run 的原始輸出（不進 git）
export TIARA_DATASETS="${TIARA_DATASETS:-$TIARA_HOME/datasets}"
export TIARA_FONTS="${TIARA_FONTS:-/usr/share/fonts/opentype/noto}"   # Noto Sans/Serif CJK TC（系統已有，fc-list :lang=zh-tw）
# m10_figs 的字型檔：系統只有 .ttc（多語系合一），沒有平台 B 的 NotoSansCJKtc-Regular.otf
export M10_CJK_FONT="${M10_CJK_FONT:-$TIARA_FONTS/NotoSansCJK-Regular.ttc}"
export PAPER_HKV_PYLIBS="${PAPER_HKV_PYLIBS:-$TIARA_HOME/pylibs}"     # ruler_tasks／m10_tasks 的 wonderwords 詞表（側裝，不進 venv）
# m10_tasks 的 T4：HotpotQA dev distractor（sha256 c20b638c…，見 $TIARA_DATASETS/hotpotqa/README.txt）
export M10_HOTPOT="${M10_HOTPOT:-$TIARA_DATASETS/hotpotqa/hotpot_qa_distractor_validation.parquet}"
export M10_LONGBENCH="${M10_LONGBENCH:-$TIARA_DATASETS/longbench}"   # m10_longbench：data/*.jsonl＋config/

# 模型：固定 revision（migration_20261013 README §1.1，和 MI300X、H200 同一版）。直接指到 snapshot，不用 *，換版會找不到而報錯
export M7_LLAMA_GLOB="${M7_LLAMA_GLOB:-$HF_HUB_CACHE/models--unsloth--Llama-3.1-8B-Instruct/snapshots/4699cc75b550f9c6f3173fb80f4703b62d946aa5}"
export M7_QWEN_GLOB="${M7_QWEN_GLOB:-$HF_HUB_CACHE/models--Qwen--Qwen2.5-7B-Instruct-1M/snapshots/e28526f7bb80e2a9c8af03b831a9af3812f18fba}"
# m7_model.py 讀的是 M7_MODEL_GLOB。預設 Llama；Qwen：M7_MODEL_GLOB=$M7_QWEN_GLOB python ...
export M7_MODEL_GLOB="${M7_MODEL_GLOB:-$M7_LLAMA_GLOB}"
# 第三階段（預測器）可能用到的小模型，同樣直接指到 snapshot。
# ⚠️ 用 --revision <sha> 下載的 repo（Llama、e5）沒有 refs/main：離線時用 repo 名稱載入會找不到（smoke 2026-10-17 踩到），
#    一律用這裡的 snapshot 路徑，或 from_pretrained(..., revision=<sha>)。
export TIARA_E5_PATH="${TIARA_E5_PATH:-$HF_HUB_CACHE/models--intfloat--multilingual-e5-small/snapshots/614241f622f53c4eeff9890bdc4f31cfecc418b3}"
export TIARA_QWEN3_06B_PATH="${TIARA_QWEN3_06B_PATH:-$HF_HUB_CACHE/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"
export TIARA_QWEN3_17B_PATH="${TIARA_QWEN3_17B_PATH:-$HF_HUB_CACHE/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"

# 第一階段 harness（m7_write_policy.py、m7_c6_vllm.py）
export M7_RES="${M7_RES:-$REPO/results/m7_write_policy_3090}"
# 限速層參數沿用 MI300X 的 tier_params.json（只換 GPU、不換「裝置」；同 H200 的 03 §R2）
export M7_TIER_PARAMS="${M7_TIER_PARAMS:-$REPO/results/m7_write_policy_mi300x/tier_params.json}"
# Guard 記錄磁碟 I/O 用的 /proc/diskstats 裝置名：/ssd8 是整顆 /dev/sdg（沒有分割區，lsblk 2026-10-17）
export M7_DISK="${M7_DISK:-sdg}"
export M7_VLLM_VENV="${M7_VLLM_VENV:-$TIARA_HOME/venv/vllm028}"   # vLLM 0.28.0+cu129（torch 2.13.0+cu129），和 harness 分開
export M7_EXTRA_PATH="${M7_EXTRA_PATH-}"                          # 平台 B 是 /opt/rocm/bin；這裡不需要（空字串）

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID      # torch 和 nvidia-smi 的編號一致（gpu_guard.smi_index）

R="$M7_RES"                              # m7_run_phase1*.sh 用 $R 當結果目錄
# 不在這裡 mkdir（同 env_h200.sh 10/15 的理由：每次 source 都建空目錄會留下空的 results/ 子目錄）。

# 記錄殼：和平台 B／C 的 m7run 同介面。每一步一個 run 目錄（code/run3090.sh；CLAUDE.md §4.1）
m7run() { local short=$1; shift; local id="$(date +%Y%m%d-%H%M%S)-$short"; RUN_ID=$id bash "$REPO/code/run3090.sh" "$short" env RUN_ID=$id "$@"; }

if [ "${1:-}" = "--print" ]; then
  _p() { printf '  %-22s %-100s %s\n' "$1" "$2" "$( [ -e "$2" ] && echo ok || echo MISSING )"; }
  echo "[env_3090] platform=$TIARA_PLATFORM host=$(hostname) cuda_visible=${CUDA_VISIBLE_DEVICES:-UNSET}"
  echo "[env_3090] venv=${VIRTUAL_ENV:-NONE} python=$(command -v python || echo NONE)"
  _p REPO "$REPO"; _p TIARA_HOME "$TIARA_HOME"; _p HF_HOME "$HF_HOME"; _p HF_HUB_CACHE "$HF_HUB_CACHE"
  _p HF_DATASETS_CACHE "$HF_DATASETS_CACHE"; _p TIARA_RUNS "$TIARA_RUNS"; _p TIARA_DATASETS "$TIARA_DATASETS"
  _p TIARA_FONTS "$TIARA_FONTS"; _p M10_CJK_FONT "$M10_CJK_FONT"; _p PAPER_HKV_PYLIBS "$PAPER_HKV_PYLIBS"; _p M10_HOTPOT "$M10_HOTPOT"; _p M10_LONGBENCH "$M10_LONGBENCH"
  _p M7_LLAMA_GLOB "$M7_LLAMA_GLOB"; _p M7_QWEN_GLOB "$M7_QWEN_GLOB"; _p M7_MODEL_GLOB "$M7_MODEL_GLOB"
  _p TIARA_E5_PATH "$TIARA_E5_PATH"; _p TIARA_QWEN3_06B_PATH "$TIARA_QWEN3_06B_PATH"; _p TIARA_QWEN3_17B_PATH "$TIARA_QWEN3_17B_PATH"
  _p M7_RES "$M7_RES"; _p M7_TIER_PARAMS "$M7_TIER_PARAMS"; _p M7_VLLM_VENV "${M7_VLLM_VENV:-<empty>}"
  _p TIARA_VENV "$TIARA_VENV"; _p TIARA_TEXBIN "$TIARA_TEXBIN"
  _p UV_CACHE_DIR "$UV_CACHE_DIR"; _p UV_PYTHON_INSTALL_DIR "$UV_PYTHON_INSTALL_DIR"; _p TORCH_HOME "$TORCH_HOME"
  _p TRITON_CACHE_DIR "$TRITON_CACHE_DIR"; _p VLLM_CACHE_ROOT "$VLLM_CACHE_ROOT"; _p XDG_CACHE_HOME "$XDG_CACHE_HOME"; _p TMPDIR "$TMPDIR"
  printf '  %-22s %-100s %s\n' M7_DISK "$M7_DISK" "$(awk -v d="$M7_DISK" '$3==d{f=1} END{print (f?"ok":"MISSING in /proc/diskstats")}' /proc/diskstats)"
  printf '  %-22s %s\n' M7_EXTRA_PATH "'$M7_EXTRA_PATH'"
  printf '  %-22s %s\n' TRANSFORMERS_CACHE "$TRANSFORMERS_CACHE"; printf '  %-22s %s\n' HUGGINGFACE_HUB_CACHE "$HUGGINGFACE_HUB_CACHE"
fi
