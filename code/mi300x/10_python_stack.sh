#!/usr/bin/env bash
# =============================================================================
# 10_python_stack.sh — 自建 Python / PyTorch(ROCm) 環境
#
# 為什麼不用平台給的 `vllm` 範本：
#   那個範本是 rocm7.0 + 別人固定好的 vLLM 版本，我們無法控制它的 commit，
#   也無法確認 OffloadingConnector 的行為與平台 A（3090, vLLM 0.28.0）一致。
#   κ 的跨硬體主張要求兩邊量的是「同一條程式路徑」，所以自己從原始碼裝。
#
# venv 放在 workspace（會存活），不是 /opt（Lab 一關就沒）。
# =============================================================================
set -Eeuo pipefail

WS=/mlsteam/workspace
VENV=$WS/venv/tiara
LOG_DIR="$WS/logs"; mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d-%H%M%S)
exec > >(tee -a "$LOG_DIR/python_stack-$STAMP.log") 2>&1
trap 'echo "[FATAL] line $LINENO 失敗，exit=$?" >&2' ERR

export PATH=/opt/rocm/bin:$PATH
echo "=== 10_python_stack.sh @ $(date -Is) ==="

# --- 0. 這顆 GPU 的 arch（MI300X = gfx942）— 不要猜，問硬體 ------------------
GFX=""
if command -v rocminfo >/dev/null; then
    GFX=$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | head -1 || true)
fi
if [ -z "$GFX" ]; then
    echo "[FATAL] 問不出 gfx arch（rocminfo 不可用）。不要用猜的值繼續。" >&2
    exit 1
fi
echo "[ok] GPU arch = $GFX"
echo "$GFX" > "$WS/.gfx_arch"

# --- 1. venv ------------------------------------------------------------------
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
    echo "[ok] 建立 venv: $VENV"
else
    echo "[skip] venv 已存在: $VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# --- 2. PyTorch for ROCm ------------------------------------------------------
# 不硬寫 wheel index：ROCm 版本與 PyTorch 官方 index 的對應會變，猜錯會裝到
# CUDA 版而靜默失敗。改成「探測哪個 index 真的存在」，由高到低取第一個。
ROCM_VER=$(cat /opt/rocm/.info/version 2>/dev/null | cut -d- -f1 || echo "unknown")
echo "[info] 容器內 ROCm 版本 = $ROCM_VER"

CANDIDATES=(
  "https://download.pytorch.org/whl/rocm7.0"
  "https://download.pytorch.org/whl/nightly/rocm7.0"
  "https://download.pytorch.org/whl/rocm6.4"
  "https://download.pytorch.org/whl/rocm6.3"
)
# 允許明確指定（例如 vLLM v0.26.0 需要 torch 2.11.0，只有 nightly 有）：
#   TORCH_INDEX=https://download.pytorch.org/whl/nightly/rocm7.0 \
#   TORCH_SPEC="torch==2.11.0.dev20260206+rocm7.0" bash 10_python_stack.sh
TORCH_INDEX="${TORCH_INDEX:-}"
[ -n "$TORCH_INDEX" ] && CANDIDATES=("$TORCH_INDEX")
TORCH_INDEX=""
for u in "${CANDIDATES[@]}"; do
    if curl -sfI "$u/torch/" >/dev/null 2>&1 || curl -sf "$u/torch/" -o /dev/null 2>/dev/null; then
        TORCH_INDEX="$u"; break
    fi
done
if [ -z "$TORCH_INDEX" ]; then
    echo "[FATAL] 探測不到任何可用的 PyTorch ROCm wheel index。" >&2
    echo "        candidates 試過： ${CANDIDATES[*]}" >&2
    echo "        不要改用 CPU 版或 CUDA 版硬撐 —— 停下來回報。" >&2
    exit 1
fi
echo "[ok] 使用 PyTorch index: $TORCH_INDEX"
echo "$TORCH_INDEX" > "$WS/.torch_index"

# 🔴 預設**釘死 2.10.0**，不要裝「該 index 上的最新」。
#    原因：vLLM v0.19.1 的 CMakeLists 寫 TORCH_SUPPORTED_VERSION_ROCM "2.10.0"，
#    而 2.11.0.dev 拿掉了 at::cuda::getCurrentHIPStreamMasqueradingAsCUDA，
#    v0.19.1 的 csrc/custom_quickreduce.cu 還在用它 → 會編不過。
#    「裝最新」在這個生態系是錯的預設值。
TORCH_SPEC_DEFAULT="torch==2.10.0+rocm7.0 torchvision torchaudio"
python -m pip install --upgrade --index-url "$TORCH_INDEX" ${TORCH_SPEC:-$TORCH_SPEC_DEFAULT}

# --- 3. 驗證：一定要是 HIP 而不是 CPU/CUDA ------------------------------------
python - <<'PYCHK'
import sys, torch
hip = getattr(torch.version, "hip", None)
cuda = getattr(torch.version, "cuda", None)
print(f"torch            = {torch.__version__}")
print(f"torch.version.hip= {hip}")
print(f"torch.version.cuda={cuda}")
print(f"is_available     = {torch.cuda.is_available()}")
if hip is None:
    sys.exit("[FATAL] 裝到的不是 ROCm build（torch.version.hip is None）。停。")
if not torch.cuda.is_available():
    sys.exit("[FATAL] torch 看不到 GPU。停，不要繼續往下裝。")
n = torch.cuda.device_count()
print(f"device_count     = {n}")
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name}  total={p.total_memory/2**30:.1f} GiB  arch={getattr(p,'gcnArchName','?')}")
PYCHK

# --- 4. 專案共用的小工具 ------------------------------------------------------
python -m pip install \
    numpy pandas matplotlib \
    datasets transformers accelerate \
    requests tqdm pyyaml

echo "=== 10_python_stack.sh 完成 @ $(date -Is) ==="
echo "venv: $VENV   （SSH 進來請先 source $VENV/bin/activate）"
echo "下一步： bash \$WS/paper-hierarchical-kv-state/code/mi300x/20_build_vllm.sh"
