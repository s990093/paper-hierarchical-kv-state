#!/usr/bin/env bash
# =============================================================================
# 21_build_vllm_v022.sh — 在「另一個 venv + git worktree」建 vLLM v0.22.0 (ROCm)
#
# 為什麼要 v0.22.0（2026-09-15 使用者決定：全部改用 v0.22.0）：
#   * v0.19.1 沒有 TieringOffloadingSpec / fs 次階 → 論文的 SSD 階與 tier_fs baseline 量不到
#     （v0.22.0 起才有；`git show <tag>:vllm/v1/kv_offload/factory.py`）
#   * v0.21.0 起 OffloadingConnector 才是 SupportsHMA（之後若要做 hybrid 用得到）
#   * v0.23.0 起多了 csrc/libtorch_stable/cuda_view.cu，那是 20_build_vllm.sh 註記的
#     「v0.26/v0.29 建不起來」的原因之一；v0.22.0 沒有這個檔案
#   * v0.20–v0.26 的 CMake 要 torch 2.11.0；ROCm 版只有 nightly（2.11.0.dev20260206）
#
# 不動既有環境：
#   venv   /mlsteam/workspace/venv/tiara-v022   （舊的 venv/tiara 保留 0.19.1）
#   原始碼 /mlsteam/workspace/src/vllm-v0.22.0  （git worktree，不 checkout 舊樹）
#
# 規則 2：任何一步失敗就停，不換 tag、不換安裝方式。
#
# 2026-09-15 結果：v0.22.0 + torch nightly 2.11.0.dev20260206 編譯失敗
#   csrc/custom_quickreduce: at::cuda::getCurrentHIPStreamMasqueradingAsCUDA 不存在
#   csrc/mamba/selective_scan_fwd: C10_HIP_CHECK 未宣告
# 同時發現 download.pytorch.org/whl/rocm7.2 有**正式版** torch 2.11–2.14（先前只查 rocm7.0 index 才以為沒有）。
# 使用者決定改建 v0.28.0（與平台 A 同版）。torch 版本依 v0.28.0 官方 docker/Dockerfile.rocm_base
# （ROCm 7.2.3、PyTorch release/2.12、vision v0.27.1）選最接近的正式 wheel：
#   VLLM_REF=v0.28.0 VENV=$WS/venv/tiara-v028 TORCH_INDEX=https://download.pytorch.org/whl/rocm7.2 \
#   TORCH_SPEC="torch==2.12.1+rocm7.2 torchvision==0.27.1+rocm7.2" bash 21_build_vllm_v022.sh
# =============================================================================
set -Eeuo pipefail

WS=/mlsteam/workspace
VLLM_REF="${VLLM_REF:-v0.22.0}"
VENV="${VENV:-$WS/venv/tiara-v022}"
SRC_MAIN=$WS/src/vllm
SRC="${SRC:-$WS/src/vllm-$VLLM_REF}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/nightly/rocm7.0}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.11.0.dev20260206+rocm7.0 torchvision==0.26.0.dev20260216+rocm7.0 torchaudio==2.11.0.dev20260216+rocm7.0}"
LOG_DIR="$WS/logs"; mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d-%H%M%S)
exec > >(tee -a "$LOG_DIR/build_vllm_${VLLM_REF}-$STAMP.log") 2>&1
trap 'echo "[FATAL] line $LINENO 失敗，exit=$? — 停下來回報，不要換方法" >&2' ERR

export PATH=/opt/rocm/bin:$PATH
echo "=== 21_build_vllm_v022.sh @ $(date -Is) ==="
echo "  VLLM_REF=$VLLM_REF  VENV=$VENV  SRC=$SRC"

GFX=$(cat "$WS/.gfx_arch")
export PYTORCH_ROCM_ARCH="$GFX" VLLM_TARGET_DEVICE=rocm
CPU_QUOTA=$(awk '{ if ($1=="max") print 0; else printf "%d", $1/$2 }' /sys/fs/cgroup/cpu.max)
export MAX_JOBS="${MAX_JOBS:-$CPU_QUOTA}"
echo "  PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH MAX_JOBS=$MAX_JOBS"

# --- 1. venv + torch nightly -------------------------------------------------
[ -d "$VENV" ] || python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel "setuptools>=77.0.3,<80.0.0"
# shellcheck disable=SC2086
python -m pip install --index-url "$TORCH_INDEX" $TORCH_SPEC
python - <<'PYCHK'
import sys, torch
print("torch", torch.__version__, "hip", torch.version.hip, "avail", torch.cuda.is_available())
if torch.version.hip is None or not torch.cuda.is_available():
    sys.exit("[FATAL] torch 不是 ROCm build 或看不到 GPU")
x = torch.randn(2048, 2048, device="cuda"); y = (x @ x).sum().item()   # 真的跑一個 kernel
print("matmul OK", y)
PYCHK

# --- 2. amdsmi 綁定（沒有它 vLLM 偵測不到 ROCm 平台）--------------------------
TMP_AMDSMI=$(mktemp -d); cp -r /opt/rocm/share/amd_smi "$TMP_AMDSMI/amd_smi"
python -m pip install "$TMP_AMDSMI/amd_smi"; rm -rf "$TMP_AMDSMI"

# --- 3. 原始碼：git worktree --------------------------------------------------
# 2026-09-15 第一次跑死在這裡：fetch 更新 origin/main 時 "cannot lock ref"（NFS）。
# 只需要 tag，本地已有就不要 fetch。
git -C "$SRC_MAIN" rev-parse -q --verify "refs/tags/$VLLM_REF" >/dev/null \
    || git -C "$SRC_MAIN" fetch origin "refs/tags/$VLLM_REF:refs/tags/$VLLM_REF"
if [ ! -d "$SRC/.git" ] && [ ! -f "$SRC/.git" ]; then
    git -C "$SRC_MAIN" worktree add --detach "$SRC" "$VLLM_REF"
fi
cd "$SRC"
git checkout --detach "$VLLM_REF"
echo "[ok] vLLM commit = $(git rev-parse HEAD)"
rm -rf "$SRC/.deps"

python -m pip install -r requirements/rocm.txt
# rocm.txt 可能把 torch 換掉 —— 檢查仍是 ROCm nightly
python -c "import torch,sys; print('torch after reqs', torch.__version__); sys.exit(0 if torch.version.hip else 1)"

# --- 4. 建置 ------------------------------------------------------------------
python -m pip install -e . --no-build-isolation 2>&1 | tee "$LOG_DIR/pip-editable-${VLLM_REF}-$STAMP.log"

# --- 5. 驗收：平台偵測 + 兩個 offloading spec + HMA ---------------------------
cd /tmp
python - <<'PYCHK'
import sys, vllm, torch
print("vllm", vllm.__version__, "torch", torch.__version__)
import vllm.platforms as p
name = type(p.current_platform).__name__
print("platform", name)
if name != "RocmPlatform":
    sys.exit("[FATAL] 平台偵測不是 RocmPlatform")
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import OffloadingConnector
from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA
print("OffloadingConnector SupportsHMA:", issubclass(OffloadingConnector, SupportsHMA))
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
print("offloading specs:", sorted(OffloadingSpecFactory._registry))
if "TieringOffloadingSpec" not in OffloadingSpecFactory._registry:
    sys.exit("[FATAL] 沒有 TieringOffloadingSpec —— 這正是升級的理由")
PYCHK
git -C "$SRC" rev-parse HEAD > "$WS/.vllm_commit_$VLLM_REF"
echo "=== 21_build_vllm_v022.sh 完成 @ $(date -Is) ==="
