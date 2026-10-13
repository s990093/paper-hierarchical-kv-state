#!/usr/bin/env bash
# =============================================================================
# 20_build_vllm.sh — 從原始碼建 vLLM (ROCm)
#
# 為什麼從原始碼：論文 §6.5 的 CPU/SSD 階是走 vLLM 的 OffloadingConnector
#   搬原始位元組。平台 A（3090）用的是 vLLM 0.28.0。平台 B 要能對照，
#   必須釘住同一個 commit —— 用平台預裝的 image 就做不到這件事。
#
# 這一步很久（MI300X 上通常 40–90 分鐘）。請用 tmux 跑，SSH 斷線才不會前功盡棄。
# =============================================================================
set -Eeuo pipefail

WS=/mlsteam/workspace
VENV=$WS/venv/tiara
SRC=$WS/src
LOG_DIR="$WS/logs"; mkdir -p "$LOG_DIR" "$SRC"
STAMP=$(date +%Y%m%d-%H%M%S)
exec > >(tee -a "$LOG_DIR/build_vllm-$STAMP.log") 2>&1
trap 'echo "[FATAL] line $LINENO 失敗，exit=$?" >&2' ERR

# ---- 版本選擇：這是研究出來的，不是挑的 ----------------------------------
# 2026-09-15 實查 vLLM 各 tag 的 torch 釘選 ↔ ROCm 版 torch 的供給：
#
#   vLLM v0.27.0–v0.29.0  要 torch 2.13.0   → ROCm 沒有，任何管道都沒有
#   vLLM v0.20.0–v0.26.0  要 torch 2.11.0   → 只有 nightly 的 .dev20260206（7個月前）
#   vLLM v0.17.0–v0.19.1  要 torch 2.10.0   → ✅ pytorch.org /whl/rocm7.0 有
#   vLLM v0.14.1–v0.16.0  要 torch 2.9.1    → repo.radeon.com 有（更舊）
#
# 選 v0.19.1 —— 「torch 供得起的最新 vLLM」。佐證：
#   * AMD 官方 rocm/vllm image 就有 `pytorch_2.10.0_vllm_0.19.1` 這個配對
#   * v0.19.1 有 vllm/distributed/.../offloading_connector.py（論文 A3 驗收條件）
#   * v0.19.1 沒有 csrc/libtorch_stable/cuda_view.cu，也沒用 torch::stable 的
#     layout() —— 那正是 v0.26/v0.29 建不起來的原因
#   * v0.19.1 的 CMakeLists 寫 TORCH_SUPPORTED_VERSION_ROCM "2.10.0"
#
# ⚠️ **跨平台可比性**：平台 A（3090）用 v0.28.0，這裡是 v0.19.1。
#    在兩邊對齊之前，κ 的跨硬體數字一律標 NOT_COMPARABLE。
#    這不是工程偏好問題，是論文能不能宣稱可比的問題。見 docs/MI300X_MLSTEAM.md §10.45。
#
#   換版本：VLLM_REF=v0.17.0 bash 20_build_vllm.sh
VLLM_REF="${VLLM_REF:-v0.19.1}"
VLLM_REF_PLATFORM_A="v0.28.0"

export PATH=/opt/rocm/bin:$PATH
# shellcheck disable=SC1091
source "$VENV/bin/activate"

GFX=$(cat "$WS/.gfx_arch" 2>/dev/null || true)
if [ -z "$GFX" ]; then
    echo "[FATAL] 找不到 $WS/.gfx_arch — 請先跑 10_python_stack.sh" >&2; exit 1
fi
export PYTORCH_ROCM_ARCH="$GFX"
export VLLM_TARGET_DEVICE=rocm
# 🔴 不要用 nproc：容器裡 nproc 回報整台 host（192），我們的 cgroup 配額只有 32。
#    用 192 開 job 會超賣 6 倍，編譯不只變慢，還可能被 OOM/throttle 打斷。
CPU_QUOTA=$(awk '{ if ($1=="max") print 0; else printf "%d", $1/$2 }' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 0)
[ "$CPU_QUOTA" -gt 0 ] 2>/dev/null || CPU_QUOTA=$(nproc)
export MAX_JOBS="${MAX_JOBS:-$CPU_QUOTA}"

echo "=== 20_build_vllm.sh @ $(date -Is) ==="
echo "  VLLM_REF          = $VLLM_REF"
echo "  PYTORCH_ROCM_ARCH = $PYTORCH_ROCM_ARCH"
echo "  MAX_JOBS          = $MAX_JOBS   (cgroup 配額=$CPU_QUOTA, host nproc=$(nproc))"

# 上一次失敗留下的 FetchContent 半成品會讓 cmake 以為已經下載好，
# 於是跳過 clone 卻又 checkout 不了。重跑前清掉。
if [ -d "$SRC/vllm/.deps" ] && [ "${KEEP_DEPS:-0}" != "1" ]; then
    echo "[info] 清掉上次的 $SRC/vllm/.deps（KEEP_DEPS=1 可保留）"
    rm -rf "$SRC/vllm/.deps"
fi

cd "$SRC"
if [ ! -d vllm/.git ]; then
    git clone https://github.com/vllm-project/vllm.git
fi
cd vllm
git fetch --all --tags
git checkout "$VLLM_REF"
echo "[ok] vLLM commit = $(git rev-parse HEAD)"
git rev-parse HEAD > "$WS/.vllm_commit"
echo "$VLLM_REF" > "$WS/.vllm_ref"
if [ "$VLLM_REF" != "$VLLM_REF_PLATFORM_A" ]; then
    echo "[!!] 注意：本次建的是 $VLLM_REF，平台 A 是 $VLLM_REF_PLATFORM_A。" >&2
    echo "     兩邊版本不同 → κ 的跨硬體比較 **NOT_COMPARABLE**，報告要標註。" >&2
fi

# ROCm 的相依（不是 requirements/cuda.txt）
if [ -f requirements/rocm.txt ]; then
    python -m pip install -r requirements/rocm.txt
elif [ -f requirements-rocm.txt ]; then
    python -m pip install -r requirements-rocm.txt
else
    echo "[FATAL] 找不到 rocm 的 requirements 檔，這個 tag 的佈局跟預期不同 —— 停下來看。" >&2
    ls requirements* -d 2>/dev/null || true
    exit 1
fi

# 就地安裝（editable），方便之後對 connector 加 instrumentation。
# `setup.py develop` 在新的 setuptools 已棄用，優先走 pip 的 editable；
# 失敗才退回舊路徑，而且**要把兩次的錯誤都留下來**，不要靜默換方法。
# ⚠️ 這裡**故意沒有 fallback**。
#    上一版寫成「pip -e 失敗就改跑 setup.py develop」，結果 2026-09-15 那次
#    兩條路死在同一個 cmake 錯誤上，白跑 6 分鐘，還把真正的原因
#    （NFS dubious ownership）埋在兩層 traceback 底下。
#    CLAUDE.md §1 規則 2：失敗就記錄、停下、回報，不要換方式硬幹到有輸出為止。
set -o pipefail
if ! python -m pip install -e . --no-build-isolation 2>&1 | tee "$LOG_DIR/pip-editable-$STAMP.log"; then
    echo >&2
    echo "[FATAL] vLLM 建置失敗。以下是 log 裡真正的錯誤（不是最外層的 traceback）：" >&2
    grep -nE "CMake Error|Could NOT find|fatal:|error: .*(not found|No such)" \
        "$LOG_DIR/pip-editable-$STAMP.log" | tail -20 >&2
    echo >&2
    echo "  完整 log: $LOG_DIR/pip-editable-$STAMP.log" >&2
    echo "  不要換 tag、不要換安裝方式 —— 先弄清楚上面那條錯誤。" >&2
    exit 1
fi

# --- 驗證 ---------------------------------------------------------------------
python - <<'PYCHK'
import sys
import vllm, torch
print("vllm      =", vllm.__version__)
print("torch     =", torch.__version__, "hip=", torch.version.hip)
try:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import OffloadingConnector  # noqa
    print("OffloadingConnector: OK (import 成功)")
except Exception as e:
    print("OffloadingConnector: IMPORT FAILED ->", repr(e))
    print("  ⚠ 這是 A3 驗收條件。失敗就要立刻回報，不要繼續往下做實驗。")
    sys.exit(1)
PYCHK

echo "=== 20_build_vllm.sh 完成 @ $(date -Is) ==="
