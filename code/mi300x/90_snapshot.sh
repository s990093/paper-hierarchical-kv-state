#!/usr/bin/env bash
# =============================================================================
# 90_snapshot.sh — 把「這個 Lab 現在長什麼樣」存進 workspace
#
# 用途：Lab 一關就沒了。留下 apt / pip 清單與 shell history，
#       下次還原若跟這次不一樣，才有東西可以 diff。
# 建議：跑完任何一次安裝、或每天收工前跑一次。
# =============================================================================
set -Eeuo pipefail
WS=/mlsteam/workspace
SNAP_DIR="$WS/snapshots/$(date +%Y%m%d-%H%M%S)-$(hostname)"
mkdir -p "$SNAP_DIR"
export PATH=/opt/rocm/bin:$PATH

echo "snapshot -> $SNAP_DIR"
dpkg-query -f '${binary:Package}\t${Version}\n' -W  > "$SNAP_DIR/apt-packages.tsv" 2>/dev/null || true
if [ -f "$WS/venv/tiara/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$WS/venv/tiara/bin/activate"
    pip freeze                                       > "$SNAP_DIR/requirements.txt"
    python -c "import torch;print(torch.__version__, torch.version.hip)" \
                                                     > "$SNAP_DIR/torch.txt" 2>&1 || true
fi
{ echo "# $(date -Is)"; uname -a; echo; cat /etc/os-release; } > "$SNAP_DIR/system.txt"
cat /opt/rocm/.info/version 2>/dev/null                > "$SNAP_DIR/rocm-version.txt" || true
rocm-smi                                               > "$SNAP_DIR/rocm-smi.txt" 2>&1 || true
rocminfo                                               > "$SNAP_DIR/rocminfo.txt" 2>&1 || true
cp /root/.bash_history "$SNAP_DIR/bash_history.txt" 2>/dev/null || true
env | sort                                             > "$SNAP_DIR/env.txt"
git -C "$WS/paper-hierarchical-kv-state" rev-parse HEAD > "$SNAP_DIR/repo_head.txt" 2>/dev/null || true
cat "$WS/.vllm_commit" 2>/dev/null                     > "$SNAP_DIR/vllm_commit.txt" || true

echo "[ok] 內容："
ls -la "$SNAP_DIR"
