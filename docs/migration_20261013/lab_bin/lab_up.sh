#!/usr/bin/env bash
# =============================================================================
# lab_up.sh — 【Lab 每次重開後跑這一支就好】
#
# 平台是 Kubernetes：Lab 一關（或改掛載、改硬體規格），container 整個重建，
# 所有 apt 裝的東西、/root 底下的東西全部消失。
# 會活下來的只有：
#   /mlsteam/workspace          （NFS，950G，本專案的家）
#   /mlsteam/data/<掛載資料夾>   （NFS，要在 Lab 設定裡打開掛載）
#
# 所以還原流程 = 重跑這支腳本。不要手動裝任何東西。
# =============================================================================
set -Eeuo pipefail
WS=/mlsteam/workspace
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "######################################################################"
echo "# Tiara MI300X — Lab 還原  $(date -Is)"
echo "# host: $(hostname)"
echo "######################################################################"

bash "$HERE/00_bootstrap.sh"          # 系統套件 + sshd + 金鑰 + PATH + git safe.directory
bash "$HERE/15_monitoring.sh"         # nvtop / radeontop / btop / iostat / gpuwatch / gpulog
bash "$HERE/10_python_stack.sh"       # venv + PyTorch(ROCm)（venv 在 workspace，通常是 no-op）
bash "$HERE/30_project.sh"            # repo + 目錄骨架 + 環境指紋
bash "$HERE/05_hwinfo.sh"             # 硬體盤點（含 cgroup 真實配額）
bash "$HERE/40_claude_code.sh"        # node 24 + Claude Code（都在 workspace，不用重登入）

echo
echo "######################  自我驗證  ######################"
set +e
bash "$HERE/selftest.sh"
SELFTEST_RC=$?
set -e

cat <<EOF

還原完成（selftest 失敗項數：$SELFTEST_RC）。
vLLM 沒有自動建（要 40–90 分鐘），需要時手動跑：
    tmux new -s build
    bash $HERE/20_build_vllm.sh

⚠ 別忘了：Lab 重開後 **expose-port 會換號碼**。
   到網頁 Lab → ⚙ 設定 → Port Forwarding 看新的 port，然後在 Mac 上跑：
       bash code/mi300x/set_ssh_port.sh <新port>
EOF

exit "$SELFTEST_RC"
