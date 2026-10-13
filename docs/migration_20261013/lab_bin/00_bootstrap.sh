#!/usr/bin/env bash
# =============================================================================
# 00_bootstrap.sh — MLSteam (Manta) MI300X Lab 每次啟動都要跑的第一支腳本
#
# 為什麼需要：平台底層是 Kubernetes，Lab 一關掉 container 就沒了。
#   只有 /mlsteam/workspace（工作區）與 /mlsteam/data/<掛載資料夾> 會留下來。
#   所以「裝過的東西」必須能靠腳本一鍵重來，不能靠手動。
#
# 冪等（idempotent）：重跑安全。
# 失敗即停：任何一步失敗就停下並印出完整錯誤（CLAUDE.md §1 規則 2）。
# =============================================================================
set -Eeuo pipefail

WS=/mlsteam/workspace
LOG_DIR="$WS/logs"
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d-%H%M%S)
exec > >(tee -a "$LOG_DIR/bootstrap-$STAMP.log") 2>&1

trap 'echo "[FATAL] line $LINENO 失敗，exit=$? — 停下來，不要繞路硬幹" >&2' ERR

echo "=== 00_bootstrap.sh @ $(date -Is) ==="
echo "hostname: $(hostname)"

# --- 1. 系統套件 -------------------------------------------------------------
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    build-essential git curl wget unzip zip rsync \
    openssh-server ca-certificates \
    iproute2 net-tools dnsutils \
    vim tmux htop jq bc \
    python3-dev python3-venv python3-pip \
    pkg-config cmake ninja-build \
    numactl libnuma-dev

# --- 2. SSH server -----------------------------------------------------------
mkdir -p /run/sshd /root/.ssh
chmod 700 /root/.ssh

# 公鑰從 workspace 還原（workspace 是唯一會跨 Lab 存活的地方）
if [ -f "$WS/.ssh/authorized_keys" ]; then
    install -m 600 "$WS/.ssh/authorized_keys" /root/.ssh/authorized_keys
    echo "[ok] authorized_keys 已從 workspace 還原 ($(wc -l < /root/.ssh/authorized_keys) 把金鑰)"
else
    echo "[FATAL] $WS/.ssh/authorized_keys 不存在 — SSH 會連不進來。" >&2
    echo "        請先在 網頁 Terminal 執行 code/mi300x/install_pubkey.sh，或手動貼上公鑰。" >&2
    exit 1
fi

# 只允許金鑰登入（這台機器是對外開 port 的，不要留密碼登入）
sshd_conf=/etc/ssh/sshd_config.d/99-tiara.conf
mkdir -p "$(dirname "$sshd_conf")"
cat > "$sshd_conf" <<'EOF'
PermitRootLogin prohibit-password
PasswordAuthentication no
KbdInteractiveAuthentication no
PubkeyAuthentication yes
ClientAliveInterval 60
ClientAliveCountMax 5
EOF

ssh-keygen -A                       # 產生 host key（fresh container 沒有）
service ssh restart || /etc/init.d/ssh restart
sleep 1
if pgrep -x sshd >/dev/null; then
    echo "[ok] sshd 已啟動，listening:"
    ss -tlnp 2>/dev/null | grep -E ':22\b' \
      || netstat -tlnp 2>/dev/null | grep -E ':22\b' \
      || { echo "[FATAL] sshd 有 process 但沒 listen 在 22" >&2; exit 1; }
else
    echo "[FATAL] sshd 沒起來" >&2; exit 1
fi

# --- 3. PATH / shell 環境 ----------------------------------------------------
# ROCm 與平台 venv 的路徑不在預設 PATH，SSH 進來會找不到 rocm-smi。
# 寫進 /root/.bashrc（互動 shell）與 /root/.profile（非互動 ssh 指令）都要。
# ⚠ 同一台機器有兩個 HOME：網頁 Terminal = /mlsteam/workspace，SSH = /root。
#   只寫其中一邊，另一邊進來就找不到 rocm-smi / gpuwatch。兩邊都寫。
#   另外 /etc/profile.d 對兩者都有效，當作第三道保險。
ENVLINE='export PATH=/mlsteam/workspace/bin:/mlsteam/workspace/venv/tiara/bin:/opt/venv/bin:/opt/rocm/bin:$PATH'
for f in /root/.bashrc /root/.profile "$WS/.bashrc" "$WS/.profile"; do
    touch "$f" 2>/dev/null || continue
    grep -qF "$ENVLINE" "$f" 2>/dev/null || echo "$ENVLINE" >> "$f"
done

# 🔴 模型權重 / HF 快取一定要指到「掛載的 Data 資料夾」，不要留在預設的 ~/.cache。
#    ~/.cache 在容器裡（/root 或 /mlsteam/workspace，取決於哪個入口），Lab 一關
#    就沒，每次都要重下載幾十 GB。$DATA 是 NFS，跨 Lab 存活。
DATA_DIR=/mlsteam/data/tiara
CACHE_LINES=$(cat <<EOF
export HF_HOME=$DATA_DIR/hf-cache
export HUGGINGFACE_HUB_CACHE=$DATA_DIR/hf-cache/hub
export TRANSFORMERS_CACHE=$DATA_DIR/hf-cache/transformers
export TORCH_HOME=$DATA_DIR/hf-cache/torch
export VLLM_CACHE_ROOT=$DATA_DIR/hf-cache/vllm
export TIARA_DATA=$DATA_DIR
export TIARA_RUNS=$DATA_DIR/runs
EOF
)
if [ -d "$DATA_DIR" ]; then
    mkdir -p "$DATA_DIR"/hf-cache/{hub,transformers,torch,vllm} "$DATA_DIR/runs"
    { printf '%s\n' "$ENVLINE"; printf '%s\n' "$CACHE_LINES"; } > /etc/profile.d/99-tiara.sh
    for f in /root/.bashrc "$WS/.bashrc"; do
        grep -q 'HF_HOME=/mlsteam/data' "$f" 2>/dev/null || printf '%s\n' "$CACHE_LINES" >> "$f"
    done
    eval "$CACHE_LINES"
    echo "[ok] HF/模型快取 -> $DATA_DIR/hf-cache（跨 Lab 存活）"
else
    printf '%s\n' "$ENVLINE" > /etc/profile.d/99-tiara.sh
    echo "[WARN] $DATA_DIR 沒掛載 —— 模型會下載到不會存活的地方。" >&2
    echo "       Lab → 🧰 資料夾 → 開啟 tiara → 套用（會重開 Lab）" >&2
fi
chmod 644 /etc/profile.d/99-tiara.sh
eval "$ENVLINE"

# Claude Code 的 SSH 連線會在這裡開 git worktree。
# 必須放在 workspace（NFS）上 —— 預設會用 $HOME=/root，Lab 一關 worktree 全沒。
mkdir -p "$WS/worktrees"
echo "[ok] worktrees -> $WS/worktrees"

# --- 3b. git 的 dubious ownership --------------------------------------------
# /mlsteam/* 是 NFS，uid 被 squash 成別的號碼，git 會拒絕操作（safe.directory 保護）。
# 這是平台的常態不是異常，所以在 bootstrap 就宣告一次，之後所有腳本才不會卡。
# 🔴 用 --system（/etc/gitconfig）不是 --global（$HOME/.gitconfig）：
#    這個平台的 **網頁 Terminal 的 HOME 是 /mlsteam/workspace，SSH 的 HOME 是 /root**
#    （平台幫網頁 Terminal 覆寫了 HOME，/etc/passwd 寫的還是 /root）。
#    用 --global 的話兩邊會寫到不同檔案 —— 網頁跑得動、SSH 進來就爆 dubious ownership。
# 用 '*'（全部）而不是列舉路徑，理由是**列舉不完**：
#   vLLM 的 CMake 會在建置期間用 FetchContent clone 出
#   src/vllm/.deps/triton_kernels-src 之類的目錄再 git checkout，
#   那些路徑事先不存在，沒辦法預先登記 —— 2026-09-15 的建置就是死在這裡。
#   safe.directory 只支援 '*'（全部）或完整路徑，沒有前綴萬用字元。
#
# 這個放寬是有界的：單人使用的拋棄式容器，/mlsteam 是使用者自己的 NFS export，
# uid squash 是平台設計如此而非權限異常，且容器本身隨時由腳本重建。
git config --system --get-all safe.directory 2>/dev/null | grep -qxF '*' \
    || git config --system --add safe.directory '*'
echo "[ok] git safe.directory (system): $(git config --system --get-all safe.directory | tr '\n' ' ')"

# --- 4. /dev/shm 大小（平台預設只有 1 GB，vLLM 會死在這裡）-------------------
SHM_KB=$(df -k /dev/shm | awk 'NR==2{print $2}')
SHM_GB=$(( SHM_KB / 1024 / 1024 ))
echo "[info] /dev/shm = ${SHM_GB} GB"
if [ "$SHM_GB" -lt 16 ]; then
    echo "[WARN] /dev/shm 只有 ${SHM_GB} GB —— vLLM 起 worker 很可能會失敗。" >&2
    echo "       到網頁 開發環境 → ⚙ 設定 → Shared Memory Size 調大（本專案設 192 GB）。" >&2
    echo "       ⚠ 改這個會重開 Lab。" >&2
fi

# --- 5. 確認 GPU 看得到 -------------------------------------------------------
echo "--- rocm-smi ---"
if command -v rocm-smi >/dev/null; then
    rocm-smi || true
else
    echo "[WARN] rocm-smi 不在 PATH（/opt/rocm/bin 存在嗎？）"
    ls -d /opt/rocm* 2>/dev/null || echo "  沒有 /opt/rocm*"
fi
echo "--- amd-smi ---"
command -v amd-smi >/dev/null && (amd-smi list || true) || echo "[info] 沒有 amd-smi"

echo "=== 00_bootstrap.sh 完成 @ $(date -Is) ==="
echo "下一步： bash $WS/paper-hierarchical-kv-state/code/mi300x/10_python_stack.sh"
