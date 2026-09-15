#!/usr/bin/env bash
# =============================================================================
# 40_claude_code.sh — 在 Lab 上裝 Claude Code，而且**重開不用重裝**
#
# 問題：預設 npm 全域裝在 /usr/lib（容器內，Lab 一關就沒），
#       Claude Code 的設定/登入/skill 放 ~/.claude（HOME=/root，也會沒）。
#       → 每次重開都要重裝、重登入、重設 skill。
#
# 對策：兩者都搬到 /mlsteam/workspace（NFS，跨 Lab 存活）：
#         npm prefix        -> $WS/npm-global
#         CLAUDE_CONFIG_DIR -> $WS/.claude
#       並寫進 /etc/profile.d，00_bootstrap.sh 每次重開會重新套用。
#
# 冪等：已裝好就只確認版本，不重裝。
# =============================================================================
set -Eeuo pipefail
WS=/mlsteam/workspace
NPM_PREFIX="$WS/npm-global"
CLAUDE_CFG="$WS/.claude"
LOG_DIR="$WS/logs"; mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d-%H%M%S)
exec > >(tee -a "$LOG_DIR/claude-code-$STAMP.log") 2>&1
trap 'echo "[FATAL] line $LINENO 失敗，exit=$?" >&2' ERR

echo "=== 40_claude_code.sh @ $(date -Is) ==="

# ---- Node.js ----------------------------------------------------------------
# apt(noble) 只有 node 18.19.1，但 Claude Code 要 >= 22（裝得上去但跑起來會出事）。
# 所以裝官方 tarball，而且裝進 workspace —— 跟 npm prefix 一樣跨 Lab 存活，
# 不必每次重開都重裝。版本是實查 nodejs.org/dist 得到的 LTS。
NODE_VERSION="${NODE_VERSION:-v24.21.0}"     # 2026-09-15 查到的最新 LTS (Krypton)
NODE_DIR="$WS/node/$NODE_VERSION"

need_node=1
if [ -x "$NODE_DIR/bin/node" ]; then
    need_node=0
elif command -v node >/dev/null; then
    cur=$(node --version | sed 's/v//' | cut -d. -f1)
    [ "$cur" -ge 22 ] 2>/dev/null && need_node=0
fi

if [ "$need_node" = "1" ]; then
    echo "[info] 安裝 Node.js $NODE_VERSION -> $NODE_DIR"
    mkdir -p "$WS/node"
    TARBALL="node-$NODE_VERSION-linux-x64.tar.xz"
    URL="https://nodejs.org/dist/$NODE_VERSION/$TARBALL"
    curl -fsSL -o "/tmp/$TARBALL" "$URL"
    sha256sum "/tmp/$TARBALL"           # 留在 log 裡備查
    rm -rf "$NODE_DIR"; mkdir -p "$NODE_DIR"
    tar -xJf "/tmp/$TARBALL" -C "$NODE_DIR" --strip-components=1
    rm -f "/tmp/$TARBALL"
fi

if [ -x "$NODE_DIR/bin/node" ]; then
    export PATH="$NODE_DIR/bin:$PATH"
    NODE_PATH_LINE="export PATH=$NODE_DIR/bin:\$PATH"
else
    NODE_PATH_LINE=""
fi
echo "[ok] node $(node --version) / npm $(npm --version)"
node -e 'const m=+process.versions.node.split(".")[0]; if(m<22){console.error(`[FATAL] node ${process.versions.node} < 22，Claude Code 會出問題`);process.exit(1)}'


mkdir -p "$NPM_PREFIX" "$CLAUDE_CFG"
npm config set prefix "$NPM_PREFIX"
export PATH="$NPM_PREFIX/bin:$PATH"
export CLAUDE_CONFIG_DIR="$CLAUDE_CFG"

if command -v claude >/dev/null && [ -x "$NPM_PREFIX/bin/claude" ]; then
    echo "[skip] claude 已裝在 workspace: $(claude --version 2>&1 | head -1)"
else
    echo "[info] 安裝 @anthropic-ai/claude-code -> $NPM_PREFIX"
    npm install -g @anthropic-ai/claude-code
fi

# 讓兩個 HOME 的 shell 都吃得到（網頁 Terminal 的 HOME 跟 SSH 不同，見 §7.5.3）
cat > /etc/profile.d/98-claude.sh <<EOF
$NODE_PATH_LINE
export NPM_CONFIG_PREFIX=$NPM_PREFIX
export CLAUDE_CONFIG_DIR=$CLAUDE_CFG
export PATH=$NPM_PREFIX/bin:\$PATH
EOF
chmod 644 /etc/profile.d/98-claude.sh
for f in /root/.bashrc "$WS/.bashrc"; do
    touch "$f" 2>/dev/null || continue
    grep -q "CLAUDE_CONFIG_DIR=$CLAUDE_CFG" "$f" 2>/dev/null \
        || cat >> "$f" <<EOF
$NODE_PATH_LINE
export NPM_CONFIG_PREFIX=$NPM_PREFIX
export CLAUDE_CONFIG_DIR=$CLAUDE_CFG
export PATH=$NPM_PREFIX/bin:\$PATH
EOF
done

echo
echo "=== 完成 ==="
echo "  node         : $(node --version)   ($NODE_DIR)"
echo "  claude       : $("$NPM_PREFIX/bin/claude" --version 2>&1 | head -1)"
echo "  安裝位置     : $NPM_PREFIX      （NFS，跨 Lab 存活）"
echo "  設定/登入    : $CLAUDE_CFG      （NFS，跨 Lab 存活 → **不用重登入**）"
echo "  worktrees    : $WS/worktrees"
echo
echo "  ⚠ 第一次要登入一次：ssh amd -t 'claude' 然後照指示做。"
echo "    登入狀態寫在 $CLAUDE_CFG，之後 Lab 重開都不用再登入。"
