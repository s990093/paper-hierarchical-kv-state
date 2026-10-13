#!/usr/bin/env bash
# =============================================================================
# install_pubkey.sh — 全新 Lab 的「雞生蛋」問題：還沒有 SSH，只能用網頁 Terminal
#
# 正常情況下不需要跑這支：公鑰已經存在 /mlsteam/workspace/.ssh/authorized_keys，
# 而 workspace 跨 Lab 存活，所以 00_bootstrap.sh 會自己還原。
# 只有在 workspace 也被清掉、或換一把新金鑰時才需要。
#
#   用法（在網頁 Terminal）： bash install_pubkey.sh
# =============================================================================
set -Eeuo pipefail
PUB='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMpVXoF+mLhvDglv1jZoZklY1OSetumRtvjkolG0mJQP lai09150915@gmail.com mlsteam-mi300x'
WS=/mlsteam/workspace
mkdir -p "$WS/.ssh" /root/.ssh
grep -qF "$PUB" "$WS/.ssh/authorized_keys" 2>/dev/null || printf '%s\n' "$PUB" >> "$WS/.ssh/authorized_keys"
install -m 600 "$WS/.ssh/authorized_keys" /root/.ssh/authorized_keys
chmod 700 "$WS/.ssh" /root/.ssh
chmod 600 "$WS/.ssh/authorized_keys"
echo "[ok] 公鑰已安裝："
ssh-keygen -lf /root/.ssh/authorized_keys
