#!/usr/bin/env bash
# =============================================================================
# sync_scripts.sh — 【在 Mac 上跑】把 code/mi300x/ 推到 Lab 的 workspace
#
# 遠端沒有 rsync（base image 很乾淨），所以走 tar over ssh。
# COPYFILE_DISABLE=1：不要把 macOS 的 ._* AppleDouble 檔一起送過去。
# 目的地 /mlsteam/workspace/bin 會跨 Lab 存活。
# =============================================================================
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${1:-mi300x}"
DEST=/mlsteam/workspace/bin

COPYFILE_DISABLE=1 tar czf - -C "$HERE" . \
  | ssh "$HOST" "mkdir -p $DEST && tar xzf - -C $DEST && rm -f $DEST/._* && chmod 755 $DEST/*.sh $DEST/runsh && ls -la $DEST"
echo "[ok] 已同步到 $HOST:$DEST"
