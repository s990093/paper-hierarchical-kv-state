#!/usr/bin/env bash
# 安全推 git（10/19 13:3xZ 起一律用這支）：不用 git stash／rebase。
# 原因：stash 會換掉工作目錄裡的檔案，正在跑的 job 開著寫的檔（例如 restores.csv）後面寫的列會進孤兒檔、關檔時消失（RUNLOG 15.12）。
# 做法：fetch → merge（只動遠端改過的檔，我們的結果檔不會被碰）→ push。用法：bash code/m10_queue_mi300x/gitpush.sh
set -euo pipefail
cd /mlsteam/workspace/paper-hierarchical-kv-state
export GIT_AUTHOR_NAME="H.W." GIT_AUTHOR_EMAIL="lai09150915@gmail.com" GIT_COMMITTER_NAME="H.W." GIT_COMMITTER_EMAIL="lai09150915@gmail.com"
git fetch -q
if ! git merge-base --is-ancestor origin/main HEAD; then
  git merge -q --no-edit origin/main
fi
git push -q origin main
git log --oneline -1
