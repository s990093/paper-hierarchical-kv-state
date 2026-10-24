#!/usr/bin/env bash
# H200 版的 gitpush.sh（2026-10-22；原理同 code/m10_queue_mi300x/gitpush.sh：不用 git stash／rebase）。
# 原因同 MI300X：stash 會換掉工作目錄裡的檔，正在跑的 job 開著寫的檔後面寫的列會進孤兒檔（RUNLOG_MI300X 15.12）；
# rebase 會把 merge 攤平、在跑的 job 底下換檔（memory h200-git-branch，10/20）。
# 做法：fetch → origin/h200 有新東西就 merge（只動遠端改過的檔）→ push 到 origin/h200。不碰 main（AMD 結束前）。
# 用法：bash code/h200_gitpush.sh
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
[ "$(git rev-parse --abbrev-ref HEAD)" = h200 ] || { echo "[h200_gitpush] 目前不在 h200 分支 → 停" >&2; exit 2; }
export GIT_AUTHOR_NAME="H.W." GIT_AUTHOR_EMAIL="lai09150915@gmail.com" GIT_COMMITTER_NAME="H.W." GIT_COMMITTER_EMAIL="lai09150915@gmail.com"
git fetch -q origin
if ! git merge-base --is-ancestor origin/h200 HEAD; then
  git merge -q --no-edit origin/h200
fi
git push -q origin HEAD:h200
git log --oneline -1
