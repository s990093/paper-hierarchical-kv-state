#!/usr/bin/env bash
# Create (or reuse) a tmux session running Claude Code in this repo.
#   code/tmux_claude.sh <name> [dir]      # make claude <name>
# `claude` is not on PATH for non-interactive shells here, so we open the user's
# shell in the session and type the command into it; quitting Claude leaves a shell.
set -euo pipefail

name=${1:-}
[[ -n $name ]] || { echo "usage: $0 <name> [dir]" >&2; exit 2; }
dir=${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

# tmux treats '.' and ':' in names as separators
[[ $name =~ ^[A-Za-z0-9_-]+$ ]] || { echo "bad session name '$name' (use letters, digits, _ -)" >&2; exit 2; }

if tmux has-session -t "=$name" 2>/dev/null; then
  echo "session '$name' already exists (not touching it)"
else
  tmux new-session -d -s "$name" -c "$dir"
  tmux send-keys -t "=$name:" claude Enter
  echo "created session '$name' in $dir"
fi
echo "attach: tmux attach -t $name"
