#!/usr/bin/env bash
# make nvtop 的本體。每個 slurm job 的 cgroup 只放行自己那張卡，
# 所以單一 nvtop 只看得到一張；多個 job 同時在跑時改用 tmux，一個 job 一格。
#   h200_nvtop.sh            自己所有 RUNNING 的 job
#   h200_nvtop.sh 21566 ...  指定 job
# 各格 synchronize-panes：按一次 q 全部一起離開（排序等按鍵也同步）。
set -euo pipefail
SLURM_BIN=${SLURM_BIN:-/cm/local/apps/slurm/current/bin}
NVTOP=${NVTOP:-$HOME/.conda/envs/tools/bin/nvtop}

if [ $# -gt 0 ]; then jobs=("$@")
else mapfile -t jobs < <("$SLURM_BIN/squeue" -h -u "$USER" -t R -o '%i' | sort -n)
fi
[ ${#jobs[@]} -gt 0 ] || { echo "沒有正在跑的 job（make status 看）"; exit 1; }

pane_cmd() { echo "$SLURM_BIN/srun --jobid=$1 --overlap -n1 --pty $NVTOP"; }

if [ ${#jobs[@]} -eq 1 ]; then exec $(pane_cmd "${jobs[0]}"); fi

title() { "$SLURM_BIN/squeue" -h -j "$1" -o '%i %j %N' 2>/dev/null || echo "$1"; }

if [ -n "${TMUX:-}" ]; then        # 已經在 tmux 裡：開新視窗，不巢狀
  target=$(tmux new-window -P -F '#{session_name}:#{window_index}' -n nvtop "$(pane_cmd "${jobs[0]}")")
  session=
else
  session=nvtop-$$
  tmux new-session -d -s "$session" -n nvtop -x "$(tput cols)" -y "$(tput lines)" "$(pane_cmd "${jobs[0]}")"
  target=$session:nvtop
fi
tmux select-pane -t "$target.0" -T "$(title "${jobs[0]}")"
for j in "${jobs[@]:1}"; do
  tmux split-window -t "$target" "$(pane_cmd "$j")"
  tmux select-pane -t "$target" -T "$(title "$j")"
  tmux select-layout -t "$target" tiled >/dev/null
done
tmux set-window-option -t "$target" pane-border-status top >/dev/null
tmux set-window-option -t "$target" pane-border-format ' #{pane_title} ' >/dev/null
tmux set-window-option -t "$target" synchronize-panes on >/dev/null
if [ -n "$session" ]; then exec tmux attach -t "$session"; fi
