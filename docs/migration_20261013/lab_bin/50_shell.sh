#!/usr/bin/env bash
# =============================================================================
# 50_shell.sh — 終端機環境（zsh + oh-my-zsh + Starship + eza/bat/fzf/… + tmux + gh）
#
# 本體全部在 /mlsteam/workspace/shell（NFS，重開不會消失）；gh 登入在
# /mlsteam/workspace/.config/gh。/root 每次重開都是空的，所以這支只做「接回」：
# 建 symlink、在 ~/.bashrc 寫入工具 PATH / GH_CONFIG_DIR 與 exec zsh。冪等，可重跑。
#   關掉 zsh：touch /mlsteam/workspace/shell/no-zsh
#   ssh amd <cmd>、Claude Code 的 Bash 工具、非互動 shell 都維持 bash。
# =============================================================================
set -Eeuo pipefail
SH=/mlsteam/workspace/shell
[[ -x $SH/zsh/bin/zsh ]] || { echo "[50_shell] 找不到 $SH/zsh/bin/zsh（要先裝 zsh-bin）" >&2; exit 1; }

mkdir -p ~/.config $SH/cache
ln -sfn $SH/zshrc          ~/.zshrc
ln -sfn $SH/starship.toml  ~/.config/starship.toml
ln -sfn $SH/tmux.conf      ~/.tmux.conf
ln -sfn $SH/terminfo       ~/.terminfo
ln -sfn /mlsteam/workspace/.config/gh ~/.config/gh   # 沒繼承 GH_CONFIG_DIR 的 process（例如早就開著的 Claude）也找得到

# 舊區塊先刪掉再重寫，所以改了這裡重跑就會更新
sed -i '/^# >>> tiara zsh >>>/,/^# <<< tiara zsh <<</d' ~/.bashrc
cat >> ~/.bashrc <<'RC'
# >>> tiara zsh >>>  (50_shell.sh) 工具 PATH + gh 設定（bash/zsh 共用）；只有互動式 tty 才換成 zsh
export PATH=/mlsteam/workspace/shell/bin:$PATH
export GH_CONFIG_DIR=/mlsteam/workspace/.config/gh
if [[ $- == *i* && -t 1 && -z "${_ZSH_FROM_BASHRC:-}" && -z "${CLAUDECODE:-}" \
      && ! -e /mlsteam/workspace/shell/no-zsh && -x /mlsteam/workspace/shell/zsh/bin/zsh ]]; then
  export _ZSH_FROM_BASHRC=1 SHELL=/mlsteam/workspace/shell/zsh/bin/zsh
  exec /mlsteam/workspace/shell/zsh/bin/zsh -l
fi
# <<< tiara zsh <<<
RC

gh_state=$(GH_CONFIG_DIR=/mlsteam/workspace/.config/gh $SH/bin/gh api user -q .login 2>/dev/null || echo "未登入")
echo "[50_shell] OK  zsh=$($SH/zsh/bin/zsh --version | cut -d' ' -f2)  starship=$($SH/bin/starship --version | head -1 | cut -d' ' -f2)  gh=$gh_state"
