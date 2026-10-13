#!/usr/bin/env bash
# =============================================================================
# set_ssh_port.sh — 【在 Mac 上跑】Lab 重開後更新 SSH 連接埠
#
# 為什麼需要：MLSteam 的 expose-port 是每次建 Port Forwarding 時動態配的，
#             Lab 重開就換一個號碼。同時 host key 也會換（全新 container）。
#
#   用法：  bash code/mi300x/set_ssh_port.sh 46211
# =============================================================================
set -Eeuo pipefail
PORT="${1:-}"
[[ "$PORT" =~ ^[0-9]+$ ]] || { echo "用法: $0 <expose-port>   例: $0 46211" >&2; exit 2; }

CFG=~/.ssh/config
KH=~/.ssh/known_hosts.mlsteam
HOSTIP=210.61.209.139

# Host 行可能有多個別名（例如 `Host amd mi300x`），所以要比對「欄位之一等於 mi300x」，
# 不能用 `^Host mi300x` 死比第一個別名 —— 加了 `amd` 捷徑之後就是這樣壞掉的。
awk '/^Host[ \t]/{for(i=2;i<=NF;i++) if($i=="mi300x") {found=1}} END{exit !found}' "$CFG" \
  || { echo "[FATAL] ~/.ssh/config 裡找不到別名含 'mi300x' 的 Host 區塊" >&2; exit 1; }

cp "$CFG" "$CFG.bak-$(date +%Y%m%d-%H%M%S)"

# 只改該區塊裡的 Port（awk 狀態機，不要用全域 sed 誤傷別的 Host）
awk -v port="$PORT" '
  /^Host[ \t]/ { inblk = 0; for (i = 2; i <= NF; i++) if ($i == "mi300x") inblk = 1 }
  inblk && /^[ \t]*Port[ \t]/ { sub(/[0-9]+[ \t]*$/, port); print; next }
  { print }
' "$CFG" > "$CFG.tmp" && mv "$CFG.tmp" "$CFG"
chmod 600 "$CFG"

# 舊 host key 一定對不上（container 重建）：清掉這個 host:port 的紀錄
touch "$KH"
ssh-keygen -R "[$HOSTIP]:$PORT" -f "$KH" >/dev/null 2>&1 || true

# 舊的 ControlMaster 連線會卡在舊 port，不清掉的話新 port 根本用不到
for h in mi300x amd; do ssh -O exit "$h" >/dev/null 2>&1 || true; done
rm -f ~/.ssh/cm-root@"$HOSTIP":* 2>/dev/null || true

echo "[ok] Port 已更新為 $PORT"
awk '/^Host[ \t].*mi300x/,/^$/' "$CFG" | sed -n '1,6p'
echo
echo "測試： ssh amd 'hostname; rocm-smi --showproductname | head -5'"
