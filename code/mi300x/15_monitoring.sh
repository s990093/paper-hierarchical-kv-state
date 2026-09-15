#!/usr/bin/env bash
# =============================================================================
# 15_monitoring.sh — 監控工具（GPU / CPU / IO）
#
# 版本不是猜的，是 2026-09-15 實際查過的：
#   nvtop      apt(noble) 3.0.2   上游 3.3.2（只有 AppImage，容器內 FUSE 常掛，不用）
#              ※ nvtop 從 2.0 起就支援 AMD GPU，3.0.2 夠用
#   radeontop  apt 1.4-2
#   btop       apt 1.3.0          上游 1.4.7
#   htop       apt 3.3.0-4build1
#   sysstat    apt 12.6.1-2       （iostat/mpstat/pidstat）
#   amdgpu_top 上游 v0.11.5 .deb  ← AMD 官方生態裡最好用的那個，選配
#
# rocm-smi / amd-smi 是 ROCm 自帶的，不用裝。
# =============================================================================
set -Eeuo pipefail
WS=/mlsteam/workspace
LOG_DIR="$WS/logs"; mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d-%H%M%S)
exec > >(tee -a "$LOG_DIR/monitoring-$STAMP.log") 2>&1
trap 'echo "[FATAL] line $LINENO 失敗，exit=$?" >&2' ERR

export DEBIAN_FRONTEND=noninteractive
export PATH=/opt/rocm/bin:$PATH
echo "=== 15_monitoring.sh @ $(date -Is) ==="

# --- apt 來源的工具（版本由 Ubuntu 24.04 決定，穩定優先）---------------------
apt-get update -qq
apt-get install -y --no-install-recommends \
    nvtop radeontop htop btop iotop sysstat ncdu tree pv

echo "[ok] 已安裝版本："
for p in nvtop radeontop htop btop iotop sysstat; do
    printf '  %-10s %s\n' "$p" "$(dpkg-query -W -f='${Version}' "$p" 2>/dev/null || echo MISSING)"
done

# --- amdgpu_top（選配，來源是 GitHub release 的 .deb）-----------------------
# 失敗不擋流程：apt 那批已經夠用，這支只是更好看。
AMDGPU_TOP_VER="${AMDGPU_TOP_VER:-0.11.5}"
if ! command -v amdgpu_top >/dev/null; then
    DEB="amdgpu-top_without_gui_${AMDGPU_TOP_VER}-1_amd64.deb"
    URL="https://github.com/Umio-Yasuno/amdgpu_top/releases/download/v${AMDGPU_TOP_VER}/${DEB}"
    echo "[info] 嘗試安裝 amdgpu_top v${AMDGPU_TOP_VER}（選配）"
    if curl -fsSL -o "/tmp/$DEB" "$URL"; then
        sha256sum "/tmp/$DEB"          # 記在 log 裡，之後可比對
        apt-get install -y "/tmp/$DEB" && echo "[ok] amdgpu_top 已安裝" \
            || echo "[warn] amdgpu_top 安裝失敗 —— 不影響其他工具"
        rm -f "/tmp/$DEB"
    else
        echo "[warn] 下載不到 amdgpu_top（網路或版本號變了）—— 跳過，不影響其他工具"
    fi
else
    echo "[skip] amdgpu_top 已存在"
fi

# --- rocmtop（PyPI，nvitop 風格的 ROCm 監控）----------------------------------
# 2026-09-15 實查：PyPI rocmtop 0.2.0，來源 github.com/Liam-zzy/rocmtop
#
# ⚠️ 裝在**獨立的 tools venv**，不裝進 $WS/venv/tiara。
#    理由跟平台 A 用 pylibs/ 側裝一樣：實驗用的 venv 裡只能有實驗需要的東西，
#    監控工具的相依（可能釘不同版的 psutil/rich）不該有機會動到 torch/vLLM 的解析。
TOOLS_VENV="$WS/venv/tools"
if [ ! -f "$TOOLS_VENV/bin/rocmtop" ]; then
    echo "[info] 建立 tools venv 並安裝 rocmtop"
    [ -d "$TOOLS_VENV" ] || python3 -m venv "$TOOLS_VENV"
    "$TOOLS_VENV/bin/pip" install -q --upgrade pip
    "$TOOLS_VENV/bin/pip" install -q rocmtop
fi
if [ -x "$TOOLS_VENV/bin/rocmtop" ]; then
    echo "[ok] rocmtop $("$TOOLS_VENV/bin/pip" show rocmtop 2>/dev/null | awk '/^Version/{print $2}')"
    ln -sf "$TOOLS_VENV/bin/rocmtop" "$WS/bin/rocmtop"
else
    echo "[warn] rocmtop 安裝失敗 —— 不影響其他工具"
fi

# --- 方便的小包裝 -------------------------------------------------------------
mkdir -p "$WS/bin"

cat > "$WS/bin/gpuwatch" <<'EOF'
#!/usr/bin/env bash
# gpuwatch — 每 1 秒刷新一次 MI300X 的關鍵欄位（純文字，SSH 友善）
export PATH=/opt/rocm/bin:$PATH
watch -n1 'rocm-smi --showuse --showmemuse --showtemp --showpower --showclocks 2>/dev/null | sed -n "1,40p"'
EOF

cat > "$WS/bin/gpulog" <<'EOF'
#!/usr/bin/env bash
# gpulog <輸出csv> [間隔秒] — 把 GPU 使用率/VRAM/溫度/功耗逐秒寫成 CSV
# 量測時開著它，事後才有東西可以對照「這段時間卡上還有誰」。
export PATH=/opt/rocm/bin:$PATH
OUT="${1:?用法: gpulog <out.csv> [interval]}"
INT="${2:-1}"
echo "ts,gpu_util_pct,vram_used_b,vram_total_b,temp_c,power_w" > "$OUT"
while true; do
    TS=$(date -Is)
    U=$(rocm-smi --showuse   --csv 2>/dev/null | awk -F, 'NR==2{print $2}')
    M=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, 'NR==2{print $2","$3}')
    T=$(rocm-smi --showtemp  --csv 2>/dev/null | awk -F, 'NR==2{print $2}')
    P=$(rocm-smi --showpower --csv 2>/dev/null | awk -F, 'NR==2{print $2}')
    echo "$TS,$U,$M,$T,$P" >> "$OUT"
    sleep "$INT"
done
EOF

chmod 755 "$WS/bin/gpuwatch" "$WS/bin/gpulog"

# 讓 $WS/bin 進 PATH（SSH 進來直接打 gpuwatch）
LINE='export PATH=/mlsteam/workspace/bin:/opt/venv/bin:/opt/rocm/bin:$PATH'
for f in /root/.bashrc /root/.profile; do
    grep -qF "$LINE" "$f" 2>/dev/null || echo "$LINE" >> "$f"
done

echo
echo "=== 15_monitoring.sh 完成 @ $(date -Is) ==="
cat <<'EOF'
可用指令：
  rocm-smi              ROCm 官方，最權威
  amd-smi monitor       ROCm 新版 CLI（有 process 層級資訊）
  nvtop                 名字叫 nvtop，但支援 AMD，TUI 圖形化
  radeontop             輕量 AMD GPU 使用率
  amdgpu_top --pci 0000:a6:00.0   最詳細。**一定要加 --pci**，不然會列到別的租戶的卡
  amdtop                nvitop 風格（含 CPU 每核心 + per-process VRAM）。已套本地 patch
  rocmtop               nvitop 風格的輕量版（PyPI，裝在 venv/tools）
  btop / htop           CPU / RAM / process
  iostat -x 1           磁碟 IO（NFS 也看得到）
  gpuwatch              ← 自製：1 秒刷新的 rocm-smi 精簡版
  gpulog out.csv 1      ← 自製：逐秒寫 CSV，量測時務必開著
EOF
