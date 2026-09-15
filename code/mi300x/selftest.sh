#!/usr/bin/env bash
# =============================================================================
# selftest.sh — 環境自我驗證：砍掉重開之後，跑這一支就知道還原乾不乾淨
#
# 設計原則：**每一項都印出實測值**，不是只印 PASS。
#   只印 PASS 的檢查等於沒檢查 —— 看不到值就沒辦法發現「通過但數字不對」。
# 退出碼：0 = 全過；非 0 = 失敗項數。
# =============================================================================
set -uo pipefail          # 注意：這裡**故意不用 -e**，要跑完所有檢查再回報總數
WS=/mlsteam/workspace
DATA=/mlsteam/data/tiara
VENV=$WS/venv/tiara
export PATH=$WS/bin:/opt/rocm/bin:$PATH

FAIL=0; PASS=0
ok()   { printf '  \033[32m✓\033[0m %-34s %s\n' "$1" "${2:-}"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31m✗\033[0m %-34s %s\n' "$1" "${2:-}"; FAIL=$((FAIL+1)); }
warn() { printf '  \033[33m!\033[0m %-34s %s\n' "$1" "${2:-}"; }

echo "======================================================================"
echo " Tiara MI300X selftest   $(date -Is)"
echo " host: $(hostname)"
echo "======================================================================"

echo; echo "[1] 持久化儲存（Lab 重開後必須還在）"
for m in "$WS" "$DATA"; do
    if mountpoint -q "$m" 2>/dev/null || [ -d "$m" ]; then
        ok "$m" "$(df -h "$m" | awk 'NR==2{print $2" total, "$4" free, "$1}')"
    else
        bad "$m" "不存在 —— Data 資料夾沒掛載？（Lab → 🧰 → ⚙ → 開關 → 套用）"
    fi
done

echo; echo "[2] 腳本與 repo"
for f in 00_bootstrap.sh 05_hwinfo.sh 10_python_stack.sh 15_monitoring.sh \
         20_build_vllm.sh 30_project.sh 90_snapshot.sh lab_up.sh runsh selftest.sh; do
    [ -x "$WS/bin/$f" ] && PASS=$((PASS+1)) || bad "bin/$f" "缺少或不可執行"
done
ok "workspace/bin 腳本" "$(ls "$WS/bin" | wc -l) 個"
if git -C "$WS/paper-hierarchical-kv-state" rev-parse --short HEAD >/dev/null 2>&1; then
    ok "論文 repo" "HEAD=$(git -C "$WS/paper-hierarchical-kv-state" rev-parse --short HEAD)"
else
    bad "論文 repo" "不是有效的 git checkout —— 跑 30_project.sh"
fi

echo; echo "[2b] HOME 陷阱（網頁 Terminal 與 SSH 的 HOME 不同）"
ok "目前 HOME" "$HOME  (網頁Terminal=/mlsteam/workspace, SSH=/root)"
if git config --system --get-all safe.directory 2>/dev/null | grep -q paper-hierarchical; then
    ok "git safe.directory" "寫在 --system(/etc/gitconfig)，兩個 HOME 都吃得到"
else
    bad "git safe.directory" "不在 --system —— SSH 進來會噴 dubious ownership"
fi

echo; echo "[3] SSH（Mac 要連得進來）"
pgrep -x sshd >/dev/null && ok "sshd process" "pid $(pgrep -x sshd | head -1)" \
                         || bad "sshd process" "沒跑 —— 跑 00_bootstrap.sh"
if command -v ss >/dev/null && ss -tln 2>/dev/null | grep -q ':22 '; then
    ok "listening :22" "$(ss -tln | awk '/:22 /{print $4}' | tr '\n' ' ')"
else
    bad "listening :22" "沒有 listen"
fi
if [ -f /root/.ssh/authorized_keys ]; then
    ok "authorized_keys" "$(ssh-keygen -lf /root/.ssh/authorized_keys 2>/dev/null | awk '{print $2}')"
else
    bad "authorized_keys" "不存在 —— 跑 install_pubkey.sh"
fi
grep -qs 'PasswordAuthentication no' /etc/ssh/sshd_config.d/99-tiara.conf \
    && ok "只允許金鑰登入" "PasswordAuthentication no" \
    || warn "只允許金鑰登入" "99-tiara.conf 不見了，sshd 可能吃預設值"

echo; echo "[4] 資源配額（🔴 用 cgroup，不要用 nproc/MemTotal）"
CPUQ=$(awk '{if($1=="max") print "unlimited"; else printf "%.0f", $1/$2}' /sys/fs/cgroup/cpu.max 2>/dev/null)
MEMQ=$(awk '{printf "%.1f GiB", $1/1073741824}' /sys/fs/cgroup/memory.max 2>/dev/null)
SHM=$(df -h /dev/shm | awk 'NR==2{print $2}')
ok "cpu 配額"    "$CPUQ  (host 看到的是 $(nproc) —— 不要用這個)"
ok "memory 配額" "$MEMQ  (host 看到的是 $(awk '/MemTotal/{printf "%.0f GiB", $2/1048576}' /proc/meminfo))"
if [ "${SHM%G}" -ge 16 ] 2>/dev/null; then ok "/dev/shm" "$SHM"; else bad "/dev/shm" "$SHM 太小，vLLM 會掛"; fi

echo; echo "[4b] 模型 / 快取路徑（必須在會存活的 Data 上）"
# shellcheck disable=SC1091
[ -f /etc/profile.d/99-tiara.sh ] && source /etc/profile.d/99-tiara.sh
if [ "${HF_HOME:-}" = "/mlsteam/data/tiara/hf-cache" ]; then
    ok "HF_HOME" "$HF_HOME  ($(du -sh "$HF_HOME" 2>/dev/null | cut -f1) 已用)"
else
    bad "HF_HOME" "${HF_HOME:-未設定} —— 模型會下載到 Lab 一關就沒的地方"
fi
[ -d /mlsteam/data/tiara/models ] && ok "models 目錄" "/mlsteam/data/tiara/models" \
                                  || warn "models 目錄" "不存在"

echo; echo "[4c] Claude Code（跨 Lab 存活，不用重裝/重登入）"
CLAUDE_BIN=/mlsteam/workspace/npm-global/bin/claude
NODE_BIN=$(ls -d /mlsteam/workspace/node/*/bin/node 2>/dev/null | tail -1)
if [ -x "$NODE_BIN" ]; then
    ok "node" "$("$NODE_BIN" --version) ($(dirname "$(dirname "$NODE_BIN")"))"
else
    warn "node" "workspace 裡沒有 node（跑 40_claude_code.sh）"
fi
if [ -x "$CLAUDE_BIN" ]; then
    ok "claude" "$(PATH=$(dirname "$NODE_BIN"):$PATH "$CLAUDE_BIN" --version 2>&1 | head -1)"
else
    warn "claude" "未安裝（跑 40_claude_code.sh）"
fi
[ -d /mlsteam/workspace/.claude ] && ok "CLAUDE_CONFIG_DIR" "/mlsteam/workspace/.claude（登入狀態存這）" \
                                  || warn "CLAUDE_CONFIG_DIR" "不存在"
[ -d /mlsteam/workspace/worktrees ] && ok "worktrees" "/mlsteam/workspace/worktrees" \
                                    || bad "worktrees" "不存在 —— Claude Code SSH 會報 no such file"

echo; echo "[5] GPU"
if command -v rocm-smi >/dev/null; then
    VRAM=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/VRAM Total Memory/{print $NF}')
    ok "rocm-smi" "$(rocm-smi --showproductname 2>/dev/null | awk -F': *' '/Card Series/{print $NF; exit}')"
    if [ -n "$VRAM" ]; then
        ok "VRAM" "$(awk -v b="$VRAM" 'BEGIN{printf "%.1f GiB", b/1073741824}')  ($VRAM B)"
    else
        bad "VRAM" "問不出來"
    fi
    ok "partition"  "compute=$(rocm-smi --showcomputepartition 2>/dev/null | awk -F': *' '/Compute Partition:/{print $NF;exit}') memory=$(rocm-smi --showmemorypartition 2>/dev/null | awk -F': *' '/Memory Partition:/{print $NF;exit}')"
    ok "gfx arch"   "$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | head -1)"
else
    bad "rocm-smi" "不在 PATH"
fi

echo; echo "[6] Python / PyTorch"
if [ -f "$VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    ok "venv" "$VENV"
    # ⚠ 不要用 2>&1 把 stderr 混進來：ROCm 會在 stderr 印
    #   "/opt/amdgpu/share/libdrm/amdgpu.ids: No such file or directory"（無害但很吵），
    #   混進去之後 `read` 會讀到那行警告而不是資料，把好的環境誤判成壞的。
    #   → stderr 另外收，只在失敗時才拿出來看；stdout 只取最後一行。
    ERRF=$(mktemp)
    OUT=$(python - 2>"$ERRF" <<'PYT' | tail -1
import torch
print(f"{torch.__version__}|{torch.version.hip}|{torch.version.cuda}|{torch.cuda.is_available()}|{torch.cuda.device_count()}|{torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0}")
PYT
)
    IFS='|' read -r TV THIP TCU TAV TCNT TMEM <<< "$OUT"
    if [ -z "$TV" ]; then
        bad "torch 匯入" "stdout 空的；stderr: $(head -3 "$ERRF" | tr '\n' ' ')"
    fi
    if [ -s "$ERRF" ]; then
        warn "torch stderr（不影響判定）" "$(head -1 "$ERRF")"
    fi
    rm -f "$ERRF"
    if [ "$THIP" != "None" ] && [ -n "$THIP" ]; then ok "torch ROCm build" "$TV  hip=$THIP"; else bad "torch ROCm build" "$OUT"; fi
    [ "$TCU" = "None" ]  && ok "不是 CUDA build" "torch.version.cuda=None" || bad "不是 CUDA build" "cuda=$TCU"
    [ "$TAV" = "True" ]  && ok "torch 看得到 GPU" "device_count=$TCNT" || bad "torch 看得到 GPU" "is_available=$TAV"
    [ -n "$TMEM" ] && [ "$TMEM" != "0" ] && ok "torch 回報 VRAM" "$(awk -v b="$TMEM" 'BEGIN{printf "%.1f GiB", b/1073741824}')"
else
    bad "venv" "$VENV 不存在 —— 跑 10_python_stack.sh"
fi

echo; echo "[7] vLLM（選配，還沒建不算失敗）"
if [ -f "$WS/.vllm_commit" ]; then
    python -c "import vllm; print(vllm.__version__)" >/dev/null 2>&1 \
        && ok "vllm" "$(python -c 'import vllm;print(vllm.__version__)') @ $(cut -c1-8 "$WS/.vllm_commit")" \
        || bad "vllm" "有 .vllm_commit 但 import 不到"
    python -c "from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import OffloadingConnector" 2>/dev/null \
        && ok "OffloadingConnector" "import OK（A3 驗收條件）" \
        || bad "OffloadingConnector" "import 失敗 —— 這是 A3 驗收條件，要回報"
else
    warn "vllm" "尚未建置（bash $WS/bin/20_build_vllm.sh，約 40–90 分鐘）"
fi

echo; echo "[8] 監控工具"
for t in rocm-smi amd-smi nvtop radeontop btop htop iostat; do
    command -v "$t" >/dev/null && PASS=$((PASS+1)) || warn "$t" "未安裝（跑 15_monitoring.sh）"
done
ok "監控工具" "$(for t in rocm-smi amd-smi nvtop radeontop amdgpu_top btop htop iostat gpuwatch gpulog; do command -v $t >/dev/null && printf '%s ' $t; done)"

echo
echo "======================================================================"
if [ "$FAIL" -eq 0 ]; then
    printf ' \033[32m全部通過\033[0m  (%d 項檢查)\n' "$PASS"
else
    printf ' \033[31m%d 項失敗\033[0m  / %d 項通過\n' "$FAIL" "$PASS"
    echo ' 照 CLAUDE.md §1 規則 2：記錄完整錯誤、停下來回報，不要換方法硬幹。'
fi
echo "======================================================================"
exit "$FAIL"
