#!/usr/bin/env bash
# =============================================================================
# 30_project.sh — 取得論文 repo、建立目錄骨架、量環境指紋
# =============================================================================
set -Eeuo pipefail
WS=/mlsteam/workspace
DATA=${TIARA_DATA:-/mlsteam/data/tiara}
REPO_URL=https://github.com/s990093/paper-hierarchical-kv-state.git
REPO=$WS/paper-hierarchical-kv-state
LOG_DIR="$WS/logs"; mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d-%H%M%S)
exec > >(tee -a "$LOG_DIR/project-$STAMP.log") 2>&1
trap 'echo "[FATAL] line $LINENO 失敗，exit=$?" >&2' ERR

export PATH=/opt/rocm/bin:$PATH
echo "=== 30_project.sh @ $(date -Is) ==="

# --- repo（public，不需要憑證）-------------------------------------------------
# NFS 的 uid squash → git 會喊 dubious ownership。單獨跑這支也要能過。
# --system 而非 --global：網頁 Terminal 與 SSH 的 HOME 不同（見 00_bootstrap.sh 註解）
for d in "$WS" "$REPO" "$WS/src/vllm" "$DATA"; do
    git config --system --get-all safe.directory 2>/dev/null | grep -qxF "$d" \
        || git config --system --add safe.directory "$d"
done

# 三種狀態要分開處理。「已存在但不是 git」那種**不能**用 git init 硬接管：
# init 出來的 repo 沒有任何 commit，HEAD 是空的，後面每個 git 指令都會怪怪的
# （上一版就是這樣，pull 噴 "no tracking information"、HEAD 顯示 UNKNOWN）。
# 正解：clone 到旁邊，再把既有檔案合併進去。
if [ -d "$REPO/.git" ] && git -C "$REPO" rev-parse HEAD >/dev/null 2>&1; then
    git -C "$REPO" pull --ff-only || echo "[warn] pull 失敗（可能有本地改動），沿用現有 checkout"
else
    TMP="$WS/.repo-clone-$$"
    rm -rf "$TMP"
    echo "[info] clone $REPO_URL -> $TMP"
    git clone "$REPO_URL" "$TMP"
    git config --system --get-all safe.directory | grep -qxF "$TMP" || git config --system --add safe.directory "$TMP"
    if [ -d "$REPO" ]; then
        echo "[info] $REPO 已存在（半殘的 git 或只有 results/）—— 保留裡面的檔案再合併"
        rm -rf "$REPO/.git"
        # --skip-old-files：repo 帶來的版本優先，我們自己產的檔案（results/*.json）才補進去
        ( cd "$REPO" && tar cf - . ) | ( cd "$TMP" && tar xf - --skip-old-files )
        rm -rf "$REPO"
    fi
    mv "$TMP" "$REPO"
fi
echo "[ok] repo HEAD = $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || git -C "$REPO" rev-parse --short FETCH_HEAD 2>/dev/null || echo UNKNOWN)"

# --- 目錄骨架 ------------------------------------------------------------------
# 大檔案一律放掛載的 Data（跨 Lab 存活且不吃 workspace 的 inode）；
# 小的、可版控的結果放 repo 的 results/（跟平台 A 的規則一致，CLAUDE.md §2）
if [ ! -d "$DATA" ]; then
    echo "[FATAL] 掛載點 $DATA 不存在。" >&2
    echo "        請到網頁 Lab → 🧰 資料夾 → 把 tiara 這個 Data 資料夾打開 → 套用。" >&2
    echo "        （套用會重開 Lab，環境會不見 —— 這正是所有東西都要寫成腳本的原因）" >&2
    exit 1
fi
mkdir -p "$DATA"/{runs,models,datasets,hf-cache,profiles}
mkdir -p "$WS"/{logs,src,venv,bin}
echo "[ok] 目錄骨架就緒"
df -h "$WS" "$DATA"

# --- 環境指紋 ------------------------------------------------------------------
FP="$REPO/results/env_mi300x.json"
mkdir -p "$(dirname "$FP")"
python3 - "$FP" <<'PYFP'
import json, os, subprocess, sys, platform, datetime
out = {"ts": datetime.datetime.now().astimezone().isoformat(), "platform": "B_MI300X_MLSteam"}
def run(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"ERROR: {e!r}"
out["hostname"]      = platform.node()
out["kernel"]        = run("uname -a")
out["os"]            = run(". /etc/os-release && echo $PRETTY_NAME")
out["rocm_version"]  = run("cat /opt/rocm/.info/version 2>/dev/null")
out["rocm_smi"]      = run("rocm-smi --showproductname --csv 2>/dev/null")
out["gfx_arch"]      = run("rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | head -1")
# 🔴 os.cpu_count() 與 /proc/meminfo 在容器裡回報的是**整台 host**，不是本 pod
#    的配額。這台 host 是 192 threads / 2.2 TB，我們實際只有 32 core / 256 GiB。
#    直接拿 host 的數字去設 worker 數 = 超賣 6 倍，量出來的時間全部不能用。
out["cpu_count_host_view"] = os.cpu_count()
out["mem_total_host_view"] = run("grep MemTotal /proc/meminfo")
_cpumax = run("cat /sys/fs/cgroup/cpu.max")
try:
    _q, _per = _cpumax.split()[:2]
    out["cpu_quota"] = None if _q == "max" else int(_q) / int(_per)
except Exception:
    out["cpu_quota"] = None
_memmax = run("cat /sys/fs/cgroup/memory.max")
out["mem_quota_bytes"] = int(_memmax) if _memmax.isdigit() else None
out["shm_bytes"] = int(run("df -B1 --output=size /dev/shm | tail -1") or 0)
out["amd_smi"]       = run("amd-smi list 2>/dev/null | head -40")
try:
    import torch
    out["torch"] = {"version": torch.__version__, "hip": torch.version.hip,
                    "cuda": torch.version.cuda,
                    "available": torch.cuda.is_available(),
                    "device_count": torch.cuda.device_count()}
except Exception as e:
    out["torch"] = f"NOT_INSTALLED ({e!r})"
p = sys.argv[1]
json.dump(out, open(p, "w"), indent=2, ensure_ascii=False)
print(f"[ok] 環境指紋寫到 {p}")
print(json.dumps({k: out[k] for k in
      ("os","rocm_version","gfx_arch","cpu_quota","cpu_count_host_view","mem_quota_bytes","shm_bytes")},
      indent=2, ensure_ascii=False))
PYFP

echo "=== 30_project.sh 完成 @ $(date -Is) ==="
