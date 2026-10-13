#!/usr/bin/env bash
# =============================================================================
# 05_hwinfo.sh — 硬體盤點：GPU / CPU / NUMA / 記憶體 / 儲存 / 網路 / PCIe
#
# 🔴 這支腳本存在的理由：**容器裡看到的數字會騙人。**
#    `nproc`、`lscpu`、`/proc/meminfo`、`numactl -H` 回報的是**整台 host**，
#    不是這個 pod 的 cgroup 配額。直接拿去設 worker 數會超賣 6 倍。
#    所以所有「我們真正能用多少」的欄位一律從 /sys/fs/cgroup 讀。
#
# 輸出：$WS/hwinfo/<ts>/  （純文字，可 diff）
#      + results/hw_mi300x.json（摘要，進 git）
# =============================================================================
set -Eeuo pipefail
trap 'echo "[FATAL] 05_hwinfo.sh line $LINENO 失敗，exit=$? —— 記錄後停止，不要略過" >&2' ERR

# 可選指令的包裝：工具不在就記一行，不要讓整支腳本 127 掛掉。
# （原本沒有這層，結果 `ip` 沒裝就把整支腳本打死，而且錯誤被重導進檔案看不到。）
opt() { if command -v "$1" >/dev/null 2>&1; then "$@"; else echo "(缺少指令: $1)"; fi; }

WS=/mlsteam/workspace
OUT="$WS/hwinfo/$(date +%Y%m%d-%H%M%S)-$(hostname)"
mkdir -p "$OUT"
export PATH=/opt/rocm/bin:$PATH

echo "=== 05_hwinfo.sh @ $(date -Is) -> $OUT ==="

# --- GPU ---------------------------------------------------------------------
{ rocm-smi --showallinfo 2>&1 || true; }            > "$OUT/rocm-smi-all.txt"
{ rocm-smi --showtoponuma --showtopo 2>&1 || true; } > "$OUT/rocm-smi-topo.txt"
{ rocminfo 2>&1 || true; }                          > "$OUT/rocminfo.txt"
{ amd-smi static 2>&1 || true; }                    > "$OUT/amd-smi-static.txt"
{ amd-smi list 2>&1 || true; }                      > "$OUT/amd-smi-list.txt"
{ amd-smi process 2>&1 || true; }                   > "$OUT/amd-smi-process.txt"

# --- CPU / NUMA（host 視角，僅供參考）-----------------------------------------
{ lscpu; echo; lscpu -e 2>/dev/null; }              > "$OUT/lscpu.txt" 2>&1
{ opt numactl -H; }                                 > "$OUT/numa.txt" 2>&1
cat /proc/cpuinfo                                    > "$OUT/cpuinfo.txt"
cat /proc/meminfo                                    > "$OUT/meminfo.txt"

# --- cgroup（**我們真正的配額**）----------------------------------------------
{
  echo "# cgroup v2 —— 這才是這個 pod 真正能用的資源"
  for f in memory.max memory.high memory.current cpu.max cpu.stat pids.max io.max; do
      printf '%-16s %s\n' "$f" "$(cat /sys/fs/cgroup/$f 2>/dev/null | tr '\n' ' ')"
  done
} > "$OUT/cgroup.txt"

# --- 儲存 --------------------------------------------------------------------
{
  echo "# ---- df ----";        df -hT
  echo; echo "# ---- mount ----"; mount | grep -vE '^(proc|sysfs|devpts|cgroup|mqueue|bpf)'
  echo; echo "# ---- /sys/block ----"
  for b in /sys/block/*; do
      n=$(basename "$b")
      case "$n" in loop*) continue;; esac
      printf '%-8s rotational=%s size=%sGB model=%s\n' "$n" \
        "$(cat "$b/queue/rotational" 2>/dev/null)" \
        "$(( $(cat "$b/size" 2>/dev/null || echo 0) * 512 / 1000000000 ))" \
        "$(cat "$b/device/model" 2>/dev/null | tr -d ' ')"
  done
  echo; echo "# ---- NVMe ----"
  ls -l /dev/nvme* 2>/dev/null || echo "(這個 pod 看不到 raw NVMe 裝置)"
} > "$OUT/storage.txt" 2>&1

# --- 網路 --------------------------------------------------------------------
{ opt ip -br addr; echo; opt ip -br link; echo; opt ss -tlnp; } > "$OUT/net.txt" 2>&1

# --- 摘要 JSON（進 git 的那份）------------------------------------------------
REPO="$WS/paper-hierarchical-kv-state"
# 只有在 repo 真的是 git checkout 時才寫進去。
# （原本無條件 mkdir -p，結果憑空造出一個非 git 的目錄，害 30_project.sh 的
#   git clone 撞「destination path already exists」。副作用要想清楚。）
if [ -d "$REPO/.git" ]; then
    JSON_OUT="${HWINFO_JSON:-$REPO/results/hw_mi300x.json}"
    mkdir -p "$(dirname "$JSON_OUT")"
else
    JSON_OUT="${HWINFO_JSON:-$OUT/hw_mi300x.json}"
    echo "[info] repo 還沒 clone，摘要先寫到 $JSON_OUT"
fi

python3 - "$JSON_OUT" "$OUT" <<'PYHW'
import json, os, subprocess, sys, datetime, re

def sh(c):
    try:
        return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"ERROR: {e!r}"

def cg(name, cast=str):
    try:
        v = open(f"/sys/fs/cgroup/{name}").read().strip()
        return cast(v) if cast is not str else v
    except Exception:
        return None

out = {
    "ts": datetime.datetime.now().astimezone().isoformat(),
    "platform": "B_MI300X_MLSteam",
    "hostname": os.uname().nodename,
    "_note": "host_view 的欄位是整台機器，不是本 pod 的配額；要用 quota 那一組。",
}

# --- 真正的配額 ---
mem_max = cg("memory.max")
cpu_max = cg("cpu.max") or ""
m = re.match(r"(\d+)\s+(\d+)", cpu_max)
out["quota"] = {
    "memory_max_bytes": int(mem_max) if mem_max and mem_max != "max" else None,
    "memory_max_gib": round(int(mem_max)/2**30, 1) if mem_max and mem_max != "max" else None,
    "cpu_count": (int(m.group(1))/int(m.group(2))) if m else None,
    "shm_bytes": int(sh("df -B1 --output=size /dev/shm | tail -1") or 0),
}
out["quota"]["shm_gib"] = round(out["quota"]["shm_bytes"]/2**30, 1)

out["host_view"] = {
    "nproc": os.cpu_count(),
    "cpu_model": sh("lscpu | awk -F: '/Model name/{gsub(/^ +/,\"\",$2);print $2;exit}'"),
    "sockets": sh("lscpu | awk -F: '/^Socket/{gsub(/ /,\"\",$2);print $2}'"),
    "numa_nodes": sh("lscpu | awk -F: '/NUMA node\\(s\\)/{gsub(/ /,\"\",$2);print $2}'"),
    "mem_total_gib": round(int(sh("awk '/MemTotal/{print $2}' /proc/meminfo") or 0)/2**20, 1),
}

vram_b = sh("rocm-smi --showmeminfo vram 2>/dev/null | awk '/VRAM Total Memory/{print $NF}'")
out["gpu"] = {
    "name":        sh("rocm-smi --showproductname 2>/dev/null | awk -F': *' '/Card Series/{print $NF; exit}'"),
    "gfx_arch":    sh("rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | head -1"),
    "vram_bytes":  int(vram_b) if vram_b.isdigit() else None,
    "vram_gib":    round(int(vram_b)/2**30, 1) if vram_b.isdigit() else None,
    "compute_units": sh("amd-smi static -g 0 2>/dev/null | awk '/NUM_COMPUTE_UNITS/{print $NF}'"),
    # 用專門的旗標，不要從 concise 表格數欄位（欄位會隨韌體/版本移位）
    "compute_partition": sh("rocm-smi --showcomputepartition 2>/dev/null | awk -F': *' '/Compute Partition:/{print $NF; exit}'"),
    "memory_partition":  sh("rocm-smi --showmemorypartition  2>/dev/null | awk -F': *' '/Memory Partition:/{print $NF; exit}'"),
    # 冒號要寫進 pattern，否則會match到 '===== Numa Nodes =====' 那條標題線
    "numa_node":   sh("rocm-smi --showtoponuma 2>/dev/null | awk -F': *' '/Numa Node:/{print $NF; exit}'"),
    "pcie":        sh("amd-smi static -g 0 2>/dev/null | awk '/MAX_PCIE_SPEED|PCIE_INTERFACE_VERSION|MAX_PCIE_WIDTH/{print}' | tr '\\n' ' '"),
    "sriov_ifwi":  sh("amd-smi static -g 0 2>/dev/null | awk -F': *' '/NAME: AMD MI300X/{print $NF; exit}'"),
}

out["storage"] = {
    "raw_nvme_visible": bool(sh("ls /dev/nvme* 2>/dev/null")),
    "mounts": [],
}
for line in sh("df -BG --output=source,fstype,size,avail,target -x tmpfs -x overlay 2>/dev/null").splitlines()[1:]:
    p = line.split()
    if len(p) >= 5:
        out["storage"]["mounts"].append(dict(zip(("source","fstype","size","avail","target"), p[:5])))
out["storage"]["root_overlay"] = sh("df -BG --output=size,avail,fstype / | tail -1").strip()
out["storage"]["root_backing_rotational"] = sh("cat /sys/block/sda/queue/rotational 2>/dev/null")
out["storage"]["root_backing_model"] = sh("cat /sys/block/sda/device/model 2>/dev/null")

p = sys.argv[1]
json.dump(out, open(p, "w"), indent=2, ensure_ascii=False)
print(f"[ok] 摘要寫到 {p}")
print(json.dumps({"quota": out["quota"], "gpu": out["gpu"]}, indent=2, ensure_ascii=False))
PYHW

echo
echo "=== 05_hwinfo.sh 完成，明細在 $OUT ==="
ls "$OUT"
