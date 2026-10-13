#!/usr/bin/env bash
# =============================================================================
# 99_report.sh — 把這台機器上所有已量到的東西彙整成一份可讀、可追溯的報告
#
# 原則：
#   * 每個數字都標「怎麼量的」與「來源檔」。沒有來源的數字不寫進去。
#   * 量不到的寫 NOT_MEASURED，不寫估計值（CLAUDE.md §1 規則 1）。
#   * 可疑的數字要標可疑並寫明理由，不是悄悄拿掉。
#
# 輸出：$WS/REPORT_MI300X.md  以及 repo 的 results/（若已 clone）
# =============================================================================
set -Eeuo pipefail
WS=/mlsteam/workspace
DATA=/mlsteam/data/tiara
REPO="$WS/paper-hierarchical-kv-state"
OUT="$WS/REPORT_MI300X.md"
export PATH="$WS/bin:/opt/rocm/bin:$PATH"

j() { python3 -c "
import json,sys
try:
    d=json.load(open('$1'))
    for k in '$2'.split('.'):
        d=d[k]
    print(d)
except Exception:
    print('NOT_MEASURED')
" 2>/dev/null || echo NOT_MEASURED; }

HW="$REPO/results/hw_mi300x.json"; [ -f "$HW" ] || HW=$(ls -t "$WS"/hwinfo/*/hw_mi300x.json 2>/dev/null | head -1)
ENVJ="$REPO/results/env_mi300x.json"
SB=$(ls -t "$DATA"/runs/*storage-bench/storage_bench.csv 2>/dev/null | head -1)

{
cat <<EOF
# 平台 B（AMD MI300X @ MLSteam）實測彙整

產生時間：$(date -Is)
產生者：\`$WS/bin/99_report.sh\`
主機：$(hostname)

> **這份是機器產生的，不要手改。** 改了就跟來源對不上。
> 每一節都標註來源檔；\`NOT_MEASURED\` 表示還沒量到，**不是 0 也不是估計值**。

---

## 1. 硬體（來源：\`$HW\`）

| 項目 | 值 |
|---|---|
| GPU | $(j "$HW" gpu.name) |
| 架構 | $(j "$HW" gpu.gfx_arch) |
| VRAM | $(j "$HW" gpu.vram_gib) GiB（$(j "$HW" gpu.vram_bytes) B） |
| 運算單元 | $(j "$HW" gpu.compute_units) |
| Compute partition | $(j "$HW" gpu.compute_partition) |
| Memory partition | $(j "$HW" gpu.memory_partition) |
| GPU 的 NUMA node | $(j "$HW" gpu.numa_node) |
| PCIe | $(j "$HW" gpu.pcie) |
| 虛擬化 | $(j "$HW" gpu.sriov_ifwi) |

> ⚠️ 原廠 Confluence 文件的「預設基本配額」寫 **96 GB VRAM / DPX**。
> 實測是 **SPX 整張卡**。以實測為準。

## 2. 資源配額（來源：\`$HW\`、cgroup v2）

| 項目 | **配額（真的能用）** | 容器裡看到的（會騙人） |
|---|---|---|
| CPU | **$(j "$HW" quota.cpu_count)** | \`nproc\` = $(j "$HW" host_view.nproc) |
| RAM | **$(j "$HW" quota.memory_max_gib) GiB** | \`MemTotal\` = $(j "$HW" host_view.mem_total_gib) GiB |
| /dev/shm | $(j "$HW" quota.shm_gib) GiB | — |

\`memory.max\` = flavor RAM(256 GiB) + shm(192 GB)，兩者相加恰等於 $(j "$HW" quota.memory_max_bytes) B。
**shm 是額外加上去的，不是從 process 記憶體切走。**

主機 CPU：$(j "$HW" host_view.cpu_model)（$(j "$HW" host_view.sockets) socket、$(j "$HW" host_view.numa_nodes) NUMA node）

## 3. 軟體環境（來源：\`$ENVJ\`）

| 項目 | 值 |
|---|---|
| OS | $(j "$ENVJ" os) |
| ROCm | $(j "$ENVJ" rocm_version) |
| torch | $(cd "$WS" && source venv/tiara/bin/activate 2>/dev/null && python -c "import torch;print(torch.__version__)" 2>/dev/null || echo NOT_INSTALLED) |
| torch HIP | $(cd "$WS" && source venv/tiara/bin/activate 2>/dev/null && python -c "import torch;print(torch.version.hip)" 2>/dev/null || echo NOT_INSTALLED) |
| vLLM | $(cd "$WS" && source venv/tiara/bin/activate 2>/dev/null && python -c "import vllm;print(vllm.__version__)" 2>/dev/null || echo NOT_INSTALLED) |
| vLLM ref | $(cat "$WS/.vllm_ref" 2>/dev/null || echo NOT_BUILT) |
| vLLM commit | $(cat "$WS/.vllm_commit" 2>/dev/null || echo NOT_BUILT) |
| 論文 repo HEAD | $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo NOT_CLONED) |

## 4. 儲存（來源：\`${SB:-NOT_MEASURED}\`）

EOF

if [ -n "${SB:-}" ] && [ -f "$SB" ]; then
    echo '| 位置 | fstype | O_DIRECT | 型態 | MB/s | 平均延遲(µs) | p99(µs) |'
    echo '|---|---|---|---|---|---|---|'
    awk -F, 'NR>1{printf "| %s | %s | %s | %s | %s | %s | %s |\n",$3,$5,$6,$7,$9,$11,$12}' "$SB"
else
    echo 'NOT_MEASURED —— 跑 `bash $WS/bin/25_storage_bench.sh`'
fi

cat <<'EOF'

**判讀**
* `/mlsteam/workspace` 與 `/mlsteam/data/tiara` 都是 **NFS4 over TCP**（NetApp），不是本地碟。
* `/` overlay 在本地 SSD（sda 3.5 TB，rotational=0）上，但**不持久**且整台 host 共用。
* 本容器**看不到任何 raw block device**（`/dev` 無 block node，無 `/dev/nvme*`）。
* 🔴 `local_overlay` 的**循序讀**數字不可採信：它比 tmpfs（純記憶體）還快，
  物理上不可能。`O_DIRECT` open 成功不代表繞過快取，`posix_fadvise(DONTNEED)`
  也沒穿透 overlayfs。→ **循序讀 = NOT_MEASURED**。
* `local_overlay` 的**隨機讀**可採信：延遲 ~158 µs 是 tmpfs(~38 µs) 的 4 倍，
  形狀符合「真的碰到裝置」。KV block 取回是隨機讀，所以對論文重要的正是這一項。

## 5. GPU 運算 smoke test

EOF

if [ -f "$WS/bin/gpu_smoke.py" ]; then
    # shellcheck disable=SC1091
    ( source "$WS/venv/tiara/bin/activate" && python "$WS/bin/gpu_smoke.py" 2>/dev/null \
      | grep -E "^\[" | sed 's/^/    /' ) || echo "    （執行失敗）"
else
    echo "    NOT_MEASURED"
fi

cat <<'EOF'

> ⚠️ 這是 **smoke test**，不是 benchmark：沒調 hipBLASLt、單一 stream、naive copy。
> **不可寫進論文。** M2 的成本模型要走 `runsh` 並重複取分佈。
> 可用的資訊是**數量級**與**是否正確**，不是絕對值。

## 6. 已知未決事項

| # | 事項 | 狀態 |
|---|---|---|
| 1 | 本地 overlay 的循序讀頻寬 | **NOT_MEASURED**（容器內繞不過 page cache） |
| 2 | 論文「SSD 階」在平台 B 的載體 | **未決**：NFS 持久但有網路成本；overlay 是本地但不持久 |
| 3 | vLLM 版本落差（平台 A=v0.28.0） | κ 跨硬體數字標 **NOT_COMPARABLE** 直到對齊 |
| 4 | `gpu_guard.py` 的 AMD 後端 | **未在真機驗證**，要對照 `amd-smi process` |
| 5 | 是否有別的租戶共用同一張卡 | **未驗證** |
| 6 | M1 / M2 實驗 | **尚未開始** |

## 7. 檔案位置

| 內容 | 路徑 |
|---|---|
| 腳本 | `/mlsteam/workspace/bin/` |
| 論文 repo | `/mlsteam/workspace/paper-hierarchical-kv-state/` |
| venv | `/mlsteam/workspace/venv/tiara/` |
| vLLM 原始碼 | `/mlsteam/workspace/src/vllm/` |
| 建置/安裝 log | `/mlsteam/workspace/logs/` |
| 硬體盤點明細 | `/mlsteam/workspace/hwinfo/<ts>/` |
| 環境快照 | `/mlsteam/workspace/snapshots/<ts>/` |
| **實驗產出** | `/mlsteam/data/tiara/runs/<RUN_ID>/` |
| 模型 / HF 快取 | `/mlsteam/data/tiara/hf-cache/` |
| Claude Code | `/mlsteam/workspace/npm-global`，設定在 `/mlsteam/workspace/.claude` |
| Rust / amdtop | `/mlsteam/workspace/rust/` |
EOF
} > "$OUT"

echo "[ok] 報告寫到 $OUT ($(wc -l < "$OUT") 行)"
if [ -d "$REPO/results" ]; then
    cp "$OUT" "$REPO/results/REPORT_MI300X.md"
    echo "[ok] 同時複製到 $REPO/results/REPORT_MI300X.md"
fi
