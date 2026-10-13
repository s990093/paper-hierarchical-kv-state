# 平台 B（AMD MI300X @ MLSteam）實測彙整

產生時間：2026-09-15T08:29:33+00:00
產生者：`/mlsteam/workspace/bin/99_report.sh`
主機：lab-u034a318-f7c8b749d-fwzpg

> **這份是機器產生的，不要手改。** 改了就跟來源對不上。
> 每一節都標註來源檔；`NOT_MEASURED` 表示還沒量到，**不是 0 也不是估計值**。

---

## 1. 硬體（來源：`/mlsteam/workspace/paper-hierarchical-kv-state/results/hw_mi300x.json`）

| 項目 | 值 |
|---|---|
| GPU | AMD Instinct MI300X |
| 架構 | gfx942 |
| VRAM | 192.0 GiB（206141652992 B） |
| 運算單元 | 304 |
| Compute partition | SPX |
| Memory partition | NPS1 |
| GPU 的 NUMA node | 1 |
| PCIe | MAX_PCIE_WIDTH: 16         MAX_PCIE_SPEED: 32 GT/s         PCIE_INTERFACE_VERSION: Gen 5 |
| 虛擬化 | AMD MI300X_HW_SRIOV_CVS_1VF |

> ⚠️ 原廠 Confluence 文件的「預設基本配額」寫 **96 GB VRAM / DPX**。
> 實測是 **SPX 整張卡**。以實測為準。

## 2. 資源配額（來源：`/mlsteam/workspace/paper-hierarchical-kv-state/results/hw_mi300x.json`、cgroup v2）

| 項目 | **配額（真的能用）** | 容器裡看到的（會騙人） |
|---|---|---|
| CPU | **32.0** | `nproc` = 192 |
| RAM | **434.8 GiB** | `MemTotal` = 2267.5 GiB |
| /dev/shm | 178.8 GiB | — |

`memory.max` = flavor RAM(256 GiB) + shm(192 GB)，兩者相加恰等於 466877906944 B。
**shm 是額外加上去的，不是從 process 記憶體切走。**

主機 CPU：AMD EPYC 9684X 96-Core Processor（2 socket、2 NUMA node）

## 3. 軟體環境（來源：`/mlsteam/workspace/paper-hierarchical-kv-state/results/env_mi300x.json`）

| 項目 | 值 |
|---|---|
| OS | Ubuntu 24.04.4 LTS |
| ROCm | 7.2.2 |
| torch | 2.10.0+rocm7.0 |
| torch HIP | 7.0.51831 |
| vLLM | 0.19.1 |
| vLLM ref | v0.19.1 |
| vLLM commit | b1388b1fbf5aaef47937fabe98931211684666a6 |
| 論文 repo HEAD | 2702df6 |

## 4. 儲存（來源：`/mlsteam/data/tiara/runs/20260915-074854-storage-bench/storage_bench.csv`）

| 位置 | fstype | O_DIRECT | 型態 | MB/s | 平均延遲(µs) | p99(µs) |
|---|---|---|---|---|---|---|
| nfs_data | nfs4 | yes | seqwrite | 1071.6 | 15339.9 | 20316.2 |
| nfs_data | nfs4 | yes | seqread | 643.9 | 25754.3 | 115867.6 |
| nfs_data | nfs4 | yes | randwrite | 91.8 | 704.9 | 1073.2 |
| nfs_data | nfs4 | yes | randread | 61.4 | 1057.8 | 11075.6 |
| nfs_workspace | nfs4 | yes | seqwrite | 684.3 | 24192.1 | 154140.7 |
| nfs_workspace | nfs4 | yes | seqread | 639.1 | 25955.6 | 143654.9 |
| nfs_workspace | nfs4 | yes | randwrite | 93.7 | 690.2 | 1036.3 |
| nfs_workspace | nfs4 | yes | randread | 62.8 | 1034.0 | 10420.2 |
| local_overlay | overlay | yes | seqwrite | 2388.7 | 6802.3 | 7569.4 |
| local_overlay | overlay | yes | seqread | 10737.4 | 1438.6 | 3424.3 |
| local_overlay | overlay | yes | randwrite | 525.7 | 118.1 | 238.6 |
| local_overlay | overlay | yes | randread | 391.4 | 158.0 | 264.2 |
| tmpfs_shm | tmpfs | yes | seqwrite | 8073.2 | 1937.7 | 2375.7 |
| tmpfs_shm | tmpfs | yes | seqread | 8589.9 | 1790.3 | 2310.1 |
| tmpfs_shm | tmpfs | yes | randwrite | 1508.1 | 41.4 | 47.9 |
| tmpfs_shm | tmpfs | yes | randread | 1624.4 | 38.4 | 45.3 |

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

    [matmul  ] bf16 8192^3       1.64 ms/iter     670.9 TFLOPS   (原廠標稱 BF16 1307.4)
    [correct ] max|I@Y - Y| = 0.000e+00   -> OK
    [HBM d2d ]   1164.0 GB/s   (原廠標稱 5300)
    [H2D pin ]     51.9 GB/s   (PCIe Gen5 x16 理論 ~63)
    [D2H pin ]     48.6 GB/s
    [mem     ] allocated=2.5 GiB  reserved=2.6 GiB

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
