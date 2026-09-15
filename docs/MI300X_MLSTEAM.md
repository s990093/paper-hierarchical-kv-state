# 平台 B 操作手冊 — AMD MI300X on MLSteam (Manta)

> 這份是**操作手冊**，寫給要在 MI300X 上跑 Tiara 實驗的人（含 agent）。
> 平台原廠教學（含截圖）在 ITRI Confluence 的「MLSteam 教學 (Manta)」外部分享頁；
> 這份把它**濃縮成本專案實際用得到的路徑**，並補上原廠文件沒講、我們踩過的坑。
>
> 論文主張看 `main.tex`；實驗計畫看 `EXPERIMENT_PLAN.md`；平台 A（3090）看 `CLAUDE.md`。
>
> 最後更新：2026-09-15

---

## 速查卡

```bash
# ── 日常 ───────────────────────────────────────────────────────────────
ssh amd                                        # 連進 MI300X
ssh amd 'bash /mlsteam/workspace/bin/selftest.sh'   # 40 項環境驗證
ssh amd -t gpuwatch                            # GPU 即時監看
bash code/mi300x/sync_scripts.sh               # 推腳本上去

# ── Lab 重開之後（三步）────────────────────────────────────────────────
# ① 網頁 Lab → ⚙ 設定 → Port Forwarding → 記下新 port（每次都不同！）
#    （全新 Lab 沒有 sshd，port forwarding 加不上去 → 先用網頁 Terminal 跑：
#       bash /mlsteam/workspace/bin/00_bootstrap.sh ）
bash code/mi300x/set_ssh_port.sh <新port>      # ②
ssh amd 'bash /mlsteam/workspace/bin/lab_up.sh'  # ③ 約 56 秒

# ── 量測（每個會產生數字的指令都要走 runsh）──────────────────────────
ssh amd '/mlsteam/workspace/bin/runsh m1-cap python code/m1_capacity.py ...'
```

| 這台機器的關鍵數字（**實測**） | |
|---|---|
| GPU | AMD Instinct MI300X, gfx942, **192.0 GiB**, 304 CU, SPX/NPS1 |
| CPU 配額 | **32**（`nproc` 騙你說 192） |
| RAM 配額 | **434.8 GiB** = 256 GiB flavor + 192 GB shm（`MemTotal` 騙你說 2268 GiB） |
| `/dev/shm` | 179 GiB |
| 持久儲存 | `/mlsteam/workspace`、`/mlsteam/data/tiara`（都是 **NFS**，各 950 GB） |
| 本地快碟 | `/` overlay on sda（SSD, 3.5 TB，**不持久、host 共用**） |
| 本地 NVMe | **沒有**（`/dev/nvme*` 不存在） |

---

## 0. 🔴 先讀這一段：這台機器會忘記你做過的事

平台底層是 **Kubernetes**。「開發環境（Lab）」是一個 Pod。
**Lab 一關、或你改了掛載／硬體規格 → container 整個重建 → 你 apt/pip 裝的東西全部消失。**

| 路徑 | 重開後 | 用途 |
|---|---|---|
| `/mlsteam/workspace` | ✅ **活著**（NFS，950 GB） | 腳本、venv、原始碼、log。**本專案的家** |
| `/mlsteam/data/tiara` | ✅ **活著**（NFS，掛載進來的 Data 資料夾） | 資料集、模型權重、runs 原始輸出 |
| `/root`、`/opt`、apt 裝的套件 | ❌ **死光** | 不要把任何東西留在這裡 |

**推論（也是這整套腳本存在的理由）：**
1. 任何安裝動作都必須寫成腳本，而且腳本要放在 `workspace`。
2. 不要手動裝東西。手動裝的東西下次沒有人記得。
3. 每次重開只跑一支 `lab_up.sh`。

---

## 1. 座標

### 平台
| 項目 | 值 |
|---|---|
| 入口 | <http://210.61.209.139/> |
| 名稱 | 教學平台（MLSteam / Manta，MyelinTek 提供） |
| 帳號 / 密碼 | 見 `SECRETS.local.md`（repo 根目錄，**已 gitignore**；本 repo 是 public） |
| 對外連線限制 | 平台只接受**已登記的來源 IP**（清單見 `SECRETS.local.md`）。換網路/VPN 會連不上 |

### 本專案在平台上的東西
| 項目 | 值 |
|---|---|
| 專案（Project） | `hugwei_amd`，id `p06dcd23` |
| 儲存配額 | **950 GB**（`Workspace` 與 `Data` 共用這個額度） |
| Data 資料夾 | `tiara` → 掛進 Lab 的 `/mlsteam/data/tiara` |
| Workspace | `<帳號>-vc-workspace` → Lab 內的 `/mlsteam/workspace` |
| 開發環境（Lab） | `tiara-mi300x`，id `u034a318` |
| Lab 用的範本 | **`dev-ubuntu-2404` / `7.2.2-complete`**（ubuntu 24.04 + ROCm 7.2.2，乾淨底） |
| 硬體規格（flavor） | **`gpu_full_single_gpu`** = CPU 32 / RAM 256 GB / GPU 1 |
| Shared Memory Size | **192 GB**（平台預設只有 1 GB，見 §2.3） |

### 硬體（**實測**，不是從文件抄的）
```
GPU[0] : Card Series  : AMD Instinct MI300X
GPU[0] : Card Model   : 0x74a1
GPU[0] : GFX Version  : gfx942
GPU[0] : Partitions   : NPS1, SPX, 0          ← 記憶體 NPS1 / 運算 SPX
       : NUM_COMPUTE_UNITS       : 304
       : VRAM Total Memory (B)   : 206141652992   = 192.0 GiB
```

> 🔴 **這一條跟原廠文件不一致，以實測為準。**
>
> 原廠教學的「預設基本配額」寫 **「GPU: 96 GB VRAM, 4 XCDs」**，而且它的
> 分割模式對照表把 **DPX**（2 邏輯裝置 / 152 CU / 96 GB）那一列框起來當預設。
> 我照著寫進這份文件，**是錯的**。
>
> 這個 Lab 實際拿到的是 **SPX 整張卡：304 CU、192 GiB HBM3**。
> `rocm-smi --showmeminfo vram` 與 `amd-smi static` 兩條獨立路徑互相對上
> （206,141,652,992 B ↔ `SIZE: 196592 MB` ↔ `NUM_COMPUTE_UNITS: 304`）。
>
> 這正是 `CLAUDE.md` §1 規則 6 講的事：**不要看欄位名／文件推斷，
> 要用資料自身交叉驗證。** 上次踩到的是 Mooncake 的 `hash_ids` block size，
> 這次差點又用文件上的 96 GB 去算容量懸崖 —— 那會讓整條 M1 的絕對數字差 2 倍。
>
> **M1 的配置腳本一定要在執行時再問一次 VRAM，不要把 192 寫死。**

原廠規格（供對照）：304 個 CDNA3 CU、192 GB HBM3、頻寬 5.3 TB/s、
記憶體匯流排 8192-bit、8×XCD + 4×IOD、台積電 5nm；
TF32 653.7 TFLOPS / FP16 1307.4 / BF16 1307.4 / FP8 2614.9 / INT8 5229.8 TOPs。

**對論文的意義**：平台 B 的單卡 VRAM 是平台 A（3090, 24 GB）的 **8 倍**，
而 CPU offload 預算（256 GB RAM）只有約 1.2 倍。
κ 的跨硬體變動就是被這種「階層間比例完全不同」撐出來的 —— 這是好消息。

---

## 2. 從零到能跑（完整流程）

### 2.1 建 Lab（網頁操作）
1. 左選單 **開發環境** → 右上 **新增**
2. 右上切到 **公用**，選 **dev-ubuntu-2404**
3. Version 選 **7.2.2-complete**（最高）
4. `name` = `tiara-mi300x`，`flavor` = `gpu_full_single_gpu`
5. **建立** → 等狀態變 **運行中**

> 為什麼不用平台給的 `vllm` 範本：那是別人釘好的 rocm7.0 + 某個 vLLM 版本，
> 我們無法確認它的 commit 與 `OffloadingConnector` 行為跟平台 A（3090, vLLM 0.28.0）一致。
> κ 的跨硬體主張要求兩邊量的是**同一條程式路徑**，所以自己從原始碼建。

### 2.2 建並掛載 Data 資料夾（網頁操作，**會重開 Lab**）
1. 左選單 **資料** → 右上 **新增** → **空白資料夾** → 名稱 `tiara`
2. 回到 Lab 的 Terminal 頁 → 上方 **🧰 資料夾** 圖示 → 右上 **⚙**
3. 把 `tiara` 的開關打開（右側會顯示掛載點 `/mlsteam/data/tiara`）→ **套用**
4. 跳出「變更設定將會重啟環境」→ **確認**

> ⚠️ **套用會重開 Lab，環境全毀，而且 expose-port 會換號碼。**
> 所以順序很重要：**先掛載，再裝東西**。

### 2.3 調大 Shared Memory（網頁操作，**會重開 Lab**）
**開發環境** 列表 → 右側 **⚙** → `Shared Memory Size`

平台預設 **1 GB**，可填 1~384。本專案設 **192 GB**。

> ⚠️ 這一項原廠教學沒提，但**不改 vLLM 會起不來** —— `/dev/shm` 是 vLLM
> worker 之間傳張量的地方，1 GB 一定爆。
>
> **為什麼是 192 而不是填滿 384**：flavor 的 RAM 是 256 GB，而 tmpfs 的頁
> 是算進 cgroup 記憶體額度的。把 shm 上限設成比 RAM 還大，寫爆的時候會變成
> **OOM-kill（整個 pod 被砍，實驗靜默消失）**，而不是 **ENOSPC（寫不進去，
> 錯誤訊息明確）**。留 64 GB 給 process，換到一個看得懂的失敗模式。
> 這跟 `CLAUDE.md` §1 規則 7 是同一個原則：**不要讓失敗變成靜默的**。

### 2.4 裝第一把鑰匙（網頁 Terminal，只有第一次要做）
全新 Lab 還沒有 SSH，是「雞生蛋」問題，只能用網頁 Terminal。
Lab 頁面點 **`>_`** 圖示開 Terminal，貼這一行（公鑰已內含）：

```bash
bash /mlsteam/workspace/bin/install_pubkey.sh
```

如果連 `workspace/bin` 都還沒有（全新專案），貼這一段：

```bash
mkdir -p /mlsteam/workspace/.ssh /root/.ssh && \
printf '%s\n' 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMpVXoF+mLhvDglv1jZoZklY1OSetumRtvjkolG0mJQP lai09150915@gmail.com mlsteam-mi300x' \
  > /mlsteam/workspace/.ssh/authorized_keys && \
cp /mlsteam/workspace/.ssh/authorized_keys /root/.ssh/authorized_keys && \
chmod 700 /mlsteam/workspace/.ssh /root/.ssh && \
chmod 600 /mlsteam/workspace/.ssh/authorized_keys /root/.ssh/authorized_keys && echo KEY_OK
```

接著裝並啟動 sshd：

```bash
export DEBIAN_FRONTEND=noninteractive; apt-get update -qq && \
apt-get install -y -qq openssh-server iproute2 && mkdir -p /run/sshd && ssh-keygen -A && \
printf 'PermitRootLogin prohibit-password\nPasswordAuthentication no\nPubkeyAuthentication yes\nClientAliveInterval 60\n' \
  > /etc/ssh/sshd_config.d/99-tiara.conf && \
service ssh restart && sleep 2 && ss -tlnp | grep ':22' && echo SSHD_LISTENING_OK
```

### 2.5 開 Port Forwarding（網頁操作）
1. Lab Terminal 頁 → 右上 **⚙ 設定** → **Port Forwarding** 的 **＋**
2. `Internal Port` = **22** → **新增**
3. 展開 Port Forwarding，記下 `22/tcp ➜ <expose-port>`（例：`46211`）

> 原廠文件的紅字提醒是對的：**服務要先在 0.0.0.0 上 listen，再開這個對話框。**
> 順序反了會加不上去。

### 2.6 設定本機 SSH（在 Mac 上）
```bash
bash code/mi300x/set_ssh_port.sh <expose-port>     # 例： 46211
ssh mi300x 'hostname; rocm-smi --showproductname | head -5'
```

### 2.7 推腳本 + 一鍵還原
```bash
bash code/mi300x/sync_scripts.sh          # Mac → /mlsteam/workspace/bin
ssh mi300x 'bash /mlsteam/workspace/bin/lab_up.sh'
```

vLLM 要另外跑（很久，用 tmux）：
```bash
ssh mi300x
tmux new -s build
bash /mlsteam/workspace/bin/20_build_vllm.sh
```

---

## 3. 硬體規格（flavor）清單

實測本專案下拉選單的完整內容：

| flavor | CPU | RAM | GPU |
|---|---|---|---|
| **`Gpu_full_single_gpu`** ← 我們用這個 | 32 | 256 GB | 1 |
| `Gpu_large_single_gpu` | 24 | 256 GB | 1 |
| `Gpu_medium_single_gpu` | 24 | 128 GB | 1 |
| `Gpu_small_single_gpu` | 16 | 128 GB | 1 |
| `Cpu_computing_full` | 32 | 256 GB | 0 |
| `Cpu_computing_large` | 24 | 256 GB | 0 |
| `Cpu_computing_medium` | 16 | 128 GB | 0 |
| `Cpu_computing_small` | 16 | 64 GB | 0 |
| `Cpu_medium` | 8 | 16 GB | 0 |
| `Cpu_small` | 2 | 4 GB | 0 |

**全部都是單卡。** 原廠教學截圖裡出現過的 `Amd_gpu_large`(GPU:6) 與
`Amd_gpu_performance`(GPU:8) **不在這個專案的清單裡**。

> 對論文其實是好事：`CLAUDE.md` §3 明文「容量懸崖必須在單卡上量，TP 會讓 §2.5 的算術失去意義」。
> 單卡 + 256 GB RAM（CPU offload 預算）正是 Tiara 要的形狀。
> CPU-only 的 flavor 拿來跑模擬器（`m4_*.py`）與品質分析剛好，不要浪費 GPU。

---

## 4. 每次 Lab 重開的還原流程（**背下來這三步**）

```bash
# ① 網頁：Lab → ⚙ 設定 → Port Forwarding → ＋ → 22 → 新增 → 記下新 port
#    （若 port forwarding 已存在就直接看號碼）

# ② Mac：更新 port（順便清掉對不上的舊 host key 與卡住的 ControlMaster）
bash code/mi300x/set_ssh_port.sh <新port>

# ③ 還原環境（**實測 55.7 秒**；vLLM 不含在內）
ssh amd 'bash /mlsteam/workspace/bin/lab_up.sh'

# ④ 驗證（40 項，退出碼 0 才算過）
ssh amd 'bash /mlsteam/workspace/bin/selftest.sh'
```

> venv 與 PyTorch 都在 `workspace`，所以第 ③ 步通常是「裝 apt 套件 + 起 sshd」，
> 不會重新下載 4.8 GB 的 torch wheel。55.7 秒就是這個情況下的實測值。

⚠️ 第 ① 步有雞生蛋問題：**新 Lab 沒有 sshd，port forwarding 加不上去**
（要先有東西 listen 在 22）。所以全新 Lab 一定要先用**網頁 Terminal** 跑 §2.4。
已經有 `workspace/bin` 的話，網頁 Terminal 只要貼：

```bash
bash /mlsteam/workspace/bin/00_bootstrap.sh
```

---

## 5. 腳本清單（`code/mi300x/` → 同步到 `/mlsteam/workspace/bin/`）

| 腳本 | 跑在哪 | 做什麼 |
|---|---|---|
| `install_pubkey.sh` | Lab（網頁 Terminal） | 把公鑰寫進 workspace 與 `/root/.ssh`。只有金鑰掉了才要跑 |
| `00_bootstrap.sh` | Lab | 系統套件 + sshd（金鑰登入 only）+ 還原 authorized_keys + PATH（兩個 HOME 都寫）+ `git safe.directory --system` + /dev/shm 檢查。**冪等** |
| `05_hwinfo.sh` | Lab | 硬體盤點：GPU / CPU / NUMA / 儲存 / 網路 / PCIe，**以及 cgroup 的真實配額**。摘要寫 `results/hw_mi300x.json` |
| `10_python_stack.sh` | Lab | 建 venv（在 workspace）+ **探測**可用的 PyTorch ROCm index + 裝 + **驗證真的是 HIP build** |
| `15_monitoring.sh` | Lab | nvtop / radeontop / btop / htop / iotop / sysstat / amdgpu_top + 自製 `gpuwatch`、`gpulog` |
| `20_build_vllm.sh` | Lab | 從原始碼建 vLLM (ROCm, gfx942)，預設 `v0.29.0`（`VLLM_REF` 可覆寫），驗 `OffloadingConnector` import 得到 |
| `16_amdtop.sh` | Lab | **amdtop 0.2.6**（nvitop 風格 AMD TUI）+ Rust 工具鏈，都裝在 workspace |
| `25_storage_bench.sh` | Lab | fio 量 NFS vs 本地 overlay vs tmpfs 的頻寬/延遲 → 決定「SSD 階」掛哪（見 §7.5.5） |
| `30_project.sh` | Lab | clone 論文 repo、建目錄骨架、寫 `results/env_mi300x.json`（含 cgroup 配額） |
| `lab_up.sh` | Lab | **一鍵還原** = 00 + 15 + 10 + 30 + 05 + selftest。回傳 selftest 失敗項數 |
| `selftest.sh` | Lab | **40 項環境驗證**，每一項都印實測值（只印 PASS 等於沒檢查）。退出碼 = 失敗項數 |
| `40_claude_code.sh` | Lab | **Node 24 + Claude Code**，安裝與登入狀態都在 workspace → 重開不用重裝、不用重登入 |
| `99_report.sh` | Lab | 把所有實測彙整成 `REPORT_MI300X.md`（機器產生，每個數字標來源） |
| `gpu_smoke.py` | Lab | 證明 GPU「真的能算」：matmul + 數值正確性 + HBM/H2D/D2H 頻寬 |
| `90_snapshot.sh` | Lab | 把 apt / pip 清單、rocm-smi、history 存進 `workspace/snapshots/` |
| `runsh` | Lab | run 記錄殼（CLAUDE.md §4.1 的平台 B 版）。**每個會產生數字的指令都要走它** |
| `set_ssh_port.sh` | **Mac** | Lab 重開後更新 `~/.ssh/config` 的 Port + 清舊 host key + 踢掉舊 ControlMaster |
| `sync_scripts.sh` | **Mac** | 把 `code/mi300x/` 推到 Lab（遠端沒有 rsync，走 tar over ssh，並擋掉 macOS 的 `._*`） |

### 腳本的設計原則
- **失敗即停**（`set -Eeuo pipefail` + ERR trap）。對應 `CLAUDE.md` §1 規則 2：
  指令失敗 → 記錄完整錯誤 → 停下來回報，**不要換個方式硬幹到有輸出為止**。
- **不猜版本**。`10_python_stack.sh` 不硬寫 PyTorch wheel index，
  而是**探測哪個 index 真的存在**；探測不到就停，不會默默裝成 CPU 版。
  `20_build_vllm.sh` 的 `gfx942` 也不是寫死的，是 `10_` 從 `rocminfo` 問出來存進
  `$WS/.gfx_arch` 的。
- **驗證是強制的**。裝完 PyTorch 一定檢查 `torch.version.hip is not None`
  且 `torch.cuda.is_available()`；任一失敗直接 exit 1。

---

## 6. SSH

### `~/.ssh/config`（已寫好）

兩個別名指向同一台：`ssh amd`（短，日常用）與 `ssh mi300x`（明確）。

```sshconfig
Host amd mi300x
    HostName 210.61.209.139
    User root
    Port 46211                              # ← 每次 Lab 重開都會變
    IdentityFile ~/.ssh/mlsteam_mi300x
    IdentitiesOnly yes
    UserKnownHostsFile ~/.ssh/known_hosts.mlsteam
    StrictHostKeyChecking accept-new
    ServerAliveInterval 60
    ServerAliveCountMax 5
    TCPKeepAlive yes
    ControlMaster auto
    ControlPersist 30m
    ControlPath ~/.ssh/cm-%r@%h:%p
```

### 兩個會咬人的地方
1. **Port 每次都變。** expose-port 是動態配的。連不上先看網頁，再跑 `set_ssh_port.sh`。
2. **Host key 每次都變。** container 是全新的，`ssh-keygen -A` 重產 host key，
   但 `HostName:Port` 可能重複 → 會噴 `REMOTE HOST IDENTIFICATION HAS CHANGED`。
   我們**沒有**全域關掉 host key 檢查（那是把保護整個丟掉），
   而是給這台機器一個獨立的 `~/.ssh/known_hosts.mlsteam`，
   由 `set_ssh_port.sh` 在換 port 時針對性地 `ssh-keygen -R` 清掉那一筆。

### 常用
```bash
ssh amd                              # 直接進去
ssh amd 'bash /mlsteam/workspace/bin/selftest.sh'   # 遠端驗證
ssh amd -t 'gpuwatch'                # GPU 即時監看
bash code/mi300x/sync_scripts.sh     # 推腳本上去
bash code/mi300x/set_ssh_port.sh N   # Lab 重開後換 port
```

### 金鑰
- 私鑰 `~/.ssh/mlsteam_mi300x`（ed25519，無 passphrase — 為了無人值守的實驗腳本）
- 公鑰指紋 `SHA256:SOspscr2X1WAjeMGtv2uMJkqBvYqAFbNqnwPG3mdNHQ`
- 公鑰本體存在 `/mlsteam/workspace/.ssh/authorized_keys`（跨 Lab 存活）

### VS Code / Cursor 遠端開發
`Remote-SSH` 直接選 `mi300x` 就會通（config 已備妥）。
原廠 `init.sh` 那行 `export PATH=/opt/venv/bin:/opt/rocm/bin:$PATH` 就是為了這個 ——
我們的 `00_bootstrap.sh` 已經把它寫進 `/root/.bashrc` 和 `/root/.profile` 兩邊
（VS Code 開的是非互動 shell，只寫 `.bashrc` 會漏）。

---

## 7. 網頁操作速查（對照原廠教學）

| 想做什麼 | 路徑 |
|---|---|
| 新增專案 | 左選單 **專案** → 右上 **新增** → 填名稱/描述 |
| 新增資料夾 | 左選單 **資料** → 右上 **新增** → 空白資料夾 / 匯入資料夾 / 掛載 NFS / 掛載 CIFS |
| 從 URL 抓檔 | **資料** → **新增** → **File From URL**（貼 GitHub 的 `.zip` 連結） |
| 解壓縮 | 勾選 `.zip` → **操作** → **Extract** → 確認 |
| 從 Dockerfile 建 image | 勾 Dockerfile → **操作** → **建置 Dockerfile** → 填 `image:tag` |
| 從 Docker Hub 拉 image | **範本** → **映像檔** → **新增** → **Internet Pull** |
| 建範本 | **範本** → 右上 **新增** → From image / From folder / From compose |
| 建 Lab | **開發環境** → **新增** → 選範本 → 填 name + flavor |
| 開 Terminal | Lab 列右側 **`>_`** |
| 掛載資料夾 | Lab Terminal 頁 → **🧰** → **⚙** → 開關 → **套用**（**會重開**） |
| Port Forwarding | Lab Terminal 頁 → **⚙ 設定** → **Port Forwarding** → **＋** |
| 看 Workspace | 左選單 **工作區** |

**映像檔命名規則**：`myelintek/...` = 平台公開的；`p<專案id>/...` = 這個專案自己建的。

---

## 7.5 🔴 量測陷阱（**這一章比其他所有章加起來重要**）

這些都是**實測撞到**才發現的，照文件做一定會踩。

### 7.5.1 容器裡的 `nproc` / `MemTotal` / `numactl` 回報的是整台 host

| 量 | 容器裡看到的 | **我們真正能用的** | 差幾倍 |
|---|---|---|---|
| CPU | `nproc` = **192** | `cpu.max` = **32** | 6× |
| RAM | `MemTotal` = **2268 GiB** | `memory.max` = **434.8 GiB** | 5.2× |
| NUMA | `numactl -H` 顯示 2 節點各 1.13 TB | 那是 host 的，不是配額 | — |

任何 `os.cpu_count()` / `multiprocessing.cpu_count()` / `MAX_JOBS=$(nproc)`
都會**超賣 6 倍**。一開始 `20_build_vllm.sh` 就是這樣寫的，已經改成讀 cgroup。

```bash
# 正確做法
CPU_QUOTA=$(awk '{if($1=="max") print 0; else printf "%d", $1/$2}' /sys/fs/cgroup/cpu.max)
MEM_QUOTA=$(cat /sys/fs/cgroup/memory.max)
```

`results/env_mi300x.json` 與 `results/hw_mi300x.json` 兩邊都同時記
`*_host_view` 與 `*_quota`，不要只記一個。

### 7.5.2 `memory.max` = flavor RAM ＋ shm（shm 是**額外加上去**的）

```
memory.max      = 466,877,906,944 B
flavor RAM      = 256 GiB = 274,877,906,944 B
shm 設定值      = 192 GB  = 192,000,000,000 B
274,877,906,944 + 192,000,000,000 = 466,877,906,944  ✅ 完全吻合
```

> **這修正了我在 §2.3 的判斷。** 我當初怕 shm 吃掉 process 記憶體、
> 所以只設 192 而不是上限 384，理由是「tmpfs 頁算進 cgroup 額度」。
> 實測顯示平台是**把 shm 加在 flavor RAM 之上**，不是從裡面切。
> 所以設 384 也不會排擠 process 記憶體。目前維持 192 是因為夠用，
> 不是因為原本那個理由 —— 兩個數字用同一條算式對上了，這個結論才站得住。

### 7.5.3 網頁 Terminal 與 SSH 的 `HOME` 不一樣

| 入口 | `$HOME` |
|---|---|
| 網頁 Terminal | `/mlsteam/workspace` ← 平台覆寫的 |
| SSH | `/root` ← `/etc/passwd` 寫的 |

**後果**：任何寫進 `$HOME` 的設定都只對其中一邊生效。
實際踩到的是 `git config --global` 寫進 `/mlsteam/workspace/.gitconfig`，
於是網頁 Terminal 跑得動、`ssh amd` 進來就噴 `dubious ownership`。

**對策**（`00_bootstrap.sh` 已實作）：
- git → `git config --system`（`/etc/gitconfig`），與 HOME 無關
- PATH → 同時寫 `/root/.bashrc`、`/root/.profile`、`$WS/.bashrc`、`$WS/.profile`，
  再加 `/etc/profile.d/99-tiara.sh` 當第三道保險
- 其他任何 `~/.xxx`（pip cache、`HF_HOME`）都要**明確指定絕對路徑**，不要靠 `~`

### 7.5.4 NFS 的 uid squash → git 拒絕操作

`/mlsteam/*` 是 NFS，檔案 owner 被 squash 成 `nobody:nogroup`（或 tar 帶來的 501），
git 會判定 dubious ownership 而拒絕。這是**平台常態不是異常**，
`00_bootstrap.sh` 每次都會重設 `safe.directory`。

### 7.5.5 🔴 **這個 Lab 沒有本地 NVMe —— 論文的「SSD 階」要重新定義**

```
/sys/block           → dm-0 dm-1 loop0..7 sda      （沒有任何 /dev/nvme*）
/mlsteam/workspace   → nfs4  <NFS_SERVER>  rsize=65536 wsize=65536
/mlsteam/data/tiara  → nfs4  <NFS_SERVER>  （同一台 NFS server）
/  (overlay)         → ext4 on sda，rotational=0（SSD），3.5 TB，本地
```

平台 A（3090）的 SSD 階是 `/ssd7`，**真的是本地 SSD**。
平台 B 這裡持久化的兩個掛載點**都是網路檔案系統**。

| 候選 | 持久 | 實體 | 成本特性 |
|---|---|---|---|
| (a) `/mlsteam/data/tiara` | ✅ | **NFS over TCP** | 含網路往返，64 KB rsize |
| (b) `/` overlay 上的 scratch | ❌ | **本地 SSD (sda)** | 真正的本地 block IO |

**論文要用 (b)。** 理由：KV cache 溢出到 SSD 本來就是**暫存**用途，
不需要跨 Lab 存活；而 (a) 會把 NFS 的網路成本混進 κ，
讓「跨硬體的成本比差異」變成「跨硬體＋跨儲存介質的差異」，歸因就毀了。

⚠️ 但 (b) 也不是平台 A 的同類物：`/` 是 **overlayfs on LVM on sda**，
中間多了 overlay 與 LVM 兩層，而且 sda 是**整台 host 共用**的
（`df /` 顯示 3.0T 已用 1.3T —— 那 1.3T 不是我們的）。
**在量 SSD 階之前，必須先用 fio 把 (a) 與 (b) 的頻寬/延遲各量一遍並寫進 RUNLOG，
在那之前不要對 SSD 階下任何結論。** 目前狀態：`NOT_MEASURED`。

### 7.5.6 venv 在 NFS 上 → `pip install` 非常慢

`venv` 放在 `/mlsteam/workspace/venv/tiara`，因為那是**唯一跨 Lab 存活**的地方。
代價是每次 `pip install` 都在 NFS 上寫幾萬個小檔：

```
$ ps aux | grep pip
root  7837  27.9  ... Dl+  python -m pip install -r requirements/rocm.txt
                      ^^^  D = uninterruptible sleep，卡在 IO 不是在算
```

**這是刻意的取捨，不是 bug**：
- venv 放 NFS → 裝很慢，但 Lab 重開不用重裝（省 4.8 GB 的 torch 下載）
- venv 放本地 `/` → 裝很快，但 Lab 一關就沒，每次都要重來

目前選前者。實測 `lab_up.sh` 還原只要 55.7 秒，就是因為 venv 沒有重建。

⚠️ **但這件事會污染任何「首次載入模型」的時間量測** ——
模型權重如果放 NFS，冷啟動時間量到的是 NFS 頻寬不是 GPU 行為。
`$DATA/hf-cache` 也在 NFS 上，所以 **M2 的成本模型量測必須先確認
權重已在 page cache 裡，或明確把冷啟動時間分開報**。

### 7.5.7 🔴 這是一台 **8 張 MI300X** 的機器，我們只拿到其中一張

裝 amdtop 時撞出來的（它一啟動就 panic，追下去才發現）：

```
/sys/bus/pci/drivers/amdgpu/ 有 8 個 BDF：
  0000:06:00.0  0000:26:00.0  0000:46:00.0  0000:66:00.0
  0000:86:00.0  0000:a6:00.0 ← 我們的   0000:c6:00.0  0000:e6:00.0

容器的 /dev/dri 只有：card41 + renderD168  （= 0000:a6:00.0）
/sys/class/drm/ 另有 56 個 amdgpu_xcp_* → 8 張 × 7 個 XCP 分割
```

**我們獨佔一張卡**（SR-IOV `MI300X_HW_SRIOV_CVS_1VF` = 整張卡給一個 VF），
**但整個節點是共用的**。這修正了 §10 風險表第 2 條的問題陳述：
問題不是「有人跟我們共用這張卡」，而是「**有 7 個鄰居跟我們共用這台主機**」。

會被鄰居影響的量測：

| 量 | 為什麼會被影響 |
|---|---|
| **H2D / D2H 頻寬**（`gpu_smoke.py` 量到 50.8 / 48.6 GB/s） | 8 張卡共用主機的 PCIe root complex 與記憶體頻寬。鄰居在搬資料時我們會變慢 |
| **本地磁碟**（`/` overlay on sda 3.5 TB） | 整台 host 共用（`df /` 已用 1.3 T，那不是我們的） |
| **NFS 頻寬** | 同一台 NetApp 服務整個叢集 |
| GPU 內的運算與 HBM | ✅ 不受影響（獨佔整張卡） |

**這對論文是直接相關的**：κ 的分母是主機鏈路頻寬。
若 H2D 量測期間有鄰居在搬資料，κ 會被高估。
→ **M2 的每一次量測都必須同時記錄鄰居狀態**，至少 `amd-smi process` 與
`iostat`；而且要重複多次取分佈而非單點。目前 50.8 GB/s 這個數字
**只是 smoke test，尚未做爭用控制**。

### 7.5.8 監控工具在多租戶容器裡會讀到別人的卡

同一個根因的第二個後果：

| 工具 | 在這個容器裡 |
|---|---|
| `rocm-smi` / `amd-smi` | ✅ 正確（只看得到我們的） |
| `rocmtop` | ✅ 正確（走 rocm-smi） |
| `radeontop` | ✅ 正確（自動選到 bus a6） |
| `amdgpu_top` | ⚠️ **預設會列出別人的卡**，一定要加 `--pci 0000:a6:00.0` |
| `amdtop` | ❌ upstream 版本**一啟動就 panic**（見下） |

`amdtop` 的 upstream 假設「機器上的 GPU 都是自己的」：
它從 sysfs 列出全部 8 張，然後對每張呼叫 `libamdgpu_top` 的 `DevicePath::get_fd()`
—— 那個函式開不到 `/dev` 節點時是 `panic!()` 而不是回傳 `Err`
（`libamdgpu_top-0.11.5/src/device_path.rs:75`）。

**我們的修法**（`16_amdtop.sh` + `patches/amdtop-skip-inaccessible-devices.patch`）：
從 git 原始碼建，在 `discover_devices()` 裡於任何東西碰到 fd 之前先濾掉節點不存在的裝置。
啟動時會印 `amdtop: skipped 7 GPU(s) visible in sysfs but not exposed to this container`。

> ⚠️ **不要用「補上缺的 /dev 節點」來繞過。** 那會讓 amdtop 顯示**別的租戶**
> 的 GPU 數據 —— 靜默的錯誤資料比崩潰更糟（`CLAUDE.md` §1 規則 7）。

### 7.5.9 ROCm 會在 stderr 印一行無害的警告

```
/opt/amdgpu/share/libdrm/amdgpu.ids: No such file or directory
```
無害，但如果你用 `2>&1` 把它混進資料流，解析就會讀到這行而不是資料。
`selftest.sh` 一開始就是這樣把好環境誤判成壞的。**解析輸出時 stderr 要分開收。**

---

## 8. 這個平台跟論文的接軌

`CLAUDE.md` §3 的「平台 B 移植面」只有三處，在這裡的狀態：

| 位置 | 狀態 |
|---|---|
| `gpu_guard.py` 的 `vendor()` / `_amd_compute_apps()` / `gpu_util()` | AMD 後端已寫好但**尚未在真機驗證**。第一次用之前必須對照 `amd-smi process` 的輸出，確認 pid 與 gpu index 對得上 |
| `env_fingerprint.py` | `torch.version.hip` 可用（`30_project.sh` 已經寫進 `results/env_mi300x.json`） |
| `CUDA_VISIBLE_DEVICES` | ROCm 接受此變數，但保險起見在 AMD 上同時設 `HIP_VISIBLE_DEVICES` |

**平台 B 上真正要重量的只有 M1 容量 與 M2 成本模型** —— κ 的跨硬體主張就是由這兩者構成。
其餘（Oracle、預算掃描、語意消融、長上下文外插）是拿新的成本常數重跑同一批腳本。

### 這台機器上要注意的量測污染
與平台 A 不同，這裡是 **k8s 配額隔離的單卡**，不是共用的整機，
所以 `gpu_guard` 那套「防別人插隊」的邏輯在這裡的風險較低。
但**不要因此就不跑 `GpuWatcher`** —— DPX 分割的另一半可能被別的租戶用，
而 `rocm-smi` 看到的是不是自己那一半，**尚未驗證**。在驗證之前，
時間類的數字一律當作「待確認」。

---

## 9. 記錄協定（平台 B 版）

跟 `CLAUDE.md` §4 一樣，只是路徑換掉：

```bash
# 每一個會產生數字的指令都走這個殼
/mlsteam/workspace/bin/runsh m1-cap-llama python code/m1_capacity.py --model llama --ctx 131072
```

產出在 `/mlsteam/data/tiara/runs/<RUN_ID>/`：`cmd.sh`、`context.txt`（含 rocm-smi、repo HEAD、vLLM commit）、`stdout.log`、`stderr.log`、`exit_code`。

- `results/**/*.csv` 一定要有 `run_id` 與 `ts` 欄。**沒有 `run_id` 的數字視同不存在。**
- 小的、可版控的結果 → git 的 `results/`；大檔案 → `/mlsteam/data/tiara/`。
- 每個 Milestone 結束補一段 `results/RUNLOG.md`，commit 開頭用 `results(mN):`。

---

## 10. 已知風險 / 待確認

| # | 事項 | 狀態 |
|---|---|---|
| 1 | GPU 分割模式 | **已實測釐清**：SPX 整張卡 192 GiB / 304 CU（`rocm-smi`、`amd-smi`、`torch` 三條路徑互相對上），**不是**原廠文件寫的 DPX 96 GB。腳本不可寫死 |
| 2 | GPU 是否獨佔 | ✅ **已釐清**：獨佔整張卡（SR-IOV 1VF），但**整個節點有 8 張卡、7 個鄰居**。見 §7.5.7 |
| 3 | `gpu_guard.py` 的 AMD 後端 | **未在真機驗證**。第一次用要對照 `amd-smi process`（已由 `05_hwinfo.sh` 存下輸出） |
| 4 | vLLM 版本落差 | 平台 B 建 **v0.29.0**、平台 A 是 **v0.28.0**。**兩邊對齊之前，κ 的跨硬體數字一律標 `NOT_COMPARABLE`** |
| 5a | **主機鏈路量測受鄰居影響** | 🔴 **未控制**。H2D 50.8 GB/s 只是 smoke test。M2 必須同時記錄鄰居狀態並重複取分佈 |
| 5 | vLLM 從原始碼建置 | 🔴 **判定不可行**（三次嘗試，根因相同）。見 §10.45。剩 C（AMD 官方 image）或 D（先做不需要 vLLM 的量測） |
| 6 | **SSD 階的載體** | 🔴 **未決**。沒有本地 NVMe；持久掛載都是 NFS。見 §7.5.5。fio 量測前一律 `NOT_MEASURED` |
| 7 | `/dev/shm` 192 GB 是否足夠 | **未實測**。vLLM 起得來之後看 `df -h /dev/shm` 實際用量 |
| 8 | 能耗分析 | 與平台 A 同樣**不做**（尚未確認 MI300X 在此平台有無可用的分軌計數器） |
| 9 | 對外連線限制 | 只接受已登記來源 IP（見 `SECRETS.local.md`）。換網路/VPN 可能連不上 |
| 10 | `nproc` 陷阱 | **已修**（`20_build_vllm.sh`、`env_mi300x.json`）。但**任何新寫的腳本都要再注意一次** |

---


## 10.4 🔴 vLLM 建置阻塞：ROCm 的 PyTorch 版本追不上 vLLM

**狀態：BLOCKED。需要你決定走哪條路。**

### 事實（全部實查，2026-09-15）

vLLM 各版本釘的 torch：

| vLLM | 需要 torch |
|---|---|
| v0.27.0 – **v0.29.0** | **2.13.0** |
| v0.22.1 – **v0.26.0** | **2.11.0** |

ROCm 版 PyTorch 實際拿得到的最高版本：

| 來源 | 最高 torch |
|---|---|
| pytorch.org `/whl/rocm7.0`（stable） | **2.10.0** ← 我們裝的 |
| pytorch.org `/whl/nightly/rocm7.0` | **2.11.0.dev20260206** |
| `repo.radeon.com/rocm/manylinux/rocm-rel-7.2.2` | **2.9.1** |

**沒有任何官方管道提供 ROCm 版的 torch 2.13.0。**
AMD 自己的 `rocm/vllm` docker image 也印證這件事 ——
最新那個是 `rocm10.0.0 + pytorch 2.12.0 + vllm 0.27.0`（2026-08-27），
其餘多數還停在 `vllm 0.23.0`。**AMD 自己也落後好幾個 vLLM 版本。**

### 實際撞到的編譯錯誤

```
csrc/libtorch_stable/hip_view.hip:21:34:
  error: no member named 'layout' in 'torch::stable::Tensor'
csrc/libtorch_stable/hip_view.hip:44:9:
  error: no viable conversion from '(lambda ...)' to 'int64_t'
```
`torch::stable` 這套 ABI 在 2.10 → 2.13 之間改過，vLLM 0.29 用的是 2.13 的形狀。
（`csrc/libtorch_stable/hip_view.hip` 這個檔 **v0.28.0 根本沒有**，是 0.29 新增的。）

### 選項（**我不會自己選 —— 這會動到論文的可比性主張**）

| | 做法 | 代價 |
|---|---|---|
| **A** | 平台 B 用 **vLLM v0.26.0 + torch 2.11.0(nightly rocm7.0)** | 這是這台機器上**實際建得起來的最新組合**。但平台 A 是 0.28.0 → **κ 不可比**，除非平台 A 也降到 0.26.0 重跑 M1/M2 |
| **B** | **兩邊都降到 v0.26.0** | κ 可比。代價是平台 A 已完成的 M1–M5 要重跑，時程吃緊（MLSys 剩 45 天） |
| **C** | 拉 AMD 官方 `rocm/vllm` image 當範本 | 省事，但版本由 AMD 決定（最新也只有 0.27.0，且綁 rocm10.0.0），而且**失去「兩邊同一份原始碼」的控制權** —— 這正是當初不用平台 `vllm` 範本的理由 |
| **D** | 先不做 vLLM，改跑**不需要 vLLM 的量測** | M1 容量懸崖的一部分、M2 的 HBM↔host 傳輸成本、儲存階 fio，都可以只靠 torch 量。把 vLLM 相關的（OffloadingConnector 路徑）押後 |

> ⚠️ **不要為了讓建置過去而隨便換 tag。** `20_build_vllm.sh` 的註解已經寫死這一條。
> 選項 A/B 的差別不是工程偏好，是**論文能不能宣稱 κ 跨硬體可比**。

---

## 10.45 vLLM 從原始碼建置：**不可行**（三次嘗試後的結論）

| # | 組合 | 結果 |
|---|---|---|
| 1 | vLLM v0.29.0 + torch 2.10.0+rocm7.0 | ❌ `torch::stable::Tensor` 無 `layout` |
| 2 | vLLM v0.26.0 + torch 2.10.0+rocm7.0 | ❌ 同上（`hip_view.hip` 是 hipify 產生的，v0.26 的 `cuda_view.cu` 一樣會產生它） |
| 3 | vLLM v0.26.0 + **torch 2.11.0.dev+rocm7.0** | ❌ `layout` 仍缺、`from_blob` 簽章不合、**`at::cuda::getCurrentHIPStreamMasqueradingAsCUDA` 不存在** |

**決定性事實**：pytorch.org 的 ROCm nightly 最新是 `torch-2.11.0.dev**20260206**`
—— **停在 2026-02-06，已經 7 個月沒更新**。
而 vLLM v0.27+ 要 torch 2.13.0，v0.22–0.26 要 2.11.0（正式版，非 dev）。

| 來源 | 最高 torch | 日期 |
|---|---|---|
| pytorch.org `/whl/rocm7.0` | 2.10.0 | — |
| pytorch.org `/whl/nightly/rocm7.0` | 2.11.0.**dev20260206** | **7 個月前** |
| `repo.radeon.com` rocm-rel-7.2.2 | 2.9.1 | — |

→ **公開管道的 ROCm PyTorch 沒有任何一版能建起近期的 vLLM。**
AMD 自家的 `rocm/vllm` image 也印證：最新 CDNA 版是
`rocm7.14.1_cdna_..._pytorch_2.11_vllm_**0.23.0**`（2026-09-01）。

### 最終決定：**vLLM v0.19.1 + torch 2.10.0+rocm7.0**

把 vLLM 每個 tag 的 torch 釘選掃過一遍後，供需關係是這樣：

| vLLM | 要 torch | ROCm 供得起？ |
|---|---|---|
| v0.27.0–v0.29.0 | 2.13.0 | ❌ 任何管道都沒有 |
| v0.20.0–v0.26.0 | 2.11.0 | ❌ 只有 7 個月前的 `.dev` |
| **v0.17.0–v0.19.1** | **2.10.0** | ✅ pytorch.org `/whl/rocm7.0` |
| v0.14.1–v0.16.0 | 2.9.1 | ✅ repo.radeon.com（更舊） |

選 **v0.19.1** —— 「torch 供得起的最新 vLLM」。四項佐證：

1. **AMD 官方 `rocm/vllm` image 就有 `pytorch_2.10.0_vllm_0.19.1` 這個配對** ← 上游自己驗證過
2. v0.19.1 有 `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py` ← **論文 A3 驗收條件成立**
3. v0.19.1 **沒有** `csrc/libtorch_stable/cuda_view.cu`，也沒用 `torch::stable` 的 `layout()` ← 正是 v0.26/v0.29 建不起來的原因
4. v0.19.1 的 CMakeLists 寫 `TORCH_SUPPORTED_VERSION_ROCM "2.10.0"`

⚠️ **torch 不能裝「該 index 上的最新」**：2.11.0.dev 拿掉了
`at::cuda::getCurrentHIPStreamMasqueradingAsCUDA`，而 v0.19.1 的
`csrc/custom_quickreduce.cu` 還在用它。`10_python_stack.sh` 已改成**釘死 2.10.0**。
**在這個生態系裡，「裝最新」是錯的預設值。**

### 其他選項（若 v0.19.1 也失敗）

| | 做法 | 代價 |
|---|---|---|
| **C** | 拉 AMD 官方 `rocm/vllm` image（Internet Pull → 建範本） | 版本由 AMD 決定（CDNA 最新 0.23.0）。**注意這不是平台的 `vllm` 範本** —— AMD 官方 image 的 tag 直接寫明 rocm/pytorch/vllm 三者版本，來源可追 |
| **D** | 先做**不需要 vLLM** 的量測 | M1 容量的算術部分、M2 的 HBM↔host 傳輸（已用 `gpu_smoke.py` 量到 H2D 50.8 / D2H 48.6 GB/s）、儲存階 fio 都不需要 vLLM |

**不要再試從原始碼建。** 三次失敗的根因相同且不會自己好轉：
ROCm 的 PyTorch 發佈節奏落後 vLLM 太多。

---

## 10.5 驗證紀錄（2026-09-15）

### 砍掉重開的端到端測試 —— **通過**

真的把 Lab 停掉再開，不是模擬：

| 步驟 | 結果 |
|---|---|
| 停止 Lab（網頁 ■ → 確認） | 狀態 `Stopping` → `完成` |
| Mac 端 `ssh amd` | `Connection refused` ← **確認機器真的沒了** |
| 重新啟動（網頁 ▶） | `Starting` → `運行中`，新 pod `lab-u034a318-f7c8b749d-fwzpg` |
| 網頁 Terminal 跑 `lab_up.sh` | **55.7 秒**完成，selftest **38 項全過** |
| expose-port | 46528 → **46331**（自己換掉了） |
| Mac 跑 `set_ssh_port.sh 46331` | OK |
| Mac `ssh amd` 跑 selftest | 修掉兩個 bug 後 **40 項全過，RC=0** |

三次 Lab 重啟量到的 expose-port：`46211` → `46528` → `46331`。
**每次都不同，沒有規律，一定要去網頁看。**

### 這一輪修掉的 bug（都是這次實測才浮出來的）

| # | 症狀 | 根因 | 修法 |
|---|---|---|---|
| 1 | `05_hwinfo.sh` 靜默 exit 127 | `iproute2` 沒裝，`ip` not found；而且錯誤被 `2>&1 >file` 吞掉看不到 | bootstrap 補裝 `iproute2/net-tools/dnsutils`；hwinfo 加 ERR trap + `opt()` 包裝可選指令 |
| 2 | `git clone` 撞 "destination already exists" | `05_hwinfo.sh` 用 `mkdir -p` 憑空造出 repo 目錄 | 只在 `$REPO/.git` 存在時才寫進 repo |
| 3 | repo 變成沒有 commit 的半殘 git | 用 `git init` 硬接管既有目錄 | 改成 clone 到旁邊再合併檔案 |
| 4 | `ssh amd` 噴 dubious ownership，網頁 Terminal 卻正常 | 兩邊 `HOME` 不同，`git config --global` 寫到不同檔案 | 改用 `git config --system` |
| 5 | selftest 把好的 torch 判成壞的 | ROCm 的 libdrm 警告被 `2>&1` 混進資料行 | stderr 分開收，stdout 只取最後一行 |
| 6 | `set_ssh_port.sh` 找不到 Host 區塊 | 加了 `amd` 別名後，`^Host mi300x` 比對不到 `Host amd mi300x` | 改成比對「欄位之一等於 mi300x」 |
| 7 | `MAX_JOBS=$(nproc)` = 192 | `nproc` 回報整台 host，配額只有 32 | 改讀 `/sys/fs/cgroup/cpu.max` |

> 這七個裡有**四個**（1、4、5、7）屬於同一類：
> **「看起來能動，但量出來的東西是錯的 / 錯誤被靜默吞掉」**。
> 這正是 `CLAUDE.md` §1 規則 7 要防的失敗模式。
> 寫新腳本時，先問「這支失敗的時候，我看得見嗎？」

### 目前狀態

| 項目 | 狀態 |
|---|---|
| Lab / 掛載 / shm / SSH / port forward | ✅ 完成並通過砍掉重開測試 |
| 監控工具 | ✅ rocm-smi, amd-smi, nvtop, radeontop, amdgpu_top, btop, htop, iostat, gpuwatch, gpulog |
| PyTorch ROCm | ✅ `2.10.0+rocm7.0`, `hip=7.0.51831`, 看得到 192 GiB |
| 論文 repo | ✅ `HEAD=2702df6` |
| 環境指紋 / 硬體盤點 | ✅ `results/env_mi300x.json`、`results/hw_mi300x.json` |
| vLLM v0.29.0 | ⏳ 建置中 |
| SSD 階的載體 | 🔴 **未決**，見 §7.5.5 |
| M1 / M2 實驗 | ⬜ 尚未開始 |

---

## 11. 密碼與金鑰放哪

**不在這個檔，也不在任何會 commit 的檔案裡。**
本 repo 是 public（<https://github.com/s990093/paper-hierarchical-kv-state>）。

→ `SECRETS.local.md`（repo 根目錄，已在 `.gitignore`，已用 `git check-ignore` 驗證）
