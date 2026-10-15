# 平台 C 操作手冊 — NVIDIA H200（學校 Slurm 叢集）

> 這份是**操作手冊**：怎麼申請、怎麼連、怎麼在 Slurm 上照本專案的記錄規則跑實驗，以及換到 H200 之後精度計劃要改什麼。
> 來源：叢集 wiki「H200 計算節點說明會」（<https://wiki.deepcat.dedyn.io/s/de18e791-6f86-4861-bb2f-50c9e848699a>，2026-10-14 讀取，六個子頁全文）與申請表單（<https://forms.gle/h7FzavvtEDunLLPL9>）。
> **標記**：〔wiki〕＝wiki 原文；〔推論〕＝我們推的、沒在叢集上試過；沒量的寫 NOT_MEASURED。登入後量到的值要填回 §3 的表，並記進 `results/RUNLOG_H200.md`（新開，比照 `RUNLOG_MI300X.md`）。
> 平台 A（3090）看 `CLAUDE.md`；平台 B（MI300X）看 [`MI300X_MLSTEAM.md`](MI300X_MLSTEAM.md)。
>
> 最後更新：2026-10-15（§4.3、§5.2、§5.3 改成以已寫好的 `env_h200.sh`、`tiara.sbatch`、`gpu_guard.py` 為準；加 `r_h200.sh`）

**一句話**：用學校帳號填表申請（Lab 選 `ChihChungHsu`）→ 收到 OpenVPN 帳密 → VPN＋ssh 登入 head node → 用 Slurm 在 `h200q` 排一張卡跑。精度這週（[`precision_20261013/03_plan_h200.md`](precision_20261013/03_plan_h200.md)）所有會產生數字的 GPU 量測都改在 H200 上跑，**不和 3090、MI300X 的數字混用**。

---

## 速查卡

```bash
# ── 連線（Mac）──────────────────────────────────────────────
# 先開 OpenVPN（匯入信裡的 .ovpn，見 §2）
ssh <username>@192.168.20.241                    # head node：bcm11-headnode01

# ── 看資源 ────────────────────────────────────────────────
sinfo -N -l                                      # 節點狀態
squeue -A ChihChungHsu                           # 全 lab 現在用了幾張卡（上限 4）
quota -gs                                        # lab 的 1.5 T 還剩多少

# ── 互動式除錯（一張卡）──────────────────────────────────────
srun -p h200q -A ChihChungHsu --gres=gpu:1 -c 16 --mem=170G --time=02:00:00 --pty bash

# ── 正式量測（走 §5 的記錄殼）─────────────────────────────────
sbatch --job-name=<短名> code/tiara.sbatch python code/<腳本>.py ...
squeue -u $USER ; scancel <job_id> ; sacct -j <job_id>
```

---

## H200 第一天照抄（10/15）

```bash
# 0. VPN＋ssh 登入（§2）；第一次先 passwd
# 1. 環境（head node，§4.1）
ml miniconda3 && conda create -n tiara python=3.12 -y && conda activate tiara && pip install vllm==0.28.0
git clone https://github.com/s990093/paper-hierarchical-kv-state.git ~/paper-hierarchical-kv-state
cd ~/paper-hierarchical-kv-state
# 2. 模型與資料，會檢查 sha256（§4.4；約 31 GB＋0.5 GB）
bash code/h200_setup_data.sh
# 3. R0 硬體報告：拿一張卡跑（03 §R0）；最後會印出建議的 M7_DISK
srun -p h200q -A ChihChungHsu --gres=gpu:1 -c 16 --mem=170G --time=02:00:00 --pty bash
source code/env_h200.sh && bash code/h200_probe.sh
# 4. 把 M7_DISK 填進 code/env_h200.sh（或在 sbatch 前 export），離開 srun，送 R1–R3（一鍵驗證、正確性、第一階段復現、和 MI300X 比方向）
sbatch code/r_h200.sh            # 被砍時：sbatch code/r_h200.sh --from <步驟名>；先看步驟：bash code/r_h200.sh --list
```

- 第 4 步過了（`repro_verdict.json` 判「重現」），才進 03 的 D0／D1：E0 確認難度（`CANDIDATE_PARAMS.json`）、δ*、挑 scale 粒度 → **凍結事先登記** → 才跑非遮蔽的 `m10_quality.py run`。
- 結果每天 push、原始 run 目錄每天備份（§5.4）。

## 0. 這台機器的關鍵數字

| 項目 | 值 | 來源 |
|:--|:--|:--|
| GPU | NVIDIA H200 | 〔wiki〕範例輸出 `使用裝置：NVIDIA H200` |
| GPU 記憶體、架構 | NVIDIA 規格是 141 GB HBM3e、Hopper（sm_90）；**本機 NOT_MEASURED** | 規格，登入後用 `nvidia-smi` 確認 |
| 每節點幾張卡、CPU 型號、host↔GPU 的 PCIe 世代 | `hgx-cn01`：8 張 H200（`Gres=gpu:h200:8`，兩個 socket）、128 CPU、RealMemory 1,393,029 MB；CPU 型號、PCIe 世代 NOT_MEASURED（要進 job 跑 `h200_probe.sh`） | 2026-10-15 head node 上 `scontrol show node hgx-cn01` |
| Driver／CUDA 版本 | NOT_MEASURED | §3；決定 vLLM wheel 能不能直接裝 |
| 佇列 | `h200q`；單一 job 最多 48 小時 | 〔wiki〕 |
| 每張卡附帶 | 16 CPU＋170 GB RAM，**固定配給** | 〔wiki〕srun 輸出：`GPUs: 1 \| Fixed: 16 CPUs, 174080MB \| Assigned QOS: h200q_1g` |
| 上限 | 單一 job ≤ 4 GPU；**整個 lab 合計** ≤ 4 GPU、64 CPU、680 GB RAM | 〔wiki〕 |
| account（`-A`） | `ChihChungHsu`（許志仲老師的 lab） | 〔wiki〕account 清單 |
| 硬碟 | 一個 lab 1.5 T，lab 資料夾下有 `shared/` 共用；`quota -gs` 查剩多少。**2026-10-15 實測：group `ChihChungHsu` 已用 1524G／上限 1536G，只剩約 12G**（個人 `quota -us` 只有 52K；用量在別的成員目錄，私有、看不到）。模型 31G＋環境放不下，見 §7 | 〔wiki〕＋2026-10-15 `quota -gs`、`quota -us` |
| 個人目錄 | `$HOME`＝`/work/ChihChungHsu/<username>`；`/work` 是 `hgx-cn01:/work` 的 NFS（28T，叢集共用）。head node 的 `/tmp` 是本機 20G、`/dev/shm` 32G | 2026-10-15 `echo $HOME`、`df -hT` |
| head node／compute node | `bcm11-headnode01`（`192.168.20.241`，要先連 VPN）／`h200q` 只有 `hgx-cn01` 一個節點（另有 `dgx-cn01/02` 在別的 partition） | 〔wiki〕＋2026-10-15 `sinfo -N -l` |
| 預設 module | `slurm/24.11`、`gcc/14.2.0` 等；`ml miniconda3` 載入 conda（`/cm/shared/apps/miniconda3`） | 〔wiki〕 |
| 對外網路（head node、compute node） | head node：huggingface.co、github.com、pypi.org 都通（HTTP 200）；compute node NOT_MEASURED | 2026-10-15 `curl -sI`；模型可以在 head node 直接下載 |
| ⚠️ 資料保存 | **「資料不保證永遠存在，請記得備份」** | 〔wiki〕→ §5.4 |

---

## 1. 申請

表單要用**學校 Google 帳號**登入才看得到〔wiki〕。題目（2026-10-14 讀取，全部必填）：

| 題目 | 怎麼填 |
|:--|:--|
| 電子郵件 | 自動記錄登入的學校帳號 |
| 姓名 | 本人 |
| 聯絡電話 | 本人 |
| 學號（學生：學號／老師：姓名） | 本人學號 |
| Username（英文） | **小寫英文、不要空白**（它會變成 Linux 帳號；預設密碼＝帳號〔wiki〕，所以第一次登入就要改密碼） |
| Lab | 下拉選 **`ChihChungHsu`** |

- 申請通過後，會用 email 寄 OpenVPN 的帳號密碼〔wiki〕。等多久：NOT_MEASURED。
- 表單底下寫「請勿利用 Google 表單送出密碼」。
- **個資（電話、學號）不要寫進這個 repo**——repo 是 public。

---

## 2. 連線

wiki 寫的是 Linux 指令〔wiki〕：

```bash
openvpn3 session-start --config <檔名>.ovpn
ssh <username>@192.168.20.241        # 密碼預設和帳號相同
```

Mac 上的做法〔推論，未試〕：
- 用 OpenVPN Connect（或 Tunnelblick）匯入信裡的 `.ovpn`，輸入信裡的 VPN 帳密；
- 連上之後 `ssh`。**第一次登入先 `passwd` 改掉預設密碼。**
- 方便起見在 `~/.ssh/config` 加：

  ```
  Host h200
      HostName 192.168.20.241
      User <username>
  ```

  **ssh key 登入可以**（2026-10-15 `ssh-copy-id` 後 `ssh -o BatchMode=yes h200` 成功）。注意：VPN 帳密和 ssh 帳密是兩套——ssh 帳號＝表單的 Username、預設密碼＝帳號本身，**不是**信裡的 VPN 密碼。
- Mac 上實際的做法：`brew install openvpn`（2.7）。伺服器協商的是舊的 AES-128-CBC，設定檔要加 `data-ciphers AES-256-GCM:AES-128-GCM:AES-128-CBC`，否則 2.6 以後的版本會拒絕。VPN 是 split tunnel（只推 192.168.20.0/24 等內網路由），一般上網不受影響。開 VPN 要 root（建 tun）。

**3090 伺服器不能當 VPN 用戶端**〔推論〕：OpenVPN 要建 tun 介面，通常要 root，而 3090 上沒有 sudo（`CLAUDE.md` §3）。所以 3090 ↔ H200 之間沒有直接的路；模型照 §4.2 在 H200 那邊直接下載，結果備份經過 Mac。

---

## 3. 第一次登入：先量，再填表

> 這一節是手動的快速版。正式的硬體報告（比照 MI300X 的 `REPORT_MI300X.md`，由腳本產生）是精度計劃的 [§R0](precision_20261013/03_plan_h200.md)。

這些都是 NOT_MEASURED。每一條都影響後面的做法，量完把值填進下表，並在 `results/RUNLOG_H200.md` 記一節（照 `CLAUDE.md` §4.1 的殼，run_id 例：`<日期時間>-h200-probe`）。

```bash
# head node 上
echo $HOME; quota -gs; df -h $HOME
sinfo -N -l; scontrol show node hgx-cn01           # 每節點幾張卡、CPU、RAM、Gres
curl -sI https://huggingface.co | head -1          # head node 有沒有對外網路
squeue -h -o '%i %u %a %b %T %N' | head            # 看不看得到別人的 job

# 進一張卡（srun）之後
nvidia-smi; nvidia-smi -L                           # ★ 看得到幾張卡？只有自己那張，還是整台？
echo $CUDA_VISIBLE_DEVICES
nvidia-smi --query-gpu=index,name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max --format=csv
nvidia-smi topo -m                                  # NVLink（NV18 之類）、PCIe、這張卡的 CPU／NUMA affinity
nvidia-smi nvlink -s                                # NVLink 狀態（我們單卡、不用 TP，只是記錄）
lscpu | head -20; numactl -H; free -g; df -h /dev/shm
cat /sys/fs/cgroup/cpu.max /sys/fs/cgroup/memory.max 2>/dev/null   # job 真的能用多少（MI300X 上 nproc、MemTotal 都會騙人）
curl -sI https://huggingface.co | head -1          # compute node 有沒有對外網路
```

| 要確認的事 | 結果 | 影響 |
|:--|:--|:--|
| GPU 記憶體、driver、CUDA | H200 143,771 MiB、driver 610.43.02、700 W、SM 最高 1980 MHz；module 有 cuda12.8、cuda13.3 toolkit（2026-10-15 srun job 20293） | vLLM 0.28.0 的 wheel 能不能直接用（§4.1） |
| 每節點幾張卡、PCIe 世代與寬度、NVLink | 8 張 H200；PCIe 最高 Gen5 x16；job 裡 `topo -m` 只看得到自己這張，NVLink 看不到。CPU：2× AMD EPYC 9554（各 64 核、無 SMT），1.5 TB RAM（2026-10-15 srun job 20293） | 速度實驗「CPU 層不限速」那個錨點的意義。NVLink 只連 GPU 和 GPU，我們的 KV 走 CPU→PCIe→GPU，用不到 |
| 這張卡靠近哪個 NUMA 節點 | 這次拿到的卡在 NUMA 1，job 的 cpuset 是 80–95（同一邊）。**每個 job 拿到的卡可能不同**，要每次記錄（2026-10-15 srun job 20293） | pinned 記憶體要配在同一邊，不然 H2D 會變慢 |
| job 的 cgroup 真的給多少 CPU／RAM | `nproc`=16、cpuset 80–95；step 層的 `memory.max`=max（job 層的上限 NOT_MEASURED，srun 宣告 174,080 MB）；`free` 看到的是整台 1.5 TB，不能用（2026-10-15 srun job 20293） | wiki 說 16 CPU、170 GB；以 cgroup 為準 |
| job 裡 `nvidia-smi` 看得到幾張卡 | **只有 1 張**（`CUDA_VISIBLE_DEVICES=0`）→ 同節點別人的卡看不到，`gpu_guard` 必須回 `UNOBSERVABLE_HOST`（§5.3 的推論成立）（2026-10-15 srun job 20293） | **決定 `gpu_guard` 能不能用（§5.3）** |
| 看不看得到同節點別人的 job（`squeue -w`） | 看得到（head node 上 `squeue -w hgx-cn01` 列出別人的 job 與 gres）（2026-10-15） | §5.3 的替代記錄 |
| head／compute node 有沒有對外網路 | 都有（huggingface.co HTTP 200）（2026-10-15 srun job 20293） | 模型、pip 套件在哪裡下載 |
| `$HOME` 實際路徑、lab 剩餘空間 | `/work/ChihChungHsu/<username>`（NFS）；lab 剩約 12G（§0）。compute node 的 `/tmp` 在本機 `/dev/md0`（1.8T，剩 1.7T），不受 lab 配額限制但不保證保留〔推論〕 | `HF_HOME`、`TIARA_RUNS` 放哪裡 |
| `/dev/shm` 大小 | compute node 756G（整台共用）；實際能用多少受 job 記憶體上限約束（2026-10-15 srun job 20293） | CPU 層用 pinned memory，上限是 170 GB 的 job 配額 |

---

## 4. 環境

### 4.1 vLLM：用和 3090 相同的 0.28.0

```bash
ml miniconda3
conda create -n tiara python=3.12 -y
conda activate tiara
pip install vllm==0.28.0      # 會帶它自己要的 torch
pip freeze > ~/tiara_pip_freeze.txt
```

- **不要照 wiki 用 conda 裝 `pytorch-cuda=12.1`**：那是給單純 PyTorch 的；vLLM 的 wheel 綁了特定 torch 版本，混裝容易衝突〔推論〕。wiki 的 `intel-openmp<2024`、`mkl<2024` 降級也是那條路徑才需要的。
- 先看 §3 的 driver 版本，再裝；driver 太舊，wheel 會起不來。
- conda 的 Terms of Service 第一次會問，回 `a`〔wiki〕。
- 版本記錄：`pip freeze` 和 `python -c "import torch, vllm; print(torch.__version__, torch.version.cuda, vllm.__version__)"` 的輸出存進 `docs/` 下的環境記錄（比照 `migration_20261013/env/`）。

### 4.2 模型：用固定的 revision

和 MI300X、3090 用同一版（[`migration_20261013/README.md`](migration_20261013/README.md) §1.1）：

```bash
export HF_HOME=/work/ChihChungHsu/<username>/hf-cache     # 路徑照 §3 的 $HOME 改
huggingface-cli download unsloth/Llama-3.1-8B-Instruct --revision 4699cc75b550f9c6f3173fb80f4703b62d946aa5
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-1M  --revision e28526f7bb80e2a9c8af03b831a9af3812f18fba
```

- 兩個合計約 31 GB（16G＋15G，migration README 的表）。lab 是 1.5 T 共用，不要另外下載用不到的模型。
- head node 沒有對外網路的話：NOT_MEASURED 的情況，屆時再定（Mac 下載後 `rsync` 上去，約 31 GB）。

### 4.3 repo 與環境腳本

```bash
git clone https://github.com/s990093/paper-hierarchical-kv-state.git ~/paper-hierarchical-kv-state
cd ~/paper-hierarchical-kv-state
source code/env_h200.sh --print      # 印出所有路徑，標出哪些還 MISSING（R0 時用）
```

**以 `code/env_h200.sh` 為準**（2026-10-15 已寫好；這一節原本的範例已過時）。它做的事：

- 啟用 conda 環境 `tiara`：先試 `module load miniconda3`，不行再找 `/cm/shared/apps/miniconda3`；batch job 裡也能用（`eval "$(conda shell.bash hook)"`）。
- 路徑全部是 `${VAR:-預設}`，source 之前先 `export` 就能蓋掉：`REPO`、`TIARA_HOME`（預設 `$HOME`，R0 確認）、`HF_HOME`、`TIARA_RUNS`、`TIARA_DATASETS`、`TIARA_FONTS`、`PAPER_HKV_PYLIBS`、`M10_HOTPOT`。
- 模型指到**固定 revision 的 snapshot**（§4.2）：`M7_LLAMA_GLOB`、`M7_QWEN_GLOB`；`M7_MODEL_GLOB` 預設 Llama。
- 第一階段程式的平台參數：`M7_RES`（結果目錄 `results/m7_write_policy_h200`）、`M7_TIER_PARAMS`（限速層參數沿用 MI300X 的 `tier_params.json`，03 §R2）、`M7_DISK`（`Guard` 記磁碟 I/O 的裝置名，**預設 `R0_UNSET`，R0 的 `h200_probe.sh` 會給建議值，填好之前 Guard 會丟錯**，規則 7）、`M7_VLLM_VENV`（＝目前的 conda 環境）。
- `CUDA_DEVICE_ORDER=PCI_BUS_ID`（多卡時 torch 和 nvidia-smi 的編號順序一致，§5.3）。**不設 `CUDA_VISIBLE_DEVICES`**：交給 Slurm 依 `--gres` 設。
- 定義 `m7run <短名> <指令...>`：和平台 B 同介面，每一步用 `code/tiara.sbatch` 包成一個 run 目錄。
- 10/15 改：source 時**不再** mkdir（原本每次都建空目錄）。run 目錄由 `tiara.sbatch` 建，`$M7_RES` 由 `r_h200.sh` 建。

### 4.4 資料（不在 git 裡的；一鍵）

`bash code/h200_setup_data.sh`（head node；`--no-models` 只下資料、先試網路）會下載並逐一檢查 sha256：

| 東西 | 放哪 | 誰用 | 檢查 |
|:--|:--|:--|:--|
| Llama-3.1-8B-Instruct、Qwen2.5-7B-Instruct-1M（固定 revision，約 31 GB） | `$HF_HOME` | 全部 | revision 寫死 |
| HotpotQA dev distractor（HF `hotpotqa/hotpot_qa` rev 1908d6a，27 MB） | `$M10_HOTPOT` | `m10_tasks` T4 | sha256 `c20b638c…`（和 MI300X 相同） |
| LongBench v1 `data.zip`（HF `THUDM/LongBench`，108 MB）＋ repo 裡的 `code/longbench_config/` | `$M10_LONGBENCH` | `m10_longbench` | sha256 `cb45b11a…`（和 MI300X 9/15 下載的相同） |
| wonderwords 3.0.1 的詞表 | `$PAPER_HKV_PYLIBS` | `m10_tasks` T1、T2、T5 | 版本寫死 |

- 2026-10-15 在 MI300X 上用假的 `$TIARA_HOME` 試過（`--no-models`）：三個 checksum 都過；用下載的資料跑 `m10_tasks --self-test`（94／0）、`m10_longbench --self-test`（9,006／0）。
- head node 沒有對外網路時：在 Mac 照腳本裡的 URL 下載、rsync 到同樣路徑，再跑 `bash code/h200_setup_data.sh --no-download`（只檢查）。
- sha256 對不上就停下來回報，不准改 checksum 硬過。

`code/m7_env.sh` 在沒有 `/mlsteam/workspace/venv/tiara` 的機器上會自動改 source `env_h200.sh`，所以第一階段的 `m7_run_phase1*.sh` 不用改。

---

## 5. 跑實驗

### 5.1 規矩

- **head node 不跑 GPU，也不跑重的 CPU 工作**〔推論，一般 HPC 慣例；wiki 沒明寫〕。下載模型、`pip install`、git 可以。
- 正式量測一律 `sbatch`；`srun --pty bash` 只用來除錯。
- 每個 job 一張卡（`--gres=gpu:1 -c 16 --mem=170G`）。wiki 兩個 sbatch 範例的 CPU／RAM 寫得不一樣（16／170G 和 1／200G），但 srun 的輸出說每張卡是固定配給，所以照 16／170G 寫。
- lab 合計 4 張卡，**是和整個 lab 共用的**。開跑前 `squeue -A ChihChungHsu` 看一下。精度計劃 D2 的「Llama 和 Qwen 各一張卡平行」只佔 2 張。

### 5.2 記錄殼（`CLAUDE.md` §4.1 的 Slurm 版）

**以 `code/tiara.sbatch` 為準**（2026-10-15 已寫好；原本這裡的範本已過時，差別如下）：

```bash
sbatch --job-name=<短名> code/tiara.sbatch <指令...>      # 在 repo 根目錄送出
# 一個 job 裡依序跑多步：在 job 裡 source code/env_h200.sh，再用 m7run <短名> <指令...>（每步一個 run 目錄）
```

- **找 repo 的順序**：`$REPO` → `$SLURM_SUBMIT_DIR` → 這支檔案的上一層 → `$HOME/paper-hierarchical-kv-state`。sbatch 會把腳本複製到 spool，所以不能只靠 `$0`；原範本寫死 `$HOME/paper-...` 的 source 路徑已拿掉。
- **只用 `-o slurm-%x-%j.out`**（`*.out` 已 gitignore）。原範本另有 `-e slurm-%x-%j.err`，`.err` 沒有被 gitignore，拿掉了；stderr 一律進 run 目錄的 `stderr.log`。
- 產出：`$TIARA_RUNS/<RUN_ID>/{cmd.sh,context.txt,stdout.log,stderr.log,exit_code,cotenants_end.txt}`。
  - `cmd.sh` 用 `printf '%q '` 寫（含空白、引號的參數不會被拆壞；runsh 2026-10-10 的修正）。
  - `context.txt`：時間、`SLURM_JOB_ID`、節點、`CUDA_VISIBLE_DEVICES`、nvidia-smi（GPU 與 compute apps）、git HEAD 與未提交的檔數、torch 版本、`TIARA_／M7_／HF_HOME／CUDA_` 環境變數、`scontrol show job`、同節點的 job（`squeue -w`）。
  - `cotenants_end.txt`：結束時同節點的 job。
- 不在 Slurm job 裡（沒有 `SLURM_JOB_ID`）就拒跑（head node 不跑會產生數字的指令，§5.1）。
- 48 小時是硬上限。精度計劃估的品質實驗約 8–9 小時、速度約 1–2 小時〔`03_plan_h200.md` §3、§4.2，推論〕，單一 job 放得下；但腳本還是要能從中斷處接著跑（已完成的 `item_id` 跳過），被砍掉不用整批重來。

**§R（第一階段復現）一個 job 跑完**：`sbatch code/r_h200.sh`（R1 正確性 → R2 校準 → R3 復現 → 兩台並排比較）。`bash code/r_h200.sh --dry-run` 在任何機器上印出每一步的指令；被砍時 `sbatch code/r_h200.sh --from <步驟名>` 接著跑（`--list` 看步驟名；接著跑之前先處理被中斷那一步已附加寫入的 CSV 列）。判定由 `code/m7_repro_compare.py compare` 產生 `results/m7_write_policy_h200/repro_verdict.json`、`repro_vs_mi300x.csv`（只比方向，03 §R3）。

### 5.3 ⚠️ `gpu_guard` 在 Slurm 裡可能誤報「乾淨」（`CLAUDE.md` 規則 7）

- `host_contention()` 在 NVIDIA 上，是看 `nvidia-smi` 列出的**其他卡**上有沒有外來 process（`code/gpu_guard.py` 的 `host_contention`）。
- 如果 Slurm 限制了 job 看得到的裝置，job 裡的 `nvidia-smi` 只會列出自己那張卡〔推論；§3 要量〕。這時同節點其他卡上別人的 job 完全看不到，函式會回 `QUIET`。
- 這正是規則 7 說的「查不到被當成沒有」。PCIe 和 host 記憶體頻寬是整台共用的，別人的 job 會拖慢「把 KV 從 CPU 搬回 GPU」的量測（3090 上量過：灌水 26–52%，`gpu_guard.py` docstring 引 RUNLOG 發現 5）。
- **已改的程式**（2026-10-15）：在 Slurm job 裡，看得到的卡比節點上的卡少（或節點卡數從 `scontrol show node` 查不到），`host_contention()` 回 `UNOBSERVABLE_HOST`（和 AMD 容器同一個處理）；`python code/gpu_guard.py --host` 印出整機的判定，`--check <卡>` 在 Slurm 裡會多印一行 `host: <level>`。`squeue -w <node>` 的快照記在每個 run 目錄（§5.2）。
- **另一個風險：看錯卡**。如果節點沒有用 cgroup 限制裝置，job 裡的 `nvidia-smi` 會列出整台的卡，job 的 `cuda:0` 可能是 nvidia-smi 的 3 號；這時 `GpuWatcher(gpu=0)` 會去看別人的卡、對自己那張視而不見，污染被標成乾淨。所以 `m7_write_policy.py`、`m7_guard_run.py` 的 GpuWatcher 改用 `gpu_guard.smi_index(0)`：把 `CUDA_VISIBLE_DEVICES`（編號或 UUID）換成 nvidia-smi 的編號，**對不上就丟 `SmiUnavailable`，不猜**。`env_h200.sh` 設 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，讓兩邊的順序一致。R0 的報告（`h200_probe.sh`）記下 `CUDA_VISIBLE_DEVICES` 和 nvidia-smi 的編號／UUID，用來確認兩邊對得上；`r_h200.sh` 開跑前也會先呼叫一次 `smi_index(0)`，對不上就不跑。
- **時間數字怎麼辦**：獨占整個節點（`--exclusive`）會超過 lab 4 張卡的上限，不可行〔推論〕。能做的是精度計劃 §4.2 本來就有的設計：**同一次重複裡 6 個配置的順序隨機打散**，配對比較，抵消幾秒內不變的鄰居負載；再加上每個 run 的鄰居快照。報告要寫明「H200 上的時間數字，鄰居負載只能部分觀測」。
- 品質（分數）不受影響，照 `CLAUDE.md` §3：保留分數，記錄爭用。

### 5.4 備份（資料不保證存在）

- `results/` 的 CSV 摘要：每個 milestone 結束就 `git push`（head node 有網路的話；沒有的話經 Mac）。
- 原始 run 目錄：每天 `rsync` 到 Mac 或 3090 的 `/ssd7/hungwei/paper-hkv/runs/`（經 Mac 轉）。只留必要的 log，profiler trace 用完就刪，不佔 lab 的 1.5 T。

---

## 6. 精度計劃

已經改成在 H200 上跑：[`precision_20261013/03_plan_h200.md`](precision_20261013/03_plan_h200.md)。開頭有「10/14 改動」一節，寫了改了什麼、為什麼、在看到任何結果之前。重點：

- **先做 §R**（跟當初上 MI300X 一樣）：R0 硬體實測報告 → R1 環境與正確性 → R2 第一階段的校準 → R3 第一階段主要結論的復現。過了才開始精度實驗。
- 模型、長度、codec、attention 路徑、限速層、判準都沒改。
- 3090、MI300X 的數字不和 H200 的數字混用。

**等帳號的這段時間**：先寫和機器無關的程式（`m10_codec.py` 和 CPU 上能跑的單元測試、任務產生器、`env_h200.sh`、`tiara.sbatch`、`gpu_guard` 的 Slurm 判斷、§R0 的探測腳本）。會寫進 `results/` 的數字，全部等 H200。

---

## 7. 待辦（照順序）

1. 填表申請（本人，學校帳號；§1）。
2. 收到 VPN 帳密 → Mac 連線 → 改密碼（§2）。
3. 跑 §3 的探測，填表，記進 `results/RUNLOG_H200.md`。
4. 建環境、下載模型、記版本（§4）。
5. ~~寫 `code/env_h200.sh`、`code/tiara.sbatch`，改 `gpu_guard.py` 的 Slurm 判斷（§4.3、§5.2、§5.3）~~（10/15 在 MI300X 上寫好、做過語法與 mock 測試；NVIDIA 上還沒跑過）。
6. 照 [`03_plan_h200.md`](precision_20261013/03_plan_h200.md) §R 做硬體報告（`bash code/h200_probe.sh`，把建議的 `M7_DISK` 填進 `env_h200.sh`）與第一階段復現（`sbatch code/r_h200.sh`）→ 過了才開始 D0（計劃本身 10/14 已改成 H200）。
7. `CLAUDE.md` 加一節「平台 C」（機器表、Slurm 用法、備份規則），指回這一頁。
