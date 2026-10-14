# 平台 C 操作手冊 — NVIDIA H200（學校 Slurm 叢集）

> 這份是**操作手冊**：怎麼申請、怎麼連、怎麼在 Slurm 上照本專案的記錄規則跑實驗，以及換到 H200 之後精度計劃要改什麼。
> 來源：叢集 wiki「H200 計算節點說明會」（<https://wiki.deepcat.dedyn.io/s/de18e791-6f86-4861-bb2f-50c9e848699a>，2026-10-14 讀取，六個子頁全文）與申請表單（<https://forms.gle/h7FzavvtEDunLLPL9>）。
> **標記**：〔wiki〕＝wiki 原文；〔推論〕＝我們推的、沒在叢集上試過；沒量的寫 NOT_MEASURED。登入後量到的值要填回 §3 的表，並記進 `results/RUNLOG_H200.md`（新開，比照 `RUNLOG_MI300X.md`）。
> 平台 A（3090）看 `CLAUDE.md`；平台 B（MI300X）看 [`MI300X_MLSTEAM.md`](MI300X_MLSTEAM.md)。
>
> 最後更新：2026-10-14

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

## 0. 這台機器的關鍵數字

| 項目 | 值 | 來源 |
|:--|:--|:--|
| GPU | NVIDIA H200 | 〔wiki〕範例輸出 `使用裝置：NVIDIA H200` |
| GPU 記憶體、架構 | NVIDIA 規格是 141 GB HBM3e、Hopper（sm_90）；**本機 NOT_MEASURED** | 規格，登入後用 `nvidia-smi` 確認 |
| 每節點幾張卡、CPU 型號、host↔GPU 的 PCIe 世代 | NOT_MEASURED | §3 |
| Driver／CUDA 版本 | NOT_MEASURED | §3；決定 vLLM wheel 能不能直接裝 |
| 佇列 | `h200q`；單一 job 最多 48 小時 | 〔wiki〕 |
| 每張卡附帶 | 16 CPU＋170 GB RAM，**固定配給** | 〔wiki〕srun 輸出：`GPUs: 1 \| Fixed: 16 CPUs, 174080MB \| Assigned QOS: h200q_1g` |
| 上限 | 單一 job ≤ 4 GPU；**整個 lab 合計** ≤ 4 GPU、64 CPU、680 GB RAM | 〔wiki〕 |
| account（`-A`） | `ChihChungHsu`（許志仲老師的 lab） | 〔wiki〕account 清單 |
| 硬碟 | 一個 lab 1.5 T，lab 資料夾下有 `shared/` 共用；`quota -gs` 查剩多少 | 〔wiki〕 |
| 個人目錄 | 大概是 `/work/ChihChungHsu/<username>` | 〔推論〕wiki 的 conda 範例路徑是 `/work/TTTTT/test4/.conda/envs/...`；登入後 `echo $HOME` 確認 |
| head node／compute node | `bcm11-headnode01`（`192.168.20.241`，要先連 VPN）／範例裡是 `hgx-cn01` | 〔wiki〕 |
| 預設 module | `slurm/24.11`、`gcc/14.2.0` 等；`ml miniconda3` 載入 conda（`/cm/shared/apps/miniconda3`） | 〔wiki〕 |
| 對外網路（head node、compute node） | NOT_MEASURED | §3；決定模型怎麼下載 |
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

  叢集允不允許 ssh key 登入：NOT_MEASURED（試 `ssh-copy-id h200`）。

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
| GPU 記憶體、driver、CUDA | NOT_MEASURED | vLLM 0.28.0 的 wheel 能不能直接用（§4.1） |
| 每節點幾張卡、PCIe 世代與寬度、NVLink | NOT_MEASURED | 速度實驗「CPU 層不限速」那個錨點的意義。NVLink 只連 GPU 和 GPU，我們的 KV 走 CPU→PCIe→GPU，用不到 |
| 這張卡靠近哪個 NUMA 節點 | NOT_MEASURED | pinned 記憶體要配在同一邊，不然 H2D 會變慢 |
| job 的 cgroup 真的給多少 CPU／RAM | NOT_MEASURED | wiki 說 16 CPU、170 GB；以 cgroup 為準 |
| job 裡 `nvidia-smi` 看得到幾張卡 | NOT_MEASURED | **決定 `gpu_guard` 能不能用（§5.3）** |
| 看不看得到同節點別人的 job（`squeue -w`） | NOT_MEASURED | §5.3 的替代記錄 |
| head／compute node 有沒有對外網路 | NOT_MEASURED | 模型、pip 套件在哪裡下載 |
| `$HOME` 實際路徑、lab 剩餘空間 | NOT_MEASURED | `HF_HOME`、`TIARA_RUNS` 放哪裡 |
| `/dev/shm` 大小 | NOT_MEASURED | CPU 層用 pinned memory，上限是 170 GB 的 job 配額 |

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
```

要新寫 `code/env_h200.sh`（比照 `code/m7_env.sh`），內容〔推論，未在叢集上試過〕：

```bash
# source 這支：平台 C（H200，Slurm）
module load miniconda3
eval "$(conda shell.bash hook)"          # batch job 裡 conda activate 需要這行
conda activate tiara
export REPO=$HOME/paper-hierarchical-kv-state
export HF_HOME=$HOME/hf-cache
export TIARA_RUNS=$HOME/tiara-runs
export M7_MODEL_GLOB="$HF_HOME/hub/models--unsloth--Llama-3.1-8B-Instruct/snapshots/*"
export PYTHONUNBUFFERED=1
# 不要設 CUDA_VISIBLE_DEVICES：Slurm 會依 --gres 自己設
```

- `code/m7_model.py` 的模型路徑預設是 `/mlsteam/...`，靠 `M7_MODEL_GLOB` 換。
- migration README §2.3：還有約 40 個檔寫死 `/mlsteam/...`。這週用得到的腳本才改，一律改成讀環境變數。

---

## 5. 跑實驗

### 5.1 規矩

- **head node 不跑 GPU，也不跑重的 CPU 工作**〔推論，一般 HPC 慣例；wiki 沒明寫〕。下載模型、`pip install`、git 可以。
- 正式量測一律 `sbatch`；`srun --pty bash` 只用來除錯。
- 每個 job 一張卡（`--gres=gpu:1 -c 16 --mem=170G`）。wiki 兩個 sbatch 範例的 CPU／RAM 寫得不一樣（16／170G 和 1／200G），但 srun 的輸出說每張卡是固定配給，所以照 16／170G 寫。
- lab 合計 4 張卡，**是和整個 lab 共用的**。開跑前 `squeue -A ChihChungHsu` 看一下。精度計劃 D2 的「Llama 和 Qwen 各一張卡平行」只佔 2 張。

### 5.2 記錄殼（`CLAUDE.md` §4.1 的 Slurm 版）

要新寫 `code/tiara.sbatch`〔推論，未在叢集上試過〕：

```bash
#!/bin/bash
#SBATCH --partition=h200q
#SBATCH --account=ChihChungHsu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=12:00:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err
# 用法：sbatch --job-name=<短名> code/tiara.sbatch <指令...>
source "$HOME/paper-hierarchical-kv-state/code/env_h200.sh"
RUN_ID=$(date +%Y%m%d-%H%M%S)-$SLURM_JOB_NAME
RUN=$TIARA_RUNS/$RUN_ID; mkdir -p "$RUN"
printf '%q ' "$@" > "$RUN/cmd.sh"; echo >> "$RUN/cmd.sh"          # 1. 指令
{ date -Is                                                        # 2. 環境
  echo "slurm_job_id=$SLURM_JOB_ID node=$SLURMD_NODENAME cuda_visible=$CUDA_VISIBLE_DEVICES"
  nvidia-smi --query-gpu=index,name,memory.used,driver_version --format=csv
  git -C "$REPO" rev-parse HEAD
  scontrol show job "$SLURM_JOB_ID"
  squeue -w "$SLURMD_NODENAME" -h -o '%i %u %b %T'                # 同節點有誰（§5.3）
} > "$RUN/context.txt" 2>&1
cd "$REPO"
RUN_ID=$RUN_ID bash "$RUN/cmd.sh" > "$RUN/stdout.log" 2> "$RUN/stderr.log"   # 3. 跑
echo $? > "$RUN/exit_code"
squeue -w "$SLURMD_NODENAME" -h -o '%i %u %b %T' > "$RUN/cotenants_end.txt" 2>&1
```

- 48 小時是硬上限。精度計劃估的品質實驗約 6–7 小時、速度約 1 小時〔`03_plan_h200.md` §3、§4.2，推論〕，單一 job 放得下；但腳本還是要能從中斷處接著跑（已完成的 `item_id` 跳過），被砍掉不用整批重來。

### 5.3 ⚠️ `gpu_guard` 在 Slurm 裡可能誤報「乾淨」（`CLAUDE.md` 規則 7）

- `host_contention()` 在 NVIDIA 上，是看 `nvidia-smi` 列出的**其他卡**上有沒有外來 process（`code/gpu_guard.py` 的 `host_contention`）。
- 如果 Slurm 限制了 job 看得到的裝置，job 裡的 `nvidia-smi` 只會列出自己那張卡〔推論；§3 要量〕。這時同節點其他卡上別人的 job 完全看不到，函式會回 `QUIET`。
- 這正是規則 7 說的「查不到被當成沒有」。PCIe 和 host 記憶體頻寬是整台共用的，別人的 job 會拖慢「把 KV 從 CPU 搬回 GPU」的量測（3090 上量過：灌水 26–52%，`gpu_guard.py` docstring 引 RUNLOG 發現 5）。
- **要改的程式**：在 Slurm job 裡，如果看得到的卡比節點上的卡少，`level` 要回 `UNOBSERVABLE_HOST`（和 AMD 容器同一個處理）。另外把 `squeue -w <node>` 的快照記進每一列（上面的殼已經記在 run 目錄）。
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
5. 寫 `code/env_h200.sh`、`code/tiara.sbatch`，改 `gpu_guard.py` 的 Slurm 判斷（§4.3、§5.2、§5.3）。
6. 照 [`03_plan_h200.md`](precision_20261013/03_plan_h200.md) §R 做硬體報告與第一階段復現 → 過了才開始 D0（計劃本身 10/14 已改成 H200）。
7. `CLAUDE.md` 加一節「平台 C」（機器表、Slurm 用法、備份規則），指回這一頁。
