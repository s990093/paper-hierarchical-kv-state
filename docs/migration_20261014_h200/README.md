# 改到 H200 跑：申請、使用規則、對實驗計劃的影響（2026-10-14）

**一句話**：用學校 Google 帳號填表申請，通過後會收到 OpenVPN 帳密；之後在 Slurm 的 `h200q` 佇列跑，一張 H200 配 16 CPU、170 GB RAM，單一工作最長 48 小時，整個實驗室最多 4 張卡。換到 H200 後顯卡記憶體不再是問題，但**所有時間數字都要重量，也不能和 3090、MI300X 的數字混用**。

**來源**：H200 計算節點說明會 wiki（`wiki.deepcat.dedyn.io`，2026-10-14 讀取，共 6 頁）。以下標〔文件〕的都出自那裡；標〔規格〕的是 NVIDIA 公開規格；標〔判讀〕的是我的推論；沒查到的寫〔未查證〕。wiki 自己寫「資料不保證永遠存在」，所以重點抄在這裡。

---

## 1. 申請

| 步驟 | 內容 | 狀態 |
|:--|:--|:--|
| 1 | 用 **學校帳號** 填 Google 表單（`https://forms.gle/h7FzavvtEDunLLPL9`）〔文件〕 | 待填：表單要先登入 Google，欄位還沒看到 |
| 2 | 申請通過後，會寄 OpenVPN 帳號密碼〔文件〕 | — |
| 3 | 連 VPN → ssh 登入 → 改預設密碼（見 §3） | — |

**表單上可能要填的資訊**（欄位還沒看到，等登入後補）：

| 項目 | 我們的答案（草稿） | 依據 |
|:--|:--|:--|
| 申請人、學號、電話 | （本人資料，不寫進公開 repo） | — |
| 學校帳號 | NYCU 學校信箱 | wiki 要求學校帳號 |
| 指導教授／計畫帳號（account） | 許志仲老師（Dr. Chih-Chung Hsu）；Slurm account 應為 **ChihChungHsu** | wiki 的 account 清單裡有這一個〔文件〕；通過後用 `sacctmgr show assoc user=$USER` 或通知信確認 |
| 用途 | 長上下文 LLM 推論的 KV cache 還原研究：比較 KV 在 BF16、FP8、INT4 下的品質與還原時間（單卡推論，不訓練） | `docs/precision_20261013/` |
| 預計資源 | 1–2 張 H200（同時最多 2 個工作），每個工作數小時，最長不超過 48 小時 | 見 §4 |
| 儲存 | 模型約 31 GB（Llama-3.1-8B 16 GB＋Qwen2.5-7B-1M 15 GB），資料與結果 < 100 GB | 〔算術〕 |

---

## 2. 叢集的規則（wiki 原文整理）

| 項目 | 規則 | 來源 |
|:--|:--|:--|
| 佇列 | `h200q`，GPU 是 H200 | 〔文件：佇列清單與說明〕 |
| 每張 GPU 配多少 | 16 CPU、170 GB RAM（固定配給） | 〔文件〕 |
| 單一工作 | 最多 4 張 GPU | 〔文件〕 |
| 同一個實驗室加總 | 最多 4 GPU、64 CPU、680 GB RAM | 〔文件〕 |
| 單一工作最長時間 | 48 小時 | 〔文件〕 |
| 儲存空間 | 一個實驗室 1.5 TB；Lab 資料夾下有 `shared` 共用資料夾；用 `quota -gs` 看剩餘空間 | 〔文件〕 |
| 資料保存 | 不保證永遠存在，自己備份 | 〔文件〕 |
| 送工作 | `sbatch` 一定要給 `--account=<計畫帳號>` | 〔文件：Slurm 常用指令〕 |
| 環境 | Environment Module（`ml`／`module`）；`ml miniconda3` 後用 conda 建環境，環境放在 `/work/<account>/<user>/.conda/envs/` | 〔文件：Module、miniconda〕 |
| 已知的坑 | conda 會裝出不能用的 Intel OpenMP／MKL 組合，要先 `conda install -c defaults "intel-openmp<2024" "mkl<2024"` 再裝 PyTorch | 〔文件：miniconda〕 |

---

## 3. 登入與第一次設定

```bash
openvpn3 session-start --config <檔名>.ovpn
```

```bash
ssh <username>@192.168.20.241
```

- 預設密碼和帳號相同〔文件〕。**第一次登入就用 `passwd` 改掉**〔判讀：預設密碼等於帳號，任何知道帳號的人都能登入〕。
- 互動式拿一張卡（測試、除錯用，10 分鐘）〔文件，account 換成自己的〕：

```bash
srun -p h200q --gres=gpu:1 --mem=170G -c 16 --time=00:10:00 -A ChihChungHsu --pty bash
```

- 常用：`sinfo`（佇列狀態）、`squeue -u $USER`（自己的工作）、`scancel <job_id>`、`sacct`（過去 24 小時的工作）〔文件〕。

**第一天的檢查清單**（每一項都記進 `results/RUNLOG.md`）：

| # | 做什麼 | 通過的標準 |
|:--|:--|:--|
| 1 | VPN、ssh、改密碼 | 能登入 |
| 2 | `quota -gs`；確認 `/work/ChihChungHsu/` 底下的空間和 `shared` 資料夾 | 知道可用多少 |
| 3 | `srun` 拿一張卡，跑 `nvidia-smi` | 看到 H200、記下 driver 與 CUDA 版本 |
| 4 | 計算節點能不能直接上網（HuggingFace、pip） | 不能的話，改在登入節點下載 |
| 5 | `ml miniconda3`，建 Python 3.12 環境，裝 vLLM 0.28.0 | `python -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.cuda.get_device_name())"` 正常 |
| 6 | 下載模型到 `/work/...`（Llama-3.1-8B-Instruct、Qwen2.5-7B-Instruct-1M，revision 照 `docs/migration_20261013/inventory/models.tsv`） | 大小對得上 |
| 7 | `git clone` repo，寫 `code/env_h200.sh`（路徑全部從這裡來） | `source` 後印出路徑 |
| 8 | `test_m7_correctness.py`（兩個模型） | 對 HF 的 logits、重算逐位元相同 |

---

## 4. 換到 H200 對實驗計劃的影響

### 4.1 變好的地方

| 項目 | 3090 | H200 | 影響 |
|:--|:--|:--|:--|
| 顯卡記憶體 | 24 GB | **141 GB**〔規格〕 | Llama 在 32K 不再緊；[05 §7](../precision_20261013/05_QA.md#7-顯卡要多大24-gb-夠嗎要不要-48-gb) 的記憶體顧慮全部消失。之後要做更長的 context、MHA 或更大的模型都放得下 |
| GPU 獨占 | 共用機器，要靠 `gpu_guard.py` 防插隊 | Slurm `--gres=gpu:1` 分到的卡是獨占的〔判讀〕 | 時間量測比較乾淨。但同一個節點上的其他卡可能有別人在跑，CPU、PCIe、記憶體頻寬仍可能被影響，**`GpuWatcher` 照用** |
| 主機記憶體 | 共用 | 每張卡固定 170 GB | 限速層用的 pinned 記憶體（Llama 每題約 7 GiB）綽綽有餘 |

### 4.2 要小心的地方

1. **架構不同（Hopper，sm_90）〔規格〕**：
   - vLLM 原生 FP8 在 H200 上會走 FlashAttention 3，**Q 也可能被轉成 FP8、attention 直接用 FP8 算**；3090 上是先轉回 BF16 再算〔程式碼，[cards/P04](../precision_20261013/cards/P04_vllm_fp8_calibration.md) §2、§2.1〕。
   - vLLM 官方部落格提過 Hopper 上 FP8 KV 的準確率問題（128K 撈針：BF16 91%、修正前 FP8 13%、修正後 89%），原因是 FA3 用 FP8 算 attention 的累加精度〔文件，P04 §2.1〕。
   - **所以這週計劃的做法更重要**：我們自己的 codec 把 FP8 反量化成 BF16，所有配置走同一個 attention kernel（[03 §1](../precision_20261013/03_plan_3090.md#1-固定的條件開跑前寫死)）。vLLM 原生 FP8 只當參考點，而且要寫明它在 H200 上「連 attention 都是 FP8」。
2. **時間數字全部重量**：H200 的重算比 3090 快很多〔判讀〕，κ（重算÷搬運）會變，Cake 有用的頻寬帶會移動。計劃用「會合點落在 block 16、32、48」來選頻寬（[03 §4.1](../precision_20261013/03_plan_3090.md#41-怎麼選頻寬)），這個做法換機器也成立。**H200 的數字不和 3090、MI300X 的數字放在同一張圖**。
3. **48 小時上限**：品質實驗要能分段跑、中斷後接著跑（依題號分批，每題算完就寫進 CSV）。
4. **實驗室最多 4 張卡，而且和同學共用**：Step 1 的兩個模型各一張卡就好，不要一次佔滿。
5. **1.5 TB 是整個實驗室共用**：模型放 `shared`（如果同學也用同一個模型），每個 run 的 log 只留必要的；結果摘要（CSV）進 git，因為叢集上的資料不保證保留。
6. **軟體版本**：vLLM 0.28.0 釘 `torch==2.13.0`〔程式碼，P04〕；叢集 wiki 的範例是 `pytorch-cuda=12.1`。要確認 driver 支不支援 vLLM 0.28 的 wheel 所需的 CUDA 版本〔未查證〕。裝不起來就照實記錄，不要換版本硬幹（CLAUDE.md §1 規則 2）。
7. **計算節點能不能上網**〔未查證〕：不能的話，模型、資料都在登入節點下載。

### 4.3 計劃要改的地方

| 計劃裡的 | 改成 |
|:--|:--|
| 03 的標題與 §1「3090 伺服器，一張 RTX 3090 24 GB」 | 「H200（Slurm `h200q`），一張 H200 141 GB」；機器確定後再改 03 |
| §1.1「Llama 在 32K 要 21–23 GB，放不下就降長度」 | 不需要了 |
| §1「先量 K／V 的分布」「torch 的 FP8 轉型」 | 照做；另外記 `torch.__version__` 與 GPU 型號 |
| §5.2「還原器為什麼比 vLLM 慢」 | 在 H200 上重做；vLLM 的預設 attention 會是 FA3（3090 是 FA2） |
| §7 時程 | 申請通過、環境裝好之後才開始算 D0 |
| 「老師：同一台機器」 | 機器換成 H200 後，這週的所有實驗都在 H200 上做，不混 3090 |

---

## 5. 我們的 sbatch 範本（照 CLAUDE.md §4.1 的記錄協定）

```bash
#!/bin/bash
#SBATCH --job-name=m10-step1
#SBATCH --partition=h200q
#SBATCH --account=ChihChungHsu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=12:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

module load miniconda3
source code/env_h200.sh

RUN_ID=$(date +%Y%m%d-%H%M%S)-m10-step1-${SLURM_JOB_ID}
RUN=$RUNS_ROOT/$RUN_ID
mkdir -p "$RUN"
CMD="conda run -n tiara python code/m10_quality.py --out $RUN ..."
printf '%s\n' "$CMD" > "$RUN/cmd.sh"
{ date -Is; echo "SLURM_JOB_ID=$SLURM_JOB_ID"; nvidia-smi; git rev-parse HEAD; } > "$RUN/context.txt"
bash "$RUN/cmd.sh" > "$RUN/stdout.log" 2> "$RUN/stderr.log"; echo $? > "$RUN/exit_code"
```

- `RUNS_ROOT`、模型路徑都在 `code/env_h200.sh` 裡設定（路徑等登入後確認）。
- `m10_quality.py` 還沒寫（[03 §6](../precision_20261013/03_plan_3090.md#6-要改的程式)）。

---

## 6. 還要問管理者或自己查的

1. 我們實驗室的計畫帳號是不是 `ChihChungHsu`（要問老師）。
2. 計算節點能不能上網、HuggingFace 能不能直接下載。
3. driver／CUDA 版本，能不能裝 vLLM 0.28.0。
4. 同一個節點有幾張 H200、別人的工作會不會和我們共用 PCIe／CPU（影響時間量測）。
5. `/work` 與實驗室資料夾的實際路徑、有沒有本地 SSD（之後若要做真實 SSD 錨點）。
