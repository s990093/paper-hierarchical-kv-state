# 換機器的復現計劃：MI300X → 3090（2026-10-13）

**一句話**：程式和結果摘要都在 `main`。到新機器只要做三件事：
1. 照 §1 下載模型和資料；
2. 照 §2 的指令就能重跑 MI300X 上付費跑過的每一個實驗；
3. 研究方向看 §3。

---

## 1. 要下載什麼

### 1.1 模型（用 MI300X 上的同一版）

| 模型 | revision | 大小 | 用途 |
|:--|:--|:--|:--|
| `unsloth/Llama-3.1-8B-Instruct` | `4699cc75b550f9c6f3173fb80f4703b62d946aa5` | 16G | 第一階段主模型；第二階段速度 |
| `Qwen/Qwen2.5-7B-Instruct-1M` | `e28526f7bb80e2a9c8af03b831a9af3812f18fba` | 15G | 第二階段品質（對精度敏感） |
| `Yukang/LongAlpaca-7B` | `14c462dc860ba1818194ef0b5a31387ba57f0e4d` | 13G | 可選：MHA 對照；3090 上最多做到 16K |

```bash
export HF_HOME=/ssd7/hungwei/hf-cache   # 3090；先看裡面是不是已經有了
huggingface-cli download unsloth/Llama-3.1-8B-Instruct --revision 4699cc75b550f9c6f3173fb80f4703b62d946aa5
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-1M  --revision e28526f7bb80e2a9c8af03b831a9af3812f18fba
```

- 其他模型（Qwen3-30B、Seed-OSS-36B 等）3090 放不下，不用下載。
- 完整清單在 [`inventory/models.tsv`](inventory/models.tsv)。

### 1.2 資料

| 資料 | 怎麼拿 |
|:--|:--|
| Mooncake trace、GSM8K、RULER 參考、LongBench | 3090 原本就有（`/ssd7/hungwei/paper-hkv/datasets/`） |
| SCBench | 從 HF 下載 `microsoft/SCBench` |
| 字型 Noto Sans/Serif CJK TC | 3090 系統已有 |
| 論文 PDF（72 篇） | 檔名清單在 [`inventory/papers_list.txt`](inventory/papers_list.txt)，檔名有 arXiv 編號，可以重新下載 |

### 1.3 軟體

- 3090 用平台 A 原本的 venv：`/ssd7/hungwei/paper-hkv/venv/vllm`，裡面是 vLLM 0.28.0（CUDA）。
- MI300X 的版本記在 [`env/versions.md`](env/versions.md) 和 `env/pip_freeze_*.txt`。
  - 那些是 ROCm 版，不能直接照裝。
  - 用途是記下「當時的數字是在哪個環境跑出來的」。

### 1.4 不用搬的

- 每次 run 的原始 log（`/mlsteam/data/tiara/runs/`，48 GB）。
  - 結果摘要 CSV 已經在 git。
  - 每個 run 的指令也已經抄進 [`inventory/run_commands.tsv`](inventory/run_commands.tsv)。
- Claude 的對話紀錄。

---

## 2. 付費跑過的實驗怎麼復現

### 2.1 找某個數字的指令

結果 CSV 每一列都有 `run_id`。到 [`inventory/run_commands.tsv`](inventory/run_commands.tsv) 查同一個 `run_id`，就有當時的完整指令和 exit code。

- 共 768 個 run，488 個 exit 0。

### 2.2 用 GPU 的實驗（MI300X 上付費跑的）

所有指令都要先 `source code/m7_env.sh`，再用 `m7run <短名> <指令>` 包起來。一張卡上要依序跑，不要並行。

| 實驗 | 日期 | 怎麼重跑 | 結果 | 報告 |
|:--|:--|:--|:--|:--|
| 正確性測試 | 10/8 | `python code/test_m7_correctness.py <out.json>` | `results/m7_write_policy_mi300x/correctness.json` | 07 |
| 第一階段：校準 C0–C7、A0、A1、A2、B | 10/8 | `code/m7_run_phase1.sh`，再依序跑 `1b`、`1c`、`1d`、`1e`、`1f` | `results/m7_write_policy_mi300x/` | `docs/phase1_20261008/07_report.md` |
| 08 消融的 GPU 確認 | 10/9 | `code/m7_run_ablate.sh` | 同上，`b2_*.csv` | `09_ablation_report.md` |
| 破解第 1 輪：MHA／MoE 的重算曲線、存活設定上 GPU（0／5） | 10/9 | `code/m7_run_round1.sh`，再依序跑 `b`、`c`、`d`、`e` | `results/m7_explore_mi300x/` | `12_round1_report.md` |
| D2：真實 vLLM 會不會丟資料 | 10/10 | `code/m8_vllm_drop.py`（vLLM 0.28 的 venv） | `results/m8_directions/d2_*.csv` | `docs/research_20261010_directions/D2_vllm_drop.md` |
| F1：開卸載反而變慢的原因 | 10/10 | `code/m9_f1_run.py`（vLLM 0.28） | `results/m9_followup/f1_*` | `docs/research_20261010_followup/F1_offload_slowdown.md` |
| F2：能耗 κ | 10/10 | `code/m9_f2_energy.py` | `results/m9_followup/f2_*` | `F2_energy_kappa.md` |
| F3：反量化成本、FP8 | 10/10 | `code/m9_f3_dequant.py`、`code/m9_f3_fp8_repro.py` | `results/m9_followup/f3_*` | `F3_dequant_fp8.md` |

- 每個實驗的參數以 `run_commands.tsv` 裡的實際指令為準。上表只是入口。
- **不用 GPU 的**（D1、D3–D8 的模擬與分析、F4 稽核、F6、F7、給老師的圖）：
  - 直接跑 `code/m8_*.py`、`code/m9_*.py` 就好；
  - 任何機器都能跑，不用付費。
- **9/15–9/20 的舊實驗**（run_id 開頭 `m1`–`m6`）：
  - 程式在 10/4「推倒重來」時刪掉了；
  - 要看可以用 `git show 52803bc^:code/<檔名>`；
  - 現在的結論沒有用到它們。

### 2.3 在 3090 上重跑要改的地方

- **路徑**：40 個檔寫死了 `/mlsteam/...`。
  - 主要是 run 目錄、模型路徑、字型路徑。
  - 改成讀環境變數，或另外寫一份 `code/env_3090.sh`。
- **模型路徑**：`code/m7_model.py` 已經可以用 `M7_MODEL_GLOB` 換。
- **GPU 監看**：`gpu_guard.py` 已支援 `nvidia-smi`。
  - 3090 是共用機，開跑前要先挑空閒的卡。
- **VRAM 24 GB**：
  - Llama-3.1-8B 做到 32K 很緊〔算術，約 21–23 GB，沒量過〕；
  - LongAlpaca 最多做到 16K；
  - Qwen3-30B 放不下。
- **數字不會一樣**：3090 的重算比較慢、PCIe 是 4.0。
  - 比的是方向：Cake 是否比「只重算」「只搬」快、S5 是否和延後版差不多；
  - 不是比絕對值。
- **ROCm 專屬的不用重做**：F1（ROCm 上開卸載就換成慢的 attention）。

---

## 3. 研究方向

### 3.1 目前結論（第一階段做完，10/12 已報告老師）

- **寫入時依位置決定放哪一層（S5）沒有用**：在所有測過的條件下，都被「滿了再搬」的延後做法追平。
  - 原因：寫入時用到的資訊（位置），等到要搬的時候一樣看得到。
- **讀取時的 Cake 有用**：頻寬 0.3–4 GiB/s 時，比「只重算」或「只搬」快 1.4–1.9 倍。
- **還有一個漏洞**：我們的重算比真實 vLLM 慢約 2 倍。
  - 模擬已經補過（F7），結論不變；
  - 但還沒有 GPU 確認。

詳細見 [`../phase1_20261008/13_advisor_update_20261012.md`](../phase1_20261008/13_advisor_update_20261012.md)。

### 3.2 下一步（提案：寫入簡單、讀取聰明；等老師決定）

| 步驟 | 內容 | 3090 可以嗎 |
|:--|:--|:--|
| 0 | 補漏洞：讓重算用快的 attention，重量重算速度 | 可以；在 3090 上就是「harness vs vLLM 0.28」的速度比 |
| 1 品質 | Qwen2.5-7B-1M 撈針，比「全精度＋Cake」「低精度全部搬」「低精度＋Cake」「PatchKV 式」 | 可以；KV 存 FP8 可行，但 3090 沒有 FP8 運算，反量化成本要重量 |
| 2 速度 | 同上幾組，不同頻寬下的還原時間 | 可以 |
| 3 動態 | GPU 忙碌程度不同時，規劃器自己選重算多少來守住品質 | 可以 |

**還在等老師決定的事**：
- 方向：改成「寫入簡單、讀取聰明」？還是寫負面結果論文，或換題目？
- 能不能做有損實驗（上表步驟 1、3）？
- `main.tex` 怎麼處理？
- CVPR 要不要追？

**投稿目標**（`VENUES.md`）：
- ICPE 2027：全文 11/16。
- CCGrid 2027：全文 12/1。

---

## 附：這個資料夾裡有什麼

| 檔案 | 內容 |
|:--|:--|
| `inventory/models.tsv` | 所有模型、revision、大小、3090 要不要 |
| `inventory/datasets.tsv` | 所有資料集、怎麼重新取得 |
| `inventory/papers_list.txt` | 論文 PDF 檔名（PDF 本身有版權，不進 git） |
| `inventory/run_commands.tsv` | 768 個 run 的指令與 exit code |
| `inventory/runs_manifest.tsv` | 每個 run 目錄的大小 |
| `env/` | MI300X 的版本與套件清單、硬體實測報告 |
| `lab_bin/` | MI300X Lab 的環境腳本（`lab_up.sh`、`runsh` 等）。10/4 的「推倒重來」把 `code/mi300x/` 刪了，這是唯一備份 |
| `claude_memory/` | Claude 的記憶。複製到新機器的 `~/.claude/projects/<repo 路徑把 / 換成 ->/memory/`；`mi300x-env.md` 到 3090 要改寫 |
