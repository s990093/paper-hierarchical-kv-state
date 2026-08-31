# 交接：2026-08-31 22:00，移到另一台機器（獨立 SSD）

> 這一份是給**接手的人或 agent** 看的。
> 讀完這份 + `CLAUDE.md` 就能繼續，不需要讀完整的 `results/RUNLOG.md`。

---

## 0. 一句話現況

Milestone 4（Oracle go/no-go）**做完了，但答案取決於怎麼算**，
而其中兩個關鍵參數在這台機器上量不到：**dedicated SSD 的寫入頻寬**與 **AWQ 剖面的成本模型**。

| 怎麼算 | toolagent | conversation | 判定 |
|---|---|---|---|
| 只算 prefill | 11.51% | 12.58% | MARGINAL |
| 只算 prefill、只比可部署的 baseline | 15.67% | 18.33% | **GO** |
| **端到端（含 decode）** | **2.83%** | **3.34%** | 🔴 **NO_GO** |

設定：`llama-bf16` 剖面、GPU 48,128 token、SSD 512 GiB、NVMe、
Mooncake 真實 trace（中位 6.3K、最長 126K）。

---

## 1. 🔴 新機器要先做的三件事

### (1) 量 dedicated SSD 的持續寫入頻寬

```bash
python code/disk_bw.py --paths <你的 SSD 掛載點> --size-mib 16384 --repeats 3
```

**為什麼這是第一件事**：今天最大的發現之一是勝出的 baseline `tier_fs`
需要 **4,666 MiB/s** 的持續寫入，而這台機器的兩顆碟都做不到
（SATA QLC 181、NVMe 2,512 MiB/s）。**在獨立 SSD 上這個數字可能完全不同，
而它直接決定「哪些策略是可部署的」，進而決定 headroom。**

⚠️ **一定要用 16 GiB 以上的測試**。1 GiB 的短測會落在 QLC 的 SLC 快取裡，
在這台機器上樂觀了 2.7 倍（492 vs 181 MiB/s）。

量完之後把值填進 `code/m4_oracle.py` 的 `DEVICE_WRITE_MIBPS`
與 `DEVICE_FS_ROOT`。

### (2) 量 AWQ 剖面的成本模型

```bash
python code/m2_cost_model.py --gpu 0 --stage retrieval --model llama-awq
python code/m2_cost_model.py --gpu 0 --stage recompute --model llama-awq \
    --positions 0 4096 8192 16384 24576 49152 81920 114688
# qwen-awq 同上
```

**為什麼重要**：上面那個 NO_GO 是 **BF16 權重**剖面的結果。
BF16 權重有 15.2 GB，decode 每一步光讀權重就要 36.8 ms，所以 decode 佔 75% 的時間。
**AWQ-INT4 權重的 decode 固定項只有約一半**（qwen-awq 實測 18.158 ms/步），
decode 變便宜 → prefill 佔比上升 → 端到端 headroom 上升。

**在量到之前，不得宣稱那個 NO_GO 適用於論文的主設定（AWQ）。**

`--positions` 掃到 114,688 是為了終結目前 5.1 倍的外插
（重算成本的位置係數只擬合到 24,576，而工作負載最大位置 126,208，
12.6–20.5% 的存取落在外插區）。這需要 AWQ 的 KV 容量才做得到。

### (3) 量 GPU-FP8 / GPU-INT4 的反量化成本

```bash
python code/m2_cost_model.py --gpu 0 --stage retrieval \
    --tiers gpu_resident gpu_fp8 gpu_int4 --ctx 16384 --n-prefixes 1 \
    --csv-suffix _precision_tiers
```

論文的動作空間有六階，M2 只量了四階。
`code/m4_oracle.load_precision_tiers()` 已寫好並用假輸入驗證過，
資料一落地就會被讀進去；`code/m4_verdict.py` 的 §F2 會顯示現況。

---

## 2. 今天確立的事（不必重做）

### 會改變結論的

| # | 發現 | 證據 |
|---|---|---|
| 1 | **decode 佔 75% 的時間，放置對它無能為力** | `results/m4_oracle/ssd_sweep.csv` 的 `decode_ms` 欄 |
| 2 | **長度是 headroom 的主因，不是重用率**。同樣 58.9% 真實重用率下，8K 只有 8.3%，32K 有 18.7% | `results/m4_oracle/headroom_surface.csv` |
| 3 | **headroom 峰值在「請求 ÷ GPU 預算 ≈ 0.68」**，超過 1.0 就 thrash | 同上 |
| 4 | **`tier_fs` 需要 4,666 MiB/s，兩顆碟都做不到** → 它不是可部署的 baseline | `disk_bw*.csv` + `ssd_sweep.csv` |
| 5 | **ε 是（精度 × 任務）的性質**。GSM8K 上四個精度差 <3pp（無法區分），大海撈針上 100% → 0% | `gsm8k_precision_n1000.csv`、`needle_pilot_32k.csv` |
| 6 | **精度階只有一級可用**：INT8 95%、FP8 5%、INT4 0%。差別在縮放係數動態算 vs 未校正靜態值 | `needle_pilot_32k.csv`、`needle_ctx_sweep.csv` |
| 7 | **vLLM 0.28 V1 無法執行 DCA**（崩潰，非推論）→ Qwen2.5-1M 的有效位置範圍是 262,144 | `dca_probe.json` |
| 8 | **單請求的 KV 必須整份在 GPU**，卸載延長不了單請求上下文 | vLLM `_check_enough_kv_cache_memory` |
| 9 | **沒有公開資料同時具備長 context 與真實低重用率**。SCBench 745K/80%、Mooncake 6.3K/37–57% | RUNLOG |

### 模擬器的驗證狀態

* 對上 M3 實測：16K 差 **8%**、32K 差 **3%**（先前是 1.5 倍，已修）
* Oracle 的重算次數**恰等於強制未命中下限**（cascade 模式）
* 不變量檢查（`code/m4_invariants.py`）：命中守恆、強制未命中下限、Oracle 是上界
* 回歸測試（`code/test_m4_regression.py`）：線上策略 20/20 與修改前相同

---

## 3. 環境

```
repo          這個目錄（GitHub: s990093/paper-hierarchical-kv-state）
大檔案        /ssd7/hungwei/paper-hkv/（venv、模型、trace、每次 run 的原始 log）
venv          /ssd7/hungwei/paper-hkv/venv/vllm（Python 3.12、vLLM 0.28.0+cu129）
模型          Qwen2.5-7B-Instruct-1M-AWQ-noDCA（本機）、llama-awq（HF 快取）
trace         /ssd7/hungwei/paper-hkv/datasets/traces/*.jsonl（Mooncake）
```

**新機器要重建的**：venv（`uv` + `--index-strategy unsafe-best-match` 裝
GitHub release 的 `+cu129` wheel，PyPI 的是 CUDA 13 裝不起來）、模型、trace。
安裝細節見 `CLAUDE.md` §3。

---

## 4. 程式的四支主檔

| 檔 | 做什麼 |
|---|---|
| `code/m4_oracle.py` | 引擎：成本模型、trace 載入、`Sim`（線上策略與 Oracle） |
| `code/m4_invariants.py` | 開跑前後的自動不變量檢查 |
| `code/m4_sweep.py` | 掃描：`--axis {ssd,budget,length,semantics,prefix,surface}` |
| `code/m4_verdict.py` | **判定材料，全部從 CSV 自動產生，不手打** |

其餘：`m1_capacity` `m2_cost_model` `m3_baseline` `m5_quality` `disk_bw`
`dca_probe` `gpu_guard` `shm_gc`。

四支舊掃描腳本（`m4_ssd_sweep` `m4_budget_sweep` `m4_semantics_ablation`
`m4_by_length` `m4_longctx` `m4_prefix_probe`）已被 `m4_sweep.py` 取代，
等驗證無誤後可刪。

---

## 5. 未完成的事

| 項目 | 狀態 |
|---|---|
| AWQ 剖面的成本模型 | 🔴 阻擋端到端 NO_GO 的判定 |
| GPU-FP8 / INT4 的反量化成本 | 🔴 六階動作空間缺兩階 |
| 重算成本掃到 114,688 | 🔴 目前 5.1 倍外插 |
| dedicated SSD 的寫入頻寬 | 🔴 決定哪些策略可部署 |
| `fp8_per_token_head` 的檢索品質 | 🟡 我的 FP8 比較不公平（靜態 vs 動態縮放） |
| 512K 的延遲 | 🟡 設定已寫好（`m3_baseline.py` 的 `qwen-awq-int8-512k`），未執行 |
| SCBench 轉成 trace | ⬜ 真實的 745K 長 context |
| 按輸出長度分箱 | ⬜ 短輸出那群的 headroom 應遠高於平均 |
| 三場景（純 prefill／混合／純 decode） | ⬜ 純 decode 是證偽測試，預期 headroom ≈ 0 |
| 論文 §7 寫入今天的結果 | ⬜ 對照表在 `PAPER_DELTAS.md` |

---

## 6. 論文的狀態

* `main.tex` 19→20 頁，編譯 0 警告 / 0 overfull / 0 未定義引用
* **§6 是計畫、§7 是已完成的量測**，兩者已分開（今天改的）
* `§app:cost` 與 `§app:oracle` 已更新（磁碟頻寬、模擬器驗證、前綴語意）
* `PAPER_DELTAS.md` 列出還沒寫進去的結果與證據檔案

**§app:oracle 有一處要補**：目前寫「逐 block 的簡化對前綴結構的工作負載無害」，
那對 baseline 成立（探針顯示缺口後只有 0.000–0.464% 還在某階），
但漏了 Oracle 那半——Oracle 的逐出不是前綴感知的，會被罰 2.3–3.1pp。
