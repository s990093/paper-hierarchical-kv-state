# PoC 執行計畫 — RTX 3090（平台 A）

> **這份文件是給執行 agent 看的。** 論文（`main.pdf`）不要交給執行 agent，那是主張文件不是執行文件。
> 對應論文：`main.tex` §6。對應 idea：`01-Ideas/idea-20260828-hierarchical-kv-state.md`

---

## 0. 給執行 agent 的規則（先讀完再動手）

### 你的任務
在一張 **RTX 3090（24 GB, sm_86）** 上，為「階層式 KV 放置」的研究建立可信的量測基準，並回答一個決定性問題：**這個問題有沒有最佳化空間（headroom）**。

### 🔴 絕對禁令

1. **不准編造任何數字。** 沒跑出來就寫 `NOT_MEASURED`，不要估、不要推、不要「依經驗約為」。
2. **不准跳過失敗。** 指令失敗就記錄完整錯誤訊息並停下來回報，不要換個方式硬幹到有輸出為止。
3. **不准改寫論文。** 你的產出是 `results/` 底下的資料檔與 `RUNLOG.md`，不要碰 `main.tex`。
4. **不准跳過 Milestone 4 的 go/no-go 判定。** 那是整個研究的停損點。
5. **每一個數字都要能追溯到一條指令與一個輸出檔。** 記錄指令、時間戳、commit hash。

### 產出結構
```
results/
  RUNLOG.md              # 逐步流水帳：指令、時間、結果、失敗
  env.json               # 環境指紋（版本、driver、GPU）
  m1_capacity/           # 容量懸崖實測
  m2_harness/            # 量測工具鏈驗證
  m3_baseline/           # baseline 數據
  m4_oracle/             # Oracle 上界 ← 決定性
```

### 何時該停下來問人
- 任何 Milestone 的驗收條件沒過
- 出現你無法歸因的效能異常（例如 baseline 比預期慢 10 倍）
- Milestone 4 的 headroom < 10%

---

## 1. 環境建立

### 硬體前提
- RTX 3090, 24 GB GDDR6X, **compute capability 8.6 (Ampere)**
- ⚠️ **Ampere 不支援原生 FP8**。論文中所有 FP8 的討論只適用於平台 B（MI300X）。
  在 3090 上量化路線是 **AWQ / GPTQ（權重 int4）** 與 **KV cache int8**。
  **開工前先驗證 vLLM 在 sm_86 上實際支援哪些 `--kv-cache-dtype`，不要假設。**

### 安裝
```bash
python -V                      # 必須 3.12（vLLM wheel 限制）
nvidia-smi                     # 記錄 driver / CUDA version
uv venv && source .venv/bin/activate
uv pip install vllm            # 記錄實際裝到的版本
```

### 驗收 A1 — 環境指紋
產出 `results/env.json`，必須包含：
```json
{
  "gpu": "...", "vram_total_mb": 0, "compute_capability": "8.6",
  "driver": "...", "cuda": "...", "python": "...",
  "vllm_version": "...", "torch_version": "...",
  "flash_attn": "...", "timestamp": "..."
}
```

### 驗收 A2 — vLLM 能跑
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-1M --max-model-len 8192   # 先用短 ctx 確認能起來
curl http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"...","prompt":"hello","max_tokens":8}'
```
**通過條件**：回傳合法 JSON。失敗就停下來回報，不要繼續。

### 驗收 A3 — 卸載連接器可用
```bash
vllm serve <model> \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
    "kv_connector_extra_config":{"spec_name":"CPUOffloadingSpec","cpu_bytes_to_use":8000000000}}'
```
**通過條件**：啟動不報錯，且 log 中出現 offloading 相關訊息。
⚠️ 若此步失敗，**整個計畫要重新設計**——立刻回報，不要自己找替代方案。

---

## 2. Milestone 1 — 容量懸崖實測

### 為什麼先做這個
論文 §2.5 宣稱「3090 上 64K 可置入、128K 超出」，那是**算出來的**。
**你的任務是量出真的懸崖在哪**，因為它取決於實際的權重量化方式、activation 開銷與碎片化。
**不要拿論文的數字當結論，拿它當假設去驗證。**

### 模型（config 全部經 HuggingFace 驗證，2026-08-29）

**選模準則**：只納入會使 κ 產生可事先計算之變化的模型，且必須滿足兩個硬條件——
**(i) 原生 context > 評測長度**（否則 RoPE 外推會污染品質量測）；**(ii) 平台上留有足夠 KV 預算**。

| 模型 | 層 | KV head | KV/tok | **原生 ctx** | κ(3090) | 角色 |
|---|---|---|---|---|---|---|
| **Qwen2.5-7B-Instruct-1M** | 28 | 4 | **56 KiB** | **1,010,000** | 108 | **主力** |
| **Llama-3.1-8B-Instruct** | 32 | 8 | 128 KiB | 131,072 | 54 | 對照 |

**為什麼主力是 Qwen2.5-7B-1M**：它在 24 GB 上的容量懸崖約在 **315K token**，遠低於 1M 原生上限
→ 你量到的是**純粹的記憶體限制**，不是模型限制。
Llama-3.1-8B 的懸崖（~132K）與原生上限（131,072）幾乎重合，兩種限制無法分離，故只作對照。

**這兩個模型的 κ 差 2 倍，而且在同一張卡上** → 免費的第二個驗證軸。

### ❌ 明確排除（不要自己加回來）
| 模型 | 排除原因 |
|---|---|
| Qwen3 全系列（8B/14B/30B-A3B） | 原生僅 **40,960**，128K 評測需外推 |
| Qwen3-30B-A3B (MoE) | int4 權重就要 ~17 GB，僅餘 4.6 GB 給 KV |
| Llama-4-Scout (MoE) | 總量 ~109B，3090 放不下（**這是平台 B 的材料**） |
| MLA（DeepSeek 系列） | 改變 KV 大小公式本身，是不同的問題 |
| Hybrid SSM（Jamba） | 狀態不隨序列成長，是 Marconi 的地盤 |

### ⚠️ 權重精度 ≠ KV 精度（不要混在一起）

這是兩個**正交**的維度：

| 維度 | 誰決定 | 取值 |
|---|---|---|
| **模型權重精度** | 部署配置，**不是本研究的變數** | BF16 / AWQ-INT4 |
| **KV 儲存狀態** | **本研究的決策變數** | GPU-BF16 / GPU-INT4 / CPU-INT8 / SSD / Drop |

Tiara **不修改模型權重**。權重精度只影響「還剩多少空間給 KV」，屬於敏感度分析，不是主軸。
**不要因為看到「主流部署有 INT4」就把五個 KV state 也改成各種權重精度。**

⚠️ **模型實際發布的是 `bfloat16` 不是 `float16`**（6/6 個 config 驗證）。
兩者都是 2 bytes 所以算術不變，但記錄時一律寫 **BF16**。

### 做法
對每個 (模型, 權重精度, KV dtype) 組合，二分搜尋最大可用 `--max-model-len`：

| 模型 | 權重 | 預期懸崖 | 用途 |
|---|---|---|---|
| Qwen2.5-7B-Instruct-1M | **AWQ-INT4** | ~315K | **主力設定** |
| Qwen2.5-7B-Instruct-1M | BF16 | 較早 | 敏感度：量化對 KV 預算的影響 |
| Llama-3.1-8B-Instruct | AWQ-INT4 | ~132K | 對照（κ 差 2 倍） |

⚠️ Ampere（sm_86）**不支援原生 FP8**，平台 A 沒有 FP8 設定。FP8 是平台 B 的事。

```bash
# 對每個設定，逐步加大直到 OOM，記錄最後成功值
for L in 8192 16384 32768 65536 131072; do
  vllm serve $MODEL --max-model-len $L --gpu-memory-utilization 0.90 ...
done
```

### 產出 `results/m1_capacity/capacity.csv`
```csv
model,weight_dtype,kv_dtype,max_model_len_ok,max_model_len_oom,kv_gb_at_ok,note
```

### 驗收 M1
- [ ] 至少一個設定找到明確的 OOM 邊界
- [ ] `RUNLOG.md` 記錄每次 OOM 的完整錯誤
- [ ] 在 `capacity.csv` 的 `note` 欄寫下**實測懸崖與論文 §2.5 算術值的差距**

---

## 3. Milestone 2 — 量測工具鏈

### 🔴 成本常數是 2×5 矩陣，不是 1×5 向量

這是最容易做錯的地方。**兩類成本的性質完全不同，必須分開量**：

| 動作 | **平時成本**（閒置時持續付出） | **被需要時的成本**（一次性） |
|---|---|---|
| GPU BF16 | GPU 位元組 | ≈ 0 |
| **GPU FP8** | GPU 位元組 ÷ 2 | 反量化 kernel |
| GPU INT4 | GPU 位元組 ÷ 4 | 反量化 kernel |
| CPU INT8 | host 位元組 ÷ 2 | PCIe + 反量化 |
| SSD | 磁碟位元組 | NVMe I/O + PCIe |
| **Drop** | **0** ← 這是它唯一的價值 | 重算(位置) |

⚠️ **Ampere（3090）不支援原生 FP8** → 平台 A 量不到 `GPU FP8` 這一階，
在 `cost_model.json` 中標記為 `NOT_SUPPORTED`，**不要用估的值填**。FP8 是平台 B 的事。

### 🔴 Drop 有依賴限制，不能獨立決定

重算 block *i* **不需要**從 token 0 重跑——用 chunked prefill 只重跑 block *i* 的 token，
attention 讀取前序 block 既有的 KV 即可。所以成本是 O(|block|) 不是 O(i)。

**但前提是前序 block 的 KV 還在：**

```
a_i = Drop 可行  ⟺  對所有 j < i，block j 的 KV 都取得得到
```

若前面也被丟了，就得先重算它們，成本沿依賴鏈累積。

**量測時要做的事**：除了單一 block 的重算成本，還要量
**「連續丟棄 N 個 block 後，重算第 N+1 個的成本」**（N = 1, 2, 4, 8），
產出 `recompute_chain.csv`。這條曲線決定策略裡「連續 Drop 上限」該設多少。

**若壓成一維，你會得到「重算最貴所以永不重算」的錯誤結論。**

⚠️ `C_recompute` **不是常數**，隨 block 的絕對位置成長（attention 二次項）。
要量成 `C_recompute(position)`，不是單一純量。對 MoE 模型，用 **active 參數量**不是總參數量。

**這組常數每個 (平台, 模型) 只量一次**，存成 `results/m2_harness/cost_model.json`，之後 policy 直接查用，不要每次實驗重量。

### 必須能量到的四件事
| 項目 | 工具 | 驗收 |
|---|---|---|
| **PCIe 傳輸量** | `nsys profile` 或 `torch.cuda` events | 能分辨 H2D / D2H 的位元組數 |
| **TTFT / TPOT** | vLLM `/metrics` 或 benchmark script | 兩者分開 |
| **GPU 記憶體** | `torch.cuda.memory_stats()` | KV pool 佔用可單獨取得 |
| **輸出品質** | 見下 | 可重現 |

⚠️ **NVIDIA 平台沒有 MI300X 的 `hbm_energy_acc`**，能耗只能量整卡。論文 §6.7 的能耗分析**不適用於平台 A**，不要嘗試。

### 品質評測（這一項最容易做錯）
**不要只用 needle-in-a-haystack。** 論文 §6.7 引用 YAKV 的發現：某些方法在單針測試上表現良好，卻在多事實抽取上崩潰。

- 主要判準：**multi-fact extraction**（YAKV 的 Text2JSON，`github.com/yandex-research/context-intensive-kv-offloading`）
- 次要：LongBench 子集
- sanity check：needle-in-a-haystack

✅ **選模準則已排除外推問題**：兩個模型的評測長度都在原生範圍內（Qwen2.5-7B-1M 懸崖 315K ≪ 1M 上限；Llama-3.1-8B 的 131K 在原生範圍內）。**不要為了掃更長的 context 而開 YaRN**——那會讓品質下降的來源無法歸因，ε 就失去意義。若真的需要超過原生長度，必須額外建立「full-KV + 相同外推設定」的基準線。

### 驗收 M2
- [ ] 同一組設定跑三次，TTFT 變異 < 10%
- [ ] 品質分數可重現（同 seed 同結果）
- [ ] 產出 `results/m2_harness/repeatability.csv`

---

## 4. Milestone 3 — Baseline

> **3090 是 CUDA，所有 baseline 都原生可跑。** 論文表 7 的「移植成本」欄位是給平台 B（ROCm）用的，
> **在這台機器上不適用**。這正是平台 A 的價值——特別是 Tier 3 那批在 MI300X 上跑不動。

### 🟢 Tier 0 — 必跑，零安裝成本（全部跑完再往下）

| # | Baseline | 怎麼跑 |
|---|---|---|
| 1 | **Full GPU（無卸載）** | 不加 `--kv-transfer-config` ← 品質上界 |
| 2 | **vLLM `lru`** | `"eviction_policy":"lru"` ← **最誠實的對手** |
| 3 | **vLLM `arc`** | `"eviction_policy":"arc"` ← 堵住「只贏最笨的」 |
| 4 | **vLLM + `fs` 磁碟層** | 加 `secondary_tiers: [{"type":"fs","root_dir":"..."}]` |
| 5 | **LMCache** | 官方 wheel |

**若 Tier 0 跑不順，停下來回報，不要往下做。**

### 🔴 然後直接做 Milestone 4（Oracle）

**不要跑完 Tier 1–3 才做 Oracle。** Oracle 只需要 Tier 0 的 workload 就能算，
而它決定整個研究要不要繼續。**先做 Oracle，再決定要不要投入 Tier 1–3 的移植工。**

### 🟡 Tier 1 — 系統類學術對手（Oracle = GO 之後才做）

| Baseline | 為什麼要 |
|---|---|
| **InfiniGen**（OSDI'24） | 唯一「真的預測未來 + 完整 CPU 階層」的系統，**審稿人一定會問** |
| **KIVI**（ICML'24） | 壓縮流派代表 |

### 🔴 Tier 2 — 學習式對手：⚠️ **三個都沒有釋出權重**（2026-08-29 逐一查證）

| | 有 checkpoint？ | 綁定模型 | 訓練需求 |
|---|---|---|---|
| **KVP**（`apple/ml-learning-to-evict`） | ❌ 只有訓練碼 | 只支援 Qwen2.5-7B-Instruct | **8 GPU DDP × 112 個 agent** |
| **ForesightKV**（`RUCAIBox/ForesightKV`） | ❌ | Qwen3 / Qwen2 各一套腳本 | **≥2 張卡** + 多卡 RL |
| **LookaheadKV**（`SamsungLabs/LookaheadKV`） | ❌ | Llama 的「最小範例」 | LoRA 微調，需求未載明 |

**在單張 3090 上訓練這三個，時程上不可行。** 不要嘗試，也不要為了湊 baseline 硬跑一個縮水版。

### ✅ 正確的替代做法：把比較移到消融，不要移到外部系統

論文的核心主張是「既有學習式方法的**對稱損失**在異質動作成本下失準」。
跟 KVP 的**整個系統**比，會混入太多無關差異（它單層、你六階；它 RL、你 GBDT）。

**真正檢驗主張的比較是「同一個系統只換損失函數」**，也就是 Milestone 5 消融表的 (B) 區塊：

```
你的系統 + 對稱 L2       ← 這就是既有方法在做的事
你的系統 + 成本敏感損失   ← 本文主張
              ↓
          差多少？
```

**這個實驗完全由你掌控，不需要訓練任何外部系統。**
→ 消融表 (B) 區塊因此是**主要證據**，不是支持證據。

**外部學習式 baseline 的處理方式**（優先序）：
1. 若某篇的**已發表數值**與你的評測設定可比 → 引用其論文回報值，明確標註「引用自原論文，未重跑」
2. 若時間允許且能取得多卡 → 挑**一個**（建議 LookaheadKV，LoRA 訓練成本最低）實跑
3. 在 §7 Limitations 誠實寫明：三者均未釋出權重，且皆為模型專屬，故未能於本文的模型上重跑

### ⚪ Tier 3 — 3090 專屬（挑 2–3 個，不要全做）

**ArkVale、Quest、ShadowKV**

這批因 `rapidsai/raft` 無 ROCm 版而在 MI300X 上**結構性地跑不動**。
在 3090 上跑得動，就是平台 A 存在的理由之一。
（ClusterKV / RetroInfer / ParisKV 機制高度重疊，可略。）

### ❌ 建議略過
- **FlexGen** — InfiniGen 建在它上面，跑 InfiniGen 就涵蓋
- **KVPR** — 做的是重算的「切分比例」，與本文 Drop 的定位不同

### 壓力軸（平台 A 專用）
論文 §6.2：**3090 上用「單請求 × context 遞增」**，不是並行度。
`context ∈ {8K, 32K, 64K, 128K, 256K, (M1 實測的懸崖點)}`
（Qwen2.5-7B-1M 的懸崖在 ~315K，所以可以掃到比 Llama 更遠）

### 產出 `results/m3_baseline/baseline.csv`
```csv
baseline,model,ctx,ttft_ms,tpot_ms,peak_vram_mb,pcie_h2d_mb,pcie_d2h_mb,quality_score,quality_metric,seed,runs,ts
```

### 驗收 M3
- [ ] baseline 1–5 全部有數據
- [ ] `lru` 與 `arc` 有可辨別的差異（若完全相同，代表卸載沒真的發生 → 回頭查）
- [ ] 品質分數與 Full GPU 的差距有記錄

---

## 5. 🔴 Milestone 4 — Oracle 上界（決定性）

### 這一步決定整個研究要不要繼續

**問題**：一個「知道未來」的完美策略，比最好的簡單策略好多少？

### 做法（作弊法）
1. 用 M3 的同一組 workload，**先完整跑一次**，記錄每個 KV block **實際上**有沒有被後續 attention 用到
2. 用這個「未來知識」離線求解最佳放置（整數規劃或貪婪近似皆可，**記錄你用的是哪一種**）
3. 在相同的記憶體預算下，比較：
   - Full GPU（品質上界）
   - 最好的 baseline（通常是 `arc`）
   - **Oracle**

### 產出 `results/m4_oracle/oracle.csv` + 一張 Pareto 圖
```csv
policy,ctx,memory_budget_gb,quality,ttft_ms,note
```

### 🚦 GO / NO-GO 判準（照這個做，不要自己解釋）

| Oracle 相對最佳 baseline 的改善 | 判定 | 你該做什麼 |
|---|---|---|
| **> 15%** | 🟢 **GO** | 回報，繼續 Milestone 5 |
| **5–15%** | 🟡 **邊緣** | **停下來問人**，附上 Pareto 圖 |
| **< 5%** | 🔴 **NO-GO** | **停止。** 完整回報並說明「此方向無 headroom」 |

**NO-GO 不是失敗。** 那是一個有價值的負面結果，會省下數個月。**不要為了讓專案繼續而美化數字。**

---

## 6. Milestone 5 — Policy（只在 GO 之後做）

先做**不需要訓練**的版本：

1. **成本模型 + recency**（不訓練）— 若這個就贏 `lru`/`arc`，代表成本模型本身有價值，這是乾淨的中間結果
2. 再加**未來效用預測器**（GBDT，見 `main.tex` §5.2）
3. 最後才是成本敏感損失的比較（§5.3）

**在 Oracle 出來之前不要訓練任何模型。**

---

## 7. 每個 Milestone 都要回報的格式

```markdown
## Milestone N — <名稱>
**狀態**: PASS / FAIL / BLOCKED
**執行時間**: <起> → <迄>
**指令**: <實際跑的指令，可複製貼上>
**產出檔**: <路徑>
**關鍵數字**: <只寫實際量到的>
**失敗與異常**: <完整錯誤訊息，沒有就寫「無」>
**與論文假設的差異**: <實測 vs main.tex 的預期>
```

---

## 8. 已知風險（遇到就回報，不要自己繞過）

| 風險 | 徵兆 | 處置 |
|---|---|---|
| Ampere 不支援某些 KV dtype | `--kv-cache-dtype fp8` 報錯 | 記錄後改用 int8 或 fp16，**並在 RUNLOG 註明** |
| Python 版本錯誤 | `libcudart.so: cannot open` | vLLM 靜默裝成 CUDA wheel，檢查 Python 是否 3.12 |
| 卸載沒真的發生 | `lru` 與 `arc` 數字完全相同 | 檢查 PCIe 流量是否為 0 |
| 品質分數不穩 | 同 seed 不同結果 | 檢查 temperature、是否啟用了非決定性 kernel |
| 24 GB 放不下目標設定 | OOM | 回到 M1 重新選權重量化方式 |

---

## 9. 明確不要做的事

- ❌ 不要在平台 A 做能耗結論（消費卡無記憶體/計算分軌計數器）
- ❌ 不要在平台 A 做多租戶／機會成本結論（24 GB 放不下多個長 session）
- ❌ 不要嘗試移植 ArkVale / Quest / ShadowKV / ClusterKV（它們在 3090 上**可以跑**，但那是 Milestone 6 的事，不要提前）
- ❌ 不要修改 `main.tex`
- ❌ 不要在 Oracle 之前調 policy 的超參數

---

## 相關文件
- 論文：`main.tex`（§6 為實驗計畫的完整版）
- Gap 分析：`../report-hierarchical-kv-state-20260828.md`
- AMD 可行性：`../report-hierarchical-kv-state-amd-feasibility-20260828.md`
- KV 容量計算器：`../../00-Inbox/kv-footprint-calculator.py`
