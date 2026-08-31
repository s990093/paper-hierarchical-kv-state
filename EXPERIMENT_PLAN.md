# PoC 執行計畫 — RTX 3090（平台 A）

> **這份文件是給執行 agent 看的。** 論文（`main.pdf`）不要交給執行 agent，那是主張文件不是執行文件。
> 對應論文：`main.tex` §6。對應 idea：`01-Ideas/idea-20260828-hierarchical-kv-state.md`

---

> ## ⚠️ 2026-08-31 更新：本計畫書有九處已被實測推翻
>
> 下表列出「計畫書原本怎麼寫」與「實測是什麼」。**遇到衝突時以實測為準**，
> 每一項都可在 `results/RUNLOG.md` 追溯到指令與輸出檔。
>
> | # | 計畫書原本寫的 | 實測 | 影響 |
> |---|---|---|---|
> | 1 | Ampere 不支援 FP8，平台 A 無此設定 | **可用**，容量恰好 2.00× | 動作空間五階單卡可驗 |
> | 2 | 只有 BF16 與 FP8 兩個 KV dtype | 另有 `int8_per_token_head`、`int4_per_token_head` | **GPU 上的四階是同一旗標的取值** |
> | 3 | 精度收益 2×/4× | **2.00× / 1.94× / 3.77×** | 量化中繼資料固定佔 2 KiB/token |
> | 4 | 重算成本隨位置**二次**成長 | **線性**，擬合偏差 <1.6% | §3 與論文 §2.6 的措辭要改 |
> | 5 | SSD 與重算「同數量級」 | 成立，但**次序在位置 7,277 token 處反轉** | 128K 時 94.4% 的 block 該用 SSD |
> | 6 | Python 版本是 vLLM 的限制 | **CUDA 版本才是**（見 §1） | 安裝流程要改 |
> | 7 | Qwen2.5-1M 可直接使用 | **vLLM 0.28 的 V1 沒有 DCA 路徑** | 要用 no-DCA 變體，上限 262,144 |
> | 8 | 主力設定是 AWQ-INT4 | 正確，但**先前一路跑 BF16**，導致容量只有 41,648 | 整個實驗曾鎖在短 context |
> | 9 | 品質用 needle/LongBench | 補上 **GSM8K many-shot**，且 n=120 不足以分辨 5pt | 需 n≈1000 |
> | 10 | Mooncake 的 `hash_ids` 可直接當 block 用 | **hash_id 是 512 token，不是 16** | 工作集與**位置**都少算 32 倍 |
> | 11 | SSD 階容量不是限制 | 工作集要 **10.4 TiB**，NVMe 只有 3.6 TiB | 「無限 SSD」是不可實作的設定 |
> | 12 | 成本模型只需要「讀回來」的價格 | **寫下去不是免費的**；`tier_fs` 需要 4,666 MiB/s | 見下方 §0.5 |
> | 13 | 磁碟階用哪顆碟只影響數字大小 | **決定策略能不能部署**（SATA 181 vs NVMe 2,512 MiB/s） | 2026-08-31 決定：**磁碟階只用 NVMe** |

---

## 0.5 🔴 寫入頻寬是一個獨立於延遲的判準（2026-08-31 新增）

原本的 go/no-go 只看一個數字：Oracle 相對最佳 baseline 的延遲節省。
但模擬的成本模型**只向「把 block 讀回來」收費，寫下去是免費的**，
而真實硬體不是。

實測（`code/disk_bw.py`，O_DIRECT，三次中位）：

| 裝置 | 1 GiB 短測 | **16 GiB 長測（持續值）** |
|---|---|---|
| Samsung 870 QVO（SATA QLC，`/ssd7`） | 寫 492 MiB/s | **寫 181 MiB/s** |
| Crucial P3（NVMe，`/`） | 寫 2,512 MiB/s | — |

短測落在 QLC 的 SLC 快取裡，樂觀 2.7 倍。**KV 階是持續寫入，只能用長測值。**

各策略需要的持續寫入頻寬（toolagent、48,128 token 預算）：

| 策略 | 需要 | SATA 181 | NVMe 2,512 |
|---|---|---|---|
| `cpu_arc` / `cpu_lru`（不用磁碟） | 0 | ✅ | ✅ |
| **`tier_fs`（無差別下放）** | **4,666 MiB/s** | 🔴 超出 26× | 🔴 超出 1.9× |
| **Oracle（成本感知）** | **1,114 MiB/s** | 🔴 超出 6.2× | ✅ |

`tier_fs` 在**兩顆碟上都不是可部署的策略**。下界檢查：即使假設
load-and-keep（每個不重複 block 只寫一次），仍需 3,086 MiB/s，結論不變。

**因此每一份 Oracle 結果都必須同時回報三件事**：
1. headroom（延遲軸）
2. 該策略需要的持續寫入頻寬
3. 目標裝置的實測持續寫入能力

`code/m4_verdict.py` 的 §G 會自動做這個比較，並且**只比在該裝置上跑得起來的策略**。

### 🔴 三組必須一起切換的參數

今天踩到三次「只切一半」的錯誤，全部是同一個形狀：

| 群組 | 成員 | 只切一半會怎樣 |
|---|---|---|
| **模型剖面** | GPU 預算、KV 每 token 位元組、成本常數的來源模型 | qwen-awq 的預算配 llama 的成本 → 預算大 5.7 倍，結論反轉 |
| **裝置** | 成本常數、持續寫入上限、掛載點 | SATA 的成本配 NVMe 的頻寬 → 可行性判定不成立 |
| **trace 解碼** | block 粒度、位置、工作集 | hash_id 當成 16 token → 位置少算 32 倍，重算太便宜 |

`m4_oracle.MODEL_PROFILES` / `DEVICE_WRITE_MIBPS` / `DEVICE_FS_ROOT` 把三組各自綁死，
`load_cost_model(require_model_key=...)` 在對不上時直接拒跑。

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

| 維度 | 誰決定 | 取值 | 實測 |
|---|---|---|---|
| **模型權重精度** | 部署配置，**不是本研究的變數** | BF16 / AWQ-INT4 | BF16 佔 15 GB、AWQ 佔 4.7 GB → KV 預算 5.9 vs 15.0 GiB |
| **KV 儲存狀態** | **本研究的決策變數** | GPU-BF16 / **FP8** / **INT8** / **INT4** / CPU / SSD / Drop | 見下 |

**KV 精度四階全部是 `--kv-cache-dtype` 的取值**（計畫書原本只寫了兩階）：

| dtype 旗標 | KiB/token | 相對 BF16 | 備註 |
|---|---|---|---|
| `auto`（BF16） | 128.11 | 1.00× | 基準 |
| `fp8` | 64.04 | **2.00×** | 純格式轉換，無中繼資料 |
| `int8_per_token_head` | 66.04 | 1.94× | 多帶 2.04 KiB/token 的 scale |
| `int4_per_token_head` | 34.02 | **3.77×** | 多帶 2.02 KiB/token |

那 2 KiB 是 `2 × 32 層 × 8 KV head = 512` 個 FP32 scale，**可由架構直接算出**。
→ **式 (1) 若只計每元素位元組數，會系統性高估低精度階的收益。**

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

⚠️ `C_recompute` **不是常數**，隨 block 的絕對位置成長。
要量成 `C_recompute(position)`，不是單一純量。對 MoE 模型，用 **active 參數量**不是總參數量。

**🔴 但成長是線性不是二次**（計畫書原文說「attention 二次項」，那是錯的）。
整段 prefill 確為 $O(L^2)$，但固定大小的 block 在位置 $P$ 重算是 $O(C \cdot P)$——**對 $P$ 線性**。
平台 A 實測（Llama-3.1-8B、chunk=2048、每點三次取中位數）：

```
C_recompute(P) = 513.0 ms + 26.9 ms × (P/1000)
線性擬合最大偏差 1.6%，五個點全在線上，找不到二次成分
P=0 → P=24,576 成長 2.29×
```

### 🔴 SSD 與 Drop 的次序會反轉，交叉點可算出

實測每 block：`SSD = 5.536 ms`（與位置無關）、`DROP = 4.008 + 0.00021 × 位置`。
兩者在**位置 7,277 token** 處交叉：

| context | 平均位置 | 平均 DROP | SSD | 誰便宜 |
|---|---|---|---|---|
| 8,192 | 4,096 | 4.87 ms | 5.54 ms | **DROP** |
| 16,384 | 8,192 | 5.73 ms | 5.54 ms | SSD |
| 131,072 | 65,536 | **17.77 ms** | 5.54 ms | **SSD** |
| 524,288 | 262,144 | **59.06 ms** | 5.54 ms | **SSD** |

**128K 時只有 5.6% 的 block 適合丟掉重算。** 量測時若只在單一 ctx 取值，
會得到「某一階被支配」的錯誤結論——**必須掃 context 長度**。

### 🔴 量 SSD 階時，CPU 階必須遠小於工作集

第一次量失敗且錯得隱蔽：CPU 階開 24 GiB、工作集 8 GiB，
東西**從未 cascade 到磁碟**。vLLM 的計數器是唯一的證據：

```
chunk_queries:('0:primary',) = 2048   ← CPU 階
chunk_queries:('1:fs',)      =    2   ← 磁碟階   ← 只有 2 次！
```

那次量到的「SSD 550.9 ms」其實是 CPU 階，所以才會跟 CPU 只差 1.2%。
**每次量分層都要記 `chunk_queries`，接近 0 就代表沒量到那一層。**

### 🔴 SSD 階的成本不是 I/O 主導

同一設定換裝置：SATA 0.49 GB/s → 端到端 5,803 ms；NVMe 1.65 GB/s（**頻寬 3.4 倍**）→ **6,600 ms（更慢）**。
磁碟讀取只佔搬運成本約 35%，其餘是分層機制的查表、cascade 排程、promotion。
→ 成本模型要拆成 `C_ssd = C_lookup + C_sched + C_io(device)`，**只有末項與裝置有關**。

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
- **🆕 推理任務：GSM8K many-shot**（`code/m5_quality.py`）

### 🆕 品質量測分兩類，量法不同

| 類別 | 動作 | 預期 ε | 怎麼驗 |
|---|---|---|---|
| **無損** | CPU / SSD / DROP | **必須 = 0** | **逐字元比對輸出**（SHA-1），不同就是 bug |
| **有損** | FP8 / INT8 / INT4 | > 0 | GSM8K 正確率相對 BF16 的掉幅 |

**無損那類已驗證**：`cpu_lru` 與 `tier_fs` 對 `full_gpu` 的輸出 **60/60 逐字元相同**。
→ 位置分層三階在 ε 約束下等價，**ε 的實質內容全部落在精度分層上**。

### 🔴 樣本數：n=120 分辨不出 5 個百分點

實測 n=120、p≈0.79 時，兩組差值的 95% CI 是 **±10.3 pt**。三個精度階的差值
（−5.83 / +0.83 / −3.33 pt）**全部落在雜訊內**，且非單調（int8 高於 BF16）——
非單調性本身就是樣本量不足的徵候。

**要以 80% 檢定力分辨 5 pt 需每組約 1,055 題**；GSM8K test 有 1,319 題，可行。
**在達到該樣本量之前，不得宣稱任何精度階之間有品質差異。**

### 🆕 品質 ↔ 容量的取捨曲線（精度不是動作空間裡的一階）

精度與 CPU/SSD/DROP 的性質不同：後者換**時間**，精度換**容量 ↔ 品質**。
所以它不進 Oracle 的動作空間，而是**外層的旋鈕**：

```
給定量化比例 f
  → 容量放大 m(f)          （已量到：FP8 2.00×、INT8 1.94×、INT4 3.77×）
  → 品質損失 ε(f)          （GSM8K，本節要量）
  → 用 gpu_blocks × m(f) 跑 Oracle 得到 T(f)
  → 畫 (ε, T) 的 Pareto 前緣 = 式 (4) 的可行解集合
```

vLLM 用 `--kv-cache-dtype-skip-layers` 支援**逐層**混合精度，故 f 的粒度是 1/32
（Llama-3.1-8B 有 32 層）。**被量化的層要等間距分散**，集中在前段會量到
「淺層的敏感度」而非「f 的效果」。指令：

```bash
python code/m5_quality.py --gpu 0 --mode mixed \
  --n-shot 128 --n-test 400 --fractions 0 0.25 0.5 0.75 1.0
```

✅ **選模準則已排除外推問題**：兩個模型的評測長度都在原生範圍內（Qwen2.5-7B-1M 懸崖 315K ≪ 1M 上限；Llama-3.1-8B 的 131K 在原生範圍內）。**不要為了掃更長的 context 而開 YaRN**——那會讓品質下降的來源無法歸因，ε 就失去意義。若真的需要超過原生長度，必須額外建立「full-KV + 相同外推設定」的基準線。

### 驗收 M2
- [ ] 同一組設定跑三次，TTFT 變異 < 10%
- [ ] 品質分數可重現（同 seed 同結果）
- [ ] 產出 `results/m2_harness/repeatability.csv`

---

## 🆕 3.5 工作負載型態決定有效動作空間（做 M3 之前必讀）

`prefill` 與 `decode` 的卸載經濟性相差**五個數量級**，這決定了哪些階在哪種負載下可用。

實測每 token 成本（平台 A、AWQ 權重）：

| 上下文 | prefill | decode | 倍數 |
|---|---|---|---|
| 16,384 | 0.316 ms | 17.0 ms | 54× |
| 126,976 | 0.903 ms | 29.7 ms | 33× |
| 258,048 | 1.264 ms | 29.4 ms | 23× |

原因：prefill 一次算 N 個 token（算術強度高、受限於算力）；
decode 一次算 1 個，但**每一步都要讀整個 KV cache**（算術強度趨近 1、受限於記憶體頻寬）。

**後果：同一筆搬運在 prefill 攤提於整段前綴，在 decode 每步重付一次。**

| 卸載比例 | decode 每步增加 | 相對全駐留 |
|---|---|---|
| 5% | +24.6 ms | 3.0× |
| 20% | +98.4 ms | 11.9× |
| 100% | +492 ms | **59.3×** |

**→ full attention 下，decode 階段的卸載在任何比例都不可行。**
唯一出路是稀疏 attention（只取回當步分數高的 block），即 InfiniGen/Quest/ArkVale 的做法。

### 三種型態的有效動作空間

| 型態 | 決策點 | 預測什麼 | 有效動作空間 |
|---|---|---|---|
| **prefill 為主** | 請求**之間**保留哪些前綴 | 該前綴是否再被請求（**時序**） | 六階全部 |
| **多輪對話** | 同上 + 思考空窗的機會成本 | 同上 | 六階全部 |
| **生成為主** | 請求**之內**哪些 block 可離開 GPU | 該 block 的 attention 分數（**語意**） | **只剩 GPU 內的精度階** |

**本計畫涵蓋前兩種。** 第三種的預測對象不同，是另一個問題。

### ⚠️ 生成長度不可設太短

原本 `GEN_TOKENS = 32`，實測導致 **99.8% 的時間都是 prefill**（ctx=258K），
TPOT 用 11–23 個 token 算、雜訊極大，且完全沒測到 decode 期間的行為。
文獻設定：CoKV 掃 1/512/1024/2048/4096、KVSwap 連續生成 1000、多數固定 256。
**現已改為 256**（`PAPER_HKV_GEN_TOKENS` 可覆寫）。

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

### 🆕 求解方法與其效力範圍（`code/m4_oracle.py`）

**方法**：trace 驅動的模擬。GPU 階的逐出用 **Bélády/MIN**（單階可證明最優），
跨階的目的地用**成本感知貪婪**。後者非加權多階問題的最優解，故所得 headroom 是
**下界**——真正的最優只會更好。**此不對稱使 GO 判定安全，NO-GO 判定須額外審視。**

多階怎麼決定目的地（這是最容易誤解的部分）：

```
GPU 滿了：
  ① Bélády 選出「下次使用最遠」的 block      ← 誰該走
  ② 再看他「多久後回來」                      ← 該去哪
       永遠不回來     → 丟掉      成本 0
       CPU 有位子     → 放 CPU
       CPU 滿、SSD 有 → 放 SSD
       兩層都滿       → 與 CPU 中「更晚才用」者交換（近的留快層）
```

**「知道未來」不只告訴你誰該走，還告訴你他多久後回來，後者決定他該放哪一層。**

### 🔴 三個必須先滿足的前提

1. **成本常數必須來自 M2 實測。** 程式在讀不到量測檔時 `SystemExit`，不使用預設值。
   這道防線實際擋下過一次——SSD 常數錯了 13.7 倍時 Oracle 直接拒跑。
2. **模擬器須先能複現已量測的行為**（`--validate`）。目前方向 2/2 正確，
   但**量級高估約 1.5×**（實測 9.00 對模擬 14.33）。
   → **只引用 headroom 的趨勢，不引用絕對值。**
3. **工作負載須有代表性。** 見下。

### 🔴 已知會使 Oracle 失真的四件事（按影響排序）

| # | 問題 | 方向 | 狀態 |
|---|---|---|---|
| 1 | **精度階完全沒模擬**——只有 {GPU, CPU, SSD, DROP} 四階，論文是六階 | Oracle **被低估** | 待修 |
| 2 | **前綴語意**：miss 記成重算一個 block，但 vLLM 是「中間缺一塊、其後全部重算」 | Oracle **被低估** | 待修 |
| 3 | **模型混用**：成本常數是 Llama-BF16 的，容量預算是 Qwen-AWQ 的，CPU 階大小寫死 128 KiB/token | 不明 | 待修 |
| 3b | **沒有預取**：取回成本在存取當下才收，等於假設「永遠來不及預取」。真正知道未來的 policy 會在請求到達前把 block 搬回 GPU，把該成本完全藏掉（vLLM 本身有 promotion 機制） | Oracle **被低估** | 待修 |
| 4 | ~~Oracle 用位置 0 的價格重算，baseline 付全價~~ | 曾灌水 8–9 pt | **已修** |

第 1、2、3b 項**全部往「低估」的方向偏**，故目前量到的 headroom 是下界。

⚠️ 預取不是免費的：提前搬回會提早佔用 GPU 記憶體、增加壓力。
**這個取捨（何時預取、預取誰）正是論文要解的問題**，也是 Oracle 贏過線上策略的
第二個來源——不只「留誰」，還有「何時搬」。

第 4 項是用「**比對命中次數而非時間**」抓到的：壓力 0.5× 時五個策略的命中數
完全相同、時間卻差 9.7%——命中相同時間不可能差。
**教訓：命中次數與時間要並列報告。前者是機制且硬體無關，後者依賴成本模型。**

### 🔴 headroom 有兩個自變數，不是一個

計畫書原本假設只掃「壓力」。實測顯示**重用率**同等重要：

| 工作負載 | 重用率 | 壓力 | headroom | 判定 |
|---|---|---|---|---|
| 合成 Zipf α=0.9 | **85.1%** | 1.0× | 0.0% | 🔴 |
| 合成 Zipf α=0.9 | 85.1% | 3.3× | **20.8%** | ✅ |
| conversation @ BF16 預算 | 36.6% | 60.8× | 16.4% | ✅ |
| toolagent @ BF16 預算 | 55.3% | 60.9× | 14.1% | 🟡 |
| conversation @ AWQ 預算 | 36.6% | 10.7× | 7.3% | 🟡 |
| **toolagent @ AWQ 預算** | 55.3% | 10.7× | **4.6%** | 🔴 |

真實流量有 **45–63% 的 block 是第一次看到**（compulsory miss），誰都躲不掉。
我的合成負載重用率 85.1%，**比真實高太多**。

**🔴 最關鍵的一列**：同一份流量只換權重精度，AWQ（**論文自己的主力設定**）
把 headroom 從 14.1% 壓到 4.6%。**配置越好，論文貢獻的空間越小。**

### ⚠️ 公開 trace 都不是長上下文

| trace | 中位數 | P99 | ≥128K 的筆數 |
|---|---|---|---|
| Mooncake conversation | 6,906 | 85,399 | **0** |
| Mooncake toolagent | 6,346 | 61,525 | **0** |

**論文的目標情境（128K–512K）在公開資料中沒有對應的 trace。**
故上表的判定只適用於短上下文流量，**不能外推到論文的目標區間**。

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

**下表的前六列是 2026-08-30/31 實際踩到的，不是預想。**

| 風險 | 徵兆 | 處置 |
|---|---|---|
| **CUDA 版本不合** | `libcudart.so.13: cannot open` | PyPI 的 vLLM 是 CUDA-13 wheel，driver 550 只到 12.4。用 GitHub release 的 **`+cu129`** wheel 加 `--index-strategy unsafe-best-match`。**不是 Python 版本問題** |
| **CUDA 看似可用但 kernel 會爆** | `torch.cuda.is_available()` 回 True，跑 matmul 卻 `driver too old` | **驗收條件必須是「跑成一個真的 kernel」**，見 `code/env_fingerprint.py` |
| **flashinfer JIT 缺 ninja** | `FileNotFoundError: 'ninja'` | ninja 在 venv 裡但直接呼叫 `$VENV/bin/vllm` 時子行程看不到。`export PATH="$VENV/bin:$PATH"` |
| **`/dev/shm` 被自己的洩漏檔塞爆** | 帶卸載的 baseline 全部啟動失敗，錯誤訊息不指向真因 | CPU 階是 `/dev/shm` 的 mmap，server 被 kill 時不回收。跑 `python code/shm_gc.py --apply` |
| **ctx 頂端等於模型定址上限** | pydantic `ValidationError: max_model_len > derived` | `max_len = ctx + GEN_TOKENS + 餘裕` 會超過。**ctx 頂端留 4,096 餘裕** |
| **記憶體「看似」已釋放** | `Free memory on device (8.51/23.68 GiB)` | 行程結束到 driver 收回有延遲。`wait_until_free()` 要求**連續三次取樣**達標 |
| ~~Ampere 不支援某些 KV dtype~~ | ~~`--kv-cache-dtype fp8` 報錯~~ | **實測可用**，見開頭的推翻表第 1 項 |
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
