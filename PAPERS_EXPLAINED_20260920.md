# 那 12 篇到底在幹嘛 —— 對抗式查新的完整解說

**日期**：2026-09-20
**這份文件是什麼**：`PAPERS_EXPLAINED.md` §9 那張「12 篇高風險論文」清單的**查證結果**。
上一份是「老師指定的五篇」（Cake / AdaptCache / Strata / Bidaw / MTDS），
**這一份是你 `refs.bib` 裡自己引了、但沒細讀的那 12 篇 + 3 個工程界的新證據。**

**攻擊目標**：2026-09-19 那兩個發現（`EXPERIMENTS_20260919.md` 實驗 #1 和 #2）。
**查證方式**：對抗式（目標是**找到殺死你的先行工作**，不是確認你 novel）。
**證據等級不一致，這很重要 —— 見 §3，不要把不同等級的結論混著用。**

---

## 0. 先給結論

| 發現 | 判決 | 一句話 |
|---|---|---|
| **(A) 量化與放置是替代品** | 🟢 **活下來，但要重寫成更窄的宣稱** | 沒有任何一篇量過「Belady headroom 隨 bit-width 變化」。而且**整個領域都假設它們正交** —— 那個假設就是你的靶子 |
| **(B) 有效傳輸頻寬是實作性質** | 🟡 **一半被搶走了** | **MLSys 2026 已經發表了 achieved-vs-peak（15 GB/s = 23% of 64 GB/s peak，vLLM+LMCache）**，而且已經點名 "transfer granularity" 是原因之一。你剩下的是**機制、量化的固定成本、跨模型 17.1×、ROCm 根因、以及修法** |

### 威脅分級總表

| # | 論文／來源 | 軸 | 含 (A)？ | 含 (B)？ | 威脅 | 證據 |
|---|---|---|---|---|---|---|
| 1 | **Understanding Bottlenecks…KV Offloading**（MLSys'26） | 量測 | ❌ | 🔴 **一半** | 🔴🔴🔴 | 全文 fetch |
| 2 | **SGLang PR #40278**（2026-09-19） | 工程 | ❌ | 🟡 同構 | 🔴🔴 | 3 票確認 |
| 3 | **SGLang PR #37701 / #37635**（ROCm） | 工程 | ❌ | 🔴 **搶走論點，機制不同** | 🔴🔴 | ✅ **已查證（見 A-3）** |
| 4 | **EvicPress**（arXiv 2512.14946） | 壓縮×逐出×階 | 🟡 **反命題** | ❌ | 🔴🔴 | abstract |
| 5 | **AdaptCache**（BigMem'25 workshop） | bit-width×階 | 🟡 **反命題** | ❌ | 🔴 | 3 票確認（全文） |
| 6 | **Where Should the KV Cache Live?**（arXiv 2609.16215，**新，不在你 bib**） | 放置 | 🟡 **搶了你的前提** | ❌ | 🔴🔴 | 3 票確認（全文） |
| 7 | **KVServe**（SIGCOMM'26） | 壓縮設定 | 🟡 修辭重疊 | ❌ | 🟡 | 3 票確認（全文） |
| 8 | **KVTuner**（ICML'25） | bit-width×品質 | ❌ **反命題** | ❌ | 🟢 | 3 票確認（全文） |
| 9 | **CacheGen**（SIGCOMM'24） | bit-width×傳輸 | ❌ **但殺掉一個框架** | ❌ | 🟡 | 3 票確認（全文） |
| 10 | **LMCache PR #5067** | 工程 | ❌ | 🟡 粒度門檻 | 🟡 | 部分確認 |
| 11 | **Tutti**（arXiv 2605.03375） | SSD I/O | ❌ | 🟡 **小 I/O 問題** | 🟡 | abstract |
| 12 | **KVDrive**（arXiv 2605.18071） | 三階放置 | ❌ | ❌ | 🟢 | abstract |
| 13 | **OrbitFlow**（PVLDB'26） | 逐層放置 ILP | ❌ | ❌ | 🟢 | abstract |
| 14 | **LeoAM**（arXiv 2506.20187） | 三階＋chunk 粒度 | ❌ | 🟡 chunk 粒度 | 🟢 | abstract |
| 15 | **HCache**（EuroSys'25） | 存 activation | ❌ | ❌ | 🟢 | abstract |
| 16 | **KVPR**（ACL Findings'25） | 算 vs 載 | ❌ | ❌ | 🟢 | abstract |
| 17 | **HetMem**（IEEE CAL'25） | HBM+DRAM 放置 | ❌ | ❌ | 🟢 | abstract |
| 18 | **YaKV**（Yandex 2604.08426） | offload **品質** | ❌ | ❌ | 🟢 **是盟友** | abstract |

---

## 1. 為什麼是這幾篇？（回答「為何要選這幾篇」）

**這 12 篇不是新選的 —— 它們全部已經在你的 `refs.bib` 裡。**
選擇規則只有一條：

> **把 (A) 和 (B) 拆成「一篇論文要同時具備什麼零件才能搶走它」，
> 然後把 `refs.bib` 裡「擁有任何一個零件」的全部抓出來。**

### (A)「量化與放置是替代品」需要的四個零件

| 零件 | 白話 | 誰有 |
|---|---|---|
| ① 多階放置策略 | HBM/DRAM/SSD 要放哪 | KVDrive、Tutti、OrbitFlow、LeoAM、HetMem、AdaptCache、EvicPress、2609.16215 |
| ② bit-width 當作決策變數 | 這塊要 4-bit 還 16-bit | KVTuner、KVServe、CacheGen、AdaptCache、EvicPress |
| ③ **Belady / offline-optimal 上界** | 「最聰明的策略最多能拿多少」 | **一篇都沒有** |
| ④ **把 ③ 畫成 ② 的函數** | headroom 隨 bit-width 怎麼變 | **一篇都沒有** |

→ **只有同時有 ①②③④ 的論文能殺死 (A)。目前 ①②有人做、③④是空的。**
這就是 (A) 存活的技術理由，也是為什麼你**必須**把 oracle 留在論文裡 —— 它是你唯一的護城河。

### (B)「有效傳輸頻寬是實作性質」需要的四個零件

| 零件 | 白話 | 誰有 |
|---|---|---|
| ① achieved vs peak | 實際拿到峰值的幾 % | 🔴 **bottlenecks2026 有了**、SGLang #40278 有了 |
| ② 粒度→頻寬曲線 | 每筆多大 → 拿到多少 | SGLang #40278（TMA op 層級）、LMCache #5067（只有門檻值） |
| ③ **每筆固定成本的數值** | 13–16 µs，與 payload 無關 | **沒有人量到**（一個被推翻的宣稱說 LMCache 有，3 票中 2 票反對） |
| ④ **同硬體跨模型變異** | 同一張卡差 17 倍 | **沒有人量到** |

→ **①被搶走了，②被搶走一半，③④還在。**
所以 (B) 現在**不能**寫成「我們首次發現 KV offload 沒吃滿頻寬」，
必須寫成「我們首次**解釋並量化**為什麼沒吃滿，並證明它跨模型差 17 倍」。

### 額外三個來源（不在 refs.bib，但這輪撈到）

- **arXiv 2609.16215**（6 天前）：新論文，直接搶你 (A) 的**前提**。必加。
- **SGLang PR #40278 / #37701**：工程界正在做 (B)。必加。
- **LMCache PR #5067**：粒度門檻的實作證據。

---

## 2. 被攻擊的兩個發現（精確表述）

這一節是為了讓你知道**到底什麼東西在受審**。數字全部來自 `EXPERIMENTS_20260919.md`。

### (A) 量化與放置是替代品，不是互補品

| trace | 臂 | 最佳 baseline (ms) | oracle (ms) | headroom |
|---|---|---|---|---|
| toolagent | 全 BF16 | 1,727,885 | 1,370,923 | **20.66%** |
| toolagent | **全 INT4 + 原廠 LRU** | **1,400,252** | 1,371,969 | **2.02%** |
| conversation | 全 BF16 | 3,556,147 | 2,812,606 | **20.91%** |
| conversation | **全 INT4 + 原廠 LRU** | **2,783,633** | 2,685,686 | **3.52%** |

三件事同時成立：
1. 只改 `--kv-cache-dtype`，拿到 oracle headroom 的 **91.8% / 103.9%**，不需要任何聰明策略。
2. 切完 INT4，剩給 learned policy 的只有 **2.02% / 3.52%**（低於你自己 `CLAUDE.md` 規則 4 的 5% NO-GO 門檻）。
3. 兩個 oracle 幾乎一樣（−0.08%）→ **oracle 靠「少重算」到達的地方，INT4 靠「多裝一點」也能到**。

穩健性：對 dequant 不確定性穩健（+19.38% / +18.96% / +18.62%）；headroom = 20.6%（95% CI 20.2–23.8, n=60 bootstrap）。

### (B) 有效傳輸頻寬是實作性質，不是硬體性質

| 每筆描述符 | 描述符數 | 實測 | 佔實測 bulk copy 天花板 **57.4 GB/s** |
|---|---|---|---|
| 8 KiB | 32,768 | 2.8 GB/s | 5.1% |
| **32 KiB** | 12,288 | **2.4 GB/s** | 4.4% |
| **64 KiB** | 6,144 | **4.7 GB/s** | 8.5% |
| 256 KiB | 1,024 | 14.7 GB/s | 27.0% |
| **1.5 MiB** | 256 | **38.9 GB/s** | 67.8% |
| 4 MiB | 64 | 48.5 GB/s | 88.9% |

- **每筆固定成本 12.9 µs（32 KiB）／16.3 µs（1.5 MiB），與 payload 大小無關。**
- 三重確認：qwen7b-1m 2.4 vs 實測 **2.27**；llama8b 4.7 vs **4.47**；qwen3-30b-a3b 38.9 vs **38.28**。
- 根因：vLLM 的 Triton 快速路徑在 ROCm 上被明確停用（`gpu_worker.py:46-51`），走 `hipMemcpyBatchAsync`，而 ROCm 7.2.1 的 bug 逼你用 `numAttrs=0`，拿不到 `srcAccessOrder=ANY`。

> ⚠️ **寫作精確度**：57.4 GB/s 是**你實測的 bulk copy 天花板**，不是 PCIe Gen5 x16 的理論峰值（單向 64 GB/s）。
> 論文裡兩個都要報，不要混用，否則審稿人會抓。換算成理論峰值：4 MiB = 75.8%、32 KiB = 3.75%。

---

## 3. ⚠️ 證據等級（先看這個，再看 §4）

**這輪的證據強度非常不平均，混用會出事。**

| 等級 | 意思 | 適用於 | 可以怎麼用 |
|---|---|---|---|
| 🟩 **A：3 票對抗式確認＋一手全文** | 有人下載 PDF/HTML、逐字 grep、算過關鍵字次數 | 2609.16215、AdaptCache、KVServe、KVTuner、CacheGen、SGLang #40278 | ✅ 可以直接寫進 related work，可以寫「X 沒有做 Y」 |
| 🟨 **B：我這輪抓的一手 abstract／全文片段** | 抓了 arXiv 頁面，但只到 abstract 或局部 | bottlenecks2026（全文片段）、EvicPress、Tutti、KVDrive、OrbitFlow、LeoAM、HCache、KVPR、HetMem、YaKV、LMCache #5067 | 🟡 可以定位、可以排威脅級，**但「它沒有做 Y」這種否定句必須先讀全文再寫** |
| 🟥 **C：未確認／被推翻** | 3 票中沒過，或根本沒查 | SGLang #37701/#37635 的**數字**、LMCache #5067 的每筆成本模型 | 🔴 **不可引用**。必須自己去讀原始 PR |

**被 3 票推翻、不可引用的宣稱（這輪共 11 條）列在 §8.3。**

---

## 4. 逐篇解說

---

### 組 A：直接威脅 (B) 的四個來源

---

#### A-1. 🔴🔴🔴 Understanding Bottlenecks for Efficiently Serving LLM Inference With KV Offloading
> arXiv:2601.19910 ｜ **MLSys 2026** ｜ William Meng (UPenn/Intel), Benjamin Lee (UPenn), Hong Wang (Intel) ｜ 2025-12-16
> 你的 bib key：`bottlenecks2026`

**這是這輪最危險的一篇。**

##### 它在解什麼問題
KV offload 到 CPU DRAM 之後，**PCIe 變成瓶頸**。它做兩件事：
1. **建一個解析框架**，推導出 `κ_crit`：cached-to-prefill token ratio 的臨界值，超過就變 memory-bound。
2. **實測刻畫**：99% 的延遲花在傳輸上，GPU 只吃到 28% 的額定 TDP。

##### 機制：κ_crit
```
κ_ratio = K / T                    （cached tokens / prefill tokens）
κ_crit  = (F_pf / B_kv) × (BW_PCIe / C_eff) = κ_M × κ_HW
                                    （模型因子 × 硬體因子）
Prefill 是 memory-bound ⟺ κ_ratio > κ_crit
```
它的結論是：**典型 workload 超過這個門檻好幾個數量級。**

##### 🔴 它已經有的東西（這是威脅）
§6.2 逐字：
> *"We measure sustained PCIe bandwidth of **15 GB/s (23% of unidirectional 64 GB/s peak)**"*

原因它也寫了：
> *"system bottlenecks including CPU-GPU memory copy overheads, NUMA effects, and **transfer granularity**"*

**也就是說：**
- ✅ achieved-vs-peak 的 KV offload 量測 —— **已經發表了**
- ✅ 量測的是 **vLLM v0.10.1 + LMCache v0.3.5**（不是自己寫的 harness）
- ✅ 甚至 **已經點名 "transfer granularity"** 是三個原因之一

→ **你不能再寫「我們首次發現 KV offload 的實際頻寬遠低於峰值」。這句話會被一槍打死。**

##### 🟢 它沒有的東西（這是你的生路）
| 你有 | 它有嗎 |
|---|---|
| 粒度 → 頻寬的完整曲線（8 KiB → 4 MiB） | ❌ 只有一個數字 15 GB/s |
| 每筆固定成本 12.9–16.3 µs | ❌ ABSENT |
| 描述符數量分析、DMA descriptor | ❌ ABSENT |
| **同硬體跨模型 17.1× 變異** | ❌ **單一數字，無跨模型／跨框架分解** |
| AMD / ROCm / MI300X | ❌ **全 NVIDIA**（H100/B200/A100；AMD EPYC 只是 host CPU） |
| 修法（描述符合併） | ❌ **完全沒提修法** |

而且它提的最佳化方向是：
1. **硬體**：NVLink C2C（900 GB/s）、unified HBM
2. 模型架構：MLA、KV 量化
3. 排程：workload-aware disaggregation
4. Power capping

→ **它把 23% 當成硬體給定的事實，然後去換更好的硬體。
這正是 (B) 要打的那個假設。它是你最好的靶子，同時也是你最大的重疊。**

##### 💡 一個可能很有價值的觀察（**假說，未驗證**）
它的 15 GB/s ÷ 你的 bulk copy 天花板 57.4 GB/s = **26.1%**。
你的表裡 256 KiB 那一列是 **14.7 GB/s = 27.0%**。**幾乎完全重合。**

如果這不是巧合，那你可以把它從「威脅」翻成「佐證」：
> 「MLSys'26 報的那個 sustained 23%，是我們這條曲線上的**一個點**。
> 我們解釋了它為什麼在那裡，並證明它會因描述符粒度在 4%–89% 之間移動。」

🔴 **但這是假說**：他們是 H100/CUDA，你是 MI300X/ROCm，機制不必然相同。
**要用這句話，必須先在 NVIDIA 卡上重現一次粒度曲線。** 現在只能當內部思路，不能寫進論文。

##### ⚠️ 符號撞名
**它用 `κ_ratio` / `κ_crit`，你的論文用 `κ` 當成本常數。**
審稿人會混淆。**建議你換符號**（例如 `c_tier` 或 `γ`），並在 related work 明確區分。

##### 對你的意義
- **必引，而且要引在 (B) 的第一段。** 用它的 99% / 28% TDP 當你的 motivation（這是別人幫你做好的背書）。
- **(B) 的 claim 必須改寫**成粒度／機制／跨模型層級，不能停在 achieved-vs-peak。
- 它是你「bandwidth is a hardware property」這個靶子的**最佳引用**：它明文提議換硬體互連來解。

---

#### A-2. 🔴🔴 SGLang PR #40278 — TMA-staged host↔device KV transfer kernel
> `sgl-project/sglang` PR #40278 ｜ 作者 cctry（Meta 工程師）｜ **2026-09-19 開，state=open，未 merge**
> 🟩 證據等級 A（3 票確認，走 GitHub API 讀 raw body）

##### 它在做什麼
SGLang HiCache 的 host↔device KV 搬運 kernel 太慢，作者改用 TMA（`cp.async.bulk`）經 shared memory staging 重寫。

##### 它已經公開的東西（威脅）
1. **achieved-vs-peak，逐字**：
   > *"The copy-engine ceilings of the link measured with `cudaMemcpyAsync` are ~210 GB/s H2D and ~192 GB/s D2H, so 'after' is at the link for large transfers."*

   出貨中的預設路徑只拿到 **97/210 = 46%**，修完 **192/210 = 91%**。
   （已驗證 `hicache_io_backend` 預設就是 `kernel`，所以 97 GB/s 是**出貨路徑**不是 opt-in。）

2. **一個 closed-form 頻寬模型**：
   > *"Each of its 1024 threads holds one 64 B slice … so a CTA has at most 64 KB outstanding … bandwidth = bytes in flight / round trip … ~48 GB/s per CTA corresponds to an effective host round trip of ~1.3 µs."*

3. **一張 op 粒度表**：TMA bulk loads 32 KB ops → 150–200 GB/s；512 B ops → **21 GB/s**。
   歸因於 *"the TMA unit handles roughly one bulk op per ~50 cycles"*。

##### 🟢 為什麼它沒殺死 (B)
| 面向 | #40278 | 你 |
|---|---|---|
| 瓶頸在哪 | **SM 上的 register staging kernel**，被 bytes-in-flight 限制 | **DMA copy engine 本身**，被每筆固定成本限制 |
| copy engine 的角色 | 🔴 **是它要追的天花板** | 🔴 **是它本身在拖** |
| 固定成本 | ~50 GPU cycles（~30 ns），on-SM TMA op | **12.9–16.3 µs**，host DMA descriptor（差 ~500×） |
| 鏈路 | NVLink-C2C（GB300） | **PCIe Gen5 x16** |
| 平台 | 明文寫 *"ROCm and pre-Hopper are unaffected"* | **MI300X / ROCm** |
| 跨模型變異 | ❌ | ✅ 17.1× |

→ **機制是反的**：它的 `cudaMemcpyAsync` 是快的參考點；你的 `hipMemcpyBatchAsync` 是慢的那一個。

##### ⚠️ 但它侵蝕了你的「大框架」
「effective KV transfer bandwidth is an implementation property」這句話，
**現在有一個 top-2 serving engine 在 GitHub 上公開示範了。**
→ (B) 的標題句不能停在這個抽象層級，必須下沉到 descriptor granularity + cross-model。

##### ⏱️ 時效性
**2026-09-19 開的，比你的實驗晚一天。** 任何頂會的 concurrency rule（通常 ≤3 個月）都算**同期工作**，不是先行工作。
→ 你可以在論文裡寫 "concurrent work"，不需要把它當 prior art 閃避。**但一定要提，不提就是不誠實。**

---

#### A-3. 🔴🔴 SGLang PR #37701 / #37635 — ROCm 的 `block_quota` 調參
> `[AMD] perf(kernel): tune JIT HiCache transfer block_quota for ROCm`（#37701，2026-09-03，open）
> `[kernel] Use block_quota=8 for the JIT HiCache transfer kernel on ROCm`（#37635，2026-09-02，closed）
> ✅ **P0 已於 2026-09-20 解決 —— 我直接讀了兩個 PR 的原始頁面。以下是查證結果。**

##### ✅ 查證結果（取代下方原本的「未確認」警告）

**PR body 逐字**（#37701 Motivation）：

> *"`python/sglang/kernels/ops/kvcache/hicache.py` uses `DEFAULT_BLOCK_QUOTA = 2` on every platform.
> **On CDNA that caps host→device at 24 GB/s where the same kernel reaches 55 GB/s at `block_quota = 8`.**"*

**實測數字（MI355X，不是 MI300X）**：

| element_size | `bq=2` | `bq=8` | 倍數 |
|---|---|---|---|
| 128 | 16.56 GB/s | 43.85 GB/s | 2.6× |
| 256 | 24.43 | 54.34 | 2.2× |
| 512 | 24.93 | 54.73 | 2.2× |
| 1024 | 24.80 | 55.46 | 2.2× |
| 1152 | 23.53 | 54.54 | 2.3× |

**機制逐字**：
> *"`bq=2` holds 4x fewer workgroups but **stays resident 2.2x longer** because its bandwidth is lower
> (one 4.1 GB load, 168 ms vs 76 ms)."*

→ **機制是 GPU workgroup residency／occupancy，不是 DMA descriptor granularity。**

**其他事實**：#37635 已 closed（09-02）、#37701 open（09-03）且 **CI 三個全部失敗**；
端到端只有 +10% throughput（TPM 5,574,054 vs 5,537,165，MiniMax-M3 TP4，concurrency 15）；
參照 AOT kernel 版本 #30024。

##### 🔴 這對 (B) 的實際傷害

| (B) 的組成 | 判決 |
|---|---|
| 「AMD 上 KV 傳輸頻寬是**軟體性質**不是硬體性質」這個**論點** | 🔴 **已被搶走。** 2026-09-02 就公開示範了（24 → 55 GB/s，2.2–2.6×，同硬體同 bytes） |
| **機制**：每筆描述符固定成本 12.9–16.3 µs | ✅ **存活。** 他們是 workgroup residency，**不同的程式路徑、不同的成因** |
| **系統**：vLLM `swap_blocks_batch` / `hipMemcpyBatchAsync`（DMA copy engine） | ✅ **存活。** 他們動的是 SGLang HiCache 自己的 **JIT 計算 kernel**，不是 DMA 路徑 |
| **幅度**：17.1× | ✅ **存活。** 他們是 2.2–2.6× |
| **跨模型變異**（KV 佈局決定粒度） | ✅ **存活。** 他們只有一個常數、一個模型，零跨模型資料 |
| **描述符大小 → 頻寬曲線** | ✅ **存活。** 他們沒有曲線 |

**→ 結論：(B) 的 headline 必須從「頻寬是軟體性質」改寫成更窄的東西。**
可存活的表述例如：

> 「在 vLLM 的 DMA copy-engine 卸載路徑上，有效頻寬由**每筆描述符 12.9–16.3 µs 的固定成本**決定，
> 因此**同一張卡上的不同模型**會因 KV 張量佈局而相差 **17.1 倍**（2.27–38.28 GB/s）。」

粗體的三個限定詞（DMA 路徑、固定成本量化、跨模型）是目前沒有人講過的部分。
若機制真的是 **kernel occupancy** 而不是 **DMA descriptor granularity**，你的機制宣稱還活著；
但 **「這是軟體性質不是硬體性質」這個結論，已經被別人在 AMD 上公開講過了。**

---

#### A-4. 🟡 LMCache PR #5067 — `cudaMemcpyBatchAsync` direct copy path
> `LMCache/LMCache` PR #5067 ｜ 作者 YuhanLiu11 ｜ 2026-09-11 ｜ **Draft (open)**
> 🟨 證據等級 B（我這輪直接抓頁面）

##### 確認存在的
- 機制：*"moves each chunk straight between the pinned host object and the engine's paged KV buffers with **one `cudaMemcpyBatchAsync` call per chunk**"*
- 🔴 **一個粒度門檻**：`--direct-copy-min-block-bytes`，**預設 128 KB**
- 動機：*"scatter kernels each waiting ~1.6 ms to be scheduled … about 50 ms of stream idle per 128k retrieve"*
- 效果：Kimi-Linear-48B（TP2, H200）token throughput 81,273 → 85,921 tok/s（+5.7%），TTFT 2736 → 2558 ms

##### 我**沒有**在頁面上找到（也被 3 票推翻過）
- ❌ 每筆 entry 的固定成本微秒數（曾有宣稱說是 0.6 µs，**2 票反對**）
- ❌ `time = max(entries × 0.6µs, bytes / 55 GB/s)` 這個模型（**推翻**）
- ❌ 34 KB break-even（**推翻**）
- ❌ 不同 entry size 的吞吐量表（**3 票全反對**）

##### 對你的意義
- ✅ **那個 128 KB 門檻本身就是證據**：LMCache 的維護者**知道**粒度太小時 batch copy 會輸，所以加了 gate。
  → 「粒度決定 KV 傳輸效率」這件事**在工程界是已知的**，你不能寫成「沒人知道」。
- ✅ 但**沒有人把它量化成一條曲線、一個固定成本、和一個跨模型的變異**。這是你的。
- ⚠️ 128 KB 這個門檻**跟你的曲線一致**（你的 128 KiB = 8.6 GB/s = 15.7%，確實還在很差的區間）。
  可以當成獨立佐證，但要小心：他們是 NVIDIA + `cudaMemcpyBatchAsync`，你是 AMD + `hipMemcpyBatchAsync`。

---

### 組 B：直接威脅 (A) 的四個來源

---

#### B-1. 🔴🔴 EvicPress: Joint KV-Cache Compression and Eviction for Efficient LLM Serving
> arXiv:2512.14946 ｜ 2025-12-16 ｜ Shaoting Feng, …, Junchen Jiang（**跟 AdaptCache 同一組人**）
> 你的 bib key：`evicpress2025` ｜ 🟨 證據等級 B（abstract）

##### 它在解什麼問題
KV cache 超過 GPU 記憶體時，前人要嘛**逐出到低階儲存**、要嘛**壓縮**。
它說前人漏掉的機會是：**把逐出和壓縮的決策放在一起最佳化**，目標是最小化平均生成延遲、且不傷品質。

##### 機制：統一效用函數
```
對每一個 context 的 KV cache：
  評估「壓縮」或「逐出」對 (品質, 延遲) 的影響 —— 而且是放在所有 context 一起看
  → 一個 unified utility function 給分
  → profiling module 週期性更新所有 (eviction, compression) 組合的分數
  → 用 fast heuristic 重排所有儲存階上的 KV cache，讓每一階的 utility 總分最大
```
對壓縮敏感的 context 就保守壓縮。

##### 控制的決策
✅ 壓縮率（lossy）｜✅ 逐出｜✅ 多階放置｜❌ bit-width 的明確掃描（abstract 沒寫）｜❌ 重算 vs 載入

##### 評測與數字
12 個 dataset × 5 個 model ｜ **TTFT 最高快 2.19×（等品質下）**

##### 🔴 為什麼它是 (A) 最強的反命題
它的**整個立論前提**就是：
> *"prior work misses an important opportunity: **jointly optimizing** the eviction and compression decisions"*

**這正是 (A) 要推翻的那句話。**
如果 (A) 成立（INT4 + 原廠 LRU 就拿到 92–104% 的 headroom），那 EvicPress 的 joint optimization 在你的 trace 上**是不必要的**。

##### 🟢 為什麼它沒殺死 (A)
- ❌ 沒有 Belady / oracle / offline-optimal baseline（abstract 層級）
- ❌ 沒有 bit-width 掃描
- ❌ 沒有 Mooncake trace
- ❌ 沒有「headroom 隨 bit-width 的函數」

→ 它證明的是「**joint 比 single 好**」（2.19×），
你證明的是「**single（bit-width）就幾乎等於 optimal，所以 joint 的邊際價值是 2–3.5%**」。
**這兩個不矛盾，但它們是對同一件事的相反敘述。**

##### 🔴 審稿人一定會問的那句話
> 「你的 substitutes 結論，是不是只是 EvicPress 的 joint optimization 在你的特定 trace 上退化成了角落解？」

**你必須先準備好答案。** 目前最好的回答是：
「EvicPress 沒有 optimal 的上界，所以它無法知道自己離 optimal 多遠。
我們有 —— 而且我們證明切到 INT4 之後，離 optimal 只剩 2.02%。」

##### 🔴 必辦事項
**EvicPress 是這 12 篇裡唯一「必須讀全文」的一篇。**
要確認的三件事：
1. 它的 compression 是不是就是 bit-width？（若是，它的 utility function 就跟 (A) 同軸）
2. 它有沒有 optimal / upper bound baseline？
3. 它的 baseline 裡有沒有「只壓縮 + LRU」這一臂？**如果有，而且那一臂很接近它自己，(A) 就被搶走了。**

---

#### B-2. 🔴 AdaptCache: KV Cache Native Storage Hierarchy
> arXiv:2509.00105 ｜ **SOSP 2025 BigMem workshop，只有 5 頁** ｜ v1 2025-08-28、**v2 2026-01-15**
> 🟩 證據等級 A（3 票確認，v1+v2+camera-ready 三個版本全文 grep）

##### 它在幹嘛
同時自適應調整 **壓縮演算法 + 壓縮率 + 裝置放置**，用一個 marginal-utility 指標統一。

##### 機制
```
形式化成 NP-hard 的 Multi-Choice Knapsack Problem (MCKP)
→ 用 Kellerer 教科書的 greedy 解，達到 Linear Programming Optimality
→ Estimator：用「歷史命中頻率」估計未來命中頻率
```

##### 🟩 確認的事實（全文 grep）
- baseline 只有四個：`Without Compression`、`KIVI LRU`、`StreamingLLM LRU`、`Prefill`
- **`oracle` = 0 次、`Belady` = 0 次**（v1、v2、camera-ready PDF 全部）
- 作者自承逐字：
  > *"Acknowledging that this design is greedy and not necessarily optimal … optimal solution is not trackable since it is NP-hard."*
- Estimator 用歷史命中頻率 → **這是 Belady 的定義上的反面**

##### 🔴 最接近 (A) 的那個數字
> *"in the coding task, KIVI LRU achieves a DRAM cache hit rate of **38%** when quantized to 2 bits. In contrast, our method achieves hit rates of **81%, 56%, 44%, and 11%**."*

一個審稿人會拿這個打你：**「2-bit（比你的 INT4 更激進）之下，adaptive policy 還是從 38% 拉到 81%，
所以激進量化之後 headroom 並沒有崩潰。」**

##### 你的三個防線（3 票驗證過的）
1. 38%→81% 的差距**混合了 placement 智慧和 per-entry 自適應壓縮率**，沒有隔離 placement。
2. 81% 是**不同品質工作點**（11%–81% 是「delay 與 quality 的不同權重」），**不是 iso-quality**。
3. 那是 **hit rate** 的差距，**不是 oracle/Belady 的延遲 headroom**；而且階是 DRAM/SSD，不是 HBM/DRAM over PCIe。

##### 對你的意義
- 🟡 **降權引用**：5 頁 workshop，評測段自題 "Preliminary Results"。
- 🔴 **但它是 (A) 的直接反命題來源**，必須正面對決，不能只放在 related work。
- ✅ **它沒有 oracle → 結構上不可能量到 headroom-vs-bit-width。(A) 的具體形式在這裡沒有被發表。**

---

#### B-3. 🔴🔴 Where Should the KV Cache Live? Placement Policies Across GPU, CPU, and SSD
> arXiv:2609.16215 ｜ **2026-09-14 投稿（6 天前）** ｜ Tumkur/Iyer/Simhadri/Kumar/Kumar/Nampelly（Vizuara）｜ cs.AI
> ⚠️ **這篇不在你的 `refs.bib` 裡。這輪才撈到。必加。**
> 🟩 證據等級 A（3 票確認，全文 6,104 字逐字 grep）

##### 它在幹嘛
用一個 **discrete-event simulator**（校準到 random-forest 執行時間預測器）掃描多階放置策略空間：
recency/LRU、reuse-frequency、predicted-reuse、EWMA × 階容量比 × workload × prefetch 深度。

##### 🔴 它的結論，跟你 (A) 的**前提**撞得很硬
逐字：
> *"These gains come from tier capacities of 1 plus 8 plus 64, **not placement policy**."*
> *"At batch one decode is compute-bound, so the placement policy barely moves throughput."*
> *"The existing predicted reuse policy is **byte identical** to recency."*
> *"Prefetching does not justify its bandwidth cost."*
> *"the oracle itself, despite seeing the future, is **slower than not prefetching in over half of** [the cells]."*

**「放置策略沒什麼價值」這句話，現在已經被別人發表了。**（雖然機制不同 —— 他們歸因於**容量比 1:8:64** 和 **batch-1 compute-bound**，不是 bit-width。）

##### 🟢 為什麼它反而幫了你
它明文寫了 (A) 要推翻的那個假設，而且**當成未經檢驗的背景假設**：
> *"KV quantization [10, 11] and eviction [12, 13] shrink the cache; **placement is orthogonal and complementary**, deciding where the (possibly compressed) blocks physically live."*
> *"placement composes with KV quantization and eviction … **we hold those fixed to isolate placement**."*

全文 grep 結果：`int4` = 0、`int8` = 0、`fp8` = 0、`bit-width` = 0、`precision` = 0、`Belady` = 0、`optimal` = 0。
它唯一的 oracle 是 **prefetch oracle**（不是 Belady 逐出 oracle），workload 是**合成生成器**，沒有 Mooncake trace。

→ **它是 (A) 的完美靶子**：一篇 6 天前的論文，把「正交且互補」當成前提寫死，然後刻意固定 bit-width。

##### 🔴 但它削弱了什麼
**「placement policy 是二階問題」這句話本身，不再是你的 novelty。**
→ (A) 的貢獻必須嚴格寫成 **「headroom 是 bit-width 的函數，而且會崩潰」**，
不能寫成「我們發現放置策略沒那麼重要」。**後面那句已經有人講了。**

##### ⚠️ 引用時的紅線
- 它是 **cs.AI 的 preprint，只有 v1，沒有 venue，純模擬器，合成 workload，batch=1**。
- 它自己的 Limitations 說：*"on real conversation traces reuse is lower and the margins compress"*、
  *"at larger batch … the value of good placement … would rise"*。
- → **可以引用它當「領域假設」的證據，不要引用它當「已證實的結論」。**
- 也**不要**單靠它一篇來論證「整個領域都假設正交」—— 要跟 KVTuner、AdaptCache、EvicPress 一起引。

---

#### B-4. 🟡 KVServe: Service-Aware KV Cache Compression for Disaggregated LLM Serving
> arXiv:2605.13734 ｜ **SIGCOMM 2026** ｜ 2026-05-13
> 🟩 證據等級 A（3 票確認，84,787 字全文 grep）

##### 它在幹嘛
**disaggregated serving**（prefill 節點 ↔ decode 節點跨網路）下，選一個壓縮策略組合來壓 KV 的傳輸量。

##### 機制：一個三段式策略空間
```
BS = C(Q(T(X)))
  T = Transformer 前處理：Delta / Hadamard / Affine
  Q = Quantizer：bit-width 降低，支援 layer-wise 和 head-wise 混合精度（新元件 MixHQ）
  C = Codec：nvCOMP
選法：Bayesian Profiling Engine（GP，<80 iterations，~20 小時，offline 搜尋省 50×）
     + Service-Aware Online Controller（解析式延遲模型 + Residual-Corrected Bandit）
延遲模型：T_p(c) = T_model(w) + V/s_p + V/(B × cr_p)
```

##### 🟡 唯一跟 (A) 有修辭重疊的地方：MixHQ
逐字：
> *"MixHQ applies aggressive ultra-low bit-width quantization to [Streaming Heads] **instead of discarding them**, while retaining Retrieval Heads in high precision."*

**這是「用量化取代丟棄」的替代宣稱** —— 它明文對標 DuoAttention 的 token dropping。
→ **「quantize rather than evict」這個修辭已經被 SIGCOMM'26 用過了。**

##### 🟢 但它沒有 (A) 的內容
全文關鍵字計數：`Belady` = 0、`oracle` = 0、`hit rate` = 0、`LRU` = 0、**`evict` = 0**、`DRAM` = 0、`PCIe` = 0、`descriptor` = 0、`INT4` = 0。
`placement` 只出現 1 次，而且指的是 **prefill/decode 階段在 GPU pool 上的放置**，不是 KV 的階層放置。
`tier` 的 14 次裡，絕大多數是**測試台的網路頻寬等級**（10/50/100 Gbps），不是記憶體階層。

→ 它的替代是在 **head 粒度、跨網路傳輸、對比 pruning baseline**；
你的替代是在 **block 粒度、多階容量、對比 Belady 上界**。**不同軸。**

##### ⚠️ 一個會被拿來打你的東西
它的 Fig. 4：*"The optimal strategy switches with bandwidth: CacheGen is optimal at very low bandwidth … overtaken by MixHQ and then KIVI; once bandwidth exceeds a threshold, communication savings no longer offset (de)compression."*

→ **「壓縮的價值取決於系統條件」這句話，KVServe 已經講過了。**
如果你的 (A) 寫成那個抽象層級，會被 KVServe 蓋掉。
**(A) 只能寫成它的實測形式：oracle/Belady headroom 隨 bit-width 崩潰。**

##### 它對 (B) 的意義（正面）
它的延遲模型 `V/(B × cr)` **把頻寬當成一個純量硬體常數**，
沒有每筆固定成本、沒有描述符數量、沒有 achieved-vs-peak 效率因子。
→ 它**體現**了 (B) 要打的那個假設。可以當引用。
（⚠️ 這條的 claim 在投票中 0-3 被推翻了 —— 被推翻的是「這件事可以用來論證 (B) 存活」的推論力道，
不是事實本身。**要用就自己再 grep 一次全文確認。**）

---

### 組 C：bit-width 軸，但不碰 placement

---

#### C-1. 🟢 KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization
> arXiv:2502.04420 ｜ **ICML 2025** ｜ Huawei Noah's Ark + CUHK ｜ v3, 36 頁
> 🟩 證據等級 A（3 票確認，PDF 117,506 字 grep）

##### 它在幹嘛
**離線**搜尋一組「每層的 KV precision pair」，在記憶體佔用和精度之間找 Pareto 前緣。

##### 機制
```
min_P (f_m(P), f_a(P))   s.t.  f_m(P) ≤ M,  f_a(P) ≤ ΔA
  搜尋空間 P ∈ S^L，S = 第 l 層的 (P^l_k, P^l_v) precision pair
  加速：intra-layer precision-pair pruning + inter-layer clustering (S^L → S_p^G)
```
典型結果：Llama-3.1-8B-Instruct 在 per-token-asym 下掃 KV8 / K4V8 / KV4 / K4V2 / KV2。

##### 🟩 確認的否定事實
`placement` = 0、`host memory` = 0、`CPU memory` = 0、`SSD` = 0、`PCIe` = 0、`LRU` = 0、**`Belady` = 0**、`NVMe` = 0、`prefix cach` = 0。
（`DRAM` 的 7 次全部是 "dramatic"；`tier` 的 26 次全部是 "Pareto frontier"；`oracle` 的 2 次都在參考文獻標題裡。）

→ **bit-width 只耦合到「品質 + 總記憶體佔用」，完全沒碰放置／階層／offload／逐出。**

##### 🟢 它是 (A) 的另一個靶子
逐字：
> *"KV cache quantization is **orthogonal** to most other KV cache management and compression methods, so it has been integrated with eviction, retrieval, and transferring."*

**又一篇頂會論文把「正交」寫成背景假設。** 跟 2609.16215 一起引，就能論證「這是領域共識」。

##### ⚠️ 用詞精確度
不要寫「KVTuner 只把 bit-width 耦合到品質」。
它的目標函數是 **(記憶體佔用 f_m, 精度損失 f_a) 的 pair**。
正確寫法：**「耦合到品質與總記憶體佔用，但不耦合到階層配置（tier assignment）」**。

---

#### C-2. 🟡 CacheGen: KV Cache Compression and Streaming for Fast LLM Serving
> arXiv:2310.07240 v6 ｜ **SIGCOMM 2024** ｜ code: github.com/UChi-JCL/CacheGen
> 🟩 證據等級 A（3 票確認，19 頁 PDF 全文 grep）

##### 它在幹嘛
把 KV cache 編碼成 bitstream，用**網路**從遠端儲存 stream 回來，並隨可用頻寬調整編碼等級。

##### 機制：per-chunk 的編碼等級選擇
逐字：
> *"each chunk can choose one of several streaming configuration: it can be sent at one of the encoding levels **or can be sent in the text format to let the LLM recompute K and V tensors**."*
> *"CacheGen estimates the bandwidth by measuring the throughput of the previous chunk. It assumes this throughput will remain constant … picks the configuration that has the least compression loss with an expected delay **still within the SLO**."*

##### 🔴 它殺死了什麼（一個你可能想寫的框架）
> **「沒有人把 bit-width 耦合到傳輸成本的決策上」—— 這句話是假的。**
> CacheGen 在 2023 就做了：per-chunk 的量化等級 × SLO 預算 × 載入 vs 重算的 fallback。

→ (A) 的存活增量**必須**明確是 **bit-width vs 放置策略智慧、對照 Belady 上界**，
不能是「bit-width 會影響系統決策」。

##### 🟢 它沒有的東西
全文計數：`Belady` = 0、`LRU` = 0、`evict` = 0、`SSD` = 0、`placement` = 0、`hit rate` = 0、`PCIe` = 0、`descriptor` = 0、`multi-tier` = 0、`substitut*` = 0。
作者自己在 §9 明文 scope out：
> *"Other aspects such as **which storage device(s) to store KV cache, caching policies**, and locating KV cache quickly are discussed in concurrent works. We leave combining CacheGen with these works to future work."*

它的儲存模型是**一台專用的網路儲存伺服器**，兩個 API（`store_kv` / `get_kv`）—— 沒有階層，沒有逐出，**所以連定義 Belady 的位置都沒有**。

##### ⚠️ Table 1 會被拿來打你
| 方法 | 大小 | 精度 |
|---|---|---|
| 8-bit quantization | 622 MB | 1.00 |
| **CacheGen** | **176 MB** | **0.98** |
| **H2O**（逐出） | **282 MB** | **0.97** |
| CacheGen on H2O | 71 MB | 0.97 |
| LLMLingua | 492 MB | 0.94 |
| CacheGen on LLMLingua | 183 MB | 0.94 |

- 潛在的**反向**論點：CacheGen 單獨（176MB/0.98）**贏過** H2O 單獨（282MB/0.97）→ 壓縮打敗逐出策略，跟 (A) 同向。
  但這是**一個 dataset 的一列**，從來沒被寫成 substitution，也從來沒對照 oracle。
- 潛在的**正向**論點（審稿人會用）：`CacheGen on H2O = 71 MB` → **堆疊有效** → complements。
  CacheGen 全文的 `complementary` 只出現一次，就是在講這個。
  → **你要預先切開軸線**：它的互補是「熵編碼 × token 剪枝」在**網路頻寬**軸；
  你的替代是「bit-width × 多階放置」在**容量**軸對照 Belady。

##### 對 (B) 的意義
它把頻寬當成**外生的網路變數**（量上一個 chunk 的吞吐、假設不變）。
全文 `PCIe` = 0、`descriptor` = 0、`theoretical` = 0、`peak` = 0、`granularity` = 0、`memcpy` = 0。
它唯一的粒度討論是 chunk 長度 1.5K tokens，理由是「太大無法及時反應頻寬變化、太小吃不滿 GPU batching」——
**沒有每筆固定成本，沒有描述符攤提。(B) 不在裡面。**

---

### 組 D：放置／I/O／重算軸，但完全不碰 bit-width

> ⚠️ 🟨 **這一組全部是 abstract 等級。** 可以用來排威脅級、可以定位，
> **但「它沒有做 X」這種否定句，必須先讀全文才能寫進論文。**

---

#### D-1. Tutti: Making SSD-Backed KV Cache Practical（arXiv 2605.03375，2026-05-05）
**問題**：KV 從 SSD 還原時 I/O 效能差、GPU stall 嚴重。
**根因（它自己講的）**：
> *"the **fragmented GPU memory layout results in a massive number of tiny random I/Os**, rendering the low-parallelism CPU a severe bottleneck even with GPU Direct Storage (GDS), which still relies on CPU intervention to initiate each I/O."*

**機制**：GPU-centric object store，把 CPU 從 HBM↔SSD 的資料／I/O 關鍵路徑上拿掉；GPU 原生抽象 + **GPU io_uring** 做非同步 direct object I/O。
**控制**：HBM/SSD 放置、slack-aware I/O 排程。**不控制**逐出策略、bit-width、重算 vs 載入。
**評測**：整合進 **vLLM**，對照 GDS-enabled **LMCache**。
**數字**：TTFT 降 78.3%（SLO 下）、request rate 2×、成本降 27%。

🔴 **這是 (B) 在「SSD 那一階」的最近鄰。**
「大量小 I/O 把 CPU 變成瓶頸」的**結構**，跟你「大量小描述符把 DMA 變成瓶頸」是**同一個故事，不同裝置**。
→ **必讀全文。** 要確認：它有沒有量 achieved-vs-peak？有沒有 I/O 大小的掃描曲線？有沒有每筆 I/O 的固定成本？
→ 對你**有利**的一面：它證明「小粒度 I/O 是 KV 系統的通病」，是 (B) 的一般性佐證。
→ 對你**不利**的一面：如果它有一條 I/O size → 吞吐的曲線，(B) 的「曲線」貢獻就被削了。

---

#### D-2. KVDrive: A Holistic Multi-Tier KV Cache Management System（arXiv 2605.18071，2026-05-18）
**問題**：現有 offload 系統把整個 cache 放 host memory、decode 時選擇性抓關鍵 entry，但
> *"sparsity cannot be pushed further without degrading accuracy … the volume of KV transfers rises sharply and becomes the dominant source of decoding latency."*

**機制**：系統層的跨階編排（GPU 記憶體 / host DRAM / SSD），不是演算法層的稀疏度精修。
**控制**：三階放置、pipeline 排程、I/O–compute overlap、跨階協調。**不控制** bit-width。
**數字**：throughput 最高 **1.74×**（保持精度）。
**對 (A)/(B)**：abstract 層級看不到 oracle、bit-width 掃描、PCIe 量測、描述符分析。
**定位**：🟢 **跟你同軸但不同層 —— 它做機制，你做「要不要做這個機制」的判準。** 引用即可。

---

#### D-3. OrbitFlow: SLO-Aware Long-Context LLM Serving with Fine-Grained KV Cache Reconfiguration（PVLDB 2026, vol.19 p.1046）
**問題**：長上下文下 request 長度和 batch 組成一直變，記憶體需求劇烈波動；靜態 offload 策略跟不上 → 過量 CPU→GPU 傳輸 → 延遲尖峰、SLO 違反。
**機制**：一個 **輕量 ILP solver**，替每個 request 決定**哪些層**的 KV 留在 GPU；用 runtime feedback 持續調整。
**「fine-grained reconfiguration」是什麼**：**逐層的 tier 放置**，**不是** precision / bit-width / layout。
**數字**：TPOT SLO 改善最高 66%、TBT 48%、P95 延遲降 38%、throughput 3.3×。
**對 (A)**：🟢 **它是「放置策略智慧有價值」的最強反證來源**——
它宣稱逐層 ILP 放置能拿到 3.3×。
🔴 **必讀全文**，要確認它的增益來自哪裡：如果來自「避免傳輸」而不是「更聰明的排序」，
那它跟 (A) 不衝突（(A) 說的是 INT4 也能避免傳輸，而且更便宜）。
**如果它的 baseline 裡有量化臂而它還是贏很多，(A) 就有麻煩。**

---

#### D-4. LeoAM: Adaptive KV Management on a Single Commodity GPU（arXiv 2506.20187，2025-06-25）
> 作者：He Sun, Li Li, Mingjun Xiao, Chengzhong Xu（⚠️ **你的 `refs.bib` 寫 `{LeoAM authors}`，要補上真實作者**）

**問題**：單張消費級 GPU 上跑長上下文，KV 必須落到磁碟，瓶頸在「token 重要性評估的開銷」+「磁碟頻寬低」。
**機制**：
1. **變動大小的 chunk 切分** —— 依各層 attention weight 的偏斜分布決定 chunk 大小，減少計算與額外傳輸。
2. **KV abstract** —— 磁碟上只存／抓每個 chunk 的「摘要」而非完整 KV。
3. 動態壓縮 + pipeline。
**控制**：GPU/CPU/Disk 放置、**chunk 粒度**、動態壓縮、pipelining。
**數字**：平均延遲 **3.46×**，大 batch 最高 **5.47×**。

🟡 **注意它的第 1 點**：**variable-sized chunk 就是一個粒度決策**。
這是 12 篇裡唯一另一個把「粒度」當成一級決策變數的。
→ **必讀全文**，確認它的粒度理由是**計算開銷**（我猜是）還是**傳輸固定成本**（那就撞到 (B)）。
→ 另外它是**單卡商用 GPU**，跟你的設備條件最像，**是最可能跑得起來的外部 baseline 之一**。

---

#### D-5. HCache: Fast State Restoration in LLM Serving（EuroSys 2025，arXiv 2410.05004）
> Shiwei Gao, Youmin Chen, Jiwu Shu（清華）

**核心點子**：**不存 KV，改存中間 activation**，還原時從 activation 重建 KV。
這在「重算（貴在算力）」和「載入 KV（貴在 I/O）」之間開了**第三條路**：載入 activation 便宜（體積小）、重建便宜（只差最後幾步）。
**兩個技術**：
1. bubble-free restoration scheduler —— 整合資源互補的方法，平衡 compute 和 IO
2. chunk-based storage manager —— 解決 layout mismatch（存的時候 layer-before-token，還原的時候 token-before-layer）

**數字**：對比 KV offload，TTFT 最高 **1.93×**，儲存空間少 **1.92–2.40×**；對比 token 重算，TTFT 最高 **5.73×**。

🟡 **它對你最重要的一點在第 2 個技術**：**layout mismatch**。
「存的順序跟讀的順序不一樣」→ 這**正是**造成小粒度散亂 I/O 的原因，跟 (B) 的描述符碎片化同源。
→ **必讀全文**，確認它的 chunk-based storage manager 有沒有量化「粒度 → 吞吐」。
**如果有，(B) 的一部分就被 EuroSys'25 搶走了。這是我對 (B) 第二擔心的一篇（僅次於 bottlenecks2026）。**

---

#### D-6. KVPR: I/O-Aware KV Cache Partial Recomputation（ACL Findings 2025，arXiv 2411.17089）
> Chaoyi Jiang, Lei Gao, Hossein Entezari Zarch, Murali Annavaram（USC）

**它在解什麼**：offload 到 CPU 之後瓶頸轉到 PCIe。前人要嘛 overlap、要嘛 CPU-GPU 異質執行，但都受限於過量資料搬運和對 CPU 能力的依賴。
**機制（跟 Cake 的差別在這裡）**：
```
Cake：       一邊從頭「算」，一邊從尾「載 KV」，兩個指標對撞
KVPR：       CPU 先傳「部分 activation」→ GPU 用它開始「重算」KV
             同時，剩下的 KV 直接從 CPU 傳過來
             → 重算與傳輸 overlap
```
**關鍵差異**：Cake 傳的是 **KV**；KVPR 傳的是 **activation**（比 KV 小），用 activation 換重算的起點。
**系統元件**：profiler（看輸入特徵 + 硬體資訊）+ scheduler（最佳化計算／通訊分配）+ runtime。
🔴 **注意：KVPR 有 profiler + scheduler，也就是它比 Cake 多了一層「決策」。**
上一份文件說 Cake「沒有成本模型、沒有估計器」—— **KVPR 可能補上了那個洞。**
**數字**：decode 延遲降最高 **35.8%**、throughput 高 **46.2%**。

🔴 **必讀全文。** 要確認三件事：
1. scheduler 的最佳化目標是不是一個**顯式的成本模型**？（若是，你對 Cake 的攻擊要改寫成「Cake 沒有，但 KVPR 有」）
2. 它有沒有 fallback？
3. 它有沒有量 PCIe 的 achieved-vs-peak？（abstract 提到 "limited bandwidth of the PCIe connection"，但沒給數字）

---

#### D-7. HetMem: Dynamic KV Cache Placement in Heterogeneous Memory System（IEEE CAL 2025，arXiv 2508.13231 v2）
**硬體設定**：**HBM + off-package DRAM**（NVLink、LPDDR5X），不是 PCIe，不是 SSD。
**它做什麼**：**把放置問題形式化，推導一個理論上界 —— 不提出具體排程策略。**
**數字／量化／oracle／頻寬分析**：abstract 層級看不到。

🟢 **威脅低**，但有一個小地方要注意：
**「推導理論上界」跟你的 oracle 是同一類東西。**
→ 讀一下它的上界怎麼定義。如果它的上界是**容量無限**的那種（不是 Belady），那跟你無關。
→ 它只有 2 頁（IEEE CAL 是 letter），讀起來很快。**CP 值最高的必讀。**

---

#### D-8. YaKV: KV Cache Offloading for Context-Intensive Tasks（Yandex Research，arXiv 2604.08426）
> Andrey Bocharnikov, Ivan Ermakov, Denis Kuznedelev, Vyacheslav Zhdanovskiy, Yegor Yershov
> 投稿 2026-04-09，**最終版 2026-09-01**。**code 公開。**

**它在幹嘛（跟你想的不一樣）**：這篇**不是系統論文，是評測／品質論文。**
前人評測 KV offload 都用「不需要從 context 撈很多資訊」的任務。他們做 context-intensive 的任務。
- 釋出 **Text2JSON** benchmark（從 raw text 抽結構化知識）
- 在 Llama 3 和 Qwen 3 上發現 **顯著的精度退化**
- 歸因兩個原因：**key 的低秩投影** + **不可靠的 landmark**
- 提出一個更簡單的替代策略

**結論句**：
> *"These findings highlight the need for a comprehensive and rigorous evaluation of long-context compression techniques."*

##### 🟢 這篇是**盟友**，而且可能是你最缺的那塊拼圖
還記得 `EXPERIMENTS_20260919.md` §1.4 那個致命缺口嗎？

> 「唯一能讓放置策略有價值的是**品質約束**，而那個約束的大小 **目前量不出來**（n=120，每個 ε 都與零不可區分）」
> 「只看 GSM8K 會得出『量化免費』的錯誤結論」

**YaKV 正好提供了一個對壓縮敏感的公開 benchmark（Text2JSON）+ 公開 code。**

🔴 **行動建議**：這是 12 篇裡**最該優先動手的一篇**，但不是拿來打，是拿來用：
```
把 Text2JSON 拿來當 ε 量測的任務
→ 如果在 Text2JSON 上 INT4 的 ε 顯著 > 0（而 GSM8K 上量不出來）
→ 你就同時得到：(i) 品質約束的真實大小 q
                (ii) 一個「為什麼大家以為量化免費」的解釋
                (iii) (A) 的完整版：headroom 是 (bit-width, 品質預算) 的二維函數
```

---

## 5. 跟目前 SOTA 差在哪（回答「跟目前 sota 差在哪」）

### 5.1 決策空間的佔領圖

```
                    │ bit-width │ 多階放置 │ 逐出 │ 算vs載 │ I/O 粒度 │ Belady 上界 │
────────────────────┼───────────┼──────────┼──────┼────────┼──────────┼─────────────┤
KVTuner (ICML'25)   │    ✅     │          │      │        │          │             │
CacheGen (SIGCOMM'24)│    ✅     │          │      │   ✅   │          │             │
KVServe (SIGCOMM'26)│    ✅     │          │      │        │          │             │
AdaptCache (BigMem) │    ✅     │    ✅    │  ✅  │        │          │             │
EvicPress (preprint)│    ✅     │    ✅    │  ✅  │        │          │             │
KVDrive (preprint)  │           │    ✅    │  ✅  │        │    ✅    │             │
OrbitFlow (PVLDB'26)│           │    ✅    │      │        │          │             │
LeoAM (preprint)    │    🟡     │    ✅    │      │        │    ✅    │             │
HetMem (IEEE CAL)   │           │    ✅    │      │        │          │    🟡 上界  │
Tutti (preprint)    │           │    ✅    │      │        │    ✅    │             │
2609.16215 (新)     │           │    ✅    │  ✅  │        │          │  🟡 prefetch│
HCache (EuroSys'25) │           │          │      │   ✅   │    🟡    │             │
KVPR (ACL'25)       │           │          │      │   ✅   │          │             │
Cake (ICML'25)      │           │          │      │   ✅   │          │             │
bottlenecks (MLSys) │           │          │      │        │  🟡 一句 │             │
────────────────────┼───────────┼──────────┼──────┼────────┼──────────┼─────────────┤
👉 你                │    ✅     │    ✅    │  ✅  │   ✅   │    ✅    │  ✅ Belady  │
```

**空白的那一欄（Belady 上界）就是你的護城河。整個領域沒有一篇有。**

### 5.2 (A) 的存活增量 —— 精確表述

> ✅ **可以宣稱**：
> 「在真實 serving trace（Mooncake）上，量化 **Belady/offline-optimal 的放置 headroom 作為 KV bit-width 的函數**，
> 並證明它在 INT4 下從 20.7%（95% CI 20.2–23.8）崩潰到 2.0–3.5% ——
> 亦即 bit-width 與放置策略智慧在延遲上是**替代品**，而文獻（KVTuner ICML'25、AdaptCache、EvicPress、arXiv 2609.16215）
> 一致地把它們當成**正交且互補**，並刻意固定其中一個來隔離另一個。」

> ❌ **不可以宣稱**：
> - 「我們發現放置策略沒那麼重要」→ **2609.16215 已發表（不同機制：容量比 + batch-1）**
> - 「我們首次把 bit-width 耦合到系統決策」→ **CacheGen 2023 已做（per-chunk 編碼等級 × SLO）**
> - 「壓縮的價值取決於系統條件」→ **KVServe Fig. 4 已講**
> - 「用量化取代丟棄」→ **KVServe MixHQ 已講（head 粒度）**
> - 「壓縮能提高快取命中率」→ **AdaptCache 已展示（2-bit，38%→81%）**

### 5.3 (B) 的存活增量 —— 精確表述

> ✅ **可以宣稱**（四點，缺一不可）：
> 1. **機制**：per-descriptor 固定成本 **12.9–16.3 µs**，與 payload 大小無關 —— 沒有人量過這個數字。
> 2. **曲線**：描述符大小 8 KiB → 4 MiB，achieved 從 **4.4% → 88.9%**（相對實測 bulk copy 57.4 GB/s）。
> 3. **跨模型 17.1×**：同一張 MI300X、同一條 PCIe Gen5 x16、同一個 vLLM，2.27 / 4.47 / 38.28 GB/s，
>    **三個端到端數字全部被 microbenchmark 重現**（2.4 / 4.7 / 38.9）。
> 4. **ROCm 根因 + 修法**：Triton 快速路徑在 ROCm 被停用 → `hipMemcpyBatchAsync` 因 ROCm 7.2.1 bug 必須 `numAttrs=0`
>    → 失去 `srcAccessOrder=ANY`。合併到 4 MiB 可望 **10.8×**。

> ❌ **不可以宣稱**：
> - 「我們首次發現 KV offload 的實際頻寬遠低於峰值」→ 🔴 **MLSys'26 已發表（15 GB/s = 23% of 64 GB/s peak，vLLM+LMCache，H100）**
> - 「沒有人研究過 KV 搬運的粒度效應」→ 🔴 **bottlenecks2026 點名 "transfer granularity"；LMCache #5067 有 128 KB 門檻；SGLang #40278 有 op 粒度表**
> - 「有效頻寬是實作性質不是硬體性質」當成**發現**→ 🟡 SGLang #40278（同期）已在生產引擎上公開示範
> - 「κ 跨硬體變動是核心貢獻」→ 🔴 **你自己的實驗 #2 已經推翻它**（變異主要來自軟體粒度，同卡跨模型 17×）

### 5.4 ⚠️ 兩個發現合成一篇時，最大的結構性風險

**(A) 和 (B) 指向相反的方向，審稿人一定會抓：**

| | (A) 說 | (B) 說 |
|---|---|---|
| 傳輸成本 | INT4 把傳輸量縮 3.77× → 放置不重要了 | 傳輸頻寬被軟體壓扁 17× → 傳輸超級重要 |
| 隱含結論 | 別做聰明放置 | 修好描述符，然後…？ |

🔴 **如果 (B) 的修法成立（CPU 階 4.5 → 48.5 GB/s，10.8×），
那 (A) 的整個成本模型要重算 —— INT4 省下的傳輸成本會變得微不足道，
而 (A) 的 20.66% → 2.02% 的結論可能會翻轉。**

**這不是缺點，這是你的論文主線。** 正確的合成方式是：

> **「多階 KV 系統的最佳決策，對一個被實作決定、而非被硬體決定的常數極度敏感。
> 我們示範了同一個決策問題，在描述符粒度改變之下，最佳解會從『量化』翻到『放置』。」**

→ 這一句同時用到 (A) 和 (B)，而且**沒有任何一篇現有工作能講**（因為沒人同時有 oracle 和粒度曲線）。
→ 🔴 **但它需要一個目前還沒做的實驗：在合併描述符之後，重跑 §2 的 headroom 表。**

---

## 6. 你原本選的 vs 這次的 —— 為什麼「不要」（回答最後一問）

### 6.1 先講清楚：**沒有任何一篇被「不要」**

這 12 篇**全部是你自己 `refs.bib` 裡的**。這輪沒有刪掉任何東西，改的是**角色**：

| 論文 | 原本的角色 | 現在的角色 | 為什麼改 |
|---|---|---|---|
| **bottlenecks2026** | related work 一行 | 🔴 **(B) 的主要對手 + 主要 motivation** | 它已經發表了 achieved-vs-peak |
| **EvicPress** | related work 一行 | 🔴 **(A) 的主要對手** | 它的前提就是 (A) 的反命題 |
| **AdaptCache** | 次要對手 | 🔴 **(A) 的反命題來源**，但降權（5 頁 workshop） | 唯一同做 bit-width×placement，但無 oracle |
| **KVTuner** | 「bit-width 決策」的威脅 | 🟢 **靶子／盟友** | 它明文寫「正交」，是你要打的共識的證據 |
| **KVServe** | 「壓縮」的威脅 | 🟡 **修辭競爭者** | MixHQ 講過「量化取代丟棄」，但不同軸 |
| **YaKV** | 「唯一跑得起來的 baseline」 | 🟢 **工具，不是對手** | 它是品質 benchmark，正好補你量不出 ε 的洞 |
| **KVPR** | 「跟 Cake 同格」 | 🟡 **可能補上 Cake 的洞** | 它有 profiler+scheduler，Cake 沒有 |
| **HCache** | 「重算 vs 載入的另一解」 | 🟡 **(B) 的隱形威脅** | layout mismatch = 粒度碎片化的同源問題 |
| **Tutti** | 「SSD 階的結論」 | 🟡 **(B) 在 SSD 階的最近鄰** | "massive number of tiny random I/Os" |
| **LeoAM** | 「設備條件相同」 | 🟡 **粒度決策的另一個玩家** | variable-sized chunk |
| **KVDrive / OrbitFlow / HetMem** | 高風險 | 🟢 **引用即可** | 同軸不同層，或只有理論上界 |

### 6.2 唯一真正的「你少了一篇」

**arXiv 2609.16215（2026-09-14）不在你的 bib 裡，而它是這輪對 (A) 最直接的威脅。**
6 天前才投的，你不可能知道。**必加。**

### 6.3 上一份文件 §8 的結論，在這輪需要修正

上一份說：
> 「不是你選錯論文，是你選對了論文卻沒跟它們比。」

**這輪之後要補一句：**
> **「而且你引的那些論文裡，有兩篇已經做掉了你一半的宣稱 ——
> bottlenecks2026 做掉了 (B) 的 achieved-vs-peak，2609.16215 做掉了 (A) 的『放置不重要』。
> 你不是沒跟它們比，你是沒讀它們。」**

這是這輪最重要的一句話。**在跟老師報告之前，至少要把 bottlenecks2026 和 EvicPress 全文讀完。**

---

## 7. 還要注意、但不在這 12 篇裡的

| 來源 | 為什麼要注意 | 優先度 |
|---|---|---|
| 🔴 **arXiv 2609.16215** | 6 天前的新論文，搶走 (A) 的前提 | **P0 必加必讀** |
| 🔴 **SGLang PR #37701 / #37635** | ROCm 的 KV 傳輸調參，比你早 17 天，**數字未確認** | **P0 必讀** |
| 🟡 **SGLang PR #40278** | 同期工作，achieved-vs-peak + op 粒度表 | P1 必提（concurrent work） |
| 🟡 **LMCache PR #5067** | 128 KB 粒度門檻 = 粒度效應在工程界已知 | P1 |
| 🟡 **Strata（OSDI'26）／Bidaw（FAST'26）** | 上一輪老師指定的，這輪沒重查 | P1 |
| 🟡 **kvsurvey2026**（ACL Findings'26, arXiv 2607.08057） | 一篇 system-aware KV survey —— **它怎麼分類你，就是審稿人怎麼分類你** | P1 |
| 🟢 **SAECache**（arXiv 2605.18825）| "Not All Tokens Are Worth Caching" —— prefix cache 的語意逐出 | P2 |
| 🟢 **QEvict**（arXiv 2608.05326）| "**Recoverable Quantized KV Eviction**" —— 標題就是量化×逐出 | 🔴 **P1，標題太像了** |
| 🟢 **TrimKV / KVAdaQuant** | bit-width 軸的其他玩家 | P2 |
| ❓ **vLLM / LMCache / TensorRT-LLM 的 achieved-vs-peak** | 除了 bottlenecks2026 用的 vLLM+LMCache 之外，**沒有找到其他公開量測** | —— |
| ❓ **`hipMemcpyBatchAsync` 粒度限制的公開陳述** | **沒有找到任何公開文獻或 issue 這樣說** → 這一條 (B) 可能是真的首次 | 🟢 **對你有利** |

> 🔴 **QEvict 被我漏掉了**：`refs.bib` 裡有 `qevict2026`，標題是
> *"Recoverable Quantized KV Eviction for Attention-Drift-Robust Long-Context Decoding"*。
> **「量化 × 逐出」寫在標題上**，但這輪的 12 篇清單沒有包含它。**應該補查。**

---

## 8. 紅線

### 8.1 必須寫進論文的三句話（不寫就是不誠實）

1. > 「Meng et al. (MLSys 2026) 已量測到 vLLM + LMCache 的 sustained PCIe 頻寬為 **15 GB/s（峰值 64 GB/s 的 23%）**，
   > 並將其歸因於 CPU-GPU memcpy 開銷、NUMA 效應與 **transfer granularity**。
   > 我們的貢獻是**隔離並量化**其中的 granularity 項……」

2. > 「Tumkur et al.（arXiv 2609.16215，concurrent）獨立報告放置策略對 throughput 影響甚微，
   > 但歸因於階容量比與 batch-1 的 compute-bound regime，**且明文固定 bit-width 不變**。」

3. > 「SGLang PR #40278（2026-09-19，concurrent work）在生產引擎上公開展示了
   > host↔device KV 傳輸的實作差異可造成 2× 的頻寬差距。」

### 8.2 不可以寫的（見 §5.2、§5.3 的 ❌ 清單）

### 8.3 這輪被 3 票推翻、**不可引用**的 11 條宣稱

> 這些宣稱在對抗式投票中沒有過半，**不代表一定是假的，代表目前沒有證據支持**。
> 想用任何一條，必須自己重新從一手來源確認。

1. ❌「AdaptCache 是 (A) 最接近的先行工作，決策空間相同」（1-2）
2. ❌「AdaptCache 的 38%→81% 直接反駁 (A)」（0-3）
3. ❌「AdaptCache 把頻寬建模成裝置常數，是 (B) 存活的積極證據」（1-2）
4. ❌「KVServe 的 `V/(B·cr)` 模型體現了 (B) 要打的假設」（0-3）—— **事實可能為真，推論力道被否**
5. ❌「KVTuner 明文的 complements 立場是 (A) 存活的積極證據」（0-3）
6. ❌「SGLang #40278 的 97→192 GB/s 就是 (B) 的同一個結構性宣稱」（0-3）
7. ❌「SGLang #40278 的 32 KB vs 512 B（7-9×）就是 (B) 的固定成本機制」（1-2）
8. ❌「SGLang #37701 在 AMD 上公開了 (B) 的論點」（1-2）—— 🔴 **但你還是要自己去讀**
9. ❌「SGLang #37701 的機制是 occupancy 不是 descriptor，所以 (B) 存活」（1-2）
10. ❌「LMCache 維護者已發表每筆 DMA 固定成本與粒度模型（0.6 µs / 34 KB break-even）」（1-2）
11. ❌「LMCache #5067 的粒度數字與 (B) 的 4-8% 有直接數值衝突」（0-3）

---

## 9. 下一步（按優先序）

| # | 事情 | 為什麼 | 工作量 |
|---|---|---|---|
| **1** | 🔴 讀 **bottlenecks2026 全文** | 它是 (B) 最大的威脅，而且我只看到片段 | 半天 |
| **2** | 🔴 讀 **EvicPress 全文** | 它是 (A) 最大的威脅，我只有 abstract | 半天 |
| **3** | 🔴 讀 **SGLang PR #37701 / #37635** | ROCm，比你早 17 天，數字未確認 | 1 小時 |
| **4** | 🟡 把 **arXiv 2609.16215** 加進 refs.bib | 6 天前的新論文，會被審稿人拿來問 | 10 分鐘 |
| **5** | 🟡 讀 **HCache §chunk-based storage manager** | layout mismatch = (B) 的同源問題 | 2 小時 |
| **6** | 🟡 補查 **QEvict**（arXiv 2608.05326） | 標題就是「量化 × 逐出」，這輪漏了 | 1 小時 |
| **7** | 🟢 **拿 YaKV 的 Text2JSON 量 ε** | 這是 (A) 完整版唯一缺的那塊 | 2–3 天 |
| **8** | 🟢 **合併描述符後重跑 headroom 表** | (A)+(B) 合成的那條主線需要它 | 3–5 天 |
| **9** | 🟢 把 `κ` 改符號 | 跟 bottlenecks2026 的 `κ_crit` 撞名 | 半天 |

---

## 附錄：來源與驗證方式

### 這輪各篇的實際證據來源

| 論文 | 怎麼查的 | 等級 |
|---|---|---|
| arXiv 2609.16215 | HTML 全文 6,104 字，3 個 agent 逐字 grep，關鍵字計數（`int4`=0、`Belady`=0、`oracle`=5 其中 1 在參考文獻標題） | 🟩 A |
| AdaptCache | v1 + v2 + BigMem'25 camera-ready PDF 三版本全文 | 🟩 A |
| KVServe | HTML 全文 84,787 字 grep | 🟩 A |
| KVTuner | v3 PDF 36 頁，117,506 字 grep | 🟩 A |
| CacheGen | v6 PDF 19 頁（SIGCOMM'24 camera-ready）+ v1 對照 | 🟩 A |
| SGLang #40278 | GitHub REST API raw body（非渲染頁），交叉驗證 `hicache_io_backend` 預設值 | 🟩 A |
| **bottlenecks2026** | **arXiv abstract + HTML 全文兩次定向擷取** | 🟨 B |
| EvicPress / Tutti / KVDrive / OrbitFlow / LeoAM / HCache / KVPR / HetMem / YaKV | **arXiv abstract 頁** | 🟨 B |
| LMCache #5067 | GitHub 頁面擷取（**非** API，可能漏留言串） | 🟨 B |
| SGLang #37701 / #37635 | **只確認存在，內容未查證** | 🟥 C |

### 已知的方法學弱點
1. **等級 B 的否定句不可信**：abstract 沒提 ≠ 全文沒有。§4 組 D 的所有「❌ 沒有 X」都要讀全文才能寫進論文。
2. **WebFetch 用小模型摘要**：逐字引號我盡量保留了原文，但**寫進論文前每一句引文都要回原文對過**。
3. **多位驗證者的 WebSearch 額度耗盡（200/200）**，所以幾條結論只靠一手全文、沒有第三方交叉比對。
   對「這篇論文有沒有 X」這種問題，一手全文本來就優於二手評論，影響不大；
   但對「有沒有**別人**做過 X」這種存在性問題，**覆蓋率是不足的**。
4. **時效**：SGLang #40278 是 1 天前、2609.16215 是 6 天前、LMCache #5067 是 9 天前。
   **這個領域的 prior art 每週都在變。投稿前要再掃一次。**
