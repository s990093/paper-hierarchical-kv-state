# F1：開 CPU 卸載反而變慢——重現、找原因、查上游

> 判準（§2）寫於 **2026-10-10T07:41:00Z**（`date -u`），在任何 GPU 執行之前。之後只做「追加」修訂，主判準不改。
> 接續 D2（`../research_20261010_directions/D2_vllm_drop.md` §4.3）。

---

## 0. 一句話結論＋判定

**判定：有看頭（系統／工程層級）**——照 §2.3 三個條件都成立；但這是工程發現，不是 Tiara 的研究貢獻。

- **一句話**：在 ROCm 上，vLLM 0.28 只要設定了任何 KV connector，就把 AMD 的 `ROCM_ATTN` 從候選拿掉，改用慢 2.4 倍的 `TRITON_ATTN`；「開卸載變慢」幾乎全是這個換 kernel 造成的，不是搬資料。
- 條件 1，**重現**：3 格 × 3 個 seed 全部變慢 ≥ 10%（doc C＝1 首輪 +96%、doc C＝16 回來 +53%、chat C＝1 回來 +86%～+98%）〔實測，§4.1〕。
- 條件 2，**機制解釋 ≥ 50%**：完全不開卸載、只強制 `TRITON_ATTN`，就解釋了 106%（doc C＝1）、105%（doc C＝16）、83%（chat C＝1）的多出時間〔實測，3 seed 中位數，§4.2〕。profiler：同一個 33K prefill，attention kernel 1.367 s → 3.277 s，GEMM 不變〔實測，§4.3〕。
- 條件 3，**上游沒修**：main（2026-10-10）和 v0.31.0 都還在排除〔原文，我自己 curl 過，§5.1〕。上游**知道**這個機制（#60316，2026-10-06 開，講的是 PD／decode／MI355X），維護者說 ROCM_ATTN 支援 connector「可能不在近期 roadmap」。
- **修法示範（只在自己腳本裡 monkeypatch）**：把每層 `(2, num_blocks, …)` 的 KV 拆成 K、V 兩個 view 註冊給 connector，就能用 ROCM_ATTN。多出來的時間拿掉 74%（doc C＝1）、67.5%（doc C＝16）、45%（chat C＝1）〔實測，3 seed 中位數，§4.4〕；KV 來回位元組比對 511／512 和上游原本的路一樣〔實測，§4.5〕；不拆的負對照 0／512、輸出被弄壞，證明檢查抓得到錯。
- 剩下的時間全是搬資料（connector 不搬時和 off 一樣：2.249 vs 2.242 s），而且跟 **copy 次數**走：這台機器上每次 copy 有固定成本（D2H 每次約 6 µs GPU 時間，資料減半時間幾乎不變；主執行緒送出約 3.5 µs／次）。簡單合併 copy 後可拿掉 70%–79%（seed 1，§4.6）。
- **對專案的影響（要先知道）**：C6 量的「vLLM 重算速度」是在 Triton 上量的（32K 冷啟動 4.58 s；預設 backend 是 2.24 s）。phase-1 報告與負面結果草稿裡「harness 和 vLLM 同一個量級、不是稻草人」的說法要重量（§6.1）。
- 附帶：chat 回來請求從 CPU 載入 16K token，比用 ROCM_ATTN 重算還慢 15%（1.678 vs 1.460 s）〔實測，§4.4〕。D2 的 NFS 層「等 83 s 不重算」重跑一次也重現（最長等 89.3 s）〔實測，§4.7〕。

---

## 1. 問題

**問題是什麼**
- D2 看到：vLLM 0.28、Llama-3.1-8B、MI300X，打開 CPU 卸載（OffloadingConnector，CPU＝工作集 50%）之後，大部分格子的 TTFT 反而變慢。
  - 例：doc、C＝1，新請求的 TTFT 2.25 s → 4.40 s。
  - D2 事先猜的原因（等寫入完成）量出來是 0，所以原因不明。
  - 只有 1 個 seed，而且 9 個 run 裡 7 個被 gpu_guard 標記（已知的 pid −1 假警報）。

**要證明什麼**
1. 換 3 個 seed 還會不會變慢？（不會 → 是假象，停。）
2. 多出來的時間花在哪？能不能用「只關掉一個東西」的實驗，找到解釋一半以上的原因？
3. 上游 vLLM 知不知道？修了沒？
4. 如果原因找得到又沒修：能不能在自己的腳本裡示範修法？

**名詞（白話）**
- **TTFT**：從送出請求到第一個輸出 token 的時間。新請求的 TTFT 幾乎就是 prefill（把整段 prompt 算一遍）的時間。
- **卸載（offload）**：把 GPU 上的 KV 複製一份到 CPU 記憶體；之後同一段文字再來，可以從 CPU 搬回來，不用重算。
- **attention backend**：vLLM 裡「算 attention 的那支 GPU 程式」。ROCm 上有好幾種：
  - `ROCM_ATTN`：AMD 寫的版本，prefill 用一支 Triton 的 prefix-prefill kernel（`_fwd_kernel`），decode 用 C++ kernel。
  - `TRITON_ATTN`：通用的 Triton 版本（`kernel_unified_attention`），哪裡都能跑。
- **KV 的排法（layout）**：
  - `ROCM_ATTN` 每層是 `(2, num_blocks, …)`：先放全部 block 的 K，再放全部 block 的 V。
  - connector 想要「同一個 block 的資料連在一起」（blocks-first）。
- **消融（ablation）**：一次只拿掉一個因素，看結果變多少。

---

## 2. 事先寫好的判準（pre-registration）

**寫於 2026-10-10T07:41:00Z，在任何 GPU 執行之前。**

### 2.1 重現（任務 1）

- 引擎、模型、設定全部照 D2 §2.1（vLLM 0.28.0，`venv/tiara-v028`，同行程，GPU KV 16 GiB，`max_num_seqs`＝16，`max_num_batched_tokens`＝8192，prefix caching 開，16 個 session × 4 輪，輸出 32 token，下一輪插到待送佇列隨機位置）。
- 兩組：`off`（不卸載）、`cpu50`（CPU＝33.0 GiB，和 D2 一樣用 `ws_bytes × 0.5`）。
- 3 格（每格的「主指標」照 D2 發現寫死）：

  | 格 | 主指標（中位數） |
  |:--|:--|
  | chat，C＝1 | 回來的請求（第 2–4 輪）TTFT |
  | doc，C＝1 | 首輪（新請求）TTFT |
  | doc，C＝16 | 回來的請求 TTFT |

- seed：**1、2、3**（新的隨機 token；D2 的 seed 0 只當參考，不算在 3 個裡）。每個 (組, seed) 一個新行程。off 和 cpu50 交錯跑。
- 每一格、每個 seed 算「變慢比例」＝ cpu50 主指標 ÷ off 主指標 − 1。
- **某一格「重現」**＝3 個 seed 裡至少 2 個的變慢比例 ≥ 10%。
- **某一格「是假象」**＝3 個 seed 裡至少 2 個的變慢比例 < 10%（照任務的寫法）。
- **整體「重現」**＝3 格裡至少 2 格重現。**整體「是假象」**＝3 格裡至少 2 格是假象 → 判 **死路**，寫完就停。
- GpuWatcher：外來者只要不是「pid −1、而且出現在自己行程結束的時候」，那個 run 作廢重跑。pid −1 的判斷規則沿用 D2 的 `guard_status`（全部外來者是 pid −1，第一次出現在最後一格寫完之後）。這條規則 D2 是看過旗標才寫的；這次是事先沿用，不改。

### 2.2 找原因（任務 2）

- 「多出來的時間」主要用 **doc，C＝1，首輪 TTFT**（沒有排隊，最乾淨）：excess ＝ cpu50 − off（3 個 seed 的中位數）。
- 每一個候選原因，用「只關掉這一個東西」的實驗（設定或自己腳本裡的 monkeypatch，不改 venv）量：關掉之後 excess 少了多少。
  - 「解釋比例」＝（原本的 excess − 關掉後的 excess）÷ 原本的 excess。
  - 如果用 profiler 歸因（不是消融），「解釋比例」＝ profiler 量到的、只在 cpu50 才有（或多出來）的時間 ÷ excess。消融優先，profiler 次之。
- 候選（照任務列的，先讀原始碼再決定哪些要量；沒量的寫 NOT_MEASURED）：D2H 複製在計算 stream 上或有同步；pageable 對 pinned 記憶體；很多小的逐 block 複製；connector 每一步的 Python 開銷（scheduler 或 worker）；prefix hash 查詢；GPU KV 變小；ROCm 特有的 memcpy 路徑；以及讀原始碼時發現的其他差異（例如開 connector 後 KV cache 的記憶體排法或 attention 後端改變）。

### 2.3 判定（照任務給的，寫死）

- **死路**：整體是假象（§2.1），**或**變慢完全被一個「已知、上游已修好」的 issue 解釋。
- **有看頭（系統／工程）**：整體重現，**而且**找到一個機制解釋 ≥ 50% 的 excess（§2.2），**而且**上游還沒修。這時提出修法，便宜的話在自己腳本裡示範。
- **其他**：**可能**。
- 「上游已修好」的認定：要有 vLLM GitHub 的 PR／commit 網址，而且合併在 0.28.0 之後；只有 issue 沒有修 = 沒修。

### 2.3b 修訂 1（2026-10-10T08:20:12Z；**已經看過** smoke run `20261010-074630-f1-smoke-rocmfix`，還沒看過任何正式格）

- 主判準（2.1、2.3）**不改**。這裡只把 §2.2 的「消融」寫具體。
- 開跑前讀 D2 的 log 和原始碼時發現（〔實測 20261010-054520、055443 的 stdout〕〔程式碼 `platforms/rocm.py:482-485`、`v1/attention/backends/rocm_attn.py:212-215`〕）：
  - `off` 選的是 `ROCM_ATTN`；開 connector 之後，vLLM 把 `ROCM_ATTN` 從候選拿掉，改用 `TRITON_ATTN`。
  - 也就是說，「開卸載」同時改了兩件事：(a) 多了搬資料；(b) attention kernel 換了。
- 消融組（每組都跑 3 格；seed 先跑 1，時間夠再補 2、3）：

  | cfg | 跟 cpu50 差在哪 | 拿來量 |
  |:--|:--|:--|
  | `off_triton` | 不卸載，強制 `TRITON_ATTN` | 「backend 換掉」單獨造成多少：(off_triton − off) ÷ (cpu50 − off) |
  | `cpu50_nostore` | 卸載開著，但 `store_threshold`＝10⁹（永遠不存） | 「搬資料」本身多少：cpu50 − cpu50_nostore |
  | `cpu50_rocmfix` | 卸載開著，強制 `ROCM_ATTN`，自己的腳本把每層 KV 拆成 K、V 兩個 view 註冊給 connector | 修法：還剩多少 excess；輸出 token 跟 `off` 一不一致 |
  | `cpu50_rocmnaive` | 強制 `ROCM_ATTN`，**不拆**（＝上游擋掉的那種用法） | 負對照：預期 KV 搬錯、輸出不一致；證明「輸出比對」抓得到錯 |

- 「解釋 ≥ 50%」用 doc C＝1 首輪 TTFT 的 `off_triton` 消融判（§2.2 的消融優先）。其他兩格一起報，但不改判定。
- 修法「對不對」的判準（事先寫）：`cpu50_rocmfix` 和 `off`（同 seed、同格、同請求）的 32 個輸出 token 完全一樣的比例，要跟 `cpu50` 對 `off_triton`（同 kernel、只差卸載）的比例相差 ≤ 5 個百分點；而且在「回來、有 CPU 命中」的請求上也要成立。做不到 → 修法不算成立，只寫「速度可以回來，但正確性沒過」。
- 和 D2 的差異（偏離）：每個行程多一個 8K token 的暖機請求（讓 Triton 長序列 kernel 先編好）；profiler 另外開行程跑（`--profile-only`），不混在正式格裡。

### 2.3c 修訂 2（追加；時間是約略的，**已經看過** seed 1 的 off／cpu50／off_triton／rocmfix 結果）

- 主判準和 §2.3b 的正確性判準都**不改**。以下三組是看過結果之後才加的**探索**，不進判定：
  - 約 09:25Z：`cpu50_rocmfix_nostore`（看修法剩下的時間是不是搬資料）。
  - 約 09:30Z：`--kvcheck`（KV 來回逐 block 位元組比對）。原因：看到輸出 token 在「沒有 CPU 參與」的請求上也會在第 0 個 token 就不同，只靠輸出比對不夠。
  - 約 10:50Z：`--coalesce`＋卸載單位 256 token（看合併 copy 能不能把剩下的時間拿掉）。

### 2.4 選做（任務 4）

- NFS fs 層「等 83 s vs 重算」那格（D2 的 E2）照原設定再跑一次（seed 1），只描述，不進判定。跑完刪掉 fs 資料，記錄刪了多少。GPU 時間不夠就不跑，寫 NOT_MEASURED。

---

## 3. 做了什麼

### 3.1 先看 log 和原始碼（還沒跑任何 GPU 之前）

- D2 的兩個 run，log 第一段就不一樣〔實測 20261010-054520-m8-d2-off-doc、20261010-055443-m8-d2-cpu50-doc 的 stdout〕：
  - `off`：`Overriding with ROCM_ATTN out of potential backends: ['ROCM_ATTN', 'TRITON_ATTN']`
  - `cpu50`：`Overriding with TRITON_ATTN out of potential backends: ['TRITON_ATTN']`
  - `cpu50` 還多一行：`Triton kernel JIT compilation during inference: kernel_unified_attention`
- 原因在原始碼〔程式碼 `vllm/platforms/rocm.py:482-485`〕：

  ```python
  # Keep ROCM_ATTN disabled for KV connectors until connector transfer
  # semantics are validated for its asymmetric native K/V cache views.
  if not use_kv_connector:
      backends.append(AttentionBackendEnum.ROCM_ATTN)
  ```
  加上 `RocmAttentionBackend.supports_kv_connector()` 回 `False`〔程式碼 `vllm/v1/attention/backends/rocm_attn.py:212-215`〕。
  `use_kv_connector` 只看「有沒有設定 KV connector」〔程式碼 `vllm/v1/attention/selector.py:137-139`〕，**跟卸載實際有沒有搬資料無關**。
- 白話：在 ROCm 上，只要打開任何 KV connector，vLLM 就把 AMD 自己的 attention 拿掉，換成通用的 Triton 版本。
- 本機沒有裝 aiter，所以沒有第三個選項，只能退到 `TRITON_ATTN`〔實測：`importlib.util.find_spec('aiter')`＝None；D2 log 的候選清單只有這兩個〕。
- 為什麼擋：connector 註冊 KV 時，把每層 KV 當成「`num_blocks` 列、每列一個 page」來切〔程式碼 `distributed/kv_transfer/kv_connector/v1/offloading/worker.py:106-127`〕。`ROCM_ATTN` 的 `(2, num_blocks, …)` 排法照這樣切，第 i 列會是「第 2i、2i+1 個 block 的 K」，不是「第 i 個 block 的 K 和 V」→ 會搬錯資料。

### 3.2 實驗（`code/m9_f1_run.py`）

- 重用 D2 的工作負載和 monkeypatch（`import m8_vllm_drop`），設定完全照 D2（§2.1）。
- 每個 (cfg, seed) 一個新行程，3 格依序跑：doc C＝1 → doc C＝16 → chat C＝1。
- 每個 run 都用 `m7run` 包、`m7_guard_run.py`（GpuWatcher）包、`flock /mlsteam/data/tiara/gpu.lock` 包。
- 為了少等鎖，後來改成「一次拿鎖、連跑幾個 run，總長 ≤ 43 分鐘」（每個 run 仍是自己的 run_id）。
- cfg（詳見 §2.3b）：`off`、`cpu50`、`off_triton`、`cpu50_nostore`、`cpu50_rocmfix`、`cpu50_rocmfix_nostore`、`cpu50_rocmnaive`。
- **修法 `cpu50_rocmfix`**（只在自己的腳本裡 monkeypatch，沒改 venv）：
  1. 讓 `RocmAttentionBackend.supports_kv_connector()` 回 `True`，並指定 `attention_backend="ROCM_ATTN"`。
  2. 換掉 `OffloadingConnectorWorker.register_kv_caches`：每層的 `(2, num_blocks, …)` 拆成 K、V 兩個 `(num_blocks, page/2)` 的 view，各自註冊。32 層 → 64 個 tensor，每個 block 每層搬 2 次 32 KiB。
  3. 這正是 `CanonicalKVCacheTensor` 的 docstring 自己寫的做法：「For attention backends where the raw tensor has num_blocks at a non-leading physical dimension (e.g. FlashAttention's (2, num_blocks, ...) layout), the tensor is split…」〔程式碼 `vllm/v1/kv_offload/base.py:444-452`〕。0.28 的註冊程式只是沒有實作這個拆法。
- **profiler**（`--profile-only`）：新的 33K prompt，只收 GPU 活動（`ProfilerActivity.CUDA`），跑 2 次取第 2 次，按 kernel 名稱分「attention／GEMM／memcpy／其他」。
- **KV 來回比對**（`--kvcheck`）：一個 8K prompt 存進 CPU → 清掉 GPU 的 prefix cache、把 GPU KV 全部寫 0 → 同一個 prompt 再跑一次（從 CPU 載入）→ 每個 block 每層算一個位元組指紋，比對兩次是否完全一樣。
- **輸出比對**：每個請求的 32 個輸出 token 都記下來，同 seed、同格、同請求，在不同 cfg 間比對。
- 合併與判定：`code/m9_f1_analyze.py` → `results/m9_followup/f1_*.csv`。

## 4. 結果

### 4.1 重現：3 個 seed 都變慢（事先寫好的判準）

主指標（中位數，秒）；變慢比例＝cpu50 ÷ off − 1〔實測，`results/m9_followup/f1_repro.csv`；比例是〔算術〕〕：

| 格 | 主指標 | seed 1 off → cpu50 | seed 2 | seed 3 | 判定 |
|:--|:--|:--|:--|:--|:--|
| chat C＝1 | 回來請求 TTFT | 0.826 → 1.542（+87%） | 0.877 → 1.734（+98%） | 0.825 → 1.531（+86%） | 重現（3／3） |
| doc C＝1 | 首輪 TTFT | 2.242 → 4.398（+96%） | 2.244 → 4.394（+96%） | 2.244 → 4.403（+96%） | 重現（3／3） |
| doc C＝16 | 回來請求 TTFT | 43.69 → 66.95（+53%） | 43.68 → 66.94（+53%） | 43.69 → 67.02（+53%） | 重現（3／3） |

- **整體：重現**（3 格都重現；門檻是 ≥ 2 格）→ 不是假象，繼續找原因。
- 數字非常穩：doc C＝1 的首輪三個 seed 差不到 0.5%；和 D2（seed 0：2.25 → 4.40）也一樣〔實測〕。
- run_id：off＝`20261010-075828-f1-s1-off`、`20261010-092506-f1-s2-off`、`20261010-094406-f1-s3-off`；cpu50＝`20261010-085015-f1-s1-cpu50`、`20261010-093230-f1-s2-cpu50`、`20261010-095124-f1-s3-cpu50`。
- 附帶發現：doc C＝1 的「回來請求 TTFT 中位數」是**雙峰**的（GPU 命中 ≈ 0.1 s、沒命中 ≈ 2.2 s），中位數落在哪一峰看命中率是不是剛好過 50%。seed 1：off 0.132 s、cpu50 0.128 s；seed 2：off 2.260 s、cpu50 1.111 s〔實測，`f1_metrics.csv`〕。D2 寫的「doc C＝1 回來請求 2.26 → 1.11 s（快 2 倍）」就是這個雙峰效應，不能當成穩定的結論〔判讀〕。這也是為什麼 doc C＝1 事先選首輪當主指標。

### 4.2 原因：不是「搬資料」，是 attention 換成了 Triton 版

三格、三個 seed 的中位數（秒）〔實測，`results/m9_followup/f1_decomp.csv` 的 `median_over_seeds` 列；比例是〔算術〕〕：

| 格（主指標） | off（ROCM_ATTN，不卸載） | **off_triton（TRITON_ATTN，不卸載）** | cpu50（TRITON，卸載） | 多出來的時間（cpu50 − off） | **backend 單獨解釋** |
|:--|--:|--:|--:|--:|--:|
| doc C＝1（首輪） | 2.244 | **4.520** | 4.398 | 2.154 | **106%** |
| doc C＝16（回來） | 43.69 | **68.10** | 66.95 | 23.26 | **105%** |
| chat C＝1（回來） | 0.826 | **1.421** | 1.542 | 0.716 | **83%** |

- 白話：**完全不開卸載**，只把 attention 換成 Triton 版，就跟開卸載一樣慢（甚至更慢一點）。
- 事先寫好的判準用 doc C＝1 首輪：106% ≥ 50% → **找到解釋一半以上的機制**。
- 每個 seed 各自算也一樣：doc C＝1 是 105.6%／105.6%／105.5%；doc C＝16 是 105.1%／104.8%／104.7%；chat C＝1 是 82%／115%／84%〔實測，同檔逐 seed 列〕。
- 超過 100% 的原因：同樣是 Triton，「沒開 connector」比「開 connector」還慢一點（doc C＝1：4.52 vs 4.40 s；profiler 的 attention 時間 3.65 vs 3.28 s，見 §4.3）。為什麼：NOT_MEASURED。

**換一種拆法（seed 1，用 `cpu50_nostore`：connector 開著、backend＝Triton，但一個 byte 都不搬）**〔實測 20261010-103048-f1-s1-nostore；比例是〔算術〕〕：

| 格 | off | cpu50_nostore | cpu50 | 「換 backend＋connector 本身」（nostore − off） | 「真的搬資料」（cpu50 − nostore） |
|:--|--:|--:|--:|--:|--:|
| doc C＝1（首輪） | 2.242 | 4.156 | 4.398 | 1.914 s（**89%**） | 0.242 s（11%） |
| doc C＝16（回來） | 43.69 | 63.12 | 66.95 | 19.43 s（**84%**） | 3.83 s（16%） |
| chat C＝1（回來） | 0.826 | 1.307 | 1.542 | 0.481 s（**67%**） | 0.235 s（33%） |

- 兩種拆法都說：大部分（67%–106%）是 backend，搬資料本身只佔 11%–33%。
- D2 的「flush 等待≈0」也對得上：變慢根本不是在等寫入。

### 4.3 profiler：同一個 33K prefill，attention kernel 慢了 1.9 秒

新的 33,024-token prompt，只收 GPU 活動，第 2 次的 kernel 時間加總（秒）〔實測，`f1_profile_summary.csv`；run：`20261010-092136-f1-prof-off`、`20261010-092234-f1-prof-cpu50`、`20261010-102417-f1-prof-offtri`、`20261010-102523-f1-prof-rocmfix`〕：

| cfg | attention kernel | 名稱 | GEMM | D2H memcpy（另一條 stream） | 其他 |
|:--|--:|:--|--:|--:|--:|
| off（ROCM_ATTN） | **1.367** | `_fwd_kernel`（160 次） | 0.759 | 0.000 | 0.076 |
| off_triton | **3.650** | `kernel_unified_attention`（160 次） | 0.753 | 0.000 | 0.058 |
| cpu50（TRITON） | **3.277** | `kernel_unified_attention` | 0.755 | 0.434（66,048 次） | 0.059 |
| cpu50_rocmfix（ROCM_ATTN） | **1.360** | `_fwd_kernel` | 0.755 | 0.780（132,096 次） | 0.075 |

- cpu50 − off 的 attention 差 **1.910 s**〔算術〕；TTFT 的 nostore − off 差 1.914 s（§4.2）。兩條獨立的路量到同一個數〔判讀〕。
- GEMM（矩陣乘法）四組都是 0.75 s，沒變。所以不是「GPU 被搬資料拖慢」，是 attention 那支程式本身慢 2.4 倍〔算術：3.277／1.367〕。
- 修法（rocmfix）的 attention 回到 1.360 s，跟 off 一樣。
- 注意：profiler 的牆鐘時間不能用。有 memcpy 的 cfg，profiler 要記十幾萬筆 memcpy，牆鐘變成 30–115 s（沒 memcpy 的 off／off_triton 是 2.2／4.5 s，接近真實 TTFT）。上表只用 GPU 上的 kernel 時間。
- 每次 D2H copy 的 GPU 時間：cpu50 是 433.7 ms／66,048 次＝6.6 µs（每次 64 KiB）；rocmfix 是 779.5 ms／132,096 次＝5.9 µs（每次 32 KiB）〔算術〕。資料少一半，時間幾乎一樣 → **每次 copy 有固定成本，搬多少 byte 不是重點**〔判讀〕。

### 4.4 修法示範：讓 connector 也能用 ROCM_ATTN

三個 seed 的中位數（秒）；「拿掉的比例」＝(cpu50 − rocmfix) ÷ (cpu50 − off)〔實測 20261010-091253-f1-s1-rocmfix、20261010-101502-f1-s2-rocmfix、20261010-105352-f1-s3-rocmfix；比例是〔算術〕〕：

| 格（主指標） | off | cpu50 | **cpu50_rocmfix** | 還多多少（rocmfix − off） | 拿掉的比例 |
|:--|--:|--:|--:|--:|--:|
| doc C＝1（首輪） | 2.244 | 4.398 | **2.804** | +0.560 s（+25%） | **74%** |
| doc C＝16（回來） | 43.69 | 66.95 | **51.25** | +7.56 s（+17%） | **67.5%** |
| chat C＝1（回來） | 0.826 | 1.542 | **1.219** | +0.393 s（+48%） | **45%** |

- 逐 seed：doc C＝1 首輪 2.796／2.804／3.078 s（拿掉 74%／74%／61%）；doc C＝16 51.26／51.21／51.25 s（67%／68%／68%）；chat C＝1 1.219／1.264／1.213 s（45%／55%／45%）〔實測，`f1_decomp.csv`〕。
- **剩下的 excess 從哪來**〔判讀，有實測支撐〕：拆成 K、V 兩半之後，每個 block 每層要搬 2 次（32 層 × 2＝64 次），copy 次數變 2 倍。而這台機器上每次 copy 有固定成本（§4.3），所以：
  - 寫入（D2H）：GPU 上 0.434 s → 0.780 s（§4.3）。
  - 載入（H2D）：doc C＝1 回來、從 CPU 載入 33K token 的請求，cpu50（每 block 32 次 copy）TTFT 中位數 1.092 s，rocmfix（64 次）2.049 s〔實測 `f1_ret_split.csv`，seed 1〕。多出 0.957 s ÷ 多出約 66,000 次 copy（每個請求約 2,064–2,073 個 block × 每 block 多 32 次）≈ **14.5 µs／次**〔算術；兩者 attention 不同，但這類請求只算 256 個新 token，attention 差可忽略〔判讀〕〕。
  - 引擎在等載入的空轉時間也變 2 倍：doc C＝1 cpu50 11.4 s → rocmfix 21.5 s；chat C＝1 7.4 s → 13.9 s〔實測，`f1_cells.csv` 的 `idle_s`，seed 1〕。
- 為什麼每次 copy 這麼貴：0.28 在 HIP ≥ 7.1 時用 `hipMemcpyBatchAsync`〔程式碼 `csrc/libtorch_stable/cache_kernels.cu:155-173`〕，但上游 #43018 說 ROCm 7.2.x 的這個 API 其實沒有批次化，效能約等於一次一次 copy〔原文 https://github.com/vllm-project/vllm/pull/43018，子 agent 讀的〕。本機是 ROCm 7.2.2〔實測 `/opt/rocm/.info/version`〕。

**回來的請求：從 CPU 載入，比重算還慢？**（seed 1，同一批請求，依「在 cpu50 裡有沒有 CPU 命中」分組，TTFT 中位數）〔實測 `f1_ret_split.csv`〕：

| 組 | off（重算，ROCM） | off_triton（重算，TRITON） | cpu50（載入，TRITON） | rocmfix（載入，ROCM） |
|:--|--:|--:|--:|--:|
| doc C＝1，CPU 命中（14 個，各載入約 33K token） | 2.272 | 4.679 | 1.092 | 2.049 |
| chat C＝1，CPU 命中（15 個，各載入約 16K token） | 1.460 | 2.708 | 1.802 | 1.678 |

- doc：用同一種 attention 比（cpu50 對 off_triton），載入快 4.3 倍；但跟最快的重算（off）比，rocmfix 的載入只快 10%。
- chat：rocmfix（載入）1.678 s **比** off（重算）1.460 s **慢 15%**〔算術〕。也就是說，在這台機器、這個 vLLM 版本，從 CPU 搬 16K token 回來，比在 GPU 上重算還慢。
- 〔判讀〕這直接關係到 Tiara 的 κ（重算成本 ÷ 傳輸成本）：用 vLLM 0.28 的 OffloadingConnector 在 MI300X 上量到的「傳輸成本」，主要是**每次 copy 的固定成本 × copy 次數**，不是頻寬。換 ROCm 版本、換 copy 合併方式，κ 就會變。

### 4.5 修法有沒有把 KV 搬錯：兩種檢查都過；負對照抓得到錯

**(a) KV 來回逐 block 位元組比對**（8,192 token，512 個 block，seed 1）〔實測 `f1_kvcheck.csv`；run：`20261010-110725-f1-kvchk-cpu50`、`20261010-110848-f1-kvchk-rocmfix`、`20261010-111008-f1-kvchk-naive`〕：

| cfg | 第 2 次從 CPU 載入 | 第 2 次有資料的 block | 和第 1 次位元組完全一樣 | 下一個 token（第 1 次 → 第 2 次） |
|:--|--:|--:|--:|:--|
| cpu50（上游原本的路，TRITON） | 8,192 token | 512 | **511／512** | 319 → 319 |
| cpu50_rocmfix（修法） | 8,192 token | 512 | **511／512** | 319 → 319 |
| cpu50_rocmnaive（不拆，負對照） | 8,192 token | **1** | **0／512** | 319 → **279** |

- 511 而不是 512〔判讀〕：vLLM 至少要重算最後 1 個 token，所以最後一個 block 裡有 1 個 token 是重新算的，數值和第 1 次（整段一起算）可能差一點。上游原本的路（cpu50）也是 511，所以修法和上游一樣「無損」。
- 負對照：不拆的話，connector 把每層當成「一列一個 block」來切，實際切到的是別的 block 的 K（§3.1）。這次切到的剛好都是空的區域，所以載回來的是 0，只剩重算的那 1 個 block 有值〔判讀〕。

**(b) 輸出 token 比對**（3 個 seed × 3 格 × 64 個請求，比 32 個輸出 token 是否完全相同）〔實測 `f1_output_match.csv`；百分比是〔算術〕〕：

| 比較（同一個 attention kernel，只差卸載） | 全部請求 | 有 CPU 命中的回來請求 |
|:--|--:|--:|
| off_triton vs cpu50（上游原本：TRITON） | 570／576（99.0%） | 90／94（95.7%） |
| off vs cpu50_rocmfix（修法：ROCM_ATTN） | 572／576（99.3%） | 91／95（95.8%） |
| off vs cpu50_rocmnaive（負對照，seed 1，chat＋doc C＝1） | 74／128（57.8%） | **0／29（0%）** |
| 參考：off vs off_triton（不同 kernel、都不卸載） | 512／576（88.9%） | — |

- 事先寫的判準（§2.3b）：修法和上游原本的差距 ≤ 5 個百分點 → 99.3% vs 99.0%、95.8% vs 95.7% → **過**。
- 為什麼不是 100%：就算完全沒有 CPU（例如 off vs off_triton，或 cpu50 裡只用 GPU 命中的請求），輸出也偶爾不同。隨機 token 的 prompt 讓第一個輸出 token 的機率很接近，計算順序一變（例如有沒有 prefix cache、chunk 怎麼切）就可能翻〔判讀〕。真正的「無損」證據是 (a) 的位元組比對。
- 負對照的回來請求中，**連沒有 CPU 命中的也有很多錯**（chat 回來請求只有 17／48 對）：不拆的寫法載入時寫到別的 block 的位置，會弄壞其他請求放在 GPU 上的 KV〔判讀〕。所以上游擋掉 ROCM_ATTN 是對的；要修就要像 rocmfix 一樣拆 K、V。

**(c) 負對照的速度**（seed 1）〔實測 20261010-110302-f1-s1-rocmnaive〕：doc C＝1 首輪 2.271 s（off 2.242、rocmfix 2.796）；chat C＝1 回來 0.832 s（off 0.826、rocmfix 1.219）。
- 〔判讀〕不拆的寫法每個 block 每層只 copy 1 次（和 cpu50 一樣是 32 次／block），速度就回到和 off 一樣。這支持 §4.4 的說法：rocmfix 剩下的 excess，主要是 copy 次數變 2 倍。（不拆的寫法結果是錯的，這裡只拿它的時間當參考。）

### 4.6 修法剩下的 excess：全是「搬資料」，而且跟 copy 次數走

seed 1，C＝1〔實測；run：`20261010-075828-f1-s1-off`、`20261010-111129-f1-s1-rocmfixns`、`20261010-091253-f1-s1-rocmfix`、`20261010-112119-f1-s1-rocmfixcoal`、`20261010-110302-f1-s1-rocmnaive`、`20261010-085015-f1-s1-cpu50`；欄位來自 `f1_cells.csv`（`handle_preempt_s`＋`start_xfer_s`＝主執行緒送出搬移的時間，`idle_s`＝等載入時引擎空轉的時間）〕：

| cfg | doc 首輪 TTFT | chat 回來 TTFT | 主執行緒送出搬移（doc 格／chat 格，合計秒） | 等載入空轉（doc／chat，秒） |
|:--|--:|--:|--:|--:|
| off（ROCM_ATTN，不卸載） | 2.242 | 0.826 | 0／0 | 0.43／0.38 |
| rocmfix_nostore（ROCM_ATTN＋connector，不搬） | **2.249** | **0.830** | 0.004／0.004 | 0.44／0.39 |
| rocmfix（拆 K、V；64 次 copy／block） | 2.796 | 1.219 | 19.80／10.81 | 21.54／13.87 |
| rocmfix＋合併 copy（offload block 256 token） | 2.685 | 1.042 | 9.05／4.02 | 10.97／8.66 |
| 參考：naive（不拆，32 次／block，**資料是錯的**） | 2.271 | 0.832 | 5.53／3.08 | 11.15／7.24 |
| 參考：cpu50（TRITON，32 次／block） | 4.398 | 1.542 | 7.21／3.40 | 11.37／7.38 |

- **connector 本身（不搬資料）在 ROCM_ATTN 上零成本**：rocmfix_nostore 和 off 差 0.007 s／0.004 s〔算術〕。所以修法剩下的 0.55 s 全是搬資料。
- **合併 copy**（`install_coalesce`：src、dst 位址都相連才合併；要把卸載單位設成 256 token，CPU 端才會相連）：
  - 正確性：KV 來回比對 511／512 相同，下一個 token 319 → 319〔實測 20261010-111954-f1-kvchk-coal〕。
  - 合併率：乾淨的 8K prompt（GPU block 編號連續）D2H 67,584 → 4,224 次、H2D 32,768 → 2,048 次（16 倍）；但真實工作負載裡 GPU block 編號是散的，只合併到 doc 格 D2H 3,607,552 → 1,914,624、H2D 1,491,648 → 708,928（約 2 倍）〔實測 `f1_kvcheck.csv`、`f1_cells.csv` 的 `coal_*`〕。
  - 效果：等載入的空轉減半（21.5 → 11.0 s，回到 cpu50 的水準）；doc 首輪 2.796 → 2.685 s，chat 回來 1.219 → 1.042 s。和 off 比，拿掉的比例從 74% → 79%（doc）、45% → 70%（chat）〔算術，seed 1，`f1_decomp.csv`〕。
- 主執行緒送出搬移的時間，跟 copy 次數一起變（rocmfix 19.8 s → 合併後 9.05 s）。合併那個 run 的 copy 次數有直接計數：9.05 s ÷ (1,914,624＋708,928) 次 ≈ **3.5 µs／次**（含合併本身的 numpy 時間）〔算術〕。其他 cfg 沒有直接計數（`stored_ok` 是去重後的 key 數，同一段被淘汰後重存只算一次，所以只是下限），NOT_MEASURED。
- 〔判讀〕vLLM 在同一個行程裡，scheduler、worker 送出搬移、和下一步的 forward 都在同一條執行緒上。送出搬移若比 GPU 跑完上一步還久，就會把下一步往後推，TTFT 就變長。naive 的送出時間（5.5 s）比 rocmfix（19.8 s）少很多，它的首輪幾乎沒變慢；這和「送出時間被藏在 GPU 計算後面、超過才露出來」的說法一致，但我沒有直接量「露出來的部分」：NOT_MEASURED。
- 結論：**要把卸載的成本降到 0，除了換回 ROCM_ATTN，還要把 copy 次數降下來**（gather kernel、更好的合併、或真正批次化的 memcpy）。本次只示範到「換回 ROCM_ATTN＋簡單合併」＝拿掉 70%–79%（seed 1）。

### 4.7 選做：NFS 層「等 89 s，不重算」再跑一次

- 設定照 D2 的 E2（`m8_vllm_drop.py --cfg tier50 --workload doc --conc 4 --n-sess 8`，seed 1）〔實測 20261010-111538-f1-tier50-nfs，`f1_tier50_summary.csv`〕：

| | D2（seed 0，`20261010-065309-m8-d2-tier50-doc`） | 本次（seed 1） |
|:--|--:|--:|
| 回來請求 TTFT 中位數 | 0.708 s | 0.306 s |
| p90 | 56.79 s | 48.90 s |
| 最大 | 84.26 s | **89.31 s** |
| 最長的 scheduler 延後（等資料搬回來） | 83.13 s | **88.18 s** |
| 回來請求 TTFT > 10 s 的個數 | 6／24 | 6／24 |
| 上一輪的 token 被重算的比例 | 0.07% | 0.06% |
| 寫進 NFS 的量 | 35.46 GB | 35.42 GB（跑完腳本自己刪掉；run 目錄現在 632 KB） |

（D2 那一列是用本次的 `m9_f1_analyze.py` 從 D2 的 `d2_requests.csv` 重算的〔實測 `20261010-112856-f1-analyze-final`，`f1_tier50_summary.csv` 第 2 列〕，和 D2 文件的數字一致。）
- **重現**：vLLM 一樣選擇「等 fs 搬回來」而不是重算。同一段 33K 重算：ROCM_ATTN 單獨 2.24 s、Triton（這個設定實際用的）約 4.4 s〔實測 off／cpu50 的 doc C＝1 首輪〕；等待是重算的 20–40 倍〔算術〕。
- 注意：這個 run 也是 Triton attention（開了 connector）。fs 在 NFS 上，是「很慢的下一層」，不代表本機 NVMe。

## 5. 前作／上游狀態

查證方式：一個子 agent 搜 GitHub／部落格（只讀），我自己再用 `curl`／WebFetch 複查最關鍵的三件事（標 ✔）。

### 5.1 「開 connector 就把 ROCM_ATTN 拿掉」是誰加的、修了沒

| 項目 | 內容 | 來源 |
|:--|:--|:--|
| 加入的 PR | #43660「[Attention][AMD] Standardize kv layout to blocks first for AMD」，2026-05-28 合併。理由：ROCM_ATTN 的 KV 是 `(2, num_blocks, …)`（K、V 分兩大塊），connector 要「每個 block 連續」的排法；「之後再補 ROCM_ATTN 的 blocks-first 支援」 | 〔原文 https://github.com/vllm-project/vllm/pull/43660〕✔ 改了什麼：我用本地 `git show 5b115bb8a3` 核過；PR 頁上的引文是子 agent 讀的 |
| PR 有沒有量速度 | 只比了 FA／UNIFIED 改前改後（例 5.628 → 5.614 req/s），**沒有比 ROCM_ATTN 和 TRITON_ATTN** | 〔原文，同上；子 agent 讀的，我沒逐字核〕 |
| 有人試著修 | #45234「[ROCm] Enable ROCm Attention Sinks and Connector-Friendly KV Layouts」，2026-07-28 作者自己關掉：「We are probably going to be deprecating ROCM_ATTN in favor of AITER_FA.」 | 〔原文 https://github.com/vllm-project/vllm/pull/45234〕（子 agent） |
| 上游 main 現在 | `if not use_kv_connector: backends.append(ROCM_ATTN)` 還在；`RocmAttentionBackend.supports_kv_connector()` 還是 `False`（main HEAD `a98247ab4d`，2026-10-10；v0.31.0 也一樣）→ **沒修** | ✔〔原文 raw.githubusercontent.com/vllm-project/vllm/main/vllm/platforms/rocm.py:501-504；…/rocm_attn.py:211-214；…/v0.31.0/vllm/platforms/rocm.py:498〕 |
| 最相關的 issue | #60316（open，2026-10-06）「KV connectors rule out ROCM_ATTN, so PD workers decode on ROCM_AITER_UNIFIED_ATTN」：MI355X、vLLM 0.30.1rc1、Qwen3；**decode** 每 token 慢 1.92–3.24 倍；作者說 PD 的額外成本主要來自換 backend，不是搬 KV | ✔〔原文 https://github.com/vllm-project/vllm/issues/60316〕 |
| 維護者回覆 | vllmellm（2026-10-07）：「a connector-supporting path for ROCM_ATTN may not be on the roadmap in the near future」；之後的最佳化會放在 ROCM_AITER_FA／ROCM_AITER_UNIFIED_ATTN | ✔〔原文，同上〕 |
| 0.28 為什麼掉到 TRITON，不是 AITER | 0.28 的 `RocmAiterUnifiedAttentionBackend` 繼承 ROCM_ATTN 的 `supports_kv_connector=False`；#53695 才讓它支援 connector（2026-09-10 合併，在 v0.30.0）。而且本機 venv **沒有裝 aiter**（`find_spec('aiter')`＝None），所以只剩 TRITON_ATTN | 〔原文 https://github.com/vllm-project/vllm/pull/53695〕（子 agent）；〔實測：本機 import 檢查〕 |

**判讀**〔判讀〕：
- 機制本身（「開 connector → 換掉 ROCM_ATTN」）上游**知道**（#60316，4 天前開的），但那個 issue 講的是 PD／decode／MI355X／AITER；**CPU 卸載＋prefill＋MI300X＋退到 TRITON_ATTN 這個組合，沒查到有人報過**〔未查證：只搜了 GitHub issue／PR 標題與幾篇部落格〕。
- 上游的方向是「不修 ROCM_ATTN，改推 AITER」。所以這不是「已修好的已知 issue」。

### 5.2 其他相關資料

| 主題 | 內容 | 來源 |
|:--|:--|:--|
| ROCM_ATTN 比 TRITON_ATTN 快 | vLLM 官方部落格（AMD 合寫，2026-02-27）：「ROCM_ATTN is faster than TRITON_ATTN for models with supported head sizes」；沒有長 context prefill 的數字 | 〔原文 https://vllm.ai/blog/rocm-attention-backend〕（子 agent） |
| 退到 TRITON_ATTN，prefill 慢 5.5 倍 | #39965：gfx1151（不是 MI300X），prefill attention kernel 813 ms → 4,486 ms；2026-08-17 因 stale 關閉 | 〔原文 https://github.com/vllm-project/vllm/issues/39965〕（子 agent） |
| 卸載對 cache miss 的 TTFT 影響小（CUDA） | vLLM 部落格（IBM，H100，2026-01-08）：「using the offloading connector has minimal effect on TTFT for cache misses」 | 〔原文 https://vllm.ai/blog/kv-offloading-connector〕（子 agent） |
| ROCm 的批次 memcpy 其實不是批次 | #43018（2026-08-21 合併，在 v0.29.0，不在 0.28）：ROCm 7.2.x 的 `hipMemcpyBatchAsync` 效能約等於一個一個複製；本機是 ROCm 7.2.2（`/opt/rocm/.info/version`），0.28 在 HIP ≥ 7.1 時就走這條（〔程式碼 `csrc/libtorch_stable/cache_kernels.cu:155-173`〕） | 〔原文 https://github.com/vllm-project/vllm/pull/43018〕（子 agent） |
| 並發時卸載讓 TTFT 變差（CUDA） | #44294：L40S、0.19.1、50 個並發共用 10K prefix，TTFT 5.95 → 11.46 s；修正 PR 沒合併，issue 2026-10-07 以 not planned 關閉 | 〔原文 https://github.com/vllm-project/vllm/issues/44294〕（子 agent） |
| 文件有沒有寫「開 connector 會換 backend」 | 沒有（`kv_offloading_usage.md`、docs.vllm.ai attention backends 頁、AMD vLLM 最佳化文件都沒寫） | 〔原文〕（子 agent） |

## 6. 如果要繼續

### 6.1 給這個專案的人（先看這段）

1. **C6 的「vLLM 重算速度」是在 Triton attention 上量的，比 vLLM 預設慢約 2 倍。**
   - C6（`20261008-141324-m7-c6-vllm`）是 vLLM 0.28＋OffloadingConnector，server log 寫的是 `Overriding with TRITON_ATTN`〔實測 該 run 的 `vllm_server.log:19`〕。
   - C6 冷啟動 32K＝4.584 s；本次 `off`（ROCM_ATTN，不開 connector）33K 首輪＝2.244 s〔實測 `results/m7_write_policy_mi300x/c6_vllm.csv`、`f1_decomp.csv`〕。
   - 有用到 C6 重算數字的地方：`docs/phase1_20261008/07_report.md` §7 的表（vLLM cold 8K／16K／32K＝443／1,304／4,584 ms，harness＝645／1,439／3,759 ms）和結論「計算端相當…32K 反而快 18%，所以 f(i) 不是一個被做弱的稻草人」；`docs/paper_negative_20261010/draft.md` §3.1 引用同一句。
   - 對照本次（設定不完全相同：本次 `max_num_batched_tokens`＝8192，C6 是 server 模式）：預設 backend 的 vLLM，8K 首輪 0.318 s、33K 首輪 2.244 s〔實測 `f1_metrics.csv`，off，3 seed〕。〔算術〕harness 的 645 ms／3,759 ms 是它的 2.0／1.7 倍。
   - 〔判讀〕「harness 和 vLLM 同一個量級」這個說法的對照組是被拖慢的 vLLM；跟預設 vLLM 比，harness 的重算慢約 1.7–2 倍，「不是稻草人」要重新量過才能講。我沒有改那些檔（規則：只建新檔）。D3 只用了 C6 的頻寬（4.1 GiB/s），不受這條影響。
   - C6 的「vLLM CPU 層只有 3.7–4.1 GiB/s」本身仍然是那條軟體路徑的實測值；但它的主因是「每次 copy 的固定成本 × copy 次數」（§4.3、§4.4），換 ROCm 版本或合併 copy 就會變。
2. **任何「卸載 vs 不卸載」或「載入 vs 重算」的比較，在 ROCm＋vLLM ≥ 含 #43660 的版本上，兩邊必須鎖同一個 attention backend。** 否則量到的是 kernel 差，不是放置策略的差。最簡單：兩邊都 `--attention-backend TRITON_ATTN`（慢但公平），或兩邊都用本文件的 rocmfix。
3. vLLM 0.19.1（`venv/tiara`）沒有這個排除邏輯〔程式碼：`/mlsteam/workspace/src/vllm/vllm/platforms/rocm.py` 搜不到 `use_kv_connector`〕，所以用 0.19.1 量的東西不受這一條影響〔判讀〕。

### 6.2 修法（可以送上游的版本）

- **第一步（主要）：讓 OffloadingConnector 接受 `(2, num_blocks, …)` 的 KV。** 在 `OffloadingConnectorWorker.register_kv_caches` 裡，遇到 KV 在外層的排法，就拆成 K、V 兩個 blocks-first 的 view 分別註冊（本文件的 `cpu50_rocmfix`，約 30 行；`CanonicalKVCacheTensor` 的 docstring 本來就這樣寫）。然後只對 OffloadingConnector 放行 ROCM_ATTN（別的 connector，例如 NIXL，要各自驗證）。
  - 效果（實測）：多出來的時間拿掉 45%–74%；KV 位元組 511／512 相同（和上游原本的路一樣）。
- **第二步：減少 copy 次數。** 拆成 K、V 之後 copy 次數變 2 倍，而這台機器上每次 copy 有固定成本。做法：
  - 合併位址相連的 copy（本次 §4.6 的試驗）；上游有類似的 PR #56110「合併連續的 DMA 複製」，還沒合併〔原文，子 agent 只看了標題，未查證〕。
  - 或用一支 gather kernel 先把散的 block 收進一塊連續的暫存區，再一次 D2H。
  - 或升級到 `hipMemcpyBatchAsync` 真的有批次化的 ROCm（#43018 說要 7.14 以上）〔原文，子 agent〕。
- **上游的方向**是「不修 ROCM_ATTN，改推 AITER」（§5.1）。本機沒有 aiter，所以 `VLLM_ROCM_USE_AITER=1` 這條路：NOT_MEASURED。如果要送 PR，要先在有 aiter 的環境量 ROCM_AITER_FA＋connector 是否已經夠快；夠快的話，這個修法的價值就只剩「沒裝 aiter 的使用者」。

### 6.3 研究上

- 這是**工程發現**，不是 Tiara 的研究貢獻。對 Tiara 的意義是「量測衛生」（§6.1），以及一個 κ 的實例：在 MI300X＋vLLM 0.28＋ROCm 7.2.2，chat 的回來請求從 CPU 載入 16K token **比重算還慢 15%**（§4.4），因為載入路徑被 copy 次數卡住。
- 下一步若要用：在同一個 backend 下，掃「載入 token 數 × copy 合併與否」，畫出「載入 vs 重算」的交叉點（κ＝1 的地方）。約 30 分鐘 GPU。

## 7. 失敗與異常

1. **GpuWatcher 旗標**：25 個正式 run 裡 19 個被標 `CONTAMINATED`，外來者全是 `pid −1`、峰值 34,137 MiB（＝自己的模型＋KV），第一次出現都在該 run 最後一筆結果寫完之後；照 §2.1 事先沿用的 D2 規則全部接受〔`results/m9_followup/f1_guard.csv`：26／26 通過（25 個 F1 run＋D2 的 tier run，後者只拿來重算 §4.7 的對照列），F1 的 25 個裡 6 個是 clean〕。profile／kvcheck／tier run 沒有 `f1_cells.csv`，我把規則裡的「最後一格」擴大成「這個 run 最後寫出的結果列（cells、profile、kvcheck、d2_cells 取最晚）」——這是規則的延伸，寫在 `code/m9_f1_analyze.py` 的 `guard_status`。
2. **等 GPU 鎖**：第一個正式 run（`20261010-075828-f1-s1-off`）07:58 送出，08:28 才拿到鎖（F2 在用）。之後我把逐個拿鎖的 driver 停掉（自己的 bash，exit 143；當時正在跑的 run 不受影響），改成「一次拿鎖、連跑幾個、≤ 43 分鐘」。實際最長一次持有 38.9 分鐘（≤ 45）。
3. **我自己的指令打錯**：用 `pkill -f "scratchpad/f1_groupE.sh"` 停第一版 group E 的等待腳本時，pattern 也比對到我自己那條 shell，shell 被殺（exit 144）。group E 第一版當時還在等，沒有跑任何 GPU 工作；之後換成 `f1_groupE2.sh` 重新排隊。
4. **smoke run 不進結果**：`20261010-074630-f1-smoke-rocmfix`（2 個 session）。它的 profiler 開了 CPU 追蹤，一個 33K prefill 的牆鐘變成 113 s；分類也把「CPU op」和「GPU kernel」重複算。之後改成只收 GPU、只算 GPU kernel。
5. **profiler 的牆鐘不能用**：有 memcpy 的 cfg（cpu50、rocmfix）profiler 要記十幾萬筆 memcpy，牆鐘 30–115 s。只用 kernel 時間（§4.3）。
6. **每一步的計時不能用**：`f1_steps.csv`（只在 run 目錄）記的是「這一步排了多少 token、`step()` 花多久」。因為 async scheduling，這一步排的 token 和這一步等到的 GPU 工作對不上（chat 8K 整塊的步只量到 0.038 s）。沒有用在任何結論。
7. **kvcheck 的「等寫入完成」沒有等到**：第 1 次之後我呼叫 `step()` 最多 200 次想讓 store 做完，但沒有請求時 `step()` 幾乎不做事（0.022 s 內跑完，`_num_write_pending_blocks` 還是 512）。之後把 GPU KV 清 0 時，理論上可能和還沒做完的 D2H copy 搶。這只會讓「比對失敗」變多，不會讓它「假成功」；cpu50／rocmfix／合併版都是 511／512，所以正面結論不受影響〔判讀〕。naive 的 0／512 我解釋成排法錯（§4.5），但不能排除這個競爭也有份。
8. **沒解釋的差異**（NOT_MEASURED）：
   - 同樣是 Triton，不開 connector（off_triton）比開 connector（cpu50_nostore）慢：doc C＝1 首輪 4.520 vs 4.156 s；attention kernel 3.650 vs 3.277 s。
   - rocmfix 的 seed 3，doc C＝1 首輪 3.078 s（seed 1、2 是 2.796、2.804 s）。
   - naive 的送出時間比 cpu50 少（5.53 vs 7.21 s），但兩者每 block 都是 32 次 copy。
9. **和 D2 不同的地方（偏離）**：每個行程多一個 8K 暖機請求；3 格放在同一個行程裡依序跑（D2 是 doc、chat 分開兩個行程）；profiler 另開行程；鎖改成分組持有。主判準、工作負載、CPU 大小（33.0 GiB）都和 D2 一樣。
10. **D2 的 doc C＝1「回來請求快 2 倍」是雙峰中位數的假象**（§4.1）：不是錯誤，但 D2 §3A 的那一格不能當結論。
11. **NFS 寫入**：`20261010-111538-f1-tier50-nfs` 在 11:15–11:19（UTC）寫了 35.42 GB 到 `/mlsteam/data/tiara/runs/20261010-111538-f1-tier50-nfs/fs`，腳本結束時自己刪掉（目錄已不存在，run 目錄 632 KB）。
12. 沒有產生 `gpucore.*`（run 目錄和 repo 都查過）；`/dev/shm` 沒有留下 `vllm_offload_*`。
13. 無害的警告：`CUDA_VISIBLE_DEVICES on ROCm is deprecated`；`Passing raw prompts to InputProcessor is deprecated`；Triton 第一次跑 `kernel_unified_attention` 時的 JIT 警告。

---

## 附：檔案與 run

- 程式：`code/m9_f1_run.py`（量測、修法 monkeypatch、profiler、KV 比對、copy 合併）、`code/m9_f1_analyze.py`（合併＋判定；最後一次合併的 run＝`20261010-112856-f1-analyze-final`）。
- 結果（`results/m9_followup/`）：`f1_repro.csv`（重現判定）、`f1_decomp.csv`（拆解）、`f1_metrics.csv`（每 cfg×seed×格）、`f1_cells.csv`、`f1_profile.csv`／`f1_profile_summary.csv`、`f1_kvcheck.csv`、`f1_output_match.csv`、`f1_ret_split.csv`、`f1_tier50_cells.csv`／`f1_tier50_summary.csv`、`f1_guard.csv`。
- 原始輸出與 profiler trace（`profile_*.json.gz`）：`/mlsteam/data/tiara/runs/<run_id>/`。
- 正式 run（25 個）：
  - 重現：`20261010-075828-f1-s1-off`、`20261010-085015-f1-s1-cpu50`、`20261010-092506-f1-s2-off`、`20261010-093230-f1-s2-cpu50`、`20261010-094406-f1-s3-off`、`20261010-095124-f1-s3-cpu50`
  - 消融：`20261010-090103-f1-s1-offtri`、`20261010-100255-f1-s2-offtri`、`20261010-104146-f1-s3-offtri`、`20261010-103048-f1-s1-nostore`、`20261010-111129-f1-s1-rocmfixns`
  - 修法：`20261010-091253-f1-s1-rocmfix`、`20261010-101502-f1-s2-rocmfix`、`20261010-105352-f1-s3-rocmfix`、`20261010-112119-f1-s1-rocmfixcoal`、`20261010-110302-f1-s1-rocmnaive`
  - profiler：`20261010-092136-f1-prof-off`、`20261010-092234-f1-prof-cpu50`、`20261010-102417-f1-prof-offtri`、`20261010-102523-f1-prof-rocmfix`
  - KV 比對：`20261010-110725-f1-kvchk-cpu50`、`20261010-110848-f1-kvchk-rocmfix`、`20261010-111008-f1-kvchk-naive`、`20261010-111954-f1-kvchk-coal`
  - NFS：`20261010-111538-f1-tier50-nfs`
  - 不進結果：`20261010-074630-f1-smoke-rocmfix`
- GPU 時間（持有鎖的時間）：約 168 分鐘〔算術：分組 34.9＋37.8＋38.9＋25.2＋12.4＋5.7 分，加 smoke 約 6 分、s1-off 約 7 分〕。另外等鎖約 30 分鐘。
