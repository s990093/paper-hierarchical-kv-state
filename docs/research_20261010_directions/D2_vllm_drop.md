# D2：真實 vLLM 裡「CPU 滿了→丟掉或卡住」常不常發生（H0 實測）

> 完成於 2026-10-10T07:01:52Z。判準（§2）寫於 05:37:21Z，在任何 GPU 執行之前；之後只有「追加」修訂，主判準一字未改。
> 引擎：vLLM 0.28.0（`venv/tiara-v028`）在 ROCm／MI300X 上**跑得起來**，不需要退回 0.19.1。
> 原始碼路徑前綴：`/mlsteam/workspace/src/vllm-v0.28.0/vllm/`。

---

## 0. 一句話結論＋判定

**判定：死路**（D1 的前提「真實系統常常因為 CPU 滿了而把 KV 丟掉」在 vLLM 0.28 的一般設定下不成立）。

- 事先寫好的主指標：`cpu50`、C＝4／16、doc＋chat 共 4 格，**寫入被跳過＝0／133,504 個 block（0.0%）**，門檻是 < 1% → 死路〔實測 20261010-055443-m8-d2-cpu50-doc、20261010-062749-m8-d2-cpu50-chat〕。
- 連壓力組（CPU 只有工作集的 25%）也一樣：12 格卸載組合計 **0／400,512**，`ALLOCATION_FAILURE`＝0〔實測，見 §4.1〕。
- **最重要的理由**：只有 CPU 層時，CPU 裡「不能淘汰」的 block 只有「正在搬」的那些（GPU→CPU 寫入中、CPU→GPU 載入中）。這個量最多約 4 GiB（＝一個 32K prompt），只佔 CPU 的 12–30%〔實測〕。所以 LRU（丟最久沒用的）永遠找得到東西丟，新的 KV 永遠存得進去。CPU 滿了，vLLM 丟的是**舊的**（LRU 淘汰，延後版的正常動作），不是**新的**。
- 只有兩種極端情況才看得到跳過，而且量不大：
  - CPU 小到 4 GiB（≈ 一個 prompt）：跳過 6.9%（C＝4）、0.27%（C＝16）〔實測 20261010-064252-m8-d2-cpu4g-doc〕。
  - CPU 下面接一層很慢的 fs（NFS）：跳過 0.21%〔實測 20261010-065309-m8-d2-tier50-doc〕。
  - 跳過少的原因：vLLM 存不進去時**下一步會重試**（〔程式碼 `distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:1326-1334`〕）。E1、E2 裡有 47%–99.7% 的 block「失敗過、之後才存進去」，真的丟掉的只有請求結束前還沒輪到的那一點。
- **兩個意外發現（不在主判定內，但比「丟資料」更值得看）**：
  1. 開 CPU 卸載會讓**新請求**的 prefill 變慢：doc、C＝1，首輪 TTFT 2.25 s → 4.40 s（1.96 倍）〔實測〕。不是 flush 等待（≈0），也不是 16 token 小塊（改 512 token 仍 4.36 s）。原因 NOT_MEASURED。
  2. 接慢的 fs 層時，**回來的請求會等 fs 把資料搬回來，而不是重算**：最久等了 83 s；同樣 33K 在 C＝1 重算只要約 2.25 s〔實測；比值 37 倍是〔算術〕〕。

---

## 1. 問題是什麼、要證明什麼

- **背景**：S5（寫入當下依位置決定放哪層）在「一次一個請求」的模型裡都被延後版追平。反方審查說：延後版能追上，是因為它永遠有空檔。唯一可能輸的區域是「沒有空檔、寫入排隊、CPU 滿、只好丟資料」。H0 讀原始碼發現：真實系統在 CPU 滿時預設是**丟掉或跳過**，不會等。但**沒有人量過這件事多常發生**。
- **要證明什麼**：在真實 vLLM 0.28 的 CPU 卸載（OffloadingConnector）裡，多個長對話同時跑、會回來、沒有空檔時：
  1. 要存到 CPU 的 KV，有多少因為「CPU 滿又沒東西可丟」而沒存進去（寫入被跳過）？
  2. 回來的請求，有多少 token 從 CPU 載入、多少重算？重算的原因是什麼？
  3. 卸載會不會讓新請求等？
- **為什麼重要**：這是 D1（另一個 agent 在模擬這個區域）的前提。如果在合理負載下幾乎不發生，D1 的前提就很弱。
- 名詞：
  - **卸載（offload）**：GPU 放不下的 KV 先複製一份到 CPU 記憶體，之後回來可以直接搬回去，不用重算。
  - **LRU 淘汰**：CPU 滿了，丟掉最久沒用的那份。
  - **寫入被跳過**：CPU 滿了，而且 CPU 裡每個 block 都在搬（不能丟），這次新的 KV 就不存了。這就是 D1 要找的事件。

---

## 2. 事先寫好的判準（pre-registration）

**寫於 2026-10-10T05:37:21Z（`date -u`），在任何 GPU 執行之前。**

### 2.1 設定（開跑前寫死）

- 引擎：vLLM 0.28.0（`venv/tiara-v028`），同一個 Python 行程內跑（`VLLM_ENABLE_V1_MULTIPROCESSING=0`），自己用 `llm_engine.add_request/step` 迴圈送請求。
- 模型：Llama-3.1-8B-Instruct（`models--unsloth--Llama-3.1-8B-Instruct`），bf16。每 token KV＝128 KiB〔算術：32 層×8 頭×128×2×2 B〕。
- GPU KV：`kv_cache_memory_bytes`＝16 GiB（約 131K token，約 4 個 32K session）。`max_num_seqs`＝16，`max_num_batched_tokens`＝8192，`max_model_len`＝40960，prefix caching 開（預設）。
- 卸載：`OffloadingConnector`＋`CPUOffloadingSpec`，`eviction_policy=lru`，`block_size` 不設（＝GPU block）。
- 工作負載：16 個 session，每個 session 4 輪，輪與輪之間沒有思考時間（沒有空檔）。一個 session 的下一輪排到「待送佇列」最後面，所以兩次回來之間隔著其他 session。
  - **doc**：第 1 輪＝32,768 個隨機 token 的文件＋問題；之後 3 輪各加 256 token 的問題。
  - **chat**：每輪加 8,192 token（8K→16K→24K→32K）。
  - 每輪輸出 32 token（`ignore_eos`），輸出 token 接進下一輪 prompt。
  - 工作集 W＝16 × 約 32K token ≈ 64 GiB〔算術〕。
- 三組卸載：`off`（不卸載）、`cpu25`＝16 GiB（W 的 25%）、`cpu50`＝32 GiB（W 的 50%）。
- 併發 C ∈ {1, 4, 16}（同時在跑的 session 數，closed-loop）。
- 每格用新的隨機 token；格與格之間 `reset_prefix_cache(reset_connector=True)`。
- 1 個 seed。如果主指標落在 0.5%–1% 或 8%–12%（離門檻很近），補 2 個 seed 再判。

### 2.2 怎麼數（用自己腳本裡的 monkeypatch，不改 venv）

- **offered（被提出要存的）**：傳進 `CPUOffloadingManager.prepare_store` 時還不在 CPU 裡的 key，每格去重。
- **accepted（真的拿到 CPU 空間）**：`prepare_store` 回傳的 `keys_to_store` 裡的 key，去重。
- **skipped（寫入時被丟）**＝ offered − accepted。也就是「CPU 滿了、又沒有可淘汰的 block，這段 KV 從頭到尾沒存進去」。
- `prepare_store` 回傳 `None` 的呼叫次數＝vLLM 自己的 `ALLOCATION_FAILURE` 計數。失敗後下一步會重試，所以「失敗過但後來存進去」另外記。
- LRU 淘汰（CPU 滿了丟最舊的）另外記，**不算進主指標**：那是延後版的正常動作（滿了才丟舊的），不是 D1 要找的「寫不進去」。
- 回來的請求：prompt token 拆成「GPU 命中＋CPU 載入＋重算」。重算的原因：第一個 MISS 的 key 是被 LRU 淘汰的、被跳過的、還是根本沒存過。
- 卡住：worker 端 `handle_preemptions` 裡等 `jobs_to_flush` 的時間；scheduler 端 `get_num_new_matched_tokens` 回 `None`（因為還在搬）的次數與延遲；搶占次數；首輪（新請求）TTFT 跟 `off` 比。

### 2.3 判定規則（照任務給的，寫死）

- **主指標**：真實設定＝`cpu50`，C ∈ {4, 16}，doc＋chat 共 4 格。合併計算 skipped ÷ offered（4 格加總後相除）。
  - **< 1%** →「丟資料區很少見」→ D1 前提的判定：**死路**。
  - **≥ 10%** → **有看頭**。
  - **1%–10%** → **可能**。
- `cpu25` 與 C＝1 的格子只當壓力測試與對照，不進主判定。
- **卡住（次要，描述用，不改主判定）**：首輪 TTFT 中位數（同一格）`cpu50` 比 `off` 高 ≥ 5%，而且 flush 等待總時間 ≥ 該格牆鐘時間的 1%，才寫「卸載會卡住新請求」。
- 0.28 在 ROCm 上跑不起來 → 記完整錯誤，改用 0.19.1（`venv/tiara`），判準照用。

### 2.4 修訂 1（2026-10-10T05:38:52Z，仍在任何執行之前，沒有看過任何結果）

- **改什麼**：2.1 原本寫「下一輪排到待送佇列最後面」。改成「下一輪插到待送佇列的**隨機位置**」（每格固定 seed）。
- **為什麼**：16 個 session 輪流（循環）存取、CPU 只放得下其中 4–8 個時，LRU 一定全部 miss（循環存取是 LRU 的最壞情況）。這樣「回來時命中」永遠是 0，CPU 裡也不會有「正在被載入、不能淘汰」的 block。而「載入中的 block 鎖住空間」正是寫入被跳過的機制之一。隨機插隊讓回來的距離有長有短，比較像真實多人使用。
- 判定規則（2.3）完全不變。

### 2.5 修訂 2（2026-10-10T06:09:17Z；**已經看過** off-doc 與 cpu50-doc 三格的結果：skipped＝0）

- 主判準（2.3）**不改**。下面兩組是**額外的探索**，不進主判定，結果另外列。
- **E1（CPU 很小）**：只有 CPU 層，`cpu_bytes_to_use`＝4 GiB（W 的約 6%，GPU KV 的 1/4），doc，C＝4、16。
  - 問題：CPU 小到跟「同時在搬的量」差不多時，跳過會不會出現？用來找門檻在哪裡。
  - 事先寫下的預期〔判讀〕：會出現。因為 cpu50-doc 量到寫入中的 block 最多佔 CPU 的 12%（約 4 GiB），載入中的最多 22%。
- **E2（CPU 下面接一層慢的 fs）**：`TieringOffloadingSpec`＋`secondary_tiers=[{"type":"fs","root_dir":<run 目錄>/fs}]`（NFS）。8 個 session 的 doc，CPU＝這個縮小工作集的 50%（約 16.5 GiB），C＝4。
  - 問題：寫往下一層的 block 會被鎖住、不能淘汰（H0：`tiering/manager.py`）。下一層比 KV 產生速度慢時，跳過會不會變多？這才是 D1 模擬的那個區域（「寫入積壓」）。
  - 判讀規則（描述用，和 2.3 同樣的門檻）：skipped ÷ offered < 1%、1–10%、≥ 10%。
  - 注意：fs 放在 NFS（本機唯一允許放大檔的地方），NFS 寫入約 0.6–0.7 GiB/s（D7 的 `20261010-054826-d7-pc-nfs` 量到 4 GiB 約 5.7–7.1 s），比本機 NVMe 慢，所以 E2 是「下一層很慢」的情況，不代表一般部署。會寫約 30 GiB 到 NFS，可能干擾 D7 同時段的 NFS 量測，跑的時間會寫在 §7。

### 2.6 修訂 3（2026-10-10T06:43:47Z；**已經看過**主格 18 格的結果）

- 主格看到：首輪（新請求）TTFT 開卸載後變慢（doc C＝1：2.25 s → 4.40 s），但 flush 等待≈0、scheduler 延後＝0。照 2.3 的「卡住」規則（要 flush 等待 ≥ 1% 牆鐘時間），**不能**寫「卸載卡住新請求」。這條規則不改。
- **E3（探索，不進任何判定）**：同 cpu50，但 `block_size`＝512（卸載單位從 16 token 變 512 token，搬移次數少 32 倍），doc，C＝1，8 個 session。
  - 問題：首輪變慢是不是「16 token 小塊、搬移次數太多」造成的？
  - 事先寫下的讀法：E3 首輪 TTFT 中位數比 cpu50（16 token）的 4.40 s 低一半以上 → 寫「小塊搬移是主因（〔判讀〕）」；和 4.40 s 差 < 10% → 寫「不是塊大小」；中間 → 寫「部分」。


---

## 3. 做了什麼

### 3.1 先讀原始碼：在哪裡數、怎麼數（任務 1）

| 事件 | vLLM 0.28 的位置〔程式碼〕 | 能不能直接拿來數 |
|:--|:--|:--|
| **寫入被跳過** | `v1/kv_offload/cpu/manager.py:186-192, 199-201`：要的 block 數 > 空的＋可丟的，`prepare_store` 回 `None` | 我用 patch 數「哪些 block 被跳過」（去重） |
| 跳過之後 | `offloading/scheduler.py:1326-1334`：`ALLOCATION_FAILURE`＋1，印 `"Request %s: cannot store chunks"`，然後 `continue`——**不推進進度，下一步會重試** | vLLM 的計數器數的是「呼叫次數」，不是 block 數（`offloading/metrics.py:38, 124-128`） |
| 什麼時候變成永久丟失 | 請求結束那一步是最後一次機會（`scheduler.py:1254-1256` 把 `finished_req_ids` 也掃一次），之後狀態被刪（`scheduler.py:1486-1493`） | 用「offered − accepted」就是永久丟失 |
| 哪些 block 不能丟 | 寫入中 `ref_cnt=−1`（`cpu/manager.py:62-63, 225-227`）；載入中 `ref_cnt>0`（`cpu/manager.py:140-145`）；只有 `ref_cnt==0` 能丟（`cpu/manager.py:247-255, 156-163`） | 每一步取樣「寫入中＋載入中」佔 CPU 的比例 |
| 部分尾段的跳過 | `scheduler.py:1201-1206`：一樣加計數器，但**不印 warning** | 一起數 |
| LRU 淘汰 | `cpu/manager.py:188-209` 的 `evicted_keys` | 另外數，不進主指標 |
| 頻率門檻的跳過（`STORES_SKIPPED`） | `cpu/manager.py:171-174, 320-325`，只有 `store_threshold≥2` 才有；預設 0 | 和本題無關（是「准入」不是「滿了」） |
| 讀取被延後 | 同一請求還有傳輸（`scheduler.py:962-967`）、block 還在寫（`HIT_PENDING`，`scheduler.py:622-631`）、block 正在被別人載入（`scheduler.py:836-860`）→ 回 `None`；scheduler 這一步就跳過該請求（`v1/core/sched/scheduler.py:834-840`） | patch 數次數與延後秒數 |
| worker 同步等待 | `offloading/worker.py:316-317` 的 `self.worker.wait(jobs_to_flush)`；`jobs_to_flush` 來自被搶占的請求、block 被重用（`scheduler.py:1452-1474`） | 量 `handle_preemptions` 的時間 |
| store 什麼時候送出 | 延到下一步開頭（`offloading/worker.py:332-344`），在 `handle_preemptions`／`start_kv_transfers` 裡送（`worker.py:309-314, 319-324`） | 量送出花的時間 |
| GPU→CPU 用什麼搬 | `ops.swap_blocks_batch`（DMA，`cpu/gpu_worker.py:41-45`）；ROCm 7.1+ 用 `hipMemcpyBatchAsync`（`csrc/libtorch_stable/cache_kernels.cu:155-173`） | — |
| Prometheus／log | `vllm:kv_offload_allocation_failure`、`..._cpu_cache_usage_perc`、`..._cpu_cache_write_usage_perc`、`..._store_bytes/time` 等（`offloading/metrics.py:22-38`；`cpu/manager.py:293-327`）；`disable_log_stats=False` 時 log 每 10 s 印一行 `KV Transfer metrics` | 拿來交叉核對 patch |

**決定**：用自己腳本裡的 monkeypatch（不改 venv），包住 `CPUOffloadingManager.prepare_store/complete_store/lookup/prepare_load`、`OffloadingConnectorScheduler.get_num_new_matched_tokens/update_state_after_alloc`、`OffloadingConnectorWorker.handle_preemptions/start_kv_transfers`、`Scheduler._preempt_request`。引擎跑在同一個行程（`VLLM_ENABLE_V1_MULTIPROCESSING=0`），patch 才有效。

**交叉核對**（照 CLAUDE.md 規則 6 的精神）：E1 裡 patch 數到 1,855 次失敗呼叫；同一個 run 的 log 有 1,855 行 `cannot store chunks`；vLLM 自己 log 的 `kv_offload_allocation_failure` 加總 1,853（最後一個 10 s 區間沒印出來）〔實測 20261010-064252-m8-d2-cpu4g-doc〕。E2：patch 102、log 101〔實測 20261010-065309-m8-d2-tier50-doc〕，差的 1 次可能是不印 warning 的部分尾段跳過〔判讀〕。

### 3.2 實驗（`code/m8_vllm_drop.py`）

- 設定照 §2.1（加修訂 1）。每格：16 個 session、每個 4 輪、輪間沒有空檔、下一輪插到待送佇列的隨機位置。
- 每個 run 用 `m7run` 包、用 `m7_guard_run.py`（GpuWatcher）包。跑之前都 `gpu_guard.py --check 0`＝CLEAN。
- 主格：{off, cpu50, cpu25} × {doc, chat} × C∈{1, 4, 16}＝18 格，6 個 run。探索：E1、E2、E3（§2.5、§2.6）。
- 最後用 `python code/m8_vllm_drop.py collect ...`（run `20261010-070054-m8-d2-collect`）把各 run 的 CSV 合併到 `results/m8_directions/`，同時檢查 GpuWatcher。

| run_id | 內容 | 牆鐘 |
|:--|:--|:--|
| 20261010-054520-m8-d2-off-doc | off，doc，C＝1/4/16 | 05:45–05:54 |
| 20261010-055443-m8-d2-cpu50-doc | cpu50（33.0 GiB），doc | 05:54–06:07 |
| 20261010-060743-m8-d2-cpu25-doc | cpu25（16.5 GiB），doc | 06:07–06:21 |
| 20261010-062141-m8-d2-off-chat | off，chat | 06:21–06:27 |
| 20261010-062749-m8-d2-cpu50-chat | cpu50，chat | 06:27–06:34 |
| 20261010-063439-m8-d2-cpu25-chat | cpu25，chat | 06:34–06:42 |
| 20261010-064252-m8-d2-cpu4g-doc | E1：CPU 4 GiB，doc，C＝4/16 | 06:42–06:53 |
| 20261010-065309-m8-d2-tier50-doc | E2：Tiering＋fs（NFS），8 session，C＝4 | 06:53–06:57 |
| 20261010-065733-m8-d2-cpu50bs512-doc | E3：block_size 512，8 session，C＝1 | 06:57–07:00 |
| 20261010-054033-m8-d2-smoke | 試跑（4 session），**不進結果** | 05:40–05:44 |

GPU 總用量約 80 分鐘（每個 run ≤ 13 分鐘）〔算術，由上表〕。

---

## 4. 結果

### 4.1 主判定：寫入被跳過（事先寫好的指標）

| 組 | 格數 | offered（block） | skipped | 比例 | 失敗呼叫 | LRU 淘汰（次） |
|:--|--:|--:|--:|--:|--:|--:|
| **cpu50，C＝4/16（主判定）** | 4 | 133,504 | **0** | **0.0%** | 0 | 263,050 |
| cpu50 全部 | 6 | 200,256 | 0 | 0.0% | 0 | 331,614 |
| cpu25 全部（壓力） | 6 | 200,256 | 0 | 0.0% | 0 | 481,200 |
| E1：CPU 4 GiB，doc | 2 | 67,776 | 2,442 | 3.6% | 1,855 | 160,976 |
| E2：Tiering＋fs（NFS） | 1 | 16,944 | 36 | 0.21% | 102 | 21,130 |
| E3：cpu50＋block 512 | 1 | 528 | 0 | 0.0% | 0 | 1,103 |

〔實測，`results/m8_directions/d2_summary.csv`、`d2_cells.csv`〕。一個 block＝16 token（E3 是 512 token）。

- **主判定：0.0% < 1% → 死路。** 不在 0.5–1% 或 8–12% 附近，照 §2.1 不需要補 seed。
- CPU 裡「不能丟」的比例（寫入中＋載入中）最高：cpu50 21.8%（doc C＝4），cpu25 30.3%（chat C＝16）〔實測 `max_nonevict_frac`〕。寫入中最多 2,048 block＝32K token＝4 GiB〔算術：0.1212 × 16,896〕，三種併發都一樣。
- 搶占＝0、scheduler 延後＝0（12 格合計）〔實測〕。C＝16 時請求是在等待佇列排隊，不是被搶占。

**E1 細節**（CPU 2,048 block＝4 GiB）：

| C | 跳過 | 失敗過、後來才存進去 | 不能丟的比例最高 | 回來請求第一個 miss 的原因（48 個） |
|:--|--:|--:|--:|:--|
| 4 | 2,352／33,888＝6.9% | 27,528（81%） | 100% | LRU 淘汰 38、被跳過 10 |
| 16 | 90／33,888＝0.27% | 33,798（99.7%） | 100% | LRU 淘汰 43、被跳過 5 |

- 〔判讀〕C＝16 跳過反而少：請求排隊久、活得久，重試的機會多。所以「跳過多少」主要看請求在 KV 產生之後還活多少步，不只看 CPU 多滿。我的負載每輪輸出 32 token（≥32 步重試機會）；輸出更短的工作負載，跳過會比這裡多。NOT_MEASURED。

**E2 細節**（CPU 8,448 block＝16.5 GiB，fs 在 NFS）：
- 跳過 36／16,944＝0.21%；失敗過、後來才存進去 7,882（47%）；不能丟的比例最高 100%（寫入中最高 96%）〔實測〕。
- 寫進 fs 的資料 35.5 GB（`FS_TIER_BYTES`＝35,458,647,792）〔實測〕。
- 〔判讀〕E2 確實出現了 D1 想要的狀態：CPU 全被「寫往下一層」鎖住。但 vLLM 不丟，而是**下一步再試**，所以最後真的丟掉的很少。

### 4.2 回來的請求：從哪裡拿到 KV

「上一輪算過的 token」裡，由 GPU 命中、CPU 載入、重算各佔多少〔實測，`d2_ttft.csv`；比例是〔算術〕〕：

| 工作負載 | C | off：GPU／重算 | cpu50：GPU／CPU／重算 | cpu25：GPU／CPU／重算 |
|:--|:--|:--|:--|:--|
| doc | 1 | 47% ／ 53% | 47% ／ 26% ／ 27% | 47% ／ 0.7% ／ 52% |
| doc | 4 | 17% ／ 83% | 15% ／ 28% ／ 56% | 15% ／ 1.7% ／ 83% |
| doc | 16 | 0% ／ 100% | 0% ／ 0% ／ 100% | 0% ／ 0% ／ 100% |
| chat | 1 | 58% ／ 42% | 58% ／ 24% ／ 18% | 58% ／ 0.2% ／ 42% |
| chat | 4 | 37% ／ 63% | 45% ／ 41% ／ 15% | 37% ／ 0.2% ／ 63% |
| chat | 16 | 1% ／ 99% | 2% ／ 24% ／ 74% | 2% ／ 1.2% ／ 97% |

- 重算的原因（回來請求第一個 CPU miss）：主格裡只有「被 LRU 淘汰」和「到了新加的部分」兩種，**「被跳過」＝0 次**〔實測 `ret_miss_reason`〕。
- 〔判讀〕CPU 沒存到的原因是**容量**（LRU 把它丟了），不是「CPU 忙到存不進去」。cpu25 幾乎全被 LRU 丟光（命中 ≤1.7%）；doc C＝16 連 cpu50 也是 0%，因為 16 個 32K 同時在跑，回來之前早被擠掉。

### 4.3 卸載會不會讓請求等

**新請求（首輪）TTFT 中位數**〔實測 `d2_ttft.csv`；倍數是〔算術〕〕：

| 工作負載 | C | off | cpu50 | cpu25 | E3（cpu50，512 token 塊） |
|:--|:--|--:|--:|--:|--:|
| doc（33K） | 1 | 2.25 s | 4.40 s（1.96×） | 4.38 s（1.95×） | 4.36 s（1.94×） |
| doc | 4 | 6.95 s | 9.30 s（1.34×） | 9.98 s（1.44×） | — |
| doc | 16 | 26.4 s | 38.9 s（1.48×） | 38.8 s（1.47×） | — |
| chat（8K） | 1 | 0.320 s | 0.409 s（1.28×） | 0.409 s（1.28×） | — |
| chat | 4 | 0.919 s | 0.835 s（0.91×） | 1.044 s（1.14×） | — |
| chat | 16 | 5.25 s | 3.84 s（0.73×） | 4.37 s（0.83×） | — |

- C＝1 最乾淨（沒有排隊）：16 個首輪的範圍很窄（off 2.219–2.259 s；cpu50 4.363–4.426 s）〔實測〕。
- 和舊資料對得上：C6（沒有 patch、server 模式、開卸載）32K 冷啟動 4.58 s、8K 0.44 s〔實測 20261008-141324-m7-c6-vllm〕。所以變慢不是我的 patch 造成的〔判讀〕。
- **但照事先寫好的規則，不能寫「卸載卡住新請求」**：flush 等待合計只有 0.079 s（12 格），遠低於 1% 牆鐘時間；scheduler 延後＝0〔實測〕。變慢**不是**從我事先猜的那條路（等 flush）來的。
- 送出搬移的 CPU 時間（`handle_preemptions`＋`start_kv_transfers`）：doc C＝1 cpu50 合計 6.2 s／164 s 牆鐘〔實測〕。16 個首輪多出來的時間約 16 × 2.15 ≈ 34 s〔算術〕，送出時間最多解釋其中約 1/5〔算術〕。
- E3：改成 512 token 的塊（搬移次數少 32 倍），首輪仍 4.36 s，和 4.40 s 差 < 10% → 照 §2.6 寫「**不是塊大小**」〔實測 20261010-065733-m8-d2-cpu50bs512-doc〕。
- 變慢的機制：NOT_MEASURED。〔判讀，未查證〕可能是 GPU→CPU 的 DMA 和 prefill 在 MI300X 上搶某種資源。
- C＝16 doc：cpu50 的 CPU 命中是 0%，卸載只有成本：回來請求 TTFT 43.7 s → 67.0 s（1.53×）〔實測〕。

**E2：回來的請求等慢的那一層**
- 回來請求 TTFT：中位數 0.71 s，p90 56.8 s，最大 84.3 s；最長的 scheduler 延後 83.1 s〔實測 20261010-065309-m8-d2-tier50-doc〕。
- 這些請求最後幾乎全部命中（重算只剩 0.07%），也就是 vLLM **選擇等 fs 搬回來，而不是重算**。
- 同樣 33K 的重算：off C＝1 首輪 2.25 s、off C＝4 回來 6.75 s〔實測〕。等的時間是重算的 12–37 倍〔算術〕。
- 限制：fs 在 NFS（約 0.6–0.7 GiB/s 寫，D7 量的），而且 35.5 GB 的寫入和讀回排在一起。本機 NVMe 的情況 NOT_MEASURED。只有 1 個 run。

---

## 5. 延後版／更簡單的做法＋前作

- **vLLM 自己就已經是「延後版」**：存不進去時不丟、下一步再試（`scheduler.py:1326-1334`）。E1、E2 裡 47–99.7% 的 block 是靠這個重試才存進去的。D1 如果模擬「存不進去就立刻丟」，會高估丟資料的量。D1 的模型應該加上「每一步重試，直到請求結束」。
- 就算真的跳過了，更簡單的修法都很便宜：
  - CPU 給大一點：CPU ≥ 約 4 GiB（一個 32K prompt）就沒有跳過（cpu25＝16.5 GiB 已經 0）。DRAM 便宜。
  - 准入門檻：`store_threshold`（`cpu/manager.py:171-174`），只存出現過兩次以上的 block。
  - LMCache 的 `force_store_wait=True`：等，不丟（H0 §0 第 2 點）。
  - Dynamo KVBM 的頻率過濾（H0 表格）。
- **前作**：
  - H0（`docs/research_20261009_explore/H0_offload_blocking.md`）只讀原始碼，這份是它 §5 的實測版（但改用 0.28，因為 0.28 有 `ALLOCATION_FAILURE` 計數器，也有 fs 層）。
  - 量測 vLLM 0.28 卸載「跳過率」的公開資料：沒查到〔未查證〕。
  - E2 的「回來時等慢層，而不是重算」就是 Cake／κ 的論點在真實系統裡的一個例子；Gill FAST'08 的「讀取要等 demotion」也是同類（12 §8 第 7 點）。

---

## 6. 如果要繼續

D2 本身（丟資料頻率）到此為止：死路。值得往下追的是兩個副產品：

1. **讀取端「等慢層 vs 重算」（E2）**——〔判讀〕這可能比寫入時放置更有看頭：
   - 現象：vLLM 0.28 tiering 在 fs 慢的時候，回來的請求等了最多 83 s，重算只要 2–7 s。
   - 下一步：fs 放本機 NVMe（不是 NFS）重跑 E2，量「等多久」的分布；再數一下多少比例的回來請求「等 > 重算」。這就是 κ＞1 的真實例子，Cake 式「前段重算、後段載入」可以直接接上。
   - 需要：本機 NVMe 路徑（規則只允許大檔放 NFS 的 run 目錄，要先問）；約 15 分鐘 GPU。
2. **開卸載讓 prefill 變慢 2 倍（ROCm）**：
   - 下一步：用 profiler 看 prefill 時 DMA 和 kernel 的時間重疊；試 `HSA_ENABLE_SDMA` 開／關。
   - 這對所有「卸載便宜」的假設都有影響（寫穿的成本不是 0）。D1、D3 的模型如果把 GPU→CPU 寫入當成免費，應該加上這個成本。
3. **如果 D1 還要用「丟資料」區域**：只能用「CPU ≈ 一個 prompt」（E1）或「下面接一層慢 fs」（E2）的設定，而且要模擬 vLLM 的「每步重試」。在這兩個設定下，永久丟失也只有 0.2–7%。

---

## 7. 失敗與異常

1. **GpuWatcher 誤報**：9 個正式 run 裡 7 個標 `CONTAMINATED`，外來者全是 `pid −1`，第一次出現都在最後一格寫完之後（2.0–34.2 s），峰值約 34 GiB（＝我自己的模型＋KV）。這是任務說明裡的已知誤報（自己的行程結束時 amd-smi 回報 pid −1）。`collect` 用寫死的規則判斷：「全部外來者都是 pid −1，而且第一次出現 ≥ 最後一格的時間」才算誤報（`code/m8_vllm_drop.py` 的 `guard_status`）。這條規則是看過試跑和前兩個 run 的旗標之後才寫的。逐 run 判斷在 `results/m8_directions/d2_guard.csv`。試跑 `20261010-054033-m8-d2-smoke` 也是同樣的誤報，但試跑本來就不進結果。
2. **我的批次迴圈印錯 exit code**：`echo "$(date) ... rc=$?"` 的 `$?` 被 `$(date)` 蓋掉，所以畫面上都印 `rc=0`。實際 exit code 以各 run 目錄的 `exit_code` 為準（誤報的 run 是 3）。不影響資料。
3. **和事先設定的差異**：
   - CPU 大小：§2.1 寫 16／32 GiB；腳本用「每 session 33K token」估工作集，實際是 16.5／33.0 GiB（8,448／16,896 block）。實際工作集 doc 66.25 GiB、chat 64.25 GiB〔算術〕，所以 cpu50 是工作集的 49.8%／51.4%。
   - E3 用 8 個 session，所以它的「cpu50」只有 16.5 GiB（264 個 512-token 塊）。首輪 TTFT 和 CPU 大小無關（cpu25 和 cpu50 首輪幾乎一樣），所以不影響 §4.3 的讀法〔判讀〕。
   - E2、E3 是 8 個 session，`d2_ttft.csv` 裡它們的 `first_ttft_ratio_vs_off` 是拿 16 個 session 的 off 當分母。只有 C＝1 的首輪比較乾淨。
4. **E2 寫了 35.5 GB 到 NFS**：06:54–06:56（UTC），目錄 `/mlsteam/data/tiara/runs/20261010-065309-m8-d2-tier50-doc/fs`，跑完腳本自己刪掉了（run 目錄現在 544 KB）。這段時間 runs 目錄裡沒有其他 agent 的 run；D7 的 NFS 量測（`20261010-054826-d7-pc-nfs`）在 05:48–05:51，沒有重疊。D7 是否有不經 `m7run` 的 NFS 量測：不知道。
5. **C＝1 等載入時引擎空轉**：同一行程模式下，回來的請求在等 CPU→GPU 載入時，`step()` 會空轉很多次（cpu50 doc C＝1：164 s 裡 92,659 步）。只影響步數，不影響 TTFT 量法。
6. 無害的警告：`CUDA_VISIBLE_DEVICES on ROCm is deprecated`；`Passing raw prompts to InputProcessor is deprecated`；結束時 `destroy_process_group() was not called`。
7. `/dev/shm` 的 `vllm_offload_*.mmap`：腳本結束時用自己的 engine_id 刪掉，現在 `/dev/shm` 沒有殘留。沒有產生 `gpucore.*`。
8. 主指標是 1 個 seed。因為是 0.0%，離門檻很遠，照事先規則不補 seed。TTFT 的比較也只有 1 個 seed（但 C＝1 的 16 個首輪範圍很窄）。

---

## 附：檔案

- 程式：`code/m8_vllm_drop.py`（量測＋`collect` 合併）
- 結果：`results/m8_directions/d2_summary.csv`（判定用）、`d2_cells.csv`（每格）、`d2_ttft.csv`（TTFT 與命中組成）、`d2_requests.csv`（每個請求）、`d2_guard.csv`（GpuWatcher 判斷）
- 原始輸出：`/mlsteam/data/tiara/runs/<run_id>/`（`d2_steps.csv` 是每 5 步的 CPU 狀態取樣，只放在 run 目錄）
