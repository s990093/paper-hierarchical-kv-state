# H0：真實系統裡，KV 搬移會不會擋住請求（讀原始碼）

**日期**：2026-10-09
**上游**：[10_breakthrough_plan.md](../phase1_20261008/10_breakthrough_plan.md) §4 H0、§3E；[11_round1_plan.md](../phase1_20261008/11_round1_plan.md) H0 判準；[08_ablation_plan.md](../phase1_20261008/08_ablation_plan.md) §3（hold 的定義）；[09_ablation_report.md](../phase1_20261008/09_ablation_report.md) §5 第 2 點
**方法**：只讀原始碼，沒有安裝、沒有執行任何 clone 下來的程式，也沒有跑 GPU。每個結論後面都附 `檔名:行號`。標〔判讀〕的是我從程式碼推出來的，程式碼本身沒有直接寫出來。標〔算術〕的是我自己算的。
**分工**：vLLM 由我直接讀。LMCache、SGLang、Dynamo 先由 3 個唯讀 subagent 找出位置，我再打開原始碼逐行核對關鍵行號（核對過的行號都列在表裡）。
**GPU 實測**：NOT_MEASURED。§5 只寫實驗設計。

---

## 0. 結論（先講）

1. **CPU 層滿了時，看的五個實作預設都是「丟掉（不存）」，不會「等 SSD 寫完」。** 五個是 vLLM 0.19.1、vLLM 0.28、LMCache、SGLang HiCache、Dynamo KVBM。正在寫往下一層的資料確實占著 CPU 空間，這一點和 hold 一樣。但新的 KV 放不進去時，系統的做法是不存或截斷，請求照常往下跑。
2. **有三個特定設定會等，和 hold 最像：**
   - **LMCache 的 `force_store_wait=True`**（非預設）：CPU 滿，而且所有 chunk 都在寫 disk 或被讀取、無法淘汰時，store 會在推論執行緒上每 0.1 s 重試一次，一直到有空間為止。只有這一種情況和 hold 的定義（等 SSD 寫完才釋放 CPU）完全同構。
   - **Dynamo KVBM「只設 disk、不設 CPU cache」的組態**：GPU→SSD 直寫（G1→G3）。GPU block 要等 SSD 寫完才釋放，所以 GPU 滿時，新請求等的是 SSD。這是「GPU 上的 KV 要先卸載出去」那一半，而且卸載的對象是 SSD。
   - **SGLang 的 `--hicache-write-policy write_back`**（非預設）：GPU 要騰空間時，排程執行緒用 `synchronize()` 等 GPU→host 的拷貝做完，才釋放 GPU。等的是 PCIe 那一段，不是 SSD。
3. **預設設定下，真實系統只會等「GPU→CPU 那一段」，而且有條件：**
   - vLLM 0.19.1：已結束請求的 GPU block，要等它的 GPU→CPU 拷貝做完才釋放。這時如果 GPU 不夠，新請求就排不進去。被搶占（preempt）時，worker 會同步等拷貝做完。
   - vLLM 0.28：GPU block 被新請求重用時，worker 會同步等舊的拷貝做完。
   - LMCache（經典 connector）：每一步的 GPU→CPU 拷貝都是同步的，滿不滿都一樣。
   - Dynamo KVBM：`request_finished` 一律回傳 True，GPU block 要等 GPU→CPU 寫完才釋放。這些寫入是一個請求接一個請求排隊做的，所以前面的請求寫得慢，後面的也會晚釋放。
4. **對 11 的 H0 判準的判定**：不是「三個系統都不會等」，所以照字面，hold 有現實意義。但要加兩個限定：
   - 預設設定下（有 CPU 層時），沒有系統等 SSD。等 SSD 的只有兩種：LMCache `force_store_wait=True`（非預設），以及 Dynamo 只設 disk 的組態（等的是 GPU→SSD，沒有 CPU 層）。
   - 預設設定下比較接近真實系統的模型是第三種：**「寫入中的資料占空間，放不下就丟掉，回來時重算」**。harness 目前沒有這個模型。08 的 hold 結果應該標成「對應同步實作（LMCache `force_store_wait=True`；Dynamo 只設 disk；SGLang `write_back` 在 GPU→CPU 那一段）」。

---

## 1. 總表

縮寫：
- 「下一層」：CPU（主機記憶體）再往下一層（SSD、檔案系統、遠端）。
- 「預設」：不改任何設定時的行為。

| 系統 | 版本／commit | CPU 滿時的動作 | 寫入下一層：同步或非同步 | 新請求會不會等（條件） | 水位線／背景搬 | 證據（檔名:行號） |
|:--|:--|:--|:--|:--|:--|:--|
| **vLLM 0.19.1**<br>`OffloadingConnector`＋`CPUOffloadingSpec` | tag `v0.19.1`<br>commit `b1388b1f`（2026-04-17）<br>本機 `src/vllm`，也是 `venv/tiara` 裝的版本 | **丟掉**：用 LRU 淘汰 `ref_cnt==0` 的 block，直接放回 free list，不寫到任何地方。<br>如果正在寫或正在讀的 block 太多、湊不出足夠空間，這次的 GPU→CPU 卸載就**跳過**，只記一條 warning | **沒有下一層**：內建只有 GPU↔CPU。<br>GPU→CPU 是非同步的：每次搬移用一條獨立的 CUDA stream，下一步一開始才送出，用 event 輪詢有沒有做完 | **CPU 滿：不等**（丟掉或跳過）。<br>**GPU 層：有條件會等**：<br>(a) 結束的請求如果還有 GPU→CPU 拷貝沒做完，GPU block 先不釋放。這時 GPU 不夠，`allocate_slots` 拿不到 block，新請求就留在等待佇列。<br>(b) 被搶占的請求：worker 在 forward 之前用 `event.synchronize()` 同步等它的拷貝。<br>(c) 這一步只要有搶占，就不排新的等待請求。<br>讀取端：要的 block 正在被另一個請求載入時，延後；還在寫入中的 block 算 miss，改成重算 | **OffloadingConnector：沒有**。GPU→CPU 是「算完一個 block 就寫」，等於寫穿。<br>另一個 `SimpleCPUOffloadConnector` 的 lazy 模式有 GPU 端水位線：保留「2×一步的 token 數」那麼多 GPU block 可以直接回收，背景先把快被淘汰的 GPU block 拷到 CPU | `v1/kv_offload/cpu/manager.py:136-148, 143-145, 78-79`<br>`v1/kv_offload/cpu/policies/lru.py:31-46`<br>`distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:225-230, 194-258, 309-333, 126-140`<br>`v1/kv_offload/factory.py:56-58`<br>`v1/kv_offload/cpu/spec.py:105-106`<br>`v1/kv_offload/worker/cpu_gpu.py:178-211, 227-229, 249-253`<br>`.../offloading/worker.py:299-332`<br>`v1/core/sched/scheduler.py:564, 746-764, 1823-1832, 2135-2138`<br>`v1/worker/gpu_model_runner.py:3805-3808`<br>`v1/kv_offload/cpu/manager.py:94-95`<br>`v1/simple_kv_offload/manager.py:186-201, 375-441` |
| **vLLM 0.28.0**（補充）<br>`TieringOffloadingSpec`＋`fs` 層 | tag `v0.28.0`<br>commit `2cf0a691`（2026-08-24）<br>本機 `src/vllm-v0.28.0`，`venv/tiara-v028`。C6 用的就是這版 | **跳過**：正在寫往 fs 的 CPU block 被 `ref_cnt` 鎖住、不能淘汰。湊不出空間時，這次的 GPU→CPU 卸載跳過，記一條 warning，`ALLOCATION_FAILURE` 計數加一 | **寫穿＋非同步**：寫進 CPU 後，對**所有**下一層各送一個寫入工作。<br>fs 層的 `submit_store` 立刻返回，交給 16 條寫入執行緒，佇列沒有上限。<br>能用 O_DIRECT 就用，不能就退回 buffered I/O | **CPU 滿：不等**（跳過）。<br>**GPU 層：有條件會等**：GPU block 被新請求重用時，如果上面還有沒做完的拷貝，worker 在 forward 之前同步 `wait`。<br>讀取端：要的 block 還在寫入中（`HIT_PENDING`），請求延後；從 SSD 搬回 CPU 時 CPU 已滿，算 miss | **沒有**（grep 不到 watermark） | `v1/kv_offload/tiering/manager.py:10-12, 581-584, 629-664, 680-689, 361-362, 403-404`<br>`v1/kv_offload/cpu/manager.py:186-201, 140-144, 125-126`<br>`.../offloading/scheduler.py:1326-1334, 1201-1206, 555-560, 1462-1474, 1602-1641, 622-631, 828-833`<br>`.../offloading/worker.py:292-317`<br>`v1/kv_offload/tiering/fs/manager.py:94, 115-116, 185-195, 217-229`<br>`v1/kv_offload/tiering/fs/thread_pool.py:77-78, 119-131` |
| **LMCache**<br>經典 in-process connector | commit `6448b447`（2026-10-09） | **丟掉**：LRU 只挑可以淘汰的（`ref_count==1` 而且沒被 pin），直接釋放，**不寫 disk**（程式裡有 `TODO: add write-back logic here`）。<br>找不到可淘汰的時候：<br>• 預設（`force_store_wait=False`）：這次 store 截斷，只存前面幾個 chunk，記一條 warning<br>• `force_store_wait=True`：每 0.1 s 重試，**沒有逾時** | **寫穿＋非同步**：每次 store 同時送到所有後端。<br>disk 寫入交給 asyncio 執行緒池，預設 4 條，佇列沒有上限。<br>寫完之前，CPU 那份的 ref 一直多 1，**不能淘汰**。<br>O_DIRECT 預設關 | **預設：CPU 滿不等**（截斷）。<br>**`force_store_wait=True`：會等**，就在推論執行緒上（store 是同步的，在 `wait_for_save` 裡做）。<br>**每一步的 GPU→CPU 拷貝都是同步的**（每個 chunk 做完就 `store_stream.synchronize()`），滿不滿都一樣。<br>讀取端：同步載入時要從 disk 讀，會 busy-loop 等 CPU 空間 | **經典路徑：沒有**。<br>另一個多行程（MP）模式有：L1 用到 80% 就在背景淘汰 20%，每 1 秒檢查一次。但淘汰是 DISCARD（直接丟），不是搬到下一層 | `lmcache/v1/storage_backend/local_cpu_backend.py:644-746`（715-732 是 sleep 迴圈）<br>`lmcache/v1/cache_engine.py:510-525, 706-719`<br>`lmcache/v1/memory_management.py:923-928`<br>`lmcache/v1/storage_backend/local_disk_backend.py:426-437, 699-712, 167-176, 386-418, 765`<br>`lmcache/v1/storage_backend/storage_manager.py:386-435, 478`<br>`lmcache/v1/gpu_connector/gpu_connectors.py:404-410`<br>`lmcache/integration/vllm/vllm_v1_adapter.py:1227-1236, 1939`<br>`lmcache/v1/distributed/config.py:474-478`<br>`lmcache/v1/distributed/storage_controllers/eviction_controller.py:160-198` |
| **SGLang HiCache**<br>L1＝GPU、L2＝host、L3＝storage | commit `5cbf9498`（2026-10-09） | **不存或丟掉，不等**。<br>寫 host 之前，空間不夠就先淘汰 host 葉節點。只淘汰已經下放到 host、而且沒被鎖的；正在寫 L3 的有 `host_lock_ref`，跳過。<br>還是不夠：<br>• `write_through`：這次備份放棄，節點只留在 GPU；之後 GPU 淘汰它時直接刪掉<br>• `write_back`：把那棵子樹丟掉 | **L1→L2**：用獨立的 CUDA stream，非同步，事件輪詢。<br>**L2→L3**：一條背景執行緒，`Queue()` 沒有上限，沒有背壓 | **預設（`write_through`）：不等**。正在拷貝的節點被鎖在 GPU，GPU 淘汰時跳過它。<br>**`write_back`：會等**。GPU 要騰空間時，先送 D2H，再對所有還沒做完的寫入 `finish_event.synchronize()`，**排程執行緒被擋住**。<br>**L2→L3 在任何設定下都不等** | **沒有水位線**。<br>`write_through` 本身就是「插入時就寫 host」。<br>`write_through_selective`：第 2 次命中才寫。<br>file 後端的 L3 有容量上限，滿了先刪到 90% | 參數：`arg_groups/fields/memory.py:118-124`（選項確實是 write_back、write_through、write_through_selective，預設 write_through）<br>`mem_cache/unified_radix_cache.py:528-530, 966-991, 770-774, 3357-3369, 1674-1692, 1640-1646, 1919-1950`<br>`mem_cache/unified_cache/unified_tree_core.py:1067-1085, 1726-1759, 2252-2271`<br>`mem_cache/pool_host/base.py:532-537`<br>`managers/cache_controller.py:465-469` |
| **NVIDIA Dynamo KVBM**<br>G1＝GPU、G2＝host、G3＝disk。<br>vLLM connector 用的是 v1 `lib/llm/src/block_manager` | commit `6ee08ec2`（2026-10-09） | **丟掉**：G2 配置先拿沒用過的 block，沒有就拿 inactive 裡排序最前面的（同優先級內最久沒用的）直接 `reset()`，不寫 disk。<br>連 inactive 都不夠就立刻回傳錯誤，不等。<br>G1→G2 在 host 配置失敗時，這次卸載放棄（送一個空的、立即完成的傳輸，讓 worker 不會卡住） | **寫穿＋非同步**：block 一註冊進 G2，就排進 G2→G3 的卸載佇列。channel 沒有上限。<br>每一層有自己的背景 worker，最多 4 個同時傳輸；背壓只在 worker 內部。<br>G3 滿了就略過。<br>**預設有頻率過濾**：同一個 block 在 G2 註冊兩次（衰減窗 600 s）才寫 disk | **G2 滿：不等**（丟掉或放棄卸載）。<br>**GPU 層：有條件會等**：`request_finished` 一律回傳 True，GPU block 要等這個請求的 G1→G2 寫完才釋放。G1→G2 由單一工作依序處理，請求之間排隊。GPU 滿時，新請求等的就是這些寫入。<br>**只設 disk 的組態**：卸載走 G1→G3，GPU block 要等 **SSD** 寫完才釋放。<br>G2→G3 不綁請求，不會擋住 GPU block | **沒有**（grep 不到淘汰用的 watermark）。<br>G2→G3 是寫穿，加頻率過濾 | `lib/llm/src/block_manager/pool/managed/inactive.rs:363-397`<br>`.../pool/managed/state.rs:140-156, 265-267`<br>`lib/llm/src/block_manager/offload.rs:73-81, 143-154, 439-476`<br>`.../offload/pending.rs:223, 238-245`<br>`.../offload/filter.rs:106-125`<br>`lib/bindings/kvbm/src/block_manager.rs:58-75`<br>`lib/bindings/kvbm/src/block_manager/vllm/connector/leader.rs:600-626`<br>`.../connector/leader/slot.rs:1536-1580, 1655-1680, 1714-1719, 1802`<br>`.../connector/worker.rs:412-417`<br>`lib/llm/src/block_manager/config.rs:280-307` |

路徑前綴：
- vLLM 0.19.1：`/mlsteam/workspace/src/vllm/vllm/`
- vLLM 0.28.0：`/mlsteam/workspace/src/vllm-v0.28.0/vllm/`
- LMCache：`/mlsteam/data/tiara/runs/20261009-164921-h0-src/LMCache/`
- SGLang：`/mlsteam/data/tiara/runs/20261009-164921-h0-src/sglang/python/sglang/srt/`
- Dynamo：`/mlsteam/data/tiara/runs/20261009-164921-h0-src/dynamo/`

---

## 2. 對 11_round1_plan.md 的 H0 判準的判定

11 的判準：
- 三個系統都不會等 → hold 只剩理論意義；08 的 hold 結果降級成「如果實作是同步的才成立」。
- 有任何一個會等 → hold 有現實意義。

**判定：不是「三個系統都不會等」。下面這些系統在這些條件下會等：**

| # | 系統與條件 | 等什麼 | 擋住誰 | 預設嗎 | 和 hold 像不像 |
|:--|:--|:--|:--|:--|:--|
| W1 | **LMCache**，`extra_config.force_store_wait=True`，CPU 滿，而且所有 chunk 都不能淘汰（還在寫 disk，或被 pin、被讀） | 等到有 chunk 能淘汰，例如 disk 寫完、ref 降回 1。每 0.1 s 重試，**沒有逾時** | 推論執行緒，也就是整個 forward step。store 在 `wait_for_save` 裡同步做 | **否**，預設 False（`cache_engine.py:513-515`） | **最像**：唯一「等 SSD 寫完才有 CPU 空間」的路徑（`local_cpu_backend.py:715-732`；`local_disk_backend.py:426, 712`；`memory_management.py:923-928`） |
| W2 | **SGLang HiCache**，`--hicache-write-policy write_back`，GPU 要騰空間 | 等所有還沒做完的 GPU→host 拷貝（`finish_event.synchronize()`） | 排程執行緒 | **否**，預設 `write_through`（`memory.py:118-124`） | 像「GPU 上的 KV 要先卸載出去」那一半，但等的是 PCIe，**不是 SSD**（`unified_radix_cache.py:966-977, 3357-3369`） |
| W3 | **vLLM 0.19.1**，請求結束時還有 GPU→CPU 拷貝沒做完，而且 GPU block 不夠 | 等拷貝做完，GPU block 才釋放 | 新請求留在等待佇列（不是忙等） | **是**（OffloadingConnector 的固定行為） | 同 W2，PCIe 那一段（`offloading/scheduler.py:332-333`；`sched/scheduler.py:746-764, 1830-1832, 2135-2138`） |
| W4 | **vLLM 0.19.1**，發生搶占 | 被搶占的請求的拷貝，`event.synchronize()` | 那一步的 forward；那一步也不排新請求 | 是 | PCIe 那一段（`offloading/worker.py:305-308`；`cpu_gpu.py:249-253`；`sched/scheduler.py:564`） |
| W5 | **vLLM 0.28**，GPU block 被新請求重用時，上面還有沒做完的拷貝 | 同步 `wait` 那些拷貝 | 那一步的 forward | 是 | PCIe 那一段（`offloading/scheduler.py:555-560, 1462-1474`；`offloading/worker.py:316-317`） |
| W6 | **LMCache**（經典 connector），每一步 | GPU→CPU 拷貝，每個 chunk `store_stream.synchronize()` | forward step（滿不滿都一樣） | 是 | 不是「滿了才等」，是「每次都付」（`gpu_connectors.py:404-410`；`vllm_v1_adapter.py:1227-1236`） |
| W7 | **Dynamo KVBM**，請求結束時 G1→G2 還沒寫完，而且 GPU block 不夠 | 等這個請求的 G1→G2 寫完。單一工作依序處理，還要等排在前面的請求 | 新請求留在 vLLM 的等待佇列 | 是（`request_finished` 一律 True） | PCIe 那一段，加上請求之間排隊（`leader.rs:600-626`；`slot.rs:1536-1554`；`worker.rs:412-417`） |
| W8 | **Dynamo KVBM**，只設 disk、不設 CPU cache（`should_bypass_cpu_cache()`），GPU block 不夠 | 等 G1→G3（GPU→SSD）寫完，GPU block 才釋放 | 新請求留在等待佇列 | 看組態：只設 disk 時，這就是固定行為 | **像**：等的是 SSD 寫入。差別是沒有 CPU 層，等的是「GPU 的 KV 寫到 SSD」，不是「CPU 騰出空間」（`config.rs:280-307`；`slot.rs:1655-1680`） |

**在預設設定下，沒有任何系統會因為「往 SSD 的寫入還沒做完」而讓新請求等。** CPU 滿時的預設做法：

| 系統 | 預設做法 | 證據 |
|:--|:--|:--|
| vLLM 0.19.1 | LRU 丟掉舊的；丟不了就不存新的 | `cpu/manager.py:136-148`、`offloading/scheduler.py:225-230` |
| vLLM 0.28 | 不存新的 | `offloading/scheduler.py:1326-1334` |
| LMCache | 截斷這次 store | `cache_engine.py:518-525` |
| SGLang | 放棄備份，或丟掉子樹 | `unified_radix_cache.py:1683-1686, 978-983` |
| Dynamo KVBM | G2 配置失敗：G1→G2 卸載放棄；G3 滿：略過 | `state.rs:146-155`；`slot.rs:1555-1580`；`offload.rs:465-476` |

**照字面，hold 有現實意義，但範圍要縮小：**
1. **完全同構（等 SSD 寫完才有 CPU 空間）**：只有 LMCache `force_store_wait=True`（W1）。這是非預設設定。
2. **等 SSD，但沒有 CPU 層**：Dynamo 只設 disk（W8）。GPU 的 KV 要寫到 SSD 才能放新請求。
3. **一半同構（GPU 上的 KV 要先卸載出去才能放新請求）**：W2–W5、W7，等的是 GPU→CPU（PCIe），不是 CPU→SSD。08 的 hold 把兩段等待接在一起（CPU 要先騰出空間，GPU 才能卸載），真實系統裡這兩段是分開的：CPU 沒空間時就不卸載（丟掉），不會等。
4. 所以 08 的 hold 結果建議標成：「**對應同步實作才成立**（例如 LMCache `force_store_wait=True`、Dynamo 只設 disk）；**預設設定的真實系統是『寫入中占空間＋放不下就丟』**」。這比 11 的兩個選項都細一點。最後要不要降級，由主 agent／老師決定。〔判讀〕

---

## 3. 各系統細節

### 3.1 vLLM 0.19.1（`OffloadingConnector`；路徑前綴 `/mlsteam/workspace/src/vllm/vllm/`）

**有哪幾層**：
- 內建只註冊 `CPUOffloadingSpec`（`v1/kv_offload/factory.py:56-58`）。
- handler 只有 GPU→CPU、CPU→GPU 兩個方向（`v1/kv_offload/cpu/spec.py:105-106`）。
- 原始碼裡沒有 disk 或 SSD 層（`grep -i "disk|ssd|tier"` 在 `v1/kv_offload` 和 `offloading*` 底下沒有結果）。
- **所以 0.19.1 本身沒有「CPU→SSD」這一段。** 要有 SSD 層，得接 LMCache（`lmcache_connector.py:106-108` 預設載入 LMCache 的經典 adapter）或升到 0.28。

**Q1：CPU 滿時怎麼做？丟掉。**
- `prepare_store` 算出要淘汰幾個 block（`cpu/manager.py:136`）。
- 交給 policy 淘汰（`cpu/manager.py:143`）。淘汰的 block 直接 `_free_block` 放回 free list（`cpu/manager.py:146-148, 78-79`），不寫到任何地方。
- LRU 只挑 `ref_cnt == 0` 而且不在保護名單的 block（`cpu/policies/lru.py:37-41`）：
  - `ref_cnt == -1`：還在寫入中，不能淘汰（`cpu/policies/abstract.py:15-16, 25`：`ref_cnt = -1` 表示 not ready）。
  - `ref_cnt > 0`：正在被載入，也不能淘汰。
- 湊不到足夠的 block：
  - `evict` 回傳 None（`lru.py:42-43`），`prepare_store` 跟著回傳 None（`cpu/manager.py:144-145`）。
  - scheduler 端只記一條 warning，然後 `continue`（`offloading/scheduler.py:225-230`）：
    ```python
    store_output = self.manager.prepare_store(new_block_hashes)
    if store_output is None:
        logger.warning("Request %s: cannot store %s blocks", req_id, num_new_blocks)
        continue
    ```

**Q1'：寫入是同步還是非同步？GPU→CPU 是非同步。**
- 每次搬移拿一條 CUDA stream。
  - stream 先 `wait_stream(current_stream)`，也就是等模型算完。
  - 再 `wait_event(上一個搬移的 end_event)`，保證照順序做。
  - 然後送出 `swap_blocks`（`v1/kv_offload/worker/cpu_gpu.py:178-211`）。
- 卸載工作延到**下一步一開始**才送出，避免拖慢這一步的取樣（`offloading/worker.py:324-332, 310-314`）。
- 有沒有做完用 `end_event.query()` 輪詢，不阻塞（`cpu_gpu.py:227-229`）。
- **寫入時機**：每一步把每個請求新算滿的 block 都卸載出去（`offloading/scheduler.py:194-258`），不是等 GPU 要淘汰才搬。〔判讀〕效果上就是 GPU→CPU 寫穿。

**Q2：新請求會不會等前一個寫入？**
- **不等 CPU 空間**：CPU 沒空就不存（上面的 Q1）。
- **會等 GPU block（有條件）**：
  1. 請求結束時，如果還有 store 沒做完，`request_finished` 回傳 True（`offloading/scheduler.py:332-333`）。
     - 這時 scheduler 不釋放它的 GPU block（`v1/core/sched/scheduler.py:1823-1832`）。
     - 要等 worker 回報 `finished_sending` 才釋放（`sched/scheduler.py:2135-2138`）。
     - 在這段期間，新請求的 `allocate_slots` 如果回傳 None，就 `break`，留在等待佇列（`sched/scheduler.py:746-764`）。
     - 〔判讀〕只有 GPU KV 快滿時才會發生。等多久取決於 D2H 拷貝。
  2. 搶占：
     - 被搶占的請求，它的 GPU block 立刻釋放（`sched/scheduler.py:965`）。
     - worker 在 forward 之前用 `handle_preemptions` → `worker.wait` → `event.synchronize()`，同步等它的 store 做完（`offloading/worker.py:305-308`；`cpu_gpu.py:249-253`；呼叫點 `v1/worker/gpu_model_runner.py:3805-3808`）。
     - 同一步只要有搶占，就不排新的等待請求（`sched/scheduler.py:564`）。
- **讀取端（不是本題，但相關）**：
  - 要的 block 正在被另一個請求載入時，`get_num_new_matched_tokens` 回傳 None，請求延後（`offloading/scheduler.py:126-140`）。
  - 還在寫入中（not ready）的 block，在 `lookup` 裡算 miss（`cpu/manager.py:94-95`），所以是**改成重算，不是等**。
  - 非同步載入的請求進入 `WAITING_FOR_REMOTE_KVS`（`sched/scheduler.py:787-791`），在 `_try_promote_blocked_waiting_request` 之前一直被跳過（`sched/scheduler.py:578-588`）。

**Q3：有沒有水位線？OffloadingConnector 沒有。**
- 另一個 `SimpleCPUOffloadConnector` 的 lazy 模式有。
  - 啟用方式：`VLLM_USE_SIMPLE_KV_OFFLOAD=1` 加上 `kv_connector_extra_config.lazy_offload=true`（`config/vllm.py:659-663`；`simple_cpu_offload_connector.py:77`）。
- 做法：
  - 目標：保留 `(1+WATERMARK_RATIO) × ⌈max_num_batched_tokens/block_size⌉` 個 GPU block 處於「空的或已經卸載過」的狀態。`WATERMARK_RATIO = 1.0`（`v1/simple_kv_offload/manager.py:186-201`）。
  - 每一步從 GPU free queue 的頭（最先被淘汰的那端）往後掃，把還沒在 CPU 的 block 拷到 CPU（`manager.py:375-441`）。
  - 一次最多拷 CPU 剩下的空 block 數（`manager.py:408-413`）。
  - CPU pool 是 `BlockPool`，配置時直接淘汰已快取的 block（`v1/core/block_pool.py:331-339`），也就是丟掉。
- 〔判讀〕這是 **GPU→CPU** 的背景提早搬，不是 CPU→SSD。

### 3.2 vLLM 0.28.0（補充；路徑前綴 `/mlsteam/workspace/src/vllm-v0.28.0/vllm/`）

**為什麼要補**：
- 第一階段 C6（3.69 GiB/s 那個數字）用的是 `venv/tiara-v028`（`07_report.md:55`）。
- 這個 venv 是 editable install，指到 `src/vllm-v0.28.0`（`__editable___vllm_0_28_0_rocm722_finder.py`）。
- 0.28 新增了 `TieringOffloadingSpec`，有真的 SSD（fs）層（`v1/kv_offload/factory.py:65-68`；`tiering/factory.py:109-113`）。

**Q1：CPU 滿時怎麼做？不存新的，不等。**
- 設計原則寫在檔頭：「Always offload to all tiers」（`tiering/manager.py:10-12`）。
- block 寫進 CPU 之後，在 `complete_store` 裡對每個下一層：
  - 先 `create_store_job`，裡面的 `prepare_read` 把 `ref_cnt` 加一，鎖住這個 CPU block；
  - 再 `submit_store`（`tiering/manager.py:653-664, 680-689`）。
- 寫完之前，這個 CPU block 不能淘汰（`cpu/manager.py:140-144, 186-192`）。〔判讀〕這一點和 hold「寫往 SSD 中的 chunk 占著 CPU 空間」一樣。
- 湊不到空間時，`prepare_store` 回傳 None（`cpu/manager.py:190-192, 200-201`；`tiering/manager.py:581-584`）。scheduler 記 `ALLOCATION_FAILURE`，印 `"cannot store chunks"`，然後 `continue`（`offloading/scheduler.py:1326-1334`）。
- 〔判讀〕**hold 會讓請求等；0.28 是不卸載這段 KV。** 代價是之後這段 KV 回來時，不在 CPU 也不在 SSD，只能重算。fs 層只透過 CPU 寫入（`tiering/manager.py:13-14`）。

**寫入下一層：非同步，沒有背壓。**
- fs 層註解：「submit_store / submit_load are non-blocking」（`tiering/fs/manager.py:94`）。
- `submit_store` 把工作丟進 `DualQueueThreadPool`（`fs/manager.py:217-229`）。
  - 佇列是沒有上限的 `deque`（`fs/thread_pool.py:77-78, 119-131`）。
  - 預設 16 條讀執行緒、16 條寫執行緒（`fs/manager.py:115-116`）。
- 能用 O_DIRECT 就用，不行就退回 buffered（`fs/manager.py:185-195`；`fs/io.py:111-117`）。〔判讀〕對 H13 有用：0.28 預設會繞過 page cache。
- fs 層本身沒有容量上限，也不淘汰（`grep -i "capacity|evict|max_bytes"` 在 `tiering/fs/` 沒有結果）。

**Q2：新請求會不會等？**
- 請求結束時 `request_finished` 回傳 False，GPU block 立刻釋放（`offloading/scheduler.py:1602-1641`）。
- 但 scheduler 記下「哪些 GPU block 上還有沒做完的 store」（`offloading/scheduler.py:555-560`）。這些 block 被新配置出去時，就把相關工作列進 `jobs_to_flush`（`offloading/scheduler.py:1462-1474`）。worker 在 forward 之前同步 `wait`（`offloading/worker.py:316-317`）。
- 讀取端：
  - CPU 裡的 block 還在寫入中，回傳 `HIT_PENDING`（`cpu/manager.py:125-126`），請求延後（`offloading/scheduler.py:622-631, 828-833`）。
  - 這和 0.19.1「算 miss、重算」不一樣：**0.28 會讓「要讀剛寫的 KV」的請求等寫入做完**，但等的是 GPU→CPU。
  - 從 fs 搬回 CPU 時 CPU 已滿，算 MISS（`tiering/manager.py:361-362, 403-404, 440-449`）。

**Q3：有沒有水位線？沒有**（`grep -i "watermark|high_water|low_water|proactive"` 在 `v1/kv_offload` 和 `offloading/` 底下沒有結果）。

### 3.3 LMCache（commit `6448b447`；路徑前綴 `.../LMCache/lmcache/`）

vLLM 0.19.1 的 `LMCacheConnectorV1` 預設載入的就是這個經典 adapter（vLLM `lmcache_connector.py:106-108`）。另外有一個多行程（MP）模式，行為不同，最後一段另外寫。

**Q1：CPU 滿時怎麼做？丟掉，不寫 disk。**
- `LocalCPUBackend.allocate`：先試一次配置（`v1/storage_backend/local_cpu_backend.py:682-684`）。失敗就進 `while True`，每次挑一個淘汰候選並 `batched_remove`（`local_cpu_backend.py:688-709`）。
- 只挑可以淘汰的：`can_evict` 要 `ref_count == 1` 而且沒被 pin（`v1/memory_management.py:923-928`）。
- 淘汰只是從 dict 拿掉、ref 減一，**沒有寫回 disk**。
  - storage manager 裡有 `# TODO (Jiayi): add write-back logic here`（`v1/storage_backend/storage_manager.py:478`）。
  - 〔判讀〕disk 上的資料，是 store 當時就以寫穿方式寫下去的，不是淘汰時才寫。
- 找不到可淘汰的候選時（`local_cpu_backend.py:715-732`）：
  ```python
  if wait_other_requests:
      if not busy_loop:
          ... break
      time_to_wait = 0.1
      logger.warning("No eviction candidates found in local cpu backend. ...")
      time.sleep(time_to_wait)
  ```
- store 呼叫時 `busy_loop` 等於 `force_store_wait`，預設 False（`v1/cache_engine.py:510-525`）：
  ```python
  memory_obj = self.storage_manager.allocate(
      kv_shapes, kv_dtypes,
      busy_loop=self.config.get_extra_config_value("force_store_wait", False), ...)
  if memory_obj is None:
      logger.warning("Local cpu memory under pressure so choosing to store only %d total chunks of KV cache.", ...)
      break
  ```
- 逐層（layerwise）store 也一樣（`cache_engine.py:706-719`）。
- **結論**：
  - 預設：截斷這次 store，只存前面幾個 chunk，後面的丟掉。
  - `force_store_wait=True`：無限等，每 0.1 s 重試一次。

**寫入下一層：寫穿＋非同步，寫完前占著 CPU。**
- `batched_put` 把同一批 chunk 送到每一個沒被略過的後端（`storage_manager.py:412-431`）。也就是寫穿：CPU、disk、remote 同時寫。
- disk 寫入：
  - 先 `memory_obj.ref_count_up()`，再 `asyncio.run_coroutine_threadsafe(...)` 丟到 disk worker（`v1/storage_backend/local_disk_backend.py:426-437`）。
  - 寫完 `write_file` 才 `ref_count_down()`（`local_disk_backend.py:699-712`）。
  - 〔判讀〕寫入期間 ref_count 是 2，`can_evict` 為假。**和 hold 一樣：寫往 disk 中的 chunk 占著 CPU 空間。**
- 預設 4 條 I/O 執行緒（`local_disk_backend.py:38, 173-176`）。佇列沒有上限（subagent 報告 `job_executor/pq_executor.py:137-148` 設 `max_size = 0  # infinite`，我沒有逐行核對）。
- O_DIRECT 預設關（`local_disk_backend.py:167-170`）。〔判讀〕對 H13 有用：LMCache 預設寫 disk 會經過 page cache。
- disk 層自己的容量：
  - 不夠就 LRU 淘汰 disk 上的檔案（`local_disk_backend.py:386-418`）。
  - 還是不夠，就略過這個 put（`local_disk_backend.py:395-403, 409-413`）。
  - 這段在呼叫者的執行緒上同步做，包括 `os.remove`（subagent 指出 `local_disk_backend.py:290`，我沒有逐行核對）。

**Q2：新請求會不會等？**
- store 是同步的，就在 forward 收尾的 `wait_for_save` 裡呼叫 `lmcache_engine.store(...)`（`integration/vllm/vllm_v1_adapter.py:1227-1236`）。vLLM 0.19.1 在每一步 forward 結束時呼叫 `wait_for_save`（vLLM `v1/worker/kv_connector_model_runner_mixin.py:116-117`）。
- GPU→CPU 拷貝每個 chunk 都 `store_stream.synchronize()`（`v1/gpu_connector/gpu_connectors.py:404-410`），而且逐 chunk 執行（`gpu_connectors.py:423-425`）。〔判讀〕每一步都付 D2H 的時間，滿不滿都一樣。
- CPU 滿：
  - 預設：截斷 store，不等。
  - `force_store_wait=True`：在推論執行緒上等（W1）。
- `request_finished` 回傳 False，GPU block 立刻釋放（`vllm_v1_adapter.py:1939`）。因為拷貝已經同步做完了。
- 讀取端：從 disk 讀進 CPU 時，用預設的 `busy_loop=True` 配置 CPU 空間（`local_disk_backend.py:765`；預設值在 `local_cpu_backend.py:650`），所以可能忙等 CPU 空間。subagent 指出非同步預取路徑改用 `busy_loop=False`（`local_disk_backend.py:618-625`）。也就是說，**「讀 SSD」可能被「CPU 被寫入中的 chunk 占滿」擋住**。〔判讀〕這是 hold 在讀取端的版本。

**Q3：有沒有水位線？**
- 經典路徑：沒有。淘汰只在 `allocate` 裡、需要時才做。
- MP（多行程）模式有背景水位線：
  - `trigger_watermark=0.8`、`eviction_ratio=0.2`（`v1/distributed/config.py:474-478`）。
  - 每 1 秒檢查一次，超過水位就淘汰（`v1/distributed/storage_controllers/eviction_controller.py:160-190`）。
  - 淘汰的去處只有 DISCARD（`eviction_controller.py:192-198`）。
  - L1→L2 是背景執行緒寫穿：「store all keys to all adapters, never delete from L1」（`storage_controllers/store_policy.py:113-125`）。
- MP connector 的 `wait_for_save` 是非同步送出（`integration/vllm/lmcache_mp_connector.py:1000-1008`）。`request_finished` 回傳 `self._can_store`，會延後釋放 GPU block（`lmcache_mp_connector.py:1564-1566`）。〔判讀〕這和 vLLM 0.19.1 的 W3 同類。

### 3.4 SGLang HiCache（commit `5cbf9498`；路徑前綴 `.../sglang/python/sglang/srt/`）

**版本注意**：
- 這個 commit 沒有 `hiradix_cache.py`。HiCache 由 `UnifiedRadixCache` 實作。
- 樹的核心邏輯預設用 Rust（`environ.py:699` `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND = EnvStr("rust")`），Python 版是對照實作。
- 下面引的樹邏輯是 Python 版（`mem_cache/unified_cache/unified_tree_core.py`）。Rust 版對應的函式我只確認存在（`rust/sglang-radix-tree/src/unified_tree_core.rs:1531` 也有 `if self.is_write_back`），沒有逐行比對。
- 阻塞的那段 `writing_check` 在 Python 的 `unified_radix_cache.py`，不管樹核心用哪個都會走到。

**`--hicache-write-policy` 查證**：選項確實是 `["write_back", "write_through", "write_through_selective"]`，預設 `"write_through"`（`arg_groups/fields/memory.py:118-124`）。
- 觸發門檻 `write_through_threshold`：write_through 是 1，其餘是 2（`mem_cache/unified_radix_cache.py:528-530`）。
- `write_back` 在插入時永遠不備份（`unified_tree_core.py:1067-1085`，`if self.is_write_back: return False`）。

**L1→L2（GPU→host）**：
- 用獨立的 CUDA stream 做非同步拷貝，完成與否用 `finish_event.query()` 輪詢（subagent 報告 `mem_cache/l2_transfer.py:56-57, 131-142`、`unified_radix_cache.py:3281-3287`，我沒有逐行核對）。
- write_through 時，正在拷貝的節點用 `inc_lock_ref` 鎖住（`unified_radix_cache.py:1644-1646`），GPU 淘汰時不會選到它（`lock_ref > 0` 就不算可淘汰的葉節點，`unified_tree_core.py:2243-2244`）。〔判讀〕**是把 GPU 記憶體釘住，不是等。**
- GPU 淘汰 `evict_device_leaf`（`unified_tree_core.py:1726-1759`）：
  - 已經備份過的：下放到 host。
  - 沒備份、write_through：直接刪掉（`# Write-through: node has no backup, delete entirely.`）。
  - 沒備份、write_back：回傳一個備份動作。
- **write_back 會等**（`unified_radix_cache.py:969-977`）：
  ```python
  backup_kv = self._evict_device_leaf(node_id, tracker)
  if node_id is not None and backup_kv is not None:
      written = self._execute_and_commit_kv_backup(backup_kv, write_back=True)
      if written > 0:
          self.writing_check(write_back=True)
          self._demote(node_id, tracker)
  ```
  `writing_check(write_back=True)`（`unified_radix_cache.py:3357-3369`）：
  ```python
  # Blocking: submit what is still queued, then wait for every ack.
  cc.start_writing()
  while self.ongoing_write_through:
      for ack in cc.ack_write_queue:
          ack.finish_event.synchronize()
  ```
  `_evict` 結束時，write_back 也會再呼叫一次（`unified_radix_cache.py:770-774`）。
  - 〔判讀〕這是在排程執行緒上（配置 GPU 空間時，`evict_for_alloc` → `_evict`）。
  - 每淘汰一個葉節點，就等**所有**還沒做完的寫入。

**L2 滿（host 滿）**：
- host 配置不阻塞：`if need_size > self.available_size(): return None`（`mem_cache/pool_host/base.py:532-537`）。
- 寫 host 前先 `evict_host(needed)`，淘汰不夠就 `return None`（`unified_radix_cache.py:1682-1686`）。
- host 葉節點要能淘汰，需要同時滿足：已經下放、已經備份、`host_lock_ref == 0`、`lock_ref == 0`、沒有子節點（`unified_tree_core.py:2252-2271`）。
  - **正在寫 L3 的節點有 `host_lock_ref`**（`unified_radix_cache.py:1947-1950`），所以會被跳過，不會等它寫完。
- 備份失敗時：
  - write_through：這次放棄（`unified_radix_cache.py:1641-1642` `return 0`）。
  - write_back：丟掉子樹（`unified_radix_cache.py:978-983`）。

**L2→L3（host→storage）**：
- 由 L1→L2 的 ack 觸發 `write_backup_storage`（`unified_radix_cache.py:1744-1748`），丟進 `backup_queue`。
- 一條背景執行緒處理，`Queue()` 沒有上限（`managers/cache_controller.py:465-469`）。
- 沒有背壓。只有 `buffer_only` 模式有積壓上限，超過就丟掉新的（subagent 報告 `mem_cache/buffer_mode/pipeline.py:653-673, 814-896`，我沒有逐行核對）。

**Q3：有沒有水位線？**
- 沒有。GPU 淘汰只在配置時需要才做（subagent 依呼叫點 grep 推得〔判讀〕）。
- write_through 本身就是「插入時就寫 host」，算是一種「提早搬」。
- L3 的 file 後端有容量上限，超過就 LRU 刪到「目前用量加上這次要寫的 ≤ 上限 × 0.9」（`environ.py:788`；`mem_cache/storage/file/lru_file_evictor.py:383-388`）。

### 3.5 NVIDIA Dynamo KVBM（commit `6ee08ec2`；路徑前綴 `.../dynamo/`）

**用的是哪套程式**：
- Python 的 vLLM connector 用的是 v1：`lib/llm/src/block_manager` 加上 `lib/bindings/kvbm`。
- `lib/kvbm-engine`（v2 的 `OffloadEngine`）在這個 commit 只有 `kvbm-engine/bin/bench_engine.rs` 用到（subagent 依呼叫點 grep 推得，我沒有重做）。
- 下面都講 v1。

**Q1：G2（host）滿時怎麼做？丟掉。**
- 配置先拿沒用過的 block，沒有就從 inactive 的 `priority_set` 拿第一個，直接 `block.reset()`（`pool/managed/inactive.rs:363-397`）。不寫 disk。
  - 排序是（優先級, 歸還時間），subagent 報告 `pool/managed/priority_key.rs:45-48`、`block.rs:494-502`，我沒有逐行核對。〔判讀〕同優先級內是 LRU。
- inactive 加上沒用過的都不夠，就立刻回傳錯誤（`pool/managed/state.rs:146-155`）：
  ```rust
  if available_blocks < count { ... return Err(BlockPoolError::NotEnoughBlocksAvailable(
  ```
- G1→G2 在 host 配置失敗時（`slot.rs:1716-1719` 的 `allocate_blocks(...).await?`），錯誤路徑送一個「空的、立即完成」的傳輸，讓 worker 不會卡住（`slot.rs:1555-1580`）。**這次卸載就放棄了。**

**寫入下一層：寫穿＋非同步，預設有頻率過濾。**
- 文件註解：「When blocks are registered … they are automatically sent to the offload manager」（`offload.rs:9`）。
- 程式：註冊時，只要 `offload_priority()` 有值就排進卸載佇列（`state.rs:265-267`）。`BasicMetadata` 一律回傳 `Some`（`block.rs:527-529`）。
- vLLM connector 裡，G1→G2 做完才註冊 host block（`slot.rs:1802`），這就觸發了 G2→G3。
- 佇列都是 `mpsc::unbounded_channel`（`offload.rs:143-154`），送出是 `self.host_offload_tx.send(request).unwrap()`（`offload.rs:618-632`），**送出端不會被擋**。
- 背壓只在 worker 內部：
  - `mpsc::channel(1)` 加上最多 `max_concurrent_transfers` 個進行中的傳輸（`offload/pending.rs:223, 238-245`）。
  - 預設 4，可用 `DYN_KVBM_MAX_CONCURRENT_TRANSFERS` 改（`offload.rs:73-81`）。
  - 〔判讀〕卡住的只是背景的卸載 worker，不是請求路徑。
- G2→G3 什麼時候略過（`offload.rs:439-476`）：
  - 來源 block 已經不在了（被淘汰）；
  - 目標層已經有了；
  - 頻率過濾沒過；
  - 目標層滿了：`"Target pool full. Skipping offload."`。
- **頻率過濾預設開著**：
  - `FrequencyFilter::new(2, Duration::from_secs(600), 1_000_000, ...)`（`bindings/kvbm/src/block_manager.rs:58-75`）。
  - 每看到一次，計數加倍；計數 ≥ 2 才放行（`offload/filter.rs:106-125`）。
  - 可用 `DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER` 關掉（`block_manager.rs:61-66`）。
  - 〔判讀〕預設下，一個 block 要在 G2 註冊兩次（衰減窗內），才會寫到 disk。
  - 沒有過濾時，程式會警告「This may result in excessive disk offloading and accelerated SSD degradation」（`offload.rs:761-764`）。〔判讀〕**真實系統在意 SSD 寫入量**，這對 H1（寫入量的 Pareto）是一個佐證。

**Q2：新請求會不會等？**
- **GPU block 要等 G1→G2 寫完**：
  - Rust 端 `request_finished` 一律回傳 `Ok(true)`，註解寫「We must ALWAYS return `true` here」（`bindings/kvbm/src/block_manager/vllm/connector/leader.rs:600-626`）。
  - slot 還有沒做完的操作，就進入 `Finishing`（`slot.rs:984-1000`）。
  - worker 要等 `is_complete`（所有操作都做完）才回報結束（`connector/worker.rs:412-417`；`lib/llm/src/block_manager/connector/scheduler.rs:174-176`）。
  - 〔判讀〕在這之前，vLLM 不釋放這個請求的 GPU block（同 vLLM 的 W3 機制）。
- **G1→G2 是排隊做的**：單一的 `LocalOffloadTask` 依序 `await` 每個請求的卸載（`slot.rs:1536-1554`）。〔判讀〕請求之間有排隊；前面的請求寫得慢，後面的請求就晚釋放 GPU block。
- **G2→G3 不擋 GPU block**：
  - G2→G3 傳輸的 `connector_req` 是 None，不綁請求。subagent 報告 `distributed/utils.rs:63-67`、`block/data/logical/distributed_leader_worker.rs:124-128`，我沒有逐行核對。
  - worker 直接執行，不記在請求的操作裡（`distributed/transfer.rs:571-573`）。
- **只設 disk 的組態（W8）**：
  - 條件：`should_bypass_cpu_cache()` 為真，也就是設了 disk cache、沒設 CPU cache（`lib/llm/src/block_manager/config.rs:280-307`）。
  - 這時卸載走 G1→G3（`slot.rs:1655-1680`），請求的操作就是「寫到 SSD」。
  - 〔判讀〕**GPU block 要等 SSD 寫完才釋放。** 這是五個實作裡，預設行為（在這個組態下）就會讓新請求等 SSD 的一個。

**Q3：有沒有水位線？沒有。**
- `grep -i "watermark|high_water|low_water"` 在 `llm/src/block_manager`、`bindings/kvbm/src`、`kvbm-engine/src`、`kvbm-logical/src`、`kvbm-config/src` 底下，只找到兩處，都和淘汰無關：
  - 一個發布順序的 watermark（`kvbm-logical/src/manager/mod.rs:106`）；
  - 一個測試（`kvbm-logical/.../lineage/mod.rs:830`）。
- 淘汰只在配置時需要才做。

**文件**：
- `lib/bindings/kvbm/README.md:20` 寫「asynchronous block offload/onboard」，和程式一致。
- `lib/kvbm-engine/docs/offload.md` 講的是 v2，vLLM connector 沒有用（subagent 報告，我沒有打開）。

---

## 4. 和我們的 hold 模型差在哪

hold 的定義（08 §3）：
- 寫往 SSD 中的 chunk 占著 CPU 空間，直到寫完；
- 下一個請求要等「CPU 已用＋寫往 SSD 中 ≤ 容量」才能開始，因為 GPU 上前一輪的 KV 必須先卸載出去。

| 面向 | hold（harness） | 真實系統（預設設定） | 證據 |
|:--|:--|:--|:--|
| 寫入中的資料占 CPU 空間 | 是 | **一樣**：vLLM 0.28（`ref_cnt` 鎖住）、LMCache（ref_count＝2）、SGLang L2→L3（`host_lock_ref`）、Dynamo G2→G3（`PendingTransfer` 持有來源 block；subagent 報告 `offload/pending.rs:61-64`，我沒有逐行核對） | §3.2–§3.5 |
| 放不下時 | **下一個請求等** | **不存或丟掉，請求照跑**：<br>• vLLM 0.19.1：LRU 丟舊的；丟不了就不存新的<br>• vLLM 0.28：不存新的<br>• LMCache：截斷新的 store，前面的 chunk 留著，後面的丟掉<br>• SGLang：放棄備份，節點只在 GPU，之後 GPU 淘汰時刪掉<br>• Dynamo：G1→G2 卸載放棄 | §2 表 |
| 代價出現在哪 | 現在：等待時間算進 TTFT | **之後**：被丟或沒存的 KV 回來時要重算（或用 Cake 從別層補）。〔判讀〕**等待換成了命中率下降** | — |
| 寫到 SSD 的時機 | 依策略：S4 系列滿了才搬，S1、S2b 寫穿 | **全部是寫穿**：<br>• vLLM 0.28：寫進 CPU 就複製到所有下一層<br>• LMCache：每次 store 同時寫所有後端<br>• SGLang L2→L3：L1→L2 做完就送<br>• Dynamo G2→G3：一註冊進 G2 就排隊（預設要過頻率過濾）<br>**沒有任何一個是「CPU 淘汰時才搬到 SSD」**。LMCache 只在註解裡寫了 `TODO … write-back`；SGLang 的 `write_back` 只用在 GPU→host | `tiering/manager.py:10-12`；`storage_manager.py:412-431, 478`；`unified_radix_cache.py:1744-1748`；`state.rs:265-267` |
| GPU 端的等待 | 和 CPU 空間綁在一起：CPU 沒空，GPU 就卸不出去 | **和 CPU 空間分開**：CPU 沒空就不卸載；只有「GPU block 要重用，但上面的 D2H 還沒做完」才等（W2–W5、W7）。<br>例外：Dynamo 只設 disk 時，等的是 GPU→SSD（W8） | §2 表 |
| 背景提早搬（水位線） | 沒有，11 新增了 S4W 對照組 | **CPU→SSD 五個都沒有。** 只有兩個相關的：<br>• GPU→CPU 的背景提早搬（vLLM `SimpleCPUOffloadConnector` lazy 模式）<br>• L1 超過水位就丟（LMCache MP 模式）<br>〔判讀〕S4W 在真實系統沒有對應，但仍是合理的對照組 | §3.1、§3.3–§3.5 |

**〔判讀〕對研究的意義：**
1. **寫穿是真實系統的預設，08 的發現和它一致。**
   - 08 在 hold 下發現寫穿（S1、S2b）最快。
   - 真實系統的設計者也都選了寫穿：CPU→SSD 不在請求路徑上；CPU 淘汰就是丟，因為 SSD 上通常已經有一份。Dynamo 預設有頻率過濾，所以不一定有。
   - 所以「滿了才搬到 SSD」（S4 系列）**不是真實系統的做法**。把 S4 系列當成「現狀」的對照組，可能不公平。真正的現狀基準應該是 S1（寫穿）加上「放不下就丟」。
2. **hold 在真實系統對應的是「寫入中占空間 → 放不下就不存」。**
   - 這條路的代價不出現在這一次的 TTFT，而是出現在**之後回來的請求要重算**。
   - harness 目前有 free 和 hold，**沒有這第三種**，暫稱 `drop`：寫入中占空間，CPU 放不下時，新的 chunk 不存在 CPU，也不存在 SSD（因為 SSD 只透過 CPU 寫入），回來時重算。這是 vLLM 0.28、LMCache、SGLang、Dynamo 的預設。vLLM 0.19.1 沒有 SSD 層，做法是丟掉 LRU 最舊的。
   - 如果要讓模擬更貼近這些預設，建議在 `m7_sim.py` 加 `--release drop`。〔判讀，要主 agent 決定〕
3. **N3（時機）方向**：
   - 有 CPU 層時，真實系統往 SSD 的寫入本來就不在關鍵路徑上。只有兩種情況在：非預設的同步設定（W1），以及沒有 CPU 層的 Dynamo 只設 disk（W8）。所以「延後要等」這個論點，在一般預設實作下不成立。
   - 但「**寫入中占空間 → 丟掉 → 之後重算**」是新的代價形式：延後搬（滿了才搬）在真實系統會變成「丟掉」。這可能讓「寫入時就決定放 SSD」有一個 hold 沒抓到的好處：寫入時直接放 SSD 的 chunk 不占 CPU 槽，比較不會被丟。
   - 這要用 `drop` 模型模擬才知道。〔判讀，未驗證〕

---

## 5. 最小實驗設計：vLLM 0.19.1 ＋ MI300X 上，CPU offload 滿了時新請求的 TTFT 會不會變長

**只寫設計，沒有跑。** 所有數字都是 NOT_MEASURED。下面的容量用 Llama-3.1-8B 的 KV 大小〔算術〕估：

> 32 層 × 8 個 KV head × 128 維 × 2（K、V）× 2 B ＝ 128 KiB／token
> 32K token ＝ 4 GiB

### 5.1 要回答的問題

- **Q-a**：CPU 層滿了（vLLM 0.19.1：LRU 丟掉；可選的 0.28 組：寫入中占空間＋不存新的）時，**新的、不相干的請求** TTFT 會不會比 CPU 沒滿時長？
- **Q-b**：同一個條件下，**被丟掉的舊 session 回來時**，TTFT 比 CPU 沒滿時長多少？也就是「丟掉」換成「重算」的代價。
- **Q-c**：如果 Q-a 有變長，是 GPU 端的等待（W3、W4）造成的，還是 PCIe 爭用？

### 5.2 組別（開跑前寫死）

| 組 | vLLM | connector | CPU 容量（`cpu_bytes_to_use`） | GPU KV（`--kv-cache-memory-bytes`） | 目的 |
|:--|:--|:--|:--|:--|:--|
| A0 | 0.19.1（`venv/tiara`） | 無 | — | 10 GiB（約 2.5 個 32K session〔算術〕） | 沒有 offload 的基準 |
| A1 | 0.19.1 | `OffloadingConnector`，`CPUOffloadingSpec`，LRU | 64 GiB（16 個 session，不會滿） | 10 GiB | CPU 沒滿 |
| A2 | 0.19.1 | 同上 | **12 GiB（3 個 session）** | 10 GiB | **CPU 滿 → LRU 丟掉** |
| A2g | 0.19.1 | 同 A2 | 12 GiB | **64 GiB**（GPU 放得下全部） | GPU 不緊。和 A2 比，分出 GPU 端的等待（W3、W4） |
| A3（可選） | 0.28（`venv/tiara-v028`） | `TieringOffloadingSpec`，`secondary_tiers=[{"type":"fs","root_dir":<本地 NVMe 或 NFS>}]` | 12 GiB | 10 GiB | **寫入中占空間＋不存新的**（最接近 hold 的預設實作） |

共同設定：
- Llama-3.1-8B-Instruct，`HIP_VISIBLE_DEVICES=0`，`--enable-prefix-caching`（預設開）。
- `--max-num-batched-tokens 8192`，`--max-num-seqs 4`，`--max-model-len 40960`。
- A1–A3 都不設 `block_size`，也就是 offload block＝GPU block（`v1/kv_offload/spec.py:99-113`）。
- connector 設定照 `code/m7_c6_vllm.py` 的寫法，只改 `cpu_bytes_to_use`。

### 5.3 工作負載（每個 rep 都用新的隨機 token，避免跨 rep 命中；照 `m7_c6_vllm.py` 的做法）

1. **暖機**：一個 512 token 的請求。
2. **灌滿**：K＝6 個互不相干的 32K session，`max_tokens=1`，**前一個回來就立刻送下一個**（gap＝0）。每個都記 TTFT。A2 從第 4 個開始，CPU 會滿〔算術〕。
3. **新請求探針**：灌滿最後一個回來後，隔 g 秒送一個新的不相干請求。
   - g ∈ {0, 0.25, 1, 10} s；長度 L_p ∈ {512, 8K}。
   - g＝10 s 當作「搬移都做完了」的基準。
4. **回來探針**：
   - 重送第 1 個 session，在原本的 32K 後面加 512 個新 token。它在 LRU 最舊，A2 會把它丟掉，A1 不會。
   - 重送第 K 個 session（最新，A1、A2 都應該還在 CPU）。
   - 記 TTFT。
5. 每格 6 個 rep，格子的順序隨機。換組時重啟 server。3 個 workload seed。

### 5.4 量什麼

- **TTFT**：client 端，用 stream，`max_tokens=1`。
- **把 TTFT 拆開**（`/metrics`，名稱已在 0.19.1 原始碼確認）：
  - `vllm:request_queue_time_seconds`：排隊時間。W3 的等待會出現在這裡。
  - `vllm:request_prefill_time_seconds`：prefill 時間。
  - `vllm:request_prefill_kv_computed_tokens`：實際重算的 token 數。用來判斷回來探針是命中還是重算。
  - `vllm:num_preemptions`。
  - `vllm:external_prefix_cache_queries`、`vllm:external_prefix_cache_hits`。
  - `vllm:kv_offload_total_bytes`、`vllm:kv_offload_total_time`。
  - 位置：`v1/metrics/loggers.py:530-583, 854-894`；`offloading/metrics.py:106-118`。
- **server log**：
  - 0.19.1：`"cannot store"` warning 的次數（`offloading/scheduler.py:227-229`）。
  - 0.28：`"cannot store chunks"` 的次數和 `ALLOCATION_FAILURE` 計數器。
- **A3 另外記**：`/proc/meminfo` 的 Dirty、Writeback（E1、E2）；fs 目錄的檔案數量和大小的時間序列。
- **記錄規則**：
  - 用 `m7run` 包住，原始輸出放 `/mlsteam/data/tiara/runs/<run_id>/`。
  - 用 `GpuWatcher` 包住，`contaminated` 的 run 不進 `results/`。
  - CSV 放 `results/m7_explore_mi300x/h0_*.csv`，要有 `run_id`、`ts` 欄。

### 5.5 判準（開跑前寫死）

- **Δ_new(組)** ＝ 新請求探針 TTFT 中位數（g＝0）− 同一組 g＝10 s 的中位數。
- **「CPU 滿會擋住新請求」成立**，三個條件都要符合：
  1. A2（或 A3）的 Δ_new ≥ g＝10 s 中位數的 5%，而且超過 rep 間 MAD 的 2 倍；
  2. A2 的 Δ_new − A1 的 Δ_new ≥ A1 g＝10 s 中位數的 5%。這是為了扣掉「只是 PCIe 還在忙」的部分；
  3. 多出來的時間主要落在 `request_queue_time`（排隊），而不是 `prefill_time`。
- **「擋住的是 GPU 端（W3、W4），不是 CPU 滿」**：A2 成立，但 A2g 不成立。
- **「丟掉的代價」**：
  - A2 的回來探針（第 1 個 session）TTFT 中位數 ÷ A1 的。
  - 同時用 `request_prefill_kv_computed_tokens` 確認 A2 真的重算了約 32K token、A1 沒有。
- 結果接回 H0：
  - Q-a 成立 → hold 在 vLLM 上有實測依據。
  - Q-a 不成立、Q-b 成立 → 照 §4，把 hold 改成 drop 模型。

### 5.6 讀原始碼得到的預期〔判讀，不是結果〕

| 項目 | 預期 | 依據 |
|:--|:--|:--|
| A2 的 Δ_new | 和 A1 差不多，CPU 滿不會讓新請求等 | §3.1 Q1 |
| A2 的回來探針（第 1 個 session） | 接近整段 prefill，被 LRU 丟掉了 | — |
| GPU 緊（A2 vs A2g）、g＝0 | 可能有一點排隊時間 | W3：結束請求的 GPU block 等 D2H 做完才釋放 |
| D2H 的量級 | 一個 32K session 是 4 GiB〔算術〕；D2H 速度在這台機器上 NOT_MEASURED | C6 的 3.69 GiB/s 是 CPU→GPU 載入，不是 D2H，不能直接套 |
| A3 | 預期 log 會出現 `cannot store chunks`；新請求一樣不等 | §3.2 |

### 5.7 成本與風險

- 成本：A0–A2g 每組大約是 6 個 32K prefill × (4 個 g × 2 個 L_p ＋ 2 個回來探針) × 6 rep × 3 seed。GPU 時間 NOT_MEASURED，要先跑一個 rep 估。
- 風險：
  1. **0.19.1 的 OffloadingConnector 在 ROCm 上還沒在本專案跑過。** C6 是 0.28。`MI300X_MLSTEAM.md:701` 只確認 0.19.1 有這個檔案。
     - `CPUOffloadingSpec.get_handlers` 要求 `is_cuda_alike()`（`cpu/spec.py:93-96`），ROCm 應該符合。〔判讀〕
     - 第一次啟動失敗要照規則記進 `results/RUNLOG_MI300X.md`，然後停下。
  2. `reset_prefix_cache` 會不會連 CPU 層一起清，0.19.1 沒有核對過。所以 §5.3 用「每個 rep 新的隨機 token」，而不是靠 reset。
  3. 本機沒有裝 LMCache 和 SGLang。W1、W2 的「正對照」（LMCache `force_store_wait=True`、SGLang `write_back`）要另外裝，而且 ROCm 支援〔待查〕。不在最小設計內。

---

## 6. 限制

1. **只讀原始碼，沒有實測。** 「會等」「不會等」都是程式碼的控制流程，不代表等多久。量級要靠 §5。
2. **LMCache、SGLang、Dynamo 是今天（2026-10-09）的 main**，不是某個 release。SGLang 這個 commit 已經沒有 `hiradix_cache.py`，和舊文件描述的結構不同。
3. **只核對了關鍵行號。** subagent 報告裡有些行號我沒有逐行打開，都標了「subagent 報告…我沒有逐行核對」。表格裡的「證據」欄都是我打開核對過的。
4. **SGLang 的樹核心預設是 Rust**，我引的是 Python 對照實作。阻塞點 `writing_check` 在 Python 層，不受影響；但淘汰順序的細節，Rust 版可能不同。
5. **vLLM 0.19.1 沒有 SSD 層**，所以 vLLM 這一列對「CPU→SSD」的回答是「不適用」。補充的 0.28 才有。
6. **Dynamo 只看了 vLLM connector 用的 v1 block manager。** v2 的 `kvbm-engine` 在這個 commit 只有 bench 用到，沒有細讀；它的設計同樣是寫穿、放不下就略過（subagent 報告，我沒有核對）。
7. 11 的判準寫「三個系統」。這裡看了五個實作（多了 vLLM 0.28 和 Dynamo），判定不受影響：只看原本的三個（vLLM 0.19.1、LMCache、SGLang），也已經有 W1–W4、W6 會等。

---

## 附錄：原始碼取得紀錄

| 系統 | 取得方式 | 位置 | commit |
|:--|:--|:--|:--|
| vLLM 0.19.1 | 本機既有 | `/mlsteam/workspace/src/vllm`（`git describe` → `v0.19.1`） | `b1388b1fbf5aaef47937fabe98931211684666a6` |
| vLLM 0.28.0 | 本機既有（`venv/tiara-v028` 的 editable 來源） | `/mlsteam/workspace/src/vllm-v0.28.0` | `2cf0a6915ce544dc493a0990f2ea38d81601128a` |
| LMCache | `git clone --depth 1 https://github.com/LMCache/LMCache`，成功 | `/mlsteam/data/tiara/runs/20261009-164921-h0-src/LMCache` | `6448b44787f2fe1b3c5ddbc0fc83aea47fe1118b` |
| SGLang | `git clone --depth 1 https://github.com/sgl-project/sglang`，成功 | `/mlsteam/data/tiara/runs/20261009-164921-h0-src/sglang` | `5cbf949839b1fc73802d20670943ccbeb805d3f2` |
| Dynamo | `git clone --depth 1 https://github.com/ai-dynamo/dynamo`，成功 | `/mlsteam/data/tiara/runs/20261009-164921-h0-src/dynamo` | `6ee08ec25010ef59b2308e8aca2f5a7709448701` |

clone 下來的程式**沒有安裝，也沒有執行**，只用 `grep`、`sed`、`awk` 讀。
