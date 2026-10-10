# D7：OS 角度（page cache、dirty writeback、反向讀、GDS、SSD 掉崖）

**判準寫於**：2026-10-10T05:38:42Z（`date -u`）。第一個 benchmark 是 `20261010-054238-d7-env`（05:42:38），在判準之後。
**§2 是當時寫的原文，一字未改**；只有章節順序（把 §0、§1 放到前面）和最前面這幾行是事後加的。
**結果寫於**：2026-10-10，約 06:10Z。
**GPU**：沒有用。
**程式**：`code/m8_os_bench.py`（子命令 env、pc、rd、seqw、dedup、bufmt、warm、analyze）。
**資料**：`results/m8_directions/d7_*.csv`（每列有 `run_id`、`ts`）；原始輸出在 `/mlsteam/data/tiara/runs/<run_id>/`。

---

## 0. 一句話結論＋判定

**一句話**：OS 這一層確實讓「SSD 層」不像我們以為的那樣（LMCache、SGLang 預設的 buffered 寫入會把 100% 的資料留在 DRAM，至少 45 s），但它能提供的好處，延後版幾乎都拿得到：page cache 隨時可以丟，OS 自己就是一個延後寫入的緩衝。**沒有找到「只有寫入當下才拿得到」的 OS 資源**。唯一結構上像的（GPU 直接寫 SSD，E9），這台機器量不到。

| 子題 | 照 §2 判準字面 | 綜合判定 | 最重要的理由 |
|:--|:--|:--|:--|
| **(a) H13／E1 page cache 占 DRAM** | 有看頭 | **可能** | DRAM 確實被占：buffered 寫完 30 s 後，自己的檔 100% 還在 page cache。但延後版（之後 `fdatasync`＋`DONTNEED`）0.5–3.1 s 就還回來。判準裡「buffered 比 O_DIRECT 慢 >20%」這一條，只在單執行緒成立（寫入時間長 60%）；4 條執行緒時 buffered 反而是單執行緒 O_DIRECT 的 2.4 倍（4.65 vs 1.945 GiB/s；多執行緒 O_DIRECT 沒量，NOT_MEASURED）。剩下的是**容量帳要修**，不是「寫入時決定」的新證據 |
| **(b) E2 dirty writeback** | 死路 | **死路** | 12 GiB dirty 之內沒有掉崖（p99／p50 = 1.5）。照算術，要 dirty 到約 63 GiB 才會開始擋寫入者〔算術〕。OS 本身就是幾十 GiB 的延後寫緩衝，這對延後版有利 |
| **(c) H10 反向讀** | 有看頭（只在 2 MiB） | **可能（偏死路）** | 本地 SSD、O_DIRECT、2 MiB 時，反向讀慢 13.8%（3 次範圍不重疊）。但 O_DIRECT 讀大檔、IO ≥8 MiB 時，差距都在 −5.3%～+9.3%，範圍重疊。讀取端改用 ≥8 MiB 的 IO 就沒差，所以讀取時版追平。影響更大的是 IO 大小，還有小 IO 被寫入干擾得更嚴重 |
| **(d) E9 ROCm GPU 直接讀寫儲存** | 可能 | **這台是死路** | LMCache 有 hipFile 後端（官方標示 alpha），但要求本地 NVMe＋ext4／xfs。這台的儲存是 MegaRAID 虛擬磁碟＋LVM＋overlay，沒有 NVMe，也沒裝 hipFile |
| **(e) E6 SSD 寫入掉崖** | 死路 | **死路（只測到 32 GiB）** | 32 GiB 連續 O_DIRECT 寫，每秒都在 2,362–2,471 MiB/s，沒有掉崖。RAID 卡把 SSD 內部擋住了，看不到 SLC 或壽命 |
| (f) 額外：NFS 去重／壓縮 | 校準沒問題 | — | 每個 4 KiB 都不同：947–981 MiB/s；4 KiB 重複：611–920 MiB/s。重複的資料沒有比較快 |

**三個關鍵數字**：
1. buffered 寫完 30 s 後，自己的檔還在 page cache 的量：本地 16,384／16,384 MiB、NFS 4,096／4,096 MiB。本地的 dirty（還沒寫到磁碟）在寫完 30.24 s 後才寫下去〔實測 20261010-054645-d7-pc-local-buf、20261010-054826-d7-pc-nfs〕。
2. 寫進 page cache 的速度，1／4／16 條執行緒分別是 1.37–1.41／4.65／9.49–9.69 GiB/s。O_DIRECT 單執行緒每 chunk 一個檔是 1.945 GiB/s，連續寫一個大檔是 2.36–2.47 GiB/s〔實測 20261010-060442-d7-bufmt、20261010-054449-d7-pc-local-odfs、20261010-060133-d7-seqw〕。
3. 本地 64 MiB 讀取：O_DIRECT 冷讀 7,816 MiB/s，buffered 冷讀 2,214 MiB/s（慢 3.5 倍），page cache 熱讀 7,744–7,775 MiB/s。所以**在這台的本地 SSD 上，page cache 對讀取沒有好處**；在 NFS 上，熱讀是 O_DIRECT 冷讀（951 MiB/s）的約 8 倍〔實測 20261010-055150-d7-rd-local、20261010-055608-d7-rd-nfs、20261010-060332-d7-warm；倍數為算術〕。

**一個之前沒想到的 OS 角度**：這台是 container，cgroup 的 `memory.max` 只有 434.8 GiB（整台 2.2 TiB），而 **page cache 算在這個額度裡**。開跑時這個 cgroup 已經有 390.9 GiB 的檔案快取，歷史上撞過上限 20 次〔實測 20261010-054238-d7-env〕。所以「CPU 層能用多少 DRAM」看的是 container 的額度，不是整台機器，而 buffered 寫 SSD 產生的 page cache 會和它搶同一個額度。

---

## 1. 問題是什麼、要證明什麼

**問題**：Tiara 的模型把 SSD 層當成兩件事：(1) 不占 DRAM；(2) 讀寫速度＝裝置速度（第一階段用 O_DIRECT 校準，`code/m7_calib.py:173-203`）。但真實系統寫 SSD 時，如果走一般的 `write()`（buffered），資料會先進 OS 的 page cache（DRAM 裡的檔案快取），兩個假設都可能錯。

**要證明（或殺掉）的事**：
- (a) 真實系統是 buffered 還是 O_DIRECT？buffered 時 DRAM 被占多少、多久？如果被占，「寫入當下選 O_DIRECT」是不是寫入時才有的價值，過不過延後測試？
- (b) dirty writeback（kernel 晚一點才把資料寫到磁碟）會不會讓寫入在某個量之後突然變慢（掉崖）？
- (c) Cake 從後面讀。反向讀會不會比較慢？IO 大小影響多大？
- (d) ROCm 上能不能讓 GPU 直接讀寫儲存（hipFile）？有沒有 KV 系統支援？
- (e) SSD 持續寫會不會掉崖（SLC cache 用完）？

**白話名詞**：
- **page cache**：OS 替檔案在 DRAM 裡留的副本。讀過或寫過的檔，OS 會先留著，需要記憶體時再丟。
- **dirty**：已經寫進 page cache、還沒寫到磁碟的資料。kernel 預設放 30 s（`dirty_expire_centisecs=3000`）或累積到一定量才寫。
- **O_DIRECT**：不經過 page cache，直接讀寫磁碟。
- **`DONTNEED`**：`posix_fadvise(POSIX_FADV_DONTNEED)`，告訴 OS「這個檔的快取可以丟了」。
- **cgroup／memcg**：container 的資源額度。這裡的 page cache 算在 container 的記憶體額度裡。

**延後測試（計劃 10 §2）怎麼套在這裡**：寫入時選 O_DIRECT（不留快取）的延後版，是「先 buffered 寫，之後再 `DONTNEED`」。只有延後版要多付一份稀缺資源（N2）、或延後做不到時，寫入時的選擇才有獨特價值。

---

## 2. 事先寫好的判準（開跑前寫死，之後不改）

名詞：
- **buffered**：一般 `write()`，資料先進 OS 的 page cache（DRAM），之後由 kernel 在背景寫到磁碟。
- **O_DIRECT**：繞過 page cache，直接寫磁碟。
- **dirty**：已經寫進 page cache、還沒寫到磁碟的資料。
- **延後版（deferred twin）**：同一個效果，等之後再做。例如「先 buffered 寫，之後再 `posix_fadvise(DONTNEED)` 把 page cache 丟掉」。

量測裝置（沿用第一階段校準，`code/m7_calib.py:5`、`code/m7_write_policy.py:446,497`）：
- 本地 SSD＝`/var/tmp/<run_id>`（container 的 overlay，底下是 `sda` 的 LVM）。
- NFS＝`/mlsteam/data/tiara/runs/<run_id>/io`（NetApp nfs4）。

### (a) H13／E1：page cache 偷 DRAM

量：本地 SSD 與 NFS，各寫 8 GiB（128 個 64 MiB chunk，每 chunk 一個檔），三種寫法：buffered、buffered＋每 chunk fsync、O_DIRECT。寫之前、寫的時候、寫完後 45 s，每 0.2 s 記一次：
- 我自己檔案留在 page cache 的量（`cachestat` 或 `mincore`，只算自己的檔）。
- `/proc/meminfo`（Cached、Dirty、Writeback、MemAvailable）與 cgroup `memory.stat`（file、file_dirty、file_writeback）。

原始碼：vLLM 0.28 fs 層、LMCache local_disk、SGLang HiCache file／nixl、Dynamo KVBM disk，預設是 O_DIRECT、buffered 還是 mmap（檔名:行號）。

| 判定 | 條件 |
|:--|:--|
| **死路** | 四個系統預設都 O_DIRECT；**或** buffered 寫完 30 s 後，兩個裝置上自己檔案留在 page cache 的量都 <5% 寫入量（計劃 10 H13 的停損） |
| **可能** | 至少一個系統預設 buffered，而且留存 ≥5%；**但**延後版（buffered 寫，之後 fsync＋`DONTNEED`）能在 ≤60 s 內把 DRAM 還回來，而且 buffered 寫入時間不比 O_DIRECT 長超過 20%。→ 容量帳要修，但「寫入當下選 O_DIRECT」可以延後，沒有寫入時才有的價值 |
| **有看頭** | 至少一個系統預設 buffered，留存 ≥50%，**而且**延後版付得起的代價過不了上面那條（DONTNEED 丟不掉、要等 >60 s，或 buffered 比 O_DIRECT 慢 >20%，也就是多吃 DRAM 頻寬＝N2） |

### (b) E2：dirty writeback

量：(a) 的每一次 64 MiB `write()` 的延遲；加一次本地 SSD 連續 buffered 寫 32 GiB，記延遲與 Dirty 的時間序列。另外讀（不改）`vm.dirty_*` 與 cgroup 的 `memory.max`，用算術推出「寫入者開始被 kernel 擋住」的 dirty 量。

| 判定 | 條件 |
|:--|:--|
| **死路** | 測到的範圍內沒有掉崖：buffered 每次 write 的 p99 ≤ 中位數×2，而且沒有 >1 s 的停頓；而且算出來的門檻 > 32 GiB（＝第一階段模擬裡最大的 CPU 層）。→ buffered I/O 只是把 SSD 成本藏到背景，等於 OS 幫你做了「延後寫」 |
| **可能** | 量不到掉崖，但算出來的門檻 ≤ 32 GiB；或有 >1 s 的停頓但不可重現 |
| **有看頭** | 測到掉崖：某段 write 延遲 ≥ 中位數×5，而且發生在 dirty ≤ 32 GiB 時 |

### (c) H10：讀取方向與 IO 大小

量：本地 SSD（8 GiB 檔）與 NFS（4 GiB 檔），IO 大小 2、8、32、64、256 MiB，順向 vs 反向（Cake 從後面讀），O_DIRECT 與 buffered（冷，先用 `DONTNEED` 清掉自己的檔並確認），每格 3 次。再加「同時有背景寫入」（O_DIRECT，64 MiB chunk）時，2 與 64 MiB 的順向 vs 反向。

| 判定 | 條件 |
|:--|:--|
| **死路** | IO ≥2 MiB 時，每個裝置、每種模式，順向與反向的吞吐量中位數差都 <10%（計劃 10 H10 的停損） |
| **可能** | 只有 buffered 模式或只在一個裝置差 ≥10%，或差 ≥10% 但 3 次的範圍重疊 |
| **有看頭** | O_DIRECT（真實系統的預設，見 (a)）在至少一個裝置差 ≥10%，3 次範圍不重疊 |

### (d) E9：ROCm 上的 GPU 直接讀寫儲存（hipFile）

只查文件、套件、核心設定，不用 GPU。

| 判定 | 條件 |
|:--|:--|
| **死路** | 這台沒有 hipFile，而且儲存不符合 hipFile 的條件（本地 NVMe＋ext4／xfs）；**或**沒有任何 KV 系統支援 |
| **可能** | 有 KV 系統支援，但這台的硬體或檔案系統不符合，要換機器才能量 |
| **有看頭** | 這台裝得起來、儲存符合、而且至少一個 KV 系統支援 |

### (e) E6：SSD 寫入掉崖

先看 `/proc/diskstats` 10 s：`sda` 被別人用的 util >10% 就跳過。裝置看得到型號才做。做法：本地 SSD 連續 O_DIRECT 順序寫 ≤40 GiB（64 MiB 一次），每 1 s 記吞吐量。

| 判定 | 條件 |
|:--|:--|
| **死路** | 40 GiB 內沒有掉崖（任何連續 10 s 的吞吐量都 ≥ 前 10 s 的 50%） |
| **可能** | 有短暫掉落（<10 s）或不可重現 |
| **有看頭** | 40 GiB 內掉崖：連續 ≥10 s 低於前 10 s 的 50% |

### (f) 額外（我加的，OS 以外的儲存端）：NFS 伺服器有沒有去重／壓縮

第一階段校準寫的是「同一個 4 KiB 隨機區塊重複」的資料（`code/m7_calib.py:166,189`）。NetApp 可能做 inline 去重或壓縮。量：NFS 上 O_DIRECT 寫 4 GiB「每個 4 KiB 都不同的隨機資料」vs 4 GiB「4 KiB 重複」，各 3 次。

| 判定 | 條件 |
|:--|:--|
| **校準沒問題** | 兩者吞吐量中位數差 <20% |
| **校準要重做** | 重複資料快 ≥20%（第一階段的 NFS 寫入參數偏樂觀） |

### 共同限制（照任務規則）

- 不用 GPU；自己的緩衝 ≤ 64 GiB；不改 sysctl、不 drop_caches；只用 `DONTNEED`／O_DIRECT 清自己的檔。
- 累計寫入 ≤ 150 GiB；跑完刪掉所有資料檔。
- 每個數字都要能追到 `m7run` 的 run_id。

---

## 3. 做了什麼

### 3.1 量測（全部走 `m7run`，exit code 都是 0，stderr 都是空的）

| run_id | 子命令 | 做什麼 | 寫入 |
|:--|:--|:--|:--|
| `20261010-054238-d7-env` | env | 讀 `vm.dirty_*`、cgroup 記憶體、裝置型號；用一個 64 MiB 檔互驗 cachestat 與 mincore | 各 64 MiB |
| `20261010-054343-d7-smoke-pc`、`…-054400-d7-smoke-pcnfs`、`…-055137-d7-smoke-rd` | 煙霧測試 | 只寫在 run 目錄，**不進 results** | 共約 5.6 GiB |
| `20261010-054449-d7-pc-local-odfs` | pc | 本地：O_DIRECT、buffered＋每 chunk fsync，各 8 GiB（128 個 64 MiB 檔），寫完觀察 45 s，再做延後版（`fdatasync`＋`DONTNEED`） | 16 GiB |
| `20261010-054645-d7-pc-local-buf` | pc | 本地：buffered 16 GiB，同上（兼作 (b) 的持續寫入） | 16 GiB |
| `20261010-054826-d7-pc-nfs` | pc | NFS：三種寫法各 4 GiB | 12 GiB |
| `20261010-055150-d7-rd-local` | rd | 本地 8 GiB 大檔＋32 個 chunk 檔；IO 2／8／32／64／256 MiB × 順向／反向 × O_DIRECT／buffered（冷）× 3 次；加上背景寫入時的 2／64 MiB | 10 GiB＋背景約 28 GiB〔算術〕 |
| `20261010-055608-d7-rd-nfs` | rd | NFS 2 GiB 大檔＋16 個 chunk 檔，同上 | 3 GiB＋背景約 30 GiB〔算術〕 |
| `20261010-060133-d7-seqw` | seqw | 先看 `sda` 10 s 有沒有別人在用，再連續 O_DIRECT 寫 32 GiB，每秒記一次 | 32 GiB |
| `20261010-060203-d7-dedup` | dedup | NFS O_DIRECT：每 4 KiB 不同 vs 4 KiB 重複，各 2 GiB × 2 次 | 8 GiB |
| `20261010-060332-d7-warm` | warm（**事後加的**） | 只讀：一個已經 100% 在 page cache 的模型權重檔（NFS 上），量熱讀速度 | 0 |
| `20261010-060442-d7-bufmt` | bufmt（**事後加的**） | 1／4／16 條執行緒 buffered 寫 4 GiB，寫完 2–4 s 內就刪掉，所以只進 page cache、不落盤（用 cgroup `io.stat` 確認） | 落盤 ≤132 MiB |
| `20261010-060721-d7-analyze` | analyze | 從原始 CSV 算彙整表（讀取方向、干擾倍數、寫入延遲） | 0 |

**怎麼量「自己的檔占了多少 page cache」**：
- NFS：`cachestat()`（Linux 6.5 以後的 syscall 451），可以分出 cached、dirty、writeback。
- 本地（overlay）：`cachestat()` 永遠回 0（env probe：mincore 數到 16,384 頁，cachestat 是 0）。改用 `mmap`＋`mincore()`，只知道「在不在快取」。dirty 只能看整個 cgroup 的 `file_dirty`＋`file_writeback`（開跑時是 0–1 MiB，所以幾乎都是我的）。
- 同時每 0.2 s 記 `/proc/meminfo`（Cached、Dirty、Writeback、MemAvailable）與 cgroup `memory.stat`。本地因為要對 256 個檔做 mincore，實際間隔約 0.5 s。

**寫入的資料**：隨機位元組，每個 4 KiB 開頭再蓋上「chunk 編號＋區塊編號」，避免儲存端去重。

### 3.2 原始碼（只讀）

路徑前綴同 H0：vLLM 0.28＝`/mlsteam/workspace/src/vllm-v0.28.0/`；LMCache、SGLang、Dynamo＝`/mlsteam/data/tiara/runs/20261009-164921-h0-src/`（commit 見 H0：LMCache `6448b44`、SGLang `5cbf949`、Dynamo `6ee08ec2`）。

### 3.3 E9 的檢查（不用 GPU，只看檔案與套件）

`find / -xdev -iname '*hipfile*'`、`dpkg -l`、兩個 venv 的 `pip list`、`/proc/modules`、`/dev/nvme*`、`/sys/bus/pci/devices/*/p2pmem`、`/opt/rocm/.info/version`，加上 LMCache 文件 `docs/source/kv_cache/storage_backends/gds.rst`。

---

## 4. 結果

### 4.1 (a) 真實系統寫磁碟用什麼

| 系統（磁碟層） | 預設 | 證據 | 寫入執行緒 |
|:--|:--|:--|:--|
| vLLM 0.28 `fs` 層 | **O_DIRECT**（開檔時試一次，不行才退回 buffered） | 〔程式碼 `vllm/v1/kv_offload/tiering/fs/io.py:41-66, 95, 111-117`〕〔程式碼 `…/fs/manager.py:185-195`〕〔程式碼 `csrc/fs_io.cpp:14-15, 51-57`〕 | 讀 16＋寫 16（`manager.py:115-116`） |
| LMCache `local_disk` | **buffered**（`use_odirect` 預設 False） | 〔程式碼 `lmcache/v1/storage_backend/local_disk_backend.py:167-171`〕寫：`778-787`；讀：`799-816` | 4（`local_disk_backend.py:38`） |
| LMCache `gds` 後端 | cuFile（預設）／hipFile；GDS 關掉時走 POSIX，`use_direct_io` 預設 False＝buffered | 〔程式碼 `gds_backend.py:291-299, 316-339, 343-357`〕〔程式碼 `lmcache/v1/config.py:355-364`〕 | 執行緒池 |
| SGLang HiCache `file`（`--hicache-storage-backend file`） | **buffered**（`numpy.tofile` 寫；`open(..., buffering=0)` 只是 Python 層不緩衝，仍經過 page cache），沒有 O_DIRECT 選項 | 〔程式碼 `sglang/srt/mem_cache/hicache_storage.py:383, 489-491, 541-543`〕 | — |
| SGLang HiCache `nixl` | **O_DIRECT**（預設 True） | 〔程式碼 `sglang/srt/environ.py:800-802`〕〔程式碼 `mem_cache/storage/nixl/nixl_utils.py:38-47, 210`〕 | — |
| Dynamo KVBM disk（G3） | **O_DIRECT**（開檔後用 `fcntl` 加上；可用 `DYN_KVBM_DISK_DISABLE_O_DIRECT` 關掉） | 〔程式碼 `dynamo/lib/llm/src/block_manager/storage/disk.rs:21, 51-55, 100-135`〕；傳輸走 NIXL 的 POSIX／GDS_MT（`block_manager/distributed/worker.rs:93-114`） | — |

- SGLang 的 `storage/mmap/mmap_allocator.py` 是主機記憶體配置器（`MAP_SHARED|MAP_ANONYMOUS`，`:100, 167`），不是磁碟層。
- **五個磁碟後端裡，兩個預設 buffered：LMCache `local_disk`、SGLang `file`**。計劃 10 H13 的停損（「都用 O_DIRECT」）不成立。

### 4.2 (a) buffered 時 DRAM 被占多少、多久

| 裝置 | 寫法 | 寫入量 | 寫入時間（GiB/s） | 寫完時在 page cache | 寫完 30 s 後 | 45 s 後 | dirty 最高 | dirty 歸零（寫完後） | 延後版 `fdatasync`＋`DONTNEED` 花多久 → 剩多少 |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| 本地 | O_DIRECT | 8 GiB | 4.114 s（1.945） | 0 | 0 | 0 | 2 MiB | — | 0 s → 0 |
| 本地 | buffered＋fsync | 8 GiB | 10.367 s（0.772） | 8,192 MiB | 8,192 | 8,192 | 65 MiB | 0.14 s | 1.877 s → 0 |
| 本地 | buffered | 16 GiB | 13.131 s（1.219） | 16,384 MiB | 16,384 | 16,384 | 12,136 MiB | **30.24 s** | 3.09 s → 0 |
| NFS | O_DIRECT | 4 GiB | 5.725 s（0.699） | 0 | 0 | 0 | 0 | — | 0 s → 0 |
| NFS | buffered | 4 GiB | 6.927 s（0.577） | 4,096 MiB | 4,096 | 4,096 | 0（取樣時） | 0.09 s | 0.524 s → 0 |
| NFS | buffered＋fsync | 4 GiB | 7.120 s（0.562） | 4,096 MiB | 4,096 | 4,096 | 0（取樣時） | 0.19 s | 0.546 s → 0 |

〔實測 `20261010-054449-d7-pc-local-odfs`、`20261010-054645-d7-pc-local-buf`、`20261010-054826-d7-pc-nfs`；`d7_pc_summary.csv`〕

讀法：
- **buffered 和 buffered＋fsync 都會把 100% 的資料留在 DRAM**，觀察的 45 s 內一點都沒少。fsync 只是讓它變乾淨（已寫到磁碟），不會讓它離開 DRAM。
- 本地 buffered 時，約 10 GiB 的 dirty 停在 DRAM 裡 30 s，然後一次寫下去（剛好是 `dirty_expire_centisecs=3000`）。也就是說，**「寫到 SSD」實際上是 30 s 後才發生**。
- 這段期間 cgroup 的 `memory.current` 從 413,431 MiB 漲到 429,903 MiB（+16 GiB），全部算在 container 的額度裡；`pgsteal_direct` 沒變（77,746,574），代表這次 cgroup 還有空間，沒有逼別人的快取被回收。`DONTNEED` 後降回 411,142 MiB〔實測 `20261010-054645-d7-pc-local-buf` 時間序列〕。
- 「多久」：在記憶體不緊的時候，乾淨的 page cache 會一直留著，直到記憶體不夠或有人丟掉它。45 s 以後會留多久：NOT_MEASURED（看同一個 container 裡其他人的記憶體用量）。
- **NFS 的 buffered 不會延後寫**：每個 chunk 一個檔，`close()` 時 NFS 就把資料送出去（close-to-open）。64 MiB 的寫入裡，`write()` 37 ms、`close()` 61 ms（中位數）〔實測 `d7_pc_latency.csv`〕。這就是計劃 10 的 E7。

**寫入當下選 O_DIRECT，過不過延後測試？**
- 延後版「之後再丟」很便宜：0.52–3.09 s 就把 4–16 GiB 還回來（上表最後一欄）。判準的「≤60 s」成立。
- 判準的另一條「buffered 寫入時間比 O_DIRECT 長 >20%」：單執行緒時成立。本地 buffered 每 GiB 0.821 s，O_DIRECT 0.514 s，長 60%〔算術，由上表〕。所以**照判準字面是「有看頭」**。
- 但這一條是單執行緒的結果。事後加量的 bufmt（判準之後才決定）顯示，寫進 page cache 的速度隨執行緒數上升：

| 執行緒 | 寫進 page cache（GiB/s），2 次 | 每個 64 MiB write 的中位數 | 落盤（cgroup `io.stat`） |
|:--|:--|:--|:--|
| 1 | 1.374、1.413 | 41.9 ms | 132、0 MiB |
| 4 | 4.650、4.649 | 43.2–43.4 ms | 0、0.1 MiB |
| 16 | 9.487、9.686 | 45.8–46.1 ms | 0、0 MiB |

〔實測 `20261010-060442-d7-bufmt`〕

  - LMCache 預設 4 條寫入執行緒（`local_disk_backend.py:38`）。這時 buffered 的 4.65 GiB/s 是單執行緒 O_DIRECT（1.945 GiB/s）的 2.4 倍〔算術〕。多執行緒 O_DIRECT：NOT_MEASURED（寫入預算已經用完，見 §7）。
  - 所以「buffered 比較慢」這個 N2 代價，在真實系統的執行緒數下不成立。〔判讀〕

**我的綜合判定：可能**。〔判讀〕
- 成立的部分：在 LMCache、SGLang `file` 的預設設定下，「SSD 層」的每個位元組都在 DRAM 裡多一份（CPU 層一份、page cache 一份），而且算在 container 的記憶體額度裡。第一階段的容量帳沒有算這一份。
- 不成立的部分：這不是「寫入時放置」的價值。page cache 隨時可以丟（延後版便宜）；而且「選 O_DIRECT」是全域設定，不是逐個 chunk 依位置決定。vLLM 0.28、Dynamo、SGLang `nixl` 本來就預設 O_DIRECT。
- 逐個 chunk 決定「進不進 page cache」有沒有意義：本地 SSD 上沒有，因為熱讀（7.7 GiB/s）和 O_DIRECT 冷讀（7.8 GiB/s）一樣快（§4.4）。NFS 上有（熱讀快 8 倍），但這等於「在 DRAM 多留一份」，就是寫穿（S1、S2b）加上之後再淘汰，延後版已經做得到。

### 4.3 (b) dirty writeback：寫入延遲與掉崖

讀到的設定〔實測 `20261010-054238-d7-env`〕：
- `vm.dirty_ratio=20`、`dirty_background_ratio=10`、`dirty_bytes=0`、`dirty_background_bytes=0`、`dirty_expire_centisecs=3000`、`dirty_writeback_centisecs=500`。
- 全機的門檻（`/proc/vmstat`）：`nr_dirty_threshold`＝112,022,081 頁＝427.3 GiB；`nr_dirty_background_threshold`＝55,942,651 頁＝213.4 GiB〔算術：×4 KiB〕。
- cgroup：`memory.max`＝466,877,906,944 B（434.8 GiB）；`memory.current`＝405.4 GiB；`file`＝390.9 GiB；`memory.events max 20`；`pgsteal_direct` 77,582,734 頁（約 296 GiB）。

每個 64 MiB 寫入的延遲〔實測；`d7_pc_latency.csv`，inclusive 分位數〕：

| 裝置／寫法 | n | 中位數 | p99 | 最大 | p99／中位數 | >1 s | 分四段的中位數 |
|:--|:--|:--|:--|:--|:--|:--|:--|
| 本地 O_DIRECT | 128 | 29.3 ms | 29.8 | 29.9 | 1.02 | 0 | 29.4／29.3／29.3／29.3 |
| 本地 buffered（16 GiB，dirty 到 12 GiB） | 256 | 42.2 ms | 63.4 | 100.0 | 1.50 | 0 | 54.0／54.1／47.6／41.8 |
| 本地 buffered＋fsync | 128 | 76.3 ms（write 42.1＋fsync 34.1） | 142.5 | 153.1 | 1.87 | 0 | 76.0／76.4／76.1／76.5 |
| NFS O_DIRECT | 64 | 63.7 ms | 629.7 | 646.5 | 9.89 | 0 | 63.7／64.0／60.7／64.3 |
| NFS buffered | 64 | 100.2 ms（write 37.2＋close 60.8） | 173.8 | 247.6 | 1.73 | 0 | 106.6／103.8／100.1／95.9 |
| NFS buffered＋fsync | 64 | 104.1 ms（write 42.0＋fsync 62.0） | 169.5 | 228.4 | 1.63 | 0 | 108.5／106.5／104.1／100.2 |

讀法：
- **沒有掉崖**：buffered 的 p99／中位數是 1.50（本地）、1.73（NFS），都 ≤2；沒有 >1 s 的停頓。本地 buffered 越寫越快（54 → 42 ms），不是越寫越慢。
- **buffered 沒有把 SSD 的成本藏起來**（單執行緒）：本地 buffered 每次 42 ms，比 O_DIRECT 的 29 ms 還慢。NFS 的 buffered 在 `close()` 就付掉了寫入成本。
- NFS O_DIRECT 有 1 次 646 ms 的停頓（64 次裡 1 次），是網路或伺服器端的抖動，不是 dirty 造成的（O_DIRECT 沒有 dirty）。〔判讀〕
- 什麼時候會開始擋寫入者：
  - 這個 cgroup 的 dirty 門檻〔算術；公式〔未查證〕：照 Linux `mm/page-writeback.c` 的 memcg 規則，可用量≈cgroup 的檔案頁＋剩餘額度，憑記憶，本機沒有 kernel 原始碼可以核對〕：可用量 ≈ 390.9＋29.5＝420.3 GiB → 背景寫回從 42.0 GiB 開始，硬門檻 84.1 GiB，開始減速約在兩者中間 63 GiB。
  - 全機門檻是 213.4／427.3 GiB（實測值），比 cgroup 鬆，所以 cgroup 的先到〔判讀，前提是本地檔案系統支援 cgroup writeback；overlay 底下是什麼檔案系統，在 container 裡看不到〕。
  - 用 4 條執行緒（4.65 GiB/s）連續寫，約 9 s 到 42 GiB；假設磁碟以 seqw 的 2.4 GiB/s 往下寫，再約 9 s 到 63 GiB〔算術，NOT_MEASURED〕。
  - 實際的掉崖：NOT_MEASURED。要讓 dirty 超過 60 GiB，至少要再寫 >60 GiB，超出寫入預算，也可能干擾 D2。
- **判定：死路**（§2 三個條件都成立：p99 ≤2×、沒有 >1 s、算出來的門檻 >32 GiB）。
- 對研究的意思〔判讀〕：OS 本身就是一個「延後寫」的緩衝，buffered 的系統（LMCache 預設）在幾十 GiB 以內，寫入者完全不會被磁碟擋住。這對延後版有利，而且會讓 D1 要找的「寫入積壓」區域更難發生。

### 4.4 (c) 讀取方向與 IO 大小

順向 vs 反向（MiB/s，3 次的中位數［最小–最大］；「分開」＝兩組 3 次的範圍不重疊）〔實測 `d7_read_summary.csv`〕：

| 裝置 | 模式 | IO | 順向 | 反向 | 反向比順向 | 分開？ |
|:--|:--|:--|:--|:--|:--|:--|
| 本地 | O_DIRECT | 2 MiB | 4,778［4,256–4,782］ | 4,120［3,939–4,129］ | **−13.8%** | 是 |
| 本地 | O_DIRECT | 8 MiB | 6,100 | 6,030 | −1.1% | 否 |
| 本地 | O_DIRECT | 32 MiB | 7,330 | 8,009 | +9.3% | 否 |
| 本地 | O_DIRECT | 64 MiB | 7,816 | 8,071 | +3.3% | 否 |
| 本地 | O_DIRECT | 256 MiB | 7,398 | 7,005 | −5.3% | 否 |
| 本地 | O_DIRECT，背景寫入中 | 2 MiB | 899 | 729 | **−18.9%** | 是 |
| 本地 | O_DIRECT，背景寫入中 | 64 MiB | 2,158 | 2,265 | +5.0% | 否 |
| 本地 | O_DIRECT，每 chunk 一個檔 | 64 MiB | 6,634 | 7,939 | +19.7% | 否 |
| 本地 | buffered 冷讀 | 2 MiB | 2,129 | 1,713 | −19.5% | 是 |
| 本地 | buffered 冷讀 | 8 MiB | 2,503 | 2,130 | −14.9% | 否 |
| 本地 | buffered 冷讀 | 32／64／256 MiB | 2,399／2,214／2,224 | 2,518／2,315／2,401 | +4.9／+4.6／+7.9% | 否 |
| NFS | O_DIRECT | 2 MiB | 772 | 846 | +9.6% | 否 |
| NFS | O_DIRECT | 8／32／64／256 MiB | 926／945／951／973 | 944／975／965／958 | +1.9／+3.2／+1.5／−1.5% | 否 |
| NFS | O_DIRECT，背景寫入中 | 2／64 MiB | 104／589 | 97／607 | −6.5／+3.0% | 否 |
| NFS | buffered 冷讀 | 2 MiB | 480 | 432（有一次 54） | −10.0% | 是 |
| NFS | buffered 冷讀 | ≥8 MiB | 459–483 | 460–474 | −3.6～+1.8% | 否 |

讀法：
- **照 §2 判準字面：有看頭**。本地 O_DIRECT 在 2 MiB 時反向慢 13.8%，範圍不重疊；有背景寫入時慢 18.9%。
- 但只在 2 MiB。大檔的 O_DIRECT、≥8 MiB 時，差距都在 −5.3%～+9.3%，範圍都重疊（每 chunk 一個檔的 64 MiB 反向快 19.7%，範圍也重疊）。NFS 全部 <10%。
- **讀取時版就能補**〔判讀〕：Cake 讀的時候把 IO 開到 ≥8 MiB（或每個 chunk 內部照順向讀，只有 chunk 之間反向），反向的代價就不見了。這不需要寫入時改布局，所以不算寫入時才有的價值。**綜合判定：可能（偏死路）**。
- **比方向大很多的是 IO 大小**：本地 O_DIRECT 2 MiB 只有 4,778 MiB/s，64 MiB 是 7,816 MiB/s（2 MiB 慢 39%）〔算術〕。
- **小 IO 被寫入干擾得更嚴重**（同一份資料，O_DIRECT，有無背景寫入的中位數比）〔實測 `d7_interference.csv`〕：

| 裝置 | IO | 單獨讀 | 有背景寫入 | 慢幾倍 | 第一階段 share 模型的 k（A2 校準） |
|:--|:--|:--|:--|:--|:--|
| 本地 | 2 MiB | 4,778 | 899 | 5.31（反向 5.65） | 3.31 |
| 本地 | 64 MiB | 7,816 | 2,158 | 3.62（反向 3.56） | 3.31 |
| NFS | 2 MiB | 772 | 104 | 7.43（反向 8.71） | 1.67 |
| NFS | 64 MiB | 951 | 589 | 1.61（反向 1.59） | 1.67 |

  64 MiB 時和第一階段的 k 吻合；2 MiB 時干擾是 k 的 1.6 倍（本地）到 4.5 倍（NFS）〔算術〕。如果實作用小 IO 讀（例如逐層、或 vLLM 用 16 token 的 block＝2 MiB〔算術：Llama-3.1-8B 每 token 128 KiB〕），share 模型會低估干擾。

- buffered 冷讀單執行緒只有 2,129–2,503 MiB/s（本地，各 IO 大小的中位數）；64 MiB 時是 O_DIRECT 的 28%〔算術〕。NFS 是 459–483 vs 926–973 MiB/s（≥8 MiB）〔實測〕。多執行緒 buffered 冷讀：NOT_MEASURED。

### 4.5 熱讀：「SSD 層」其實從 DRAM 讀（事後加的）

一個 3,812 MiB、讀之前與之後都 100% 在 page cache 的檔（NFS 上的模型權重，只讀）〔實測 `20261010-060332-d7-warm`〕：
- 2 MiB IO：8,356–8,397 MiB/s；64 MiB IO：7,744–7,775 MiB/s。
- 對照：本地 O_DIRECT 冷讀 64 MiB 7,816 MiB/s（一樣快）；NFS O_DIRECT 冷讀 951 MiB/s（熱讀約 8 倍）〔算術〕。
- 意思〔判讀〕：在 LMCache 預設（buffered）＋慢儲存（像 NFS）時，最近寫過的 chunk 讀起來像 DRAM。這一部分不受 KV 管理器控制，而是 kernel 的 LRU 在管。拿 LMCache 預設當對照組時，模擬器用裝置速度會低估它。

### 4.6 (d) E9：ROCm 上 GPU 直接讀寫儲存

| 項目 | 這台 | 證據 |
|:--|:--|:--|
| hipFile 套件／函式庫 | **沒有**（`find / -xdev`、`dpkg -l`、兩個 venv 都找不到；也沒有 `/opt/rocm/bin/ais-check`） | 檢查指令（§3.3） |
| ROCm | 7.2.2 | `/opt/rocm/.info/version` |
| 本地儲存 | **沒有 NVMe**。`sda`＝BROADCOM `GBT3916-MR-32PD`（`megaraid_sas`），上面是 LVM（`ubuntu--vg-data--lv`），container 裡再包一層 overlay | 〔實測 `20261010-054238-d7-env`〕、`/proc/modules`、`ls /dev/nvme*` |
| NFS | nfs4.1、TCP、rsize／wsize＝64 KiB、NetApp ONTAP 9.13.1P3；沒載 `rpcrdma` | `/proc/self/mountstats` |
| 核心 P2PDMA | MI300X（`0000:06:00.0`）有 `p2pmem`，`available`＝274,877,906,944 B（256 GiB）→ 核心有開 P2PDMA〔判讀〕 | `/sys/bus/pci/devices/0000:06:00.0/p2pmem/available` |
| KV 系統支援 | **LMCache**：`gds_backend: hipfile`，文件寫明 alpha、「local NVMe drives only」、「ext4 (data=ordered) and xfs」、要 `CONFIG_PCI_P2PDMA` | 〔程式碼 `gds_backend.py:327-335`、`hipfile_shim.py:1-80`〕〔原文 `LMCache/docs/source/kv_cache/storage_backends/gds.rst:159-203`〕 |
| | Dynamo KVBM：block manager 相依 `cudarc`（CUDA），GDS 走 NIXL `GDS_MT`（NVIDIA） | 〔程式碼 `dynamo/lib/llm/Cargo.toml:25`、`block_manager/distributed/worker.rs:106-109`〕 |
| | SGLang `nixl`：GDS 是 NVIDIA 的 | 〔原文 `sglang/.../storage/nixl/README.md:538-543`〕 |
| | vLLM 0.28：KV 卸載沒有 GDS（`cufile`／`kvikio` 只出現在載入權重的 `model_loader/weight_utils.py`） | 〔程式碼 grep〕 |

- **判定：照 §2 是「可能」（有 KV 系統支援，但這台的硬體不符合，要換機器）；在這台是死路**。
- 小發現〔判讀〕：LMCache 在 overlay 上自動關 GDS 的判斷寫的是 `fstype in ["tmpfs", "overlayfs"]`（`gds_backend.py:291`），但 `/proc/mounts` 裡 overlay 的型別字串是 `overlay`，所以在這台的 `/var/tmp` 上不會自動關。之後會不會出錯：NOT_MEASURED（沒有 hipFile 可以試）。
- 為什麼還值得記〔判讀，來自 lit_B C9〕：GPU 直接寫 SSD 是唯一「延後版結構上做不到」的 N2。延後版的資料已經在 host，第二段只能走 host→SSD；寫入時就決定「前段不進 CPU」才能整段跳過 DRAM。要驗證必須換有本地 NVMe 的機器。

### 4.7 (e) E6：SSD 持續寫

- 開跑前 10 s，`sda` 的 util＝0.00%，讀 0 MiB、寫 0.4 MiB（沒有別人在用）〔實測 `20261010-060133-d7-seqw`〕。
- 32 GiB O_DIRECT 連續寫一個大檔，13.6 s；每秒吞吐量 2,362–2,471 MiB/s（最後一秒只有部分），`sda` util 87.8–89.9%。**沒有任何一秒掉到前 10 s 的 50% 以下**。
- 限制：`sda` 是 RAID 卡上的虛擬磁碟，看不到底下有幾顆 SSD、型號、SMART、寫入壽命。32 GiB 可能還沒用完各顆 SSD 的 SLC cache。
- **判定：死路（只到 32 GiB）**。

### 4.8 (f) NFS 去重／壓縮

〔實測 `20261010-060203-d7-dedup`〕每個 4 KiB 都不同：981.0、947.1 MiB/s；4 KiB 重複（第一階段校準的寫法）：919.8、610.7 MiB/s。重複的資料沒有比較快 → **第一階段的 NFS 寫入參數沒有被去重灌水**。

### 4.9 其他發現

1. **container 的記憶體額度**（新的 OS 角度）〔實測 env〕：見 §0。補充：page cache 可以一路長到 cgroup 額度滿；之後任何新的配置，都要先在配置者自己的執行緒上回收別人的快取（direct reclaim；env run 記到這個 cgroup 累計 `pgscan_direct`＝77,597,697 頁）。KV 系統的 pinned 池是開機時一次配好，page cache 搶不走它〔判讀〕，但會讓「下一個要記憶體的人」付回收的時間〔判讀，NOT_MEASURED〕。
2. **NFS 今天比校準時快**：O_DIRECT 讀 772–973 MiB/s（2–256 MiB），校準時 377–447 MiB/s；O_DIRECT 寫 0.70 GiB/s（每 chunk 一個檔）、947–981 MiB/s（dedup），校準時 684–822 MiB/s。和 07 說的一樣，NFS 參數代表「慢的 NFS」。
3. 本地「每 chunk 一個檔」的 O_DIRECT 寫（1.945 GiB/s）比連續寫一個大檔（2.36–2.47 GiB/s）慢，差在開檔與建檔。

---

## 5. 延後版／更簡單的做法＋前作

**延後版與更簡單的做法**：

| 寫入時的想法 | 延後版／更簡單的做法 | 結果 |
|:--|:--|:--|
| 寫入時用 O_DIRECT，不占 page cache | buffered 寫，之後 `fdatasync`＋`DONTNEED` | 0.52–3.09 s 還回 4–16 GiB〔實測〕；唯一贏的「單執行緒比較快」在 4 條執行緒時不成立 |
| 寫入時用 buffered，讓 page cache 當免費的 DRAM 快取 | 寫穿（CPU 層多留一份），之後再淘汰 | 等價〔判讀〕；本地 SSD 上熱讀沒有比 O_DIRECT 冷讀快，所以沒有好處 |
| 寫入時就寫到磁碟（避免積壓） | OS 的 dirty writeback（背景版，免費） | 12 GiB 內沒掉崖；算出來約 63 GiB 才擋〔算術〕 |
| 寫入時把資料排成反序（給 Cake） | 讀取時用 ≥8 MiB 的 IO | ≥8 MiB 時差距在 ±9.3% 內〔實測〕 |
| 寫入時就用 GPU 直接寫 SSD（E9） | 先寫 CPU，之後 host→SSD | **延後版結構上做不到「不經過 DRAM」**；這台量不到 |

**前作**：
- **py-kvcache**（arXiv 2609.11744，Kanichai、De Matteis、Trivedi）：
  - 指出 llm-d 預設用 POSIX 讀寫、經過 page cache，「for our tests this is unfavourable, and is likely to distort results, especially when cached data is re-read」，所以自己 fork 成 direct I/O〔原文 p.6〕；實驗也用 direct I/O「to avoid page-cache effects」〔原文 p.11〕。
  - 結論是快取效能取決於「transfer granularity, intermediate memory use, and when transfers enter the request schedule, not only on device bandwidth」〔原文 p.1〕。這和本文的 IO 大小結果方向一致。
  - 用 KvikIO（cuFile／GDS）的版本「was slower than all our other implementations」〔原文 p.14〕。
  - 〔判讀〕前作把 page cache 當成要排除的干擾。本文是量它在真實系統預設下有多大，沒有看到有人量過。
- **Tutti**（arXiv 2605.03375）：認為 GDS「still relies on CPU intervention to initiate each I/O and thus remains CPU-centric」，提出 GPU io_uring；對照組是開 GDS 的 LMCache〔原文 p.1–2〕。
- Linux 文件（vm.rst 的 `dirty_ratio`、open(2) 的 O_DIRECT、nfs(5) 的 close-to-open）：見 `../research_20261009_explore/lit_B_summary.md` E1、E2、E7 的摘錄；這次沒有重新打開原文〔未查證〕。
- Ziggurat 讓大的非同步寫入走 page cache 再寫回（lit_B 引〔Ziggurat 原文 p.5〕，這次沒有重新核對）。

---

## 6. 如果要繼續

1. **給 D1（併發／積壓）**：真實系統要分兩類建模〔判讀〕。
   - O_DIRECT 類（vLLM 0.28、Dynamo、SGLang `nixl`）：寫入中的 chunk 占 CPU 空間，直到磁碟寫完（H0 的 hold）。
   - buffered 類（LMCache、SGLang `file` 預設）：寫入只要「複製進 page cache」的時間（單執行緒每 64 MiB 約 42 ms），CPU 空間很快就放出來；要等 dirty 累積到幾十 GiB（這台約 42–63 GiB〔算術〕）才會被擋。延後版吃虧的「積壓」區域在這類系統裡更難出現。
   - 如果要實測掉崖：用一個獨立、額度小的 cgroup 或專用機器，寫 >60 GiB dirty。這次在共用 container 裡不做。
2. **模擬器的 SSD 參數要分兩套**：O_DIRECT 用現在的校準值；buffered 冷讀用約 2.2 GiB/s（本地、單執行緒），最近寫過的 chunk 用熱讀 7.7 GiB/s，並加上「page cache 占 DRAM」的容量帳。
3. **share 模型的 k 要依 IO 大小**：2 MiB 時本地 5.3–5.7、NFS 7.4–8.7；64 MiB 時 3.6、1.6（`d7_interference.csv`）。
4. **E9 要換機器**：找有本地 NVMe（ext4／xfs）的 MI300X 節點，裝 hipFile，先跑 `ais-check`，再用 LMCache `gds_backend: hipfile` 量 GPU→SSD 直寫。這是 OS 層唯一可能過延後測試的方向（N2，結構上）。
5. 還沒量的（寫入預算用完，§7）：多執行緒 O_DIRECT 寫、多執行緒 buffered 冷讀、40 GiB 以上的持續寫。

---

## 7. 失敗與異常

1. **超過寫入預算**：累計落盤約 161 GiB〔算術〕，超過 150 GiB（本地約 107 GiB、NFS 約 54 GiB）。
   - 原因：rd 的背景寫入器沒有設上限。NFS 的 2 MiB 讀取在寫入干擾下慢 7–9 倍，背景寫入跟著跑更久。兩個 rd run 的背景寫入合計約 58 GiB〔算術：`writer_MiBps ×（secs＋0.3 s）`〕，我事前估計約 30 GiB。
   - 同一時間在磁碟上最多 32 GiB（seqw）；**所有資料檔都已刪除**（`/var/tmp/d7_*` 不存在；各 run 目錄裡沒有 `.bin`，空的 `io/` 也刪了）。
   - 發現之後就停止所有會落盤的量測：bufmt 只進 page cache、2–4 s 內刪掉（`io.stat` 落盤 0–132 MiB）；warm 只讀。
2. **cachestat 在 overlay 上永遠回 0**（`20261010-054238-d7-env`：`probe.local.after_write_fsync cachestat_cached_pages=0 … mincore_pages=16384`）。本地改用 mincore。本地的 dirty 只能看整個 cgroup 的 `file_dirty`（container 裡其他行程也會算進去；開跑時是 0–1 MiB）。
3. **`d7_pc_summary.csv` 的 p99 算錯**：用了 `statistics.quantiles` 的預設（exclusive），樣本少時 p99 會大於最大值（例：NFS O_DIRECT p99 655.77 ms > 最大 646.46 ms）。程式已改成 inclusive；**以 `d7_pc_latency.csv` 為準**。
4. 本地取樣實際間隔約 0.5 s（要對 256 個檔做 mincore），不是 0.2 s。
5. 和 §2 計劃的偏離：
   - (a) NFS 每種寫法 4 GiB（計劃 8 GiB）；本地 buffered 16 GiB（計劃 8 GiB＋(b) 另外 32 GiB，合併成一次 16 GiB，少一點 page cache 對 D2 的影響）。
   - (c) NFS 大檔 2 GiB（計劃 4 GiB）；背景寫入那組只讀前 1 GiB（本地）／0.5 GiB（NFS）。
   - (e) 32 GiB（計劃上限 40 GiB）。
   - (f) 每種 2 GiB × 2 次（計劃 4 GiB × 3 次）。
   - 事後才加的量測：bufmt、warm（判準之後才決定，已在 §3 標示）。bufmt 讓 (a) 的綜合判定從「有看頭」降成「可能」；這是事後調整，§0 兩個判定都列出來了。
6. 異常值（沒有排除）：NFS O_DIRECT 寫入有 1 次 646 ms；NFS buffered 2 MiB 反向讀有 1 次 54 MiB/s；本地「每 chunk 一個檔」O_DIRECT 順向有 1 次 7,955 MiB/s（其他兩次 6,300–6,634）。
7. 煙霧測試的 3 個 run（`*-d7-smoke-*`）用 `D7_SMOKE=1`，只寫在 run 目錄，沒進 `results/`。
8. 對 D2 的影響：本地 buffered 那次 page cache 最多多 16 GiB（cgroup `memory.current` 413,431 → 429,903 MiB），`pgsteal_direct` 沒變（沒有逼別人的快取被回收），之後全部丟掉。NFS 那次多約 4–5 GiB，也丟掉了。warm 讀了一個已經在快取裡的共用權重檔，沒有丟它的快取。
9. m7run 與 runsh 各算一次時間戳的問題：這次沒有發生（stderr 都是空的，沒有「run dir 不存在」的警告）。
