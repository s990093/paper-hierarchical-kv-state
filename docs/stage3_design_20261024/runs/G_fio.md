# G-fio：hgx-cn01 的下層實測（/tmp＝md0、/work）

> **性質**：登記（量法事先寫死；這一關沒有過／停，只是把算術用的常數換成 H200 的實測）。來源：[mechanisms/06 §1、§2](../mechanisms/06_eval_gates.md)、[01 §3](../mechanisms/01_regime.md)。
> **量時間、獨佔**。程式：`code/s3/gfio.py`（這台沒有 fio：Python 的 O_DIRECT `preadv`，讀之前另外 `POSIX_FADV_DONTNEED`）。

## 1. 量法
- 目錄：`/tmp/hungweilai-s3/gfio`（md0）與 `/work/ChihChungHsu/hungweilai/tmp/gfio`（在 compute node 上看到的 /work）。每個目錄先記 `df -T`、`findmnt`（回答 01 §3「/work 是不是 NFS」：在 head node 上 /work 是 `hgx-cn01:/work` 的 NFS；在 hgx-cn01 自己身上是什麼，由這一步記錄）。
- 測試檔：/tmp 16 GiB、/work 4 GiB（群組配額只剩約 49G）；O_DIRECT 寫入；量完刪掉。
- 循序讀 1／4／64 MiB、隨機讀 128 KiB／2 MiB／64 MiB（一個 KV 塊的大小）× 1／4／8 個執行緒；每格讀 8 GiB（或整個檔）。
## 2. 用法（事先寫死）
- A2／算術的「本地 NVMe」常數改用 /tmp 的 64 MiB 循序讀、8 執行緒的 GiB/s；「NFS」的列：如果 compute node 上的 /work 是本地檔案系統，NFS 的列一律繼續用限速層代替並標明（01 §3）。
## 3. 成本：約 10 分鐘。

## 登記
- 登記時間（`date -u`）：**2026-10-25T12:48:11Z**；commit 訊息開頭 `docs(s3-w0,gfio): prereg`。登記時沒有任何數字。

---

## 偏離（登記之後）

### 偏離 1（2026-10-25T13:06:45Z；使用者要求；G-fio 還沒跑）：每格量 6 次＋雜訊欄位
- **改了什麼**：每個（目錄, 樣式, 大小, 執行緒）讀 6 次（每次 /tmp 讀 4 GiB、/work 讀 2 GiB；原本每格 1 次 8 GiB），CSV 每列加 `n_rep`、`sigma_rel`、`delta_star`；常數用 6 次的中位數。G-fio 沒有過／停的判定，只換算術常數；2σ > 5% 的格子照實標出、不拿來當常數（先回報）。
- **為什麼**：同 runs/README.md「共同規則」。**受影響**：G-fio 的全部 run（都還沒跑）。
