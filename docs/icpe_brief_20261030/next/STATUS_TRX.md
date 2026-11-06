# STATUS_TRX（icpe-trx，TRX40）

> 只由 `icpe-trx` 寫。格式照 STATUS_DATA：日期時間／做完什麼（commit、run_id）／卡住什麼（完整錯誤）／需要別的 session 做什麼／需要使用者決定什麼。最新的一節在最上面。
> 工作＝STATUS_DATA 2026-11-05T09:00+08:00 節「給 `icpe-trx`（TRX40）的步驟」0–7：SemiAnalysis 的 `sa_main` → `sa_main_ttl7` → `sens_rep --only 'sa_*'`。不解讀上限、不下判定。

## 2026-11-07T00:50+08:00 — Mooncake P4：**2／6 完成**（rc=0、核對通過），第 3 個 `conversation_P_B2` 跑到約 53%；toolagent 比原本估的慢，**完成時間改為 11/7 約 06:00～11:00**〔推算〕

回覆 icpe-data 的詢問（訊息，見 STATUS_DATA 00:50）。

- **佇列**：pgid 2231622，log `$ROOT/runs/queue-p4mc-rust-w36-20261106-210352.log`，沒有 FAILED／KILLED，三個守門程式都沒觸發。

  | block | 起 → 迄 | 牆鐘 | job 數 | `run_s` 合計（執行緒秒） | 每 job 平均 | 中位數／p90／max |
  |:--|:--|--:|--:|--:|--:|:--|
  | `p4_b1/toolagent_P_B1` | 21:03:52 → 22:28:44 | 5,092 s | 196,718 | 181,817 | 0.924 | 0.083／2.856／7.561 |
  | `p4_b1/toolagent_M_B1` | 22:28:44 → 00:30:49 | 7,325 s | 148,738 | 262,866 | 1.767 | 0.082／5.570／8.238 |
  | `p4_b2/conversation_P_B2` | 00:30:49 → 跑中 | — | 108,557／205,160（00:48:43） | 36,615（107,856 個 job 時） | 0.339（到目前） | — |
  | `p4_b2/conversation_M_B2` | 未開始 | | 153,870 | | | |
  | `p4_b2/toolagent_P_B2` | 未開始 | | 191,920 | | | |
  | `p4_b2/toolagent_M_B2` | 未開始 | | 143,940 | | | |

  - `run_s` 取自各 run 的 `rust_timing.jsonl`（單一執行緒秒數）。兩個完成的 block 有效並行度約 35.7、35.9（`run_s` 合計 ÷ 牆鐘）。
- **完成的 2 個 block 逐項核對**都通過：
  - `exit_code` 0；`kept_jobs` 等於 count（196,718、148,738）。
  - `unmatched_cells=0`；`py_fallback_jobs=0`；`req/` 0 個檔；`rust_timing.jsonl` 行數等於 kept。
  - stderr 只有 `session groups…` 與 `x1fast serve: trace_load_s…` 兩種行。
  - `conversation_P_B2` 開頭的同樣檢查也正常：`kept_jobs=205160`，`unmatched_cells=0`，`py_fallback_jobs=0`。
- **資源**（memwatch 21:03:54 → 00:46）：
  - MemAvailable 最低 52,342 MiB；行程群組 PSS 最高 1,977 MiB。
  - swap 一直 790 MiB；load1 最高 42.56；/ssd4 剩 541 GiB。
- **為什麼比 3 h 久**：
  - 21:15 的 3 h 是用 acvlab 的 0.374 執行緒秒／job 算的。TRX40 的 toolagent B1 實測是 0.924（P）和 1.767（M）。
  - 重的 job 集中在 job 順序的 40%–80%：`toolagent_P_B1` 跑完一半的 job 時，`run_s` 只累積了 26%。所以不能用「跑完幾成 job」線性外推。
- **ETA**〔推算〕：
  - `conversation_P_B2`：用 toolagent B1 的「job 序位 → 累積 `run_s`」曲線外推，總共約 110,500–113,900 執行緒秒。約 **01:25** 結束。
  - `conversation_M_B2`：假設 M／P 每 job 比例和 toolagent B1 一樣（1.91 倍），約 161,000 執行緒秒、約 75 分鐘。
  - toolagent 兩個 B2 block：用 B1 的每 job 速率 × B2 倍數 f。f 在 TRX40 的 Mooncake 上 NOT_MEASURED。
  - 剩下約 668,000 ＋ 431,800 ×（f − 1）執行緒秒，用 35.8 的並行度：
    - f＝1.0 → 約 **06:00**；
    - f＝1.5 → 約 **07:40**；
    - f＝2.5（acvlab Bailian 的倍數）→ 約 **11:00**。
  - 之後 collect＋核對＋commit 另外要約 10–20 分鐘〔推算〕。
  - `toolagent_P_B2` 跑到一部分時（〔推算〕約 03:00 之後），可以用同樣的曲線和 `toolagent_P_B1` 比，量出 f，再更新這裡。
- **接下來**：照 21:15 的計畫，逐 block 核對；全部完成後 `collect --plans p4_b1 p4_b2`，commit／push，通知 icpe-data。

## 2026-11-06T21:15+08:00 — **sa_p4 完成**（48 個 block、1,004,366 個 job、1,434,342 列，已進 git）；Mooncake 對拍 0 差；**6 個 Mooncake P4 block 21:03:52 開跑**

**sa_p4**
- **佇列**：15:16:47 → 20:58:03，48 個 block 都 rc=0，沒有 FAILED／KILLED，三個守門程式都沒觸發。log `$ROOT/runs/queue-sa_p4-rust-w36-20261106-151647.log`。
- **逐 block 核對**，48／48 通過：
  - `kept_jobs`＝`count`（run `20261106-150924-icpe-trx-count-sa_p4`），合計 1,004,366。
  - `unmatched_cells=0`、`py_fallback_jobs=0`、exit 0、`req/` 0 個檔。
  - `rust_timing.jsonl` 的筆數＝`kept_jobs`。
- **`collect --plans sa_p4_b1 sa_p4_b2`**：
  - `sa_p4_b1` 24 個 block、727,272 列；`sa_p4_b2` 24 個 block、707,070 列。
  - 合計 1,434,342 列，和 15:15 那節估 npz 時數的 view 數相同。
  - 每個檔都檢查過：
    - `run_id` 單一值，且是該 block 的 run。
    - `host`＝`acvlab-TRX40-AORUS-MASTER`，`code_sha`＝`3280b1c8799b+rs62b0932b`。
    - `ts` 沒有空值。
    - `noise_model`／`noise_draw` 都和 block 名稱一致。
- **程式**：block 的 context 記的 git HEAD 有 4f5c831（1 個）、cdea0d4（38 個）、afc6e76（1 個）、313a9af（8 個）。`git diff 4f5c831 313a9af -- code` 是空的，中間只改了 STATUS 檔。
- **時間**：sweep 時間每個 block 20–1,059 s；`sa_p4_b1` 合計 10,701 s，`sa_p4_b2` 合計 9,758 s。
  - `rust_timing` 1,004,366 筆：`run_s` 中位數 0.726 s、p90 0.985 s、最長 2.270 s。
- **記憶體**（677 筆）：MemAvailable 最低 47.3 GiB（48,435 MiB）；PSS 合計最高 3.3 GiB；swap 790–947 MiB，沒有超過開跑時的量；load1 最高 43.8。/ssd4 剩 547–548 GiB。
- **檔案大小**：48 個 xz（preset 9）合計 101 MB，36 個超過 1 MiB，最大 4.5 MB（`sa_p4_b1/semianalysis_N72_s3_P_B1`）。
  - CLAUDE.md 寫進 git 的檔要 < 1 MB。git 是資料到 acvlab 和分析 session 的唯一管道。
  - **使用者決定**（本 session，11/6 約 21:10，AskUserQuestion）：照原樣 commit。

**Mooncake 的雜訊對拍**（`$ROOT/tools/mcheck.sh 4f5c831`，19:15:35–19:20:39）
- 參數：`--traces toolagent --rules R1 R2 --cpu 8 --policies LRU O-BEL O-PF O-ALL --slows ssd --modes cake --gpu-gib 111.0625 --gpu-write wb --gpu-policies LRU TTLB:10 --pf-target cpu --noise-models P M --noise-lambdas 0.3 --noise-draws B1 B2 --noise-seeds 0`，`--workers 8`。
- Rust：run `20261106-191535-icpe-trx-mc_noise-rust-4f5c831`，19 s，`py_fallback_jobs=0`。
- Python：run `20261106-191535-icpe-trx-mc_noise-py-4f5c831`，285 s。
- `next_sim_cmp.py`（compare run `20261106-192039-icpe-trx-compare-mc_noise-4f5c831`）：96 列 × 41 欄，鍵含 4 個雜訊欄，不同的格子 0，**CMP PASS**。

**merge 和對拍**
- merge `origin/icpe-next-data` @ 58fdfae（STATUS_DATA 19:20 的最終清單）→ **313a9af**。只改了文件；`code/` 和 4f5c831 相同。
- 三項對拍（`verify_rust2.sh 313a9af`，21:01:05–21:03:00；compare run `20261106-210300-icpe-trx-compare-313a9af`）都是 0 個不同：
  - (a) G0：128,064 格，IDENTICAL。
  - (b1) xc_sa 對 icpe-data：1,280 格。
  - (b2) xc_sa 對 TRX40 的 Python：CMP PASS。
  - (c) `sa_N16_s1` 整個 block：293,760 格。
  - `code_sha`＝`3280b1c8799b+rs62b0932b`。

**Mooncake P4 開跑**
- 指令：`run --plans 'p4_b1:toolagent_*' 'p4_b2:conversation_*,toolagent_*' --engine rust --workers 36`，21:03:52，pgid 2231622。
  - log `$ROOT/runs/queue-p4mc-rust-w36-20261106-210352.log`。
  - 守門程式：memwatch 6,144、swapguard 2,933、diskguard 50 GiB。
  - 開跑前：MemAvailable 53.8 GiB，swap 790 MiB，/ssd4 剩 542 GiB。
- **第一個 block**（`p4_b1/toolagent_P_B1`，run `20261106-210352-icpe-trx-p4_b1-toolagent_P_B1`）：
  - `kept_jobs=196718`（＝`count`），`unmatched_cells=0`，`py_fallback_jobs=0`。
  - `export RUN_DIR=`（空的），`req/` 0 個檔，`code_sha=3280b1c8799b+rs62b0932b`。
- 〔推算〕約 3 h（acvlab 的 0.374 執行緒秒／job）。TRX40 跑 Mooncake 的速度 NOT_MEASURED。

**接下來**：每個 block 照上面核對；全部跑完後 `collect --plans p4_b1 p4_b2`，只會有這 6 個目錄；檢查後 commit／push，並通知 icpe-data。

**需要別的 session 做什麼**：無。
**需要使用者決定什麼**：無。

## 2026-11-06T19:20+08:00 — sa_p4 跑到 38／48；**使用者決定：sa_p4 跑完後，TRX40 接 acvlab P4 裡還沒開始的 Mooncake block**（清單等 icpe-data 定）

**依據**
- icpe-main 的訊息（11/6 約 19:10）轉述使用者在 icpe-main 的話：「那帮我在 RTX 上面继续跑 P4 的任务，可以直接交给他。」
  - 哪些 block 搬由 icpe-data 定；CacheWise 一律留在 acvlab；同一個 block 不能兩台都跑。
- **使用者在本 session 確認**（AskUserQuestion，11/6 約 19:15）：**只接 Mooncake**。
  - 這改了 11/6 早上「TRX40 只做 SemiAnalysis」的決定，也把我寫 results 的範圍擴大到這幾個 Mooncake block（`results/icpe_next/b/p4_b1|p4_b2/<block>/`）。
  - Bailian、TraceLab 不搬：TRX40 上沒有這兩份資料。

**sa_p4 進度**
- 19:05 時 38／48 個 block rc=0，沒有 FAILED／KILLED。swap 947 MiB 沒變，/ssd4 剩 547 GiB。
- 剩 10 個 B2 block（`N64_s2`–`N72_s3`）。〔推算〕用同一批變體的 B1 時間（519–1,060 s），約 21:00 跑完。

**TRX40 上的 Mooncake 資料**
- `$ROOT/datasets/mooncake/traces` 的 sha256：
  - `conversation_trace.jsonl` `b8cbb061…`、`toolagent_trace.jsonl` `48a2db1a…`
  - 和 STATUS_DATA 11/4 23:55 的值相同。
- 不用搬任何資料。

**job 數**（run `20261106-191245-icpe-trx-count-p4-acvlab`，`count --plans p4_b1 p4_b2`）
- 合計 2,297,276，和 STATUS_DATA 相同。
- 依 STATUS_DATA 18:20，acvlab 還沒開始的 Mooncake block 有 6 個，共 **1,040,346** 個 job：

  | block | job |
  |:--|--:|
  | `p4_b1/toolagent_P_B1` | 196,718 |
  | `p4_b1/toolagent_M_B1` | 148,738 |
  | `p4_b2/conversation_P_B2` | 205,160 |
  | `p4_b2/conversation_M_B2` | 153,870 |
  | `p4_b2/toolagent_P_B2` | 191,920 |
  | `p4_b2/toolagent_M_B2` | 143,940 |

- 〔推算〕用 acvlab 實測的 0.374 執行緒秒／job（STATUS_DATA 18:40），36 workers 約 3 h。TRX40 跑 Mooncake 的速度 NOT_MEASURED。
- 〔推算〕acvlab 照 18:40 的速率，約 22:00 會跑到 `toolagent_P_B1`。所以跳過的設定要在那之前做好。
- **dry-run**：
  - 指令：`run --plans p4_b1:toolagent_P_B1,toolagent_M_B1 p4_b2:conversation_P_B2,conversation_M_B2,toolagent_P_B2,toolagent_M_B2 --engine rust --workers 36 --dry-run`
  - 剛好這 6 個 block。都是 `export RUN_DIR=`（空的），都有 `--jobs-from`，`X1_TRACE_DIR` 是上面的目錄。沒有建立 run 目錄。

**計畫**（清單以 icpe-data 在 STATUS_DATA 寫的為準）
1. sa_p4 跑完 → `collect`、核對、commit／push（照 15:20 那節）。
2. 等 icpe-data 在 STATUS_DATA 寫好 TRX40 的 block 清單，以及 acvlab 佇列（pid 3531487）怎麼跳過這些 block。
3. `git merge --no-ff origin/icpe-next-data`。
   - `git diff 4f5c831 HEAD -- code/p3x code/icpe/p3_fvar.py code/p3_platform.py` 要是空的。不是空的就重做三項對拍。
   - `code_sha` 要等於 acvlab P4 的 `3280b1c8799b+rs62b0932b`。
4. **Mooncake 的雜訊對拍**：G0 本身就是 Mooncake conversation，15:05 在這份程式上已經 0 差，但它不含雜訊。所以另外比一個 Mooncake toolagent 的 B1＋B2 子集：同樣參數在 TRX40 用 Python 和 Rust 各跑一次，要 0 個不同的格子。
5. 照清單用 `--plans p4_b1:<…> p4_b2:<…>` 指定 block，`--engine rust --workers 36`。
   - 守門程式照舊：memwatch 6,144、swapguard 2,933、diskguard 50 GiB。
   - 每個 block 核對：`kept_jobs`＝`count`、`unmatched_cells=0`、`py_fallback_jobs=0`、`req/` 是空的。
6. 跑完 `collect --plans p4_b1 p4_b2`。
   - 只收 TRX40 跑的這幾個 block，`host` 要是 `acvlab-TRX40-AORUS-MASTER`。
   - commit／push 前在這裡寫清單，避免和 acvlab 收的檔重疊。
7. 任何一步失敗或有差，就停下來，把完整錯誤寫在這裡，並回報使用者。

**舊計畫的兩步不重跑**（使用者問的，run `20261106-163455-icpe-trx-oldplan-coverage`）
- 舊計畫的 `sa_main_ttl7`（20 個 block）和 `sens_rep sa_*`（4 個 block）：dry-run 後，逐一拿參數去和已完成的 P1–P3 比。
- 24 個都是某個已完成 block 的子集，0 個沒涵蓋：
  - `sa_main_ttl7` → `sa_p1 *_ttl`
  - nvme807、load → `sa_p2`
  - wt、pfgpu → `sa_p3`
- 唯一沒涵蓋的 `sa_t128k` 已經在 11:22 跑完。
- 注意：比的是參數。「參數是子集，所以 job 也是子集」是〔推論〕。

**需要別的 session 做什麼**：**icpe-data**：在 STATUS_DATA 寫 TRX40 的 block 清單，以及 acvlab 佇列跳過這些 block 的做法（附 commit）。已傳訊息。
**需要使用者決定什麼**：無（已決定，見上）。

## 2026-11-06T15:20+08:00 — merge icpe-data 的「P4 不存 npz」（f828e79）；**sa_p4 開跑**（48 個 block、1,004,366 個 job、36 workers）

**依據**：STATUS_DATA 15:25（@ f828e79，git 上讀過）。PREREG 的 P4 只要 seed 間的中位數、最小–最大、平均±標準差，`next_sim_analyze.py` 只讀 sweep CSV，所以不需要逐請求 npz。P4 的 block 帶 `no_npz=True` → `RUN_DIR` 是空字串，x1_sim 不存 npz。

**做完什麼**
- `git merge --no-ff origin/icpe-next-data` → **4f5c831**。這次只改了 `code/icpe/next_b_queue.py`（6 行）。
  - `git diff 424d390 HEAD -- code/p3x code/icpe/p3_fvar.py code/p3_platform.py` 是空的：模擬程式和 15:15 那節對拍過的相同，所以沒有重做對拍。
  - `git diff origin/icpe-next-data HEAD -- code/` 也是空的。
  - 我查過 `RUN_DIR` 在 `x1_rust.py`、`p3_fvar.py`、`p3_platform.py` 都沒有用到，x1_sim 只用在 `_job_rows` 存 npz（`if rd:`）。
- **dry-run**：48 個 block 都是 `export RUN_DIR=`（空的），都有 `--jobs-from` 和 `--workers 36 --engine rust`。
- **開跑**：`run --plans sa_p4_b1 sa_p4_b2 --engine rust --workers 36`，15:16:47，pgid 2121436，log `$ROOT/runs/queue-sa_p4-rust-w36-20261106-151647.log`。
  - 守門程式：memwatch（6,144 MiB）、swapguard（2,933 MiB），另加 **diskguard**（新的機器工具 `$ROOT/tools/diskguard.sh`，/ssd4 剩不到 50 GiB 就停）。
  - 開跑前：swap 947 MiB、MemAvailable 57.8 GiB、/ssd4 剩 548 GiB。
- **第一個 block**（`sa_p4_b1/semianalysis_N16_s1_P_B1`，run `20261106-151647-icpe-trx-sa_p4_b1-semianalysis_N16_s1_P_B1`）：
  - `jobs_from=` 那行：rows 287、cells 287、`kept_jobs=8323`（＝`count`）、`dropped_jobs=39893`、`unmatched_cells=0`。
  - `py_fallback_jobs=0`，`code_sha=3280b1c8799b+rs62b0932b`。
  - `req/` 是空的（0 個檔），run 目錄 1.6 MB。
- 〔推算〕時間：1,004,366 job × 0.578 s（`sa_p3 _wt` 的 `run_s` 中位數）÷ 36 ≈ 4.5 h，約 19:45 跑完。B2 的速度還沒量。

**接下來**：每個 block 跑完核對 `kept_jobs`＝`count`、`unmatched_cells=0`、`py_fallback_jobs=0`。全部跑完後 `collect`，檢查每個檔的 `run_id`／`host`／`code_sha`，再 commit／push。任何 block 失敗就停下來，把完整錯誤寫在這裡。

**需要別的 session 做什麼**：無。
**需要使用者決定什麼**：無。

## 2026-11-06T15:15+08:00 — merge icpe-next-data（v5＋`--jobs-from`＋P4 plan）、對拍全部 0 差、count／dry-run 正確；**sa_p4 沒有開跑：逐請求 npz 估計約 1,164 GiB，/ssd4 只剩 548 GiB** → 使用者選「請 icpe-data 加不寫 npz 的開關」

**做完什麼**
- 依據 STATUS_DATA 15:10（@ 7947e95）：「icpe-trx 可以開始 sa_p4」。
- `git merge --no-ff origin/icpe-next-data` → **424d390**（無衝突）。
  - `git diff origin/icpe-next-data HEAD -- code/` 是空的；`git diff 1bf35bd HEAD -- code/p3x code/icpe/p3_fvar.py code/p3_platform.py` 也是空的。
  - main.rs 和 9167f11 相同，Rust 沒有重編（src `62b0932b…`）。`code_sha`＝`3280b1c8799b+rs62b0932b`，和 icpe-data 的 G0 相同。
- **對拍**（`verify_rust2.sh 424d390`＋`b2check.sh 424d390`，15:05:11–15:08:54），全部不同的格子 0，`py_fallback_jobs=0`：
  - G0：128,064 格。
  - xc_sa：對 icpe-data 的 Python 1,280 格；對 TRX40 的 Python，CMP PASS。
  - `sa_N16_s1` 整個 block：293,760 格。
  - B2 子集 Python 對 Rust：24 列 × 44 欄，CMP PASS。
- **count**：run `20261106-150924-icpe-trx-count-sa_p4`。48 個 block；`sa_p4_b1` 509,256、`sa_p4_b2` 495,110，**合計 1,004,366**，和 STATUS_DATA 相同。
- **dry-run**：`run --plans sa_p4_b1 sa_p4_b2 --engine rust --workers 36 --dry-run`。48 個 block 都有 `--jobs-from <run>/jobs.csv` 和 `--engine rust`，沒有建立 run 目錄。

**為什麼沒開跑：磁碟**
- 每個 job 跑完，`x1_sim._job_rows` 會在 `$RUN_DIR`（佇列設成 `<run>/req`）存一個逐請求 TTFT 的 npz。檔名含雜訊參數，所以每個 job 一個檔，不會互相覆蓋。
- 實測：`sa_p1` 的 `_ttl` block 每個 job 1 個 view，**每個 view 700–922 kB**，依變體而定。`sa_p3` 的 `sa_N72_s3_wt` 的 run 目錄 6.3 GiB 幾乎全是 `req/`。`$ROOT/runs` 目前共 400 GiB。
- 〔推算〕把 P4 每個 job 的 view 數（用同一個 `jobs_for`＋`--jobs-from` 的過濾算）乘上同一變體的每 view 位元組。
  - run `20261106-151238-icpe-trx-p4-npz-est`，程式 `$ROOT/tools/p4_npz_est.py`，只讀程式、不跑模擬。
  - 結果：1,434,342 個 view，**約 1,250 GB（1,164 GiB）**。
  - `/ssd4` 剩 587.8 GB（548 GiB，`df -B1`）→ **放不下**，約跑到 4 成就會滿。/ssd4 是共用磁碟，滿了也會影響別人。
- 使用者決定（本 session，11/6 約 15:12，AskUserQuestion）：**請 icpe-data 加一個讓 P4 不寫 npz 的開關**。npz 是 `x1_boot.py` 成對 bootstrap 用的，所以先請 icpe-data 確認 P4 的分析要不要用；不用的話，sa_p4 只寫 sweep CSV。TRX40 在那之前不跑 P4。

**需要別的 session 做什麼**
- **icpe-data**：
  1. 確認 PREREG 的 P4 分析需不需要逐請求 npz。必要時問 icpe-main／icpe-ana。
  2. 不需要的話，在 `next_b_queue.py` 加開關，讓 P4 的 block 不設 `RUN_DIR`、或不寫 npz；需要改 `x1_sim` 的話請 icpe-sim 改。
  3. 好了在 STATUS_DATA 寫 commit。
- **acvlab 的 P4 也要看一下**：`p4_b1`＋`p4_b2` 有 2,297,276 個 job。TraceLab 每個 job 約 4.4 MB（STATUS_DATA 11/4 的數字），應該也要先算磁碟。

**需要使用者決定什麼**：無（已決定，見上）。

## 2026-11-06T14:30+08:00 — 先 merge v5（B2 進 Rust，icpe-next-sim 4356484，程式 4a4baa4）；對拍全部 0 差，B2 在 TRX40 走 Rust

**依據**：STATUS_SIM 14:25（@ 4356484，git 上讀過）「需要 icpe-trx 做什麼：merge 後重做 G0；venv 不是 3.12 的話 B2 會退回 Python；用 `--engine rust` 跑 xc_sa 和 Python 版逐格比」；icpe-sim 也直接傳了訊息。sa_p4 還要等 `--jobs-from`。TRX40 現在是閒的，所以先把 v5 併進來驗證，提早發現 TRX40 特有的問題（例如編譯失敗、B2 退回 Python）。

**做完什麼**
- TRX40 的 venv 是 **Python 3.12.15**（`$ROOT/venv/bin/python -I -c 'import sys;print(sys.version)'`），所以 B2 會交給 Rust。
- `git merge --no-ff origin/icpe-next-sim` → **9167f11**（無衝突）。`git diff 4356484 HEAD -- code/p3x code/icpe/p3_fvar.py code/p3_platform.py` 是空的。
- Rust 自動重新編譯：src_sha256 `62b0932b…`，bin `x1fast-work/build-62b0932bb6cd6d9d/fast/x1fast3`（bin_sha256 `576b78a0…`）。新的 `code_sha`＝`fd0dccbf7093+rs62b0932b`。
- **三項對拍**（`verify_rust2.sh 9167f11`，14:21:50–14:24:05；compare run `20261106-142405-icpe-trx-compare-9167f11`）：
  - (a) G0：128,064 格，不同 0。
  - (b1) xc_sa 對 icpe-data 的 Python：1,280 格，不同 0。
  - (b2) xc_sa 對 TRX40 的 Python：32 列 × 40 欄，不同 0，CMP PASS。
  - (c) `sa_N16_s1` 整個 block：293,760 格，不同 0。
  - 都是 `py_fallback_jobs=0`。
- **B2 對拍**（新的機器工具 `$ROOT/tools/b2check.sh`）：
  - 參數照 SIM_CHANGES 20 的 SemiAnalysis B2 子集：`--traces nb:semianalysis_N16_s1 --rules nb --cpu 8 --policies LRU O-BEL O-PF O-ALL --slows ssd --modes cake --gpu-gib 0 111.0625 --gpu-write wt --gpu-policies TTLB:10 --pf-target cpu --noise-models P M --noise-lambdas 0.3 --noise-draws B2 --noise-seeds 0`，`--workers 8`。
  - Rust：run `20261106-142406-icpe-trx-sa_b2-rust-9167f11`，3 s，`py_fallback_jobs=0`，stdout 沒有 `noise_B2_py`。
  - Python：run `20261106-142406-icpe-trx-sa_b2-py-9167f11`，91 s。
  - `next_sim_cmp.py`（compare run `20261106-142540-icpe-trx-compare-b2-9167f11`）：24 列 × 44 欄，鍵含 4 個 noise 欄，不同的格子 0，**CMP PASS**。
- 記憶體：MemAvailable 最低 53.8 GiB，swap 959 MiB 沒變。

**還沒做**：sa_p4 仍然等 `--jobs-from`。icpe-sim 預計 15:30–16:15 push；之後 icpe-data 要 merge 進 icpe-next-data、做 G0。到時我 merge icpe-next-data、重做上面的對拍，再 dry-run／count，然後開跑（見 14:20 那節）。

**需要別的 session 做什麼**：無。
**需要使用者決定什麼**：無。

## 2026-11-06T14:20+08:00 — 接下 P4 的 SemiAnalysis（`sa_p4_b1`、`sa_p4_b2`），36 workers；等 icpe-data 通知

**依據**
- STATUS_DATA 14:05（@ 8f1ef80，git 上讀過）：使用者決定 P4 照 PREREG 全跑；`sa_p4_b1`（509,256 job）和 `sa_p4_b2`（495,110 job）交給 TRX40，共 48 個 block、1,004,366 個 job。icpe-data 也直接傳了訊息。
- **使用者決定**（本 session，11/6 約 14:15，AskUserQuestion）：TRX40 接 sa_p4；用 **36 workers**，和 P1–P3 相同（icpe-data 建議的是 32）。

**開跑的條件**（都還沒到）
1. icpe-sim push v5（B2 進 Rust）和 `x1 sweep --jobs-from`。
2. icpe-data merge 進 `icpe-next-data`、做完 G0，然後在 STATUS_DATA 寫好。

**我的步驟**
1. `git merge --no-ff origin/icpe-next-data`。
2. Rust 在 TRX40 重新編譯，重做三項對拍（G0、xc_sa 對 Python、`sa_N16_s1`）。如果 STATUS_SIM 對 v5／`--jobs-from` 指定了 TRX40 要做的驗證，照它加做。
3. `run --plans sa_p4_b1 sa_p4_b2 --engine rust --workers 36`：先 `--dry-run` 和 `count`，確認 48 個 block、1,004,366 個 job；memwatch（6,144 MiB）和 swapguard（2,933 MiB）都開。
4. 任何一步失敗或對拍有差，就停下來，把完整錯誤寫在這裡，回報使用者。

**需要別的 session 做什麼**：icpe-data 在 STATUS_DATA 寫「icpe-trx 可以開始 sa_p4」時，請附上 icpe-next-data 的 commit。
**需要使用者決定什麼**：無。

## 2026-11-06T13:45+08:00 — **`sa_p3` 的 12 個 `_wt` 完成 → TRX40 的 SemiAnalysis 偏離 6（P1–P3）＋`sa_t128k` 全部完成**；目前 TRX40 沒有佇列在跑

**做完什麼**
- 佇列 `run --plans sa_p3 --only 'sa_N*_wt' --engine rust --workers 36`：12 個 block 都 rc=0，13:12:59 → 13:37:22（24 分鐘）。log `$ROOT/runs/queue-sa_p3_wt-rust-w36-20261106-131259.log`；沒有 `*_KILLED`，swapguard log 是空的。
- `collect --plans sa_p3` → `results/icpe_next/b/sa_p3/<變體>_wt/`：12 個 block，各 7,344 列（job 6,174），合計 74,088 job、88,128 列。之前收的 12 個 `_pfgpu` 沒有變動。
  - 12 個檔都 `run_id` 單一值，等於各自的 run（`20261106-131259…sa_N16_s1_wt` 到 `20261106-133506…sa_N72_s3_wt`）。
  - `host` 都是 `acvlab-TRX40-AORUS-MASTER`，`code_sha` 都是 `fd0dccbf7093+rse9497446`，`gpu_write` 都是 `wt`，`ts` 沒有空值。
  - 12 個 run 都 `py_fallback_jobs=0`，`rust_timing.jsonl` 各 6,174 筆。
- 程式：第一個 block 的 context 記 git HEAD 2b73df1，其餘 11 個記 a1608f2。中間的 commit 只改了 STATUS_TRX（`git diff --stat 2b73df1 a1608f2` 只有這一個檔），所以 12 個 block 的程式相同，也和三項對拍相同。
- sweep 每個 block 88.8–121.8 s，合計 1,296 s。`rust_timing` 74,088 筆：`run_s` 中位數 0.578 s、p90 0.902 s、最長 1.851 s。
- 記憶體（49 筆，每 30 s）：PSS 最高 2.8 GiB，MemAvailable 最低 53.5 GiB；swap 959 MiB，整段沒變；load1 最高 39.4。
- 早上失敗的 run `20261106-093803-icpe-trx-sa_p3-sa_N16_s1_wt`（舊程式）不收；同一個 block 這次用新程式跑完。

**TRX40 的 SemiAnalysis 總表**（全部在 git 裡）

| 目錄 | block | 列 | code_sha |
|:--|--:|--:|:--|
| `sa_main` | 12 | 88,128 | （Python，見 11/5 各節） |
| `sa_p1`（`_ttl`、`_trunc`） | 24 | 139,968 | `a27559138492+rs43dbe828` |
| `sa_p2`（`_nvme807`、`_load`） | 24 | 146,880 | `a27559138492+rs43dbe828` |
| `sa_p3` `_pfgpu` | 12 | 15,552 | `a27559138492+rs43dbe828` |
| `sa_p3` `_wt` | 12 | 88,128 | `fd0dccbf7093+rse9497446` |
| `sens_rep/sa_t128k` | 1 | 1,632 | `a27559138492+rs43dbe828` |

- `_wt` 用的是修正後的程式，其他用修正前的程式。照 STATUS_SIM 10:10／13:15，wb＋GPU 的格子在兩版之間不變；本機的三項對拍（13:15 那節）也顯示 wb 的 xc_sa 和 `sa_N16_s1` 在新程式下 0 差。
- 不解讀上限。P4（雜訊）不在 TRX40 的工作範圍。

**需要別的 session 做什麼**：無。
**需要使用者決定什麼**：無。

## 2026-11-06T13:15+08:00 — merge wt 修正（icpe-next-sim 776486c，程式＝7193a2d）；三項對拍全部 0 差；12 個 `_wt` 開跑

**依據**：STATUS_SIM 13:15（@ 776486c，git 上讀過）「需要 icpe-trx 做什麼：merge 後重做 G0；main.rs 變了會自動重編；用 `--engine rust` 跑 xc_sa 和 Python 版逐格比對，之後再跑 12 個 `_wt` block」；icpe-sim 也直接傳訊息請我 merge 776486c。icpe-data 還沒把它併進 `icpe-next-data`（13:09 查過），所以直接 merge icpe-next-sim。

**做完什麼**
- `git merge --no-ff origin/icpe-next-sim` → **2b73df1**（無衝突）。`git diff 776486c HEAD -- code/p3x code/icpe/p3_fvar.py code/p3_platform.py` 是空的。
  - 一起進來的還有準備快取 c7ef740（`X1_PREP_CACHE` 預設 1，快取寫在 `X1_RUST_WORK`＝`$ROOT/x1fast-work`）。`code/icpe/next_b_queue.py` 沒變（icpe-next-data 的 23865ee 沒併）。
- Rust 自動重新編譯：src_sha256 `e9497446…`，bin `x1fast-work/build-e94974467f6f608c/fast/x1fast3`（bin_sha256 `f4c5e7d7…`）；新的 `code_sha`＝`fd0dccbf7093+rse9497446`。
- **三項對拍**（`verify_rust2.sh 2b73df1`，`--engine rust --workers 32`，13:10:30–13:12:47，memwatch／swapguard 有開；compare run `20261106-131244-icpe-trx-compare-2b73df1`）：

| 項目 | run | 對照 | 結果 |
|:--|:--|:--|:--|
| (a) G0 | `20261106-131030-icpe-trx-g0-rust-2b73df1`（26.5 s） | H200 `results/icpe_h200/t1/p01/sim_V1/x1/x1_sweep.csv.gz` | 4,416 列，128,064 格，不同 0 |
| (b1) xc_sa | `20261106-131030-icpe-trx-xc_sa-rust-2b73df1`（17.6 s） | `results/icpe_next/b/xcheck/xc_sa`（icpe-data 的 Python） | 32 列，1,280 格，不同 0 |
| (b2) xc_sa | 同上 | TRX40 的 Python xc_sa（run `20261105-105555-icpe-trx-xcheck-xc_sa-54fa2d8`），`next_sim_cmp.py` | 32 列 × 40 欄，不同 0，CMP PASS |
| (c) sa_N16_s1 整個 block | `20261106-131030-icpe-trx-sa_N16_s1-rust-2b73df1`（89.9 s） | `results/icpe_next/b/sa_main/sa_N16_s1` | 7,344 列，293,760 格，不同 0 |

  - 三個 run 都 `py_fallback_jobs=0`。(b)(c) 都是 wb＋GPU，照 STATUS_SIM 不受這次修正影響，結果也的確沒變。
- **12 個 `_wt` 開跑**：`run --plans sa_p3 --only 'sa_N*_wt' --engine rust --workers 36`。先 `--dry-run` 確認剛好選到 12 個 `_wt` block，沒有建立任何 run 目錄。13:12:59 開始，pgid 2091475，log `$ROOT/runs/queue-sa_p3_wt-rust-w36-20261106-131259.log`。memwatch（6,144 MiB）和 swapguard（2,933 MiB）都有開。
  - 開跑前 swap 959 MiB、MemAvailable 57.7 GiB。
  - 舊的失敗 run `20261106-093803-icpe-trx-sa_p3-sa_N16_s1_wt` 不收。

**需要別的 session 做什麼**：無。
**需要使用者決定什麼**：無。

## 2026-11-06T11:22+08:00 — `sa_p3` 的 12 個 `_pfgpu`＋`sens_rep sa_t128k` 完成；`_wt` 等 icpe-sim 的修正（7193a2d）

**使用者決定（本 session，11/6 約 11:05，AskUserQuestion）**：等 icpe-sim 修，期間先用現在這版 Rust 跑 `_pfgpu` 和 `sa_t128k`，出錯就停；修好後 merge → 重做三項對拍 → 跑 12 個 `_wt`。已直接通知 icpe-sim（SendMessage）。

**icpe-sim 的回覆**（STATUS_SIM 10:10 @ 88a81e3，已在 git 上核對）：
- Python 在同一個 job 也出錯（O-PF、CPU 8、wt、TTLB:10；`demote()` 的 `self.sb[s][-1]` → IndexError）→ 模型的邊界情況，不是 Rust 獨有。
- 原因：新請求沒有完整 block 時，回合結束只做 `cpu_set.discard(s)`，順序結構（LRU／ARC／S3FIFO）留下過期項目。SemiAnalysis 每個變體 198 個這種請求；`semianalysis_N16_s1_t128k` 125 個。
- **wb＋GPU 不會發生**（GPU 回合開始時已 `_order_remove`）→ `sa_main`、`sa_p1`、`sa_p2`、`_pfgpu`、`sa_t128k`、三項對拍不受影響。我查過本機每個 SA run 的 cmd.sh：除了失敗的 `sa_p3/sa_N16_s1_wt`，全部是 `--gpu-gib 111.0625 --gpu-write wb`。
- 修法：使用者在 icpe-sim 選 A（兩個回合結束都 `order_remove`，兩個引擎都改），commit 7193a2d；驗證 run `20261106-111012-icpe-sim-g1-7193a2d`（最後跑整個 `sa_N16_s1_wt` Rust block），預計 12:30，過了寫 STATUS_SIM。`main.rs` 有改 → TRX40 merge 後會重新編譯。

**做完什麼**
- 佇列：`run --plans sa_p3 sens_rep --only 'sa_N*_pfgpu' 'sa_t128k' --engine rust --workers 36`（先 `--dry-run` 確認只選到這 13 個 block；`'*_pfgpu'` 會選到 sens_rep 其他資料集的格子，所以不用）。11:06:08 → 11:19:19，log `$ROOT/runs/queue-sa_pfgpu_t128k-rust-w36-20261106-110608.log`；程式同三項對拍（git HEAD 98b0d2f，`code/` 和 eeaecef 相同）。
- `results/icpe_next/b/sa_p3/<變體>_pfgpu/`：12 個 block，各 1,296 列（job 1,296），sweep 45.9–53.8 s，合計 610 s。
- `results/icpe_next/b/sens_rep/sa_t128k/`：run `20261106-111913-icpe-trx-sens_rep-sa_t128k`，1,372 job、1,632 列，sweep 4.1 s（`--traces nb:semianalysis_N16_s1_t128k --gpu-gib 111.0625 --gpu-write wb --pf-target cpu`）。
- 13 個檔都 `run_id` 單一值、`host`＝`acvlab-TRX40-AORUS-MASTER`、`code_sha`＝`a27559138492+rs43dbe828`，13 個 run 都 `py_fallback_jobs=0`。
- 記憶體（27 筆）：PSS 最高 3.2 GiB，MemAvailable 最低 53.2 GiB；swap 959 MiB；load1 最高 32.5。

**還沒跑**：`sa_p3` 的 12 個 `_wt`（74,088 job）。失敗的 run `20261106-093803-icpe-trx-sa_p3-sa_N16_s1_wt` 不收。

## 2026-11-06T09:45+08:00 — `sa_p2` 完成；**`sa_p3` 第一個 block（`sa_N16_s1_wt`）失敗：Rust 引擎 panic（`src/main.rs:667`）→ 佇列停了**

**`sa_p2` 完成**
- 24 個 block 都 rc=0，08:53:56 → 09:38:03（44 分鐘）→ `results/icpe_next/b/sa_p2/`（commit dc8892c，12 MB）：12 個 `_nvme807` 各 2,448 列、12 個 `_load` 各 9,792 列；合計 124,416 job、146,880 列。每個檔 `run_id` 單一值、`host`＝`acvlab-TRX40-AORUS-MASTER`、`code_sha`＝`a27559138492+rs43dbe828`；24 個 run 都 `py_fallback_jobs=0`。
- sweep 每個 block 30.3–171.0 s，合計 2,251 s。`rust_timing.jsonl` 124,416 筆：`run_s` 中位數 0.619 s、p90 0.916 s、最長 1.919 s。
- 記憶體（88 筆）：PSS 最高 3.0 GiB，MemAvailable 最低 53.3 GiB；swap 959 MiB；load1 最高 40.8。

**`sa_p3` 失敗**（完整錯誤；沒有重試）
- run `20261106-093803-icpe-trx-sa_p3-sa_N16_s1_wt`，exit 1。git HEAD b68d034（程式＝eeaecef，sim 1304345，`bin_sha256=ceaf3065…`，和三項對拍相同）。
- 指令（cmd.sh 最後一行的尾段）：`… x1 sweep --traces nb:semianalysis_N16_s1 --rules nb --cpu 8 16 32 64 128 256 512 1024 111.0625 --policies LRU LRU2 ARC S3FIFO GDSF PENS FREQ2Q LRU+NONE LRU+2HW LRU+2HA LRU+RPF O-BEL O-BELS O-DROP O-NOW O-PF O-ALL --slows ssd b32 nfs --modes cake --gpu-gib 111.0625 --gpu-write wt --gpu-policies LRU BEL TTLA:0.3 TTLA:10 TTLA:300 TTLB:0.3 TTLB:10 TTLB:300 --pf-target cpu --workers 36 --engine rust`
- 跑到 400／6,174 個 job（stdout 最後一行 `[400/6174 8s] … O-ALL …`；`rust_timing.jsonl` 400 筆）。
- stderr.log 全文：
  ```
  session groups: 550 groups for 2090 sessions, largest 153
  x1fast serve: trace_load_s 0.649105179 threads 36

  thread '<unnamed>' (2052850) panicked at src/main.rs:667:41:
  index out of bounds: the len is 0 but the index is 18446744073709551615
  note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace

  thread '<unnamed>' (2052865) panicked at src/main.rs:667:41:
  index out of bounds: the len is 0 but the index is 18446744073709551615
  Traceback (most recent call last):
    File ".../code/icpe/p3_fvar.py", line 163, in <module>
      main()
    File ".../code/icpe/p3_fvar.py", line 137, in main
      XS.main()
    File ".../code/p3x/x1_sim.py", line 2740, in main
      {"selftest": cmd_selftest, "sweep": cmd_sweep, "analyze": cmd_analyze}[a.cmd](a)
    File ".../code/p3x/x1_sim.py", line 2347, in cmd_sweep
      _rust_sweep(JR, a.workers, put)
    File ".../code/p3x/x1_sim.py", line 2401, in _rust_sweep
      raise RuntimeError(f"x1fast（常駐，{ev[1].tdir}）提早結束，exit {ev[1].p.wait()}（錯誤訊息在 stderr）")
  RuntimeError: x1fast（常駐，/ssd4/hungwei/icpe-next/x1fast-work/trace-7cfa6556228f0469c7dc5113）提早結束，exit -6（錯誤訊息在 stderr）
  ```
- 佇列 log：`BLOCK FAILED sa_p3/sa_N16_s1_wt rc=1` → 佇列 09:38 結束（`sa_p3` 其餘 23 個 block 沒跑）。保護沒觸發，沒有殘留行程。
- `src/main.rs:667` 是 `demote()` 裡開預取時的 `let t = self.tr.next_access(bl[bl.len() - 1], self.cur);`：`bl`（`self.sbl(s)`）是空的，`len()-1` 溢位。只看到這一行，沒有找出是哪個 job（哪個策略／TTL／CPU 大小）；Python 引擎在同一個 job 會不會出錯 NOT_MEASURED。
- **需要 icpe-sim 做什麼**：看 `--gpu-write wt` 時 `demote()` 為什麼會拿到沒有 block 的 session（重現：上面的指令、`nb:semianalysis_N16_s1`）。程式歸 icpe-sim，我不改。
- **需要使用者決定什麼**：`sa_p3` 怎麼繼續（已問）。

## 2026-11-06T08:58+08:00 — **`sa_p1` 完成**（24 個 block，Rust，36 workers）；`sa_p2` 在跑

- 佇列 `queue-sa_p123-rust-w36-20261106-080648.log`：`sa_p1` 24 個 block 都 rc=0，08:06:48 → 08:53:56（47 分鐘）。run_id＝`20261106-HHMMSS-icpe-trx-sa_p1-<block>`（第一個 `…-080648-…-sa_N16_s1_ttl`，最後一個 `…-085152-…-sa_N72_s3_trunc`）。
- collect → `results/icpe_next/b/sa_p1/<block>/`（commit c800ea8，9.2 MB）：
  - 12 個 `_ttl`：各 6,480 列（job 6,480）；12 個 `_trunc`：各 5,184 列（job 4,824）。合計 135,648 job、139,968 列。
  - 每個檔：`run_id` 單一值、`host`＝`acvlab-TRX40-AORUS-MASTER`、`code_sha`＝`a27559138492+rs43dbe828`；24 個 run 都 `py_fallback_jobs=0`。
- 時間：sweep 每個 block 69.0–132.4 s，合計 2,467 s。`rust_timing.jsonl`（135,648 筆＝job 數）：Rust 引擎每 job `run_s` 中位數 0.629 s、p90 0.940 s、最長 1.924 s；從派出到讀回中位數 0.637 s、最長 2.288 s。
- 記憶體（mem.log 08:06:48–08:53:56，93 筆）：行程群組 PSS 最高 2.9 GiB，MemAvailable 最低 53.4 GiB；swap 958–959 MiB；load1 最高 40.2。保護都沒觸發。
- 下一個：`sa_p2/sa_N16_s1_nvme807` 08:53:56 開跑。〔算術〕照 `sa_p1` 的速度（約 48 job/s，含每個 block 的準備時間），`sa_p2`＋`sa_p3` 共 214,056 job 約 75 分鐘 → 約 10:10 跑完；之後 merge 23865ee、跑 `sens_rep --only sa_t128k`。
- 不解讀上限（分析由新的分析 session 做）。

## 2026-11-06T08:12+08:00 — **三項對拍都 0 個不同**；Rust 佇列 `sa_p1 sa_p2 sa_p3` 開跑（36 workers，使用者決定）

**對拍**（STATUS_DATA 08:10 第 3 步；`$ROOT/tools/verify_rust2.sh eeaecef`；`--engine rust --workers 32`；比對 run `20261106-080609-icpe-trx-compare-eeaecef`）
- 執行檔：`src_sha256=43dbe828…`、`bin_sha256=ceaf3065f96e46e1fefb834ca3586806771cace604a75456ca46a5c3922c197b`（TRX40 用 rustc 1.99.1 自己編）；code_sha `a27559138492+rs43dbe828`（和 acvlab 的 G0 rust run `20261106-074940-icpe-data-g0-rust-merged-1304345` 相同）；三個 run 的 `py_fallback_jobs=0`。
- (a) G0：run `20261106-080343-icpe-trx-g0-rust-eeaecef`，sweep 18.4 s（`time -v` 牆鐘 25.8 s）→ H200 參考：4,416 列 × 29 欄＝**128,064 格，0 個不同**，`G0_RESULT IDENTICAL`。
- (b) xc_sa：run `20261106-080343-icpe-trx-xc_sa-rust-eeaecef`，sweep 1.9 s → `results/icpe_next/b/xcheck/xc_sa`：32 列 × 40 欄＝**1,280 格，0 個不同**；→ TRX40 的 Python xc_sa（run `20261105-105555-icpe-trx-xcheck-xc_sa-54fa2d8`）：`next_sim_cmp.py` **CMP PASS**（欄位、列數相同，不同的格子 0）。
- (c) 整個 `sa_N16_s1` block：run `20261106-080343-icpe-trx-sa_N16_s1-rust-eeaecef`，6,174 job，sweep 90.1 s（`time -v` 牆鐘 1:43.5，最大 RSS 2.9 GB）→ `results/icpe_next/b/sa_main/sa_N16_s1/x1_sweep.csv.xz`：7,344 列 × 40 欄＝**293,760 格，0 個不同**，`G0_RESULT IDENTICAL`。〔算術〕同一個 block 的 Python（40 workers）sweep 3,903 s。
- 記憶體（mem.log，10 筆）：行程群組 PSS 最高 2.7 GiB，MemAvailable 最低 53.4 GiB，swap 958 MiB。

**使用者決定（本 session，11/6 08:0x，AskUserQuestion）**：使用者說「用更多核心，最後 load 90 上下」；我說明 TRX40 只有 48 個硬體執行緒、load 90 不會更快後，使用者選 **36 workers（約 45 CPU）**。所以不是 STATUS_DATA 08:10 的 32。

**開跑**：08:06:48，`env … NEXT_B_TAG=icpe-trx RUSTC=… X1_CARGO=… CARGO_HOME=… RUSTUP_HOME=… X1_RUST_WORK=$ROOT/x1fast-work python -I code/icpe/next_b_queue.py run --plans sa_p1 sa_p2 sa_p3 --engine rust --workers 36`（git HEAD 9a78df5；log `$ROOT/runs/queue-sa_p123-rust-w36-20261106-080648.log`；memwatch 6 GiB、swapguard 2,933 MiB）。第一個 block `sa_p1/sa_N16_s1_ttl`（run `20261106-080648-icpe-trx-sa_p1-sa_N16_s1_ttl`）：stdout 印 `jobs=6480 workers=36 code_sha=a27559138492+rs43dbe828`。
- 每個 plan 跑完：`collect` → commit → push，記 run_id、時間、記憶體、`rust_timing.jsonl`。
- **`sa_t128k`**：icpe-data 在 STATUS_DATA 08:20（23865ee）更正，要跑。佇列已照舊指令開跑，不中斷；`sa_p3` 跑完後 merge 23865ee（只改 `next_b_queue.py`），再跑 `run --plans sens_rep --only sa_t128k --engine rust --workers 36`。`sens_rep` 其餘 4 個 `sa_*` 不跑。

## 2026-11-06T08:06+08:00 — 使用者決定：Rust 跑 SemiAnalysis 的 P1–P3；裝好 rustc 1.99.1、merge 到 9be4925；三項對拍在跑

**使用者決定**
- 在本 session（AskUserQuestion，11/6 07:5x）：「改用 Rust 跑偏離 6」。
- 在 icpe-data（STATUS_DATA 08:10 @ 9be4925，以這節為準）：TRX40 裝 Rust 自己跑 `sa_p1` → `sa_p2` → `sa_p3`；`sa_main_ttl7`、`sens_rep sa_*` **不跑**（理由寫在該節：P2／P3 已涵蓋全部格子）。
- 注意（事實，不是建議）：`sens_rep` 的 5 個 `sa_*` 格裡，`sa_t128k`（截斷 128K 的紀錄 `semianalysis_N16_s1_t128k`）在 `sa_p1`–`sa_p3` 裡沒有對應的 block；另外 4 個（nvme807、load、wt、pfgpu）有。

**做完什麼**
- **Rust 工具鏈**：`micromamba create -p $ROOT/tools/rust -c conda-forge rust=1.99.1`（使用者權限；log `$ROOT/tools/rust-install-*.log`）→ `rustc 1.99.1 (58cf8f9c3 2026-10-13)`、`cargo 1.99.1 (5f94df478 2026-08-27)`。版本和 STATUS_DATA 08:10 要的相同；裝法是 conda-forge 不是 rustup（使用者在本 session 選的選項寫明 conda-forge）。cargo 要 `RUSTC=$ROOT/tools/rust/bin/rustc`（不然找不到 rustc）。環境：`X1_CARGO=$ROOT/tools/rust/bin/cargo CARGO_HOME=$ROOT/tools/cargo-home RUSTUP_HOME=$ROOT/tools/rustup-unused X1_RUST_WORK=$ROOT/x1fast-work`。試編（scratchpad 目標目錄）4.9 s 成功。
- **merge**：
  - 741bc2c：`origin/icpe-next-data` @ bc1e040（Rust 0b0ef1e、`run --engine rust`、偏離 6 的 `sa_p1`–`sa_p3`）。
  - **eeaecef**：`origin/icpe-next-data` @ 9be4925（含 icpe-sim 47a4617）。`git diff 1304345 HEAD -- code/p3x code/icpe/p3_fvar.py code/p3_platform.py` **是空的**。
- **job 數**：run `20261106-075249-icpe-trx-count-sa-p1p3`（`count --plans sa_p1 sa_p2 sa_p3 sens_rep`，在 741bc2c；`code/icpe/next_b_queue.py` 在 eeaecef 沒變）：`sa_p1` 135,648、`sa_p2` 124,416、`sa_p3` 89,640（合計 349,704，和 STATUS_DATA 的數相同）；`sens_rep` 的 `sa_*` 5 格合計 5,336（不跑）。
- **作廢**：07:52 在 741bc2c 上開的 G0／xc_sa（py＋rust）四個 run `20261106-075222-icpe-trx-{g0,xc_sa}-{py,rust}-741bc2c`：08:02 手動 SIGTERM（py 兩個還在跑；rust 兩個 exit 0 但沒比）。理由：STATUS_DATA 08:10 要用 47a4617／1304345 重做。目錄裡留了 `SUPERSEDED`，不當驗證證據。

**現在在跑**：STATUS_DATA 08:10 第 3 步的三項對拍（`$ROOT/tools/verify_rust2.sh eeaecef`，08:03:43 開始，log `$ROOT/runs/verify-rust2-eeaecef-20261106-080343.log`；memwatch 6 GiB、swapguard 2,933 MiB）。都是 `--engine rust --workers 32`，依序：
- (a) G0：run `20261106-080343-icpe-trx-g0-rust-eeaecef` → H200 參考（`next_b_g0.py`）。
- (b) xc_sa：run `20261106-080343-icpe-trx-xc_sa-rust-eeaecef` → `results/icpe_next/b/xcheck/xc_sa`（`next_b_g0.py`）＋TRX40 的 Python xc_sa（run `20261105-105555-icpe-trx-xcheck-xc_sa-54fa2d8`；`next_sim_cmp.py`）。
- (c) 整個 `sa_N16_s1` block：run `20261106-080343-icpe-trx-sa_N16_s1-rust-eeaecef`（manifest 的 cmd＋`--engine rust --workers 32`）→ `results/icpe_next/b/sa_main/sa_N16_s1/x1_sweep.csv.xz`（`next_b_g0.py`）。
- 三項都 0 個不同 → `run --plans sa_p1 sa_p2 sa_p3 --engine rust --workers 32`。有任何不同 → 停、寫這裡。

## 2026-11-06T07:55+08:00 — **`sa_main` 12／12 完成**；3 個 TTL 的分析＋`ttl7cells` 完成；下一步等使用者決定

**做完什麼**
- 最後一個 block：run `20261106-030157-icpe-trx-sa_main-sa_N72_s3`，exit 0，sweep 5,805 s（queue 03:01:57 → 04:38:54），7,344 列 → `results/icpe_next/b/sa_main/sa_N72_s3/`（commit de566f7）。記憶體（183 筆）：PSS 合計最高 10.4 GiB，MemAvailable 最低 45.6 GiB；swap 980 MiB。佇列 04:38:54 結束，兩個保護跟著結束、都沒觸發。
- `sa_main` 12 個 block 都在 `results/icpe_next/b/sa_main/`：每個 7,344 列、`run_id` 單一值、`host`＝`acvlab-TRX40-AORUS-MASTER`；程式都是 8ef0ada（＝sim 54fa2d8，maxtasksperchild 4），40 worker。sweep 時間 3,789–5,888 s，12 個合計 60,634 s。
- **分析**（STATUS_DATA 09:00 第 6 步之 1）：run `20261106-074822-icpe-trx-analyze-sa_main-ttl3`，exit 0，stderr 空。`next_sim_analyze.py --sweep results/icpe_next/b/sa_main/*/x1_sweep.csv.xz --out-prefix sa_main_ttl3 --pf-target cpu`（`export RUN_ID`、`nice -n 10`；git HEAD de566f7，`code/` 和 8ef0ada 相同）。headroom_components 3,888 列、violations 5,322 列 → `results/icpe_next/b/analysis/sa_main_ttl3/`（兩個都 xz，解壓後和 run 目錄的原檔 `cmp` 相同；`manifest.json` 照 `analysis/cw_main_ttl3/` 的格式）。
- **`ttl7cells`**（第 6 步之 2）：run `20261106-074827-icpe-trx-ttl7cells-sa_main`，exit 0：`cells=466／648  (trace,rule,cpu,slow)=253  jobs=40480` → `results/icpe_next/b/sa_main_ttl7/cells.csv`（每列有 run_id、ts、host、src_analysis_run_id）。
- 上兩項 commit f0b13bb。不解讀、不下判定。

**使用者決定（11/6，AskUserQuestion）**：TRX40 **只做 SemiAnalysis**。FreeInference、Bailian 的 P1–P3 不在 TRX40 跑。

**還沒決定**：`sa_main` 之後 TRX40 跑什麼（Python 照 09:00 跑 `sa_main_ttl7`＋`sens_rep sa_*`，或改 Rust 跑偏離 6 的 `sa_p1`→`sens_rep sa_*`→`sa_p2`→`sa_p3`）。已再問使用者。在那之前 TRX40 不跑模擬、不 merge。
- git 上（fetch 07:5x）：`origin/icpe-next-data` @ bc1e040 已 merge Rust 引擎（875fb2b → 304fd72）、G0（rust）PASS、佇列加 `run --engine rust`；`origin/icpe-next-sim` @ 47a4617 再加 W4 雜訊 B1（G1 PASS，STATUS_SIM 06:14），icpe-data 還沒 merge 這個。

## 2026-11-06T03:25+08:00 — Rust 引擎 G1 PASS（STATUS_SIM 03:15）＋偏離 6 已登記 → `sa_main` 之後跑什麼，問使用者

- **git 上看到的**（fetch 03:2x）：
  - `origin/icpe-next-sim` @ 875fb2b，STATUS_SIM 03:15：`x1 sweep --engine rust`（程式 0b0ef1e），G1 py／rust 都 4,416×29、0 格不同；xcheck 14 個 block（含 xc_sa）0 格不同。預設仍是 py。code/p3x 有改 → merge 後要重做 G0、xc_sa。
  - `origin/icpe-next-data` @ 3102df4：PREREG 偏離 6（1546086；P1＝全部格子 7 個 TTL、② ④ 同一組＋截尾規則；P2 nvme807＋只載入；P3 wt＋預取進 GPU）；`next_b_queue.py` 加了 `sa_p1`／`sa_p2`／`sa_p3`（3385802）；使用者 11/6：Rust 的佇列 CPU 約 1.25×workers。**icpe-data 還沒 merge 0b0ef1e。**
  - TRX40 沒有 Rust 工具鏈（`cargo`、`rustc` 都不在）。
- 〔算術〕SemiAnalysis 的 P1–P3＝每個變體 11,304＋10,368＋7,470＝29,142 job × 12＝349,704 job。照 `sa_main` 在 TRX40 的 Python 速度（40 worker 約 1.2 job/s）約 80 h，11/9 前跑不完；Rust 在 TRX40 的速度 NOT_MEASURED。
- **問使用者**：`sa_main` 跑完（約 04:40）之後，TRX40 照 STATUS_DATA 09:00 用 Python 跑 `sa_main_ttl7`＋`sens_rep sa_*`，還是改成 Rust 跑偏離 6 的 `sa_p1`（取代 `sa_main_ttl7`）→ `sens_rep sa_*` → `sa_p2` → `sa_p3`（要在 `$ROOT` 裝 Rust 工具鏈、merge、G0＋xc_sa py／rust 都比）。不管選哪個，3 個 TTL 的分析＋`ttl7cells` 照做（幾秒，偏離 6：沒做完的優先度照偏離 4／5 報告）。

## 2026-11-06T03:03+08:00 — `sa_main` 第 11／12 個 block 完成（`sa_N64_s3`）

- run `20261106-012556-icpe-trx-sa_main-sa_N64_s3`：exit 0，sweep 5,750 s（queue 01:25:56 → 03:01:57），6,174 job，7,344 列 → collect 進 `results/icpe_next/b/sa_main/sa_N64_s3/`（`x1_sweep.csv.xz` 601 KiB＋`manifest.json`；`run_id` 單一值、`host`＝`acvlab-TRX40-AORUS-MASTER`）。
- 記憶體（mem.log 01:25:56–03:01:57，181 筆）：本組 PSS 合計最高 10.4 GiB，MemAvailable 最低 45.8 GiB；swap 980–981 MiB。兩個保護都沒觸發。
- 最後一個 `sa_N72_s3` 03:01:57 開跑。〔算術〕照 s1／s2 的 N72 約 5,850–5,890 s → 約 11/6 04:40 跑完。

## 2026-11-06T01:27+08:00 — `sa_main` 第 10／12 個 block 完成（`sa_N32_s3`）

- run `20261106-000537-icpe-trx-sa_main-sa_N32_s3`：exit 0，sweep 4,807 s（queue 00:05:37 → 01:25:56），6,174 job，7,344 列 → collect 進 `results/icpe_next/b/sa_main/sa_N32_s3/`（`x1_sweep.csv.xz` 571 KiB＋`manifest.json`；`run_id` 單一值、`host`＝`acvlab-TRX40-AORUS-MASTER`）。
- 記憶體（mem.log 00:05:37–01:25:56，151 筆）：本組 PSS 合計最高 10.0 GiB，MemAvailable 最低 45.9 GiB；swap 981 MiB。兩個保護都沒觸發。
- 下一個 `sa_N64_s3` 01:25:56 開跑。〔算術〕剩 2 個 block，照 s1／s2 同 N 的時間約 3.2 h → `sa_main` 約 11/6 04:40 跑完。

## 2026-11-06T00:08+08:00 — `sa_main` 第 9／12 個 block 完成（`sa_N16_s3`）；icpe-ana 的 Rust 引擎與補跑草案（記錄，現在不動）

- run `20261105-230217-icpe-trx-sa_main-sa_N16_s3`：exit 0，sweep 3,789 s（queue 23:02:17 → 00:05:37），6,174 job，7,344 列 → collect 進 `results/icpe_next/b/sa_main/sa_N16_s3/`（`x1_sweep.csv.xz` 537 KiB＋`manifest.json`；`run_id` 單一值、`host`＝`acvlab-TRX40-AORUS-MASTER`）。
- 記憶體（mem.log 23:02:17–00:05:37，120 筆）：本組 PSS 合計最高 9.8 GiB，MemAvailable 最低 46.5 GiB；swap 981 MiB。兩個保護都沒觸發。
- 下一個 `sa_N32_s3` 00:05:37 開跑。〔算術〕剩 3 個 block，照 s1／s2 同 N 的時間約 4.6 h → `sa_main` 約 11/6 04:40 跑完。
- **sa_main 跑完後的步驟已備好**（本機 `$ROOT/tools/sa_post_main.sh`，不進 repo，只做過 `bash -n`）：§4.1 的殼跑 `next_sim_analyze.py … --out-prefix sa_main_ttl3 --pf-target cpu`（`export RUN_ID`、`nice -n 10`）→ 另一個殼跑 `ttl7cells --plan sa_main`。表的放法照 icpe-data 的 `analysis/cw_main_ttl3/`（headroom_components xz＋violations＋manifest.json）。檢查內容後才 commit。
- **icpe-ana 的訊息**（資訊；以 git 為準，已核對 `origin/icpe-next-ana` 的 STATUS_ANA）：
  - aee1527（23:58 節）：Rust 版 x1 引擎，和 `x1_sim.py` 逐位元相同；用本分支 `sa_main/sa_N16_s1` 對過 7,344／7,344 列相同。要等 icpe-sim 在 STATUS_SIM 寫 G1／xcheck PASS 才用；只用在還沒開始的 block。**`sa_main` 照舊用 Python 跑完。**
  - 2e32946（00:20 節）：使用者 11/6 決定補回偏離 4 砍掉的格子（P1–P4），PREREG 偏離 6 由 icpe-data 寫（目前是草稿）。TRX40 的分工**草案**：SemiAnalysis 的補跑＋`sens_rep sa_*`，以及 SemiAnalysis、FreeInference、Bailian 的 P1–P3（Rust 要在 TRX40 另外編、另外對拍）。
  - icpe-trx 的立場：等偏離 6 正式登記＋STATUS_SIM 的驗證；在那之前照 STATUS_DATA 09:00 的步驟走（`sa_main_ttl7`、`sens_rep sa_*` 用目前的 Python 佇列，除非 STATUS 改了）。
  - **草案裡超出 TRX40 目前授權的兩點**（要使用者決定，輪到時問）：
    1. FreeInference 的資料能不能放 TRX40：STATUS_DATA 09:00 的授權清單只有 SemiAnalysis、Bailian（Apache-2.0）、TraceLab（CC BY 4.0）；FreeInference 不在清單上。
    2. 寫入範圍：使用者給 icpe-trx 的只有 `results/icpe_next/b/` 底下的 `sa_*`（＋`analysis/sa_*`、`sens_rep/sa_*`）。FreeInference、Bailian 的下載、轉換、結果目錄都在範圍外。

## 2026-11-05T23:03+08:00 — `sa_main` 第 8／12 個 block 完成（`sa_N72_s2`）；s2 四個都進 results

- run `20261105-212443-icpe-trx-sa_main-sa_N72_s2`：exit 0，sweep 5,842 s（queue 21:24:43 → 23:02:17），6,174 job，7,344 列 → collect 進 `results/icpe_next/b/sa_main/sa_N72_s2/`（`x1_sweep.csv.xz` 607 KiB＋`manifest.json`；`run_id` 單一值、`host`＝`acvlab-TRX40-AORUS-MASTER`）。
- 記憶體（mem.log 21:24:43–23:02:17，184 筆）：本組 PSS 合計最高 10.8 GiB，MemAvailable 最低 45.5 GiB；swap 981 MiB。兩個保護都沒觸發。
- 下一個 `sa_N16_s3` 23:02:17 開跑。
- 〔算術〕s2 四個 block 合計 20,207 s（s1：20,276 s）。剩 s3 的 4 個 → 約 5.6 h → `sa_main` 約 11/6 04:40 跑完。

## 2026-11-05T21:26+08:00 — `sa_main` 第 7／12 個 block 完成（`sa_N64_s2`）

- run `20261105-194923-icpe-trx-sa_main-sa_N64_s2`：exit 0，sweep 5,708 s（queue 19:49:23 → 21:24:43），6,174 job，7,344 列 → collect 進 `results/icpe_next/b/sa_main/sa_N64_s2/`（`x1_sweep.csv.xz` 601 KiB＋`manifest.json`；`run_id` 單一值、`host`＝`acvlab-TRX40-AORUS-MASTER`）。
- 記憶體（mem.log 19:49:23–21:24:43，180 筆）：本組 PSS 合計最高 10.2 GiB，MemAvailable 最低 46.0 GiB；swap 977–981 MiB。兩個保護都沒觸發。
- 下一個 `sa_N72_s2` 21:24:43 開跑。
- 〔算術〕剩 5 個 block；照 s1 同 N 的時間 → 約 7.3 h → `sa_main` 約 11/6 04:40 跑完。

## 2026-11-05T19:50+08:00 — `sa_main` 第 6／12 個 block 完成（`sa_N32_s2`）

- run `20261105-182807-icpe-trx-sa_main-sa_N32_s2`：exit 0，sweep 4,865 s（queue 18:28:07 → 19:49:23），6,174 job，7,344 列 → collect 進 `results/icpe_next/b/sa_main/sa_N32_s2/`（`x1_sweep.csv.xz` 576 KiB＋`manifest.json`；`run_id` 單一值、`host`＝`acvlab-TRX40-AORUS-MASTER`）。
- 記憶體（mem.log 18:28:07–19:49:23，153 筆）：本組 PSS 合計最高 10.1 GiB，MemAvailable 最低 45.8 GiB；swap 959–977 MiB。兩個保護都沒觸發。
- 下一個 `sa_N64_s2` 19:49:23 開跑。
- 〔算術〕剩 6 個 block；照 s1 同 N 的時間 → 約 8.9 h → `sa_main` 約 11/6 04:40 跑完。

## 2026-11-05T18:29+08:00 — `sa_main` 第 5／12 個 block 完成（`sa_N16_s2`）

- run `20261105-172446-icpe-trx-sa_main-sa_N16_s2`：exit 0，sweep 3,792 s（queue 17:24:46 → 18:28:07），6,174 job，7,344 列 → collect 進 `results/icpe_next/b/sa_main/sa_N16_s2/`（`x1_sweep.csv.xz` 536 KiB＋`manifest.json`；`run_id` 單一值、`host`＝`acvlab-TRX40-AORUS-MASTER`）。
- 記憶體（mem.log 17:24:46–18:28:07，119 筆）：本組 PSS 合計最高 9.5 GiB，MemAvailable 最低 46.6 GiB；swap 958–959 MiB。兩個保護都沒觸發。
- 下一個 `sa_N32_s2` 18:28:07 開跑。
- 〔算術〕剩 7 個 block；照 s1 同 N 的時間 → 約 10.2 h → `sa_main` 約 11/6 04:40 跑完（和 17:26 的估計相同）。

## 2026-11-05T17:26+08:00 — `sa_main` 第 4／12 個 block 完成（`sa_N72_s1`）；s1 四個都進 results

- run `20261105-154628-icpe-trx-sa_main-sa_N72_s1`：exit 0，sweep 5,888 s（queue 15:46:28 → 17:24:46），6,174 job，7,344 列 → collect 進 `results/icpe_next/b/sa_main/sa_N72_s1/`（`x1_sweep.csv.xz` 606 KiB＋`manifest.json`；`run_id` 單一值、`host`＝`acvlab-TRX40-AORUS-MASTER`）。
- 記憶體（mem.log 15:46:28–17:24:46，186 筆）：本組 PSS 合計最高 10.4 GiB，MemAvailable 最低 44.1 GiB；swap 959–978 MiB。兩個保護都沒觸發。
- 下一個 `sa_N16_s2` 17:24:46 開跑。
- 〔算術〕s1 四個 block 合計 20,276 s（N16 3,903／N32 4,771／N64 5,714／N72 5,888）。剩 8 個＝s2、s3 各 4 個；假設和 s1 同 N 一樣快 → 約 11.3 h → `sa_main` 約 11/6 04:40 跑完。

## 2026-11-05T15:48+08:00 — `sa_main` 第 3／12 個 block 完成（`sa_N64_s1`）

- run `20261105-141103-icpe-trx-sa_main-sa_N64_s1`：exit 0，sweep 5,714 s（queue 14:11:03 → 15:46:28），6,174 job，7,344 列 → collect 進 `results/icpe_next/b/sa_main/sa_N64_s1/`（`x1_sweep.csv.xz` 601 KiB＋`manifest.json`；`run_id` 單一值、`host`＝`acvlab-TRX40-AORUS-MASTER`）。
- 記憶體（mem.log 14:11:03–15:46:28，180 筆）：本組 PSS 合計最高 10.3 GiB，MemAvailable 最低 46.0 GiB；swap 2,357 → 978 MiB（往下掉：別人的頁換回記憶體）。兩個保護都沒觸發。
- 下一個 `sa_N72_s1` 15:46:28 開跑。
- 〔算術〕s1 三個 block：N16 3,903 s、N32 4,771 s、N64 5,714 s。剩 9 個 block（N72_s1＋s2、s3 各 4 個）；假設 s2、s3 和 s1 同 N 的時間一樣、N72 每個 5,700–6,500 s → 合計約 12.7–13.4 h → `sa_main` 約 11/6 04:30–05:30 跑完。

## 2026-11-05T14:13+08:00 — `sa_main` 第 2／12 個 block 完成（`sa_N32_s1`）

- run `20261105-125122-icpe-trx-sa_main-sa_N32_s1`：exit 0，sweep 4,771 s（queue 12:51:22 → 14:11:03），6,174 job，7,344 列 → collect 進 `results/icpe_next/b/sa_main/sa_N32_s1/`（`x1_sweep.csv.xz` 577 KiB＋`manifest.json`；`run_id` 單一值、`host`＝`acvlab-TRX40-AORUS-MASTER`）。commit 8523791。
- 記憶體（同一個 mem.log，12:51:22–14:11:03，150 筆）：本組 PSS 合計最高 10.2 GiB，MemAvailable 最低 47.8 GiB；swap 2,357–2,401 MiB（沒有往上漲）。兩個保護都還在跑、都沒觸發。
- 下一個 `sa_N64_s1` 14:11:03 開跑。
- 〔算術〕N32 比 N16 多 22%（4,771 vs 3,903 s）。剩 10 個 block：N16×2 約 130 分、N32×2 約 160 分，N64／N72 共 6 個還沒量到——每個抓 80–110 分 → 合計約 13–16 h → `sa_main` 約 11/6 03:00–06:00 跑完。

## 2026-11-05T12:53+08:00 — `sa_main` 第 1／12 個 block 完成（`sa_N16_s1`）

- run `20261105-114609-icpe-trx-sa_main-sa_N16_s1`：exit 0，sweep 3,903 s（queue 記 3,912.7 s），6,174 job，7,344 列 → collect 進 `results/icpe_next/b/sa_main/sa_N16_s1/`（`x1_sweep.csv.xz` 548 KiB＋`manifest.json`；`host`＝`acvlab-TRX40-AORUS-MASTER`）。
- 記憶體（`queue-sa_main-w40-20261105-114609.mem.log`）：本組 PSS 合計最高 9.7 GiB（11:55:10），MemAvailable 最低 49.6 GiB；swap 2,442 → 2,406 MiB（沒有增加）。兩個保護都沒觸發。
- 下一個 `sa_N32_s1` 12:51:22 開跑（同一條佇列）。
- 〔算術〕剩 11 個 block × 65 分鐘 ≈ 12 h；CacheWise 的 block 隨 N 變大最多慢約 40%，所以約 12–17 h → `sa_main` 約 11/6 01:00–06:00 跑完。之後是分析＋`ttl7cells`、`sa_main_ttl7`、`sens_rep 'sa_*'`。

## 2026-11-05T11:47+08:00 — G1 PASS＋merge；**`sa_main` 開跑（40 worker、maxtasks 4）**

- icpe-sim：54fa2d8 的 **G1 PASS**（STATUS_SIM 984dc4f 11:45，run `20261105-100635-icpe-sim-g1-54fa2d8`，4,416 列 × 29 欄 0 格不同、自檢 997／997；我在 git 上看過那一節）。原因：開 GPU 層時 Engine 有循環參照、fork 出來的 worker 幾乎不做 gen2 gc（SIM_CHANGES 第 15 項）。
- icpe-data：merge 進 `icpe-next-data` @ 2dc6c8a。
- 我：merge `origin/icpe-next-data` @ 2dc6c8a → **`8ef0ada`**。`git diff 54fa2d8 HEAD -- code/p3x code/icpe/p3_fvar.py code/p3_platform.py` **是空的** → 11:08 節在 54fa2d8 上做的 G0（128,064 格 0 不同）、xc_sa（1,280 格 0 不同）適用。`next_b_queue.py`、`next_b_g0.py` 和 828e1d3 相同。
- **開跑**：11:46:09，`python -I code/icpe/next_b_queue.py run --plans sa_main --workers 40`（程式 8ef0ada；`X1_MAXTASKS` 沒設 → stdout `maxtasksperchild=4`；log `$ROOT/runs/queue-sa_main-w40-20261105-114609.log`）。保護：memwatch（MemAvailable < 6 GiB 停）、swapguard（swap > 2,933 MiB 停，使用者 09:4x 的上限）。開跑時 MemAvailable 58.3 GiB、swap 2,442 MiB（09:57 那次留下的別人的頁）、load 0.5。
- 第一個 block `sa_N16_s1`（run `20261105-114609-icpe-trx-sa_main-sa_N16_s1`）。每個 block 跑完：`collect --plans sa_main` → commit → push。

## 2026-11-05T11:08+08:00 — 54fa2d8 上提前做 G0、xc_sa：都 IDENTICAL；仍等 G1 PASS＋icpe-data merge

- 為了 G1／merge 一到就能開跑，先在 detached worktree `$ROOT/wt-sim-54fa2d8`（icpe-sim 54fa2d8，`--maxtasks` 預設 4）做我這邊要重做的兩個對拍：
  - **G0**：run `20261105-105553-icpe-trx-g0-54fa2d8`（16 worker、新功能全關；stdout 印 `maxtasksperchild=4`；sweep 679 s、`/usr/bin/time` 11:21）。
  - **xc_sa**：run `20261105-105555-icpe-trx-xcheck-xc_sa-54fa2d8`（同 queue 的 xc_sa 參數、16 worker）。不 collect。
  - 兩個 run 的 cmd.sh 最後一步的比對失敗（`next_b_g0.py` 是 icpe-data 的檔，不在 icpe-sim 的 worktree：`can't open file '.../wt-sim-54fa2d8/code/icpe/next_b_g0.py'`，exit 2）；sweep 本身都跑完。兩個目錄留了 `COMPARE_FAILED_SEE_NEXT_RUN`。
  - **比對**：run `20261105-110736-icpe-trx-compare-54fa2d8`（repo 828e1d3 的 `next_b_g0.py`）：
    - G0 對 `results/icpe_h200/t1/p01/sim_V1/x1/x1_sweep.csv.gz`：**4,416 列 × 29 欄＝128,064 格，0 個不同**。
    - xc_sa 對 `results/icpe_next/b/xcheck/xc_sa/x1_sweep.csv.xz`：**32 列 × 40 欄＝1,280 格，0 個不同**。
- 開跑條件：STATUS_SIM 寫 54fa2d8 的 G1 PASS ＋ icpe-data merge ＋ 我 merge 後 `git diff 54fa2d8 HEAD -- code/p3x code/icpe/p3_fvar.py code/p3_platform.py` 是空的（是空的，上面兩個對拍就適用；不是空的就重做）。然後 `sa_main` 40 worker。

## 2026-11-05T10:25+08:00 — icpe-sim 的修正 54fa2d8（`--maxtasks`，預設 4）在 TRX40 試跑：40 worker 記憶體持平；等 G1＋merge

- icpe-sim 54fa2d8（`origin/icpe-next-sim`）：`x1 sweep --maxtasks`（預設 4、`$X1_MAXTASKS`；0＝原本）→ `Pool(maxtasksperchild)`。G1 跑中（icpe-sim 10:1x 的訊息；以 STATUS_SIM 為準）。
- **試跑**（不進 results/；程式＝detached worktree `$ROOT/wt-sim-54fa2d8` @ 54fa2d8，G1 未過）：run `20261105-100804-icpe-trx-trial-maxtasks4-w40`——sa_N16_s1、和 `sa_main` 相同的 sweep 參數、**40 worker**、`maxtasksperchild=4`（stdout 有印）。跑到 1,329／6,174 job（968 s，涵蓋最重的 O-ALL、O-PF、LRU+RPF 段）後手動停（`STOPPED` 檔）。
  - 本組 PSS 合計：最高 **9.4 GiB**（10:09:46），之後在 7.3–8.8 GiB 之間，沒有往上漲（修正前同一段：889 job 時 39.4 GiB）。
  - MemAvailable 最低 50.2 GiB；swap 2,495 → 2,486 MiB（沒有增加）。
  - 速度：1,329 job／968 s＝每秒 1.37 個（修正前 40 worker 前 540 個：1.37）。
- **計畫**：G1 PASS（STATUS_SIM）→ icpe-data merge 進 `icpe-next-data` → 我 merge、G0＋xc_sa（新程式）→ `sa_main` 用 **40 worker**（使用者先前決定的值；上面的量測顯示記憶體夠）。
- 〔算術〕照每秒 1.37 個 job：一個 block 約 75 分鐘，`sa_main` 12 個約 15 h（N 大的 block 可能更久，CacheWise 多約 40%）。

## 2026-11-05T10:05+08:00 — **卡住**：worker 的記憶體每個 job 都在累積、不釋放 → `sa_main` 在 62 GiB 上跑不完；20 worker 的 block 被 swap 保護停掉

**發生什麼**
- 20 worker 的 run `20261105-092452-icpe-trx-sa_main-sa_N16_s1`：**09:57:27 swapguard 觸發**（swap 3,125 MiB > 使用者允許的 2,933 MiB），停在 1,958／6,174 job（1,946 s）。留了 `ABORTED`、`SWAPGUARD_KILLED`，不 collect。停掉前本組 PSS 49.3 GiB、MemAvailable 10.4 GiB。停掉後 MemAvailable 59.5 GiB、swap 2,503 MiB（別人的頁，慢慢換回）。
- 兩次 run 的本組 PSS 合計 vs 已跑的 job 數（`mem.log`＋stdout 的進度列）：

| run | 已跑 job | 本組 PSS 合計 | 合計 ÷ job 數 |
|:--|:--|:--|:--|
| 40 worker | 889 | 39.4 GiB | 44 MiB |
| 20 worker | 約 950（09:41:54） | 36.5 GiB | 約 38 MiB |
| 20 worker | 約 1,410（09:49:12） | 43.0 GiB | 約 30 MiB |
| 20 worker | 1,958（09:57） | 49.3 GiB | 約 25 MiB |

  合計大致跟「已跑的 job 數」走，和 worker 數關係不大。重的 job（O-ALL、O-PF、LRU+RPF）漲得快，後面的（O-BEL、PENS）慢一些，但**沒有持平**。

**診斷試跑**（不進 results/）：run `20261105-095815-icpe-trx-trial-memgrowth-w1`——1 worker、SA N16 s1、CPU 128、PENS／LRU2／GDSF／ARC／S3FIFO／LRU × GPU LRU／BEL、ssd、cake，14 個 job（142 s）；每 2 s 記 worker 的 `smaps_rollup`（`worker_rss.log`）。
- worker 的 Private_Dirty：開始 148 MiB → 14 個 job 後 388 MiB，**只升不降**；有的 job 一次 +45 MiB，有的持平（GDSF 4 個 job 只在第一個 +44）。RSS 958 → 1,198 MiB。
- 結論（這台量到的）：worker 每跑一個 job 都留下一些記憶體、之後不還。`sa_main` 一個 block 6,174 個 job，20 worker 時每個 worker 約 309 個 → 照上面的量推〔算術〕每個 worker 5–14 GiB，這台 62 GiB 不管幾個 worker 都裝不下。
- 是程式裡的洩漏還是 malloc 碎片，我分不出來；沒看程式改什麼，也沒有改。

**需要 icpe-sim 做什麼**（擋 TRX40 的全部 SA 工作）
- `x1_sim.py` 2278 行的 `mp.get_context("fork").Pool(a.workers)` 加 `maxtasksperchild`（例如 8–16），或找出跨 job 留下的東西。前者不改任何 job 的計算，worker 只是跑完 k 個 job 就換一個新的（從父行程重新 fork）。要做 G1。
- acvlab 也一樣會累積，只是 440 GiB 看不出來；TraceLab（記憶體較大）40 worker 可能也值得看一下 mem-watch。

**需要 icpe-data 做什麼**：上面的修正 G1 過了以後 merge 進 `icpe-next-data`；我 merge、重做 G0＋xc_sa、再開 `sa_main`。

**使用者決定（11/5 10:0x，在 icpe-trx）**：選「請 icpe-sim 加 maxtasksperchild」。已用 SendMessage 通知 icpe-sim（請求）與 icpe-data（告知）；以本檔為準。在修正 G1 過、merge 進 `icpe-next-data` 之前，TRX40 不跑任何 SA 的模擬。

## 2026-11-05T09:41+08:00 — 20 worker 的記憶體也在漲；**使用者：允許 swap 小幅增加**

- 20 worker（run `20261105-092452-icpe-trx-sa_main-sa_N16_s1`）09:40:20：本組 PSS 34.4 GiB（09:36 是 25.8 GiB），MemAvailable 23.5 GiB，swap 885 MiB（還沒增加）。09:38 時每個 worker PSS 中位數 1.54 GiB、最大 1.87 GiB；RSS 中位數 2.29 GiB、最大 2.62 GiB。
- 父行程（p3_fvar）RSS 0.95 GiB，其中和 worker 共用約 0.76 GiB；worker 的 RSS 已經比父行程大，所以不只是 copy-on-write。
- `x1_sim.py` 把重的 job（預取、scan）排在最前面先跑（`J.sort(...)`，2266 行），所以現在這段是記憶體最高的一段；每個 worker 的峰值**沒量到**。程式裡除了 `_PCACHE`（每份紀錄一個，父行程 fork 前就載好）沒有跨 job 的快取。
- 09:39 加了 swap 保護 `$ROOT/tools/swapguard.sh`（swap 用量 > 門檻就 SIGTERM 整組），門檻先設 1,141 MiB（基準＋256）。
- **使用者決定（11/5 09:4x，在 icpe-trx）**：選「允許 swap 小幅增加」＝swap 最多比基準多 2 GiB（門檻 **2,933 MiB**），MemAvailable < 6 GiB 才停。被換出去的是別的使用者閒置的桌面程式（gnome-shell 等）的頁，不影響模擬結果。這是對「不准用到 swap」的放寬，使用者明講。
  - 另兩個選項（沒選）：繼續 20、被停就降 12；現在就停、改 12 重跑。

## 2026-11-05T09:26+08:00 — 40 worker 記憶體不夠、swap 增加 → 停掉作廢；**`sa_main` 改 20 worker 重跑**

**發生什麼**
- run `20261105-091241-icpe-trx-sa_main-sa_N16_s1`（40 worker）跑到 889／6174 job（681 s），**09:24 我手動停掉**（`kill -TERM` 整個行程群組），留了 `ABORTED` 檔，不 collect。
- 記憶體（`$ROOT/runs/queue-sa_main-w40-20261105-091241.{mem.log,pss-5min.txt,pss-abort.txt}`）：

| 時間 | 本組 PSS 合計 | 每個 worker PSS | MemAvailable | swap 用量 |
|:--|:--|:--|:--|:--|
| 09:13:14 | 5.5 GiB | — | 50.2 GiB | 245 MiB |
| 09:17:49（開跑 5 分鐘） | 20.0 GiB | 中位數 501 MiB、p90 552、最大 637 | 36 GiB | 244 MiB |
| 09:21:06 | 28.9 GiB | — | 26.9 GiB | 264 MiB（開始增加） |
| 09:23:4x（停掉前） | 39.4 GiB | 中位數 968 MiB、最大 1,248；私有中位數 949 | 19.4 GiB | 885 MiB |

- swap 的增加不是本組被換出（本組每個行程 `VmSwap` 都是 0），是記憶體壓力把別的使用者閒置的頁（gnome-shell、update-manager、uid 1010 的行程）擠進 swap。照規則「不准用到 swap」，仍然算本組造成的，所以停。
- 漲法：09:17:49 → 09:19:00 只多 1.5 GiB；09:19 以後每 30 s 多約 2 GiB，時間上和 job 從 O-ALL（前 432 個）換到 O-PF／LRU+RPF 一致。所以看起來是這類 job 比較吃記憶體，不像單純越跑越漏；每個 worker 的上限**沒量到**。
- 停掉後 swap 留在 885 MiB（被換出的頁要等那些行程用到才換回）。之後的基準改成 885 MiB。

**處理**：照使用者 09:08 的決定（被停就作廢、降 worker 重跑）。09:24:52 重開 `run --plans sa_main --workers 20`（程式 828e1d3；log `$ROOT/runs/queue-sa_main-w20-20261105-092452.log`；記憶體監看同前）。20 是我依上面的量測選的：每個 worker 以 1.25–1.5 GiB 估，合計 25–30 GiB；本組約 29 GiB 時 swap 開始增加。之後在 block 之間依量到的峰值調整。

**對時間的影響**〔算術〕：40 worker 時每秒約 1.37 個 job（前 540 個）。20 worker 約一半，一個 block 約 2.5 h，`sa_main` 12 個約 30–40 h。實際時間等第一個 block 跑完再寫。

**需要 icpe-sim 知道**（不擋，只是告知）：SemiAnalysis（98,827 請求、4,200 萬 chunk）每個 worker 在 O-PF／LRU+RPF 的 job 上 PSS 約 1–1.25 GiB 而且還在漲（PSS 接近 RSS 的 2 GiB）。如果是 fork 後的 copy-on-write（refcount 碰到父行程的物件），共用的部分會慢慢變私有。這台 62 GiB，worker 數因此受限。

## 2026-11-05T09:14+08:00 — 指紋（不含 meta_json）13／13 相同；G0 PASS；xc_sa IDENTICAL；**`sa_main` 開跑（40 worker）**

**做完什麼**
- **merge** `origin/icpe-next-data` @ ec74563 → `828e1d3`（`--no-ff`）。模擬程式 `code/p3x`、`code/icpe/p3_fvar.py`、`code/p3_platform.py` 和 f4f5668 逐位元相同（`git diff` 空）；改到的只有 `next_b_queue.py`（digest 跳過 `meta_json`）、`next_sim_analyze.py`（`--rival-only-gpu`）、`next_a_*` 與文件。
- **指紋**：run `20261105-091102-icpe-trx-npz-digest-nometa`（程式 828e1d3 的 `next_b_queue.py digest`）對 `npz_digest.csv` 的 `content_sha256_no_meta`（acvlab run `20261105-090851-icpe-data-npz-digest-nometa`）：**13／13 SAME**（`DIGEST_RESULT IDENTICAL`）。和本檔 09:07 節的表也相同。icpe-data 的訊息說它比過也是 13／13；這裡是我自己重算、重比的結果。
- **4. G0 PASS**：run `20261105-090000-icpe-trx-g0-f4f5668`（程式 f4f5668、新功能全關、16 worker、nice 10）：
  - 指令：`P3_CONST=h200 P3_TIER=h200cpu M7_TIER_PARAMS=results/icpe_h200/t1/tier_params_h200cpu.json M7_RES=results/m7_write_policy_h200 X1_TRACE_DIR=$ROOT/datasets/mooncake/traces RUN_DIR=<run>/req OMP_NUM_THREADS=1 python -I code/icpe/p3_fvar.py --variant V1 --calib results/icpe_h200/p01/calib --out-root <run>/out x1 sweep --workers 16`（acvlab 的 G0 cmd.sh 在 acvlab 上、這裡看不到；照 STATUS_SIM G1 的指令＋queue 的環境變數組的）。
  - `next_b_g0.py results/icpe_h200/t1/p01/sim_V1/x1/x1_sweep.csv.gz <run>/out/x1/x1_sweep.csv`：**4,416 列 × 29 欄＝128,064 格，0 個不同**（`G0_RESULT IDENTICAL`）。
  - sweep 1,328 個 job 687 s（16 worker）；最大 RSS（單一行程）385 MiB；整組 PSS 最多 2.96 GiB。
  - G0 跑在 merge 前的 f4f5668；merge 沒有改模擬程式（上面），所以沒有重做。
- **5. xc_sa IDENTICAL**：run `20261105-091126-icpe-trx-xcheck-xc_sa`（程式 828e1d3、16 worker、28 job、sweep 40.6 s），比對 run `20261105-091229-icpe-trx-xcheck-compare-xc_sa`：`next_b_g0.py results/icpe_next/b/xcheck/xc_sa/x1_sweep.csv.xz <run>/out/x1/x1_sweep.csv` → **32 列 × 40 欄＝1,280 格，0 個不同**。沒有 collect。
  - 記憶體（09:12:03 一次取樣，`/proc/<pid>/smaps_rollup`）：16 個 worker 的 PSS 149–205 MiB、parent 174 MiB、queue 11 MiB，合計 3.2 GiB。這是 CPU 64 GiB 的小格子；`sa_main` 有到 1,024 GiB 的容量，worker 可能更大，開跑後再量。
- **6. `sa_main` 開跑**：09:12:41，`python -I code/icpe/next_b_queue.py run --plans sa_main --workers 40`（程式 828e1d3；setsid；log `$ROOT/runs/queue-sa_main-w40-20261105-091241.log`；記憶體監看 `…091241.mem.log`，每 30 s，MemAvailable < 6 GiB 就停整組）。開跑時 MemAvailable 55 GiB、swap 245 MiB（基準）、load 6.1。第一個 block `sa_N16_s1`。

**卡住什麼**：無。

**需要別的 session 做什麼**：無。

**需要使用者決定什麼**：無。

## 2026-11-05T09:07+08:00 — 轉換完成；**npz 內容指紋 13／13 和 acvlab 不同 → 停**（原因：指紋含 `meta_json`）；G0 跑中；**等 icpe-data 的新指紋**

**做完什麼**
- **3. 轉換**：run `20261105-085858-icpe-trx-convert-semianalysis`（exit 0）。p3x＝`git archive 6b17191 code/p3x` 解到 `$ROOT/code-archive/p3x-6b17191`。
  - `TIARA_DATASETS=$ROOT/datasets python -I code/icpe/next_b_convert.py semianalysis --N 16 32 64 72 --seeds 1 2 3 --out-dir $ROOT/datasets/x1_nb --p3x <archive>/code/p3x`：3 分 50 秒，最大 RSS 2.03 GiB（`/usr/bin/time -v`）。
  - 截斷版 `semianalysis --N 16 --seeds 1 --trunc-128k ...`：48 秒，最大 RSS 1.66 GiB。
  - 規則 6：`len(hash_ids)×64=in` 98827／98827。
- **指紋比對**：run `20261105-090355-icpe-trx-npz-digest`（exit 1）：`next_b_queue.py digest <13 個名字>` 和 `results/icpe_next/b/traces/npz_digest.csv` **13 個全部不同**（`DIGEST_RESULT DIFFER n=13 bad=13`）。照步驟 3「不同就停」：**SA 的 xcheck／正式模擬都沒開始。**

**原因（診斷，沒有改任何程式）**
- npz 裡除了 8 個資料陣列（`full_key`、`full_off`、`kind`、`nall`、`parent`、`sess`、`t`、`user_trigger`）還有一個 **`meta_json`** 字串陣列，內容含 `run_id`、`ts`、`host`、`argv`（含 `--out-dir`、`--p3x` 的絕對路徑）、`source.file`（絕對路徑）。
- `cmd_digest` 把 `meta_json` 也算進去 → 換一台機器、甚至同一台重轉一次，指紋必然不同。所以 `npz_digest.csv` 的值在別台**照定義就不可能相同**。
- 診斷 run `20261105-090538-icpe-trx-digest-diag`（腳本在該 run 目錄的 `digest_diag.py`，不在 repo）：
  - (a) 同一套算法、**跳過 `meta_json`** 的指紋 `content_sha256_no_meta`（下表）——acvlab 沒有這個值可比。
  - (b) `meta_json` 解出來的統計欄，和 git 裡 acvlab 的 `results/icpe_next/b/traces/trace_summary.csv` 同名列逐欄比（n_conv、active_median／p10、n_req、n_sess、n_cont、full_chunks_total、n_keys、span_h、inp_tokens_median／p90、frac_inp_gt_128k、trunc_frac_req_cut、time_fix_within_session、rule6）：**13／13 全部相同**（`SUMMARY_FIELDS_RESULT ALL_SAME`）。這是彙總量相同，不是逐位元相同。
  - 先前兩個診斷 run `…090500…`（腳本 bug：`meta_json` 是 0 維，`ABORTED`）、`…090513…`（欄位對應錯：n_keys 在 `meta.check`，`SUPERSEDED`）作廢。

| name | content_sha256_no_meta（TRX40） |
|:--|:--|
| semianalysis_N16_s1 | 3d6192cae774cdb66036527301ded814692b0b457d1f8c926907a9cb4c962b04 |
| semianalysis_N16_s1_t128k | f51da2d56ca961a47110ea13061ae2ee28b01879ad42b18956b25a461e81608f |
| semianalysis_N16_s2 | 460bb9572953b039ba3d589a97afa80549fd6ec309aee8fc62f09f93890b52fd |
| semianalysis_N16_s3 | 8ab19616850d09c155f58b9d53838dab92ebf0b4321cd9133d05c7068120eefe |
| semianalysis_N32_s1 | 517b8bbe9ce428ba7bb76eb76aba54ddf9404bb6c5162fb2f6680efa743a8b7a |
| semianalysis_N32_s2 | 0fb05f6661031aa0b8b8b93bee3a4a4484c9d02875d66f4598904ff05656ff4c |
| semianalysis_N32_s3 | 6ceac31f0c802142e7ba0f8ac9fe6d8f9ba8a6699bc0604743b3258773481584 |
| semianalysis_N64_s1 | f844e1a3b954bfccbff9814c25d4cddb8da50e69030aae5675f48fddc3b0c038 |
| semianalysis_N64_s2 | f5fd5d2534b099cf57e7f63261992c3c4444df73da48bde2378ace2aa18e54b4 |
| semianalysis_N64_s3 | 2074092728cf504f15fcf7337bff9114ada9008b11bec727b91ac7a8fa2d09b6 |
| semianalysis_N72_s1 | ef0f0a9fbd1832f7025886376353475a1320328ba2bc7235144352867d5553a1 |
| semianalysis_N72_s2 | d12b3468afc7c667bea791332e1f7a879bf9e748cd1733cce8ce4ef6049872e6 |
| semianalysis_N72_s3 | af8752b4bf015d9b2227be6d51c6d71594d7cdf36edc8bf829958de1ff190686 |

**跑中**：G0 run `20261105-090000-icpe-trx-g0-f4f5668`（Mooncake，16 worker；和 SA 的紀錄無關，所以沒停）。

**卡住什麼**：SA 紀錄的跨機器相等性沒有逐位元的證據（上面的原因）。

**需要 icpe-data 做什麼**
- `cmd_digest` 跳過 `meta_json`（或只留 meta 裡和機器無關的欄），在 acvlab 對同 13 個名字重算，寫進 `npz_digest.csv`（或另一個欄）。我會和上表逐名比。上表的算法＝`cmd_digest` 原樣、只在 `k == "meta_json"` 時 `continue`。
- 同一個問題也影響 h200 的 Bailian／FI 指紋比對。

**使用者決定（11/5 09:08，在 icpe-trx）**
- **等 icpe-data 的新指紋**：acvlab 用「不含 `meta_json`」的算法重算、寫進 git，13 個都和上表相同，才跑 `xc_sa` → `sa_main`。在那之前 SA 什麼都不跑（xcheck 也不跑）。
- **`sa_main` 直接 40 worker**（使用者：「CPU：48 執行緒 改吃40?」→ 選「sa_main 直接 40」）：第一個 block 就開 40，不先用 16 量。保險＝記憶體監看（MemAvailable < 6 GiB 就停整組；被停的 block 作廢、降 worker 重跑）。swap 以 245 MiB 為基準，增加就記錄並回報。

## 2026-11-05T09:00+08:00 — repo、venv、下載完成；轉換＋G0 跑中

**機器**：`acvlab-TRX40-AORUS-MASTER`，Threadripper 3960X 48 執行緒，62 GiB RAM，swap 8 GiB（開跑前別人已用約 245 MiB，以這個為基準，不准再增加）。Titan RTX 不碰。

**做完什麼**
- **0. Repo**：用本機已有的 clone（`/home/hungwei/paper-hierarchical-kv-state`），分支 `icpe-next-trx` 從 `origin/icpe-next-data` @ `f4f5668` 開。
  - `$ROOT=/ssd4/hungwei/icpe-next`（/ssd4 剩 958G、45%），底下 `venv/`、`uv-python/`、`datasets/`、`runs/`、`hf-home/`、`tools/`、`code-archive/`。程式在 repo 裡，和資料分開。
- **1. venv**：`$ROOT/venv`，Python **3.12.15**、numpy **2.3.5**、pandas **3.0.5**、pyarrow **25.0.1**、python-dateutil **4.0.0**、tzdata **2026.6**（`uv pip freeze` 只有這 5 個）。
  - 這台沒有 `uv`、`curl`、`pip`。另外用 micromamba（conda-forge）建工具環境 `$ROOT/tools/env`（uv 0.13.7、curl 8.23.0、huggingface_hub 的 `hf` 2.4.0），只用來裝 Python 與下載，和模擬用的 venv 分開。
  - Python：`uv python install 3.12.15 --install-dir $ROOT/uv-python --no-bin`（cpython-3.12.15-linux-x86_64-gnu）。
  - repo 的 `.claude/settings.json` 把 `HF_HOME`、`UV_*` 指到 `/ssd7`（這台沒有），還設了 `HF_HUB_ENABLE_HF_TRANSFER=1`；每條指令都明確蓋掉／unset。
- **2. 下載**：run `20261105-085632-icpe-trx-dl-semianalysis-mooncake`（exit 0）。`TIARA_SKIP_VENV=1 TIARA_HOME=$ROOT TIARA_DATASETS=$ROOT/datasets HF_HOME=$ROOT/hf-home TIARA_PY=$ROOT/venv/bin/python TIARA_HF_CLI=$ROOT/tools/env/bin/hf bash code/setup_3090_data.sh semianalysis mooncake`。
  - SemiAnalysis：revision 23f152f6，`hf_verify semianalysisai/cc-traces-weka-062126 sha=23f152f6f0f9399a85901b89a6458def0ef16729 files=3 bad=0`；`traces.jsonl` sha256 `29b6a19e…`。
  - Mooncake：commit d750cdda，6 個檔 git-blob OK；conversation sha256 `b8cbb061…`、toolagent `48a2db1a…`（和指定的相同）；單位驗算 4 份都是 rate=1.000000。

**卡住什麼**：無。

**需要別的 session 做什麼**：無。

**需要使用者決定什麼**：無。
