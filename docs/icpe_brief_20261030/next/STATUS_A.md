# STATUS_A（`h200` session，H200 head node，分支 `icpe-next-a`）

> 只有 `h200` session 寫本檔。格式照 NEXT_PLAN §3b：每次更新加一節。內容以 git 上的版本為準。

## 2026-11-04 23:20（台灣）

**做完什麼**
- **PREREG 已登記**：[`PREREG.md`](PREREG.md)（§6b 八項＋補問的 A–D＋W6 都經使用者逐項確認）。**本檔所在的 commit 推上 `origin/icpe-next-a` 之後，正式結果可以開跑**；在那之前只准試跑。
- W8 先核了影響 §6b 的數字：[`W8_source_check.md`](W8_source_check.md) §1。
- 給 `icpe-sim`／`icpe-data` 的重點（細節以 PREREG 為準）：
  - GPU 層＝**1,777 塊**（512-token、BF16，111.06 GiB；vLLM 0.28.0 預設 `gpu_memory_utilization`＝0.92，不是 0.9）。
  - TTL 網格 **0.3／1／3／10／30／100／300 s**，語意見 PREREG 第 3 項。
  - 8.07 那格：讀 8.07、寫 2.209 GiB/s、`k_read` 3.306。
  - 慢層上限＝⌊0.5 × `n_blocks_full`⌋ 個 INT4 塊（第二批）。
  - 雜訊兩種模型（PBKV 式混合 λ 0–40%；乘法 λ 0–30%）× 兩種抽法 × 10 seed。**PBKV 的雜訊不是時間乘法誤差**（計畫 §6b 草案寫錯）。
  - 並行 **16／32／64／72**；佇列順序 3 個 seed 隨機打亂；CacheWise 一律 5 份複本（470 個對話）。
  - 預取：主結果到 CPU；另跑「直接到 GPU」的敏感度（舊紀錄全部格子）。
  - 舊紀錄（Mooncake、FreeInference T1／T4／T5）**全部格子**用新模擬器重跑。
  - 前綴快取式（APC）在 session 粒度的定義：請 `icpe-sim` 在 G1 前寫進 STATUS_SIM。

**卡住什麼**：無。

**需要對方做什麼**
- `icpe-sim`：照 PREREG 第 1–6 項實作；APC 式的定義寫進 STATUS_SIM。
- `icpe-data`：照 PREREG 第 7 項與 §3 C、D 做到達合成。

**需要使用者決定什麼**（已在 `h200` session 直接問）
- GPU 預取敏感度要不要也跑新紀錄（W2）——PREREG 未登記。
- W6c 敏感度重跑的「代表格」是哪些——PREREG 未登記，要在 W6c 有結果之前定。
- `code/icpe/x1_gpu.py`（W6b 要改）由哪個 session 改——§3b 沒有歸屬。

**下一步**：W1（Mooncake 命中率，只用現有輸出）→ W8 其餘項目。

## 2026-11-04 23:20（台灣）之二

**做完什麼**：使用者回答了上一節「需要使用者決定」的三題（「都做 gpu 的話 h200 做」）→ 寫成 PREREG 的 **偏離 1**：GPU 預取敏感度也跑 W2 的全部正式格子；W6c 敏感度重跑全部正式格子；`code/icpe/x1_gpu.py`（W6b）由 `h200` 改，W6 仍然現在不排。

**需要對方做什麼**：`icpe-data` 跑 W2 時，GPU 預取敏感度（等 `icpe-sim` 的 G1 有這個旗標）也排進去。

**需要使用者決定什麼**：無（目前）。

**下一步**：W1。

## 2026-11-04 23:30（台灣）

**做完什麼**：**W1 完成**（只用現有輸出、只報告）。run `20261104-231951-icpe-next-a-w1-hitrate`；結果與說明 `results/icpe_next/a/w1/`（README.md、`w1_summary_ssd_idle.csv`、`w1_hitrate_mooncake.csv`、`w1_hitrate_fi_t4.csv`、`w1_checks.json`）。自檢 K1–K3 全過。同為 CPU 128 GiB、SSD 級、GPU 閒：Mooncake 的 LRU CPU 命中率 5.87–41.49%，FreeInference 99.29% → 照 W1 的事先登記，「差距取決於固定規則命中多少」這個解釋不用拿掉。

**卡住什麼**：無。

**收到 `icpe-sim` 的訊息**：使用者在 `icpe-sim` 決定了 3 項 PREREG 偏離（TTLB、write-through 敏感度、APC 分兩批）；權威文字在 STATUS_SIM 的 2026-11-04T23:19+08:00 那節，**登記時 origin 上還沒有那節**。等它上 origin、使用者在 `h200` 確認後才寫進 PREREG。

**需要使用者決定什麼**：上面 3 項偏離要不要登記（已在 `h200` 問）。

**下一步**：W8 剩下的項目。

## 2026-11-04 23:45（台灣）

**做完什麼**
- PREREG **偏離 2**（`a209253`）：TTLB 進 ②、`wt` 敏感度跑全部格子、APC 分兩批（照 STATUS_SIM 23:19 原文）；8.07 那格改用未四捨五入的 8.068／2.2081 GiB/s；② ＝GPU 規則 × CPU 規則全部組合取最快。`icpe-sim` 回覆：程式本來就是 2.2081，STATUS_SIM 的 2.2087 是文件筆誤。
- **W8 完成**：[`W8_source_check.md`](W8_source_check.md)（13 篇 arXiv HTML 全文逐字核對；self-review，非 cross-model）。和計畫不一致的：
  1. CacheWise 的「22–2.6×」**不是作者的排版錯誤**：原文是 2–2.6×，「22」是 11/4 抓取工具把數學元件接錯造成的。
  2. CacheScout「預取最多 +1 個百分點」是**只開預取**時；和預測淘汰一起開時 GAIA 每輪延遲再降 28%。
  3. SAGA 的 25 GB/s 是它論證時**假設**的典型值，不是平台設定。
  4. InferCept 的「上下文上限 2K」在全文沒找到 → NOT_VERIFIED。
  5. PBKV 的雜訊是混合式（已在 PREREG 第 6 項）。
  其餘（CacheWise 1.19×、KVFlow 1.08×、Cake 5 個頻寬點、Pensieve 60／600 s、CachedAttention λ＝1、並行數、模型與 GPU）和原文一致；計畫裡寫的頁碼 HTML 看不到，沒核。

**卡住什麼**：無。

**需要使用者決定什麼**：NEXT_PLAN 的 W8 與 §5.9 有上面 1、3、4 三處寫法要改（計畫檔只由使用者或規劃的 session 改，`h200` 不改）。

**下一步**：`h200` 的三件事（W1、W8、PREREG）都做完了；等使用者指示。

## 2026-11-04 23:40（台灣）

**做完什麼**：使用者「你決定」→ `h200` 不改 NEXT_PLAN（W8 的更正寫在 `W8_source_check.md` §4 當勘誤），改做 **W6 的準備（不送出、不用 GPU）**：
- runner `code/icpe/next_a_w6.sh`（`--part w6c`：C1 延伸到 256 塊、fa3；`--part w6a`：C2 的 GPU 重播 36 個 run）；head node `--dry-run` 過。
- 分析 `code/icpe/next_a_w6_analyze.py`（判準事先寫死）。用 T1G 舊的 rep 0 自檢：O-PF 比 HINT-PF 快 3.98%（128 GiB）／3.83%（256 GiB）、HINT-PF 拿到 O-PF 改善的 0.81／0.79（自檢輸出放 scratchpad，不進 results）。
- PREREG **偏離 3**：W6c 引用的校準 run 更正成 V1 的 `20261022-205934-icpe-p01-c1`（fa3）；W6c 敏感度整份換 f(i)；W6b 的 `x1_gpu.py` 其實已能重播 O-BEL，缺的是 O-ORD 其他固定對手。

**GPU 時間估計**〔估計〕：W6c 約 1–3 分鐘（80 塊 × 3 次花 17 s）；W6a 37 個 run × 約 16 分（T1G 每個 run 15–17 分）≈ 10 小時。

**需要使用者決定什麼**：W6 什麼時候送出（使用者說現在不排）。

**下一步**：等使用者指示。

## 2026-11-04 23:50（台灣）

**做完什麼**：使用者「W6 now」→ 送出 W6（W6b 的程式還沒寫，不送）。
- **W6c**＝Slurm job **24443**（`icpe-next-w6c`，1 小時上限）；**W6a**＝job **24444**（`icpe-next-w6a`，14 小時上限）。從 worktree 送出，程式版本＝`9a6523d`（開跑時 runner 改執行 worktree 裡的版本；job 跑完之前不原地改 `code/icpe/next_a_w6.sh`）。
- 排法（使用者選「改排在那些任務後」）：兩個都 `afterany` 掛在同一個 Slurm 使用者的 5 個 DL_Lab5 job 後面（執行中 24269、24282；排隊中 24265、24270、24271），W6a 另外掛在 W6c 後面。不動 DL_Lab5 的 job。實驗室帳號 4 張卡目前用滿（MaxGRESPerAccount），所以實際開跑還要等帳號有空卡。
- 輸出：`results/icpe_next/a/w6c/`、`results/icpe_next/a/w6a/`；分析：`code/icpe/next_a_w6_analyze.py w6c`、`w6a`（head node）。

**卡住什麼**：無（在排隊）。風險：如果之後又有新的 DL_Lab5 job 在 W6 跑的途中開跑，runner 只在開跑時檢查，途中的主機爭用 GpuWatcher 看不到——跑完要對 `cotenants_end.txt` 與 sacct 查重疊。

**需要使用者決定什麼**：無。

## 2026-11-05 00:20（台灣）

**做完什麼**：PREREG **偏離 4**（方案 A，使用者在 `icpe-data` 決定、在 `h200` 確認）：CPU 規則 17、TTL 0.3／10／300 s（上限 ≥10% 的格子用 7 個 TTL 補跑 ②）、主結果＝所有紀錄變體 × 主時間設定、敏感度只跑代表格（CW N16 s1、TL c1、SA N16 s1、Bailian coder、Mooncake conv R1／toolagent R1、FI 第 1 週 × CPU 111.0625／128）；CacheWise 400 個對話、前綴主＝結構推得／敏感度＝快取命中、resume 不去重、Bailian traceB 跳過。另外使用者在 `h200` 選：W6c 敏感度也只跑代表格；W6 不趕 11/9、照原樣排在 DL_Lab5 後面。

**需要對方做什麼**：`icpe-data`、`icpe-sim` 照偏離 4。

**需要使用者決定什麼**：無。

## 2026-11-05 09:00（台灣）— 回 STATUS_DATA 2026-11-05T09:00 節（`origin/icpe-next-data` @ `9f9e291`）

**回 `icpe-data` 請求 1（H200 能不能送純 CPU 的 Slurm job）：不能送。**使用者在 `h200` 選「回『不能送』」→ `old_main`、`bl_main`、兩者的 `_ttl7`、`sens_rep` 的 `mc_*`／`fi_*`／`bl_*` 照你原本的安排留在 acvlab 跑。查到的事實（2026-11-05 08:53–08:55 台灣，head node）：
- 我們的帳號（`chihchunghsu`）只能用 `h200q`（`sacctmgr show assoc`）；送到 `defq` 回「Invalid account or account/partition combination specified」。
- `h200q` 的送件檢查會把每個 job 改成 1 GPU＋16 核＋170 GB：`sbatch --test-only` 不帶 GPU、`-c 1`／`-c 4`／`-c 16` 都回「GPUs: 1 | Fixed: 16 CPUs, 174080MB | Assigned QOS: h200q_1g」；`-c 32` 被拒（「每張 GPU 最多僅能配給 16 核心 CPU」）。`--test-only` 沒有真的送出 job（squeue／sacct 查不到那幾個編號）。
- 帳號上限（QOS `lab_limit` 的 GrpTRES）＝cpu 64、gpu 4、mem 680G；目前 4 張 GPU 都在用（MaxGRESPerAccount）。`--test-only` 估計的開跑時間是 2026-11-12T05:57:56。
- 節點 hgx-cn01：128 核、RealMemory 1,393,029 MB、時間上限 2 天。H200 head node：32 核、62 GiB（可用 45 GiB）、53 個使用者登入；超過約 1 小時的模擬照使用者規則不在 head node 跑。

**請求 2（PREREG 偏離 4 第 2 項的釐清）**：還沒登記，在等使用者確認怎麼登記。登記之前，PREREG 上仍只有偏離 4 原文。

**W6 排程**：DL_Lab5 在 2026-11-05 08:44–08:48 送了 3 個新 job（24507、24508、24509），原本 W6 沒有排在它們後面。W6 開跑時若節點上有我們自己的 job，會等 15 分鐘後 exit 2。使用者選「W6c 加排在後面」→ `scontrol update JobId=24443 Dependency=afterany:24271:24507:24508:24509`（2026-11-05T00:55:07Z）。24444（W6a）原本就掛在 24443 後面，不動。DL_Lab5 的 job 不動。

**需要使用者決定什麼**：PREREG 釐清怎麼登記（已在 `h200` 問）。

## 2026-11-05 09:05（台灣）— PREREG 釐清交給 `icpe-data` 寫

- 使用者在 `h200` 選「交給 acvlab 寫」：STATUS_DATA 9f9e291 請求 2 的那句（「補跑 ② 時，④ 的 GPU 規則集合維持原本 3 個 TTL」）**由 `icpe-data` 自己寫進 `PREREG.md`**（在 `icpe-next-data` 分支上）。從這次起 PREREG 不再只由 `h200` 寫。
- `h200` 這邊不登記這一句。在看到 `icpe-data` 那個 commit 之前，`h200` 不改 `PREREG.md`，避免兩個分支同時改同一處、合併時衝突。
- 建議寫法（照本檔偏離程序 §6，由 `icpe-data` 決定）：接在偏離 4 後面另起一節，不改偏離 4 原文；寫明 `date -u`、出處、理由，以及登記時已經看過哪些正式結果（CacheWise `cw_main_ttl3`／`cw_main_ttl7` 的分析 run `20261105-084414-icpe-data-analyze-cw_main-ttl7`）。

## 2026-11-05 09:10（台灣）— 收到 PREREG 偏離 5

- `icpe-data` 把偏離 5（偏離 4 第 2 項的釐清）寫在 `origin/icpe-next-data` @ `8f362d8`（acvlab `date -u` 2026-11-05T00:59:15Z）。我核過：只在檔尾加一節、偏離 4 原文沒動（和本分支 20fe61a 比，沒有刪掉任何一行）；內容就是使用者交給 `icpe-data` 的那句，另寫了出處、理由和當下已看過的正式結果。
- 本分支**只同步 `PREREG.md` 這一個檔**（取 8f362d8 的版本，內容和 23195e5 相同），**不 merge 整個 `icpe-next-data`**：那樣會把 `code/p3x/x1_sim.py` 改掉約 1,300 行，而 W6（job 24443／24444，還在排隊）開跑時執行的是本 worktree 的程式，W6a 的 `x1_gpu.py` 會 import `x1_sim`。W6 跑完再看要不要 merge。
- 之後若 `h200` 再登記，編號從偏離 6 開始。

## 2026-11-06 00:15（台灣）— 收到 PREREG 偏離 6

- `icpe-data` 把偏離 6 寫在 `origin/icpe-next-data` @ `1546086`（acvlab `date -u` 2026-11-05T16:09:01Z；使用者在 `icpe-ana` 決定由 `icpe-data` 寫，STATUS_ANA 00:20 @ `1ea44c9`）。我核過：只在檔尾加一節，前面各節（含偏離 4、5）沒有刪改任何一行。內容：補回偏離 4 砍掉的格子（P1 截尾規則只報告＋全部格子 ② 和 ④ 都用 7 個 TTL → P2 nvme807／只載入 → P3 `wt`／預取進 GPU → P4 W4 雜訊）；Rust 引擎在 `icpe-sim` 驗過 G1／xcheck 之後才用。W6c 敏感度不在偏離 6 裡，仍是代表格。
- 本分支照上次的做法**只同步 `PREREG.md`**（取 1546086 的版本，和 `origin/icpe-next-data` 目前的 tip 相同），不 merge 整個分支：W6（24443／24444）還在排隊（2026-11-06 00:10 台灣：24443 等 DL_Lab5 的 24507、24509），開跑時用本 worktree 的程式。
- **更正 `h200` 先前的說法**：我在偏離 4 第 2 項的註、以及問使用者偏離 5 時說「④ 維持 3 個 TTL 比較保守（補跑只會讓上限變小）」。這只就「≥10% 選格規則不會漏格」而言成立；就論文主張（增益來自位元組、不是預測）而言，上限被算低反而有利於主張，所以**對主張不保守**。偏離 6 的說明是對的。
- 之後若 `h200` 再登記，編號從偏離 7 開始（先看 `origin/icpe-next-data` 有沒有更新的偏離）。

## 2026-11-06 07:55（台灣）— 收到偏離 6 補記

- `icpe-data` 在 `origin/icpe-next-data` @ `813617a` 加了「偏離 6 補記」（acvlab `date -u` 2026-11-05T23:49:08Z；使用者在 `icpe-data` 選「兩個都停，全改 Rust」）：正在用 Python 跑的 `tl_main/tl_c4`、`old_main/fi6w` 停掉、整個 block 用 Rust 從頭重跑，Python 半成品不收；W4 雜訊 B1 也用 Rust（`icpe-sim` 1304345）。不改格子、規則、判準。我核過：只在檔尾加一節，沒有刪改前面任何一行。沒有佔新的偏離編號（`h200` 下一個仍是偏離 7）。
- 本分支照前兩次的做法**只同步 `PREREG.md`**（取 813617a 的版本，和 `origin/icpe-next-data` 目前的 tip 7a59fb1 相同）。
- W6（2026-11-06 07:51 台灣）：DL_Lab5 的 24508、24509 已結束，24507 從 05:39 開始在跑；24443（W6c）等它結束，24444（W6a）接在後面。

## 2026-11-06 09:00（台灣）— 收到 PREREG 偏離 7（`icpe-main` 寫）

- 新的 session `icpe-main`（分支 `icpe-next-main`）把偏離 7 寫在 `origin/icpe-next-main` @ `bcd0bad`（acvlab `date -u` 2026-11-06T00:53:20Z；使用者在 `icpe-main` 選「由 icpe-main 寫」）：判定方式的釐清——原始／合成到達各自下判定、Mooncake R1／R2 各自判定（`rule=min` 只報告）、寫入飽和的格子照算並另報格數；另寫了 `icpe-main` 的解讀（算判定的元件、工具類的分法、「≥15%」照字面、「沒有簡單線索能拿走大部分」＝NOT_MEASURED → 推翻條件寫「未定」）。不改格子、規則、門檻、主指標。
- 我核過：和本分支 c987bbd 的 PREREG 比，只在檔尾加一節（0 行刪除），前面含偏離 6 補記都在。`origin/icpe-next-data` 的 PREREG 目前和本分支 c987bbd 相同（還沒有偏離 7）。
- 節內照實寫了：使用者做決定 1–3 時，`icpe-main` 已看過 SemiAnalysis N16 s1（合成到達）有 ≥15% 的格子、CacheWise N16 s1／s2 部分元件最大 <1.2%；選「登記」時已看過版本 A（3 個 TTL）的全部判定表。
- 本分支照前幾次的做法**只同步 `PREREG.md`**（取 bcd0bad 的版本）。`h200` 下一個偏離編號是 8（登記前先看 `icpe-next-data`、`icpe-next-main` 兩個分支有沒有更新的偏離）。
- W6（2026-11-06 08:55 台灣）：仍等 DL_Lab5 的 24507（05:39 開始跑）。

## 2026-11-06 09:05（台灣）— 收到偏離 7 更正

- `icpe-main` 在 `origin/icpe-next-main` @ `ed4863d` 加了「偏離 7 更正」（acvlab `date -u` 2026-11-06T00:57:25Z）：只把偏離 7 裡「使用者選『登記』的時候（09:0x）」更正為 08:51–08:53（acvlab 本地），先後順序不變；不改偏離 7 內容、不佔編號。我核過：只在檔尾加一節、0 行刪除。
- 本分支**只同步 `PREREG.md`**（取 ed4863d 的版本）。同步前，`origin/icpe-next-data` 的 PREREG 和本分支 1c0973f 相同（已含偏離 7）。`h200` 下一個偏離編號仍是 8。

## 2026-11-06（台灣）— 偏離 8 的編號給 `icpe-main`

- `icpe-main` 問偏離 8 能不能用（P4 雜訊選格的解讀；使用者在 `icpe-main` 決定登記、由 `icpe-main` 寫）。我查了 origin 上所有 `icpe-next-*` 分支和 `main` 的 `PREREG.md`（a e594171、ana 43e7ffc、b 19f3e85、data 16bbb24、main-branch c2bd12c、sim 88a81e3、trx 6209515、main ff71545）：都沒有偏離 8，最大是偏離 7；`h200` 也沒有待登記的偏離 → **偏離 8 給 `icpe-main`，`h200` 下一個從 9 開始**（登記前仍先查各分支）。

## 2026-11-06 13:55（台灣）— 收到 PREREG 偏離 8（`icpe-main` 寫）

- `icpe-main` 把偏離 8 寫在 `origin/icpe-next-main` @ `df8b74c`（acvlab `date -u` 2026-11-06T04:41:49Z）：P4（W4 雜訊）選格＝版本 B（P1 正式值）上限 ≥0.15 的每個（格, 元件）；元件＝5 個判定元件＋O-ALL(report)（只報告）；Mooncake 只列 R1／R2；寫入飽和照列另標。清單 2,582 個（格, 元件）、975 個格（run `20261106-124122-icpe-main-p4cells`）。節內寫明使用者決定時已看過版本 A、B 的全部輸出，W4 雜訊結果還沒看過。我核過：只在檔尾加一節、0 行刪除；`origin/icpe-next-main` tip c868ef2 的 PREREG 和 df8b74c 相同。
- 本分支**只同步 `PREREG.md`**（取 df8b74c 的版本）。`h200` 下一個偏離編號是 9。
- W6（2026-11-06 13:51 台灣）：仍等 DL_Lab5 的 24507（05:39 開始，已跑 8 小時 11 分）。

## 2026-11-07 08:35（台灣）— 偏離 9 的編號給 `icpe-rux`

- `icpe-rux`（跨 session 的名字 `rux-proj`，分支 `icpe-next-rux`）問下一個偏離編號，用來登記 W3 線索 (a) 的規則（使用者 11/7 約 08:45 經 `icpe-main` 轉達；要在 TRX40 跑 W3 sweep 之前登記；push 前會先問使用者）。我查了 origin 上的 icpe-next-a 6b84a5c、ana 43e7ffc、b 19f3e85、data a25dcba、main-branch c868ef2、rux 4558329、sim 90dae78、trx 6872edf 和 main ff71545 的 `PREREG.md`：最大都是偏離 8，沒有 9；`h200` 沒有待登記的偏離 → **偏離 9 給 `icpe-rux`，`h200` 下一個從 10 開始**（登記前仍先查各分支）。
- W6（2026-11-07 08:33 台灣）：仍等 DL_Lab5 的 24507（2026-11-06 05:39 開始，已跑 1 天 2 小時 53 分；h200q 上限 2 天）。

## 2026-11-07（台灣）— 收到 PREREG 偏離 9（`icpe-rux` 寫）

- `icpe-rux` 把偏離 9 寫在 `origin/icpe-next-rux` @ `8f04a91`（RUX3DPRO `date -u` 2026-11-07T00:36:50Z）：W3 線索 (a) 的規則——cue 的定義（TraceLab：該輪 `tools` 非空；FreeInference：下一個請求 `step_trigger`＝tool）、③ 的兩個 GPU 規則 CUEP／CUEO × `NON_ORACLE` 10 個 CPU 規則、紀錄＝TraceLab c1／c2／c4＋FI 6 週、格＝主結果的格、f_c＝(T② − T③) ÷ (T② − T④c)、判定門檻 f > 0.5、逐格任一格成立就算。節內照實列了寫的時候已看過的結果。我核過：和本分支 b87bca8 的 PREREG 比，只在檔尾加一節、0 行刪除；push 時 data／main／trx 分支最大都是偏離 8。
- 本分支**只同步 `PREREG.md`**（取 8f04a91 的版本）。`h200` 下一個偏離編號是 10。

## 2026-11-07 08:45（台灣）— 收到偏離 9 更正

- `icpe-rux` 在 `origin/icpe-next-rux` @ `e7e0c77`（RUX3DPRO `date -u` 2026-11-07T00:40:03Z）加了「偏離 9 更正」：更正偏離 9「經過」的兩個時間（使用者在 `icpe-main` 決定約 08:28–08:30；在 `icpe-rux` 定三點約 08:29–08:35），先後順序不變；補一句方向：逐格任一成立的彙總偏向「推翻條件成立」，和第 5 點的方向相反。規則、門檻、判定不改，不佔編號。我核過：只在檔尾加一節、0 行刪除；data／main／trx 分支最大仍是偏離 8。
- 本分支**只同步 `PREREG.md`**（取 e7e0c77 的版本）。`h200` 下一個偏離編號仍是 10。
- W6（08:40 台灣）：仍等 DL_Lab5 的 24507（已跑 1 天 3 小時）。

## 2026-11-07 08:55（台灣）— 收到偏離 9 補記

- `icpe-rux` 在 `origin/icpe-next-rux` @ `aec99d1`（RUX3DPRO `date -u` 2026-11-07T00:52:12Z）加了「偏離 9 補記」：W3 前的版本 B 重現檢查要涵蓋 W3 會判的全部格（9 個 CPU 容量 × ssd／b32／nfs × GPU 閒／忙 × Cake，每個變體各自做，任一格不同 → 該變體不判）。TraceLab c1 先前只重現 3 個容量就算了 W3 判定（icpe-next-trx `9e1064f0`），依本補記改為待確認。節內照實寫了使用者與 `icpe-rux` 決定前已看過 TraceLab c1 的 W3 結果。規則、門檻、判定不改，不佔編號。我核過：只在檔尾加一節、0 行刪除；data／main／trx 分支最大仍是偏離 8。
- 本分支**只同步 `PREREG.md`**（取 aec99d1 的版本）。`h200` 下一個偏離編號仍是 10。

## 2026-11-07 09:00（台灣）— 偏離 10 的編號給 `icpe-rux`

- `icpe-rux` 再要一個偏離編號，用來登記 block 粒度 APC（第二批；APC 加進 ② 和 ④、O-GPU 的 ④ 加 BELP、重算全部登記變體的主結果格、APC 版上限表成為正式主表；使用者在 `icpe-rux` 決定，push 前會再問使用者）。我查了 origin 上的 icpe-next-a 282b06d、ana 43e7ffc、b 19f3e85、data a25dcba、main-branch c868ef2、rux aec99d1、sim 90dae78、trx dce4005 和 main ff71545：最大是偏離 9，沒有 10；`h200` 沒有待登記的偏離（W6b 的 run set 還沒寫程式、沒排）→ **偏離 10 給 `icpe-rux`，`h200` 下一個從 11 開始**（登記前仍先查各分支）。

## 2026-11-07 09:05（台灣）— 收到 PREREG 偏離 10（`icpe-rux` 寫）

- `icpe-rux` 把偏離 10 寫在 `origin/icpe-next-rux` @ `935dcd7`（RUX3DPRO `date -u` 2026-11-07T00:59:09Z）：block 粒度 APC 的定義（session LRU 選 session、尾端逐塊逐出；wb／wt）；APC 加進 ② 與 O-ORD／O-DROP／O-NOW／O-PF 的 ④；O-GPU 的 ④＝{BEL, BELP}；40 個登記變體的主結果格全部重算（18 個 GPU 規則，先過版本 B 重現檢查）；APC 版（版本 C）取代版本 B 成為正式主表，版本 B 保留並列；W3 的 f、P4 選格改用版本 C。節內照實列了決定時已看過的 APC／BELP 探索摘要。我核過：和本分支 ab8955c 的 PREREG 比，只在檔尾加一節、0 行刪除；data／main／trx 分支最大仍是偏離 8。
- 本分支**只同步 `PREREG.md`**（取 935dcd7 的版本）。`h200` 下一個偏離編號是 11。
- W6（09:01 台灣）：仍等 DL_Lab5 的 24507。

## 2026-11-07 09:05（台灣）之二 — 偏離 11 的編號給 `icpe-rux`

- `icpe-rux` 再要一個編號，用來把 W5（CPU 層 INT4＋一次拆完，敏感度）與排隊（一階近似 qpost，敏感度）合登一節（使用者在 `icpe-rux` 定了範圍與地位，push 前會再問使用者）。origin 上 icpe-next-a c0952cc、ana 43e7ffc、b 19f3e85、data a25dcba、main-branch c868ef2、rux 7a89a9b、sim 90dae78、trx 4ca0d65、main ff71545：最大是偏離 10，沒有 11；`h200` 沒有待登記的偏離 → **偏離 11 給 `icpe-rux`，`h200` 下一個從 12 開始**（登記前仍先查各分支）。

## 2026-11-07 09:10（台灣）— 收到 PREREG 偏離 11（`icpe-rux` 寫）

- `icpe-rux` 把偏離 11 寫在`origin/icpe-next-rux` @ `4a71338`（RUX3DPRO `date -u` 2026-11-07T01:05:49Z）：W5（CPU 層 INT4＋一次拆完；7 個代表變體 × CPU 111.0625／128／256；每組事後挑較快格式；只報告）與排隊（一階近似 `qpost`；固定並行紀錄 SemiAnalysis／CacheWise 全部主結果格；重負載 ≥20 s 只報告、標不可靠；主結果標「上限（不含排隊）」）。主結果、門檻、主指標不改。節內照實列了決定時已看過的探索摘要（含「排隊負載越重上限越大」）。我核過：和本分支 96050c8 的 PREREG 比，只在檔尾加一節、0 行刪除；data／main／trx 分支最大仍是偏離 8。
- 本分支**只同步 `PREREG.md`**（取 4a71338 的版本）。`h200` 下一個偏離編號是 12。

## 2026-11-07 10:15（台灣）— 收到 PREREG 偏離 12（`icpe-rux` 寫）

- `icpe-rux` 把偏離 12 寫在 `origin/icpe-next-rux` @ `294d2bd`（RUX3DPRO `date -u` 2026-11-07T02:13:34Z）：W7 合成參數地圖（9 維全部；LHS 90 點 × 3 seed＝270 份合成紀錄；Rust 生成器 `w7gen`，引擎 v8 不改；版本 C 全套網格，GPU 固定 111.0625 GiB，CPU 9 個水準 WS_ref × 2^k，k＝−6..2；單位＝(點, k, 慢層, 閒／忙, 元件)，3 seed 中位數 ≥0.15 算在區域內；最多加密一輪，需先再登記；和真實紀錄分開報）。**只給 CCGrid 版，ICPE 論文不用**；主結果、門檻、主指標不改。節內照實揭露了登記前的試跑與煙霧測試（上限表有算出、沒讀；進度行看到 5 個 job 的 mean_ret）。我核過：和本分支 0fb651e 的 PREREG 比，只在檔尾加一節（85 行）、0 行刪除；origin 上 a 0fb651e、ana 43e7ffc、b 19f3e85、data a25dcba、main-branch c868ef2、sim 90dae78、trx e3af7c2、main ff71545 都沒有偏離 12（最大 11）；rux 之後的 f971708 沒再動 PREREG。
- 號碼：我在 09:10 回 `icpe-rux` 寫「h200 下一個偏離編號是 12」，原意是 `h200` 自己（W6b）下一個用 12，`icpe-rux` 讀成給它的號碼。各分支都沒有撞號（`h200` 尚未登記任何 run set），所以照收；**`h200` 下一個從 13 開始**（登記前仍先查各分支）。
- 本分支**只同步 `PREREG.md`**（取 294d2bd 的版本）。
- W6（10:15 台灣）：24443、24444 仍等 DL_Lab5 的 24507（剩 19 小時 24 分）；目前 squeue 上本帳號沒有新的 DL_Lab5 job。

## 2026-11-07 11:00（台灣）— 對話資料集補登記：使用者決定編號偏離 13，但先請 `icpe-data` 補寫版本 B／C

- `icpe-data` 請 `h200` 把 `origin/icpe-next-data` @ `688ad0b` 的 STATUS_DATA「2026-11-07T11:00+08:00 — 對話資料集補登記」（ShareGPT、WildChat-1M、LMSYS-Chat-1M、UltraChat 的主網格；acvlab `date -u` 2026-11-07T02:57:51Z；commit 10:58:29 +0800；寫明還沒跑任何模擬、只看過 ShareGPT 的轉換統計）收進 PREREG，建議用不編號的附錄。
- 我核對時看到兩點：
  1. PREREG §6 寫 push 之後的任何改動都在文末「偏離」加一節；§4 寫「另補登記」的 W3、W5、W7 先前都登記成偏離 9、11、12；§4 的選跑清單只列 ShareGPT＋Pensieve，另外三份不在清單裡。
  2. 這份登記寫「指標、表格、門檻都和主結果相同」，但網格是 `next_b_queue.py` 的 `sweep_args()` 預設（688ad0b 第 68–70、88–90 行）：GPU 規則 LRU、BEL、TTLA／TTLB × 0.3／10／300，即版本 A；不含偏離 6 的 7 個 TTL（版本 B），也不含偏離 10 的 block 粒度 APC、BELP（版本 C，正式主表）。偏離 10 的重算只涵蓋原本 40 個登記變體。
- 使用者決定（AskUserQuestion，約 11:00 台灣）：
  - 形式：**「偏離 13，標『補登記』」**——由 `h200` 照抄 STATUS_DATA 原文、引用 688ad0b，登記時間以那節的 commit 時間為準，進 PREREG 的時間另記。**偏離 13 保留給這份**；`h200` 自己（W6b）下一個改從 14 開始。
  - 版本：**「請 icpe-data 先補寫」**——`h200` 先不收；請 `icpe-data` 在開跑前於 STATUS_DATA 補寫這四份要不要跑版本 B／C、由哪台用哪版引擎跑；補好後 `h200` 連同原節一起收成偏離 13。

## 2026-11-07 11:10（台灣）— PREREG 偏離 13（補登記）：公開對話資料集，`h200` 寫

- `icpe-data` 照使用者 11:00 的決定補寫好了：`origin/icpe-next-data` @ `e56c2be` 的 STATUS_DATA「2026-11-07T11:05+08:00 — 對話資料集補登記的補寫」（acvlab `date -u` 03:04:39Z）：版本 C 為正式表、版本 B 並列、版本 A 不單獨報；版本 B＝`chat_main`＋`chat_p1` 的 `_ttl`（v7 62b0932b），版本 C＝v8 48e4efce（固定 icpe-next-rux 4a713386 的檔，重現檢查要 REPRODUCED）；acvlab 跑 ShareGPT／WildChat、TRX40 跑 LMSYS／UltraChat，B、C 同台；判定由 icpe-main 做、歸「合成到達」、單獨成一節、不跑 P4；UltraChat 一行兩個 JSON 物件用 raw_decode 拆。補寫開頭照實記了：688ad0b 之後、我的訊息到之前，acvlab 已跑完版本 A 的 `sg_T60_s1`、`sg_T600_s1`（留作版本 B 的輸入），`sg_T60_s2` 停下標 `STOPPED_NOT_USED`；只看過兩行進度輸出。
- 我寫進 PREREG 的「偏離 13（補登記）」（H200 `date -u` 2026-11-07T03:06:54Z）：收錄說明＋核對（STATUS_DATA 0 行刪除、11:00 節在兩個 commit 逐字相同；v8 `src/main.rs` sha256 48e4efce…；collect_v8.py 在 4a713386 之後改過，固定用 4a713386 的版本；STATUS_TRX「11:15」節的 commit 334317cd 時間是 11:02:11 +0800）＋兩節原文照抄（只把節標題降成 `####`）。寫之前查過各分支：最大是 12，沒有 13。
- `h200` 下一個偏離編號是 14。

## 2026-11-07 11:08（台灣）— 偏離 14 的編號給 `icpe-sim`

- `icpe-sim` 要一個編號，用來登記 P4 雜訊斷崖的修正規則（使用者 11/7 約 11:12 在 `icpe-main` 選 B：補規則、全部重跑、修正版取代正式結果；細節在 `origin/icpe-next-sim` @ 631d17c 的 STATUS_SIM「2026-11-07T11:02+08:00」節：(1) O-PF 預取副本在預測的請求過了還沒被讀就丟，λ＝0 也會變，主結果的 O-PF 要重算；(2) 模型 P 抽到的時間已過時，改從經驗分佈「比已過時間更長」的部分重抽，取代夾成下一個請求）。由誰寫、push 前要先問使用者，`icpe-sim` 正在問。
- origin 上 icpe-next-a 021cadf、ana 43e7ffc、b 19f3e85、data c715f05、main-branch cc3d7bd、rux f054a1a、sim 631d17c、trx 334317c、main ff71545：最大是偏離 13，沒有 14；`h200` 沒有待登記的偏離（W6b 的 run set 還沒寫程式、沒排）→ **偏離 14 給 `icpe-sim`，`h200` 下一個從 15 開始**（登記前仍先查各分支）。

## 2026-11-07 11:19（台灣）— 收到 PREREG 偏離 14（`icpe-sim` 寫）

- `icpe-sim` 把偏離 14 寫在 `origin/icpe-next-sim` @ `fe34522`（acvlab `date -u` 2026-11-07T03:12:47Z；底＝icpe-next-data c715f05，和本分支 021cadf 的 PREREG 相同）：P4 雜訊斷崖的兩條修正規則——(i) O-PF／O-ALL 預取副本在預測被讀的請求過了就丟（λ＝0 也適用）；(a) 模型 P 抽到已過的時間，從經驗分佈「間隔 > 已過時間」的部分重抽。兩條都是旗標，關閉時逐位元不變；修正版取代正式結果（規則 (i)：主結果、P1–P4、版本 C、偏離 13 對話資料集的 O-PF／O-ALL job；規則 (a)：P4 模型 P λ>0）。門檻、主指標、規則集合、比較層級不改。節內寫明「看過 P4 結果之後才補」、規則 (i) 是看過診斷效果後選的（窄版沒選）、使用者先選 C 後改 B，並列了看過的結果。我核過：和本分支 9ea1641 的 PREREG 比，只在檔尾加一節（75 行）、0 行刪除；其他分支最大 13。
- 本分支**只同步 `PREREG.md`**（取 fe34522 的版本）。`h200` 下一個偏離編號是 15。
- `icpe-next-sim` 以前沒有 PREREG.md；之後若 merge 它，PREREG 可能出現「兩邊都新增」的衝突，取內容較長、只增不刪的那版（目前＝fe34522）。

## 2026-11-07 11:47（台灣）— 偏離 15 的編號給 `icpe-main`；W6 歸屬的轉達

- `icpe-main` 要一個編號，用來登記「CacheWise 從論文拿掉、內部分析保留；論文的表、圖、摘要只用 6 份紀錄、28 個變體」（使用者 11/7 約 11:4x 在 `icpe-main` 決定；理由是授權，使用者在 `icpe-data` 說「CacheWise 下先停 放不了paper」；`icpe-main` 會寫明是看過結果之後的決定、以及拿掉前後的判定字：合成工具類 21/26 → 14/14，原始工具類和推翻條件不變；push 前先給使用者看）。origin 上 icpe-next-a e93074f、ana 43e7ffc、b 19f3e85、data 8637e7b、main-branch dfeaeb7、rux f054a1a、sim fe34522、trx 291e7d6、main ff71545：最大是偏離 14，沒有 15；`h200` 沒有待登記的偏離 → **偏離 15 給 `icpe-main`，`h200` 下一個從 16 開始**（登記前仍先查各分支）。
- `icpe-data` 轉達使用者 11/7 約 11:55（acvlab 的鐘）的話「W6：H200 上的 GPU 實測 改叫 h200 session 做」。W6 本來就歸 `h200`（偏離 1「GPU 的工作都歸 h200」），而且 24443（W6c）→ 24444（W6a）已排，等 DL_Lab5 的 24507（11:46 時剩 17 小時 52 分）。`icpe-data` 列的待定項目已在偏離 1、3、4 定了：W6c 敏感度跑偏離 4 第 4 點的代表格（偏離 4 第 6 點）；x1_gpu.py 由 `h200` 改（偏離 1）；W6b 缺的是 O-ORD 的其他固定對手，跑哪些、幾次在程式改好後另外登記（偏離 3 第 3 點）。

## 2026-11-07 11:56（台灣）— W6b 的程式準備好（還沒登記、還沒送出）

- 使用者 11:4x（AskUserQuestion）選「現在準備，不動排隊中的程式」。
- **不改** `code/icpe/x1_gpu.py`、`code/p3x/x1_sim.py`、`code/icpe/next_a_w6.sh`（24443／24444 開跑時會讀）。新寫兩支：
  - `code/icpe/next_a_w6b.py`：import x1_gpu，照原樣呼叫它的 replay／cmd_sim／cmd_gpu，做法＝O-ORD 的 ②（LRU、LRU2、ARC、S3FIFO、GDSF、PENS，＝x1_sim `RIVALS["O-ORD"]`）和 ④（O-BEL、O-BELS，＝`COMPONENT_POLICIES["O-ORD"]`）；子命令 selftest／sim／gpu／analyze。
  - `code/icpe/next_a_w6b.sh`：GPU runner（結構同 next_a_w6.sh；`--reps` 必填；8 個做法各冒煙 100 個請求；每輪 16 個設定用 seed 20261108＋輪次打亂；自己人同節點就等、污染重跑最多 3 次）。檔名不在 `next_a_*.py` 樣式內（同偏離 3 第 5 點的 next_a_w6.sh），只有 `h200` 用。
- **x1_gpu.py 原本跑不了 GDSF**：`make_engine` 不傳 `gdsf_view`，GDSF 的 Engine 在建構時失敗。第一次自測 run `20261107-115225-icpe-next-w6b-selftest` 在 GDSF 停下（`AttributeError: 'NoneType' object has no attribute 'slow_name'`，x1_sim.py:413），run 目錄標 `FAILED_NOT_USED`。修法：next_a_w6b.py 換掉 `X1G.make_engine`，GDSF 照 x1_sim 自己 sweep 的寫法傳 `views[0]`（GDSF 只用這個 view 的 slow_name、d、f；GpuView 是 View 的子類別、同一組 VIEW 參數）。
- **自測**（head node，CPU）：run `20261107-115434-icpe-next-w6b-selftest` → `results/icpe_next/a/w6b/selftest.csv`：8 個做法 × CPU 128／256，重播迴圈（模型時間）≡ Engine.run 逐請求逐位元（TTFT、慢層讀寫位元組、預取位元組）**16／16 PASS**，各 1,535 個請求。只記 PASS／FAIL，沒有記、沒有印任何 TTFT（登記前不看模擬數字）。程式 sha256 前 16 碼：x1_gpu.py ca74a722a923246d、x1_sim.py 047b06485b0bdb3d、x1_trace.py 8f659c970f386486。
- runner `--dry-run`／`--list` 在 head node 跑過（reps＝3 時 8 個冒煙＋48 個 run）；analyze 只用假資料在 scratchpad 測過（PASS／INCOMPLETE 兩條路徑）。
- **還要使用者決定、再登記成偏離 16**：每個設定跑幾次（rep 數）、多個 rep 怎麼彙總進判準。S_orig 在登記之後才算。

## 2026-11-07 12:00（台灣）— PREREG 偏離 16（W6b 的 run set）登記、S_orig 算完、W6b 送出（job 24687）

- 使用者（AskUserQuestion，約 11:58）：每個設定 **6 次**；多個 rep **每格取中位數**再套判準。（第一題「現在準備」的回答時間在 11:47–11:52 之間：上一節寫 11:4x、偏離 16 寫 11:5x，都在這個範圍。）
- **PREREG 偏離 16**（`h200` 寫，commit `9e52a5f`，H200 `date -u` 2026-11-07T03:58:xxZ）：W6b＝O-ORD 的 ② 6 個＋④ 2 個 × CPU 128／256（k＝8，同 W6a／T1G P02）× 6 次＝96 個 run；判準＝P0-2 §6 三條，每格取 6 個 rep 的中位數；PASS／FAIL／INCOMPLETE。偏離 15 已給 `icpe-main`、還沒寫入，所以 16 排在它前面。寫之前查過各分支：最大是 14。
- **S_orig**（登記 push 之後、任何 GPU 資料之前；head node）：run `20261107-115902-icpe-next-w6b-sim-orig`（exit 0）→ `results/icpe_next/a/w6b/sim_requests.csv.xz`（24,560 列；未壓縮的 `sim_requests.csv` 留在工作目錄、不進 git，之後 S_real 會接在後面）。
  - 檢查：LRU 128 的回來請求平均 TTFT 0.30484398001421004 s，和 T1G P02 的 selftest（run `20261028-162428-icpe-t1g-p02-selftest`）逐位相同。
  - O-ORD 上限〔模擬〕：CPU 128 → 最好的 ② LRU2 0.291972 s、最好的 ④ O-BEL 0.262309 s，上限 10.16%；CPU 256 → GDSF 0.284631 s、O-BEL 0.246323 s，上限 13.46%。兩格都 ≥2%，判準 (3) 兩格都要判。
- **送出**：`env -u TMPDIR -u CLAUDE_CODE_TMPDIR sbatch --dependency=afterany:24444 code/icpe/next_a_w6b.sh --reps 6` → **job 24687**（1 GPU、36 小時上限；排在 24444 之後）。`--test-only` 的預估開始時間 2026-11-12T08:04（Slurm 的保守估計，照實記）。
- 鏈：24507（DL_Lab5，剩約 17 小時 40 分）→ 24443（W6c）→ 24444（W6a）→ 24687（W6b）。之前不能改 `next_a_w6.sh`、`next_a_w6b.py`、`next_a_w6b.sh`、`x1_gpu.py`、`x1_sim.py`、`m7_calib.py`、`m7_model.py`。
- 群組配額（12:00）：/work 1452G／1536G。

## 2026-11-07 12:13（台灣）— 收到 PREREG 偏離 15（`icpe-main` 寫），接在偏離 16 後面

- `icpe-main` 把偏離 15 寫在 `origin/icpe-next-main` @ `2e071aa`（acvlab `date -u` 2026-11-07T03:47:09Z；底＝本分支 e93074f 的 PREREG，經 d0e6516 同步）：CacheWise 從論文拿掉（授權），內部分析保留；論文只用 6 份紀錄、28 個變體；CacheWise 的新工作（偏離 10 的版本 C、偏離 11、偏離 14 的重跑）先不跑；對話資料集不受影響。照實列了前後：合成到達・工具類 21／26 → 14／14（≥15% 成立的變體數），原始到達・工具類 3／9 不變，推翻條件不變。**方向**（節內寫明）：拿掉的 12 個裡有 5 個沒過門檻，拿掉後合成工具類更一致地過門檻，偏向「預測仍有空間」，也就是對「增益不是來自預測」的主張**不利**的方向。2e071aa 的 commit 訊息寫「number pending h200; not pushed」是寫 commit 時的狀態，內文寫 15、已 push。
- 我核過：2e071aa 的 PREREG＝e93074f 原樣＋檔尾 18 行，0 行刪除。本分支在 e93074f 之後已經加了偏離 16（9e52a5f），所以**不整份取 2e071aa**，而是把那 18 行原樣接在偏離 16 後面。檔案裡的順序因此是 14、16、15（偏離 15 寫入 acvlab 的時間 03:47:09Z 早於偏離 16 的 03:58Z，但 push 較晚）。
- 更正我 11:47 對使用者的口頭說法：我說拿掉 CacheWise「方向對我們的主張有利」，**說反了**。21／26 → 14／14 數的是過 15% 門檻的變體，拿掉後全部過門檻，是對主張不利的方向（照偏離 15 原文）。這句話沒有寫進任何檔案。

## 2026-11-07 12:49（台灣）— 「TraceLab c1 的 GPU 層縮小重播」可行性與時間〔推算〕（使用者在 `icpe-main` 問，只估計、不排）

**1. 程式要改多少**
- **不需要 vLLM。** x1_gpu 的重播用 harness（HF 模型＋自己的 KV 緩衝），時間和 KV 內容無關（P0-2 §5）。「GPU 層留住」是 Engine 的帳；GPU 上只要放得下這個請求要用的那段 KV。
- **決策程式**：icpe-sim 分支的 Python `x1_sim.py` 已經有 GPU 層（session 粒度；LRU、BEL、TTLA、TTLB；wb／wt）。**block 粒度 APC 和 BELP 只在 icpe-rux 的 Rust v8**，Python 沒有（各分支的 x1_sim.py 都查不到）。
- **要做的**：
  - a. 重播改用有 GPU 層的 Engine。本 worktree 的 x1_sim.py 是舊版，W6 跑完前不能換，所以要另開 worktree 或等 W6 跑完。
  - b. GpuView.restore 接新參數（g＝已在 GPU 的前綴塊數；lay 4＝預取進 GPU）。`m7_restore_harness.Restorer` 要能從第 g 塊開始還原：Cake 從 g 重算、從後面載入，前 g 塊留在 KV 緩衝。在包裝程式裡做，不改原檔。
  - c. 載入 TraceLab（x1_nb 格式；轉換程式在 icpe-next-data；原始檔 H200 上有，`datasets/tracelab` 97 MB）。
  - d. 縮小方式與暖機（見 2）。
  - e. 自測：重播（模型時間）≡ Engine.run 逐位元；Python 和 Rust 在同一份縮小紀錄上對拍。
  - f. 若要 APC／BELP：移植到 Python，並和 Rust v8 對拍。
- **工作量**〔推算〕：LRU／TTL／BEL 版約 1–1.5 天；加 APC／BELP 再 1–2 天（還要 icpe-rux 對拍）。

**2. 可不可行、GPU 時數**
- **TraceLab 的規模**（run `20261029-104721-icpe-agent-traces-char`，`results/icpe_h200/agent/trace_stats.csv`）：665,453 請求、8,058 session；輸入中位數 132,092、p90 338,662、p99 856,464、最大 999,944 token。
- **記憶體**：KV 128 KiB/token → 中位數 16.1、p99 104.5、最大 122.1 GiB。H200 總 139.79 GiB，權重＋非 torch 15.81 GiB（PREREG §2 第 1 項）→ 約 850K token 以上的請求放不下，要排除或截斷。這會改變工作負載，要登記。
- **照 session 縮小（W6b 的做法）不能用**：GPU 層也要 ÷k。k＝8 → 13.9 GiB，比一個中位數 session 的 16.1 GiB 還小，GPU 層的行為就變了。k＝2（55.5 GiB）約可放 3.4 個中位數 session，全尺寸約 6.9 個。
- **GPU 時間**〔推算：V1 的 f(i)（run `20261022-205934-icpe-p01-c1`，第 80 塊以後線性外插：17.929 ms＋0.2104 ms × 塊序）；每個請求的新 token 數假設約 4K（中位數 input − prefix），實際分佈 NOT_MEASURED〕：
  - 全部命中 GPU：中位數請求約 0.58 s（132K 上下文上 8 塊新 token），p90 約 1.26 s。
  - 全部重算：中位數 11.5 s，p90 57.7 s。
  - 全紀錄一個 run 至少約 665K × 0.6–1 s ≈ 110–185 GPU 小時；k＝2 也要 55–90 小時／run。**全紀錄或 k＝2 在 11/17 前不可行。**
- **可行的形式**：
  - 取一段時間窗，GPU 層、CPU 用全尺寸；窗口之前用模擬暖機（重播迴圈先用模型時間跑到窗口開始，再切到 GPU）。
  - 每個 run 的預算 6–8 小時 → 約 2.5–4 萬個請求（全紀錄的 4–6%）。
  - LRU、BEL、最好的 TTL × 1 個 CPU 容量 × 1–2 次 ≈ 20–50 GPU 小時。
  - 窗口怎麼選（要在看任何 TraceLab GPU 數字之前定規則）、要不要排除超長請求，都要先登記。
- **其他限制**：
  - f(i) 只校準到 40K（V1）／128K（W6c），TraceLab 大半超過，重播量到的長上下文時間可能和模擬有系統差（這也正是能驗到的東西）。
  - 重播一次只跑一個請求，窗口內負載高時會排隊（排隊時間另記，S_real 診斷）。

**3. 最快什麼時候**〔推算〕
- 現在的鏈：24507（DL_Lab5，最晚 11/8 05:39）→ W6c（≤1 h）→ W6a（約 10 h）→ W6b（約 26 h）→ W6 最快約 11/9 晚上跑完（假設實驗室的 GPU 名額一空就開始）。
- 程式可以在 W6 跑的時候寫（另開 worktree）→ 縮小版最快 11/9 晚上開跑，20–50 小時 → 11/10 晚上到 11/11 晚上跑完，分析 11/12。只做 LRU／TTL／BEL。
- 要更早：把 W6b（26 h）往後移，可提早約一天（使用者決定）。
- 實驗室上限 4 張 GPU，chihchung、hao 也在用，開始時間可能更晚。

## 2026-11-07 12:54（台灣）— PREREG 偏離 17（補登記）：agent 軌跡 mini-SWE-agent、OpenHands，`h200` 寫

- `icpe-data` 請 `h200` 把 `origin/icpe-next-data` @ `1a92a02` 的 STATUS_DATA「2026-11-07T12:55+08:00 — agent 軌跡補登記」（acvlab `date -u` 04:52:16Z；還沒寫轉換程式、沒跑模擬）收成偏離 17。內容涵蓋版本 B＋C（同偏離 13 補寫）、偏離 14 的重跑、歸「合成到達」單獨成一節，沒有偏離 13 那時的版本缺口，所以照偏離 13 使用者選的形式直接收（照抄原文＋h200 的收錄說明）。
- 號碼：我對 `icpe-main` 說的「下一個是 17」原意是 `h200` 自己的，又被讀成給別人的號碼（第二次，偏離 12 也是）。各分支最大 16，沒有撞號，照用；**`h200` 下一個從 18 開始**。之後回覆時寫成「h200 自己要用的下一號是 X，別的 session 要號請另外問」。
- 寫進 PREREG（H200 `date -u` 2026-11-07T04:54:11Z），只在檔尾加一節、0 行刪除。`icpe-data` 等這節之後才開跑模擬。

## 2026-11-07 13:00（台灣）— PREREG「偏離 17 更正」（OpenHands 規則 6 (b) 照實記錄後照跑），`h200` 收錄

- `icpe-data` 請 `h200` 把 `origin/icpe-next-data` @ `1af8839` 的 STATUS_DATA「13:10 — 偏離 17 的更正（登記）」記成偏離 17 的更正（不佔號）：OpenHands 的規則 6 (b) 中位數 0.795 < 0.8（p10 0.717、p90 0.875、100% 在 [0.5, 2]），使用者在 `icpe-data` 選「照實記錄後照跑」；門檻不改，nb 的 meta 記 pass=false 與 override 理由。
- 照抄原文＋h200 的收錄說明（含時間不一致：節名 13:10／13:00 晚於 commit 12:58:45／12:57:05，以 commit 為準；輸入偏短、對上限的方向 NOT_MEASURED）。寫進 PREREG（H200 `date -u` 2026-11-07T05:00:18Z），只在檔尾加一節、0 行刪除。

## 2026-11-07 13:06（台灣）— W6c 完成；W6a 和 DL_Lab5 24707 重疊 → 停掉重送；runner 改成每個 run 前等、跑完查重疊；TraceLab 重播的設計已定

- **W6c（24443）完成**：DL_Lab5 的 24507 在 12:07:49 提早結束，W6c 12:07:49 → 12:08:55（exit 0）。run `20261107-120752-icpe-next-w6c-c1` exit 0、沒有污染 → `results/icpe_next/a/w6c/calib_c1.csv`。開跑時同節點只有別人的 job（slurm 輸出照記）。還沒分析。
- **W6a（24444）**：12:08:55 開始。DL_Lab5 的 24706（12:47:04–12:47:11）、24707（12:47:11 送出、12:47:13 開始，GPU 4／CPU 64–79；W6a 是 GPU 2／CPU 32–47）在 W6a 中途開始。DL_Lab5 算我們自己的 job，T2 的「不同 NUMA 可並行」只量過兩個計時 job，沒量過訓練 job。
  - 使用者（AskUserQuestion，約 13:03）選「停掉，等 DL_Lab5 跑完再重送」→ 13:03:56 `scancel 24444`。使用者另說「DL_Lab5 那邊要結束讓他跑」→ 24707 不碰。
  - W6a 的 run（`results/icpe_next/a/w6a/overlap.csv` 用 `t1g.py overlap` 事後補記）：
    - smoke（12:08:59–12:09:39）exit 0。
    - `128-O-PF-r0`（run 20261107-121049，12:10:50–12:27:10）exit 0、**沒有重疊 → 有效**。
    - `256-LRU-r0` 第 1 次（run 20261107-122723）gpu_guard 污染（exit 3）；第 2 次（run 20261107-124406，12:44:07–13:00:35）**和 24706、24707 重疊 → 不用**。
    - `256-HINT-PF-r0`（run 20261107-130048）被 scancel 中斷、和 24707 重疊 → 不用。
- **runner 改法**（使用者選「每個 run 前等、跑完查重疊」）：`next_a_w6.sh`、`next_a_w6b.sh` 開跑前與每個 run 前都跑 `t1g.py wait-clear`（等到節點上沒有我們自己的別的 job，最多 1,440 分鐘），每個 run 跑完跑 `t1g.py overlap`（sacct）寫 `<part>/overlap.csv`；污染或重疊就重跑，最多 3 次，3 次都不乾淨就停下回報。分析（`next_a_w6_analyze.py w6a`、`next_a_w6b.py analyze`）只用 overlap.csv 裡 overlap＝0 的 run。這是 PREREG §5「計時不和自己的其他 GPU 工作同時跑」的實作，判準不變。
- **TraceLab c1 GPU 層重播**：使用者在 `icpe-main` 決定做（acvlab 12:5x），我更正估計後直接問使用者（AskUserQuestion，約 13:00），選：
  - 格子：NFS、CPU 8 GiB、GPU 閒；CPU 規則 LRU+RPF，GPU 規則 LRU、TTLB:30、BEL。
  - 時間窗：從紀錄時間的中位數起連續 5,000 個請求，之前用模擬暖機。
  - 超長請求：整份紀錄每個 session 遇到第一個 >786,432 token 的請求就截掉那一輪和之後。
  - 次數：2 次。
  - 順序：W6a → TraceLab 重播 → W6b。W6b（24687）先 `scontrol hold`，TraceLab 的 job 送出後改成排在它後面再放開。
  - 我在 head node 用 `python -I` 讀 TraceLab 原始檔算的數字（只讀輸入，不是模擬結果）：665,453 請求、8,058 session，跨 7,282.8 小時（2025-09-23 → 2026-07-24），每小時請求數中位數 7、p90 285、最多 4,741；>786,432 token 的請求 1.66%（47 個 session）、>524,288 的 4.92%（79 個）；連續 5,000 個請求的窗口長度中位數 21.0 小時（p10 5.2、p90 75.0）；從時間中位數起的 5,000 個請求跨 25.2 小時。
- **icpe-data 的時間更正**：偏離 17 更正的使用者決定實際在 12:57:05（32160d0）之後、12:58:45（1af8839）之前（icpe-data 13:1x 的訊息），順序不變；PREREG 不補更正。
