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
